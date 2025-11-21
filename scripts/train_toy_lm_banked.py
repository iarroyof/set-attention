import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention.data import configure_hf_cache, resolve_data_root
from set_attention.data.wikitext import chunk_tokens, load_wikitext_lines, tokenize_lines
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.kernels.sketches import MinHasher
from set_attention.utils.profiling import profiler
from set_attention.utils.sample_logging import format_text_samples
from set_attention.utils.wandb import init_wandb
from set_attention.experiments.nlp_eval import corpus_bleu
from set_attention.training.text_utils import (
    ids_to_tokens,
    text_batch_iterator,
)


def make_char_data(n=2000, seq_len=64, vocab=32, seed=3):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, vocab, (n, seq_len), generator=g)
    Y = torch.roll(X, shifts=-1, dims=1)
    return X, Y


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]


def build_vocab_from_tokens(tokens):
    vocab = list(SPECIAL_TOKENS)
    seen = set(vocab)
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            vocab.append(tok)
    stoi = {tok: idx for idx, tok in enumerate(vocab)}
    itos = {idx: tok for tok, idx in stoi.items()}
    return stoi, itos


def encode_token_sequences(token_sequences, stoi):
    unk_id = stoi.get("<unk>", 0)
    encoded = []
    for seq in token_sequences:
        ids = [stoi.get(tok, unk_id) for tok in seq]
        encoded.append(torch.tensor(ids, dtype=torch.long))
    if not encoded:
        return torch.empty(0, 0, dtype=torch.long)
    return torch.stack(encoded, dim=0)


def tensors_to_text_pairs(X, Y, itos):
    pairs = []
    refs = []
    for src, tgt in zip(X, Y):
        src_tokens = [itos[int(tok)] for tok in src.tolist()]
        tgt_tokens = [itos[int(tok)] for tok in tgt.tolist()]
        pairs.append((" ".join(src_tokens), " ".join(tgt_tokens)))
        refs.append(tgt_tokens)
    return pairs, refs


def load_wikitext_dataset(args, cache_dir):
    line_limit = args.dataset_lines if args.dataset_lines > 0 else None
    train_lines = load_wikitext_lines(args.dataset, "train", cache_dir, line_limit)
    val_lines = load_wikitext_lines(args.dataset, "validation", cache_dir, line_limit)
    train_tokens = tokenize_lines(train_lines)
    val_tokens = tokenize_lines(val_lines)
    train_chunks = chunk_tokens(train_tokens, args.seq_len, args.seq_stride)
    val_chunks = chunk_tokens(val_tokens, args.seq_len, args.seq_stride)
    if not train_chunks:
        raise RuntimeError("No training sequences were generated from Wikitext.")
    stoi, itos = build_vocab_from_tokens(train_tokens)
    train_X = encode_token_sequences(train_chunks, stoi)
    val_X = encode_token_sequences(val_chunks, stoi)
    train_Y = torch.roll(train_X, shifts=-1, dims=1)
    val_Y = torch.roll(val_X, shifts=-1, dims=1)
    return train_X, train_Y, val_X, val_Y, stoi, itos


class TinyLMBackbone(nn.Module):
    def __init__(self, vocab: int, d_model: int, nhead: int, layers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(self.emb(x))


def run_lm_benchmark(
    args,
    backbone,
    set_attn,
    router,
    head,
    adapter,
    atom_emb,
    phi_snapshot,
    cache,
    optimizer,
    train_X,
    train_Y,
    max_len,
    device,
    wandb_run,
):
    bench_batch = min(args.batch, train_X.size(0))
    if bench_batch == 0:
        print("[benchmark] no data available.")
        return
    batch_idx = torch.arange(bench_batch, dtype=torch.long, device=device)
    src_ids = train_X[:bench_batch].to(device)
    tgt_ids = train_Y[:bench_batch].to(device)
    tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

    def step():
        optimizer.zero_grad(set_to_none=True)
        phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
        Phi, Sig, Size, q_ptrs = cache.gather_flat(batch_idx, phi_dynamic)
        token_states = backbone(src_ids)
        Z, q_ptrs = set_attn(Phi, Sig, Size, q_ptrs, Phi, Sig, Size, q_ptrs)
        routed = router(token_states, Z, Phi, q_ptrs)
        logits = head(routed)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
        loss.backward()
        optimizer.step()

    for _ in range(args.bench_warmup):
        step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.bench_iters):
        step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tokens = bench_batch * max_len * args.bench_iters
    throughput = tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[benchmark][lm] backend={args.ska_backend} precision={args.precision} "
        f"tokens/s={throughput:.1f} elapsed={elapsed:.3f}s"
    )
    if wandb_run.enabled:
        wandb_run.log(
            {
                "benchmark/tokens_per_s": throughput,
                "benchmark/elapsed_s": elapsed,
                "benchmark/batch_tokens": tokens,
            }
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska_true"], default="dot")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--minhash-k", type=int, default=64)
    ap.add_argument("--adapter-rank", type=int, default=0)
    ap.add_argument("--router-topk", type=int, default=0)
    ap.add_argument("--profile", "--prof", action="store_true", dest="profile")
    ap.add_argument("--ska-backend", choices=["python", "triton", "keops"], default="python")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--dataset", choices=["", "wikitext2", "wikitext103"], default="")
    ap.add_argument("--dataset-lines", type=int, default=0, help="Limit number of text lines per split (0 = all).")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seq-stride", type=int, default=128)
    ap.add_argument("--data-root", type=str, default="")
    ap.add_argument("--hf-cache-dir", type=str, default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="")
    ap.add_argument("--wandb-run-name", type=str, default="")
    ap.add_argument("--wandb-tags", type=str, default="")
    ap.add_argument("--sample-count", type=int, default=10, help="Number of validation samples to log per epoch.")
    ap.add_argument("--sample-seed", type=int, default=1337, help="Seed controlling which samples are logged.")
    args = ap.parse_args()

    wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_config = {
        "script": "train_toy_lm_banked",
        "dataset": args.dataset or "char",
        "seq_len": args.seq_len,
        "seq_stride": args.seq_stride,
        "window": args.window,
        "stride": args.stride,
        "minhash_k": args.minhash_k,
        "router_topk": args.router_topk,
        "adapter_rank": args.adapter_rank,
        "ska_backend": args.ska_backend,
        "precision": args.precision,
        "batch": args.batch,
        "sample_count": args.sample_count,
    }
    wandb_run = init_wandb(
        args.wandb,
        args.wandb_project or None,
        args.wandb_run_name or None,
        wandb_config,
        wandb_tags,
    )

    data_root = resolve_data_root(args.data_root)
    hf_cache = configure_hf_cache(data_root, args.hf_cache_dir)

    if args.dataset:
        train_X, train_Y, val_X, val_Y, stoi, itos = load_wikitext_dataset(args, hf_cache)
        print(f"[Data] Loaded {args.dataset} into cache {hf_cache}")
    else:
        X, Y = make_char_data(seq_len=args.seq_len)
        n_val = int(0.2 * X.size(0))
        train_X, val_X = X[:-n_val], X[-n_val:]
        train_Y, val_Y = Y[:-n_val], Y[-n_val:]
        raw_vocab = int(X.max().item() + 1)
        token_list = [f"tok{i}" for i in range(raw_vocab)]
        vocab_tokens = SPECIAL_TOKENS + token_list
        stoi = {tok: idx for idx, tok in enumerate(vocab_tokens)}
        itos = {idx: tok for tok, idx in stoi.items()}

    train_text_pairs, train_refs = tensors_to_text_pairs(train_X, train_Y, itos)
    val_text_pairs, val_refs = tensors_to_text_pairs(val_X, val_Y, itos)

    max_len = train_X.size(1)
    vocab_size = len(stoi)
    device = torch.device(args.device)

    sequences_train = [row.clone() for row in train_X]
    sequences_val = [row.clone() for row in val_X] if val_X.numel() > 0 else []
    train_bank = build_windowed_bank_from_ids(sequences_train, window=args.window, stride=args.stride).to(device)
    val_bank = build_windowed_bank_from_ids(sequences_val, window=args.window, stride=args.stride).to(device)

    atom_emb = nn.Embedding(vocab_size, args.d_model).to(device)
    adapter = None
    if args.adapter_rank > 0:
        adapter = AtomFeatureAdapter(args.d_model, rank=args.adapter_rank).to(device)
        with torch.no_grad():
            phi_snapshot = adapter(atom_emb.weight).detach()
    else:
        phi_snapshot = atom_emb.weight

    universe_ids = torch.arange(vocab_size, device=device, dtype=torch.long)
    universe = UniversePool(universe_ids, metadata={"task": "toy_lm"}).to(device)
    train_minhash = MinHasher(k=args.minhash_k, device=train_bank.values.device)
    val_minhash = MinHasher(k=args.minhash_k, device=val_bank.values.device)
    train_cache = SetFeatureCache(universe, train_bank.values, train_bank.set_offsets, train_bank.seq_offsets, minhash=train_minhash).to(device)
    val_cache = SetFeatureCache(universe, val_bank.values, val_bank.set_offsets, val_bank.seq_offsets, minhash=val_minhash).to(device)

    backbone = TinyLMBackbone(vocab_size, args.d_model, args.nhead, args.layers).to(device)
    set_attn = SetBankAttention(
        d_model=args.d_model,
        num_heads=args.nhead,
        tau=1.0,
        gamma=0.3,
        beta=1.0,
        score_mode="delta_plus_dot",
        eta=1.0,
        backend=args.ska_backend,
        precision=args.precision,
    ).to(device)
    router = TokenSetRouter(d_model=args.d_model, num_heads=args.nhead, topk=args.router_topk).to(device)
    head = nn.Linear(args.d_model, vocab_size).to(device)

    params = list(backbone.parameters()) + list(set_attn.parameters()) + list(router.parameters()) + list(head.parameters()) + list(atom_emb.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)

    if args.benchmark:
        run_lm_benchmark(
            args,
            backbone,
            set_attn,
            router,
            head,
            adapter,
            atom_emb,
            phi_snapshot,
            train_cache,
            optimizer,
            train_X,
            train_Y,
            max_len,
            device,
            wandb_run,
        )
        return

    def run_epoch(train: bool):
        model_list = [backbone, router, head]
        if adapter is not None:
            model_list.append(adapter)
        for m in model_list:
            m.train(train)
        refs_epoch, hyps_epoch = [], []
        total_loss = 0.0
        total_tokens = 0
        iterator = text_batch_iterator(
            train_text_pairs if train else val_text_pairs,
            stoi,
            stoi,
            train_refs if train else val_refs,
            max_len,
            args.batch,
            shuffle=train,
        )
        for batch_idx, src_ids, tgt_ids, ref_tokens in iterator:
            batch_idx = batch_idx.to(device)
            src_ids = src_ids.to(device, non_blocking=True)
            tgt_ids = tgt_ids.to(device, non_blocking=True)
            tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

            phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
            cache = train_cache if train else val_cache
            Phi, Sig, Size, q_ptrs = cache.gather_flat(batch_idx, phi_dynamic)
            token_states = backbone(src_ids)
            Z, q_ptrs = set_attn(Phi, Sig, Size, q_ptrs, Phi, Sig, Size, q_ptrs)
            routed = router(token_states, Z, Phi, q_ptrs)
            logits = head(routed)

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach()) * tgt_ids.numel()
            total_tokens += tgt_ids.numel()
            preds = logits.argmax(dim=-1)
            refs_epoch.extend(ref_tokens)
            hyps_epoch.extend([ids_to_tokens(row, itos) for row in preds.detach().cpu()])

        return total_loss / max(1, total_tokens), refs_epoch, hyps_epoch

    for epoch in range(1, args.epochs + 1):
        with profiler(args.profile) as prof:
            train_loss, train_refs_epoch, train_hyps_epoch = run_epoch(train=True)
            val_loss, val_refs_epoch, val_hyps_epoch = run_epoch(train=False)

        train_bleu = corpus_bleu(train_refs_epoch, train_hyps_epoch)
        val_bleu = corpus_bleu(val_refs_epoch, val_hyps_epoch)
        train_ppl = math.exp(min(train_loss, 20.0))
        val_ppl = math.exp(min(val_loss, 20.0))
        msg = (
            f"[LM-Banked][{args.attn}] epoch {epoch:02d} "
            f"train loss {train_loss:.4f} ppl {train_ppl:.2f} BLEU {train_bleu:.3f} | "
            f"val loss {val_loss:.4f} ppl {val_ppl:.2f} BLEU {val_bleu:.3f}"
        )
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
        print(msg)
        if wandb_run.enabled:
            payload = {
                "train/loss": train_loss,
                "train/ppl": train_ppl,
                "train/bleu": train_bleu,
                "val/loss": val_loss,
                "val/ppl": val_ppl,
                "val/bleu": val_bleu,
            }
            if args.profile:
                payload["train/time_s"] = prof["time_s"]
                if torch.cuda.is_available():
                    payload["train/peak_vram_mib"] = prof["gpu_peak_mem_mib"]
            sample_text = format_text_samples(
                val_refs_epoch,
                val_hyps_epoch,
                args.sample_count,
                args.sample_seed + epoch,
            )
            if sample_text:
                payload["samples/val_text"] = sample_text
            wandb_run.log(payload, step=epoch)

    wandb_run.finish()
if __name__ == "__main__":
    main()
