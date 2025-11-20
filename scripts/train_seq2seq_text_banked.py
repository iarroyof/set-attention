import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention.tokenizers.active_tokenizer import ACTIVE_TOKENIZER_TYPE
from set_attention.tokenizers.hf_bpe import HF_BPE_TYPE
from set_attention.tokenizers.hf_unigram import HF_UNIGRAM_TYPE
from set_attention.tokenizers.registry import (
    available_tokenizer_types,
    create_tokenizer,
    load_tokenizer,
    save_tokenizer,
)
from set_attention.training.seq_loaders import get_seq2seq_datasets
from set_attention.sets.bank_builders import build_windowed_bank_from_texts
from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.training.text_utils import (
    encode_sentence,
    ids_to_tokens,
    text_batch_iterator,
    SPECIAL_TOKENS,
)
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.kernels.sketches import MinHasher
from set_attention.utils.profiling import profiler
from set_attention.utils.wandb import init_wandb
from set_attention.experiments.nlp_eval import corpus_bleu, rouge_l


def _parse_tokenizer_config_arg(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text())
    return json.loads(value)


def _default_tokenizer_config(kind: str, max_len: int) -> Dict[str, Any]:
    if kind == ACTIVE_TOKENIZER_TYPE:
        return {"seed_lengths": (3, 4, 5), "min_freq": 2, "max_len": max_len}
    if kind in {HF_UNIGRAM_TYPE, HF_BPE_TYPE}:
        return {"vocab_size": 16000, "min_frequency": 2}
    return {}


def _normalize_tokenizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(config)
    if "seed_lengths" in out:
        out["seed_lengths"] = tuple(int(v) for v in out["seed_lengths"])
    if "special_tokens" in out:
        out["special_tokens"] = tuple(str(v) for v in out["special_tokens"])
    for key in ("min_freq", "min_frequency", "vocab_size", "max_len"):
        if key in out:
            out[key] = int(out[key])
    return out

class TinySeq2SeqBackbone(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, nhead: int, layers: int):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.dec_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=1,
        )

    def forward(self, src_ids: torch.Tensor, tgt_in_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = self.enc(self.src_emb(src_ids))
        dec = self.dec_self(self.tgt_emb(tgt_in_ids))
        return dec, mem



def run_seq2seq_benchmark(
    args,
    model,
    set_attn,
    router,
    out_proj,
    adapter,
    atom_emb,
    phi_snapshot,
    cache_src,
    cache_tgt,
    train_pairs,
    train_src_stoi,
    train_tgt_stoi,
    train_ref_tokens,
    max_len,
    opt,
    device,
    wandb_run,
):
    iterator = text_batch_iterator(
        train_pairs,
        train_src_stoi,
        train_tgt_stoi,
        train_ref_tokens,
        max_len,
        args.batch,
        shuffle=False,
    )
    try:
        batch_idx_tensor, src_ids, tgt_ids, _ = next(iterator)
    except StopIteration:
        print("[benchmark] insufficient data for benchmark batch.")
        return
    batch_idx_tensor = batch_idx_tensor.to(device)
    src_ids = src_ids.to(device, non_blocking=True)
    tgt_ids = tgt_ids.to(device, non_blocking=True)
    tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

    def step():
        opt.zero_grad(set_to_none=True)
        phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
        Phi_q, Sig_q, Size_q, q_ptrs = cache_tgt.gather_flat(batch_idx_tensor, phi_dynamic)
        Phi_k, Sig_k, Size_k, k_ptrs = cache_src.gather_flat(batch_idx_tensor, phi_dynamic)
        dec_h, _ = model(src_ids, tgt_in)
        Z_sets, q_ptrs = set_attn(Phi_q, Sig_q, Size_q, q_ptrs, Phi_k, Sig_k, Size_k, k_ptrs)
        tok_out = router(dec_h, Z_sets, Phi_q, q_ptrs)
        logits = out_proj(tok_out)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
        loss.backward()
        opt.step()

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
    tokens = tgt_ids.size(0) * max_len * args.bench_iters
    throughput = tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[benchmark][seq2seq] backend={args.ska_backend} precision={args.precision} "
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
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=False)
    parser.add_argument("--tgt", required=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-samples", type=int, default=200)
    parser.add_argument("--dataset", choices=["", "wmt16_en_ro", "cnn_dailymail"], default="")
    parser.add_argument("--limit", type=int, default=200)
    tokenizer_choices = tuple(available_tokenizer_types())
    parser.add_argument(
        "--tokenizer-dir",
        "--tokenizer",
        dest="tokenizer_dir",
        default="",
        help="Optional directory to load/save tokenizer state.",
    )
    parser.add_argument(
        "--tokenizer-type",
        choices=tokenizer_choices,
        default=ACTIVE_TOKENIZER_TYPE,
        help="Tokenizer backend to use when training a new tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=str,
        default="",
        help="JSON string or path with tokenizer config overrides.",
    )
    parser.add_argument("--atom-dim", type=int, default=128)
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--minhash-k", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--router-topk", type=int, default=0)
    parser.add_argument("--set-kernel", choices=["delta_rbf", "delta_plus_dot", "intersect_norm", "intersect_plus_dot", "dot"], default="delta_plus_dot")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--ska-backend", choices=["python", "triton", "keops"], default="python")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    args = parser.parse_args()

    wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_config = {
        "script": "train_seq2seq_text_banked",
        "dataset": args.dataset or "custom",
        "tokenizer_type": args.tokenizer_type,
        "ska_backend": args.ska_backend,
        "precision": args.precision,
        "window": args.window,
        "stride": args.stride,
        "minhash_k": args.minhash_k,
        "router_topk": args.router_topk,
        "adapter_rank": args.adapter_rank,
        "set_kernel": args.set_kernel,
        "batch": args.batch,
    }
    wandb_run = init_wandb(
        args.wandb,
        args.wandb_project or None,
        args.wandb_run_name or None,
        wandb_config,
        wandb_tags,
    )

    train_ds, val_ds = get_seq2seq_datasets(
        dataset=args.dataset,
        limit=args.limit,
        val_limit=args.limit,
        src_path=args.src,
        tgt_path=args.tgt,
        demo=args.demo,
        demo_samples=args.demo_samples,
        max_len=64,
    )

    train_pairs = train_ds.pairs
    src_texts = [s for (s, _) in train_pairs]
    tgt_texts = [t for (_, t) in train_pairs]
    train_src_stoi = train_ds.src_stoi
    train_tgt_stoi = train_ds.tgt_stoi
    train_tgt_itos = train_ds.tgt_itos
    max_len = train_ds.max_len
    train_ref_tokens = [t.split() for _, t in train_pairs]

    tokenizer_dir = args.tokenizer_dir
    config_overrides = _parse_tokenizer_config_arg(args.tokenizer_config)
    tokenizer_config = _default_tokenizer_config(args.tokenizer_type, max_len)
    if config_overrides:
        tokenizer_config.update(config_overrides)
    tokenizer_config = _normalize_tokenizer_config(tokenizer_config)

    if tokenizer_dir and os.path.isdir(tokenizer_dir):
        print(f"[Tokenizer] Loading {args.tokenizer_type} tokenizer from {tokenizer_dir}")
        load_cfg = config_overrides if config_overrides else None
        tokenizer = load_tokenizer(tokenizer_dir, kind=args.tokenizer_type, config=load_cfg)
    else:
        print(f"[Tokenizer] Training new {args.tokenizer_type} tokenizer on source corpus")
        tokenizer = create_tokenizer(args.tokenizer_type, tokenizer_config)
        tokenizer.fit(src_texts)
        if tokenizer_dir:
            os.makedirs(tokenizer_dir, exist_ok=True)
            save_tokenizer(tokenizer, tokenizer_dir)
            print(f"[Tokenizer] Saved to {tokenizer_dir}")

    src_bank = build_windowed_bank_from_texts(tokenizer, src_texts, window=args.window, stride=args.stride).to(args.device)
    tgt_bank = build_windowed_bank_from_texts(tokenizer, tgt_texts, window=args.window, stride=args.stride).to(args.device)

    has_val = val_ds is not None
    if has_val:
        val_pairs = val_ds.pairs
        val_src_texts = [s for (s, _) in val_pairs]
        val_tgt_texts = [t for (_, t) in val_pairs]
        val_ref_tokens = [t.split() for _, t in val_pairs]
        val_src_bank = build_windowed_bank_from_texts(tokenizer, val_src_texts, window=args.window, stride=args.stride).to(args.device)
        val_tgt_bank = build_windowed_bank_from_texts(tokenizer, val_tgt_texts, window=args.window, stride=args.stride).to(args.device)
    else:
        val_pairs = []
        val_ref_tokens = []

    device = torch.device(args.device)
    V = tokenizer.vocab_size()
    atom_emb = nn.Embedding(V, args.atom_dim).to(device)
    adapter = None
    if args.adapter_rank > 0:
        adapter = AtomFeatureAdapter(args.atom_dim, rank=args.adapter_rank).to(device)
        print("[Adapter] enabled (low-rank adapter trainable; Φ pooled per-batch with autograd)")
        with torch.no_grad():
            phi_snapshot = adapter(atom_emb.weight).detach()
    else:
        for p in atom_emb.parameters():
            p.requires_grad = False
        phi_snapshot = atom_emb.weight
        print("[Adapter] disabled (atom pool frozen; Φ pooled from fixed atom_emb)")

    universe_ids = torch.arange(V, device=device, dtype=torch.long)
    universe = UniversePool(
        universe_ids,
        metadata={
            "tokenizer_type": args.tokenizer_type,
            "tokenizer_path": tokenizer_dir or "inline",
        },
    ).to(device)
    print(universe.log_summary(prefix="[Universe]"))

    train_minhash = MinHasher(k=args.minhash_k, device=src_bank.values.device)
    cache_src = SetFeatureCache(universe, src_bank.values, src_bank.set_offsets, src_bank.seq_offsets, minhash=train_minhash).to(device)
    cache_tgt = SetFeatureCache(universe, tgt_bank.values, tgt_bank.set_offsets, tgt_bank.seq_offsets, minhash=train_minhash).to(device)

    if has_val:
        val_minhash = MinHasher(k=args.minhash_k, device=val_src_bank.values.device)
        val_cache_src = SetFeatureCache(universe, val_src_bank.values, val_src_bank.set_offsets, val_src_bank.seq_offsets, minhash=val_minhash).to(device)
        val_cache_tgt = SetFeatureCache(universe, val_tgt_bank.values, val_tgt_bank.set_offsets, val_tgt_bank.seq_offsets, minhash=val_minhash).to(device)

    model = TinySeq2SeqBackbone(
        src_vocab=len(train_src_stoi),
        tgt_vocab=len(train_tgt_stoi),
        d_model=args.atom_dim,
        nhead=args.heads,
        layers=args.layers,
    ).to(device)
    set_attn = SetBankAttention(
        d_model=args.atom_dim,
        num_heads=args.heads,
        tau=args.tau,
        gamma=0.3,
        beta=args.beta,
        score_mode=args.set_kernel,
        eta=args.eta,
        backend=args.ska_backend,
        precision=args.precision,
    ).to(device)
    router = TokenSetRouter(d_model=args.atom_dim, num_heads=args.heads, topk=args.router_topk).to(device)
    out_proj = nn.Linear(args.atom_dim, len(train_tgt_stoi)).to(device)

    params = list(model.parameters()) + list(set_attn.parameters()) + list(router.parameters()) + list(out_proj.parameters())
    if adapter is not None:
        params += list(adapter.parameters()) + list(atom_emb.parameters())
    else:
        params += list(atom_emb.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4)

    if args.benchmark:
        run_seq2seq_benchmark(
            args,
            model,
            set_attn,
            router,
            out_proj,
            adapter,
            atom_emb,
            phi_snapshot,
            cache_src,
            cache_tgt,
            train_pairs,
            train_src_stoi,
            train_tgt_stoi,
            train_ref_tokens,
            max_len,
            opt,
            device,
            wandb_run,
        )
        wandb_run.finish()
        return

    for epoch in range(1, args.epochs + 1):
        model.train()
        set_attn.train()
        router.train()
        if adapter is not None:
            adapter.train()

        train_refs, train_hyps = [], []
        train_loss_total = 0.0
        train_token_total = 0
        with profiler(True) as prof:
            for batch_idx_tensor, src_ids, tgt_ids, ref_tokens in text_batch_iterator(
                train_pairs,
                train_src_stoi,
                train_tgt_stoi,
                train_ref_tokens,
                max_len,
                args.batch,
                shuffle=True,
            ):
                batch_idx_tensor = batch_idx_tensor.to(device)
                src_ids = src_ids.to(device, non_blocking=True)
                tgt_ids = tgt_ids.to(device, non_blocking=True)
                tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

                phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                Phi_q, Sig_q, Size_q, q_ptrs = cache_tgt.gather_flat(batch_idx_tensor, phi_dynamic)
                Phi_k, Sig_k, Size_k, k_ptrs = cache_src.gather_flat(batch_idx_tensor, phi_dynamic)

                dec_h, _ = model(src_ids, tgt_in)
                Z_sets, q_ptrs = set_attn(Phi_q, Sig_q, Size_q, q_ptrs, Phi_k, Sig_k, Size_k, k_ptrs)
                tok_out = router(dec_h, Z_sets, Phi_q, q_ptrs)
                logits = out_proj(tok_out)

                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                train_loss_total += float(loss.detach()) * tgt_ids.numel()
                train_token_total += tgt_ids.numel()
                preds = logits.argmax(dim=-1)
                train_refs.extend(ref_tokens)
                train_hyps.extend([ids_to_tokens(row, train_tgt_itos) for row in preds.detach().cpu()])

        tr_loss = train_loss_total / max(1, train_token_total)
        tr_ppl = math.exp(min(tr_loss, 20.0))
        tr_bleu = corpus_bleu(train_refs, train_hyps)
        tr_rouge = rouge_l(train_refs, train_hyps)
        msg = (
            f"[Seq2Seq-Text-Banked] epoch {epoch:02d} "
            f"train loss {tr_loss:.4f} ppl {tr_ppl:.2f} BLEU {tr_bleu:.3f} | ROUGE-L {tr_rouge:.3f} | time {prof['time_s']:.2f}s"
        )
        if prof.get("cpu_pct") is not None:
            msg += f" | CPU {prof['cpu_pct']:.1f}%"
        if torch.cuda.is_available():
            msg += f" | VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
            if prof.get("gpu_util_pct") is not None:
                msg += f" | GPU {prof['gpu_util_pct']:.1f}%"
            if prof.get("gpu_mem_util_pct") is not None:
                msg += f" | GPU-MEM {prof['gpu_mem_util_pct']:.1f}%"
            if prof.get("gpu_active_s") is not None:
                msg += f" | GPU-ACT {prof['gpu_active_s']:.2f}s"
        print(msg)
        if wandb_run.enabled:
            wandb_payload = {
                "train/bleu": tr_bleu,
                "train/rougeL": tr_rouge,
                "train/time_s": prof["time_s"],
            }
            if prof.get("gpu_peak_mem_mib") is not None:
                wandb_payload["train/peak_vram_mib"] = prof["gpu_peak_mem_mib"]
            wandb_run.log(wandb_payload, step=epoch)

        if has_val:
            model.eval()
            set_attn.eval()
            router.eval()
            if adapter is not None:
                adapter.eval()
            val_refs, val_hyps = [], []
            val_loss_total = 0.0
            val_token_total = 0
            with torch.no_grad():
                for batch_idx_tensor, src_ids, tgt_ids, ref_tokens in text_batch_iterator(
                    val_pairs,
                    train_src_stoi,
                    train_tgt_stoi,
                    val_ref_tokens,
                    max_len,
                    args.batch,
                    shuffle=False,
                ):
                    batch_idx_tensor = batch_idx_tensor.to(device)
                    src_ids = src_ids.to(device, non_blocking=True)
                    tgt_ids = tgt_ids.to(device, non_blocking=True)
                    tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

                    phi_dynamic = adapter(atom_emb.weight) if adapter is not None else phi_snapshot
                    Phi_q, Sig_q, Size_q, q_ptrs = val_cache_tgt.gather_flat(batch_idx_tensor, phi_dynamic)
                    Phi_k, Sig_k, Size_k, k_ptrs = val_cache_src.gather_flat(batch_idx_tensor, phi_dynamic)

                    dec_h, _ = model(src_ids, tgt_in)
                    Z_sets, q_ptrs = set_attn(Phi_q, Sig_q, Size_q, q_ptrs, Phi_k, Sig_k, Size_k, k_ptrs)
                    tok_out = router(dec_h, Z_sets, Phi_q, q_ptrs)
                    logits = out_proj(tok_out)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
                    val_loss_total += float(loss.detach()) * tgt_ids.numel()
                    val_token_total += tgt_ids.numel()
                    preds = logits.argmax(dim=-1)
                    val_refs.extend(ref_tokens)
                    val_hyps.extend([ids_to_tokens(row, train_tgt_itos) for row in preds.detach().cpu()])

            v_loss = val_loss_total / max(1, val_token_total)
            v_ppl = math.exp(min(v_loss, 20.0))
            v_bleu = corpus_bleu(val_refs, val_hyps)
            v_rouge = rouge_l(val_refs, val_hyps)
            print(
                f"[Seq2Seq-Text-Banked] epoch {epoch:02d} VAL loss {v_loss:.4f} ppl {v_ppl:.2f} "
                f"BLEU {v_bleu:.3f} | ROUGE-L {v_rouge:.3f}"
            )
        else:
            v_loss = v_ppl = v_bleu = v_rouge = None
            val_refs = val_hyps = []

        if wandb_run.enabled:
            payload = {
                "train/loss": tr_loss,
                "train/ppl": tr_ppl,
                "train/bleu": tr_bleu,
                "train/rougeL": tr_rouge,
            }
            if v_loss is not None:
                payload.update(
                    {
                        "val/loss": v_loss,
                        "val/ppl": v_ppl,
                        "val/bleu": v_bleu,
                        "val/rougeL": v_rouge,
                    }
                )
            if val_refs and val_hyps:
                ref_sample = " ".join(val_refs[0])
                pred_sample = " ".join(val_hyps[0])
                payload["samples/val_text"] = f"REF: {ref_sample}\nPRED: {pred_sample}"
            wandb_run.log(payload, step=epoch)

    wandb_run.finish()
            if wandb_run.enabled:
                wandb_run.log(
                    {
                        "val/bleu": v_bleu,
                        "val/rougeL": v_rouge,
                    },
                    step=epoch,
                )

    wandb_run.finish()


if __name__ == "__main__":
    main()

