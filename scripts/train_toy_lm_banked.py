import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention.sets.atom_adapter import AtomFeatureAdapter
from set_attention.sets.bank_builders import build_windowed_bank_from_ids
from set_attention.heads.banked_attention import SetBankAttention
from set_attention.heads.token_router import TokenSetRouter
from set_attention.universe import SetFeatureCache, UniversePool
from set_attention.kernels.sketches import MinHasher
from set_attention.utils.profiling import profiler
from set_attention.experiments.nlp_eval import corpus_bleu
from set_attention.training.text_utils import (
    ids_to_tokens,
    ints_to_text,
    text_batch_iterator,
)


def make_char_data(n=2000, seq_len=64, vocab=32, seed=3):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, vocab, (n, seq_len), generator=g)
    Y = torch.roll(X, shifts=-1, dims=1)
    return X, Y


class TinyLMBackbone(nn.Module):
    def __init__(self, vocab: int, d_model: int, nhead: int, layers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(self.emb(x))


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
    args = ap.parse_args()

    X, Y = make_char_data()
    n_val = int(0.2 * X.size(0))
    train_X, val_X = X[:-n_val], X[-n_val:]
    train_Y, val_Y = Y[:-n_val], Y[-n_val:]
    raw_vocab = int(X.max().item() + 1)
    token_list = [f"tok{i}" for i in range(raw_vocab)]
    vocab_tokens = ["<pad>", "<s>", "</s>", "<unk>"] + token_list
    stoi = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    itos = {idx: tok for tok, idx in stoi.items()}
    vocab_size = len(vocab_tokens)

    train_pairs_int = list(zip(train_X.tolist(), train_Y.tolist()))
    val_pairs_int = list(zip(val_X.tolist(), val_Y.tolist()))
    train_text_pairs = [(ints_to_text(torch.tensor(src)), ints_to_text(torch.tensor(tgt))) for src, tgt in train_pairs_int]
    val_text_pairs = [(ints_to_text(torch.tensor(src)), ints_to_text(torch.tensor(tgt))) for src, tgt in val_pairs_int]
    train_refs = [tgt.split() for _, tgt in train_text_pairs]
    val_refs = [tgt.split() for _, tgt in val_text_pairs]

    max_len = X.size(1) + 2
    vocab_size = len(vocab_tokens)
    device = torch.device(args.device)

    def remap_to_vocab(seq):
        # Map raw integer tokens (0..raw_vocab) onto the padded vocabulary indices.
        return torch.tensor([stoi[f"tok{int(tok)}"] for tok in seq], dtype=torch.long)

    sequences_train = [remap_to_vocab(src) for src, _ in train_pairs_int]
    sequences_val = [remap_to_vocab(src) for src, _ in val_pairs_int]
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
    ).to(device)
    router = TokenSetRouter(d_model=args.d_model, num_heads=args.nhead, topk=args.router_topk).to(device)
    head = nn.Linear(args.d_model, vocab_size).to(device)

    params = list(backbone.parameters()) + list(set_attn.parameters()) + list(router.parameters()) + list(head.parameters()) + list(atom_emb.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)

    def run_epoch(train: bool):
        model_list = [backbone, router, head]
        if adapter is not None:
            model_list.append(adapter)
        for m in model_list:
            m.train(train)
        refs_epoch, hyps_epoch = [], []
        total_loss = 0.0
        count = 0
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
            Phi, Sig, Size, mask = cache.gather_padded(batch_idx, phi_dynamic)
            token_states = backbone(src_ids)
            Z = set_attn(Phi, Sig, Size, mask, Phi, Sig, Size, mask)
            routed = router(token_states, Z, Phi, mask)
            logits = head(routed)

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach()) * src_ids.size(0)
            count += src_ids.size(0)
            preds = logits.argmax(dim=-1)
            refs_epoch.extend(ref_tokens)
            hyps_epoch.extend([ids_to_tokens(row, itos) for row in preds.detach().cpu()])

        return total_loss / max(1, count), refs_epoch, hyps_epoch

    for epoch in range(1, args.epochs + 1):
        with profiler(args.profile) as prof:
            train_loss, train_refs_epoch, train_hyps_epoch = run_epoch(train=True)
            val_loss, val_refs_epoch, val_hyps_epoch = run_epoch(train=False)

        train_bleu = corpus_bleu(train_refs_epoch, train_hyps_epoch)
        val_bleu = corpus_bleu(val_refs_epoch, val_hyps_epoch)
        msg = f"[LM-Banked][{args.attn}] epoch {epoch:02d} train loss {train_loss:.4f} BLEU {train_bleu:.3f} | val loss {val_loss:.4f} BLEU {val_bleu:.3f}"
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
        print(msg)
if __name__ == "__main__":
    main()
