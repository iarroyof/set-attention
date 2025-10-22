import argparse
import os
import random
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention.tokenizers.active_tokenizer import ActiveUniverseTokenizer, TokenizerConfig
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
from set_attention.experiments.nlp_eval import corpus_bleu, rouge_l

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
    parser.add_argument("--tokenizer", type=str, default="")
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
    args = parser.parse_args()

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
    train_src_stoi = train_ds.src_stoi
    train_tgt_stoi = train_ds.tgt_stoi
    train_tgt_itos = train_ds.tgt_itos
    max_len = train_ds.max_len
    train_ref_tokens = [t.split() for _, t in train_pairs]

    if args.tokenizer and os.path.isdir(args.tokenizer):
        print(f"[Tokenizer] Loading from {args.tokenizer}")
        tokenizer = ActiveUniverseTokenizer.load(args.tokenizer)
    else:
        print("[Tokenizer] Training new tokenizer on source corpus")
        src_lines = [s for (s, _) in train_ds.pairs]
        tokenizer = ActiveUniverseTokenizer(TokenizerConfig(seed_lengths=(3, 4, 5), min_freq=2, max_len=64))
        tokenizer.fit(src_lines)
        if args.tokenizer:
            os.makedirs(args.tokenizer, exist_ok=True)
            tokenizer.save(args.tokenizer)
            print(f"[Tokenizer] Saved to {args.tokenizer}")

    src_texts = [s for (s, _) in train_pairs]
    tgt_texts = [t for (_, t) in train_pairs]
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
    V = len(tokenizer.sym2id) + 1
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
    universe = UniversePool(universe_ids, metadata={"tokenizer": args.tokenizer or "inline"}).to(device)
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
    ).to(device)
    router = TokenSetRouter(d_model=args.atom_dim, num_heads=args.heads, topk=args.router_topk).to(device)
    out_proj = nn.Linear(args.atom_dim, len(train_tgt_stoi)).to(device)

    params = list(model.parameters()) + list(set_attn.parameters()) + list(router.parameters()) + list(out_proj.parameters())
    if adapter is not None:
        params += list(adapter.parameters()) + list(atom_emb.parameters())
    else:
        params += list(atom_emb.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        set_attn.train()
        router.train()
        if adapter is not None:
            adapter.train()

        train_refs, train_hyps = [], []
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
                Phi_q, Sig_q, Size_q, mask_q = cache_tgt.gather_padded(batch_idx_tensor, phi_dynamic)
                Phi_k, Sig_k, Size_k, mask_k = cache_src.gather_padded(batch_idx_tensor, phi_dynamic)

                dec_h, _ = model(src_ids, tgt_in)
                Z_sets = set_attn(Phi_q, Sig_q, Size_q, mask_q, Phi_k, Sig_k, Size_k, mask_k)
                tok_out = router(dec_h, Z_sets, Phi_q, mask_q)
                logits = out_proj(tok_out)

                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                preds = logits.argmax(dim=-1)
                train_refs.extend(ref_tokens)
                train_hyps.extend([ids_to_tokens(row, train_tgt_itos) for row in preds.detach().cpu()])

        tr_bleu = corpus_bleu(train_refs, train_hyps)
        tr_rouge = rouge_l(train_refs, train_hyps)
        msg = f"[Seq2Seq-Text-Banked] epoch {epoch:02d} BLEU {tr_bleu:.3f} | ROUGE-L {tr_rouge:.3f} | time {prof['time_s']:.2f}s"
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

        if has_val:
            model.eval()
            set_attn.eval()
            router.eval()
            if adapter is not None:
                adapter.eval()
            val_refs, val_hyps = [], []
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
                    Phi_q, Sig_q, Size_q, mask_q = val_cache_tgt.gather_padded(batch_idx_tensor, phi_dynamic)
                    Phi_k, Sig_k, Size_k, mask_k = val_cache_src.gather_padded(batch_idx_tensor, phi_dynamic)

                    dec_h, _ = model(src_ids, tgt_in)
                    Z_sets = set_attn(Phi_q, Sig_q, Size_q, mask_q, Phi_k, Sig_k, Size_k, mask_k)
                    tok_out = router(dec_h, Z_sets, Phi_q, mask_q)
                    logits = out_proj(tok_out)
                    preds = logits.argmax(dim=-1)
                    val_refs.extend(ref_tokens)
                    val_hyps.extend([ids_to_tokens(row, train_tgt_itos) for row in preds.detach().cpu()])

            v_bleu = corpus_bleu(val_refs, val_hyps)
            v_rouge = rouge_l(val_refs, val_hyps)
            print(f"[Seq2Seq-Text-Banked] epoch {epoch:02d} VAL BLEU {v_bleu:.3f} | ROUGE-L {v_rouge:.3f}")


if __name__ == "__main__":
    main()

