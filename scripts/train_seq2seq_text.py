import argparse
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention.training.seq_loaders import get_seq2seq_datasets
from set_attention.patch import replace_multihead_attn
from set_attention.heads.ska_tokenized import SetKernelMultiheadAttentionTokenized
from set_attention.tokenizers.active_tokenizer import ActiveUniverseTokenizer, TokenizerConfig
from set_attention.training.text_utils import (
    build_token_set_store,
    encode_sentence,
    ids_to_tokens,
    text_batch_iterator,
)
from set_attention.experiments.nlp_eval import corpus_bleu, rouge_l
from set_attention.utils.profiling import profiler


class TinySeq2Seq(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, nhead: int, layers: int):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=layers)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        memory = self.enc(self.src_emb(src_ids))
        out = self.dec(self.tgt_emb(tgt_ids), memory)
        return self.out(out)


class TinySeq2SeqSKATok(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, nhead: int, layers: int, ska_gamma: float, gate_topk: int):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.dec_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=1,
        )
        self.cross_attn = SetKernelMultiheadAttentionTokenized(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            sim="rbf",
            rbf_gamma=ska_gamma,
        )
        self.out = nn.Linear(d_model, tgt_vocab)
        self.gate_topk = gate_topk

    def enable_gating(self, vocab_size: int, atom_dim: int):
        self.cross_attn.enable_gating(vocab_size=vocab_size, atom_dim=atom_dim, gate_topk=self.gate_topk)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        src_vals: torch.Tensor,
        src_offs: torch.Tensor,
        tgt_vals: torch.Tensor,
        tgt_offs: torch.Tensor,
        src_sigs: torch.Tensor,
        tgt_sigs: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.enc(self.src_emb(src_ids))
        dec_h = self.dec_self(self.tgt_emb(tgt_in_ids))
        h, _ = self.cross_attn(
            dec_h,
            memory,
            memory,
            token_sets_q=tgt_vals,
            token_offs_q=tgt_offs,
            token_sets_k=src_vals,
            token_offs_k=src_offs,
            token_sigs_q=tgt_sigs,
            token_sigs_k=src_sigs,
        )
        return self.out(h)


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
    parser.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska_true", "ska_tok"], default="dot")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ska-gamma", type=float, default=0.3)
    parser.add_argument("--gate-topk", type=int, default=8)
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--token-set-k", type=int, default=64)
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
    train_refs = [t.split() for _, t in train_pairs]

    has_val = val_ds is not None
    if has_val:
        val_pairs = val_ds.pairs
        val_refs = [t.split() for _, t in val_pairs]
    else:
        val_pairs = []
        val_refs = []

    device = torch.device(args.device)

    if args.attn == "ska_tok" or args.tokenizer:
        if args.tokenizer and os.path.isdir(args.tokenizer):
            print(f"[Tokenizer] Loading from {args.tokenizer}")
            tokenizer = ActiveUniverseTokenizer.load(args.tokenizer)
        else:
            print("[Tokenizer] Training new tokenizer on source corpus")
            src_lines = [s for (s, _) in train_pairs]
            tokenizer = ActiveUniverseTokenizer(TokenizerConfig(seed_lengths=(3, 4, 5), min_freq=2, max_len=max_len))
            tokenizer.fit(src_lines)
            if args.tokenizer:
                os.makedirs(args.tokenizer, exist_ok=True)
                tokenizer.save(args.tokenizer)
                print(f"[Tokenizer] Saved to {args.tokenizer}")
    else:
        tokenizer = None

    if args.attn == "ska_tok":
        src_store, tgt_store = build_token_set_store(train_ds, tokenizer, args.token_set_k, device)
        if has_val:
            val_src_store, val_tgt_store = build_token_set_store(val_ds, tokenizer, args.token_set_k, device)
        else:
            val_src_store = val_tgt_store = None
    else:
        src_store = tgt_store = val_src_store = val_tgt_store = None

    model: nn.Module
    if args.attn == "ska_tok":
        model = TinySeq2SeqSKATok(
            src_vocab=len(train_src_stoi),
            tgt_vocab=len(train_tgt_stoi),
            d_model=args.d_model,
            nhead=args.nhead,
            layers=args.layers,
            ska_gamma=args.ska_gamma,
            gate_topk=args.gate_topk,
        )
        model.enable_gating(vocab_size=len(tokenizer.sym2id) + 1, atom_dim=args.d_model)
    else:
        model = TinySeq2Seq(
            src_vocab=len(train_src_stoi),
            tgt_vocab=len(train_tgt_stoi),
            d_model=args.d_model,
            nhead=args.nhead,
            layers=args.layers,
        )

    if args.attn not in {"dot", "ska_tok"}:
        replace_multihead_attn(model, sim=args.attn)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_refs_epoch, train_hyps_epoch = [], []
        with profiler(True) as prof:
            for batch_idx, src_ids, tgt_ids, refs in text_batch_iterator(
                train_pairs, train_src_stoi, train_tgt_stoi, train_refs, max_len, args.batch, shuffle=True
            ):
                batch_idx = batch_idx.to(device)
                src_ids = src_ids.to(device, non_blocking=True)
                tgt_ids = tgt_ids.to(device, non_blocking=True)
                tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

                if args.attn == "ska_tok":
                    src_vals, src_offs, src_sigs = src_store.gather(batch_idx, device)
                    tgt_vals, tgt_offs, tgt_sigs = tgt_store.gather(batch_idx, device)
                    logits = model(src_ids, tgt_in, src_vals, src_offs, tgt_vals, tgt_offs, src_sigs, tgt_sigs)
                else:
                    logits = model(src_ids, tgt_in)

                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=-1)
                train_refs_epoch.extend(refs)
                train_hyps_epoch.extend([ids_to_tokens(row, train_tgt_itos) for row in preds.detach().cpu()])

        tr_bleu = corpus_bleu(train_refs_epoch, train_hyps_epoch)
        tr_rouge = rouge_l(train_refs_epoch, train_hyps_epoch)
        msg = f"[Seq2Seq][{args.attn}] epoch {epoch:02d} train BLEU {tr_bleu:.3f} | ROUGE-L {tr_rouge:.3f} | time {prof['time_s']:.2f}s"
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
            val_refs_epoch, val_hyps_epoch = [], []
            with torch.no_grad():
                for batch_idx, src_ids, tgt_ids, refs in text_batch_iterator(
                    val_pairs, train_src_stoi, train_tgt_stoi, val_refs, max_len, args.batch, shuffle=False
                ):
                    batch_idx = batch_idx.to(device)
                    src_ids = src_ids.to(device, non_blocking=True)
                    tgt_ids = tgt_ids.to(device, non_blocking=True)
                    tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

                    if args.attn == "ska_tok" and val_src_store is not None:
                        src_vals, src_offs, src_sigs = val_src_store.gather(batch_idx, device)
                        tgt_vals, tgt_offs, tgt_sigs = val_tgt_store.gather(batch_idx, device)
                        logits = model(src_ids, tgt_in, src_vals, src_offs, tgt_vals, tgt_offs, src_sigs, tgt_sigs)
                    else:
                        logits = model(src_ids, tgt_in)

                    preds = logits.argmax(dim=-1)
                    val_refs_epoch.extend(refs)
                    val_hyps_epoch.extend([ids_to_tokens(row, train_tgt_itos) for row in preds.detach().cpu()])

            v_bleu = corpus_bleu(val_refs_epoch, val_hyps_epoch)
            v_rouge = rouge_l(val_refs_epoch, val_hyps_epoch)
            print(f"[Seq2Seq][{args.attn}] epoch {epoch:02d} val BLEU {v_bleu:.3f} | ROUGE-L {v_rouge:.3f}")


if __name__ == "__main__":
    main()

