import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention.heads.ska_tokenized import SetKernelMultiheadAttentionTokenized
from set_attention.patch import replace_multihead_attn
from set_attention.tokenizers.active_tokenizer import ACTIVE_TOKENIZER_TYPE
from set_attention.tokenizers.hf_bpe import HF_BPE_TYPE
from set_attention.tokenizers.hf_unigram import HF_UNIGRAM_TYPE
from set_attention.tokenizers.registry import available_tokenizer_types, create_tokenizer, load_tokenizer, save_tokenizer
from set_attention.tokenizers.cache import (
    default_tokenizer_dir,
    meta_matches,
    save_tokenizer_meta,
    tokenizer_fingerprint,
)
from set_attention.data.hf_cache import ensure_hf_cache
from set_attention.training.text_utils import (
    build_token_set_store,
    ids_to_tokens,
    ints_to_text,
    text_batch_iterator,
)
from set_attention.experiments.data_nlp import InMemoryTextPairDataset
from set_attention.utils.profiling import profiler


def _parse_tokenizer_config_arg(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text())
    return json.loads(value)


def _default_tokenizer_config(kind: str, max_len: int) -> Dict[str, Any]:
    if kind == ACTIVE_TOKENIZER_TYPE:
        return {"seed_lengths": (3, 4, 5), "min_freq": 1, "max_len": max_len}
    if kind in {HF_UNIGRAM_TYPE, HF_BPE_TYPE}:
        return {"vocab_size": 512, "min_frequency": 1}
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


def make_toy_pairs(n=1000, seq_len=16, vocab=32, seed=7):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(1, vocab, (n, seq_len), generator=g)
    tgt = (torch.flip(src, dims=[1]) + 1) % vocab
    return src, tgt


class TinySeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, layers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=layers)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        memory = self.enc(self.emb(src_ids))
        dec_h = self.dec(self.emb(tgt_ids), memory)
        return self.proj(dec_h)


class TinySeq2SeqSKATok(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, layers: int, ska_gamma: float, gate_topk: int):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.dec_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=1,
        )
        self.cross = SetKernelMultiheadAttentionTokenized(embed_dim=d_model, num_heads=nhead, batch_first=True, gamma=ska_gamma)
        self.proj = nn.Linear(d_model, vocab_size)
        self.gate_topk = gate_topk

    def enable_gating(self, vocab_size: int, atom_dim: int):
        self.cross.enable_gating(vocab_size=vocab_size, atom_dim=atom_dim, gate_topk=self.gate_topk)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in: torch.Tensor,
        src_vals: torch.Tensor,
        src_offs: torch.Tensor,
        tgt_vals: torch.Tensor,
        tgt_offs: torch.Tensor,
        src_sigs: torch.Tensor,
        tgt_sigs: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.enc(self.src_emb(src_ids))
        dec_h = self.dec_self(self.tgt_emb(tgt_in))
        h, _ = self.cross(
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
        return self.proj(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska_true", "ska_tok"], default="dot")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--inter-topk", type=int, default=16)
    ap.add_argument("--inter-norm", type=int, default=1)
    ap.add_argument("--token-set-k", type=int, default=32)
    ap.add_argument("--profile", action="store_true")
    tokenizer_choices = tuple(available_tokenizer_types())
    ap.add_argument(
        "--tokenizer-dir",
        "--tokenizer",
        dest="tokenizer_dir",
        default="",
        help="Optional directory to load/save tokenizer state.",
    )
    ap.add_argument(
        "--tokenizer-type",
        choices=tokenizer_choices,
        default=ACTIVE_TOKENIZER_TYPE,
        help="Tokenizer backend to use when training a new tokenizer.",
    )
    ap.add_argument(
        "--tokenizer-config",
        type=str,
        default="",
        help="JSON string or path with tokenizer config overrides.",
    )
    args = ap.parse_args()
    ensure_hf_cache(None)

    src_int, tgt_int = make_toy_pairs()
    src_texts = [ints_to_text(row) for row in src_int]
    tgt_texts = [ints_to_text(row) for row in tgt_int]
    dataset = InMemoryTextPairDataset(src_texts, tgt_texts, max_len=src_int.size(1) + 2)

    tokenizer = None
    tokenizer_dir = args.tokenizer_dir
    needs_tokenizer = args.attn == "ska_tok" or bool(tokenizer_dir)
    if needs_tokenizer:
        config_overrides = _parse_tokenizer_config_arg(args.tokenizer_config)
        tokenizer_config = _default_tokenizer_config(args.tokenizer_type, dataset.max_len)
        if config_overrides:
            tokenizer_config.update(config_overrides)
        tokenizer_config = _normalize_tokenizer_config(tokenizer_config)

        if not tokenizer_dir:
            hf_home = Path(os.environ.get("HF_HOME") or "~/.cache/set-attention/hf").expanduser()
            dataset_id = "toy_seq2seq"
            fingerprint = tokenizer_fingerprint(
                args.tokenizer_type,
                tokenizer_config,
                dataset_id,
                dataset.max_len,
                limit=None,
                extra={"pairs": len(src_texts)},
            )
            tokenizer_dir = str(default_tokenizer_dir(hf_home, dataset_id, args.tokenizer_type, fingerprint))

        expected_meta = {
            "fingerprint": tokenizer_fingerprint(
                args.tokenizer_type,
                tokenizer_config,
                "toy_seq2seq",
                dataset.max_len,
                limit=None,
                extra={"pairs": len(src_texts)},
            ),
            "dataset": "toy_seq2seq",
            "tokenizer_type": args.tokenizer_type,
            "tokenizer_config": tokenizer_config,
            "max_len": dataset.max_len,
        }

        if tokenizer_dir and Path(tokenizer_dir).is_dir() and meta_matches(Path(tokenizer_dir), expected_meta):
            print(f"[Tokenizer] Loading {args.tokenizer_type} tokenizer from {tokenizer_dir}")
            load_cfg = config_overrides if config_overrides else None
            tokenizer = load_tokenizer(tokenizer_dir, kind=args.tokenizer_type, config=load_cfg)
        else:
            if tokenizer_dir and Path(tokenizer_dir).is_dir():
                print("[Tokenizer] Cache mismatch; retraining to avoid inconsistent token IDs.")
            print(f"[Tokenizer] Training new {args.tokenizer_type} tokenizer on synthetic corpus")
            tokenizer = create_tokenizer(args.tokenizer_type, tokenizer_config)
            tokenizer.fit(src_texts)
            if tokenizer_dir:
                Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)
                save_tokenizer(tokenizer, tokenizer_dir)
                save_tokenizer_meta(Path(tokenizer_dir), expected_meta)
                print(f"[Tokenizer] Saved to {tokenizer_dir}")

    device = torch.device(args.device)

    if args.attn == "ska_tok":
        if tokenizer is None:
            raise RuntimeError("Tokenizer is required for --attn ska_tok.")
        src_store, tgt_store = build_token_set_store(dataset, tokenizer, args.token_set_k, device)
    else:
        src_store = tgt_store = None

    if args.attn == "ska_tok":
        model = TinySeq2SeqSKATok(
            vocab_size=len(dataset.src_stoi),
            d_model=args.d_model,
            nhead=args.nhead,
            layers=args.layers,
            ska_gamma=args.gamma,
            gate_topk=args.inter_topk,
        )
        model.enable_gating(vocab_size=tokenizer.vocab_size(), atom_dim=args.d_model)
    else:
        model = TinySeq2Seq(
            vocab_size=len(dataset.src_stoi),
            d_model=args.d_model,
            nhead=args.nhead,
            layers=args.layers,
        )
        sim_mode = args.attn
        if sim_mode in {"ska", "ska_tok"}:
            sim_mode = "rbf"
        elif sim_mode == "ska_intersect":
            sim_mode = "intersect"
        if sim_mode != "dot":
            replace_multihead_attn(
                model,
                sim=sim_mode,
                rbf_gamma=args.gamma,
                inter_topk=args.inter_topk,
                inter_normalize=bool(args.inter_norm),
            )

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    pairs = list(zip(src_texts, tgt_texts))
    refs = [t.split() for t in tgt_texts]

    for epoch in range(1, args.epochs + 1):
        with profiler(args.profile) as prof:
            model.train()
            batch_refs, batch_hyps = [], []
            for batch_idx, src_ids, tgt_ids, ref_tokens in text_batch_iterator(
                pairs, dataset.src_stoi, dataset.tgt_stoi, refs, dataset.max_len, batch_size=64, shuffle=True
            ):
                batch_idx = batch_idx.to(device)
                src_ids = src_ids.to(device, non_blocking=True)
                tgt_ids = tgt_ids.to(device, non_blocking=True)
                tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

                if args.attn == "ska_tok" and src_store is not None:
                    src_vals, src_offs, src_sigs = src_store.gather(batch_idx, device)
                    tgt_vals, tgt_offs, tgt_sigs = tgt_store.gather(batch_idx, device)
                    logits = model(src_ids, tgt_in, src_vals, src_offs, tgt_vals, tgt_offs, src_sigs, tgt_sigs)
                else:
                    logits = model(src_ids, tgt_in)

                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                preds = logits.argmax(dim=-1)
                batch_refs.extend(ref_tokens)
                batch_hyps.extend([ids_to_tokens(row, dataset.tgt_itos) for row in preds.detach().cpu()])

        msg = f"[Seq2Seq-Toy][{args.attn}] epoch {epoch:02d} loss {loss.item():.4f}"
        if args.profile:
            msg += f" | time {prof['time_s']:.2f}s"
            if torch.cuda.is_available():
                msg += f" | peak VRAM {prof['gpu_peak_mem_mib']:.1f} MiB"
        print(msg)


if __name__ == "__main__":
    main()
