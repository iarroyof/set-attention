import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention.training.seq_loaders import get_seq2seq_datasets
from set_attention.patch import replace_multihead_attn
from set_attention.heads.ska_tokenized import SetKernelMultiheadAttentionTokenized
from set_attention.tokenizers.active_tokenizer import ACTIVE_TOKENIZER_TYPE
from set_attention.tokenizers.hf_bpe import HF_BPE_TYPE
from set_attention.tokenizers.hf_unigram import HF_UNIGRAM_TYPE
from set_attention.tokenizers.registry import (
    available_tokenizer_types,
    create_tokenizer,
    load_tokenizer,
    save_tokenizer,
)
from set_attention.training.text_utils import (
    build_token_set_store,
    encode_sentence,
    ids_to_tokens,
    text_batch_iterator,
)
from set_attention.experiments.nlp_eval import corpus_bleu, rouge_l
from set_attention.utils.profiling import profiler
from set_attention.utils.sample_logging import format_text_samples
from set_attention.utils.wandb import init_wandb


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
        self.cross_attn = SetKernelMultiheadAttentionTokenized(embed_dim=d_model, num_heads=nhead, batch_first=True, gamma=ska_gamma)
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


def run_seq2seq_benchmark(
    args,
    model,
    optimizer,
    tokenizer,
    src_store,
    tgt_store,
    train_pairs,
    train_src_stoi,
    train_tgt_stoi,
    train_refs,
    max_len,
    device,
    wandb_run,
):
    iterator = text_batch_iterator(
        train_pairs,
        train_src_stoi,
        train_tgt_stoi,
        train_refs,
        max_len,
        args.batch,
        shuffle=False,
    )
    try:
        batch_idx, src_ids, tgt_ids, refs = next(iterator)
    except StopIteration:
        print("[benchmark] insufficient data for benchmark batch.")
        return
    batch_idx = batch_idx.to(device)
    src_ids = src_ids.to(device, non_blocking=True)
    tgt_ids = tgt_ids.to(device, non_blocking=True)
    tgt_in = torch.cat([tgt_ids[:, :1], tgt_ids[:, :-1]], dim=1)

    def step():
        optimizer.zero_grad(set_to_none=True)
        if args.attn == "ska_tok" and src_store is not None and tgt_store is not None:
            src_vals, src_offs, src_sigs = src_store.gather(batch_idx, device)
            tgt_vals, tgt_offs, tgt_sigs = tgt_store.gather(batch_idx, device)
            logits = model(src_ids, tgt_in, src_vals, src_offs, tgt_vals, tgt_offs, src_sigs, tgt_sigs)
        else:
            logits = model(src_ids, tgt_in)
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
    tokens = tgt_ids.size(0) * max_len * args.bench_iters
    throughput = tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[benchmark][seq2seq-baseline] backend={args.attn} precision={args.precision if hasattr(args,'precision') else 'fp32'} "
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
    parser.add_argument("--attn", choices=["dot", "cosine", "rbf", "intersect", "ska_true", "ska_tok"], default="dot")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ska-gamma", type=float, default=0.3)
    parser.add_argument("--gate-topk", type=int, default=8)
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
    parser.add_argument("--token-set-k", type=int, default=64)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=10, help="Number of validation samples to log.")
    parser.add_argument("--sample-seed", type=int, default=1337, help="Seed for selecting logged samples.")
    args = parser.parse_args()

    if not args.dataset and not args.demo and not args.src and args.benchmark:
        print("[Info] No dataset provided for benchmark; enabling --demo mode.")
        args.demo = True

    wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_config = {
        "script": "train_seq2seq_text",
        "dataset": args.dataset or "custom",
        "attn": args.attn,
        "precision": args.precision,
        "tokenizer_type": args.tokenizer_type,
        "token_set_k": args.token_set_k,
        "batch": args.batch,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "layers": args.layers,
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

    tokenizer = None
    tokenizer_dir = args.tokenizer_dir
    needs_tokenizer = args.attn == "ska_tok" or bool(tokenizer_dir)
    if needs_tokenizer:
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

    if args.attn == "ska_tok":
        if tokenizer is None:
            raise RuntimeError("Tokenizer is required for --attn ska_tok.")
        src_store, tgt_store = build_token_set_store(train_ds, tokenizer, args.token_set_k, device)
        if has_val:
            val_src_store, val_tgt_store = build_token_set_store(val_ds, tokenizer, args.token_set_k, device)
        else:
            val_src_store = val_tgt_store = None
    else:
        src_store = tgt_store = val_src_store = val_tgt_store = None

    model: nn.Module
    if args.attn == "ska_tok":
        if tokenizer is None:
            raise RuntimeError("Tokenizer is required for --attn ska_tok.")
        model = TinySeq2SeqSKATok(
            src_vocab=len(train_src_stoi),
            tgt_vocab=len(train_tgt_stoi),
            d_model=args.d_model,
            nhead=args.nhead,
            layers=args.layers,
            ska_gamma=args.ska_gamma,
            gate_topk=args.gate_topk,
        )
        model.enable_gating(vocab_size=tokenizer.vocab_size(), atom_dim=args.d_model)
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

    if args.benchmark:
        run_seq2seq_benchmark(
            args,
            model,
            optimizer,
            tokenizer,
            src_store,
            tgt_store,
            train_pairs,
            train_src_stoi,
            train_tgt_stoi,
            train_refs,
            max_len,
            device,
            wandb_run,
        )
        wandb_run.finish()
        return

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_refs_epoch, train_hyps_epoch = [], []
        train_loss_total = 0.0
        train_token_total = 0
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

                train_loss_total += float(loss.detach()) * tgt_ids.numel()
                train_token_total += tgt_ids.numel()
                preds = logits.argmax(dim=-1)
                train_refs_epoch.extend(refs)
                train_hyps_epoch.extend([ids_to_tokens(row, train_tgt_itos) for row in preds.detach().cpu()])

        tr_loss = train_loss_total / max(1, train_token_total)
        tr_ppl = math.exp(min(tr_loss, 20.0))
        tr_bleu = corpus_bleu(train_refs_epoch, train_hyps_epoch)
        tr_rouge = rouge_l(train_refs_epoch, train_hyps_epoch)
        msg = (
            f"[Seq2Seq][{args.attn}] epoch {epoch:02d} "
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

        val_refs_epoch: List[List[str]] = []
        val_hyps_epoch: List[List[str]] = []
        val_src_epoch: List[str] = []
        v_loss = v_ppl = v_bleu = v_rouge = None
        if has_val:
            model.eval()
            val_loss_total = 0.0
            val_token_total = 0
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

                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_ids.reshape(-1))
                    val_loss_total += float(loss.detach()) * tgt_ids.numel()
                    val_token_total += tgt_ids.numel()
                    preds = logits.argmax(dim=-1)
                    val_refs_epoch.extend(refs)
                    val_hyps_epoch.extend([ids_to_tokens(row, train_tgt_itos) for row in preds.detach().cpu()])
                    batch_indices = batch_idx.detach().cpu().tolist()
                    val_src_epoch.extend(val_pairs[i][0] for i in batch_indices)

            v_loss = val_loss_total / max(1, val_token_total)
            v_ppl = math.exp(min(v_loss, 20.0))
            v_bleu = corpus_bleu(val_refs_epoch, val_hyps_epoch)
            v_rouge = rouge_l(val_refs_epoch, val_hyps_epoch)
            msg += f" | val loss {v_loss:.4f} ppl {v_ppl:.2f} BLEU {v_bleu:.3f} | ROUGE-L {v_rouge:.3f}"
        print(msg)

        if wandb_run.enabled:
            payload = {
                "train/loss": tr_loss,
                "train/ppl": tr_ppl,
                "train/bleu": tr_bleu,
                "train/rougeL": tr_rouge,
                "train/time_s": prof["time_s"],
            }
            if torch.cuda.is_available():
                payload["train/peak_vram_mib"] = prof["gpu_peak_mem_mib"]
            if v_loss is not None:
                payload.update(
                    {
                        "val/loss": v_loss,
                        "val/ppl": v_ppl,
                        "val/bleu": v_bleu,
                        "val/rougeL": v_rouge,
                    }
                )
                sample_text = format_text_samples(
                    val_refs_epoch,
                    val_hyps_epoch,
                    args.sample_count,
                    args.sample_seed + epoch,
                    sources=val_src_epoch,
                )
                if sample_text:
                    payload["samples/val_text"] = sample_text
            wandb_run.log(payload, step=epoch)

    wandb_run.finish()


if __name__ == "__main__":
    main()
