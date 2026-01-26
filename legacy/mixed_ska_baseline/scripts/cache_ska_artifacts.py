#!/usr/bin/env python3
"""Precompute SKA bank+routing caches for LM/Seq2Seq/TextDiff tasks."""
from __future__ import annotations

import argparse
import subprocess
import sys
import shlex


def main() -> int:
    ap = argparse.ArgumentParser(description="Build full SKA caches (tokens + banks + routing).")
    ap.add_argument("--task", choices=["lm", "seq2seq", "textdiff"], required=True)
    ap.add_argument("--artifact-cache-root", type=str, default="")
    ap.add_argument("--overwrite-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--common-args",
        action="append",
        default=[],
        help="Extra CLI args appended to the task cache command (repeatable, quoted string).",
    )
    ap.add_argument(
        "--lm-args",
        action="append",
        default=[],
        help="Extra CLI args appended to LM cache commands (repeatable, quoted string).",
    )
    ap.add_argument(
        "--seq-args",
        action="append",
        default=[],
        help="Extra CLI args appended to Seq2Seq cache commands (repeatable, quoted string).",
    )
    ap.add_argument(
        "--textdiff-args",
        action="append",
        default=[],
        help="Extra CLI args appended to TextDiff cache commands (repeatable, quoted string).",
    )

    # LM args
    ap.add_argument("--lm-dataset", type=str, default="wikitext2")
    ap.add_argument("--lm-subset-path", type=str, default="")
    ap.add_argument("--lm-limit", type=int, default=None, help="Alias for --limit in LM runs.")
    ap.add_argument("--lm-seq-len", type=int, default=256)
    ap.add_argument("--lm-seq-stride", type=int, default=256)
    ap.add_argument("--lm-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--lm-window", type=int, default=64)
    ap.add_argument("--lm-stride", type=int, default=32)
    ap.add_argument("--lm-minhash-k", type=int, default=128)
    ap.add_argument("--lm-router-topk", type=int, default=4)

    # Seq2Seq args
    ap.add_argument("--seq-dataset", type=str, default="wmt16_en_ro")
    ap.add_argument("--seq-subset-path", type=str, default="")
    ap.add_argument("--seq-limit", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Alias for --seq-limit.")
    ap.add_argument("--seq-tokenizer-type", type=str, default="whitespace")
    ap.add_argument("--seq-max-len", type=int, default=256)
    ap.add_argument("--seq-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--seq-window", type=int, default=64)
    ap.add_argument("--seq-stride", type=int, default=32)
    ap.add_argument("--seq-minhash-k", type=int, default=128)
    ap.add_argument("--seq-router-topk", type=int, default=4)

    # TextDiff args
    ap.add_argument("--textdiff-dataset", type=str, default="wikitext2")
    ap.add_argument("--textdiff-subset-path", type=str, default="")
    ap.add_argument("--textdiff-seq-len", type=int, default=256)
    ap.add_argument("--textdiff-stride", type=int, default=256)
    ap.add_argument("--textdiff-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--textdiff-window", type=int, default=64)
    ap.add_argument("--textdiff-bank-stride", type=int, default=32)
    ap.add_argument("--textdiff-minhash-k", type=int, default=128)
    ap.add_argument("--textdiff-router-topk", type=int, default=4)

    args, unknown = ap.parse_known_args()
    if args.limit is not None and args.seq_limit is None:
        args.seq_limit = args.limit

    def _parse_extra(values: list[str]) -> list[str]:
        extra: list[str] = []
        for value in values:
            if value:
                extra.extend(shlex.split(value))
        return extra

    common_args = _parse_extra(args.common_args)
    lm_args = _parse_extra(args.lm_args)
    seq_args = _parse_extra(args.seq_args)
    textdiff_args = _parse_extra(args.textdiff_args)

    cmd: list[str]
    if args.task == "lm":
        cmd = [
            sys.executable,
            "scripts/train_toy_lm_banked.py",
            "--dataset",
            args.lm_dataset,
            "--seq-len",
            str(args.lm_seq_len),
            "--seq-stride",
            str(args.lm_seq_stride),
            "--precision",
            args.lm_precision,
            "--window",
            str(args.lm_window),
            "--stride",
            str(args.lm_stride),
            "--minhash-k",
            str(args.lm_minhash_k),
            "--router-topk",
            str(args.lm_router_topk),
            "--cache-mode",
            "full",
            "--cache-only",
            "--precompute-bank",
        ]
        if args.lm_subset_path:
            cmd.extend(["--subset-path", args.lm_subset_path])
        if args.lm_limit is not None:
            cmd.extend(["--limit", str(args.lm_limit)])
    elif args.task == "seq2seq":
        cmd = [
            sys.executable,
            "scripts/train_seq2seq_text_banked.py",
            "--dataset",
            args.seq_dataset,
            "--max-len",
            str(args.seq_max_len),
            "--tokenizer-type",
            args.seq_tokenizer_type,
            "--window",
            str(args.seq_window),
            "--stride",
            str(args.seq_stride),
            "--minhash-k",
            str(args.seq_minhash_k),
            "--router-topk",
            str(args.seq_router_topk),
            "--precision",
            args.seq_precision,
            "--cache-mode",
            "full",
            "--cache-only",
            "--precompute-bank",
        ]
        if args.seq_subset_path:
            cmd.extend(["--subset-path", args.seq_subset_path])
        if args.seq_limit is not None:
            cmd.extend(["--limit", str(args.seq_limit)])
    else:
        cmd = [
            sys.executable,
            "scripts/train_toy_diffusion_banked.py",
            "--data-mode",
            "text",
            "--text-dataset",
            args.textdiff_dataset,
            "--text-seq-len",
            str(args.textdiff_seq_len),
            "--text-stride",
            str(args.textdiff_stride),
            "--precision",
            args.textdiff_precision,
            "--window",
            str(args.textdiff_window),
            "--stride",
            str(args.textdiff_bank_stride),
            "--minhash-k",
            str(args.textdiff_minhash_k),
            "--router-topk",
            str(args.textdiff_router_topk),
            "--cache-mode",
            "full",
            "--cache-only",
            "--precompute-bank",
        ]
        if args.textdiff_subset_path:
            cmd.extend(["--text-subset-path", args.textdiff_subset_path])

    if args.artifact_cache_root:
        cmd.extend(["--artifact-cache-root", args.artifact_cache_root])
    if args.overwrite_cache:
        cmd.append("--overwrite-cache")
    if common_args:
        cmd.extend(common_args)
    if args.task == "lm" and lm_args:
        cmd.extend(lm_args)
    if args.task == "seq2seq" and seq_args:
        cmd.extend(seq_args)
    if args.task == "textdiff" and textdiff_args:
        cmd.extend(textdiff_args)
    if unknown:
        cmd.extend(unknown)

    print("[cache_ska_artifacts]", " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
