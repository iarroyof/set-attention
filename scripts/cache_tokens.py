#!/usr/bin/env python3
"""Precompute token caches for LM/Seq2Seq/TextDiff tasks."""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Build token caches for Stage A/B runs.")
    ap.add_argument("--task", choices=["lm", "seq2seq", "textdiff"], required=True)
    ap.add_argument("--artifact-cache-root", type=str, default="")
    ap.add_argument("--overwrite-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    # LM args
    ap.add_argument("--lm-dataset", type=str, default="wikitext2")
    ap.add_argument("--lm-subset-path", type=str, default="")
    ap.add_argument("--lm-seq-len", type=int, default=256)
    ap.add_argument("--lm-seq-stride", type=int, default=256)

    # Seq2Seq args
    ap.add_argument("--seq-dataset", type=str, default="wmt16_en_ro")
    ap.add_argument("--seq-limit", type=int, default=None)
    ap.add_argument("--seq-tokenizer-type", type=str, default="whitespace")

    # TextDiff args
    ap.add_argument("--textdiff-dataset", type=str, default="wikitext2")
    ap.add_argument("--textdiff-seq-len", type=int, default=64)
    ap.add_argument("--textdiff-stride", type=int, default=64)

    args = ap.parse_args()

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
            "--cache-mode",
            "tokens",
            "--cache-only",
        ]
        if args.lm_subset_path:
            cmd.extend(["--subset-path", args.lm_subset_path])
    elif args.task == "seq2seq":
        cmd = [
            sys.executable,
            "scripts/train_seq2seq_text_banked.py",
            "--dataset",
            args.seq_dataset,
            "--cache-mode",
            "tokens",
            "--cache-only",
            "--tokenizer-type",
            args.seq_tokenizer_type,
        ]
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
            "--cache-mode",
            "tokens",
            "--cache-only",
        ]

    if args.artifact_cache_root:
        cmd.extend(["--artifact-cache-root", args.artifact_cache_root])
    if args.overwrite_cache:
        cmd.append("--overwrite-cache")

    print("[cache_tokens]", " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
