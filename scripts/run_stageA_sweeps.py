#!/usr/bin/env python3
"""
Stage A launcher: run training+eval (metrics CSV) at fixed budgets for baseline vs SKA.
Uses the per-task scripts with metrics logging enabled; not a benchmark-only runner.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _parse_seeds(text: str, default: int) -> List[int]:
    if not text:
        return [default]
    seeds: List[int] = []
    for part in text.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError:
            continue
    return seeds or [default]


def _run(cmd: List[str], dry_run: bool) -> int:
    print("[stageA]", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Run Stage A quality sweeps (training + metrics logging).")
    ap.add_argument("--output-dir", type=str, default="out/stageA_runs")
    ap.add_argument("--seeds", type=str, default="")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true")

    # LM defaults
    ap.add_argument("--lm-dataset", type=str, default="wikitext2")
    ap.add_argument("--lm-subset-path", type=str, default="")
    ap.add_argument("--lm-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--lm-batch", type=int, default=8)
    ap.add_argument("--lm-seq-len", type=int, default=256)
    ap.add_argument("--lm-seq-stride", type=int, default=256)

    # Seq2Seq defaults
    ap.add_argument("--seq-dataset", type=str, default="wmt16_en_ro")
    ap.add_argument("--seq-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--seq-batch", type=int, default=32)

    # Diffusion text defaults
    ap.add_argument("--textdiff-dataset", type=str, default="wikitext2")
    ap.add_argument("--textdiff-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--textdiff-batch", type=int, default=64)
    ap.add_argument("--textdiff-seq-len", type=int, default=64)
    ap.add_argument("--textdiff-stride", type=int, default=64)

    # ViT defaults
    ap.add_argument("--vit-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--vit-batch", type=int, default=128)

    args = ap.parse_args()
    seeds = _parse_seeds(args.seeds, default=2024)
    reps = max(1, int(args.reps))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Helper to build a metrics filename
    def mpath(task: str, dataset: str, variant: str, precision: str, seed: int, rep: int) -> Path:
        return out_dir / f"metrics_{task}_{dataset}_{variant}_{precision}_s{seed}_r{rep}.csv"

    # LM runs (baseline and SKA)
    for variant in ("dot_explicit", "ska_python"):
        for seed in seeds:
            for rep in range(1, reps + 1):
                csv_path = mpath("lm", args.lm_dataset, variant, args.lm_precision, seed, rep)
                cmd = [
                    sys.executable,
                    "scripts/train_toy_lm_banked.py",
                    "--dataset",
                    args.lm_dataset,
                    "--epochs",
                    str(args.epochs),
                    "--batch",
                    str(args.lm_batch),
                    "--seq-len",
                    str(args.lm_seq_len),
                    "--seq-stride",
                    str(args.lm_seq_stride),
                    "--precision",
                    args.lm_precision,
                    "--eval-seed",
                    str(args.eval_seed),
                    "--metrics-csv",
                    str(csv_path),
                    "--seed",
                    str(seed),
                    "--reps",
                    str(1),
                ]
                if args.lm_subset_path:
                    cmd.extend(["--subset-path", args.lm_subset_path])
                if variant == "dot_explicit":
                    cmd.extend(["--sdpa-baseline", "--attn-baseline", "explicit", "--dot-naive"])
                else:
                    cmd.extend(
                        [
                            "--ska-backend",
                            "python",
                            "--window",
                            "64",
                            "--stride",
                            "32",
                            "--minhash-k",
                            "128",
                            "--router-topk",
                            "4",
                        ]
                    )
                rc = _run(cmd, args.dry_run)
                if rc != 0:
                    print(f"[stageA] command failed: {' '.join(cmd)}")

    # Seq2Seq runs
    for variant in ("dot_explicit", "ska_python"):
        for seed in seeds:
            for rep in range(1, reps + 1):
                csv_path = mpath("seq", args.seq_dataset, variant, args.seq_precision, seed, rep)
                cmd = [
                    sys.executable,
                    "scripts/train_seq2seq_text_banked.py",
                    "--dataset",
                    args.seq_dataset,
                    "--epochs",
                    str(args.epochs),
                    "--batch",
                    str(args.seq_batch),
                    "--precision",
                    args.seq_precision,
                    "--eval-seed",
                    str(args.eval_seed),
                    "--metrics-csv",
                    str(csv_path),
                    "--seed",
                    str(seed),
                    "--reps",
                    str(1),
                ]
                if variant == "dot_explicit":
                    cmd.extend(["--sdpa-baseline", "--attn-baseline", "explicit", "--dot-naive"])
                else:
                    cmd.extend(
                        [
                            "--ska-backend",
                            "python",
                            "--window",
                            "64",
                            "--stride",
                            "32",
                            "--minhash-k",
                            "128",
                            "--router-topk",
                            "4",
                        ]
                    )
                rc = _run(cmd, args.dry_run)
                if rc != 0:
                    print(f"[stageA] command failed: {' '.join(cmd)}")

    # Text diffusion runs
    for variant in ("dot_explicit", "ska_python"):
        for seed in seeds:
            for rep in range(1, reps + 1):
                csv_path = mpath("textdiff", args.textdiff_dataset, variant, args.textdiff_precision, seed, rep)
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
                    "--epochs",
                    str(args.epochs),
                    "--batch",
                    str(args.textdiff_batch),
                    "--precision",
                    args.textdiff_precision,
                    "--eval-seed",
                    str(args.eval_seed),
                    "--metrics-csv",
                    str(csv_path),
                    "--seed",
                    str(seed),
                    "--reps",
                    str(1),
                ]
                if variant == "dot_explicit":
                    cmd.extend(["--sdpa-baseline", "--attn-baseline", "explicit", "--dot-naive"])
                else:
                    cmd.extend(
                        [
                            "--ska-backend",
                            "python",
                            "--window",
                            "64",
                            "--stride",
                            "32",
                            "--minhash-k",
                            "128",
                            "--router-topk",
                            "4",
                        ]
                    )
                rc = _run(cmd, args.dry_run)
                if rc != 0:
                    print(f"[stageA] command failed: {' '.join(cmd)}")

    # ViT runs
    for variant in ("dot_explicit", "ska_python"):
        for seed in seeds:
            for rep in range(1, reps + 1):
                csv_path = mpath("vit", "cifar10", variant, args.vit_precision, seed, rep)
                cmd = [
                    sys.executable,
                    "scripts/train_tiny_vit_banked.py",
                    "--data-mode",
                    "cifar10",
                    "--epochs",
                    str(args.epochs),
                    "--batch",
                    str(args.vit_batch),
                    "--precision",
                    args.vit_precision,
                    "--eval-seed",
                    str(args.eval_seed),
                    "--metrics-csv",
                    str(csv_path),
                    "--seed",
                    str(seed),
                    "--reps",
                    str(1),
                ]
                if variant == "dot_explicit":
                    cmd.extend(["--sdpa-baseline", "--attn-baseline", "explicit", "--dot-naive"])
                else:
                    cmd.extend(
                        [
                            "--ska-backend",
                            "python",
                            "--window",
                            "8",
                            "--stride",
                            "4",
                            "--minhash-k",
                            "64",
                            "--router-topk",
                            "0",
                        ]
                    )
                rc = _run(cmd, args.dry_run)
                if rc != 0:
                    print(f"[stageA] command failed: {' '.join(cmd)}")


if __name__ == "__main__":
    main()
