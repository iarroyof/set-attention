#!/usr/bin/env python3
"""
Stage B scaling sweep launcher.
Runs length/budget/capacity grids for the benchmark scripts and respects skip/oom flags.
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
    print("[sweep]", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Run Stage B scaling sweeps with skip-aware configs.")
    ap.add_argument("--output-dir", type=str, default="out/benchmarks_scale")
    ap.add_argument("--gpu-vram", type=float, default=24.0, help="GPU VRAM in GB for skip estimation.")
    ap.add_argument("--seeds", type=str, default="")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")

    # LM length sweep defaults
    ap.add_argument("--lm-lengths", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--lm-batch", type=int, default=8)
    ap.add_argument("--lm-dataset", type=str, default="wikitext103")
    ap.add_argument("--lm-subset-path", type=str, default="subsets/wikitext103_train_10pct.json")
    ap.add_argument("--lm-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])

    # Seq2Seq lengths (optional; default empty)
    ap.add_argument("--seq-lengths", type=int, nargs="+", default=[])
    ap.add_argument("--seq-batch", type=int, default=16)
    ap.add_argument("--seq-dataset", type=str, default="wmt16_en_ro")
    ap.add_argument("--seq-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])

    # Diffusion/text lengths (optional; default empty)
    ap.add_argument("--textdiff-lengths", type=int, nargs="+", default=[])
    ap.add_argument("--textdiff-batch", type=int, default=64)
    ap.add_argument("--textdiff-dataset", type=str, default="wikitext2")
    ap.add_argument("--textdiff-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])

    # ViT (no length sweep; uses patch-based)
    ap.add_argument("--vit-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])

    args = ap.parse_args()
    seeds = _parse_seeds(args.seeds, default=2024)
    reps = max(1, int(args.reps))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # LM sweeps
    for L in args.lm_lengths:
        for mode in ("dot_explicit", "ska_python"):
            for seed in seeds:
                for rep in range(1, reps + 1):
                    csv_name = f"lm_{args.lm_dataset}_L{L}_{mode}_s{seed}_r{rep}.csv"
                    cmd = [
                        sys.executable,
                        "scripts/train_toy_lm_banked.py",
                        "--dataset",
                        args.lm_dataset,
                        "--subset-path",
                        args.lm_subset_path,
                        "--benchmark",
                        "--bench-warmup",
                        "5",
                        "--bench-iters",
                        "20",
                        "--batch",
                        str(args.lm_batch),
                        "--seq-len",
                        str(L),
                        "--seq-stride",
                        str(L),
                        "--precision",
                        args.lm_precision,
                        "--gpu-vram",
                        str(args.gpu_vram),
                        "--skip-oom",
                        "--benchmark-csv",
                        str(out_dir / csv_name),
                        "--seed",
                        str(seed),
                        "--reps",
                        str(1),
                    ]
                    if mode == "dot_explicit":
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
                        print(f"[sweep] command failed with code {rc}: {' '.join(cmd)}")

    # Seq2Seq sweeps (optional)
    for L in args.seq_lengths:
        for mode in ("dot_explicit", "ska_python"):
            for seed in seeds:
                for rep in range(1, reps + 1):
                    csv_name = f"seq_{args.seq_dataset}_L{L}_{mode}_s{seed}_r{rep}.csv"
                    cmd = [
                        sys.executable,
                        "scripts/train_seq2seq_text_banked.py",
                        "--dataset",
                        args.seq_dataset,
                        "--benchmark",
                        "--bench-warmup",
                        "5",
                        "--bench-iters",
                        "20",
                        "--batch",
                        str(args.seq_batch),
                        "--precision",
                        args.seq_precision,
                        "--gpu-vram",
                        str(args.gpu_vram),
                        "--skip-oom",
                        "--benchmark-csv",
                        str(out_dir / csv_name),
                        "--seed",
                        str(seed),
                        "--reps",
                        str(1),
                    ]
                    if mode == "dot_explicit":
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
                        print(f"[sweep] command failed with code {rc}: {' '.join(cmd)}")

    # Text diffusion sweeps (optional)
    for L in args.textdiff_lengths:
        for mode in ("dot_explicit", "ska_python"):
            for seed in seeds:
                for rep in range(1, reps + 1):
                    csv_name = f"textdiff_{args.textdiff_dataset}_L{L}_{mode}_s{seed}_r{rep}.csv"
                    cmd = [
                        sys.executable,
                        "scripts/train_toy_diffusion_banked.py",
                        "--data-mode",
                        "text",
                        "--text-dataset",
                        args.textdiff_dataset,
                        "--text-seq-len",
                        str(L),
                        "--text-stride",
                        str(L),
                        "--benchmark",
                        "--bench-warmup",
                        "5",
                        "--bench-iters",
                        "20",
                        "--batch",
                        str(args.textdiff_batch),
                        "--precision",
                        args.textdiff_precision,
                        "--gpu-vram",
                        str(args.gpu_vram),
                        "--skip-oom",
                        "--benchmark-csv",
                        str(out_dir / csv_name),
                        "--seed",
                        str(seed),
                        "--reps",
                        str(1),
                    ]
                    if mode == "dot_explicit":
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
                        print(f"[sweep] command failed with code {rc}: {' '.join(cmd)}")

    # ViT sweeps (no length axis; single run per variant)
    for mode in ("dot_explicit", "ska_python"):
        for seed in seeds:
            for rep in range(1, reps + 1):
                csv_name = f"vit_{mode}_s{seed}_r{rep}.csv"
                cmd = [
                    sys.executable,
                    "scripts/train_tiny_vit_banked.py",
                    "--data-mode",
                    "cifar10",
                    "--benchmark",
                    "--bench-warmup",
                    "5",
                    "--bench-iters",
                    "20",
                    "--batch",
                    "128",
                    "--precision",
                    args.vit_precision,
                    "--gpu-vram",
                    str(args.gpu_vram),
                    "--skip-oom",
                    "--benchmark-csv",
                    str(out_dir / csv_name),
                    "--seed",
                    str(seed),
                    "--reps",
                    str(1),
                ]
                if mode == "dot_explicit":
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
                    print(f"[sweep] command failed with code {rc}: {' '.join(cmd)}")


if __name__ == "__main__":
    main()
