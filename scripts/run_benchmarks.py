#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _cmd(script: str, args: List[str], csv_path: Path) -> List[str]:
    cmd = [sys.executable, script]
    cmd.extend(args)
    cmd.extend(["--benchmark-csv", str(csv_path)])
    return cmd


LM_BENCHES = [
    {
        "name": "lm_sdpa",
        "script": "scripts/train_toy_lm_banked.py",
        "args": [
            "--dataset",
            "wikitext2",
            "--streaming",
            "--sdpa-baseline",
            "--benchmark",
            "--bench-warmup",
            "20",
            "--bench-iters",
            "100",
            "--batch",
            "8",
            "--seq-len",
            "256",
            "--seq-stride",
            "256",
        ],
    },
    {
        "name": "lm_ska_keops",
        "script": "scripts/train_toy_lm_banked.py",
        "args": [
            "--dataset",
            "wikitext2",
            "--streaming",
            "--ska-backend",
            "keops",
            "--precision",
            "bf16",
            "--router-topk",
            "4",
            "--window",
            "64",
            "--stride",
            "32",
            "--minhash-k",
            "128",
            "--benchmark",
            "--bench-warmup",
            "20",
            "--bench-iters",
            "100",
            "--batch",
            "8",
            "--seq-len",
            "256",
            "--seq-stride",
            "256",
        ],
    },
]

SEQ_BENCHES = [
    {
        "name": "seq_sdpa",
        "script": "scripts/train_seq2seq_text_banked.py",
        "args": [
            "--dataset",
            "wmt16_en_ro",
            "--sdpa-baseline",
            "--benchmark",
            "--bench-warmup",
            "10",
            "--bench-iters",
            "50",
            "--batch",
            "16",
        ],
    },
    {
        "name": "seq_ska_keops",
        "script": "scripts/train_seq2seq_text_banked.py",
        "args": [
            "--dataset",
            "wmt16_en_ro",
            "--ska-backend",
            "keops",
            "--precision",
            "bf16",
            "--router-topk",
            "4",
            "--window",
            "8",
            "--stride",
            "4",
            "--minhash-k",
            "128",
            "--benchmark",
            "--bench-warmup",
            "10",
            "--bench-iters",
            "50",
            "--batch",
            "16",
        ],
    },
]

DIFF_BENCHES = [
    {
        "name": "diff_sdpa",
        "script": "scripts/train_toy_diffusion_banked.py",
        "args": [
            "--config",
            "configs/diffusion_toy.yaml",
            "--sdpa-baseline",
            "--benchmark",
            "--bench-warmup",
            "10",
            "--bench-iters",
            "50",
            "--batch",
            "64",
        ],
    },
    {
        "name": "diff_ska_keops",
        "script": "scripts/train_toy_diffusion_banked.py",
        "args": [
            "--config",
            "configs/diffusion_toy.yaml",
            "--ska-backend",
            "keops",
            "--precision",
            "bf16",
            "--router-topk",
            "4",
            "--window",
            "8",
            "--stride",
            "8",
            "--minhash-k",
            "128",
            "--benchmark",
            "--bench-warmup",
            "10",
            "--bench-iters",
            "50",
            "--batch",
            "64",
        ],
    },
]

VIT_BENCHES = [
    {
        "name": "vit_sdpa",
        "script": "scripts/train_tiny_vit_banked.py",
        "args": [
            "--data-mode",
            "cifar10",
            "--sdpa-baseline",
            "--limit",
            "1024",
            "--benchmark",
            "--bench-warmup",
            "10",
            "--bench-iters",
            "50",
            "--batch",
            "128",
        ],
    },
    {
        "name": "vit_ska_keops",
        "script": "scripts/train_tiny_vit_banked.py",
        "args": [
            "--data-mode",
            "cifar10",
            "--limit",
            "1024",
            "--ska-backend",
            "keops",
            "--precision",
            "bf16",
            "--router-topk",
            "4",
            "--window",
            "14",
            "--stride",
            "14",
            "--minhash-k",
            "128",
            "--benchmark",
            "--bench-warmup",
            "10",
            "--bench-iters",
            "50",
            "--batch",
            "128",
        ],
    },
]


BENCHMARKS: Dict[str, List[Dict[str, List[str]]]] = {
    "lm": LM_BENCHES,
    "seq2seq": SEQ_BENCHES,
    "diffusion": DIFF_BENCHES,
    "vit": VIT_BENCHES,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SKA vs SDPA benchmark suites.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default=["all"],
        help="Which task suites to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/benchmarks",
        help="Directory to store benchmark CSV files.",
    )
    parser.add_argument("--device", type=str, default="", help="CUDA_VISIBLE_DEVICES value (optional).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    tasks = list(BENCHMARKS.keys()) if "all" in args.tasks else args.tasks
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.device:
        env["CUDA_VISIBLE_DEVICES"] = args.device

    for task in tasks:
        specs = BENCHMARKS[task]
        print(f"\n[bench-suite] task={task} ({len(specs)} runs)")
        for spec in specs:
            csv_path = out_dir / f"{spec['name']}.csv"
            cmd = _cmd(spec["script"], spec["args"], csv_path)
            print(f"[bench-suite] running: {' '.join(cmd)}")
            if args.dry_run:
                continue
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                print(f"[bench-suite] command failed with code {result.returncode}")
                sys.exit(result.returncode)


if __name__ == "__main__":
    main()
