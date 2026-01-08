#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def _cmd(
    script: str,
    base_args: List[str],
    csv_path: Path,
    seed: int,
    rep: int,
    wandb_name: str,
    deterministic: bool,
    benchmark_deterministic: bool,
) -> List[str]:
    cmd = [sys.executable, script]
    cmd.extend(base_args)
    cmd.extend(["--benchmark-csv", str(csv_path)])
    cmd.extend(["--seed", str(seed)])
    cmd.extend(["--reps", "1"])
    if wandb_name:
        cmd.extend(["--wandb-run-name", wandb_name])
    if deterministic:
        cmd.append("--deterministic")
    if benchmark_deterministic:
        cmd.append("--benchmark-deterministic")
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
            "--dot-naive",
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
            "--dot-naive",
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
            "--dot-naive",
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
            "--dot-naive",
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
            "--dot-naive",
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
            "--dot-naive",
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
            "--dot-naive",
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
            "--dot-naive",
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
    parser.add_argument("--seed", type=int, default=2024, help="Default seed when --seeds is not provided.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma/space separated list of seeds (overrides --seed when provided).",
    )
    parser.add_argument("--reps", type=int, default=1, help="Repetitions per seed.")
    parser.add_argument(
        "--wandb-name-template",
        type=str,
        default="",
        help="Optional template for wandb run names, e.g. '{name}-s{seed}-r{rep}'.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Pass --deterministic to benchmark commands.",
    )
    parser.add_argument(
        "--benchmark-deterministic",
        action="store_true",
        help="Pass --benchmark-deterministic to benchmark commands.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    seed_values: List[int] = []
    if args.seeds:
        for part in args.seeds.replace(",", " ").split():
            part = part.strip()
            if not part:
                continue
            try:
                seed_values.append(int(part))
            except ValueError:
                continue
    if not seed_values:
        seed_values = [args.seed]
    reps = max(1, int(args.reps))

    tasks = list(BENCHMARKS.keys()) if "all" in args.tasks else args.tasks
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.device:
        env["CUDA_VISIBLE_DEVICES"] = args.device

    def append_status(csv_path: Path, row: dict) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        if csv_path.exists():
            with csv_path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                existing = reader.fieldnames or []
                rows = list(reader)
            extras = [c for c in fieldnames if c not in existing]
            if extras:
                new_fields = existing + extras
                with csv_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=new_fields)
                    writer.writeheader()
                    for prev in rows:
                        writer.writerow({c: prev.get(c, "") for c in new_fields})
                existing = new_fields
            with csv_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=existing)
                writer.writerow({c: row.get(c, "") for c in existing})
            return
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    for task in tasks:
        specs = BENCHMARKS[task]
        print(f"\n[bench-suite] task={task} ({len(specs)} runs)")
        for spec in specs:
            csv_path = out_dir / f"{spec['name']}.csv"
            if csv_path.exists():
                csv_path.unlink()
            for seed in seed_values:
                for rep in range(1, reps + 1):
                    wandb_name = ""
                    if args.wandb_name_template:
                        wandb_name = args.wandb_name_template.format(
                            name=spec["name"],
                            seed=seed,
                            rep=rep,
                            script=spec["script"],
                        )
                    cmd = _cmd(
                        spec["script"],
                        spec["args"],
                        csv_path,
                        seed,
                        rep,
                        wandb_name,
                        args.deterministic,
                        args.benchmark_deterministic,
                    )
                    print(f"[bench-suite] running (seed={seed} rep={rep}): {' '.join(cmd)}")
                    if args.dry_run:
                        continue
                    result = subprocess.run(cmd, env=env)
                    if result.returncode != 0:
                        append_status(
                            csv_path,
                            {
                                "script": spec["script"],
                                "task": task,
                                "dataset": spec.get("dataset", ""),
                                "dataset_id": spec.get("dataset", ""),
                                "seed": seed,
                                "rep": rep,
                                "run_uid": f"exitcode-{int(time.time())}-{seed}-{rep}",
                                "status": "exitcode",
                                "exit_code": result.returncode,
                                "skip_reason": f"exitcode={result.returncode}",
                            },
                        )
                        print(f"[bench-suite] command failed with code {result.returncode}")
                        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
