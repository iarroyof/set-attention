#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import tempfile
import shlex
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


def _visible_gpu_indices() -> List[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        return []
    indices: List[int] = []
    for part in visible.split(","):
        part = part.strip()
        if not part or part == "-1":
            continue
        try:
            indices.append(int(part))
        except ValueError:
            continue
    return indices


def _gpu_uuid_map() -> dict[int, str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
        )
    except Exception:
        return {}
    mapping: dict[int, str] = {}
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        mapping[idx] = parts[1]
    return mapping


def _selected_gpu_uuids() -> List[str]:
    mapping = _gpu_uuid_map()
    if not mapping:
        return []
    visible = _visible_gpu_indices()
    if visible:
        return [mapping[idx] for idx in visible if idx in mapping]
    return list(mapping.values())


def _gpu_compute_procs() -> List[dict]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception as exc:
        print(f"[bench-suite] warn: nvidia-smi compute query failed ({exc}); skipping process gate.")
        return []
    selected = set(_selected_gpu_uuids())
    procs: List[dict] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid_raw, mem_raw = parts[:3]
        if selected and uuid not in selected:
            continue
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        try:
            mem_mb = float(mem_raw)
        except ValueError:
            mem_mb = 0.0
        procs.append({"gpu_uuid": uuid, "pid": pid, "used_mb": mem_mb})
    return procs


def _format_procs(procs: List[dict]) -> str:
    return ", ".join([f"pid={p['pid']} mem={p['used_mb']:.0f}MB" for p in procs])


def _gpu_free_gb() -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception as exc:
        print(f"[bench-suite] warn: nvidia-smi unavailable ({exc}); skipping GPU wait.")
        return None
    values: List[float] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        part = line.split()[0].replace(",", "")
        try:
            values.append(float(part))
        except ValueError:
            continue
    if not values:
        return None
    visible = _visible_gpu_indices()
    if visible:
        selected = [values[i] for i in visible if i < len(values)]
        if selected:
            return min(selected) / 1024.0
    return values[0] / 1024.0


def _wait_for_gpu(min_free_gb: float, interval_s: float, timeout_s: float, require_idle: bool) -> bool:
    start = time.time()
    while True:
        procs = _gpu_compute_procs()
        free_gb = _gpu_free_gb()
        free_ok = free_gb is None or free_gb >= min_free_gb or min_free_gb <= 0
        idle_ok = not procs if require_idle else True
        if idle_ok and free_ok:
            return True
        if timeout_s > 0 and (time.time() - start) >= timeout_s:
            if procs:
                print(f"[bench-suite] warn: GPU busy ({_format_procs(procs)}) (timeout).")
            if free_gb is not None:
                print(f"[bench-suite] warn: GPU free {free_gb:.2f} GB < {min_free_gb:.2f} GB (timeout).")
            return False
        if require_idle and procs:
            print(f"[bench-suite] waiting for GPU idle; busy: {_format_procs(procs)}")
        if min_free_gb > 0 and free_gb is not None and free_gb < min_free_gb:
            print(f"[bench-suite] waiting for GPU free {free_gb:.2f} GB < {min_free_gb:.2f} GB")
        time.sleep(interval_s)


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
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.0,
        help="Wait for this much free GPU memory before running each job (0=disable).",
    )
    parser.add_argument(
        "--wait-gpu-interval",
        type=float,
        default=10.0,
        help="Seconds between GPU free-memory checks.",
    )
    parser.add_argument(
        "--wait-gpu-timeout",
        type=float,
        default=0.0,
        help="Timeout in seconds for GPU wait (0=wait forever).",
    )
    parser.add_argument(
        "--require-idle-gpu",
        action="store_true",
        default=True,
        help="Wait for GPU to have no active compute processes (default: enabled).",
    )
    parser.add_argument(
        "--no-require-idle-gpu",
        dest="require_idle_gpu",
        action="store_false",
        help="Disable GPU process-idle gating.",
    )
    parser.add_argument(
        "--post-run-grace",
        type=float,
        default=2.0,
        help="Seconds to wait after each job before checking GPU processes.",
    )
    parser.add_argument(
        "--post-run-wait",
        action="store_true",
        help="Wait for GPU idle after each job (in addition to warnings).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--common-args",
        action="append",
        default=[],
        help="Extra CLI args appended to all task runs (repeatable, quoted string).",
    )
    parser.add_argument(
        "--lm-args",
        action="append",
        default=[],
        help="Extra CLI args appended to LM runs (repeatable, quoted string).",
    )
    parser.add_argument(
        "--seq-args",
        action="append",
        default=[],
        help="Extra CLI args appended to Seq2Seq runs (repeatable, quoted string).",
    )
    parser.add_argument(
        "--diff-args",
        action="append",
        default=[],
        help="Extra CLI args appended to Diffusion runs (repeatable, quoted string).",
    )
    parser.add_argument(
        "--vit-args",
        action="append",
        default=[],
        help="Extra CLI args appended to ViT runs (repeatable, quoted string).",
    )
    args = parser.parse_args()

    def _parse_extra(values: list[str]) -> list[str]:
        extra: list[str] = []
        for value in values:
            if value:
                extra.extend(shlex.split(value))
        return extra

    common_args = _parse_extra(args.common_args)
    task_args = {
        "lm": _parse_extra(args.lm_args),
        "seq2seq": _parse_extra(args.seq_args),
        "diffusion": _parse_extra(args.diff_args),
        "vit": _parse_extra(args.vit_args),
    }

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
                        list(spec["args"]) + common_args + task_args.get(task, []),
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
                    if not _wait_for_gpu(
                        args.min_free_gb,
                        args.wait_gpu_interval,
                        args.wait_gpu_timeout,
                        args.require_idle_gpu,
                    ):
                        append_status(
                            csv_path,
                            {
                                "script": spec["script"],
                                "task": task,
                                "dataset": spec.get("dataset", ""),
                                "dataset_id": spec.get("dataset", ""),
                                "seed": seed,
                                "rep": rep,
                                "run_uid": f"skip-{int(time.time())}-{seed}-{rep}",
                                "status": "skipped",
                                "skip_reason": "gpu_busy",
                            },
                        )
                        continue
                    with tempfile.TemporaryFile(mode="w+") as stderr_file:
                        proc = subprocess.Popen(cmd, env=env, stderr=stderr_file)
                        returncode = proc.wait()
                        if returncode != 0:
                            stderr_file.seek(0)
                            tail = stderr_file.read().splitlines()[-20:]
                            if tail:
                                print("[bench-suite] stderr (tail):")
                                for line in tail:
                                    print(f"[bench-suite] | {line}")
                    if args.post_run_grace > 0:
                        time.sleep(args.post_run_grace)
                    procs = _gpu_compute_procs()
                    if procs:
                        still = _format_procs(procs)
                        if proc.pid in {p["pid"] for p in procs}:
                            print(f"[bench-suite] warn: job pid {proc.pid} still on GPU: {still}")
                        else:
                            print(f"[bench-suite] warn: GPU still busy after run: {still}")
                        if args.post_run_wait:
                            _wait_for_gpu(
                                args.min_free_gb,
                                args.wait_gpu_interval,
                                args.wait_gpu_timeout,
                                args.require_idle_gpu,
                            )
                    if returncode != 0:
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
                                "exit_code": returncode,
                                "skip_reason": f"exitcode={returncode}",
                            },
                        )
                        print(f"[bench-suite] command failed with code {returncode}")
                        sys.exit(returncode)


if __name__ == "__main__":
    main()
