#!/usr/bin/env python3
"""
Stage A launcher: run training+eval (metrics CSV) at fixed budgets for baseline vs SKA.
Uses the per-task scripts with metrics logging enabled; not a benchmark-only runner.
"""
import argparse
import csv
import os
import subprocess
import tempfile
import sys
import time
from pathlib import Path
from typing import List, Optional


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
        print(f"[stageA] warn: nvidia-smi compute query failed ({exc}); skipping process gate.")
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
        print(f"[stageA] warn: nvidia-smi unavailable ({exc}); skipping GPU wait.")
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
                print(f"[stageA] warn: GPU busy ({_format_procs(procs)}) (timeout).")
            print(
                f"[stageA] warn: GPU free {free_gb:.2f} GB < {min_free_gb:.2f} GB (timeout)."
            )
            return False
        if require_idle and procs:
            print(f"[stageA] waiting for GPU idle; busy: {_format_procs(procs)}")
        if min_free_gb > 0 and free_gb is not None and free_gb < min_free_gb:
            print(f"[stageA] waiting for GPU free {free_gb:.2f} GB < {min_free_gb:.2f} GB")
        time.sleep(interval_s)


def _run(
    cmd: List[str],
    dry_run: bool,
    min_free_gb: float,
    wait_interval: float,
    wait_timeout: float,
    require_idle: bool,
    post_run_grace: float,
    post_run_wait: bool,
) -> int:
    print("[stageA]", " ".join(cmd))
    if dry_run:
        return 0
    if not _wait_for_gpu(min_free_gb, wait_interval, wait_timeout, require_idle):
        return 1
    with tempfile.TemporaryFile(mode="w+") as stderr_file:
        proc = subprocess.Popen(cmd, stderr=stderr_file)
        rc = proc.wait()
        if rc != 0:
            stderr_file.seek(0)
            tail = stderr_file.read().splitlines()[-20:]
            if tail:
                print("[stageA] stderr (tail):")
                for line in tail:
                    print(f"[stageA] | {line}")
    if post_run_grace > 0:
        time.sleep(post_run_grace)
    procs = _gpu_compute_procs()
    if procs:
        still = _format_procs(procs)
        if proc.pid in {p["pid"] for p in procs}:
            print(f"[stageA] warn: job pid {proc.pid} still on GPU: {still}")
        else:
            print(f"[stageA] warn: GPU still busy after run: {still}")
        if post_run_wait:
            _wait_for_gpu(min_free_gb, wait_interval, wait_timeout, require_idle)
    return rc


def _append_status_row(csv_path: Path, row: dict) -> None:
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


def main():
    ap = argparse.ArgumentParser(description="Run Stage A quality sweeps (training + metrics logging).")
    ap.add_argument("--output-dir", type=str, default="out/stageA_runs")
    ap.add_argument("--seeds", type=str, default="")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--profile", action="store_true", help="Enable per-epoch profiling (time/VRAM) in task scripts.")
    ap.add_argument(
        "--skip-oom",
        action="store_true",
        default=True,
        help="Skip OOMs in task scripts and record status rows (default: enabled).",
    )
    ap.add_argument(
        "--no-skip-oom",
        dest="skip_oom",
        action="store_false",
        help="Disable OOM skipping and let task scripts raise.",
    )
    ap.add_argument("--cache-mode", choices=["none", "tokens", "full"], default="none")
    ap.add_argument("--artifact-cache-root", type=str, default="")
    ap.add_argument("--overwrite-cache", action="store_true")
    ap.add_argument("--precache", action="store_true", help="Precompute caches before running Stage A.")
    ap.add_argument("--min-free-gb", type=float, default=0.0, help="Wait for this much free GPU memory before running each job (0=disable).")
    ap.add_argument("--wait-gpu-interval", type=float, default=10.0, help="Seconds between GPU free-memory checks.")
    ap.add_argument("--wait-gpu-timeout", type=float, default=0.0, help="Timeout in seconds for GPU wait (0=wait forever).")
    ap.add_argument("--require-idle-gpu", action="store_true", default=True, help="Wait for GPU to have no active compute processes (default: enabled).")
    ap.add_argument("--no-require-idle-gpu", dest="require_idle_gpu", action="store_false", help="Disable GPU process-idle gating.")
    ap.add_argument("--post-run-grace", type=float, default=2.0, help="Seconds to wait after each job before checking GPU processes.")
    ap.add_argument("--post-run-wait", action="store_true", help="Wait for GPU idle after each job (in addition to warnings).")
    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="If >0, set OMP/MKL/OPENBLAS/NUMEXPR threads for child runs.",
    )

    # LM defaults
    ap.add_argument("--lm-dataset", type=str, default="wikitext2")
    ap.add_argument("--lm-subset-path", type=str, default="")
    ap.add_argument("--lm-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--lm-batch", type=int, default=8)
    ap.add_argument("--lm-seq-len", type=int, default=256)
    ap.add_argument("--lm-seq-stride", type=int, default=256)
    ap.add_argument("--lm-num-workers", type=int, default=0)

    # Seq2Seq defaults
    ap.add_argument("--seq-dataset", type=str, default="wmt16_en_ro")
    ap.add_argument("--seq-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--seq-batch", type=int, default=32)
    ap.add_argument("--seq-num-workers", type=int, default=0)

    # Diffusion text defaults
    ap.add_argument("--textdiff-dataset", type=str, default="wikitext2")
    ap.add_argument("--textdiff-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--textdiff-batch", type=int, default=64)
    ap.add_argument("--textdiff-seq-len", type=int, default=64)
    ap.add_argument("--textdiff-stride", type=int, default=64)
    ap.add_argument("--textdiff-num-workers", type=int, default=0)

    # ViT defaults
    ap.add_argument("--vit-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--vit-batch", type=int, default=128)
    ap.add_argument("--vit-num-workers", type=int, default=0)

    args = ap.parse_args()
    seeds = _parse_seeds(args.seeds, default=2024)
    reps = max(1, int(args.reps))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu_threads > 0:
        value = str(args.cpu_threads)
        os.environ["OMP_NUM_THREADS"] = value
        os.environ["MKL_NUM_THREADS"] = value
        os.environ["OPENBLAS_NUM_THREADS"] = value
        os.environ["NUMEXPR_NUM_THREADS"] = value

    if args.precache and args.cache_mode != "none":
        cache_script = "scripts/cache_tokens.py" if args.cache_mode == "tokens" else "scripts/cache_ska_artifacts.py"
        common = [sys.executable, cache_script]
        if args.artifact_cache_root:
            common += ["--artifact-cache-root", args.artifact_cache_root]
        if args.overwrite_cache:
            common.append("--overwrite-cache")

        cache_cmds = []
        lm_cmd = common + [
            "--task",
            "lm",
            "--lm-dataset",
            args.lm_dataset,
            "--lm-subset-path",
            args.lm_subset_path,
            "--lm-seq-len",
            str(args.lm_seq_len),
            "--lm-seq-stride",
            str(args.lm_seq_stride),
        ]
        if args.cache_mode == "full":
            lm_cmd.extend(["--lm-window", "64", "--lm-stride", "32", "--lm-minhash-k", "128", "--lm-router-topk", "4"])
        cache_cmds.append(lm_cmd)

        seq_cmd = common + [
            "--task",
            "seq2seq",
            "--seq-dataset",
            args.seq_dataset,
        ]
        if args.cache_mode == "full":
            seq_cmd.extend(["--seq-window", "64", "--seq-stride", "32", "--seq-minhash-k", "128", "--seq-router-topk", "4"])
        cache_cmds.append(seq_cmd)

        text_cmd = common + [
            "--task",
            "textdiff",
            "--textdiff-dataset",
            args.textdiff_dataset,
            "--textdiff-seq-len",
            str(args.textdiff_seq_len),
            "--textdiff-stride",
            str(args.textdiff_stride),
        ]
        if args.cache_mode == "full":
            text_cmd.extend(
                ["--textdiff-window", "64", "--textdiff-bank-stride", "32", "--textdiff-minhash-k", "128", "--textdiff-router-topk", "4"]
            )
        cache_cmds.append(text_cmd)
        for cmd in cache_cmds:
            _run(
                cmd,
                args.dry_run,
                args.min_free_gb,
                args.wait_gpu_interval,
                args.wait_gpu_timeout,
                args.require_idle_gpu,
                args.post_run_grace,
                args.post_run_wait,
            )

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
                    "--num-workers",
                    str(args.lm_num_workers),
                ]
                if args.lm_subset_path:
                    cmd.extend(["--subset-path", args.lm_subset_path])
                if args.cache_mode != "none":
                    cmd.extend(["--cache-mode", args.cache_mode])
                if args.artifact_cache_root:
                    cmd.extend(["--artifact-cache-root", args.artifact_cache_root])
                if args.overwrite_cache:
                    cmd.append("--overwrite-cache")
                if args.skip_oom:
                    cmd.append("--skip-oom")
                if args.profile:
                    cmd.append("--profile")
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
            rc = _run(
                cmd,
                args.dry_run,
                args.min_free_gb,
                args.wait_gpu_interval,
                args.wait_gpu_timeout,
                args.require_idle_gpu,
                args.post_run_grace,
                args.post_run_wait,
            )
            if rc != 0:
                    _append_status_row(
                        csv_path,
                        {
                            "script": "train_toy_lm_banked",
                            "task": "lm",
                            "dataset": args.lm_dataset,
                            "dataset_id": args.lm_dataset,
                            "mode": "sdpa" if variant == "dot_explicit" else "ska/python",
                            "attn_impl": "dot_explicit" if variant == "dot_explicit" else "ska/python",
                            "precision": args.lm_precision,
                            "seed": seed,
                            "rep": rep,
                            "run_uid": f"exitcode-{int(time.time())}-{seed}-{rep}",
                            "status": "exitcode",
                            "exit_code": rc,
                            "skip_reason": f"exitcode={rc}",
                        },
                    )
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
                cmd.extend(["--num-workers", str(args.seq_num_workers)])
                if args.cache_mode != "none":
                    cmd.extend(["--cache-mode", args.cache_mode])
                if args.artifact_cache_root:
                    cmd.extend(["--artifact-cache-root", args.artifact_cache_root])
                if args.overwrite_cache:
                    cmd.append("--overwrite-cache")
                if args.skip_oom:
                    cmd.append("--skip-oom")
                if args.profile:
                    cmd.append("--profile")
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
                rc = _run(
                    cmd,
                    args.dry_run,
                    args.min_free_gb,
                    args.wait_gpu_interval,
                    args.wait_gpu_timeout,
                    args.require_idle_gpu,
                    args.post_run_grace,
                    args.post_run_wait,
                )
                if rc != 0:
                    _append_status_row(
                        csv_path,
                        {
                            "script": "train_seq2seq_text_banked",
                            "task": "seq2seq",
                            "dataset": args.seq_dataset,
                            "dataset_id": args.seq_dataset,
                            "mode": "sdpa" if variant == "dot_explicit" else "ska/python",
                            "attn_impl": "dot_explicit" if variant == "dot_explicit" else "ska/python",
                            "precision": args.seq_precision,
                            "seed": seed,
                            "rep": rep,
                            "run_uid": f"exitcode-{int(time.time())}-{seed}-{rep}",
                            "status": "exitcode",
                            "exit_code": rc,
                            "skip_reason": f"exitcode={rc}",
                        },
                    )
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
                cmd.extend(["--num-workers", str(args.textdiff_num_workers)])
                if args.cache_mode != "none":
                    cmd.extend(["--cache-mode", args.cache_mode])
                if args.artifact_cache_root:
                    cmd.extend(["--artifact-cache-root", args.artifact_cache_root])
                if args.overwrite_cache:
                    cmd.append("--overwrite-cache")
                if args.skip_oom:
                    cmd.append("--skip-oom")
                if args.profile:
                    cmd.append("--profile")
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
                rc = _run(
                    cmd,
                    args.dry_run,
                    args.min_free_gb,
                    args.wait_gpu_interval,
                    args.wait_gpu_timeout,
                    args.require_idle_gpu,
                    args.post_run_grace,
                    args.post_run_wait,
                )
                if rc != 0:
                    _append_status_row(
                        csv_path,
                        {
                            "script": "train_toy_diffusion_banked",
                            "task": "textdiff",
                            "dataset": args.textdiff_dataset,
                            "dataset_id": args.textdiff_dataset,
                            "mode": "sdpa" if variant == "dot_explicit" else "ska/python",
                            "attn_impl": "dot_explicit" if variant == "dot_explicit" else "ska/python",
                            "precision": args.textdiff_precision,
                            "seed": seed,
                            "rep": rep,
                            "run_uid": f"exitcode-{int(time.time())}-{seed}-{rep}",
                            "status": "exitcode",
                            "exit_code": rc,
                            "skip_reason": f"exitcode={rc}",
                        },
                    )
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
                cmd.extend(["--num-workers", str(args.vit_num_workers)])
                if args.skip_oom:
                    cmd.append("--skip-oom")
                if args.profile:
                    cmd.append("--profile")
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
                rc = _run(
                    cmd,
                    args.dry_run,
                    args.min_free_gb,
                    args.wait_gpu_interval,
                    args.wait_gpu_timeout,
                    args.require_idle_gpu,
                    args.post_run_grace,
                    args.post_run_wait,
                )
                if rc != 0:
                    _append_status_row(
                        csv_path,
                        {
                            "script": "train_tiny_vit_banked",
                            "task": "vit",
                            "dataset": "cifar10",
                            "dataset_id": "cifar10",
                            "mode": "sdpa" if variant == "dot_explicit" else "ska/python",
                            "attn_impl": "dot_explicit" if variant == "dot_explicit" else "ska/python",
                            "precision": args.vit_precision,
                            "seed": seed,
                            "rep": rep,
                            "run_uid": f"exitcode-{int(time.time())}-{seed}-{rep}",
                            "status": "exitcode",
                            "exit_code": rc,
                            "skip_reason": f"exitcode={rc}",
                        },
                    )
                    print(f"[stageA] command failed: {' '.join(cmd)}")


if __name__ == "__main__":
    main()
