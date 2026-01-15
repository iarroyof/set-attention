#!/usr/bin/env python3
"""
Stage B scaling sweep launcher.
Runs length/budget/capacity grids for the benchmark scripts and respects skip/oom flags.
"""
import argparse
import csv
import os
import subprocess
import tempfile
import shlex
import sys
import time
from pathlib import Path
from typing import List, Optional

from set_attention.data.artifact_cache import resolve_hf_root

def _parse_seeds(raw: str | List[str] | None, default: int) -> List[int]:
    if not raw:
        return [default]
    if isinstance(raw, list):
        text = " ".join(raw)
    else:
        text = raw
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
        print(f"[sweep] warn: nvidia-smi compute query failed ({exc}); skipping process gate.")
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
        print(f"[sweep] warn: nvidia-smi unavailable ({exc}); skipping GPU wait.")
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
                print(f"[sweep] warn: GPU busy ({_format_procs(procs)}) (timeout).")
            print(
                f"[sweep] warn: GPU free {free_gb:.2f} GB < {min_free_gb:.2f} GB (timeout)."
            )
            return False
        if require_idle and procs:
            print(f"[sweep] waiting for GPU idle; busy: {_format_procs(procs)}")
        if min_free_gb > 0 and free_gb is not None and free_gb < min_free_gb:
            print(f"[sweep] waiting for GPU free {free_gb:.2f} GB < {min_free_gb:.2f} GB")
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
    print("[sweep]", " ".join(cmd))
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
                print("[sweep] stderr (tail):")
                for line in tail:
                    print(f"[sweep] | {line}")
    if post_run_grace > 0:
        time.sleep(post_run_grace)
    procs = _gpu_compute_procs()
    if procs:
        still = _format_procs(procs)
        if proc.pid in {p["pid"] for p in procs}:
            print(f"[sweep] warn: job pid {proc.pid} still on GPU: {still}")
        else:
            print(f"[sweep] warn: GPU still busy after run: {still}")
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
    ap = argparse.ArgumentParser(description="Run Stage B scaling sweeps with skip-aware configs.")
    ap.add_argument("--output-dir", type=str, default="out/benchmarks_scale")
    ap.add_argument("--gpu-vram", type=float, default=24.0, help="GPU VRAM in GB for skip estimation.")
    ap.add_argument("--seeds", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache-mode", choices=["none", "tokens", "full"], default="none")
    ap.add_argument("--artifact-cache-root", type=str, default="")
    ap.add_argument("--overwrite-cache", action="store_true")
    ap.add_argument("--precache", action="store_true", help="Precompute caches before running Stage B.")
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

    # LM length sweep defaults
    ap.add_argument("--lm-lengths", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--lm-batch", type=int, default=8)
    ap.add_argument("--lm-dataset", type=str, default="wikitext103")
    ap.add_argument("--lm-subset-path", type=str, default="subsets/wikitext103_train_10pct.json")
    ap.add_argument("--lm-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--lm-seq-stride", type=int, default=0, help="Seq stride override (0 uses seq length).")
    ap.add_argument("--lm-window", type=int, default=64)
    ap.add_argument("--lm-stride", type=int, default=32)
    ap.add_argument("--lm-minhash-k", type=int, default=128)
    ap.add_argument("--lm-router-topk", type=int, default=4)
    ap.add_argument("--lm-num-workers", type=int, default=0)
    ap.add_argument("--lm-limit", type=int, default=None, help="Alias for --limit in LM runs.")

    # Seq2Seq lengths (optional; default empty)
    ap.add_argument("--seq-lengths", type=int, nargs="+", default=[])
    ap.add_argument("--seq-batch", type=int, default=16)
    ap.add_argument("--seq-dataset", type=str, default="wmt16_en_ro")
    ap.add_argument("--seq-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--seq-max-len", type=int, default=256)
    ap.add_argument("--seq-tokenizer-type", type=str, default="whitespace")
    ap.add_argument("--seq-window", type=int, default=64)
    ap.add_argument("--seq-stride", type=int, default=32)
    ap.add_argument("--seq-minhash-k", type=int, default=128)
    ap.add_argument("--seq-router-topk", type=int, default=4)
    ap.add_argument("--seq-num-workers", type=int, default=0)
    ap.add_argument("--seq-limit", type=int, default=None, help="Alias for --limit in Seq2Seq runs.")

    # Diffusion/text lengths (optional; default empty)
    ap.add_argument("--textdiff-lengths", type=int, nargs="+", default=[])
    ap.add_argument("--textdiff-batch", type=int, default=64)
    ap.add_argument("--textdiff-dataset", type=str, default="wikitext2")
    ap.add_argument("--textdiff-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--textdiff-seq-len", type=int, default=256)
    ap.add_argument("--textdiff-stride", type=int, default=256)
    ap.add_argument("--textdiff-window", type=int, default=64)
    ap.add_argument("--textdiff-bank-stride", type=int, default=32)
    ap.add_argument("--textdiff-minhash-k", type=int, default=128)
    ap.add_argument("--textdiff-router-topk", type=int, default=4)
    ap.add_argument("--textdiff-num-workers", type=int, default=0)

    # ViT (no length sweep; uses patch-based)
    ap.add_argument("--vit-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--vit-batch", type=int, default=128)
    ap.add_argument("--vit-window", type=int, default=8)
    ap.add_argument("--vit-stride", type=int, default=4)
    ap.add_argument("--vit-minhash-k", type=int, default=64)
    ap.add_argument("--vit-router-topk", type=int, default=0)
    ap.add_argument("--vit-num-workers", type=int, default=0)
    ap.add_argument("--vit-limit", type=int, default=None, help="Alias for --limit in ViT runs.")
    ap.add_argument(
        "--common-args",
        action="append",
        default=[],
        help="Extra CLI args appended to all task runs (repeatable, quoted string).",
    )
    ap.add_argument(
        "--lm-args",
        action="append",
        default=[],
        help="Extra CLI args appended to LM runs (repeatable, quoted string).",
    )
    ap.add_argument(
        "--seq-args",
        action="append",
        default=[],
        help="Extra CLI args appended to Seq2Seq runs (repeatable, quoted string).",
    )
    ap.add_argument(
        "--textdiff-args",
        action="append",
        default=[],
        help="Extra CLI args appended to TextDiff runs (repeatable, quoted string).",
    )
    ap.add_argument(
        "--vit-args",
        action="append",
        default=[],
        help="Extra CLI args appended to ViT runs (repeatable, quoted string).",
    )
    ap.add_argument(
        "--lm-cache-args",
        action="append",
        default=[],
        help="Extra CLI args appended to LM cache commands (repeatable, quoted string).",
    )
    ap.add_argument(
        "--seq-cache-args",
        action="append",
        default=[],
        help="Extra CLI args appended to Seq2Seq cache commands (repeatable, quoted string).",
    )
    ap.add_argument(
        "--textdiff-cache-args",
        action="append",
        default=[],
        help="Extra CLI args appended to TextDiff cache commands (repeatable, quoted string).",
    )

    args = ap.parse_args()

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
    vit_args = _parse_extra(args.vit_args)
    lm_cache_args = _parse_extra(args.lm_cache_args)
    seq_cache_args = _parse_extra(args.seq_cache_args)
    textdiff_cache_args = _parse_extra(args.textdiff_cache_args)
    seeds = _parse_seeds(args.seeds, default=2024)
    reps = max(1, int(args.reps))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_mode == "full" and not args.precache:
        hf_root = resolve_hf_root(args.artifact_cache_root or None)
        missing = []
        lm_root = hf_root / "artifacts" / "lm" / args.lm_dataset
        lm_required = ["bank_train.pt", "bank_val.pt", "routing_train.pt", "routing_val.pt"]

        def _has_artifacts(root: Path, required: list[str]) -> bool:
            if not root.exists():
                return False
            for fp_dir in root.iterdir():
                if not fp_dir.is_dir():
                    continue
                if all((fp_dir / name).exists() for name in required):
                    return True
            return False

        if not _has_artifacts(lm_root, lm_required):
            missing.append(f"lm:{args.lm_dataset}")
        if args.seq_lengths:
            seq_root = hf_root / "artifacts" / "seq2seq" / args.seq_dataset
            seq_required = [
                "bank_src_train.pt",
                "bank_tgt_train.pt",
                "bank_src_val.pt",
                "bank_tgt_val.pt",
                "routing_src_train.pt",
                "routing_tgt_train.pt",
                "routing_src_val.pt",
                "routing_tgt_val.pt",
            ]
            if not _has_artifacts(seq_root, seq_required):
                missing.append(f"seq2seq:{args.seq_dataset}")
        if args.textdiff_lengths:
            text_root = hf_root / "artifacts" / "textdiff" / args.textdiff_dataset
            text_required = ["bank_train.pt", "bank_val.pt", "routing_train.pt", "routing_val.pt"]
            if not _has_artifacts(text_root, text_required):
                missing.append(f"textdiff:{args.textdiff_dataset}")
        if missing:
            raise RuntimeError(
                "Full cache requested but no precomputed banks found for: "
                + ", ".join(missing)
                + ". Run with --precache first."
            )

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

        for L in args.lm_lengths:
            lm_stride = args.lm_seq_stride if args.lm_seq_stride > 0 else L
            lm_cmd = common + [
                "--task",
                "lm",
                "--lm-dataset",
                args.lm_dataset,
                "--lm-subset-path",
                args.lm_subset_path,
                "--lm-seq-len",
                str(L),
                "--lm-seq-stride",
                str(lm_stride),
                "--lm-precision",
                args.lm_precision,
            ]
            if args.lm_limit is not None:
                lm_cmd.extend(["--lm-limit", str(args.lm_limit)])
            if args.cache_mode == "full":
                lm_cmd.extend(
                    [
                        "--lm-window",
                        str(args.lm_window),
                        "--lm-stride",
                        str(args.lm_stride),
                        "--lm-minhash-k",
                        str(args.lm_minhash_k),
                        "--lm-router-topk",
                        str(args.lm_router_topk),
                        "--ska-backend",
                        "python",
                    ]
                )
            if lm_cache_args:
                lm_cmd.extend(lm_cache_args)
            _run(
                lm_cmd,
                args.dry_run,
                args.min_free_gb,
                args.wait_gpu_interval,
                args.wait_gpu_timeout,
                args.require_idle_gpu,
                args.post_run_grace,
                args.post_run_wait,
            )

        if args.seq_lengths:
            for L in args.seq_lengths:
                seq_cmd = common + [
                    "--task",
                    "seq2seq",
                    "--seq-dataset",
                    args.seq_dataset,
                    "--seq-max-len",
                    str(L),
                    "--seq-tokenizer-type",
                    args.seq_tokenizer_type,
                    "--seq-precision",
                    args.seq_precision,
                ]
                if args.seq_limit is not None:
                    seq_cmd.extend(["--seq-limit", str(args.seq_limit)])
                if args.cache_mode == "full":
                    seq_cmd.extend(
                        [
                            "--seq-window",
                            str(args.seq_window),
                            "--seq-stride",
                            str(args.seq_stride),
                            "--seq-minhash-k",
                            str(args.seq_minhash_k),
                            "--seq-router-topk",
                            str(args.seq_router_topk),
                            "--ska-backend",
                            "python",
                        ]
                    )
                if seq_cache_args:
                    seq_cmd.extend(seq_cache_args)
                _run(
                    seq_cmd,
                    args.dry_run,
                    args.min_free_gb,
                    args.wait_gpu_interval,
                    args.wait_gpu_timeout,
                    args.require_idle_gpu,
                    args.post_run_grace,
                    args.post_run_wait,
                )

        for L in args.textdiff_lengths:
            text_stride = args.textdiff_stride if args.textdiff_stride > 0 else L
            text_cmd = common + [
                "--task",
                "textdiff",
                "--textdiff-dataset",
                args.textdiff_dataset,
                "--textdiff-seq-len",
                str(L),
                "--textdiff-stride",
                str(text_stride),
                "--textdiff-precision",
                args.textdiff_precision,
            ]
            if args.cache_mode == "full":
                text_cmd.extend(
                    [
                        "--textdiff-window",
                        str(args.textdiff_window),
                        "--textdiff-bank-stride",
                        str(args.textdiff_bank_stride),
                        "--textdiff-minhash-k",
                        str(args.textdiff_minhash_k),
                        "--textdiff-router-topk",
                        str(args.textdiff_router_topk),
                        "--ska-backend",
                        "python",
                    ]
                )
            if textdiff_cache_args:
                text_cmd.extend(textdiff_cache_args)
            _run(
                text_cmd,
                args.dry_run,
                args.min_free_gb,
                args.wait_gpu_interval,
                args.wait_gpu_timeout,
                args.require_idle_gpu,
                args.post_run_grace,
                args.post_run_wait,
            )
        print("[sweep] precache complete; exiting.")
        return

    # LM sweeps
    for L in args.lm_lengths:
        for mode in ("dot_explicit", "ska_python"):
            for seed in seeds:
                for rep in range(1, reps + 1):
                    csv_name = f"lm_{args.lm_dataset}_L{L}_{mode}_s{seed}_r{rep}.csv"
                    lm_stride = args.lm_seq_stride if args.lm_seq_stride > 0 else L
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
                        str(lm_stride),
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
                        "--num-workers",
                        str(args.lm_num_workers),
                    ]
                    if args.lm_limit is not None:
                        cmd.extend(["--limit", str(args.lm_limit)])
                    if args.cache_mode != "none":
                        cmd.extend(["--cache-mode", args.cache_mode])
                    if args.artifact_cache_root:
                        cmd.extend(["--artifact-cache-root", args.artifact_cache_root])
                    if args.overwrite_cache:
                        cmd.append("--overwrite-cache")
                    if mode == "dot_explicit":
                        cmd.extend(["--sdpa-baseline", "--attn-baseline", "explicit", "--dot-naive"])
                    else:
                        cmd.extend(
                            [
                                "--ska-backend",
                                "python",
                                "--window",
                                str(args.lm_window),
                                "--stride",
                                str(args.lm_stride),
                                "--minhash-k",
                                str(args.lm_minhash_k),
                                "--router-topk",
                                str(args.lm_router_topk),
                            ]
                        )
                    if common_args:
                        cmd.extend(common_args)
                    if lm_args:
                        cmd.extend(lm_args)
                    if "--precompute-bank" in cmd:
                        raise RuntimeError("BUG: --precompute-bank must not appear in Stage B sweeps")
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
                            out_dir / csv_name,
                            {
                                "script": "train_toy_lm_banked",
                                "task": "lm",
                                "dataset": args.lm_dataset,
                                "dataset_id": args.lm_dataset,
                                "mode": "sdpa" if mode == "dot_explicit" else "ska/python",
                                "attn_impl": "dot_explicit" if mode == "dot_explicit" else "ska/python",
                                "precision": args.lm_precision,
                                "seed": seed,
                                "rep": rep,
                                "run_uid": f"exitcode-{int(time.time())}-{seed}-{rep}",
                                "status": "exitcode",
                                "exit_code": rc,
                                "skip_reason": f"exitcode={rc}",
                            },
                        )
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
                        "--max-len",
                        str(L),
                        "--tokenizer-type",
                        args.seq_tokenizer_type,
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
                        "--num-workers",
                        str(args.seq_num_workers),
                    ]
                    if args.seq_limit is not None:
                        cmd.extend(["--limit", str(args.seq_limit)])
                    if args.cache_mode != "none":
                        cmd.extend(["--cache-mode", args.cache_mode])
                    if args.artifact_cache_root:
                        cmd.extend(["--artifact-cache-root", args.artifact_cache_root])
                    if args.overwrite_cache:
                        cmd.append("--overwrite-cache")
                    if mode == "dot_explicit":
                        cmd.extend(["--sdpa-baseline", "--attn-baseline", "explicit", "--dot-naive"])
                    else:
                        cmd.extend(
                            [
                                "--ska-backend",
                                "python",
                                "--window",
                                str(args.seq_window),
                                "--stride",
                                str(args.seq_stride),
                                "--minhash-k",
                                str(args.seq_minhash_k),
                                "--router-topk",
                                str(args.seq_router_topk),
                            ]
                        )
                    if common_args:
                        cmd.extend(common_args)
                    if seq_args:
                        cmd.extend(seq_args)
                    if "--precompute-bank" in cmd:
                        raise RuntimeError("BUG: --precompute-bank must not appear in Stage B sweeps")
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
                            out_dir / csv_name,
                            {
                                "script": "train_seq2seq_text_banked",
                                "task": "seq2seq",
                                "dataset": args.seq_dataset,
                                "dataset_id": args.seq_dataset,
                                "mode": "sdpa" if mode == "dot_explicit" else "ska/python",
                                "attn_impl": "dot_explicit" if mode == "dot_explicit" else "ska/python",
                                "precision": args.seq_precision,
                                "seed": seed,
                                "rep": rep,
                                "run_uid": f"exitcode-{int(time.time())}-{seed}-{rep}",
                                "status": "exitcode",
                                "exit_code": rc,
                                "skip_reason": f"exitcode={rc}",
                            },
                        )
                        print(f"[sweep] command failed with code {rc}: {' '.join(cmd)}")

    # Text diffusion sweeps (optional)
    for L in args.textdiff_lengths:
        for mode in ("dot_explicit", "ska_python"):
            for seed in seeds:
                for rep in range(1, reps + 1):
                    csv_name = f"textdiff_{args.textdiff_dataset}_L{L}_{mode}_s{seed}_r{rep}.csv"
                    text_stride = args.textdiff_stride if args.textdiff_stride > 0 else L
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
                        str(text_stride),
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
                        "--num-workers",
                        str(args.textdiff_num_workers),
                    ]
                    if args.cache_mode != "none":
                        cmd.extend(["--cache-mode", args.cache_mode])
                    if args.artifact_cache_root:
                        cmd.extend(["--artifact-cache-root", args.artifact_cache_root])
                    if args.overwrite_cache:
                        cmd.append("--overwrite-cache")
                    if mode == "dot_explicit":
                        cmd.extend(["--sdpa-baseline", "--attn-baseline", "explicit", "--dot-naive"])
                    else:
                        cmd.extend(
                            [
                                "--ska-backend",
                                "python",
                                "--window",
                                str(args.textdiff_window),
                                "--stride",
                                str(args.textdiff_bank_stride),
                                "--minhash-k",
                                str(args.textdiff_minhash_k),
                                "--router-topk",
                                str(args.textdiff_router_topk),
                            ]
                        )
                    if common_args:
                        cmd.extend(common_args)
                    if textdiff_args:
                        cmd.extend(textdiff_args)
                    if "--precompute-bank" in cmd:
                        raise RuntimeError("BUG: --precompute-bank must not appear in Stage B sweeps")
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
                            out_dir / csv_name,
                            {
                                "script": "train_toy_diffusion_banked",
                                "task": "textdiff",
                                "dataset": args.textdiff_dataset,
                                "dataset_id": args.textdiff_dataset,
                                "mode": "sdpa" if mode == "dot_explicit" else "ska/python",
                                "attn_impl": "dot_explicit" if mode == "dot_explicit" else "ska/python",
                                "precision": args.textdiff_precision,
                                "seed": seed,
                                "rep": rep,
                                "run_uid": f"exitcode-{int(time.time())}-{seed}-{rep}",
                                "status": "exitcode",
                                "exit_code": rc,
                                "skip_reason": f"exitcode={rc}",
                            },
                        )
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
                    str(args.vit_batch),
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
                cmd.extend(["--num-workers", str(args.vit_num_workers)])
                if args.vit_limit is not None:
                    cmd.extend(["--limit", str(args.vit_limit)])
                if mode == "dot_explicit":
                    cmd.extend(["--sdpa-baseline", "--attn-baseline", "explicit", "--dot-naive"])
                else:
                    cmd.extend(
                        [
                            "--ska-backend",
                            "python",
                            "--window",
                            str(args.vit_window),
                            "--stride",
                            str(args.vit_stride),
                            "--minhash-k",
                            str(args.vit_minhash_k),
                            "--router-topk",
                            str(args.vit_router_topk),
                        ]
                    )
                if common_args:
                    cmd.extend(common_args)
                if vit_args:
                    cmd.extend(vit_args)
                if "--precompute-bank" in cmd:
                    raise RuntimeError("BUG: --precompute-bank must not appear in Stage B sweeps")
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
                        out_dir / csv_name,
                        {
                            "script": "train_tiny_vit_banked",
                            "task": "vit",
                            "dataset": "cifar10",
                            "dataset_id": "cifar10",
                            "mode": "sdpa" if mode == "dot_explicit" else "ska/python",
                            "attn_impl": "dot_explicit" if mode == "dot_explicit" else "ska/python",
                            "precision": args.vit_precision,
                            "seed": seed,
                            "rep": rep,
                            "run_uid": f"exitcode-{int(time.time())}-{seed}-{rep}",
                            "status": "exitcode",
                            "exit_code": rc,
                            "skip_reason": f"exitcode={rc}",
                        },
                    )
                    print(f"[sweep] command failed with code {rc}: {' '.join(cmd)}")


if __name__ == "__main__":
    main()
