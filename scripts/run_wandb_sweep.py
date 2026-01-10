import argparse
import csv
import os
import subprocess
import time
from pathlib import Path
from typing import List
import yaml


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
        print(f"[wandb-sweep] warn: nvidia-smi compute query failed ({exc}); skipping process gate.")
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
        print(f"[wandb-sweep] warn: nvidia-smi unavailable ({exc}); skipping GPU wait.")
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
                print(f"[wandb-sweep] warn: GPU busy ({_format_procs(procs)}) (timeout).")
            if free_gb is not None:
                print(f"[wandb-sweep] warn: GPU free {free_gb:.2f} GB < {min_free_gb:.2f} GB (timeout).")
            return False
        if require_idle and procs:
            print(f"[wandb-sweep] waiting for GPU idle; busy: {_format_procs(procs)}")
        if min_free_gb > 0 and free_gb is not None and free_gb < min_free_gb:
            print(f"[wandb-sweep] waiting for GPU free {free_gb:.2f} GB < {min_free_gb:.2f} GB")
        time.sleep(interval_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="Path to sweep yaml (transformer/diffusion/vit)")
    ap.add_argument("--min-free-gb", type=float, default=0.0, help="Wait for this much free GPU memory before running each job (0=disable).")
    ap.add_argument("--wait-gpu-interval", type=float, default=10.0, help="Seconds between GPU free-memory checks.")
    ap.add_argument("--wait-gpu-timeout", type=float, default=0.0, help="Timeout in seconds for GPU wait (0=wait forever).")
    ap.add_argument("--require-idle-gpu", action="store_true", default=True, help="Wait for GPU to have no active compute processes (default: enabled).")
    ap.add_argument("--no-require-idle-gpu", dest="require_idle_gpu", action="store_false", help="Disable GPU process-idle gating.")
    ap.add_argument("--post-run-grace", type=float, default=2.0, help="Seconds to wait after each job before checking GPU processes.")
    ap.add_argument("--post-run-wait", action="store_true", help="Wait for GPU idle after each job (in addition to warnings).")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.sweep, "r", encoding="utf-8"))
    program = cfg["program"]
    params = cfg["parameters"]

    attns = params.get("attn", {}).get("values", ["dot"])  # type: ignore
    seeds = params.get("seed", {}).get("values", ["1337"])  # type: ignore
    fixed = {k: v for k, v in params.items() if isinstance(v, dict) and "value" in v}

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

    csv_path = Path("out/sweeps/wandb.csv")
    for a in attns:
        for s in seeds:
            cmd = ["python", program, "--attn", a]
            if "epochs" in fixed:
                cmd += ["--epochs", str(fixed["epochs"]["value"])]
            cmd += ["--seed", str(s)] if program.endswith("transformer.py") else []
            print("→", " ".join(cmd))
            env = os.environ.copy()
            base_row = {"script": program, "task": "wandb_sweep", "seed": s, "attn": a}
            if not _wait_for_gpu(
                args.min_free_gb, args.wait_gpu_interval, args.wait_gpu_timeout, args.require_idle_gpu
            ):
                append_status(
                    csv_path,
                    {
                        **base_row,
                        "run_uid": f"skip-{int(time.time())}-{s}",
                        "status": "skipped",
                        "skip_reason": "gpu_busy",
                    },
                )
                continue
            proc = subprocess.Popen(cmd, env=env)
            returncode = proc.wait()
            if args.post_run_grace > 0:
                time.sleep(args.post_run_grace)
            procs = _gpu_compute_procs()
            if procs:
                still = _format_procs(procs)
                if proc.pid in {p["pid"] for p in procs}:
                    print(f"[wandb-sweep] warn: job pid {proc.pid} still on GPU: {still}")
                else:
                    print(f"[wandb-sweep] warn: GPU still busy after run: {still}")
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
                        **base_row,
                        "run_uid": f"exitcode-{int(time.time())}-{s}",
                        "status": "exitcode",
                        "exit_code": returncode,
                        "skip_reason": f"exitcode={returncode}",
                    },
                )


if __name__ == "__main__":
    main()
