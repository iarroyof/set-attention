import argparse
import csv
import itertools
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import yaml

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
            if free_gb is not None:
                print(
                    f"[sweep] warn: GPU free {free_gb:.2f} GB < {min_free_gb:.2f} GB (timeout)."
                )
            return False
        if require_idle and procs:
            print(f"[sweep] waiting for GPU idle; busy: {_format_procs(procs)}")
        if min_free_gb > 0 and free_gb is not None and free_gb < min_free_gb:
            print(f"[sweep] waiting for GPU free {free_gb:.2f} GB < {min_free_gb:.2f} GB")
        time.sleep(interval_s)


def _extract_seed(cmd: List[str]) -> str:
    if "--seed" not in cmd:
        return ""
    try:
        idx = cmd.index("--seed")
    except ValueError:
        return ""
    if idx + 1 >= len(cmd):
        return ""
    return str(cmd[idx + 1])


def run(
    cmd,
    csv_path: Path,
    min_free_gb: float,
    wait_interval: float,
    wait_timeout: float,
    require_idle: bool,
    post_run_grace: float,
    post_run_wait: bool,
    base_row: dict | None = None,
):
    print("→", " ".join(cmd))
    if not _wait_for_gpu(min_free_gb, wait_interval, wait_timeout, require_idle):
        if base_row is not None:
            row = dict(base_row)
            row.setdefault("script", cmd[1] if len(cmd) > 1 else "")
            row.setdefault("task", "sweep")
            row.setdefault("seed", _extract_seed(cmd))
            row.update(
                {
                    "run_uid": f"skip-{int(time.time())}",
                    "status": "skipped",
                    "skip_reason": "gpu_busy",
                }
            )
            append_status(csv_path, row)
        return 1
    with tempfile.TemporaryFile(mode="w+") as stderr_file:
        proc = subprocess.Popen(cmd, stderr=stderr_file)
        returncode = proc.wait()
        if returncode != 0:
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
    if returncode != 0:
        if base_row is None:
            append_status(
                csv_path,
                {
                    "script": cmd[1] if len(cmd) > 1 else "",
                    "task": "sweep",
                    "seed": _extract_seed(cmd),
                    "run_uid": f"exitcode-{int(time.time())}",
                    "status": "exitcode",
                    "exit_code": returncode,
                    "skip_reason": f"exitcode={returncode}",
                },
            )
        else:
            row = dict(base_row)
            row.setdefault("script", cmd[1] if len(cmd) > 1 else "")
            row.setdefault("task", "sweep")
            row.setdefault("seed", _extract_seed(cmd))
            row.update(
                {
                    "run_uid": f"exitcode-{int(time.time())}",
                    "status": "exitcode",
                    "exit_code": returncode,
                    "skip_reason": f"exitcode={returncode}",
                }
            )
            append_status(csv_path, row)
    return returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", help="Optional YAML sweep definition (same schema as run_wandb_sweep.py).")
    ap.add_argument("--csv", help="Optional CSV path for sweep results.")
    ap.add_argument("--which", choices=["transformer", "diffusion", "vit"], default="transformer")
    ap.add_argument("--attn", nargs="*", default=["dot", "cosine", "rbf", "intersect", "ska", "ska_true"])  # ska alias to rbf
    ap.add_argument("--seeds", nargs="*", default=["1337", "2024", "7"])  # strings for convenience
    ap.add_argument("--min-free-gb", type=float, default=0.0, help="Wait for this much free GPU memory before running each job (0=disable).")
    ap.add_argument("--wait-gpu-interval", type=float, default=10.0, help="Seconds between GPU free-memory checks.")
    ap.add_argument("--wait-gpu-timeout", type=float, default=0.0, help="Timeout in seconds for GPU wait (0=wait forever).")
    ap.add_argument("--require-idle-gpu", action="store_true", default=True, help="Wait for GPU to have no active compute processes (default: enabled).")
    ap.add_argument("--no-require-idle-gpu", dest="require_idle_gpu", action="store_false", help="Disable GPU process-idle gating.")
    ap.add_argument("--post-run-grace", type=float, default=2.0, help="Seconds to wait after each job before checking GPU processes.")
    ap.add_argument("--post-run-wait", action="store_true", help="Wait for GPU idle after each job (in addition to warnings).")
    args = ap.parse_args()

    if args.sweep:
        cfg = yaml.safe_load(open(args.sweep, "r", encoding="utf-8"))
        program = cfg["program"]
        params = cfg.get("parameters", {})

        ordered_keys = list(params.keys())
        fixed: dict[str, object] = {}
        sweep_keys: List[str] = []
        sweep_values: List[list] = []
        for key in ordered_keys:
            spec = params.get(key, {})
            if not isinstance(spec, dict):
                continue
            if "value" in spec:
                fixed[key] = spec["value"]
                continue
            if "values" in spec:
                values = spec["values"]
                if not isinstance(values, list):
                    values = [values]
                sweep_keys.append(key)
                sweep_values.append(values)

        def _param_to_cli(key: str, value: object) -> list[str]:
            if value is None:
                return []
            flag = key if key.startswith("-") else f"--{key}"
            if isinstance(value, bool):
                return [flag] if value else []
            if isinstance(value, (list, tuple)):
                return [flag, *[str(v) for v in value]]
            return [flag, str(value)]

        def _param_to_row(key: str, value: object) -> str:
            if isinstance(value, (list, tuple, dict)):
                return json.dumps(value, sort_keys=True)
            return str(value)

        csv_path = Path(args.csv) if args.csv else Path("out/sweeps") / f"{Path(program).stem}.csv"
        combos = list(itertools.product(*sweep_values)) if sweep_keys else [()]
        for combo in combos:
            sweep_params = {k: v for k, v in zip(sweep_keys, combo)}
            all_params = {**fixed, **sweep_params}
            cmd = ["python", program]
            for key in ordered_keys:
                if key not in all_params:
                    continue
                cmd.extend(_param_to_cli(key, all_params[key]))
            base_row = {"script": program, "task": "sweep"}
            for key, value in all_params.items():
                base_row[key] = _param_to_row(key, value)
            run(
                cmd,
                csv_path,
                args.min_free_gb,
                args.wait_gpu_interval,
                args.wait_gpu_timeout,
                args.require_idle_gpu,
                args.post_run_grace,
                args.post_run_wait,
                base_row,
            )
        return

    if args.which == "transformer":
        csv_path = Path("out/sweeps/transformer.csv")
        for a, s in itertools.product(args.attn, args.seeds):
            run(
                ["python", "scripts/train_toy_transformer.py", "--attn", a, "--seed", s],
                csv_path,
                args.min_free_gb,
                args.wait_gpu_interval,
                args.wait_gpu_timeout,
                args.require_idle_gpu,
                args.post_run_grace,
                args.post_run_wait,
                {"script": "scripts/train_toy_transformer.py", "task": "sweep", "attn": a, "seed": s},
            )
    elif args.which == "diffusion":
        csv_path = Path("out/sweeps/diffusion.csv")
        for a, s in itertools.product(args.attn, args.seeds):
            run(
                ["python", "scripts/train_toy_diffusion.py", "--attn", a, "--seed", s],
                csv_path,
                args.min_free_gb,
                args.wait_gpu_interval,
                args.wait_gpu_timeout,
                args.require_idle_gpu,
                args.post_run_grace,
                args.post_run_wait,
                {"script": "scripts/train_toy_diffusion.py", "task": "sweep", "attn": a, "seed": s},
            )
    else:
        csv_path = Path("out/sweeps/vit.csv")
        for a, s in itertools.product(args.attn, args.seeds):
            run(
                ["python", "scripts/train_tiny_vit_cifar.py", "--attn", a],
                csv_path,
                args.min_free_gb,
                args.wait_gpu_interval,
                args.wait_gpu_timeout,
                args.require_idle_gpu,
                args.post_run_grace,
                args.post_run_wait,
                {"script": "scripts/train_tiny_vit_cifar.py", "task": "sweep", "attn": a, "seed": s},
            )


if __name__ == "__main__":
    main()
