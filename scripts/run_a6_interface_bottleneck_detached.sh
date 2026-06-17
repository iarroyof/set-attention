#!/usr/bin/env bash
# Host-side detached launcher for A6.3. Starts resume-capable workers inside
# the already-running set-attention container.
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a6_interface_bottleneck}"
mkdir -p "${LOG_ROOT}" audit

python3 - <<'PY'
import json
import subprocess
from pathlib import Path


def run(cmd):
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip().splitlines(),
        "stderr": proc.stderr.strip().splitlines(),
    }


Path("audit/A6_3_interface_bottleneck_detached_relaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "docker_ps": run(["docker", "compose", "ps", "-a"]),
    "note": (
        "Detached relaunch after host-side docker compose exec loops failed with "
        "Docker exec-instance interruptions. Workers skip complete epoch-10 CSVs "
        "and overwrite incomplete/missing A6.3 rows."
    ),
}, indent=2) + "\n")
PY

docker exec -d \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_DATASETS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e WANDB_MODE=offline \
  set-attention \
  bash -lc 'cd /workspace && bash scripts/run_a6_interface_bottleneck_worker_inside_container.sh 0 > logs/a6_interface_bottleneck/container_worker_gpu0.log 2>&1'

docker exec -d \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e WANDB_MODE=offline \
  set-attention \
  bash -lc 'cd /workspace && bash scripts/run_a6_interface_bottleneck_worker_inside_container.sh 1 > logs/a6_interface_bottleneck/container_worker_gpu1.log 2>&1'

echo "A6.3 detached container workers launched"
