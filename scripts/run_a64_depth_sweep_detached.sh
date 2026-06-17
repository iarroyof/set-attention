#!/usr/bin/env bash
# Host-side detached launcher for A6.4. Starts resume-capable workers inside
# the already-running set-attention container via docker exec -d to avoid
# SSH session / Docker exec-instance interruption failures.
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a64_depth_sweep}"
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


Path("audit/A6_4_depth_sweep_detached_launch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "docker_ps": run(["docker", "compose", "ps", "-a"]),
    "note": (
        "A6.4 set-stack depth sweep detached launch. Tests num_layers in {8,10} "
        "at capacity pairs (set_state_dim=384,d_phi=384) and (set_state_dim=768,d_phi=512). "
        "Depth=6 rows are reused from A6.3 TSV. Workers skip complete epoch-10 CSVs "
        "and can be safely re-run for resume."
    ),
}, indent=2) + "\n")
PY

docker exec -d \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_DATASETS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e WANDB_MODE=offline \
  set-attention \
  bash -lc 'cd /workspace && bash scripts/run_a64_depth_sweep_worker_inside_container.sh 0 > logs/a64_depth_sweep/container_worker_gpu0.log 2>&1'

docker exec -d \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e WANDB_MODE=offline \
  set-attention \
  bash -lc 'cd /workspace && bash scripts/run_a64_depth_sweep_worker_inside_container.sh 1 > logs/a64_depth_sweep/container_worker_gpu1.log 2>&1'

echo "A6.4 detached container workers launched (GPU0 and GPU1)"
