#!/usr/bin/env bash
# A8.3 selective L=8192 follow-up on lizmark.
#
# Runs the only smoke-supported paper-bound follow-up:
# baseline_linear_landmark vs set_linear_landmark at (w,s)=(8,4), seeds 0..4.
# Full cached WikiText-2 is used (no data.limit override), 10 epochs, batch 1.
set -euo pipefail

cd "${REPO_ROOT:-$HOME/set-attention}"

LOG_ROOT="${LOG_ROOT:-logs/a8_l8192_linear_followup}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a8_l8192_linear_followup}"
PROJECT="${PROJECT:-set-attention}"
SOURCE_HEAD="${SOURCE_HEAD:-unknown}"
IMAGE="${IMAGE:-set-attention:latest}"
SEQ_LEN="${SEQ_LEN:-8192}"
BATCH="${BATCH:-1}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
WINDOW=8
STRIDE=4

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" audit
STATUS_PATH="${OUT_ROOT}/a8_l8192_linear_followup_status.tsv"
printf "name\tgpu\tconfig\twindow\tstride\tseed\tcsv\texit_code\n" > "${STATUS_PATH}"

record_prelaunch() {
  /usr/bin/python3 - <<'PY'
import json
import os
import shutil
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

payload = {
    "phase": "A8.3 L8192 linear follow-up",
    "host": run(["hostname"]),
    "date": run(["date", "-Is"]),
    "source_head": os.environ.get("SOURCE_HEAD", "unknown"),
    "repo_has_git": shutil.which("git") is not None and Path(".git").exists(),
    "git_head": run(["git", "rev-parse", "HEAD"]) if Path(".git").exists() else None,
    "git_status": run(["git", "status", "--short"]) if Path(".git").exists() else None,
    "docker_image": os.environ.get("IMAGE", "set-attention:latest"),
    "seq_len": 8192,
    "batch_size": int(os.environ.get("BATCH", "1")),
    "epochs": int(os.environ.get("EPOCHS", "10")),
    "lr": os.environ.get("LR", "1e-4"),
    "seeds": [0, 1, 2, 3, 4],
    "rows": [
        {
            "family": "baseline_linear_landmark",
            "backend": "landmark",
            "landmark_coverage": 0.25,
        },
        {
            "family": "set_linear_landmark",
            "backend": "landmark",
            "window_size": 8,
            "stride": 4,
            "M": ((8192 - 8) // 4) + 1,
            "set_causality_mode": "strict_past",
            "output_residual_mode": "empty_only",
            "landmark_coverage": 0.25,
        },
    ],
    "note": "Full cached WikiText-2 run; no data.limit override.",
}
Path("audit/A8_3_l8192_linear_followup_prelaunch.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

run_case() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  local seed="$4"
  local window="${5:-}"
  local stride="${6:-}"
  local group="a8_l8192_linear_followup_${name}"
  local csv="${OUT_ROOT}/${group}/${name}.csv"
  local log="${LOG_ROOT}/${name}.log"
  local args=(
    "training.output_dir=${OUT_ROOT}/${group}/${name}"
    "data.dataset=wikitext2"
    "data.batch_size=${BATCH}"
    "data.seq_len=${SEQ_LEN}"
    "training.seed=${seed}"
    "training.epochs=${EPOCHS}"
    "training.lr=${LR}"
    "training.warmup_steps=1000"
    "model.d_model=384"
    "model.dim_feedforward=1536"
    "model.num_layers=6"
    "model.num_heads=8"
    "model.max_seq_len=${SEQ_LEN}"
    "model.backend_params.landmark_coverage=0.25"
  )

  if [[ -n "${window}" ]]; then
    args+=(
      "model.window_size=${window}"
      "model.stride=${stride}"
      "model.set_causality_mode=strict_past"
      "model.output_residual_mode=empty_only"
    )
  fi

  mkdir -p "${OUT_ROOT}/${group}"
  echo "=== ${name} on GPU ${gpu} ===" | tee "${log}"
  set +e
  docker run --rm \
    --gpus "device=${gpu}" \
    --ipc=host \
    -u "$(id -u):$(id -g)" \
    -e HOME=/workspace \
    -e XDG_CACHE_HOME=/workspace/.cache \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_DATASETS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e WANDB_MODE=offline \
    -e WANDB_PROJECT="${PROJECT}" \
    -e WANDB_NAME="${name}" \
    -e WANDB_RUN_GROUP="${group}" \
    -v "${PWD}:/workspace" \
    -w /workspace \
    "${IMAGE}" \
    /usr/bin/python scripts/run_experiment.py \
      --config "configs/paper_lr_norm/${config}" \
      --csv-path "${csv}" \
      --override "${args[@]}" \
    >> "${log}" 2>&1
  local rc="$?"
  set -e
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${name}" "${gpu}" "${config}" "${window:-NA}" "${stride:-NA}" "${seed}" "${csv}" "${rc}" \
    >> "${STATUS_PATH}"
  return 0
}

record_prelaunch

(
  for seed in 0 1 2 3 4; do
    run_case 0 "baseline_linear_landmark_L8192_seed${seed}" "baseline_linear_landmark.yaml" "${seed}"
  done
  for seed in 0 2 4; do
    run_case 0 "set_linear_landmark_L8192_w8_s4_seed${seed}" "set_linear_landmark.yaml" "${seed}" "${WINDOW}" "${STRIDE}"
  done
) > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"

(
  for seed in 1 3; do
    run_case 1 "set_linear_landmark_L8192_w8_s4_seed${seed}" "set_linear_landmark.yaml" "${seed}" "${WINDOW}" "${STRIDE}"
  done
) > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

echo "${PID0}" > "${LOG_ROOT}/worker_gpu0.pid"
echo "${PID1}" > "${LOG_ROOT}/worker_gpu1.pid"
echo "A8.3 L8192 linear follow-up workers launched: GPU0 PID=${PID0}, GPU1 PID=${PID1}"
echo "Status path: ${STATUS_PATH}"
wait "${PID0}"
wait "${PID1}"
echo "=== A8.3 L8192 linear follow-up complete ==="
