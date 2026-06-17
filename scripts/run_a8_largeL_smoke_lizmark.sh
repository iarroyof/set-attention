#!/usr/bin/env bash
# A8.3 direct large-L smoke on lizmark.
#
# Seed-0, one-epoch, two-sample smoke at L=8192 for matched dense/linear token
# baselines and SetDense/SetLinear at the observed near-2 and near-4 topologies.
# This is a fit/config/provenance gate, not a paper-bound full training grid.
set -euo pipefail

cd "${REPO_ROOT:-$HOME/set-attention}"

LOG_ROOT="${LOG_ROOT:-logs/a8_largeL_smoke}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a8_largeL_smoke}"
PROJECT="${PROJECT:-set-attention}"
SOURCE_HEAD="${SOURCE_HEAD:-unknown}"
IMAGE="${IMAGE:-set-attention:latest}"
SEQ_LEN="${SEQ_LEN:-8192}"
BATCH="${BATCH:-1}"
# WikiText limit is line-count, not sample-count. At L=8192, limit=500 gives
# two training chunks in the copied offline cache; smaller values may yield none.
LIMIT="${LIMIT:-500}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-0}"
LR="${LR:-1e-4}"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" audit
STATUS_PATH="${OUT_ROOT}/a8_largeL_smoke_status.tsv"
printf "name\tgpu\tconfig\twindow\tstride\tcsv\texit_code\n" > "${STATUS_PATH}"

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
    "phase": "A8.3 large-L smoke",
    "host": run(["hostname"]),
    "date": run(["date", "-Is"]),
    "source_head": os.environ.get("SOURCE_HEAD", "unknown"),
    "repo_has_git": shutil.which("git") is not None and Path(".git").exists(),
    "git_head": run(["git", "rev-parse", "HEAD"]) if Path(".git").exists() else None,
    "git_status": run(["git", "status", "--short"]) if Path(".git").exists() else None,
    "docker_image": os.environ.get("IMAGE", "set-attention:latest"),
    "seq_len": int(os.environ.get("SEQ_LEN", "8192")),
    "batch_size": int(os.environ.get("BATCH", "1")),
    "data_limit": int(os.environ.get("LIMIT", "2")),
    "epochs": int(os.environ.get("EPOCHS", "1")),
    "seed": int(os.environ.get("SEED", "0")),
    "lr": os.environ.get("LR", "1e-4"),
    "rows": [
        {"family": "baseline_dense_exact", "topology": "NA"},
        {"family": "baseline_linear_landmark", "topology": "NA", "landmark_coverage": 0.25},
        {"family": "set_dense_exact", "topology": [4, 2], "M": ((8192 - 4) // 2) + 1},
        {"family": "set_dense_exact", "topology": [8, 4], "M": ((8192 - 8) // 4) + 1},
        {"family": "set_linear_landmark", "topology": [4, 2], "M": ((8192 - 4) // 2) + 1, "landmark_coverage": 0.25},
        {"family": "set_linear_landmark", "topology": [8, 4], "M": ((8192 - 8) // 4) + 1, "landmark_coverage": 0.25},
    ],
}
Path("audit/A8_3_largeL_smoke_prelaunch.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

run_case() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  local window="${4:-}"
  local stride="${5:-}"
  local group="a8_largeL_smoke_${name}"
  local csv="${OUT_ROOT}/${group}/${name}.csv"
  local log="${LOG_ROOT}/${name}.log"
  local args=(
    "training.output_dir=${OUT_ROOT}/${group}/${name}"
    "data.dataset=wikitext2"
    "data.batch_size=${BATCH}"
    "data.seq_len=${SEQ_LEN}"
    "data.limit=${LIMIT}"
    "training.seed=${SEED}"
    "training.epochs=${EPOCHS}"
    "training.lr=${LR}"
    "training.warmup_steps=1000"
    "model.d_model=384"
    "model.dim_feedforward=1536"
    "model.num_layers=6"
    "model.num_heads=8"
    "model.max_seq_len=${SEQ_LEN}"
  )

  if [[ -n "${window}" ]]; then
    args+=(
      "model.window_size=${window}"
      "model.stride=${stride}"
      "model.set_causality_mode=strict_past"
      "model.output_residual_mode=empty_only"
    )
  fi

  if [[ "${name}" == *linear* ]]; then
    args+=("model.backend_params.landmark_coverage=0.25")
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
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${name}" "${gpu}" "${config}" "${window:-NA}" "${stride:-NA}" "${csv}" "${rc}" \
    >> "${STATUS_PATH}"
  return 0
}

record_prelaunch

(
  run_case 0 "baseline_dense_exact_L8192_seed${SEED}" "baseline_dense_exact.yaml"
  run_case 0 "set_dense_exact_L8192_w4_s2_seed${SEED}" "set_dense_exact.yaml" 4 2
  run_case 0 "set_dense_exact_L8192_w8_s4_seed${SEED}" "set_dense_exact.yaml" 8 4
) > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"

(
  run_case 1 "baseline_linear_landmark_L8192_seed${SEED}" "baseline_linear_landmark.yaml"
  run_case 1 "set_linear_landmark_L8192_w4_s2_seed${SEED}" "set_linear_landmark.yaml" 4 2
  run_case 1 "set_linear_landmark_L8192_w8_s4_seed${SEED}" "set_linear_landmark.yaml" 8 4
) > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

echo "${PID0}" > "${LOG_ROOT}/worker_gpu0.pid"
echo "${PID1}" > "${LOG_ROOT}/worker_gpu1.pid"
echo "A8.3 large-L smoke workers launched: GPU0 PID=${PID0}, GPU1 PID=${PID1}"
echo "Status path: ${STATUS_PATH}"
wait "${PID0}"
wait "${PID1}"
echo "=== A8.3 large-L smoke complete ==="
