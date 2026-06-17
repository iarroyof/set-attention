#!/usr/bin/env bash
# A2.4 matched token-backend controls at LR-norm headline reference.
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a2_baseline_controls}"
OUT_ROOT="${OUT_ROOT:-out/paper_lr_norm/baseline_controls_A2_D384_FF1536}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a2_baseline_controls}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LRS=(1e-4 2e-4 3e-4 5e-4 7e-4)
SEEDS=(0 1 2)
SEQ_LEN=512
BATCH=16

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" audit

record_prelaunch () {
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

manifest = Path("out/paper_integrated_evidence/checks/a3_pooltau_sweep_manifest.json")
manifest_data = json.loads(manifest.read_text()) if manifest.exists() else {}
Path("audit/A2_4_baseline_controls_prelaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "a3_2_manifest_exists": manifest.exists(),
    "a3_2_manifest_status": manifest_data.get("status"),
    "locked_design_note": (
        "A2.4 v2.7 matched token-backend controls: baseline_sparse_local_band "
        "and baseline_linear_landmark at LR-norm headline reference D=384,d_ff=1536,"
        "L=512,w=16,s=8,M=63. Token landmark operates over L token positions, so "
        "landmark_coverage=0.25 resolves to K=128 for baseline_linear_landmark."
    ),
}, indent=2) + "\n")
PY
}

config_for_family () {
  case "$1" in
    baseline_sparse_local_band) echo "configs/paper_lr_norm/baseline_sparse_local_band.yaml" ;;
    baseline_linear_landmark) echo "configs/paper_lr_norm/baseline_linear_landmark.yaml" ;;
    *) echo "unknown family: $1" >&2; return 1 ;;
  esac
}

run_one () {
  local GPU="$1"
  local FAMILY="$2"
  local LR="$3"
  local SEED="$4"
  local CFG
  CFG="$(config_for_family "${FAMILY}")"

  local LR_TAG="${LR//./p}"
  local GROUP="${GROUP_PREFIX}_${FAMILY}_D384_FF1536"
  local NAME="a2_controls_${FAMILY}_D384_FF1536_L512_w16_s8_lr${LR_TAG}_seed${SEED}"
  local CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  local LOG_PATH="${LOG_ROOT}/${NAME}.log"

  mkdir -p "${OUT_ROOT}/${GROUP}"
  echo "=== Running ${NAME} on GPU ${GPU} ==="
  docker compose exec -T \
    -e CUDA_VISIBLE_DEVICES="${GPU}" \
    -e HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    -e WANDB_MODE="${WANDB_MODE}" \
    -e WANDB_PROJECT="${PROJECT}" \
    -e WANDB_NAME="${NAME}" \
    -e WANDB_RUN_GROUP="${GROUP}" \
    set-attention \
    python scripts/run_experiment.py \
    --config "${CFG}" \
    --wandb \
    --wandb-project "${PROJECT}" \
    --csv-path "${CSV_PATH}" \
    --override \
      training.output_dir="${OUT_ROOT}/${GROUP}/${NAME}" \
      logging.wandb.enable=true \
      logging.wandb.project="${PROJECT}" \
      logging.wandb.run_name="${NAME}" \
      data.dataset=wikitext2 \
      data.batch_size="${BATCH}" \
      data.seq_len="${SEQ_LEN}" \
      training.seed="${SEED}" \
      training.epochs=10 \
      training.lr="${LR}" \
      training.warmup_steps=1000 \
      model.d_model=384 \
      model.dim_feedforward=1536 \
      model.num_layers=6 \
      model.num_heads=8 \
      model.max_seq_len="${SEQ_LEN}" \
    | tee "${LOG_PATH}"
}

run_worker () {
  local GPU="$1"
  local FAMILY="$2"
  local SEED LR
  for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
      run_one "${GPU}" "${FAMILY}" "${LR}" "${SEED}"
    done
  done
}

record_prelaunch
run_worker 0 baseline_sparse_local_band > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"
run_worker 1 baseline_linear_landmark > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

wait "${PID0}"
wait "${PID1}"
echo "=== A2.4 baseline controls complete ==="
