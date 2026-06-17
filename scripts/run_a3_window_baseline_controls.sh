#!/usr/bin/env bash
# A3.1-control matched token-backend controls for the fixed-stride window sweep.
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a3_window_baseline_controls}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a3_window_baseline_controls}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a3_window_baseline_controls}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"
STRIDE=4
WINDOWS=(6 8 12 16 20 24)
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

manifest = Path("out/paper_integrated_evidence/checks/a3_window_sweep_manifest.json")
manifest_data = json.loads(manifest.read_text()) if manifest.exists() else {}
Path("audit/A3_1_baseline_controls_prelaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "a3_1_manifest_exists": manifest.exists(),
    "a3_1_manifest_status": manifest_data.get("status"),
    "locked_design_note": (
        "A3.1-control v2.7 token-backend controls over the same fixed-stride "
        "window labels as A3.1: s=4, w={6,8,12,16,20,24}. Token baselines do not "
        "construct set windows; w and M are comparison-axis metadata."
    ),
}, indent=2) + "\n")
PY
}

config_for_family () {
  case "$1" in
    baseline_sparse_local_band) echo "configs/paper_complements/baseline_sparse_local_band.yaml" ;;
    baseline_linear_landmark) echo "configs/paper_complements/baseline_linear_landmark.yaml" ;;
    *) echo "unknown family: $1" >&2; return 1 ;;
  esac
}

seeds_for_window () {
  case "$1" in
    6|16|24) echo "0 1 2" ;;
    8|12|20) echo "0" ;;
    *) echo "unknown window: $1" >&2; return 1 ;;
  esac
}

run_one () {
  local GPU="$1"
  local FAMILY="$2"
  local WINDOW="$3"
  local SEED="$4"
  local CFG
  CFG="$(config_for_family "${FAMILY}")"

  local LR_TAG="${LR//./p}"
  local GROUP="${GROUP_PREFIX}_${FAMILY}_D384_FF1536_s${STRIDE}"
  local NAME="a3_window_controls_${FAMILY}_D384_FF1536_w${WINDOW}_s${STRIDE}_lr${LR_TAG}_seed${SEED}"
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
  local WINDOW SEED
  for WINDOW in "${WINDOWS[@]}"; do
    for SEED in $(seeds_for_window "${WINDOW}"); do
      run_one "${GPU}" "${FAMILY}" "${WINDOW}" "${SEED}"
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
echo "=== A3.1 baseline controls complete ==="
