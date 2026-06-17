#!/usr/bin/env bash
# A9 candidate-gather router comparison.
#
# Exact matched rerun of compressed A7 empty_only operating points, changing
# only model.router.score_mode from the historical dense masked router to the
# candidate_gather router.
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a9_candidate_gather}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a9_candidate_gather}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a9_candidate_gather}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"
SEQ_LEN=512
BATCH=16
EPOCHS=10

FAMILIES=(set_dense_exact set_sparse_local_band set_linear_landmark)
SPECS=("4 2" "8 4")
SEEDS=(0 1 2)

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" audit out/paper_integrated_evidence/checks

config_for_family () {
  case "$1" in
    set_dense_exact) echo "configs/paper_lr_norm/set_dense_exact.yaml" ;;
    set_sparse_local_band) echo "configs/paper_lr_norm/set_sparse_local_band.yaml" ;;
    set_linear_landmark) echo "configs/paper_lr_norm/set_linear_landmark.yaml" ;;
    *) echo "unknown family: $1" >&2; return 1 ;;
  esac
}

backend_for_family () {
  case "$1" in
    set_dense_exact) echo "exact" ;;
    set_sparse_local_band) echo "local_band" ;;
    set_linear_landmark) echo "landmark" ;;
    *) echo "unknown family: $1" >&2; return 1 ;;
  esac
}

completed_csv () {
  local CSV_PATH="$1"
  python3 - "$CSV_PATH" "$EPOCHS" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
with path.open(newline="") as fh:
    rows = list(csv.DictReader(fh))
if len(rows) >= expected and rows[-1].get("epoch") == str(expected):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

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


Path("audit/A9_candidate_gather_prelaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "purpose": (
        "Compare exact candidate-gather learned routing against already-run A7 "
        "dense masked-router empty_only references at the same causal LM "
        "operating points."
    ),
    "only_intended_change_vs_reference": "model.router.score_mode=candidate_gather",
    "reference_table": "out/paper_integrated_evidence/tables/a7_backend_family_empty_only_augmented_summary.tsv",
    "fixed": {
        "D": 384,
        "d_ff": 1536,
        "L": 512,
        "lr": "1e-4",
        "epochs": 10,
        "seeds": [0, 1, 2],
        "set_causality_mode": "strict_past",
        "output_residual_mode": "empty_only",
        "feature_mode": "geometry_only",
        "pooling.mode": "mean",
        "d_phi": 384,
        "set_state_dim": 384,
        "landmark_coverage": 0.25,
    },
    "topologies": [
        {"w": 4, "s": 2, "M": 255, "compression": 512 / 255},
        {"w": 8, "s": 4, "M": 127, "compression": 512 / 127},
    ],
    "families": ["set_dense_exact", "set_sparse_local_band", "set_linear_landmark"],
    "expected_runs": 18,
}, indent=2) + "\n")
PY
}

run_one () {
  local GPU="$1"
  local FAMILY="$2"
  local WINDOW="$3"
  local STRIDE="$4"
  local SEED="$5"
  local CFG
  local BACKEND
  CFG="$(config_for_family "${FAMILY}")"
  BACKEND="$(backend_for_family "${FAMILY}")"

  local M=$(( (SEQ_LEN - WINDOW) / STRIDE + 1 ))
  local LR_TAG="${LR//./p}"
  local GROUP="${GROUP_PREFIX}_${FAMILY}_D384_FF1536"
  local NAME="a9_cg_${FAMILY}_D384_FF1536_L${SEQ_LEN}_w${WINDOW}_s${STRIDE}_M${M}_lr${LR_TAG}_seed${SEED}"
  local CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  local LOG_PATH="${LOG_ROOT}/${NAME}.log"

  mkdir -p "${OUT_ROOT}/${GROUP}"
  if completed_csv "${CSV_PATH}"; then
    echo "=== Skipping complete ${NAME} ==="
    return 0
  fi

  echo "=== Running ${NAME} on GPU ${GPU} ==="
  local OVERRIDES=(
    training.output_dir="${OUT_ROOT}/${GROUP}/${NAME}"
    logging.wandb.enable=true
    logging.wandb.project="${PROJECT}"
    logging.wandb.run_name="${NAME}"
    data.dataset=wikitext2
    data.batch_size="${BATCH}"
    data.seq_len="${SEQ_LEN}"
    training.seed="${SEED}"
    training.epochs="${EPOCHS}"
    training.lr="${LR}"
    training.warmup_steps=1000
    model.d_model=384
    model.dim_feedforward=1536
    model.num_layers=6
    model.num_heads=8
    model.max_seq_len="${SEQ_LEN}"
    model.window_size="${WINDOW}"
    model.stride="${STRIDE}"
    model.set_causality_mode=strict_past
    model.output_residual_mode=empty_only
    model.allow_token_token=false
    model.d_phi=384
    model.set_state_dim=384
    model.adapter_type=auto
    model.router_topk=16
    model.router_temperature=1.0
    model.router_multihead=true
    model.router.min_temp=0.5
    model.router.score_mode=candidate_gather
    model.pooling.mode=mean
    model.pooling_multihead=false
    model.feature_mode=geometry_only
    model.geometry.enabled=false
    model.geometry.apply_as_bias=false
    model.geometry.apply_in_phi_attn=false
    model.token_mlp.enabled=false
  )

  if [[ "${BACKEND}" == "local_band" ]]; then
    OVERRIDES+=(model.backend_params.radius=4)
  fi
  if [[ "${BACKEND}" == "landmark" ]]; then
    OVERRIDES+=(model.backend_params.landmark_coverage=0.25)
  fi

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
    --override "${OVERRIDES[@]}" \
    | tee "${LOG_PATH}"
}

run_worker () {
  local GPU="$1"
  local MOD="$2"
  local INDEX=0
  local FAMILY SPEC WINDOW STRIDE SEED
  for FAMILY in "${FAMILIES[@]}"; do
    for SPEC in "${SPECS[@]}"; do
      read -r WINDOW STRIDE <<<"${SPEC}"
      for SEED in "${SEEDS[@]}"; do
        if (( INDEX % 2 == MOD )); then
          run_one "${GPU}" "${FAMILY}" "${WINDOW}" "${STRIDE}" "${SEED}"
        fi
        INDEX=$((INDEX + 1))
      done
    done
  done
}

record_prelaunch
run_worker 0 0 > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"
run_worker 1 1 > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

echo "A9 candidate-gather workers launched: GPU0 PID=${PID0}, GPU1 PID=${PID1}"
wait "${PID0}"
wait "${PID1}"
echo "=== A9 candidate-gather comparison complete ==="
