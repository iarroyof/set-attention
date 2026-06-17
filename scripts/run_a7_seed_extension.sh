#!/usr/bin/env bash
# A7.5 targeted seed extension for convergence-critical empty_only points.
#
# Adds seeds 3 and 4 for matched token baselines and for set families at
# w,s in {(1,1), (2,1), (3,1)}. This strengthens the empirical convergence
# claim without rerunning the full compressed-topology grid.
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a7_seed_extension}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a7_seed_extension}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a7_seed_extension}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"
SEQ_LEN=512
BATCH=16
EPOCHS=10
D_MODEL=384
D_FF=1536
SEEDS=(3 4)
SET_SPECS=("1 1" "2 1" "3 1")
BASELINE_FAMILIES=(baseline_dense_exact baseline_sparse_local_band baseline_linear_landmark)
SET_FAMILIES=(set_dense_exact set_sparse_local_band set_linear_landmark)

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" audit out/paper_integrated_evidence/checks

config_for_family () {
  case "$1" in
    baseline_dense_exact) echo "configs/paper_lr_norm/baseline_dense_exact.yaml" ;;
    baseline_sparse_local_band) echo "configs/paper_lr_norm/baseline_sparse_local_band.yaml" ;;
    baseline_linear_landmark) echo "configs/paper_lr_norm/baseline_linear_landmark.yaml" ;;
    set_dense_exact) echo "configs/paper_lr_norm/set_dense_exact.yaml" ;;
    set_sparse_local_band) echo "configs/paper_lr_norm/set_sparse_local_band.yaml" ;;
    set_linear_landmark) echo "configs/paper_lr_norm/set_linear_landmark.yaml" ;;
    *) echo "unknown family: $1" >&2; return 1 ;;
  esac
}

backend_for_family () {
  case "$1" in
    baseline_dense_exact|set_dense_exact) echo "exact" ;;
    baseline_sparse_local_band|set_sparse_local_band) echo "local_band" ;;
    baseline_linear_landmark|set_linear_landmark) echo "landmark" ;;
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


Path("audit/A7_seed_extension_prelaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "purpose": (
        "Targeted seeds 3 and 4 for A7 empirical convergence claims. "
        "This is not a full topology rerun; it covers token baselines and "
        "set empty_only families at w,s in {(1,1),(2,1),(3,1)}."
    ),
    "fixed": {
        "D": 384,
        "d_ff": 1536,
        "L": 512,
        "lr": "1e-4",
        "epochs": 10,
        "new_seeds": [3, 4],
        "set_causality_mode": "strict_past",
        "output_residual_mode": "empty_only",
        "feature_mode": "geometry_only",
        "pooling.mode": "mean",
        "landmark_coverage": 0.25,
    },
    "baseline_families": [
        "baseline_dense_exact",
        "baseline_sparse_local_band",
        "baseline_linear_landmark",
    ],
    "set_families": [
        "set_dense_exact",
        "set_sparse_local_band",
        "set_linear_landmark",
    ],
    "set_topologies": [
        {"w": 1, "s": 1, "M": 512},
        {"w": 2, "s": 1, "M": 511},
        {"w": 3, "s": 1, "M": 510},
    ],
    "expected_new_runs": 24,
}, indent=2) + "\n")
PY
}

run_baseline () {
  local GPU="$1"
  local FAMILY="$2"
  local SEED="$3"
  local CFG BACKEND GROUP NAME CSV_PATH LOG_PATH LR_TAG
  CFG="$(config_for_family "${FAMILY}")"
  BACKEND="$(backend_for_family "${FAMILY}")"
  LR_TAG="${LR//./p}"
  GROUP="${GROUP_PREFIX}_${FAMILY}_D${D_MODEL}_FF${D_FF}"
  NAME="a7_seedext_${FAMILY}_D${D_MODEL}_FF${D_FF}_L${SEQ_LEN}_lr${LR_TAG}_seed${SEED}"
  CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  LOG_PATH="${LOG_ROOT}/${NAME}.log"

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
    model.d_model="${D_MODEL}"
    model.dim_feedforward="${D_FF}"
    model.num_layers=6
    model.num_heads=8
    model.max_seq_len="${SEQ_LEN}"
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

run_set () {
  local GPU="$1"
  local FAMILY="$2"
  local WINDOW="$3"
  local STRIDE="$4"
  local SEED="$5"
  local CFG BACKEND M GROUP NAME CSV_PATH LOG_PATH ALLOW_TOKEN_TOKEN LR_TAG
  CFG="$(config_for_family "${FAMILY}")"
  BACKEND="$(backend_for_family "${FAMILY}")"
  M=$(( (SEQ_LEN - WINDOW) / STRIDE + 1 ))
  LR_TAG="${LR//./p}"
  GROUP="${GROUP_PREFIX}_${FAMILY}_D${D_MODEL}_FF${D_FF}"
  NAME="a7_seedext_${FAMILY}_D${D_MODEL}_FF${D_FF}_L${SEQ_LEN}_w${WINDOW}_s${STRIDE}_M${M}_lr${LR_TAG}_seed${SEED}"
  CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  LOG_PATH="${LOG_ROOT}/${NAME}.log"
  ALLOW_TOKEN_TOKEN=false

  if [[ "${WINDOW}" == "1" && "${STRIDE}" == "1" ]]; then
    ALLOW_TOKEN_TOKEN=true
  fi

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
    model.d_model="${D_MODEL}"
    model.dim_feedforward="${D_FF}"
    model.num_layers=6
    model.num_heads=8
    model.max_seq_len="${SEQ_LEN}"
    model.window_size="${WINDOW}"
    model.stride="${STRIDE}"
    model.set_causality_mode=strict_past
    model.output_residual_mode=empty_only
    model.allow_token_token="${ALLOW_TOKEN_TOKEN}"
    model.d_phi="${D_MODEL}"
    model.set_state_dim="${D_MODEL}"
    model.adapter_type=auto
    model.router_topk=16
    model.router_temperature=1.0
    model.router_multihead=true
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

  for FAMILY in "${BASELINE_FAMILIES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      if (( INDEX % 2 == MOD )); then
        run_baseline "${GPU}" "${FAMILY}" "${SEED}"
      fi
      INDEX=$((INDEX + 1))
    done
  done

  for FAMILY in "${SET_FAMILIES[@]}"; do
    for SPEC in "${SET_SPECS[@]}"; do
      read -r WINDOW STRIDE <<<"${SPEC}"
      for SEED in "${SEEDS[@]}"; do
        if (( INDEX % 2 == MOD )); then
          run_set "${GPU}" "${FAMILY}" "${WINDOW}" "${STRIDE}" "${SEED}"
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

echo "A7 seed-extension workers launched: GPU0 PID=${PID0}, GPU1 PID=${PID1}"
wait "${PID0}"
wait "${PID1}"
echo "=== A7 seed-extension sweep complete ==="
