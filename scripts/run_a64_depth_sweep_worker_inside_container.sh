#!/usr/bin/env bash
# Container-side A6.4 worker. Run inside the set-attention container.
# Tests set-stack depth (num_layers) in {8,10} at two capacity settings:
#   (set_state_dim=384, d_phi=384) and (set_state_dim=768, d_phi=512).
# Depth=6 rows are reused from the A6.3 TSV; only depths 8 and 10 are new here.
set -euo pipefail

cd /workspace

MOD="${1:?usage: $0 <worker_mod_0_or_1>}"
LOG_ROOT="${LOG_ROOT:-logs/a64_depth_sweep}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a64_depth_sweep}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a64_depth_sweep}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"
SEQ_LEN=512
WINDOW=16
STRIDE=8
BATCH=16
EPOCHS=10
D_MODEL=384
D_FF=1536
# Only new depths — depth=6 is reused from the A6.3 TSV
DEPTHS=(8 10)
# (set_state_dim:d_phi) — the two capacity settings from A6.4 design
CAPACITIES=("384:384" "768:512")
SEEDS=(0 1 2)
FAMILIES=(set_dense_exact set_sparse_local_band set_linear_landmark)

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

config_for_family () {
  case "$1" in
    set_dense_exact)      echo "configs/paper_lr_norm/set_dense_exact.yaml" ;;
    set_sparse_local_band) echo "configs/paper_lr_norm/set_sparse_local_band.yaml" ;;
    set_linear_landmark)  echo "configs/paper_lr_norm/set_linear_landmark.yaml" ;;
    *) echo "unknown family: $1" >&2; return 1 ;;
  esac
}

csv_complete () {
  local CSV_PATH="$1"
  python - "${CSV_PATH}" "${EPOCHS}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
with path.open(newline="") as fh:
    rows = list(csv.DictReader(fh))
if len(rows) == expected and rows[-1].get("epoch") == str(expected):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

run_one () {
  local FAMILY="$1"
  local SET_STATE_DIM="$2"
  local DPHI="$3"
  local NUM_LAYERS="$4"
  local SEED="$5"
  local CFG
  CFG="$(config_for_family "${FAMILY}")"

  local LR_TAG="${LR//./p}"
  local GROUP="${GROUP_PREFIX}_${FAMILY}_D${D_MODEL}_FF${D_FF}"
  local NAME="a64_depth_${FAMILY}_D${D_MODEL}_FF${D_FF}_setdim${SET_STATE_DIM}_dphi${DPHI}_nl${NUM_LAYERS}_w${WINDOW}_s${STRIDE}_lr${LR_TAG}_seed${SEED}"
  local CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  local LOG_PATH="${LOG_ROOT}/${NAME}.log"

  mkdir -p "${OUT_ROOT}/${GROUP}"
  if csv_complete "${CSV_PATH}"; then
    echo "=== Skipping complete ${NAME} ==="
    return 0
  fi

  echo "=== Running ${NAME} ==="
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
    model.num_layers="${NUM_LAYERS}"
    model.num_heads=8
    model.max_seq_len="${SEQ_LEN}"
    model.window_size="${WINDOW}"
    model.stride="${STRIDE}"
    model.set_causality_mode=strict_past
    model.d_phi="${DPHI}"
    model.set_state_dim="${SET_STATE_DIM}"
    model.adapter_type=auto
    model.router_topk=16
    model.router_temperature=1.0
    model.pooling.mode=soft_trimmed_boltzmann
    model.pooling.tau=0.1
    model.pooling.q=0.85
    model.router_multihead=true
    model.pooling_multihead=false
  )

  if [[ "${FAMILY}" == "set_linear_landmark" ]]; then
    OVERRIDES+=(model.backend_params.landmark_coverage=0.25)
  fi

  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${PROJECT}" \
  WANDB_NAME="${NAME}" \
  WANDB_RUN_GROUP="${GROUP}" \
  python scripts/run_experiment.py \
    --config "${CFG}" \
    --wandb \
    --wandb-project "${PROJECT}" \
    --csv-path "${CSV_PATH}" \
    --override "${OVERRIDES[@]}" \
    | tee "${LOG_PATH}"
}

INDEX=0
for FAMILY in "${FAMILIES[@]}"; do
  for CAP in "${CAPACITIES[@]}"; do
    SET_STATE_DIM="${CAP%%:*}"
    DPHI="${CAP##*:}"
    for NUM_LAYERS in "${DEPTHS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        if (( INDEX % 2 == MOD )); then
          run_one "${FAMILY}" "${SET_STATE_DIM}" "${DPHI}" "${NUM_LAYERS}" "${SEED}"
        fi
        INDEX=$((INDEX + 1))
      done
    done
  done
done

echo "=== A6.4 container worker MOD=${MOD} complete ==="
