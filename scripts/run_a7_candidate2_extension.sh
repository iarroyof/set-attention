#!/usr/bin/env bash
# A7 candidate-count-near-2 extension for empty_only SetDense calibration.
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a7_candidate2_extension}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a7_empty_only_calibration}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a7_empty_only_calibration}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"
SEQ_LEN=512
BATCH=16
EPOCHS=10

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" audit out/paper_integrated_evidence/checks

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

run_one () {
  local GPU="$1"
  local WINDOW="$2"
  local STRIDE="$3"
  local SEED="$4"
  local M=$(( (SEQ_LEN - WINDOW) / STRIDE + 1 ))
  local LR_TAG="${LR//./p}"
  local GROUP="${GROUP_PREFIX}_set_dense_D384_FF1536"
  local NAME="a7_empty_set_dense_D384_FF1536_L${SEQ_LEN}_w${WINDOW}_s${STRIDE}_M${M}_lr${LR_TAG}_seed${SEED}"
  local CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  local LOG_PATH="${LOG_ROOT}/${NAME}.log"
  local ALLOW_TOKEN_TOKEN=false

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
    model.d_model=384
    model.dim_feedforward=1536
    model.num_layers=6
    model.num_heads=8
    model.max_seq_len="${SEQ_LEN}"
    model.window_size="${WINDOW}"
    model.stride="${STRIDE}"
    model.set_causality_mode=strict_past
    model.output_residual_mode=empty_only
    model.allow_token_token="${ALLOW_TOKEN_TOKEN}"
    model.d_phi=384
    model.set_state_dim=384
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
    --config configs/paper_lr_norm/set_dense_exact.yaml \
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
  local SPEC WINDOW STRIDE SEEDS SEED
  local SPECS=(
    "2 1 0,1,2"
    "3 1 0,1,2"
    "2 2 0,1,2"
  )
  for SPEC in "${SPECS[@]}"; do
    read -r WINDOW STRIDE SEEDS <<<"${SPEC}"
    IFS=',' read -ra SEED_LIST <<<"${SEEDS}"
    for SEED in "${SEED_LIST[@]}"; do
      if (( INDEX % 2 == MOD )); then
        run_one "${GPU}" "${WINDOW}" "${STRIDE}" "${SEED}"
      fi
      INDEX=$((INDEX + 1))
    done
  done
}

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

Path("audit/A7_candidate2_extension_prelaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "purpose": "Complete and control the mean-candidate-count-near-2 empty_only topology question.",
    "fixed": {
        "D": 384,
        "d_ff": 1536,
        "L": 512,
        "lr": "1e-4",
        "epochs": 10,
        "set_causality_mode": "strict_past",
        "output_residual_mode": "empty_only",
        "feature_mode": "geometry_only",
        "pooling.mode": "mean",
    },
    "matrix": [
        {"w": 2, "s": 1, "seeds": [0, 1, 2], "note": "reuse seed 0 if already complete"},
        {"w": 3, "s": 1, "seeds": [0, 1, 2]},
        {"w": 2, "s": 2, "seeds": [0, 1, 2]},
    ],
}, indent=2) + "\n")
PY

run_worker 0 0 > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"
run_worker 1 1 > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

echo "A7 candidate-count extension workers launched: GPU0 PID=${PID0}, GPU1 PID=${PID1}"
wait "${PID0}"
wait "${PID1}"
echo "=== A7 candidate-count extension complete ==="
