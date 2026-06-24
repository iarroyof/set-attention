#!/usr/bin/env bash
# SD-5 S1 dense set-dictionary ladder: anchor_span, anchor disabled, CE only.
set -euo pipefail

cd ~/set-attention

SMOKE="${SMOKE:-0}"
PROJECT="${PROJECT:-set-attention}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"

if [[ "${SMOKE}" == "1" ]]; then
  CONFIG="configs/set_dictionary/s1_anchor_span_dense_smoke.yaml"
  LOG_ROOT="${LOG_ROOT:-logs/sd5_s1_anchor_span_dense_smoke}"
  OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/sd5_s1_anchor_span_dense_smoke}"
  GROUP_PREFIX="${GROUP_PREFIX:-sd5_s1_anchor_span_dense_smoke}"
  SEQ_LEN=64
  BATCH=2
  EPOCHS=1
  D_MODEL=64
  D_FF=128
  LAYERS=1
  HEADS=4
  D_PHI=64
  SET_STATE_DIM=64
  HASH_BINS=32
  WARMUP=0
  SPECS=("4 2 0")
else
  CONFIG="configs/set_dictionary/s1_anchor_span_dense.yaml"
  LOG_ROOT="${LOG_ROOT:-logs/sd5_s1_anchor_span_dense}"
  OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/sd5_s1_anchor_span_dense}"
  GROUP_PREFIX="${GROUP_PREFIX:-sd5_s1_anchor_span_dense}"
  SEQ_LEN=512
  BATCH=16
  EPOCHS=10
  D_MODEL=384
  D_FF=1536
  LAYERS=6
  HEADS=8
  D_PHI=384
  SET_STATE_DIM=384
  HASH_BINS=128
  WARMUP=1000
  SPECS=("16 8 0,1,2" "4 2 0,1,2")
fi

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

record_prelaunch () {
  python3 - "$SMOKE" "$CONFIG" "$OUT_ROOT" "$LOG_ROOT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

smoke = sys.argv[1] == "1"
config = sys.argv[2]
out_root = sys.argv[3]
log_root = sys.argv[4]

def run(cmd):
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip().splitlines(),
        "stderr": proc.stderr.strip().splitlines(),
    }

payload = {
    "phase": "SD-5",
    "step": "S1",
    "smoke": smoke,
    "config": config,
    "out_root": out_root,
    "log_root": log_root,
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "contract": {
        "output_residual_mode": "anchor_span",
        "anchor.enabled": False,
        "objective": "CE only",
        "backend": "exact",
        "candidate_fiber": "endpoint_window",
        "token_mlp.enabled": False,
        "multivector_basis.enabled": False,
        "multivector_basis.r": 1,
    },
    "full_topologies": [
        {"w": 16, "s": 8, "M": 63},
        {"w": 4, "s": 2, "M": 255},
    ],
    "reference_policy": (
        "Reuse existing direct/SKA references and dense token baseline; do not rerun refs."
    ),
}
suffix = "_smoke" if smoke else ""
Path(f"audit/SD_5_s1_anchor_span_dense_prelaunch{suffix}.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

run_one () {
  local GPU="$1"
  local WINDOW="$2"
  local STRIDE="$3"
  local SEED="$4"
  local M=$(( (SEQ_LEN - WINDOW) / STRIDE + 1 ))
  local LR_TAG="${LR//./p}"
  local GROUP="${GROUP_PREFIX}_D${D_MODEL}_FF${D_FF}"
  local NAME="${GROUP_PREFIX}_D${D_MODEL}_FF${D_FF}_L${SEQ_LEN}_w${WINDOW}_s${STRIDE}_M${M}_lr${LR_TAG}_seed${SEED}"
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
    training.warmup_steps="${WARMUP}"
    model.d_model="${D_MODEL}"
    model.dim_feedforward="${D_FF}"
    model.num_layers="${LAYERS}"
    model.num_heads="${HEADS}"
    model.max_seq_len="${SEQ_LEN}"
    model.window_size="${WINDOW}"
    model.stride="${STRIDE}"
    model.set_causality_mode=strict_past
    model.output_residual_mode=anchor_span
    model.allow_token_token=false
    model.candidate_fiber=endpoint_window
    model.d_phi="${D_PHI}"
    model.set_state_dim="${SET_STATE_DIM}"
    model.adapter_type=auto
    model.router_topk=16
    model.router_temperature=1.0
    model.router_multihead=true
    model.router.min_temp=0.5
    model.router.score_mode=candidate_gather
    model.pooling.mode=soft_trimmed_boltzmann
    model.pooling.tau=0.1
    model.pooling.q=0.85
    model.pooling_multihead=false
    model.feature_mode=hashed_counts
    model.feature_params.num_bins="${HASH_BINS}"
    model.feature_params.hash_seed=13
    model.feature_params.normalize=true
    model.geometry.enabled=true
    model.geometry.apply_as_bias=true
    model.geometry.apply_in_phi_attn=true
    model.token_mlp.enabled=false
    model.anchor.enabled=false
    model.anchor.teacher.enabled=false
    model.set_diversity.lambda_div=0.0
    model.multivector_basis.enabled=false
    model.multivector_basis.r=1
  )
  if [[ "${SMOKE}" == "1" ]]; then
    OVERRIDES+=(data.limit=8 data.val_limit=4 model.router_topk=8)
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
    --config "${CONFIG}" \
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

record_prelaunch
if [[ "${SMOKE}" == "1" ]]; then
  run_worker 0 0
else
  run_worker 0 0 > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
  PID0="$!"
  run_worker 1 1 > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
  PID1="$!"
  echo "SD-5 S1 anchor_span dense workers launched: GPU0 PID=${PID0}, GPU1 PID=${PID1}"
  wait "${PID0}"
  wait "${PID1}"
fi
echo "=== SD-5 S1 anchor_span dense complete ==="
