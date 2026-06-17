#!/usr/bin/env bash
# A4.2 long-context family slice -- L=2048, LR-norm headline reference (D=384, d_ff=1536, w=16, s=8).
# Families: baseline_token (dense), set_dense (exact), set_sparse (local_band), set_linear (landmark).
# Seeds: 0, 1, 2 for each family.  12 total runs.
# GPU0: baseline_token + set_dense (sequential, 6 runs).
# GPU1: set_sparse + set_linear (sequential, 6 runs).
# BATCH=4 confirmed safe at L=2048 fp32 (peak ~18.6 GiB for dense; less for sparse/linear).
# M at L=2048, w=16, s=8: floor((2048-16)/8)+1 = 255.
# Landmark count: max(round(0.25 * 255), 2) = 64.
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a42_slice}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a42_slice}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a42_slice}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"
SEQ_LEN=2048
WINDOW=16
STRIDE=8
SEEDS=(0 1 2)
BATCH=4   # same as A4.1: fp32 dense attn at L=2048 peaks ~22 GiB at B=16; B=4 fits safely

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

manifest_path = Path("out/paper_integrated_evidence/checks/a41_smoke_manifest.json")
handoff_path  = Path("audit/A4_1_smoke.md")
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

Path("audit/A4_2_slice_prelaunch.json").write_text(json.dumps({
    "branch":       run(["git", "branch", "--show-current"]),
    "head":         run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "a4_1_manifest_exists":   manifest_path.exists(),
    "a4_1_manifest_status":   manifest.get("status"),
    "a4_1_validated_runs":    manifest.get("validated_runs"),
    "a4_1_expected_runs":     manifest.get("expected_runs"),
    "a4_1_audit_status_line": handoff_path.read_text().splitlines()[2] if handoff_path.exists() else None,
    "locked_design_note": (
        "A4.2 long-context family slice: L=2048, D=384, d_ff=1536, w=16, s=8, seeds={0,1,2}. "
        "Families: baseline_token (dense), set_dense (exact), set_sparse (local_band), set_linear (landmark). "
        "12 total runs. BATCH=4 (confirmed safe in A4.1). M=255 for all SKA variants. "
        "Landmark count = max(round(0.25*255),2) = 64."
    ),
}, indent=2) + "\n")
PY
}

run_one () {
  local GPU="$1"
  local FAMILY_SLUG="$2"   # baseline_token | set_dense | set_sparse | set_linear
  local CFG="$3"
  local SEED="$4"
  local IS_SET="$5"         # "yes" or "no"

  local LR_TAG="${LR//./p}"
  local GROUP="${GROUP_PREFIX}_${FAMILY_SLUG}_D384_FF1536_L${SEQ_LEN}"
  local LOG_PATH="${LOG_ROOT}/${FAMILY_SLUG}_seed${SEED}.log"
  local CSV_PATH="${OUT_ROOT}/${GROUP}"

  mkdir -p "${OUT_ROOT}/${GROUP}"

  local NAME
  if [[ "${IS_SET}" == "yes" ]]; then
    NAME="a42_${FAMILY_SLUG}_D384_FF1536_L${SEQ_LEN}_w${WINDOW}_s${STRIDE}_lr${LR_TAG}_seed${SEED}"
  else
    NAME="a42_${FAMILY_SLUG}_D384_FF1536_L${SEQ_LEN}_lr${LR_TAG}_seed${SEED}"
  fi

  CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"

  echo "=== Running ${NAME} on GPU ${GPU} ==="

  local OVERRIDES=(
    training.output_dir="${OUT_ROOT}/${GROUP}/${NAME}"
    data.dataset=wikitext2
    data.batch_size="${BATCH}"
    data.seq_len="${SEQ_LEN}"
    training.seed="${SEED}"
    training.epochs=10
    training.lr="${LR}"
    training.warmup_steps=1000
    model.d_model=384
    model.dim_feedforward=1536
    model.num_layers=6
    model.num_heads=8
    model.max_seq_len="${SEQ_LEN}"
  )

  if [[ "${IS_SET}" == "yes" ]]; then
    OVERRIDES+=(
      model.window_size="${WINDOW}"
      model.stride="${STRIDE}"
      model.set_causality_mode=strict_past
      model.router_topk=16
      model.router_temperature=1.0
      model.pooling.mode=soft_trimmed_boltzmann
      model.pooling.tau=0.1
      model.pooling.q=0.85
      model.router_multihead=true
      model.pooling_multihead=false
    )
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

run_gpu0_worker () {
  local SEED
  # baseline_token (dense, no set overrides)
  for SEED in "${SEEDS[@]}"; do
    run_one 0 "baseline_token" "configs/a4_long_context/baseline_dense_lc.yaml" "${SEED}" "no"
  done
  # set_dense (exact backend)
  for SEED in "${SEEDS[@]}"; do
    run_one 0 "set_dense" "configs/a4_long_context/set_dense_lc.yaml" "${SEED}" "yes"
  done
}

run_gpu1_worker () {
  local SEED
  # set_sparse (local_band backend)
  for SEED in "${SEEDS[@]}"; do
    run_one 1 "set_sparse" "configs/a4_long_context/set_sparse_lc.yaml" "${SEED}" "yes"
  done
  # set_linear (landmark backend)
  for SEED in "${SEEDS[@]}"; do
    run_one 1 "set_linear" "configs/a4_long_context/set_linear_lc.yaml" "${SEED}" "yes"
  done
}

record_prelaunch

run_gpu0_worker > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"
run_gpu1_worker > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

wait "${PID0}"
wait "${PID1}"
echo "=== A4.2 long-context slice complete ==="
