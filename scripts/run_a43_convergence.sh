#!/usr/bin/env bash
# A4.3 convergence panel -- 30 epochs at LR-norm headline reference.
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a43_convergence}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a43_convergence}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a43_convergence}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
SEQ_LEN=512
WINDOW=16
STRIDE=8
SEED=0
BATCH=16
EPOCHS=30

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

manifests = {}
for name, path in {
    "a2_baseline_controls": "out/paper_integrated_evidence/checks/a2_baseline_controls_manifest.json",
    "a3_window_baseline_controls": "out/paper_integrated_evidence/checks/a3_window_baseline_controls_manifest.json",
    "a4_long_context_baseline_controls": "out/paper_integrated_evidence/checks/a4_long_context_baseline_controls_manifest.json",
}.items():
    p = Path(path)
    manifests[name] = json.loads(p.read_text()) if p.exists() else {"status": "missing"}

Path("audit/A4_3_convergence_prelaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "control_manifests": manifests,
    "locked_design_note": (
        "A4.3 convergence panel: 6 families, seed=0, 30 epochs, D=384,d_ff=1536,"
        "L=512,w=16,s=8,M=63. All selected LRs are 1e-4 by lowest mean val PPL "
        "across seeds in A2/A2.4 summaries."
    ),
}, indent=2) + "\n")
PY
}

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

is_set_family () {
  case "$1" in
    set_*) return 0 ;;
    *) return 1 ;;
  esac
}

run_one () {
  local GPU="$1"
  local FAMILY="$2"
  local LR="$3"
  local CFG
  CFG="$(config_for_family "${FAMILY}")"

  local LR_TAG="${LR//./p}"
  local GROUP="${GROUP_PREFIX}_${FAMILY}_D384_FF1536_L${SEQ_LEN}"
  local NAME="a43_${FAMILY}_D384_FF1536_L${SEQ_LEN}_w${WINDOW}_s${STRIDE}_lr${LR_TAG}_seed${SEED}_ep${EPOCHS}"
  local CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  local LOG_PATH="${LOG_ROOT}/${NAME}.log"

  mkdir -p "${OUT_ROOT}/${GROUP}"
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
  )

  if is_set_family "${FAMILY}"; then
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
  run_one 0 baseline_dense_exact 1e-4
  run_one 0 set_dense_exact 1e-4
  run_one 0 set_sparse_local_band 1e-4
}

run_gpu1_worker () {
  run_one 1 baseline_sparse_local_band 1e-4
  run_one 1 baseline_linear_landmark 1e-4
  run_one 1 set_linear_landmark 1e-4
}

record_prelaunch
run_gpu0_worker > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"
run_gpu1_worker > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

wait "${PID0}"
wait "${PID1}"
echo "=== A4.3 convergence panel complete ==="
