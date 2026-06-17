#!/usr/bin/env bash
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a3_pooltau_sweep}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a3_pooltau_sweep}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a3_pooltau_sweep}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"
WINDOW=16
STRIDE=4
TAUS=(0.05 0.1 0.2)
SEEDS=(0 1 2)

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" audit

config_for_family () {
  case "$1" in
    dense_exact) echo "configs/paper_complements/family_dense_exact.yaml" ;;
    sparse_local_band) echo "configs/paper_complements/family_sparse_local_band.yaml" ;;
    linear_landmark) echo "configs/paper_complements/family_linear_landmark.yaml" ;;
    *) echo "unknown family: $1" >&2; return 1 ;;
  esac
}

tau_tag () {
  echo "$1" | sed 's/\./p/g'
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

manifest_path = Path("out/paper_integrated_evidence/checks/a3_window_sweep_manifest.json")
handoff_path = Path("audit/A3_1_window_sweep.md")
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
stale = run([
    "grep", "-RInE",
    r"M=64|M = 64|ceil\(512 / 8\)|num_landmarks|backend:[[:space:]]*nystrom|noncausal|all-A2|override all A2",
    "scripts/gpu1_run_pooltau_family_complements.sh",
    "scripts/summarize_paper_family_complements.py",
    "scripts/run_a3_pooltau_sweep.sh",
    "configs/paper_complements",
])

Path("audit/A3_2_pooltau_sweep_prelaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "a3_1_manifest_exists": manifest_path.exists(),
    "a3_1_manifest_status": manifest.get("status"),
    "a3_1_expected_runs": manifest.get("expected_runs"),
    "a3_1_validated_runs": manifest.get("validated_runs"),
    "a3_1_audit_status_line": handoff_path.read_text().splitlines()[2] if handoff_path.exists() else None,
    "stale_assumption_check": stale,
    "locked_stride_confirmation": (
        "A3.2 pooling-temperature mechanism sweep uses anchor topology reference "
        "s=4,w=16. A2.2/A2.3 LR-normalized headline/family grid remains s=8 "
        "and is not overridden."
    ),
}, indent=2) + "\n")
PY
}

run_one () {
  local GPU="$1"
  local FAMILY="$2"
  local TAU="$3"
  local SEED="$4"
  local CFG
  CFG="$(config_for_family "${FAMILY}")"

  local LR_TAG="${LR//./p}"
  local TAU_TAG
  TAU_TAG="$(tau_tag "${TAU}")"
  local NAME="a3_pooltau_${FAMILY}_D384_FF1536_w${WINDOW}_s${STRIDE}_tau${TAU_TAG}_lr${LR_TAG}_seed${SEED}"
  local GROUP="${GROUP_PREFIX}_${FAMILY}_D384_FF1536_s${STRIDE}_w${WINDOW}"
  local CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  local LOG_PATH="${LOG_ROOT}/${NAME}.log"

  echo "=== Running ${NAME} on GPU ${GPU}"
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
      data.batch_size=16 \
      data.seq_len=512 \
      training.seed="${SEED}" \
      training.epochs=10 \
      training.lr="${LR}" \
      training.warmup_steps=1000 \
      model.d_model=384 \
      model.dim_feedforward=1536 \
      model.num_layers=6 \
      model.num_heads=8 \
      model.window_size="${WINDOW}" \
      model.stride="${STRIDE}" \
      model.set_causality_mode=strict_past \
      model.router_topk=16 \
      model.router_temperature=1.0 \
      model.pooling.mode=soft_trimmed_boltzmann \
      model.pooling.tau="${TAU}" \
      model.pooling.q=0.85 \
      model.router_multihead=true \
      model.pooling_multihead=false \
    | tee "${LOG_PATH}"
}

run_dense_worker () {
  local TAU SEED
  for TAU in "${TAUS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      run_one 0 dense_exact "${TAU}" "${SEED}"
    done
  done
  for SEED in "${SEEDS[@]}"; do
    run_one 0 sparse_local_band 0.05 "${SEED}"
  done
}

run_sparse_linear_worker () {
  local TAU SEED
  for TAU in 0.1 0.2; do
    for SEED in "${SEEDS[@]}"; do
      run_one 1 sparse_local_band "${TAU}" "${SEED}"
    done
  done
  for TAU in "${TAUS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      run_one 1 linear_landmark "${TAU}" "${SEED}"
    done
  done
}

record_prelaunch

run_dense_worker > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"
run_sparse_linear_worker > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

wait "${PID0}"
wait "${PID1}"
