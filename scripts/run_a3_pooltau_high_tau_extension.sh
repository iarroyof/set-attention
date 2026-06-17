#!/usr/bin/env bash
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a3_pooltau_sweep_high_tau}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a3_pooltau_sweep}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a3_pooltau_sweep}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"
WINDOW=16
STRIDE=4
TAUS=(0.5 0.95)
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

manifest_path = Path("out/paper_integrated_evidence/checks/a3_pooltau_sweep_manifest.json")
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
Path("audit/A3_2_pooltau_high_tau_extension_prelaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "existing_a3_2_manifest_status": manifest.get("status"),
    "existing_a3_2_expected_runs": manifest.get("expected_runs"),
    "existing_a3_2_validated_runs": manifest.get("validated_runs"),
    "extension": {
        "tau_pool": [0.5, 0.95],
        "families": ["dense_exact", "sparse_local_band", "linear_landmark"],
        "seeds": [0, 1, 2],
        "D": 384,
        "d_ff": 1536,
        "L": 512,
        "w": 16,
        "s": 4,
        "M": 125,
        "lr": "1e-4",
        "set_causality_mode": "strict_past",
        "landmark_coverage": 0.25,
    },
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

  if [[ -s "${CSV_PATH}" && -s "${CSV_PATH%.csv}.json" ]]; then
    echo "=== Skipping existing ${NAME}"
    return 0
  fi

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

run_gpu0 () {
  local TAU SEED
  for TAU in "${TAUS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      run_one 0 dense_exact "${TAU}" "${SEED}"
    done
  done
  for SEED in "${SEEDS[@]}"; do
    run_one 0 sparse_local_band 0.5 "${SEED}"
  done
}

run_gpu1 () {
  local TAU SEED
  for SEED in "${SEEDS[@]}"; do
    run_one 1 sparse_local_band 0.95 "${SEED}"
  done
  for TAU in "${TAUS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      run_one 1 linear_landmark "${TAU}" "${SEED}"
    done
  done
}

record_prelaunch

run_gpu0 > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"
run_gpu1 > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

wait "${PID0}"
wait "${PID1}"
