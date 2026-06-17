#!/usr/bin/env bash
# A6.3 set-token interface bottleneck sweep.
#
# Tests whether fixed d_phi=384 bottlenecks wider set states by adding matched
# and partially matched interface rows:
#   (set_state_dim,d_phi) in {(512,512), (768,512), (768,768)}
# Existing A6.2 rows provide the bottleneck references (512,384) and (768,384).
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a6_interface_bottleneck}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a6_interface_bottleneck}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a6_interface_bottleneck}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"
SEQ_LEN=512
WINDOW=16
STRIDE=8
BATCH=16
EPOCHS=10
D_MODEL=384
D_FF=1536
PAIRS=("512:512" "768:512" "768:768")
SEEDS=(0 1 2)
FAMILIES=(set_dense_exact set_sparse_local_band set_linear_landmark)

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" audit

config_for_family () {
  case "$1" in
    set_dense_exact) echo "configs/paper_lr_norm/set_dense_exact.yaml" ;;
    set_sparse_local_band) echo "configs/paper_lr_norm/set_sparse_local_band.yaml" ;;
    set_linear_landmark) echo "configs/paper_lr_norm/set_linear_landmark.yaml" ;;
    *) echo "unknown family: $1" >&2; return 1 ;;
  esac
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


manifest_path = Path("out/paper_integrated_evidence/checks/a6_set_state_width_manifest.json")
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"status": "missing"}
Path("audit/A6_3_interface_bottleneck_prelaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "a6_2_status": manifest.get("status"),
    "design_note": (
        "A6.3 tests whether d_phi bottlenecks the set-token routing/interface. "
        "Token width is fixed at D=384, d_ff=1536 and set_state_dim is held at "
        "512 or 768 while d_phi is raised to 512 or 768. Existing A6.2 rows "
        "provide d_phi=384 references."
    ),
}, indent=2) + "\n")
PY
}

run_one () {
  local GPU="$1"
  local FAMILY="$2"
  local SET_STATE_DIM="$3"
  local DPHI="$4"
  local SEED="$5"
  local CFG
  CFG="$(config_for_family "${FAMILY}")"

  local LR_TAG="${LR//./p}"
  local GROUP="${GROUP_PREFIX}_${FAMILY}_D${D_MODEL}_FF${D_FF}"
  local NAME="a6_iface_${FAMILY}_D${D_MODEL}_FF${D_FF}_setdim${SET_STATE_DIM}_dphi${DPHI}_w${WINDOW}_s${STRIDE}_lr${LR_TAG}_seed${SEED}"
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
    model.d_model="${D_MODEL}"
    model.dim_feedforward="${D_FF}"
    model.num_layers=6
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
  local FAMILY PAIR SET_STATE_DIM DPHI SEED
  for FAMILY in "${FAMILIES[@]}"; do
    for PAIR in "${PAIRS[@]}"; do
      SET_STATE_DIM="${PAIR%%:*}"
      DPHI="${PAIR##*:}"
      for SEED in "${SEEDS[@]}"; do
        if (( INDEX % 2 == MOD )); then
          run_one "${GPU}" "${FAMILY}" "${SET_STATE_DIM}" "${DPHI}" "${SEED}"
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

echo "A6.3 interface bottleneck workers launched: GPU0 PID=${PID0}, GPU1 PID=${PID1}"
wait "${PID0}"
wait "${PID1}"
echo "=== A6.3 interface bottleneck sweep complete ==="
