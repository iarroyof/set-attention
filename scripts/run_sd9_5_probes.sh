#!/usr/bin/env bash
# SD-9.5 mechanism probes for the verified SD-9 mixed-resolution winner.
#
# ROLE=short: L=512, batch=16, exact backend probe retraining.
# ROLE=scale: L=${SEQ_LEN:-16384}, batch=1, landmark backend with coverage 0.25.
# SMOKE=1 runs the production-shape mixed row for seed 0 and one epoch.
set -euo pipefail

cd "${REPO_ROOT:-$HOME/set-attention}"

ROLE="${ROLE:-short}"
SMOKE="${SMOKE:-0}"
VARIANTS="${VARIANTS:-all}"
PROJECT="${PROJECT:-set-attention}"
IMAGE="${IMAGE:-set-attention:latest}"
LR="${LR:-1e-4}"
CONFIG="${CONFIG:-configs/set_dictionary/sd9_multiresolution.yaml}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

case "${ROLE}" in
  short)
    SEQ_LEN="${SEQ_LEN:-512}"
    BATCH="${BATCH:-16}"
    BACKEND="exact"
    ATTENTION_FAMILY="dense"
    COVERAGE_ARG=()
    DEFAULT_LOG_ROOT="logs/sd9_5_probes_short"
    DEFAULT_OUT_ROOT="out/paper_mechanisms/sd9_5_probes_short"
    ;;
  scale)
    SEQ_LEN="${SEQ_LEN:-16384}"
    BATCH="${BATCH:-1}"
    BACKEND="landmark"
    ATTENTION_FAMILY="linear"
    COVERAGE_ARG=("model.backend_params.landmark_coverage=0.25")
    DEFAULT_LOG_ROOT="logs/sd9_5_scaleL_L${SEQ_LEN}"
    DEFAULT_OUT_ROOT="out/paper_mechanisms/sd9_5_scaleL_L${SEQ_LEN}"
    ;;
  *)
    echo "ROLE must be short or scale" >&2
    exit 2
    ;;
esac

if [[ "${SMOKE}" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
  WARMUP="${WARMUP:-0}"
  LIMIT="${LIMIT:-500}"
  LOG_ROOT="${LOG_ROOT:-${DEFAULT_LOG_ROOT}_smoke}"
  OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}_smoke}"
  if [[ "${ROLE}" == "scale" ]]; then
    ROW_SPECS=("mixed 3 5 0")
  else
    ROW_SPECS=("mixed 6 2 0")
  fi
else
  EPOCHS="${EPOCHS:-10}"
  WARMUP="${WARMUP:-1000}"
  LIMIT="${LIMIT:-}"
  LOG_ROOT="${LOG_ROOT:-${DEFAULT_LOG_ROOT}}"
  OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"
  if [[ "${ROLE}" == "scale" ]]; then
    ROW_SPECS=(
      "mixed 3 5 0"
      "all_fine 8 0 0"
      "all_coarse 0 8 0"
    )
  else
    ROW_SPECS=(
      "mixed 6 2 0,1,2"
      "all_fine 8 0 0,1,2"
      "all_coarse 0 8 0,1,2"
    )
  fi
fi

if [[ "${VARIANTS}" != "all" ]]; then
  IFS=',' read -r -a REQUESTED_VARIANTS <<< "${VARIANTS}"
  FILTERED_ROW_SPECS=()
  for spec in "${ROW_SPECS[@]}"; do
    read -r variant _fine _coarse _seeds <<< "${spec}"
    for requested in "${REQUESTED_VARIANTS[@]}"; do
      if [[ "${variant}" == "${requested}" ]]; then
        FILTERED_ROW_SPECS+=("${spec}")
        break
      fi
    done
  done
  if [[ "${#FILTERED_ROW_SPECS[@]}" -eq 0 ]]; then
    echo "VARIANTS=${VARIANTS} selected no rows" >&2
    exit 2
  fi
  ROW_SPECS=("${FILTERED_ROW_SPECS[@]}")
fi

D_MODEL="${D_MODEL:-384}"
D_FF="${D_FF:-1536}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"
D_PHI="${D_PHI:-384}"
SET_STATE_DIM="${SET_STATE_DIM:-384}"
HASH_BINS="${HASH_BINS:-128}"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" audit out/paper_integrated_evidence/checks
STATUS_PATH="${OUT_ROOT}/sd9_5_${ROLE}_status.tsv"
printf "role\tvariant\tgpu\tseed\tfine_heads\tcoarse_heads\tfine_w\tfine_s\tcoarse_w\tcoarse_s\tbackend\tcoverage\tseq_len\tbatch\tcsv\texit_code\n" > "${STATUS_PATH}"

completed_csv () {
  local csv_path="$1"
  local expected_epochs="$2"
  python3 - "$csv_path" "$expected_epochs" <<'PY'
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

groups_yaml () {
  local fine_heads="$1"
  local coarse_heads="$2"
  if [[ "${fine_heads}" -gt 0 && "${coarse_heads}" -gt 0 ]]; then
    printf '[{name: fine, num_heads: %s, window_size: 2, stride: 1}, {name: coarse, num_heads: %s, window_size: 4, stride: 2}]' "${fine_heads}" "${coarse_heads}"
  elif [[ "${fine_heads}" -gt 0 ]]; then
    printf '[{name: fine, num_heads: %s, window_size: 2, stride: 1}]' "${fine_heads}"
  else
    printf '[{name: coarse, num_heads: %s, window_size: 4, stride: 2}]' "${coarse_heads}"
  fi
}

record_prelaunch () {
  python3 - "$ROLE" "$SMOKE" "$CONFIG" "$OUT_ROOT" "$LOG_ROOT" "$SEQ_LEN" "$BATCH" "$BACKEND" "$LIMIT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

role, smoke, config, out_root, log_root, seq_len, batch, backend, limit = sys.argv[1:]

def run(cmd):
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip().splitlines(),
        "stderr": proc.stderr.strip().splitlines(),
    }

payload = {
    "phase": "SD-9.5",
    "role": role,
    "smoke": smoke == "1",
    "config": config,
    "out_root": out_root,
    "log_root": log_root,
    "seq_len": int(seq_len),
    "batch_size": int(batch),
    "backend": backend,
    "landmark_coverage": 0.25 if backend == "landmark" else "NA",
    "data_limit": int(limit) if limit else "NA",
    "contract": {
        "objective": "CE only",
        "anchor.enabled": False,
        "candidate_fiber": "endpoint_window",
        "output_residual_mode": "anchor_span",
        "token_mlp.enabled": False,
        "set_state_dim": 384,
        "d_model": 384,
        "num_layers": 6,
        "num_heads": 8,
        "dim_feedforward": 1536,
        "lr": "1e-4",
    },
    "rows": [
        {"variant": "mixed", "fine_heads": 6 if role == "short" else 3, "coarse_heads": 2 if role == "short" else 5},
        {"variant": "all_fine", "fine_heads": 8, "coarse_heads": 0},
        {"variant": "all_coarse", "fine_heads": 0, "coarse_heads": 8},
    ],
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
}
suffix = f"_{role}" + ("_smoke" if smoke == "1" else "")
Path(f"audit/SD_9_5_probes_prelaunch{suffix}.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

run_one () {
  local gpu="$1"
  local variant="$2"
  local fine_heads="$3"
  local coarse_heads="$4"
  local seed="$5"
  local group_yaml
  group_yaml="$(groups_yaml "${fine_heads}" "${coarse_heads}")"
  local blur_pct=$(( coarse_heads * 100 / HEADS ))
  local lr_tag="${LR//./p}"
  local group="sd9_5_${ROLE}_${variant}"
  local name="sd9_5_${ROLE}_${variant}_blur${blur_pct}_L${SEQ_LEN}_B${BATCH}_${BACKEND}_seed${seed}"
  local csv_path="${OUT_ROOT}/${group}/${name}.csv"
  local log_path="${LOG_ROOT}/${name}.log"

  mkdir -p "${OUT_ROOT}/${group}"
  if completed_csv "${csv_path}" "${EPOCHS}"; then
    echo "=== Skipping complete ${name} ==="
    return 0
  fi

  echo "=== Running ${name} on GPU ${gpu} ==="
  local overrides=(
    "training.output_dir=${OUT_ROOT}/${group}/${name}"
    "logging.wandb.enable=false"
    "logging.wandb.project=${PROJECT}"
    "logging.wandb.run_name=${name}"
    "data.dataset=wikitext2"
    "data.batch_size=${BATCH}"
    "data.seq_len=${SEQ_LEN}"
    "training.seed=${seed}"
    "training.epochs=${EPOCHS}"
    "training.lr=${LR}"
    "training.warmup_steps=${WARMUP}"
    "model.attention_family=${ATTENTION_FAMILY}"
    "model.backend=${BACKEND}"
    "model.max_seq_len=${SEQ_LEN}"
    "model.d_model=${D_MODEL}"
    "model.dim_feedforward=${D_FF}"
    "model.num_layers=${LAYERS}"
    "model.num_heads=${HEADS}"
    "model.d_phi=${D_PHI}"
    "model.set_state_dim=${SET_STATE_DIM}"
    "model.feature_params.num_bins=${HASH_BINS}"
    "model.window_size=$([[ "${fine_heads}" -gt 0 ]] && echo 2 || echo 4)"
    "model.stride=$([[ "${fine_heads}" -gt 0 ]] && echo 1 || echo 2)"
    "model.output_residual_mode=anchor_span"
    "model.token_mlp.enabled=false"
    "model.multiresolution.enabled=true"
    "model.multiresolution.groups=${group_yaml}"
    "${COVERAGE_ARG[@]}"
  )
  if [[ -n "${LIMIT}" ]]; then
    overrides+=("data.limit=${LIMIT}")
  fi

  set +e
  docker run --rm \
    --gpus "device=${gpu}" \
    --ipc=host \
    -u "$(id -u):$(id -g)" \
    -e HOME=/workspace \
    -e XDG_CACHE_HOME=/workspace/.cache \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_DATASETS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e WANDB_MODE=offline \
    -e WANDB_PROJECT="${PROJECT}" \
    -e WANDB_NAME="${name}" \
    -e WANDB_RUN_GROUP="${group}" \
    -v "${PWD}:/workspace" \
    -w /workspace \
    "${IMAGE}" \
    /usr/bin/python scripts/run_experiment.py \
      --config "${CONFIG}" \
      --csv-path "${csv_path}" \
      --override "${overrides[@]}" \
    > "${log_path}" 2>&1
  local rc="$?"
  set -e
  printf "%s\t%s\t%s\t%s\t%s\t%s\t2\t1\t4\t2\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${ROLE}" "${variant}" "${gpu}" "${seed}" "${fine_heads}" "${coarse_heads}" \
    "${BACKEND}" "$([[ "${BACKEND}" == "landmark" ]] && echo 0.25 || echo NA)" \
    "${SEQ_LEN}" "${BATCH}" "${csv_path}" "${rc}" >> "${STATUS_PATH}"
  return 0
}

record_prelaunch

worker_for_gpu () {
  local gpu="$1"
  local parity="$2"
  local row_index=0
  for spec in "${ROW_SPECS[@]}"; do
    read -r variant short_fine short_coarse seeds_csv <<< "${spec}"
    local fine_heads="${short_fine}"
    local coarse_heads="${short_coarse}"
    if [[ "${ROLE}" == "scale" && "${variant}" == "mixed" ]]; then
      fine_heads=3
      coarse_heads=5
    fi
    IFS=',' read -r -a seeds <<< "${seeds_csv}"
    for seed in "${seeds[@]}"; do
      if [[ $(( row_index % 2 )) -eq "${parity}" ]]; then
        run_one "${gpu}" "${variant}" "${fine_heads}" "${coarse_heads}" "${seed}"
      fi
      row_index=$((row_index + 1))
    done
  done
}

worker_for_gpu "${GPU0}" 0 > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"
worker_for_gpu "${GPU1}" 1 > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

echo "${PID0}" > "${LOG_ROOT}/worker_gpu0.pid"
echo "${PID1}" > "${LOG_ROOT}/worker_gpu1.pid"
echo "SD-9.5 ${ROLE} workers launched: GPU0 PID=${PID0}, GPU1 PID=${PID1}"
echo "Status path: ${STATUS_PATH}"
wait "${PID0}"
wait "${PID1}"
echo "=== SD-9.5 ${ROLE} complete ==="
