#!/usr/bin/env bash
# SD-9.6 long-context multiresolution queue.
#
# Purpose: discover/sweep guarded long-context operating points. Use
# HOST_TAG=blue for blue-demon and HOST_TAG=lizmark for lizmark recovery rows.
#
# Defaults run a cheap support probe:
#   LENGTHS="4096 2048" ROW_SET=support_probe MODE=probe
#
# Useful row sets:
#   support_probe:  mixed65, all_fine, all_coarse at seed 0
#   blur_sweep:     all_fine, mixed25, mixed50, mixed65, mixed75, all_coarse
#   complement_5s:  mixed25, all_fine, all_coarse at seeds 3,4
set -euo pipefail

cd "${REPO_ROOT:-$HOME/set-attention}"

PROJECT="${PROJECT:-set-attention}"
IMAGE="${IMAGE:-set-attention:latest}"
CONFIG="${CONFIG:-configs/set_dictionary/sd9_multiresolution.yaml}"
LENGTHS="${LENGTHS:-4096 2048}"
ROW_SET="${ROW_SET:-support_probe}"
MODE="${MODE:-probe}"
SEEDS="${SEEDS:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
VARIANTS="${VARIANTS:-}"
HOST_TAG="${HOST_TAG:-blue}"

LR="${LR:-1e-4}"
BATCH="${BATCH:-1}"
BACKEND="landmark"
ATTENTION_FAMILY="linear"
COVERAGE="0.25"

case "${MODE}" in
  probe)
    EPOCHS="${EPOCHS:-1}"
    WARMUP="${WARMUP:-0}"
    LIMIT="${LIMIT:-500}"
    ;;
  full)
    EPOCHS="${EPOCHS:-10}"
    WARMUP="${WARMUP:-1000}"
    LIMIT="${LIMIT:-}"
    ;;
  *)
    echo "MODE must be probe or full" >&2
    exit 2
    ;;
esac

D_MODEL="${D_MODEL:-384}"
D_FF="${D_FF:-1536}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"
D_PHI="${D_PHI:-384}"
SET_STATE_DIM="${SET_STATE_DIM:-384}"
HASH_BINS="${HASH_BINS:-128}"

row_specs () {
  case "${ROW_SET}" in
    support_probe)
      printf '%s\n' \
        "mixed65 3 5 0" \
        "all_fine 8 0 0" \
        "all_coarse 0 8 0"
      ;;
    blur_sweep)
      printf '%s\n' \
        "all_fine 8 0 ${SEEDS}" \
        "mixed25 6 2 ${SEEDS}" \
        "mixed50 4 4 ${SEEDS}" \
        "mixed65 3 5 ${SEEDS}" \
        "mixed75 2 6 ${SEEDS}" \
        "all_coarse 0 8 ${SEEDS}"
      ;;
    complement_5s)
      printf '%s\n' \
        "mixed25 6 2 3,4" \
        "all_fine 8 0 3,4" \
        "all_coarse 0 8 3,4"
      ;;
    *)
      echo "unknown ROW_SET=${ROW_SET}" >&2
      exit 2
      ;;
  esac
}

variant_enabled () {
  local variant="$1"
  local wanted
  if [[ -z "${VARIANTS}" ]]; then
    return 0
  fi
  for wanted in ${VARIANTS//,/ }; do
    if [[ "${variant}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

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
  local seq_len="$1"
  local out_root="$2"
  local log_root="$3"
  python3 - "$seq_len" "$out_root" "$log_root" "$ROW_SET" "$MODE" "$EPOCHS" "$LIMIT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

seq_len, out_root, log_root, row_set, mode, epochs, limit = sys.argv[1:]

def run(cmd):
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip().splitlines(),
        "stderr": proc.stderr.strip().splitlines(),
    }

payload = {
    "phase": "SD-9.6-long-multires",
    "seq_len": int(seq_len),
    "row_set": row_set,
    "mode": mode,
    "epochs": int(epochs),
    "data_limit": int(limit) if limit else "NA",
    "out_root": out_root,
    "log_root": log_root,
    "backend": "landmark",
    "landmark_coverage": 0.25,
    "batch_size": 1,
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
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
}
host_tag = Path(out_root).parts[2].replace("sd9_6_", "").replace("_long", "") if len(Path(out_root).parts) > 2 else "unknown"
Path(f"audit/SD_9_6_{host_tag}_long_multires_prelaunch_L{seq_len}_{row_set}_{mode}.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

run_one () {
  local seq_len="$1"
  local gpu="$2"
  local variant="$3"
  local fine_heads="$4"
  local coarse_heads="$5"
  local seed="$6"
  local out_root="$7"
  local log_root="$8"
  local status_path="$9"
  local group_yaml
  group_yaml="$(groups_yaml "${fine_heads}" "${coarse_heads}")"
  local blur_pct=$(( coarse_heads * 100 / HEADS ))
  local group="sd9_6_${HOST_TAG}_${ROW_SET}_${variant}"
  local name="sd9_6_${HOST_TAG}_${ROW_SET}_${variant}_blur${blur_pct}_L${seq_len}_B${BATCH}_${BACKEND}_seed${seed}"
  local csv_path="${out_root}/${group}/${name}.csv"
  local log_path="${log_root}/${name}.log"

  mkdir -p "${out_root}/${group}"
  if completed_csv "${csv_path}" "${EPOCHS}"; then
    echo "=== Skipping complete ${name} ==="
    return 0
  fi

  echo "=== Running ${name} on GPU ${gpu} ==="
  local overrides=(
    "training.output_dir=${out_root}/${group}/${name}"
    "logging.wandb.enable=false"
    "logging.wandb.project=${PROJECT}"
    "logging.wandb.run_name=${name}"
    "data.dataset=wikitext2"
    "data.batch_size=${BATCH}"
    "data.seq_len=${seq_len}"
    "training.seed=${seed}"
    "training.epochs=${EPOCHS}"
    "training.lr=${LR}"
    "training.warmup_steps=${WARMUP}"
    "model.attention_family=${ATTENTION_FAMILY}"
    "model.backend=${BACKEND}"
    "model.backend_params.landmark_coverage=${COVERAGE}"
    "model.max_seq_len=${seq_len}"
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
  printf "scale\t%s\t%s\t%s\t%s\t%s\t2\t1\t4\t2\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${variant}" "${gpu}" "${seed}" "${fine_heads}" "${coarse_heads}" \
    "${BACKEND}" "${COVERAGE}" "${seq_len}" "${BATCH}" "${csv_path}" "${rc}" >> "${status_path}"
  return 0
}

run_length () {
  local seq_len="$1"
  local out_root="out/paper_mechanisms/sd9_6_${HOST_TAG}_long/${ROW_SET}/L${seq_len}_${MODE}"
  local log_root="logs/sd9_6_${HOST_TAG}_long/${ROW_SET}/L${seq_len}_${MODE}"
  local status_path="${out_root}/sd9_6_${HOST_TAG}_long_status.tsv"
  mkdir -p "${out_root}" "${log_root}" audit
  if [[ ! -f "${status_path}" ]]; then
    printf "role\tvariant\tgpu\tseed\tfine_heads\tcoarse_heads\tfine_w\tfine_s\tcoarse_w\tcoarse_s\tbackend\tcoverage\tseq_len\tbatch\tcsv\texit_code\n" > "${status_path}"
  fi
  record_prelaunch "${seq_len}" "${out_root}" "${log_root}"

  mapfile -t specs < <(row_specs)
  worker_for_gpu () {
    local gpu="$1"
    local parity="$2"
    local row_index=0
    for spec in "${specs[@]}"; do
      read -r variant fine_heads coarse_heads seeds_csv <<< "${spec}"
      if ! variant_enabled "${variant}"; then
        continue
      fi
      read -r -a seeds <<< "${seeds_csv//,/ }"
      for seed in "${seeds[@]}"; do
        if [[ $(( row_index % 2 )) -eq "${parity}" ]]; then
          run_one "${seq_len}" "${gpu}" "${variant}" "${fine_heads}" "${coarse_heads}" "${seed}" "${out_root}" "${log_root}" "${status_path}"
        fi
        row_index=$((row_index + 1))
      done
    done
  }

  worker_for_gpu "${GPU0}" 0 > "${log_root}/worker_gpu0.log" 2>&1 &
  local pid0="$!"
  worker_for_gpu "${GPU1}" 1 > "${log_root}/worker_gpu1.log" 2>&1 &
  local pid1="$!"
  echo "${pid0}" > "${log_root}/worker_gpu0.pid"
  echo "${pid1}" > "${log_root}/worker_gpu1.pid"
  echo "SD-9.6 ${HOST_TAG} ${ROW_SET}/${MODE} L=${seq_len}: GPU0 worker ${pid0}, GPU1 worker ${pid1}"
  wait "${pid0}"
  wait "${pid1}"
  echo "=== SD-9.6 ${HOST_TAG} ${ROW_SET}/${MODE} L=${seq_len} complete ==="
}

echo "=== SD-9.6 ${HOST_TAG} long queue start: ROW_SET=${ROW_SET} MODE=${MODE} LENGTHS=${LENGTHS} VARIANTS=${VARIANTS:-all} ==="
for seq_len in ${LENGTHS}; do
  run_length "${seq_len}"
done
echo "=== SD-9.6 ${HOST_TAG} long queue complete ==="
