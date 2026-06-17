#!/usr/bin/env bash
# A8 hybrid sparse/local-band progressive topology sweep.
#
# Runs layer-level token/set hybrids with one shared token stream. The set
# layers use strict-past endpoint routing and progressive near-2/near-4
# topologies: early set layers (w=4,s=2), later set layers (w=8,s=4).
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a8_hybrid_sparse_progressive}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a8_hybrid_sparse_progressive}"
STATUS_PATH="${OUT_ROOT}/a8_hybrid_sparse_progressive_status.tsv"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a8_hybrid_sparse_progressive}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"

CONFIGS=(
  "TTSSSS configs/a8_hybrid/sparse_progressive_TTSSSS.yaml"
  "TSTSTS configs/a8_hybrid/sparse_progressive_TSTSTS.yaml"
  "TTTTSS configs/a8_hybrid/sparse_progressive_TTTTSS.yaml"
)
SEEDS=(0 1 2)

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" audit out/paper_integrated_evidence/checks

config_fields () {
  local CFG="$1"
  docker compose exec -T -e PYTHONPATH=. set-attention python - "${CFG}" <<'PY'
import sys
from config.load import load_config

cfg = load_config(sys.argv[1])
model = cfg["model"]
data = cfg["data"]
training = cfg["training"]
lr = float(training["lr"])
lr_tag = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
print("\t".join(str(x) for x in [
    model["d_model"],
    model["dim_feedforward"],
    data["seq_len"],
    data["batch_size"],
    training["epochs"],
    training["lr"],
    lr_tag,
]))
PY
}

completed_csv () {
  local CSV_PATH="$1"
  local EXPECTED_EPOCHS="$2"
  python3 - "$CSV_PATH" "$EXPECTED_EPOCHS" <<'PY'
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


Path("audit/A8_hybrid_sparse_progressive_prelaunch.json").write_text(json.dumps({
    "branch": run(["git", "branch", "--show-current"]),
    "head": run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "purpose": (
        "Test whether layer-level token/set hybrids preserve fine-grained "
        "information through a shared token stream while using strict-past "
        "set layers only in selected layers."
    ),
    "config_contract": (
        "Model/data/training hyperparameters are read from configs/a8_hybrid/*.yaml. "
        "The launcher overrides only seed, output_dir, csv path, and W&B run identity."
    ),
    "configs": [
        "configs/a8_hybrid/sparse_progressive_TTSSSS.yaml",
        "configs/a8_hybrid/sparse_progressive_TSTSTS.yaml",
        "configs/a8_hybrid/sparse_progressive_TTTTSS.yaml",
    ],
    "seeds": [0, 1, 2],
    "patterns": {
        "TTSSSS": "two token layers, then four set layers; set topologies 4:2,4:2,8:4,8:4",
        "TSTSTS": "alternating token/set bridge; set topologies 4:2,4:2,8:4",
        "TTTTSS": "four token layers, then two set layers; set topologies 4:2,8:4",
    },
    "expected_runs": 9,
    "seed_extension_policy": "Seeds 3,4 are intentionally held for follow-up approval after this first 3-seed pass.",
}, indent=2) + "\n")
PY
}

run_one () {
  local GPU="$1"
  local PATTERN="$2"
  local CFG="$3"
  local SEED="$4"
  local D_MODEL D_FF SEQ_LEN BATCH EPOCHS LR LR_TAG
  IFS=$'\t' read -r D_MODEL D_FF SEQ_LEN BATCH EPOCHS LR LR_TAG <<<"$(config_fields "${CFG}")"
  local GROUP="${GROUP_PREFIX}_${PATTERN}_D${D_MODEL}_FF${D_FF}"
  local NAME="a8_hybrid_sparse_${PATTERN}_D${D_MODEL}_FF${D_FF}_L${SEQ_LEN}_lr${LR_TAG}_seed${SEED}"
  local CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  local LOG_PATH="${LOG_ROOT}/${NAME}.log"

  mkdir -p "${OUT_ROOT}/${GROUP}"
  if completed_csv "${CSV_PATH}" "${EPOCHS}"; then
    echo "=== Skipping complete ${NAME} ==="
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${NAME}" "${PATTERN}" "${SEED}" "${GPU}" "0" "skipped_complete" >> "${STATUS_PATH}"
    return 0
  fi

  echo "=== Running ${NAME} on GPU ${GPU} ==="
  local OVERRIDES=(
    training.output_dir="${OUT_ROOT}/${GROUP}/${NAME}"
    logging.wandb.project="${PROJECT}"
    logging.wandb.run_name="${NAME}"
    training.seed="${SEED}"
  )

  set +e
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
  local EXIT_CODE="${PIPESTATUS[0]}"
  set -e
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${NAME}" "${PATTERN}" "${SEED}" "${GPU}" "${EXIT_CODE}" "${CSV_PATH}" >> "${STATUS_PATH}"
  return "${EXIT_CODE}"
}

run_worker () {
  local GPU="$1"
  local MOD="$2"
  local INDEX=0
  local ENTRY PATTERN CFG SEED
  for ENTRY in "${CONFIGS[@]}"; do
    read -r PATTERN CFG <<<"${ENTRY}"
    for SEED in "${SEEDS[@]}"; do
      if (( INDEX % 2 == MOD )); then
        run_one "${GPU}" "${PATTERN}" "${CFG}" "${SEED}"
      fi
      INDEX=$((INDEX + 1))
    done
  done
}

record_prelaunch
printf "run_name\tpattern\tseed\tgpu\texit_code\tcsv_path\n" > "${STATUS_PATH}"
run_worker 0 0 > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"
run_worker 1 1 > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

echo "A8 hybrid sparse progressive workers launched: GPU0 PID=${PID0}, GPU1 PID=${PID1}"
wait "${PID0}"
wait "${PID1}"
echo "=== A8 hybrid sparse progressive sweep complete ==="
