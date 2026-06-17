#!/usr/bin/env bash
# A4.1 long-context smoke -- L=2048, dense baseline + dense SKA (w=16, s=8), seed 0.
# One run per GPU (GPU0: baseline, GPU1: set_dense). 10 epochs.
# M at L=2048, w=16, s=8: floor((2048-16)/8)+1 = 255.
set -euo pipefail

cd ~/set-attention

LOG_ROOT="${LOG_ROOT:-logs/a41_smoke}"
OUT_ROOT="${OUT_ROOT:-out/paper_mechanisms/a41_smoke}"
PROJECT="${PROJECT:-set-attention}"
GROUP_PREFIX="${GROUP_PREFIX:-a41_smoke}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LR="${LR:-1e-4}"
SEQ_LEN=2048
WINDOW=16
STRIDE=8
SEED=0
BATCH=4   # reduced from 16: fp32 dense attn at L=2048 peaks ~22 GiB at B=16; B=4 fits ~6 GiB

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

manifest_path = Path("out/paper_integrated_evidence/checks/a3_stride_sweep_manifest.json")
handoff_path  = Path("audit/A3_3_stride_sweep.md")
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

Path("audit/A4_1_smoke_prelaunch.json").write_text(json.dumps({
    "branch":       run(["git", "branch", "--show-current"]),
    "head":         run(["git", "rev-parse", "HEAD"]),
    "status_short": run(["git", "status", "--short"]),
    "a3_3_manifest_exists":   manifest_path.exists(),
    "a3_3_manifest_status":   manifest.get("status"),
    "a3_3_validated_runs":    manifest.get("validated_runs"),
    "a3_3_expected_runs":     manifest.get("expected_runs"),
    "a3_3_audit_status_line": handoff_path.read_text().splitlines()[2] if handoff_path.exists() else None,
    "locked_design_note": (
        "A4.1 long-context smoke: L=2048, D=384, d_ff=1536, batch=16. "
        "dense baseline (baseline_token) on GPU0; dense SKA (set_only, w=16, s=8) on GPU1. "
        "1 seed, 10 epochs. M=255 for SKA at L=2048, w=16, s=8. "
        "Memory budget verified: ~15.6 GB peak per GPU (fp32) < 24 GB RTX4090 VRAM."
    ),
}, indent=2) + "\n")
PY
}

run_baseline () {
  local LR_TAG="${LR//./p}"
  local NAME="a41_baseline_dense_D384_FF1536_L${SEQ_LEN}_lr${LR_TAG}_seed${SEED}"
  local GROUP="${GROUP_PREFIX}_baseline_dense_L${SEQ_LEN}"
  local CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  local LOG_PATH="${LOG_ROOT}/${NAME}.log"

  mkdir -p "${OUT_ROOT}/${GROUP}"

  echo "=== Running ${NAME} on GPU 0 ==="
  docker compose exec -T \
    -e CUDA_VISIBLE_DEVICES="0" \
    -e HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    -e WANDB_MODE="${WANDB_MODE}" \
    -e WANDB_PROJECT="${PROJECT}" \
    -e WANDB_NAME="${NAME}" \
    -e WANDB_RUN_GROUP="${GROUP}" \
    set-attention \
    python scripts/run_experiment.py \
    --config "configs/a4_long_context/baseline_dense_lc.yaml" \
    --wandb \
    --wandb-project "${PROJECT}" \
    --csv-path "${CSV_PATH}" \
    --override \
      training.output_dir="${OUT_ROOT}/${GROUP}/${NAME}" \
      data.dataset=wikitext2 \
      data.batch_size="${BATCH}" \
      data.seq_len="${SEQ_LEN}" \
      training.seed="${SEED}" \
      training.epochs=10 \
      training.lr="${LR}" \
      training.warmup_steps=1000 \
      model.d_model=384 \
      model.dim_feedforward=1536 \
      model.num_layers=6 \
      model.num_heads=8 \
      model.max_seq_len="${SEQ_LEN}" \
    | tee "${LOG_PATH}"
}

run_set_dense () {
  local LR_TAG="${LR//./p}"
  local NAME="a41_set_dense_D384_FF1536_L${SEQ_LEN}_w${WINDOW}_s${STRIDE}_lr${LR_TAG}_seed${SEED}"
  local GROUP="${GROUP_PREFIX}_set_dense_L${SEQ_LEN}"
  local CSV_PATH="${OUT_ROOT}/${GROUP}/${NAME}.csv"
  local LOG_PATH="${LOG_ROOT}/${NAME}.log"

  mkdir -p "${OUT_ROOT}/${GROUP}"

  echo "=== Running ${NAME} on GPU 1 ==="
  docker compose exec -T \
    -e CUDA_VISIBLE_DEVICES="1" \
    -e HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    -e WANDB_MODE="${WANDB_MODE}" \
    -e WANDB_PROJECT="${PROJECT}" \
    -e WANDB_NAME="${NAME}" \
    -e WANDB_RUN_GROUP="${GROUP}" \
    set-attention \
    python scripts/run_experiment.py \
    --config "configs/a4_long_context/set_dense_lc.yaml" \
    --wandb \
    --wandb-project "${PROJECT}" \
    --csv-path "${CSV_PATH}" \
    --override \
      training.output_dir="${OUT_ROOT}/${GROUP}/${NAME}" \
      data.dataset=wikitext2 \
      data.batch_size="${BATCH}" \
      data.seq_len="${SEQ_LEN}" \
      training.seed="${SEED}" \
      training.epochs=10 \
      training.lr="${LR}" \
      training.warmup_steps=1000 \
      model.d_model=384 \
      model.dim_feedforward=1536 \
      model.num_layers=6 \
      model.num_heads=8 \
      model.max_seq_len="${SEQ_LEN}" \
      model.window_size="${WINDOW}" \
      model.stride="${STRIDE}" \
      model.set_causality_mode=strict_past \
      model.router_topk=16 \
      model.router_temperature=1.0 \
      model.pooling.mode=soft_trimmed_boltzmann \
      model.pooling.tau=0.1 \
      model.pooling.q=0.85 \
      model.router_multihead=true \
      model.pooling_multihead=false \
    | tee "${LOG_PATH}"
}

record_prelaunch

run_baseline  > "${LOG_ROOT}/worker_gpu0.log" 2>&1 &
PID0="$!"
run_set_dense > "${LOG_ROOT}/worker_gpu1.log" 2>&1 &
PID1="$!"

wait "${PID0}"
wait "${PID1}"
echo "=== A4.1 long-context smoke complete ==="
