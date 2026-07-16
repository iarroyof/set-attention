#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LAUNCH=0
PREFLIGHT=0
RUN_B25=0
for arg in "$@"; do
  case "$arg" in
    --launch) LAUNCH=1 ;;
    --preflight-one-step) PREFLIGHT=1 ;;
    --b25) RUN_B25=1 ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "${MRP3_MQAR_LAUNCH:-}" != "approved" || "$LAUNCH" != "1" ]]; then
  cat >&2 <<'MSG'
Refusing to launch the registered MRP-3 MQAR calibration.

Required:
  MRP3_MQAR_LAUNCH=approved
  scripts/run_mqar_calibration.sh --launch

Use --preflight-one-step for a one-update validation pass.
Use --b25 only after selecting MQAR_CALIBRATION_SELECTED_LR from token calibration.
MSG
  exit 3
fi

COMMON_OVERRIDES=(
  "stage=mqar_calibration_registered"
  "training.seed=0"
  "training.max_updates=${MQAR_CALIBRATION_MAX_UPDATES:-20000}"
  "training.eval_every_updates=${MQAR_CALIBRATION_EVAL_EVERY:-500}"
  "training.calibration_accuracy_threshold=0.99"
  "training.calibration_consecutive_evals=2"
  "training.checkpoint.save_final=${MQAR_CALIBRATION_SAVE_FINAL:-true}"
  "data.seq_len=512"
  "data.num_kv_pairs=64"
  "data.num_train_examples=100000"
  "data.num_val_examples=3000"
  "data.batch_size=16"
)

COMMON_ARGS=()
if [[ "$PREFLIGHT" == "1" ]]; then
  COMMON_ARGS+=(--preflight-one-step)
fi

run_token_lr() {
  local lr="$1"
  local tag="${lr//./p}"
  python scripts/run_mqar.py \
    --config configs/mqar/primary_token.yaml \
    --csv-path "out/mqar_calibration/token_lr${tag}_seed0.csv" \
    --override \
      "${COMMON_OVERRIDES[@]}" \
      "training.lr=${lr}" \
      "training.output_dir=out/mqar_calibration/token_lr${tag}_seed0" \
    "${COMMON_ARGS[@]}"
}

run_b25_lr() {
  local lr="$1"
  local tag="${lr//./p}"
  python scripts/run_mqar.py \
    --config configs/mqar/primary_b25.yaml \
    --csv-path "out/mqar_calibration/b25_lr${tag}_seed0.csv" \
    --override \
      "${COMMON_OVERRIDES[@]}" \
      "training.lr=${lr}" \
      "training.output_dir=out/mqar_calibration/b25_lr${tag}_seed0" \
      "data.mqar_row=b25" \
    "${COMMON_ARGS[@]}"
}

if [[ "$RUN_B25" == "1" ]]; then
  if [[ -z "${MQAR_CALIBRATION_SELECTED_LR:-}" ]]; then
    echo "MQAR_CALIBRATION_SELECTED_LR is required with --b25" >&2
    exit 4
  fi
  run_b25_lr "$MQAR_CALIBRATION_SELECTED_LR"
  exit 0
fi

for lr in 0.0001 0.0003 0.001; do
  run_token_lr "$lr"
done
