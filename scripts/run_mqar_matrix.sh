#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LAUNCH=0
SMOKE=0
PREFLIGHT=0
for arg in "$@"; do
  case "$arg" in
    --launch) LAUNCH=1 ;;
    --smoke) SMOKE=1 ;;
    --preflight-one-step) PREFLIGHT=1 ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "$SMOKE" == "1" ]]; then
  python scripts/run_mqar.py --config configs/mqar/token_smoke.yaml --preflight-one-step --device "${MQAR_DEVICE:-cpu}"
  exit 0
fi

if [[ "${MRP3_MQAR_LAUNCH:-}" != "approved" || "$LAUNCH" != "1" ]]; then
  cat >&2 <<'MSG'
Refusing to launch the registered MRP-3 MQAR matrix.

This wrapper is launch-guarded. To run the registered matrix, both are required:
  MRP3_MQAR_LAUNCH=approved
  scripts/run_mqar_matrix.sh --launch

For CPU-local validation, use:
  scripts/run_mqar_matrix.sh --smoke
MSG
  exit 3
fi

COMMON_ARGS=()
if [[ "$PREFLIGHT" == "1" ]]; then
  COMMON_ARGS+=(--preflight-one-step)
fi
OUT_ROOT="${MQAR_OUT_ROOT:-out/mqar_primary}"
: "${MQAR_LR:?set calibrated MQAR_LR, e.g. 0.001}"
: "${MQAR_MAX_UPDATES:?set frozen MQAR_MAX_UPDATES, e.g. 12500}"
PRIMARY_LR="${MQAR_LR}"
PRIMARY_MAX_UPDATES="${MQAR_MAX_UPDATES}"
SAVE_FINAL="${MQAR_SAVE_FINAL:-true}"
BATCH_SIZE="${MQAR_BATCH_SIZE:-4}"
NUM_TRAIN_EXAMPLES="${MQAR_NUM_TRAIN_EXAMPLES:-100000}"
NUM_VAL_EXAMPLES="${MQAR_NUM_VAL_EXAMPLES:-3000}"
SEQ_LEN="${MQAR_SEQ_LEN:-2048}"
NUM_KV_PAIRS="${MQAR_NUM_KV_PAIRS:-256}"
SEEDS="${MQAR_SEEDS:-0 1 2}"
ROWS="${MQAR_ROWS:-token b0 b25 b50 b75 b100}"
FORCE="${MQAR_FORCE:-0}"

for numeric in PRIMARY_MAX_UPDATES BATCH_SIZE NUM_TRAIN_EXAMPLES NUM_VAL_EXAMPLES SEQ_LEN NUM_KV_PAIRS; do
  [[ "${!numeric}" =~ ^[0-9]+$ ]] || {
    echo "${numeric} must be a non-negative integer" >&2
    exit 2
  }
done

skip_if_done() {
  local out="$1"
  if [[ "$FORCE" != "1" && "$SAVE_FINAL" == "true" && -s "${out}/checkpoints/final.pt" ]]; then
    echo "SKIP checkpoint exists out=${out}/checkpoints/final.pt"
    return 0
  fi
  return 1
}

run_token() {
  local seed="$1"
  local out="${OUT_ROOT}/token_seed${seed}_B${BATCH_SIZE}"
  skip_if_done "$out" && return 0
  python scripts/run_mqar.py \
    --config configs/mqar/primary_token.yaml \
    --csv-path "${out}.csv" \
    --override \
      "stage=mqar_primary_registered" \
      "training.seed=${seed}" \
      "training.lr=${PRIMARY_LR}" \
      "training.max_updates=${PRIMARY_MAX_UPDATES}" \
      "training.checkpoint.save_final=${SAVE_FINAL}" \
      "training.output_dir=${out}" \
      "data.batch_size=${BATCH_SIZE}" \
      "data.seq_len=${SEQ_LEN}" \
      "data.num_kv_pairs=${NUM_KV_PAIRS}" \
      "data.num_train_examples=${NUM_TRAIN_EXAMPLES}" \
      "data.num_val_examples=${NUM_VAL_EXAMPLES}" \
    "${COMMON_ARGS[@]}"
}

run_set_row() {
  local row="$1"
  local fine_heads="$2"
  local coarse_heads="$3"
  local seed="$4"
  local groups
  if [[ "$fine_heads" == "0" ]]; then
    groups="[{name: coarse, num_heads: ${coarse_heads}, window_size: 4, stride: 2}]"
  elif [[ "$coarse_heads" == "0" ]]; then
    groups="[{name: fine, num_heads: ${fine_heads}, window_size: 2, stride: 1}]"
  else
    groups="[{name: fine, num_heads: ${fine_heads}, window_size: 2, stride: 1}, {name: coarse, num_heads: ${coarse_heads}, window_size: 4, stride: 2}]"
  fi
  local out="${OUT_ROOT}/${row}_seed${seed}_B${BATCH_SIZE}"
  skip_if_done "$out" && return 0
  python scripts/run_mqar.py \
    --config configs/mqar/primary_b25.yaml \
    --csv-path "${out}.csv" \
    --override \
      "stage=mqar_primary_registered" \
      "training.seed=${seed}" \
      "training.lr=${PRIMARY_LR}" \
      "training.max_updates=${PRIMARY_MAX_UPDATES}" \
      "training.checkpoint.save_final=${SAVE_FINAL}" \
      "training.output_dir=${out}" \
      "data.batch_size=${BATCH_SIZE}" \
      "data.seq_len=${SEQ_LEN}" \
      "data.num_kv_pairs=${NUM_KV_PAIRS}" \
      "data.num_train_examples=${NUM_TRAIN_EXAMPLES}" \
      "data.num_val_examples=${NUM_VAL_EXAMPLES}" \
      "data.mqar_row=${row}" \
      "model.multiresolution.groups=${groups}" \
    "${COMMON_ARGS[@]}"
}

for seed in $SEEDS; do
  for row in $ROWS; do
    case "$row" in
      token) run_token "$seed" ;;
      b0) run_set_row b0 8 0 "$seed" ;;
      b25) run_set_row b25 6 2 "$seed" ;;
      b50) run_set_row b50 4 4 "$seed" ;;
      b75) run_set_row b75 2 6 "$seed" ;;
      b100) run_set_row b100 0 8 "$seed" ;;
      *) echo "unknown MQAR row: $row" >&2; exit 2 ;;
    esac
  done
done
