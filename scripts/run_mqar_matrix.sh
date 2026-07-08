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

This wrapper is approval-gated. To run the registered matrix, both are required:
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

run_token() {
  local seed="$1"
  python scripts/run_mqar.py \
    --config configs/mqar/primary_token.yaml \
    --override "training.seed=${seed}" "data.batch_size=${MQAR_BATCH_SIZE:-1}" \
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
  python scripts/run_mqar.py \
    --config configs/mqar/primary_b25.yaml \
    --override \
      "training.seed=${seed}" \
      "data.batch_size=${MQAR_BATCH_SIZE:-1}" \
      "data.mqar_row=${row}" \
      "model.multiresolution.groups=${groups}" \
    "${COMMON_ARGS[@]}"
}

for seed in 0 1 2; do
  run_token "$seed"
  run_set_row b0 8 0 "$seed"
  run_set_row b25 6 2 "$seed"
  run_set_row b50 4 4 "$seed"
  run_set_row b75 2 6 "$seed"
  run_set_row b100 0 8 "$seed"
done
