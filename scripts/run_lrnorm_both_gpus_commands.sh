#!/usr/bin/env bash
set -euo pipefail

# Run on blue-demon from ~/set-attention.
# Launches the A2 LR-normalized grid on GPU 0 and the A2 anchor/family rows on GPU 1.
cd ~/set-attention
LOG_ROOT="${LOG_ROOT:-logs/a2_grid}"
mkdir -p "${LOG_ROOT}"

bash scripts/gpu0_run_lrnorm_headline_pairs.sh > "${LOG_ROOT}/gpu0_lrnorm_headline_pairs.nohup.log" 2>&1 &
PID0=$!
echo "GPU0 headline LR-normalization PID: ${PID0}"

bash scripts/gpu1_run_lrnorm_family_anchor.sh > "${LOG_ROOT}/gpu1_lrnorm_family_anchor.nohup.log" 2>&1 &
PID1=$!
echo "GPU1 family/anchor LR-normalization PID: ${PID1}"

printf "%s\n" "${PID0}" > "${LOG_ROOT}/gpu0.pid"
printf "%s\n" "${PID1}" > "${LOG_ROOT}/gpu1.pid"

set +e
REMAINING=2
while [[ "${REMAINING}" -gt 0 ]]; do
  wait -n
  RC=$?
  if [[ "${RC}" -ne 0 ]]; then
    echo "An A2 grid worker failed with exit code ${RC}; stopping remaining workers." >&2
    for PID in "${PID0}" "${PID1}"; do
      if kill -0 "${PID}" 2>/dev/null; then
        kill "${PID}" 2>/dev/null || true
      fi
    done
    exit "${RC}"
  fi
  REMAINING=$((REMAINING - 1))
done

exit 0
