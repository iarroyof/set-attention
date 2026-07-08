#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$HOME/set-attention}"
PATCH_TAR="${PATCH_TAR:-/tmp/set_attention_diag_patch_lizmark.tar}"
INTERVAL="${INTERVAL:-300}"
MAX_GPU_MEMORY_MIB="${MAX_GPU_MEMORY_MIB:-4000}"
LOG="${LOG:-/tmp/defer_sd_grid_seeded_v1_lizmark.log}"
LOCK="/tmp/defer_sd_grid_seeded_v1_lizmark.lock"

exec 9>"$LOCK"
flock -n 9 || {
  printf '[%s] another deferred deployer holds %s; exiting\n' "$(date '+%F %T')" "$LOCK" >> "$LOG"
  exit 0
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"
}

active_training() {
  pgrep -f '^bash scripts/run_sd_grid[.]sh$|/usr/bin/python scripts/run_experiment[.]py' >/dev/null 2>&1
}

gpu_memory_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{sum += $1} END {print sum + 0}'
}

log "deferred corrected lizmark deployment enqueued"
while active_training; do
  log "legacy grid/training process remains; waiting ${INTERVAL}s"
  sleep "$INTERVAL"
done

while [ "$(gpu_memory_used)" -gt "$MAX_GPU_MEMORY_MIB" ]; do
  log "GPUs remain busy; waiting ${INTERVAL}s"
  sleep "$INTERVAL"
done

[ -f "$PATCH_TAR" ] || {
  log "missing patch archive $PATCH_TAR"
  exit 2
}

cd "$REPO"
log "host idle; beginning registry cleanup and deployment"

if [ -f out/paper_mechanisms/sd_grid/grid_runs_lizmark.tsv ]; then
  staging="$(mktemp -d /tmp/sd_grid_cleanup.XXXXXX)"
  tar -xf "$PATCH_TAR" -C "$staging" scripts/archive_sd_grid_duplicate_records.py
  python3 "$staging/scripts/archive_sd_grid_duplicate_records.py" \
    out/paper_mechanisms/sd_grid/grid_runs_lizmark.tsv \
    out/_archive/duplicate_launch_records_20260701_lizmark \
    >> "$LOG" 2>&1
fi

tar -xf "$PATCH_TAR" -C "$REPO"
chmod +x \
  scripts/run_sd_grid.sh \
  scripts/archive_sd_grid_duplicate_records.py

docker run --rm -u "$(id -u):$(id -g)" \
  -e HOME=/workspace \
  -v "$REPO:/workspace" -w /workspace set-attention:latest \
  /usr/bin/python -c \
  'import runpy; a=runpy.run_path("tests/test_multiresolution_diagnostics.py"); a["test_multiresolution_training_diagnostics_are_grouped_and_complete"](); a["test_multiresolution_eval_probes_are_grouped"](); b=runpy.run_path("tests/test_run_experiment_seed.py"); [b[n]() for n in ("test_same_seed_reproduces_initial_parameters","test_different_seed_changes_initial_parameters","test_applied_seed_provenance_is_explicit","test_missing_seed_fails_closed")]' \
  >> "$LOG" 2>&1

env \
  HOST_TAG=lizmark \
  GRID_PROFILE=paper5 \
  SEEDS="0 1 2 3 4" \
  GRID_NAMESPACE=sd_grid_seeded_v1 \
  RUN_TAG=seeded_v1 \
  REQUIRE_APPLIED_SEED=1 \
  TRAINING_DETERMINISTIC=true \
  DRY_RUN=1 \
  REPO_ROOT="$REPO" \
  bash scripts/run_sd_grid.sh > /tmp/sd_grid_lizmark_seeded_v1_strict_plan.txt

plans="$(grep -c '^PLAN  run' /tmp/sd_grid_lizmark_seeded_v1_strict_plan.txt || true)"
skips="$(grep -c '^SKIP' /tmp/sd_grid_lizmark_seeded_v1_strict_plan.txt || true)"
if [ "$plans" -ne 135 ] || [ "$skips" -ne 0 ]; then
  log "strict dry plan mismatch: plans=$plans skips=$skips; launch blocked"
  exit 3
fi
log "strict dry plan passed: plans=135 skips=0"

setsid -f env \
  HOST_TAG=lizmark \
  GRID_PROFILE=paper5 \
  SEEDS="0 1 2 3 4" \
  GRID_NAMESPACE=sd_grid_seeded_v1 \
  RUN_TAG=seeded_v1 \
  REQUIRE_APPLIED_SEED=1 \
  TRAINING_DETERMINISTIC=true \
  GPU0=0 \
  GPU1=1 \
  REPO_ROOT="$REPO" \
  bash scripts/run_sd_grid.sh \
  > logs/sd_grid_lizmark_paper5_seeded_v1.log 2>&1 < /dev/null

sleep 15
driver_count="$(pgrep -fc '^bash scripts/run_sd_grid[.]sh$' || true)"
failure_count="$(
  { grep -h '=== FAIL\|=== FATAL CONFIG' \
      logs/sd_grid_seeded_v1/lizmark/worker_gpu0.log \
      logs/sd_grid_seeded_v1/lizmark/worker_gpu1.log 2>/dev/null || true; } \
    | wc -l
)"
if [ "$driver_count" -lt 1 ] || [ "$failure_count" -ne 0 ]; then
  log "post-launch health check failed: drivers=$driver_count failures=$failure_count"
  exit 4
fi

log "corrected queue launched; drivers=$driver_count failures=0"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader >> "$LOG" 2>&1
tail -n 5 logs/sd_grid_lizmark_paper5_seeded_v1.log >> "$LOG" 2>&1
log "deferred deployer complete; stop-polling policy active"
