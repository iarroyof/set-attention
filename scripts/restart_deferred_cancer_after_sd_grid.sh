#!/usr/bin/env bash
set -euo pipefail

GRID_PID="${1:?usage: $0 GRID_PID}"
CONTAINER="${CONTAINER:-cancer_rl_agent}"
RESTORE_NAME="${RESTORE_NAME:-cancer_rl_agent}"
ENV_FILE="${ENV_FILE:-$HOME/.local/state/gpu-handoff/cancer_rl_agent_semantic_eval.env}"
LOG="${LOG:-$HOME/.local/state/gpu-handoff/restart_after_sd_grid.log}"
INTERVAL="${INTERVAL:-120}"
WORKDIR="/app"
WORKLOAD="semantic_triplet_verification_ppo_rag_eval_v3_abstention_raw_triplets_with_explanations.py"
OUTPUT="semantic_triplet_verification_ppo_rag_eval_v3_abstention_raw_triplets_with_explanations.out"

mkdir -p "$(dirname "$LOG")"
exec 9>"${LOG}.lock"
flock -n 9 || {
  printf '[%s] another GPU handoff watcher is active\n' "$(date '+%F %T')" >> "$LOG"
  exit 0
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"
}

grid_driver_alive() {
  kill -0 "$GRID_PID" 2>/dev/null &&
    ps -o args= -p "$GRID_PID" 2>/dev/null | grep -q '^bash scripts/run_sd_grid[.]sh$'
}

grid_containers_alive() {
  docker ps --filter 'name=^sdgrid_lizmark_' --format '{{.Names}}' | grep -q .
}

cuda_processes_alive() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'
}

log "waiting for set-attention grid PID ${GRID_PID} to release both GPUs"
while grid_driver_alive; do
  sleep "$INTERVAL"
done
while grid_containers_alive || cuda_processes_alive; do
  log "grid driver exited but a GPU process/container remains; waiting"
  sleep "$INTERVAL"
done

[ -f "$ENV_FILE" ] || {
  log "missing preserved environment: $ENV_FILE"
  exit 2
}

if [ "$CONTAINER" != "$RESTORE_NAME" ]; then
  if docker inspect "$RESTORE_NAME" >/dev/null 2>&1; then
    log "cannot restore container name; $RESTORE_NAME already exists"
    exit 3
  fi
  docker rename "$CONTAINER" "$RESTORE_NAME"
  CONTAINER="$RESTORE_NAME"
fi

state="$(docker inspect "$CONTAINER" --format '{{.State.Status}}' 2>/dev/null || true)"
if [ "$state" != running ]; then
  docker start "$CONTAINER" >> "$LOG" 2>&1
fi

if docker exec "$CONTAINER" pgrep -f "^python ${WORKLOAD}$" >/dev/null 2>&1; then
  log "deferred workload is already running; no duplicate launched"
  exit 0
fi

docker exec -d \
  --env-file "$ENV_FILE" \
  -w "$WORKDIR" \
  "$CONTAINER" \
  sh -lc "exec /usr/bin/python '$WORKLOAD' > '$OUTPUT' 2>&1"

sleep 10
if ! docker exec "$CONTAINER" pgrep -f "^/usr/bin/python ${WORKLOAD}$|^python ${WORKLOAD}$" >/dev/null 2>&1; then
  log "container restarted but deferred workload did not remain active"
  exit 4
fi

log "deferred container and workload restarted after GPU release"
