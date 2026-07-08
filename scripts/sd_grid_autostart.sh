#!/usr/bin/env bash
# Host-local watcher: wait until the legacy SD-9.6/9.7 queues are gone AND the GPUs are idle,
# then auto-start run_sd_grid.sh ONCE on this host. Idempotent and safe to leave running:
#  * never starts a 2nd grid driver (skips if run_sd_grid.sh already running),
#  * never starts onto busy GPUs (avoids OOM / contention with draining legacy workers),
#  * one-shot: starts the grid then exits (re-run the watcher later for a fresh wave).
# Deploy:  HOST_TAG=blue    nohup bash scripts/sd_grid_autostart.sh >/dev/null 2>&1 &
#          HOST_TAG=lizmark nohup bash scripts/sd_grid_autostart.sh >/dev/null 2>&1 &
set -u
cd "${REPO_ROOT:-$HOME/set-attention}" 2>/dev/null || true   # caller already cd's into the repo
HOST_TAG="${HOST_TAG:?set HOST_TAG=blue|lizmark}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
INTERVAL="${INTERVAL:-300}"
MAXMEM_IDLE="${MAXMEM_IDLE:-4000}"   # summed GPU MiB below which the host counts as "free"
mkdir -p logs
LOG="logs/sd_grid_autostart_${HOST_TAG}.log"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# robust self-dedup: only one watcher per host (flock, not pgrep — pgrep -f self-matches)
exec 9>"logs/.sd_grid_autostart_${HOST_TAG}.lock"
if command -v flock >/dev/null 2>&1; then
  flock -n 9 || { log "another watcher holds the lock; exiting"; exit 0; }
fi

log "watcher start tag=$HOST_TAG interval=${INTERVAL}s maxmem_idle=${MAXMEM_IDLE}MiB"
while true; do
  if pgrep -f 'run_sd_grid.sh' >/dev/null 2>&1; then
    log "grid driver already running -> nothing to do, watcher exits"; break
  fi
  a=$(pgrep -fc 'run_sd9_6_blue_long_multires_queue.sh' 2>/dev/null); a=${a:-0}
  b=$(pgrep -fc 'run_sd9_7_token_baseline.sh' 2>/dev/null); b=${b:-0}
  legacy=$((a + b))
  if [ "${legacy:-0}" -gt 0 ]; then
    log "legacy queue still running (${legacy} proc); wait ${INTERVAL}s"; sleep "$INTERVAL"; continue
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  if [ "${used:-999999}" -gt "$MAXMEM_IDLE" ]; then
    log "no legacy launcher but GPUs busy (${used}MiB > ${MAXMEM_IDLE}); wait ${INTERVAL}s"; sleep "$INTERVAL"; continue
  fi
  log "CONDITIONS MET (no legacy, GPUs idle ${used}MiB) -> starting grid"
  HOST_TAG="$HOST_TAG" GPU0="$GPU0" GPU1="$GPU1" nohup bash scripts/run_sd_grid.sh \
    > "logs/sd_grid_${HOST_TAG}.log" 2>&1 &
  log "grid launched (pid $!); watcher exits"
  break
done
