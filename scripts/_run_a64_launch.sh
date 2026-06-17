#!/usr/bin/env bash
# Local WSL launcher for A6.4.
# SCPs worker scripts and summarizer to blue-demon, then starts detached container workers.
set -euo pipefail

REMOTE="iarroyof@192.168.241.149"
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"
REPO="~/set-attention"

echo "=== [A6.4] Syncing worker scripts to blue-demon ==="
$SSHPASS scp \
  scripts/run_a64_depth_sweep_worker_inside_container.sh \
  scripts/run_a64_depth_sweep_detached.sh \
  "$REMOTE:${REPO}/scripts/"

echo "=== [A6.4] Syncing summarizer to blue-demon ==="
$SSHPASS scp \
  scripts/summarize_a64_depth_sweep.py \
  "$REMOTE:${REPO}/scripts/"

echo "=== [A6.4] Creating log dir on blue-demon ==="
$SSHPASS ssh "$REMOTE" "mkdir -p ${REPO}/logs/a64_depth_sweep"

echo "=== [A6.4] Launching detached workers on blue-demon ==="
$SSHPASS ssh "$REMOTE" "cd ${REPO} && bash scripts/run_a64_depth_sweep_detached.sh"

echo ""
echo "=== A6.4 launch complete ==="
echo "Workers running detached inside the container. Monitor with:"
echo "  $SSHPASS ssh $REMOTE 'tail -f ${REPO}/logs/a64_depth_sweep/container_worker_gpu0.log'"
echo "  $SSHPASS ssh $REMOTE 'tail -f ${REPO}/logs/a64_depth_sweep/container_worker_gpu1.log'"
