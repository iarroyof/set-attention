#!/usr/bin/env bash
# A4.1 launch: SCP configs + run script to blue-demon and start the smoke (nohup, background).
set -euo pipefail

REPO="/mnt/d/UserFolders/Documents/GitHub/set-attention"
REMOTE="iarroyof@192.168.241.149"
REMOTE_REPO="~/set-attention"
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"

cd "$REPO"

echo "=== Step 1: SCP run_a41_smoke.sh to blue-demon ==="
$SSHPASS scp scripts/run_a41_smoke.sh \
    "$REMOTE:$REMOTE_REPO/scripts/run_a41_smoke.sh"
echo "SCP OK"

echo "=== Step 2: SCP a4_long_context configs to blue-demon ==="
$SSHPASS ssh "$REMOTE" "mkdir -p $REMOTE_REPO/configs/a4_long_context"
$SSHPASS scp configs/a4_long_context/baseline_dense_lc.yaml \
    "$REMOTE:$REMOTE_REPO/configs/a4_long_context/baseline_dense_lc.yaml"
$SSHPASS scp configs/a4_long_context/set_dense_lc.yaml \
    "$REMOTE:$REMOTE_REPO/configs/a4_long_context/set_dense_lc.yaml"
echo "Configs SCP OK"

echo "=== Step 3: Start A4.1 smoke on blue-demon (nohup, detached) ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && mkdir -p logs/a41_smoke out/paper_mechanisms/a41_smoke audit && \
     nohup bash scripts/run_a41_smoke.sh \
     > logs/a41_smoke/nohup_a41.log 2>&1 &
     echo \"PID: \$!\""
echo "Smoke launched on blue-demon. Check logs/a41_smoke/nohup_a41.log for progress."
echo "When complete, run run_a41_sync.bat to collect artifacts."
