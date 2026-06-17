#!/usr/bin/env bash
# A3.3 launch: SCP run script to blue-demon and start the sweep (nohup, background).
set -euo pipefail

REPO="/mnt/d/UserFolders/Documents/GitHub/set-attention"
REMOTE="iarroyof@192.168.241.149"
REMOTE_REPO="~/set-attention"
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"

cd "$REPO"

echo "=== Step 1: SCP run_a3_stride_sweep.sh to blue-demon ==="
$SSHPASS scp scripts/run_a3_stride_sweep.sh \
    "$REMOTE:$REMOTE_REPO/scripts/run_a3_stride_sweep.sh"
echo "SCP OK"

echo "=== Step 2: Start A3.3 sweep on blue-demon (nohup, detached) ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && mkdir -p logs/a3_stride_sweep out/paper_mechanisms/a3_stride_sweep audit && \
     nohup bash scripts/run_a3_stride_sweep.sh \
     > logs/a3_stride_sweep/nohup_a33.log 2>&1 &
     echo \"PID: \$!\""
echo "Sweep launched on blue-demon. Check logs/a3_stride_sweep/nohup_a33.log for progress."
echo "When complete, run run_a33_sync.bat to collect artifacts."
