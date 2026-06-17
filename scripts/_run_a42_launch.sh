#!/usr/bin/env bash
# A4.2 launch: SCP configs + run script to blue-demon and start the slice (nohup, background).
set -euo pipefail

REPO="/mnt/d/UserFolders/Documents/GitHub/set-attention"
REMOTE="iarroyof@192.168.241.149"
REMOTE_REPO="~/set-attention"
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"

cd "$REPO"

echo "=== Step 1: SCP run_a42_slice.sh to blue-demon ==="
$SSHPASS scp scripts/run_a42_slice.sh \
    "$REMOTE:$REMOTE_REPO/scripts/run_a42_slice.sh"
echo "SCP OK"

echo "=== Step 2: SCP a4_long_context configs to blue-demon ==="
$SSHPASS ssh "$REMOTE" "mkdir -p $REMOTE_REPO/configs/a4_long_context"
$SSHPASS scp configs/a4_long_context/baseline_dense_lc.yaml \
    "$REMOTE:$REMOTE_REPO/configs/a4_long_context/baseline_dense_lc.yaml"
$SSHPASS scp configs/a4_long_context/set_dense_lc.yaml \
    "$REMOTE:$REMOTE_REPO/configs/a4_long_context/set_dense_lc.yaml"
$SSHPASS scp configs/a4_long_context/set_sparse_lc.yaml \
    "$REMOTE:$REMOTE_REPO/configs/a4_long_context/set_sparse_lc.yaml"
$SSHPASS scp configs/a4_long_context/set_linear_lc.yaml \
    "$REMOTE:$REMOTE_REPO/configs/a4_long_context/set_linear_lc.yaml"
echo "Configs SCP OK"

echo "=== Step 3: Start A4.2 slice on blue-demon (nohup, detached) ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && mkdir -p logs/a42_slice out/paper_mechanisms/a42_slice audit && \
     nohup bash scripts/run_a42_slice.sh \
     > logs/a42_slice/nohup_a42.log 2>&1 &
     echo \"PID: \$!\""
echo "Slice launched on blue-demon. Check logs/a42_slice/nohup_a42.log for progress."
echo "When complete, run run_a42_sync.bat to collect artifacts."
