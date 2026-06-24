#!/usr/bin/env bash
# SD-8 full launch: sync all_past dense-router files to blue-demon and start the 6-run ladder.
set -euo pipefail

REPO="/mnt/d/UserFolders/Documents/GitHub/set-attention"
REMOTE="iarroyof@192.168.241.149"
REMOTE_REPO="~/set-attention"
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"

cd "$REPO"

echo "=== Step 1: Prepare remote directories ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && mkdir -p src/models/set_only src/config configs/set_dictionary scripts logs/sd8_all_past_dense_routerdense out/paper_mechanisms/sd8_all_past_dense_routerdense audit out/paper_integrated_evidence/tables out/paper_integrated_evidence/checks"

echo "=== Step 2: SCP SD-8 implementation/config/scripts ==="
$SSHPASS scp \
    src/models/set_only/banks.py \
    src/models/set_only/set_only_lm.py \
    src/config/compatibility.py \
    "$REMOTE:$REMOTE_REPO/"
$SSHPASS ssh "$REMOTE" "cd $REMOTE_REPO && mv banks.py src/models/set_only/banks.py && mv set_only_lm.py src/models/set_only/set_only_lm.py && mv compatibility.py src/config/compatibility.py"
$SSHPASS scp \
    configs/set_dictionary/sd8_all_past_dense.yaml \
    configs/set_dictionary/sd8_all_past_dense_smoke.yaml \
    "$REMOTE:$REMOTE_REPO/configs/set_dictionary/"
$SSHPASS scp \
    scripts/run_sd8_all_past_dense.sh \
    scripts/summarize_sd8_all_past_dense.py \
    "$REMOTE:$REMOTE_REPO/scripts/"

echo "=== Step 3: Start SD-8 all_past dense-router full ladder on blue-demon (nohup, detached) ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && nohup env SMOKE=0 bash scripts/run_sd8_all_past_dense.sh > logs/sd8_all_past_dense_routerdense/nohup_sd8_full.log 2>&1 < /dev/null & echo \"PID: \$!\""

echo "Full ladder launched. Check logs/sd8_all_past_dense_routerdense/nohup_sd8_full.log on blue-demon."
echo "When complete, run run_sd8_full_sync.bat or scripts/_run_sd8_full_steps.sh."
