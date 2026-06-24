#!/usr/bin/env bash
# SD-8 smoke launch: sync all_past dense-router implementation to blue-demon and start detached smoke.
set -euo pipefail

REPO="/mnt/d/UserFolders/Documents/GitHub/set-attention"
REMOTE="iarroyof@192.168.241.149"
REMOTE_REPO="~/set-attention"
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"

cd "$REPO"

echo "=== Step 1: Prepare remote directories ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && mkdir -p src/models/set_only src/config tests configs/set_dictionary scripts logs/sd8_all_past_dense_routerdense_smoke out/paper_mechanisms/sd8_all_past_dense_routerdense_smoke audit out/paper_integrated_evidence/tables out/paper_integrated_evidence/checks"

echo "=== Step 2: SCP SD-8 implementation and tests ==="
$SSHPASS scp \
    src/models/set_only/banks.py \
    src/models/set_only/set_only_lm.py \
    src/config/compatibility.py \
    tests/test_banks_option1.py \
    tests/test_set_dictionary_causality.py \
    tests/test_output_residual_mode.py \
    "$REMOTE:$REMOTE_REPO/"
$SSHPASS ssh "$REMOTE" "cd $REMOTE_REPO && mv banks.py src/models/set_only/banks.py && mv set_only_lm.py src/models/set_only/set_only_lm.py && mv compatibility.py src/config/compatibility.py && mv test_banks_option1.py tests/test_banks_option1.py && mv test_set_dictionary_causality.py tests/test_set_dictionary_causality.py && mv test_output_residual_mode.py tests/test_output_residual_mode.py"

echo "=== Step 3: SCP SD-8 configs and scripts ==="
$SSHPASS scp \
    configs/set_dictionary/sd8_all_past_dense.yaml \
    configs/set_dictionary/sd8_all_past_dense_smoke.yaml \
    "$REMOTE:$REMOTE_REPO/configs/set_dictionary/"
$SSHPASS scp \
    scripts/run_sd8_all_past_dense.sh \
    scripts/summarize_sd8_all_past_dense.py \
    "$REMOTE:$REMOTE_REPO/scripts/"

echo "=== Step 4: Run focused container checks ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && docker compose exec -T set-attention python -m py_compile src/models/set_only/banks.py src/models/set_only/set_only_lm.py src/config/compatibility.py tests/test_banks_option1.py tests/test_set_dictionary_causality.py tests/test_output_residual_mode.py && docker compose exec -T set-attention python tests/test_set_dictionary_causality.py"

echo "=== Step 5: Start SD-8 all_past dense-router smoke on blue-demon (nohup, detached) ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && nohup env SMOKE=1 bash scripts/run_sd8_all_past_dense.sh > logs/sd8_all_past_dense_routerdense_smoke/nohup_sd8_smoke.log 2>&1 < /dev/null & echo \"PID: \$!\""

echo "Smoke launched. Check logs/sd8_all_past_dense_routerdense_smoke/nohup_sd8_smoke.log on blue-demon."
echo "When complete, run run_sd8_smoke_sync.bat or scripts/_run_sd8_smoke_steps.sh."
