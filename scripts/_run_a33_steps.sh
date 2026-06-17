#!/usr/bin/env bash
# A3.3 post-run steps: scp summarizer to blue-demon, run it, sync artifacts back.
# Run this AFTER run_a3_stride_sweep.sh has completed on blue-demon.
set -euo pipefail

REPO="/mnt/d/UserFolders/Documents/GitHub/set-attention"
REMOTE="iarroyof@192.168.241.149"
REMOTE_REPO="~/set-attention"
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"

cd "$REPO"

echo "=== Step 1: SCP patched summarizer to blue-demon ==="
$SSHPASS scp scripts/summarize_a3_stride_sweep.py \
    "$REMOTE:$REMOTE_REPO/scripts/summarize_a3_stride_sweep.py"
echo "SCP OK"

echo "=== Step 2: Run summarizer on blue-demon inside container ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && docker compose exec -T set-attention \
     python scripts/summarize_a3_stride_sweep.py"
echo "Summarizer OK"

echo "=== Step 3: Sync artifacts back locally ==="
for f in \
    "out/paper_integrated_evidence/tables/a3_stride_sweep_all_runs.tsv" \
    "out/paper_integrated_evidence/tables/a3_stride_sweep_summary.tsv" \
    "out/paper_integrated_evidence/checks/a3_stride_sweep_manifest.json" \
    "audit/A3_3_stride_sweep.md"; do
    mkdir -p "$REPO/$(dirname $f)"
    $SSHPASS scp "$REMOTE:$REMOTE_REPO/$f" "$REPO/$f"
    echo "  synced: $f"
done

echo "=== Step 4: Verify manifest ==="
python3 -c "
import json, sys
m = json.load(open('$REPO/out/paper_integrated_evidence/checks/a3_stride_sweep_manifest.json'))
print('status:', m['status'])
print('validated_runs:', m['validated_runs'], '/ expected:', m['expected_runs'])
if m['failures']:
    print('FAILURES:')
    for f in m['failures']: print(' -', f)
    sys.exit(1)
print('ALL CLEAR')
"
