#!/usr/bin/env bash
# SD-8 full post-run steps: summarize full ladder and sync artifacts back.
set -euo pipefail

REPO="/mnt/d/UserFolders/Documents/GitHub/set-attention"
REMOTE="iarroyof@192.168.241.149"
REMOTE_REPO="~/set-attention"
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"

cd "$REPO"

echo "=== Step 1: SCP summarizer to blue-demon ==="
$SSHPASS scp scripts/summarize_sd8_all_past_dense.py \
    "$REMOTE:$REMOTE_REPO/scripts/summarize_sd8_all_past_dense.py"

echo "=== Step 2: Run full summarizer in container ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && docker compose exec -T set-attention python scripts/summarize_sd8_all_past_dense.py --mode full"

echo "=== Step 3: Sync full artifacts back locally ==="
for f in \
    "out/paper_integrated_evidence/tables/sd8_all_past_dense_routerdense_runs.tsv" \
    "out/paper_integrated_evidence/tables/sd8_all_past_dense_routerdense_summary.tsv" \
    "out/paper_integrated_evidence/checks/sd8_all_past_dense_routerdense_manifest.json" \
    "audit/SD_8_all_past_dense_routerdense.md" \
    "audit/SD_8_all_past_dense_routerdense_prelaunch.json"; do
    mkdir -p "$REPO/$(dirname "$f")"
    $SSHPASS scp "$REMOTE:$REMOTE_REPO/$f" "$REPO/$f"
    echo "  synced: $f"
done

echo "=== Step 4: Verify full manifest ==="
python3 -c "
import json, sys
m = json.load(open('$REPO/out/paper_integrated_evidence/checks/sd8_all_past_dense_routerdense_manifest.json'))
print('status:', m['status'])
print('validated_runs:', m['validated_runs'], '/ expected:', m['expected_runs'])
if m['status'] != 'pass' or m['validated_runs'] != m['expected_runs']:
    sys.exit(1)
print('SD-8 FULL PASS')
"
