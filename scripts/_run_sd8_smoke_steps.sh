#!/usr/bin/env bash
# SD-8 smoke post-run steps: summarize smoke and sync artifacts back.
set -euo pipefail

REPO="/mnt/d/UserFolders/Documents/GitHub/set-attention"
REMOTE="iarroyof@192.168.241.149"
REMOTE_REPO="~/set-attention"
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"

cd "$REPO"

echo "=== Step 1: SCP summarizer to blue-demon ==="
$SSHPASS scp scripts/summarize_sd8_all_past_dense.py \
    "$REMOTE:$REMOTE_REPO/scripts/summarize_sd8_all_past_dense.py"

echo "=== Step 2: Run smoke summarizer in container ==="
$SSHPASS ssh "$REMOTE" \
    "cd $REMOTE_REPO && docker compose exec -T set-attention python scripts/summarize_sd8_all_past_dense.py --mode smoke"

echo "=== Step 3: Sync smoke artifacts back locally ==="
for f in \
    "out/paper_integrated_evidence/tables/sd8_all_past_dense_routerdense_smoke_runs.tsv" \
    "out/paper_integrated_evidence/tables/sd8_all_past_dense_routerdense_smoke_summary.tsv" \
    "out/paper_integrated_evidence/checks/sd8_all_past_dense_routerdense_smoke_manifest.json" \
    "audit/SD_8_all_past_dense_routerdense_smoke.md" \
    "audit/SD_8_all_past_dense_routerdense_prelaunch_smoke.json"; do
    mkdir -p "$REPO/$(dirname "$f")"
    $SSHPASS scp "$REMOTE:$REMOTE_REPO/$f" "$REPO/$f"
    echo "  synced: $f"
done

echo "=== Step 4: Verify smoke manifest ==="
python3 -c "
import json, sys
m = json.load(open('$REPO/out/paper_integrated_evidence/checks/sd8_all_past_dense_routerdense_smoke_manifest.json'))
print('status:', m['status'])
print('validated_runs:', m['validated_runs'], '/ expected:', m['expected_runs'])
if m['status'] != 'pass' or m['validated_runs'] != m['expected_runs']:
    sys.exit(1)
print('SD-8 SMOKE PASS')
"
