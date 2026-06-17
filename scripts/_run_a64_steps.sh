#!/usr/bin/env bash
# Local WSL post-run sync for A6.4.
# Runs the summarizer inside the container on blue-demon, then syncs 3 artifacts + audit doc.
set -euo pipefail

REMOTE="iarroyof@192.168.241.149"
SSHPASS="sshpass -f $HOME/.ssh/.sshpass"
REPO="~/set-attention"
LOCAL_EVIDENCE="out/paper_integrated_evidence"

echo "=== [A6.4] Running summarizer inside container on blue-demon ==="
$SSHPASS ssh "$REMOTE" \
  "docker exec set-attention bash -lc 'cd /workspace && python scripts/summarize_a64_depth_sweep.py'"

echo "=== [A6.4] Syncing artifacts from blue-demon ==="
mkdir -p "${LOCAL_EVIDENCE}/tables" "${LOCAL_EVIDENCE}/checks"

$SSHPASS scp \
  "$REMOTE:${REPO}/out/paper_integrated_evidence/tables/a64_depth_sweep_all_runs.tsv" \
  "${LOCAL_EVIDENCE}/tables/"

$SSHPASS scp \
  "$REMOTE:${REPO}/out/paper_integrated_evidence/tables/a64_depth_sweep_summary.tsv" \
  "${LOCAL_EVIDENCE}/tables/"

$SSHPASS scp \
  "$REMOTE:${REPO}/out/paper_integrated_evidence/checks/a64_depth_sweep_manifest.json" \
  "${LOCAL_EVIDENCE}/checks/"

echo "=== [A6.4] Syncing audit doc from blue-demon ==="
$SSHPASS scp \
  "$REMOTE:${REPO}/audit/A6_4_depth_sweep.md" \
  "audit/"

echo ""
echo "=== A6.4 sync complete ==="
python3 - <<'PY'
import json, sys
from pathlib import Path
p = Path("out/paper_integrated_evidence/checks/a64_depth_sweep_manifest.json")
if not p.exists():
    print("manifest not found", file=sys.stderr)
    sys.exit(1)
m = json.loads(p.read_text())
print(f"status           : {m['status']}")
print(f"validated_runs   : {m['validated_runs']} / {m['expected_runs']}")
print(f"validated_new    : {m['validated_new_runs']} / {m['expected_new_runs']}")
print(f"reused_rows      : {m['reused_rows']}")
if m.get("failures"):
    print("FAILURES:")
    for f in m["failures"][:10]:
        print(f"  {f}")
PY
