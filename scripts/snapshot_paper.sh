#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$(pwd)}"
SRC="$ROOT/out/final_paper_bundle/overleaf_ready/example_paper.tex"
DSTDIR="$ROOT/out/final_paper_bundle/checks/snapshots"
mkdir -p "$DSTDIR"
TS="$(date +%Y%m%d_%H%M%S)"
cp "$SRC" "$DSTDIR/example_paper_$TS.tex"
echo "Snapshot: $DSTDIR/example_paper_$TS.tex"
