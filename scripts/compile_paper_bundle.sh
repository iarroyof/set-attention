#!/usr/bin/env bash
# scripts/compile_paper_bundle.sh

set -euo pipefail

ROOT="${1:-$(pwd)}"
PAPER_DIR="$ROOT/out/final_paper_bundle/overleaf_ready"
MAIN_TEX="${2:-example_paper.tex}"
LOG_DIR="$ROOT/out/final_paper_bundle/checks/compile_logs"
mkdir -p "$LOG_DIR"

cd "$PAPER_DIR"

echo "Compiling: $PAPER_DIR/$MAIN_TEX"

latexmk -pdf -interaction=nonstopmode -file-line-error \
  -outdir="$LOG_DIR" \
  "$MAIN_TEX" | tee "$LOG_DIR/latexmk.stdout.txt"

EXIT_CODE=${PIPESTATUS[0]}

echo
echo "==== LAST 80 LINES OF BUILD LOG ===="
tail -n 80 "$LOG_DIR/latexmk.stdout.txt" || true

echo
echo "==== ERROR/WARNING GREP ===="
grep -nE "(^!|LaTeX Error|Undefined control sequence|Emergency stop|Warning|Citation.*undefined|Reference.*undefined|Overfull|Underfull)" \
  "$LOG_DIR"/latexmk.stdout.txt || true

if [[ $EXIT_CODE -ne 0 ]]; then
  echo
  echo "BUILD FAILED"
  exit $EXIT_CODE
fi

PDF_PATH="$LOG_DIR/${MAIN_TEX%.tex}.pdf"
if [[ -f "$PDF_PATH" ]]; then
  echo
  echo "BUILD SUCCEEDED"
  echo "PDF: $PDF_PATH"
else
  echo
  echo "BUILD DID NOT PRODUCE PDF"
  exit 1
fi
