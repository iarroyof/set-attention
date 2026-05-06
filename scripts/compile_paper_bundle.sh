#!/usr/bin/env bash
# scripts/compile_paper_bundle.sh

set -euo pipefail

ROOT="${1:-$(pwd)}"
PAPER_DIR="$ROOT/out/final_paper_bundle/overleaf_ready"
MAIN_TEX="${2:-example_paper.tex}"
LOG_DIR="$ROOT/out/final_paper_bundle/checks/compile_logs"
mkdir -p "$LOG_DIR"
RUN_DIR="$(mktemp -d "$LOG_DIR/run_XXXXXX")"
LATEST_RUN_FILE="$LOG_DIR/latest_run_dir.txt"
LATEST_MAIN_FILE="$LOG_DIR/latest_main_tex.txt"

cd "$PAPER_DIR"

echo "Compiling: $PAPER_DIR/$MAIN_TEX"
echo "Build directory: $RUN_DIR"

set +e
latexmk -pdf -interaction=nonstopmode -file-line-error \
  -outdir="$RUN_DIR" \
  "$MAIN_TEX" | tee "$LOG_DIR/latexmk.stdout.txt"
EXIT_CODE=${PIPESTATUS[0]}
set -e

printf '%s\n' "$RUN_DIR" > "$LATEST_RUN_FILE"
printf '%s\n' "$MAIN_TEX" > "$LATEST_MAIN_FILE"

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

PDF_PATH="$RUN_DIR/${MAIN_TEX%.tex}.pdf"
if [[ -f "$PDF_PATH" ]]; then
  echo
  echo "BUILD SUCCEEDED"
  echo "PDF: $PDF_PATH"
else
  echo
  echo "BUILD DID NOT PRODUCE PDF"
  exit 1
fi
