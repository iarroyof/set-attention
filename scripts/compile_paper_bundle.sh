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

if [[ -f "$ROOT/docs/benchmark_task_diagrams.tikz.tex" ]]; then
  bash "$ROOT/scripts/build_benchmark_task_diagrams.sh" "$ROOT"
fi

cd "$PAPER_DIR"

echo "Compiling: $PAPER_DIR/$MAIN_TEX"
echo "Build directory: $RUN_DIR"

# Compile in the source directory, as Overleaf does.  Using only -outdir here
# can make BibTeX consume a stale source-directory .aux file instead of the
# auxiliary file from the current run.
set +e
latexmk -pdf -interaction=nonstopmode -file-line-error \
  "$MAIN_TEX" | tee "$LOG_DIR/latexmk.stdout.txt"
EXIT_CODE=${PIPESTATUS[0]}
set -e

STEM="${MAIN_TEX%.tex}"
if [[ $EXIT_CODE -eq 0 && -f "$PAPER_DIR/$STEM.pdf" ]]; then
  cp "$PAPER_DIR/$STEM.pdf" "$RUN_DIR/$STEM.pdf"
  if [[ -f "$PAPER_DIR/$STEM.log" ]]; then
    cp "$PAPER_DIR/$STEM.log" "$RUN_DIR/$STEM.log"
  fi
fi

printf '%s\n' "$RUN_DIR" > "$LATEST_RUN_FILE"
printf '%s\n' "$MAIN_TEX" > "$LATEST_MAIN_FILE"

echo
echo "==== LAST 80 LINES OF BUILD LOG ===="
tail -n 80 "$LOG_DIR/latexmk.stdout.txt" || true

echo
echo "==== ERROR/WARNING GREP ===="
FINAL_LOG="$PAPER_DIR/$STEM.log"
if [[ -f "$FINAL_LOG" ]]; then
  grep -nE "(^!|LaTeX Error|Undefined control sequence|Emergency stop|Warning|Citation.*undefined|Reference.*undefined|Overfull|Underfull)" \
    "$FINAL_LOG" || true
else
  echo "Final TeX log not found: $FINAL_LOG"
fi

if [[ $EXIT_CODE -ne 0 ]]; then
  echo
  echo "BUILD FAILED"
  exit $EXIT_CODE
fi

PDF_PATH="$RUN_DIR/$STEM.pdf"
if [[ -f "$PDF_PATH" ]]; then
  echo
  echo "BUILD SUCCEEDED"
  echo "PDF: $PDF_PATH"
else
  echo
  echo "BUILD DID NOT PRODUCE PDF"
  exit 1
fi
