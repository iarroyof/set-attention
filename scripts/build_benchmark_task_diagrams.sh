#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
SOURCE="$ROOT/docs/benchmark_task_diagrams.tikz.tex"
DOCS_PDF="$ROOT/docs/benchmark_task_diagrams.tikz.pdf"
PAPER_DIR="$ROOT/out/final_paper_bundle/overleaf_ready"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

mkdir -p "$PAPER_DIR"

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory "$BUILD_DIR" "$SOURCE"

cp "$BUILD_DIR/benchmark_task_diagrams.tikz.pdf" "$DOCS_PDF"
cp "$DOCS_PDF" "$PAPER_DIR/benchmark_task_diagrams.tikz.pdf"
cp "$SOURCE" "$PAPER_DIR/benchmark_task_diagrams.tikz.tex"

echo "Built: $DOCS_PDF"
echo "Copied PDF and TikZ source to: $PAPER_DIR"
