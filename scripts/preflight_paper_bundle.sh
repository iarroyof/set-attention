#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
PAPER_DIR="$ROOT/out/final_paper_bundle/overleaf_ready"
MAIN="$PAPER_DIR/example_paper.tex"
NEURIPS_STY="$PAPER_DIR/neurips_2025.sty"
CHECKLIST="$PAPER_DIR/neurips_checklist.tex"

echo "Checking bundle preflight in: $PAPER_DIR"

[[ -f "$MAIN" ]] || { echo "Missing main TeX: $MAIN"; exit 1; }
[[ -f "$NEURIPS_STY" ]] || { echo "Missing NeurIPS style file: $NEURIPS_STY"; exit 1; }
[[ -f "$CHECKLIST" ]] || { echo "Missing NeurIPS checklist scaffold: $CHECKLIST"; exit 1; }

echo "Referenced style/class/bib commands:"
grep -nE '\\usepackage|\\bibliographystyle|\\bibliography|\\input|\\includegraphics' "$MAIN" || true

echo
echo "NeurIPS-specific checks:"
grep -n '\\usepackage.*{neurips_2025}' "$MAIN" || { echo "Missing neurips_2025 package use in main TeX"; exit 1; }
grep -n '\\section\*{NeurIPS Paper Checklist}' "$CHECKLIST" || { echo "Checklist scaffold does not define the required NeurIPS checklist heading"; exit 1; }

echo
echo "Local files in overleaf_ready:"
find "$PAPER_DIR" -maxdepth 1 -type f | sort

echo
echo "Preflight compile..."
"$ROOT/scripts/compile_paper_bundle.sh" "$ROOT"
