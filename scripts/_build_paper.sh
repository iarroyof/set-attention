#!/usr/bin/env bash
# Local WSL script: build example_paper_working_agent.tex with pdflatex.
# Gathers tex, sty, bib into out/paper_build/ and runs a 3-pass build.
set -euo pipefail

REPO="/mnt/d/UserFolders/Documents/GitHub/set-attention"
BUILD="${REPO}/out/paper_build"

mkdir -p "${BUILD}"

echo "=== Copying source files ==="
cp "${REPO}/docs/example_paper_working_agent.tex" "${BUILD}/example_paper_working_agent.tex"
cp "${REPO}/out/final_paper_bundle/overleaf_ready/icml2026.sty"     "${BUILD}/"
cp "${REPO}/out/final_paper_bundle/overleaf_ready/icml2026.bst"     "${BUILD}/"
cp "${REPO}/out/final_paper_bundle/overleaf_ready/example_paper.bib" "${BUILD}/"

cd "${BUILD}"

echo "=== pdflatex pass 1 ==="
pdflatex -interaction=nonstopmode example_paper_working_agent.tex 2>&1 | tee build.log || true

echo "=== bibtex ==="
if grep -q '\\citation{' example_paper_working_agent.aux; then
  bibtex example_paper_working_agent 2>&1 | tee -a build.log || true
else
  {
    echo "No citation commands found; writing an empty bibliography file."
    printf '\\begin{thebibliography}{}\n\\end{thebibliography}\n'
  } | tee example_paper_working_agent.bbl | tee -a build.log >/dev/null
fi

echo "=== pdflatex pass 2 ==="
pdflatex -interaction=nonstopmode example_paper_working_agent.tex 2>&1 | tee -a build.log || true

echo "=== pdflatex pass 3 ==="
pdflatex -interaction=nonstopmode example_paper_working_agent.tex 2>&1 | tee -a build.log || true

echo "=== Extracting issues ==="
grep -n "Error\|error\|Overfull\|Underfull\|Warning\|LaTeX Warning" build.log \
  | grep -v "^Binary\|pdftex.map\|size subst\|Font Warning.*substituted\|Font shape" \
  > build_issues_all.txt || true
head -80 build_issues_all.txt | tee build_issues.txt

echo ""
echo "=== Build complete ==="
if grep -q "^!" build.log; then
  echo "ERRORS FOUND — see build.log"
  grep -n "^!" build.log | head -20
else
  echo "No fatal errors."
fi

echo "PDF: ${BUILD}/example_paper_working_agent.pdf"
echo "Log: ${BUILD}/build.log"
