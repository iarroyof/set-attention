#!/usr/bin/env bash
set -euo pipefail

cd ~/set-attention
mkdir -p logs

docker compose exec -T set-attention python scripts/summarize_paper_family_complements.py \
  | tee logs/summarize_paper_family_complements.log

docker compose exec -T set-attention python scripts/plot_paper_family_complements.py \
  | tee logs/plot_paper_family_complements.log

echo
echo "Done."
echo "Bundle root: out/paper_complements_bundle"
echo "Tables: out/paper_complements_bundle/tables/"
echo "Latex:  out/paper_complements_bundle/latex/"
echo "Plots:  out/paper_complements_bundle/plots/"
echo "Checks: out/paper_complements_bundle/checks/"
