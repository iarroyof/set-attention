#!/usr/bin/env bash
set -euo pipefail

# Run on blue-demon from ~/set-attention.
# This launches the headline baseline-vs-dense LR normalization on GPU 0 and
# the sparse/linear anchor-family LR normalization on GPU 1.
mkdir -p logs
nohup bash scripts/gpu0_run_lrnorm_headline_pairs.sh > logs/gpu0_lrnorm_headline_pairs.nohup.log 2>&1 &
echo "GPU0 headline LR-normalization PID: $!"
nohup bash scripts/gpu1_run_lrnorm_family_anchor.sh > logs/gpu1_lrnorm_family_anchor.nohup.log 2>&1 &
echo "GPU1 family-anchor LR-normalization PID: $!"
