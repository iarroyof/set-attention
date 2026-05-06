#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-iarroyof@192.168.241.149}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/iarroyof/set-attention}"
SSH_PASS_FILE="${SSH_PASS_FILE:-$HOME/.ssh/.sshpass}"

mkdir -p \
  out/paper_lr_norm/paper_lr_norm_family_D384_FF1536 \
  out/paper_lr_norm/paper_lr_norm_headline_D384_FF1536

sshpass -f "$SSH_PASS_FILE" scp -o StrictHostKeyChecking=no \
  "$REMOTE:$REMOTE_ROOT/out/paper_lr_norm/paper_lr_norm_family_D384_FF1536/*.csv" \
  out/paper_lr_norm/paper_lr_norm_family_D384_FF1536/

sshpass -f "$SSH_PASS_FILE" scp -o StrictHostKeyChecking=no \
  "$REMOTE:$REMOTE_ROOT/out/paper_lr_norm/paper_lr_norm_headline_D384_FF1536/*.csv" \
  out/paper_lr_norm/paper_lr_norm_headline_D384_FF1536/

echo "Fetched partial LR-normalized D384_FF1536 evidence into out/paper_lr_norm/."
