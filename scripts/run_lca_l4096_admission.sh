#!/usr/bin/env bash
# MRP-lca-cmp L4096 STAGE-A ADMISSION probe (user-approved amendment
# 2026-07-27, docs/agent_plans/mrp_lca_cmp.md). Memory/admission ONLY:
# measures peak VRAM at L=4096 for token vs b75 full routing. No
# scientific claims; labels l4096adm_*; never pooled with matrix or
# diagnostic training rows.
#
# Rows (native B4 first, max_updates=30, seed 0, prefix supervision):
#   l4096adm_token     token dense
#   l4096adm_b75full   b75 all_past dense topk=4095 (full)
# OOM fallback: if a native row OOMs, record it and retry B2 x accum2
# (label suffix _mb2) as a memory-control measurement. The native peak
# (or native OOM) is the headline number per the un-optimized-memory
# tracking directive.
set -uo pipefail
cd "${REPO_ROOT:-$HOME/set-attention}"

HOST_TAG="${HOST_TAG:-lizmark}"
IMAGE="${IMAGE:-set-attention:latest}"
MAX_UPDATES="${MAX_UPDATES:-30}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DRY_RUN="${DRY_RUN:-0}"

GRID_ROOT="out/lca_cmp/l4096admission"
DONE_ROOT="${GRID_ROOT}/markers"
LOG_ROOT="logs/lca_cmp/${HOST_TAG}"
RESULT_TSV="${GRID_ROOT}/l4096admission_${HOST_TAG}.tsv"
mkdir -p "$DONE_ROOT" "$LOG_ROOT"
[ -f "$RESULT_TSV" ] || printf "ts\thost\tlabel\tgpu\texit\tpeak_vram_mib\ttrain_loss\tval_loss\tval_acc\tcsv\n" > "$RESULT_TSV"

B75_GROUPS='[{name: fine, num_heads: 2, window_size: 2, stride: 1}, {name: coarse, num_heads: 6, window_size: 4, stride: 2}]'

csv_field () {
  python3 - "$1" "$2" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    with p.open(newline="") as fh:
        rows = list(csv.DictReader(line.replace("\0", "") for line in fh))
    print(rows[-1].get(sys.argv[2], "NA") if rows else "NA")
except Exception:
    print("NA")
PY
}

run_row () { # label family topk gpu bsz accum
  local label="$1" family="$2" topk="$3" gpu="$4" bsz="$5" accum="$6"
  local name="${label}_L4096_seed0"
  local csv log donemark container_name cfg
  csv="${GRID_ROOT}/${family}/L4096/${name}.csv"
  log="${LOG_ROOT}/${name}.log"
  donemark="${DONE_ROOT}/${name}.done"
  container_name="lcacmp_${HOST_TAG}_${name}"

  if [ -f "$donemark" ]; then
    echo "SKIP done    $name"
    return 0
  fi

  if [ "$family" = token ]; then cfg="configs/lca_cmp/base_token.yaml"; else cfg="configs/lca_cmp/base_set.yaml"; fi
  local -a ov=(
    "data.seq_len=4096" "model.max_seq_len=4096"
    "data.batch_size=${bsz}" "data.supervision=prefix"
    "training.grad_accum_steps=${accum}" "training.eval_microbatch_size=null"
    "training.seed=0" "training.max_updates=${MAX_UPDATES}"
    "training.output_dir=${csv%.csv}"
    "logging.wandb.enable=false" "logging.wandb.run_name=${name}"
  )
  if [ "$family" != token ]; then
    ov+=("model.multiresolution.groups=${B75_GROUPS}"
         "model.candidate_fiber=all_past" "model.router.score_mode=dense"
         "model.router_topk=${topk}")
  fi

  echo "=== RUN gpu${gpu} $name (updates=${MAX_UPDATES}, B${bsz}xaccum${accum}) ==="
  if [ "$DRY_RUN" = 1 ]; then
    echo "PLAN  $name -> $csv"
    return 0
  fi
  mkdir -p "$(dirname "$csv")"
  docker run --rm --name "$container_name" --gpus "device=${gpu}" --ipc=host \
    -u "$(id -u):$(id -g)" \
    -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e WANDB_MODE=offline \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" \
    /usr/bin/python scripts/run_lca_cmp.py --config "$cfg" --csv-path "$csv" \
    --override "${ov[@]}" > "$log" 2>&1
  local rc=$?

  local peak tloss vloss vacc
  peak="$(csv_field "$csv" "train/peak_vram_mib")"
  tloss="$(csv_field "$csv" "train/loss")"
  vloss="$(csv_field "$csv" "val/loss")"
  vacc="$(csv_field "$csv" "val/accuracy")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "$HOST_TAG" "$name" "$gpu" "$rc" "$peak" "$tloss" "$vloss" "$vacc" "$csv" \
    >> "$RESULT_TSV"

  if [ "$rc" -eq 0 ] && ! grep -qiE "traceback|out of memory|outofmemoryerror" "$log"; then
    touch "$donemark"
    echo "=== DONE $name peak=${peak}MiB ==="
    return 0
  fi
  if grep -qiE "out of memory|outofmemoryerror" "$log"; then
    echo "=== OOM $name ==="
    return 3
  fi
  echo "=== FAIL rc=${rc} $name ==="
  return 1
}

echo "=== LCA L4096ADMISSION ${HOST_TAG}: MAX_UPDATES=${MAX_UPDATES}, DRY_RUN=${DRY_RUN} ==="
run_row l4096adm_b75full b75 4095 "$GPU0" 4 1 &
p0=$!
run_row l4096adm_token token 0 "$GPU1" 4 1 &
p1=$!
rc0=0; rc1=0
wait "$p0" || rc0=$?
wait "$p1" || rc1=$?
# OOM fallbacks (sequential, same GPUs): B2 x accum2, effective batch 4.
if [ "$rc0" -eq 3 ]; then
  run_row l4096adm_b75full_mb2 b75 4095 "$GPU0" 2 2 || true
fi
if [ "$rc1" -eq 3 ]; then
  run_row l4096adm_token_mb2 token 0 "$GPU1" 2 2 || true
fi
echo "=== LCA L4096ADMISSION ${HOST_TAG} pass complete ==="
