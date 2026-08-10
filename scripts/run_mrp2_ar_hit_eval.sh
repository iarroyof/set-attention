#!/usr/bin/env bash
# MRP-2 registered natural AR-hit evaluation driver.
# Evaluates the 12 registered checkpoints (token/b0/b25/b100 x seeds 0,1,2,
# L=2048/B=4, exact dense) with scripts/evaluate_ar_hits.py and appends one
# TSV row per evaluation. b25 rows additionally run the registered
# fine/coarse group span-ablation.
#
# Env knobs: HOST_TAG SEEDS ROWS GPU0_ROWS GPU1_ROWS IMAGE DRY_RUN
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST_TAG="${HOST_TAG:-$(hostname -s 2>/dev/null || echo unknown)}"
SEEDS="${SEEDS:-0 1 2}"
ROWS="${ROWS:-token b0 b25 b100}"
GPU0_ROWS="${GPU0_ROWS:-}"
GPU1_ROWS="${GPU1_ROWS:-}"
IMAGE="${IMAGE:-set-attention:latest}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="logs/mrp2_ar_hits/${HOST_TAG}"
EVAL_ROOT="out/mrp2_ar_hits/eval"
RESULT_TSV="${EVAL_ROOT}/mrp2_ar_hit_eval_${HOST_TAG}.tsv"
DRIVER_LOG="${LOG_DIR}/mrp2_eval_driver.log"
mkdir -p "$LOG_DIR" "$EVAL_ROOT"
[ -f "$RESULT_TSV" ] || printf "date\thost\trow\tseed\tgpu\trc\tout_json\n" > "$RESULT_TSV"

log() { echo "$*" | tee -a "$DRIVER_LOG"; }

run_one() {
  local gpu="$1" row="$2" seed="$3"
  local name="mrp2eval_${row}_seed${seed}"
  local ckpt="out/mrp2_ar_hits/retrain/${row}_seed${seed}/checkpoints/final.pt"
  local out_json="${EVAL_ROOT}/${row}_seed${seed}.json"
  local out_csv="${EVAL_ROOT}/${row}_seed${seed}.csv"
  local logf="${LOG_DIR}/${name}.log"

  if [ ! -f "$ckpt" ]; then
    log "=== SKIP $(date '+%F %T') $name missing checkpoint $ckpt ==="
    return 1
  fi
  if [ -f "$out_json" ] && [ "$DRY_RUN" != 1 ]; then
    log "=== SKIP $(date '+%F %T') $name already evaluated ==="
    return 0
  fi

  local extra=()
  if [ "$row" = "b25" ]; then
    extra+=(--group-ablation)
  fi

  log "=== RUN $(date '+%F %T') gpu${gpu} $name ==="
  if [ "$DRY_RUN" = 1 ]; then
    log "PLAN  $name -> $out_json"
    return 0
  fi
  docker run --rm --name "$name" --gpus "device=${gpu}" --ipc=host \
    -u "$(id -u):$(id -g)" \
    -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e WANDB_MODE=offline \
    -v "${PWD}:/workspace" -w /workspace "$IMAGE" \
    /usr/bin/python scripts/evaluate_ar_hits.py \
      --config "configs/eval/ar_hits/${row}.yaml" \
      --checkpoint "$ckpt" \
      --row "$row" --seed "$seed" \
      --device cuda \
      --out-json "$out_json" --out-csv "$out_csv" \
      "${extra[@]}" > "$logf" 2>&1
  local rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "$HOST_TAG" "$row" "$seed" "$gpu" "$rc" "$out_json" \
    >> "$RESULT_TSV"
  if [ "$rc" -eq 0 ] && [ -f "$out_json" ] \
     && ! grep -qiE "traceback|out of memory|outofmemoryerror" "$logf"; then
    log "=== DONE $(date '+%F %T') $name ==="
  else
    log "=== FAIL $(date '+%F %T') rc=${rc} $name ==="
  fi
  return "$rc"
}

run_queue() {
  local gpu="$1"; shift
  local rows="$*"
  local rc_all=0
  for row in $rows; do
    for seed in $SEEDS; do
      run_one "$gpu" "$row" "$seed" || rc_all=1
    done
  done
  return "$rc_all"
}

if [ -z "$GPU0_ROWS" ] && [ -z "$GPU1_ROWS" ]; then
  # split ROWS alternately across the two GPUs
  i=0
  for row in $ROWS; do
    if [ $((i % 2)) -eq 0 ]; then GPU0_ROWS="$GPU0_ROWS $row"; else GPU1_ROWS="$GPU1_ROWS $row"; fi
    i=$((i + 1))
  done
fi

log "=== MRP-2 AR-HIT EVAL ${HOST_TAG}: GPU0_ROWS='${GPU0_ROWS}', GPU1_ROWS='${GPU1_ROWS}', SEEDS='${SEEDS}', DRY_RUN=${DRY_RUN} ==="

pids=()
[ -n "$GPU0_ROWS" ] && run_queue 0 $GPU0_ROWS & pids+=($!)
[ -n "$GPU1_ROWS" ] && run_queue 1 $GPU1_ROWS & pids+=($!)
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

if [ "$rc" -eq 0 ]; then
  log "=== MRP-2 AR-HIT EVAL ${HOST_TAG} pass complete ==="
else
  log "=== MRP-2 AR-HIT EVAL ${HOST_TAG} completed with FAIL rows ==="
fi
exit "$rc"
