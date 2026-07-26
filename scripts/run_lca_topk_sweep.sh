#!/usr/bin/env bash
# MRP-lca-cmp b75 TOP-K BANDWIDTH SWEEP (user-approved 2026-07-24,
# docs/agent_plans/mrp_lca_cmp.md "Approved Diagnostic Probes").
#
# Question: how much routing bandwidth does the frontier blur allocation
# (b75) need before quality collapses? Rows: b75/L1024, all_past,
# score_mode=dense, data.supervision=prefix, router_topk in
# {16,32,64,128,256,512,1023}, seeds 0-2, native B4, max_updates=2000.
# Diagnostic labels (topksweep_*), never pooled with matrix rows.
#
# Resumable: row with .done marker or complete final CSV is skipped.
# Strict log scan before .done. Host: blue-demon (<=24GB rows).
set -uo pipefail
cd "${REPO_ROOT:-$HOME/set-attention}"

HOST_TAG="${HOST_TAG:-blue}"
IMAGE="${IMAGE:-set-attention:latest}"
MAX_UPDATES="${MAX_UPDATES:-2000}"
SEEDS="${SEEDS:-0 1 2}"
TOPKS="${TOPKS:-16 32 64 128 256 512 1023}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DRY_RUN="${DRY_RUN:-0}"

GRID_ROOT="out/lca_cmp/topksweep"
DONE_ROOT="${GRID_ROOT}/markers"
LOG_ROOT="logs/lca_cmp/${HOST_TAG}"
RESULT_TSV="${GRID_ROOT}/topksweep_${HOST_TAG}.tsv"
mkdir -p "$DONE_ROOT" "$LOG_ROOT"
[ -f "$RESULT_TSV" ] || printf "ts\thost\tlabel\tgpu\texit\tpeak_vram_mib\ttrain_loss\tval_loss\tval_acc\tcsv\n" > "$RESULT_TSV"

B75_GROUPS='[{name: fine, num_heads: 2, window_size: 2, stride: 1}, {name: coarse, num_heads: 6, window_size: 4, stride: 2}]'

csv_complete () {
  python3 - "$1" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
try:
    with p.open(newline="") as fh:
        rows = list(csv.DictReader(line.replace("\0", "") for line in fh))
except Exception:
    raise SystemExit(1)
if not rows:
    raise SystemExit(1)
row = rows[-1]
bad = {"nan", "inf", "-inf", "infinity", "-infinity"}
for key in ("train/loss", "val/loss", "val/accuracy", "train/peak_vram_mib"):
    value = str(row.get(key, "")).strip().lower()
    if value in {"", "na", "none"} or value in bad:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

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

run_row () { # topk seed gpu
  local topk="$1" seed="$2" gpu="$3"
  local name="topksweep_b75_L1024_topk${topk}_seed${seed}"
  local csv log donemark lock container_name
  csv="${GRID_ROOT}/b75/L1024/${name}.csv"
  log="${LOG_ROOT}/${name}.log"
  donemark="${DONE_ROOT}/${name}.done"
  lock="${DONE_ROOT}/${name}.lock"
  container_name="lcacmp_${HOST_TAG}_${name}"

  if [ -f "$donemark" ] || csv_complete "$csv"; then
    echo "SKIP done    $name"
    return 0
  fi
  if ! mkdir "$lock" 2>/dev/null; then
    echo "SKIP locked  $name"
    return 0
  fi

  local -a ov=(
    "data.seq_len=1024" "model.max_seq_len=1024"
    "data.batch_size=4" "data.supervision=prefix"
    "training.grad_accum_steps=1" "training.eval_microbatch_size=null"
    "training.seed=${seed}" "training.max_updates=${MAX_UPDATES}"
    "training.output_dir=${csv%.csv}"
    "logging.wandb.enable=false" "logging.wandb.run_name=${name}"
    "model.multiresolution.groups=${B75_GROUPS}"
    "model.candidate_fiber=all_past" "model.router.score_mode=dense"
    "model.router_topk=${topk}"
  )

  echo "=== RUN gpu${gpu} $name (updates=${MAX_UPDATES}) ==="
  if [ "$DRY_RUN" = 1 ]; then
    echo "PLAN  $name -> $csv"
    rmdir "$lock" 2>/dev/null || true
    return 0
  fi
  mkdir -p "$(dirname "$csv")"
  docker run --rm --name "$container_name" --gpus "device=${gpu}" --ipc=host \
    -u "$(id -u):$(id -g)" \
    -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e WANDB_MODE=offline \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" \
    /usr/bin/python scripts/run_lca_cmp.py --config configs/lca_cmp/base_set.yaml \
    --csv-path "$csv" --override "${ov[@]}" > "$log" 2>&1
  local rc=$?

  local peak tloss vloss vacc
  peak="$(csv_field "$csv" "train/peak_vram_mib")"
  tloss="$(csv_field "$csv" "train/loss")"
  vloss="$(csv_field "$csv" "val/loss")"
  vacc="$(csv_field "$csv" "val/accuracy")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "$HOST_TAG" "$name" "$gpu" "$rc" "$peak" "$tloss" "$vloss" "$vacc" "$csv" \
    >> "$RESULT_TSV"

  if [ "$rc" -eq 0 ] && csv_complete "$csv" \
     && ! grep -qiE "traceback|out of memory|outofmemoryerror" "$log"; then
    touch "$donemark"
    echo "=== DONE $name peak=${peak}MiB val_loss=${vloss} val_acc=${vacc} ==="
  else
    echo "=== FAIL rc=${rc} $name ==="
  fi
  rmdir "$lock" 2>/dev/null || true
  return 0
}

worker () { # gpu topks...
  local gpu="$1"; shift
  local topk seed
  for topk in "$@"; do
    for seed in $SEEDS; do
      run_row "$topk" "$seed" "$gpu"
    done
  done
}

echo "=== LCA TOPKSWEEP b75 ${HOST_TAG}: TOPKS='${TOPKS}', SEEDS='${SEEDS}', MAX_UPDATES=${MAX_UPDATES}, DRY_RUN=${DRY_RUN} ==="
worker "$GPU0" 16 64 256 1023 &
w0=$!
worker "$GPU1" 32 128 512 &
w1=$!
wait "$w0" "$w1"
echo "=== LCA TOPKSWEEP ${HOST_TAG} pass complete ==="
