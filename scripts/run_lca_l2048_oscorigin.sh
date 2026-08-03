#!/usr/bin/env bash
# MRP-lca-cmp L2048 OSCILLATION-ORIGIN probes (2026-08-01; follows the
# lr-decay probe verdict in audit/LCA_calibration_20260718.md: cosine decay
# did NOT damp the oscillation; aligned troughs between the const/cosine
# twins point to a data-driven component; dropout 0.1 is the other
# untested suspect). Two b75 full-routing rows at L2048/prefix/B4, 4000
# updates, validation every 500, against the existing const control
# (l2048lr_b75const_L2048_seed0):
#   b75seed3  — training.seed=3: SAME dataset (dataset_seed=1729) and SAME
#               val set, but different batch order + init + dropout masks.
#               If the const row's trough pattern (dips at updates
#               1500-2000) reappears at the same updates, the oscillation
#               is update-indexed dynamics; if it moves/vanishes, it tracks
#               the data sequence.
#   b75nodrop — all dropout knobs 0.0: if the oscillation vanishes or
#               shrinks materially, dropout is a major contributor.
# Blue-only (b75 L2048 peaks ~7.2 GB). Labels l2048osc_*; never pooled
# with matrix/pilot/budget/stageb/lrdecay rows.
set -uo pipefail
cd "${REPO_ROOT:-$HOME/set-attention}"

HOST_TAG="${HOST_TAG:-blue}"
IMAGE="${IMAGE:-set-attention:latest}"
MAX_UPDATES="${MAX_UPDATES:-4000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
ROWS="${ROWS:-b75seed3:3 b75nodrop:0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DRY_RUN="${DRY_RUN:-0}"

GRID_ROOT="out/lca_cmp/l2048oscorigin"
DONE_ROOT="${GRID_ROOT}/markers"
LOG_ROOT="logs/lca_cmp/${HOST_TAG}"
RESULT_TSV="${GRID_ROOT}/l2048oscorigin_${HOST_TAG}.tsv"
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

run_row () { # label seed gpu extra_overrides...
  local label="$1" seed="$2" gpu="$3"; shift 3
  local name="${label}_L2048_seed${seed}"
  local csv log donemark lock container_name
  csv="${GRID_ROOT}/b75/L2048/${name}.csv"
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
    "data.seq_len=2048" "model.max_seq_len=2048"
    "data.batch_size=4" "data.supervision=prefix"
    "training.grad_accum_steps=1" "training.eval_microbatch_size=null"
    "training.seed=${seed}" "training.max_updates=${MAX_UPDATES}"
    "training.eval_every=${EVAL_EVERY}"
    "training.output_dir=${csv%.csv}"
    "logging.wandb.enable=false" "logging.wandb.run_name=${name}"
    "model.multiresolution.groups=${B75_GROUPS}"
    "model.candidate_fiber=all_past" "model.router.score_mode=dense"
    "model.router_topk=2047"
  )
  ov+=("$@")

  echo "=== RUN $(date '+%F %T') gpu${gpu} $name (updates=${MAX_UPDATES}, eval_every=${EVAL_EVERY}, extra: $*) ==="
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
    echo "=== DONE $(date '+%F %T') $name peak=${peak}MiB val_loss=${vloss} val_acc=${vacc} ==="
  else
    echo "=== FAIL $(date '+%F %T') rc=${rc} $name ==="
  fi
  rmdir "$lock" 2>/dev/null || true
  return 0
}

echo "=== LCA L2048 OSCORIGIN ${HOST_TAG}: ROWS='${ROWS}', MAX_UPDATES=${MAX_UPDATES}, EVAL_EVERY=${EVAL_EVERY}, DRY_RUN=${DRY_RUN} ==="
set -- $ROWS
while [ $# -gt 0 ]; do
  spec1="$1"; shift
  spec2="${1:-}"; [ $# -gt 0 ] && shift
  name1="${spec1%%:*}"; seed1="${spec1##*:}"
  case "$name1" in
    b75seed3)  extra1=("training.seed=${seed1}") ;;
    b75nodrop) extra1=("model.dropout=0.0" "model.attn_dropout=0.0" "model.resid_dropout=0.0" "model.ffn_dropout=0.0") ;;
    *) echo "ERROR unknown row $name1" >&2; exit 1 ;;
  esac
  run_row "l2048osc_${name1}" "$seed1" "$GPU0" "${extra1[@]}" &
  p0=$!
  if [ -n "$spec2" ]; then
    name2="${spec2%%:*}"; seed2="${spec2##*:}"
    case "$name2" in
      b75seed3)  extra2=("training.seed=${seed2}") ;;
      b75nodrop) extra2=("model.dropout=0.0" "model.attn_dropout=0.0" "model.resid_dropout=0.0" "model.ffn_dropout=0.0") ;;
      *) echo "ERROR unknown row $name2" >&2; exit 1 ;;
    esac
    run_row "l2048osc_${name2}" "$seed2" "$GPU1" "${extra2[@]}" &
    p1=$!
    wait "$p0" "$p1"
  else
    wait "$p0"
  fi
done
echo "=== LCA L2048 OSCORIGIN ${HOST_TAG} pass complete ==="
