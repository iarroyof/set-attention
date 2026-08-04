#!/usr/bin/env bash
# MRP-lca-cmp L4096 DROPOUT-FREE frontier re-measurement (pre-approved
# sequence after the l2048nd confirmation, 2026-08-04): the dropout=0
# recipe is confirmed at L2048 (ceiling 0.880 -> 0.9358+-0.0067, VRAM
# -25%), but the L2048 memory edge shrank to -7.5% because token VRAM
# also drops -36%. At L4096 the dense score tensor is 4x larger, so
# the structural b75 memory edge should hold better — this wave is the
# decisive frontier row under the confirmed recipe.
# Rows: b75nodrop + tokennodrop at L=4096, prefix supervision,
# all_past + score_mode=dense, native B4, 8000 updates, validation
# every EVAL_EVERY updates, ALL dropout knobs 0.0. Seed0 first; seeds
# 1-2 only if seed0 remains promising (user pre-approved).
# Lizmark-only (with dropout: token 33.7GB, b75 24.9GB; nodrop reduces
# both but token stays well above Blue's 24GB).
# Labels l4096nd_*; never pooled with dropout=0.1 rows.
set -uo pipefail
cd "${REPO_ROOT:-$HOME/set-attention}"

HOST_TAG="${HOST_TAG:-lizmark}"
IMAGE="${IMAGE:-set-attention:latest}"
MAX_UPDATES="${MAX_UPDATES:-8000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
ROWS="${ROWS:-b75nodrop:0 tokennodrop:0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DRY_RUN="${DRY_RUN:-0}"

GRID_ROOT="out/lca_cmp/l4096nodrop"
DONE_ROOT="${GRID_ROOT}/markers"
LOG_ROOT="logs/lca_cmp/${HOST_TAG}"
RESULT_TSV="${GRID_ROOT}/l4096nodrop_${HOST_TAG}.tsv"
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

row_spec () { # name -> family topk
  case "$1" in
    tokennodrop) echo "token 0" ;;
    b75nodrop)   echo "b75 4095" ;;
    *) echo "ERROR unknown row $1" >&2; return 1 ;;
  esac
}

run_row () { # label family topk seed gpu
  local label="$1" family="$2" topk="$3" seed="$4" gpu="$5"
  local name="${label}_L4096_seed${seed}"
  local csv log donemark lock container_name cfg
  csv="${GRID_ROOT}/${family}/L4096/${name}.csv"
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

  if [ "$family" = token ]; then cfg="configs/lca_cmp/base_token.yaml"; else cfg="configs/lca_cmp/base_set.yaml"; fi
  local -a ov=(
    "data.seq_len=4096" "model.max_seq_len=4096"
    "data.batch_size=4" "data.supervision=prefix"
    "training.grad_accum_steps=1" "training.eval_microbatch_size=null"
    "training.seed=${seed}" "training.max_updates=${MAX_UPDATES}"
    "training.eval_every=${EVAL_EVERY}"
    "training.output_dir=${csv%.csv}"
    "model.dropout=0.0" "model.attn_dropout=0.0"
    "model.resid_dropout=0.0" "model.ffn_dropout=0.0"
    "logging.wandb.enable=false" "logging.wandb.run_name=${name}"
  )
  if [ "$family" != token ]; then
    ov+=("model.multiresolution.groups=${B75_GROUPS}"
         "model.candidate_fiber=all_past" "model.router.score_mode=dense"
         "model.router_topk=${topk}")
  fi

  echo "=== RUN $(date '+%F %T') gpu${gpu} $name (updates=${MAX_UPDATES}, eval_every=${EVAL_EVERY}, dropout=0) ==="
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

echo "=== LCA L4096 NODROP ${HOST_TAG}: ROWS='${ROWS}', MAX_UPDATES=${MAX_UPDATES}, EVAL_EVERY=${EVAL_EVERY}, DRY_RUN=${DRY_RUN} ==="
set -- $ROWS
while [ $# -gt 0 ]; do
  spec1="$1"; shift
  spec2="${1:-}"; [ $# -gt 0 ] && shift
  name1="${spec1%%:*}"; seed1="${spec1##*:}"
  read -r fam1 topk1 <<< "$(row_spec "$name1")"
  run_row "l4096nd_${name1}" "$fam1" "$topk1" "$seed1" "$GPU0" &
  p0=$!
  if [ -n "$spec2" ]; then
    name2="${spec2%%:*}"; seed2="${spec2##*:}"
    read -r fam2 topk2 <<< "$(row_spec "$name2")"
    run_row "l4096nd_${name2}" "$fam2" "$topk2" "$seed2" "$GPU1" &
    p1=$!
    wait "$p0" "$p1"
  else
    wait "$p0"
  fi
done
echo "=== LCA L4096 NODROP ${HOST_TAG} pass complete ==="
