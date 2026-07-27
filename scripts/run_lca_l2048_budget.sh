#!/usr/bin/env bash
# MRP-lca-cmp L2048 BUDGET probe (diagnostic, user-approved parallel track
# 2026-07-25). Question: is 2000 updates too short at L2048? Rerun the two
# decisive pilot rows at MAX_UPDATES=4000, seed 0:
#   l2048budget_token    token dense
#   l2048budget_b75full  b75 all_past dense topk=2047 (full)
# Separate grid root (out/lca_cmp/l2048budget) and labels so results can
# never be pooled/confused with l2048pilot rows. Resumable; strict log scan.
set -uo pipefail
cd "${REPO_ROOT:-$HOME/set-attention}"

HOST_TAG="${HOST_TAG:-lizmark}"
IMAGE="${IMAGE:-set-attention:latest}"
MAX_UPDATES="${MAX_UPDATES:-4000}"
SEEDS="${SEEDS:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DRY_RUN="${DRY_RUN:-0}"

GRID_ROOT="out/lca_cmp/l2048budget"
DONE_ROOT="${GRID_ROOT}/markers"
LOG_ROOT="logs/lca_cmp/${HOST_TAG}"
RESULT_TSV="${GRID_ROOT}/l2048budget_${HOST_TAG}.tsv"
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

run_row () { # label family topk seed gpu ; family=token|b75 ; topk=0 for token
  local label="$1" family="$2" topk="$3" seed="$4" gpu="$5"
  local name="${label}_L2048_seed${seed}"
  local csv log donemark lock container_name cfg
  csv="${GRID_ROOT}/${family}/L2048/${name}.csv"
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
    "data.seq_len=2048" "model.max_seq_len=2048"
    "data.batch_size=4" "data.supervision=prefix"
    "training.grad_accum_steps=1" "training.eval_microbatch_size=null"
    "training.seed=${seed}" "training.max_updates=${MAX_UPDATES}"
    "training.output_dir=${csv%.csv}"
    "logging.wandb.enable=false" "logging.wandb.run_name=${name}"
  )
  if [ "$family" != token ]; then
    ov+=("model.multiresolution.groups=${B75_GROUPS}"
         "model.candidate_fiber=all_past" "model.router.score_mode=dense"
         "model.router_topk=${topk}")
  fi

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
    echo "=== DONE $name peak=${peak}MiB val_loss=${vloss} val_acc=${vacc} ==="
  else
    echo "=== FAIL rc=${rc} $name ==="
  fi
  rmdir "$lock" 2>/dev/null || true
  return 0
}

echo "=== LCA L2048BUDGET ${HOST_TAG}: SEEDS='${SEEDS}', MAX_UPDATES=${MAX_UPDATES}, DRY_RUN=${DRY_RUN} ==="
for seed in $SEEDS; do
  run_row l2048budget_b75full b75 2047 "$seed" "$GPU0" &
  p0=$!
  run_row l2048budget_token token 0 "$seed" "$GPU1" &
  p1=$!
  wait "$p0" "$p1"
done
echo "=== LCA L2048BUDGET ${HOST_TAG} pass complete ==="
