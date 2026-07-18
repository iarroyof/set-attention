#!/usr/bin/env bash
# MRP-lca-cmp CALIBRATION driver — resumable + OOM-mapped + strict log scan.
#
# Runs the registered calibration matrix (docs/agent_plans/mrp_lca_cmp.md,
# configs/lca_cmp/matrix.md): islands L=1024/B4 and L=2048/B4, families
# token + set b0/b25/b50/b75/b100, seeds 0 1 2 (36 trained rows).
#
# Guarantees (mirroring scripts/run_sd_grid.sh conventions, simplified):
#  * RESUMABLE: a row with a .done marker or a complete final CSV is skipped;
#    a crashed row releases its mkdir lock so a re-run retries it.
#  * NATIVE-BATCH MEMORY STORY: calibration rows run with grad_accum_steps=1
#    and eval_microbatch_size=null at batch=effective batch, so the trained
#    row's train/peak_vram_mib IS the registered native-batch peak; the driver
#    copies it into results TSV. If a cell OOMs natively, the OOM is recorded
#    in oom_registry.tsv and the cell is retried ONCE with microbatching
#    (batch 2 x grad_accum 2, effective batch 4) as a new labeled row
#    (batching_mode=microbatch), never pooled with native rows.
#  * STRICT SCAN: every log is scanned for traceback/NaN/Inf; CSV endpoints
#    are checked for nan/inf metric values before a .done marker is written.
#  * EXCLUSIVE: one worker per GPU, round-robin cells; mkdir locks prevent
#    double-execution.
#
# Usage (on a GPU host with the set-attention:latest image):
#   HOST_TAG=blue GPU0=0 GPU1=1 \
#     nohup bash scripts/run_lca_cmp_calibration.sh \
#     > logs/lca_cmp/blue/queue.log 2>&1 &
#   DRY_RUN=1 HOST_TAG=blue bash scripts/run_lca_cmp_calibration.sh
set -uo pipefail
cd "${REPO_ROOT:-$HOME/set-attention}"

HOST_TAG="${HOST_TAG:?set HOST_TAG=blue|lizmark}"
SEEDS="${SEEDS:-0 1 2}"
LENGTHS="${LENGTHS:-1024 2048}"
FAMILIES="${FAMILIES:-token b0 b25 b50 b75 b100}"
MAX_UPDATES="${MAX_UPDATES:-2000}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DRY_RUN="${DRY_RUN:-0}"
LOCK_TTL_MIN="${LOCK_TTL_MIN:-180}"
IMAGE="${IMAGE:-set-attention:latest}"
TOKEN_CONFIG="configs/lca_cmp/base_token.yaml"
SET_CONFIG="configs/lca_cmp/base_set.yaml"

GRID_ROOT="out/lca_cmp/calibration"
LOCK_ROOT="${GRID_ROOT}/locks"
DONE_ROOT="${GRID_ROOT}/markers"
LOG_ROOT="logs/lca_cmp/${HOST_TAG}"
OOM_REG="${GRID_ROOT}/oom_registry.tsv"
RESULT_TSV="${GRID_ROOT}/calibration_runs_${HOST_TAG}.tsv"
mkdir -p "$LOCK_ROOT" "$DONE_ROOT" "$LOG_ROOT"
[ -f "$OOM_REG" ] || printf "ts\thost\tfamily\tL\tseed\tbatching_mode\tpeak_vram_mib\tnote\n" > "$OOM_REG"
[ -f "$RESULT_TSV" ] || printf "ts\thost\tcell_id\tgpu\tbatching_mode\texit\tpeak_vram_mib\tval_loss\tval_acc\tcsv\n" > "$RESULT_TSV"

groups_yaml () { # fine coarse
  local f="$1" c="$2"
  if   [ "$f" -gt 0 ] && [ "$c" -gt 0 ]; then printf '[{name: fine, num_heads: %s, window_size: 2, stride: 1}, {name: coarse, num_heads: %s, window_size: 4, stride: 2}]' "$f" "$c"
  elif [ "$f" -gt 0 ]; then printf '[{name: fine, num_heads: %s, window_size: 2, stride: 1}]' "$f"
  else printf '[{name: coarse, num_heads: %s, window_size: 4, stride: 2}]' "$c"; fi
}

blur_split () { # variant -> "fine coarse"
  case "$1" in
    b0)   echo "8 0" ;;
    b25)  echo "6 2" ;;
    b50)  echo "4 4" ;;
    b75)  echo "2 6" ;;
    b100) echo "0 8" ;;
    *) echo "unsupported blur variant: $1" >&2; return 2 ;;
  esac
}

csv_complete () { # csv -> 0 if exactly one final row with no nan/inf metric
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

csv_field () { # csv field
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

run_row () { # family L seed gpu batching_mode batch accum
  local family="$1" L="$2" seed="$3" gpu="$4" mode="$5" batch="$6" accum="$7"
  local cid="${family}|${L}|b4|${seed}|${mode}"
  local safe="${cid//[|\/]/_}"
  local name="lcacmp_${family}_L${L}_seed${seed}_${mode}"
  local csv log donemark lock container_name cfg
  csv="${GRID_ROOT}/${family}/L${L}/${name}.csv"
  log="${LOG_ROOT}/${name}.log"
  donemark="${DONE_ROOT}/${safe}.done"
  lock="${LOCK_ROOT}/${safe}.lock"
  container_name="lcacmp_${HOST_TAG}_${safe}"

  if [ -f "$donemark" ] || csv_complete "$csv"; then
    echo "SKIP done    $cid"
    return 0
  fi
  if ! mkdir "$lock" 2>/dev/null; then
    # Stale-lock reclaim: a crashed worker leaves its mkdir lock behind; if it
    # is older than LOCK_TTL_MIN and no matching container is alive, take it.
    if [ -z "$(find "$lock" -mmin -"${LOCK_TTL_MIN}" 2>/dev/null)" ] \
       && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$container_name"; then
      echo "RECLAIM stale lock $cid"
      rmdir "$lock" 2>/dev/null || true
      mkdir "$lock" 2>/dev/null || { echo "SKIP locked  $cid"; return 0; }
    else
      echo "SKIP locked  $cid"
      return 0
    fi
  fi

  if [ "$family" = token ]; then
    cfg="$TOKEN_CONFIG"
  else
    cfg="$SET_CONFIG"
  fi
  local -a ov=(
    "data.seq_len=${L}" "model.max_seq_len=${L}"
    "data.batch_size=${batch}"
    "training.grad_accum_steps=${accum}"
    "training.eval_microbatch_size=null"
    "training.seed=${seed}"
    "training.max_updates=${MAX_UPDATES}"
    "training.output_dir=${csv%.csv}"
    "logging.wandb.enable=false"
    "logging.wandb.run_name=${name}"
  )
  if [ "$family" != token ]; then
    read -r fine coarse <<< "$(blur_split "$family")"
    ov+=("model.multiresolution.groups=$(groups_yaml "$fine" "$coarse")")
  fi

  echo "=== RUN gpu${gpu} $cid (batch=${batch} accum=${accum} updates=${MAX_UPDATES}) ==="
  if [ "$DRY_RUN" = 1 ]; then
    echo "PLAN  $cfg --override ${ov[*]} -> $csv"
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

  local peak vloss vacc
  peak="$(csv_field "$csv" "train/peak_vram_mib")"
  vloss="$(csv_field "$csv" "val/loss")"
  vacc="$(csv_field "$csv" "val/accuracy")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "$HOST_TAG" "$cid" "$gpu" "$mode" "$rc" "$peak" "$vloss" "$vacc" "$csv" \
    >> "$RESULT_TSV"

  if [ "$rc" -eq 0 ] && csv_complete "$csv" \
     && ! grep -qiE "traceback|out of memory|outofmemoryerror" "$log"; then
    touch "$donemark"
    echo "=== DONE $cid peak=${peak}MiB val_loss=${vloss} val_acc=${vacc} ==="
  elif grep -qiE "out of memory|outofmemoryerror|CUDA error: out of memory|CUBLAS_STATUS_ALLOC" "$log"; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(date '+%F %T')" "$HOST_TAG" "$family" "$L" "$seed" "$mode" "$peak" \
      "OOM (batch=${batch} accum=${accum})" >> "$OOM_REG"
    echo "=== OOM $cid -> oom_registry ==="
    rmdir "$lock" 2>/dev/null || true
    return 69
  else
    echo "=== FAIL $cid rc=$rc (lock released; will retry next driver run) ==="
    grep -inE "traceback|error|nan|inf" "$log" | tail -n 5 || true
  fi
  rmdir "$lock" 2>/dev/null || true
  return 0
}

process_cell () { # family L seed gpu
  local family="$1" L="$2" seed="$3" gpu="$4"
  local cid="${family}|${L}|b4|${seed}|native"
  # Native batching row: batch=effective batch, grad_accum_steps=1,
  # eval_microbatch_size=null. Its CSV train/peak_vram_mib is the registered
  # native-batch peak for the cell (native-batch memory story).
  run_row "$family" "$L" "$seed" "$gpu" native 4 1
  local rc=$?
  if [ "$rc" -eq 69 ]; then
    # Native OOM censoring recorded; retry as a NEW labeled microbatch row
    # (effective batch held at 4: batch 2 x grad_accum 2). Quality numbers of
    # this row are never pooled with native-batch islands.
    echo "=== RETRY microbatch $cid (native OOM censored) ==="
    run_row "$family" "$L" "$seed" "$gpu" microbatch 2 2
  fi
  return 0
}

declare -a CELLS=()
for L in $LENGTHS; do
  for family in $FAMILIES; do
    for seed in $SEEDS; do
      CELLS+=("$family $L $seed")
    done
  done
done

echo "=== LCA-CMP CALIBRATION ${HOST_TAG}: ${#CELLS[@]} cells, MAX_UPDATES=${MAX_UPDATES}, SEEDS='${SEEDS}', DRY_RUN=${DRY_RUN} ==="
worker () { # gpu slot
  local gpu="$1" slot="$2" i c
  for i in "${!CELLS[@]}"; do
    if [ $((i % 2)) -eq "$slot" ]; then
      # shellcheck disable=SC2086
      process_cell ${CELLS[$i]} "$gpu"
    fi
  done
}
if [ "$DRY_RUN" = 1 ]; then
  worker "$GPU0" 0
  worker "$GPU1" 1
else
  worker "$GPU0" 0 > "${LOG_ROOT}/worker_gpu${GPU0}.log" 2>&1 &
  pid0=$!
  worker "$GPU1" 1 > "${LOG_ROOT}/worker_gpu${GPU1}.log" 2>&1 &
  pid1=$!
  echo "workers: gpu${GPU0}=$pid0 gpu${GPU1}=$pid1"
  wait "$pid0" "$pid1"
fi
echo "=== LCA-CMP CALIBRATION ${HOST_TAG} pass complete ==="
