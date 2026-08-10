#!/usr/bin/env bash
# MRP-2 AR-hit NEW-RECIPE BRIDGE checkpoint retraining driver.
# Trains the bridge matrix (token/b0/b25/b75 x seeds 0,1,2) at the registered
# MRP-2 island L=2048/B=4, WT2, 10 epochs, lr 1e-4, but under the repaired
# global recipe: candidate_fiber=all_past, router.score_mode=dense,
# router_topk=L-1 (full routing), dropout=0.  These rows carry NO experiment
# contract and must never be pooled with registered MRP-2.
#
# Required: MRP2_AR_HITS_BRIDGE=approved and --launch.
# Env knobs: HOST_TAG SEEDS ROWS GPU0_ROWS GPU1_ROWS IMAGE DRY_RUN
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LAUNCH=0
for arg in "$@"; do
  case "$arg" in
    --launch) LAUNCH=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [[ "${MRP2_AR_HITS_BRIDGE:-}" != "approved" || "$LAUNCH" != "1" ]]; then
  cat >&2 <<'MSG'
Refusing to launch the MRP-2 AR-hit new-recipe bridge retraining.

Required:
  MRP2_AR_HITS_BRIDGE=approved
  scripts/run_mrp2_ar_hit_bridge.sh --launch

Rows: token, b0, b25, b75 at L=2048, B=4, exact dense, seeds 0,1,2,
all_past + dense scoring + full routing + dropout 0, with final checkpoint
saving.  Bridge rows carry no experiment contract; never pool with MRP-2.
MSG
  exit 3
fi

HOST_TAG="${HOST_TAG:-$(hostname -s 2>/dev/null || echo unknown)}"
SEEDS="${SEEDS:-0 1 2}"
ROWS="${ROWS:-token b0 b25 b75}"
GPU0_ROWS="${GPU0_ROWS:-}"
GPU1_ROWS="${GPU1_ROWS:-}"
IMAGE="${IMAGE:-set-attention:latest}"
DRY_RUN="${DRY_RUN:-0}"
SEQ_LEN=2048
BATCH=4
EPOCHS=10
LR=0.0001
WARMUP=1000
TOPK_FULL=$((SEQ_LEN - 1))

OUT_ROOT="out/mrp2_ar_hits_bridge/retrain"
LOG_DIR="logs/mrp2_ar_hits_bridge/${HOST_TAG}"
RESULT_TSV="out/mrp2_ar_hits_bridge/mrp2_ar_hit_bridge_${HOST_TAG}.tsv"
DRIVER_LOG="${LOG_DIR}/mrp2_bridge_driver.log"
mkdir -p "$LOG_DIR" "$OUT_ROOT"
[ -f "$RESULT_TSV" ] || printf "date\thost\tname\tgpu\trc\tepochs\tpeak_mib\ttrain_ppl\tval_ppl\tcsv\n" > "$RESULT_TSV"

log() { echo "$*" | tee -a "$DRIVER_LOG"; }

csv_field() { # csv field
  python3 - "$1" "$2" <<'PY'
import csv, sys
path, field = sys.argv[1], sys.argv[2]
try:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(rows[-1].get(field, "") if rows else "")
except Exception:
    print("")
PY
}

csv_complete() { # csv epochs
  python3 - "$1" "$2" <<'PY'
import csv, sys
path, epochs = sys.argv[1], int(sys.argv[2])
try:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    sys.exit(0 if len(rows) >= epochs else 1)
except Exception:
    sys.exit(1)
PY
}

set_groups() {
  local row="$1"
  case "$row" in
    b0) echo "[{name: fine, num_heads: 8, window_size: 2, stride: 1}]" ;;
    b25) echo "[{name: fine, num_heads: 6, window_size: 2, stride: 1}, {name: coarse, num_heads: 2, window_size: 4, stride: 2}]" ;;
    b75) echo "[{name: fine, num_heads: 2, window_size: 2, stride: 1}, {name: coarse, num_heads: 6, window_size: 4, stride: 2}]" ;;
    *) echo "unknown set row: $row" >&2; return 2 ;;
  esac
}

run_one() {
  local gpu="$1" row="$2" seed="$3"
  local name="mrp2bridge_${row}_seed${seed}"
  local out="${OUT_ROOT}/${row}_seed${seed}"
  local csv="${out}.csv"
  local ckpt="${out}/checkpoints/final.pt"
  local logf="${LOG_DIR}/${name}.log"

  if [ -s "$ckpt" ] && [ "$DRY_RUN" != 1 ]; then
    log "=== SKIP $(date '+%F %T') $name checkpoint exists ==="
    return 0
  fi
  mkdir -p "$out"

  local cfg ov
  if [[ "$row" == "token" ]]; then
    cfg="configs/paper_lr_norm/baseline_dense_exact.yaml"
    ov=(
      "model.dropout=0.0" "model.attn_dropout=0.0"
      "model.resid_dropout=0.0" "model.ffn_dropout=0.0"
    )
  else
    cfg="configs/set_dictionary/sd9_multiresolution.yaml"
    local groups
    groups="$(set_groups "$row")"
    ov=(
      "model.output_residual_mode=anchor_span"
      "model.token_mlp.enabled=false"
      "model.anchor.enabled=false"
      "model.allow_token_token=false"
      "model.candidate_fiber=all_past"
      "model.router.score_mode=dense"
      "model.router_topk=${TOPK_FULL}"
      "model.dropout=0.0" "model.attn_dropout=0.0"
      "model.resid_dropout=0.0" "model.ffn_dropout=0.0"
      "model.window_size=2"
      "model.stride=1"
      "model.multiresolution.enabled=true"
      "model.multiresolution.groups=${groups}"
    )
  fi
  ov+=(
    "data.dataset=wikitext2"
    "data.seq_len=${SEQ_LEN}"
    "data.batch_size=${BATCH}"
    "training.epochs=${EPOCHS}"
    "training.lr=${LR}"
    "training.warmup_steps=${WARMUP}"
    "training.deterministic=true"
    "training.benchmark_mode=false"
    "training.seed=${seed}"
    "training.output_dir=${out}"
    "training.checkpoint.directory=${out}/checkpoints"
    "training.checkpoint.save_final=true"
    "training.checkpoint.save_every_epochs=0"
    "logging.wandb.enable=false"
    "model.attention_family=dense"
    "model.backend=exact"
    "model.max_seq_len=${SEQ_LEN}"
    "model.d_model=384"
    "model.dim_feedforward=1536"
    "model.num_layers=6"
    "model.num_heads=8"
  )

  log "=== RUN $(date '+%F %T') gpu${gpu} $name (epochs=${EPOCHS}, L=${SEQ_LEN}, B=${BATCH}) ==="
  if [ "$DRY_RUN" = 1 ]; then
    log "PLAN  $name -> $ckpt"
    return 0
  fi
  docker run --rm --name "$name" --gpus "device=${gpu}" --ipc=host \
    -u "$(id -u):$(id -g)" \
    -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e WANDB_MODE=offline \
    -v "${PWD}:/workspace" -w /workspace "$IMAGE" \
    /usr/bin/python scripts/run_experiment.py --config "$cfg" --csv-path "$csv" \
    --override "${ov[@]}" > "$logf" 2>&1
  local rc=$?

  local peak tppl vppl ep
  ep=$(($( { wc -l < "$csv"; } 2>/dev/null || echo 1) - 1)); [ "$ep" -lt 0 ] && ep=0
  peak="$(csv_field "$csv" "train/peak_vram_mib")"
  tppl="$(csv_field "$csv" "train/ppl")"
  vppl="$(csv_field "$csv" "val/ppl")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "$HOST_TAG" "$name" "$gpu" "$rc" "$ep" "$peak" "$tppl" "$vppl" "$csv" \
    >> "$RESULT_TSV"

  if [ "$rc" -eq 0 ] && [ -s "$ckpt" ] && csv_complete "$csv" "$EPOCHS" \
     && ! grep -qiE "traceback|out of memory|outofmemoryerror" "$logf"; then
    log "=== DONE $(date '+%F %T') $name epochs=${ep} peak=${peak}MiB val_ppl=${vppl} ==="
  else
    log "=== FAIL $(date '+%F %T') rc=${rc} $name ==="
    return 1
  fi
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
  i=0
  for row in $ROWS; do
    if [ $((i % 2)) -eq 0 ]; then GPU0_ROWS="$GPU0_ROWS $row"; else GPU1_ROWS="$GPU1_ROWS $row"; fi
    i=$((i + 1))
  done
fi

log "=== MRP-2 AR-HIT BRIDGE ${HOST_TAG}: GPU0_ROWS='${GPU0_ROWS}', GPU1_ROWS='${GPU1_ROWS}', SEEDS='${SEEDS}', DRY_RUN=${DRY_RUN} ==="

pids=()
if [ -n "$GPU0_ROWS" ]; then run_queue 0 $GPU0_ROWS & pids+=($!); fi
if [ -n "$GPU1_ROWS" ]; then run_queue 1 $GPU1_ROWS & pids+=($!); fi
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

if [ "$rc" -eq 0 ]; then
  log "=== MRP-2 AR-HIT BRIDGE ${HOST_TAG} pass complete ==="
else
  log "=== MRP-2 AR-HIT BRIDGE ${HOST_TAG} completed with FAIL rows ==="
fi
exit "$rc"
