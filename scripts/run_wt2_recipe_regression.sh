#!/usr/bin/env bash
# WT2 RECIPE REGRESSION bridge (user-approved 2026-08-05): does the
# repaired LCA recipe (all_past + dense router scoring + full routing +
# zero dropout) transfer back to language modeling, or does it change the
# WikiText-2 quality-memory operating point measured under the old
# local-fiber recipe (endpoint_window + candidate_gather + topk16 +
# dropout 0.1)? NOTE: prefix supervision does not apply to LM — next-token
# loss is already dense all-position supervision.
#
# Island: the GOLD controlled operating point of the registered matrix —
# exact / B16 / L512, EPOCHS=10, lr=1e-4, warmup=1000, d_model=384, 6
# layers, 8 heads — same architecture operating point as the matrix.
#
# Rows (seed 0 first pass):
#   tokennodrop  matched token control under dropout=0
#   b25nodrop    old WT2 blur optimum (6 fine + 2 coarse) under repaired recipe
#   b75nodrop    LCA blur optimum (2 fine + 6 coarse) under repaired recipe
#   b75drop      b75 repaired fiber/scoring/routing but dropout=0.1 —
#                separates the dropout effect from the fiber/scoring effect
#
# Labels wt2rr_*; GRID_ROOT=out/wt2_recipe_regression. NEVER pooled with
# the registered sd_grid matrix — this is a bridge/control, not a
# replacement matrix. Interpretation guard (user): if the old frontier
# does not survive, say "the original WT2 frontier was measured under the
# local-fiber recipe; the repaired global-fiber recipe changes the
# quality-memory operating point" — NOT "the frontier was an artifact".
set -uo pipefail
cd "${REPO_ROOT:-$HOME/set-attention}"

HOST_TAG="${HOST_TAG:-blue}"
IMAGE="${IMAGE:-set-attention:latest}"
PROJECT="${PROJECT:-set-attention}"
L="${L:-512}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-0.0001}"
WARMUP="${WARMUP:-1000}"
SEEDS="${SEEDS:-0}"
ROWS="${ROWS:-tokennodrop b25nodrop b75nodrop b75drop}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DRY_RUN="${DRY_RUN:-0}"

GRID_ROOT="out/wt2_recipe_regression"
DONE_ROOT="${GRID_ROOT}/markers"
LOG_ROOT="logs/wt2_recipe_regression/${HOST_TAG}"
RESULT_TSV="${GRID_ROOT}/wt2_recipe_regression_${HOST_TAG}.tsv"
mkdir -p "$DONE_ROOT" "$LOG_ROOT"
[ -f "$RESULT_TSV" ] || printf "ts\thost\tlabel\tgpu\texit\tepochs\tpeak_vram_mib\ttrain_ppl\tval_ppl\tcsv\n" > "$RESULT_TSV"

SET_CFG="configs/set_dictionary/sd9_multiresolution.yaml"
TOKEN_CFG="configs/paper_lr_norm/baseline_dense_exact.yaml"
B25_GROUPS='[{name: fine, num_heads: 6, window_size: 2, stride: 1}, {name: coarse, num_heads: 2, window_size: 4, stride: 2}]'
B75_GROUPS='[{name: fine, num_heads: 2, window_size: 2, stride: 1}, {name: coarse, num_heads: 6, window_size: 4, stride: 2}]'
TOPK_FULL=$((L - 1))

csv_complete () {
  python3 - "$1" "$EPOCHS" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1]); exp = int(sys.argv[2])
if not p.exists():
    raise SystemExit(1)
try:
    with p.open(newline="") as fh:
        rows = list(csv.DictReader(line.replace("\0", "") for line in fh))
except Exception:
    raise SystemExit(1)
if len(rows) < exp:
    raise SystemExit(1)
row = rows[-1]
bad = {"nan", "inf", "-inf", "infinity", "-infinity"}
for key in ("val/ppl", "train/ppl", "train/peak_vram_mib"):
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

run_row () { # label seed gpu
  local label="$1" seed="$2" gpu="$3"
  local family groups drop cfg name csv log donemark lock container_name
  case "$label" in
    tokennodrop) family=token; drop=0 ;;
    b25nodrop)   family=set; groups="$B25_GROUPS"; drop=0 ;;
    b75nodrop)   family=set; groups="$B75_GROUPS"; drop=0 ;;
    b75drop)     family=set; groups="$B75_GROUPS"; drop=1 ;;
    *) echo "ERROR unknown row $label" >&2; return 1 ;;
  esac
  name="wt2rr_${label}_L${L}b${BATCH}_seed${seed}"
  csv="${GRID_ROOT}/${family}/L${L}/${name}.csv"
  log="${LOG_ROOT}/${name}.log"
  donemark="${DONE_ROOT}/${name}.done"
  lock="${DONE_ROOT}/${name}.lock"
  container_name="wt2rr_${HOST_TAG}_${name}"

  if [ -f "$donemark" ] || csv_complete "$csv"; then
    echo "SKIP done    $name"
    return 0
  fi
  if ! mkdir "$lock" 2>/dev/null; then
    echo "SKIP locked  $name"
    return 0
  fi

  local -a ov=(
    "training.output_dir=${csv%.csv}" logging.wandb.enable=false "logging.wandb.run_name=${name}"
    data.dataset=wikitext2 "data.batch_size=${BATCH}" "data.seq_len=${L}"
    "training.seed=${seed}" "training.epochs=${EPOCHS}" "training.lr=${LR}" "training.warmup_steps=${WARMUP}"
    training.deterministic=false training.benchmark_mode=false
    training.experiment_contract=sd_grid_seeded_v1 training.diagnostics_contract=current_matrix_v1
    model.d_model=384 model.dim_feedforward=1536 model.num_layers=6 model.num_heads=8
    "model.max_seq_len=${L}"
  )
  if [ "$family" = token ]; then
    cfg="$TOKEN_CFG"
    ov+=("model.attention_family=dense" "model.backend=exact")
  else
    cfg="$SET_CFG"
    ov+=("model.attention_family=dense" "model.backend=exact"
         model.d_phi=384 model.set_state_dim=384 model.feature_params.num_bins=128
         model.window_size=2 model.stride=1
         model.output_residual_mode=anchor_span model.token_mlp.enabled=false
         model.multiresolution.enabled=true "model.multiresolution.groups=${groups}"
         model.candidate_fiber=all_past model.router.score_mode=dense
         "model.router_topk=${TOPK_FULL}")
  fi
  if [ "$drop" = 0 ]; then
    ov+=("model.dropout=0.0" "model.attn_dropout=0.0"
         "model.resid_dropout=0.0" "model.ffn_dropout=0.0")
  fi

  echo "=== RUN $(date '+%F %T') gpu${gpu} $name (epochs=${EPOCHS}, L=${L}, B=${BATCH}, dropout_knob=${drop}) ==="
  if [ "$DRY_RUN" = 1 ]; then
    echo "PLAN  $name -> $csv"
    rmdir "$lock" 2>/dev/null || true
    return 0
  fi
  mkdir -p "$(dirname "$csv")"
  docker run --rm --name "$container_name" --gpus "device=${gpu}" --ipc=host \
    -u "$(id -u):$(id -g)" \
    -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e WANDB_MODE=offline \
    -e WANDB_PROJECT="${PROJECT}" -e WANDB_NAME="${name}" -e WANDB_RUN_GROUP="wt2rr_${family}_L${L}" \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" \
    /usr/bin/python scripts/run_experiment.py --config "$cfg" --csv-path "$csv" \
    --override "${ov[@]}" > "$log" 2>&1
  local rc=$?

  local peak tppl vppl ep
  ep=$(($( { wc -l < "$csv"; } 2>/dev/null || echo 1) - 1)); [ "$ep" -lt 0 ] && ep=0
  peak="$(csv_field "$csv" "train/peak_vram_mib")"
  tppl="$(csv_field "$csv" "train/ppl")"
  vppl="$(csv_field "$csv" "val/ppl")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "$HOST_TAG" "$name" "$gpu" "$rc" "$ep" "$peak" "$tppl" "$vppl" "$csv" \
    >> "$RESULT_TSV"

  if [ "$rc" -eq 0 ] && csv_complete "$csv" \
     && ! grep -qiE "traceback|out of memory|outofmemoryerror" "$log"; then
    touch "$donemark"
    echo "=== DONE $(date '+%F %T') $name epochs=${ep} peak=${peak}MiB val_ppl=${vppl} ==="
  else
    echo "=== FAIL $(date '+%F %T') rc=${rc} $name ==="
  fi
  rmdir "$lock" 2>/dev/null || true
  return 0
}

echo "=== WT2 RECIPE REGRESSION ${HOST_TAG}: ROWS='${ROWS}', SEEDS='${SEEDS}', L=${L}, B=${BATCH}, EPOCHS=${EPOCHS}, DRY_RUN=${DRY_RUN} ==="
for seed in $SEEDS; do
  set -- $ROWS
  while [ $# -gt 0 ]; do
    row1="$1"; shift
    row2="${1:-}"; [ $# -gt 0 ] && shift
    run_row "$row1" "$seed" "$GPU0" &
    p0=$!
    if [ -n "$row2" ]; then
      run_row "$row2" "$seed" "$GPU1" &
      p1=$!
      wait "$p0" "$p1"
    else
      wait "$p0"
    fi
  done
done
echo "=== WT2 RECIPE REGRESSION ${HOST_TAG} pass complete ==="
