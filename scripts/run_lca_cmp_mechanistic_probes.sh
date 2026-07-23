#!/usr/bin/env bash
# MRP-lca-cmp MECHANISTIC PROBE SERIES driver (user-approved 2026-07-20,
# docs/agent_plans/mrp_lca_cmp.md "Approved Diagnostic Probes").
#
# Rows (all L=1024, seed 0, native B4, grad_accum_steps=1,
# eval_microbatch_size=null, max_updates=2000; never pooled with matrix rows):
#   GPU0: P1 allpast_routerdense_fulltopk_probe  (b25, all_past, dense, topk=0)
#   GPU0: P2 prefixsup_b25_probe                 (b25, all_past, dense, topk=16,
#                                                 data.supervision=prefix)
#   GPU0: P3 oracle_b25_probe                    (b25, all_past, dense, topk=16,
#                                                 data.oracle_count_token=true)
#   GPU1: P2 prefixsup_token_probe               (token, data.supervision=prefix)
#
# Resumable: a row with a .done marker or a complete final CSV is skipped.
# Strict log scan (traceback/NaN/Inf/OOM) before any .done marker is written.
set -uo pipefail
cd "${REPO_ROOT:-$HOME/set-attention}"

HOST_TAG="${HOST_TAG:-lizmark}"
IMAGE="${IMAGE:-set-attention:latest}"
MAX_UPDATES="${MAX_UPDATES:-2000}"
DRY_RUN="${DRY_RUN:-0}"

GRID_ROOT="out/lca_cmp/calibration"
DONE_ROOT="${GRID_ROOT}/markers"
LOG_ROOT="logs/lca_cmp/${HOST_TAG}"
RESULT_TSV="${GRID_ROOT}/mechanistic_probes_${HOST_TAG}.tsv"
mkdir -p "$DONE_ROOT" "$LOG_ROOT"
[ -f "$RESULT_TSV" ] || printf "ts\thost\tlabel\tgpu\texit\tpeak_vram_mib\ttrain_loss\tval_loss\tval_acc\tcsv\n" > "$RESULT_TSV"

B25_GROUPS='[{name: fine, num_heads: 6, window_size: 2, stride: 1}, {name: coarse, num_heads: 2, window_size: 4, stride: 2}]'

csv_complete () { # csv -> 0 if final row has no nan/inf/empty metric
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

run_probe () { # label family gpu extra_overrides...
  local label="$1" family="$2" gpu="$3"; shift 3
  local cfg name csv log donemark
  if [ "$family" = token ]; then
    cfg="configs/lca_cmp/base_token.yaml"
  else
    cfg="configs/lca_cmp/base_set.yaml"
  fi
  name="lcacmp_${family}_L1024_seed0_${label}"
  csv="${GRID_ROOT}/${family}/L1024/${name}.csv"
  log="${LOG_ROOT}/${name}.log"
  donemark="${DONE_ROOT}/${family}_L1024_seed0_${label}.done"

  if [ -f "$donemark" ] || csv_complete "$csv"; then
    echo "SKIP done    $label"
    return 0
  fi

  local -a ov=(
    "data.seq_len=1024" "model.max_seq_len=1024"
    "data.batch_size=4"
    "training.grad_accum_steps=1"
    "training.eval_microbatch_size=null"
    "training.seed=0"
    "training.max_updates=${MAX_UPDATES}"
    "training.output_dir=${csv%.csv}"
    "logging.wandb.enable=false"
    "logging.wandb.run_name=${name}"
  )
  if [ "$family" != token ]; then
    ov+=("model.multiresolution.groups=${B25_GROUPS}")
  fi
  ov+=("$@")

  echo "=== RUN gpu${gpu} ${label} (${name}) ==="
  if [ "$DRY_RUN" = 1 ]; then
    echo "PLAN  $cfg --override ${ov[*]} -> $csv"
    return 0
  fi
  mkdir -p "$(dirname "$csv")"
  docker run --rm --name "lcacmp_${HOST_TAG}_${label}" --gpus "device=${gpu}" --ipc=host \
    -u "$(id -u):$(id -g)" \
    -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e WANDB_MODE=offline \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" \
    /usr/bin/python scripts/run_lca_cmp.py --config "$cfg" --csv-path "$csv" \
    --override "${ov[@]}" > "$log" 2>&1
  local rc=$?

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "$HOST_TAG" "$label" "$gpu" "$rc" \
    "$(csv_field "$csv" "train/peak_vram_mib")" \
    "$(csv_field "$csv" "train/loss")" \
    "$(csv_field "$csv" "val/loss")" \
    "$(csv_field "$csv" "val/accuracy")" \
    "$csv" >> "$RESULT_TSV"

  if [ "$rc" -eq 0 ] && csv_complete "$csv" \
     && ! grep -qiE "traceback|out of memory|outofmemoryerror|nan|inf" "$log"; then
    touch "$donemark"
    echo "=== DONE ${label} ==="
  else
    echo "=== FAIL ${label} rc=${rc} (no .done; rerunnable) ==="
    grep -inE "traceback|error|nan|inf|out of memory" "$log" | tail -n 5 || true
  fi
  return 0
}

worker_gpu0 () {
  # P1: full routing. router_topk=0 is rejected by config validation
  # ("learned router_topk must be >= 1"); the sanctioned full-softmax
  # mechanism is topk = max_sets (compatibility.py: "router_topk == max_sets
  # is equivalent to full softmax"). max_sets = 1023 for the L=1024 fine bank
  # (w=2, s=1, strict_past); 1023 also exceeds the coarse bank's 511 sets.
  run_probe allpast_routerdense_fulltopk_probe b25 0 \
    "model.candidate_fiber=all_past" "model.router.score_mode=dense" "model.router_topk=1023"
  run_probe prefixsup_b25_probe b25 0 \
    "model.candidate_fiber=all_past" "model.router.score_mode=dense" "data.supervision=prefix"
  run_probe oracle_b25_probe b25 0 \
    "model.candidate_fiber=all_past" "model.router.score_mode=dense" "data.oracle_count_token=true"
}

worker_gpu1 () {
  run_probe prefixsup_token_probe token 1 \
    "data.supervision=prefix"
}

echo "=== LCA-CMP MECHANISTIC PROBES ${HOST_TAG}: P1 + P2(token,b25) + P3, MAX_UPDATES=${MAX_UPDATES}, DRY_RUN=${DRY_RUN} ==="
if [ "$DRY_RUN" = 1 ]; then
  worker_gpu0
  worker_gpu1
else
  worker_gpu0 > "${LOG_ROOT}/probes_worker_gpu0.log" 2>&1 &
  pid0=$!
  worker_gpu1 > "${LOG_ROOT}/probes_worker_gpu1.log" 2>&1 &
  pid1=$!
  echo "workers: gpu0=$pid0 gpu1=$pid1"
  wait "$pid0" "$pid1"
fi
echo "=== LCA-CMP MECHANISTIC PROBES pass complete ==="
