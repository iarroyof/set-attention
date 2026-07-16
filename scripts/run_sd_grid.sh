#!/usr/bin/env bash
# Unified SD-9.x FULL-GRID driver — duplication-proof + resumable + OOM-mapping.
#
# Guarantees:
#  * 100% MUTUALLY EXCLUSIVE: every (family,backend,L,variant,seed) cell is assigned to exactly
#    ONE host (manifest column), and within a host an ATOMIC mkdir lock means no two workers/launchers
#    ever run the same cell. Safe to run alongside the legacy launchers: a cell that is already
#    complete (metadata scan), already done-marked, OOM-marked, or has a LIVE training process is skipped.
#  * 100% RESUMABLE: completed cells leave a .done marker AND are detected via metadata; a crashed/killed
#    cell releases its lock so a re-run retries it cleanly (CSV overwritten, no corruption). Re-running
#    the driver any number of times only ever advances missing cells.
#  * OOM-MAPPED: an uncontended OOM appends to oom_registry.tsv (the VRAM-ceiling map) and leaves a
#    .oom marker. An OOM with another CUDA workload present is non-terminal, is written to the separate
#    contention registry, and remains queued for a clean retry.
#  * GPU-ADMITTED: strict exclusivity is the default and current paper policy. An occupied or
#    unqueryable device defers before container creation, with a second check immediately before
#    `docker run`. Optional co-residency remains only for non-paper exploratory use when exclusivity
#    is explicitly disabled; it never creates terminal OOM evidence.
#  * 3 SEEDS default (SEEDS="0 1 2"); add seeds 3,4 later only for the best variants.
#  * DIAGNOSTICS preserved: uses the standard configs (full eval diagnostics + span ablation + VRAM in CSV)
#    and keeps offline wandb dirs.
#
# Usage:
#   HOST_TAG=blue    DRY_RUN=1 bash scripts/run_sd_grid.sh         # plan only (prove exclusivity)
#   HOST_TAG=blue    GPU0=0 GPU1=1 nohup bash scripts/run_sd_grid.sh > logs/sd_grid_blue.log 2>&1 &
#   HOST_TAG=lizmark GPU0=0 GPU1=1 nohup bash scripts/run_sd_grid.sh > logs/sd_grid_lizmark.log 2>&1 &
# Optional filters: ONLY_LENGTHS="8192 16384"  ONLY_FAMILY="set|token"  SEEDS="0 1 2 3 4"
# Isolated rerun: GRID_NAMESPACE=sd_grid_seeded_v1 RUN_TAG=seeded_v1
#                 REQUIRE_APPLIED_SEED=1 TRAINING_DETERMINISTIC=true
# Exact-dense grid profiles:
#   GRID_PROFILE=primary   -> completed registered matrix (default)
#   GRID_PROFILE=frontier  -> L2048/B3 bridge + L3584/{B4,B3} + L4096/B3
#   GRID_PROFILE=paper5    -> five-seed main-paper rows selected after frontier analysis
#   GRID_PROFILE=short_b3  -> missing short-context B3 bridge rows at L512,L1024
#   GRID_PROFILE=b2        -> fallback L2048/B2 bridge + L3584/L4096 B2
set -euo pipefail
cd "${REPO_ROOT:-$HOME/set-attention}"

HOST_TAG="${HOST_TAG:?set HOST_TAG=blue|lizmark}"
SEEDS="${SEEDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
GPU_WORKERS_PER_DEVICE="${GPU_WORKERS_PER_DEVICE:-1}"
IMAGE="${IMAGE:-set-attention:latest}"
PROJECT="${PROJECT:-set-attention}"
LR="${LR:-0.0001}"
EPOCHS="${EPOCHS:-10}"; WARMUP="${WARMUP:-1000}"
ONLY_LENGTHS="${ONLY_LENGTHS:-}"
ONLY_FAMILY="${ONLY_FAMILY:-}"
GRID_PROFILE="${GRID_PROFILE:-primary}"
GRID_NAMESPACE="${GRID_NAMESPACE:-sd_grid}"
RUN_TAG="${RUN_TAG:-}"
REQUIRE_APPLIED_SEED="${REQUIRE_APPLIED_SEED:-0}"
TRAINING_DETERMINISTIC="${TRAINING_DETERMINISTIC:-false}"
SET_CONFIG="configs/set_dictionary/sd9_multiresolution.yaml"
TOKEN_CONFIG_EXACT="configs/paper_lr_norm/baseline_dense_exact.yaml"
TOKEN_CONFIG_LANDMARK="configs/paper_lr_norm/baseline_linear_landmark.yaml"
TOKEN_CONFIG_SPARSE="configs/paper_lr_norm/baseline_sparse_local_band.yaml"
LOCK_TTL_MIN="${LOCK_TTL_MIN:-180}"   # stale-lock reclaim window (a live proc also protects it)
ALLOW_GPU_CORESIDENCY="${ALLOW_GPU_CORESIDENCY:-0}"
REQUIRE_EXCLUSIVE_GPU="${REQUIRE_EXCLUSIVE_GPU:-1}"
GPU_ADMISSION_HEADROOM_MIB="${GPU_ADMISSION_HEADROOM_MIB:-4096}"
GPU_PEAK_ESTIMATE_MARGIN_MIB="${GPU_PEAK_ESTIMATE_MARGIN_MIB:-1024}"
GPU_ADMISSION_RETRY_SEC="${GPU_ADMISSION_RETRY_SEC:-60}"
PEAK_REFERENCE_ROOT="${PEAK_REFERENCE_ROOT:-out/paper_mechanisms}"

[[ "$GRID_NAMESPACE" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "GRID_NAMESPACE must contain only letters, digits, underscores, or hyphens" >&2
  exit 2
}
if [ -n "$RUN_TAG" ]; then
  [[ "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || {
    echo "RUN_TAG must contain only letters, digits, underscores, or hyphens" >&2
    exit 2
  }
fi
if [ "$REQUIRE_APPLIED_SEED" = 1 ] && [ "$TRAINING_DETERMINISTIC" != true ]; then
  echo "REQUIRE_APPLIED_SEED=1 requires TRAINING_DETERMINISTIC=true" >&2
  exit 2
fi
for numeric_var in GPU_ADMISSION_HEADROOM_MIB GPU_PEAK_ESTIMATE_MARGIN_MIB GPU_ADMISSION_RETRY_SEC; do
  [[ "${!numeric_var}" =~ ^[0-9]+$ ]] || {
    echo "${numeric_var} must be a non-negative integer" >&2
    exit 2
  }
done
[[ "$GPU_WORKERS_PER_DEVICE" =~ ^[1-9][0-9]*$ ]] || {
  echo "GPU_WORKERS_PER_DEVICE must be a positive integer" >&2
  exit 2
}
case "$ALLOW_GPU_CORESIDENCY" in
  0|1) ;;
  *) echo "ALLOW_GPU_CORESIDENCY must be 0 or 1" >&2; exit 2 ;;
esac
case "$REQUIRE_EXCLUSIVE_GPU" in
  0|1) ;;
  *) echo "REQUIRE_EXCLUSIVE_GPU must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$REQUIRE_EXCLUSIVE_GPU" = 1 ] && [ "$ALLOW_GPU_CORESIDENCY" = 1 ]; then
  echo "REQUIRE_EXCLUSIVE_GPU=1 conflicts with ALLOW_GPU_CORESIDENCY=1" >&2
  exit 2
fi
case "$HOST_TAG" in
  blue|lizmark) ;;
  *) echo "HOST_TAG must be blue or lizmark" >&2; exit 2 ;;
esac

GRID_ROOT="out/paper_mechanisms/${GRID_NAMESPACE}"
LOCK_ROOT="${GRID_ROOT}/locks"; DONE_ROOT="${GRID_ROOT}/markers"
LOG_ROOT="logs/${GRID_NAMESPACE}/${HOST_TAG}"
OOM_REG="${GRID_ROOT}/oom_registry.tsv"
CONTENTION_OOM_REG="${GRID_ROOT}/contention_oom_registry.tsv"
GPU_ADMISSION_REG="${GRID_ROOT}/gpu_admission_${HOST_TAG}.tsv"
RESULT_TSV="${GRID_ROOT}/grid_runs_${HOST_TAG}.tsv"
FATAL_MARKER="${GRID_ROOT}/.fatal_config_error"
mkdir -p "$LOCK_ROOT" "$DONE_ROOT" "$LOG_ROOT" audit "$GRID_ROOT"
[ -f "$OOM_REG" ]    || printf "ts\thost\tfamily\tL\tvariant\tseed\tfine\tcoarse\tbackend\tbatch\tcoverage\tpeak_vram_mib\tnote\n" > "$OOM_REG"
[ -f "$CONTENTION_OOM_REG" ] || printf "ts\thost\tcell_id\tgpu\texpected_peak_mib\trequired_free_mib\tstart_free_mib\tend_free_mib\tstart_processes\tend_processes\tnote\n" > "$CONTENTION_OOM_REG"
[ -f "$GPU_ADMISSION_REG" ] || printf "ts\thost\tevent\tcell_id\tgpu\texpected_peak_mib\trequired_free_mib\ttotal_mib\tused_mib\tfree_mib\tcuda_processes\tgrid_processes\tco_resident\tmetric_scope\tnote\n" > "$GPU_ADMISSION_REG"
[ -f "$RESULT_TSV" ] || printf "ts\thost\tcell_id\tgpu\texit\tepochs\tcsv\n" > "$RESULT_TSV"

HOST_LOCK_DIR=""
DRIVER_BASHPID="$BASHPID"
declare -a WORKER_PIDS=()
cleanup_driver () {
  local pid container
  [ "$BASHPID" = "$DRIVER_BASHPID" ] || return 0
  trap - EXIT TERM INT HUP
  # Signal workers first so they cannot advance to another cell, then stop
  # their foreground containers to unblock the pending worker signal.
  for pid in "${WORKER_PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  while IFS= read -r container; do
    [ -n "$container" ] || continue
    docker stop -t 30 "$container" >/dev/null 2>&1 || true
  done < <(docker ps --filter "name=^sdgrid_${HOST_TAG}_" --format '{{.Names}}' 2>/dev/null || true)
  for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  [ -z "$HOST_LOCK_DIR" ] || rm -rf "$HOST_LOCK_DIR"
}
stop_driver () {
  cleanup_driver
  exit 143
}
trap cleanup_driver EXIT
trap stop_driver TERM INT HUP

# HOST-LEVEL SINGLE-GRID GUARD (robust): only one grid driver per host, ever. Prevents the
# double-grid overload (which caused batch contention OOMs) regardless of how it was launched
# (stray watcher, late-firing hung ssh, manual double-run). DRY_RUN is exempt so plans never block.
if [ "${DRY_RUN:-0}" != 1 ]; then
  HOST_LOCK_ROOT="out/paper_mechanisms/.grid_host_locks"
  mkdir -p "$HOST_LOCK_ROOT"
  if command -v flock >/dev/null 2>&1; then
    exec 8>"${HOST_LOCK_ROOT}/grid_${HOST_TAG}.lock"
    flock -n 8 || { echo "=== another grid driver already holds the ${HOST_TAG} host lock; exiting ==="; exit 0; }
  else
    # flock not installed (these hosts): atomic mkdir guard (race-free, unlike PID check-then-write)
    LOCKD="${HOST_LOCK_ROOT}/grid_${HOST_TAG}.lockd"
    if ! mkdir "$LOCKD" 2>/dev/null; then
      opid="$(cat "$LOCKD/pid" 2>/dev/null || true)"
      if [ -n "$opid" ] && kill -0 "$opid" 2>/dev/null; then
        echo "=== another grid driver (pid $opid) is live on ${HOST_TAG}; exiting ==="; exit 0
      fi
      echo "=== reclaiming stale host lock (owner $opid dead) ==="   # owner gone: take it over
    fi
    echo $$ > "$LOCKD/pid"
    HOST_LOCK_DIR="$LOCKD"
  fi
fi

# ---------------- grid manifest ----------------
# family L variant fine coarse host backend batch coverage   (token rows: fine/coarse = NA)
grid_rows () {
# DENSE-only matrix (2026-06-25). Landmark is CANCELED for any efficiency claim (not sub-quadratic:
# dense content-bias adapter [B,H,M,M], geom bias, causal mask upstream). All cells are exact/dense,
# arch d384/ff1536/L6, lr1e-4, warmup1000, 10 epochs, 3 seeds. Two islands, both internally controlled
# (batch is in the cell-id so they never collide):
#  (1) GOLD — exact/B16/L512 full blur + MATCHED dense token. The controlled operating point that
#      batch16 motivated; preserved and completed (b50/b62/b75 were the contention-OOM gaps).
#  (2) B4 cross-L — exact/B4 at L in {512,1024,2048,4096}, full blur + MATCHED dense token at EACH L,
#      so every set point has a same-config token at the same (backend,batch,L,arch). B4 fits to ~2048
#      on 24GB (blue) and ~4096 on 49GB (lizmark) -> blue owns L<=1024, lizmark owns L2048,4096. The
#      L512 overlap with the GOLD island bridges B16<->B4 (quantifies the batch offset). OOM rows map
#      the dense ceiling. This is the clean intuitive-picture test: does a set-vs-token gain hold/emerge
#      across L under fully matched conditions (the existing noisy token bucket cannot answer this).
# family L variant fine coarse host backend batch param
if [ "$GRID_PROFILE" = primary ]; then
cat <<'ROWS'
set 512 b0 8 0 blue exact 16 NA
set 512 b25 6 2 blue exact 16 NA
set 512 b50 4 4 blue exact 16 NA
set 512 b62 3 5 blue exact 16 NA
set 512 b75 2 6 blue exact 16 NA
set 512 b100 0 8 blue exact 16 NA
token 512 token NA NA blue exact 16 NA
set 512 b0 8 0 blue exact 4 NA
set 512 b25 6 2 blue exact 4 NA
set 512 b50 4 4 blue exact 4 NA
set 512 b62 3 5 blue exact 4 NA
set 512 b75 2 6 blue exact 4 NA
set 512 b100 0 8 blue exact 4 NA
token 512 token NA NA blue exact 4 NA
set 1024 b0 8 0 blue exact 4 NA
set 1024 b25 6 2 blue exact 4 NA
set 1024 b50 4 4 blue exact 4 NA
set 1024 b62 3 5 blue exact 4 NA
set 1024 b75 2 6 blue exact 4 NA
set 1024 b100 0 8 blue exact 4 NA
token 1024 token NA NA blue exact 4 NA
set 2048 b0 8 0 lizmark exact 4 NA
set 2048 b25 6 2 lizmark exact 4 NA
set 2048 b50 4 4 lizmark exact 4 NA
set 2048 b62 3 5 lizmark exact 4 NA
set 2048 b75 2 6 lizmark exact 4 NA
set 2048 b100 0 8 lizmark exact 4 NA
token 2048 token NA NA lizmark exact 4 NA
set 4096 b0 8 0 lizmark exact 4 NA
set 4096 b25 6 2 lizmark exact 4 NA
set 4096 b50 4 4 lizmark exact 4 NA
set 4096 b62 3 5 lizmark exact 4 NA
set 4096 b75 2 6 lizmark exact 4 NA
set 4096 b100 0 8 lizmark exact 4 NA
token 4096 token NA NA lizmark exact 4 NA
ROWS
elif [ "$GRID_PROFILE" = frontier ]; then
cat <<'ROWS'
set 2048 b0 8 0 blue exact 3 NA
set 2048 b25 6 2 blue exact 3 NA
set 2048 b50 4 4 blue exact 3 NA
set 2048 b62 3 5 blue exact 3 NA
set 2048 b75 2 6 blue exact 3 NA
set 2048 b100 0 8 blue exact 3 NA
token 2048 token NA NA blue exact 3 NA
set 3584 b0 8 0 lizmark exact 4 NA
set 3584 b25 6 2 lizmark exact 4 NA
set 3584 b50 4 4 lizmark exact 4 NA
set 3584 b62 3 5 lizmark exact 4 NA
set 3584 b75 2 6 lizmark exact 4 NA
set 3584 b100 0 8 lizmark exact 4 NA
token 3584 token NA NA lizmark exact 4 NA
set 3584 b0 8 0 lizmark exact 3 NA
set 3584 b25 6 2 lizmark exact 3 NA
set 3584 b50 4 4 lizmark exact 3 NA
set 3584 b62 3 5 lizmark exact 3 NA
set 3584 b75 2 6 lizmark exact 3 NA
set 3584 b100 0 8 lizmark exact 3 NA
token 3584 token NA NA lizmark exact 3 NA
set 4096 b0 8 0 lizmark exact 3 NA
set 4096 b25 6 2 lizmark exact 3 NA
set 4096 b50 4 4 lizmark exact 3 NA
set 4096 b62 3 5 lizmark exact 3 NA
set 4096 b75 2 6 lizmark exact 3 NA
set 4096 b100 0 8 lizmark exact 3 NA
token 4096 token NA NA lizmark exact 3 NA
ROWS
elif [ "$GRID_PROFILE" = paper5 ]; then
# Main-paper five-seed matrix. Run with SEEDS="0 1 2 3 4"; completed seeds are skipped.
# Every supported island uses the same set rows {b0,b25,b50,b75,b100} plus exact token. This regular
# matrix supports full within-island Pareto curves without post-hoc row selection. L4096/B4 excludes
# token/b0/b25 because all three are closed 3/3 OOM cells; its supported rows are b50/b75/b100.
cat <<'ROWS'
set 512 b0 8 0 blue exact 16 NA
set 512 b25 6 2 blue exact 16 NA
set 512 b50 4 4 blue exact 16 NA
set 512 b75 2 6 blue exact 16 NA
set 512 b100 0 8 blue exact 16 NA
token 512 token NA NA blue exact 16 NA
set 512 b0 8 0 blue exact 4 NA
set 512 b25 6 2 blue exact 4 NA
set 512 b50 4 4 blue exact 4 NA
set 512 b75 2 6 blue exact 4 NA
set 512 b100 0 8 blue exact 4 NA
token 512 token NA NA blue exact 4 NA
set 1024 b0 8 0 blue exact 4 NA
set 1024 b25 6 2 blue exact 4 NA
set 1024 b50 4 4 blue exact 4 NA
set 1024 b75 2 6 blue exact 4 NA
set 1024 b100 0 8 blue exact 4 NA
token 1024 token NA NA blue exact 4 NA
set 2048 b0 8 0 blue exact 3 NA
set 2048 b25 6 2 blue exact 3 NA
set 2048 b50 4 4 blue exact 3 NA
set 2048 b75 2 6 blue exact 3 NA
set 2048 b100 0 8 blue exact 3 NA
token 2048 token NA NA blue exact 3 NA
set 2048 b0 8 0 lizmark exact 4 NA
set 2048 b25 6 2 lizmark exact 4 NA
set 2048 b50 4 4 lizmark exact 4 NA
set 2048 b75 2 6 lizmark exact 4 NA
set 2048 b100 0 8 lizmark exact 4 NA
token 2048 token NA NA lizmark exact 4 NA
set 3584 b0 8 0 lizmark exact 4 NA
set 3584 b25 6 2 lizmark exact 4 NA
set 3584 b50 4 4 lizmark exact 4 NA
set 3584 b75 2 6 lizmark exact 4 NA
set 3584 b100 0 8 lizmark exact 4 NA
token 3584 token NA NA lizmark exact 4 NA
set 3584 b0 8 0 lizmark exact 3 NA
set 3584 b25 6 2 lizmark exact 3 NA
set 3584 b50 4 4 lizmark exact 3 NA
set 3584 b75 2 6 lizmark exact 3 NA
set 3584 b100 0 8 lizmark exact 3 NA
token 3584 token NA NA lizmark exact 3 NA
set 4096 b0 8 0 lizmark exact 3 NA
set 4096 b25 6 2 lizmark exact 3 NA
set 4096 b50 4 4 lizmark exact 3 NA
set 4096 b75 2 6 lizmark exact 3 NA
set 4096 b100 0 8 lizmark exact 3 NA
token 4096 token NA NA lizmark exact 3 NA
set 4096 b50 4 4 lizmark exact 4 NA
set 4096 b75 2 6 lizmark exact 4 NA
set 4096 b100 0 8 lizmark exact 4 NA
ROWS
elif [ "$GRID_PROFILE" = short_b3 ]; then
# Short-context B3 bridge for the full exact-dense picture. This fills the
# missing intermediate-batch islands between B4 and the larger/smaller
# operating points without changing the registered paper5 profile.
cat <<'ROWS'
set 512 b0 8 0 lizmark exact 3 NA
set 512 b25 6 2 lizmark exact 3 NA
set 512 b50 4 4 lizmark exact 3 NA
set 512 b75 2 6 lizmark exact 3 NA
set 512 b100 0 8 lizmark exact 3 NA
token 512 token NA NA lizmark exact 3 NA
set 1024 b0 8 0 blue exact 3 NA
set 1024 b25 6 2 blue exact 3 NA
set 1024 b50 4 4 blue exact 3 NA
set 1024 b75 2 6 blue exact 3 NA
set 1024 b100 0 8 blue exact 3 NA
token 1024 token NA NA blue exact 3 NA
ROWS
elif [ "$GRID_PROFILE" = b2 ]; then
cat <<'ROWS'
set 2048 b0 8 0 blue exact 2 NA
set 2048 b25 6 2 blue exact 2 NA
set 2048 b50 4 4 blue exact 2 NA
set 2048 b62 3 5 blue exact 2 NA
set 2048 b75 2 6 blue exact 2 NA
set 2048 b100 0 8 blue exact 2 NA
token 2048 token NA NA blue exact 2 NA
set 3584 b0 8 0 lizmark exact 2 NA
set 3584 b25 6 2 lizmark exact 2 NA
set 3584 b50 4 4 lizmark exact 2 NA
set 3584 b62 3 5 lizmark exact 2 NA
set 3584 b75 2 6 lizmark exact 2 NA
set 3584 b100 0 8 lizmark exact 2 NA
token 3584 token NA NA lizmark exact 2 NA
set 4096 b0 8 0 lizmark exact 2 NA
set 4096 b25 6 2 lizmark exact 2 NA
set 4096 b50 4 4 lizmark exact 2 NA
set 4096 b62 3 5 lizmark exact 2 NA
set 4096 b75 2 6 lizmark exact 2 NA
set 4096 b100 0 8 lizmark exact 2 NA
token 4096 token NA NA lizmark exact 2 NA
ROWS
else
  echo "unsupported GRID_PROFILE: $GRID_PROFILE" >&2
  return 2
fi
}

# ---------------- done-set from metadata ----------------
declare -A DONE
if [ "$GRID_NAMESPACE" = sd_grid ]; then
  STATUS_SCAN_ROOT="out/paper_mechanisms"
else
  STATUS_SCAN_ROOT="$GRID_ROOT"
fi
STATUS_SNAPSHOT="${GRID_ROOT}/status_preflight_${HOST_TAG}.tsv"
if [ "$REQUIRE_APPLIED_SEED" = 1 ]; then
  SD_GRID_TARGET_EPOCHS="${EPOCHS}" \
  SD_GRID_REQUIRE_CONTRACT=sd_grid_seeded_v1 \
    python3 scripts/sd_grid_status.py "$STATUS_SCAN_ROOT" > "$STATUS_SNAPSHOT"
else
  SD_GRID_TARGET_EPOCHS="${EPOCHS}" \
    python3 scripts/sd_grid_status.py "$STATUS_SCAN_ROOT" > "$STATUS_SNAPSHOT"
fi
while IFS=$'\t' read -r cid ep _rest; do
  [[ "${ep:-0}" =~ ^[0-9]+$ ]] && [ "${ep}" -ge "${EPOCHS}" ] && DONE["$cid"]=1
done < "$STATUS_SNAPSHOT"

# Peak admission references deliberately accept legacy rows here: a seed/config
# defect can invalidate a PPL estimate without invalidating its measured allocation
# peak. Only complete rows are used, and the maximum across seeds is retained.
PEAK_REFERENCE_SNAPSHOT="${GRID_ROOT}/peak_reference_${HOST_TAG}.tsv"
SD_GRID_TARGET_EPOCHS="${EPOCHS}" \
  python3 scripts/sd_grid_status.py "$PEAK_REFERENCE_ROOT" \
    > "$PEAK_REFERENCE_SNAPSHOT" \
    2> "${GRID_ROOT}/peak_reference_${HOST_TAG}.warnings.log"
declare -A PEAK_ESTIMATE
while IFS=$'\t' read -r ref_cid ref_ep _ref_ppl ref_peak _rest; do
  [[ "${ref_ep:-0}" =~ ^[0-9]+$ ]] && [ "$ref_ep" -ge "$EPOCHS" ] || continue
  [[ "${ref_peak:-}" =~ ^[0-9]+([.][0-9]+)?$ ]] || continue
  ref_key="${ref_cid%|*}"
  ref_peak_int=$((${ref_peak%.*} + 1))
  if [ -z "${PEAK_ESTIMATE[$ref_key]:-}" ] || [ "$ref_peak_int" -gt "${PEAK_ESTIMATE[$ref_key]}" ]; then
    PEAK_ESTIMATE["$ref_key"]="$ref_peak_int"
  fi
done < "$PEAK_REFERENCE_SNAPSHOT"

groups_yaml () { # fine coarse
  local f="$1" c="$2"
  if   [ "$f" -gt 0 ] && [ "$c" -gt 0 ]; then printf '[{name: fine, num_heads: %s, window_size: 2, stride: 1}, {name: coarse, num_heads: %s, window_size: 4, stride: 2}]' "$f" "$c"
  elif [ "$f" -gt 0 ]; then printf '[{name: fine, num_heads: %s, window_size: 2, stride: 1}]' "$f"
  else printf '[{name: coarse, num_heads: %s, window_size: 4, stride: 2}]' "$c"; fi
}

cell_inflight () { # L batch seed signature
  local L="$1" batch="$2" seed="$3" sig="$4"
  ps -eo args 2>/dev/null | grep -F "run_experiment.py" | grep -F "data.seq_len=${L}" \
    | grep -F "data.batch_size=${batch}" | grep -F "training.seed=${seed}" \
    | grep -F -- "$sig" | grep -qv "grep -F"
}

csv_complete () { # csv epochs requested_seed require_applied_seed family fine coarse
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" <<'PY'
import csv,sys
from pathlib import Path
p=Path(sys.argv[1]); exp=int(sys.argv[2]); seed=int(sys.argv[3]); require=sys.argv[4]=="1"
family=sys.argv[5]; fine=int(sys.argv[6]) if sys.argv[6] != "NA" else 0
coarse=int(sys.argv[7]) if sys.argv[7] != "NA" else 0
if not p.exists(): raise SystemExit(1)
try:
    with p.open(newline="") as fh: rows=list(csv.DictReader(l.replace("\0","") for l in fh))
except Exception: raise SystemExit(1)
complete=len(rows)>=exp and rows[-1].get("epoch")==str(exp)
if require and complete:
    row=rows[-1]
    def present(key):
        return str(row.get(key, "")).strip().lower() not in {"", "na", "none", "null"}
    complete=(
        row.get("training.seed_applied", "").lower() in {"true", "1"}
        and row.get("training.applied_seed") == str(seed)
        and row.get("training.torch_initial_seed") == str(seed)
        and row.get("training.deterministic", "").lower() in {"true", "1"}
        and row.get("training.benchmark_mode", "").lower() in {"false", "0"}
        and row.get("training.experiment_contract") == "sd_grid_seeded_v1"
        and row.get("training.diagnostics_contract") == "current_matrix_v1"
    )
    required=[
        "val/ppl",
        "train/peak_vram_mib",
        "val/loss_early_freq",
        "val/loss_early_rare",
        "val/loss_late_freq",
        "val/loss_late_rare",
    ]
    if family == "token":
        required += [
            "baseline/attention_entropy_mean",
            "baseline/attention_top1_mean",
            "baseline/attention_gradient_norm",
            "baseline/attention_param_norm",
        ]
    else:
        required += ["val/span_ablation_delta_ppl"]
        for group, heads in (("fine", fine), ("coarse", coarse)):
            if heads <= 0:
                continue
            required += [
                f"val/span_ablation_{group}_delta_ppl",
                f"val/effective_range_{group}",
                f"val/routing_entropy_{group}",
                f"val/routing_top1_{group}",
                f"ausa/{group}/routing_entropy_norm",
                f"ausa/{group}/router_top1_weight",
                f"ausa/{group}/pooling_effective_support",
                f"ausa/{group}/router_gradient_norm",
                f"ausa/{group}/router_param_norm",
                f"ausa/{group}/grad_norm_token_pre_pool",
                f"ausa/{group}/grad_norm_set_post_pool",
                f"ausa/{group}/grad_norm_set_post_blocks",
            ]
    complete = complete and all(present(key) for key in required)
raise SystemExit(0 if complete else 1)
PY
}

snapshot_gpu () { # physical GPU index; writes GPU_* globals
  local gpu="$1" raw pid mem cmd
  GPU_TOTAL_MIB=0
  GPU_USED_MIB=0
  GPU_FREE_MIB=0
  GPU_PROCESS_COUNT=0
  GPU_GRID_PROCESS_COUNT=0
  GPU_PROCESSES="none"
  if ! raw="$(nvidia-smi --id="$gpu" \
      --query-gpu=memory.total,memory.used,memory.free \
      --format=csv,noheader,nounits 2>/dev/null)"; then
    return 1
  fi
  IFS=',' read -r GPU_TOTAL_MIB GPU_USED_MIB GPU_FREE_MIB <<< "$raw"
  GPU_TOTAL_MIB="${GPU_TOTAL_MIB//[[:space:]]/}"
  GPU_USED_MIB="${GPU_USED_MIB//[[:space:]]/}"
  GPU_FREE_MIB="${GPU_FREE_MIB//[[:space:]]/}"
  while IFS=',' read -r pid mem; do
    pid="${pid//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"
    [ -n "$pid" ] || continue
    GPU_PROCESS_COUNT=$((GPU_PROCESS_COUNT + 1))
    if [ "$GPU_PROCESSES" = none ]; then GPU_PROCESSES="${pid}:${mem}"; else GPU_PROCESSES="${GPU_PROCESSES},${pid}:${mem}"; fi
    cmd="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    case "$cmd" in
      *scripts/run_experiment.py*) GPU_GRID_PROCESS_COUNT=$((GPU_GRID_PROCESS_COUNT + 1)) ;;
    esac
  done < <(nvidia-smi --id="$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
}

log_gpu_admission () { # event cid gpu expected required co_resident note
  local event="$1" cid="$2" gpu="$3" expected="$4" required="$5" co_resident="$6" note="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "$HOST_TAG" "$event" "$cid" "$gpu" "$expected" "$required" \
    "$GPU_TOTAL_MIB" "$GPU_USED_MIB" "$GPU_FREE_MIB" "$GPU_PROCESSES" \
    "$GPU_GRID_PROCESS_COUNT" "$co_resident" \
    "$([ "$co_resident" = 1 ] && echo ppl_and_allocated_peak_only || echo all_metrics)" "$note" \
    >> "$GPU_ADMISSION_REG"
}

admit_gpu () { # cid gpu; sets ADMISSION_* globals, returns 75 when deferred
  local cid="$1" gpu="$2" key="${1%|*}" expected required note
  expected="${PEAK_ESTIMATE[$key]:-NA}"
  if [ "$expected" = NA ]; then
    # Short-B3 bridge rows use the same exact-dense architecture as the
    # completed B4 rows, at a smaller batch. Use B4 as a conservative admission
    # reference so co-resident scheduling is still memory-gated.
    case "$key" in
      *"|b3") expected="${PEAK_ESTIMATE[${key%|b3}|b4]:-NA}" ;;
    esac
  fi
  required="NA"
  if [ "$expected" != NA ]; then
    required=$((expected + GPU_PEAK_ESTIMATE_MARGIN_MIB + GPU_ADMISSION_HEADROOM_MIB))
  fi
  if ! snapshot_gpu "$gpu"; then
    GPU_TOTAL_MIB=NA GPU_USED_MIB=NA GPU_FREE_MIB=NA
    GPU_PROCESSES=unknown GPU_GRID_PROCESS_COUNT=NA
    log_gpu_admission defer "$cid" "$gpu" "$expected" "$required" 0 "nvidia-smi query failed"
    return 75
  fi

  ADMISSION_EXPECTED_PEAK_MIB="$expected"
  ADMISSION_REQUIRED_FREE_MIB="$required"
  ADMISSION_START_FREE_MIB="$GPU_FREE_MIB"
  ADMISSION_START_PROCESSES="$GPU_PROCESSES"
  ADMISSION_CORESIDENT=0

  if [ "$GPU_PROCESS_COUNT" -eq 0 ]; then
    log_gpu_admission admit "$cid" "$gpu" "$expected" "$required" 0 "exclusive GPU"
    return 0
  fi
  if [ "$REQUIRE_EXCLUSIVE_GPU" = 1 ]; then
    log_gpu_admission defer "$cid" "$gpu" "$expected" "$required" 0 "strict exclusivity: another CUDA process is using this GPU"
    return 75
  fi
  if [ "$ALLOW_GPU_CORESIDENCY" != 1 ]; then
    log_gpu_admission defer "$cid" "$gpu" "$expected" "$required" 0 "co-residency disabled"
    return 75
  fi
  if [ "$expected" = NA ]; then
    log_gpu_admission defer "$cid" "$gpu" "$expected" "$required" 0 "no complete matching peak reference"
    return 75
  fi
  if [ "$GPU_FREE_MIB" -lt "$required" ]; then
    note="free ${GPU_FREE_MIB} MiB is below estimated peak ${expected} + estimate margin ${GPU_PEAK_ESTIMATE_MARGIN_MIB} + headroom ${GPU_ADMISSION_HEADROOM_MIB}"
    log_gpu_admission defer "$cid" "$gpu" "$expected" "$required" 0 "$note"
    return 75
  fi

  ADMISSION_CORESIDENT=1
  log_gpu_admission admit "$cid" "$gpu" "$expected" "$required" 1 "co-resident; exclude wall-time and throughput"
  return 0
}

process_cell () { # family L variant fine coarse backend batch coverage seed gpu
  local family="$1" L="$2" variant="$3" fine="$4" coarse="$5" backend="$6" batch="$7" cov="$8" seed="$9" gpu="${10}"
  local cid name csv log donemark oommark lock sig container_name
  if [ -f "$FATAL_MARKER" ]; then
    echo "STOP fatal config marker present: $FATAL_MARKER"
    return 99
  fi
  if [ "$family" = set ]; then
    cid="set|${backend}|${L}|f${fine}c${coarse}|b${batch}|${seed}"
    name="sdgrid${RUN_TAG:+_${RUN_TAG}}_set_${variant}_L${L}_${backend}_b${batch}_seed${seed}"
    sig="$(groups_yaml "$fine" "$coarse")"
  else
    cid="token|${backend}|${L}|token|b${batch}|${seed}"
    name="sdgrid${RUN_TAG:+_${RUN_TAG}}_token_${backend}_L${L}_b${batch}_seed${seed}"
    case "$backend" in
      exact)                  sig="baseline_dense_exact.yaml" ;;
      landmark)               sig="baseline_linear_landmark.yaml" ;;
      local_band|sparse_topk) sig="baseline_sparse_local_band.yaml" ;;
      *) echo "unsupported token backend: $backend"; return 2 ;;
    esac
  fi
  local safe="${cid//[|\/]/_}"
  csv="${GRID_ROOT}/${family}/L${L}/${name}.csv"
  log="${LOG_ROOT}/${name}.log"
  donemark="${DONE_ROOT}/${safe}.done"; oommark="${DONE_ROOT}/${safe}.oom"; lock="${LOCK_ROOT}/${safe}.lock"
  container_name="sdgrid_${HOST_TAG}_${safe}"

  if [ -n "${DONE[$cid]:-}" ] || [ -f "$donemark" ]; then echo "SKIP done    $cid"; return 0; fi
  if [ -f "$oommark" ]; then echo "SKIP oom     $cid (recorded ceiling)"; return 0; fi
  if cell_inflight "$L" "$batch" "$seed" "$sig"; then echo "SKIP inflight $cid (legacy/other launcher)"; return 0; fi

  if ! mkdir "$lock" 2>/dev/null; then
    if [ -z "$(find "$lock" -mmin -"${LOCK_TTL_MIN}" 2>/dev/null)" ] && \
       ! cell_inflight "$L" "$batch" "$seed" "$sig"; then
      echo "RECLAIM stale lock $cid"; rmdir "$lock" 2>/dev/null || true
      mkdir "$lock" 2>/dev/null || { echo "SKIP locked  $cid"; return 0; }
    else
      echo "SKIP locked  $cid"; return 0
    fi
  fi
  # shellcheck disable=SC2064
  trap "rmdir '$lock' 2>/dev/null || true" RETURN

  if [ "$DRY_RUN" = 1 ]; then echo "PLAN  run gpu${gpu}  $cid  -> $csv"; return 0; fi
  local admission_rc
  if admit_gpu "$cid" "$gpu"; then
    admission_rc=0
  else
    admission_rc=$?
    echo "DEFER gpu${gpu} $cid (admission rc=${admission_rc})"
    return "$admission_rc"
  fi
  echo "=== RUN gpu${gpu} $cid ==="
  mkdir -p "$(dirname "$csv")"

  local -a ov
  if [ "$family" = set ]; then
    local fam="linear"
    case "$backend" in exact) fam="dense";; local_band|sparse_topk) fam="sparse";; esac
    ov=( "training.output_dir=${csv%.csv}" logging.wandb.enable=false "logging.wandb.run_name=${name}"
         data.dataset=wikitext2 "data.batch_size=${batch}" "data.seq_len=${L}"
         "training.seed=${seed}" "training.epochs=${EPOCHS}" "training.lr=${LR}" "training.warmup_steps=${WARMUP}"
         "training.deterministic=${TRAINING_DETERMINISTIC}" training.benchmark_mode=false
         training.experiment_contract=sd_grid_seeded_v1 training.diagnostics_contract=current_matrix_v1
         "model.attention_family=${fam}" "model.backend=${backend}" "model.max_seq_len=${L}"
         model.d_model=384 model.dim_feedforward=1536 model.num_layers=6 model.num_heads=8
         model.d_phi=384 model.set_state_dim=384 model.feature_params.num_bins=128
         "model.window_size=$([ "$fine" -gt 0 ] && echo 2 || echo 4)" "model.stride=$([ "$fine" -gt 0 ] && echo 1 || echo 2)"
         model.output_residual_mode=anchor_span model.token_mlp.enabled=false
         model.multiresolution.enabled=true "model.multiresolution.groups=${sig}" )
    # column 9 (cov) is backend-specific: landmark coverage, or local_band radius
    case "$backend" in
      landmark)   ov+=( "model.backend_params.landmark_coverage=${cov}" );;
      local_band) ov+=( "model.backend_params.radius=${cov}" );;
    esac
  else
    local tfam="linear"
    case "$backend" in exact) tfam="dense";; local_band|sparse_topk) tfam="sparse";; esac
    ov=( "training.output_dir=${csv%.csv}" logging.wandb.enable=false "logging.wandb.run_name=${name}"
         data.dataset=wikitext2 "data.batch_size=${batch}" "data.seq_len=${L}"
         "training.seed=${seed}" "training.epochs=${EPOCHS}" "training.lr=${LR}" training.warmup_steps="${WARMUP}"
         "training.deterministic=${TRAINING_DETERMINISTIC}" training.benchmark_mode=false
         training.experiment_contract=sd_grid_seeded_v1 training.diagnostics_contract=current_matrix_v1
         "model.attention_family=${tfam}" "model.backend=${backend}"
         model.d_model=384 model.dim_feedforward=1536 model.num_layers=6 model.num_heads=8
         "model.max_seq_len=${L}" )
    case "$backend" in
      landmark)   ov+=( "model.backend_params.landmark_coverage=${cov}" );;
      local_band) ov+=( "model.backend_params.radius=${cov}" );;
    esac
  fi
  local cfg="$SET_CONFIG"
  if [ "$family" = token ]; then
    case "$backend" in
      exact)                  cfg="$TOKEN_CONFIG_EXACT" ;;
      landmark)               cfg="$TOKEN_CONFIG_LANDMARK" ;;
      local_band|sparse_topk) cfg="$TOKEN_CONFIG_SPARSE" ;;
      *) echo "unsupported token backend: $backend"; return 2 ;;
    esac
  fi

  # Recheck immediately before container creation. This cannot control an
  # arbitrary concurrent Docker client, but it closes the launcher-local
  # check/config-resolution window.
  if [ "$REQUIRE_EXCLUSIVE_GPU" = 1 ]; then
    if ! snapshot_gpu "$gpu"; then
      GPU_TOTAL_MIB=NA GPU_USED_MIB=NA GPU_FREE_MIB=NA
      GPU_PROCESSES=unknown GPU_GRID_PROCESS_COUNT=NA
      log_gpu_admission prestart_defer "$cid" "$gpu" "$ADMISSION_EXPECTED_PEAK_MIB" "$ADMISSION_REQUIRED_FREE_MIB" 0 "strict exclusivity: prestart nvidia-smi query failed"
      return 75
    fi
    if [ "$GPU_PROCESS_COUNT" -ne 0 ]; then
      log_gpu_admission prestart_defer "$cid" "$gpu" "$ADMISSION_EXPECTED_PEAK_MIB" "$ADMISSION_REQUIRED_FREE_MIB" 0 "strict exclusivity: GPU became occupied before docker run"
      return 75
    fi
  fi

  # A driver/worker stop must stop the active container as well. Without this,
  # killing a detached queue can orphan docker runs that bypass the host lock.
  trap "docker stop -t 30 '$container_name' >/dev/null 2>&1 || true; rmdir '$lock' 2>/dev/null || true; exit 143" TERM INT HUP
  set +e
  docker run --rm --name "$container_name" --gpus "device=${gpu}" --ipc=host -u "$(id -u):$(id -g)" \
    -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e WANDB_MODE=offline \
    -e WANDB_PROJECT="${PROJECT}" -e WANDB_NAME="${name}" -e WANDB_RUN_GROUP="sdgrid_${family}_L${L}" \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" \
    /usr/bin/python scripts/run_experiment.py --config "${cfg}" --csv-path "${csv}" --override "${ov[@]}" \
    > "$log" 2>&1
  local rc=$?
  set -e
  trap stop_driver TERM INT HUP

  local ep; ep=$(($( { wc -l < "$csv"; } 2>/dev/null || echo 1) - 1)); [ "$ep" -lt 0 ] && ep=0
  local end_free="NA" end_processes="unknown" end_process_count=0 end_snapshot_ok=0
  local run_co_resident="$ADMISSION_CORESIDENT"
  if snapshot_gpu "$gpu"; then
    end_snapshot_ok=1
    end_free="$GPU_FREE_MIB"
    end_processes="$GPU_PROCESSES"
    end_process_count="$GPU_PROCESS_COUNT"
    [ "$end_process_count" -eq 0 ] || run_co_resident=1
  else
    # Unknown end occupancy can never certify an exclusive capacity failure.
    run_co_resident=1
    GPU_TOTAL_MIB=NA GPU_USED_MIB=NA GPU_FREE_MIB=NA
    GPU_PROCESSES=unknown GPU_GRID_PROCESS_COUNT=NA
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$HOST_TAG" "$cid" "$gpu" "$rc" "$ep" "$csv" >> "$RESULT_TSV"
  if [ "$rc" -eq 0 ] && csv_complete "$csv" "$EPOCHS" "$seed" "$REQUIRE_APPLIED_SEED" "$family" "$fine" "$coarse"; then
    touch "$donemark"
    log_gpu_admission finish "$cid" "$gpu" "$ADMISSION_EXPECTED_PEAK_MIB" "$ADMISSION_REQUIRED_FREE_MIB" "$run_co_resident" "success rc=0 epochs=${ep}"
    echo "=== DONE $cid (${ep}ep, co_resident=${run_co_resident}) ==="
  elif grep -qE "config[.]schema[.]ConfigError|training[.]experiment_contract|sd_grid_seeded_v1:" "$log" 2>/dev/null; then
    printf "%s\n" "$cid" > "$FATAL_MARKER"
    echo "=== FATAL CONFIG $cid rc=$rc -> stopping host queue ==="
    return 99
  elif grep -qiE "out of memory|outofmemoryerror|CUDA error: out of memory|CUBLAS_STATUS_ALLOC" "$log" 2>/dev/null; then
    local pk; pk=$(python3 - "$csv" <<'PY' 2>/dev/null || echo NA
import csv,sys
from pathlib import Path
p=Path(sys.argv[1])
try:
    with p.open(newline="") as fh: r=list(csv.DictReader(l.replace("\0","") for l in fh))
    print(r[-1].get("train/peak_vram_mib","NA") if r else "NA")
except Exception: print("NA")
PY
)
    if [ "$run_co_resident" = 1 ]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date '+%F %T')" "$HOST_TAG" "$cid" "$gpu" "$ADMISSION_EXPECTED_PEAK_MIB" \
        "$ADMISSION_REQUIRED_FREE_MIB" "$ADMISSION_START_FREE_MIB" "$end_free" \
        "$ADMISSION_START_PROCESSES" "$end_processes" \
        "NON_TERMINAL_CONTENTION_OR_UNKNOWN_OOM; end_snapshot_ok=${end_snapshot_ok}; no .oom marker; retry when admitted" \
        >> "$CONTENTION_OOM_REG"
      log_gpu_admission contention_oom "$cid" "$gpu" "$ADMISSION_EXPECTED_PEAK_MIB" "$ADMISSION_REQUIRED_FREE_MIB" 1 "non-terminal; retry required"
      echo "=== CONTENTION OOM $cid -> deferred; not a VRAM ceiling ==="
    else
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$HOST_TAG" "$family" "$L" "$variant" "$seed" "$fine" "$coarse" "$backend" "$batch" "$cov" "$pk" "exclusive OOM" >> "$OOM_REG"
      touch "$oommark"
      log_gpu_admission exclusive_oom "$cid" "$gpu" "$ADMISSION_EXPECTED_PEAK_MIB" "$ADMISSION_REQUIRED_FREE_MIB" 0 "terminal VRAM ceiling"
      echo "=== OOM  $cid -> registry (exclusive VRAM ceiling) ==="
    fi
  else
    log_gpu_admission finish "$cid" "$gpu" "$ADMISSION_EXPECTED_PEAK_MIB" "$ADMISSION_REQUIRED_FREE_MIB" "$run_co_resident" "failure rc=${rc} epochs=${ep}; retry required"
    echo "=== FAIL $cid rc=$rc (lock released; will retry next run) ==="
  fi
}

# Validate both family configs through the exact runner path before workers can
# iterate the matrix. This catches YAML/override type mismatches and contract
# drift without creating a training artifact.
if [ "$DRY_RUN" != 1 ] && [ "$REQUIRE_APPLIED_SEED" = 1 ]; then
  common_preflight=(
    training.seed=0 "training.epochs=${EPOCHS}" "training.lr=${LR}"
    "training.warmup_steps=${WARMUP}" training.deterministic=true
    training.benchmark_mode=false
    training.experiment_contract=sd_grid_seeded_v1
    training.diagnostics_contract=current_matrix_v1
    logging.wandb.enable=false
  )
  for preflight_cfg in "$SET_CONFIG" "$TOKEN_CONFIG_EXACT"; do
    preflight_name="$(basename "$preflight_cfg" .yaml)"
    docker run --rm --ipc=host -u "$(id -u):$(id -g)" \
      -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache \
      -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e WANDB_MODE=offline \
      -v "${PWD}:/workspace" -w /workspace "${IMAGE}" \
      /usr/bin/python scripts/run_experiment.py --config "$preflight_cfg" \
      --dry-run --override "${common_preflight[@]}" \
      > "${GRID_ROOT}/contract_preflight_${preflight_name}.log" 2>&1
  done
fi

# ---------------- enumerate this host's pending cells, split across GPUs ----------------
mapfile -t ROWS < <(grid_rows)
declare -a CELLS=()
for row in "${ROWS[@]}"; do
  read -r family L variant fine coarse host backend batch cov <<< "$row"
  [ "$host" = "$HOST_TAG" ] || continue
  [ -z "$ONLY_FAMILY" ] || [ "$ONLY_FAMILY" = "$family" ] || continue
  if [ -n "$ONLY_LENGTHS" ]; then case " $ONLY_LENGTHS " in *" $L "*) ;; *) continue;; esac; fi
  for seed in $SEEDS; do CELLS+=("$family $L $variant $fine $coarse $backend $batch $cov $seed"); done
done

echo "=== SD-GRID ${HOST_TAG} profile=${GRID_PROFILE}: ${#CELLS[@]} candidate cells, SEEDS='${SEEDS}', DRY_RUN=${DRY_RUN}, GPU_WORKERS_PER_DEVICE=${GPU_WORKERS_PER_DEVICE} ==="
worker () { # gpu slot total_slots
  local gpu="$1" slot="$2" total_slots="$3" i c rc deferred
  while true; do
    i=0
    deferred=0
    for c in "${CELLS[@]}"; do
      if [ $((i % total_slots)) -eq "$slot" ]; then
        # shellcheck disable=SC2086
        if process_cell $c "$gpu"; then rc=0; else rc=$?; fi
        case "$rc" in
          75) deferred=$((deferred + 1)) ;;
          99) return 99 ;;
        esac
      fi
      i=$((i + 1))
    done
    [ "$deferred" -gt 0 ] || break
    echo "=== GPU${gpu}: ${deferred} cells admission-deferred; retrying in ${GPU_ADMISSION_RETRY_SEC}s ==="
    sleep "$GPU_ADMISSION_RETRY_SEC"
  done
}
if [ "$DRY_RUN" = 1 ]; then
  total_slots=$((GPU_WORKERS_PER_DEVICE * 2))
  for ((slot=0; slot<GPU_WORKERS_PER_DEVICE; slot++)); do worker "$GPU0" "$slot" "$total_slots"; done
  for ((slot=0; slot<GPU_WORKERS_PER_DEVICE; slot++)); do worker "$GPU1" "$((slot + GPU_WORKERS_PER_DEVICE))" "$total_slots"; done
else
  total_slots=$((GPU_WORKERS_PER_DEVICE * 2))
  declare -a pids=()
  for ((slot=0; slot<GPU_WORKERS_PER_DEVICE; slot++)); do
    worker "$GPU0" "$slot" "$total_slots" > "${LOG_ROOT}/worker_gpu0_slot${slot}.log" 2>&1 &
    pids+=("$!")
  done
  for ((slot=0; slot<GPU_WORKERS_PER_DEVICE; slot++)); do
    worker "$GPU1" "$((slot + GPU_WORKERS_PER_DEVICE))" "$total_slots" > "${LOG_ROOT}/worker_gpu1_slot${slot}.log" 2>&1 &
    pids+=("$!")
  done
  WORKER_PIDS=("${pids[@]}")
  echo "workers: ${pids[*]}"
  for pid in "${pids[@]}"; do wait "$pid"; done
fi
echo "=== SD-GRID ${HOST_TAG} complete ==="
