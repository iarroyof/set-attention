#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LAUNCH=0
for arg in "$@"; do
  case "$arg" in
    --launch) LAUNCH=1 ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "${MRP2_AR_HITS_RETRAIN:-}" != "approved" || "$LAUNCH" != "1" ]]; then
  cat >&2 <<'MSG'
Refusing to launch MRP-2 AR-hit checkpoint retraining.

Required:
  MRP2_AR_HITS_RETRAIN=approved
  scripts/run_mrp2_ar_hit_retrain.sh --launch

This runs only the registered MRP-2 rows: token, b0, b25, b100 at L=2048,
B=4, exact dense, seeds 0,1,2, with final checkpoint saving enabled.
MSG
  exit 3
fi

OUT_ROOT="${MRP2_OUT_ROOT:-out/mrp2_ar_hits/retrain}"
SEEDS="${MRP2_SEEDS:-0 1 2}"
EPOCHS="${MRP2_EPOCHS:-10}"
LR="${MRP2_LR:-0.0001}"
WARMUP="${MRP2_WARMUP:-1000}"
SEQ_LEN="${MRP2_SEQ_LEN:-2048}"
BATCH="${MRP2_BATCH:-4}"
FORCE="${MRP2_FORCE:-0}"

COMMON_OVERRIDES=(
  "data.dataset=wikitext2"
  "data.seq_len=${SEQ_LEN}"
  "data.batch_size=${BATCH}"
  "training.epochs=${EPOCHS}"
  "training.lr=${LR}"
  "training.warmup_steps=${WARMUP}"
  "training.deterministic=true"
  "training.benchmark_mode=false"
  "training.experiment_contract=sd_grid_seeded_v1"
  "training.diagnostics_contract=current_matrix_v1"
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

set_groups() {
  local row="$1"
  case "$row" in
    b0) echo "[{name: fine, num_heads: 8, window_size: 2, stride: 1}]" ;;
    b25) echo "[{name: fine, num_heads: 6, window_size: 2, stride: 1}, {name: coarse, num_heads: 2, window_size: 4, stride: 2}]" ;;
    b100) echo "[{name: coarse, num_heads: 8, window_size: 4, stride: 2}]" ;;
    *) echo "unknown set row: $row" >&2; return 2 ;;
  esac
}

run_one() {
  local row="$1" seed="$2" cfg out csv ckpt
  out="${OUT_ROOT}/${row}_seed${seed}"
  csv="${out}.csv"
  ckpt="${out}/checkpoints/final.pt"
  if [[ "$FORCE" != "1" && -s "$ckpt" ]]; then
    echo "SKIP checkpoint exists row=${row} seed=${seed} ckpt=${ckpt}"
    return 0
  fi
  mkdir -p "$out"
  if [[ "$row" == "token" ]]; then
    cfg="configs/paper_lr_norm/baseline_dense_exact.yaml"
    python scripts/run_experiment.py \
      --config "$cfg" \
      --csv-path "$csv" \
      --override \
        "${COMMON_OVERRIDES[@]}" \
        "training.seed=${seed}" \
        "training.output_dir=${out}" \
        "training.checkpoint.directory=${out}/checkpoints" \
        "logging.wandb.run_name=mrp2_ar_hits_${row}_seed${seed}"
  else
    cfg="configs/set_dictionary/sd9_multiresolution.yaml"
    local groups
    groups="$(set_groups "$row")"
    python scripts/run_experiment.py \
      --config "$cfg" \
      --csv-path "$csv" \
      --override \
        "${COMMON_OVERRIDES[@]}" \
        "training.seed=${seed}" \
        "training.output_dir=${out}" \
        "training.checkpoint.directory=${out}/checkpoints" \
        "logging.wandb.run_name=mrp2_ar_hits_${row}_seed${seed}" \
        "model.output_residual_mode=anchor_span" \
        "model.token_mlp.enabled=false" \
        "model.anchor.enabled=false" \
        "model.allow_token_token=false" \
        "model.candidate_fiber=endpoint_window" \
        "model.window_size=$([[ "$row" == "b100" ]] && echo 4 || echo 2)" \
        "model.stride=$([[ "$row" == "b100" ]] && echo 2 || echo 1)" \
        "model.multiresolution.enabled=true" \
        "model.multiresolution.groups=${groups}"
  fi
}

for seed in $SEEDS; do
  for row in token b0 b25 b100; do
    run_one "$row" "$seed"
  done
done
