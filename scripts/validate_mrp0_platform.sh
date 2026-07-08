#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT:-$HOME/set-attention}"

OUT_ROOT="${OUT_ROOT:-out/mrp0_validation}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${OUT_ROOT}/${RUN_ID}"
mkdir -p "$ROOT"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

python -m pytest -q \
  tests/test_mrp0_checkpoints.py \
  tests/test_mrp0_ordered_data.py \
  tests/test_mrp0_masked_metrics.py \
  tests/test_mrp0_determinism.py \
  tests/test_mrp0_config_contract.py \
  tests/test_run_experiment_seed.py \
  tests/test_multiresolution_diagnostics.py \
  tests/test_sd_grid_contract.py \
  tests/test_sd_grid_status.py

common=(
  data.limit=8
  data.streaming=false
  data.seq_len=32
  data.batch_size=2
  data.num_workers=0
  model.max_seq_len=32
  model.d_model=32
  model.dim_feedforward=64
  model.num_layers=1
  model.num_heads=4
  training.seed=17
  training.epochs=1
  training.lr=0.0001
  training.deterministic=true
  training.strict_deterministic=true
  training.benchmark_mode=false
  training.checkpoint.save_final=true
  training.checkpoint.save_every_epochs=0
  logging.wandb.enable=false
  'logging.metric_columns=[train/accuracy,train/valid_tokens,val/accuracy,val/valid_tokens]'
)

run_token() {
  local name="$1"
  local out="${ROOT}/${name}"
  python scripts/run_experiment.py \
    --config configs/paper_lr_norm/baseline_dense_exact.yaml \
    --csv-path "${out}/metrics.csv" \
    --override \
      "${common[@]}" \
      "training.output_dir=${out}" \
      "training.checkpoint.directory=${out}/checkpoints"
}

run_token token_a
run_token token_b
python scripts/verify_checkpoint_replay.py \
  "${ROOT}/token_a/checkpoints/final.pt" \
  "${ROOT}/token_b/checkpoints/final.pt" \
  > "${ROOT}/token_strict_replay.json"

set_overrides=(
  "${common[@]}"
  model.d_phi=32
  model.set_state_dim=32
  model.feature_params.num_bins=16
  model.router_topk=4
  'model.multiresolution.groups=[{name: fine, num_heads: 3, window_size: 2, stride: 1}, {name: coarse, num_heads: 1, window_size: 4, stride: 2}]'
)

run_b25() {
  local name="$1"
  local out="${ROOT}/${name}"
  python scripts/run_experiment.py \
    --config configs/set_dictionary/sd9_multiresolution.yaml \
    --csv-path "${out}/metrics.csv" \
    --override \
      "${set_overrides[@]}" \
      "training.output_dir=${out}" \
      "training.checkpoint.directory=${out}/checkpoints"
}

run_b25 b25_a
run_b25 b25_b
python scripts/verify_checkpoint_replay.py \
  "${ROOT}/b25_a/checkpoints/final.pt" \
  "${ROOT}/b25_b/checkpoints/final.pt" \
  > "${ROOT}/b25_strict_replay.json"

token_checkpoint="${ROOT}/token_a/checkpoints/final.pt"
before_sha="$(sha256sum "$token_checkpoint" | cut -d' ' -f1)"
eval_out="${ROOT}/token_eval"
python scripts/run_experiment.py \
  --config configs/paper_lr_norm/baseline_dense_exact.yaml \
  --csv-path "${eval_out}/metrics.csv" \
  --override \
    "${common[@]}" \
    training.checkpoint.save_final=false \
    "training.checkpoint.eval_only_from=${token_checkpoint}" \
    "training.output_dir=${eval_out}" \
    "training.checkpoint.directory=${eval_out}/checkpoints"
after_sha="$(sha256sum "$token_checkpoint" | cut -d' ' -f1)"
test "$before_sha" = "$after_sha"
test ! -e "${eval_out}/checkpoints/final.pt"

printf 'MRP-0 validation PASS: %s\n' "$ROOT"
