#!/usr/bin/env bash
set -euo pipefail

cd ~/set-attention
mkdir -p logs out/paper_lr_norm

PROJECT="set-attention"
GROUP_PREFIX="paper_lr_norm_headline"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

run_one () {
  local GPU="$1"
  local CFG="$2"
  local IMPL="$3"
  local SPEC="$4"
  local D="$5"
  local FF="$6"
  local LR="$7"

  local LR_TAG="${LR//./p}"
  local NAME="paper_lrnorm_${IMPL}_${SPEC}_lr${LR_TAG}_seed0"
  local GROUP="${GROUP_PREFIX}_${SPEC}"

  echo "=== Running ${NAME}"
  docker compose exec -T \
    -e CUDA_VISIBLE_DEVICES="${GPU}" \
    -e HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    -e WANDB_PROJECT="${PROJECT}" \
    -e WANDB_NAME="${NAME}" \
    -e WANDB_RUN_GROUP="${GROUP}" \
    set-attention \
    python scripts/run_experiment.py \
    --config "${CFG}" \
    --wandb \
    --wandb-project "${PROJECT}" \
    --csv-path "out/paper_lr_norm/${GROUP}/${NAME}.csv" \
    --override \
      training.output_dir="out/paper_lr_norm/${GROUP}/${NAME}" \
      logging.wandb.enable=true \
      logging.wandb.project="${PROJECT}" \
      logging.wandb.run_name="${NAME}" \
      data.dataset=wikitext2 \
      data.batch_size=16 \
      data.seq_len=512 \
      training.seed=0 \
      training.epochs=10 \
      training.lr="${LR}" \
      training.warmup_steps=1000 \
      model.d_model="${D}" \
      model.dim_feedforward="${FF}" \
      model.num_layers=6 \
      model.num_heads=8
}

run_pair () {
  local SPEC="$1"
  local D="$2"
  local FF="$3"
  local LR="$4"
  run_one 0 configs/paper_lr_norm/baseline_dense_exact.yaml baseline "${SPEC}" "${D}" "${FF}" "${LR}" \
    | tee "logs/paper_lrnorm_baseline_${SPEC}_lr${LR//./p}.log"
  run_one 0 configs/paper_lr_norm/set_dense_exact.yaml set_dense "${SPEC}" "${D}" "${FF}" "${LR}" \
    | tee "logs/paper_lrnorm_set_dense_${SPEC}_lr${LR//./p}.log"
}

# Resume after the failed run:
# completed before failure:
#   D384_FF1536 all LRs; D512_FF2048 lr=1e-4; D512_FF2048 baseline lr=2e-4.
run_one 0 configs/paper_lr_norm/set_dense_exact.yaml set_dense D512_FF2048 512 2048 2e-4 \
  | tee "logs/paper_lrnorm_set_dense_D512_FF2048_lr2e-4.log"
run_pair D512_FF2048 512 2048 3e-4

for LR in 1e-4 2e-4 3e-4; do
  run_pair D384_FF3072 384 3072 "${LR}"
done

for LR in 1e-4 2e-4 3e-4; do
  run_pair D512_FF1024 512 1024 "${LR}"
done
