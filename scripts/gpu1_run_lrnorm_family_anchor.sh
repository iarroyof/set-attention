#!/usr/bin/env bash
set -euo pipefail

cd ~/set-attention
mkdir -p logs out/paper_lr_norm

PROJECT="set-attention"
GROUP="paper_lr_norm_family_D384_FF1536"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
LRS=(1e-4 2e-4 3e-4)

run_one () {
  local GPU="$1"
  local CFG="$2"
  local FAMILY="$3"
  local LR="$4"

  local LR_TAG="${LR//./p}"
  local NAME="paper_lrnorm_${FAMILY}_D384_FF1536_lr${LR_TAG}_seed0"

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
      model.d_model=384 \
      model.dim_feedforward=1536 \
      model.num_layers=6 \
      model.num_heads=8 \
      model.window_size=16 \
      model.stride=8 \
      model.router_topk=16 \
      model.router_temperature=1.0 \
      model.pooling.mode=soft_trimmed_boltzmann \
      model.pooling.tau=0.1 \
      model.pooling.q=0.85 \
      model.router_multihead=true \
      model.pooling_multihead=false
}

for LR in "${LRS[@]}"; do
  run_one 1 configs/paper_lr_norm/set_sparse_local_band.yaml set_sparse "${LR}" \
    | tee "logs/paper_lrnorm_set_sparse_D384_FF1536_lr${LR//./p}.log"
  run_one 1 configs/paper_lr_norm/set_linear_landmark.yaml set_linear "${LR}" \
    | tee "logs/paper_lrnorm_set_linear_D384_FF1536_lr${LR//./p}.log"
done
