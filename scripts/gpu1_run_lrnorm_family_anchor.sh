#!/usr/bin/env bash
set -euo pipefail

cd ~/set-attention
LOG_ROOT="${LOG_ROOT:-logs/a2_grid}"
OUT_ROOT="${OUT_ROOT:-out/paper_lr_norm}"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

PROJECT="set-attention"
FAMILY_GROUP="${FAMILY_GROUP:-paper_lr_norm_family_A2_D384_FF1536}"
ANCHOR_GROUP="${ANCHOR_GROUP:-paper_lr_norm_anchor_A2_D384_FF1536_s4}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LRS=(1e-4 2e-4 3e-4 5e-4 7e-4)
SEEDS=(0 1 2)

run_one () {
  local GPU="$1"
  local CFG="$2"
  local FAMILY="$3"
  local LR="$4"
  local SEED="$5"
  local STRIDE="${6:-8}"
  local GROUP="${7:-${FAMILY_GROUP}}"

  local LR_TAG="${LR//./p}"
  local NAME="paper_lrnorm_${FAMILY}_D384_FF1536"
  if [[ "${STRIDE}" != "8" ]]; then
    NAME="${NAME}_s${STRIDE}"
  fi
  NAME="${NAME}_lr${LR_TAG}_seed${SEED}"

  echo "=== Running ${NAME}"
  docker compose exec -T \
    -e CUDA_VISIBLE_DEVICES="${GPU}" \
    -e HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    -e WANDB_MODE="${WANDB_MODE}" \
    -e WANDB_PROJECT="${PROJECT}" \
    -e WANDB_NAME="${NAME}" \
    -e WANDB_RUN_GROUP="${GROUP}" \
    set-attention \
    python scripts/run_experiment.py \
    --config "${CFG}" \
    --wandb \
    --wandb-project "${PROJECT}" \
    --csv-path "${OUT_ROOT}/${GROUP}/${NAME}.csv" \
    --override \
      training.output_dir="${OUT_ROOT}/${GROUP}/${NAME}" \
      logging.wandb.enable=true \
      logging.wandb.project="${PROJECT}" \
      logging.wandb.run_name="${NAME}" \
      data.dataset=wikitext2 \
      data.batch_size=16 \
      data.seq_len=512 \
      training.seed="${SEED}" \
      training.epochs=10 \
      training.lr="${LR}" \
      training.warmup_steps=1000 \
      model.d_model=384 \
      model.dim_feedforward=1536 \
      model.num_layers=6 \
      model.num_heads=8 \
      model.window_size=16 \
      model.stride="${STRIDE}" \
      model.set_causality_mode=strict_past \
      model.router_topk=16 \
      model.router_temperature=1.0 \
      model.pooling.mode=soft_trimmed_boltzmann \
      model.pooling.tau=0.1 \
      model.pooling.q=0.85 \
      model.router_multihead=true \
      model.pooling_multihead=false
}

for SEED in "${SEEDS[@]}"; do
  run_one 1 configs/paper_lr_norm/set_dense_exact.yaml anchor_set_dense 1e-4 "${SEED}" 4 "${ANCHOR_GROUP}" \
    | tee "${LOG_ROOT}/paper_lrnorm_anchor_set_dense_D384_FF1536_s4_lr1e-4_seed${SEED}.log"
done

for SEED in "${SEEDS[@]}"; do
  for LR in "${LRS[@]}"; do
    run_one 1 configs/paper_lr_norm/set_sparse_local_band.yaml set_sparse "${LR}" "${SEED}" \
      | tee "${LOG_ROOT}/paper_lrnorm_set_sparse_D384_FF1536_lr${LR//./p}_seed${SEED}.log"
    run_one 1 configs/paper_lr_norm/set_linear_landmark.yaml set_linear "${LR}" "${SEED}" \
      | tee "${LOG_ROOT}/paper_lrnorm_set_linear_D384_FF1536_lr${LR//./p}_seed${SEED}.log"
  done
done
