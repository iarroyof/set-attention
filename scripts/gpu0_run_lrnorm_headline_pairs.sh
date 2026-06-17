#!/usr/bin/env bash
set -euo pipefail

cd ~/set-attention
LOG_ROOT="${LOG_ROOT:-logs/a2_grid}"
OUT_ROOT="${OUT_ROOT:-out/paper_lr_norm}"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

PROJECT="set-attention"
GROUP_PREFIX="${GROUP_PREFIX:-paper_lr_norm_headline_A2}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
LRS=(1e-4 2e-4 3e-4 5e-4 7e-4)
SEEDS=(0 1 2)
SPECS=(
  "D384_FF1536 384 1536"
  "D512_FF2048 512 2048"
  "D384_FF3072 384 3072"
  "D512_FF1024 512 1024"
)

run_one () {
  local GPU="$1"
  local CFG="$2"
  local IMPL="$3"
  local SPEC="$4"
  local D="$5"
  local FF="$6"
  local LR="$7"
  local SEED="$8"

  local LR_TAG="${LR//./p}"
  local NAME="paper_lrnorm_${IMPL}_${SPEC}_lr${LR_TAG}_seed${SEED}"
  local GROUP="${GROUP_PREFIX}_${SPEC}"
  local EXTRA_OVERRIDES=()
  if [[ "${IMPL}" != "baseline" ]]; then
    EXTRA_OVERRIDES=(
      model.window_size=16
      model.stride=8
      model.set_causality_mode=strict_past
      model.router_topk=16
      model.router_temperature=1.0
      model.pooling.mode=soft_trimmed_boltzmann
      model.pooling.tau=0.1
      model.pooling.q=0.85
      model.router_multihead=true
      model.pooling_multihead=false
    )
  fi

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
      model.d_model="${D}" \
      model.dim_feedforward="${FF}" \
      model.num_layers=6 \
      model.num_heads=8 \
      "${EXTRA_OVERRIDES[@]}"
}

for item in "${SPECS[@]}"; do
  read -r SPEC D FF <<< "${item}"
  for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
      run_one 0 configs/paper_lr_norm/baseline_dense_exact.yaml baseline "${SPEC}" "${D}" "${FF}" "${LR}" "${SEED}" \
        | tee "${LOG_ROOT}/paper_lrnorm_baseline_${SPEC}_lr${LR//./p}_seed${SEED}.log"
      run_one 0 configs/paper_lr_norm/set_dense_exact.yaml set_dense "${SPEC}" "${D}" "${FF}" "${LR}" "${SEED}" \
        | tee "${LOG_ROOT}/paper_lrnorm_set_dense_${SPEC}_lr${LR//./p}_seed${SEED}.log"
    done
  done
done
