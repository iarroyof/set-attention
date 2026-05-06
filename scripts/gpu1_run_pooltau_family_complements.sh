#!/usr/bin/env bash
set -euo pipefail

cd ~/set-attention
mkdir -p logs out/paper_complements

PROJECT="set-attention"
GROUP_PREFIX="paper_comp_pooltau"

run_one () {
  local GPU="$1"
  local CFG="$2"
  local FAMILY="$3"
  local TAU="$4"

  local NAME="paper_pooltau_${FAMILY}_tau${TAU}_s4_w16_Tr1_lr1e4_seed0"
  local GROUP="${GROUP_PREFIX}_${FAMILY}"

  echo "=== Running ${NAME}"
  docker compose exec -T \
    -e CUDA_VISIBLE_DEVICES="${GPU}" \
    -e WANDB_PROJECT="${PROJECT}" \
    -e WANDB_NAME="${NAME}" \
    -e WANDB_RUN_GROUP="${GROUP}" \
    set-attention \
    python scripts/run_experiment.py \
    --config "${CFG}" \
    --wandb \
    --wandb-project "${PROJECT}" \
    --csv-path "out/paper_complements/${GROUP}/${NAME}.csv" \
    --override \
      training.output_dir="out/paper_complements/${GROUP}/${NAME}" \
      logging.wandb.enable=true \
      logging.wandb.project="${PROJECT}" \
      logging.wandb.run_name="${NAME}" \
      model.window_size=16 \
      model.stride=4 \
      model.router_temperature=1.0 \
      model.pooling.tau="${TAU}" \
      model.pooling.q=0.85 \
      training.seed=0 \
      training.lr=1e-4 \
      training.epochs=10 \
    | tee "logs/${NAME}.log"
}

for TAU in 0.05 0.075 0.1 0.15 0.2; do
  run_one 1 configs/paper_complements/family_dense_exact.yaml       dense_exact       "${TAU}"
  run_one 1 configs/paper_complements/family_sparse_local_band.yaml sparse_local_band "${TAU}"
  run_one 1 configs/paper_complements/family_linear_landmark.yaml   linear_landmark   "${TAU}"
done
