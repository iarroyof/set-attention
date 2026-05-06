#!/usr/bin/env bash
set -euo pipefail

cd ~/set-attention
mkdir -p logs out/paper_complements

PROJECT="set-attention"
GROUP_PREFIX="paper_comp_boundary"

run_one () {
  local GPU="$1"
  local CFG="$2"
  local FAMILY="$3"
  local STRIDE="$4"

  local NAME="paper_boundary_${FAMILY}_s${STRIDE}_w16_Tr1_taup0.1_lr1e4_seed0"
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
      model.stride="${STRIDE}" \
      model.router_temperature=1.0 \
      model.pooling.tau=0.1 \
      model.pooling.q=0.85 \
      training.seed=0 \
      training.lr=1e-4 \
      training.epochs=10 \
    | tee "logs/${NAME}.log"
}

for S in 3 4 5 6 8; do
  run_one 0 configs/paper_complements/family_dense_exact.yaml       dense_exact       "${S}"
  run_one 0 configs/paper_complements/family_sparse_local_band.yaml sparse_local_band "${S}"
  run_one 0 configs/paper_complements/family_linear_landmark.yaml   linear_landmark   "${S}"
done
