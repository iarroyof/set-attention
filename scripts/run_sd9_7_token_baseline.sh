#!/usr/bin/env bash
# SD-9.7 matched landmark TOKEN baselines for the intermediate lengths, full WikiText-2.
# Mirrors A8.3 (baseline_linear_landmark) at the SD-9 budget so set-vs-token is matched.
# Output is homogenized downstream by scripts/normalize_sd9x_runs.py (token rows get
# set-specific columns = NA). No data.limit (full data). One GPU per invocation.
#
#   HOST_TAG=blue LENGTHS="2048" SEEDS="0 1 2" GPU=0 bash scripts/run_sd9_7_token_baseline.sh
set -euo pipefail
cd "${REPO_ROOT:-$HOME/set-attention}"

HOST_TAG="${HOST_TAG:-blue}"
LENGTHS="${LENGTHS:-4096 2048}"
SEEDS="${SEEDS:-0 1 2}"
GPU="${GPU:-0}"
IMAGE="${IMAGE:-set-attention:latest}"
PROJECT="${PROJECT:-set-attention}"
BATCH=1
EPOCHS=10
LR=1e-4

OUT_ROOT="out/paper_mechanisms/sd9_7_token_baseline_${HOST_TAG}"
LOG_ROOT="logs/sd9_7_token_baseline_${HOST_TAG}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" audit
STATUS_PATH="${OUT_ROOT}/sd9_7_token_baseline_status.tsv"
[[ -f "${STATUS_PATH}" ]] || printf "name\tgpu\tseq_len\tseed\tcsv\texit_code\n" > "${STATUS_PATH}"

run_case () {
  local seq_len="$1" seed="$2" gpu="$3"
  local name="sd9_7_baseline_landmark_token_L${seq_len}_seed${seed}"
  local group="sd9_7_token_baseline_${HOST_TAG}_L${seq_len}"
  local csv="${OUT_ROOT}/${group}/${name}.csv"
  mkdir -p "${OUT_ROOT}/${group}"
  echo "=== ${name} on GPU ${gpu} ==="
  set +e
  docker run --rm --gpus "device=${gpu}" --ipc=host -u "$(id -u):$(id -g)" \
    -e HOME=/workspace -e XDG_CACHE_HOME=/workspace/.cache -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_DATASETS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e WANDB_MODE=offline \
    -e WANDB_PROJECT="${PROJECT}" -e WANDB_NAME="${name}" -e WANDB_RUN_GROUP="${group}" \
    -v "${PWD}:/workspace" -w /workspace "${IMAGE}" \
    /usr/bin/python scripts/run_experiment.py \
      --config configs/paper_lr_norm/baseline_linear_landmark.yaml \
      --csv-path "${csv}" \
      --override \
        "training.output_dir=${OUT_ROOT}/${group}/${name}" \
        logging.wandb.enable=false "logging.wandb.project=${PROJECT}" "logging.wandb.run_name=${name}" \
        data.dataset=wikitext2 "data.batch_size=${BATCH}" "data.seq_len=${seq_len}" \
        "training.seed=${seed}" "training.epochs=${EPOCHS}" "training.lr=${LR}" training.warmup_steps=1000 \
        model.d_model=384 model.dim_feedforward=1536 model.num_layers=6 model.num_heads=8 \
        "model.max_seq_len=${seq_len}" model.backend_params.landmark_coverage=0.25 \
    > "${LOG_ROOT}/${name}.log" 2>&1
  local rc="$?"
  set -e
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${name}" "${gpu}" "${seq_len}" "${seed}" "${csv}" "${rc}" >> "${STATUS_PATH}"
}

for seq_len in ${LENGTHS}; do
  for seed in ${SEEDS}; do
    run_case "${seq_len}" "${seed}" "${GPU}"
  done
done
echo "=== SD-9.7 token baselines complete (${HOST_TAG}) ==="
