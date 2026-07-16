#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry-run}"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
PROJECT="${WANDB_PROJECT:-}"

if [[ "$MODE" != "dry-run" && "$MODE" != "launch" ]]; then
  echo "usage: $0 [dry-run|launch]" >&2
  exit 2
fi

run_cell() {
  local family="$1"
  local seq_len="$2"
  local batch_size="$3"
  local seed="$4"
  local window_size="${5:-}"
  local stride="${6:-}"
  local config="configs/lca_cmp/base_token.yaml"
  local tag="token"
  local csv="out/lca_cmp/${family}_L${seq_len}_B${batch_size}_seed${seed}.csv"

  if [[ "$family" == set_* ]]; then
    config="configs/lca_cmp/base_set.yaml"
    tag="${family}"
  fi

  local -a cmd=(
    python scripts/run_lca_cmp.py
    --config "$config"
    --device "cuda"
    --csv-path "$csv"
    --override
    "data.seq_len=${seq_len}"
    "data.batch_size=${batch_size}"
    "model.max_seq_len=${seq_len}"
    "training.seed=${seed}"
    "logging.csv.path=${csv}"
  )

  if [[ "$family" == set_* ]]; then
    cmd+=(
      "model.window_size=${window_size}"
      "model.stride=${stride}"
    )
  fi

  if [[ -n "$PROJECT" ]]; then
    cmd+=(--wandb --wandb-project "$PROJECT" --wandb-tags "mrp-lca-cmp,${tag},L${seq_len},B${batch_size}")
  fi

  if [[ "$MODE" == "dry-run" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s' "$DEVICE"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    mkdir -p out/lca_cmp/logs
    local log="out/lca_cmp/logs/${family}_L${seq_len}_B${batch_size}_seed${seed}.log"
    CUDA_VISIBLE_DEVICES="$DEVICE" nohup "${cmd[@]}" > "$log" 2>&1 &
    echo "$! $family L=$seq_len B=$batch_size seed=$seed log=$log"
  fi
}

for seed in 0 1 2; do
  for spec in "1024 4" "2048 4" "3584 4" "4096 3"; do
    read -r seq_len batch_size <<< "$spec"
    run_cell token "$seq_len" "$batch_size" "$seed"
    run_cell set_w2_s1 "$seq_len" "$batch_size" "$seed" 2 1
    run_cell set_w4_s2 "$seq_len" "$batch_size" "$seed" 4 2
    run_cell set_w8_s4 "$seq_len" "$batch_size" "$seed" 8 4
  done
done
