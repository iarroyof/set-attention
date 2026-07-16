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
  local fine_heads="${5:-}"
  local coarse_heads="${6:-}"
  local config="configs/lca_cmp/base_token.yaml"
  local tag="token"
  local csv="out/lca_cmp/${family}_L${seq_len}_B${batch_size}_seed${seed}.csv"

  if [[ "$family" == b* ]]; then
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

  if [[ "$family" == b* ]]; then
    local groups="["
    if [[ "$fine_heads" -gt 0 ]]; then
      groups+="{name: fine, num_heads: ${fine_heads}, window_size: 2, stride: 1}"
    fi
    if [[ "$coarse_heads" -gt 0 ]]; then
      if [[ "$fine_heads" -gt 0 ]]; then
        groups+=", "
      fi
      groups+="{name: coarse, num_heads: ${coarse_heads}, window_size: 4, stride: 2}"
    fi
    groups+="]"
    cmd+=("model.multiresolution.groups=${groups}")
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
    run_cell b0 "$seq_len" "$batch_size" "$seed" 8 0
    run_cell b25 "$seq_len" "$batch_size" "$seed" 6 2
    run_cell b50 "$seq_len" "$batch_size" "$seed" 4 4
    run_cell b75 "$seq_len" "$batch_size" "$seed" 2 6
    run_cell b100 "$seq_len" "$batch_size" "$seed" 0 8
  done
done
