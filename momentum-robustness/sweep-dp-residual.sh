#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TAUS=("0.1" "1.0" "10.0")
LRS=("0.05" "0.1" "0.01")
SEEDS=("0")
TARGET_EPSILON="10"
TARGET_DELTA="2e-5"

USE_CUDA_FLAG="${USE_CUDA_FLAG:---use-cuda}"
WANDB_ARGS=(
  --wandb
  --wandb-entity ae-gourari-cole-polytechnique
  --wandb-project DP-FedAvg_VS_CC-FedAvg
)

for seed in "${SEEDS[@]}"; do
  for tau in "${TAUS[@]}"; do
    for lr in "${LRS[@]}"; do
      PYTHONPATH=. python3 momentum-robustness/cifar10-all-noattack.py \
        ${USE_CUDA_FLAG} \
        "${WANDB_ARGS[@]}" \
        --model resnet20 \
        --agg dp-residual \
        --clip-tau "$tau" \
        --target-epsilon "$TARGET_EPSILON" \
        --target-delta "$TARGET_DELTA" \
        --residual-alpha 0.1 \
        --lr "$lr" \
        --local-steps 1 \
        --seed "$seed"
    done
  done
done
