#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TAUS=("0.1" "1.0" "5.0")
LRS=("0.05" "0.1" "0.01")
SEEDS=("0")
NOISE_MULTIPLIER="${NOISE_MULTIPLIER:-0.1}"
ANCHOR_NOISE_MULTIPLIER="${ANCHOR_NOISE_MULTIPLIER:-$NOISE_MULTIPLIER}"
TARGET_DELTA="${TARGET_DELTA:-2e-5}"
RESIDUAL_ALPHA="${RESIDUAL_ALPHA:-0.1}"
ANCHOR_PERIOD="${ANCHOR_PERIOD:-100}"
RESIDUAL_CENTER_MODE="${RESIDUAL_CENTER_MODE:-ema}"
RESIDUAL_CENTER_BETA="${RESIDUAL_CENTER_BETA:-0.9}"

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
        --agg dp-fedavg \
        --clip-tau "$tau" \
        --dp-noise-multiplier "$NOISE_MULTIPLIER" \
        --target-delta "$TARGET_DELTA" \
        --lr "$lr" \
        --momentum 0.0 \
        --momentum-mode ema \
        --local-steps 1 \
        --seed "$seed"

      PYTHONPATH=. python3 momentum-robustness/cifar10-all-noattack.py \
        ${USE_CUDA_FLAG} \
        "${WANDB_ARGS[@]}" \
        --model resnet20 \
        --agg dp-residual \
        --clip-tau "$tau" \
        --dp-noise-multiplier "$NOISE_MULTIPLIER" \
        --target-delta "$TARGET_DELTA" \
        --residual-center-mode clipped-residual-ema \
        --residual-alpha "$RESIDUAL_ALPHA" \
        --lr "$lr" \
        --local-steps 1 \
        --seed "$seed"

      PYTHONPATH=. python3 momentum-robustness/cifar10-all-noattack.py \
        ${USE_CUDA_FLAG} \
        "${WANDB_ARGS[@]}" \
        --model resnet20 \
        --agg dp-residual-anchor \
        --clip-tau "$tau" \
        --anchor-clip-tau "$tau" \
        --dp-noise-multiplier "$NOISE_MULTIPLIER" \
        --dp-anchor-noise-multiplier "$ANCHOR_NOISE_MULTIPLIER" \
        --target-delta "$TARGET_DELTA" \
        --residual-center-mode "$RESIDUAL_CENTER_MODE" \
        --residual-center-beta "$RESIDUAL_CENTER_BETA" \
        --residual-alpha "$RESIDUAL_ALPHA" \
        --anchor-period "$ANCHOR_PERIOD" \
        --lr "$lr" \
        --local-steps 1 \
        --seed "$seed"
    done
  done
done
