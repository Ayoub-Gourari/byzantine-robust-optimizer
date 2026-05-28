#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROJECT="${WANDB_PROJECT:-ARDP-FedAvg_VS_DP-FedAvg}"
ENTITY="${WANDB_ENTITY:-ae-gourari-cole-polytechnique}"
MODEL="${MODEL:-resnet20}"
USE_CUDA_FLAG="${USE_CUDA_FLAG:---use-cuda}"

SEEDS=(${SEEDS:-0})
LRS=(${LRS:-0.03 0.05 0.07})

MOMENTUM="${MOMENTUM:-0.9}"
MOMENTUM_MODE="${MOMENTUM_MODE:-ema}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"
NOISE_MULTIPLIER="${NOISE_MULTIPLIER:-0.1}"
ANCHOR_NOISE_MULTIPLIER="${ANCHOR_NOISE_MULTIPLIER:-$NOISE_MULTIPLIER}"
TARGET_DELTA="${TARGET_DELTA:-2e-5}"
RUN_FULL_GRID="${RUN_FULL_GRID:-1}"
RUN_FEDAVG_SWEEP="${RUN_FEDAVG_SWEEP:-0}"

# Phase 1: keep the search small and identify the best learning rate first.
LR_SWEEP_FEDAVG_CLIPS=(${LR_SWEEP_FEDAVG_CLIPS:-1.6})
LR_SWEEP_ARDP_CRES=(${LR_SWEEP_ARDP_CRES:-0.7})
LR_SWEEP_ARDP_CANCHOR=(${LR_SWEEP_ARDP_CANCHOR:-1.5})
LR_SWEEP_ARDP_ALPHAS=(${LR_SWEEP_ARDP_ALPHAS:-0.2})
LR_SWEEP_ARDP_PERIODS=(${LR_SWEEP_ARDP_PERIODS:-200})

# Phase 2: refine after choosing BEST_LR from the phase-1 runs.
RUN_REFINE="${RUN_REFINE:-0}"
BEST_LR="${BEST_LR:-0.05}"
REFINE_FEDAVG_CLIPS=(${REFINE_FEDAVG_CLIPS:-1.2 1.6 2.0})
REFINE_ARDP_CRES=(${REFINE_ARDP_CRES:-0.6 0.7})
REFINE_ARDP_CANCHOR=(${REFINE_ARDP_CANCHOR:-1.2 1.5})
REFINE_ARDP_ALPHAS=(${REFINE_ARDP_ALPHAS:-0.2 0.3})
REFINE_ARDP_PERIODS=(${REFINE_ARDP_PERIODS:-100 200})

# Keep the default ARDP center search minimal: tracker-style centers first.
RUN_EMA_CENTER="${RUN_EMA_CENTER:-0}"
EMA_BETAS=(${EMA_BETAS:-0.8 0.7})

# Full-grid mode: trimmed by default to the strongest practical candidates.
FULL_GRID_FEDAVG_CLIPS=(${FULL_GRID_FEDAVG_CLIPS:-1.6})
FULL_GRID_ARDP_CRES=(${FULL_GRID_ARDP_CRES:-0.55 0.7 0.8})
FULL_GRID_ARDP_CANCHOR=(${FULL_GRID_ARDP_CANCHOR:-0.85 1.5 1.8})
FULL_GRID_ARDP_ALPHAS=(${FULL_GRID_ARDP_ALPHAS:-0.2 0.3})
FULL_GRID_ARDP_PERIODS=(${FULL_GRID_ARDP_PERIODS:-100 200})
FULL_GRID_EMA_BETAS=(${FULL_GRID_EMA_BETAS:-0.8 0.7})
FULL_GRID_INCLUDE_EMA_CENTER="${FULL_GRID_INCLUDE_EMA_CENTER:-0}"

COMMON_ARGS=(
  --wandb
  --wandb-entity "$ENTITY"
  --wandb-project "$PROJECT"
  --model "$MODEL"
  --dp-noise-multiplier "$NOISE_MULTIPLIER"
  --target-delta "$TARGET_DELTA"
  --momentum "$MOMENTUM"
  --momentum-mode "$MOMENTUM_MODE"
  --local-steps "$LOCAL_STEPS"
)

run_fedavg() {
  local lr="$1"
  local clip_tau="$2"
  local seed="$3"
  local tag="$4"

  PYTHONPATH=. python3 momentum-robustness/cifar10-all-noattack.py \
    ${USE_CUDA_FLAG} \
    "${COMMON_ARGS[@]}" \
    --wandb-run-name "${tag}-dpfedavg-C${clip_tau}-lr${lr}-m${MOMENTUM}-mom${MOMENTUM_MODE}-sigma${NOISE_MULTIPLIER}-local${LOCAL_STEPS}-seed${seed}" \
    --agg dp-fedavg \
    --clip-tau "$clip_tau" \
    --lr "$lr" \
    --seed "$seed"
}

run_ardp_clipres() {
  local lr="$1"
  local c_res="$2"
  local c_anchor="$3"
  local alpha="$4"
  local anchor_period="$5"
  local seed="$6"
  local tag="$7"

  PYTHONPATH=. python3 momentum-robustness/cifar10-all-noattack.py \
    ${USE_CUDA_FLAG} \
    "${COMMON_ARGS[@]}" \
    --wandb-run-name "${tag}-ardp-clipres-Cres${c_res}-Canchor${c_anchor}-alpha${alpha}-period${anchor_period}-lr${lr}-m${MOMENTUM}-mom${MOMENTUM_MODE}-sigma${NOISE_MULTIPLIER}-local${LOCAL_STEPS}-seed${seed}" \
    --agg dp-residual-anchor \
    --clip-tau "$c_res" \
    --anchor-clip-tau "$c_anchor" \
    --dp-anchor-noise-multiplier "$ANCHOR_NOISE_MULTIPLIER" \
    --residual-center-mode clipped-residual-ema \
    --residual-alpha "$alpha" \
    --anchor-period "$anchor_period" \
    --lr "$lr" \
    --seed "$seed"
}

run_ardp_ema() {
  local lr="$1"
  local c_res="$2"
  local c_anchor="$3"
  local alpha="$4"
  local beta="$5"
  local anchor_period="$6"
  local seed="$7"
  local tag="$8"

  PYTHONPATH=. python3 momentum-robustness/cifar10-all-noattack.py \
    ${USE_CUDA_FLAG} \
    "${COMMON_ARGS[@]}" \
    --wandb-run-name "${tag}-ardp-ema-Cres${c_res}-Canchor${c_anchor}-alpha${alpha}-beta${beta}-period${anchor_period}-lr${lr}-m${MOMENTUM}-mom${MOMENTUM_MODE}-sigma${NOISE_MULTIPLIER}-local${LOCAL_STEPS}-seed${seed}" \
    --agg dp-residual-anchor \
    --clip-tau "$c_res" \
    --anchor-clip-tau "$c_anchor" \
    --dp-anchor-noise-multiplier "$ANCHOR_NOISE_MULTIPLIER" \
    --residual-center-mode ema \
    --residual-center-beta "$beta" \
    --residual-alpha "$alpha" \
    --anchor-period "$anchor_period" \
    --lr "$lr" \
    --seed "$seed"
}

run_full_grid() {
  echo "=== Full grid: learning-rate-outer sweep ==="
  for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [[ "$RUN_FEDAVG_SWEEP" == "1" ]]; then
        for clip_tau in "${FULL_GRID_FEDAVG_CLIPS[@]}"; do
          run_fedavg "$lr" "$clip_tau" "$seed" "fullgrid"
        done
      fi

      for c_res in "${FULL_GRID_ARDP_CRES[@]}"; do
        for c_anchor in "${FULL_GRID_ARDP_CANCHOR[@]}"; do
          for alpha in "${FULL_GRID_ARDP_ALPHAS[@]}"; do
            for anchor_period in "${FULL_GRID_ARDP_PERIODS[@]}"; do
              run_ardp_clipres \
                "$lr" "$c_res" "$c_anchor" "$alpha" "$anchor_period" "$seed" "fullgrid"
            done
          done
        done
      done

      if [[ "$FULL_GRID_INCLUDE_EMA_CENTER" == "1" ]]; then
        for c_res in "${FULL_GRID_ARDP_CRES[@]}"; do
          for c_anchor in "${FULL_GRID_ARDP_CANCHOR[@]}"; do
            for alpha in "${FULL_GRID_ARDP_ALPHAS[@]}"; do
              for beta in "${FULL_GRID_EMA_BETAS[@]}"; do
                for anchor_period in "${FULL_GRID_ARDP_PERIODS[@]}"; do
                  run_ardp_ema \
                    "$lr" "$c_res" "$c_anchor" "$alpha" "$beta" \
                    "$anchor_period" "$seed" "fullgrid"
                done
              done
            done
          done
        done
      fi
    done
  done
}

if [[ "$RUN_FULL_GRID" == "1" ]]; then
  run_full_grid
  exit 0
fi

echo "=== Phase 1: learning-rate-first sweep ==="
for lr in "${LRS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for clip_tau in "${LR_SWEEP_FEDAVG_CLIPS[@]}"; do
      run_fedavg "$lr" "$clip_tau" "$seed" "phase1"
    done

    for c_res in "${LR_SWEEP_ARDP_CRES[@]}"; do
      for c_anchor in "${LR_SWEEP_ARDP_CANCHOR[@]}"; do
        for alpha in "${LR_SWEEP_ARDP_ALPHAS[@]}"; do
          for anchor_period in "${LR_SWEEP_ARDP_PERIODS[@]}"; do
            run_ardp_clipres \
              "$lr" "$c_res" "$c_anchor" "$alpha" "$anchor_period" "$seed" "phase1"
          done
        done
      done
    done
  done
done

if [[ "$RUN_REFINE" != "1" ]]; then
  echo "Phase 1 complete. Set RUN_REFINE=1 BEST_LR=<chosen_lr> to launch the refine sweep."
  exit 0
fi

echo "=== Phase 2: refine around BEST_LR=${BEST_LR} ==="
for seed in "${SEEDS[@]}"; do
  for clip_tau in "${REFINE_FEDAVG_CLIPS[@]}"; do
    run_fedavg "$BEST_LR" "$clip_tau" "$seed" "phase2"
  done

  for c_res in "${REFINE_ARDP_CRES[@]}"; do
    for c_anchor in "${REFINE_ARDP_CANCHOR[@]}"; do
      for alpha in "${REFINE_ARDP_ALPHAS[@]}"; do
        for anchor_period in "${REFINE_ARDP_PERIODS[@]}"; do
          run_ardp_clipres \
            "$BEST_LR" "$c_res" "$c_anchor" "$alpha" "$anchor_period" "$seed" "phase2"
        done
      done
    done
  done

  if [[ "$RUN_EMA_CENTER" == "1" ]]; then
    for c_res in "${REFINE_ARDP_CRES[@]}"; do
      for c_anchor in "${REFINE_ARDP_CANCHOR[@]}"; do
        for alpha in "${REFINE_ARDP_ALPHAS[@]}"; do
          for beta in "${EMA_BETAS[@]}"; do
            for anchor_period in "${REFINE_ARDP_PERIODS[@]}"; do
              run_ardp_ema \
                "$BEST_LR" "$c_res" "$c_anchor" "$alpha" "$beta" \
                "$anchor_period" "$seed" "phase2"
            done
          done
        done
      done
    done
  fi
done
