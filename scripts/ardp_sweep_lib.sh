#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN="${DRY_RUN:-1}"
SWEEP_SIZE="${SWEEP_SIZE:-small}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-ae-gourari-cole-polytechnique}"
WANDB_PROJECT="${WANDB_PROJECT:-ARDP-FedAvg_VS_DP-FedAvg}"
USE_CUDA="${USE_CUDA:-1}"
MODEL="${MODEL:-resnet20}"
SEEDS=(${SEEDS:-0})
EPOCHS="${EPOCHS:-70}"
LOCAL_STEPS="${LOCAL_STEPS:-1}"
MOMENTUM_MODE="${MOMENTUM_MODE:-ema}"
TARGET_DELTA="${TARGET_DELTA:-2e-5}"
TARGET_EPSILON="${TARGET_EPSILON:-8.0}"
SIGMA="${SIGMA:-0.1}"
ANCHOR_SIGMA="${ANCHOR_SIGMA:-$SIGMA}"

CUDA_ARGS=()
case "$USE_CUDA" in
  1|true|TRUE|yes|YES)
    CUDA_ARGS=(--use-cuda)
    ;;
  0|false|FALSE|no|NO)
    CUDA_ARGS=()
    ;;
  *)
    echo "Unknown USE_CUDA=$USE_CUDA. Use 1 or 0." >&2
    exit 2
    ;;
esac

case "$SWEEP_SIZE" in
  small)
    MUS=(${MUS:-0.9})
    C_RES_VALUES=(${C_RES_VALUES:-0.03 0.1 0.3})
    C_ANCHOR_VALUES=(${C_ANCHOR_VALUES:-1.0 3.0})
    ANCHOR_PERIODS=(${ANCHOR_PERIODS:-100 200 -1})
    LRS=(${LRS:-0.05})
    ;;
  full)
    MUS=(${MUS:-0.9 0.95})
    C_RES_VALUES=(${C_RES_VALUES:-0.01 0.03 0.1 0.3 1.0})
    C_ANCHOR_VALUES=(${C_ANCHOR_VALUES:-0.3 1.0 3.0 10.0})
    ANCHOR_PERIODS=(${ANCHOR_PERIODS:-5 10 20 50 -1})
    LRS=(${LRS:-0.03 0.05 0.1})
    ;;
  *)
    echo "Unknown SWEEP_SIZE=$SWEEP_SIZE. Use small or full." >&2
    exit 2
    ;;
esac

safe_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-1/noanchor}"
  echo "$value"
}

period_label() {
  local period="$1"
  if [[ "$period" == "-1" ]]; then
    echo "noanchor"
  else
    echo "K${period}"
  fi
}

print_shell_command() {
  local log_file="$1"
  shift
  printf 'nohup env PYTHONPATH=. '
  printf '%q ' "$@"
  printf '> %q 2>&1 &\n' "$log_file"
}

emit_or_run() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"

  if [[ "$DRY_RUN" == "1" ]]; then
    print_shell_command "$log_file" "$@"
  else
    echo "Starting: $log_file"
    env PYTHONPATH=. "$@" > "$log_file" 2>&1
  fi
}

run_ardp_grid() {
  local mode="$1"
  local run_prefix="$2"
  local log_root="$3"
  local sigma="$4"
  local anchor_sigma="$5"
  local force_oracle_clips="${6:-0}"
  local count=0

  echo "# mode=$mode sweep_size=$SWEEP_SIZE dry_run=$DRY_RUN"
  echo "# mus=${MUS[*]}"
  echo "# C_res=${C_RES_VALUES[*]}"
  echo "# C_anchor=${C_ANCHOR_VALUES[*]}"
  echo "# anchor_periods=${ANCHOR_PERIODS[*]}"
  echo "# lrs=${LRS[*]}"
  echo "# seeds=${SEEDS[*]}"

  for mu in "${MUS[@]}"; do
    for c_res in "${C_RES_VALUES[@]}"; do
      for c_anchor in "${C_ANCHOR_VALUES[@]}"; do
        for anchor_period in "${ANCHOR_PERIODS[@]}"; do
          for lr in "${LRS[@]}"; do
            for seed in "${SEEDS[@]}"; do
              local effective_c_res="$c_res"
              local effective_c_anchor="$c_anchor"
              if [[ "$force_oracle_clips" == "1" ]]; then
                effective_c_res="1000000.0"
                effective_c_anchor="1000000.0"
              fi

              local mu_label
              local cres_label
              local canchor_label
              local lr_label
              local period_name
              mu_label="$(safe_value "$mu")"
              cres_label="$(safe_value "$effective_c_res")"
              canchor_label="$(safe_value "$effective_c_anchor")"
              lr_label="$(safe_value "$lr")"
              period_name="$(period_label "$anchor_period")"

              local run_name
              run_name="${run_prefix}-mu${mu_label}-Cres${cres_label}-Canchor${canchor_label}-${period_name}-lr${lr_label}-seed${seed}"
              if [[ -n "$TARGET_EPSILON" ]]; then
                run_name="${run_name}-eps$(safe_value "$TARGET_EPSILON")"
              else
                run_name="${run_name}-sigma$(safe_value "$sigma")"
              fi
              local log_file
              log_file="${log_root}/${run_name}.out"

              local cmd=(
                python3
                momentum-robustness/cifar10-all-noattack.py
              )
              if [[ ${#CUDA_ARGS[@]} -gt 0 ]]; then
                cmd+=("${CUDA_ARGS[@]}")
              fi
              cmd+=(
                --model "$MODEL"
                --agg dp-residual-anchor
                --clip-tau "$effective_c_res"
                --anchor-clip-tau "$effective_c_anchor"
                --dp-noise-multiplier "$sigma"
                --dp-anchor-noise-multiplier "$anchor_sigma"
                --target-delta "$TARGET_DELTA"
                --momentum "$mu"
                --momentum-mode "$MOMENTUM_MODE"
                --residual-center-mode clipped-residual-ema
                --residual-alpha 1.0
                --anchor-period "$anchor_period"
                --lr "$lr"
                --local-steps "$LOCAL_STEPS"
                --epochs "$EPOCHS"
                --seed "$seed"
              )

              if [[ -n "$TARGET_EPSILON" ]]; then
                cmd+=(--target-epsilon "$TARGET_EPSILON")
              fi

              if [[ "$WANDB_ENABLED" == "1" ]]; then
                cmd+=(
                  --wandb
                  --wandb-entity "$WANDB_ENTITY"
                  --wandb-project "$WANDB_PROJECT"
                  --wandb-run-name "$run_name"
                )
              fi

              emit_or_run "$log_file" "${cmd[@]}"
              count=$((count + 1))
            done
          done
        done
      done
    done
  done

  echo "# total_commands=$count"
}
