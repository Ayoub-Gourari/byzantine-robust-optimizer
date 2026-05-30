#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN="${DRY_RUN:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-ae-gourari-cole-polytechnique}"
WANDB_PROJECT="${WANDB_PROJECT:-non_private_residual_clipping}"
USE_CUDA="${USE_CUDA:-1}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-128}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
BETA="${BETA:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
LOG_ROOT="${LOG_ROOT:-logs/residual_clipping_cifar10}"

LRS=(${LRS:-0.05 0.1})
CLIP_CS=(${CLIP_CS:-0.1 0.3 1.0 3.0})
C_RES_VALUES=(${C_RES_VALUES:-0.03 0.1 0.3 1.0})

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

COMMON_ARGS=(
  --model resnet20
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --test-batch-size "$TEST_BATCH_SIZE"
  --loader-workers "$LOADER_WORKERS"
  --beta "$BETA"
  --weight-decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --log-interval "$LOG_INTERVAL"
)
if [[ ${#CUDA_ARGS[@]} -gt 0 ]]; then
  COMMON_ARGS+=("${CUDA_ARGS[@]}")
fi
if [[ "$WANDB_ENABLED" == "1" ]]; then
  COMMON_ARGS+=(
    --wandb
    --wandb-entity "$WANDB_ENTITY"
    --wandb-project "$WANDB_PROJECT"
  )
fi
if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
  COMMON_ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ -n "${MAX_TEST_BATCHES:-}" ]]; then
  COMMON_ARGS+=(--max-test-batches "$MAX_TEST_BATCHES")
fi

safe_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  echo "$value"
}

emit_or_run() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'env PYTHONPATH=. '
    printf '%q ' "$@"
    printf '> %q 2>&1\n' "$log_file"
  else
    echo "Starting: $log_file"
    env PYTHONPATH=. "$@" > "$log_file" 2>&1
  fi
}

count=0

for lr in "${LRS[@]}"; do
  lr_label="$(safe_value "$lr")"
  run_name="central-resnet20-sgd-mom-lr${lr_label}-seed${SEED}"
  log_file="${LOG_ROOT}/${run_name}.out"
  cmd=(python3 momentum-robustness/cifar10_residual_clipping.py "${COMMON_ARGS[@]}")
  cmd+=(
    --optimizer-mode sgd_momentum
    --lr "$lr"
    --run-name "$run_name"
    --wandb-run-name "$run_name"
  )
  emit_or_run "$log_file" "${cmd[@]}"
  count=$((count + 1))
done

for lr in "${LRS[@]}"; do
  for clip_c in "${CLIP_CS[@]}"; do
    lr_label="$(safe_value "$lr")"
    c_label="$(safe_value "$clip_c")"
    run_name="central-resnet20-clipmom-C${c_label}-lr${lr_label}-seed${SEED}"
    log_file="${LOG_ROOT}/${run_name}.out"
    cmd=(python3 momentum-robustness/cifar10_residual_clipping.py "${COMMON_ARGS[@]}")
    cmd+=(
      --optimizer-mode clipped_momentum
      --clip-c "$clip_c"
      --lr "$lr"
      --run-name "$run_name"
      --wandb-run-name "$run_name"
    )
    emit_or_run "$log_file" "${cmd[@]}"
    count=$((count + 1))
  done
done

for lr in "${LRS[@]}"; do
  for c_res in "${C_RES_VALUES[@]}"; do
    lr_label="$(safe_value "$lr")"
    cres_label="$(safe_value "$c_res")"
    run_name="central-resnet20-resclipmom-Cres${cres_label}-lr${lr_label}-seed${SEED}"
    log_file="${LOG_ROOT}/${run_name}.out"
    cmd=(python3 momentum-robustness/cifar10_residual_clipping.py "${COMMON_ARGS[@]}")
    cmd+=(
      --optimizer-mode residual_clipped_momentum
      --clip-c-res "$c_res"
      --lr "$lr"
      --run-name "$run_name"
      --wandb-run-name "$run_name"
    )
    emit_or_run "$log_file" "${cmd[@]}"
    count=$((count + 1))
  done
done

echo "# total_commands=$count"
