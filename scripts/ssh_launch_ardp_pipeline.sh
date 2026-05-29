#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SWEEP_SIZE="${SWEEP_SIZE:-small}"
USE_CUDA="${USE_CUDA:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ssh_launch_ardp_pipeline.sh dry-run
  bash scripts/ssh_launch_ardp_pipeline.sh oracle
  bash scripts/ssh_launch_ardp_pipeline.sh clipping
  bash scripts/ssh_launch_ardp_pipeline.sh dp-small
  bash scripts/ssh_launch_ardp_pipeline.sh dp-full

Examples contained in this launcher:
  DRY_RUN=1 SWEEP_SIZE=small bash scripts/run_ardp_dp_sweep.sh
  nohup env DRY_RUN=0 SWEEP_SIZE=small USE_CUDA=1 bash scripts/run_ardp_oracle_debug.sh > /tmp/ardp_oracle_driver.out 2>&1 &
  nohup env DRY_RUN=0 SWEEP_SIZE=small USE_CUDA=1 bash scripts/run_ardp_clipping_debug.sh > /tmp/ardp_clipping_driver.out 2>&1 &
  nohup env DRY_RUN=0 SWEEP_SIZE=small USE_CUDA=1 bash scripts/run_ardp_dp_sweep.sh > /tmp/ardp_dp_small_driver.out 2>&1 &
EOF
}

case "${1:-}" in
  dry-run)
    DRY_RUN=1 SWEEP_SIZE="$SWEEP_SIZE" bash scripts/run_ardp_dp_sweep.sh
    ;;
  oracle)
    nohup env DRY_RUN=0 SWEEP_SIZE="$SWEEP_SIZE" USE_CUDA="$USE_CUDA" bash scripts/run_ardp_oracle_debug.sh > /tmp/ardp_oracle_driver.out 2>&1 &
    echo "oracle driver pid=$!"
    ;;
  clipping)
    nohup env DRY_RUN=0 SWEEP_SIZE="$SWEEP_SIZE" USE_CUDA="$USE_CUDA" bash scripts/run_ardp_clipping_debug.sh > /tmp/ardp_clipping_driver.out 2>&1 &
    echo "clipping driver pid=$!"
    ;;
  dp-small)
    nohup env DRY_RUN=0 SWEEP_SIZE=small USE_CUDA="$USE_CUDA" bash scripts/run_ardp_dp_sweep.sh > /tmp/ardp_dp_small_driver.out 2>&1 &
    echo "small DP sweep driver pid=$!"
    ;;
  dp-full)
    nohup env DRY_RUN=0 SWEEP_SIZE=full USE_CUDA="$USE_CUDA" bash scripts/run_ardp_dp_sweep.sh > /tmp/ardp_dp_full_driver.out 2>&1 &
    echo "full DP sweep driver pid=$!"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown action: $1" >&2
    usage >&2
    exit 2
    ;;
esac
