#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ardp_sweep_lib.sh"

# Clipping-only debug: use the requested clipping grid but remove DP noise.
SIGMA="0.0"
ANCHOR_SIGMA="0.0"
TARGET_EPSILON=""

run_ardp_grid \
  "clipping" \
  "clipdebug-ardp-prevmom" \
  "logs/ardp_clipping_${SWEEP_SIZE}" \
  "$SIGMA" \
  "$ANCHOR_SIGMA" \
  "0"
