#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ardp_sweep_lib.sh"

# DP sweep: default is the small first-pass grid. Use SWEEP_SIZE=full for the
# full Cartesian product requested in the experiment plan.
SIGMA="${SIGMA:-0.1}"
ANCHOR_SIGMA="${ANCHOR_SIGMA:-$SIGMA}"

run_ardp_grid \
  "dp" \
  "dp-ardp-prevmom" \
  "logs/ardp_dp_${SWEEP_SIZE}" \
  "$SIGMA" \
  "$ANCHOR_SIGMA" \
  "0"

