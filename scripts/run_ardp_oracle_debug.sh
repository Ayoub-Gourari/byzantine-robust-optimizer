#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/ardp_sweep_lib.sh"

# Oracle debug: no DP noise and effectively no clipping. This checks whether the
# previous-momentum ARDP dynamics can optimize before privacy/clipping effects.
SIGMA="0.0"
ANCHOR_SIGMA="0.0"
TARGET_EPSILON=""
C_RES_VALUES=(1.0)
C_ANCHOR_VALUES=(1.0)

run_ardp_grid \
  "oracle" \
  "oracle-ardp-prevmom" \
  "logs/ardp_oracle_${SWEEP_SIZE}" \
  "$SIGMA" \
  "$ANCHOR_SIGMA" \
  "1"
