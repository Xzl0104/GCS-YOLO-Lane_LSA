#!/usr/bin/env bash
set -euo pipefail

# Interval-5 wrapper for the full official-val protocol.
# It keeps training-time official_best, post-train sweeps, and final test
# reporting for both official_best.pt and best.pt.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export RUN_NAME="${RUN_NAME:-query_alpha05_gt5short_geom_w2_interval5_v1}"
export OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-5}"

exec bash "${SCRIPT_DIR}/run_gt5_short_geom_full_protocol_v1.sh" "$@"
