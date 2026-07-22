#!/usr/bin/env bash
set -euo pipefail

# Narrow env30 follow-up for GT4 short near-miss geometry.
# This wrapper does not change reference banks, decode, count losses, or valid-negative pressure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-query_alpha05_env30_gt4_near20_geom_refine_v1}"
export RUN_TESTS="${RUN_TESTS:-0}"

# First-pass defaults: only matched q0/q10 GT4 short lanes with x-APE in [20, 25] px.
export GT4_NEAR20_GEOM_REFINE="${GT4_NEAR20_GEOM_REFINE:-2.0}"
export GT4_NEAR20_VISIBLE_THR="${GT4_NEAR20_VISIBLE_THR:-10}"
export GT4_NEAR20_LOWER_PX="${GT4_NEAR20_LOWER_PX:-20.0}"
export GT4_NEAR20_UPPER_PX="${GT4_NEAR20_UPPER_PX:-25.0}"
export GT4_NEAR20_MIN_OVERLAP="${GT4_NEAR20_MIN_OVERLAP:-3}"
export GT4_NEAR20_ALLOWED_QUERIES="${GT4_NEAR20_ALLOWED_QUERIES:-0,10}"
export GT4_NEAR20_GEOM_CURVE="${GT4_NEAR20_GEOM_CURVE:-0.05}"

export OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-5}"
export OFFICIAL_WARMUP="${OFFICIAL_WARMUP:-5}"
export VALID_BEFORE_MAXDET="${VALID_BEFORE_MAXDET:-1}"

exec bash "${SCRIPT_DIR}/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh" "$@"
