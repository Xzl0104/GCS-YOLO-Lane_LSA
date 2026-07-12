#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the remote CUDA server.
# Tiered matched GT4/GT5 weak-positive geometry rescue, GT5 ultra-short point-valid rescue, and env30 boundary mask.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-query_alpha05_gt4gt5_tiered_geom_env30_nocount_v1}"
export SHORT_GEOM="${SHORT_GEOM:-1.0}"
export SHORT_GEOM_TIERED="${SHORT_GEOM_TIERED:-1}"
export SHORT_GEOM_GT4_ULTRA_VISIBLE_THR="${SHORT_GEOM_GT4_ULTRA_VISIBLE_THR:-10}"
export SHORT_GEOM_GT4_ULTRA_WEIGHT="${SHORT_GEOM_GT4_ULTRA_WEIGHT:-1.35}"
export SHORT_GEOM_GT4_MID_VISIBLE_THR="${SHORT_GEOM_GT4_MID_VISIBLE_THR:-20}"
export SHORT_GEOM_GT4_MID_WEIGHT="${SHORT_GEOM_GT4_MID_WEIGHT:-1.0}"
export SHORT_GEOM_GT5_ULTRA_VISIBLE_THR="${SHORT_GEOM_GT5_ULTRA_VISIBLE_THR:-10}"
export SHORT_GEOM_GT5_ULTRA_WEIGHT="${SHORT_GEOM_GT5_ULTRA_WEIGHT:-2.25}"
export SHORT_GEOM_GT5_MID_VISIBLE_THR="${SHORT_GEOM_GT5_MID_VISIBLE_THR:-20}"
export SHORT_GEOM_GT5_MID_WEIGHT="${SHORT_GEOM_GT5_MID_WEIGHT:-1.5}"
export SHORT_GEOM_MAX_WEIGHT="${SHORT_GEOM_MAX_WEIGHT:-3.0}"
export SHORT_GEOM_CURVE="${SHORT_GEOM_CURVE:-1.0}"
export GT5_SHORT_VISIBLE_THR="${GT5_SHORT_VISIBLE_THR:-10}"
export GT5_SHORT_POINT_VALID_WEIGHT="${GT5_SHORT_POINT_VALID_WEIGHT:-1.5}"

exec bash "${SCRIPT_DIR}/run_query_alpha05_gt4gt5weak_geom_w15w2_env30_nocount_v1.sh" "$@"
