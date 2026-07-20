#!/usr/bin/env bash
set -euo pipefail

# GT4/GT5 constrained static reference v2 full protocol.
# Training-time official-val selection runs every 5 epochs with a coarse grid.
# Post-train official-val sweeps use a finer grid, then both official_best.pt
# and best.pt are evaluated on test as reporting-only results.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-query_alpha05_env30_gt45staticref_v2}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-env30-gt45staticref-v2.yaml}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_TESTS="${RUN_TESTS:-1}"
export OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-5}"
export OFFICIAL_CONFS="${OFFICIAL_CONFS:-0.003 0.005 0.008 0.01 0.015 0.02}"
export OFFICIAL_POINT_VALID_THRS="${OFFICIAL_POINT_VALID_THRS:-0.45 0.5 0.55 0.6}"
export OFFICIAL_NMS_DIST_PXS="${OFFICIAL_NMS_DIST_PXS:-0 18 30}"
export OFFICIAL_MAX_DETS="${OFFICIAL_MAX_DETS:-5 6}"
export OFFICIAL_MIN_POINTS="${OFFICIAL_MIN_POINTS:-2 3 4}"
export OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"
export SWEEP_CONFS="${SWEEP_CONFS:-0.001 0.003 0.005 0.008 0.01 0.012 0.015 0.02}"
export SWEEP_POINT_VALID_THRS="${SWEEP_POINT_VALID_THRS:-0.45 0.475 0.5 0.525 0.55 0.575 0.6}"
export SWEEP_NMS_DIST_PXS="${SWEEP_NMS_DIST_PXS:-0 12 18 24 30}"
export SWEEP_MAX_DETS="${SWEEP_MAX_DETS:-5 6 8}"
export SWEEP_MIN_POINTS="${SWEEP_MIN_POINTS:-2 3 4}"
export SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-score_sum}"
export BOUNDARY_PSEUDO_VISIBLE_THR="${BOUNDARY_PSEUDO_VISIBLE_THR:-12}"
export BOUNDARY_PSEUDO_DIST_THR="${BOUNDARY_PSEUDO_DIST_THR:-80}"
export BOUNDARY_PSEUDO_VALID_THR="${BOUNDARY_PSEUDO_VALID_THR:-0.5}"
export BOUNDARY_PSEUDO_MIN_VALID="${BOUNDARY_PSEUDO_MIN_VALID:-4}"
export BOUNDARY_PSEUDO_GT_COUNT="${BOUNDARY_PSEUDO_GT_COUNT:-4}"
export BOUNDARY_PSEUDO_MAX_GT_COUNT="${BOUNDARY_PSEUDO_MAX_GT_COUNT:-5}"
export BOUNDARY_PSEUDO_SCORE_THR="${BOUNDARY_PSEUDO_SCORE_THR:-0.05}"
export BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX="${BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX:-30}"
export BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR="${BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR:-0.75}"
export BOUNDARY_PSEUDO_VALID_NEG_WEIGHT="${BOUNDARY_PSEUDO_VALID_NEG_WEIGHT:-0.25}"

exec bash "${SCRIPT_DIR}/run_query_alpha05_env30_gt5staticref_v1.sh"
