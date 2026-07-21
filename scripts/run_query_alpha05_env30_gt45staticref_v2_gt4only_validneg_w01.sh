#!/usr/bin/env bash
set -euo pipefail

# GT4-only boundary-pseudo valid-negative ablation on top of the constrained
# env30 GT4/GT5 static-reference v2 bank.
#
# Default protocol:
# - keep the v2 static reference bank and env30 short-geometry settings
# - scope boundary-pseudo negatives to GT4 images only
# - reduce point-valid negative pressure from v2's 0.25 to 0.10
# - keep TEST closed by default; set RUN_TESTS=1 only after official-val and
#   train0601/train0531 diagnostics pass the gate.
#
# To run the no-valid-neg companion without another script:
#   BOUNDARY_PSEUDO_VALID_NEG_WEIGHT=0.0 \
#   RUN_NAME=query_alpha05_env30_gt45staticref_v2_gt4only_validneg_w00 \
#   bash scripts/run_query_alpha05_env30_gt45staticref_v2_gt4only_validneg_w01.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-query_alpha05_env30_gt45staticref_v2_gt4only_validneg_w01}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-env30-gt45staticref-v2.yaml}"

export BOUNDARY_PSEUDO_GT_COUNT="${BOUNDARY_PSEUDO_GT_COUNT:-4}"
export BOUNDARY_PSEUDO_MAX_GT_COUNT="${BOUNDARY_PSEUDO_MAX_GT_COUNT:-4}"
export BOUNDARY_PSEUDO_VALID_NEG_WEIGHT="${BOUNDARY_PSEUDO_VALID_NEG_WEIGHT:-0.1}"

# Preserve v2's narrower clear-far definition unless explicitly overridden.
export BOUNDARY_PSEUDO_VISIBLE_THR="${BOUNDARY_PSEUDO_VISIBLE_THR:-12}"
export BOUNDARY_PSEUDO_DIST_THR="${BOUNDARY_PSEUDO_DIST_THR:-80}"
export BOUNDARY_PSEUDO_VALID_THR="${BOUNDARY_PSEUDO_VALID_THR:-0.5}"
export BOUNDARY_PSEUDO_MIN_VALID="${BOUNDARY_PSEUDO_MIN_VALID:-4}"
export BOUNDARY_PSEUDO_SCORE_THR="${BOUNDARY_PSEUDO_SCORE_THR:-0.05}"
export BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX="${BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX:-30}"
export BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR="${BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR:-0.75}"

export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-5}"

exec bash "${SCRIPT_DIR}/run_query_alpha05_env30_gt45staticref_v2.sh"
