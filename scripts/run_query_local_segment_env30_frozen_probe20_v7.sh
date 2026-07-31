#!/usr/bin/env bash
set -euo pipefail

# Frozen-env30 local short-segment v7 probe. It reuses the v6 proposal-local
# head, but aligns training and diagnostic selection with dense geometry-quality
# negatives and a combined quality+replace selector score.

export RUN_NAME="${RUN_NAME:-query_local_segment_env30_frozen_probe20_v7}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v7.yaml}"

export SHORT_SEGMENT_BCE_WEIGHT="${SHORT_SEGMENT_BCE_WEIGHT:-0.0}"
export SHORT_SEGMENT_LISTWISE_WEIGHT="${SHORT_SEGMENT_LISTWISE_WEIGHT:-1.0}"
export SHORT_SEGMENT_LISTWISE_ALL_CANDIDATES="${SHORT_SEGMENT_LISTWISE_ALL_CANDIDATES:-1}"
export SHORT_SEGMENT_DENSE_QUALITY_WEIGHT="${SHORT_SEGMENT_DENSE_QUALITY_WEIGHT:-1.0}"
export SHORT_SEGMENT_DENSE_NEG_WEIGHT="${SHORT_SEGMENT_DENSE_NEG_WEIGHT:-0.05}"
export SHORT_SEGMENT_REPLACE_WEIGHT="${SHORT_SEGMENT_REPLACE_WEIGHT:-1.0}"
export SHORT_SEGMENT_REPLACE_DENSE_NEG_WEIGHT="${SHORT_SEGMENT_REPLACE_DENSE_NEG_WEIGHT:-0.25}"
export SHORT_SEGMENT_BASE_PRESERVE="${SHORT_SEGMENT_BASE_PRESERVE:-1}"

export HARD_CANDIDATE_SCORE_THR="${HARD_CANDIDATE_SCORE_THR:-0.5}"
export HARD_SEGMENT_SELECTION_SCORE_MODE="${HARD_SEGMENT_SELECTION_SCORE_MODE:-combined}"
export HARD_SEGMENT_SELECTION_PRED_VALID_OVERLAP_MIN="${HARD_SEGMENT_SELECTION_PRED_VALID_OVERLAP_MIN:-0}"

bash scripts/run_query_local_segment_env30_frozen_probe20_v5.sh
