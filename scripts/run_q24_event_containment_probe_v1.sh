#!/usr/bin/env bash
set -euo pipefail

# Q24 event-mined containment probe. Keep TEST closed.
# Isolates clean GT5 true-short carriers and contains high-risk event carriers.

export RUN_NAME="${RUN_NAME:-query_alpha05_env30_q24_event_containment_probe40_v1}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q24-k56-protected-static.yaml}"
export EPOCHS="${EPOCHS:-40}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OVERWRITE_SWEEPS="${OVERWRITE_SWEEPS:-0}"

export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---gcs-boundary-pseudo-gt5-safe --gcs-boundary-pseudo-score-thr 0.0 --gcs-boundary-pseudo-protect-short-visible-thr 10 --gcs-boundary-pseudo-protect-dist-px 40 --gcs-boundary-pseudo-protect-min-overlap 3 --gcs-boundary-pseudo-protect-queries 12,15,20 --gcs-q24-event-contain 0.35 --gcs-q24-event-valid-weight 0.20 --gcs-q24-event-matcher --gcs-q24-event-clean-gt5-queries 12,15,20 --gcs-q24-event-risk-queries 13,21,22,23 --gcs-q24-event-gt4-queries 14,17,18,19 --gcs-q24-event-gt4-visible-thr 20 --gcs-q24-event-gt5-visible-thr 10 --gcs-q24-event-suppress-gt5-risk}"

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
