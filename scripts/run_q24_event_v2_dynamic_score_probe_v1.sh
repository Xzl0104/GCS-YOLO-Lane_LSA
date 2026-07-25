#!/usr/bin/env bash
set -euo pipefail

# Q24 event-v2 dynamic containment + true-short score calibration probe.
# Keep TEST closed; this is only a 20-40 epoch official-val/train-side gate.

export RUN_NAME="${RUN_NAME:-query_alpha05_env30_q24_event_v2_dynamic_score_probe40_v1}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q24-k56-protected-static.yaml}"
export EPOCHS="${EPOCHS:-40}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OVERWRITE_SWEEPS="${OVERWRITE_SWEEPS:-0}"

export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---gcs-boundary-pseudo-gt5-safe --gcs-boundary-pseudo-score-thr 0.0 --gcs-boundary-pseudo-protect-short-visible-thr 10 --gcs-boundary-pseudo-protect-dist-px 40 --gcs-boundary-pseudo-protect-min-overlap 3 --gcs-boundary-pseudo-protect-queries 12,13,15 --gcs-q24-event-contain 0.35 --gcs-q24-event-valid-weight 0.20 --gcs-q24-event-matcher --gcs-q24-event-clean-gt5-queries 13,15 --gcs-q24-event-risk-queries 16,21,22,23 --gcs-q24-event-gt4-queries 14,17,18,19 --gcs-q24-event-gt4-visible-thr 20 --gcs-q24-event-gt5-visible-thr 10 --gcs-q24-event-suppress-gt5-risk --gcs-q24-event-gt5-risk-protect --gcs-q24-event-gt5-risk-protect-short-visible-thr 10 --gcs-q24-event-gt5-risk-protect-dist-px 25 --gcs-q24-event-gt5-risk-protect-min-overlap 3 --gcs-q24-event-dynamic --gcs-q24-event-dynamic-queries 0,1,11,20 --gcs-q24-event-dynamic-valid-thr 0.5 --gcs-q24-event-dynamic-min-valid 3 --gcs-q24-event-dynamic-max-visible 20 --gcs-q24-event-dynamic-score-thr 0.0 --gcs-q24-event-dynamic-protect --gcs-q24-event-dynamic-protect-visible-thr 10 --gcs-q24-event-dynamic-protect-dist-px 25 --gcs-q24-event-dynamic-protect-min-overlap 3 --gcs-q24-event-score-calib 0.15 --gcs-q24-event-score-queries 13,15 --gcs-q24-event-score-visible-thr 10 --gcs-q24-event-score-valid-thr 0.5 --gcs-q24-event-score-dist-px 40 --gcs-q24-event-score-min-overlap 3 --gcs-q24-event-score-target 0.75}"

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
