#!/usr/bin/env bash
set -euo pipefail

# Q24 protected-static GT5-safe boundary-pseudo probe. Keep TEST closed.
# This probe does not enable hard role-partition matcher constraints by default.

export RUN_NAME="${RUN_NAME:-query_alpha05_env30_q24_gt5safe_boundary_probe40_v1}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q24-k56-protected-static.yaml}"
export EPOCHS="${EPOCHS:-40}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OVERWRITE_SWEEPS="${OVERWRITE_SWEEPS:-0}"

export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---gcs-boundary-pseudo-gt5-safe --gcs-boundary-pseudo-score-thr 0.0 --gcs-boundary-pseudo-protect-short-visible-thr 10 --gcs-boundary-pseudo-protect-dist-px 40 --gcs-boundary-pseudo-protect-min-overlap 3 --gcs-boundary-pseudo-protect-queries 12,13,15,16,20,21,23}"

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
