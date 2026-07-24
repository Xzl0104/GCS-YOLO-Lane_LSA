#!/usr/bin/env bash
set -euo pipefail

# Q24 protected-static role-containment probe. Keep TEST closed.

export RUN_NAME="${RUN_NAME:-query_alpha05_env30_q24_role_containment_probe40_v1}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q24-k56-protected-static.yaml}"
export EPOCHS="${EPOCHS:-40}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OVERWRITE_SWEEPS="${OVERWRITE_SWEEPS:-0}"

export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---gcs-role-contain 0.5 --gcs-role-contain-valid-weight 0.25 --gcs-role-contain-matcher --gcs-role-gt5-queries 12-17 --gcs-role-gt4-queries 18-23 --gcs-role-gt4-visible-thr 20 --gcs-role-gt5-visible-thr 20}"

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
