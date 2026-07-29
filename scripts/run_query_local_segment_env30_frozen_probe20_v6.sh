#!/usr/bin/env bash
set -euo pipefail

# Frozen-env30 local short-segment v6 probe. It reuses the v5 train/diagnostic
# protocol but enables proposal-local evidence in the v6 YAML.

export RUN_NAME="${RUN_NAME:-query_local_segment_env30_frozen_probe20_v6}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v6.yaml}"

bash scripts/run_query_local_segment_env30_frozen_probe20_v5.sh
