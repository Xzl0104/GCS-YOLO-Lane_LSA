#!/usr/bin/env bash
set -euo pipefail

# Q12/env30 Count Head only probe.
# Count logits choose k_hat; original pred_logits keep lane ranking/thresholding.
# No Quality Head, no Q24 bank, no role/event containment, and TEST stays closed.

export RUN_NAME="${RUN_NAME:-query_count_head_ce025_env30_probe100_v1}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count.yaml}"
export EPOCHS="${EPOCHS:-100}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-count_logits}"
export SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-${OFFICIAL_COUNT_MODES}}"
export COUNT_AWARE_TOPK="${COUNT_AWARE_TOPK:-1}"
export SWEEP_COUNT_AWARE_TOPK="${SWEEP_COUNT_AWARE_TOPK:-${COUNT_AWARE_TOPK}}"
export COUNT_AWARE_MIN_K="${COUNT_AWARE_MIN_K:-2}"
export COUNT_AWARE_MAX_K="${COUNT_AWARE_MAX_K:-5}"
export COUNT_AWARE_LENGTH_NORM="${COUNT_AWARE_LENGTH_NORM:-12.0}"
export COUNT_AWARE_EXTRA_MARGINS="${COUNT_AWARE_EXTRA_MARGINS:-0}"
export SWEEP_COUNT_AWARE_EXTRA_MARGINS="${SWEEP_COUNT_AWARE_EXTRA_MARGINS:-${COUNT_AWARE_EXTRA_MARGINS}}"

COUNT_ONLY_EXTRA_ARGS=(
  --gcs-count 0.0
  --gcs-count-under5 0.0
  --gcs-count-boundary 0.0
  --gcs-query-count-ce "${GCS_QUERY_COUNT_CE:-0.25}"
  --gcs-query-quality 0.0
  --gcs-role-contain 0.0
  --gcs-q24-event-contain 0.0
  --gcs-q24-event-score-calib 0.0
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} ${COUNT_ONLY_EXTRA_ARGS[*]}"
else
  export EXTRA_TRAIN_ARGS="${COUNT_ONLY_EXTRA_ARGS[*]}"
fi

echo "Q12 Count Head only probe: model=${MODEL}, epochs=${EPOCHS}, tests=${RUN_TESTS}"
echo "Decode: count_logits count-aware top-k chooses k; pred_logits ranks lanes because this model has no Quality Head."
echo "Count CE gain: gcs_query_count_ce=${GCS_QUERY_COUNT_CE:-0.25}; gcs_query_quality=0.0"
echo "Inherited env30 geometry path: short-geom and lightweight boundary-pseudo from parent script."
echo "Disabled: score-sum count losses, Quality Head loss, Q24 role/event losses."

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
