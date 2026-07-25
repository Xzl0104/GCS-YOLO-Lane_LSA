#!/usr/bin/env bash
set -euo pipefail

# Default-off Q12/env30 dual-head probe.
# Count is supplied by pred_count_logits; lane ranking/threshold score is supplied
# by pred_quality_logits. TEST stays closed by default.

export RUN_NAME="${RUN_NAME:-query_dualhead_quality_count_env30_probe40_v1}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-dualhead.yaml}"
export EPOCHS="${EPOCHS:-40}"
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

DUAL_EXTRA_ARGS=(
  --gcs-count 0.0
  --gcs-count-under5 0.0
  --gcs-count-boundary 0.0
  --gcs-query-count-ce "${GCS_QUERY_COUNT_CE:-0.5}"
  --gcs-query-quality "${GCS_QUERY_QUALITY:-0.5}"
  --gcs-boundary-pseudo-neg 0.0
  --gcs-role-contain 0.0
  --gcs-q24-event-contain 0.0
  --gcs-q24-event-score-calib 0.0
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} ${DUAL_EXTRA_ARGS[*]}"
else
  export EXTRA_TRAIN_ARGS="${DUAL_EXTRA_ARGS[*]}"
fi

echo "Dual-head probe: model=${MODEL}, count_mode=${OFFICIAL_COUNT_MODES}, count_aware_topk=${COUNT_AWARE_TOPK}, tests=${RUN_TESTS}"
echo "Dual-head gains: gcs_query_count_ce=${GCS_QUERY_COUNT_CE:-0.5}, gcs_query_quality=${GCS_QUERY_QUALITY:-0.5}"
echo "Disabled in this probe: gcs-count, gcs-count-under5, gcs-count-boundary, gcs-boundary-pseudo-neg, Q24 role/event losses."

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
