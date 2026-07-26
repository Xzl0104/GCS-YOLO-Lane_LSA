#!/usr/bin/env bash
set -euo pipefail

# Default-off Q12/env30 query extent probe.
# Adds query start/end extent supervision on top of the env30 parent protocol and
# sweeps no-extent vs interval/intersect visibility decode on official-val.
# Count Head and Quality Head stay disabled.

export RUN_NAME="${RUN_NAME:-query_extent_env30_probe40_v1}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-extent.yaml}"
export EPOCHS="${EPOCHS:-40}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OFFICIAL_EXTENT_DECODE_MODES="${OFFICIAL_EXTENT_DECODE_MODES:-none interval intersect}"
export SWEEP_EXTENT_DECODE_MODES="${SWEEP_EXTENT_DECODE_MODES:-${OFFICIAL_EXTENT_DECODE_MODES}}"
export OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"
export SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-${OFFICIAL_COUNT_MODES}}"
export COUNT_AWARE_TOPK="${COUNT_AWARE_TOPK:-0}"
export SWEEP_COUNT_AWARE_TOPK="${SWEEP_COUNT_AWARE_TOPK:-${COUNT_AWARE_TOPK}}"
export GCS_BOUNDARY_PSEUDO_NEG="${GCS_BOUNDARY_PSEUDO_NEG:-0.0}"

EXTENT_EXTRA_ARGS=(
  --gcs-query-extent "${GCS_QUERY_EXTENT:-0.5}"
  --gcs-query-extent-short-visible-thr "${GCS_QUERY_EXTENT_SHORT_VISIBLE_THR:-10}"
  --gcs-query-extent-short-weight "${GCS_QUERY_EXTENT_SHORT_WEIGHT:-2.0}"
  --gcs-query-extent-gt-min-lanes "${GCS_QUERY_EXTENT_GT_MIN_LANES:-4}"
  --gcs-query-count-ce 0.0
  --gcs-query-quality 0.0
  --gcs-count 0.0
  --gcs-count-under5 0.0
  --gcs-count-boundary 0.0
  --gcs-boundary-pseudo-neg 0.0
  --gcs-role-contain 0.0
  --gcs-q24-event-contain 0.0
  --gcs-q24-event-score-calib 0.0
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} ${EXTENT_EXTRA_ARGS[*]}"
else
  export EXTRA_TRAIN_ARGS="${EXTENT_EXTRA_ARGS[*]}"
fi

echo "Q12 query extent probe: model=${MODEL}, epochs=${EPOCHS}, tests=${RUN_TESTS}"
echo "Extent loss gain: gcs_query_extent=${GCS_QUERY_EXTENT:-0.5}"
echo "Official extent decode modes: ${OFFICIAL_EXTENT_DECODE_MODES}"
echo "Count-aware top-k: ${COUNT_AWARE_TOPK} (keep 0 for extent-only interpretation; if enabled, ranking still uses point-valid quality)."
echo "Inherited env30 parent protocol: gcs_short_geom=1.0 from the parent launch script."
echo "Disabled: Count Head CE, Quality Head loss, score-sum count losses, boundary-pseudo loss, Q24 role/event losses."

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
