#!/usr/bin/env bash
set -euo pipefail

# Default-off Q12/env30 short-lane coarse-to-fine local x-refine probe.
# Keeps Count Head, Quality Head, and query extent decode disabled.
# TEST stays closed; use official-val and raw/refined geometry gates first.

export RUN_NAME="${RUN_NAME:-query_short_local_refine_env30_probe40_v1}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-short-local-refine.yaml}"
export EPOCHS="${EPOCHS:-40}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OFFICIAL_EXTENT_DECODE_MODES="${OFFICIAL_EXTENT_DECODE_MODES:-none}"
export SWEEP_EXTENT_DECODE_MODES="${SWEEP_EXTENT_DECODE_MODES:-${OFFICIAL_EXTENT_DECODE_MODES}}"
export OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"
export SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-${OFFICIAL_COUNT_MODES}}"
export COUNT_AWARE_TOPK="${COUNT_AWARE_TOPK:-0}"
export SWEEP_COUNT_AWARE_TOPK="${SWEEP_COUNT_AWARE_TOPK:-${COUNT_AWARE_TOPK}}"

SHORT_REFINE_EXTRA_ARGS=(
  --gcs-short-local-refine "${GCS_SHORT_LOCAL_REFINE:-0.25}"
  --gcs-short-local-refine-visible-thr "${GCS_SHORT_LOCAL_REFINE_VISIBLE_THR:-10}"
  --gcs-short-local-refine-gt-min-lanes "${GCS_SHORT_LOCAL_REFINE_GT_MIN_LANES:-4}"
  --gcs-short-local-refine-beta-px "${GCS_SHORT_LOCAL_REFINE_BETA_PX:-5.0}"
  --gcs-query-count-ce 0.0
  --gcs-query-quality 0.0
  --gcs-query-extent 0.0
  --gcs-count 0.0
  --gcs-count-under5 0.0
  --gcs-count-boundary 0.0
  --gcs-role-contain 0.0
  --gcs-q24-event-contain 0.0
  --gcs-q24-event-score-calib 0.0
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} ${SHORT_REFINE_EXTRA_ARGS[*]}"
else
  export EXTRA_TRAIN_ARGS="${SHORT_REFINE_EXTRA_ARGS[*]}"
fi

echo "Q12/env30 short local x-refine probe: model=${MODEL}, epochs=${EPOCHS}, tests=${RUN_TESTS}"
echo "Short local refine loss gain: gcs_short_local_refine=${GCS_SHORT_LOCAL_REFINE:-0.25}"
echo "Official extent decode modes: ${OFFICIAL_EXTENT_DECODE_MODES}"
echo "Count-aware top-k: ${COUNT_AWARE_TOPK}"
echo "Inherited env30 parent protocol, including gcs_short_geom=1.0 and parent boundary-pseudo settings."
echo "Disabled: Count Head CE, Quality Head loss, query extent loss/decode, score-sum count losses, Q24 role/event losses."

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
