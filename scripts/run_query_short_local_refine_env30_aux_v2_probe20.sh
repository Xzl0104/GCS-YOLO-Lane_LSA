#!/usr/bin/env bash
set -euo pipefail

# Superseded Q12/env30 auxiliary short-lane local x-refine v2 probe.
# Do not use as the current next experiment; use window v3 instead.
# Main pred_points/decode stay on the env30 path; pred_short_refined_points is
# auxiliary-only and is judged by raw/refined geometry diagnostics.

export RUN_NAME="${RUN_NAME:-query_short_local_refine_env30_aux_v2_probe20}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-short-local-refine.yaml}"
export PRETRAINED="${PRETRAINED:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt}"
export EPOCHS="${EPOCHS:-20}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OFFICIAL_EXTENT_DECODE_MODES="${OFFICIAL_EXTENT_DECODE_MODES:-none}"
export SWEEP_EXTENT_DECODE_MODES="${SWEEP_EXTENT_DECODE_MODES:-${OFFICIAL_EXTENT_DECODE_MODES}}"
export OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"
export SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-${OFFICIAL_COUNT_MODES}}"
export COUNT_AWARE_TOPK="${COUNT_AWARE_TOPK:-0}"
export SWEEP_COUNT_AWARE_TOPK="${SWEEP_COUNT_AWARE_TOPK:-${COUNT_AWARE_TOPK}}"

SHORT_REFINE_EXTRA_ARGS=(
  --gcs-short-local-refine "${GCS_SHORT_LOCAL_REFINE:-0.02}"
  --gcs-short-local-refine-visible-thr "${GCS_SHORT_LOCAL_REFINE_VISIBLE_THR:-10}"
  --gcs-short-local-refine-gt-min-lanes "${GCS_SHORT_LOCAL_REFINE_GT_MIN_LANES:-4}"
  --gcs-short-local-refine-beta-px "${GCS_SHORT_LOCAL_REFINE_BETA_PX:-5.0}"
  --gcs-short-local-refine-max-delta-px "${GCS_SHORT_LOCAL_REFINE_MAX_DELTA_PX:-40.0}"
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

echo "Q12/env30 auxiliary short local x-refine v2: model=${MODEL}, epochs=${EPOCHS}, tests=${RUN_TESTS}"
echo "Pretrained: ${PRETRAINED}"
echo "Short local refine: gain=${GCS_SHORT_LOCAL_REFINE:-0.02}, beta_px=${GCS_SHORT_LOCAL_REFINE_BETA_PX:-5.0}, max_delta_px=${GCS_SHORT_LOCAL_REFINE_MAX_DELTA_PX:-40.0}"
echo "Official extent decode modes: ${OFFICIAL_EXTENT_DECODE_MODES}"
echo "Count-aware top-k: ${COUNT_AWARE_TOPK}"
echo "Disabled: Count Head CE, Quality Head loss, query extent loss/decode, score-sum count losses, Q24 role/event losses."

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
