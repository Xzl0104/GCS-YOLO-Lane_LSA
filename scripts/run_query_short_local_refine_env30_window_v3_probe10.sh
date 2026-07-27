#!/usr/bin/env bash
set -euo pipefail

# Default-off Q12/env30 short-lane local x-refine v3 probe.
# The env30 main path is frozen; only the windowed auxiliary local-refine head trains.
# TEST stays closed until official-val and raw/refined geometry gates pass.

export RUN_NAME="${RUN_NAME:-query_short_local_refine_env30_window_v3_probe10}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-short-local-refine-v3.yaml}"
export PRETRAINED="${PRETRAINED:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt}"
export EPOCHS="${EPOCHS:-10}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OFFICIAL_EXTENT_DECODE_MODES="${OFFICIAL_EXTENT_DECODE_MODES:-none}"
export SWEEP_EXTENT_DECODE_MODES="${SWEEP_EXTENT_DECODE_MODES:-${OFFICIAL_EXTENT_DECODE_MODES}}"
export OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"
export SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-${OFFICIAL_COUNT_MODES}}"
export COUNT_AWARE_TOPK="${COUNT_AWARE_TOPK:-0}"
export SWEEP_COUNT_AWARE_TOPK="${SWEEP_COUNT_AWARE_TOPK:-${COUNT_AWARE_TOPK}}"

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if is_true "${RUN_TESTS}"; then
  echo "RUN_TESTS must stay 0 for the first short local x-refine v3 probe. TEST is closed." >&2
  exit 2
fi

SHORT_REFINE_EXTRA_ARGS=(
  --gcs-short-local-refine "${GCS_SHORT_LOCAL_REFINE:-0.05}"
  --gcs-short-local-refine-visible-thr "${GCS_SHORT_LOCAL_REFINE_VISIBLE_THR:-10}"
  --gcs-short-local-refine-gt-min-lanes "${GCS_SHORT_LOCAL_REFINE_GT_MIN_LANES:-4}"
  --gcs-short-local-refine-beta-px "${GCS_SHORT_LOCAL_REFINE_BETA_PX:-3.0}"
  --gcs-short-local-refine-max-delta-px "${GCS_SHORT_LOCAL_REFINE_MAX_DELTA_PX:-60.0}"
  --gcs-short-local-refine-window-search
  --gcs-short-local-refine-window-radius-px "${GCS_SHORT_LOCAL_REFINE_WINDOW_RADIUS_PX:-60.0}"
  --gcs-short-local-refine-window-step-px "${GCS_SHORT_LOCAL_REFINE_WINDOW_STEP_PX:-20.0}"
  --gcs-short-local-refine-freeze-base
  --gcs-short-local-refine-identity-guard
  --gcs-short-local-refine-identity-thr-px "${GCS_SHORT_LOCAL_REFINE_IDENTITY_THR_PX:-20.0}"
  --gcs-short-local-refine-nearmiss-thr-px "${GCS_SHORT_LOCAL_REFINE_NEARMISS_THR_PX:-80.0}"
  --gcs-short-local-refine-identity-weight "${GCS_SHORT_LOCAL_REFINE_IDENTITY_WEIGHT:-1.0}"
  --gcs-short-local-refine-nearmiss-weight "${GCS_SHORT_LOCAL_REFINE_NEARMISS_WEIGHT:-1.0}"
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

echo "Q12/env30 short local x-refine v3: model=${MODEL}, epochs=${EPOCHS}, tests=${RUN_TESTS}"
echo "Pretrained: ${PRETRAINED}"
echo "Short local refine: gain=${GCS_SHORT_LOCAL_REFINE:-0.05}, beta_px=${GCS_SHORT_LOCAL_REFINE_BETA_PX:-3.0}, max_delta_px=${GCS_SHORT_LOCAL_REFINE_MAX_DELTA_PX:-60.0}"
echo "Window search: radius_px=${GCS_SHORT_LOCAL_REFINE_WINDOW_RADIUS_PX:-60.0}, step_px=${GCS_SHORT_LOCAL_REFINE_WINDOW_STEP_PX:-20.0}"
echo "Identity guard: identity_thr_px=${GCS_SHORT_LOCAL_REFINE_IDENTITY_THR_PX:-20.0}, nearmiss_thr_px=${GCS_SHORT_LOCAL_REFINE_NEARMISS_THR_PX:-80.0}"
echo "Freeze: env30/base parameters frozen; only query_short_local_refine head trains."
echo "Official extent decode modes: ${OFFICIAL_EXTENT_DECODE_MODES}"
echo "Count-aware top-k: ${COUNT_AWARE_TOPK}"
echo "Disabled: Count Head CE, Quality Head loss, query extent loss/decode, score-sum count losses, Q24 role/event losses."

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
