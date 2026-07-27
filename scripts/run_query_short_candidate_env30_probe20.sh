#!/usr/bin/env bash
set -euo pipefail

# Q12/env30 lateral candidate-generation probe.
# The env30/base parameters are frozen; only the candidate score head trains.
# Candidate decode is enabled for official-val selection. TEST remains closed.

export RUN_NAME="${RUN_NAME:-query_short_candidate_env30_probe20}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-short-candidate.yaml}"
export PRETRAINED="${PRETRAINED:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt}"
export EPOCHS="${EPOCHS:-20}"
export RUN_TESTS="${RUN_TESTS:-0}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export CANDIDATE_DECODE=1
export COUNT_AWARE_TOPK=0
export OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"
export SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-${OFFICIAL_COUNT_MODES}}"
export OFFICIAL_EXTENT_DECODE_MODES=none
export SWEEP_EXTENT_DECODE_MODES=none

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if is_true "${RUN_TESTS}"; then
  echo "RUN_TESTS must stay 0 for the first lateral candidate probe. TEST is closed." >&2
  exit 2
fi

CANDIDATE_EXTRA_ARGS=(
  --gcs-short-candidate "${GCS_SHORT_CANDIDATE:-0.05}"
  --gcs-short-candidate-visible-thr "${GCS_SHORT_CANDIDATE_VISIBLE_THR:-10}"
  --gcs-short-candidate-gt-min-lanes "${GCS_SHORT_CANDIDATE_GT_MIN_LANES:-4}"
  --gcs-short-candidate-beta-px "${GCS_SHORT_CANDIDATE_BETA_PX:-3.0}"
  --gcs-short-candidate-score-temperature "${GCS_SHORT_CANDIDATE_SCORE_TEMPERATURE:-1.0}"
  --gcs-short-candidate-count "${GCS_SHORT_CANDIDATE_COUNT:-7}"
  --gcs-short-candidate-radius-px "${GCS_SHORT_CANDIDATE_RADIUS_PX:-60.0}"
  --gcs-short-candidate-step-px "${GCS_SHORT_CANDIDATE_STEP_PX:-20.0}"
  --gcs-short-candidate-freeze-base
  --gcs-query-count-ce 0.0
  --gcs-query-quality 0.0
  --gcs-query-extent 0.0
  --gcs-count 0.0
  --gcs-count-under5 0.0
  --gcs-count-boundary 0.0
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} ${CANDIDATE_EXTRA_ARGS[*]}"
else
  export EXTRA_TRAIN_ARGS="${CANDIDATE_EXTRA_ARGS[*]}"
fi

echo "Q12/env30 lateral candidate probe: model=${MODEL}, epochs=${EPOCHS}, tests=${RUN_TESTS}"
echo "Pretrained: ${PRETRAINED}"
echo "Candidate offsets: count=${GCS_SHORT_CANDIDATE_COUNT:-7}, radius_px=${GCS_SHORT_CANDIDATE_RADIUS_PX:-60.0}, step_px=${GCS_SHORT_CANDIDATE_STEP_PX:-20.0}"
echo "Candidate loss: gain=${GCS_SHORT_CANDIDATE:-0.05}, beta_px=${GCS_SHORT_CANDIDATE_BETA_PX:-3.0}, temperature=${GCS_SHORT_CANDIDATE_SCORE_TEMPERATURE:-1.0}"
echo "Freeze: env30/base parameters frozen; only query_short_candidate head trains."
echo "Decode: candidate_decode=true, extent=false, count-aware-topk=false."
echo "TEST is closed."

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
