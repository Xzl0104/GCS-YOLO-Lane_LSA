#!/usr/bin/env bash
set -euo pipefail

# Q12/env30 gated lateral-candidate probe v2.
# The env30 main path stays frozen; candidates are used only for prediction-only
# short-lane queries, and the original query score is preserved.

export RUN_NAME="${RUN_NAME:-query_short_candidate_env30_gated_probe20_v2}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-short-candidate.yaml}"
export PRETRAINED="${PRETRAINED:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt}"
export EPOCHS="${EPOCHS:-20}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_TESTS="${RUN_TESTS:-0}"
export CANDIDATE_DECODE=1
export CANDIDATE_SHORT_GATE="${CANDIDATE_SHORT_GATE:-1}"
export CANDIDATE_GATE_VALID_THR="${CANDIDATE_GATE_VALID_THR:-0.5}"
export CANDIDATE_GATE_MIN_VISIBLE="${CANDIDATE_GATE_MIN_VISIBLE:-2}"
export CANDIDATE_GATE_MAX_VISIBLE="${CANDIDATE_GATE_MAX_VISIBLE:-10}"
export CANDIDATE_PRESERVE_BASE_SCORE="${CANDIDATE_PRESERVE_BASE_SCORE:-1}"
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
  echo "RUN_TESTS must stay 0 for the gated candidate probe. TEST is closed." >&2
  exit 2
fi

if ! is_true "${CANDIDATE_SHORT_GATE}"; then
  echo "CANDIDATE_SHORT_GATE must stay enabled for v2." >&2
  exit 2
fi

if ! is_true "${CANDIDATE_PRESERVE_BASE_SCORE}"; then
  echo "CANDIDATE_PRESERVE_BASE_SCORE must stay enabled for v2." >&2
  exit 2
fi

EXTRA_ARGS=(
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
  export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} ${EXTRA_ARGS[*]}"
else
  export EXTRA_TRAIN_ARGS="${EXTRA_ARGS[*]}"
fi

echo "Q12/env30 gated lateral-candidate probe v2"
echo "model=${MODEL}"
echo "pretrained=${PRETRAINED}"
echo "epochs=${EPOCHS}"
echo "candidate gate: valid_thr=${CANDIDATE_GATE_VALID_THR}, visible=[${CANDIDATE_GATE_MIN_VISIBLE},${CANDIDATE_GATE_MAX_VISIBLE}]"
echo "candidate score: preserve_base=${CANDIDATE_PRESERVE_BASE_SCORE}"
echo "Count Head/Quality Head/Extent/TEST: disabled"

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
