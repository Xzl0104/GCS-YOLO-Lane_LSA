#!/usr/bin/env bash
set -euo pipefail

# Q12/env30 query extent + Count Head CE ablation.
# This is the query_extent_env30_probe100_v1 protocol with only the explicit
# query Count Head and query count CE added. TEST stays closed unless RUN_TESTS=1.

export RUN_NAME="${RUN_NAME:-query_extent_count_head_ce025_env30_probe100_v1}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-extent-count.yaml}"
export EPOCHS="${EPOCHS:-100}"
export RUN_TESTS="${RUN_TESTS:-0}"
export VALID_BEFORE_MAXDET="${VALID_BEFORE_MAXDET:-1}"

export OFFICIAL_EXTENT_DECODE_MODES="${OFFICIAL_EXTENT_DECODE_MODES:-none interval intersect}"
export SWEEP_EXTENT_DECODE_MODES="${SWEEP_EXTENT_DECODE_MODES:-${OFFICIAL_EXTENT_DECODE_MODES}}"
export OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-count_logits}"
export SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-${OFFICIAL_COUNT_MODES}}"
export COUNT_AWARE_TOPK="${COUNT_AWARE_TOPK:-1}"
export SWEEP_COUNT_AWARE_TOPK="${SWEEP_COUNT_AWARE_TOPK:-${COUNT_AWARE_TOPK}}"
export COUNT_AWARE_MIN_K="${COUNT_AWARE_MIN_K:-2}"
export COUNT_AWARE_MAX_K="${COUNT_AWARE_MAX_K:-5}"
export COUNT_AWARE_LENGTH_NORM="${COUNT_AWARE_LENGTH_NORM:-12.0}"
export COUNT_AWARE_EXTRA_MARGINS="${COUNT_AWARE_EXTRA_MARGINS:-0}"
export SWEEP_COUNT_AWARE_EXTRA_MARGINS="${SWEEP_COUNT_AWARE_EXTRA_MARGINS:-${COUNT_AWARE_EXTRA_MARGINS}}"
export GCS_BOUNDARY_PSEUDO_NEG="${GCS_BOUNDARY_PSEUDO_NEG:-0.0}"

export GCS_QUERY_COUNT_CE="${GCS_QUERY_COUNT_CE:-0.25}"
export GCS_QUERY_EXTENT="${GCS_QUERY_EXTENT:-0.5}"
export GCS_QUERY_EXTENT_SHORT_VISIBLE_THR="${GCS_QUERY_EXTENT_SHORT_VISIBLE_THR:-10}"
export GCS_QUERY_EXTENT_SHORT_WEIGHT="${GCS_QUERY_EXTENT_SHORT_WEIGHT:-2.0}"
export GCS_QUERY_EXTENT_GT_MIN_LANES="${GCS_QUERY_EXTENT_GT_MIN_LANES:-4}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-ssh_lane}"
fi

python - "${MODEL}" <<'PY'
import sys

from ultralytics import YOLO

model_path = sys.argv[1]
model = YOLO(model_path).model
head = model.model[-1]
bad = []
if not bool(getattr(head, "query_count_head", False)):
    bad.append("query_count_head is not enabled")
if not bool(getattr(head, "query_extent_head", False)):
    bad.append("query_extent_head is not enabled")
if bool(getattr(head, "query_quality_head", False)):
    bad.append("query_quality_head must stay disabled")
if bad:
    raise SystemExit("; ".join(bad))
print(
    "preflight ok: query_count_head=True, query_extent_head=True, "
    f"count_classes={getattr(head, 'count_classes', None)}"
)
PY

COUNT_EXTENT_EXTRA_ARGS=(
  --gcs-query-extent "${GCS_QUERY_EXTENT}"
  --gcs-query-extent-short-visible-thr "${GCS_QUERY_EXTENT_SHORT_VISIBLE_THR}"
  --gcs-query-extent-short-weight "${GCS_QUERY_EXTENT_SHORT_WEIGHT}"
  --gcs-query-extent-gt-min-lanes "${GCS_QUERY_EXTENT_GT_MIN_LANES}"
  --gcs-query-count-ce "${GCS_QUERY_COUNT_CE}"
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
  export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} ${COUNT_EXTENT_EXTRA_ARGS[*]}"
else
  export EXTRA_TRAIN_ARGS="${COUNT_EXTENT_EXTRA_ARGS[*]}"
fi

echo "Q12 query extent + Count Head CE probe"
echo "RUN_NAME=${RUN_NAME}"
echo "MODEL=${MODEL}"
echo "EPOCHS=${EPOCHS}, RUN_TESTS=${RUN_TESTS}, VALID_BEFORE_MAXDET=${VALID_BEFORE_MAXDET}"
echo "Extent loss: gain=${GCS_QUERY_EXTENT}, short_thr=${GCS_QUERY_EXTENT_SHORT_VISIBLE_THR}, short_weight=${GCS_QUERY_EXTENT_SHORT_WEIGHT}, gt_min_lanes=${GCS_QUERY_EXTENT_GT_MIN_LANES}"
echo "Count CE: gcs_query_count_ce=${GCS_QUERY_COUNT_CE}; count-aware=${COUNT_AWARE_TOPK}; count_modes=${OFFICIAL_COUNT_MODES}; extra_margins=${COUNT_AWARE_EXTRA_MARGINS}"
echo "Disabled: Quality Head loss, score-sum count losses, boundary-pseudo loss, Q24 role/event losses."

if [[ "$(printf '%s' "${DRY_RUN:-0}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
  echo "DRY_RUN=1: preflight passed; not launching parent env30 protocol."
  exit 0
fi

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
