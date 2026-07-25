#!/usr/bin/env bash
set -euo pipefail

# Q24 protected-static + explicit Count Head + Quality Head isolation probe.
# The only intended change versus the previous Q24 dual-head probe is disabling
# boundary-pseudo negative pressure. Selection stays official-val-only; TEST is off.

export RUN_NAME="${RUN_NAME:-query_alpha05_env30_q24_dualhead_quality_count_bneg0_probe100_v1}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q24-k56-dualhead.yaml}"
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
export GCS_BOUNDARY_PSEUDO_NEG="${GCS_BOUNDARY_PSEUDO_NEG:-0.0}"

# Keep optimizer and loss gains fixed to isolate the boundary-pseudo variable.
export Q24_OPTIMIZER="${Q24_OPTIMIZER:-AdamW}"
export Q24_LR0="${Q24_LR0:-0.0005}"
export Q24_LRF="${Q24_LRF:-0.05}"
export Q24_WEIGHT_DECAY="${Q24_WEIGHT_DECAY:-0.0001}"
export Q24_WARMUP_EPOCHS="${Q24_WARMUP_EPOCHS:-0.0}"

Q24_EXTRA_ARGS=(
  --optimizer "${Q24_OPTIMIZER}"
  --lr0 "${Q24_LR0}"
  --lrf "${Q24_LRF}"
  --weight-decay "${Q24_WEIGHT_DECAY}"
  --warmup-epochs "${Q24_WARMUP_EPOCHS}"
  --gcs-count 0.0
  --gcs-count-under5 0.0
  --gcs-count-boundary 0.0
  --gcs-query-count-ce "${GCS_QUERY_COUNT_CE:-0.25}"
  --gcs-query-quality "${GCS_QUERY_QUALITY:-0.25}"
  --gcs-role-contain 0.0
  --gcs-q24-event-contain 0.0
  --gcs-q24-event-score-calib 0.0
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} ${Q24_EXTRA_ARGS[*]}"
else
  export EXTRA_TRAIN_ARGS="${Q24_EXTRA_ARGS[*]}"
fi

echo "Q24 dual-head bneg0 isolation probe: model=${MODEL}, epochs=${EPOCHS}, tests=${RUN_TESTS}"
echo "Boundary pseudo neg: ${GCS_BOUNDARY_PSEUDO_NEG}"
echo "Count/Quality gains: gcs_query_count_ce=${GCS_QUERY_COUNT_CE:-0.25}, gcs_query_quality=${GCS_QUERY_QUALITY:-0.25}"
echo "Optimizer: ${Q24_OPTIMIZER}, lr0=${Q24_LR0}, lrf=${Q24_LRF}, warmup_epochs=${Q24_WARMUP_EPOCHS}"
echo "Disabled: score-sum count losses, boundary-pseudo neg, role containment, Q24 event containment/calibration."

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
