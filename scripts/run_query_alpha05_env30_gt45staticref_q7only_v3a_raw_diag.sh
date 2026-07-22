#!/usr/bin/env bash
set -euo pipefail

# No-training diagnostic for the env30 q7-only GT4/GT5 static-reference
# counterfactual. It loads env30 official_best weights into the q7-only YAML
# and runs raw-Q12 diagnostics on official-val plus train0601/train0531 only.
# TEST remains closed.

RUN_NAME="${RUN_NAME:-query_alpha05_env30_gt45staticref_q7only_v3a}"
MODEL_CFG="${MODEL_CFG:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-env30-gt45staticref-q7only-v3a.yaml}"
WEIGHTS="${WEIGHTS:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt}"
DECODE_YAML="${DECODE_YAML:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best_decode.yaml}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
VAL_GT_JSON="${VAL_GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-1}"
MAX_IMAGES="${MAX_IMAGES:-0}"
WARMUP="${WARMUP:-20}"

COMMON_ARGS=(
  tools/diagnose_tusimple_raw_q12_filters.py
  --archive-root "${ARCHIVE_ROOT}"
  --weights "${WEIGHTS}"
  --model-cfg "${MODEL_CFG}"
  --decode-yaml "${DECODE_YAML}"
  --imgsz 544 960
  --device "${DEVICE}"
  --pool-max-det 12
  --max-images "${MAX_IMAGES}"
  --warmup "${WARMUP}"
)

if [[ "${HALF}" == "1" || "${HALF}" == "true" || "${HALF}" == "TRUE" ]]; then
  COMMON_ARGS+=(--half)
fi

python "${COMMON_ARGS[@]}" \
  --split val \
  --gt-json "${VAL_GT_JSON}" \
  --save-dir "runs/gcs_lane/${RUN_NAME}_official_best_val_raw_q12_counterfactual"

python "${COMMON_ARGS[@]}" \
  --split train \
  --gt-json "${ARCHIVE_ROOT}/train_set/label_data_0601.json" \
  --save-dir "runs/gcs_lane/${RUN_NAME}_official_best_train0601_raw_q12_counterfactual"

python "${COMMON_ARGS[@]}" \
  --split train \
  --gt-json "${ARCHIVE_ROOT}/train_set/label_data_0531.json" \
  --save-dir "runs/gcs_lane/${RUN_NAME}_official_best_train0531_raw_q12_counterfactual"

echo "q7-only raw-Q12 diagnostic complete: runs/gcs_lane/${RUN_NAME}_official_best_*_raw_q12_counterfactual"
