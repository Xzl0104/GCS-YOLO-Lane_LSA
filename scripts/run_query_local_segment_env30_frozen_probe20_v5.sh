#!/usr/bin/env bash
set -euo pipefail

# Frozen-env30 local short-segment v5 probe: keep the v4 raw segment proposal,
# train listwise selector + base-preserve replace gate, run hard official-GT
# diagnostics on available checkpoints, select diagnostic segment_best.pt, and
# keep TEST closed.

RUN_NAME="${RUN_NAME:-query_local_segment_env30_frozen_probe20_v5}"
PROJECT="${PROJECT:-runs/gcs_lane}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v5.yaml}"
DATA="${DATA:-data/tusimple_gcs_fixed_y_960x544.yaml}"
PRETRAINED="${PRETRAINED:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
TRAIN0601_GT_JSON="${TRAIN0601_GT_JSON:-archive/TUSimple/train_set/label_data_0601.json}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-20}"
WORKERS="${WORKERS:-4}"
OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-5}"
OFFICIAL_WARMUP="${OFFICIAL_WARMUP:-5}"
OFFICIAL_MAX_IMAGES="${OFFICIAL_MAX_IMAGES:-0}"
SWEEP_WARMUP="${SWEEP_WARMUP:-20}"
HARD_WARMUP="${HARD_WARMUP:-20}"
HALF="${HALF:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_BASE_SWEEP="${RUN_BASE_SWEEP:-1}"
RUN_HARD_DIAG="${RUN_HARD_DIAG:-1}"
RUN_TESTS="${RUN_TESTS:-0}"

SCALE="${SCALE:-0.0}"
ERASING="${ERASING:-0.0}"
SHORT_SEGMENT="${SHORT_SEGMENT:-0.05}"
SHORT_SEGMENT_TOPK="${SHORT_SEGMENT_TOPK:-8}"
SHORT_SEGMENT_VISIBLE_THR="${SHORT_SEGMENT_VISIBLE_THR:-10}"
SHORT_SEGMENT_MIN_VISIBLE="${SHORT_SEGMENT_MIN_VISIBLE:-3}"
SHORT_SEGMENT_MIN_OVERLAP="${SHORT_SEGMENT_MIN_OVERLAP:-3}"
SHORT_SEGMENT_POS_PX="${SHORT_SEGMENT_POS_PX:-20}"
SHORT_SEGMENT_SOFT_PX="${SHORT_SEGMENT_SOFT_PX:-40}"
SHORT_SEGMENT_TAU="${SHORT_SEGMENT_TAU:-25}"
SHORT_SEGMENT_POINT_WEIGHT="${SHORT_SEGMENT_POINT_WEIGHT:-0.05}"
SHORT_SEGMENT_GT4_WEIGHT="${SHORT_SEGMENT_GT4_WEIGHT:-1.0}"
SHORT_SEGMENT_GT5_WEIGHT="${SHORT_SEGMENT_GT5_WEIGHT:-1.5}"
SHORT_SEGMENT_NEG_SCORE_THR="${SHORT_SEGMENT_NEG_SCORE_THR:-0.6}"
SHORT_SEGMENT_BCE_WEIGHT="${SHORT_SEGMENT_BCE_WEIGHT:-0.0}"
SHORT_SEGMENT_LISTWISE_WEIGHT="${SHORT_SEGMENT_LISTWISE_WEIGHT:-1.0}"
SHORT_SEGMENT_REPLACE_WEIGHT="${SHORT_SEGMENT_REPLACE_WEIGHT:-1.0}"
SHORT_SEGMENT_REPLACE_MARGIN_PX="${SHORT_SEGMENT_REPLACE_MARGIN_PX:-5.0}"
SHORT_SEGMENT_BASE_PRESERVE="${SHORT_SEGMENT_BASE_PRESERVE:-1}"

OFFICIAL_CONFS="${OFFICIAL_CONFS:-0.005 0.01 0.02 0.05 0.1}"
OFFICIAL_POINT_VALID_THRS="${OFFICIAL_POINT_VALID_THRS:-0.45 0.50}"
OFFICIAL_NMS_DIST_PXS="${OFFICIAL_NMS_DIST_PXS:-0 18 30 50}"
OFFICIAL_MAX_DETS="${OFFICIAL_MAX_DETS:-5 6 8}"
OFFICIAL_MIN_POINTS="${OFFICIAL_MIN_POINTS:-4 5 6}"
OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"
HARD_WEIGHT_NAMES="${HARD_WEIGHT_NAMES:-last best official_best}"

read -r -a OFFICIAL_CONFS_ARR <<< "${OFFICIAL_CONFS}"
read -r -a OFFICIAL_POINT_VALID_THRS_ARR <<< "${OFFICIAL_POINT_VALID_THRS}"
read -r -a OFFICIAL_NMS_DIST_PXS_ARR <<< "${OFFICIAL_NMS_DIST_PXS}"
read -r -a OFFICIAL_MAX_DETS_ARR <<< "${OFFICIAL_MAX_DETS}"
read -r -a OFFICIAL_MIN_POINTS_ARR <<< "${OFFICIAL_MIN_POINTS}"
read -r -a OFFICIAL_COUNT_MODES_ARR <<< "${OFFICIAL_COUNT_MODES}"
read -r -a HARD_WEIGHT_NAMES_ARR <<< "${HARD_WEIGHT_NAMES}"

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-ssh_lane}"
fi

if [[ ! -f "${PRETRAINED}" ]]; then
  echo "Missing env30 pretrained checkpoint: ${PRETRAINED}" >&2
  exit 2
fi
if [[ ! -f "${GT_JSON}" ]]; then
  echo "Missing canonical official-val GT JSON: ${GT_JSON}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN0601_GT_JSON}" ]]; then
  echo "Missing train0601 GT JSON: ${TRAIN0601_GT_JSON}" >&2
  exit 2
fi
if [[ ! -d "${ARCHIVE_ROOT}" ]]; then
  echo "Missing TuSimple archive root: ${ARCHIVE_ROOT}" >&2
  exit 2
fi
if is_true "${RUN_TESTS}"; then
  echo "RUN_TESTS must stay 0 for this local-segment v5 probe. Select on official-val before any TEST run." >&2
  exit 2
fi

RUN_DIR="${PROJECT}/${RUN_NAME}"
OFFICIAL_BEST_WEIGHTS="${RUN_DIR}/weights/official_best.pt"
BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"
LAST_WEIGHTS="${RUN_DIR}/weights/last.pt"
BASE_WEIGHTS_FOR_SWEEP="${BASE_WEIGHTS_FOR_SWEEP:-${OFFICIAL_BEST_WEIGHTS}}"
BASE_SWEEP_DIR="${BASE_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_official_best_val_base_sweep}"
HARD_DIAG_ROOT="${HARD_DIAG_ROOT:-${PROJECT}/${RUN_NAME}_hard_segment_diag}"
SEGMENT_BEST_JSON="${SEGMENT_BEST_JSON:-${RUN_DIR}/weights/segment_best_hard_gate.json}"
SEGMENT_BEST_WEIGHTS="${SEGMENT_BEST_WEIGHTS:-${RUN_DIR}/weights/segment_best.pt}"

HALF_ARGS=()
OFFICIAL_HALF_ARGS=()
if is_true "${HALF}"; then
  HALF_ARGS=(--half)
  OFFICIAL_HALF_ARGS=(--gcs-official-half)
fi

BASE_PRESERVE_ARGS=(--gcs-short-segment-base-preserve)
if ! is_true "${SHORT_SEGMENT_BASE_PRESERVE}"; then
  BASE_PRESERVE_ARGS=(--no-gcs-short-segment-base-preserve)
fi

if is_true "${RUN_TRAIN}"; then
  if [[ -e "${RUN_DIR}" ]]; then
    echo "Run directory already exists: ${RUN_DIR}" >&2
    echo "Set RUN_TRAIN=0 to reuse it, or choose a new RUN_NAME." >&2
    exit 2
  fi

  python tools/train_gcs.py \
    --dataset tusimple \
    --model "${MODEL}" \
    --data "${DATA}" \
    --pretrained "${PRETRAINED}" \
    --imgsz 544 960 \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --workers "${WORKERS}" \
    --device "${DEVICE}" \
    --scale "${SCALE}" \
    --erasing "${ERASING}" \
    --mosaic 0.0 \
    --gcs-exist 0.0 \
    --gcs-point 0.0 \
    --gcs-point-valid 0.0 \
    --gcs-smooth 0.0 \
    --gcs-curve 0.0 \
    --gcs-mask 0.0 \
    --gcs-edge 0.0 \
    --gcs-count 0.0 \
    --gcs-count-under5 0.0 \
    --gcs-count-boundary 0.0 \
    --gcs-query-count-ce 0.0 \
    --gcs-short-candidate 0.0 \
    --gcs-short-segment "${SHORT_SEGMENT}" \
    --gcs-short-segment-topk "${SHORT_SEGMENT_TOPK}" \
    --gcs-short-segment-visible-thr "${SHORT_SEGMENT_VISIBLE_THR}" \
    --gcs-short-segment-min-visible "${SHORT_SEGMENT_MIN_VISIBLE}" \
    --gcs-short-segment-min-overlap "${SHORT_SEGMENT_MIN_OVERLAP}" \
    --gcs-short-segment-pos-px "${SHORT_SEGMENT_POS_PX}" \
    --gcs-short-segment-soft-px "${SHORT_SEGMENT_SOFT_PX}" \
    --gcs-short-segment-tau "${SHORT_SEGMENT_TAU}" \
    --gcs-short-segment-point-weight "${SHORT_SEGMENT_POINT_WEIGHT}" \
    --gcs-short-segment-gt4-weight "${SHORT_SEGMENT_GT4_WEIGHT}" \
    --gcs-short-segment-gt5-weight "${SHORT_SEGMENT_GT5_WEIGHT}" \
    --gcs-short-segment-neg-score-thr "${SHORT_SEGMENT_NEG_SCORE_THR}" \
    --gcs-short-segment-bce-weight "${SHORT_SEGMENT_BCE_WEIGHT}" \
    --gcs-short-segment-listwise-weight "${SHORT_SEGMENT_LISTWISE_WEIGHT}" \
    --gcs-short-segment-replace-weight "${SHORT_SEGMENT_REPLACE_WEIGHT}" \
    --gcs-short-segment-replace-margin-px "${SHORT_SEGMENT_REPLACE_MARGIN_PX}" \
    "${BASE_PRESERVE_ARGS[@]}" \
    --gcs-short-segment-freeze-base \
    --gcs-official-best \
    --gcs-official-interval "${OFFICIAL_INTERVAL}" \
    --gcs-official-archive-root "${ARCHIVE_ROOT}" \
    --gcs-official-gt-json "${GT_JSON}" \
    --gcs-official-max-images "${OFFICIAL_MAX_IMAGES}" \
    --gcs-official-warmup "${OFFICIAL_WARMUP}" \
    --gcs-official-confs "${OFFICIAL_CONFS_ARR[@]}" \
    --gcs-official-point-valid-thrs "${OFFICIAL_POINT_VALID_THRS_ARR[@]}" \
    --gcs-official-nms-dist-pxs "${OFFICIAL_NMS_DIST_PXS_ARR[@]}" \
    --gcs-official-max-dets "${OFFICIAL_MAX_DETS_ARR[@]}" \
    --gcs-official-min-points "${OFFICIAL_MIN_POINTS_ARR[@]}" \
    --gcs-official-count-modes "${OFFICIAL_COUNT_MODES_ARR[@]}" \
    "${OFFICIAL_HALF_ARGS[@]}" \
    --project "${PROJECT}" \
    --name "${RUN_NAME}" \
    --exist-ok
fi

if [[ ! -f "${BASE_WEIGHTS_FOR_SWEEP}" ]]; then
  if [[ -f "${BEST_WEIGHTS}" ]]; then
    BASE_WEIGHTS_FOR_SWEEP="${BEST_WEIGHTS}"
  elif [[ -f "${LAST_WEIGHTS}" ]]; then
    BASE_WEIGHTS_FOR_SWEEP="${LAST_WEIGHTS}"
  else
    echo "Missing base-decode weights for sweep: ${BASE_WEIGHTS_FOR_SWEEP}" >&2
    exit 2
  fi
fi

echo "Base-decode sweep weights: ${BASE_WEIGHTS_FOR_SWEEP}"

if is_true "${RUN_BASE_SWEEP}"; then
  python tools/sweep_tusimple_official.py \
    --weights "${BASE_WEIGHTS_FOR_SWEEP}" \
    --archive-root "${ARCHIVE_ROOT}" \
    --split val \
    --gt-json "${GT_JSON}" \
    --imgsz 544 960 \
    --confs "${OFFICIAL_CONFS_ARR[@]}" \
    --point-valid-thrs "${OFFICIAL_POINT_VALID_THRS_ARR[@]}" \
    --nms-dist-pxs "${OFFICIAL_NMS_DIST_PXS_ARR[@]}" \
    --max-dets "${OFFICIAL_MAX_DETS_ARR[@]}" \
    --min-points "${OFFICIAL_MIN_POINTS_ARR[@]}" \
    --count-modes "${OFFICIAL_COUNT_MODES_ARR[@]}" \
    --warmup "${SWEEP_WARMUP}" \
    --device "${DEVICE}" \
    "${HALF_ARGS[@]}" \
    --save-dir "${BASE_SWEEP_DIR}"
fi

if is_true "${RUN_HARD_DIAG}"; then
  SELECT_ARGS=()
  for WEIGHT_NAME in "${HARD_WEIGHT_NAMES_ARR[@]}"; do
    WEIGHTS="${RUN_DIR}/weights/${WEIGHT_NAME}.pt"
    if [[ ! -f "${WEIGHTS}" ]]; then
      echo "Skip hard segment diagnostic for missing weights: ${WEIGHTS}" >&2
      continue
    fi
    VAL_DIR="${HARD_DIAG_ROOT}_val_${WEIGHT_NAME}"
    TRAIN0601_DIR="${HARD_DIAG_ROOT}_train0601_${WEIGHT_NAME}"
    python tools/diagnose_gcs_short_candidate_hard_coverage.py \
      --weights "${WEIGHTS}" \
      --archive-root "${ARCHIVE_ROOT}" \
      --split val \
      --gt-json "${GT_JSON}" \
      --imgsz 544 960 \
      --device "${DEVICE}" \
      --warmup "${HARD_WARMUP}" \
      "${HALF_ARGS[@]}" \
      --save-dir "${VAL_DIR}"

    python tools/diagnose_gcs_short_candidate_hard_coverage.py \
      --weights "${WEIGHTS}" \
      --archive-root "${ARCHIVE_ROOT}" \
      --split train \
      --gt-json "${TRAIN0601_GT_JSON}" \
      --raw-file-contains clips/0601/ \
      --imgsz 544 960 \
      --device "${DEVICE}" \
      --warmup "${HARD_WARMUP}" \
      "${HALF_ARGS[@]}" \
      --allow-noncanonical-gt \
      --save-dir "${TRAIN0601_DIR}"
    SELECT_ARGS+=(--candidate "${WEIGHT_NAME}" "${WEIGHTS}" "${VAL_DIR}/short_candidate_hard_summary.json" "${TRAIN0601_DIR}/short_candidate_hard_summary.json")
  done

  if [[ "${#SELECT_ARGS[@]}" -gt 0 ]]; then
    python tools/select_gcs_short_segment_hard_best.py \
      "${SELECT_ARGS[@]}" \
      --output-json "${SEGMENT_BEST_JSON}" \
      --copy-to "${SEGMENT_BEST_WEIGHTS}"
  fi
fi

echo "Local short-segment v5 probe complete. TEST remained closed."
