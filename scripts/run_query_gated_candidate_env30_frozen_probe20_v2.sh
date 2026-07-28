#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the remote CUDA server.
# Frozen-env30 probe: load env30 official_best.pt, freeze every non-short-candidate
# parameter, train only the gated-candidate v2 selector, verify base decode on
# official-val, run raw candidate coverage, and keep TEST closed.

RUN_NAME="${RUN_NAME:-query_gated_candidate_env30_frozen_probe20_v2}"
PROJECT="${PROJECT:-runs/gcs_lane}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-gated-candidate-v2.yaml}"
DATA="${DATA:-data/tusimple_gcs_fixed_y_960x544.yaml}"
PRETRAINED="${PRETRAINED:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-20}"
WORKERS="${WORKERS:-4}"
OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-5}"
OFFICIAL_WARMUP="${OFFICIAL_WARMUP:-5}"
OFFICIAL_MAX_IMAGES="${OFFICIAL_MAX_IMAGES:-0}"
SWEEP_WARMUP="${SWEEP_WARMUP:-20}"
HALF="${HALF:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_BASE_SWEEP="${RUN_BASE_SWEEP:-1}"
RUN_CANDIDATE_DECODE_SWEEP="${RUN_CANDIDATE_DECODE_SWEEP:-0}"
RUN_RAW_COVERAGE="${RUN_RAW_COVERAGE:-1}"
RUN_TESTS="${RUN_TESTS:-0}"

SCALE="${SCALE:-0.0}"
ERASING="${ERASING:-0.0}"
SHORT_CANDIDATE="${SHORT_CANDIDATE:-0.05}"
SHORT_CANDIDATE_TOPK="${SHORT_CANDIDATE_TOPK:-4}"
SHORT_CANDIDATE_VISIBLE_THR="${SHORT_CANDIDATE_VISIBLE_THR:-10}"
SHORT_CANDIDATE_MIN_VISIBLE="${SHORT_CANDIDATE_MIN_VISIBLE:-2}"
SHORT_CANDIDATE_POS_PX="${SHORT_CANDIDATE_POS_PX:-20}"
SHORT_CANDIDATE_SOFT_PX="${SHORT_CANDIDATE_SOFT_PX:-40}"
SHORT_CANDIDATE_TAU="${SHORT_CANDIDATE_TAU:-25}"
SHORT_CANDIDATE_PULL_WEIGHT="${SHORT_CANDIDATE_PULL_WEIGHT:-0.0}"
SHORT_CANDIDATE_GT4_WEIGHT="${SHORT_CANDIDATE_GT4_WEIGHT:-1.0}"
SHORT_CANDIDATE_GT5_WEIGHT="${SHORT_CANDIDATE_GT5_WEIGHT:-1.5}"
SHORT_CANDIDATE_NEG_SCORE_THR="${SHORT_CANDIDATE_NEG_SCORE_THR:-0.6}"
CANDIDATE_SCORE_THR="${CANDIDATE_SCORE_THR:-0.05}"
CANDIDATE_SHORT_MIN_POINTS="${CANDIDATE_SHORT_MIN_POINTS:-2}"
CANDIDATE_SHORT_MAX_POINTS="${CANDIDATE_SHORT_MAX_POINTS:-10}"

OFFICIAL_CONFS="${OFFICIAL_CONFS:-0.005 0.01 0.02 0.05 0.1}"
OFFICIAL_POINT_VALID_THRS="${OFFICIAL_POINT_VALID_THRS:-0.45 0.50}"
OFFICIAL_NMS_DIST_PXS="${OFFICIAL_NMS_DIST_PXS:-0 18 30 50}"
OFFICIAL_MAX_DETS="${OFFICIAL_MAX_DETS:-5 6 8}"
OFFICIAL_MIN_POINTS="${OFFICIAL_MIN_POINTS:-4 5 6}"
OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"

read -r -a OFFICIAL_CONFS_ARR <<< "${OFFICIAL_CONFS}"
read -r -a OFFICIAL_POINT_VALID_THRS_ARR <<< "${OFFICIAL_POINT_VALID_THRS}"
read -r -a OFFICIAL_NMS_DIST_PXS_ARR <<< "${OFFICIAL_NMS_DIST_PXS}"
read -r -a OFFICIAL_MAX_DETS_ARR <<< "${OFFICIAL_MAX_DETS}"
read -r -a OFFICIAL_MIN_POINTS_ARR <<< "${OFFICIAL_MIN_POINTS}"
read -r -a OFFICIAL_COUNT_MODES_ARR <<< "${OFFICIAL_COUNT_MODES}"

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
if [[ ! -d "${ARCHIVE_ROOT}" ]]; then
  echo "Missing TuSimple archive root: ${ARCHIVE_ROOT}" >&2
  exit 2
fi
if is_true "${RUN_TESTS}"; then
  echo "RUN_TESTS must stay 0 for this frozen-env30 probe. Select on official-val before any TEST run." >&2
  exit 2
fi

RUN_DIR="${PROJECT}/${RUN_NAME}"
OFFICIAL_BEST_WEIGHTS="${RUN_DIR}/weights/official_best.pt"
BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"
LAST_WEIGHTS="${RUN_DIR}/weights/last.pt"
BASE_WEIGHTS_FOR_SWEEP="${BASE_WEIGHTS_FOR_SWEEP:-${OFFICIAL_BEST_WEIGHTS}}"
CANDIDATE_WEIGHTS_FOR_DIAG="${CANDIDATE_WEIGHTS_FOR_DIAG:-${LAST_WEIGHTS}}"
BASE_SWEEP_DIR="${BASE_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_official_best_val_base_sweep}"
CANDIDATE_SWEEP_DIR="${CANDIDATE_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_last_val_candidate_decode_sweep}"
COVERAGE_DIR="${COVERAGE_DIR:-${PROJECT}/${RUN_NAME}_candidate_coverage}"

HALF_ARGS=()
OFFICIAL_HALF_ARGS=()
if is_true "${HALF}"; then
  HALF_ARGS=(--half)
  OFFICIAL_HALF_ARGS=(--gcs-official-half)
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
    --gcs-short-candidate "${SHORT_CANDIDATE}" \
    --gcs-short-candidate-topk "${SHORT_CANDIDATE_TOPK}" \
    --gcs-short-candidate-visible-thr "${SHORT_CANDIDATE_VISIBLE_THR}" \
    --gcs-short-candidate-min-visible "${SHORT_CANDIDATE_MIN_VISIBLE}" \
    --gcs-short-candidate-pos-px "${SHORT_CANDIDATE_POS_PX}" \
    --gcs-short-candidate-soft-px "${SHORT_CANDIDATE_SOFT_PX}" \
    --gcs-short-candidate-tau "${SHORT_CANDIDATE_TAU}" \
    --gcs-short-candidate-pull-weight "${SHORT_CANDIDATE_PULL_WEIGHT}" \
    --gcs-short-candidate-gt4-weight "${SHORT_CANDIDATE_GT4_WEIGHT}" \
    --gcs-short-candidate-gt5-weight "${SHORT_CANDIDATE_GT5_WEIGHT}" \
    --gcs-short-candidate-neg-score-thr "${SHORT_CANDIDATE_NEG_SCORE_THR}" \
    --gcs-short-candidate-freeze-base \
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
  else
    echo "Missing base-decode weights for sweep: ${BASE_WEIGHTS_FOR_SWEEP}" >&2
    exit 2
  fi
fi
if [[ ! -f "${CANDIDATE_WEIGHTS_FOR_DIAG}" ]]; then
  if [[ -f "${OFFICIAL_BEST_WEIGHTS}" ]]; then
    CANDIDATE_WEIGHTS_FOR_DIAG="${OFFICIAL_BEST_WEIGHTS}"
  elif [[ -f "${BEST_WEIGHTS}" ]]; then
    CANDIDATE_WEIGHTS_FOR_DIAG="${BEST_WEIGHTS}"
  else
    echo "Missing candidate weights for diagnostics: ${CANDIDATE_WEIGHTS_FOR_DIAG}" >&2
    exit 2
  fi
fi

echo "Base-decode sweep weights: ${BASE_WEIGHTS_FOR_SWEEP}"
echo "Candidate diagnostics weights: ${CANDIDATE_WEIGHTS_FOR_DIAG}"

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

if is_true "${RUN_CANDIDATE_DECODE_SWEEP}"; then
  python tools/sweep_tusimple_official.py \
    --weights "${CANDIDATE_WEIGHTS_FOR_DIAG}" \
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
    --candidate-decode \
    --candidate-score-thr "${CANDIDATE_SCORE_THR}" \
    --candidate-short-min-points "${CANDIDATE_SHORT_MIN_POINTS}" \
    --candidate-short-max-points "${CANDIDATE_SHORT_MAX_POINTS}" \
    --warmup "${SWEEP_WARMUP}" \
    --device "${DEVICE}" \
    "${HALF_ARGS[@]}" \
    --save-dir "${CANDIDATE_SWEEP_DIR}"
fi

if is_true "${RUN_RAW_COVERAGE}"; then
  mkdir -p "${COVERAGE_DIR}"
  python tools/diagnose_gcs_short_candidate_coverage.py \
    --weights "${CANDIDATE_WEIGHTS_FOR_DIAG}" \
    --data "${DATA}" \
    --split train \
    --imgsz 544 960 \
    --batch "${BATCH}" \
    --workers "${WORKERS}" \
    --device "${DEVICE}" \
    "${HALF_ARGS[@]}" \
    --visible-thr "${SHORT_CANDIDATE_VISIBLE_THR}" \
    --save-json "${COVERAGE_DIR}/train_short_candidate_coverage.json"
  python tools/diagnose_gcs_short_candidate_coverage.py \
    --weights "${CANDIDATE_WEIGHTS_FOR_DIAG}" \
    --data "${DATA}" \
    --split val \
    --imgsz 544 960 \
    --batch "${BATCH}" \
    --workers "${WORKERS}" \
    --device "${DEVICE}" \
    "${HALF_ARGS[@]}" \
    --visible-thr "${SHORT_CANDIDATE_VISIBLE_THR}" \
    --save-json "${COVERAGE_DIR}/val_short_candidate_coverage.json"
fi

echo "Frozen-env30 probe complete. TEST remained closed."
