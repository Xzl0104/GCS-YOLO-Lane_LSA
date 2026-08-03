#!/usr/bin/env bash
set -euo pipefail

# Full-lane proposal v2 warmup.
# This run is diagnostic-only: full proposals receive proposal-only positive
# assignment while env30/base parameters stay frozen. TEST remains closed.

RUN_NAME="${RUN_NAME:-full_lane_proposal_env30_auxwarm_v2}"
PROJECT="${PROJECT:-runs/gcs_lane}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-full-lane-proposal.yaml}"
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
HALF="${HALF:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_ORACLE="${RUN_ORACLE:-1}"
RUN_FINAL_SWEEP="${RUN_FINAL_SWEEP:-0}"
RUN_TESTS="${RUN_TESTS:-0}"

FULL_LANE_GAIN="${FULL_LANE_GAIN:-1.0}"
FULL_LANE_QUALITY_TAU="${FULL_LANE_QUALITY_TAU:-25.0}"
FULL_LANE_UNMATCHED_VALID_WEIGHT="${FULL_LANE_UNMATCHED_VALID_WEIGHT:-0.0}"
FULL_LANE_UNMATCHED_WEIGHT="${FULL_LANE_UNMATCHED_WEIGHT:-0.0}"
FULL_LANE_AUX_MATCH_MIN_OVERLAP="${FULL_LANE_AUX_MATCH_MIN_OVERLAP:-2}"
FULL_LANE_AUX_MATCH_GATE_PX="${FULL_LANE_AUX_MATCH_GATE_PX:-0.0}"
FULL_LANE_HARD_FOCUS="${FULL_LANE_HARD_FOCUS:-0}"
FULL_LANE_FOCUS_HIT_PX="${FULL_LANE_FOCUS_HIT_PX:-20.0}"
FULL_LANE_FOCUS_BASE_VALID_THR="${FULL_LANE_FOCUS_BASE_VALID_THR:-0.6}"
FULL_LANE_FOCUS_BASE_MIN_COVERAGE="${FULL_LANE_FOCUS_BASE_MIN_COVERAGE:-1.0}"
FULL_LANE_BASE_HIT_WEIGHT="${FULL_LANE_BASE_HIT_WEIGHT:-1.0}"
FULL_LANE_BASE_MISS_WEIGHT="${FULL_LANE_BASE_MISS_WEIGHT:-1.0}"
FULL_LANE_GT4_WEIGHT="${FULL_LANE_GT4_WEIGHT:-1.0}"
FULL_LANE_GT5_WEIGHT="${FULL_LANE_GT5_WEIGHT:-1.0}"
FULL_LANE_SHORT_VISIBLE_THR="${FULL_LANE_SHORT_VISIBLE_THR:-10}"
FULL_LANE_SHORT_VISIBLE_WEIGHT="${FULL_LANE_SHORT_VISIBLE_WEIGHT:-1.0}"
FULL_LANE_DECODE="${FULL_LANE_DECODE:-0}"
ORACLE_WARMUP="${ORACLE_WARMUP:-20}"
ORACLE_POINT_VALID_THR="${ORACLE_POINT_VALID_THR:-0.5}"
ORACLE_HIT_PX="${ORACLE_HIT_PX:-20}"

OFFICIAL_CONFS="${OFFICIAL_CONFS:-0.001 0.003 0.005 0.008 0.01 0.02}"
OFFICIAL_POINT_VALID_THRS="${OFFICIAL_POINT_VALID_THRS:-0.45 0.50 0.55 0.60}"
OFFICIAL_NMS_DIST_PXS="${OFFICIAL_NMS_DIST_PXS:-0 18 30}"
OFFICIAL_MAX_DETS="${OFFICIAL_MAX_DETS:-5 6 8}"
OFFICIAL_MIN_POINTS="${OFFICIAL_MIN_POINTS:-2 3 4 5}"
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

if is_true "${RUN_TESTS}"; then
  echo "RUN_TESTS must stay 0 for v2 warmup. Select on official-val before any TEST run." >&2
  exit 2
fi
if is_true "${RUN_TRAIN}" && [[ ! -f "${PRETRAINED}" ]]; then
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

RUN_DIR="${PROJECT}/${RUN_NAME}"
if is_true "${RUN_TRAIN}"; then
  if [[ -e "${RUN_DIR}" ]]; then
    echo "Run directory already exists: ${RUN_DIR}" >&2
    echo "Set RUN_TRAIN=0 to reuse it, or choose a new RUN_NAME." >&2
    exit 2
  fi

  OFFICIAL_HALF_ARGS=()
  if is_true "${HALF}"; then
    OFFICIAL_HALF_ARGS=(--gcs-official-half)
  fi
  FULL_DECODE_ARGS=()
  if is_true "${FULL_LANE_DECODE}"; then
    FULL_DECODE_ARGS=(--gcs-full-lane-decode)
  fi
  HARD_FOCUS_ARGS=()
  if is_true "${FULL_LANE_HARD_FOCUS}"; then
    HARD_FOCUS_ARGS=(--gcs-full-lane-hard-focus)
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
    --scale 0.0 \
    --erasing 0.0 \
    --mosaic 0.0 \
    --gcs-full-lane-proposal "${FULL_LANE_GAIN}" \
    --gcs-full-lane-quality-tau "${FULL_LANE_QUALITY_TAU}" \
    --gcs-full-lane-unmatched-valid-weight "${FULL_LANE_UNMATCHED_VALID_WEIGHT}" \
    --gcs-full-lane-unmatched-weight "${FULL_LANE_UNMATCHED_WEIGHT}" \
    --gcs-full-lane-aux-assignment \
    --no-gcs-full-lane-unified-matching \
    --gcs-full-lane-aux-match-min-overlap "${FULL_LANE_AUX_MATCH_MIN_OVERLAP}" \
    --gcs-full-lane-aux-match-gate-px "${FULL_LANE_AUX_MATCH_GATE_PX}" \
    "${HARD_FOCUS_ARGS[@]}" \
    --gcs-full-lane-focus-hit-px "${FULL_LANE_FOCUS_HIT_PX}" \
    --gcs-full-lane-focus-base-valid-thr "${FULL_LANE_FOCUS_BASE_VALID_THR}" \
    --gcs-full-lane-focus-base-min-coverage "${FULL_LANE_FOCUS_BASE_MIN_COVERAGE}" \
    --gcs-full-lane-base-hit-weight "${FULL_LANE_BASE_HIT_WEIGHT}" \
    --gcs-full-lane-base-miss-weight "${FULL_LANE_BASE_MISS_WEIGHT}" \
    --gcs-full-lane-gt4-weight "${FULL_LANE_GT4_WEIGHT}" \
    --gcs-full-lane-gt5-weight "${FULL_LANE_GT5_WEIGHT}" \
    --gcs-full-lane-short-visible-thr "${FULL_LANE_SHORT_VISIBLE_THR}" \
    --gcs-full-lane-short-visible-weight "${FULL_LANE_SHORT_VISIBLE_WEIGHT}" \
    --gcs-full-lane-freeze-base \
    "${FULL_DECODE_ARGS[@]}" \
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
    --name "${RUN_NAME}"
fi

WEIGHTS="${WEIGHTS:-${RUN_DIR}/weights/last.pt}"
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Missing v2 warmup checkpoint: ${WEIGHTS}" >&2
  exit 2
fi
echo "Full-lane v2 diagnostic weights: ${WEIGHTS}"

if is_true "${RUN_ORACLE}"; then
  ORACLE_HALF_ARGS=()
  if is_true "${HALF}"; then
    ORACLE_HALF_ARGS=(--half)
  fi

  python tools/diagnose_gcs_full_lane_oracle.py \
    --weights "${WEIGHTS}" \
    --archive-root "${ARCHIVE_ROOT}" \
    --split val \
    --gt-json "${GT_JSON}" \
    --imgsz 544 960 \
    --device "${DEVICE}" \
    --warmup "${ORACLE_WARMUP}" \
    --point-valid-thr "${ORACLE_POINT_VALID_THR}" \
    --hit-px "${ORACLE_HIT_PX}" \
    "${ORACLE_HALF_ARGS[@]}" \
    --save-dir "${PROJECT}/${RUN_NAME}_full_lane_oracle_val_last"

  python tools/diagnose_gcs_full_lane_oracle.py \
    --weights "${WEIGHTS}" \
    --archive-root "${ARCHIVE_ROOT}" \
    --split train \
    --gt-json "${TRAIN0601_GT_JSON}" \
    --raw-file-contains clips/0601/ \
    --allow-noncanonical-gt \
    --imgsz 544 960 \
    --device "${DEVICE}" \
    --warmup "${ORACLE_WARMUP}" \
    --point-valid-thr "${ORACLE_POINT_VALID_THR}" \
    --hit-px "${ORACLE_HIT_PX}" \
    "${ORACLE_HALF_ARGS[@]}" \
    --save-dir "${PROJECT}/${RUN_NAME}_full_lane_oracle_train0601_last"
fi

if is_true "${RUN_FINAL_SWEEP}"; then
  if ! is_true "${FULL_LANE_DECODE}"; then
    echo "RUN_FINAL_SWEEP=1 requires FULL_LANE_DECODE=1 for a full-lane decode sweep." >&2
    exit 2
  fi
  SWEEP_HALF_ARGS=()
  if is_true "${HALF}"; then
    SWEEP_HALF_ARGS=(--half)
  fi
  python tools/sweep_tusimple_official_cached.py \
    --weights "${WEIGHTS}" \
    --archive-root "${ARCHIVE_ROOT}" \
    --split val \
    --gt-json "${GT_JSON}" \
    --imgsz 544 960 \
    --device "${DEVICE}" \
    --full-lane-decode \
    --rebuild-cache \
    --warmup "${OFFICIAL_WARMUP}" \
    --confs "${OFFICIAL_CONFS_ARR[@]}" \
    --point-valid-thrs "${OFFICIAL_POINT_VALID_THRS_ARR[@]}" \
    --nms-dist-pxs "${OFFICIAL_NMS_DIST_PXS_ARR[@]}" \
    --max-dets "${OFFICIAL_MAX_DETS_ARR[@]}" \
    --min-points "${OFFICIAL_MIN_POINTS_ARR[@]}" \
    --count-modes "${OFFICIAL_COUNT_MODES_ARR[@]}" \
    "${SWEEP_HALF_ARGS[@]}" \
    --save-dir "${PROJECT}/${RUN_NAME}_full_lane_official_val_sweep"
fi

echo "Full-lane proposal v2 warmup complete. TEST remained closed."
