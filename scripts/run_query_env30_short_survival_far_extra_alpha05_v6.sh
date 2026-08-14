#!/usr/bin/env bash
set -euo pipefail

# Env30 follow-up: preserve the env30 existence-quality alpha while adding the
# default-off short-survival and selective far-extra guard ablations.
#
# Required:
#   PRETRAINED=/path/to/env30/weights/official_best.pt bash scripts/run_query_env30_short_survival_far_extra_alpha05_v6.sh

RUN_NAME="${RUN_NAME:-query_env30_short_survival_far_extra005_d80_alpha05_env30recipe_b32_amp_v6}"
PROJECT="${PROJECT:-runs/gcs_lane}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml}"
DATA="${DATA:-data/tusimple_gcs_fixed_y_960x544.yaml}"
PRETRAINED="${PRETRAINED:-}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-80}"
WORKERS="${WORKERS:-4}"
OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-5}"
OFFICIAL_WARMUP="${OFFICIAL_WARMUP:-5}"
OFFICIAL_MAX_IMAGES="${OFFICIAL_MAX_IMAGES:-0}"
HALF="${HALF:-1}"

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

if [[ -z "${PRETRAINED}" ]]; then
  echo "Set PRETRAINED to the env30 weights/official_best.pt path." >&2
  exit 2
fi
if [[ ! -f "${PRETRAINED}" ]]; then
  echo "Missing PRETRAINED checkpoint: ${PRETRAINED}" >&2
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

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-ssh_lane}"
fi

OFFICIAL_HALF_ARGS=()
if is_true "${HALF}"; then
  OFFICIAL_HALF_ARGS=(--gcs-official-half)
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
  --scale 0.15 \
  --erasing 0.10 \
  --mosaic 0.0 \
  --gcs-exist 2.0 \
  --gcs-point 15.0 \
  --gcs-point-valid 1.0 \
  --gcs-smooth 0.05 \
  --gcs-curve 0.1 \
  --gcs-mask 0.2 \
  --gcs-edge 0.2 \
  --gcs-count 0.0 \
  --gcs-count-under5 0.0 \
  --gcs-count-boundary 0.0 \
  --gcs-query-count-ce 0.0 \
  --gcs-exist-quality-alpha 0.5 \
  --gcs-short-geom 1.0 \
  --gcs-short-geom-visible-thr 10 \
  --gcs-short-geom-gt4-weight 1.0 \
  --gcs-short-geom-gt5-weight 2.0 \
  --gcs-short-geom-max-weight 3.0 \
  --gcs-short-geom-curve 1.0 \
  --gcs-gt5-short-visible-thr 0 \
  --gcs-short-survival 0.2 \
  --gcs-short-survival-visible-thr 10 \
  --gcs-short-survival-gt4-weight 1.0 \
  --gcs-short-survival-gt5-weight 1.0 \
  --gcs-short-survival-exist-weight 0.2 \
  --gcs-short-survival-valid-weight 1.0 \
  --gcs-boundary-pseudo-neg 0.02 \
  --gcs-boundary-pseudo-visible-thr 10 \
  --gcs-boundary-pseudo-dist-thr 80 \
  --gcs-boundary-pseudo-valid-thr 0.5 \
  --gcs-boundary-pseudo-min-valid 4 \
  --gcs-boundary-pseudo-gt-count 5 \
  --gcs-boundary-pseudo-score-thr 0.2 \
  --gcs-boundary-pseudo-envelope-margin-px 30 \
  --gcs-boundary-pseudo-envelope-ratio-thr 0.75 \
  --gcs-far-extra-neg 0.05 \
  --gcs-far-extra-min-gt-lanes 3 \
  --gcs-far-extra-max-gt-lanes 5 \
  --gcs-far-extra-min-valid 4 \
  --gcs-far-extra-max-valid 24 \
  --gcs-far-extra-valid-thr 0.5 \
  --gcs-far-extra-score-thr 0.02 \
  --gcs-far-extra-dist-thr 80 \
  --gcs-far-extra-gt3-weight 1.0 \
  --gcs-far-extra-gt4-weight 1.0 \
  --gcs-far-extra-gt5-weight 0.5 \
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
  --gcs-official-valid-before-maxdet \
  "${OFFICIAL_HALF_ARGS[@]}" \
  --project "${PROJECT}" \
  --name "${RUN_NAME}"
