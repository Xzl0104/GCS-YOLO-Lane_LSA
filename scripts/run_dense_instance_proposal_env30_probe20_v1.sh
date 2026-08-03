#!/usr/bin/env bash
set -euo pipefail

# Dense instance/keypoint evidence probe.
# This path preserves the env30 structured lane outputs, keeps dense evidence
# out of the default decoder, and never runs TEST.

RUN_NAME="${RUN_NAME:-dense_instance_proposal_env30_probe20_v1}"
PROJECT="${PROJECT:-runs/gcs_lane}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-dense-instance-proposal.yaml}"
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
RUN_DIAG="${RUN_DIAG:-1}"
RUN_TESTS="${RUN_TESTS:-0}"

DENSE_GAIN="${DENSE_GAIN:-1.0}"
DENSE_CENTERLINE_WEIGHT="${DENSE_CENTERLINE_WEIGHT:-1.0}"
DENSE_ENDPOINT_WEIGHT="${DENSE_ENDPOINT_WEIGHT:-1.0}"
DENSE_EMBED_PULL_WEIGHT="${DENSE_EMBED_PULL_WEIGHT:-0.25}"
DENSE_EMBED_PUSH_WEIGHT="${DENSE_EMBED_PUSH_WEIGHT:-0.25}"
DENSE_EMBED_MARGIN="${DENSE_EMBED_MARGIN:-0.5}"
DENSE_SIGMA_PX="${DENSE_SIGMA_PX:-3.0}"
DENSE_POS_WEIGHT_MAX="${DENSE_POS_WEIGHT_MAX:-50.0}"
DENSE_FREEZE_BASE="${DENSE_FREEZE_BASE:-1}"

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
  echo "RUN_TESTS must stay 0 for the dense evidence probe." >&2
  exit 2
fi
if is_true "${RUN_TRAIN}" && [[ ! -f "${PRETRAINED}" ]]; then
  echo "Missing env30 pretrained checkpoint: ${PRETRAINED}" >&2
  echo "Set PRETRAINED=/path/to/env30/official_best.pt explicitly." >&2
  exit 2
fi
if is_true "${RUN_DIAG}" && [[ ! -f "${GT_JSON}" ]]; then
  echo "Missing canonical official-val GT JSON: ${GT_JSON}" >&2
  exit 2
fi
if is_true "${RUN_DIAG}" && [[ ! -f "${TRAIN0601_GT_JSON}" ]]; then
  echo "Missing train0601 GT JSON: ${TRAIN0601_GT_JSON}" >&2
  exit 2
fi
if is_true "${RUN_DIAG}" && [[ ! -d "${ARCHIVE_ROOT}" ]]; then
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
  DENSE_FREEZE_ARGS=()
  if is_true "${DENSE_FREEZE_BASE}"; then
    DENSE_FREEZE_ARGS=(--gcs-dense-freeze-base)
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
    --gcs-dense-instance "${DENSE_GAIN}" \
    --gcs-dense-centerline-weight "${DENSE_CENTERLINE_WEIGHT}" \
    --gcs-dense-endpoint-weight "${DENSE_ENDPOINT_WEIGHT}" \
    --gcs-dense-embed-pull-weight "${DENSE_EMBED_PULL_WEIGHT}" \
    --gcs-dense-embed-push-weight "${DENSE_EMBED_PUSH_WEIGHT}" \
    --gcs-dense-embed-margin "${DENSE_EMBED_MARGIN}" \
    --gcs-dense-sigma-px "${DENSE_SIGMA_PX}" \
    --gcs-dense-pos-weight-max "${DENSE_POS_WEIGHT_MAX}" \
    "${DENSE_FREEZE_ARGS[@]}" \
    --gcs-official-best \
    --gcs-official-interval "${OFFICIAL_INTERVAL}" \
    --gcs-official-archive-root "${ARCHIVE_ROOT}" \
    --gcs-official-gt-json "${GT_JSON}" \
    --gcs-official-max-images "${OFFICIAL_MAX_IMAGES}" \
    --gcs-official-warmup "${OFFICIAL_WARMUP}" \
    --gcs-official-confs 0.001 0.003 0.005 0.01 0.02 \
    --gcs-official-point-valid-thrs 0.45 0.50 0.55 0.60 \
    --gcs-official-nms-dist-pxs 0 18 30 \
    --gcs-official-max-dets 5 6 8 \
    --gcs-official-min-points 2 3 4 5 \
    --gcs-official-count-modes score_sum \
    "${OFFICIAL_HALF_ARGS[@]}" \
    --project "${PROJECT}" \
    --name "${RUN_NAME}"
fi

WEIGHTS="${WEIGHTS:-${RUN_DIR}/weights/last.pt}"
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Missing dense probe checkpoint: ${WEIGHTS}" >&2
  exit 2
fi
echo "Dense instance probe weights: ${WEIGHTS}"

if is_true "${RUN_DIAG}"; then
  DIAG_HALF_ARGS=()
  if is_true "${HALF}"; then
    DIAG_HALF_ARGS=(--half)
  fi

  python tools/diagnose_gcs_dense_instance_oracle.py \
    --weights "${WEIGHTS}" \
    --archive-root "${ARCHIVE_ROOT}" \
    --split val \
    --gt-json "${GT_JSON}" \
    --imgsz 544 960 \
    --device "${DEVICE}" \
    --warmup 20 \
    --centerline-thr 0.5 \
    --endpoint-thr 0.5 \
    --support-radius-px 4 \
    "${DIAG_HALF_ARGS[@]}" \
    --save-dir "${PROJECT}/${RUN_NAME}_dense_diag_val_last"

  python tools/diagnose_gcs_dense_instance_oracle.py \
    --weights "${WEIGHTS}" \
    --archive-root "${ARCHIVE_ROOT}" \
    --split train \
    --gt-json "${TRAIN0601_GT_JSON}" \
    --raw-file-contains clips/0601/ \
    --allow-noncanonical-gt \
    --imgsz 544 960 \
    --device "${DEVICE}" \
    --warmup 20 \
    --centerline-thr 0.5 \
    --endpoint-thr 0.5 \
    --support-radius-px 4 \
    "${DIAG_HALF_ARGS[@]}" \
    --save-dir "${PROJECT}/${RUN_NAME}_dense_diag_train0601_last"
fi

echo "Dense instance/keypoint evidence probe complete. TEST remained closed."
