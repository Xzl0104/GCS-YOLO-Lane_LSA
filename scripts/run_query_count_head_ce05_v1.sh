#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the remote CUDA server.
# Override variables from the shell when needed, e.g. BATCH=16 DEVICE=1 bash scripts/run_query_count_head_ce05_v1.sh

RUN_NAME="${RUN_NAME:-query_count_head_ce05_v1}"
PROJECT="${PROJECT:-runs/gcs_lane}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count.yaml}"
DATA="${DATA:-data/tusimple_gcs_fixed_y_960x544.yaml}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-220}"
WORKERS="${WORKERS:-4}"
CONF="${CONF:-0.005}"
POINT_VALID_THR="${POINT_VALID_THR:-0.45}"
NMS_DIST_PX="${NMS_DIST_PX:-0}"
MAX_DET="${MAX_DET:-5}"
MIN_POINTS="${MIN_POINTS:-2}"
VALID_BEFORE_MAXDET="${VALID_BEFORE_MAXDET:-1}"
COUNT_AWARE_TOPK="${COUNT_AWARE_TOPK:-1}"
COUNT_MODE="${COUNT_MODE:-count_logits}"
ACC_DIR="${ACC_DIR:-${PROJECT}/${RUN_NAME}_official_val_acc_fixed_conf005_pv045_nms0_max5_min2_${COUNT_MODE}}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-ssh_lane}"
fi

if [[ ! -f "${GT_JSON}" ]]; then
  echo "Missing canonical official-val GT JSON: ${GT_JSON}" >&2
  exit 2
fi
if [[ -e "${PROJECT}/${RUN_NAME}" ]]; then
  echo "Run directory already exists: ${PROJECT}/${RUN_NAME}" >&2
  echo "Set RUN_NAME to a new value before launching this protocol." >&2
  exit 2
fi

python tools/train_gcs.py \
  --dataset tusimple \
  --model "${MODEL}" \
  --data "${DATA}" \
  --pretrained yolo11s-seg.pt \
  --imgsz 544 960 \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --scale 0.15 \
  --erasing 0.10 \
  --mosaic 0.0 \
  --gcs-query-count-ce 0.5 \
  --gcs-count 0.0 \
  --gcs-count-under5 0.0 \
  --gcs-count-boundary 0.0 \
  --gcs-official-best \
  --gcs-official-gt-json "${GT_JSON}" \
  --gcs-official-valid-before-maxdet \
  --gcs-official-half \
  --project "${PROJECT}" \
  --name "${RUN_NAME}"

WEIGHTS="${PROJECT}/${RUN_NAME}/weights/official_best.pt"

DECODE_ARGS=()
if [[ "${VALID_BEFORE_MAXDET}" == "1" || "${VALID_BEFORE_MAXDET}" == "true" ]]; then
  DECODE_ARGS+=(--valid-before-maxdet)
fi
if [[ "${COUNT_AWARE_TOPK}" == "1" || "${COUNT_AWARE_TOPK}" == "true" ]]; then
  DECODE_ARGS+=(--count-aware-topk --count-mode "${COUNT_MODE}")
fi

python tools/eval_tusimple_official.py \
  --archive-root "${ARCHIVE_ROOT}" \
  --split val \
  --gt-json "${GT_JSON}" \
  --weights "${WEIGHTS}" \
  --imgsz 544 960 \
  --device "${DEVICE}" \
  --half \
  --conf "${CONF}" \
  --point-valid-thr "${POINT_VALID_THR}" \
  --nms-dist-px "${NMS_DIST_PX}" \
  --max-det "${MAX_DET}" \
  --min-points "${MIN_POINTS}" \
  --save-dir "${ACC_DIR}" \
  --save-records \
  "${DECODE_ARGS[@]}"
