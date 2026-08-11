#!/usr/bin/env bash
set -euo pipefail

# Default-off lane-instance-set probe. TuSimple TEST stays closed.
RUN_NAME=${RUN_NAME:-lane_instance_set_env30_probe20_seed0_v1}
PROJECT=${PROJECT:-runs/gcs_lane}
MODEL=${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-lane-instance-set-decoder.yaml}
DATA=${DATA:-data/tusimple_gcs_fixed_y_960x544.yaml}
PRETRAINED=${PRETRAINED:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-archive/TUSimple}
GT_JSON=${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-32}
WORKERS=${WORKERS:-4}
DEVICE=${DEVICE:-0}
SEED=${SEED:-0}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}
RUN_TESTS=${RUN_TESTS:-0}

is_true() { [[ ${1,,} =~ ^(1|true|yes|on)$ ]]; }

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate ${CONDA_ENV:-ssh_lane}
fi

if is_true "${RUN_TESTS}"; then
  echo "RUN_TESTS must stay 0."
  exit 2
fi
if [[ "${MODEL}" != *lane-instance-set-decoder* ]]; then
  echo "Invalid MODEL: ${MODEL}"
  exit 2
fi
for path in "${MODEL}" "${DATA}" "${PRETRAINED}" "${GT_JSON}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing file: ${path}"
    exit 2
  fi
done
if [[ ! -d "${ARCHIVE_ROOT}" ]]; then
  echo "Missing archive: ${ARCHIVE_ROOT}"
  exit 2
fi

RUN_DIR="${PROJECT}/${RUN_NAME}"
if is_true "${RUN_TRAIN}"; then
  if [[ -e "${RUN_DIR}" ]]; then
    echo "Run exists: ${RUN_DIR}"
    exit 2
  fi
  RUN_TESTS=0 python -c 'from ultralytics.cfg import entrypoint; entrypoint()' task=gcs_lane mode=train \
    model="${MODEL}" data="${DATA}" pretrained="${PRETRAINED}" \
    imgsz='[544,960]' gcs_imgsz='[544,960]' \
    epochs="${EPOCHS}" batch="${BATCH}" workers="${WORKERS}" device="${DEVICE}" seed="${SEED}" \
    amp=False scale=0.0 erasing=0.0 mosaic=0.0 project="${PROJECT}" name="${RUN_NAME}" val=True exist_ok=False \
    gcs_lane_instance_set=1.0 gcs_lane_instance_set_noop_weight=0.5 gcs_lane_instance_set_margin=0.5 \
    gcs_official_best=True gcs_official_interval="${OFFICIAL_INTERVAL:-5}" gcs_official_warmup="${OFFICIAL_WARMUP:-5}" \
    gcs_official_archive_root="${ARCHIVE_ROOT}" gcs_official_gt_json="${GT_JSON}" \
    gcs_official_confs='[0.03,0.05,0.10,0.15,0.25]' \
    gcs_official_point_valid_thrs='[0.35,0.40,0.45,0.50,0.60]' \
    gcs_official_min_points='[2,3,4]' gcs_official_half=True
fi

if ! is_true "${RUN_EVAL}"; then
  exit 0
fi

WEIGHTS="${WEIGHTS:-${RUN_DIR}/weights/official_best.pt}"
if [[ ! -f "${WEIGHTS}" ]]; then
  WEIGHTS="${RUN_DIR}/weights/last.pt"
fi
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Missing checkpoint: ${WEIGHTS}"
  exit 2
fi

SWEEP_DIR="${SWEEP_DIR:-${PROJECT}/${RUN_NAME}_canonical_official_sweep}"
RUN_TESTS=0 python tools/sweep_tusimple_official_cached.py \
  --weights "${WEIGHTS}" --archive-root "${ARCHIVE_ROOT}" --split val --gt-json "${GT_JSON}" \
  --imgsz 544 960 --decode-mode lane_instance_set --half \
  --confs 0.03 0.05 0.10 0.15 0.25 \
  --point-valid-thrs 0.35 0.40 0.45 0.50 0.60 --min-points 2 3 4 \
  --lane-instance-max-dets 5 --lane-instance-duplicate-thrs 0.50 0.65 0.80 \
  --lane-instance-min-survivors 2 --rebuild-cache --save-dir "${SWEEP_DIR}"

echo "Completed ${RUN_NAME}; TEST remained closed."
