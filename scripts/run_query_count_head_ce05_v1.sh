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
SWEEP_DIR="${SWEEP_DIR:-${PROJECT}/${RUN_NAME}_official_val_sweep_count_modes}"
ACC_DIR="${ACC_DIR:-${PROJECT}/${RUN_NAME}_official_val_acc_from_sweep_best}"

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

python tools/sweep_tusimple_official.py \
  --archive-root "${ARCHIVE_ROOT}" \
  --split val \
  --gt-json "${GT_JSON}" \
  --weights "${WEIGHTS}" \
  --imgsz 544 960 \
  --device "${DEVICE}" \
  --half \
  --confs 0.003 0.005 0.008 0.01 0.02 \
  --point-valid-thrs 0.45 0.5 0.55 0.6 \
  --nms-dist-pxs 0 18 30 \
  --max-dets 5 6 \
  --min-points 2 3 4 \
  --valid-before-maxdet \
  --count-aware-topk \
  --count-modes score_sum count_logits \
  --save-dir "${SWEEP_DIR}"

SWEEP_SUMMARY="${SWEEP_DIR}/tusimple_official_sweep_summary.json"
BEST_ARGS="$(
  python - "${SWEEP_SUMMARY}" <<'PY'
import json
import shlex
import sys

summary_path = sys.argv[1]
with open(summary_path, "r", encoding="utf-8") as f:
    best = json.load(f)["best"]

args = [
    "--conf", str(best["conf"]),
    "--point-valid-thr", str(best["point_valid_thr"]),
    "--nms-dist-px", str(best["nms_dist_px"]),
    "--max-det", str(best["max_det"]),
    "--min-points", str(best["min_points"]),
    "--count-aware-min-k", str(best.get("count_aware_min_k", 3)),
    "--count-aware-max-k", str(best.get("count_aware_max_k", 5)),
    "--count-aware-length-norm", str(best.get("count_aware_length_norm", 12.0)),
    "--count-mode", str(best.get("count_mode", "score_sum")),
]
if best.get("valid_before_maxdet", False):
    args.append("--valid-before-maxdet")
if best.get("count_aware_topk", False):
    args.append("--count-aware-topk")
print(" ".join(shlex.quote(x) for x in args))
PY
)"

# shellcheck disable=SC2086
python tools/eval_tusimple_official.py \
  --archive-root "${ARCHIVE_ROOT}" \
  --split val \
  --gt-json "${GT_JSON}" \
  --weights "${WEIGHTS}" \
  --imgsz 544 960 \
  --device "${DEVICE}" \
  --half \
  --save-dir "${ACC_DIR}" \
  --save-records \
  ${BEST_ARGS}
