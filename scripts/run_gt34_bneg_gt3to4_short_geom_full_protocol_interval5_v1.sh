#!/usr/bin/env bash
set -euo pipefail

# Full protocol for GT3/GT4-only clear-far boundary pseudo negatives plus
# conservative GT4/GT5 short-geometry rescue. This intentionally excludes GT5
# images from boundary pseudo-negative pressure through BNEG_MAX_GT_COUNT=4.
# It runs training-time official-val selection every 5 epochs, post-train
# official-val sweeps for official_best.pt and best.pt, then reporting-only
# TEST for both checkpoints from their val-selected decodes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

RUN_NAME="${RUN_NAME:-query_alpha05_gt34_bneg_gt3to4_short_geom_interval5_v1}"
PROJECT="${PROJECT:-runs/gcs_lane}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml}"
DATA="${DATA:-data/tusimple_gcs_fixed_y_960x544.yaml}"
PRETRAINED="${PRETRAINED:-yolo11s-seg.pt}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-220}"
WORKERS="${WORKERS:-4}"
OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-5}"
OFFICIAL_WARMUP="${OFFICIAL_WARMUP:-5}"
OFFICIAL_MAX_IMAGES="${OFFICIAL_MAX_IMAGES:-0}"
VAL_MAX_IMAGES="${VAL_MAX_IMAGES:-0}"
TEST_MAX_IMAGES="${TEST_MAX_IMAGES:-0}"
SWEEP_WARMUP="${SWEEP_WARMUP:-20}"
TEST_WARMUP="${TEST_WARMUP:-20}"
HALF="${HALF:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
OVERWRITE_SWEEPS="${OVERWRITE_SWEEPS:-0}"
OVERWRITE_TESTS="${OVERWRITE_TESTS:-0}"
VALID_BEFORE_MAXDET="${VALID_BEFORE_MAXDET:-1}"

OFFICIAL_CONFS="${OFFICIAL_CONFS:-0.001 0.003 0.005 0.008 0.01 0.015 0.02}"
OFFICIAL_POINT_VALID_THRS="${OFFICIAL_POINT_VALID_THRS:-0.45 0.5 0.55 0.575 0.6}"
OFFICIAL_NMS_DIST_PXS="${OFFICIAL_NMS_DIST_PXS:-0 18 30 50}"
OFFICIAL_MAX_DETS="${OFFICIAL_MAX_DETS:-5 6 8}"
OFFICIAL_MIN_POINTS="${OFFICIAL_MIN_POINTS:-2 3 4 5}"
OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"

SWEEP_CONFS="${SWEEP_CONFS:-${OFFICIAL_CONFS}}"
SWEEP_POINT_VALID_THRS="${SWEEP_POINT_VALID_THRS:-${OFFICIAL_POINT_VALID_THRS}}"
SWEEP_NMS_DIST_PXS="${SWEEP_NMS_DIST_PXS:-${OFFICIAL_NMS_DIST_PXS}}"
SWEEP_MAX_DETS="${SWEEP_MAX_DETS:-${OFFICIAL_MAX_DETS}}"
SWEEP_MIN_POINTS="${SWEEP_MIN_POINTS:-${OFFICIAL_MIN_POINTS}}"
SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-${OFFICIAL_COUNT_MODES}}"

SHORT_GEOM_GAIN="${SHORT_GEOM_GAIN:-1.0}"
SHORT_GEOM_VISIBLE_THR="${SHORT_GEOM_VISIBLE_THR:-20}"
SHORT_GEOM_FOCUS_VISIBLE_THR="${SHORT_GEOM_FOCUS_VISIBLE_THR:-10}"
SHORT_GEOM_FOCUS_WEIGHT="${SHORT_GEOM_FOCUS_WEIGHT:-1.5}"
SHORT_GEOM_GT4_WEIGHT="${SHORT_GEOM_GT4_WEIGHT:-1.0}"
SHORT_GEOM_GT5_WEIGHT="${SHORT_GEOM_GT5_WEIGHT:-2.0}"
SHORT_GEOM_MAX_WEIGHT="${SHORT_GEOM_MAX_WEIGHT:-3.0}"
SHORT_GEOM_CURVE="${SHORT_GEOM_CURVE:-1.0}"

BNEG_GAIN="${BNEG_GAIN:-0.02}"
BNEG_VISIBLE_THR="${BNEG_VISIBLE_THR:-10}"
BNEG_DIST_THR="${BNEG_DIST_THR:-80}"
BNEG_VALID_THR="${BNEG_VALID_THR:-0.5}"
BNEG_MIN_VALID="${BNEG_MIN_VALID:-4}"
BNEG_GT_COUNT="${BNEG_GT_COUNT:-3}"
BNEG_MAX_GT_COUNT="${BNEG_MAX_GT_COUNT:-4}"
BNEG_SCORE_THR="${BNEG_SCORE_THR:-0.2}"
BNEG_ENVELOPE_MARGIN_PX="${BNEG_ENVELOPE_MARGIN_PX:-30}"
BNEG_ENVELOPE_RATIO_THR="${BNEG_ENVELOPE_RATIO_THR:-0.75}"

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

if [[ ! -f "${GT_JSON}" ]]; then
  echo "Missing canonical official-val GT JSON: ${GT_JSON}" >&2
  exit 2
fi
if [[ ! -d "${ARCHIVE_ROOT}" ]]; then
  echo "Missing TuSimple archive root: ${ARCHIVE_ROOT}" >&2
  exit 2
fi

RUN_DIR="${PROJECT}/${RUN_NAME}"
OFFICIAL_BEST_WEIGHTS="${RUN_DIR}/weights/official_best.pt"
BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"
DECODE_TAG="default_decode"
TRAIN_VALID_ARGS=(--no-gcs-official-valid-before-maxdet)
SWEEP_VALID_ARGS=()
if is_true "${VALID_BEFORE_MAXDET}"; then
  DECODE_TAG="valid_before_maxdet"
  TRAIN_VALID_ARGS=(--gcs-official-valid-before-maxdet)
  SWEEP_VALID_ARGS=(--valid-before-maxdet)
fi

OFFICIAL_HALF_ARGS=()
SWEEP_HALF_ARGS=()
if is_true "${HALF}"; then
  OFFICIAL_HALF_ARGS=(--gcs-official-half)
  SWEEP_HALF_ARGS=(--half)
fi

OFFICIAL_BEST_SWEEP_DIR="${OFFICIAL_BEST_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_official_best_val_sweep_${DECODE_TAG}}"
BEST_SWEEP_DIR="${BEST_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_best_val_sweep_${DECODE_TAG}}"
OFFICIAL_BEST_TEST_DIR="${OFFICIAL_BEST_TEST_DIR:-${PROJECT}/${RUN_NAME}_official_best_test_from_val_sweep_${DECODE_TAG}}"
BEST_TEST_DIR="${BEST_TEST_DIR:-${PROJECT}/${RUN_NAME}_best_test_from_val_sweep_${DECODE_TAG}}"
PROTOCOL_SUMMARY="${PROTOCOL_SUMMARY:-${PROJECT}/${RUN_NAME}_official_best_and_best_test_protocol_summary.json}"

run_train() {
  if [[ -e "${RUN_DIR}" ]]; then
    echo "Run directory already exists: ${RUN_DIR}" >&2
    echo "Set RUN_TRAIN=0 to reuse it, or set RUN_NAME to a new value." >&2
    exit 2
  fi

  echo "Run name: ${RUN_NAME}"
  echo "Official interval: ${OFFICIAL_INTERVAL}"
  echo "Short geom: gain=${SHORT_GEOM_GAIN}, visible<=${SHORT_GEOM_VISIBLE_THR}, focus<=${SHORT_GEOM_FOCUS_VISIBLE_THR}, gt4=${SHORT_GEOM_GT4_WEIGHT}, gt5=${SHORT_GEOM_GT5_WEIGHT}"
  echo "Boundary pseudo neg: gain=${BNEG_GAIN}, gt_count=${BNEG_GT_COUNT}..${BNEG_MAX_GT_COUNT}, dist>=${BNEG_DIST_THR}, visible<=${BNEG_VISIBLE_THR}, min_valid=${BNEG_MIN_VALID}, score>=${BNEG_SCORE_THR}, envelope_margin=${BNEG_ENVELOPE_MARGIN_PX}"

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
    --gcs-count 0.0 \
    --gcs-count-under5 0.0 \
    --gcs-count-boundary 0.0 \
    --gcs-query-count-ce 0.0 \
    --gcs-exist-quality-alpha 0.5 \
    --gcs-short-geom "${SHORT_GEOM_GAIN}" \
    --gcs-short-geom-visible-thr "${SHORT_GEOM_VISIBLE_THR}" \
    --gcs-short-geom-focus-visible-thr "${SHORT_GEOM_FOCUS_VISIBLE_THR}" \
    --gcs-short-geom-focus-weight "${SHORT_GEOM_FOCUS_WEIGHT}" \
    --gcs-short-geom-gt4-weight "${SHORT_GEOM_GT4_WEIGHT}" \
    --gcs-short-geom-gt5-weight "${SHORT_GEOM_GT5_WEIGHT}" \
    --gcs-short-geom-max-weight "${SHORT_GEOM_MAX_WEIGHT}" \
    --gcs-short-geom-curve "${SHORT_GEOM_CURVE}" \
    --gcs-gt5-short-visible-thr 0 \
    --gcs-boundary-pseudo-neg "${BNEG_GAIN}" \
    --gcs-boundary-pseudo-visible-thr "${BNEG_VISIBLE_THR}" \
    --gcs-boundary-pseudo-dist-thr "${BNEG_DIST_THR}" \
    --gcs-boundary-pseudo-valid-thr "${BNEG_VALID_THR}" \
    --gcs-boundary-pseudo-min-valid "${BNEG_MIN_VALID}" \
    --gcs-boundary-pseudo-gt-count "${BNEG_GT_COUNT}" \
    --gcs-boundary-pseudo-max-gt-count "${BNEG_MAX_GT_COUNT}" \
    --gcs-boundary-pseudo-score-thr "${BNEG_SCORE_THR}" \
    --gcs-boundary-pseudo-envelope-margin-px "${BNEG_ENVELOPE_MARGIN_PX}" \
    --gcs-boundary-pseudo-envelope-ratio-thr "${BNEG_ENVELOPE_RATIO_THR}" \
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
    "${TRAIN_VALID_ARGS[@]}" \
    "${OFFICIAL_HALF_ARGS[@]}" \
    --project "${PROJECT}" \
    --name "${RUN_NAME}"
}

run_val_sweep() {
  local label="$1"
  local weights="$2"
  local save_dir="$3"

  if [[ ! -f "${weights}" ]]; then
    echo "Missing ${label} weights: ${weights}" >&2
    exit 2
  fi
  if [[ -e "${save_dir}" ]] && ! is_true "${OVERWRITE_SWEEPS}"; then
    echo "Sweep directory already exists: ${save_dir}" >&2
    echo "Set OVERWRITE_SWEEPS=1 or choose a new RUN_NAME/SWEEP dir." >&2
    exit 2
  fi

  python tools/sweep_tusimple_official_cached.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split val \
    --gt-json "${GT_JSON}" \
    --weights "${weights}" \
    --imgsz 544 960 \
    --device "${DEVICE}" \
    --max-images "${VAL_MAX_IMAGES}" \
    --warmup "${SWEEP_WARMUP}" \
    "${SWEEP_HALF_ARGS[@]}" \
    --confs ${SWEEP_CONFS} \
    --point-valid-thrs ${SWEEP_POINT_VALID_THRS} \
    --nms-dist-pxs ${SWEEP_NMS_DIST_PXS} \
    --max-dets ${SWEEP_MAX_DETS} \
    --min-points ${SWEEP_MIN_POINTS} \
    --count-modes ${SWEEP_COUNT_MODES} \
    "${SWEEP_VALID_ARGS[@]}" \
    --save-dir "${save_dir}"
}

run_test_from_sweep() {
  local label="$1"
  local weights="$2"
  local sweep_dir="$3"
  local save_dir="$4"
  local summary="${sweep_dir}/tusimple_official_sweep_summary.json"

  if [[ ! -f "${summary}" ]]; then
    echo "Missing ${label} sweep summary: ${summary}" >&2
    exit 2
  fi
  if [[ -e "${save_dir}" ]] && ! is_true "${OVERWRITE_TESTS}"; then
    echo "Test directory already exists: ${save_dir}" >&2
    echo "Set OVERWRITE_TESTS=1 or choose a new RUN_NAME/TEST dir." >&2
    exit 2
  fi

  python - "${summary}" "${weights}" "${save_dir}" "${ARCHIVE_ROOT}" "${DEVICE}" "${TEST_MAX_IMAGES}" "${TEST_WARMUP}" "${HALF}" <<'PY'
import json
import shlex
import subprocess
import sys

summary_path, weights, save_dir, archive_root, device, max_images, warmup, half = sys.argv[1:]
with open(summary_path, "r", encoding="utf-8") as f:
    best = json.load(f)["best"]

cmd = [
    sys.executable,
    "tools/eval_tusimple_official.py",
    "--archive-root",
    archive_root,
    "--split",
    "test",
    "--weights",
    weights,
    "--imgsz",
    "544",
    "960",
    "--device",
    device,
    "--conf",
    str(best["conf"]),
    "--point-valid-thr",
    str(best["point_valid_thr"]),
    "--nms-dist-px",
    str(best["nms_dist_px"]),
    "--max-det",
    str(int(best["max_det"])),
    "--min-points",
    str(int(best["min_points"])),
    "--count-mode",
    str(best.get("count_mode", "score_sum")),
    "--warmup",
    str(int(max(0, int(warmup)))),
    "--save-dir",
    save_dir,
    "--save-records",
]
if str(half).strip().lower() in {"1", "true", "yes", "on"}:
    cmd.append("--half")
if int(max_images) > 0:
    cmd.extend(["--max-images", str(int(max_images))])
if bool(best.get("valid_before_maxdet", False)):
    cmd.append("--valid-before-maxdet")

print("[test] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
subprocess.run(cmd, check=True)
PY
}

write_protocol_summary() {
  python - \
    "${RUN_NAME}" \
    "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" "${OFFICIAL_BEST_TEST_DIR}" \
    "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}" "${BEST_TEST_DIR}" \
    "${PROTOCOL_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

(
    run_name,
    official_best_weights,
    official_best_sweep_dir,
    official_best_test_dir,
    best_weights,
    best_sweep_dir,
    best_test_dir,
    protocol_summary,
) = sys.argv[1:]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def package(label: str, weights: str, sweep_dir: str, test_dir: str) -> dict:
    sweep_summary = str(Path(sweep_dir) / "tusimple_official_sweep_summary.json")
    test_summary = str(Path(test_dir) / "tusimple_official_summary.json")
    return {
        "label": label,
        "weights": weights,
        "val_sweep_summary": sweep_summary,
        "val_best": load_json(sweep_summary)["best"],
        "test_summary": test_summary,
        "test_metrics": load_json(test_summary)["metrics"],
    }


output = {
    "run_name": run_name,
    "primary_checkpoint": "official_best.pt",
    "selection_split": "official-val",
    "test_usage": "reporting_only_from_each_val_selected_decode",
    "do_not_select_from_test": True,
    "best_pt_test_role": "reporting_only_not_selection",
    "official_best": package("official_best.pt", official_best_weights, official_best_sweep_dir, official_best_test_dir),
    "best": package("best.pt", best_weights, best_sweep_dir, best_test_dir),
}
Path(protocol_summary).parent.mkdir(parents=True, exist_ok=True)
Path(protocol_summary).write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(
    {
        "summary": protocol_summary,
        "official_best_test_metrics": output["official_best"]["test_metrics"],
        "best_test_metrics": output["best"]["test_metrics"],
    },
    indent=2,
))
PY
}

if is_true "${RUN_TRAIN}"; then
  run_train
else
  echo "RUN_TRAIN=0: reusing existing run directory ${RUN_DIR}" >&2
fi

run_val_sweep "official_best.pt" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}"
run_val_sweep "best.pt" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}"
run_test_from_sweep "official_best.pt" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" "${OFFICIAL_BEST_TEST_DIR}"
run_test_from_sweep "best.pt" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}" "${BEST_TEST_DIR}"
write_protocol_summary
