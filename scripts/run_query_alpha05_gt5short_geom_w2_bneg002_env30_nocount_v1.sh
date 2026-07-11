#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the remote CUDA server.
# This is the true boundary-pseudo mask-v2 / envelope-protected protocol.
# Override variables from the shell when needed, e.g. BATCH=16 DEVICE=1 bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh

RUN_NAME="${RUN_NAME:-query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1}"
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
RUN_TESTS="${RUN_TESTS:-0}"
VALID_BEFORE_MAXDET="${VALID_BEFORE_MAXDET:-1}"

OFFICIAL_CONFS="${OFFICIAL_CONFS:-0.001 0.003 0.005 0.008 0.01 0.02}"
OFFICIAL_POINT_VALID_THRS="${OFFICIAL_POINT_VALID_THRS:-0.45 0.50 0.55 0.60}"
OFFICIAL_NMS_DIST_PXS="${OFFICIAL_NMS_DIST_PXS:-0 18 30}"
OFFICIAL_MAX_DETS="${OFFICIAL_MAX_DETS:-5 6 8}"
OFFICIAL_MIN_POINTS="${OFFICIAL_MIN_POINTS:-2 3 4 5}"
OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"

SWEEP_CONFS="${SWEEP_CONFS:-${OFFICIAL_CONFS}}"
SWEEP_POINT_VALID_THRS="${SWEEP_POINT_VALID_THRS:-${OFFICIAL_POINT_VALID_THRS}}"
SWEEP_NMS_DIST_PXS="${SWEEP_NMS_DIST_PXS:-${OFFICIAL_NMS_DIST_PXS}}"
SWEEP_MAX_DETS="${SWEEP_MAX_DETS:-${OFFICIAL_MAX_DETS}}"
SWEEP_MIN_POINTS="${SWEEP_MIN_POINTS:-${OFFICIAL_MIN_POINTS}}"
SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-${OFFICIAL_COUNT_MODES}}"

read -r -a OFFICIAL_CONFS_ARR <<< "${OFFICIAL_CONFS}"
read -r -a OFFICIAL_POINT_VALID_THRS_ARR <<< "${OFFICIAL_POINT_VALID_THRS}"
read -r -a OFFICIAL_NMS_DIST_PXS_ARR <<< "${OFFICIAL_NMS_DIST_PXS}"
read -r -a OFFICIAL_MAX_DETS_ARR <<< "${OFFICIAL_MAX_DETS}"
read -r -a OFFICIAL_MIN_POINTS_ARR <<< "${OFFICIAL_MIN_POINTS}"
read -r -a OFFICIAL_COUNT_MODES_ARR <<< "${OFFICIAL_COUNT_MODES}"
read -r -a SWEEP_CONFS_ARR <<< "${SWEEP_CONFS}"
read -r -a SWEEP_POINT_VALID_THRS_ARR <<< "${SWEEP_POINT_VALID_THRS}"
read -r -a SWEEP_NMS_DIST_PXS_ARR <<< "${SWEEP_NMS_DIST_PXS}"
read -r -a SWEEP_MAX_DETS_ARR <<< "${SWEEP_MAX_DETS}"
read -r -a SWEEP_MIN_POINTS_ARR <<< "${SWEEP_MIN_POINTS}"
read -r -a SWEEP_COUNT_MODES_ARR <<< "${SWEEP_COUNT_MODES}"

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
    --gcs-exist-quality-alpha 0.5 \
    --gcs-count 0.0 \
    --gcs-count-under5 0.0 \
    --gcs-count-boundary 0.0 \
    --gcs-query-count-ce 0.0 \
    --gcs-short-geom 1.0 \
    --gcs-short-geom-visible-thr 10 \
    --gcs-short-geom-gt5-weight 2.0 \
    --gcs-short-geom-max-weight 3.0 \
    --gcs-short-geom-curve 1.0 \
    --gcs-gt5-short-visible-thr 0 \
    --gcs-boundary-pseudo-neg 0.02 \
    --gcs-boundary-pseudo-visible-thr 10 \
    --gcs-boundary-pseudo-dist-thr 80 \
    --gcs-boundary-pseudo-valid-thr 0.5 \
    --gcs-boundary-pseudo-min-valid 4 \
    --gcs-boundary-pseudo-gt-count 5 \
    --gcs-boundary-pseudo-score-thr 0.2 \
    --gcs-boundary-pseudo-envelope-margin-px 30 \
    --gcs-boundary-pseudo-envelope-ratio-thr 0.75 \
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
    --confs "${SWEEP_CONFS_ARR[@]}" \
    --point-valid-thrs "${SWEEP_POINT_VALID_THRS_ARR[@]}" \
    --nms-dist-pxs "${SWEEP_NMS_DIST_PXS_ARR[@]}" \
    --max-dets "${SWEEP_MAX_DETS_ARR[@]}" \
    --min-points "${SWEEP_MIN_POINTS_ARR[@]}" \
    --count-modes "${SWEEP_COUNT_MODES_ARR[@]}" \
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

if best.get("decode_mode", "query") == "ordered_slot":
    raise SystemExit("This protocol script is for the query mask-v2 run, not ordered_slot decode.")

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
if bool(best.get("count_aware_topk", False)):
    cmd.append("--count-aware-topk")
    cmd.extend(["--count-aware-min-k", str(int(best.get("count_aware_min_k", 3)))])
    cmd.extend(["--count-aware-max-k", str(int(best.get("count_aware_max_k", 5)))])
    cmd.extend(["--count-aware-length-norm", str(float(best.get("count_aware_length_norm", 12.0)))])

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
    "mask_v2_params": {
        "gcs_boundary_pseudo_neg": 0.02,
        "gcs_boundary_pseudo_dist_thr": 80,
        "gcs_boundary_pseudo_min_valid": 4,
        "gcs_boundary_pseudo_score_thr": 0.2,
        "gcs_boundary_pseudo_envelope_margin_px": 30,
        "gcs_boundary_pseudo_envelope_ratio_thr": 0.75,
    },
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

echo "Run name: ${RUN_NAME}"
echo "Mask-v2 boundary pseudo params: neg=0.02 dist_thr=80 min_valid=4 score_thr=0.2 envelope_margin_px=30 envelope_ratio_thr=0.75"
echo "Selection GT: ${GT_JSON}"
echo "Training-time official_best and post-train sweeps use tools/sweep_tusimple_official_cached.py."
echo "RUN_TESTS=${RUN_TESTS}: official test is reporting-only and must stay off until official-val and diagnostics pass."

if is_true "${RUN_TRAIN}"; then
  run_train
else
  echo "RUN_TRAIN=0: reusing existing run directory ${RUN_DIR}" >&2
fi

run_val_sweep "official_best.pt" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}"
run_val_sweep "best.pt" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}"
if is_true "${RUN_TESTS}"; then
  run_test_from_sweep "official_best.pt" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" "${OFFICIAL_BEST_TEST_DIR}"
  run_test_from_sweep "best.pt" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}" "${BEST_TEST_DIR}"
  write_protocol_summary
else
  echo "RUN_TESTS=0: skipping official test and test protocol summary." >&2
  echo "Run official-val diagnostics first; set RUN_TESTS=1 only for reporting-only test after gates pass." >&2
fi
