#!/usr/bin/env bash
set -euo pipefail

# Full protocol for:
# Q12 ultrashort dataref + env30 boundary mask + short geometry loss.
# Run from the repository root on the remote CUDA server.
#
# The protocol is:
# train with training-time official_best
# -> post-train official-val sweep for official_best.pt
# -> post-train official-val sweep for best.pt
# -> optional reporting-only TEST evaluation for both val-selected decodes
# -> raw-Q12 diagnostics for both checkpoints on val and optional test
#
# TEST is never used for selection.

RUN_NAME="${RUN_NAME:-query_alpha05_env30_q12_ultrashort_dataref_geom1_full_protocol_v1}"
PROJECT="${PROJECT:-runs/gcs_lane}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-ultrashort-dataref.yaml}"
DATA="${DATA:-data/tusimple_gcs_fixed_y_960x544.yaml}"
PRETRAINED="${PRETRAINED:-yolo11s-seg.pt}"
BANK="${BANK:-data/gcs_reference_banks/q12_ultrashort_env30_train_v1.json}"
REFERENCE_AUDIT="${REFERENCE_AUDIT:-data/gcs_reference_banks/q12_ultrashort_env30_train_v1_val_reference_audit.json}"
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
DIAG_WARMUP="${DIAG_WARMUP:-20}"
HALF="${HALF:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
RUN_TEST_DIAGNOSTICS="${RUN_TEST_DIAGNOSTICS:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
OVERWRITE_SWEEPS="${OVERWRITE_SWEEPS:-0}"
OVERWRITE_TESTS="${OVERWRITE_TESTS:-0}"
OVERWRITE_DIAGNOSTICS="${OVERWRITE_DIAGNOSTICS:-0}"
ENFORCE_REFERENCE_GATE="${ENFORCE_REFERENCE_GATE:-1}"
VALID_BEFORE_MAXDET="${VALID_BEFORE_MAXDET:-1}"

# Experiment-specific loss contract. These values are intentionally checked
# below so an accidental environment override cannot change the candidate.
SHORT_GEOM="${SHORT_GEOM:-1.0}"
SHORT_GEOM_VISIBLE_THR="${SHORT_GEOM_VISIBLE_THR:-10}"
SHORT_GEOM_GT4_WEIGHT="${SHORT_GEOM_GT4_WEIGHT:-1.0}"
SHORT_GEOM_GT5_WEIGHT="${SHORT_GEOM_GT5_WEIGHT:-2.0}"
SHORT_GEOM_MAX_WEIGHT="${SHORT_GEOM_MAX_WEIGHT:-3.0}"
SHORT_GEOM_CURVE="${SHORT_GEOM_CURVE:-1.0}"
SHORT_GEOM_TIERED="${SHORT_GEOM_TIERED:-0}"
GT4_SHORT_VISIBLE_THR="${GT4_SHORT_VISIBLE_THR:-0}"
GT4_SHORT_POINT_VALID_WEIGHT="${GT4_SHORT_POINT_VALID_WEIGHT:-1.0}"
GT5_SHORT_VISIBLE_THR="${GT5_SHORT_VISIBLE_THR:-0}"
GT5_SHORT_POINT_VALID_WEIGHT="${GT5_SHORT_POINT_VALID_WEIGHT:-1.0}"

# Env30 boundary-mask contract.
BOUNDARY_PSEUDO_NEG="${BOUNDARY_PSEUDO_NEG:-0.02}"
BOUNDARY_PSEUDO_VISIBLE_THR="${BOUNDARY_PSEUDO_VISIBLE_THR:-10}"
BOUNDARY_PSEUDO_DIST_THR="${BOUNDARY_PSEUDO_DIST_THR:-80}"
BOUNDARY_PSEUDO_VALID_THR="${BOUNDARY_PSEUDO_VALID_THR:-0.5}"
BOUNDARY_PSEUDO_MIN_VALID="${BOUNDARY_PSEUDO_MIN_VALID:-4}"
BOUNDARY_PSEUDO_GT_COUNT="${BOUNDARY_PSEUDO_GT_COUNT:-5}"
BOUNDARY_PSEUDO_SCORE_THR="${BOUNDARY_PSEUDO_SCORE_THR:-0.2}"
BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX="${BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX:-30}"
BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR="${BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR:-0.75}"

# Use the same complete 864-row grid for training-time and post-train
# official-val selection: 6 * 4 * 3 * 3 * 4 * 1.
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

DIAG_MATCH_THR_PX="${DIAG_MATCH_THR_PX:-20}"
DIAG_MATCH_MIN_OVERLAP="${DIAG_MATCH_MIN_OVERLAP:-3}"
DIAG_SHORT_VISIBLE_MAX="${DIAG_SHORT_VISIBLE_MAX:-10}"

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

if [[ ! -f "${MODEL}" ]]; then
  echo "Missing model YAML: ${MODEL}" >&2
  exit 2
fi
if [[ ! -f "${DATA}" ]]; then
  echo "Missing data YAML: ${DATA}" >&2
  exit 2
fi
if [[ ! -f "${BANK}" ]]; then
  echo "Missing reference bank: ${BANK}" >&2
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

validate_contract() {
  python - \
    "${SHORT_GEOM}" "${SHORT_GEOM_VISIBLE_THR}" \
    "${SHORT_GEOM_GT4_WEIGHT}" "${SHORT_GEOM_GT5_WEIGHT}" \
    "${SHORT_GEOM_MAX_WEIGHT}" "${SHORT_GEOM_CURVE}" "${SHORT_GEOM_TIERED}" \
    "${GT4_SHORT_VISIBLE_THR}" "${GT4_SHORT_POINT_VALID_WEIGHT}" \
    "${GT5_SHORT_VISIBLE_THR}" "${GT5_SHORT_POINT_VALID_WEIGHT}" \
    "${BOUNDARY_PSEUDO_NEG}" "${BOUNDARY_PSEUDO_VISIBLE_THR}" \
    "${BOUNDARY_PSEUDO_DIST_THR}" "${BOUNDARY_PSEUDO_VALID_THR}" \
    "${BOUNDARY_PSEUDO_MIN_VALID}" "${BOUNDARY_PSEUDO_GT_COUNT}" \
    "${BOUNDARY_PSEUDO_SCORE_THR}" "${BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX}" \
    "${BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR}" \
    "${VALID_BEFORE_MAXDET}" "${OFFICIAL_CONFS}" \
    "${OFFICIAL_POINT_VALID_THRS}" "${OFFICIAL_NMS_DIST_PXS}" \
    "${OFFICIAL_MAX_DETS}" "${OFFICIAL_MIN_POINTS}" \
    "${OFFICIAL_COUNT_MODES}" "${SWEEP_CONFS}" \
    "${SWEEP_POINT_VALID_THRS}" "${SWEEP_NMS_DIST_PXS}" \
    "${SWEEP_MAX_DETS}" "${SWEEP_MIN_POINTS}" "${SWEEP_COUNT_MODES}" <<'PY'
from decimal import Decimal
import sys

(
    short_geom,
    short_geom_visible_thr,
    short_geom_gt4_weight,
    short_geom_gt5_weight,
    short_geom_max_weight,
    short_geom_curve,
    short_geom_tiered,
    gt4_short_visible_thr,
    gt4_short_point_valid_weight,
    gt5_short_visible_thr,
    gt5_short_point_valid_weight,
    boundary_pseudo_neg,
    boundary_pseudo_visible_thr,
    boundary_pseudo_dist_thr,
    boundary_pseudo_valid_thr,
    boundary_pseudo_min_valid,
    boundary_pseudo_gt_count,
    boundary_pseudo_score_thr,
    boundary_pseudo_margin,
    boundary_pseudo_ratio,
    valid_before_maxdet,
    official_confs,
    official_valid,
    official_nms,
    official_max_dets,
    official_min_points,
    official_count_modes,
    sweep_confs,
    sweep_valid,
    sweep_nms,
    sweep_max_dets,
    sweep_min_points,
    sweep_count_modes,
) = sys.argv[1:]


def decimal(value):
    return Decimal(value)


def require_decimal(name, actual, expected):
    if decimal(actual) != decimal(expected):
        raise SystemExit(f"{name} must be {expected}, got {actual}")


def require_int(name, actual, expected):
    if int(actual) != expected:
        raise SystemExit(f"{name} must be {expected}, got {actual}")


require_decimal("SHORT_GEOM", short_geom, "1.0")
require_int("SHORT_GEOM_VISIBLE_THR", short_geom_visible_thr, 10)
require_decimal("SHORT_GEOM_GT4_WEIGHT", short_geom_gt4_weight, "1.0")
require_decimal("SHORT_GEOM_GT5_WEIGHT", short_geom_gt5_weight, "2.0")
require_decimal("SHORT_GEOM_MAX_WEIGHT", short_geom_max_weight, "3.0")
require_decimal("SHORT_GEOM_CURVE", short_geom_curve, "1.0")
if short_geom_tiered.strip().lower() not in {"0", "false", "no", "off"}:
    raise SystemExit("SHORT_GEOM_TIERED must remain disabled for this experiment")
require_int("GT4_SHORT_VISIBLE_THR", gt4_short_visible_thr, 0)
require_decimal("GT4_SHORT_POINT_VALID_WEIGHT", gt4_short_point_valid_weight, "1.0")
require_int("GT5_SHORT_VISIBLE_THR", gt5_short_visible_thr, 0)
require_decimal("GT5_SHORT_POINT_VALID_WEIGHT", gt5_short_point_valid_weight, "1.0")

require_decimal("BOUNDARY_PSEUDO_NEG", boundary_pseudo_neg, "0.02")
require_int("BOUNDARY_PSEUDO_VISIBLE_THR", boundary_pseudo_visible_thr, 10)
require_decimal("BOUNDARY_PSEUDO_DIST_THR", boundary_pseudo_dist_thr, "80")
require_decimal("BOUNDARY_PSEUDO_VALID_THR", boundary_pseudo_valid_thr, "0.5")
require_int("BOUNDARY_PSEUDO_MIN_VALID", boundary_pseudo_min_valid, 4)
require_int("BOUNDARY_PSEUDO_GT_COUNT", boundary_pseudo_gt_count, 5)
require_decimal("BOUNDARY_PSEUDO_SCORE_THR", boundary_pseudo_score_thr, "0.2")
require_decimal("BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX", boundary_pseudo_margin, "30")
require_decimal("BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR", boundary_pseudo_ratio, "0.75")

if valid_before_maxdet.strip().lower() not in {"1", "true", "yes", "on"}:
    raise SystemExit("VALID_BEFORE_MAXDET must remain enabled for this protocol")

expected_grid = {
    "confs": ["0.001", "0.003", "0.005", "0.008", "0.01", "0.02"],
    "valid": ["0.45", "0.50", "0.55", "0.60"],
    "nms": ["0", "18", "30"],
    "max_dets": ["5", "6", "8"],
    "min_points": ["2", "3", "4", "5"],
    "count_modes": ["score_sum"],
}


def split_grid(value):
    return value.split()


for name, actual, key in [
    ("OFFICIAL_CONFS", official_confs, "confs"),
    ("OFFICIAL_POINT_VALID_THRS", official_valid, "valid"),
    ("OFFICIAL_NMS_DIST_PXS", official_nms, "nms"),
    ("OFFICIAL_MAX_DETS", official_max_dets, "max_dets"),
    ("OFFICIAL_MIN_POINTS", official_min_points, "min_points"),
    ("OFFICIAL_COUNT_MODES", official_count_modes, "count_modes"),
    ("SWEEP_CONFS", sweep_confs, "confs"),
    ("SWEEP_POINT_VALID_THRS", sweep_valid, "valid"),
    ("SWEEP_NMS_DIST_PXS", sweep_nms, "nms"),
    ("SWEEP_MAX_DETS", sweep_max_dets, "max_dets"),
    ("SWEEP_MIN_POINTS", sweep_min_points, "min_points"),
    ("SWEEP_COUNT_MODES", sweep_count_modes, "count_modes"),
]:
    expected = expected_grid[key]
    actual_values = split_grid(actual)
    if actual_values != expected:
        raise SystemExit(f"{name} must be {' '.join(expected)}, got {actual}")

print("Experiment parameter contract passed.")
print("Official-val grid rows: 864")
PY
}

validate_reference_gate() {
  if ! is_true "${ENFORCE_REFERENCE_GATE}"; then
    echo "ENFORCE_REFERENCE_GATE=0: bypassing static reference gate. Use only for debugging." >&2
    return
  fi
  if [[ ! -f "${REFERENCE_AUDIT}" ]]; then
    echo "Missing static reference audit: ${REFERENCE_AUDIT}" >&2
    echo "Run tools/check_gcs_reference_bank_coverage.py before training." >&2
    exit 2
  fi
  python - "${REFERENCE_AUDIT}" "${BANK}" "${GT_JSON}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
bank_path = Path(sys.argv[2])
expected_gt_json = str(sys.argv[3]).replace("\\", "/")

with audit_path.open("r", encoding="utf-8") as f:
    audit = json.load(f)
with bank_path.open("r", encoding="utf-8") as f:
    bank = json.load(f)

if audit.get("schema") != "gcs_q12_fixed_y_reference_bank_coverage_audit_v1":
    raise SystemExit(f"Unexpected static audit schema: {audit.get('schema')!r}")
if audit.get("bank_schema") != "gcs_q12_fixed_y_reference_bank_v1":
    raise SystemExit(f"Unexpected bank schema recorded in audit: {audit.get('bank_schema')!r}")
if bank.get("schema") != "gcs_q12_fixed_y_reference_bank_v1":
    raise SystemExit(f"Unexpected bank schema: {bank.get('schema')!r}")
if bank.get("point_mode") != "fixed_y":
    raise SystemExit(f"Reference bank point_mode must be fixed_y, got {bank.get('point_mode')!r}")
if int(bank.get("num_queries", -1)) != 12 or int(bank.get("num_points", -1)) != 56:
    raise SystemExit("Reference bank must declare num_queries=12 and num_points=56")
x_norm = bank.get("x_norm")
if not isinstance(x_norm, list) or len(x_norm) != 12 or any(len(row) != 56 for row in x_norm):
    raise SystemExit("Reference bank x_norm must have shape 12 x 56")

actual_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest()
if audit.get("bank_sha256") != actual_hash:
    raise SystemExit(
        f"Static reference audit does not match bank. "
        f"audit_hash={audit.get('bank_sha256')!r} bank_hash={actual_hash!r}"
    )

audit_gt_json = [str(p).replace("\\", "/") for p in audit.get("gt_json", [])]
if audit_gt_json != [expected_gt_json]:
    raise SystemExit(f"Static audit GT mismatch. expected={[expected_gt_json]!r} actual={audit_gt_json!r}")
gate = audit.get("gate", {})
if not bool(gate.get("pass", False)):
    raise SystemExit(f"Static reference gate failed in {audit_path}: {gate}")

print(f"Static reference gate passed: {audit_path}")
print(f"Reference bank shape: {len(x_norm)} x {len(x_norm[0])}")
PY
}

validate_contract
validate_reference_gate

if is_true "${PREFLIGHT_ONLY}"; then
  echo "PREFLIGHT_ONLY=1: validation completed; training, sweep, TEST, and diagnostics were not started."
  exit 0
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
OFFICIAL_BEST_VAL_DIAG_DIR="${OFFICIAL_BEST_VAL_DIAG_DIR:-${PROJECT}/${RUN_NAME}_official_best_val_raw_q12_${DECODE_TAG}}"
BEST_VAL_DIAG_DIR="${BEST_VAL_DIAG_DIR:-${PROJECT}/${RUN_NAME}_best_val_raw_q12_${DECODE_TAG}}"
OFFICIAL_BEST_TEST_DIAG_DIR="${OFFICIAL_BEST_TEST_DIAG_DIR:-${PROJECT}/${RUN_NAME}_official_best_test_raw_q12_${DECODE_TAG}}"
BEST_TEST_DIAG_DIR="${BEST_TEST_DIAG_DIR:-${PROJECT}/${RUN_NAME}_best_test_raw_q12_${DECODE_TAG}}"
PROTOCOL_SUMMARY="${PROTOCOL_SUMMARY:-${PROJECT}/${RUN_NAME}_official_best_and_best_test_protocol_summary.json}"

run_train() {
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
    --scale 0.15 \
    --erasing 0.10 \
    --mosaic 0.0 \
    --gcs-exist-quality-alpha 0.5 \
    --gcs-count 0.0 \
    --gcs-count-under5 0.0 \
    --gcs-count-boundary 0.0 \
    --gcs-query-count-ce 0.0 \
    --gcs-short-geom "${SHORT_GEOM}" \
    --gcs-short-geom-visible-thr "${SHORT_GEOM_VISIBLE_THR}" \
    --gcs-short-geom-gt4-weight "${SHORT_GEOM_GT4_WEIGHT}" \
    --gcs-short-geom-gt5-weight "${SHORT_GEOM_GT5_WEIGHT}" \
    --gcs-short-geom-max-weight "${SHORT_GEOM_MAX_WEIGHT}" \
    --gcs-short-geom-curve "${SHORT_GEOM_CURVE}" \
    --gcs-gt4-short-visible-thr "${GT4_SHORT_VISIBLE_THR}" \
    --gcs-gt4-short-point-valid-weight "${GT4_SHORT_POINT_VALID_WEIGHT}" \
    --gcs-gt5-short-visible-thr "${GT5_SHORT_VISIBLE_THR}" \
    --gcs-gt5-short-point-valid-weight "${GT5_SHORT_POINT_VALID_WEIGHT}" \
    --gcs-boundary-pseudo-neg "${BOUNDARY_PSEUDO_NEG}" \
    --gcs-boundary-pseudo-visible-thr "${BOUNDARY_PSEUDO_VISIBLE_THR}" \
    --gcs-boundary-pseudo-dist-thr "${BOUNDARY_PSEUDO_DIST_THR}" \
    --gcs-boundary-pseudo-valid-thr "${BOUNDARY_PSEUDO_VALID_THR}" \
    --gcs-boundary-pseudo-min-valid "${BOUNDARY_PSEUDO_MIN_VALID}" \
    --gcs-boundary-pseudo-gt-count "${BOUNDARY_PSEUDO_GT_COUNT}" \
    --gcs-boundary-pseudo-score-thr "${BOUNDARY_PSEUDO_SCORE_THR}" \
    --gcs-boundary-pseudo-envelope-margin-px "${BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX}" \
    --gcs-boundary-pseudo-envelope-ratio-thr "${BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR}" \
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
    echo "Set OVERWRITE_SWEEPS=1 or choose a new sweep directory." >&2
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
    echo "Set OVERWRITE_TESTS=1 or choose a new TEST directory." >&2
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
    raise SystemExit("This protocol is for query-mode checkpoints, not ordered_slot.")

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
    cmd.extend(["--count-aware-extra-margin", str(int(best.get("count_aware_extra_margin", 0)))])

print("[test] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
subprocess.run(cmd, check=True)
PY
}

run_raw_q12_diag_from_sweep() {
  local label="$1"
  local weights="$2"
  local sweep_dir="$3"
  local split="$4"
  local save_dir="$5"
  local max_images="$6"
  local summary="${sweep_dir}/tusimple_official_sweep_summary.json"

  if [[ ! -f "${summary}" ]]; then
    echo "Missing ${label} sweep summary for raw-Q12 diagnostic: ${summary}" >&2
    exit 2
  fi
  if [[ ! -f "${weights}" ]]; then
    echo "Missing ${label} weights for raw-Q12 diagnostic: ${weights}" >&2
    exit 2
  fi
  if [[ -e "${save_dir}" ]] && ! is_true "${OVERWRITE_DIAGNOSTICS}"; then
    echo "Diagnostic directory already exists: ${save_dir}" >&2
    echo "Set OVERWRITE_DIAGNOSTICS=1 or choose a new diagnostic directory." >&2
    exit 2
  fi

  python - \
    "${summary}" "${weights}" "${save_dir}" "${ARCHIVE_ROOT}" "${DEVICE}" \
    "${max_images}" "${DIAG_WARMUP}" "${HALF}" "${split}" "${GT_JSON}" \
    "${DIAG_MATCH_THR_PX}" "${DIAG_MATCH_MIN_OVERLAP}" "${DIAG_SHORT_VISIBLE_MAX}" <<'PY'
import json
import shlex
import subprocess
import sys

(
    summary_path,
    weights,
    save_dir,
    archive_root,
    device,
    max_images,
    warmup,
    half,
    split,
    gt_json,
    match_thr_px,
    match_min_overlap,
    short_visible_max,
) = sys.argv[1:]

with open(summary_path, "r", encoding="utf-8") as f:
    best = json.load(f)["best"]

if best.get("decode_mode", "query") == "ordered_slot":
    raise SystemExit("This protocol is for query-mode raw-Q12 diagnostics, not ordered_slot.")

cmd = [
    sys.executable,
    "tools/diagnose_tusimple_raw_q12_filters.py",
    "--archive-root",
    archive_root,
    "--split",
    split,
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
    "--match-thr-px",
    str(float(match_thr_px)),
    "--match-min-overlap",
    str(int(match_min_overlap)),
    "--short-visible-max",
    str(int(short_visible_max)),
    "--warmup",
    str(int(max(0, int(warmup)))),
    "--save-dir",
    save_dir,
]
if split == "val":
    cmd.extend(["--gt-json", gt_json])
elif split == "test":
    cmd.append("--allow-test-oracle")
else:
    raise SystemExit(f"Unsupported diagnostic split: {split}")
if str(half).strip().lower() in {"1", "true", "yes", "on"}:
    cmd.append("--half")
if int(max_images) > 0:
    cmd.extend(["--max-images", str(int(max_images))])
if bool(best.get("valid_before_maxdet", False)):
    cmd.append("--valid-before-maxdet")

print("[diag] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
subprocess.run(cmd, check=True)
PY
}

write_protocol_summary() {
  python - \
    "${RUN_NAME}" "${PROJECT}" "${MODEL}" "${DATA}" "${BANK}" "${REFERENCE_AUDIT}" "${GT_JSON}" \
    "${BATCH}" "${EPOCHS}" "${WORKERS}" "${OFFICIAL_INTERVAL}" "${OFFICIAL_WARMUP}" \
    "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" "${OFFICIAL_BEST_TEST_DIR}" \
    "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}" "${BEST_TEST_DIR}" \
    "${OFFICIAL_BEST_VAL_DIAG_DIR}" "${BEST_VAL_DIAG_DIR}" \
    "${OFFICIAL_BEST_TEST_DIAG_DIR}" "${BEST_TEST_DIAG_DIR}" \
    "${PROTOCOL_SUMMARY}" \
    "${SHORT_GEOM}" "${SHORT_GEOM_VISIBLE_THR}" "${SHORT_GEOM_GT4_WEIGHT}" \
    "${SHORT_GEOM_GT5_WEIGHT}" "${SHORT_GEOM_MAX_WEIGHT}" "${SHORT_GEOM_CURVE}" \
    "${GT4_SHORT_VISIBLE_THR}" "${GT4_SHORT_POINT_VALID_WEIGHT}" \
    "${GT5_SHORT_VISIBLE_THR}" "${GT5_SHORT_POINT_VALID_WEIGHT}" \
    "${VALID_BEFORE_MAXDET}" "${HALF}" <<'PY'
import json
import sys
from pathlib import Path

(
    run_name,
    project,
    model,
    data,
    bank,
    reference_audit,
    gt_json,
    batch,
    epochs,
    workers,
    official_interval,
    official_warmup,
    official_best_weights,
    official_best_sweep_dir,
    official_best_test_dir,
    best_weights,
    best_sweep_dir,
    best_test_dir,
    official_best_val_diag_dir,
    best_val_diag_dir,
    official_best_test_diag_dir,
    best_test_diag_dir,
    protocol_summary,
    short_geom,
    short_geom_visible_thr,
    short_geom_gt4_weight,
    short_geom_gt5_weight,
    short_geom_max_weight,
    short_geom_curve,
    gt4_short_visible_thr,
    gt4_short_point_valid_weight,
    gt5_short_visible_thr,
    gt5_short_point_valid_weight,
    valid_before_maxdet,
    half,
) = sys.argv[1:]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def diagnostic_package(save_dir):
    summary = Path(save_dir) / "raw_q12_filter_summary.json"
    if not summary.exists():
        return None
    return {
        "save_dir": save_dir,
        "summary": str(summary),
        "raw_gt_lane_diagnostics": str(Path(save_dir) / "raw_gt_lane_diagnostics.csv"),
        "per_image_filter_trace": str(Path(save_dir) / "per_image_filter_trace.csv"),
        "raw_ape_group_summary": str(Path(save_dir) / "raw_ape_group_summary.csv"),
        "point_valid_group_summary": str(Path(save_dir) / "point_valid_group_summary.csv"),
    }


def package(label, weights, sweep_dir, test_dir):
    sweep_summary = Path(sweep_dir) / "tusimple_official_sweep_summary.json"
    test_summary = Path(test_dir) / "tusimple_official_summary.json"
    sweep = load_json(sweep_summary)
    test = load_json(test_summary)
    return {
        "label": label,
        "weights": weights,
        "val_sweep_summary": str(sweep_summary),
        "val_best": sweep["best"],
        "test_summary": str(test_summary),
        "test_metrics": test["metrics"],
    }


output = {
    "schema": "gcs_q12_ultrashort_dataref_geom_full_protocol_v1",
    "run_name": run_name,
    "project": project,
    "model": model,
    "data": data,
    "reference_bank": bank,
    "reference_audit": reference_audit,
    "official_val_gt_json": gt_json,
    "selection_split": "official-val",
    "test_usage": "reporting_only_from_each_val_selected_decode",
    "do_not_select_from_test": True,
    "decode": {
        "valid_before_maxdet": valid_before_maxdet.strip().lower() in {"1", "true", "yes", "on"},
        "half": half.strip().lower() in {"1", "true", "yes", "on"},
    },
    "training_contract": {
        "imgsz": [544, 960],
        "batch": int(batch),
        "epochs": int(epochs),
        "workers": int(workers),
        "scale": 0.15,
        "erasing": 0.10,
        "mosaic": 0.0,
        "official_interval": int(official_interval),
        "official_warmup": int(official_warmup),
    },
    "loss_contract": {
        "gcs_short_geom": float(short_geom),
        "gcs_short_geom_visible_thr": int(short_geom_visible_thr),
        "gcs_short_geom_gt4_weight": float(short_geom_gt4_weight),
        "gcs_short_geom_gt5_weight": float(short_geom_gt5_weight),
        "gcs_short_geom_max_weight": float(short_geom_max_weight),
        "gcs_short_geom_curve": float(short_geom_curve),
        "gcs_short_geom_tiered": False,
        "gcs_gt4_short_visible_thr": int(gt4_short_visible_thr),
        "gcs_gt4_short_point_valid_weight": float(gt4_short_point_valid_weight),
        "gcs_gt5_short_visible_thr": int(gt5_short_visible_thr),
        "gcs_gt5_short_point_valid_weight": float(gt5_short_point_valid_weight),
        "gcs_count": 0.0,
        "gcs_count_under5": 0.0,
        "gcs_count_boundary": 0.0,
        "gcs_query_count_ce": 0.0,
    },
    "official_val_grid": {
        "confs": [0.001, 0.003, 0.005, 0.008, 0.01, 0.02],
        "point_valid_thrs": [0.45, 0.50, 0.55, 0.60],
        "nms_dist_pxs": [0, 18, 30],
        "max_dets": [5, 6, 8],
        "min_points": [2, 3, 4, 5],
        "count_modes": ["score_sum"],
        "rows": 864,
    },
    "official_best": package(
        "official_best.pt",
        official_best_weights,
        official_best_sweep_dir,
        official_best_test_dir,
    ),
    "best": package("best.pt", best_weights, best_sweep_dir, best_test_dir),
    "diagnostics": {
        "official_best_val": diagnostic_package(official_best_val_diag_dir),
        "best_val": diagnostic_package(best_val_diag_dir),
        "official_best_test": diagnostic_package(official_best_test_diag_dir),
        "best_test": diagnostic_package(best_test_diag_dir),
    },
}

Path(protocol_summary).parent.mkdir(parents=True, exist_ok=True)
Path(protocol_summary).write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps({
    "summary": protocol_summary,
    "official_best_test_metrics": output["official_best"]["test_metrics"],
    "best_test_metrics": output["best"]["test_metrics"],
}, indent=2))
PY
}

echo "Run name: ${RUN_NAME}"
echo "Model: ${MODEL}"
echo "Reference bank: ${BANK}"
echo "Reference audit: ${REFERENCE_AUDIT}"
echo "Geometry: short_geom=${SHORT_GEOM} visible_thr=${SHORT_GEOM_VISIBLE_THR} gt4_weight=${SHORT_GEOM_GT4_WEIGHT} gt5_weight=${SHORT_GEOM_GT5_WEIGHT} max_weight=${SHORT_GEOM_MAX_WEIGHT} curve=${SHORT_GEOM_CURVE} tiered=${SHORT_GEOM_TIERED}"
echo "Point-valid: gt4=${GT4_SHORT_VISIBLE_THR}/${GT4_SHORT_POINT_VALID_WEIGHT} gt5=${GT5_SHORT_VISIBLE_THR}/${GT5_SHORT_POINT_VALID_WEIGHT}"
echo "Official-val grid: 864 rows; valid_before_maxdet=${VALID_BEFORE_MAXDET}"
echo "RUN_TRAIN=${RUN_TRAIN} RUN_TESTS=${RUN_TESTS} RUN_DIAGNOSTICS=${RUN_DIAGNOSTICS} RUN_TEST_DIAGNOSTICS=${RUN_TEST_DIAGNOSTICS}"

if is_true "${RUN_TRAIN}"; then
  run_train
else
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "RUN_TRAIN=0 but run directory does not exist: ${RUN_DIR}" >&2
    exit 2
  fi
  echo "RUN_TRAIN=0: reusing existing run directory ${RUN_DIR}" >&2
fi

run_val_sweep "official_best.pt" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}"
run_val_sweep "best.pt" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}"

if is_true "${RUN_DIAGNOSTICS}"; then
  run_raw_q12_diag_from_sweep \
    "official_best.pt val" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" \
    "val" "${OFFICIAL_BEST_VAL_DIAG_DIR}" "${VAL_MAX_IMAGES}"
  run_raw_q12_diag_from_sweep \
    "best.pt val" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}" \
    "val" "${BEST_VAL_DIAG_DIR}" "${VAL_MAX_IMAGES}"
else
  echo "RUN_DIAGNOSTICS=0: skipping validation raw-Q12 diagnostics." >&2
fi

if is_true "${RUN_TESTS}"; then
  run_test_from_sweep \
    "official_best.pt" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" \
    "${OFFICIAL_BEST_TEST_DIR}"
  run_test_from_sweep \
    "best.pt" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}" \
    "${BEST_TEST_DIR}"

  if is_true "${RUN_TEST_DIAGNOSTICS}"; then
    run_raw_q12_diag_from_sweep \
      "official_best.pt test" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" \
      "test" "${OFFICIAL_BEST_TEST_DIAG_DIR}" "${TEST_MAX_IMAGES}"
    run_raw_q12_diag_from_sweep \
      "best.pt test" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}" \
      "test" "${BEST_TEST_DIAG_DIR}" "${TEST_MAX_IMAGES}"
  else
    echo "RUN_TEST_DIAGNOSTICS=0: skipping test raw-Q12 diagnostics." >&2
  fi

  write_protocol_summary
else
  echo "RUN_TESTS=0: skipping official TEST and protocol summary." >&2
  echo "TEST remains reporting-only; set RUN_TESTS=1 only after official-val selection is complete." >&2
fi
