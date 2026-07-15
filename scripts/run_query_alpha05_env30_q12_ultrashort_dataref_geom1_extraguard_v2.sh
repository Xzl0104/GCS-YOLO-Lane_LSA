#!/usr/bin/env bash
set -euo pipefail

# Q12 ultrashort dataref v2 + env30 boundary mask + short geometry + small train-only final-extra guard.
# TEST is gated by official-val/train final-query extra diagnostics and is off by default.

RUN_NAME="${RUN_NAME:-query_alpha05_env30_q12_ultrashort_dataref_geom1_extraguard_v2}"
PROJECT="${PROJECT:-runs/gcs_lane}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-ultrashort-dataref-v2.yaml}"
DATA="${DATA:-data/tusimple_gcs_fixed_y_960x544.yaml}"
PRETRAINED="${PRETRAINED:-yolo11s-seg.pt}"
BANK="${BANK:-data/gcs_reference_banks/q12_ultrashort_env30_train_v2.json}"
REFERENCE_AUDIT="${REFERENCE_AUDIT:-data/gcs_reference_banks/q12_ultrashort_env30_train_v2_reference_audit.json}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
TRAIN0601_GT_JSON="${TRAIN0601_GT_JSON:-${ARCHIVE_ROOT}/train_set/label_data_0601.json}"
TRAIN0531_GT_JSON="${TRAIN0531_GT_JSON:-${ARCHIVE_ROOT}/train_set/label_data_0531.json}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-220}"
WORKERS="${WORKERS:-4}"
OFFICIAL_INTERVAL="${OFFICIAL_INTERVAL:-5}"
OFFICIAL_WARMUP="${OFFICIAL_WARMUP:-5}"
OFFICIAL_MAX_IMAGES="${OFFICIAL_MAX_IMAGES:-0}"
VAL_MAX_IMAGES="${VAL_MAX_IMAGES:-0}"
TRAIN_MAX_IMAGES="${TRAIN_MAX_IMAGES:-0}"
TEST_MAX_IMAGES="${TEST_MAX_IMAGES:-0}"
SWEEP_WARMUP="${SWEEP_WARMUP:-20}"
EXTRA_DIAG_WARMUP="${EXTRA_DIAG_WARMUP:-20}"
TEST_WARMUP="${TEST_WARMUP:-20}"
HALF="${HALF:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TESTS="${RUN_TESTS:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
OVERWRITE_SWEEPS="${OVERWRITE_SWEEPS:-0}"
OVERWRITE_EXTRA_DIAGS="${OVERWRITE_EXTRA_DIAGS:-0}"
OVERWRITE_TESTS="${OVERWRITE_TESTS:-0}"
ENFORCE_REFERENCE_GATE="${ENFORCE_REFERENCE_GATE:-1}"
ENFORCE_FINAL_QUERY_EXTRA_GATE="${ENFORCE_FINAL_QUERY_EXTRA_GATE:-1}"
VALID_BEFORE_MAXDET="${VALID_BEFORE_MAXDET:-1}"

# Env30 official-val selected baseline used only for final-query extra comparison.
ENV30_WEIGHTS="${ENV30_WEIGHTS:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt}"
ENV30_DECODE_YAML="${ENV30_DECODE_YAML:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best_decode.yaml}"
RUN_ENV30_EXTRA_DIAGS="${RUN_ENV30_EXTRA_DIAGS:-1}"

SHORT_GEOM="${SHORT_GEOM:-1.0}"
SHORT_GEOM_VISIBLE_THR="${SHORT_GEOM_VISIBLE_THR:-10}"
SHORT_GEOM_GT4_WEIGHT="${SHORT_GEOM_GT4_WEIGHT:-1.0}"
SHORT_GEOM_GT5_WEIGHT="${SHORT_GEOM_GT5_WEIGHT:-2.0}"
SHORT_GEOM_MAX_WEIGHT="${SHORT_GEOM_MAX_WEIGHT:-3.0}"
SHORT_GEOM_CURVE="${SHORT_GEOM_CURVE:-1.0}"
GT4_SHORT_VISIBLE_THR="${GT4_SHORT_VISIBLE_THR:-0}"
GT4_SHORT_POINT_VALID_WEIGHT="${GT4_SHORT_POINT_VALID_WEIGHT:-1.0}"
GT5_SHORT_VISIBLE_THR="${GT5_SHORT_VISIBLE_THR:-0}"
GT5_SHORT_POINT_VALID_WEIGHT="${GT5_SHORT_POINT_VALID_WEIGHT:-1.0}"

BOUNDARY_PSEUDO_NEG="${BOUNDARY_PSEUDO_NEG:-0.02}"
BOUNDARY_PSEUDO_VISIBLE_THR="${BOUNDARY_PSEUDO_VISIBLE_THR:-10}"
BOUNDARY_PSEUDO_DIST_THR="${BOUNDARY_PSEUDO_DIST_THR:-80}"
BOUNDARY_PSEUDO_VALID_THR="${BOUNDARY_PSEUDO_VALID_THR:-0.5}"
BOUNDARY_PSEUDO_MIN_VALID="${BOUNDARY_PSEUDO_MIN_VALID:-4}"
BOUNDARY_PSEUDO_GT_COUNT="${BOUNDARY_PSEUDO_GT_COUNT:-5}"
BOUNDARY_PSEUDO_SCORE_THR="${BOUNDARY_PSEUDO_SCORE_THR:-0.2}"
BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX="${BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX:-30}"
BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR="${BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR:-0.75}"

FINAL_EXTRA_GUARD="${FINAL_EXTRA_GUARD:-0.02}"
FINAL_EXTRA_GUARD_SCOPE="${FINAL_EXTRA_GUARD_SCOPE:-1,3,4,5,6,7,8}"
FINAL_EXTRA_GUARD_SCORE_THR="${FINAL_EXTRA_GUARD_SCORE_THR:-0.15}"
FINAL_EXTRA_GUARD_VALID_THR="${FINAL_EXTRA_GUARD_VALID_THR:-0.55}"
FINAL_EXTRA_GUARD_MIN_VALID="${FINAL_EXTRA_GUARD_MIN_VALID:-2}"
FINAL_EXTRA_GUARD_MIN_OVERLAP="${FINAL_EXTRA_GUARD_MIN_OVERLAP:-3}"
FINAL_EXTRA_GUARD_CLEAR_FAR_PX="${FINAL_EXTRA_GUARD_CLEAR_FAR_PX:-50}"
FINAL_EXTRA_GUARD_DUPLICATE_PX="${FINAL_EXTRA_GUARD_DUPLICATE_PX:-30}"
FINAL_EXTRA_GUARD_PROTECT_PX="${FINAL_EXTRA_GUARD_PROTECT_PX:-30}"
FINAL_EXTRA_GUARD_PROTECT_ACC="${FINAL_EXTRA_GUARD_PROTECT_ACC:-0.85}"

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

for required_path in "${MODEL}" "${DATA}" "${BANK}" "${GT_JSON}" "${TRAIN0601_GT_JSON}" "${TRAIN0531_GT_JSON}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 2
  fi
done
if [[ ! -d "${ARCHIVE_ROOT}" ]]; then
  echo "Missing TuSimple archive root: ${ARCHIVE_ROOT}" >&2
  exit 2
fi

validate_reference_gate() {
  if ! is_true "${ENFORCE_REFERENCE_GATE}"; then
    echo "ENFORCE_REFERENCE_GATE=0: bypassing static reference gate. Use only for debugging." >&2
    return
  fi
  python - "${REFERENCE_AUDIT}" "${BANK}" "${GT_JSON}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
bank_path = Path(sys.argv[2])
expected_gt_json = str(sys.argv[3]).replace("\\", "/")
if not audit_path.exists():
    raise SystemExit(f"Missing static reference audit: {audit_path}")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
bank = json.loads(bank_path.read_text(encoding="utf-8"))
if audit.get("schema") != "gcs_q12_fixed_y_reference_bank_coverage_audit_v1":
    raise SystemExit(f"Unexpected static audit schema: {audit.get('schema')!r}")
if bank.get("schema") != "gcs_q12_fixed_y_reference_bank_v1":
    raise SystemExit(f"Unexpected bank schema: {bank.get('schema')!r}")
if int(bank.get("num_queries", -1)) != 12 or int(bank.get("num_points", -1)) != 56:
    raise SystemExit("Reference bank must declare num_queries=12 and num_points=56")
actual_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest()
if audit.get("bank_sha256") != actual_hash:
    raise SystemExit(f"Static audit hash mismatch. audit={audit.get('bank_sha256')!r} bank={actual_hash!r}")
audit_gt_json = [str(p).replace("\\", "/") for p in audit.get("gt_json", [])]
if audit_gt_json != [expected_gt_json]:
    raise SystemExit(f"Static audit GT mismatch. expected={[expected_gt_json]!r} actual={audit_gt_json!r}")
if not bool(audit.get("gate", {}).get("pass", False)):
    raise SystemExit(f"Static reference gate failed: {audit.get('gate')}")
print(f"Static reference gate passed: {audit_path}")
PY
}

validate_loss_contract() {
  python - \
    "${SHORT_GEOM}" "${SHORT_GEOM_VISIBLE_THR}" "${SHORT_GEOM_GT4_WEIGHT}" \
    "${SHORT_GEOM_GT5_WEIGHT}" "${SHORT_GEOM_MAX_WEIGHT}" "${SHORT_GEOM_CURVE}" \
    "${GT4_SHORT_VISIBLE_THR}" "${GT4_SHORT_POINT_VALID_WEIGHT}" \
    "${GT5_SHORT_VISIBLE_THR}" "${GT5_SHORT_POINT_VALID_WEIGHT}" \
    "${BOUNDARY_PSEUDO_NEG}" "${BOUNDARY_PSEUDO_VISIBLE_THR}" \
    "${BOUNDARY_PSEUDO_DIST_THR}" "${BOUNDARY_PSEUDO_VALID_THR}" \
    "${BOUNDARY_PSEUDO_MIN_VALID}" "${BOUNDARY_PSEUDO_GT_COUNT}" \
    "${BOUNDARY_PSEUDO_SCORE_THR}" "${BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX}" \
    "${BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR}" "${FINAL_EXTRA_GUARD}" \
    "${FINAL_EXTRA_GUARD_SCOPE}" "${FINAL_EXTRA_GUARD_SCORE_THR}" \
    "${FINAL_EXTRA_GUARD_VALID_THR}" "${FINAL_EXTRA_GUARD_MIN_VALID}" \
    "${FINAL_EXTRA_GUARD_MIN_OVERLAP}" \
    "${FINAL_EXTRA_GUARD_CLEAR_FAR_PX}" "${FINAL_EXTRA_GUARD_DUPLICATE_PX}" \
    "${FINAL_EXTRA_GUARD_PROTECT_PX}" "${FINAL_EXTRA_GUARD_PROTECT_ACC}" <<'PY'
from decimal import Decimal
import sys

(
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
    boundary_pseudo_neg,
    boundary_pseudo_visible_thr,
    boundary_pseudo_dist_thr,
    boundary_pseudo_valid_thr,
    boundary_pseudo_min_valid,
    boundary_pseudo_gt_count,
    boundary_pseudo_score_thr,
    boundary_pseudo_margin,
    boundary_pseudo_ratio,
    final_extra_guard,
    final_extra_guard_scope,
    final_extra_guard_score_thr,
    final_extra_guard_valid_thr,
    final_extra_guard_min_valid,
    final_extra_guard_min_overlap,
    final_extra_guard_clear_far_px,
    final_extra_guard_duplicate_px,
    final_extra_guard_protect_px,
    final_extra_guard_protect_acc,
) = sys.argv[1:]

def require_decimal(name, actual, expected):
    if Decimal(actual) != Decimal(expected):
        raise SystemExit(f"{name} must be {expected}, got {actual}")

require_decimal("SHORT_GEOM", short_geom, "1.0")
if int(short_geom_visible_thr) != 10:
    raise SystemExit(f"SHORT_GEOM_VISIBLE_THR must be 10, got {short_geom_visible_thr}")
require_decimal("SHORT_GEOM_GT4_WEIGHT", short_geom_gt4_weight, "1.0")
require_decimal("SHORT_GEOM_GT5_WEIGHT", short_geom_gt5_weight, "2.0")
require_decimal("SHORT_GEOM_MAX_WEIGHT", short_geom_max_weight, "3.0")
require_decimal("SHORT_GEOM_CURVE", short_geom_curve, "1.0")
if int(gt4_short_visible_thr) != 0 or Decimal(gt4_short_point_valid_weight) != Decimal("1.0"):
    raise SystemExit("GT4 point-valid rescue must stay disabled for v2")
if int(gt5_short_visible_thr) != 0 or Decimal(gt5_short_point_valid_weight) != Decimal("1.0"):
    raise SystemExit("GT5 point-valid rescue must stay disabled for v2")
require_decimal("BOUNDARY_PSEUDO_NEG", boundary_pseudo_neg, "0.02")
if int(boundary_pseudo_visible_thr) != 10:
    raise SystemExit(f"BOUNDARY_PSEUDO_VISIBLE_THR must be 10, got {boundary_pseudo_visible_thr}")
require_decimal("BOUNDARY_PSEUDO_DIST_THR", boundary_pseudo_dist_thr, "80")
require_decimal("BOUNDARY_PSEUDO_VALID_THR", boundary_pseudo_valid_thr, "0.5")
if int(boundary_pseudo_min_valid) != 4:
    raise SystemExit(f"BOUNDARY_PSEUDO_MIN_VALID must be 4, got {boundary_pseudo_min_valid}")
if int(boundary_pseudo_gt_count) != 5:
    raise SystemExit(f"BOUNDARY_PSEUDO_GT_COUNT must be 5, got {boundary_pseudo_gt_count}")
require_decimal("BOUNDARY_PSEUDO_SCORE_THR", boundary_pseudo_score_thr, "0.2")
require_decimal("BOUNDARY_PSEUDO_ENVELOPE_MARGIN_PX", boundary_pseudo_margin, "30")
require_decimal("BOUNDARY_PSEUDO_ENVELOPE_RATIO_THR", boundary_pseudo_ratio, "0.75")
require_decimal("FINAL_EXTRA_GUARD", final_extra_guard, "0.02")
if final_extra_guard_scope != "1,3,4,5,6,7,8":
    raise SystemExit(f"FINAL_EXTRA_GUARD_SCOPE must be 1,3,4,5,6,7,8, got {final_extra_guard_scope}")
require_decimal("FINAL_EXTRA_GUARD_SCORE_THR", final_extra_guard_score_thr, "0.15")
require_decimal("FINAL_EXTRA_GUARD_VALID_THR", final_extra_guard_valid_thr, "0.55")
if int(final_extra_guard_min_valid) != 2:
    raise SystemExit(f"FINAL_EXTRA_GUARD_MIN_VALID must be 2, got {final_extra_guard_min_valid}")
if int(final_extra_guard_min_overlap) != 3:
    raise SystemExit(f"FINAL_EXTRA_GUARD_MIN_OVERLAP must be 3, got {final_extra_guard_min_overlap}")
require_decimal("FINAL_EXTRA_GUARD_CLEAR_FAR_PX", final_extra_guard_clear_far_px, "50")
require_decimal("FINAL_EXTRA_GUARD_DUPLICATE_PX", final_extra_guard_duplicate_px, "30")
require_decimal("FINAL_EXTRA_GUARD_PROTECT_PX", final_extra_guard_protect_px, "30")
require_decimal("FINAL_EXTRA_GUARD_PROTECT_ACC", final_extra_guard_protect_acc, "0.85")
print("v2 loss contract passed.")
PY
}

validate_reference_gate
validate_loss_contract

if is_true "${PREFLIGHT_ONLY}"; then
  echo "PREFLIGHT_ONLY=1: validation completed; no training/sweep/diagnostic/TEST started."
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
HALF_ARGS=()
if is_true "${HALF}"; then
  OFFICIAL_HALF_ARGS=(--gcs-official-half)
  HALF_ARGS=(--half)
fi

OFFICIAL_BEST_SWEEP_DIR="${OFFICIAL_BEST_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_official_best_val_sweep_${DECODE_TAG}}"
BEST_SWEEP_DIR="${BEST_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_best_val_sweep_${DECODE_TAG}}"
OFFICIAL_BEST_VAL_EXTRA_DIR="${OFFICIAL_BEST_VAL_EXTRA_DIR:-${PROJECT}/${RUN_NAME}_official_best_val_final_query_extra_${DECODE_TAG}}"
BEST_VAL_EXTRA_DIR="${BEST_VAL_EXTRA_DIR:-${PROJECT}/${RUN_NAME}_best_val_final_query_extra_${DECODE_TAG}}"
OFFICIAL_BEST_TRAIN0601_EXTRA_DIR="${OFFICIAL_BEST_TRAIN0601_EXTRA_DIR:-${PROJECT}/${RUN_NAME}_official_best_train0601_final_query_extra_${DECODE_TAG}}"
OFFICIAL_BEST_TRAIN0531_EXTRA_DIR="${OFFICIAL_BEST_TRAIN0531_EXTRA_DIR:-${PROJECT}/${RUN_NAME}_official_best_train0531_final_query_extra_${DECODE_TAG}}"
ENV30_VAL_EXTRA_DIR="${ENV30_VAL_EXTRA_DIR:-${PROJECT}/env30_official_best_val_final_query_extra_${DECODE_TAG}}"
ENV30_TRAIN0601_EXTRA_DIR="${ENV30_TRAIN0601_EXTRA_DIR:-${PROJECT}/env30_official_best_train0601_final_query_extra_${DECODE_TAG}}"
ENV30_TRAIN0531_EXTRA_DIR="${ENV30_TRAIN0531_EXTRA_DIR:-${PROJECT}/env30_official_best_train0531_final_query_extra_${DECODE_TAG}}"
FINAL_QUERY_EXTRA_GATE_JSON="${FINAL_QUERY_EXTRA_GATE_JSON:-${PROJECT}/${RUN_NAME}_final_query_extra_gate.json}"
OFFICIAL_BEST_TEST_DIR="${OFFICIAL_BEST_TEST_DIR:-${PROJECT}/${RUN_NAME}_official_best_test_from_val_sweep_${DECODE_TAG}}"
BEST_TEST_DIR="${BEST_TEST_DIR:-${PROJECT}/${RUN_NAME}_best_test_from_val_sweep_${DECODE_TAG}}"

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
    --gcs-final-extra-guard "${FINAL_EXTRA_GUARD}" \
    --gcs-final-extra-guard-scope "${FINAL_EXTRA_GUARD_SCOPE}" \
    --gcs-final-extra-guard-score-thr "${FINAL_EXTRA_GUARD_SCORE_THR}" \
    --gcs-final-extra-guard-valid-thr "${FINAL_EXTRA_GUARD_VALID_THR}" \
    --gcs-final-extra-guard-min-valid "${FINAL_EXTRA_GUARD_MIN_VALID}" \
    --gcs-final-extra-guard-min-overlap "${FINAL_EXTRA_GUARD_MIN_OVERLAP}" \
    --gcs-final-extra-guard-clear-far-px "${FINAL_EXTRA_GUARD_CLEAR_FAR_PX}" \
    --gcs-final-extra-guard-duplicate-px "${FINAL_EXTRA_GUARD_DUPLICATE_PX}" \
    --gcs-final-extra-guard-protect-px "${FINAL_EXTRA_GUARD_PROTECT_PX}" \
    --gcs-final-extra-guard-protect-acc "${FINAL_EXTRA_GUARD_PROTECT_ACC}" \
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
    "${HALF_ARGS[@]}" \
    --confs "${SWEEP_CONFS_ARR[@]}" \
    --point-valid-thrs "${SWEEP_POINT_VALID_THRS_ARR[@]}" \
    --nms-dist-pxs "${SWEEP_NMS_DIST_PXS_ARR[@]}" \
    --max-dets "${SWEEP_MAX_DETS_ARR[@]}" \
    --min-points "${SWEEP_MIN_POINTS_ARR[@]}" \
    --count-modes "${SWEEP_COUNT_MODES_ARR[@]}" \
    "${SWEEP_VALID_ARGS[@]}" \
    --save-dir "${save_dir}"
}

run_extra_diag_from_sweep() {
  local label="$1"
  local weights="$2"
  local sweep_dir="$3"
  local split="$4"
  local gt_json="$5"
  local save_dir="$6"
  local max_images="$7"
  local summary="${sweep_dir}/tusimple_official_sweep_summary.json"
  if [[ ! -f "${summary}" ]]; then
    echo "Missing ${label} sweep summary: ${summary}" >&2
    exit 2
  fi
  if [[ -e "${save_dir}" ]] && ! is_true "${OVERWRITE_EXTRA_DIAGS}"; then
    echo "Final-query extra diagnostic directory already exists: ${save_dir}" >&2
    echo "Set OVERWRITE_EXTRA_DIAGS=1 or choose a new directory." >&2
    exit 2
  fi
  python - "${summary}" "${weights}" "${split}" "${gt_json}" "${save_dir}" "${max_images}" "${EXTRA_DIAG_WARMUP}" "${HALF}" "${DEVICE}" "${ARCHIVE_ROOT}" <<'PY'
import json
import shlex
import subprocess
import sys

summary_path, weights, split, gt_json, save_dir, max_images, warmup, half, device, archive_root = sys.argv[1:]
best = json.load(open(summary_path, encoding="utf-8"))["best"]
cmd = [
    sys.executable,
    "tools/diagnose_tusimple_final_query_extra_lanes.py",
    "--archive-root", archive_root,
    "--split", split,
    "--gt-json", gt_json,
    "--weights", weights,
    "--imgsz", "544", "960",
    "--device", device,
    "--conf", str(best["conf"]),
    "--point-valid-thr", str(best["point_valid_thr"]),
    "--nms-dist-px", str(best["nms_dist_px"]),
    "--max-det", str(int(best["max_det"])),
    "--min-points", str(int(best["min_points"])),
    "--count-mode", str(best.get("count_mode", "score_sum")),
    "--warmup", str(int(max(0, int(warmup)))),
    "--focus-queries", "1,3,4,5,6,8",
    "--save-dir", save_dir,
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
print("[final-query-extra] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
subprocess.run(cmd, check=True)
PY
}

run_extra_diag_from_decode_yaml() {
  local label="$1"
  local weights="$2"
  local decode_yaml="$3"
  local split="$4"
  local gt_json="$5"
  local save_dir="$6"
  local max_images="$7"
  if [[ ! -f "${weights}" || ! -f "${decode_yaml}" ]]; then
    echo "Missing ${label} baseline weights/decode yaml: ${weights} ${decode_yaml}" >&2
    exit 2
  fi
  if [[ -e "${save_dir}" ]] && ! is_true "${OVERWRITE_EXTRA_DIAGS}"; then
    echo "Final-query extra diagnostic directory already exists: ${save_dir}" >&2
    echo "Set OVERWRITE_EXTRA_DIAGS=1 or choose a new directory." >&2
    exit 2
  fi
  python tools/diagnose_tusimple_final_query_extra_lanes.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split "${split}" \
    --gt-json "${gt_json}" \
    --weights "${weights}" \
    --decode-yaml "${decode_yaml}" \
    --imgsz 544 960 \
    --device "${DEVICE}" \
    --warmup "${EXTRA_DIAG_WARMUP}" \
    "${HALF_ARGS[@]}" \
    --focus-queries 1,3,4,5,6,8 \
    --max-images "${max_images}" \
    --save-dir "${save_dir}"
}

run_final_query_extra_gate() {
  if ! is_true "${ENFORCE_FINAL_QUERY_EXTRA_GATE}"; then
    echo "ENFORCE_FINAL_QUERY_EXTRA_GATE=0: skipping final-query extra gate. TEST remains unsafe for selection." >&2
    return
  fi
  python tools/check_gcs_final_query_extra_gate.py \
    --candidate-val-summary "${OFFICIAL_BEST_VAL_EXTRA_DIR}/final_query_extra_summary.json" \
    --baseline-val-summary "${ENV30_VAL_EXTRA_DIR}/final_query_extra_summary.json" \
    --candidate-train-summary "train0601=${OFFICIAL_BEST_TRAIN0601_EXTRA_DIR}/final_query_extra_summary.json" \
    --baseline-train-summary "train0601=${ENV30_TRAIN0601_EXTRA_DIR}/final_query_extra_summary.json" \
    --candidate-train-summary "train0531=${OFFICIAL_BEST_TRAIN0531_EXTRA_DIR}/final_query_extra_summary.json" \
    --baseline-train-summary "train0531=${ENV30_TRAIN0531_EXTRA_DIR}/final_query_extra_summary.json" \
    --save-json "${FINAL_QUERY_EXTRA_GATE_JSON}"
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
    echo "TEST directory already exists: ${save_dir}" >&2
    echo "Set OVERWRITE_TESTS=1 or choose a new TEST directory." >&2
    exit 2
  fi
  python - "${summary}" "${weights}" "${save_dir}" "${ARCHIVE_ROOT}" "${DEVICE}" "${TEST_MAX_IMAGES}" "${TEST_WARMUP}" "${HALF}" <<'PY'
import json
import shlex
import subprocess
import sys

summary_path, weights, save_dir, archive_root, device, max_images, warmup, half = sys.argv[1:]
best = json.load(open(summary_path, encoding="utf-8"))["best"]
cmd = [
    sys.executable,
    "tools/eval_tusimple_official.py",
    "--archive-root", archive_root,
    "--split", "test",
    "--weights", weights,
    "--imgsz", "544", "960",
    "--device", device,
    "--conf", str(best["conf"]),
    "--point-valid-thr", str(best["point_valid_thr"]),
    "--nms-dist-px", str(best["nms_dist_px"]),
    "--max-det", str(int(best["max_det"])),
    "--min-points", str(int(best["min_points"])),
    "--count-mode", str(best.get("count_mode", "score_sum")),
    "--warmup", str(int(max(0, int(warmup)))),
    "--save-dir", save_dir,
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
print("[reporting-only test] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
subprocess.run(cmd, check=True)
PY
}

echo "Run name: ${RUN_NAME}"
echo "Model: ${MODEL}"
echo "Reference bank: ${BANK}"
echo "RUN_TRAIN=${RUN_TRAIN} RUN_TESTS=${RUN_TESTS} ENFORCE_FINAL_QUERY_EXTRA_GATE=${ENFORCE_FINAL_QUERY_EXTRA_GATE}"

if is_true "${RUN_TRAIN}"; then
  run_train
else
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "RUN_TRAIN=0 but run directory does not exist: ${RUN_DIR}" >&2
    exit 2
  fi
  echo "RUN_TRAIN=0: reusing ${RUN_DIR}" >&2
fi

run_val_sweep "official_best.pt" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}"
run_val_sweep "best.pt" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}"

run_extra_diag_from_sweep \
  "official_best.pt val" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" \
  "val" "${GT_JSON}" "${OFFICIAL_BEST_VAL_EXTRA_DIR}" "${VAL_MAX_IMAGES}"
run_extra_diag_from_sweep \
  "best.pt val" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}" \
  "val" "${GT_JSON}" "${BEST_VAL_EXTRA_DIR}" "${VAL_MAX_IMAGES}"
run_extra_diag_from_sweep \
  "official_best.pt train0601" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" \
  "train" "${TRAIN0601_GT_JSON}" "${OFFICIAL_BEST_TRAIN0601_EXTRA_DIR}" "${TRAIN_MAX_IMAGES}"
run_extra_diag_from_sweep \
  "official_best.pt train0531" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" \
  "train" "${TRAIN0531_GT_JSON}" "${OFFICIAL_BEST_TRAIN0531_EXTRA_DIR}" "${TRAIN_MAX_IMAGES}"

if is_true "${RUN_ENV30_EXTRA_DIAGS}"; then
  run_extra_diag_from_decode_yaml \
    "env30 val" "${ENV30_WEIGHTS}" "${ENV30_DECODE_YAML}" \
    "val" "${GT_JSON}" "${ENV30_VAL_EXTRA_DIR}" "${VAL_MAX_IMAGES}"
  run_extra_diag_from_decode_yaml \
    "env30 train0601" "${ENV30_WEIGHTS}" "${ENV30_DECODE_YAML}" \
    "train" "${TRAIN0601_GT_JSON}" "${ENV30_TRAIN0601_EXTRA_DIR}" "${TRAIN_MAX_IMAGES}"
  run_extra_diag_from_decode_yaml \
    "env30 train0531" "${ENV30_WEIGHTS}" "${ENV30_DECODE_YAML}" \
    "train" "${TRAIN0531_GT_JSON}" "${ENV30_TRAIN0531_EXTRA_DIR}" "${TRAIN_MAX_IMAGES}"
fi

run_final_query_extra_gate

if is_true "${RUN_TESTS}"; then
  echo "Final-query extra gate passed; running reporting-only TEST."
  run_test_from_sweep "official_best.pt" "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" "${OFFICIAL_BEST_TEST_DIR}"
  run_test_from_sweep "best.pt" "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}" "${BEST_TEST_DIR}"
else
  echo "RUN_TESTS=0: gate completed; official TEST remains closed. Set RUN_TESTS=1 only after reviewing the gate."
fi
