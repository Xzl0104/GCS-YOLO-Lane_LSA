#!/usr/bin/env bash
set -euo pipefail

# End-to-end Q12/env30 query extent protocol.
# Selection stays official-val only. TEST is reporting-only and runs only when
# RUN_TESTS=1 is explicitly set by the operator.

export RUN_NAME="${RUN_NAME:-query_extent_env30_probe40_v1}"
export PROJECT="${PROJECT:-runs/gcs_lane}"
REQUIRED_MODEL="ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-extent.yaml"
export MODEL="${MODEL:-${REQUIRED_MODEL}}"
export EPOCHS="${EPOCHS:-40}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export VALID_BEFORE_MAXDET="${VALID_BEFORE_MAXDET:-1}"
export OFFICIAL_EXTENT_DECODE_MODES="${OFFICIAL_EXTENT_DECODE_MODES:-none interval intersect}"
export SWEEP_EXTENT_DECODE_MODES="${SWEEP_EXTENT_DECODE_MODES:-${OFFICIAL_EXTENT_DECODE_MODES}}"
export OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"
export SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-${OFFICIAL_COUNT_MODES}}"
export COUNT_AWARE_TOPK="${COUNT_AWARE_TOPK:-0}"
export SWEEP_COUNT_AWARE_TOPK="${SWEEP_COUNT_AWARE_TOPK:-${COUNT_AWARE_TOPK}}"
export COUNT_AWARE_EXTRA_MARGINS="${COUNT_AWARE_EXTRA_MARGINS:-0}"
export SWEEP_COUNT_AWARE_EXTRA_MARGINS="${SWEEP_COUNT_AWARE_EXTRA_MARGINS:-${COUNT_AWARE_EXTRA_MARGINS}}"
export GCS_QUERY_EXTENT="${GCS_QUERY_EXTENT:-0.5}"
export GCS_QUERY_EXTENT_SHORT_VISIBLE_THR="${GCS_QUERY_EXTENT_SHORT_VISIBLE_THR:-10}"
export GCS_QUERY_EXTENT_SHORT_WEIGHT="${GCS_QUERY_EXTENT_SHORT_WEIGHT:-2.0}"
export GCS_QUERY_EXTENT_GT_MIN_LANES="${GCS_QUERY_EXTENT_GT_MIN_LANES:-4}"
export GCS_BOUNDARY_PSEUDO_NEG="${GCS_BOUNDARY_PSEUDO_NEG:-0.0}"

ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-1}"
TEST_MAX_IMAGES="${TEST_MAX_IMAGES:-0}"
VAL_DIAG_MAX_IMAGES="${VAL_DIAG_MAX_IMAGES:-0}"
TEST_DIAG_MAX_IMAGES="${TEST_DIAG_MAX_IMAGES:-${TEST_MAX_IMAGES}}"
TEST_WARMUP="${TEST_WARMUP:-20}"
DIAG_WARMUP="${DIAG_WARMUP:-20}"
DIAG_EXTENT_DECODE_MODES="${DIAG_EXTENT_DECODE_MODES:-none interval intersect}"
RUN_VAL_DIAGS="${RUN_VAL_DIAGS:-1}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_TEST_DIAGS="${RUN_TEST_DIAGS:-${RUN_TESTS}}"
OVERWRITE_DIAGS="${OVERWRITE_DIAGS:-0}"
OVERWRITE_TESTS="${OVERWRITE_TESTS:-0}"

RUN_DIR="${PROJECT}/${RUN_NAME}"
DECODE_TAG="default_decode"
if [[ "$(printf '%s' "${VALID_BEFORE_MAXDET}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
  DECODE_TAG="valid_before_maxdet"
fi

OFFICIAL_BEST_WEIGHTS="${RUN_DIR}/weights/official_best.pt"
BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"
ARGS_YAML="${RUN_DIR}/args.yaml"
TRAIN_OFFICIAL_BEST_DECODE_YAML="${RUN_DIR}/weights/official_best_decode.yaml"
TRAIN_OFFICIAL_BEST_SWEEP_JSON="${RUN_DIR}/weights/official_best_sweep.json"
OFFICIAL_BEST_SWEEP_DIR="${OFFICIAL_BEST_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_official_best_val_sweep_${DECODE_TAG}}"
BEST_SWEEP_DIR="${BEST_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_best_val_sweep_${DECODE_TAG}}"
OFFICIAL_BEST_POST_DECODE_YAML="${OFFICIAL_BEST_POST_DECODE_YAML:-${OFFICIAL_BEST_SWEEP_DIR}/official_best_posttrain_decode.yaml}"
BEST_POST_DECODE_YAML="${BEST_POST_DECODE_YAML:-${BEST_SWEEP_DIR}/best_posttrain_decode.yaml}"
OFFICIAL_BEST_TEST_DIR="${OFFICIAL_BEST_TEST_DIR:-${PROJECT}/${RUN_NAME}_official_best_test_from_posttrain_val_${DECODE_TAG}}"
BEST_TEST_DIR="${BEST_TEST_DIR:-${PROJECT}/${RUN_NAME}_best_test_from_posttrain_val_${DECODE_TAG}}"
DIAG_DIR="${DIAG_DIR:-${PROJECT}/${RUN_NAME}_official_best_best_diagnostics_${DECODE_TAG}}"
PROTOCOL_SUMMARY="${PROTOCOL_SUMMARY:-${PROJECT}/${RUN_NAME}_full_protocol_summary_${DECODE_TAG}.json}"

read -r -a OFFICIAL_EXTENT_DECODE_MODES_ARR <<< "${OFFICIAL_EXTENT_DECODE_MODES}"
read -r -a SWEEP_EXTENT_DECODE_MODES_ARR <<< "${SWEEP_EXTENT_DECODE_MODES}"
read -r -a OFFICIAL_COUNT_MODES_ARR <<< "${OFFICIAL_COUNT_MODES}"
read -r -a SWEEP_COUNT_MODES_ARR <<< "${SWEEP_COUNT_MODES}"
read -r -a DIAG_EXTENT_DECODE_MODES_ARR <<< "${DIAG_EXTENT_DECODE_MODES}"

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

check_inputs() {
  if [[ ! -f "${GT_JSON}" ]]; then
    echo "Missing canonical official-val GT JSON: ${GT_JSON}" >&2
    exit 2
  fi
  if [[ ! -d "${ARCHIVE_ROOT}" ]]; then
    echo "Missing TuSimple archive root: ${ARCHIVE_ROOT}" >&2
    exit 2
  fi
  if [[ "${MODEL}" != "${REQUIRED_MODEL}" ]]; then
    echo "MODEL must be exactly ${REQUIRED_MODEL}, got: ${MODEL}" >&2
    exit 2
  fi
  if is_true "${COUNT_AWARE_TOPK}" || is_true "${SWEEP_COUNT_AWARE_TOPK}"; then
    echo "COUNT_AWARE_TOPK/SWEEP_COUNT_AWARE_TOPK is enabled. This is not extent-only because ranking uses point-valid quality." >&2
  fi
}

run_training_and_posttrain_sweeps() {
  # The probe script delegates to the env30 parent launcher, which enables
  # training-time official_best and runs post-train val sweeps for official_best.pt and best.pt.
  RUN_TESTS=0 bash scripts/run_query_extent_env30_probe40_v1.sh
}

check_training_contract() {
  if [[ ! -f "${ARGS_YAML}" ]]; then
    echo "Missing training args: ${ARGS_YAML}" >&2
    exit 2
  fi
  if [[ ! -f "${OFFICIAL_BEST_WEIGHTS}" ]]; then
    echo "Missing official_best weights: ${OFFICIAL_BEST_WEIGHTS}" >&2
    exit 2
  fi
  if [[ ! -f "${BEST_WEIGHTS}" ]]; then
    echo "Missing best weights: ${BEST_WEIGHTS}" >&2
    exit 2
  fi
  if [[ ! -f "${TRAIN_OFFICIAL_BEST_DECODE_YAML}" ]]; then
    echo "Missing training-time official_best decode YAML: ${TRAIN_OFFICIAL_BEST_DECODE_YAML}" >&2
    exit 2
  fi
  if [[ ! -f "${TRAIN_OFFICIAL_BEST_SWEEP_JSON}" ]]; then
    echo "Missing training-time official_best sweep JSON: ${TRAIN_OFFICIAL_BEST_SWEEP_JSON}" >&2
    exit 2
  fi
  if ! compgen -G "${RUN_DIR}/official_sweeps/epoch*/prediction_cache/manifest.json" > /dev/null; then
    echo "Missing training-time official_best prediction cache manifest under ${RUN_DIR}/official_sweeps/epoch*/prediction_cache/" >&2
    exit 2
  fi

  python - "${ARGS_YAML}" "${TRAIN_OFFICIAL_BEST_DECODE_YAML}" "${TRAIN_OFFICIAL_BEST_SWEEP_JSON}" \
    "${REQUIRED_MODEL}" \
    "${GCS_QUERY_EXTENT}" "${GCS_QUERY_EXTENT_SHORT_VISIBLE_THR}" \
    "${GCS_QUERY_EXTENT_SHORT_WEIGHT}" "${GCS_QUERY_EXTENT_GT_MIN_LANES}" \
    "${VALID_BEFORE_MAXDET}" "${OFFICIAL_EXTENT_DECODE_MODES}" \
    "${OFFICIAL_COUNT_MODES}" "${COUNT_AWARE_TOPK}" "${COUNT_AWARE_EXTRA_MARGINS}" <<'PY'
import json
import math
import sys

import yaml

(
    args_path,
    decode_path,
    official_best_sweep_path,
    required_model,
    expected_extent,
    expected_short_thr,
    expected_short_weight,
    expected_min_lanes,
    expected_valid_before,
    expected_extent_modes,
    expected_count_modes,
    expected_count_aware,
    expected_extra_margins,
) = sys.argv[1:]

with open(args_path, "r", encoding="utf-8") as f:
    args = yaml.safe_load(f) or {}
with open(decode_path, "r", encoding="utf-8") as f:
    decode = yaml.safe_load(f) or {}
with open(official_best_sweep_path, "r", encoding="utf-8") as f:
    official_best_sweep = json.load(f)

def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def as_list(value: str) -> list[str]:
    return [str(x).strip() for x in str(value).split() if str(x).strip()]

def cfg_list(value) -> list[str]:
    if isinstance(value, str):
        return as_list(value)
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return []

bad = []
float_expected = {
    "gcs_query_extent": float(expected_extent),
    "gcs_query_extent_short_weight": float(expected_short_weight),
    "gcs_short_geom": 1.0,
    "gcs_query_count_ce": 0.0,
    "gcs_query_quality": 0.0,
    "gcs_count": 0.0,
    "gcs_count_under5": 0.0,
    "gcs_count_boundary": 0.0,
    "gcs_boundary_pseudo_neg": 0.0,
    "gcs_role_contain": 0.0,
    "gcs_q24_event_contain": 0.0,
    "gcs_q24_event_score_calib": 0.0,
}
for key, expected in float_expected.items():
    try:
        actual = float(args.get(key))
    except (TypeError, ValueError):
        bad.append(f"{key}={args.get(key)!r}")
        continue
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        bad.append(f"{key}={actual!r}, expected {expected!r}")

int_expected = {
    "gcs_query_extent_short_visible_thr": int(expected_short_thr),
    "gcs_query_extent_gt_min_lanes": int(expected_min_lanes),
}
for key, expected in int_expected.items():
    try:
        actual = int(args.get(key))
    except (TypeError, ValueError):
        bad.append(f"{key}={args.get(key)!r}")
        continue
    if actual != expected:
        bad.append(f"{key}={actual!r}, expected {expected!r}")

if str(args.get("model", "")) != required_model:
    bad.append(f"model={args.get('model')!r}")
if not bool(args.get("gcs_official_best", False)):
    bad.append("gcs_official_best is not true")
if bool(args.get("gcs_official_valid_before_maxdet", False)) != as_bool(expected_valid_before):
    bad.append(
        "gcs_official_valid_before_maxdet="
        f"{args.get('gcs_official_valid_before_maxdet')!r}, expected {as_bool(expected_valid_before)!r}"
    )
if sorted(cfg_list(args.get("gcs_official_extent_decode_modes"))) != sorted(as_list(expected_extent_modes)):
    bad.append(
        "gcs_official_extent_decode_modes="
        f"{args.get('gcs_official_extent_decode_modes')!r}, expected {as_list(expected_extent_modes)!r}"
    )
if sorted(cfg_list(args.get("gcs_official_count_modes"))) != sorted(as_list(expected_count_modes)):
    bad.append(
        "gcs_official_count_modes="
        f"{args.get('gcs_official_count_modes')!r}, expected {as_list(expected_count_modes)!r}"
    )
if bool(args.get("gcs_official_count_aware_topk", False)) != as_bool(expected_count_aware):
    bad.append(
        "gcs_official_count_aware_topk="
        f"{args.get('gcs_official_count_aware_topk')!r}, expected {as_bool(expected_count_aware)!r}"
    )
if sorted(cfg_list(args.get("gcs_official_count_aware_extra_margins"))) != sorted(as_list(expected_extra_margins)):
    bad.append(
        "gcs_official_count_aware_extra_margins="
        f"{args.get('gcs_official_count_aware_extra_margins')!r}, expected {as_list(expected_extra_margins)!r}"
    )
if decode.get("decode_mode") != "query":
    bad.append(f"training official_best decode_mode={decode.get('decode_mode')!r}")
extent_mode = str(decode.get("extent_decode_mode", "none") or "none")
if extent_mode not in {"none", "interval", "intersect"}:
    bad.append(f"training official_best extent_decode_mode={extent_mode!r}")
selection_policy = official_best_sweep.get("selection_policy") or official_best_sweep.get("official_selection_policy") or {}
if selection_policy.get("name") != "official_best_v4":
    bad.append(f"training official_best selection_policy={selection_policy!r}")
if int((official_best_sweep.get("official_best") or {}).get("epoch", 0) or 0) <= 0:
    bad.append("training official_best source epoch is missing or non-positive")
if "best" not in (official_best_sweep.get("official_best") or {}):
    bad.append("training official_best metadata is missing best row")
if bad:
    raise SystemExit("Query extent full protocol contract check failed: " + "; ".join(bad))
print(json.dumps({"status": "ok", "contract": "query_extent_full_protocol"}))
PY
}

write_decode_yaml_from_sweep() {
  local label="$1"
  local sweep_dir="$2"
  local out_yaml="$3"
  local summary="${sweep_dir}/tusimple_official_sweep_summary.json"

  if [[ ! -f "${summary}" ]]; then
    echo "Missing ${label} post-train sweep summary: ${summary}" >&2
    exit 2
  fi
  python - "${summary}" "${out_yaml}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

from ultralytics.models.gcs.decode_summary import query_decode_cfg

summary_path, out_yaml = sys.argv[1:]
with open(summary_path, "r", encoding="utf-8") as f:
    data = json.load(f)
best = dict(data["best"])
if str(best.get("decode_mode", "query")) != "query":
    raise SystemExit(f"Expected query decode best row, got {best.get('decode_mode')!r}")
cfg = query_decode_cfg(best)
out = Path(out_yaml)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(json.dumps({"decode_yaml": str(out), "extent_decode_mode": cfg.get("extent_decode_mode")}))
PY
}

run_test_eval() {
  local label="$1"
  local weights="$2"
  local decode_yaml="$3"
  local save_dir="$4"
  local half_args=()
  local max_images_args=()
  if is_true "${HALF}"; then
    half_args=(--half)
  fi
  if [[ "${TEST_MAX_IMAGES}" -gt 0 ]]; then
    max_images_args=(--max-images "${TEST_MAX_IMAGES}")
  fi
  if [[ -e "${save_dir}" ]] && ! is_true "${OVERWRITE_TESTS}"; then
    echo "TEST directory already exists for ${label}: ${save_dir}" >&2
    echo "Set OVERWRITE_TESTS=1 or choose a new RUN_NAME." >&2
    exit 2
  fi
  python tools/eval_tusimple_official.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split test \
    --weights "${weights}" \
    --decode-yaml "${decode_yaml}" \
    --imgsz 544 960 \
    --device "${DEVICE}" \
    --warmup "${TEST_WARMUP}" \
    "${max_images_args[@]}" \
    "${half_args[@]}" \
    --save-records \
    --save-dir "${save_dir}"
}

ensure_diag_dir() {
  local dir="$1"
  if [[ -e "${dir}" ]] && ! is_true "${OVERWRITE_DIAGS}"; then
    echo "Diagnostic directory already exists: ${dir}" >&2
    echo "Set OVERWRITE_DIAGS=1 or choose a new RUN_NAME/DIAG_DIR." >&2
    exit 2
  fi
}

run_extent_diag() {
  local label="$1"
  local split_name="$2"
  local split="$3"
  local gt_json="$4"
  local weights="$5"
  local decode_yaml="$6"
  local save_dir="${DIAG_DIR}/${label}/extent_gate_${split_name}"
  local half_args=()
  local gt_args=()
  local test_args=()
  local max_images="${VAL_DIAG_MAX_IMAGES}"
  if [[ "${split}" == "test" ]]; then
    max_images="${TEST_DIAG_MAX_IMAGES}"
    test_args=(--allow-test-oracle)
  fi
  if is_true "${HALF}"; then
    half_args=(--half)
  fi
  if [[ -n "${gt_json}" ]]; then
    gt_args=(--gt-json "${gt_json}")
  fi
  ensure_diag_dir "${save_dir}"
  python tools/diagnose_tusimple_query_extent_gate.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split "${split}" \
    "${gt_args[@]}" \
    "${test_args[@]}" \
    --weights "${weights}" \
    --decode-yaml "${decode_yaml}" \
    --imgsz 544 960 \
    --extent-decode-modes "${DIAG_EXTENT_DECODE_MODES_ARR[@]}" \
    --match-thr-px 20 \
    --match-min-overlap 3 \
    --short-visible-max 10 \
    --max-images "${max_images}" \
    --warmup "${DIAG_WARMUP}" \
    --device "${DEVICE}" \
    "${half_args[@]}" \
    --save-dir "${save_dir}"
}

run_oracle_rank_diag() {
  local label="$1"
  local split_name="$2"
  local split="$3"
  local gt_json="$4"
  local weights="$5"
  local decode_yaml="$6"
  local save_dir="${DIAG_DIR}/${label}/oracle_rank_${split_name}"
  local half_args=()
  local gt_args=()
  local test_args=()
  local max_images="${VAL_DIAG_MAX_IMAGES}"
  if [[ "${split}" == "test" ]]; then
    max_images="${TEST_DIAG_MAX_IMAGES}"
    test_args=(--allow-test-oracle)
  fi
  if is_true "${HALF}"; then
    half_args=(--half)
  fi
  if [[ -n "${gt_json}" ]]; then
    gt_args=(--gt-json "${gt_json}")
  fi
  ensure_diag_dir "${save_dir}"
  python tools/diagnose_tusimple_oracle_rank.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split "${split}" \
    "${gt_args[@]}" \
    "${test_args[@]}" \
    --weights "${weights}" \
    --decode-yaml "${decode_yaml}" \
    --imgsz 544 960 \
    --pool-max-det 12 \
    --max-images "${max_images}" \
    --warmup "${DIAG_WARMUP}" \
    --device "${DEVICE}" \
    "${half_args[@]}" \
    --save-dir "${save_dir}" \
    --save-records
}

run_val_diagnostics_for_checkpoint() {
  local label="$1"
  local weights="$2"
  local decode_yaml="$3"
  if is_true "${RUN_VAL_DIAGS}"; then
    run_extent_diag "${label}" val val "${GT_JSON}" "${weights}" "${decode_yaml}"
    run_oracle_rank_diag "${label}" val val "${GT_JSON}" "${weights}" "${decode_yaml}"
  fi
}

run_test_diagnostics_for_checkpoint() {
  local label="$1"
  local weights="$2"
  local decode_yaml="$3"
  if is_true "${RUN_TEST_DIAGS}"; then
    if ! is_true "${RUN_TESTS}"; then
      echo "RUN_TEST_DIAGS=1 requires RUN_TESTS=1 because TEST diagnostics are reporting-only after TEST eval." >&2
      exit 2
    fi
    run_extent_diag "${label}" test test "" "${weights}" "${decode_yaml}"
    run_oracle_rank_diag "${label}" test test "" "${weights}" "${decode_yaml}"
  fi
}

write_full_summary() {
  python - \
    "${RUN_NAME}" \
    "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_SWEEP_DIR}" "${OFFICIAL_BEST_POST_DECODE_YAML}" "${OFFICIAL_BEST_TEST_DIR}" \
    "${BEST_WEIGHTS}" "${BEST_SWEEP_DIR}" "${BEST_POST_DECODE_YAML}" "${BEST_TEST_DIR}" \
    "${DIAG_DIR}" "${PROTOCOL_SUMMARY}" "${RUN_TESTS}" "${RUN_TEST_DIAGS}" <<'PY'
import json
import sys
from pathlib import Path

(
    run_name,
    official_best_weights,
    official_best_sweep_dir,
    official_best_decode_yaml,
    official_best_test_dir,
    best_weights,
    best_sweep_dir,
    best_decode_yaml,
    best_test_dir,
    diag_dir,
    protocol_summary,
    run_tests,
    run_test_diags,
) = sys.argv[1:]

def maybe_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def package(label: str, weights: str, sweep_dir: str, decode_yaml: str, test_dir: str) -> dict:
    test_summary = Path(test_dir) / "tusimple_official_summary.json"
    return {
        "label": label,
        "weights": weights,
        "posttrain_val_sweep_summary": str(Path(sweep_dir) / "tusimple_official_sweep_summary.json"),
        "posttrain_val_best": maybe_json(str(Path(sweep_dir) / "tusimple_official_sweep_summary.json")).get("best"),
        "posttrain_decode_yaml": decode_yaml,
        "test_summary": str(test_summary) if test_summary.exists() else None,
        "test_metrics": maybe_json(str(test_summary)).get("metrics") if test_summary.exists() else None,
        "diagnostics": {
            "val_extent_gate": str(Path(diag_dir) / label / "extent_gate_val" / "summary.json"),
            "val_oracle_rank": str(Path(diag_dir) / label / "oracle_rank_val" / "oracle_rank_summary.json"),
            "test_extent_gate": str(Path(diag_dir) / label / "extent_gate_test" / "summary.json"),
            "test_oracle_rank": str(Path(diag_dir) / label / "oracle_rank_test" / "oracle_rank_summary.json"),
        },
    }

summary = {
    "run_name": run_name,
    "selection_split": "official-val",
    "training_time_official_best": "weights/official_best.pt selected by periodic official-val cached sweeps",
    "posttrain_selection": "official_best.pt and best.pt each receive an independent post-train official-val sweep",
    "test_usage": "paired_reporting_only_from_each_posttrain_val_selected_decode",
    "paired_test_requested_by_user": True,
    "do_not_compare_test_to_select_checkpoint": True,
    "run_tests": str(run_tests).strip().lower() in {"1", "true", "yes", "on"},
    "run_test_diagnostics": str(run_test_diags).strip().lower() in {"1", "true", "yes", "on"},
    "do_not_select_from_test": True,
    "official_best": package("official_best", official_best_weights, official_best_sweep_dir, official_best_decode_yaml, official_best_test_dir),
    "best": package("best", best_weights, best_sweep_dir, best_decode_yaml, best_test_dir),
}
Path(protocol_summary).parent.mkdir(parents=True, exist_ok=True)
Path(protocol_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(
    {
        "summary": protocol_summary,
        "official_best_test_metrics": summary["official_best"]["test_metrics"],
        "best_test_metrics": summary["best"]["test_metrics"],
    },
    indent=2,
))
PY
}

echo "Q12/env30 query extent full protocol"
echo "RUN_NAME=${RUN_NAME}"
echo "EPOCHS=${EPOCHS}, RUN_TRAIN=${RUN_TRAIN}, RUN_TESTS=${RUN_TESTS}, RUN_TEST_DIAGS=${RUN_TEST_DIAGS}"
echo "Selection GT: ${GT_JSON}"
echo "Training official extent modes: ${OFFICIAL_EXTENT_DECODE_MODES}"
echo "Post-train sweep extent modes: ${SWEEP_EXTENT_DECODE_MODES}"
echo "Extent loss: gain=${GCS_QUERY_EXTENT}, short_thr=${GCS_QUERY_EXTENT_SHORT_VISIBLE_THR}, short_weight=${GCS_QUERY_EXTENT_SHORT_WEIGHT}, gt_min_lanes=${GCS_QUERY_EXTENT_GT_MIN_LANES}"
echo "Count-aware top-k: train=${COUNT_AWARE_TOPK}, sweep=${SWEEP_COUNT_AWARE_TOPK}"
echo "TEST policy: if RUN_TESTS=1, official_best and best are both reported because this protocol explicitly requests paired TEST reporting; never use TEST to select between them."

check_inputs
run_training_and_posttrain_sweeps
check_training_contract
write_decode_yaml_from_sweep official_best "${OFFICIAL_BEST_SWEEP_DIR}" "${OFFICIAL_BEST_POST_DECODE_YAML}"
write_decode_yaml_from_sweep best "${BEST_SWEEP_DIR}" "${BEST_POST_DECODE_YAML}"

run_val_diagnostics_for_checkpoint official_best "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_POST_DECODE_YAML}"
run_val_diagnostics_for_checkpoint best "${BEST_WEIGHTS}" "${BEST_POST_DECODE_YAML}"

if is_true "${RUN_TESTS}"; then
  run_test_eval official_best "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_POST_DECODE_YAML}" "${OFFICIAL_BEST_TEST_DIR}"
  run_test_diagnostics_for_checkpoint official_best "${OFFICIAL_BEST_WEIGHTS}" "${OFFICIAL_BEST_POST_DECODE_YAML}"
  run_test_eval best "${BEST_WEIGHTS}" "${BEST_POST_DECODE_YAML}" "${BEST_TEST_DIR}"
  run_test_diagnostics_for_checkpoint best "${BEST_WEIGHTS}" "${BEST_POST_DECODE_YAML}"
else
  if is_true "${RUN_TEST_DIAGS}"; then
    echo "RUN_TEST_DIAGS=1 requires RUN_TESTS=1 because TEST diagnostics are reporting-only after TEST eval." >&2
    exit 2
  fi
  echo "RUN_TESTS=0: skipped TEST official ACC. Set RUN_TESTS=1 only for final reporting after official-val gates." >&2
fi

write_full_summary
