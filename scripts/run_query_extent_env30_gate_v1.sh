#!/usr/bin/env bash
set -euo pipefail

# Official-val/train-side diagnostic gate for the Q12/env30 query extent probe.
# Uses GT only for diagnostics. TEST stays closed.

export RUN_NAME="${RUN_NAME:-query_extent_env30_probe40_v1}"
export PROJECT="${PROJECT:-runs/gcs_lane}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OVERWRITE_DIAGS="${OVERWRITE_DIAGS:-0}"

RUN_DIR="${PROJECT}/${RUN_NAME}"
WEIGHTS="${WEIGHTS:-${RUN_DIR}/weights/official_best.pt}"
DECODE_YAML="${DECODE_YAML:-${RUN_DIR}/weights/official_best_decode.yaml}"
ARGS_YAML="${ARGS_YAML:-${RUN_DIR}/args.yaml}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
TRAIN0601_GT_JSON="${TRAIN0601_GT_JSON:-${ARCHIVE_ROOT}/train_set/label_data_0601.json}"
TRAIN0531_GT_JSON="${TRAIN0531_GT_JSON:-${ARCHIVE_ROOT}/train_set/label_data_0531.json}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-1}"
WARMUP="${WARMUP:-20}"
MAX_IMAGES="${MAX_IMAGES:-0}"
EXTENT_DECODE_MODES="${EXTENT_DECODE_MODES:-interval intersect}"

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

if [[ ! -f "${ARGS_YAML}" ]]; then
  echo "Missing training args for contract check: ${ARGS_YAML}" >&2
  exit 2
fi
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Missing weights: ${WEIGHTS}" >&2
  exit 2
fi
if [[ ! -f "${DECODE_YAML}" ]]; then
  echo "Missing decode yaml: ${DECODE_YAML}" >&2
  exit 2
fi

python - "${ARGS_YAML}" "${DECODE_YAML}" <<'PY'
import math
import sys

import yaml

args_path, decode_path = sys.argv[1:]
with open(args_path, "r", encoding="utf-8") as f:
    args = yaml.safe_load(f) or {}
with open(decode_path, "r", encoding="utf-8") as f:
    decode = yaml.safe_load(f) or {}

bad = []
model = str(args.get("model", ""))
if not model.endswith("gcs-yolo-lane-s-q12-k56-extent.yaml"):
    bad.append(f"model={model!r}")

expected = {
    "gcs_query_extent": 0.5,
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
for key, target in expected.items():
    value = args.get(key)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        bad.append(f"{key}={value!r}")
        continue
    if not math.isclose(numeric, target, rel_tol=0.0, abs_tol=1e-12):
        bad.append(f"{key}={value!r}, expected {target}")

extent_mode = str(decode.get("extent_decode_mode", "none") or "none")
if extent_mode not in {"none", "interval", "intersect"}:
    bad.append(f"decode.extent_decode_mode={extent_mode!r}")
if bool(decode.get("extent_decode", False)) and extent_mode == "none":
    bad.append("decode.extent_decode=true with mode none")

if bad:
    raise SystemExit("Q12 query extent contract check failed: " + ", ".join(bad))

print("Q12 query extent contract check passed: extent YAML/loss active on env30 parent, Count/Quality/count/boundary/Q24 losses disabled.")
PY

HALF_ARGS=()
if is_true "${HALF}"; then
  HALF_ARGS=(--half)
fi

read -r -a EXTENT_DECODE_MODES_ARR <<< "${EXTENT_DECODE_MODES}"

ensure_empty_or_overwrite() {
  local dir="$1"
  if [[ -e "${dir}" ]] && ! is_true "${OVERWRITE_DIAGS}"; then
    echo "Diagnostic directory already exists: ${dir}" >&2
    echo "Set OVERWRITE_DIAGS=1 or choose a new RUN_NAME/diagnostic path." >&2
    exit 2
  fi
}

run_extent_gate() {
  local split_name="$1"
  local split="$2"
  local gt_json="$3"
  local save_dir="${RUN_DIR}/query_extent_gate_${split_name}_official_best_decode"
  ensure_empty_or_overwrite "${save_dir}"
  python tools/diagnose_tusimple_query_extent_gate.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split "${split}" \
    --gt-json "${gt_json}" \
    --weights "${WEIGHTS}" \
    --decode-yaml "${DECODE_YAML}" \
    --imgsz 544 960 \
    --extent-decode-modes "${EXTENT_DECODE_MODES_ARR[@]}" \
    --match-thr-px 20 \
    --match-min-overlap 3 \
    --short-visible-max 10 \
    --max-images "${MAX_IMAGES}" \
    --warmup "${WARMUP}" \
    --device "${DEVICE}" \
    "${HALF_ARGS[@]}" \
    --save-dir "${save_dir}"
}

run_oracle_rank() {
  local save_dir="${RUN_DIR}/oracle_rank_val_extent_official_best_decode"
  ensure_empty_or_overwrite "${save_dir}"
  python tools/diagnose_tusimple_oracle_rank.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split val \
    --gt-json "${GT_JSON}" \
    --weights "${WEIGHTS}" \
    --decode-yaml "${DECODE_YAML}" \
    --imgsz 544 960 \
    --pool-max-det 12 \
    --max-images "${MAX_IMAGES}" \
    --warmup "${WARMUP}" \
    --device "${DEVICE}" \
    "${HALF_ARGS[@]}" \
    --save-dir "${save_dir}" \
    --save-records
}

echo "Q12 query extent diagnostic gate: run=${RUN_NAME}, tests=${RUN_TESTS}"
echo "TEST remains closed; running official-val and train-side endpoint/interval diagnostics."

run_extent_gate val val "${GT_JSON}"
run_extent_gate train0601 train "${TRAIN0601_GT_JSON}"
run_extent_gate train0531 train "${TRAIN0531_GT_JSON}"
run_oracle_rank
