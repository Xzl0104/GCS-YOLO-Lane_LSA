#!/usr/bin/env bash
set -euo pipefail

# Official-val/train-side diagnostics for the Q12/env30 Count Head only probe.
# This gate uses GT only for diagnostics and must not run TEST.

export RUN_NAME="${RUN_NAME:-query_count_head_ce025_env30_probe100_v1}"
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
if not model.endswith("gcs-yolo-lane-s-q12-k56-count.yaml"):
    bad.append(f"model={model!r}")

expected = {
    "gcs_query_count_ce": 0.25,
    "gcs_query_quality": 0.0,
    "gcs_count": 0.0,
    "gcs_count_under5": 0.0,
    "gcs_count_boundary": 0.0,
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

if decode.get("count_mode") != "count_logits":
    bad.append(f"decode.count_mode={decode.get('count_mode')!r}")
if not bool(decode.get("count_aware_topk", False)):
    bad.append("decode.count_aware_topk is not true")

if bad:
    raise SystemExit("Q12 count-only contract check failed: " + ", ".join(bad))

print("Q12 count-only contract check passed: count_logits top-k, no Quality Head loss, Q24 losses disabled.")
PY

HALF_ARGS=()
if is_true "${HALF}"; then
  HALF_ARGS=(--half)
fi

ensure_empty_or_overwrite() {
  local dir="$1"
  if [[ -e "${dir}" ]] && ! is_true "${OVERWRITE_DIAGS}"; then
    echo "Diagnostic directory already exists: ${dir}" >&2
    echo "Set OVERWRITE_DIAGS=1 or choose a new RUN_NAME/diagnostic path." >&2
    exit 2
  fi
}

run_raw_diag() {
  local split_name="$1"
  local split="$2"
  local gt_json="$3"
  local save_dir="${RUN_DIR}/raw_q12_filters_${split_name}_official_best_decode"
  ensure_empty_or_overwrite "${save_dir}"
  python tools/diagnose_tusimple_raw_q12_filters.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split "${split}" \
    --gt-json "${gt_json}" \
    --weights "${WEIGHTS}" \
    --decode-yaml "${DECODE_YAML}" \
    --imgsz 544 960 \
    --pool-max-det 12 \
    --match-thr-px 20 \
    --match-min-overlap 3 \
    --short-visible-max 10 \
    --max-images "${MAX_IMAGES}" \
    --warmup "${WARMUP}" \
    --device "${DEVICE}" \
    "${HALF_ARGS[@]}" \
    --save-dir "${save_dir}"
}

ORACLE_DIR="${RUN_DIR}/oracle_rank_val_official_best_decode"
ensure_empty_or_overwrite "${ORACLE_DIR}"

echo "Q12 Count Head only diagnostic gate: run=${RUN_NAME}, tests=${RUN_TESTS}"
echo "TEST remains closed; running official-val and train-side raw/oracle diagnostics."

run_raw_diag val val "${GT_JSON}"
run_raw_diag train0601 train "${TRAIN0601_GT_JSON}"
run_raw_diag train0531 train "${TRAIN0531_GT_JSON}"

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
  --require-count-logits \
  --save-dir "${ORACLE_DIR}" \
  --save-records
