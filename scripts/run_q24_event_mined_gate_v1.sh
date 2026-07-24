#!/usr/bin/env bash
set -euo pipefail

# Build official-val/train-side Q24 event-mined diagnostics for an existing run.
# This script uses GT only for diagnostics and blocks TEST by default.

RUN_NAME="${RUN_NAME:-query_alpha05_env30_q24_gt5safe_boundary_probe40_v1}"
PROJECT="${PROJECT:-runs/gcs_lane}"
RUN_DIR="${PROJECT}/${RUN_NAME}"
WEIGHTS="${WEIGHTS:-${RUN_DIR}/weights/official_best.pt}"
DECODE_YAML="${DECODE_YAML:-${RUN_DIR}/weights/official_best_decode.yaml}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
TRAIN0601_GT_JSON="${TRAIN0601_GT_JSON:-${ARCHIVE_ROOT}/train_set/label_data_0601.json}"
TRAIN0531_GT_JSON="${TRAIN0531_GT_JSON:-${ARCHIVE_ROOT}/train_set/label_data_0531.json}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-1}"
WARMUP="${WARMUP:-20}"
MAX_IMAGES="${MAX_IMAGES:-0}"
OVERWRITE_DIAGS="${OVERWRITE_DIAGS:-0}"
GT5_QUERIES="${GT5_QUERIES:-12,13,15,16,20,21,23}"
GT4_QUERIES="${GT4_QUERIES:-14,17,18,19,22}"
WATCH_QUERIES="${WATCH_QUERIES:-0,1,11}"

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

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Missing weights: ${WEIGHTS}" >&2
  exit 2
fi
if [[ ! -f "${DECODE_YAML}" ]]; then
  echo "Missing decode yaml: ${DECODE_YAML}" >&2
  exit 2
fi

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
  local label="$1"
  local split="$2"
  local gt_json="$3"
  local save_dir="$4"
  ensure_empty_or_overwrite "${save_dir}"
  python tools/diagnose_tusimple_raw_q12_filters.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split "${split}" \
    --gt-json "${gt_json}" \
    --weights "${WEIGHTS}" \
    --decode-yaml "${DECODE_YAML}" \
    --imgsz 544 960 \
    --pool-max-det 24 \
    --match-thr-px 20 \
    --match-min-overlap 3 \
    --short-visible-max 10 \
    --max-images "${MAX_IMAGES}" \
    --warmup "${WARMUP}" \
    --device "${DEVICE}" \
    "${HALF_ARGS[@]}" \
    --save-dir "${save_dir}"
}

run_extra_diag() {
  local gt_count="$1"
  local pred_count="$2"
  local save_dir="$3"
  ensure_empty_or_overwrite "${save_dir}"
  python tools/diagnose_tusimple_gt5_extra_lanes.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split val \
    --gt-json "${GT_JSON}" \
    --weights "${WEIGHTS}" \
    --decode-yaml "${DECODE_YAML}" \
    --imgsz 544 960 \
    --target-gt-count "${gt_count}" \
    --target-pred-count "${pred_count}" \
    --match-min-overlap 3 \
    --boundary-visible-max 10 \
    --boundary-edge-margin-px 80 \
    --max-images "${MAX_IMAGES}" \
    --warmup "${WARMUP}" \
    --device "${DEVICE}" \
    "${HALF_ARGS[@]}" \
    --save-dir "${save_dir}"
}

VAL_RAW_DIR="${RUN_DIR}/raw_q24_filters_val_official_best_decode"
TRAIN0601_RAW_DIR="${RUN_DIR}/raw_q24_filters_train0601_official_best_decode"
TRAIN0531_RAW_DIR="${RUN_DIR}/raw_q24_filters_train0531_official_best_decode"
GT3_TO4_DIR="${RUN_DIR}/extra_lane_diag_val_gt3_to4_official_best_decode"
GT3_TO5_DIR="${RUN_DIR}/extra_lane_diag_val_gt3_to5_official_best_decode"
GT4_TO5_DIR="${RUN_DIR}/extra_lane_diag_val_gt4_to5_official_best_decode"
GT4_TO6_DIR="${RUN_DIR}/extra_lane_diag_val_gt4_to6_official_best_decode"
GT5_TO6_DIR="${RUN_DIR}/extra_lane_diag_val_gt5_to6_official_best_decode"
EVENT_GATE_DIR="${RUN_DIR}/event_mined_gate"

echo "Run event-mined diagnostics for ${RUN_NAME}"
echo "TEST remains closed; diagnostics use official-val and train-side GT only."

run_raw_diag val val "${GT_JSON}" "${VAL_RAW_DIR}"
run_raw_diag train0601 train "${TRAIN0601_GT_JSON}" "${TRAIN0601_RAW_DIR}"
run_raw_diag train0531 train "${TRAIN0531_GT_JSON}" "${TRAIN0531_RAW_DIR}"
run_extra_diag 3 4 "${GT3_TO4_DIR}"
run_extra_diag 3 5 "${GT3_TO5_DIR}"
run_extra_diag 4 5 "${GT4_TO5_DIR}"
run_extra_diag 4 6 "${GT4_TO6_DIR}"
run_extra_diag 5 6 "${GT5_TO6_DIR}"

ensure_empty_or_overwrite "${EVENT_GATE_DIR}"
python tools/diagnose_q24_event_mined_gate.py \
  --run-name "${RUN_NAME}" \
  --raw "val=${VAL_RAW_DIR}/raw_gt_lane_diagnostics.csv" \
  --raw "train0601=${TRAIN0601_RAW_DIR}/raw_gt_lane_diagnostics.csv" \
  --raw "train0531=${TRAIN0531_RAW_DIR}/raw_gt_lane_diagnostics.csv" \
  --gt5-extra "val_gt5_to6=${GT5_TO6_DIR}/gt5_extra_lanes.csv" \
  --false-extra "val_gt3_to4=${GT3_TO4_DIR}/gt5_extra_lanes.csv" \
  --false-extra "val_gt3_to5=${GT3_TO5_DIR}/gt5_extra_lanes.csv" \
  --false-extra "val_gt4_to5=${GT4_TO5_DIR}/gt5_extra_lanes.csv" \
  --false-extra "val_gt4_to6=${GT4_TO6_DIR}/gt5_extra_lanes.csv" \
  --gt5-queries "${GT5_QUERIES}" \
  --gt4-queries "${GT4_QUERIES}" \
  --watch-queries "${WATCH_QUERIES}" \
  --save-dir "${EVENT_GATE_DIR}"
