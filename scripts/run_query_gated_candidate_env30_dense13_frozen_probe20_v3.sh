#!/usr/bin/env bash
set -euo pipefail

# Dense13 follow-up to the frozen-env30 gated-candidate probe.
# It changes only the candidate offset grid to:
#   0, +/-10, +/-20, +/-30, +/-40, +/-50, +/-60 px
# The env30/base path remains frozen by the underlying v2 runner.

RUN_NAME="${RUN_NAME:-query_gated_candidate_env30_dense13_frozen_probe20_v3}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-gated-candidate-dense13-v3.yaml}"
PRETRAINED="${PRETRAINED:-runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_PRETRAIN_RAW_ORACLE="${RUN_PRETRAIN_RAW_ORACLE:-1}"
REQUIRE_PRETRAIN_RAW_GATE="${REQUIRE_PRETRAIN_RAW_GATE:-0}"

PROJECT="${PROJECT:-runs/gcs_lane}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
TRAIN0601_GT_JSON="${TRAIN0601_GT_JSON:-archive/TUSimple/train_set/label_data_0601.json}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-1}"
WARMUP="${HARD_COVERAGE_WARMUP:-20}"
PRETRAIN_VAL_GT5_RAW_MIN="${PRETRAIN_VAL_GT5_RAW_MIN:-44}"
PRETRAIN_TRAIN0601_GT5_RAW_MIN="${PRETRAIN_TRAIN0601_GT5_RAW_MIN:-160}"
PRETRAIN_TRAIN0601_GT4_RAW_MIN="${PRETRAIN_TRAIN0601_GT4_RAW_MIN:-15}"

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

hard_coverage_args() {
  if is_true "${HALF}"; then
    printf '%s\n' --half
  fi
}

check_pretrain_gate() {
  local val_json="$1"
  local train_json="$2"
  python - "$val_json" "$train_json" \
    "${PRETRAIN_VAL_GT5_RAW_MIN}" \
    "${PRETRAIN_TRAIN0601_GT5_RAW_MIN}" \
    "${PRETRAIN_TRAIN0601_GT4_RAW_MIN}" <<'PY'
import json
import sys

val_path, train_path = sys.argv[1], sys.argv[2]
val_min = int(float(sys.argv[3]))
train_gt5_min = int(float(sys.argv[4]))
train_gt4_min = int(float(sys.argv[5]))

def group(path, name):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["groups"][name]

val_gt5 = group(val_path, "short_gt5")
train_gt5 = group(train_path, "train0601_short_gt5")
train_gt4 = group(train_path, "train0601_short_gt4")
checks = [
    ("official-val short GT5 raw_candidate_hit20", int(val_gt5["raw_candidate_hit20"]), val_min),
    ("train0601 short GT5 raw_candidate_hit20", int(train_gt5["raw_candidate_hit20"]), train_gt5_min),
    ("train0601 short GT4 raw_candidate_hit20", int(train_gt4["raw_candidate_hit20"]), train_gt4_min),
]
failed = [(name, got, need) for name, got, need in checks if got < need]
for name, got, need in checks:
    print(f"{name}: {got} / required >= {need}")
if failed:
    raise SystemExit(2)
PY
}

if is_true "${RUN_PRETRAIN_RAW_ORACLE}"; then
  mapfile -t HALF_ARGS < <(hard_coverage_args)
  PRETRAIN_VAL_DIR="${PROJECT}/${RUN_NAME}_pretrain_oracle_hard_candidate_diag_val"
  PRETRAIN_TRAIN0601_DIR="${PROJECT}/${RUN_NAME}_pretrain_oracle_hard_candidate_diag_train0601"

  python tools/diagnose_gcs_short_candidate_hard_coverage.py \
    --model-yaml "${MODEL}" \
    --pretrained-base "${PRETRAINED}" \
    --archive-root "${ARCHIVE_ROOT}" \
    --split val \
    --gt-json "${GT_JSON}" \
    --imgsz 544 960 \
    --device "${DEVICE}" \
    --warmup "${WARMUP}" \
    "${HALF_ARGS[@]}" \
    --save-dir "${PRETRAIN_VAL_DIR}"

  python tools/diagnose_gcs_short_candidate_hard_coverage.py \
    --model-yaml "${MODEL}" \
    --pretrained-base "${PRETRAINED}" \
    --archive-root "${ARCHIVE_ROOT}" \
    --split train \
    --gt-json "${TRAIN0601_GT_JSON}" \
    --raw-file-contains clips/0601/ \
    --imgsz 544 960 \
    --device "${DEVICE}" \
    --warmup "${WARMUP}" \
    "${HALF_ARGS[@]}" \
    --allow-noncanonical-gt \
    --save-dir "${PRETRAIN_TRAIN0601_DIR}"

  if is_true "${REQUIRE_PRETRAIN_RAW_GATE}"; then
    check_pretrain_gate \
      "${PRETRAIN_VAL_DIR}/short_candidate_hard_summary.json" \
      "${PRETRAIN_TRAIN0601_DIR}/short_candidate_hard_summary.json"
  fi
fi

export RUN_NAME MODEL PRETRAINED RUN_TESTS

bash scripts/run_query_gated_candidate_env30_frozen_probe20_v2.sh

RUN_HARD_COVERAGE="${RUN_HARD_COVERAGE:-1}"

if is_true "${RUN_HARD_COVERAGE}"; then
  mapfile -t HALF_ARGS < <(hard_coverage_args)
  for WEIGHT_NAME in official_best last; do
    WEIGHTS="${PROJECT}/${RUN_NAME}/weights/${WEIGHT_NAME}.pt"
    if [[ ! -f "${WEIGHTS}" ]]; then
      echo "Skip hard coverage for missing weights: ${WEIGHTS}" >&2
      continue
    fi
    python tools/diagnose_gcs_short_candidate_hard_coverage.py \
      --weights "${WEIGHTS}" \
      --archive-root "${ARCHIVE_ROOT}" \
      --split val \
      --gt-json "${GT_JSON}" \
      --imgsz 544 960 \
      --device "${DEVICE}" \
      --warmup "${WARMUP}" \
      "${HALF_ARGS[@]}" \
      --save-dir "${PROJECT}/${RUN_NAME}_hard_candidate_diag_val_${WEIGHT_NAME}"

    python tools/diagnose_gcs_short_candidate_hard_coverage.py \
      --weights "${WEIGHTS}" \
      --archive-root "${ARCHIVE_ROOT}" \
      --split train \
      --gt-json "${TRAIN0601_GT_JSON}" \
      --raw-file-contains clips/0601/ \
      --imgsz 544 960 \
      --device "${DEVICE}" \
      --warmup "${WARMUP}" \
      "${HALF_ARGS[@]}" \
      --allow-noncanonical-gt \
      --save-dir "${PROJECT}/${RUN_NAME}_hard_candidate_diag_train0601_${WEIGHT_NAME}"
  done
fi

echo "Dense13 frozen-env30 probe complete. TEST remained closed."
