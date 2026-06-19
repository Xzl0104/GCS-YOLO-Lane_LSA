#!/usr/bin/env bash
set -euo pipefail

# K56 fifth-gate v2 from the epoch152 parent.
# Builds a train-only near-miss manifest from count diagnostics, then fine-tunes
# the survival-head model with count-conditioned fifth-gate supervision.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT="${PROJECT:-runs/gcs_lane}"
DATA="${DATA:-data/tusimple_gcs_fixed_y_k56_960x544.yaml}"
DATASET_ROOT="${DATASET_ROOT:-datasets/tusimple_fixed_y_k56_960x544}"
MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-survival.yaml}"
PARENT_WEIGHTS="${PARENT_WEIGHTS:-runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/weights/official_best.pt}"
OFFICIAL_GT_JSON="${OFFICIAL_GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
OFFICIAL_ARCHIVE_ROOT="${OFFICIAL_ARCHIVE_ROOT:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset}"

EPOCHS="${EPOCHS:-4}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-1}"
LR0="${LR0:-0.0001}"
LRF="${LRF:-}"
DEVICE="${DEVICE:-}"
EXIST_OK="${EXIST_OK:-0}"
FORCE_DIAG="${FORCE_DIAG:-0}"
FORCE_MANIFEST="${FORCE_MANIFEST:-0}"

SURVIVAL_GAIN="${SURVIVAL_GAIN:-0.2}"
FIFTH_GATE_HARD_NEGATIVE_WEIGHT="${FIFTH_GATE_HARD_NEGATIVE_WEIGHT:-3.0}"
COUNT_BOUNDARY_HARD_MARGIN="${COUNT_BOUNDARY_HARD_MARGIN:-0.2}"
COUNT_BOUNDARY_HARD_MARGIN_GAIN="${COUNT_BOUNDARY_HARD_MARGIN_GAIN:-0.05}"

GT4_COUNT5_PROB_THR="${GT4_COUNT5_PROB_THR:-0.05}"
GT4_UNMATCHED_SCORE_THR="${GT4_UNMATCHED_SCORE_THR:-0.75}"
GT4_UNMATCHED_MIN_POINTS="${GT4_UNMATCHED_MIN_POINTS:-5}"
GT5_COUNT4_PROB_THR="${GT5_COUNT4_PROB_THR:-0.05}"
NEARMISS_TOPK="${NEARMISS_TOPK:-8}"

RUN_NAME="${RUN_NAME:-gcs_yolo_lane_s_q12_k56_fifthgate_nearmiss_v2_ft${EPOCHS}_seed${SEED}_b${BATCH}w${WORKERS}}"
TRAIN_DIAG_DIR="${TRAIN_DIAG_DIR:-runs/gcs_lane/k56_parent_count_diag_train_min1}"
HARD_SAMPLE_FILE="${HARD_SAMPLE_FILE:-runs/gcs_lane/hard_samples/k56_fifth_gate_nearmiss_train.txt}"

run_cmd() {
  printf '\n+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Required directory not found: $path" >&2
    exit 2
  fi
}

prepare_nearmiss_manifest() {
  local diag_jsonl="$TRAIN_DIAG_DIR/per_image.jsonl"
  local diag_summary="$TRAIN_DIAG_DIR/summary.json"
  local sidecar="${HARD_SAMPLE_FILE}.summary.json"

  if [[ ! -f "$diag_jsonl" || ! -f "$diag_summary" || "$FORCE_DIAG" == "1" ]]; then
    mkdir -p "$TRAIN_DIAG_DIR"
    run_cmd "$PYTHON_BIN" tools/diagnose_gcs_count_errors.py \
      --data "$DATA" \
      --split train \
      --weights "$PARENT_WEIGHTS" \
      --imgsz 544 960 \
      --gcs-count-min-gt-points 1 \
      --diagnostic-topk "$NEARMISS_TOPK" \
      --out "$TRAIN_DIAG_DIR" \
      --write-hard-samples
  else
    echo "Reusing existing train count diagnostics: $TRAIN_DIAG_DIR"
  fi

  if [[ -f "$HARD_SAMPLE_FILE" && -f "$sidecar" && "$FORCE_MANIFEST" != "1" && "$FORCE_DIAG" != "1" ]]; then
    echo "Reusing existing near-miss hard manifest: $HARD_SAMPLE_FILE"
    return 0
  fi

  local overwrite_args=()
  if [[ "$FORCE_MANIFEST" == "1" || "$FORCE_DIAG" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi

  mkdir -p "$(dirname "$HARD_SAMPLE_FILE")"
  run_cmd "$PYTHON_BIN" tools/build_gcs_hard_samples_from_eval.py \
    --eval-summary "$TRAIN_DIAG_DIR" \
    --preset k56_fifth_gate_nearmiss_hardsamples \
    --dataset-root "$DATASET_ROOT" \
    --target-splits train \
    --require-target-match \
    --gt4-count5-prob-thr "$GT4_COUNT5_PROB_THR" \
    --gt4-unmatched-score-thr "$GT4_UNMATCHED_SCORE_THR" \
    --gt4-unmatched-min-points "$GT4_UNMATCHED_MIN_POINTS" \
    --gt5-count4-prob-thr "$GT5_COUNT4_PROB_THR" \
    --nearmiss-topk "$NEARMISS_TOPK" \
    --output "$HARD_SAMPLE_FILE" \
    "${overwrite_args[@]}"
}

main() {
  require_file "$DATA"
  require_file "$MODEL"
  require_file "$PARENT_WEIGHTS"
  require_file "$OFFICIAL_GT_JSON"
  require_dir "$OFFICIAL_ARCHIVE_ROOT"
  require_dir "$DATASET_ROOT/labels_gcs/train"

  echo "K56 fifth-gate near-miss v2"
  echo "root=$ROOT_DIR"
  echo "parent=$PARENT_WEIGHTS"
  echo "diag=$TRAIN_DIAG_DIR"
  echo "manifest=$HARD_SAMPLE_FILE"
  echo "run=$RUN_NAME"
  echo "epochs=$EPOCHS lr0=$LR0 lrf=${LRF:-train_gcs_default} batch=$BATCH workers=$WORKERS seed=$SEED device=${DEVICE:-default}"

  prepare_nearmiss_manifest

  local extra_args=()
  if [[ "$EXIST_OK" == "1" ]]; then
    extra_args+=(--exist-ok)
  fi
  if [[ -n "$DEVICE" ]]; then
    extra_args+=(--device "$DEVICE")
  fi
  if [[ -n "$LRF" ]]; then
    extra_args+=(--lrf "$LRF")
  fi

  run_cmd "$PYTHON_BIN" tools/train_gcs.py \
    --model "$MODEL" \
    --data "$DATA" \
    --imgsz 544 960 \
    --name "$RUN_NAME" \
    --project "$PROJECT" \
    --pretrained "$PARENT_WEIGHTS" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --lr0 "$LR0" \
    --gcs-count-min-gt-points 1 \
    --gcs-survival "$SURVIVAL_GAIN" \
    --gcs-survival-target-mode count_conditioned_fifth \
    --gcs-fifth-gate-hard-negative-weight "$FIFTH_GATE_HARD_NEGATIVE_WEIGHT" \
    --gcs-count-boundary-hard-margin "$COUNT_BOUNDARY_HARD_MARGIN" \
    --gcs-count-boundary-hard-margin-gain "$COUNT_BOUNDARY_HARD_MARGIN_GAIN" \
    --gcs-count-adjacent-margin-gain 0.0 \
    --gcs-hard-loss-file "$HARD_SAMPLE_FILE" \
    --gcs-hard-edge-loss-weight-by-count none \
    --gcs-hard-edge-loss-terms none \
    --gcs-official-best \
    --gcs-official-best-period 1 \
    --gcs-official-best-top-k 5 \
    --gcs-official-best-gt-json "$OFFICIAL_GT_JSON" \
    --gcs-official-best-archive-root "$OFFICIAL_ARCHIVE_ROOT" \
    "${extra_args[@]}"
}

main "$@"
