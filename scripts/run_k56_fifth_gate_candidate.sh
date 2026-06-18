#!/usr/bin/env bash
set -euo pipefail

# Count-conditioned fifth-gate candidate from the K56 epoch152 parent.
# Builds a train-only hard manifest, then fine-tunes the survival-head model.

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

EPOCHS="${EPOCHS:-6}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-1}"
LR0="${LR0:-0.0001}"
LRF="${LRF:-}"
DEVICE="${DEVICE:-}"
EXIST_OK="${EXIST_OK:-0}"
FORCE_MANIFEST="${FORCE_MANIFEST:-0}"

SURVIVAL_GAIN="${SURVIVAL_GAIN:-0.2}"
FIFTH_GATE_HARD_NEGATIVE_WEIGHT="${FIFTH_GATE_HARD_NEGATIVE_WEIGHT:-2.0}"
COUNT_BOUNDARY_HARD_MARGIN="${COUNT_BOUNDARY_HARD_MARGIN:-0.2}"
COUNT_BOUNDARY_HARD_MARGIN_GAIN="${COUNT_BOUNDARY_HARD_MARGIN_GAIN:-0.05}"

RUN_NAME="${RUN_NAME:-gcs_yolo_lane_s_q12_k56_countcond_fifthgate_ft${EPOCHS}_seed${SEED}_b${BATCH}w${WORKERS}}"
TRAIN_EVAL_DIR="${TRAIN_EVAL_DIR:-runs/gcs_lane/k56_fifth_gate_train_parent_eval}"
TRAIN_EVAL_SUMMARY="${TRAIN_EVAL_SUMMARY:-$TRAIN_EVAL_DIR/eval_summary.json}"
HARD_SAMPLE_FILE="${HARD_SAMPLE_FILE:-runs/gcs_lane/hard_samples/k56_fifth_gate_train.txt}"

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

prepare_fifth_gate_hard_samples() {
  local sidecar="${HARD_SAMPLE_FILE}.summary.json"
  if [[ -f "$HARD_SAMPLE_FILE" && -f "$sidecar" && "$FORCE_MANIFEST" != "1" ]]; then
    echo "Reusing existing train fifth-gate hard manifest: $HARD_SAMPLE_FILE"
    return 0
  fi

  mkdir -p "$(dirname "$HARD_SAMPLE_FILE")" "$TRAIN_EVAL_DIR"

  if [[ ! -f "$TRAIN_EVAL_SUMMARY" || "$FORCE_MANIFEST" == "1" ]]; then
    run_cmd "$PYTHON_BIN" tools/eval_gcs.py \
      --weights "$PARENT_WEIGHTS" \
      --data "$DATA" \
      --split train \
      --imgsz 544 960 \
      --conf 0.005 \
      --point-valid-thr 0.35 \
      --nms-dist-px 18.0 \
      --max-det 5 \
      --min-points 6 \
      --use-count-head-decode \
      --save-dir "$TRAIN_EVAL_DIR" \
      --save-json "$TRAIN_EVAL_SUMMARY"
  else
    echo "Reusing existing train eval summary: $TRAIN_EVAL_SUMMARY"
  fi

  local overwrite_args=()
  if [[ "$FORCE_MANIFEST" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi

  run_cmd "$PYTHON_BIN" tools/build_gcs_hard_samples_from_eval.py \
    --eval-summary "$TRAIN_EVAL_SUMMARY" \
    --preset k56_fifth_gate_hardsamples \
    --dataset-root "$DATASET_ROOT" \
    --target-splits train \
    --require-target-match \
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

  echo "K56 count-conditioned fifth-gate candidate"
  echo "root=$ROOT_DIR"
  echo "parent=$PARENT_WEIGHTS"
  echo "run=$RUN_NAME"
  echo "epochs=$EPOCHS lr0=$LR0 lrf=${LRF:-train_gcs_default} batch=$BATCH workers=$WORKERS seed=$SEED device=${DEVICE:-default}"

  prepare_fifth_gate_hard_samples

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
