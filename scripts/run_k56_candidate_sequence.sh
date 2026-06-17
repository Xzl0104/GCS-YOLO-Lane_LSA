#!/usr/bin/env bash
set -euo pipefail

# Run the five validation-only K56 candidates in the declared order:
# 1. k56_no_gt5_erasing
# 2. k56_viscountsum_hardsamples
# 3. k56_survival_head
# 4. k56_decoder_aux_loss
# 5. k56_lane_aware_erasing
#
# Intended use on the remote CUDA server from the repository root:
#   source /root/miniconda3/etc/profile.d/conda.sh
#   conda activate ssh_lane
#   bash scripts/run_k56_candidate_sequence.sh
#
# Common overrides:
#   EPOCHS=8 BATCH=32 WORKERS=4 DEVICE=0 bash scripts/run_k56_candidate_sequence.sh
#   SKIP_COMPLETED=0 EXIST_OK=1 bash scripts/run_k56_candidate_sequence.sh

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT="${PROJECT:-runs/gcs_lane}"
DATA="${DATA:-data/tusimple_gcs_fixed_y_k56_960x544.yaml}"
DATASET_ROOT="${DATASET_ROOT:-datasets/tusimple_fixed_y_k56_960x544}"
BASE_MODEL="${BASE_MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml}"
SURVIVAL_MODEL="${SURVIVAL_MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-survival.yaml}"
DECODER_AUX_MODEL="${DECODER_AUX_MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-decoder-aux.yaml}"
PARENT_WEIGHTS="${PARENT_WEIGHTS:-runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/weights/official_best.pt}"
OFFICIAL_GT_JSON="${OFFICIAL_GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
OFFICIAL_ARCHIVE_ROOT="${OFFICIAL_ARCHIVE_ROOT:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset}"

EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
EXIST_OK="${EXIST_OK:-0}"

TRAIN_EVAL_DIR="${TRAIN_EVAL_DIR:-runs/gcs_lane/k56_viscountsum_train_parent_eval}"
TRAIN_EVAL_SUMMARY="${TRAIN_EVAL_SUMMARY:-$TRAIN_EVAL_DIR/eval_summary.json}"
HARD_SAMPLE_FILE="${HARD_SAMPLE_FILE:-runs/gcs_lane/hard_samples/k56_viscountsum_train.txt}"
FORCE_MANIFEST="${FORCE_MANIFEST:-0}"

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

is_completed_run() {
  local run_dir="$1"
  [[ -f "$run_dir/official_best_summary.json" || -f "$run_dir/weights/official_best.pt" ]]
}

run_candidate() {
  local name="$1"
  local model="$2"
  shift 2

  local run_dir="$PROJECT/$name"
  if is_completed_run "$run_dir"; then
    if [[ "$SKIP_COMPLETED" == "1" ]]; then
      echo "Skipping completed candidate: $name ($run_dir)"
      return 0
    fi
  fi
  if [[ -d "$run_dir" && "$EXIST_OK" != "1" ]]; then
    echo "Run directory already exists and is not marked complete: $run_dir" >&2
    echo "Set EXIST_OK=1 to let train_gcs.py reuse the run name, or move the directory first." >&2
    exit 2
  fi

  local extra_args=()
  if [[ "$EXIST_OK" == "1" ]]; then
    extra_args+=(--exist-ok)
  fi
  if [[ -n "$DEVICE" ]]; then
    extra_args+=(--device "$DEVICE")
  fi

  run_cmd "$PYTHON_BIN" tools/train_gcs.py \
    --model "$model" \
    --data "$DATA" \
    --imgsz 544 960 \
    --name "$name" \
    --project "$PROJECT" \
    --pretrained "$PARENT_WEIGHTS" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --seed "$SEED" \
    --gcs-official-best \
    --gcs-official-best-period 1 \
    --gcs-official-best-top-k 5 \
    --gcs-official-best-gt-json "$OFFICIAL_GT_JSON" \
    --gcs-official-best-archive-root "$OFFICIAL_ARCHIVE_ROOT" \
    "${extra_args[@]}" \
    "$@"
}

prepare_viscountsum_hard_samples() {
  local sidecar="${HARD_SAMPLE_FILE}.summary.json"
  if [[ -f "$HARD_SAMPLE_FILE" && -f "$sidecar" && "$FORCE_MANIFEST" != "1" ]]; then
    echo "Reusing existing train hard-sample manifest: $HARD_SAMPLE_FILE"
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
    --preset k56_viscountsum_hardsamples \
    --dataset-root "$DATASET_ROOT" \
    --target-splits train \
    --require-target-match \
    --output "$HARD_SAMPLE_FILE" \
    "${overwrite_args[@]}"
}

main() {
  require_file "$DATA"
  require_file "$BASE_MODEL"
  require_file "$SURVIVAL_MODEL"
  require_file "$DECODER_AUX_MODEL"
  require_file "$PARENT_WEIGHTS"
  require_file "$OFFICIAL_GT_JSON"
  require_dir "$OFFICIAL_ARCHIVE_ROOT"
  require_dir "$DATASET_ROOT/labels_gcs/train"

  echo "K56 candidate sequence"
  echo "root=$ROOT_DIR"
  echo "parent=$PARENT_WEIGHTS"
  echo "epochs=$EPOCHS batch=$BATCH workers=$WORKERS seed=$SEED device=${DEVICE:-default}"

  run_candidate \
    gcs_yolo_lane_s_q12_k56_no_gt5_erasing_ft8_seed1_b32w4 \
    "$BASE_MODEL" \
    --gcs-gt5-erasing 0.0

  prepare_viscountsum_hard_samples
  run_candidate \
    gcs_yolo_lane_s_q12_k56_viscountsum_hardsamples_ft8_seed1_b32w4 \
    "$BASE_MODEL" \
    --gcs-visible-count-sum 0.05 \
    --gcs-visible-count-sum-quality-weight 0.25 \
    --gcs-visible-count-boundary 0.25 \
    --gcs-hard-loss-file "$HARD_SAMPLE_FILE" \
    --gcs-visible-count-sum-hard-weight 2.0

  run_candidate \
    gcs_yolo_lane_s_q12_k56_survival_head_ft8_seed1_b32w4 \
    "$SURVIVAL_MODEL" \
    --gcs-survival 0.2

  run_candidate \
    gcs_yolo_lane_s_q12_k56_decoder_aux_ft8_seed1_b32w4 \
    "$DECODER_AUX_MODEL" \
    --gcs-decoder-aux 0.1

  run_candidate \
    gcs_yolo_lane_s_q12_k56_laneaware_erasing_ft8_seed1_b32w4 \
    "$BASE_MODEL" \
    --gcs-gt5-erasing 0.15 \
    --gcs-gt5-lane-aware-erasing \
    --gcs-gt5-erasing-lane-margin-px 8.0

  echo "K56 candidate sequence finished."
}

main "$@"
