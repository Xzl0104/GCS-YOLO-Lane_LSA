#!/usr/bin/env bash
set -euo pipefail

# Official-val-only prediction-only geometry-source ablation.
# The main decode remains the baseline; this script never runs TEST.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-query_short_local_refine_env30_window_v3_probe10}"
RUN_DIR="${RUN_DIR:-runs/gcs_lane/${RUN_NAME}}"
WEIGHTS="${WEIGHTS:-${RUN_DIR}/weights/official_best.pt}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-1}"

MAIN_SAVE_DIR="${MAIN_SAVE_DIR:-${RUN_DIR}/official_val_refined_decode_main}"
REFINED_SAVE_DIR="${REFINED_SAVE_DIR:-${RUN_DIR}/official_val_refined_decode_short_refined}"

SWEEP_ARGS=(
  --archive-root "$ARCHIVE_ROOT"
  --split val
  --gt-json "$GT_JSON"
  --weights "$WEIGHTS"
  --imgsz 544 960
  --device "$DEVICE"
  --confs 0.001 0.003 0.005
  --point-valid-thrs 0.50 0.55 0.60
  --nms-dist-pxs 0 18 30
  --max-dets 5 6
  --min-points 2 5
  --valid-before-maxdet
  --extent-decode-modes none
  --count-modes score_sum
)

if [[ "$HALF" == "1" ]]; then
  SWEEP_ARGS+=(--half)
fi

echo "[1/3] Rebuild official_best prediction cache and sweep main geometry"
python tools/sweep_tusimple_official_cached.py \
  "${SWEEP_ARGS[@]}" \
  --query-points-source main \
  --rebuild-cache \
  --save-dir "$MAIN_SAVE_DIR"

echo "[2/3] Sweep prediction-only short-refined geometry from the same cache"
python tools/sweep_tusimple_official_cached.py \
  "${SWEEP_ARGS[@]}" \
  --query-points-source short_refined \
  --sweep-only \
  --save-dir "$REFINED_SAVE_DIR"

echo "[3/3] Compare official-val summaries"
MAIN_SUMMARY="$MAIN_SAVE_DIR/tusimple_official_sweep_summary.json"
REFINED_SUMMARY="$REFINED_SAVE_DIR/tusimple_official_sweep_summary.json"
python - "$MAIN_SUMMARY" "$REFINED_SUMMARY" <<'PY'
import json
import sys

main = json.load(open(sys.argv[1], encoding="utf-8"))["best"]
refined = json.load(open(sys.argv[2], encoding="utf-8"))["best"]

keys = (
    "official_acc",
    "official_FP",
    "official_FN",
    "official_score",
    "count_acc",
    "count_acc_3",
    "count_acc_4",
    "count_acc_5",
)
print("main:", json.dumps({key: main.get(key) for key in keys}, sort_keys=True))
print("short_refined:", json.dumps({key: refined.get(key) for key in keys}, sort_keys=True))
print(
    "delta_short_refined_minus_main:",
    json.dumps(
        {
            key: round(float(refined.get(key, 0.0)) - float(main.get(key, 0.0)), 6)
            for key in keys
        },
        sort_keys=True,
    ),
)
print("main_decode:", json.dumps({key: main.get(key) for key in ("conf", "point_valid_thr", "nms_dist_px", "max_det", "min_points")}, sort_keys=True))
print("short_refined_decode:", json.dumps({key: refined.get(key) for key in ("conf", "point_valid_thr", "nms_dist_px", "max_det", "min_points")}, sort_keys=True))
PY

echo "main summary: $MAIN_SUMMARY"
echo "short-refined summary: $REFINED_SUMMARY"
