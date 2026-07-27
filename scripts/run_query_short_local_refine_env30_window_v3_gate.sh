#!/usr/bin/env bash
set -euo pipefail

# Official-val/train-side geometry gate for Q12/env30 short local x-refine v3.
# Uses GT only for diagnostics. TEST stays closed.

export RUN_NAME="${RUN_NAME:-query_short_local_refine_env30_window_v3_probe10}"
export PROJECT="${PROJECT:-runs/gcs_lane}"
export RUN_TESTS="${RUN_TESTS:-0}"
export OVERWRITE_DIAGS="${OVERWRITE_DIAGS:-0}"
export ENV30_OFFICIAL_VAL_ACC="${ENV30_OFFICIAL_VAL_ACC:-0.973330}"
export VAL_SHORT_GT5_TARGET="${VAL_SHORT_GT5_TARGET:-48}"
export TRAIN0601_SHORT_GT5_TARGET="${TRAIN0601_SHORT_GT5_TARGET:-160}"
export TRAIN0601_SHORT_GT4_TARGET="${TRAIN0601_SHORT_GT4_TARGET:-14}"

RUN_DIR="${PROJECT}/${RUN_NAME}"
WEIGHTS="${WEIGHTS:-${RUN_DIR}/weights/official_best.pt}"
DECODE_YAML="${DECODE_YAML:-${RUN_DIR}/weights/official_best_decode.yaml}"
ARGS_YAML="${ARGS_YAML:-${RUN_DIR}/args.yaml}"
OFFICIAL_BEST_SWEEP_SUMMARY="${OFFICIAL_BEST_SWEEP_SUMMARY:-${PROJECT}/${RUN_NAME}_official_best_val_sweep_valid_before_maxdet/tusimple_official_sweep_summary.json}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
TRAIN0601_GT_JSON="${TRAIN0601_GT_JSON:-${ARCHIVE_ROOT}/train_set/label_data_0601.json}"
TRAIN0531_GT_JSON="${TRAIN0531_GT_JSON:-${ARCHIVE_ROOT}/train_set/label_data_0531.json}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-1}"
WARMUP="${WARMUP:-20}"
MAX_IMAGES="${MAX_IMAGES:-0}"
POOL_MAX_DET="${POOL_MAX_DET:-12}"

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if is_true "${RUN_TESTS}"; then
  echo "RUN_TESTS must stay 0 for the short local x-refine v3 gate. TEST is closed." >&2
  exit 2
fi

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
if [[ ! -f "${OFFICIAL_BEST_SWEEP_SUMMARY}" ]]; then
  echo "Missing official_best post-train official-val sweep summary: ${OFFICIAL_BEST_SWEEP_SUMMARY}" >&2
  echo "Run scripts/run_query_short_local_refine_env30_window_v3_probe10.sh first, or set OFFICIAL_BEST_SWEEP_SUMMARY." >&2
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


def as_float(key, default=None):
    value = args.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"{key} must be numeric, got {value!r}")


def as_int(key, default=None):
    value = args.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise SystemExit(f"{key} must be integer, got {value!r}")


def as_bool_value(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def as_bool(key, default=False):
    return as_bool_value(args.get(key, default))


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [x for x in str(value).replace(",", " ").split() if x]


def close(key, expected):
    return math.isclose(as_float(key, expected), float(expected), rel_tol=0.0, abs_tol=1e-12)


bad = []
model = str(args.get("model", ""))
if not model.endswith("gcs-yolo-lane-s-q12-k56-short-local-refine-v3.yaml"):
    bad.append(f"model={model!r}")

if not close("gcs_short_local_refine", 0.05):
    bad.append(f"gcs_short_local_refine={args.get('gcs_short_local_refine')!r}, expected 0.05")
if as_int("gcs_short_local_refine_visible_thr", 10) != 10:
    bad.append(f"gcs_short_local_refine_visible_thr={args.get('gcs_short_local_refine_visible_thr')!r}, expected 10")
if as_int("gcs_short_local_refine_gt_min_lanes", 4) != 4:
    bad.append(f"gcs_short_local_refine_gt_min_lanes={args.get('gcs_short_local_refine_gt_min_lanes')!r}, expected 4")
if not close("gcs_short_local_refine_beta_px", 3.0):
    bad.append(f"gcs_short_local_refine_beta_px={args.get('gcs_short_local_refine_beta_px')!r}, expected 3.0")
if not close("gcs_short_local_refine_max_delta_px", 60.0):
    bad.append(f"gcs_short_local_refine_max_delta_px={args.get('gcs_short_local_refine_max_delta_px')!r}, expected 60.0")
if not as_bool("gcs_short_local_refine_window_search", False):
    bad.append("gcs_short_local_refine_window_search must be true")
if not close("gcs_short_local_refine_window_radius_px", 60.0):
    bad.append(
        f"gcs_short_local_refine_window_radius_px={args.get('gcs_short_local_refine_window_radius_px')!r}, expected 60.0"
    )
if not close("gcs_short_local_refine_window_step_px", 20.0):
    bad.append(
        f"gcs_short_local_refine_window_step_px={args.get('gcs_short_local_refine_window_step_px')!r}, expected 20.0"
    )
if not as_bool("gcs_short_local_refine_freeze_base", False):
    bad.append("gcs_short_local_refine_freeze_base must be true")
if not as_bool("gcs_short_local_refine_identity_guard", False):
    bad.append("gcs_short_local_refine_identity_guard must be true")
if not close("gcs_short_local_refine_identity_thr_px", 20.0):
    bad.append(
        f"gcs_short_local_refine_identity_thr_px={args.get('gcs_short_local_refine_identity_thr_px')!r}, expected 20.0"
    )
if not close("gcs_short_local_refine_nearmiss_thr_px", 80.0):
    bad.append(
        f"gcs_short_local_refine_nearmiss_thr_px={args.get('gcs_short_local_refine_nearmiss_thr_px')!r}, expected 80.0"
    )
if not close("gcs_short_local_refine_identity_weight", 1.0):
    bad.append(
        f"gcs_short_local_refine_identity_weight={args.get('gcs_short_local_refine_identity_weight')!r}, expected 1.0"
    )
if not close("gcs_short_local_refine_nearmiss_weight", 1.0):
    bad.append(
        f"gcs_short_local_refine_nearmiss_weight={args.get('gcs_short_local_refine_nearmiss_weight')!r}, expected 1.0"
    )

expected_zero = (
    "gcs_query_count_ce",
    "gcs_query_quality",
    "gcs_query_extent",
    "gcs_count",
    "gcs_count_under5",
    "gcs_count_boundary",
    "gcs_role_contain",
    "gcs_q24_event_contain",
    "gcs_q24_event_score_calib",
)
for key in expected_zero:
    if not close(key, 0.0):
        bad.append(f"{key}={args.get(key)!r}, expected 0.0")

if not close("gcs_short_geom", 1.0):
    bad.append(f"gcs_short_geom={args.get('gcs_short_geom')!r}, expected inherited env30 value 1.0")
if as_list(args.get("gcs_official_extent_decode_modes")) != ["none"]:
    bad.append(f"gcs_official_extent_decode_modes={args.get('gcs_official_extent_decode_modes')!r}, expected ['none']")
if as_list(args.get("gcs_official_count_modes")) != ["score_sum"]:
    bad.append(f"gcs_official_count_modes={args.get('gcs_official_count_modes')!r}, expected ['score_sum']")
if as_bool("gcs_official_count_aware_topk", False):
    bad.append("gcs_official_count_aware_topk must be false")

decode_extent = bool(decode.get("extent_decode", False))
decode_extent_mode = str(decode.get("extent_decode_mode", "none") or "none")
if decode_extent or decode_extent_mode != "none":
    bad.append(f"decode extent must be disabled, got extent_decode={decode_extent}, mode={decode_extent_mode!r}")
if as_bool_value(decode.get("count_aware_topk", False)):
    bad.append("decode count_aware_topk must be false")
if str(decode.get("count_mode", "score_sum") or "score_sum") != "score_sum":
    bad.append(f"decode count_mode={decode.get('count_mode')!r}, expected score_sum")

if bad:
    raise SystemExit("Q12 short local x-refine v3 contract check failed: " + "; ".join(bad))

print("Q12 short local x-refine v3 contract check passed: window/freeze/identity active; Count/Quality/Extent/count-aware/TEST paths disabled.")
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

run_raw_refined_gate() {
  local split_name="$1"
  local split="$2"
  local gt_json="$3"
  local save_dir="${RUN_DIR}/short_local_refine_window_v3_gate_${split_name}_official_best_decode"
  ensure_empty_or_overwrite "${save_dir}"
  python tools/diagnose_tusimple_raw_q12_filters.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split "${split}" \
    --gt-json "${gt_json}" \
    --weights "${WEIGHTS}" \
    --decode-yaml "${DECODE_YAML}" \
    --imgsz 544 960 \
    --pool-max-det "${POOL_MAX_DET}" \
    --match-thr-px 20 \
    --match-min-overlap 3 \
    --raw-match-thrs 20 30 40 \
    --short-visible-max 10 \
    --max-images "${MAX_IMAGES}" \
    --warmup "${WARMUP}" \
    --device "${DEVICE}" \
    "${HALF_ARGS[@]}" \
    --save-dir "${save_dir}"
}

echo "Q12/env30 short local x-refine v3 gate: run=${RUN_NAME}, tests=${RUN_TESTS}"
echo "TEST remains closed; running official-val and train-side coarse/refined geometry diagnostics."

run_raw_refined_gate val val "${GT_JSON}"
run_raw_refined_gate train0601 train "${TRAIN0601_GT_JSON}"
run_raw_refined_gate train0531 train "${TRAIN0531_GT_JSON}"

python - \
  "${OFFICIAL_BEST_SWEEP_SUMMARY}" \
  "${RUN_DIR}/short_local_refine_window_v3_gate_val_official_best_decode/raw_q12_filter_summary.json" \
  "${RUN_DIR}/short_local_refine_window_v3_gate_train0601_official_best_decode/raw_q12_filter_summary.json" \
  "${RUN_DIR}/short_local_refine_window_v3_gate_train0531_official_best_decode/raw_q12_filter_summary.json" \
  "${ENV30_OFFICIAL_VAL_ACC}" \
  "${VAL_SHORT_GT5_TARGET}" \
  "${TRAIN0601_SHORT_GT5_TARGET}" \
  "${TRAIN0601_SHORT_GT4_TARGET}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, val_path, train0601_path, train0531_path, env30_acc_s, val_gt5_target_s, train0601_gt5_target_s, train0601_gt4_target_s = sys.argv[1:]
env30_acc = float(env30_acc_s)
val_gt5_target = int(val_gt5_target_s)
train0601_gt5_target = int(train0601_gt5_target_s)
train0601_gt4_target = int(train0601_gt4_target_s)


def load_json(path_s):
    return json.loads(Path(path_s).read_text(encoding="utf-8"))


def metric_acc(sweep_summary):
    best = sweep_summary.get("best", {})
    if "official_acc" in best:
        return float(best["official_acc"])
    metrics = best.get("metrics", {})
    if "official_acc" in metrics:
        return float(metrics["official_acc"])
    raise SystemExit(f"Cannot find official_acc in {summary_path}")


def refined_hit_count(data, group):
    return int(data.get("coarse_refined", {}).get(group, {}).get("refined", {}).get("has_match_20px", 0) or 0)


def gain_loss_counts(data, group):
    item = data.get("coarse_refined", {}).get(group, {})
    return int(item.get("gain20_count", 0) or 0), int(item.get("loss20_count", 0) or 0)


summary = load_json(summary_path)
val_data = load_json(val_path)
train0601_data = load_json(train0601_path)
train0531_data = load_json(train0531_path)
failures = []

acc = metric_acc(summary)
if acc < env30_acc:
    failures.append(f"official-val ACC {acc:.6f} < env30 {env30_acc:.6f}")

checks = [
    ("official-val short_gt5 refined hit20", refined_hit_count(val_data, "short_gt5"), val_gt5_target),
    ("train0601 short_gt5 refined hit20", refined_hit_count(train0601_data, "short_gt5"), train0601_gt5_target),
    ("train0601 short_gt4 refined hit20", refined_hit_count(train0601_data, "short_gt4"), train0601_gt4_target),
]
for name, value, target in checks:
    if value < target:
        failures.append(f"{name} {value} < target {target}")

for label, data in (("official-val", val_data), ("train0601", train0601_data), ("train0531", train0531_data)):
    for group in ("short_gt4", "short_gt5"):
        gain20, loss20 = gain_loss_counts(data, group)
        if loss20 > gain20:
            failures.append(f"{label} {group} loss20 {loss20} > gain20 {gain20}")

path_by_label = {
    "official-val": val_path,
    "train0601": train0601_path,
    "train0531": train0531_path,
}
for label, data in (("official-val", val_data), ("train0601", train0601_data), ("train0531", train0531_data)):
    cr = data.get("coarse_refined", {})
    print(f"\n[{label}] {path_by_label[label]}")
    if not cr.get("has_short_refined_points"):
        print("  missing pred_short_refined_points: this checkpoint is not a v3 short-local-refine model")
        continue
    for group in ("short_gt4", "short_gt5"):
        item = cr.get(group, {})
        coarse = item.get("coarse", {})
        refined = item.get("refined", {})
        print(
            "  {group}: coarse_hit20={ch20} refined_hit20={rh20} "
            "gain20={gain} loss20={loss} refined_p90={p90}".format(
                group=group,
                ch20=coarse.get("has_match_20px"),
                rh20=refined.get("has_match_20px"),
                gain=item.get("gain20_count"),
                loss=item.get("loss20_count"),
                p90=refined.get("best_ape_px_p90"),
            )
        )

print(f"\n[official-val] ACC={acc:.6f}, env30_floor={env30_acc:.6f}")
print("[gate note] GT3/GT4 false-extra is not_checked by this raw/refined gate; inspect official-val sweep/count confusion before promotion.")
if failures:
    print("\nV3 gate failed:")
    for item in failures:
        print(f"  - {item}")
    raise SystemExit(2)
print("\nV3 gate passed.")
PY
