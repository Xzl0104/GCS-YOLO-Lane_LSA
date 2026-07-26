#!/usr/bin/env bash
set -euo pipefail

# Full reporting protocol for the default-off Q12/env30 short local x-refine probe.
# Selection is official-val only. TEST results are reporting-only from each
# checkpoint's post-train official-val-selected decode.

export RUN_NAME="${RUN_NAME:-query_short_local_refine_env30_full_protocol_v1}"
export PROJECT="${PROJECT:-runs/gcs_lane}"
export MODEL="${MODEL:-ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-short-local-refine.yaml}"
export DATA="${DATA:-data/tusimple_gcs_fixed_y_960x544.yaml}"
export EPOCHS="${EPOCHS:-40}"
export RUN_TESTS="${RUN_TESTS:-1}"
export RUN_PREFLIGHT_CHECKS="${RUN_PREFLIGHT_CHECKS:-1}"
export RUN_FINAL_DIAGS="${RUN_FINAL_DIAGS:-1}"
export OVERWRITE_DIAGS="${OVERWRITE_DIAGS:-0}"

# Keep this experiment isolated from Count Head, Quality Head, query extent,
# and count-aware decode. The env30 parent script still supplies
# gcs_short_geom=1.0 and boundary-pseudo defaults.
export OFFICIAL_EXTENT_DECODE_MODES="${OFFICIAL_EXTENT_DECODE_MODES:-none}"
export SWEEP_EXTENT_DECODE_MODES="${SWEEP_EXTENT_DECODE_MODES:-none}"
export OFFICIAL_COUNT_MODES="${OFFICIAL_COUNT_MODES:-score_sum}"
export SWEEP_COUNT_MODES="${SWEEP_COUNT_MODES:-score_sum}"
export COUNT_AWARE_TOPK="${COUNT_AWARE_TOPK:-0}"
export SWEEP_COUNT_AWARE_TOPK="${SWEEP_COUNT_AWARE_TOPK:-0}"
export COUNT_AWARE_EXTRA_MARGINS="${COUNT_AWARE_EXTRA_MARGINS:-0}"
export SWEEP_COUNT_AWARE_EXTRA_MARGINS="${SWEEP_COUNT_AWARE_EXTRA_MARGINS:-0}"
export VALID_BEFORE_MAXDET="${VALID_BEFORE_MAXDET:-1}"
export GCS_BOUNDARY_PSEUDO_NEG="${GCS_BOUNDARY_PSEUDO_NEG:-0.02}"

# Pin the full threshold grid here so server-local parent-script edits cannot
# silently narrow the training-time official_best or post-train sweep protocol.
export OFFICIAL_CONFS="${OFFICIAL_CONFS:-0.001 0.003 0.005 0.008 0.01 0.02}"
export OFFICIAL_POINT_VALID_THRS="${OFFICIAL_POINT_VALID_THRS:-0.45 0.50 0.55 0.60}"
export OFFICIAL_NMS_DIST_PXS="${OFFICIAL_NMS_DIST_PXS:-0 18 30}"
export OFFICIAL_MAX_DETS="${OFFICIAL_MAX_DETS:-5 6 8}"
export OFFICIAL_MIN_POINTS="${OFFICIAL_MIN_POINTS:-2 3 4 5}"
export SWEEP_CONFS="${SWEEP_CONFS:-${OFFICIAL_CONFS}}"
export SWEEP_POINT_VALID_THRS="${SWEEP_POINT_VALID_THRS:-${OFFICIAL_POINT_VALID_THRS}}"
export SWEEP_NMS_DIST_PXS="${SWEEP_NMS_DIST_PXS:-${OFFICIAL_NMS_DIST_PXS}}"
export SWEEP_MAX_DETS="${SWEEP_MAX_DETS:-${OFFICIAL_MAX_DETS}}"
export SWEEP_MIN_POINTS="${SWEEP_MIN_POINTS:-${OFFICIAL_MIN_POINTS}}"

ARCHIVE_ROOT="${ARCHIVE_ROOT:-archive/TUSimple}"
GT_JSON="${GT_JSON:-runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json}"
TRAIN0601_GT_JSON="${TRAIN0601_GT_JSON:-${ARCHIVE_ROOT}/train_set/label_data_0601.json}"
TRAIN0531_GT_JSON="${TRAIN0531_GT_JSON:-${ARCHIVE_ROOT}/train_set/label_data_0531.json}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-1}"
WARMUP="${WARMUP:-20}"
MAX_IMAGES="${MAX_IMAGES:-0}"
POOL_MAX_DET="${POOL_MAX_DET:-12}"

export ARCHIVE_ROOT
export GT_JSON

RUN_DIR="${PROJECT}/${RUN_NAME}"
OFFICIAL_BEST_WEIGHTS="${RUN_DIR}/weights/official_best.pt"
BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"

DECODE_TAG="default_decode"
if [[ "$(printf '%s' "${VALID_BEFORE_MAXDET}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
  DECODE_TAG="valid_before_maxdet"
fi

export OFFICIAL_BEST_SWEEP_DIR="${OFFICIAL_BEST_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_official_best_val_sweep_${DECODE_TAG}}"
export BEST_SWEEP_DIR="${BEST_SWEEP_DIR:-${PROJECT}/${RUN_NAME}_best_val_sweep_${DECODE_TAG}}"
export OFFICIAL_BEST_TEST_DIR="${OFFICIAL_BEST_TEST_DIR:-${PROJECT}/${RUN_NAME}_official_best_test_from_val_sweep_${DECODE_TAG}}"
export BEST_TEST_DIR="${BEST_TEST_DIR:-${PROJECT}/${RUN_NAME}_best_test_from_val_sweep_${DECODE_TAG}}"
export PROTOCOL_SUMMARY="${PROTOCOL_SUMMARY:-${PROJECT}/${RUN_NAME}_official_best_and_best_test_protocol_summary.json}"
FULL_PROTOCOL_SUMMARY="${FULL_PROTOCOL_SUMMARY:-${PROJECT}/${RUN_NAME}_short_local_refine_full_protocol_summary.json}"

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

SHORT_REFINE_EXTRA_ARGS=(
  --gcs-short-local-refine "${GCS_SHORT_LOCAL_REFINE:-0.25}"
  --gcs-short-local-refine-visible-thr "${GCS_SHORT_LOCAL_REFINE_VISIBLE_THR:-10}"
  --gcs-short-local-refine-gt-min-lanes "${GCS_SHORT_LOCAL_REFINE_GT_MIN_LANES:-4}"
  --gcs-short-local-refine-beta-px "${GCS_SHORT_LOCAL_REFINE_BETA_PX:-5.0}"
  --gcs-query-count-ce 0.0
  --gcs-query-quality 0.0
  --gcs-query-extent 0.0
  --gcs-count 0.0
  --gcs-count-under5 0.0
  --gcs-count-boundary 0.0
  --gcs-role-contain 0.0
  --gcs-q24-event-contain 0.0
  --gcs-q24-event-score-calib 0.0
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} ${SHORT_REFINE_EXTRA_ARGS[*]}"
else
  export EXTRA_TRAIN_ARGS="${SHORT_REFINE_EXTRA_ARGS[*]}"
fi

run_preflight_checks() {
  python - "${MODEL}" <<'PY'
import os
import sys
from pathlib import Path

from ultralytics.nn.modules import GCSLaneHead
from ultralytics.nn.tasks import GCSLaneModel

model_path = Path(sys.argv[1])
bad: list[str] = []

if not model_path.exists():
    bad.append(f"missing model YAML: {model_path}")
else:
    model = GCSLaneModel(str(model_path), nc=1, verbose=False)
    head = model.model[-1]
    if not isinstance(head, GCSLaneHead):
        bad.append(f"{model_path} did not build a GCSLaneHead")
    else:
        if int(getattr(head, "num_queries", -1)) != 12:
            bad.append(f"num_queries={getattr(head, 'num_queries', None)!r}, expected 12")
        if int(getattr(head, "num_points", -1)) != 56:
            bad.append(f"num_points={getattr(head, 'num_points', None)!r}, expected 56")
        if str(getattr(head, "gcs_mode", "")) != "query":
            bad.append(f"gcs_mode={getattr(head, 'gcs_mode', None)!r}, expected query")
        if str(getattr(head, "point_mode", "")) != "fixed_y":
            bad.append(f"point_mode={getattr(head, 'point_mode', None)!r}, expected fixed_y")
        if not bool(getattr(head, "query_short_local_refine_head", False)):
            bad.append("query_short_local_refine_head must be enabled")
        if bool(getattr(head, "query_count_head", False)):
            bad.append("query_count_head must stay disabled")
        if bool(getattr(head, "query_quality_head", False)):
            bad.append("query_quality_head must stay disabled")
        if bool(getattr(head, "query_extent_head", False)):
            bad.append("query_extent_head must stay disabled")


def is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str) -> list[str]:
    return [x for x in str(os.environ.get(name, "")).replace(",", " ").split() if x]


if env_list("OFFICIAL_EXTENT_DECODE_MODES") != ["none"]:
    bad.append(f"OFFICIAL_EXTENT_DECODE_MODES={env_list('OFFICIAL_EXTENT_DECODE_MODES')!r}, expected ['none']")
if env_list("SWEEP_EXTENT_DECODE_MODES") != ["none"]:
    bad.append(f"SWEEP_EXTENT_DECODE_MODES={env_list('SWEEP_EXTENT_DECODE_MODES')!r}, expected ['none']")
if env_list("OFFICIAL_COUNT_MODES") != ["score_sum"]:
    bad.append(f"OFFICIAL_COUNT_MODES={env_list('OFFICIAL_COUNT_MODES')!r}, expected ['score_sum']")
if env_list("SWEEP_COUNT_MODES") != ["score_sum"]:
    bad.append(f"SWEEP_COUNT_MODES={env_list('SWEEP_COUNT_MODES')!r}, expected ['score_sum']")
if is_true(os.environ.get("COUNT_AWARE_TOPK", "0")):
    bad.append("COUNT_AWARE_TOPK must stay false")
if is_true(os.environ.get("SWEEP_COUNT_AWARE_TOPK", "0")):
    bad.append("SWEEP_COUNT_AWARE_TOPK must stay false")

if float(os.environ.get("GCS_SHORT_LOCAL_REFINE", "0.25")) <= 0.0:
    bad.append("GCS_SHORT_LOCAL_REFINE must be > 0")
if int(os.environ.get("GCS_SHORT_LOCAL_REFINE_VISIBLE_THR", "10")) != 10:
    bad.append("GCS_SHORT_LOCAL_REFINE_VISIBLE_THR must be 10")
if int(os.environ.get("GCS_SHORT_LOCAL_REFINE_GT_MIN_LANES", "4")) != 4:
    bad.append("GCS_SHORT_LOCAL_REFINE_GT_MIN_LANES must be 4")
if abs(float(os.environ.get("GCS_SHORT_LOCAL_REFINE_BETA_PX", "5.0")) - 5.0) > 1e-12:
    bad.append("GCS_SHORT_LOCAL_REFINE_BETA_PX must be 5.0")

if bad:
    raise SystemExit("short-local-refine full-protocol preflight failed: " + "; ".join(bad))

print(
    "preflight ok: Q12/K56 query_short_local_refine_head=True; "
    "query_count_head/query_quality_head/query_extent_head/count-aware/extent-decode all disabled."
)
PY
}

ensure_empty_or_overwrite() {
  local dir="$1"
  if [[ -e "${dir}" ]] && ! is_true "${OVERWRITE_DIAGS}"; then
    echo "Diagnostic directory already exists: ${dir}" >&2
    echo "Set OVERWRITE_DIAGS=1 or choose a new RUN_NAME/diagnostic path." >&2
    exit 2
  fi
}

make_decode_yaml_from_sweep() {
  local label="$1"
  local sweep_summary="$2"
  local output_yaml="$3"

  if [[ ! -f "${sweep_summary}" ]]; then
    echo "Missing ${label} sweep summary: ${sweep_summary}" >&2
    exit 2
  fi

  python - "${label}" "${sweep_summary}" "${output_yaml}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

from ultralytics.models.gcs.decode_summary import query_decode_cfg, validate_decode_yaml_for_model

label, summary_path, output_yaml = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
best = dict(summary["best"])
cfg = query_decode_cfg(best)
validate_decode_yaml_for_model(cfg, model_mode="query")

bad: list[str] = []
if bool(cfg.get("extent_decode", False)) or str(cfg.get("extent_decode_mode", "none")) != "none":
    bad.append(f"{label}: extent decode must be disabled, got {cfg.get('extent_decode')}/{cfg.get('extent_decode_mode')}")
if bool(cfg.get("count_aware_topk", False)):
    bad.append(f"{label}: count-aware top-k must be disabled")
if str(cfg.get("count_mode", "score_sum")) != "score_sum":
    bad.append(f"{label}: count_mode must be score_sum, got {cfg.get('count_mode')!r}")
if bad:
    raise SystemExit("; ".join(bad))

payload = {
    "source": "post_train_official_val_sweep_best",
    "checkpoint_label": label,
    "sweep_summary": str(Path(summary_path).resolve()),
    "decode": cfg,
    "val_best": best,
}
out = Path(output_yaml)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
print(f"wrote {label} sweep-selected decode yaml: {out}")
PY
}

run_raw_refined_diag() {
  local label="$1"
  local weights="$2"
  local decode_yaml="$3"
  local split_name="$4"
  local split="$5"
  local gt_json="$6"
  local save_dir="${RUN_DIR}/short_local_refine_full_protocol_${label}_${split_name}_val_sweep_decode"

  ensure_empty_or_overwrite "${save_dir}"

  local half_args=()
  if is_true "${HALF}"; then
    half_args=(--half)
  fi

  python tools/diagnose_tusimple_raw_q12_filters.py \
    --archive-root "${ARCHIVE_ROOT}" \
    --split "${split}" \
    --gt-json "${gt_json}" \
    --weights "${weights}" \
    --decode-yaml "${decode_yaml}" \
    --imgsz 544 960 \
    --pool-max-det "${POOL_MAX_DET}" \
    --match-thr-px 20 \
    --match-min-overlap 3 \
    --raw-match-thrs 20 30 40 \
    --short-visible-max 10 \
    --max-images "${MAX_IMAGES}" \
    --warmup "${WARMUP}" \
    --device "${DEVICE}" \
    "${half_args[@]}" \
    --save-dir "${save_dir}"
}

write_full_protocol_summary() {
  python - \
    "${RUN_NAME}" \
    "${RUN_DIR}" \
    "${PROTOCOL_SUMMARY}" \
    "${OFFICIAL_BEST_SWEEP_DIR}" "${OFFICIAL_BEST_TEST_DIR}" \
    "${BEST_SWEEP_DIR}" "${BEST_TEST_DIR}" \
    "${RUN_DIR}/weights/official_best_from_posttrain_sweep_decode.yaml" \
    "${RUN_DIR}/weights/best_from_posttrain_sweep_decode.yaml" \
    "${FULL_PROTOCOL_SUMMARY}" \
    "${RUN_TESTS}" \
    "${RUN_FINAL_DIAGS}" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

import yaml

(
    run_name,
    run_dir,
    protocol_summary,
    official_best_sweep_dir,
    official_best_test_dir,
    best_sweep_dir,
    best_test_dir,
    official_best_decode_yaml,
    best_decode_yaml,
    full_summary,
    run_tests,
    run_final_diags,
) = sys.argv[1:]

run_dir_p = Path(run_dir)


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"missing": str(p)}
    return json.loads(p.read_text(encoding="utf-8"))


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"missing": str(p)}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def as_float(data: dict, key: str, default=0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{key} must be numeric, got {data.get(key)!r}") from exc


def as_int(data: dict, key: str, default=0) -> int:
    try:
        return int(data.get(key, default))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{key} must be integer, got {data.get(key)!r}") from exc


def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [x for x in str(value).replace(",", " ").split() if x]


def sweep_best(sweep_dir: str | Path) -> dict:
    return load_json(Path(sweep_dir) / "tusimple_official_sweep_summary.json").get("best", {})


def test_metrics(test_dir: str | Path) -> dict:
    return load_json(Path(test_dir) / "tusimple_official_summary.json").get("metrics", {})


def diag_summary(label: str, split: str) -> dict:
    path = run_dir_p / f"short_local_refine_full_protocol_{label}_{split}_val_sweep_decode/raw_q12_filter_summary.json"
    return load_json(path)


args = load_yaml(run_dir_p / "args.yaml")
official_best_decode = load_yaml(run_dir_p / "weights/official_best_decode.yaml")
posttrain_official_best_decode = load_yaml(official_best_decode_yaml)
posttrain_best_decode = load_yaml(best_decode_yaml)

bad: list[str] = []
required_paths = [
    run_dir_p / "args.yaml",
    run_dir_p / "weights/official_best.pt",
    run_dir_p / "weights/best.pt",
    run_dir_p / "weights/official_best_decode.yaml",
    Path(official_best_sweep_dir) / "tusimple_official_sweep_summary.json",
    Path(best_sweep_dir) / "tusimple_official_sweep_summary.json",
    Path(official_best_decode_yaml),
    Path(best_decode_yaml),
]
if as_bool(run_tests):
    required_paths.extend(
        [
            Path(official_best_test_dir) / "tusimple_official_summary.json",
            Path(best_test_dir) / "tusimple_official_summary.json",
        ]
    )
if as_bool(run_final_diags):
    for label in ("official_best", "best"):
        for split in ("val", "train0601", "train0531"):
            required_paths.append(
                run_dir_p
                / f"short_local_refine_full_protocol_{label}_{split}_val_sweep_decode/raw_q12_filter_summary.json"
            )
for path in required_paths:
    if not Path(path).exists():
        bad.append(f"missing required artifact: {path}")

model = str(args.get("model", ""))
if not model.endswith("gcs-yolo-lane-s-q12-k56-short-local-refine.yaml"):
    bad.append(f"model={model!r}")
if as_float(args, "gcs_short_local_refine", 0.0) <= 0.0:
    bad.append(f"gcs_short_local_refine={args.get('gcs_short_local_refine')!r}, expected > 0")
if as_int(args, "gcs_short_local_refine_visible_thr", 10) != 10:
    bad.append(f"gcs_short_local_refine_visible_thr={args.get('gcs_short_local_refine_visible_thr')!r}, expected 10")
if as_int(args, "gcs_short_local_refine_gt_min_lanes", 4) != 4:
    bad.append(f"gcs_short_local_refine_gt_min_lanes={args.get('gcs_short_local_refine_gt_min_lanes')!r}, expected 4")
if not math.isclose(as_float(args, "gcs_short_local_refine_beta_px", 5.0), 5.0, rel_tol=0.0, abs_tol=1e-12):
    bad.append(f"gcs_short_local_refine_beta_px={args.get('gcs_short_local_refine_beta_px')!r}, expected 5.0")

for key in (
    "gcs_query_count_ce",
    "gcs_query_quality",
    "gcs_query_extent",
    "gcs_count",
    "gcs_count_under5",
    "gcs_count_boundary",
    "gcs_role_contain",
    "gcs_q24_event_contain",
    "gcs_q24_event_score_calib",
):
    if not math.isclose(as_float(args, key, 0.0), 0.0, rel_tol=0.0, abs_tol=1e-12):
        bad.append(f"{key}={args.get(key)!r}, expected 0.0")

if not math.isclose(as_float(args, "gcs_short_geom", 0.0), 1.0, rel_tol=0.0, abs_tol=1e-12):
    bad.append(f"gcs_short_geom={args.get('gcs_short_geom')!r}, expected inherited env30 value 1.0")
if not math.isclose(
    as_float(args, "gcs_boundary_pseudo_neg", 0.0),
    float(os.environ.get("GCS_BOUNDARY_PSEUDO_NEG", "0.02")),
    rel_tol=0.0,
    abs_tol=1e-12,
):
    bad.append(
        "gcs_boundary_pseudo_neg="
        f"{args.get('gcs_boundary_pseudo_neg')!r}, expected {os.environ.get('GCS_BOUNDARY_PSEUDO_NEG', '0.02')}"
    )
if as_list(args.get("gcs_official_extent_decode_modes")) != ["none"]:
    bad.append(f"gcs_official_extent_decode_modes={args.get('gcs_official_extent_decode_modes')!r}, expected ['none']")
if as_list(args.get("gcs_official_count_modes")) != ["score_sum"]:
    bad.append(f"gcs_official_count_modes={args.get('gcs_official_count_modes')!r}, expected ['score_sum']")
if as_bool(args.get("gcs_official_count_aware_topk", False)):
    bad.append("gcs_official_count_aware_topk must be false")

for label, yaml_data in (
    ("training_time_official_best_decode", official_best_decode.get("decode", official_best_decode)),
    ("posttrain_official_best_decode", posttrain_official_best_decode.get("decode", {})),
    ("posttrain_best_decode", posttrain_best_decode.get("decode", {})),
):
    if not yaml_data:
        bad.append(f"{label} missing")
        continue
    if bool(yaml_data.get("extent_decode", False)) or str(yaml_data.get("extent_decode_mode", "none")) != "none":
        bad.append(f"{label}: extent decode must be disabled")
    if as_bool(yaml_data.get("count_aware_topk", False)):
        bad.append(f"{label}: count-aware top-k must be disabled")
    if str(yaml_data.get("count_mode", "score_sum")) != "score_sum":
        bad.append(f"{label}: count_mode must be score_sum")

for label, row in (("official_best_val_best", sweep_best(official_best_sweep_dir)), ("best_val_best", sweep_best(best_sweep_dir))):
    if not row:
        bad.append(f"{label} missing")
        continue
    if bool(row.get("extent_decode", False)) or str(row.get("extent_decode_mode", "none")) != "none":
        bad.append(f"{label}: extent decode must be disabled")
    if as_bool(row.get("count_aware_topk", False)):
        bad.append(f"{label}: count-aware top-k must be disabled")
    if str(row.get("count_mode", "score_sum")) != "score_sum":
        bad.append(f"{label}: count_mode must be score_sum")

if bad:
    raise SystemExit("short-local-refine full protocol contract check failed: " + "; ".join(bad))

diagnostics = {}
if as_bool(run_final_diags):
    for label in ("official_best", "best"):
        diagnostics[label] = {
            "official_val": diag_summary(label, "val"),
            "train0601": diag_summary(label, "train0601"),
            "train0531": diag_summary(label, "train0531"),
        }

summary = {
    "run_name": run_name,
    "selection_split": "official-val",
    "test_usage": "reporting_only_from_val_selected_decode" if as_bool(run_tests) else "not_run",
    "do_not_select_from_test": True,
    "contract_check": "passed",
    "params": {
        "model": model,
        "gcs_short_local_refine": as_float(args, "gcs_short_local_refine", 0.0),
        "gcs_short_local_refine_visible_thr": as_int(args, "gcs_short_local_refine_visible_thr", 10),
        "gcs_short_local_refine_gt_min_lanes": as_int(args, "gcs_short_local_refine_gt_min_lanes", 4),
        "gcs_short_local_refine_beta_px": as_float(args, "gcs_short_local_refine_beta_px", 5.0),
        "gcs_short_geom": as_float(args, "gcs_short_geom", 0.0),
        "gcs_boundary_pseudo_neg": as_float(args, "gcs_boundary_pseudo_neg", 0.0),
        "extent_decode_modes": as_list(args.get("gcs_official_extent_decode_modes")),
        "count_modes": as_list(args.get("gcs_official_count_modes")),
        "count_aware_topk": as_bool(args.get("gcs_official_count_aware_topk", False)),
    },
    "training_time_official_best_decode_yaml": str((run_dir_p / "weights/official_best_decode.yaml").resolve()),
    "parent_protocol_summary": str(Path(protocol_summary).resolve()),
    "official_best": {
        "weights": str((run_dir_p / "weights/official_best.pt").resolve()),
        "posttrain_val_sweep": str((Path(official_best_sweep_dir) / "tusimple_official_sweep_summary.json").resolve()),
        "posttrain_val_best": sweep_best(official_best_sweep_dir),
        "posttrain_decode_yaml": str(Path(official_best_decode_yaml).resolve()),
        "test_summary": str((Path(official_best_test_dir) / "tusimple_official_summary.json").resolve()),
        "test_metrics": test_metrics(official_best_test_dir) if as_bool(run_tests) else {},
    },
    "best": {
        "weights": str((run_dir_p / "weights/best.pt").resolve()),
        "posttrain_val_sweep": str((Path(best_sweep_dir) / "tusimple_official_sweep_summary.json").resolve()),
        "posttrain_val_best": sweep_best(best_sweep_dir),
        "posttrain_decode_yaml": str(Path(best_decode_yaml).resolve()),
        "test_summary": str((Path(best_test_dir) / "tusimple_official_summary.json").resolve()),
        "test_metrics": test_metrics(best_test_dir) if as_bool(run_tests) else {},
    },
    "diagnostics": diagnostics,
}

out = Path(full_summary)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(
    {
        "summary": str(out),
        "official_best_val_acc": summary["official_best"]["posttrain_val_best"].get("official_acc"),
        "best_val_acc": summary["best"]["posttrain_val_best"].get("official_acc"),
        "official_best_test_acc": summary["official_best"]["test_metrics"].get("official_acc"),
        "best_test_acc": summary["best"]["test_metrics"].get("official_acc"),
        "contract_check": "passed",
    },
    indent=2,
))
PY
}

echo "Q12/env30 short local x-refine full protocol"
echo "run=${RUN_NAME}"
echo "model=${MODEL}"
echo "epochs=${EPOCHS}"
echo "RUN_TESTS=${RUN_TESTS} (TEST is reporting-only from official-val-selected decode)"
echo "official extent modes=${OFFICIAL_EXTENT_DECODE_MODES}; sweep extent modes=${SWEEP_EXTENT_DECODE_MODES}"
echo "official count modes=${OFFICIAL_COUNT_MODES}; count-aware=${COUNT_AWARE_TOPK}/${SWEEP_COUNT_AWARE_TOPK}"
echo "short local refine: gain=${GCS_SHORT_LOCAL_REFINE:-0.25}, visible_thr=${GCS_SHORT_LOCAL_REFINE_VISIBLE_THR:-10}, gt_min_lanes=${GCS_SHORT_LOCAL_REFINE_GT_MIN_LANES:-4}, beta_px=${GCS_SHORT_LOCAL_REFINE_BETA_PX:-5.0}"
echo "inherited env30 parent: gcs_short_geom=1.0, gcs_boundary_pseudo_neg=${GCS_BOUNDARY_PSEUDO_NEG}"

if is_true "${RUN_PREFLIGHT_CHECKS}"; then
  run_preflight_checks
fi

bash scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh

make_decode_yaml_from_sweep \
  "official_best" \
  "${OFFICIAL_BEST_SWEEP_DIR}/tusimple_official_sweep_summary.json" \
  "${RUN_DIR}/weights/official_best_from_posttrain_sweep_decode.yaml"
make_decode_yaml_from_sweep \
  "best" \
  "${BEST_SWEEP_DIR}/tusimple_official_sweep_summary.json" \
  "${RUN_DIR}/weights/best_from_posttrain_sweep_decode.yaml"

if is_true "${RUN_FINAL_DIAGS}"; then
  echo "Running raw/refined geometry diagnostics on official-val, train0601, and train0531. No TEST oracle diagnostics are run."
  run_raw_refined_diag "official_best" "${OFFICIAL_BEST_WEIGHTS}" "${RUN_DIR}/weights/official_best_from_posttrain_sweep_decode.yaml" val val "${GT_JSON}"
  run_raw_refined_diag "official_best" "${OFFICIAL_BEST_WEIGHTS}" "${RUN_DIR}/weights/official_best_from_posttrain_sweep_decode.yaml" train0601 train "${TRAIN0601_GT_JSON}"
  run_raw_refined_diag "official_best" "${OFFICIAL_BEST_WEIGHTS}" "${RUN_DIR}/weights/official_best_from_posttrain_sweep_decode.yaml" train0531 train "${TRAIN0531_GT_JSON}"
  run_raw_refined_diag "best" "${BEST_WEIGHTS}" "${RUN_DIR}/weights/best_from_posttrain_sweep_decode.yaml" val val "${GT_JSON}"
  run_raw_refined_diag "best" "${BEST_WEIGHTS}" "${RUN_DIR}/weights/best_from_posttrain_sweep_decode.yaml" train0601 train "${TRAIN0601_GT_JSON}"
  run_raw_refined_diag "best" "${BEST_WEIGHTS}" "${RUN_DIR}/weights/best_from_posttrain_sweep_decode.yaml" train0531 train "${TRAIN0531_GT_JSON}"
else
  echo "RUN_FINAL_DIAGS=0: skipping raw/refined geometry diagnostics." >&2
fi

write_full_protocol_summary
