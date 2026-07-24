from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE_WIDTH = 1280.0
FIXED_Y_DESC = np.arange(710.0, 150.0, -10.0, dtype=np.float32)
PROTECTED_Q12 = tuple(range(12))

DEFAULT_VAL_GT = (
    ROOT
    / "runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/"
    / "tusimple_official_val_363_folder_aware_seed20260602.json"
)
DEFAULT_TRAIN0601_JSON = ROOT / "archive/TUSimple/train_set/label_data_0601.json"
DEFAULT_TRAIN0531_JSON = ROOT / "archive/TUSimple/train_set/label_data_0531.json"
DEFAULT_VAL_DIAG_CANDIDATES = [
    ROOT / ".tmp/env30_analysis/raw_gt_lane_diagnostics.csv",
    ROOT
    / "runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1_official_best_val_raw_q12_binding_diag_v1/"
    / "raw_gt_lane_diagnostics.csv",
]
DEFAULT_TRAIN0601_DIAG_CANDIDATES = [
    ROOT / ".tmp/env30_train_raw/train0601/raw_gt_lane_diagnostics.csv",
    ROOT / "runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/raw_q12_filters_train0601_official_best_decode/raw_gt_lane_diagnostics.csv",
]
DEFAULT_TRAIN0531_DIAG_CANDIDATES = [
    ROOT / ".tmp/env30_train_raw/train0531/raw_gt_lane_diagnostics.csv",
    ROOT / "runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/raw_q12_filters_train0531_official_best_decode/raw_gt_lane_diagnostics.csv",
]
DEFAULT_IMPACT_CSV_CANDIDATES = [
    ROOT / ".tmp/env30_tail_positive_triage/official_impact_tail_triage.csv",
    ROOT / ".tmp/goal_9700_image_appearance/input/env30_official_impact_tail_triage.csv",
]
DEFAULT_SAVE_DIR = ROOT / ".tmp/q20_dualbank_static_gate"
DEFAULT_PUBLISH_BANK = ROOT / "data/gcs_reference_banks/q20_dualbank_env30_best.json"
DEFAULT_Q24_SAVE_DIR = ROOT / ".tmp/q24_protected_static_gate"
DEFAULT_Q24_PUBLISH_BANK = ROOT / "data/gcs_reference_banks/q24_protected_static_env30_best.json"


@dataclass(frozen=True)
class Lane:
    split: str
    raw_file: str
    gt_lane_id: int
    gt_count: int
    lane_position: str
    side_group: str
    visible_points_gt: int
    group: str
    x_norm: np.ndarray
    visible_mask: np.ndarray
    source_row: dict


def parse_args(
    default_num_queries: int = 20,
    default_allocations: str = "5:3,4:4,6:2",
    default_save_dir: Path = DEFAULT_SAVE_DIR,
    default_publish_bank: Path = DEFAULT_PUBLISH_BANK,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and gate a default-off QN protected static reference design. "
            "This tool uses official-val and train-side raw diagnostics only; TEST is never read."
        )
    )
    parser.add_argument("--val-diag", default=None, help="official-val raw_gt_lane_diagnostics.csv")
    parser.add_argument("--train0601-diag", default=None, help="train0601 raw_gt_lane_diagnostics.csv")
    parser.add_argument("--train0531-diag", default=None, help="train0531 raw_gt_lane_diagnostics.csv")
    parser.add_argument("--impact-csv", default=None, help="Optional official impact triage CSV.")
    parser.add_argument("--val-gt-json", default=str(DEFAULT_VAL_GT))
    parser.add_argument("--train0601-json", default=str(DEFAULT_TRAIN0601_JSON))
    parser.add_argument("--train0531-json", default=str(DEFAULT_TRAIN0531_JSON))
    parser.add_argument("--save-dir", default=str(default_save_dir))
    parser.add_argument("--num-queries", type=int, default=int(default_num_queries))
    parser.add_argument("--beam-width", type=int, default=300)
    parser.add_argument("--top-report", type=int, default=25)
    parser.add_argument("--pool-cap-per-group", type=int, default=24)
    parser.add_argument(
        "--allocations",
        default=default_allocations,
        help="Comma-separated GT5:GT4 extra-query allocations. Each allocation must sum to num_queries-12.",
    )
    parser.add_argument("--publish-bank", action="store_true", help=f"Publish passed bank to {default_publish_bank}.")
    parser.add_argument("--publish-path", default=str(default_publish_bank))
    return parser.parse_args()


def parse_allocations(text: str, total_extra_queries: int) -> list[tuple[int, int]]:
    """Parse unique GT5:GT4 extra-query allocations."""
    allocations: list[tuple[int, int]] = []
    seen = set()
    for part in str(text or "").split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid allocation {item!r}; expected GT5:GT4 such as 5:3.")
        left, right = item.split(":", 1)
        try:
            gt5_slots = int(left)
            gt4_slots = int(right)
        except ValueError as exc:
            raise ValueError(f"Invalid allocation {item!r}; slot counts must be integers.") from exc
        if gt5_slots <= 0 or gt4_slots <= 0 or gt5_slots + gt4_slots != int(total_extra_queries):
            raise ValueError(
                f"Invalid allocation {item!r}; GT5 and GT4 slots must be positive and sum to {int(total_extra_queries)}."
            )
        key = (gt5_slots, gt4_slots)
        if key in seen:
            continue
        seen.add(key)
        allocations.append(key)
    if not allocations:
        raise ValueError("At least one allocation is required.")
    return allocations


def _resolve_default(path_arg: str | None, candidates: list[Path], name: str) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
        return path
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No default {name} found. Checked: {[str(p) for p in candidates]}")


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def _has_test_path_token(value: str) -> bool:
    text = norm_raw_file(value).lower()
    parts = [part for part in text.split("/") if part]
    return "test" in parts or "test_set" in parts or "test_label" in text


def _assert_no_test_path(path: Path, name: str) -> None:
    if _has_test_path_token(str(path)):
        raise ValueError(f"{name} must not point to TEST data: {path}")


def _assert_no_test_rows(rows: list[dict], name: str) -> None:
    for idx, row in enumerate(rows, start=1):
        raw_file = norm_raw_file(row.get("raw_file", ""))
        if _has_test_path_token(raw_file):
            raise ValueError(f"{name} row {idx} uses TEST raw_file {raw_file!r}; protected static gate cannot use TEST.")


def validate_gate_inputs(
    val_diag: Path,
    train0601_diag: Path,
    train0531_diag: Path,
    impact_csv: Path,
    val_gt_json: Path,
    train0601_json: Path,
    train0531_json: Path,
) -> None:
    """Enforce official-val/train-only inputs before recording test_used=false."""
    for name, path in (
        ("val diagnostic CSV", val_diag),
        ("train0601 diagnostic CSV", train0601_diag),
        ("train0531 diagnostic CSV", train0531_diag),
        ("impact CSV", impact_csv),
        ("val GT JSON", val_gt_json),
        ("train0601 GT JSON", train0601_json),
        ("train0531 GT JSON", train0531_json),
    ):
        _assert_no_test_path(path, name)
    if not _same_file(val_gt_json, DEFAULT_VAL_GT):
        raise ValueError(f"val GT JSON must be the canonical 363-image official-val JSON: {DEFAULT_VAL_GT}")
    if train0601_json.name != "label_data_0601.json":
        raise ValueError(f"train0601 JSON must be label_data_0601.json, got {train0601_json}")
    if train0531_json.name != "label_data_0531.json":
        raise ValueError(f"train0531 JSON must be label_data_0531.json, got {train0531_json}")


def norm_raw_file(raw_file: object) -> str:
    return str(raw_file or "").replace("\\", "/").strip()


def read_json_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_gt_index(paths: list[Path]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for path in paths:
        for record in read_json_records(path):
            raw_file = norm_raw_file(record.get("raw_file", ""))
            if not raw_file:
                continue
            if raw_file in index:
                raise ValueError(f"Duplicate raw_file across GT inputs: {raw_file}")
            index[raw_file] = record
    return index


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row, raw_file=norm_raw_file(row.get("raw_file", ""))) for row in csv.DictReader(f)]


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def int_or_none(value: object) -> int | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def float_or_none(value: object) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def q12_default_references(num_points: int = 56) -> np.ndarray:
    bottom_x = np.linspace(0.05, 0.95, 12, dtype=np.float32)
    top_x = 0.5 + (bottom_x - 0.5) * 0.25
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    return (bottom_x[:, None] * (1.0 - t[None, :]) + top_x[:, None] * t[None, :]).astype(np.float32)


def lane_to_fixed_reference(record: dict, gt_lane_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    lanes = record.get("lanes", [])
    if gt_lane_id < 0 or gt_lane_id >= len(lanes):
        return None
    lane = np.asarray(lanes[gt_lane_id], dtype=np.float32)
    h_samples = np.asarray(record.get("h_samples", []), dtype=np.float32)
    if lane.shape[0] != h_samples.shape[0]:
        raise ValueError(f"Lane/h_samples length mismatch for {record.get('raw_file')} lane {gt_lane_id}")
    valid = np.isfinite(lane) & (lane >= 0.0)
    if int(valid.sum()) < 2:
        return None

    ys = h_samples[valid].astype(np.float32)
    xs = lane[valid].astype(np.float32)
    order = np.argsort(ys, kind="stable")
    ys = ys[order]
    xs = xs[order]
    unique_y, unique_idx = np.unique(ys, return_index=True)
    ys = unique_y.astype(np.float32)
    xs = xs[unique_idx].astype(np.float32)
    if ys.size < 2:
        return None

    fixed_y = FIXED_Y_DESC.astype(np.float32)
    interp_x = np.interp(fixed_y, ys, xs).astype(np.float32)
    top_slope = (xs[1] - xs[0]) / max(float(ys[1] - ys[0]), 1e-6)
    bottom_slope = (xs[-1] - xs[-2]) / max(float(ys[-1] - ys[-2]), 1e-6)
    above_top = fixed_y < ys[0]
    below_bottom = fixed_y > ys[-1]
    interp_x[above_top] = xs[0] + top_slope * (fixed_y[above_top] - ys[0])
    interp_x[below_bottom] = xs[-1] + bottom_slope * (fixed_y[below_bottom] - ys[-1])
    visible_mask = (fixed_y >= ys[0]) & (fixed_y <= ys[-1])
    x_norm = np.clip(interp_x / IMAGE_WIDTH, 0.001, 0.999).astype(np.float32)
    return x_norm, visible_mask.astype(bool)


def group_for(row: dict) -> str:
    gt_count = int(float(row.get("gt_count", 0) or 0))
    side_group = str(row.get("side_group", "")).strip()
    if gt_count == 5 and side_group in {"center", "left_side"}:
        return "gt5cl"
    if gt_count == 5 and side_group == "right_side":
        return "gt5r"
    if gt_count == 4 and side_group == "center":
        return "gt4c"
    if gt_count == 4 and side_group == "right_side":
        return "gt4r"
    if gt_count == 4:
        return "gt4other"
    return "other"


def is_primary_hard(row: dict) -> bool:
    gt_count = int(float(row.get("gt_count", 0) or 0))
    visible = int(float(row.get("visible_points_gt", 0) or 0))
    if gt_count not in {4, 5} or visible > 20:
        return False
    raw_has_match = bool_value(row.get("raw_has_match_20px", ""))
    valid_len = int_or_none(row.get("raw_best_valid_len@0.6", ""))
    valid_count = int_or_none(row.get("raw_best_valid_count@0.6", ""))
    min_points = int_or_none(row.get("min_points", "")) or 0
    recall = float_or_none(row.get("point_valid_recall@0.6", ""))
    survival_bad = False
    if valid_len is None or valid_len < min_points:
        survival_bad = True
    if valid_count is None or valid_count < min_points:
        survival_bad = True
    if recall is None or recall < 0.75:
        survival_bad = True
    return (not raw_has_match) or survival_bad


def is_normal_risk_row(row: dict) -> bool:
    gt_count = int(float(row.get("gt_count", 0) or 0))
    visible = int(float(row.get("visible_points_gt", 0) or 0))
    if gt_count not in {3, 4} or visible <= 20:
        return False
    return not is_primary_hard(row)


def is_impact_target(row: dict) -> bool:
    gt_count = int(float(row.get("gt_count", 0) or 0))
    visible = int(float(row.get("visible_points_gt", 0) or 0))
    if gt_count != 5 or visible > 20:
        return False
    return (
        bool_value(row.get("no20", ""))
        or bool_value(row.get("valid_lt_min", ""))
        or bool_value(row.get("missing_at_match_thr", ""))
    )


def lane_from_row(split: str, row: dict, gt_index: dict[str, dict]) -> Lane | None:
    raw_file = norm_raw_file(row.get("raw_file", ""))
    record = gt_index.get(raw_file)
    if record is None:
        return None
    gt_lane_id = int(float(row.get("gt_lane_id", -1) or -1))
    fixed = lane_to_fixed_reference(record, gt_lane_id)
    if fixed is None:
        return None
    x_norm, visible_mask = fixed
    return Lane(
        split=split,
        raw_file=raw_file,
        gt_lane_id=gt_lane_id,
        gt_count=int(float(row.get("gt_count", 0) or 0)),
        lane_position=str(row.get("lane_position", "")).strip(),
        side_group=str(row.get("side_group", "")).strip(),
        visible_points_gt=int(float(row.get("visible_points_gt", 0) or 0)),
        group=group_for(row),
        x_norm=x_norm,
        visible_mask=visible_mask,
        source_row=row,
    )


def dedup_lanes(lanes: list[Lane | None], include_split: bool = True) -> list[Lane]:
    out: list[Lane] = []
    seen = set()
    for lane in lanes:
        if lane is None:
            continue
        key = (lane.split, lane.raw_file, lane.gt_lane_id) if include_split else (lane.raw_file, lane.gt_lane_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(lane)
    return out


def distance_matrix(refs: np.ndarray, lanes: list[Lane]) -> np.ndarray:
    if not lanes:
        return np.zeros((0, int(refs.shape[0])), dtype=np.float32)
    lane_x = np.stack([lane.x_norm for lane in lanes], axis=0).astype(np.float32)
    mask = np.stack([lane.visible_mask.astype(np.float32) for lane in lanes], axis=0)
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    diff = np.abs(lane_x[:, None, :] - refs.astype(np.float32)[None, :, :]) * mask[:, None, :]
    return (diff.sum(axis=2) / denom * IMAGE_WIDTH).astype(np.float32)


def metrics_from_distances(distances: np.ndarray) -> dict:
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return {"count": 0, "p50": None, "p90": None, "match20": None, "matched20_count": 0}
    return {
        "count": int(finite.size),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "match20": float(np.mean(finite <= 20.0)),
        "matched20_count": int(np.sum(finite <= 20.0)),
    }


def metrics(refs: np.ndarray, lanes: list[Lane]) -> dict:
    mat = distance_matrix(refs, lanes)
    if mat.shape[0] == 0:
        out = metrics_from_distances(np.asarray([], dtype=np.float32))
        out["nearest_query_hist"] = {}
        return out
    nearest = np.nanargmin(mat, axis=1).astype(np.int64)
    distances = mat[np.arange(mat.shape[0]), nearest].astype(np.float32)
    out = metrics_from_distances(distances)
    out["nearest_query_hist"] = {str(int(q)): int((nearest == q).sum()) for q in sorted(set(nearest.tolist()))}
    return out


def count_by_group(lanes: list[Lane]) -> dict:
    out = {}
    for lane in lanes:
        key = f"GT{lane.gt_count}:{lane.group}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def lane_summary(lane: Lane) -> dict:
    return {
        "split": lane.split,
        "raw_file": lane.raw_file,
        "gt_lane_id": int(lane.gt_lane_id),
        "gt_count": int(lane.gt_count),
        "lane_position": lane.lane_position,
        "side_group": lane.side_group,
        "visible_points_gt": int(lane.visible_points_gt),
        "group": lane.group,
    }


def build_eval_sets(
    raw_rows: dict[str, list[dict]],
    impact_rows: list[dict],
    gt_indexes: dict[str, dict[str, dict]],
) -> tuple[dict[str, list[Lane]], str, bool]:
    primary_lanes = {}
    normal_lanes = {}
    for split, rows in raw_rows.items():
        lanes = [lane_from_row(split, row, gt_indexes[split]) for row in rows if is_primary_hard(row)]
        primary_lanes[split] = dedup_lanes(lanes)
        normal = [lane_from_row(split, row, gt_indexes[split]) for row in rows if is_normal_risk_row(row)]
        normal_lanes[split] = dedup_lanes(normal)

    impact_target_lanes = dedup_lanes(
        [lane_from_row("val_impact", row, gt_indexes["val"]) for row in impact_rows if is_impact_target(row)]
    )
    impact_source = "impact_csv"
    formal_impact = True
    if not impact_target_lanes:
        impact_target_lanes = [lane for lane in primary_lanes["val"] if lane.gt_count == 5]
        impact_source = "val_gt5_primary_fallback"
        formal_impact = False

    eval_sets = {
        "val_gt5_primary": [lane for lane in primary_lanes["val"] if lane.gt_count == 5],
        "val_gt4_primary": [lane for lane in primary_lanes["val"] if lane.gt_count == 4],
        "train0601_gt5_primary": [lane for lane in primary_lanes["train0601"] if lane.gt_count == 5],
        "train0601_gt4_primary": [lane for lane in primary_lanes["train0601"] if lane.gt_count == 4],
        "train0531_gt5_primary": [lane for lane in primary_lanes["train0531"] if lane.gt_count == 5],
        "train0531_gt4_primary": [lane for lane in primary_lanes["train0531"] if lane.gt_count == 4],
        "val_gt5_impact_target": impact_target_lanes,
        "val_gt3gt4_normal_risk": [lane for lane in normal_lanes["val"] if lane.gt_count in {3, 4}],
        "train0601_gt3gt4_normal_risk": [lane for lane in normal_lanes["train0601"] if lane.gt_count in {3, 4}],
        "train0531_gt3gt4_normal_risk": [lane for lane in normal_lanes["train0531"] if lane.gt_count in {3, 4}],
    }
    return eval_sets, impact_source, formal_impact


def candidate_target_score(lane: Lane, target_lanes: list[Lane]) -> float:
    if not target_lanes:
        return float("inf")
    mat = distance_matrix(np.asarray([lane.x_norm], dtype=np.float32), target_lanes)
    values = mat[:, 0]
    return float(np.percentile(values, 50) + 0.5 * np.percentile(values, 90))


def trim_candidate_pool(train_lanes: list[Lane], eval_sets: dict[str, list[Lane]], cap_per_group: int) -> list[Lane]:
    targets_by_group = {
        "gt5cl": [
            lane
            for key in ("val_gt5_primary", "train0601_gt5_primary")
            for lane in eval_sets[key]
            if lane.group == "gt5cl"
        ],
        "gt5r": [
            lane
            for key in ("val_gt5_primary", "train0601_gt5_primary")
            for lane in eval_sets[key]
            if lane.group == "gt5r"
        ],
        "gt4c": [
            lane
            for key in ("val_gt4_primary", "train0601_gt4_primary", "train0531_gt4_primary")
            for lane in eval_sets[key]
            if lane.group == "gt4c"
        ],
        "gt4r": [
            lane
            for key in ("val_gt4_primary", "train0601_gt4_primary", "train0531_gt4_primary")
            for lane in eval_sets[key]
            if lane.group == "gt4r"
        ],
        "gt4other": [
            lane
            for key in ("val_gt4_primary", "train0601_gt4_primary", "train0531_gt4_primary")
            for lane in eval_sets[key]
            if lane.group == "gt4other"
        ],
    }
    grouped: dict[str, list[Lane]] = {}
    for lane in train_lanes:
        grouped.setdefault(lane.group, []).append(lane)

    trimmed: list[Lane] = []
    for group, lanes in sorted(grouped.items()):
        ranked = sorted(
            lanes,
            key=lambda lane: (
                candidate_target_score(lane, targets_by_group.get(group, [])),
                lane.split,
                lane.raw_file,
                int(lane.gt_lane_id),
            ),
        )
        trimmed.extend(ranked[: max(1, int(cap_per_group))])
    return trimmed


def precompute(base_refs: np.ndarray, candidates: list[Lane], lanes: list[Lane]) -> dict:
    base_dist = distance_matrix(base_refs, lanes).min(axis=1).astype(np.float32) if lanes else np.asarray([], dtype=np.float32)
    cand_refs = np.stack([lane.x_norm for lane in candidates], axis=0).astype(np.float32) if candidates else np.zeros((0, 56), dtype=np.float32)
    cand_dist = distance_matrix(cand_refs, lanes).T.astype(np.float32) if lanes else np.zeros((len(candidates), 0), dtype=np.float32)
    return {"base": base_dist, "cand": cand_dist}


def distances_for(pre: dict, selected: tuple[int, ...]) -> np.ndarray:
    out = pre["base"].copy()
    if selected and pre["cand"].size:
        out = np.minimum(out, pre["cand"][list(selected), :].min(axis=0))
    return out


def added_only_distances_for(pre: dict, selected: tuple[int, ...]) -> np.ndarray:
    if not selected or not pre["cand"].size:
        return np.asarray([], dtype=np.float32)
    return pre["cand"][list(selected), :].min(axis=0).astype(np.float32)


def metrics_for_selected(precomputed: dict[str, dict], selected: tuple[int, ...]) -> dict:
    return {name: metrics_from_distances(distances_for(pre, selected)) for name, pre in precomputed.items()}


def added_risk_for_selected(precomputed: dict[str, dict], selected: tuple[int, ...]) -> dict:
    out = {}
    for name in ("val_gt3gt4_normal_risk", "train0601_gt3gt4_normal_risk", "train0531_gt3gt4_normal_risk"):
        out[name] = metrics_from_distances(added_only_distances_for(precomputed[name], selected))
    return out


def group_counts(candidates: list[Lane], selected: tuple[int, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx in selected:
        group = candidates[idx].group
        out[group] = out.get(group, 0) + 1
    return out


def _metric_value(m: dict, name: str, key: str, default: float) -> float:
    value = m.get(name, {}).get(key)
    return default if value is None else float(value)


def objective_score(m: dict, risk: dict, candidates: list[Lane], selected: tuple[int, ...], phase: str) -> float:
    def p90_over(name: str, limit: float, weight: float) -> float:
        return weight * max(0.0, _metric_value(m, name, "p90", 1e6) - limit)

    def match_under(name: str, limit: float, weight: float) -> float:
        return weight * max(0.0, limit - _metric_value(m, name, "match20", 0.0))

    score = (
        p90_over("val_gt5_primary", 72.0, 20.0)
        + p90_over("train0601_gt5_primary", 85.0, 10.0)
        + match_under("val_gt5_primary", 0.50, 12000.0)
        + match_under("train0601_gt5_primary", 0.26, 12000.0)
        + match_under("val_gt5_impact_target", 0.75, 12000.0)
        - 250.0 * _metric_value(m, "val_gt5_primary", "match20", 0.0)
        - 200.0 * _metric_value(m, "train0601_gt5_primary", "match20", 0.0)
        - 250.0 * _metric_value(m, "val_gt5_impact_target", "match20", 0.0)
    )
    counts = group_counts(candidates, selected)
    gt4_selected_count = sum(1 for idx in selected if candidates[idx].gt_count == 4)
    if counts.get("gt5cl", 0) < 2:
        score += 6000.0
    if counts.get("gt5r", 0) < 1:
        score += 4000.0

    if phase == "joint":
        score += (
            p90_over("val_gt4_primary", 95.0, 12.0)
            + p90_over("train0601_gt4_primary", 115.0, 8.0)
            + p90_over("train0531_gt4_primary", 95.0, 8.0)
            + match_under("val_gt4_primary", 0.25, 10000.0)
            + match_under("train0601_gt4_primary", 0.20, 10000.0)
            + match_under("train0531_gt4_primary", 0.2857, 10000.0)
            - 200.0 * _metric_value(m, "val_gt4_primary", "match20", 0.0)
            - 200.0 * _metric_value(m, "train0601_gt4_primary", "match20", 0.0)
            - 200.0 * _metric_value(m, "train0531_gt4_primary", "match20", 0.0)
        )
        min_gt4c = 2 if gt4_selected_count >= 3 else 1
        min_gt4r = 1 if gt4_selected_count >= 2 else 0
        if counts.get("gt4c", 0) < min_gt4c:
            score += 5000.0
        if counts.get("gt4r", 0) < min_gt4r:
            score += 3000.0
        for name in risk:
            score += 100.0 * _metric_value(risk, name, "match20", 0.0)
    return score


def evaluate_selected(
    precomputed: dict[str, dict],
    candidates: list[Lane],
    selected: tuple[int, ...],
    phase: str,
    eval_cache: dict[tuple[int, ...], tuple[dict, dict]],
    score_cache: dict[tuple[str, tuple[int, ...]], float],
) -> tuple[float, dict, dict]:
    key = tuple(sorted(selected))
    cached_eval = eval_cache.get(key)
    if cached_eval is None:
        m = metrics_for_selected(precomputed, key)
        risk = added_risk_for_selected(precomputed, key)
        eval_cache[key] = (m, risk)
    else:
        m, risk = cached_eval
    score_key = (phase, key)
    score = score_cache.get(score_key)
    if score is None:
        score = objective_score(m, risk, candidates, key, phase)
        score_cache[score_key] = score
    return float(score), m, risk


def beam_select(
    precomputed: dict[str, dict],
    candidates: list[Lane],
    allowed_indices: list[int],
    seed_selected: tuple[int, ...],
    slots: int,
    beam_width: int,
    phase: str,
    eval_cache: dict[tuple[int, ...], tuple[dict, dict]] | None = None,
    score_cache: dict[tuple[str, tuple[int, ...]], float] | None = None,
) -> list[tuple[float, tuple[int, ...], dict, dict]]:
    eval_cache = {} if eval_cache is None else eval_cache
    score_cache = {} if score_cache is None else score_cache
    beam: list[tuple[float, tuple[int, ...]]] = [(0.0, seed_selected)]
    allowed = tuple(int(i) for i in allowed_indices)
    for _ in range(slots):
        expanded: list[tuple[float, tuple[int, ...]]] = []
        seen = set()
        for _, selected in beam:
            selected_set = set(selected)
            for idx in allowed:
                if idx in selected_set:
                    continue
                nxt = tuple((*selected, idx))
                key = tuple(sorted(nxt))
                if key in seen:
                    continue
                seen.add(key)
                score, _, _ = evaluate_selected(precomputed, candidates, key, phase, eval_cache, score_cache)
                expanded.append((score, key))
        expanded.sort(key=lambda item: item[0])
        beam = expanded[: max(1, int(beam_width))]
    finals = []
    for score, selected in beam:
        key = tuple(sorted(selected))
        _, m, risk = evaluate_selected(precomputed, candidates, key, phase, eval_cache, score_cache)
        finals.append((float(score), key, m, risk))
    finals.sort(key=lambda item: item[0])
    return finals


def gate_from_metrics(base: dict, m: dict, risk: dict, formal_impact: bool) -> dict:
    base_val_gt4 = float(base["val_gt4_primary"]["match20"] or 0.0)
    base_train0601_gt4 = float(base["train0601_gt4_primary"]["match20"] or 0.0)
    base_train0531_gt4 = float(base["train0531_gt4_primary"]["match20"] or 0.0)
    base_val_normal = float(base["val_gt3gt4_normal_risk"]["match20"] or 0.0)
    base_train0601_normal = float(base["train0601_gt3gt4_normal_risk"]["match20"] or 0.0)
    base_train0531_normal = float(base["train0531_gt3gt4_normal_risk"]["match20"] or 0.0)
    eps = 1e-12
    checks = {
        "formal_impact_source": bool(formal_impact),
        "q12_references_preserved": True,
        "val_gt5_primary_match20_ge_0p50": _metric_value(m, "val_gt5_primary", "match20", 0.0) >= 0.50,
        "train0601_gt5_primary_match20_ge_0p26": _metric_value(m, "train0601_gt5_primary", "match20", 0.0) >= 0.26,
        "val_gt5_impact_match20_ge_0p75": _metric_value(m, "val_gt5_impact_target", "match20", 0.0) >= 0.75,
        "val_gt5_primary_p90_le_85": _metric_value(m, "val_gt5_primary", "p90", 1e6) <= 85.0,
        "train0601_gt5_primary_p90_le_90": _metric_value(m, "train0601_gt5_primary", "p90", 1e6) <= 90.0,
        "val_gt4_primary_match20_gt_base": _metric_value(m, "val_gt4_primary", "match20", 0.0) > base_val_gt4,
        "train0601_gt4_primary_match20_gt_base": _metric_value(m, "train0601_gt4_primary", "match20", 0.0) > base_train0601_gt4,
        "train0531_gt4_primary_match20_gt_base": _metric_value(m, "train0531_gt4_primary", "match20", 0.0) > base_train0531_gt4,
        "val_gt4_primary_match20_ge_0p25": _metric_value(m, "val_gt4_primary", "match20", 0.0) >= 0.25,
        "train0601_gt4_primary_match20_ge_0p20": _metric_value(m, "train0601_gt4_primary", "match20", 0.0) >= 0.20,
        "train0531_gt4_primary_match20_ge_0p2857": _metric_value(m, "train0531_gt4_primary", "match20", 0.0) >= 0.2857,
        "val_gt3gt4_normal_risk_no_increase": _metric_value(m, "val_gt3gt4_normal_risk", "match20", 0.0)
        <= base_val_normal + eps,
        "train0601_gt3gt4_normal_risk_no_increase": _metric_value(m, "train0601_gt3gt4_normal_risk", "match20", 0.0)
        <= base_train0601_normal + eps,
        "train0531_gt3gt4_normal_risk_no_increase": _metric_value(m, "train0531_gt3gt4_normal_risk", "match20", 0.0)
        <= base_train0531_normal + eps,
    }
    checks["strict_pass"] = all(bool(v) for v in checks.values())
    loose_keys = [
        "formal_impact_source",
        "q12_references_preserved",
        "val_gt5_primary_match20_ge_0p50",
        "train0601_gt5_primary_match20_ge_0p26",
        "val_gt5_impact_match20_ge_0p75",
        "val_gt4_primary_match20_gt_base",
        "train0601_gt4_primary_match20_gt_base",
        "train0531_gt4_primary_match20_gt_base",
        "val_gt3gt4_normal_risk_no_increase",
        "train0601_gt3gt4_normal_risk_no_increase",
        "train0531_gt3gt4_normal_risk_no_increase",
    ]
    checks["loose_pass"] = all(bool(checks[k]) for k in loose_keys)
    checks["risk_added_normal_match20"] = {
        name: risk[name]["match20"] for name in sorted(risk)
    }
    return checks


def build_static_references(
    selected: tuple[int, ...],
    candidates: list[Lane],
    num_queries: int,
) -> tuple[np.ndarray, list[dict], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    refs = np.zeros((int(num_queries), 56), dtype=np.float32)
    q12 = q12_default_references()
    refs[:12] = q12
    extra_queries = tuple(range(12, int(num_queries)))
    selected_gt5 = [candidates[idx] for idx in selected if candidates[idx].gt_count == 5]
    selected_gt4 = [candidates[idx] for idx in selected if candidates[idx].gt_count == 4]
    if len(selected_gt5) + len(selected_gt4) != len(extra_queries):
        raise ValueError(
            f"Expected {len(extra_queries)} extra prototypes, got {len(selected_gt5)} GT5 and {len(selected_gt4)} GT4."
        )
    gt5_bank_queries = tuple(extra_queries[: len(selected_gt5)])
    gt4_bank_queries = tuple(extra_queries[len(selected_gt5) :])

    references: list[dict] = []
    for qid in PROTECTED_Q12:
        references.append(
            {
                "query_id": int(qid),
                "role": "protected_q12",
                "source": "q12_linear",
                "x_norm": [round(float(x), 8) for x in refs[qid].tolist()],
            }
        )
    for qid, lane in zip(gt5_bank_queries, selected_gt5):
        refs[qid] = lane.x_norm
        item = lane_summary(lane)
        item.update(
            {
                "query_id": int(qid),
                "role": "gt5_short_weak_visible_bank",
                "source": "train_hard_lane",
                "x_norm": [round(float(x), 8) for x in lane.x_norm.tolist()],
            }
        )
        references.append(item)
    for qid, lane in zip(gt4_bank_queries, selected_gt4):
        refs[qid] = lane.x_norm
        item = lane_summary(lane)
        item.update(
            {
                "query_id": int(qid),
                "role": "gt4_hard_short_weak_visible_bank",
                "source": "train_hard_lane",
                "x_norm": [round(float(x), 8) for x in lane.x_norm.tolist()],
            }
        )
        references.append(item)
    return refs, references, extra_queries, gt5_bank_queries, gt4_bank_queries


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_candidate_row(
    rank: int,
    allocation: tuple[int, int],
    score: float,
    selected: tuple[int, ...],
    m: dict,
    risk: dict,
    gate: dict,
    candidates: list[Lane],
) -> dict:
    row = {
        "rank": int(rank),
        "allocation": f"{int(allocation[0])}:{int(allocation[1])}",
        "score": f"{float(score):.6f}",
        "strict_pass": bool(gate.get("strict_pass")),
        "loose_pass": bool(gate.get("loose_pass")),
        "selected_indices": ";".join(str(int(i)) for i in selected),
        "selected_groups": ";".join(candidates[i].group for i in selected),
        "selected_sources": ";".join(f"{candidates[i].split}:{candidates[i].raw_file}#{candidates[i].gt_lane_id}" for i in selected),
    }
    for name in (
        "val_gt5_primary",
        "train0601_gt5_primary",
        "val_gt5_impact_target",
        "val_gt4_primary",
        "train0601_gt4_primary",
        "train0531_gt4_primary",
    ):
        row[f"{name}_p90"] = "" if m[name]["p90"] is None else f"{float(m[name]['p90']):.6f}"
        row[f"{name}_match20"] = "" if m[name]["match20"] is None else f"{float(m[name]['match20']):.6f}"
    for name in sorted(risk):
        row[f"{name}_added_match20"] = "" if risk[name]["match20"] is None else f"{float(risk[name]['match20']):.6f}"
    return row


def main(
    default_num_queries: int = 20,
    default_allocations: str = "5:3,4:4,6:2",
    default_save_dir: Path = DEFAULT_SAVE_DIR,
    default_publish_bank: Path = DEFAULT_PUBLISH_BANK,
) -> None:
    args = parse_args(default_num_queries, default_allocations, default_save_dir, default_publish_bank)
    num_queries = int(args.num_queries)
    if num_queries < 20:
        raise ValueError(f"protected static gate requires num_queries >= 20, got {num_queries}.")
    if num_queries <= len(PROTECTED_Q12):
        raise ValueError(f"num_queries must exceed protected Q12 size {len(PROTECTED_Q12)}, got {num_queries}.")
    extra_queries = tuple(range(len(PROTECTED_Q12), num_queries))
    val_diag = _resolve_default(args.val_diag, DEFAULT_VAL_DIAG_CANDIDATES, "val diagnostic CSV")
    train0601_diag = _resolve_default(args.train0601_diag, DEFAULT_TRAIN0601_DIAG_CANDIDATES, "train0601 diagnostic CSV")
    train0531_diag = _resolve_default(args.train0531_diag, DEFAULT_TRAIN0531_DIAG_CANDIDATES, "train0531 diagnostic CSV")
    impact_csv = _resolve_default(args.impact_csv, DEFAULT_IMPACT_CSV_CANDIDATES, "impact CSV")
    val_gt_json = Path(args.val_gt_json)
    train0601_json = Path(args.train0601_json)
    train0531_json = Path(args.train0531_json)
    if not val_gt_json.is_absolute():
        val_gt_json = ROOT / val_gt_json
    if not train0601_json.is_absolute():
        train0601_json = ROOT / train0601_json
    if not train0531_json.is_absolute():
        train0531_json = ROOT / train0531_json
    for path in (val_gt_json, train0601_json, train0531_json):
        if not path.exists():
            raise FileNotFoundError(path)
    validate_gate_inputs(val_diag, train0601_diag, train0531_diag, impact_csv, val_gt_json, train0601_json, train0531_json)

    gt_indexes = {
        "val": load_gt_index([val_gt_json]),
        "train0601": load_gt_index([train0601_json]),
        "train0531": load_gt_index([train0531_json]),
    }
    raw_rows = {
        "val": read_csv_rows(val_diag),
        "train0601": read_csv_rows(train0601_diag),
        "train0531": read_csv_rows(train0531_diag),
    }
    impact_rows = read_csv_rows(impact_csv)
    for split, rows in raw_rows.items():
        _assert_no_test_rows(rows, f"{split} diagnostic CSV")
    _assert_no_test_rows(impact_rows, "impact CSV")
    eval_sets, impact_source, formal_impact = build_eval_sets(raw_rows, impact_rows, gt_indexes)

    train_proto_lanes = dedup_lanes(
        [lane_from_row("train0601", row, gt_indexes["train0601"]) for row in raw_rows["train0601"] if is_primary_hard(row)]
        + [lane_from_row("train0531", row, gt_indexes["train0531"]) for row in raw_rows["train0531"] if is_primary_hard(row)]
    )
    train_proto_lanes = [
        lane for lane in train_proto_lanes if lane.group in {"gt5cl", "gt5r", "gt4c", "gt4r", "gt4other"}
    ]
    candidates = trim_candidate_pool(train_proto_lanes, eval_sets, int(args.pool_cap_per_group))
    gt5_indices = [i for i, lane in enumerate(candidates) if lane.gt_count == 5]
    gt4_indices = [i for i, lane in enumerate(candidates) if lane.gt_count == 4]
    allocations = parse_allocations(args.allocations, len(extra_queries))
    max_gt5_slots = max(gt5_slots for gt5_slots, _ in allocations)
    max_gt4_slots = max(gt4_slots for _, gt4_slots in allocations)
    if len(gt5_indices) < max_gt5_slots or len(gt4_indices) < max_gt4_slots:
        raise RuntimeError(
            f"Need at least {max_gt5_slots} GT5 and {max_gt4_slots} GT4 candidates for allocations {allocations}, "
            f"got {len(gt5_indices)} and {len(gt4_indices)}."
        )

    base_refs = q12_default_references()
    base_metrics = {name: metrics(base_refs, lanes) for name, lanes in eval_sets.items()}
    precomputed = {name: precompute(base_refs, candidates, lanes) for name, lanes in eval_sets.items()}

    joint_rows: list[tuple[tuple[int, int], float, tuple[int, ...], dict, dict, dict]] = []
    eval_cache: dict[tuple[int, ...], tuple[dict, dict]] = {}
    score_cache: dict[tuple[str, tuple[int, ...]], float] = {}
    for allocation in allocations:
        gt5_slots, gt4_slots = allocation
        gt5_beam = beam_select(
            precomputed,
            candidates,
            gt5_indices,
            tuple(),
            gt5_slots,
            int(args.beam_width),
            "gt5",
            eval_cache,
            score_cache,
        )
        for _, gt5_selected, _, _ in gt5_beam[: max(1, min(len(gt5_beam), int(args.beam_width)))]:
            joint = beam_select(
                precomputed,
                candidates,
                gt4_indices,
                gt5_selected,
                gt4_slots,
                max(1, int(args.beam_width) // 4),
                "joint",
                eval_cache,
                score_cache,
            )
            for score, selected, m, risk in joint:
                gate = gate_from_metrics(base_metrics, m, risk, formal_impact)
                joint_rows.append((allocation, float(score), selected, m, risk, gate))
    if not joint_rows:
        raise RuntimeError("No protected static joint rows were evaluated.")
    joint_rows.sort(key=lambda item: (not item[5]["strict_pass"], not item[5]["loose_pass"], item[1]))
    top_rows = joint_rows[: int(args.top_report)]
    best_allocation, best_score, best_selected, best_metrics, best_risk, best_gate = top_rows[0]
    best_refs, references, extra_queries, best_gt5_bank_queries, best_gt4_bank_queries = build_static_references(
        best_selected,
        candidates,
        num_queries,
    )

    q12_preserved = bool(np.allclose(best_refs[:12], base_refs, atol=1e-8))
    best_gate["q12_references_preserved"] = q12_preserved
    best_gate["strict_pass"] = bool(best_gate["strict_pass"] and q12_preserved)
    best_gate["loose_pass"] = bool(best_gate["loose_pass"] and q12_preserved)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"q{num_queries}_protected_static"
    bank_path = save_dir / f"{prefix}_env30_best.json"
    top_csv = save_dir / f"{prefix}_top_candidates.csv"
    summary_path = save_dir / "summary.json"

    bank = {
        "schema": f"q{num_queries}_protected_static_reference_bank_v1",
        "result_type": "default_off_experimental_reference_bank",
        "formal_gate_passed": bool(best_gate["strict_pass"]),
        "loose_gate_passed": bool(best_gate["loose_pass"]),
        "selection_protocol": "official_val_train_raw_static_gate_no_test",
        "test_used": False,
        "source_splits": ["official_val", "train0601", "train0531"],
        "input_provenance": {
            "val_diag": str(val_diag.relative_to(ROOT)) if val_diag.is_relative_to(ROOT) else val_diag.name,
            "train0601_diag": str(train0601_diag.relative_to(ROOT)) if train0601_diag.is_relative_to(ROOT) else train0601_diag.name,
            "train0531_diag": str(train0531_diag.relative_to(ROOT)) if train0531_diag.is_relative_to(ROOT) else train0531_diag.name,
            "impact_csv": str(impact_csv.relative_to(ROOT)) if impact_csv.is_relative_to(ROOT) else impact_csv.name,
            "val_gt_json": str(val_gt_json.relative_to(ROOT)) if val_gt_json.is_relative_to(ROOT) else val_gt_json.name,
            "train0601_json": str(train0601_json.relative_to(ROOT)) if train0601_json.is_relative_to(ROOT) else train0601_json.name,
            "train0531_json": str(train0531_json.relative_to(ROOT)) if train0531_json.is_relative_to(ROOT) else train0531_json.name,
        },
        "num_queries": int(num_queries),
        "num_points": 56,
        "point_mode": "fixed_y",
        "reference_mode": "dualbank",
        "image_width": IMAGE_WIDTH,
        "fixed_y_px_desc": [int(x) for x in FIXED_Y_DESC.tolist()],
        "protected_queries": [int(q) for q in PROTECTED_Q12],
        "extra_queries": [int(q) for q in extra_queries],
        "selected_allocation": f"{int(best_allocation[0])}:{int(best_allocation[1])}",
        "gt5_bank_queries": [int(q) for q in best_gt5_bank_queries],
        "gt4_bank_queries": [int(q) for q in best_gt4_bank_queries],
        "gate": best_gate,
        "metrics": best_metrics,
        "added_normal_risk": best_risk,
        "x_norm": [[round(float(x), 8) for x in row] for row in best_refs.tolist()],
        "references": references,
    }
    bank_path.write_text(json.dumps(bank, indent=2), encoding="utf-8")

    csv_rows = [
        flatten_candidate_row(rank + 1, allocation, score, selected, m, risk, gate, candidates)
        for rank, (allocation, score, selected, m, risk, gate) in enumerate(top_rows)
    ]
    if csv_rows:
        write_csv(top_csv, csv_rows, list(csv_rows[0].keys()))

    summary = {
        "schema": f"q{num_queries}_protected_static_gate_v1",
        "inputs": {
            "val_diag": str(val_diag),
            "train0601_diag": str(train0601_diag),
            "train0531_diag": str(train0531_diag),
            "impact_csv": str(impact_csv),
            "val_gt_json": str(val_gt_json),
            "train0601_json": str(train0601_json),
            "train0531_json": str(train0531_json),
            "test_used": False,
        },
        "definition": {
            "num_queries": int(num_queries),
            "protected_queries": [int(q) for q in PROTECTED_Q12],
            "extra_queries": [int(q) for q in extra_queries],
            "allocations": [f"{int(gt5)}:{int(gt4)}" for gt5, gt4 in allocations],
            "selected_allocation": f"{int(best_allocation[0])}:{int(best_allocation[1])}",
            "gt5_bank_queries": [int(q) for q in best_gt5_bank_queries],
            "gt4_bank_queries": [int(q) for q in best_gt4_bank_queries],
            "impact_source": impact_source,
            "formal_impact_source": bool(formal_impact),
            "beam_width": int(args.beam_width),
            "pool_cap_per_group": int(args.pool_cap_per_group),
        },
        "counts": {
            "raw_rows": {k: len(v) for k, v in raw_rows.items()},
            "eval_sets": {k: len(v) for k, v in eval_sets.items()},
            "train_prototype_pool": len(train_proto_lanes),
            "train_prototype_pool_by_group": count_by_group(train_proto_lanes),
            "trimmed_candidates": len(candidates),
            "trimmed_candidates_by_group": count_by_group(candidates),
            "joint_rows_evaluated": len(joint_rows),
            "strict_pass_rows": int(sum(1 for row in joint_rows if row[5]["strict_pass"])),
            "loose_pass_rows": int(sum(1 for row in joint_rows if row[5]["loose_pass"])),
        },
        "base_metrics": base_metrics,
        "best": {
            "selected_allocation": f"{int(best_allocation[0])}:{int(best_allocation[1])}",
            "score": float(best_score),
            "selected_indices": [int(i) for i in best_selected],
            "selected_prototypes": [lane_summary(candidates[i]) for i in best_selected],
            "metrics": best_metrics,
            "added_normal_risk": best_risk,
            "gate": best_gate,
            "bank_path": str(bank_path),
        },
        "outputs": {
            "summary": str(summary_path),
            "top_candidates": str(top_csv),
            "bank": str(bank_path),
        },
    }

    if args.publish_bank:
        publish_path = Path(args.publish_path)
        if not publish_path.is_absolute():
            publish_path = ROOT / publish_path
        if not best_gate["strict_pass"]:
            summary["outputs"]["published_bank"] = None
            summary["publish_error"] = (
                "Bank was not published because strict gate failed. "
                "Do not train from a failed bank."
            )
        else:
            publish_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bank_path, publish_path)
            summary["outputs"]["published_bank"] = str(publish_path)
            bank["published_path"] = str(publish_path)
            publish_path.write_text(json.dumps(bank, indent=2), encoding="utf-8")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"strict_pass": best_gate["strict_pass"], "loose_pass": best_gate["loose_pass"], "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
