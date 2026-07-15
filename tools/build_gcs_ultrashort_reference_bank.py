# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Build a Q12 fixed-y ultrashort reference bank from train hard lanes.

The generated bank is a controlled, default-off experiment artifact. Prototype
lanes are selected only from train diagnostics and train TuSimple JSON. The
official-val diagnostics are used as the static validation gate; no official-val
lane geometry is used as a prototype.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA = "gcs_q12_fixed_y_reference_bank_v1"
IMAGE_WIDTH = 1280.0
FIXED_Y_DESC = np.arange(710.0, 150.0, -10.0, dtype=np.float32)
GROUP_K = {
    "gt5_vle10_center_left": 2,
    "gt4_vle10_center": 2,
    "gt4_vle10_right_side": 2,
}
TRAIN_NORMAL_SCORE_CAP = 512
STATIC_GATE_TOP_K = 5000
DEFAULT_QUERY_STATIC_PENALTY = "4:0,10:2"


@dataclass(frozen=True)
class LaneShape:
    raw_file: str
    gt_lane_id: int
    gt_count: int
    lane_order: int | None
    lane_position: str
    side_group: str
    visible_points_gt: int
    group: str
    x_norm: np.ndarray
    visible_mask: np.ndarray
    hard: bool
    raw_has_match_20px: bool
    raw_best_valid_len_06: int | None
    raw_best_valid_count_06: int | None
    point_valid_recall_06: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-diag-csv", action="append", required=True, help="Train raw_gt_lane_diagnostics.csv.")
    parser.add_argument("--val-diag-csv", action="append", required=True, help="Official-val raw_gt_lane_diagnostics.csv.")
    parser.add_argument("--train-label-json", action="append", required=True, help="TuSimple train label_data_*.json.")
    parser.add_argument("--val-gt-json", action="append", required=True, help="Official-val GT JSON, audit only.")
    parser.add_argument("--num-queries", type=int, default=12)
    parser.add_argument("--num-points", type=int, default=56)
    parser.add_argument("--replace-queries", default="1,3,4,5,6,7,8,10")
    parser.add_argument("--keep-queries", default="0,2,9,11")
    parser.add_argument(
        "--required-replace-queries",
        default="",
        help="Comma-separated query ids that must be replaced in the selected assignment. Empty means optional.",
    )
    parser.add_argument(
        "--query-static-penalty",
        default=DEFAULT_QUERY_STATIC_PENALTY,
        help="Comma-separated query_id:penalty terms added when a query is replaced, e.g. '4:0,10:2'.",
    )
    parser.add_argument(
        "--prototype-pool-per-group",
        type=int,
        default=4,
        help="Legacy medoid pool size retained in the summary; exhaustive static-gate selection uses all train hard lanes.",
    )
    parser.add_argument(
        "--static-gate-top-k",
        type=int,
        default=STATIC_GATE_TOP_K,
        help="Number of hard-gate shortlist candidates to fully audit against normal lanes.",
    )
    parser.add_argument("--save-bank", required=True)
    parser.add_argument("--save-summary", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_int_list(value: str) -> list[int]:
    if not str(value).strip():
        return []
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _parse_query_penalties(value: str) -> dict[int, float]:
    penalties: dict[int, float] = {}
    if not str(value).strip():
        return penalties
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Query static penalty item must be query_id:penalty, got {item!r}.")
        query_text, penalty_text = item.split(":", 1)
        query_id = int(query_text.strip())
        penalty = float(penalty_text.strip())
        if not math.isfinite(penalty):
            raise ValueError(f"Query static penalty must be finite for q{query_id}, got {penalty_text!r}.")
        penalties[query_id] = penalty
    return penalties


def _read_json_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}.")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_gt_index(paths: Iterable[Path]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for path in paths:
        for record in _read_json_records(path):
            raw_file = _norm_raw_file(record.get("raw_file", ""))
            if not raw_file:
                continue
            if raw_file in index:
                raise ValueError(f"Duplicate raw_file {raw_file!r} across GT JSON inputs.")
            index[raw_file] = record
    return index


def _norm_raw_file(raw_file: str) -> str:
    return str(raw_file).replace("\\", "/").strip()


def _read_diag_rows(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_diag_csv"] = str(path)
                row["raw_file"] = _norm_raw_file(row.get("raw_file", ""))
                rows.append(row)
    return rows


def _bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int_or_none(value) -> int | None:
    text = str(value).strip()
    if text == "":
        return None
    return int(float(text))


def _float_or_none(value) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    out = float(text)
    if not math.isfinite(out):
        return None
    return out


def _safe_lane_order(value) -> int | None:
    text = str(value).strip()
    if text == "":
        return None
    return int(float(text))


def _default_references(num_queries: int, num_points: int) -> np.ndarray:
    bottom_x = np.linspace(0.05, 0.95, num_queries, dtype=np.float32)
    top_x = 0.5 + (bottom_x - 0.5) * 0.25
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    return (bottom_x[:, None] * (1.0 - t[None, :]) + top_x[:, None] * t[None, :]).astype(np.float32)


def _group_for_row(row: dict) -> str:
    gt_count = int(float(row.get("gt_count", 0) or 0))
    visible = int(float(row.get("visible_points_gt", 0) or 0))
    side_group = str(row.get("side_group", "")).strip()
    if visible > 10:
        return ""
    if gt_count == 5 and side_group in {"center", "left_side"}:
        return "gt5_vle10_center_left"
    if gt_count == 4 and side_group == "center":
        return "gt4_vle10_center"
    if gt_count == 4 and side_group == "right_side":
        return "gt4_vle10_right_side"
    return ""


def _is_hard(row: dict) -> bool:
    gt_count = int(float(row.get("gt_count", 0) or 0))
    visible = int(float(row.get("visible_points_gt", 0) or 0))
    if gt_count not in {4, 5} or visible > 10:
        return False
    raw_has_match = _bool_value(row.get("raw_has_match_20px", ""))
    valid_len = _int_or_none(row.get("raw_best_valid_len@0.6", ""))
    min_points = _int_or_none(row.get("min_points", "")) or 0
    recall = _float_or_none(row.get("point_valid_recall@0.6", ""))
    return (not raw_has_match) or (valid_len is None or valid_len < min_points) or (recall is None or recall < 0.75)


def _lane_to_fixed_reference(record: dict, gt_lane_id: int, num_points: int) -> tuple[np.ndarray, np.ndarray] | None:
    if num_points != len(FIXED_Y_DESC):
        raise ValueError(f"This tool supports K56 only, got num_points={num_points}.")
    lanes = record.get("lanes", [])
    if gt_lane_id < 0 or gt_lane_id >= len(lanes):
        return None
    lane = np.asarray(lanes[gt_lane_id], dtype=np.float32)
    h_samples = np.asarray(record.get("h_samples", []), dtype=np.float32)
    if lane.shape[0] != h_samples.shape[0]:
        raise ValueError(f"Lane/h_samples length mismatch for {record.get('raw_file')} lane {gt_lane_id}.")
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


def _shape_from_row(row: dict, gt_index: dict[str, dict], num_points: int) -> LaneShape | None:
    raw_file = _norm_raw_file(row.get("raw_file", ""))
    record = gt_index.get(raw_file)
    if record is None:
        return None
    gt_lane_id = int(float(row.get("gt_lane_id", -1) or -1))
    fixed = _lane_to_fixed_reference(record, gt_lane_id, num_points)
    if fixed is None:
        return None
    x_norm, visible_mask = fixed
    group = _group_for_row(row)
    return LaneShape(
        raw_file=raw_file,
        gt_lane_id=gt_lane_id,
        gt_count=int(float(row.get("gt_count", 0) or 0)),
        lane_order=_safe_lane_order(row.get("lane_order", "")),
        lane_position=str(row.get("lane_position", "")).strip(),
        side_group=str(row.get("side_group", "")).strip(),
        visible_points_gt=int(float(row.get("visible_points_gt", 0) or 0)),
        group=group,
        x_norm=x_norm,
        visible_mask=visible_mask,
        hard=_is_hard(row),
        raw_has_match_20px=_bool_value(row.get("raw_has_match_20px", "")),
        raw_best_valid_len_06=_int_or_none(row.get("raw_best_valid_len@0.6", "")),
        raw_best_valid_count_06=_int_or_none(row.get("raw_best_valid_count@0.6", "")),
        point_valid_recall_06=_float_or_none(row.get("point_valid_recall@0.6", "")),
    )


def _distance_px(reference_x: np.ndarray, lane: LaneShape) -> float:
    mask = lane.visible_mask.astype(bool)
    if int(mask.sum()) == 0:
        return float("inf")
    return float(np.mean(np.abs(reference_x[mask] - lane.x_norm[mask])) * IMAGE_WIDTH)


def _distance_matrix(refs: np.ndarray, lanes: list[LaneShape]) -> np.ndarray:
    """Return N lanes x R references mean visible-anchor x error in pixels."""
    if not lanes:
        return np.zeros((0, int(refs.shape[0])), dtype=np.float32)
    lane_x = np.stack([lane.x_norm for lane in lanes], axis=0).astype(np.float32)
    mask = np.stack([lane.visible_mask.astype(np.float32) for lane in lanes], axis=0)
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    diff = np.abs(lane_x[:, None, :] - refs.astype(np.float32)[None, :, :]) * mask[:, None, :]
    return (diff.sum(axis=2) / denom * IMAGE_WIDTH).astype(np.float32)


def _nearest_distances(refs: np.ndarray, lanes: list[LaneShape]) -> tuple[np.ndarray, np.ndarray]:
    mat = _distance_matrix(refs, lanes)
    if mat.shape[0] == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int64)
    nearest = np.nanargmin(mat, axis=1).astype(np.int64)
    distances = mat[np.arange(mat.shape[0]), nearest].astype(np.float32)
    return distances, nearest


def _percentile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, q))


def _metrics(refs: np.ndarray, lanes: list[LaneShape]) -> dict:
    if not lanes:
        return {"count": 0, "p50": None, "p90": None, "match20": None, "nearest_ref_query_hist": {}}
    distances, nearest = _nearest_distances(refs, lanes)
    hist = {str(int(q)): int((nearest == q).sum()) for q in sorted(set(nearest.tolist()))}
    return {
        "count": int(len(lanes)),
        "p50": _percentile(distances, 50),
        "p90": _percentile(distances, 90),
        "match20": float(np.mean(distances <= 20.0)),
        "nearest_ref_query_hist": hist,
    }


def _greedy_medoids(lanes: list[LaneShape], k: int) -> list[LaneShape]:
    if not lanes or k <= 0:
        return []
    selected: list[LaneShape] = []
    remaining = list(lanes)
    while remaining and len(selected) < k:
        best_idx = 0
        best_key = (float("inf"), float("inf"), "")
        for idx, candidate in enumerate(remaining):
            refs = np.stack([lane.x_norm for lane in [*selected, candidate]], axis=0)
            distances, _ = _nearest_distances(refs, lanes)
            key = (_percentile(distances, 50) or float("inf"), _percentile(distances, 90) or float("inf"), candidate.raw_file)
            if key < best_key:
                best_key = key
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected


def _dedup_lanes(lanes: Iterable[LaneShape]) -> list[LaneShape]:
    out: list[LaneShape] = []
    seen: set[tuple[str, int]] = set()
    for lane in lanes:
        key = (lane.raw_file, lane.gt_lane_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(lane)
    return out


def _combo_allowed(group: str, lanes: tuple[LaneShape, ...]) -> bool:
    if group == "gt5_vle10_center_left":
        sides = {lane.side_group for lane in lanes}
        return "center" in sides and "left_side" in sides
    return True


def _lane_package(lane: LaneShape, proto_index: int) -> dict:
    return {
        "name": f"{lane.group}_proto{proto_index}",
        "source_raw_file": lane.raw_file,
        "source_gt_lane_id": int(lane.gt_lane_id),
        "group": lane.group,
        "gt_count": int(lane.gt_count),
        "lane_order": lane.lane_order,
        "lane_position": lane.lane_position,
        "side_group": lane.side_group,
        "visible_points_gt": int(lane.visible_points_gt),
        "x_norm": [round(float(x), 8) for x in lane.x_norm.tolist()],
    }


def _build_eval_groups(lanes: list[LaneShape]) -> dict[str, list[LaneShape]]:
    groups = {
        "hard_target": [lane for lane in lanes if lane.hard and lane.group in GROUP_K],
        "target_ultrashort_all": [lane for lane in lanes if lane.group in GROUP_K],
        "gt5_vle10_center_left": [lane for lane in lanes if lane.group == "gt5_vle10_center_left"],
        "gt4_vle10_center": [lane for lane in lanes if lane.group == "gt4_vle10_center"],
        "gt4_vle10_right_side": [lane for lane in lanes if lane.group == "gt4_vle10_right_side"],
        "gt5_short_all": [lane for lane in lanes if lane.gt_count == 5 and lane.visible_points_gt <= 10],
        "normal_lane": [lane for lane in lanes if not (lane.group in GROUP_K) and lane.visible_points_gt > 20],
        "all": lanes,
    }
    return groups


def _sample_train_normal_lanes(lanes: list[LaneShape], cap: int) -> list[LaneShape]:
    normal = [
        lane
        for lane in lanes
        if lane.group not in GROUP_K and lane.visible_points_gt > 20
    ]
    normal = sorted(normal, key=lambda lane: (lane.raw_file, lane.gt_lane_id))
    if len(normal) <= int(cap):
        return normal
    idx = np.linspace(0, len(normal) - 1, int(cap), dtype=np.int64)
    return [normal[int(i)] for i in idx]


def _score_assignment(base_refs: np.ndarray, new_refs: np.ndarray, lanes: list[LaneShape], keep_queries: set[int]) -> tuple[float, dict]:
    groups = _build_eval_groups(lanes)
    base_hard = _metrics(base_refs, groups["hard_target"])
    new_hard = _metrics(new_refs, groups["hard_target"])
    base_normal = _metrics(base_refs, groups["normal_lane"])
    new_normal = _metrics(new_refs, groups["normal_lane"])
    base_gt5 = _metrics(base_refs, groups["gt5_short_all"])
    new_gt5 = _metrics(new_refs, groups["gt5_short_all"])

    def val(metrics: dict, key: str, default: float = 0.0) -> float:
        x = metrics.get(key)
        return default if x is None else float(x)

    hard_ape_gain = val(base_hard, "p50") - val(new_hard, "p50")
    hard_p90_gain = val(base_hard, "p90") - val(new_hard, "p90")
    hard_match_gain = val(new_hard, "match20") - val(base_hard, "match20")
    normal_penalty = 2.0 * max(0.0, val(new_normal, "p50") - val(base_normal, "p50"))
    normal_penalty += max(0.0, val(new_normal, "p90") - val(base_normal, "p90"))
    gt5_penalty = 2.0 * max(0.0, val(new_gt5, "p50") - val(base_gt5, "p50"))
    gt5_penalty += max(0.0, val(new_gt5, "p90") - val(base_gt5, "p90"))
    changed_queries = {q for q in range(new_refs.shape[0]) if not np.allclose(new_refs[q], base_refs[q], atol=1e-8)}
    protected_penalty = 50.0 * len(changed_queries & keep_queries)
    hard_gate_penalty = 1000.0 * max(0.0, val(new_hard, "p50", 1e9) - 30.0)
    hard_gate_penalty += 1000.0 * max(0.0, val(new_hard, "p90", 1e9) - 50.0)
    score = (
        hard_ape_gain
        + hard_p90_gain
        + 100.0 * hard_match_gain
        - normal_penalty
        - gt5_penalty
        - protected_penalty
        - hard_gate_penalty
    )
    return score, {
        "score": score,
        "hard_ape_p50_gain": hard_ape_gain,
        "hard_ape_p90_gain": hard_p90_gain,
        "hard_match20_gain": hard_match_gain,
        "normal_lane_regression_penalty": normal_penalty,
        "gt5_short_regression_penalty": gt5_penalty,
        "protected_query_change_penalty": protected_penalty,
        "hard_gate_penalty": hard_gate_penalty,
        "base_hard_target": base_hard,
        "new_hard_target": new_hard,
        "base_normal_lane": base_normal,
        "new_normal_lane": new_normal,
        "base_gt5_short_all": base_gt5,
        "new_gt5_short_all": new_gt5,
    }


def _all_group_combinations(group: str, lanes: list[LaneShape], k: int) -> list[tuple[LaneShape, ...]]:
    return [
        combo
        for combo in itertools.combinations(lanes, k)
        if _combo_allowed(group, combo)
    ]


def _combination_distance_matrix(combos: list[tuple[LaneShape, ...]], lanes: list[LaneShape]) -> np.ndarray:
    if not combos:
        return np.zeros((0, len(lanes)), dtype=np.float32)
    rows = []
    for combo in combos:
        refs = np.stack([lane.x_norm for lane in combo], axis=0)
        distances, _ = _nearest_distances(refs, lanes)
        rows.append(distances.astype(np.float32))
    return np.stack(rows, axis=0).astype(np.float32)


def _push_near_candidate(heap: list, limit: int, item: tuple) -> None:
    if limit <= 0:
        return
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _static_gate_candidate_score(audit: dict, qset: tuple[int, ...], query_penalties: dict[int, float]) -> float:
    gate = audit["gate"]
    hard_p50 = float(audit["new_nearest_reference_ape_p50"] or 1e9)
    hard_p90 = float(audit["new_nearest_reference_ape_p90"] or 1e9)
    hard_match20 = float(audit["new_nearest_reference_match20"] or 0.0)
    normal = audit["normal_lane_regression"]
    normal_p50_delta = float(normal["delta_p50"] or 0.0)
    normal_p90_delta = float(normal["delta_p90"] or 0.0)
    gt5_short = audit["per_group"]["gt5_short_all"]
    gt5_base = gt5_short["base"]
    gt5_new = gt5_short["new"]
    gt5_p50_delta = 0.0
    gt5_p90_delta = 0.0
    if gt5_base["p50"] is not None and gt5_new["p50"] is not None:
        gt5_p50_delta = float(gt5_new["p50"] - gt5_base["p50"])
    if gt5_base["p90"] is not None and gt5_new["p90"] is not None:
        gt5_p90_delta = float(gt5_new["p90"] - gt5_base["p90"])
    gate_penalty = 0.0
    if not gate["hard_p50_lt_30"]:
        gate_penalty += 1_000_000.0 + 10_000.0 * max(0.0, hard_p50 - 30.0)
    if not gate["hard_p90_lt_50"]:
        gate_penalty += 1_000_000.0 + 10_000.0 * max(0.0, hard_p90 - 50.0)
    if not gate["hard_match20_gain_ge_0p20"]:
        gate_penalty += 1_000_000.0
    if not gate["normal_p50_regression_le_5"]:
        gate_penalty += 1_000_000.0 + 10_000.0 * max(0.0, normal_p50_delta - 5.0)
    if not gate["normal_p90_regression_le_10"]:
        gate_penalty += 1_000_000.0 + 10_000.0 * max(0.0, normal_p90_delta - 10.0)

    query_penalty = sum(float(query_penalties.get(int(q), 0.0)) for q in qset)
    gt5_short_regression = 10.0 * max(0.0, gt5_p50_delta) + max(0.0, gt5_p90_delta)
    normal_penalty = 2.0 * max(0.0, normal_p50_delta) + max(0.0, normal_p90_delta)
    return (
        gate_penalty
        + 1000.0 * gt5_short_regression
        + 100.0 * normal_penalty
        + query_penalty
        + 0.01 * hard_p50
        + 0.0025 * hard_p90
        - 0.25 * hard_match20
    )


def _best_query_assignment(
    base_refs: np.ndarray,
    prototypes: list[dict],
    qset: tuple[int, ...],
) -> tuple[dict[int, dict], np.ndarray, float]:
    proto_x = [np.asarray(p["x_norm"], dtype=np.float32) for p in prototypes]
    best_cost = float("inf")
    best_perm: tuple[int, ...] | None = None
    for query_tuple in itertools.permutations(qset, len(prototypes)):
        cost = 0.0
        for proto_idx, query_id in enumerate(query_tuple):
            cost += float(np.mean(np.abs(proto_x[proto_idx] - base_refs[int(query_id)])) * IMAGE_WIDTH)
        if cost < best_cost:
            best_cost = cost
            best_perm = tuple(int(q) for q in query_tuple)
    if best_perm is None:
        raise RuntimeError("No query assignment permutation was produced.")
    refs = base_refs.copy()
    mapping: dict[int, dict] = {}
    for proto_idx, query_id in enumerate(best_perm):
        refs[query_id] = proto_x[proto_idx]
        mapping[query_id] = {k: v for k, v in prototypes[proto_idx].items() if k != "x_norm"}
    return mapping, refs, best_cost


def _select_static_gate_candidate(
    base_refs: np.ndarray,
    replace_queries: list[int],
    keep_queries: set[int],
    required_queries: set[int],
    query_penalties: dict[int, float],
    train_lanes: list[LaneShape],
    train_score_lanes: list[LaneShape],
    val_lanes: list[LaneShape],
    top_k: int,
) -> dict:
    group_names = list(GROUP_K)
    total_prototypes = sum(GROUP_K.values())
    val_groups = _build_eval_groups(val_lanes)
    val_hard_lanes = val_groups["hard_target"]
    if not val_hard_lanes:
        raise ValueError("No official-val hard target lanes are available for the static gate.")

    group_lanes = {
        group: [lane for lane in train_lanes if lane.group == group]
        for group in group_names
    }
    group_combos = {
        group: _all_group_combinations(group, group_lanes[group], GROUP_K[group])
        for group in group_names
    }
    for group, combos in group_combos.items():
        if not combos:
            raise ValueError(f"No train hard prototype combinations available for {group}.")

    combo_distances = {
        group: _combination_distance_matrix(group_combos[group], val_hard_lanes)
        for group in group_names
    }
    hard_base = _metrics(base_refs, val_hard_lanes)
    base_match20 = float(hard_base["match20"] or 0.0)

    hard_candidates: list[tuple] = []
    near_heap: list[tuple] = []
    qsets = [
        tuple(int(q) for q in qset)
        for qset in itertools.combinations(replace_queries, total_prototypes)
        if required_queries.issubset(set(qset))
    ]
    if not qsets:
        raise ValueError(
            f"No replace-query sets remain after applying required queries {sorted(required_queries)} "
            f"to replace_queries={replace_queries}."
        )
    for qset in qsets:
        kept_queries = [q for q in range(base_refs.shape[0]) if q not in qset]
        if kept_queries:
            kept_dist = _distance_matrix(base_refs[kept_queries], val_hard_lanes).min(axis=1).astype(np.float32)
        else:
            kept_dist = np.full((len(val_hard_lanes),), np.inf, dtype=np.float32)

        gt5_dist = combo_distances[group_names[0]]
        gt4_center_dist = combo_distances[group_names[1]]
        gt4_right_dist = combo_distances[group_names[2]]

        for a0 in range(0, gt5_dist.shape[0], 32):
            min_a = np.minimum(gt5_dist[a0 : a0 + 32, None, None, :], kept_dist[None, None, None, :])
            for b0 in range(0, gt4_center_dist.shape[0], 128):
                min_ab = np.minimum(min_a, gt4_center_dist[None, b0 : b0 + 128, None, :])
                for c0 in range(0, gt4_right_dist.shape[0], 64):
                    dist = np.minimum(min_ab, gt4_right_dist[None, None, c0 : c0 + 64, :])
                    p50 = np.percentile(dist, 50, axis=3)
                    p90 = np.percentile(dist, 90, axis=3)
                    match20 = (dist <= 20.0).mean(axis=3)
                    hard_gate_mask = (
                        (p50 < 30.0)
                        & (p90 < 50.0)
                        & ((match20 - base_match20) >= 0.20 - 1e-9)
                    )
                    for aa, bb, cc in np.argwhere(hard_gate_mask):
                        hard_score = float(p50[aa, bb, cc] * 1000.0 + p90[aa, bb, cc] * 10.0 - match20[aa, bb, cc])
                        hard_candidates.append(
                            (
                                hard_score,
                                tuple(int(q) for q in qset),
                                (a0 + int(aa), b0 + int(bb), c0 + int(cc)),
                            )
                        )

                    near_score = -(p50 * 1000.0 + p90 * 10.0 - match20)
                    flat_count = min(8, near_score.size)
                    if flat_count > 0:
                        flat_indices = np.argpartition(near_score.ravel(), -flat_count)[-flat_count:]
                        for flat_idx in flat_indices:
                            aa, bb, cc = np.unravel_index(int(flat_idx), near_score.shape)
                            _push_near_candidate(
                                near_heap,
                                max(top_k, 1),
                                (
                                    float(near_score[aa, bb, cc]),
                                    tuple(int(q) for q in qset),
                                    (a0 + int(aa), b0 + int(bb), c0 + int(cc)),
                                ),
                            )

    hard_gate_candidate_count = len(hard_candidates)
    if hard_candidates:
        hard_candidates = sorted(hard_candidates, key=lambda x: x[0])[: max(1, int(top_k))]
        shortlist = hard_candidates
        shortlist_source = "official_val_hard_gate"
    else:
        shortlist = [
            (-item[0], item[1], item[2])
            for item in sorted(near_heap, reverse=True)
        ]
        shortlist_source = "nearest_official_val_hard_gate_miss"

    best: tuple[float, dict] | None = None
    full_gate_pass_count = 0
    evaluated_full_audits = 0
    for _, qset, combo_indices in shortlist:
        selected_lanes: list[LaneShape] = []
        for group, combo_idx in zip(group_names, combo_indices):
            selected_lanes.extend(group_combos[group][int(combo_idx)])
        prototypes = []
        group_seen: dict[str, int] = {}
        for lane in selected_lanes:
            idx = group_seen.get(lane.group, 0)
            prototypes.append(_lane_package(lane, idx))
            group_seen[lane.group] = idx + 1
        mapping, refs, assignment_cost = _best_query_assignment(base_refs, prototypes, qset)
        official_val_audit = _audit_refs(base_refs, refs, val_lanes)
        train_selection_audit = _audit_refs(base_refs, refs, train_score_lanes)
        evaluated_full_audits += 1
        if official_val_audit["gate"]["pass"]:
            full_gate_pass_count += 1
        score = _static_gate_candidate_score(official_val_audit, qset, query_penalties=query_penalties)
        score += 0.001 * assignment_cost
        if best is None or score < best[0]:
            best = (
                score,
                {
                    "refs": refs,
                    "mapping": mapping,
                    "prototypes": prototypes,
                    "qset": qset,
                    "assignment_cost": assignment_cost,
                    "official_val_audit": official_val_audit,
                    "train_selection_audit": train_selection_audit,
                    "score": score,
                },
            )
    if best is None:
        raise RuntimeError("Static-gate candidate enumeration produced no candidates.")
    result = best[1]
    result["selection_stats"] = {
        "selection_method": "train_prototypes_enumerated_official_val_static_gate",
        "shortlist_source": shortlist_source,
        "qsets_evaluated": len(qsets),
        "hard_gate_candidate_count": hard_gate_candidate_count,
        "shortlisted_candidate_count": len(shortlist),
        "full_audits_evaluated": evaluated_full_audits,
        "full_gate_pass_count": full_gate_pass_count,
        "top_k": int(top_k),
        "group_combination_counts": {group: len(group_combos[group]) for group in group_names},
        "group_train_hard_counts": {group: len(group_lanes[group]) for group in group_names},
        "required_replace_queries": sorted(int(q) for q in required_queries),
        "query_static_penalty": {str(int(q)): float(v) for q, v in sorted(query_penalties.items())},
        "protected_queries": sorted(keep_queries),
    }
    return result


def _audit_refs(base_refs: np.ndarray, new_refs: np.ndarray, lanes: list[LaneShape]) -> dict:
    groups = _build_eval_groups(lanes)
    per_group = {}
    for name, group_lanes in groups.items():
        per_group[name] = {
            "base": _metrics(base_refs, group_lanes),
            "new": _metrics(new_refs, group_lanes),
        }
    hard_base = per_group["hard_target"]["base"]
    hard_new = per_group["hard_target"]["new"]
    normal_base = per_group["normal_lane"]["base"]
    normal_new = per_group["normal_lane"]["new"]
    normal_regression = {
        "base_p50": normal_base["p50"],
        "new_p50": normal_new["p50"],
        "delta_p50": None if normal_base["p50"] is None or normal_new["p50"] is None else normal_new["p50"] - normal_base["p50"],
        "base_p90": normal_base["p90"],
        "new_p90": normal_new["p90"],
        "delta_p90": None if normal_base["p90"] is None or normal_new["p90"] is None else normal_new["p90"] - normal_base["p90"],
        "count": normal_new["count"],
    }
    gate = {
        "hard_p50_lt_30": hard_new["p50"] is not None and hard_new["p50"] < 30.0,
        "hard_p90_lt_50": hard_new["p90"] is not None and hard_new["p90"] < 50.0,
        "hard_match20_gain_ge_0p20": (
            hard_base["match20"] is not None
            and hard_new["match20"] is not None
            and hard_new["match20"] - hard_base["match20"] >= 0.20 - 1e-9
        ),
        "normal_p50_regression_le_5": normal_regression["delta_p50"] is None or normal_regression["delta_p50"] <= 5.0,
        "normal_p90_regression_le_10": normal_regression["delta_p90"] is None or normal_regression["delta_p90"] <= 10.0,
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    return {
        "base_nearest_reference_ape_p50": hard_base["p50"],
        "base_nearest_reference_ape_p90": hard_base["p90"],
        "new_nearest_reference_ape_p50": hard_new["p50"],
        "new_nearest_reference_ape_p90": hard_new["p90"],
        "base_nearest_reference_match20": hard_base["match20"],
        "new_nearest_reference_match20": hard_new["match20"],
        "per_group": per_group,
        "nearest_ref_query_hist": {
            "base": hard_base["nearest_ref_query_hist"],
            "new": hard_new["nearest_ref_query_hist"],
        },
        "normal_lane_regression": normal_regression,
        "gate": gate,
    }


def main() -> None:
    args = parse_args()
    num_queries = int(args.num_queries)
    num_points = int(args.num_points)
    if num_queries != 12 or num_points != 56:
        raise ValueError("Q12-dataref-ultrashort-v1 supports only num_queries=12 and num_points=56.")
    replace_queries = _parse_int_list(args.replace_queries)
    keep_queries = set(_parse_int_list(args.keep_queries))
    required_queries = set(_parse_int_list(args.required_replace_queries))
    query_penalties = _parse_query_penalties(args.query_static_penalty)
    if sorted(set(replace_queries)) != sorted(replace_queries):
        raise ValueError(f"replace-queries contains duplicates: {replace_queries}")
    if set(replace_queries) & keep_queries:
        raise ValueError(f"replace-queries overlap keep-queries: {sorted(set(replace_queries) & keep_queries)}")
    if required_queries - set(replace_queries):
        raise ValueError(
            f"required-replace-queries must be a subset of replace-queries; "
            f"missing={sorted(required_queries - set(replace_queries))}"
        )
    if required_queries & keep_queries:
        raise ValueError(
            f"required-replace-queries overlap keep-queries: {sorted(required_queries & keep_queries)}"
        )

    train_json_paths = [Path(p) for p in args.train_label_json]
    val_json_paths = [Path(p) for p in args.val_gt_json]
    train_diag_paths = [Path(p) for p in args.train_diag_csv]
    val_diag_paths = [Path(p) for p in args.val_diag_csv]
    train_index = _load_gt_index(train_json_paths)
    val_index = _load_gt_index(val_json_paths)
    val_raw_files = set(val_index)

    train_lanes_by_key: dict[tuple[str, int], LaneShape] = {}
    train_audit_lanes_by_key: dict[tuple[str, int], LaneShape] = {}
    skipped = {"missing_gt_record": 0, "too_few_points": 0, "official_val_excluded": 0, "not_target_group": 0, "not_hard": 0}
    for row in _read_diag_rows(train_diag_paths):
        raw_file = _norm_raw_file(row.get("raw_file", ""))
        if raw_file in val_raw_files:
            skipped["official_val_excluded"] += 1
            continue
        lane = _shape_from_row(row, train_index, num_points)
        if lane is None:
            if raw_file not in train_index:
                skipped["missing_gt_record"] += 1
            else:
                skipped["too_few_points"] += 1
            continue
        train_audit_lanes_by_key[(lane.raw_file, lane.gt_lane_id)] = lane
        if lane.group not in GROUP_K:
            skipped["not_target_group"] += 1
            continue
        if not lane.hard:
            skipped["not_hard"] += 1
            continue
        train_lanes_by_key[(lane.raw_file, lane.gt_lane_id)] = lane
    train_lanes = list(train_lanes_by_key.values())
    train_audit_lanes = list(train_audit_lanes_by_key.values())
    train_normal_score_lanes = _sample_train_normal_lanes(train_audit_lanes, TRAIN_NORMAL_SCORE_CAP)
    train_score_lanes = [*train_lanes, *train_normal_score_lanes]

    val_lanes: list[LaneShape] = []
    for row in _read_diag_rows(val_diag_paths):
        lane = _shape_from_row(row, val_index, num_points)
        if lane is not None:
            val_lanes.append(lane)
    if not val_lanes:
        raise ValueError("No official-val lanes could be joined from --val-diag-csv and --val-gt-json.")

    if len(replace_queries) < sum(GROUP_K.values()):
        raise ValueError(f"Need at least {sum(GROUP_K.values())} replace queries, got {replace_queries}.")

    base_refs = _default_references(num_queries, num_points)
    selection = _select_static_gate_candidate(
        base_refs=base_refs,
        replace_queries=replace_queries,
        keep_queries=keep_queries,
        required_queries=required_queries,
        query_penalties=query_penalties,
        train_lanes=train_lanes,
        train_score_lanes=train_score_lanes,
        val_lanes=val_lanes,
        top_k=int(args.static_gate_top_k),
    )
    best_refs = selection["refs"]
    best_mapping: dict[int, dict] = selection["mapping"]
    best_prototypes: list[dict] = selection["prototypes"]
    train_selection_audit = selection["train_selection_audit"]
    audit = selection["official_val_audit"]
    best_score_details = {
        "score": float(selection["score"]),
        "assignment_identity_tie_penalty": float(selection["assignment_cost"]) * 0.001,
        "assignment_cost_px": float(selection["assignment_cost"]),
        "assignment_qset": [int(q) for q in selection["qset"]],
        "selection_stats": selection["selection_stats"],
        "base_hard_target": train_selection_audit["per_group"]["hard_target"]["base"],
        "new_hard_target": train_selection_audit["per_group"]["hard_target"]["new"],
        "base_normal_lane": train_selection_audit["per_group"]["normal_lane"]["base"],
        "new_normal_lane": train_selection_audit["per_group"]["normal_lane"]["new"],
    }

    replacements = {
        str(query_id): {
            **best_mapping[query_id],
            "x_norm": [round(float(x), 8) for x in best_refs[query_id].tolist()],
        }
        for query_id in sorted(best_mapping)
    }
    bank = {
        "schema": SCHEMA,
        "num_queries": num_queries,
        "num_points": num_points,
        "point_mode": "fixed_y",
        "fixed_y_px_desc": [int(x) for x in FIXED_Y_DESC.tolist()],
        "x_norm": [[round(float(x), 8) for x in row] for row in best_refs.tolist()],
        "replacements": replacements,
        "base": "default_q12_linear_perspective",
        "provenance": {
            "generated_from": "train hard lanes only",
            "official_val_usage": "static_gate_selection_and_audit_only_no_prototype_generation",
            "hard_filter": "raw_has_match20 false OR raw_best_valid_len@0.6 < min_points OR point_valid_recall@0.6 < 0.75",
            "hard_survival_note": "raw_best_valid_count@0.6 is diagnostic only; raw_best_valid_len@0.6 is the contiguous-span survival field",
            "assignment_method": "enumerated_train_prototypes_official_val_static_gate",
            "assignment_score_lanes": "official-val hard target gate plus official-val normal-lane regression audit",
            "required_replace_queries": sorted(int(q) for q in required_queries),
            "query_static_penalty": {str(int(q)): float(v) for q, v in sorted(query_penalties.items())},
            "train_normal_score_cap": TRAIN_NORMAL_SCORE_CAP,
            "excluded_official_val_raw_files": len(val_raw_files),
            "train_diag_csv": [str(p) for p in train_diag_paths],
            "val_diag_csv": [str(p) for p in val_diag_paths],
            "train_label_json": [str(p) for p in train_json_paths],
            "val_gt_json": [str(p) for p in val_json_paths],
            "input_sha256": {str(p): _sha256(p) for p in [*train_diag_paths, *val_diag_paths, *train_json_paths, *val_json_paths]},
        },
    }

    audit = _audit_refs(base_refs, best_refs, val_lanes)
    audit["assignment_mapping"] = {str(k): v for k, v in sorted(best_mapping.items())}
    train_selection_audit = _audit_refs(base_refs, best_refs, train_score_lanes)
    train_selection_audit["assignment_mapping"] = {str(k): v for k, v in sorted(best_mapping.items())}
    train_selection_audit["assignment_score"] = best_score_details
    train_pool_by_group = {
        group: [lane for lane in train_lanes if lane.group == group]
        for group in GROUP_K
    }
    selection_stats = best_score_details["selection_stats"]
    summary = {
        "schema": "gcs_q12_ultrashort_reference_bank_build_summary_v1",
        "bank_path": str(Path(args.save_bank)),
        "candidate_assignments_evaluated": selection_stats["hard_gate_candidate_count"],
        "coverage_sets_evaluated": selection_stats["full_audits_evaluated"],
        "replace_queries": replace_queries,
        "keep_queries": sorted(keep_queries),
        "required_replace_queries": sorted(int(q) for q in required_queries),
        "query_static_penalty": {str(int(q)): float(v) for q, v in sorted(query_penalties.items())},
        "selection_stats": selection_stats,
        "prototype_pool_per_group": {group: len(train_pool_by_group[group]) for group in GROUP_K},
        "selected_prototype_counts": {group: sum(1 for p in best_prototypes if p["group"] == group) for group in GROUP_K},
        "train_hard_target_count": len(train_lanes),
        "train_hard_target_by_group": {group: sum(1 for lane in train_lanes if lane.group == group) for group in GROUP_K},
        "train_score_lane_count": len(train_score_lanes),
        "train_normal_score_lane_count": len(train_normal_score_lanes),
        "skipped_train_rows": skipped,
        "prototype_pool": {
            group: [
                {k: v for k, v in _lane_package(lane, idx).items() if k != "x_norm"}
                for idx, lane in enumerate(train_pool_by_group[group])
            ]
            for group in GROUP_K
        },
        "train_selection_audit": train_selection_audit,
        "prototypes": [{k: v for k, v in p.items() if k != "x_norm"} for p in best_prototypes],
        "official_val_audit": audit,
        "do_not_train_unless_gate_passes": not bool(audit["gate"]["pass"]),
    }

    bank_path = Path(args.save_bank)
    summary_path = Path(args.save_summary)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"bank": str(bank_path), "summary": str(summary_path), "gate": audit["gate"]}, indent=2))


if __name__ == "__main__":
    main()
