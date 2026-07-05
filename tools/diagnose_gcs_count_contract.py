from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    DEFAULT_OFFICIAL_SCORE_FN_WEIGHT,
    DEFAULT_OFFICIAL_SCORE_FP_WEIGHT,
    TUSIMPLE_ORIGINAL_SHAPE,
    TuSimpleOfficialLaneEval,
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    official_metric_score,
    read_tusimple_json_lines,
    resolve_tusimple_gt_json,
)
from tools.eval_gcs import label_path_for_image, load_gcs_label  # noqa: E402
from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image  # noqa: E402
from ultralytics.utils.gcs_fixed_y import validate_training_fixed_y_desc  # noqa: E402
from ultralytics.utils.gcs_count_contract import shortside_visible_decision  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions, lane_nms, longest_contiguous_valid_mask  # noqa: E402
from ultralytics.utils.gcs_shape import assert_gcs_shape, normalize_imgsz  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_DATASET_ROOT = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544"
TARGET_BUCKETS = ("GT4_4->3", "GT4_4->5", "GT5_5->4", "GT5_5->6")
REQUIRED_LABEL_FIELDS = {
    "semantic_mask",
    "edge_mask",
    "lanes",
    "lane_valid",
    "num_lanes",
    "point_mode",
    "fixed_y",
    "num_points",
    "raw_file",
    "image_shape",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose GT4/GT5 GCS count-contract failures by query decode stage.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--splits", nargs="+", default=["val"], choices=("train", "val", "test"))
    parser.add_argument(
        "--allow-test-diagnostics",
        action="store_true",
        help="Allow detailed test split diagnostics. This is reporting-only and must not be used for model selection.",
    )
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--point-valid-thr", type=float, default=0.5)
    parser.add_argument("--nms-dist-px", type=float, default=0.0)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--valid-before-maxdet", action="store_true")
    parser.add_argument("--raw-match-dist-px", type=float, default=30.0)
    parser.add_argument("--side-raw-match-dist-px", type=float, default=40.0)
    parser.add_argument("--shortside-min-gt-lanes", type=int, default=4)
    parser.add_argument("--shortside-min-valid-points", type=int, default=6)
    parser.add_argument("--shortside-ultra-min-valid-points", type=int, default=2)
    parser.add_argument("--shortside-reliable-min-valid-points", type=int, default=None)
    parser.add_argument("--shortside-ultra-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shortside-ultra-score-floor-gain", type=float, default=0.0)
    parser.add_argument("--shortside-ultra-valid-gain", type=float, default=0.0)
    parser.add_argument("--shortside-ultra-rank-pos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shortside-visible-max", type=int, default=42)
    parser.add_argument("--shortside-use-median", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shortside-median-margin", type=float, default=0.0)
    parser.add_argument("--rank-focus-shortside", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rank-pos-scope",
        choices=("shortside_reliable", "shortside_with_ultra", "gt4gt5_matched", "all_matched"),
        default="shortside_reliable",
    )
    parser.add_argument("--rank-gt-min-lanes", type=int, default=4)
    parser.add_argument("--rank-side-duplicate-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rank-side-dup-margin-px", type=float, default=5.0)
    parser.add_argument("--rank-side-dup-min-overlap", type=int, default=6)
    parser.add_argument(
        "--rank-pair-reduction",
        choices=("global_pair_mean", "image_mean"),
        default="global_pair_mean",
    )
    parser.add_argument("--base-ignore-raw-rescue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base-ignore-rank-near", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base-ignore-farspur-near", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base-ignore-duplicate-like", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--legacy-spurious-active", action="store_true")
    parser.add_argument("--near-dist-px", type=float, default=40.0)
    parser.add_argument("--side-ignore-dist-px", type=float, default=60.0)
    parser.add_argument("--clear-dist-px", type=float, default=80.0)
    parser.add_argument("--farspur-score-thr", type=float, default=0.05)
    parser.add_argument("--farspur-min-valid-points", type=int, default=2)
    parser.add_argument("--rank-dup-close-px", type=float, default=30.0)
    parser.add_argument("--rank-dup-min-overlap", type=int, default=6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0, help="Per-split image limit. 0 means all.")
    parser.add_argument("--archive-root", default=str(ROOT / "archive"), help="TuSimple archive root for optional official metrics.")
    parser.add_argument("--official-gt-json", default=None, help="Optional TuSimple GT json-lines path. Use with one split.")
    parser.add_argument("--official-runtime-ms", type=float, default=1.0)
    parser.add_argument("--score-fp-weight", type=float, default=DEFAULT_OFFICIAL_SCORE_FP_WEIGHT)
    parser.add_argument("--score-fn-weight", type=float, default=DEFAULT_OFFICIAL_SCORE_FN_WEIGHT)
    parser.add_argument("--save-dir", required=True)
    return parser.parse_args()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default, allow_nan=False), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=json_default, allow_nan=False) + "\n")


def clean_float(value, ndigits: int = 6):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, ndigits)


def array_scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def load_label_meta(label_path: Path, expected_shape: tuple[int, int]) -> dict:
    meta = {"raw_file": "", "image_shape": None, "label_contract": "validated_k56_fixed_y"}
    with np.load(label_path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_LABEL_FIELDS.difference(data.files))
        if missing:
            raise KeyError(f"{label_path} missing required K56 fixed-y label fields: {missing}")

        raw_file = array_scalar_str(data["raw_file"])
        if not raw_file:
            raise ValueError(f"{label_path}: raw_file must be non-empty for count-contract diagnostics.")
        meta["raw_file"] = raw_file

        point_mode = array_scalar_str(data["point_mode"]).lower()
        if point_mode != "fixed_y":
            raise ValueError(f"{label_path}: point_mode must be 'fixed_y', got {point_mode!r}.")
        meta["point_mode"] = point_mode

        num_points = int(np.asarray(data["num_points"]).reshape(-1)[0])
        if num_points != 56:
            raise ValueError(f"{label_path}: num_points must be 56 for K56 diagnostics, got {num_points}.")
        meta["num_points"] = num_points

        lanes = np.asarray(data["lanes"])
        lane_valid = np.asarray(data["lane_valid"])
        if lanes.ndim != 3 or lanes.shape[1:] != (56, 2):
            raise ValueError(f"{label_path}: lanes must have shape N x 56 x 2, got {lanes.shape}.")
        if lane_valid.shape != lanes.shape[:2]:
            raise ValueError(f"{label_path}: lane_valid shape {lane_valid.shape} must match lanes N x 56.")
        num_lanes = int(np.asarray(data["num_lanes"]).reshape(-1)[0])
        if num_lanes != int(lanes.shape[0]):
            raise ValueError(f"{label_path}: num_lanes={num_lanes} does not match lanes.shape[0]={lanes.shape[0]}.")
        meta["num_lanes"] = num_lanes

        shape = np.asarray(data["image_shape"]).reshape(-1)
        if shape.size < 2:
            raise ValueError(f"{label_path}: image_shape must contain H,W, got {shape.tolist()}.")
        stored_shape = (int(shape[0]), int(shape[1]))
        expected_shape = (int(expected_shape[0]), int(expected_shape[1]))
        if stored_shape != expected_shape:
            raise ValueError(f"{label_path}: image_shape={stored_shape} does not match image/imgsz shape={expected_shape}.")
        meta["image_shape"] = [stored_shape[0], stored_shape[1]]

        for field in ("semantic_mask", "edge_mask"):
            field_shape = np.asarray(data[field]).shape[-2:]
            if tuple(int(x) for x in field_shape) != expected_shape:
                raise ValueError(f"{label_path}: {field} shape {field_shape} does not match {expected_shape}.")

        validate_training_fixed_y_desc(data["fixed_y"], name=f"{label_path}: fixed_y")
    return meta


def visible_span(valid: np.ndarray) -> dict:
    idx = np.where(np.asarray(valid) > 0.5)[0]
    if idx.size == 0:
        return {"start_idx": None, "end_idx": None, "count": 0}
    return {"start_idx": int(idx[0]), "end_idx": int(idx[-1]), "count": int(idx.size)}


def side_ranks_by_lower_x(gt_lanes: np.ndarray, gt_valid: np.ndarray) -> dict[int, int]:
    if gt_lanes.size == 0:
        return {}
    k = int(gt_lanes.shape[1])
    # K56 fixed-y is bottom-to-top: index 0 is bottom y=710, index K-1 is top y=160.
    # Therefore lower/bottom half is indices < K//2.
    lower = np.arange(k) < (k // 2)
    xs = []
    for lane_i, (lane, valid) in enumerate(zip(gt_lanes, gt_valid)):
        valid_mask = valid > 0.5
        if not valid_mask.any():
            continue
        lower_valid = valid_mask & lower
        use_valid = lower_valid if lower_valid.any() else valid_mask
        xs.append((lane_i, float(lane[use_valid, 0].mean())))
    xs.sort(key=lambda x: (x[1], x[0]))
    return {lane_i: rank for rank, (lane_i, _) in enumerate(xs)}


def longest_valid_count(valid_scores: torch.Tensor, point_valid_thr: float, min_points: int) -> int:
    mask = longest_contiguous_valid_mask(valid_scores >= float(point_valid_thr), min_points=min_points)
    return int(mask.sum().item())


def sorted_query_arrays(
    pred_points_t: torch.Tensor,
    pred_valid_t: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    points = pred_points_t.detach().float().cpu().clamp(0.0, 1.0)
    valid_scores = pred_valid_t.detach().float().cpu().sigmoid()
    sorted_points = []
    sorted_valid = []
    for q in range(points.shape[0]):
        order = torch.argsort(points[q, :, 1], descending=True, stable=True)
        sorted_points.append(points[q][order].numpy().astype(np.float32))
        sorted_valid.append(valid_scores[q][order].numpy().astype(np.float32))
    return np.stack(sorted_points, axis=0), np.stack(sorted_valid, axis=0)


def anchor_query_arrays(
    pred_points_t: torch.Tensor,
    pred_valid_t: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return query geometry in K56 anchor order, matching training loss role masks."""
    points = pred_points_t.detach().float().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    valid_scores = pred_valid_t.detach().float().cpu().sigmoid().numpy().astype(np.float32)
    return points, valid_scores


def mean_dx_to_gt_px(pred_lane: np.ndarray, gt_lane: np.ndarray, gt_valid: np.ndarray, width: int) -> float:
    mask = gt_valid > 0.5
    if not mask.any():
        return float("inf")
    return float(np.abs(pred_lane[mask, 0] - gt_lane[mask, 0]).mean() * float(width))


def query_gt_distance(
    pred_lane: np.ndarray,
    pred_valid: np.ndarray,
    gt_lanes: np.ndarray,
    gt_valid: np.ndarray,
    width: int,
    point_valid_thr: float,
) -> tuple[float, int, float, float]:
    nearest_gt = -1
    min_dist = float("inf")
    min_bottom = float("inf")
    side_min = float("inf")
    ranks = side_ranks_by_lower_x(gt_lanes, gt_valid)
    side_ids = {gt_i for gt_i, rank in ranks.items() if rank in {0, max(ranks.values(), default=0)}}
    valid_q = pred_valid >= float(point_valid_thr)
    for gt_i, (lane, valid) in enumerate(zip(gt_lanes, gt_valid)):
        lane_valid = valid > 0.5
        if not lane_valid.any():
            continue
        raw = mean_dx_to_gt_px(pred_lane, lane, valid, width)
        common = valid_q & lane_valid
        common_dist = float(np.abs(pred_lane[common, 0] - lane[common, 0]).mean() * float(width)) if common.any() else float("inf")
        dist = min(raw, common_dist)
        if dist < min_dist:
            min_dist = dist
            nearest_gt = int(gt_i)
        bottom_idx = int(np.where(lane_valid)[0][0])
        bottom_dx = float(abs(pred_lane[bottom_idx, 0] - lane[bottom_idx, 0]) * float(width))
        min_bottom = min(min_bottom, bottom_dx)
        if gt_i in side_ids:
            side_min = min(side_min, dist, bottom_dx)
    return min_dist, nearest_gt, min_bottom, side_min


def query_stage_trace(
    pred_points_t: torch.Tensor,
    pred_logits_t: torch.Tensor,
    pred_valid_t: torch.Tensor,
    image_shape: tuple[int, int],
    args: argparse.Namespace,
) -> dict:
    if pred_logits_t.ndim == 2 and pred_logits_t.shape[-1] == 1:
        pred_logits_t = pred_logits_t.squeeze(-1)
    points = pred_points_t.detach().float().cpu().clamp(0.0, 1.0)
    scores = pred_logits_t.detach().float().cpu().sigmoid()
    valid_scores = pred_valid_t.detach().float().cpu().sigmoid()
    query_indices = torch.arange(points.shape[0], dtype=torch.long)
    stage = {
        "conf": set(),
        "nms": set(),
        "valid": set(),
        "maxdet": set(),
        "final": set(),
        "rank_by_score": {},
        "rank_after_valid": {},
        "valid_count": {},
        "point_valid_any": {},
    }
    for rank, q in enumerate(torch.argsort(scores, descending=True).tolist(), start=1):
        stage["rank_by_score"][int(q)] = int(rank)
    for q in range(points.shape[0]):
        order_q = torch.argsort(points[q, :, 1], descending=True, stable=True)
        valid_count = longest_valid_count(valid_scores[q][order_q], args.point_valid_thr, args.min_points)
        stage["valid_count"][int(q)] = int(valid_count)
        stage["point_valid_any"][int(q)] = int((valid_scores[q][order_q] >= float(args.point_valid_thr)).any().item())

    keep = torch.nonzero(scores >= float(args.conf), as_tuple=False).flatten()
    stage["conf"] = set(int(x) for x in keep.tolist())
    if keep.numel() == 0:
        return stage

    sorted_points = []
    sorted_valid_scores = []
    for i in keep:
        order_i = torch.argsort(points[i, :, 1], descending=True, stable=True)
        sorted_points.append(points[i][order_i])
        sorted_valid_scores.append(valid_scores[i][order_i])
    points = torch.stack(sorted_points, dim=0)
    valid_scores = torch.stack(sorted_valid_scores, dim=0)
    scores = scores[keep]
    query_indices = query_indices[keep]
    order = torch.argsort(scores, descending=True)

    if args.nms_dist_px > 0.0:
        sorted_points = points[order]
        sorted_scores = scores[order]
        sorted_queries = query_indices[order]
        sorted_valid = valid_scores[order]
        valid_masks = torch.stack(
            [
                longest_contiguous_valid_mask(v >= float(args.point_valid_thr), min_points=args.min_points)
                for v in sorted_valid
            ],
            dim=0,
        )
        keep_sorted = lane_nms(
            sorted_points,
            sorted_scores,
            image_shape=image_shape,
            dist_thr_px=float(args.nms_dist_px),
            valid_masks=valid_masks,
        )
        points = sorted_points[keep_sorted]
        scores = sorted_scores[keep_sorted]
        query_indices = sorted_queries[keep_sorted]
        valid_scores = sorted_valid[keep_sorted]
        order = torch.arange(scores.shape[0], dtype=torch.long)
    stage["nms"] = set(int(x) for x in query_indices.tolist())

    if args.valid_before_maxdet:
        valid_masks = torch.stack(
            [
                longest_contiguous_valid_mask(v >= float(args.point_valid_thr), min_points=args.min_points)
                for v in valid_scores
            ],
            dim=0,
        )
        keep_valid = torch.nonzero(valid_masks.sum(dim=1) >= int(args.min_points), as_tuple=False).flatten()
        points = points[keep_valid]
        scores = scores[keep_valid]
        query_indices = query_indices[keep_valid]
        valid_scores = valid_scores[keep_valid]
        order = torch.argsort(scores, descending=True)
    stage["valid"] = set(int(x) for x in query_indices.tolist())
    for rank, q in enumerate(query_indices[torch.argsort(scores, descending=True)].tolist(), start=1):
        stage["rank_after_valid"][int(q)] = int(rank)

    if args.max_det is not None and args.max_det > 0:
        order = order[: int(args.max_det)]
    query_indices = query_indices[order]
    valid_scores = valid_scores[order]
    stage["maxdet"] = set(int(x) for x in query_indices.tolist())
    final = []
    for lane_i, q in enumerate(query_indices.tolist()):
        valid_mask = longest_contiguous_valid_mask(
            valid_scores[lane_i] >= float(args.point_valid_thr),
            min_points=args.min_points,
        )
        if int(valid_mask.sum().item()) >= int(args.min_points):
            final.append(int(q))
    stage["final"] = set(final)
    return stage


def best_raw_query_for_gt(
    pred_points: np.ndarray,
    gt_lanes: np.ndarray,
    gt_valid: np.ndarray,
    gt_idx: int,
    width: int,
) -> tuple[int, float]:
    distances = [mean_dx_to_gt_px(pred_points[q], gt_lanes[gt_idx], gt_valid[gt_idx], width) for q in range(pred_points.shape[0])]
    if not distances:
        return -1, float("inf")
    q = int(np.argmin(np.asarray(distances, dtype=np.float32)))
    return q, float(distances[q])


def shortside_contract_decision(
    gt_count: int,
    is_side: bool,
    visible: int,
    visible_counts: list[int],
    args: argparse.Namespace,
):
    decision = shortside_visible_decision(
        visible,
        visible_counts,
        min_valid_points=int(args.shortside_min_valid_points),
        ultra_min_valid_points=int(args.shortside_ultra_min_valid_points),
        reliable_min_valid_points=int(
            args.shortside_reliable_min_valid_points
            if args.shortside_reliable_min_valid_points is not None
            else args.shortside_min_valid_points
        ),
        visible_max=int(args.shortside_visible_max),
        use_median=bool(args.shortside_use_median),
        median_margin=float(args.shortside_median_margin),
    )
    if gt_count < int(args.shortside_min_gt_lanes) or not is_side:
        return type(decision)(
            eligible=False,
            selected_by_abs=False,
            selected_by_median=False,
            visible=decision.visible,
            image_visible_median=decision.image_visible_median,
        )
    return decision


def bucket_name(gt_count: int, pred_count: int) -> str | None:
    if gt_count == 4 and pred_count == 3:
        return "GT4_4->3"
    if gt_count == 4 and pred_count == 5:
        return "GT4_4->5"
    if gt_count == 5 and pred_count == 4:
        return "GT5_5->4"
    if gt_count == 5 and pred_count == 6:
        return "GT5_5->6"
    return None


def classify_drop_stage(raw_ok: bool, q: int, stage: dict, final_query_kept: bool, args: argparse.Namespace) -> tuple[str, str]:
    return classify_drop_stage_for_gt(raw_ok, q, stage, final_query_kept, False, args)


def classify_drop_stage_for_gt(
    raw_ok: bool,
    q: int,
    stage: dict,
    final_query_kept: bool,
    final_gt_matched: bool,
    args: argparse.Namespace,
) -> tuple[str, str]:
    if final_gt_matched:
        return "kept", "kept"
    if not raw_ok:
        return "raw_match", "raw_match_missing"
    if q < 0:
        return "raw_match", "no_matched_query"
    if q not in stage["conf"]:
        return "after_conf", "exist_score_below_conf"
    if q not in stage["nms"]:
        return "after_nms", "lane_nms_suppressed"
    if args.valid_before_maxdet:
        if not bool(stage["point_valid_any"].get(q, 0)):
            return "after_point_valid", "no_pred_valid_anchor_before_maxdet"
        if int(stage["valid_count"].get(q, 0)) < int(args.min_points):
            return "after_min_points", "pred_valid_points_below_min_points_before_maxdet"
    if q not in stage["maxdet"]:
        return "after_maxdet", "score_rank_below_max_det"
    if not args.valid_before_maxdet:
        if not bool(stage["point_valid_any"].get(q, 0)):
            return "after_point_valid", "no_pred_valid_anchor_after_maxdet"
        if int(stage["valid_count"].get(q, 0)) < int(args.min_points):
            return "after_min_points", "pred_valid_points_below_min_points_after_maxdet"
    if not final_query_kept:
        return "final_decode", "final_min_points_or_geometry_drop"
    return "kept", "kept"


def record_stage_flags(
    raw_ok: bool,
    q: int,
    stage: dict,
    final_query_kept: bool,
    final_gt_matched: bool,
    args: argparse.Namespace,
) -> dict[str, bool]:
    raw = bool(raw_ok and q >= 0)
    after_conf = bool(raw and q in stage["conf"])
    after_nms = bool(after_conf and q in stage["nms"])
    point_valid_ok = bool(after_nms and stage["point_valid_any"].get(q, 0))
    min_points_ok = bool(point_valid_ok and int(stage["valid_count"].get(q, 0)) >= int(args.min_points))
    if args.valid_before_maxdet:
        after_point_valid = point_valid_ok
        after_min_points = min_points_ok
        after_maxdet = bool(after_min_points and q in stage["maxdet"])
    else:
        after_maxdet = bool(after_nms and q in stage["maxdet"])
        after_point_valid = bool(after_maxdet and stage["point_valid_any"].get(q, 0))
        after_min_points = bool(after_point_valid and int(stage["valid_count"].get(q, 0)) >= int(args.min_points))
    return {
        "raw_match": bool(raw_ok),
        "after_point_valid": after_point_valid,
        "after_min_points": after_min_points,
        "after_conf": after_conf,
        "after_maxdet": after_maxdet,
        "after_nms": after_nms,
        "final_query": bool(raw_ok and final_query_kept),
        "final_gt_matched": bool(raw_ok and final_gt_matched),
        "final_decode": bool(raw_ok and final_gt_matched),
    }


def final_match_for_gt(
    decoded_lanes: list[dict],
    gt_lane: np.ndarray,
    gt_valid: np.ndarray,
    width: int,
    args: argparse.Namespace,
    *,
    match_dist_px: float,
) -> bool:
    for lane in decoded_lanes:
        pred = np.asarray(lane["points_norm"], dtype=np.float32)
        valid = np.asarray(lane.get("point_valid", np.ones(pred.shape[0])), dtype=np.float32)
        common = (valid > 0.5) & (gt_valid > 0.5)
        if int(common.sum()) < int(args.min_points):
            continue
        dx = float(np.abs(pred[common, 0] - gt_lane[common, 0]).mean() * float(width))
        if dx <= float(match_dist_px):
            return True
    return False


def namespace_to_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def make_contract_loss(model: torch.nn.Module, args: argparse.Namespace, imgsz: tuple[int, int]) -> GCSLoss:
    """Build a diagnostic GCSLoss instance that only supplies matcher/mask logic."""
    cfg = namespace_to_dict(getattr(model, "args", None))
    cfg.update(
        {
        "gcs_imgsz": [int(imgsz[0]), int(imgsz[1])],
        "gcs_shortside_rawmatch_boost": 0.0,
        "gcs_shortside_rawmatch_dist_px": float(args.raw_match_dist_px),
        "gcs_shortside_side_dist_px": float(args.side_raw_match_dist_px),
        "gcs_shortside_max_valid_points": int(args.shortside_visible_max),
        "gcs_shortside_min_gt_lanes": int(args.shortside_min_gt_lanes),
        "gcs_shortside_min_valid_points": int(args.shortside_min_valid_points),
        "gcs_shortside_ultra_min_valid_points": int(args.shortside_ultra_min_valid_points),
        "gcs_shortside_reliable_min_valid_points": (
            None if args.shortside_reliable_min_valid_points is None else int(args.shortside_reliable_min_valid_points)
        ),
        "gcs_shortside_ultra_enable": bool(args.shortside_ultra_enable),
        "gcs_shortside_ultra_score_floor_gain": float(args.shortside_ultra_score_floor_gain),
        "gcs_shortside_ultra_valid_gain": float(args.shortside_ultra_valid_gain),
        "gcs_shortside_ultra_rank_pos": bool(args.shortside_ultra_rank_pos),
        "gcs_shortside_visible_max": int(args.shortside_visible_max),
        "gcs_shortside_rawmatch_px": None,
        "gcs_shortside_side_rawmatch_px": None,
        "gcs_shortside_use_median": bool(args.shortside_use_median),
        "gcs_shortside_median_margin": float(args.shortside_median_margin),
        "gcs_spurious_neg": 0.0,
        "gcs_allow_legacy_spurious_with_new_contract": False,
        "gcs_farspur_ignore_first": True,
        "gcs_farspur_weight": 0.0,
        "gcs_farspur_score_thr": float(args.farspur_score_thr),
        "gcs_farspur_near_dist_px": float(args.near_dist_px),
        "gcs_farspur_side_ignore_dist_px": float(args.side_ignore_dist_px),
        "gcs_farspur_clear_dist_px": float(args.clear_dist_px),
        "gcs_farspur_min_valid_points": int(args.farspur_min_valid_points),
        "gcs_rank_topk_weight": 0.0,
        "gcs_rank_focus_shortside": bool(args.rank_focus_shortside),
        "gcs_rank_pos_scope": str(args.rank_pos_scope),
        "gcs_rank_gt_min_lanes": int(args.rank_gt_min_lanes),
        "gcs_rank_dup_close_px": float(args.rank_dup_close_px),
        "gcs_rank_dup_min_overlap": int(args.rank_dup_min_overlap),
        "gcs_rank_side_duplicate_enable": bool(args.rank_side_duplicate_enable),
        "gcs_rank_side_dup_margin_px": float(args.rank_side_dup_margin_px),
        "gcs_rank_side_dup_min_overlap": int(args.rank_side_dup_min_overlap),
        "gcs_rank_pair_reduction": str(args.rank_pair_reduction),
        "gcs_base_ignore_raw_rescue": bool(args.base_ignore_raw_rescue),
        "gcs_base_ignore_rank_near": bool(args.base_ignore_rank_near),
        "gcs_base_ignore_farspur_near": bool(args.base_ignore_farspur_near),
        "gcs_base_ignore_duplicate_like": bool(args.base_ignore_duplicate_like),
        "gcs_rank_near_gt_ignore_px": float(args.near_dist_px),
        "gcs_rank_side_ignore_px": float(args.side_ignore_dist_px),
        "gcs_eval_point_valid_thr": float(args.point_valid_thr),
        }
    )
    return GCSLoss(cfg)


def contract_scalar_count(masks: dict[str, torch.Tensor | None], name: str) -> int:
    """Return a scalar count from a count-contract mask dictionary."""
    value = masks.get(name)
    if not isinstance(value, torch.Tensor):
        return 0
    return int(float(value.detach().cpu().item()))


def contract_mask_count(masks: dict[str, torch.Tensor | None], name: str) -> int:
    """Return the number of true entries in a count-contract boolean mask."""
    value = masks.get(name)
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.detach().cpu().bool().sum().item())


def build_training_contract_masks(
    criterion: GCSLoss,
    pred_points_t: torch.Tensor,
    pred_logits_t: torch.Tensor,
    pred_valid_t: torch.Tensor,
    gt_lanes_np: np.ndarray,
    gt_valid_np: np.ndarray,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict[str, torch.Tensor | None]]:
    """Return training-equivalent Hungarian assignment and count-contract masks for one image."""
    device = pred_points_t.device
    pred_points_b = pred_points_t.unsqueeze(0)
    pred_logits_b = pred_logits_t.unsqueeze(0)
    pred_valid_b = pred_valid_t.unsqueeze(0)
    gt_points = [torch.as_tensor(gt_lanes_np, device=device, dtype=pred_points_t.dtype)]
    gt_valid = [torch.as_tensor(gt_valid_np, device=device, dtype=pred_points_t.dtype)]
    gt_count = pred_logits_t.new_tensor([float(gt_lanes_np.shape[0])])
    indices = criterion.matcher(pred_points_b, pred_logits_b, gt_points, gt_valid)
    masks = criterion.build_count_contract_masks(
        pred_points_b,
        pred_logits_b,
        pred_valid_b,
        gt_points,
        gt_valid,
        indices,
        gt_count,
    )
    return indices, masks


def query_role_flags(q: int, masks: dict[str, torch.Tensor | None]) -> dict[str, object]:
    """Return training-mask role flags for one query from B=1 count-contract masks."""
    def mask_bool(name: str) -> bool:
        value = masks.get(name)
        return bool(isinstance(value, torch.Tensor) and bool(value[0, q].detach().cpu().item()))

    gt_idx = -1
    hungarian_gt_idx = masks.get("hungarian_gt_idx")
    if isinstance(hungarian_gt_idx, torch.Tensor):
        gt_idx = int(hungarian_gt_idx[0, q].detach().cpu().item())
    raw_rescue_gt_idx = masks.get("raw_rescue_gt_idx")
    rescue_gt_idx = -1
    if isinstance(raw_rescue_gt_idx, torch.Tensor):
        rescue_gt_idx = int(raw_rescue_gt_idx[0, q].detach().cpu().item())
    selected_gt_idx_t = masks.get("shortside_selected_gt_idx")
    selected_gt_idx = -1
    if isinstance(selected_gt_idx_t, torch.Tensor):
        selected_gt_idx = int(selected_gt_idx_t[0, q].detach().cpu().item())
    selected_reason_t = masks.get("shortside_selected_reason_code")
    selected_reason_code = 0
    if isinstance(selected_reason_t, torch.Tensor):
        selected_reason_code = int(selected_reason_t[0, q].detach().cpu().item())
    selected_reason = {1: "hungarian_rawmatch", 2: "unmatched_raw_rescue"}.get(selected_reason_code, "none")
    return {
        "is_hungarian_positive": mask_bool("hungarian_pos"),
        "is_rank_positive": mask_bool("rank_pos"),
        "hungarian_gt_idx": gt_idx,
        "is_near_gt_corridor": mask_bool("near_gt_corridor"),
        "is_ambiguous_side_region": mask_bool("ambiguous_side_region"),
        "is_rank_near_gt_corridor": mask_bool("near_gt_corridor"),
        "is_rank_ambiguous_side_region": mask_bool("ambiguous_side_region"),
        "is_farspur_near_gt_corridor": mask_bool("farspur_near_gt_corridor"),
        "is_farspur_ambiguous_side_region": mask_bool("farspur_ambiguous_side_region"),
        "is_clear_far_spurious": mask_bool("clear_far_spurious"),
        "is_farspur_clear_far_spurious": mask_bool("farspur_clear_far_spurious"),
        "is_duplicate_like_rank_neg": mask_bool("duplicate_like"),
        "is_normal_duplicate_like_rank_neg": mask_bool("normal_duplicate_like"),
        "is_side_duplicate_like_rank_neg": mask_bool("side_duplicate_like"),
        "is_rank_side_ambiguous_ignored": mask_bool("rank_side_ambiguous_ignored"),
        "is_rank_negative": mask_bool("rank_neg"),
        "is_raw_rescue_query": mask_bool("raw_rescue_pos"),
        "raw_rescue_gt_idx": rescue_gt_idx,
        "selected_shortside_gt_idx": selected_gt_idx,
        "selected_shortside_reason": selected_reason,
    }


def selected_shortside_for_gt(gt_idx: int, masks: dict[str, torch.Tensor | None]) -> tuple[int, str]:
    """Return the training count-contract selected shortside query/reason for one GT lane."""
    selected_gt_idx_t = masks.get("shortside_selected_gt_idx")
    selected_reason_t = masks.get("shortside_selected_reason_code")
    if not isinstance(selected_gt_idx_t, torch.Tensor) or not isinstance(selected_reason_t, torch.Tensor):
        return -1, "none"
    selected_gt_idx = selected_gt_idx_t[0].detach().cpu()
    matches = (selected_gt_idx == int(gt_idx)).nonzero(as_tuple=False).reshape(-1)
    if int(matches.numel()) == 0:
        return -1, "none"
    q = int(matches[0].item())
    code = int(selected_reason_t[0, q].detach().cpu().item())
    return q, {1: "hungarian_rawmatch", 2: "unmatched_raw_rescue"}.get(code, "none")


def count_metrics_from_pairs(pairs: list[tuple[int, int]]) -> dict:
    n = max(len(pairs), 1)
    correct = sum(1 for gt, pred in pairs if int(gt) == int(pred))
    metrics = {
        "count_acc": round(float(correct) / float(n), 6),
        "count_confusion": {f"{gt}->{pred}": int(v) for (gt, pred), v in sorted(Counter(pairs).items())},
    }
    for gt_count in sorted({int(gt) for gt, _ in pairs}):
        rows = [pred for gt, pred in pairs if int(gt) == gt_count]
        metrics[f"count_acc_{gt_count}"] = round(sum(1 for pred in rows if int(pred) == gt_count) / max(len(rows), 1), 6)
    return metrics


def official_metrics_for_summaries(image_summaries: list[dict], args: argparse.Namespace) -> dict:
    """Compute optional TuSimple official metrics from per-image decoded prediction records."""
    pred_records = [item.get("tusimple_prediction") for item in image_summaries if item.get("tusimple_prediction")]
    if not pred_records:
        return {"available": False, "reason": "no_tusimple_prediction_records"}
    splits = sorted({str(item.get("split")) for item in image_summaries})
    if len(splits) != 1:
        return {"available": False, "reason": "official_metrics_require_one_split", "splits": splits}
    split = splits[0]
    try:
        archive_root = find_tusimple_archive_root(args.archive_root)
        gt_json = resolve_tusimple_gt_json(archive_root, split=split, gt_json=args.official_gt_json)
        gt_records_all = read_tusimple_json_lines(gt_json)
        gt_by_raw = {str(item["raw_file"]): item for item in gt_records_all}
        gt_records = [gt_by_raw[str(pred["raw_file"])] for pred in pred_records if str(pred["raw_file"]) in gt_by_raw]
        if len(gt_records) != len(pred_records):
            missing = sorted(str(pred["raw_file"]) for pred in pred_records if str(pred["raw_file"]) not in gt_by_raw)
            return {
                "available": False,
                "reason": "official_gt_missing_raw_files",
                "gt_json": str(gt_json),
                "missing_raw_files": missing[:20],
                "missing_count": int(len(missing)),
            }
        pred_records = [
            {**pred, "h_samples": list(gt["h_samples"])}
            for pred, gt in zip(pred_records, gt_records)
        ]
        result, _ = TuSimpleOfficialLaneEval.bench_records(pred_records, gt_records, strict_length=False)
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__, "message": str(exc)}
    metrics = result.as_dict()
    official_acc = metrics.pop("Accuracy")
    official_fp = metrics.pop("FP")
    official_fn = metrics.pop("FN")
    count_metrics = count_metrics_from_pairs(
        [
            (sum(1 for lane in gt.get("lanes", []) if any(float(x) >= 0.0 for x in lane)), len(pred.get("lanes", [])))
            for pred, gt in zip(pred_records, gt_records)
        ]
    )
    return {
        "available": True,
        "metric_contract": "tusimple_official_reporting_only",
        "gt_json": str(gt_json),
        "official_acc": official_acc,
        "official_FP": official_fp,
        "official_FN": official_fn,
        "official_score": round(
            official_metric_score(
                official_acc,
                official_fp,
                official_fn,
                fp_weight=float(args.score_fp_weight),
                fn_weight=float(args.score_fn_weight),
            ),
            6,
        ),
        **count_metrics,
        **metrics,
    }


def acceptance_metric_aliases(official_metrics: dict, diagnostic_count_metrics: dict) -> dict:
    """Expose the requested acceptance names while preserving metric provenance."""
    out = {
        "official_acc": None,
        "official_FP": None,
        "official_FN": None,
        "count_acc": diagnostic_count_metrics.get("diagnostic_decode_count_acc"),
        "count_acc_4": diagnostic_count_metrics.get("diagnostic_decode_count_acc_4"),
        "count_acc_5": diagnostic_count_metrics.get("diagnostic_decode_count_acc_5"),
    }
    if official_metrics.get("available"):
        for key in ("official_acc", "official_FP", "official_FN", "count_acc", "count_acc_4", "count_acc_5"):
            if key in official_metrics:
                out[key] = official_metrics[key]
    return out


def load_official_h_samples_by_raw(args: argparse.Namespace, split: str) -> dict[str, list] | None:
    """Best-effort map for converting decoded lanes with the exact official h_samples."""
    try:
        archive_root = find_tusimple_archive_root(args.archive_root)
        gt_json = resolve_tusimple_gt_json(archive_root, split=split, gt_json=args.official_gt_json)
        return {str(item["raw_file"]): list(item["h_samples"]) for item in read_tusimple_json_lines(gt_json)}
    except Exception:
        return None


def diagnose_image(
    model: torch.nn.Module,
    criterion: GCSLoss,
    image_path: Path,
    labels_dir: Path,
    *,
    split: str,
    dataset_root: Path,
    imgsz: tuple[int, int],
    device: torch.device,
    half: bool,
    args: argparse.Namespace,
    official_h_samples_by_raw: dict[str, list] | None = None,
) -> tuple[dict, list[dict], list[dict], dict | None]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    assert_gcs_shape(img.shape[:2], imgsz, name="diagnostic image", context=f"{image_path}")
    label_path = label_path_for_image(image_path, labels_dir)
    gt_lanes, gt_valid = load_gcs_label(label_path)
    meta = load_label_meta(label_path, img.shape[:2])

    tensor = preprocess_image(img, imgsz=imgsz, device=device, half=half)
    preds = model(tensor)
    if "pred_points" not in preds or "pred_logits" not in preds or "pred_valid_logits" not in preds:
        raise KeyError("count-contract diagnostic requires query-mode pred_points, pred_logits, and pred_valid_logits.")
    pred_points_t = preds["pred_points"][0].detach().float()
    pred_logits_t = preds["pred_logits"][0].detach().float()
    pred_valid_t = preds["pred_valid_logits"][0].detach().float()
    if pred_logits_t.ndim == 2 and pred_logits_t.shape[-1] == 1:
        pred_logits_t = pred_logits_t.squeeze(-1)
    if tuple(pred_points_t.shape) != (12, 56, 2) or tuple(pred_valid_t.shape) != (12, 56):
        raise ValueError(
            f"{image_path}: expected active Q12/K56 query outputs, got "
            f"pred_points={tuple(pred_points_t.shape)} pred_valid={tuple(pred_valid_t.shape)}."
        )

    stage = query_stage_trace(pred_points_t, pred_logits_t, pred_valid_t, img.shape[:2], args)
    decoded = decode_gcs_predictions(
        pred_points_t,
        pred_logits_t,
        pred_valid_logits=pred_valid_t,
        image_shape=img.shape[:2],
        score_thr=args.conf,
        point_valid_thr=args.point_valid_thr,
        min_points=args.min_points,
        max_det=args.max_det,
        nms_dist_px=args.nms_dist_px,
        valid_before_maxdet=bool(args.valid_before_maxdet),
    )
    decoded_queries = {int(lane.get("query", -1)) for lane in decoded}
    if decoded_queries != stage["final"]:
        raise AssertionError(
            f"{image_path}: traced final queries {sorted(stage['final'])} "
            f"do not match decode_gcs_predictions queries {sorted(decoded_queries)}"
        )

    gt_count = int(gt_lanes.shape[0])
    pred_count = int(len(decoded))
    bucket = bucket_name(gt_count, pred_count)
    indices, contract_masks = build_training_contract_masks(
        criterion,
        pred_points_t.to(device=device),
        pred_logits_t.to(device=device),
        pred_valid_t.to(device=device),
        gt_lanes,
        gt_valid,
    )
    src_idx, tgt_idx = indices[0]
    hungarian_query_by_gt = {int(t.item()): int(s.item()) for s, t in zip(src_idx.detach().cpu(), tgt_idx.detach().cpu())}
    hungarian_matched_queries = set(int(x.item()) for x in src_idx.detach().cpu().reshape(-1))
    scores = pred_logits_t.sigmoid().cpu().numpy().astype(np.float32)
    raw_pred_points, raw_pred_valid_scores = anchor_query_arrays(pred_points_t, pred_valid_t)
    width = int(img.shape[1])
    ranks = side_ranks_by_lower_x(gt_lanes, gt_valid)
    side_rank_max = max(ranks.values(), default=-1)
    gt_visible_counts = [int((gt_valid[i] > 0.5).sum()) for i in range(gt_count)]
    raw_rescue_candidates: list[tuple[int, int, float]] = []
    raw_rescue_conflict_count = 0
    raw_geometry_by_q: dict[int, list[tuple[float, int, bool]]] = defaultdict(list)
    gt_records = []
    stage_flags_for_summary = []
    shortside_selected_by_abs_count = 0
    shortside_selected_by_median_count = 0
    shortside_selected_total_count = 0

    if bucket is not None:
        for gt_idx in range(gt_count):
            q, raw_dist = best_raw_query_for_gt(raw_pred_points, gt_lanes, gt_valid, gt_idx, width)
            rank = ranks.get(gt_idx)
            is_side = rank in {0, side_rank_max}
            gt_visible_points = int((gt_valid[gt_idx] > 0.5).sum())
            shortside_decision = shortside_contract_decision(
                gt_count, bool(is_side), gt_visible_points, gt_visible_counts, args
            )
            shortside_eligible = bool(shortside_decision.eligible)
            if shortside_eligible:
                shortside_selected_total_count += 1
                if bool(shortside_decision.selected_by_abs):
                    shortside_selected_by_abs_count += 1
                if bool(shortside_decision.selected_by_median):
                    shortside_selected_by_median_count += 1
            raw_thr = float(args.side_raw_match_dist_px if is_side else args.raw_match_dist_px)
            raw_geometry_ok = bool(q >= 0 and np.isfinite(raw_dist) and raw_dist <= raw_thr)
            shortside_rawmatch_ok = bool(shortside_eligible and raw_geometry_ok)
            if raw_geometry_ok:
                raw_geometry_by_q[int(q)].append((float(raw_dist), int(gt_idx), bool(shortside_eligible)))
            if shortside_rawmatch_ok:
                raw_rescue_candidates.append((int(q), int(gt_idx), float(raw_dist)))
            final_gt_matched = final_match_for_gt(
                decoded,
                gt_lanes[gt_idx],
                gt_valid[gt_idx],
                width,
                args,
                match_dist_px=raw_thr,
            )
            final_query_kept = bool(q in stage["final"]) if q >= 0 else False
            dropped_stage, drop_reason = classify_drop_stage_for_gt(
                raw_geometry_ok, q, stage, final_query_kept, final_gt_matched, args
            )
            flags = record_stage_flags(raw_geometry_ok, q, stage, final_query_kept, final_gt_matched, args)
            stage_flags_for_summary.append(flags)
            hungarian_q = hungarian_query_by_gt.get(int(gt_idx), -1)
            hungarian_raw_dist = (
                mean_dx_to_gt_px(raw_pred_points[hungarian_q], gt_lanes[gt_idx], gt_valid[gt_idx], width)
                if hungarian_q >= 0
                else float("inf")
            )
            best_raw_query_idx = int(q) if raw_geometry_ok else -1
            best_raw_dist = float(raw_dist) if raw_geometry_ok else float("inf")
            selected_shortside_q, selected_shortside_reason = selected_shortside_for_gt(gt_idx, contract_masks)
            gt_records.append(
                {
                    "bucket": bucket,
                    "split": split,
                    "image": str(image_path),
                    "label": str(label_path),
                    "raw_file": meta.get("raw_file", ""),
                    "gt_lanes": gt_count,
                    "pred_lanes": pred_count,
                    "gt_lane_idx": int(gt_idx),
                    "side_rank": None if rank is None else int(rank),
                    "is_leftmost_or_rightmost": bool(is_side),
                    "gt_valid_points": gt_visible_points,
                    "image_gt_visible_median": clean_float(shortside_decision.image_visible_median, 4),
                    "shortside_selected_by_abs": bool(shortside_decision.selected_by_abs),
                    "shortside_selected_by_median": bool(shortside_decision.selected_by_median),
                    "shortside_is_reliable": bool(shortside_decision.is_reliable),
                    "shortside_is_ultra_short": bool(shortside_decision.is_ultra_short),
                    "shortside_skipped_visible_lt2": bool(shortside_decision.skipped_visible_lt2),
                    "raw_geometry_ok": bool(raw_geometry_ok),
                    "shortside_contract_eligible": bool(shortside_eligible),
                    "shortside_rawmatch_ok": bool(shortside_rawmatch_ok),
                    "gt_visible_span": visible_span(gt_valid[gt_idx]),
                    "hungarian_query_idx": int(hungarian_q),
                    "hungarian_raw_dist_px": clean_float(hungarian_raw_dist, 4),
                    "hungarian_query_final_kept": bool(hungarian_q in decoded_queries) if hungarian_q >= 0 else False,
                    "best_raw_query_idx": int(best_raw_query_idx),
                    "best_raw_dist_px": clean_float(best_raw_dist, 4),
                    "best_raw_is_hungarian": bool(best_raw_query_idx >= 0 and best_raw_query_idx == int(hungarian_q)),
                    "best_raw_is_unmatched": bool(best_raw_query_idx >= 0 and best_raw_query_idx not in hungarian_matched_queries),
                    "selected_shortside_query_idx": int(selected_shortside_q),
                    "selected_shortside_reason": selected_shortside_reason,
                    "matched_query_idx": int(best_raw_query_idx),
                    "matched_query_idx_deprecated": True,
                    "matched_query_semantics": "deprecated_alias_of_best_raw_query_idx_not_hungarian",
                    "raw_match_dist": clean_float(best_raw_dist, 4),
                    "exist_score": clean_float(scores[q], 8) if q >= 0 else None,
                    "quality_score": None,
                    "lane_score": clean_float(scores[q], 8) if q >= 0 else None,
                    "pred_valid_points": int(stage["valid_count"].get(q, 0)) if q >= 0 else 0,
                    "final_query_kept": bool(final_query_kept),
                    "final_gt_matched": bool(final_gt_matched),
                    "dropped_stage": dropped_stage,
                    "drop_reason": drop_reason,
                }
            )

    candidates_by_q: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for q, gt_idx, raw_dist in raw_rescue_candidates:
        candidates_by_q[int(q)].append((float(raw_dist), int(gt_idx)))
    for q, candidates in candidates_by_q.items():
        candidates.sort(key=lambda item: item[0])
        best_dist, _ = candidates[0]
        closer_non_shortside = any(
            (not shortside_eligible) and other_dist < best_dist
            for other_dist, _, shortside_eligible in raw_geometry_by_q.get(int(q), [])
        )
        if closer_non_shortside:
            raw_rescue_conflict_count += 1
        if len(candidates) > 1:
            raw_rescue_conflict_count += len(candidates) - 1

    unmatched_records = []
    if bucket is not None:
        for q in range(raw_pred_points.shape[0]):
            if q in hungarian_matched_queries:
                continue
            min_dist, nearest_gt, min_bottom, side_min = query_gt_distance(
                raw_pred_points[q],
                raw_pred_valid_scores[q],
                gt_lanes,
                gt_valid,
                width,
                args.point_valid_thr,
            )
            pred_valid_points = int(stage["valid_count"].get(q, 0))
            roles = query_role_flags(q, contract_masks)
            unmatched_records.append(
                {
                    "bucket": bucket,
                    "split": split,
                    "image": str(image_path),
                    "raw_file": meta.get("raw_file", ""),
                    "gt_lanes": gt_count,
                    "pred_lanes": pred_count,
                    "query_idx": int(q),
                    "lane_score": clean_float(scores[q], 8),
                    "pred_valid_points": pred_valid_points,
                    "min_dist_to_gt": clean_float(min_dist, 4),
                    "nearest_gt_idx": int(nearest_gt),
                    "min_bottom_dx_to_gt": clean_float(min_bottom, 4),
                    "min_side_dist_to_gt": clean_float(side_min, 4),
                    "query_role_semantics": "training_hungarian_count_contract_masks",
                    **roles,
                    "final_kept": bool(q in decoded_queries),
                }
            )

    tusimple_prediction = None
    raw_file = meta.get("raw_file", "")
    if raw_file:
        h_samples = (
            list(official_h_samples_by_raw[raw_file])
            if official_h_samples_by_raw is not None and raw_file in official_h_samples_by_raw
            else list(range(160, 720, 10))
        )
        tusimple_prediction = {
            "lanes": gcs_lanes_to_tusimple_lanes(
                decoded,
                h_samples,
                image_shape=TUSIMPLE_ORIGINAL_SHAPE,
            ),
            "h_samples": h_samples,
            "raw_file": raw_file,
            "run_time": float(args.official_runtime_ms),
        }
    rank_pos_total_count = contract_scalar_count(contract_masks, "rank_pos_total_count")
    rank_neg_total_count = contract_mask_count(contract_masks, "rank_neg")
    rank_active_for_image = gt_count >= int(getattr(criterion, "rank_gt_min_lanes", args.rank_gt_min_lanes))
    rank_noop_because_no_pos = int(bool(rank_active_for_image and rank_pos_total_count == 0))
    rank_noop_because_no_neg = int(bool(rank_active_for_image and rank_neg_total_count == 0))
    rank_loss_noop_images = int(bool(rank_noop_because_no_pos or rank_noop_because_no_neg))
    image_summary = {
        "split": split,
        "image": str(image_path),
        "raw_file": meta.get("raw_file", ""),
        "gt_lanes": gt_count,
        "pred_lanes": pred_count,
        "bucket": bucket,
        "raw_geometry_rescue_proxy_candidate_count": int(len(raw_rescue_candidates)),
        "raw_geometry_rescue_proxy_conflict_count": int(raw_rescue_conflict_count),
        "shortside_hungarian_rawmatch_candidate_count": contract_scalar_count(
            contract_masks, "shortside_hungarian_rawmatch_candidate_count"
        ),
        "shortside_unmatched_raw_rescue_candidate_count": contract_scalar_count(
            contract_masks, "shortside_unmatched_raw_rescue_candidate_count"
        ),
        "shortside_rawmatch_candidate_total_count": contract_scalar_count(
            contract_masks, "shortside_rawmatch_candidate_total_count"
        ),
        "raw_rescue_candidate_count": int(float(contract_masks["raw_rescue_candidate_count"].detach().cpu().item())),
        "raw_rescue_final_count": int(float(contract_masks["raw_rescue_final_count"].detach().cpu().item())),
        "raw_rescue_conflict_count": int(float(contract_masks["raw_rescue_conflict_count"].detach().cpu().item())),
        "shortside_selected_hungarian_rawmatch": int(
            float(contract_masks["shortside_selected_hungarian_rawmatch_count"].detach().cpu().item())
        ),
        "shortside_selected_unmatched_rescue": int(
            float(contract_masks["shortside_selected_unmatched_rescue_count"].detach().cpu().item())
        ),
        "shortside_rescue_conflict": int(float(contract_masks["shortside_rescue_conflict_count"].detach().cpu().item())),
        "shortside_missing_no_rawmatch": int(
            float(contract_masks["shortside_missing_no_rawmatch_count"].detach().cpu().item())
        ),
        "shortside_nearest_was_duplicate_but_hungarian_boosted": int(
            float(contract_masks["shortside_nearest_was_duplicate_but_hungarian_boosted_count"].detach().cpu().item())
        ),
        "shortside_selected_by_abs_count": int(shortside_selected_by_abs_count),
        "shortside_selected_by_median_count": int(shortside_selected_by_median_count),
        "shortside_selected_total_count": int(shortside_selected_total_count),
        "shortside_reliable_count": contract_scalar_count(contract_masks, "shortside_reliable_count"),
        "shortside_reliable_selected_count": contract_scalar_count(contract_masks, "shortside_reliable_selected_count"),
        "shortside_ultra_seen_count": contract_scalar_count(contract_masks, "shortside_ultra_seen_count"),
        "shortside_ultra_short_count": contract_scalar_count(contract_masks, "shortside_ultra_short_count"),
        "shortside_ultra_enabled_count": contract_scalar_count(contract_masks, "shortside_ultra_enabled_count"),
        "shortside_ultra_score_floor_count": contract_scalar_count(
            contract_masks, "shortside_ultra_score_floor_count"
        ),
        "shortside_ultra_valid_count": contract_scalar_count(contract_masks, "shortside_ultra_valid_count"),
        "shortside_visible_lt2_skipped_count": contract_scalar_count(
            contract_masks, "shortside_visible_lt2_skipped_count"
        ),
        "shortside_skipped_visible_lt2_count": contract_scalar_count(
            contract_masks, "shortside_skipped_visible_lt2_count"
        ),
        "shortside_ultra_in_gt4_4to3_count": contract_scalar_count(
            contract_masks, "shortside_ultra_in_gt4_4to3_count"
        ),
        "shortside_ultra_in_gt5_5to4_count": contract_scalar_count(
            contract_masks, "shortside_ultra_in_gt5_5to4_count"
        ),
        "shortside_visible_2_5_in_gt4_4to3": contract_scalar_count(
            contract_masks, "shortside_visible_2_5_in_gt4_4to3_count"
        ),
        "shortside_visible_2_5_in_gt5_5to4": contract_scalar_count(
            contract_masks, "shortside_visible_2_5_in_gt5_5to4_count"
        ),
        "rank_pos_scope": str(getattr(criterion, "rank_pos_scope", args.rank_pos_scope)),
        "rank_pos_total": rank_pos_total_count,
        "rank_pos_hungarian_all": contract_scalar_count(contract_masks, "rank_pos_hungarian_all_count"),
        "rank_pos_shortside_matched": contract_scalar_count(contract_masks, "rank_pos_shortside_matched_count"),
        "rank_pos_shortside_rescue": contract_scalar_count(contract_masks, "rank_pos_shortside_rescue_count"),
        "rank_pos_shortside_reliable": contract_scalar_count(contract_masks, "rank_pos_shortside_reliable_count"),
        "rank_pos_shortside_ultra": contract_scalar_count(contract_masks, "rank_pos_shortside_ultra_count"),
        "rank_pos_gt4gt5_matched": contract_scalar_count(contract_masks, "rank_pos_gt4gt5_matched_count"),
        "rank_pos_all_matched": contract_scalar_count(contract_masks, "rank_pos_all_matched_count"),
        "rank_neg_duplicate_like": contract_scalar_count(contract_masks, "rank_neg_duplicate_like_count"),
        "rank_neg_clear_far": contract_scalar_count(contract_masks, "rank_neg_clear_far_count"),
        "rank_neg_near_ignored": contract_scalar_count(contract_masks, "rank_neg_near_ignored_count"),
        "rank_neg_side_duplicate_like_count": contract_scalar_count(
            contract_masks, "rank_neg_side_duplicate_like_count"
        ),
        "rank_neg_normal_duplicate_like_count": contract_scalar_count(
            contract_masks, "rank_neg_normal_duplicate_like_count"
        ),
        "rank_side_ambiguous_ignored_count": contract_scalar_count(
            contract_masks, "rank_side_ambiguous_ignored_count"
        ),
        "rank_side_duplicate_rejected_better_than_true_count": contract_scalar_count(
            contract_masks, "rank_side_duplicate_rejected_better_than_true_count"
        ),
        "rank_loss_noop_images": rank_loss_noop_images,
        "rank_noop_because_no_pos": rank_noop_because_no_pos,
        "rank_noop_because_no_neg": rank_noop_because_no_neg,
        "clear_far_boundary_count": contract_scalar_count(contract_masks, "clear_far_boundary_count"),
        "training_hungarian_positive_count": int(len(hungarian_matched_queries)),
        "label_contract": meta.get("label_contract", "validated_k56_fixed_y"),
        "stage_flags": stage_flags_for_summary,
    }
    return image_summary, gt_records, unmatched_records, tusimple_prediction


def summarize_stage(records: list[dict]) -> dict:
    totals = {
        name: 0
        for name in (
            "raw_match",
            "after_point_valid",
            "after_min_points",
            "after_conf",
            "after_maxdet",
            "after_nms",
            "final_query",
            "final_gt_matched",
            "final_decode",
        )
    }
    n = 0
    for record in records:
        for flags in record.get("stage_flags", []):
            n += 1
            for key in totals:
                totals[key] += int(bool(flags.get(key)))
    out = {"gt_lane_records": int(n)}
    for key, value in totals.items():
        out[f"{key}_count"] = int(value)
        out[f"{key}_recall"] = None if n == 0 else round(float(value) / float(n), 6)
    return out


def build_summary(
    image_summaries: list[dict],
    gt_records: list[dict],
    unmatched_records: list[dict],
    tusimple_predictions: list[dict],
    args: argparse.Namespace,
) -> dict:
    bucket_images = defaultdict(list)
    for item in image_summaries:
        if item.get("bucket") in TARGET_BUCKETS:
            bucket_images[item["bucket"]].append(item)

    count_confusion = Counter()
    count_pairs: list[tuple[int, int]] = []
    for item in image_summaries:
        count_confusion[f"{item['gt_lanes']}->{item['pred_lanes']}"] += 1
        count_pairs.append((int(item["gt_lanes"]), int(item["pred_lanes"])))
    diagnostic_count_metrics = count_metrics_from_pairs(count_pairs)
    diagnostic_count_metrics = {f"diagnostic_decode_{k}": v for k, v in diagnostic_count_metrics.items()}
    official_metrics = official_metrics_for_summaries(
        [
            {**item, "tusimple_prediction": pred}
            for item, pred in zip(image_summaries, tusimple_predictions)
            if pred is not None
        ],
        args,
    )
    acceptance_metrics = acceptance_metric_aliases(official_metrics, diagnostic_count_metrics)

    per_bucket = {}
    for bucket in TARGET_BUCKETS:
        imgs = bucket_images.get(bucket, [])
        gt_bucket = [r for r in gt_records if r["bucket"] == bucket]
        unmatched_bucket = [r for r in unmatched_records if r["bucket"] == bucket]
        shortside_hungarian_rawmatch_candidate_count = int(
            sum(i.get("shortside_hungarian_rawmatch_candidate_count", 0) for i in imgs)
        )
        shortside_unmatched_raw_rescue_candidate_count = int(
            sum(i.get("shortside_unmatched_raw_rescue_candidate_count", 0) for i in imgs)
        )
        shortside_rawmatch_candidate_total_count = int(
            sum(i.get("shortside_rawmatch_candidate_total_count", 0) for i in imgs)
        )
        raw_rescue_candidate_count = int(sum(i.get("raw_rescue_candidate_count", 0) for i in imgs))
        raw_rescue_final_count = int(sum(i.get("raw_rescue_final_count", 0) for i in imgs))
        raw_rescue_conflict_count = int(sum(i.get("raw_rescue_conflict_count", 0) for i in imgs))
        shortside_selected_hungarian_rawmatch = int(sum(i.get("shortside_selected_hungarian_rawmatch", 0) for i in imgs))
        shortside_selected_unmatched_rescue = int(sum(i.get("shortside_selected_unmatched_rescue", 0) for i in imgs))
        shortside_rescue_conflict = int(sum(i.get("shortside_rescue_conflict", 0) for i in imgs))
        shortside_missing_no_rawmatch = int(sum(i.get("shortside_missing_no_rawmatch", 0) for i in imgs))
        shortside_nearest_was_duplicate_but_hungarian_boosted = int(
            sum(i.get("shortside_nearest_was_duplicate_but_hungarian_boosted", 0) for i in imgs)
        )
        shortside_selected_by_abs_count = int(sum(i.get("shortside_selected_by_abs_count", 0) for i in imgs))
        shortside_selected_by_median_count = int(sum(i.get("shortside_selected_by_median_count", 0) for i in imgs))
        shortside_selected_total_count = int(sum(i.get("shortside_selected_total_count", 0) for i in imgs))
        shortside_reliable_count = int(sum(i.get("shortside_reliable_count", 0) for i in imgs))
        shortside_reliable_selected_count = int(sum(i.get("shortside_reliable_selected_count", 0) for i in imgs))
        shortside_ultra_seen_count = int(sum(i.get("shortside_ultra_seen_count", 0) for i in imgs))
        shortside_ultra_short_count = int(sum(i.get("shortside_ultra_short_count", 0) for i in imgs))
        shortside_ultra_enabled_count = int(sum(i.get("shortside_ultra_enabled_count", 0) for i in imgs))
        shortside_ultra_score_floor_count = int(sum(i.get("shortside_ultra_score_floor_count", 0) for i in imgs))
        shortside_ultra_valid_count = int(sum(i.get("shortside_ultra_valid_count", 0) for i in imgs))
        shortside_visible_lt2_skipped_count = int(sum(i.get("shortside_visible_lt2_skipped_count", 0) for i in imgs))
        shortside_skipped_visible_lt2_count = int(sum(i.get("shortside_skipped_visible_lt2_count", 0) for i in imgs))
        shortside_ultra_in_gt4_4to3_count = int(sum(i.get("shortside_ultra_in_gt4_4to3_count", 0) for i in imgs))
        shortside_ultra_in_gt5_5to4_count = int(sum(i.get("shortside_ultra_in_gt5_5to4_count", 0) for i in imgs))
        shortside_visible_2_5_in_gt4_4to3 = int(sum(i.get("shortside_visible_2_5_in_gt4_4to3", 0) for i in imgs))
        shortside_visible_2_5_in_gt5_5to4 = int(sum(i.get("shortside_visible_2_5_in_gt5_5to4", 0) for i in imgs))
        rank_pos_total = int(sum(i.get("rank_pos_total", 0) for i in imgs))
        rank_pos_hungarian_all = int(sum(i.get("rank_pos_hungarian_all", 0) for i in imgs))
        rank_pos_shortside_matched = int(sum(i.get("rank_pos_shortside_matched", 0) for i in imgs))
        rank_pos_shortside_rescue = int(sum(i.get("rank_pos_shortside_rescue", 0) for i in imgs))
        rank_pos_shortside_reliable = int(sum(i.get("rank_pos_shortside_reliable", 0) for i in imgs))
        rank_pos_shortside_ultra = int(sum(i.get("rank_pos_shortside_ultra", 0) for i in imgs))
        rank_pos_gt4gt5_matched = int(sum(i.get("rank_pos_gt4gt5_matched", 0) for i in imgs))
        rank_pos_all_matched = int(sum(i.get("rank_pos_all_matched", 0) for i in imgs))
        rank_neg_duplicate_like = int(sum(i.get("rank_neg_duplicate_like", 0) for i in imgs))
        rank_neg_clear_far = int(sum(i.get("rank_neg_clear_far", 0) for i in imgs))
        rank_neg_near_ignored = int(sum(i.get("rank_neg_near_ignored", 0) for i in imgs))
        rank_neg_side_duplicate_like_count = int(sum(i.get("rank_neg_side_duplicate_like_count", 0) for i in imgs))
        rank_neg_normal_duplicate_like_count = int(sum(i.get("rank_neg_normal_duplicate_like_count", 0) for i in imgs))
        rank_side_ambiguous_ignored_count = int(sum(i.get("rank_side_ambiguous_ignored_count", 0) for i in imgs))
        rank_side_duplicate_rejected_better_than_true_count = int(
            sum(i.get("rank_side_duplicate_rejected_better_than_true_count", 0) for i in imgs)
        )
        rank_loss_noop_images = int(sum(i.get("rank_loss_noop_images", 0) for i in imgs))
        rank_noop_because_no_pos = int(sum(i.get("rank_noop_because_no_pos", 0) for i in imgs))
        rank_noop_because_no_neg = int(sum(i.get("rank_noop_because_no_neg", 0) for i in imgs))
        near_gt_ignore_count = int(sum(1 for r in unmatched_bucket if r["is_near_gt_corridor"]))
        duplicate_like_rank_neg_count = int(sum(1 for r in unmatched_bucket if r.get("is_duplicate_like_rank_neg")))
        clear_far_rank_neg_count = int(sum(1 for r in unmatched_bucket if r["is_clear_far_spurious"]))
        clear_far_boundary_count = int(sum(i.get("clear_far_boundary_count", 0) for i in imgs))
        per_bucket[bucket] = {
            "images": int(len(imgs)),
            "gt_lane_records": int(len(gt_bucket)),
            "unmatched_query_records": int(len(unmatched_bucket)),
            "shortside_hungarian_rawmatch_candidate_count": shortside_hungarian_rawmatch_candidate_count,
            "shortside_unmatched_raw_rescue_candidate_count": shortside_unmatched_raw_rescue_candidate_count,
            "shortside_rawmatch_candidate_total_count": shortside_rawmatch_candidate_total_count,
            "raw_rescue_candidate_count": raw_rescue_candidate_count,
            "raw_rescue_final_count": raw_rescue_final_count,
            "raw_rescue_conflict_count": raw_rescue_conflict_count,
            "shortside_selected_hungarian_rawmatch": shortside_selected_hungarian_rawmatch,
            "shortside_selected_unmatched_rescue": shortside_selected_unmatched_rescue,
            "shortside_rescue_conflict": shortside_rescue_conflict,
            "shortside_missing_no_rawmatch": shortside_missing_no_rawmatch,
            "shortside_nearest_was_duplicate_but_hungarian_boosted": shortside_nearest_was_duplicate_but_hungarian_boosted,
            "shortside_selected_by_abs_count": shortside_selected_by_abs_count,
            "shortside_selected_by_median_count": shortside_selected_by_median_count,
            "shortside_selected_total_count": shortside_selected_total_count,
            "shortside_reliable_count": shortside_reliable_count,
            "shortside_reliable_selected_count": shortside_reliable_selected_count,
            "shortside_ultra_seen_count": shortside_ultra_seen_count,
            "shortside_ultra_short_count": shortside_ultra_short_count,
            "shortside_ultra_enabled_count": shortside_ultra_enabled_count,
            "shortside_ultra_score_floor_count": shortside_ultra_score_floor_count,
            "shortside_ultra_valid_count": shortside_ultra_valid_count,
            "shortside_visible_lt2_skipped_count": shortside_visible_lt2_skipped_count,
            "shortside_skipped_visible_lt2_count": shortside_skipped_visible_lt2_count,
            "shortside_ultra_in_gt4_4to3_count": shortside_ultra_in_gt4_4to3_count,
            "shortside_ultra_in_gt5_5to4_count": shortside_ultra_in_gt5_5to4_count,
            "shortside_visible_2_5_in_gt4_4to3": shortside_visible_2_5_in_gt4_4to3,
            "shortside_visible_2_5_in_gt5_5to4": shortside_visible_2_5_in_gt5_5to4,
            "rank_pos_scope": str(args.rank_pos_scope),
            "rank_pos_total": rank_pos_total,
            "rank_pos_hungarian_all": rank_pos_hungarian_all,
            "rank_pos_shortside_matched": rank_pos_shortside_matched,
            "rank_pos_shortside_rescue": rank_pos_shortside_rescue,
            "rank_pos_shortside_reliable": rank_pos_shortside_reliable,
            "rank_pos_shortside_ultra": rank_pos_shortside_ultra,
            "rank_pos_gt4gt5_matched": rank_pos_gt4gt5_matched,
            "rank_pos_all_matched": rank_pos_all_matched,
            "rank_neg_duplicate_like": rank_neg_duplicate_like,
            "rank_neg_clear_far": rank_neg_clear_far,
            "rank_neg_near_ignored": rank_neg_near_ignored,
            "rank_neg_side_duplicate_like_count": rank_neg_side_duplicate_like_count,
            "rank_neg_normal_duplicate_like_count": rank_neg_normal_duplicate_like_count,
            "rank_neg_clear_far_count": rank_neg_clear_far,
            "rank_side_ambiguous_ignored_count": rank_side_ambiguous_ignored_count,
            "rank_side_duplicate_rejected_better_than_true_count": rank_side_duplicate_rejected_better_than_true_count,
            "rank_loss_noop_images": rank_loss_noop_images,
            "rank_noop_because_no_pos": rank_noop_because_no_pos,
            "rank_noop_because_no_neg": rank_noop_because_no_neg,
            "raw_geometry_rescue_proxy_candidate_count": int(
                sum(i.get("raw_geometry_rescue_proxy_candidate_count", 0) for i in imgs)
            ),
            "raw_geometry_rescue_proxy_conflict_count": int(
                sum(i.get("raw_geometry_rescue_proxy_conflict_count", 0) for i in imgs)
            ),
            "near_gt_ignore_count": near_gt_ignore_count,
            "near_gt_ignored_count": near_gt_ignore_count,
            "duplicate_like_rank_neg_count": duplicate_like_rank_neg_count,
            "clear_far_rank_neg_count": clear_far_rank_neg_count,
            "clear_far_boundary_count": clear_far_boundary_count,
            "stage": summarize_stage(imgs),
            "drop_reason_hist": dict(sorted(Counter(r["drop_reason"] for r in gt_bucket).items())),
            "unmatched_query_flags": {
                "near_gt_corridor": near_gt_ignore_count,
                "ambiguous_side_region": int(sum(1 for r in unmatched_bucket if r["is_ambiguous_side_region"])),
                "clear_far_spurious": clear_far_rank_neg_count,
                "duplicate_like_rank_neg": duplicate_like_rank_neg_count,
                "normal_duplicate_like_rank_neg": int(
                    sum(1 for r in unmatched_bucket if r.get("is_normal_duplicate_like_rank_neg"))
                ),
                "side_duplicate_like_rank_neg": int(
                    sum(1 for r in unmatched_bucket if r.get("is_side_duplicate_like_rank_neg"))
                ),
                "rank_side_ambiguous_ignored": int(
                    sum(1 for r in unmatched_bucket if r.get("is_rank_side_ambiguous_ignored"))
                ),
                "final_kept": int(sum(1 for r in unmatched_bucket if r["final_kept"])),
            },
        }

    has_test = "test" in set(args.splits)
    shortside_hungarian_rawmatch_candidate_total = int(
        sum(i.get("shortside_hungarian_rawmatch_candidate_count", 0) for i in image_summaries)
    )
    shortside_unmatched_raw_rescue_candidate_total = int(
        sum(i.get("shortside_unmatched_raw_rescue_candidate_count", 0) for i in image_summaries)
    )
    shortside_rawmatch_candidate_total = int(
        sum(i.get("shortside_rawmatch_candidate_total_count", 0) for i in image_summaries)
    )
    raw_rescue_candidate_total = int(sum(i.get("raw_rescue_candidate_count", 0) for i in image_summaries))
    raw_rescue_final_total = int(sum(i.get("raw_rescue_final_count", 0) for i in image_summaries))
    raw_rescue_conflict_total = int(sum(i.get("raw_rescue_conflict_count", 0) for i in image_summaries))
    shortside_selected_hungarian_rawmatch_total = int(
        sum(i.get("shortside_selected_hungarian_rawmatch", 0) for i in image_summaries)
    )
    shortside_selected_unmatched_rescue_total = int(
        sum(i.get("shortside_selected_unmatched_rescue", 0) for i in image_summaries)
    )
    shortside_rescue_conflict_total = int(sum(i.get("shortside_rescue_conflict", 0) for i in image_summaries))
    shortside_missing_no_rawmatch_total = int(sum(i.get("shortside_missing_no_rawmatch", 0) for i in image_summaries))
    shortside_nearest_was_duplicate_but_hungarian_boosted_total = int(
        sum(i.get("shortside_nearest_was_duplicate_but_hungarian_boosted", 0) for i in image_summaries)
    )
    shortside_selected_by_abs_total = int(sum(i.get("shortside_selected_by_abs_count", 0) for i in image_summaries))
    shortside_selected_by_median_total = int(sum(i.get("shortside_selected_by_median_count", 0) for i in image_summaries))
    shortside_selected_total = int(sum(i.get("shortside_selected_total_count", 0) for i in image_summaries))
    shortside_reliable_total = int(sum(i.get("shortside_reliable_count", 0) for i in image_summaries))
    shortside_reliable_selected_total = int(sum(i.get("shortside_reliable_selected_count", 0) for i in image_summaries))
    shortside_ultra_seen_total = int(sum(i.get("shortside_ultra_seen_count", 0) for i in image_summaries))
    shortside_ultra_short_total = int(sum(i.get("shortside_ultra_short_count", 0) for i in image_summaries))
    shortside_ultra_enabled_total = int(sum(i.get("shortside_ultra_enabled_count", 0) for i in image_summaries))
    shortside_ultra_score_floor_total = int(sum(i.get("shortside_ultra_score_floor_count", 0) for i in image_summaries))
    shortside_ultra_valid_total = int(sum(i.get("shortside_ultra_valid_count", 0) for i in image_summaries))
    shortside_visible_lt2_skipped_total = int(sum(i.get("shortside_visible_lt2_skipped_count", 0) for i in image_summaries))
    shortside_skipped_visible_lt2_total = int(sum(i.get("shortside_skipped_visible_lt2_count", 0) for i in image_summaries))
    shortside_ultra_in_gt4_4to3_total = int(sum(i.get("shortside_ultra_in_gt4_4to3_count", 0) for i in image_summaries))
    shortside_ultra_in_gt5_5to4_total = int(sum(i.get("shortside_ultra_in_gt5_5to4_count", 0) for i in image_summaries))
    shortside_visible_2_5_in_gt4_4to3_total = int(
        sum(i.get("shortside_visible_2_5_in_gt4_4to3", 0) for i in image_summaries)
    )
    shortside_visible_2_5_in_gt5_5to4_total = int(
        sum(i.get("shortside_visible_2_5_in_gt5_5to4", 0) for i in image_summaries)
    )
    rank_pos_total = int(sum(i.get("rank_pos_total", 0) for i in image_summaries))
    rank_pos_hungarian_all = int(sum(i.get("rank_pos_hungarian_all", 0) for i in image_summaries))
    rank_pos_shortside_matched = int(sum(i.get("rank_pos_shortside_matched", 0) for i in image_summaries))
    rank_pos_shortside_rescue = int(sum(i.get("rank_pos_shortside_rescue", 0) for i in image_summaries))
    rank_pos_shortside_reliable = int(sum(i.get("rank_pos_shortside_reliable", 0) for i in image_summaries))
    rank_pos_shortside_ultra = int(sum(i.get("rank_pos_shortside_ultra", 0) for i in image_summaries))
    rank_pos_gt4gt5_matched = int(sum(i.get("rank_pos_gt4gt5_matched", 0) for i in image_summaries))
    rank_pos_all_matched = int(sum(i.get("rank_pos_all_matched", 0) for i in image_summaries))
    rank_neg_duplicate_like = int(sum(i.get("rank_neg_duplicate_like", 0) for i in image_summaries))
    rank_neg_clear_far = int(sum(i.get("rank_neg_clear_far", 0) for i in image_summaries))
    rank_neg_near_ignored = int(sum(i.get("rank_neg_near_ignored", 0) for i in image_summaries))
    rank_neg_side_duplicate_like_total = int(
        sum(i.get("rank_neg_side_duplicate_like_count", 0) for i in image_summaries)
    )
    rank_neg_normal_duplicate_like_total = int(
        sum(i.get("rank_neg_normal_duplicate_like_count", 0) for i in image_summaries)
    )
    rank_side_ambiguous_ignored_total = int(
        sum(i.get("rank_side_ambiguous_ignored_count", 0) for i in image_summaries)
    )
    rank_side_duplicate_rejected_better_than_true_total = int(
        sum(i.get("rank_side_duplicate_rejected_better_than_true_count", 0) for i in image_summaries)
    )
    rank_loss_noop_images_total = int(sum(i.get("rank_loss_noop_images", 0) for i in image_summaries))
    rank_noop_because_no_pos_total = int(sum(i.get("rank_noop_because_no_pos", 0) for i in image_summaries))
    rank_noop_because_no_neg_total = int(sum(i.get("rank_noop_because_no_neg", 0) for i in image_summaries))
    near_gt_ignore_total = int(sum(1 for r in unmatched_records if r["is_near_gt_corridor"]))
    duplicate_like_rank_neg_total = int(sum(1 for r in unmatched_records if r.get("is_duplicate_like_rank_neg")))
    clear_far_rank_neg_total = int(sum(1 for r in unmatched_records if r["is_clear_far_spurious"]))
    clear_far_boundary_total = int(sum(i.get("clear_far_boundary_count", 0) for i in image_summaries))
    return {
        "result_type": "reporting_only_test_diagnostic" if has_test else "count_contract_diagnostic",
        "not_for_selection": bool(has_test),
        "label_contract": "validated_k56_fixed_y",
        "query_role_semantics": "training_hungarian_count_contract_masks",
        "raw_rescue_count_semantics": "training_hungarian_count_contract_masks",
        "raw_geometry_rescue_proxy_semantics": "diagnostic_geometry_proxy_not_training_assignment",
        "config": {
            "weights": str(Path(args.weights)),
            "dataset_root": str(Path(args.dataset_root)),
            "splits": list(args.splits),
            "allow_test_diagnostics": bool(args.allow_test_diagnostics),
            "imgsz": [int(x) for x in normalize_imgsz(args.imgsz)],
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "valid_before_maxdet": bool(args.valid_before_maxdet),
            "raw_match_dist_px": float(args.raw_match_dist_px),
            "side_raw_match_dist_px": float(args.side_raw_match_dist_px),
            "shortside_min_gt_lanes": int(args.shortside_min_gt_lanes),
            "shortside_min_valid_points": int(args.shortside_min_valid_points),
            "shortside_ultra_min_valid_points": int(args.shortside_ultra_min_valid_points),
            "shortside_reliable_min_valid_points": (
                None if args.shortside_reliable_min_valid_points is None else int(args.shortside_reliable_min_valid_points)
            ),
            "shortside_ultra_enable": bool(args.shortside_ultra_enable),
            "shortside_ultra_score_floor_gain": float(args.shortside_ultra_score_floor_gain),
            "shortside_ultra_valid_gain": float(args.shortside_ultra_valid_gain),
            "shortside_ultra_rank_pos": bool(args.shortside_ultra_rank_pos),
            "shortside_visible_max": int(args.shortside_visible_max),
            "shortside_use_median": bool(args.shortside_use_median),
            "shortside_median_margin": float(args.shortside_median_margin),
            "rank_focus_shortside": bool(args.rank_focus_shortside),
            "rank_pos_scope": str(args.rank_pos_scope),
            "rank_gt_min_lanes": int(args.rank_gt_min_lanes),
            "rank_side_duplicate_enable": bool(args.rank_side_duplicate_enable),
            "rank_side_dup_margin_px": float(args.rank_side_dup_margin_px),
            "rank_side_dup_min_overlap": int(args.rank_side_dup_min_overlap),
            "rank_pair_reduction": str(args.rank_pair_reduction),
            "base_ignore_raw_rescue": bool(args.base_ignore_raw_rescue),
            "base_ignore_rank_near": bool(args.base_ignore_rank_near),
            "base_ignore_farspur_near": bool(args.base_ignore_farspur_near),
            "base_ignore_duplicate_like": bool(args.base_ignore_duplicate_like),
            "legacy_spurious_active": bool(args.legacy_spurious_active),
            "near_dist_px": float(args.near_dist_px),
            "side_ignore_dist_px": float(args.side_ignore_dist_px),
            "clear_dist_px": float(args.clear_dist_px),
            "farspur_score_thr": float(args.farspur_score_thr),
            "farspur_min_valid_points": int(args.farspur_min_valid_points),
            "rank_dup_close_px": float(args.rank_dup_close_px),
            "rank_dup_min_overlap": int(args.rank_dup_min_overlap),
            "archive_root": str(Path(args.archive_root)),
            "official_gt_json": None if args.official_gt_json is None else str(Path(args.official_gt_json)),
            "official_runtime_ms": float(args.official_runtime_ms),
        },
        "images": int(len(image_summaries)),
        "target_bucket_images": int(sum(len(v) for v in bucket_images.values())),
        **acceptance_metrics,
        "count_confusion": dict(sorted(count_confusion.items())),
        **diagnostic_count_metrics,
        "official_metrics": official_metrics,
        "shortside_hungarian_rawmatch_candidate_count": shortside_hungarian_rawmatch_candidate_total,
        "shortside_unmatched_raw_rescue_candidate_count": shortside_unmatched_raw_rescue_candidate_total,
        "shortside_rawmatch_candidate_total_count": shortside_rawmatch_candidate_total,
        "raw_rescue_candidate_count": raw_rescue_candidate_total,
        "raw_rescue_final_count": raw_rescue_final_total,
        "raw_rescue_conflict_count": raw_rescue_conflict_total,
        "shortside_selected_hungarian_rawmatch": shortside_selected_hungarian_rawmatch_total,
        "shortside_selected_unmatched_rescue": shortside_selected_unmatched_rescue_total,
        "shortside_rescue_conflict": shortside_rescue_conflict_total,
        "shortside_missing_no_rawmatch": shortside_missing_no_rawmatch_total,
        "shortside_nearest_was_duplicate_but_hungarian_boosted": shortside_nearest_was_duplicate_but_hungarian_boosted_total,
        "shortside_selected_by_abs_count": shortside_selected_by_abs_total,
        "shortside_selected_by_median_count": shortside_selected_by_median_total,
        "shortside_selected_total_count": shortside_selected_total,
        "shortside_reliable_count": shortside_reliable_total,
        "shortside_reliable_selected_count": shortside_reliable_selected_total,
        "shortside_ultra_seen_count": shortside_ultra_seen_total,
        "shortside_ultra_short_count": shortside_ultra_short_total,
        "shortside_ultra_enabled_count": shortside_ultra_enabled_total,
        "shortside_ultra_score_floor_count": shortside_ultra_score_floor_total,
        "shortside_ultra_valid_count": shortside_ultra_valid_total,
        "shortside_visible_lt2_skipped_count": shortside_visible_lt2_skipped_total,
        "shortside_skipped_visible_lt2_count": shortside_skipped_visible_lt2_total,
        "shortside_ultra_in_gt4_4to3_count": shortside_ultra_in_gt4_4to3_total,
        "shortside_ultra_in_gt5_5to4_count": shortside_ultra_in_gt5_5to4_total,
        "shortside_visible_2_5_in_gt4_4to3": shortside_visible_2_5_in_gt4_4to3_total,
        "shortside_visible_2_5_in_gt5_5to4": shortside_visible_2_5_in_gt5_5to4_total,
        "rank_pos_scope": str(args.rank_pos_scope),
        "rank_pos_total": rank_pos_total,
        "rank_pos_hungarian_all": rank_pos_hungarian_all,
        "rank_pos_shortside_matched": rank_pos_shortside_matched,
        "rank_pos_shortside_rescue": rank_pos_shortside_rescue,
        "rank_pos_shortside_reliable": rank_pos_shortside_reliable,
        "rank_pos_shortside_ultra": rank_pos_shortside_ultra,
        "rank_pos_gt4gt5_matched": rank_pos_gt4gt5_matched,
        "rank_pos_all_matched": rank_pos_all_matched,
        "rank_neg_duplicate_like": rank_neg_duplicate_like,
        "rank_neg_clear_far": rank_neg_clear_far,
        "rank_neg_near_ignored": rank_neg_near_ignored,
        "rank_neg_side_duplicate_like_count": rank_neg_side_duplicate_like_total,
        "rank_neg_normal_duplicate_like_count": rank_neg_normal_duplicate_like_total,
        "rank_neg_clear_far_count": rank_neg_clear_far,
        "rank_side_ambiguous_ignored_count": rank_side_ambiguous_ignored_total,
        "rank_side_duplicate_rejected_better_than_true_count": (
            rank_side_duplicate_rejected_better_than_true_total
        ),
        "rank_loss_noop_images": rank_loss_noop_images_total,
        "rank_noop_because_no_pos": rank_noop_because_no_pos_total,
        "rank_noop_because_no_neg": rank_noop_because_no_neg_total,
        "near_gt_ignore_count": near_gt_ignore_total,
        "near_gt_ignored_count": near_gt_ignore_total,
        "legacy_spurious_active": bool(args.legacy_spurious_active),
        "duplicate_like_rank_neg_count": duplicate_like_rank_neg_total,
        "clear_far_rank_neg_count": clear_far_rank_neg_total,
        "clear_far_boundary_count": clear_far_boundary_total,
        "buckets": per_bucket,
    }


def main() -> None:
    args = parse_args()
    if "test" in set(args.splits) and not bool(args.allow_test_diagnostics):
        raise SystemExit(
            "Detailed test diagnostics are reporting-only and can leak model-selection signal. "
            "Use --allow-test-diagnostics only after the candidate has been selected by official-val."
        )
    dataset_root = Path(args.dataset_root)
    imgsz = normalize_imgsz(args.imgsz)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    model = load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    criterion = make_contract_loss(model, args, imgsz)
    args.rank_pos_scope = str(getattr(criterion, "rank_pos_scope", args.rank_pos_scope))

    if args.warmup > 0:
        dummy = torch.zeros(1, 3, imgsz[0], imgsz[1], device=device)
        dummy = dummy.half() if args.half else dummy.float()
        with torch.inference_mode():
            for _ in range(int(args.warmup)):
                _ = model(dummy)

    image_summaries: list[dict] = []
    gt_records: list[dict] = []
    unmatched_records: list[dict] = []
    tusimple_predictions: list[dict] = []
    with torch.inference_mode():
        for split in args.splits:
            image_dir = dataset_root / "images" / split
            labels_dir = dataset_root / "labels_gcs" / split
            images = collect_images(image_dir, max_images=int(args.max_images))
            official_h_samples_by_raw = load_official_h_samples_by_raw(args, split)
            for image_path in images:
                image_summary, gt_rows, unmatched_rows, tusimple_prediction = diagnose_image(
                    model,
                    criterion,
                    image_path,
                    labels_dir,
                    split=split,
                    dataset_root=dataset_root,
                    imgsz=imgsz,
                    device=device,
                    half=bool(args.half),
                    args=args,
                    official_h_samples_by_raw=official_h_samples_by_raw,
                )
                image_summaries.append(image_summary)
                gt_records.extend(gt_rows)
                unmatched_records.extend(unmatched_rows)
                if tusimple_prediction is not None:
                    tusimple_predictions.append(tusimple_prediction)

    summary = build_summary(image_summaries, gt_records, unmatched_records, tusimple_predictions, args)
    write_json(save_dir / "summary.json", summary)
    write_jsonl(save_dir / "gt_lane_records.jsonl", gt_records)
    write_jsonl(save_dir / "unmatched_query_records.jsonl", unmatched_records)
    print(f"Wrote {save_dir / 'summary.json'}")
    print(f"Wrote {save_dir / 'gt_lane_records.jsonl'}")
    print(f"Wrote {save_dir / 'unmatched_query_records.jsonl'}")


if __name__ == "__main__":
    main()
