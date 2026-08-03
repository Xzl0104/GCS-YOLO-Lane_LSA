"""Audit prediction-only local short-segment replacements.

This diagnostic reuses the formal query decoder and TuSimple evaluator. It
does not change decode behavior and is intended to separate useful segment
replacements from replacements that damage the frozen base query.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    TuSimpleOfficialLaneEval,
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    official_gt_contract_summary,
    read_tusimple_json_lines,
    resolve_tusimple_gt_json,
    tusimple_image_path,
    valid_tusimple_lanes,
)
from tools.infer_gcs import load_gcs_model, preprocess_image  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_VAL_GT = (
    ROOT
    / "runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/"
    / "tusimple_official_val_363_folder_aware_seed20260602.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit prediction-only local short-segment replacements against the frozen base decoder."
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--gt-json", default=None)
    parser.add_argument("--allow-noncanonical-gt", action="store_true")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--point-valid-thr", type=float, default=0.6)
    parser.add_argument("--nms-dist-px", type=float, default=0.0)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--segment-score-thrs", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--segment-short-min-points", type=int, default=3)
    parser.add_argument("--segment-short-max-points", type=int, default=10)
    parser.add_argument("--segment-pred-valid-overlap-min", type=int, default=0)
    parser.add_argument("--runtime-ms", type=float, default=1.0)
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _default_gt_json(args: argparse.Namespace) -> str | None:
    if args.gt_json:
        return args.gt_json
    if str(args.split).lower() == "val" and DEFAULT_VAL_GT.exists():
        return str(DEFAULT_VAL_GT)
    return None


def _limit_records(records: list[dict], max_images: int) -> list[dict]:
    if max_images and max_images > 0:
        return records[: int(max_images)]
    return records


def _threshold_logit(probability: float) -> float:
    value = float(probability)
    if value <= 0.0:
        return float("-inf")
    if value >= 1.0:
        return float("inf")
    return float(math.log(value / (1.0 - value)))


def _segment_apply_details(
    preds: dict[str, torch.Tensor],
    *,
    point_valid_thr: float,
    segment_score_thr: float,
    segment_short_min_points: int,
    segment_short_max_points: int,
    segment_pred_valid_overlap_min: int,
) -> dict[str, torch.Tensor]:
    """Mirror the segment gate in ``decode_gcs_predictions`` for audit rows."""
    pred_valid_logits = preds["pred_valid_logits"][0].detach().float().cpu()
    segment_logits = preds["pred_short_segment_logits"][0].detach().float().cpu()
    window_mask = preds["pred_short_segment_window_mask"].detach().bool().cpu()
    if window_mask.ndim == 3 and window_mask.shape[0] == 1:
        window_mask = window_mask[0]
    if window_mask.ndim != 2:
        raise RuntimeError(f"Expected segment window mask S x K, got {tuple(window_mask.shape)}.")
    if segment_logits.ndim != 2:
        raise RuntimeError(f"Expected segment logits Q x S, got {tuple(segment_logits.shape)}.")
    if window_mask.shape[0] != segment_logits.shape[1]:
        raise RuntimeError(
            f"Segment window count mismatch: mask={tuple(window_mask.shape)}, logits={tuple(segment_logits.shape)}."
        )

    valid_scores = pred_valid_logits.sigmoid()
    query_valid = valid_scores >= float(point_valid_thr)
    segment_lengths = window_mask.sum(dim=1)
    segment_gate = (segment_lengths >= int(segment_short_min_points)) & (
        segment_lengths <= int(segment_short_max_points)
    )
    segment_gate = segment_gate.view(1, -1).expand(segment_logits.shape[0], -1)
    if int(segment_pred_valid_overlap_min) > 0:
        valid_overlap = (
            window_mask.view(1, window_mask.shape[0], window_mask.shape[1])
            & query_valid.view(query_valid.shape[0], 1, query_valid.shape[1])
        ).sum(dim=2)
        segment_gate = segment_gate & (valid_overlap >= int(segment_pred_valid_overlap_min))

    masked_logits = segment_logits.masked_fill(~segment_gate, float("-inf"))
    best_logits, best_indices = masked_logits.max(dim=1)
    apply = torch.isfinite(best_logits) & (best_logits >= _threshold_logit(segment_score_thr))
    return {
        "segment_lengths": segment_lengths,
        "segment_gate": segment_gate,
        "best_logits": best_logits,
        "best_indices": best_indices,
        "best_scores": best_logits.sigmoid(),
        "apply": apply,
        "valid_scores": valid_scores,
    }


def _decode(
    preds: dict[str, torch.Tensor],
    *,
    image_shape: tuple[int, int],
    args: argparse.Namespace,
    segment_decode: bool,
    segment_score_thr: float,
    min_points: int | None = None,
    conf: float | None = None,
) -> list[dict]:
    pred_valid = preds.get("pred_valid_logits")
    segment_points = preds.get("pred_short_segment_points")
    segment_logits = preds.get("pred_short_segment_logits")
    segment_window_mask = preds.get("pred_short_segment_window_mask")
    return decode_gcs_predictions(
        preds["pred_points"][0],
        preds["pred_logits"][0],
        pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
        pred_short_segment_points=segment_points[0] if segment_points is not None else None,
        pred_short_segment_logits=segment_logits[0] if segment_logits is not None else None,
        pred_short_segment_window_mask=segment_window_mask,
        image_shape=image_shape,
        score_thr=float(args.conf if conf is None else conf),
        point_valid_thr=float(args.point_valid_thr),
        min_points=int(args.min_points if min_points is None else min_points),
        max_det=int(args.max_det),
        nms_dist_px=float(args.nms_dist_px),
        segment_decode=bool(segment_decode),
        segment_score_thr=float(segment_score_thr),
        segment_short_min_points=int(args.segment_short_min_points),
        segment_short_max_points=int(args.segment_short_max_points),
        segment_pred_valid_overlap_min=int(args.segment_pred_valid_overlap_min),
    )


def _query_lane_map(
    preds: dict[str, torch.Tensor],
    *,
    image_shape: tuple[int, int],
    args: argparse.Namespace,
    segment_decode: bool,
    segment_score_thr: float,
) -> dict[int, dict]:
    """Decode every query with permissive output filtering for replacement audit."""
    lanes = _decode(
        preds,
        image_shape=image_shape,
        args=args,
        segment_decode=segment_decode,
        segment_score_thr=segment_score_thr,
        min_points=1,
        conf=-1.0,
    )
    return {int(lane["query"]): lane for lane in lanes}


def _lane_xs(lanes: list[list[int]], index: int) -> np.ndarray:
    return np.asarray(lanes[index], dtype=np.float32)


def _best_mean_ape(pred_lane: list[int] | None, gt_lanes: list[list[float]]) -> tuple[float, int]:
    if pred_lane is None:
        return float("inf"), -1
    pred = _lane_xs([pred_lane], 0)
    best = float("inf")
    best_idx = -1
    for idx, gt in enumerate(gt_lanes):
        gt_arr = np.asarray(gt, dtype=np.float32)
        valid = (pred >= 0.0) & (gt_arr >= 0.0)
        if not bool(valid.any()):
            continue
        ape = float(np.mean(np.abs(pred[valid] - gt_arr[valid])))
        if ape < best:
            best = ape
            best_idx = int(idx)
    return best, best_idx


def _official_match_flags(pred_lanes: list[list[int]], record: dict) -> list[bool]:
    gt_lanes = valid_tusimple_lanes(record.get("lanes", []))
    y_samples = list(record["h_samples"])
    if not gt_lanes:
        return []
    angles = [TuSimpleOfficialLaneEval.get_angle(gt, y_samples) for gt in gt_lanes]
    thresholds = [
        TuSimpleOfficialLaneEval.pixel_thresh / max(float(np.cos(angle)), 1e-12) for angle in angles
    ]
    flags: list[bool] = []
    for gt, threshold in zip(gt_lanes, thresholds):
        max_acc = max(
            (
                TuSimpleOfficialLaneEval.line_accuracy(pred, gt, threshold)
                for pred in pred_lanes
            ),
            default=0.0,
        )
        flags.append(bool(max_acc >= TuSimpleOfficialLaneEval.pt_thresh))
    return flags


def _official_metrics(pred_lanes: list[list[int]], record: dict, runtime_ms: float) -> dict[str, float]:
    accuracy, fp, fn = TuSimpleOfficialLaneEval.bench(
        pred=list(pred_lanes),
        gt=list(record.get("lanes", [])),
        y_samples=list(record["h_samples"]),
        running_time=float(runtime_ms),
    )
    return {
        "official_acc": float(accuracy),
        "official_fp": float(fp),
        "official_fn": float(fn),
    }


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _new_threshold_summary() -> dict[str, Any]:
    return {
        "images": 0,
        "official_acc_sum": 0.0,
        "official_fp_sum": 0.0,
        "official_fn_sum": 0.0,
        "segment_applied_queries": 0,
        "segment_applied_images": 0,
        "base_hit_to_segment_miss": 0,
        "base_miss_to_segment_hit": 0,
        "base_hit_to_segment_hit": 0,
        "base_miss_to_segment_miss": 0,
        "segment_better_images": 0,
        "segment_worse_images": 0,
        "segment_tied_images": 0,
        "replacement_queries_near_base_gt20": 0,
        "replacement_queries_near_base_gt40": 0,
        "replacement_queries_near_segment_gt20": 0,
        "replacement_queries_recovered20": 0,
        "replacement_queries_harmed20": 0,
        "replacement_queries_unmatched_base": 0,
    }


def _finalize_threshold_summary(summary: dict[str, Any]) -> dict[str, Any]:
    images = max(int(summary["images"]), 1)
    out = dict(summary)
    out["official_acc"] = round(float(summary["official_acc_sum"]) / images, 6)
    out["official_FP"] = round(float(summary["official_fp_sum"]) / images, 6)
    out["official_FN"] = round(float(summary["official_fn_sum"]) / images, 6)
    out["replacement_rate_per_image"] = round(
        float(summary["segment_applied_queries"]) / images, 6
    )
    for key in ("official_acc_sum", "official_fp_sum", "official_fn_sum"):
        out.pop(key, None)
    return out


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_path = resolve_tusimple_gt_json(
        archive_root,
        split=args.split,
        gt_json=_default_gt_json(args),
    )
    records = _limit_records(read_tusimple_json_lines(gt_path), args.max_images)
    if not records:
        raise ValueError(f"No TuSimple records found in {gt_path}.")
    gt_contract = official_gt_contract_summary(
        split=args.split,
        gt_json=gt_path,
        gt_records=records,
        allow_noncanonical_gt=bool(args.allow_noncanonical_gt),
    )
    imgsz = normalize_imgsz(args.imgsz, dataset="tusimple")
    score_thrs = sorted({float(x) for x in args.segment_score_thrs})
    if any(x < 0.0 or x > 1.0 for x in score_thrs):
        raise ValueError(f"segment score thresholds must be in [0, 1], got {score_thrs}.")

    device = select_device(args.device)
    model = load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    if args.warmup > 0:
        warm_path = tusimple_image_path(archive_root, str(records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device, half=bool(args.half))
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device)

    threshold_summaries = {str(thr): _new_threshold_summary() for thr in score_thrs}
    base_metrics_sum = {"official_acc": 0.0, "official_fp": 0.0, "official_fn": 0.0}
    image_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    infer_time_s = 0.0
    post_time_s = 0.0

    for record in records:
        raw_file = str(record["raw_file"]).replace("\\", "/")
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_shape = image.shape[:2]
        tensor = preprocess_image(image, imgsz, device=device, half=bool(args.half))
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            preds = model(tensor)
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        if "pred_short_segment_points" not in preds or "pred_short_segment_logits" not in preds:
            raise RuntimeError("Checkpoint does not emit pred_short_segment_points/logits.")
        if "pred_short_segment_window_mask" not in preds:
            raise RuntimeError("Checkpoint does not emit pred_short_segment_window_mask.")

        base_lanes = _decode(
            preds,
            image_shape=image_shape,
            args=args,
            segment_decode=False,
            segment_score_thr=0.5,
        )
        base_tusimple = gcs_lanes_to_tusimple_lanes(base_lanes, record["h_samples"], image_shape=image_shape)
        base_metrics = _official_metrics(base_tusimple, record, args.runtime_ms)
        base_flags = _official_match_flags(base_tusimple, record)
        base_metrics_sum["official_acc"] += base_metrics["official_acc"]
        base_metrics_sum["official_fp"] += base_metrics["official_fp"]
        base_metrics_sum["official_fn"] += base_metrics["official_fn"]
        gt_lanes = valid_tusimple_lanes(record.get("lanes", []))

        base_query_map = _query_lane_map(
            preds,
            image_shape=image_shape,
            args=args,
            segment_decode=False,
            segment_score_thr=0.5,
        )
        apply_details_by_threshold: dict[str, dict[str, torch.Tensor]] = {}
        segment_lanes_by_threshold: dict[str, list[dict]] = {}
        segment_tusimple_by_threshold: dict[str, list[list[int]]] = {}
        segment_flags_by_threshold: dict[str, list[bool]] = {}
        segment_metrics_by_threshold: dict[str, dict[str, float]] = {}
        for thr in score_thrs:
            key = str(thr)
            details = _segment_apply_details(
                preds,
                point_valid_thr=args.point_valid_thr,
                segment_score_thr=thr,
                segment_short_min_points=args.segment_short_min_points,
                segment_short_max_points=args.segment_short_max_points,
                segment_pred_valid_overlap_min=args.segment_pred_valid_overlap_min,
            )
            apply_details_by_threshold[key] = details
            segment_lanes = _decode(
                preds,
                image_shape=image_shape,
                args=args,
                segment_decode=True,
                segment_score_thr=thr,
            )
            segment_tusimple = gcs_lanes_to_tusimple_lanes(
                segment_lanes,
                record["h_samples"],
                image_shape=image_shape,
            )
            segment_metrics = _official_metrics(segment_tusimple, record, args.runtime_ms)
            segment_flags = _official_match_flags(segment_tusimple, record)
            segment_lanes_by_threshold[key] = segment_lanes
            segment_tusimple_by_threshold[key] = segment_tusimple
            segment_flags_by_threshold[key] = segment_flags
            segment_metrics_by_threshold[key] = segment_metrics

            summary = threshold_summaries[key]
            summary["images"] += 1
            summary["official_acc_sum"] += segment_metrics["official_acc"]
            summary["official_fp_sum"] += segment_metrics["official_fp"]
            summary["official_fn_sum"] += segment_metrics["official_fn"]
            applied_queries = int(details["apply"].sum().item())
            summary["segment_applied_queries"] += applied_queries
            summary["segment_applied_images"] += int(applied_queries > 0)
            if segment_metrics["official_acc"] > base_metrics["official_acc"]:
                summary["segment_better_images"] += 1
            elif segment_metrics["official_acc"] < base_metrics["official_acc"]:
                summary["segment_worse_images"] += 1
            else:
                summary["segment_tied_images"] += 1

            for gt_idx, (base_flag, segment_flag) in enumerate(zip(base_flags, segment_flags)):
                if base_flag and segment_flag:
                    summary["base_hit_to_segment_hit"] += 1
                elif base_flag and not segment_flag:
                    summary["base_hit_to_segment_miss"] += 1
                elif not base_flag and segment_flag:
                    summary["base_miss_to_segment_hit"] += 1
                else:
                    summary["base_miss_to_segment_miss"] += 1
                gt_rows.append(
                    {
                        "split": args.split,
                        "raw_file": raw_file,
                        "gt_lane_id": gt_idx,
                        "gt_count": len(gt_lanes),
                        "base_match": bool(base_flag),
                        "segment_score_thr": thr,
                        "segment_match": bool(segment_flag),
                        "base_to_segment": (
                            "hit_to_hit"
                            if base_flag and segment_flag
                            else "hit_to_miss"
                            if base_flag
                            else "miss_to_hit"
                            if segment_flag
                            else "miss_to_miss"
                        ),
                    }
                )

            segment_query_map = _query_lane_map(
                preds,
                image_shape=image_shape,
                args=args,
                segment_decode=True,
                segment_score_thr=thr,
            )
            details = apply_details_by_threshold[key]
            for query_idx in torch.nonzero(details["apply"], as_tuple=False).flatten().tolist():
                base_lane = base_query_map.get(int(query_idx))
                segment_lane = segment_query_map.get(int(query_idx))
                base_lane_tusimple = (
                    gcs_lanes_to_tusimple_lanes([base_lane], record["h_samples"], image_shape=image_shape)
                    if base_lane is not None
                    else []
                )
                segment_lane_tusimple = (
                    gcs_lanes_to_tusimple_lanes([segment_lane], record["h_samples"], image_shape=image_shape)
                    if segment_lane is not None
                    else []
                )
                base_ape, base_gt_idx = _best_mean_ape(
                    base_lane_tusimple[0] if base_lane_tusimple else None,
                    gt_lanes,
                )
                segment_ape, segment_gt_idx = _best_mean_ape(
                    segment_lane_tusimple[0] if segment_lane_tusimple else None,
                    gt_lanes,
                )
                base_near20 = bool(base_ape <= 20.0)
                base_near40 = bool(base_ape <= 40.0)
                segment_near20 = bool(segment_ape <= 20.0)
                recovered20 = bool((not base_near20) and segment_near20)
                harmed20 = bool(base_near20 and (not segment_near20))
                if base_near20:
                    summary["replacement_queries_near_base_gt20"] += 1
                if base_near40:
                    summary["replacement_queries_near_base_gt40"] += 1
                if segment_near20:
                    summary["replacement_queries_near_segment_gt20"] += 1
                if recovered20:
                    summary["replacement_queries_recovered20"] += 1
                if harmed20:
                    summary["replacement_queries_harmed20"] += 1
                if not base_near40:
                    summary["replacement_queries_unmatched_base"] += 1
                query_rows.append(
                    {
                        "split": args.split,
                        "raw_file": raw_file,
                        "segment_score_thr": thr,
                        "query": int(query_idx),
                        "segment_index": int(details["best_indices"][query_idx].item()),
                        "segment_score": float(details["best_scores"][query_idx].item()),
                        "segment_length": int(details["segment_lengths"][details["best_indices"][query_idx]].item()),
                        "base_ape_px": _safe_float(base_ape),
                        "segment_ape_px": _safe_float(segment_ape),
                        "base_gt_lane_id": int(base_gt_idx),
                        "segment_gt_lane_id": int(segment_gt_idx),
                        "base_near20": base_near20,
                        "base_near40": base_near40,
                        "segment_near20": segment_near20,
                        "recovered20": recovered20,
                        "harmed20": harmed20,
                        "base_query_present_after_decode": base_lane is not None,
                        "segment_query_present_after_decode": segment_lane is not None,
                    }
                )

            image_rows.append(
                {
                    "split": args.split,
                    "raw_file": raw_file,
                    "segment_score_thr": thr,
                    "base_acc": base_metrics["official_acc"],
                    "base_fp": base_metrics["official_fp"],
                    "base_fn": base_metrics["official_fn"],
                    "segment_acc": segment_metrics["official_acc"],
                    "segment_fp": segment_metrics["official_fp"],
                    "segment_fn": segment_metrics["official_fn"],
                    "delta_acc": segment_metrics["official_acc"] - base_metrics["official_acc"],
                    "delta_fp": segment_metrics["official_fp"] - base_metrics["official_fp"],
                    "delta_fn": segment_metrics["official_fn"] - base_metrics["official_fn"],
                    "applied_queries": int(details["apply"].sum().item()),
                    "pred_lanes_base": len(base_tusimple),
                    "pred_lanes_segment": len(segment_tusimple),
                }
            )

        t2 = time.perf_counter()
        infer_time_s += t1 - t0
        post_time_s += t2 - t1

    images = max(len(records), 1)
    base_summary = {
        "official_acc": round(base_metrics_sum["official_acc"] / images, 6),
        "official_FP": round(base_metrics_sum["official_fp"] / images, 6),
        "official_FN": round(base_metrics_sum["official_fn"] / images, 6),
    }
    finalized_thresholds = {
        key: _finalize_threshold_summary(value) for key, value in threshold_summaries.items()
    }
    save_dir = (
        Path(args.save_dir)
        if args.save_dir
        else ROOT
        / "runs/gcs_lane"
        / f"{Path(args.weights).stem}_replacement_audit_{args.split}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(save_dir / "replacement_image_rows.csv", image_rows)
    _write_csv(save_dir / "replacement_gt_rows.csv", gt_rows)
    _write_csv(save_dir / "replacement_query_rows.csv", query_rows)
    output = {
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "archive_root": str(archive_root.resolve()),
            "split": args.split,
            "gt_json": str(gt_path.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "device": str(args.device),
            "half": bool(args.half),
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "segment_score_thrs": score_thrs,
            "segment_short_min_points": int(args.segment_short_min_points),
            "segment_short_max_points": int(args.segment_short_max_points),
            "segment_pred_valid_overlap_min": int(args.segment_pred_valid_overlap_min),
            "runtime_ms": float(args.runtime_ms),
            "test_used": False,
        },
        "base": base_summary,
        "by_segment_score_thr": finalized_thresholds,
        "timing": {
            "images": len(records),
            "avg_inference_ms": round(infer_time_s * 1000.0 / images, 4),
            "avg_audit_postprocess_ms": round(post_time_s * 1000.0 / images, 4),
        },
        **gt_contract,
    }
    (save_dir / "replacement_audit_summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))
    print(f"saved to: {save_dir.resolve()}")
    return output


def main() -> None:
    run_audit(parse_args())


if __name__ == "__main__":
    main()
