from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image
from ultralytics.models.gcs.decode_ordered_slot import decode_ordered_slot_predictions
from ultralytics.models.gcs.decode_summary import (
    build_ordered_slot_decode_summary,
    ordered_slot_decode_params,
    ordered_slot_decode_runtime_config,
)
from ultralytics.models.gcs.mode_utils import resolve_decode_mode
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, assert_gcs_shape, normalize_imgsz, shape_str
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions, draw_gcs_lanes, save_gcs_lanes_txt
from ultralytics.utils.torch_utils import select_device


DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "overfit20" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544" / "images" / "val"
DEFAULT_LABELS = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544" / "labels_gcs" / "val"
DEFAULT_CULANE_ARCHIVE_ROOT = Path(r"D:\BaiduNetdiskDownload\CULane")
CULANE_RAW_SHAPE = (590, 1640)
CULANE_LANE_WIDTH = 30
CULANE_IOU_THRESHOLD = 0.5


def dataset_defaults(dataset: str) -> dict[str, Path]:
    """Return conventional validation paths for a converted GCS dataset."""
    root = ROOT / "datasets" / ("tusimple_fixed_y_k56_960x544" if dataset.lower() == "tusimple" else dataset.lower())
    return {
        "source": root / "images" / "val",
        "labels": root / "labels_gcs" / "val",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GCS-YOLO-Lane structured lane predictions.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt or model yaml.")
    parser.add_argument("--source", default=None, help="Image file, image directory, or txt list.")
    parser.add_argument("--labels", default=None, help="labels_gcs directory. Empty means infer from image path.")
    parser.add_argument(
        "--metric",
        choices=("ape", "culane_iou"),
        default="ape",
        help="Primary matching metric. Use culane_iou for CULane benchmark F1.",
    )
    parser.add_argument(
        "--culane-archive-root",
        default=str(DEFAULT_CULANE_ARCHIVE_ROOT),
        help="Extracted CULane root containing the original .lines.txt annotations.",
    )
    parser.add_argument(
        "--culane-raw-imgsz",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=CULANE_RAW_SHAPE,
        help="Original CULane image shape used by the official evaluator.",
    )
    parser.add_argument(
        "--culane-lane-width",
        type=int,
        default=CULANE_LANE_WIDTH,
        help="Lane mask width in original CULane pixels.",
    )
    parser.add_argument(
        "--culane-iou-thr",
        type=float,
        default=CULANE_IOU_THRESHOLD,
        help="Strict CULane lane IoU threshold; IoU must be greater than this value.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS inference shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Lane existence confidence threshold.")
    parser.add_argument("--decode-mode", choices=("auto", "query", "ordered_slot"), default="auto", help="Decode path.")
    parser.add_argument("--gcs-min-lanes", type=int, default=2, help="ordered_slot minimum supported lane count.")
    parser.add_argument("--gcs-max-lanes", type=int, default=5, help="ordered_slot maximum supported lane count.")
    parser.add_argument("--gcs-num-slots", type=int, default=5, help="ordered_slot slot count.")
    parser.add_argument(
        "--gcs-min-interval-points",
        type=int,
        default=2,
        help="ordered_slot minimum decoded start/end interval length.",
    )
    parser.add_argument(
        "--gcs-bottom-order-margin-px",
        type=float,
        default=2.0,
        help="ordered_slot bottom-x left-to-right order margin in pixels.",
    )
    parser.add_argument(
        "--point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold for fixed-y lane decoding.",
    )
    parser.add_argument("--ape-thr", type=float, default=20.0, help="APE threshold in pixels for TP matching.")
    parser.add_argument("--match-gate-px", type=float, default=None, help="Strict eval APE gate in pixels. Defaults to --ape-thr.")
    parser.add_argument("--max-x-dist", type=float, default=0.0, help="Optional strict eval mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--min-overlap", type=int, default=2, help="Minimum valid overlapping GT points required for eval matching.")
    parser.add_argument("--nms-dist-px", type=float, default=50.0, help="Optional lane duplicate suppression distance in pixels. 0 disables.")
    parser.add_argument("--max-det", type=int, default=8, help="Maximum decoded lane queries per image.")
    parser.add_argument("--count-aware-topk", action="store_true", help="Use count_score to keep only the quality-best dynamic lane count.")
    parser.add_argument("--count-aware-min-k", type=int, default=3, help="Minimum k_hat for --count-aware-topk.")
    parser.add_argument("--count-aware-max-k", type=int, default=5, help="Maximum k_hat for --count-aware-topk.")
    parser.add_argument("--count-aware-length-norm", type=float, default=12.0, help="Visible-point count that saturates count-aware length quality.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images. 0 means all.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of untimed warmup forwards before benchmarking.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--save-dir", default="runs/gcs_lane/eval", help="Directory for eval_summary.json and outputs.")
    parser.add_argument(
        "--save-json",
        nargs="?",
        const=True,
        default=False,
        help="Save per-image predictions and matching details. Optionally pass a JSON output path.",
    )
    parser.add_argument("--save-img", action="store_true", help="Save rendered prediction images.")
    parser.add_argument("--save-txt", action="store_true", help="Save normalized lane predictions as txt.")
    parser.add_argument("--line-width", type=int, default=2, help="Polyline width for saved prediction images.")
    return parser.parse_args()


def label_path_for_image(image_path: Path, label_dir: str | Path | None) -> Path:
    """Map an image path to its GCS npz label."""
    if label_dir:
        return Path(label_dir) / f"{image_path.stem}.npz"

    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels_gcs"
        return Path(*parts).with_suffix(".npz")
    return image_path.parent.parent / "labels_gcs" / image_path.parent.name / f"{image_path.stem}.npz"


def load_gcs_label(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized GT lanes and valid masks from one GCS npz label."""
    if not label_path.exists():
        raise FileNotFoundError(f"Missing GCS label: {label_path}")
    with np.load(label_path, allow_pickle=False) as data:
        required = {"lanes", "lane_valid"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{label_path} missing required arrays: {sorted(missing)}")
        lanes = data["lanes"].astype(np.float32)
        valid = data["lane_valid"].astype(np.float32)
        point_mode = str(np.asarray(data["point_mode"]).item()) if "point_mode" in data else "free"

    if lanes.ndim != 3 or lanes.shape[-1] != 2:
        raise ValueError(f"{label_path}: lanes must have shape N x K x 2, got {lanes.shape}")
    if valid.shape != lanes.shape[:2]:
        raise ValueError(f"{label_path}: lane_valid shape {valid.shape} must match lanes {lanes.shape[:2]}")
    if not np.isfinite(lanes).all() or not np.isfinite(valid).all():
        raise ValueError(f"{label_path}: GT lanes contain NaN or Inf values.")

    valid = (valid > 0.5).astype(np.float32)
    lanes = np.clip(lanes, 0.0, 1.0)
    keep = valid.sum(axis=1) >= 2
    lanes = lanes[keep]
    valid = valid[keep]
    if point_mode.lower() in {"fixed_y", "fixed-y", "fixedy"}:
        for i, (lane, lane_valid) in enumerate(zip(lanes, valid)):
            ys = lane[lane_valid > 0.5, 1]
            if ys.shape[0] >= 2 and not np.all(np.diff(ys) <= 1e-6):
                raise ValueError(f"{label_path}: fixed_y lane {i} valid y anchors must be bottom-to-top.")
        return lanes.astype(np.float32), valid.astype(np.float32)

    ordered_lanes = np.zeros_like(lanes, dtype=np.float32)
    ordered_valid = np.zeros_like(valid, dtype=np.float32)
    for i, (lane, lane_valid) in enumerate(zip(lanes, valid)):
        points = lane[lane_valid > 0.5]
        order = np.argsort(-points[:, 1], kind="stable")
        points = points[order]
        ordered_lanes[i, : points.shape[0]] = points
        ordered_valid[i, : points.shape[0]] = 1.0
    return ordered_lanes, ordered_valid


def _scalar_label_string(value: np.ndarray) -> str:
    """Read a scalar string stored in a converted CULane NPZ label."""
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(array.reshape(-1)[0])


def parse_culane_lines_file(path: Path) -> list[list[tuple[float, float]]]:
    """Parse one original CULane ``.lines.txt`` file into pixel-space lanes."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing CULane lane annotation: {path}")

    lanes: list[list[tuple[float, float]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        tokens = line.strip().split()
        if not tokens:
            continue
        if len(tokens) % 2:
            raise ValueError(f"{path}:{line_number} has an odd number of coordinate values.")
        points: list[tuple[float, float]] = []
        for index in range(0, len(tokens), 2):
            try:
                x = float(tokens[index])
                y = float(tokens[index + 1])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} contains non-numeric coordinates.") from exc
            if np.isfinite(x) and np.isfinite(y):
                points.append((x, y))
        if len(points) >= 2:
            lanes.append(points)
    return lanes


def culane_lines_path(label_path: Path, archive_root: str | Path) -> Path:
    """Resolve the original CULane ``.lines.txt`` path recorded in one NPZ label."""
    with np.load(label_path, allow_pickle=False) as data:
        if "source_lines" not in data.files:
            raise KeyError(f"{label_path} does not contain source_lines metadata.")
        source_lines = _scalar_label_string(data["source_lines"])
    if not source_lines:
        raise ValueError(f"{label_path} contains an empty source_lines metadata value.")
    relative = Path(source_lines.replace("/", os.sep))
    path = Path(archive_root).expanduser().resolve() / relative
    if not path.is_file():
        raise FileNotFoundError(f"Original CULane annotation does not exist: {path}")
    return path


def culane_spline_interp_points(
    points: list[tuple[float, float]],
    samples_per_segment: int = 50,
) -> np.ndarray:
    """Match the official CULane natural cubic spline interpolation."""
    if samples_per_segment <= 0:
        raise ValueError(f"samples_per_segment must be positive, got {samples_per_segment}")
    clean = np.asarray(points, dtype=np.float64)
    if clean.ndim != 2 or clean.shape[1] != 2 or clean.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    if not np.isfinite(clean).all():
        raise ValueError("CULane lane points contain NaN or Inf.")

    segment_lengths = np.linalg.norm(np.diff(clean, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-9))
    clean = clean[keep]
    if clean.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(clean, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    if clean.shape[0] == 2:
        fractions = np.linspace(0.0, 1.0, samples_per_segment + 1, dtype=np.float64)
        return clean[0] + fractions[:, None] * (clean[1] - clean[0])

    spline = CubicSpline(arc, clean, axis=0, bc_type="natural")
    chunks = []
    for index, length in enumerate(segment_lengths):
        local_arc = np.linspace(0.0, length, samples_per_segment, endpoint=False, dtype=np.float64)
        chunks.append(spline(arc[index] + local_arc))
    chunks.append(clean[-1][None, :])
    return np.concatenate(chunks, axis=0)


def culane_lane_mask(
    lane: list[tuple[float, float]] | np.ndarray,
    image_shape: tuple[int, int],
    lane_width: int = CULANE_LANE_WIDTH,
) -> np.ndarray:
    """Rasterize one lane using the official CULane spline and mask width."""
    height, width = int(image_shape[0]), int(image_shape[1])
    lane_width = int(lane_width)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid CULane mask shape: {image_shape}")
    if lane_width <= 0:
        raise ValueError(f"lane_width must be positive, got {lane_width}")

    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(lane, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        return mask
    interpolated = culane_spline_interp_points(points.tolist(), samples_per_segment=50)
    if interpolated.shape[0] < 2:
        return mask
    integer_points = np.rint(interpolated).astype(np.int32)
    for first, second in zip(integer_points[:-1], integer_points[1:]):
        cv2.line(
            mask,
            (int(first[0]), int(first[1])),
            (int(second[0]), int(second[1])),
            color=1,
            thickness=lane_width,
            lineType=cv2.LINE_8,
        )
    return mask


def culane_lane_iou(
    lane_a: list[tuple[float, float]] | np.ndarray,
    lane_b: list[tuple[float, float]] | np.ndarray,
    image_shape: tuple[int, int],
    lane_width: int = CULANE_LANE_WIDTH,
) -> float:
    """Calculate CULane region IoU for one lane pair."""
    mask_a = culane_lane_mask(lane_a, image_shape=image_shape, lane_width=lane_width)
    mask_b = culane_lane_mask(lane_b, image_shape=image_shape, lane_width=lane_width)
    intersection = int(np.logical_and(mask_a > 0, mask_b > 0).sum())
    union = int(np.logical_or(mask_a > 0, mask_b > 0).sum())
    return float(intersection / union) if union else 0.0


def load_culane_ground_truth(
    label_path: Path,
    archive_root: str | Path,
    raw_shape: tuple[int, int],
) -> list[list[tuple[float, float]]]:
    """Load and validate original CULane lanes for official-style evaluation."""
    with np.load(label_path, allow_pickle=False) as data:
        if "raw_image_shape" in data.files:
            recorded_shape = tuple(int(x) for x in np.asarray(data["raw_image_shape"]).reshape(-1)[:2])
            if recorded_shape != tuple(raw_shape):
                raise ValueError(
                    f"{label_path}: raw_image_shape={recorded_shape} does not match requested {tuple(raw_shape)}."
                )
    return parse_culane_lines_file(culane_lines_path(label_path, archive_root))


def prediction_lane_points_raw(
    lane: dict,
    raw_shape: tuple[int, int],
) -> np.ndarray:
    """Map one decoded normalized lane back to original CULane pixels."""
    points = np.asarray(lane.get("visible_points_norm", lane["points_norm"]), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Decoded lane points must have shape N x 2, got {points.shape}.")
    raw_height, raw_width = int(raw_shape[0]), int(raw_shape[1])
    return points * np.asarray([raw_width, raw_height], dtype=np.float64)


def match_culane_lanes(
    pred_lanes: list[dict],
    gt_lanes: list[list[tuple[float, float]]],
    raw_shape: tuple[int, int],
    lane_width: int = CULANE_LANE_WIDTH,
    iou_threshold: float = CULANE_IOU_THRESHOLD,
) -> tuple[dict, list[dict]]:
    """Match lane regions with the official CULane IoU/Hungarian protocol."""
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    pred_points = [prediction_lane_points_raw(lane, raw_shape=raw_shape) for lane in pred_lanes]
    pred_masks = [
        culane_lane_mask(pred_lane, image_shape=raw_shape, lane_width=lane_width)
        for pred_lane in pred_points
    ]
    gt_masks = [
        culane_lane_mask(gt_lane, image_shape=raw_shape, lane_width=lane_width)
        for gt_lane in gt_lanes
    ]
    pred_binary = [mask > 0 for mask in pred_masks]
    gt_binary = [mask > 0 for mask in gt_masks]
    ious = np.zeros((len(pred_points), len(gt_lanes)), dtype=np.float64)
    for pred_index, pred_mask in enumerate(pred_binary):
        pred_area = int(pred_mask.sum())
        for gt_index, gt_mask in enumerate(gt_binary):
            intersection = int(np.logical_and(pred_mask, gt_mask).sum())
            union = pred_area + int(gt_mask.sum()) - intersection
            ious[pred_index, gt_index] = float(intersection / union) if union else 0.0

    if ious.size:
        pred_indices, gt_indices = linear_sum_assignment(-ious)
    else:
        pred_indices = np.zeros((0,), dtype=np.int64)
        gt_indices = np.zeros((0,), dtype=np.int64)

    matched: list[dict] = []
    tp = 0
    iou_tp: list[float] = []
    iou_matched_all: list[float] = []
    iou_fp_matched: list[float] = []
    for pred_index, gt_index in zip(pred_indices.tolist(), gt_indices.tolist()):
        value = float(ious[pred_index, gt_index])
        is_tp = value > float(iou_threshold)
        tp += int(is_tp)
        iou_matched_all.append(value)
        if is_tp:
            iou_tp.append(value)
        else:
            iou_fp_matched.append(value)
        matched.append(
            {
                "pred": int(pred_index),
                "gt": int(gt_index),
                "culane_iou": round(value, 6),
                "tp": bool(is_tp),
            }
        )

    metrics = {
        "metric_name": "culane_iou",
        "metric_threshold": float(iou_threshold),
        "tp": int(tp),
        "fp": int(len(pred_lanes) - tp),
        "fn": int(len(gt_lanes) - tp),
        "culane_iou_tp": iou_tp,
        "culane_iou_matched_all": iou_matched_all,
        "culane_iou_fp_matched": iou_fp_matched,
        "strict_match_count": len(matched),
        "diagnostic_match_count": len(matched),
        "diagnostic_matches": matched,
        "curvature_error": [],
    }
    return metrics, matched


def lane_ape_px(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, scale: np.ndarray) -> float:
    """Average point error in pixels for one predicted/GT lane pair."""
    mask = valid > 0.5
    if int(mask.sum()) < 2:
        return float("inf")
    return float(np.linalg.norm((pred[mask] - gt[mask]) * scale, axis=-1).mean())


def curvature_error_px(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, scale: np.ndarray) -> float | None:
    """Mean second-order curve error in pixels for one matched lane pair."""
    mask = (valid[2:] > 0.5) & (valid[1:-1] > 0.5) & (valid[:-2] > 0.5)
    if int(mask.sum()) < 1:
        return None
    pred_curve = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
    gt_curve = gt[2:] - 2.0 * gt[1:-1] + gt[:-2]
    return float(np.linalg.norm((pred_curve[mask] - gt_curve[mask]) * scale, axis=-1).mean())


def pair_geometry(
    pred: np.ndarray,
    gt_lanes: np.ndarray,
    gt_valid: np.ndarray,
    scale: np.ndarray,
    pred_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pairwise APE, mean x-distance, and overlap counts for predicted/GT lanes."""
    n_pred = int(pred.shape[0])
    n_gt = int(gt_lanes.shape[0])
    if n_pred == 0 or n_gt == 0:
        return (
            np.zeros((n_pred, n_gt), dtype=np.float32),
            np.zeros((n_pred, n_gt), dtype=np.float32),
            np.zeros((n_pred, n_gt), dtype=np.int32),
        )

    valid = (gt_valid > 0.5).astype(np.float32)
    if pred_valid is None:
        pred_valid = np.ones(pred.shape[:2], dtype=np.float32)
    pred_valid = (pred_valid > 0.5).astype(np.float32)
    if pred_valid.shape != pred.shape[:2]:
        raise ValueError(f"pred_valid shape {pred_valid.shape} must match pred point dims {pred.shape[:2]}.")
    overlap_mask = pred_valid[:, None, :] * valid[None]
    overlap_per_pair = overlap_mask.sum(axis=2).astype(np.int32)
    denom = np.maximum(overlap_per_pair.astype(np.float32), 1.0)
    diff_px = (pred[:, None] - gt_lanes[None]) * scale.reshape(1, 1, 1, 2)
    point_error = np.linalg.norm(diff_px, axis=-1)
    ape = (point_error * overlap_mask).sum(axis=2) / denom
    mean_x = (np.abs(diff_px[..., 0]) * overlap_mask).sum(axis=2) / denom
    ape = np.where(overlap_per_pair > 0, ape, np.inf)
    mean_x = np.where(overlap_per_pair > 0, mean_x, np.inf)
    overlap = overlap_per_pair.copy()
    return ape.astype(np.float32), mean_x.astype(np.float32), overlap


def gated_assignment(cost: np.ndarray, gate: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run Hungarian assignment and drop pairs that fail the finite/gate mask."""
    if cost.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    finite = np.isfinite(cost)
    if gate is not None:
        finite = finite & gate
    if not finite.any():
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    safe_cost = np.where(finite, cost, 1e9)
    rows, cols = linear_sum_assignment(safe_cost)
    keep = finite[rows, cols]
    return rows[keep].astype(np.int64), cols[keep].astype(np.int64)


def match_lanes(
    pred_lanes: list[dict],
    gt_lanes: np.ndarray,
    gt_valid: np.ndarray,
    image_shape: tuple[int, int],
    ape_thr: float,
    match_gate_px: float | None = None,
    max_x_dist: float = 0.0,
    min_overlap: int = 2,
) -> tuple[dict, list[dict]]:
    """Strictly match decoded predictions to GT lanes and compute per-image metrics.

    The strict assignment is gated before TP/FP/FN accounting. A separate raw
    Hungarian diagnostic is still reported so very bad forced pairs are visible
    as ape_matched_all/ape_fp_matched rather than hidden inside a single APE.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    scale = np.array([w, h], dtype=np.float32)
    pred = (
        np.stack([np.asarray(x["points_norm"], dtype=np.float32) for x in pred_lanes], axis=0)
        if pred_lanes
        else np.zeros((0, gt_lanes.shape[1] if gt_lanes.ndim == 3 else 0, 2), dtype=np.float32)
    )
    pred_valid = (
        np.stack(
            [
                np.asarray(x.get("point_valid", np.ones(np.asarray(x["points_norm"]).shape[0])), dtype=np.float32)
                for x in pred_lanes
            ],
            axis=0,
        )
        if pred_lanes
        else np.zeros((0, gt_lanes.shape[1] if gt_lanes.ndim == 3 else 0), dtype=np.float32)
    )
    n_pred = int(pred.shape[0])
    n_gt = int(gt_lanes.shape[0])

    if n_pred == 0 or n_gt == 0:
        return {
            "metric_name": "ape",
            "metric_threshold": float(ape_thr),
            "tp": 0,
            "fp": n_pred,
            "fn": n_gt,
            "ape_tp": [],
            "ape_matched_all": [],
            "ape_fp_matched": [],
            "ape": [],
            "curvature_error": [],
            "strict_match_count": 0,
            "diagnostic_match_count": 0,
            "diagnostic_matches": [],
        }, []

    ape, mean_x, overlap = pair_geometry(pred, gt_lanes, gt_valid, scale, pred_valid=pred_valid)
    diagnostic_rows, diagnostic_cols = gated_assignment(ape)
    gate_px = float(ape_thr) if match_gate_px is None else float(match_gate_px)
    gate = overlap >= max(int(min_overlap), 0)
    if max_x_dist and max_x_dist > 0.0:
        gate = gate & (mean_x <= float(max_x_dist))
    if gate_px > 0.0:
        gate = gate & (ape <= gate_px)
    rows, cols = gated_assignment(ape, gate=gate)

    matched = []
    diagnostic_matches = []
    matched_tp_ape = []
    matched_all_ape = []
    matched_fp_ape = []
    curve_errors = []
    strict_pairs = {(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())}
    for row, col in zip(diagnostic_rows, diagnostic_cols):
        value = float(ape[row, col])
        matched_all_ape.append(value)
        gate_ok = bool(gate[row, col])
        is_tp = bool((int(row), int(col)) in strict_pairs and value < float(ape_thr))
        if not is_tp:
            matched_fp_ape.append(value)
        diagnostic_matches.append(
            {
                "pred": int(row),
                "gt": int(col),
                "ape_px": round(value, 4),
                "mean_x_dist_px": round(float(mean_x[row, col]), 4),
                "overlap_points": int(overlap[row, col]),
                "gate_ok": gate_ok,
                "tp": is_tp,
            }
        )

    for row, col in zip(rows, cols):
        value = float(ape[row, col])
        is_good = value < float(ape_thr)
        curve = curvature_error_px(pred[row], gt_lanes[col], gt_valid[col], scale)
        if is_good:
            matched_tp_ape.append(value)
        if curve is not None and is_good:
            curve_errors.append(curve)
        matched.append(
            {
                "pred": int(row),
                "gt": int(col),
                "ape_px": round(value, 4),
                "mean_x_dist_px": round(float(mean_x[row, col]), 4),
                "overlap_points": int(overlap[row, col]),
                "curvature_error_px": None if curve is None else round(float(curve), 4),
                "tp": bool(is_good),
            }
        )
    tp = len(matched_tp_ape)

    return {
        "metric_name": "ape",
        "metric_threshold": float(ape_thr),
        "tp": tp,
        "fp": n_pred - tp,
        "fn": n_gt - tp,
        "ape_tp": matched_tp_ape,
        "ape_matched_all": matched_all_ape,
        "ape_fp_matched": matched_fp_ape,
        "ape": matched_tp_ape,
        "curvature_error": curve_errors,
        "strict_match_count": len(matched),
        "diagnostic_match_count": len(diagnostic_matches),
        "diagnostic_matches": diagnostic_matches,
    }, matched


def stat_mean(values: list[float]) -> float | None:
    """Return rounded mean or None for an empty sequence."""
    return None if not values else round(float(np.mean(values)), 4)


def stat_median(values: list[float]) -> float | None:
    """Return rounded median or None for an empty sequence."""
    return None if not values else round(float(np.median(values)), 4)


def stat_max(values: list[float]) -> float | None:
    """Return rounded max or None for an empty sequence."""
    return None if not values else round(float(np.max(values)), 4)


def stat_min(values: list[float]) -> float | None:
    """Return rounded min or None for an empty sequence."""
    return None if not values else round(float(np.min(values)), 4)


def summarize(
    records: list[dict],
    total_infer: float,
    total_post: float,
    ape_thr: float,
    metric_name: str = "ape",
    metric_threshold: float | None = None,
) -> dict:
    """Aggregate per-image GCS metrics."""
    tp = sum(int(x["metrics"]["tp"]) for x in records)
    fp = sum(int(x["metrics"]["fp"]) for x in records)
    fn = sum(int(x["metrics"]["fn"]) for x in records)
    ape_tp = [float(v) for x in records for v in x["metrics"].get("ape_tp", x["metrics"].get("ape", []))]
    ape_matched_all = [float(v) for x in records for v in x["metrics"].get("ape_matched_all", [])]
    ape_fp_matched = [float(v) for x in records for v in x["metrics"].get("ape_fp_matched", [])]
    curve = [float(v) for x in records for v in x["metrics"].get("curvature_error", [])]
    iou_tp = [float(v) for x in records for v in x["metrics"].get("culane_iou_tp", [])]
    iou_matched_all = [float(v) for x in records for v in x["metrics"].get("culane_iou_matched_all", [])]
    iou_fp_matched = [float(v) for x in records for v in x["metrics"].get("culane_iou_fp_matched", [])]
    pred_counts = [int(x["pred_lanes"]) for x in records]
    gt_counts = [int(x["gt_lanes"]) for x in records]
    lane_count_abs_error = sum(abs(p - g) for p, g in zip(pred_counts, gt_counts))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    n = max(len(records), 1)
    total_time = total_infer + total_post
    infer_ms = np.asarray([float(x["inference_ms"]) for x in records], dtype=np.float32)
    post_ms = np.asarray([float(x["postprocess_ms"]) for x in records], dtype=np.float32)

    def timing_stats(values: np.ndarray, prefix: str) -> dict[str, float | None]:
        if values.size == 0:
            return {
                f"{prefix}_p50_ms": None,
                f"{prefix}_p95_ms": None,
                f"{prefix}_p99_ms": None,
                f"{prefix}_max_ms": None,
            }
        return {
            f"{prefix}_p50_ms": round(float(np.percentile(values, 50)), 4),
            f"{prefix}_p95_ms": round(float(np.percentile(values, 95)), 4),
            f"{prefix}_p99_ms": round(float(np.percentile(values, 99)), 4),
            f"{prefix}_max_ms": round(float(np.max(values)), 4),
        }

    summary = {
        "images": len(records),
        "metric_name": str(metric_name),
        "metric_threshold": None if metric_threshold is None else float(metric_threshold),
        "ape_threshold_px": float(ape_thr),
        "ape_mean_px": stat_mean(ape_tp),
        "ape_median_px": stat_median(ape_tp),
        "ape_min_px": stat_min(ape_tp),
        "ape_max_px": stat_max(ape_tp),
        "ape_tp_mean_px": stat_mean(ape_tp),
        "ape_tp_median_px": stat_median(ape_tp),
        "ape_tp_max_px": stat_max(ape_tp),
        "ape_matched_all_mean_px": stat_mean(ape_matched_all),
        "ape_matched_all_median_px": stat_median(ape_matched_all),
        "ape_matched_all_max_px": stat_max(ape_matched_all),
        "ape_all_matched_mean_px": stat_mean(ape_matched_all),
        "ape_all_matched_median_px": stat_median(ape_matched_all),
        "ape_all_matched_max_px": stat_max(ape_matched_all),
        "ape_fp_matched_mean_px": stat_mean(ape_fp_matched),
        "ape_fp_matched_median_px": stat_median(ape_fp_matched),
        "ape_fp_matched_max_px": stat_max(ape_fp_matched),
        "fp_matched_ape_mean_px": stat_mean(ape_fp_matched),
        "fp_matched_ape_median_px": stat_median(ape_fp_matched),
        "fp_matched_ape_max_px": stat_max(ape_fp_matched),
        "culane_iou_tp_mean": stat_mean(iou_tp),
        "culane_iou_tp_median": stat_median(iou_tp),
        "culane_iou_tp_min": stat_min(iou_tp),
        "culane_iou_tp_max": stat_max(iou_tp),
        "culane_iou_matched_all_mean": stat_mean(iou_matched_all),
        "culane_iou_matched_all_median": stat_median(iou_matched_all),
        "culane_iou_matched_all_min": stat_min(iou_matched_all),
        "culane_iou_matched_all_max": stat_max(iou_matched_all),
        "culane_iou_fp_matched_mean": stat_mean(iou_fp_matched),
        "curvature_error_mean_px": None if not curve else round(float(np.mean(curve)), 4),
        "curvature_error_median_px": None if not curve else round(float(np.median(curve)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "fp_per_image": round(float(fp) / n, 6),
        "fn_per_image": round(float(fn) / n, 6),
        "lane_count_mae": round(float(lane_count_abs_error) / n, 6),
        "pred_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(pred_counts).items())},
        "gt_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(gt_counts).items())},
        "gt_pred_lanes_hist": {
            f"{gt}->{pred}": int(count)
            for (gt, pred), count in sorted(Counter(zip(gt_counts, pred_counts)).items())
        },
        "avg_inference_ms": round(total_infer * 1000.0 / n, 4),
        "avg_postprocess_ms": round(total_post * 1000.0 / n, 4),
        "fps_infer_post": round(n / max(total_time, 1e-9), 4),
    }
    summary.update(timing_stats(infer_ms, "inference"))
    summary.update(timing_stats(post_ms, "postprocess"))
    return summary


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_eval_config(
    *,
    weights: str | Path,
    source: str | Path,
    label_dir: Path | None,
    imgsz: tuple[int, int],
    decode_mode: str,
    conf: float,
    point_valid_thr: float,
    ape_thr: float,
    match_gate_px: float | None,
    max_x_dist: float,
    min_overlap: int,
    nms_dist_px: float,
    max_det: int,
    count_aware_topk: bool,
    count_aware_min_k: int,
    count_aware_max_k: int,
    count_aware_length_norm: float,
    warmup: int,
    device: str,
    half: bool,
    gcs_min_lanes: int = 2,
    gcs_max_lanes: int = 5,
    gcs_num_slots: int = 5,
    gcs_min_interval_points: int = 2,
    gcs_bottom_order_margin_px: float = 2.0,
    metric: str = "ape",
    culane_archive_root: str | Path | None = None,
    culane_raw_shape: tuple[int, int] = CULANE_RAW_SHAPE,
    culane_lane_width: int = CULANE_LANE_WIDTH,
    culane_iou_thr: float = CULANE_IOU_THRESHOLD,
) -> dict:
    """Build the eval_summary.json config block for query or ordered-slot decode."""
    config = {
        "weights": str(Path(weights).resolve()) if not isinstance(weights, Path) else str(weights.resolve()),
        "source": str(Path(source).resolve()),
        "labels": None if label_dir is None else str(label_dir.resolve()),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "decode_mode": str(decode_mode),
        "metric_name": str(metric),
        "ape_threshold_px": float(ape_thr),
        "match_gate_px": float(ape_thr if match_gate_px is None else match_gate_px),
        "max_x_dist": float(max_x_dist),
        "min_overlap": int(min_overlap),
        "warmup": int(warmup),
        "device": str(device),
        "half": bool(half),
    }
    if str(metric) == "culane_iou":
        config.update(
            {
                "culane_archive_root": None
                if culane_archive_root is None
                else str(Path(culane_archive_root).expanduser().resolve()),
                "culane_raw_shape": [int(culane_raw_shape[0]), int(culane_raw_shape[1])],
                "culane_lane_width": int(culane_lane_width),
                "culane_iou_threshold": float(culane_iou_thr),
            }
        )
    if decode_mode == "ordered_slot":
        runtime_cfg = ordered_slot_decode_runtime_config(context="eval_gcs")
        ordered_params = ordered_slot_decode_params(
            {
                "gcs_min_lanes": gcs_min_lanes,
                "gcs_max_lanes": gcs_max_lanes,
                "gcs_num_slots": gcs_num_slots,
                "gcs_min_interval_points": gcs_min_interval_points,
                "gcs_bottom_order_margin_px": gcs_bottom_order_margin_px,
            }
        )
        effective_decode = build_ordered_slot_decode_summary(
            min_lanes=ordered_params["min_lanes"],
            max_lanes=ordered_params["max_lanes"],
            num_slots=ordered_params["num_slots"],
            min_interval_points=ordered_params["min_interval_points"],
            order_margin_px=ordered_params["order_margin_px"],
            output_order=runtime_cfg["output_order"],
            order_check=runtime_cfg["order_check"],
        )
        config.update(effective_decode)
        config["effective_decode"] = effective_decode
        return config
    if decode_mode != "query":
        raise RuntimeError(f"Unsupported eval_gcs decode_mode={decode_mode!r}.")
    config.update(
        {
            "conf": float(conf),
            "point_valid_thr": float(point_valid_thr),
            "nms_dist_px": float(nms_dist_px),
            "max_det": int(max_det),
            "count_aware_topk": bool(count_aware_topk),
            "count_aware_min_k": int(count_aware_min_k),
            "count_aware_max_k": int(count_aware_max_k),
            "count_aware_length_norm": float(count_aware_length_norm),
        }
    )
    return config


@torch.inference_mode()
def evaluate(
    weights: str | Path,
    source: str | Path,
    labels: str | Path | None = DEFAULT_LABELS,
    imgsz: int | tuple[int, int] | list[int] = (544, 960),
    conf: float = 0.2,
    point_valid_thr: float = 0.5,
    ape_thr: float = 20.0,
    match_gate_px: float | None = None,
    max_x_dist: float = 0.0,
    min_overlap: int = 2,
    nms_dist_px: float = 50.0,
    max_det: int = 8,
    count_aware_topk: bool = False,
    count_aware_min_k: int = 3,
    count_aware_max_k: int = 5,
    count_aware_length_norm: float = 12.0,
    gcs_min_lanes: int = 2,
    gcs_max_lanes: int = 5,
    gcs_num_slots: int = 5,
    gcs_min_interval_points: int = 2,
    gcs_bottom_order_margin_px: float = 2.0,
    decode_mode: str = "auto",
    max_images: int = 0,
    warmup: int = 20,
    device: str = "0",
    half: bool = False,
    save_dir: str | Path = "runs/gcs_lane/eval",
    save_json: bool | str | Path = False,
    save_img: bool = False,
    save_txt: bool = False,
    line_width: int = 2,
    metric: str = "ape",
    culane_archive_root: str | Path | None = DEFAULT_CULANE_ARCHIVE_ROOT,
    culane_raw_shape: tuple[int, int] = CULANE_RAW_SHAPE,
    culane_lane_width: int = CULANE_LANE_WIDTH,
    culane_iou_thr: float = CULANE_IOU_THRESHOLD,
) -> dict:
    """Evaluate a GCS-YOLO-Lane checkpoint on image/labels_gcs pairs."""
    metric = str(metric).lower()
    if metric not in {"ape", "culane_iou"}:
        raise ValueError(f"Unsupported evaluation metric: {metric!r}")
    if metric == "culane_iou":
        if culane_archive_root is None:
            raise ValueError("--culane-archive-root is required for metric='culane_iou'.")
        if tuple(int(x) for x in culane_raw_shape) != CULANE_RAW_SHAPE:
            raise ValueError(
                "CULane official evaluation expects raw H,W=(590, 1640); "
                f"got {tuple(culane_raw_shape)}."
            )
        if int(culane_lane_width) <= 0:
            raise ValueError(f"--culane-lane-width must be positive, got {culane_lane_width}")
        if not 0.0 <= float(culane_iou_thr) <= 1.0:
            raise ValueError(f"--culane-iou-thr must be in [0, 1], got {culane_iou_thr}")
    imgsz = normalize_imgsz(imgsz)
    device_obj = select_device(device, verbose=False)
    model = load_gcs_model(weights, device=device_obj, half=half, gcs_imgsz=imgsz)
    active_decode_mode = resolve_decode_mode(decode_mode, model)
    images = collect_images(source, max_images=max_images)
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")

    save_dir = Path(save_dir)
    image_dir = save_dir / "images"
    label_out_dir = save_dir / "labels"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json_path = None
    if isinstance(save_json, (str, Path)) and str(save_json).strip().lower() not in {"", "false", "none", "0"}:
        save_json_path = Path(save_json)
        if not save_json_path.is_absolute():
            save_json_path = (ROOT / save_json_path).resolve()
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
    if save_img:
        image_dir.mkdir(parents=True, exist_ok=True)
    if save_txt:
        label_out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_infer = 0.0
    total_post = 0.0
    label_dir = None if labels is None or str(labels).strip() == "" else Path(labels)

    if warmup > 0 and images:
        warm_img = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {images[0]}")
        warm_tensor = preprocess_image(warm_img, imgsz=imgsz, device=device_obj, half=half)
        for _ in range(int(warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    ordered_params = (
        ordered_slot_decode_params(
            {
                "gcs_min_lanes": gcs_min_lanes,
                "gcs_max_lanes": gcs_max_lanes,
                "gcs_num_slots": gcs_num_slots,
                "gcs_min_interval_points": gcs_min_interval_points,
                "gcs_bottom_order_margin_px": gcs_bottom_order_margin_px,
            }
        )
        if active_decode_mode == "ordered_slot"
        else None
    )

    for image_path in images:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        source_shape = DATASET_IMAGE_SHAPES["culane"] if metric == "culane_iou" else imgsz
        assert_gcs_shape(
            img.shape[:2],
            source_shape,
            name="evaluation image",
            context=f"eval_gcs.evaluate({image_path})",
        )
        label_path = label_path_for_image(image_path, label_dir)
        gt_lanes, gt_valid = load_gcs_label(label_path)
        raw_gt_lanes = (
            load_culane_ground_truth(
                label_path,
                archive_root=culane_archive_root,
                raw_shape=tuple(int(x) for x in culane_raw_shape),
            )
            if metric == "culane_iou"
            else None
        )

        tensor = preprocess_image(img, imgsz=imgsz, device=device_obj, half=half)
        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        infer_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if active_decode_mode == "ordered_slot":
            runtime_cfg = ordered_slot_decode_runtime_config(context="eval_gcs")
            lanes = decode_ordered_slot_predictions(
                preds,
                batch_index=0,
                image_shape=img.shape[:2],
                min_lanes=ordered_params["min_lanes"],
                max_lanes=ordered_params["max_lanes"],
                min_interval_points=ordered_params["min_interval_points"],
                order_margin_px=ordered_params["order_margin_px"],
                img_w=float(img.shape[1]),
                order_check=runtime_cfg["order_check"],
                output_order=runtime_cfg["output_order"],
            )
        else:
            pred_valid = preds.get("pred_valid_logits")
            lanes = decode_gcs_predictions(
                preds["pred_points"][0],
                preds["pred_logits"][0],
                pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
                image_shape=img.shape[:2],
                score_thr=conf,
                point_valid_thr=point_valid_thr,
                max_det=max_det,
                nms_dist_px=nms_dist_px,
                count_aware_topk=count_aware_topk,
                count_aware_min_k=count_aware_min_k,
                count_aware_max_k=count_aware_max_k,
                count_aware_length_norm=count_aware_length_norm,
            )
        if metric == "culane_iou":
            metrics, matches = match_culane_lanes(
                lanes,
                raw_gt_lanes or [],
                raw_shape=tuple(int(x) for x in culane_raw_shape),
                lane_width=culane_lane_width,
                iou_threshold=culane_iou_thr,
            )
        else:
            metrics, matches = match_lanes(
                lanes,
                gt_lanes,
                gt_valid,
                img.shape[:2],
                ape_thr=ape_thr,
                match_gate_px=match_gate_px,
                max_x_dist=max_x_dist,
                min_overlap=min_overlap,
            )
        post_s = time.perf_counter() - t1
        total_infer += infer_s
        total_post += post_s

        if save_img:
            cv2.imwrite(str(image_dir / image_path.name), draw_gcs_lanes(img, lanes, line_width=line_width))
        if save_txt:
            save_gcs_lanes_txt(label_out_dir / f"{image_path.stem}.txt", lanes, save_conf=True)

        records.append(
            {
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
                "height": int(img.shape[0]),
                "width": int(img.shape[1]),
                "pred_lanes": len(lanes),
                "gt_lanes": int(len(raw_gt_lanes) if metric == "culane_iou" else gt_lanes.shape[0]),
                "inference_ms": round(infer_s * 1000.0, 4),
                "postprocess_ms": round(post_s * 1000.0, 4),
                "metrics": metrics,
                "matches": matches,
            }
        )

    summary = summarize(
        records,
        total_infer=total_infer,
        total_post=total_post,
        ape_thr=ape_thr,
        metric_name=metric,
        metric_threshold=culane_iou_thr if metric == "culane_iou" else ape_thr,
    )
    output = {
        "summary": summary,
        "config": build_eval_config(
            weights=weights,
            source=source,
            label_dir=label_dir,
            imgsz=imgsz,
            decode_mode=str(active_decode_mode),
            conf=conf,
            point_valid_thr=point_valid_thr,
            ape_thr=ape_thr,
            match_gate_px=match_gate_px,
            max_x_dist=max_x_dist,
            min_overlap=min_overlap,
            nms_dist_px=nms_dist_px,
            max_det=max_det,
            count_aware_topk=count_aware_topk,
            count_aware_min_k=count_aware_min_k,
            count_aware_max_k=count_aware_max_k,
            count_aware_length_norm=count_aware_length_norm,
            gcs_min_lanes=gcs_min_lanes,
            gcs_max_lanes=gcs_max_lanes,
            gcs_num_slots=gcs_num_slots,
            gcs_min_interval_points=gcs_min_interval_points,
            gcs_bottom_order_margin_px=gcs_bottom_order_margin_px,
            warmup=warmup,
            device=device,
            half=half,
            metric=metric,
            culane_archive_root=culane_archive_root,
            culane_raw_shape=tuple(int(x) for x in culane_raw_shape),
            culane_lane_width=culane_lane_width,
            culane_iou_thr=culane_iou_thr,
        ),
    }
    if active_decode_mode == "ordered_slot":
        output["effective_decode"] = output["config"]["effective_decode"]
        output["query_decode_args"] = "not_applicable"
    if save_json:
        output["records"] = records
    (save_dir / "eval_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    if save_json_path is not None:
        save_json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved to: {save_dir.resolve()}")
    if save_json_path is not None:
        print(f"json saved to: {save_json_path}")
    return output


def main() -> None:
    args = parse_args()
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    defaults = dataset_defaults(args.dataset)
    evaluate(
        weights=args.weights,
        source=args.source or defaults["source"],
        labels=args.labels or defaults["labels"],
        imgsz=imgsz,
        conf=args.conf,
        point_valid_thr=args.point_valid_thr,
        ape_thr=args.ape_thr,
        match_gate_px=args.match_gate_px,
        max_x_dist=args.max_x_dist,
        min_overlap=args.min_overlap,
        nms_dist_px=args.nms_dist_px,
        max_det=args.max_det,
        count_aware_topk=args.count_aware_topk,
        count_aware_min_k=args.count_aware_min_k,
        count_aware_max_k=args.count_aware_max_k,
        count_aware_length_norm=args.count_aware_length_norm,
        gcs_min_lanes=args.gcs_min_lanes,
        gcs_max_lanes=args.gcs_max_lanes,
        gcs_num_slots=args.gcs_num_slots,
        gcs_min_interval_points=args.gcs_min_interval_points,
        gcs_bottom_order_margin_px=args.gcs_bottom_order_margin_px,
        decode_mode=args.decode_mode,
        max_images=args.max_images,
        warmup=args.warmup,
        device=args.device,
        half=args.half,
        save_dir=args.save_dir,
        save_json=args.save_json,
        save_img=args.save_img,
        save_txt=args.save_txt,
        line_width=args.line_width,
        metric=args.metric,
        culane_archive_root=args.culane_archive_root,
        culane_raw_shape=tuple(args.culane_raw_imgsz),
        culane_lane_width=args.culane_lane_width,
        culane_iou_thr=args.culane_iou_thr,
    )


if __name__ == "__main__":
    main()
