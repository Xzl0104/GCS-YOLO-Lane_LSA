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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.infer_gcs import (
    collect_images,
    count_calibration_from_args,
    count_head_decode_kwargs_from_args,
    load_gcs_model,
    preprocess_image,
    warn_max_det_mismatch,
)
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules import GCSLaneHead
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, assert_gcs_shape, normalize_imgsz, shape_str
from ultralytics.utils.gcs_postprocess import (
    GCS_DEFAULT_MAX_DET,
    count_head_decode_meta,
    decode_gcs_predictions,
    draw_gcs_lanes,
    empty_decode_count_state,
    save_gcs_lanes_txt,
    summarize_decode_count_state,
    update_decode_count_state,
)
from ultralytics.utils.torch_utils import select_device


DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "overfit20" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "images" / "val"
DEFAULT_LABELS = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "labels_gcs" / "val"


def dataset_defaults(dataset: str, split: str = "val") -> dict[str, Path]:
    """Return conventional validation paths for a converted GCS dataset."""
    root = ROOT / "datasets" / ("tusimple_fixed_y_960x544" if dataset.lower() == "tusimple" else dataset.lower())
    return {
        "source": root / "images" / split,
        "labels": root / "labels_gcs" / split,
    }


def default_data_yaml(dataset: str) -> Path:
    """Prefer the fixed-y GCS data yaml used by current experiments when it exists."""
    candidates = [
        ROOT / "data" / f"{dataset}_gcs_fixed_y_960x544.yaml",
        ROOT / "data" / f"{dataset}_gcs_stratified_960x544.yaml",
        ROOT / "data" / f"{dataset}_gcs.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def resolve_dataset(data: str | Path | None, dataset: str) -> dict | None:
    """Resolve a GCS data yaml, returning None when no yaml is requested or available."""
    data_path = Path(data) if data else default_data_yaml(dataset)
    if not data_path.exists():
        if data:
            raise FileNotFoundError(f"GCS data yaml not found: {data_path}")
        return None
    resolved = check_det_dataset(str(data_path))
    resolved["yaml_file"] = str(data_path)
    return resolved


def labels_from_source(source: str | Path | None) -> Path | None:
    """Map an images/<split> directory to labels_gcs/<split> when possible."""
    if source is None:
        return None
    path = Path(source)
    parts = list(path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels_gcs"
        return Path(*parts)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GCS-YOLO-Lane structured lane predictions.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--data", default=None, help="GCS data yaml. Defaults to local fixed-y TuSimple yaml when present.")
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate when --source is not provided.",
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt or model yaml.")
    parser.add_argument("--source", default=None, help="Image file, image directory, or txt list. Overrides --data/--split.")
    parser.add_argument("--labels", default=None, help="labels_gcs directory. Empty means infer from image path.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS inference shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Lane existence confidence threshold.")
    parser.add_argument(
        "--point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold for fixed-y lane decoding.",
    )
    parser.add_argument("--ape-thr", type=float, default=20.0, help="APE threshold in pixels for TP matching.")
    parser.add_argument("--match-gate-px", type=float, default=None, help="Strict eval APE gate in pixels. Defaults to --ape-thr.")
    parser.add_argument("--max-x-dist", type=float, default=0.0, help="Optional strict eval mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--min-overlap", type=int, default=6, help="Minimum overlapping visible anchors required for eval matching.")
    parser.add_argument("--min-points", type=int, default=6, help="Minimum decoded visible anchors required to keep a predicted lane.")
    parser.add_argument(
        "--min-gt-cover-ratio",
        type=float,
        default=0.3,
        help="Minimum GT visible-anchor coverage ratio required for eval matching.",
    )
    parser.add_argument(
        "--min-pred-cover-ratio",
        type=float,
        default=0.3,
        help="Minimum predicted visible-anchor coverage ratio required for eval matching.",
    )
    parser.add_argument("--nms-dist-px", type=float, default=18.0, help="Lane duplicate suppression distance in pixels. 0 disables.")
    parser.add_argument("--max-det", type=int, default=GCS_DEFAULT_MAX_DET, help="Maximum decoded lane queries per image.")
    count_head_group = parser.add_mutually_exclusive_group()
    count_head_group.add_argument("--use-count-head-decode", dest="use_count_head_decode", action="store_true", help="Use explicit Count Head K for final Top-K lane selection.")
    count_head_group.add_argument("--no-count-head-decode", dest="use_count_head_decode", action="store_false", help="Disable Count Head K and use max-det rank selection.")
    parser.set_defaults(use_count_head_decode=True)
    parser.add_argument("--count-head-temp", type=float, default=1.0, help="Temperature for Count Head count=2/3/4/5 softmax.")
    parser.add_argument("--candidate-conf", type=float, default=0.05, help="Relaxed candidate-pool existence threshold for Count Head Top-K decode.")
    parser.add_argument("--candidate-point-valid-thr", type=float, default=0.20, help="Relaxed candidate-pool point-valid threshold for Count Head Top-K decode.")
    parser.add_argument("--candidate-min-points", type=int, default=5, help="Relaxed candidate-pool visible-anchor floor before final Top-K.")
    rescue_group = parser.add_mutually_exclusive_group()
    rescue_group.add_argument("--enable-rescue-candidate-pool", dest="enable_rescue_candidate_pool", action="store_true", help="Use weaker real-query candidates only when Count Head K exceeds the normal candidate pool.")
    rescue_group.add_argument("--no-enable-rescue-candidate-pool", dest="enable_rescue_candidate_pool", action="store_false", help="Disable the weaker rescue candidate pool.")
    parser.set_defaults(enable_rescue_candidate_pool=True)
    parser.add_argument("--rescue-candidate-conf", type=float, default=0.005, help="Rescue candidate-pool existence threshold for Count Head Top-K decode.")
    parser.add_argument("--rescue-candidate-point-valid-thr", type=float, default=0.08, help="Rescue candidate-pool point-valid threshold for Count Head Top-K decode.")
    parser.add_argument("--rescue-candidate-min-points", type=int, default=4, help="Rescue candidate-pool visible-anchor floor before final Top-K.")
    parser.add_argument("--final-min-points", type=int, default=6, help="Final visible-anchor floor for selected ranks 1-4.")
    parser.add_argument("--fifth-min-points", type=int, default=5, help="Final visible-anchor floor for selected rank 5.")
    parser.add_argument("--line-nms-min-overlap", type=int, default=6, help="Minimum shared visible anchors for lane-NMS duplicate suppression.")
    parser.add_argument("--line-nms-rescue-dist-px", type=float, default=30.0, help="Duplicate distance used when rescuing lanes from pre-NMS candidates.")
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument("--quality-rescue-5th", dest="quality_rescue_5th", action="store_true", help="Enable quality-gated fifth-lane rescue when pred_quality_logits are present.")
    quality_group.add_argument("--no-quality-rescue-5th", dest="quality_rescue_5th", action="store_false", help="Disable quality-gated fifth-lane rescue.")
    parser.set_defaults(quality_rescue_5th=True)
    parser.add_argument("--quality-rescue-count5-thr", type=float, default=0.70, help="Minimum Count Head P(count=5) for quality-gated fifth-lane rescue.")
    parser.add_argument("--quality-rescue-conf-thr", type=float, default=0.03, help="Minimum lane existence probability for quality-gated fifth-lane rescue.")
    parser.add_argument("--quality-rescue-mean-valid-thr", type=float, default=0.45, help="Minimum mean point-valid probability for quality-gated fifth-lane rescue.")
    parser.add_argument("--quality-rescue-quality-thr", type=float, default=0.55, help="Minimum Quality Head probability for quality-gated fifth-lane rescue.")
    parser.add_argument("--quality-rescue-min-points", type=int, default=5, help="Minimum visible anchors for quality-gated fifth-lane rescue.")
    parser.add_argument("--quality-rescue-dist-px", type=float, default=24.0, help="Minimum distance to existing lanes for quality-gated fifth-lane rescue.")
    parser.add_argument("--soft-count-decision", action="store_true", help="Choose K by candidate quality when Count Head probabilities are close.")
    parser.add_argument("--soft-count-prob-margin", type=float, default=0.08)
    parser.add_argument("--soft-count-quality-weight", type=float, default=1.0)
    parser.add_argument("--soft-count-prior-weight", type=float, default=0.5)
    parser.add_argument("--soft-count-duplicate-penalty", type=float, default=1.0)
    parser.add_argument("--soft-count-invalid-penalty", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images. 0 means all.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of untimed warmup forwards before benchmarking.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory for eval_summary.json and outputs. Defaults to a parameter-specific folder under the weight run.",
    )
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


def model_fixed_y_anchors(model: torch.nn.Module) -> np.ndarray | None:
    """Return fixed-y anchors from the model head, or None for free-point heads."""
    model = getattr(model, "module", model)
    for module in model.modules():
        if isinstance(module, GCSLaneHead):
            mode = str(getattr(module, "point_mode", "free")).lower()
            if mode in {"fixed-y", "fixedy"}:
                mode = "fixed_y"
            if mode != "fixed_y":
                return None
            if all(hasattr(module, name) for name in ("fixed_y_start", "fixed_y_end", "num_points")):
                return np.linspace(
                    float(module.fixed_y_start),
                    float(module.fixed_y_end),
                    int(module.num_points),
                    dtype=np.float32,
                ).reshape(-1)
            anchors = module.fixed_y_anchors.detach().float().cpu().numpy().astype(np.float32)
            return anchors.reshape(-1)
    return None


def label_fixed_y_anchors(label_path: Path) -> np.ndarray | None:
    """Return fixed-y anchors stored in one label file, or None for free-point labels."""
    with np.load(label_path, allow_pickle=False) as data:
        point_mode = str(np.asarray(data["point_mode"]).item()) if "point_mode" in data else "free"
        if point_mode.lower() in {"fixed-y", "fixedy"}:
            point_mode = "fixed_y"
        if point_mode.lower() != "fixed_y":
            return None
        if "fixed_y" in data:
            anchors = np.asarray(data["fixed_y"], dtype=np.float32).reshape(-1)
        elif "lanes" in data and data["lanes"].ndim == 3 and data["lanes"].shape[0] > 0:
            anchors = np.asarray(data["lanes"][0, :, 1], dtype=np.float32).reshape(-1)
        else:
            raise ValueError(f"{label_path}: fixed_y label is missing fixed_y anchors.")
    if anchors.size < 2:
        raise ValueError(f"{label_path}: fixed_y anchors must contain at least two points.")
    if not np.all(np.diff(anchors) < 0.0):
        raise ValueError(f"{label_path}: fixed_y anchors must be strictly descending from bottom to top.")
    if float(anchors.min()) < -1e-4 or float(anchors.max()) > 1.0 + 1e-4:
        raise ValueError(f"{label_path}: fixed_y anchors must be normalized to [0, 1].")
    return np.clip(anchors, 0.0, 1.0).astype(np.float32)


def assert_label_fixed_y_compatible(
    label_path: Path,
    expected_anchors: np.ndarray | None,
    tol: float = 5e-5,
    image_shape: tuple[int, int] | None = None,
    pixel_tol: float = 0.5,
) -> None:
    """Fail fast if a label file uses meaningfully different fixed-y anchors than the model.

    Old fixed-y labels may differ from checkpoint anchors by tiny normalized
    rounding amounts. Compare in pixel space when image_shape is available so a
    sub-pixel mismatch does not block evaluation/sweeps.
    """
    if expected_anchors is None:
        return
    anchors = label_fixed_y_anchors(label_path)
    if anchors is None:
        raise ValueError(f"{label_path}: model uses fixed_y anchors but label point_mode is free.")
    expected_anchors = np.asarray(expected_anchors, dtype=np.float32).reshape(-1)
    if anchors.shape != expected_anchors.shape:
        raise ValueError(
            f"{label_path}: fixed_y anchor shape mismatch, labels={anchors.shape}, model={expected_anchors.shape}."
        )
    max_err = float(np.max(np.abs(anchors - expected_anchors))) if anchors.size else 0.0
    eff_tol = float(tol)
    max_err_px = None
    eff_tol_px = None
    if image_shape is not None:
        img_h = max(float(image_shape[0]), 1.0)
        eff_tol = max(eff_tol, float(pixel_tol) / img_h)
        max_err_px = max_err * img_h
        eff_tol_px = eff_tol * img_h
    if max_err > eff_tol:
        px_msg = "" if max_err_px is None else f", max_err_px={max_err_px:.4f}, tol_px={eff_tol_px:.4f}"
        raise ValueError(
            f"{label_path}: fixed_y anchor mismatch, labels first/last=({anchors[0]:.9f}, {anchors[-1]:.9f}) "
            f"but model first/last=({expected_anchors[0]:.9f}, {expected_anchors[-1]:.9f}), "
            f"max_err={max_err:.6g}, tol={eff_tol:.6g}{px_msg}."
        )


def _fmt_float_for_path(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _count_calibration_tag(config: dict | None) -> str | None:
    """Deprecated: score-gap count calibration has been removed."""
    if not config:
        return None
    raise ValueError("Score-gap count calibration has been removed. Use Count Head Top-K decode instead.")


def _weight_run_dir(weights: str | Path) -> Path | None:
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def resolve_eval_save_dir(
    save_dir: str | Path | None,
    weights: str | Path,
    split: str | None,
    conf: float,
    point_valid_thr: float,
    nms_dist_px: float,
    max_det: int,
    max_images: int,
    min_points: int = 6,
    min_overlap: int = 6,
    min_gt_cover_ratio: float = 0.3,
    min_pred_cover_ratio: float = 0.3,
    count_calibration: dict | None = None,
    candidate_score_thr: float | None = None,
    candidate_point_valid_thr: float | None = None,
) -> Path:
    """Resolve a non-overlapping default eval output directory."""
    if save_dir is not None and str(save_dir).strip():
        return Path(save_dir)
    tag_parts = [
        f"eval_{split or 'custom'}",
        f"conf{_fmt_float_for_path(conf)}",
        f"pvalid{_fmt_float_for_path(point_valid_thr)}",
        f"candconf{_fmt_float_for_path(conf if candidate_score_thr is None else candidate_score_thr)}",
        f"candpvalid{_fmt_float_for_path(point_valid_thr if candidate_point_valid_thr is None else candidate_point_valid_thr)}",
        f"nms{_fmt_float_for_path(nms_dist_px)}",
        f"maxdet{int(max_det)}",
        f"minp{int(min_points)}",
        f"overlap{int(min_overlap)}",
    ]
    if min_gt_cover_ratio > 0.0 or min_pred_cover_ratio > 0.0:
        tag_parts.append(
            f"coverg{_fmt_float_for_path(min_gt_cover_ratio)}p{_fmt_float_for_path(min_pred_cover_ratio)}"
        )
    calibration_tag = _count_calibration_tag(count_calibration)
    if calibration_tag:
        tag_parts.append(calibration_tag)
    if max_images and max_images > 0:
        tag_parts.append(f"maximg{int(max_images)}")
    run_dir = _weight_run_dir(weights)
    if run_dir is not None:
        return run_dir / "eval_gcs" / "_".join(tag_parts)
    return ROOT / "runs" / "gcs_lane" / "eval" / Path(weights).stem / "_".join(tag_parts)


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
    min_overlap: int = 6,
    min_gt_cover_ratio: float = 0.3,
    min_pred_cover_ratio: float = 0.3,
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
    gt_visible = np.maximum((gt_valid > 0.5).sum(axis=1).astype(np.float32), 1.0)
    pred_visible = np.maximum((pred_valid > 0.5).sum(axis=1).astype(np.float32), 1.0)
    overlap_ratio = overlap.astype(np.float32) / gt_visible.reshape(1, -1)
    pred_cover_ratio = overlap.astype(np.float32) / pred_visible.reshape(-1, 1)
    if min_gt_cover_ratio > 0.0:
        gate = gate & (overlap_ratio >= float(min_gt_cover_ratio))
    if min_pred_cover_ratio > 0.0:
        gate = gate & (pred_cover_ratio >= float(min_pred_cover_ratio))
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
                "overlap_ratio": round(float(overlap_ratio[row, col]), 4),
                "pred_cover_ratio": round(float(pred_cover_ratio[row, col]), 4),
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
                "overlap_ratio": round(float(overlap_ratio[row, col]), 4),
                "pred_cover_ratio": round(float(pred_cover_ratio[row, col]), 4),
                "curvature_error_px": None if curve is None else round(float(curve), 4),
                "tp": bool(is_good),
            }
        )
    tp = len(matched_tp_ape)

    return {
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


def summarize(records: list[dict], total_infer: float, total_post: float, ape_thr: float) -> dict:
    """Aggregate per-image GCS metrics."""
    tp = sum(int(x["metrics"]["tp"]) for x in records)
    fp = sum(int(x["metrics"]["fp"]) for x in records)
    fn = sum(int(x["metrics"]["fn"]) for x in records)
    ape_tp = [float(v) for x in records for v in x["metrics"].get("ape_tp", x["metrics"].get("ape", []))]
    ape_matched_all = [float(v) for x in records for v in x["metrics"].get("ape_matched_all", [])]
    ape_fp_matched = [float(v) for x in records for v in x["metrics"].get("ape_fp_matched", [])]
    curve = [float(v) for x in records for v in x["metrics"]["curvature_error"]]
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


@torch.inference_mode()
def evaluate(
    weights: str | Path,
    source: str | Path,
    labels: str | Path | None = DEFAULT_LABELS,
    imgsz: int | tuple[int, int] | list[int] = (544, 960),
    data: str | Path | None = None,
    split: str | None = None,
    conf: float = 0.2,
    point_valid_thr: float = 0.5,
    ape_thr: float = 20.0,
    match_gate_px: float | None = None,
    max_x_dist: float = 0.0,
    min_overlap: int = 6,
    min_points: int = 6,
    min_gt_cover_ratio: float = 0.3,
    min_pred_cover_ratio: float = 0.3,
    nms_dist_px: float = 0.0,
    max_det: int = GCS_DEFAULT_MAX_DET,
    max_images: int = 0,
    warmup: int = 20,
    device: str = "0",
    half: bool = False,
    save_dir: str | Path | None = None,
    save_json: bool | str | Path = False,
    save_img: bool = False,
    save_txt: bool = False,
    line_width: int = 2,
    count_calibration: dict | None = None,
    use_count_head_decode: bool = True,
    count_head_temperature: float = 1.0,
    dataset_name: str = "tusimple",
    candidate_score_thr: float = 0.05,
    candidate_point_valid_thr: float = 0.20,
    candidate_min_points: int = 5,
    enable_rescue_candidate_pool: bool = True,
    rescue_candidate_score_thr: float = 0.005,
    rescue_candidate_point_valid_thr: float = 0.08,
    rescue_candidate_min_points: int = 4,
    final_min_points: int = 6,
    fifth_min_points: int = 5,
    line_nms_min_overlap: int = 6,
    line_nms_rescue_dist_px: float = 30.0,
    quality_rescue_5th: bool = True,
    quality_rescue_count5_thr: float = 0.70,
    quality_rescue_conf_thr: float = 0.03,
    quality_rescue_mean_valid_thr: float = 0.45,
    quality_rescue_quality_thr: float = 0.55,
    quality_rescue_min_points: int = 5,
    quality_rescue_dist_px: float = 24.0,
    enable_soft_count_decision: bool = False,
    soft_count_prob_margin: float = 0.08,
    soft_count_quality_weight: float = 1.0,
    soft_count_prior_weight: float = 0.5,
    soft_count_duplicate_penalty: float = 1.0,
    soft_count_invalid_penalty: float = 1.0,
) -> dict:
    """Evaluate a GCS-YOLO-Lane checkpoint on image/labels_gcs pairs."""
    imgsz = normalize_imgsz(imgsz)
    device_obj = select_device(device, verbose=False)
    model = load_gcs_model(weights, device=device_obj, half=half, gcs_imgsz=imgsz)
    expected_fixed_y = model_fixed_y_anchors(model)
    images = collect_images(source, max_images=max_images)
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")

    save_dir = resolve_eval_save_dir(
        save_dir=save_dir,
        weights=weights,
        split=split,
        conf=conf,
        point_valid_thr=point_valid_thr,
        nms_dist_px=nms_dist_px,
        max_det=max_det,
        max_images=max_images,
        min_points=min_points,
        min_overlap=min_overlap,
        min_gt_cover_ratio=min_gt_cover_ratio,
        min_pred_cover_ratio=min_pred_cover_ratio,
        count_calibration=count_calibration,
        candidate_score_thr=candidate_score_thr,
        candidate_point_valid_thr=candidate_point_valid_thr,
    )
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
    decode_count_state = empty_decode_count_state()
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

    for image_path in images:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        assert_gcs_shape(
            img.shape[:2],
            imgsz,
            name="evaluation image",
            context=f"eval_gcs.evaluate({image_path})",
        )
        label_path = label_path_for_image(image_path, label_dir)
        assert_label_fixed_y_compatible(label_path, expected_fixed_y, image_shape=img.shape[:2])
        gt_lanes, gt_valid = load_gcs_label(label_path)

        tensor = preprocess_image(img, imgsz=imgsz, device=device_obj, half=half)
        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        infer_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred_valid = preds.get("pred_valid_logits")
        pred_count = preds.get("pred_count_logits")
        pred_count_boundary = preds.get("pred_count_boundary_logits")
        pred_quality = preds.get("pred_quality_logits")
        count_meta = count_head_decode_meta(
            pred_count[0] if pred_count is not None else None,
            pred_count_boundary[0] if pred_count_boundary is not None else None,
            use_count_head_decode=use_count_head_decode,
            count_head_temperature=count_head_temperature,
            dataset_name=dataset_name,
            max_det=max_det,
        )
        lanes = decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
            pred_count_logits=pred_count[0] if pred_count is not None else None,
            pred_count_boundary_logits=pred_count_boundary[0] if pred_count_boundary is not None else None,
            pred_quality_logits=pred_quality[0] if pred_quality is not None else None,
            image_shape=img.shape[:2],
            score_thr=conf,
            point_valid_thr=point_valid_thr,
            min_points=min_points,
            max_det=max_det,
            nms_dist_px=nms_dist_px,
            count_calibration=count_calibration,
            use_count_head_decode=use_count_head_decode,
            count_head_temperature=count_head_temperature,
            dataset_name=dataset_name,
            candidate_score_thr=candidate_score_thr,
            candidate_point_valid_thr=candidate_point_valid_thr,
            candidate_min_points=candidate_min_points,
            enable_rescue_candidate_pool=enable_rescue_candidate_pool,
            rescue_candidate_score_thr=rescue_candidate_score_thr,
            rescue_candidate_point_valid_thr=rescue_candidate_point_valid_thr,
            rescue_candidate_min_points=rescue_candidate_min_points,
            final_min_points=final_min_points,
            fifth_min_points=fifth_min_points,
            line_nms_min_overlap=line_nms_min_overlap,
            line_nms_rescue_dist_px=line_nms_rescue_dist_px,
            quality_rescue_5th=quality_rescue_5th,
            quality_rescue_count5_thr=quality_rescue_count5_thr,
            quality_rescue_conf_thr=quality_rescue_conf_thr,
            quality_rescue_mean_valid_thr=quality_rescue_mean_valid_thr,
            quality_rescue_quality_thr=quality_rescue_quality_thr,
            quality_rescue_min_points=quality_rescue_min_points,
            quality_rescue_dist_px=quality_rescue_dist_px,
            enable_soft_count_decision=enable_soft_count_decision,
            soft_count_prob_margin=soft_count_prob_margin,
            soft_count_quality_weight=soft_count_quality_weight,
            soft_count_prior_weight=soft_count_prior_weight,
            soft_count_duplicate_penalty=soft_count_duplicate_penalty,
            soft_count_invalid_penalty=soft_count_invalid_penalty,
        )
        metrics, matches = match_lanes(
            lanes,
            gt_lanes,
            gt_valid,
            img.shape[:2],
            ape_thr=ape_thr,
            match_gate_px=match_gate_px,
            max_x_dist=max_x_dist,
            min_overlap=min_overlap,
            min_gt_cover_ratio=min_gt_cover_ratio,
            min_pred_cover_ratio=min_pred_cover_ratio,
        )
        post_s = time.perf_counter() - t1
        total_infer += infer_s
        total_post += post_s
        update_decode_count_state(decode_count_state, count_meta, len(lanes))

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
                "gt_lanes": int(gt_lanes.shape[0]),
                "decode_count_head_raw_count": None if count_meta is None else int(count_meta["count_head_raw_count"]),
                "decode_count_head_k": None if count_meta is None else int(count_meta["count_head_policy_count"]),
                "decode_count_head_margin": None if count_meta is None else round(float(count_meta["count_head_margin"]), 6),
                "decode_count_shortfall": None
                if count_meta is None
                else max(0, int(count_meta["count_head_policy_count"]) - len(lanes)),
                "inference_ms": round(infer_s * 1000.0, 4),
                "postprocess_ms": round(post_s * 1000.0, 4),
                "metrics": metrics,
                "matches": matches,
            }
        )

    summary = summarize(records, total_infer=total_infer, total_post=total_post, ape_thr=ape_thr)
    summary.update(summarize_decode_count_state(decode_count_state, prefix="decode/"))
    output = {
        "summary": summary,
        "config": {
            "weights": str(Path(weights).resolve()) if not isinstance(weights, Path) else str(weights.resolve()),
            "data": None if data is None else str(Path(data).resolve()),
            "split": split,
            "source": str(Path(source).resolve()),
            "labels": None if label_dir is None else str(label_dir.resolve()),
            "save_dir": str(save_dir.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(conf),
            "point_valid_thr": float(point_valid_thr),
            "ape_threshold_px": float(ape_thr),
            "match_gate_px": float(ape_thr if match_gate_px is None else match_gate_px),
            "max_x_dist": float(max_x_dist),
            "min_overlap": int(min_overlap),
            "min_points": int(min_points),
            "min_gt_cover_ratio": float(min_gt_cover_ratio),
            "min_pred_cover_ratio": float(min_pred_cover_ratio),
            "nms_dist_px": float(nms_dist_px),
            "max_det": int(max_det),
            "use_count_head_decode": bool(use_count_head_decode),
            "count_head_temperature": float(count_head_temperature),
            "candidate_score_thr": float(candidate_score_thr),
            "candidate_point_valid_thr": float(candidate_point_valid_thr),
            "candidate_min_points": int(candidate_min_points),
            "enable_rescue_candidate_pool": bool(enable_rescue_candidate_pool),
            "rescue_candidate_score_thr": float(rescue_candidate_score_thr),
            "rescue_candidate_point_valid_thr": float(rescue_candidate_point_valid_thr),
            "rescue_candidate_min_points": int(rescue_candidate_min_points),
            "final_min_points": int(final_min_points),
            "fifth_min_points": int(fifth_min_points),
            "line_nms_min_overlap": int(line_nms_min_overlap),
            "line_nms_rescue_dist_px": float(line_nms_rescue_dist_px),
            "quality_rescue_5th": bool(quality_rescue_5th),
            "quality_rescue_count5_thr": float(quality_rescue_count5_thr),
            "quality_rescue_conf_thr": float(quality_rescue_conf_thr),
            "quality_rescue_mean_valid_thr": float(quality_rescue_mean_valid_thr),
            "quality_rescue_quality_thr": float(quality_rescue_quality_thr),
            "quality_rescue_min_points": int(quality_rescue_min_points),
            "quality_rescue_dist_px": float(quality_rescue_dist_px),
            "soft_count_decision": bool(enable_soft_count_decision),
            "soft_count_prob_margin": float(soft_count_prob_margin),
            "soft_count_quality_weight": float(soft_count_quality_weight),
            "soft_count_prior_weight": float(soft_count_prior_weight),
            "soft_count_duplicate_penalty": float(soft_count_duplicate_penalty),
            "soft_count_invalid_penalty": float(soft_count_invalid_penalty),
            "warmup": int(warmup),
            "device": str(device),
            "half": bool(half),
        },
    }
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
    warn_max_det_mismatch(args.weights, max_det=args.max_det, context="GCS eval")
    data = resolve_dataset(args.data, args.dataset)
    imgsz = normalize_imgsz(
        args.imgsz or (data.get("gcs_imgsz") if data else None) or (data.get("image_shape") if data else None),
        dataset=args.dataset,
    )
    defaults = dataset_defaults(args.dataset, split=args.split)
    source = args.source or (data.get(args.split) if data else None) or defaults["source"]
    if source is None:
        raise ValueError(f"{args.split!r} source is required via --source or data yaml {args.split}.")
    labels = args.labels or labels_from_source(source) or defaults["labels"]
    evaluate(
        weights=args.weights,
        source=source,
        labels=labels,
        imgsz=imgsz,
        data=data.get("yaml_file") if data else None,
        split=args.split,
        conf=args.conf,
        point_valid_thr=args.point_valid_thr,
        ape_thr=args.ape_thr,
        match_gate_px=args.match_gate_px,
        max_x_dist=args.max_x_dist,
        min_overlap=args.min_overlap,
        min_points=args.min_points,
        min_gt_cover_ratio=args.min_gt_cover_ratio,
        min_pred_cover_ratio=args.min_pred_cover_ratio,
        nms_dist_px=args.nms_dist_px,
        max_det=args.max_det,
        max_images=args.max_images,
        warmup=args.warmup,
        device=args.device,
        half=args.half,
        save_dir=args.save_dir,
        save_json=args.save_json,
        save_img=args.save_img,
        save_txt=args.save_txt,
        line_width=args.line_width,
        count_calibration=count_calibration_from_args(args),
        **count_head_decode_kwargs_from_args(args, dataset_name=args.dataset),
    )


if __name__ == "__main__":
    main()
