from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.eval_gcs import (
    CULANE_IOU_THRESHOLD,
    CULANE_LANE_WIDTH,
    CULANE_RAW_SHAPE,
    culane_lane_mask,
    culane_lane_iou,
    culane_spline_interp_points,
    label_path_for_image,
    load_culane_ground_truth,
)
from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image
from ultralytics.utils.gcs_postprocess import lane_nms
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, assert_gcs_shape, normalize_imgsz
from ultralytics.utils.torch_utils import select_device


CACHE_VERSION = 1
WEAK_IOU_THRESHOLD = 0.10
MIN_POINTS = 2
_SWEEP_CACHE: dict[str, np.ndarray] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache one CULane validation forward pass, then sweep query decode parameters without repeat inference."
    )
    parser.add_argument("--weights", required=True, help="CULane query checkpoint to evaluate.")
    parser.add_argument("--source", required=True, help="CULane validation image directory or list.")
    parser.add_argument("--labels", required=True, help="CULane labels_gcs/val directory.")
    parser.add_argument("--culane-archive-root", required=True, help="Original CULane archive containing .lines.txt labels.")
    parser.add_argument("--imgsz", nargs=2, type=int, metavar=("H", "W"), default=(384, 960))
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true", help="Run the single cached model forward in FP16.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16, help="Images per single cached model forward batch.")
    parser.add_argument(
        "--iou-workers",
        type=int,
        default=0,
        help="Parallel CPU workers for cached CULane mask/IoU construction; 0 keeps sequential behavior.",
    )
    parser.add_argument(
        "--sweep-workers",
        type=int,
        default=0,
        help="Parallel CPU workers for cached decode configurations; 0 keeps sequential behavior.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Validation-only debug limit. 0 means all images.")
    parser.add_argument("--cache-dir", required=True, help="Directory for raw outputs and cached region-IoU tensors.")
    parser.add_argument("--save-dir", required=True, help="Directory for sweep rows and diagnosis reports.")
    parser.add_argument("--force-cache", action="store_true", help="Rebuild the raw forward/IoU cache even when compatible files exist.")
    parser.add_argument("--confs", nargs="+", type=float, default=[0.03, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--point-valid-thrs", nargs="+", type=float, default=[0.30, 0.40, 0.50])
    parser.add_argument("--nms-dist-pxs", nargs="+", type=float, default=[0.0, 18.0, 30.0, 50.0])
    parser.add_argument("--max-dets", nargs="+", type=int, default=[5, 6, 8])
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _path_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _cache_contract(args: argparse.Namespace, images: list[Path], imgsz: tuple[int, int]) -> dict[str, Any]:
    weights = Path(args.weights).expanduser().resolve()
    return {
        "cache_version": CACHE_VERSION,
        "weights": _path_signature(weights),
        "source": str(Path(args.source).expanduser().resolve()),
        "labels": str(Path(args.labels).expanduser().resolve()),
        "culane_archive_root": str(Path(args.culane_archive_root).expanduser().resolve()),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "half": bool(args.half),
        "min_points": MIN_POINTS,
        "point_valid_thrs": [float(value) for value in args.point_valid_thrs],
        "culane_raw_shape": list(CULANE_RAW_SHAPE),
        "culane_lane_width": CULANE_LANE_WIDTH,
        "culane_iou_threshold": CULANE_IOU_THRESHOLD,
        "max_images": int(args.max_images),
        "image_count": len(images),
        "first_image": str(images[0]) if images else "",
        "last_image": str(images[-1]) if images else "",
    }


def _sorted_points_and_valid(points: np.ndarray, valid_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match decoder point ordering: clamp then stable descending y sort per query."""
    points = np.clip(np.asarray(points, dtype=np.float32), 0.0, 1.0)
    valid_logits = np.asarray(valid_logits, dtype=np.float32)
    order = np.argsort(-points[..., 1], axis=1, kind="stable")
    sorted_points = np.take_along_axis(points, order[..., None], axis=1)
    sorted_valid_logits = np.take_along_axis(valid_logits, order, axis=1)
    return sorted_points, sorted_valid_logits


def _longest_contiguous_mask(mask: np.ndarray, min_points: int = MIN_POINTS) -> np.ndarray:
    """Numpy equivalent of decode_gcs_predictions longest-contiguous visibility filtering."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    best_start = 0
    best_len = 0
    start: int | None = None
    for index, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            length = index - start
            if length > best_len:
                best_start, best_len = start, length
            start = None
    out = np.zeros_like(mask, dtype=bool)
    if best_len >= int(min_points):
        out[best_start : best_start + best_len] = True
    return out


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values, dtype=np.float32))


def _lane_iou_for_points(points_norm: np.ndarray, gt_lane: list[tuple[float, float]]) -> float:
    raw_points = np.asarray(points_norm, dtype=np.float64) * np.asarray(
        [CULANE_RAW_SHAPE[1], CULANE_RAW_SHAPE[0]], dtype=np.float64
    )
    return culane_lane_iou(
        raw_points,
        gt_lane,
        image_shape=CULANE_RAW_SHAPE,
        lane_width=CULANE_LANE_WIDTH,
    )


def _lane_ious_against_masks(
    points_norm: np.ndarray,
    gt_masks: list[np.ndarray],
    gt_areas: np.ndarray,
) -> np.ndarray:
    """Rasterize one prediction once, then compare it with all GT masks."""
    raw_points = np.asarray(points_norm, dtype=np.float64) * np.asarray(
        [CULANE_RAW_SHAPE[1], CULANE_RAW_SHAPE[0]], dtype=np.float64
    )
    pred_mask = _fast_culane_lane_mask(
        raw_points,
        image_shape=CULANE_RAW_SHAPE,
        lane_width=CULANE_LANE_WIDTH,
    ) > 0
    pred_area = int(pred_mask.sum())
    values = np.zeros((len(gt_masks),), dtype=np.float32)
    for gt_index, gt_mask in enumerate(gt_masks):
        intersection = int(np.logical_and(pred_mask, gt_mask).sum())
        union = pred_area + int(gt_areas[gt_index]) - intersection
        values[gt_index] = float(intersection / union) if union else 0.0
    return values


def _fast_culane_lane_mask(
    lane: list[tuple[float, float]] | np.ndarray,
    image_shape: tuple[int, int],
    lane_width: int,
) -> np.ndarray:
    """Byte-for-byte equivalent to culane_lane_mask without Python segment loops."""
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(lane, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        return mask
    interpolated = culane_spline_interp_points(points.tolist(), samples_per_segment=50)
    if interpolated.shape[0] < 2:
        return mask
    cv2.polylines(
        mask,
        [np.rint(interpolated).astype(np.int32)],
        isClosed=False,
        color=1,
        thickness=int(lane_width),
        lineType=cv2.LINE_8,
    )
    return mask


def _gt_masks_for_lanes(
    gt_lanes: list[list[tuple[float, float]]],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build each image's GT masks once for the cached IoU calculations."""
    masks = [
        _fast_culane_lane_mask(
            np.asarray(gt_lane, dtype=np.float64),
            image_shape=CULANE_RAW_SHAPE,
            lane_width=CULANE_LANE_WIDTH,
        )
        > 0
        for gt_lane in gt_lanes
    ]
    areas = np.asarray([int(mask.sum()) for mask in masks], dtype=np.int32)
    return masks, areas


def _configure_iou_worker() -> None:
    """Avoid CPU thread oversubscription when several raster workers run together."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _build_image_iou(
    item: tuple[int, np.ndarray, np.ndarray, list[list[tuple[float, float]]], list[float]],
) -> tuple[int, np.ndarray, np.ndarray]:
    """Build all query/GT IoUs for one image from cached raw model outputs."""
    image_index, points_i, valid_i_np, gt_lanes, thresholds = item
    sorted_points, sorted_valid_logits = _sorted_points_and_valid(points_i, valid_i_np)
    valid_scores = _sigmoid(sorted_valid_logits)
    gt_masks, gt_areas = _gt_masks_for_lanes(gt_lanes)
    query_count = sorted_points.shape[0]
    gt_count = len(gt_lanes)
    full_iou = np.zeros((query_count, gt_count), dtype=np.float32)
    visible_iou = np.zeros((len(thresholds), query_count, gt_count), dtype=np.float32)
    for query_index in range(query_count):
        full_iou[query_index] = _lane_ious_against_masks(
            sorted_points[query_index],
            gt_masks,
            gt_areas,
        )
        for threshold_index, threshold in enumerate(thresholds):
            visible_mask = _longest_contiguous_mask(valid_scores[query_index] >= threshold)
            if int(visible_mask.sum()) < MIN_POINTS:
                continue
            visible_iou[threshold_index, query_index] = _lane_ious_against_masks(
                sorted_points[query_index, visible_mask],
                gt_masks,
                gt_areas,
            )
    return image_index, full_iou, visible_iou


def _load_raw_gt(images: list[Path], label_dir: Path, archive_root: Path) -> tuple[list[Path], list[list[list[tuple[float, float]]]]]:
    label_paths: list[Path] = []
    all_gt: list[list[list[tuple[float, float]]]] = []
    for image_path in tqdm(images, desc="Loading CULane val ground truth", unit="image"):
        label_path = label_path_for_image(image_path, label_dir)
        gt_lanes = load_culane_ground_truth(label_path, archive_root=archive_root, raw_shape=CULANE_RAW_SHAPE)
        label_paths.append(label_path)
        all_gt.append(gt_lanes)
    return label_paths, all_gt


def _build_cache(
    args: argparse.Namespace,
    cache_dir: Path,
    images: list[Path],
    label_paths: list[Path],
    all_gt: list[list[list[tuple[float, float]]]],
    imgsz: tuple[int, int],
    contract: dict[str, Any],
) -> dict[str, np.ndarray]:
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    max_gt = max((len(lanes) for lanes in all_gt), default=0)
    if max_gt <= 0:
        raise ValueError("CULane validation source has no ground-truth lanes.")

    if int(args.warmup) > 0 and images:
        warm_image = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if warm_image is None:
            raise FileNotFoundError(f"Failed to read warmup image: {images[0]}")
        warm_tensor = preprocess_image(warm_image, imgsz=imgsz, device=device, half=bool(args.half))
        warm_tensor = warm_tensor.repeat(int(min(args.batch_size, len(images))), 1, 1, 1)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device)

    pred_points: np.ndarray | None = None
    pred_logits: np.ndarray | None = None
    pred_valid_logits: np.ndarray | None = None
    full_iou: np.ndarray | None = None
    visible_iou: np.ndarray | None = None
    gt_counts = np.asarray([len(lanes) for lanes in all_gt], dtype=np.int16)
    inference_ms = np.zeros((len(images),), dtype=np.float32)
    thresholds = [float(value) for value in args.point_valid_thrs]

    batch_size = int(args.batch_size)
    batch_ranges = range(0, len(images), batch_size)
    for batch_start in tqdm(batch_ranges, total=(len(images) + batch_size - 1) // batch_size, desc="Caching one CULane val forward", unit="batch"):
        batch_paths = images[batch_start : batch_start + batch_size]
        batch_gt = all_gt[batch_start : batch_start + batch_size]
        tensors = []
        for image_path in batch_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")
            assert_gcs_shape(
                image.shape[:2],
                DATASET_IMAGE_SHAPES["culane"],
                name="CULane validation image",
                context=f"sweep_culane_cached({image_path})",
            )
            tensors.append(preprocess_image(image, imgsz=imgsz, device=device, half=bool(args.half)))
        tensor = torch.cat(tensors, dim=0)
        _sync_if_cuda(device)
        start = time.perf_counter()
        outputs = model(tensor)
        _sync_if_cuda(device)
        batch_inference_ms = float((time.perf_counter() - start) * 1000.0)
        per_image_inference_ms = batch_inference_ms / max(len(batch_paths), 1)

        points_batch = outputs["pred_points"].detach().float().cpu().numpy().astype(np.float32)
        logits_batch = outputs["pred_logits"].detach().float().cpu().numpy().astype(np.float32)
        valid_batch = outputs.get("pred_valid_logits")
        if valid_batch is None:
            valid_batch_np = np.full(points_batch.shape[:3], 20.0, dtype=np.float32)
        else:
            valid_batch_np = valid_batch.detach().float().cpu().numpy().astype(np.float32)

        if pred_points is None:
            query_count, point_count = points_batch.shape[1:3]
            pred_points = np.empty((len(images), query_count, point_count, 2), dtype=np.float32)
            pred_logits = np.empty((len(images), query_count), dtype=np.float32)
            pred_valid_logits = np.empty((len(images), query_count, point_count), dtype=np.float32)
            full_iou = np.zeros((len(images), query_count, max_gt), dtype=np.float32)
            visible_iou = np.zeros((len(images), len(thresholds), query_count, max_gt), dtype=np.float32)

        assert pred_points is not None and pred_logits is not None and pred_valid_logits is not None
        for local_index, gt_lanes in enumerate(batch_gt):
            image_index = batch_start + local_index
            points_i = points_batch[local_index]
            logits_i = logits_batch[local_index]
            valid_i_np = valid_batch_np[local_index]
            inference_ms[image_index] = per_image_inference_ms
            pred_points[image_index] = points_i
            pred_logits[image_index] = logits_i
            pred_valid_logits[image_index] = valid_i_np

    assert pred_points is not None and pred_logits is not None and pred_valid_logits is not None
    assert full_iou is not None and visible_iou is not None
    iou_items = (
        (
            image_index,
            pred_points[image_index],
            pred_valid_logits[image_index],
            all_gt[image_index],
            thresholds,
        )
        for image_index in range(len(images))
    )
    iou_workers = int(args.iou_workers)
    if iou_workers < 0:
        raise ValueError(f"--iou-workers must be non-negative, got {iou_workers}")
    if iou_workers > 1:
        worker_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=iou_workers,
            mp_context=worker_context,
            initializer=_configure_iou_worker,
        ) as executor:
            iou_results = executor.map(_build_image_iou, iou_items, chunksize=8)
            for image_index, full_iou_i, visible_iou_i in tqdm(
                iou_results,
                total=len(images),
                desc="Caching CULane region IoUs",
                unit="image",
            ):
                gt_count = full_iou_i.shape[1]
                full_iou[image_index, :, :gt_count] = full_iou_i
                visible_iou[image_index, :, :, :gt_count] = visible_iou_i
    else:
        for image_index, full_iou_i, visible_iou_i in tqdm(
            map(_build_image_iou, iou_items),
            total=len(images),
            desc="Caching CULane region IoUs",
            unit="image",
        ):
            gt_count = full_iou_i.shape[1]
            full_iou[image_index, :, :gt_count] = full_iou_i
            visible_iou[image_index, :, :, :gt_count] = visible_iou_i

    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = cache_dir / "culane_val_cache.npz"
    temporary_path = cache_dir / "culane_val_cache.tmp.npz"
    np.savez_compressed(
        temporary_path,
        pred_points=pred_points,
        pred_logits=pred_logits,
        pred_valid_logits=pred_valid_logits,
        full_iou=full_iou,
        visible_iou=visible_iou,
        gt_counts=gt_counts,
        inference_ms=inference_ms,
    )
    temporary_path.replace(arrays_path)
    metadata = {
        "contract": contract,
        "images": [str(path.resolve()) for path in images],
        "labels": [str(path.resolve()) for path in label_paths],
        "forward_count": len(images),
        "max_gt_lanes": int(max_gt),
        "average_inference_ms": float(inference_ms.mean()),
        "iou_workers": int(args.iou_workers),
    }
    (cache_dir / "culane_val_cache.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return {
        "pred_points": pred_points,
        "pred_logits": pred_logits,
        "pred_valid_logits": pred_valid_logits,
        "full_iou": full_iou,
        "visible_iou": visible_iou,
        "gt_counts": gt_counts,
        "inference_ms": inference_ms,
    }


def _load_or_build_cache(
    args: argparse.Namespace,
    cache_dir: Path,
    images: list[Path],
    label_paths: list[Path],
    all_gt: list[list[list[tuple[float, float]]]],
    imgsz: tuple[int, int],
    contract: dict[str, Any],
) -> dict[str, np.ndarray]:
    arrays_path = cache_dir / "culane_val_cache.npz"
    metadata_path = cache_dir / "culane_val_cache.json"
    if not args.force_cache and arrays_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("contract") == contract:
            with np.load(arrays_path, allow_pickle=False) as data:
                return {name: np.asarray(data[name]) for name in data.files}
        print("Cached CULane val contract does not match this request; rebuilding cache.")
    return _build_cache(args, cache_dir, images, label_paths, all_gt, imgsz, contract)


def _decode_stages(
    points: np.ndarray,
    logits: np.ndarray,
    valid_logits: np.ndarray,
    conf: float,
    point_valid_thr: float,
    nms_dist_px: float,
    max_det: int,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Return query ids after every query decoder stage for cached metric and diagnosis use."""
    sorted_points, sorted_valid_logits = _sorted_points_and_valid(points, valid_logits)
    scores = _sigmoid(logits)
    valid_masks = np.stack(
        [_longest_contiguous_mask(row >= float(point_valid_thr)) for row in _sigmoid(sorted_valid_logits)], axis=0
    )
    selected = np.flatnonzero(scores >= float(conf)).astype(np.int64)
    if selected.size == 0:
        return {
            "scores": scores,
            "valid_masks": valid_masks,
            "score_queries": np.empty((0,), dtype=np.int64),
            "nms_queries": np.empty((0,), dtype=np.int64),
            "prevalid_queries": np.empty((0,), dtype=np.int64),
            "final_queries": np.empty((0,), dtype=np.int64),
        }

    score_order = np.argsort(-scores[selected], kind="stable")
    score_queries = selected[score_order]
    if float(nms_dist_px) > 0.0 and score_queries.size > 1:
        nms_keep = lane_nms(
            torch.from_numpy(sorted_points[score_queries]),
            torch.from_numpy(scores[score_queries]),
            image_shape=image_shape,
            dist_thr_px=float(nms_dist_px),
            valid_masks=torch.from_numpy(valid_masks[score_queries]),
        ).cpu().numpy()
        nms_queries = score_queries[nms_keep]
    else:
        nms_queries = score_queries

    prevalid_queries = nms_queries[: int(max_det)] if int(max_det) > 0 else nms_queries
    final_queries = prevalid_queries[valid_masks[prevalid_queries].sum(axis=1) >= MIN_POINTS]
    return {
        "scores": scores,
        "valid_masks": valid_masks,
        "score_queries": score_queries,
        "nms_queries": nms_queries,
        "prevalid_queries": prevalid_queries,
        "final_queries": final_queries,
    }


def _match_iou_matrix(ious: np.ndarray) -> tuple[int, list[tuple[int, int, float]], dict[int, int]]:
    if ious.size == 0:
        return 0, [], {}
    pred_indices, gt_indices = linear_sum_assignment(-ious)
    pairs: list[tuple[int, int, float]] = []
    assignment: dict[int, int] = {}
    tp = 0
    for pred_index, gt_index in zip(pred_indices.tolist(), gt_indices.tolist()):
        value = float(ious[pred_index, gt_index])
        pairs.append((int(pred_index), int(gt_index), value))
        assignment[int(pred_index)] = int(gt_index)
        tp += int(value > CULANE_IOU_THRESHOLD)
    return tp, pairs, assignment


def _row_base(conf: float, point_valid_thr: float, nms_dist_px: float, max_det: int) -> dict[str, Any]:
    return {
        "conf": float(conf),
        "point_valid_thr": float(point_valid_thr),
        "nms_dist_px": float(nms_dist_px),
        "max_det": int(max_det),
    }


def _evaluate_cache(
    cache: dict[str, np.ndarray],
    point_valid_index: int,
    conf: float,
    point_valid_thr: float,
    nms_dist_px: float,
    max_det: int,
    capture: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    gt_counts = cache["gt_counts"]
    total_tp = total_fp = total_fn = 0
    count_error = 0
    gt_pred_hist: Counter[tuple[int, int]] = Counter()
    records: list[dict[str, Any]] | None = [] if capture else None
    start = time.perf_counter()

    for image_index, gt_count_raw in enumerate(gt_counts.tolist()):
        gt_count = int(gt_count_raw)
        stages = _decode_stages(
            cache["pred_points"][image_index],
            cache["pred_logits"][image_index],
            cache["pred_valid_logits"][image_index],
            conf=conf,
            point_valid_thr=point_valid_thr,
            nms_dist_px=nms_dist_px,
            max_det=max_det,
            image_shape=DATASET_IMAGE_SHAPES["culane"],
        )
        final_queries = stages["final_queries"]
        ious = cache["visible_iou"][image_index, point_valid_index, final_queries, :gt_count]
        tp, pairs, assignment = _match_iou_matrix(ious)
        pred_count = int(final_queries.shape[0])
        total_tp += int(tp)
        total_fp += pred_count - int(tp)
        total_fn += gt_count - int(tp)
        count_error += abs(pred_count - gt_count)
        gt_pred_hist[(gt_count, pred_count)] += 1
        if records is not None:
            records.append(
                {
                    "final_queries": final_queries,
                    "nms_queries": stages["nms_queries"],
                    "prevalid_queries": stages["prevalid_queries"],
                    "score_queries": stages["score_queries"],
                    "valid_masks": stages["valid_masks"],
                    "scores": stages["scores"],
                    "pairs": pairs,
                    "assignment": assignment,
                    "tp": int(tp),
                }
            )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    row = _row_base(conf, point_valid_thr, nms_dist_px, max_det)
    row.update(
        {
            "F1": float(f1),
            "Precision": float(precision),
            "Recall": float(recall),
            "TP": int(total_tp),
            "FP": int(total_fp),
            "FN": int(total_fn),
            "lane_count_mae": float(count_error / max(len(gt_counts), 1)),
            "cached_decode_match_ms": float(elapsed_ms),
        }
    )
    for (gt_count, pred_count), value in sorted(gt_pred_hist.items()):
        row[f"GT{gt_count}->Pred{pred_count}"] = int(value)
    return row, records


def _init_sweep_worker() -> None:
    """Use one BLAS/OpenCV thread per process for configuration-level parallelism."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _evaluate_config_worker(
    item: tuple[int, float, float, float, int],
) -> dict[str, Any]:
    if _SWEEP_CACHE is None:
        raise RuntimeError("Cached sweep worker was not initialized with a cache.")
    point_valid_index, conf, point_valid_thr, nms_dist_px, max_det = item
    row, _ = _evaluate_cache(
        _SWEEP_CACHE,
        point_valid_index=point_valid_index,
        conf=conf,
        point_valid_thr=point_valid_thr,
        nms_dist_px=nms_dist_px,
        max_det=max_det,
    )
    return row


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, int, int]:
    return (-float(row["F1"]), -float(row["Precision"]), -float(row["Recall"]), int(row["FP"]), int(row["FN"]))


def _diagnose_misses(
    cache: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    point_valid_index: int,
    best_row: dict[str, Any],
) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    by_gt_count: dict[str, Counter[str]] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    threshold = float(best_row["point_valid_thr"])
    conf = float(best_row["conf"])

    for image_index, record in enumerate(records):
        gt_count = int(cache["gt_counts"][image_index])
        if gt_count <= 0:
            continue
        final_queries = np.asarray(record["final_queries"], dtype=np.int64)
        final_ious = cache["visible_iou"][image_index, point_valid_index, final_queries, :gt_count]
        matched_tp_gts = {
            gt_index
            for pred_index, gt_index, value in record["pairs"]
            if float(value) > CULANE_IOU_THRESHOLD
        }
        final_assignment = dict(record["assignment"])
        final_query_to_row = {int(query): row_index for row_index, query in enumerate(final_queries.tolist())}

        raw_full = cache["full_iou"][image_index, :, :gt_count]
        raw_visible = cache["visible_iou"][image_index, point_valid_index, :, :gt_count]
        scores = np.asarray(record["scores"], dtype=np.float32)
        valid_masks = np.asarray(record["valid_masks"], dtype=bool)
        nms_queries = set(np.asarray(record["nms_queries"], dtype=np.int64).tolist())
        prevalid_queries = set(np.asarray(record["prevalid_queries"], dtype=np.int64).tolist())

        for gt_index in range(gt_count):
            if gt_index in matched_tp_gts:
                continue
            full_good = np.flatnonzero(raw_full[:, gt_index] > CULANE_IOU_THRESHOLD)
            weak = np.flatnonzero(raw_full[:, gt_index] > WEAK_IOU_THRESHOLD)
            if weak.size == 0:
                category = "candidate_missing"
            elif full_good.size == 0:
                category = "geometry_bad"
            else:
                score_good = full_good[scores[full_good] >= conf]
                if score_good.size == 0:
                    category = "low_existence_score"
                else:
                    valid_good = score_good[valid_masks[score_good].sum(axis=1) >= MIN_POINTS]
                    if valid_good.size == 0:
                        category = "low_point_valid"
                    else:
                        visible_good = valid_good[raw_visible[valid_good, gt_index] > CULANE_IOU_THRESHOLD]
                        if visible_good.size == 0:
                            category = "region_iou_below_0.5"
                        elif not any(int(query) in nms_queries for query in visible_good.tolist()):
                            category = "suppressed_by_nms"
                        elif not any(int(query) in prevalid_queries for query in visible_good.tolist()):
                            category = "removed_by_max_det"
                        else:
                            final_good = [
                                int(query)
                                for query in visible_good.tolist()
                                if int(query) in final_query_to_row
                            ]
                            conflict = False
                            for query in final_good:
                                pred_row = final_query_to_row[query]
                                assigned_gt = final_assignment.get(pred_row)
                                if assigned_gt is not None and assigned_gt != gt_index:
                                    conflict = True
                                    break
                            category = "assignment_conflict" if conflict else "region_iou_below_0.5"

            categories[category] += 1
            by_gt_count.setdefault(str(gt_count), Counter())[category] += 1
            bucket = examples.setdefault(category, [])
            if len(bucket) < 10:
                bucket.append(
                    {
                        "image_index": image_index,
                        "gt_index": gt_index,
                        "gt_count": gt_count,
                        "best_full_iou": float(raw_full[:, gt_index].max(initial=0.0)),
                        "best_visible_iou": float(raw_visible[:, gt_index].max(initial=0.0)),
                        "best_score": float(scores.max(initial=0.0)),
                        "point_valid_thr": threshold,
                    }
                )

    return {
        "diagnostic_contract": {
            "official_tp_iou_rule": f"IoU > {CULANE_IOU_THRESHOLD}",
            "weak_candidate_iou_rule": f"IoU > {WEAK_IOU_THRESHOLD}",
            "point_valid_thr": threshold,
            "decode_uses_ground_truth": False,
        },
        "fn_total": int(sum(categories.values())),
        "categories": dict(categories),
        "by_gt_lane_count": {key: dict(value) for key, value in sorted(by_gt_count.items())},
        "examples": examples,
    }


def _write_rows(save_dir: Path, rows: list[dict[str, Any]]) -> None:
    json_path = save_dir / "sweep_rows.json"
    csv_path = save_dir / "sweep_rows.csv"
    json_path.write_text(json.dumps(rows, indent=2, default=_json_default) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global _SWEEP_CACHE
    args = parse_args()
    imgsz = normalize_imgsz(args.imgsz, dataset="culane")
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(args.source, max_images=int(args.max_images))
    label_dir = Path(args.labels).expanduser().resolve()
    archive_root = Path(args.culane_archive_root).expanduser().resolve()
    label_paths, all_gt = _load_raw_gt(images, label_dir=label_dir, archive_root=archive_root)
    contract = _cache_contract(args, images, imgsz)
    cache = _load_or_build_cache(args, cache_dir, images, label_paths, all_gt, imgsz, contract)

    rows: list[dict[str, Any]] = []
    threshold_to_index = {float(value): index for index, value in enumerate(args.point_valid_thrs)}
    combinations = list(product(args.confs, args.point_valid_thrs, args.nms_dist_pxs, args.max_dets))
    sweep_items = [
        (
            threshold_to_index[float(point_valid_thr)],
            float(conf),
            float(point_valid_thr),
            float(nms_dist_px),
            int(max_det),
        )
        for conf, point_valid_thr, nms_dist_px, max_det in combinations
    ]
    sweep_workers = int(args.sweep_workers)
    if sweep_workers < 0:
        raise ValueError(f"--sweep-workers must be non-negative, got {sweep_workers}")
    if sweep_workers > 1:
        _SWEEP_CACHE = cache
        if "fork" in multiprocessing.get_all_start_methods():
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=sweep_workers,
                mp_context=context,
                initializer=_init_sweep_worker,
            ) as executor:
                row_iter = executor.map(_evaluate_config_worker, sweep_items, chunksize=1)
                rows.extend(
                    tqdm(
                        row_iter,
                        total=len(sweep_items),
                        desc="Parallel cached decode sweep",
                        unit="config",
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=sweep_workers) as executor:
                row_iter = executor.map(_evaluate_config_worker, sweep_items)
                rows.extend(
                    tqdm(
                        row_iter,
                        total=len(sweep_items),
                        desc="Parallel cached decode sweep",
                        unit="config",
                    )
                )
    else:
        for item in tqdm(sweep_items, desc="Cached decode sweep", unit="config"):
            rows.append(_evaluate_config_worker(item) if _SWEEP_CACHE is not None else _evaluate_cache(
                cache,
                point_valid_index=item[0],
                conf=item[1],
                point_valid_thr=item[2],
                nms_dist_px=item[3],
                max_det=item[4],
            )[0])
    rows.sort(key=_sort_key)
    _write_rows(save_dir, rows)

    best = rows[0]
    point_valid_index = threshold_to_index[float(best["point_valid_thr"])]
    _, records = _evaluate_cache(
        cache,
        point_valid_index=point_valid_index,
        conf=float(best["conf"]),
        point_valid_thr=float(best["point_valid_thr"]),
        nms_dist_px=float(best["nms_dist_px"]),
        max_det=int(best["max_det"]),
        capture=True,
    )
    assert records is not None
    diagnosis = _diagnose_misses(cache, records, point_valid_index=point_valid_index, best_row=best)
    (save_dir / "raw_query_miss_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    summary = {
        "protocol": {
            "split": "val",
            "test_used": False,
            "model_forward_count": int(cache["pred_points"].shape[0]),
            "decode_combinations": len(combinations),
            "cache_dir": str(cache_dir),
            "weights": str(Path(args.weights).expanduser().resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "half": bool(args.half),
            "batch_size": int(args.batch_size),
        },
        "cache": {
            "average_inference_ms": float(cache["inference_ms"].mean()),
            "total_images": int(cache["pred_points"].shape[0]),
        },
        "best": best,
        "top10": rows[:10],
        "diagnosis": diagnosis,
    }
    (save_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"Saved cached CULane sweep to: {save_dir}")


if __name__ == "__main__":
    main()
