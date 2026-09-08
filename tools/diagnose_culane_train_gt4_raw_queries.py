from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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

from tools.eval_gcs import (  # noqa: E402
    CULANE_IOU_THRESHOLD,
    CULANE_LANE_WIDTH,
    CULANE_RAW_SHAPE,
    label_path_for_image,
    load_culane_ground_truth,
)
from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image  # noqa: E402
from tools.sweep_culane_cached import (  # noqa: E402
    MIN_POINTS,
    _decode_stages,
    _fast_culane_lane_mask,
    _lane_ious_against_masks,
    _longest_contiguous_mask,
    _sigmoid,
    _sorted_points_and_valid,
)
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, assert_gcs_shape, normalize_imgsz  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming CULane train-split GT4 raw-query geometry diagnosis."
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--culane-archive-root", required=True)
    parser.add_argument("--imgsz", nargs=2, type=int, metavar=("H", "W"), default=(384, 960))
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--iou-workers",
        type=int,
        default=0,
        help="Parallel CPU workers for per-image raw IoU/decode diagnosis; 0 keeps sequential behavior.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--gt-count",
        type=int,
        default=4,
        help="Only run images with this ground-truth lane count; 0 keeps all images.",
    )
    parser.add_argument("--point-valid-thr", type=float, default=0.30)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms-dist-px", type=float, default=60.0)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing contiguous gt4_train_hardset.jsonl/csv prefix.",
    )
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


def _percentile(values: list[float], quantiles: list[float]) -> dict[str, float | None]:
    if not values:
        return {str(q): None for q in quantiles}
    array = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.quantile(array, q)) for q in quantiles}


def _visible_span(
    sorted_points: np.ndarray,
    sorted_valid_logits: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int, int]:
    scores = _sigmoid(sorted_valid_logits)
    mask = _longest_contiguous_mask(scores >= float(threshold))
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        return mask, -1, -1
    return mask, int(valid_indices[0]), int(valid_indices[-1])


def _interp_gt_x(gt_lane: list[tuple[float, float]], ys: np.ndarray) -> np.ndarray:
    points = np.asarray(gt_lane, dtype=np.float64)
    order = np.argsort(points[:, 1], kind="stable")
    x = points[order, 0]
    y = points[order, 1]
    unique_y, unique_indices = np.unique(y, return_index=True)
    unique_x = x[unique_indices]
    if unique_y.size < 2:
        return np.full_like(ys, np.nan, dtype=np.float64)
    return np.interp(ys, unique_y, unique_x, left=np.nan, right=np.nan)


def _geometry_errors(
    pred_points_norm: np.ndarray,
    gt_lane: list[tuple[float, float]],
) -> tuple[float | None, float | None, float | None]:
    pred = np.asarray(pred_points_norm, dtype=np.float64)
    pred_raw = pred * np.asarray([CULANE_RAW_SHAPE[1], CULANE_RAW_SHAPE[0]], dtype=np.float64)
    gt_x = _interp_gt_x(gt_lane, pred_raw[:, 1])
    valid = np.isfinite(gt_x)
    if int(valid.sum()) == 0:
        return None, None, None
    errors = np.abs(pred_raw[:, 0] - gt_x)
    mean_error = float(errors[valid].mean())
    valid_indices = np.flatnonzero(valid)
    bottom_error = float(errors[valid_indices[np.argmax(pred_raw[valid_indices, 1])]])
    top_error = float(errors[valid_indices[np.argmin(pred_raw[valid_indices, 1])]])
    return mean_error, bottom_error, top_error


def _best_query_stats(
    points: np.ndarray,
    valid_logits: np.ndarray,
    logits: np.ndarray,
    gt_lanes: list[list[tuple[float, float]]],
    valid_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sorted_points, sorted_valid_logits = _sorted_points_and_valid(points, valid_logits)
    scores = _sigmoid(logits)
    gt_masks = [
        _fast_culane_lane_mask(
            np.asarray(gt_lane, dtype=np.float64),
            image_shape=CULANE_RAW_SHAPE,
            lane_width=CULANE_LANE_WIDTH,
        )
        > 0
        for gt_lane in gt_lanes
    ]
    gt_areas = np.asarray([int(mask.sum()) for mask in gt_masks], dtype=np.int32)
    query_count = sorted_points.shape[0]
    gt_count = len(gt_lanes)
    full_iou = np.zeros((query_count, gt_count), dtype=np.float32)
    visible_iou = np.zeros((query_count, gt_count), dtype=np.float32)
    visible_masks: list[np.ndarray] = []
    visible_ranges: list[tuple[int, int]] = []
    for query_index in range(query_count):
        full_iou[query_index] = _lane_ious_against_masks(
            sorted_points[query_index],
            gt_masks,
            gt_areas,
        )
        visible_mask, span_start, span_end = _visible_span(
            sorted_points[query_index],
            sorted_valid_logits[query_index],
            valid_threshold,
        )
        visible_masks.append(visible_mask)
        visible_ranges.append((span_start, span_end))
        if int(visible_mask.sum()) >= MIN_POINTS:
            visible_iou[query_index] = _lane_ious_against_masks(
                sorted_points[query_index, visible_mask],
                gt_masks,
                gt_areas,
            )

    query_best_for_gt = np.argmax(visible_iou, axis=0) if gt_count else np.empty((0,), dtype=np.int64)
    query_best_counts = Counter(int(query) for query in query_best_for_gt.tolist())
    image_rows: list[dict[str, Any]] = []
    for gt_index, gt_lane in enumerate(gt_lanes):
        visible_order = np.argsort(-visible_iou[:, gt_index], kind="stable")
        full_order = np.argsort(-full_iou[:, gt_index], kind="stable")
        best_visible_query = int(visible_order[0])
        best_full_query = int(full_order[0])
        best_visible_iou = float(visible_iou[best_visible_query, gt_index])
        second_visible_iou = float(visible_iou[visible_order[1], gt_index]) if query_count > 1 else 0.0
        best_full_iou = float(full_iou[best_full_query, gt_index])
        second_full_iou = float(full_iou[full_order[1], gt_index]) if query_count > 1 else 0.0
        best_mask = visible_masks[best_visible_query]
        span_start, span_end = visible_ranges[best_visible_query]
        mean_error, bottom_error, top_error = _geometry_errors(
            sorted_points[best_visible_query, best_mask] if int(best_mask.sum()) >= MIN_POINTS else sorted_points[best_full_query],
            gt_lane,
        )
        weak_query_count = int((visible_iou[:, gt_index] > 0.10).sum())
        best_visible_score = float(scores[best_visible_query])
        if best_visible_iou > CULANE_IOU_THRESHOLD:
            if best_visible_score < 0.30:
                category = "score_failure"
            else:
                category = "raw_candidate_good"
        elif best_full_iou > CULANE_IOU_THRESHOLD:
            category = "valid_span_failure"
        elif best_visible_iou > 0.10:
            category = "geometry_near_miss"
        else:
            category = "geometry_missing"
        image_rows.append(
            {
                "gt_index": int(gt_index),
                "gt_count": int(gt_count),
                "category": category,
                "best_visible_iou": best_visible_iou,
                "second_best_visible_iou": second_visible_iou,
                "best_full_iou": best_full_iou,
                "second_best_full_iou": second_full_iou,
                "best_visible_query": best_visible_query,
                "best_full_query": best_full_query,
                "best_visible_score": best_visible_score,
                "best_visible_valid_count": int(best_mask.sum()),
                "best_visible_span_start": int(span_start),
                "best_visible_span_end": int(span_end),
                "best_visible_span_y_bottom_px": (
                    float(sorted_points[best_visible_query, span_start, 1] * CULANE_RAW_SHAPE[0])
                    if span_start >= 0
                    else None
                ),
                "best_visible_span_y_top_px": (
                    float(sorted_points[best_visible_query, span_end, 1] * CULANE_RAW_SHAPE[0])
                    if span_end >= 0
                    else None
                ),
                "weak_visible_query_count": weak_query_count,
                "best_mean_abs_x_error_px": mean_error,
                "best_bottom_x_error_px": bottom_error,
                "best_top_x_error_px": top_error,
                "best_query_collision_count": int(query_best_counts[best_visible_query]),
            }
        )
    return full_iou, visible_iou, image_rows


def _final_decode_gt_presence(
    points: np.ndarray,
    logits: np.ndarray,
    valid_logits: np.ndarray,
    gt_lanes: list[list[tuple[float, float]]],
    conf: float,
    point_valid_thr: float,
    nms_dist_px: float,
    max_det: int,
) -> np.ndarray:
    stages = _decode_stages(
        points,
        logits,
        valid_logits,
        conf=conf,
        point_valid_thr=point_valid_thr,
        nms_dist_px=nms_dist_px,
        max_det=max_det,
        image_shape=DATASET_IMAGE_SHAPES["culane"],
    )
    final_queries = np.asarray(stages["final_queries"], dtype=np.int64)
    if final_queries.size == 0 or not gt_lanes:
        return np.zeros((len(gt_lanes),), dtype=bool)
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
    sorted_points, _ = _sorted_points_and_valid(points, valid_logits)
    ious = np.zeros((final_queries.size, len(gt_lanes)), dtype=np.float32)
    for row_index, query_index in enumerate(final_queries.tolist()):
        visible_mask = np.asarray(stages["valid_masks"][query_index], dtype=bool)
        if int(visible_mask.sum()) >= MIN_POINTS:
            ious[row_index] = _lane_ious_against_masks(
                sorted_points[query_index, visible_mask],
                masks,
                areas,
            )
    pred_indices, gt_indices = linear_sum_assignment(-ious)
    presence = np.zeros((len(gt_lanes),), dtype=bool)
    for pred_index, gt_index in zip(pred_indices.tolist(), gt_indices.tolist()):
        presence[gt_index] = bool(float(ious[pred_index, gt_index]) > CULANE_IOU_THRESHOLD)
    return presence


def _configure_diagnosis_worker() -> None:
    """Avoid CPU thread oversubscription when several diagnostic workers run."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _restore_partial_outputs(
    jsonl_path: Path,
    csv_path: Path,
    images: list[Path],
    fields: list[str],
) -> tuple[int, Counter[str], Counter[str], list[float], list[float], list[float], list[float], list[float], int]:
    """Restore statistics from a contiguous, lane-complete partial output."""
    if not jsonl_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            "--resume requires both existing output files: "
            f"{jsonl_path} and {csv_path}"
        )

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot parse {jsonl_path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{jsonl_path} line {line_number} is not a JSON object.")
            rows.append(row)

    if len(rows) % 4 != 0:
        raise ValueError(
            f"Partial JSONL has {len(rows)} rows, which is not a complete GT4-image prefix."
        )
    resume_image_count = len(rows) // 4
    if resume_image_count > len(images):
        raise ValueError(
            f"Partial JSONL covers {resume_image_count} images, but only {len(images)} "
            "filtered GT4 images are available."
        )

    expected_image_index = 0
    expected_gt_index = 0
    category_counts: Counter[str] = Counter()
    category_image_counts: Counter[str] = Counter()
    all_best_visible: list[float] = []
    all_second_visible: list[float] = []
    all_best_full: list[float] = []
    all_scores: list[float] = []
    all_valid_counts: list[float] = []
    collision_rows = 0
    image_categories: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        missing = [field for field in fields if field not in row]
        if missing:
            raise ValueError(
                f"{jsonl_path} line {row_number} is missing fields: {', '.join(missing)}"
            )
        image_index = int(row["image_index"])
        gt_index = int(row["gt_index"])
        if image_index != expected_image_index or gt_index != expected_gt_index:
            raise ValueError(
                f"Partial JSONL is not contiguous at line {row_number}: "
                f"expected image_index={expected_image_index}, gt_index={expected_gt_index}, "
                f"got image_index={image_index}, gt_index={gt_index}."
            )
        if str(row["image"]) != str(images[image_index]):
            raise ValueError(
                f"Partial JSONL image mismatch at line {row_number}: "
                f"expected {images[image_index]}, got {row['image']}."
            )
        category = str(row["category"])
        category_counts[category] += 1
        image_categories.add(category)
        all_best_visible.append(float(row["best_visible_iou"]))
        all_second_visible.append(float(row["second_best_visible_iou"]))
        all_best_full.append(float(row["best_full_iou"]))
        all_scores.append(float(row["best_visible_score"]))
        all_valid_counts.append(float(row["best_visible_valid_count"]))
        if int(row["best_query_collision_count"]) > 1:
            collision_rows += 1

        expected_gt_index += 1
        if expected_gt_index == 4:
            for image_category in image_categories:
                category_image_counts[image_category] += 1
            image_categories.clear()
            expected_image_index += 1
            expected_gt_index = 0

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(
                f"CSV header mismatch in {csv_path}: expected {fields}, got {reader.fieldnames}."
            )
        csv_rows = sum(1 for _ in reader)
    if csv_rows != len(rows):
        raise ValueError(
            f"Partial CSV/JSONL row mismatch: csv={csv_rows}, jsonl={len(rows)}."
        )

    return (
        resume_image_count,
        category_counts,
        category_image_counts,
        all_best_visible,
        all_second_visible,
        all_best_full,
        all_scores,
        all_valid_counts,
        collision_rows,
    )


def _diagnose_image(
    item: tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[list[tuple[float, float]]],
        float,
        float,
        float,
        int,
    ],
) -> tuple[int, list[dict[str, Any]], np.ndarray]:
    """Run all CPU-heavy E2 diagnosis for one image in an isolated worker."""
    (
        image_index,
        points,
        valid_logits,
        logits,
        gt_lanes,
        valid_threshold,
        conf,
        nms_dist_px,
        max_det,
    ) = item
    _, _, rows = _best_query_stats(
        points,
        valid_logits,
        logits,
        gt_lanes,
        valid_threshold=valid_threshold,
    )
    final_presence = _final_decode_gt_presence(
        points,
        logits,
        valid_logits,
        gt_lanes,
        conf=conf,
        point_valid_thr=valid_threshold,
        nms_dist_px=nms_dist_px,
        max_det=max_det,
    )
    return image_index, rows, final_presence


def main() -> None:
    args = parse_args()
    imgsz = normalize_imgsz(args.imgsz, dataset="culane")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.iou_workers < 0:
        raise ValueError("--iou-workers must be non-negative")
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    source_images = collect_images(args.source, max_images=int(args.max_images))
    images = source_images
    label_dir = Path(args.labels).expanduser().resolve()
    archive_root = Path(args.culane_archive_root).expanduser().resolve()
    source_gt_lane_total = 0
    source_gt_count_distribution: Counter[str] = Counter()
    if int(args.gt_count) > 0:
        filtered_images: list[Path] = []
        for image_path in tqdm(source_images, desc="Filtering GT lane count", unit="image"):
            label_path = label_path_for_image(image_path, label_dir)
            gt_lanes = load_culane_ground_truth(
                label_path,
                archive_root=archive_root,
                raw_shape=CULANE_RAW_SHAPE,
            )
            source_gt_lane_total += len(gt_lanes)
            source_gt_count_distribution[str(len(gt_lanes))] += 1
            if len(gt_lanes) == int(args.gt_count):
                filtered_images.append(image_path)
        images = filtered_images
        if not images:
            raise ValueError(f"No images with gt-count={args.gt_count} found in {args.source}.")
    else:
        for image_path in tqdm(source_images, desc="Loading GT lane counts", unit="image"):
            label_path = label_path_for_image(image_path, label_dir)
            gt_lanes = load_culane_ground_truth(
                label_path,
                archive_root=archive_root,
                raw_shape=CULANE_RAW_SHAPE,
            )
            source_gt_lane_total += len(gt_lanes)
            source_gt_count_distribution[str(len(gt_lanes))] += 1
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)

    if args.warmup > 0 and images:
        image = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(images[0])
        tensor = preprocess_image(image, imgsz=imgsz, device=device, half=bool(args.half))
        tensor = tensor.repeat(min(args.batch_size, len(images)), 1, 1, 1)
        for _ in range(args.warmup):
            _ = model(tensor)
        _sync_if_cuda(device)

    jsonl_path = save_dir / "gt4_train_hardset.jsonl"
    csv_path = save_dir / "gt4_train_hardset.csv"
    summary_path = save_dir / "gt4_train_hardset_summary.json"
    fields = [
        "image_index",
        "image",
        "label",
        "gt_index",
        "gt_count",
        "category",
        "best_visible_iou",
        "second_best_visible_iou",
        "best_full_iou",
        "second_best_full_iou",
        "best_visible_query",
        "best_full_query",
        "best_visible_score",
        "best_visible_valid_count",
        "best_visible_span_start",
        "best_visible_span_end",
        "best_visible_span_y_bottom_px",
        "best_visible_span_y_top_px",
        "weak_visible_query_count",
        "best_mean_abs_x_error_px",
        "best_bottom_x_error_px",
        "best_top_x_error_px",
        "best_query_collision_count",
        "final_decode_presence",
    ]
    category_counts: Counter[str] = Counter()
    category_image_counts: Counter[str] = Counter()
    all_best_visible: list[float] = []
    all_second_visible: list[float] = []
    all_best_full: list[float] = []
    all_scores: list[float] = []
    all_valid_counts: list[float] = []
    collision_rows = 0
    total_gt_lanes = 0
    gt4_images = 0
    total_gt4_image_candidates = 0
    resume_image_count = 0
    if args.resume:
        (
            resume_image_count,
            category_counts,
            category_image_counts,
            all_best_visible,
            all_second_visible,
            all_best_full,
            all_scores,
            all_valid_counts,
            collision_rows,
        ) = _restore_partial_outputs(jsonl_path, csv_path, images, fields)
        gt4_images = resume_image_count
        total_gt_lanes = resume_image_count * 4
        total_gt4_image_candidates = resume_image_count * 12
        print(
            f"Resuming from {resume_image_count} complete GT4 images "
            f"({total_gt_lanes} lane rows)."
        )
    start_time = time.perf_counter()
    batch_size = int(args.batch_size)
    diagnosis_executor: ProcessPoolExecutor | None = None
    if args.iou_workers > 1:
        diagnosis_executor = ProcessPoolExecutor(
            max_workers=int(args.iou_workers),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_configure_diagnosis_worker,
        )
    csv_mode = "a" if args.resume else "w"
    jsonl_mode = "a" if args.resume else "w"
    csv_handle = csv_path.open(csv_mode, newline="", encoding="utf-8")
    jsonl_handle = jsonl_path.open(jsonl_mode, encoding="utf-8")
    writer = csv.DictWriter(csv_handle, fieldnames=fields)
    if not args.resume:
        writer.writeheader()
    try:
        for batch_start in tqdm(
            range(resume_image_count, len(images), batch_size),
            total=(len(images) + batch_size - 1) // batch_size,
            desc="E2 train forward",
            unit="batch",
        ):
            batch_paths = images[batch_start : batch_start + batch_size]
            tensors: list[torch.Tensor] = []
            gt_batch: list[list[list[tuple[float, float]]]] = []
            labels: list[Path] = []
            for image_path in batch_paths:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(image_path)
                assert_gcs_shape(
                    image.shape[:2],
                    DATASET_IMAGE_SHAPES["culane"],
                    name="CULane train image",
                    context=f"diagnose_culane_train_gt4_raw_queries({image_path})",
                )
                tensors.append(preprocess_image(image, imgsz=imgsz, device=device, half=bool(args.half)))
                label_path = label_path_for_image(image_path, label_dir)
                gt_lanes = load_culane_ground_truth(
                    label_path,
                    archive_root=archive_root,
                    raw_shape=CULANE_RAW_SHAPE,
                )
                labels.append(label_path)
                gt_batch.append(gt_lanes)
            tensor = torch.cat(tensors, dim=0)
            _sync_if_cuda(device)
            forward_start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(tensor)
            _sync_if_cuda(device)
            _ = (time.perf_counter() - forward_start) * 1000.0
            points_batch = outputs["pred_points"].detach().float().cpu().numpy()
            logits_batch = outputs["pred_logits"].detach().float().cpu().numpy()
            valid_output = outputs.get("pred_valid_logits")
            if valid_output is None:
                valid_batch = np.full(points_batch.shape[:3], 20.0, dtype=np.float32)
            else:
                valid_batch = valid_output.detach().float().cpu().numpy()
            diagnosis_items = [
                (
                    batch_start + local_index,
                    points_batch[local_index],
                    valid_batch[local_index],
                    logits_batch[local_index],
                    gt_lanes,
                    float(args.point_valid_thr),
                    float(args.conf),
                    float(args.nms_dist_px),
                    int(args.max_det),
                )
                for local_index, gt_lanes in enumerate(gt_batch)
                if len(gt_lanes) == 4
            ]
            if diagnosis_executor is None:
                diagnosis_results = map(_diagnose_image, diagnosis_items)
            else:
                diagnosis_results = diagnosis_executor.map(_diagnose_image, diagnosis_items, chunksize=4)
            for current_index, rows, final_presence in diagnosis_results:
                local_index = current_index - batch_start
                gt_lanes = gt_batch[local_index]
                gt4_images += 1
                total_gt4_image_candidates += points_batch.shape[1]
                image_categories: set[str] = set()
                for row in rows:
                    row["image_index"] = int(current_index)
                    row["image"] = str(images[current_index])
                    row["label"] = str(labels[local_index])
                    row["final_decode_presence"] = bool(final_presence[row["gt_index"]])
                    if row["category"] == "raw_candidate_good" and not bool(row["final_decode_presence"]):
                        row["category"] = "decoder_failure"
                    writer.writerow(row)
                    jsonl_handle.write(json.dumps(row, default=_json_default) + "\n")
                    category = str(row["category"])
                    category_counts[category] += 1
                    image_categories.add(category)
                    total_gt_lanes += 1
                    all_best_visible.append(float(row["best_visible_iou"]))
                    all_second_visible.append(float(row["second_best_visible_iou"]))
                    all_best_full.append(float(row["best_full_iou"]))
                    all_scores.append(float(row["best_visible_score"]))
                    all_valid_counts.append(float(row["best_visible_valid_count"]))
                    if int(row["best_query_collision_count"]) > 1:
                        collision_rows += 1
                for category in image_categories:
                    category_image_counts[category] += 1
    finally:
        if diagnosis_executor is not None:
            diagnosis_executor.shutdown(wait=True)
        csv_handle.close()
        jsonl_handle.close()

    elapsed = time.perf_counter() - start_time
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    summary = {
        "protocol": {
            "task": "E2 CULane train GT4 raw-query geometry diagnosis",
            "split": "train",
            "test_used": False,
            "gt_used_only_for_offline_diagnosis": True,
            "decode_uses_ground_truth": False,
            "weights": str(Path(args.weights).expanduser().resolve()),
            "source": str(Path(args.source).expanduser().resolve()),
            "labels": str(label_dir),
            "culane_archive_root": str(archive_root),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "half": bool(args.half),
            "point_valid_thr": float(args.point_valid_thr),
            "conf": float(args.conf),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "gt_count_filter": int(args.gt_count),
            "official_iou_threshold": float(CULANE_IOU_THRESHOLD),
            "culane_raw_shape": list(CULANE_RAW_SHAPE),
            "culane_lane_width": int(CULANE_LANE_WIDTH),
            "raw_query_count": 12,
            "raw_point_count": 56,
            "iou_workers": int(args.iou_workers),
        },
        "counts": {
            "train_images_seen": len(source_images),
            "gt4_filter_input_images": len(source_images),
            "gt4_images": int(gt4_images),
            "gt4_lanes": int(total_gt_lanes),
            "all_gt_lanes_seen": int(source_gt_lane_total),
            "source_gt_count_distribution": dict(
                sorted(source_gt_count_distribution.items(), key=lambda item: int(item[0]))
            ),
            "category_image_counts": dict(category_image_counts),
            "category_lane_counts": dict(category_counts),
            "best_query_collision_rows": int(collision_rows),
            "best_query_collision_rate": float(collision_rows / max(total_gt_lanes, 1)),
        },
        "category_rates": {
            key: float(value / max(total_gt_lanes, 1))
            for key, value in sorted(category_counts.items())
        },
        "statistics": {
            "best_visible_iou": _percentile(all_best_visible, quantiles),
            "second_best_visible_iou": _percentile(all_second_visible, quantiles),
            "best_full_iou": _percentile(all_best_full, quantiles),
            "best_visible_score": _percentile(all_scores, quantiles),
            "best_visible_valid_count": _percentile(all_valid_counts, quantiles),
        },
        "runtime": {
            "elapsed_seconds": float(elapsed),
            "images_per_second": float(len(images) / max(elapsed, 1e-9)),
        },
        "outputs": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
