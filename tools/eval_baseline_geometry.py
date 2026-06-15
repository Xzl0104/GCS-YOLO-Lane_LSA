from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.label_utils import resample_polyline
from tools.eval_gcs import label_path_for_image, load_gcs_label, match_lanes, summarize
from tools.infer_gcs import collect_images
from ultralytics import YOLO
from ultralytics.utils.gcs_postprocess import GCS_DEFAULT_MAX_DET, draw_gcs_lanes, save_gcs_lanes_txt


DEFAULT_WEIGHTS = ROOT / "runs" / "baseline" / "yolo11s_seg_tusimple-2" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "images" / "val"
DEFAULT_LABELS = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "labels_gcs" / "val"
DEFAULT_DATA = ROOT / "data" / "tusimple_yolo.yaml"

MASK_COLORS = (
    (0, 180, 255),
    (0, 255, 0),
    (255, 120, 0),
    (255, 0, 180),
    (80, 180, 255),
    (180, 255, 80),
    (255, 80, 80),
    (180, 120, 255),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a YOLO11 segmentation baseline with the same structured lane geometry metrics used by "
            "GCS-YOLO-Lane."
        )
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO11 segmentation checkpoint.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image file, image directory, or txt list.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="labels_gcs directory for geometric GT.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="YOLO segmentation data yaml for optional mAP val.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference long-side image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO mask confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--ape-thr", type=float, default=20.0, help="APE threshold in pixels for TP matching.")
    parser.add_argument("--max-det", type=int, default=GCS_DEFAULT_MAX_DET, help="Maximum baseline mask instances per image.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images. 0 means all.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--num-points", type=int, default=32, help="Number of structured points per decoded lane.")
    parser.add_argument("--mask-thr", type=float, default=0.5, help="Binary threshold for predicted masks.")
    parser.add_argument("--min-area", type=int, default=40, help="Minimum mask area in pixels after binarization.")
    parser.add_argument("--min-row-pixels", type=int, default=2, help="Minimum foreground pixels needed to keep a row.")
    parser.add_argument("--smooth-window", type=int, default=7, help="Odd moving-average window for extracted centerline x.")
    parser.add_argument("--save-dir", default="runs/baseline/geometry_eval", help="Directory for eval_summary.json.")
    parser.add_argument("--save-json", action="store_true", help="Save per-image predictions and matches.")
    parser.add_argument("--save-img", action="store_true", help="Save rendered mask-centerline predictions.")
    parser.add_argument("--save-txt", action="store_true", help="Save decoded normalized lane points as txt.")
    parser.add_argument("--line-width", type=int, default=2, help="Polyline width for saved prediction images.")
    parser.add_argument("--run-map", action="store_true", help="Also run YOLO segmentation validation and store mAP metrics.")
    parser.add_argument("--map-batch", type=int, default=2, help="Batch size for --run-map.")
    parser.add_argument("--map-workers", type=int, default=0, help="Workers for --run-map.")
    return parser.parse_args()


def _as_numpy_masks(result) -> np.ndarray | None:
    """Return result masks as an N x H x W numpy array, or None when no masks were predicted."""
    if result.masks is None or result.masks.data is None:
        return None
    masks = result.masks.data
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().float().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"Expected YOLO masks with shape N x H x W, got {masks.shape}.")
    return masks


def _prediction_scores(result, n: int) -> np.ndarray:
    """Read YOLO instance confidences, falling back to ones if boxes are unavailable."""
    if result.boxes is None or result.boxes.conf is None:
        return np.ones((n,), dtype=np.float32)
    scores = result.boxes.conf.detach().float().cpu().numpy().astype(np.float32)
    if scores.shape[0] != n:
        return np.ones((n,), dtype=np.float32)
    return scores


def _largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Keep the largest connected foreground component to avoid averaging unrelated fragments."""
    if int(mask.sum()) < min_area:
        return np.zeros_like(mask, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = int(np.argmax(areas)) + 1
    if int(stats[keep, cv2.CC_STAT_AREA]) < min_area:
        return np.zeros_like(mask, dtype=np.uint8)
    return (labels == keep).astype(np.uint8)


def _smooth_centerline(points: list[tuple[float, float]], window: int) -> list[tuple[float, float]]:
    """Smooth x coordinates while preserving y order."""
    if window <= 1 or len(points) < 3:
        return points
    if window % 2 == 0:
        window += 1
    window = min(window, len(points) if len(points) % 2 == 1 else len(points) - 1)
    if window <= 1:
        return points

    arr = np.asarray(points, dtype=np.float32)
    pad = window // 2
    padded = np.pad(arr[:, 0], (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    xs = np.convolve(padded, kernel, mode="valid")
    return [(float(x), float(y)) for x, y in zip(xs, arr[:, 1])]


def mask_to_lane_points(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    num_points: int,
    mask_thr: float,
    min_area: int,
    min_row_pixels: int,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Convert one predicted lane mask to normalized bottom-to-top structured points."""
    h, w = int(image_shape[0]), int(image_shape[1])
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    binary = (mask >= float(mask_thr)).astype(np.uint8)
    binary = _largest_component(binary, min_area=min_area)
    if int(binary.sum()) < min_area:
        return None

    points: list[tuple[float, float]] = []
    ys = np.flatnonzero(binary.any(axis=1))
    for y in ys:
        xs = np.flatnonzero(binary[y] > 0)
        if xs.size < min_row_pixels:
            continue
        points.append((float(np.median(xs)), float(y)))
    if len(points) < 2:
        return None

    points = sorted(points, key=lambda p: p[1], reverse=True)
    points = _smooth_centerline(points, window=smooth_window)
    sampled, valid = resample_polyline(points, num_points=num_points)
    if valid.sum() < 2:
        return None

    sampled[:, 0] = sampled[:, 0] / max(float(w), 1.0)
    sampled[:, 1] = sampled[:, 1] / max(float(h), 1.0)
    sampled = np.clip(sampled, 0.0, 1.0).astype(np.float32)
    return sampled, valid.astype(np.float32)


def decode_baseline_result(
    result,
    image_shape: tuple[int, int],
    num_points: int,
    mask_thr: float,
    min_area: int,
    min_row_pixels: int,
    smooth_window: int,
    max_det: int,
) -> tuple[list[dict], list[np.ndarray]]:
    """Decode YOLO segmentation masks into GCS-compatible lane dictionaries."""
    masks = _as_numpy_masks(result)
    if masks is None:
        return [], []

    scores = _prediction_scores(result, masks.shape[0])
    order = np.argsort(-scores)
    if max_det and max_det > 0:
        order = order[: int(max_det)]

    h, w = int(image_shape[0]), int(image_shape[1])
    decoded: list[dict] = []
    decoded_masks: list[np.ndarray] = []
    for rank, idx in enumerate(order.tolist()):
        lane = mask_to_lane_points(
            masks[idx],
            image_shape=image_shape,
            num_points=num_points,
            mask_thr=mask_thr,
            min_area=min_area,
            min_row_pixels=min_row_pixels,
            smooth_window=smooth_window,
        )
        if lane is None:
            continue
        points_norm, _ = lane
        points_px = points_norm * np.array([w, h], dtype=np.float32)
        decoded.append(
            {
                "score": float(scores[idx]),
                "query": int(idx),
                "points_norm": points_norm,
                "points": points_px.astype(np.float32),
                "rank": int(rank),
            }
        )

        mask = masks[idx]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        decoded_masks.append((mask >= float(mask_thr)).astype(np.uint8))
    return decoded, decoded_masks


def draw_baseline_lanes(img: np.ndarray, lanes: list[dict], masks: list[np.ndarray], line_width: int) -> np.ndarray:
    """Render predicted baseline masks plus extracted centerlines."""
    out = img.copy()
    overlay = out.copy()
    for i, mask in enumerate(masks):
        color = np.asarray(MASK_COLORS[i % len(MASK_COLORS)], dtype=np.uint8)
        overlay[mask.astype(bool)] = color
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    return draw_gcs_lanes(out, lanes, show_scores=True, line_width=line_width)


def _json_lane(lane: dict) -> dict:
    """Convert a decoded lane dict to JSON-serializable values."""
    return {
        "query": int(lane.get("query", -1)),
        "rank": int(lane.get("rank", -1)),
        "score": round(float(lane["score"]), 6),
        "points_norm": np.asarray(lane["points_norm"], dtype=float).round(6).tolist(),
        "points": np.asarray(lane["points"], dtype=float).round(2).tolist(),
    }


def run_segmentation_map(
    model: YOLO,
    data: str | Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    half: bool,
    save_dir: Path,
) -> dict:
    """Run ordinary YOLO segmentation validation so mAP(M) can be preserved as an auxiliary metric."""
    metrics = model.val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        half=half,
        split="val",
        plots=False,
        project=str(save_dir),
        name="segmentation_map",
        exist_ok=True,
        verbose=False,
    )
    results = getattr(metrics, "results_dict", {}) or {}
    keep = (
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )
    return {k: round(float(results[k]), 6) for k in keep if k in results}


def evaluate(
    weights: str | Path,
    source: str | Path,
    labels: str | Path | None = DEFAULT_LABELS,
    data: str | Path = DEFAULT_DATA,
    imgsz: int = 960,
    conf: float = 0.25,
    iou: float = 0.7,
    ape_thr: float = 20.0,
    max_det: int = GCS_DEFAULT_MAX_DET,
    max_images: int = 0,
    device: str = "0",
    half: bool = False,
    num_points: int = 32,
    mask_thr: float = 0.5,
    min_area: int = 40,
    min_row_pixels: int = 2,
    smooth_window: int = 7,
    save_dir: str | Path = "runs/baseline/geometry_eval",
    save_json: bool = False,
    save_img: bool = False,
    save_txt: bool = False,
    line_width: int = 2,
    run_map: bool = False,
    map_batch: int = 2,
    map_workers: int = 0,
) -> dict:
    """Evaluate YOLO-seg baseline predictions with unified structured lane metrics."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Baseline weights not found: {weights}")

    model = YOLO(str(weights))
    images = collect_images(source, max_images=max_images)

    save_dir = Path(save_dir)
    image_dir = save_dir / "images"
    label_out_dir = save_dir / "labels"
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_img:
        image_dir.mkdir(parents=True, exist_ok=True)
    if save_txt:
        label_out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_pre = 0.0
    total_infer = 0.0
    total_yolo_post = 0.0
    total_lane_post = 0.0
    total_wall = 0.0
    label_dir = None if labels is None or str(labels).strip() == "" else Path(labels)

    for image_path in images:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        label_path = label_path_for_image(image_path, label_dir)
        gt_lanes, gt_valid = load_gcs_label(label_path)

        t0 = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            half=half,
            retina_masks=True,
            verbose=False,
        )[0]
        wall_s = time.perf_counter() - t0
        speed = getattr(result, "speed", {}) or {}
        pre_s = float(speed.get("preprocess", 0.0)) / 1000.0
        infer_s = float(speed.get("inference", wall_s * 1000.0)) / 1000.0
        yolo_post_s = float(speed.get("postprocess", 0.0)) / 1000.0

        t1 = time.perf_counter()
        lanes, masks = decode_baseline_result(
            result,
            image_shape=img.shape[:2],
            num_points=num_points,
            mask_thr=mask_thr,
            min_area=min_area,
            min_row_pixels=min_row_pixels,
            smooth_window=smooth_window,
            max_det=max_det,
        )
        metrics, matches = match_lanes(lanes, gt_lanes, gt_valid, img.shape[:2], ape_thr=ape_thr)
        lane_post_s = time.perf_counter() - t1

        total_pre += pre_s
        total_infer += infer_s
        total_yolo_post += yolo_post_s
        total_lane_post += lane_post_s
        total_wall += wall_s

        if save_img:
            cv2.imwrite(str(image_dir / image_path.name), draw_baseline_lanes(img, lanes, masks, line_width=line_width))
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
                "preprocess_ms": round(pre_s * 1000.0, 4),
                "inference_ms": round(infer_s * 1000.0, 4),
                "yolo_postprocess_ms": round(yolo_post_s * 1000.0, 4),
                "mask_to_lane_ms": round(lane_post_s * 1000.0, 4),
                "wall_ms": round(wall_s * 1000.0, 4),
                "metrics": metrics,
                "matches": matches,
                "lanes": [_json_lane(x) for x in lanes],
            }
        )

    summary = summarize(
        records,
        total_infer=total_infer,
        total_post=total_yolo_post + total_lane_post,
        ape_thr=ape_thr,
    )
    n = max(len(records), 1)
    summary.update(
        {
            "avg_yolo_preprocess_ms": round(total_pre * 1000.0 / n, 4),
            "avg_yolo_postprocess_ms": round(total_yolo_post * 1000.0 / n, 4),
            "avg_mask_to_lane_ms": round(total_lane_post * 1000.0 / n, 4),
            "avg_wall_ms": round(total_wall * 1000.0 / n, 4),
            "num_points": int(num_points),
            "mask_threshold": float(mask_thr),
            "baseline_note": "YOLO-seg masks are converted to centerline point sequences before APE/F1 evaluation.",
        }
    )

    output = {"summary": summary}
    if run_map:
        output["segmentation_map"] = run_segmentation_map(
            model=model,
            data=data,
            imgsz=imgsz,
            batch=map_batch,
            workers=map_workers,
            device=device,
            half=half,
            save_dir=save_dir,
        )
    if save_json:
        output["records"] = records
    (save_dir / "eval_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if run_map:
        print("segmentation_map:", json.dumps(output["segmentation_map"], indent=2))
    print(f"saved to: {save_dir.resolve()}")
    return output


def main() -> None:
    args = parse_args()
    evaluate(
        weights=args.weights,
        source=args.source,
        labels=args.labels,
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        ape_thr=args.ape_thr,
        max_det=args.max_det,
        max_images=args.max_images,
        device=args.device,
        half=args.half,
        num_points=args.num_points,
        mask_thr=args.mask_thr,
        min_area=args.min_area,
        min_row_pixels=args.min_row_pixels,
        smooth_window=args.smooth_window,
        save_dir=args.save_dir,
        save_json=args.save_json,
        save_img=args.save_img,
        save_txt=args.save_txt,
        line_width=args.line_width,
        run_map=args.run_map,
        map_batch=args.map_batch,
        map_workers=args.map_workers,
    )


if __name__ == "__main__":
    main()
