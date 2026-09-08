from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.eval_gcs import (  # noqa: E402
    CULANE_IOU_THRESHOLD,
    CULANE_LANE_WIDTH,
    CULANE_RAW_SHAPE,
    culane_spline_interp_points,
    load_culane_ground_truth,
    match_culane_lanes,
)
from tools.eval_gcs import label_path_for_image  # noqa: E402
from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image  # noqa: E402
from ultralytics.models.gcs.mode_utils import resolve_decode_mode  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import assert_gcs_shape, normalize_imgsz  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


GT_COLOR = (0, 210, 0)
TP_COLOR = (255, 80, 0)
FP_COLOR = (0, 0, 255)
FN_COLOR = (0, 215, 255)
TEXT_COLOR = (235, 235, 235)
HEADER_COLOR = (28, 28, 28)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize CULane GT/TP/FP/FN lane matches on prediction images."
    )
    parser.add_argument("--weights", required=True, help="GCS checkpoint .pt or model yaml.")
    parser.add_argument("--source", required=True, help="Image file, directory, or txt list.")
    parser.add_argument("--labels", required=True, help="Converted CULane labels_gcs directory.")
    parser.add_argument("--archive-root", required=True, help="Original CULane root.")
    parser.add_argument("--imgsz", nargs=2, type=int, default=(544, 960), metavar=("H", "W"))
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--point-valid-thr", type=float, default=0.5)
    parser.add_argument("--nms-dist-px", type=float, default=50.0)
    parser.add_argument("--max-det", type=int, default=8)
    parser.add_argument("--lane-width", type=int, default=CULANE_LANE_WIDTH)
    parser.add_argument("--iou-thr", type=float, default=CULANE_IOU_THRESHOLD)
    parser.add_argument("--raw-imgsz", nargs=2, type=int, default=CULANE_RAW_SHAPE, metavar=("H", "W"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--line-width", type=int, default=4)
    return parser.parse_args()


def draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    label: str | None = None,
) -> None:
    """Draw one lane polyline and an optional label in display coordinates."""
    if points.ndim != 2 or points.shape[0] < 2:
        return
    height, width = image.shape[:2]
    points = points.astype(np.float32, copy=True)
    points[:, 0] = np.clip(points[:, 0], 0.0, float(width - 1))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(height - 1))
    points_i = np.rint(points).astype(np.int32)
    cv2.polylines(
        image,
        [points_i],
        isClosed=False,
        color=color,
        thickness=max(int(thickness), 1),
        lineType=cv2.LINE_AA,
    )
    if label:
        x, y = points_i[0].tolist()
        cv2.putText(
            image,
            label,
            (int(x), max(48, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def gt_points_for_display(
    lane: list[tuple[float, float]],
    display_shape: tuple[int, int],
    raw_shape: tuple[int, int],
) -> np.ndarray:
    """Interpolate a raw CULane lane and scale it to the converted image."""
    raw_points = culane_spline_interp_points(lane, samples_per_segment=50)
    raw_h, raw_w = int(raw_shape[0]), int(raw_shape[1])
    display_h, display_w = int(display_shape[0]), int(display_shape[1])
    return raw_points * np.asarray([display_w / raw_w, display_h / raw_h], dtype=np.float64)


def pred_points_for_display(
    lane: dict,
    display_shape: tuple[int, int],
) -> np.ndarray:
    """Convert normalized decoded prediction points to display pixels."""
    points = np.asarray(lane.get("visible_points_norm", lane["points_norm"]), dtype=np.float32)
    display_h, display_w = int(display_shape[0]), int(display_shape[1])
    return points * np.asarray([display_w, display_h], dtype=np.float32)


def draw_legend(image: np.ndarray, stats: dict[str, int]) -> None:
    """Draw a compact legend without covering the roadway below the header."""
    height, width = image.shape[:2]
    header_h = 38
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, header_h), HEADER_COLOR, -1)
    cv2.addWeighted(overlay, 0.88, image, 0.12, 0.0, dst=image)
    entries = (
        ("GT", GT_COLOR),
        ("TP", TP_COLOR),
        ("FP", FP_COLOR),
        ("FN", FN_COLOR),
    )
    x = 8
    for name, color in entries:
        cv2.line(image, (x, 19), (x + 24, 19), color, 4, cv2.LINE_AA)
        cv2.putText(image, name, (x + 30, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)
        x += 78
    count_text = f"TP={stats['tp']} FP={stats['fp']} FN={stats['fn']}"
    cv2.putText(image, count_text, (max(x, width - 235), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)


def render_match_image(
    image: np.ndarray,
    pred_lanes: list[dict],
    gt_lanes: list[list[tuple[float, float]]],
    matches: list[dict],
    raw_shape: tuple[int, int],
    lane_width: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Render GT, TP predictions, FP predictions, and FN GT lanes."""
    output = image.copy()
    display_shape = output.shape[:2]
    matched_pred_to_gt = {int(item["pred"]): item for item in matches if item["tp"]}
    matched_gt = {int(item["gt"]) for item in matches if item["tp"]}
    matched_pred_indices = {int(item["pred"]) for item in matches}

    for gt_index, gt_lane in enumerate(gt_lanes):
        points = gt_points_for_display(gt_lane, display_shape=display_shape, raw_shape=raw_shape)
        draw_polyline(output, points, GT_COLOR, thickness=max(lane_width // 5, 2))
        if gt_index not in matched_gt:
            draw_polyline(output, points, FN_COLOR, thickness=max(lane_width // 4, 3), label="FN")

    for pred_index, pred_lane in enumerate(pred_lanes):
        points = pred_points_for_display(pred_lane, display_shape=display_shape)
        match = matched_pred_to_gt.get(pred_index)
        if match is not None:
            label = f"TP {float(match['culane_iou']):.2f}"
            draw_polyline(output, points, TP_COLOR, thickness=4, label=label)
        elif pred_index not in matched_pred_indices:
            label = f"FP {float(pred_lane['score']):.2f}"
            draw_polyline(output, points, FP_COLOR, thickness=4, label=label)
        else:
            label = f"FP {float(pred_lane['score']):.2f}"
            draw_polyline(output, points, FP_COLOR, thickness=4, label=label)

    stats = {
        "tp": len(matched_pred_to_gt),
        "fp": len(pred_lanes) - len(matched_pred_to_gt),
        "fn": len(gt_lanes) - len(matched_gt),
    }
    draw_legend(output, stats)
    return output, stats


@torch.inference_mode()
def visualize(args: argparse.Namespace) -> dict:
    imgsz = normalize_imgsz(args.imgsz)
    raw_shape = tuple(int(x) for x in args.raw_imgsz)
    if raw_shape != CULANE_RAW_SHAPE:
        raise ValueError(f"CULane raw image shape must be {CULANE_RAW_SHAPE}, got {raw_shape}.")
    if int(args.lane_width) <= 0:
        raise ValueError(f"--lane-width must be positive, got {args.lane_width}")
    if not 0.0 <= float(args.iou_thr) <= 1.0:
        raise ValueError(f"--iou-thr must be in [0, 1], got {args.iou_thr}")

    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=imgsz)
    if resolve_decode_mode("auto", model) != "query":
        raise ValueError("visualize_culane_matches.py currently expects a query-mode GCS model.")
    images = collect_images(args.source, max_images=args.max_images)
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.warmup > 0:
        warm_image = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if warm_image is None:
            raise FileNotFoundError(f"Failed to read warmup image: {images[0]}")
        warm_tensor = preprocess_image(warm_image, imgsz=imgsz, device=device, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    summary = {
        "images": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "metric_name": "culane_iou",
        "culane_iou_threshold": float(args.iou_thr),
        "culane_lane_width": int(args.lane_width),
        "raw_image_shape_hw": list(raw_shape),
        "imgsz_hw": list(imgsz),
        "weights": str(Path(args.weights).resolve()),
    }
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        assert_gcs_shape(
            image.shape[:2],
            (384, 960),
            name="CULane display image",
            context=f"visualize_culane_matches({image_path})",
        )
        label_path = label_path_for_image(image_path, args.labels)
        gt_lanes = load_culane_ground_truth(label_path, args.archive_root, raw_shape=raw_shape)
        tensor = preprocess_image(image, imgsz=imgsz, device=device, half=args.half)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        predictions = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        lanes = decode_gcs_predictions(
            predictions["pred_points"][0],
            predictions["pred_logits"][0],
            pred_valid_logits=predictions.get("pred_valid_logits", None)[0]
            if predictions.get("pred_valid_logits", None) is not None
            else None,
            image_shape=image.shape[:2],
            score_thr=float(args.conf),
            point_valid_thr=float(args.point_valid_thr),
            max_det=int(args.max_det),
            nms_dist_px=float(args.nms_dist_px),
        )
        metrics, matches = match_culane_lanes(
            lanes,
            gt_lanes,
            raw_shape=raw_shape,
            lane_width=int(args.lane_width),
            iou_threshold=float(args.iou_thr),
        )
        rendered, stats = render_match_image(
            image,
            lanes,
            gt_lanes,
            matches,
            raw_shape=raw_shape,
            lane_width=int(args.lane_width),
        )
        output_path = image_dir / image_path.name
        if not cv2.imwrite(str(output_path), rendered):
            raise OSError(f"Failed to write visualization: {output_path}")
        summary["images"] += 1
        summary["tp"] += int(stats["tp"])
        summary["fp"] += int(stats["fp"])
        summary["fn"] += int(stats["fn"])

    summary["precision"] = summary["tp"] / max(summary["tp"] + summary["fp"], 1)
    summary["recall"] = summary["tp"] / max(summary["tp"] + summary["fn"], 1)
    summary["f1"] = (
        2.0 * summary["precision"] * summary["recall"] / max(summary["precision"] + summary["recall"], 1e-12)
    )
    summary_path = output_dir / "visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved to: {output_dir.resolve()}")
    return summary


def main() -> None:
    visualize(parse_args())


if __name__ == "__main__":
    main()
