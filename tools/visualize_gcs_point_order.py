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

from tools.infer_gcs import load_gcs_model, preprocess_image
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str
from ultralytics.utils.torch_utils import select_device


DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_refquery_e220" / "weights" / "best.pt"
COLORS = (
    (0, 255, 0),
    (0, 200, 255),
    (255, 120, 0),
    (255, 0, 180),
    (80, 180, 255),
    (180, 255, 80),
    (255, 80, 80),
    (180, 120, 255),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT and raw prediction point order with 0..K-1 indices.")
    parser.add_argument("--dataset-root", default="datasets/tusimple_fixed_y_k56_960x544", help="Converted dataset root.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Dataset split.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint. Use empty string to skip predictions.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS input shape as H W. Defaults to the dataset preset.",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Existence confidence threshold for raw predictions.")
    parser.add_argument("--max-det", type=int, default=8, help="Maximum raw prediction lanes to draw.")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum images to visualize.")
    parser.add_argument("--device", default="cpu", help="Inference device.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--save-dir", default="runs/gcs_lane/point_order_vis", help="Output directory.")
    return parser.parse_args()


def load_label(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(label_path, allow_pickle=False) as data:
        lanes = data["lanes"].astype(np.float32)
        valid = data["lane_valid"].astype(np.float32)
    return lanes, valid


def put_title(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_index_box(image: np.ndarray, point: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    x, y = point
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.32
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0 = int(np.clip(x + 3, 0, image.shape[1] - tw - 2))
    y0 = int(np.clip(y - 3, th + 2, image.shape[0] - baseline - 2))
    cv2.rectangle(image, (x0 - 1, y0 - th - 1), (x0 + tw + 1, y0 + baseline + 1), (0, 0, 0), -1)
    cv2.putText(image, text, (x0, y0), font, scale, color, thickness, cv2.LINE_AA)


def draw_ordered_lanes(
    image: np.ndarray,
    lanes: np.ndarray,
    valid: np.ndarray | None,
    title: str,
    scores: np.ndarray | None = None,
    query_ids: np.ndarray | None = None,
) -> np.ndarray:
    h, w = image.shape[:2]
    out = put_title(image, title)
    for lane_idx, lane in enumerate(lanes):
        if valid is None:
            mask = np.ones((lane.shape[0],), dtype=bool)
        else:
            mask = valid[lane_idx] > 0.5
        pts_norm = lane[mask]
        orig_indices = np.flatnonzero(mask)
        if pts_norm.shape[0] < 1:
            continue

        pts = pts_norm * np.array([w, h], dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts_i = np.round(pts).astype(np.int32)
        color = COLORS[lane_idx % len(COLORS)]
        if pts_i.shape[0] >= 2:
            cv2.polylines(out, [pts_i], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
            for i in range(pts_i.shape[0] - 1):
                if i % 4 == 0:
                    cv2.arrowedLine(
                        out,
                        tuple(int(x) for x in pts_i[i]),
                        tuple(int(x) for x in pts_i[i + 1]),
                        color,
                        1,
                        tipLength=0.28,
                    )

        for point_idx, point in zip(orig_indices, pts_i):
            cv2.circle(out, tuple(int(x) for x in point), 3, color, -1, lineType=cv2.LINE_AA)
            draw_index_box(out, tuple(int(x) for x in point), str(int(point_idx)), color)

        label_parts = []
        if query_ids is not None:
            label_parts.append(f"q{int(query_ids[lane_idx])}")
        else:
            label_parts.append(f"lane{lane_idx}")
        if scores is not None:
            label_parts.append(f"{float(scores[lane_idx]):.2f}")
        x0, y0 = (int(pts_i[0, 0]), max(45, int(pts_i[0, 1]) - 8))
        cv2.putText(out, " ".join(label_parts), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


@torch.inference_mode()
def predict_raw_lanes(
    model: torch.nn.Module,
    image: np.ndarray,
    imgsz: tuple[int, int],
    device: torch.device,
    half: bool,
    conf: float,
    max_det: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    tensor = preprocess_image(image, imgsz=imgsz, device=device, half=half)
    preds = model(tensor)
    if not isinstance(preds, dict) or "pred_points" not in preds or "pred_logits" not in preds:
        raise ValueError("GCS model output must contain pred_points and pred_logits.")

    points = preds["pred_points"][0].detach().float().cpu().clamp(0.0, 1.0)
    logits = preds["pred_logits"][0].detach().float().cpu()
    if logits.ndim == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    scores = logits.sigmoid()
    keep = torch.nonzero(scores >= float(conf), as_tuple=False).flatten()
    if keep.numel() == 0:
        return (
            np.zeros((0, points.shape[1], 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )

    order = keep[torch.argsort(scores[keep], descending=True)]
    if max_det and max_det > 0:
        order = order[: int(max_det)]
    kept_points = points[order].numpy().astype(np.float32)
    kept_scores = scores[order].numpy().astype(np.float32)
    kept_queries = order.numpy().astype(np.int64)

    monotonic = []
    for lane, score, query in zip(kept_points, kept_scores, kept_queries):
        ys = lane[:, 1]
        monotonic.append(
            {
                "query": int(query),
                "score": float(score),
                "bottom_to_top": bool(np.all(np.diff(ys) <= 1e-6)),
            }
        )
    return kept_points, kept_scores, kept_queries, monotonic


def main() -> None:
    args = parse_args()
    dataset_root = ROOT / args.dataset_root if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels_gcs" / args.split
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    save_dir = (ROOT / args.save_dir if not Path(args.save_dir).is_absolute() else Path(args.save_dir)) / args.split
    save_dir.mkdir(parents=True, exist_ok=True)

    model = None
    device = select_device(args.device, verbose=False)
    weights = str(args.weights).strip()
    if weights:
        model = load_gcs_model(weights, device=device, half=args.half, gcs_imgsz=imgsz)

    image_paths = sorted(image_dir.glob("*.jpg"))[: int(args.max_images)]
    records: list[dict] = []
    written = 0
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.npz"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        gt_lanes, gt_valid = load_label(label_path)
        gt_panel = draw_ordered_lanes(image, gt_lanes, gt_valid, "GT point order")
        panels = [put_title(image, image_path.name), gt_panel]

        pred_count = 0
        pred_monotonic: list[dict] = []
        if model is not None:
            pred_lanes, pred_scores, pred_queries, pred_monotonic = predict_raw_lanes(
                model=model,
                image=image,
                imgsz=imgsz,
                device=device,
                half=args.half,
                conf=args.conf,
                max_det=args.max_det,
            )
            pred_count = int(pred_lanes.shape[0])
            pred_panel = draw_ordered_lanes(
                image,
                pred_lanes,
                valid=None,
                title="Raw pred point order",
                scores=pred_scores,
                query_ids=pred_queries,
            )
            panels.append(pred_panel)

        canvas = np.concatenate(panels, axis=1)
        cv2.imwrite(str(save_dir / f"{image_path.stem}.jpg"), canvas)
        records.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "gt_lanes": int(gt_lanes.shape[0]),
                "pred_lanes": pred_count,
                "raw_pred_monotonic": pred_monotonic,
            }
        )
        written += 1

    (save_dir / "point_order_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved images: {written}")
    print(f"output: {save_dir}")


if __name__ == "__main__":
    main()
