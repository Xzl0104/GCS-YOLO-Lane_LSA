from __future__ import annotations

import argparse
import csv
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

from tools.eval_gcs import (
    assert_label_fixed_y_compatible,
    gated_assignment,
    label_path_for_image,
    load_gcs_label,
    match_lanes,
    model_fixed_y_anchors,
    pair_geometry,
)
from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image
from ultralytics.utils.gcs_postprocess import GCS_DEFAULT_MAX_DET, decode_gcs_predictions, sort_lane_bottom_to_top
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, assert_gcs_shape, normalize_imgsz, shape_str
from ultralytics.utils.torch_utils import select_device


DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_strict_exist_ft" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "images" / "test"
DEFAULT_LABELS = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "labels_gcs" / "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run oracle-count and oracle-geometry diagnostics for GCS lanes.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image file, directory, or txt list.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="labels_gcs directory. Empty means infer from image path.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS input shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Baseline confidence threshold.")
    parser.add_argument(
        "--point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold for fixed-y baseline decoding.",
    )
    parser.add_argument("--ape-thr", type=float, default=20.0, help="APE threshold in pixels for TP matching.")
    parser.add_argument("--match-gate-px", type=float, default=None, help="Strict eval APE gate. Defaults to --ape-thr.")
    parser.add_argument("--max-x-dist", type=float, default=0.0, help="Optional strict mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--min-overlap", type=int, default=6, help="Minimum overlapping visible anchors for matching.")
    parser.add_argument("--min-points", type=int, default=6, help="Minimum decoded visible anchors required to keep a baseline predicted lane.")
    parser.add_argument("--nms-dist-px", type=float, default=18.0, help="Baseline Lane-NMS threshold. 0 disables.")
    parser.add_argument("--max-det", type=int, default=GCS_DEFAULT_MAX_DET, help="Maximum decoded baseline lanes.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images. 0 means all.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--save-dir", default="runs/gcs_lane/oracle_analysis", help="Output directory.")
    return parser.parse_args()


def empty_state() -> dict:
    return {
        "images": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "apes": [],
        "apes_matched_all": [],
        "apes_fp_matched": [],
        "pred_counts": [],
        "gt_counts": [],
    }


def update_state(state: dict, metrics: dict, pred_count: int, gt_count: int) -> None:
    state["images"] += 1
    state["tp"] += int(metrics["tp"])
    state["fp"] += int(metrics["fp"])
    state["fn"] += int(metrics["fn"])
    state["apes"].extend(float(x) for x in metrics.get("ape_tp", metrics.get("ape", [])))
    state["apes_matched_all"].extend(float(x) for x in metrics.get("ape_matched_all", []))
    state["apes_fp_matched"].extend(float(x) for x in metrics.get("ape_fp_matched", []))
    state["pred_counts"].append(int(pred_count))
    state["gt_counts"].append(int(gt_count))


def summarize_state(state: dict, ape_thr: float) -> dict:
    tp, fp, fn = int(state["tp"]), int(state["fp"]), int(state["fn"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    apes = np.asarray(state["apes"], dtype=np.float32)
    apes_all = np.asarray(state["apes_matched_all"], dtype=np.float32)
    apes_fp = np.asarray(state["apes_fp_matched"], dtype=np.float32)
    pred_counts = state["pred_counts"]
    gt_counts = state["gt_counts"]
    n = max(int(state["images"]), 1)
    return {
        "images": int(state["images"]),
        "ape_threshold_px": float(ape_thr),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_per_image": round(float(fp) / n, 6),
        "fn_per_image": round(float(fn) / n, 6),
        "lane_count_mae": round(float(sum(abs(p - g) for p, g in zip(pred_counts, gt_counts))) / n, 6),
        "ape_tp_mean_px": None if apes.size == 0 else round(float(apes.mean()), 4),
        "ape_matched_all_mean_px": None if apes_all.size == 0 else round(float(apes_all.mean()), 4),
        "ape_fp_matched_mean_px": None if apes_fp.size == 0 else round(float(apes_fp.mean()), 4),
    }


def lanes_from_indices(
    points: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    image_shape: tuple[int, int],
) -> list[dict]:
    h, w = int(image_shape[0]), int(image_shape[1])
    scale = np.array([w, h], dtype=np.float32).reshape(1, 2)
    lanes = []
    for idx in indices.tolist():
        lane_norm = points[int(idx)].astype(np.float32)
        lanes.append(
            {
                "query": int(idx),
                "score": float(scores[int(idx)]),
                "points_norm": lane_norm,
                "points": lane_norm * scale,
            }
        )
    return lanes


def oracle_geometry_indices(
    points: np.ndarray,
    gt_lanes: np.ndarray,
    gt_valid: np.ndarray,
    image_shape: tuple[int, int],
    ape_thr: float,
    match_gate_px: float | None,
    max_x_dist: float,
    min_overlap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] == 0 or gt_lanes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    h, w = int(image_shape[0]), int(image_shape[1])
    scale = np.array([w, h], dtype=np.float32)
    ape, mean_x, overlap = pair_geometry(points, gt_lanes, gt_valid, scale)
    gate_px = float(ape_thr if match_gate_px is None else match_gate_px)
    gate = overlap >= max(int(min_overlap), 0)
    if gate_px > 0.0:
        gate = gate & (ape <= gate_px)
    if max_x_dist > 0.0:
        gate = gate & (mean_x <= float(max_x_dist))
    rows, cols = gated_assignment(ape, gate=gate)
    keep = ape[rows, cols] < float(ape_thr)
    rows, cols = rows[keep], cols[keep]
    return rows, cols, ape[rows, cols].astype(np.float32)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=imgsz)
    expected_fixed_y = model_fixed_y_anchors(model)
    images = collect_images(args.source, max_images=args.max_images)
    label_dir = None if args.labels is None or str(args.labels).strip() == "" else Path(args.labels)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    baseline = empty_state()
    oracle_count = empty_state()
    oracle_geometry = empty_state()
    per_image_rows = []

    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"oracle analyzing {len(images)} images at conf={args.conf:.3f}, ape_thr={args.ape_thr:.1f}")

    for image_path in images:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        assert_gcs_shape(img.shape[:2], imgsz, name="oracle image", context=f"analyze_gcs_oracle({image_path})")
        label_path = label_path_for_image(image_path, label_dir)
        assert_label_fixed_y_compatible(label_path, expected_fixed_y, image_shape=img.shape[:2])
        gt_lanes, gt_valid = load_gcs_label(label_path)
        gt_count = int(gt_lanes.shape[0])

        tensor = preprocess_image(img, imgsz=imgsz, device=device, half=args.half)
        preds = model(tensor)
        pred_points_t = preds["pred_points"][0].detach().float().cpu().clamp(0.0, 1.0)
        pred_logits_t = preds["pred_logits"][0].detach().float().cpu()
        pred_valid_t = preds.get("pred_valid_logits")
        pred_valid_t = pred_valid_t[0].detach().float().cpu() if pred_valid_t is not None else None
        pred_count_t = preds.get("pred_count_logits")
        pred_count_t = pred_count_t[0].detach().float().cpu() if pred_count_t is not None else None
        pred_count_boundary_t = preds.get("pred_count_boundary_logits")
        pred_count_boundary_t = (
            pred_count_boundary_t[0].detach().float().cpu() if pred_count_boundary_t is not None else None
        )
        pred_quality_t = preds.get("pred_quality_logits")
        pred_quality_t = pred_quality_t[0].detach().float().cpu() if pred_quality_t is not None else None
        if pred_logits_t.ndim == 2 and pred_logits_t.shape[-1] == 1:
            pred_logits_t = pred_logits_t.squeeze(-1)
        scores = pred_logits_t.sigmoid().numpy().astype(np.float32)
        sorted_points = torch.stack([sort_lane_bottom_to_top(x) for x in pred_points_t], dim=0).numpy().astype(np.float32)

        baseline_lanes = decode_gcs_predictions(
            pred_points_t,
            pred_logits_t,
            pred_valid_logits=pred_valid_t,
            pred_count_logits=pred_count_t,
            pred_count_boundary_logits=pred_count_boundary_t,
            pred_quality_logits=pred_quality_t,
            image_shape=img.shape[:2],
            score_thr=args.conf,
            point_valid_thr=args.point_valid_thr,
            min_points=args.min_points,
            max_det=args.max_det,
            nms_dist_px=args.nms_dist_px,
            candidate_score_thr=args.conf,
            candidate_point_valid_thr=args.point_valid_thr,
            line_nms_min_overlap=6,
        )
        baseline_metrics, _ = match_lanes(
            baseline_lanes,
            gt_lanes,
            gt_valid,
            img.shape[:2],
            ape_thr=args.ape_thr,
            match_gate_px=args.match_gate_px,
            max_x_dist=args.max_x_dist,
            min_overlap=args.min_overlap,
        )
        update_state(baseline, baseline_metrics, len(baseline_lanes), gt_count)

        topk = min(gt_count, sorted_points.shape[0])
        count_lanes = (
            decode_gcs_predictions(
                pred_points_t,
                pred_logits_t,
                pred_valid_logits=pred_valid_t,
                pred_count_boundary_logits=pred_count_boundary_t,
                pred_quality_logits=pred_quality_t,
                image_shape=img.shape[:2],
                score_thr=0.0,
                point_valid_thr=args.point_valid_thr,
                min_points=args.min_points,
                max_det=topk,
                nms_dist_px=0.0,
                use_count_head_decode=False,
                candidate_score_thr=0.0,
                candidate_point_valid_thr=args.point_valid_thr,
            )
            if topk > 0
            else []
        )
        count_metrics, _ = match_lanes(
            count_lanes,
            gt_lanes,
            gt_valid,
            img.shape[:2],
            ape_thr=args.ape_thr,
            match_gate_px=args.match_gate_px,
            max_x_dist=args.max_x_dist,
            min_overlap=args.min_overlap,
        )
        update_state(oracle_count, count_metrics, len(count_lanes), gt_count)

        geom_rows, geom_cols, geom_apes = oracle_geometry_indices(
            sorted_points,
            gt_lanes,
            gt_valid,
            img.shape[:2],
            ape_thr=args.ape_thr,
            match_gate_px=args.match_gate_px,
            max_x_dist=args.max_x_dist,
            min_overlap=args.min_overlap,
        )
        geometry_lanes = lanes_from_indices(sorted_points, scores, geom_rows, img.shape[:2])
        geometry_metrics, _ = match_lanes(
            geometry_lanes,
            gt_lanes,
            gt_valid,
            img.shape[:2],
            ape_thr=args.ape_thr,
            match_gate_px=args.match_gate_px,
            max_x_dist=args.max_x_dist,
            min_overlap=args.min_overlap,
        )
        update_state(oracle_geometry, geometry_metrics, len(geometry_lanes), gt_count)

        per_image_rows.append(
            {
                "image": str(image_path.resolve()),
                "gt_lanes": gt_count,
                "baseline_pred": len(baseline_lanes),
                "baseline_tp": int(baseline_metrics["tp"]),
                "baseline_fp": int(baseline_metrics["fp"]),
                "baseline_fn": int(baseline_metrics["fn"]),
                "oracle_count_pred": len(count_lanes),
                "oracle_count_tp": int(count_metrics["tp"]),
                "oracle_count_fp": int(count_metrics["fp"]),
                "oracle_count_fn": int(count_metrics["fn"]),
                "oracle_geometry_pred": len(geometry_lanes),
                "oracle_geometry_tp": int(geometry_metrics["tp"]),
                "oracle_geometry_fp": int(geometry_metrics["fp"]),
                "oracle_geometry_fn": int(geometry_metrics["fn"]),
                "oracle_geometry_apes": ";".join(f"{float(x):.3f}" for x in geom_apes.tolist()),
                "oracle_geometry_queries": ";".join(str(int(x)) for x in geom_rows.tolist()),
                "oracle_geometry_gts": ";".join(str(int(x)) for x in geom_cols.tolist()),
            }
        )

    output = {
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "source": str(Path(args.source).resolve()),
            "labels": None if label_dir is None else str(label_dir.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(args.conf),
            "ape_threshold_px": float(args.ape_thr),
            "match_gate_px": float(args.ape_thr if args.match_gate_px is None else args.match_gate_px),
            "max_x_dist": float(args.max_x_dist),
            "min_overlap": int(args.min_overlap),
            "min_points": int(args.min_points),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
        },
        "baseline": summarize_state(baseline, args.ape_thr),
        "oracle_count": summarize_state(oracle_count, args.ape_thr),
        "oracle_geometry": summarize_state(oracle_geometry, args.ape_thr),
    }
    (save_dir / "oracle_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (save_dir / "oracle_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()) if per_image_rows else [])
        if per_image_rows:
            writer.writeheader()
            writer.writerows(per_image_rows)

    print(json.dumps(output, indent=2))
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
