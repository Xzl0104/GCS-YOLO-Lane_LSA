from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
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

from tools.eval_gcs import (
    assert_label_fixed_y_compatible,
    label_path_for_image,
    load_gcs_label,
    match_lanes,
    model_fixed_y_anchors,
    pair_geometry,
)
from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, assert_gcs_shape, normalize_imgsz, shape_str
from ultralytics.utils.gcs_postprocess import GCS_DEFAULT_MAX_DET, decode_gcs_predictions
from ultralytics.utils.torch_utils import select_device


DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_refquery_e220" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "images" / "test"
DEFAULT_LABELS = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "labels_gcs" / "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GCS exist calibration, strict matching, and error distribution.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt or model yaml.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image file, directory, or txt list.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="labels_gcs directory. Empty means infer from image path.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS inference shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Lane existence threshold to analyze.")
    parser.add_argument(
        "--point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold for fixed-y lane decoding.",
    )
    parser.add_argument("--ape-thr", type=float, default=20.0, help="APE threshold in pixels for TP/good-query labels.")
    parser.add_argument("--match-gate-px", type=float, default=None, help="Strict eval APE gate. Defaults to --ape-thr.")
    parser.add_argument("--max-x-dist", type=float, default=0.0, help="Optional mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--min-overlap", type=int, default=6, help="Minimum overlapping visible anchors for matching.")
    parser.add_argument("--min-points", type=int, default=6, help="Minimum decoded visible anchors required to keep a predicted lane.")
    parser.add_argument("--nms-dist-px", type=float, default=18.0, help="Lane duplicate suppression distance in pixels. 0 disables.")
    parser.add_argument("--max-det", type=int, default=GCS_DEFAULT_MAX_DET, help="Maximum decoded lane queries per image.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images. 0 means all.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or 0.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--save-dir", default="runs/gcs_lane/error_analysis", help="Output directory.")
    return parser.parse_args()


def stat(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def safe_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def score_bin(score: float) -> str:
    lo = min(int(score * 10), 9) / 10.0
    hi = lo + 0.1
    return f"{lo:.1f}-{hi:.1f}"


def ape_bin(ape: float) -> str:
    if not np.isfinite(ape):
        return "inf"
    if ape <= 10:
        return "0-10"
    if ape <= 20:
        return "10-20"
    if ape <= 50:
        return "20-50"
    if ape <= 100:
        return "50-100"
    return ">100"


def best_query_geometry(
    pred_points: np.ndarray,
    gt_lanes: np.ndarray,
    gt_valid: np.ndarray,
    image_shape: tuple[int, int],
    ape_thr: float,
    max_x_dist: float,
    min_overlap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-query best APE, best mean-x distance, overlap, and geometry-good mask."""
    n_query = int(pred_points.shape[0])
    if gt_lanes.shape[0] == 0:
        best_ape = np.full((n_query,), np.inf, dtype=np.float32)
        best_x = np.full((n_query,), np.inf, dtype=np.float32)
        best_overlap = np.zeros((n_query,), dtype=np.int32)
        return best_ape, best_x, best_overlap, np.zeros((n_query,), dtype=bool)

    h, w = int(image_shape[0]), int(image_shape[1])
    scale = np.array([w, h], dtype=np.float32)
    ape, mean_x, overlap = pair_geometry(pred_points, gt_lanes, gt_valid, scale)
    best_idx = np.argmin(ape, axis=1)
    rows = np.arange(n_query)
    best_ape = ape[rows, best_idx]
    best_x = mean_x[rows, best_idx]
    best_overlap = overlap[rows, best_idx]
    good = (best_overlap >= max(int(min_overlap), 0)) & (best_ape <= float(ape_thr))
    if max_x_dist > 0.0:
        good = good & (best_x <= float(max_x_dist))
    return best_ape, best_x, best_overlap, good


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=imgsz)
    expected_fixed_y = model_fixed_y_anchors(model)
    images = collect_images(args.source, max_images=args.max_images)
    label_dir = None if args.labels is None or str(args.labels).strip() == "" else Path(args.labels)

    tp = fp = fn = 0
    records: list[dict] = []
    query_rows: list[dict] = []
    pred_counts: list[int] = []
    gt_counts: list[int] = []
    apes_tp: list[float] = []
    apes_all: list[float] = []
    apes_fp: list[float] = []
    score_good: list[float] = []
    score_bad: list[float] = []
    ape_bin_scores: dict[str, list[float]] = defaultdict(list)
    score_bin_good: dict[str, list[int]] = defaultdict(list)
    count_group = defaultdict(lambda: {"images": 0, "tp": 0, "fp": 0, "fn": 0})
    active_good = active_bad = inactive_good = inactive_bad = 0
    total_infer = 0.0

    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"analyzing {len(images)} images at conf={args.conf:.3f}, ape_thr={args.ape_thr:.1f}")

    for idx, image_path in enumerate(images, start=1):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        assert_gcs_shape(img.shape[:2], imgsz, name="analysis image", context=f"analyze_gcs_errors({image_path})")
        label_path = label_path_for_image(image_path, label_dir)
        assert_label_fixed_y_compatible(label_path, expected_fixed_y, image_shape=img.shape[:2])
        gt_lanes, gt_valid = load_gcs_label(label_path)

        tensor = preprocess_image(img, imgsz=imgsz, device=device, half=args.half)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        preds = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_infer += time.perf_counter() - t0

        pred_points_t = preds["pred_points"][0].detach().float()
        pred_logits_t = preds["pred_logits"][0].detach().float()
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
        scores = pred_logits_t.sigmoid().cpu().numpy().astype(np.float32)
        pred_points = pred_points_t.cpu().numpy().astype(np.float32)

        best_ape, best_x, best_overlap, geom_good = best_query_geometry(
            pred_points,
            gt_lanes,
            gt_valid,
            img.shape[:2],
            ape_thr=args.ape_thr,
            max_x_dist=args.max_x_dist,
            min_overlap=args.min_overlap,
        )

        for q, score in enumerate(scores.tolist()):
            good = bool(geom_good[q])
            active = bool(score >= args.conf)
            score_good.append(float(score)) if good else score_bad.append(float(score))
            if active and good:
                active_good += 1
            elif active and not good:
                active_bad += 1
            elif (not active) and good:
                inactive_good += 1
            else:
                inactive_bad += 1
            ape_bin_scores[ape_bin(float(best_ape[q]))].append(float(score))
            score_bin_good[score_bin(float(score))].append(1 if good else 0)
            query_rows.append(
                {
                    "image": str(image_path.resolve()),
                    "query": int(q),
                    "score": round(float(score), 8),
                    "active": int(active),
                    "best_ape_px": None if not np.isfinite(best_ape[q]) else round(float(best_ape[q]), 4),
                    "best_mean_x_dist_px": None if not np.isfinite(best_x[q]) else round(float(best_x[q]), 4),
                    "best_overlap_points": int(best_overlap[q]),
                    "geometry_good": int(good),
                }
            )

        pred_lanes = decode_gcs_predictions(
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
        metrics, matches = match_lanes(
            pred_lanes,
            gt_lanes,
            gt_valid,
            img.shape[:2],
            ape_thr=args.ape_thr,
            match_gate_px=args.match_gate_px,
            max_x_dist=args.max_x_dist,
            min_overlap=args.min_overlap,
        )

        gt_count = int(gt_lanes.shape[0])
        pred_count = int(len(pred_lanes))
        image_tp, image_fp, image_fn = int(metrics["tp"]), int(metrics["fp"]), int(metrics["fn"])
        tp += image_tp
        fp += image_fp
        fn += image_fn
        pred_counts.append(pred_count)
        gt_counts.append(gt_count)
        apes_tp.extend(float(x) for x in metrics.get("ape_tp", metrics.get("ape", [])))
        apes_all.extend(float(x) for x in metrics.get("ape_matched_all", []))
        apes_fp.extend(float(x) for x in metrics.get("ape_fp_matched", []))
        group = count_group[(gt_count, pred_count)]
        group["images"] += 1
        group["tp"] += image_tp
        group["fp"] += image_fp
        group["fn"] += image_fn

        diag = metrics.get("diagnostic_matches", [])
        fp_diag = [float(x["ape_px"]) for x in diag if not bool(x.get("tp"))]
        records.append(
            {
                "image": str(image_path.resolve()),
                "gt_lanes": gt_count,
                "pred_lanes": pred_count,
                "lane_count_error": int(pred_count - gt_count),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "max_diagnostic_ape_px": None if not diag else round(max(float(x["ape_px"]) for x in diag), 4),
                "fp_diagnostic_ape_mean_px": None if not fp_diag else round(float(np.mean(fp_diag)), 4),
                "scores_desc": ";".join(f"{float(x):.4f}" for x in sorted(scores.tolist(), reverse=True)),
                "active_queries": int((scores >= args.conf).sum()),
                "geometry_good_queries": int(geom_good.sum()),
                "strict_matches": len(matches),
            }
        )

        if idx % 250 == 0 or idx == len(images):
            print(f"processed {idx}/{len(images)}")

    precision, recall, f1 = safe_f1(tp, fp, fn)
    n = max(len(images), 1)
    count_rows = []
    for (gt_count, pred_count), data in sorted(count_group.items()):
        gp, gr, gf1 = safe_f1(int(data["tp"]), int(data["fp"]), int(data["fn"]))
        count_rows.append(
            {
                "gt_lanes": gt_count,
                "pred_lanes": pred_count,
                "images": int(data["images"]),
                "tp": int(data["tp"]),
                "fp": int(data["fp"]),
                "fn": int(data["fn"]),
                "precision": round(float(gp), 6),
                "recall": round(float(gr), 6),
                "f1": round(float(gf1), 6),
            }
        )

    score_bins = {}
    for key in sorted(score_bin_good):
        vals = score_bin_good[key]
        score_bins[key] = {"queries": len(vals), "geometry_good_fraction": round(float(np.mean(vals)), 6)}

    ape_bins = {}
    for key in ["0-10", "10-20", "20-50", "50-100", ">100", "inf"]:
        vals = ape_bin_scores.get(key, [])
        ape_bins[key] = {
            **stat(vals),
            "active_fraction": None if not vals else round(float(np.mean(np.asarray(vals) >= args.conf)), 6),
        }

    summary = {
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
        "strict_eval": {
            "images": len(images),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "fp_per_image": round(float(fp) / n, 6),
            "fn_per_image": round(float(fn) / n, 6),
            "lane_count_mae": round(float(np.mean(np.abs(np.asarray(pred_counts) - np.asarray(gt_counts)))), 6),
            "pred_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(pred_counts).items())},
            "gt_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(gt_counts).items())},
            "gt_pred_lanes_hist": {
                f"{gt}->{pred}": int(v)
                for (gt, pred), v in sorted(Counter(zip(gt_counts, pred_counts)).items())
            },
            "ape_tp": stat(apes_tp),
            "ape_matched_all": stat(apes_all),
            "ape_fp_matched": stat(apes_fp),
            "ape_all_matched": stat(apes_all),
            "fp_matched_ape": stat(apes_fp),
            "avg_inference_ms": round(total_infer * 1000.0 / n, 4),
        },
        "exist_calibration": {
            "query_count": int(len(query_rows)),
            "active_queries": int(active_good + active_bad),
            "geometry_good_queries": int(active_good + inactive_good),
            "active_good": int(active_good),
            "active_bad": int(active_bad),
            "inactive_good": int(inactive_good),
            "inactive_bad": int(inactive_bad),
            "active_geometry_precision": round(active_good / max(active_good + active_bad, 1), 6),
            "geometry_good_active_recall": round(active_good / max(active_good + inactive_good, 1), 6),
            "score_geometry_good": stat(score_good),
            "score_geometry_bad": stat(score_bad),
            "score_bins": score_bins,
            "score_by_ape_bin": ape_bins,
        },
        "top_error_images": sorted(
            [r for r in records if r["fp"] > 0 or r["fn"] > 0 or r["lane_count_error"] != 0],
            key=lambda r: (r["fp"] + r["fn"], abs(r["lane_count_error"]), r["max_diagnostic_ape_px"] or 0),
            reverse=True,
        )[:50],
    }

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(
        save_dir / "error_images.csv",
        sorted(records, key=lambda r: (r["fp"] + r["fn"], abs(r["lane_count_error"]), r["max_diagnostic_ape_px"] or 0), reverse=True),
        [
            "image",
            "gt_lanes",
            "pred_lanes",
            "lane_count_error",
            "tp",
            "fp",
            "fn",
            "max_diagnostic_ape_px",
            "fp_diagnostic_ape_mean_px",
            "scores_desc",
            "active_queries",
            "geometry_good_queries",
            "strict_matches",
        ],
    )
    write_csv(
        save_dir / "query_calibration.csv",
        query_rows,
        [
            "image",
            "query",
            "score",
            "active",
            "best_ape_px",
            "best_mean_x_dist_px",
            "best_overlap_points",
            "geometry_good",
        ],
    )
    write_csv(
        save_dir / "lane_count_groups.csv",
        count_rows,
        ["gt_lanes", "pred_lanes", "images", "tp", "fp", "fn", "precision", "recall", "f1"],
    )

    print(json.dumps({"strict_eval": summary["strict_eval"], "exist_calibration": summary["exist_calibration"]}, indent=2))
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
