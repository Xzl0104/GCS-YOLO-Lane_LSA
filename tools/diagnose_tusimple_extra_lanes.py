from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    DEFAULT_OFFICIAL_SCORE_FN_WEIGHT,
    DEFAULT_OFFICIAL_SCORE_FP_WEIGHT,
    TuSimpleOfficialLaneEval,
    default_tusimple_gt_json,
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    official_metric_score,
    read_tusimple_json_lines,
    tusimple_image_path,
    valid_tusimple_lanes,
    write_tusimple_predictions,
)
from tools.eval_tusimple_official import _count_diagnostics  # noqa: E402
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = (
    ROOT
    / "runs"
    / "gcs_lane"
    / "gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03"
    / "weights"
    / "best.pt"
)
DEFAULT_COUNT_PAIRS = ("3->4", "3->5", "4->5", "5->4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose TuSimple decoded count errors and dump per-extra-lane "
            "score, visibility, nearest-GT distance, nearest-pred distance, "
            "and duplicate/spurious labels."
        )
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="TuSimple split.")
    parser.add_argument("--gt-json", default=None, help="TuSimple GT json-lines file.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.01, help="Lane existence confidence threshold.")
    parser.add_argument("--point-valid-thr", type=float, default=0.45, help="Point-valid threshold.")
    parser.add_argument("--nms-dist-px", type=float, default=30.0, help="Lane-NMS distance in original-image px.")
    parser.add_argument("--max-det", type=int, default=5, help="Maximum decoded lanes.")
    parser.add_argument("--min-points", type=int, default=2, help="Minimum visible points per decoded lane.")
    parser.add_argument("--count-pairs", nargs="+", default=list(DEFAULT_COUNT_PAIRS), help="Count pairs to list/visualize.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit GT records before filtering. 0 means all.")
    parser.add_argument("--match-overlap", type=int, default=3, help="Minimum shared h_samples for strict GT/pred matching.")
    parser.add_argument("--match-x-thr", type=float, default=20.0, help="Strict GT/pred mean-x match threshold in px.")
    parser.add_argument("--duplicate-overlap", type=int, default=3, help="Minimum shared h_samples for duplicate-like labels.")
    parser.add_argument(
        "--duplicate-x-thr",
        type=float,
        default=50.0,
        help="Mean-x distance threshold in px for possible duplicate/near-duplicate extra lanes.",
    )
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup forwards on the first image.")
    parser.add_argument("--vis-limit-per-pair", type=int, default=50, help="Visualization cap for each selected count pair.")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization output.")
    parser.add_argument("--save-dir", default=None, help="Output directory.")
    parser.add_argument("--runtime-ms", type=float, default=1.0, help="Constant TuSimple run_time for prediction export.")
    parser.add_argument("--score-fp-weight", type=float, default=DEFAULT_OFFICIAL_SCORE_FP_WEIGHT)
    parser.add_argument("--score-fn-weight", type=float, default=DEFAULT_OFFICIAL_SCORE_FN_WEIGHT)
    return parser.parse_args()


def parse_count_pairs(values: list[str]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if "->" not in text:
            raise ValueError(f"Count pair must use A->B format, got {value!r}.")
        left, right = text.split("->", 1)
        pairs.add((int(left.strip()), int(right.strip())))
    if not pairs:
        raise ValueError("--count-pairs must contain at least one A->B pair.")
    return pairs


def resolve_save_dir(args: argparse.Namespace, weights: Path) -> Path:
    if args.save_dir is not None and str(args.save_dir).strip():
        return Path(args.save_dir)
    tag = (
        f"extra_lane_diag_{args.split}_conf{float(args.conf):.4g}_"
        f"pvalid{float(args.point_valid_thr):.4g}_nms{float(args.nms_dist_px):.4g}_"
        f"maxdet{int(args.max_det)}_minp{int(args.min_points)}"
    ).replace(".", "p")
    if weights.parent.name == "weights":
        return weights.parent.parent / tag
    return ROOT / "runs" / "gcs_lane" / tag


def limit_records(records: list[dict], max_images: int) -> list[dict]:
    if max_images and max_images > 0:
        return records[: int(max_images)]
    return records


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_name(raw_file: str) -> str:
    return raw_file.strip("/").replace("/", "__").replace("\\", "__").replace(":", "_")


def lane_pair_stats(a: list[float], b: list[float]) -> tuple[int, float]:
    if len(a) != len(b):
        raise ValueError(f"lane lengths differ: {len(a)} vs {len(b)}")
    diffs = [abs(float(xa) - float(xb)) for xa, xb in zip(a, b) if float(xa) >= 0.0 and float(xb) >= 0.0]
    if not diffs:
        return 0, math.inf
    return len(diffs), float(sum(diffs) / len(diffs))


def nearest_lane_stats(
    lane: list[float],
    lanes: list[list[float]],
    min_overlap: int,
    skip_idx: int | None = None,
) -> dict:
    best: dict | None = None
    for idx, other in enumerate(lanes):
        if skip_idx is not None and int(idx) == int(skip_idx):
            continue
        overlap, mean_abs_x = lane_pair_stats(lane, other)
        if overlap < int(min_overlap) or not math.isfinite(mean_abs_x):
            continue
        row = {
            "lane_idx": int(idx),
            "overlap": int(overlap),
            "mean_abs_x_error": float(mean_abs_x),
        }
        if best is None or (float(row["mean_abs_x_error"]), -int(row["overlap"]), int(row["lane_idx"])) < (
            float(best["mean_abs_x_error"]),
            -int(best["overlap"]),
            int(best["lane_idx"]),
        ):
            best = row
    return best or {"lane_idx": None, "overlap": 0, "mean_abs_x_error": math.inf}


def match_lanes(
    pred_lanes: list[list[float]],
    gt_lanes: list[list[float]],
    min_overlap: int,
    x_thr: float,
) -> tuple[list[dict], set[int], set[int]]:
    candidates: list[tuple[float, int, int, int]] = []
    for pred_idx, pred_lane in enumerate(pred_lanes):
        for gt_idx, gt_lane in enumerate(gt_lanes):
            overlap, mean_abs_x = lane_pair_stats(pred_lane, gt_lane)
            if overlap >= int(min_overlap) and mean_abs_x <= float(x_thr):
                candidates.append((mean_abs_x, -overlap, pred_idx, gt_idx))
    candidates.sort()

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[dict] = []
    for mean_abs_x, neg_overlap, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(int(pred_idx))
        matched_gt.add(int(gt_idx))
        matches.append(
            {
                "pred_idx": int(pred_idx),
                "gt_idx": int(gt_idx),
                "overlap": int(-neg_overlap),
                "mean_abs_x_error": round(float(mean_abs_x), 6),
            }
        )
    return matches, matched_pred, matched_gt


def valid_point_count(lane: dict) -> int:
    if "point_valid" in lane:
        return int(np.asarray(lane["point_valid"], dtype=float).sum())
    if "visible_points" in lane:
        return int(np.asarray(lane["visible_points"]).shape[0])
    if "visible_points_norm" in lane:
        return int(np.asarray(lane["visible_points_norm"]).shape[0])
    return int(np.asarray(lane["points_norm"]).shape[0])


def convert_decoded_lanes(lanes: list[dict], h_samples: list[int] | list[float], image_shape: tuple[int, int]) -> list[dict]:
    items: list[dict] = []
    for rank, lane in enumerate(lanes):
        tusimple = gcs_lanes_to_tusimple_lanes([lane], h_samples, image_shape=image_shape)
        if not tusimple:
            continue
        point_valid_scores = np.asarray(lane.get("point_valid_scores", []), dtype=np.float32)
        items.append(
            {
                "rank": int(rank),
                "query": int(lane.get("query", -1)),
                "score": float(lane["score"]),
                "valid_point_count": valid_point_count(lane),
                "point_valid_mean": None if point_valid_scores.size == 0 else float(point_valid_scores.mean()),
                "point_valid_max": None if point_valid_scores.size == 0 else float(point_valid_scores.max()),
                "lane": tusimple[0],
                "raw_lane": lane,
            }
        )
    return items


def draw_lane_xs(
    image: np.ndarray,
    lane: list[float],
    h_samples: list[int] | list[float],
    color: tuple[int, int, int],
    thickness: int,
    label: str,
) -> None:
    h, w = image.shape[:2]
    pts = []
    for x, y in zip(lane, h_samples):
        x = float(x)
        y = float(y)
        if x < 0.0:
            continue
        pts.append((int(round(np.clip(x, 0, w - 1))), int(round(np.clip(y, 0, h - 1)))))
    if len(pts) < 2:
        return
    arr = np.asarray(pts, dtype=np.int32)
    cv2.polylines(image, [arr], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for x, y in pts[:: max(len(pts) // 8, 1)]:
        cv2.circle(image, (x, y), max(2, thickness), color, -1, lineType=cv2.LINE_AA)
    if label:
        x0, y0 = pts[-1]
        cv2.putText(image, label, (x0 + 4, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def save_visualization(
    path: Path,
    image: np.ndarray,
    h_samples: list[int] | list[float],
    gt_lanes: list[list[float]],
    pred_items: list[dict],
    matched_pred: set[int],
    matched_gt: set[int],
    extra_pred: list[int],
    missing_gt: list[int],
    title: str,
) -> None:
    vis = image.copy()
    for gt_idx, lane in enumerate(gt_lanes):
        color = (0, 255, 255) if gt_idx in missing_gt else (0, 210, 0)
        draw_lane_xs(vis, lane, h_samples, color=color, thickness=2, label=f"GT{gt_idx}")
    for pred_idx, item in enumerate(pred_items):
        if pred_idx in extra_pred:
            color = (0, 0, 255)
            thickness = 3
            label = f"EX{pred_idx} q{item['query']} {item['score']:.3f}"
        elif pred_idx in matched_pred:
            color = (255, 120, 0)
            thickness = 2
            label = f"P{pred_idx} q{item['query']}"
        else:
            color = (180, 180, 180)
            thickness = 1
            label = f"P{pred_idx}"
        draw_lane_xs(vis, item["lane"], h_samples, color=color, thickness=thickness, label=label)
    cv2.putText(vis, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def compact_float(value: float | None, ndigits: int = 6):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, ndigits)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    count_pairs = parse_count_pairs(args.count_pairs)
    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_path = Path(args.gt_json) if args.gt_json else default_tusimple_gt_json(archive_root, split=args.split)
    gt_records = limit_records(read_tusimple_json_lines(gt_path), args.max_images)
    if not gt_records:
        raise ValueError(f"No TuSimple GT records found in {gt_path}")

    weights = Path(args.weights)
    imgsz = normalize_imgsz(args.imgsz, dataset="tusimple")
    save_dir = resolve_save_dir(args, weights)
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / "vis"
    image_list_dir = save_dir / "image_lists"

    warn_max_det_mismatch(weights, max_det=int(args.max_det), context="TuSimple extra-lane diagnostic")
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)

    pred_records: list[dict] = []
    per_image_rows: list[dict] = []
    extra_rows: list[dict] = []
    missing_rows: list[dict] = []
    selected_by_pair: dict[str, list[dict]] = defaultdict(list)
    vis_counts: Counter[str] = Counter()
    infer_time_s = 0.0
    post_time_s = 0.0

    if args.warmup > 0 and gt_records:
        first_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        first_img = cv2.imread(str(first_path), cv2.IMREAD_COLOR)
        if first_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {first_path}")
        first_tensor = preprocess_image(first_img, imgsz, device=device, half=bool(args.half))
        for _ in range(int(args.warmup)):
            _ = model(first_tensor)
        sync_if_cuda(device)

    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"diagnosing {len(gt_records)} records at conf={args.conf}, pvalid={args.point_valid_thr}, nms={args.nms_dist_px}")
    t0 = time.perf_counter()

    for idx, record in enumerate(gt_records, start=1):
        raw_file = str(record["raw_file"])
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_shape = image.shape[:2]
        gt_lanes = valid_tusimple_lanes(record.get("lanes", []))
        h_samples = record["h_samples"]

        tensor = preprocess_image(image, imgsz, device=device, half=bool(args.half))
        sync_if_cuda(device)
        infer_start = time.perf_counter()
        preds = model(tensor)
        sync_if_cuda(device)
        infer_end = time.perf_counter()

        pred_valid = preds.get("pred_valid_logits")
        lanes = decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
            image_shape=image_shape,
            score_thr=float(args.conf),
            point_valid_thr=float(args.point_valid_thr),
            min_points=int(args.min_points),
            max_det=int(args.max_det),
            nms_dist_px=float(args.nms_dist_px),
        )
        pred_items = convert_decoded_lanes(lanes, h_samples, image_shape=image_shape)
        pred_lanes = [item["lane"] for item in pred_items]
        post_end = time.perf_counter()
        infer_time_s += infer_end - infer_start
        post_time_s += post_end - infer_end

        pred_records.append(
            {
                "lanes": pred_lanes,
                "h_samples": h_samples,
                "raw_file": raw_file,
                "run_time": float(args.runtime_ms),
            }
        )

        matches, matched_pred, matched_gt = match_lanes(
            pred_lanes,
            gt_lanes,
            min_overlap=int(args.match_overlap),
            x_thr=float(args.match_x_thr),
        )
        gt_count = len(gt_lanes)
        pred_count = len(pred_lanes)
        count_pair = (gt_count, pred_count)
        count_pair_text = f"{gt_count}->{pred_count}"
        extra_pred = [i for i in range(pred_count) if i not in matched_pred]
        missing_gt = [i for i in range(gt_count) if i not in matched_gt]

        image_extra_labels: list[str] = []
        for pred_idx in extra_pred:
            item = pred_items[pred_idx]
            nearest_gt = nearest_lane_stats(item["lane"], gt_lanes, min_overlap=int(args.duplicate_overlap))
            nearest_pred = nearest_lane_stats(
                item["lane"],
                pred_lanes,
                min_overlap=int(args.duplicate_overlap),
                skip_idx=pred_idx,
            )
            nearest_matched_pred = nearest_lane_stats(
                item["lane"],
                [pred_lanes[i] for i in sorted(matched_pred)],
                min_overlap=int(args.duplicate_overlap),
            )
            duplicate_like = (
                (
                    int(nearest_gt["overlap"]) >= int(args.duplicate_overlap)
                    and float(nearest_gt["mean_abs_x_error"]) <= float(args.duplicate_x_thr)
                )
                or (
                    int(nearest_pred["overlap"]) >= int(args.duplicate_overlap)
                    and float(nearest_pred["mean_abs_x_error"]) <= float(args.duplicate_x_thr)
                )
            )
            label = "duplicate_like_extra" if duplicate_like else "spurious_extra"
            image_extra_labels.append(label)
            extra_rows.append(
                {
                    "raw_file": raw_file,
                    "count_pair": count_pair_text,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "pred_idx": int(pred_idx),
                    "rank": int(item["rank"]),
                    "query": int(item["query"]),
                    "score": round(float(item["score"]), 8),
                    "valid_point_count": int(item["valid_point_count"]),
                    "point_valid_mean": compact_float(item["point_valid_mean"]),
                    "point_valid_max": compact_float(item["point_valid_max"]),
                    "nearest_gt_lane_id": nearest_gt["lane_idx"],
                    "nearest_gt_overlap": int(nearest_gt["overlap"]),
                    "nearest_gt_mean_abs_x_error": compact_float(nearest_gt["mean_abs_x_error"]),
                    "nearest_pred_lane_id": nearest_pred["lane_idx"],
                    "nearest_pred_overlap": int(nearest_pred["overlap"]),
                    "nearest_pred_mean_abs_x_error": compact_float(nearest_pred["mean_abs_x_error"]),
                    "nearest_matched_pred_lane_id": nearest_matched_pred["lane_idx"],
                    "nearest_matched_pred_overlap": int(nearest_matched_pred["overlap"]),
                    "nearest_matched_pred_mean_abs_x_error": compact_float(nearest_matched_pred["mean_abs_x_error"]),
                    "possible_duplicate": int(duplicate_like),
                    "possible_spurious": int(not duplicate_like),
                    "extra_label": label,
                    "within_decode_nms_dist": int(
                        math.isfinite(float(nearest_pred["mean_abs_x_error"]))
                        and float(nearest_pred["mean_abs_x_error"]) <= float(args.nms_dist_px)
                    ),
                    "within_50px_pred": int(
                        math.isfinite(float(nearest_pred["mean_abs_x_error"]))
                        and float(nearest_pred["mean_abs_x_error"]) <= 50.0
                    ),
                    "image_path": str(image_path.resolve()),
                }
            )

        for gt_idx in missing_gt:
            nearest_pred = nearest_lane_stats(gt_lanes[gt_idx], pred_lanes, min_overlap=int(args.match_overlap))
            missing_rows.append(
                {
                    "raw_file": raw_file,
                    "count_pair": count_pair_text,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "gt_lane_id": int(gt_idx),
                    "gt_visible_points": sum(1 for x in gt_lanes[gt_idx] if float(x) >= 0.0),
                    "nearest_pred_lane_id": nearest_pred["lane_idx"],
                    "nearest_pred_overlap": int(nearest_pred["overlap"]),
                    "nearest_pred_mean_abs_x_error": compact_float(nearest_pred["mean_abs_x_error"]),
                    "image_path": str(image_path.resolve()),
                }
            )

        row = {
            "raw_file": raw_file,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "count_pair": count_pair_text,
            "matched_lanes": len(matches),
            "unmatched_pred_lanes": len(extra_pred),
            "unmatched_gt_lanes": len(missing_gt),
            "extra_pred_indices": ";".join(str(i) for i in extra_pred),
            "missing_gt_indices": ";".join(str(i) for i in missing_gt),
            "duplicate_like_extra": sum(1 for x in image_extra_labels if x == "duplicate_like_extra"),
            "spurious_extra": sum(1 for x in image_extra_labels if x == "spurious_extra"),
            "decoded_queries": ";".join(str(item["query"]) for item in pred_items),
            "decoded_scores": ";".join(f"{float(item['score']):.6f}" for item in pred_items),
            "decoded_valid_point_counts": ";".join(str(int(item["valid_point_count"])) for item in pred_items),
            "image_path": str(image_path.resolve()),
        }
        per_image_rows.append(row)

        if count_pair in count_pairs:
            selected_by_pair[count_pair_text].append(row)
            if not args.no_vis and vis_counts[count_pair_text] < int(args.vis_limit_per_pair):
                out_path = vis_dir / count_pair_text.replace("->", "to") / f"{vis_counts[count_pair_text]:04d}_{safe_name(raw_file)}.jpg"
                save_visualization(
                    out_path,
                    image,
                    h_samples,
                    gt_lanes,
                    pred_items,
                    matched_pred,
                    matched_gt,
                    extra_pred,
                    missing_gt,
                    title=f"{count_pair_text} dup={row['duplicate_like_extra']} spur={row['spurious_extra']}",
                )
                vis_counts[count_pair_text] += 1

        if idx % 50 == 0 or idx == len(gt_records):
            print(f"processed {idx}/{len(gt_records)}", flush=True)

    elapsed = time.perf_counter() - t0
    result, _ = TuSimpleOfficialLaneEval.bench_records(pred_records, gt_records, strict_length=True, return_records=False)
    metrics = result.as_dict()
    metrics = {
        "official_acc": metrics["Accuracy"],
        "official_FP": metrics["FP"],
        "official_FN": metrics["FN"],
        "images": metrics["images"],
        "official_score": round(
            official_metric_score(
                metrics["Accuracy"],
                metrics["FP"],
                metrics["FN"],
                fp_weight=float(args.score_fp_weight),
                fn_weight=float(args.score_fn_weight),
            ),
            6,
        ),
        **_count_diagnostics(pred_records, gt_records),
    }

    extra_counter = Counter(row["extra_label"] for row in extra_rows)
    pair_summary: dict[str, dict] = {}
    for pair_text in sorted({f"{a}->{b}" for a, b in count_pairs}):
        rows = selected_by_pair.get(pair_text, [])
        pair_extra = [row for row in extra_rows if row["count_pair"] == pair_text]
        pair_missing = [row for row in missing_rows if row["count_pair"] == pair_text]
        pair_summary[pair_text] = {
            "images": len(rows),
            "extra_lanes": len(pair_extra),
            "duplicate_like_extra": sum(1 for row in pair_extra if row["extra_label"] == "duplicate_like_extra"),
            "spurious_extra": sum(1 for row in pair_extra if row["extra_label"] == "spurious_extra"),
            "missing_lanes": len(pair_missing),
            "visualizations_saved": int(vis_counts[pair_text]),
        }

    summary = {
        "config": {
            "weights": str(weights.resolve()),
            "archive_root": str(archive_root.resolve()),
            "split": str(args.split),
            "gt_json": str(gt_path.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "match_overlap": int(args.match_overlap),
            "match_x_thr": float(args.match_x_thr),
            "duplicate_overlap": int(args.duplicate_overlap),
            "duplicate_x_thr": float(args.duplicate_x_thr),
            "count_pairs": sorted(pair_summary),
            "save_dir": str(save_dir.resolve()),
        },
        "metrics": metrics,
        "selected_count_pairs": pair_summary,
        "extra_lane_labels": dict(sorted(extra_counter.items())),
        "outputs": {
            "summary_json": str((save_dir / "summary.json").resolve()),
            "per_image_csv": str((save_dir / "per_image.csv").resolve()),
            "extra_lanes_csv": str((save_dir / "extra_lanes.csv").resolve()),
            "missing_lanes_csv": str((save_dir / "missing_lanes.csv").resolve()),
            "pred_json": str((save_dir / "tusimple_predictions.json").resolve()),
            "image_list_dir": str(image_list_dir.resolve()),
            "vis_dir": None if args.no_vis else str(vis_dir.resolve()),
        },
        "timing": {
            "avg_inference_ms": round(infer_time_s * 1000.0 / max(len(gt_records), 1), 4),
            "avg_postprocess_ms": round(post_time_s * 1000.0 / max(len(gt_records), 1), 4),
            "seconds": round(elapsed, 3),
        },
    }

    write_tusimple_predictions(save_dir / "tusimple_predictions.json", pred_records)
    write_csv(
        save_dir / "per_image.csv",
        per_image_rows,
        [
            "raw_file",
            "gt_count",
            "pred_count",
            "count_pair",
            "matched_lanes",
            "unmatched_pred_lanes",
            "unmatched_gt_lanes",
            "extra_pred_indices",
            "missing_gt_indices",
            "duplicate_like_extra",
            "spurious_extra",
            "decoded_queries",
            "decoded_scores",
            "decoded_valid_point_counts",
            "image_path",
        ],
    )
    write_csv(
        save_dir / "extra_lanes.csv",
        extra_rows,
        [
            "raw_file",
            "count_pair",
            "gt_count",
            "pred_count",
            "pred_idx",
            "rank",
            "query",
            "score",
            "valid_point_count",
            "point_valid_mean",
            "point_valid_max",
            "nearest_gt_lane_id",
            "nearest_gt_overlap",
            "nearest_gt_mean_abs_x_error",
            "nearest_pred_lane_id",
            "nearest_pred_overlap",
            "nearest_pred_mean_abs_x_error",
            "nearest_matched_pred_lane_id",
            "nearest_matched_pred_overlap",
            "nearest_matched_pred_mean_abs_x_error",
            "possible_duplicate",
            "possible_spurious",
            "extra_label",
            "within_decode_nms_dist",
            "within_50px_pred",
            "image_path",
        ],
    )
    write_csv(
        save_dir / "missing_lanes.csv",
        missing_rows,
        [
            "raw_file",
            "count_pair",
            "gt_count",
            "pred_count",
            "gt_lane_id",
            "gt_visible_points",
            "nearest_pred_lane_id",
            "nearest_pred_overlap",
            "nearest_pred_mean_abs_x_error",
            "image_path",
        ],
    )

    image_list_dir.mkdir(parents=True, exist_ok=True)
    for pair_text, rows in sorted(selected_by_pair.items()):
        tag = pair_text.replace("->", "to")
        (image_list_dir / f"{tag}.txt").write_text("\n".join(row["raw_file"] for row in rows) + ("\n" if rows else ""), encoding="utf-8")
        write_csv(
            image_list_dir / f"{tag}.csv",
            rows,
            [
                "raw_file",
                "gt_count",
                "pred_count",
                "count_pair",
                "matched_lanes",
                "unmatched_pred_lanes",
                "unmatched_gt_lanes",
                "duplicate_like_extra",
                "spurious_extra",
                "decoded_queries",
                "decoded_scores",
                "decoded_valid_point_counts",
                "image_path",
            ],
        )

    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "selected_count_pairs": pair_summary, "save_dir": str(save_dir.resolve())}, indent=2))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
