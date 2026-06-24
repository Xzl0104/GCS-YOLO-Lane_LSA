from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    default_tusimple_gt_json,
    find_tusimple_archive_root,
    read_tusimple_json_lines,
    tusimple_image_path,
)
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions, longest_contiguous_valid_mask  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = (
    ROOT
    / "runs"
    / "gcs_lane"
    / "gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03"
    / "weights"
    / "best.pt"
)
VALID_POINT_THRESHOLDS = (0.5, 0.45, 0.4, 0.35, 0.3)
SCORE_BINS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.000001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose GT4 4->3 missing lanes by dumping raw Q=12 GCS queries "
            "before confidence, point-valid, min-points, and final decode filters."
        )
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt to diagnose.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="TuSimple split to diagnose.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--gt-json", default=None, help="TuSimple GT json-lines file. Required for the 363-image official-val subset.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.005, help="Normal decode lane existence threshold.")
    parser.add_argument("--point-valid-thr", type=float, default=0.5, help="Normal decode point-valid threshold.")
    parser.add_argument("--nms-dist-px", type=float, default=0.0, help="Normal decode Lane-NMS threshold in original-image px.")
    parser.add_argument("--max-det", type=int, default=8, help="Normal decode max kept lanes.")
    parser.add_argument("--min-points", type=int, default=6, help="Normal decode minimum visible points.")
    parser.add_argument(
        "--only-count-pair",
        default="4->3",
        help="Only analyze normal-decode count pair, e.g. 4->3. Use 'all' to disable. Default: 4->3.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Limit GT records before filtering. 0 means all.")
    parser.add_argument("--save-dir", default=None, help="Output directory. Defaults under the checkpoint run directory.")
    parser.add_argument("--match-overlap", type=int, default=3, help="Minimum common visible h_samples for a raw-query/GT match.")
    parser.add_argument("--match-x-thr", type=float, default=20.0, help="Max mean absolute x error in px for a raw-query/GT match.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--warmup", type=int, default=0, help="Number of warmup forwards on the first processed image.")
    parser.add_argument("--vis-limit", type=int, default=50, help="Save visualizations for the first N selected images.")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization output.")
    return parser.parse_args()


def _limit_records(records: list[dict], max_images: int) -> list[dict]:
    if max_images and max_images > 0:
        return records[: int(max_images)]
    return records


def _disabled_filter(value: str | None) -> bool:
    return value is None or str(value).strip().lower() in {"", "all", "none", "null", "false", "0"}


def parse_count_pair(value: str | None) -> tuple[int, int] | None:
    if _disabled_filter(value):
        return None
    text = str(value).strip()
    if "->" not in text:
        raise ValueError(f"--only-count-pair must use A->B format, got {value!r}.")
    left, right = text.split("->", 1)
    return int(left.strip()), int(right.strip())


def weight_run_tag(weights: Path) -> str:
    if weights.parent.name == "weights":
        return weights.parent.parent.name
    return weights.stem


def resolve_save_dir(args: argparse.Namespace, weights: Path, count_pair: tuple[int, int] | None) -> Path:
    if args.save_dir is not None and str(args.save_dir).strip():
        return Path(args.save_dir)
    pair_tag = "all_pairs" if count_pair is None else f"{count_pair[0]}to{count_pair[1]}"
    decode_tag = (
        f"gt4_missing_raw_queries_{args.split}_{pair_tag}_"
        f"conf{float(args.conf):.4g}_pvalid{float(args.point_valid_thr):.4g}_"
        f"nms{float(args.nms_dist_px):.4g}_maxdet{int(args.max_det)}_minp{int(args.min_points)}"
    ).replace(".", "p")
    run_dir = weights.parent.parent if weights.parent.name == "weights" else None
    if run_dir is not None:
        return run_dir / decode_tag
    return ROOT / "runs" / "gcs_lane" / decode_tag


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def finite_round(value: float | np.floating | None, ndigits: int = 6):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, ndigits)


def valid_gt_lanes(record: dict) -> list[list[float]]:
    lanes = []
    for lane in record.get("lanes", []):
        lane = [float(x) for x in lane]
        if any(x >= 0.0 for x in lane):
            lanes.append(lane)
    return lanes


def visible_count(lane: list[float]) -> int:
    return sum(1 for x in lane if float(x) >= 0.0)


def lane_pair_stats(a: list[float], b: list[float]) -> tuple[int, float]:
    if len(a) != len(b):
        raise ValueError(f"lane lengths differ: {len(a)} vs {len(b)}")
    diffs = [abs(float(xa) - float(xb)) for xa, xb in zip(a, b) if float(xa) >= 0.0 and float(xb) >= 0.0]
    if not diffs:
        return 0, math.inf
    return len(diffs), float(sum(diffs) / len(diffs))


def is_match(stats: dict | None, min_overlap: int, x_thr: float) -> bool:
    if not stats:
        return False
    return int(stats["overlap_points"]) >= int(min_overlap) and float(stats["mean_abs_x_error"]) <= float(x_thr)


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
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches.append(
            {
                "pred_idx": int(pred_idx),
                "gt_idx": int(gt_idx),
                "overlap_points": int(-neg_overlap),
                "mean_abs_x_error": round(float(mean_abs_x), 6),
            }
        )
    return matches, matched_pred, matched_gt


def query_points_to_tusimple_lane(
    points_norm: np.ndarray,
    h_samples: list[int] | list[float],
    image_shape: tuple[int, int],
    valid_mask: np.ndarray | None = None,
) -> list[float]:
    points_norm = np.asarray(points_norm, dtype=np.float32)
    if points_norm.ndim != 2 or points_norm.shape[-1] != 2:
        raise ValueError(f"points_norm must have shape K x 2, got {points_norm.shape}")
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != points_norm.shape[:1]:
            raise ValueError(f"valid_mask shape {valid.shape} does not match points {points_norm.shape[:1]}")
        points_norm = points_norm[valid]

    if points_norm.shape[0] < 2:
        return [-2.0 for _ in h_samples]

    h, w = int(image_shape[0]), int(image_shape[1])
    points = np.clip(points_norm, 0.0, 1.0) * np.array([w, h], dtype=np.float32).reshape(1, 2)
    points = points[np.argsort(points[:, 1], kind="stable")]
    rounded_y = np.round(points[:, 1], decimals=4)
    _, unique_idx = np.unique(rounded_y, return_index=True)
    points = points[np.sort(unique_idx)]
    if points.shape[0] < 2:
        return [-2.0 for _ in h_samples]

    xs = points[:, 0].astype(np.float32)
    ys = points[:, 1].astype(np.float32)
    sample_y = np.asarray(h_samples, dtype=np.float32)
    pred_x = np.interp(sample_y, ys, xs, left=np.nan, right=np.nan)
    out: list[float] = []
    for x in pred_x:
        if not np.isfinite(x):
            out.append(-2.0)
            continue
        xf = float(x)
        out.append(xf if 0.0 <= xf < float(w) else -2.0)
    return out


def decoded_lanes_to_tusimple_items(
    lanes: list[dict],
    h_samples: list[int] | list[float],
    image_shape: tuple[int, int],
) -> list[dict]:
    items = []
    for rank, lane in enumerate(lanes):
        valid_mask = lane.get("point_valid")
        lane_xs = query_points_to_tusimple_lane(
            np.asarray(lane["points_norm"], dtype=np.float32),
            h_samples=h_samples,
            image_shape=image_shape,
            valid_mask=np.asarray(valid_mask, dtype=bool) if valid_mask is not None else None,
        )
        if visible_count(lane_xs) < 2:
            continue
        items.append(
            {
                "rank": int(rank),
                "query": int(lane["query"]),
                "score": float(lane["score"]),
                "lane": lane_xs,
                "point_valid": lane.get("point_valid"),
            }
        )
    return items


def contiguous_valid_mask(prob: np.ndarray, threshold: float, min_points: int) -> np.ndarray:
    mask_t = torch.from_numpy(np.asarray(prob, dtype=np.float32) >= float(threshold))
    return longest_contiguous_valid_mask(mask_t, min_points=int(min_points)).numpy().astype(bool)


def valid_run_count(prob: np.ndarray, threshold: float) -> int:
    mask_t = torch.from_numpy(np.asarray(prob, dtype=np.float32) >= float(threshold))
    return int(longest_contiguous_valid_mask(mask_t, min_points=1).sum().item())


def best_query_for_gt(
    query_lanes: list[list[float]],
    gt_lane: list[float],
    min_overlap: int,
) -> dict | None:
    stats = []
    for q, lane in enumerate(query_lanes):
        overlap, mean_abs_x = lane_pair_stats(lane, gt_lane)
        if overlap <= 0 or not math.isfinite(mean_abs_x):
            continue
        stats.append(
            {
                "query_id": int(q),
                "overlap_points": int(overlap),
                "mean_abs_x_error": float(mean_abs_x),
            }
        )
    if not stats:
        return None
    enough_overlap = [row for row in stats if int(row["overlap_points"]) >= int(min_overlap)]
    pool = enough_overlap if enough_overlap else stats
    return min(pool, key=lambda row: (float(row["mean_abs_x_error"]), -int(row["overlap_points"]), int(row["query_id"])))


def best_gt_for_query(query_lane: list[float], gt_lanes: list[list[float]], min_overlap: int) -> dict | None:
    stats = []
    for gt_idx, gt_lane in enumerate(gt_lanes):
        overlap, mean_abs_x = lane_pair_stats(query_lane, gt_lane)
        if overlap <= 0 or not math.isfinite(mean_abs_x):
            continue
        stats.append(
            {
                "gt_lane_id": int(gt_idx),
                "overlap_points": int(overlap),
                "mean_abs_x_error": float(mean_abs_x),
            }
        )
    if not stats:
        return None
    enough_overlap = [row for row in stats if int(row["overlap_points"]) >= int(min_overlap)]
    pool = enough_overlap if enough_overlap else stats
    return min(pool, key=lambda row: (float(row["mean_abs_x_error"]), -int(row["overlap_points"]), int(row["gt_lane_id"])))


def gt_lane_order_and_side(gt_lanes: list[list[float]], h_samples: list[int] | list[float]) -> dict[int, tuple[int, str]]:
    bottoms = []
    for idx, lane in enumerate(gt_lanes):
        visible = [(float(y), float(x)) for x, y in zip(lane, h_samples) if float(x) >= 0.0]
        if not visible:
            key = float("inf")
        else:
            max_y = max(y for y, _ in visible)
            xs = [x for y, x in visible if y == max_y]
            key = float(np.mean(xs))
        bottoms.append((key, idx))
    order = [idx for _, idx in sorted(bottoms)]
    side_names_4 = ["left_outer", "left_inner", "right_inner", "right_outer"]
    out: dict[int, tuple[int, str]] = {}
    for rank, idx in enumerate(order):
        if len(order) == 4:
            side = side_names_4[rank]
        elif rank == 0:
            side = "leftmost"
        elif rank == len(order) - 1:
            side = "rightmost"
        else:
            side = f"inner_{rank}"
        out[idx] = (int(rank), side)
    return out


def classify_drop_reason(
    best: dict | None,
    raw_match: bool,
    stage1_match: bool,
    stage2_match: bool,
    stage3_match: bool,
    final_match: bool,
) -> str:
    if best is None:
        return "no_raw_match"
    if not raw_match:
        return "geometry_bad"
    if not stage1_match:
        return "low_point_valid"
    if not stage2_match:
        return "low_min_points"
    if not stage3_match:
        return "low_score"
    if not final_match:
        return "ranked_out"
    return "unknown"


def histogram_int(values: list[int]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(int(x) for x in values).items())}


def bucket_visible(value: int) -> str:
    value = int(value)
    if value <= 5:
        return "<=5"
    if value <= 10:
        return "6..10"
    if value <= 20:
        return "11..20"
    if value <= 30:
        return "21..30"
    if value <= 40:
        return "31..40"
    return ">40"


def bucket_score(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "none"
    value = float(value)
    for lo, hi in zip(SCORE_BINS[:-1], SCORE_BINS[1:]):
        if lo <= value < hi:
            return f"[{lo:g},{hi:g})"
    return f">={SCORE_BINS[-2]:g}"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def safe_name(raw_file: str) -> str:
    return raw_file.lstrip("/").replace("\\", "_").replace("/", "_").replace(":", "_")


def draw_lane_xs(
    image: np.ndarray,
    lane: list[float],
    h_samples: list[int] | list[float],
    color: tuple[int, int, int],
    thickness: int,
    label: str | None = None,
) -> None:
    h, w = image.shape[:2]
    pts = [
        (int(round(float(x))), int(round(float(y))))
        for x, y in zip(lane, h_samples)
        if float(x) >= 0.0 and 0.0 <= float(x) < w and 0.0 <= float(y) < h
    ]
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
    final_items: list[dict],
    missing_rows: list[dict],
    raw_query_lanes: list[list[float]],
) -> None:
    vis = image.copy()
    for gt_idx, lane in enumerate(gt_lanes):
        color = (0, 210, 0) if not any(int(row["gt_lane_id"]) == gt_idx for row in missing_rows) else (0, 255, 255)
        draw_lane_xs(vis, lane, h_samples, color=color, thickness=2, label=f"GT{gt_idx}")
    for item in final_items:
        draw_lane_xs(vis, item["lane"], h_samples, color=(255, 120, 0), thickness=2, label=f"Pq{item['query']}")
    for row in missing_rows:
        q = row.get("best_query_id")
        if q is None or q == "":
            continue
        q = int(q)
        if 0 <= q < len(raw_query_lanes):
            label = f"miss GT{row['gt_lane_id']} q{q} {row['drop_reason']}"
            draw_lane_xs(vis, raw_query_lanes[q], h_samples, color=(0, 0, 255), thickness=3, label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def rate(num: int, den: int) -> float:
    return round(float(num) / max(int(den), 1), 6)


@torch.inference_mode()
def run_diagnostic(args: argparse.Namespace) -> dict:
    """Run the raw-query diagnostic and write all configured artifacts."""
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")
    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_path = Path(args.gt_json) if args.gt_json else default_tusimple_gt_json(archive_root, split=args.split)
    if args.split == "val" and args.gt_json is None:
        print(
            "WARNING: --split val without --gt-json uses the archive default labels, not necessarily the "
            "363-image official-val subset.",
            file=sys.stderr,
        )
    gt_records = _limit_records(read_tusimple_json_lines(gt_path), int(args.max_images))
    if not gt_records:
        raise ValueError(f"No GT records found in {gt_path}.")

    imgsz = normalize_imgsz(args.imgsz, dataset="tusimple")
    count_pair = parse_count_pair(args.only_count_pair)
    save_dir = resolve_save_dir(args, weights=weights, count_pair=count_pair)
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / "vis"

    device = select_device(args.device, verbose=False)
    model = load_gcs_model(weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    warn_max_det_mismatch(weights, max_det=int(args.max_det), context="GT4 missing-lane raw-query diagnostic")

    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"diagnosing split={args.split} count_pair={args.only_count_pair} records={len(gt_records)}")

    did_warmup = False
    selected_images = 0
    image_rows: list[dict] = []
    raw_query_rows: list[dict] = []
    missing_rows: list[dict] = []
    vis_count = 0
    t0 = time.perf_counter()

    for idx, record in enumerate(gt_records, start=1):
        raw_file = str(record["raw_file"])
        gt_lanes = valid_gt_lanes(record)
        gt_count = len(gt_lanes)
        if count_pair is not None and gt_count != int(count_pair[0]):
            continue

        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_shape = image.shape[:2]
        h_samples = [int(y) for y in record["h_samples"]]

        tensor = preprocess_image(image, imgsz=imgsz, device=device, half=bool(args.half))
        if int(args.warmup) > 0 and not did_warmup:
            for _ in range(int(args.warmup)):
                _ = model(tensor)
            sync_if_cuda(device)
            did_warmup = True

        preds = model(tensor)
        pred_points_t = preds["pred_points"][0].detach().float()
        pred_logits_t = preds["pred_logits"][0].detach().float()
        if pred_logits_t.ndim == 2 and pred_logits_t.shape[-1] == 1:
            pred_logits_t = pred_logits_t.squeeze(-1)
        pred_valid_logits = preds.get("pred_valid_logits")
        pred_valid_t = pred_valid_logits[0].detach().float() if pred_valid_logits is not None else None

        final_lanes = decode_gcs_predictions(
            pred_points_t,
            pred_logits_t,
            pred_valid_logits=pred_valid_t,
            image_shape=image_shape,
            score_thr=float(args.conf),
            point_valid_thr=float(args.point_valid_thr),
            min_points=int(args.min_points),
            max_det=int(args.max_det),
            nms_dist_px=float(args.nms_dist_px),
        )
        final_items = decoded_lanes_to_tusimple_items(final_lanes, h_samples=h_samples, image_shape=image_shape)
        pred_count = len(final_items)
        pair_text = f"{gt_count}->{pred_count}"
        if count_pair is not None and pred_count != int(count_pair[1]):
            continue

        selected_images += 1
        pred_points = pred_points_t.cpu().numpy().astype(np.float32)
        exist_logits = pred_logits_t.cpu().numpy().astype(np.float32)
        scores = 1.0 / (1.0 + np.exp(-exist_logits))
        q_count, k_count = pred_points.shape[:2]
        if pred_valid_t is None:
            point_valid_prob = np.ones((q_count, k_count), dtype=np.float32)
        else:
            point_valid_prob = pred_valid_t.sigmoid().cpu().numpy().astype(np.float32)

        raw_query_lanes = [
            query_points_to_tusimple_lane(pred_points[q], h_samples=h_samples, image_shape=image_shape)
            for q in range(q_count)
        ]
        point_valid_masks = [
            np.asarray(point_valid_prob[q] >= float(args.point_valid_thr), dtype=bool)
            for q in range(q_count)
        ]
        stage1_query_lanes = [
            query_points_to_tusimple_lane(pred_points[q], h_samples=h_samples, image_shape=image_shape, valid_mask=point_valid_masks[q])
            for q in range(q_count)
        ]
        min_point_masks = [
            contiguous_valid_mask(point_valid_prob[q], float(args.point_valid_thr), int(args.min_points))
            for q in range(q_count)
        ]
        stage2_query_lanes = [
            query_points_to_tusimple_lane(pred_points[q], h_samples=h_samples, image_shape=image_shape, valid_mask=min_point_masks[q])
            for q in range(q_count)
        ]

        final_pred_lanes = [item["lane"] for item in final_items]
        final_matches, matched_pred, matched_gt = match_lanes(
            final_pred_lanes,
            gt_lanes,
            min_overlap=int(args.match_overlap),
            x_thr=float(args.match_x_thr),
        )
        final_query_by_gt = {
            int(match["gt_idx"]): int(final_items[int(match["pred_idx"])]["query"])
            for match in final_matches
            if int(match["pred_idx"]) < len(final_items)
        }
        final_rank_by_query = {int(item["query"]): int(item["rank"]) for item in final_items}
        final_decoded_queries = sorted(final_rank_by_query, key=lambda q: final_rank_by_query[q])
        missing_gt_ids = [gt_idx for gt_idx in range(gt_count) if gt_idx not in matched_gt]
        order_side = gt_lane_order_and_side(gt_lanes, h_samples)

        image_missing_rows: list[dict] = []
        for q in range(q_count):
            raw_best = best_gt_for_query(raw_query_lanes[q], gt_lanes, min_overlap=int(args.match_overlap))
            stage1_best = best_gt_for_query(stage1_query_lanes[q], gt_lanes, min_overlap=int(args.match_overlap))
            stage2_best = best_gt_for_query(stage2_query_lanes[q], gt_lanes, min_overlap=int(args.match_overlap))
            valid_counts = {
                f"valid_points_at_{str(thr).replace('.', '_')}": valid_run_count(point_valid_prob[q], thr)
                for thr in VALID_POINT_THRESHOLDS
            }
            raw_query_rows.append(
                {
                    "raw_file": raw_file,
                    "split": args.split,
                    "gt_count": int(gt_count),
                    "pred_count": int(pred_count),
                    "count_pair": pair_text,
                    "query_id": int(q),
                    "exist_logit": finite_round(exist_logits[q]),
                    "score": finite_round(scores[q], ndigits=8),
                    **valid_counts,
                    "survives_min_points": int(valid_run_count(point_valid_prob[q], float(args.point_valid_thr)) >= int(args.min_points)),
                    "survives_conf": int(float(scores[q]) >= float(args.conf)),
                    "survives_final_decode": int(q in final_rank_by_query),
                    "final_decode_rank": final_rank_by_query.get(q),
                    "raw_best_gt_lane_id": None if raw_best is None else raw_best["gt_lane_id"],
                    "raw_best_mean_abs_x_error": None if raw_best is None else finite_round(raw_best["mean_abs_x_error"]),
                    "raw_best_overlap_points": None if raw_best is None else raw_best["overlap_points"],
                    "raw_matches_any_gt": int(is_match(raw_best, int(args.match_overlap), float(args.match_x_thr))),
                    "stage1_best_gt_lane_id": None if stage1_best is None else stage1_best["gt_lane_id"],
                    "stage1_best_mean_abs_x_error": None if stage1_best is None else finite_round(stage1_best["mean_abs_x_error"]),
                    "stage1_best_overlap_points": None if stage1_best is None else stage1_best["overlap_points"],
                    "stage1_matches_any_gt": int(is_match(stage1_best, int(args.match_overlap), float(args.match_x_thr))),
                    "stage2_best_gt_lane_id": None if stage2_best is None else stage2_best["gt_lane_id"],
                    "stage2_best_mean_abs_x_error": None if stage2_best is None else finite_round(stage2_best["mean_abs_x_error"]),
                    "stage2_best_overlap_points": None if stage2_best is None else stage2_best["overlap_points"],
                    "stage2_matches_any_gt": int(is_match(stage2_best, int(args.match_overlap), float(args.match_x_thr))),
                }
            )

        for gt_idx in missing_gt_ids:
            gt_lane = gt_lanes[gt_idx]
            best = best_query_for_gt(raw_query_lanes, gt_lane, min_overlap=int(args.match_overlap))
            if best is None:
                q = None
                raw_match = stage1_match = stage2_match = stage3_match = final_match = False
                stage1_stats = stage2_stats = None
                valid_counts = {f"best_query_valid_points_at_{str(thr).replace('.', '_')}": None for thr in VALID_POINT_THRESHOLDS}
                exist_logit = score = None
            else:
                q = int(best["query_id"])
                raw_match = is_match(best, int(args.match_overlap), float(args.match_x_thr))
                stage1_stats = best_query_for_gt([stage1_query_lanes[q]], gt_lane, min_overlap=int(args.match_overlap))
                stage2_stats = best_query_for_gt([stage2_query_lanes[q]], gt_lane, min_overlap=int(args.match_overlap))
                stage1_match = is_match(stage1_stats, int(args.match_overlap), float(args.match_x_thr))
                stage2_match = is_match(stage2_stats, int(args.match_overlap), float(args.match_x_thr))
                stage3_match = bool(stage2_match and float(scores[q]) >= float(args.conf))
                final_match = bool(final_query_by_gt.get(gt_idx) == q)
                valid_counts = {
                    f"best_query_valid_points_at_{str(thr).replace('.', '_')}": valid_run_count(point_valid_prob[q], thr)
                    for thr in VALID_POINT_THRESHOLDS
                }
                exist_logit = finite_round(exist_logits[q])
                score = finite_round(scores[q], ndigits=8)

            order, side = order_side.get(gt_idx, (gt_idx, "unknown"))
            drop_reason = classify_drop_reason(best, raw_match, stage1_match, stage2_match, stage3_match, final_match)
            row = {
                "raw_file": raw_file,
                "split": args.split,
                "gt_count": int(gt_count),
                "pred_count": int(pred_count),
                "count_pair": pair_text,
                "gt_lane_id": int(gt_idx),
                "gt_visible_points": visible_count(gt_lane),
                "gt_lane_order": int(order),
                "gt_lane_side": side,
                "best_query_id": q,
                "best_query_mean_abs_x_error": None if best is None else finite_round(best["mean_abs_x_error"]),
                "best_query_overlap_points": None if best is None else int(best["overlap_points"]),
                "best_query_exist_logit": exist_logit,
                "best_query_score": score,
                **valid_counts,
                "stage1_mean_abs_x_error": None if stage1_stats is None else finite_round(stage1_stats["mean_abs_x_error"]),
                "stage1_overlap_points": None if stage1_stats is None else int(stage1_stats["overlap_points"]),
                "stage2_mean_abs_x_error": None if stage2_stats is None else finite_round(stage2_stats["mean_abs_x_error"]),
                "stage2_overlap_points": None if stage2_stats is None else int(stage2_stats["overlap_points"]),
                "survives_point_valid": int(stage1_match),
                "survives_min_points": int(stage2_match),
                "survives_conf": int(stage3_match),
                "survives_final_decode": int(final_match),
                "final_decode_rank": None if q is None else final_rank_by_query.get(q),
                "drop_reason": drop_reason,
            }
            missing_rows.append(row)
            image_missing_rows.append(row)

        image_rows.append(
            {
                "raw_file": raw_file,
                "split": args.split,
                "image_path": str(image_path),
                "gt_count": int(gt_count),
                "pred_count": int(pred_count),
                "count_pair": pair_text,
                "gt_visible_counts": ";".join(str(visible_count(lane)) for lane in gt_lanes),
                "missing_gt_lane_ids": ";".join(str(x) for x in missing_gt_ids),
                "missing_gt_lanes": int(len(missing_gt_ids)),
                "matched_gt_lanes": int(len(matched_gt)),
                "final_decoded_queries": ";".join(str(x) for x in final_decoded_queries),
                "stage0_raw_all_queries": int(q_count),
                "stage1_after_point_valid_queries": int(
                    sum(valid_run_count(point_valid_prob[q], float(args.point_valid_thr)) > 0 for q in range(q_count))
                ),
                "stage2_after_min_points_queries": int(
                    sum(valid_run_count(point_valid_prob[q], float(args.point_valid_thr)) >= int(args.min_points) for q in range(q_count))
                ),
                "stage3_after_conf_queries": int(
                    sum(
                        valid_run_count(point_valid_prob[q], float(args.point_valid_thr)) >= int(args.min_points)
                        and float(scores[q]) >= float(args.conf)
                        for q in range(q_count)
                    )
                ),
                "stage4_final_decode_lanes": int(pred_count),
                "missing_raw_match": int(sum(1 for row in image_missing_rows if row["drop_reason"] != "no_raw_match" and row["drop_reason"] != "geometry_bad")),
                "missing_after_point_valid": int(sum(int(row["survives_point_valid"]) for row in image_missing_rows)),
                "missing_after_min_points": int(sum(int(row["survives_min_points"]) for row in image_missing_rows)),
                "missing_after_conf": int(sum(int(row["survives_conf"]) for row in image_missing_rows)),
                "drop_reasons": ";".join(str(row["drop_reason"]) for row in image_missing_rows),
            }
        )

        if not args.no_vis and vis_count < int(args.vis_limit):
            out_path = vis_dir / f"{vis_count:04d}_{safe_name(raw_file)}.jpg"
            save_visualization(out_path, image, h_samples, gt_lanes, final_items, image_missing_rows, raw_query_lanes)
            vis_count += 1

        if selected_images % 50 == 0:
            print(f"selected {selected_images} images at source index {idx}/{len(gt_records)}", flush=True)

    elapsed = time.perf_counter() - t0
    total_missing = len(missing_rows)
    raw_match_count = sum(1 for row in missing_rows if row["drop_reason"] not in {"no_raw_match", "geometry_bad"})
    stage_point_valid = sum(int(row["survives_point_valid"]) for row in missing_rows)
    stage_min_points = sum(int(row["survives_min_points"]) for row in missing_rows)
    stage_conf = sum(int(row["survives_conf"]) for row in missing_rows)
    stage_final = sum(int(row["survives_final_decode"]) for row in missing_rows)
    drop_reason_hist = Counter(row["drop_reason"] for row in missing_rows)

    visible_values = [int(row["gt_visible_points"]) for row in missing_rows]
    score_values = [row["best_query_score"] for row in missing_rows]
    side_values = [str(row["gt_lane_side"]) for row in missing_rows]
    order_values = [str(row["gt_lane_order"]) for row in missing_rows]
    summary = {
        "config": {
            "weights": str(weights.resolve()),
            "archive_root": str(archive_root.resolve()),
            "split": args.split,
            "gt_json": str(gt_path.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "only_count_pair": None if count_pair is None else f"{count_pair[0]}->{count_pair[1]}",
            "match_overlap": int(args.match_overlap),
            "match_x_thr": float(args.match_x_thr),
            "max_images": int(args.max_images),
            "device": str(args.device),
            "half": bool(args.half),
            "warmup": int(args.warmup),
            "save_dir": str(save_dir.resolve()),
        },
        "records_scanned": int(len(gt_records)),
        "total_gt4_4to3_images": int(selected_images) if count_pair == (4, 3) else None,
        "selected_images": int(selected_images),
        "total_missing_gt_lanes": int(total_missing),
        "drop_reason_histogram": {str(k): int(v) for k, v in sorted(drop_reason_hist.items())},
        "gt4_diagnostics": {
            "gt4_missing_lanes": int(total_missing),
            "geometry_bad": int(drop_reason_hist.get("geometry_bad", 0)),
            "low_point_valid": int(drop_reason_hist.get("low_point_valid", 0)),
            "low_score": int(drop_reason_hist.get("low_score", 0)),
            "raw_match_recall": rate(raw_match_count, total_missing),
            "raw_match_count": int(raw_match_count),
            "denominator_missing_gt_lanes": int(total_missing),
        },
        "missing_lane_visible_point_histogram": histogram_int(visible_values),
        "missing_lane_visible_point_bucket_histogram": {str(k): int(v) for k, v in sorted(Counter(bucket_visible(x) for x in visible_values).items())},
        "missing_lane_side_histogram": {str(k): int(v) for k, v in sorted(Counter(side_values).items())},
        "missing_lane_order_histogram": {str(k): int(v) for k, v in sorted(Counter(order_values).items())},
        "best_query_score_histogram": {str(k): int(v) for k, v in sorted(Counter(bucket_score(x) for x in score_values).items())},
        "best_query_valid_points_histograms": {
            f"at_{str(thr).replace('.', '_')}": histogram_int(
                [
                    int(row[f"best_query_valid_points_at_{str(thr).replace('.', '_')}"])
                    for row in missing_rows
                    if row.get(f"best_query_valid_points_at_{str(thr).replace('.', '_')}") is not None
                ]
            )
            for thr in VALID_POINT_THRESHOLDS
        },
        "stage_recall": {
            "denominator_missing_gt_lanes": int(total_missing),
            "raw_match_recall": rate(raw_match_count, total_missing),
            "after_point_valid_recall": rate(stage_point_valid, total_missing),
            "after_min_points_recall": rate(stage_min_points, total_missing),
            "after_conf_recall": rate(stage_conf, total_missing),
            "final_decode_recall": rate(stage_final, total_missing),
            "counts": {
                "raw_match": int(raw_match_count),
                "after_point_valid": int(stage_point_valid),
                "after_min_points": int(stage_min_points),
                "after_conf": int(stage_conf),
                "final_decode": int(stage_final),
            },
        },
        "artifacts": {
            "summary_json": str((save_dir / "summary.json").resolve()),
            "per_missing_lane_csv": str((save_dir / "per_missing_lane.csv").resolve()),
            "per_image_summary_csv": str((save_dir / "per_image_summary.csv").resolve()),
            "raw_queries_csv": str((save_dir / "raw_queries.csv").resolve()),
            "vis_dir": None if args.no_vis else str(vis_dir.resolve()),
            "visualizations_saved": int(vis_count),
        },
        "seconds": round(float(elapsed), 3),
    }

    missing_fields = [
        "raw_file",
        "split",
        "gt_count",
        "pred_count",
        "count_pair",
        "gt_lane_id",
        "gt_visible_points",
        "gt_lane_order",
        "gt_lane_side",
        "best_query_id",
        "best_query_mean_abs_x_error",
        "best_query_overlap_points",
        "best_query_exist_logit",
        "best_query_score",
        "best_query_valid_points_at_0_5",
        "best_query_valid_points_at_0_45",
        "best_query_valid_points_at_0_4",
        "best_query_valid_points_at_0_35",
        "best_query_valid_points_at_0_3",
        "stage1_mean_abs_x_error",
        "stage1_overlap_points",
        "stage2_mean_abs_x_error",
        "stage2_overlap_points",
        "survives_point_valid",
        "survives_min_points",
        "survives_conf",
        "survives_final_decode",
        "final_decode_rank",
        "drop_reason",
    ]
    image_fields = [
        "raw_file",
        "split",
        "image_path",
        "gt_count",
        "pred_count",
        "count_pair",
        "gt_visible_counts",
        "missing_gt_lane_ids",
        "missing_gt_lanes",
        "matched_gt_lanes",
        "final_decoded_queries",
        "stage0_raw_all_queries",
        "stage1_after_point_valid_queries",
        "stage2_after_min_points_queries",
        "stage3_after_conf_queries",
        "stage4_final_decode_lanes",
        "missing_raw_match",
        "missing_after_point_valid",
        "missing_after_min_points",
        "missing_after_conf",
        "drop_reasons",
    ]
    raw_query_fields = [
        "raw_file",
        "split",
        "gt_count",
        "pred_count",
        "count_pair",
        "query_id",
        "exist_logit",
        "score",
        "valid_points_at_0_5",
        "valid_points_at_0_45",
        "valid_points_at_0_4",
        "valid_points_at_0_35",
        "valid_points_at_0_3",
        "survives_min_points",
        "survives_conf",
        "survives_final_decode",
        "final_decode_rank",
        "raw_best_gt_lane_id",
        "raw_best_mean_abs_x_error",
        "raw_best_overlap_points",
        "raw_matches_any_gt",
        "stage1_best_gt_lane_id",
        "stage1_best_mean_abs_x_error",
        "stage1_best_overlap_points",
        "stage1_matches_any_gt",
        "stage2_best_gt_lane_id",
        "stage2_best_mean_abs_x_error",
        "stage2_best_overlap_points",
        "stage2_matches_any_gt",
    ]

    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(save_dir / "per_missing_lane.csv", missing_rows, missing_fields)
    write_csv(save_dir / "per_image_summary.csv", image_rows, image_fields)
    write_csv(save_dir / "raw_queries.csv", raw_query_rows, raw_query_fields)
    print(json.dumps({k: summary[k] for k in ("selected_images", "total_gt4_4to3_images", "total_missing_gt_lanes", "drop_reason_histogram", "stage_recall")}, indent=2))
    print(f"saved: {save_dir.resolve()}")
    return summary


def main() -> None:
    run_diagnostic(parse_args())


if __name__ == "__main__":
    main()
