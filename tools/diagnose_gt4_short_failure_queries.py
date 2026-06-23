from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.eval_gcs import gated_assignment, label_path_for_image, load_gcs_label, pair_geometry  # noqa: E402
from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image  # noqa: E402
from ultralytics.utils.gcs_matcher import GCSHungarianMatcher  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions, longest_contiguous_valid_mask  # noqa: E402
from ultralytics.utils.gcs_shape import assert_gcs_shape, normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_WEIGHTS = (
    ROOT
    / "runs"
    / "gcs_lane"
    / "gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03"
    / "weights"
    / "best.pt"
)
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace raw GCS queries on fixed-y train/val images. The diagnostic is "
            "self-contained and does not depend on legacy count-confusion helpers."
        )
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint to diagnose.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Converted fixed-y GCS dataset root.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=("train", "val"), help="Splits to diagnose.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="Inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.15, help="Selected lane existence threshold.")
    parser.add_argument("--point-valid-thr", type=float, default=0.5, help="Selected point visibility threshold.")
    parser.add_argument("--nms-dist-px", type=float, default=0.0, help="Selected Lane-NMS threshold. 0 disables.")
    parser.add_argument("--max-det", type=int, default=6, help="Selected maximum decoded lanes.")
    parser.add_argument("--min-points", type=int, default=4, help="Selected minimum visible points.")
    parser.add_argument(
        "--gt-counts",
        nargs="+",
        type=int,
        default=None,
        help="GT lane counts to trace. Defaults to --gt-count for backwards compatibility.",
    )
    parser.add_argument("--gt-count", type=int, default=4, help="Backwards-compatible single GT lane count.")
    parser.add_argument(
        "--short-visible-max",
        type=int,
        default=20,
        help="Max shortest visible GT lane length. Use 0 to disable the short-lane filter.",
    )
    parser.add_argument("--bucket-edges", nargs="+", type=int, default=[10, 20, 30, 40], help="Min-visible bucket edges.")
    parser.add_argument("--trace-scope", choices=("failures", "candidates"), default="failures", help="Trace only count failures or every selected candidate image.")
    parser.add_argument("--ape-thr", type=float, default=20.0, help="APE threshold for a true-lane-like query.")
    parser.add_argument(
        "--duplicate-ape-thr",
        type=float,
        default=20.0,
        help="Decoded unmatched query with best APE at or below this is duplicate-like.",
    )
    parser.add_argument(
        "--duplicate-visible-iou-thr",
        type=float,
        default=0.5,
        help="Decoded unmatched query with visible IoU at or above this can be duplicate-like.",
    )
    parser.add_argument("--min-overlap", type=int, default=2, help="Minimum overlapping visible points for geometry matching.")
    parser.add_argument("--match-cost-point", type=float, default=5.0, help="Training Hungarian point cost.")
    parser.add_argument("--match-cost-curve", type=float, default=0.05, help="Training Hungarian curvature cost.")
    parser.add_argument("--match-cost-exist", type=float, default=0.1, help="Training Hungarian existence cost.")
    parser.add_argument("--match-min-overlap", type=int, default=2, help="Training Hungarian minimum GT visible points.")
    parser.add_argument("--match-max-x-dist", type=float, default=0.0, help="Training Hungarian max mean x-distance gate. 0 disables.")
    parser.add_argument("--match-gate-px", type=float, default=160.0, help="Training Hungarian APE gate in pixels. 0 disables.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup forwards on the first image per split.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional per-split image limit. 0 means all.")
    parser.add_argument("--run-tag", default=None, help="Tag written to outputs. Defaults to the checkpoint run directory name.")
    parser.add_argument("--save-dir", default=None, help="Output directory. Defaults under runs/gcs_lane/query_trace.")
    return parser.parse_args()


def npz_scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def date_id(raw_file: str) -> str:
    rel = raw_file.lstrip("/").replace("\\", "/")
    parts = Path(rel).parts
    if len(parts) >= 2 and parts[0] == "clips":
        return str(parts[1])
    return str(parts[0]) if parts else "unknown"


def bucket_label(value: int, edges: list[int]) -> str:
    value = int(value)
    if value <= 0:
        return "none"
    prev = 1
    for edge in sorted({int(x) for x in edges}):
        if value <= edge:
            return f"<={edge}" if prev <= 1 else f"{prev}..{edge}"
        prev = edge + 1
    return f">{max(edges)}" if edges else "all"


def weight_run_tag(weights: Path) -> str:
    if weights.parent.name == "weights":
        return weights.parent.parent.name
    return weights.stem


def resolve_save_dir(args: argparse.Namespace, weights: Path, gt_counts: list[int]) -> Path:
    if args.save_dir is not None and str(args.save_dir).strip():
        return Path(args.save_dir)
    tag = args.run_tag or weight_run_tag(weights)
    gt_tag = "gt" + "-".join(str(x) for x in gt_counts)
    short_tag = "allvisible" if int(args.short_visible_max) <= 0 else f"short{int(args.short_visible_max)}"
    return ROOT / "runs" / "gcs_lane" / "query_trace" / f"{tag}_{gt_tag}_{short_tag}_{args.trace_scope}_train_val"


def load_label_metadata(label_path: Path) -> dict:
    with np.load(label_path, allow_pickle=False) as data:
        raw_file = npz_scalar_text(data["raw_file"]) if "raw_file" in data.files else label_path.with_suffix(".jpg").name
        lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
        if "num_lanes" in data.files:
            gt_count = int(np.asarray(data["num_lanes"]).reshape(-1)[0])
        else:
            gt_count = int((lane_valid.sum(axis=1) >= 2).sum())
    visible_counts = [int(x) for x in lane_valid.sum(axis=1).tolist()[:gt_count]]
    return {
        "raw_file": raw_file,
        "date": date_id(raw_file),
        "gt_count": int(gt_count),
        "visible_counts": visible_counts,
        "min_visible_points": min(visible_counts) if visible_counts else 0,
    }


def visible_iou_matrix(pred_valid: np.ndarray, gt_valid: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_valid, dtype=np.float32).clip(0.0, 1.0)
    gt = (np.asarray(gt_valid, dtype=np.float32) > 0.5).astype(np.float32)
    inter = (pred[:, None, :] * gt[None]).sum(axis=2)
    union = pred.sum(axis=1)[:, None] + gt.sum(axis=1)[None] - inter
    return np.divide(inter, np.maximum(union, 1e-6), out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def pred_visible_lengths(pred_valid_scores: torch.Tensor | None, point_valid_thr: float, q_count: int, k_count: int) -> np.ndarray:
    if pred_valid_scores is None:
        return np.full((q_count,), int(k_count), dtype=np.int32)
    lengths = []
    for q in range(q_count):
        mask = longest_contiguous_valid_mask(pred_valid_scores[q].detach().cpu() >= float(point_valid_thr), min_points=1)
        lengths.append(int(mask.sum().item()))
    return np.asarray(lengths, dtype=np.int32)


def finite_round(value: float | np.floating | None, ndigits: int = 4):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, ndigits)


def summarize_numeric(values: list[float]) -> dict:
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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def query_reason(
    role: str,
    decoded: bool,
    strict_matched: bool,
    score: float,
    conf: float,
    visible_len: int,
    min_points: int,
    best_ape: float,
    best_visible_iou: float,
    best_gt_visible: int,
    duplicate_ape_thr: float,
    duplicate_visible_iou_thr: float,
) -> str:
    near_gt = np.isfinite(best_ape) and best_ape <= float(duplicate_ape_thr)
    duplicate_like = near_gt or best_visible_iou >= float(duplicate_visible_iou_thr)
    short_gt = 0 < int(best_gt_visible) <= 20
    if role == "matched_qpos":
        if decoded and strict_matched:
            return "matched_decoded_tp_short_gt" if short_gt else "matched_decoded_tp"
        if score < float(conf):
            return "matched_low_score_short_gt" if short_gt else "matched_low_score"
        if visible_len < int(min_points):
            return "matched_min_points_filtered_short_gt" if short_gt else "matched_min_points_filtered"
        return "matched_rank_or_nms_filtered_short_gt" if short_gt else "matched_rank_or_nms_filtered"
    if decoded:
        if duplicate_like:
            return "unmatched_duplicate_like_extra_short_gt" if short_gt else "unmatched_duplicate_like_extra"
        return "unmatched_spurious_extra"
    if score >= float(conf) and visible_len >= int(min_points):
        if duplicate_like:
            return "unmatched_duplicate_like_suppressed_short_gt" if short_gt else "unmatched_duplicate_like_suppressed"
        return "unmatched_spurious_suppressed"
    if score < float(conf):
        if duplicate_like:
            return "unmatched_low_score_near_gt_short_gt" if short_gt else "unmatched_low_score_near_gt"
        return "unmatched_low_score_background"
    return "unmatched_min_points_filtered"


def empty_role_state() -> dict:
    return {
        "queries": 0,
        "decoded": 0,
        "strict_decoded_matched": 0,
        "scores": [],
        "best_apes": [],
        "best_visible_ious": [],
        "visible_lengths": [],
    }


def update_role_state(state: dict, row: dict) -> None:
    state["queries"] += 1
    state["decoded"] += int(row["decoded"])
    state["strict_decoded_matched"] += int(row["strict_decoded_matched"])
    state["scores"].append(float(row["score"]))
    if row.get("best_ape_px") is not None:
        state["best_apes"].append(float(row["best_ape_px"]))
    if row.get("best_visible_iou") is not None:
        state["best_visible_ious"].append(float(row["best_visible_iou"]))
    state["visible_lengths"].append(float(row["pred_visible_len"]))


def finalize_role_state(state: dict) -> dict:
    q = max(int(state["queries"]), 1)
    return {
        "queries": int(state["queries"]),
        "decoded": int(state["decoded"]),
        "decoded_rate": round(float(state["decoded"]) / q, 6),
        "strict_decoded_matched": int(state["strict_decoded_matched"]),
        "strict_decoded_matched_rate": round(float(state["strict_decoded_matched"]) / q, 6),
        "score": summarize_numeric(state["scores"]),
        "best_ape_px": summarize_numeric(state["best_apes"]),
        "best_visible_iou": summarize_numeric(state["best_visible_ious"]),
        "pred_visible_len": summarize_numeric(state["visible_lengths"]),
    }


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    dataset_root = Path(args.dataset_root)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    gt_counts = sorted({int(x) for x in (args.gt_counts if args.gt_counts is not None else [args.gt_count])})
    if not gt_counts or min(gt_counts) < 1:
        raise ValueError(f"--gt-counts must contain positive lane counts, got {gt_counts}.")
    imgsz = normalize_imgsz(args.imgsz, dataset="tusimple")
    bucket_edges = sorted({int(x) for x in args.bucket_edges})
    run_tag = args.run_tag or weight_run_tag(weights)
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    matcher = GCSHungarianMatcher(
        cost_point=float(args.match_cost_point),
        cost_curve=float(args.match_cost_curve),
        cost_exist=float(args.match_cost_exist),
        image_size=imgsz,
        min_overlap=int(args.match_min_overlap),
        max_x_dist=float(args.match_max_x_dist),
        match_gate_px=float(args.match_gate_px),
    )

    save_dir = resolve_save_dir(args, weights=weights, gt_counts=gt_counts)
    save_dir.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict] = []
    query_rows: list[dict] = []
    role_states: dict[tuple[str, str, str], dict] = defaultdict(empty_role_state)
    reason_counts: Counter[tuple[str, str, str, str]] = Counter()
    summary = {
        "config": {
            "run_tag": run_tag,
            "weights": str(weights.resolve()),
            "dataset_root": str(dataset_root.resolve()),
            "splits": list(args.splits),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "gt_counts": gt_counts,
            "short_visible_max": int(args.short_visible_max),
            "trace_scope": str(args.trace_scope),
            "ape_thr": float(args.ape_thr),
            "duplicate_ape_thr": float(args.duplicate_ape_thr),
            "duplicate_visible_iou_thr": float(args.duplicate_visible_iou_thr),
            "min_overlap": int(args.min_overlap),
            "match_cost_point": float(args.match_cost_point),
            "match_cost_curve": float(args.match_cost_curve),
            "match_cost_exist": float(args.match_cost_exist),
            "match_min_overlap": int(args.match_min_overlap),
            "match_max_x_dist": float(args.match_max_x_dist),
            "match_gate_px": float(args.match_gate_px),
            "device": str(args.device),
            "half": bool(args.half),
            "warmup": int(args.warmup),
            "max_images": int(args.max_images),
        },
        "splits": {},
    }

    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"query trace run={run_tag} gt_counts={gt_counts} trace_scope={args.trace_scope}")

    for split in args.splits:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels_gcs" / split
        images = collect_images(image_dir, max_images=int(args.max_images))
        split_seen = 0
        split_candidate = 0
        split_traced = 0
        split_failures = 0
        split_direction = Counter()
        split_candidate_buckets = Counter()
        split_traced_buckets = Counter()
        split_failure_buckets = Counter()
        split_gt_count_hist = Counter()
        split_pred_count_hist = Counter()
        split_confusion = Counter()
        t0 = time.perf_counter()

        if int(args.warmup) > 0 and images:
            warm_img = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
            if warm_img is None:
                raise FileNotFoundError(f"Failed to read warmup image: {images[0]}")
            warm_tensor = preprocess_image(warm_img, imgsz=imgsz, device=device, half=bool(args.half))
            for _ in range(int(args.warmup)):
                _ = model(warm_tensor)
            sync_if_cuda(device)

        for idx, image_path in enumerate(images, start=1):
            label_path = label_path_for_image(image_path, label_dir)
            gt_lanes, gt_valid = load_gcs_label(label_path)
            meta = load_label_metadata(label_path)
            split_seen += 1
            gt_count = int(meta["gt_count"])
            min_visible = int(meta["min_visible_points"])
            if gt_count not in gt_counts:
                continue
            if int(args.short_visible_max) > 0 and min_visible > int(args.short_visible_max):
                continue
            split_candidate += 1
            gt_key = str(gt_count)
            bucket = bucket_label(min_visible, bucket_edges)
            split_candidate_buckets[f"GT{gt_count}|{bucket}"] += 1

            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")
            assert_gcs_shape(img.shape[:2], imgsz, name="diagnostic image", context=f"{split}:{image_path}")
            tensor = preprocess_image(img, imgsz=imgsz, device=device, half=bool(args.half))
            preds = model(tensor)

            pred_points_t = preds["pred_points"][0].detach().float()
            pred_logits_t = preds["pred_logits"][0].detach().float()
            if pred_logits_t.ndim == 2 and pred_logits_t.shape[-1] == 1:
                pred_logits_t = pred_logits_t.squeeze(-1)
            pred_valid_logits = preds.get("pred_valid_logits")
            pred_valid_t = pred_valid_logits[0].detach().float() if pred_valid_logits is not None else None

            pred_lanes = decode_gcs_predictions(
                pred_points_t,
                pred_logits_t,
                pred_valid_logits=pred_valid_t,
                image_shape=img.shape[:2],
                score_thr=float(args.conf),
                point_valid_thr=float(args.point_valid_thr),
                min_points=int(args.min_points),
                max_det=int(args.max_det),
                nms_dist_px=float(args.nms_dist_px),
            )
            pred_count = len(pred_lanes)
            count_error = int(pred_count - gt_count)
            count_failure = count_error != 0
            split_gt_count_hist[gt_count] += 1
            split_pred_count_hist[pred_count] += 1
            split_confusion[f"{gt_count}->{pred_count}"] += 1
            if count_failure:
                split_failures += 1
                split_failure_buckets[f"GT{gt_count}|{bucket}"] += 1
                split_direction["overcount" if count_error > 0 else "undercount"] += 1
            if args.trace_scope == "failures" and not count_failure:
                continue
            split_traced += 1
            split_traced_buckets[f"GT{gt_count}|{bucket}"] += 1

            pred_points = pred_points_t.cpu().numpy().astype(np.float32)
            scores = pred_logits_t.sigmoid().cpu().numpy().astype(np.float32)
            q_count, k_count = pred_points.shape[:2]
            if pred_valid_t is not None:
                pred_valid_scores_t = pred_valid_t.sigmoid().detach().cpu()
                pred_valid_prob = pred_valid_scores_t.numpy().astype(np.float32)
                pred_valid_mask = (pred_valid_prob >= float(args.point_valid_thr)).astype(np.float32)
            else:
                pred_valid_scores_t = None
                pred_valid_prob = np.ones((q_count, k_count), dtype=np.float32)
                pred_valid_mask = np.ones((q_count, k_count), dtype=np.float32)
            visible_lengths = pred_visible_lengths(pred_valid_scores_t, float(args.point_valid_thr), q_count, k_count)

            h, w = int(img.shape[0]), int(img.shape[1])
            scale = np.array([w, h], dtype=np.float32)
            ape, mean_x, overlap = pair_geometry(pred_points, gt_lanes, gt_valid, scale, pred_valid=pred_valid_mask)
            viou = visible_iou_matrix(pred_valid_prob, gt_valid)
            best_gt_idx = np.argmin(ape, axis=1) if ape.shape[1] else np.full((q_count,), -1, dtype=np.int64)
            best_rows = np.arange(q_count)
            best_ape = ape[best_rows, best_gt_idx] if ape.shape[1] else np.full((q_count,), np.inf, dtype=np.float32)
            best_mean_x = mean_x[best_rows, best_gt_idx] if mean_x.shape[1] else np.full((q_count,), np.inf, dtype=np.float32)
            best_overlap = overlap[best_rows, best_gt_idx] if overlap.shape[1] else np.zeros((q_count,), dtype=np.int32)
            best_viou = viou[best_rows, best_gt_idx] if viou.shape[1] else np.zeros((q_count,), dtype=np.float32)

            raw_gate = (overlap >= int(args.min_overlap)) & np.isfinite(ape)
            raw_rows, raw_cols = gated_assignment(ape, gate=raw_gate)
            raw_assign = {int(row): int(col) for row, col in zip(raw_rows.tolist(), raw_cols.tolist())}

            gt_points_t = torch.from_numpy(gt_lanes).to(device=pred_points_t.device, dtype=pred_points_t.dtype)
            gt_valid_t = torch.from_numpy(gt_valid).to(device=pred_points_t.device, dtype=pred_points_t.dtype)
            train_src, train_tgt = matcher(
                pred_points_t.unsqueeze(0),
                pred_logits_t.unsqueeze(0),
                [gt_points_t],
                [gt_valid_t],
            )[0]
            train_assign = {int(src): int(tgt) for src, tgt in zip(train_src.tolist(), train_tgt.tolist())}

            decoded_queries = {int(lane["query"]): rank for rank, lane in enumerate(pred_lanes)}
            decoded_points = (
                np.stack([np.asarray(x["points_norm"], dtype=np.float32) for x in pred_lanes], axis=0)
                if pred_lanes
                else np.zeros((0, gt_lanes.shape[1], 2), dtype=np.float32)
            )
            decoded_valid = (
                np.stack([np.asarray(x.get("point_valid", np.ones(gt_lanes.shape[1])), dtype=np.float32) for x in pred_lanes], axis=0)
                if pred_lanes
                else np.zeros((0, gt_lanes.shape[1]), dtype=np.float32)
            )
            dec_ape, _, dec_overlap = pair_geometry(decoded_points, gt_lanes, gt_valid, scale, pred_valid=decoded_valid)
            dec_gate = (dec_overlap >= int(args.min_overlap)) & np.isfinite(dec_ape) & (dec_ape <= float(args.ape_thr))
            dec_rows, dec_cols = gated_assignment(dec_ape, gate=dec_gate)
            strict_query_matches = {
                int(pred_lanes[row]["query"]): int(col)
                for row, col in zip(dec_rows.tolist(), dec_cols.tolist())
                if row < len(pred_lanes)
            }

            image_row = {
                "run_tag": run_tag,
                "split": split,
                "raw_file": meta["raw_file"],
                "date": meta["date"],
                "image": str(image_path),
                "label": str(label_path),
                "gt_count": gt_count,
                "pred_count": int(pred_count),
                "count_correct": int(not count_failure),
                "count_error": count_error,
                "direction": "correct" if not count_failure else ("overcount" if count_error > 0 else "undercount"),
                "min_visible_points": min_visible,
                "min_visible_bucket": bucket,
                "visible_counts": ";".join(str(x) for x in meta["visible_counts"]),
                "decoded_queries": ";".join(str(q) for q in sorted(decoded_queries, key=lambda x: decoded_queries[x])),
            }
            image_rows.append(image_row)

            gt_visible_counts = [int(x) for x in gt_valid.sum(axis=1).tolist()]
            for q in range(q_count):
                decoded = q in decoded_queries
                strict_gt = strict_query_matches.get(q)
                raw_gt = raw_assign.get(q)
                train_gt = train_assign.get(q)
                role = "matched_qpos" if train_gt is not None else "unmatched_qminus"
                best_gt = int(best_gt_idx[q]) if int(best_gt_idx[q]) >= 0 else None
                best_gt_visible = gt_visible_counts[best_gt] if best_gt is not None and best_gt < len(gt_visible_counts) else 0
                reason = query_reason(
                    role=role,
                    decoded=decoded,
                    strict_matched=strict_gt is not None,
                    score=float(scores[q]),
                    conf=float(args.conf),
                    visible_len=int(visible_lengths[q]),
                    min_points=int(args.min_points),
                    best_ape=float(best_ape[q]),
                    best_visible_iou=float(best_viou[q]),
                    best_gt_visible=int(best_gt_visible),
                    duplicate_ape_thr=float(args.duplicate_ape_thr),
                    duplicate_visible_iou_thr=float(args.duplicate_visible_iou_thr),
                )
                row = {
                    **image_row,
                    "query": int(q),
                    "q_role": role,
                    "score": round(float(scores[q]), 8),
                    "above_conf": int(float(scores[q]) >= float(args.conf)),
                    "decoded": int(decoded),
                    "decoded_rank": decoded_queries.get(q),
                    "training_matched": int(train_gt is not None),
                    "training_matched_gt": train_gt,
                    "raw_geometry_matched": int(raw_gt is not None),
                    "raw_geometry_gt": raw_gt,
                    "strict_decoded_matched": int(strict_gt is not None),
                    "strict_decoded_gt": strict_gt,
                    "best_gt": best_gt,
                    "best_gt_visible_points": int(best_gt_visible),
                    "best_ape_px": finite_round(best_ape[q]),
                    "best_visible_iou": finite_round(best_viou[q], ndigits=6),
                    "min_distance_to_gt_px": finite_round(best_ape[q]),
                    "min_mean_x_dist_to_gt_px": finite_round(best_mean_x[q]),
                    "best_overlap_points": int(best_overlap[q]),
                    "pred_visible_len": int(visible_lengths[q]),
                    "reason": reason,
                }
                query_rows.append(row)
                role_key = (split, gt_key, role)
                update_role_state(role_states[role_key], row)
                reason_counts[(split, gt_key, role, reason)] += 1

            if idx % 250 == 0 or idx == len(images):
                print(f"{split}: processed {idx}/{len(images)}", flush=True)

        elapsed = time.perf_counter() - t0
        summary["splits"][split] = {
            "images_seen": int(split_seen),
            "candidate_images": int(split_candidate),
            "traced_images": int(split_traced),
            "failure_images": int(split_failures),
            "failure_rate_on_candidates": round(split_failures / max(split_candidate, 1), 6),
            "direction": {k: int(v) for k, v in sorted(split_direction.items())},
            "candidate_min_visible_buckets": {k: int(v) for k, v in sorted(split_candidate_buckets.items())},
            "traced_min_visible_buckets": {k: int(v) for k, v in sorted(split_traced_buckets.items())},
            "failure_min_visible_buckets": {k: int(v) for k, v in sorted(split_failure_buckets.items())},
            "gt_lanes_hist": {str(k): int(v) for k, v in sorted(split_gt_count_hist.items())},
            "pred_lanes_hist": {str(k): int(v) for k, v in sorted(split_pred_count_hist.items())},
            "count_confusion": {str(k): int(v) for k, v in sorted(split_confusion.items())},
            "seconds": round(float(elapsed), 3),
        }
        print(f"{split}: candidates={split_candidate}, traced={split_traced}, failures={split_failures}", flush=True)

    role_summary = {}
    for (split, gt_key, role), state in sorted(role_states.items()):
        role_summary.setdefault(split, {}).setdefault(f"GT{gt_key}", {})[role] = finalize_role_state(state)
    reason_summary = {}
    for (split, gt_key, role, reason), count in sorted(reason_counts.items()):
        reason_summary.setdefault(split, {}).setdefault(f"GT{gt_key}", {}).setdefault(role, {})[reason] = int(count)
    summary["role_summary"] = role_summary
    summary["reason_counts"] = reason_summary
    summary["all"] = {
        "images": int(len(image_rows)),
        "query_rows": int(len(query_rows)),
        "count_confusion": {
            str(k): int(v)
            for k, v in sorted(Counter(f"{row['gt_count']}->{row['pred_count']}" for row in image_rows).items())
        },
        "failure_direction": {k: int(v) for k, v in sorted(Counter(row["direction"] for row in image_rows).items())},
        "query_reason_counts": {k: int(v) for k, v in sorted(Counter(row["reason"] for row in query_rows).items())},
        "query_role_counts": {k: int(v) for k, v in sorted(Counter(row["q_role"] for row in query_rows).items())},
    }

    image_fields = [
        "run_tag",
        "split",
        "raw_file",
        "date",
        "image",
        "label",
        "gt_count",
        "pred_count",
        "count_correct",
        "count_error",
        "direction",
        "min_visible_points",
        "min_visible_bucket",
        "visible_counts",
        "decoded_queries",
    ]
    query_fields = image_fields + [
        "query",
        "q_role",
        "score",
        "above_conf",
        "decoded",
        "decoded_rank",
        "training_matched",
        "training_matched_gt",
        "raw_geometry_matched",
        "raw_geometry_gt",
        "strict_decoded_matched",
        "strict_decoded_gt",
        "best_gt",
        "best_gt_visible_points",
        "best_ape_px",
        "best_visible_iou",
        "min_distance_to_gt_px",
        "min_mean_x_dist_to_gt_px",
        "best_overlap_points",
        "pred_visible_len",
        "reason",
    ]
    role_rows = []
    for (split, gt_key, role), state in sorted(role_states.items()):
        stats = finalize_role_state(state)
        role_rows.append(
            {
                "run_tag": run_tag,
                "split": split,
                "gt_count": gt_key,
                "q_role": role,
                "queries": stats["queries"],
                "decoded": stats["decoded"],
                "decoded_rate": stats["decoded_rate"],
                "strict_decoded_matched": stats["strict_decoded_matched"],
                "strict_decoded_matched_rate": stats["strict_decoded_matched_rate"],
                "score_mean": stats["score"]["mean"],
                "score_median": stats["score"]["median"],
                "score_p90": stats["score"]["p90"],
                "best_ape_mean_px": stats["best_ape_px"]["mean"],
                "best_ape_median_px": stats["best_ape_px"]["median"],
                "best_visible_iou_mean": stats["best_visible_iou"]["mean"],
                "pred_visible_len_mean": stats["pred_visible_len"]["mean"],
            }
        )

    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(save_dir / "images.csv", image_rows, image_fields)
    write_csv(save_dir / "query_trace.csv", query_rows, query_fields)
    write_csv(
        save_dir / "role_summary.csv",
        role_rows,
        [
            "run_tag",
            "split",
            "gt_count",
            "q_role",
            "queries",
            "decoded",
            "decoded_rate",
            "strict_decoded_matched",
            "strict_decoded_matched_rate",
            "score_mean",
            "score_median",
            "score_p90",
            "best_ape_mean_px",
            "best_ape_median_px",
            "best_visible_iou_mean",
            "pred_visible_len_mean",
        ],
    )
    print(json.dumps(summary["all"], indent=2))
    print(f"saved: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
