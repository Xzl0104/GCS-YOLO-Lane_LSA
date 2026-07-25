from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    official_gt_contract_summary,
    read_tusimple_json_lines,
    resolve_tusimple_gt_json,
    tusimple_image_path,
    valid_tusimple_lanes,
)
from tools.diagnose_tusimple_raw_q12_filters import (  # noqa: E402
    DEFAULT_VAL_GT,
    _ape_px,
    _csv_value,
    _interp_lane_xs,
    _limit_records,
    _parse_date_session,
    _sync_if_cuda,
    _valid_gt_lanes_with_ids,
    _write_csv,
)
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.models.gcs.decode_summary import load_decode_yaml, validate_decode_yaml_for_model  # noqa: E402
from ultralytics.models.gcs.mode_utils import resolve_decode_mode  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions, lane_x_distance_px  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = ROOT / "runs/gcs_lane/query_alpha05_gt5short_geom_w2_v1/weights/official_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose extra lanes on GT5->6 TuSimple images from final query decode."
    )
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="TuSimple archive split.")
    parser.add_argument("--gt-json", default=None, help="Official TuSimple GT json-lines file.")
    parser.add_argument("--allow-noncanonical-gt", action="store_true")
    parser.add_argument(
        "--allow-test-oracle",
        action="store_true",
        help="Explicitly allow GT-based test diagnostics. Do not use output for threshold/checkpoint selection.",
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Query GCS checkpoint .pt.")
    parser.add_argument("--decode-yaml", default=None, help="Optional query official_best_decode.yaml.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.003)
    parser.add_argument("--point-valid-thr", type=float, default=0.5)
    parser.add_argument("--nms-dist-px", type=float, default=0.0)
    parser.add_argument("--max-det", type=int, default=6)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--valid-before-maxdet", action="store_true")
    parser.add_argument("--count-aware-topk", action="store_true")
    parser.add_argument("--count-aware-min-k", type=int, default=3)
    parser.add_argument("--count-aware-max-k", type=int, default=5)
    parser.add_argument("--count-aware-length-norm", type=float, default=12.0)
    parser.add_argument("--count-mode", choices=("score_sum", "count_logits"), default="score_sum")
    parser.add_argument("--target-gt-count", type=int, default=5)
    parser.add_argument("--target-pred-count", type=int, default=6)
    parser.add_argument("--match-min-overlap", type=int, default=3)
    parser.add_argument("--duplicate-dist-px", type=float, default=30.0)
    parser.add_argument("--duplicate-gt-ape-px", type=float, default=30.0)
    parser.add_argument("--spurious-gt-ape-px", type=float, default=50.0)
    parser.add_argument("--high-score-thr", type=float, default=0.5)
    parser.add_argument("--boundary-visible-max", type=int, default=10)
    parser.add_argument("--boundary-edge-margin-px", type=float, default=80.0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def _default_gt_json(args: argparse.Namespace) -> str | None:
    if args.gt_json:
        return args.gt_json
    if str(args.split).lower() == "val" and DEFAULT_VAL_GT.exists():
        return str(DEFAULT_VAL_GT)
    return None


def _weight_run_dir(weights: str | Path) -> Path | None:
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def _resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir is not None and str(args.save_dir).strip():
        return Path(args.save_dir)
    run_dir = _weight_run_dir(args.weights)
    tag = (
        f"gt5_extra_lanes_{args.split}_conf{float(args.conf):.4g}_"
        f"pvalid{float(args.point_valid_thr):.4g}_nms{float(args.nms_dist_px):.4g}_"
        f"maxdet{int(args.max_det)}_minp{int(args.min_points)}"
    ).replace(".", "p")
    if bool(args.valid_before_maxdet):
        tag += "_validbeforemaxdet"
    if run_dir is not None:
        return run_dir / tag
    return ROOT / "runs/gcs_lane/gt5_extra_lanes" / tag


def _apply_query_decode_yaml(args: argparse.Namespace, decode_yaml_cfg: dict) -> None:
    args.conf = float(decode_yaml_cfg["conf"])
    args.point_valid_thr = float(decode_yaml_cfg["point_valid_thr"])
    args.nms_dist_px = float(decode_yaml_cfg["nms_dist_px"])
    args.max_det = int(decode_yaml_cfg["max_det"])
    args.min_points = int(decode_yaml_cfg["min_points"])
    args.valid_before_maxdet = bool(decode_yaml_cfg.get("valid_before_maxdet", False))
    args.count_aware_topk = bool(decode_yaml_cfg["count_aware_topk"])
    args.count_aware_min_k = int(decode_yaml_cfg["count_aware_min_k"])
    args.count_aware_max_k = int(decode_yaml_cfg["count_aware_max_k"])
    args.count_aware_length_norm = float(decode_yaml_cfg["count_aware_length_norm"])
    args.count_mode = str(decode_yaml_cfg.get("count_mode", "score_sum"))


def _lane_visible_length(lane: dict) -> int:
    if "point_valid" not in lane:
        return int(np.asarray(lane["points_norm"]).shape[0])
    return int((np.asarray(lane["point_valid"], dtype=np.float32) > 0.5).sum())


def _lane_mean_valid_prob(lane: dict) -> float:
    scores = lane.get("point_valid_scores")
    valid = lane.get("point_valid")
    if scores is None:
        return 1.0
    scores_arr = np.asarray(scores, dtype=np.float32)
    if valid is None:
        return float(scores_arr.mean()) if scores_arr.size else 0.0
    mask = np.asarray(valid, dtype=np.float32) > 0.5
    return float(scores_arr[mask].mean()) if int(mask.sum()) > 0 else 0.0


def _visible_x_stats(lane: dict, image_shape: tuple[int, int]) -> dict:
    h, w = int(image_shape[0]), int(image_shape[1])
    pts = np.asarray(lane.get("visible_points", []), dtype=np.float32)
    if pts.size == 0:
        pts_norm = np.asarray(lane["points_norm"], dtype=np.float32)
        pts = pts_norm * np.array([w, h], dtype=np.float32).reshape(1, 2)
    xs = pts[:, 0] if pts.ndim == 2 and pts.shape[0] > 0 else np.asarray([], dtype=np.float32)
    ys = pts[:, 1] if pts.ndim == 2 and pts.shape[0] > 0 else np.asarray([], dtype=np.float32)
    return {
        "mean_x": float(xs.mean()) if xs.size else float("nan"),
        "min_x": float(xs.min()) if xs.size else float("nan"),
        "max_x": float(xs.max()) if xs.size else float("nan"),
        "bottom_x": float(xs[np.argmax(ys)]) if xs.size and ys.size else float("nan"),
        "top_x": float(xs[np.argmin(ys)]) if xs.size and ys.size else float("nan"),
    }


def _lane_distance_px(a: dict, b: dict, image_shape: tuple[int, int]) -> float:
    pts_a = torch.from_numpy(np.asarray(a["points_norm"], dtype=np.float32))
    pts_b = torch.from_numpy(np.asarray(b["points_norm"], dtype=np.float32))
    valid_a = torch.from_numpy(np.asarray(a.get("point_valid", np.ones(pts_a.shape[0])), dtype=np.float32) > 0.5)
    valid_b = torch.from_numpy(np.asarray(b.get("point_valid", np.ones(pts_b.shape[0])), dtype=np.float32) > 0.5)
    return float(lane_x_distance_px(pts_a, pts_b, image_shape=image_shape, valid_a=valid_a, valid_b=valid_b))


def _best_assignment(ape: np.ndarray) -> tuple[dict[int, int], int, float]:
    """Assign all GT lanes to distinct predictions, leaving one extra prediction."""
    n_pred, n_gt = int(ape.shape[0]), int(ape.shape[1])
    if n_pred <= n_gt:
        return {i: i for i in range(min(n_pred, n_gt))}, -1, float("inf")

    best_total = float("inf")
    best_map: dict[int, int] = {}
    best_extra = -1
    pred_indices = range(n_pred)
    gt_indices = range(n_gt)
    for extra_idx in pred_indices:
        remaining = [i for i in pred_indices if i != extra_idx]
        for assigned_preds in itertools.permutations(remaining, n_gt):
            total = 0.0
            valid = True
            mapping: dict[int, int] = {}
            for gt_i, pred_i in zip(gt_indices, assigned_preds):
                value = float(ape[pred_i, gt_i])
                if not math.isfinite(value):
                    valid = False
                    break
                total += value
                mapping[int(pred_i)] = int(gt_i)
            if valid and total < best_total:
                best_total = total
                best_map = mapping
                best_extra = int(extra_idx)
    return best_map, best_extra, best_total


def _percentile(values: list[float], pct: float) -> float | None:
    finite = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * float(pct) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    return finite[lo] * (hi - pos) + finite[hi] * (pos - lo)


def _summarize(rows: list[dict]) -> dict:
    total = len(rows)
    apes = [float(r["extra_best_gt_ape"]) for r in rows if math.isfinite(float(r["extra_best_gt_ape"]))]
    dists = [float(r["nearest_retained_mean_x_dist"]) for r in rows if math.isfinite(float(r["nearest_retained_mean_x_dist"]))]
    scores = [float(r["exist_score"]) for r in rows]
    lengths = [int(r["visible_length"]) for r in rows]
    return {
        "gt5_to_6_images": int(total),
        "duplicate_count": int(sum(1 for r in rows if bool(r["is_duplicate"]))),
        "spurious_count": int(sum(1 for r in rows if bool(r["is_spurious"]))),
        "boundary_pseudo_count": int(sum(1 for r in rows if bool(r["is_boundary_pseudo"]))),
        "duplicate_rate": None if total == 0 else round(sum(1 for r in rows if bool(r["is_duplicate"])) / total, 6),
        "spurious_rate": None if total == 0 else round(sum(1 for r in rows if bool(r["is_spurious"])) / total, 6),
        "boundary_pseudo_rate": None if total == 0 else round(sum(1 for r in rows if bool(r["is_boundary_pseudo"])) / total, 6),
        "extra_best_gt_ape_mean": None if not apes else round(sum(apes) / len(apes), 6),
        "extra_best_gt_ape_p50": None if not apes else round(median(apes), 6),
        "extra_best_gt_ape_p90": None if not apes else round(float(_percentile(apes, 90.0)), 6),
        "nearest_retained_dist_mean": None if not dists else round(sum(dists) / len(dists), 6),
        "nearest_retained_dist_p50": None if not dists else round(median(dists), 6),
        "nearest_retained_dist_p90": None if not dists else round(float(_percentile(dists, 90.0)), 6),
        "exist_score_mean": None if not scores else round(sum(scores) / len(scores), 6),
        "exist_score_p50": None if not scores else round(median(scores), 6),
        "visible_length_mean": None if not lengths else round(sum(lengths) / len(lengths), 6),
        "visible_length_p50": None if not lengths else round(median(lengths), 6),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if str(args.split).lower() == "test" and not bool(args.allow_test_oracle):
        raise ValueError(
            "GT5 extra-lane diagnostic uses GT to classify test errors. "
            "Pass --allow-test-oracle only for a reporting-only audit, never for selection."
        )

    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_json = _default_gt_json(args)
    gt_path = resolve_tusimple_gt_json(archive_root, split=args.split, gt_json=gt_json)
    gt_records = _limit_records(read_tusimple_json_lines(gt_path), args.max_images)
    if not gt_records:
        raise ValueError(f"No TuSimple GT records found in {gt_path}.")
    gt_contract = official_gt_contract_summary(
        split=args.split,
        gt_json=gt_path,
        gt_records=gt_records,
        allow_noncanonical_gt=bool(args.allow_noncanonical_gt),
    )
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)

    device_obj = select_device(args.device)
    model = load_gcs_model(args.weights, device=device_obj, half=args.half, gcs_imgsz=imgsz)
    if args.decode_yaml:
        _, decode_yaml_cfg = load_decode_yaml(args.decode_yaml)
        decode_mode = resolve_decode_mode(decode_yaml_cfg.get("decode_mode"), model)
        validate_decode_yaml_for_model(decode_yaml_cfg, model_mode=decode_mode)
        if decode_mode != "query":
            raise ValueError(f"GT5 extra-lane diagnostic supports query decode only, got {decode_mode!r}.")
        _apply_query_decode_yaml(args, decode_yaml_cfg)
    else:
        decode_mode = resolve_decode_mode("auto", model)
        if decode_mode != "query":
            raise ValueError(f"GT5 extra-lane diagnostic supports query decode only, got {decode_mode!r}.")

    warn_max_det_mismatch(args.weights, max_det=args.max_det, context="TuSimple GT5 extra-lane diagnostic")

    if args.warmup > 0 and gt_records:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device_obj, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    save_dir = _resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    all_target_images = 0
    infer_time_s = 0.0
    post_time_s = 0.0

    for record in gt_records:
        raw_file = str(record["raw_file"])
        gt_lanes = _valid_gt_lanes_with_ids(record)
        gt_count = int(len(valid_tusimple_lanes(record.get("lanes", []))))
        if gt_count != int(args.target_gt_count):
            continue
        all_target_images += 1

        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_shape = img.shape[:2]
        h_samples = [float(x) for x in record["h_samples"]]
        tensor = preprocess_image(img, imgsz, device=device_obj, half=args.half)

        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        t1 = time.perf_counter()

        pred_valid = preds.get("pred_valid_logits")
        pred_count_logits = preds.get("pred_count_logits")
        pred_quality_logits = preds.get("pred_quality_logits")
        final_lanes = decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_quality_logits=pred_quality_logits[0] if pred_quality_logits is not None else None,
            pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
            pred_count_logits=pred_count_logits[0] if pred_count_logits is not None else None,
            image_shape=image_shape,
            score_thr=args.conf,
            point_valid_thr=args.point_valid_thr,
            min_points=args.min_points,
            max_det=args.max_det,
            nms_dist_px=args.nms_dist_px,
            valid_before_maxdet=args.valid_before_maxdet,
            count_aware_topk=args.count_aware_topk,
            count_aware_min_k=args.count_aware_min_k,
            count_aware_max_k=args.count_aware_max_k,
            count_aware_length_norm=args.count_aware_length_norm,
            count_mode=args.count_mode,
        )
        tusimple_lanes = gcs_lanes_to_tusimple_lanes(final_lanes, h_samples, image_shape=image_shape)
        t2 = time.perf_counter()
        infer_time_s += t1 - t0
        post_time_s += t2 - t1

        if len(tusimple_lanes) != int(args.target_pred_count):
            continue

        pred_xs = [
            _interp_lane_xs(np.asarray(lane["points_norm"], dtype=np.float32), h_samples, image_shape)
            for lane in final_lanes
        ]
        gt_lane_ids = [lane_id for lane_id, _ in gt_lanes]
        ape = np.full((len(final_lanes), len(gt_lanes)), float("inf"), dtype=np.float32)
        overlap = np.zeros((len(final_lanes), len(gt_lanes)), dtype=np.int32)
        for pred_i, xs in enumerate(pred_xs):
            for gt_i, (_, gt_lane) in enumerate(gt_lanes):
                value, ov = _ape_px(xs, gt_lane, min_overlap=args.match_min_overlap)
                ape[pred_i, gt_i] = value
                overlap[pred_i, gt_i] = ov

        assignment, extra_idx, assignment_total_ape = _best_assignment(ape)
        if extra_idx < 0:
            continue
        extra = final_lanes[extra_idx]
        best_gt_i = int(np.nanargmin(ape[extra_idx]))
        best_gt_id = int(gt_lane_ids[best_gt_i])
        best_gt_ape = float(ape[extra_idx, best_gt_i])
        best_gt_overlap = int(overlap[extra_idx, best_gt_i])

        nearest_idx = -1
        nearest_dist = float("inf")
        for pred_i, lane in enumerate(final_lanes):
            if pred_i == extra_idx:
                continue
            dist = _lane_distance_px(extra, lane, image_shape=image_shape)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = int(pred_i)
        nearest_assigned_gt_i = assignment.get(nearest_idx, -1)
        nearest_assigned_gt_id = "" if nearest_assigned_gt_i < 0 else int(gt_lane_ids[nearest_assigned_gt_i])

        assigned_same_gt_exists = any(gt_i == best_gt_i for pred_i, gt_i in assignment.items() if pred_i != extra_idx)
        visible_length = _lane_visible_length(extra)
        mean_valid_prob = _lane_mean_valid_prob(extra)
        x_stats = _visible_x_stats(extra, image_shape)
        w = int(image_shape[1])
        near_left = math.isfinite(x_stats["mean_x"]) and x_stats["mean_x"] <= float(args.boundary_edge_margin_px)
        near_right = math.isfinite(x_stats["mean_x"]) and x_stats["mean_x"] >= float(w) - float(args.boundary_edge_margin_px)
        is_image_edge_boundary = bool(visible_length <= int(args.boundary_visible_max) and (near_left or near_right))
        is_side_boundary_short_far = bool(
            visible_length <= int(args.boundary_visible_max)
            and best_gt_i in {0, len(gt_lanes) - 1}
            and best_gt_ape >= float(args.spurious_gt_ape_px)
        )
        is_duplicate = bool(
            nearest_dist <= float(args.duplicate_dist_px)
            or (assigned_same_gt_exists and best_gt_ape <= float(args.duplicate_gt_ape_px))
        )
        is_spurious = bool(best_gt_ape >= float(args.spurious_gt_ape_px) and float(extra["score"]) >= float(args.high_score_thr))
        is_boundary_pseudo = bool(is_image_edge_boundary or is_side_boundary_short_far)
        if is_duplicate:
            category = "duplicate"
        elif is_boundary_pseudo:
            category = "boundary_pseudo"
        elif is_spurious:
            category = "spurious"
        else:
            category = "ambiguous"

        date, session = _parse_date_session(raw_file)
        rows.append(
            {
                "raw_file": raw_file,
                "date": date,
                "session": session,
                "gt_count": gt_count,
                "pred_count": len(tusimple_lanes),
                "extra_lane_index": int(extra_idx),
                "extra_query": int(extra["query"]),
                "extra_best_gt_id": best_gt_id,
                "extra_best_gt_ape": best_gt_ape,
                "extra_best_gt_overlap": best_gt_overlap,
                "assigned_same_gt_exists": assigned_same_gt_exists,
                "exist_score": float(extra["score"]),
                "mean_valid_prob": mean_valid_prob,
                "visible_length": int(visible_length),
                "nearest_retained_lane_index": nearest_idx,
                "nearest_retained_query": "" if nearest_idx < 0 else int(final_lanes[nearest_idx]["query"]),
                "nearest_retained_assigned_gt_id": nearest_assigned_gt_id,
                "nearest_retained_mean_x_dist": nearest_dist,
                "is_duplicate": is_duplicate,
                "is_spurious": is_spurious,
                "is_boundary_pseudo": is_boundary_pseudo,
                "is_image_edge_boundary": is_image_edge_boundary,
                "is_side_boundary_short_far": is_side_boundary_short_far,
                "category": category,
                "extra_mean_x": x_stats["mean_x"],
                "extra_min_x": x_stats["min_x"],
                "extra_max_x": x_stats["max_x"],
                "extra_bottom_x": x_stats["bottom_x"],
                "extra_top_x": x_stats["top_x"],
                "near_left_edge": near_left,
                "near_right_edge": near_right,
                "assignment_total_ape": float(assignment_total_ape),
                "decode_conf": float(args.conf),
                "decode_point_valid_thr": float(args.point_valid_thr),
                "decode_nms_dist_px": float(args.nms_dist_px),
                "decode_max_det": int(args.max_det),
                "decode_min_points": int(args.min_points),
                "decode_valid_before_maxdet": bool(args.valid_before_maxdet),
            }
        )

    fields = [
        "raw_file",
        "date",
        "session",
        "gt_count",
        "pred_count",
        "extra_lane_index",
        "extra_query",
        "extra_best_gt_id",
        "extra_best_gt_ape",
        "extra_best_gt_overlap",
        "assigned_same_gt_exists",
        "exist_score",
        "mean_valid_prob",
        "visible_length",
        "nearest_retained_lane_index",
        "nearest_retained_query",
        "nearest_retained_assigned_gt_id",
        "nearest_retained_mean_x_dist",
        "is_duplicate",
        "is_spurious",
        "is_boundary_pseudo",
        "is_image_edge_boundary",
        "is_side_boundary_short_far",
        "category",
        "extra_mean_x",
        "extra_min_x",
        "extra_max_x",
        "extra_bottom_x",
        "extra_top_x",
        "near_left_edge",
        "near_right_edge",
        "assignment_total_ape",
        "decode_conf",
        "decode_point_valid_thr",
        "decode_nms_dist_px",
        "decode_max_det",
        "decode_min_points",
        "decode_valid_before_maxdet",
    ]
    _write_csv(save_dir / "gt5_extra_lanes.csv", rows, fields)

    by_category: dict[str, int] = {}
    for row in rows:
        by_category[str(row["category"])] = by_category.get(str(row["category"]), 0) + 1
    summary = {
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "decode_yaml": None if not args.decode_yaml else str(Path(args.decode_yaml).resolve()),
            "archive_root": str(archive_root.resolve()),
            "split": args.split,
            "gt_json": str(gt_path.resolve()),
            "save_dir": str(save_dir.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "valid_before_maxdet": bool(args.valid_before_maxdet),
            "target_gt_count": int(args.target_gt_count),
            "target_pred_count": int(args.target_pred_count),
            "duplicate_dist_px": float(args.duplicate_dist_px),
            "duplicate_gt_ape_px": float(args.duplicate_gt_ape_px),
            "spurious_gt_ape_px": float(args.spurious_gt_ape_px),
            "high_score_thr": float(args.high_score_thr),
            "boundary_visible_max": int(args.boundary_visible_max),
            "boundary_edge_margin_px": float(args.boundary_edge_margin_px),
            "allow_test_oracle": bool(args.allow_test_oracle),
        },
        "target_gt_images": int(all_target_images),
        "summary": _summarize(rows),
        "category_counts": by_category,
        "outputs": {
            "gt5_extra_lanes": str((save_dir / "gt5_extra_lanes.csv").resolve()),
        },
        "timing": {
            "avg_inference_ms_over_gt5": round(infer_time_s * 1000.0 / max(all_target_images, 1), 4),
            "avg_postprocess_ms_over_gt5": round(post_time_s * 1000.0 / max(all_target_images, 1), 4),
        },
        **gt_contract,
    }
    (save_dir / "gt5_extra_lanes_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["summary"] | {"category_counts": by_category}, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
