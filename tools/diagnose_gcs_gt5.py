from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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
    normalize_tusimple_gt_record,
    read_tusimple_json_lines,
    tusimple_image_path,
)
from tools.infer_gcs import count_calibration_from_args, count_head_decode_kwargs_from_args, load_gcs_model, preprocess_image  # noqa: E402
from ultralytics.utils.gcs_postprocess import (  # noqa: E402
    GCS_DEFAULT_MAX_DET,
    count_head_decode_meta,
    decode_gcs_predictions,
    lane_x_distance_px,
)
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = (
    ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_q12_tusimple_hard45_count03" / "weights" / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose where the 5th lane dies on TuSimple GT=5 samples."
    )
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="test", choices=("test", "train", "val"), help="TuSimple archive split.")
    parser.add_argument("--gt-json", default=None, help="TuSimple official GT json-lines file.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.25, help="Raw exist threshold used by formal postprocess.")
    parser.add_argument("--point-valid-thr", type=float, default=0.5, help="Per-point visibility threshold.")
    parser.add_argument("--min-points", type=int, default=6, help="Minimum contiguous visible anchors for a kept lane.")
    parser.add_argument("--nms-dist-px", type=float, default=18.0, help="Lane-NMS distance in original image pixels. 0 disables.")
    parser.add_argument("--max-det", type=int, default=GCS_DEFAULT_MAX_DET, help="Maximum formal decoded lanes.")
    count_head_group = parser.add_mutually_exclusive_group()
    count_head_group.add_argument("--use-count-head-decode", dest="use_count_head_decode", action="store_true", help="Use explicit Count Head K for final Top-K lane selection.")
    count_head_group.add_argument("--no-count-head-decode", dest="use_count_head_decode", action="store_false", help="Disable Count Head K and use max-det rank selection.")
    parser.set_defaults(use_count_head_decode=True)
    parser.add_argument("--count-head-temp", type=float, default=1.0)
    parser.add_argument("--candidate-conf", type=float, default=0.05)
    parser.add_argument("--candidate-point-valid-thr", type=float, default=0.20)
    parser.add_argument("--candidate-min-points", type=int, default=5)
    rescue_group = parser.add_mutually_exclusive_group()
    rescue_group.add_argument("--enable-rescue-candidate-pool", dest="enable_rescue_candidate_pool", action="store_true")
    rescue_group.add_argument("--no-enable-rescue-candidate-pool", dest="enable_rescue_candidate_pool", action="store_false")
    parser.set_defaults(enable_rescue_candidate_pool=True)
    parser.add_argument("--rescue-candidate-conf", type=float, default=0.005)
    parser.add_argument("--rescue-candidate-point-valid-thr", type=float, default=0.08)
    parser.add_argument("--rescue-candidate-min-points", type=int, default=4)
    parser.add_argument("--final-min-points", type=int, default=6)
    parser.add_argument("--fifth-min-points", type=int, default=5)
    parser.add_argument("--line-nms-min-overlap", type=int, default=6)
    parser.add_argument("--line-nms-rescue-dist-px", type=float, default=30.0)
    last_lane_group = parser.add_mutually_exclusive_group()
    last_lane_group.add_argument("--last-lane-rescue", dest="last_lane_rescue", action="store_true")
    last_lane_group.add_argument("--no-last-lane-rescue", dest="last_lane_rescue", action="store_false")
    parser.set_defaults(last_lane_rescue=False)
    parser.add_argument("--last-lane-rescue-min-policy-count", type=int, default=4)
    parser.add_argument("--last-lane-rescue-conf-thr", type=float, default=None)
    parser.add_argument("--last-lane-rescue-point-valid-thr", type=float, default=0.08)
    parser.add_argument("--last-lane-rescue-min-points", type=int, default=4)
    parser.add_argument("--last-lane-rescue-mean-valid-thr", type=float, default=0.40)
    parser.add_argument("--last-lane-rescue-quality-thr", type=float, default=0.50)
    parser.add_argument("--last-lane-rescue-dist-px", type=float, default=24.0)
    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument("--edge-last-lane-rescue", dest="edge_last_lane_rescue", action="store_true")
    edge_group.add_argument("--no-edge-last-lane-rescue", dest="edge_last_lane_rescue", action="store_false")
    parser.set_defaults(edge_last_lane_rescue=False)
    parser.add_argument("--edge-rescue-conf-thr", type=float, default=0.02)
    parser.add_argument("--edge-rescue-point-valid-thr", type=float, default=0.06)
    parser.add_argument("--edge-rescue-min-points", type=int, default=4)
    parser.add_argument("--edge-rescue-mean-valid-thr", type=float, default=0.35)
    parser.add_argument("--edge-rescue-quality-thr", type=float, default=0.45)
    parser.add_argument("--edge-rescue-outside-gap-px", type=float, default=28.0)
    parser.add_argument("--edge-rescue-dist-px", type=float, default=24.0)
    parser.add_argument("--edge-rescue-min-policy-count", type=int, default=4)
    edge_upgrade_group = parser.add_mutually_exclusive_group()
    edge_upgrade_group.add_argument("--edge-count4-to5-upgrade", dest="edge_count4_to5_upgrade", action="store_true")
    edge_upgrade_group.add_argument(
        "--no-edge-count4-to5-upgrade",
        dest="edge_count4_to5_upgrade",
        action="store_false",
    )
    parser.set_defaults(edge_count4_to5_upgrade=True)
    parser.add_argument("--edge-count4-to5-prob-margin", type=float, default=0.20)
    parser.add_argument("--soft-count-decision", action="store_true", help="Choose K by candidate quality when Count Head probabilities are close.")
    parser.add_argument("--soft-count-prob-margin", type=float, default=0.08)
    parser.add_argument("--soft-count-quality-weight", type=float, default=1.0)
    parser.add_argument("--soft-count-prior-weight", type=float, default=0.5)
    parser.add_argument("--soft-count-duplicate-penalty", type=float, default=1.0)
    parser.add_argument("--soft-count-invalid-penalty", type=float, default=1.0)
    parser.add_argument(
        "--fifth-rescue-point-valid-thrs",
        nargs="+",
        type=float,
        default=[0.10, 0.12, 0.15],
        help="Point-valid thresholds used only for Count Head K<5 rank5 rescue diagnostics.",
    )
    parser.add_argument(
        "--fifth-rescue-close-gaps",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Small invalid gaps to close when measuring rank5 rescue visibility.",
    )
    parser.add_argument("--rank-len-norm", type=float, default=12.0, help="Length saturation used by rank quality.")
    parser.add_argument("--s5-low-thr", type=float, default=0.25, help="Rank-score threshold for weak s5 diagnosis.")
    parser.add_argument("--exist-low-thr", type=float, default=0.25, help="Raw exist threshold for weak exist diagnosis.")
    parser.add_argument("--topk", type=int, default=6, help="Number of rank-sorted raw candidates to export.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit GT=5 images after filtering. 0 means all.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--save-dir", default=None, help="Output directory. Defaults under the weight run.")
    return parser.parse_args()


def _weight_run_dir(weights: str | Path) -> Path | None:
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    run_dir = _weight_run_dir(args.weights)
    base = run_dir if run_dir is not None else ROOT / "runs" / "gcs_lane"
    conf = str(float(args.conf)).replace(".", "p")
    pvalid = str(float(args.point_valid_thr)).replace(".", "p")
    cand_conf = str(float(args.candidate_conf)).replace(".", "p")
    cand_pvalid = str(float(args.candidate_point_valid_thr)).replace(".", "p")
    return base / "gt5_diagnostics" / (
        f"{args.split}_conf{conf}_pvalid{pvalid}_candconf{cand_conf}_"
        f"candpvalid{cand_pvalid}_minp{int(args.min_points)}"
    )


def _close_small_false_gaps(mask: np.ndarray, close_gap: int = 0) -> np.ndarray:
    """Fill false runs of length <= close_gap when they are bracketed by true values."""
    mask = np.asarray(mask, dtype=bool).copy()
    close_gap = int(close_gap)
    if close_gap <= 0 or mask.size == 0:
        return mask
    i = 0
    while i < mask.size:
        if mask[i]:
            i += 1
            continue
        start = i
        while i < mask.size and not mask[i]:
            i += 1
        end = i
        if start > 0 and end < mask.size and mask[start - 1] and mask[end] and (end - start) <= close_gap:
            mask[start:end] = True
    return mask


def _longest_true_run(mask: np.ndarray, close_gap: int = 0) -> tuple[int | None, int | None]:
    best_start: int | None = None
    best_len = 0
    start: int | None = None
    mask = _close_small_false_gaps(mask, close_gap=close_gap)
    values = [bool(x) for x in mask.tolist()] + [False]
    for i, value in enumerate(values):
        if value and start is None:
            start = i
        elif not value and start is not None:
            length = i - start
            if length > best_len:
                best_start = start
                best_len = length
            start = None
    if best_start is None or best_len <= 0:
        return None, None
    return int(best_start), int(best_start + best_len)


def _candidate_quality(
    lane_points: np.ndarray,
    exist_score: float,
    valid_scores: np.ndarray | None,
    image_shape: tuple[int, int],
    point_valid_thr: float,
    min_points: int,
    rank_len_norm: float,
    close_gap: int = 0,
) -> dict:
    if valid_scores is None:
        visible = np.ones((lane_points.shape[0],), dtype=bool)
        valid_scores = np.ones((lane_points.shape[0],), dtype=np.float32)
    else:
        visible = np.asarray(valid_scores, dtype=np.float32) >= float(point_valid_thr)

    rank_valid_count = int(np.asarray(visible, dtype=bool).sum())
    mean_valid_score_all = float(np.asarray(valid_scores, dtype=np.float32).mean()) if len(valid_scores) else 0.0
    visible = _close_small_false_gaps(visible, close_gap=close_gap)
    start, end = _longest_true_run(visible, close_gap=0)
    if start is None or end is None:
        valid_points = 0
        lane_length = 0
        mean_valid_score = 0.0
        smooth_factor = 0.5
        jump_factor = 0.5
        lane_length_px = 0.0
    else:
        segment = lane_points[start:end]
        segment_scores = np.asarray(valid_scores[start:end], dtype=np.float32)
        valid_points = int(segment.shape[0])
        lane_length = int(end - start)
        mean_valid_score = float(segment_scores.mean()) if valid_points > 0 else 0.0
        h, w = int(image_shape[0]), int(image_shape[1])
        points_px = segment * np.array([float(w), float(h)], dtype=np.float32)
        x = points_px[:, 0]
        if valid_points >= 3:
            ddx = float(np.mean(np.abs(x[2:] - 2.0 * x[1:-1] + x[:-2])))
            smooth_factor = float(np.exp(-ddx / 60.0))
        else:
            smooth_factor = 0.5
        if valid_points >= 2:
            max_dx = float(np.max(np.abs(x[1:] - x[:-1])))
            jump_factor = float(np.exp(-max(max_dx - 80.0, 0.0) / 80.0))
            lane_length_px = float(np.sum(np.linalg.norm(points_px[1:] - points_px[:-1], axis=1)))
        else:
            jump_factor = 0.5
        lane_length_px = 0.0

    length_factor = min(1.0, float(lane_length) / max(float(rank_len_norm), 1e-6))
    total_points = max(int(lane_points.shape[0]), 1)
    valid_count_score = float(
        np.clip((float(rank_valid_count) - float(min_points) + 1.0) / float(total_points), 0.0, 1.0)
    )
    quality_proxy = mean_valid_score_all * valid_count_score
    rank_score = float(exist_score) * quality_proxy
    return {
        "rank_score": float(rank_score),
        "exist_score": float(exist_score),
        "quality_proxy": float(quality_proxy),
        "valid_points": int(valid_points),
        "rank_valid_count": int(rank_valid_count),
        "lane_length": int(lane_length),
        "lane_length_px": float(lane_length_px),
        "mean_valid_score": float(mean_valid_score_all),
        "valid_count_score": float(valid_count_score),
        "length_factor": float(length_factor),
        "smooth_factor": float(smooth_factor),
        "jump_factor": float(jump_factor),
        "visible_start": None if start is None else int(start),
        "visible_end": None if end is None else int(end - 1),
        "points_norm": lane_points.astype(np.float32),
        "point_valid": visible.astype(bool),
        "point_valid_thr": float(point_valid_thr),
        "close_gap": int(close_gap),
    }


def rank_query_candidates(
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None,
    image_shape: tuple[int, int],
    point_valid_thr: float,
    min_points: int,
    rank_len_norm: float,
    close_gap: int = 0,
) -> list[dict]:
    if pred_logits.ndim == 2 and pred_logits.shape[-1] == 1:
        pred_logits = pred_logits.squeeze(-1)
    points = pred_points.detach().float().cpu().clamp(0.0, 1.0)
    scores = pred_logits.detach().float().cpu().sigmoid().numpy().astype(np.float32)
    valid_scores = pred_valid_logits.detach().float().cpu().sigmoid() if pred_valid_logits is not None else None

    rows: list[dict] = []
    for query in range(int(points.shape[0])):
        order = torch.argsort(points[query, :, 1], descending=True, stable=True)
        lane_points = points[query][order].numpy().astype(np.float32)
        valid_i = None
        if valid_scores is not None:
            valid_i = valid_scores[query][order].numpy().astype(np.float32)
        quality = _candidate_quality(
            lane_points=lane_points,
            exist_score=float(scores[query]),
            valid_scores=valid_i,
            image_shape=image_shape,
            point_valid_thr=point_valid_thr,
            min_points=min_points,
            rank_len_norm=rank_len_norm,
            close_gap=close_gap,
        )
        quality["query"] = int(query)
        rows.append(quality)
    rows.sort(key=lambda x: float(x["rank_score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def decoded_query_set(
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None,
    pred_count_logits: torch.Tensor | None,
    pred_quality_logits: torch.Tensor | None,
    image_shape: tuple[int, int],
    conf: float,
    point_valid_thr: float,
    min_points: int,
    nms_dist_px: float,
    max_det: int | None,
    count_calibration: dict | None,
    decode_kwargs: dict | None = None,
    pred_count_boundary_logits: torch.Tensor | None = None,
) -> set[int]:
    lanes = decoded_lanes(
        pred_points,
        pred_logits,
        pred_valid_logits=pred_valid_logits,
        pred_count_logits=pred_count_logits,
        pred_quality_logits=pred_quality_logits,
        image_shape=image_shape,
        conf=conf,
        point_valid_thr=point_valid_thr,
        min_points=min_points,
        max_det=max_det,
        nms_dist_px=nms_dist_px,
        count_calibration=count_calibration,
        decode_kwargs=decode_kwargs,
        pred_count_boundary_logits=pred_count_boundary_logits,
    )
    return {int(x["query"]) for x in lanes}


def decoded_lanes(
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None,
    pred_count_logits: torch.Tensor | None,
    pred_quality_logits: torch.Tensor | None,
    image_shape: tuple[int, int],
    conf: float,
    point_valid_thr: float,
    min_points: int,
    nms_dist_px: float,
    max_det: int | None,
    count_calibration: dict | None,
    decode_kwargs: dict | None = None,
    return_meta: bool = False,
    pred_count_boundary_logits: torch.Tensor | None = None,
) -> list[dict] | tuple[list[dict], dict]:
    decode_kwargs = dict(decode_kwargs or {})
    return decode_gcs_predictions(
        pred_points,
        pred_logits,
        pred_valid_logits=pred_valid_logits,
        pred_count_logits=pred_count_logits,
        pred_count_boundary_logits=pred_count_boundary_logits,
        pred_quality_logits=pred_quality_logits,
        image_shape=image_shape,
        score_thr=conf,
        point_valid_thr=point_valid_thr,
        min_points=min_points,
        max_det=max_det,
        nms_dist_px=nms_dist_px,
        count_calibration=count_calibration,
        return_meta=return_meta,
        **decode_kwargs,
    )


def rank5_required_min_points(args: argparse.Namespace, decode_kwargs: dict) -> int:
    """Return the final visible-anchor requirement for the selected 5th output lane."""
    final_min_points = int(decode_kwargs.get("final_min_points", args.min_points))
    if bool(decode_kwargs.get("use_count_head_decode", True)):
        return int(decode_kwargs.get("fifth_min_points", min(final_min_points, 5)))
    return final_min_points


def no_count_head_decode_kwargs(args: argparse.Namespace, decode_kwargs: dict) -> dict:
    """Build a decode config that keeps the same candidate/final gates but ignores Count Head K."""
    out = {
        "use_count_head_decode": False,
        "candidate_score_thr": float(args.candidate_conf),
        "candidate_point_valid_thr": float(args.candidate_point_valid_thr),
        "candidate_min_points": int(args.candidate_min_points),
        "enable_rescue_candidate_pool": bool(args.enable_rescue_candidate_pool),
        "rescue_candidate_score_thr": float(args.rescue_candidate_conf),
        "rescue_candidate_point_valid_thr": float(args.rescue_candidate_point_valid_thr),
        "rescue_candidate_min_points": int(args.rescue_candidate_min_points),
        "final_min_points": int(decode_kwargs.get("final_min_points", args.min_points)),
        "line_nms_min_overlap": int(args.line_nms_min_overlap),
        "line_nms_rescue_dist_px": float(args.line_nms_rescue_dist_px),
    }
    if bool(decode_kwargs.get("use_count_head_decode", True)):
        out["fifth_min_points"] = int(decode_kwargs.get("fifth_min_points", min(out["final_min_points"], 5)))
    return out


def count_head_meta_for_image(
    pred_count_logits: torch.Tensor | None,
    args: argparse.Namespace,
    decode_kwargs: dict,
    pred_count_boundary_logits: torch.Tensor | None = None,
) -> dict | None:
    """Return Count Head K metadata for one image using the same policy as decode."""
    return count_head_decode_meta(
        pred_count_logits,
        pred_count_boundary_logits,
        use_count_head_decode=bool(decode_kwargs.get("use_count_head_decode", True)),
        count_head_temperature=float(decode_kwargs.get("count_head_temperature", args.count_head_temp)),
        dataset_name=str(decode_kwargs.get("dataset_name", args.dataset)),
        max_det=args.max_det,
    )


def deletion_stage(
    rank5: dict | None,
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None,
    pred_count_logits: torch.Tensor | None,
    pred_quality_logits: torch.Tensor | None,
    image_shape: tuple[int, int],
    args: argparse.Namespace,
    count_calibration: dict | None,
    decode_kwargs: dict,
    final_queries: set[int],
    pred_count_boundary_logits: torch.Tensor | None = None,
) -> str:
    if rank5 is None:
        return "no_5th_candidate"
    query = int(rank5["query"])
    if query in final_queries:
        return "kept"
    below_conf = float(rank5["exist_score"]) < float(args.candidate_conf)
    below_min_points = int(rank5["valid_points"]) < rank5_required_min_points(args, decode_kwargs)
    if below_conf and below_min_points:
        return "conf_and_min_points"
    if below_conf:
        return "conf"
    if below_min_points:
        return "min_points"

    legacy_decode_kwargs = no_count_head_decode_kwargs(args, decode_kwargs)
    after_min = decoded_query_set(
        pred_points,
        pred_logits,
        pred_valid_logits,
        None,
        pred_quality_logits,
        image_shape,
        conf=args.conf,
        point_valid_thr=args.point_valid_thr,
        min_points=args.min_points,
        nms_dist_px=0.0,
        max_det=None,
        count_calibration=None,
        decode_kwargs=legacy_decode_kwargs,
    )
    if query not in after_min:
        return "pre_nms_filter"

    after_nms = decoded_query_set(
        pred_points,
        pred_logits,
        pred_valid_logits,
        None,
        pred_quality_logits,
        image_shape,
        conf=args.conf,
        point_valid_thr=args.point_valid_thr,
        min_points=args.min_points,
        nms_dist_px=args.nms_dist_px,
        max_det=None,
        count_calibration=None,
        decode_kwargs=legacy_decode_kwargs,
    )
    if query not in after_nms:
        return "nms"

    if args.max_det and args.max_det > 0:
        after_max_det = decoded_query_set(
            pred_points,
            pred_logits,
            pred_valid_logits,
            None,
            pred_quality_logits,
            image_shape,
            conf=args.conf,
            point_valid_thr=args.point_valid_thr,
            min_points=args.min_points,
            nms_dist_px=args.nms_dist_px,
            max_det=args.max_det,
            count_calibration=None,
            decode_kwargs=legacy_decode_kwargs,
        )
        if query not in after_max_det:
            return "max_det"

    after_count_head = decoded_query_set(
        pred_points,
        pred_logits,
        pred_valid_logits,
        pred_count_logits,
        pred_quality_logits,
        image_shape,
        conf=args.conf,
        point_valid_thr=args.point_valid_thr,
        min_points=args.min_points,
        nms_dist_px=args.nms_dist_px,
        max_det=args.max_det,
        count_calibration=count_calibration,
        decode_kwargs=decode_kwargs,
        pred_count_boundary_logits=pred_count_boundary_logits,
    )
    if query not in after_count_head:
        return "count_head_topk"
    return "postprocess"


def classify_rank5(rank5: dict | None, deletion: str, args: argparse.Namespace, required_min_points: int) -> str:
    if rank5 is None:
        return "no_5th_candidate"
    if deletion == "kept":
        return "kept"
    if int(rank5["valid_points"]) < int(required_min_points):
        return "point_valid"
    if float(rank5["exist_score"]) < float(args.exist_low_thr) or float(rank5["rank_score"]) < float(args.s5_low_thr):
        return "exist_count"
    return "postprocess"


def rank5_candidate_reason(rank5: dict | None, deletion: str, args: argparse.Namespace, required_min_points: int) -> str:
    """Classify why the rank-5 candidate fails before considering Count Head K."""
    if rank5 is None:
        return "no_5th_candidate"
    if deletion == "kept":
        return "kept"
    below_score = float(rank5["exist_score"]) < float(args.candidate_conf)
    below_valid = int(rank5["valid_points"]) < int(required_min_points)
    if below_score and below_valid:
        return "score_and_valid_low"
    if below_score:
        return "score_low"
    if below_valid:
        return "valid_points_low"
    if deletion == "nms":
        return "nms"
    if deletion == "max_det":
        return "max_det"
    return "postprocess_other"


def rank5_primary_drop_reason(
    rank5: dict | None,
    deletion: str,
    args: argparse.Namespace,
    required_min_points: int,
    count_head_meta: dict | None,
) -> str:
    """Classify the primary final blocker for the rank-5 candidate on GT=5 images."""
    if rank5 is None:
        return "no_5th_candidate"
    if deletion == "kept":
        return "kept"
    if count_head_meta is not None and int(count_head_meta["count_head_policy_count"]) < 5:
        return "count_head_wrong"
    return rank5_candidate_reason(rank5, deletion, args, required_min_points)


def gt5_output_drop_reason(
    final_pred_lanes: int,
    rank5: dict | None,
    deletion: str,
    args: argparse.Namespace,
    required_min_points: int,
    count_head_meta: dict | None,
) -> str:
    """Classify why a GT=5 image did not decode 5 final lanes."""
    if int(final_pred_lanes) >= 5:
        return "kept"
    if (
        count_head_meta is not None
        and int(count_head_meta["count_head_policy_count"]) < 5
        and not bool(count_head_meta.get("quality_count5_upgrade_eligible", False))
    ):
        return "count_head_under_predict"
    if count_head_meta is not None and int(count_head_meta.get("candidate_pool_shortfall", 0) or 0) > 0:
        return "candidate_pool_shortfall"
    if rank5 is not None and int(rank5["valid_points"]) < int(required_min_points):
        return "valid_points_fail"
    top5_quality = None if count_head_meta is None else count_head_meta.get("top5_candidate_quality_before_nms")
    quality_thr = float(getattr(args, "quality_rescue_quality_thr", 0.55))
    if top5_quality is not None and float(top5_quality) < quality_thr:
        return "quality_too_low"
    if count_head_meta is not None and bool(count_head_meta.get("top5_suppressed_by_nms", False)):
        return "nms_suppressed"
    if rank5 is not None and (
        float(rank5["exist_score"]) < float(args.exist_low_thr) or float(rank5["rank_score"]) < float(args.s5_low_thr)
    ):
        return "rank_score_low"
    reason = rank5_candidate_reason(rank5, deletion, args, required_min_points)
    return "unknown_shortfall" if reason in {"kept", "postprocess_other"} else reason


def _lane_tensors_from_candidate(candidate: dict) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.from_numpy(np.asarray(candidate["points_norm"], dtype=np.float32))
    valid = torch.from_numpy(np.asarray(candidate["point_valid"], dtype=bool))
    return points, valid


def _lane_tensors_from_decoded(lane: dict) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.from_numpy(np.asarray(lane["points_norm"], dtype=np.float32))
    valid = lane.get("point_valid")
    if valid is None:
        valid_np = np.ones((points.shape[0],), dtype=bool)
    else:
        valid_np = np.asarray(valid, dtype=np.float32).reshape(-1) > 0.5
        if valid_np.shape[0] != points.shape[0]:
            valid_np = np.ones((points.shape[0],), dtype=bool)
    return points, torch.from_numpy(valid_np)


def duplicate_with_decoded_lanes(
    candidate: dict,
    final_lanes: list[dict],
    image_shape: tuple[int, int],
    dist_thr_px: float,
    min_overlap: int,
) -> bool:
    if float(dist_thr_px) <= 0.0:
        return False
    cand_points, cand_valid = _lane_tensors_from_candidate(candidate)
    for lane in final_lanes:
        lane_points, lane_valid = _lane_tensors_from_decoded(lane)
        dist = lane_x_distance_px(
            cand_points,
            lane_points,
            image_shape=image_shape,
            min_overlap=int(min_overlap),
            valid_a=cand_valid,
            valid_b=lane_valid,
        )
        if bool(torch.isfinite(dist)) and float(dist) <= float(dist_thr_px):
            return True
    return False


def candidate_nms_suppressed(
    candidate: dict,
    ranked_candidates: list[dict],
    image_shape: tuple[int, int],
    dist_thr_px: float,
    min_overlap: int,
    min_points: int,
    score_thr: float,
) -> bool:
    """Return whether Lane-NMS would suppress candidate behind a higher-ranked candidate."""
    if float(dist_thr_px) <= 0.0:
        return False
    query = int(candidate["query"])
    cand_points, cand_valid = _lane_tensors_from_candidate(candidate)
    for other in ranked_candidates:
        if int(other["query"]) == query:
            return False
        if float(other["exist_score"]) < float(score_thr) or int(other["valid_points"]) < int(min_points):
            continue
        other_points, other_valid = _lane_tensors_from_candidate(other)
        dist = lane_x_distance_px(
            cand_points,
            other_points,
            image_shape=image_shape,
            min_overlap=int(min_overlap),
            valid_a=cand_valid,
            valid_b=other_valid,
        )
        if bool(torch.isfinite(dist)) and float(dist) <= float(dist_thr_px):
            return True
    return False


def rescue_sweep_rows_for_image(
    raw_file: str,
    base_rank5: dict | None,
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None,
    image_shape: tuple[int, int],
    args: argparse.Namespace,
    count_meta: dict | None,
    final_lanes: list[dict],
) -> list[dict]:
    if count_meta is None or int(count_meta["count_head_policy_count"]) >= 5 or len(final_lanes) >= 5:
        return []
    base_query = None if base_rank5 is None else int(base_rank5["query"])
    rows: list[dict] = []
    for rescue_thr in args.fifth_rescue_point_valid_thrs:
        for close_gap in args.fifth_rescue_close_gaps:
            rescue_thr_f = float(rescue_thr)
            close_gap_i = int(close_gap)
            reason = "no_reliable_rank5"
            rank5 = None
            valid_ok = False
            score_ok = False
            nms_suppressed = False
            duplicate_final = False
            if base_query is not None:
                ranked = rank_query_candidates(
                    pred_points=pred_points,
                    pred_logits=pred_logits,
                    pred_valid_logits=pred_valid_logits,
                    image_shape=image_shape,
                    point_valid_thr=rescue_thr_f,
                    min_points=int(args.candidate_min_points),
                    rank_len_norm=args.rank_len_norm,
                    close_gap=close_gap_i,
                )
                by_query = {int(x["query"]): x for x in ranked}
                rank5 = by_query.get(base_query)
                if rank5 is not None:
                    valid_ok = int(rank5["valid_points"]) >= int(args.fifth_min_points)
                    score_ok = float(rank5["exist_score"]) >= float(args.candidate_conf)
                    if not valid_ok:
                        reason = "valid_insufficient"
                    elif not score_ok:
                        reason = "no_reliable_rank5"
                    else:
                        nms_suppressed = candidate_nms_suppressed(
                            rank5,
                            ranked,
                            image_shape=image_shape,
                            dist_thr_px=float(args.nms_dist_px),
                            min_overlap=int(args.line_nms_min_overlap),
                            min_points=int(args.candidate_min_points),
                            score_thr=float(args.candidate_conf),
                        )
                        if nms_suppressed:
                            reason = "nms_suppressed"
                        else:
                            duplicate_final = duplicate_with_decoded_lanes(
                                rank5,
                                final_lanes,
                                image_shape=image_shape,
                                dist_thr_px=float(args.nms_dist_px),
                                min_overlap=int(args.line_nms_min_overlap),
                            )
                            reason = "no_reliable_rank5" if duplicate_final else "rescue_ok"
            rows.append(
                {
                    "raw_file": raw_file,
                    "fifth_rescue_point_valid_thr": round(rescue_thr_f, 6),
                    "close_gap": close_gap_i,
                    "count_head_policy_count": int(count_meta["count_head_policy_count"]),
                    "final_pred_lanes": int(len(final_lanes)),
                    "rank5_query": base_query,
                    "rank5_exists": int(rank5 is not None),
                    "rank5_exist_score": None if rank5 is None else round(float(rank5["exist_score"]), 8),
                    "rank5_rank_score": None if rank5 is None else round(float(rank5["rank_score"]), 8),
                    "rank5_valid_points": None if rank5 is None else int(rank5["valid_points"]),
                    "rank5_valid_ge5": int(valid_ok),
                    "rank5_score_ok": int(score_ok),
                    "rank5_nms_suppressed": int(nms_suppressed),
                    "rank5_duplicate_final": int(duplicate_final),
                    "rescue_ok": int(reason == "rescue_ok"),
                    "rescue_reason": reason,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def add_topk_fields(row: dict, candidates: list[dict], topk: int) -> None:
    for rank in range(1, int(topk) + 1):
        item = candidates[rank - 1] if rank - 1 < len(candidates) else None
        prefix = str(rank)
        row[f"query_{prefix}"] = None if item is None else int(item["query"])
        row[f"s{prefix}"] = None if item is None else round(float(item["rank_score"]), 8)
        row[f"exist_{prefix}"] = None if item is None else round(float(item["exist_score"]), 8)
        row[f"quality_proxy_{prefix}"] = None if item is None else round(float(item["quality_proxy"]), 8)
        row[f"valid_points_{prefix}"] = None if item is None else int(item["valid_points"])
        row[f"lane_length_{prefix}"] = None if item is None else int(item["lane_length"])
        row[f"lane_length_px_{prefix}"] = None if item is None else round(float(item["lane_length_px"]), 4)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_path = Path(args.gt_json) if args.gt_json else default_tusimple_gt_json(archive_root, split=args.split)
    gt_records_all = read_tusimple_json_lines(gt_path)
    gt5_records = [x for x in gt_records_all if len(normalize_tusimple_gt_record(x).get("lanes", [])) == 5]
    if args.max_images and args.max_images > 0:
        gt5_records = gt5_records[: int(args.max_images)]
    if not gt5_records:
        raise ValueError(f"No GT=5 records found in {gt_path}.")

    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=imgsz)
    count_calibration = count_calibration_from_args(args)
    decode_kwargs = count_head_decode_kwargs_from_args(args, dataset_name=args.dataset)
    save_dir = resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    query_rows: list[dict] = []
    rescue_rows: list[dict] = []
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"diagnosing {len(gt5_records)} GT=5 images from {gt_path}")

    for index, gt in enumerate(gt5_records, start=1):
        raw_file = str(gt["raw_file"])
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_shape = (int(img.shape[0]), int(img.shape[1]))
        tensor = preprocess_image(img, imgsz=imgsz, device=device, half=args.half)
        preds = model(tensor)

        pred_points = preds["pred_points"][0].detach().float()
        pred_logits = preds["pred_logits"][0].detach().float()
        pred_valid = preds.get("pred_valid_logits")
        pred_valid = pred_valid[0].detach().float() if pred_valid is not None else None
        pred_count = preds.get("pred_count_logits")
        pred_count = pred_count[0].detach().float() if pred_count is not None else None
        pred_count_boundary = preds.get("pred_count_boundary_logits")
        pred_count_boundary = (
            pred_count_boundary[0].detach().float() if pred_count_boundary is not None else None
        )
        pred_quality = preds.get("pred_quality_logits")
        pred_quality = pred_quality[0].detach().float() if pred_quality is not None else None

        candidates = rank_query_candidates(
            pred_points=pred_points,
            pred_logits=pred_logits,
            pred_valid_logits=pred_valid,
            image_shape=image_shape,
            point_valid_thr=args.candidate_point_valid_thr,
            min_points=int(args.candidate_min_points),
            rank_len_norm=args.rank_len_norm,
        )
        final_lanes, decode_meta = decoded_lanes(
            pred_points,
            pred_logits,
            pred_valid,
            pred_count,
            pred_quality,
            image_shape,
            conf=args.conf,
            point_valid_thr=args.point_valid_thr,
            min_points=args.min_points,
            nms_dist_px=args.nms_dist_px,
            max_det=args.max_det,
            count_calibration=count_calibration,
            decode_kwargs=decode_kwargs,
            return_meta=True,
            pred_count_boundary_logits=pred_count_boundary,
        )
        final_queries = {int(x["query"]) for x in final_lanes}
        count_meta = count_head_meta_for_image(
            pred_count,
            args,
            decode_kwargs,
            pred_count_boundary_logits=pred_count_boundary,
        )
        if count_meta is not None:
            count_meta = {**count_meta, **decode_meta}
            if count_meta.get("effective_policy_count") is not None:
                count_meta["count_head_policy_count"] = int(count_meta["effective_policy_count"])
        else:
            count_meta = decode_meta
        rank5 = candidates[4] if len(candidates) >= 5 else None
        rescue_rows.extend(
            rescue_sweep_rows_for_image(
                raw_file=raw_file,
                base_rank5=rank5,
                pred_points=pred_points,
                pred_logits=pred_logits,
                pred_valid_logits=pred_valid,
                image_shape=image_shape,
                args=args,
                count_meta=count_meta,
                final_lanes=final_lanes,
            )
        )
        required_min_points = rank5_required_min_points(args, decode_kwargs)
        deletion = deletion_stage(
            rank5,
            pred_points,
            pred_logits,
            pred_valid,
            pred_count,
            pred_quality,
            image_shape,
            args,
            count_calibration,
            decode_kwargs,
            final_queries,
            pred_count_boundary_logits=pred_count_boundary,
        )
        diagnosis = classify_rank5(rank5, deletion, args, required_min_points)
        candidate_reason = rank5_candidate_reason(rank5, deletion, args, required_min_points)
        primary_reason = rank5_primary_drop_reason(rank5, deletion, args, required_min_points, count_meta)
        output_reason = gt5_output_drop_reason(
            len(final_queries), rank5, deletion, args, required_min_points, count_meta
        )
        count_prob = None if count_meta is None else list(count_meta["count_head_prob"])
        row = {
            "raw_file": raw_file,
            "image": str(image_path.resolve()),
            "gt_lanes": 5,
            "final_pred_lanes": int(len(final_queries)),
            "final_queries": ";".join(str(x) for x in sorted(final_queries)),
            "count_head_raw_count": None if count_meta is None else int(count_meta["count_head_raw_count"]),
            "count_head_policy_count": None if count_meta is None else int(count_meta["count_head_policy_count"]),
            "count_head_top1_prob": None if count_prob is None else round(float(max(count_prob)), 8),
            "count_head_prob_2": None if count_prob is None else round(float(count_prob[0]), 8),
            "count_head_prob_3": None if count_prob is None else round(float(count_prob[1]), 8),
            "count_head_prob_4": None if count_prob is None else round(float(count_prob[2]), 8),
            "count_head_prob_5": None if count_prob is None else round(float(count_prob[3]), 8),
            "count_head_margin": None if count_meta is None else round(float(count_meta["count_head_margin"]), 8),
            "candidate_pool_shortfall": int(count_meta.get("candidate_pool_shortfall", 0) or 0),
            "candidate_pool_shortfall_before_rescue": int(
                count_meta.get("candidate_pool_shortfall_before_rescue", 0) or 0
            ),
            "candidate_pool_shortfall_after_rescue": int(
                count_meta.get("candidate_pool_shortfall_after_rescue", 0) or 0
            ),
            "top5_suppressed_by_nms": int(bool(count_meta.get("top5_suppressed_by_nms", False))),
            "candidate_count_normal": int(count_meta.get("candidate_count_normal", 0) or 0),
            "candidate_count_rescue": int(count_meta.get("candidate_count_rescue", 0) or 0),
            "candidate_count_last_lane": int(count_meta.get("candidate_count_last_lane", 0) or 0),
            "candidate_count_edge_rescue": int(count_meta.get("candidate_count_edge_rescue", 0) or 0),
            "candidate_count_after_rescue": int(count_meta.get("candidate_count_after_rescue", 0) or 0),
            "candidate_count_after_nms": int(count_meta.get("candidate_count_after_nms", 0) or 0),
            "nms_suppressed_count": int(count_meta.get("nms_suppressed_count", 0) or 0),
            "last_lane_rescue_attempt_count": int(count_meta.get("last_lane_rescue_attempt_count", 0) or 0),
            "last_lane_rescue_success_count": int(count_meta.get("last_lane_rescue_success_count", 0) or 0),
            "last_lane_rescue_reason": str(count_meta.get("last_lane_rescue_reason", "not_attempted")),
            "edge_last_lane_rescue_attempt_count": int(
                count_meta.get("edge_last_lane_rescue_attempt_count", 0) or 0
            ),
            "edge_last_lane_rescue_success_count": int(
                count_meta.get("edge_last_lane_rescue_success_count", 0) or 0
            ),
            "edge_last_lane_rescue_reason": str(
                count_meta.get("edge_last_lane_rescue_reason", "not_attempted")
            ),
            "edge_last_lane_rescue_candidate_side": count_meta.get("edge_last_lane_rescue_candidate_side"),
            "edge_last_lane_rescue_candidate_outside_gap_px": count_meta.get(
                "edge_last_lane_rescue_candidate_outside_gap_px"
            ),
            "edge_last_lane_rescue_candidate_valid_points": count_meta.get(
                "edge_last_lane_rescue_candidate_valid_points"
            ),
            "edge_last_lane_rescue_candidate_quality": count_meta.get(
                "edge_last_lane_rescue_candidate_quality"
            ),
            "edge_count4_to5_upgrade": int(bool(count_meta.get("edge_count4_to5_upgrade", False))),
            "edge_count4_to5_upgrade_reason": str(
                count_meta.get("edge_count4_to5_upgrade_reason", "not_attempted")
            ),
            "rank5_query": None if rank5 is None else int(rank5["query"]),
            "rank5_required_min_points": int(required_min_points),
            "rank5_kept": int(deletion == "kept"),
            "rank5_deletion_stage": deletion,
            "rank5_diagnosis": diagnosis,
            "rank5_candidate_reason": candidate_reason,
            "rank5_primary_drop_reason": primary_reason,
            "gt5_output_drop_reason": output_reason,
        }
        add_topk_fields(row, candidates, args.topk)
        rows.append(row)

        for item in candidates[: int(args.topk)]:
            query_rows.append(
                {
                    "raw_file": raw_file,
                    "image": str(image_path.resolve()),
                    "rank": int(item["rank"]),
                    "query": int(item["query"]),
                    "rank_score": round(float(item["rank_score"]), 8),
                    "exist_score": round(float(item["exist_score"]), 8),
                    "quality_proxy": round(float(item["quality_proxy"]), 8),
                    "valid_points": int(item["valid_points"]),
                    "lane_length": int(item["lane_length"]),
                    "lane_length_px": round(float(item["lane_length_px"]), 4),
                    "mean_valid_score": round(float(item["mean_valid_score"]), 8),
                    "length_factor": round(float(item["length_factor"]), 8),
                    "smooth_factor": round(float(item["smooth_factor"]), 8),
                    "jump_factor": round(float(item["jump_factor"]), 8),
                    "visible_start": item["visible_start"],
                    "visible_end": item["visible_end"],
                    "kept_final": int(int(item["query"]) in final_queries),
                }
            )

        if index % 100 == 0 or index == len(gt5_records):
            print(f"processed {index}/{len(gt5_records)}")

    fields = [
        "raw_file",
        "image",
        "gt_lanes",
        "final_pred_lanes",
        "final_queries",
        "count_head_raw_count",
        "count_head_policy_count",
        "count_head_top1_prob",
        "count_head_prob_2",
        "count_head_prob_3",
        "count_head_prob_4",
        "count_head_prob_5",
        "count_head_margin",
        "candidate_pool_shortfall",
        "candidate_pool_shortfall_before_rescue",
        "candidate_pool_shortfall_after_rescue",
        "top5_suppressed_by_nms",
        "candidate_count_normal",
        "candidate_count_rescue",
        "candidate_count_last_lane",
        "candidate_count_edge_rescue",
        "candidate_count_after_rescue",
        "candidate_count_after_nms",
        "nms_suppressed_count",
        "last_lane_rescue_attempt_count",
        "last_lane_rescue_success_count",
        "last_lane_rescue_reason",
        "edge_last_lane_rescue_attempt_count",
        "edge_last_lane_rescue_success_count",
        "edge_last_lane_rescue_reason",
        "edge_last_lane_rescue_candidate_side",
        "edge_last_lane_rescue_candidate_outside_gap_px",
        "edge_last_lane_rescue_candidate_valid_points",
        "edge_last_lane_rescue_candidate_quality",
        "edge_count4_to5_upgrade",
        "edge_count4_to5_upgrade_reason",
        "rank5_query",
        "rank5_required_min_points",
        "rank5_kept",
        "rank5_deletion_stage",
        "rank5_diagnosis",
        "rank5_candidate_reason",
        "rank5_primary_drop_reason",
        "gt5_output_drop_reason",
    ]
    for rank in range(1, int(args.topk) + 1):
        fields.extend(
            [
                f"query_{rank}",
                f"s{rank}",
                f"exist_{rank}",
                f"quality_proxy_{rank}",
                f"valid_points_{rank}",
                f"lane_length_{rank}",
                f"lane_length_px_{rank}",
            ]
        )
    write_csv(save_dir / "gt5_rank_diagnostics.csv", rows, fields)
    write_csv(
        save_dir / "gt5_topk_queries.csv",
        query_rows,
        [
            "raw_file",
            "image",
            "rank",
            "query",
            "rank_score",
            "exist_score",
            "quality_proxy",
            "valid_points",
            "lane_length",
            "lane_length_px",
            "mean_valid_score",
            "length_factor",
            "smooth_factor",
            "jump_factor",
            "visible_start",
            "visible_end",
            "kept_final",
        ],
    )
    rescue_fields = [
        "raw_file",
        "fifth_rescue_point_valid_thr",
        "close_gap",
        "count_head_policy_count",
        "final_pred_lanes",
        "rank5_query",
        "rank5_exists",
        "rank5_exist_score",
        "rank5_rank_score",
        "rank5_valid_points",
        "rank5_valid_ge5",
        "rank5_score_ok",
        "rank5_nms_suppressed",
        "rank5_duplicate_final",
        "rescue_ok",
        "rescue_reason",
    ]
    write_csv(save_dir / "gt5_count_head_klt5_rescue_sweep.csv", rescue_rows, rescue_fields)

    diagnosis_counts = Counter(str(x["rank5_diagnosis"]) for x in rows)
    deletion_counts = Counter(str(x["rank5_deletion_stage"]) for x in rows)
    candidate_reason_counts = Counter(str(x["rank5_candidate_reason"]) for x in rows)
    primary_reason_counts = Counter(str(x["rank5_primary_drop_reason"]) for x in rows)
    output_reason_counts = Counter(str(x["gt5_output_drop_reason"]) for x in rows)
    last_lane_rescue_reason_counts = Counter(
        str(x["last_lane_rescue_reason"])
        for x in rows
        if int(x.get("last_lane_rescue_attempt_count", 0) or 0) > 0
    )
    edge_rescue_reason_counts = Counter(
        str(x["edge_last_lane_rescue_reason"])
        for x in rows
        if int(x.get("edge_last_lane_rescue_attempt_count", 0) or 0) > 0
    )
    edge_upgrade_reason_counts = Counter(str(x["edge_count4_to5_upgrade_reason"]) for x in rows)
    count_head_counts = Counter(
        "disabled" if x.get("count_head_policy_count") in (None, "") else str(int(x["count_head_policy_count"]))
        for x in rows
    )
    total_rows = max(len(rows), 1)
    gt5_count_head_under_rate = sum(
        int((x.get("count_head_policy_count") not in (None, "")) and int(x["count_head_policy_count"]) < 5)
        for x in rows
    ) / total_rows
    gt5_candidate_pool_shortfall_rate = sum(
        int(int(x.get("candidate_pool_shortfall", 0) or 0) > 0) for x in rows
    ) / total_rows
    gt5_valid_points_fail_rate = sum(
        int(str(x.get("gt5_output_drop_reason")) == "valid_points_fail") for x in rows
    ) / total_rows
    gt5_top5_suppressed_by_nms_rate = sum(
        int(bool(x.get("top5_suppressed_by_nms", 0))) for x in rows
    ) / total_rows
    gt5_rank5_score_low_rate = sum(
        int(str(x.get("gt5_output_drop_reason")) == "rank_score_low") for x in rows
    ) / total_rows
    gt5_unknown_shortfall_rate = sum(
        int(str(x.get("gt5_output_drop_reason")) == "unknown_shortfall") for x in rows
    ) / total_rows
    s5_values = [float(x["s5"]) for x in rows if x.get("s5") is not None]
    v5_values = [int(x["valid_points_5"]) for x in rows if x.get("valid_points_5") is not None]
    required_values = [int(x["rank5_required_min_points"]) for x in rows if x.get("rank5_required_min_points") is not None]
    rescue_summary: dict[str, dict] = {}
    for rescue_row in rescue_rows:
        key = f"thr{float(rescue_row['fifth_rescue_point_valid_thr']):.2f}_gap{int(rescue_row['close_gap'])}"
        item = rescue_summary.setdefault(
            key,
            {
                "fifth_rescue_point_valid_thr": float(rescue_row["fifth_rescue_point_valid_thr"]),
                "close_gap": int(rescue_row["close_gap"]),
                "klt5_images": 0,
                "rank5_exists": 0,
                "rank5_valid_ge5": 0,
                "rank5_nms_suppressed": 0,
                "rank5_duplicate_final": 0,
                "rescue_ok": 0,
                "rescue_reason_counts": {},
            },
        )
        item["klt5_images"] += 1
        item["rank5_exists"] += int(rescue_row["rank5_exists"])
        item["rank5_valid_ge5"] += int(rescue_row["rank5_valid_ge5"])
        item["rank5_nms_suppressed"] += int(rescue_row["rank5_nms_suppressed"])
        item["rank5_duplicate_final"] += int(rescue_row["rank5_duplicate_final"])
        item["rescue_ok"] += int(rescue_row["rescue_ok"])
        reason = str(rescue_row["rescue_reason"])
        item["rescue_reason_counts"][reason] = int(item["rescue_reason_counts"].get(reason, 0)) + 1
    rescue_summary = {k: rescue_summary[k] for k in sorted(rescue_summary)}
    summary = {
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "archive_root": str(archive_root.resolve()),
            "gt_json": str(gt_path.resolve()),
            "split": args.split,
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "candidate_conf": float(args.candidate_conf),
            "candidate_point_valid_thr": float(args.candidate_point_valid_thr),
            "min_points": int(args.min_points),
            "nms_dist_px": float(args.nms_dist_px),
            "line_nms_min_overlap": int(args.line_nms_min_overlap),
            "max_det": int(args.max_det),
            "count_head_decode": decode_kwargs,
            "s5_low_thr": float(args.s5_low_thr),
            "exist_low_thr": float(args.exist_low_thr),
            "rank_len_norm": float(args.rank_len_norm),
            "fifth_rescue_point_valid_thrs": [float(x) for x in args.fifth_rescue_point_valid_thrs],
            "fifth_rescue_close_gaps": [int(x) for x in args.fifth_rescue_close_gaps],
        },
        "images": int(len(rows)),
        "count_head_policy_count_counts": {k: int(v) for k, v in sorted(count_head_counts.items())},
        "rank5_diagnosis_counts": {k: int(v) for k, v in sorted(diagnosis_counts.items())},
        "rank5_deletion_stage_counts": {k: int(v) for k, v in sorted(deletion_counts.items())},
        "rank5_candidate_reason_counts": {k: int(v) for k, v in sorted(candidate_reason_counts.items())},
        "rank5_primary_drop_reason_counts": {k: int(v) for k, v in sorted(primary_reason_counts.items())},
        "gt5_output_drop_reason_counts": {k: int(v) for k, v in sorted(output_reason_counts.items())},
        "last_lane_rescue_attempt_count": int(sum(x["last_lane_rescue_attempt_count"] for x in rows)),
        "last_lane_rescue_success_count": int(sum(x["last_lane_rescue_success_count"] for x in rows)),
        "last_lane_rescue_reason_counts": {
            k: int(v) for k, v in sorted(last_lane_rescue_reason_counts.items())
        },
        "edge_last_lane_rescue_attempt_count": int(
            sum(x["edge_last_lane_rescue_attempt_count"] for x in rows)
        ),
        "edge_last_lane_rescue_success_count": int(
            sum(x["edge_last_lane_rescue_success_count"] for x in rows)
        ),
        "edge_last_lane_rescue_reason_counts": {
            k: int(v) for k, v in sorted(edge_rescue_reason_counts.items())
        },
        "edge_count4_to5_upgrade_count": int(sum(x["edge_count4_to5_upgrade"] for x in rows)),
        "edge_count4_to5_upgrade_reason_counts": {
            k: int(v) for k, v in sorted(edge_upgrade_reason_counts.items())
        },
        "gt5_count_head_under_rate": round(float(gt5_count_head_under_rate), 6),
        "gt5_candidate_pool_shortfall_rate": round(float(gt5_candidate_pool_shortfall_rate), 6),
        "gt5_valid_points_fail_rate": round(float(gt5_valid_points_fail_rate), 6),
        "gt5_top5_suppressed_by_nms_rate": round(float(gt5_top5_suppressed_by_nms_rate), 6),
        "gt5_rank5_score_low_rate": round(float(gt5_rank5_score_low_rate), 6),
        "gt5_unknown_shortfall_rate": round(float(gt5_unknown_shortfall_rate), 6),
        "s5": {
            "mean": None if not s5_values else round(float(np.mean(s5_values)), 6),
            "median": None if not s5_values else round(float(np.median(s5_values)), 6),
            "p10": None if not s5_values else round(float(np.percentile(s5_values, 10)), 6),
            "p90": None if not s5_values else round(float(np.percentile(s5_values, 90)), 6),
        },
        "valid_points_5": {
            "mean": None if not v5_values else round(float(np.mean(v5_values)), 6),
            "median": None if not v5_values else round(float(np.median(v5_values)), 6),
            "lt_required_min_points": int(
                sum(v < required for v, required in zip(v5_values, required_values, strict=False))
            ),
        },
        "fifth_rescue_sweep_counts": rescue_summary,
        "outputs": {
            "per_image_csv": str((save_dir / "gt5_rank_diagnostics.csv").resolve()),
            "topk_query_csv": str((save_dir / "gt5_topk_queries.csv").resolve()),
            "rescue_sweep_csv": str((save_dir / "gt5_count_head_klt5_rescue_sweep.csv").resolve()),
        },
    }
    (save_dir / "gt5_rank_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
