from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.eval_gcs import label_path_for_image, load_gcs_label  # noqa: E402
from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image  # noqa: E402
from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.utils.gcs_matcher import GCSHungarianMatcher  # noqa: E402
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether GT-close but Hungarian-unmatched query lanes are "
            "being supervised as all-zero point-valid negatives."
        )
    )
    parser.add_argument("--dataset", default="culane", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--weights", required=True, help="GCS checkpoint .pt.")
    parser.add_argument("--source", required=True, help="Image directory, image file, or txt list.")
    parser.add_argument("--labels", default=None, help="labels_gcs directory. Empty means infer from image path.")
    parser.add_argument(
        "--imgsz",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=None,
        help="GCS inference/loss shape as H W. Defaults to the dataset profile.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    parser.add_argument(
        "--x-thrs",
        nargs="+",
        type=float,
        default=[20.0, 30.0, 40.0, 60.0],
        help="Mean x-error thresholds in training-image pixels for GT-close unmatched queries.",
    )
    parser.add_argument(
        "--anchor-x-thrs",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Per-anchor x-error thresholds in pixels. If omitted, each value in --x-thrs is reused "
            "for the same-threshold anchor mask."
        ),
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=3,
        help="Minimum GT-visible fixed-y anchors required before a query/GT pair can be GT-close.",
    )
    parser.add_argument("--cost-point", type=float, default=5.0)
    parser.add_argument("--cost-curve", type=float, default=0.05)
    parser.add_argument("--cost-exist", type=float, default=0.1)
    parser.add_argument("--match-min-overlap", type=int, default=2)
    parser.add_argument("--match-max-x-dist", type=float, default=0.0)
    parser.add_argument("--match-gate-px", type=float, default=160.0)
    parser.add_argument("--valid-prob-thrs", nargs="+", type=float, default=[0.3, 0.5])
    parser.add_argument("--examples-per-thr", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--save-dir", required=True)
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _new_accumulator() -> dict[str, float]:
    return {
        "images": 0.0,
        "gt_lanes": 0.0,
        "gt_visible_anchors": 0.0,
        "queries": 0.0,
        "matched_queries": 0.0,
        "unmatched_queries": 0.0,
        "point_valid_bce_sum": 0.0,
        "point_valid_anchor_count": 0.0,
        "unmatched_neg_bce_sum": 0.0,
        "unmatched_anchor_count": 0.0,
        "close_unmatched_queries": 0.0,
        "close_unmatched_pairs": 0.0,
        "close_unmatched_anchors": 0.0,
        "close_unmatched_bce_sum": 0.0,
        "close_valid_prob_sum": 0.0,
        "close_valid_prob_count": 0.0,
        "far_unmatched_queries": 0.0,
        "far_unmatched_anchors": 0.0,
        "far_unmatched_bce_sum": 0.0,
        "far_valid_prob_sum": 0.0,
        "far_valid_prob_count": 0.0,
    }


def _add(dst: dict[str, float], key: str, value: float | int) -> None:
    dst[key] = float(dst.get(key, 0.0)) + float(value)


def _add_dict(dst: dict[str, float], src: dict[str, float]) -> None:
    for key, value in src.items():
        _add(dst, key, value)


def _format_threshold(value: float) -> str:
    return f"{float(value):g}"


def _finalize_accumulator(stats: dict[str, float], valid_prob_thrs: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in stats.items():
        if key.endswith("_count") or key.endswith("_queries") or key.endswith("_anchors") or key in {
            "images",
            "gt_lanes",
            "queries",
            "matched_queries",
            "unmatched_queries",
            "close_unmatched_pairs",
            "point_valid_anchor_count",
            "unmatched_anchor_count",
            "gt_visible_anchors",
            "far_unmatched_queries",
        }:
            out[key] = int(round(value))
        else:
            out[key] = float(value)

    unmatched_queries = float(stats.get("unmatched_queries", 0.0))
    unmatched_anchors = float(stats.get("unmatched_anchor_count", 0.0))
    point_valid_bce = float(stats.get("point_valid_bce_sum", 0.0))
    unmatched_bce = float(stats.get("unmatched_neg_bce_sum", 0.0))
    close_queries = float(stats.get("close_unmatched_queries", 0.0))
    close_anchors = float(stats.get("close_unmatched_anchors", 0.0))
    close_bce = float(stats.get("close_unmatched_bce_sum", 0.0))
    far_bce = float(stats.get("far_unmatched_bce_sum", 0.0))
    close_prob_count = float(stats.get("close_valid_prob_count", 0.0))
    far_prob_count = float(stats.get("far_valid_prob_count", 0.0))

    out["close_query_ratio_of_unmatched"] = close_queries / unmatched_queries if unmatched_queries else 0.0
    out["close_anchor_ratio_of_unmatched"] = close_anchors / unmatched_anchors if unmatched_anchors else 0.0
    out["close_bce_share_of_unmatched_neg"] = close_bce / unmatched_bce if unmatched_bce else 0.0
    out["close_bce_share_of_total_point_valid"] = close_bce / point_valid_bce if point_valid_bce else 0.0
    out["far_bce_share_of_unmatched_neg"] = far_bce / unmatched_bce if unmatched_bce else 0.0
    out["close_valid_prob_mean"] = (
        float(stats.get("close_valid_prob_sum", 0.0)) / close_prob_count if close_prob_count else 0.0
    )
    out["far_valid_prob_mean"] = (
        float(stats.get("far_valid_prob_sum", 0.0)) / far_prob_count if far_prob_count else 0.0
    )
    out["point_valid_bce_mean"] = point_valid_bce / float(stats.get("point_valid_anchor_count", 0.0)) if stats.get("point_valid_anchor_count", 0.0) else 0.0
    out["unmatched_neg_bce_mean"] = unmatched_bce / unmatched_anchors if unmatched_anchors else 0.0
    out["close_unmatched_bce_mean"] = close_bce / close_anchors if close_anchors else 0.0
    out["far_unmatched_bce_mean"] = far_bce / float(stats.get("far_unmatched_anchors", 0.0)) if stats.get("far_unmatched_anchors", 0.0) else 0.0

    for thr in valid_prob_thrs:
        key = f"close_valid_prob_ge_{_format_threshold(thr)}"
        count = float(stats.get(key, 0.0))
        out[key] = int(round(count))
        out[f"{key}_ratio"] = count / close_prob_count if close_prob_count else 0.0

    return out


def _batch_tensors(
    image_paths: list[Path],
    label_dir: str | Path | None,
    imgsz: tuple[int, int],
    device: torch.device,
    half: bool,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]:
    images: list[torch.Tensor] = []
    gt_points: list[torch.Tensor] = []
    gt_valid: list[torch.Tensor] = []
    meta: list[dict[str, Any]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        label_path = label_path_for_image(image_path, label_dir)
        lanes_np, valid_np = load_gcs_label(label_path)
        images.append(preprocess_image(image, imgsz=imgsz, device=device, half=half))
        gt_points.append(torch.from_numpy(lanes_np).to(device=device))
        gt_valid.append(torch.from_numpy(valid_np > 0.5).to(device=device))
        meta.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "gt_count": int(lanes_np.shape[0]),
                "original_shape": [int(image.shape[0]), int(image.shape[1])],
            }
        )

    return torch.cat(images, dim=0), gt_points, gt_valid, meta


def _point_valid_target(
    pred_valid_logits: torch.Tensor,
    gt_valid: list[torch.Tensor],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    target = torch.zeros_like(pred_valid_logits, dtype=torch.float32)
    device = pred_valid_logits.device
    for b, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() == 0:
            continue
        src_idx = src_idx.to(device=device, dtype=torch.long)
        tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
        target[b, src_idx] = gt_valid[b].to(device=device, dtype=torch.float32)[tgt_idx]
    return target


def _patch_legacy_gcs_head_attrs(model: torch.nn.Module) -> int:
    """Patch missing metadata on older pickled GCS heads without enabling new branches."""
    patched = 0
    for module in model.modules():
        if not isinstance(module, GCSLaneHead):
            continue
        if not hasattr(module, "gcs_mode"):
            if all(hasattr(module, name) for name in ("count_mlp", "start_mlp", "end_mlp")):
                module.gcs_mode = "ordered_slot"
            elif all(hasattr(module, name) for name in ("count_head", "start_head", "end_head")):
                module.gcs_mode = "ordered_slot"
            else:
                module.gcs_mode = "query"
            patched += 1
        inferred = {
            "geometry_aware_exist": hasattr(module, "geometry_exist_delta_mlp"),
            "two_stage_refine": hasattr(module, "point_refine_mlp_stage2"),
            "gated_multiscale": hasattr(module, "multiscale_gate"),
            "proposal_state_refine": hasattr(module, "proposal_state_update_mlp"),
        }
        for name, value in inferred.items():
            if not hasattr(module, name):
                setattr(module, name, bool(value))
                patched += 1
        if not hasattr(module, "return_aux"):
            module.return_aux = False
            patched += 1
    return patched


@torch.no_grad()
def _safe_match(
    matcher: GCSHungarianMatcher,
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_points: list[torch.Tensor],
    gt_valid: list[torch.Tensor],
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
    """Run the training matcher while dropping infeasible rows/columns for diagnosis continuity."""
    if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
        pred_logits = pred_logits.squeeze(-1)
    if pred_points.ndim != 4 or pred_points.shape[-1] != 2:
        raise ValueError(f"pred_points must have shape B x Q x K x 2, got {tuple(pred_points.shape)}.")
    if pred_logits.ndim != 2:
        raise ValueError(f"pred_logits must have shape B x Q, got {tuple(pred_logits.shape)}.")
    if pred_points.shape[:2] != pred_logits.shape:
        raise ValueError(
            f"pred_points B,Q must match pred_logits, got {tuple(pred_points.shape[:2])} vs {tuple(pred_logits.shape)}."
        )

    device = pred_points.device
    dtype = pred_points.dtype
    indices: list[tuple[torch.Tensor, torch.Tensor]] = []
    dropped_infeasible_gt = 0
    empty = torch.empty(0, dtype=torch.long, device=device)

    for b in range(pred_points.shape[0]):
        pp = pred_points[b]
        pl = pred_logits[b]
        gp = gt_points[b].to(device=device, dtype=dtype)
        gv = gt_valid[b].to(device=device, dtype=dtype)
        if gp.numel() == 0:
            indices.append((empty, empty))
            continue
        if gp.ndim != 3 or gp.shape[-1] != 2:
            raise ValueError(f"Each GT lane tensor must have shape N x K x 2, got {tuple(gp.shape)}.")
        if gv.shape != gp.shape[:2]:
            raise ValueError(f"GT valid mask must match GT lane first two dims, got {tuple(gv.shape)} vs {tuple(gp.shape[:2])}.")

        valid_lane = gv.sum(dim=1) >= 2
        if not bool(valid_lane.any()):
            indices.append((empty, empty))
            continue
        original_cols = torch.arange(gp.shape[0], device=device)[valid_lane]
        gp = gp[valid_lane]
        gv = gv[valid_lane]
        cost = matcher.cost_matrix(pp, pl, gp, gv)
        finite = torch.isfinite(cost)
        row_keep = finite.any(dim=1)
        col_keep = finite.any(dim=0)
        dropped_infeasible_gt += int((~col_keep).sum().item())
        if not bool(row_keep.any()) or not bool(col_keep.any()):
            indices.append((empty, empty))
            continue
        kept_rows = torch.arange(cost.shape[0], device=device)[row_keep]
        kept_cols = torch.arange(cost.shape[1], device=device)[col_keep]
        cost_sub = cost[row_keep][:, col_keep]
        finite_sub = finite[row_keep][:, col_keep]
        large_cost = torch.nan_to_num(cost_sub, posinf=1e9, neginf=1e9).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(large_cost)
        rows_all = torch.as_tensor(row_ind, dtype=torch.long, device=device)
        cols_all = torch.as_tensor(col_ind, dtype=torch.long, device=device)
        keep = finite_sub[rows_all, cols_all]
        rows = kept_rows[rows_all[keep]]
        cols = original_cols[kept_cols[cols_all[keep]]]
        indices.append((rows, cols))

    return indices, dropped_infeasible_gt


def _analyze_image(
    pred_points: torch.Tensor,
    pred_valid_logits: torch.Tensor,
    gt_points: torch.Tensor,
    gt_valid: torch.Tensor,
    indices: tuple[torch.Tensor, torch.Tensor],
    image_size: tuple[int, int],
    x_thr: float,
    anchor_x_thr: float,
    min_overlap: int,
    valid_prob_thrs: list[float],
) -> tuple[dict[str, float], dict[int, dict[str, float]], list[dict[str, Any]]]:
    q, k = pred_valid_logits.shape
    device = pred_valid_logits.device
    matched = torch.zeros((q,), dtype=torch.bool, device=device)
    src_idx, _ = indices
    if src_idx.numel():
        matched[src_idx.to(device=device, dtype=torch.long)] = True
    unmatched = ~matched
    valid_prob = pred_valid_logits.float().sigmoid()
    bce_zero = F.softplus(pred_valid_logits.float())

    stats = _new_accumulator()
    stats["images"] = 1.0
    stats["gt_lanes"] = float(gt_points.shape[0])
    stats["gt_visible_anchors"] = float(gt_valid.sum().item())
    stats["queries"] = float(q)
    stats["matched_queries"] = float(matched.sum().item())
    stats["unmatched_queries"] = float(unmatched.sum().item())
    stats["unmatched_anchor_count"] = float(unmatched.sum().item() * k)
    stats["unmatched_neg_bce_sum"] = float(bce_zero[unmatched].sum().item()) if bool(unmatched.any()) else 0.0

    per_lane: dict[int, dict[str, float]] = {}
    examples: list[dict[str, Any]] = []

    if gt_points.numel() == 0 or not bool(unmatched.any()):
        stats["far_unmatched_queries"] = float(unmatched.sum().item())
        stats["far_unmatched_anchors"] = float(unmatched.sum().item() * k)
        stats["far_unmatched_bce_sum"] = stats["unmatched_neg_bce_sum"]
        if bool(unmatched.any()):
            far_prob = valid_prob[unmatched]
            stats["far_valid_prob_sum"] = float(far_prob.sum().item())
            stats["far_valid_prob_count"] = float(far_prob.numel())
        return stats, per_lane, examples

    h, w = image_size
    del h
    gt_valid_bool = gt_valid.to(dtype=torch.bool)
    diff_x_px = (pred_points[:, None, :, 0].float() - gt_points[None, :, :, 0].float()).abs() * float(w)
    valid_pair = gt_valid_bool[None].expand(q, -1, -1)
    overlap = valid_pair.sum(dim=2)
    mean_x = (diff_x_px * valid_pair.float()).sum(dim=2) / overlap.clamp_min(1).float()
    close_pair = (
        unmatched[:, None]
        & (overlap >= int(min_overlap))
        & (mean_x <= float(x_thr))
    )
    close_query = close_pair.any(dim=1)
    close_anchor = (
        close_pair[:, :, None]
        & valid_pair
        & (diff_x_px <= float(anchor_x_thr))
    ).any(dim=1)
    unmatched_anchor = unmatched[:, None].expand(q, k)
    far_anchor = unmatched_anchor & ~close_anchor

    stats["close_unmatched_queries"] = float(close_query.sum().item())
    stats["close_unmatched_pairs"] = float(close_pair.sum().item())
    stats["close_unmatched_anchors"] = float(close_anchor.sum().item())
    stats["close_unmatched_bce_sum"] = float(bce_zero[close_anchor].sum().item()) if bool(close_anchor.any()) else 0.0
    stats["far_unmatched_queries"] = float((unmatched & ~close_query).sum().item())
    stats["far_unmatched_anchors"] = float(far_anchor.sum().item())
    stats["far_unmatched_bce_sum"] = float(bce_zero[far_anchor].sum().item()) if bool(far_anchor.any()) else 0.0

    if bool(close_anchor.any()):
        close_prob = valid_prob[close_anchor]
        stats["close_valid_prob_sum"] = float(close_prob.sum().item())
        stats["close_valid_prob_count"] = float(close_prob.numel())
        for prob_thr in valid_prob_thrs:
            stats[f"close_valid_prob_ge_{_format_threshold(prob_thr)}"] = float((close_prob >= float(prob_thr)).sum().item())
    if bool(far_anchor.any()):
        far_prob = valid_prob[far_anchor]
        stats["far_valid_prob_sum"] = float(far_prob.sum().item())
        stats["far_valid_prob_count"] = float(far_prob.numel())

    pair_rows = torch.nonzero(close_pair, as_tuple=False)
    for row in pair_rows.tolist():
        query_index, lane_index = int(row[0]), int(row[1])
        lane_stats = per_lane.setdefault(lane_index, _new_accumulator())
        lane_anchor = close_pair[query_index, lane_index] & gt_valid_bool[lane_index] & (diff_x_px[query_index, lane_index] <= float(anchor_x_thr))
        _add(lane_stats, "gt_lanes", 1)
        _add(lane_stats, "close_unmatched_pairs", 1)
        _add(lane_stats, "close_unmatched_queries", 1)
        _add(lane_stats, "close_unmatched_anchors", int(lane_anchor.sum().item()))
        _add(lane_stats, "close_unmatched_bce_sum", float(bce_zero[query_index][lane_anchor].sum().item()))
        if bool(lane_anchor.any()):
            lane_prob = valid_prob[query_index][lane_anchor]
            _add(lane_stats, "close_valid_prob_sum", float(lane_prob.sum().item()))
            _add(lane_stats, "close_valid_prob_count", int(lane_prob.numel()))
            for prob_thr in valid_prob_thrs:
                _add(
                    lane_stats,
                    f"close_valid_prob_ge_{_format_threshold(prob_thr)}",
                    int((lane_prob >= float(prob_thr)).sum().item()),
                )
        examples.append(
            {
                "query": query_index,
                "gt_index": lane_index,
                "mean_abs_x_error_px": float(mean_x[query_index, lane_index].item()),
                "anchor_count": int(lane_anchor.sum().item()),
                "anchor_bce_sum": float(bce_zero[query_index][lane_anchor].sum().item()),
                "anchor_valid_prob_mean": float(valid_prob[query_index][lane_anchor].mean().item()) if bool(lane_anchor.any()) else 0.0,
            }
        )

    return stats, per_lane, examples


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Point-Valid Unmatched Diagnosis",
        "",
        "This is a training-objective diagnostic. It does not change decode, metrics, weights, or labels.",
        "",
        "## Dataset",
        "",
        f"- images: {summary['totals']['images']}",
        f"- gt count histogram: `{summary['gt_count_hist']}`",
        f"- weights: `{summary['config']['weights']}`",
        f"- source: `{summary['config']['source']}`",
        "",
        "## Threshold Summary",
        "",
        "| x_thr_px | close_unmatched_queries | close_query_ratio | close_anchors | close_bce_share_unmatched | close_bce_share_total | close_valid_prob_mean | far_valid_prob_mean |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in summary["thresholds"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(key),
                    str(row["close_unmatched_queries"]),
                    f"{row['close_query_ratio_of_unmatched']:.6f}",
                    str(row["close_unmatched_anchors"]),
                    f"{row['close_bce_share_of_unmatched_neg']:.6f}",
                    f"{row['close_bce_share_of_total_point_valid']:.6f}",
                    f"{row['close_valid_prob_mean']:.6f}",
                    f"{row['far_valid_prob_mean']:.6f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    imgsz = normalize_imgsz(args.imgsz or DATASET_IMAGE_SHAPES[args.dataset])
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    x_thrs = [float(x) for x in args.x_thrs]
    if not x_thrs:
        raise ValueError("--x-thrs must contain at least one threshold.")
    if args.anchor_x_thrs:
        anchor_x_thrs = [float(x) for x in args.anchor_x_thrs]
        if len(anchor_x_thrs) != len(x_thrs):
            raise ValueError("--anchor-x-thrs must have the same length as --x-thrs when provided.")
    else:
        anchor_x_thrs = list(x_thrs)

    valid_prob_thrs = [float(x) for x in args.valid_prob_thrs]
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    legacy_head_patches = _patch_legacy_gcs_head_attrs(model)
    matcher = GCSHungarianMatcher(
        cost_point=float(args.cost_point),
        cost_curve=float(args.cost_curve),
        cost_exist=float(args.cost_exist),
        image_size=imgsz,
        min_overlap=int(args.match_min_overlap),
        max_x_dist=float(args.match_max_x_dist),
        match_gate_px=float(args.match_gate_px),
    )
    images = collect_images(args.source, max_images=int(args.max_images))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"images: {len(images)}")
    print(f"thresholds: {x_thrs}")

    if int(args.warmup) > 0 and images:
        warmup_images = images[: min(len(images), int(args.batch_size))]
        warmup_batch, _, _, _ = _batch_tensors(warmup_images, args.labels, imgsz, device, bool(args.half))
        for _ in range(int(args.warmup)):
            _ = model(warmup_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    total = _new_accumulator()
    dropped_infeasible_gt = 0
    thresholds = {_format_threshold(thr): _new_accumulator() for thr in x_thrs}
    by_gt_count = {_format_threshold(thr): {} for thr in x_thrs}
    by_lane_index = {_format_threshold(thr): {} for thr in x_thrs}
    examples = {_format_threshold(thr): [] for thr in x_thrs}
    gt_count_hist: Counter[int] = Counter()
    started = time.perf_counter()

    for start in tqdm(range(0, len(images), int(args.batch_size)), desc="diagnose"):
        batch_paths = images[start : start + int(args.batch_size)]
        batch, gt_points, gt_valid, meta = _batch_tensors(batch_paths, args.labels, imgsz, device, bool(args.half))
        preds = model(batch)
        if not isinstance(preds, dict) or "pred_points" not in preds or "pred_valid_logits" not in preds:
            raise ValueError("GCS model output must contain pred_points and pred_valid_logits.")
        pred_points = preds["pred_points"].detach()
        pred_logits = preds["pred_logits"].detach()
        pred_valid_logits = preds["pred_valid_logits"].detach()
        if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
            pred_logits = pred_logits.squeeze(-1)
        indices, dropped = _safe_match(matcher, pred_points, pred_logits, gt_points, gt_valid)
        dropped_infeasible_gt += int(dropped)
        target = _point_valid_target(pred_valid_logits, gt_valid, indices)
        point_valid_bce = F.binary_cross_entropy_with_logits(pred_valid_logits.float(), target, reduction="none")

        for b, info in enumerate(meta):
            gt_count = int(info["gt_count"])
            gt_count_hist[gt_count] += 1
            q, k = pred_valid_logits[b].shape
            src_idx, _ = indices[b]
            image_base = _new_accumulator()
            image_base["images"] = 1.0
            image_base["gt_lanes"] = float(gt_count)
            image_base["gt_visible_anchors"] = float(gt_valid[b].sum().item())
            image_base["queries"] = float(q)
            image_base["matched_queries"] = float(src_idx.numel())
            image_base["unmatched_queries"] = float(q - src_idx.numel())
            image_base["point_valid_bce_sum"] = float(point_valid_bce[b].sum().item())
            image_base["point_valid_anchor_count"] = float(point_valid_bce[b].numel())
            image_base["unmatched_anchor_count"] = float((q - src_idx.numel()) * k)
            matched_mask = torch.zeros((q,), dtype=torch.bool, device=device)
            if src_idx.numel():
                matched_mask[src_idx.to(device=device, dtype=torch.long)] = True
            unmatched_mask = ~matched_mask
            image_base["unmatched_neg_bce_sum"] = (
                float(F.softplus(pred_valid_logits[b].float())[unmatched_mask].sum().item())
                if bool(unmatched_mask.any())
                else 0.0
            )
            _add_dict(total, image_base)

            for x_thr, anchor_x_thr in zip(x_thrs, anchor_x_thrs):
                key = _format_threshold(x_thr)
                image_thr_stats, lane_stats, image_examples = _analyze_image(
                    pred_points[b],
                    pred_valid_logits[b],
                    gt_points[b],
                    gt_valid[b],
                    indices[b],
                    image_size=imgsz,
                    x_thr=float(x_thr),
                    anchor_x_thr=float(anchor_x_thr),
                    min_overlap=int(args.min_overlap),
                    valid_prob_thrs=valid_prob_thrs,
                )
                for base_key in (
                    "point_valid_bce_sum",
                    "point_valid_anchor_count",
                    "unmatched_neg_bce_sum",
                    "unmatched_anchor_count",
                ):
                    image_thr_stats[base_key] = image_base[base_key]
                _add_dict(thresholds[key], image_thr_stats)
                group = by_gt_count[key].setdefault(str(gt_count), _new_accumulator())
                _add_dict(group, image_thr_stats)
                for lane_index, stats in lane_stats.items():
                    lane_group = by_lane_index[key].setdefault(str(lane_index), _new_accumulator())
                    _add_dict(lane_group, stats)
                if image_examples and len(examples[key]) < int(args.examples_per_thr):
                    for row in image_examples:
                        row = {
                            **row,
                            "image": info["image"],
                            "label": info["label"],
                            "gt_count": gt_count,
                            "x_thr_px": float(x_thr),
                            "anchor_x_thr_px": float(anchor_x_thr),
                        }
                        examples[key].append(row)
                        if len(examples[key]) >= int(args.examples_per_thr):
                            break

    elapsed_s = time.perf_counter() - started
    summary = {
        "diagnostic_contract": {
            "diagnostic_name": "point_valid_gt_close_unmatched_v1",
            "uses_ground_truth_for_decode_or_metric_selection": False,
            "changes_model_or_training": False,
            "close_query_rule": "Hungarian-unmatched AND GT-visible overlap >= min_overlap AND mean_abs_x_error_px <= x_thr",
            "close_anchor_rule": "GT-visible anchor on a close query/GT pair AND abs_x_error_px <= anchor_x_thr",
            "point_valid_current_target": "matched query uses GT visible mask; unmatched query uses all-zero target",
            "pixel_scale": "training image width from --imgsz, not CULane raw-mask IoU",
            "safe_matcher_for_diagnosis": "GT columns with no finite candidate under the training gate are left unmatched instead of aborting.",
        },
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "source": str(Path(args.source).resolve()),
            "labels": str(Path(args.labels).resolve()) if args.labels else None,
            "dataset": str(args.dataset),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "device": str(args.device),
            "half": bool(args.half),
            "batch_size": int(args.batch_size),
            "max_images": int(args.max_images),
            "x_thrs": x_thrs,
            "anchor_x_thrs": anchor_x_thrs,
            "min_overlap": int(args.min_overlap),
            "valid_prob_thrs": valid_prob_thrs,
            "matcher": {
                "cost_point": float(args.cost_point),
                "cost_curve": float(args.cost_curve),
                "cost_exist": float(args.cost_exist),
                "match_min_overlap": int(args.match_min_overlap),
                "match_max_x_dist": float(args.match_max_x_dist),
                "match_gate_px": float(args.match_gate_px),
            },
            "legacy_gcs_head_attr_patches": int(legacy_head_patches),
        },
        "totals": _finalize_accumulator(total, valid_prob_thrs),
        "gt_count_hist": {str(key): int(value) for key, value in sorted(gt_count_hist.items())},
        "dropped_infeasible_gt_under_match_gate": int(dropped_infeasible_gt),
        "thresholds": {
            key: _finalize_accumulator(value, valid_prob_thrs)
            for key, value in thresholds.items()
        },
        "by_gt_count": {
            key: {
                group_key: _finalize_accumulator(group_stats, valid_prob_thrs)
                for group_key, group_stats in sorted(groups.items(), key=lambda item: int(item[0]))
            }
            for key, groups in by_gt_count.items()
        },
        "by_lane_index": {
            key: {
                lane_key: _finalize_accumulator(lane_stats, valid_prob_thrs)
                for lane_key, lane_stats in sorted(groups.items(), key=lambda item: int(item[0]))
            }
            for key, groups in by_lane_index.items()
        },
        "examples": examples,
        "elapsed_s": elapsed_s,
        "images_per_second": len(images) / max(elapsed_s, 1e-9),
    }

    summary_json = save_dir / "point_valid_unmatched_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_summary_markdown(save_dir / "point_valid_unmatched_summary.md", summary)

    rows: list[dict[str, Any]] = []
    for key, row in summary["thresholds"].items():
        rows.append({"scope": "all", "group": "all", "x_thr_px": key, **row})
        for group_key, group_row in summary["by_gt_count"][key].items():
            rows.append({"scope": "gt_count", "group": group_key, "x_thr_px": key, **group_row})
        for lane_key, lane_row in summary["by_lane_index"][key].items():
            rows.append({"scope": "lane_index", "group": lane_key, "x_thr_px": key, **lane_row})
    if rows:
        csv_path = save_dir / "point_valid_unmatched_summary.csv"
        fields = sorted({field for row in rows for field in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print(f"summary: {summary_json}")
    for key, row in summary["thresholds"].items():
        print(
            "x_thr={key}px close_q={close_q} "
            "close_q_ratio={close_q_ratio:.6f} close_anchor={close_anchor} "
            "bce_share_unmatched={bce_unmatched:.6f} bce_share_total={bce_total:.6f} "
            "close_prob={close_prob:.6f} far_prob={far_prob:.6f}".format(
                key=key,
                close_q=row["close_unmatched_queries"],
                close_q_ratio=row["close_query_ratio_of_unmatched"],
                close_anchor=row["close_unmatched_anchors"],
                bce_unmatched=row["close_bce_share_of_unmatched_neg"],
                bce_total=row["close_bce_share_of_total_point_valid"],
                close_prob=row["close_valid_prob_mean"],
                far_prob=row["far_valid_prob_mean"],
            )
        )
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
