"""Official-GT hard-set coverage diagnostic for GCS short candidates.

This tool uses TuSimple official json-lines records and original image shapes,
not converted GCS label visibility, so it can reproduce the historical
``40/53`` and ``142/183`` short-GT5 denominators.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    find_tusimple_archive_root,
    official_gt_contract_summary,
    read_tusimple_json_lines,
    resolve_tusimple_gt_json,
    tusimple_image_path,
    valid_tusimple_lanes,
)
from tools.infer_gcs import load_gcs_model, preprocess_image  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel, load_checkpoint  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_VAL_GT = (
    ROOT
    / "runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/"
    / "tusimple_official_val_363_folder_aware_seed20260602.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure base/raw-candidate/selected-candidate coverage on TuSimple "
            "official-GT short GT4/GT5 hard lanes."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", default=None, help="Gated-candidate checkpoint.")
    source.add_argument("--model-yaml", default=None, help="Candidate-head model YAML for untrained raw-oracle checks.")
    parser.add_argument(
        "--pretrained-base",
        default=None,
        help="Optional base checkpoint loaded into --model-yaml with shape-safe partial transfer.",
    )
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--gt-json", default=None, help="TuSimple official json-lines GT.")
    parser.add_argument("--allow-noncanonical-gt", action="store_true")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS input shape as H W.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--raw-file-contains",
        default=None,
        help="Optional substring filter, e.g. clips/0601/ for the train0601 hard subset.",
    )
    parser.add_argument("--target-visible-max", type=int, default=10)
    parser.add_argument("--match-min-overlap", type=int, default=3)
    parser.add_argument("--hit-pxs", nargs="+", type=float, default=[20.0, 30.0, 40.0])
    parser.add_argument(
        "--affine-oracle",
        action="store_true",
        help="Ignore candidate-head outputs and enumerate affine candidates from base pred_points.",
    )
    parser.add_argument(
        "--static-proposal-oracle",
        action="store_true",
        help="Ignore candidate-head outputs and add image-independent static proposal templates to the base pool.",
    )
    parser.add_argument(
        "--mask-proposal-oracle",
        action="store_true",
        help="Ignore candidate-head outputs and add connected-component proposals from predicted aux_mask_logits.",
    )
    parser.add_argument(
        "--affine-offsets-px",
        nargs="+",
        type=float,
        default=[0.0, -20.0, 20.0, -40.0, 40.0, -60.0, 60.0],
        help="Pixel offsets used by --affine-oracle.",
    )
    parser.add_argument(
        "--affine-slopes-px",
        nargs="+",
        type=float,
        default=[0.0, -20.0, 20.0, -40.0, 40.0],
        help="Pixel slopes used by --affine-oracle. The value is applied over a centered unit y range.",
    )
    parser.add_argument(
        "--affine-y-mode",
        choices=("normalized_y", "index"),
        default="normalized_y",
        help="Centered y coordinate used by --affine-oracle.",
    )
    parser.add_argument("--proposal-bottom-count", type=int, default=16)
    parser.add_argument("--proposal-top-count", type=int, default=6)
    parser.add_argument("--proposal-bottom-min", type=float, default=0.04)
    parser.add_argument("--proposal-bottom-max", type=float, default=0.96)
    parser.add_argument("--proposal-top-min", type=float, default=0.20)
    parser.add_argument("--proposal-top-max", type=float, default=0.80)
    parser.add_argument(
        "--proposal-curves-px",
        nargs="+",
        type=float,
        default=[0.0],
        help="Mid-lane quadratic curve offsets, in original-image pixels, used by --static-proposal-oracle.",
    )
    parser.add_argument("--mask-proposal-thrs", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--mask-proposal-min-area", type=int, default=20)
    parser.add_argument("--mask-proposal-max-components", type=int, default=16)
    parser.add_argument("--mask-proposal-row-radius", type=int, default=2)
    parser.add_argument("--point-valid-thr", type=float, default=0.6)
    parser.add_argument("--candidate-score-thr", type=float, default=0.05)
    parser.add_argument("--candidate-short-min-points", type=int, default=2)
    parser.add_argument("--candidate-short-max-points", type=int, default=10)
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _default_gt_json(args: argparse.Namespace) -> str | None:
    if args.gt_json:
        return args.gt_json
    if str(args.split).lower() == "val" and DEFAULT_VAL_GT.exists():
        return str(DEFAULT_VAL_GT)
    return None


def _limit_and_filter_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.raw_file_contains:
        needle = str(args.raw_file_contains).replace("\\", "/")
        records = [r for r in records if needle in str(r["raw_file"]).replace("\\", "/")]
    if args.max_images and args.max_images > 0:
        records = records[: int(args.max_images)]
    return records


def _resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    if bool(getattr(args, "affine_oracle", False)):
        mode_prefix = "affine_oracle_"
    elif bool(getattr(args, "static_proposal_oracle", False)):
        mode_prefix = "static_proposal_oracle_"
    elif bool(getattr(args, "mask_proposal_oracle", False)):
        mode_prefix = "mask_proposal_oracle_"
    else:
        mode_prefix = ""
    if args.model_yaml:
        stem = mode_prefix + Path(args.model_yaml).stem
        if args.pretrained_base:
            stem += "_from_" + Path(args.pretrained_base).stem
    else:
        stem = mode_prefix + Path(args.weights).stem
    split_tag = str(args.split)
    if args.raw_file_contains:
        safe = str(args.raw_file_contains).replace("\\", "/").strip("/").replace("/", "_")
        split_tag += f"_{safe}"
    return ROOT / "runs/gcs_lane/short_candidate_hard_coverage" / stem / split_tag


def _load_diag_model(args: argparse.Namespace, device: torch.device, imgsz: tuple[int, int]) -> torch.nn.Module:
    """Load either a trained checkpoint or a YAML model with optional base-weight transfer."""
    if args.model_yaml:
        model = GCSLaneModel(str(args.model_yaml), nc=1, verbose=False).to(device).eval()
        if args.pretrained_base:
            base_model, _ = load_checkpoint(args.pretrained_base, device=device, fuse=False)
            model.load(base_model, verbose=True)
        model.task = "gcs_lane"
        model.gcs_imgsz = [int(imgsz[0]), int(imgsz[1])]
        if isinstance(getattr(model, "args", None), dict):
            model.args["gcs_imgsz"] = [int(imgsz[0]), int(imgsz[1])]
        elif getattr(model, "args", None) is not None:
            model.args.gcs_imgsz = [int(imgsz[0]), int(imgsz[1])]
        if bool(args.half):
            if device.type != "cuda":
                raise ValueError("--half requires a CUDA device.")
            model.half()
        return model
    return load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)


def _set_return_aux(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "return_aux"):
            module.return_aux = bool(enabled)


def _parse_date(raw_file: str) -> str:
    parts = raw_file.replace("\\", "/").split("/")
    return parts[1] if len(parts) >= 2 and parts[0] == "clips" else ""


def _valid_lanes_with_ids(record: dict) -> list[tuple[int, list[float]]]:
    out: list[tuple[int, list[float]]] = []
    for idx, lane in enumerate(record.get("lanes", [])):
        lane_f = [float(x) for x in lane]
        if any(x >= 0.0 for x in lane_f):
            out.append((int(idx), lane_f))
    return out


def _visible_count(lane: list[float]) -> int:
    return int(sum(1 for x in lane if float(x) >= 0.0))


def _target_group_names(split: str, gt_count: int, visible: int, raw_file: str, visible_max: int) -> list[str]:
    if gt_count not in {4, 5} or visible < 1 or visible > int(visible_max):
        return []
    date = _parse_date(raw_file)
    names = [f"short_gt{gt_count}", "short_gt4_gt5"]
    if date:
        names.append(f"date{date}_short_gt{gt_count}")
    if str(split) == "train" and date == "0601":
        names.append(f"train0601_short_gt{gt_count}")
    return names


def _interp_points_to_hsamples(
    points_norm: np.ndarray,
    h_samples: list[float],
    image_shape: tuple[int, int],
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    points = np.asarray(points_norm, dtype=np.float32).reshape(-1, 2).copy()
    if valid_mask is not None:
        keep = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if keep.shape[0] != points.shape[0]:
            raise ValueError(f"valid_mask must have {points.shape[0]} elements, got {keep.shape[0]}.")
        points = points[keep]
    if points.shape[0] < 2:
        return np.full(len(h_samples), np.nan, dtype=np.float32)
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * float(w)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * float(h)
    order = np.argsort(points[:, 1], kind="stable")
    points = points[order]
    ys = points[:, 1]
    xs = points[:, 0]
    unique_ys, unique_idx = np.unique(np.round(ys, decimals=4), return_index=True)
    if unique_ys.shape[0] < 2:
        return np.full(len(h_samples), np.nan, dtype=np.float32)
    unique_xs = xs[unique_idx]
    return np.interp(np.asarray(h_samples, dtype=np.float32), unique_ys, unique_xs, left=np.nan, right=np.nan).astype(
        np.float32
    )


def _ape_px(pred_xs: np.ndarray, gt_xs: list[float], min_overlap: int) -> tuple[float, int]:
    gt = np.asarray(gt_xs, dtype=np.float32)
    pred = np.asarray(pred_xs, dtype=np.float32)
    valid = (gt >= 0.0) & np.isfinite(pred)
    overlap = int(valid.sum())
    if overlap < int(min_overlap):
        return float("inf"), overlap
    return float(np.mean(np.abs(pred[valid] - gt[valid]))), overlap


def _best_ape(pool_xs: list[np.ndarray], gt_lane: list[float], min_overlap: int) -> tuple[float, int, int]:
    best_ape = float("inf")
    best_idx = -1
    best_overlap = 0
    for idx, xs in enumerate(pool_xs):
        ape, overlap = _ape_px(xs, gt_lane, min_overlap=min_overlap)
        if ape < best_ape:
            best_ape = float(ape)
            best_idx = int(idx)
            best_overlap = int(overlap)
    return best_ape, best_idx, best_overlap


def _centered_y(points: torch.Tensor, mode: str) -> torch.Tensor:
    """Return Q x K centered y coordinates in roughly [-0.5, 0.5]."""
    q, k = int(points.shape[0]), int(points.shape[1])
    if mode == "index":
        values = torch.linspace(-0.5, 0.5, k, dtype=points.dtype, device=points.device)
        return values.view(1, k).expand(q, k)
    y = points[..., 1]
    y_min = y.min(dim=1, keepdim=True).values
    y_max = y.max(dim=1, keepdim=True).values
    denom = (y_max - y_min).clamp_min(1e-6)
    return (y - (y_min + y_max) * 0.5) / denom


def _make_affine_candidate_points(
    base_points: torch.Tensor,
    image_shape: tuple[int, int],
    offsets_px: list[float],
    slopes_px: list[float],
    y_mode: str,
) -> tuple[torch.Tensor, list[tuple[float, float]]]:
    """Generate base + offset + slope * centered_y candidates in normalized coordinates."""
    if base_points.ndim != 3 or base_points.shape[-1] != 2:
        raise RuntimeError(f"Bad base point shape for affine oracle: {tuple(base_points.shape)}")
    image_w = float(image_shape[1])
    if image_w <= 0.0:
        raise ValueError(f"Invalid image width for affine oracle: {image_shape}")
    pairs = [(float(offset), float(slope)) for offset in offsets_px for slope in slopes_px]
    if not pairs:
        raise ValueError("Affine oracle requires at least one offset/slope pair.")
    q, k = int(base_points.shape[0]), int(base_points.shape[1])
    pair_t = torch.tensor(pairs, dtype=base_points.dtype, device=base_points.device)
    offsets = pair_t[:, 0].view(1, -1, 1) / image_w
    slopes = pair_t[:, 1].view(1, -1, 1) / image_w
    centered = _centered_y(base_points, y_mode).view(q, 1, k)
    candidates = base_points.unsqueeze(1).expand(q, len(pairs), k, 2).clone()
    candidates[..., 0] = (candidates[..., 0] + offsets + slopes * centered).clamp(0.0, 1.0)
    candidates[..., 1] = candidates[..., 1].clamp(0.0, 1.0)
    return candidates, pairs


def _linspace_values(min_value: float, max_value: float, count: int) -> list[float]:
    if int(count) <= 0:
        raise ValueError(f"Proposal grid count must be positive, got {count}.")
    if int(count) == 1:
        return [float((float(min_value) + float(max_value)) * 0.5)]
    return [float(x) for x in np.linspace(float(min_value), float(max_value), int(count), dtype=np.float32)]


def _make_static_proposal_points(
    base_points: torch.Tensor,
    image_shape: tuple[int, int],
    bottom_xs: list[float],
    top_xs: list[float],
    curves_px: list[float],
) -> tuple[torch.Tensor, list[tuple[float, float, float]]]:
    """Generate image-independent straight/quadratic lane proposal templates."""
    if base_points.ndim != 3 or base_points.shape[-1] != 2:
        raise RuntimeError(f"Bad base point shape for static proposal oracle: {tuple(base_points.shape)}")
    image_w = float(image_shape[1])
    if image_w <= 0.0:
        raise ValueError(f"Invalid image width for static proposal oracle: {image_shape}")
    pairs = [(float(bottom_x), float(top_x), float(curve)) for bottom_x in bottom_xs for top_x in top_xs for curve in curves_px]
    if not pairs:
        raise ValueError("Static proposal oracle requires at least one proposal template.")
    k = int(base_points.shape[1])
    y = base_points[0, :, 1].clamp(0.0, 1.0)
    t = torch.linspace(0.0, 1.0, k, dtype=base_points.dtype, device=base_points.device)
    curve_basis = 4.0 * t * (1.0 - t)
    proposal_t = torch.tensor(pairs, dtype=base_points.dtype, device=base_points.device)
    bottom = proposal_t[:, 0:1]
    top = proposal_t[:, 1:2]
    curve = proposal_t[:, 2:3] / image_w
    x = bottom * (1.0 - t.view(1, k)) + top * t.view(1, k) + curve * curve_basis.view(1, k)
    points = torch.stack((x.clamp(0.0, 1.0), y.view(1, k).expand(len(pairs), k)), dim=-1)
    return points.unsqueeze(1), pairs


def _mask_lane_probability(aux_mask_logits: torch.Tensor) -> np.ndarray:
    logits = aux_mask_logits.detach().float().cpu()
    if logits.ndim != 3:
        raise RuntimeError(f"aux_mask_logits[0] must have shape C x H x W, got {tuple(logits.shape)}")
    if int(logits.shape[0]) == 1:
        prob = logits[0].sigmoid()
    elif int(logits.shape[0]) >= 2:
        prob = logits.softmax(dim=0)[1]
    else:
        raise RuntimeError(f"aux_mask_logits has no channels: {tuple(logits.shape)}")
    return prob.numpy().astype(np.float32)


def _mask_component_proposal_pool(
    aux_mask_logits: torch.Tensor,
    h_samples: list[float],
    image_shape: tuple[int, int],
    thresholds: list[float],
    min_area: int,
    max_components: int,
    row_radius: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Extract sparse x-at-hsample proposals from predicted semantic mask components."""
    prob = _mask_lane_probability(aux_mask_logits)
    mask_h, mask_w = int(prob.shape[0]), int(prob.shape[1])
    orig_h, orig_w = int(image_shape[0]), int(image_shape[1])
    if mask_h <= 0 or mask_w <= 0 or orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid mask/image shape: mask={prob.shape}, image={image_shape}")
    pool: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for thr in thresholds:
        binary = (prob >= float(thr)).astype(np.uint8)
        if int(binary.sum()) == 0:
            continue
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        components: list[tuple[int, int]] = []
        for label in range(1, int(num_labels)):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= int(min_area):
                components.append((area, label))
        components.sort(reverse=True)
        for area, label in components[: int(max_components)]:
            ys, xs = np.where(labels == int(label))
            pred_xs = np.full(len(h_samples), np.nan, dtype=np.float32)
            for i, y_orig in enumerate(h_samples):
                y_mask = int(round(float(y_orig) / max(float(orig_h - 1), 1.0) * float(mask_h - 1)))
                lo = max(0, y_mask - int(row_radius))
                hi = min(mask_h - 1, y_mask + int(row_radius))
                sel = (ys >= lo) & (ys <= hi)
                if bool(np.any(sel)):
                    x_mask = float(np.median(xs[sel]))
                    pred_xs[i] = x_mask / max(float(mask_w - 1), 1.0) * float(orig_w - 1)
            finite_count = int(np.isfinite(pred_xs).sum())
            if finite_count <= 0:
                continue
            pool.append(pred_xs)
            cx, cy = centroids[int(label)]
            meta.append(
                {
                    "source": "mask_component",
                    "query": -1,
                    "index": len(meta),
                    "mask_thr": float(thr),
                    "mask_area": int(area),
                    "mask_finite_points": int(finite_count),
                    "mask_centroid_x": float(cx),
                    "mask_centroid_y": float(cy),
                }
            )
    return pool, meta


def _percentile(values: list[float], pct: float) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite, dtype=np.float32), float(pct)))


def _hit_key(px: float) -> str:
    return str(int(px)) if float(px).is_integer() else str(px).replace(".", "p")


def _empty_stats(hit_pxs: list[float]) -> dict[str, Any]:
    return {
        "total": 0,
        "base_apes": [],
        "raw_candidate_apes": [],
        "selected_all_apes": [],
        "selected_gated_apes": [],
        "oracle_query_gate_count": 0,
        "oracle_query_selected_count": 0,
        **{f"base_hit{_hit_key(px)}": 0 for px in hit_pxs},
        **{f"raw_candidate_hit{_hit_key(px)}": 0 for px in hit_pxs},
        **{f"selected_all_hit{_hit_key(px)}": 0 for px in hit_pxs},
        **{f"selected_gated_hit{_hit_key(px)}": 0 for px in hit_pxs},
        **{f"base_to_raw_gain{_hit_key(px)}": 0 for px in hit_pxs},
        **{f"base_to_raw_loss{_hit_key(px)}": 0 for px in hit_pxs},
        **{f"base_to_selected_gated_gain{_hit_key(px)}": 0 for px in hit_pxs},
        **{f"base_to_selected_gated_loss{_hit_key(px)}": 0 for px in hit_pxs},
        **{f"raw_oracle_missed_by_selected_gated{_hit_key(px)}": 0 for px in hit_pxs},
    }


def _update_stats(stats: dict[str, Any], row: dict[str, Any], hit_pxs: list[float]) -> None:
    stats["total"] += 1
    stats["base_apes"].append(float(row["base_ape_px"]))
    stats["raw_candidate_apes"].append(float(row["raw_candidate_ape_px"]))
    stats["selected_all_apes"].append(float(row["selected_all_ape_px"]))
    stats["selected_gated_apes"].append(float(row["selected_gated_ape_px"]))
    stats["oracle_query_gate_count"] += int(bool(row["oracle_query_gate_applies"]))
    stats["oracle_query_selected_count"] += int(bool(row["oracle_query_selected_oracle_candidate"]))
    for px in hit_pxs:
        key = _hit_key(px)
        base_hit = float(row["base_ape_px"]) <= float(px)
        raw_hit = float(row["raw_candidate_ape_px"]) <= float(px)
        selected_all_hit = float(row["selected_all_ape_px"]) <= float(px)
        selected_gated_hit = float(row["selected_gated_ape_px"]) <= float(px)
        stats[f"base_hit{key}"] += int(base_hit)
        stats[f"raw_candidate_hit{key}"] += int(raw_hit)
        stats[f"selected_all_hit{key}"] += int(selected_all_hit)
        stats[f"selected_gated_hit{key}"] += int(selected_gated_hit)
        stats[f"base_to_raw_gain{key}"] += int((not base_hit) and raw_hit)
        stats[f"base_to_raw_loss{key}"] += int(base_hit and (not raw_hit))
        stats[f"base_to_selected_gated_gain{key}"] += int((not base_hit) and selected_gated_hit)
        stats[f"base_to_selected_gated_loss{key}"] += int(base_hit and (not selected_gated_hit))
        stats[f"raw_oracle_missed_by_selected_gated{key}"] += int(raw_hit and (not selected_gated_hit))


def _summarize_stats(stats: dict[str, Any], hit_pxs: list[float]) -> dict[str, Any]:
    total = int(stats["total"])

    def ape_summary(name: str) -> dict[str, float | None]:
        vals = [float(v) for v in stats[name] if math.isfinite(float(v))]
        return {
            f"{name[:-1]}mean_px": round(float(sum(vals) / len(vals)), 6) if vals else None,
            f"{name[:-1]}median_px": round(float(median(vals)), 6) if vals else None,
            f"{name[:-1]}p90_px": None if (p90 := _percentile(vals, 90.0)) is None else round(float(p90), 6),
        }

    out: dict[str, Any] = {
        "total": total,
        "oracle_query_gate_count": int(stats["oracle_query_gate_count"]),
        "oracle_query_gate_rate": round(float(stats["oracle_query_gate_count"]) / max(total, 1), 6),
        "oracle_query_selected_count": int(stats["oracle_query_selected_count"]),
        "oracle_query_selected_rate": round(float(stats["oracle_query_selected_count"]) / max(total, 1), 6),
    }
    for name in ("base_apes", "raw_candidate_apes", "selected_all_apes", "selected_gated_apes"):
        out.update(ape_summary(name))
    for px in hit_pxs:
        key = _hit_key(px)
        for prefix in (
            "base_hit",
            "raw_candidate_hit",
            "selected_all_hit",
            "selected_gated_hit",
            "base_to_raw_gain",
            "base_to_raw_loss",
            "base_to_selected_gated_gain",
            "base_to_selected_gated_loss",
            "raw_oracle_missed_by_selected_gated",
        ):
            field = f"{prefix}{key}"
            value = int(stats[field])
            out[field] = value
            if prefix.endswith("hit"):
                out[f"{field}_rate"] = round(float(value) / max(total, 1), 6)
    return out


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    oracle_modes = [bool(args.affine_oracle), bool(args.static_proposal_oracle), bool(args.mask_proposal_oracle)]
    if sum(int(x) for x in oracle_modes) > 1:
        raise ValueError("--affine-oracle, --static-proposal-oracle, and --mask-proposal-oracle are mutually exclusive.")
    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_json = _default_gt_json(args)
    gt_path = resolve_tusimple_gt_json(archive_root, split=args.split, gt_json=gt_json)
    gt_records = _limit_and_filter_records(read_tusimple_json_lines(gt_path), args)
    if not gt_records:
        raise ValueError("No TuSimple GT records remain after filtering.")

    gt_contract = official_gt_contract_summary(
        split=args.split,
        gt_json=gt_path,
        gt_records=gt_records,
        allow_noncanonical_gt=bool(args.allow_noncanonical_gt),
    )
    imgsz = normalize_imgsz(args.imgsz, dataset="tusimple")
    hit_pxs = sorted({float(x) for x in args.hit_pxs} | {20.0, 30.0, 40.0})

    device_obj = select_device(args.device)
    model = _load_diag_model(args, device=device_obj, imgsz=imgsz)
    _set_return_aux(model, enabled=bool(args.mask_proposal_oracle))

    if args.warmup > 0:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device_obj, half=bool(args.half))
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    rows: list[dict[str, Any]] = []
    stats = defaultdict(lambda: _empty_stats(hit_pxs))
    infer_time_s = 0.0
    post_time_s = 0.0

    for record in gt_records:
        raw_file = str(record["raw_file"]).replace("\\", "/")
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        original_shape = img.shape[:2]
        h_samples = [float(x) for x in record["h_samples"]]
        tensor = preprocess_image(img, imgsz, device=device_obj, half=bool(args.half))

        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        t1 = time.perf_counter()

        base_points = preds["pred_points"][0].detach().float().cpu().clamp(0.0, 1.0)
        pred_valid_logits = preds.get("pred_valid_logits")
        if pred_valid_logits is None:
            raise RuntimeError("This diagnostic requires pred_valid_logits for candidate gate analysis.")
        candidate_points_t = preds.get("pred_short_candidate_points")
        candidate_logits_t = preds.get("pred_short_candidate_logits")
        segment_points_t = preds.get("pred_short_segment_points")
        segment_logits_t = preds.get("pred_short_segment_logits")
        segment_replace_logits_t = preds.get("pred_short_segment_replace_logits")
        segment_window_mask_t = preds.get("pred_short_segment_window_mask")
        affine_pairs: list[tuple[float, float]] | None = None
        static_pairs: list[tuple[float, float, float]] | None = None
        candidate_window_masks: torch.Tensor | None = None
        candidate_replace_logits: torch.Tensor | None = None
        model_candidate_source = "candidate_head"
        if bool(args.affine_oracle):
            candidate_points, affine_pairs = _make_affine_candidate_points(
                base_points,
                original_shape,
                offsets_px=[float(x) for x in args.affine_offsets_px],
                slopes_px=[float(x) for x in args.affine_slopes_px],
                y_mode=str(args.affine_y_mode),
            )
            candidate_logits = torch.zeros(candidate_points.shape[:2], dtype=torch.float32)
        elif bool(args.static_proposal_oracle):
            static_points, static_pairs = _make_static_proposal_points(
                base_points,
                original_shape,
                bottom_xs=_linspace_values(args.proposal_bottom_min, args.proposal_bottom_max, args.proposal_bottom_count),
                top_xs=_linspace_values(args.proposal_top_min, args.proposal_top_max, args.proposal_top_count),
                curves_px=[float(x) for x in args.proposal_curves_px],
            )
            candidate_points = torch.cat((base_points.unsqueeze(1), static_points), dim=0)
            candidate_logits = torch.zeros(candidate_points.shape[:2], dtype=torch.float32)
        elif bool(args.mask_proposal_oracle):
            if preds.get("aux_mask_logits") is None:
                raise RuntimeError("--mask-proposal-oracle requires aux_mask_logits.")
            candidate_points = base_points.unsqueeze(1)
            candidate_logits = torch.zeros(candidate_points.shape[:2], dtype=torch.float32)
        elif candidate_points_t is not None and candidate_logits_t is not None:
            candidate_points = candidate_points_t[0].detach().float().cpu().clamp(0.0, 1.0)
            candidate_logits = candidate_logits_t[0].detach().float().cpu()
        elif segment_points_t is not None and segment_logits_t is not None and segment_window_mask_t is not None:
            candidate_points = segment_points_t[0].detach().float().cpu().clamp(0.0, 1.0)
            candidate_logits = segment_logits_t[0].detach().float().cpu()
            if segment_replace_logits_t is not None:
                candidate_replace_logits = segment_replace_logits_t[0].detach().float().cpu()
                if candidate_replace_logits.shape != candidate_logits.shape:
                    raise RuntimeError(
                        "Bad pred_short_segment_replace_logits shape: "
                        f"{tuple(candidate_replace_logits.shape)} vs logits={tuple(candidate_logits.shape)}"
                    )
            mask_t = segment_window_mask_t.detach().cpu()
            if mask_t.ndim == 2:
                candidate_window_masks = mask_t.bool().view(1, *mask_t.shape).expand(candidate_points.shape[0], -1, -1)
            elif mask_t.ndim == 4:
                candidate_window_masks = mask_t[0].bool()
            else:
                raise RuntimeError(f"Bad pred_short_segment_window_mask shape: {tuple(mask_t.shape)}")
            model_candidate_source = "segment_head"
        else:
            raise RuntimeError(
                "This diagnostic requires pred_short_candidate_points/logits or "
                "pred_short_segment_points/logits/window_mask."
            )

        valid_scores = pred_valid_logits[0].detach().float().cpu().sigmoid()
        if candidate_points.ndim != 4 or candidate_logits.shape != candidate_points.shape[:2]:
            raise RuntimeError(
                f"Bad candidate output shapes: points={tuple(candidate_points.shape)}, logits={tuple(candidate_logits.shape)}"
            )

        visible_counts = (valid_scores >= float(args.point_valid_thr)).sum(dim=1)
        short_gate = (visible_counts >= int(args.candidate_short_min_points)) & (
            visible_counts <= int(args.candidate_short_max_points)
        )
        if bool(args.affine_oracle) or bool(args.static_proposal_oracle) or bool(args.mask_proposal_oracle):
            candidate_prob = candidate_logits.sigmoid()
            best_scores, best_indices = candidate_prob.max(dim=1)
            short_gate_full = torch.zeros(candidate_points.shape[0], dtype=torch.bool)
            short_gate_full[: short_gate.shape[0]] = short_gate
            short_gate = short_gate_full
            apply_gate = torch.zeros(candidate_points.shape[0], dtype=torch.bool)
            best_replace_scores = torch.zeros_like(best_scores)
            selected_segment_lengths = torch.zeros_like(best_indices)
        elif candidate_replace_logits is not None and candidate_window_masks is not None:
            segment_lengths = candidate_window_masks.sum(dim=2)
            segment_length_gate = (segment_lengths >= int(args.candidate_short_min_points)) & (
                segment_lengths <= int(args.candidate_short_max_points)
            )
            selection_logits = candidate_logits + candidate_replace_logits
            selection_scores = selection_logits.sigmoid().masked_fill(~segment_length_gate, -1.0)
            best_scores, best_indices = selection_scores.max(dim=1)
            replace_scores = candidate_replace_logits.sigmoid()
            best_replace_scores = replace_scores.gather(1, best_indices.view(-1, 1)).squeeze(1)
            selected_segment_lengths = segment_lengths.gather(1, best_indices.view(-1, 1)).squeeze(1)
            short_gate = segment_length_gate.gather(1, best_indices.view(-1, 1)).squeeze(1)
            apply_gate = short_gate & (best_replace_scores >= float(args.candidate_score_thr))
        else:
            candidate_prob = candidate_logits.sigmoid()
            best_scores, best_indices = candidate_prob.max(dim=1)
            apply_gate = short_gate & (best_scores >= float(args.candidate_score_thr))
            best_replace_scores = torch.zeros_like(best_scores)
            selected_segment_lengths = torch.zeros_like(best_indices)

        q_count = int(base_points.shape[0])
        candidate_q_count = int(candidate_points.shape[0])
        cand_count = int(candidate_points.shape[1])
        base_pool_xs = [
            _interp_points_to_hsamples(base_points[q].numpy(), h_samples, original_shape) for q in range(q_count)
        ]
        raw_candidate_pool: list[np.ndarray] = []
        raw_candidate_meta: list[dict[str, Any]] = []
        for q in range(candidate_q_count):
            for m in range(cand_count):
                valid_mask = None
                if candidate_window_masks is not None:
                    valid_mask = candidate_window_masks[q, m].numpy()
                raw_candidate_pool.append(
                    _interp_points_to_hsamples(
                        candidate_points[q, m].numpy(),
                        h_samples,
                        original_shape,
                        valid_mask=valid_mask,
                    )
                )
                meta: dict[str, Any] = {
                    "source": model_candidate_source,
                    "query": int(q),
                    "index": int(m),
                    "offset_px": "",
                    "slope_px": "",
                    "segment_start": "",
                    "segment_end": "",
                    "segment_length": "",
                    "bottom_x": "",
                    "top_x": "",
                    "curve_px": "",
                    "mask_thr": "",
                    "mask_area": "",
                    "mask_finite_points": "",
                    "mask_centroid_x": "",
                    "mask_centroid_y": "",
                }
                if affine_pairs is not None:
                    meta["source"] = "affine"
                    if 0 <= m < len(affine_pairs):
                        meta["offset_px"] = float(affine_pairs[m][0])
                        meta["slope_px"] = float(affine_pairs[m][1])
                elif static_pairs is not None:
                    if 0 <= q < q_count:
                        meta["source"] = "base"
                    else:
                        static_idx = int(q - q_count)
                        meta["source"] = "static_proposal"
                        if 0 <= static_idx < len(static_pairs):
                            meta["bottom_x"] = float(static_pairs[static_idx][0])
                            meta["top_x"] = float(static_pairs[static_idx][1])
                            meta["curve_px"] = float(static_pairs[static_idx][2])
                elif bool(args.mask_proposal_oracle):
                    meta["source"] = "base"
                elif candidate_window_masks is not None:
                    valid_idx = torch.nonzero(candidate_window_masks[q, m], as_tuple=False).flatten()
                    if valid_idx.numel() > 0:
                        meta["segment_start"] = int(valid_idx[0].item())
                        meta["segment_end"] = int(valid_idx[-1].item())
                        meta["segment_length"] = int(valid_idx.numel())
                raw_candidate_meta.append(meta)
        if bool(args.mask_proposal_oracle):
            mask_pool, mask_meta = _mask_component_proposal_pool(
                preds["aux_mask_logits"][0],
                h_samples,
                original_shape,
                thresholds=[float(x) for x in args.mask_proposal_thrs],
                min_area=int(args.mask_proposal_min_area),
                max_components=int(args.mask_proposal_max_components),
                row_radius=int(args.mask_proposal_row_radius),
            )
            raw_candidate_pool.extend(mask_pool)
            raw_candidate_meta.extend(mask_meta)
        selected_all_pool_xs = []
        selected_gated_pool_xs = []
        for q in range(q_count):
            m = int(best_indices[q].item())
            selected_mask = candidate_window_masks[q, m].numpy() if candidate_window_masks is not None else None
            selected_all_pool_xs.append(
                _interp_points_to_hsamples(
                    candidate_points[q, m].numpy(),
                    h_samples,
                    original_shape,
                    valid_mask=selected_mask,
                )
            )
            if bool(apply_gate[q].item()):
                selected_gated_pool_xs.append(selected_all_pool_xs[-1])
            else:
                selected_gated_pool_xs.append(base_pool_xs[q])

        gt_count = int(len(valid_tusimple_lanes(record.get("lanes", []))))
        for gt_lane_id, gt_lane in _valid_lanes_with_ids(record):
            visible = _visible_count(gt_lane)
            group_names = _target_group_names(args.split, gt_count, visible, raw_file, args.target_visible_max)
            if not group_names:
                continue

            base_ape, base_idx, base_overlap = _best_ape(
                base_pool_xs, gt_lane, min_overlap=int(args.match_min_overlap)
            )
            raw_ape, raw_idx, raw_overlap = _best_ape(
                raw_candidate_pool, gt_lane, min_overlap=int(args.match_min_overlap)
            )
            selected_all_ape, selected_all_idx, selected_all_overlap = _best_ape(
                selected_all_pool_xs, gt_lane, min_overlap=int(args.match_min_overlap)
            )
            selected_gated_ape, selected_gated_idx, selected_gated_overlap = _best_ape(
                selected_gated_pool_xs, gt_lane, min_overlap=int(args.match_min_overlap)
            )

            oracle_q = int(raw_idx // cand_count) if raw_idx >= 0 else -1
            oracle_m = int(raw_idx % cand_count) if raw_idx >= 0 else -1
            raw_meta = raw_candidate_meta[raw_idx] if 0 <= raw_idx < len(raw_candidate_meta) else {}
            raw_source = str(raw_meta.get("source", "candidate_head"))
            oracle_has_base_query = 0 <= oracle_q < int(visible_counts.shape[0])
            oracle_has_candidate_query = 0 <= oracle_q < int(best_scores.shape[0])
            row = {
                "split": str(args.split),
                "raw_file": raw_file,
                "date": _parse_date(raw_file),
                "gt_count": int(gt_count),
                "gt_lane_id": int(gt_lane_id),
                "visible_points_gt": int(visible),
                "groups": ";".join(group_names),
                "base_ape_px": float(base_ape),
                "base_best_query": int(base_idx),
                "base_best_overlap": int(base_overlap),
                "raw_candidate_ape_px": float(raw_ape),
                "raw_candidate_source": raw_source,
                "raw_candidate_query": int(raw_meta.get("query", oracle_q)),
                "raw_candidate_index": int(raw_meta.get("index", oracle_m)),
                "raw_candidate_offset_px": raw_meta.get("offset_px", ""),
                "raw_candidate_slope_px": raw_meta.get("slope_px", ""),
                "raw_candidate_segment_start": raw_meta.get("segment_start", ""),
                "raw_candidate_segment_end": raw_meta.get("segment_end", ""),
                "raw_candidate_segment_length": raw_meta.get("segment_length", ""),
                "raw_candidate_bottom_x": raw_meta.get("bottom_x", ""),
                "raw_candidate_top_x": raw_meta.get("top_x", ""),
                "raw_candidate_curve_px": raw_meta.get("curve_px", ""),
                "raw_candidate_mask_thr": raw_meta.get("mask_thr", ""),
                "raw_candidate_mask_area": raw_meta.get("mask_area", ""),
                "raw_candidate_mask_finite_points": raw_meta.get("mask_finite_points", ""),
                "raw_candidate_mask_centroid_x": raw_meta.get("mask_centroid_x", ""),
                "raw_candidate_mask_centroid_y": raw_meta.get("mask_centroid_y", ""),
                "raw_candidate_overlap": int(raw_overlap),
                "selected_all_ape_px": float(selected_all_ape),
                "selected_all_query": int(selected_all_idx),
                "selected_all_overlap": int(selected_all_overlap),
                "selected_gated_ape_px": float(selected_gated_ape),
                "selected_gated_query": int(selected_gated_idx),
                "selected_gated_overlap": int(selected_gated_overlap),
                "oracle_query_visible_count": int(visible_counts[oracle_q].item()) if oracle_has_base_query else -1,
                "oracle_query_short_gate": bool(short_gate[oracle_q].item()) if 0 <= oracle_q < int(short_gate.shape[0]) else False,
                "oracle_query_best_score": float(best_scores[oracle_q].item()) if oracle_has_candidate_query else float("nan"),
                "oracle_query_best_replace_score": float(best_replace_scores[oracle_q].item())
                if oracle_has_candidate_query
                else float("nan"),
                "oracle_query_gate_applies": bool(apply_gate[oracle_q].item()) if 0 <= oracle_q < int(apply_gate.shape[0]) else False,
                "oracle_query_selected_candidate": int(best_indices[oracle_q].item()) if oracle_has_candidate_query else -1,
                "oracle_query_selected_segment_length": int(selected_segment_lengths[oracle_q].item())
                if oracle_has_candidate_query
                else -1,
                "oracle_query_selected_oracle_candidate": bool(int(best_indices[oracle_q].item()) == oracle_m)
                if oracle_has_candidate_query
                else False,
            }
            for px in hit_pxs:
                key = _hit_key(px)
                row[f"base_hit{key}"] = bool(float(base_ape) <= float(px))
                row[f"raw_candidate_hit{key}"] = bool(float(raw_ape) <= float(px))
                row[f"selected_all_hit{key}"] = bool(float(selected_all_ape) <= float(px))
                row[f"selected_gated_hit{key}"] = bool(float(selected_gated_ape) <= float(px))
            rows.append(row)
            for group in group_names:
                _update_stats(stats[group], row, hit_pxs)
        t2 = time.perf_counter()
        infer_time_s += t1 - t0
        post_time_s += t2 - t1

    save_dir = _resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(save_dir / "short_candidate_hard_lane_rows.csv", rows)
    group_summary = {name: _summarize_stats(value, hit_pxs) for name, value in sorted(stats.items())}
    summary = {
        "config": {
            "oracle_type": (
                "affine_from_base_pred_points"
                if bool(args.affine_oracle)
                else "static_proposal_templates"
                if bool(args.static_proposal_oracle)
                else "mask_component_proposals"
                if bool(args.mask_proposal_oracle)
                else "model_short_segment_head"
                if rows and any(str(r.get("raw_candidate_source", "")) == "segment_head" for r in rows)
                else "model_candidate_head"
            ),
            "weights": str(Path(args.weights).resolve()) if args.weights else None,
            "model_yaml": str(Path(args.model_yaml).resolve()) if args.model_yaml else None,
            "pretrained_base": str(Path(args.pretrained_base).resolve()) if args.pretrained_base else None,
            "archive_root": str(archive_root.resolve()),
            "split": str(args.split),
            "gt_json": str(gt_path.resolve()),
            "raw_file_contains": args.raw_file_contains,
            "records": int(len(gt_records)),
            "target_visible_max": int(args.target_visible_max),
            "match_min_overlap": int(args.match_min_overlap),
            "hit_pxs": hit_pxs,
            "affine_oracle": bool(args.affine_oracle),
            "affine_offsets_px": [float(x) for x in args.affine_offsets_px] if bool(args.affine_oracle) else None,
            "affine_slopes_px": [float(x) for x in args.affine_slopes_px] if bool(args.affine_oracle) else None,
            "affine_y_mode": str(args.affine_y_mode) if bool(args.affine_oracle) else None,
            "static_proposal_oracle": bool(args.static_proposal_oracle),
            "static_proposal_count": int(args.proposal_bottom_count)
            * int(args.proposal_top_count)
            * max(len(args.proposal_curves_px), 1)
            if bool(args.static_proposal_oracle)
            else None,
            "static_candidate_pool_includes_base_q12": bool(args.static_proposal_oracle),
            "proposal_bottom_count": int(args.proposal_bottom_count) if bool(args.static_proposal_oracle) else None,
            "proposal_top_count": int(args.proposal_top_count) if bool(args.static_proposal_oracle) else None,
            "proposal_bottom_min": float(args.proposal_bottom_min) if bool(args.static_proposal_oracle) else None,
            "proposal_bottom_max": float(args.proposal_bottom_max) if bool(args.static_proposal_oracle) else None,
            "proposal_top_min": float(args.proposal_top_min) if bool(args.static_proposal_oracle) else None,
            "proposal_top_max": float(args.proposal_top_max) if bool(args.static_proposal_oracle) else None,
            "proposal_curves_px": [float(x) for x in args.proposal_curves_px]
            if bool(args.static_proposal_oracle)
            else None,
            "mask_proposal_oracle": bool(args.mask_proposal_oracle),
            "mask_proposal_thrs": [float(x) for x in args.mask_proposal_thrs]
            if bool(args.mask_proposal_oracle)
            else None,
            "mask_proposal_min_area": int(args.mask_proposal_min_area) if bool(args.mask_proposal_oracle) else None,
            "mask_proposal_max_components": int(args.mask_proposal_max_components)
            if bool(args.mask_proposal_oracle)
            else None,
            "mask_proposal_row_radius": int(args.mask_proposal_row_radius) if bool(args.mask_proposal_oracle) else None,
            "point_valid_thr": float(args.point_valid_thr),
            "candidate_score_thr": float(args.candidate_score_thr),
            "candidate_short_min_points": int(args.candidate_short_min_points),
            "candidate_short_max_points": int(args.candidate_short_max_points),
            "selected_gate_mode": (
                "segment_length_and_replace_score"
                if rows and any(str(r.get("raw_candidate_source", "")) == "segment_head" for r in rows)
                and any(math.isfinite(float(r.get("oracle_query_best_replace_score", float("nan")))) for r in rows)
                else "base_visible_count_and_candidate_score"
            ),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "device": str(args.device),
            "half": bool(args.half),
            "save_dir": str(save_dir.resolve()),
            "test_closed": True,
        },
        "groups": group_summary,
        "outputs": {
            "rows_csv": str((save_dir / "short_candidate_hard_lane_rows.csv").resolve()),
            "summary_json": str((save_dir / "short_candidate_hard_summary.json").resolve()),
        },
        "timing": {
            "avg_inference_ms": round(infer_time_s * 1000.0 / max(len(gt_records), 1), 4),
            "avg_postprocess_ms": round(post_time_s * 1000.0 / max(len(gt_records), 1), 4),
            "avg_total_ms": round((infer_time_s + post_time_s) * 1000.0 / max(len(gt_records), 1), 4),
        },
        **gt_contract,
    }
    (save_dir / "short_candidate_hard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"groups": group_summary, "test_closed": True}, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
