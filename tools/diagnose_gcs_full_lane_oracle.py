"""Diagnose full-visible-span proposal coverage for the independent full-lane head.

This is a prediction-only upper-bound diagnostic.  A proposal counts as a
full-visible-span match only when its predicted visible mask covers every
visible GT h_sample.  Partial-window overlap is reported separately and is
never promoted to a full-lane hit.
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
from ultralytics.utils.gcs_fixed_y import validate_tusimple_h_samples_asc  # noqa: E402
from ultralytics.utils.gcs_full_lane import (  # noqa: E402
    full_lane_interval_mask,
    full_lane_proposal_score_probability,
)
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_VAL_GT = (
    ROOT
    / "runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/"
    / "tusimple_official_val_363_folder_aware_seed20260602.json"
)
FULL_KEYS = (
    "pred_full_lane_points",
    "pred_full_lane_valid_logits",
    "pred_full_lane_exist_logits",
    "pred_full_lane_quality_logits",
    "pred_full_lane_start_logits",
    "pred_full_lane_end_logits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure full-visible-span oracle coverage for base and independent "
            "full-lane proposals on TuSimple official GT."
        )
    )
    parser.add_argument("--weights", required=True, help="Full-lane proposal checkpoint (.pt).")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--gt-json", default=None, help="TuSimple official GT json-lines file.")
    parser.add_argument(
        "--allow-noncanonical-gt",
        action="store_true",
        help="Allow noncanonical split=val GT for diagnostics; result is not comparable to canonical val.",
    )
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS input shape as H W.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=0, help="Limit GT records. 0 means all.")
    parser.add_argument(
        "--raw-file-contains",
        default=None,
        help="Optional raw_file substring filter, e.g. clips/0601/.",
    )
    parser.add_argument(
        "--point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold used for the full-visible-span mask.",
    )
    parser.add_argument(
        "--hit-px",
        type=float,
        default=20.0,
        help="APE threshold for the strict full-visible-span hit.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.0,
        help="Minimum shared proposal score for model-score top-1 selection.",
    )
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
        records = [record for record in records if needle in str(record["raw_file"]).replace("\\", "/")]
    if args.max_images and args.max_images > 0:
        records = records[: int(args.max_images)]
    return records


def _parse_date(raw_file: str) -> str:
    parts = str(raw_file).replace("\\", "/").split("/")
    return parts[1] if len(parts) >= 2 and parts[0] == "clips" else ""


def _group_names(split: str, raw_file: str, gt_count: int, gt_visible_points: int | None = None) -> list[str]:
    names = ["all", f"gt{int(gt_count)}"]
    date = _parse_date(raw_file)
    if date:
        names.append(f"date{date}_gt{int(gt_count)}")
    if str(split) == "train" and date == "0601":
        names.append(f"train0601_gt{int(gt_count)}")
    if gt_visible_points is not None:
        visible = int(gt_visible_points)
        if visible <= 3:
            visible_group = "vis_le3"
        elif visible <= 5:
            visible_group = "vis_4_5"
        elif visible <= 10:
            visible_group = "vis_6_10"
        else:
            visible_group = "vis_gt10"
        names.append(visible_group)
        names.append(f"gt{int(gt_count)}_{visible_group}")
    return names


def _safe_float(value: float) -> float | None:
    value = float(value)
    return round(value, 6) if math.isfinite(value) else None


def _candidate_x_and_coverage(
    points_norm: np.ndarray,
    valid_mask: np.ndarray,
    h_samples: list[int | float],
    image_shape: tuple[int, int],
    gt_lane: list[float],
) -> dict[str, Any]:
    """Evaluate one proposal against one GT lane at official h_samples."""
    points = np.asarray(points_norm, dtype=np.float32).reshape(-1, 2)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if points.shape[0] != valid.shape[0] or points.shape[0] < 2:
        raise ValueError(f"Proposal points/mask mismatch or too few points: {points.shape}, {valid.shape}.")

    h, w = int(image_shape[0]), int(image_shape[1])
    points_px = points * np.asarray([float(w), float(h)], dtype=np.float32).reshape(1, 2)
    order = np.argsort(points_px[:, 1], kind="stable")
    points_px = points_px[order]
    valid = valid[order]
    ys = points_px[:, 1]
    xs = points_px[:, 0]

    gt = np.asarray(gt_lane, dtype=np.float32).reshape(-1)
    sample_y = np.asarray(h_samples, dtype=np.float32).reshape(-1)
    gt_visible = gt >= 0.0
    if gt_visible.shape != sample_y.shape:
        raise ValueError(f"GT lane length {gt.shape[0]} does not match h_samples {sample_y.shape[0]}.")

    nearest = np.abs(ys[:, None] - sample_y[None, :]).argmin(axis=0)
    nearest_distance = np.abs(ys[nearest] - sample_y)
    exact_anchor = nearest_distance <= 1.5
    candidate_visible = exact_anchor & valid[nearest]
    pred_x = np.interp(sample_y, ys, xs, left=np.nan, right=np.nan)
    comparable = gt_visible & np.isfinite(pred_x)
    covered = gt_visible & candidate_visible & np.isfinite(pred_x)
    gt_visible_count = int(gt_visible.sum())
    overlap_count = int(covered.sum())
    partial_ape = (
        float(np.mean(np.abs(pred_x[covered] - gt[covered])))
        if overlap_count > 0
        else float("inf")
    )
    full_span = bool(gt_visible_count > 0 and overlap_count == gt_visible_count)
    full_ape = float(np.mean(np.abs(pred_x[gt_visible] - gt[gt_visible]))) if full_span else float("inf")
    comparable_ape = (
        float(np.mean(np.abs(pred_x[comparable] - gt[comparable])))
        if bool(comparable.any())
        else float("inf")
    )
    return {
        "full_span": full_span,
        "gt_visible_count": gt_visible_count,
        "overlap_count": overlap_count,
        "coverage": float(overlap_count / max(gt_visible_count, 1)),
        "partial_ape_px": partial_ape,
        "full_ape_px": full_ape,
        "comparable_ape_px": comparable_ape,
    }


def _best_full(
    evaluations: list[dict[str, Any]],
    indices: list[int],
) -> tuple[int, dict[str, Any]]:
    valid = [idx for idx in indices if math.isfinite(float(evaluations[idx]["full_ape_px"]))]
    if not valid:
        return -1, {
            "full_span": False,
            "coverage": 0.0,
            "overlap_count": 0,
            "full_ape_px": float("inf"),
            "partial_ape_px": float("inf"),
        }
    best = min(
        valid,
        key=lambda idx: (
            float(evaluations[idx]["full_ape_px"]),
            -int(evaluations[idx]["overlap_count"]),
            int(idx),
        ),
    )
    return int(best), evaluations[best]


def _best_partial(
    evaluations: list[dict[str, Any]],
    indices: list[int],
) -> tuple[int, dict[str, Any]]:
    valid = [idx for idx in indices if math.isfinite(float(evaluations[idx]["partial_ape_px"]))]
    if not valid:
        return -1, {
            "full_span": False,
            "coverage": 0.0,
            "overlap_count": 0,
            "full_ape_px": float("inf"),
            "partial_ape_px": float("inf"),
        }
    best = min(
        valid,
        key=lambda idx: (
            float(evaluations[idx]["partial_ape_px"]),
            -int(evaluations[idx]["overlap_count"]),
            int(idx),
        ),
    )
    return int(best), evaluations[best]


def _score_rank(scores: np.ndarray, indices: list[int], candidate_idx: int) -> int | None:
    """Return 1-based descending score rank for a candidate, with index tie-break."""
    if candidate_idx < 0 or candidate_idx not in indices:
        return None
    ordered = sorted(indices, key=lambda idx: (-float(scores[idx]), int(idx)))
    return int(ordered.index(candidate_idx) + 1)


def _init_stats() -> dict[str, Any]:
    return {
        "lanes": 0,
        "base_full_span": 0,
        "full_full_span": 0,
        "union_full_span": 0,
        "base_full_hit": 0,
        "full_full_hit": 0,
        "union_full_hit": 0,
        "model_top1_full_span": 0,
        "model_top1_full_hit": 0,
        "full_no_full_span": 0,
        "full_span_miss20": 0,
        "full_span_base_no_span": 0,
        "full_hit_base_miss": 0,
        "union_span_gain_over_base": 0,
        "union_hit_gain_over_base": 0,
        "full_oracle_score_top1_all": 0,
        "full_oracle_score_top5_all": 0,
        "full_oracle_score_top1_full": 0,
        "full_oracle_score_top3_full": 0,
        "full_oracle_score_top5_full": 0,
        "base_partial_apes": [],
        "full_partial_apes": [],
        "union_partial_apes": [],
        "union_full_apes": [],
        "model_top1_full_apes": [],
        "model_top1_scores": [],
        "full_full_apes": [],
        "full_partial_coverages": [],
        "full_oracle_score_ranks_all": [],
        "full_oracle_score_ranks_full": [],
    }


def _update_stats(
    stats: dict[str, Any],
    row: dict[str, Any],
    hit_px: float,
) -> None:
    stats["lanes"] += 1
    for name in ("base", "full", "union"):
        stats[f"{name}_full_span"] += int(bool(row[f"{name}_oracle_full_span"]))
        stats[f"{name}_full_hit"] += int(bool(row[f"{name}_oracle_full_hit"]))
        partial = row[f"{name}_oracle_partial_ape_px"]
        if partial is not None:
            stats[f"{name}_partial_apes"].append(float(partial))
    stats["model_top1_full_span"] += int(bool(row["model_top1_full_span"]))
    stats["model_top1_full_hit"] += int(bool(row["model_top1_full_hit"]))
    full_span = bool(row["full_oracle_full_span"])
    full_hit = bool(row["full_oracle_full_hit"])
    base_span = bool(row["base_oracle_full_span"])
    base_hit = bool(row["base_oracle_full_hit"])
    union_span = bool(row["union_oracle_full_span"])
    union_hit = bool(row["union_oracle_full_hit"])
    stats["full_no_full_span"] += int(not full_span)
    stats["full_span_miss20"] += int(full_span and not full_hit)
    stats["full_span_base_no_span"] += int(full_span and not base_span)
    stats["full_hit_base_miss"] += int(full_hit and not base_hit)
    stats["union_span_gain_over_base"] += int(union_span and not base_span)
    stats["union_hit_gain_over_base"] += int(union_hit and not base_hit)
    rank_all = row.get("full_oracle_score_rank_all")
    rank_full = row.get("full_oracle_score_rank_full")
    if rank_all is not None:
        rank_all = int(rank_all)
        stats["full_oracle_score_ranks_all"].append(rank_all)
        stats["full_oracle_score_top1_all"] += int(rank_all <= 1)
        stats["full_oracle_score_top5_all"] += int(rank_all <= 5)
    if rank_full is not None:
        rank_full = int(rank_full)
        stats["full_oracle_score_ranks_full"].append(rank_full)
        stats["full_oracle_score_top1_full"] += int(rank_full <= 1)
        stats["full_oracle_score_top3_full"] += int(rank_full <= 3)
        stats["full_oracle_score_top5_full"] += int(rank_full <= 5)
    for key in ("union_oracle_full_ape_px", "model_top1_full_ape_px", "model_top1_score"):
        value = row[key]
        if value is not None:
            stats[
                "union_full_apes"
                if key == "union_oracle_full_ape_px"
                else "model_top1_full_apes"
                if key == "model_top1_full_ape_px"
                else "model_top1_scores"
            ].append(float(value))
    full_ape = row.get("full_oracle_full_ape_px")
    if full_ape is not None:
        stats["full_full_apes"].append(float(full_ape))
    coverage = row.get("full_oracle_partial_coverage")
    if coverage is not None:
        stats["full_partial_coverages"].append(float(coverage))


def _summary_stats(stats: dict[str, Any], hit_px: float) -> dict[str, Any]:
    lanes = max(int(stats["lanes"]), 1)

    def rate(name: str) -> float:
        return round(float(stats[name]) / lanes, 6)

    def mean(name: str) -> float | None:
        values = [float(x) for x in stats[name] if math.isfinite(float(x))]
        return round(float(np.mean(values)), 6) if values else None

    def median(name: str) -> float | None:
        values = [float(x) for x in stats[name] if math.isfinite(float(x))]
        return round(float(np.median(values)), 6) if values else None

    return {
        "lanes": int(stats["lanes"]),
        "base_oracle_full_span": int(stats["base_full_span"]),
        "base_oracle_full_span_rate": rate("base_full_span"),
        "full_proposal_oracle_full_span": int(stats["full_full_span"]),
        "full_proposal_oracle_full_span_rate": rate("full_full_span"),
        "union_oracle_full_span": int(stats["union_full_span"]),
        "union_oracle_full_span_rate": rate("union_full_span"),
        "base_oracle_full_hit": int(stats["base_full_hit"]),
        "full_proposal_oracle_full_hit": int(stats["full_full_hit"]),
        "union_oracle_full_hit": int(stats["union_full_hit"]),
        "base_oracle_full_hit_rate": rate("base_full_hit"),
        "full_proposal_oracle_full_hit_rate": rate("full_full_hit"),
        "union_oracle_full_hit_rate": rate("union_full_hit"),
        "model_top1_full_span": int(stats["model_top1_full_span"]),
        "model_top1_full_span_rate": rate("model_top1_full_span"),
        "model_top1_full_hit": int(stats["model_top1_full_hit"]),
        "model_top1_full_hit_rate": rate("model_top1_full_hit"),
        "full_proposal_no_full_span": int(stats["full_no_full_span"]),
        "full_proposal_no_full_span_rate": rate("full_no_full_span"),
        "full_proposal_full_span_miss20": int(stats["full_span_miss20"]),
        "full_proposal_full_span_miss20_rate": rate("full_span_miss20"),
        "full_proposal_full_span_base_no_span": int(stats["full_span_base_no_span"]),
        "full_proposal_full_span_base_no_span_rate": rate("full_span_base_no_span"),
        "full_proposal_full_hit_base_miss": int(stats["full_hit_base_miss"]),
        "full_proposal_full_hit_base_miss_rate": rate("full_hit_base_miss"),
        "union_oracle_full_span_gain_over_base": int(stats["union_span_gain_over_base"]),
        "union_oracle_full_span_gain_over_base_rate": rate("union_span_gain_over_base"),
        "union_oracle_full_hit_gain_over_base": int(stats["union_hit_gain_over_base"]),
        "union_oracle_full_hit_gain_over_base_rate": rate("union_hit_gain_over_base"),
        "full_oracle_score_top1_all": int(stats["full_oracle_score_top1_all"]),
        "full_oracle_score_top1_all_rate": rate("full_oracle_score_top1_all"),
        "full_oracle_score_top5_all": int(stats["full_oracle_score_top5_all"]),
        "full_oracle_score_top5_all_rate": rate("full_oracle_score_top5_all"),
        "full_oracle_score_top1_full": int(stats["full_oracle_score_top1_full"]),
        "full_oracle_score_top1_full_rate": rate("full_oracle_score_top1_full"),
        "full_oracle_score_top3_full": int(stats["full_oracle_score_top3_full"]),
        "full_oracle_score_top3_full_rate": rate("full_oracle_score_top3_full"),
        "full_oracle_score_top5_full": int(stats["full_oracle_score_top5_full"]),
        "full_oracle_score_top5_full_rate": rate("full_oracle_score_top5_full"),
        "full_oracle_score_rank_all_mean": mean("full_oracle_score_ranks_all"),
        "full_oracle_score_rank_all_median": median("full_oracle_score_ranks_all"),
        "full_oracle_score_rank_full_mean": mean("full_oracle_score_ranks_full"),
        "full_oracle_score_rank_full_median": median("full_oracle_score_ranks_full"),
        "base_oracle_partial_ape_mean_px": mean("base_partial_apes"),
        "full_proposal_oracle_partial_ape_mean_px": mean("full_partial_apes"),
        "full_proposal_oracle_full_ape_mean_px": mean("full_full_apes"),
        "full_proposal_oracle_partial_coverage_mean": mean("full_partial_coverages"),
        "union_oracle_partial_ape_mean_px": mean("union_partial_apes"),
        "union_oracle_full_ape_mean_px": mean("union_full_apes"),
        "model_top1_full_ape_mean_px": mean("model_top1_full_apes"),
        "model_top1_score_mean": mean("model_top1_scores"),
        "full_span_definition": "all visible GT h_samples covered by proposal valid mask",
        "full_hit_definition": f"full span plus full APE <= {float(hit_px):g}px",
        "partial_overlap_is_auxiliary_only": True,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    stem = Path(args.weights).stem
    split_tag = str(args.split)
    if args.raw_file_contains:
        safe = str(args.raw_file_contains).replace("\\", "/").strip("/").replace("/", "_")
        split_tag += f"_{safe}"
    return ROOT / "runs" / "gcs_lane" / "full_lane_oracle" / stem / split_tag


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.point_valid_thr) <= 1.0:
        raise ValueError(f"--point-valid-thr must be in [0, 1], got {args.point_valid_thr}.")
    if float(args.hit_px) < 0.0:
        raise ValueError(f"--hit-px must be >= 0, got {args.hit_px}.")
    if float(args.score_thr) < 0.0:
        raise ValueError(f"--score-thr must be >= 0, got {args.score_thr}.")

    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_json = _default_gt_json(args)
    if gt_json is None:
        gt_path = resolve_tusimple_gt_json(archive_root, args.split, None)
    else:
        gt_path = Path(gt_json)
    gt_records = read_tusimple_json_lines(gt_path)
    gt_contract = official_gt_contract_summary(
        split=args.split,
        gt_json=gt_path,
        gt_records=gt_records,
        allow_noncanonical_gt=bool(args.allow_noncanonical_gt),
    )
    gt_records = _limit_and_filter_records(gt_records, args)
    imgsz = normalize_imgsz(args.imgsz if args.imgsz is not None else (544, 960))
    device = select_device(args.device)
    model = load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    model.eval()

    if args.warmup > 0 and gt_records:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device, half=bool(args.half))
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device)

    rows: list[dict[str, Any]] = []
    stats = defaultdict(_init_stats)
    infer_time_s = 0.0
    for record in gt_records:
        raw_file = str(record["raw_file"])
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_shape = (int(image.shape[0]), int(image.shape[1]))
        validate_tusimple_h_samples_asc(record["h_samples"], name=f"{raw_file} h_samples")

        tensor = preprocess_image(image, imgsz, device=device, half=bool(args.half))
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device)
        infer_time_s += time.perf_counter() - t0
        if not isinstance(preds, dict):
            raise ValueError("Full-lane oracle expects dictionary model outputs.")
        missing = [key for key in ("pred_points", "pred_logits", "pred_valid_logits", *FULL_KEYS) if key not in preds]
        if missing:
            raise KeyError(f"Full-lane checkpoint did not emit required outputs: {missing}")

        base_points = preds["pred_points"][0].detach().float().cpu()
        base_valid_logits = preds["pred_valid_logits"][0].detach().float().cpu()
        base_valid_mask = base_valid_logits.sigmoid() >= float(args.point_valid_thr)
        base_scores = full_lane_proposal_score_probability(
            preds["pred_logits"][0].detach().float().cpu(),
            quality_logits=None,
            valid_logits=base_valid_logits,
        ).cpu()

        full_points = preds["pred_full_lane_points"][0].detach().float().cpu()
        full_valid_logits = preds["pred_full_lane_valid_logits"][0].detach().float().cpu()
        full_exist_logits = preds["pred_full_lane_exist_logits"][0].detach().float().cpu()
        full_quality_logits = preds["pred_full_lane_quality_logits"][0].detach().float().cpu()
        full_start_logits = preds["pred_full_lane_start_logits"][0].detach().float().cpu()
        full_end_logits = preds["pred_full_lane_end_logits"][0].detach().float().cpu()
        full_interval = full_lane_interval_mask(full_start_logits, full_end_logits)
        full_valid_mask = (full_valid_logits.sigmoid() >= float(args.point_valid_thr)) & full_interval
        full_scores = full_lane_proposal_score_probability(
            full_exist_logits,
            quality_logits=full_quality_logits,
            valid_logits=full_valid_logits,
            start_logits=full_start_logits,
            end_logits=full_end_logits,
        ).cpu()

        all_points = torch.cat((base_points, full_points), dim=0).numpy()
        all_masks = torch.cat((base_valid_mask, full_valid_mask), dim=0).numpy()
        all_scores = torch.cat((base_scores, full_scores), dim=0).numpy()
        base_count = int(base_points.shape[0])
        full_count = int(full_points.shape[0])
        gt_lanes = valid_tusimple_lanes(record.get("lanes", []))
        for gt_lane_id, gt_lane in enumerate(gt_lanes):
            evaluations = [
                _candidate_x_and_coverage(points, mask, record["h_samples"], image_shape, gt_lane)
                for points, mask in zip(all_points, all_masks)
            ]
            base_indices = list(range(base_count))
            full_indices = list(range(base_count, base_count + full_count))
            union_indices = list(range(base_count + full_count))
            base_oracle_idx, base_oracle = _best_full(evaluations, base_indices)
            full_oracle_idx, full_oracle = _best_full(evaluations, full_indices)
            union_oracle_idx, union_oracle = _best_full(evaluations, union_indices)
            base_partial_idx, base_partial = _best_partial(evaluations, base_indices)
            full_partial_idx, full_partial = _best_partial(evaluations, full_indices)
            union_partial_idx, union_partial = _best_partial(evaluations, union_indices)
            full_score_rank_all = _score_rank(all_scores, union_indices, full_oracle_idx)
            full_score_rank_full = _score_rank(all_scores, full_indices, full_oracle_idx)
            model_top1_idx = int(np.argmax(all_scores)) if all_scores.size else -1
            model_top1 = evaluations[model_top1_idx] if model_top1_idx >= 0 else {
                "full_span": False,
                "overlap_count": 0,
                "coverage": 0.0,
                "full_ape_px": float("inf"),
                "partial_ape_px": float("inf"),
            }
            if model_top1_idx >= 0 and float(all_scores[model_top1_idx]) < float(args.score_thr):
                model_top1_idx = -1
                model_top1 = {
                    "full_span": False,
                    "overlap_count": 0,
                    "coverage": 0.0,
                    "full_ape_px": float("inf"),
                    "partial_ape_px": float("inf"),
                }

            def source(idx: int) -> str:
                if idx < 0:
                    return ""
                return "base_query" if idx < base_count else "full_lane_proposal"

            group_names = _group_names(
                args.split,
                raw_file,
                len(gt_lanes),
                int(sum(float(x) >= 0.0 for x in gt_lane)),
            )
            row = {
                "split": str(args.split),
                "raw_file": raw_file,
                "date": _parse_date(raw_file),
                "gt_count": int(len(gt_lanes)),
                "gt_lane_id": int(gt_lane_id),
                "gt_visible_points": int(sum(float(x) >= 0.0 for x in gt_lane)),
                "base_oracle_source": source(base_oracle_idx),
                "base_oracle_index": int(base_oracle_idx),
                "base_oracle_full_span": bool(base_oracle.get("full_span", False)),
                "base_oracle_full_hit": bool(
                    base_oracle.get("full_span", False)
                    and float(base_oracle.get("full_ape_px", float("inf"))) <= float(args.hit_px)
                ),
                "base_oracle_full_ape_px": _safe_float(base_oracle.get("full_ape_px", float("inf"))),
                "base_oracle_partial_ape_px": _safe_float(base_partial.get("partial_ape_px", float("inf"))),
                "base_oracle_partial_overlap": int(base_partial.get("overlap_count", 0)),
                "full_oracle_source": source(full_oracle_idx),
                "full_oracle_index": int(full_oracle_idx - base_count) if full_oracle_idx >= base_count else -1,
                "full_oracle_full_span": bool(full_oracle.get("full_span", False)),
                "full_oracle_full_hit": bool(
                    full_oracle.get("full_span", False)
                    and float(full_oracle.get("full_ape_px", float("inf"))) <= float(args.hit_px)
                ),
                "full_oracle_full_ape_px": _safe_float(full_oracle.get("full_ape_px", float("inf"))),
                "full_oracle_partial_ape_px": _safe_float(full_partial.get("partial_ape_px", float("inf"))),
                "full_oracle_partial_overlap": int(full_partial.get("overlap_count", 0)),
                "full_oracle_partial_coverage": _safe_float(full_partial.get("coverage", 0.0)),
                "full_oracle_score_rank_all": full_score_rank_all,
                "full_oracle_score_rank_full": full_score_rank_full,
                "union_oracle_source": source(union_oracle_idx),
                "union_oracle_index": int(union_oracle_idx),
                "union_oracle_full_span": bool(union_oracle.get("full_span", False)),
                "union_oracle_full_hit": bool(
                    union_oracle.get("full_span", False)
                    and float(union_oracle.get("full_ape_px", float("inf"))) <= float(args.hit_px)
                ),
                "union_oracle_full_ape_px": _safe_float(union_oracle.get("full_ape_px", float("inf"))),
                "union_oracle_partial_ape_px": _safe_float(union_partial.get("partial_ape_px", float("inf"))),
                "union_oracle_partial_overlap": int(union_partial.get("overlap_count", 0)),
                "model_top1_source": source(model_top1_idx),
                "model_top1_index": int(model_top1_idx - base_count) if model_top1_idx >= base_count else int(model_top1_idx),
                "model_top1_score": _safe_float(all_scores[model_top1_idx]) if model_top1_idx >= 0 else None,
                "model_top1_full_span": bool(model_top1.get("full_span", False)),
                "model_top1_full_hit": bool(
                    model_top1.get("full_span", False)
                    and float(model_top1.get("full_ape_px", float("inf"))) <= float(args.hit_px)
                ),
                "model_top1_full_ape_px": _safe_float(model_top1.get("full_ape_px", float("inf"))),
                "model_top1_partial_ape_px": _safe_float(model_top1.get("partial_ape_px", float("inf"))),
                "model_top1_partial_overlap": int(model_top1.get("overlap_count", 0)),
                "groups": ";".join(group_names),
            }
            rows.append(row)
            for group in group_names:
                _update_stats(stats[group], row, float(args.hit_px))

    save_dir = _resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = save_dir / "full_lane_oracle_rows.csv"
    summary_json = save_dir / "full_lane_oracle_summary.json"
    _write_csv(rows_csv, rows)
    summary = {
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "archive_root": str(archive_root.resolve()),
            "split": str(args.split),
            "gt_json": str(gt_path.resolve()),
            "raw_file_contains": args.raw_file_contains,
            "records": int(len(gt_records)),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "point_valid_thr": float(args.point_valid_thr),
            "hit_px": float(args.hit_px),
            "score_thr": float(args.score_thr),
            "device": str(args.device),
            "half": bool(args.half),
            "uses_gt_for_inference": False,
            "diagnostic_only": True,
            "test_closed": True,
            "partial_overlap_is_auxiliary_only": True,
        },
        "groups": {name: _summary_stats(value, float(args.hit_px)) for name, value in sorted(stats.items())},
        "outputs": {
            "rows_csv": str(rows_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
        },
        "timing": {
            "avg_inference_ms": round(infer_time_s * 1000.0 / max(len(gt_records), 1), 4),
        },
        **gt_contract,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"groups": summary["groups"], "test_closed": True}, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
