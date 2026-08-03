"""Diagnose dense lane evidence on complete visible GT spans.

This tool is prediction-only and diagnostic-only.  Ground truth is used only
to score the dense evidence after inference; it is never passed into model
forward or decode.  It does not produce official TuSimple predictions and
must not be used for TEST selection.
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
import torch.nn.functional as F

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
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_VAL_GT = (
    ROOT
    / "runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/"
    / "tusimple_official_val_363_folder_aware_seed20260602.json"
)


def parse_args() -> argparse.Namespace:
    """Parse dense evidence diagnostic arguments."""
    parser = argparse.ArgumentParser(
        description="Measure dense centerline/endpoint/instance evidence on complete TuSimple GT spans."
    )
    parser.add_argument("--weights", required=True, help="Dense-instance checkpoint (.pt).")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--gt-json", default=None, help="TuSimple official GT json-lines file.")
    parser.add_argument(
        "--allow-noncanonical-gt",
        action="store_true",
        help="Allow noncanonical GT for diagnostics; result is not comparable to canonical official-val evidence.",
    )
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS input shape as H W.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=0, help="Limit GT records. 0 means all.")
    parser.add_argument("--raw-file-contains", default=None, help="Optional raw_file substring filter.")
    parser.add_argument("--centerline-thr", type=float, default=0.5)
    parser.add_argument("--endpoint-thr", type=float, default=0.5)
    parser.add_argument("--support-radius-px", type=int, default=4)
    parser.add_argument("--point-valid-thr", type=float, default=0.5)
    parser.add_argument("--base-hit-px", type=float, default=20.0)
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


def _safe_float(value: float) -> float | None:
    value = float(value)
    return round(value, 6) if math.isfinite(value) else None


def _group_names(split: str, raw_file: str, gt_count: int, visible_count: int, base_full_hit: bool) -> list[str]:
    """Return stable per-lane diagnostic groups."""
    if visible_count <= 3:
        visible_group = "vis_le3"
    elif visible_count <= 5:
        visible_group = "vis_4_5"
    elif visible_count <= 10:
        visible_group = "vis_6_10"
    else:
        visible_group = "vis_gt10"
    names = ["all", f"gt{int(gt_count)}", visible_group, f"gt{int(gt_count)}_{visible_group}"]
    date = _parse_date(raw_file)
    if date:
        names.append(f"date{date}_gt{int(gt_count)}")
    if str(split) == "train" and date == "0601":
        names.append(f"train0601_gt{int(gt_count)}")
    if not base_full_hit:
        names.append("base_miss")
        names.append(f"gt{int(gt_count)}_base_miss")
        names.append(f"gt{int(gt_count)}_{visible_group}_base_miss")
    return names


def _init_stats() -> dict[str, Any]:
    return {
        "lanes": 0,
        "dense_centerline_full_span": 0,
        "dense_endpoint_full_support": 0,
        "dense_full_evidence": 0,
        "base_full_hit": 0,
        "dense_full_evidence_on_base_miss": 0,
        "dense_full_span_on_base_miss": 0,
        "centerline_support_ratios": [],
        "endpoint_support_ratios": [],
        "embed_pull_losses": [],
        "embed_pair_distances": [],
        "embed_margin_violations": [],
    }


def _update_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["lanes"] += 1
    stats["dense_centerline_full_span"] += int(bool(row["dense_centerline_full_span"]))
    stats["dense_endpoint_full_support"] += int(bool(row["dense_endpoint_full_support"]))
    stats["dense_full_evidence"] += int(bool(row["dense_full_evidence"]))
    stats["base_full_hit"] += int(bool(row["base_full_hit"]))
    if not row["base_full_hit"]:
        stats["dense_full_evidence_on_base_miss"] += int(bool(row["dense_full_evidence"]))
        stats["dense_full_span_on_base_miss"] += int(bool(row["dense_centerline_full_span"]))
    stats["centerline_support_ratios"].append(float(row["centerline_support_ratio"]))
    stats["endpoint_support_ratios"].append(float(row["endpoint_support_ratio"]))
    if row["embed_pull_loss"] is not None:
        stats["embed_pull_losses"].append(float(row["embed_pull_loss"]))
    if row["embed_pair_distance"] is not None:
        stats["embed_pair_distances"].append(float(row["embed_pair_distance"]))
    if row["embed_margin_violation"] is not None:
        stats["embed_margin_violations"].append(float(row["embed_margin_violation"]))


def _mean(values: list[float]) -> float | None:
    return _safe_float(sum(values) / len(values)) if values else None


def _summary_stats(stats: dict[str, Any]) -> dict[str, Any]:
    lanes = max(int(stats["lanes"]), 1)
    return {
        "lanes": int(stats["lanes"]),
        "dense_centerline_full_span": int(stats["dense_centerline_full_span"]),
        "dense_centerline_full_span_rate": round(float(stats["dense_centerline_full_span"]) / lanes, 6),
        "dense_endpoint_full_support": int(stats["dense_endpoint_full_support"]),
        "dense_endpoint_full_support_rate": round(float(stats["dense_endpoint_full_support"]) / lanes, 6),
        "dense_full_evidence": int(stats["dense_full_evidence"]),
        "dense_full_evidence_rate": round(float(stats["dense_full_evidence"]) / lanes, 6),
        "base_full_hit": int(stats["base_full_hit"]),
        "base_full_hit_rate": round(float(stats["base_full_hit"]) / lanes, 6),
        "dense_full_span_on_base_miss": int(stats["dense_full_span_on_base_miss"]),
        "dense_full_evidence_on_base_miss": int(stats["dense_full_evidence_on_base_miss"]),
        "centerline_support_ratio_mean": _mean(stats["centerline_support_ratios"]),
        "endpoint_support_ratio_mean": _mean(stats["endpoint_support_ratios"]),
        "embed_pull_loss_mean": _mean(stats["embed_pull_losses"]),
        "embed_pair_distance_mean": _mean(stats["embed_pair_distances"]),
        "embed_margin_violation_mean": _mean(stats["embed_margin_violations"]),
    }


def _model_coords(
    h_samples: list[int | float],
    gt_lane: list[float],
    image_shape: tuple[int, int],
    imgsz: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map visible original-image GT points into resized model pixel coordinates."""
    original_h, original_w = int(image_shape[0]), int(image_shape[1])
    model_h, model_w = int(imgsz[0]), int(imgsz[1])
    h_values = np.asarray(h_samples, dtype=np.float32).reshape(-1)
    x_values = np.asarray(gt_lane, dtype=np.float32).reshape(-1)
    if h_values.shape != x_values.shape:
        raise ValueError(f"GT h_samples/lane shape mismatch: {h_values.shape} vs {x_values.shape}.")
    visible = x_values >= 0.0
    # GCS labels use x / W and y / H normalization. Keep the diagnostic in
    # that same coordinate system as the dense loss and official decoder.
    x_model = x_values[visible] / float(max(original_w, 1)) * float(model_w)
    y_model = h_values[visible] / float(max(original_h, 1)) * float(model_h)
    return np.stack((x_model, y_model), axis=1), np.flatnonzero(visible)


def _support_values(prob: torch.Tensor, coords: np.ndarray, radius_px: int) -> np.ndarray:
    """Sample max dense probability in a square neighborhood around GT points."""
    if coords.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"Dense probability map must be H x W, got {tuple(prob.shape)}.")
    radius = max(int(radius_px), 0)
    source = prob.float().unsqueeze(0).unsqueeze(0)
    if radius:
        source = F.max_pool2d(source, kernel_size=2 * radius + 1, stride=1, padding=radius)
    height, width = int(source.shape[-2]), int(source.shape[-1])
    xy = np.rint(coords).astype(np.int64)
    x = np.clip(xy[:, 0], 0, width - 1)
    y = np.clip(xy[:, 1], 0, height - 1)
    return source[0, 0, torch.from_numpy(y), torch.from_numpy(x)].cpu().numpy().astype(np.float32)


def _base_lane_quality(
    points: torch.Tensor,
    valid_logits: torch.Tensor,
    h_samples: list[int | float],
    gt_lane: list[float],
    image_shape: tuple[int, int],
    point_valid_thr: float,
    hit_px: float,
) -> tuple[bool, float]:
    """Return best base-query full-span hit and its visible-point hit ratio."""
    original_h, original_w = int(image_shape[0]), int(image_shape[1])
    gt_x = np.asarray(gt_lane, dtype=np.float32)
    gt_h = np.asarray(h_samples, dtype=np.float32)
    visible = gt_x >= 0.0
    if not visible.any():
        return False, 0.0
    gt_x = gt_x[visible]
    gt_y_px = gt_h[visible]
    points = points.detach().float().cpu()
    valid = valid_logits.detach().float().sigmoid().cpu()
    best_ratio = 0.0
    best_full = False
    for query_points, query_valid in zip(points, valid):
        pred_y_px = query_points[:, 1].numpy() * float(max(original_h, 1))
        pred_x = query_points[:, 0].numpy()
        nearest = np.abs(pred_y_px.reshape(1, -1) - gt_y_px.reshape(-1, 1)).argmin(axis=1)
        errors = np.abs(pred_x[nearest] * float(max(original_w, 1)) - gt_x)
        hit = (query_valid.numpy()[nearest] >= float(point_valid_thr)) & (errors <= float(hit_px))
        ratio = float(hit.mean()) if hit.size else 0.0
        best_ratio = max(best_ratio, ratio)
        best_full = best_full or bool(hit.all())
    return best_full, best_ratio


def _embedding_stats(
    embedding: torch.Tensor,
    coords: np.ndarray,
    lane_ids: np.ndarray,
    lane_count: int,
    imgsz: tuple[int, int],
) -> tuple[float | None, float | None, float | None]:
    """Measure same-lane pull and different-lane separation at GT coordinates."""
    if coords.size == 0 or lane_count <= 0:
        return None, None, None
    model_h, model_w = int(imgsz[0]), int(imgsz[1])
    norm_x = coords[:, 0] / float(max(model_w, 1))
    norm_y = coords[:, 1] / float(max(model_h, 1))
    grid = torch.from_numpy(np.stack((norm_x * 2.0 - 1.0, norm_y * 2.0 - 1.0), axis=1)).float()
    sampled = F.grid_sample(
        F.normalize(embedding.float(), dim=0, eps=1.0e-6).unsqueeze(0),
        grid.view(1, -1, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, :, :, 0].transpose(0, 1)
    sampled = F.normalize(sampled, dim=1, eps=1.0e-6)

    means: list[torch.Tensor] = []
    pull_losses: list[torch.Tensor] = []
    lane_ids = np.asarray(lane_ids, dtype=np.int64)
    for lane_id in range(int(lane_count)):
        mask = torch.from_numpy(lane_ids == lane_id)
        if int(mask.sum()) < 2:
            continue
        lane_embed = sampled[mask]
        mean_embed = F.normalize(lane_embed.mean(dim=0, keepdim=True), dim=1, eps=1.0e-6)[0]
        means.append(mean_embed)
        pull_losses.append((1.0 - (lane_embed * mean_embed.view(1, -1)).sum(dim=1)).mean())
    pull = float(torch.stack(pull_losses).mean().item()) if pull_losses else None
    if len(means) < 2:
        return pull, None, None

    pair_distances: list[torch.Tensor] = []
    violations: list[torch.Tensor] = []
    for left in range(len(means) - 1):
        distances = torch.linalg.vector_norm(torch.stack(means[left + 1 :]) - means[left], dim=1)
        pair_distances.append(distances)
        violations.append((distances < 0.5).float())
    distance = float(torch.cat(pair_distances).mean().item())
    violation = float(torch.cat(violations).mean().item())
    return pull, distance, violation


def _resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    stem = Path(args.weights).stem
    split_tag = str(args.split)
    if args.raw_file_contains:
        safe = str(args.raw_file_contains).replace("\\", "/").strip("/").replace("/", "_")
        split_tag += f"_{safe}"
    return ROOT / "runs" / "gcs_lane" / "dense_instance_oracle" / stem / split_tag


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.centerline_thr) <= 1.0:
        raise ValueError("--centerline-thr must be in [0, 1].")
    if not 0.0 <= float(args.endpoint_thr) <= 1.0:
        raise ValueError("--endpoint-thr must be in [0, 1].")
    if not 0.0 <= float(args.point_valid_thr) <= 1.0:
        raise ValueError("--point-valid-thr must be in [0, 1].")
    if int(args.support_radius_px) < 0:
        raise ValueError("--support-radius-px must be >= 0.")
    if float(args.base_hit_px) < 0.0:
        raise ValueError("--base-hit-px must be >= 0.")

    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_json = _default_gt_json(args)
    gt_path = resolve_tusimple_gt_json(archive_root, args.split, None) if gt_json is None else Path(gt_json)
    records = read_tusimple_json_lines(gt_path)
    gt_contract = official_gt_contract_summary(
        split=args.split,
        gt_json=gt_path,
        gt_records=records,
        allow_noncanonical_gt=bool(args.allow_noncanonical_gt),
    )
    records = _limit_and_filter_records(records, args)
    imgsz = normalize_imgsz(args.imgsz if args.imgsz is not None else (544, 960))
    device = select_device(args.device)
    model = load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    model.eval()

    if args.warmup > 0 and records:
        warm_path = tusimple_image_path(archive_root, str(records[0]["raw_file"]), split=args.split)
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
    for record in records:
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
            raise ValueError("Dense evidence diagnostic expects dictionary model outputs.")
        required = (
            "pred_points",
            "pred_valid_logits",
            "pred_dense_centerline_logits",
            "pred_dense_endpoint_logits",
            "pred_dense_instance_embed",
        )
        missing = [key for key in required if key not in preds]
        if missing:
            raise KeyError(f"Dense checkpoint did not emit required outputs: {missing}")

        centerline = preds["pred_dense_centerline_logits"][0].float().sigmoid()
        endpoint = preds["pred_dense_endpoint_logits"][0].float().sigmoid()
        embedding = preds["pred_dense_instance_embed"][0].float().detach().cpu()
        centerline = (
            F.interpolate(centerline.unsqueeze(0), size=imgsz, mode="bilinear", align_corners=False)[0, 0]
            .detach()
            .cpu()
        )
        endpoint = (
            F.interpolate(endpoint.unsqueeze(0), size=imgsz, mode="bilinear", align_corners=False)[0]
            .detach()
            .cpu()
        )
        gt_lanes = valid_tusimple_lanes(record.get("lanes", []))
        base_points = preds["pred_points"][0].detach().float().cpu()
        base_valid_logits = preds["pred_valid_logits"][0].detach().float().cpu()

        lane_coords: list[np.ndarray] = []
        lane_ids: list[np.ndarray] = []
        for lane_id, gt_lane in enumerate(gt_lanes):
            coords, _ = _model_coords(record["h_samples"], gt_lane, image_shape, imgsz)
            lane_coords.append(coords)
            lane_ids.append(np.full((coords.shape[0],), lane_id, dtype=np.int64))
        all_coords = np.concatenate(lane_coords, axis=0) if lane_coords else np.zeros((0, 2), dtype=np.float32)
        all_lane_ids = np.concatenate(lane_ids, axis=0) if lane_ids else np.zeros((0,), dtype=np.int64)
        pull, pair_distance, margin_violation = _embedding_stats(
            embedding,
            all_coords,
            all_lane_ids,
            len(gt_lanes),
            imgsz,
        )

        for gt_lane_id, gt_lane in enumerate(gt_lanes):
            coords, _ = _model_coords(record["h_samples"], gt_lane, image_shape, imgsz)
            visible_count = int(coords.shape[0])
            center_values = _support_values(centerline, coords, int(args.support_radius_px))
            endpoint_coords = coords[[0, -1]] if visible_count else coords
            endpoint_values = np.zeros((0,), dtype=np.float32)
            if visible_count:
                endpoint_start = _support_values(endpoint[0], endpoint_coords[[0]], int(args.support_radius_px))
                endpoint_end = _support_values(endpoint[1], endpoint_coords[[-1]], int(args.support_radius_px))
                endpoint_values = np.concatenate((endpoint_start, endpoint_end))
            center_ratio = float((center_values >= float(args.centerline_thr)).mean()) if visible_count else 0.0
            endpoint_ratio = float((endpoint_values >= float(args.endpoint_thr)).mean()) if endpoint_values.size else 0.0
            dense_centerline_full_span = bool(visible_count > 0 and center_ratio >= 1.0)
            dense_endpoint_full_support = bool(endpoint_values.size == 2 and endpoint_ratio >= 1.0)
            dense_full_evidence = dense_centerline_full_span and dense_endpoint_full_support
            base_full_hit, base_hit_ratio = _base_lane_quality(
                base_points,
                base_valid_logits,
                record["h_samples"],
                gt_lane,
                image_shape,
                point_valid_thr=float(args.point_valid_thr),
                hit_px=float(args.base_hit_px),
            )
            group_names = _group_names(
                args.split,
                raw_file,
                len(gt_lanes),
                visible_count,
                base_full_hit,
            )
            row = {
                "split": str(args.split),
                "raw_file": raw_file,
                "date": _parse_date(raw_file),
                "gt_count": int(len(gt_lanes)),
                "gt_lane_id": int(gt_lane_id),
                "gt_visible_points": visible_count,
                "base_full_hit": bool(base_full_hit),
                "base_hit_ratio": round(base_hit_ratio, 6),
                "dense_centerline_full_span": dense_centerline_full_span,
                "dense_endpoint_full_support": dense_endpoint_full_support,
                "dense_full_evidence": dense_full_evidence,
                "centerline_support_ratio": round(center_ratio, 6),
                "endpoint_support_ratio": round(endpoint_ratio, 6),
                "centerline_support_min": _safe_float(float(center_values.min())) if center_values.size else None,
                "endpoint_support_min": _safe_float(float(endpoint_values.min())) if endpoint_values.size else None,
                "embed_pull_loss": pull,
                "embed_pair_distance": pair_distance,
                "embed_margin_violation": margin_violation,
                "groups": ";".join(group_names),
            }
            rows.append(row)
            for group in group_names:
                _update_stats(stats[group], row)

    save_dir = _resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = save_dir / "dense_instance_oracle_rows.csv"
    summary_json = save_dir / "dense_instance_oracle_summary.json"
    _write_csv(rows_csv, rows)
    summary = {
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "archive_root": str(archive_root.resolve()),
            "split": str(args.split),
            "gt_json": str(gt_path.resolve()),
            "raw_file_contains": args.raw_file_contains,
            "records": int(len(records)),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "centerline_thr": float(args.centerline_thr),
            "endpoint_thr": float(args.endpoint_thr),
            "support_radius_px": int(args.support_radius_px),
            "point_valid_thr": float(args.point_valid_thr),
            "base_hit_px": float(args.base_hit_px),
            "device": str(args.device),
            "half": bool(args.half),
            "uses_gt_for_inference": False,
            "diagnostic_only": True,
            "test_closed": True,
            "dense_full_evidence_definition": "all visible GT points have centerline support and both endpoints have endpoint support",
        },
        "groups": {name: _summary_stats(value) for name, value in sorted(stats.items())},
        "outputs": {
            "rows_csv": str(rows_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
        },
        "timing": {
            "avg_inference_ms": round(infer_time_s * 1000.0 / max(len(records), 1), 4),
        },
        **gt_contract,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"groups": summary["groups"], "diagnostic_only": True, "test_closed": True}, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
