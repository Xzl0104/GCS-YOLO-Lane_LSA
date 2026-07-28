"""Measure raw lateral-candidate geometry coverage for short GT4/GT5 lanes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.infer_gcs import load_gcs_model  # noqa: E402
from ultralytics.data.dataset_gcs import GCSLaneDataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose raw GCS short lateral-candidate coverage.")
    parser.add_argument("--weights", required=True, help="Checkpoint or candidate-head YAML.")
    parser.add_argument("--data", default="data/tusimple_gcs_fixed_y_960x544.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS input shape as H W.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--visible-thr", type=int, default=10)
    parser.add_argument("--min-visible", type=int, default=2)
    parser.add_argument("--hit20-px", type=float, default=20.0)
    parser.add_argument("--hit40-px", type=float, default=40.0)
    parser.add_argument("--save-json", default=None)
    return parser.parse_args()


def _empty_stats() -> dict[str, Any]:
    return {"total": 0, "hit20": 0, "hit40": 0, "best_apes": []}


def _update(stats: dict[str, Any], best_ape: float, hit20_px: float, hit40_px: float) -> None:
    stats["total"] += 1
    stats["hit20"] += int(best_ape <= float(hit20_px))
    stats["hit40"] += int(best_ape <= float(hit40_px))
    stats["best_apes"].append(float(best_ape))


def _summarize(stats: dict[str, Any]) -> dict[str, Any]:
    apes = np.asarray(stats["best_apes"], dtype=np.float32)
    total = int(stats["total"])
    return {
        "total": total,
        "hit20": int(stats["hit20"]),
        "hit40": int(stats["hit40"]),
        "hit20_rate": round(float(stats["hit20"]) / max(total, 1), 6),
        "hit40_rate": round(float(stats["hit40"]) / max(total, 1), 6),
        "best_ape_mean_px": round(float(apes.mean()), 4) if apes.size else None,
        "best_ape_median_px": round(float(np.median(apes)), 4) if apes.size else None,
        "best_ape_p90_px": round(float(np.percentile(apes, 90)), 4) if apes.size else None,
    }


def _candidate_best_ape(candidate_points: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, scale: torch.Tensor) -> float:
    flat = candidate_points.reshape(-1, candidate_points.shape[-2], 2)
    valid = valid.clamp(0.0, 1.0)
    valid_count = valid.sum().clamp_min(1.0)
    point_error = torch.norm((flat - target.view(1, target.shape[0], 2)) * scale.view(1, 1, 2), dim=-1)
    ape = (point_error * valid.view(1, target.shape[0])).sum(dim=1) / valid_count
    return float(ape.min().detach().cpu().item())


def main() -> None:
    args = parse_args()
    data = check_det_dataset(args.data)
    imgsz = normalize_imgsz(args.imgsz or data.get("image_shape") or data.get("gcs_imgsz"), dataset="tusimple")
    image_dir = data[args.split]
    dataset = GCSLaneDataset(img_path=image_dir, imgsz=imgsz, fraction=1.0, strict=True, augment=False)
    if args.max_images and args.max_images > 0:
        dataset.im_files = dataset.im_files[: int(args.max_images)]
        dataset.label_files = dataset.label_files[: int(args.max_images)]
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=GCSLaneDataset.collate_fn,
        pin_memory=False,
    )
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    scale = torch.tensor([float(imgsz[1]), float(imgsz[0])], device=device, dtype=torch.float32)
    stats = defaultdict(_empty_stats)
    images = 0
    with torch.inference_mode():
        for batch in loader:
            img = batch["img"].to(device=device, non_blocking=device.type == "cuda").float() / 255.0
            if args.half:
                img = img.half()
            preds = model(img)
            candidate_points = preds.get("pred_short_candidate_points")
            if candidate_points is None:
                raise RuntimeError("Model did not emit pred_short_candidate_points; use the gated-candidate-v2 YAML/checkpoint.")
            candidate_points = candidate_points.detach().float()
            for i, (lanes, valid, gt_count_t, path) in enumerate(
                zip(batch["lanes"], batch["lane_valid"], batch["num_lanes"], batch["im_file"])
            ):
                images += 1
                gt_count = int(gt_count_t.item())
                if gt_count not in {4, 5}:
                    continue
                lanes = lanes.to(device=device, dtype=torch.float32)
                valid = valid.to(device=device, dtype=torch.float32)
                visible_counts = valid.sum(dim=1)
                short_mask = (visible_counts >= float(args.min_visible)) & (visible_counts <= float(args.visible_thr))
                for lane_idx in torch.nonzero(short_mask, as_tuple=False).flatten().tolist():
                    best_ape = _candidate_best_ape(candidate_points[i], lanes[lane_idx], valid[lane_idx], scale)
                    group = f"short_gt{gt_count}"
                    _update(stats[group], best_ape, args.hit20_px, args.hit40_px)
                    _update(stats["short_gt4_gt5"], best_ape, args.hit20_px, args.hit40_px)
                    if gt_count == 5 and "0601" in str(path).replace("\\", "/"):
                        _update(stats["train0601_short_gt5"], best_ape, args.hit20_px, args.hit40_px)

    summary = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "split": str(args.split),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "images": int(images),
        "visible_thr": int(args.visible_thr),
        "min_visible": int(args.min_visible),
        "hit20_px": float(args.hit20_px),
        "hit40_px": float(args.hit40_px),
        "groups": {name: _summarize(value) for name, value in sorted(stats.items())},
        "test_closed": True,
    }
    text = json.dumps(summary, indent=2)
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
