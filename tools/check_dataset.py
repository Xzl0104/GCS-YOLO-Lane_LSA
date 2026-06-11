from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.data.dataset_gcs import GCSLaneDataset, gcs_collate_fn
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str


def dataset_defaults(dataset: str) -> dict[str, Path]:
    """Return conventional paths for a converted GCS dataset."""
    name = dataset.lower()
    root = ROOT / "datasets" / ("tusimple_fixed_y_960x544" if name == "tusimple" else name)
    return {
        "image_dir": root / "images" / "train",
        "label_dir": root / "labels_gcs" / "train",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GCSLaneDataset and GCS collate output.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--image-dir", default=None, help="Image split directory.")
    parser.add_argument("--label-dir", default=None, help="GCS npz label directory.")
    parser.add_argument(
        "--imgsz",
        "--img-size",
        nargs="+",
        type=int,
        default=None,
        help="GCS input shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for the smoke test.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction to inspect.")
    return parser.parse_args()


def assert_lane_contract(batch: dict) -> None:
    """Validate normalized coordinates and bottom-to-top lane point order."""
    for lanes, lane_valid in zip(batch["lanes"], batch["lane_valid"]):
        if lanes.ndim != 3 or lanes.shape[-1] != 2:
            raise AssertionError(f"lanes must be N x K x 2, got {tuple(lanes.shape)}")
        if lane_valid.shape != lanes.shape[:2]:
            raise AssertionError(f"lane_valid shape {tuple(lane_valid.shape)} does not match {tuple(lanes.shape[:2])}")

        valid_coords = lanes[lane_valid > 0.5]
        if valid_coords.numel() and (valid_coords.min() < 0.0 or valid_coords.max() > 1.0):
            raise AssertionError("lane coordinates must be normalized to [0, 1]")

        for lane, valid in zip(lanes, lane_valid):
            ys = lane[valid > 0.5, 1]
            if ys.numel() > 1 and torch.any(ys[1:] > ys[:-1] + 1e-6):
                raise AssertionError("lane points must be ordered bottom-to-top by descending y")


def label_lane_histogram(label_dir: Path) -> dict[int, int]:
    """Count labels by lane count without loading image data."""
    hist: dict[int, int] = {}
    for label in sorted(label_dir.glob("*.npz")):
        with np.load(label, allow_pickle=False) as data:
            n = int(data["lanes"].shape[0])
        hist[n] = hist.get(n, 0) + 1
    return dict(sorted(hist.items()))


def main() -> None:
    args = parse_args()
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    defaults = dataset_defaults(args.dataset)
    image_dir = Path(args.image_dir or defaults["image_dir"])
    label_dir = Path(args.label_dir or defaults["label_dir"])
    dataset = GCSLaneDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        img_size=imgsz,
        fraction=args.fraction,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=gcs_collate_fn)
    batch = next(iter(loader))
    assert_lane_contract(batch)

    print(f"dataset: {len(dataset)} samples")
    print(f"point_mode: {dataset.point_mode}")
    print(f"input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"lane_count_hist: {label_lane_histogram(label_dir)}")
    print("img:", tuple(batch["img"].shape), batch["img"].dtype)
    print("num_lanes:", batch["num_lanes"].tolist())
    print("lanes[0]:", tuple(batch["lanes"][0].shape))
    print("lane_valid[0]:", tuple(batch["lane_valid"][0].shape))
    print("OK: GCSLaneDataset contract is valid.")


if __name__ == "__main__":
    main()
