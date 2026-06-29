from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.models.gcs.slot_targets import build_ordered_lane_slots_single  # noqa: E402
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str  # noqa: E402


DEFAULT_IMAGE_DIR = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544" / "images" / "train"
DEFAULT_LABEL_DIR = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544" / "labels_gcs" / "train"
COLORS = (
    (0, 255, 0),
    (0, 200, 255),
    (255, 120, 0),
    (255, 0, 180),
    (80, 180, 255),
)


def _save_rgb_image(path: str | Path, image: np.ndarray) -> bool:
    path = Path(path)
    if image.ndim == 3 and image.shape[2] == 3:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image
    ok = bool(cv2.imwrite(str(path), image_to_save))
    if not ok:
        raise IOError(f"Failed to write ordered-slot target visualization: {path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ordered-slot targets built from GCS labels.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR), help="Image directory or txt list.")
    parser.add_argument("--label-dir", default=str(DEFAULT_LABEL_DIR), help="labels_gcs directory.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS shape as H W. Defaults to dataset shape.")
    parser.add_argument("--num-images", type=int, default=20, help="Number of random samples to render.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default=str(ROOT / "runs/gcs_lane/ordered_slot_targets"), help="Output directory.")
    return parser.parse_args()


def _draw_slot(image: np.ndarray, points: np.ndarray, valid: np.ndarray, slot: int, start: int, end: int) -> None:
    h, w = image.shape[:2]
    color = COLORS[slot % len(COLORS)]
    valid = valid > 0.5
    pts = points.copy()
    pts[:, 0] *= float(w)
    pts[:, 1] *= float(h)
    visible = pts[valid]
    if visible.shape[0] >= 2:
        cv2.polylines(image, [np.round(visible).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    for k, (x, y) in enumerate(pts):
        if not valid[k]:
            continue
        radius = 5 if k in {start, end} else 3
        dot = (0, 255, 0) if k == start else (0, 0, 255) if k == end else color
        cv2.circle(image, (int(round(x)), int(round(y))), radius, dot, -1, cv2.LINE_AA)
    if visible.shape[0]:
        x0, y0 = np.round(visible[0]).astype(np.int32).tolist()
        cv2.putText(
            image,
            f"s{slot} [{start},{end}]",
            (int(x0), max(16, int(y0) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    from ultralytics.data.dataset_gcs import GCSLaneDataset

    args = parse_args()
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    dataset = GCSLaneDataset(img_path=args.image_dir, label_dir=args.label_dir, imgsz=imgsz, strict=True, augment=False)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No samples found in {args.image_dir}")
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: min(int(args.num_images), len(indices))]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for out_i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["img"].permute(1, 2, 0).numpy().copy()
        targets = build_ordered_lane_slots_single(sample["lanes"], sample["lane_valid"], max_lanes=5)
        slot_points = targets["slot_points"].cpu().numpy()
        slot_valid = targets["slot_valid"].cpu().numpy()
        slot_exist = targets["slot_exist"].cpu().numpy()
        start_labels = targets["start_labels"].cpu().numpy()
        end_labels = targets["end_labels"].cpu().numpy()
        sorted_gt_indices = targets["sorted_gt_indices"].cpu().numpy()

        for slot in range(5):
            if slot_exist[slot] > 0.5:
                _draw_slot(
                    image,
                    slot_points[slot],
                    slot_valid[slot],
                    slot,
                    int(start_labels[slot]),
                    int(end_labels[slot]),
                )
        out_path = save_dir / f"{out_i:02d}_{Path(sample['im_file']).stem}.jpg"
        _save_rgb_image(out_path, image)
        manifest.append(
            {
                "index": int(idx),
                "image": sample["im_file"],
                "label": sample["label_file"],
                "output": str(out_path.resolve()),
                "num_lanes": int(targets["num_lanes"].item()),
                "sorted_gt_indices": [int(x) for x in sorted_gt_indices.tolist()],
                "start_labels": [int(x) for x in start_labels.tolist()],
                "end_labels": [int(x) for x in end_labels.tolist()],
            }
        )

    (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"rendered {len(manifest)} ordered-slot target images to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
