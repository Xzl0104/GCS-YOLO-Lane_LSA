from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


REQUIRED_KEYS = ("lanes", "lane_valid", "num_lanes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize converted GCS TuSimple labels.")
    parser.add_argument("--dataset-root", default="datasets/tusimple_fixed_y_960x544", help="Converted dataset root.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"), help="Dataset split.")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum number of samples to visualize.")
    parser.add_argument("--save-dir", default="runs/label_vis", help="Visualization output directory.")
    return parser.parse_args()


def load_label(label_path: Path) -> dict[str, np.ndarray]:
    """Load and validate one GCS npz label."""
    with np.load(label_path) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise KeyError(f"{label_path} missing keys: {missing}")
        label = {key: data[key] for key in REQUIRED_KEYS}

    lanes = label["lanes"]
    lane_valid = label["lane_valid"]
    if lanes.ndim != 3 or lanes.shape[-1] != 2:
        raise ValueError(f"{label_path} lanes must be N x K x 2, got {lanes.shape}")
    if lane_valid.shape != lanes.shape[:2]:
        raise ValueError(f"{label_path} lane_valid must be N x K, got {lane_valid.shape}")
    if int(label["num_lanes"][0]) != lanes.shape[0]:
        raise ValueError(f"{label_path} num_lanes mismatch")
    if lanes.size and (not np.isfinite(lanes).all() or lanes.min() < 0.0 or lanes.max() > 1.0):
        raise ValueError(f"{label_path} lanes must be normalized to [0, 1]")
    return label


def draw_lanes(img: np.ndarray, lanes: np.ndarray, lane_valid: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for lane, valid in zip(lanes, lane_valid):
        pts: list[tuple[int, int]] = []
        for (x, y), v in zip(lane, valid):
            if v < 0.5:
                continue
            px = int(np.clip(x * w, 0, w - 1))
            py = int(np.clip(y * h, 0, h - 1))
            pts.append((px, py))
        if len(pts) >= 2:
            pts_np = np.asarray(pts, dtype=np.int32)
            cv2.polylines(out, [pts_np], isClosed=False, color=(0, 255, 0), thickness=2)
            stride = max(1, len(pts) // 8)
            for i in range(0, len(pts) - 1, stride):
                cv2.arrowedLine(out, pts[i], pts[i + 1], (0, 220, 255), 2, tipLength=0.35)

        denom = max(1, len(pts) - 1)
        for i, point in enumerate(pts):
            ratio = i / denom
            color = (int(255 * ratio), 40, int(255 * (1.0 - ratio)))
            cv2.circle(out, point, 3, color, -1)
        if pts:
            cv2.circle(out, pts[0], 6, (0, 140, 255), 2)
            cv2.circle(out, pts[-1], 6, (255, 90, 0), 2)
    return out


def main() -> None:
    args = parse_args()
    dataset_root = ROOT / args.dataset_root if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels_gcs" / args.split
    save_dir = (ROOT / args.save_dir if not Path(args.save_dir).is_absolute() else Path(args.save_dir)) / args.split
    save_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.jpg"))[: args.max_images]
    written = 0
    missing_labels = 0
    read_failures = 0
    for image_path in tqdm(image_paths, desc="visualize"):
        label_path = label_dir / f"{image_path.stem}.npz"
        if not label_path.exists():
            missing_labels += 1
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            read_failures += 1
            continue
        data = load_label(label_path)
        lane_img = draw_lanes(img, data["lanes"], data["lane_valid"])
        canvas = np.concatenate([img, lane_img], axis=1)
        cv2.imwrite(str(save_dir / f"{image_path.stem}.jpg"), canvas)
        written += 1

    print(f"saved: {written}")
    print(f"missing labels: {missing_labels}")
    print(f"read failures: {read_failures}")
    print(f"output: {save_dir}")


if __name__ == "__main__":
    main()
