from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.label_utils import mask_to_yolo_segments, points_to_mask
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO segmentation labels from GCS labels_gcs npz files.")
    parser.add_argument("--dataset-root", default="datasets/tusimple_fixed_y_960x544", help="Converted dataset root.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to export.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="Image shape as H W.")
    parser.add_argument("--line-width", type=int, default=12, help="Lane rasterization width in pixels.")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id for lane instances.")
    parser.add_argument("--summary", default="runs/gcs_lane/fixed_y_yolo_labels_summary.json", help="Summary JSON path.")
    return parser.parse_args()


def _load_shape(data: np.lib.npyio.NpzFile, fallback: tuple[int, int]) -> tuple[int, int]:
    if "image_shape" not in data.files:
        return fallback
    shape = np.asarray(data["image_shape"]).reshape(-1)
    if shape.size < 2:
        return fallback
    return int(shape[0]), int(shape[1])


def _label_lines_from_npz(label_path: Path, fallback_shape: tuple[int, int], line_width: int, class_id: int) -> tuple[list[str], int]:
    with np.load(label_path, allow_pickle=False) as data:
        missing = {"lanes", "lane_valid"}.difference(data.files)
        if missing:
            raise KeyError(f"{label_path} missing required arrays: {sorted(missing)}")
        lanes = data["lanes"].astype(np.float32)
        lane_valid = data["lane_valid"].astype(np.float32)
        img_h, img_w = _load_shape(data, fallback_shape)

    if lanes.ndim != 3 or lanes.shape[-1] != 2:
        raise ValueError(f"{label_path}: lanes must have shape N x K x 2, got {lanes.shape}")
    if lane_valid.shape != lanes.shape[:2]:
        raise ValueError(f"{label_path}: lane_valid shape {lane_valid.shape} must match lanes {lanes.shape[:2]}")
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"{label_path}: invalid image shape {(img_h, img_w)}")

    lines: list[str] = []
    kept_lanes = 0
    for lane, valid in zip(lanes, lane_valid):
        keep = valid > 0.5
        if int(keep.sum()) < 2:
            continue
        points = [(float(x) * img_w, float(y) * img_h) for x, y in lane[keep]]
        mask = points_to_mask(points, h=img_h, w=img_w, line_width=line_width)
        segments = mask_to_yolo_segments(mask, class_id=class_id)
        if segments:
            kept_lanes += 1
            lines.extend(segments)
    return lines, kept_lanes


def _validate_yolo_txt(label_path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 7:
            errors.append(f"line {line_idx}: expected class plus at least 3 xy points")
            continue
        coords = parts[1:]
        if len(coords) % 2:
            errors.append(f"line {line_idx}: odd coordinate count")
            continue
        try:
            values = [float(v) for v in coords]
        except ValueError:
            errors.append(f"line {line_idx}: non-numeric coordinate")
            continue
        if any(v < -1e-6 or v > 1.0 + 1e-6 for v in values):
            errors.append(f"line {line_idx}: coordinate outside [0, 1]")
    return len(lines), errors


def export_split(dataset_root: Path, split: str, img_shape: tuple[int, int], line_width: int, class_id: int) -> dict:
    image_dir = dataset_root / "images" / split
    gcs_dir = dataset_root / "labels_gcs" / split
    yolo_dir = dataset_root / "labels" / split
    yolo_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.jpg"))
    gcs_labels = sorted(gcs_dir.glob("*.npz"))
    if len(images) != len(gcs_labels):
        raise ValueError(f"{split}: images={len(images)} but labels_gcs={len(gcs_labels)}")

    image_stems = {p.stem for p in images}
    gcs_stems = {p.stem for p in gcs_labels}
    if image_stems != gcs_stems:
        missing = sorted(image_stems - gcs_stems)[:5]
        extra = sorted(gcs_stems - image_stems)[:5]
        raise ValueError(f"{split}: image/labels_gcs stem mismatch, missing={missing}, extra={extra}")

    exported = 0
    total_lines = 0
    total_gcs_lanes = 0
    file_errors: dict[str, list[str]] = {}
    for label_path in tqdm(gcs_labels, desc=f"export {split}"):
        lines, gcs_lanes = _label_lines_from_npz(label_path, img_shape, line_width=line_width, class_id=class_id)
        out_path = yolo_dir / f"{label_path.stem}.txt"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        line_count, errors = _validate_yolo_txt(out_path)
        if errors:
            file_errors[str(out_path)] = errors[:5]
        exported += 1
        total_lines += line_count
        total_gcs_lanes += gcs_lanes

    yolo_labels = sorted(yolo_dir.glob("*.txt"))
    yolo_stems = {p.stem for p in yolo_labels}
    if yolo_stems != image_stems:
        missing = sorted(image_stems - yolo_stems)[:5]
        extra = sorted(yolo_stems - image_stems)[:5]
        raise ValueError(f"{split}: image/labels stem mismatch after export, missing={missing}, extra={extra}")

    return {
        "images": len(images),
        "labels_gcs": len(gcs_labels),
        "labels": len(yolo_labels),
        "exported": exported,
        "yolo_lines": total_lines,
        "gcs_lanes_with_segments": total_gcs_lanes,
        "file_errors": file_errors,
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    img_shape = normalize_imgsz(args.imgsz)
    print(f"dataset_root: {dataset_root}")
    print(f"image shape: {shape_str(img_shape)} (W x H), stored as H,W={img_shape}")

    summary = {"dataset_root": str(dataset_root), "splits": {}}
    for split in args.splits:
        summary["splits"][split] = export_split(
            dataset_root,
            split,
            img_shape=img_shape,
            line_width=args.line_width,
            class_id=args.class_id,
        )
        print(f"{split}: {summary['splits'][split]}")

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
