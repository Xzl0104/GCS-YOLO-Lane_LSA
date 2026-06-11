from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.label_utils import mask_to_yolo_segments, points_to_mask
from gcs_tools.tusimple_utils import (
    ensure_dataset_dirs,
    find_archive_root,
    iter_split_samples,
    lane_count_histogram,
    load_image,
    resize_image_and_lanes,
)
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TuSimple lane labels to YOLO segmentation format.")
    parser.add_argument("--archive-root", default="archive/TUSimple", help="TuSimple root or archive directory.")
    parser.add_argument("--output-root", default="datasets/tusimple_fixed_y_960x544", help="Converted dataset root.")
    parser.add_argument(
        "--dataset",
        default="tusimple",
        choices=sorted(DATASET_IMAGE_SHAPES),
        help="Dataset shape preset used when --imgsz is omitted.",
    )
    parser.add_argument(
        "--imgsz",
        "--img-shape",
        nargs="+",
        type=int,
        default=None,
        help="Output image shape as H W. Defaults to TuSimple 544 960.",
    )
    parser.add_argument("--img-size", type=int, default=None, help="Legacy square output size.")
    parser.add_argument("--line-width", type=int, default=12, help="Lane rasterization width in output pixels.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio from training labels.")
    parser.add_argument("--split-seed", type=int, default=0, help="Seed for deterministic train/val split.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split-by-clip",
        dest="split_by_clip",
        action="store_true",
        default=True,
        help="Keep one TuSimple clip group in only one split. This is the default.",
    )
    split_group.add_argument(
        "--no-split-by-clip",
        dest="split_by_clip",
        action="store_false",
        help="Use the older lane-count sample-level split, which can leak adjacent frames across train/val.",
    )
    parser.add_argument("--no-test", action="store_true", help="Do not convert labeled test samples.")
    return parser.parse_args()


def convert_one(split: str, sample, output_root: Path, img_shape: tuple[int, int], line_width: int) -> bool:
    img_h, img_w = img_shape
    image = load_image(sample.image_path)
    resized, lanes = resize_image_and_lanes(image, sample.lanes, img_shape)

    img_path = output_root / "images" / split / f"{sample.sample_id}.jpg"
    label_path = output_root / "labels" / split / f"{sample.sample_id}.txt"

    if not cv2.imwrite(str(img_path), resized):
        raise OSError(f"Failed to write image: {img_path}")

    lines: list[str] = []
    for lane in lanes:
        if len(lane) < 2:
            continue
        mask = points_to_mask(lane, img_h, img_w, line_width=line_width)
        lines.extend(mask_to_yolo_segments(mask, class_id=0))

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    img_shape = normalize_imgsz(args.img_size if args.img_size is not None else args.imgsz, dataset=args.dataset)
    archive_root = find_archive_root(ROOT / args.archive_root if not Path(args.archive_root).is_absolute() else args.archive_root)
    output_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    include_test = not args.no_test
    ensure_dataset_dirs(output_root, include_test=include_test)
    print(f"output image shape: {shape_str(img_shape)} (W x H), stored as H,W={img_shape}")
    print(f"train/val split: {'clip-group' if args.split_by_clip else 'sample-level lane-count'}")

    samples = list(
        iter_split_samples(
            archive_root,
            include_test=include_test,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
            group_by_clip=args.split_by_clip,
        )
    )
    for split in ["train", "val"] + (["test"] if include_test else []):
        split_samples = [sample for sample_split, sample in samples if sample_split == split]
        print(f"{split}: source_samples={len(split_samples)} lane_count_hist={lane_count_histogram(split_samples)}")

    ok = 0
    missing = 0
    for split, sample in tqdm(samples, desc="YOLO labels"):
        if not sample.image_path.exists():
            print(f"missing image: {sample.image_path}")
            missing += 1
            continue
        convert_one(split, sample, output_root, img_shape=img_shape, line_width=args.line_width)
        ok += 1

    print(f"converted: {ok}")
    print(f"missing: {missing}")
    print(f"output: {output_root}")


if __name__ == "__main__":
    main()
