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

from gcs_tools.label_utils import (
    TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM,
    resample_polyline,
    sample_polyline_fixed_y,
)
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
    parser = argparse.ArgumentParser(description="Convert TuSimple labels to GCS structured npz labels.")
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
    parser.add_argument("--num-points", type=int, default=32, help="Fixed number of points per lane.")
    parser.add_argument(
        "--point-mode",
        choices=("free", "fixed_y"),
        default="fixed_y",
        help="free uses arc-length Kx2 labels; fixed_y samples x at shared y anchors for x-only heads.",
    )
    parser.add_argument(
        "--fixed-y-start",
        type=float,
        default=TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM,
        help="Bottom normalized y anchor for fixed_y mode. Defaults to TuSimple h=710 over H=720.",
    )
    parser.add_argument("--fixed-y-end", type=float, default=0.25, help="Top normalized y anchor for fixed_y mode.")
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


def build_gcs_arrays(
    lanes: list[list[tuple[float, float]]],
    img_shape: tuple[int, int],
    num_points: int,
    point_mode: str = "free",
    fixed_y_start: float = TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM,
    fixed_y_end: float = 0.25,
) -> dict[str, np.ndarray]:
    img_h, img_w = img_shape
    point_mode = str(point_mode).lower()
    if point_mode not in {"free", "fixed_y"}:
        raise ValueError(f"Unsupported point_mode={point_mode!r}; use 'free' or 'fixed_y'.")

    all_lanes: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    fixed_y = None
    for lane in lanes:
        if point_mode == "fixed_y":
            sampled, valid, fixed_y = sample_polyline_fixed_y(
                lane,
                img_h=img_h,
                img_w=img_w,
                num_points=num_points,
                y_start=fixed_y_start,
                y_end=fixed_y_end,
            )
        else:
            sampled, valid = resample_polyline(lane, num_points=num_points)
            sampled[:, 0] = sampled[:, 0] / float(img_w)
            sampled[:, 1] = sampled[:, 1] / float(img_h)
        if valid.sum() < 2 or not np.isfinite(sampled).all():
            continue
        sampled = np.clip(sampled, 0.0, 1.0).astype(np.float32)
        all_lanes.append(sampled)
        all_valid.append(valid.astype(np.float32))

    if all_lanes:
        lane_arr = np.stack(all_lanes, axis=0).astype(np.float32)
        valid_arr = np.stack(all_valid, axis=0).astype(np.float32)
    else:
        lane_arr = np.zeros((0, num_points, 2), dtype=np.float32)
        valid_arr = np.zeros((0, num_points), dtype=np.float32)

    arrays = {
        "lanes": lane_arr,
        "lane_valid": valid_arr,
        "num_lanes": np.array([lane_arr.shape[0]], dtype=np.int64),
        "point_mode": np.array(point_mode),
    }
    if point_mode == "fixed_y":
        if fixed_y is None:
            _, _, fixed_y = sample_polyline_fixed_y(
                [],
                img_h=img_h,
                img_w=img_w,
                num_points=num_points,
                y_start=fixed_y_start,
                y_end=fixed_y_end,
            )
        arrays["fixed_y"] = fixed_y.astype(np.float32)
    return arrays


def validate_gcs_arrays(arrays: dict[str, np.ndarray], img_shape: tuple[int, int], num_points: int) -> None:
    """Fail fast when a generated label violates the structured lane contract."""
    required = {"lanes", "lane_valid", "num_lanes", "point_mode"}
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"Missing GCS arrays: {sorted(missing)}")

    lanes = arrays["lanes"]
    lane_valid = arrays["lane_valid"]
    num_lanes = arrays["num_lanes"]
    point_mode = str(np.asarray(arrays.get("point_mode", np.array("free"))).item())

    if lanes.ndim != 3 or lanes.shape[1:] != (num_points, 2):
        raise ValueError(f"lanes shape must be N x {num_points} x 2, got {lanes.shape}")
    if lane_valid.shape != lanes.shape[:2]:
        raise ValueError(f"lane_valid shape must match lanes first two dims, got {lane_valid.shape} vs {lanes.shape[:2]}")
    if int(num_lanes[0]) != lanes.shape[0]:
        raise ValueError(f"num_lanes mismatch: {int(num_lanes[0])} vs {lanes.shape[0]}")
    if lanes.size and (not np.isfinite(lanes).all() or lanes.min() < 0.0 or lanes.max() > 1.0):
        raise ValueError("lanes must be finite normalized coordinates in [0, 1]")
    if point_mode == "fixed_y":
        fixed_y = arrays.get("fixed_y")
        if fixed_y is None or fixed_y.shape != (num_points,):
            raise ValueError(f"fixed_y labels require fixed_y shape ({num_points},), got {None if fixed_y is None else fixed_y.shape}")
        if not np.all(np.diff(fixed_y) < 0.0):
            raise ValueError("fixed_y anchors must be strictly descending from bottom to top")
        if lanes.size:
            expected_y = fixed_y.reshape(1, num_points)
            y_err = np.abs(lanes[..., 1] - expected_y) * (lane_valid > 0.5)
            if y_err.size and float(y_err.max()) > 1e-5:
                raise ValueError("fixed_y lane labels must keep y coordinates aligned with fixed_y anchors")


def convert_one(
    split: str,
    sample,
    output_root: Path,
    img_shape: tuple[int, int],
    num_points: int,
    point_mode: str,
    fixed_y_start: float,
    fixed_y_end: float,
) -> bool:
    image = load_image(sample.image_path)
    resized, lanes = resize_image_and_lanes(image, sample.lanes, img_shape)

    img_path = output_root / "images" / split / f"{sample.sample_id}.jpg"
    label_path = output_root / "labels_gcs" / split / f"{sample.sample_id}.npz"

    if not cv2.imwrite(str(img_path), resized):
        raise OSError(f"Failed to write image: {img_path}")

    arrays = build_gcs_arrays(
        lanes,
        img_shape=img_shape,
        num_points=num_points,
        point_mode=point_mode,
        fixed_y_start=fixed_y_start,
        fixed_y_end=fixed_y_end,
    )
    validate_gcs_arrays(arrays, img_shape=img_shape, num_points=num_points)
    np.savez_compressed(
        label_path,
        **arrays,
        raw_file=np.array(sample.raw_file),
        image_shape=np.array(img_shape, dtype=np.int32),
        num_points=np.array([num_points], dtype=np.int32),
    )
    return True


def report_outputs(output_root: Path, include_test: bool, img_shape: tuple[int, int], num_points: int) -> None:
    """Print split-level pairing counts and inspect one npz per split."""
    for split in ["train", "val"] + (["test"] if include_test else []):
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels_gcs" / split
        images = sorted(image_dir.glob("*.jpg"))
        labels = sorted(label_dir.glob("*.npz"))
        lane_hist: dict[int, int] = {}
        for label in labels:
            with np.load(label) as data:
                n = int(data["lanes"].shape[0])
            lane_hist[n] = lane_hist.get(n, 0) + 1
        print(f"{split}: images={len(images)} labels_gcs={len(labels)} lane_count_hist={dict(sorted(lane_hist.items()))}")
        if not labels:
            continue
        with np.load(labels[0]) as data:
            arrays = {k: data[k] for k in data.files if k in {"lanes", "lane_valid", "num_lanes", "point_mode", "fixed_y"}}
            validate_gcs_arrays(
                arrays,
                img_shape,
                num_points,
            )
            point_mode = str(np.asarray(data["point_mode"]).item()) if "point_mode" in data.files else "free"
            print(
                f"{split}: sample={labels[0].name} lanes_shape={data['lanes'].shape} "
                f"point_mode={point_mode}"
            )


def main() -> None:
    args = parse_args()
    img_shape = normalize_imgsz(args.img_size if args.img_size is not None else args.imgsz, dataset=args.dataset)
    archive_root = find_archive_root(ROOT / args.archive_root if not Path(args.archive_root).is_absolute() else args.archive_root)
    output_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    include_test = not args.no_test
    ensure_dataset_dirs(output_root, include_test=include_test)
    print(f"output image shape: {shape_str(img_shape)} (W x H), stored as H,W={img_shape}")
    print(f"point mode: {args.point_mode}")
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
    for split, sample in tqdm(samples, desc="GCS labels"):
        if not sample.image_path.exists():
            print(f"missing image: {sample.image_path}")
            missing += 1
            continue
        convert_one(
            split,
            sample,
            output_root,
            img_shape=img_shape,
            num_points=args.num_points,
            point_mode=args.point_mode,
            fixed_y_start=args.fixed_y_start,
            fixed_y_end=args.fixed_y_end,
        )
        ok += 1

    print(f"converted: {ok}")
    print(f"missing: {missing}")
    print(f"output: {output_root}")
    report_outputs(output_root, include_test=include_test, img_shape=img_shape, num_points=args.num_points)


if __name__ == "__main__":
    main()
