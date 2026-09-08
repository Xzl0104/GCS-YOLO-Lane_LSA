from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.label_utils import build_edge_mask, build_semantic_mask, sample_polyline_fixed_y
from ultralytics.utils.gcs_fixed_y import build_fixed_y_anchors, validate_fixed_y_contract


TEST_CATEGORY_FILES = {
    "test0_normal": "Normal",
    "test1_crowd": "Crowded",
    "test2_hlight": "Dazzle",
    "test3_shadow": "Shadow",
    "test4_noline": "No line",
    "test5_arrow": "Arrow",
    "test6_curve": "Curve",
    "test7_cross": "Crossroad",
    "test8_night": "Night",
}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the official CULane lists and lane points into GCS structured labels."
    )
    parser.add_argument(
        "--archive-root",
        default=r"D:\BaiduNetdiskDownload\CULane",
        help="Extracted CULane root containing driver_* directories and list/.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "datasets" / "culane_fixed_y_590x960"),
        help="Output dataset root containing images/, labels_gcs/, manifests/.",
    )
    parser.add_argument("--imgsz", nargs=2, type=int, metavar=("H", "W"), default=(384, 960))
    parser.add_argument(
        "--raw-imgsz",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=(590, 1640),
        help="Original CULane image shape. Official CULane images are 590x1640.",
    )
    parser.add_argument("--num-points", type=int, default=56)
    parser.add_argument(
        "--fixed-y-start-px",
        type=float,
        default=589.0,
        help="Bottom source-image y anchor for CULane fixed-y labels.",
    )
    parser.add_argument(
        "--fixed-y-end-px",
        type=float,
        default=39.0,
        help="Top source-image y anchor for CULane fixed-y labels.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=8,
        help="Lane-mask line width after scaling to the output H,W shape.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Only convert the selected official splits. Defaults to train val test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files already generated at the output root.",
    )
    parser.add_argument(
        "--verify-image-shapes",
        action="store_true",
        help="Decode every source image and fail if its shape differs from --raw-imgsz.",
    )
    parser.add_argument(
        "--reuse-images-root",
        default=None,
        help=(
            "Optional root containing verified preprocessed images as "
            "<root>/<split>/<flattened-source-stem>.jpg. This allows labels to be rebuilt from "
            "official .lines.txt files when raw source JPGs are unavailable on the conversion host."
        ),
    )
    parser.add_argument(
        "--reuse-image-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to materialize --reuse-images-root inputs into the converted dataset.",
    )
    return parser.parse_args()


def normalize_rel_path(value: str) -> str:
    """Normalize a CULane list path to a slash-separated root-relative key."""
    value = str(value).strip().replace("\\", "/")
    value = value.lstrip("/")
    return str(Path(value)).replace("\\", "/")


def archive_path_for(root: Path, raw_file: str) -> Path:
    """Resolve a CULane list entry without treating its leading slash as a drive root."""
    rel = normalize_rel_path(raw_file)
    return root / Path(rel.replace("/", os.sep))


def source_path_for(root: Path, raw_file: str) -> Path:
    """Return a required raw CULane source image path."""
    path = archive_path_for(root, raw_file)
    if not path.is_file():
        raise FileNotFoundError(f"CULane source image does not exist: {path}")
    return path


def source_lines_path_for(root: Path, raw_file: str) -> Path:
    """Return the required official polyline annotation for one CULane source image."""
    path = archive_path_for(root, raw_file).with_suffix(".lines.txt")
    if not path.is_file():
        raise FileNotFoundError(f"CULane lane annotation does not exist: {path}")
    return path


def read_list(path: Path) -> list[str]:
    """Read non-empty CULane list entries and normalize their path spelling."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing CULane list file: {path}")
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            values.append(normalize_rel_path(line))
    return values


def parse_lines_file(path: Path) -> list[list[tuple[float, float]]]:
    """Parse one CULane .lines.txt file into pixel-space lane polylines."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing CULane lane annotation: {path}")

    lanes: list[list[tuple[float, float]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        tokens = line.strip().split()
        if not tokens:
            continue
        if len(tokens) % 2:
            raise ValueError(f"{path}:{line_number} has an odd number of coordinate values.")

        points: list[tuple[float, float]] = []
        for index in range(0, len(tokens), 2):
            try:
                x = float(tokens[index])
                y = float(tokens[index + 1])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} contains non-numeric coordinates.") from exc
            if np.isfinite(x) and np.isfinite(y) and x >= 0.0 and y >= 0.0:
                points.append((x, y))
        if len(points) >= 2:
            lanes.append(points)
    return lanes


def build_gcs_arrays(
    raw_lanes: list[list[tuple[float, float]]],
    raw_shape: tuple[int, int],
    output_shape: tuple[int, int],
    num_points: int,
    line_width: int,
    fixed_y_start_px: float,
    fixed_y_end_px: float,
) -> tuple[dict[str, np.ndarray], int]:
    """Create fixed-y GCS arrays and return the number of discarded raw lanes."""
    raw_h, raw_w = raw_shape
    out_h, out_w = output_shape
    if raw_h <= 0 or raw_w <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError(f"Image shapes must be positive, got raw={raw_shape}, output={output_shape}.")

    sampled_lanes: list[np.ndarray] = []
    valid_lanes: list[np.ndarray] = []
    mask_lanes: list[list[tuple[float, float]]] = []
    dropped = 0
    scale_x = float(out_w) / float(raw_w)
    scale_y = float(out_h) / float(raw_h)
    fixed_y = build_fixed_y_anchors(
        original_h=raw_h,
        start_px=fixed_y_start_px,
        end_px=fixed_y_end_px,
        k=num_points,
    )

    for raw_lane in raw_lanes:
        sampled, valid, lane_fixed_y = sample_polyline_fixed_y(
            raw_lane,
            img_h=raw_h,
            img_w=raw_w,
            num_points=num_points,
            y_start=float(fixed_y_start_px) / float(raw_h),
            y_end=float(fixed_y_end_px) / float(raw_h),
        )
        if valid.sum() < 2 or not np.isfinite(sampled).all():
            dropped += 1
            continue
        if not np.allclose(lane_fixed_y, fixed_y, atol=1e-7):
            raise ValueError("CULane fixed-y sampling returned anchors inconsistent with the conversion contract.")
        sampled = np.clip(sampled, 0.0, 1.0).astype(np.float32)
        sampled_lanes.append(sampled)
        valid_lanes.append(valid.astype(np.float32))
        mask_lanes.append([(float(x) * scale_x, float(y) * scale_y) for x, y in raw_lane])

    semantic_mask = build_semantic_mask(mask_lanes, h=out_h, w=out_w, line_width=line_width)
    edge_mask = build_edge_mask(semantic_mask)
    if sampled_lanes:
        lanes = np.stack(sampled_lanes, axis=0).astype(np.float32)
        lane_valid = np.stack(valid_lanes, axis=0).astype(np.float32)
    else:
        lanes = np.zeros((0, num_points, 2), dtype=np.float32)
        lane_valid = np.zeros((0, num_points), dtype=np.float32)

    arrays = {
        "lanes": lanes,
        "lane_valid": lane_valid,
        "semantic_mask": semantic_mask.astype(np.uint8),
        "edge_mask": edge_mask.astype(np.float32),
        "num_lanes": np.array([lanes.shape[0]], dtype=np.int64),
        "point_mode": np.array("fixed_y"),
        "fixed_y": fixed_y.astype(np.float32),
        "fixed_y_original_h": np.array([raw_h], dtype=np.int32),
        "fixed_y_start_px": np.array([fixed_y_start_px], dtype=np.float32),
        "fixed_y_end_px": np.array([fixed_y_end_px], dtype=np.float32),
        "num_points": np.array([num_points], dtype=np.int32),
        "image_shape": np.array(output_shape, dtype=np.int32),
    }
    return arrays, dropped


def output_stem(raw_file: str) -> str:
    """Flatten the CULane relative path while preserving a globally unique name."""
    rel = Path(normalize_rel_path(raw_file))
    return rel.with_suffix("").as_posix().replace("/", "__")


def validate_arrays(
    arrays: dict[str, np.ndarray],
    output_shape: tuple[int, int],
    num_points: int,
    raw_shape: tuple[int, int],
    fixed_y_start_px: float,
    fixed_y_end_px: float,
) -> None:
    """Validate the on-disk GCS label contract before writing an NPZ."""
    out_h, out_w = output_shape
    lanes = arrays["lanes"]
    lane_valid = arrays["lane_valid"]
    if arrays["semantic_mask"].shape != (out_h, out_w):
        raise ValueError(f"semantic_mask shape mismatch: {arrays['semantic_mask'].shape}")
    if arrays["edge_mask"].shape != (out_h, out_w):
        raise ValueError(f"edge_mask shape mismatch: {arrays['edge_mask'].shape}")
    if lanes.shape != (lanes.shape[0], num_points, 2):
        raise ValueError(f"lanes shape mismatch: {lanes.shape}")
    if lane_valid.shape != lanes.shape[:2]:
        raise ValueError(f"lane_valid shape mismatch: {lane_valid.shape} vs {lanes.shape[:2]}")
    if int(arrays["num_lanes"].reshape(-1)[0]) != lanes.shape[0]:
        raise ValueError("num_lanes does not match lanes.shape[0].")
    if lanes.size and (not np.isfinite(lanes).all() or lanes.min() < 0.0 or lanes.max() > 1.0):
        raise ValueError("lanes contain invalid normalized coordinates.")
    fixed_y = np.asarray(arrays["fixed_y"], dtype=np.float32).reshape(-1)
    validate_fixed_y_contract(
        fixed_y,
        original_h=int(raw_shape[0]),
        start_px=fixed_y_start_px,
        end_px=fixed_y_end_px,
        k=num_points,
        name="CULane fixed_y",
    )
    if int(np.asarray(arrays["fixed_y_original_h"]).reshape(-1)[0]) != int(raw_shape[0]):
        raise ValueError("fixed_y_original_h does not match the raw CULane image height.")
    if lanes.shape[0]:
        expected_y = fixed_y.reshape(1, num_points)
        if not np.allclose(lanes[..., 1], expected_y, atol=1e-6):
            raise ValueError("fixed-y lane labels must retain the shared y anchor for valid and invalid rows.")
        invalid_x = lanes[..., 0][lane_valid <= 0.5]
        if invalid_x.size and not np.allclose(invalid_x, 0.0, atol=1e-6):
            raise ValueError("fixed-y invalid anchors must use x=0.")
        if np.any(lane_valid.sum(axis=1) < 2):
            raise ValueError("CULane labels must not retain lanes with fewer than two valid fixed-y anchors.")


def write_resized_image(path: Path, output_path: Path, raw_shape: tuple[int, int], output_shape: tuple[int, int]) -> None:
    """Decode a CULane image, verify its source shape, and write the model input shape."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Failed to decode CULane image: {path}")
    if tuple(image.shape[:2]) != tuple(raw_shape):
        raise ValueError(f"Unexpected CULane image shape for {path}: {image.shape[:2]} != {raw_shape}")
    if tuple(image.shape[:2]) != tuple(output_shape):
        image = cv2.resize(image, (int(output_shape[1]), int(output_shape[0])), interpolation=cv2.INTER_LINEAR)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to write resized CULane image: {output_path}")


def materialize_reused_image(
    source_path: Path,
    output_path: Path,
    output_shape: tuple[int, int],
    mode: str,
) -> None:
    """Validate and hard-link or copy one verified preprocessed image into the new dataset."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Reused CULane image does not exist: {source_path}")
    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Failed to decode reused CULane image: {source_path}")
    if tuple(image.shape[:2]) != tuple(output_shape):
        raise ValueError(
            f"Reused CULane image shape for {source_path} is {image.shape[:2]}, expected {output_shape}."
        )
    if output_path.exists():
        output_path.unlink()
    if mode == "hardlink":
        try:
            os.link(source_path, output_path)
        except OSError as exc:
            raise OSError(
                f"Could not hard-link reused CULane image {source_path} -> {output_path}. "
                "Use --reuse-image-mode copy when the two roots are on different filesystems."
            ) from exc
    elif mode == "copy":
        shutil.copy2(source_path, output_path)
    else:
        raise ValueError(f"Unsupported reused image mode: {mode!r}")


class JsonArrayWriter:
    """Stream a JSON array so the 133k-image manifest does not require a large list in RAM."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.first = True

    def __enter__(self) -> "JsonArrayWriter":
        self.handle = self.path.open("w", encoding="utf-8", newline="\n")
        self.handle.write("[\n")
        return self

    def write(self, value: dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("JsonArrayWriter is not open.")
        if not self.first:
            self.handle.write(",\n")
        self.handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        self.first = False

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is not None:
            self.handle.write("\n]\n")
            self.handle.close()


def ensure_output_is_safe(output_root: Path, overwrite: bool) -> None:
    """Refuse accidental mixing with an existing conversion unless explicitly requested."""
    if not output_root.exists():
        return
    generated_paths = (
        output_root / "images",
        output_root / "labels_gcs",
        output_root / "manifests",
        output_root / "dataset.yaml",
    )
    existing = [path for path in generated_paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output already contains generated CULane paths:\n{joined}\n"
            "Use a new output root or pass --overwrite."
        )


def load_test_categories(list_root: Path) -> dict[str, str]:
    """Map every official test image to one mutually exclusive display category."""
    mapping: dict[str, str] = {}
    for file_stem, display_name in TEST_CATEGORY_FILES.items():
        path = list_root / "test_split" / f"{file_stem}.txt"
        for raw_file in read_list(path):
            previous = mapping.get(raw_file)
            if previous is not None and previous != display_name:
                raise ValueError(f"Test category overlap for {raw_file}: {previous} vs {display_name}")
            mapping[raw_file] = display_name
    return mapping


def write_dataset_yaml(
    output_root: Path,
    output_shape: tuple[int, int],
    raw_shape: tuple[int, int],
    num_points: int,
    fixed_y_start_px: float,
    fixed_y_end_px: float,
) -> None:
    """Write a self-contained dataset YAML for direct use from the converted root."""
    out_h, out_w = output_shape
    dataset_root = output_root.resolve().as_posix()
    content = (
        f"path: {json.dumps(dataset_root)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"image_shape: [{out_h}, {out_w}]\n"
        "point_mode: fixed_y\n"
        f"num_points: {num_points}\n"
        f"fixed_y_original_h: {int(raw_shape[0])}\n"
        f"fixed_y_start_px: {float(fixed_y_start_px):.8g}\n"
        f"fixed_y_end_px: {float(fixed_y_end_px):.8g}\n"
        f"fixed_y: [{float(fixed_y_start_px) / float(raw_shape[0]):.16g}, "
        f"{float(fixed_y_end_px) / float(raw_shape[0]):.16g}]\n"
        "names:\n"
        "  0: lane\n"
    )
    (output_root / "dataset.yaml").write_text(content, encoding="utf-8")


def convert(args: argparse.Namespace) -> dict[str, Any]:
    archive_root = Path(args.archive_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    reuse_images_root = (
        Path(args.reuse_images_root).expanduser().resolve() if args.reuse_images_root else None
    )
    list_root = archive_root / "list"
    output_shape = (int(args.imgsz[0]), int(args.imgsz[1]))
    raw_shape = (int(args.raw_imgsz[0]), int(args.raw_imgsz[1]))
    selected_splits = tuple(dict.fromkeys(str(split) for split in args.splits))
    if not selected_splits:
        raise ValueError("--splits must contain at least one split.")
    if int(args.num_points) <= 0:
        raise ValueError(f"--num-points must be positive, got {args.num_points}")
    if int(args.line_width) <= 0:
        raise ValueError(f"--line-width must be positive, got {args.line_width}")
    if reuse_images_root is not None and not reuse_images_root.is_dir():
        raise FileNotFoundError(f"--reuse-images-root does not exist or is not a directory: {reuse_images_root}")
    build_fixed_y_anchors(
        original_h=raw_shape[0],
        start_px=float(args.fixed_y_start_px),
        end_px=float(args.fixed_y_end_px),
        k=int(args.num_points),
    )

    ensure_output_is_safe(output_root, overwrite=bool(args.overwrite))
    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels_gcs" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)

    split_files = {split: read_list(list_root / f"{split}.txt") for split in selected_splits}
    split_sets = {split: set(values) for split, values in split_files.items()}
    for index, first in enumerate(selected_splits):
        for second in selected_splits[index + 1 :]:
            overlap = sorted(split_sets[first].intersection(split_sets[second]))
            if overlap:
                raise ValueError(f"Official CULane split overlap: {first} vs {second}, examples={overlap[:3]}")

    test_categories = load_test_categories(list_root) if "test" in selected_splits else {}
    if "test" in selected_splits:
        test_set = split_sets["test"]
        if set(test_categories) != test_set:
            missing = sorted(test_set.difference(test_categories))
            extra = sorted(set(test_categories).difference(test_set))
            raise ValueError(
                f"CULane test categories do not exactly cover test.txt: missing={missing[:3]}, extra={extra[:3]}"
            )

    all_seen: set[str] = set()
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    lane_count_hist: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    dropped_lane_count = 0
    missing_segmentation_count = 0
    image_shape_checked = 0

    manifest_path = output_root / "manifests" / "manifest.json"
    split_manifest_paths = {
        split: output_root / "manifests" / f"{split}.json" for split in SPLITS
    }

    with JsonArrayWriter(manifest_path) as manifest_writer:
        split_writers = {split: JsonArrayWriter(path) for split, path in split_manifest_paths.items()}
        with split_writers["train"] as train_writer, split_writers["val"] as val_writer, split_writers["test"] as test_writer:
            writers = {"train": train_writer, "val": val_writer, "test": test_writer}
            for split in selected_splits:
                values: Iterable[str] = tqdm(
                    split_files[split],
                    desc=f"Converting {split}",
                    unit="image",
                )
                for raw_file in values:
                    if raw_file in all_seen:
                        raise ValueError(f"Duplicate source image across conversion: {raw_file}")
                    all_seen.add(raw_file)

                    source_image = archive_path_for(archive_root, raw_file)
                    source_lines = source_lines_path_for(archive_root, raw_file)
                    raw_lanes = parse_lines_file(source_lines)

                    stem = output_stem(raw_file)
                    output_image = output_root / "images" / split / f"{stem}.jpg"
                    output_label = output_root / "labels_gcs" / split / f"{stem}.npz"
                    if output_image.exists() or output_label.exists():
                        if not args.overwrite:
                            raise FileExistsError(
                                f"Output pair already exists for {raw_file}: {output_image}, {output_label}"
                            )

                    output_image.parent.mkdir(parents=True, exist_ok=True)
                    output_label.parent.mkdir(parents=True, exist_ok=True)
                    if reuse_images_root is None:
                        source_image = source_path_for(archive_root, raw_file)
                        write_resized_image(
                            source_image,
                            output_image,
                            raw_shape=raw_shape,
                            output_shape=output_shape,
                        )
                    else:
                        reused_image = reuse_images_root / split / f"{stem}.jpg"
                        materialize_reused_image(
                            reused_image,
                            output_image,
                            output_shape=output_shape,
                            mode=str(args.reuse_image_mode),
                        )
                    image_shape_checked += 1

                    arrays, dropped = build_gcs_arrays(
                        raw_lanes,
                        raw_shape=raw_shape,
                        output_shape=output_shape,
                        num_points=int(args.num_points),
                        line_width=int(args.line_width),
                        fixed_y_start_px=float(args.fixed_y_start_px),
                        fixed_y_end_px=float(args.fixed_y_end_px),
                    )
                    validate_arrays(
                        arrays,
                        output_shape=output_shape,
                        num_points=int(args.num_points),
                        raw_shape=raw_shape,
                        fixed_y_start_px=float(args.fixed_y_start_px),
                        fixed_y_end_px=float(args.fixed_y_end_px),
                    )
                    category = test_categories.get(raw_file, "")
                    source_segmentation = archive_root / "laneseg_label_w16" / Path(
                        raw_file.replace("/", os.sep)
                    ).with_suffix(".png")
                    if not source_segmentation.is_file():
                        missing_segmentation_count += 1
                        source_segmentation_value = ""
                    else:
                        source_segmentation_value = str(source_segmentation.relative_to(archive_root)).replace(
                            "\\", "/"
                        )

                    np.savez_compressed(
                        output_label,
                        **arrays,
                        raw_file=np.array(raw_file),
                        source_image=np.array(normalize_rel_path(raw_file)),
                        source_lines=np.array(str(source_lines.relative_to(archive_root)).replace("\\", "/")),
                        source_segmentation=np.array(source_segmentation_value),
                        split=np.array(split),
                        category=np.array(category),
                        raw_image_shape=np.array(raw_shape, dtype=np.int32),
                    )

                    record = {
                        "split": split,
                        "category": category,
                        "raw_file": raw_file,
                        "raw_lines": str(source_lines.relative_to(archive_root)).replace("\\", "/"),
                        "raw_segmentation": source_segmentation_value,
                        "output_image": str(output_image.relative_to(output_root)).replace("\\", "/"),
                        "output_label": str(output_label.relative_to(output_root)).replace("\\", "/"),
                        "num_lanes": int(arrays["lanes"].shape[0]),
                        "raw_lane_count": len(raw_lanes),
                        "dropped_lane_count": int(dropped),
                    }
                    manifest_writer.write(record)
                    writers[split].write(record)
                    split_counts[split] += 1
                    lane_count_hist[split][int(arrays["lanes"].shape[0])] += 1
                    dropped_lane_count += int(dropped)
                    if category:
                        category_counts[category] += 1

    write_dataset_yaml(
        output_root,
        output_shape=output_shape,
        raw_shape=raw_shape,
        num_points=int(args.num_points),
        fixed_y_start_px=float(args.fixed_y_start_px),
        fixed_y_end_px=float(args.fixed_y_end_px),
    )
    summary = {
        "dataset": "CULane",
        "archive_root": str(archive_root),
        "output_root": str(output_root),
        "image_shape_hw": list(output_shape),
        "raw_image_shape_hw": list(raw_shape),
        "selected_splits": list(selected_splits),
        "point_mode": "fixed_y",
        "num_points": int(args.num_points),
        "fixed_y_original_h": int(raw_shape[0]),
        "fixed_y_start_px": float(args.fixed_y_start_px),
        "fixed_y_end_px": float(args.fixed_y_end_px),
        "line_width": int(args.line_width),
        "image_source": "reused_preprocessed" if reuse_images_root is not None else "raw_archive",
        "reuse_images_root": str(reuse_images_root) if reuse_images_root is not None else None,
        "reuse_image_mode": str(args.reuse_image_mode) if reuse_images_root is not None else None,
        "split_counts": dict(split_counts),
        "category_counts": dict(category_counts),
        "lane_count_hist": {
            split: {str(key): int(value) for key, value in sorted(hist.items())}
            for split, hist in lane_count_hist.items()
        },
        "dropped_lane_count": int(dropped_lane_count),
        "missing_segmentation_count": int(missing_segmentation_count),
        "image_shape_checked": int(image_shape_checked),
        "manifest": str(manifest_path),
        "split_manifests": {split: str(path) for split, path in split_manifest_paths.items()},
    }
    (output_root / "manifests" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = convert(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
