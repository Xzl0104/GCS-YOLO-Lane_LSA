"""Audit the converted CULane fixed-y GCS dataset before training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.utils.gcs_fixed_y import build_fixed_y_anchors, validate_fixed_y_contract


SPLITS = ("train", "val", "test")
TEST_CATEGORIES = (
    "Normal",
    "Crowded",
    "Dazzle",
    "Shadow",
    "No line",
    "Arrow",
    "Curve",
    "Crossroad",
    "Night",
)
CULANE_RAW_SHAPE = (590, 1640)
CULANE_FIXED_Y_ORIGINAL_H = 590
CULANE_FIXED_Y_START_PX = 589.0
CULANE_FIXED_Y_END_PX = 39.0
CULANE_MAX_LANES = 4


def parse_args() -> argparse.Namespace:
    """Parse CULane label-audit arguments."""
    parser = argparse.ArgumentParser(description="Validate a converted CULane fixed-y GCS dataset.")
    parser.add_argument(
        "--dataset-root",
        default=str(ROOT / "datasets" / "culane_fixed_y_590x960"),
        help="Converted root containing images/<split> and labels_gcs/<split>.",
    )
    parser.add_argument(
        "--archive-root",
        default=None,
        help="Optional extracted CULane archive root. Enables source image and .lines.txt checks.",
    )
    parser.add_argument(
        "--source-lines-root",
        default=None,
        help=(
            "Optional root containing official .lines.txt files at their CULane-relative paths. "
            "Use this when the conversion host retains source annotations but not raw source JPGs."
        ),
    )
    parser.add_argument("--imgsz", nargs=2, type=int, metavar=("H", "W"), default=(384, 960))
    parser.add_argument("--num-points", type=int, default=56)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Deep-check at most this many labels per split; 0 deep-checks every label.",
    )
    parser.add_argument("--save-json", default=None, help="Optional validation report path.")
    return parser.parse_args()


def scalar_text(data: Any, key: str, default: str = "") -> str:
    """Read a scalar NPZ metadata field as text."""
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape {value.shape}.")
    return str(value.reshape(-1)[0])


def scalar_number(data: Any, key: str) -> float:
    """Read a scalar NPZ metadata field as a finite number."""
    if key not in data.files:
        raise KeyError(f"missing key: {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape {value.shape}.")
    result = float(value.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{key} must be finite, got {result}.")
    return result


def path_is_relative_to(path: Path, root: Path) -> bool:
    """Return whether a resolved path remains under a resolved root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_archive_relative(archive_root: Path, value: str, field: str) -> Path:
    """Resolve one metadata path while rejecting absolute or escaping values."""
    text = str(value).strip().replace("\\", "/")
    if not text:
        raise ValueError(f"{field} is empty.")
    path = PurePosixPath(text)
    if path.is_absolute() or path.drive or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a safe archive-relative path, got {value!r}.")
    resolved = (archive_root / Path(*path.parts)).resolve()
    if not path_is_relative_to(resolved, archive_root):
        raise ValueError(f"{field} escapes archive root: {value!r}.")
    return resolved


def add_error(errors: list[dict[str, Any]], location: Path | str, message: str) -> None:
    """Append one structured audit error."""
    errors.append({"path": str(location), "error": str(message)})


def expected_fixed_y(num_points: int) -> np.ndarray:
    """Return the canonical CULane bottom-to-top normalized anchors."""
    return build_fixed_y_anchors(
        original_h=CULANE_FIXED_Y_ORIGINAL_H,
        start_px=CULANE_FIXED_Y_START_PX,
        end_px=CULANE_FIXED_Y_END_PX,
        k=num_points,
    )


def label_metadata(label_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Read small label metadata needed for every file, even under --max-files."""
    errors: list[str] = []
    metadata: dict[str, Any] = {
        "raw_file": "",
        "split": "",
        "category": "",
        "num_lanes": None,
    }
    try:
        with np.load(label_path, allow_pickle=False) as data:
            for key in ("raw_file", "split", "category"):
                try:
                    metadata[key] = scalar_text(data, key)
                except (KeyError, ValueError) as exc:
                    errors.append(str(exc))
            try:
                lane_count = scalar_number(data, "num_lanes")
                if lane_count < 0 or not np.isclose(lane_count, round(lane_count), atol=1e-6):
                    raise ValueError(f"num_lanes must be a non-negative integer, got {lane_count}.")
                metadata["num_lanes"] = int(round(lane_count))
            except (KeyError, ValueError) as exc:
                errors.append(str(exc))
    except Exception as exc:
        errors.append(f"could not read NPZ metadata: {type(exc).__name__}: {exc}")
    return metadata, errors


def inspect_label(
    label_path: Path,
    image_path: Path | None,
    expected_shape: tuple[int, int],
    num_points: int,
    archive_root: Path | None,
    source_lines_root: Path | None,
) -> list[str]:
    """Deep-check label arrays, paired output image, and optional archive source paths."""
    errors: list[str] = []
    anchors = expected_fixed_y(num_points)
    try:
        with np.load(label_path, allow_pickle=False) as data:
            required = {
                "lanes",
                "lane_valid",
                "semantic_mask",
                "edge_mask",
                "num_lanes",
                "point_mode",
                "fixed_y",
                "fixed_y_original_h",
                "fixed_y_start_px",
                "fixed_y_end_px",
                "num_points",
                "image_shape",
                "raw_image_shape",
                "raw_file",
                "source_lines",
                "split",
                "category",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                return [f"missing required keys: {missing}"]

            lanes = np.asarray(data["lanes"], dtype=np.float32)
            lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
            semantic_mask = np.asarray(data["semantic_mask"])
            edge_mask = np.asarray(data["edge_mask"])
            fixed_y = np.asarray(data["fixed_y"], dtype=np.float32).reshape(-1)
            image_shape = np.asarray(data["image_shape"], dtype=np.int64).reshape(-1)
            raw_image_shape = np.asarray(data["raw_image_shape"], dtype=np.int64).reshape(-1)
            raw_file = scalar_text(data, "raw_file")
            source_lines = scalar_text(data, "source_lines")
            point_mode = scalar_text(data, "point_mode").lower()
            declared_lanes = scalar_number(data, "num_lanes")
            declared_points = scalar_number(data, "num_points")
            original_h = scalar_number(data, "fixed_y_original_h")
            start_px = scalar_number(data, "fixed_y_start_px")
            end_px = scalar_number(data, "fixed_y_end_px")
    except Exception as exc:
        return [f"could not read NPZ arrays: {type(exc).__name__}: {exc}"]

    if point_mode != "fixed_y":
        errors.append(f"point_mode={point_mode!r}, expected 'fixed_y'.")
    if lanes.ndim != 3 or lanes.shape[1:] != (num_points, 2):
        errors.append(f"lanes shape {lanes.shape} != (N,{num_points},2).")
    if lane_valid.shape != lanes.shape[:2]:
        errors.append(f"lane_valid shape {lane_valid.shape} != {lanes.shape[:2]}.")
    if semantic_mask.shape != expected_shape:
        errors.append(f"semantic_mask shape {semantic_mask.shape} != {expected_shape}.")
    if edge_mask.shape != expected_shape:
        errors.append(f"edge_mask shape {edge_mask.shape} != {expected_shape}.")
    if tuple(image_shape.tolist()) != expected_shape:
        errors.append(f"image_shape {tuple(image_shape.tolist())} != {expected_shape}.")
    if tuple(raw_image_shape.tolist()) != CULANE_RAW_SHAPE:
        errors.append(f"raw_image_shape {tuple(raw_image_shape.tolist())} != {CULANE_RAW_SHAPE}.")
    if not np.isclose(declared_points, num_points, atol=1e-6):
        errors.append(f"num_points={declared_points} != {num_points}.")
    if not np.isclose(original_h, CULANE_FIXED_Y_ORIGINAL_H, atol=1e-6):
        errors.append(f"fixed_y_original_h={original_h} != {CULANE_FIXED_Y_ORIGINAL_H}.")
    if not np.isclose(start_px, CULANE_FIXED_Y_START_PX, atol=1e-6):
        errors.append(f"fixed_y_start_px={start_px} != {CULANE_FIXED_Y_START_PX}.")
    if not np.isclose(end_px, CULANE_FIXED_Y_END_PX, atol=1e-6):
        errors.append(f"fixed_y_end_px={end_px} != {CULANE_FIXED_Y_END_PX}.")
    try:
        validate_fixed_y_contract(
            fixed_y,
            original_h=CULANE_FIXED_Y_ORIGINAL_H,
            start_px=CULANE_FIXED_Y_START_PX,
            end_px=CULANE_FIXED_Y_END_PX,
            k=num_points,
            name=f"{label_path}: fixed_y",
        )
    except ValueError as exc:
        errors.append(str(exc))
    if fixed_y.shape != anchors.shape or not np.allclose(fixed_y, anchors, atol=1e-6):
        errors.append("fixed_y does not equal the canonical 589..39/590 anchor sequence.")

    if lanes.ndim == 3 and lanes.shape[1:] == (num_points, 2) and lane_valid.shape == lanes.shape[:2]:
        lane_count = lanes.shape[0]
        if not np.isclose(declared_lanes, lane_count, atol=1e-6):
            errors.append(f"num_lanes={declared_lanes} != lanes.shape[0]={lane_count}.")
        if lane_count > CULANE_MAX_LANES:
            errors.append(f"lane count {lane_count} exceeds CULane maximum {CULANE_MAX_LANES}.")
        if not np.isfinite(lanes).all() or not np.isfinite(lane_valid).all():
            errors.append("lanes or lane_valid contain NaN/Inf.")
        if lanes.size and (lanes.min() < 0.0 or lanes.max() > 1.0):
            errors.append("lanes contain out-of-range normalized coordinates.")
        if lane_valid.size and (lane_valid.min() < 0.0 or lane_valid.max() > 1.0):
            errors.append("lane_valid contains values outside [0,1].")
        if lane_count:
            if not np.allclose(lanes[..., 1], anchors.reshape(1, -1), atol=1e-6):
                errors.append("every CULane lane y coordinate must equal the shared fixed-y anchors.")
            invalid_x = lanes[..., 0][lane_valid <= 0.5]
            if invalid_x.size and not np.allclose(invalid_x, 0.0, atol=1e-6):
                errors.append("fixed-y invalid points must store x=0.")
            valid_per_lane = (lane_valid > 0.5).sum(axis=1)
            if np.any(valid_per_lane < 2):
                errors.append("every stored lane must contain at least two valid fixed-y points.")
    elif 0 <= declared_lanes <= CULANE_MAX_LANES:
        errors.append("cannot compare num_lanes because lanes/lane_valid shapes are malformed.")

    if image_path is not None:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            errors.append(f"could not decode converted image: {image_path}.")
        elif tuple(image.shape[:2]) != expected_shape:
            errors.append(f"converted image shape {image.shape[:2]} != {expected_shape}.")

    if archive_root is not None:
        try:
            source_image = resolve_archive_relative(archive_root, raw_file, "raw_file")
            expected_lines = source_image.with_suffix(".lines.txt")
            source_lines_path = resolve_archive_relative(archive_root, source_lines, "source_lines")
            if source_lines_path != expected_lines:
                errors.append("source_lines does not match raw_file with the .lines.txt suffix.")
            if not source_image.is_file():
                errors.append(f"source image is missing: {source_image}.")
            else:
                raw_image = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
                if raw_image is None:
                    errors.append(f"could not decode source image: {source_image}.")
                elif tuple(raw_image.shape[:2]) != CULANE_RAW_SHAPE:
                    errors.append(f"source image shape {raw_image.shape[:2]} != {CULANE_RAW_SHAPE}.")
            if not source_lines_path.is_file():
                errors.append(f"source .lines.txt is missing: {source_lines_path}.")
        except ValueError as exc:
            errors.append(str(exc))
    elif source_lines_root is not None:
        try:
            expected_lines_rel = str(PurePosixPath(raw_file).with_suffix(".lines.txt"))
            source_lines_path = resolve_archive_relative(source_lines_root, source_lines, "source_lines")
            expected_lines_path = resolve_archive_relative(
                source_lines_root,
                expected_lines_rel,
                "expected source_lines",
            )
            if source_lines_path != expected_lines_path:
                errors.append("source_lines does not match raw_file with the .lines.txt suffix.")
            if not source_lines_path.is_file():
                errors.append(f"source .lines.txt is missing: {source_lines_path}.")
        except ValueError as exc:
            errors.append(str(exc))

    return errors


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """Validate all split metadata and a configurable number of full labels."""
    root = Path(args.dataset_root).expanduser().resolve()
    archive_root = Path(args.archive_root).expanduser().resolve() if args.archive_root else None
    source_lines_root = (
        Path(args.source_lines_root).expanduser().resolve() if args.source_lines_root else None
    )
    expected_shape = (int(args.imgsz[0]), int(args.imgsz[1]))
    num_points = int(args.num_points)
    if expected_shape != (384, 960):
        raise ValueError(f"CULane training image shape must be H,W=(384, 960), got {expected_shape}.")
    if num_points != 56:
        raise ValueError(f"CULane fixed-y contract requires K=56, got {num_points}.")
    if archive_root is not None and not archive_root.is_dir():
        raise FileNotFoundError(f"CULane archive root does not exist: {archive_root}.")
    if source_lines_root is not None and not source_lines_root.is_dir():
        raise FileNotFoundError(f"CULane source-lines root does not exist: {source_lines_root}.")

    all_errors: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    raw_files_by_split: dict[str, set[str]] = {}

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels_gcs" / split
        images = sorted(path for path in image_dir.glob("*.jpg") if path.is_file())
        labels = sorted(path for path in label_dir.glob("*.npz") if path.is_file())
        image_by_stem = {path.stem: path for path in images}
        label_by_stem = {path.stem: path for path in labels}
        missing_labels = sorted(set(image_by_stem).difference(label_by_stem))
        missing_images = sorted(set(label_by_stem).difference(image_by_stem))
        for stem in missing_labels:
            add_error(all_errors, image_by_stem[stem], "missing paired .npz label.")
        for stem in missing_images:
            add_error(all_errors, label_by_stem[stem], "missing paired .jpg image.")

        lane_hist: Counter[int] = Counter()
        categories: Counter[str] = Counter()
        raw_files: set[str] = set()
        metadata_error_count = 0
        for label_path in labels:
            metadata, metadata_errors = label_metadata(label_path)
            if metadata_errors:
                metadata_error_count += len(metadata_errors)
                for error in metadata_errors:
                    add_error(all_errors, label_path, error)
                continue

            raw_file = str(metadata["raw_file"])
            split_field = str(metadata["split"])
            category = str(metadata["category"])
            lane_count = int(metadata["num_lanes"])
            lane_hist[lane_count] += 1
            categories[category] += 1
            try:
                resolve_archive_relative(root, raw_file, "raw_file")
            except ValueError as exc:
                metadata_error_count += 1
                add_error(all_errors, label_path, str(exc))
            else:
                raw_files.add(raw_file.replace("\\", "/"))
            if split_field != split:
                metadata_error_count += 1
                add_error(all_errors, label_path, f"split={split_field!r}, expected {split!r}.")
            if lane_count > CULANE_MAX_LANES:
                metadata_error_count += 1
                add_error(
                    all_errors,
                    label_path,
                    f"num_lanes={lane_count} exceeds CULane maximum {CULANE_MAX_LANES}.",
                )
            if split in {"train", "val"} and category:
                metadata_error_count += 1
                add_error(all_errors, label_path, f"{split} category must be empty, got {category!r}.")
            if split == "test" and category not in TEST_CATEGORIES:
                metadata_error_count += 1
                add_error(all_errors, label_path, f"invalid test category {category!r}.")

        labels_to_check = labels if args.max_files <= 0 else labels[: int(args.max_files)]
        for label_path in labels_to_check:
            errors = inspect_label(
                label_path,
                image_by_stem.get(label_path.stem),
                expected_shape=expected_shape,
                num_points=num_points,
                archive_root=archive_root,
                source_lines_root=source_lines_root,
            )
            for error in errors:
                add_error(all_errors, label_path, error)

        split_summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "checked_labels": len(labels_to_check),
            "missing_label_count": len(missing_labels),
            "missing_image_count": len(missing_images),
            "missing_labels_for_images": missing_labels[:10],
            "missing_images_for_labels": missing_images[:10],
            "lane_count_hist": {str(key): int(value) for key, value in sorted(lane_hist.items())},
            "category_hist": {key: int(value) for key, value in sorted(categories.items())},
            "metadata_error_count": metadata_error_count,
        }
        raw_files_by_split[split] = raw_files

    raw_file_leakage: dict[str, Any] = {}
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1 :]:
            overlap = sorted(raw_files_by_split[first].intersection(raw_files_by_split[second]))
            raw_file_leakage[f"{first}_vs_{second}"] = {"count": len(overlap), "examples": overlap[:10]}
            for raw_file in overlap:
                add_error(all_errors, f"{first}/{second}", f"raw_file leaks across splits: {raw_file}.")

    test_categories = Counter(split_summary["test"]["category_hist"])
    has_test_labels = bool(split_summary["test"]["labels"])
    if has_test_labels:
        invalid_categories = sorted(set(test_categories).difference(TEST_CATEGORIES))
        missing_categories = sorted(set(TEST_CATEGORIES).difference(test_categories))
        test_category_audit = "validated"
        for category in invalid_categories:
            add_error(all_errors, "test category audit", f"invalid category present: {category!r}.")
        for category in missing_categories:
            add_error(all_errors, "test category audit", f"missing required category: {category!r}.")
    else:
        # Train/val-only roots are valid before the final held-out test set is prepared.
        invalid_categories = []
        missing_categories = []
        test_category_audit = "skipped_no_test_labels"

    return {
        "dataset": "CULane",
        "dataset_root": str(root),
        "archive_root": str(archive_root) if archive_root is not None else None,
        "source_lines_root": str(source_lines_root) if source_lines_root is not None else None,
        "expected_image_shape_hw": list(expected_shape),
        "expected_raw_image_shape_hw": list(CULANE_RAW_SHAPE),
        "fixed_y": {
            "point_mode": "fixed_y",
            "num_points": num_points,
            "original_h": CULANE_FIXED_Y_ORIGINAL_H,
            "start_px": CULANE_FIXED_Y_START_PX,
            "end_px": CULANE_FIXED_Y_END_PX,
        },
        "splits": split_summary,
        "raw_file_leakage": raw_file_leakage,
        "test_category_counts": dict(sorted(test_categories.items())),
        "test_category_audit": test_category_audit,
        "missing_test_categories": missing_categories,
        "invalid_test_categories": invalid_categories,
        "error_count": len(all_errors),
        "error_examples": all_errors[:30],
    }


def main() -> None:
    """Run the audit, write the report, and return a non-zero exit code on contract failure."""
    args = parse_args()
    report = validate(args)
    save_json = Path(args.save_json) if args.save_json else Path(args.dataset_root) / "manifests" / "validation.json"
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["error_count"]:
        raise SystemExit(1)
    print(f"saved: {save_json.resolve()}")


if __name__ == "__main__":
    main()
