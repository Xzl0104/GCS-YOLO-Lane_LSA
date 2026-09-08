from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.label_utils import remove_duplicate_y, sort_lane_bottom_to_top
from gcs_tools.tusimple_utils import (
    find_archive_root,
    load_train_samples,
    parse_lanes,
    read_json_lines,
    resize_image_and_lanes,
    split_train_val,
)
from tools.convert_tusimple_to_gcs import (
    build_gcs_arrays,
    convert_one,
    validate_gcs_arrays,
)
from ultralytics.utils.gcs_shape import normalize_imgsz


DEFAULT_ARCHIVE = ROOT / "archive" / "TUSimple"
DEFAULT_OUTPUT = ROOT / "datasets" / "tusimple_final_train_fixed_y_k56_960x544_0530_1000"
DEFAULT_SEED = 20260829
FIXED_Y_START = 710.0 / 720.0
FIXED_Y_END = 160.0 / 720.0
EXTRA_LABEL_SOURCE = "pseudo_label_optical_flow_with_reference_endpoint_extension_from_test_frame_20"
ENDPOINT_EXTENSION_METHOD = "reference_20_local_tangent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a final train-only TuSimple GCS dataset with 0530 optical-flow pseudo labels."
    )
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extra-source", type=Path, default=None, help="Optional test_set/clips/0530 directory.")
    parser.add_argument("--test-label", type=Path, default=None, help="Optional test_label.json path.")
    parser.add_argument("--extra-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--imgsz", nargs=2, type=int, default=[544, 960], metavar=("H", "W"))
    parser.add_argument("--flow-fb-thr", type=float, default=4.0)
    parser.add_argument("--line-width", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-extra-only",
        action="store_true",
        help="Regenerate the manifest-selected 0530 pseudo labels without replacing base samples or visualizations.",
    )
    parser.add_argument("--limit-base", type=int, default=0, help="Smoke-test limit for rebuilt train+val samples.")
    parser.add_argument("--limit-extra", type=int, default=0, help="Smoke-test limit for selected 0530 samples.")
    return parser.parse_args()


def path_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_0530_label_map(test_label_path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for raw in read_json_lines(test_label_path):
        rel = raw["raw_file"].lstrip("/").replace("\\", "/")
        parts = Path(rel).parts
        if len(parts) == 4 and parts[:2] == ("clips", "0530") and parts[3] == "20.jpg":
            labels[parts[2]] = raw
    return labels


def collect_candidates(source_root: Path, label_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for clip_dir in sorted(source_root.iterdir()):
        if not clip_dir.is_dir() or clip_dir.name not in label_map:
            continue
        for image_path in sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem)):
            if image_path.stem == "20":
                continue
            candidates.append(
                {
                    "clip": clip_dir.name,
                    "frame": int(image_path.stem),
                    "image_path": image_path,
                    "raw_file": f"clips/0530/{clip_dir.name}/{image_path.name}",
                }
            )
    return candidates


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def track_points(
    source_gray: np.ndarray,
    target_gray: np.ndarray,
    points: np.ndarray,
    fb_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 2:
        return points.copy(), np.zeros(points.shape[0], dtype=bool)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        source_gray,
        target_gray,
        points.reshape(-1, 1, 2).astype(np.float32),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return points.copy(), np.zeros(points.shape[0], dtype=bool)
    tracked = tracked.reshape(-1, 2)
    valid = status.reshape(-1).astype(bool)
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        target_gray,
        source_gray,
        tracked.reshape(-1, 1, 2).astype(np.float32),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if backward is None or back_status is None:
        return tracked, np.zeros(points.shape[0], dtype=bool)
    backward = backward.reshape(-1, 2)
    valid &= back_status.reshape(-1).astype(bool)
    valid &= np.linalg.norm(backward - points, axis=1) <= float(fb_thr)
    valid &= np.isfinite(tracked).all(axis=1)
    valid &= (tracked[:, 0] >= 0.0) & (tracked[:, 0] < source_gray.shape[1])
    valid &= (tracked[:, 1] >= 0.0) & (tracked[:, 1] < source_gray.shape[0])
    return tracked, valid


def append_to_boundary(
    points: np.ndarray,
    boundary_y: float,
    *,
    extend_bottom: bool,
    image_width: int,
    tangent_points: int = 6,
) -> np.ndarray:
    """Extend one lane endpoint to a target y using its local x(y) tangent."""
    if points.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float32)

    endpoint_index = 0 if extend_bottom else -1
    endpoint = points[endpoint_index]
    remaining = float(boundary_y - endpoint[1])
    if (extend_bottom and remaining <= 1e-3) or (not extend_bottom and remaining >= -1e-3):
        return np.empty((0, 2), dtype=np.float32)

    local = points[: min(tangent_points, len(points))] if extend_bottom else points[-min(tangent_points, len(points)) :]
    local = np.asarray(
        remove_duplicate_y([(float(x), float(y)) for x, y in local]),
        dtype=np.float32,
    )
    if local.shape[0] < 2 or float(np.ptp(local[:, 1])) < 1e-3:
        return np.empty((0, 2), dtype=np.float32)

    slope = float(np.polyfit(local[:, 1], local[:, 0], deg=1)[0])
    y_steps = np.abs(np.diff(local[:, 1]))
    step = float(np.median(y_steps[y_steps > 1e-3])) if np.any(y_steps > 1e-3) else 10.0
    step = float(np.clip(step, 5.0, 20.0))

    direction = 1.0 if extend_bottom else -1.0
    distance = abs(remaining)
    offsets = np.arange(step, distance, step, dtype=np.float32)
    ys = endpoint[1] + direction * offsets
    ys = np.append(ys, np.float32(boundary_y))

    extension: list[tuple[float, float]] = []
    previous = endpoint.astype(np.float32)
    for y in ys:
        x = float(endpoint[0] + slope * (float(y) - float(endpoint[1])))
        candidate = np.array([x, float(y)], dtype=np.float32)
        if 0.0 <= x < float(image_width):
            extension.append((x, float(y)))
            previous = candidate
            continue

        boundary_x = 0.0 if x < 0.0 else float(image_width - 1)
        delta_x = float(candidate[0] - previous[0])
        if abs(delta_x) > 1e-6:
            fraction = float(np.clip((boundary_x - previous[0]) / delta_x, 0.0, 1.0))
            boundary_crossing_y = float(previous[1] + fraction * (candidate[1] - previous[1]))
            extension.append((boundary_x, boundary_crossing_y))
        break
    return np.asarray(extension, dtype=np.float32)


def extend_lane_to_reference_coverage(
    lane: np.ndarray,
    reference_lane: list[tuple[float, float]],
    image_width: int,
) -> list[tuple[float, float]]:
    """Preserve the tracked segment and restore only its missing reference-covered endpoints."""
    current = np.asarray(
        remove_duplicate_y(sort_lane_bottom_to_top([(float(x), float(y)) for x, y in lane])),
        dtype=np.float32,
    )
    reference = np.asarray(
        remove_duplicate_y(sort_lane_bottom_to_top(reference_lane)),
        dtype=np.float32,
    )
    if current.shape[0] < 2 or reference.shape[0] < 2:
        return [(float(x), float(y)) for x, y in current]

    bottom_extension = append_to_boundary(
        current,
        boundary_y=float(reference[0, 1]),
        extend_bottom=True,
        image_width=image_width,
    )
    if bottom_extension.size:
        current = np.concatenate((bottom_extension[::-1], current), axis=0)

    top_extension = append_to_boundary(
        current,
        boundary_y=float(reference[-1, 1]),
        extend_bottom=False,
        image_width=image_width,
    )
    if top_extension.size:
        current = np.concatenate((current, top_extension), axis=0)

    return remove_duplicate_y(sort_lane_bottom_to_top([(float(x), float(y)) for x, y in current]))


def propagate_clip_lanes(
    source_root: Path,
    clip: str,
    source_row: dict[str, Any],
    target_frame: int,
    fb_thr: float,
) -> tuple[list[list[tuple[float, float]]], float]:
    source_lanes = parse_lanes(source_row)
    lane_points = [np.asarray(sort_lane_bottom_to_top(lane), dtype=np.float32) for lane in source_lanes]
    current_gray = read_gray(source_root / clip / "20.jpg")
    total_points = sum(len(points) for points in lane_points)
    tracked_points = 0

    for frame in range(19, target_frame - 1, -1):
        target_gray = read_gray(source_root / clip / f"{frame}.jpg")
        tracked_lanes: list[np.ndarray] = []
        valid_lanes: list[np.ndarray] = []
        all_old: list[np.ndarray] = []
        all_new: list[np.ndarray] = []
        for points in lane_points:
            tracked, valid = track_points(current_gray, target_gray, points, fb_thr=fb_thr)
            tracked_lanes.append(tracked)
            valid_lanes.append(valid)
            if valid.any():
                all_old.append(points[valid])
                all_new.append(tracked[valid])

        if all_old:
            old_points = np.concatenate(all_old, axis=0)
            new_points = np.concatenate(all_new, axis=0)
            global_delta = np.median(new_points - old_points, axis=0)
        else:
            global_delta = np.zeros(2, dtype=np.float32)

        next_lanes: list[np.ndarray] = []
        for old_points, tracked, valid in zip(lane_points, tracked_lanes, valid_lanes):
            tracked_points += int(valid.sum())
            valid_idx = np.flatnonzero(valid)
            if valid_idx.size >= 2:
                # Interpolate local flow displacement rather than absolute coordinates.
                # This keeps failed endpoint points at their own longitudinal positions.
                deltas = tracked[valid] - old_points[valid]
                sample_idx = np.arange(len(old_points), dtype=np.float32)
                filled = old_points.copy()
                filled[:, 0] += np.interp(sample_idx, valid_idx, deltas[:, 0])
                filled[:, 1] += np.interp(sample_idx, valid_idx, deltas[:, 1])
                next_lanes.append(filled.astype(np.float32))
            elif valid_idx.size == 1:
                delta = tracked[valid_idx[0]] - old_points[valid_idx[0]]
                shifted = old_points.copy()
                shifted += delta
                next_lanes.append(shifted.astype(np.float32))
            else:
                shifted = old_points.copy()
                shifted += global_delta
                next_lanes.append(shifted.astype(np.float32))
        lane_points = next_lanes
        current_gray = target_gray

    lanes: list[list[tuple[float, float]]] = []
    for points, reference_lane in zip(lane_points, source_lanes):
        if points.shape[0] >= 2:
            extended = extend_lane_to_reference_coverage(
                points,
                reference_lane,
                image_width=current_gray.shape[1],
            )
            if len(extended) >= 2:
                lanes.append(extended)
    transitions = 20 - target_frame
    expected_points = total_points * max(transitions, 1)
    ratio = float(tracked_points / expected_points) if expected_points else 0.0
    return lanes, ratio


def write_extra_sample(
    candidate: dict[str, Any],
    source_row: dict[str, Any],
    source_root: Path,
    output_root: Path,
    img_shape: tuple[int, int],
    line_width: int,
    fb_thr: float,
    flow_cache: dict[tuple[str, int], tuple[list[list[tuple[float, float]]], float]],
) -> dict[str, Any]:
    clip = candidate["clip"]
    frame = candidate["frame"]
    cache_key = (clip, frame)
    if cache_key not in flow_cache:
        flow_cache[cache_key] = propagate_clip_lanes(
            source_root,
            clip,
            source_row,
            target_frame=frame,
            fb_thr=fb_thr,
        )
    lanes, flow_ratio = flow_cache[cache_key]
    image = cv2.imread(str(candidate["image_path"]))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {candidate['image_path']}")
    resized, scaled_lanes = resize_image_and_lanes(image, lanes, img_shape)
    arrays = build_gcs_arrays(
        scaled_lanes,
        img_shape=img_shape,
        num_points=56,
        line_width=line_width,
        point_mode="fixed_y",
        fixed_y_start=FIXED_Y_START,
        fixed_y_end=FIXED_Y_END,
    )
    validate_gcs_arrays(arrays, img_shape=img_shape, num_points=56)

    sample_id = f"extra_0530_{clip}_{frame:02d}"
    image_path = output_root / "images" / "train" / f"{sample_id}.jpg"
    label_path = output_root / "labels_gcs" / "train" / f"{sample_id}.npz"
    if not cv2.imwrite(str(image_path), resized):
        raise OSError(f"Failed to write image: {image_path}")
    np.savez_compressed(
        label_path,
        **arrays,
        raw_file=np.array(candidate["raw_file"]),
        source_label_raw_file=np.array(f"clips/0530/{clip}/20.jpg"),
        label_source=np.array(EXTRA_LABEL_SOURCE),
        endpoint_extension_method=np.array(ENDPOINT_EXTENSION_METHOD),
        flow_valid_ratio=np.array([flow_ratio], dtype=np.float32),
        image_shape=np.array(img_shape, dtype=np.int32),
        num_points=np.array([56], dtype=np.int32),
    )
    return {
        "sample_id": sample_id,
        "raw_file": candidate["raw_file"],
        "source_label_raw_file": f"clips/0530/{clip}/20.jpg",
        "frame": frame,
        "clip": clip,
        "flow_valid_ratio": flow_ratio,
        "label_source": EXTRA_LABEL_SOURCE,
        "endpoint_extension_method": ENDPOINT_EXTENSION_METHOD,
        "num_lanes": int(arrays["lanes"].shape[0]),
        "image": str(image_path.relative_to(ROOT)),
        "label": str(label_path.relative_to(ROOT)),
    }


def write_dataset_yaml(output_root: Path) -> None:
    content = """# Train-only dataset. The former val split was merged into train.
# The 0530 non-20 frames use optical-flow pseudo labels derived from test frame 20.
train: images/train
val: null
nc: 1
names: [lane]
point_mode: fixed_y
fixed_y: [0.9861111111111112, 0.2222222222222222]
imgsz: [544, 960]
"""
    (output_root / "dataset.yaml").write_text(content, encoding="utf-8")


def refresh_extra_only(
    output_root: Path,
    source_root: Path,
    label_map: dict[str, dict[str, Any]],
    img_shape: tuple[int, int],
    line_width: int,
    fb_thr: float,
) -> None:
    """Replace only the existing 0530 extra records, preserving all other dataset artifacts."""
    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest required for --refresh-extra-only: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_records = manifest.get("extra_records")
    if not isinstance(prior_records, list) or not prior_records:
        raise ValueError(f"Manifest has no extra_records to refresh: {manifest_path}")

    flow_cache: dict[tuple[str, int], tuple[list[list[tuple[float, float]]], float]] = {}
    refreshed_records: list[dict[str, Any]] = []
    for record in tqdm(prior_records, desc="Refreshing 0530 pseudo labels"):
        clip = str(record["clip"])
        frame = int(record["frame"])
        image_path = source_root / clip / f"{frame}.jpg"
        if clip not in label_map:
            raise KeyError(f"Missing 20.jpg test label for 0530 clip {clip}")
        if not image_path.exists():
            raise FileNotFoundError(f"Missing selected 0530 image: {image_path}")
        refreshed_records.append(
            write_extra_sample(
                {
                    "clip": clip,
                    "frame": frame,
                    "image_path": image_path,
                    "raw_file": f"clips/0530/{clip}/{frame}.jpg",
                },
                label_map[clip],
                source_root,
                output_root,
                img_shape=img_shape,
                line_width=line_width,
                fb_thr=fb_thr,
                flow_cache=flow_cache,
            )
        )

    manifest["extra_records"] = refreshed_records
    manifest["extra_sampling"]["label_method"] = (
        "optical_flow_backward_with_reference_20_local_tangent_endpoint_extension"
    )
    manifest["extra_sampling"]["endpoint_extension_method"] = ENDPOINT_EXTENSION_METHOD
    manifest["extra_sampling"]["flow_fb_threshold_pixels"] = fb_thr
    manifest["counts"]["extra_images"] = len(refreshed_records)
    manifest["counts"]["total_images"] = int(manifest["counts"]["base_images"]) + len(refreshed_records)
    manifest["counts"]["extra_lane_count_hist"] = dict(
        sorted(Counter(record["num_lanes"] for record in refreshed_records).items())
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(f"refreshed_extra={len(refreshed_records)}")
    print(f"manifest={manifest_path}")


def main() -> None:
    args = parse_args()
    archive_root = find_archive_root(path_from_root(args.archive_root))
    output_root = path_from_root(args.output_root)
    source_root = path_from_root(args.extra_source) if args.extra_source else archive_root / "test_set" / "clips" / "0530"
    test_label_path = path_from_root(args.test_label) if args.test_label else archive_root / "test_label.json"
    img_shape = normalize_imgsz(args.imgsz)

    if args.refresh_extra_only:
        if not output_root.exists():
            raise FileNotFoundError(f"Output root does not exist for --refresh-extra-only: {output_root}")
        label_map = load_0530_label_map(test_label_path)
        refresh_extra_only(
            output_root,
            source_root,
            label_map,
            img_shape=img_shape,
            line_width=args.line_width,
            fb_thr=args.flow_fb_thr,
        )
        return

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to rebuild it.")
        if output_root.resolve() == ROOT.resolve() or ROOT.resolve() not in output_root.resolve().parents:
            raise ValueError(f"Refusing to overwrite unsafe output path: {output_root}")
        shutil.rmtree(output_root)

    for path in (
        output_root / "images" / "train",
        output_root / "labels_gcs" / "train",
    ):
        path.mkdir(parents=True, exist_ok=True)

    label_map = load_0530_label_map(test_label_path)
    candidates = collect_candidates(source_root, label_map)
    if args.extra_count > len(candidates):
        raise ValueError(f"Requested {args.extra_count} extras, but only {len(candidates)} candidates are available.")
    selected = random.Random(args.seed).sample(candidates, args.extra_count)
    selected.sort(key=lambda item: item["raw_file"])
    if args.limit_extra > 0:
        selected = selected[: args.limit_extra]

    split = split_train_val(load_train_samples(archive_root), val_ratio=0.1, split_seed=0, group_by_clip=True)
    base_samples = sorted(split["train"] + split["val"], key=lambda sample: sample.sample_id)
    if args.limit_base > 0:
        base_samples = base_samples[: args.limit_base]

    print(f"archive_root={archive_root}")
    print(f"base_samples={len(base_samples)} (train={len(split['train'])}, val={len(split['val'])})")
    print(f"0530_labelled_clips={len(label_map)} candidates_excluding_20={len(candidates)}")
    print(f"selected_extra={len(selected)} seed={args.seed}")
    print(f"output_root={output_root}")

    base_records: list[dict[str, Any]] = []
    for sample in tqdm(base_samples, desc="Rebuilding base train+val labels"):
        convert_one(
            "train",
            sample,
            output_root,
            img_shape=img_shape,
            num_points=56,
            line_width=args.line_width,
            point_mode="fixed_y",
            fixed_y_start=FIXED_Y_START,
            fixed_y_end=FIXED_Y_END,
        )
        base_records.append(
            {
                "sample_id": sample.sample_id,
                "raw_file": sample.raw_file,
                "source_split": "train" if sample in split["train"] else "val",
                "label_source": "official_train_json",
            }
        )

    flow_cache: dict[tuple[str, int], tuple[list[list[tuple[float, float]]], float]] = {}
    extra_records: list[dict[str, Any]] = []
    for candidate in tqdm(selected, desc="Generating 0530 pseudo labels"):
        extra_records.append(
            write_extra_sample(
                candidate,
                label_map[candidate["clip"]],
                source_root,
                output_root,
                img_shape=img_shape,
                line_width=args.line_width,
                fb_thr=args.flow_fb_thr,
                flow_cache=flow_cache,
            )
        )

    write_dataset_yaml(output_root)
    manifest = {
        "created_date": "2026-08-29",
        "dataset_type": "train_only_final_training_dataset",
        "archive_root": str(archive_root),
        "output_root": str(output_root),
        "image_shape_hw": list(img_shape),
        "point_mode": "fixed_y",
        "num_points": 56,
        "fixed_y_pixels_desc": list(range(710, 159, -10)),
        "base_split_protocol": {
            "source_jsons": ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"],
            "val_ratio": 0.1,
            "split_seed": 0,
            "group_by_clip": True,
            "original_train_count": len(split["train"]),
            "original_val_count": len(split["val"]),
            "merged_base_count": len(base_samples),
        },
        "extra_sampling": {
            "source": str(source_root),
            "test_label": str(test_label_path),
            "seed": args.seed,
            "requested_count": args.extra_count,
            "selected_count": len(selected),
            "excluded_rule": "all 0530 clips' 20.jpg frames, which are the official test-labelled frames",
            "candidate_count": len(candidates),
            "label_method": "optical_flow_backward_with_reference_20_local_tangent_endpoint_extension",
            "endpoint_extension_method": ENDPOINT_EXTENSION_METHOD,
            "flow_fb_threshold_pixels": args.flow_fb_thr,
            "pseudo_label_warning": "Non-20 frames do not have independent official TuSimple GT in this archive.",
        },
        "counts": {
            "total_images": len(base_records) + len(extra_records),
            "base_images": len(base_records),
            "extra_images": len(extra_records),
            "extra_lane_count_hist": dict(sorted(Counter(r["num_lanes"] for r in extra_records).items())),
        },
        "base_records": base_records,
        "extra_records": extra_records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(f"manifest={output_root / 'manifest.json'}")
    print(f"dataset_yaml={output_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
