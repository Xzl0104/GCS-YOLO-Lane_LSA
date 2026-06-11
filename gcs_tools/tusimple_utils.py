from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from gcs_tools.label_utils import lane_to_points, sort_lane_bottom_to_top
from ultralytics.utils.gcs_shape import normalize_imgsz


TRAIN_JSON_NAMES = (
    "label_data_0313.json",
    "label_data_0531.json",
    "label_data_0601.json",
)
TEST_JSON_NAMES = (
    "test_label.json",
    "test_label_new.json",
)


@dataclass(frozen=True)
class TuSimpleSample:
    sample_id: str
    raw_file: str
    image_path: Path
    lanes: list[list[tuple[float, float]]]
    split: str


def find_archive_root(path: str | Path) -> Path:
    """Resolve either archive/ or archive/TUSimple to the TuSimple root."""
    root = Path(path)
    if (root / "train_set").exists() and (root / "test_set").exists():
        return root
    candidate = root / "TUSimple"
    if (candidate / "train_set").exists() and (candidate / "test_set").exists():
        return candidate
    raise FileNotFoundError(f"Cannot find TuSimple train_set/test_set under {root}")


def read_json_lines(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_lanes(sample: dict) -> list[list[tuple[float, float]]]:
    h_samples = sample["h_samples"]
    lanes: list[list[tuple[float, float]]] = []
    for lane_xs in sample["lanes"]:
        points = lane_to_points(lane_xs, h_samples)
        if len(points) >= 2:
            lanes.append(sort_lane_bottom_to_top(points))
    return lanes


def _resolve_image(root: Path, raw_file: str, source_split: str) -> Path:
    rel = raw_file.lstrip("/").replace("\\", "/")
    split_root = root / ("test_set" if source_split == "test" else "train_set")
    return split_root / rel


def _make_sample_id(source_split: str, raw_file: str, idx: int) -> str:
    rel = raw_file.lstrip("/").replace("\\", "/")
    parts = Path(rel).parts
    if len(parts) >= 4 and parts[0] == "clips":
        day = parts[1]
        clip = parts[2]
        frame = Path(parts[3]).stem
        return f"{source_split}_{day}_{clip}_{frame}"
    return f"{source_split}_{idx:06d}"


def load_train_samples(root: Path) -> list[TuSimpleSample]:
    all_samples: list[TuSimpleSample] = []
    idx = 0
    for name in TRAIN_JSON_NAMES:
        path = root / "train_set" / name
        if not path.exists():
            continue
        for raw in read_json_lines(path):
            raw_file = raw["raw_file"]
            sample = TuSimpleSample(
                sample_id=_make_sample_id("train", raw_file, idx),
                raw_file=raw_file,
                image_path=_resolve_image(root, raw_file, "train"),
                lanes=parse_lanes(raw),
                split="train",
            )
            all_samples.append(sample)
            idx += 1
    return all_samples


def load_test_samples(root: Path) -> list[TuSimpleSample]:
    candidates = [root / name for name in TEST_JSON_NAMES]
    candidates.append(root.parent / "test_label_new.json")
    candidates.append(root / "train_set" / "seg_label" / "test.json")

    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        return []

    samples: list[TuSimpleSample] = []
    for idx, raw in enumerate(read_json_lines(json_path)):
        raw_file = raw["raw_file"]
        sample = TuSimpleSample(
            sample_id=_make_sample_id("test", raw_file, idx),
            raw_file=raw_file,
            image_path=_resolve_image(root, raw_file, "test"),
            lanes=parse_lanes(raw),
            split="test",
        )
        samples.append(sample)
    return samples


def lane_count_histogram(samples: Iterable[TuSimpleSample]) -> dict[int, int]:
    """Count samples by the number of valid lane instances."""
    hist: dict[int, int] = {}
    for sample in samples:
        n = len(sample.lanes)
        hist[n] = hist.get(n, 0) + 1
    return dict(sorted(hist.items()))


def tusimple_group_id(raw_file: str) -> str:
    """Return a TuSimple clip group id from a raw_file path."""
    rel = raw_file.lstrip("/").replace("\\", "/")
    parts = Path(rel).parts
    if len(parts) >= 3 and parts[0] == "clips":
        return f"{parts[1]}/{parts[2]}"
    return rel


def _split_train_val_by_lane_count(
    samples: list[TuSimpleSample],
    val_ratio: float,
    split_seed: int,
) -> dict[str, list[TuSimpleSample]]:
    """Create a deterministic lane-count-stratified sample-level train/val split."""
    val_ratio = min(max(float(val_ratio), 0.0), 0.9)
    n_val = int(round(len(samples) * val_ratio))
    n_val = max(1, n_val) if len(samples) > 1 and val_ratio > 0 else 0
    if n_val == 0:
        return {"train": sorted(samples, key=lambda x: x.sample_id), "val": []}

    rng = random.Random(int(split_seed))
    groups: dict[int, list[TuSimpleSample]] = defaultdict(list)
    for sample in samples:
        groups[len(sample.lanes)].append(sample)

    train: list[TuSimpleSample] = []
    val: list[TuSimpleSample] = []
    for lane_count in sorted(groups):
        group = groups[lane_count][:]
        rng.shuffle(group)
        group_val = int(round(len(group) * val_ratio))
        if group_val == 0 and len(group) > 1:
            group_val = 1
        group_val = min(group_val, max(len(group) - 1, 0))
        val.extend(group[:group_val])
        train.extend(group[group_val:])

    if len(val) > n_val:
        rng.shuffle(val)
        overflow = val[n_val:]
        val = val[:n_val]
        train.extend(overflow)
    elif len(val) < n_val:
        rng.shuffle(train)
        need = min(n_val - len(val), len(train))
        val.extend(train[:need])
        train = train[need:]

    return {
        "train": sorted(train, key=lambda x: x.sample_id),
        "val": sorted(val, key=lambda x: x.sample_id),
    }


def _split_train_val_by_group(
    samples: list[TuSimpleSample],
    val_ratio: float,
    split_seed: int,
) -> dict[str, list[TuSimpleSample]]:
    """Create a deterministic train/val split where a TuSimple clip group appears in only one split."""
    val_ratio = min(max(float(val_ratio), 0.0), 0.9)
    n_val = int(round(len(samples) * val_ratio))
    n_val = max(1, n_val) if len(samples) > 1 and val_ratio > 0 else 0
    if n_val == 0:
        return {"train": sorted(samples, key=lambda x: x.sample_id), "val": []}

    grouped: dict[str, list[TuSimpleSample]] = defaultdict(list)
    for sample in samples:
        grouped[tusimple_group_id(sample.raw_file)].append(sample)

    rng = random.Random(int(split_seed))
    group_items = list(grouped.items())
    rng.shuffle(group_items)

    val_groups: set[str] = set()
    val_count = 0
    for group_id, group_samples in group_items:
        if val_count >= n_val:
            break
        if len(samples) - (val_count + len(group_samples)) <= 0:
            continue
        val_groups.add(group_id)
        val_count += len(group_samples)

    train = [sample for sample in samples if tusimple_group_id(sample.raw_file) not in val_groups]
    val = [sample for sample in samples if tusimple_group_id(sample.raw_file) in val_groups]
    return {
        "train": sorted(train, key=lambda x: x.sample_id),
        "val": sorted(val, key=lambda x: x.sample_id),
    }


def split_train_val(
    samples: list[TuSimpleSample],
    val_ratio: float = 0.1,
    split_seed: int = 0,
    group_by_clip: bool = True,
) -> dict[str, list[TuSimpleSample]]:
    """Create a deterministic train/val split.

    TuSimple JSON files are ordered by drive sequence. Taking the final 10%
    as validation can put entire lane-count/domain modes into val only, which
    makes structured lane training look much worse than the model actually is.
    The default is clip-group splitting so adjacent frames/groups do not cross
    the train/val boundary.
    """
    if not samples:
        return {"train": [], "val": []}
    if group_by_clip:
        return _split_train_val_by_group(samples, val_ratio=val_ratio, split_seed=split_seed)
    return _split_train_val_by_lane_count(samples, val_ratio=val_ratio, split_seed=split_seed)


def ensure_dataset_dirs(output_root: Path, include_test: bool = True) -> None:
    splits = ["train", "val"] + (["test"] if include_test else [])
    for split in splits:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels_gcs" / split).mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def resize_image_and_lanes(
    image: np.ndarray,
    lanes: list[list[tuple[float, float]]],
    img_size: int | tuple[int, int] | list[int],
) -> tuple[np.ndarray, list[list[tuple[float, float]]]]:
    out_h, out_w = normalize_imgsz(img_size)
    h, w = image.shape[:2]
    resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    sx = float(out_w) / float(w)
    sy = float(out_h) / float(h)
    scaled_lanes: list[list[tuple[float, float]]] = []
    for lane in lanes:
        scaled = [(float(x) * sx, float(y) * sy) for x, y in lane]
        if len(scaled) >= 2:
            scaled_lanes.append(sort_lane_bottom_to_top(scaled))
    return resized, scaled_lanes


def iter_split_samples(
    root: Path,
    include_test: bool,
    val_ratio: float,
    split_seed: int = 0,
    group_by_clip: bool = True,
) -> Iterable[tuple[str, TuSimpleSample]]:
    split_map = split_train_val(
        load_train_samples(root),
        val_ratio=val_ratio,
        split_seed=split_seed,
        group_by_clip=group_by_clip,
    )
    for split in ("train", "val"):
        for sample in split_map[split]:
            yield split, sample
    if include_test:
        for sample in load_test_samples(root):
            yield "test", sample
