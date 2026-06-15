from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.label_utils import TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM, sort_lane_bottom_to_top
from gcs_tools.tusimple_utils import (
    ensure_dataset_dirs,
    find_archive_root,
    load_train_samples,
)
from tools.convert_tusimple_to_gcs import build_gcs_arrays, validate_gcs_arrays
from tools.export_gcs_yolo_labels import export_split
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    raw_file: str
    folder: str
    image_path: Path
    arrays: dict[str, np.ndarray]
    lane_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild TuSimple fixed-y train/val split in-place from archive labels, "
            "then overwrite labels_gcs and YOLO labels."
        )
    )
    parser.add_argument("--archive-root", default="archive/TUSimple", help="TuSimple root or archive directory.")
    parser.add_argument("--dataset-root", default="datasets/tusimple_fixed_y_960x544", help="Converted dataset root.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="Output image shape as H W.")
    parser.add_argument("--num-points", type=int, default=32, help="Fixed-y point count per lane.")
    parser.add_argument(
        "--fixed-y-start",
        type=float,
        default=TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM,
        help="Bottom normalized y anchor. Defaults to TuSimple official h=710 / H=720.",
    )
    parser.add_argument("--fixed-y-end", type=float, default=0.25, help="Top normalized y anchor.")
    parser.add_argument("--val-size", type=int, default=363, help="Number of validation images.")
    parser.add_argument("--seed", type=int, default=20260530, help="Deterministic stratified split seed.")
    parser.add_argument(
        "--folder-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Balance validation sampling across raw_file folders inside each selected lane count. "
            "Use --no-folder-balanced to restore legacy lane-count-only random sampling."
        ),
    )
    parser.add_argument(
        "--folder-index",
        type=int,
        default=1,
        help="raw_file path component used as the balancing folder. For clips/0601/x/20.jpg, index 1 gives 0601.",
    )
    parser.add_argument(
        "--lane-classes",
        default="3,4,5",
        help="Comma-separated lane counts used for validation ratio allocation.",
    )
    parser.add_argument(
        "--val-counts",
        default=None,
        help="Optional explicit validation targets, e.g. 3:150,4:150,5:63. "
        "When omitted, current labels_gcs/val ratio is scaled to --val-size.",
    )
    parser.add_argument("--line-width", type=int, default=12, help="YOLO segmentation rasterization width.")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id.")
    parser.add_argument(
        "--summary",
        default="runs/gcs_lane/tusimple_fixed_y_restratified_20260530.json",
        help="Summary JSON path.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return {str(k): int(v) for k, v in sorted(value.items())}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_int_list(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one lane class")
    return out


def _parse_count_map(value: str | None) -> dict[int, int] | None:
    if value is None:
        return None
    out: dict[int, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, raw_count = item.split(":", maxsplit=1)
        out[int(key.strip())] = int(raw_count.strip())
    if not out:
        raise ValueError("--val-counts was provided but no counts were parsed")
    return out


def _label_lane_count(label_path: Path) -> int:
    with np.load(label_path, allow_pickle=False) as data:
        if "lane_valid" in data.files:
            return int((np.asarray(data["lane_valid"]).sum(axis=1) >= 2).sum())
        if "num_lanes" in data.files:
            return int(np.asarray(data["num_lanes"]).reshape(-1)[0])
        return int(np.asarray(data["lanes"]).shape[0])


def source_folder(raw_file: str, folder_index: int) -> str:
    """Return the raw_file path component used for folder-aware split balancing."""
    parts = str(raw_file).replace("\\", "/").strip("/").split("/")
    if not parts:
        return ""
    if folder_index < 0:
        folder_index = len(parts) + folder_index
    if folder_index < 0 or folder_index >= len(parts):
        raise ValueError(f"raw_file={raw_file!r} has no component at folder_index={folder_index}.")
    return parts[folder_index]


def current_val_hist(dataset_root: Path, lane_classes: list[int]) -> Counter[int]:
    hist: Counter[int] = Counter()
    for label_path in sorted((dataset_root / "labels_gcs" / "val").glob("*.npz")):
        lane_count = _label_lane_count(label_path)
        if lane_count in lane_classes:
            hist[lane_count] += 1
    return hist


def allocate_val_counts(hist: Counter[int], lane_classes: list[int], val_size: int) -> dict[int, int]:
    total = sum(hist[c] for c in lane_classes)
    if total <= 0:
        raise ValueError("Cannot allocate validation counts because current val has no selected lane classes")
    raw = {c: float(val_size) * float(hist[c]) / float(total) for c in lane_classes}
    counts = {c: int(np.floor(raw[c])) for c in lane_classes}
    remain = int(val_size) - sum(counts.values())
    order = sorted(lane_classes, key=lambda c: (raw[c] - counts[c], hist[c], c), reverse=True)
    for c in order[:remain]:
        counts[c] += 1
    return counts


def scaled_lanes_for_image(candidate: Any, img_shape: tuple[int, int]) -> list[list[tuple[float, float]]]:
    image = cv2.imread(str(candidate.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {candidate.image_path}")
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = img_shape
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    scaled: list[list[tuple[float, float]]] = []
    for lane in candidate.lanes:
        points = [(float(x) * sx, float(y) * sy) for x, y in lane]
        if len(points) >= 2:
            scaled.append(sort_lane_bottom_to_top(points))
    return scaled


def build_candidates(
    archive_root: Path,
    img_shape: tuple[int, int],
    num_points: int,
    fixed_y_start: float,
    fixed_y_end: float,
    folder_index: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    samples = load_train_samples(archive_root)
    for sample in tqdm(samples, desc="scan archive train labels"):
        lanes = scaled_lanes_for_image(sample, img_shape=img_shape)
        arrays = build_gcs_arrays(
            lanes,
            img_shape=img_shape,
            num_points=num_points,
            point_mode="fixed_y",
            fixed_y_start=fixed_y_start,
            fixed_y_end=fixed_y_end,
        )
        validate_gcs_arrays(arrays, img_shape=img_shape, num_points=num_points)
        lane_count = int(np.asarray(arrays["num_lanes"]).reshape(-1)[0])
        candidates.append(
            Candidate(
                sample_id=sample.sample_id,
                raw_file=sample.raw_file,
                folder=source_folder(sample.raw_file, folder_index),
                image_path=sample.image_path,
                arrays=arrays,
                lane_count=lane_count,
            )
        )
    return candidates


def even_folder_targets(total: int, available: dict[str, int]) -> dict[str, int]:
    """Allocate validation samples as evenly as possible across folders, capped by availability."""
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    folders = sorted(k for k, v in available.items() if v > 0)
    if total == 0:
        return {}
    if not folders:
        raise ValueError("No folders are available for a positive validation target.")
    if sum(available[k] for k in folders) < total:
        raise ValueError(f"Only {sum(available[k] for k in folders)} records are available, cannot sample {total}.")

    remaining = int(total)
    active = list(folders)
    targets = {k: 0 for k in folders}
    while remaining > 0 and active:
        quota, extra = divmod(remaining, len(active))
        progressed = False
        next_active = []
        for i, folder in enumerate(active):
            want = quota + (1 if i < extra else 0)
            room = available[folder] - targets[folder]
            take = min(want, room)
            if take > 0:
                targets[folder] += take
                remaining -= take
                progressed = True
            if targets[folder] < available[folder]:
                next_active.append(folder)
        active = next_active
        if not progressed:
            break
    if remaining != 0:
        raise ValueError(f"Could not allocate {total} samples across folders; remaining={remaining}.")
    return {k: v for k, v in targets.items() if v > 0}


def split_candidates(
    candidates: list[Candidate],
    val_counts: dict[int, int],
    seed: int,
    folder_balanced: bool,
) -> tuple[list[Candidate], list[Candidate]]:
    rng = random.Random(int(seed))
    by_lane_count: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_lane_count[candidate.lane_count].append(candidate)

    val_ids: set[str] = set()
    for lane_count, target in sorted(val_counts.items()):
        group = by_lane_count.get(lane_count, [])[:]
        if len(group) < target:
            raise ValueError(f"Need {target} validation samples for {lane_count}-lane, found {len(group)}")
        if folder_balanced:
            by_folder: dict[str, list[Candidate]] = defaultdict(list)
            for candidate in group:
                by_folder[candidate.folder].append(candidate)
            folder_targets = even_folder_targets(
                int(target),
                {folder: len(folder_group) for folder, folder_group in by_folder.items()},
            )
            for folder, folder_target in sorted(folder_targets.items()):
                folder_group = sorted(by_folder[folder], key=lambda c: c.sample_id)
                rng.shuffle(folder_group)
                val_ids.update(candidate.sample_id for candidate in folder_group[:folder_target])
        else:
            rng.shuffle(group)
            val_ids.update(candidate.sample_id for candidate in group[:target])

    train = sorted((c for c in candidates if c.sample_id not in val_ids), key=lambda c: c.sample_id)
    val = sorted((c for c in candidates if c.sample_id in val_ids), key=lambda c: c.sample_id)
    return train, val


def split_membership(dataset_root: Path) -> dict[str, str]:
    membership: dict[str, str] = {}
    for split in ("train", "val"):
        for image_path in (dataset_root / "images" / split).glob("*.jpg"):
            membership[image_path.stem] = split
    return membership


def clear_split(dataset_root: Path, split: str) -> dict[str, int]:
    removed: dict[str, int] = {}
    for folder, pattern in (("images", "*.jpg"), ("labels", "*.txt"), ("labels_gcs", "*.npz")):
        directory = dataset_root / folder / split
        directory.mkdir(parents=True, exist_ok=True)
        count = 0
        for path in directory.glob(pattern):
            path.unlink()
            count += 1
        removed[f"{folder}/{split}"] = count
    return removed


def write_candidate(dataset_root: Path, split: str, candidate: Candidate, img_shape: tuple[int, int], num_points: int) -> None:
    image = cv2.imread(str(candidate.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {candidate.image_path}")
    dst_h, dst_w = img_shape
    resized = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    image_path = dataset_root / "images" / split / f"{candidate.sample_id}.jpg"
    label_path = dataset_root / "labels_gcs" / split / f"{candidate.sample_id}.npz"
    if not cv2.imwrite(str(image_path), resized):
        raise OSError(f"Failed to write image: {image_path}")

    validate_gcs_arrays(candidate.arrays, img_shape=img_shape, num_points=num_points)
    np.savez_compressed(
        label_path,
        **candidate.arrays,
        raw_file=np.array(candidate.raw_file),
        image_shape=np.array(img_shape, dtype=np.int32),
        num_points=np.array([num_points], dtype=np.int32),
    )


def write_split(dataset_root: Path, split: str, candidates: list[Candidate], img_shape: tuple[int, int], num_points: int) -> None:
    for candidate in tqdm(candidates, desc=f"write {split}"):
        write_candidate(dataset_root, split, candidate, img_shape=img_shape, num_points=num_points)


def split_hist(candidates: list[Candidate]) -> Counter[int]:
    return Counter(candidate.lane_count for candidate in candidates)


def folder_hist(candidates: list[Candidate]) -> Counter[str]:
    return Counter(candidate.folder for candidate in candidates)


def lane_folder_hist(candidates: list[Candidate]) -> Counter[tuple[int, str]]:
    return Counter((candidate.lane_count, candidate.folder) for candidate in candidates)


def nested_lane_folder_hist(hist: Counter[tuple[int, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for (lane_count, folder), value in sorted(hist.items()):
        out[str(lane_count)][str(folder)] = int(value)
    return dict(out)


def check_stems(dataset_root: Path, split: str) -> dict[str, Any]:
    image_stems = {p.stem for p in (dataset_root / "images" / split).glob("*.jpg")}
    yolo_stems = {p.stem for p in (dataset_root / "labels" / split).glob("*.txt")}
    gcs_stems = {p.stem for p in (dataset_root / "labels_gcs" / split).glob("*.npz")}
    return {
        "images": len(image_stems),
        "labels": len(yolo_stems),
        "labels_gcs": len(gcs_stems),
        "images_vs_labels_missing": sorted(image_stems - yolo_stems)[:10],
        "labels_vs_images_extra": sorted(yolo_stems - image_stems)[:10],
        "images_vs_labels_gcs_missing": sorted(image_stems - gcs_stems)[:10],
        "labels_gcs_vs_images_extra": sorted(gcs_stems - image_stems)[:10],
        "ok": image_stems == yolo_stems == gcs_stems,
    }


def output_label_hist(dataset_root: Path, split: str) -> Counter[int]:
    hist: Counter[int] = Counter()
    for label_path in sorted((dataset_root / "labels_gcs" / split).glob("*.npz")):
        hist[_label_lane_count(label_path)] += 1
    return hist


def main() -> None:
    args = parse_args()
    archive_root = find_archive_root(ROOT / args.archive_root if not Path(args.archive_root).is_absolute() else args.archive_root)
    dataset_root = ROOT / args.dataset_root if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    img_shape = normalize_imgsz(args.imgsz)
    lane_classes = _parse_int_list(args.lane_classes)
    explicit_val_counts = _parse_count_map(args.val_counts)

    ensure_dataset_dirs(dataset_root, include_test=True)
    before_membership = split_membership(dataset_root)
    ratio_source_hist = current_val_hist(dataset_root, lane_classes)
    val_counts = explicit_val_counts or allocate_val_counts(ratio_source_hist, lane_classes, args.val_size)
    if sum(val_counts.values()) != int(args.val_size):
        raise ValueError(f"Validation counts sum to {sum(val_counts.values())}, expected {args.val_size}")

    print(f"archive_root: {archive_root}")
    print(f"dataset_root: {dataset_root}")
    print(f"image shape: {shape_str(img_shape)} (W x H), stored as H,W={img_shape}")
    print(f"seed: {args.seed}")
    print(f"folder_balanced: {bool(args.folder_balanced)}")
    print(f"folder_index: {int(args.folder_index)}")
    print(f"ratio_source_hist: {dict(sorted(ratio_source_hist.items()))}")
    print(f"target_val_counts: {dict(sorted(val_counts.items()))}")

    candidates = build_candidates(
        archive_root=archive_root,
        img_shape=img_shape,
        num_points=args.num_points,
        fixed_y_start=args.fixed_y_start,
        fixed_y_end=args.fixed_y_end,
        folder_index=int(args.folder_index),
    )
    train_candidates, val_candidates = split_candidates(
        candidates,
        val_counts=val_counts,
        seed=args.seed,
        folder_balanced=bool(args.folder_balanced),
    )
    print(f"candidate_hist: {dict(sorted(split_hist(candidates).items()))}")
    print(f"train_hist: {dict(sorted(split_hist(train_candidates).items()))}")
    print(f"val_hist: {dict(sorted(split_hist(val_candidates).items()))}")
    print(f"val_lane_folder_hist: {_jsonable(nested_lane_folder_hist(lane_folder_hist(val_candidates)))}")

    removed: dict[str, int] = {}
    for split in ("train", "val"):
        removed.update(clear_split(dataset_root, split))
    write_split(dataset_root, "train", train_candidates, img_shape=img_shape, num_points=args.num_points)
    write_split(dataset_root, "val", val_candidates, img_shape=img_shape, num_points=args.num_points)

    export_summary: dict[str, Any] = {}
    for split in ("train", "val"):
        export_summary[split] = export_split(
            dataset_root=dataset_root,
            split=split,
            img_shape=img_shape,
            line_width=args.line_width,
            class_id=args.class_id,
        )
        print(f"{split} export: {export_summary[split]}")

    after_membership = split_membership(dataset_root)
    changed_split = sum(
        1
        for stem, split in after_membership.items()
        if stem in before_membership and before_membership[stem] != split
    )
    new_or_rebuilt = sum(1 for stem in after_membership if stem not in before_membership)

    summary = {
        "archive_root": archive_root,
        "dataset_root": dataset_root,
        "img_shape_hw": img_shape,
        "fixed_y": [float(args.fixed_y_start), float(args.fixed_y_end)],
        "num_points": int(args.num_points),
        "seed": int(args.seed),
        "split_strategy": "lane_count_folder" if bool(args.folder_balanced) else "lane_count",
        "folder_balanced": bool(args.folder_balanced),
        "folder_index": int(args.folder_index),
        "val_size": int(args.val_size),
        "lane_classes": lane_classes,
        "ratio_source_hist": ratio_source_hist,
        "target_val_counts": val_counts,
        "candidate_hist": split_hist(candidates),
        "candidate_folder_hist": folder_hist(candidates),
        "candidate_lane_folder_hist": nested_lane_folder_hist(lane_folder_hist(candidates)),
        "splits": {
            "train": {
                "total": len(train_candidates),
                "lane_count_hist": split_hist(train_candidates),
                "folder_hist": folder_hist(train_candidates),
                "lane_folder_hist": nested_lane_folder_hist(lane_folder_hist(train_candidates)),
                "output_label_hist": output_label_hist(dataset_root, "train"),
                "stem_check": check_stems(dataset_root, "train"),
            },
            "val": {
                "total": len(val_candidates),
                "lane_count_hist": split_hist(val_candidates),
                "folder_hist": folder_hist(val_candidates),
                "lane_folder_hist": nested_lane_folder_hist(lane_folder_hist(val_candidates)),
                "output_label_hist": output_label_hist(dataset_root, "val"),
                "stem_check": check_stems(dataset_root, "val"),
            },
            "test": {
                "output_label_hist": output_label_hist(dataset_root, "test"),
                "stem_check": check_stems(dataset_root, "test"),
            },
        },
        "removed_before_rebuild": removed,
        "changed_train_val_split_membership": changed_split,
        "new_or_rebuilt_train_val_stems": new_or_rebuilt,
        "export_summary": export_summary,
    }

    summary_path = ROOT / args.summary if not Path(args.summary).is_absolute() else Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
