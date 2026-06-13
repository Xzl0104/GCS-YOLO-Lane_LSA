from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.label_utils import TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM
from gcs_tools.tusimple_utils import ensure_dataset_dirs, find_archive_root, load_test_samples, load_train_samples
from tools.convert_tusimple_to_gcs import convert_one
from tools.export_gcs_yolo_labels import export_split
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str


TUSIMPLE_OFFICIAL_TOP_Y_NORM = 160.0 / 720.0
DEFAULT_OUTPUT_ROOT = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544"
DEFAULT_REFERENCE_ROOT = ROOT / "datasets" / "tusimple_fixed_y_960x544"
DEFAULT_SUMMARY = ROOT / "runs" / "gcs_lane" / "tusimple_fixed_y_k56_official_h_samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the TuSimple Q12-K56 fixed-y dataset from original TuSimple JSON and images, "
            "using the current converted dataset only as the train/val split reference."
        )
    )
    parser.add_argument("--archive-root", default="archive/TUSimple", help="TuSimple root or archive directory.")
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT), help="Existing split reference root.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Destination converted dataset root.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="Output image shape as H W.")
    parser.add_argument("--num-points", type=int, default=56, help="Fixed-y point count per lane.")
    parser.add_argument(
        "--fixed-y-start",
        type=float,
        default=TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM,
        help="Bottom normalized y anchor. Defaults to TuSimple official h=710 / H=720.",
    )
    parser.add_argument(
        "--fixed-y-end",
        type=float,
        default=TUSIMPLE_OFFICIAL_TOP_Y_NORM,
        help="Top normalized y anchor. Defaults to TuSimple official h=160 / H=720.",
    )
    parser.add_argument("--line-width", type=int, default=12, help="YOLO segmentation rasterization width.")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Summary JSON path.")
    parser.add_argument("--overwrite", action="store_true", help="Clear existing output split files before writing.")
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


def _array_scalar_str(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def reference_raw_file_split(reference_root: Path) -> dict[str, str]:
    """Return raw_file -> split from an existing converted dataset."""
    mapping: dict[str, str] = {}
    for split in ("train", "val"):
        label_dir = reference_root / "labels_gcs" / split
        labels = sorted(label_dir.glob("*.npz"))
        if not labels:
            raise FileNotFoundError(f"No reference labels found under {label_dir}")
        for label_path in labels:
            with np.load(label_path, allow_pickle=False) as data:
                if "raw_file" not in data.files:
                    raise KeyError(f"{label_path} is missing raw_file; cannot preserve split membership.")
                raw_file = _array_scalar_str(data["raw_file"]).replace("\\", "/").lstrip("/")
            previous = mapping.get(raw_file)
            if previous is not None and previous != split:
                raise ValueError(f"raw_file appears in both {previous} and {split}: {raw_file}")
            mapping[raw_file] = split
    return mapping


def clear_output_splits(output_root: Path, splits: tuple[str, ...]) -> dict[str, int]:
    removed: dict[str, int] = {}
    for split in splits:
        for folder, pattern in (("images", "*.jpg"), ("labels", "*.txt"), ("labels_gcs", "*.npz")):
            directory = output_root / folder / split
            directory.mkdir(parents=True, exist_ok=True)
            count = 0
            for path in directory.glob(pattern):
                path.unlink()
                count += 1
            removed[f"{folder}/{split}"] = count
    return removed


def label_lane_count(label_path: Path) -> int:
    with np.load(label_path, allow_pickle=False) as data:
        lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
    return int((lane_valid.sum(axis=1) >= 2).sum())


def split_label_hist(output_root: Path, split: str) -> Counter[int]:
    hist: Counter[int] = Counter()
    for label_path in sorted((output_root / "labels_gcs" / split).glob("*.npz")):
        hist[label_lane_count(label_path)] += 1
    return hist


def split_stem_check(output_root: Path, split: str) -> dict[str, Any]:
    image_stems = {p.stem for p in (output_root / "images" / split).glob("*.jpg")}
    yolo_stems = {p.stem for p in (output_root / "labels" / split).glob("*.txt")}
    gcs_stems = {p.stem for p in (output_root / "labels_gcs" / split).glob("*.npz")}
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


def convert_split_samples(
    split_samples: dict[str, list[Any]],
    output_root: Path,
    img_shape: tuple[int, int],
    num_points: int,
    fixed_y_start: float,
    fixed_y_end: float,
) -> None:
    for split in ("train", "val", "test"):
        for sample in tqdm(split_samples[split], desc=f"write K56 {split}"):
            if not sample.image_path.exists():
                raise FileNotFoundError(f"Missing source image for {sample.raw_file}: {sample.image_path}")
            convert_one(
                split,
                sample,
                output_root,
                img_shape=img_shape,
                num_points=num_points,
                point_mode="fixed_y",
                fixed_y_start=fixed_y_start,
                fixed_y_end=fixed_y_end,
            )


def main() -> None:
    args = parse_args()
    if int(args.num_points) != 56:
        raise ValueError("This K56 builder is intentionally pinned to --num-points 56.")

    archive_root = find_archive_root(ROOT / args.archive_root if not Path(args.archive_root).is_absolute() else args.archive_root)
    reference_root = ROOT / args.reference_root if not Path(args.reference_root).is_absolute() else Path(args.reference_root)
    output_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    img_shape = normalize_imgsz(args.imgsz)

    ensure_dataset_dirs(output_root, include_test=True)
    removed = clear_output_splits(output_root, ("train", "val", "test")) if args.overwrite else {}

    ref_split = reference_raw_file_split(reference_root)
    split_samples: dict[str, list[Any]] = {"train": [], "val": [], "test": []}
    missing_reference: list[str] = []

    for sample in load_train_samples(archive_root):
        raw_file = sample.raw_file.replace("\\", "/").lstrip("/")
        split = ref_split.get(raw_file)
        if split is None:
            missing_reference.append(raw_file)
            continue
        split_samples[split].append(sample)
    if missing_reference:
        raise ValueError(f"{len(missing_reference)} train JSON samples are missing from the reference split. First: {missing_reference[0]}")

    split_samples["test"] = load_test_samples(archive_root)
    if not split_samples["test"]:
        raise FileNotFoundError(f"No TuSimple test samples found under {archive_root}")

    print(f"archive_root: {archive_root}")
    print(f"reference_root: {reference_root}")
    print(f"output_root: {output_root}")
    print(f"image shape: {shape_str(img_shape)} (W x H), stored as H,W={img_shape}")
    print(f"fixed_y: [{float(args.fixed_y_start)}, {float(args.fixed_y_end)}], num_points={int(args.num_points)}")
    print({split: len(samples) for split, samples in split_samples.items()})

    convert_split_samples(
        split_samples=split_samples,
        output_root=output_root,
        img_shape=img_shape,
        num_points=int(args.num_points),
        fixed_y_start=float(args.fixed_y_start),
        fixed_y_end=float(args.fixed_y_end),
    )

    export_summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        export_summary[split] = export_split(
            dataset_root=output_root,
            split=split,
            img_shape=img_shape,
            line_width=int(args.line_width),
            class_id=int(args.class_id),
        )
        print(f"{split} export: {export_summary[split]}")

    summary = {
        "archive_root": archive_root,
        "reference_root": reference_root,
        "output_root": output_root,
        "img_shape_hw": list(img_shape),
        "point_mode": "fixed_y",
        "fixed_y": [float(args.fixed_y_start), float(args.fixed_y_end)],
        "num_points": int(args.num_points),
        "official_h_samples": {
            "bottom": 710,
            "top": 160,
            "step": 10,
            "count": 56,
            "order": "bottom_to_top",
        },
        "split_source": "reference_raw_file_membership",
        "reference_raw_files": len(ref_split),
        "splits": {
            split: {
                "source_samples": len(split_samples[split]),
                "output_label_hist": split_label_hist(output_root, split),
                "stem_check": split_stem_check(output_root, split),
            }
            for split in ("train", "val", "test")
        },
        "removed_before_rebuild": removed,
        "export_summary": export_summary,
    }

    summary_path = ROOT / args.summary if not Path(args.summary).is_absolute() else Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
