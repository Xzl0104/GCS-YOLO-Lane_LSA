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
DEFAULT_SUMMARY = ROOT / "runs" / "gcs_lane" / "tusimple_fixed_y_k56_official_h_samples.json"
DEFAULT_VAL_GT_JSON = (
    ROOT
    / "runs"
    / "gcs_lane"
    / "tusimple_official_val_363_folder_aware_seed20260602_subset"
    / "labels"
    / "tusimple_official_val_363_folder_aware_seed20260602.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the TuSimple Q12-K56 fixed-y dataset from original TuSimple JSON and images, "
            "using an official-val raw_file JSON as the default train/val split manifest."
        )
    )
    parser.add_argument("--archive-root", default="archive/TUSimple", help="TuSimple root or archive directory.")
    parser.add_argument(
        "--val-gt-json",
        default=str(DEFAULT_VAL_GT_JSON),
        help="Official-val json-lines file whose raw_file entries define the validation split.",
    )
    parser.add_argument(
        "--reference-root",
        default=None,
        help="Optional legacy converted split reference root. Explicit migration aid only; not used by default.",
    )
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


def val_gt_raw_file_split(val_gt_json: Path) -> dict[str, str]:
    """Return raw_file -> split using an official-val JSON as the validation manifest."""
    if not val_gt_json.exists():
        raise FileNotFoundError(f"Official-val split manifest not found: {val_gt_json}")
    mapping: dict[str, str] = {}
    with val_gt_json.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            raw_file = normalized_raw_file(str(payload.get("raw_file", "")))
            if not raw_file:
                raise ValueError(f"{val_gt_json}:{lineno} is missing raw_file")
            previous = mapping.get(raw_file)
            if previous is not None:
                raise ValueError(f"raw_file appears more than once in official-val manifest: {raw_file}")
            mapping[raw_file] = "val"
    if not mapping:
        raise ValueError(f"Official-val split manifest is empty: {val_gt_json}")
    return mapping


def normalized_raw_file(value: str) -> str:
    """Normalize TuSimple raw_file strings for split-membership checks."""
    return str(value).replace("\\", "/").lstrip("/")


def sample_raw_file_set(samples: list[Any]) -> set[str]:
    """Return normalized raw_file ids for a split sample list."""
    return {normalized_raw_file(sample.raw_file) for sample in samples}


def assert_disjoint_raw_file_splits(split_samples: dict[str, list[Any]]) -> dict[str, Any]:
    """Fail fast if any raw_file appears in more than one split."""
    split_sets = {split: sample_raw_file_set(samples) for split, samples in split_samples.items()}
    overlaps: dict[str, list[str]] = {}
    split_names = sorted(split_sets)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = sorted(split_sets[left].intersection(split_sets[right]))
            if overlap:
                overlaps[f"{left}_vs_{right}"] = overlap[:10]
    if overlaps:
        raise ValueError(f"TuSimple split raw_file overlap detected: {overlaps}")
    return {
        f"{left}_vs_{right}": 0
        for i, left in enumerate(split_names)
        for right in split_names[i + 1 :]
    }


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
    reference_root = (
        ROOT / args.reference_root if args.reference_root and not Path(args.reference_root).is_absolute() else Path(args.reference_root)
    ) if args.reference_root else None
    val_gt_json = ROOT / args.val_gt_json if not Path(args.val_gt_json).is_absolute() else Path(args.val_gt_json)
    output_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    img_shape = normalize_imgsz(args.imgsz)

    ensure_dataset_dirs(output_root, include_test=True)
    removed = clear_output_splits(output_root, ("train", "val", "test")) if args.overwrite else {}

    ref_split = reference_raw_file_split(reference_root) if reference_root is not None else val_gt_raw_file_split(val_gt_json)
    split_samples: dict[str, list[Any]] = {"train": [], "val": [], "test": []}
    missing_reference: list[str] = []

    for sample in load_train_samples(archive_root):
        raw_file = normalized_raw_file(sample.raw_file)
        if reference_root is not None and raw_file not in ref_split:
            missing_reference.append(raw_file)
            continue
        split = ref_split.get(raw_file, "train")
        split_samples[split].append(sample)
    if missing_reference:
        raise ValueError(f"{len(missing_reference)} train JSON samples are missing from the reference split. First: {missing_reference[0]}")
    if reference_root is not None:
        rebuilt_train_val = len(split_samples["train"]) + len(split_samples["val"])
        if rebuilt_train_val != len(ref_split):
            raise FileNotFoundError(
                "Archive train JSON did not cover the reference train/val split: "
                f"rebuilt={rebuilt_train_val}, reference={len(ref_split)}. "
                "Check that archive/TUSimple/train_set contains the original TuSimple label_data_*.json files."
            )
    else:
        missing_val = sorted(raw_file for raw_file, split in ref_split.items() if split == "val" and raw_file not in sample_raw_file_set(split_samples["val"]))
        if missing_val:
            raise FileNotFoundError(
                "Archive train JSON did not cover the official-val split manifest: "
                f"missing={len(missing_val)}, first={missing_val[0]}"
            )

    split_samples["test"] = load_test_samples(archive_root)
    if not split_samples["test"]:
        raise FileNotFoundError(f"No TuSimple test samples found under {archive_root}")
    split_overlap_counts = assert_disjoint_raw_file_splits(split_samples)

    print(f"archive_root: {archive_root}")
    print(f"split_manifest: {reference_root if reference_root is not None else val_gt_json}")
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
        "val_gt_json": val_gt_json if reference_root is None else None,
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
        "split_source": "reference_raw_file_membership" if reference_root is not None else "official_val_gt_json_raw_file_membership",
        "reference_raw_files": len(ref_split),
        "splits": {
            split: {
                "source_samples": len(split_samples[split]),
                "output_label_hist": split_label_hist(output_root, split),
                "stem_check": split_stem_check(output_root, split),
            }
            for split in ("train", "val", "test")
        },
        "split_raw_file_overlap_counts": split_overlap_counts,
        "removed_before_rebuild": removed,
        "export_summary": export_summary,
    }

    summary_path = ROOT / args.summary if not Path(args.summary).is_absolute() else Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
