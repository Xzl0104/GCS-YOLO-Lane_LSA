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

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_utils import (  # noqa: E402
    TuSimpleSample,
    find_archive_root,
    load_test_samples,
    load_train_samples,
    split_train_val,
)
from tools.convert_tusimple_to_gcs import convert_one  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive" / "TUSimple"
DEFAULT_OUTPUT = ROOT / "datasets" / "tusimple_final_train_fixed_y_k56_960x544_0530_500_test_2282"
DEFAULT_SEED = 20260829
FIXED_Y_START = 710.0 / 720.0
FIXED_Y_END = 160.0 / 720.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a final TuSimple train/test split with the current train+val data "
            "and 500 official 0530 test samples moved into train."
        )
    )
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-count-to-train", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--imgsz", nargs=2, type=int, default=[544, 960], metavar=("H", "W"))
    parser.add_argument("--line-width", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-base", type=int, default=0, help="Smoke-test limit for merged train+val samples.")
    parser.add_argument("--limit-test", type=int, default=0, help="Smoke-test limit for remaining test samples.")
    return parser.parse_args()


def path_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def day_from_raw_file(raw_file: str) -> str:
    parts = Path(raw_file.lstrip("/").replace("\\", "/")).parts
    return parts[1] if len(parts) >= 2 and parts[0] == "clips" else "unknown"


def sample_record(sample: TuSimpleSample, *, destination_split: str, source_split: str) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "raw_file": sample.raw_file,
        "day": day_from_raw_file(sample.raw_file),
        "frame": Path(sample.raw_file).stem,
        "source_split": source_split,
        "destination_split": destination_split,
        "label_source": "official_train_json" if source_split == "train" else "official_test_json",
    }


def convert_samples(
    samples: list[TuSimpleSample],
    destination_split: str,
    output_root: Path,
    img_shape: tuple[int, int],
    line_width: int,
    description: str,
) -> None:
    for sample in tqdm(samples, desc=description):
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing source image: {sample.image_path}")
        convert_one(
            destination_split,
            sample,
            output_root,
            img_shape=img_shape,
            num_points=56,
            line_width=line_width,
            point_mode="fixed_y",
            fixed_y_start=FIXED_Y_START,
            fixed_y_end=FIXED_Y_END,
        )


def write_dataset_yaml(output_root: Path) -> None:
    content = """# Final training split: original train + current val + 500 official 0530 test samples.
# The remaining official test samples are kept under images/test for final evaluation only.
train: images/train
val: null
test: images/test
nc: 1
names: [lane]
point_mode: fixed_y
fixed_y: [0.9861111111111112, 0.2222222222222222]
imgsz: [544, 960]
"""
    (output_root / "dataset.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    archive_root = find_archive_root(path_from_root(args.archive_root))
    output_root = path_from_root(args.output_root)
    img_shape = normalize_imgsz(args.imgsz)

    if args.test_count_to_train <= 0:
        raise ValueError(f"--test-count-to-train must be positive, got {args.test_count_to_train}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to rebuild it.")
        resolved = output_root.resolve()
        if resolved == ROOT.resolve() or ROOT.resolve() not in resolved.parents:
            raise ValueError(f"Refusing to overwrite unsafe output path: {output_root}")
        shutil.rmtree(output_root)

    (output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output_root / "labels_gcs" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "labels_gcs" / "test").mkdir(parents=True, exist_ok=True)

    all_train = load_train_samples(archive_root)
    split = split_train_val(all_train, val_ratio=0.1, split_seed=0, group_by_clip=True)
    merged_base = sorted(split["train"] + split["val"], key=lambda sample: sample.sample_id)
    if args.limit_base > 0:
        merged_base = merged_base[: args.limit_base]

    all_test = sorted(load_test_samples(archive_root), key=lambda sample: sample.sample_id)
    if not all_test:
        raise FileNotFoundError(f"No official test samples found under {archive_root}")
    for sample in all_test:
        if Path(sample.raw_file).stem != "20":
            raise ValueError(f"Expected official test frame 20.jpg, got {sample.raw_file}")

    test_0530 = [sample for sample in all_test if day_from_raw_file(sample.raw_file) == "0530"]
    if args.test_count_to_train > len(test_0530):
        raise ValueError(
            f"Requested {args.test_count_to_train} samples from 0530, but only {len(test_0530)} are available."
        )
    selected_to_train = sorted(
        random.Random(args.seed).sample(test_0530, args.test_count_to_train),
        key=lambda sample: sample.sample_id,
    )
    selected_raw_files = {sample.raw_file for sample in selected_to_train}
    remaining_test = [sample for sample in all_test if sample.raw_file not in selected_raw_files]
    if args.limit_test > 0:
        remaining_test = remaining_test[: args.limit_test]

    base_ids = {sample.raw_file for sample in merged_base}
    if base_ids.intersection(sample.raw_file for sample in all_test):
        raise ValueError("Train and official test raw_file sets overlap before the requested test move.")
    if len(selected_to_train) + len(remaining_test) != len(all_test) and args.limit_test <= 0:
        raise AssertionError("Selected and remaining test samples do not partition the original test set.")

    print(f"archive_root={archive_root}")
    print(f"original_train={len(split['train'])} original_val={len(split['val'])} merged_base={len(merged_base)}")
    print(f"official_test={len(all_test)} official_0530={len(test_0530)}")
    print(f"selected_test_0530_to_train={len(selected_to_train)} seed={args.seed}")
    print(f"remaining_test={len(remaining_test)}")
    print(f"output_root={output_root}")

    convert_samples(
        merged_base,
        "train",
        output_root,
        img_shape=img_shape,
        line_width=args.line_width,
        description="Converting original train+val to final train",
    )
    convert_samples(
        selected_to_train,
        "train",
        output_root,
        img_shape=img_shape,
        line_width=args.line_width,
        description="Moving official 0530 test samples into train",
    )
    convert_samples(
        remaining_test,
        "test",
        output_root,
        img_shape=img_shape,
        line_width=args.line_width,
        description="Converting remaining official test samples",
    )

    write_dataset_yaml(output_root)
    base_records = [
        sample_record(
            sample,
            destination_split="train",
            source_split="val" if sample in split["val"] else "train",
        )
        for sample in merged_base
    ]
    moved_records = [sample_record(sample, destination_split="train", source_split="test") for sample in selected_to_train]
    test_records = [sample_record(sample, destination_split="test", source_split="test") for sample in remaining_test]
    manifest = {
        "created_date": "2026-08-29",
        "dataset_type": "final_train_test_split_with_official_test_subset_moved_to_train",
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
            "merged_base_count": len(merged_base),
        },
        "test_move_protocol": {
            "source_json": "test_label.json",
            "source_split": "test",
            "candidate_filter": "raw_file day is 0530 and frame is 20.jpg",
            "candidate_count": len(test_0530),
            "moved_count": len(selected_to_train),
            "seed": args.seed,
            "selection": "random.sample after sorting by sample_id",
            "moved_label_method": "official_test_json_direct_fixed_y_k56_conversion",
            "test_count_before": len(all_test),
            "test_count_after": len(remaining_test),
            "test_reduction": len(all_test) - len(remaining_test),
        },
        "counts": {
            "train_images": len(base_records) + len(moved_records),
            "test_images": len(test_records),
            "total_images": len(base_records) + len(moved_records) + len(test_records),
            "train_lane_count_hist": dict(
                sorted(Counter(len(sample.lanes) for sample in merged_base + selected_to_train).items())
            ),
            "test_lane_count_hist": dict(sorted(Counter(len(sample.lanes) for sample in remaining_test).items())),
            "train_source_hist": dict(
                sorted(Counter(record["source_split"] for record in base_records + moved_records).items())
            ),
        },
        "train_base_records": base_records,
        "moved_test_0530_records": moved_records,
        "test_records": test_records,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "moved_test_0530_raw_files.json").write_text(
        json.dumps([sample.raw_file for sample in selected_to_train], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(f"manifest={output_root / 'manifest.json'}")
    print(f"dataset_yaml={output_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
