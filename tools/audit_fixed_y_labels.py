from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.utils.gcs_fixed_y import validate_training_fixed_y_desc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fixed-y K56 GCS label files.")
    parser.add_argument("label_root", nargs="?", help="labels_gcs directory or one .npz file.")
    parser.add_argument("--data", default=None, help="Dataset YAML with train/val/test image paths.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional per-split file limit for quick checks.")
    parser.add_argument("--check-lane-count", action="store_true", help="Report 2/3/4/5 lane-count histograms.")
    parser.add_argument("--check-fixed-y", action="store_true", help="Validate descending 710..160 fixed-y anchors.")
    parser.add_argument("--check-contiguity", action="store_true", help="Report non-contiguous valid masks.")
    parser.add_argument(
        "--fail-on-missing-2lane",
        action="store_true",
        help="Exit nonzero when the train split has no 2-lane samples.",
    )
    return parser.parse_args()


def collect_files(path: Path, max_files: int = 0) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.rglob("*.npz"))
    if max_files and max_files > 0:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No .npz labels found under {path}.")
    return files


def is_contiguous(valid: np.ndarray) -> bool:
    idx = np.where(valid > 0.5)[0]
    if idx.size == 0:
        return True
    return int((valid[idx.min() : idx.max() + 1] > 0.5).sum()) == int(idx.max() - idx.min() + 1)


def _label_root_for_image_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lowered = [p.lower() for p in parts]
    if "images" in lowered:
        idx = lowered.index("images")
        parts[idx] = "labels_gcs"
        return Path(*parts)
    return image_path.parent / "labels_gcs" / image_path.name


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base_dir / path).resolve()


def label_roots_from_data_yaml(data_yaml: Path) -> dict[str, Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Dataset YAML {data_yaml} must contain a mapping.")
    roots: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        image_path = _resolve_path(data[split], data_yaml.parent)
        roots[split] = _label_root_for_image_path(image_path)
    if not roots:
        raise RuntimeError(f"Dataset YAML {data_yaml} has no train/val/test paths.")
    return roots


def audit_files(
    files: list[Path],
    check_fixed_y: bool = True,
    check_contiguity: bool = True,
) -> dict[str, Any]:
    lane_count_hist: Counter[int] = Counter()
    total_lanes = 0
    noncontiguous_lanes = 0
    hole_points = 0

    for label_file in files:
        with np.load(label_file, allow_pickle=False) as data:
            if "lanes" not in data or "lane_valid" not in data:
                raise KeyError(f"{label_file}: missing lanes or lane_valid.")
            lanes = np.asarray(data["lanes"], dtype=np.float32)
            lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
            if lanes.ndim != 3 or lane_valid.shape != lanes.shape[:2]:
                raise ValueError(f"{label_file}: invalid lanes/lane_valid shapes {lanes.shape}/{lane_valid.shape}.")
            if check_fixed_y and "fixed_y" in data:
                validate_training_fixed_y_desc(data["fixed_y"], name=f"{label_file}: fixed_y")
            kept = (lane_valid > 0.5).sum(axis=1) >= 2
            if check_fixed_y and bool(kept.any()):
                first = int(np.where(kept)[0][0])
                validate_training_fixed_y_desc(lanes[first, :, 1], name=f"{label_file}: lane fixed_y")

        valid_counts = (lane_valid > 0.5).sum(axis=1)
        kept = valid_counts >= 2
        num_lanes = int(kept.sum())
        lane_count_hist[num_lanes] += 1
        total_lanes += num_lanes
        if check_contiguity:
            for valid in lane_valid[kept]:
                idx = np.where(valid > 0.5)[0]
                if idx.size == 0 or is_contiguous(valid):
                    continue
                noncontiguous_lanes += 1
                hole_points += int((idx.max() - idx.min() + 1) - idx.size)

    lane_count_2_to_5 = {str(k): int(lane_count_hist.get(k, 0)) for k in range(2, 6)}
    other = {str(k): int(v) for k, v in sorted(lane_count_hist.items()) if k < 2 or k > 5}
    return {
        "files": len(files),
        "total_lanes": int(total_lanes),
        "lane_count_hist": lane_count_2_to_5,
        "other_lane_count_hist": other,
        "noncontiguous_lanes": int(noncontiguous_lanes),
        "hole_points": int(hole_points),
    }


def main() -> None:
    args = parse_args()
    if not any((args.check_lane_count, args.check_fixed_y, args.check_contiguity)):
        args.check_lane_count = True
        args.check_fixed_y = True
        args.check_contiguity = True

    warnings: list[str] = []
    if args.data:
        data_yaml = Path(args.data)
        if not data_yaml.is_absolute():
            data_yaml = (ROOT / data_yaml).resolve()
        roots = label_roots_from_data_yaml(data_yaml)
        splits = {}
        for split, root in roots.items():
            files = collect_files(root, max_files=args.max_files)
            splits[split] = audit_files(files, check_fixed_y=args.check_fixed_y, check_contiguity=args.check_contiguity)
        train_hist = splits.get("train", {}).get("lane_count_hist", {})
        if train_hist and int(train_hist.get("2", 0)) == 0:
            msg = (
                "WARNING: train split contains 0 two-lane labels. "
                "ordered_slot_v2 class 0 cannot be learned unless 2-lane samples are preserved."
            )
            warnings.append(msg)
            print(msg, file=sys.stderr)
            if args.fail_on_missing_2lane:
                raise RuntimeError(msg)
        output = {
            "data": str(data_yaml),
            "splits": splits,
            "warnings": warnings,
        }
    else:
        if not args.label_root:
            raise ValueError("Provide either label_root or --data.")
        root = Path(args.label_root)
        if not root.is_absolute():
            root = (ROOT / root).resolve()
        files = collect_files(root, max_files=args.max_files)
        output = audit_files(files, check_fixed_y=args.check_fixed_y, check_contiguity=args.check_contiguity)
        output["label_root"] = str(root)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
