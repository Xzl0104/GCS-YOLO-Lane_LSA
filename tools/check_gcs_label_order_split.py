from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_utils import tusimple_group_id
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str


REQUIRED_KEYS = ("lanes", "lane_valid", "num_lanes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GCS label point order and TuSimple split grouping.")
    parser.add_argument("--dataset-root", default="datasets/tusimple_fixed_y_960x544", help="Converted dataset root.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to inspect.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="Expected GCS image shape as H W. Defaults to the dataset preset.",
    )
    parser.add_argument(
        "--max-order-files",
        type=int,
        default=0,
        help="Maximum labels per split for point-order checks. 0 checks every label.",
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="Tolerance for descending-y point order.")
    parser.add_argument("--save-dir", default="runs/gcs_lane/label_split_check", help="Directory for summary.json.")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return {str(k): int(v) for k, v in sorted(value.items())}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_scalar_str(value: np.ndarray) -> str:
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return str(value)


def _day_id(raw_file: str) -> str:
    rel = raw_file.lstrip("/").replace("\\", "/")
    parts = Path(rel).parts
    if len(parts) >= 2 and parts[0] == "clips":
        return parts[1]
    return "unknown"


def _select_for_order_check(labels: list[Path], max_files: int) -> list[Path]:
    if max_files <= 0 or max_files >= len(labels):
        return labels
    return labels[:max_files]


def check_label(
    label_path: Path,
    expected_imgsz: tuple[int, int],
    eps: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    with np.load(label_path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            return {}, [], [f"missing keys: {missing}"]

        lanes = data["lanes"]
        lane_valid = data["lane_valid"]
        num_lanes = data["num_lanes"]
        raw_file = _array_scalar_str(data["raw_file"]) if "raw_file" in data else ""
        image_shape = tuple(int(x) for x in data["image_shape"].reshape(-1)) if "image_shape" in data else None
        point_mode = _array_scalar_str(data["point_mode"]).lower() if "point_mode" in data else "free"
        if point_mode in {"fixed-y", "fixedy"}:
            point_mode = "fixed_y"
        fixed_y = data["fixed_y"].astype(np.float32) if "fixed_y" in data else None

    errors: list[str] = []
    bad_order: list[dict[str, Any]] = []
    img_h, img_w = expected_imgsz

    if image_shape is not None and image_shape != (img_h, img_w):
        errors.append(f"image_shape {image_shape} != {(img_h, img_w)}")
    if lanes.ndim != 3 or lanes.shape[-1] != 2:
        errors.append(f"lanes shape must be N,K,2, got {lanes.shape}")
    if lane_valid.shape != lanes.shape[:2]:
        errors.append(f"lane_valid shape {lane_valid.shape} != {lanes.shape[:2]}")
    if int(num_lanes.reshape(-1)[0]) != int(lanes.shape[0]):
        errors.append(f"num_lanes {int(num_lanes.reshape(-1)[0])} != lanes.shape[0] {lanes.shape[0]}")
    if lanes.size and (not np.isfinite(lanes).all() or float(lanes.min()) < 0.0 or float(lanes.max()) > 1.0):
        errors.append("lanes are not finite normalized coordinates in [0, 1]")
    if point_mode not in {"free", "fixed_y"}:
        errors.append(f"unsupported point_mode={point_mode!r}")
    if point_mode == "fixed_y":
        if fixed_y is None:
            errors.append("fixed_y point_mode requires fixed_y anchors")
        elif fixed_y.shape != (lanes.shape[1],):
            errors.append(f"fixed_y shape {fixed_y.shape} != ({lanes.shape[1]},)")
        elif not np.all(np.diff(fixed_y) < 0.0):
            errors.append("fixed_y anchors are not strictly descending")

    if lanes.ndim == 3 and lane_valid.shape == lanes.shape[:2]:
        for lane_idx, lane in enumerate(lanes):
            valid = lane_valid[lane_idx] > 0.5
            ys = lane[valid, 1]
            if len(ys) >= 2 and not np.all(np.diff(ys) <= eps):
                bad_order.append(
                    {
                        "label": str(label_path),
                        "lane_index": int(lane_idx),
                        "ys": [round(float(y), 6) for y in ys[:64]],
                    }
                )
            if point_mode == "fixed_y" and fixed_y is not None and fixed_y.shape == (lanes.shape[1],):
                y_err = np.abs(lane[:, 1] - fixed_y) * valid
                if y_err.size and float(y_err.max()) > max(float(eps), 1e-6):
                    errors.append(
                        f"lane {lane_idx} fixed_y valid rows differ from anchors, max_err={float(y_err.max()):.8f}"
                    )

    return (
        {
            "raw_file": raw_file,
            "group_id": tusimple_group_id(raw_file) if raw_file else "",
            "day_id": _day_id(raw_file) if raw_file else "unknown",
            "num_lanes": int(lanes.shape[0]) if lanes.ndim >= 1 else 0,
            "point_mode": point_mode,
        },
        bad_order,
        errors,
    )


def inspect_split(
    dataset_root: Path,
    split: str,
    expected_imgsz: tuple[int, int],
    max_order_files: int,
    eps: float,
) -> dict[str, Any]:
    label_dir = dataset_root / "labels_gcs" / split
    labels = sorted(label_dir.glob("*.npz"))
    order_labels = set(_select_for_order_check(labels, max_order_files))

    lane_hist: Counter[int] = Counter()
    point_mode_hist: Counter[str] = Counter()
    group_hist: Counter[str] = Counter()
    day_hist: Counter[str] = Counter()
    bad_order: list[dict[str, Any]] = []
    file_errors: list[dict[str, Any]] = []
    raw_files_missing = 0

    for label in labels:
        meta, label_bad_order, errors = check_label(
            label,
            expected_imgsz=expected_imgsz,
            eps=eps,
        )
        if errors:
            file_errors.append({"label": str(label), "errors": errors})
        if not meta:
            continue
        lane_hist[int(meta["num_lanes"])] += 1
        point_mode_hist[str(meta["point_mode"])] += 1
        if meta["group_id"]:
            group_hist[str(meta["group_id"])] += 1
        else:
            raw_files_missing += 1
        day_hist[str(meta["day_id"])] += 1
        if label in order_labels:
            bad_order.extend(label_bad_order)

    top_groups = group_hist.most_common(10)
    return {
        "label_count": len(labels),
        "order_checked_files": len(order_labels),
        "lane_count_hist": dict(sorted(lane_hist.items())),
        "point_mode_hist": dict(sorted(point_mode_hist.items())),
        "day_hist": dict(sorted(day_hist.items())),
        "group_count": len(group_hist),
        "top_group_sizes": top_groups,
        "raw_files_missing": raw_files_missing,
        "bad_order_lanes": len(bad_order),
        "bad_order_examples": bad_order[:5],
        "file_error_count": len(file_errors),
        "file_error_examples": file_errors[:5],
        "_groups": set(group_hist.keys()),
    }


def main() -> None:
    args = parse_args()
    dataset_root = ROOT / args.dataset_root if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    expected_imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    save_dir = ROOT / args.save_dir if not Path(args.save_dir).is_absolute() else Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, dict[str, Any]] = {}
    group_sets: dict[str, set[str]] = {}
    for split in args.splits:
        summary = inspect_split(
            dataset_root=dataset_root,
            split=split,
            expected_imgsz=expected_imgsz,
            max_order_files=args.max_order_files,
            eps=args.eps,
        )
        group_sets[split] = set(summary.pop("_groups"))
        split_summaries[split] = summary

    leakage: dict[str, Any] = {}
    splits = list(args.splits)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            overlap = sorted(group_sets[a].intersection(group_sets[b]))
            leakage[f"{a}_vs_{b}"] = {
                "overlap_count": len(overlap),
                "examples": overlap[:10],
            }

    out = {
        "dataset_root": str(dataset_root),
        "expected_imgsz_hw": list(expected_imgsz),
        "expected_shape": shape_str(expected_imgsz),
        "splits": split_summaries,
        "group_leakage": leakage,
    }
    (save_dir / "summary.json").write_text(json.dumps(_jsonable(out), indent=2), encoding="utf-8")

    print(f"dataset_root: {dataset_root}")
    print(f"expected image shape: {shape_str(expected_imgsz)} (W x H), stored as H,W={expected_imgsz}")
    for split, summary in split_summaries.items():
        print(
            f"{split}: labels={summary['label_count']} order_checked={summary['order_checked_files']} "
            f"bad_lanes={summary['bad_order_lanes']} file_errors={summary['file_error_count']} "
            f"lane_count_hist={summary['lane_count_hist']} point_mode_hist={summary['point_mode_hist']} "
            f"day_hist={summary['day_hist']}"
        )
    for name, item in leakage.items():
        print(f"{name}: group_overlap={item['overlap_count']} examples={item['examples'][:3]}")
    print(f"saved: {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
