from __future__ import annotations

import argparse
import csv
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

from gcs_tools.tusimple_official_eval import (
    TUSIMPLE_ORIGINAL_SHAPE,
    TuSimpleOfficialLaneEval,
    default_tusimple_gt_json,
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    normalize_tusimple_gt_record,
    read_tusimple_json_lines,
    write_tusimple_predictions,
)


DEFAULT_DATASET_ROOT = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544"
DEFAULT_VAL_GT_JSON = (
    ROOT
    / "runs"
    / "gcs_lane"
    / "tusimple_official_val_363_folder_aware_seed20260602_subset"
    / "labels"
    / "tusimple_official_val_363_folder_aware_seed20260602.json"
)
DEFAULT_SAVE_DIR = ROOT / "runs" / "gcs_lane" / "tusimple_fixed_y_k56_label_oracle_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate converted fixed-y GCS labels as a TuSimple official-format label oracle."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Converted fixed-y dataset root.")
    parser.add_argument("--label-split", default="val", choices=("train", "val", "test"), help="labels_gcs split to read.")
    parser.add_argument("--archive-root", default="archive", help="TuSimple archive root used when --gt-json is omitted.")
    parser.add_argument("--gt-json", default=None, help="Official TuSimple json-lines GT.")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR), help="Output directory.")
    parser.add_argument("--runtime-ms", type=float, default=1.0, help="Constant oracle run_time in ms.")
    parser.add_argument("--allow-test", action="store_true", help="Allow oracle evaluation on test labels.")
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


def label_index(label_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for label_path in sorted(label_dir.glob("*.npz")):
        with np.load(label_path, allow_pickle=False) as data:
            if "raw_file" not in data.files:
                raise KeyError(f"{label_path} is missing raw_file")
            raw_file = _array_scalar_str(data["raw_file"]).replace("\\", "/").lstrip("/")
        previous = index.get(raw_file)
        if previous is not None:
            raise ValueError(f"Duplicate raw_file in labels: {raw_file} ({previous}, {label_path})")
        index[raw_file] = label_path
    if not index:
        raise FileNotFoundError(f"No .npz labels found under {label_dir}")
    return index


def load_label_as_decoded_lanes(label_path: Path) -> tuple[list[dict], dict[str, Any]]:
    with np.load(label_path, allow_pickle=False) as data:
        missing = {"lanes", "lane_valid", "fixed_y", "num_points", "point_mode"}.difference(data.files)
        if missing:
            raise KeyError(f"{label_path} missing required fixed-y arrays: {sorted(missing)}")
        lanes = np.asarray(data["lanes"], dtype=np.float32)
        lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
        fixed_y = np.asarray(data["fixed_y"], dtype=np.float32).reshape(-1)
        point_mode = _array_scalar_str(data["point_mode"]).lower()
        num_points = int(np.asarray(data["num_points"]).reshape(-1)[0])

    if point_mode in {"fixed-y", "fixedy"}:
        point_mode = "fixed_y"
    if point_mode != "fixed_y":
        raise ValueError(f"{label_path}: point_mode must be fixed_y, got {point_mode!r}")
    if lanes.ndim != 3 or lanes.shape[1:] != (num_points, 2):
        raise ValueError(f"{label_path}: lanes shape must be N x {num_points} x 2, got {lanes.shape}")
    if lane_valid.shape != lanes.shape[:2]:
        raise ValueError(f"{label_path}: lane_valid shape {lane_valid.shape} does not match lanes {lanes.shape[:2]}")
    if fixed_y.shape != (num_points,):
        raise ValueError(f"{label_path}: fixed_y shape {fixed_y.shape} does not match ({num_points},)")
    if not np.all(np.diff(fixed_y) < 0.0):
        raise ValueError(f"{label_path}: fixed_y anchors must be strictly descending")
    if lanes.size:
        y_err = np.abs(lanes[..., 1] - fixed_y.reshape(1, num_points)) * (lane_valid > 0.5)
        max_y_err = float(y_err.max()) if y_err.size else 0.0
        if max_y_err > 5e-5:
            raise ValueError(f"{label_path}: lane y coordinates do not match fixed_y, max_err={max_y_err:.6g}")

    decoded: list[dict] = []
    for lane, valid in zip(lanes, lane_valid):
        if int((valid > 0.5).sum()) < 2:
            continue
        decoded.append({"points_norm": lane.astype(np.float32), "point_valid": (valid > 0.5).astype(np.float32)})
    meta = {
        "num_points": int(num_points),
        "fixed_y_first": float(fixed_y[0]) if fixed_y.size else None,
        "fixed_y_last": float(fixed_y[-1]) if fixed_y.size else None,
        "label_lanes": int(len(decoded)),
    }
    return decoded, meta


def resolve_gt_json(args: argparse.Namespace) -> Path:
    gt_arg = str(args.gt_json or "").strip()
    if gt_arg:
        path = Path(gt_arg)
        return path if path.is_absolute() else ROOT / path
    if args.label_split == "val" and DEFAULT_VAL_GT_JSON.exists():
        return DEFAULT_VAL_GT_JSON
    if args.label_split == "test":
        raise SystemExit("ERROR: --label-split test requires an explicit --gt-json with --allow-test.")
    if args.label_split == "train":
        raise SystemExit("ERROR: --label-split train requires an explicit --gt-json.")
    archive_root = find_tusimple_archive_root(args.archive_root)
    return default_tusimple_gt_json(archive_root, split=args.label_split)


def main() -> None:
    args = parse_args()
    if args.label_split == "test" and not args.allow_test:
        raise SystemExit(
            "ERROR: test label-oracle evaluation is final-only/protected. Pass --allow-test only for an explicit final audit."
        )

    dataset_root = ROOT / args.dataset_root if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    label_dir = dataset_root / "labels_gcs" / args.label_split
    gt_json = resolve_gt_json(args)
    save_dir = ROOT / args.save_dir if not Path(args.save_dir).is_absolute() else Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    labels_by_raw = label_index(label_dir)
    gt_records = [normalize_tusimple_gt_record(r) for r in read_tusimple_json_lines(gt_json)]
    predictions: list[dict] = []
    missing: list[str] = []
    label_lane_hist: Counter[int] = Counter()
    gt_lane_hist: Counter[int] = Counter()
    fixed_y_meta: dict[str, Any] | None = None

    for record in gt_records:
        raw_file = str(record["raw_file"]).replace("\\", "/").lstrip("/")
        label_path = labels_by_raw.get(raw_file)
        if label_path is None:
            missing.append(raw_file)
            continue
        decoded, meta = load_label_as_decoded_lanes(label_path)
        fixed_y_meta = fixed_y_meta or {k: meta[k] for k in ("num_points", "fixed_y_first", "fixed_y_last")}
        label_lane_hist[int(meta["label_lanes"])] += 1
        gt_lane_hist[len(record.get("lanes", []))] += 1
        tusimple_lanes = gcs_lanes_to_tusimple_lanes(
            decoded,
            h_samples=list(record["h_samples"]),
            image_shape=TUSIMPLE_ORIGINAL_SHAPE,
        )
        predictions.append(
            {
                "raw_file": raw_file,
                "lanes": tusimple_lanes,
                "run_time": round(float(args.runtime_ms), 3),
            }
        )

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} labels for GT records. First missing raw_file={missing[0]}")

    result, per_image = TuSimpleOfficialLaneEval.bench_records(
        predictions,
        gt_records,
        strict_length=True,
        return_records=True,
    )

    pred_json = save_dir / "label_oracle_predictions.json"
    write_tusimple_predictions(pred_json, predictions)
    with (save_dir / "label_oracle_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("raw_file", "Accuracy", "FP", "FN", "pred_lanes", "gt_lanes", "run_time"),
        )
        writer.writeheader()
        writer.writerows(per_image)

    summary = {
        "dataset_root": dataset_root,
        "label_split": args.label_split,
        "label_dir": label_dir,
        "gt_json": gt_json,
        "prediction_json": pred_json,
        "metric": result.as_dict(),
        "fixed_y": fixed_y_meta,
        "records": len(gt_records),
        "labels_indexed": len(labels_by_raw),
        "label_lane_hist": label_lane_hist,
        "gt_lane_hist": gt_lane_hist,
        "image_shape_for_official": list(TUSIMPLE_ORIGINAL_SHAPE),
        "note": "This is a label representation oracle; no model weights, decode GT, or official metric changes are used.",
    }
    (save_dir / "label_oracle_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
