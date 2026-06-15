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
    validate_tusimple_selection_source,
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
    parser.add_argument(
        "--diagnose-one-anchor-impact",
        action="store_true",
        help=(
            "Official-val-only GT-assisted diagnostic: append exact raw one-anchor GT lanes to a copy of the "
            "current label-oracle predictions and report the oracle delta. This is not model evidence."
        ),
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


def valid_hsample_count(lane: list[float] | tuple[float, ...]) -> int:
    """Return the number of valid TuSimple x samples in one lane."""
    return sum(1 for x in lane if float(x) >= 0.0)


def one_anchor_gt_lanes(record: dict) -> list[list[int]]:
    """Return exact raw GT lanes with one valid official h-sample."""
    out: list[list[int]] = []
    for lane in normalize_tusimple_gt_record(record).get("lanes", []):
        if valid_hsample_count(lane) == 1:
            out.append([int(x) if float(x) >= 0.0 else -2 for x in lane])
    return out


def add_one_anchor_gt_lanes_to_predictions(
    predictions: list[dict],
    gt_records: list[dict],
    *,
    runtime_ms: float = 1.0,
) -> tuple[list[dict], dict[str, Any]]:
    """Append exact one-anchor raw GT lanes to a prediction copy for oracle-only diagnostics."""
    pred_by_raw: dict[str, dict] = {}
    for pred in predictions:
        raw_file = str(pred["raw_file"]).replace("\\", "/").lstrip("/")
        if raw_file in pred_by_raw:
            raise ValueError(f"Duplicate prediction raw_file: {raw_file}")
        pred_by_raw[raw_file] = pred

    augmented: list[dict] = []
    added_hist: Counter[int] = Counter()
    examples: list[dict] = []
    total_added = 0
    for record in gt_records:
        raw_file = str(record["raw_file"]).replace("\\", "/").lstrip("/")
        pred = pred_by_raw.get(raw_file, {"raw_file": raw_file, "lanes": [], "run_time": round(float(runtime_ms), 3)})
        one_anchor_lanes = one_anchor_gt_lanes(record)
        lanes = [list(lane) for lane in pred.get("lanes", [])] + one_anchor_lanes
        augmented.append({"raw_file": raw_file, "lanes": lanes, "run_time": pred.get("run_time", round(float(runtime_ms), 3))})
        added_hist[len(one_anchor_lanes)] += 1
        total_added += len(one_anchor_lanes)
        if one_anchor_lanes and len(examples) < 20:
            h_samples = list(record.get("h_samples", []))
            examples.append(
                {
                    "raw_file": raw_file,
                    "added": len(one_anchor_lanes),
                    "valid_h_samples": [
                        [int(y) for x, y in zip(lane, h_samples) if float(x) >= 0.0]
                        for lane in one_anchor_lanes
                    ],
                    "valid_x": [
                        [int(x) for x in lane if float(x) >= 0.0]
                        for lane in one_anchor_lanes
                    ],
                }
            )

    stats = {
        "total_added_lanes": int(total_added),
        "images_with_added_lanes": int(sum(count for added, count in added_hist.items() if int(added) > 0)),
        "added_lanes_per_image_hist": added_hist,
        "examples": examples,
    }
    return augmented, stats


def exact_gt_predictions(gt_records: list[dict], *, runtime_ms: float = 1.0) -> list[dict]:
    """Return exact normalized GT lanes as TuSimple prediction records."""
    predictions: list[dict] = []
    for record in gt_records:
        gt = normalize_tusimple_gt_record(record)
        raw_file = str(gt["raw_file"]).replace("\\", "/").lstrip("/")
        predictions.append(
            {
                "raw_file": raw_file,
                "lanes": [[int(x) if float(x) >= 0.0 else -2 for x in lane] for lane in gt.get("lanes", [])],
                "run_time": round(float(runtime_ms), 3),
            }
        )
    return predictions


def metric_delta(updated: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    """Return rounded official-metric deltas in updated-minus-baseline order."""
    return {
        "Accuracy": round(float(updated["Accuracy"]) - float(baseline["Accuracy"]), 6),
        "FP": round(float(updated["FP"]) - float(baseline["FP"]), 6),
        "FN": round(float(updated["FN"]) - float(baseline["FN"]), 6),
    }


def one_anchor_per_image_delta(current_rows: list[dict], augmented_rows: list[dict]) -> list[dict]:
    """Return per-image deltas after appending exact one-anchor GT lanes."""
    by_raw = {str(row["raw_file"]): row for row in augmented_rows}
    rows: list[dict] = []
    for row in current_rows:
        raw_file = str(row["raw_file"])
        other = by_raw[raw_file]
        rows.append(
            {
                "raw_file": raw_file,
                "current_Accuracy": row["Accuracy"],
                "augmented_Accuracy": other["Accuracy"],
                "delta_Accuracy": round(float(other["Accuracy"]) - float(row["Accuracy"]), 6),
                "current_FP": row["FP"],
                "augmented_FP": other["FP"],
                "delta_FP": round(float(other["FP"]) - float(row["FP"]), 6),
                "current_FN": row["FN"],
                "augmented_FN": other["FN"],
                "delta_FN": round(float(other["FN"]) - float(row["FN"]), 6),
                "current_pred_lanes": row["pred_lanes"],
                "augmented_pred_lanes": other["pred_lanes"],
                "gt_lanes": row["gt_lanes"],
            }
        )
    rows.sort(key=lambda x: (float(x["delta_Accuracy"]), -float(x["delta_FN"])), reverse=True)
    return rows


def diagnose_one_anchor_impact(
    *,
    predictions: list[dict],
    gt_records: list[dict],
    current_metric: dict[str, Any],
    current_rows: list[dict],
    runtime_ms: float,
    save_dir: Path,
) -> dict[str, Any]:
    """Quantify the official-val oracle upper bound from exact raw one-anchor GT lanes."""
    augmented, added_stats = add_one_anchor_gt_lanes_to_predictions(
        predictions,
        gt_records,
        runtime_ms=float(runtime_ms),
    )
    augmented_result, augmented_rows = TuSimpleOfficialLaneEval.bench_records(
        augmented,
        gt_records,
        strict_length=True,
        return_records=True,
    )
    raw_gt_result, _ = TuSimpleOfficialLaneEval.bench_records(
        exact_gt_predictions(gt_records, runtime_ms=float(runtime_ms)),
        gt_records,
        strict_length=True,
        return_records=False,
    )
    delta_rows = one_anchor_per_image_delta(current_rows, augmented_rows)
    with (save_dir / "one_anchor_per_image_delta.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "raw_file",
                "current_Accuracy",
                "augmented_Accuracy",
                "delta_Accuracy",
                "current_FP",
                "augmented_FP",
                "delta_FP",
                "current_FN",
                "augmented_FN",
                "delta_FN",
                "current_pred_lanes",
                "augmented_pred_lanes",
                "gt_lanes",
            ),
        )
        writer.writeheader()
        writer.writerows(delta_rows)
    augmented_prediction_json = save_dir / "one_anchor_gt_assisted_not_for_selection_predictions.json"
    protocol_json = save_dir / "one_anchor_gt_assisted_not_for_selection_protocol.json"
    write_tusimple_predictions(augmented_prediction_json, augmented)
    protocol = {
        "gt_assisted": True,
        "official_val_only": True,
        "not_for_selection": True,
        "prediction_json": augmented_prediction_json,
        "summary_json": save_dir / "one_anchor_impact_summary.json",
        "warning": (
            "This prediction file contains exact raw GT one-anchor lanes appended for an official-val oracle "
            "diagnostic. It must not be evaluated or reported as model output."
        ),
    }
    protocol_json.write_text(json.dumps(_jsonable(protocol), indent=2), encoding="utf-8")

    augmented_metric = augmented_result.as_dict()
    diagnostic = {
        "gt_assisted": True,
        "official_val_only": True,
        "not_for_selection": True,
        "artifacts": {
            "gt_assisted_prediction_json": augmented_prediction_json,
            "protocol_json": protocol_json,
            "per_image_delta_csv": save_dir / "one_anchor_per_image_delta.csv",
        },
        "current_label_oracle_metric": current_metric,
        "plus_exact_one_anchor_gt_metric": augmented_metric,
        "raw_gt_exact_metric": raw_gt_result.as_dict(),
        "delta_plus_minus_current": metric_delta(augmented_metric, current_metric),
        "added_one_anchor_stats": added_stats,
        "top_positive_deltas": [row for row in delta_rows if float(row["delta_Accuracy"]) > 0.0][:20],
        "note": (
            "This is a GT-assisted representation/export oracle diagnostic on official-val only. "
            "It is not model evidence, must not be used on test, and does not change official metric logic."
        ),
    }
    (save_dir / "one_anchor_impact_summary.json").write_text(
        json.dumps(_jsonable(diagnostic), indent=2),
        encoding="utf-8",
    )
    return diagnostic


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


def validate_one_anchor_diagnostic_args(args: argparse.Namespace, gt_json: str | Path | None = None) -> None:
    """Fail unless the one-anchor oracle diagnostic is explicitly limited to official-val."""
    if not bool(getattr(args, "diagnose_one_anchor_impact", False)):
        return
    label_split = str(getattr(args, "label_split", "")).lower()
    if label_split != "val":
        raise SystemExit("ERROR: --diagnose-one-anchor-impact is official-val-only and requires --label-split val.")
    gt_path = gt_json if gt_json is not None else getattr(args, "gt_json", None)
    validate_tusimple_selection_source(
        "val",
        gt_json=gt_path,
        context="K56 one-anchor label-oracle impact diagnostic",
    )


def validate_one_anchor_diagnostic_records(
    args: argparse.Namespace,
    *,
    gt_json: str | Path,
    gt_records: list[dict],
) -> None:
    """Reject renamed test GT records for the one-anchor diagnostic."""
    if not bool(getattr(args, "diagnose_one_anchor_impact", False)):
        return
    validate_tusimple_selection_source(
        "val",
        gt_json=gt_json,
        gt_records=gt_records,
        archive_root=getattr(args, "archive_root", None),
        context="K56 one-anchor label-oracle impact diagnostic",
    )


def main() -> None:
    args = parse_args()
    if args.label_split == "test" and not args.allow_test:
        raise SystemExit(
            "ERROR: test label-oracle evaluation is final-only/protected. Pass --allow-test only for an explicit final audit."
        )
    validate_one_anchor_diagnostic_args(args)

    dataset_root = ROOT / args.dataset_root if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    label_dir = dataset_root / "labels_gcs" / args.label_split
    gt_json = resolve_gt_json(args)
    validate_one_anchor_diagnostic_args(args, gt_json=gt_json)
    save_dir = ROOT / args.save_dir if not Path(args.save_dir).is_absolute() else Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    labels_by_raw = label_index(label_dir)
    gt_records = [normalize_tusimple_gt_record(r) for r in read_tusimple_json_lines(gt_json)]
    validate_one_anchor_diagnostic_records(args, gt_json=gt_json, gt_records=gt_records)
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

    current_metric = result.as_dict()
    summary = {
        "dataset_root": dataset_root,
        "label_split": args.label_split,
        "label_dir": label_dir,
        "gt_json": gt_json,
        "prediction_json": pred_json,
        "metric": current_metric,
        "fixed_y": fixed_y_meta,
        "records": len(gt_records),
        "labels_indexed": len(labels_by_raw),
        "label_lane_hist": label_lane_hist,
        "gt_lane_hist": gt_lane_hist,
        "image_shape_for_official": list(TUSIMPLE_ORIGINAL_SHAPE),
        "note": "This is a label representation oracle; no model weights, decode GT, or official metric changes are used.",
    }
    if args.diagnose_one_anchor_impact:
        summary["one_anchor_impact"] = diagnose_one_anchor_impact(
            predictions=predictions,
            gt_records=gt_records,
            current_metric=current_metric,
            current_rows=per_image,
            runtime_ms=float(args.runtime_ms),
            save_dir=save_dir,
        )
    (save_dir / "label_oracle_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
