from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    normalize_tusimple_gt_record,
    read_tusimple_json_lines,
    valid_tusimple_lanes,
)
from gcs_tools.tusimple_split_guard import reject_tusimple_test_search_gt_json  # noqa: E402


DEFAULT_BUCKET_EDGES = (10, 20, 30, 40)
DEFAULT_WEIGHTS = (
    ROOT
    / "runs"
    / "gcs_lane"
    / "gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03"
    / "weights"
    / "best.pt"
)
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose TuSimple lane-count confusion by date, GT lane count, "
            "and shortest visible GT lane bucket."
        )
    )
    parser.add_argument("--split", default="val", choices=("train", "val"), help="Diagnostic split for JSON mode. Test is intentionally disallowed.")
    parser.add_argument("--gt-json", default=None, help="TuSimple GT json-lines file for the train/val diagnostic split.")
    parser.add_argument("--pred-json", default=None, help="TuSimple-format prediction json-lines file.")
    parser.add_argument("--weights", default=None, help="GCS checkpoint for direct fixed-y dataset inference mode.")
    parser.add_argument("--dataset-root", default=None, help="Converted fixed-y GCS dataset root for direct inference mode.")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=None, help="Dataset splits for direct inference mode.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.05, help="Lane existence threshold for direct inference mode.")
    parser.add_argument("--point-valid-thr", type=float, default=0.5, help="Point-valid threshold for direct inference mode.")
    parser.add_argument("--nms-dist-px", type=float, default=50.0, help="Lane-NMS distance for direct inference mode.")
    parser.add_argument("--max-det", type=int, default=8, help="Maximum decoded lanes for direct inference mode.")
    parser.add_argument("--min-points", type=int, default=5, help="Minimum visible anchors for direct inference mode.")
    parser.add_argument("--device", default="0", help="Inference device for direct inference mode.")
    parser.add_argument("--half", action="store_true", help="Use FP16 for direct inference mode on CUDA.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional image limit for direct inference mode. 0 means all.")
    parser.add_argument(
        "--bucket-edges",
        nargs="+",
        type=int,
        default=list(DEFAULT_BUCKET_EDGES),
        help="Upper edges for min-visible-point buckets. Default: 10 20 30 40.",
    )
    parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Skip GT records without predictions. Default is to fail fast.",
    )
    parser.add_argument("--save-json", default=None, help="Optional path for full diagnostic summary JSON.")
    parser.add_argument("--save-csv", default=None, help="Optional path for grouped diagnostic CSV.")
    parser.add_argument("--topk", type=int, default=0, help="Print the largest error groups. 0 disables.")
    return parser.parse_args()


def date_id(raw_file: str) -> str:
    """Return the TuSimple date/domain id from a raw_file path."""
    rel = raw_file.lstrip("/").replace("\\", "/")
    parts = Path(rel).parts
    if len(parts) >= 2 and parts[0] == "clips":
        return parts[1]
    return "unknown"


def valid_point_count(lane: Iterable[float]) -> int:
    """Count visible x coordinates in one TuSimple lane."""
    return sum(1 for x in lane if float(x) >= 0.0)


def lane_count(record: dict) -> int:
    """Count valid TuSimple lanes in a GT or prediction record."""
    return len(valid_tusimple_lanes(record.get("lanes", [])))


def min_visible_points(record: dict) -> int:
    """Return the shortest visible-lane length in one normalized GT record."""
    lanes = valid_tusimple_lanes(record.get("lanes", []))
    if not lanes:
        return 0
    return min(valid_point_count(lane) for lane in lanes)


def bucket_label(value: int, edges: list[int]) -> str:
    """Map a point count to a stable bucket label."""
    if not edges:
        return "all"
    prev = 0
    for edge in edges:
        if int(value) <= int(edge):
            if prev == 0:
                return f"<= {int(edge)}"
            return f"{prev + 1}-{int(edge)}"
        prev = int(edge)
    return f"> {prev}"


def confusion_dict(pairs: Iterable[tuple[int, int]]) -> dict[str, int]:
    """Return sorted GT->pred confusion counts."""
    return {f"{gt}->{pred}": int(count) for (gt, pred), count in sorted(Counter(pairs).items())}


def compact_confusion(confusion: dict[str, int]) -> str:
    """Format a confusion dict for table output."""
    return ", ".join(f"{key}:{value}" for key, value in confusion.items())


def records_by_raw(records: Iterable[dict], path: Path) -> dict[str, dict]:
    """Index records by raw_file and reject duplicates."""
    out: dict[str, dict] = {}
    for record in records:
        raw_file = str(record.get("raw_file", ""))
        if not raw_file:
            raise ValueError(f"{path}: record is missing raw_file")
        if raw_file in out:
            raise ValueError(f"{path}: duplicate raw_file={raw_file!r}")
        out[raw_file] = record
    return out


def summarize_group(rows: list[dict]) -> dict:
    """Aggregate count correctness and confusion for one record subset."""
    pairs = [(int(row["gt_count"]), int(row["pred_count"])) for row in rows]
    images = len(rows)
    correct = sum(1 for gt, pred in pairs if gt == pred)
    return {
        "images": int(images),
        "correct": int(correct),
        "count_acc": round(correct / max(images, 1), 6),
        "gt_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(gt for gt, _ in pairs).items())},
        "pred_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(pred for _, pred in pairs).items())},
        "count_confusion": confusion_dict(pairs),
    }


def summarize_rows(rows: list[dict], bucket_edges: list[int], missing: list[str] | None = None) -> dict:
    """Build the common count-confusion summary from normalized row dictionaries."""
    if not rows:
        raise ValueError("No matched records to diagnose.")
    grouped = group_rows(rows)
    return {
        "overall": summarize_group(rows),
        "groups": grouped,
        "matched_images": len(rows),
        "missing_predictions": list(missing or []),
        "bucket_edges": [int(x) for x in bucket_edges],
    }


def group_rows(rows: list[dict]) -> list[dict]:
    """Group diagnostics by date, GT lane count, and min-visible-point bucket."""
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for row in rows:
        key = (str(row["date"]), int(row["gt_count"]), str(row["min_visible_bucket"]))
        grouped.setdefault(key, []).append(row)

    output = []
    def sort_key(item: tuple[tuple[str, int, str], list[dict]]) -> tuple[str, int, int, str]:
        (date, gt_count, bucket), subset = item
        return date, gt_count, min(int(row["min_visible_points"]) for row in subset), bucket

    for (date, gt_count, bucket), subset in sorted(grouped.items(), key=sort_key):
        summary = summarize_group(subset)
        output.append(
            {
                "date": date,
                "gt_count": int(gt_count),
                "min_visible_bucket": bucket,
                "min_visible_min": min(int(row["min_visible_points"]) for row in subset),
                "min_visible_max": max(int(row["min_visible_points"]) for row in subset),
                **summary,
            }
        )
    return output


def build_diagnostics(
    gt_records: list[dict],
    pred_records: list[dict],
    bucket_edges: list[int],
    allow_missing_predictions: bool = False,
) -> dict:
    """Build grouped lane-count diagnostics from TuSimple GT and prediction records."""
    pred_by_raw = records_by_raw(pred_records, Path("<pred-json>"))
    rows: list[dict] = []
    missing: list[str] = []

    for raw_gt in gt_records:
        gt = normalize_tusimple_gt_record(raw_gt)
        raw_file = str(gt.get("raw_file", ""))
        if not raw_file:
            raise ValueError("GT record is missing raw_file")
        pred = pred_by_raw.get(raw_file)
        if pred is None:
            missing.append(raw_file)
            continue
        min_visible = min_visible_points(gt)
        rows.append(
            {
                "raw_file": raw_file,
                "date": date_id(raw_file),
                "gt_count": lane_count(gt),
                "pred_count": lane_count(pred),
                "min_visible_points": int(min_visible),
                "min_visible_bucket": bucket_label(int(min_visible), bucket_edges),
            }
        )

    if missing and not allow_missing_predictions:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing predictions for {len(missing)} GT records, first missing: {preview}")
    if not rows:
        raise ValueError("No matched GT/prediction records to diagnose.")

    return summarize_rows(rows, bucket_edges=bucket_edges, missing=missing)


def print_summary(summary: dict, topk: int = 0) -> None:
    """Print a compact human-readable diagnostic table."""
    overall = summary["overall"]
    print(
        "Overall: "
        f"images={overall['images']} correct={overall['correct']} "
        f"count_acc={overall['count_acc']:.6f} "
        f"confusion={compact_confusion(overall['count_confusion'])}"
    )
    print()
    print("date       gt  min_visible  images  correct  count_acc  confusion")
    for row in summary["groups"]:
        print(
            f"{row['date']:<10} {row['gt_count']:>2}  {row['min_visible_bucket']:<11} "
            f"{row['images']:>6}  {row['correct']:>7}  {row['count_acc']:.6f}  "
            f"{compact_confusion(row['count_confusion'])}"
        )
    if topk > 0:
        ranked = sorted(
            summary["groups"],
            key=lambda row: (row["images"] - row["correct"], row["images"]),
            reverse=True,
        )
        print()
        print(f"Top {min(topk, len(ranked))} Error Groups")
        for row in ranked[:topk]:
            errors = row["images"] - row["correct"]
            print(
                f"{row['date']} gt={row['gt_count']} min_visible={row['min_visible_bucket']} "
                f"errors={errors} images={row['images']} count_acc={row['count_acc']:.6f} "
                f"confusion={compact_confusion(row['count_confusion'])}"
            )


def save_json(path: str | Path, summary: dict, config: dict) -> None:
    """Save full diagnostics as JSON."""
    out = {"config": config, **summary}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def save_csv(path: str | Path, groups: list[dict]) -> None:
    """Save grouped diagnostics as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    if any("split" in row for row in groups):
        fieldnames.append("split")
    fieldnames.extend([
        "date",
        "gt_count",
        "min_visible_bucket",
        "min_visible_min",
        "min_visible_max",
        "images",
        "correct",
        "count_acc",
        "gt_lanes_hist",
        "pred_lanes_hist",
        "count_confusion",
    ])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in groups:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True) if isinstance(row[key], dict) else row[key]
                    for key in fieldnames
                }
            )


def _npz_scalar_text(value) -> str:
    """Read a scalar string/bytes value from a numpy npz array."""
    import numpy as np

    arr = np.asarray(value)
    item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _label_row(label_path: Path, pred_count: int, bucket_edges: list[int]) -> dict:
    """Build one normalized diagnostic row from a fixed-y GCS label and predicted count."""
    import numpy as np

    with np.load(label_path, allow_pickle=False) as data:
        lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
        raw_file = _npz_scalar_text(data["raw_file"]) if "raw_file" in data.files else label_path.with_suffix(".jpg").name
        if "num_lanes" in data.files:
            gt_count = int(np.asarray(data["num_lanes"]).reshape(-1)[0])
        else:
            gt_count = int((lane_valid.sum(axis=1) >= 2).sum())
    visible_counts = [int(x) for x in lane_valid.sum(axis=1).tolist()[:gt_count]]
    min_visible = min(visible_counts) if visible_counts else 0
    return {
        "raw_file": raw_file,
        "date": date_id(raw_file),
        "gt_count": int(gt_count),
        "pred_count": int(pred_count),
        "min_visible_points": int(min_visible),
        "min_visible_bucket": bucket_label(int(min_visible), bucket_edges),
    }


def _label_path_for_image(image_path: Path, label_dir: Path) -> Path:
    """Map one fixed-y dataset image to its labels_gcs npz file."""
    return label_dir / f"{image_path.stem}.npz"


def _collect_split_images(image_dir: Path, max_images: int = 0) -> list[Path]:
    """Collect split images for direct inference mode."""
    from tools.infer_gcs import collect_images

    return collect_images(image_dir, max_images=max_images)


def run_direct_inference(args: argparse.Namespace, bucket_edges: list[int]) -> dict:
    """Run checkpoint inference on fixed-y train/val splits and summarize count confusion."""
    import cv2
    import torch

    from tools.infer_gcs import load_gcs_model, preprocess_image
    from ultralytics.utils.gcs_postprocess import decode_gcs_predictions
    from ultralytics.utils.gcs_shape import normalize_imgsz
    from ultralytics.utils.torch_utils import select_device

    weights = Path(args.weights or DEFAULT_WEIGHTS)
    dataset_root = Path(args.dataset_root or DEFAULT_DATASET_ROOT)
    splits = args.splits or [args.split]
    if any(split not in {"train", "val"} for split in splits):
        raise ValueError("Direct count-confusion diagnostics only allow train/val splits.")
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    imgsz = normalize_imgsz(args.imgsz, dataset="tusimple")
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)

    output = {
        "config": {
            "mode": "direct_fixed_y_inference",
            "weights": str(weights.resolve()),
            "dataset_root": str(dataset_root.resolve()),
            "splits": list(splits),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "device": str(args.device),
            "half": bool(args.half),
            "max_images": int(args.max_images),
            "bucket_edges": bucket_edges,
        },
        "splits": {},
    }

    for split in splits:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels_gcs" / split
        images = _collect_split_images(image_dir, max_images=int(args.max_images))
        rows: list[dict] = []
        records: list[dict] = []
        for idx, image_path in enumerate(images, start=1):
            label_path = _label_path_for_image(image_path, label_dir)
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")
            tensor = preprocess_image(img, imgsz=imgsz, device=device, half=bool(args.half))
            with torch.inference_mode():
                preds = model(tensor)
            pred_valid = preds.get("pred_valid_logits")
            pred_lanes = decode_gcs_predictions(
                preds["pred_points"][0],
                preds["pred_logits"][0],
                pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
                image_shape=img.shape[:2],
                score_thr=float(args.conf),
                point_valid_thr=float(args.point_valid_thr),
                min_points=int(args.min_points),
                max_det=int(args.max_det),
                nms_dist_px=float(args.nms_dist_px),
            )
            row = _label_row(label_path, pred_count=len(pred_lanes), bucket_edges=bucket_edges)
            rows.append(row)
            records.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    **row,
                }
            )
            if idx % 250 == 0 or idx == len(images):
                print(f"{split}: processed {idx}/{len(images)}", flush=True)
        output["splits"][split] = {**summarize_rows(rows, bucket_edges=bucket_edges), "records": records}
    return output


def main() -> None:
    args = parse_args()
    bucket_edges = sorted({int(x) for x in args.bucket_edges})
    direct_mode = bool(args.weights or args.dataset_root or args.splits)
    if direct_mode:
        output = run_direct_inference(args, bucket_edges=bucket_edges)
        for split, summary in output["splits"].items():
            print()
            print(f"Split: {split}")
            print_summary(summary, topk=int(args.topk))
        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"saved JSON: {Path(args.save_json).resolve()}")
        if args.save_csv:
            rows = []
            for split, summary in output["splits"].items():
                for row in summary["groups"]:
                    rows.append({"split": split, **row})
            save_csv(args.save_csv, rows)
            print(f"saved CSV: {Path(args.save_csv).resolve()}")
        return

    if not args.gt_json or not args.pred_json:
        raise ValueError("JSON mode requires --gt-json and --pred-json. Direct mode requires --weights/--dataset-root/--splits.")
    gt_path = Path(args.gt_json)
    pred_path = Path(args.pred_json)
    reject_tusimple_test_search_gt_json(gt_path, context="TuSimple count-confusion diagnostics")
    summary = build_diagnostics(
        gt_records=read_tusimple_json_lines(gt_path),
        pred_records=read_tusimple_json_lines(pred_path),
        bucket_edges=bucket_edges,
        allow_missing_predictions=bool(args.allow_missing_predictions),
    )
    print_summary(summary, topk=int(args.topk))
    config = {
        "mode": "json_records",
        "gt_json": str(gt_path.resolve()),
        "pred_json": str(pred_path.resolve()),
        "split": str(args.split),
        "bucket_edges": bucket_edges,
        "allow_missing_predictions": bool(args.allow_missing_predictions),
    }
    if args.save_json:
        save_json(args.save_json, summary, config=config)
        print(f"saved JSON: {Path(args.save_json).resolve()}")
    if args.save_csv:
        save_csv(args.save_csv, summary["groups"])
        print(f"saved CSV: {Path(args.save_csv).resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
