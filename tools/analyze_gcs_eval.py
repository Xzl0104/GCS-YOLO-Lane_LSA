from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GCS-YOLO-Lane eval_summary.json by lane count and source.")
    parser.add_argument("summary", help="Path to eval_summary.json produced by tools/eval_gcs.py.")
    parser.add_argument("--topk", type=int, default=20, help="Number of worst images to print.")
    return parser.parse_args()


def source_group(image: str) -> str:
    """Group TuSimple test images by source date prefix."""
    name = Path(image).name
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def aggregate(records: list[dict]) -> dict:
    """Aggregate precision, recall, F1, and APE for a record subset."""
    tp = sum(int(x["metrics"]["tp"]) for x in records)
    fp = sum(int(x["metrics"]["fp"]) for x in records)
    fn = sum(int(x["metrics"]["fn"]) for x in records)
    apes = [float(v) for x in records for v in x["metrics"].get("ape", [])]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "images": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ape_mean": None if not apes else float(np.mean(apes)),
        "ape_median": None if not apes else float(np.median(apes)),
    }


def print_group(title: str, groups: dict) -> None:
    """Print grouped aggregate metrics in a compact table."""
    print(f"\n{title}")
    for key in sorted(groups):
        stats = aggregate(groups[key])
        mean = "n/a" if stats["ape_mean"] is None else f"{stats['ape_mean']:.2f}"
        median = "n/a" if stats["ape_median"] is None else f"{stats['ape_median']:.2f}"
        print(
            f"{key}: images={stats['images']} tp={stats['tp']} fp={stats['fp']} fn={stats['fn']} "
            f"P={stats['precision']:.4f} R={stats['recall']:.4f} F1={stats['f1']:.4f} "
            f"APEmean={mean} APEmed={median}"
        )


def print_worst(records: list[dict], topk: int) -> None:
    """Print images with the largest number of bad matched lanes."""
    rows = []
    for record in records:
        apes = record["metrics"].get("ape", []) or []
        bad = [float(x) for x in apes if float(x) >= 20.0]
        rows.append(
            (
                len(bad),
                max([float(x) for x in apes], default=0.0),
                float(np.mean(bad)) if bad else 0.0,
                Path(record["image"]).name,
                int(record["gt_lanes"]),
                int(record["pred_lanes"]),
                int(record["metrics"]["tp"]),
                int(record["metrics"]["fp"]),
                int(record["metrics"]["fn"]),
            )
        )

    print(f"\nWorst {topk} Images")
    for bad_count, max_ape, bad_mean, name, gt, pred, tp, fp, fn in sorted(rows, reverse=True)[:topk]:
        print(
            f"{name} gt={gt} pred={pred} tp={tp} fp={fp} fn={fn} "
            f"bad_matches={bad_count} max_ape={max_ape:.1f} bad_mean={bad_mean:.1f}"
        )


def main() -> None:
    args = parse_args()
    path = Path(args.summary)
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", data)
    records = data.get("records")
    if not records:
        raise SystemExit(f"{path} does not contain per-image records. Re-run eval_gcs.py with --save-json.")

    print("Summary")
    for key in (
        "images",
        "ape_threshold_px",
        "ape_mean_px",
        "ape_median_px",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ):
        print(f"{key}: {summary.get(key)}")

    by_gt = defaultdict(list)
    by_pred = defaultdict(list)
    by_source = defaultdict(list)
    for record in records:
        by_gt[int(record["gt_lanes"])].append(record)
        by_pred[int(record["pred_lanes"])].append(record)
        by_source[source_group(record["image"])].append(record)

    print_group("By GT Lane Count", by_gt)
    print_group("By Pred Lane Count", by_pred)
    print_group("By Source", by_source)
    print("\nGT/Pred Count Pairs")
    for (gt, pred), count in Counter((int(r["gt_lanes"]), int(r["pred_lanes"])) for r in records).most_common(20):
        print(f"gt={gt}, pred={pred}: {count}")
    print_worst(records, topk=args.topk)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
