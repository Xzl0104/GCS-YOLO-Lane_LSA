from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


CATEGORIES = OrderedDict(
    [
        ("Normal", "Normal"),
        ("Crowded", "Crowded"),
        ("Dazzle", "Dazzle"),
        ("Shadow", "Shadow"),
        ("No line", "No line"),
        ("Arrow", "Arrow"),
        ("Curve", "Curve"),
        ("Crossroad", "Crossroad"),
        ("Night", "Night"),
    ]
)
TABLE_FIELDS = ["Method", *CATEGORIES.keys(), "FPS", "Total"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CULane-style per-category table from eval_gcs.py records."
    )
    parser.add_argument("--eval-summary", required=True, help="eval_summary.json created with --save-json.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Converted CULane manifests/manifest.json. Required to map output images to test categories.",
    )
    parser.add_argument("--method", default="GCS-YOLO-Lane")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-tex", default=None)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError(
            f"{path} does not contain per-image records. Rerun tools/eval_gcs.py with --save-json."
        )
    return records


def load_metric_name(path: Path) -> str:
    """Return the primary metric recorded by eval_gcs.py."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an eval_gcs.py JSON object.")
    summary = payload.get("summary")
    config = payload.get("config")
    for section in (summary, config):
        if isinstance(section, dict) and section.get("metric_name"):
            return str(section["metric_name"])
    records = payload.get("records")
    if isinstance(records, list) and records:
        metrics = records[0].get("metrics")
        if isinstance(metrics, dict) and metrics.get("metric_name"):
            return str(metrics["metric_name"])
    return ""


def load_category_by_image(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{manifest_path} must be the JSON array written by convert_culane_to_gcs.py.")
    mapping: dict[str, str] = {}
    for item in payload:
        if item.get("split") != "test":
            continue
        category = str(item.get("category", ""))
        image_name = Path(str(item.get("output_image", ""))).name
        if category and image_name:
            mapping[image_name] = category
    return mapping


def aggregate(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate CULane TP/FP/FN and derive F1/FPS."""
    tp = sum(int(record["metrics"]["tp"]) for record in records)
    fp = sum(int(record["metrics"]["fp"]) for record in records)
    fn = sum(int(record["metrics"]["fn"]) for record in records)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    elapsed = sum(
        float(record.get("inference_ms", 0.0)) + float(record.get("postprocess_ms", 0.0))
        for record in records
    ) / 1000.0
    fps = len(records) / max(elapsed, 1e-9)
    return {
        "images": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": 100.0 * f1,
        "fps": fps,
    }


def format_metric(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def build_row(
    records: list[dict[str, Any]],
    category_by_image: dict[str, str],
    method: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORIES}
    unmapped: list[str] = []
    for record in records:
        category = category_by_image.get(Path(str(record.get("image", ""))).name)
        if category not in grouped:
            unmapped.append(str(record.get("image", "")))
        else:
            grouped[category].append(record)
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} evaluation records could not be mapped to a CULane test category; "
            f"examples={unmapped[:3]}"
        )

    stats = {name: aggregate(items) for name, items in grouped.items()}
    total = aggregate(records)
    row = {"Method": method}
    for name in CATEGORIES:
        if name == "Crossroad":
            row[name] = format_metric(stats[name]["fp"], digits=0)
        else:
            row[name] = format_metric(stats[name]["f1"])
    row["FPS"] = format_metric(total["fps"])
    row["Total"] = format_metric(total["f1"])
    return row, {
        "metric_name": "culane_iou",
        "iou_rule": "IoU > 0.5",
        "crossroad_field": "fp",
        "categories": stats,
        "total": total,
    }


def markdown_table(row: dict[str, str]) -> str:
    header = "| " + " | ".join(TABLE_FIELDS) + " |"
    divider = "| " + " | ".join("---" for _ in TABLE_FIELDS) + " |"
    values = "| " + " | ".join(row[field] for field in TABLE_FIELDS) + " |"
    return "\n".join([header, divider, values, ""])


def latex_table(row: dict[str, str]) -> str:
    fields = ["Method", *CATEGORIES.keys(), "FPS", "Total"]
    lines = [
        r"\begin{tabular}{lrrrrrrrrrrr}",
        r"\toprule",
        " & ".join(fields) + r" \\",
        r"\midrule",
        " & ".join(row[field] for field in fields) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    eval_summary = Path(args.eval_summary).expanduser().resolve()
    manifest = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else ROOT / "datasets" / "culane" / "manifests" / "manifest.json"
    )
    records = load_records(eval_summary)
    metric_name = load_metric_name(eval_summary)
    if metric_name != "culane_iou":
        raise ValueError(
            f"{eval_summary} was evaluated with metric={metric_name or 'unknown'!r}. "
            "Run tools/eval_gcs.py with --metric culane_iou before building a CULane table."
        )
    category_by_image = load_category_by_image(manifest)
    row, details = build_row(records, category_by_image, args.method)

    output_base = eval_summary.parent
    output_csv = Path(args.output_csv) if args.output_csv else output_base / "culane_table.csv"
    output_md = Path(args.output_md) if args.output_md else output_base / "culane_table.md"
    output_tex = Path(args.output_tex) if args.output_tex else output_base / "culane_table.tex"
    for path in (output_csv, output_md, output_tex):
        path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    output_md.write_text(markdown_table(row), encoding="utf-8")
    output_tex.write_text(latex_table(row), encoding="utf-8")
    details_path = output_base / "culane_table_details.json"
    details_path.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(markdown_table(row), end="")
    print("metric: CULane region-mask IoU with strict IoU > 0.5; Crossroad column = FP count")
    print(f"csv: {output_csv.resolve()}")
    print(f"markdown: {output_md.resolve()}")
    print(f"latex: {output_tex.resolve()}")
    print(f"details: {details_path.resolve()}")


if __name__ == "__main__":
    main()
