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

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    default_tusimple_gt_json,
    find_tusimple_archive_root,
    normalize_tusimple_gt_record,
    read_tusimple_json_lines,
)
from gcs_tools.label_utils import (  # noqa: E402
    TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM,
    TUSIMPLE_OFFICIAL_TOP_Y_NORM,
    fixed_y_anchors,
)


DEFAULT_SAVE = ROOT / "runs" / "gcs_lane" / "tusimple_hsample_endpoint_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TuSimple official h_sample coverage for the K56 fixed-y contract.")
    parser.add_argument("--archive-root", default="archive", help="TuSimple archive root used when --gt-json is omitted.")
    parser.add_argument("--gt-json", default=None, help="TuSimple json-lines GT file. Defaults to official val/test path.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Split used for default GT lookup.")
    parser.add_argument("--save", default=str(DEFAULT_SAVE), help="Summary JSON path.")
    parser.add_argument("--k56-start", type=float, default=TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM)
    parser.add_argument("--k56-end", type=float, default=TUSIMPLE_OFFICIAL_TOP_Y_NORM)
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


def official_valid_y_range(record: dict, lane: list[float]) -> tuple[int, int] | None:
    """Return first/last official h_sample y for valid lane x entries."""
    h_samples = list(record.get("h_samples", []))
    valid_y = [int(y) for x, y in zip(lane, h_samples) if float(x) >= 0.0]
    if not valid_y:
        return None
    return min(valid_y), max(valid_y)


def _anchor_pixels(num_points: int, start: float, end: float) -> list[int]:
    return [int(round(float(y) * 720.0)) for y in fixed_y_anchors(num_points=num_points, y_start=start, y_end=end)]


def _coverage(valid_y: set[int], anchors: set[int]) -> int:
    return len(valid_y.intersection(anchors))


def analyze_records(records: list[dict], *, k56_start: float, k56_end: float) -> dict:
    """Summarize official h_sample lane lengths and fixed-y anchor coverage."""
    k56_anchors = set(_anchor_pixels(56, k56_start, k56_end))

    lane_len_hist: Counter[int] = Counter()
    endpoint_hist: Counter[str] = Counter()
    k56_cover_hist: Counter[int] = Counter()
    k56_lost_endpoint_hist: Counter[str] = Counter()
    ultra_short_examples: list[dict] = []
    total_lanes = 0

    for record in records:
        gt = normalize_tusimple_gt_record(record)
        h_samples = list(gt.get("h_samples", []))
        for lane in gt.get("lanes", []):
            valid_y = [int(y) for x, y in zip(lane, h_samples) if float(x) >= 0.0]
            if not valid_y:
                continue
            total_lanes += 1
            valid_set = set(valid_y)
            lane_len = len(valid_y)
            lane_len_hist[lane_len] += 1
            top_y, bottom_y = min(valid_y), max(valid_y)
            endpoint_hist[f"{top_y}->{bottom_y}"] += 1

            k56_cover = _coverage(valid_set, k56_anchors)
            k56_cover_hist[k56_cover] += 1

            if top_y not in k56_anchors:
                k56_lost_endpoint_hist[f"top:{top_y}"] += 1
            if bottom_y not in k56_anchors:
                k56_lost_endpoint_hist[f"bottom:{bottom_y}"] += 1

            if lane_len <= 3 and len(ultra_short_examples) < 20:
                ultra_short_examples.append(
                    {
                        "raw_file": str(gt.get("raw_file", "")),
                        "valid_h_samples": valid_y,
                        "k56_anchor_hits": sorted(valid_set.intersection(k56_anchors), reverse=True),
                    }
                )

    one_to_three = sum(v for k, v in lane_len_hist.items() if int(k) <= 3)
    summary = {
        "records": len(records),
        "lanes": int(total_lanes),
        "lane_valid_h_sample_count_hist": lane_len_hist,
        "official_endpoint_hist": endpoint_hist,
        "ultra_short_lanes_1_to_3_h_samples": int(one_to_three),
        "ultra_short_lane_rate": round(float(one_to_three) / max(total_lanes, 1), 8),
        "k56": {
            "anchors_px": sorted(k56_anchors, reverse=True),
            "coverage_hist": k56_cover_hist,
            "zero_anchor_lanes": int(k56_cover_hist.get(0, 0)),
            "one_anchor_lanes": int(k56_cover_hist.get(1, 0)),
            "lost_endpoint_hist": k56_lost_endpoint_hist,
        },
        "ultra_short_examples": ultra_short_examples,
        "note": (
            "This analyzes raw TuSimple GT h_samples only. It does not fabricate lanes, "
            "change labels, or use test for selection."
        ),
    }
    return summary


def resolve_gt_json(args: argparse.Namespace) -> Path:
    if args.gt_json:
        path = Path(args.gt_json)
        return path if path.is_absolute() else ROOT / path
    archive_root = find_tusimple_archive_root(args.archive_root)
    return default_tusimple_gt_json(archive_root, split=args.split)


def main() -> None:
    args = parse_args()
    gt_json = resolve_gt_json(args)
    records = read_tusimple_json_lines(gt_json)
    summary = analyze_records(
        records,
        k56_start=float(args.k56_start),
        k56_end=float(args.k56_end),
    )
    summary["gt_json"] = str(gt_json.resolve())
    summary["split"] = str(args.split)
    save = Path(args.save)
    save = save if save.is_absolute() else ROOT / save
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
