from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit GT5-short proposal recovery headroom from raw-Q12 CSV diagnostics.")
    parser.add_argument("--split-dir", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--hit-px", type=float, default=20.0)
    parser.add_argument("--near-px", type=float, default=40.0)
    parser.add_argument("--moderate-px", type=float, default=60.0)
    parser.add_argument("--min-valid-points", type=int, default=4)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else math.inf


def audit(name: str, directory: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    diagnostics = read_csv(directory / "raw_gt_lane_diagnostics.csv")
    query_rows = read_csv(directory / "raw_query_gt_ape_long.csv")
    traces = {row["raw_file"]: row for row in read_csv(directory / "per_image_filter_trace.csv")}
    by_lane: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_image: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in query_rows:
        by_lane[(row["raw_file"], row["gt_lane_id"])].append(row)
        by_image[row["raw_file"]].append(row)

    raw_miss = [row for row in diagnostics if row["raw_has_match_20px"] != "True"]
    categories = Counter()
    detailed: list[dict] = []
    for row in raw_miss:
        candidates = sorted(by_lane[(row["raw_file"], row["gt_lane_id"])], key=lambda item: number(item, "ape_px"))
        nearest = candidates[0]
        nearest_ape = number(nearest, "ape_px")
        valid_len = number(nearest, "valid_len@0.6")
        occupied = any(
            other["query"] == nearest["query"]
            and other["gt_lane_id"] != row["gt_lane_id"]
            and number(other, "ape_px") <= args.hit_px
            for other in by_image[row["raw_file"]]
        )
        if nearest_ape <= args.near_px and valid_len >= args.min_valid_points and not occupied:
            category = "R1_near_geometry_recoverable"
        elif nearest_ape <= args.moderate_px and valid_len >= args.min_valid_points and not occupied:
            category = "R2_moderate_geometry_recoverable"
        elif nearest_ape <= args.moderate_px and valid_len >= args.min_valid_points and occupied:
            category = "R3_query_competition_recoverable"
        elif nearest_ape > args.moderate_px and valid_len >= args.min_valid_points:
            category = "R4_far_geometry_hard"
        else:
            category = "R5_valid_span_or_geometry_hard"
        categories[category] += 1
        trace = traces.get(row["raw_file"], {})
        detailed.append({
            **row,
            "nearest_ape_px": None if math.isinf(nearest_ape) else round(nearest_ape, 4),
            "nearest_query": nearest["query"],
            "nearest_valid_len_at_0_6": None if math.isinf(valid_len) else int(valid_len),
            "nearest_query_occupied_by_other_gt": occupied,
            "category": category,
            "final_count": trace.get("final_count", ""),
            "trace_gt_count": trace.get("gt_count", row["gt_count"]),
        })

    image_groups: defaultdict[str, list[dict]] = defaultdict(list)
    for row in detailed:
        image_groups[row["raw_file"]].append(row)
    capacity = Counter()
    for raw_file, rows in image_groups.items():
        trace = traces.get(raw_file, {})
        final_count = int(trace.get("final_count", 0) or 0)
        gt_count = int(trace.get("gt_count", rows[0]["gt_count"]))
        key = "capacity_room" if final_count < gt_count else "replacement_required"
        capacity[key + "_images"] += 1
        capacity[key + "_lanes"] += len(rows)
        if gt_count == 5:
            capacity["gt5_images"] += 1
            capacity["gt5_" + key + "_images"] += 1
            capacity["gt5_" + key + "_lanes"] += len(rows)

    summary = {
        "split": name,
        "config": {"hit_px": args.hit_px, "near_px": args.near_px, "moderate_px": args.moderate_px, "min_valid_points": args.min_valid_points},
        "gt_lanes": len(diagnostics),
        "raw_hit20": sum(row["raw_has_match_20px"] == "True" for row in diagnostics),
        "raw_miss20": len(raw_miss),
        "raw_coverage20": round(sum(row["raw_has_match_20px"] == "True" for row in diagnostics) / max(len(diagnostics), 1), 6),
        "raw_miss_short": sum(row["short_visible_lane"] == "True" for row in raw_miss),
        "raw_miss_gt4": sum(row["gt_count"] == "4" for row in raw_miss),
        "raw_miss_gt5": sum(row["gt_count"] == "5" for row in raw_miss),
        "categories": dict(categories),
        "capacity": dict(capacity),
        "ideal_proposal_coverage20": 1.0 if diagnostics else 0.0,
    }
    return summary, detailed


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = []
    for item in args.split_dir:
        if "=" not in item:
            raise ValueError(f"--split-dir must use NAME=DIR: {item}")
        name, raw_directory = item.split("=", 1)
        summary, rows = audit(name, Path(raw_directory), args)
        summaries.append(summary)
        all_rows.extend(rows)
    result = {"splits": summaries}
    (save_dir / "proposal_gate_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if all_rows:
        with (save_dir / "proposal_gate_raw_miss.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
