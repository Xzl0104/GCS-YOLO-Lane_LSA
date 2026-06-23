from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TuSimple official-val prediction failure modes across "
            "multiple already-generated tusimple_predictions.json files."
        )
    )
    parser.add_argument("--gt-json", required=True, help="TuSimple GT json-lines file.")
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named TuSimple prediction json-lines file. Repeat for each run.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for summary.json and per_image_failures.csv.")
    parser.add_argument("--baseline-run", default="count03", help="Run name used for count03-vs-dupmargin comparisons.")
    parser.add_argument("--dupmargin-run", default="dupmargin005", help="Run name used for dupmargin comparisons.")
    parser.add_argument("--gt4short-run", default="gt4short15", help="Run name used for gt4short15-vs-dupmargin comparisons.")
    parser.add_argument("--match-overlap", type=int, default=3, help="Minimum common visible points for a matched lane.")
    parser.add_argument("--match-x-thr", type=float, default=20.0, help="Max mean abs x error in px for a matched lane.")
    parser.add_argument(
        "--duplicate-overlap",
        type=int,
        default=3,
        help="Minimum common visible points for duplicate-like extra classification.",
    )
    parser.add_argument(
        "--duplicate-x-thr",
        type=float,
        default=20.0,
        help="Max mean abs x error in px for duplicate-like extra classification.",
    )
    parser.add_argument(
        "--short-visible-max",
        type=int,
        default=20,
        help="Unmatched GT lanes with <= this many visible points are missed_short_gt.",
    )
    return parser.parse_args()


def read_json_lines(path: str | Path) -> list[dict]:
    path = Path(path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON line: {exc}") from exc
    return records


def parse_named_paths(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--pred must use NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--pred must use NAME=PATH, got {item!r}")
        if name in out:
            raise ValueError(f"duplicate prediction name: {name}")
        out[name] = Path(path)
    return out


def valid_lane(lane: list[float]) -> bool:
    return any(float(x) >= 0.0 for x in lane)


def valid_lanes(lanes: list[list[float]]) -> list[list[float]]:
    return [list(lane) for lane in lanes if valid_lane(lane)]


def visible_count(lane: list[float]) -> int:
    return sum(1 for x in lane if float(x) >= 0.0)


def lane_pair_stats(a: list[float], b: list[float]) -> tuple[int, float]:
    if len(a) != len(b):
        raise ValueError(f"lane lengths differ: {len(a)} vs {len(b)}")
    diffs = [abs(float(xa) - float(xb)) for xa, xb in zip(a, b) if float(xa) >= 0.0 and float(xb) >= 0.0]
    if not diffs:
        return 0, math.inf
    return len(diffs), float(sum(diffs) / len(diffs))


def is_close_lane(a: list[float], b: list[float], min_overlap: int, x_thr: float) -> bool:
    overlap, mean_abs_x = lane_pair_stats(a, b)
    return overlap >= int(min_overlap) and mean_abs_x <= float(x_thr)


def match_lanes(
    pred_lanes: list[list[float]],
    gt_lanes: list[list[float]],
    min_overlap: int,
    x_thr: float,
) -> tuple[list[dict], set[int], set[int]]:
    candidates: list[tuple[float, int, int, int]] = []
    for pred_idx, pred_lane in enumerate(pred_lanes):
        for gt_idx, gt_lane in enumerate(gt_lanes):
            overlap, mean_abs_x = lane_pair_stats(pred_lane, gt_lane)
            if overlap >= int(min_overlap) and mean_abs_x <= float(x_thr):
                candidates.append((mean_abs_x, -overlap, pred_idx, gt_idx))
    candidates.sort()

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[dict] = []
    for mean_abs_x, neg_overlap, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches.append(
            {
                "pred_idx": int(pred_idx),
                "gt_idx": int(gt_idx),
                "overlap": int(-neg_overlap),
                "mean_abs_x_error": round(float(mean_abs_x), 6),
            }
        )
    return matches, matched_pred, matched_gt


def analyze_image(
    gt_record: dict,
    pred_record: dict,
    match_overlap: int,
    match_x_thr: float,
    duplicate_overlap: int,
    duplicate_x_thr: float,
    short_visible_max: int,
) -> dict:
    raw_file = str(gt_record["raw_file"])
    if str(pred_record["raw_file"]) != raw_file:
        raise ValueError(f"raw_file mismatch: GT {raw_file!r}, pred {pred_record.get('raw_file')!r}")
    gt_lanes = valid_lanes(gt_record.get("lanes", []))
    pred_lanes = [list(lane) for lane in pred_record.get("lanes", [])]

    matches, matched_pred, matched_gt = match_lanes(
        pred_lanes,
        gt_lanes,
        min_overlap=match_overlap,
        x_thr=match_x_thr,
    )
    gt_count = len(gt_lanes)
    pred_count = len(pred_lanes)
    unmatched_pred = [idx for idx in range(pred_count) if idx not in matched_pred]
    unmatched_gt = [idx for idx in range(gt_count) if idx not in matched_gt]

    duplicate_like_extra = 0
    spurious_extra = 0
    extra_labels: list[str] = []
    for pred_idx in unmatched_pred:
        pred_lane = pred_lanes[pred_idx]
        close_to_gt = any(
            is_close_lane(pred_lane, gt_lane, min_overlap=duplicate_overlap, x_thr=duplicate_x_thr)
            for gt_lane in gt_lanes
        )
        close_to_matched_pred = any(
            is_close_lane(pred_lane, pred_lanes[other_idx], min_overlap=duplicate_overlap, x_thr=duplicate_x_thr)
            for other_idx in matched_pred
        )
        if close_to_gt or close_to_matched_pred:
            duplicate_like_extra += 1
            extra_labels.append("duplicate_like_extra")
        else:
            spurious_extra += 1
            extra_labels.append("spurious_extra")

    missed_short_gt = 0
    missed_gt = 0
    missed_labels: list[str] = []
    for gt_idx in unmatched_gt:
        if visible_count(gt_lanes[gt_idx]) <= int(short_visible_max):
            missed_short_gt += 1
            missed_labels.append("missed_short_gt")
        else:
            missed_gt += 1
            missed_labels.append("missed_gt")

    count_error = pred_count - gt_count
    return {
        "raw_file": raw_file,
        "gt_count": int(gt_count),
        "pred_count": int(pred_count),
        "count_pair": f"{gt_count}->{pred_count}",
        "count_correct": bool(count_error == 0),
        "overcount": bool(count_error > 0),
        "undercount": bool(count_error < 0),
        "matched_lanes": int(len(matches)),
        "unmatched_pred": int(len(unmatched_pred)),
        "unmatched_gt": int(len(unmatched_gt)),
        "duplicate_like_extra": int(duplicate_like_extra),
        "spurious_extra": int(spurious_extra),
        "missed_short_gt": int(missed_short_gt),
        "missed_gt": int(missed_gt),
        "extra_labels": extra_labels,
        "missed_labels": missed_labels,
        "matches": matches,
    }


def summarize_run(rows: list[dict]) -> dict:
    images = len(rows)
    count_confusion = Counter(row["count_pair"] for row in rows)
    gt_hist = Counter(int(row["gt_count"]) for row in rows)
    pred_hist = Counter(int(row["pred_count"]) for row in rows)
    correct = sum(1 for row in rows if row["count_correct"])
    failure_rows = [row for row in rows if not row["count_correct"]]

    by_gt_count: dict[str, dict] = {}
    for gt_count in sorted(gt_hist):
        bucket = [row for row in rows if int(row["gt_count"]) == int(gt_count)]
        failure_bucket = [row for row in bucket if not row["count_correct"]]
        n = len(bucket)
        by_gt_count[str(gt_count)] = {
            "images": int(n),
            "count_acc": round(sum(1 for row in bucket if row["count_correct"]) / max(n, 1), 6),
            "count_confusion": dict(sorted(Counter(row["count_pair"] for row in bucket).items())),
            "overcount_images": int(sum(1 for row in bucket if row["overcount"])),
            "undercount_images": int(sum(1 for row in bucket if row["undercount"])),
            "duplicate_like_extra": int(sum(int(row["duplicate_like_extra"]) for row in bucket)),
            "spurious_extra": int(sum(int(row["spurious_extra"]) for row in bucket)),
            "missed_short_gt": int(sum(int(row["missed_short_gt"]) for row in bucket)),
            "missed_gt": int(sum(int(row["missed_gt"]) for row in bucket)),
            "count_failure_only": {
                "images": int(len(failure_bucket)),
                "count_confusion": dict(sorted(Counter(row["count_pair"] for row in failure_bucket).items())),
                "duplicate_like_extra": int(sum(int(row["duplicate_like_extra"]) for row in failure_bucket)),
                "spurious_extra": int(sum(int(row["spurious_extra"]) for row in failure_bucket)),
                "missed_short_gt": int(sum(int(row["missed_short_gt"]) for row in failure_bucket)),
                "missed_gt": int(sum(int(row["missed_gt"]) for row in failure_bucket)),
            },
        }

    return {
        "images": int(images),
        "count_acc": round(correct / max(images, 1), 6),
        "count_correct_images": int(correct),
        "overcount_images": int(sum(1 for row in rows if row["overcount"])),
        "undercount_images": int(sum(1 for row in rows if row["undercount"])),
        "count_confusion": dict(sorted(count_confusion.items())),
        "gt_lanes_hist": {str(k): int(v) for k, v in sorted(gt_hist.items())},
        "pred_lanes_hist": {str(k): int(v) for k, v in sorted(pred_hist.items())},
        "matched_lanes": int(sum(int(row["matched_lanes"]) for row in rows)),
        "unmatched_pred_lanes": int(sum(int(row["unmatched_pred"]) for row in rows)),
        "unmatched_gt_lanes": int(sum(int(row["unmatched_gt"]) for row in rows)),
        "duplicate_like_extra": int(sum(int(row["duplicate_like_extra"]) for row in rows)),
        "duplicate_like_extra_images": int(sum(1 for row in rows if int(row["duplicate_like_extra"]) > 0)),
        "spurious_extra": int(sum(int(row["spurious_extra"]) for row in rows)),
        "spurious_extra_images": int(sum(1 for row in rows if int(row["spurious_extra"]) > 0)),
        "missed_short_gt": int(sum(int(row["missed_short_gt"]) for row in rows)),
        "missed_short_gt_images": int(sum(1 for row in rows if int(row["missed_short_gt"]) > 0)),
        "missed_gt": int(sum(int(row["missed_gt"]) for row in rows)),
        "missed_gt_images": int(sum(1 for row in rows if int(row["missed_gt"]) > 0)),
        "count_failure_only": {
            "images": int(len(failure_rows)),
            "count_confusion": dict(sorted(Counter(row["count_pair"] for row in failure_rows).items())),
            "duplicate_like_extra": int(sum(int(row["duplicate_like_extra"]) for row in failure_rows)),
            "spurious_extra": int(sum(int(row["spurious_extra"]) for row in failure_rows)),
            "missed_short_gt": int(sum(int(row["missed_short_gt"]) for row in failure_rows)),
            "missed_gt": int(sum(int(row["missed_gt"]) for row in failure_rows)),
        },
        "by_gt_count": by_gt_count,
    }


def summarize_image_set(raw_files: list[str], rows_by_run: dict[str, dict[str, dict]], run_name: str) -> dict:
    gt_hist: Counter[int] = Counter()
    count_pairs: Counter[str] = Counter()
    failure_modes: Counter[str] = Counter()
    for raw_file in raw_files:
        row = rows_by_run[run_name][raw_file]
        gt_hist[int(row["gt_count"])] += 1
        count_pairs[str(row["count_pair"])] += 1
        for key in ("duplicate_like_extra", "spurious_extra", "missed_short_gt", "missed_gt"):
            failure_modes[key] += int(row[key])
    return {
        "images": int(len(raw_files)),
        "by_gt_count": {str(k): int(v) for k, v in sorted(gt_hist.items())},
        "count_pairs": dict(sorted(count_pairs.items())),
        "failure_modes_for_run": {str(k): int(v) for k, v in sorted(failure_modes.items())},
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    pred_paths = parse_named_paths(args.pred)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_records = read_json_lines(args.gt_json)
    gt_by_raw = {str(record["raw_file"]): record for record in gt_records}
    if len(gt_by_raw) != len(gt_records):
        raise ValueError("GT json has duplicate raw_file values.")

    rows_by_run: dict[str, dict[str, dict]] = {}
    run_rows: dict[str, list[dict]] = {}
    for run_name, pred_path in pred_paths.items():
        pred_records = read_json_lines(pred_path)
        pred_by_raw = {str(record["raw_file"]): record for record in pred_records}
        if len(pred_by_raw) != len(pred_records):
            raise ValueError(f"{pred_path} has duplicate raw_file values.")
        missing = sorted(set(gt_by_raw) - set(pred_by_raw))
        extra = sorted(set(pred_by_raw) - set(gt_by_raw))
        if missing or extra:
            raise ValueError(
                f"{run_name}: prediction raw_file set differs from GT "
                f"(missing={len(missing)}, extra={len(extra)})."
            )
        rows: list[dict] = []
        by_raw: dict[str, dict] = {}
        for gt_record in gt_records:
            raw_file = str(gt_record["raw_file"])
            row = analyze_image(
                gt_record,
                pred_by_raw[raw_file],
                match_overlap=args.match_overlap,
                match_x_thr=args.match_x_thr,
                duplicate_overlap=args.duplicate_overlap,
                duplicate_x_thr=args.duplicate_x_thr,
                short_visible_max=args.short_visible_max,
            )
            rows.append(row)
            by_raw[raw_file] = row
        run_rows[run_name] = rows
        rows_by_run[run_name] = by_raw

    missing_required = [
        name
        for name in (args.baseline_run, args.dupmargin_run, args.gt4short_run)
        if name not in rows_by_run
    ]
    if missing_required:
        raise ValueError(f"Missing required comparison run(s): {missing_required}. Available: {sorted(rows_by_run)}")

    baseline = args.baseline_run
    dup = args.dupmargin_run
    gt4 = args.gt4short_run
    raw_files = [str(record["raw_file"]) for record in gt_records]
    dupmargin_fixed_images = [
        raw_file
        for raw_file in raw_files
        if not rows_by_run[baseline][raw_file]["count_correct"] and rows_by_run[dup][raw_file]["count_correct"]
    ]
    dupmargin_regressed_images = [
        raw_file
        for raw_file in raw_files
        if rows_by_run[baseline][raw_file]["count_correct"] and not rows_by_run[dup][raw_file]["count_correct"]
    ]
    gt4short_correct_dupmargin_wrong_images = [
        raw_file
        for raw_file in raw_files
        if rows_by_run[gt4][raw_file]["count_correct"] and not rows_by_run[dup][raw_file]["count_correct"]
    ]

    summary_by_run = {run_name: summarize_run(rows) for run_name, rows in run_rows.items()}
    summary = {
        "config": {
            "gt_json": str(Path(args.gt_json).resolve()),
            "predictions": {name: str(path.resolve()) for name, path in pred_paths.items()},
            "match_overlap": int(args.match_overlap),
            "match_x_thr": float(args.match_x_thr),
            "duplicate_overlap": int(args.duplicate_overlap),
            "duplicate_x_thr": float(args.duplicate_x_thr),
            "short_visible_max": int(args.short_visible_max),
            "baseline_run": baseline,
            "dupmargin_run": dup,
            "gt4short_run": gt4,
        },
        "runs": summary_by_run,
        "count_confusion": {run_name: item["count_confusion"] for run_name, item in summary_by_run.items()},
        "GT4_GT5_count_acc": {
            run_name: {
                "GT4": item["by_gt_count"].get("4", {}).get("count_acc"),
                "GT5": item["by_gt_count"].get("5", {}).get("count_acc"),
            }
            for run_name, item in summary_by_run.items()
        },
        "spurious_extra": {run_name: int(item["spurious_extra"]) for run_name, item in summary_by_run.items()},
        "duplicate_like_extra": {run_name: int(item["duplicate_like_extra"]) for run_name, item in summary_by_run.items()},
        "missed_short_gt": {run_name: int(item["missed_short_gt"]) for run_name, item in summary_by_run.items()},
        "count_failure_only": {run_name: item["count_failure_only"] for run_name, item in summary_by_run.items()},
        "buckets_by_gt_count": {
            str(gt_count): {
                run_name: summary_by_run[run_name]["by_gt_count"].get(str(gt_count), {})
                for run_name in summary_by_run
            }
            for gt_count in (3, 4, 5)
        },
        "focus_GT4_GT5": {
            str(gt_count): {
                run_name: summary_by_run[run_name]["by_gt_count"].get(str(gt_count), {})
                for run_name in summary_by_run
            }
            for gt_count in (4, 5)
        },
        "dupmargin_fixed_images": dupmargin_fixed_images,
        "dupmargin_regressed_images": dupmargin_regressed_images,
        "gt4short15_correct_dupmargin_wrong_images": gt4short_correct_dupmargin_wrong_images,
        "set_summaries": {
            "dupmargin_fixed_images": summarize_image_set(dupmargin_fixed_images, rows_by_run, dup),
            "dupmargin_regressed_images": summarize_image_set(dupmargin_regressed_images, rows_by_run, dup),
            "gt4short15_correct_dupmargin_wrong_images": summarize_image_set(
                gt4short_correct_dupmargin_wrong_images,
                rows_by_run,
                dup,
            ),
        },
    }

    csv_rows: list[dict] = []
    run_names = list(pred_paths.keys())
    for raw_file in raw_files:
        base_row = {
            "raw_file": raw_file,
            "gt_count": rows_by_run[run_names[0]][raw_file]["gt_count"],
            "dupmargin_fixed": int(raw_file in set(dupmargin_fixed_images)),
            "dupmargin_regressed": int(raw_file in set(dupmargin_regressed_images)),
            "gt4short15_correct_dupmargin_wrong": int(raw_file in set(gt4short_correct_dupmargin_wrong_images)),
        }
        for run_name in run_names:
            row = rows_by_run[run_name][raw_file]
            prefix = f"{run_name}_"
            base_row.update(
                {
                    prefix + "pred_count": row["pred_count"],
                    prefix + "count_pair": row["count_pair"],
                    prefix + "count_correct": int(row["count_correct"]),
                    prefix + "overcount": int(row["overcount"]),
                    prefix + "undercount": int(row["undercount"]),
                    prefix + "matched_lanes": row["matched_lanes"],
                    prefix + "unmatched_pred": row["unmatched_pred"],
                    prefix + "unmatched_gt": row["unmatched_gt"],
                    prefix + "duplicate_like_extra": row["duplicate_like_extra"],
                    prefix + "spurious_extra": row["spurious_extra"],
                    prefix + "missed_short_gt": row["missed_short_gt"],
                    prefix + "missed_gt": row["missed_gt"],
                }
            )
        csv_rows.append(base_row)

    fields = ["raw_file", "gt_count", "dupmargin_fixed", "dupmargin_regressed", "gt4short15_correct_dupmargin_wrong"]
    for run_name in run_names:
        prefix = f"{run_name}_"
        fields.extend(
            [
                prefix + "pred_count",
                prefix + "count_pair",
                prefix + "count_correct",
                prefix + "overcount",
                prefix + "undercount",
                prefix + "matched_lanes",
                prefix + "unmatched_pred",
                prefix + "unmatched_gt",
                prefix + "duplicate_like_extra",
                prefix + "spurious_extra",
                prefix + "missed_short_gt",
                prefix + "missed_gt",
            ]
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "per_image_failures.csv", csv_rows, fields)
    print(json.dumps({"images": len(raw_files), "runs": list(pred_paths), "out_dir": str(out_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
