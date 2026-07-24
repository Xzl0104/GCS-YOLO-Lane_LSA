from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Q24 raw GT-lane diagnostics and final extra-lane diagnostics into a "
            "per-query event-mined gate table. This is GT-based diagnostics only."
        )
    )
    parser.add_argument("--run-name", default="", help="Run name written to the summary.")
    parser.add_argument(
        "--raw",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Raw GT-lane diagnostic CSV, e.g. val=.../raw_gt_lane_diagnostics.csv. Can repeat.",
    )
    parser.add_argument("--val-raw", default=None, help="Convenience alias for --raw val=CSV.")
    parser.add_argument("--train0601-raw", default=None, help="Convenience alias for --raw train0601=CSV.")
    parser.add_argument("--train0531-raw", default=None, help="Convenience alias for --raw train0531=CSV.")
    parser.add_argument(
        "--gt5-extra",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="GT5->6 final extra-lane CSV. Path-only form defaults to val_gt5_to6.",
    )
    parser.add_argument(
        "--false-extra",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="GT3/GT4 false-extra CSV, e.g. val_gt4_to5=.../gt5_extra_lanes.csv. Can repeat.",
    )
    parser.add_argument("--gt5-queries", default="", help="Query ids considered GT5 bank, e.g. '12,13,15,21,23'.")
    parser.add_argument("--gt4-queries", default="", help="Query ids considered GT4 bank.")
    parser.add_argument("--watch-queries", default="0,1,11", help="Protected/default queries to include in the table.")
    parser.add_argument("--gt5-short-visible-max", type=int, default=10)
    parser.add_argument("--gt5-weak-visible-max", type=int, default=20)
    parser.add_argument("--gt34-visible-max", type=int, default=20)
    parser.add_argument("--hit-thr-px", type=float, default=20.0)
    parser.add_argument("--near-thr-px", type=float, default=40.0)
    parser.add_argument("--score-risk-thr", type=float, default=0.2)
    parser.add_argument("--min-val-gt5-hit20", type=int, default=4)
    parser.add_argument("--max-risk-events", type=int, default=2)
    parser.add_argument("--high-risk-events", type=int, default=10)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--allow-test-oracle",
        action="store_true",
        help="Explicitly allow test-derived CSV paths for a reporting-only audit. Never use for selection.",
    )
    return parser.parse_args()


def parse_query_spec(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        values = [int(v) for v in value]
    else:
        text = str(value).strip()
        if text.lower() in {"", "none", "false", "off"}:
            return ()
        values = []
        for raw_part in text.split(","):
            part = raw_part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start = int(start_s.strip())
                end = int(end_s.strip())
                step = 1 if end >= start else -1
                values.extend(range(start, end + step, step))
            else:
                values.append(int(part))
    if any(v < 0 for v in values):
        raise ValueError(f"Query ids must be non-negative, got {values}.")
    return tuple(sorted(set(values)))


def labeled_path(value: str, default_label: str) -> tuple[str, Path]:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty labeled path.")
    if "=" in text:
        label, raw_path = text.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
    else:
        label = default_label
        path = Path(text)
    if not label:
        raise ValueError(f"Missing label in {value!r}.")
    return label, path


def read_csv(path: Path, *, allow_test_oracle: bool) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    normalized = str(path).replace("\\", "/").lower()
    if not allow_test_oracle and "/test" in normalized:
        raise ValueError(
            f"{path} looks test-derived. Pass --allow-test-oracle only for a reporting-only audit, "
            "not for model selection."
        )
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_int(value, default: int | None = None) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return default


def to_float(value, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    if text.lower() == "inf":
        return float("inf")
    if text.lower() == "-inf":
        return float("-inf")
    try:
        return float(text)
    except ValueError:
        return default


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def csv_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key, "")
                if (value is None or value == "") and key not in {"role"} and not key.endswith("_score_mean"):
                    value = 0
                out[key] = csv_value(value)
            writer.writerow(out)


def ensure(stats: dict[int, dict], query: int) -> dict:
    if query not in stats:
        stats[query] = {"query": int(query), "_score_lists": {}}
    return stats[query]


def inc(item: dict, key: str, value: int = 1) -> None:
    item[key] = int(item.get(key, 0)) + int(value)


def add_score(item: dict, key: str, value: float) -> None:
    if math.isfinite(value):
        item.setdefault("_score_lists", {}).setdefault(key, []).append(float(value))


def add_raw_events(
    stats: dict[int, dict],
    rows: list[dict],
    *,
    label: str,
    gt5_short_visible_max: int,
    gt5_weak_visible_max: int,
    gt34_visible_max: int,
    hit_thr_px: float,
    near_thr_px: float,
) -> None:
    for row in rows:
        query = to_int(row.get("raw_best_query_id"), default=None)
        if query is None:
            continue
        gt_count = to_int(row.get("gt_count"), default=0) or 0
        visible = to_int(row.get("visible_points_gt"), default=0) or 0
        ape = to_float(row.get("raw_best_ape_px"))
        score = to_float(row.get("raw_best_exist_score"))
        item = ensure(stats, query)

        if gt_count == 5 and visible <= int(gt5_short_visible_max):
            prefix = f"{label}_true_gt5_le{int(gt5_short_visible_max)}"
            inc(item, f"{prefix}_total")
            add_score(item, f"{prefix}_score", score)
            if math.isfinite(ape) and ape <= float(hit_thr_px):
                inc(item, f"{prefix}_hit20")
            elif math.isfinite(ape) and ape <= float(near_thr_px):
                inc(item, f"{prefix}_near20_40")
            else:
                inc(item, f"{prefix}_miss40")

        if gt_count == 5 and visible <= int(gt5_weak_visible_max):
            prefix = f"{label}_true_gt5_le{int(gt5_weak_visible_max)}"
            inc(item, f"{prefix}_total")
            if math.isfinite(ape) and ape <= float(hit_thr_px):
                inc(item, f"{prefix}_hit20")

        if gt_count in {3, 4} and visible <= int(gt34_visible_max):
            inc(item, f"{label}_gt34_le{int(gt34_visible_max)}_rawbest_total")


def add_gt5_extra_events(
    stats: dict[int, dict],
    rows: list[dict],
    *,
    prefix: str,
    score_risk_thr: float,
) -> None:
    for row in rows:
        query = to_int(row.get("extra_query"), default=None)
        if query is None:
            continue
        item = ensure(stats, query)
        inc(item, f"{prefix}_extra_total")
        score = to_float(row.get("exist_score"))
        add_score(item, f"{prefix}_score", score)
        if math.isfinite(score) and score < float(score_risk_thr):
            inc(item, f"{prefix}_score_lt_02")

        category = str(row.get("category", "")).strip().lower()
        if category == "boundary_pseudo" or to_bool(row.get("is_boundary_pseudo")):
            inc(item, f"{prefix}_boundary")
        elif category == "ambiguous":
            inc(item, f"{prefix}_ambiguous")
        elif category == "spurious" or to_bool(row.get("is_spurious")):
            inc(item, f"{prefix}_spurious")
        elif category == "duplicate" or to_bool(row.get("is_duplicate")):
            inc(item, f"{prefix}_duplicate")
        else:
            inc(item, f"{prefix}_ambiguous")


def add_false_extra_events(stats: dict[int, dict], rows: list[dict], *, prefix: str) -> None:
    for row in rows:
        query = to_int(row.get("extra_query"), default=None)
        if query is None:
            continue
        item = ensure(stats, query)
        inc(item, f"{prefix}_extra_total")
        category = str(row.get("category", "")).strip().lower()
        if category == "boundary_pseudo" or to_bool(row.get("is_boundary_pseudo")):
            inc(item, f"{prefix}_boundary")
        elif category == "ambiguous":
            inc(item, f"{prefix}_ambiguous")
        elif category == "spurious" or to_bool(row.get("is_spurious")):
            inc(item, f"{prefix}_spurious")
        elif category == "duplicate" or to_bool(row.get("is_duplicate")):
            inc(item, f"{prefix}_duplicate")
        else:
            inc(item, f"{prefix}_ambiguous")


def finalize_scores(item: dict) -> None:
    for key, values in item.pop("_score_lists", {}).items():
        item[f"{key}_mean"] = "" if not values else round(float(mean(values)), 6)


def main() -> None:
    args = parse_args()
    if int(args.gt5_short_visible_max) <= 0:
        raise ValueError("--gt5-short-visible-max must be > 0.")
    if int(args.gt5_weak_visible_max) < int(args.gt5_short_visible_max):
        raise ValueError("--gt5-weak-visible-max must be >= --gt5-short-visible-max.")

    raw_specs: list[tuple[str, Path]] = []
    if args.val_raw:
        raw_specs.append(("val", Path(args.val_raw)))
    if args.train0601_raw:
        raw_specs.append(("train0601", Path(args.train0601_raw)))
    if args.train0531_raw:
        raw_specs.append(("train0531", Path(args.train0531_raw)))
    raw_specs.extend(labeled_path(x, "raw") for x in args.raw)

    gt5_extra_specs = [labeled_path(x, "val_gt5_to6") for x in args.gt5_extra]
    false_extra_specs = [labeled_path(x, "val_gt34_false") for x in args.false_extra]

    if not raw_specs:
        raise ValueError("At least one raw diagnostic CSV is required.")
    if not gt5_extra_specs and not false_extra_specs:
        raise ValueError("At least one extra-lane diagnostic CSV is required.")

    stats: dict[int, dict] = {}
    gt5_queries = set(parse_query_spec(args.gt5_queries))
    gt4_queries = set(parse_query_spec(args.gt4_queries))
    watch_queries = set(parse_query_spec(args.watch_queries))
    if gt5_queries & gt4_queries:
        raise ValueError(f"GT5 and GT4 query sets overlap: {sorted(gt5_queries & gt4_queries)}")
    for query in sorted(gt5_queries | gt4_queries | watch_queries):
        ensure(stats, query)

    raw_inputs = []
    for label, path in raw_specs:
        rows = read_csv(path, allow_test_oracle=bool(args.allow_test_oracle))
        add_raw_events(
            stats,
            rows,
            label=label,
            gt5_short_visible_max=int(args.gt5_short_visible_max),
            gt5_weak_visible_max=int(args.gt5_weak_visible_max),
            gt34_visible_max=int(args.gt34_visible_max),
            hit_thr_px=float(args.hit_thr_px),
            near_thr_px=float(args.near_thr_px),
        )
        raw_inputs.append({"label": label, "path": str(path), "rows": len(rows)})

    gt5_extra_inputs = []
    for label, path in gt5_extra_specs:
        rows = read_csv(path, allow_test_oracle=bool(args.allow_test_oracle))
        add_gt5_extra_events(stats, rows, prefix=label, score_risk_thr=float(args.score_risk_thr))
        gt5_extra_inputs.append({"label": label, "path": str(path), "rows": len(rows)})

    false_extra_inputs = []
    for label, path in false_extra_specs:
        rows = read_csv(path, allow_test_oracle=bool(args.allow_test_oracle))
        add_false_extra_events(stats, rows, prefix="val_gt34_false")
        false_extra_inputs.append({"label": label, "path": str(path), "rows": len(rows)})

    short_label = f"true_gt5_le{int(args.gt5_short_visible_max)}"
    weak_label = f"true_gt5_le{int(args.gt5_weak_visible_max)}"
    gt34_label = f"gt34_le{int(args.gt34_visible_max)}_rawbest_total"

    output_rows = []
    for query in sorted(stats):
        item = stats[query]
        finalize_scores(item)
        if query in gt5_queries:
            role = "gt5_bank"
        elif query in gt4_queries:
            role = "gt4_bank"
        elif query in watch_queries:
            role = "protected_watch"
        else:
            role = "observed"
        row = {"query": int(query), "role": role}
        row.update({k: v for k, v in item.items() if not k.startswith("_") and k != "query"})

        val_hit = int(row.get(f"val_{short_label}_hit20", 0))
        val_near = int(row.get(f"val_{short_label}_near20_40", 0))
        risk = int(row.get("val_gt5_to6_extra_total", 0)) + int(row.get("val_gt34_false_extra_total", 0))
        useful = val_hit + val_near
        row["risk_events"] = risk
        row["useful_events"] = useful
        row["net_useful_minus_risk"] = useful - risk
        row["gate_high_true_low_risk"] = bool(val_hit >= int(args.min_val_gt5_hit20) and risk <= int(args.max_risk_events))
        output_rows.append(row)

    raw_labels = [label for label, _ in raw_specs]
    fields = ["query", "role"]
    for label in raw_labels:
        fields.extend(
            [
                f"{label}_{short_label}_total",
                f"{label}_{short_label}_hit20",
                f"{label}_{short_label}_near20_40",
                f"{label}_{short_label}_miss40",
                f"{label}_{short_label}_score_mean",
                f"{label}_{weak_label}_total",
                f"{label}_{weak_label}_hit20",
                f"{label}_{gt34_label}",
            ]
        )
    fields.extend(
        [
            "val_gt5_to6_extra_total",
            "val_gt5_to6_boundary",
            "val_gt5_to6_ambiguous",
            "val_gt5_to6_spurious",
            "val_gt5_to6_duplicate",
            "val_gt5_to6_score_mean",
            "val_gt5_to6_score_lt_02",
            "val_gt34_false_extra_total",
            "val_gt34_false_boundary",
            "val_gt34_false_ambiguous",
            "val_gt34_false_spurious",
            "val_gt34_false_duplicate",
            "risk_events",
            "useful_events",
            "net_useful_minus_risk",
            "gate_high_true_low_risk",
        ]
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    per_query_csv = save_dir / "per_query_event_gate.csv"
    write_csv(per_query_csv, output_rows, fields)

    clean_queries = [
        {
            "query": int(row["query"]),
            "role": row["role"],
            "val_true_gt5_hit20": int(row.get(f"val_{short_label}_hit20", 0)),
            "risk_events": int(row.get("risk_events", 0)),
        }
        for row in output_rows
        if bool(row.get("gate_high_true_low_risk"))
    ]
    high_risk_queries = [
        {
            "query": int(row["query"]),
            "role": row["role"],
            "risk_events": int(row.get("risk_events", 0)),
            "val_true_gt5_hit20": int(row.get(f"val_{short_label}_hit20", 0)),
            "val_true_gt5_near20_40": int(row.get(f"val_{short_label}_near20_40", 0)),
            "val_gt5_to6_extra_total": int(row.get("val_gt5_to6_extra_total", 0)),
            "val_gt34_false_extra_total": int(row.get("val_gt34_false_extra_total", 0)),
        }
        for row in output_rows
        if int(row.get("risk_events", 0)) >= int(args.high_risk_events)
    ]
    high_risk_queries = sorted(high_risk_queries, key=lambda x: (-x["risk_events"], x["query"]))

    summary = {
        "run": args.run_name,
        "test_used": bool(args.allow_test_oracle),
        "queries": [int(row["query"]) for row in output_rows],
        "gt5_bank": sorted(gt5_queries),
        "gt4_bank": sorted(gt4_queries),
        "watch_queries": sorted(watch_queries),
        "per_query_csv": str(per_query_csv),
        "raw_inputs": raw_inputs,
        "gt5_extra_inputs": gt5_extra_inputs,
        "false_extra_inputs": false_extra_inputs,
        "total_val_gt5_to6": int(sum(int(row.get("val_gt5_to6_extra_total", 0)) for row in output_rows)),
        "total_val_gt34_false_extra": int(sum(int(row.get("val_gt34_false_extra_total", 0)) for row in output_rows)),
        "total_val_true_gt5_le10_rawbest": int(sum(int(row.get(f"val_{short_label}_total", 0)) for row in output_rows)),
        "total_train0601_true_gt5_le10_rawbest": int(
            sum(int(row.get(f"train0601_{short_label}_total", 0)) for row in output_rows)
        ),
        "clean_high_true_low_risk_queries": clean_queries,
        "no_query_has_high_true_low_risk": not bool(clean_queries),
        "high_risk_queries": high_risk_queries,
        "gate": {
            "min_val_gt5_hit20": int(args.min_val_gt5_hit20),
            "max_risk_events": int(args.max_risk_events),
            "high_risk_events": int(args.high_risk_events),
            "hit_thr_px": float(args.hit_thr_px),
            "near_thr_px": float(args.near_thr_px),
            "score_risk_thr": float(args.score_risk_thr),
        },
    }
    summary_path = save_dir / "event_gate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("total_val_gt5_to6", "total_val_gt34_false_extra", "no_query_has_high_true_low_risk")}, indent=2))
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
