# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Check final-query extra-lane gates before any reporting-only TuSimple TEST run.

The input summaries are produced by ``tools/diagnose_tusimple_final_query_extra_lanes.py``.
They use GT after decode for diagnostics, so this checker is intentionally for
official-val and train-derived gates only. It must not be used to choose from
TEST summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


FOCUS_GT5_TO6_QUERIES = (1, 3, 5, 6, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-val-summary", required=True, help="Candidate official-val final_query_extra_summary.json.")
    parser.add_argument("--baseline-val-summary", required=True, help="Env30 official-val final_query_extra_summary.json.")
    parser.add_argument(
        "--candidate-train-summary",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Candidate train split final_query_extra_summary.json for q4 gate, e.g. train0601=path.",
    )
    parser.add_argument(
        "--baseline-train-summary",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Env30 train split final_query_extra_summary.json for q4 gate, e.g. train0601=path.",
    )
    parser.add_argument("--over-images-margin", type=int, default=3)
    parser.add_argument("--under-images-margin", type=int, default=2)
    parser.add_argument("--q5-q6-total-extra-max", type=int, default=2)
    parser.add_argument("--save-json", default=None, help="Optional gate report JSON path.")
    parser.add_argument(
        "--allow-test-summary",
        action="store_true",
        help="Allow split=test summaries only for reporting audits. Do not use with pre-TEST gating.",
    )
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    data["_path"] = str(path)
    return data


def _parse_labeled_paths(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in str(value):
            raise ValueError(f"Expected LABEL=PATH, got {value!r}.")
        label, path = str(value).split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Expected non-empty LABEL=PATH, got {value!r}.")
        if label in out:
            raise ValueError(f"Duplicate train summary label {label!r}.")
        out[label] = path
    return out


def _hist(summary: dict[str, Any], key: str) -> dict[str, int]:
    value = summary.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{summary.get('_path')} is missing dict field {key!r}.")
    return {str(k): int(v) for k, v in value.items()}


def _hist_sum(summary: dict[str, Any], key: str, queries: tuple[int, ...] | list[int] | None = None) -> int:
    hist = _hist(summary, key)
    if queries is None:
        return int(sum(hist.values()))
    return int(sum(hist.get(str(int(q)), 0) for q in queries))


def _int_field(summary: dict[str, Any], key: str) -> int:
    if key not in summary:
        raise ValueError(f"{summary.get('_path')} is missing field {key!r}.")
    return int(summary[key])


def _split(summary: dict[str, Any]) -> str:
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{summary.get('_path')} is missing config object.")
    return str(config.get("split", "")).strip()


def _validate_summary(summary: dict[str, Any], expected_split: str | None, *, allow_test: bool) -> None:
    split = _split(summary)
    if split == "test" and not allow_test:
        raise ValueError(
            f"{summary.get('_path')} has split=test. Final-query extra gate is pre-TEST only; "
            "do not use TEST GT diagnostics for selection."
        )
    if expected_split is not None and split != expected_split:
        raise ValueError(f"{summary.get('_path')} split must be {expected_split!r}, got {split!r}.")
    config = summary["config"]
    if bool(config.get("uses_gt_for_decode", False)):
        raise ValueError(f"{summary.get('_path')} reports uses_gt_for_decode=True, which is invalid for this gate.")
    required = (
        "over_images",
        "under_images",
        "gt4_to5_extra_query_hist",
        "gt4_to6_extra_query_hist",
        "gt5_to6_extra_query_hist",
        "final_query_hist_extra_candidate",
        "nearest_gt_far_extra_by_query",
    )
    for key in required:
        if key not in summary:
            raise ValueError(f"{summary.get('_path')} is missing required field {key!r}.")


def _check(checks: list[dict[str, Any]], name: str, ok: bool, **details: Any) -> None:
    checks.append({"name": name, "pass": bool(ok), **details})


def main() -> None:
    args = parse_args()
    candidate_val = _load_json(args.candidate_val_summary)
    baseline_val = _load_json(args.baseline_val_summary)
    _validate_summary(candidate_val, "val", allow_test=bool(args.allow_test_summary))
    _validate_summary(baseline_val, "val", allow_test=bool(args.allow_test_summary))

    candidate_train_paths = _parse_labeled_paths(args.candidate_train_summary)
    baseline_train_paths = _parse_labeled_paths(args.baseline_train_summary)
    if set(candidate_train_paths) != set(baseline_train_paths):
        raise ValueError(
            "Candidate and baseline train summary labels must match. "
            f"candidate={sorted(candidate_train_paths)} baseline={sorted(baseline_train_paths)}"
        )

    checks: list[dict[str, Any]] = []

    cand_over = _int_field(candidate_val, "over_images")
    base_over = _int_field(baseline_val, "over_images")
    over_limit = base_over + int(args.over_images_margin)
    _check(checks, "official_val_over_images", cand_over <= over_limit, candidate=cand_over, baseline=base_over, limit=over_limit)

    gt5_to6 = _hist_sum(candidate_val, "gt5_to6_extra_query_hist")
    _check(checks, "official_val_gt5_5to6_extra_zero", gt5_to6 == 0, candidate=gt5_to6, limit=0)

    gt4_to6 = _hist_sum(candidate_val, "gt4_to6_extra_query_hist")
    _check(checks, "official_val_gt4_4to6_extra_zero", gt4_to6 == 0, candidate=gt4_to6, limit=0)

    cand_gt4_to5 = _hist_sum(candidate_val, "gt4_to5_extra_query_hist")
    base_gt4_to5 = _hist_sum(baseline_val, "gt4_to5_extra_query_hist")
    _check(
        checks,
        "official_val_gt4_4to5_extra_le_baseline",
        cand_gt4_to5 <= base_gt4_to5,
        candidate=cand_gt4_to5,
        baseline=base_gt4_to5,
    )

    focus_gt5_to6 = _hist_sum(candidate_val, "gt5_to6_extra_query_hist", list(FOCUS_GT5_TO6_QUERIES))
    _check(
        checks,
        "official_val_focus_queries_gt5_5to6_zero",
        focus_gt5_to6 == 0,
        queries=list(FOCUS_GT5_TO6_QUERIES),
        candidate=focus_gt5_to6,
        limit=0,
    )

    q5_q6_extra = _hist_sum(candidate_val, "final_query_hist_extra_candidate", [5, 6])
    _check(
        checks,
        "official_val_q5_q6_total_extra",
        q5_q6_extra <= int(args.q5_q6_total_extra_max),
        candidate=q5_q6_extra,
        limit=int(args.q5_q6_total_extra_max),
    )

    cand_under = _int_field(candidate_val, "under_images")
    base_under = _int_field(baseline_val, "under_images")
    under_limit = base_under + int(args.under_images_margin)
    _check(
        checks,
        "official_val_under_images",
        cand_under <= under_limit,
        candidate=cand_under,
        baseline=base_under,
        limit=under_limit,
    )

    train_reports = {}
    for label in sorted(candidate_train_paths):
        cand = _load_json(candidate_train_paths[label])
        base = _load_json(baseline_train_paths[label])
        _validate_summary(cand, "train", allow_test=bool(args.allow_test_summary))
        _validate_summary(base, "train", allow_test=bool(args.allow_test_summary))

        cand_q4_4to5 = _hist_sum(cand, "gt4_to5_extra_query_hist", [4])
        base_q4_4to5 = _hist_sum(base, "gt4_to5_extra_query_hist", [4])
        _check(
            checks,
            f"{label}_q4_gt4_4to5_extra_le_baseline",
            cand_q4_4to5 <= base_q4_4to5,
            candidate=cand_q4_4to5,
            baseline=base_q4_4to5,
        )

        cand_q4_4to6 = _hist_sum(cand, "gt4_to6_extra_query_hist", [4])
        _check(
            checks,
            f"{label}_q4_gt4_4to6_extra_zero",
            cand_q4_4to6 == 0,
            candidate=cand_q4_4to6,
            limit=0,
        )

        cand_q4_far = _hist_sum(cand, "nearest_gt_far_extra_by_query", [4])
        base_q4_far = _hist_sum(base, "nearest_gt_far_extra_by_query", [4])
        _check(
            checks,
            f"{label}_q4_far_extra_le_baseline",
            cand_q4_far <= base_q4_far,
            candidate=cand_q4_far,
            baseline=base_q4_far,
        )
        train_reports[label] = {
            "candidate": cand.get("_path"),
            "baseline": base.get("_path"),
            "q4_gt4_4to5_extra": cand_q4_4to5,
            "baseline_q4_gt4_4to5_extra": base_q4_4to5,
            "q4_gt4_4to6_extra": cand_q4_4to6,
            "q4_far_extra": cand_q4_far,
            "baseline_q4_far_extra": base_q4_far,
        }

    passed = all(bool(item["pass"]) for item in checks)
    output = {
        "schema": "gcs_final_query_extra_gate_v1",
        "pass": passed,
        "candidate_val_summary": candidate_val.get("_path"),
        "baseline_val_summary": baseline_val.get("_path"),
        "test_summary_allowed": bool(args.allow_test_summary),
        "official_val": {
            "candidate_over_images": cand_over,
            "baseline_over_images": base_over,
            "candidate_under_images": cand_under,
            "baseline_under_images": base_under,
            "candidate_gt5_5to6_extra": gt5_to6,
            "candidate_gt4_4to6_extra": gt4_to6,
            "candidate_gt4_4to5_extra": cand_gt4_to5,
            "baseline_gt4_4to5_extra": base_gt4_to5,
            "candidate_q5_q6_total_extra": q5_q6_extra,
        },
        "train_q4": train_reports,
        "checks": checks,
    }

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps({"pass": passed, "failed": [c for c in checks if not c["pass"]]}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
