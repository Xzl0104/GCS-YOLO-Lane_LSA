"""Select a diagnostic short-segment checkpoint by hard selected-gated hit20.

This tool reads hard official-GT diagnostic summaries. It does not run TEST,
does not change official metrics, and only copies an already-produced training
checkpoint to a diagnostic ``segment_best.pt`` path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a short-segment checkpoint from hard diagnostic summaries.")
    parser.add_argument(
        "--candidate",
        nargs=4,
        action="append",
        metavar=("NAME", "WEIGHTS", "VAL_SUMMARY", "TRAIN_SUMMARY"),
        required=True,
        help="Candidate name, checkpoint path, val hard summary, and train0601 hard summary.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--copy-to", default=None, help="Optional destination for the selected checkpoint copy.")
    return parser.parse_args()


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing summary JSON: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _group(summary: dict[str, Any], name: str) -> dict[str, Any]:
    groups = summary.get("groups", {})
    value = groups.get(name, {})
    return value if isinstance(value, dict) else {}


def _metric(group: dict[str, Any], key: str) -> float:
    value = group.get(key, 0)
    return float(value) if value is not None else 0.0


def _candidate_record(name: str, weights: str, val_summary: str, train_summary: str) -> dict[str, Any]:
    val = _read_json(val_summary)
    train = _read_json(train_summary)
    val_gt5 = _group(val, "short_gt5")
    val_gt4 = _group(val, "short_gt4")
    train_gt5 = _group(train, "train0601_short_gt5")
    train_gt4 = _group(train, "train0601_short_gt4")
    return {
        "name": str(name),
        "weights": str(Path(weights)),
        "val_summary": str(Path(val_summary)),
        "train_summary": str(Path(train_summary)),
        "val_short_gt5_total": int(_metric(val_gt5, "total")),
        "val_short_gt5_selected_gated_hit20": int(_metric(val_gt5, "selected_gated_hit20")),
        "val_short_gt4_total": int(_metric(val_gt4, "total")),
        "val_short_gt4_selected_gated_hit20": int(_metric(val_gt4, "selected_gated_hit20")),
        "train0601_short_gt5_total": int(_metric(train_gt5, "total")),
        "train0601_short_gt5_selected_gated_hit20": int(_metric(train_gt5, "selected_gated_hit20")),
        "train0601_short_gt4_total": int(_metric(train_gt4, "total")),
        "train0601_short_gt4_selected_gated_hit20": int(_metric(train_gt4, "selected_gated_hit20")),
        "val_short_gt5_raw_oracle_missed_by_selected_gated20": int(
            _metric(val_gt5, "raw_oracle_missed_by_selected_gated20")
        ),
        "train0601_short_gt5_raw_oracle_missed_by_selected_gated20": int(
            _metric(train_gt5, "raw_oracle_missed_by_selected_gated20")
        ),
    }


def _sort_key(record: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(record["val_short_gt5_selected_gated_hit20"]),
        float(record["train0601_short_gt5_selected_gated_hit20"]),
        float(record["train0601_short_gt4_selected_gated_hit20"]),
        float(record["val_short_gt4_selected_gated_hit20"]),
        -float(record["val_short_gt5_raw_oracle_missed_by_selected_gated20"]),
        -float(record["train0601_short_gt5_raw_oracle_missed_by_selected_gated20"]),
    )


def main() -> None:
    args = parse_args()
    records = [_candidate_record(*candidate) for candidate in args.candidate]
    existing = [r for r in records if Path(r["weights"]).exists()]
    if not existing:
        raise FileNotFoundError("No candidate checkpoint exists.")
    selected = max(existing, key=_sort_key)
    output = {
        "selection_policy": (
            "max val short_gt5 selected_gated_hit20, then train0601 short_gt5, "
            "then train0601 short_gt4, then val short_gt4; TEST closed"
        ),
        "selected": selected,
        "candidates": records,
        "test_closed": True,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.copy_to:
        copy_to = Path(args.copy_to)
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected["weights"], copy_to)
        output["selected"]["copied_to"] = str(copy_to)
        output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected["name"], "test_closed": True}, indent=2))


if __name__ == "__main__":
    main()
