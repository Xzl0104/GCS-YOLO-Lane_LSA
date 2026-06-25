"""Check Q18 countguard dry-run metric acceptance fields."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer


REQUIRED_PROGRESS_FIELDS = (
    "cnt_ce",
    "cnt_acc",
    "gt4_pt",
    "gt4_lbp",
    "uvneg_loss",
    "uvneg_prob",
    "vlb_vprob",
    "vlb_vsum",
)
OPTIONAL_DISABLED_GT4_PROGRESS_FIELDS = ("gt4_vprob", "gt4_vsum")
REQUIRED_RESULT_FIELDS = {
    "cnt_ce": ("train/count_ce_loss", "count_ce_loss", "cnt_ce"),
    "cnt_acc": ("train/count_ce_acc", "count_ce_acc", "cnt_acc"),
    "gt4_pt": ("train/gt4_short_lane_loss", "gt4_short_lane_loss", "gt4_pt"),
    "gt4_lbp": ("train/gt4_lane_balanced_point_loss", "gt4_lane_balanced_point_loss", "gt4_lbp"),
    "uvneg_loss": ("train/unmatched_valid_neg_loss", "unmatched_valid_neg_loss", "uvneg_loss"),
    "uvneg_prob": ("train/unmatched_valid_prob_mean", "unmatched_valid_prob_mean", "uvneg_prob"),
    "vlb_vprob": ("train/valid_lb_gt4_short_pred_prob_mean", "valid_lb_gt4_short_pred_prob_mean", "vlb_vprob"),
    "vlb_vsum": ("train/valid_lb_gt4_short_pred_sum_mean", "valid_lb_gt4_short_pred_sum_mean", "vlb_vsum"),
}
DISABLED_GT4_RESULT_FIELDS = {
    "gt4_vprob": ("train/gt4_short_pred_valid_prob_mean", "gt4_short_pred_valid_prob_mean", "gt4_vprob"),
    "gt4_vsum": ("train/gt4_short_pred_valid_sum_mean", "gt4_short_pred_valid_sum_mean", "gt4_vsum"),
}


def parse_args() -> argparse.Namespace:
    """Parse optional dry-run artifact paths."""
    parser = argparse.ArgumentParser(description="Check Q18 countguard dry-run metric field contract.")
    parser.add_argument("--results-csv", type=Path, default=None, help="Optional training results.csv to validate.")
    parser.add_argument("--args-yaml", type=Path, default=None, help="Optional training args.yaml to validate disabled flags.")
    parser.add_argument("--zero-tol", type=float, default=1e-12, help="Tolerance for disabled gt4_vprob/gt4_vsum checks.")
    return parser.parse_args()


def _read_last_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if any(str(v).strip() for v in row.values())]
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows[-1]


def _find_field(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    fields = {str(key).strip(): key for key in row}
    for candidate in candidates:
        if candidate in fields:
            return fields[candidate]
    raise KeyError(f"Missing any of result fields: {candidates}")


def _float_field(row: dict[str, str], candidates: tuple[str, ...]) -> float:
    key = _find_field(row, candidates)
    value = str(row[key]).strip()
    if value == "":
        raise ValueError(f"Empty metric value for {key}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Non-finite metric value for {key}: {value}")
    return out


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_disabled_flags(path: Path | None) -> tuple[bool, bool]:
    if path is None:
        return False, False
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (
        _to_bool(data.get("gcs_gt4_short_valid_recall")),
        _to_bool(data.get("gcs_gt4_short_valid_count_floor")),
    )


def main() -> None:
    """Assert dry-run acceptance checks use the intended progress/result fields."""
    args = parse_args()
    progress_fields = tuple(GCSLaneTrainer.progress_loss_names)
    missing_progress = [field for field in REQUIRED_PROGRESS_FIELDS if field not in progress_fields]
    if missing_progress:
        raise AssertionError(f"Missing required dry-run progress fields: {missing_progress}")
    for field in OPTIONAL_DISABLED_GT4_PROGRESS_FIELDS:
        if field not in progress_fields:
            raise AssertionError(f"Missing disabled-state diagnostic progress field: {field}")

    print("OK: dry-run progress field contract is correct.")
    print("required_progress_fields:", ",".join(REQUIRED_PROGRESS_FIELDS))
    print("disabled_expected_zero_fields:", ",".join(OPTIONAL_DISABLED_GT4_PROGRESS_FIELDS))

    if args.results_csv is None:
        return

    row = _read_last_csv_row(args.results_csv)
    values = {name: _float_field(row, candidates) for name, candidates in REQUIRED_RESULT_FIELDS.items()}
    recall_enabled, floor_enabled = _load_disabled_flags(args.args_yaml)
    disabled_values = {
        name: _float_field(row, candidates)
        for name, candidates in DISABLED_GT4_RESULT_FIELDS.items()
        if any(candidate in row for candidate in candidates)
    }
    if not recall_enabled and not floor_enabled:
        nonzero_disabled = {
            name: value for name, value in disabled_values.items() if abs(float(value)) > float(args.zero_tol)
        }
        if nonzero_disabled:
            raise AssertionError(
                "gt4_vprob/gt4_vsum are non-zero while gcs_gt4_short_valid_recall and "
                f"gcs_gt4_short_valid_count_floor are disabled: {nonzero_disabled}"
            )
    elif any(abs(float(value)) > float(args.zero_tol) for value in disabled_values.values()):
        print("NOTE: gt4_vprob/gt4_vsum are non-zero because a GT4 short-valid repair flag is enabled.")

    print("OK: dry-run results fields are finite.")
    print("checked_values:", {k: round(float(v), 6) for k, v in values.items()})
    if disabled_values:
        print("gt4_disabled_values:", {k: round(float(v), 6) for k, v in disabled_values.items()})


if __name__ == "__main__":
    main()
