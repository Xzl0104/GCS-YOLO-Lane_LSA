"""Selection-key definitions for TuSimple official-val sweeps and checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SWEEP_SELECTION_KEYS = (
    {"key": "official_acc", "direction": "max"},
    {"key": "official_score", "direction": "max"},
    {"key": "official_FP", "direction": "min"},
    {"key": "official_FN", "direction": "min"},
    {"key": "count_acc_4", "direction": "max"},
    {"key": "count_acc", "direction": "max"},
    {"key": "count_acc_5", "direction": "max"},
    {"key": "conf", "direction": "min"},
    {"key": "nms_dist_px", "direction": "min"},
    {"key": "point_valid_thr", "direction": "min"},
    {"key": "max_det", "direction": "min"},
    {"key": "min_points", "direction": "min"},
)

ROBUST_SWEEP_SELECTION_KEYS = (
    {"key": "robust_score", "direction": "max"},
    {"key": "official_acc", "direction": "max"},
    {"key": "official_score", "direction": "max"},
    {"key": "official_FP", "direction": "min"},
    {"key": "official_FN", "direction": "min"},
    {"key": "count_acc_4", "direction": "max"},
    {"key": "count_acc", "direction": "max"},
    {"key": "count_acc_5", "direction": "max"},
    {"key": "conf", "direction": "min"},
    {"key": "nms_dist_px", "direction": "min"},
    {"key": "point_valid_thr", "direction": "min"},
    {"key": "max_det", "direction": "min"},
    {"key": "min_points", "direction": "min"},
)

OFFICIAL_BEST_SELECTION_KEYS = (
    {"key": "strict_order_valid", "direction": "max"},
    {"key": "ordered_slot_order_violations", "direction": "min"},
    {"key": "official_acc", "direction": "max"},
    {"key": "official_score", "direction": "max"},
    {"key": "official_FP", "direction": "min"},
    {"key": "official_FN", "direction": "min"},
    {"key": "count_acc_4", "direction": "max"},
    {"key": "count_acc", "direction": "max"},
    {"key": "count_acc_5", "direction": "max"},
    {"key": "epoch", "direction": "earliest"},
)

ROBUST_OFFICIAL_BEST_SELECTION_KEYS = (
    {"key": "strict_order_valid", "direction": "max"},
    {"key": "ordered_slot_order_violations", "direction": "min"},
    {"key": "robust_score", "direction": "max"},
    {"key": "official_acc", "direction": "max"},
    {"key": "official_score", "direction": "max"},
    {"key": "official_FP", "direction": "min"},
    {"key": "official_FN", "direction": "min"},
    {"key": "count_acc_4", "direction": "max"},
    {"key": "count_acc", "direction": "max"},
    {"key": "count_acc_5", "direction": "max"},
    {"key": "epoch", "direction": "earliest"},
)


def selection_policy(name: str, ordered_keys: tuple[dict[str, str], ...]) -> dict[str, Any]:
    """Return a JSON-serializable selection-policy object."""
    return {"name": name, "ordered_keys": [dict(item) for item in ordered_keys]}


SWEEP_SELECTION_POLICY = selection_policy("official_sweep_v4", SWEEP_SELECTION_KEYS)
ROBUST_SWEEP_SELECTION_POLICY = selection_policy("official_sweep_robust_v1", ROBUST_SWEEP_SELECTION_KEYS)
OFFICIAL_SELECTION_POLICY = selection_policy("official_best_v4", OFFICIAL_BEST_SELECTION_KEYS)
ROBUST_OFFICIAL_SELECTION_POLICY = selection_policy("official_best_robust_v1", ROBUST_OFFICIAL_BEST_SELECTION_KEYS)
CONSTRAINED_SWEEP_SELECTION_POLICY = {
    "name": "count_safe_constrained_then_official_sweep_v1",
    "prefilter": "hard official-val gates, then official_sweep_v4 ordering",
    "sweep_policy": SWEEP_SELECTION_POLICY,
}
COUNT_CONFUSION_KEY_PATTERN = re.compile(r"^[0-9]+->[0-9]+$")


DEFAULT_COUNT_SAFE_THRESHOLDS = {
    "acc_min_exclusive": 0.973330,
    "fp_max": 0.015748,
    "fn_max": 0.009642,
    "count_acc4_min": 0.969697,
    "count_acc5_min": 0.986486,
    "gt4_to3_max": 1,
    "gt4_to5_max": 0,
    "gt5_to4_max": 1,
    "allow_output6": False,
}


def _copy_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Return a defensive copy of a JSON-serializable selection policy."""
    return {"name": str(policy["name"]), "ordered_keys": [dict(item) for item in policy["ordered_keys"]]}


def sweep_selection_policy() -> dict[str, Any]:
    """Return the policy used by tools/sweep_tusimple_official.py."""
    return _copy_policy(SWEEP_SELECTION_POLICY)


def constrained_sweep_selection_policy() -> dict[str, Any]:
    """Return the policy used by pre-registered count-safe sweep selection."""
    return {
        "name": str(CONSTRAINED_SWEEP_SELECTION_POLICY["name"]),
        "prefilter": str(CONSTRAINED_SWEEP_SELECTION_POLICY["prefilter"]),
        "sweep_policy": _copy_policy(SWEEP_SELECTION_POLICY),
    }


def policy_sha256(policy: dict[str, Any]) -> str:
    """Return a stable SHA256 for a JSON-serializable selection policy object."""
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def robust_sweep_selection_policy() -> dict[str, Any]:
    """Return the robust sweep-selection policy for constrained official-val sweeps."""
    return _copy_policy(ROBUST_SWEEP_SELECTION_POLICY)


def official_best_selection_policy() -> dict[str, Any]:
    """Return the cross-epoch policy used by training-time official_best."""
    return _copy_policy(OFFICIAL_SELECTION_POLICY)


def robust_official_best_selection_policy() -> dict[str, Any]:
    """Return the robust cross-epoch policy used by training-time official_best."""
    return _copy_policy(ROBUST_OFFICIAL_SELECTION_POLICY)


def _numeric(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def selection_sort_key(row: dict[str, Any], ordered_keys: tuple[dict[str, str], ...]) -> tuple:
    """Build a max()-compatible tuple from an ordered selection-key definition."""
    values = []
    for item in ordered_keys:
        key = item["key"]
        direction = item["direction"]
        value = row.get(key, 0)
        if direction == "max":
            values.append(_numeric(value))
        elif direction == "min":
            values.append(-_numeric(value))
        elif direction == "latest":
            values.append(int(value or 0))
        elif direction == "earliest":
            values.append(-int(value or 0))
        else:
            raise ValueError(f"Unsupported selection direction {direction!r} for key {key!r}.")
    return tuple(values)


def sweep_sort_key(row: dict[str, Any]) -> tuple:
    """Return the exact official sweep sort key."""
    return selection_sort_key(row, SWEEP_SELECTION_KEYS)


def robust_sweep_sort_key(row: dict[str, Any]) -> tuple:
    """Return the robust official sweep sort key."""
    return selection_sort_key(row, ROBUST_SWEEP_SELECTION_KEYS)


def official_best_sort_key(best: dict[str, Any], epoch: int) -> tuple:
    """Return the exact training-time official_best sort key."""
    row = dict(best)
    row["epoch"] = int(epoch)
    return selection_sort_key(row, OFFICIAL_BEST_SELECTION_KEYS)


def count_safe_thresholds(**overrides: Any) -> dict[str, Any]:
    """Return count-safe official-val hard gates with optional overrides."""
    thresholds = dict(DEFAULT_COUNT_SAFE_THRESHOLDS)
    for key, value in overrides.items():
        if value is not None:
            thresholds[key] = value
    thresholds["allow_output6"] = bool(thresholds.get("allow_output6", False))
    return thresholds


def _valid_count_confusion(confusion: Any) -> bool:
    if not isinstance(confusion, dict) or not confusion:
        return False
    for key, value in confusion.items():
        if not COUNT_CONFUSION_KEY_PATTERN.match(str(key)):
            return False
        try:
            int(value)
        except (TypeError, ValueError):
            try:
                int(float(value))
            except (TypeError, ValueError):
                return False
    return True


def _count(confusion: dict[str, Any], key: str) -> int:
    value = confusion.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(float(value))


def _has_output_at_least(confusion: dict[str, Any], minimum_count: int) -> bool:
    for key in confusion:
        if _count(confusion, key) <= 0:
            continue
        parts = str(key).split("->", maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pred_count = int(parts[1])
        except ValueError:
            continue
        if pred_count >= minimum_count:
            return True
    return False


def _metric_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check_float(
    checks: list[dict[str, Any]],
    name: str,
    value: float | None,
    op: str,
    threshold: float,
) -> None:
    if value is None:
        passed = False
    elif op == ">":
        passed = value > threshold
    elif op == "<=":
        passed = value <= threshold
    elif op == ">=":
        passed = value >= threshold
    else:
        raise ValueError(op)
    checks.append({"name": name, "value": value, "op": op, "threshold": threshold, "passed": passed})


def _check_int(checks: list[dict[str, Any]], name: str, value: int, op: str, threshold: int) -> None:
    if op == "<=":
        passed = value <= threshold
    elif op == "==":
        passed = value == threshold
    else:
        raise ValueError(op)
    checks.append({"name": name, "value": value, "op": op, "threshold": threshold, "passed": passed})


def count_safe_row_checks(row: dict[str, Any], thresholds: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return count-safe hard-gate checks for one official sweep row."""
    thresholds = count_safe_thresholds(**(thresholds or {}))
    checks: list[dict[str, Any]] = []
    raw_confusion = row.get("count_confusion")
    if not _valid_count_confusion(raw_confusion):
        confusion = {}
        _check_int(checks, "count_confusion_present", 0, "==", 1)
    else:
        confusion = raw_confusion
        _check_int(checks, "count_confusion_present", 1, "==", 1)

    _check_float(checks, "official_acc", _metric_float(row, "official_acc"), ">", float(thresholds["acc_min_exclusive"]))
    _check_float(checks, "official_FP", _metric_float(row, "official_FP"), "<=", float(thresholds["fp_max"]))
    _check_float(checks, "official_FN", _metric_float(row, "official_FN"), "<=", float(thresholds["fn_max"]))
    _check_float(
        checks, "count_acc_4", _metric_float(row, "count_acc_4"), ">=", float(thresholds["count_acc4_min"])
    )
    _check_float(
        checks, "count_acc_5", _metric_float(row, "count_acc_5"), ">=", float(thresholds["count_acc5_min"])
    )
    _check_int(checks, "GT4 4->3", _count(confusion, "4->3"), "<=", int(thresholds["gt4_to3_max"]))
    _check_int(checks, "GT4 4->5", _count(confusion, "4->5"), "<=", int(thresholds["gt4_to5_max"]))
    _check_int(checks, "GT5 5->4", _count(confusion, "5->4"), "<=", int(thresholds["gt5_to4_max"]))
    if not bool(thresholds["allow_output6"]):
        _check_int(checks, "output6", int(_has_output_at_least(confusion, 6)), "==", 0)
    return checks


def count_safe_row_eligible(row: dict[str, Any], thresholds: dict[str, Any] | None = None) -> bool:
    """Return whether one official sweep row passes the count-safe hard gates."""
    return all(item["passed"] for item in count_safe_row_checks(row, thresholds))


def select_count_safe_sweep_row(
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select the best count-safe row, or return ``None`` when no row passes."""
    eligible_rows = [row for row in rows if count_safe_row_eligible(row, thresholds)]
    selected = dict(max(eligible_rows, key=sweep_sort_key)) if eligible_rows else None
    return selected, eligible_rows


def official_best_robust_sort_key(best: dict[str, Any], epoch: int) -> tuple:
    """Return the robust training-time official_best sort key."""
    row = dict(best)
    row["epoch"] = int(epoch)
    return selection_sort_key(row, ROBUST_OFFICIAL_BEST_SELECTION_KEYS)
