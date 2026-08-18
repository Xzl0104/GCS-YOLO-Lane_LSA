"""Selection-key definitions for TuSimple official-val sweeps and checkpoints."""

from __future__ import annotations

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


def selection_policy(name: str, ordered_keys: tuple[dict[str, str], ...]) -> dict[str, Any]:
    """Return a JSON-serializable selection-policy object."""
    return {"name": name, "ordered_keys": [dict(item) for item in ordered_keys]}


SWEEP_SELECTION_POLICY = selection_policy("official_sweep_v3", SWEEP_SELECTION_KEYS)
OFFICIAL_SELECTION_POLICY = selection_policy("official_best_v4", OFFICIAL_BEST_SELECTION_KEYS)


def _copy_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Return a defensive copy of a JSON-serializable selection policy."""
    return {"name": str(policy["name"]), "ordered_keys": [dict(item) for item in policy["ordered_keys"]]}


def sweep_selection_policy() -> dict[str, Any]:
    """Return the policy used by tools/sweep_tusimple_official.py."""
    return _copy_policy(SWEEP_SELECTION_POLICY)


def official_best_selection_policy() -> dict[str, Any]:
    """Return the cross-epoch policy used by training-time official_best."""
    return _copy_policy(OFFICIAL_SELECTION_POLICY)


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


def official_best_sort_key(best: dict[str, Any], epoch: int) -> tuple:
    """Return the exact training-time official_best sort key."""
    row = dict(best)
    row["epoch"] = int(epoch)
    return selection_sort_key(row, OFFICIAL_BEST_SELECTION_KEYS)
