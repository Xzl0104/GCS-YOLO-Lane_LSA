"""Shared Q20-dataref missing-lane loading and source-lane dedup helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_lane_id(value: Any):
    """Return a stable lane id, preserving missing values for signature fallback."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def normalize_raw_file(raw_file: str) -> str:
    """Normalize raw_file text for source-lane deduplication."""
    raw = str(raw_file or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    parts = []
    for part in raw.split("/"):
        if not part or part == ".":
            continue
        if part == ".." and parts:
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts).lower()


def _normalize_x(x, image_width: float) -> np.ndarray:
    """Normalize pixel or normalized x values into [0, 1], keeping invalid as nan."""
    arr = np.asarray(x, dtype=np.float32)
    arr = np.where(arr < 0, np.nan, arr)
    if np.isfinite(arr).any() and float(np.nanmax(arr)) > 2.0:
        arr = arr / float(image_width)
    return arr


def valid_count(row: dict, image_width: float = 1280.0) -> int:
    """Count finite valid anchors after x normalization."""
    x = _normalize_x(row["x"], image_width=image_width)
    valid = np.asarray(row["valid"], dtype=bool) & np.isfinite(x)
    return int(valid.sum())


def _lane_signature(row: dict, image_width: float) -> tuple:
    """Build an x/valid signature when raw_file or lane_id is unavailable."""
    x = _normalize_x(row["x"], image_width=image_width)
    valid = np.asarray(row["valid"], dtype=bool) & np.isfinite(x)
    x_sig = tuple(None if not bool(v) else round(float(xi), 6) for xi, v in zip(x.tolist(), valid.tolist()))
    valid_sig = "".join("1" if bool(v) else "0" for v in valid.tolist())
    return valid_sig, x_sig


def lane_key(row: dict, image_width: float = 1280.0) -> tuple:
    """Return the canonical source-lane key used for deduplication."""
    raw_file = normalize_raw_file(row.get("raw_file", ""))
    lane_id = parse_lane_id(row.get("lane_id", None))
    if raw_file and lane_id not in {None, "", -1, "-1"}:
        return "raw_lane", raw_file, str(lane_id)
    return "signature", _lane_signature(row, image_width=image_width)


def load_jsonl(path: Path, group: str | None = None) -> list[dict]:
    """Load JSONL rows with x[56] and valid[56] arrays."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "x" not in row or "valid" not in row:
                raise KeyError(f"{path}:{line_no} must contain x and valid.")
            loaded = {
                "raw_file": str(row.get("raw_file", "")),
                "lane_id": parse_lane_id(row.get("lane_id")),
                "x": list(row["x"]),
                "valid": list(row["valid"]),
                "drop_reason": str(row.get("drop_reason", "")),
                "source": str(row.get("source", path.name)),
            }
            if group is not None:
                loaded["group"] = group
            rows.append(loaded)
    return rows


def _jsonable_key(key: tuple) -> str:
    """Return a compact printable key for duplicate reports."""
    if key and key[0] == "raw_lane":
        return f"{key[1]}#{key[2]}"
    return "signature:" + str(abs(hash(key)))


def _duplicate_group_examples(groups: dict[tuple, list[int]], rows: list[dict], max_examples: int) -> list[dict]:
    """Return compact duplicate group examples for error messages and reports."""
    examples = []
    for key, indices in groups.items():
        if len(indices) <= 1:
            continue
        first = rows[indices[0]]
        examples.append(
            {
                "key": _jsonable_key(key),
                "count": int(len(indices)),
                "indices": [int(i) for i in indices[:10]],
                "raw_file": first.get("raw_file", ""),
                "lane_id": first.get("lane_id"),
                "sources": [rows[i].get("source", "") for i in indices[:10]],
            }
        )
        if len(examples) >= int(max_examples):
            break
    return examples


def build_dedup_report(
    rows: list[dict],
    image_width: float = 1280.0,
    dedup_enabled: bool = True,
    max_duplicate_group_examples: int = 5,
) -> dict:
    """Build a stable duplicate-source-lane report."""
    groups: dict[tuple, list[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(lane_key(row, image_width=image_width), []).append(idx)

    duplicate_indices = [idx for indices in groups.values() for idx in indices[1:]]
    duplicate_rows = int(sum(max(len(indices) - 1, 0) for indices in groups.values()))
    duplicate_groups = int(sum(1 for indices in groups.values() if len(indices) > 1))
    output_rows = int(len(groups) if dedup_enabled else len(rows))
    return {
        "input_rows": int(len(rows)),
        "unique_source_lanes": int(len(groups)),
        "output_rows_after_dedup": output_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_groups": duplicate_groups,
        "duplicate_group_examples": _duplicate_group_examples(groups, rows, max_duplicate_group_examples),
        "dropped_duplicate_indices": sorted(int(i) for i in duplicate_indices) if dedup_enabled else [],
        "dedup_enabled": bool(dedup_enabled),
    }


def format_duplicate_error(report: dict) -> str:
    """Format duplicate evidence details for formal-mode failures."""
    return (
        "Duplicate Q20-dataref evidence rows are not formal evidence: "
        f"input_rows={report.get('input_rows')}, "
        f"unique_source_lanes={report.get('unique_source_lanes')}, "
        f"duplicate_rows={report.get('duplicate_rows')}, "
        f"duplicate_groups={report.get('duplicate_groups')}, "
        f"duplicate_group_examples={report.get('duplicate_group_examples')}. "
        "Use --dedup-evidence for formal deduplicated evidence or "
        "--allow-duplicate-evidence for debug-only duplicate-weighted coverage."
    )


def deduplicate_rows(
    rows: list[dict],
    image_width: float = 1280.0,
    allow_duplicate_weighting: bool = False,
    require_no_duplicates: bool = False,
) -> tuple[list[dict], dict]:
    """Deduplicate rows by source lane unless duplicate weighting is explicitly enabled."""
    dedup_enabled = not bool(allow_duplicate_weighting)
    report = build_dedup_report(rows, image_width=image_width, dedup_enabled=dedup_enabled)
    if require_no_duplicates and int(report["duplicate_rows"]) > 0:
        raise RuntimeError(format_duplicate_error(report))

    if not dedup_enabled:
        return rows, report

    seen = set()
    output_rows = []
    for row in rows:
        key = lane_key(row, image_width=image_width)
        if key in seen:
            continue
        seen.add(key)
        output_rows.append(row)
    return output_rows, report
