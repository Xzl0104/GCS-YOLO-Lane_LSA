"""Regression check for Q20-dataref reference-coverage duplicate evidence guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.gcs_lane import Q20_DATAREF_X


COVERAGE = ROOT / "tools" / "check_q20_dataref_reference_coverage.py"
WORK_DIR = ROOT / ".tmp" / "q20_dataref_duplicate_guard_contract"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_duplicate_fixture() -> tuple[Path, Path]:
    """Create 42 rows with 22 unique source lanes and 20 duplicate rows."""
    valid = [1] * 56
    side_templates = [Q20_DATAREF_X[i] for i in (0, 1, 2, 3, 16, 17, 18, 19)]
    old_rows = []
    for idx in range(22):
        template = side_templates[idx % len(side_templates)]
        old_rows.append(
            {
                "raw_file": f"clips/synth/{idx:05d}/20.jpg",
                "lane_id": 0,
                "x": list(template),
                "valid": valid,
                "drop_reason": "synthetic_duplicate_guard",
                "source": "synthetic_old",
            }
        )
    current_rows = []
    for idx in range(20):
        row = dict(old_rows[idx])
        row["source"] = "synthetic_current"
        current_rows.append(row)

    old_path = WORK_DIR / "old_missing.jsonl"
    current_path = WORK_DIR / "current_missing.jsonl"
    write_jsonl(old_path, old_rows)
    write_jsonl(current_path, current_rows)
    return old_path, current_path


def run_coverage(old_path: Path, current_path: Path, out_path: Path, *extra_args: str) -> subprocess.CompletedProcess:
    """Run the coverage script with the duplicate fixture."""
    cmd = [
        sys.executable,
        str(COVERAGE),
        "--old-missing-jsonl",
        str(old_path),
        "--current-missing-jsonl",
        str(current_path),
        "--out",
        str(out_path),
        *extra_args,
    ]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=120)


def assert_contains(text: str, needle: str) -> None:
    """Assert substring presence with useful failure text."""
    if needle not in text:
        raise AssertionError(f"Expected {needle!r} in:\n{text}")


def main() -> None:
    """Run duplicate-guard regression cases."""
    old_path, current_path = make_duplicate_fixture()

    default_out = WORK_DIR / "default_should_fail.json"
    default = run_coverage(old_path, current_path, default_out)
    if default.returncode == 0:
        raise AssertionError("Default formal mode must fail on duplicate evidence.")
    combined = default.stdout + default.stderr
    assert_contains(combined, "input_rows=42")
    assert_contains(combined, "unique_source_lanes=22")
    assert_contains(combined, "duplicate_rows=20")
    assert_contains(combined, "duplicate_group_examples")

    debug_out = WORK_DIR / "debug_duplicate_weighted.json"
    debug = run_coverage(old_path, current_path, debug_out, "--allow-duplicate-evidence")
    if debug.returncode != 0:
        raise AssertionError(debug.stdout + debug.stderr)
    assert_contains(debug.stdout, "DEBUG ONLY")
    debug_summary = json.loads(debug_out.read_text(encoding="utf-8"))
    assert debug_summary["evidence_mode"] == "debug_duplicate_weighted"
    assert debug_summary["dedup_report"]["input_rows"] == 42
    assert debug_summary["dedup_report"]["unique_source_lanes"] == 22
    assert debug_summary["dedup_report"]["duplicate_rows"] == 20
    assert debug_summary["gate"]["debug_passed"] is True
    assert debug_summary["gate"]["formal_eligible"] is False
    assert debug_summary["gate"]["passed"] is False

    dedup_out = WORK_DIR / "formal_deduplicated.json"
    dedup = run_coverage(old_path, current_path, dedup_out, "--dedup-evidence")
    if dedup.returncode != 0:
        raise AssertionError(dedup.stdout + dedup.stderr)
    assert_contains(dedup.stdout, "FORMAL evidence")
    dedup_summary = json.loads(dedup_out.read_text(encoding="utf-8"))
    assert dedup_summary["evidence_mode"] == "formal_deduplicated"
    assert dedup_summary["lanes"] == 22
    assert dedup_summary["dedup_report"]["input_rows"] == 42
    assert dedup_summary["dedup_report"]["unique_source_lanes"] == 22
    assert dedup_summary["dedup_report"]["duplicate_rows"] == 20
    assert dedup_summary["dedup_report"]["output_rows_after_dedup"] == 22
    assert dedup_summary["gate"]["formal_eligible"] is True
    assert dedup_summary["gate"]["passed"] is True

    print("OK: Q20 dataref reference coverage duplicate guard contract is correct.")


if __name__ == "__main__":
    main()
