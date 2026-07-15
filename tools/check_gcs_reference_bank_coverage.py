# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Audit Q12 fixed-y reference-bank coverage against raw GT diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_gcs_ultrashort_reference_bank import (
    FIXED_Y_DESC,
    GROUP_K,
    SCHEMA,
    _audit_refs,
    _build_eval_groups,
    _default_references,
    _load_gt_index,
    _metrics,
    _read_diag_rows,
    _sha256,
    _shape_from_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, help="Reference bank JSON.")
    parser.add_argument("--diag-csv", action="append", required=True, help="raw_gt_lane_diagnostics.csv.")
    parser.add_argument("--gt-json", action="append", required=True, help="TuSimple GT JSON matching diagnostics.")
    parser.add_argument("--save-json", required=True)
    return parser.parse_args()


def _load_bank(path: Path) -> dict:
    bank = json.loads(path.read_text(encoding="utf-8"))
    if bank.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported bank schema {bank.get('schema')!r}; expected {SCHEMA!r}.")
    if bank.get("point_mode") != "fixed_y":
        raise ValueError(f"Unsupported point_mode {bank.get('point_mode')!r}; expected 'fixed_y'.")
    if int(bank.get("num_queries", -1)) != 12:
        raise ValueError(f"Bank num_queries must be 12, got {bank.get('num_queries')}.")
    if int(bank.get("num_points", -1)) != 56:
        raise ValueError(f"Bank num_points must be 56, got {bank.get('num_points')}.")
    fixed_y = [int(x) for x in bank.get("fixed_y_px_desc", [])]
    expected = [int(x) for x in FIXED_Y_DESC.tolist()]
    if fixed_y != expected:
        raise ValueError(f"Bank fixed_y_px_desc mismatch: expected {expected}, got {fixed_y}.")
    refs = np.asarray(bank.get("x_norm"), dtype=np.float32)
    if refs.shape != (12, 56):
        raise ValueError(f"Bank x_norm shape must be (12, 56), got {tuple(refs.shape)}.")
    if not np.isfinite(refs).all():
        raise ValueError("Bank x_norm contains non-finite values.")
    if float(refs.min()) < 0.001 - 1e-7 or float(refs.max()) > 0.999 + 1e-7:
        raise ValueError(f"Bank x_norm values must be clipped to [0.001, 0.999], got min={refs.min()} max={refs.max()}.")
    return bank


def main() -> None:
    args = parse_args()
    bank_path = Path(args.bank)
    bank = _load_bank(bank_path)
    gt_index = _load_gt_index([Path(p) for p in args.gt_json])
    lanes = []
    skipped = {"missing_or_short_gt": 0}
    for row in _read_diag_rows([Path(p) for p in args.diag_csv]):
        lane = _shape_from_row(row, gt_index, num_points=56)
        if lane is None:
            skipped["missing_or_short_gt"] += 1
            continue
        lanes.append(lane)
    if not lanes:
        raise ValueError("No lanes could be joined from diagnostics and GT JSON.")

    base_refs = _default_references(12, 56)
    new_refs = np.asarray(bank["x_norm"], dtype=np.float32)
    audit = _audit_refs(base_refs, new_refs, lanes)
    groups = _build_eval_groups(lanes)
    per_group_compact = {}
    for name, group_lanes in groups.items():
        base = _metrics(base_refs, group_lanes)
        new = _metrics(new_refs, group_lanes)
        per_group_compact[name] = {
            "count": new["count"],
            "base_p50": base["p50"],
            "base_p90": base["p90"],
            "base_match20": base["match20"],
            "new_p50": new["p50"],
            "new_p90": new["p90"],
            "new_match20": new["match20"],
            "delta_p50": None if base["p50"] is None or new["p50"] is None else new["p50"] - base["p50"],
            "delta_p90": None if base["p90"] is None or new["p90"] is None else new["p90"] - base["p90"],
            "delta_match20": None
            if base["match20"] is None or new["match20"] is None
            else new["match20"] - base["match20"],
        }

    output = {
        "schema": "gcs_q12_fixed_y_reference_bank_coverage_audit_v1",
        "bank": str(bank_path),
        "bank_sha256": _sha256(bank_path),
        "diag_csv": [str(Path(p)) for p in args.diag_csv],
        "gt_json": [str(Path(p)) for p in args.gt_json],
        "bank_schema": bank.get("schema"),
        "bank_base": bank.get("base"),
        "assignment_mapping": bank.get("replacements", {}),
        "base_nearest_reference_ape_p50": audit["base_nearest_reference_ape_p50"],
        "base_nearest_reference_ape_p90": audit["base_nearest_reference_ape_p90"],
        "new_nearest_reference_ape_p50": audit["new_nearest_reference_ape_p50"],
        "new_nearest_reference_ape_p90": audit["new_nearest_reference_ape_p90"],
        "base_nearest_reference_match20": audit["base_nearest_reference_match20"],
        "new_nearest_reference_match20": audit["new_nearest_reference_match20"],
        "per_group": per_group_compact,
        "nearest_ref_query_hist": audit["nearest_ref_query_hist"],
        "normal_lane_regression": audit["normal_lane_regression"],
        "gate": audit["gate"],
        "joined_lane_count": len(lanes),
        "skipped": skipped,
    }
    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"save_json": str(save_path), "gate": output["gate"]}, indent=2))


if __name__ == "__main__":
    main()
