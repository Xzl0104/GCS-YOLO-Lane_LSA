"""Compare Q18, Q20 sidegeom, and Q20 dataref nearest-reference coverage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.gcs_lane import GCSLaneHead

from q20_dataref_common import build_dedup_report, deduplicate_rows, format_duplicate_error, load_jsonl


EXPECTED_Y = np.linspace(710 / 720, 160 / 720, 56, dtype=np.float32)
DEFAULT_OUT = ROOT / "runs" / "debug" / "q20_dataref_reference_coverage.json"


def parse_args() -> argparse.Namespace:
    """Parse reference coverage arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-missing-jsonl", nargs="*", type=Path, default=[], help="Old-denominator missing JSONL.")
    parser.add_argument("--current-missing-jsonl", nargs="*", type=Path, default=[], help="Current missing JSONL.")
    parser.add_argument(
        "--missing-jsonl",
        nargs="*",
        type=Path,
        default=[],
        help="Compatibility alias: extra old-denominator missing JSONL.",
    )
    parser.add_argument("--old-missing-csv", nargs="*", type=Path, default=[], help="Old-denominator missing CSV.")
    parser.add_argument("--current-missing-csv", nargs="*", type=Path, default=[], help="Current missing CSV.")
    parser.add_argument("--label-dir", type=Path, default=None, help="labels_gcs split dir for CSV inputs.")
    parser.add_argument("--label-source-prefix", default="train", choices=("train", "test"), help="Converted label prefix.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON report.")
    parser.add_argument("--image-width", type=float, default=1280.0, help="Original image width for TuSimple x pixels.")
    parser.add_argument("--improve-px", type=float, default=5.0, help="Minimum mean-x reduction counted as improved.")
    parser.add_argument("--min-old-improved", type=int, default=15, help="Old-denominator required improved-vs-Q18 count.")
    parser.add_argument("--require-old-min-lanes", type=int, default=20, help="Apply old gate at or above this count.")
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument(
        "--dedup-evidence",
        action="store_true",
        help="Deduplicate source lanes before computing formal reference-only coverage.",
    )
    evidence.add_argument(
        "--allow-duplicate-evidence",
        action="store_true",
        help="Debug only: compute duplicate-weighted coverage, but force gate.passed=false.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve repo-relative paths."""
    return path if path.is_absolute() else ROOT / path


def npz_scalar_str(value) -> str:
    """Return a scalar npz value as text."""
    item = np.asarray(value).reshape(-1)[0].item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def sample_id_from_raw_file(raw_file: str, source_prefix: str) -> str:
    """Mirror the common TuSimple converted-label sample id."""
    rel = raw_file.lstrip("/").replace("\\", "/")
    parts = PurePosixPath(rel).parts
    if len(parts) >= 4 and parts[0] == "clips":
        return f"{source_prefix}_{parts[1]}_{parts[2]}_{Path(parts[3]).stem}"
    raise ValueError(f"Cannot infer label sample_id from raw_file={raw_file!r}.")


def build_label_index(label_dir: Path) -> dict[str, Path]:
    """Build raw_file -> label path index as a fallback for nonstandard names."""
    index = {}
    for label_path in sorted(label_dir.glob("*.npz")):
        with np.load(label_path, allow_pickle=False) as data:
            if "raw_file" in data.files:
                index[npz_scalar_str(data["raw_file"])] = label_path
    return index


def label_path_for_raw(raw_file: str, label_dir: Path, source_prefix: str, index: dict[str, Path] | None) -> Path:
    """Resolve a converted GCS label path for a raw_file."""
    try:
        sample_id = sample_id_from_raw_file(raw_file, source_prefix)
        label_path = label_dir / f"{sample_id}.npz"
        if label_path.exists():
            return label_path
    except ValueError:
        label_path = label_dir / "<unknown>.npz"
    if index is not None and raw_file in index:
        return index[raw_file]
    raise FileNotFoundError(f"Missing label for raw_file={raw_file!r}: tried {label_path}")


def load_csv_rows(path: Path, group: str, label_dir: Path, source_prefix: str) -> list[dict]:
    """Load missing-lane CSV metadata and recover x/valid from converted labels."""
    rows = []
    label_index: dict[str, Path] | None = None
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_file = str(row.get("raw_file", ""))
            if not raw_file:
                raise KeyError(f"{path} row is missing raw_file.")
            if label_index is None:
                try:
                    label_path = label_path_for_raw(raw_file, label_dir, source_prefix, index=None)
                except (FileNotFoundError, ValueError):
                    label_index = build_label_index(label_dir)
                    label_path = label_path_for_raw(raw_file, label_dir, source_prefix, index=label_index)
            else:
                label_path = label_path_for_raw(raw_file, label_dir, source_prefix, index=label_index)

            lane_id = int(row.get("gt_lane_id") or row.get("lane_id") or -1)
            with np.load(label_path, allow_pickle=False) as data:
                lanes = np.asarray(data["lanes"], dtype=np.float32)
                lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
                fixed_y = np.asarray(data["fixed_y"] if "fixed_y" in data.files else lanes[0, :, 1], dtype=np.float32)
            fixed_y = fixed_y.reshape(-1)
            if lane_id < 0 or lane_id >= lanes.shape[0]:
                raise IndexError(f"{raw_file}: lane_id={lane_id} outside label lanes shape {lanes.shape}.")
            if lanes.shape[1] != 56:
                raise ValueError(f"{label_path}: expected K=56, got lanes shape {lanes.shape}.")
            if not np.allclose(fixed_y, EXPECTED_Y, atol=1e-6):
                raise ValueError(f"{label_path}: fixed_y anchors do not match 710/720 -> 160/720 K56.")
            rows.append(
                {
                    "group": group,
                    "raw_file": raw_file,
                    "lane_id": lane_id,
                    "drop_reason": str(row.get("drop_reason") or row.get("new_status") or row.get("old_drop_reason") or ""),
                    "source": path.name,
                    "x": lanes[lane_id, :, 0].astype(np.float32),
                    "valid": lane_valid[lane_id].astype(np.float32) > 0.5,
                }
            )
    return rows


def build_ref_x(num_queries: int, image_width: float, reference_mode: str = "default") -> np.ndarray:
    """Build reference x coordinates in original-image pixels."""
    head = GCSLaneHead(
        c1=128,
        num_queries=int(num_queries),
        num_points=56,
        num_decoder_layers=3,
        nhead=8,
        aux=True,
        point_mode="fixed_y",
        fixed_y_start=710 / 720,
        fixed_y_end=160 / 720,
        reference_mode=reference_mode,
    )
    return torch.sigmoid(head.point_reference_logits.detach().cpu()).numpy() * float(image_width)


def normalize_gt_x(x: np.ndarray, image_width: float) -> np.ndarray:
    """Normalize row x values to original-image pixels, masking invalid values."""
    arr = np.asarray(x, dtype=np.float32)
    arr = np.where(arr < 0, np.nan, arr)
    if np.isfinite(arr).any() and float(np.nanmax(arr)) <= 2.0:
        arr = arr * float(image_width)
    return arr


def nearest_ref_metrics(gt_x: np.ndarray, valid, ref_x: np.ndarray, image_width: float) -> dict | None:
    """Return nearest reference metrics for one GT lane."""
    gt_x = normalize_gt_x(gt_x, image_width=image_width)
    valid_arr = np.asarray(valid, dtype=bool) & np.isfinite(gt_x)
    if gt_x.shape[0] != ref_x.shape[1]:
        raise ValueError(f"GT length {gt_x.shape[0]} != ref K {ref_x.shape[1]}")
    if valid_arr.shape[0] != gt_x.shape[0]:
        raise ValueError(f"valid length {valid_arr.shape[0]} != GT length {gt_x.shape[0]}")
    if int(valid_arr.sum()) < 2:
        return None

    gt = gt_x[valid_arr]
    refs = ref_x[:, valid_arr]
    abs_err = np.abs(refs - gt[None, :])
    mean_err = abs_err.mean(axis=1)
    best_idx = int(np.argmin(mean_err))
    best_abs = abs_err[best_idx]
    return {
        "best_query": best_idx,
        "mean_abs_x": float(best_abs.mean()),
        "p75_abs_x": float(np.percentile(best_abs, 75)),
        "max_abs_x": float(best_abs.max()),
        "valid_points": int(valid_arr.sum()),
    }


def summarize_group(records: list[dict], improve_px: float, old_gate_min: int, old_gate_lanes: int) -> dict:
    """Summarize one coverage group."""
    if not records:
        return {
            "lanes": 0,
            "q18_median_mean_abs_x": None,
            "q20_sidegeom_median_mean_abs_x": None,
            "q20_dataref_median_mean_abs_x": None,
            "improved_vs_q18": 0,
            "improved_vs_sidegeom": 0,
            "worsened_vs_sidegeom": 0,
            "gate": {"applied": False, "passed": None},
        }

    q18 = np.asarray([row["q18_mean_abs_x"] for row in records], dtype=np.float32)
    sidegeom = np.asarray([row["q20_sidegeom_mean_abs_x"] for row in records], dtype=np.float32)
    dataref = np.asarray([row["q20_dataref_mean_abs_x"] for row in records], dtype=np.float32)
    improved_vs_q18 = int(np.sum(dataref <= q18 - float(improve_px)))
    improved_vs_sidegeom = int(np.sum(dataref <= sidegeom - float(improve_px)))
    worsened_vs_sidegeom = int(np.sum(dataref >= sidegeom + float(improve_px)))
    median_sidegeom = float(np.median(sidegeom))
    median_dataref = float(np.median(dataref))
    gate_applied = len(records) >= int(old_gate_lanes)
    return {
        "lanes": int(len(records)),
        "q18_median_mean_abs_x": float(np.median(q18)),
        "q20_sidegeom_median_mean_abs_x": median_sidegeom,
        "q20_dataref_median_mean_abs_x": median_dataref,
        "dataref_median_delta_vs_q18": float(np.median(dataref - q18)),
        "dataref_median_delta_vs_sidegeom": float(np.median(dataref - sidegeom)),
        "improved_vs_q18": improved_vs_q18,
        "improved_vs_sidegeom": improved_vs_sidegeom,
        "worsened_vs_sidegeom": worsened_vs_sidegeom,
        "gate": {
            "applied": bool(gate_applied),
            "min_old_improved": int(old_gate_min),
            "require_old_min_lanes": int(old_gate_lanes),
            "passed": (
                bool(improved_vs_q18 >= int(old_gate_min) and median_dataref < median_sidegeom)
                if gate_applied
                else None
            ),
        },
    }


def main() -> None:
    """Run Q18/Q20-sidegeom/Q20-dataref reference-only comparison."""
    args = parse_args()
    label_dir = resolve_path(args.label_dir) if args.label_dir is not None else None
    if (args.old_missing_csv or args.current_missing_csv) and label_dir is None:
        raise ValueError("--label-dir is required for CSV inputs.")

    rows = []
    for path in [*args.old_missing_jsonl, *args.missing_jsonl]:
        rows.extend(load_jsonl(resolve_path(path), group="old"))
    for path in args.current_missing_jsonl:
        rows.extend(load_jsonl(resolve_path(path), group="current"))
    for path in args.old_missing_csv:
        assert label_dir is not None
        rows.extend(load_csv_rows(resolve_path(path), group="old", label_dir=label_dir, source_prefix=str(args.label_source_prefix)))
    for path in args.current_missing_csv:
        assert label_dir is not None
        rows.extend(
            load_csv_rows(resolve_path(path), group="current", label_dir=label_dir, source_prefix=str(args.label_source_prefix))
        )
    if not rows:
        raise ValueError("No missing-lane rows loaded.")

    if args.dedup_evidence:
        rows, dedup_report = deduplicate_rows(
            rows,
            image_width=float(args.image_width),
            allow_duplicate_weighting=False,
            require_no_duplicates=False,
        )
        evidence_mode = "formal_deduplicated"
        formal_eligible = True
    elif args.allow_duplicate_evidence:
        dedup_report = build_dedup_report(rows, image_width=float(args.image_width), dedup_enabled=False)
        evidence_mode = "debug_duplicate_weighted"
        formal_eligible = False
    else:
        dedup_report = build_dedup_report(rows, image_width=float(args.image_width), dedup_enabled=False)
        if int(dedup_report["duplicate_rows"]) > 0:
            raise RuntimeError(format_duplicate_error(dedup_report))
        evidence_mode = "formal_unique"
        formal_eligible = True

    refs = {
        "q18": build_ref_x(18, image_width=float(args.image_width)),
        "q20_sidegeom": build_ref_x(20, image_width=float(args.image_width), reference_mode="sidegeom"),
        "q20_dataref": build_ref_x(20, image_width=float(args.image_width), reference_mode="dataref"),
    }

    records = []
    skipped_empty = 0
    for row in rows:
        metrics = {}
        for name, ref in refs.items():
            metrics[name] = nearest_ref_metrics(row["x"], row["valid"], ref, image_width=float(args.image_width))
        if any(value is None for value in metrics.values()):
            skipped_empty += 1
            continue
        rec = {
            "group": row["group"],
            "raw_file": row.get("raw_file", ""),
            "lane_id": row.get("lane_id"),
            "drop_reason": row.get("drop_reason", ""),
            "source": row.get("source", ""),
            "valid_points": metrics["q20_dataref"]["valid_points"],
        }
        for name in refs:
            rec[f"{name}_best_query"] = metrics[name]["best_query"]
            rec[f"{name}_mean_abs_x"] = metrics[name]["mean_abs_x"]
            rec[f"{name}_p75_abs_x"] = metrics[name]["p75_abs_x"]
            rec[f"{name}_max_abs_x"] = metrics[name]["max_abs_x"]
        rec["dataref_delta_vs_q18"] = rec["q20_dataref_mean_abs_x"] - rec["q18_mean_abs_x"]
        rec["dataref_delta_vs_sidegeom"] = rec["q20_dataref_mean_abs_x"] - rec["q20_sidegeom_mean_abs_x"]
        rec["improved_vs_q18"] = rec["dataref_delta_vs_q18"] <= -float(args.improve_px)
        rec["improved_vs_sidegeom"] = rec["dataref_delta_vs_sidegeom"] <= -float(args.improve_px)
        records.append(rec)

    old_records = [row for row in records if row["group"] == "old"]
    current_records = [row for row in records if row["group"] == "current"]
    all_summary = summarize_group(records, float(args.improve_px), int(args.min_old_improved), int(args.require_old_min_lanes))
    old_summary = summarize_group(old_records, float(args.improve_px), int(args.min_old_improved), int(args.require_old_min_lanes))
    current_summary = summarize_group(current_records, float(args.improve_px), 0, int(args.require_old_min_lanes))

    dataref_better_than_sidegeom = bool(
        all_summary["lanes"] > 0
        and all_summary["q20_dataref_median_mean_abs_x"] < all_summary["q20_sidegeom_median_mean_abs_x"]
    )
    old_gate_ok = old_summary["gate"]["passed"] if old_summary["gate"]["applied"] else True
    debug_passed = bool(dataref_better_than_sidegeom and old_gate_ok)
    gate_passed = bool(formal_eligible and debug_passed)

    summary = {
        "evidence_mode": evidence_mode,
        "lanes": int(len(records)),
        "skipped_empty": int(skipped_empty),
        "image_width": float(args.image_width),
        "improve_px": float(args.improve_px),
        "dedup_report": dedup_report,
        "all": all_summary,
        "old": old_summary,
        "current": current_summary,
        "gate": {
            "passed": gate_passed,
            "debug_passed": debug_passed,
            "formal_eligible": bool(formal_eligible),
            "dataref_better_than_sidegeom_median": dataref_better_than_sidegeom,
            "old_gate_passed": old_gate_ok,
        },
        "records": records,
    }

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key != "records"}
    print(json.dumps(printable, indent=2))

    if gate_passed:
        print("OK: Q20 dataref reference-only coverage passed as FORMAL evidence.")
        return
    if debug_passed and not formal_eligible:
        print("DEBUG ONLY: Q20 dataref duplicate-weighted coverage threshold passed, but this is not formal evidence.")
        return
    if not debug_passed:
        raise AssertionError(
            "Q20 dataref reference coverage gate failed: "
            f"dataref_better_than_sidegeom={dataref_better_than_sidegeom}, old_gate={old_gate_ok}"
        )
    raise AssertionError("Q20 dataref reference coverage is not formal evidence.")


if __name__ == "__main__":
    main()
