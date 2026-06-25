"""Compare Q18 and Q20 nearest-reference coverage on fixed missing GT lanes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.gcs_lane import GCSLaneHead


DEFAULT_OUT = ROOT / "runs" / "debug" / "q20_reference_coverage.json"
DEFAULT_LABEL_DIR = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544" / "labels_gcs" / "train"
EXPECTED_Y = torch.linspace(710 / 720, 160 / 720, 56).numpy()


def parse_args() -> argparse.Namespace:
    """Parse reference coverage arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--missing-jsonl", type=Path, help="JSONL rows with raw_file, lane_id, x, and optional valid.")
    inputs.add_argument(
        "--missing-csv",
        type=Path,
        help="per_missing_lane.csv from tools/diagnose_gt4_missing_lane_raw_queries.py.",
    )
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR, help="labels_gcs split dir for --missing-csv.")
    parser.add_argument(
        "--label-source-prefix",
        default="train",
        choices=("train", "test"),
        help="Converted label filename prefix for raw_file paths when --missing-csv is used.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON report.")
    parser.add_argument("--im-width", type=float, default=1280.0, help="Original TuSimple image width for x pixels.")
    parser.add_argument("--improve-px", type=float, default=5.0, help="Minimum mean-x reduction counted as improved.")
    parser.add_argument("--min-improved", type=int, default=13, help="Old-22 gate: required improved lane count.")
    parser.add_argument("--require-min-lanes", type=int, default=20, help="Only apply hard gate at or above this count.")
    parser.add_argument("--gt-x-normalized", action="store_true", help="Treat JSONL x values as normalized [0,1].")
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
    """Mirror gcs_tools.tusimple_utils._make_sample_id for common TuSimple paths."""
    rel = raw_file.lstrip("/").replace("\\", "/")
    parts = Path(rel).parts
    if len(parts) >= 4 and parts[0] == "clips":
        day = parts[1]
        clip = parts[2]
        frame = Path(parts[3]).stem
        return f"{source_prefix}_{day}_{clip}_{frame}"
    raise ValueError(f"Cannot infer label sample_id from raw_file={raw_file!r}; use --missing-jsonl with x/valid.")


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


def load_jsonl_rows(path: Path, im_width: float, x_normalized: bool) -> list[dict]:
    """Load missing-lane rows that already contain x and valid arrays."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "x" not in row:
                raise KeyError(f"{path}:{line_no} is missing x.")
            x = np.asarray(row["x"], dtype=np.float32)
            if x_normalized:
                x = x * float(im_width)
            valid = np.asarray(row.get("valid", x >= 0.0), dtype=bool)
            rows.append(
                {
                    "raw_file": str(row.get("raw_file", "")),
                    "lane_id": row.get("lane_id"),
                    "drop_reason": str(row.get("drop_reason", "")),
                    "x": x,
                    "valid": valid,
                }
            )
    return rows


def load_csv_rows(path: Path, label_dir: Path, source_prefix: str, im_width: float) -> list[dict]:
    """Load missing-lane rows by combining diagnostic CSV metadata with GCS labels."""
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label dir for --missing-csv: {label_dir}")
    label_index: dict[str, Path] | None = None
    rows = []
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
                fixed_y = np.asarray(data["fixed_y"] if "fixed_y" in data.files else lanes[0, :, 1], dtype=np.float32).reshape(-1)
            if lane_id < 0 or lane_id >= lanes.shape[0]:
                raise IndexError(f"{raw_file}: lane_id={lane_id} outside label lanes shape {lanes.shape}.")
            if lanes.shape[1] != 56:
                raise ValueError(f"{label_path}: expected K=56, got lanes shape {lanes.shape}.")
            if not np.allclose(fixed_y, EXPECTED_Y, atol=1e-6):
                raise ValueError(f"{label_path}: fixed_y anchors do not match 710/720 -> 160/720 K56.")
            rows.append(
                {
                    "raw_file": raw_file,
                    "lane_id": lane_id,
                    "drop_reason": str(row.get("drop_reason", "")),
                    "x": lanes[lane_id, :, 0].astype(np.float32) * float(im_width),
                    "valid": lane_valid[lane_id].astype(np.float32) > 0.5,
                }
            )
    return rows


def build_ref_x(num_queries: int, im_width: float, num_points: int = 56) -> np.ndarray:
    """Build reference x coordinates in original-image pixels."""
    head = GCSLaneHead(
        c1=128,
        num_queries=int(num_queries),
        num_points=int(num_points),
        num_decoder_layers=3,
        nhead=8,
        aux=True,
        point_mode="fixed_y",
        fixed_y_start=710 / 720,
        fixed_y_end=160 / 720,
    )
    ref = torch.sigmoid(head.point_reference_logits.detach().cpu()).numpy()
    return ref * float(im_width)


def nearest_ref_metrics(gt_x: np.ndarray, valid: np.ndarray, ref_x: np.ndarray) -> dict | None:
    """Return nearest reference metrics for one GT lane."""
    gt_x = np.asarray(gt_x, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if gt_x.shape[0] != ref_x.shape[1]:
        raise ValueError(f"GT length {gt_x.shape[0]} != ref K {ref_x.shape[1]}")
    if valid.shape[0] != gt_x.shape[0]:
        raise ValueError(f"valid length {valid.shape[0]} != GT length {gt_x.shape[0]}")
    if int(valid.sum()) == 0:
        return None

    gt = gt_x[valid]
    refs = ref_x[:, valid]
    abs_err = np.abs(refs - gt[None, :])
    mean_err = abs_err.mean(axis=1)
    best_idx = int(np.argmin(mean_err))
    best_abs = abs_err[best_idx]
    return {
        "best_query": best_idx,
        "mean_abs_x": float(best_abs.mean()),
        "p75_abs_x": float(np.percentile(best_abs, 75)),
        "max_abs_x": float(best_abs.max()),
        "valid_points": int(valid.sum()),
    }


def summarize_by_reason(records: list[dict]) -> dict[str, dict]:
    """Summarize deltas by drop_reason."""
    out = {}
    for reason in sorted({str(row.get("drop_reason", "")) for row in records}):
        group = [row for row in records if str(row.get("drop_reason", "")) == reason]
        q18 = [row["q18_mean_abs_x"] for row in group]
        q20 = [row["q20_mean_abs_x"] for row in group]
        out[reason or "unknown"] = {
            "lanes": len(group),
            "improved": int(sum(1 for row in group if row["status"] == "improved")),
            "same": int(sum(1 for row in group if row["status"] == "same")),
            "worsened": int(sum(1 for row in group if row["status"] == "worsened")),
            "q18_median_mean_abs_x": float(np.median(q18)) if q18 else None,
            "q20_median_mean_abs_x": float(np.median(q20)) if q20 else None,
            "median_delta": float(np.median(np.asarray(q20) - np.asarray(q18))) if q18 else None,
        }
    return out


def main() -> None:
    """Run Q18-vs-Q20 reference coverage comparison."""
    args = parse_args()
    out = resolve_path(args.out)
    im_width = float(args.im_width)

    if args.missing_jsonl:
        input_path = resolve_path(args.missing_jsonl)
        rows = load_jsonl_rows(input_path, im_width=im_width, x_normalized=bool(args.gt_x_normalized))
        input_kind = "jsonl"
    else:
        input_path = resolve_path(args.missing_csv)
        rows = load_csv_rows(
            input_path,
            label_dir=resolve_path(args.label_dir),
            source_prefix=str(args.label_source_prefix),
            im_width=im_width,
        )
        input_kind = "csv+labels"

    if not rows:
        raise ValueError(f"No missing-lane rows loaded from {input_path}")

    q18_ref = build_ref_x(18, im_width=im_width)
    q20_ref = build_ref_x(20, im_width=im_width)

    records = []
    improved = 0
    worsened = 0
    same = 0
    skipped_empty = 0
    for row in rows:
        m18 = nearest_ref_metrics(row["x"], row["valid"], q18_ref)
        m20 = nearest_ref_metrics(row["x"], row["valid"], q20_ref)
        if m18 is None or m20 is None:
            skipped_empty += 1
            continue

        delta = m20["mean_abs_x"] - m18["mean_abs_x"]
        if delta <= -float(args.improve_px):
            status = "improved"
            improved += 1
        elif delta >= float(args.improve_px):
            status = "worsened"
            worsened += 1
        else:
            status = "same"
            same += 1

        records.append(
            {
                "raw_file": row.get("raw_file", ""),
                "lane_id": row.get("lane_id"),
                "drop_reason": row.get("drop_reason", ""),
                "valid_points": m20["valid_points"],
                "q18_best_query": m18["best_query"],
                "q18_mean_abs_x": m18["mean_abs_x"],
                "q18_p75_abs_x": m18["p75_abs_x"],
                "q18_max_abs_x": m18["max_abs_x"],
                "q20_best_query": m20["best_query"],
                "q20_mean_abs_x": m20["mean_abs_x"],
                "q20_p75_abs_x": m20["p75_abs_x"],
                "q20_max_abs_x": m20["max_abs_x"],
                "delta_mean_abs_x": delta,
                "status": status,
            }
        )

    q18_means = [row["q18_mean_abs_x"] for row in records]
    q20_means = [row["q20_mean_abs_x"] for row in records]
    hard_gate_applied = len(records) >= int(args.require_min_lanes)
    summary = {
        "input": str(input_path),
        "input_kind": input_kind,
        "lanes": len(records),
        "skipped_empty": int(skipped_empty),
        "im_width": im_width,
        "improve_px": float(args.improve_px),
        "improved": int(improved),
        "same": int(same),
        "worsened": int(worsened),
        "drop_reason_histogram": {str(k): int(v) for k, v in sorted(Counter(row["drop_reason"] for row in records).items())},
        "q18_median_mean_abs_x": float(np.median(q18_means)) if q18_means else None,
        "q20_median_mean_abs_x": float(np.median(q20_means)) if q20_means else None,
        "median_delta": float(np.median(np.asarray(q20_means) - np.asarray(q18_means))) if q18_means else None,
        "by_drop_reason": summarize_by_reason(records),
        "gate": {
            "applied": bool(hard_gate_applied),
            "require_min_lanes": int(args.require_min_lanes),
            "min_improved": int(args.min_improved),
            "passed": None,
        },
        "records": records,
    }
    if hard_gate_applied:
        summary["gate"]["passed"] = bool(
            improved >= int(args.min_improved)
            and summary["q20_median_mean_abs_x"] is not None
            and summary["q18_median_mean_abs_x"] is not None
            and summary["q20_median_mean_abs_x"] < summary["q18_median_mean_abs_x"]
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key != "records"}
    print(json.dumps(printable, indent=2))

    if hard_gate_applied and not summary["gate"]["passed"]:
        raise AssertionError(
            "Q20 reference coverage gate failed: "
            f"improved={improved}, q18_median={summary['q18_median_mean_abs_x']}, "
            f"q20_median={summary['q20_median_mean_abs_x']}"
        )
    print("OK: Q20 reference-only coverage check passed." if hard_gate_applied else "OK: Q20 reference coverage report saved.")


if __name__ == "__main__":
    main()
