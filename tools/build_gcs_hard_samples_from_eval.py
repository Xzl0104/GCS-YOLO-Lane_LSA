from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


DEFAULT_DATASET_ROOT = ROOT / "datasets" / "tusimple_fixed_y_960x544"


def parse_list(value: str) -> list[str]:
    """Parse a comma/semicolon/space separated string list."""
    return [x for x in value.replace(";", ",").replace(" ", ",").split(",") if x]


def parse_transitions(value: str) -> set[tuple[int, int]]:
    """Parse GT->pred lane-count transitions such as '4->3,4->5,3->5'."""
    transitions: set[tuple[int, int]] = set()
    for token in value.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "->" not in token:
            raise ValueError(f"Invalid transition {token!r}; expected GT->pred, e.g. 5->4.")
        gt, pred = token.split("->", 1)
        transitions.add((int(gt.strip()), int(pred.strip())))
    if not transitions:
        raise ValueError("At least one transition is required.")
    return transitions


def is_test_summary(payload: dict, summary_path: Path) -> bool:
    """Detect likely TuSimple test summaries so hard files are not built from test failures by accident."""
    config = payload.get("config") or {}
    split = str(config.get("split", "")).lower()
    gt_json = str(config.get("gt_json", "")).replace("\\", "/").lower()
    path_text = summary_path.as_posix().lower()
    return split == "test" or "test_label" in gt_json or "/test/" in path_text or "\\test\\" in str(summary_path).lower()


def default_output_path(summary_path: Path, fmt: str) -> Path:
    """Return a default hard-sample manifest path next to the eval summary."""
    suffix = "json" if fmt == "json" else "txt"
    return summary_path.with_name(f"hard_samples_from_{summary_path.stem}.{suffix}")


def normalize_sample_id(value: Any) -> str:
    """Normalize image/raw-file identifiers for manifest matching."""
    return str(value).strip().strip("\"'").replace("\\", "/")


def sample_id_variants(value: Any) -> set[str]:
    """Return exact and stem variants for one image/raw-file identifier."""
    norm = normalize_sample_id(value)
    if not norm:
        return set()
    variants = {norm, norm.lstrip("./")}
    # Do not collapse path-like raw_file ids such as clips/.../20.jpg to "20";
    # TuSimple frame basenames are not globally unique and would cause split leakage.
    if "/" not in norm:
        stem = Path(norm).stem
        variants.add(stem)
    return variants


def sample_matches_ids(sample_ids: set[str], target_ids: set[str]) -> bool:
    """Match exact/stem/path-suffix identifiers, mirroring the trainer hard-file behavior."""
    if sample_ids & target_ids:
        return True
    path_like_ids = [x for x in target_ids if "/" in x]
    return any(candidate.endswith(target_id) for candidate in sample_ids for target_id in path_like_ids)


def scalar_to_str(value: Any) -> str:
    """Convert numpy scalar/object values from .npz labels to a plain string."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0].item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return normalize_sample_id(value)


def label_raw_file(label_path: Path) -> str:
    """Read raw_file from one GCS .npz label if present."""
    try:
        with np.load(label_path, allow_pickle=True) as data:
            if "raw_file" not in data:
                return ""
            return scalar_to_str(data["raw_file"])
    except Exception:
        return ""


def collect_dataset_split_ids(dataset_root: Path, split: str) -> set[str]:
    """Collect sample id variants from labels_gcs/<split> for target-hit auditing."""
    label_dir = dataset_root / "labels_gcs" / split
    image_dir = dataset_root / "images" / split
    if not label_dir.exists():
        raise FileNotFoundError(f"Target label split does not exist: {label_dir}")
    ids: set[str] = set()
    for label_path in sorted(label_dir.glob("*.npz")):
        raw_file = label_raw_file(label_path)
        for value in (
            raw_file,
            label_path,
            label_path.as_posix(),
            label_path.stem,
            image_dir / f"{label_path.stem}.jpg",
            (image_dir / f"{label_path.stem}.jpg").as_posix(),
        ):
            ids.update(sample_id_variants(value))
    return ids


def audit_target_matches(rows: list[dict], dataset_root: Path, target_splits: list[str]) -> dict:
    """Report how many exported failures can actually be sampled from target splits."""
    split_ids = {split: collect_dataset_split_ids(dataset_root, split) for split in target_splits}
    split_match_counts: Counter[str] = Counter()
    unmatched: list[str] = []
    matched_any = 0
    for row in rows:
        row_ids = sample_id_variants(row["raw_file"])
        row_matches = [split for split, ids in split_ids.items() if sample_matches_ids(row_ids, ids)]
        if row_matches:
            matched_any += 1
            for split in row_matches:
                split_match_counts[split] += 1
        else:
            unmatched.append(row["raw_file"])
    return {
        "dataset_root": str(dataset_root),
        "target_splits": target_splits,
        "matched_unique_samples": int(matched_any),
        "unmatched_unique_samples": int(len(unmatched)),
        "split_match_counts": dict(sorted(split_match_counts.items())),
        "unmatched_examples": unmatched[:20],
    }


def extract_records(payload: dict, summary_path: Path) -> list[dict]:
    """Return per-image records and fail loudly on aggregate-only sweep summaries."""
    records = payload.get("records")
    if not isinstance(records, list):
        raise SystemExit(
            f"{summary_path} does not contain a top-level records list. "
            "Build hard samples from eval_tusimple_official.py or eval_gcs.py outputs, not aggregate sweep summaries."
        )
    return records


def record_raw_file(record: dict) -> str:
    """Return the best available image identifier from official or custom eval records."""
    for key in ("raw_file", "image", "im_file", "file"):
        value = normalize_sample_id(record.get(key, ""))
        if value:
            return value
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GCS hard-sample manifest from validation eval failures.")
    parser.add_argument("--eval-summary", required=True, help="Path to tusimple_official_summary.json or similar eval JSON.")
    parser.add_argument(
        "--transitions",
        default="4->3,4->5,3->5",
        help="Comma-separated GT->pred lane-count failures to export.",
    )
    parser.add_argument("--output", default=None, help="Output txt/json manifest. Defaults next to --eval-summary.")
    parser.add_argument("--format", choices=("txt", "json"), default="txt")
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Converted GCS dataset root used to audit target split hits. Use empty string to disable.",
    )
    parser.add_argument(
        "--target-splits",
        default="train",
        help="Comma/space separated dataset splits that the next training loader will sample from.",
    )
    parser.add_argument(
        "--require-target-match",
        action="store_true",
        help="Fail if exported failures do not match at least one sample in --target-splits.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Allow building from a test split summary. Avoid this for training to prevent test leakage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.eval_summary)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if is_test_summary(payload, summary_path) and not args.allow_test:
        raise SystemExit(
            "Refusing to build hard samples from a likely test summary. "
            "Use a validation/official-val summary, or pass --allow-test only for analysis artifacts."
        )

    transitions = parse_transitions(args.transitions)
    records = extract_records(payload, summary_path)
    rows = []
    transition_counts: Counter[str] = Counter()
    for record in records:
        gt = int(record.get("gt_lanes", -1))
        pred = int(record.get("pred_lanes", -1))
        if (gt, pred) not in transitions:
            continue
        raw_file = record_raw_file(record)
        if not raw_file:
            continue
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        rows.append(
            {
                "raw_file": raw_file,
                "gt_lanes": gt,
                "pred_lanes": pred,
                "Accuracy": record.get("Accuracy"),
                "FP": record.get("FP", metrics.get("fp")),
                "FN": record.get("FN", metrics.get("fn")),
            }
        )
        transition_counts[f"{gt}->{pred}"] += 1

    seen = set()
    unique_rows = []
    for row in rows:
        if row["raw_file"] in seen:
            continue
        seen.add(row["raw_file"])
        unique_rows.append(row)

    output = Path(args.output) if args.output else default_output_path(summary_path, args.format)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        output.write_text(json.dumps(unique_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        output.write_text("\n".join(row["raw_file"] for row in unique_rows) + ("\n" if unique_rows else ""), encoding="utf-8")

    target_match_audit = None
    dataset_root_arg = str(args.dataset_root or "").strip()
    if dataset_root_arg:
        dataset_root = Path(dataset_root_arg)
        if not dataset_root.is_absolute():
            dataset_root = ROOT / dataset_root
        target_splits = parse_list(args.target_splits)
        if not target_splits:
            raise SystemExit("--target-splits must name at least one split when --dataset-root is set.")
        target_match_audit = audit_target_matches(unique_rows, dataset_root, target_splits)
        if args.require_target_match and unique_rows and int(target_match_audit["matched_unique_samples"]) <= 0:
            raise SystemExit(
                "Exported failures do not match any sample in target splits. "
                f"Audit: {json.dumps(target_match_audit, ensure_ascii=False)}"
            )

    summary = {
        "eval_summary": str(summary_path),
        "output": str(output),
        "transitions": sorted(f"{a}->{b}" for a, b in transitions),
        "input_records": len(records),
        "records": len(rows),
        "unique_samples": len(unique_rows),
        "transition_counts": dict(sorted(transition_counts.items())),
        "test_summary_allowed": bool(args.allow_test),
        "target_match_audit": target_match_audit,
    }
    summary_path_out = output.with_suffix(output.suffix + ".summary.json")
    summary_path_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
