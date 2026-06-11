"""Split a GCS hard-sample manifest by GT lane count.

This is intended for loss-only hard-edge manifests that are dominated by one
lane-count class. It preserves the original manifest entries in the output files
and uses labels_gcs/<split>/*.npz num_lanes to decide the bucket.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input txt/json hard-sample manifest.")
    parser.add_argument(
        "--dataset-root",
        default="datasets/tusimple_fixed_y_960x544",
        help="Dataset root containing labels_gcs/<split>.",
    )
    parser.add_argument(
        "--target-splits",
        nargs="+",
        default=["train"],
        help="Splits used for fallback id/raw_file lookup.",
    )
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=[4, 5],
        help="GT lane counts to write, e.g. --counts 4 5.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to the input manifest directory.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Output file prefix. Defaults to the input manifest stem.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Skip entries whose labels cannot be resolved.")
    return parser.parse_args()


def collect_json_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(collect_json_strings(v))
        return out
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            out.extend(collect_json_strings(v))
        return out
    return []


def read_manifest(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        return [x.strip() for x in collect_json_strings(json.loads(path.read_text(encoding="utf-8"))) if x.strip()]
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def np_scalar_to_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="ignore")
    return str(item)


def normalize_id(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def add_index(index: dict[str, Path | None], key: str, path: Path) -> None:
    key = normalize_id(key)
    if not key:
        return
    if key in index and index[key] != path:
        index[key] = None
    else:
        index[key] = path


def build_label_index(dataset_root: Path, splits: list[str]) -> dict[str, Path | None]:
    index: dict[str, Path | None] = {}
    for split in splits:
        label_dir = dataset_root / "labels_gcs" / split
        if not label_dir.exists():
            continue
        for label in label_dir.glob("*.npz"):
            add_index(index, label.stem, label)
            add_index(index, label.as_posix(), label)
            with np.load(label) as data:
                raw_file = np_scalar_to_str(data["raw_file"]) if "raw_file" in data.files else ""
            if raw_file:
                add_index(index, raw_file, label)
    return index


def direct_label_path(entry: str, dataset_root: Path) -> Path | None:
    norm = normalize_id(entry)
    path = Path(entry)
    if path.suffix.lower() == ".npz" and path.exists():
        return path

    for marker in ("/labels_gcs/", "labels_gcs/"):
        if marker in norm:
            tail = norm.split(marker, 1)[1]
            parts = tail.split("/", 1)
            if len(parts) == 2:
                return dataset_root / "labels_gcs" / parts[0] / Path(parts[1]).with_suffix(".npz").name

    for marker in ("/images/", "images/"):
        if marker in norm:
            tail = norm.split(marker, 1)[1]
            parts = tail.split("/", 1)
            if len(parts) == 2:
                return dataset_root / "labels_gcs" / parts[0] / f"{Path(parts[1]).stem}.npz"
    return None


def resolve_label(entry: str, dataset_root: Path, index: dict[str, Path | None]) -> Path | None:
    direct = direct_label_path(entry, dataset_root)
    if direct is not None and direct.exists():
        return direct

    norm = normalize_id(entry)
    candidates = [norm, Path(norm).stem]
    for key in candidates:
        value = index.get(normalize_id(key))
        if value is not None:
            return value
    return None


def label_lane_count(path: Path) -> int:
    with np.load(path) as data:
        if "num_lanes" in data.files:
            return int(np.asarray(data["num_lanes"]).reshape(-1)[0])
        return int(np.asarray(data["lanes"]).shape[0])


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else manifest.parent
    prefix = args.prefix or manifest.stem
    wanted_counts = {int(x) for x in args.counts}
    if any(x <= 0 for x in wanted_counts):
        raise ValueError(f"--counts must be positive integers, got {sorted(wanted_counts)}.")

    entries = read_manifest(manifest)
    index = build_label_index(dataset_root, list(args.target_splits))
    buckets: dict[int, list[str]] = {count: [] for count in sorted(wanted_counts)}
    all_counts: Counter[int] = Counter()
    missing: list[str] = []

    for entry in entries:
        label = resolve_label(entry, dataset_root, index)
        if label is None or not label.exists():
            missing.append(entry)
            continue
        count = label_lane_count(label)
        all_counts[count] += 1
        if count in buckets:
            buckets[count].append(entry)

    if missing and not args.allow_missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} manifest entries could not be resolved to labels_gcs. First: {preview}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for count, values in buckets.items():
        out_path = output_dir / f"{prefix}_gt{count}.txt"
        out_path.write_text("".join(f"{x}\n" for x in values), encoding="utf-8")
        outputs[str(count)] = out_path.as_posix()

    summary = {
        "manifest": manifest.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "target_splits": list(args.target_splits),
        "total_entries": len(entries),
        "resolved_entries": sum(all_counts.values()),
        "missing_entries": len(missing),
        "lane_count_hist": {str(k): v for k, v in sorted(all_counts.items())},
        "outputs": outputs,
        "output_counts": {str(k): len(v) for k, v in sorted(buckets.items())},
    }
    summary_path = output_dir / f"{prefix}_by_count_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
