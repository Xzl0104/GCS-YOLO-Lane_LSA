from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    default_tusimple_gt_json,
    find_tusimple_archive_root,
    normalize_tusimple_gt_record,
    read_tusimple_json_lines,
)

DEFAULT_DATASET_ROOT = ROOT / "datasets" / "tusimple_fixed_y_960x544"
DEFAULT_OUTPUT = ROOT / "runs" / "gcs_lane" / "tusimple_official_trainval_500_test_ratio_seed20260529.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small TuSimple official-GT json subset from current train/val split by test lane-count ratio."
    )
    parser.add_argument("--archive-root", default=str(ROOT / "archive" / "TUSimple"), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Converted GCS dataset root with labels_gcs splits.")
    parser.add_argument("--pool-json", default=None, help="Official train/val json-lines pool. Defaults to TuSimple train_val json.")
    parser.add_argument("--reference-json", default=None, help="Json-lines used for target lane-count ratio. Defaults to official test labels.")
    parser.add_argument("--source-splits", nargs="+", default=["val", "train"], choices=("train", "val"), help="Converted splits used as the sampling pool.")
    parser.add_argument("--num-images", type=int, default=500, help="Number of records to sample.")
    parser.add_argument("--seed", type=int, default=20260529, help="Deterministic sampling seed.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output TuSimple official json-lines path.")
    parser.add_argument("--summary", default=None, help="Optional summary json path. Defaults to <output>.summary.json.")
    return parser.parse_args()


def lane_count(record: dict) -> int:
    """Count TuSimple lanes with at least one valid x coordinate."""
    return sum(1 for lane in record.get("lanes", []) if any(float(x) >= 0.0 for x in lane))


def read_split_raw_files(dataset_root: Path, splits: list[str]) -> dict[str, dict]:
    """Map official raw_file ids to the current converted split membership."""
    out: dict[str, dict] = {}
    for split in splits:
        label_dir = dataset_root / "labels_gcs" / split
        if not label_dir.exists():
            raise FileNotFoundError(f"labels_gcs split not found: {label_dir}")
        for label_path in sorted(label_dir.glob("*.npz")):
            with np.load(label_path, allow_pickle=True) as data:
                if "raw_file" not in data:
                    raise KeyError(f"{label_path} is missing raw_file")
                raw_file = str(np.asarray(data["raw_file"]).reshape(-1)[0]).replace("\\", "/").lstrip("/")
                num_lanes = int(np.asarray(data["num_lanes"]).reshape(-1)[0]) if "num_lanes" in data else None
            if raw_file in out:
                prev = out[raw_file]
                raise ValueError(f"raw_file appears in multiple requested splits: {raw_file} ({prev['split']} and {split})")
            out[raw_file] = {"split": split, "label": str(label_path), "num_lanes": num_lanes}
    return out


def proportional_targets(reference_hist: Counter, total: int, available_hist: Counter) -> dict[int, int]:
    """Allocate total samples by largest-remainder apportionment, capped by availability."""
    if total <= 0:
        raise ValueError("--num-images must be positive")
    ref_total = sum(reference_hist.values())
    if ref_total <= 0:
        raise ValueError("reference histogram is empty")

    quotas = {k: (total * v / ref_total) for k, v in sorted(reference_hist.items())}
    targets = {k: int(quotas[k]) for k in quotas}
    for k, _frac in sorted(((k, quotas[k] - targets[k]) for k in quotas), key=lambda x: (-x[1], x[0])):
        if sum(targets.values()) >= total:
            break
        targets[k] += 1

    capped = {k: min(targets.get(k, 0), available_hist.get(k, 0)) for k in sorted(set(targets) | set(available_hist))}
    deficit = total - sum(capped.values())
    if deficit <= 0:
        return {k: v for k, v in capped.items() if v > 0}

    priority = sorted(available_hist, key=lambda k: (-reference_hist.get(k, 0), k))
    while deficit > 0:
        progressed = False
        for k in priority:
            if capped.get(k, 0) < available_hist[k]:
                capped[k] = capped.get(k, 0) + 1
                deficit -= 1
                progressed = True
                if deficit == 0:
                    break
        if not progressed:
            raise ValueError(f"Only {sum(available_hist.values())} records are available, cannot sample {total}.")
    return {k: v for k, v in capped.items() if v > 0}


def main() -> None:
    args = parse_args()
    archive_root = find_tusimple_archive_root(args.archive_root)
    dataset_root = Path(args.dataset_root)
    pool_json = Path(args.pool_json) if args.pool_json else default_tusimple_gt_json(archive_root, split="train")
    reference_json = Path(args.reference_json) if args.reference_json else default_tusimple_gt_json(archive_root, split="test")
    output = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")

    split_meta = read_split_raw_files(dataset_root, list(args.source_splits))
    reference_records = read_tusimple_json_lines(reference_json)
    reference_hist = Counter(lane_count(r) for r in reference_records)

    pool_records = read_tusimple_json_lines(pool_json)
    groups: dict[int, list[dict]] = defaultdict(list)
    skipped = 0
    for record in pool_records:
        raw_file = str(record["raw_file"]).replace("\\", "/").lstrip("/")
        meta = split_meta.get(raw_file)
        if meta is None:
            skipped += 1
            continue
        enriched = dict(record)
        enriched["_gcs_source_split"] = meta["split"]
        enriched["_gcs_label"] = meta["label"]
        enriched["_gcs_num_lanes"] = meta["num_lanes"]
        groups[lane_count(enriched)].append(enriched)

    available_hist = Counter({k: len(v) for k, v in groups.items()})
    targets = proportional_targets(reference_hist, int(args.num_images), available_hist)

    rng = random.Random(int(args.seed))
    selected: list[dict] = []
    for count, need in sorted(targets.items()):
        candidates = sorted(groups[count], key=lambda r: str(r["raw_file"]))
        rng.shuffle(candidates)
        selected.extend(candidates[:need])
    rng.shuffle(selected)

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in selected:
            clean = {k: v for k, v in record.items() if not k.startswith("_gcs_")}
            clean = normalize_tusimple_gt_record(clean)
            f.write(json.dumps(clean, separators=(",", ":")) + "\n")

    selected_hist = Counter(lane_count(r) for r in selected)
    source_split_hist = Counter(str(r["_gcs_source_split"]) for r in selected)
    summary = {
        "output": str(output.resolve()),
        "summary": str(summary_path.resolve()),
        "archive_root": str(archive_root.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "pool_json": str(pool_json.resolve()),
        "reference_json": str(reference_json.resolve()),
        "source_splits": list(args.source_splits),
        "seed": int(args.seed),
        "num_images": int(args.num_images),
        "reference_hist": {str(k): int(v) for k, v in sorted(reference_hist.items())},
        "available_hist": {str(k): int(v) for k, v in sorted(available_hist.items())},
        "target_hist": {str(k): int(v) for k, v in sorted(targets.items())},
        "selected_hist": {str(k): int(v) for k, v in sorted(selected_hist.items())},
        "source_split_hist": {str(k): int(v) for k, v in sorted(source_split_hist.items())},
        "skipped_pool_records_not_in_requested_splits": int(skipped),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
