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


DEFAULT_OUTPUT = ROOT / "runs" / "gcs_lane" / "tusimple_official_val_363_folder_aware_seed20260602.json"
DEFAULT_DATASET_ROOT = ROOT / "datasets" / "tusimple_fixed_y_960x544"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TuSimple official-GT json subset with lane-count targets and "
            "near-uniform sampling across raw_file folders inside each lane count."
        )
    )
    parser.add_argument(
        "--archive-root",
        default=str(ROOT / "archive" / "TUSimple"),
        help="Path to archive/ or archive/TUSimple.",
    )
    parser.add_argument(
        "--pool-json",
        default=None,
        help="Official json-lines sampling pool. Defaults to train_set/seg_label/train_val.json.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Converted GCS dataset root used to filter the pool by current split membership.",
    )
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=["val"],
        choices=("train", "val"),
        help=(
            "Converted split(s) allowed in the sampling pool. Defaults to val so official-val "
            "does not overlap the current train split. Pass train val only for diagnostic train+val subsets."
        ),
    )
    parser.add_argument(
        "--reference-json",
        default=None,
        help="Json-lines used for target lane-count ratio when --target-hist is not set. Defaults to test labels.",
    )
    parser.add_argument("--num-images", type=int, default=363, help="Number of records to sample.")
    parser.add_argument(
        "--target-hist",
        default="auto",
        help=(
            "Optional exact lane-count targets, e.g. '3:227,4:62,5:74'. "
            "Use 'auto' to derive targets from --reference-json and --num-images."
        ),
    )
    parser.add_argument(
        "--folder-index",
        type=int,
        default=1,
        help="raw_file path component used as the balancing folder. For clips/0601/x/20.jpg, index 1 gives 0601.",
    )
    parser.add_argument("--seed", type=int, default=20260602, help="Deterministic sampling seed.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output TuSimple official json-lines path.")
    parser.add_argument("--summary", default=None, help="Optional summary json path. Defaults to <output>.summary.json.")
    return parser.parse_args()


def lane_count(record: dict) -> int:
    """Count valid TuSimple lanes after dropping all-negative placeholders."""
    return len(normalize_tusimple_gt_record(record).get("lanes", []))


def read_split_raw_files(dataset_root: Path, splits: list[str]) -> dict[str, dict]:
    """Map official raw_file ids to current converted split membership."""
    out: dict[str, dict] = {}
    for split in splits:
        label_dir = dataset_root / "labels_gcs" / split
        if not label_dir.exists():
            raise FileNotFoundError(f"labels_gcs split not found: {label_dir}")
        for label_path in sorted(label_dir.glob("*.npz")):
            with np.load(label_path, allow_pickle=True) as data:
                if "raw_file" not in data.files:
                    raise KeyError(f"{label_path} is missing raw_file")
                raw_file = str(np.asarray(data["raw_file"]).reshape(-1)[0]).replace("\\", "/").lstrip("/")
                num_lanes = int(np.asarray(data["num_lanes"]).reshape(-1)[0]) if "num_lanes" in data.files else None
            if raw_file in out:
                prev = out[raw_file]
                raise ValueError(f"raw_file appears in multiple requested splits: {raw_file} ({prev['split']} and {split})")
            out[raw_file] = {"split": split, "label": display_path(label_path), "num_lanes": num_lanes}
    return out


def source_folder(raw_file: str, folder_index: int) -> str:
    """Return the path component used for folder-balanced sampling."""
    parts = str(raw_file).replace("\\", "/").strip("/").split("/")
    if not parts:
        return ""
    if folder_index < 0:
        folder_index = len(parts) + folder_index
    if folder_index < 0 or folder_index >= len(parts):
        raise ValueError(f"raw_file={raw_file!r} has no component at folder_index={folder_index}.")
    return parts[folder_index]


def parse_target_hist(value: str, reference_hist: Counter[int], total: int, available_hist: Counter[int]) -> dict[int, int]:
    """Parse exact lane-count targets or derive them from a reference histogram."""
    text = str(value or "").strip()
    if not text or text.lower() == "auto":
        return proportional_targets(reference_hist, total, available_hist)

    out: dict[int, int] = {}
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            key, raw_count = chunk.split(":", 1)
        elif "=" in chunk:
            key, raw_count = chunk.split("=", 1)
        else:
            raise ValueError(f"Invalid --target-hist item {chunk!r}; expected count:target.")
        lane_count_i = int(key.strip())
        target_i = int(raw_count.strip())
        if target_i < 0:
            raise ValueError(f"Target for lane count {lane_count_i} must be >= 0, got {target_i}.")
        if target_i > 0:
            out[lane_count_i] = target_i

    if not out:
        raise ValueError("--target-hist did not contain any positive targets.")
    if sum(out.values()) != int(total):
        raise ValueError(
            f"--target-hist sums to {sum(out.values())}, but --num-images is {int(total)}. "
            "Keep them equal so summaries and downstream sweep configs are unambiguous."
        )
    return dict(sorted(out.items()))


def proportional_targets(reference_hist: Counter[int], total: int, available_hist: Counter[int]) -> dict[int, int]:
    """Allocate total samples by largest-remainder apportionment, capped by availability."""
    if total <= 0:
        raise ValueError("--num-images must be positive")
    ref_total = sum(reference_hist.values())
    if ref_total <= 0:
        raise ValueError("reference histogram is empty")

    quotas = {k: (total * v / ref_total) for k, v in sorted(reference_hist.items()) if available_hist.get(k, 0) > 0}
    targets = {k: int(quotas[k]) for k in quotas}
    for k, _frac in sorted(((k, quotas[k] - targets[k]) for k in quotas), key=lambda x: (-x[1], x[0])):
        if sum(targets.values()) >= total:
            break
        targets[k] += 1

    capped = {k: min(targets.get(k, 0), available_hist.get(k, 0)) for k in sorted(targets)}
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


def even_folder_targets(total: int, available: dict[str, int]) -> dict[str, int]:
    """Allocate total records as evenly as possible across folders, capped by availability."""
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    folders = sorted(k for k, v in available.items() if v > 0)
    if total == 0:
        return {}
    if not folders:
        raise ValueError("No folders are available for a positive target.")
    if sum(available[k] for k in folders) < total:
        raise ValueError(f"Only {sum(available[k] for k in folders)} records are available, cannot sample {total}.")

    remaining = int(total)
    active = list(folders)
    targets = {k: 0 for k in folders}
    while remaining > 0 and active:
        quota, extra = divmod(remaining, len(active))
        progressed = False
        next_active = []
        for i, folder in enumerate(active):
            want = quota + (1 if i < extra else 0)
            room = available[folder] - targets[folder]
            take = min(want, room)
            if take > 0:
                targets[folder] += take
                remaining -= take
                progressed = True
            if targets[folder] < available[folder]:
                next_active.append(folder)
        active = next_active
        if not progressed:
            break
    if remaining != 0:
        raise ValueError(f"Could not allocate {total} samples across folders; remaining={remaining}.")
    return {k: v for k, v in targets.items() if v > 0}


def display_path(path: Path) -> str:
    """Write project-relative paths in summaries when possible."""
    path = Path(path)
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def nested_counter_to_dict(counter: Counter[tuple[int, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for (count, folder), value in sorted(counter.items()):
        out[str(count)][str(folder)] = int(value)
    return dict(out)


def main() -> None:
    args = parse_args()
    archive_root = find_tusimple_archive_root(args.archive_root)
    pool_json = Path(args.pool_json) if args.pool_json else default_tusimple_gt_json(archive_root, split="train")
    reference_json = Path(args.reference_json) if args.reference_json else default_tusimple_gt_json(archive_root, split="test")
    dataset_root = Path(args.dataset_root)
    output = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")

    split_meta = read_split_raw_files(dataset_root, list(args.source_splits))
    pool_records = [normalize_tusimple_gt_record(r) for r in read_tusimple_json_lines(pool_json)]
    reference_records = [normalize_tusimple_gt_record(r) for r in read_tusimple_json_lines(reference_json)]

    groups: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    skipped_pool_records_not_in_requested_splits = 0
    for record in pool_records:
        raw_file = str(record["raw_file"]).replace("\\", "/").lstrip("/")
        meta = split_meta.get(raw_file)
        if meta is None:
            skipped_pool_records_not_in_requested_splits += 1
            continue
        count = lane_count(record)
        folder = source_folder(raw_file, int(args.folder_index))
        enriched = dict(record)
        enriched["_folder"] = folder
        enriched["_gcs_source_split"] = meta["split"]
        enriched["_gcs_label"] = meta["label"]
        enriched["_gcs_num_lanes"] = meta["num_lanes"]
        groups[count][folder].append(enriched)

    available_hist = Counter({count: sum(len(v) for v in by_folder.values()) for count, by_folder in groups.items()})
    reference_hist = Counter(lane_count(r) for r in reference_records)
    targets = parse_target_hist(str(args.target_hist), reference_hist, int(args.num_images), available_hist)

    rng = random.Random(int(args.seed))
    selected: list[dict] = []
    target_lane_folder_hist: Counter[tuple[int, str]] = Counter()
    available_lane_folder_hist: Counter[tuple[int, str]] = Counter()
    for count, by_folder in sorted(groups.items()):
        for folder, candidates in sorted(by_folder.items()):
            available_lane_folder_hist[(count, folder)] = len(candidates)

    for count, need in sorted(targets.items()):
        by_folder = groups.get(count, {})
        available_by_folder = {folder: len(candidates) for folder, candidates in by_folder.items()}
        folder_targets = even_folder_targets(int(need), available_by_folder)
        for folder, folder_need in sorted(folder_targets.items()):
            candidates = sorted(by_folder[folder], key=lambda r: str(r["raw_file"]))
            rng.shuffle(candidates)
            selected.extend(candidates[:folder_need])
            target_lane_folder_hist[(count, folder)] = int(folder_need)

    rng.shuffle(selected)

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for record in selected:
            clean = {k: v for k, v in record.items() if not k.startswith("_")}
            f.write(json.dumps(normalize_tusimple_gt_record(clean), separators=(",", ":")) + "\n")

    selected_hist = Counter(lane_count(r) for r in selected)
    selected_folder_hist = Counter(str(r["_folder"]) for r in selected)
    selected_lane_folder_hist = Counter((lane_count(r), str(r["_folder"])) for r in selected)
    selected_source_split_hist = Counter(str(r["_gcs_source_split"]) for r in selected)
    folder_notes = {}
    for count, by_folder in sorted(groups.items()):
        if count in targets:
            folders = sorted(k for k, v in by_folder.items() if len(v) > 0)
            folder_notes[str(count)] = {
                "available_folders": folders,
                "balanced_across_folders": len(folders) > 1,
            }

    summary = {
        "output": display_path(output),
        "summary": display_path(summary_path),
        "archive_root": display_path(archive_root),
        "dataset_root": display_path(dataset_root),
        "pool_json": display_path(pool_json),
        "reference_json": display_path(reference_json),
        "source_splits": list(args.source_splits),
        "seed": int(args.seed),
        "num_images": len(selected),
        "requested_num_images": int(args.num_images),
        "target_hist": {str(k): int(v) for k, v in sorted(targets.items())},
        "selected_hist": {str(k): int(v) for k, v in sorted(selected_hist.items())},
        "available_hist": {str(k): int(v) for k, v in sorted(available_hist.items())},
        "reference_hist": {str(k): int(v) for k, v in sorted(reference_hist.items())},
        "folder_index": int(args.folder_index),
        "selected_folder_hist": {str(k): int(v) for k, v in sorted(selected_folder_hist.items())},
        "selected_source_split_hist": {str(k): int(v) for k, v in sorted(selected_source_split_hist.items())},
        "target_lane_folder_hist": nested_counter_to_dict(target_lane_folder_hist),
        "selected_lane_folder_hist": nested_counter_to_dict(selected_lane_folder_hist),
        "available_lane_folder_hist": nested_counter_to_dict(available_lane_folder_hist),
        "folder_balance_notes": folder_notes,
        "split_filter": {
            "requested_splits": list(args.source_splits),
            "raw_files_in_requested_splits": len(split_meta),
            "skipped_pool_records_not_in_requested_splits": int(skipped_pool_records_not_in_requested_splits),
            "train_overlap_possible": "train" in set(args.source_splits),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
