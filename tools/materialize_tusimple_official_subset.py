from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    find_tusimple_archive_root,
    normalize_tusimple_gt_record,
    read_tusimple_json_lines,
    tusimple_image_path,
)


DEFAULT_INPUT = ROOT / "runs" / "gcs_lane" / "tusimple_official_val_363.json"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "gcs_lane" / "tusimple_official_val_363_subset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a TuSimple official json-lines subset and its images into an isolated archive-like folder."
    )
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT), help="Official TuSimple json-lines subset.")
    parser.add_argument("--archive-root", default=str(ROOT / "archive" / "TUSimple"), help="Source archive/ or archive/TUSimple.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Destination subset folder.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Image lookup/output split.")
    parser.add_argument(
        "--label-name",
        default=None,
        help="Output label json filename under labels/. Defaults to the input json filename.",
    )
    parser.add_argument(
        "--per-image-labels",
        action="store_true",
        help="Also write one normalized json file per image under labels/per_image/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied images and label files in the output folder.",
    )
    return parser.parse_args()


def lane_count(record: dict) -> int:
    """Count valid TuSimple lanes after dropping all-negative placeholders."""
    record = normalize_tusimple_gt_record(record)
    return len(record.get("lanes", []))


def sanitize_raw_file(raw_file: str) -> str:
    """Return a filename-safe id that still preserves the original path information."""
    return raw_file.replace("\\", "/").strip("/").replace("/", "__")


def copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    """Copy one file and return True when bytes were written."""
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def display_path(path: Path) -> str:
    """Write portable project-relative paths in summaries when possible."""
    path = Path(path)
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    source_archive = find_tusimple_archive_root(args.archive_root)
    output_root = Path(args.output_root)
    label_name = args.label_name or input_json.name

    records = [normalize_tusimple_gt_record(r) for r in read_tusimple_json_lines(input_json)]
    image_split_dir = "test_set" if args.split == "test" else "train_set"
    output_label = output_root / "labels" / label_name
    output_summary = output_root / "summary.json"

    # Keep both split roots so the folder is accepted by find_tusimple_archive_root().
    (output_root / "train_set").mkdir(parents=True, exist_ok=True)
    (output_root / "test_set").mkdir(parents=True, exist_ok=True)
    (output_root / "labels").mkdir(parents=True, exist_ok=True)

    copied = 0
    reused = 0
    missing: list[dict] = []
    for record in records:
        raw_file = str(record["raw_file"]).replace("\\", "/").lstrip("/")
        try:
            src = tusimple_image_path(source_archive, raw_file, split=args.split)
        except FileNotFoundError as exc:
            missing.append({"raw_file": raw_file, "error": str(exc)})
            continue
        dst = output_root / image_split_dir / raw_file
        if copy_file(src, dst, overwrite=bool(args.overwrite)):
            copied += 1
        else:
            reused += 1

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} subset images. First missing: {missing[0]}")

    if output_label.exists() and not args.overwrite:
        raise FileExistsError(f"Output label exists: {output_label}. Pass --overwrite to replace it.")
    with output_label.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    per_image_written = 0
    if args.per_image_labels:
        per_image_dir = output_root / "labels" / "per_image"
        per_image_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            raw_file = str(record["raw_file"]).replace("\\", "/").lstrip("/")
            path = per_image_dir / f"{sanitize_raw_file(raw_file)}.json"
            if path.exists() and not args.overwrite:
                continue
            path.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
            per_image_written += 1

    hist = Counter(lane_count(r) for r in records)
    summary = {
        "input_json": display_path(input_json),
        "source_archive": display_path(source_archive),
        "output_root": display_path(output_root),
        "label_json": display_path(output_label),
        "split": str(args.split),
        "image_split_dir": image_split_dir,
        "images": len(records),
        "images_copied": int(copied),
        "images_reused": int(reused),
        "per_image_labels_written": int(per_image_written),
        "lane_count_hist": {str(k): int(v) for k, v in sorted(hist.items())},
        "usage": {
            "archive_root": display_path(output_root),
            "gt_json": display_path(output_label),
            "split": str(args.split),
        },
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
