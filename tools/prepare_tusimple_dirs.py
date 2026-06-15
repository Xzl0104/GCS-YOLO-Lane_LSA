from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_utils import ensure_dataset_dirs, find_archive_root, load_test_samples, load_train_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create GCS-YOLO-Lane TuSimple project directories.")
    parser.add_argument("--archive-root", default="archive/TUSimple", help="TuSimple root or archive directory.")
    parser.add_argument("--output-root", default="datasets/tusimple_fixed_y_960x544", help="Converted dataset root.")
    parser.add_argument("--no-test", action="store_true", help="Do not create test output directories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_root = find_archive_root(ROOT / args.archive_root if not Path(args.archive_root).is_absolute() else args.archive_root)
    output_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    ensure_dataset_dirs(output_root, include_test=not args.no_test)

    train_count = len(load_train_samples(archive_root))
    test_count = len(load_test_samples(archive_root))
    print(f"archive_root: {archive_root}")
    print(f"output_root: {output_root}")
    print(f"labeled train samples: {train_count}")
    print(f"labeled test samples: {test_count}")
    print("created images/labels/labels_gcs directories")


if __name__ == "__main__":
    main()
