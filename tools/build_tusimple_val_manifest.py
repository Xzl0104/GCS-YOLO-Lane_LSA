from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    read_tusimple_json_lines,
    stable_gt_content_hash,
    stable_raw_file_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical TuSimple official-val manifest.")
    parser.add_argument("--gt-json", required=True, help="Canonical 363-image TuSimple official-val GT jsonl.")
    parser.add_argument(
        "--out",
        default=str(ROOT / "gcs_tools" / "canonical_tusimple_val_363_manifest.json"),
        help="Output manifest JSON path.",
    )
    parser.add_argument("--name", default="tusimple_official_val_363", help="Manifest name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_records = read_tusimple_json_lines(args.gt_json)
    manifest = {
        "name": args.name,
        "num_images": len(gt_records),
        "raw_file_sha256": stable_raw_file_hash(gt_records),
        "gt_content_sha256": stable_gt_content_hash(gt_records),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
