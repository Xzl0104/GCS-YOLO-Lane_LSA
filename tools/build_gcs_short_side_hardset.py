from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.diagnose_gcs_short_side_hardset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    build_config,
    clean_float,
    diagnose_one_image,
    json_default,
    raw_match,
    summarize_subset,
    warmup_model,
    write_json,
    write_jsonl,
)
from tools.infer_gcs import collect_images, load_gcs_model  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed hard short-side raw-geometry GT lane set.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=("train", "val"))
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.003)
    parser.add_argument("--point-valid-thr", type=float, default=0.6)
    parser.add_argument("--nms-dist-px", type=float, default=18.0)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--valid-before-maxdet", action="store_true")
    parser.add_argument("--min-overlap", type=int, default=3)
    parser.add_argument("--hard-gt-counts", nargs="+", type=int, default=[4, 5])
    parser.add_argument("--hard-visible-max", type=int, default=20)
    parser.add_argument("--hard-raw-x-thr-px", type=float, default=20.0)
    parser.add_argument("--max-images", type=int, default=0, help="Limit images per split. 0 means all.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--save-dir", required=True)
    return parser.parse_args()


def is_hard_lane(record: dict, args: argparse.Namespace) -> bool:
    if int(record.get("gt_lanes", -1)) not in {int(x) for x in args.hard_gt_counts}:
        return False
    if not bool(record.get("side_lane", False)):
        return False
    if int(record.get("visible_count", 0)) > int(args.hard_visible_max):
        return False
    mean_x = record.get("best_raw_mean_x_px")
    return (
        mean_x is None
        or float(mean_x) > float(args.hard_raw_x_thr_px)
        or int(record.get("best_raw_overlap", 0)) < int(args.min_overlap)
    )


def build_hardset_summary(
    hard_records: list[dict],
    *,
    args: argparse.Namespace,
    imgsz: tuple[int, int],
    images_seen: int,
    candidate_lanes: int,
    runtime_sec: float,
) -> dict:
    summary = summarize_subset(hard_records, min_overlap=args.min_overlap, include_groups=True)
    summary.update(
        {
            "config": {
                **build_config(args, imgsz),
                "splits": list(args.splits),
                "hard_gt_counts": [int(x) for x in args.hard_gt_counts],
                "hard_visible_max": int(args.hard_visible_max),
                "hard_raw_x_thr_px": clean_float(args.hard_raw_x_thr_px),
            },
            "images": int(images_seen),
            "candidate_gt_lanes": int(candidate_lanes),
            "runtime_sec": round(float(runtime_sec), 3),
            "contains_test": False,
            "selection_eligible": True,
            "not_for_tuning": False,
        }
    )
    summary["baseline_raw_match_20px_count"] = sum(
        1 for record in hard_records if raw_match(record, thr_px=20.0, min_overlap=args.min_overlap)
    )
    return summary


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    t0 = time.time()
    imgsz = normalize_imgsz(args.imgsz, dataset="tusimple")
    dataset_root = Path(args.dataset_root)
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=imgsz)

    all_hard_records: list[dict] = []
    images_seen = 0
    candidate_lanes = 0
    warmed = False
    for split in args.splits:
        source = dataset_root / "images" / split
        labels_dir = dataset_root / "labels_gcs" / split
        images = collect_images(source, max_images=args.max_images)
        print(f"building hardset split={split} images={len(images)}")
        for img_i, image_path in enumerate(images, start=1):
            if not warmed:
                warmup_model(model, image_path, imgsz, device, args.half, args.warmup)
                warmed = True
            records = diagnose_one_image(
                model,
                image_path,
                labels_dir,
                split=split,
                dataset_root=dataset_root,
                imgsz=imgsz,
                device=device,
                half=args.half,
                args=args,
            )
            images_seen += 1
            candidate_lanes += len(records)
            all_hard_records.extend(record for record in records if is_hard_lane(record, args))
            if img_i % 500 == 0 or img_i == len(images):
                print(f"  {split}: {img_i}/{len(images)} hard_lanes={len(all_hard_records)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = build_hardset_summary(
        all_hard_records,
        args=args,
        imgsz=imgsz,
        images_seen=images_seen,
        candidate_lanes=candidate_lanes,
        runtime_sec=time.time() - t0,
    )
    write_jsonl(save_dir / "records.jsonl", all_hard_records)
    write_json(save_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=json_default, allow_nan=False))
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
