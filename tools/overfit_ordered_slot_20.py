from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str, trainer_imgsz  # noqa: E402


DEFAULT_MODEL = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q5-slot-k56.yaml"
OVERFIT_THRESHOLDS = {
    "slot_count_acc": {"op": ">=", "value": 1.0},
    "slot_exist_acc": {"op": ">=", "value": 1.0},
    "interval_start_mae": {"op": "<=", "value": 0.5},
    "interval_end_mae": {"op": "<=", "value": 0.5},
    "order_violation_rate": {"op": "<=", "value": 0.01},
    "point_l1_px": {"op": "<=", "value": 3.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 20-image overfit check for GCS ordered_slot mode.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data", default=None)
    parser.add_argument("--train-images", default=None)
    parser.add_argument("--train-gcs-labels", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS shape as H W. Defaults to dataset shape.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--pretrained", default="yolo11s-seg.pt")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--lrf", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--project", default=str((ROOT / "runs/gcs_lane").resolve()))
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--gcs-num-slots", type=int, default=5)
    parser.add_argument("--gcs-min-lanes", type=int, default=2)
    parser.add_argument("--gcs-max-lanes", type=int, default=5)
    parser.add_argument("--gcs-count-classes", type=int, default=4)
    parser.add_argument("--gcs-contiguity-policy", choices=("strict", "repair_interp"), default="strict")
    parser.add_argument("--gcs-exist", type=float, default=2.0)
    parser.add_argument("--gcs-point", type=float, default=15.0)
    parser.add_argument("--gcs-point-valid", type=float, default=1.0)
    parser.add_argument("--gcs-count-ce", type=float, default=1.0)
    parser.add_argument("--gcs-interval", type=float, default=1.0)
    parser.add_argument("--gcs-order", type=float, default=1.0)
    parser.add_argument("--gcs-slot-exist-w4", type=float, default=2.0)
    parser.add_argument("--gcs-slot-exist-w5", type=float, default=2.0)
    parser.add_argument("--gcs-order-margin-px", type=float, default=5.0)
    parser.add_argument("--target-count-acc", type=float, default=1.0)
    parser.add_argument("--target-exist-acc", type=float, default=1.0)
    parser.add_argument("--max-interval-mae", type=float, default=0.5)
    parser.add_argument("--max-order-violation-rate", type=float, default=0.01)
    parser.add_argument("--max-point-l1-px", type=float, default=3.0)
    return parser.parse_args()


def _read_last_results(save_dir: Path) -> dict[str, float]:
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    out = {}
    for key, value in rows[-1].items():
        if value in {None, ""}:
            continue
        try:
            out[key.strip()] = float(value)
        except ValueError:
            continue
    return out


def _metric(row: dict[str, float], name: str) -> float | None:
    return row.get(f"val/{name}") if f"val/{name}" in row else row.get(f"train/{name}")


def overfit_thresholds_from_args(args: argparse.Namespace) -> dict[str, dict[str, float | str]]:
    """Build ordered-slot overfit thresholds from CLI args."""
    thresholds = {key: dict(rule) for key, rule in OVERFIT_THRESHOLDS.items()}
    thresholds["slot_count_acc"]["value"] = float(args.target_count_acc)
    thresholds["slot_exist_acc"]["value"] = float(args.target_exist_acc)
    thresholds["interval_start_mae"]["value"] = float(args.max_interval_mae)
    thresholds["interval_end_mae"]["value"] = float(args.max_interval_mae)
    thresholds["order_violation_rate"]["value"] = float(args.max_order_violation_rate)
    thresholds["point_l1_px"]["value"] = float(args.max_point_l1_px)
    return thresholds


def overfit_failures(metrics: dict[str, float | None], thresholds: dict[str, dict[str, float | str]]) -> list[str]:
    """Return human-readable ordered-slot overfit threshold failures."""
    failures = []
    for key, rule in thresholds.items():
        value = metrics.get(key)
        if value is None:
            failures.append(f"{key}: missing")
            continue
        if key.startswith("slot_count_acc_"):
            count = key.rsplit("_", 1)[-1]
            total = metrics.get(f"slot_count_total_{count}")
            if total is not None and float(total) == 0.0 and not math.isfinite(float(value)):
                continue

        op = str(rule["op"])
        target = float(rule["value"])
        got = float(value)
        if not math.isfinite(got):
            failures.append(f"{key}: got {got}, expected {op} {target}")
            continue
        if op == ">=":
            passed = got >= target
        elif op == "<=":
            passed = got <= target
        elif op == "==":
            passed = got == target
        else:
            failures.append(f"{key}: unsupported op {op}")
            continue
        if not passed:
            failures.append(f"{key}: got {got}, expected {op} {target}")
    return failures


def assert_overfit_passed(metrics: dict[str, float | None], thresholds: dict[str, dict[str, float | str]]) -> None:
    """Raise when the ordered-slot overfit metrics do not satisfy the threshold contract."""
    failures = overfit_failures(metrics, thresholds)
    if failures:
        raise AssertionError("ordered_slot overfit20 failed:\n" + "\n".join(failures))


def summarize_ordered_overfit(save_dir: Path, args: argparse.Namespace) -> tuple[Path, dict]:
    row = _read_last_results(save_dir)
    metrics = {
        "slot_count_acc": _metric(row, "slot_count_acc"),
        "slot_exist_acc": _metric(row, "slot_exist_acc"),
        "interval_start_mae": _metric(row, "interval_start_mae"),
        "interval_end_mae": _metric(row, "interval_end_mae"),
        "order_violation_rate": _metric(row, "order_violation_rate"),
        "point_l1_px": _metric(row, "point_l1_px"),
        "slot_count_acc_2": _metric(row, "slot_count_acc_2"),
        "slot_count_acc_3": _metric(row, "slot_count_acc_3"),
        "slot_count_acc_4": _metric(row, "slot_count_acc_4"),
        "slot_count_acc_5": _metric(row, "slot_count_acc_5"),
        "slot_count_correct_2": _metric(row, "slot_count_correct_2"),
        "slot_count_correct_3": _metric(row, "slot_count_correct_3"),
        "slot_count_correct_4": _metric(row, "slot_count_correct_4"),
        "slot_count_correct_5": _metric(row, "slot_count_correct_5"),
        "slot_count_total_2": _metric(row, "slot_count_total_2"),
        "slot_count_total_3": _metric(row, "slot_count_total_3"),
        "slot_count_total_4": _metric(row, "slot_count_total_4"),
        "slot_count_total_5": _metric(row, "slot_count_total_5"),
        "slot4_exist_acc": _metric(row, "slot4_exist_acc"),
        "slot5_exist_acc": _metric(row, "slot5_exist_acc"),
    }
    thresholds = overfit_thresholds_from_args(args)
    summary = {
        "passed": not overfit_failures(metrics, thresholds),
        "thresholds": thresholds,
        "last_metrics": metrics,
    }
    out = save_dir / "ordered_slot_overfit_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out, summary


def main() -> None:
    from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer
    from tools.overfit_gcs_20 import (
        check_subset_contract,
        collect_overfit_pairs,
        dataset_defaults,
        parse_pretrained,
        write_subset_files,
    )

    args = parse_args()
    gcs_imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    defaults = dataset_defaults(args.dataset)
    data = args.data or str(defaults["data"])
    train_images = args.train_images or str(defaults["train_images"])
    train_gcs_labels = args.train_gcs_labels or str(defaults["train_labels"])
    name = args.name or f"overfit20_ordered_slot_{args.dataset}"
    project = Path(args.project)

    pairs = collect_overfit_pairs(train_images, train_gcs_labels, args.limit)
    image_list, manifest_path = write_subset_files(pairs, project=project, name=name)
    check_subset_contract(image_list, train_gcs_labels, gcs_imgsz)

    overrides = {
        "task": "gcs_lane",
        "model": args.model,
        "data": data,
        "pretrained": parse_pretrained(args.pretrained),
        "imgsz": trainer_imgsz(gcs_imgsz),
        "gcs_imgsz": list(gcs_imgsz),
        "epochs": args.epochs,
        "batch": args.batch,
        "nbs": args.nbs if args.nbs > 0 else args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": str(project.resolve()),
        "name": name,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "patience": max(args.epochs, 100),
        "fraction": 1.0,
        "mosaic": 0.0,
        "scale": 0.0,
        "erasing": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "fliplr": 0.0,
        "val": True,
        "exist_ok": args.exist_ok,
        "train_images": str(image_list.resolve()),
        "train_gcs_labels": train_gcs_labels,
        "val_images": str(image_list.resolve()),
        "val_gcs_labels": train_gcs_labels,
        "gcs_mode": "ordered_slot",
        "gcs_num_slots": args.gcs_num_slots,
        "gcs_min_lanes": args.gcs_min_lanes,
        "gcs_max_lanes": args.gcs_max_lanes,
        "gcs_count_classes": args.gcs_count_classes,
        "gcs_contiguity_policy": args.gcs_contiguity_policy,
        "gcs_exist": args.gcs_exist,
        "gcs_point": args.gcs_point,
        "gcs_point_valid": args.gcs_point_valid,
        "gcs_count_ce": args.gcs_count_ce,
        "gcs_interval": args.gcs_interval,
        "gcs_order": args.gcs_order,
        "gcs_slot_exist_w4": args.gcs_slot_exist_w4,
        "gcs_slot_exist_w5": args.gcs_slot_exist_w5,
        "gcs_order_margin_px": args.gcs_order_margin_px,
        "gcs_eval_nms_dist_px": 0.0,
        "gcs_eval_point_valid_thr": 0.5,
        "gcs_eval_max_det": 5,
    }

    print(f"GCS input shape: {shape_str(gcs_imgsz)} (W x H), stored as H,W={gcs_imgsz}")
    print(f"ordered_slot overfit subset: {len(pairs)} images")
    print(f"image list: {image_list.resolve()}")
    print(f"manifest: {manifest_path.resolve()}")

    trainer = GCSLaneTrainer(overrides=overrides)
    trainer.train()

    save_dir = Path(trainer.save_dir)
    summary_path, summary = summarize_ordered_overfit(save_dir, args)
    print(f"training run: {save_dir.resolve()}")
    print(f"ordered_slot overfit summary: {summary_path.resolve()}")
    assert_overfit_passed(summary["last_metrics"], summary["thresholds"])


if __name__ == "__main__":
    main()
