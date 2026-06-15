from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.data.dataset_gcs import GCSLaneDataset
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer
from tools.eval_gcs import match_lanes
from tools.infer_gcs import run_inference
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str, trainer_imgsz
from ultralytics.utils.gcs_postprocess import GCS_DEFAULT_MAX_DET


DEFAULT_MODEL = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q12.yaml"
DEFAULT_DATA = ROOT / "data" / "tusimple_gcs_fixed_y_960x544.yaml"
DEFAULT_TRAIN_IMAGES = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "images" / "train"
DEFAULT_TRAIN_LABELS = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "labels_gcs" / "train"
DEFAULT_LOSS_GAINS = {
    "exist_loss": 1.0,
    "point_loss": 5.0,
    "point_valid_loss": 0.5,
    "line_iou_loss": 0.3,
    "count_cls_loss": 0.3,
    "count_sum_loss": 0.02,
    "quality_loss": 0.3,
}


def dataset_defaults(dataset: str) -> dict[str, Path]:
    """Return conventional local paths for a converted GCS dataset."""
    name = dataset.lower()
    root = ROOT / "datasets" / ("tusimple_fixed_y_960x544" if name == "tusimple" else name)
    return {
        "data": ROOT / "data" / ("tusimple_gcs_fixed_y_960x544.yaml" if name == "tusimple" else f"{name}_gcs.yaml"),
        "train_images": root / "images" / "train",
        "train_labels": root / "labels_gcs" / "train",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the required 20-image GCS-YOLO-Lane overfit test.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="GCS-YOLO-Lane model yaml.")
    parser.add_argument("--data", default=None, help="Dataset yaml.")
    parser.add_argument("--train-images", default=None, help="Training image directory.")
    parser.add_argument("--train-gcs-labels", default=None, help="Training labels_gcs directory.")
    parser.add_argument("--limit", type=int, default=20, help="Number of image/npz pairs to overfit.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS input shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=0, help="Nominal batch size. 0 uses --batch for true overfit updates.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--pretrained", default="yolo11s-seg.pt", help="Matching YOLO/GCS checkpoint, or false.")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--lrf", type=float, default=1.0, help="Final LR ratio. Keep 1.0 for small overfit tests.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=0.0, help="Disable warmup by default for overfit tests.")
    parser.add_argument("--warmup-bias-lr", type=float, default=0.0, help="Disable bias warmup by default for overfit tests.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA algorithms. Off by default because GCS uses ops that warn on CUDA.",
    )
    parser.add_argument("--project", default=str((ROOT / "runs/gcs_lane").resolve()))
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="Mosaic probability for the overfit check. Default stays 0.0 so mosaic GT lanes do not exceed the query budget.",
    )
    parser.add_argument("--gcs-exist", type=float, default=1.0, help="Overfit gain for quality-aware lane existence loss.")
    parser.add_argument("--gcs-point", type=float, default=5.0, help="Overfit gain for structured point loss.")
    parser.add_argument("--gcs-point-valid", type=float, default=0.5, help="Overfit gain for per-point visibility loss.")
    parser.add_argument(
        "--gcs-point-invalid-x",
        type=float,
        default=0.05,
        help="Relative pseudo-x penalty inside point loss for matched invisible anchors weighted by point-valid probability.",
    )
    parser.add_argument("--gcs-line-iou", type=float, default=0.3, help="Overfit gain for whole-lane LineIoU loss.")
    parser.add_argument("--gcs-quality", type=float, default=0.3, help="Overfit gain for lane-level Quality Head loss.")
    parser.add_argument(
        "--gcs-quality-dist-thr-px",
        type=float,
        default=20.0,
        help="Pixel inlier threshold used to build Quality Head point-inlier targets.",
    )
    parser.add_argument(
        "--gcs-quality-neg-weight",
        type=float,
        default=0.25,
        help="Unmatched-query BCE weight for Quality Head loss.",
    )
    parser.add_argument("--gcs-quality-hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--gcs-quality-duplicate-negative-weight", type=float, default=1.5)
    parser.add_argument(
        "--gcs-line-iou-width-px",
        type=float,
        default=15.0,
        help="Half-width in pixels used to expand lane points into horizontal strips for LineIoU.",
    )
    parser.add_argument("--gcs-count-cls", type=float, default=0.3)
    parser.add_argument("--gcs-count-head-warmup-epochs", type=float, default=5.0)
    parser.add_argument("--gcs-count-min-gt-points", type=int, default=1)
    parser.add_argument("--gcs-count-cls-w2", type=float, default=0.5)
    parser.add_argument("--gcs-count-cls-w3", type=float, default=1.2)
    parser.add_argument("--gcs-count-cls-w4", type=float, default=1.4)
    parser.add_argument("--gcs-count-cls-w5", type=float, default=2.0)
    parser.add_argument("--gcs-exist-pos-weight", type=float, default=1.0, help="Positive query weight for existence BCE.")
    parser.add_argument("--gcs-exist-focal-gamma", type=float, default=2.0, help="Quality focal gamma for existence BCE.")
    parser.add_argument(
        "--gcs-exist-focal-alpha",
        type=float,
        default=-1.0,
        help="Optional focal alpha for existence BCE. Use a value in [0, 1] to enable alpha weighting.",
    )
    parser.add_argument("--gcs-hard-negative-quality-thr", type=float, default=0.5)
    parser.add_argument("--gcs-hard-negative-topk", type=int, default=2)
    parser.add_argument("--gcs-hard-negative-exist-weight", type=float, default=4.0)
    parser.add_argument("--gcs-duplicate-negative-exist-weight", type=float, default=4.0)
    parser.add_argument("--gcs-duplicate-dist-thr-px", type=float, default=25.0)
    parser.add_argument("--gcs-duplicate-iou-thr", type=float, default=0.30)
    parser.add_argument("--gcs-exist-margin", type=float, default=0.5, help="Relative exist probability margin loss gain.")
    parser.add_argument("--gcs-exist-pos-margin", type=float, default=0.55)
    parser.add_argument("--gcs-exist-neg-margin", type=float, default=0.20)
    parser.add_argument(
        "--gcs-exist-quality-alpha",
        type=float,
        default=1.0,
        help="Blend factor for quality-aware existence targets. Keep 1.0 so poor-geometry matches are not high-confidence lanes.",
    )
    parser.add_argument(
        "--gcs-exist-quality-lane-iou-alpha",
        type=float,
        default=1.0,
        help="Blend factor inside existence geometry quality. 1 uses LineIoU quality, 0 uses APE quality.",
    )
    parser.add_argument(
        "--gcs-exist-quality-mode",
        choices=("linear", "exp"),
        default="linear",
        help="Quality target shape. linear makes APE >= neg-px a zero target; exp uses exponential APE decay.",
    )
    parser.add_argument(
        "--gcs-exist-quality-tau",
        type=float,
        default=25.0,
        help="APE decay scale in pixels for exp quality mode.",
    )
    parser.add_argument(
        "--gcs-exist-quality-floor",
        type=float,
        default=0.0,
        help="Minimum geometry quality used for exp quality mode.",
    )
    parser.add_argument(
        "--gcs-exist-quality-pos-px",
        type=float,
        default=10.0,
        help="APE at or below this value receives quality 1.0 in linear quality mode.",
    )
    parser.add_argument(
        "--gcs-exist-quality-neg-px",
        type=float,
        default=25.0,
        help="APE at or above this value receives quality 0.0 in linear quality mode.",
    )
    parser.add_argument(
        "--gcs-point-valid-pos-weight-max",
        type=float,
        default=10.0,
        help="Maximum positive weight for per-point visibility loss.",
    )
    parser.add_argument("--gcs-point-valid-gt5-pos-weight", type=float, default=3.0)
    parser.add_argument("--gcs-point-valid-unmatched-weight", type=float, default=0.35)
    parser.add_argument("--gcs-point-valid-hard-negative-weight", type=float, default=1.25)
    parser.add_argument("--gcs-point-valid-duplicate-negative-weight", type=float, default=1.5)
    parser.add_argument("--gcs-point-valid-neg", type=float, default=0.25)
    parser.add_argument("--gcs-point-valid-neg-thr", type=float, default=0.20)
    parser.add_argument("--gcs-cost-point", type=float, default=5.0, help="Hungarian matching point cost weight.")
    parser.add_argument("--gcs-cost-exist", type=float, default=0.1, help="Hungarian matching existence cost weight.")
    parser.add_argument("--gcs-match-min-overlap", type=int, default=2, help="Minimum valid GT points for training Hungarian matching.")
    parser.add_argument("--gcs-match-max-x-dist", type=float, default=0.0, help="Optional training matcher mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--gcs-match-gate-px", type=float, default=160.0, help="Training matcher APE gate in pixels. 0 disables.")
    parser.add_argument("--gcs-eval-match-gate-px", type=float, default=None, help="Strict validation APE gate in pixels. Defaults to --ape-thr.")
    parser.add_argument("--gcs-eval-max-x-dist", type=float, default=0.0, help="Optional strict validation mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--gcs-eval-min-overlap", type=int, default=6, help="Minimum overlapping visible anchors for strict validation matching.")
    parser.add_argument("--gcs-eval-min-points", type=int, default=6, help="Minimum decoded visible anchors required to keep a validation prediction.")
    parser.add_argument("--gcs-eval-min-gt-cover-ratio", type=float, default=0.3, help="Minimum GT visible-anchor coverage ratio for strict validation matching.")
    parser.add_argument("--gcs-eval-min-pred-cover-ratio", type=float, default=0.3, help="Minimum predicted visible-anchor coverage ratio for strict validation matching.")
    parser.add_argument("--gcs-eval-nms-dist-px", type=float, default=18.0, help="Validation/inference lane NMS distance in pixels. 0 disables.")
    parser.add_argument(
        "--gcs-eval-point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold used when decoding fixed-y lanes for validation/inference.",
    )
    parser.add_argument("--hsv-h", type=float, default=0.0, help="Overfit HSV hue gain. Keep 0 for memorization checks.")
    parser.add_argument("--hsv-s", type=float, default=0.0, help="Overfit HSV saturation gain. Keep 0 for memorization checks.")
    parser.add_argument("--hsv-v", type=float, default=0.0, help="Overfit HSV value gain. Keep 0 for memorization checks.")
    parser.add_argument("--fliplr", type=float, default=0.0, help="Overfit horizontal flip probability. Keep 0 for memorization checks.")
    parser.add_argument("--ape-thr", type=float, default=20.0, help="APE threshold in pixels for overfit F1.")
    parser.add_argument(
        "--pred-conf",
        type=float,
        default=0.2,
        help="Confidence threshold for post-training render. Quality-aware exist targets require a lower calibrated threshold.",
    )
    parser.add_argument("--skip-predict", action="store_true", help="Skip rendering predictions after training.")
    return parser.parse_args()


def parse_pretrained(value: str) -> str | bool:
    normalized = str(value).strip().lower()
    if normalized in {"", "false", "none", "no", "0"}:
        return False
    return value


def gcs_label_lane_count(label_path: str | Path) -> int:
    """Read the number of valid GT lanes from one GCS npz label."""
    with np.load(label_path, allow_pickle=False) as data:
        if "num_lanes" in data:
            return int(np.asarray(data["num_lanes"]).reshape(-1)[0])
        valid = data["lane_valid"].astype(np.float32)
    return int((valid.sum(axis=1) >= 2).sum())


def collect_overfit_pairs(
    image_dir: str | Path,
    label_dir: str | Path,
    limit: int,
) -> list[tuple[Path, Path]]:
    """Select a GT lane-count stratified image/label subset without copying dataset files."""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    if limit <= 0:
        raise ValueError(f"--limit must be positive, got {limit}.")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"GCS label directory not found: {label_dir}")

    images = sorted(p for p in image_dir.rglob("*.*") if p.suffix[1:].lower() in IMG_FORMATS)
    pairs = []
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.npz"
        if label_path.exists():
            pairs.append((image_path.resolve(), label_path.resolve()))
    if len(pairs) < limit:
        raise FileNotFoundError(
            f"Only found {len(pairs)} valid image/labels_gcs pairs under {image_dir} and {label_dir}; "
            f"{limit} are required for this overfit test."
        )

    groups: dict[int, list[tuple[Path, Path]]] = {}
    for image_path, label_path in pairs:
        groups.setdefault(gcs_label_lane_count(label_path), []).append((image_path, label_path))

    selected: list[tuple[Path, Path]] = []
    counts = sorted(groups)
    while len(selected) < limit and any(groups[count] for count in counts):
        for count in counts:
            if groups[count]:
                selected.append(groups[count].pop(0))
                if len(selected) >= limit:
                    break
    return selected


def write_subset_files(pairs: list[tuple[Path, Path]], project: str | Path, name: str) -> tuple[Path, Path]:
    """Write a deterministic image-list txt and manifest for the overfit subset."""
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    list_path = project / f"{name}_images.txt"
    manifest_path = project / f"{name}_manifest.json"

    list_path.write_text("\n".join(str(image) for image, _ in pairs) + "\n", encoding="utf-8")
    manifest = [{"image": str(image), "label": str(label), "lane_count": gcs_label_lane_count(label)} for image, label in pairs]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return list_path, manifest_path


def check_subset_contract(image_list: Path, label_dir: str | Path, imgsz: tuple[int, int]) -> None:
    """Fail early if the 20-image subset violates GCS label contract."""
    dataset = GCSLaneDataset(img_path=image_list, imgsz=imgsz, label_dir=label_dir, strict=True)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        lanes = sample["lanes"]
        lane_valid = sample["lane_valid"]
        if lanes.numel():
            coords = lanes[lane_valid > 0.5]
            if coords.numel() and (coords.min() < 0 or coords.max() > 1):
                raise AssertionError(f"{sample['label_file']}: GCS overfit labels must be normalized to [0, 1].")
            for lane, valid in zip(lanes, lane_valid):
                ys = lane[valid > 0.5, 1]
                if ys.numel() > 1 and torch.any(ys[1:] > ys[:-1] + 1e-6):
                    raise AssertionError(
                        f"{sample['label_file']}: GCS overfit lane points must be bottom-to-top by descending y."
                    )


def _load_loss_gains(save_dir: Path) -> dict[str, float]:
    """Load GCS loss gains from args.yaml, falling back to the GCSLoss defaults."""
    gains = dict(DEFAULT_LOSS_GAINS)
    args_path = save_dir / "args.yaml"
    if not args_path.exists():
        return gains
    with args_path.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f) or {}
    key_map = {
        "exist_loss": "gcs_exist",
        "point_loss": "gcs_point",
        "point_valid_loss": "gcs_point_valid",
        "line_iou_loss": "gcs_line_iou",
        "count_cls_loss": "gcs_count_cls",
        "count_sum_loss": "gcs_count_sum",
        "quality_loss": "gcs_quality",
    }
    for loss_name, arg_name in key_map.items():
        if isinstance(arg_name, tuple):
            primary, legacy = arg_name
            if primary in args:
                gains[loss_name] = float(args[primary])
            elif legacy in args:
                gains[loss_name] = float(args[legacy])
        elif arg_name in args:
            gains[loss_name] = float(args[arg_name])
    return gains


def _weighted_total(row: dict[str, str], prefix: str, gains: dict[str, float]) -> float:
    """Compute the same weighted joint loss form used by GCSLoss from logged components."""
    total = 0.0
    for loss_name, gain in gains.items():
        key = f"{prefix}/{loss_name}"
        if row.get(key) not in {None, ""}:
            total += gain * float(row[key])
    return total


def summarize_overfit_results(save_dir: str | Path) -> Path | None:
    """Summarize first/last train losses from Ultralytics results.csv for the overfit check."""
    save_dir = Path(save_dir)
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    first = rows[0]
    last = rows[-1]
    gains = _load_loss_gains(save_dir)
    loss_keys = (
        "train/exist_loss",
        "train/point_loss",
        "train/point_valid_loss",
        "train/line_iou_loss",
        "train/count_cls_loss",
        "train/count_sum_loss",
        "train/quality_loss",
        "val/exist_loss",
        "val/point_loss",
        "val/point_valid_loss",
        "val/line_iou_loss",
        "val/count_cls_loss",
        "val/count_sum_loss",
        "val/quality_loss",
    )

    losses = {}
    for key in loss_keys:
        if key not in first or key not in last or first[key] == "" or last[key] == "":
            continue
        start = float(first[key])
        end = float(last[key])
        losses[key] = {
            "start": round(start, 6),
            "end": round(end, 6),
            "delta": round(end - start, 6),
            "decreased": end < start,
        }

    train_total_start = _weighted_total(first, "train", gains)
    train_total_end = _weighted_total(last, "train", gains)
    val_total_start = _weighted_total(first, "val", gains)
    val_total_end = _weighted_total(last, "val", gains)

    summary = {
        "epochs_recorded": len(rows),
        "loss_gains": gains,
        "train_weighted_total_loss": {
            "start": round(train_total_start, 6),
            "end": round(train_total_end, 6),
            "delta": round(train_total_end - train_total_start, 6),
            "decreased": train_total_end < train_total_start,
        },
        "val_weighted_total_loss": {
            "start": round(val_total_start, 6),
            "end": round(val_total_end, 6),
            "delta": round(val_total_end - val_total_start, 6),
            "decreased": val_total_end < val_total_start,
        },
        "losses": losses,
        "interpretation": (
            "20-image overfit is considered healthy only when total, existence, point, point-valid, LineIoU, "
            "Count Head, and Quality Head losses trend down and rendered predictions align with GT lanes."
        ),
    }
    out_path = save_dir / "overfit_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def evaluate_overfit_predictions(
    predictions_path: str | Path,
    manifest_path: str | Path,
    ape_thr: float = 20.0,
    match_gate_px: float | None = None,
    max_x_dist: float = 0.0,
    min_overlap: int = 6,
) -> Path | None:
    """Evaluate decoded overfit predictions against the exact 20-image GT subset."""
    predictions_path = Path(predictions_path)
    manifest_path = Path(manifest_path)
    if not predictions_path.exists() or not manifest_path.exists():
        return None

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_by_image = {str(Path(x["image"]).resolve()): Path(x["label"]) for x in manifest}

    matched_apes = []
    matched_all_apes = []
    matched_fp_apes = []
    tp = fp = fn = 0
    per_image = []
    pred_counts = []
    gt_counts = []
    blank_gt_images = []
    for rec in predictions:
        image_path = str(Path(rec["image"]).resolve())
        label_path = label_by_image.get(image_path)
        if label_path is None or not label_path.exists():
            continue

        with np.load(label_path) as data:
            gt = data["lanes"].astype(np.float32)
            valid = data["lane_valid"].astype(np.float32)
            label_shape = data["image_shape"].astype(np.float32) if "image_shape" in data else np.array([544.0, 960.0])

        pred_lanes = rec.get("lanes", [])
        h = float(rec.get("height", label_shape[0]))
        w = float(rec.get("width", label_shape[1]))

        metrics, matches = match_lanes(
            pred_lanes,
            gt,
            valid,
            image_shape=(int(h), int(w)),
            ape_thr=ape_thr,
            match_gate_px=match_gate_px,
            max_x_dist=max_x_dist,
            min_overlap=min_overlap,
        )
        matched_apes.extend(float(x) for x in metrics.get("ape_tp", metrics.get("ape", [])))
        matched_all_apes.extend(float(x) for x in metrics.get("ape_matched_all", []))
        matched_fp_apes.extend(float(x) for x in metrics.get("ape_fp_matched", []))
        image_tp = int(metrics["tp"])
        tp += image_tp
        fp += int(metrics["fp"])
        fn += int(metrics["fn"])
        pred_counts.append(int(len(pred_lanes)))
        gt_counts.append(int(gt.shape[0]))
        if int(gt.shape[0]) > 0 and int(len(pred_lanes)) == 0:
            blank_gt_images.append(image_path)
        per_image.append(
            {
                "image": image_path,
                "pred": int(len(pred_lanes)),
                "gt": int(gt.shape[0]),
                "ape": [round(float(x), 3) for x in metrics.get("ape_tp", metrics.get("ape", []))],
                "ape_matched_all": [round(float(x), 3) for x in metrics.get("ape_matched_all", [])],
                "ape_fp_matched": [round(float(x), 3) for x in metrics.get("ape_fp_matched", [])],
                "matches": matches,
                "tp": image_tp,
            }
        )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    lane_count_mae = float(np.mean([abs(p - g) for p, g in zip(pred_counts, gt_counts)])) if pred_counts else 0.0
    count_diversity_ok = len(set(gt_counts)) > 1
    metrics = {
        "ape_threshold_px": float(ape_thr),
        "ape_mean_px": round(float(np.mean(matched_apes)), 3) if matched_apes else None,
        "ape_median_px": round(float(np.median(matched_apes)), 3) if matched_apes else None,
        "ape_min_px": round(float(np.min(matched_apes)), 3) if matched_apes else None,
        "ape_max_px": round(float(np.max(matched_apes)), 3) if matched_apes else None,
        "ape_tp_mean_px": round(float(np.mean(matched_apes)), 3) if matched_apes else None,
        "ape_matched_all_mean_px": round(float(np.mean(matched_all_apes)), 3) if matched_all_apes else None,
        "ape_fp_matched_mean_px": round(float(np.mean(matched_fp_apes)), 3) if matched_fp_apes else None,
        "ape_all_matched_mean_px": round(float(np.mean(matched_all_apes)), 3) if matched_all_apes else None,
        "fp_matched_ape_mean_px": round(float(np.mean(matched_fp_apes)), 3) if matched_fp_apes else None,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "fp_per_image": round(float(fp) / max(len(per_image), 1), 6),
        "fn_per_image": round(float(fn) / max(len(per_image), 1), 6),
        "lane_count_mae": round(lane_count_mae, 6),
        "pred_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(pred_counts).items())},
        "gt_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(gt_counts).items())},
        "gt_pred_lanes_hist": {
            f"{gt}->{pred}": int(count) for (gt, pred), count in sorted(Counter(zip(gt_counts, pred_counts)).items())
        },
        "blank_gt_images": blank_gt_images,
        "blank_gt_image_count": int(len(blank_gt_images)),
        "count_diversity_ok": bool(count_diversity_ok),
        "passed": bool(
            matched_apes
            and np.mean(matched_apes) < ape_thr
            and f1 > 0.8
            and count_diversity_ok
            and not blank_gt_images
        ),
        "per_image": per_image,
    }
    out_path = predictions_path.parent / "overfit_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    gcs_imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    defaults = dataset_defaults(args.dataset)
    data = args.data or str(defaults["data"])
    train_images = args.train_images or str(defaults["train_images"])
    train_gcs_labels = args.train_gcs_labels or str(defaults["train_labels"])
    name = args.name or f"overfit20_{args.dataset}"
    project = Path(args.project)
    pairs = collect_overfit_pairs(train_images, train_gcs_labels, args.limit)
    image_list, manifest_path = write_subset_files(pairs, project=project, name=name)
    check_subset_contract(image_list, train_gcs_labels, gcs_imgsz)

    print(f"GCS input shape: {shape_str(gcs_imgsz)} (W x H), stored as H,W={gcs_imgsz}")
    print(f"overfit subset: {len(pairs)} images")
    print(f"image list: {image_list.resolve()}")
    print(f"manifest: {manifest_path.resolve()}")

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
        "mosaic": args.mosaic,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "fliplr": args.fliplr,
        "val": True,
        "exist_ok": args.exist_ok,
        "box": 0.0,
        "cls": 0.0,
        "dfl": 0.0,
        "pose": 0.0,
        "kobj": 0.0,
        "rle": 0.0,
        "angle": 0.0,
        "train_images": str(image_list.resolve()),
        "train_gcs_labels": train_gcs_labels,
        "val_images": str(image_list.resolve()),
        "val_gcs_labels": train_gcs_labels,
        "gcs_exist": args.gcs_exist,
        "gcs_point": args.gcs_point,
        "gcs_point_valid": args.gcs_point_valid,
        "gcs_point_invalid_x": args.gcs_point_invalid_x,
        "gcs_line_iou": args.gcs_line_iou,
        "gcs_quality": args.gcs_quality,
        "gcs_quality_dist_thr_px": args.gcs_quality_dist_thr_px,
        "gcs_quality_neg_weight": args.gcs_quality_neg_weight,
        "gcs_quality_hard_negative_weight": args.gcs_quality_hard_negative_weight,
        "gcs_quality_duplicate_negative_weight": args.gcs_quality_duplicate_negative_weight,
        "gcs_line_iou_width_px": args.gcs_line_iou_width_px,
        "gcs_count_cls": args.gcs_count_cls,
        "gcs_count_head_warmup_epochs": args.gcs_count_head_warmup_epochs,
        "gcs_count_min_gt_points": args.gcs_count_min_gt_points,
        "gcs_count_cls_w2": args.gcs_count_cls_w2,
        "gcs_count_cls_w3": args.gcs_count_cls_w3,
        "gcs_count_cls_w4": args.gcs_count_cls_w4,
        "gcs_count_cls_w5": args.gcs_count_cls_w5,
        "gcs_exist_pos_weight": args.gcs_exist_pos_weight,
        "gcs_exist_focal_gamma": args.gcs_exist_focal_gamma,
        "gcs_exist_focal_alpha": args.gcs_exist_focal_alpha,
        "gcs_hard_negative_quality_thr": args.gcs_hard_negative_quality_thr,
        "gcs_hard_negative_topk": args.gcs_hard_negative_topk,
        "gcs_hard_negative_exist_weight": args.gcs_hard_negative_exist_weight,
        "gcs_duplicate_negative_exist_weight": args.gcs_duplicate_negative_exist_weight,
        "gcs_duplicate_dist_thr_px": args.gcs_duplicate_dist_thr_px,
        "gcs_duplicate_iou_thr": args.gcs_duplicate_iou_thr,
        "gcs_exist_margin": args.gcs_exist_margin,
        "gcs_exist_pos_margin": args.gcs_exist_pos_margin,
        "gcs_exist_neg_margin": args.gcs_exist_neg_margin,
        "gcs_exist_quality_alpha": args.gcs_exist_quality_alpha,
        "gcs_exist_quality_lane_iou_alpha": args.gcs_exist_quality_lane_iou_alpha,
        "gcs_exist_quality_mode": args.gcs_exist_quality_mode,
        "gcs_exist_quality_tau": args.gcs_exist_quality_tau,
        "gcs_exist_quality_floor": args.gcs_exist_quality_floor,
        "gcs_exist_quality_pos_px": args.gcs_exist_quality_pos_px,
        "gcs_exist_quality_neg_px": args.gcs_exist_quality_neg_px,
        "gcs_point_valid_pos_weight_max": args.gcs_point_valid_pos_weight_max,
        "gcs_point_valid_gt5_pos_weight": args.gcs_point_valid_gt5_pos_weight,
        "gcs_point_valid_unmatched_weight": args.gcs_point_valid_unmatched_weight,
        "gcs_point_valid_hard_negative_weight": args.gcs_point_valid_hard_negative_weight,
        "gcs_point_valid_duplicate_negative_weight": args.gcs_point_valid_duplicate_negative_weight,
        "gcs_point_valid_neg": args.gcs_point_valid_neg,
        "gcs_point_valid_neg_thr": args.gcs_point_valid_neg_thr,
        "gcs_cost_point": args.gcs_cost_point,
        "gcs_cost_exist": args.gcs_cost_exist,
        "gcs_match_min_overlap": args.gcs_match_min_overlap,
        "gcs_match_max_x_dist": args.gcs_match_max_x_dist,
        "gcs_match_gate_px": args.gcs_match_gate_px,
        "gcs_eval_match_gate_px": args.gcs_eval_match_gate_px,
        "gcs_eval_max_x_dist": args.gcs_eval_max_x_dist,
        "gcs_eval_min_overlap": args.gcs_eval_min_overlap,
        "gcs_eval_min_points": args.gcs_eval_min_points,
        "gcs_eval_min_gt_cover_ratio": args.gcs_eval_min_gt_cover_ratio,
        "gcs_eval_min_pred_cover_ratio": args.gcs_eval_min_pred_cover_ratio,
        "gcs_eval_nms_dist_px": args.gcs_eval_nms_dist_px,
        "gcs_eval_point_valid_thr": args.gcs_eval_point_valid_thr,
        "gcs_sampler_mode": "none",
        "gcs_hard_sampling": False,
        "gcs_hard_sample_file": "",
        "gcs_gt5_extra_aug": False,
    }

    trainer = GCSLaneTrainer(overrides=overrides)
    trainer.train()

    save_dir = Path(trainer.save_dir)
    summary_path = summarize_overfit_results(save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    weights = best if best.exists() else last

    if not args.skip_predict and weights.exists():
        run_inference(
            weights=weights,
            source=image_list,
            save_dir=save_dir / "overfit_predictions",
            imgsz=gcs_imgsz,
            conf=args.pred_conf,
            point_valid_thr=args.gcs_eval_point_valid_thr,
            min_points=args.gcs_eval_min_points,
            nms_dist_px=args.gcs_eval_nms_dist_px,
            device=args.device,
            half=False,
            max_det=GCS_DEFAULT_MAX_DET,
            max_images=args.limit,
            save_img=True,
            save_txt=True,
            save_json=True,
            line_width=2,
        )
        metrics_path = evaluate_overfit_predictions(
            save_dir / "overfit_predictions" / "predictions.json",
            manifest_path,
            ape_thr=args.ape_thr,
            match_gate_px=args.gcs_eval_match_gate_px,
            max_x_dist=args.gcs_eval_max_x_dist,
            min_overlap=args.gcs_eval_min_overlap,
        )
    else:
        metrics_path = None

    print(f"training run: {save_dir.resolve()}")
    if summary_path is not None:
        print(f"overfit summary: {summary_path.resolve()}")
    if metrics_path is not None:
        print(f"overfit metrics: {metrics_path.resolve()}")
    print(
        "success signals: weighted total, existence, point, point-valid, LineIoU, and Count Head losses should fall; "
        "overfit_metrics.json should show low APE and high F1."
    )


if __name__ == "__main__":
    main()
