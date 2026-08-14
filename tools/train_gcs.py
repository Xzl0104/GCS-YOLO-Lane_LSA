from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer
from ultralytics.models.gcs.mode_utils import assert_ordered_slot_scale_contract
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str, trainer_imgsz


DEFAULT_MODEL = "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"
DEFAULT_DATA = "data/tusimple_gcs_fixed_y_960x544.yaml"
ORDERED_SLOT_DEFAULT_MODEL = "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q5-slot-k56.yaml"
ORDERED_SLOT_MODEL = ROOT / ORDERED_SLOT_DEFAULT_MODEL


def str2bool(value: str | bool) -> bool:
    """Parse shell-friendly boolean values for argparse options that may take True/False."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def dataset_defaults(dataset: str) -> dict[str, Path]:
    """Return conventional local paths for a converted GCS dataset."""
    name = dataset.lower()
    if name == "tusimple":
        fixed_root = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544"
        fixed_data = Path(DEFAULT_DATA)
        if fixed_data.exists():
            return {
                "data": fixed_data,
                "train_images": fixed_root / "images" / "train",
                "train_labels": fixed_root / "labels_gcs" / "train",
                "val_images": fixed_root / "images" / "val",
                "val_labels": fixed_root / "labels_gcs" / "val",
            }
    root = ROOT / "datasets" / name
    return {
        "data": ROOT / "data" / f"{name}_gcs.yaml",
        "train_images": root / "images" / "train",
        "train_labels": root / "labels_gcs" / "train",
        "val_images": root / "images" / "val",
        "val_labels": root / "labels_gcs" / "val",
    }


def is_ordered_slot_model_yaml(model_path: str | Path) -> bool:
    """Return whether a model yaml path names an ordered-slot config."""
    name = Path(str(model_path)).name.lower()
    return "q5-slot" in name or "ordered_slot" in name or "ordered-slot" in name


def maybe_switch_ordered_slot_model(args: argparse.Namespace) -> argparse.Namespace:
    """Switch ordered-slot runs onto the ordered-slot default model yaml unless disabled."""
    assert_ordered_slot_scale_contract(getattr(args, "gcs_mode", "query"), getattr(args, "scale", 0.0))
    if getattr(args, "gcs_mode", "query") != "ordered_slot":
        return args
    if is_ordered_slot_model_yaml(getattr(args, "model", "")):
        return args
    if bool(getattr(args, "gcs_disable_auto_model_switch", False)):
        raise RuntimeError(
            f"--gcs-mode ordered_slot requires an ordered_slot model yaml, but got {args.model}. "
            f"Please use {ORDERED_SLOT_DEFAULT_MODEL}."
        )
    old_model = args.model
    args.model = ORDERED_SLOT_DEFAULT_MODEL
    print(f"[GCS][STRICT] ordered_slot auto-switch: {old_model} -> {args.model}")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GCS-YOLO-Lane on structured lane labels.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=None)
    parser.add_argument("--pretrained", default="yolo11s-seg.pt")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS input shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str((ROOT / "runs/gcs_lane").resolve()))
    parser.add_argument("--name", default=None)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument(
        "--cos_lr",
        "--cos-lr",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="Use cosine LR scheduling. Accepts '--cos_lr', '--cos_lr True', or '--cos-lr true'.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument(
        "--warmup-bias-lr",
        type=float,
        default=0.0,
        help="Initial warmup LR for bias parameter groups. Keep 0.0 for GCS heads unless deliberately testing warmup.",
    )
    parser.add_argument("--nbs", type=int, default=0, help="Nominal batch size. 0 uses --batch for GCS training.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA algorithms. Off by default for GCS because several CUDA ops warn/fallback.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision if a run shows numerical instability.",
    )
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="GCS-safe random center scaling gain. 0 disables; 0.3 samples scale from 0.7 to 1.3.",
    )
    parser.add_argument(
        "--erasing",
        type=float,
        default=0.0,
        help="Random erasing probability applied to training images only for occlusion robustness.",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="GCS-safe mosaic probability. Keep 0.0 unless num_queries is raised above mosaic GT lane count.",
    )
    parser.add_argument("--train-images", default=None)
    parser.add_argument("--train-gcs-labels", default=None)
    parser.add_argument("--val-images", default=None)
    parser.add_argument("--val-gcs-labels", default=None)
    parser.add_argument("--gcs-mode", choices=("query", "ordered_slot"), default="query")
    parser.add_argument(
        "--gcs-disable-auto-model-switch",
        action="store_true",
        help="Disable automatic ordered_slot model YAML switching and fail on non-slot model YAMLs.",
    )
    parser.add_argument("--gcs-num-slots", type=int, default=5)
    parser.add_argument("--gcs-min-lanes", type=int, default=2, help="ordered_slot minimum supported lane count.")
    parser.add_argument("--gcs-max-lanes", type=int, default=5, help="ordered_slot maximum supported lane count.")
    parser.add_argument("--gcs-count-classes", type=int, default=4, help="ordered_slot count classes for 2/3/4/5 lanes.")
    parser.add_argument(
        "--gcs-contiguity-policy",
        choices=("strict", "repair_interp"),
        default="strict",
        help="ordered_slot fixed-y valid-mask policy.",
    )
    parser.add_argument("--gcs-exist", type=float, default=2.0)
    parser.add_argument("--gcs-point", type=float, default=15.0)
    parser.add_argument("--gcs-point-valid", type=float, default=1.0)
    parser.add_argument("--gcs-smooth", type=float, default=0.05)
    parser.add_argument("--gcs-curve", type=float, default=0.1)
    parser.add_argument("--gcs-mask", type=float, default=0.2)
    parser.add_argument("--gcs-edge", type=float, default=0.2)
    parser.add_argument(
        "--gcs-count",
        type=float,
        default=0.0,
        help="Cardinality loss gain for matching sum(sigmoid(pred_logits)) to GT lane count. 0 disables.",
    )
    parser.add_argument(
        "--gcs-count-under5",
        type=float,
        default=0.0,
        help="Extra undercount loss gain for samples with GT lane count >= --gcs-count-under5-min-lanes. 0 disables.",
    )
    parser.add_argument(
        "--gcs-count-under5-min-lanes",
        type=int,
        default=5,
        help="Minimum GT lane count that enables the targeted undercount penalty.",
    )
    parser.add_argument(
        "--gcs-count-boundary",
        type=float,
        default=0.0,
        help="GT3/GT4/GT5 adjacent lane-count boundary loss gain. 0 disables.",
    )
    parser.add_argument(
        "--gcs-count-boundary-gt4-weight",
        type=float,
        default=2.0,
        help="Weight for the GT4 lower/upper count boundary terms.",
    )
    parser.add_argument(
        "--gcs-count-boundary-gt5-weight",
        type=float,
        default=1.5,
        help="Weight for the GT5 undercount boundary term.",
    )
    parser.add_argument(
        "--gcs-count-boundary-gt5-under-weight",
        type=float,
        default=None,
        help="Optional separate weight for the GT5 undercount boundary term. Defaults to --gcs-count-boundary-gt5-weight.",
    )
    parser.add_argument(
        "--gcs-count-boundary-margin34",
        type=float,
        default=0.35,
        help="Margin around the 3/4 boundary: GT3 upper=3+margin, GT4 lower=4-margin.",
    )
    parser.add_argument(
        "--gcs-count-boundary-margin45",
        type=float,
        default=0.35,
        help="Margin around the 4/5 boundary: GT4 upper=4+margin, GT5 lower=5-margin.",
    )
    parser.add_argument("--gcs-query-count-ce", type=float, default=0.0)
    parser.add_argument("--gcs-query-count-min-lanes", type=int, default=2)
    parser.add_argument("--gcs-query-count-max-lanes", type=int, default=5)
    parser.add_argument(
        "--gcs-count-ce",
        nargs="?",
        const=1.0,
        type=float,
        default=1.0,
        help="ordered_slot count CE gain for 2/3/4/5 classification. Use without a value for 1.0.",
    )
    parser.add_argument(
        "--gcs-interval",
        nargs="?",
        const=1.0,
        type=float,
        default=1.0,
        help="ordered_slot start/end interval CE gain. Use without a value for 1.0.",
    )
    parser.add_argument(
        "--gcs-order",
        nargs="?",
        const=1.0,
        type=float,
        default=0.2,
        help="ordered_slot common-anchor adjacent-slot order loss gain. Use without a value for 1.0.",
    )
    parser.add_argument(
        "--gcs-gt-bottom-order",
        type=float,
        default=1.0,
        help="ordered_slot GT-bottom adjacent-slot order loss gain.",
    )
    parser.add_argument(
        "--gcs-decoded-bottom-order",
        type=float,
        default=1.0,
        help="ordered_slot decoded-bottom adjacent-slot order loss gain.",
    )
    parser.add_argument(
        "--gcs-slot-gt-bottom-x",
        type=float,
        default=0.0,
        help="ordered_slot decoded bottom-x to GT bottom-x regression gain. 0 disables.",
    )
    parser.add_argument(
        "--gcs-slot-gt-bottom-x-beta",
        type=float,
        default=0.05,
        help="SmoothL1 beta for ordered_slot GT bottom-x regression in normalized x.",
    )
    parser.add_argument(
        "--gcs-slot-gt-bottom-x-detach-interval",
        type=int,
        default=1,
        help="1 detaches start/end logits when selecting decoded bottom index for GT bottom-x loss; 0 keeps them attached.",
    )
    parser.add_argument(
        "--gcs-slot-gt-bottom-x-soft",
        type=float,
        default=0.0,
        help="Gain for differentiable soft-start GT bottom-x loss. 0 disables.",
    )
    parser.add_argument(
        "--gcs-slot-gt-bottom-x-soft-tau",
        type=float,
        default=0.5,
        help="Temperature for softmax over start logits in soft GT bottom-x loss.",
    )
    parser.add_argument(
        "--gcs-slot-gt-bottom-x-soft-beta",
        type=float,
        default=0.05,
        help="SmoothL1 beta for soft GT bottom-x loss in normalized x.",
    )
    parser.add_argument(
        "--gcs-slot-start-index-l1",
        type=float,
        default=0.0,
        help="Gain for differentiable soft start-index L1 loss. 0 disables.",
    )
    parser.add_argument(
        "--gcs-slot-start-index-l1-beta",
        type=float,
        default=2.0,
        help="SmoothL1 beta for soft start-index L1 loss in anchor units.",
    )
    parser.add_argument(
        "--gcs-allow-disable-order-loss",
        action="store_true",
        help="Allow ordered_slot order losses <= 0 only for explicit ablations.",
    )
    parser.add_argument("--gcs-slot-exist-w4", type=float, default=1.0, help="ordered_slot BCE weight for slot 4.")
    parser.add_argument("--gcs-slot-exist-w5", type=float, default=1.0, help="ordered_slot BCE weight for slot 5.")
    parser.add_argument(
        "--gcs-min-interval-points",
        type=int,
        default=2,
        help="ordered_slot minimum decoded start/end interval length.",
    )
    parser.add_argument("--gcs-order-margin-px", type=float, default=5.0, help="ordered_slot left-to-right margin in pixels.")
    parser.add_argument(
        "--gcs-bottom-order-margin-px",
        type=float,
        default=2.0,
        help="ordered_slot bottom-x left-to-right margin in pixels.",
    )
    parser.add_argument(
        "--gcs-ordered-point-loss",
        choices=("aspect_l1", "pixel_smooth_l1", "normalized_smooth_l1"),
        default="normalized_smooth_l1",
        help="ordered_slot point regression loss. Use aspect_l1 only for an explicit point-loss ablation.",
    )
    parser.add_argument("--gcs-point-y-weight", type=float, default=0.25, help="ordered_slot y-coordinate point-loss weight.")
    parser.add_argument("--gcs-point-x-only", action="store_true", help="Train ordered_slot point regression on x only.")
    parser.add_argument(
        "--gcs-pixel-smoothl1-beta",
        type=float,
        default=1.0,
        help="Pixel-space SmoothL1 beta when --gcs-ordered-point-loss pixel_smooth_l1 is selected.",
    )
    parser.add_argument(
        "--gcs-spurious-neg",
        type=float,
        default=0.0,
        help="Extra BCE negative loss gain for short unmatched duplicate lane queries. 0 disables.",
    )
    parser.add_argument(
        "--gcs-spurious-neg-weight",
        type=float,
        default=1.0,
        help="Multiplier inside the spurious-negative loss term.",
    )
    parser.add_argument(
        "--gcs-spurious-gt3-weight",
        type=float,
        default=1.0,
        help="Per-image multiplier for spurious-negative loss on GT3-or-sparser samples.",
    )
    parser.add_argument(
        "--gcs-spurious-gt4-weight",
        type=float,
        default=1.0,
        help="Per-image multiplier for spurious-negative loss on GT4 samples.",
    )
    parser.add_argument(
        "--gcs-spurious-gt5-weight",
        type=float,
        default=1.0,
        help="Per-image multiplier for spurious-negative loss on GT5-or-denser samples.",
    )
    parser.add_argument(
        "--gcs-spurious-disable-gt5",
        action="store_true",
        help="Disable spurious-negative loss on GT5-or-denser samples.",
    )
    parser.add_argument(
        "--gcs-spurious-max-points",
        type=int,
        default=12,
        help="Maximum visible anchors for an unmatched query to be treated as a short spurious duplicate.",
    )
    parser.add_argument(
        "--gcs-spurious-close-px",
        type=float,
        default=30.0,
        help="Maximum mean x distance in pixels to a matched query over overlapping visible anchors.",
    )
    parser.add_argument(
        "--gcs-spurious-min-overlap",
        type=int,
        default=3,
        help="Minimum overlapping visible anchors with a matched query for spurious duplicate detection.",
    )
    parser.add_argument(
        "--gcs-spurious-gt-protect",
        action="store_true",
        help="Protect duplicate-like spurious candidates that are close to a GT lane.",
    )
    parser.add_argument(
        "--gcs-spurious-gt-protect-px",
        type=float,
        default=25.0,
        help="Maximum mean x distance in pixels for GT-aware spurious candidate protection.",
    )
    parser.add_argument(
        "--gcs-spurious-gt-protect-min-overlap",
        type=int,
        default=3,
        help="Minimum overlapping anchors with a GT lane for GT-aware spurious candidate protection.",
    )
    parser.add_argument(
        "--gcs-spurious-gt-protect-margin-px",
        type=float,
        default=5.0,
        help="Protect if candidate mean GT dx plus this margin is below the matched query mean GT dx.",
    )
    parser.add_argument(
        "--gcs-spurious-gt-protect-mode",
        type=str,
        default="better_matched",
        choices=("better_matched",),
        help="GT-aware spurious protection policy.",
    )
    parser.add_argument(
        "--gcs-gt5-short-visible-thr",
        type=int,
        default=0,
        help="Boost matched GT5 lane point-valid positives when visible_count <= this threshold. 0 disables.",
    )
    parser.add_argument(
        "--gcs-gt5-short-point-valid-weight",
        type=float,
        default=1.0,
        help="Extra point-valid BCE multiplier for visible anchors of matched short GT5 lanes.",
    )
    parser.add_argument("--gcs-short-geom", type=float, default=0.0)
    parser.add_argument("--gcs-short-geom-visible-thr", type=int, default=10)
    parser.add_argument("--gcs-short-geom-gt4-weight", type=float, default=1.0)
    parser.add_argument("--gcs-short-geom-gt5-weight", type=float, default=2.0)
    parser.add_argument("--gcs-short-geom-max-weight", type=float, default=3.0)
    parser.add_argument("--gcs-short-geom-curve", type=float, default=1.0)
    parser.add_argument("--gcs-short-survival", type=float, default=0.0)
    parser.add_argument("--gcs-short-survival-visible-thr", type=int, default=10)
    parser.add_argument("--gcs-short-survival-gt4-weight", type=float, default=1.0)
    parser.add_argument("--gcs-short-survival-gt5-weight", type=float, default=1.0)
    parser.add_argument("--gcs-short-survival-exist-weight", type=float, default=1.0)
    parser.add_argument("--gcs-short-survival-valid-weight", type=float, default=1.0)
    parser.add_argument("--gcs-boundary-pseudo-neg", type=float, default=0.0)
    parser.add_argument("--gcs-boundary-pseudo-visible-thr", type=int, default=10)
    parser.add_argument("--gcs-boundary-pseudo-dist-thr", type=float, default=60.0)
    parser.add_argument("--gcs-boundary-pseudo-valid-thr", type=float, default=0.5)
    parser.add_argument("--gcs-boundary-pseudo-min-valid", type=int, default=3)
    parser.add_argument("--gcs-boundary-pseudo-gt-count", type=int, default=5)
    parser.add_argument("--gcs-boundary-pseudo-score-thr", type=float, default=0.0)
    parser.add_argument("--gcs-boundary-pseudo-envelope-margin-px", type=float, default=-1.0)
    parser.add_argument("--gcs-boundary-pseudo-envelope-ratio-thr", type=float, default=0.75)
    parser.add_argument("--gcs-exist-pos-weight", type=float, default=1.0)
    parser.add_argument("--gcs-exist-focal-gamma", type=float, default=0.0, help="Optional focal gamma for existence BCE. 0 disables focal weighting.")
    parser.add_argument(
        "--gcs-exist-focal-alpha",
        type=float,
        default=-1.0,
        help="Optional focal alpha for existence BCE. Use a value in [0, 1] to enable alpha weighting.",
    )
    parser.add_argument(
        "--gcs-exist-quality-alpha",
        type=float,
        default=1.0,
        help="Blend factor for quality-aware existence targets. 1 uses pure geometry quality, 0 restores hard matched labels.",
    )
    parser.add_argument(
        "--gcs-exist-quality-mode",
        choices=("linear", "exp"),
        default="linear",
        help="Quality target shape. linear makes APE >= neg-px a zero target; exp keeps the old exp(-APE/tau) behavior.",
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
        default=20.0,
        help="APE at or above this value receives quality 0.0 in linear quality mode.",
    )
    parser.add_argument("--gcs-mask-pos-weight-max", type=float, default=20.0)
    parser.add_argument("--gcs-point-valid-pos-weight-max", type=float, default=10.0)
    parser.add_argument("--gcs-edge-pos-weight-max", type=float, default=50.0)
    parser.add_argument("--gcs-aux-dice", type=float, default=0.5)
    parser.add_argument("--gcs-cost-point", type=float, default=5.0)
    parser.add_argument("--gcs-cost-curve", type=float, default=0.05)
    parser.add_argument("--gcs-cost-exist", type=float, default=0.1)
    parser.add_argument("--gcs-match-min-overlap", type=int, default=2, help="Minimum valid GT points for training Hungarian matching.")
    parser.add_argument("--gcs-match-max-x-dist", type=float, default=0.0, help="Optional training matcher mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--gcs-match-gate-px", type=float, default=160.0, help="Training matcher APE gate in pixels. 0 disables.")
    parser.add_argument("--gcs-eval-conf", type=float, default=0.2)
    parser.add_argument("--gcs-eval-ape-thr", type=float, default=20.0)
    parser.add_argument("--gcs-eval-match-gate-px", type=float, default=None, help="Strict validation APE gate in pixels. Defaults to --gcs-eval-ape-thr.")
    parser.add_argument("--gcs-eval-max-x-dist", type=float, default=0.0, help="Optional strict validation mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--gcs-eval-min-overlap", type=int, default=2, help="Minimum valid overlapping GT points for strict validation matching.")
    parser.add_argument("--gcs-eval-nms-dist-px", type=float, default=50.0, help="Optional validation lane NMS distance in pixels. 0 disables.")
    parser.add_argument(
        "--gcs-eval-point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold used when decoding fixed-y lanes for validation metrics.",
    )
    parser.add_argument("--gcs-eval-max-det", type=int, default=8)
    parser.add_argument(
        "--gcs-official-best",
        action="store_true",
        help="Every --gcs-official-interval epochs, run a TuSimple official-val sweep and preserve weights/official_best.pt.",
    )
    parser.add_argument(
        "--gcs-allow-internal-best",
        action="store_true",
        help="Allow ordered_slot debug training without --gcs-official-best. Formal runs should not use this.",
    )
    parser.add_argument(
        "--gcs-official-interval",
        type=int,
        default=5,
        help="Epoch interval for --gcs-official-best. The final/early-stop epoch is evaluated as well.",
    )
    parser.add_argument(
        "--gcs-official-archive-root",
        default=str(ROOT / "archive"),
        help="Path to archive/ or archive/TUSimple for training-time TuSimple official-val selection.",
    )
    parser.add_argument(
        "--gcs-official-gt-json",
        default=None,
        help="Explicit canonical 363-image official-val GT json-lines file for training-time official selection.",
    )
    parser.add_argument(
        "--gcs-official-allow-noncanonical-gt",
        action="store_true",
        help="Allow non-363 official-val GT only for diagnostics; summary marks it incomparable with E1/spurious.",
    )
    parser.add_argument("--gcs-official-max-images", type=int, default=0, help="Limit official-val images per hook. 0 means all.")
    parser.add_argument("--gcs-official-warmup", type=int, default=5, help="Warmup forwards for each training-time official sweep.")
    parser.add_argument("--gcs-official-confs", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--gcs-official-point-valid-thrs", nargs="+", type=float, default=[0.45, 0.5])
    parser.add_argument("--gcs-official-nms-dist-pxs", nargs="+", type=float, default=[0.0, 18.0, 30.0, 50.0])
    parser.add_argument("--gcs-official-max-dets", nargs="+", type=int, default=[5, 6, 8])
    parser.add_argument("--gcs-official-min-points", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument(
        "--gcs-official-count-modes",
        nargs="+",
        choices=("score_sum", "count_logits"),
        default=["score_sum"],
        help="Count source modes for training-time query official-val sweeps.",
    )
    official_valid_group = parser.add_mutually_exclusive_group()
    official_valid_group.add_argument(
        "--gcs-official-valid-before-maxdet",
        dest="gcs_official_valid_before_maxdet",
        action="store_true",
        default=None,
        help="During training-time official sweeps, filter point-valid/min_points failures before max_det truncation.",
    )
    official_valid_group.add_argument(
        "--no-gcs-official-valid-before-maxdet",
        dest="gcs_official_valid_before_maxdet",
        action="store_false",
        default=None,
        help="Disable valid-before-maxdet during training-time official sweeps when overriding a resumed run.",
    )
    parser.add_argument("--gcs-official-score-fp-weight", type=float, default=0.02)
    parser.add_argument("--gcs-official-score-fn-weight", type=float, default=0.02)
    parser.add_argument(
        "--gcs-official-half",
        action="store_true",
        help="Use FP16 inference during training-time official-val sweeps.",
    )
    parser.add_argument(
        "--gcs-lane-count-balanced",
        action="store_true",
        default=False,
        help="Enable lane-count-balanced sampling. Disabled by default for fair query/ordered_slot comparison.",
    )
    parser.add_argument(
        "--gcs-lane-count-balance-power",
        type=float,
        default=1.0,
        help="Exponent for lane-count balancing. With min-group smoothing, 1.0 strongly balances 2/3/4/5-lane modes.",
    )
    parser.add_argument(
        "--gcs-lane-count-min-group",
        type=int,
        default=50,
        help="Minimum group size used when balancing lane counts, preventing tiny groups from dominating an epoch.",
    )
    parser.add_argument(
        "--gcs-hard-sampling",
        action="store_true",
        help="Use train-only weighted hard sampling for 0601 and short-visible GT3/GT4/GT5 samples.",
    )
    parser.add_argument(
        "--gcs-hard-date-0601-weight",
        type=float,
        default=2.0,
        help="Hard-sampling multiplier for samples whose TuSimple date is 0601.",
    )
    parser.add_argument(
        "--gcs-hard-gt4-le10-weight",
        type=float,
        default=4.0,
        help="Hard-sampling multiplier for GT4 samples with min_visible_points <= --gcs-hard-visible-thr.",
    )
    parser.add_argument(
        "--gcs-hard-gt5-le10-weight",
        type=float,
        default=3.0,
        help="Hard-sampling multiplier for GT5 samples with min_visible_points <= --gcs-hard-visible-thr.",
    )
    parser.add_argument(
        "--gcs-hard-gt3-le20-weight",
        type=float,
        default=1.5,
        help="Hard-sampling multiplier for GT3 samples with min_visible_points <= --gcs-hard-gt3-visible-thr.",
    )
    parser.add_argument(
        "--gcs-hard-0313-2-gt4-le10-weight",
        type=float,
        default=4.0,
        help="Extra hard-sampling multiplier for 0313-2 GT4 samples with min_visible_points <= --gcs-hard-visible-thr.",
    )
    parser.add_argument(
        "--gcs-hard-visible-thr",
        type=int,
        default=10,
        help="Visible-point threshold for GT4/GT5 hard-sampling groups.",
    )
    parser.add_argument(
        "--gcs-hard-gt3-visible-thr",
        type=int,
        default=20,
        help="Visible-point threshold for GT3 hard-sampling group.",
    )
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume from the latest run when used alone, or from an explicit checkpoint path.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args(argv)


def parse_pretrained(value: str) -> str | bool:
    """Convert CLI pretrained values to the form expected by Ultralytics."""
    normalized = str(value).strip().lower()
    if normalized in {"", "false", "none", "no", "0"}:
        return False
    return value


def resolve_project(value: str) -> str:
    """Keep run outputs under the project root when a relative project path is passed."""
    path = Path(value)
    return str(path if path.is_absolute() else (ROOT / path).resolve())


def main() -> None:
    args = maybe_switch_ordered_slot_model(parse_args())
    defaults = dataset_defaults(args.dataset)
    gcs_imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    model_path = args.model

    overrides = {
        "task": "gcs_lane",
        "model": model_path,
        "data": args.data or str(defaults["data"]),
        "pretrained": parse_pretrained(args.pretrained),
        "imgsz": trainer_imgsz(gcs_imgsz),
        "gcs_imgsz": list(gcs_imgsz),
        "epochs": args.epochs,
        "batch": args.batch,
        "nbs": args.nbs if args.nbs > 0 else args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": resolve_project(args.project),
        "name": args.name or f"gcs_yolo_lane_s_{args.dataset}",
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": args.cos_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "amp": not args.no_amp,
        "patience": args.patience,
        "fraction": args.fraction,
        "scale": args.scale,
        "erasing": args.erasing,
        "mosaic": args.mosaic,
        "val": not args.no_val,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "train_images": args.train_images,
        "train_gcs_labels": args.train_gcs_labels,
        "val_images": args.val_images,
        "val_gcs_labels": args.val_gcs_labels,
        "gcs_mode": args.gcs_mode,
        "gcs_num_slots": args.gcs_num_slots,
        "gcs_min_lanes": args.gcs_min_lanes,
        "gcs_max_lanes": args.gcs_max_lanes,
        "gcs_count_classes": args.gcs_count_classes,
        "gcs_contiguity_policy": args.gcs_contiguity_policy,
        "gcs_exist": args.gcs_exist,
        "gcs_point": args.gcs_point,
        "gcs_point_valid": args.gcs_point_valid,
        "gcs_smooth": args.gcs_smooth,
        "gcs_curve": args.gcs_curve,
        "gcs_mask": args.gcs_mask,
        "gcs_edge": args.gcs_edge,
        "gcs_count": args.gcs_count,
        "gcs_count_under5": args.gcs_count_under5,
        "gcs_count_under5_min_lanes": args.gcs_count_under5_min_lanes,
        "gcs_count_boundary": args.gcs_count_boundary,
        "gcs_count_boundary_gt4_weight": args.gcs_count_boundary_gt4_weight,
        "gcs_count_boundary_gt5_weight": args.gcs_count_boundary_gt5_weight,
        "gcs_count_boundary_gt5_under_weight": args.gcs_count_boundary_gt5_under_weight,
        "gcs_count_boundary_margin34": args.gcs_count_boundary_margin34,
        "gcs_count_boundary_margin45": args.gcs_count_boundary_margin45,
        "gcs_query_count_ce": args.gcs_query_count_ce,
        "gcs_query_count_min_lanes": args.gcs_query_count_min_lanes,
        "gcs_query_count_max_lanes": args.gcs_query_count_max_lanes,
        "gcs_count_ce": args.gcs_count_ce,
        "gcs_interval": args.gcs_interval,
        "gcs_order": args.gcs_order,
        "gcs_gt_bottom_order": args.gcs_gt_bottom_order,
        "gcs_decoded_bottom_order": args.gcs_decoded_bottom_order,
        "gcs_slot_gt_bottom_x": args.gcs_slot_gt_bottom_x,
        "gcs_slot_gt_bottom_x_beta": args.gcs_slot_gt_bottom_x_beta,
        "gcs_slot_gt_bottom_x_detach_interval": args.gcs_slot_gt_bottom_x_detach_interval,
        "gcs_slot_gt_bottom_x_soft": args.gcs_slot_gt_bottom_x_soft,
        "gcs_slot_gt_bottom_x_soft_tau": args.gcs_slot_gt_bottom_x_soft_tau,
        "gcs_slot_gt_bottom_x_soft_beta": args.gcs_slot_gt_bottom_x_soft_beta,
        "gcs_slot_start_index_l1": args.gcs_slot_start_index_l1,
        "gcs_slot_start_index_l1_beta": args.gcs_slot_start_index_l1_beta,
        "gcs_allow_disable_order_loss": args.gcs_allow_disable_order_loss,
        "gcs_slot_exist_w4": args.gcs_slot_exist_w4,
        "gcs_slot_exist_w5": args.gcs_slot_exist_w5,
        "gcs_min_interval_points": args.gcs_min_interval_points,
        "gcs_order_margin_px": args.gcs_order_margin_px,
        "gcs_bottom_order_margin_px": args.gcs_bottom_order_margin_px,
        "gcs_ordered_point_loss": args.gcs_ordered_point_loss,
        "gcs_point_y_weight": args.gcs_point_y_weight,
        "gcs_point_x_only": args.gcs_point_x_only,
        "gcs_pixel_smoothl1_beta": args.gcs_pixel_smoothl1_beta,
        "gcs_spurious_neg": args.gcs_spurious_neg,
        "gcs_spurious_neg_weight": args.gcs_spurious_neg_weight,
        "gcs_spurious_gt3_weight": args.gcs_spurious_gt3_weight,
        "gcs_spurious_gt4_weight": args.gcs_spurious_gt4_weight,
        "gcs_spurious_gt5_weight": args.gcs_spurious_gt5_weight,
        "gcs_spurious_disable_gt5": args.gcs_spurious_disable_gt5,
        "gcs_spurious_max_points": args.gcs_spurious_max_points,
        "gcs_spurious_close_px": args.gcs_spurious_close_px,
        "gcs_spurious_min_overlap": args.gcs_spurious_min_overlap,
        "gcs_spurious_gt_protect": args.gcs_spurious_gt_protect,
        "gcs_spurious_gt_protect_px": args.gcs_spurious_gt_protect_px,
        "gcs_spurious_gt_protect_min_overlap": args.gcs_spurious_gt_protect_min_overlap,
        "gcs_spurious_gt_protect_margin_px": args.gcs_spurious_gt_protect_margin_px,
        "gcs_spurious_gt_protect_mode": args.gcs_spurious_gt_protect_mode,
        "gcs_gt5_short_visible_thr": args.gcs_gt5_short_visible_thr,
        "gcs_gt5_short_point_valid_weight": args.gcs_gt5_short_point_valid_weight,
        "gcs_short_geom": args.gcs_short_geom,
        "gcs_short_geom_visible_thr": args.gcs_short_geom_visible_thr,
        "gcs_short_geom_gt4_weight": args.gcs_short_geom_gt4_weight,
        "gcs_short_geom_gt5_weight": args.gcs_short_geom_gt5_weight,
        "gcs_short_geom_max_weight": args.gcs_short_geom_max_weight,
        "gcs_short_geom_curve": args.gcs_short_geom_curve,
        "gcs_short_survival": args.gcs_short_survival,
        "gcs_short_survival_visible_thr": args.gcs_short_survival_visible_thr,
        "gcs_short_survival_gt4_weight": args.gcs_short_survival_gt4_weight,
        "gcs_short_survival_gt5_weight": args.gcs_short_survival_gt5_weight,
        "gcs_short_survival_exist_weight": args.gcs_short_survival_exist_weight,
        "gcs_short_survival_valid_weight": args.gcs_short_survival_valid_weight,
        "gcs_boundary_pseudo_neg": args.gcs_boundary_pseudo_neg,
        "gcs_boundary_pseudo_visible_thr": args.gcs_boundary_pseudo_visible_thr,
        "gcs_boundary_pseudo_dist_thr": args.gcs_boundary_pseudo_dist_thr,
        "gcs_boundary_pseudo_valid_thr": args.gcs_boundary_pseudo_valid_thr,
        "gcs_boundary_pseudo_min_valid": args.gcs_boundary_pseudo_min_valid,
        "gcs_boundary_pseudo_gt_count": args.gcs_boundary_pseudo_gt_count,
        "gcs_boundary_pseudo_score_thr": args.gcs_boundary_pseudo_score_thr,
        "gcs_boundary_pseudo_envelope_margin_px": args.gcs_boundary_pseudo_envelope_margin_px,
        "gcs_boundary_pseudo_envelope_ratio_thr": args.gcs_boundary_pseudo_envelope_ratio_thr,
        "gcs_exist_pos_weight": args.gcs_exist_pos_weight,
        "gcs_exist_focal_gamma": args.gcs_exist_focal_gamma,
        "gcs_exist_focal_alpha": args.gcs_exist_focal_alpha,
        "gcs_exist_quality_alpha": args.gcs_exist_quality_alpha,
        "gcs_exist_quality_mode": args.gcs_exist_quality_mode,
        "gcs_exist_quality_tau": args.gcs_exist_quality_tau,
        "gcs_exist_quality_floor": args.gcs_exist_quality_floor,
        "gcs_exist_quality_pos_px": args.gcs_exist_quality_pos_px,
        "gcs_exist_quality_neg_px": args.gcs_exist_quality_neg_px,
        "gcs_point_valid_pos_weight_max": args.gcs_point_valid_pos_weight_max,
        "gcs_mask_pos_weight_max": args.gcs_mask_pos_weight_max,
        "gcs_edge_pos_weight_max": args.gcs_edge_pos_weight_max,
        "gcs_aux_dice": args.gcs_aux_dice,
        "gcs_cost_point": args.gcs_cost_point,
        "gcs_cost_curve": args.gcs_cost_curve,
        "gcs_cost_exist": args.gcs_cost_exist,
        "gcs_match_min_overlap": args.gcs_match_min_overlap,
        "gcs_match_max_x_dist": args.gcs_match_max_x_dist,
        "gcs_match_gate_px": args.gcs_match_gate_px,
        "gcs_eval_conf": args.gcs_eval_conf,
        "gcs_eval_ape_thr": args.gcs_eval_ape_thr,
        "gcs_eval_match_gate_px": args.gcs_eval_match_gate_px,
        "gcs_eval_max_x_dist": args.gcs_eval_max_x_dist,
        "gcs_eval_min_overlap": args.gcs_eval_min_overlap,
        "gcs_eval_nms_dist_px": args.gcs_eval_nms_dist_px,
        "gcs_eval_point_valid_thr": args.gcs_eval_point_valid_thr,
        "gcs_eval_max_det": args.gcs_eval_max_det,
        "gcs_official_best": args.gcs_official_best,
        "gcs_allow_internal_best": args.gcs_allow_internal_best,
        "gcs_official_interval": args.gcs_official_interval,
        "gcs_official_archive_root": args.gcs_official_archive_root,
        "gcs_official_gt_json": args.gcs_official_gt_json,
        "gcs_official_allow_noncanonical_gt": args.gcs_official_allow_noncanonical_gt,
        "gcs_official_max_images": args.gcs_official_max_images,
        "gcs_official_warmup": args.gcs_official_warmup,
        "gcs_official_confs": args.gcs_official_confs,
        "gcs_official_point_valid_thrs": args.gcs_official_point_valid_thrs,
        "gcs_official_nms_dist_pxs": args.gcs_official_nms_dist_pxs,
        "gcs_official_max_dets": args.gcs_official_max_dets,
        "gcs_official_min_points": args.gcs_official_min_points,
        "gcs_official_count_modes": args.gcs_official_count_modes,
        "gcs_official_score_fp_weight": args.gcs_official_score_fp_weight,
        "gcs_official_score_fn_weight": args.gcs_official_score_fn_weight,
        "gcs_official_half": args.gcs_official_half,
        "gcs_lane_count_balanced": args.gcs_lane_count_balanced,
        "gcs_lane_count_balance_power": args.gcs_lane_count_balance_power,
        "gcs_lane_count_min_group": args.gcs_lane_count_min_group,
        "gcs_hard_sampling": args.gcs_hard_sampling,
        "gcs_hard_date_0601_weight": args.gcs_hard_date_0601_weight,
        "gcs_hard_gt4_le10_weight": args.gcs_hard_gt4_le10_weight,
        "gcs_hard_gt5_le10_weight": args.gcs_hard_gt5_le10_weight,
        "gcs_hard_gt3_le20_weight": args.gcs_hard_gt3_le20_weight,
        "gcs_hard_0313_2_gt4_le10_weight": args.gcs_hard_0313_2_gt4_le10_weight,
        "gcs_hard_visible_thr": args.gcs_hard_visible_thr,
        "gcs_hard_gt3_visible_thr": args.gcs_hard_gt3_visible_thr,
    }
    if args.gcs_official_valid_before_maxdet is not None:
        overrides["gcs_official_valid_before_maxdet"] = args.gcs_official_valid_before_maxdet

    print(f"GCS input shape: {shape_str(gcs_imgsz)} (W x H), stored as H,W={gcs_imgsz}")
    trainer = GCSLaneTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
