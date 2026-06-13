from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.models.yolo.gcs_lane.train import (
    GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT,
    GCS_MAINLINE_COUNT_CLS_WEIGHTS,
    GCS_MAINLINE_COUNT_BOUNDARY_GAIN,
    GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT,
    GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING,
    GCS_MAINLINE_COUNT_SUM_GAIN,
    GCS_MAINLINE_GROUP_SAMPLER_RATIOS,
    GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT,
    GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT,
    GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY,
    GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR,
    GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT,
    GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_MIN_POINTS,
    GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_THR,
    GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT,
    GCS_MAINLINE_QUALITY_HARD_NEGATIVE_FROM_HEAD,
    GCS_MAINLINE_QUALITY_GAIN,
    GCS_MAINLINE_QUALITY_GT5_EDGE_FLOOR,
    GCS_MAINLINE_QUALITY_NEG_WEIGHT,
    GCSLaneTrainer,
)
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str, trainer_imgsz
from ultralytics.utils.gcs_postprocess import GCS_DEFAULT_MAX_DET


DEFAULT_MODEL = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q12.yaml"


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
        fixed_root = ROOT / "datasets" / "tusimple_fixed_y_960x544"
        fixed_data = ROOT / "data" / "tusimple_gcs_fixed_y_960x544.yaml"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GCS-YOLO-Lane on structured lane labels.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
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
        "--translate",
        type=float,
        default=0.0,
        help="GCS-safe random translation gain as a fraction of image width/height.",
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
    parser.add_argument(
        "--gcs-train-include-val",
        action="store_true",
        help="Append the configured val images/labels to the training loader for final train+val fitting. Use with --no-val.",
    )
    parser.add_argument("--gcs-exist", type=float, default=1.0, help="Quality-aware lane existence/confidence loss gain.")
    parser.add_argument("--gcs-point", type=float, default=5.0)
    parser.add_argument("--gcs-point-valid", type=float, default=0.5)
    parser.add_argument(
        "--gcs-point-invalid-x",
        type=float,
        default=0.05,
        help="Relative pseudo-x penalty inside point loss for matched invisible anchors weighted by predicted point-valid probability.",
    )
    parser.add_argument(
        "--gcs-line-iou",
        type=float,
        default=0.3,
        help="Whole-lane LineIoU loss gain. 0 disables the LineIoU shape term.",
    )
    parser.add_argument(
        "--gcs-line-iou-width-px",
        type=float,
        default=15.0,
        help="Half-width in pixels used to expand lane points into horizontal strips for LineIoU.",
    )
    parser.add_argument("--gcs-count-cls", type=float, default=0.3, help="Explicit Count Head count=2/3/4/5 CE loss gain.")
    parser.add_argument(
        "--gcs-count-sum",
        "--count-sum-loss-weight",
        dest="gcs_count_sum",
        type=float,
        default=GCS_MAINLINE_COUNT_SUM_GAIN,
        help="Exist-count consistency loss gain for sum(sigmoid(pred_logits)) ~= GT lane count.",
    )
    parser.add_argument(
        "--gcs-count-sum-normalize",
        "--count-sum-loss-normalize",
        dest="gcs_count_sum_normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize count_sum_loss by GT lane count.",
    )
    parser.add_argument("--gcs-quality", type=float, default=GCS_MAINLINE_QUALITY_GAIN, help="Lane-level Quality Head BCE loss gain.")
    parser.add_argument("--gcs-quality-dist-thr-px", type=float, default=20.0, help="Pixel threshold for quality target point-inlier score.")
    parser.add_argument("--gcs-quality-neg-weight", type=float, default=GCS_MAINLINE_QUALITY_NEG_WEIGHT, help="Relative weight for unmatched-query quality negatives.")
    parser.add_argument(
        "--gcs-quality-gt5-edge-floor",
        type=float,
        default=GCS_MAINLINE_QUALITY_GT5_EDGE_FLOOR,
        help="Minimum Quality Head target for matched left/right GT5 edge lanes. 0 disables.",
    )
    parser.add_argument("--gcs-quality-hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--gcs-quality-duplicate-negative-weight", type=float, default=1.5)
    parser.add_argument(
        "--gcs-quality-hard-negative-from-head",
        action=argparse.BooleanOptionalAction,
        default=GCS_MAINLINE_QUALITY_HARD_NEGATIVE_FROM_HEAD,
        help="Also mine Quality Head hard negatives directly from high pred_quality_logits on unmatched queries.",
    )
    parser.add_argument(
        "--gcs-hard-negative-visible-segment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mine unmatched hard negatives with visible-segment evidence instead of all-anchor point-valid mean.",
    )
    parser.add_argument(
        "--gcs-hard-negative-visible-thr",
        type=float,
        default=0.5,
        help="Point-valid probability threshold for visible-segment hard-negative mining.",
    )
    parser.add_argument(
        "--gcs-hard-negative-visible-support-points",
        type=float,
        default=12.0,
        help="Visible-segment length saturation for hard-negative mining support.",
    )
    parser.add_argument("--gcs-count-head-warmup-epochs", type=float, default=5.0, help="Linearly ramp Count Head loss over this many epochs. 0 disables warmup.")
    parser.add_argument("--gcs-count-min-gt-points", type=int, default=1, help="Minimum visible anchors for a GT lane to count in Count Head targets.")
    parser.add_argument("--gcs-count-cls-w2", type=float, default=GCS_MAINLINE_COUNT_CLS_WEIGHTS[0])
    parser.add_argument("--gcs-count-cls-w3", type=float, default=GCS_MAINLINE_COUNT_CLS_WEIGHTS[1])
    parser.add_argument("--gcs-count-cls-w4", type=float, default=GCS_MAINLINE_COUNT_CLS_WEIGHTS[2])
    parser.add_argument(
        "--gcs-count-cls-w5",
        type=float,
        default=GCS_MAINLINE_COUNT_CLS_WEIGHTS[3],
        help="Count Head CE class weight for 5-lane images. Current default balances GT5 recall and count generalization; sweep 1.6/1.8/2.0 if needed.",
    )
    parser.add_argument(
        "--gcs-count-boundary",
        type=float,
        default=GCS_MAINLINE_COUNT_BOUNDARY_GAIN,
        help="Count Boundary BCE gain for count>=4 and count>=5 calibration inside count_cls_loss.",
    )
    parser.add_argument(
        "--gcs-count-boundary-label-smoothing",
        type=float,
        default=GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING,
        help="Label smoothing for Count Boundary count>=4/count>=5 targets.",
    )
    parser.add_argument(
        "--gcs-count-boundary-gt5-pos-weight",
        type=float,
        default=GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT,
        help="Extra Count Boundary BCE weight for count>=5 positive targets. 1 disables.",
    )
    parser.add_argument(
        "--gcs-count-adjacent-margin",
        type=float,
        default=0.2,
        help="Target-vs-neighbor Count Head logit margin for adjacent count classes.",
    )
    parser.add_argument(
        "--gcs-count-adjacent-margin-gain",
        type=float,
        default=0.0,
        help="Adjacent count margin gain inside count_cls_loss. 0 disables the experimental margin term.",
    )
    parser.add_argument(
        "--gcs-count-adjacent-margin-gt45-weight",
        type=float,
        default=1.0,
        help="Extra adjacent-margin sample weight for GT4/GT5 images. 1 disables.",
    )
    parser.add_argument("--gcs-exist-pos-weight", type=float, default=1.0)
    parser.add_argument("--gcs-exist-focal-gamma", type=float, default=2.0, help="Quality focal gamma for existence BCE. 0 disables focal weighting.")
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
    parser.add_argument(
        "--gcs-exist-margin",
        type=float,
        default=0.5,
        help="Relative exist probability margin loss gain for matched positives and unmatched negatives.",
    )
    parser.add_argument(
        "--gcs-exist-pos-margin",
        type=float,
        default=0.55,
        help="Matched positive queries with high quality are pushed above this exist probability.",
    )
    parser.add_argument(
        "--gcs-exist-neg-margin",
        type=float,
        default=0.20,
        help="Unmatched queries are pushed below this exist probability.",
    )
    parser.add_argument(
        "--gcs-exist-quality-alpha",
        type=float,
        default=1.0,
        help="Blend factor for quality-aware existence targets. 1 uses pure geometry quality, 0 restores hard matched labels.",
    )
    parser.add_argument(
        "--gcs-exist-quality-lane-iou-alpha",
        type=float,
        default=1.0,
        help="Legacy compatibility option; matched exist quality now always uses LineIoU, point error, and visibility.",
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
        help="Optional matched exist-target floor above the mandatory 0.5 lower bound.",
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
    parser.add_argument("--gcs-point-valid-pos-weight-max", type=float, default=10.0)
    parser.add_argument("--gcs-point-valid-unmatched-weight", type=float, default=0.35)
    parser.add_argument("--gcs-point-valid-hard-negative-weight", type=float, default=1.25)
    parser.add_argument("--gcs-point-valid-duplicate-negative-weight", type=float, default=1.5)
    parser.add_argument(
        "--gcs-point-valid-gt5-pos-weight",
        type=float,
        default=GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT,
        help=(
            "Extra multiplier for positive point-valid anchors on images with at least 5 GT lanes. 1 disables. "
            "Current default avoids over-concentrating GT5 while preserving fifth-lane visibility pressure."
        ),
    )
    parser.add_argument(
        "--gcs-gt5-edge-loss-weight",
        "--gt5-edge-loss-weight",
        dest="gcs_gt5_edge_loss_weight",
        type=float,
        default=GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT,
        help="Multiplier for matched left/right edge lanes on images with at least 4 GT lanes. 1 disables.",
    )
    parser.add_argument(
        "--gcs-candidate-gt5-edge-weight",
        type=float,
        default=GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT,
        help="Matched left/right GT5 edge-query/lane loss multiplier. 1 disables.",
    )
    parser.add_argument(
        "--gcs-point-valid-gt5-edge-continuity",
        type=float,
        default=GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY,
        help="Point-valid continuity gain for matched left/right edge lanes on GT>=5 images. 0 disables.",
    )
    parser.add_argument(
        "--gcs-point-valid-gt5-edge-continuity-thr",
        type=float,
        default=GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR,
        help="Minimum adjacent point-valid probability used by the GT>=5 edge continuity penalty.",
    )
    parser.add_argument(
        "--gcs-point-valid-gt5-edge-segment",
        type=float,
        default=GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT,
        help="GT5 edge-lane longest-visible-segment support loss gain. 0 disables.",
    )
    parser.add_argument(
        "--gcs-point-valid-gt5-edge-segment-thr",
        type=float,
        default=GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_THR,
        help="Point-valid probability floor for the GT5 edge visible-segment support loss.",
    )
    parser.add_argument(
        "--gcs-point-valid-gt5-edge-segment-min-points",
        type=int,
        default=GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_MIN_POINTS,
        help="Minimum GT-visible anchors required for the GT5 edge segment support loss.",
    )
    parser.add_argument(
        "--gcs-hard-loss-file",
        default="",
        help="Optional txt/json manifest of hard images for loss-only hard-edge weighting. Does not affect sampling.",
    )
    parser.add_argument(
        "--gcs-hard-loss-lane-counts",
        default="",
        help="Optional GT lane-count filter for --gcs-hard-loss-file, e.g. '5' or '4,5'. Empty keeps all manifest hits.",
    )
    parser.add_argument(
        "--gcs-hard-edge-loss-weight-by-count",
        default="4:1.15,5:1.6",
        help="Per-GT-count hard-edge loss multipliers for manifest hits, e.g. '4:1.15,5:1.6'.",
    )
    parser.add_argument(
        "--gcs-hard-edge-loss-terms",
        default="exist,point,point_valid,line_iou",
        help="Comma/space separated terms that receive hard-edge weighting: exist, point, point_valid, line_iou, quality.",
    )
    parser.add_argument(
        "--gcs-hard-edge-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply hard loss multipliers only to matched left/right edge lanes.",
    )
    parser.add_argument(
        "--gcs-point-valid-neg",
        type=float,
        default=0.25,
        help="Relative margin penalty for over-confident invisible anchors.",
    )
    parser.add_argument(
        "--gcs-point-valid-neg-thr",
        type=float,
        default=0.20,
        help="Invisible anchors are penalized when point-valid probability exceeds this threshold.",
    )
    parser.add_argument("--gcs-cost-point", type=float, default=5.0)
    parser.add_argument("--gcs-cost-exist", type=float, default=0.1)
    parser.add_argument("--gcs-match-min-overlap", type=int, default=2, help="Minimum valid GT points for training Hungarian matching.")
    parser.add_argument("--gcs-match-max-x-dist", type=float, default=0.0, help="Optional training matcher mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--gcs-match-gate-px", type=float, default=160.0, help="Training matcher APE gate in pixels. 0 disables.")
    parser.add_argument("--gcs-eval-conf", type=float, default=0.2)
    parser.add_argument("--gcs-eval-ape-thr", type=float, default=20.0)
    parser.add_argument("--gcs-eval-match-gate-px", type=float, default=None, help="Strict validation APE gate in pixels. Defaults to --gcs-eval-ape-thr.")
    parser.add_argument("--gcs-eval-max-x-dist", type=float, default=0.0, help="Optional strict validation mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--gcs-eval-min-overlap", type=int, default=6, help="Minimum overlapping visible anchors for strict validation matching.")
    parser.add_argument("--gcs-eval-min-points", type=int, default=6, help="Minimum decoded visible anchors required to keep a validation prediction.")
    parser.add_argument(
        "--gcs-eval-min-gt-cover-ratio",
        type=float,
        default=0.3,
        help="Minimum GT visible-anchor coverage ratio required for strict validation matching.",
    )
    parser.add_argument(
        "--gcs-eval-min-pred-cover-ratio",
        type=float,
        default=0.3,
        help="Minimum predicted visible-anchor coverage ratio required for strict validation matching.",
    )
    parser.add_argument("--gcs-eval-nms-dist-px", type=float, default=18.0, help="Validation lane NMS distance in pixels. 0 disables.")
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument("--gcs-use-count-head-decode", dest="gcs_use_count_head_decode", action="store_true", help="Use explicit Count Head K during validation/inference decode.")
    decode_group.add_argument("--no-gcs-use-count-head-decode", dest="gcs_use_count_head_decode", action="store_false", help="Disable Count Head K and use max-det rank selection.")
    parser.set_defaults(gcs_use_count_head_decode=True)
    parser.add_argument("--gcs-count-head-temp", type=float, default=1.0)
    parser.add_argument("--gcs-decode-candidate-conf", type=float, default=0.05)
    parser.add_argument("--gcs-decode-candidate-point-valid-thr", type=float, default=0.20)
    parser.add_argument("--gcs-decode-candidate-min-points", type=int, default=5)
    rescue_candidate_group = parser.add_mutually_exclusive_group()
    rescue_candidate_group.add_argument("--gcs-enable-rescue-candidate-pool", dest="gcs_enable_rescue_candidate_pool", action="store_true", help="Use weaker real-query candidates only when Count Head K exceeds the normal candidate pool.")
    rescue_candidate_group.add_argument("--no-gcs-enable-rescue-candidate-pool", dest="gcs_enable_rescue_candidate_pool", action="store_false", help="Disable the weaker rescue candidate pool.")
    parser.set_defaults(gcs_enable_rescue_candidate_pool=True)
    parser.add_argument("--gcs-decode-rescue-candidate-conf", type=float, default=0.005)
    parser.add_argument("--gcs-decode-rescue-candidate-point-valid-thr", type=float, default=0.08)
    parser.add_argument("--gcs-decode-rescue-candidate-min-points", type=int, default=4)
    parser.add_argument("--gcs-decode-final-min-points", type=int, default=6)
    parser.add_argument("--gcs-decode-fifth-min-points", type=int, default=5)
    parser.add_argument("--gcs-line-nms-min-overlap", type=int, default=6)
    parser.add_argument("--gcs-line-nms-rescue-dist-px", type=float, default=30.0)
    parser.add_argument("--gcs-quality-rescue-5th", action=argparse.BooleanOptionalAction, default=True, help="Enable quality-gated fifth-lane rescue when pred_quality_logits are present.")
    parser.add_argument("--gcs-quality-rescue-count5-thr", type=float, default=0.70)
    parser.add_argument("--gcs-quality-rescue-conf-thr", type=float, default=0.03)
    parser.add_argument("--gcs-quality-rescue-mean-valid-thr", type=float, default=0.45)
    parser.add_argument("--gcs-quality-rescue-quality-thr", type=float, default=0.55)
    parser.add_argument("--gcs-quality-rescue-min-points", type=int, default=5)
    parser.add_argument("--gcs-quality-rescue-dist-px", type=float, default=24.0)
    parser.add_argument("--gcs-last-lane-rescue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gcs-last-lane-rescue-min-policy-count", type=int, default=4)
    parser.add_argument("--gcs-last-lane-rescue-conf-thr", type=float, default=None)
    parser.add_argument("--gcs-last-lane-rescue-point-valid-thr", type=float, default=0.08)
    parser.add_argument("--gcs-last-lane-rescue-min-points", type=int, default=4)
    parser.add_argument("--gcs-last-lane-rescue-mean-valid-thr", type=float, default=0.40)
    parser.add_argument("--gcs-last-lane-rescue-quality-thr", type=float, default=0.50)
    parser.add_argument("--gcs-last-lane-rescue-dist-px", type=float, default=24.0)
    parser.add_argument("--gcs-edge-last-lane-rescue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gcs-edge-rescue-conf-thr", type=float, default=0.02)
    parser.add_argument("--gcs-edge-rescue-point-valid-thr", type=float, default=0.06)
    parser.add_argument("--gcs-edge-rescue-min-points", type=int, default=4)
    parser.add_argument("--gcs-edge-rescue-mean-valid-thr", type=float, default=0.35)
    parser.add_argument("--gcs-edge-rescue-quality-thr", type=float, default=0.45)
    parser.add_argument("--gcs-edge-rescue-outside-gap-px", type=float, default=28.0)
    parser.add_argument("--gcs-edge-rescue-dist-px", type=float, default=24.0)
    parser.add_argument("--gcs-edge-rescue-min-policy-count", type=int, default=4)
    parser.add_argument("--gcs-edge-count4-to5-upgrade", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gcs-edge-count4-to5-prob-margin", type=float, default=0.20)
    parser.add_argument(
        "--gcs-soft-count-decision",
        "--soft-count-decision",
        dest="gcs_soft_count_decision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When Count Head probabilities are close, choose K by candidate quality instead of hard argmax.",
    )
    parser.add_argument("--gcs-soft-count-prob-margin", "--soft-count-prob-margin", dest="gcs_soft_count_prob_margin", type=float, default=0.08)
    parser.add_argument("--gcs-soft-count-quality-weight", "--soft-count-quality-weight", dest="gcs_soft_count_quality_weight", type=float, default=1.0)
    parser.add_argument("--gcs-soft-count-prior-weight", "--soft-count-prior-weight", dest="gcs_soft_count_prior_weight", type=float, default=0.5)
    parser.add_argument("--gcs-soft-count-duplicate-penalty", "--soft-count-duplicate-penalty", dest="gcs_soft_count_duplicate_penalty", type=float, default=1.0)
    parser.add_argument("--gcs-soft-count-invalid-penalty", "--soft-count-invalid-penalty", dest="gcs_soft_count_invalid_penalty", type=float, default=1.0)
    parser.add_argument(
        "--gcs-eval-point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold used when decoding fixed-y lanes for validation metrics.",
    )
    parser.add_argument("--gcs-eval-max-det", type=int, default=GCS_DEFAULT_MAX_DET)
    parser.add_argument(
        "--gcs-sampler-mode",
        default="group_cycle",
        choices=("group_cycle", "weighted", "none"),
        help="Training sampler mode. group_cycle uses lane-count target ratios and cycles each group without replacement.",
    )
    parser.add_argument(
        "--gcs-group-sampler-ratios",
        default=GCS_MAINLINE_GROUP_SAMPLER_RATIOS,
        help=(
            "Target lane-count ratios for --gcs-sampler-mode group_cycle. "
            f"Mainline default: '{GCS_MAINLINE_GROUP_SAMPLER_RATIOS}'."
        ),
    )
    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument(
        "--gcs-lane-count-balanced",
        dest="gcs_lane_count_balanced",
        action="store_true",
        help="Use inverse-frequency sampling by GT lane count to expose rare 2/5-lane cases more often.",
    )
    balance_group.add_argument(
        "--no-gcs-lane-count-balanced",
        dest="gcs_lane_count_balanced",
        action="store_false",
        help="Disable lane-count balanced sampling.",
    )
    parser.set_defaults(gcs_lane_count_balanced=True)
    parser.add_argument(
        "--gcs-lane-count-balance-power",
        type=float,
        default=1.0,
        help="Exponent for lane-count balancing. With min-group smoothing, 1.0 strongly balances common 3/4/5-lane modes.",
    )
    parser.add_argument(
        "--gcs-lane-count-min-group",
        type=int,
        default=50,
        help="Minimum group size used when balancing lane counts, preventing tiny groups from dominating an epoch.",
    )
    hard_group = parser.add_mutually_exclusive_group()
    hard_group.add_argument(
        "--gcs-hard-sampling",
        dest="gcs_hard_sampling",
        nargs="?",
        const=True,
        type=str2bool,
        help="Boost hard lane-count groups in the GCS weighted sampler. Accepts '--gcs-hard-sampling' or '--gcs-hard-sampling True'.",
    )
    hard_group.add_argument(
        "--no-gcs-hard-sampling",
        dest="gcs_hard_sampling",
        action="store_false",
        help="Disable hard lane-count boosting while keeping optional lane-count balancing.",
    )
    parser.set_defaults(gcs_hard_sampling=False)
    parser.add_argument(
        "--gcs-hard-lane-counts",
        default="",
        help="Comma/space separated GT lane counts boosted by --gcs-hard-sampling in weighted legacy mode.",
    )
    parser.add_argument(
        "--gcs-hard-sampling-boost",
        type=float,
        default=1.5,
        help="Weight multiplier for samples whose GT lane count is in --gcs-hard-lane-counts.",
    )
    parser.add_argument(
        "--gcs-hard-sampling-boost-by-count",
        default="",
        help="Optional per-count final multipliers for legacy weighted sampling, e.g. '4:1.5,5:2.0'.",
    )
    parser.add_argument(
        "--gcs-hard-sample-file",
        default="",
        help="Optional txt/json manifest of hard image stems, paths, or raw_file ids to boost in the sampler.",
    )
    parser.add_argument(
        "--gcs-hard-sample-boost",
        type=float,
        default=2.0,
        help="Weight multiplier for samples matched by --gcs-hard-sample-file.",
    )
    parser.add_argument(
        "--gcs-gt5-oversample-weight",
        "--gt5-oversample-weight",
        dest="gcs_gt5_oversample_weight",
        type=float,
        default=GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT,
        help="Additional sampler multiplier/ratio boost for GT=5 training images. Mainline default 1 disables it.",
    )
    gt5_aug_group = parser.add_mutually_exclusive_group()
    gt5_aug_group.add_argument(
        "--gcs-gt5-extra-aug",
        dest="gcs_gt5_extra_aug",
        action="store_true",
        help="Enable extra image-only multi-view augmentation for samples with at least 5 GT lanes.",
    )
    gt5_aug_group.add_argument(
        "--no-gcs-gt5-extra-aug",
        dest="gcs_gt5_extra_aug",
        action="store_false",
        help="Disable extra image-only augmentation for >=5-lane samples.",
    )
    parser.set_defaults(gcs_gt5_extra_aug=True)
    parser.add_argument("--gcs-gt5-aug-min-lanes", type=int, default=5)
    parser.add_argument("--gcs-gt5-erasing", type=float, default=0.15)
    parser.add_argument("--gcs-gt5-blur", type=float, default=0.15)
    parser.add_argument("--gcs-gt5-noise", type=float, default=0.15)
    parser.add_argument("--gcs-gt5-shadow", type=float, default=0.20)
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save extra epoch checkpoint files every N epochs. Not required for --gcs-official-best.",
    )
    parser.add_argument(
        "--gcs-official-best",
        action="store_true",
        help=(
            "Maintain weights/official_best.pt using official Accuracy only in periodic TuSimple official val sweeps. "
            "Configured thresholds are diagnostic only; weights/best.pt remains ordinary val F1 best."
        ),
    )
    parser.add_argument(
        "--gcs-official-best-period",
        type=int,
        default=0,
        help="Epoch interval for official best sweeps. 0 uses --save-period once for legacy commands, otherwise defaults to 10.",
    )
    parser.add_argument(
        "--gcs-official-best-top-k",
        type=int,
        default=1,
        help="Preserve the top K official-val candidate checkpoints under weights/official_topk. 1 keeps legacy behavior.",
    )
    parser.add_argument("--gcs-official-best-gt-json", default="", help="Stratified official-val json-lines used for official_best.pt selection.")
    parser.add_argument("--gcs-official-best-archive-root", default="archive", help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--gcs-official-best-split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--gcs-official-best-confs", default="0.005 0.01 0.015 0.02 0.03 0.05 0.08 0.10")
    parser.add_argument("--gcs-official-best-point-valid-thrs", default="0.20 0.25 0.30 0.35")
    parser.add_argument("--gcs-official-best-nms-dist-pxs", default="18.0")
    parser.add_argument("--gcs-official-best-max-dets", default="5")
    parser.add_argument("--gcs-official-best-min-points", default="6")
    parser.add_argument("--gcs-official-best-rank-min-points", default="none 5:5")
    parser.add_argument("--gcs-official-best-last-lane-rescue-point-valid-thrs", default="0.08")
    parser.add_argument("--gcs-official-best-last-lane-rescue-min-points", default="4")
    parser.add_argument("--gcs-official-best-last-lane-rescue-mean-valid-thrs", default="0.40")
    parser.add_argument("--gcs-official-best-last-lane-rescue-quality-thrs", default="0.50")
    parser.add_argument("--gcs-official-best-last-lane-rescue-dist-pxs", default="24.0")
    parser.add_argument("--gcs-official-best-max-images", type=int, default=0)
    parser.add_argument("--gcs-official-best-warmup", type=int, default=0)
    parser.add_argument("--gcs-official-best-half", action="store_true")
    parser.add_argument("--gcs-official-best-score-fp-weight", type=float, default=0.02)
    parser.add_argument("--gcs-official-best-score-fn-weight", type=float, default=0.02)
    parser.add_argument("--gcs-official-best-count-acc3-weight", type=float, default=0.0)
    parser.add_argument("--gcs-official-best-count-acc4-weight", type=float, default=0.006)
    parser.add_argument("--gcs-official-best-count-acc5-weight", type=float, default=0.004)
    parser.add_argument("--gcs-official-best-rate-4-to-5-weight", type=float, default=0.004)
    parser.add_argument("--gcs-official-best-rate-3-to-5-weight", type=float, default=0.0025)
    parser.add_argument("--gcs-official-best-rate-4-to-3-weight", type=float, default=0.0015)
    parser.add_argument("--gcs-official-best-rate-3-to-4-weight", type=float, default=0.001)
    parser.add_argument("--gcs-official-best-rate-5-to-4-weight", type=float, default=0.0)
    parser.add_argument(
        "--gcs-official-best-min-count-acc3",
        type=float,
        default=-1.0,
        help="Soft diagnostic floor for GT=3 count accuracy. Use a negative value to disable.",
    )
    parser.add_argument(
        "--gcs-official-best-min-count-acc4",
        type=float,
        default=-1.0,
        help="Soft diagnostic floor for GT=4 count accuracy. Use a negative value to disable.",
    )
    parser.add_argument(
        "--gcs-official-best-min-count-acc5",
        type=float,
        default=-1.0,
        help="Legacy soft diagnostic floor for GT=5 count accuracy. Use a negative value to disable.",
    )
    parser.add_argument(
        "--gcs-official-best-min-gt5-output5-rate",
        type=float,
        default=0.80,
        help="Soft diagnostic floor for GT=5 images whose final decoded output has 5 lanes. Negative disables.",
    )
    parser.add_argument(
        "--gcs-official-best-max-gt5-count-head-under-rate",
        type=float,
        default=0.15,
        help="Soft diagnostic ceiling for GT=5 images where Count Head policy K is below 5. Negative disables.",
    )
    parser.add_argument(
        "--gcs-official-best-max-gt5-valid-points-fail-rate",
        type=float,
        default=0.10,
        help="Soft diagnostic ceiling for GT=5 images where Count Head K=5 but final output has fewer than 5 lanes. Use a negative value to disable.",
    )
    parser.add_argument(
        "--gcs-official-best-max-rate-3-to-4",
        type=float,
        default=-1.0,
        help="Soft diagnostic ceiling for GT=3 decoded as 4 lanes. Use a negative value to disable.",
    )
    parser.add_argument(
        "--gcs-official-best-max-rate-4-to-3",
        type=float,
        default=-1.0,
        help="Soft diagnostic ceiling for GT=4 decoded as 3 lanes. Use a negative value to disable.",
    )
    parser.add_argument(
        "--gcs-official-best-max-rate-4-to-5",
        type=float,
        default=-1.0,
        help="Optional soft diagnostic ceiling for GT=4 decoded as 5 lanes. Negative disables.",
    )
    parser.add_argument(
        "--gcs-official-best-max-rate-3-to-5",
        type=float,
        default=-1.0,
        help="Optional soft diagnostic ceiling for GT=3 decoded as 5 lanes. Negative disables.",
    )
    parser.add_argument(
        "--gcs-official-best-max-rate-5-to-4",
        type=float,
        default=-1.0,
        help="Soft diagnostic ceiling for GT=5 decoded as 4 lanes. Use a negative value to disable.",
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
    return parser.parse_args()


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


def resolve_existing_path(value: str, *, flag: str) -> str:
    """Resolve a CLI path relative to the project root and require that it exists."""
    text = str(value).strip()
    if not text or text.lower() in {"none", "false", "0"}:
        raise SystemExit(f"ERROR: --gcs-official-best requires {flag}.")
    path = Path(text)
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise SystemExit(f"ERROR: {flag} does not exist: {resolved}")
    return text


def main() -> None:
    args = parse_args()
    defaults = dataset_defaults(args.dataset)
    gcs_imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    save_period = args.save_period
    gcs_official_best_period = args.gcs_official_best_period
    if args.gcs_official_best and gcs_official_best_period <= 0:
        gcs_official_best_period = save_period if save_period > 0 else 10
        if save_period > 0:
            # Older commands used --save-period as the official sweep cadence. Keep that cadence without
            # retaining epochN.pt checkpoint files; best.pt is still the ordinary val-F1 best.
            save_period = -1
    if args.gcs_official_best:
        from tools.sweep_tusimple_official import validate_official_sweep_split

        try:
            args.gcs_official_best_split = validate_official_sweep_split(
                args.gcs_official_best_split,
                context="Training official_best selection",
            )
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
        args.gcs_official_best_gt_json = resolve_existing_path(
            args.gcs_official_best_gt_json,
            flag="--gcs-official-best-gt-json",
        )
        args.gcs_official_best_archive_root = resolve_existing_path(
            args.gcs_official_best_archive_root,
            flag="--gcs-official-best-archive-root",
        )

    overrides = {
        "task": "gcs_lane",
        "model": args.model,
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
        "translate": args.translate,
        "erasing": args.erasing,
        "auto_augment": None,
        "mosaic": args.mosaic,
        "val": not args.no_val,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "save_period": save_period,
        "box": 0.0,
        "cls": 0.0,
        "dfl": 0.0,
        "pose": 0.0,
        "kobj": 0.0,
        "rle": 0.0,
        "angle": 0.0,
        "train_images": args.train_images,
        "train_gcs_labels": args.train_gcs_labels,
        "val_images": args.val_images,
        "val_gcs_labels": args.val_gcs_labels,
        "gcs_train_include_val": args.gcs_train_include_val,
        "gcs_exist": args.gcs_exist,
        "gcs_point": args.gcs_point,
        "gcs_point_valid": args.gcs_point_valid,
        "gcs_point_invalid_x": args.gcs_point_invalid_x,
        "gcs_line_iou": args.gcs_line_iou,
        "gcs_line_iou_width_px": args.gcs_line_iou_width_px,
        "gcs_count_cls": args.gcs_count_cls,
        "gcs_count_sum": args.gcs_count_sum,
        "gcs_count_sum_normalize": args.gcs_count_sum_normalize,
        "gcs_quality": args.gcs_quality,
        "gcs_quality_dist_thr_px": args.gcs_quality_dist_thr_px,
        "gcs_quality_neg_weight": args.gcs_quality_neg_weight,
        "gcs_quality_gt5_edge_floor": args.gcs_quality_gt5_edge_floor,
        "gcs_quality_hard_negative_weight": args.gcs_quality_hard_negative_weight,
        "gcs_quality_duplicate_negative_weight": args.gcs_quality_duplicate_negative_weight,
        "gcs_quality_hard_negative_from_head": args.gcs_quality_hard_negative_from_head,
        "gcs_hard_negative_visible_segment": args.gcs_hard_negative_visible_segment,
        "gcs_hard_negative_visible_thr": args.gcs_hard_negative_visible_thr,
        "gcs_hard_negative_visible_support_points": args.gcs_hard_negative_visible_support_points,
        "gcs_count_head_warmup_epochs": args.gcs_count_head_warmup_epochs,
        "gcs_count_min_gt_points": args.gcs_count_min_gt_points,
        "gcs_count_cls_w2": args.gcs_count_cls_w2,
        "gcs_count_cls_w3": args.gcs_count_cls_w3,
        "gcs_count_cls_w4": args.gcs_count_cls_w4,
        "gcs_count_cls_w5": args.gcs_count_cls_w5,
        "gcs_count_boundary": args.gcs_count_boundary,
        "gcs_count_boundary_label_smoothing": args.gcs_count_boundary_label_smoothing,
        "gcs_count_boundary_gt5_pos_weight": args.gcs_count_boundary_gt5_pos_weight,
        "gcs_count_adjacent_margin": args.gcs_count_adjacent_margin,
        "gcs_count_adjacent_margin_gain": args.gcs_count_adjacent_margin_gain,
        "gcs_count_adjacent_margin_gt45_weight": args.gcs_count_adjacent_margin_gt45_weight,
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
        "gcs_gt5_edge_loss_weight": args.gcs_gt5_edge_loss_weight,
        "gcs_candidate_gt5_edge_weight": args.gcs_candidate_gt5_edge_weight,
        "gcs_point_valid_gt5_edge_continuity": args.gcs_point_valid_gt5_edge_continuity,
        "gcs_point_valid_gt5_edge_continuity_thr": args.gcs_point_valid_gt5_edge_continuity_thr,
        "gcs_point_valid_gt5_edge_segment": args.gcs_point_valid_gt5_edge_segment,
        "gcs_point_valid_gt5_edge_segment_thr": args.gcs_point_valid_gt5_edge_segment_thr,
        "gcs_point_valid_gt5_edge_segment_min_points": args.gcs_point_valid_gt5_edge_segment_min_points,
        "gcs_hard_loss_file": args.gcs_hard_loss_file,
        "gcs_hard_loss_lane_counts": args.gcs_hard_loss_lane_counts,
        "gcs_hard_edge_loss_weight_by_count": args.gcs_hard_edge_loss_weight_by_count,
        "gcs_hard_edge_loss_terms": args.gcs_hard_edge_loss_terms,
        "gcs_hard_edge_only": args.gcs_hard_edge_only,
        "gcs_point_valid_neg": args.gcs_point_valid_neg,
        "gcs_point_valid_neg_thr": args.gcs_point_valid_neg_thr,
        "gcs_cost_point": args.gcs_cost_point,
        "gcs_cost_exist": args.gcs_cost_exist,
        "gcs_match_min_overlap": args.gcs_match_min_overlap,
        "gcs_match_max_x_dist": args.gcs_match_max_x_dist,
        "gcs_match_gate_px": args.gcs_match_gate_px,
        "gcs_eval_conf": args.gcs_eval_conf,
        "gcs_eval_ape_thr": args.gcs_eval_ape_thr,
        "gcs_eval_match_gate_px": args.gcs_eval_match_gate_px,
        "gcs_eval_max_x_dist": args.gcs_eval_max_x_dist,
        "gcs_eval_min_overlap": args.gcs_eval_min_overlap,
        "gcs_eval_min_points": args.gcs_eval_min_points,
        "gcs_eval_min_gt_cover_ratio": args.gcs_eval_min_gt_cover_ratio,
        "gcs_eval_min_pred_cover_ratio": args.gcs_eval_min_pred_cover_ratio,
        "gcs_eval_nms_dist_px": args.gcs_eval_nms_dist_px,
        "gcs_use_count_head_decode": args.gcs_use_count_head_decode,
        "gcs_count_head_temp": args.gcs_count_head_temp,
        "gcs_decode_candidate_conf": args.gcs_decode_candidate_conf,
        "gcs_decode_candidate_point_valid_thr": args.gcs_decode_candidate_point_valid_thr,
        "gcs_decode_candidate_min_points": args.gcs_decode_candidate_min_points,
        "gcs_enable_rescue_candidate_pool": args.gcs_enable_rescue_candidate_pool,
        "gcs_decode_rescue_candidate_conf": args.gcs_decode_rescue_candidate_conf,
        "gcs_decode_rescue_candidate_point_valid_thr": args.gcs_decode_rescue_candidate_point_valid_thr,
        "gcs_decode_rescue_candidate_min_points": args.gcs_decode_rescue_candidate_min_points,
        "gcs_decode_final_min_points": args.gcs_decode_final_min_points,
        "gcs_decode_fifth_min_points": args.gcs_decode_fifth_min_points,
        "gcs_line_nms_min_overlap": args.gcs_line_nms_min_overlap,
        "gcs_line_nms_rescue_dist_px": args.gcs_line_nms_rescue_dist_px,
        "gcs_quality_rescue_5th": args.gcs_quality_rescue_5th,
        "gcs_quality_rescue_count5_thr": args.gcs_quality_rescue_count5_thr,
        "gcs_quality_rescue_conf_thr": args.gcs_quality_rescue_conf_thr,
        "gcs_quality_rescue_mean_valid_thr": args.gcs_quality_rescue_mean_valid_thr,
        "gcs_quality_rescue_quality_thr": args.gcs_quality_rescue_quality_thr,
        "gcs_quality_rescue_min_points": args.gcs_quality_rescue_min_points,
        "gcs_quality_rescue_dist_px": args.gcs_quality_rescue_dist_px,
        "gcs_last_lane_rescue": args.gcs_last_lane_rescue,
        "gcs_last_lane_rescue_min_policy_count": args.gcs_last_lane_rescue_min_policy_count,
        "gcs_last_lane_rescue_conf_thr": args.gcs_last_lane_rescue_conf_thr,
        "gcs_last_lane_rescue_point_valid_thr": args.gcs_last_lane_rescue_point_valid_thr,
        "gcs_last_lane_rescue_min_points": args.gcs_last_lane_rescue_min_points,
        "gcs_last_lane_rescue_mean_valid_thr": args.gcs_last_lane_rescue_mean_valid_thr,
        "gcs_last_lane_rescue_quality_thr": args.gcs_last_lane_rescue_quality_thr,
        "gcs_last_lane_rescue_dist_px": args.gcs_last_lane_rescue_dist_px,
        "gcs_edge_last_lane_rescue": args.gcs_edge_last_lane_rescue,
        "gcs_edge_rescue_conf_thr": args.gcs_edge_rescue_conf_thr,
        "gcs_edge_rescue_point_valid_thr": args.gcs_edge_rescue_point_valid_thr,
        "gcs_edge_rescue_min_points": args.gcs_edge_rescue_min_points,
        "gcs_edge_rescue_mean_valid_thr": args.gcs_edge_rescue_mean_valid_thr,
        "gcs_edge_rescue_quality_thr": args.gcs_edge_rescue_quality_thr,
        "gcs_edge_rescue_outside_gap_px": args.gcs_edge_rescue_outside_gap_px,
        "gcs_edge_rescue_dist_px": args.gcs_edge_rescue_dist_px,
        "gcs_edge_rescue_min_policy_count": args.gcs_edge_rescue_min_policy_count,
        "gcs_edge_count4_to5_upgrade": args.gcs_edge_count4_to5_upgrade,
        "gcs_edge_count4_to5_prob_margin": args.gcs_edge_count4_to5_prob_margin,
        "gcs_soft_count_decision": args.gcs_soft_count_decision,
        "gcs_soft_count_prob_margin": args.gcs_soft_count_prob_margin,
        "gcs_soft_count_quality_weight": args.gcs_soft_count_quality_weight,
        "gcs_soft_count_prior_weight": args.gcs_soft_count_prior_weight,
        "gcs_soft_count_duplicate_penalty": args.gcs_soft_count_duplicate_penalty,
        "gcs_soft_count_invalid_penalty": args.gcs_soft_count_invalid_penalty,
        "gcs_eval_point_valid_thr": args.gcs_eval_point_valid_thr,
        "gcs_eval_max_det": args.gcs_eval_max_det,
        "gcs_sampler_mode": args.gcs_sampler_mode,
        "gcs_group_sampler_ratios": args.gcs_group_sampler_ratios,
        "gcs_lane_count_balanced": args.gcs_lane_count_balanced,
        "gcs_lane_count_balance_power": args.gcs_lane_count_balance_power,
        "gcs_lane_count_min_group": args.gcs_lane_count_min_group,
        "gcs_hard_sampling": args.gcs_hard_sampling,
        "gcs_hard_lane_counts": args.gcs_hard_lane_counts,
        "gcs_hard_sampling_boost": args.gcs_hard_sampling_boost,
        "gcs_hard_sampling_boost_by_count": args.gcs_hard_sampling_boost_by_count,
        "gcs_hard_sample_file": args.gcs_hard_sample_file,
        "gcs_hard_sample_boost": args.gcs_hard_sample_boost,
        "gcs_gt5_oversample_weight": args.gcs_gt5_oversample_weight,
        "gcs_gt5_extra_aug": args.gcs_gt5_extra_aug,
        "gcs_gt5_aug_min_lanes": args.gcs_gt5_aug_min_lanes,
        "gcs_gt5_erasing": args.gcs_gt5_erasing,
        "gcs_gt5_blur": args.gcs_gt5_blur,
        "gcs_gt5_noise": args.gcs_gt5_noise,
        "gcs_gt5_shadow": args.gcs_gt5_shadow,
        "gcs_official_best": args.gcs_official_best,
        "gcs_official_best_period": gcs_official_best_period,
        "gcs_official_best_top_k": args.gcs_official_best_top_k,
        "gcs_official_best_gt_json": args.gcs_official_best_gt_json,
        "gcs_official_best_archive_root": args.gcs_official_best_archive_root,
        "gcs_official_best_split": args.gcs_official_best_split,
        "gcs_official_best_confs": args.gcs_official_best_confs,
        "gcs_official_best_point_valid_thrs": args.gcs_official_best_point_valid_thrs,
        "gcs_official_best_nms_dist_pxs": args.gcs_official_best_nms_dist_pxs,
        "gcs_official_best_max_dets": args.gcs_official_best_max_dets,
        "gcs_official_best_min_points": args.gcs_official_best_min_points,
        "gcs_official_best_rank_min_points": args.gcs_official_best_rank_min_points,
        "gcs_official_best_last_lane_rescue_point_valid_thrs": args.gcs_official_best_last_lane_rescue_point_valid_thrs,
        "gcs_official_best_last_lane_rescue_min_points": args.gcs_official_best_last_lane_rescue_min_points,
        "gcs_official_best_last_lane_rescue_mean_valid_thrs": args.gcs_official_best_last_lane_rescue_mean_valid_thrs,
        "gcs_official_best_last_lane_rescue_quality_thrs": args.gcs_official_best_last_lane_rescue_quality_thrs,
        "gcs_official_best_last_lane_rescue_dist_pxs": args.gcs_official_best_last_lane_rescue_dist_pxs,
        "gcs_official_best_max_images": args.gcs_official_best_max_images,
        "gcs_official_best_warmup": args.gcs_official_best_warmup,
        "gcs_official_best_half": args.gcs_official_best_half,
        "gcs_official_best_score_fp_weight": args.gcs_official_best_score_fp_weight,
        "gcs_official_best_score_fn_weight": args.gcs_official_best_score_fn_weight,
        "gcs_official_best_count_acc3_weight": args.gcs_official_best_count_acc3_weight,
        "gcs_official_best_count_acc4_weight": args.gcs_official_best_count_acc4_weight,
        "gcs_official_best_count_acc5_weight": args.gcs_official_best_count_acc5_weight,
        "gcs_official_best_rate_4_to_5_weight": args.gcs_official_best_rate_4_to_5_weight,
        "gcs_official_best_rate_3_to_5_weight": args.gcs_official_best_rate_3_to_5_weight,
        "gcs_official_best_rate_4_to_3_weight": args.gcs_official_best_rate_4_to_3_weight,
        "gcs_official_best_rate_3_to_4_weight": args.gcs_official_best_rate_3_to_4_weight,
        "gcs_official_best_rate_5_to_4_weight": args.gcs_official_best_rate_5_to_4_weight,
        "gcs_official_best_min_count_acc3": args.gcs_official_best_min_count_acc3,
        "gcs_official_best_min_count_acc4": args.gcs_official_best_min_count_acc4,
        "gcs_official_best_min_count_acc5": args.gcs_official_best_min_count_acc5,
        "gcs_official_best_min_gt5_output5_rate": args.gcs_official_best_min_gt5_output5_rate,
        "gcs_official_best_max_gt5_count_head_under_rate": args.gcs_official_best_max_gt5_count_head_under_rate,
        "gcs_official_best_max_gt5_valid_points_fail_rate": args.gcs_official_best_max_gt5_valid_points_fail_rate,
        "gcs_official_best_max_rate_3_to_4": args.gcs_official_best_max_rate_3_to_4,
        "gcs_official_best_max_rate_4_to_5": args.gcs_official_best_max_rate_4_to_5,
        "gcs_official_best_max_rate_4_to_3": args.gcs_official_best_max_rate_4_to_3,
        "gcs_official_best_max_rate_3_to_5": args.gcs_official_best_max_rate_3_to_5,
        "gcs_official_best_max_rate_5_to_4": args.gcs_official_best_max_rate_5_to_4,
    }

    print(f"GCS input shape: {shape_str(gcs_imgsz)} (W x H), stored as H,W={gcs_imgsz}")
    trainer = GCSLaneTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
