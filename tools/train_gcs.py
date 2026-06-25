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
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str, trainer_imgsz


DEFAULT_MODEL = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q12-k56.yaml"


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
        fixed_data = ROOT / "data" / "tusimple_gcs_fixed_y_k56_960x544.yaml"
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
    parser.add_argument("--gcs-exist", type=float, default=2.0)
    parser.add_argument("--gcs-point", type=float, default=15.0)
    parser.add_argument(
        "--gcs-lane-balanced-point-loss",
        "--gcs_lane_balanced_point_loss",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="Replace the base matched x point loss with lane-balanced reduction. Default off.",
    )
    parser.add_argument(
        "--gcs-lane-balanced-point",
        type=float,
        default=0.0,
        help="Extra lane-balanced matched point loss gain. 0 disables.",
    )
    parser.add_argument(
        "--gcs-gt4-short-lane-weight",
        type=float,
        default=1.0,
        help="Lane-balanced point-loss multiplier for GT4 short lanes when --gcs-lane-balanced-point-loss is enabled.",
    )
    parser.add_argument(
        "--gcs-gt4-short-lane-max-points",
        type=int,
        default=20,
        help="Maximum GT-visible anchors for GT4 short-lane point-loss weighting.",
    )
    parser.add_argument(
        "--gcs-gt4-lane-balanced-point",
        type=float,
        default=0.0,
        help="GT4-only weak matched-lane point reweighting gain. 0 disables.",
    )
    parser.add_argument(
        "--gcs-gt4-lane-balanced-topk",
        type=int,
        default=1,
        help="Number of highest point-loss matched GT4 lanes to upweight.",
    )
    parser.add_argument(
        "--gcs-gt4-lane-balanced-max-mult",
        type=float,
        default=2.0,
        help="Maximum weak-lane multiplier before per-image mean normalization.",
    )
    parser.add_argument("--gcs-point-valid", type=float, default=1.0)
    parser.add_argument(
        "--gcs-lane-balanced-valid-loss",
        "--gcs_lane_balanced_valid_loss",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="Replace base point-valid BCE with matched per-lane balanced reduction. Default off.",
    )
    parser.add_argument(
        "--gcs-gt4-short-valid-lane-weight",
        type=float,
        default=1.5,
        help="Lane-balanced valid-loss multiplier for matched GT4 short lanes.",
    )
    parser.add_argument(
        "--gcs-gt4-short-valid-pos-weight",
        type=float,
        default=1.0,
        help="Optional positive-point multiplier inside lane-balanced valid BCE.",
    )
    parser.add_argument(
        "--gcs-unmatched-valid-neg-weight",
        type=float,
        default=0.5,
        help="Unmatched-query zero-target valid BCE weight inside lane-balanced valid replacement.",
    )
    parser.add_argument(
        "--gcs-short-valid-recall",
        type=float,
        default=0.0,
        help="Positive-only point-valid recall loss gain for matched short GT lanes. 0 disables.",
    )
    parser.add_argument(
        "--gcs-short-valid-max-visible",
        type=int,
        default=20,
        help="Maximum GT-visible anchors for short-valid recall loss.",
    )
    parser.add_argument(
        "--gcs-short-valid-min-visible",
        type=int,
        default=4,
        help="Minimum GT-visible anchors for short-valid recall loss.",
    )
    parser.add_argument(
        "--gcs-short-valid-max-ape-px",
        type=float,
        default=40.0,
        help="Maximum matched APE in pixels for short-valid recall loss.",
    )
    parser.add_argument(
        "--gcs-short-valid-min-visible-iou",
        type=float,
        default=0.3,
        help="Minimum matched visible-IoU for short-valid recall loss.",
    )
    parser.add_argument(
        "--gcs-gt4-short-valid-recall",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="Enable positive-only valid recall loss for matched GT4 short lanes. Default off.",
    )
    parser.add_argument(
        "--gcs-gt4-short-valid-recall-weight",
        type=float,
        default=0.2,
        help="Gain for GT4 short valid recall loss.",
    )
    parser.add_argument(
        "--gcs-gt4-short-valid-max-points",
        type=int,
        default=20,
        help="Maximum GT-visible anchors for GT4 short valid repair.",
    )
    parser.add_argument(
        "--gcs-gt4-short-valid-count-floor",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="Enable GT4 short valid probability count floor loss. Default off.",
    )
    parser.add_argument(
        "--gcs-gt4-short-valid-count-floor-weight",
        type=float,
        default=0.05,
        help="Gain for GT4 short valid count floor loss.",
    )
    parser.add_argument(
        "--gcs-gt4-short-valid-count-floor-ratio",
        type=float,
        default=0.6,
        help="Target floor ratio of GT valid points for GT4 short valid count floor.",
    )
    parser.add_argument(
        "--gcs-gt4-short-valid-count-floor-min",
        type=int,
        default=3,
        help="Minimum target floor before clamping to GT valid points.",
    )
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
        "--gcs-count-ce",
        type=float,
        default=0.0,
        help="Cross-entropy loss gain for the explicit 3/4/5 lane-count head. 0 disables.",
    )
    parser.add_argument(
        "--gcs-duplicate-margin",
        type=float,
        default=0.0,
        help="Pairwise duplicate-like unmatched query margin loss gain. 0 disables.",
    )
    parser.add_argument(
        "--gcs-duplicate-margin-logit",
        type=float,
        default=1.0,
        help="Required logit margin between reliable matched q+ and duplicate-like q-.",
    )
    parser.add_argument(
        "--gcs-duplicate-gt-count",
        type=int,
        default=4,
        help="Only apply duplicate margin loss to images with this GT lane count.",
    )
    parser.add_argument(
        "--gcs-duplicate-short-visible-max",
        type=int,
        default=20,
        help="Only apply duplicate margin loss to GT lanes with at most this many visible anchors.",
    )
    parser.add_argument(
        "--gcs-duplicate-min-overlap",
        type=int,
        default=2,
        help="Minimum hard visible-anchor overlap between q- and the short GT lane.",
    )
    parser.add_argument(
        "--gcs-duplicate-min-visible-iou",
        type=float,
        default=0.4,
        help="Minimum soft visibility IoU for q+ reliability and q- duplicate selection.",
    )
    parser.add_argument(
        "--gcs-duplicate-pos-ape-px",
        type=float,
        default=20.0,
        help="Maximum APE in pixels for the matched q+ to be treated as reliable.",
    )
    parser.add_argument(
        "--gcs-duplicate-neg-ape-px",
        type=float,
        default=120.0,
        help="Maximum APE in pixels for q- duplicate selection; filters far background queries.",
    )
    parser.add_argument(
        "--gcs-duplicate-ape-gap-px",
        type=float,
        default=5.0,
        help="q- must be this much worse than q+, unless above the absolute q+ APE threshold.",
    )
    parser.add_argument(
        "--gcs-duplicate-max-pairs-per-gt",
        type=int,
        default=2,
        help="Maximum high-logit q- pairs kept per matched GT lane.",
    )
    parser.add_argument(
        "--gcs-spurious-margin",
        type=float,
        default=0.0,
        help="Pairwise far-unmatched spurious query margin loss gain. 0 disables.",
    )
    parser.add_argument(
        "--gcs-spurious-margin-logit",
        type=float,
        default=1.0,
        help="Required logit margin between reliable matched q+ and far spurious q-.",
    )
    parser.add_argument(
        "--gcs-spurious-gt-counts",
        default="3,4",
        help="Comma-separated GT lane counts that enable spurious margin loss, e.g. '3,4'.",
    )
    parser.add_argument(
        "--gcs-spurious-pos-ape-px",
        type=float,
        default=20.0,
        help="Maximum APE in pixels for matched q+ to be treated as reliable.",
    )
    parser.add_argument(
        "--gcs-spurious-pos-min-visible-iou",
        type=float,
        default=0.4,
        help="Minimum soft visible IoU for matched q+ to be treated as reliable.",
    )
    parser.add_argument(
        "--gcs-spurious-neg-min-ape-px",
        type=float,
        default=50.0,
        help="q- is treated as far/spurious when its best GT APE is above this threshold.",
    )
    parser.add_argument(
        "--gcs-spurious-neg-max-visible-iou",
        type=float,
        default=0.2,
        help="q- is treated as far/spurious when its best GT visible IoU is below this threshold.",
    )
    parser.add_argument(
        "--gcs-spurious-duplicate-ape-px",
        type=float,
        default=50.0,
        help="Exclude near-GT duplicate-like q- at or below this best-GT APE.",
    )
    parser.add_argument(
        "--gcs-spurious-duplicate-visible-iou",
        type=float,
        default=0.4,
        help="Exclude near-GT duplicate-like q- at or above this best-GT visible IoU.",
    )
    parser.add_argument(
        "--gcs-spurious-max-pairs-per-image",
        type=int,
        default=4,
        help="Maximum high-risk spurious ranking pairs kept per image.",
    )
    parser.add_argument(
        "--gcs-far-spurious-survival",
        type=float,
        default=0.0,
        help="Absolute-logit loss gain for decode-risk far-spurious unmatched q-. 0 disables.",
    )
    parser.add_argument(
        "--gcs-far-spurious-gt-counts",
        default="3,4",
        help="Comma-separated GT lane counts that enable far-spurious survival loss, e.g. '3,4'.",
    )
    parser.add_argument(
        "--gcs-far-spurious-score-thr",
        type=float,
        default=0.03,
        help="Target score threshold for far-spurious q- suppression.",
    )
    parser.add_argument(
        "--gcs-far-spurious-min-score",
        type=float,
        default=0.03,
        help="Minimum q- score required before applying far-spurious survival loss.",
    )
    parser.add_argument(
        "--gcs-far-spurious-min-ape-px",
        type=float,
        default=50.0,
        help="q- is treated as far-spurious if best-GT APE is above this or visible IoU is low.",
    )
    parser.add_argument(
        "--gcs-far-spurious-max-visible-iou",
        type=float,
        default=0.2,
        help="q- is treated as far-spurious if best-GT visible IoU is below this or APE is high.",
    )
    parser.add_argument(
        "--gcs-far-spurious-point-valid-thr",
        type=float,
        default=0.5,
        help="Point-valid threshold used for the decode-risk visible-run gate.",
    )
    parser.add_argument(
        "--gcs-far-spurious-min-visible-run",
        type=int,
        default=5,
        help="Minimum contiguous predicted visible anchors for q- decode survival risk.",
    )
    parser.add_argument(
        "--gcs-far-spurious-max-neg-per-image",
        type=int,
        default=1,
        help="Maximum highest-logit far-spurious q- penalties per image.",
    )
    parser.add_argument(
        "--gcs-far-spurious-loss-type",
        default="relu",
        choices=("relu", "softplus"),
        help="Absolute-logit penalty form for far-spurious survival loss.",
    )
    parser.add_argument(
        "--gcs-gt5-rank-consistency",
        type=float,
        default=0.0,
        help="GT5-only weakest matched q+ vs top unmatched q- rank consistency loss gain. 0 disables.",
    )
    parser.add_argument(
        "--gcs-gt5-rank-margin-logit",
        type=float,
        default=0.5,
        help="Required logit margin between GT5 weakest q+ and top q-.",
    )
    parser.add_argument(
        "--gcs-gt5-rank-min-qminus-score",
        type=float,
        default=0.02,
        help="Minimum top q- score required before applying GT5 rank loss.",
    )
    parser.add_argument(
        "--gcs-gt5-rank-max-pairs-per-image",
        type=int,
        default=1,
        help="Maximum high-logit unmatched q- pairs per GT5 image.",
    )
    parser.add_argument(
        "--gcs-gt3-extra-survival",
        type=float,
        default=0.0,
        help="GT3-only top unmatched q- vs weakest matched q+ hinge loss gain. 0 disables.",
    )
    parser.add_argument(
        "--gcs-gt3-extra-margin-logit",
        type=float,
        default=0.05,
        help="Required logit margin between GT3 weakest q+ and top extra q-.",
    )
    parser.add_argument(
        "--gcs-gt3-extra-topk",
        type=int,
        default=1,
        help="Number of highest-logit unmatched GT3 queries to rank below weakest q+.",
    )
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
    parser.add_argument(
        "--gcs-gt4-short-match-endpoint",
        type=float,
        default=0.0,
        help="Extra normalized-x endpoint cost gain for short GT lanes in GT4 images during Hungarian matching. 0 disables.",
    )
    parser.add_argument(
        "--gcs-gt4-short-match-max-points",
        type=int,
        default=20,
        help="Maximum GT-visible anchors for the GT4 short-lane endpoint matcher cost.",
    )
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
    parser.add_argument(
        "--gcs-gt4-sample-gain",
        type=float,
        default=1.0,
        help="Train-only sampler gain for GT4 images after lane-count balancing. 1 disables; keep first-pass <= 2.",
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


def main() -> None:
    args = parse_args()
    defaults = dataset_defaults(args.dataset)
    gcs_imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)

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
        "erasing": args.erasing,
        "mosaic": args.mosaic,
        "val": not args.no_val,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "train_images": args.train_images,
        "train_gcs_labels": args.train_gcs_labels,
        "val_images": args.val_images,
        "val_gcs_labels": args.val_gcs_labels,
        "gcs_exist": args.gcs_exist,
        "gcs_point": args.gcs_point,
        "gcs_lane_balanced_point_loss": args.gcs_lane_balanced_point_loss,
        "gcs_lane_balanced_point": args.gcs_lane_balanced_point,
        "gcs_gt4_short_lane_weight": args.gcs_gt4_short_lane_weight,
        "gcs_gt4_short_lane_max_points": args.gcs_gt4_short_lane_max_points,
        "gcs_gt4_lane_balanced_point": args.gcs_gt4_lane_balanced_point,
        "gcs_gt4_lane_balanced_topk": args.gcs_gt4_lane_balanced_topk,
        "gcs_gt4_lane_balanced_max_mult": args.gcs_gt4_lane_balanced_max_mult,
        "gcs_point_valid": args.gcs_point_valid,
        "gcs_lane_balanced_valid_loss": args.gcs_lane_balanced_valid_loss,
        "gcs_gt4_short_valid_lane_weight": args.gcs_gt4_short_valid_lane_weight,
        "gcs_gt4_short_valid_pos_weight": args.gcs_gt4_short_valid_pos_weight,
        "gcs_unmatched_valid_neg_weight": args.gcs_unmatched_valid_neg_weight,
        "gcs_short_valid_recall": args.gcs_short_valid_recall,
        "gcs_short_valid_max_visible": args.gcs_short_valid_max_visible,
        "gcs_short_valid_min_visible": args.gcs_short_valid_min_visible,
        "gcs_short_valid_max_ape_px": args.gcs_short_valid_max_ape_px,
        "gcs_short_valid_min_visible_iou": args.gcs_short_valid_min_visible_iou,
        "gcs_gt4_short_valid_recall": args.gcs_gt4_short_valid_recall,
        "gcs_gt4_short_valid_recall_weight": args.gcs_gt4_short_valid_recall_weight,
        "gcs_gt4_short_valid_max_points": args.gcs_gt4_short_valid_max_points,
        "gcs_gt4_short_valid_count_floor": args.gcs_gt4_short_valid_count_floor,
        "gcs_gt4_short_valid_count_floor_weight": args.gcs_gt4_short_valid_count_floor_weight,
        "gcs_gt4_short_valid_count_floor_ratio": args.gcs_gt4_short_valid_count_floor_ratio,
        "gcs_gt4_short_valid_count_floor_min": args.gcs_gt4_short_valid_count_floor_min,
        "gcs_smooth": args.gcs_smooth,
        "gcs_curve": args.gcs_curve,
        "gcs_mask": args.gcs_mask,
        "gcs_edge": args.gcs_edge,
        "gcs_count": args.gcs_count,
        "gcs_count_under5": args.gcs_count_under5,
        "gcs_count_under5_min_lanes": args.gcs_count_under5_min_lanes,
        "gcs_count_ce": args.gcs_count_ce,
        "gcs_duplicate_margin": args.gcs_duplicate_margin,
        "gcs_duplicate_margin_logit": args.gcs_duplicate_margin_logit,
        "gcs_duplicate_gt_count": args.gcs_duplicate_gt_count,
        "gcs_duplicate_short_visible_max": args.gcs_duplicate_short_visible_max,
        "gcs_duplicate_min_overlap": args.gcs_duplicate_min_overlap,
        "gcs_duplicate_min_visible_iou": args.gcs_duplicate_min_visible_iou,
        "gcs_duplicate_pos_ape_px": args.gcs_duplicate_pos_ape_px,
        "gcs_duplicate_neg_ape_px": args.gcs_duplicate_neg_ape_px,
        "gcs_duplicate_ape_gap_px": args.gcs_duplicate_ape_gap_px,
        "gcs_duplicate_max_pairs_per_gt": args.gcs_duplicate_max_pairs_per_gt,
        "gcs_spurious_margin": args.gcs_spurious_margin,
        "gcs_spurious_margin_logit": args.gcs_spurious_margin_logit,
        "gcs_spurious_gt_counts": args.gcs_spurious_gt_counts,
        "gcs_spurious_pos_ape_px": args.gcs_spurious_pos_ape_px,
        "gcs_spurious_pos_min_visible_iou": args.gcs_spurious_pos_min_visible_iou,
        "gcs_spurious_neg_min_ape_px": args.gcs_spurious_neg_min_ape_px,
        "gcs_spurious_neg_max_visible_iou": args.gcs_spurious_neg_max_visible_iou,
        "gcs_spurious_duplicate_ape_px": args.gcs_spurious_duplicate_ape_px,
        "gcs_spurious_duplicate_visible_iou": args.gcs_spurious_duplicate_visible_iou,
        "gcs_spurious_max_pairs_per_image": args.gcs_spurious_max_pairs_per_image,
        "gcs_far_spurious_survival": args.gcs_far_spurious_survival,
        "gcs_far_spurious_gt_counts": args.gcs_far_spurious_gt_counts,
        "gcs_far_spurious_score_thr": args.gcs_far_spurious_score_thr,
        "gcs_far_spurious_min_score": args.gcs_far_spurious_min_score,
        "gcs_far_spurious_min_ape_px": args.gcs_far_spurious_min_ape_px,
        "gcs_far_spurious_max_visible_iou": args.gcs_far_spurious_max_visible_iou,
        "gcs_far_spurious_point_valid_thr": args.gcs_far_spurious_point_valid_thr,
        "gcs_far_spurious_min_visible_run": args.gcs_far_spurious_min_visible_run,
        "gcs_far_spurious_max_neg_per_image": args.gcs_far_spurious_max_neg_per_image,
        "gcs_far_spurious_loss_type": args.gcs_far_spurious_loss_type,
        "gcs_gt5_rank_consistency": args.gcs_gt5_rank_consistency,
        "gcs_gt5_rank_margin_logit": args.gcs_gt5_rank_margin_logit,
        "gcs_gt5_rank_min_qminus_score": args.gcs_gt5_rank_min_qminus_score,
        "gcs_gt5_rank_max_pairs_per_image": args.gcs_gt5_rank_max_pairs_per_image,
        "gcs_gt3_extra_survival": args.gcs_gt3_extra_survival,
        "gcs_gt3_extra_margin_logit": args.gcs_gt3_extra_margin_logit,
        "gcs_gt3_extra_topk": args.gcs_gt3_extra_topk,
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
        "gcs_gt4_short_match_endpoint": args.gcs_gt4_short_match_endpoint,
        "gcs_gt4_short_match_max_points": args.gcs_gt4_short_match_max_points,
        "gcs_eval_conf": args.gcs_eval_conf,
        "gcs_eval_ape_thr": args.gcs_eval_ape_thr,
        "gcs_eval_match_gate_px": args.gcs_eval_match_gate_px,
        "gcs_eval_max_x_dist": args.gcs_eval_max_x_dist,
        "gcs_eval_min_overlap": args.gcs_eval_min_overlap,
        "gcs_eval_nms_dist_px": args.gcs_eval_nms_dist_px,
        "gcs_eval_point_valid_thr": args.gcs_eval_point_valid_thr,
        "gcs_eval_max_det": args.gcs_eval_max_det,
        "gcs_lane_count_balanced": args.gcs_lane_count_balanced,
        "gcs_lane_count_balance_power": args.gcs_lane_count_balance_power,
        "gcs_lane_count_min_group": args.gcs_lane_count_min_group,
        "gcs_gt4_sample_gain": args.gcs_gt4_sample_gain,
    }

    print(f"GCS input shape: {shape_str(gcs_imgsz)} (W x H), stored as H,W={gcs_imgsz}")
    trainer = GCSLaneTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
