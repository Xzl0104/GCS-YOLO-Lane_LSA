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
        help="Optional explicit official-val GT json-lines file for training-time official selection.",
    )
    parser.add_argument("--gcs-official-max-images", type=int, default=0, help="Limit official-val images per hook. 0 means all.")
    parser.add_argument("--gcs-official-warmup", type=int, default=5, help="Warmup forwards for each training-time official sweep.")
    parser.add_argument("--gcs-official-confs", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--gcs-official-point-valid-thrs", nargs="+", type=float, default=[0.45, 0.5])
    parser.add_argument("--gcs-official-nms-dist-pxs", nargs="+", type=float, default=[0.0, 18.0, 30.0, 50.0])
    parser.add_argument("--gcs-official-max-dets", nargs="+", type=int, default=[5, 6, 8])
    parser.add_argument("--gcs-official-min-points", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--gcs-official-score-fp-weight", type=float, default=0.02)
    parser.add_argument("--gcs-official-score-fn-weight", type=float, default=0.02)
    parser.add_argument(
        "--gcs-official-half",
        action="store_true",
        help="Use FP16 inference during training-time official-val sweeps.",
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
        "gcs_count_boundary_margin34": args.gcs_count_boundary_margin34,
        "gcs_count_boundary_margin45": args.gcs_count_boundary_margin45,
        "gcs_spurious_neg": args.gcs_spurious_neg,
        "gcs_spurious_neg_weight": args.gcs_spurious_neg_weight,
        "gcs_spurious_gt5_weight": args.gcs_spurious_gt5_weight,
        "gcs_spurious_disable_gt5": args.gcs_spurious_disable_gt5,
        "gcs_spurious_max_points": args.gcs_spurious_max_points,
        "gcs_spurious_close_px": args.gcs_spurious_close_px,
        "gcs_spurious_min_overlap": args.gcs_spurious_min_overlap,
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
        "gcs_official_interval": args.gcs_official_interval,
        "gcs_official_archive_root": args.gcs_official_archive_root,
        "gcs_official_gt_json": args.gcs_official_gt_json,
        "gcs_official_max_images": args.gcs_official_max_images,
        "gcs_official_warmup": args.gcs_official_warmup,
        "gcs_official_confs": args.gcs_official_confs,
        "gcs_official_point_valid_thrs": args.gcs_official_point_valid_thrs,
        "gcs_official_nms_dist_pxs": args.gcs_official_nms_dist_pxs,
        "gcs_official_max_dets": args.gcs_official_max_dets,
        "gcs_official_min_points": args.gcs_official_min_points,
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

    print(f"GCS input shape: {shape_str(gcs_imgsz)} (W x H), stored as H,W={gcs_imgsz}")
    trainer = GCSLaneTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
