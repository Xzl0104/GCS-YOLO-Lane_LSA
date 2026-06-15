"""Smoke-test GCS-YOLO-Lane module registration and model construction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules import GCSLaneHead, LSEM, LaneBiFPN, LaneFeatureProjection
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str

DEFAULT_CFG = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q12.yaml"


def parse_args():
    """Parse command line arguments for the smoke test."""
    parser = argparse.ArgumentParser(description="Check GCS-YOLO-Lane model registration and forward output shapes.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="Path to the GCS-YOLO-Lane YAML.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="Input shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument("--batch", type=int, default=2, help="Batch size for the forward smoke test.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--detailed", action="store_true", help="Print extra task and registration checks.")
    return parser.parse_args()


def main():
    """Build the GCS model and verify inference/training outputs."""
    args = parse_args()
    cfg = args.cfg if args.cfg.is_absolute() else ROOT / args.cfg
    if not cfg.exists():
        raise FileNotFoundError(f"GCS model YAML not found: {cfg}")

    yolo = YOLO(str(cfg), task="gcs_lane", verbose=True)
    model = yolo.model.to(args.device)

    has_lsem = any(isinstance(m, LSEM) for m in model.modules())
    has_bifpn = any(isinstance(m, LaneBiFPN) for m in model.modules())
    has_projection = any(isinstance(m, LaneFeatureProjection) for m in model.modules())
    has_head = isinstance(model.model[-1], GCSLaneHead)
    head = model.model[-1] if has_head else None

    cfg_stem = cfg.stem.lower()
    expects_no_lsem = "no-lsem" in cfg_stem
    expects_projection = "proj-no-bifpn" in cfg_stem
    valid_lsem = not has_lsem if expects_no_lsem else has_lsem
    valid_fusion = has_projection if expects_projection else has_bifpn
    if not (valid_lsem and valid_fusion and has_head):
        raise RuntimeError("GCS-YOLO-Lane registration check failed.")

    img_h, img_w = normalize_imgsz(args.imgsz, dataset=args.dataset)
    model.gcs_imgsz = (img_h, img_w)
    x = torch.randn(args.batch, 3, img_h, img_w, device=args.device)

    model.train()
    with torch.no_grad():
        y = model(x)

    expected = {
        "pred_points",
        "pred_logits",
        "pred_valid_logits",
        "pred_count_logits",
        "pred_count_boundary_logits",
        "pred_quality_logits",
    }
    if not isinstance(y, dict):
        raise RuntimeError(f"Expected GCSLaneHead to return a dict, got {type(y).__name__}.")
    missing = sorted(expected - set(y))
    extra = sorted(set(y) - expected)
    if missing or extra:
        raise RuntimeError(f"Unexpected GCSLaneHead output keys: missing={missing}, extra={extra}.")
    expected_points_shape = (args.batch, head.num_queries, head.num_points, 2)
    if tuple(y["pred_points"].shape) != expected_points_shape:
        raise RuntimeError(
            f"pred_points must have shape B x Q x K x 2, got {tuple(y['pred_points'].shape)} "
            f"vs expected {expected_points_shape}."
        )
    if y["pred_logits"].shape != y["pred_points"].shape[:2]:
        raise RuntimeError(
            "pred_logits must have shape B x Q matching pred_points, "
            f"got {tuple(y['pred_logits'].shape)} vs {tuple(y['pred_points'].shape[:2])}."
        )
    if y["pred_valid_logits"].shape != y["pred_points"].shape[:3]:
        raise RuntimeError(
            "pred_valid_logits must have shape B x Q x K matching pred_points, "
            f"got {tuple(y['pred_valid_logits'].shape)} vs {tuple(y['pred_points'].shape[:3])}."
        )
    if y["pred_count_logits"].shape != (args.batch, 4):
        raise RuntimeError(f"pred_count_logits must have shape B x 4, got {tuple(y['pred_count_logits'].shape)}.")
    if y["pred_count_boundary_logits"].shape != (args.batch, 2):
        raise RuntimeError(
            "pred_count_boundary_logits must have shape B x 2, "
            f"got {tuple(y['pred_count_boundary_logits'].shape)}."
        )
    if y["pred_quality_logits"].shape != y["pred_points"].shape[:2]:
        raise RuntimeError(
            "pred_quality_logits must have shape B x Q matching pred_points, "
            f"got {tuple(y['pred_quality_logits'].shape)} vs {tuple(y['pred_points'].shape[:2])}."
        )
    if getattr(head, "point_mode", "free") == "fixed_y":
        if int(getattr(head, "point_dims", 2)) != 1:
            raise RuntimeError("fixed_y GCSLaneHead must use point_dims=1 for x-only prediction.")
        anchors = head.fixed_y_anchors.to(device=y["pred_points"].device, dtype=y["pred_points"].dtype)
        y_pred = y["pred_points"][..., 1]
        max_y_err = float((y_pred - anchors.view(1, 1, -1)).abs().max().cpu().item())
        if max_y_err > 1e-6:
            raise RuntimeError(f"fixed_y GCSLaneHead produced non-anchor y coordinates, max error={max_y_err:.6g}.")
        final = head.point_mlp[-1]
        if getattr(final, "out_features", None) != head.num_points:
            raise RuntimeError(
                f"fixed_y GCSLaneHead point MLP must output K x values, got out_features={final.out_features}."
            )

    if args.detailed:
        print(f"task: {yolo.task}")
        print(f"model: {type(model).__name__}")
        print(f"input shape: {shape_str((img_h, img_w))} (W x H), stored as H,W={(img_h, img_w)}")
        print(f"registered LSEM: {has_lsem}")
        print(f"registered LaneBiFPN: {has_bifpn}")
        print(f"registered LaneFeatureProjection: {has_projection}")
        print(f"registered GCSLaneHead: {has_head}")
        print(f"GCSLaneHead point_mode: {getattr(head, 'point_mode', None)}")
        print(f"GCSLaneHead point_dims: {getattr(head, 'point_dims', None)}")

    print(type(y))
    for k, v in y.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
