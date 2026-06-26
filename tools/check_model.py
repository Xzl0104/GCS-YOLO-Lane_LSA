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
from ultralytics.nn.modules import GCSLaneHead, LSEM, LaneBiFPN
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str

DEFAULT_CFG = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q12-k56.yaml"


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
    has_head = isinstance(model.model[-1], GCSLaneHead)
    head = model.model[-1] if has_head else None

    if not (has_lsem and has_bifpn and has_head):
        raise RuntimeError("GCS-YOLO-Lane registration check failed.")

    img_h, img_w = normalize_imgsz(args.imgsz, dataset=args.dataset)
    model.gcs_imgsz = (img_h, img_w)
    x = torch.randn(args.batch, 3, img_h, img_w, device=args.device)

    model.train()
    with torch.no_grad():
        y = model(x)

    expected = {"pred_points", "pred_logits", "pred_valid_logits", "aux_mask_logits", "aux_edge_logits"}
    if not isinstance(y, dict) or not expected.issubset(y):
        raise RuntimeError("Unexpected GCSLaneHead output keys.")
    if y["pred_valid_logits"].shape != y["pred_points"].shape[:3]:
        raise RuntimeError(
            "pred_valid_logits must have shape B x Q x K matching pred_points, "
            f"got {tuple(y['pred_valid_logits'].shape)} vs {tuple(y['pred_points'].shape[:3])}."
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
        print(f"registered GCSLaneHead: {has_head}")
        print(f"GCSLaneHead point_mode: {getattr(head, 'point_mode', None)}")
        print(f"GCSLaneHead point_dims: {getattr(head, 'point_dims', None)}")

    print(type(y))
    for k, v in y.items():
        print(k, v.shape if hasattr(v, "shape") else v)


if __name__ == "__main__":
    main()
