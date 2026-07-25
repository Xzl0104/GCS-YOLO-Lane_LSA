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
from ultralytics.utils.gcs_fixed_y import validate_training_fixed_y_desc
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

    expected = {"pred_points", "pred_logits", "pred_valid_logits"}
    if getattr(head, "aux", False):
        expected |= {"aux_mask_logits", "aux_edge_logits"}
    if not isinstance(y, dict) or not expected.issubset(y):
        raise RuntimeError("Unexpected GCSLaneHead output keys.")
    gcs_mode = str(getattr(head, "gcs_mode", "query"))
    expected_q = int(getattr(head, "num_queries", 0))
    expected_k = int(getattr(head, "num_points", 0))
    expected_points_shape = (args.batch, expected_q, expected_k, 2)
    if tuple(y["pred_points"].shape) != expected_points_shape:
        raise RuntimeError(f"pred_points must have shape {expected_points_shape}, got {tuple(y['pred_points'].shape)}.")
    if tuple(y["pred_logits"].shape) != (args.batch, expected_q):
        raise RuntimeError(f"pred_logits must have shape {(args.batch, expected_q)}, got {tuple(y['pred_logits'].shape)}.")
    if tuple(y["pred_valid_logits"].shape) != (args.batch, expected_q, expected_k):
        raise RuntimeError(
            f"pred_valid_logits must have shape {(args.batch, expected_q, expected_k)}, "
            f"got {tuple(y['pred_valid_logits'].shape)}."
        )
    if getattr(head, "aux", False):
        if tuple(y["aux_mask_logits"].shape) != (args.batch, 2, img_h, img_w):
            raise RuntimeError(
                f"aux_mask_logits must have shape {(args.batch, 2, img_h, img_w)}, got {tuple(y['aux_mask_logits'].shape)}."
            )
        if tuple(y["aux_edge_logits"].shape) != (args.batch, 1, img_h, img_w):
            raise RuntimeError(
                f"aux_edge_logits must have shape {(args.batch, 1, img_h, img_w)}, got {tuple(y['aux_edge_logits'].shape)}."
            )
    if gcs_mode == "ordered_slot":
        ordered_expected = {"pred_exist_logits", "pred_start_logits", "pred_end_logits", "pred_count_logits"}
        if not ordered_expected.issubset(y):
            raise RuntimeError(f"ordered_slot output is missing keys: {sorted(ordered_expected - set(y))}.")
        expected_slots = int(getattr(head, "num_slots", 5))
        expected_k = int(getattr(head, "num_points", 56))
        if tuple(y["pred_points"].shape[1:3]) != (expected_slots, expected_k):
            raise RuntimeError(
                f"ordered_slot pred_points must be B x {expected_slots} x {expected_k} x 2, "
                f"got {tuple(y['pred_points'].shape)}."
            )
        if y["pred_start_logits"].shape != y["pred_points"].shape[:3]:
            raise RuntimeError("ordered_slot pred_start_logits must have shape B x 5 x K.")
        if y["pred_end_logits"].shape != y["pred_points"].shape[:3]:
            raise RuntimeError("ordered_slot pred_end_logits must have shape B x 5 x K.")
        expected_count_classes = int(getattr(head, "count_classes", 4))
        if tuple(y["pred_count_logits"].shape) != (args.batch, expected_count_classes):
            raise RuntimeError(
                f"ordered_slot pred_count_logits must have shape B x {expected_count_classes}, "
                f"got {tuple(y['pred_count_logits'].shape)}."
            )
    elif getattr(head, "query_count_head", False):
        expected_count_classes = int(getattr(head, "count_classes", 4))
        if tuple(y.get("pred_count_logits", torch.empty(0)).shape) != (args.batch, expected_count_classes):
            raise RuntimeError(
                f"query Count Head pred_count_logits must have shape B x {expected_count_classes}, "
                f"got {tuple(y.get('pred_count_logits', torch.empty(0)).shape)}."
            )
    elif "pred_count_logits" in y:
        raise RuntimeError("default query GCSLaneHead must not emit pred_count_logits.")
    if getattr(head, "query_quality_head", False):
        if tuple(y.get("pred_quality_logits", torch.empty(0)).shape) != (args.batch, expected_q):
            raise RuntimeError(
                f"query Quality Head pred_quality_logits must have shape B x {expected_q}, "
                f"got {tuple(y.get('pred_quality_logits', torch.empty(0)).shape)}."
            )
    elif "pred_quality_logits" in y:
        raise RuntimeError("default query GCSLaneHead must not emit pred_quality_logits.")
    if y["pred_valid_logits"].shape != y["pred_points"].shape[:3]:
        raise RuntimeError(
            "pred_valid_logits must have shape B x Q x K matching pred_points, "
            f"got {tuple(y['pred_valid_logits'].shape)} vs {tuple(y['pred_points'].shape[:3])}."
        )
    if getattr(head, "point_mode", "free") == "fixed_y":
        if int(getattr(head, "point_dims", 2)) != 1:
            raise RuntimeError("fixed_y GCSLaneHead must use point_dims=1 for x-only prediction.")
        anchors = head.fixed_y_anchors.to(device=y["pred_points"].device, dtype=y["pred_points"].dtype)
        validate_training_fixed_y_desc(anchors.detach().cpu().numpy(), name="check_model fixed_y anchors")
        y_pred = y["pred_points"][..., 1]
        max_y_err = float((y_pred - anchors.view(1, 1, -1)).abs().max().cpu().item())
        if max_y_err > 1e-6:
            raise RuntimeError(f"fixed_y GCSLaneHead produced non-anchor y coordinates, max error={max_y_err:.6g}.")
        final = head.point_mlp[-1]
        if getattr(final, "out_features", None) != head.num_points:
            raise RuntimeError(
                f"fixed_y GCSLaneHead point MLP must output K x values, got out_features={final.out_features}."
            )
    if getattr(head, "reference_mode", "linear") == "dualbank":
        if expected_q not in {20, 24}:
            raise RuntimeError("dualbank GCSLaneHead must use num_queries=20 or 24.")
        if getattr(head, "point_mode", None) != "fixed_y":
            raise RuntimeError("dualbank GCSLaneHead must use point_mode='fixed_y'.")
        refs = torch.sigmoid(head.point_reference_logits.detach().float().cpu())
        if tuple(refs.shape) != (expected_q, expected_k):
            raise RuntimeError(
                f"dualbank point references must have shape {expected_q} x {expected_k}, got {tuple(refs.shape)}."
            )
        t = torch.linspace(0.0, 1.0, expected_k)
        bottom_x = torch.linspace(0.05, 0.95, 12)
        top_x = 0.5 + (bottom_x - 0.5) * 0.25
        q12_refs = bottom_x[:, None] * (1.0 - t[None, :]) + top_x[:, None] * t[None, :]
        max_ref_err = float((refs[:12] - q12_refs).abs().max().item())
        if max_ref_err > 1e-6:
            raise RuntimeError(f"dualbank must preserve Q12 references in q0..q11, max error={max_ref_err:.6g}.")

    if args.detailed:
        print(f"task: {yolo.task}")
        print(f"model: {type(model).__name__}")
        print(f"input shape: {shape_str((img_h, img_w))} (W x H), stored as H,W={(img_h, img_w)}")
        print(f"registered LSEM: {has_lsem}")
        print(f"registered LaneBiFPN: {has_bifpn}")
        print(f"registered GCSLaneHead: {has_head}")
        print(f"GCSLaneHead point_mode: {getattr(head, 'point_mode', None)}")
        print(f"GCSLaneHead gcs_mode: {getattr(head, 'gcs_mode', None)}")
        print(f"GCSLaneHead point_dims: {getattr(head, 'point_dims', None)}")
        print(f"GCSLaneHead reference_mode: {getattr(head, 'reference_mode', None)}")

    print(type(y))
    for k, v in y.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
