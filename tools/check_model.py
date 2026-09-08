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
from ultralytics.utils.gcs_fixed_y import validate_fixed_y_contract, validate_training_fixed_y_desc
from ultralytics.nn.modules import GCSLaneHead, LSEM, LaneBiFPN
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str

DEFAULT_CFG = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q12-k56.yaml"

DATASET_CONTRACTS = {
    "tusimple": {
        "imgsz": (544, 960),
        "fixed_y_original_h": 720,
        "fixed_y_start_px": 710.0,
        "fixed_y_end_px": 160.0,
        "num_points": 56,
        "gcs_modes": {"query", "ordered_slot"},
        "min_lanes": 2,
        "max_lanes": 5,
        "count_classes": 4,
    },
    "culane": {
        "imgsz": (384, 960),
        "fixed_y_original_h": 590,
        "fixed_y_start_px": 589.0,
        "fixed_y_end_px": 39.0,
        "num_points": 56,
        "gcs_modes": {"query"},
        "min_lanes": 0,
        "max_lanes": 4,
        "count_classes": 5,
    },
}


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
    contract = DATASET_CONTRACTS[args.dataset]
    expected_imgsz = tuple(contract["imgsz"])
    if (img_h, img_w) != expected_imgsz:
        raise RuntimeError(
            f"{args.dataset} requires input shape H,W={expected_imgsz}, got {(img_h, img_w)}."
        )
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
    if gcs_mode not in contract["gcs_modes"]:
        raise RuntimeError(
            f"{args.dataset} does not support GCSLaneHead.gcs_mode={gcs_mode!r}; "
            f"allowed modes are {sorted(contract['gcs_modes'])}."
        )
    lane_contract = (
        int(getattr(head, "min_lanes", -1)),
        int(getattr(head, "max_lanes", -1)),
        int(getattr(head, "count_classes", -1)),
    )
    expected_lane_contract = (
        int(contract["min_lanes"]),
        int(contract["max_lanes"]),
        int(contract["count_classes"]),
    )
    if lane_contract != expected_lane_contract:
        raise RuntimeError(
            f"{args.dataset} lane contract must be "
            f"{expected_lane_contract[0]}..{expected_lane_contract[1]} with "
            f"{expected_lane_contract[2]} count classes, got "
            f"{lane_contract[0]}..{lane_contract[1]} with {lane_contract[2]} classes."
        )
    expected_num_points = int(contract["num_points"])
    actual_num_points = int(getattr(head, "num_points", -1))
    if actual_num_points != expected_num_points:
        raise RuntimeError(
            f"{args.dataset} requires GCSLaneHead.num_points={expected_num_points}, "
            f"got {actual_num_points}."
        )
    expected_points_shape = (args.batch, int(getattr(head, "num_queries", 0)), expected_num_points, 2)
    if tuple(y["pred_points"].shape) != expected_points_shape:
        raise RuntimeError(
            f"pred_points must have shape B x Q x K x 2={expected_points_shape}, "
            f"got {tuple(y['pred_points'].shape)}."
        )
    if gcs_mode == "ordered_slot":
        ordered_expected = {"pred_exist_logits", "pred_start_logits", "pred_end_logits", "pred_count_logits"}
        if not ordered_expected.issubset(y):
            raise RuntimeError(f"ordered_slot output is missing keys: {sorted(ordered_expected - set(y))}.")
        expected_slots = int(getattr(head, "num_slots", 5))
        expected_k = expected_num_points
        if tuple(y["pred_points"].shape[1:3]) != (expected_slots, expected_k):
            raise RuntimeError(
                f"ordered_slot pred_points must be B x {expected_slots} x {expected_k} x 2, "
                f"got {tuple(y['pred_points'].shape)}."
            )
        if y["pred_start_logits"].shape != y["pred_points"].shape[:3]:
            raise RuntimeError("ordered_slot pred_start_logits must have shape B x 5 x K.")
        if y["pred_end_logits"].shape != y["pred_points"].shape[:3]:
            raise RuntimeError("ordered_slot pred_end_logits must have shape B x 5 x K.")
        expected_count_classes = int(getattr(head, "count_classes", contract["count_classes"]))
        if tuple(y["pred_count_logits"].shape) != (args.batch, expected_count_classes):
            raise RuntimeError(
                f"ordered_slot pred_count_logits must have shape B x {expected_count_classes}, "
                f"got {tuple(y['pred_count_logits'].shape)}."
            )
    elif bool(getattr(head, "query_count_head", False)):
        if "pred_count_logits" not in y:
            raise RuntimeError("Query Count Head is enabled but pred_count_logits is missing from the model output.")
        expected_count_classes = int(contract["count_classes"])
        if tuple(y["pred_count_logits"].shape) != (args.batch, expected_count_classes):
            raise RuntimeError(
                f"query pred_count_logits must have shape B x {expected_count_classes}, "
                f"got {tuple(y['pred_count_logits'].shape)}."
            )
    elif "pred_count_logits" in y:
        raise RuntimeError(
            "The default query model must not emit pred_count_logits; use the explicit query Count Head YAML "
            "when count supervision is requested."
        )
    if y["pred_valid_logits"].shape != y["pred_points"].shape[:3]:
        raise RuntimeError(
            "pred_valid_logits must have shape B x Q x K matching pred_points, "
            f"got {tuple(y['pred_valid_logits'].shape)} vs {tuple(y['pred_points'].shape[:3])}."
        )
    if getattr(head, "point_mode", "free") == "fixed_y":
        if int(getattr(head, "point_dims", 2)) != 1:
            raise RuntimeError("fixed_y GCSLaneHead must use point_dims=1 for x-only prediction.")
        anchors = head.fixed_y_anchors.to(device=y["pred_points"].device, dtype=y["pred_points"].dtype)
        anchors_np = anchors.detach().cpu().numpy()
        if args.dataset == "tusimple":
            validate_training_fixed_y_desc(
                anchors_np,
                original_h=int(contract["fixed_y_original_h"]),
                name="check_model TuSimple fixed_y anchors",
            )
        else:
            validate_fixed_y_contract(
                anchors_np,
                original_h=int(contract["fixed_y_original_h"]),
                start_px=float(contract["fixed_y_start_px"]),
                end_px=float(contract["fixed_y_end_px"]),
                k=expected_num_points,
                name="check_model CULane fixed_y anchors",
            )
        for field in ("fixed_y_original_h", "fixed_y_start_px", "fixed_y_end_px"):
            actual = float(getattr(head, field))
            expected_value = float(contract[field])
            if abs(actual - expected_value) > 1e-6:
                raise RuntimeError(
                    f"{args.dataset} {field} must be {expected_value:g}, got {actual:g}."
                )
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
        print(f"GCSLaneHead gcs_mode: {getattr(head, 'gcs_mode', None)}")
        print(f"GCSLaneHead point_dims: {getattr(head, 'point_dims', None)}")

    print(type(y))
    for k, v in y.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
