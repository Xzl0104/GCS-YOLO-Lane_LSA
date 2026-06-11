from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image
from ultralytics.nn.modules import GCSLaneHead
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, assert_gcs_shape, normalize_imgsz, shape_str
from ultralytics.utils.torch_utils import select_device


DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_refquery_e220" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "datasets" / "tusimple_fixed_y_960x544" / "images" / "val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether GCSLaneHead uses spatial tokens and image content.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt or model yaml.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image file, directory, or txt list.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS input shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--min-tokens", type=int, default=1000, help="Minimum acceptable P2-P5 spatial token count.")
    parser.add_argument(
        "--fail-under-point-diff",
        type=float,
        default=0.0,
        help="Fail if all real-vs-variant point mean-abs diffs are below this value. 0 disables.",
    )
    return parser.parse_args()


def find_gcs_head(model: torch.nn.Module) -> GCSLaneHead:
    heads = [m for m in model.modules() if isinstance(m, GCSLaneHead)]
    if len(heads) != 1:
        raise RuntimeError(f"Expected exactly one GCSLaneHead, found {len(heads)}.")
    head = heads[0]
    required = (
        "point_embed",
        "point_coord_mlp",
        "point_refine_norm",
        "point_image_norm",
        "point_refine_mlp",
        "point_valid_mlp",
        "point_valid_refine_mlp",
    )
    missing = [name for name in required if not hasattr(head, name)]
    if missing:
        raise RuntimeError(f"GCSLaneHead is missing current fixed-y modules: {missing}. Retrain with the current model YAML.")
    return head


def tensor_mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().mean().cpu().item())


def y_monotonic_summary(points: torch.Tensor) -> dict:
    y = points.detach().float()[..., 1]
    if y.shape[-1] < 2:
        return {"monotonic_fraction": 1.0, "monotonic": 0, "total": 0}
    monotonic = (y[..., :-1] >= y[..., 1:] - 1e-6).all(dim=-1)
    return {
        "monotonic_fraction": round(float(monotonic.float().mean().cpu().item()), 6),
        "monotonic": int(monotonic.sum().cpu().item()),
        "total": int(monotonic.numel()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=imgsz)
    head = find_gcs_head(model)
    head.min_spatial_tokens = int(args.min_tokens)

    image_path = collect_images(args.source, max_images=1)[0]
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    assert_gcs_shape(img.shape[:2], imgsz, name="dependency-check image", context=f"check_gcs_head_dependency({image_path})")

    real = preprocess_image(img, imgsz=imgsz, device=device, half=args.half)
    variants = {
        "zero": torch.zeros_like(real),
        "noise": torch.rand_like(real),
        "flip": torch.flip(real, dims=[3]),
    }

    out_real = model(real)
    if "pred_valid_logits" not in out_real:
        raise RuntimeError("GCSLaneHead output is missing pred_valid_logits.")
    if out_real["pred_valid_logits"].shape != out_real["pred_points"].shape[:3]:
        raise RuntimeError(
            "pred_valid_logits must have shape B x Q x K matching pred_points, "
            f"got {tuple(out_real['pred_valid_logits'].shape)} vs {tuple(out_real['pred_points'].shape[:3])}."
        )
    spatial_debug = getattr(head, "_last_spatial_debug", None)
    if not spatial_debug:
        raise RuntimeError("GCSLaneHead did not record spatial debug metadata.")

    outputs = {name: model(tensor) for name, tensor in variants.items()}
    point_diffs = {
        f"real_{name}": round(tensor_mean_abs(out_real["pred_points"], out["pred_points"]), 8)
        for name, out in outputs.items()
    }
    logit_diffs = {
        f"real_{name}": round(tensor_mean_abs(out_real["pred_logits"], out["pred_logits"]), 8)
        for name, out in outputs.items()
    }
    valid_diffs = {
        f"real_{name}": round(tensor_mean_abs(out_real["pred_valid_logits"], out["pred_valid_logits"]), 8)
        for name, out in outputs.items()
    }

    memory_tokens = int(spatial_debug["memory_shape"][1])
    if memory_tokens < int(args.min_tokens):
        raise AssertionError(f"Too few spatial tokens: {memory_tokens} < {args.min_tokens}.")
    if args.fail_under_point_diff > 0.0 and max(point_diffs.values()) < float(args.fail_under_point_diff):
        raise AssertionError(
            f"All point diffs are below {args.fail_under_point_diff}: {point_diffs}. "
            "The head may be dominated by query/template priors instead of image tokens."
        )

    report = {
        "image": str(image_path.resolve()),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "shape": shape_str(imgsz),
        "head": spatial_debug,
        "point_mean_abs_diff": point_diffs,
        "logit_mean_abs_diff": logit_diffs,
        "point_valid_mean_abs_diff": valid_diffs,
        "real_pred_y_monotonic": y_monotonic_summary(out_real["pred_points"]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
