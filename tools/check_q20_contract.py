"""Check the Q20 K56 side-geometry fixed-y GCSLaneHead and model contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.gcs_lane import GCSLaneHead
from ultralytics.nn.tasks import GCSLaneModel


DEFAULT_CFG = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q20-k56-sidegeom.yaml"
EXPECTED_BOTTOM_X = torch.tensor(
    [
        0.01,
        0.03,
        0.06,
        0.10,
        *torch.linspace(0.05, 0.95, 12).tolist(),
        0.90,
        0.94,
        0.97,
        0.99,
    ],
    dtype=torch.float32,
)
EXPECTED_TOP_PULL = torch.tensor(
    [
        0.65,
        0.55,
        0.45,
        0.35,
        *torch.full((12,), 0.25).tolist(),
        0.35,
        0.45,
        0.55,
        0.65,
    ],
    dtype=torch.float32,
)
EXPECTED_TOP_X = 0.5 + (EXPECTED_BOTTOM_X - 0.5) * EXPECTED_TOP_PULL
EXPECTED_Y = torch.linspace(710 / 720, 160 / 720, 56)


def parse_args() -> argparse.Namespace:
    """Parse Q20 contract check arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="Q20 model YAML.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for the forward shape check.")
    parser.add_argument("--device", default="cpu", help="Torch device for the forward shape check.")
    parser.add_argument("--skip-forward", action="store_true", help="Only check the head reference tensors.")
    return parser.parse_args()


def find_gcs_head(model: torch.nn.Module) -> GCSLaneHead:
    """Return the single GCSLaneHead from a model."""
    for module in model.modules():
        if isinstance(module, GCSLaneHead):
            return module
    raise RuntimeError("GCSLaneHead not found")


def unwrap_output(out):
    """Return the GCS output dict from common model output wrappers."""
    if isinstance(out, dict):
        return out
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, dict):
                return item
    raise TypeError(f"Unsupported model output type: {type(out).__name__}")


def check_head_reference(head: GCSLaneHead) -> None:
    """Assert the Q20 side-geometry reference and fixed-y anchors."""
    ref = torch.sigmoid(head.point_reference_logits.detach().cpu())
    fixed_y = head.fixed_y_anchors.detach().cpu()

    assert int(head.num_queries) == 20, head.num_queries
    assert int(head.num_points) == 56, head.num_points
    assert head.point_mode == "fixed_y", head.point_mode
    assert tuple(ref.shape) == (20, 56), ref.shape
    assert tuple(fixed_y.shape) == (56,), fixed_y.shape
    assert torch.allclose(ref[:, 0], EXPECTED_BOTTOM_X, atol=1e-6), ref[:, 0]
    assert torch.allclose(ref[:, -1], EXPECTED_TOP_X, atol=1e-6), ref[:, -1]
    assert torch.allclose(fixed_y, EXPECTED_Y, atol=1e-6), fixed_y

    assert ref[0, -1] < 0.25, ref[0, -1]
    assert ref[-1, -1] > 0.75, ref[-1, -1]
    assert ref[4:16].shape == (12, 56), ref[4:16].shape


def check_model_forward(cfg: Path, batch: int, device: str) -> None:
    """Build the Q20 model and assert primary output shapes."""
    model = GCSLaneModel(str(cfg), nc=1, ch=3, verbose=False).to(device)
    model.train()
    head = find_gcs_head(model)
    check_head_reference(head)

    x = torch.zeros(int(batch), 3, 544, 960, device=device)
    with torch.no_grad():
        out = unwrap_output(model(x))

    expected_shapes = {
        "pred_points": (int(batch), 20, 56, 2),
        "pred_logits": (int(batch), 20),
        "pred_valid_logits": (int(batch), 20, 56),
        "pred_count_logits": (int(batch), 3),
        "pred_reference_x": (int(batch), 20, 56),
    }
    for key, shape in expected_shapes.items():
        assert key in out, out.keys()
        assert tuple(out[key].shape) == shape, (key, out[key].shape, shape)
        assert torch.isfinite(out[key]).all(), key
    assert out.get("reference_mode") != "dataref", out.get("reference_mode")
    assert "is_dataref_reference" in out, out.keys()
    assert float(out["is_dataref_reference"].detach().cpu().item()) == 0.0, out["is_dataref_reference"]

    anchors = head.fixed_y_anchors.to(device=out["pred_points"].device, dtype=out["pred_points"].dtype)
    y_err = (out["pred_points"][..., 1] - anchors.view(1, 1, -1)).abs().max()
    assert float(y_err.detach().cpu()) <= 1e-6, float(y_err.detach().cpu())
    ref_err = (
        out["pred_reference_x"].detach().cpu()
        - torch.sigmoid(head.point_reference_logits.detach().cpu()).view(1, 20, 56)
    ).abs().max()
    assert float(ref_err) <= 1e-6, float(ref_err)


def main() -> None:
    """Run all Q20 contract assertions."""
    args = parse_args()
    cfg = args.cfg if args.cfg.is_absolute() else ROOT / args.cfg
    if not cfg.exists():
        raise FileNotFoundError(f"Missing Q20 model YAML: {cfg}")

    head = GCSLaneHead(
        c1=128,
        num_queries=20,
        num_points=56,
        num_decoder_layers=3,
        nhead=8,
        aux=True,
        point_mode="fixed_y",
        fixed_y_start=710 / 720,
        fixed_y_end=160 / 720,
    )
    check_head_reference(head)

    if not args.skip_forward:
        check_model_forward(cfg, batch=int(args.batch), device=str(args.device))

    ref = torch.sigmoid(head.point_reference_logits.detach().cpu())
    print("OK: Q20 side-geometry fixed-y contract is correct.")
    print("point_reference_logits:", tuple(head.point_reference_logits.shape))
    print("bottom_x:", [round(float(x), 4) for x in ref[:, 0]])
    print("top_x:", [round(float(x), 4) for x in ref[:, -1]])
    print("fixed_y:", round(float(head.fixed_y_anchors[0]), 6), "->", round(float(head.fixed_y_anchors[-1]), 6))


if __name__ == "__main__":
    main()
