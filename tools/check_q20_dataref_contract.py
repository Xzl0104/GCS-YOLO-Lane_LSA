"""Check the Q20 dataref fixed-y GCSLaneHead contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.gcs_lane import GCSLaneHead, Q20_DATAREF_X
from ultralytics.nn.tasks import GCSLaneModel


DEFAULT_CFG = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q20-k56-dataref.yaml"
EXPECTED_Y = torch.linspace(710 / 720, 160 / 720, 56)
EXPECTED_NORMAL_BOTTOM = torch.linspace(0.05, 0.95, 12)
EXPECTED_NORMAL_TOP = 0.5 + (EXPECTED_NORMAL_BOTTOM - 0.5) * 0.25


def parse_args() -> argparse.Namespace:
    """Parse dataref contract arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="Q20 dataref model YAML.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for optional forward check.")
    parser.add_argument("--device", default="cpu", help="Torch device for optional forward check.")
    parser.add_argument("--skip-forward", action="store_true", help="Only check the head reference tensors.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve repo-relative paths."""
    return path if path.is_absolute() else ROOT / path


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
    """Assert Q20 dataref references and fixed-y anchors."""
    ref = torch.sigmoid(head.point_reference_logits.detach().cpu())
    fixed_y = head.fixed_y_anchors.detach().cpu()
    dataref = torch.tensor(Q20_DATAREF_X, dtype=torch.float32)

    assert int(head.num_queries) == 20, head.num_queries
    assert int(head.num_points) == 56, head.num_points
    assert head.point_mode == "fixed_y", head.point_mode
    assert head.reference_mode == "dataref", head.reference_mode
    assert tuple(ref.shape) == (20, 56), ref.shape
    assert tuple(dataref.shape) == (20, 56), dataref.shape
    assert tuple(fixed_y.shape) == (56,), fixed_y.shape
    assert torch.isfinite(ref).all()
    assert float(ref.min()) > 0.0 and float(ref.max()) < 1.0
    assert torch.allclose(ref, dataref, atol=2e-6), (ref[:, 0], dataref[:, 0])

    assert torch.all(ref[:4, 0] < 0.35), ref[:4, 0]
    assert torch.all(ref[-4:, 0] > 0.65), ref[-4:, 0]
    assert torch.allclose(ref[4:16, 0], EXPECTED_NORMAL_BOTTOM, atol=2e-6), ref[4:16, 0]
    assert torch.allclose(ref[4:16, -1], EXPECTED_NORMAL_TOP, atol=2e-6), ref[4:16, -1]
    assert torch.allclose(fixed_y, EXPECTED_Y, atol=1e-6), fixed_y


def expect_value_error(**kwargs) -> None:
    """Assert that invalid GCSLaneHead reference-mode combinations fail early."""
    try:
        GCSLaneHead(
            c1=128,
            num_points=56,
            num_decoder_layers=3,
            nhead=8,
            aux=True,
            fixed_y_start=710 / 720,
            fixed_y_end=160 / 720,
            **kwargs,
        )
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError for GCSLaneHead args: {kwargs}")


def check_reference_mode_guards() -> None:
    """Assert dataref/sidegeom modes cannot silently fall back on non-Q20 heads."""
    expect_value_error(num_queries=18, point_mode="fixed_y", reference_mode="dataref")
    expect_value_error(num_queries=12, point_mode="fixed_y", reference_mode="dataref")
    expect_value_error(num_queries=18, point_mode="fixed_y", reference_mode="sidegeom")
    expect_value_error(num_queries=20, point_mode="free", reference_mode="dataref")

    dataref = GCSLaneHead(
        c1=128,
        num_queries=20,
        num_points=56,
        num_decoder_layers=3,
        nhead=8,
        aux=True,
        point_mode="fixed_y",
        fixed_y_start=710 / 720,
        fixed_y_end=160 / 720,
        reference_mode="dataref",
    )
    check_head_reference(dataref)

    sidegeom = GCSLaneHead(
        c1=128,
        num_queries=20,
        num_points=56,
        num_decoder_layers=3,
        nhead=8,
        aux=True,
        point_mode="fixed_y",
        fixed_y_start=710 / 720,
        fixed_y_end=160 / 720,
        reference_mode="sidegeom",
    )
    assert int(sidegeom.num_queries) == 20
    assert sidegeom.reference_mode == "sidegeom"


def check_model_forward(cfg: Path, batch: int, device: str) -> None:
    """Build the Q20 dataref model and assert primary output shapes."""
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
    }
    for key, shape in expected_shapes.items():
        assert key in out, out.keys()
        assert tuple(out[key].shape) == shape, (key, out[key].shape, shape)
        assert torch.isfinite(out[key]).all(), key

    anchors = head.fixed_y_anchors.to(device=out["pred_points"].device, dtype=out["pred_points"].dtype)
    y_err = (out["pred_points"][..., 1] - anchors.view(1, 1, -1)).abs().max()
    assert float(y_err.detach().cpu()) <= 1e-6, float(y_err.detach().cpu())


def main() -> None:
    """Run all Q20 dataref contract assertions."""
    args = parse_args()
    cfg = resolve_path(args.cfg)
    if not cfg.exists():
        raise FileNotFoundError(f"Missing Q20 dataref model YAML: {cfg}")

    check_reference_mode_guards()

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
        reference_mode="dataref",
    )
    check_head_reference(head)

    if not args.skip_forward:
        check_model_forward(cfg, batch=int(args.batch), device=str(args.device))

    ref = torch.sigmoid(head.point_reference_logits.detach().cpu())
    print("OK: Q20 dataref fixed-y contract is correct.")
    print("point_reference_logits:", tuple(head.point_reference_logits.shape))
    print("bottom_x:", [round(float(x), 4) for x in ref[:, 0]])
    print("top_x:", [round(float(x), 4) for x in ref[:, -1]])
    print("fixed_y:", round(float(head.fixed_y_anchors[0]), 6), "->", round(float(head.fixed_y_anchors[-1]), 6))


if __name__ == "__main__":
    main()
