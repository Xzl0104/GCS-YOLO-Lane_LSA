"""Check Q18 K56 model construction and forward output shapes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.gcs_lane import GCSLaneHead
from ultralytics.nn.tasks import GCSLaneModel


CFG = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q18-k56-side.yaml"


def find_gcs_head(model: torch.nn.Module) -> GCSLaneHead:
    """Return the single GCSLaneHead from a constructed model."""
    for module in model.modules():
        if isinstance(module, GCSLaneHead):
            return module
    raise RuntimeError("GCSLaneHead not found.")


def normalize_output(output):
    """Handle direct dict output and tuple/list wrappers."""
    if isinstance(output, dict):
        return output
    if isinstance(output, (list, tuple)):
        dicts = [item for item in output if isinstance(item, dict)]
        if dicts:
            return dicts[0]
    raise TypeError(f"Expected dict-like GCS output, got {type(output)}.")


def main() -> None:
    """Build the Q18 model and assert core output tensor shapes."""
    model = GCSLaneModel(str(CFG), nc=1, ch=3, verbose=False)
    model.eval()

    head = find_gcs_head(model)
    assert head.num_queries == 18, head.num_queries
    assert head.num_points == 56, head.num_points
    assert head.point_mode == "fixed_y", head.point_mode

    x = torch.zeros(1, 3, 544, 960)
    with torch.no_grad():
        output = normalize_output(model(x))

    expected_keys = {"pred_points", "pred_logits", "pred_valid_logits", "pred_count_logits"}
    assert expected_keys.issubset(output), output.keys()
    assert output["pred_points"].shape == (1, 18, 56, 2), output["pred_points"].shape
    assert output["pred_logits"].shape == (1, 18), output["pred_logits"].shape
    assert output["pred_valid_logits"].shape == (1, 18, 56), output["pred_valid_logits"].shape
    assert output["pred_count_logits"].shape == (1, 3), output["pred_count_logits"].shape

    pred_points = output["pred_points"]
    assert torch.isfinite(pred_points).all()
    assert float(pred_points.min()) >= 0.0
    assert float(pred_points.max()) <= 1.0
    assert torch.isfinite(output["pred_count_logits"]).all()

    print("OK: Q18 model forward works.")
    print("pred_points:", tuple(output["pred_points"].shape))
    print("pred_logits:", tuple(output["pred_logits"].shape))
    print("pred_valid_logits:", tuple(output["pred_valid_logits"].shape))
    print("pred_count_logits:", tuple(output["pred_count_logits"].shape))


if __name__ == "__main__":
    main()
