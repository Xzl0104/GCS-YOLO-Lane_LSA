"""Check the Q18 K56 fixed-y GCSLaneHead reference contract."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.gcs_lane import GCSLaneHead


EXPECTED_BOTTOM_X = torch.tensor(
    [
        0.02,
        0.05,
        0.08,
        0.12,
        0.17,
        0.23,
        0.30,
        0.38,
        0.46,
        0.54,
        0.62,
        0.70,
        0.77,
        0.83,
        0.88,
        0.92,
        0.95,
        0.98,
    ],
    dtype=torch.float32,
)


def main() -> None:
    """Assert Q18 side-dense references and TuSimple K56 y anchors."""
    head = GCSLaneHead(
        c1=128,
        num_queries=18,
        num_points=56,
        num_decoder_layers=3,
        nhead=8,
        aux=True,
        point_mode="fixed_y",
        fixed_y_start=710 / 720,
        fixed_y_end=160 / 720,
    )

    ref = torch.sigmoid(head.point_reference_logits.detach().cpu())
    fixed_y = head.fixed_y_anchors.detach().cpu()
    expected_y = torch.linspace(710 / 720, 160 / 720, 56)

    assert head.num_queries == 18, head.num_queries
    assert head.num_points == 56, head.num_points
    assert head.point_mode == "fixed_y", head.point_mode
    assert ref.shape == (18, 56), ref.shape
    assert fixed_y.shape == (56,), fixed_y.shape
    assert torch.allclose(ref[:, 0], EXPECTED_BOTTOM_X, atol=1e-6), ref[:, 0]
    assert torch.allclose(fixed_y, expected_y, atol=1e-6), fixed_y

    top_x = ref[:, -1]
    assert torch.all((top_x > 0.0) & (top_x < 1.0)), top_x
    assert top_x[0] > ref[0, 0], (top_x[0], ref[0, 0])
    assert top_x[-1] < ref[-1, 0], (top_x[-1], ref[-1, 0])

    print("OK: Q18 fixed-y reference contract is correct.")
    print("point_reference_logits:", tuple(head.point_reference_logits.shape))
    print("bottom_x:", [round(float(x), 4) for x in ref[:, 0]])
    print("top_x:", [round(float(x), 4) for x in top_x])
    print("fixed_y_first_last:", float(fixed_y[0]), float(fixed_y[-1]))


if __name__ == "__main__":
    main()
