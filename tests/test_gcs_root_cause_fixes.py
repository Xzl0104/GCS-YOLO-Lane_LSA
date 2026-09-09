from __future__ import annotations

import torch

from pathlib import Path

from ultralytics.nn.tasks import GCSLaneModel
from ultralytics.utils.gcs_loss import GCSLoss


def test_visible_line_iou_keeps_gradient_for_far_error() -> None:
    criterion = GCSLoss({"gcs_imgsz": [384, 960], "gcs_line_iou_width_px": 18.0})
    gt = torch.zeros((1, 4, 2), dtype=torch.float32)
    gt[..., 0] = 0.5
    gt[..., 1] = torch.linspace(589.0 / 590.0, 39.0 / 590.0, 4)
    pred = gt.unsqueeze(0).clone()
    pred[..., 0] += 50.0 / 960.0
    pred.requires_grad_(True)

    loss = criterion.visible_line_iou_loss(
        pred,
        [gt],
        [torch.ones((1, 4), dtype=torch.bool)],
        [(torch.tensor([0]), torch.tensor([0]))],
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert float(pred.grad[..., 0].abs().sum()) > 0.0


def test_proposal_state_refinement_uses_bounded_residual_scale() -> None:
    root = Path(__file__).resolve().parents[1]
    model = GCSLaneModel(
        str(root / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-culane.yaml"),
        nc=1,
        verbose=False,
    )
    head = model.model[-1]
    assert bool(head.proposal_state_refine)
    assert 0.0 < float(head.proposal_state_delta_scale) < 1.0
