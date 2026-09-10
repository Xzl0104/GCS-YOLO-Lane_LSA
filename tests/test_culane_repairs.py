"""CPU tests for the explicit CULane R1/R2 loss ablations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.gcs_loss import GCSLoss


def _lane_case():
    pred_points = torch.tensor(
        [[[[0.20, 0.90], [0.25, 0.70], [0.30, 0.50], [0.35, 0.30]]]],
        dtype=torch.float32,
    )
    gt_points = [pred_points[0].clone()]
    gt_valid = [torch.ones((1, 4), dtype=torch.bool)]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return pred_points, gt_points, gt_valid, indices


def test_r1_disabled_preserves_coordinate_only_loss():
    pred_points, gt_points, gt_valid, indices = _lane_case()
    criterion = GCSLoss({"gcs_imgsz": [384, 960], "gcs_line_iou_visibility": False})
    exact = criterion.visible_line_iou_loss(pred_points, gt_points, gt_valid, indices)
    short_visibility = criterion.visible_line_iou_loss(
        pred_points,
        gt_points,
        gt_valid,
        indices,
        pred_valid_logits=torch.tensor([[[-8.0, -8.0, 8.0, 8.0]]]),
    )
    assert torch.allclose(exact, torch.zeros_like(exact))
    assert torch.allclose(short_visibility, exact)


def test_r1_soft_iou_penalizes_short_predicted_visibility_and_backpropagates():
    pred_points, gt_points, gt_valid, indices = _lane_case()
    criterion = GCSLoss({"gcs_imgsz": [384, 960], "gcs_line_iou_visibility": True})
    full_logits = torch.full((1, 1, 4), 8.0, requires_grad=True)
    short_logits = torch.tensor([[[-8.0, -8.0, 8.0, 8.0]]], requires_grad=True)

    full_loss = criterion.visible_line_iou_loss(
        pred_points, gt_points, gt_valid, indices, pred_valid_logits=full_logits
    )
    short_loss = criterion.visible_line_iou_loss(
        pred_points, gt_points, gt_valid, indices, pred_valid_logits=short_logits
    )
    assert short_loss > full_loss
    short_loss.backward()
    assert short_logits.grad is not None
    assert torch.isfinite(short_logits.grad).all()
    assert short_logits.grad.abs().sum() > 0


def test_r2_quality_target_is_detached_and_unmatched_remains_zero():
    criterion = GCSLoss({"gcs_imgsz": [384, 960], "gcs_exist_region_quality": True})
    pred_logits = torch.zeros((1, 2), requires_grad=True)
    matched_quality = torch.tensor([[0.25, 0.9]], requires_grad=True)
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss = criterion.exist_loss(pred_logits, indices, matched_quality=matched_quality)
    expected = F.binary_cross_entropy_with_logits(
        pred_logits,
        torch.tensor([[0.25, 0.0]]),
        reduction="mean",
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert matched_quality.grad is None
    assert pred_logits.grad is not None
    assert torch.isfinite(pred_logits.grad).all()


def test_r1_r2_forward_keeps_five_loss_items():
    pred_points, gt_points, gt_valid, _ = _lane_case()
    pred_points = pred_points.repeat(1, 2, 1, 1).requires_grad_(True)
    pred_logits = torch.tensor([[0.0, -1.0]], requires_grad=True)
    pred_valid_logits = torch.full((1, 2, 4), 8.0, requires_grad=True)
    criterion = GCSLoss(
        {
            "gcs_imgsz": [384, 960],
            "gcs_line_iou_visibility": True,
            "gcs_exist_region_quality": True,
        }
    )
    criterion.matcher = lambda points, logits, lanes, valid: [
        (torch.tensor([0]), torch.tensor([0]))
    ]
    batch = {
        "lanes": gt_points,
        "lane_valid": gt_valid,
    }
    total, items = criterion(
        {
            "pred_points": pred_points,
            "pred_logits": pred_logits,
            "pred_valid_logits": pred_valid_logits,
        },
        batch,
    )
    assert items.shape == (5,)
    assert torch.isfinite(total)
    assert torch.isfinite(items).all()
