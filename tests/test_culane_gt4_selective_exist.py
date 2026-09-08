from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.gcs_loss import GCSLoss


def _criterion(**overrides):
    args = {"gcs_imgsz": [384, 960], **overrides}
    return GCSLoss(args)


def test_query_loss_names_are_only_five_terms():
    assert GCSLoss.loss_names == (
        "exist_loss",
        "point_loss",
        "point_valid_loss",
        "curve_loss",
        "visible_line_iou_loss",
    )


def test_hard_existence_bce_matches_hungarian_targets():
    pred_logits = torch.tensor([[-2.0, 0.5, 1.0]], dtype=torch.float32)
    indices = [(torch.tensor([1]), torch.tensor([0]))]
    loss = _criterion().exist_loss(pred_logits, indices)

    target = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    expected = F.binary_cross_entropy_with_logits(pred_logits, target)
    assert torch.allclose(loss, expected)


def test_hard_existence_ignores_legacy_quality_and_rescue_args():
    pred_logits = torch.tensor([[-1.0, 2.0]], dtype=torch.float32)
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    baseline = _criterion().exist_loss(pred_logits, indices)
    legacy_args = _criterion(
        gcs_exist_quality_alpha=1.0,
        gcs_exist_quality_pos_px=1.0,
        gcs_exist_quality_neg_px=2.0,
        gcs_gt4_selective_exist_target_floor=0.9,
        gcs_gt4_selective_exist_target_ape_max=1.0,
        gcs_gt4_selective_exist_target_visible_thr=1,
    ).exist_loss(pred_logits, indices)
    assert torch.equal(baseline, legacy_args)


def test_hard_existence_gradient_lifts_matched_and_suppresses_unmatched():
    pred_logits = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss = _criterion().exist_loss(pred_logits, indices)
    loss.backward()

    assert pred_logits.grad is not None
    assert float(pred_logits.grad[0, 0]) < 0.0
    assert float(pred_logits.grad[0, 1]) > 0.0


def test_legacy_loss_gains_do_not_change_total_or_loss_vector():
    k, q = 4, 2
    anchors = torch.linspace(589.0 / 590.0, 39.0 / 590.0, k)
    gt_lane = torch.zeros((1, k, 2), dtype=torch.float32)
    gt_lane[0, :, 0] = torch.linspace(0.45, 0.55, k)
    gt_lane[0, :, 1] = anchors
    gt_valid = torch.ones((1, k), dtype=torch.float32)
    pred_points = torch.zeros((1, q, k, 2), dtype=torch.float32)
    pred_points[0, 0] = gt_lane[0]
    pred_points[0, 0, :, 0] += 0.005
    pred_points[0, 1, :, 0] = 0.1
    pred_points[0, :, :, 1] = anchors.view(1, k)
    preds = {
        "pred_points": pred_points,
        "pred_logits": torch.zeros((1, q), dtype=torch.float32),
        "pred_valid_logits": torch.zeros((1, q, k), dtype=torch.float32),
        "pred_count_logits": torch.zeros((1, 5), dtype=torch.float32),
        "aux_mask_logits": torch.randn(1, 2, 384, 960),
        "aux_edge_logits": torch.randn(1, 1, 384, 960),
    }
    batch = {
        "lanes": [gt_lane],
        "lane_valid": [gt_valid],
        "semantic_mask": torch.zeros((1, 384, 960), dtype=torch.long),
        "edge_mask": torch.zeros((1, 384, 960), dtype=torch.float32),
    }

    baseline_total, baseline_items = _criterion()(preds, batch)
    legacy_total, legacy_items = _criterion(
        gcs_smooth=100.0,
        gcs_mask=100.0,
        gcs_edge=100.0,
        gcs_count=100.0,
        gcs_count_under5=100.0,
        gcs_count_boundary=100.0,
        gcs_query_count_ce=100.0,
        gcs_spurious_neg=100.0,
        gcs_boundary_pseudo_neg=100.0,
        gcs_exist_quality_alpha=1.0,
        gcs_gt4_selective_exist_target_floor=0.9,
    )(preds, batch)

    assert int(legacy_items.numel()) == 5
    assert torch.equal(baseline_items, legacy_items)
    assert torch.equal(baseline_total, legacy_total)
