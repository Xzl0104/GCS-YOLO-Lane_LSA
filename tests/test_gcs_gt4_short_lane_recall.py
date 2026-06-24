import torch

from ultralytics.utils.gcs_loss import GCSLoss
from ultralytics.utils.gcs_matcher import GCSHungarianMatcher


def _fixed_y_points(num_lanes: int, k: int) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, k)
    points = torch.zeros(num_lanes, k, 2)
    points[..., 1] = y
    return points


def test_lane_balanced_point_loss_keeps_short_lane_from_valid_point_dilution():
    k = 56
    gt = _fixed_y_points(4, k)
    pred = gt.clone()
    valid = torch.zeros(4, k)
    valid[0, :5] = 1.0
    valid[1:, :40] = 1.0

    pred[0, :5, 0] = 0.20
    pred[1, :40, 0] = 0.01
    pred_points = pred.unsqueeze(0)
    indices = [(torch.arange(4), torch.arange(4))]

    legacy = GCSLoss(model={"gcs_imgsz": [544, 960], "gcs_lane_balanced_point_loss": False})
    balanced = GCSLoss(model={"gcs_imgsz": [544, 960], "gcs_lane_balanced_point_loss": True})

    legacy_loss = legacy.point_loss(pred_points, [gt], [valid], indices)
    balanced_loss, stats = balanced.point_loss(
        pred_points,
        [gt],
        [valid],
        indices,
        target_counts=torch.tensor([4.0]),
        return_stats=True,
    )

    assert balanced_loss > legacy_loss * 4.0
    assert torch.isclose(stats["gt4_short_lane_valid_points_mean"], torch.tensor(5.0))
    assert torch.isclose(stats["gt4_short_lane_count"], torch.tensor(1.0))
    assert torch.isclose(stats["gt4_short_lane_loss"], torch.tensor(0.20), atol=1e-6)


def test_gt4_short_endpoint_cost_only_applies_to_gt4_short_lanes():
    k = 6
    pred = _fixed_y_points(1, k)
    logits = torch.zeros(1)
    gt4 = _fixed_y_points(4, k)
    valid4 = torch.ones(4, k)
    valid4[0, 3:] = 0.0
    gt4[0, 0, 0] = 0.4
    gt4[0, 2, 0] = 0.4
    gt4[1:, :, 0] = 0.1

    base = GCSHungarianMatcher(cost_point=1.0, cost_curve=0.0, cost_exist=0.0, image_size=[544, 960])
    endpoint = GCSHungarianMatcher(
        cost_point=1.0,
        cost_curve=0.0,
        cost_exist=0.0,
        image_size=[544, 960],
        gt4_short_match_endpoint=2.0,
        gt4_short_match_max_points=3,
    )

    delta = endpoint.cost_matrix(pred, logits, gt4, valid4) - base.cost_matrix(pred, logits, gt4, valid4)
    assert delta[0, 0] > 0.0
    assert torch.allclose(delta[0, 1:], torch.zeros(3))

    gt3 = gt4[:3]
    valid3 = valid4[:3]
    delta_non_gt4 = endpoint.cost_matrix(pred, logits, gt3, valid3) - base.cost_matrix(pred, logits, gt3, valid3)
    assert torch.allclose(delta_non_gt4, torch.zeros_like(delta_non_gt4))
