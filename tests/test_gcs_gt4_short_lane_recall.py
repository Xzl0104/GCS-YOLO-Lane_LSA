from types import SimpleNamespace

import torch

from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer
from ultralytics.utils.gcs_loss import GCSLoss
from ultralytics.utils.gcs_matcher import GCSHungarianMatcher


def _fixed_y_points(num_lanes: int, k: int) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, k)
    points = torch.zeros(num_lanes, k, 2)
    points[..., 1] = y
    return points


def test_forward_logs_lane_balanced_and_gt4_short_point_items():
    k = 56
    gt = _fixed_y_points(4, k)
    gt[:, :, 0] = torch.tensor([0.05, 0.35, 0.65, 0.95]).view(4, 1)
    pred = gt.clone()
    valid = torch.zeros(4, k)
    valid[0, :5] = 1.0
    valid[1:, :40] = 1.0

    pred[0, :5, 0] += 0.02
    pred[1:, :40, 0] += 0.001
    preds = {
        "pred_points": pred.unsqueeze(0),
        "pred_logits": torch.ones(1, 4),
    }
    batch = {
        "lanes": [gt],
        "lane_valid": [valid],
        "num_lanes": torch.tensor([4]),
    }
    loss_fn = GCSLoss(
        model={
            "gcs_imgsz": [544, 960],
            "gcs_lane_balanced_point_loss": True,
            "gcs_gt4_short_lane_weight": 1.5,
            "gcs_match_gate_px": 0.0,
            "gcs_cost_curve": 0.0,
            "gcs_cost_exist": 0.0,
        }
    )

    _, items = loss_fn(preds, batch)
    item_map = dict(zip(loss_fn.loss_names, items))

    assert loss_fn.loss_names[2] == "lane_balanced_point_loss"
    assert loss_fn.loss_names[3] == "gt4_short_lane_loss"
    assert loss_fn.loss_names[4] == "gt4_lane_balanced_point_loss"
    assert item_map["lane_balanced_point_loss"] > 0.0
    assert item_map["gt4_short_lane_loss"] > 0.0
    assert torch.isclose(item_map["point_loss"], item_map["lane_balanced_point_loss"])
    assert torch.isclose(item_map["gt4_short_lane_loss"], torch.tensor(0.03), atol=1e-6)


def test_gt4_lbp_item_is_separate_gain_weighted_contribution():
    k = 56
    gt = _fixed_y_points(4, k)
    gt[:, :, 0] = torch.tensor([0.05, 0.35, 0.65, 0.95]).view(4, 1)
    pred = gt.clone()
    pred[0, :, 0] += 0.10
    pred[1:, :, 0] += 0.01
    valid = torch.ones(4, k)
    preds = {
        "pred_points": pred.unsqueeze(0),
        "pred_logits": torch.ones(1, 4),
    }
    batch = {
        "lanes": [gt],
        "lane_valid": [valid],
        "num_lanes": torch.tensor([4]),
    }
    loss_fn = GCSLoss(
        model={
            "gcs_imgsz": [544, 960],
            "gcs_exist": 0.0,
            "gcs_point": 0.0,
            "gcs_lane_balanced_point": 0.0,
            "gcs_gt4_lane_balanced_point": 0.25,
            "gcs_gt4_lane_balanced_topk": 1,
            "gcs_gt4_lane_balanced_max_mult": 2.0,
            "gcs_point_valid": 0.0,
            "gcs_short_valid_recall": 0.0,
            "gcs_smooth": 0.0,
            "gcs_curve": 0.0,
            "gcs_mask": 0.0,
            "gcs_edge": 0.0,
            "gcs_count": 0.0,
            "gcs_count_under5": 0.0,
            "gcs_duplicate_margin": 0.0,
            "gcs_spurious_margin": 0.0,
            "gcs_far_spurious_survival": 0.0,
            "gcs_gt5_rank_consistency": 0.0,
            "gcs_gt3_extra_survival": 0.0,
            "gcs_match_gate_px": 0.0,
            "gcs_cost_curve": 0.0,
            "gcs_cost_exist": 0.0,
        }
    )

    total, items = loss_fn(preds, batch)
    item_map = dict(zip(loss_fn.loss_names, items))

    assert item_map["gt4_lane_balanced_point_loss"] > 0.0
    assert torch.isclose(total, 0.25 * item_map["gt4_lane_balanced_point_loss"], atol=1e-6)


def test_progress_header_marks_disabled_experiment_columns():
    trainer = object.__new__(GCSLaneTrainer)
    trainer.args = SimpleNamespace(gcs_gt4_lane_balanced_point=0.0, gcs_short_valid_recall=0.0)
    disabled_header = trainer.progress_string()

    assert "gt4lbp_off" in disabled_header
    assert "short_off" in disabled_header

    trainer.args = SimpleNamespace(gcs_gt4_lane_balanced_point=0.25, gcs_short_valid_recall=0.5)
    enabled_header = trainer.progress_string()

    assert "gt4_lbp" in enabled_header
    assert "short_rec" in enabled_header
    assert "gt4lbp_off" not in enabled_header
    assert "short_off" not in enabled_header


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
