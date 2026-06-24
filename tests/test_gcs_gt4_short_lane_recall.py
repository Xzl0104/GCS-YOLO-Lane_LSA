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


def test_gt4_short_valid_recall_only_targets_gt4_short_lanes():
    k = 24
    logits = torch.zeros(4, k)
    logits[1:] = -100.0
    valid = torch.zeros(4, k, dtype=torch.bool)
    valid[0, :3] = True
    valid[1, :3] = True
    valid[2, :21] = True
    valid[3, :1] = True
    gt_lane_count = torch.tensor([4, 3, 4, 4])
    loss_fn = GCSLoss(model={"gcs_imgsz": [544, 960]})

    loss, stats = loss_fn.gt4_short_valid_recall_loss(
        logits,
        valid,
        gt_lane_count,
        short_max_points=20,
        min_valid_points=2,
    )

    assert torch.isclose(loss, torch.log(torch.tensor(2.0)), atol=1e-6)
    assert torch.isclose(stats["gt4_short_valid_lane_count"], torch.tensor(1.0))
    assert torch.isclose(stats["gt4_short_gt_valid_points_mean"], torch.tensor(3.0))
    assert torch.isclose(stats["gt4_short_pred_valid_prob_mean"], torch.tensor(0.5))
    assert torch.isclose(stats["gt4_short_pred_valid_sum_mean"], torch.tensor(1.5))


def test_lane_balanced_valid_loss_hits_and_weights_gt4_short_lanes():
    k = 24
    valid = torch.zeros(3, k, dtype=torch.bool)
    valid[0, :4] = True
    valid[1, :22] = True
    valid[2, :4] = True
    gt_lane_count = torch.tensor([4, 4, 3])

    logits = torch.full((3, k), -5.0)
    logits[1, :22] = 5.0
    logits[2, :4] = 5.0
    loss_fn = GCSLoss(model={"gcs_imgsz": [544, 960]})

    unweighted, _ = loss_fn.lane_balanced_valid_loss(
        logits,
        valid,
        gt_lane_count,
        short_max_points=20,
        short_lane_weight=1.0,
        pos_weight=1.0,
    )
    weighted, stats = loss_fn.lane_balanced_valid_loss(
        logits,
        valid,
        gt_lane_count,
        short_max_points=20,
        short_lane_weight=2.0,
        pos_weight=1.0,
    )

    assert weighted > unweighted
    assert torch.isclose(stats["valid_lb_gt4_short_count"], torch.tensor(1.0))
    assert torch.isclose(stats["valid_lb_gt4_short_gt_points_mean"], torch.tensor(4.0))
    assert stats["valid_lb_gt4_short_pred_prob_mean"] < 0.01
    assert stats["valid_lb_gt4_short_pred_sum_mean"] < 0.05


def test_lane_balanced_valid_loss_empty_input_is_zero():
    loss_fn = GCSLoss(model={"gcs_imgsz": [544, 960]})

    loss, stats = loss_fn.lane_balanced_valid_loss(
        torch.zeros(0, 56),
        torch.zeros(0, 56, dtype=torch.bool),
        torch.zeros(0, dtype=torch.long),
    )

    assert torch.isclose(loss, torch.tensor(0.0))
    assert torch.isclose(stats["valid_lb_gt4_short_count"], torch.tensor(0.0))
    assert torch.isclose(stats["valid_lb_gt4_short_pred_sum_mean"], torch.tensor(0.0))


def test_unmatched_valid_negative_loss_penalizes_high_logits_in_forward():
    k = 24
    gt = _fixed_y_points(2, k)
    gt[:, :, 0] = torch.tensor([0.2, 0.8]).view(2, 1)
    pred = _fixed_y_points(4, k)
    pred[:2] = gt
    pred[2:, :, 0] = torch.tensor([0.4, 0.6]).view(2, 1)
    valid = torch.zeros(2, k)
    valid[:, :8] = 1.0
    pred_valid_logits = torch.zeros(1, 4, k)
    pred_valid_logits[:, 2:] = 10.0
    preds = {
        "pred_points": pred.unsqueeze(0),
        "pred_logits": torch.ones(1, 4),
        "pred_valid_logits": pred_valid_logits,
    }
    batch = {
        "lanes": [gt],
        "lane_valid": [valid],
        "num_lanes": torch.tensor([2]),
    }
    loss_fn = GCSLoss(
        model={
            "gcs_imgsz": [544, 960],
            "gcs_lane_balanced_valid_loss": True,
            "gcs_unmatched_valid_neg_weight": 0.5,
            "gcs_exist": 0.0,
            "gcs_point": 0.0,
            "gcs_point_valid": 1.0,
            "gcs_short_valid_recall": 0.0,
            "gcs_smooth": 0.0,
            "gcs_curve": 0.0,
            "gcs_mask": 0.0,
            "gcs_edge": 0.0,
            "gcs_count": 0.0,
            "gcs_count_under5": 0.0,
            "gcs_match_gate_px": 0.0,
            "gcs_cost_curve": 0.0,
            "gcs_cost_exist": 0.0,
        }
    )

    total, items = loss_fn(preds, batch)
    item_map = dict(zip(loss_fn.loss_names, items))

    assert item_map["unmatched_valid_neg_loss"] > 9.0
    assert torch.isclose(item_map["unmatched_valid_query_count"], torch.tensor(2.0))
    assert item_map["unmatched_valid_prob_mean"] > 0.99
    assert item_map["point_valid_loss"] > torch.log(torch.tensor(2.0))
    assert torch.isclose(total, item_map["point_valid_loss"], atol=1e-6)


def test_unmatched_valid_negative_loss_low_logits_is_small():
    logits = torch.full((1, 3, 12), -10.0)
    matched_query_mask = torch.tensor([[True, False, False]])
    loss_fn = GCSLoss(model={"gcs_imgsz": [544, 960]})

    loss, stats = loss_fn.unmatched_valid_negative_loss(logits, matched_query_mask)

    assert loss < 1e-4
    assert torch.isclose(stats["unmatched_valid_query_count"], torch.tensor(2.0))
    assert stats["unmatched_valid_prob_mean"] < 1e-4


def test_unmatched_valid_negative_loss_no_unmatched_is_zero():
    logits = torch.full((1, 2, 12), 10.0)
    matched_query_mask = torch.ones(1, 2, dtype=torch.bool)
    loss_fn = GCSLoss(model={"gcs_imgsz": [544, 960]})

    loss, stats = loss_fn.unmatched_valid_negative_loss(logits, matched_query_mask)

    assert torch.isclose(loss, torch.tensor(0.0))
    assert torch.isclose(stats["unmatched_valid_query_count"], torch.tensor(0.0))
    assert torch.isclose(stats["unmatched_valid_prob_mean"], torch.tensor(0.0))


def test_gt4_short_valid_count_floor_only_penalizes_below_floor():
    k = 24
    valid = torch.zeros(3, k, dtype=torch.bool)
    valid[0, :5] = True
    valid[1, :5] = True
    valid[2, :21] = True
    gt_lane_count = torch.tensor([4, 3, 4])
    loss_fn = GCSLoss(model={"gcs_imgsz": [544, 960]})

    high_logits = torch.full((3, k), -10.0)
    high_logits[0, :5] = 10.0
    high_loss, high_stats = loss_fn.gt4_short_valid_count_floor_loss(
        high_logits,
        valid,
        gt_lane_count,
        short_max_points=20,
        floor_ratio=0.6,
        floor_min=3,
        min_valid_points=2,
    )

    low_logits = torch.full((3, k), -10.0)
    low_loss, low_stats = loss_fn.gt4_short_valid_count_floor_loss(
        low_logits,
        valid,
        gt_lane_count,
        short_max_points=20,
        floor_ratio=0.6,
        floor_min=3,
        min_valid_points=2,
    )

    assert high_loss < 1e-4
    assert low_loss > 0.99
    assert torch.isclose(high_stats["gt4_short_valid_lane_count"], torch.tensor(1.0))
    assert torch.isclose(low_stats["gt4_short_valid_lane_count"], torch.tensor(1.0))
    assert torch.isclose(low_stats["gt4_short_gt_valid_points_mean"], torch.tensor(5.0))


def test_gt4_short_valid_forward_uses_true_gt_lane_count_not_matched_count():
    k = 24
    gt = _fixed_y_points(4, k)
    gt[:, :, 0] = torch.tensor([0.05, 0.35, 0.65, 0.95]).view(4, 1)
    pred = gt[:3].clone()
    valid = torch.zeros(4, k)
    valid[0, :3] = 1.0
    valid[1:, :21] = 1.0
    preds = {
        "pred_points": pred.unsqueeze(0),
        "pred_logits": torch.ones(1, 3),
        "pred_valid_logits": torch.zeros(1, 3, k),
    }
    batch = {
        "lanes": [gt],
        "lane_valid": [valid],
        "num_lanes": torch.tensor([4]),
    }
    loss_fn = GCSLoss(
        model={
            "gcs_imgsz": [544, 960],
            "gcs_gt4_short_valid_recall": True,
            "gcs_exist": 0.0,
            "gcs_point": 0.0,
            "gcs_point_valid": 0.0,
            "gcs_short_valid_recall": 0.0,
            "gcs_smooth": 0.0,
            "gcs_curve": 0.0,
            "gcs_mask": 0.0,
            "gcs_edge": 0.0,
            "gcs_count": 0.0,
            "gcs_count_under5": 0.0,
            "gcs_match_gate_px": 0.0,
            "gcs_cost_curve": 0.0,
            "gcs_cost_exist": 0.0,
        }
    )

    total, items = loss_fn(preds, batch)
    item_map = dict(zip(loss_fn.loss_names, items))

    assert torch.isclose(item_map["gt4_short_valid_recall_loss"], torch.log(torch.tensor(2.0)), atol=1e-6)
    assert torch.isclose(item_map["gt4_short_valid_lane_count"], torch.tensor(1.0))
    assert torch.isclose(total, 0.2 * item_map["gt4_short_valid_recall_loss"], atol=1e-6)


def test_lane_balanced_valid_forward_uses_true_gt_lane_count_not_matched_count():
    k = 24
    gt = _fixed_y_points(4, k)
    gt[:, :, 0] = torch.tensor([0.05, 0.35, 0.65, 0.95]).view(4, 1)
    pred = gt[:3].clone()
    valid = torch.zeros(4, k)
    valid[0, :3] = 1.0
    valid[1:, :21] = 1.0
    preds = {
        "pred_points": pred.unsqueeze(0),
        "pred_logits": torch.ones(1, 3),
        "pred_valid_logits": torch.zeros(1, 3, k),
    }
    batch = {
        "lanes": [gt],
        "lane_valid": [valid],
        "num_lanes": torch.tensor([4]),
    }
    loss_fn = GCSLoss(
        model={
            "gcs_imgsz": [544, 960],
            "gcs_lane_balanced_valid_loss": True,
            "gcs_gt4_short_valid_lane_weight": 1.5,
            "gcs_unmatched_valid_neg_weight": 0.0,
            "gcs_exist": 0.0,
            "gcs_point": 0.0,
            "gcs_point_valid": 1.0,
            "gcs_short_valid_recall": 0.0,
            "gcs_smooth": 0.0,
            "gcs_curve": 0.0,
            "gcs_mask": 0.0,
            "gcs_edge": 0.0,
            "gcs_count": 0.0,
            "gcs_count_under5": 0.0,
            "gcs_match_gate_px": 0.0,
            "gcs_cost_curve": 0.0,
            "gcs_cost_exist": 0.0,
        }
    )

    total, items = loss_fn(preds, batch)
    item_map = dict(zip(loss_fn.loss_names, items))

    assert torch.isclose(item_map["valid_lb_gt4_short_count"], torch.tensor(1.0))
    assert torch.isclose(item_map["valid_lb_gt4_short_gt_points_mean"], torch.tensor(3.0))
    assert torch.isclose(item_map["point_valid_loss"], torch.log(torch.tensor(2.0)), atol=1e-6)
    assert torch.isclose(total, item_map["point_valid_loss"], atol=1e-6)


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
