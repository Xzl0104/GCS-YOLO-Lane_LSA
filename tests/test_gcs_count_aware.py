from __future__ import annotations

import math
import sys

import pytest
import torch

from tools import train_gcs
from ultralytics.cfg import CFG_FLOAT_KEYS, CFG_FRACTION_KEYS
from ultralytics.models.yolo.gcs_lane.train import (
    GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT,
    GCS_MAINLINE_COUNT_CLS_WEIGHTS,
    GCS_MAINLINE_COUNT_BOUNDARY_GAIN,
    GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT,
    GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING,
    GCS_MAINLINE_COUNT_SUM_GAIN,
    GCS_MAINLINE_GROUP_SAMPLER_RATIOS,
    GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT,
    GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT,
    GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY,
    GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR,
    GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT,
    GCS_MAINLINE_QUALITY_GAIN,
    GCS_MAINLINE_QUALITY_NEG_WEIGHT,
    GCSLaneTrainer,
    apply_gt5_oversample_weight_to_ratios,
)
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.gcs_lane import CandidateAwareCountHead, GCSLaneHead
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.gcs_candidate_matching import GCSLaneCandidate
from ultralytics.utils.gcs_count_diagnostics import build_candidates_from_predictions, diagnose_count_errors
from ultralytics.utils.gcs_loss import GCSLoss
from ultralytics.utils.gcs_postprocess import (
    count_aware_refill,
    decode_gcs_predictions,
    soft_count_decision,
)


def _gt(xs: list[float], points: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(0.98, 0.25, points)
    lanes = torch.stack([torch.stack((torch.full_like(y, x), y), dim=-1) for x in xs], dim=0)
    valid = torch.ones((len(xs), points), dtype=torch.float32)
    return lanes, valid


def _cand(
    x: float,
    *,
    q: int,
    rank: int,
    valid_points: int = 6,
    score: float = 1.0,
    keep: bool = True,
    suppressed_by: int | None = None,
) -> GCSLaneCandidate:
    y = torch.linspace(0.98, 0.25, 6)
    points = torch.stack((torch.full_like(y, x), y), dim=-1)
    valid = torch.ones(6)
    if valid_points < 6:
        valid[valid_points:] = 0.0
    return GCSLaneCandidate(
        image_id="synthetic",
        query_idx=q,
        points=points,
        valid_probs=valid,
        exist_logit=6.0,
        exist_score=score,
        point_valid_mean=float(valid.mean()),
        point_valid_max=float(valid.max()),
        valid_points=valid_points,
        lane_quality=score,
        pre_nms_rank=rank,
        pre_nms_score=score,
        keep_after_nms=keep,
        suppressed_by=suppressed_by,
    )


def _diagnose(gt_xs, candidates, final, pred_count):
    gt_lanes, gt_valid = _gt(gt_xs)
    logits = torch.full((4,), -5.0)
    logits[pred_count - 2] = 5.0
    return diagnose_count_errors(
        image_id="synthetic",
        gt_lanes=gt_lanes,
        gt_valid=gt_valid,
        candidates=candidates,
        final_candidates=final,
        pred_count_logits=logits,
        diagnostic_topk=8,
        normal_min_points=5,
        image_shape=(544, 960),
    )


def test_error_type_a_count_head_wrong():
    xs = [0.1, 0.25, 0.4, 0.55, 0.7]
    candidates = [_cand(x, q=i, rank=i + 1) for i, x in enumerate(xs)]
    row = _diagnose(xs, candidates, candidates, pred_count=4)
    assert row["count_error_primary"] == "A_COUNT_HEAD_WRONG"


def test_error_type_b_candidate_pool_missing():
    xs = [0.1, 0.25, 0.4, 0.55, 0.7]
    candidates = [_cand(x, q=i, rank=i + 1) for i, x in enumerate(xs[:4])]
    row = _diagnose(xs, candidates, candidates, pred_count=5)
    assert row["count_error_primary"] == "B_CANDIDATE_POOL_MISSING"
    assert math.isclose(row["candidate_recall_all"], 4 / 5)


def test_error_type_c_true_lane_rank_low():
    xs = [0.1, 0.25, 0.4, 0.55, 0.7]
    candidates = [_cand(x, q=i, rank=i + 1) for i, x in enumerate(xs[:4])]
    candidates.extend(_cand(0.85 + 0.01 * i, q=20 + i, rank=5 + i, score=0.5) for i in range(7))
    candidates.append(_cand(xs[4], q=99, rank=12, score=0.4))
    row = _diagnose(xs, candidates, candidates[:4], pred_count=5)
    assert row["count_error_primary"] == "C_TRUE_LANE_RANK_LOW"
    assert row["missing_gt_best_rank"] == 12


def test_error_type_d_valid_points_low():
    xs = [0.1, 0.25, 0.4, 0.55, 0.7]
    candidates = [_cand(x, q=i, rank=i + 1) for i, x in enumerate(xs[:4])]
    candidates.append(_cand(xs[4], q=4, rank=5, valid_points=4, score=0.9))
    row = _diagnose(xs, candidates, candidates[:4], pred_count=5)
    assert row["count_error_primary"] == "D_TRUE_LANE_VALID_POINTS_LOW"
    assert row["missing_gt_best_valid_points"] == 4


def test_error_type_e_suppressed_by_nms():
    xs = [0.1, 0.25, 0.4, 0.55, 0.7]
    candidates = [_cand(x, q=i, rank=i + 1) for i, x in enumerate(xs[:4])]
    candidates.append(_cand(xs[4], q=4, rank=5, keep=False, suppressed_by=1, score=0.9))
    row = _diagnose(xs, candidates, candidates[:4], pred_count=5)
    assert row["count_error_primary"] == "E_TRUE_LANE_SUPPRESSED_BY_NMS"
    assert row["missing_gt_suppressed_by_nms"] == 1


def test_error_type_f_final_count_ok_but_false_or_duplicate():
    xs = [0.1, 0.25, 0.4, 0.55]
    candidates = [_cand(x, q=i, rank=i + 1) for i, x in enumerate(xs)]
    final = candidates[:3] + [_cand(0.9, q=9, rank=5)]
    row = _diagnose(xs, candidates, final, pred_count=4)
    assert row["count_error_primary"] == "F_FINAL_COUNT_OK_BUT_FALSE_OR_DUP"
    assert row["has_false_lane"] == 1


def test_candidate_aware_count_head_shape_and_grad():
    torch.manual_seed(1)
    head = CandidateAwareCountHead([16, 16, 16, 16], query_dim=16, hidden_dim=32, topq=8)
    feats = [torch.randn(2, 16, 4, 8) for _ in range(4)]
    query = torch.randn(2, 12, 16, requires_grad=True)
    logits = torch.randn(2, 12, requires_grad=True)
    valid = torch.randn(2, 12, 6, requires_grad=True)
    points = torch.rand(2, 12, 6, 2)
    out = head(feats, query, pred_logits=logits, pred_valid_logits=valid, pred_points=points)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert logits.grad is not None
    assert valid.grad is not None


def test_gcs_lane_head_count_backward_isolated_from_shared_branches():
    torch.manual_seed(2)
    head = GCSLaneHead(
        c1=16,
        num_queries=6,
        num_points=8,
        num_decoder_layers=1,
        nhead=4,
        point_mode="fixed_y",
    )
    head.min_spatial_tokens = 0
    feats = [
        torch.randn(2, 16, 8, 16, requires_grad=True),
        torch.randn(2, 16, 4, 8, requires_grad=True),
        torch.randn(2, 16, 3, 4, requires_grad=True),
        torch.randn(2, 16, 2, 3, requires_grad=True),
    ]

    out = head(feats)
    out["pred_count_logits"].sum().backward()

    count_grads = [
        param.grad
        for name, param in head.named_parameters()
        if name.startswith("count_head.") and param.requires_grad
    ]
    assert any(grad is not None and torch.count_nonzero(grad).item() > 0 for grad in count_grads)
    assert all(feat.grad is None for feat in feats)
    assert all(
        param.grad is None
        for name, param in head.named_parameters()
        if not name.startswith("count_head.") and param.requires_grad
    )


def test_gcs_lane_head_count_boundary_backward_isolated_from_shared_branches():
    torch.manual_seed(3)
    head = GCSLaneHead(
        c1=16,
        num_queries=6,
        num_points=8,
        num_decoder_layers=1,
        nhead=4,
        point_mode="fixed_y",
    )
    head.min_spatial_tokens = 0
    feats = [
        torch.randn(2, 16, 8, 16, requires_grad=True),
        torch.randn(2, 16, 4, 8, requires_grad=True),
        torch.randn(2, 16, 3, 4, requires_grad=True),
        torch.randn(2, 16, 2, 3, requires_grad=True),
    ]

    out = head(feats)
    out["pred_count_boundary_logits"].sum().backward()

    count_grads = [
        param.grad
        for name, param in head.named_parameters()
        if name.startswith("count_head.") and param.requires_grad
    ]
    assert any(grad is not None and torch.count_nonzero(grad).item() > 0 for grad in count_grads)
    assert all(feat.grad is None for feat in feats)
    assert all(
        param.grad is None
        for name, param in head.named_parameters()
        if not name.startswith("count_head.") and param.requires_grad
    )


def test_count_sum_loss_backward():
    criterion = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960], "gcs_count_sum": 0.02})
    pred_logits = torch.randn(2, 12, requires_grad=True)
    gt_valid = [_gt([0.1, 0.2, 0.3])[1], _gt([0.1, 0.2, 0.3, 0.4, 0.5])[1]]
    loss = criterion.count_sum_loss(pred_logits, {}, gt_valid)
    assert float(loss.detach()) > 0
    loss.backward()
    assert pred_logits.grad is not None


def test_gt5_oversample_ratio_boost():
    ratios = apply_gt5_oversample_weight_to_ratios({3: 0.5, 5: 0.2}, 2.0)
    assert ratios[5] == 0.4
    assert ratios[3] == 0.5


def test_mainline_sampler_defaults_and_ratio_boost_boundaries(monkeypatch):
    assert DEFAULT_CFG_DICT["gcs_group_sampler_ratios"] == GCS_MAINLINE_GROUP_SAMPLER_RATIOS
    assert DEFAULT_CFG_DICT["gcs_gt5_oversample_weight"] == GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT
    assert math.isclose(DEFAULT_CFG_DICT["gcs_count_sum"], GCS_MAINLINE_COUNT_SUM_GAIN)
    assert math.isclose(DEFAULT_CFG_DICT["gcs_quality"], GCS_MAINLINE_QUALITY_GAIN)
    assert math.isclose(DEFAULT_CFG_DICT["gcs_quality_neg_weight"], GCS_MAINLINE_QUALITY_NEG_WEIGHT)
    assert tuple(DEFAULT_CFG_DICT[f"gcs_count_cls_w{i}"] for i in range(2, 6)) == GCS_MAINLINE_COUNT_CLS_WEIGHTS
    assert math.isclose(
        DEFAULT_CFG_DICT["gcs_point_valid_gt5_pos_weight"], GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT
    )
    assert math.isclose(DEFAULT_CFG_DICT["gcs_gt5_edge_loss_weight"], GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT)
    assert DEFAULT_CFG_DICT["gcs_count_boundary"] == GCS_MAINLINE_COUNT_BOUNDARY_GAIN
    assert DEFAULT_CFG_DICT["gcs_count_boundary_label_smoothing"] == GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING
    assert DEFAULT_CFG_DICT["gcs_count_boundary_gt5_pos_weight"] == GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT
    assert DEFAULT_CFG_DICT["gcs_candidate_gt5_edge_weight"] == GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT
    assert DEFAULT_CFG_DICT["gcs_point_valid_gt5_edge_continuity"] == GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY
    assert (
        DEFAULT_CFG_DICT["gcs_point_valid_gt5_edge_continuity_thr"]
        == GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR
    )

    monkeypatch.setattr(sys, "argv", ["train_gcs.py"])
    args = train_gcs.parse_args()
    assert args.gcs_group_sampler_ratios == GCS_MAINLINE_GROUP_SAMPLER_RATIOS
    assert args.gcs_gt5_oversample_weight == GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT
    assert math.isclose(args.gcs_count_sum, GCS_MAINLINE_COUNT_SUM_GAIN)
    assert math.isclose(args.gcs_quality, GCS_MAINLINE_QUALITY_GAIN)
    assert math.isclose(args.gcs_quality_neg_weight, GCS_MAINLINE_QUALITY_NEG_WEIGHT)
    assert tuple(getattr(args, f"gcs_count_cls_w{i}") for i in range(2, 6)) == GCS_MAINLINE_COUNT_CLS_WEIGHTS
    assert math.isclose(args.gcs_point_valid_gt5_pos_weight, GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT)
    assert math.isclose(args.gcs_gt5_edge_loss_weight, GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT)
    assert args.gcs_count_boundary == GCS_MAINLINE_COUNT_BOUNDARY_GAIN
    assert args.gcs_count_boundary_label_smoothing == GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING
    assert args.gcs_count_boundary_gt5_pos_weight == GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT
    assert args.gcs_candidate_gt5_edge_weight == GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT
    assert args.gcs_point_valid_gt5_edge_continuity == GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY
    assert args.gcs_point_valid_gt5_edge_continuity_thr == GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR

    trainer_overrides = {}
    monkeypatch.setattr(BaseTrainer, "__init__", lambda self, cfg, overrides, callbacks: trainer_overrides.update(overrides))
    monkeypatch.setattr(GCSLaneTrainer, "_lock_gcs_shape_contract", lambda self: None)
    GCSLaneTrainer()
    assert trainer_overrides["gcs_group_sampler_ratios"] == GCS_MAINLINE_GROUP_SAMPLER_RATIOS
    assert trainer_overrides["gcs_gt5_oversample_weight"] == GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT
    assert math.isclose(trainer_overrides["gcs_count_sum"], GCS_MAINLINE_COUNT_SUM_GAIN)
    assert math.isclose(trainer_overrides["gcs_quality"], GCS_MAINLINE_QUALITY_GAIN)
    assert math.isclose(trainer_overrides["gcs_quality_neg_weight"], GCS_MAINLINE_QUALITY_NEG_WEIGHT)
    assert tuple(trainer_overrides[f"gcs_count_cls_w{i}"] for i in range(2, 6)) == GCS_MAINLINE_COUNT_CLS_WEIGHTS
    assert math.isclose(
        trainer_overrides["gcs_point_valid_gt5_pos_weight"], GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT
    )
    assert math.isclose(trainer_overrides["gcs_gt5_edge_loss_weight"], GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT)
    assert trainer_overrides["gcs_count_boundary"] == GCS_MAINLINE_COUNT_BOUNDARY_GAIN
    assert trainer_overrides["gcs_count_boundary_label_smoothing"] == GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING
    assert trainer_overrides["gcs_count_boundary_gt5_pos_weight"] == GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT
    assert trainer_overrides["gcs_candidate_gt5_edge_weight"] == GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT
    assert trainer_overrides["gcs_point_valid_gt5_edge_continuity"] == GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY
    assert (
        trainer_overrides["gcs_point_valid_gt5_edge_continuity_thr"]
        == GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR
    )

    criterion = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960]})
    assert math.isclose(criterion.count_sum_gain, GCS_MAINLINE_COUNT_SUM_GAIN)
    assert math.isclose(criterion.quality_gain, GCS_MAINLINE_QUALITY_GAIN)
    assert math.isclose(criterion.quality_neg_weight, GCS_MAINLINE_QUALITY_NEG_WEIGHT)
    assert criterion.count_cls_weights == GCS_MAINLINE_COUNT_CLS_WEIGHTS
    assert math.isclose(criterion.point_valid_gt5_pos_weight, GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT)
    assert math.isclose(criterion.gt5_edge_loss_weight, GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT)
    assert math.isclose(criterion.count_boundary_gain, GCS_MAINLINE_COUNT_BOUNDARY_GAIN)
    assert math.isclose(criterion.count_boundary_label_smoothing, GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING)
    assert math.isclose(criterion.count_boundary_gt5_pos_weight, GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT)
    assert math.isclose(criterion.candidate_gt5_edge_weight, GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT)
    assert math.isclose(criterion.point_valid_gt5_edge_continuity, GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY)
    assert math.isclose(
        criterion.point_valid_gt5_edge_continuity_thr, GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR
    )

    ratios = {2: 0.01, 3: 0.29, 4: 0.42, 5: 0.28}
    assert apply_gt5_oversample_weight_to_ratios(ratios, 1.0) == ratios
    assert ratios[5] == 0.28
    with pytest.raises(ValueError, match="must be > 0"):
        apply_gt5_oversample_weight_to_ratios(ratios, 0.0)


def test_gcs_loss_item_names_stay_stable():
    expected = (
        "exist_loss",
        "point_loss",
        "point_valid_loss",
        "line_iou_loss",
        "count_cls_loss",
        "count_sum_loss",
        "quality_loss",
    )
    assert GCSLoss.loss_names == expected
    assert GCSLaneTrainer.loss_names == expected
    assert GCSLaneTrainer.progress_loss_names == expected


def test_gt5_candidate_cfg_keys_have_expected_types():
    assert {
        "gcs_count_boundary_gt5_pos_weight",
        "gcs_candidate_gt5_edge_weight",
        "gcs_point_valid_gt5_edge_continuity",
    } <= CFG_FLOAT_KEYS
    assert "gcs_point_valid_gt5_edge_continuity_thr" in CFG_FRACTION_KEYS


def test_count_boundary_gt5_pos_weight_increases_count_loss():
    _, valid = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    preds = {
        "pred_count_logits": torch.zeros(1, 4),
        "pred_count_boundary_logits": torch.tensor([[0.0, -2.0]]),
    }
    pred_points = torch.zeros(1, 5, 6, 2)
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_count_boundary": 1.0,
        "gcs_count_boundary_label_smoothing": 0.0,
    }
    base = GCSLoss(model={**common, "gcs_count_boundary_gt5_pos_weight": 1.0})
    boosted = GCSLoss(model={**common, "gcs_count_boundary_gt5_pos_weight": 2.0})

    base_loss = base.count_head_loss(preds, pred_points, [valid])
    boosted_loss = boosted.count_head_loss(preds, pred_points, [valid])

    assert boosted_loss > base_loss


def test_candidate_gt5_edge_weight_targets_real_edge_queries():
    lanes, valid = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_gt5_edge_loss_weight": 1.0,
            "gcs_candidate_gt5_edge_weight": 1.5,
        }
    )

    weights = criterion._matched_target_weights(
        lanes,
        valid,
        torch.tensor([0, 1, 2, 3, 4]),
        device=torch.device("cpu"),
        dtype=torch.float32,
        term="point",
    )

    assert torch.isclose(weights[0], torch.tensor(1.5))
    assert torch.isclose(weights[2], torch.tensor(1.0))
    assert torch.isclose(weights[4], torch.tensor(1.5))


def test_gt5_edge_weight_reaches_quality_loss():
    lanes, valid = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    pred_points = lanes.unsqueeze(0).clone()
    pred_quality_logits = torch.tensor([[-3.0, 3.0, 3.0, 3.0, -3.0]], requires_grad=True)
    indices = [(torch.arange(5), torch.arange(5))]
    hard_negative_mask = torch.zeros(1, 5, dtype=torch.bool)
    duplicate_negative_mask = torch.zeros(1, 5, dtype=torch.bool)
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_gt5_edge_loss_weight": 1.0,
        "gcs_quality_dist_thr_px": 100.0,
    }
    base = GCSLoss(model={**common, "gcs_candidate_gt5_edge_weight": 1.0})
    boosted = GCSLoss(model={**common, "gcs_candidate_gt5_edge_weight": 2.0})

    base_loss = base.quality_loss(
        pred_quality_logits,
        pred_points,
        [lanes],
        [valid],
        indices,
        hard_negative_mask=hard_negative_mask,
        duplicate_negative_mask=duplicate_negative_mask,
    )
    boosted_loss = boosted.quality_loss(
        pred_quality_logits,
        pred_points,
        [lanes],
        [valid],
        indices,
        hard_negative_mask=hard_negative_mask,
        duplicate_negative_mask=duplicate_negative_mask,
    )

    assert boosted_loss > base_loss
    boosted_loss.backward()
    assert pred_quality_logits.grad is not None


def test_point_valid_gt5_edge_continuity_adds_loss():
    lanes, valid = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    pred_points = lanes.unsqueeze(0).clone()
    logits = torch.full((1, 5, 6), 4.0)
    logits[0, 0, 2] = -4.0
    logits[0, 4, 3] = -4.0
    pred_valid_logits = logits.requires_grad_()
    indices = [(torch.arange(5), torch.arange(5))]
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_gt5_edge_loss_weight": 1.0,
        "gcs_candidate_gt5_edge_weight": 1.0,
        "gcs_point_valid_gt5_pos_weight": 1.0,
        "gcs_point_valid_unmatched_weight": 1.0,
    }
    base = GCSLoss(model={**common, "gcs_point_valid_gt5_edge_continuity": 0.0})
    continuity = GCSLoss(
        model={
            **common,
            "gcs_point_valid_gt5_edge_continuity": 0.5,
            "gcs_point_valid_gt5_edge_continuity_thr": 0.8,
        }
    )

    base_loss = base.point_valid_loss(pred_valid_logits, pred_points, [valid], indices, gt_points=[lanes])
    continuity_loss = continuity.point_valid_loss(pred_valid_logits, pred_points, [valid], indices, gt_points=[lanes])

    assert continuity_loss > base_loss
    continuity_loss.backward()
    assert pred_valid_logits.grad is not None


def test_edge_lane_weights_do_not_affect_count_sum():
    criterion = GCSLoss(
        model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960], "gcs_gt5_edge_loss_weight": 1.5}
    )
    lanes, valid = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    weights = criterion._matched_target_weights(
        lanes,
        valid,
        torch.tensor([0, 1, 2, 3, 4]),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert weights[0] > weights[2]
    assert weights[4] > weights[2]
    pred_logits = torch.randn(1, 12, requires_grad=True)
    loss = criterion.count_sum_loss(pred_logits, {}, [valid])
    loss.backward()
    assert pred_logits.grad is not None


def test_hard_edge_loss_weights_match_manifest_and_count(tmp_path):
    manifest = tmp_path / "hard.txt"
    manifest.write_text(
        "D:/data/images/train/hard4.jpg\nD:/data/images/train/hard5.jpg\n",
        encoding="utf-8",
    )
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_gt5_edge_loss_weight": 1.0,
            "gcs_candidate_gt5_edge_weight": 1.0,
            "gcs_hard_loss_file": str(manifest),
            "gcs_hard_loss_lane_counts": "5",
            "gcs_hard_edge_loss_weight_by_count": "4:1.15,5:1.6",
            "gcs_hard_edge_loss_terms": "exist,point,point_valid,line_iou",
            "gcs_hard_edge_only": True,
        }
    )
    _, valid4 = _gt([0.1, 0.25, 0.55, 0.7])
    lanes5, valid5 = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    hard_mask = criterion.hard_loss_mask(
        {
            "im_file": ["D:/data/images/train/hard4.jpg", "D:/data/images/train/hard5.jpg"],
            "label_file": [
                "D:/data/labels_gcs/train/hard4.npz",
                "D:/data/labels_gcs/train/hard5.npz",
            ],
        },
        2,
        torch.device("cpu"),
        gt_valid=[valid4, valid5],
    )
    assert hard_mask.tolist() == [False, True]

    weights5 = criterion._matched_target_weights(
        lanes5,
        valid5,
        torch.tensor([0, 1, 2, 3, 4]),
        device=torch.device("cpu"),
        dtype=torch.float32,
        hard_image=True,
        term="point",
    )
    assert torch.isclose(weights5[0], torch.tensor(1.6))
    assert torch.isclose(weights5[2], torch.tensor(1.0))
    assert torch.isclose(weights5[4], torch.tensor(1.6))

    lanes4, _ = _gt([0.1, 0.25, 0.55, 0.7])
    weights4 = criterion._matched_target_weights(
        lanes4,
        valid4,
        torch.tensor([0, 1, 2, 3]),
        device=torch.device("cpu"),
        dtype=torch.float32,
        hard_image=True,
        term="exist",
    )
    assert torch.isclose(weights4[0], torch.tensor(1.15))
    assert torch.isclose(weights4[1], torch.tensor(1.0))
    assert torch.isclose(weights4[3], torch.tensor(1.15))

    non_hard = criterion._matched_target_weights(
        lanes5,
        valid5,
        torch.tensor([0, 1, 2, 3, 4]),
        device=torch.device("cpu"),
        dtype=torch.float32,
        hard_image=False,
        term="point",
    )
    quality_term = criterion._matched_target_weights(
        lanes5,
        valid5,
        torch.tensor([0, 1, 2, 3, 4]),
        device=torch.device("cpu"),
        dtype=torch.float32,
        hard_image=True,
        term="quality",
    )
    assert torch.equal(non_hard, torch.ones_like(non_hard))
    assert torch.equal(quality_term, torch.ones_like(quality_term))


def test_count_aware_refill_does_not_fabricate_lanes():
    selected = [{"query": i, "points_norm": _cand(0.1 + i * 0.1, q=i, rank=i + 1).points.numpy(), "valid_count": 6, "rank_score": 1.0} for i in range(4)]
    rescue = selected + [{"query": 4, "points_norm": _cand(0.8, q=4, rank=5).points.numpy(), "valid_count": 6, "rank_score": 0.9}]
    out = count_aware_refill(selected, rescue, 5, (544, 960), 5, rescue_dist_px=0.0)
    assert len(out) == 5
    assert out[-1]["query"] == 4
    assert out[-1]["source"] == "rescue_refill"
    out2 = count_aware_refill(selected, selected, 5, (544, 960), 5, rescue_dist_px=0.0)
    assert len(out2) == 4


def test_normal_rescue_dual_thresholds():
    y = torch.linspace(0.98, 0.25, 6)
    points = torch.stack((torch.full_like(y, 0.5), y), dim=-1).unsqueeze(0)
    logits = torch.tensor([math.log(0.02 / 0.98)])
    valid = torch.full((1, 6), math.log(0.10 / 0.90))
    cands = build_candidates_from_predictions(
        image_id="synthetic",
        pred_points=points,
        pred_logits=logits,
        pred_valid_logits=valid,
        normal_candidate_score_thr=0.03,
        normal_point_valid_thr=0.15,
        normal_min_points=5,
        rescue_candidate_score_thr=0.015,
        rescue_point_valid_thr=0.08,
        rescue_min_points=4,
        nms_dist_px=0.0,
    )
    assert len(cands) == 1
    assert cands[0].source == "rescue"


def test_soft_count_decision_can_upgrade_or_stay():
    lanes = [{"rank_score": 1.0, "quality_score": 1.0, "valid_count": 6, "points_norm": _cand(0.1 + i * 0.1, q=i, rank=i + 1).points.numpy()} for i in range(5)]
    meta = soft_count_decision([0.01, 0.10, 0.46, 0.43], lanes, prob_margin=0.08, min_points=5)
    assert meta["pred_count_cls_raw"] == 4
    assert meta["pred_count_cls_soft"] == 5
    lanes[-1]["quality_score"] = -5.0
    meta2 = soft_count_decision([0.01, 0.10, 0.46, 0.43], lanes, prob_margin=0.08, min_points=5)
    assert meta2["pred_count_cls_soft"] == 4
