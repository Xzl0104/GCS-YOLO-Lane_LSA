from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import train_gcs
from ultralytics.data.dataset_gcs import GCSLaneDataset
from ultralytics.cfg import CFG_BOOL_KEYS, CFG_FLOAT_KEYS, CFG_FRACTION_KEYS, CFG_INT_KEYS
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
    GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT,
    GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_MIN_POINTS,
    GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_THR,
    GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT,
    GCS_MAINLINE_QUALITY_HARD_NEGATIVE_FROM_HEAD,
    GCS_MAINLINE_QUALITY_GAIN,
    GCS_MAINLINE_QUALITY_GT5_EDGE_FLOOR,
    GCS_MAINLINE_QUALITY_NEG_WEIGHT,
    GCS_MAINLINE_QUALITY_POINT_WEIGHT,
    GCSLaneTrainer,
    apply_gt5_oversample_weight_to_ratios,
)
from ultralytics.models.yolo.gcs_lane.val import GCSLaneValidator, LOSS_NAMES as VAL_LOSS_NAMES
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules.gcs_lane import (
    CandidateAwareCountHead,
    GCSLaneHead,
    LaneStripPyramidAttention,
    TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM,
    TUSIMPLE_OFFICIAL_TOP_Y_NORM,
)
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


def _gt_fixed_y56(
    xs: list[float],
    *,
    visible_start: int = 34,
    visible_end: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM, TUSIMPLE_OFFICIAL_TOP_Y_NORM, 56)
    lanes = torch.stack([torch.stack((torch.full_like(y, x), y), dim=-1) for x in xs], dim=0)
    valid = torch.zeros((len(xs), 56), dtype=torch.float32)
    valid[:, visible_start:visible_end] = 1.0
    return lanes, valid


def _logit_prob(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


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


def test_count_head_visible_segment_evidence_keeps_short_edge_lane_count_visible():
    head = CandidateAwareCountHead([16, 16, 16, 16], query_dim=16, hidden_dim=32, topq=8)
    high = math.log(0.95 / 0.05)
    low = math.log(0.05 / 0.95)
    pred_logits = torch.full((1, 6), high)
    pred_valid_logits = torch.full((1, 6, 56), low)
    pred_valid_logits[0, 0:4, :] = high
    pred_valid_logits[0, 4, 34:40] = high
    pred_valid_logits[0, 5, [3, 12, 21, 30, 39, 48]] = high

    valid_prob = pred_valid_logits.sigmoid()
    visible_mean, visible_support, visible_points, all_anchor_mean = head._visible_segment_stats(valid_prob)

    assert torch.isclose(visible_points[0, 4, 0], torch.tensor(6.0))
    assert visible_mean[0, 4, 0] > 0.94
    assert torch.isclose(visible_support[0, 4, 0], torch.tensor(0.5))
    assert all_anchor_mean[0, 4, 0] < 0.23

    _, lane_quality = head._candidate_extra_features(
        pred_logits,
        pred_valid_logits,
        pred_points=None,
        pred_quality_logits=None,
    )
    old_all_anchor_quality = pred_logits.sigmoid()[0, 4] * all_anchor_mean[0, 4, 0]
    assert lane_quality[0, 4] > 0.45
    assert old_all_anchor_quality < 0.22
    assert int(lane_quality.topk(k=5, dim=1).indices[0, 4]) == 4

    features = head._cardinality_features(pred_logits, pred_valid_logits, pred_quality_logits=None)
    top5_idx = head.cardinality_feature_names.index("lane_quality_top5")
    valid_mean_idx = head.cardinality_feature_names.index("valid_mean")
    assert features[0, top5_idx] > 0.45
    assert features[0, valid_mean_idx] < 0.9


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


def test_gcs_lane_head_count_quality_calib_keeps_output_contract_and_quality_grad():
    torch.manual_seed(4)
    head = GCSLaneHead(
        c1=16,
        num_queries=6,
        num_points=8,
        num_decoder_layers=1,
        nhead=4,
        point_mode="fixed_y",
        count_quality_calib_dim=8,
    )
    head.min_spatial_tokens = 0
    feats = [
        torch.randn(2, 16, 8, 16),
        torch.randn(2, 16, 4, 8),
        torch.randn(2, 16, 3, 4),
        torch.randn(2, 16, 2, 3),
    ]

    out = head(feats)

    assert out["pred_quality_logits"].shape == (2, 6)
    assert out["pred_count_logits"].shape == (2, 4)
    assert out["pred_count_boundary_logits"].shape == (2, 2)
    out["pred_quality_logits"].sum().backward()
    calib_grads = [
        param.grad
        for name, param in head.named_parameters()
        if name.startswith("count_quality_calib.") and param.requires_grad
    ]
    assert any(grad is not None and torch.count_nonzero(grad).item() > 0 for grad in calib_grads)


def test_gcs_lane_head_count_quality_calib_preserves_count_head_isolation():
    torch.manual_seed(5)
    head = GCSLaneHead(
        c1=16,
        num_queries=6,
        num_points=8,
        num_decoder_layers=1,
        nhead=4,
        point_mode="fixed_y",
        count_quality_calib_dim=8,
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

    assert any(
        param.grad is not None and torch.count_nonzero(param.grad).item() > 0
        for name, param in head.named_parameters()
        if name.startswith("count_head.") and param.requires_grad
    )
    assert all(feat.grad is None for feat in feats)
    assert all(
        param.grad is None
        for name, param in head.named_parameters()
        if name.startswith("count_quality_calib.") and param.requires_grad
    )


def test_gcs_lane_head_survival_and_decoder_aux_outputs_keep_shapes():
    torch.manual_seed(7)
    head = GCSLaneHead(
        c1=16,
        num_queries=6,
        num_points=56,
        num_decoder_layers=3,
        nhead=4,
        point_mode="fixed_y",
        survival_head=True,
        decoder_aux_loss=True,
    )
    head.min_spatial_tokens = 0
    feats = [
        torch.randn(2, 16, 8, 16),
        torch.randn(2, 16, 4, 8),
        torch.randn(2, 16, 3, 4),
        torch.randn(2, 16, 2, 3),
    ]

    out = head(feats)

    assert out["pred_survival_logits"].shape == (2, 6)
    assert len(out["aux_outputs"]) == 2
    assert all(aux["pred_points"].shape == (2, 6, 56, 2) for aux in out["aux_outputs"])
    assert all(aux["pred_valid_logits"].shape == (2, 6, 56) for aux in out["aux_outputs"])
    out["pred_survival_logits"].sum().backward()
    assert any(
        param.grad is not None and torch.count_nonzero(param.grad).item() > 0
        for name, param in head.named_parameters()
        if name.startswith("survival_mlp.") and param.requires_grad
    )


def test_gcs_lane_head_missing_optional_flags_matches_old_checkpoint_defaults():
    torch.manual_seed(8)
    head = GCSLaneHead(
        c1=16,
        num_queries=6,
        num_points=56,
        num_decoder_layers=3,
        nhead=4,
        point_mode="fixed_y",
    )
    head.min_spatial_tokens = 0
    delattr(head, "decoder_aux_outputs")
    delattr(head, "survival_head_enabled")
    feats = [
        torch.randn(1, 16, 8, 16),
        torch.randn(1, 16, 4, 8),
        torch.randn(1, 16, 3, 4),
        torch.randn(1, 16, 2, 3),
    ]

    out = head(feats)

    assert "aux_outputs" not in out
    assert "pred_survival_logits" not in out
    assert out["pred_points"].shape == (1, 6, 56, 2)
    assert out["pred_count_logits"].shape == (1, 4)
    assert out["pred_count_boundary_logits"].shape == (1, 2)


def test_survival_head_keeps_quality_logits_out_of_count_head(monkeypatch):
    def run_head(survival_head: bool):
        head = GCSLaneHead(
            c1=16,
            num_queries=6,
            num_points=56,
            num_decoder_layers=3,
            nhead=4,
            point_mode="fixed_y",
            survival_head=survival_head,
        )
        head.min_spatial_tokens = 0
        captured = {}

        def fake_forward_with_boundary(_feats, _query_embed, **kwargs):
            captured["pred_quality_logits"] = kwargs.get("pred_quality_logits")
            pred_logits = kwargs["pred_logits"]
            return pred_logits.new_zeros((pred_logits.shape[0], 4)), pred_logits.new_zeros((pred_logits.shape[0], 2))

        monkeypatch.setattr(head.count_head, "forward_with_boundary", fake_forward_with_boundary)
        feats = [
            torch.randn(1, 16, 8, 16),
            torch.randn(1, 16, 4, 8),
            torch.randn(1, 16, 3, 4),
            torch.randn(1, 16, 2, 3),
        ]
        head(feats)
        return captured["pred_quality_logits"]

    assert run_head(survival_head=False) is not None
    assert run_head(survival_head=True) is None


def test_lane_strip_pyramid_attention_is_zero_init_residual_for_selected_levels():
    torch.manual_seed(6)
    module = LaneStripPyramidAttention(16, levels="p2,p3", k=9)
    xs = [
        torch.randn(2, 16, 8, 16),
        torch.randn(2, 16, 4, 8),
        torch.randn(2, 16, 3, 4),
        torch.randn(2, 16, 2, 3),
    ]

    out = module(xs)

    assert [tuple(x.shape) for x in out] == [tuple(x.shape) for x in xs]
    assert module.selected == {0, 1}
    assert torch.allclose(out[0], xs[0])
    assert torch.allclose(out[1], xs[1])
    assert torch.equal(out[2], xs[2])
    assert torch.equal(out[3], xs[3])


def test_count_sum_loss_backward():
    criterion = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960], "gcs_count_sum": 0.02})
    pred_logits = torch.randn(2, 12, requires_grad=True)
    gt_valid = [_gt([0.1, 0.2, 0.3])[1], _gt([0.1, 0.2, 0.3, 0.4, 0.5])[1]]
    loss = criterion.count_sum_loss(pred_logits, {}, gt_valid)
    assert float(loss.detach()) > 0
    loss.backward()
    assert pred_logits.grad is not None


def test_query_count_losses_share_count_min_gt_points():
    gt_valid = [
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    ]
    pred_logits = torch.zeros(1, 4)
    default = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960], "gcs_count_min_gt_points": 1})
    strict = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960], "gcs_count_min_gt_points": 2})

    assert default.target_lane_count(pred_logits, {}, gt_valid).item() == 2
    assert default.count_head_targets(torch.zeros(1, 4), gt_valid)[2].item() == 2
    assert strict.target_lane_count(pred_logits, {}, gt_valid).item() == 1
    assert strict.count_head_targets(torch.zeros(1, 4), gt_valid)[2].item() == 1


def test_visible_count_sum_loss_backprops_to_exist_valid_quality_and_survival():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_visible_count_sum": 1.0,
            "gcs_visible_count_sum_quality_weight": 0.5,
            "gcs_visible_count_sum_survival_weight": 0.5,
        }
    )
    _, valid = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.zeros(1, 5, requires_grad=True)
    pred_valid_logits = torch.full((1, 5, 56), -4.0)
    pred_valid_logits[:, :, 30:44] = 4.0
    pred_valid_logits.requires_grad_()
    pred_quality_logits = torch.zeros(1, 5, requires_grad=True)
    pred_survival_logits = torch.zeros(1, 5, requires_grad=True)

    loss = criterion.visible_count_sum_loss(
        pred_logits,
        pred_valid_logits,
        pred_quality_logits,
        pred_survival_logits,
        {},
        [valid],
    )

    assert float(loss.detach()) > 0.0
    loss.backward()
    for tensor in (pred_logits, pred_valid_logits, pred_quality_logits, pred_survival_logits):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad).item() > 0


def test_visible_count_sum_loss_soft_fallback_backprops_when_no_segment_is_visible():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_visible_count_sum": 1.0,
        }
    )
    _, valid = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.zeros(1, 5, requires_grad=True)
    pred_valid_logits = torch.full((1, 5, 56), -8.0, requires_grad=True)
    loss = criterion.visible_count_sum_loss(
        pred_logits,
        pred_valid_logits,
        None,
        None,
        {},
        [valid],
    )

    assert float(loss.detach()) > 0.0
    loss.backward()
    assert pred_valid_logits.grad is not None
    assert torch.count_nonzero(pred_valid_logits.grad).item() > 0


def test_visible_count_sum_loss_uses_count4_count5_ordinal_targets():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_visible_count_sum": 1.0,
            "gcs_visible_count_sum_normalize": False,
            "gcs_visible_count_boundary": 1.0,
            "gcs_visible_count_boundary_temperature": 0.5,
        }
    )
    targets = torch.tensor([3.0, 4.0, 5.0])
    pred_logits = torch.logit((targets / 6.0).view(-1, 1).expand(-1, 6)).clone()
    gt_valid = [_gt_fixed_y56([0.1 + 0.1 * i for i in range(count)])[1] for count in (3, 4, 5)]

    loss = criterion.visible_count_sum_loss(pred_logits, None, None, None, {}, gt_valid)

    boundary_logits = (targets.view(-1, 1) - torch.tensor([3.5, 4.5]).view(1, 2)) / 0.5
    boundary_targets = torch.stack((targets.ge(4).float(), targets.ge(5).float()), dim=1)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        boundary_logits,
        boundary_targets,
        reduction="none",
    ).mean(dim=1).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_survival_loss_targets_matched_lanes_and_unmatched_negatives():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_survival": 1.0,
            "gcs_survival_neg_weight": 1.0,
        }
    )
    pred_survival_logits = torch.zeros(1, 4, requires_grad=True)
    pred_points = torch.zeros(1, 4, 56, 2)
    indices = [(torch.tensor([0, 2]), torch.tensor([0, 1]))]

    loss = criterion.survival_loss(pred_survival_logits, pred_points, indices)

    assert float(loss.detach()) > 0.0
    loss.backward()
    assert pred_survival_logits.grad is not None
    assert pred_survival_logits.grad[0, 0] < 0.0
    assert pred_survival_logits.grad[0, 2] < 0.0
    assert pred_survival_logits.grad[0, 1] > 0.0
    assert pred_survival_logits.grad[0, 3] > 0.0


def test_survival_loss_negative_weight_changes_unmatched_gradient():
    pred_points = torch.zeros(1, 4, 56, 2)
    indices = [(torch.tensor([0, 2]), torch.tensor([0, 1]))]

    def grad_for_weight(weight: float) -> torch.Tensor:
        criterion = GCSLoss(
            model={
                "gcs_point_mode": "fixed_y",
                "gcs_imgsz": [544, 960],
                "gcs_survival": 1.0,
                "gcs_survival_neg_weight": weight,
            }
        )
        logits = torch.zeros(1, 4, requires_grad=True)
        loss = criterion.survival_loss(logits, pred_points, indices)
        loss.backward()
        return logits.grad.detach().clone()

    low = grad_for_weight(0.5)
    high = grad_for_weight(2.0)

    assert high[0, 1] > low[0, 1]
    assert high[0, 3] > low[0, 3]
    assert torch.allclose(high[0, [0, 2]], low[0, [0, 2]])


def test_count_conditioned_fifth_survival_targets_edges_and_fifth_candidates_only():
    lanes, valid = _gt_fixed_y56([0.1, 0.25, 0.4, 0.55, 0.7])
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_survival": 1.0,
            "gcs_survival_target_mode": "count_conditioned_fifth",
            "gcs_survival_neg_weight": 1.0,
            "gcs_fifth_gate_hard_negative_weight": 3.0,
        }
    )
    pred_survival_logits = torch.zeros(1, 9, requires_grad=True)
    pred_points = torch.zeros(1, 9, 56, 2)
    indices = [(torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4]))]
    hard_negative_mask = torch.zeros(1, 9, dtype=torch.bool)
    duplicate_negative_mask = torch.zeros(1, 9, dtype=torch.bool)
    fifth_candidate_negative_mask = torch.zeros(1, 9, dtype=torch.bool)
    hard_negative_mask[0, 6] = True
    duplicate_negative_mask[0, 7] = True
    fifth_candidate_negative_mask[0, 8] = True

    loss = criterion.survival_loss(
        pred_survival_logits,
        pred_points,
        indices,
        hard_negative_mask=hard_negative_mask,
        duplicate_negative_mask=duplicate_negative_mask,
        fifth_candidate_negative_mask=fifth_candidate_negative_mask,
        gt_points=[lanes],
        gt_valid=[valid],
        hard_loss_mask=torch.tensor([True]),
    )

    loss.backward()
    grad = pred_survival_logits.grad
    assert grad is not None
    assert grad[0, 0] < 0.0
    assert grad[0, 4] < 0.0
    assert torch.allclose(grad[0, 1:4], torch.zeros(3))
    assert torch.allclose(grad[0, 5:], torch.zeros(4))


def test_count_conditioned_fifth_survival_edge_uses_count_min_gt_points():
    lanes, valid = _gt_fixed_y56([0.1, 0.25, 0.4, 0.55, 0.7])
    valid[0] = 0.0
    valid[0, 40] = 1.0
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_count_min_gt_points": 1,
            "gcs_survival": 1.0,
            "gcs_survival_target_mode": "count_conditioned_fifth",
            "gcs_survival_neg_weight": 1.0,
        }
    )
    pred_survival_logits = torch.zeros(1, 6, requires_grad=True)
    pred_points = torch.zeros(1, 6, 56, 2)
    indices = [(torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4]))]

    loss = criterion.survival_loss(
        pred_survival_logits,
        pred_points,
        indices,
        hard_negative_mask=torch.zeros(1, 6, dtype=torch.bool),
        duplicate_negative_mask=torch.zeros(1, 6, dtype=torch.bool),
        fifth_candidate_negative_mask=torch.zeros(1, 6, dtype=torch.bool),
        gt_points=[lanes],
        gt_valid=[valid],
    )

    loss.backward()
    grad = pred_survival_logits.grad
    assert grad is not None
    assert grad[0, 0] < 0.0
    assert grad[0, 4] < 0.0
    assert torch.allclose(grad[0, 1:4], torch.zeros(3))
    assert torch.allclose(grad[0, 5:], torch.zeros(1))


def test_survival_loss_requires_survival_logits_when_enabled():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_survival": 1.0,
            "gcs_survival_target_mode": "count_conditioned_fifth",
        }
    )

    with pytest.raises(ValueError, match="pred_survival_logits is missing"):
        criterion.survival_loss(
            None,
            torch.zeros(1, 2, 56, 2),
            [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))],
            gt_points=[torch.zeros(0, 56, 2)],
            gt_valid=[torch.zeros(0, 56)],
        )


def test_fifth_candidate_negative_mask_selects_unmatched_rank5_candidate():
    criterion = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960]})
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    _, valid5 = _gt_fixed_y56([0.1, 0.25, 0.4, 0.55, 0.7])
    pred_logits = torch.tensor([[5.0, 4.8, 4.6, 4.4, 4.2, 1.0]])
    pred_valid_logits = torch.full((1, 6, 56), 5.0)
    indices = [(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))]

    mask = criterion.fifth_candidate_negative_mask(pred_logits, pred_valid_logits, [valid4], indices)
    gt5_mask = criterion.fifth_candidate_negative_mask(pred_logits, pred_valid_logits, [valid5], indices)

    assert mask.tolist() == [[False, False, False, False, True, False]]
    assert gt5_mask.tolist() == [[False, False, False, False, False, False]]


def test_fifth_candidate_negative_mask_skips_unmatched_candidates_before_rank5():
    criterion = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960]})
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.tensor([[5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 1.0]])
    pred_valid_logits = torch.full((1, 7, 56), 5.0)
    indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))]

    mask = criterion.fifth_candidate_negative_mask(pred_logits, pred_valid_logits, [valid4], indices)

    assert mask.tolist() == [[False, False, False, False, True, False, False]]


def test_fifth_candidate_negative_mask_uses_decode_candidate_visibility_thresholds():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_decode_candidate_point_valid_thr": 0.20,
            "gcs_decode_candidate_min_points": 5,
        }
    )
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.tensor([[5.0, 4.9, 4.8, 4.7, 4.6, 4.5]])
    pred_valid_logits = torch.full((1, 6, 56), _logit_prob(0.05))
    pred_valid_logits[0, :, 10:16] = _logit_prob(0.95)
    pred_valid_logits[0, 3, :] = _logit_prob(0.05)
    pred_valid_logits[0, 3, 10:14] = _logit_prob(0.95)
    indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))]

    mask = criterion.fifth_candidate_negative_mask(pred_logits, pred_valid_logits, [valid4], indices)

    assert mask.tolist() == [[False, False, False, False, False, True]]


def test_fifth_candidate_negative_mask_uses_rescue_pool_when_normal_pool_shortfalls():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_decode_candidate_point_valid_thr": 0.20,
            "gcs_decode_candidate_min_points": 5,
            "gcs_enable_rescue_candidate_pool": True,
            "gcs_decode_rescue_candidate_point_valid_thr": 0.08,
            "gcs_decode_rescue_candidate_min_points": 4,
        }
    )
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.tensor([[5.0, 4.9, 4.8, 4.7, 4.6, 4.5]])
    pred_valid_logits = torch.full((1, 6, 56), _logit_prob(0.05))
    pred_valid_logits[0, 0:4, 10:16] = _logit_prob(0.95)
    pred_valid_logits[0, 4:6, 10:14] = _logit_prob(0.10)
    indices = [(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))]

    mask = criterion.fifth_candidate_negative_mask(pred_logits, pred_valid_logits, [valid4], indices)

    assert mask.tolist() == [[False, False, False, False, True, False]]


def test_fifth_candidate_negative_mask_does_not_use_rescue_when_normal_pool_has_five():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_decode_candidate_point_valid_thr": 0.20,
            "gcs_decode_candidate_min_points": 5,
            "gcs_enable_rescue_candidate_pool": True,
            "gcs_decode_rescue_candidate_point_valid_thr": 0.08,
            "gcs_decode_rescue_candidate_min_points": 4,
        }
    )
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.tensor([[5.0, 4.9, 4.8, 4.7, 4.6, 20.0]])
    pred_valid_logits = torch.full((1, 6, 56), _logit_prob(0.05))
    pred_valid_logits[0, 0:5, 10:16] = _logit_prob(0.95)
    pred_valid_logits[0, 5, 10:14] = _logit_prob(0.10)
    indices = [(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))]

    mask = criterion.fifth_candidate_negative_mask(pred_logits, pred_valid_logits, [valid4], indices)

    assert mask.tolist() == [[False, False, False, False, True, False]]


def test_fifth_candidate_negative_mask_sorts_after_rescue_shortfall_like_decode():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_decode_candidate_point_valid_thr": 0.20,
            "gcs_decode_candidate_min_points": 5,
            "gcs_enable_rescue_candidate_pool": True,
            "gcs_decode_rescue_candidate_point_valid_thr": 0.08,
            "gcs_decode_rescue_candidate_min_points": 4,
        }
    )
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.tensor([[5.0, 4.9, 4.8, 20.0, 4.7, 4.6]])
    pred_valid_logits = torch.full((1, 6, 56), _logit_prob(0.05))
    pred_valid_logits[0, 0:3, 10:16] = _logit_prob(0.95)
    pred_valid_logits[0, 3, 10:14] = _logit_prob(0.10)
    pred_valid_logits[0, 4, 10:14] = _logit_prob(0.09)
    pred_valid_logits[0, 5, 10:14] = _logit_prob(0.08)
    indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))]

    mask = criterion.fifth_candidate_negative_mask(pred_logits, pred_valid_logits, [valid4], indices)

    assert mask.tolist() == [[False, False, False, False, True, False]]


def test_fifth_candidate_negative_mask_filters_close_rescue_like_decode():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_decode_candidate_point_valid_thr": 0.20,
            "gcs_decode_candidate_min_points": 5,
            "gcs_enable_rescue_candidate_pool": True,
            "gcs_decode_rescue_candidate_point_valid_thr": 0.08,
            "gcs_decode_rescue_candidate_min_points": 4,
            "gcs_line_nms_rescue_dist_px": 30.0,
            "gcs_line_nms_min_overlap": 4,
        }
    )
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.tensor([[5.0, 4.9, 4.8, 4.7, 20.0, 4.6]])
    pred_valid_logits = torch.full((1, 6, 56), _logit_prob(0.05))
    pred_valid_logits[0, 0:4, 10:16] = _logit_prob(0.95)
    pred_valid_logits[0, 4, 10:14] = _logit_prob(0.10)
    pred_valid_logits[0, 5, 10:14] = _logit_prob(0.09)
    pred_points = torch.zeros(1, 6, 56, 2)
    pred_points[..., 1] = torch.linspace(0.9861111111111112, 0.2222222222222222, 56).view(1, 1, 56)
    for q, x in enumerate((0.10, 0.25, 0.45, 0.65, 0.105, 0.85)):
        pred_points[0, q, :, 0] = x
    indices = [(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))]

    mask = criterion.fifth_candidate_negative_mask(
        pred_logits,
        pred_valid_logits,
        [valid4],
        indices,
        pred_points=pred_points,
    )

    assert mask.tolist() == [[False, False, False, False, False, True]]


def test_fifth_candidate_negative_mask_filters_close_rescue_against_accepted_rescue():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_decode_candidate_point_valid_thr": 0.20,
            "gcs_decode_candidate_min_points": 5,
            "gcs_enable_rescue_candidate_pool": True,
            "gcs_decode_rescue_candidate_point_valid_thr": 0.08,
            "gcs_decode_rescue_candidate_min_points": 4,
            "gcs_line_nms_rescue_dist_px": 30.0,
            "gcs_line_nms_min_overlap": 4,
        }
    )
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.tensor([[5.0, 4.9, 4.8, 20.0, 19.0, 4.6]])
    pred_valid_logits = torch.full((1, 6, 56), _logit_prob(0.05))
    pred_valid_logits[0, 0:3, 10:16] = _logit_prob(0.95)
    pred_valid_logits[0, 3, 10:14] = _logit_prob(0.10)
    pred_valid_logits[0, 4, 10:14] = _logit_prob(0.10)
    pred_valid_logits[0, 5, 10:14] = _logit_prob(0.09)
    pred_points = torch.zeros(1, 6, 56, 2)
    pred_points[..., 1] = torch.linspace(0.9861111111111112, 0.2222222222222222, 56).view(1, 1, 56)
    for q, x in enumerate((0.10, 0.25, 0.45, 0.70, 0.705, 0.85)):
        pred_points[0, q, :, 0] = x
    indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))]

    mask = criterion.fifth_candidate_negative_mask(
        pred_logits,
        pred_valid_logits,
        [valid4],
        indices,
        pred_points=pred_points,
    )

    assert mask.tolist() == [[False, False, False, False, False, True]]


def test_fifth_candidate_negative_mask_includes_duplicate_fifth_candidate_only_for_gt34():
    criterion = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960]})
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    _, valid5 = _gt_fixed_y56([0.1, 0.25, 0.4, 0.55, 0.7])
    pred_logits = torch.tensor([[5.0, 4.8, 4.6, 4.4, 1.0, 4.2, 0.9]])
    pred_valid_logits = torch.full((1, 7, 56), 5.0)
    indices = [(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))]
    duplicate_negative_mask = torch.zeros(1, 7, dtype=torch.bool)
    duplicate_negative_mask[0, 4] = True
    duplicate_negative_mask[0, 5] = True
    duplicate_negative_mask[0, 6] = True

    mask = criterion.fifth_candidate_negative_mask(
        pred_logits,
        pred_valid_logits,
        [valid4],
        indices,
        duplicate_negative_mask=duplicate_negative_mask,
    )
    gt5_mask = criterion.fifth_candidate_negative_mask(
        pred_logits,
        pred_valid_logits,
        [valid5],
        indices,
        duplicate_negative_mask=duplicate_negative_mask,
    )

    assert mask.tolist() == [[False, False, False, False, True, True, True]]
    assert gt5_mask.tolist() == [[False, False, False, False, False, False, False]]


def test_fifth_candidate_negative_mask_requires_duplicate_to_be_decode_candidate():
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_decode_candidate_point_valid_thr": 0.20,
            "gcs_decode_candidate_min_points": 5,
            "gcs_enable_rescue_candidate_pool": False,
        }
    )
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.tensor([[5.0, 4.8, 4.6, 4.4, 1.0, 4.2, 0.9]])
    pred_valid_logits = torch.full((1, 7, 56), _logit_prob(0.05))
    pred_valid_logits[0, 0:6, 10:16] = _logit_prob(0.95)
    indices = [(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))]
    duplicate_negative_mask = torch.zeros(1, 7, dtype=torch.bool)
    duplicate_negative_mask[0, 5] = True
    duplicate_negative_mask[0, 6] = True

    mask = criterion.fifth_candidate_negative_mask(
        pred_logits,
        pred_valid_logits,
        [valid4],
        indices,
        duplicate_negative_mask=duplicate_negative_mask,
    )

    assert mask.tolist() == [[False, False, False, False, False, True, False]]


def test_fifth_candidate_negative_mask_marks_unmatched_top5_even_when_rank5_is_matched():
    criterion = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960]})
    _, valid4 = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_logits = torch.tensor([[5.0, 4.8, 4.6, 4.5, 4.4, 4.3]])
    pred_valid_logits = torch.full((1, 6, 56), 5.0)
    indices = [(torch.tensor([0, 1, 2, 4]), torch.tensor([0, 1, 2, 3]))]

    mask = criterion.fifth_candidate_negative_mask(pred_logits, pred_valid_logits, [valid4], indices)

    assert mask.tolist() == [[False, False, False, False, False, False]]


def test_count_conditioned_fifth_survival_negatives_are_gt34_rank5_candidates_only():
    lanes, valid = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_survival": 1.0,
            "gcs_survival_target_mode": "count_conditioned_fifth",
            "gcs_survival_neg_weight": 1.0,
            "gcs_fifth_gate_hard_negative_weight": 3.0,
        }
    )
    pred_logits = torch.tensor(
        [[_logit_prob(p) for p in (0.99, 0.98, 0.97, 0.80, 0.96, 0.95, 0.30, 0.20)]]
    )
    pred_survival_logits = torch.zeros(1, 8, requires_grad=True)
    pred_points = torch.zeros(1, 8, 56, 2)
    indices = [(torch.tensor([0, 1, 2, 4]), torch.tensor([0, 1, 2, 3]))]

    fifth_candidate_negative_mask = criterion.fifth_candidate_negative_mask(pred_logits, None, [valid], indices)
    expected_fifth = torch.zeros(1, 8, dtype=torch.bool)
    expected_fifth[0, 5] = True
    assert torch.equal(fifth_candidate_negative_mask.cpu(), expected_fifth)

    hard_negative_mask = torch.zeros(1, 8, dtype=torch.bool)
    hard_negative_mask[0, 5] = True
    hard_negative_mask[0, 6] = True
    duplicate_negative_mask = torch.zeros(1, 8, dtype=torch.bool)
    duplicate_negative_mask[0, 5] = True
    duplicate_negative_mask[0, 7] = True

    loss = criterion.survival_loss(
        pred_survival_logits,
        pred_points,
        indices,
        hard_negative_mask=hard_negative_mask,
        duplicate_negative_mask=duplicate_negative_mask,
        fifth_candidate_negative_mask=fifth_candidate_negative_mask,
        gt_points=[lanes],
        gt_valid=[valid],
        hard_loss_mask=torch.tensor([True]),
    )

    loss.backward()
    grad = pred_survival_logits.grad
    assert grad is not None
    assert torch.allclose(grad[0, [0, 1, 2, 3, 4]], torch.zeros(5))
    assert grad[0, 5] > 0.0
    assert torch.allclose(grad[0, 6:], torch.zeros(2))


def test_count_conditioned_fifth_survival_trains_duplicate_fifth_candidate_negative():
    lanes, valid = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_survival": 1.0,
            "gcs_survival_target_mode": "count_conditioned_fifth",
            "gcs_survival_neg_weight": 1.0,
        }
    )
    pred_survival_logits = torch.zeros(1, 7, requires_grad=True)
    pred_points = torch.zeros(1, 7, 56, 2)
    indices = [(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))]
    duplicate_negative_mask = torch.zeros(1, 7, dtype=torch.bool)
    duplicate_negative_mask[0, 5] = True
    fifth_candidate_negative_mask = torch.zeros(1, 7, dtype=torch.bool)
    fifth_candidate_negative_mask[0, 5] = True

    loss = criterion.survival_loss(
        pred_survival_logits,
        pred_points,
        indices,
        hard_negative_mask=torch.zeros(1, 7, dtype=torch.bool),
        duplicate_negative_mask=duplicate_negative_mask,
        fifth_candidate_negative_mask=fifth_candidate_negative_mask,
        gt_points=[lanes],
        gt_valid=[valid],
        hard_loss_mask=torch.tensor([False]),
    )

    loss.backward()
    grad = pred_survival_logits.grad
    assert grad is not None
    assert torch.allclose(grad[0, :5], torch.zeros(5))
    assert grad[0, 5] > 0.0
    assert torch.allclose(grad[0, 6:], torch.zeros(1))


def test_count_conditioned_fifth_survival_ignores_general_survival_hard_weights():
    lanes, valid = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    pred_points = torch.zeros(1, 7, 56, 2)
    indices = [(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))]
    fifth_candidate_negative_mask = torch.zeros(1, 7, dtype=torch.bool)
    fifth_candidate_negative_mask[0, 5] = True
    hard_negative_mask = torch.zeros(1, 7, dtype=torch.bool)
    duplicate_negative_mask = torch.zeros(1, 7, dtype=torch.bool)
    hard_negative_mask[0, 5] = True
    duplicate_negative_mask[0, 5] = True

    def grad_for_weights(hard_weight: float, duplicate_weight: float) -> torch.Tensor:
        criterion = GCSLoss(
            model={
                "gcs_point_mode": "fixed_y",
                "gcs_imgsz": [544, 960],
                "gcs_survival": 1.0,
                "gcs_survival_target_mode": "count_conditioned_fifth",
                "gcs_survival_neg_weight": 1.0,
                "gcs_survival_hard_negative_weight": hard_weight,
                "gcs_survival_duplicate_negative_weight": duplicate_weight,
                "gcs_fifth_gate_hard_negative_weight": 3.0,
            }
        )
        logits = torch.zeros(1, 7, requires_grad=True)
        loss = criterion.survival_loss(
            logits,
            pred_points,
            indices,
            hard_negative_mask=hard_negative_mask,
            duplicate_negative_mask=duplicate_negative_mask,
            fifth_candidate_negative_mask=fifth_candidate_negative_mask,
            gt_points=[lanes],
            gt_valid=[valid],
            hard_loss_mask=torch.tensor([True]),
        )
        loss.backward()
        return logits.grad.detach().clone()

    low = grad_for_weights(1.0, 1.0)
    high = grad_for_weights(20.0, 30.0)

    assert torch.allclose(high, low)
    assert high[0, 5] > 0.0


def test_count_conditioned_fifth_survival_ignores_global_hard_negatives_without_manifest_hit():
    lanes, valid = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_survival": 1.0,
            "gcs_survival_target_mode": "count_conditioned_fifth",
            "gcs_survival_neg_weight": 1.0,
        }
    )
    pred_survival_logits = torch.zeros(1, 7, requires_grad=True)
    pred_points = torch.zeros(1, 7, 56, 2)
    indices = [(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))]
    hard_negative_mask = torch.zeros(1, 7, dtype=torch.bool)
    hard_negative_mask[0, 5] = True
    duplicate_negative_mask = torch.zeros(1, 7, dtype=torch.bool)
    fifth_candidate_negative_mask = torch.zeros(1, 7, dtype=torch.bool)
    fifth_candidate_negative_mask[0, 6] = True

    loss = criterion.survival_loss(
        pred_survival_logits,
        pred_points,
        indices,
        hard_negative_mask=hard_negative_mask,
        duplicate_negative_mask=duplicate_negative_mask,
        fifth_candidate_negative_mask=fifth_candidate_negative_mask,
        gt_points=[lanes],
        gt_valid=[valid],
        hard_loss_mask=torch.tensor([False]),
    )

    loss.backward()
    grad = pred_survival_logits.grad
    assert grad is not None
    assert torch.allclose(grad[0, :6], torch.zeros(6))
    assert grad[0, 6] > 0.0


def test_count_conditioned_fifth_survival_rejects_bad_candidate_mask_shape():
    lanes, valid = _gt_fixed_y56([0.1, 0.25, 0.45, 0.65])
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_survival": 1.0,
            "gcs_survival_target_mode": "count_conditioned_fifth",
        }
    )

    with pytest.raises(ValueError, match="fifth_candidate_negative_mask must match"):
        criterion.survival_loss(
            torch.zeros(1, 7),
            torch.zeros(1, 7, 56, 2),
            [(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))],
            fifth_candidate_negative_mask=torch.zeros(1, 6, dtype=torch.bool),
            gt_points=[lanes],
            gt_valid=[valid],
        )


def test_decoder_aux_loss_backprops_from_intermediate_decoder_outputs():
    torch.manual_seed(8)
    head = GCSLaneHead(
        c1=16,
        num_queries=6,
        num_points=56,
        num_decoder_layers=3,
        nhead=4,
        point_mode="fixed_y",
        decoder_aux_loss=True,
    )
    head.min_spatial_tokens = 0
    feats = [
        torch.randn(1, 16, 8, 16),
        torch.randn(1, 16, 4, 8),
        torch.randn(1, 16, 3, 4),
        torch.randn(1, 16, 2, 3),
    ]
    out = head(feats)
    lanes, valid = _gt_fixed_y56([0.1, 0.3, 0.5])
    criterion = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960], "gcs_decoder_aux": 0.2})

    loss = criterion.decoder_aux_loss(out["aux_outputs"], [lanes], [valid])

    assert float(loss.detach()) > 0.0
    loss.backward()
    assert any(
        param.grad is not None and torch.count_nonzero(param.grad).item() > 0
        for name, param in head.named_parameters()
        if not name.startswith("count_head.") and param.requires_grad
    )


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
    assert math.isclose(DEFAULT_CFG_DICT["gcs_quality_point_weight"], GCS_MAINLINE_QUALITY_POINT_WEIGHT)
    assert math.isclose(DEFAULT_CFG_DICT["gcs_quality_gt5_edge_floor"], GCS_MAINLINE_QUALITY_GT5_EDGE_FLOOR)
    assert tuple(DEFAULT_CFG_DICT[f"gcs_count_cls_w{i}"] for i in range(2, 6)) == GCS_MAINLINE_COUNT_CLS_WEIGHTS
    assert math.isclose(
        DEFAULT_CFG_DICT["gcs_point_valid_gt5_pos_weight"], GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT
    )
    assert math.isclose(DEFAULT_CFG_DICT["gcs_gt5_edge_loss_weight"], GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT)
    assert DEFAULT_CFG_DICT["gcs_count_boundary"] == GCS_MAINLINE_COUNT_BOUNDARY_GAIN
    assert DEFAULT_CFG_DICT["gcs_count_boundary_label_smoothing"] == GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING
    assert DEFAULT_CFG_DICT["gcs_count_boundary_gt5_pos_weight"] == GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT
    assert DEFAULT_CFG_DICT["gcs_count_adjacent_margin"] == 0.2
    assert DEFAULT_CFG_DICT["gcs_count_adjacent_margin_gain"] == 0.0
    assert DEFAULT_CFG_DICT["gcs_count_adjacent_margin_gt45_weight"] == 1.0
    assert DEFAULT_CFG_DICT["gcs_candidate_gt5_edge_weight"] == GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT
    assert DEFAULT_CFG_DICT["gcs_point_valid_gt5_edge_continuity"] == GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY
    assert (
        DEFAULT_CFG_DICT["gcs_point_valid_gt5_edge_continuity_thr"]
        == GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR
    )
    assert DEFAULT_CFG_DICT["gcs_quality_hard_negative_from_head"] == GCS_MAINLINE_QUALITY_HARD_NEGATIVE_FROM_HEAD
    assert DEFAULT_CFG_DICT["gcs_hard_negative_visible_segment"] is False
    assert DEFAULT_CFG_DICT["gcs_hard_negative_visible_thr"] == 0.5
    assert DEFAULT_CFG_DICT["gcs_hard_negative_visible_support_points"] == 12.0
    assert DEFAULT_CFG_DICT["gcs_point_valid_gt5_edge_segment"] == GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT
    assert DEFAULT_CFG_DICT["gcs_point_valid_gt5_edge_segment_thr"] == GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_THR
    assert (
        DEFAULT_CFG_DICT["gcs_point_valid_gt5_edge_segment_min_points"]
        == GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_MIN_POINTS
    )
    assert DEFAULT_CFG_DICT["gcs_visible_count_sum"] == 0.0
    assert DEFAULT_CFG_DICT["gcs_visible_count_sum_quality_weight"] == 0.0
    assert DEFAULT_CFG_DICT["gcs_visible_count_sum_survival_weight"] == 0.0
    assert DEFAULT_CFG_DICT["gcs_survival"] == 0.0
    assert DEFAULT_CFG_DICT["gcs_survival_target_mode"] == "matched"
    assert DEFAULT_CFG_DICT["gcs_fifth_gate_hard_negative_weight"] == 2.0
    assert DEFAULT_CFG_DICT["gcs_decoder_aux"] == 0.0
    assert DEFAULT_CFG_DICT["gcs_count_boundary_hard_margin"] == 0.2
    assert DEFAULT_CFG_DICT["gcs_count_boundary_hard_margin_gain"] == 0.0
    assert DEFAULT_CFG_DICT["gcs_gt5_lane_aware_erasing"] is False
    assert DEFAULT_CFG_DICT["gcs_official_best_top_k"] == 1

    monkeypatch.setattr(sys, "argv", ["train_gcs.py"])
    args = train_gcs.parse_args()
    assert args.gcs_group_sampler_ratios == GCS_MAINLINE_GROUP_SAMPLER_RATIOS
    assert args.gcs_gt5_oversample_weight == GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT
    assert math.isclose(args.gcs_count_sum, GCS_MAINLINE_COUNT_SUM_GAIN)
    assert math.isclose(args.gcs_quality, GCS_MAINLINE_QUALITY_GAIN)
    assert math.isclose(args.gcs_quality_neg_weight, GCS_MAINLINE_QUALITY_NEG_WEIGHT)
    assert math.isclose(args.gcs_quality_point_weight, GCS_MAINLINE_QUALITY_POINT_WEIGHT)
    assert math.isclose(args.gcs_quality_gt5_edge_floor, GCS_MAINLINE_QUALITY_GT5_EDGE_FLOOR)
    assert tuple(getattr(args, f"gcs_count_cls_w{i}") for i in range(2, 6)) == GCS_MAINLINE_COUNT_CLS_WEIGHTS
    assert math.isclose(args.gcs_point_valid_gt5_pos_weight, GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT)
    assert math.isclose(args.gcs_gt5_edge_loss_weight, GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT)
    assert args.gcs_count_boundary == GCS_MAINLINE_COUNT_BOUNDARY_GAIN
    assert args.gcs_count_boundary_label_smoothing == GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING
    assert args.gcs_count_boundary_gt5_pos_weight == GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT
    assert args.gcs_count_adjacent_margin == 0.2
    assert args.gcs_count_adjacent_margin_gain == 0.0
    assert args.gcs_count_adjacent_margin_gt45_weight == 1.0
    assert args.gcs_candidate_gt5_edge_weight == GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT
    assert args.gcs_point_valid_gt5_edge_continuity == GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY
    assert args.gcs_point_valid_gt5_edge_continuity_thr == GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR
    assert args.gcs_quality_hard_negative_from_head == GCS_MAINLINE_QUALITY_HARD_NEGATIVE_FROM_HEAD
    assert args.gcs_hard_negative_visible_segment is False
    assert args.gcs_hard_negative_visible_thr == 0.5
    assert args.gcs_hard_negative_visible_support_points == 12.0
    assert args.gcs_point_valid_gt5_edge_segment == GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT
    assert args.gcs_point_valid_gt5_edge_segment_thr == GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_THR
    assert args.gcs_point_valid_gt5_edge_segment_min_points == GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_MIN_POINTS
    assert args.gcs_visible_count_sum == 0.0
    assert args.gcs_visible_count_sum_quality_weight == 0.0
    assert args.gcs_visible_count_sum_survival_weight == 0.0
    assert args.gcs_survival == 0.0
    assert args.gcs_survival_target_mode == "matched"
    assert args.gcs_fifth_gate_hard_negative_weight == 2.0
    assert args.gcs_decoder_aux == 0.0
    assert args.gcs_count_boundary_hard_margin == 0.2
    assert args.gcs_count_boundary_hard_margin_gain == 0.0
    assert args.gcs_gt5_lane_aware_erasing is False
    assert args.gcs_official_best_top_k == 1

    trainer_overrides = {}
    monkeypatch.setattr(BaseTrainer, "__init__", lambda self, cfg, overrides, callbacks: trainer_overrides.update(overrides))
    monkeypatch.setattr(GCSLaneTrainer, "_lock_gcs_shape_contract", lambda self: None)
    GCSLaneTrainer()
    assert trainer_overrides["gcs_group_sampler_ratios"] == GCS_MAINLINE_GROUP_SAMPLER_RATIOS
    assert trainer_overrides["gcs_gt5_oversample_weight"] == GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT
    assert math.isclose(trainer_overrides["gcs_count_sum"], GCS_MAINLINE_COUNT_SUM_GAIN)
    assert math.isclose(trainer_overrides["gcs_quality"], GCS_MAINLINE_QUALITY_GAIN)
    assert math.isclose(trainer_overrides["gcs_quality_neg_weight"], GCS_MAINLINE_QUALITY_NEG_WEIGHT)
    assert math.isclose(trainer_overrides["gcs_quality_point_weight"], GCS_MAINLINE_QUALITY_POINT_WEIGHT)
    assert math.isclose(trainer_overrides["gcs_quality_gt5_edge_floor"], GCS_MAINLINE_QUALITY_GT5_EDGE_FLOOR)
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
    assert trainer_overrides["gcs_quality_hard_negative_from_head"] == GCS_MAINLINE_QUALITY_HARD_NEGATIVE_FROM_HEAD
    assert trainer_overrides["gcs_hard_negative_visible_segment"] is False
    assert trainer_overrides["gcs_hard_negative_visible_thr"] == 0.5
    assert trainer_overrides["gcs_hard_negative_visible_support_points"] == 12.0
    assert trainer_overrides["gcs_point_valid_gt5_edge_segment"] == GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT
    assert (
        trainer_overrides["gcs_point_valid_gt5_edge_segment_thr"]
        == GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_THR
    )
    assert (
        trainer_overrides["gcs_point_valid_gt5_edge_segment_min_points"]
        == GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_MIN_POINTS
    )
    assert trainer_overrides["gcs_visible_count_sum"] == 0.0
    assert trainer_overrides["gcs_visible_count_sum_quality_weight"] == 0.0
    assert trainer_overrides["gcs_visible_count_sum_survival_weight"] == 0.0
    assert trainer_overrides["gcs_survival"] == 0.0
    assert trainer_overrides["gcs_survival_target_mode"] == "matched"
    assert trainer_overrides["gcs_fifth_gate_hard_negative_weight"] == 2.0
    assert trainer_overrides["gcs_decoder_aux"] == 0.0
    assert trainer_overrides["gcs_count_boundary_hard_margin"] == 0.2
    assert trainer_overrides["gcs_count_boundary_hard_margin_gain"] == 0.0
    assert trainer_overrides["gcs_gt5_lane_aware_erasing"] is False

    criterion = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960]})
    assert math.isclose(criterion.count_sum_gain, GCS_MAINLINE_COUNT_SUM_GAIN)
    assert math.isclose(criterion.quality_gain, GCS_MAINLINE_QUALITY_GAIN)
    assert math.isclose(criterion.quality_neg_weight, GCS_MAINLINE_QUALITY_NEG_WEIGHT)
    assert math.isclose(criterion.quality_point_weight, GCS_MAINLINE_QUALITY_POINT_WEIGHT)
    assert math.isclose(criterion.quality_gt5_edge_floor, GCS_MAINLINE_QUALITY_GT5_EDGE_FLOOR)
    assert criterion.count_cls_weights == GCS_MAINLINE_COUNT_CLS_WEIGHTS
    assert math.isclose(criterion.point_valid_gt5_pos_weight, GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT)
    assert math.isclose(criterion.gt5_edge_loss_weight, GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT)
    assert math.isclose(criterion.count_boundary_gain, GCS_MAINLINE_COUNT_BOUNDARY_GAIN)
    assert math.isclose(criterion.count_boundary_label_smoothing, GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING)
    assert math.isclose(criterion.count_boundary_gt5_pos_weight, GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT)
    assert math.isclose(criterion.count_adjacent_margin, 0.2)
    assert math.isclose(criterion.count_adjacent_margin_gain, 0.0)
    assert math.isclose(criterion.count_adjacent_margin_gt45_weight, 1.0)
    assert math.isclose(criterion.candidate_gt5_edge_weight, GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT)
    assert math.isclose(criterion.point_valid_gt5_edge_continuity, GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY)
    assert math.isclose(
        criterion.point_valid_gt5_edge_continuity_thr, GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR
    )
    assert criterion.quality_hard_negative_from_head == GCS_MAINLINE_QUALITY_HARD_NEGATIVE_FROM_HEAD
    assert criterion.hard_negative_visible_segment is False
    assert criterion.hard_negative_visible_thr == 0.5
    assert criterion.hard_negative_visible_support_points == 12.0
    assert math.isclose(criterion.point_valid_gt5_edge_segment, GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT)
    assert math.isclose(
        criterion.point_valid_gt5_edge_segment_thr, GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_THR
    )
    assert criterion.point_valid_gt5_edge_segment_min_points == GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT_MIN_POINTS
    assert criterion.visible_count_sum_gain == 0.0
    assert criterion.visible_count_sum_quality_weight == 0.0
    assert criterion.visible_count_sum_survival_weight == 0.0
    assert criterion.survival_gain == 0.0
    assert criterion.survival_target_mode == "matched"
    assert math.isclose(criterion.fifth_gate_hard_negative_weight, 2.0)
    assert criterion.decoder_aux_gain == 0.0
    assert math.isclose(criterion.count_boundary_hard_margin, 0.2)
    assert math.isclose(criterion.count_boundary_hard_margin_gain, 0.0)

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
        "visible_count_sum_loss",
        "quality_loss",
        "survival_loss",
        "decoder_aux_loss",
    )
    assert GCSLoss.loss_names == expected
    assert GCSLaneTrainer.loss_names == expected
    assert GCSLaneTrainer.progress_loss_names == expected
    assert VAL_LOSS_NAMES == expected

    validator = GCSLaneValidator(args=SimpleNamespace())
    gains = validator._loss_gains(torch.device("cpu"))
    assert gains.shape == (len(expected),)
    assert torch.isclose(gains[expected.index("count_sum_loss")], torch.tensor(0.03))
    assert torch.isclose(gains[expected.index("visible_count_sum_loss")], torch.tensor(0.0))
    assert torch.isclose(gains[expected.index("quality_loss")], torch.tensor(0.4))
    assert torch.isclose(gains[expected.index("survival_loss")], torch.tensor(0.0))
    assert torch.isclose(gains[expected.index("decoder_aux_loss")], torch.tensor(0.0))


def test_gt5_candidate_cfg_keys_have_expected_types():
    assert {
        "gcs_count_boundary_gt5_pos_weight",
        "gcs_count_boundary_hard_margin",
        "gcs_count_boundary_hard_margin_gain",
        "gcs_count_adjacent_margin",
        "gcs_count_adjacent_margin_gain",
        "gcs_count_adjacent_margin_gt45_weight",
        "gcs_candidate_gt5_edge_weight",
        "gcs_quality_point_weight",
        "gcs_quality_gt5_edge_floor",
        "gcs_hard_negative_visible_support_points",
        "gcs_point_valid_gt5_edge_continuity",
        "gcs_point_valid_gt5_edge_segment",
        "gcs_visible_count_sum",
        "gcs_visible_count_sum_support_points",
        "gcs_visible_count_sum_hard_weight",
        "gcs_visible_count_boundary",
        "gcs_visible_count_boundary_temperature",
        "gcs_survival",
        "gcs_survival_pos_weight",
        "gcs_survival_neg_weight",
        "gcs_fifth_gate_hard_negative_weight",
        "gcs_decoder_aux",
        "gcs_gt5_erasing_lane_margin_px",
    } <= CFG_FLOAT_KEYS
    assert "gcs_hard_negative_visible_thr" in CFG_FRACTION_KEYS
    assert "gcs_quality_point_weight" in CFG_FRACTION_KEYS
    assert "gcs_quality_gt5_edge_floor" in CFG_FRACTION_KEYS
    assert "gcs_visible_count_sum_visible_thr" in CFG_FRACTION_KEYS
    assert "gcs_visible_count_sum_quality_weight" in CFG_FRACTION_KEYS
    assert "gcs_visible_count_sum_survival_weight" in CFG_FRACTION_KEYS
    assert "gcs_visible_count_boundary_label_smoothing" in CFG_FRACTION_KEYS
    assert "gcs_point_valid_gt5_edge_continuity_thr" in CFG_FRACTION_KEYS
    assert "gcs_point_valid_gt5_edge_segment_thr" in CFG_FRACTION_KEYS
    assert "gcs_point_valid_gt5_edge_segment_min_points" in CFG_INT_KEYS
    assert "gcs_quality_hard_negative_from_head" in CFG_BOOL_KEYS
    assert "gcs_hard_negative_visible_segment" in CFG_BOOL_KEYS
    assert "gcs_visible_count_sum_normalize" in CFG_BOOL_KEYS
    assert "gcs_gt5_lane_aware_erasing" in CFG_BOOL_KEYS
    assert "gcs_official_best_top_k" in CFG_INT_KEYS


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


def test_count_boundary_hard_margin_pushes_only_hard_gt4_gt5_samples():
    _, valid_gt4 = _gt([0.1, 0.25, 0.4, 0.55])
    _, valid_gt5 = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    _, valid_gt3 = _gt([0.2, 0.4, 0.6])
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_count_boundary_hard_margin": 0.2,
            "gcs_count_boundary_hard_margin_gain": 1.0,
        }
    )
    boundary_logits = torch.tensor([[0.0, 0.5], [0.0, -0.5], [0.0, 0.5]], requires_grad=True)

    loss = criterion.count_boundary_hard_margin_loss(
        boundary_logits,
        [valid_gt4, valid_gt5, valid_gt3],
        torch.tensor([True, True, False]),
    )

    assert loss > 0.0
    loss.backward()
    assert boundary_logits.grad is not None
    assert boundary_logits.grad[0, 1] > 0.0
    assert boundary_logits.grad[1, 1] < 0.0
    assert torch.allclose(boundary_logits.grad[2], torch.zeros(2))


def test_count_boundary_hard_margin_rejects_bad_hard_mask_shape():
    _, valid_gt4 = _gt([0.1, 0.25, 0.4, 0.55])
    _, valid_gt5 = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_count_boundary_hard_margin_gain": 1.0,
        }
    )
    boundary_logits = torch.zeros(2, 2, requires_grad=True)

    with pytest.raises(ValueError, match="hard_loss_mask must contain one value per image"):
        criterion.count_boundary_hard_margin_loss(
            boundary_logits,
            [valid_gt4, valid_gt5],
            torch.tensor([True]),
        )


def test_count_adjacent_margin_is_default_off_for_count_loss():
    _, valid = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    pred_count_logits = torch.tensor([[0.0, 0.0, 2.0, 1.7]], requires_grad=True)
    preds = {"pred_count_logits": pred_count_logits}
    pred_points = torch.zeros(1, 5, 6, 2)
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_count_boundary": 0.0,
        "gcs_count_adjacent_margin": 0.2,
    }
    base = GCSLoss(model={**common, "gcs_count_adjacent_margin_gain": 0.0})
    explicit_off = GCSLoss(model={**common, "gcs_count_adjacent_margin_gain": 0.0})

    base_loss = base.count_head_loss(preds, pred_points, [valid])
    explicit_off_loss = explicit_off.count_head_loss(preds, pred_points, [valid])

    assert torch.isclose(explicit_off_loss, base_loss)


def test_count_adjacent_margin_penalizes_neighbor_count_confusion():
    _, valid = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    pred_count_logits = torch.tensor([[0.0, 0.0, 2.0, 1.7]], requires_grad=True)
    preds = {"pred_count_logits": pred_count_logits}
    pred_points = torch.zeros(1, 5, 6, 2)
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_count_boundary": 0.0,
        "gcs_count_adjacent_margin": 0.2,
    }
    base = GCSLoss(model={**common, "gcs_count_adjacent_margin_gain": 0.0})
    margin = GCSLoss(
        model={
            **common,
            "gcs_count_adjacent_margin_gain": 1.0,
            "gcs_count_adjacent_margin_gt45_weight": 1.5,
        }
    )

    gt_count, gt_count_cls, _ = margin.count_head_targets(pred_count_logits, [valid])
    margin_term = margin.count_adjacent_margin_loss(pred_count_logits, gt_count_cls, gt_count)
    base_loss = base.count_head_loss(preds, pred_points, [valid])
    adjacent_count_loss = margin.count_head_loss(preds, pred_points, [valid])

    assert margin_term > 0
    assert adjacent_count_loss > base_loss
    adjacent_count_loss.backward()
    assert pred_count_logits.grad is not None
    assert pred_count_logits.grad[0, 3] < 0


def test_count_adjacent_margin_gt45_weight_boosts_mixed_batch_gradient():
    _, valid_gt2 = _gt([0.2, 0.6])
    _, valid_gt5 = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    logits_base = torch.tensor([[1.0, 0.95, 0.0, 0.0], [0.0, 0.0, 0.95, 1.0]], requires_grad=True)
    logits_boosted = logits_base.detach().clone().requires_grad_(True)
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_count_boundary": 0.0,
        "gcs_count_adjacent_margin": 0.2,
    }
    base = GCSLoss(model={**common, "gcs_count_adjacent_margin_gt45_weight": 1.0})
    boosted = GCSLoss(model={**common, "gcs_count_adjacent_margin_gt45_weight": 1.5})

    gt_count, gt_count_cls, _ = boosted.count_head_targets(logits_base, [valid_gt2, valid_gt5])
    base.count_adjacent_margin_loss(logits_base, gt_count_cls, gt_count).backward()
    boosted.count_adjacent_margin_loss(logits_boosted, gt_count_cls, gt_count).backward()

    assert torch.isclose(logits_base.grad[0, 0].abs(), logits_base.grad[1, 3].abs())
    assert logits_boosted.grad[1, 3].abs() > logits_boosted.grad[0, 0].abs()
    assert logits_boosted.grad[1, 3].abs() > logits_base.grad[1, 3].abs()


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


def test_quality_hard_negative_from_head_increases_quality_loss():
    lanes, valid = _gt([0.3])
    pred_points = torch.zeros(1, 3, 6, 2)
    pred_points[0, 0] = lanes[0]
    pred_quality_logits = torch.tensor([[4.0, 4.0, -4.0]], requires_grad=True)
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    hard_negative_mask = torch.zeros(1, 3, dtype=torch.bool)
    duplicate_negative_mask = torch.zeros(1, 3, dtype=torch.bool)
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_quality_dist_thr_px": 100.0,
        "gcs_quality_neg_weight": 0.5,
        "gcs_quality_hard_negative_weight": 3.0,
        "gcs_hard_negative_quality_thr": 0.5,
        "gcs_hard_negative_topk": 1,
    }
    base = GCSLoss(model={**common, "gcs_quality_hard_negative_from_head": False})
    head_mined = GCSLoss(model={**common, "gcs_quality_hard_negative_from_head": True})

    base_loss = base.quality_loss(
        pred_quality_logits,
        pred_points,
        [lanes],
        [valid],
        indices,
        hard_negative_mask=hard_negative_mask,
        duplicate_negative_mask=duplicate_negative_mask,
    )
    head_loss = head_mined.quality_loss(
        pred_quality_logits,
        pred_points,
        [lanes],
        [valid],
        indices,
        hard_negative_mask=hard_negative_mask,
        duplicate_negative_mask=duplicate_negative_mask,
    )

    assert head_loss > base_loss
    head_loss.backward()
    assert pred_quality_logits.grad is not None


def test_quality_head_hard_negative_from_head_ignores_matched_zero_quality_lane():
    lanes, valid = _gt_fixed_y56([0.8])
    pred_points = torch.zeros(1, 3, 56, 2)
    pred_points[0, :, :, 1] = lanes[0, :, 1]
    pred_quality_logits = torch.tensor([[4.0, -4.0, -4.0]], requires_grad=True)
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    hard_negative_mask = torch.zeros(1, 3, dtype=torch.bool)
    duplicate_negative_mask = torch.zeros(1, 3, dtype=torch.bool)
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_quality_dist_thr_px": 5.0,
        "gcs_quality_neg_weight": 0.5,
        "gcs_quality_hard_negative_weight": 3.0,
        "gcs_hard_negative_quality_thr": 0.5,
        "gcs_hard_negative_topk": 0,
    }
    base = GCSLoss(model={**common, "gcs_quality_hard_negative_from_head": False})
    head_mined = GCSLoss(model={**common, "gcs_quality_hard_negative_from_head": True})

    target_quality = base.build_quality_targets(pred_quality_logits, pred_points, [lanes], [valid], indices)
    assert torch.isclose(target_quality[0, 0], torch.tensor(0.0))

    base_loss = base.quality_loss(
        pred_quality_logits,
        pred_points,
        [lanes],
        [valid],
        indices,
        hard_negative_mask=hard_negative_mask,
        duplicate_negative_mask=duplicate_negative_mask,
    )
    head_loss = head_mined.quality_loss(
        pred_quality_logits,
        pred_points,
        [lanes],
        [valid],
        indices,
        hard_negative_mask=hard_negative_mask,
        duplicate_negative_mask=duplicate_negative_mask,
    )

    assert torch.isclose(head_loss, base_loss)
    head_loss.backward()
    assert pred_quality_logits.grad is not None


def test_quality_gt5_edge_floor_only_boosts_matched_edge_targets():
    lanes, valid = _gt_fixed_y56([0.1, 0.3, 0.5, 0.7, 0.9])
    extra_lane = lanes[:1].clone()
    pred_points = torch.cat([lanes, extra_lane], dim=0).unsqueeze(0)
    pred_points[..., 0] = 0.0
    pred_quality_logits = torch.zeros(1, 6)
    indices = [(torch.arange(5), torch.arange(5))]
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_quality_dist_thr_px": 5.0,
    }
    base = GCSLoss(model={**common, "gcs_quality_gt5_edge_floor": 0.0})
    floored = GCSLoss(model={**common, "gcs_quality_gt5_edge_floor": 0.65})

    base_target = base.build_quality_targets(pred_quality_logits, pred_points, [lanes], [valid], indices)
    floor_target = floored.build_quality_targets(pred_quality_logits, pred_points, [lanes], [valid], indices)

    assert torch.allclose(floor_target[0, [0, 4]], torch.full((2,), 0.65))
    assert torch.allclose(floor_target[0, 1:4], base_target[0, 1:4])
    assert floor_target[0, 5].item() == 0.0

    lanes4, valid4 = _gt_fixed_y56([0.1, 0.3, 0.5, 0.7])
    pred_points4 = lanes4.unsqueeze(0).clone()
    pred_points4[..., 0] = 0.0
    pred_quality_logits4 = torch.zeros(1, 4)
    indices4 = [(torch.arange(4), torch.arange(4))]

    base_target4 = base.build_quality_targets(pred_quality_logits4, pred_points4, [lanes4], [valid4], indices4)
    floor_target4 = floored.build_quality_targets(pred_quality_logits4, pred_points4, [lanes4], [valid4], indices4)
    assert torch.allclose(floor_target4, base_target4)


def test_quality_point_weight_blends_point_inlier_and_line_iou_targets():
    lanes, valid = _gt([0.5], points=6)
    pred_points = lanes.unsqueeze(0).clone()
    pred_points[0, 0, :3, 0] += 30.0 / 960.0
    pred_quality_logits = torch.zeros(1, 1)
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_quality_dist_thr_px": 20.0,
        "gcs_line_iou_width_px": 15.0,
    }

    point_only = GCSLoss(model={**common, "gcs_quality_point_weight": 1.0})
    line_only = GCSLoss(model={**common, "gcs_quality_point_weight": 0.0})
    default = GCSLoss(model={**common, "gcs_quality_point_weight": 0.5})
    point_heavy = GCSLoss(model={**common, "gcs_quality_point_weight": 0.8})

    point_target = point_only.build_quality_targets(pred_quality_logits, pred_points, [lanes], [valid], indices)
    line_target = line_only.build_quality_targets(pred_quality_logits, pred_points, [lanes], [valid], indices)
    default_target = default.build_quality_targets(pred_quality_logits, pred_points, [lanes], [valid], indices)
    point_heavy_target = point_heavy.build_quality_targets(pred_quality_logits, pred_points, [lanes], [valid], indices)

    assert point_target[0, 0] > line_target[0, 0]
    assert torch.allclose(default_target, 0.5 * point_target + 0.5 * line_target)
    assert torch.allclose(point_heavy_target, 0.8 * point_target + 0.2 * line_target)


def test_visible_segment_hard_negative_mining_is_default_off():
    lanes, valid = _gt_fixed_y56([0.2])
    high = math.log(0.95 / 0.05)
    low = math.log(0.05 / 0.95)
    pred_logits = torch.full((1, 3), high)
    pred_valid_logits = torch.full((1, 3, 56), low)
    pred_valid_logits[0, 0, :] = high
    pred_valid_logits[0, 1, 20:26] = high
    pred_points = torch.zeros(1, 3, 56, 2)
    pred_points[0, :, :, 1] = lanes[0, :, 1]
    pred_points[0, 0, :, 0] = 0.2
    pred_points[0, 1, :, 0] = 0.85
    pred_points[0, 2, :, 0] = 0.55
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_hard_negative_quality_thr": 0.4,
        "gcs_hard_negative_topk": 0,
    }
    base = GCSLoss(model=common)
    explicit_off = GCSLoss(model={**common, "gcs_hard_negative_visible_segment": False})

    base_hard, _ = base.negative_query_masks(pred_logits, pred_points, pred_valid_logits, [lanes], [valid], indices)
    off_hard, _ = explicit_off.negative_query_masks(
        pred_logits,
        pred_points,
        pred_valid_logits,
        [lanes],
        [valid],
        indices,
    )

    assert torch.equal(base_hard, off_hard)
    assert base_hard.tolist() == [[False, False, False]]


def test_visible_segment_hard_negative_mining_selects_short_unmatched_candidate():
    lanes, valid = _gt_fixed_y56([0.2])
    high = math.log(0.95 / 0.05)
    low = math.log(0.05 / 0.95)
    pred_logits = torch.full((1, 3), high)
    pred_valid_logits = torch.full((1, 3, 56), low)
    pred_valid_logits[0, 0, :] = high
    pred_valid_logits[0, 1, 20:26] = high
    pred_points = torch.zeros(1, 3, 56, 2)
    pred_points[0, :, :, 1] = lanes[0, :, 1]
    pred_points[0, 0, :, 0] = 0.2
    pred_points[0, 1, :, 0] = 0.85
    pred_points[0, 2, :, 0] = 0.55
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_hard_negative_quality_thr": 0.4,
            "gcs_hard_negative_topk": 0,
            "gcs_hard_negative_visible_segment": True,
            "gcs_hard_negative_visible_thr": 0.5,
            "gcs_hard_negative_visible_support_points": 12.0,
        }
    )

    all_anchor_quality = pred_logits.sigmoid() * pred_valid_logits.sigmoid().mean(dim=-1)
    visible_mean, visible_support = GCSLoss._visible_segment_mean_and_support(
        pred_valid_logits.sigmoid(),
        visible_thr=0.5,
        support_points=12.0,
    )
    hard_negative, _ = criterion.negative_query_masks(
        pred_logits,
        pred_points,
        pred_valid_logits,
        [lanes],
        [valid],
        indices,
    )

    assert all_anchor_quality[0, 1] < 0.4
    assert visible_mean[0, 1] > 0.94
    assert torch.isclose(visible_support[0, 1], torch.tensor(0.5))
    assert hard_negative.tolist() == [[False, True, False]]


def test_visible_segment_hard_negative_recipe_keeps_matched_queries_protected():
    lanes, valid = _gt_fixed_y56([0.2])
    high = math.log(0.95 / 0.05)
    low = math.log(0.05 / 0.95)
    pred_logits = torch.full((1, 2), high)
    pred_valid_logits = torch.full((1, 2, 56), low)
    pred_valid_logits[0, 0, 20:26] = high
    pred_valid_logits[0, 1, 20:26] = high
    pred_points = torch.zeros(1, 2, 56, 2)
    pred_points[0, :, :, 1] = lanes[0, :, 1]
    pred_points[0, 0, :, 0] = 0.2
    pred_points[0, 1, :, 0] = 0.85
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_hard_negative_quality_thr": 0.4,
            "gcs_hard_negative_topk": 2,
            "gcs_hard_negative_visible_segment": True,
            "gcs_hard_negative_visible_thr": 0.5,
            "gcs_hard_negative_visible_support_points": 12.0,
        }
    )

    hard_negative, _ = criterion.negative_query_masks(
        pred_logits,
        pred_points,
        pred_valid_logits,
        [lanes],
        [valid],
        indices,
    )

    assert hard_negative.tolist() == [[False, True]]


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


def test_point_valid_gt5_edge_segment_adds_loss():
    lanes, valid = _gt([0.1, 0.25, 0.4, 0.55, 0.7])
    pred_points = lanes.unsqueeze(0).clone()
    logits = torch.full((1, 5, 6), 4.0)
    logits[0, 0, 1:4] = -3.0
    logits[0, 4, 2:5] = -3.0
    pred_valid_logits = logits.requires_grad_()
    indices = [(torch.arange(5), torch.arange(5))]
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_gt5_edge_loss_weight": 1.0,
        "gcs_candidate_gt5_edge_weight": 1.0,
        "gcs_point_valid_gt5_pos_weight": 1.0,
        "gcs_point_valid_unmatched_weight": 1.0,
        "gcs_point_valid_gt5_edge_continuity": 0.0,
    }
    base = GCSLoss(model={**common, "gcs_point_valid_gt5_edge_segment": 0.0})
    segment = GCSLoss(
        model={
            **common,
            "gcs_point_valid_gt5_edge_segment": 0.5,
            "gcs_point_valid_gt5_edge_segment_thr": 0.8,
            "gcs_point_valid_gt5_edge_segment_min_points": 5,
        }
    )

    base_loss = base.point_valid_loss(pred_valid_logits, pred_points, [valid], indices, gt_points=[lanes])
    segment_loss = segment.point_valid_loss(pred_valid_logits, pred_points, [valid], indices, gt_points=[lanes])

    assert segment_loss > base_loss
    segment_loss.backward()
    assert pred_valid_logits.grad is not None


def test_point_valid_gt5_edge_segment_uses_fixed_y56_edge_queries_only():
    visible_start, visible_end = 34, 42
    lanes5, valid5 = _gt_fixed_y56(
        [0.1, 0.25, 0.4, 0.55, 0.7],
        visible_start=visible_start,
        visible_end=visible_end,
    )
    pred_points5 = lanes5.unsqueeze(0).clone()
    indices5 = [(torch.arange(5), torch.arange(5))]
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_gt5_edge_loss_weight": 1.0,
        "gcs_candidate_gt5_edge_weight": 1.0,
        "gcs_point_valid_gt5_pos_weight": 1.0,
        "gcs_point_valid_unmatched_weight": 1.0,
        "gcs_point_valid_gt5_edge_continuity": 0.0,
    }
    base = GCSLoss(model={**common, "gcs_point_valid_gt5_edge_segment": 0.0})
    segment = GCSLoss(
        model={
            **common,
            "gcs_point_valid_gt5_edge_segment": 0.5,
            "gcs_point_valid_gt5_edge_segment_thr": 0.8,
            "gcs_point_valid_gt5_edge_segment_min_points": 5,
        }
    )

    edge_logits = torch.full((1, 5, 56), 4.0)
    edge_logits[0, 0, visible_start:visible_end] = -3.0
    edge_logits[0, 4, visible_start:visible_end] = -3.0
    edge_logits = edge_logits.requires_grad_()
    edge_base_loss = base.point_valid_loss(edge_logits, pred_points5, [valid5], indices5, gt_points=[lanes5])
    edge_segment_loss = segment.point_valid_loss(edge_logits, pred_points5, [valid5], indices5, gt_points=[lanes5])
    assert edge_segment_loss > edge_base_loss
    edge_segment_loss.backward()
    assert edge_logits.grad is not None

    middle_logits = torch.full((1, 5, 56), 4.0)
    middle_logits[0, 2, visible_start:visible_end] = -3.0
    middle_base_loss = base.point_valid_loss(middle_logits, pred_points5, [valid5], indices5, gt_points=[lanes5])
    middle_segment_loss = segment.point_valid_loss(middle_logits, pred_points5, [valid5], indices5, gt_points=[lanes5])
    assert torch.isclose(middle_segment_loss, middle_base_loss)

    lanes4, valid4 = _gt_fixed_y56(
        [0.1, 0.25, 0.55, 0.7],
        visible_start=visible_start,
        visible_end=visible_end,
    )
    pred_points4 = lanes4.unsqueeze(0).clone()
    indices4 = [(torch.arange(4), torch.arange(4))]
    gt4_logits = torch.full((1, 4, 56), 4.0)
    gt4_logits[0, 0, visible_start:visible_end] = -3.0
    gt4_logits[0, 3, visible_start:visible_end] = -3.0
    gt4_base_loss = base.point_valid_loss(gt4_logits, pred_points4, [valid4], indices4, gt_points=[lanes4])
    gt4_segment_loss = segment.point_valid_loss(gt4_logits, pred_points4, [valid4], indices4, gt_points=[lanes4])
    assert torch.isclose(gt4_segment_loss, gt4_base_loss)


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

    quality_criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_gt5_edge_loss_weight": 1.0,
            "gcs_candidate_gt5_edge_weight": 1.0,
            "gcs_hard_loss_file": str(manifest),
            "gcs_hard_loss_lane_counts": "5",
            "gcs_hard_edge_loss_weight_by_count": "4:1.15,5:1.6",
            "gcs_hard_edge_loss_terms": "exist,point,point_valid,line_iou,quality",
            "gcs_hard_edge_only": True,
        }
    )
    quality_weighted = quality_criterion._matched_target_weights(
        lanes5,
        valid5,
        torch.tensor([0, 1, 2, 3, 4]),
        device=torch.device("cpu"),
        dtype=torch.float32,
        hard_image=True,
        term="quality",
    )
    assert torch.isclose(quality_weighted[0], torch.tensor(1.6))
    assert torch.isclose(quality_weighted[2], torch.tensor(1.0))
    assert torch.isclose(quality_weighted[4], torch.tensor(1.6))


def test_hard_loss_lane_count_filter_uses_shared_count_min_gt_points(tmp_path):
    manifest = tmp_path / "hard.txt"
    manifest.write_text("short5.jpg\n", encoding="utf-8")
    valid = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    batch = {
        "raw_file": ["short5.jpg"],
        "im_file": ["D:/data/images/train/short5.jpg"],
        "label_file": ["D:/data/labels_gcs/train/short5.npz"],
    }

    count_min1 = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_hard_loss_file": str(manifest),
            "gcs_hard_loss_lane_counts": "5",
            "gcs_count_min_gt_points": 1,
        }
    )
    count_min2 = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_hard_loss_file": str(manifest),
            "gcs_hard_loss_lane_counts": "5",
            "gcs_count_min_gt_points": 2,
        }
    )

    assert count_min1.hard_loss_mask(batch, 1, torch.device("cpu"), gt_valid=valid).tolist() == [True]
    assert count_min2.hard_loss_mask(batch, 1, torch.device("cpu"), gt_valid=valid).tolist() == [False]


def test_survival_head_replaces_quality_for_fifth_lane_rescue_gate():
    y = torch.linspace(0.98, 0.25, 6)
    pred_points = torch.stack(
        [torch.stack((torch.full_like(y, x), y), dim=-1) for x in (0.1, 0.25, 0.4, 0.55, 0.7)],
        dim=0,
    )
    common = {
        "pred_points": pred_points,
        "pred_logits": torch.tensor([5.0, 4.8, 4.6, 4.4, 4.2]),
        "pred_valid_logits": torch.full((5, 6), 5.0),
        "pred_count_logits": torch.tensor([-5.0, -5.0, -5.0, 5.0]),
        "image_shape": (544, 960),
        "score_thr": 0.0,
        "point_valid_thr": 0.5,
        "min_points": 5,
        "max_det": 5,
        "nms_dist_px": 0.0,
        "quality_rescue_count5_thr": 0.7,
        "quality_rescue_quality_thr": 0.55,
        "quality_rescue_dist_px": 0.0,
        "return_meta": True,
    }

    blocked, blocked_meta = decode_gcs_predictions(
        pred_quality_logits=torch.full((5,), _logit_prob(0.95)),
        pred_survival_logits=torch.tensor(
            [_logit_prob(0.95), _logit_prob(0.95), _logit_prob(0.95), _logit_prob(0.95), _logit_prob(0.05)]
        ),
        **common,
    )
    rescued, rescued_meta = decode_gcs_predictions(
        pred_quality_logits=torch.tensor(
            [_logit_prob(0.95), _logit_prob(0.95), _logit_prob(0.95), _logit_prob(0.95), _logit_prob(0.05)]
        ),
        pred_survival_logits=torch.full((5,), _logit_prob(0.95)),
        **common,
    )

    assert len(blocked) == 4
    assert blocked_meta["rescue_reason"] == "survival_too_low"
    assert blocked_meta["rescue_candidate_gate_source"] == "survival"
    assert blocked_meta["rescue_candidate_gate_score"] < 0.55
    assert blocked_meta["rescue_candidate_quality"] > 0.9
    assert len(rescued) == 5
    assert rescued_meta["rescue_reason"] == "rescued"
    assert rescued_meta["rescue_candidate_gate_source"] == "survival"
    assert rescued_meta["rescue_candidate_gate_score"] > 0.9
    assert rescued_meta["rescue_candidate_quality"] < 0.1
    assert rescued[-1]["survival_score"] > 0.9
    assert rescued[-1]["quality_score"] < 0.1


def test_edge_count4_to5_upgrade_uses_survival_gated_edge_candidate():
    y = torch.linspace(0.98, 0.25, 6)
    pred_points = torch.stack(
        [torch.stack((torch.full_like(y, x), y), dim=-1) for x in (0.1, 0.25, 0.4, 0.55, 0.70, 0.88)],
        dim=0,
    )

    lanes, meta = decode_gcs_predictions(
        pred_points=pred_points,
        pred_logits=torch.tensor([5.0, 4.9, 4.8, 4.7, 4.6, 4.0]),
        pred_valid_logits=torch.full((6, 6), 5.0),
        pred_count_logits=torch.tensor([-6.0, -6.0, 5.0, 4.95]),
        pred_quality_logits=torch.full((6,), _logit_prob(0.95)),
        pred_survival_logits=torch.tensor(
            [
                _logit_prob(0.95),
                _logit_prob(0.95),
                _logit_prob(0.95),
                _logit_prob(0.95),
                _logit_prob(0.05),
                _logit_prob(0.95),
            ]
        ),
        image_shape=(544, 960),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=5,
        max_det=5,
        nms_dist_px=0.0,
        edge_rescue_dist_px=0.0,
        return_meta=True,
    )

    queries = {int(lane["query"]) for lane in lanes}
    assert len(lanes) == 5
    assert 5 in queries
    assert 4 not in queries
    assert meta["edge_count4_to5_upgrade"] is True
    assert meta["edge_count4_to5_upgrade_success"] is True
    assert meta["edge_last_lane_rescue_reason"] == "rescued"
    assert meta["edge_last_lane_rescue_candidate_gate_source"] == "survival"
    assert meta["edge_last_lane_rescue_candidate_gate_score"] > 0.9


def test_quality_head_keeps_quality_too_low_rescue_reason_without_survival_head():
    y = torch.linspace(0.98, 0.25, 6)
    pred_points = torch.stack(
        [torch.stack((torch.full_like(y, x), y), dim=-1) for x in (0.1, 0.25, 0.4, 0.55, 0.7)],
        dim=0,
    )

    lanes, meta = decode_gcs_predictions(
        pred_points=pred_points,
        pred_logits=torch.tensor([5.0, 4.8, 4.6, 4.4, 4.2]),
        pred_valid_logits=torch.full((5, 6), 5.0),
        pred_count_logits=torch.tensor([-5.0, -5.0, -5.0, 5.0]),
        pred_quality_logits=torch.tensor(
            [_logit_prob(0.95), _logit_prob(0.95), _logit_prob(0.95), _logit_prob(0.95), _logit_prob(0.05)]
        ),
        image_shape=(544, 960),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=5,
        max_det=5,
        nms_dist_px=0.0,
        quality_rescue_count5_thr=0.7,
        quality_rescue_quality_thr=0.55,
        quality_rescue_dist_px=0.0,
        return_meta=True,
    )

    assert len(lanes) == 4
    assert meta["rescue_reason"] == "quality_too_low"
    assert meta["rescue_candidate_gate_source"] == "quality"


def test_lane_aware_gt5_erasing_avoids_visible_lane_anchors(monkeypatch):
    dataset = object.__new__(GCSLaneDataset)
    dataset.gt5_erasing_lane_margin_px = 8.0
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lanes = np.array([[[0.5, 0.5]]], dtype=np.float32)
    lane_valid = np.array([[1.0]], dtype=np.float32)
    uniform_values = iter([0.04, 1.0, 0.04, 1.0])
    randint_values = iter([40, 40, 0, 0])
    calls = []

    monkeypatch.setattr("ultralytics.data.dataset_gcs.random.random", lambda: 0.0)
    monkeypatch.setattr("ultralytics.data.dataset_gcs.random.uniform", lambda *_: next(uniform_values))
    monkeypatch.setattr("ultralytics.data.dataset_gcs.random.randint", lambda *_: next(randint_values))

    out = dataset._apply_lane_aware_random_erasing(img, lanes, lane_valid, probability=1.0)

    assert np.array_equal(out[50, 50], np.array([0, 0, 0], dtype=np.uint8))
    assert np.array_equal(out[5, 5], np.array([114, 114, 114], dtype=np.uint8))


def test_lane_aware_gt5_erasing_avoids_visible_lane_segments(monkeypatch):
    dataset = object.__new__(GCSLaneDataset)
    dataset.gt5_erasing_lane_margin_px = 4.0
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lanes = np.array([[[0.2, 0.5], [0.8, 0.5]]], dtype=np.float32)
    lane_valid = np.array([[1.0, 1.0]], dtype=np.float32)
    uniform_values = iter([0.04, 1.0, 0.04, 1.0])
    randint_values = iter([40, 40, 0, 0])

    monkeypatch.setattr("ultralytics.data.dataset_gcs.random.random", lambda: 0.0)
    monkeypatch.setattr("ultralytics.data.dataset_gcs.random.uniform", lambda *_: next(uniform_values))
    monkeypatch.setattr("ultralytics.data.dataset_gcs.random.randint", lambda *_: next(randint_values))

    out = dataset._apply_lane_aware_random_erasing(img, lanes, lane_valid, probability=1.0)

    assert np.array_equal(out[50, 50], np.array([0, 0, 0], dtype=np.uint8))
    assert np.array_equal(out[5, 5], np.array([114, 114, 114], dtype=np.uint8))


def test_lane_aware_gt5_erasing_does_not_bridge_invalid_anchor_gaps():
    lanes = np.array([[[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]]], dtype=np.float32)
    lane_valid = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)

    boxes = GCSLaneDataset._visible_lane_anchor_boxes(lanes, lane_valid, (100, 100), margin_px=4.0)

    bridge_box = (48, 48, 53, 53)
    left_anchor_box = (18, 48, 23, 53)
    assert not any(GCSLaneDataset._boxes_overlap(bridge_box, box) for box in boxes)
    assert any(GCSLaneDataset._boxes_overlap(left_anchor_box, box) for box in boxes)


def test_gt5_lane_aware_erasing_also_guards_base_erasing(monkeypatch):
    dataset = object.__new__(GCSLaneDataset)
    dataset.erasing = 1.0
    dataset.gt5_lane_aware_erasing = True
    dataset.gt5_erasing_lane_margin_px = 8.0
    dataset.gt5_blur = 0.0
    dataset.gt5_noise = 0.0
    dataset.gt5_shadow = 0.0
    dataset.gt5_erasing = 0.0
    dataset.scale = 0.0
    dataset.translate = 0.0
    dataset.fliplr = 0.0
    dataset.flipud = 0.0
    dataset.hsv_h = 0.0
    dataset.hsv_s = 0.0
    dataset.hsv_v = 0.0
    dataset.img_h = 100
    dataset.img_w = 100
    dataset.imgsz = (100, 100)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    lanes = np.array([[[0.5, 0.6], [0.5, 0.4]]], dtype=np.float32)
    lane_valid = np.array([[1.0, 1.0]], dtype=np.float32)
    uniform_values = iter([0.04, 1.0, 0.04, 1.0])
    randint_values = iter([40, 40, 0, 0])
    calls = []

    monkeypatch.setattr("ultralytics.data.dataset_gcs.random.random", lambda: 0.0)
    monkeypatch.setattr("ultralytics.data.dataset_gcs.random.uniform", lambda *_: next(uniform_values))
    monkeypatch.setattr("ultralytics.data.dataset_gcs.random.randint", lambda *_: next(randint_values))
    original_lane_aware = GCSLaneDataset._apply_lane_aware_random_erasing
    original_plain = GCSLaneDataset._apply_random_erasing

    def tracked_lane_aware(self, *args, **kwargs):
        calls.append("lane_aware")
        return original_lane_aware(self, *args, **kwargs)

    def tracked_plain(self, *args, **kwargs):
        calls.append("plain")
        return original_plain(self, *args, **kwargs)

    monkeypatch.setattr(GCSLaneDataset, "_apply_lane_aware_random_erasing", tracked_lane_aware)
    monkeypatch.setattr(GCSLaneDataset, "_apply_random_erasing", tracked_plain)

    out, out_lanes, out_valid = dataset._apply_augmentations(img, lanes, lane_valid, gt5_extra=True)

    assert np.any(out == 114)
    assert np.array_equal(out_lanes, lanes)
    assert np.array_equal(out_valid, lane_valid)
    assert calls == ["lane_aware"]


def test_decoder_aux_loss_forward_path_returns_full_loss_vector():
    torch.manual_seed(9)
    head = GCSLaneHead(
        c1=16,
        num_queries=6,
        num_points=56,
        num_decoder_layers=3,
        nhead=4,
        point_mode="fixed_y",
        decoder_aux_loss=True,
    )
    head.min_spatial_tokens = 0
    feats = [
        torch.randn(1, 16, 8, 16),
        torch.randn(1, 16, 4, 8),
        torch.randn(1, 16, 3, 4),
        torch.randn(1, 16, 2, 3),
    ]
    preds = head(feats)
    lanes, valid = _gt_fixed_y56([0.1, 0.3, 0.5])
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_decoder_aux": 0.2,
        }
    )

    total, items = criterion(
        preds,
        {
            "img": torch.zeros(1, 3, 544, 960),
            "lanes": [lanes],
            "lane_valid": [valid],
            "num_lanes": torch.tensor([3]),
        },
    )

    assert torch.isfinite(total)
    assert items.shape == (len(GCSLoss.loss_names),)
    aux_idx = GCSLoss.loss_names.index("decoder_aux_loss")
    assert items[aux_idx] > 0.0


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


def test_soft_count_decision_prefers_survival_score_when_present():
    lanes = [
        {
            "rank_score": 1.0,
            "quality_score": 1.0,
            "survival_score": 1.0,
            "valid_count": 6,
            "points_norm": _cand(0.1 + i * 0.1, q=i, rank=i + 1).points.numpy(),
        }
        for i in range(5)
    ]
    lanes[-1]["quality_score"] = 1.0
    lanes[-1]["survival_score"] = -5.0

    meta = soft_count_decision([0.01, 0.10, 0.46, 0.43], lanes, prob_margin=0.08, min_points=5)

    assert meta["pred_count_cls_raw"] == 4
    assert meta["pred_count_cls_soft"] == 4
