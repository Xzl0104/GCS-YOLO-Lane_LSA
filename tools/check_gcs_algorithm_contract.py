# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""End-to-end contract checks for the current GCS lane algorithm."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import sweep_tusimple_official as official_sweep  # noqa: E402
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.nn.modules.gcs_lane import GCSLaneHead  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402
from ultralytics.utils.gcs_postprocess import (  # noqa: E402
    _edge_side_and_gap,
    count_head_decode_meta,
    decode_gcs_predictions,
    lane_nms,
)


ALLOWED_LOSSES = (
    "exist_loss",
    "point_loss",
    "point_valid_loss",
    "line_iou_loss",
    "count_cls_loss",
    "count_sum_loss",
    "quality_loss",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _lane_points(xs: list[float], points: int = 6, width: int = 960) -> torch.Tensor:
    y = torch.linspace(0.98, 0.25, points)
    return torch.stack(
        [torch.stack((torch.full_like(y, float(x) / float(width)), y), dim=-1) for x in xs],
        dim=0,
    )


def _valid_logits(count: int, points: int = 6) -> torch.Tensor:
    return torch.full((count, points), 10.0)


def check_loss_items() -> None:
    _assert(tuple(GCSLoss.loss_names) == ALLOWED_LOSSES, f"unexpected GCS loss names: {GCSLoss.loss_names}")


def check_count_sum_loss_gradient() -> None:
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_count_sum": 0.02,
            "gcs_line_iou": 0.0,
            "gcs_exist_quality_lane_iou_alpha": 0.0,
            "gcs_quality": 0.0,
        }
    )
    pred_logits = torch.randn(2, 6, requires_grad=True)
    gt_valid = [torch.ones((3, 6), dtype=torch.float32), torch.ones((5, 6), dtype=torch.float32)]
    loss = criterion.count_sum_loss(pred_logits, {}, gt_valid)
    _assert(torch.isfinite(loss) and float(loss.detach()) > 0.0, "count_sum_loss must be finite and positive")
    loss.backward()
    _assert(pred_logits.grad is not None, "count_sum_loss must backpropagate to pred_logits")


def check_quality_head_shape_and_gradient() -> None:
    torch.manual_seed(1)
    head = GCSLaneHead(
        c1=32,
        num_queries=6,
        num_points=8,
        num_decoder_layers=1,
        nhead=4,
        point_mode="fixed_y",
    )
    head.min_spatial_tokens = 0
    xs = [
        torch.randn(2, 32, 8, 16),
        torch.randn(2, 32, 4, 8),
        torch.randn(2, 32, 3, 4),
        torch.randn(2, 32, 2, 3),
    ]
    out = head(xs)
    _assert(out["pred_quality_logits"].shape == out["pred_logits"].shape, "quality logits must be B x Q")
    _assert(out["pred_count_logits"].shape == (2, 4), "count logits must be B x 4")

    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_quality": 0.3,
            "gcs_quality_dist_thr_px": 20.0,
            "gcs_quality_neg_weight": 0.25,
        }
    )
    gt_points = [out["pred_points"][0, :2].detach().clone(), out["pred_points"][1, :2].detach().clone()]
    gt_valid = [torch.ones((2, 8), dtype=torch.float32), torch.ones((2, 8), dtype=torch.float32)]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1])), (torch.tensor([0, 1]), torch.tensor([0, 1]))]
    loss = criterion.quality_loss(out["pred_quality_logits"], out["pred_points"], gt_points, gt_valid, indices)
    loss.backward()
    has_quality_grad = any(
        "quality" in name and param.grad is not None and torch.isfinite(param.grad).all() and float(param.grad.abs().sum()) > 0.0
        for name, param in head.named_parameters()
    )
    _assert(has_quality_grad, "quality head parameters must receive gradient from quality_loss")


def check_quality_targets() -> None:
    criterion = GCSLoss(model={"gcs_point_mode": "fixed_y", "gcs_imgsz": [544, 960], "gcs_quality": 0.3})
    y = torch.linspace(0.98, 0.25, 6)
    gt0 = torch.stack((torch.full_like(y, 0.20), y), dim=-1)
    gt1 = torch.stack((torch.full_like(y, 0.60), y), dim=-1)
    pred_points = torch.stack(
        (
            gt0,
            torch.stack((torch.full_like(y, 0.70), y), dim=-1),
            torch.stack((torch.full_like(y, 0.90), y), dim=-1),
        ),
        dim=0,
    ).unsqueeze(0)
    pred_quality_logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
    targets = criterion.build_quality_targets(
        pred_quality_logits,
        pred_points,
        [torch.stack((gt0, gt1), dim=0)],
        [torch.ones((2, 6), dtype=torch.float32)],
        [(torch.tensor([0, 1]), torch.tensor([0, 1]))],
    )
    _assert(targets.requires_grad is False, "quality targets must be detached")
    _assert(torch.isfinite(targets).all(), "quality targets must be finite")
    _assert(float(targets.min()) >= 0.0 and float(targets.max()) <= 1.0, "quality targets must be in [0, 1]")
    _assert(float(targets[0, 2]) == 0.0, "unmatched quality target must be zero")
    _assert(not torch.allclose(targets[0, :2], torch.ones_like(targets[0, :2])), "matched quality must not be all one")


def check_exist_quality_and_negative_mining() -> None:
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_quality": 0.3,
            "gcs_hard_negative_quality_thr": 0.5,
            "gcs_hard_negative_topk": 1,
            "gcs_duplicate_dist_thr_px": 25.0,
            "gcs_duplicate_iou_thr": 0.30,
        }
    )
    points = _lane_points([200, 210, 500, 800])
    gt_points = [points[:1].clone()]
    gt_valid = [torch.ones((1, points.shape[1]), dtype=torch.float32)]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    pred_logits = torch.tensor([[5.0, 5.0, 5.0, -5.0]])
    pred_valid_logits = torch.full((1, 4, points.shape[1]), 5.0)

    perfect_quality = criterion._matched_exist_quality(
        points[:1],
        gt_points[0],
        gt_valid[0],
        pred_valid_logits[0, :1],
    )
    poor_quality = criterion._matched_exist_quality(
        points[3:4],
        gt_points[0],
        gt_valid[0],
        torch.full_like(pred_valid_logits[0, :1], -5.0),
    )
    _assert(perfect_quality.requires_grad is False, "matched exist quality target must be detached")
    _assert(float(perfect_quality.min()) > 0.99, "perfect lane quality should remain near 1")
    _assert(torch.allclose(poor_quality, torch.full_like(poor_quality, 0.5)), "poor matched quality must clamp to 0.5")

    hard_mask, duplicate_mask = criterion.negative_query_masks(
        pred_logits,
        points.unsqueeze(0),
        pred_valid_logits,
        gt_points,
        gt_valid,
        indices,
    )
    _assert(bool(duplicate_mask[0, 1]), "near unmatched prediction must be a duplicate negative")
    _assert(bool(hard_mask[0, 1]), "duplicate negative must also be a hard negative")
    _assert(bool(hard_mask[0, 2]), "high exist*valid unmatched prediction must be a hard negative")
    _assert(not bool(hard_mask[0, 0]), "matched query must never be mined as a hard negative")


def check_decode_quality_rank_and_rescue() -> None:
    points = _lane_points([100, 260, 420, 580, 740], points=8)
    pred_logits = torch.full((5,), 6.0)
    valid_logits = torch.full((5, 8), 10.0)

    rank_valid_logits = valid_logits[:2].clone()
    rank_valid_logits[0, 6:] = -10.0
    ranked = decode_gcs_predictions(
        pred_points=points[:2],
        pred_logits=pred_logits[:2],
        pred_valid_logits=rank_valid_logits,
        pred_quality_logits=torch.tensor([5.0, -5.0]),
        image_shape=(720, 960),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=6,
        max_det=1,
        nms_dist_px=0.0,
        use_count_head_decode=False,
    )
    _assert(len(ranked) == 1 and int(ranked[0]["query"]) == 1, "visibility-aware rank must prefer complete lane")
    _assert(ranked[0]["rank_score_source"] == "exist_visibility", "composite rank source should be recorded")
    expected = float(pred_logits[1].sigmoid()) * float(rank_valid_logits[1].sigmoid().mean()) * (3.0 / 8.0)
    _assert(abs(float(ranked[0]["rank_score"]) - expected) < 1e-5, "rank score formula mismatch")
    _assert(float(ranked[0]["quality_head_score"]) < 0.01, "Quality Head score should remain diagnostic only")

    high_p5, meta = decode_gcs_predictions(
        pred_points=points,
        pred_logits=pred_logits,
        pred_valid_logits=valid_logits,
        pred_quality_logits=torch.full((5,), 5.0),
        pred_count_logits=torch.tensor([-5.0, -5.0, 2.0, 1.9]),
        image_shape=(720, 960),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=6,
        max_det=5,
        nms_dist_px=0.0,
        use_count_head_decode=True,
        dataset_name="culane",
        candidate_score_thr=0.0,
        candidate_point_valid_thr=0.5,
        candidate_min_points=5,
        final_min_points=6,
        fifth_min_points=5,
        quality_rescue_5th=True,
        quality_rescue_count5_thr=0.30,
        return_meta=True,
    )
    _assert(meta["count_head_policy_count"] == 4, "fixture should keep Count Head policy at K=4")
    _assert(meta["effective_policy_count"] == 5, "gated high-P5 rescue may raise effective K to 5")
    _assert(meta["quality_count5_upgrade_success"], "K=4 to K=5 upgrade should be explicitly marked")
    _assert(len(high_p5) == 5 and any(x.get("quality_rescue_5th") for x in high_p5), "fifth lane should be rescued")
    _assert({int(x["query"]) for x in high_p5}.issubset(set(range(5))), "rescue must use real model query ids")
    _assert(len(high_p5) <= int(meta["effective_policy_count"]), "final output must not exceed effective policy K")

    low_p5, low_meta = decode_gcs_predictions(
        pred_points=points,
        pred_logits=pred_logits,
        pred_valid_logits=valid_logits,
        pred_quality_logits=torch.full((5,), 5.0),
        pred_count_logits=torch.tensor([-5.0, -5.0, 2.0, -1.0]),
        image_shape=(720, 960),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=6,
        max_det=5,
        nms_dist_px=0.0,
        use_count_head_decode=True,
        dataset_name="culane",
        candidate_score_thr=0.0,
        candidate_point_valid_thr=0.5,
        candidate_min_points=5,
        final_min_points=6,
        fifth_min_points=5,
        quality_rescue_5th=True,
        quality_rescue_count5_thr=0.30,
        return_meta=True,
    )
    _assert(len(low_p5) == 4, "low P5 must not force a fifth output")
    _assert(low_meta["effective_policy_count"] == 4, "effective K should remain 4 when gated rescue is not eligible")
    _assert(
        low_meta["top5_candidate_quality_before_nms"] is not None,
        "K=4 decode must still expose real rank-5 candidate quality for GT5 diagnostics",
    )


def check_last_required_lane_rescue() -> None:
    points = _lane_points([100, 260, 420, 580])
    pred_logits = torch.full((4,), 6.0)
    valid_logits = _valid_logits(4)
    valid_logits[3, 4:] = -6.0

    lanes, meta = decode_gcs_predictions(
        pred_points=points,
        pred_logits=pred_logits,
        pred_valid_logits=valid_logits,
        pred_quality_logits=torch.full((4,), 5.0),
        pred_count_logits=torch.tensor([-5.0, -5.0, 5.0, -5.0]),
        image_shape=(720, 960),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=6,
        max_det=5,
        nms_dist_px=0.0,
        use_count_head_decode=True,
        dataset_name="culane",
        candidate_score_thr=0.0,
        candidate_point_valid_thr=0.5,
        candidate_min_points=6,
        final_min_points=6,
        fifth_min_points=5,
        last_lane_rescue=True,
        last_lane_rescue_min_policy_count=4,
        last_lane_rescue_point_valid_thr=0.5,
        last_lane_rescue_min_points=4,
        last_lane_rescue_mean_valid_thr=0.40,
        last_lane_rescue_quality_thr=0.50,
        last_lane_rescue_dist_px=24.0,
        edge_last_lane_rescue=False,
        return_meta=True,
    )
    _assert(meta["count_head_policy_count"] == 4, "fixture should keep Count Head policy at K=4")
    _assert(len(lanes) == 4, "edge-prioritized last-lane rescue should fill K=4 shortfall")
    _assert(meta["edge_last_lane_rescue_success_count"] == 1, "edge last-lane rescue success should be recorded")
    _assert(meta["last_lane_rescue_success_count"] == 0, "ordinary last-lane rescue should not run after edge success")
    _assert(any(x.get("edge_last_lane_rescue") for x in lanes), "rescued lane should be marked as edge rescue")
    _assert(not any(x.get("quality_rescue_5th") for x in lanes), "K=4 shortfall rescue is not fifth-lane upgrade")
    _assert(meta["count_head_shortfall"] == 0, "last-lane rescue should clear Count Head output shortfall")


def check_edge_rescue_shared_y_geometry() -> None:
    """A short upper edge lane must be compared with selected lanes at the same y coordinates."""
    ys = torch.linspace(710.0, 220.0, 8)

    def lane(xs: list[float], valid: list[float]) -> dict:
        return {
            "points": torch.stack((torch.tensor(xs, dtype=torch.float32), ys), dim=-1).numpy(),
            "point_valid": torch.tensor(valid, dtype=torch.float32).numpy(),
        }

    selected = [
        lane([100, 120, 140, 160, 180, 200, 220, 240], [1] * 8),
        lane([500, 490, 480, 470, 460, 450, 440, 430], [1] * 8),
    ]
    short_left_edge = lane([80, 100, 120, 140, 140, 160, 180, 200], [0, 0, 0, 0, 1, 1, 1, 1])
    side, gap, _ = _edge_side_and_gap(
        short_left_edge,
        selected,
        image_shape=(720, 960),
        outside_gap_px=28.0,
    )
    _assert(side == "left", "short edge lane should be classified from shared-y lateral ordering")
    _assert(gap is not None and gap >= 39.0, "shared-y edge gap should preserve the true outside separation")


def check_edge_last_lane_rescue_and_count_upgrade() -> None:
    points = _lane_points([100, 260, 420, 580, 740])
    pred_logits = torch.full((5,), 6.0)
    valid_logits = _valid_logits(5)
    valid_logits[0, 4:] = -6.0

    lanes, meta = decode_gcs_predictions(
        pred_points=points,
        pred_logits=pred_logits,
        pred_valid_logits=valid_logits,
        pred_quality_logits=torch.full((5,), 5.0),
        pred_count_logits=torch.tensor([-5.0, -5.0, 2.0, 1.9]),
        image_shape=(720, 960),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=6,
        max_det=5,
        nms_dist_px=0.0,
        use_count_head_decode=True,
        dataset_name="culane",
        candidate_score_thr=0.0,
        candidate_point_valid_thr=0.5,
        candidate_min_points=6,
        enable_rescue_candidate_pool=True,
        rescue_candidate_score_thr=0.0,
        rescue_candidate_point_valid_thr=0.5,
        rescue_candidate_min_points=6,
        final_min_points=6,
        fifth_min_points=6,
        edge_last_lane_rescue=False,
        edge_count4_to5_upgrade=True,
        edge_count4_to5_prob_margin=0.12,
        edge_rescue_conf_thr=0.02,
        edge_rescue_point_valid_thr=0.5,
        edge_rescue_min_points=4,
        edge_rescue_mean_valid_thr=0.35,
        edge_rescue_quality_thr=0.45,
        edge_rescue_outside_gap_px=28.0,
        edge_rescue_dist_px=24.0,
        enable_soft_count_decision=False,
        return_meta=True,
    )
    _assert(meta["count_head_raw_count"] == 4, "fixture should keep raw Count Head at K=4")
    _assert(not meta["soft_count_decision_enabled"], "fixture should keep soft-count disabled")
    _assert(meta["edge_count4_to5_upgrade"], "near P4/P5 plus outside edge candidate should upgrade K=4 to K=5")
    _assert(meta["edge_last_lane_rescue_active"], "edge upgrade should activate edge rescue for this decode")
    _assert(meta["edge_last_lane_rescue_success_count"] == 1, "edge rescue success should be recorded")
    _assert(len(lanes) == 5, "edge rescue should fill the promoted fifth output")
    _assert(any(x.get("edge_last_lane_rescue") for x in lanes), "rescued lane should be marked as edge rescue")
    _assert({int(x["query"]) for x in lanes}.issubset(set(range(5))), "edge rescue must use real query ids")
    _assert(meta["count_head_shortfall"] == 0, "edge rescue should clear effective-K shortfall")


def check_last_lane_quality_fallback() -> None:
    points = _lane_points([100, 260, 420, 580, 740])
    pred_logits = torch.full((5,), 6.0)
    valid_logits = _valid_logits(5)
    valid_logits[4, 5:] = -6.0

    lanes, meta = decode_gcs_predictions(
        pred_points=points,
        pred_logits=pred_logits,
        pred_valid_logits=valid_logits,
        pred_quality_logits=torch.full((5,), 5.0),
        pred_count_logits=torch.tensor([-5.0, -5.0, -5.0, 5.0]),
        image_shape=(720, 960),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=6,
        max_det=5,
        nms_dist_px=0.0,
        use_count_head_decode=True,
        dataset_name="culane",
        candidate_score_thr=0.0,
        candidate_point_valid_thr=0.5,
        candidate_min_points=5,
        final_min_points=6,
        fifth_min_points=6,
        quality_rescue_5th=True,
        quality_rescue_count5_thr=0.70,
        quality_rescue_min_points=5,
        last_lane_rescue=True,
        last_lane_rescue_min_policy_count=5,
        last_lane_rescue_point_valid_thr=0.5,
        last_lane_rescue_min_points=6,
        last_lane_rescue_mean_valid_thr=0.40,
        last_lane_rescue_quality_thr=0.50,
        last_lane_rescue_dist_px=24.0,
        edge_last_lane_rescue=False,
        edge_rescue_outside_gap_px=9999.0,
        return_meta=True,
    )
    _assert(meta["count_head_policy_count"] == 5, "fixture should keep Count Head policy at K=5")
    _assert(meta["last_lane_rescue_attempt_count"] == 1, "last-lane rescue should be tried first")
    _assert(meta["last_lane_rescue_success_count"] == 0, "last-lane rescue should fail in this fixture")
    _assert(meta["quality_rescue_fallback_after_last_lane"], "quality fallback should be explicitly marked")
    _assert(meta["quality_rescue_success_count"] == 1, "quality rescue fallback should recover the fifth lane")
    _assert(len(lanes) == 5 and any(x.get("quality_rescue_5th") for x in lanes), "fifth lane should be quality-rescued")
    _assert(meta["count_head_shortfall"] == 0, "quality fallback should clear K=5 shortfall")


def check_nms_suppressed_contract() -> None:
    points = _lane_points([100, 105, 300])
    kept, suppressed = lane_nms(
        points,
        torch.tensor([0.9, 0.8, 0.7]),
        image_shape=(720, 960),
        dist_thr_px=18.0,
        valid_masks=torch.ones((3, 6), dtype=torch.bool),
        point_valid_scores=torch.full((3, 6), 0.75),
        exist_scores=torch.tensor([0.95, 0.77, 0.66]),
        min_overlap=6,
        return_suppressed=True,
    )
    _assert(kept.tolist() == [0, 2], f"unexpected NMS keep list: {kept.tolist()}")
    required = {
        "candidate_id",
        "query_id",
        "rank_score",
        "lane_conf",
        "mean_point_valid",
        "valid_points",
        "suppressed_by",
        "distance_to_suppressor",
    }
    _assert(bool(suppressed) and required.issubset(suppressed[0]), f"suppressed fields missing: {suppressed}")


def check_official_sweep_summary_contract() -> None:
    state = official_sweep.empty_state()
    state["images"] = 1
    state["accuracy_sum"] = 1.0
    state["gt_lanes_hist"][5] = 1
    state["pred_lanes_hist"][5] = 1
    state["gt_pred_lanes_hist"][(5, 5)] = 1
    state["all_pred_quality_sum"] = 4.0
    state["all_pred_quality_count"] = 5
    state["matched_pred_quality_sum"] = 0.9
    state["matched_pred_quality_count"] = 1
    state["unmatched_pred_quality_sum"] = 0.2
    state["unmatched_pred_quality_count"] = 1
    state["rank5_quality_gt5_k4_sum"] = 0.7
    state["rank5_quality_gt5_k4_count"] = 1
    state["rescue_attempt_count"] = 1
    state["rescue_success_count"] = 1
    state["rescue_tp_count"] = 1
    state["last_lane_rescue_success_count"] = 1
    state["last_lane_rescue_tp_count"] = 2
    state["last_lane_rescue_fp_count"] = 1
    combo = {
        "conf": 0.05,
        "point_valid_thr": 0.20,
        "nms_dist_px": 18.0,
        "max_det": 5,
        "min_points": 6,
        "rank_min_points_tag": "none",
        "rank_min_points": None,
    }
    row = official_sweep.summarize_state(combo, state)
    required = {
        "official_acc",
        "candidate_pool_shortfall_rate",
        "top5_suppressed_by_nms_rate",
        "all_pred_quality_mean",
        "matched_pred_quality_mean",
        "unmatched_pred_quality_mean",
        "TP_quality_mean",
        "FP_quality_mean",
        "rank5_quality_mean_on_gt5_k4",
        "rescue_attempt_count",
        "rescue_success_count",
        "rescue_tp_count",
        "rescue_fp_count",
        "rescue_precision",
        "last_lane_rescue_precision",
    }
    _assert(required.issubset(row), f"official sweep row missing fields: {required - set(row)}")
    _assert(0.0 <= row["last_lane_rescue_precision"] <= 1.0, "last-lane precision must be bounded")
    _assert(row["last_lane_rescue_precision"] == 0.666667, "last-lane precision should use TP/(TP+FP)")
    sweep_text = (ROOT / "tools" / "sweep_tusimple_official.py").read_text(encoding="utf-8")
    _assert('default="official_acc"' in sweep_text, "official sweep best selection must default to official_acc")


def check_official_best_selector_priority() -> None:
    common = {
        "count_acc_3": 0.0,
        "count_acc_4": 0.0,
        "count_acc_5": 0.0,
        "gt5_count_head_under_rate": 0.0,
        "gt5_valid_points_fail_rate": 0.0,
        "gt5_candidate_pool_shortfall_rate": 0.0,
        "gt5_top5_suppressed_by_nms_rate": 0.0,
        "rate_3_to_4": 0.0,
        "rate_3_to_5": 0.0,
        "rate_4_to_3": 0.0,
        "rate_4_to_5": 0.0,
        "rate_5_to_4": 0.0,
    }
    high_acc = {
        **common,
        "official_acc": 0.95193,
        "official_score": 0.95133,
        "official_fp": 0.03,
        "official_fn": 0.02,
        "gt5_output5_rate": 0.743243,
    }
    high_gt5 = {
        **common,
        "official_acc": 0.951353,
        "official_score": 0.999999,
        "official_fp": 0.05,
        "official_fn": 0.00,
        "gt5_output5_rate": 0.810811,
    }
    args = SimpleNamespace(
        select_best_metric="official_score",
        baseline_fp=None,
        baseline_fn=None,
    )
    selected = official_sweep.select_best([high_gt5, high_acc], args)
    _assert(selected["official_acc"] == high_acc["official_acc"], "requested metrics must not override official_acc")

    guarded_args = SimpleNamespace(
        select_best_metric="official_acc",
        baseline_fp=None,
        baseline_fn=None,
        min_gt5_output5_rate=0.80,
    )
    selected = official_sweep.select_best([high_gt5, high_acc], guarded_args)
    _assert(
        selected["official_acc"] == high_acc["official_acc"],
        "soft GT5 thresholds must not reject the higher-accuracy candidate",
    )
    _assert(not selected["selection_constraints_satisfied"], "violated soft threshold should remain visible in diagnostics")
    _assert(selected["selection_constraints_mode"] == "diagnostic_only", "selection thresholds must be diagnostic only")

    baseline_args = SimpleNamespace(
        select_best_metric="official_acc",
        baseline_fp=0.01,
        baseline_fn=0.01,
        fp_tol=0.0,
        fn_tol=0.0,
    )
    selected = official_sweep.select_best([high_gt5, high_acc], baseline_args)
    _assert(selected["official_acc"] == high_acc["official_acc"], "baseline FP/FN thresholds must not filter ACC best")
    _assert(not selected["selection_constraints_satisfied"], "baseline threshold violations should remain diagnostic")

    low_fn = {**common, "official_acc": 0.95, "official_score": 0.949, "official_fp": 0.05, "official_fn": 0.02, "gt5_output5_rate": 0.70}
    low_fp = {**common, "official_acc": 0.95, "official_score": 0.949, "official_fp": 0.02, "official_fn": 0.04, "gt5_output5_rate": 0.90}
    selected = official_sweep.select_best([low_fp, low_fn], args)
    _assert(selected["official_fn"] == low_fp["official_fn"], "equal ACC must not use FN as a tie-breaker")

    low_fp = {**common, "official_acc": 0.95, "official_score": 0.949, "official_fp": 0.02, "official_fn": 0.03, "gt5_output5_rate": 0.70}
    high_fp = {**common, "official_acc": 0.95, "official_score": 0.949, "official_fp": 0.04, "official_fn": 0.03, "gt5_output5_rate": 0.90}
    selected = official_sweep.select_best([high_fp, low_fp], args)
    _assert(selected["official_fp"] == high_fp["official_fp"], "equal ACC must not use FP as a tie-breaker")


def check_official_best_cross_epoch_priority() -> None:
    previous = {
        "official_acc": 0.9500,
        "official_score": 0.9490,
        "official_fp": 0.02,
        "official_fn": 0.03,
        "gt5_output5_rate": 0.80,
        "rate_4_to_5": 0.06,
        "rate_3_to_5": 0.005,
        "count_acc_5": 0.80,
        "count_acc_4": 0.86,
    }
    clear_acc_gain = {**previous, "official_acc": 0.950001, "official_score": 0.9480, "official_fn": 0.05}
    _assert(
        GCSLaneTrainer._official_best_candidate_is_better(clear_acc_gain, previous),
        "official_best should update on any strict official_acc gain",
    )
    near_acc_better_tie = {**previous, "official_acc": 0.9497, "official_score": 0.9492, "official_fn": 0.02}
    _assert(
        not GCSLaneTrainer._official_best_candidate_is_better(near_acc_better_tie, previous),
        "official_best must reject lower official_acc even when diagnostics improve",
    )
    equal_acc_better_diagnostics = {**previous, "official_score": 0.9492, "official_fn": 0.02}
    _assert(
        not GCSLaneTrainer._official_best_candidate_is_better(equal_acc_better_diagnostics, previous),
        "official_best must keep the existing checkpoint when official_acc is equal",
    )
    trainer_text = (ROOT / "ultralytics" / "models" / "yolo" / "gcs_lane" / "train.py").read_text(encoding="utf-8")
    _assert("official_non_best" not in trainer_text, "training must not save official_non_best checkpoints")


def main() -> None:
    checks = (
        check_loss_items,
        check_count_sum_loss_gradient,
        check_quality_head_shape_and_gradient,
        check_quality_targets,
        check_exist_quality_and_negative_mining,
        check_decode_quality_rank_and_rescue,
        check_last_required_lane_rescue,
        check_edge_rescue_shared_y_geometry,
        check_edge_last_lane_rescue_and_count_upgrade,
        check_last_lane_quality_fallback,
        check_nms_suppressed_contract,
        check_official_sweep_summary_contract,
        check_official_best_selector_priority,
        check_official_best_cross_epoch_priority,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("All GCS algorithm contract checks passed.")


if __name__ == "__main__":
    main()
