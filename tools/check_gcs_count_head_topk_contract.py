# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Contract checks for explicit Count Head + Top-K GCS lane decode."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.gcs_lane import GCSLaneHead
from ultralytics.utils.gcs_loss import GCSLoss
from ultralytics.utils.gcs_postprocess import (
    apply_count_policy,
    count_head_decode_meta,
    decode_gcs_predictions,
    empty_decode_count_state,
    lane_nms,
    summarize_decode_count_state,
    update_decode_count_state,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _valid_lanes(count: int, points: int = 8) -> torch.Tensor:
    valid = torch.zeros((count, points), dtype=torch.float32)
    valid[:, :2] = 1.0
    return valid


def _lane_points(query_count: int, points: int = 6, width: int = 960) -> torch.Tensor:
    y = torch.linspace(0.98, 0.25, points)
    xs = torch.linspace(100.0, 500.0, query_count) / float(width)
    lanes = []
    for x in xs:
        lanes.append(torch.stack((torch.full_like(y, x), y), dim=-1))
    return torch.stack(lanes, dim=0)


def _valid_logits(valid_counts: list[int], points: int = 6) -> torch.Tensor:
    logits = torch.full((len(valid_counts), points), -10.0)
    for i, n in enumerate(valid_counts):
        logits[i, : int(n)] = 10.0
    return logits


def check_head_shapes() -> None:
    torch.manual_seed(0)
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
    _assert(out["pred_points"].shape == (2, 6, 8, 2), "pred_points shape mismatch")
    _assert(out["pred_logits"].shape == (2, 6), "pred_logits shape mismatch")
    _assert(out["pred_valid_logits"].shape == (2, 6, 8), "pred_valid_logits shape mismatch")
    _assert(out["pred_count_logits"].shape == (2, 4), "pred_count_logits must be B x 4")
    _assert(out["pred_count_boundary_logits"].shape == (2, 2), "pred_count_boundary_logits must be B x 2")
    _assert(out["pred_quality_logits"].shape == (2, 6), "pred_quality_logits must be B x Q")


def check_count_targets_and_cls_loss() -> None:
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_count_min_gt_points": 1,
            "gcs_line_iou": 0.0,
            "gcs_exist_quality_lane_iou_alpha": 0.0,
            "gcs_quality": 0.0,
            "gcs_imgsz": [544, 960],
        }
    )
    pred_count_logits = torch.zeros((3, 4), dtype=torch.float32)
    gt_valid = [_valid_lanes(3), _valid_lanes(4), _valid_lanes(5)]
    gt_count, gt_cls, gt_raw = criterion.count_head_targets(pred_count_logits, gt_valid)
    _assert(gt_raw.tolist() == [3, 4, 5], f"raw count mismatch: {gt_raw.tolist()}")
    _assert(gt_count.tolist() == [3, 4, 5], f"clamped count mismatch: {gt_count.tolist()}")
    _assert(gt_cls.tolist() == [1, 2, 3], f"class target mismatch: {gt_cls.tolist()}")

    pred_count_logits_grad = torch.zeros((3, 4), dtype=torch.float32, requires_grad=True)
    pred_count_boundary_logits_grad = torch.zeros((3, 2), dtype=torch.float32, requires_grad=True)
    boundary_target = criterion.count_boundary_targets(pred_count_boundary_logits_grad, gt_valid)
    _assert(
        torch.allclose(boundary_target, torch.tensor([[0.025, 0.025], [0.975, 0.025], [0.975, 0.975]])),
        f"boundary target mismatch: {boundary_target}",
    )
    loss_cls = criterion.count_head_loss(
        {
            "pred_count_logits": pred_count_logits_grad,
            "pred_count_boundary_logits": pred_count_boundary_logits_grad,
        },
        torch.zeros((3, 1, 1, 2)),
        gt_valid,
    )
    loss_cls.backward()
    grad = pred_count_logits_grad.grad
    _assert(grad[0, 1] < 0, "GT=3 CE must push class 3-lane up")
    _assert(grad[1, 2] < 0, "GT=4 CE must push class 4-lane up")
    _assert(grad[2, 3] < 0, "GT=5 CE must push class 5-lane up")
    _assert(pred_count_boundary_logits_grad.grad is not None, "Count Boundary logits must receive gradient")

    try:
        criterion.count_head_loss(
            {"pred_count_logits": torch.zeros((3, 4))},
            torch.zeros((3, 1, 1, 2)),
            gt_valid,
        )
    except ValueError as exc:
        _assert("pred_count_boundary_logits is missing" in str(exc), "enabled Count Boundary must require logits")
    else:
        raise AssertionError("enabled Count Boundary BCE must not silently become zero when logits are missing")

    try:
        criterion.count_head_loss({}, torch.zeros((1, 1, 1, 2)), [_valid_lanes(3)])
    except ValueError as exc:
        _assert("pred_count_logits is missing" in str(exc), "enabled Count Head CE should require logits")
    else:
        raise AssertionError("enabled Count Head CE must not silently become zero when pred_count_logits is missing")

    disabled = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_count_cls": 0.0,
            "gcs_line_iou": 0.0,
            "gcs_exist_quality_lane_iou_alpha": 0.0,
            "gcs_quality": 0.0,
            "gcs_imgsz": [544, 960],
        }
    )
    zero = disabled.count_head_loss({}, torch.ones((1, 1, 1, 2), requires_grad=True), [_valid_lanes(3)])
    _assert(float(zero.detach()) == 0.0, "disabled Count Head CE may return differentiable zero")


def check_quality_targets_and_loss() -> None:
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "fixed_y",
            "gcs_imgsz": [544, 960],
            "gcs_quality": 0.3,
            "gcs_quality_dist_thr_px": 20.0,
            "gcs_quality_neg_weight": 0.25,
        }
    )
    y = torch.linspace(0.98, 0.25, 6)
    gt0 = torch.stack((torch.full_like(y, 0.20), y), dim=-1)
    gt1 = torch.stack((torch.full_like(y, 0.60), y), dim=-1)
    pred0 = gt0.clone()
    pred1 = torch.stack((torch.full_like(y, 0.70), y), dim=-1)
    pred2 = torch.stack((torch.full_like(y, 0.85), y), dim=-1)
    pred_points = torch.stack((pred0, pred1, pred2), dim=0).unsqueeze(0)
    gt_points = [torch.stack((gt0, gt1), dim=0)]
    gt_valid = [torch.ones((2, 6), dtype=torch.float32)]
    pred_quality_logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    target_quality = criterion.build_quality_targets(pred_quality_logits, pred_points, gt_points, gt_valid, indices)
    _assert(target_quality.requires_grad is False, "quality targets must be detached")
    _assert(torch.isfinite(target_quality).all(), "quality targets must be finite")
    _assert(float(target_quality.min()) >= 0.0 and float(target_quality.max()) <= 1.0, "quality target range")
    _assert(float(target_quality[0, 2]) == 0.0, "unmatched quality target must be zero")
    _assert(
        not torch.allclose(target_quality[0, :2], torch.ones_like(target_quality[0, :2])),
        "matched quality targets must not collapse to all ones",
    )

    loss = criterion.quality_loss(pred_quality_logits, pred_points, gt_points, gt_valid, indices)
    _assert(torch.isfinite(loss) and float(loss.detach()) >= 0.0, "quality loss must be finite")
    loss.backward()
    _assert(pred_quality_logits.grad is not None, "quality logits must receive gradient")
    _assert(torch.isfinite(pred_quality_logits.grad).all(), "quality gradient must be finite")


def check_count_policy() -> None:
    _assert(apply_count_policy(2, dataset_name="tusimple") == 3, "TuSimple K=2 must merge to 3")
    _assert(apply_count_policy(3, dataset_name="tusimple") == 3, "TuSimple K=3 mismatch")
    _assert(apply_count_policy(4, dataset_name="tusimple") == 4, "TuSimple K=4 mismatch")
    _assert(apply_count_policy(5, dataset_name="tusimple") == 5, "TuSimple K=5 mismatch")
    _assert(apply_count_policy(2, dataset_name="culane") == 2, "non-TuSimple K=2 must stay 2")


def check_count_head_meta_and_shortfall_logs() -> None:
    meta = count_head_decode_meta(
        torch.tensor([-5.0, -5.0, -5.0, 5.0]),
        use_count_head_decode=True,
        dataset_name="tusimple",
        max_det=5,
    )
    _assert(meta["count_head_policy_count"] == 5, f"policy K mismatch: {meta}")
    _assert("count_head_margin" in meta, "Count Head margin should be exposed as diagnostic metadata")

    state = empty_decode_count_state()
    update_decode_count_state(state, meta, final_pred_lanes=4)
    summary = summarize_decode_count_state(state, prefix="decode/")
    _assert(summary["decode/count_head_k"] == 5.0, "decode/count_head_k should average Count Head policy K")
    _assert(summary["decode/final_pred_lanes"] == 4.0, "decode/final_pred_lanes should average final output count")
    _assert(summary["decode/count_shortfall_rate"] == 1.0, "shortfall rate should flag K > final output")
    _assert(summary["decode/k5_to_output4_rate"] == 1.0, "K=5 to output=4 rate should be tracked")

    try:
        count_head_decode_meta(None, use_count_head_decode=True)
    except ValueError as exc:
        _assert("pred_count_logits is missing" in str(exc), "missing Count Head logits should fail loudly")
    else:
        raise AssertionError("Count Head decode must not fall back to max_det when logits are missing")


def check_count_boundary_calibration() -> None:
    logits_34 = torch.tensor([-5.0, 2.0, 1.9, -5.0])
    raw = count_head_decode_meta(logits_34, dataset_name="culane")
    zero = count_head_decode_meta(logits_34, torch.zeros(2), dataset_name="culane")
    _assert(zero["count_head_calibrated_count"] == raw["count_head_raw_count"], "zero boundary must be neutral")
    _assert(zero["count_head_prob"] == raw["count_head_prob"], "zero boundary must preserve Count Head probs")
    _assert(zero["count_boundary_prob"] == [0.5, 0.5], "zero boundary sigmoid should be [0.5, 0.5]")
    _assert(not zero["count_boundary_applied"], "zero boundary must not mark calibration applied")

    up34 = count_head_decode_meta(logits_34, torch.tensor([3.0, -3.0]), dataset_name="culane")
    _assert(up34["count_head_raw_count"] == 3 and up34["count_head_calibrated_count"] == 4, "boundary should calibrate 3->4")

    logits_45 = torch.tensor([-5.0, -5.0, 2.0, 1.9])
    up45 = count_head_decode_meta(logits_45, torch.tensor([3.0, 3.0]), dataset_name="culane")
    _assert(up45["count_head_raw_count"] == 4 and up45["count_head_calibrated_count"] == 5, "boundary should calibrate 4->5")

    no_jump = count_head_decode_meta(
        torch.tensor([-5.0, 2.0, 1.9, 1.8]),
        torch.tensor([8.0, 8.0]),
        dataset_name="culane",
    )
    _assert(no_jump["count_head_raw_count"] == 3, "fixture raw count mismatch")
    _assert(no_jump["count_head_calibrated_count"] in {3, 4}, "boundary must stay raw or move to adjacent class")
    _assert(no_jump["count_head_prob"][3] == 0.0, "non-adjacent class must be masked out")


def check_line_nms_order() -> None:
    width = 960
    points = _lane_points(3, points=6, width=width)
    points[0, :, 0] = 100.0 / width
    points[1, :, 0] = 105.0 / width
    points[2, :, 0] = 300.0 / width
    scores = torch.tensor([0.9, 0.8, 0.7])
    valid_masks = torch.ones((3, 6), dtype=torch.bool)
    keep = lane_nms(points, scores, image_shape=(720, width), dist_thr_px=18.0, valid_masks=valid_masks, min_overlap=6)
    _assert(keep.tolist() == [0, 2], f"Line-NMS should keep query0 and query2 before Top-K, got {keep.tolist()}")


def check_final_min_points() -> None:
    points = _lane_points(5)
    pred_logits = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0])
    valid_logits = _valid_logits([6, 6, 6, 5, 6])
    common = dict(
        pred_points=points,
        pred_logits=pred_logits,
        pred_valid_logits=valid_logits,
        image_shape=(720, 960),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=6,
        nms_dist_px=0.0,
        use_count_head_decode=True,
        dataset_name="culane",
        candidate_score_thr=0.0,
        candidate_point_valid_thr=0.5,
        candidate_min_points=5,
        final_min_points=6,
        fifth_min_points=5,
    )

    k4_logits = torch.tensor([-5.0, -5.0, 5.0, -5.0])
    lanes_k4 = decode_gcs_predictions(pred_count_logits=k4_logits, **common)
    selected_k4 = {lane["query"] for lane in lanes_k4}
    _assert(len(lanes_k4) == 4, f"K=4 should select 4 lanes, got {len(lanes_k4)}")
    _assert(3 not in selected_k4 and 4 in selected_k4, "5-point candidate cannot fill selected rank 4")

    k5_logits = torch.tensor([-5.0, -5.0, -5.0, 5.0])
    valid_logits_k5 = _valid_logits([6, 6, 6, 6, 5])
    lanes_k5 = decode_gcs_predictions(pred_count_logits=k5_logits, pred_valid_logits=valid_logits_k5, **{k: v for k, v in common.items() if k != "pred_valid_logits"})
    selected_k5 = {lane["query"] for lane in lanes_k5}
    _assert(len(lanes_k5) == 5, f"K=5 should select 5 lanes, got {len(lanes_k5)}")
    _assert(4 in selected_k5, "5-point candidate may fill selected rank 5")


def check_quality_rank_and_rescue() -> None:
    points = _lane_points(5, points=8)
    pred_logits = torch.full((5,), 6.0)
    valid_logits = _valid_logits([8, 8, 8, 8, 8], points=8)

    rank_valid_logits = _valid_logits([6, 8], points=8)
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
        candidate_score_thr=0.0,
        candidate_point_valid_thr=0.5,
        candidate_min_points=6,
    )
    _assert(len(ranked) == 1 and ranked[0]["query"] == 1, "composite rank should prefer the complete lane")
    _assert(ranked[0]["rank_score_source"] == "exist_visibility", "composite rank source should be recorded")
    _assert(float(ranked[0]["quality_head_score"]) < 0.01, "Quality Head must not override composite ranking")

    common = dict(
        pred_points=points,
        pred_logits=pred_logits,
        pred_valid_logits=valid_logits,
        pred_quality_logits=torch.full((5,), 5.0),
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
        quality_rescue_count5_thr=0.70,
        quality_rescue_conf_thr=0.03,
        quality_rescue_mean_valid_thr=0.45,
        quality_rescue_quality_thr=0.55,
        quality_rescue_min_points=5,
        quality_rescue_dist_px=24.0,
    )
    low_p5 = decode_gcs_predictions(pred_count_logits=torch.tensor([0.0, 0.0, 0.1, 0.2]), **common)
    _assert(len(low_p5) == 4, f"low P(count=5) should block fifth rescue, got {len(low_p5)}")
    high_p5 = decode_gcs_predictions(pred_count_logits=torch.tensor([-5.0, -5.0, -5.0, 5.0]), **common)
    _assert(len(high_p5) == 5, f"high-quality fifth rescue should output 5 lanes, got {len(high_p5)}")
    _assert(any(lane.get("quality_rescue_5th") for lane in high_p5), "rescued fifth lane should be marked")


def check_pre_nms_rescue() -> None:
    width = 960
    points = _lane_points(2, width=width)
    points[0, :, 0] = 100.0 / width
    points[1, :, 0] = 105.0 / width
    lanes = decode_gcs_predictions(
        pred_points=points,
        pred_logits=torch.tensor([8.0, 7.0]),
        pred_valid_logits=_valid_logits([6, 6]),
        pred_count_logits=torch.tensor([8.0, -5.0, -5.0, -5.0]),
        image_shape=(720, width),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=6,
        nms_dist_px=18.0,
        use_count_head_decode=True,
        dataset_name="culane",
        candidate_score_thr=0.0,
        candidate_point_valid_thr=0.5,
        candidate_min_points=6,
        final_min_points=6,
        fifth_min_points=5,
        line_nms_rescue_dist_px=0.0,
    )
    _assert(len(lanes) == 2, f"pre-NMS rescue should fill K=2 shortfall, got {len(lanes)}")
    _assert(any(lane.get("count_head_rescue") for lane in lanes), "rescue lane should be marked")


def check_stale_rule_calibration_is_not_active() -> None:
    points = _lane_points(2)
    try:
        decode_gcs_predictions(
            pred_points=points,
            pred_logits=torch.tensor([8.0, 7.0]),
            pred_valid_logits=_valid_logits([6, 6]),
            image_shape=(720, 960),
            count_calibration={"mode": "rule"},
        )
    except ValueError as exc:
        _assert("removed" in str(exc), "stale rule calibration should fail loudly")
    else:
        raise AssertionError("stale rule calibration must not silently decide lane count")


def check_no_score_contamination() -> None:
    pattern = re.compile(
        r"(count_prob.*rank|rank.*count_prob|pred_count.*rank|rank_score.*count|count.*rank_score)"
    )
    hits: list[str] = []
    for rel in ("ultralytics", "tools"):
        for path in (ROOT / rel).rglob("*.py"):
            if path.name.startswith("check_"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    normalized = line.strip()
                    if "Count Head K; query scores only decide" in normalized:
                        continue
                    hits.append(f"{path.relative_to(ROOT)}:{lineno}: {normalized}")
    _assert(not hits, "Count Head must not contaminate rank_score:\n" + "\n".join(hits[:20]))


def main() -> None:
    checks = (
        check_head_shapes,
        check_count_targets_and_cls_loss,
        check_quality_targets_and_loss,
        check_count_policy,
        check_count_head_meta_and_shortfall_logs,
        check_count_boundary_calibration,
        check_line_nms_order,
        check_final_min_points,
        check_quality_rank_and_rescue,
        check_pre_nms_rescue,
        check_stale_rule_calibration_is_not_active,
        check_no_score_contamination,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("All Count Head + Top-K decode contract checks passed.")


if __name__ == "__main__":
    main()
