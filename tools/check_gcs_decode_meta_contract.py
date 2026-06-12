# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Contract checks for GCS decode metadata, rescue pools, and NMS suppression diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.gcs_postprocess import decode_gcs_predictions, lane_nms  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _lane_points(xs: list[float], points: int = 6, width: int = 960) -> torch.Tensor:
    y = torch.linspace(0.98, 0.25, points)
    lanes = []
    for x in xs:
        lanes.append(torch.stack((torch.full_like(y, float(x) / float(width)), y), dim=-1))
    return torch.stack(lanes, dim=0)


def _valid_logits(count: int, points: int = 6) -> torch.Tensor:
    return torch.full((count, points), 10.0)


def check_lane_nms_return_suppressed() -> None:
    points = _lane_points([100, 105, 300])
    scores = torch.tensor([0.9, 0.8, 0.7])
    valid = torch.ones((3, 6), dtype=torch.bool)
    point_valid_scores = torch.tensor(
        [
            [0.90, 0.90, 0.90, 0.90, 0.90, 0.90],
            [0.70, 0.70, 0.70, 0.70, 0.70, 0.70],
            [0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        ],
        dtype=torch.float32,
    )
    exist_scores = torch.tensor([0.95, 0.77, 0.66])
    kept, suppressed = lane_nms(
        points,
        scores,
        image_shape=(720, 960),
        dist_thr_px=18.0,
        valid_masks=valid,
        point_valid_scores=point_valid_scores,
        exist_scores=exist_scores,
        min_overlap=6,
        return_suppressed=True,
    )
    _assert(kept.tolist() == [0, 2], f"unexpected NMS keep list: {kept.tolist()}")
    _assert(len(suppressed) == 1 and suppressed[0]["index"] == 1, f"unexpected suppressed list: {suppressed}")
    _assert(suppressed[0]["suppressed_by_index"] == 0, "suppressed record should include suppressor index")
    _assert(abs(float(suppressed[0]["exist_score"]) - 0.77) < 1e-6, "suppressed record should include exist score")
    _assert(abs(float(suppressed[0]["point_valid_mean"]) - 0.70) < 1e-6, "suppressed record should include point-valid mean")


def check_rescue_pool_meta_and_no_fabrication() -> None:
    points = _lane_points([100, 250, 400, 550, 700])
    logits = torch.tensor([8.0, 7.0, 6.0, 5.0, -3.5])
    lanes, meta = decode_gcs_predictions(
        pred_points=points,
        pred_logits=logits,
        pred_valid_logits=_valid_logits(5),
        pred_count_logits=torch.tensor([-5.0, -5.0, -5.0, 5.0]),
        image_shape=(720, 960),
        score_thr=0.5,
        point_valid_thr=0.5,
        min_points=6,
        max_det=5,
        nms_dist_px=0.0,
        use_count_head_decode=True,
        dataset_name="culane",
        candidate_score_thr=0.5,
        candidate_point_valid_thr=0.5,
        candidate_min_points=6,
        enable_rescue_candidate_pool=True,
        rescue_candidate_score_thr=0.01,
        rescue_candidate_point_valid_thr=0.5,
        rescue_candidate_min_points=6,
        final_min_points=6,
        fifth_min_points=5,
        return_meta=True,
    )
    _assert(meta["candidate_count_normal"] == 4, f"normal pool should have 4 candidates: {meta}")
    _assert(meta["candidate_count_after_rescue"] == 5, f"rescue pool should fill K=5: {meta}")
    _assert(meta["candidate_pool_shortfall_before_rescue"] == 1, "before-rescue shortfall should be 1")
    _assert(meta["candidate_pool_shortfall"] == 0, "after-rescue shortfall should be 0")
    _assert(len(lanes) == 5, f"Count Head K=5 should control final output, got {len(lanes)}")
    _assert({int(x["query"]) for x in lanes}.issubset(set(range(5))), "rescue must not fabricate query ids")
    _assert(any(x.get("candidate_pool_rescue") for x in lanes), "one final lane should be marked as pool rescue")


def check_top5_nms_suppression_meta() -> None:
    points = _lane_points([100, 300, 430, 560, 105])
    logits = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0])
    lanes, meta = decode_gcs_predictions(
        pred_points=points,
        pred_logits=logits,
        pred_valid_logits=_valid_logits(5),
        pred_count_logits=torch.tensor([-5.0, -5.0, -5.0, 5.0]),
        image_shape=(720, 960),
        score_thr=0.0,
        point_valid_thr=0.5,
        min_points=6,
        max_det=5,
        nms_dist_px=18.0,
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
        fifth_min_points=5,
        line_nms_rescue_dist_px=0.0,
        return_meta=True,
    )
    _assert("candidate_pool_shortfall" in meta, "decode meta should expose candidate_pool_shortfall")
    _assert("top5_suppressed_by_nms" in meta, "decode meta should expose top5_suppressed_by_nms")
    _assert(meta["top5_candidate_exists_before_nms"], f"top5 candidate should exist before NMS: {meta}")
    _assert(meta["top5_suppressed_by_nms"], f"5th pre-NMS candidate should be marked NMS-suppressed: {meta}")
    _assert(meta["nms_suppressed_count"] == 1, f"expected one NMS-suppressed candidate: {meta}")
    _assert(len(lanes) == 5, f"suppressed recovery should still respect Count Head K=5, got {len(lanes)}")


def check_short_visible_segment_rank_metadata() -> None:
    points = _lane_points([100, 250, 400, 550, 700], points=32)
    valid_logits = torch.full((5, 32), 10.0)
    valid_logits[4, 5:] = -10.0
    lanes = decode_gcs_predictions(
        pred_points=points,
        pred_logits=torch.full((5,), 6.0),
        pred_valid_logits=valid_logits,
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
        fifth_min_points=5,
    )
    short_lane = next((lane for lane in lanes if int(lane["query"]) == 4), None)
    _assert(short_lane is not None, "short visible edge query should remain selectable as fifth lane")
    _assert(float(short_lane["mean_valid_score"]) > 0.9, "mean_valid_score should track visible segment quality")
    _assert(float(short_lane["mean_valid_score_all"]) < 0.2, "decode metadata should retain all-anchor mean")
    _assert(float(short_lane["valid_count_score"]) > float(short_lane["anchor_valid_count_score"]), "visible support should avoid all-anchor underweighting")


def main() -> None:
    checks = (
        check_lane_nms_return_suppressed,
        check_rescue_pool_meta_and_no_fabrication,
        check_top5_nms_suppression_meta,
        check_short_visible_segment_rank_metadata,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("All GCS decode metadata contract checks passed.")


if __name__ == "__main__":
    main()
