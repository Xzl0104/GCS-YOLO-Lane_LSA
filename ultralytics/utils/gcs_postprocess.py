# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Post-processing helpers for GCS-YOLO-Lane structured lane predictions."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.utils.gcs_full_lane import (
    full_lane_interval_mask,
    full_lane_proposal_score_probability,
)

__all__ = (
    "decode_gcs_predictions",
    "draw_gcs_lanes",
    "lane_x_distance_px",
    "lane_mean_distance_px",
    "lane_nms",
    "save_gcs_lanes_txt",
    "sort_lane_bottom_to_top",
)


GCS_LANE_COLORS = (
    (0, 255, 0),
    (0, 200, 255),
    (255, 120, 0),
    (255, 0, 180),
    (80, 180, 255),
    (180, 255, 80),
    (255, 80, 80),
    (180, 120, 255),
)


def sort_lane_bottom_to_top(points: torch.Tensor) -> torch.Tensor:
    """Sort one K x 2 lane by descending y, matching the GCS bottom-to-top point order."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"Expected one lane with shape K x 2, got {tuple(points.shape)}.")
    order = torch.argsort(points[:, 1], descending=True, stable=True)
    return points[order]


def longest_contiguous_valid_mask(mask: torch.Tensor, min_points: int = 2) -> torch.Tensor:
    """Keep only the longest continuous visible point run in bottom-to-top point order."""
    mask = mask.detach().bool().cpu()
    if mask.ndim != 1:
        raise ValueError(f"point visibility mask must be 1D, got {tuple(mask.shape)}.")
    best_start = best_len = 0
    start = None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            length = i - start
            if length > best_len:
                best_start, best_len = start, length
            start = None
    out = torch.zeros_like(mask, dtype=torch.bool)
    if best_len >= int(min_points):
        out[best_start : best_start + best_len] = True
    return out


def lane_x_distance_px(
    a: torch.Tensor,
    b: torch.Tensor,
    image_shape: tuple[int, int],
    min_overlap: int = 2,
    valid_a: torch.Tensor | None = None,
    valid_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mean absolute x-distance in pixels between two normalized K x 2 lanes.

    GCS lanes use a fixed point order, so Lane-NMS compares points at the same
    sample index and suppresses duplicates by lateral distance. For fixed-y
    lanes, optional visibility masks restrict the comparison to the common
    visible anchor run so invalid endpoints do not decide duplicate suppression.
    """
    if a.shape != b.shape or a.ndim != 2 or a.shape[-1] != 2:
        raise ValueError(f"Lane NMS expects matching K x 2 lanes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if valid_a is not None or valid_b is not None:
        if valid_a is None or valid_b is None:
            raise ValueError("Lane NMS visibility comparison requires both valid_a and valid_b.")
        valid_a = valid_a.detach().bool().to(device=a.device)
        valid_b = valid_b.detach().bool().to(device=a.device)
        if valid_a.shape != a.shape[:1] or valid_b.shape != b.shape[:1]:
            raise ValueError(f"Lane NMS valid masks must have shape K, got {tuple(valid_a.shape)} and {tuple(valid_b.shape)}.")
        keep = valid_a & valid_b
        if int(keep.sum()) < int(min_overlap):
            return a.new_tensor(float("inf"))
        a = a[keep]
        b = b[keep]
    elif a.shape[0] < int(min_overlap):
        return a.new_tensor(float("inf"))
    w = int(image_shape[1])
    return torch.mean(torch.abs(a[:, 0] - b[:, 0]) * float(w))


def lane_mean_distance_px(a: torch.Tensor, b: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    """Backward-compatible alias for the Lane-NMS lateral distance."""
    return lane_x_distance_px(a, b, image_shape=image_shape)


def lane_nms(
    points: torch.Tensor,
    scores: torch.Tensor,
    image_shape: tuple[int, int],
    dist_thr_px: float,
    valid_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedily suppress duplicate lane predictions by mean x-distance."""
    if dist_thr_px <= 0.0 or points.shape[0] <= 1:
        return torch.arange(points.shape[0], dtype=torch.long, device=points.device)
    if valid_masks is not None and valid_masks.shape != points.shape[:2]:
        raise ValueError(f"Lane-NMS valid_masks must have shape N x K, got {tuple(valid_masks.shape)} vs {tuple(points.shape[:2])}.")
    order = torch.argsort(scores, descending=True)
    keep: list[int] = []
    for idx in order.tolist():
        lane = points[idx]
        duplicate = False
        for kept_idx in keep:
            valid_a = valid_masks[idx] if valid_masks is not None else None
            valid_b = valid_masks[kept_idx] if valid_masks is not None else None
            if float(lane_x_distance_px(lane, points[kept_idx], image_shape, valid_a=valid_a, valid_b=valid_b)) <= float(dist_thr_px):
                duplicate = True
                break
        if not duplicate:
            keep.append(int(idx))
    return torch.tensor(keep, dtype=torch.long, device=points.device)


def _validate_count_aware_topk(min_k: int, max_k: int, length_norm: float, extra_margin: int = 0) -> tuple[int, int, float, int]:
    """Validate count-aware top-k controls and return normalized values."""
    min_k = int(min_k)
    max_k = int(max_k)
    length_norm = float(length_norm)
    extra_margin = int(extra_margin)
    if min_k < 0 or max_k < 0:
        raise ValueError(f"count-aware k bounds must be >= 0, got min_k={min_k}, max_k={max_k}.")
    if min_k > max_k:
        raise ValueError(f"count-aware min_k must be <= max_k, got min_k={min_k}, max_k={max_k}.")
    if length_norm <= 0.0:
        raise ValueError(f"count-aware length_norm must be > 0, got {length_norm}.")
    if extra_margin < 0:
        raise ValueError(f"count-aware extra_margin must be >= 0, got {extra_margin}.")
    return min_k, max_k, length_norm, extra_margin


def _count_aware_k_hat(count_score: float, min_k: int, max_k: int) -> int:
    """Round the count score and clamp it to the configured lane-count range."""
    k_hat = int(round(float(count_score)))
    return max(int(min_k), min(int(max_k), k_hat))


def _count_aware_lane_quality(lane: dict, length_norm: float) -> float:
    """Return count-aware lane quality from existence, visible-valid probability, and visible length."""
    exist_prob = float(lane["score"])
    valid_mask = None
    if "point_valid" in lane:
        valid_mask = np.asarray(lane["point_valid"], dtype=np.float32) > 0.5
        visible_count = int(valid_mask.sum())
    elif "visible_points_norm" in lane:
        visible_count = int(np.asarray(lane["visible_points_norm"], dtype=np.float32).shape[0])
    else:
        visible_count = int(np.asarray(lane["points_norm"], dtype=np.float32).shape[0])

    mean_valid_prob = 1.0
    if "point_valid_scores" in lane and valid_mask is not None:
        valid_scores = np.asarray(lane["point_valid_scores"], dtype=np.float32)
        if valid_scores.shape[0] == valid_mask.shape[0] and int(valid_mask.sum()) > 0:
            mean_valid_prob = float(valid_scores[valid_mask].mean())
        else:
            mean_valid_prob = 0.0

    length_factor = min(float(visible_count) / float(length_norm), 1.0)
    return float(exist_prob * mean_valid_prob * length_factor)


def _prob_threshold_to_logit_floor(probability: float) -> float:
    """Convert a probability gate to a finite logit threshold."""
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability threshold must be in [0, 1], got {probability}.")
    if probability <= 0.0:
        return float("-inf")
    if probability >= 1.0:
        return float("inf")
    return float(math.log(probability / (1.0 - probability)))


def _apply_count_aware_topk(lanes: list[dict], k_hat: int, length_norm: float) -> list[dict]:
    """Keep the highest-quality lanes after ordinary conf filtering and Lane-NMS."""
    if not lanes:
        return lanes
    qualities = [_count_aware_lane_quality(lane, length_norm=length_norm) for lane in lanes]
    for lane, quality in zip(lanes, qualities):
        lane["count_aware_quality"] = float(quality)
    if len(lanes) <= int(k_hat):
        return lanes
    keep_by_quality = sorted(range(len(lanes)), key=lambda i: (-qualities[i], i))[: int(k_hat)]
    keep = set(keep_by_quality)
    return [lane for i, lane in enumerate(lanes) if i in keep]


def _decode_full_lane_proposals(
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None,
    pred_full_lane_points: torch.Tensor,
    pred_full_lane_valid_logits: torch.Tensor,
    pred_full_lane_exist_logits: torch.Tensor,
    pred_full_lane_quality_logits: torch.Tensor,
    pred_full_lane_start_logits: torch.Tensor,
    pred_full_lane_end_logits: torch.Tensor,
    image_shape: tuple[int, int] | None,
    score_thr: float,
    point_valid_thr: float,
    min_points: int,
    max_det: int | None,
    nms_dist_px: float,
) -> list[dict]:
    """Decode base and independent full-lane proposals as one prediction set."""
    if pred_valid_logits is None:
        raise ValueError("full_lane_decode requires base pred_valid_logits.")
    if pred_points.ndim != 3 or pred_points.shape[-1] != 2:
        raise ValueError(f"pred_points must have shape Q x K x 2, got {tuple(pred_points.shape)}.")
    if pred_full_lane_points.ndim != 3 or pred_full_lane_points.shape[-1] != 2:
        raise ValueError(
            "pred_full_lane_points must have shape P x K x 2, "
            f"got {tuple(pred_full_lane_points.shape)}."
        )
    proposal_count, point_count = pred_full_lane_points.shape[:2]
    if pred_points.shape[1] != point_count:
        raise ValueError(
            "Base and full-lane proposals must use the same K, "
            f"got {pred_points.shape[1]} and {point_count}."
        )
    expected_pk = (proposal_count, point_count)
    if tuple(pred_full_lane_valid_logits.shape) != expected_pk:
        raise ValueError(
            "pred_full_lane_valid_logits must have shape P x K, "
            f"got {tuple(pred_full_lane_valid_logits.shape)} vs {expected_pk}."
        )
    if tuple(pred_full_lane_exist_logits.shape) != (proposal_count,):
        raise ValueError(
            "pred_full_lane_exist_logits must have shape P, "
            f"got {tuple(pred_full_lane_exist_logits.shape)}."
        )
    if tuple(pred_full_lane_quality_logits.shape) != (proposal_count,):
        raise ValueError(
            "pred_full_lane_quality_logits must have shape P, "
            f"got {tuple(pred_full_lane_quality_logits.shape)}."
        )
    if tuple(pred_full_lane_start_logits.shape) != expected_pk or tuple(pred_full_lane_end_logits.shape) != expected_pk:
        raise ValueError(
            "pred_full_lane_start_logits and pred_full_lane_end_logits must have shape P x K, "
            f"got {tuple(pred_full_lane_start_logits.shape)} and {tuple(pred_full_lane_end_logits.shape)}."
        )
    base_points = pred_points.detach().float().cpu().clamp(0.0, 1.0)
    base_valid_scores = pred_valid_logits.detach().float().cpu().sigmoid()
    base_valid_mask = base_valid_scores >= float(point_valid_thr)
    base_scores = full_lane_proposal_score_probability(
        pred_logits.detach().float().cpu(),
        quality_logits=None,
        valid_logits=pred_valid_logits.detach().float().cpu(),
    ).cpu()

    full_points = pred_full_lane_points.detach().float().cpu().clamp(0.0, 1.0)
    full_valid_scores = pred_full_lane_valid_logits.detach().float().cpu().sigmoid()
    full_start = pred_full_lane_start_logits.detach().float().cpu()
    full_end = pred_full_lane_end_logits.detach().float().cpu()
    full_interval = full_lane_interval_mask(full_start, full_end).cpu()
    full_valid_mask = (full_valid_scores >= float(point_valid_thr)) & full_interval
    full_scores = full_lane_proposal_score_probability(
        pred_full_lane_exist_logits.detach().float().cpu(),
        pred_full_lane_quality_logits.detach().float().cpu(),
        pred_full_lane_valid_logits.detach().float().cpu(),
        pred_full_lane_start_logits.detach().float().cpu(),
        pred_full_lane_end_logits.detach().float().cpu(),
    ).cpu()

    points = torch.cat((base_points, full_points), dim=0)
    valid_scores = torch.cat((base_valid_scores, full_valid_scores), dim=0)
    valid_masks = torch.cat((base_valid_mask, full_valid_mask), dim=0)
    scores = torch.cat((base_scores, full_scores), dim=0)
    source = torch.cat(
        (
            torch.zeros((base_points.shape[0],), dtype=torch.long),
            torch.ones((full_points.shape[0],), dtype=torch.long),
        ),
        dim=0,
    )
    proposal_indices = torch.cat(
        (
            torch.arange(base_points.shape[0], dtype=torch.long),
            torch.arange(full_points.shape[0], dtype=torch.long),
        ),
        dim=0,
    )
    if int(min_points) < 1 or int(min_points) > int(point_count):
        raise ValueError(f"min_points must be in [1, K], got {min_points} for K={point_count}.")

    keep = (scores >= float(score_thr)) & (valid_masks.sum(dim=1) >= int(min_points))
    keep_indices = torch.nonzero(keep, as_tuple=False).flatten()
    if keep_indices.numel() == 0:
        return []

    points = points[keep_indices]
    valid_scores = valid_scores[keep_indices]
    valid_masks = valid_masks[keep_indices]
    scores = scores[keep_indices]
    source = source[keep_indices]
    proposal_indices = proposal_indices[keep_indices]
    order = torch.argsort(scores, descending=True)
    points = points[order]
    valid_scores = valid_scores[order]
    valid_masks = valid_masks[order]
    scores = scores[order]
    source = source[order]
    proposal_indices = proposal_indices[order]

    if nms_dist_px > 0.0:
        if image_shape is None:
            raise ValueError("full_lane_decode requires image_shape when nms_dist_px > 0.")
        keep_nms = lane_nms(
            points,
            scores,
            image_shape=image_shape,
            dist_thr_px=float(nms_dist_px),
            valid_masks=valid_masks,
        )
        points = points[keep_nms]
        valid_scores = valid_scores[keep_nms]
        valid_masks = valid_masks[keep_nms]
        scores = scores[keep_nms]
        source = source[keep_nms]
        proposal_indices = proposal_indices[keep_nms]

    if max_det is not None and int(max_det) > 0:
        keep_max = torch.arange(min(int(max_det), points.shape[0]), dtype=torch.long)
        points = points[keep_max]
        valid_scores = valid_scores[keep_max]
        valid_masks = valid_masks[keep_max]
        scores = scores[keep_max]
        source = source[keep_max]
        proposal_indices = proposal_indices[keep_max]

    scale = None
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        scale = torch.tensor([w, h], dtype=points.dtype).view(1, 1, 2)

    lanes: list[dict] = []
    for lane_points, lane_valid_scores, lane_valid_mask, lane_score, lane_source, lane_idx in zip(
        points,
        valid_scores,
        valid_masks,
        scores,
        source,
        proposal_indices,
    ):
        visible_mask = longest_contiguous_valid_mask(lane_valid_mask, min_points=int(min_points))
        if int(visible_mask.sum()) < int(min_points):
            continue
        lane_norm = lane_points.numpy().astype(np.float32)
        item = {
            "score": float(lane_score),
            "query": int(lane_idx),
            "source": "full_lane_proposal" if int(lane_source) else "base_query",
            "points_norm": lane_norm,
            "point_valid_scores": lane_valid_scores.numpy().astype(np.float32),
            "point_valid": visible_mask.numpy().astype(np.float32),
            "visible_points_norm": lane_norm[visible_mask.numpy()],
        }
        if int(lane_source):
            item["full_lane_proposal_index"] = int(lane_idx)
        if scale is not None:
            points_px = (lane_points.unsqueeze(0) * scale).squeeze(0).numpy().astype(np.float32)
            item["points"] = points_px
            item["visible_points"] = points_px[visible_mask.numpy()]
        lanes.append(item)
    return lanes


def decode_gcs_predictions(
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None = None,
    pred_count_logits: torch.Tensor | None = None,
    pred_short_candidate_points: torch.Tensor | None = None,
    pred_short_candidate_logits: torch.Tensor | None = None,
    oracle_count: int | None = None,
    image_shape: tuple[int, int] | None = None,
    score_thr: float = 0.5,
    point_valid_thr: float = 0.5,
    min_points: int = 2,
    max_det: int | None = None,
    nms_dist_px: float = 0.0,
    valid_before_maxdet: bool = False,
    count_aware_topk: bool = False,
    count_aware_min_k: int = 3,
    count_aware_max_k: int = 5,
    count_aware_length_norm: float = 12.0,
    count_aware_extra_margin: int = 0,
    count_mode: str = "score_sum",
    candidate_decode: bool = False,
    candidate_score_thr: float = 0.05,
    candidate_short_min_points: int = 2,
    candidate_short_max_points: int = 10,
    pred_short_segment_points: torch.Tensor | None = None,
    pred_short_segment_logits: torch.Tensor | None = None,
    pred_short_segment_window_mask: torch.Tensor | None = None,
    segment_decode: bool = False,
    segment_score_thr: float = 0.5,
    segment_short_min_points: int = 3,
    segment_short_max_points: int = 10,
    segment_pred_valid_overlap_min: int = 0,
    segment_score_as_lane_score: bool = False,
    segment_rescue_base_miss_only: bool = False,
    pred_short_segment_choice_logits: torch.Tensor | None = None,
    full_lane_decode: bool = False,
    pred_full_lane_points: torch.Tensor | None = None,
    pred_full_lane_valid_logits: torch.Tensor | None = None,
    pred_full_lane_exist_logits: torch.Tensor | None = None,
    pred_full_lane_quality_logits: torch.Tensor | None = None,
    pred_full_lane_start_logits: torch.Tensor | None = None,
    pred_full_lane_end_logits: torch.Tensor | None = None,
) -> list[dict]:
    """Decode ``pred_points`` and ``pred_logits`` into ordered lane point sequences.

    Args:
        pred_points: Q x K x 2 normalized point predictions in the GCS training coordinate system.
        pred_logits: Q existence logits for the lane queries.
        pred_valid_logits: Optional Q x K visibility logits. When present, decoded lanes keep full K points
            for metrics but drawing/export uses the longest visible contiguous point run.
        oracle_count: Optional GT lane count used only with diagnostic ``count_mode='oracle_gt'``.
        image_shape: Optional original image shape as (height, width). If provided, pixel points are added.
        score_thr: Existence probability threshold.
        point_valid_thr: Per-point visibility probability threshold.
        min_points: Minimum number of points required to keep a lane.
        max_det: Optional maximum number of kept lanes after score sorting.
        nms_dist_px: Optional duplicate-lane suppression threshold in pixels. 0 disables lane NMS.
        valid_before_maxdet: If true, discard point-valid/min_points failures before ``max_det`` truncation.
        count_aware_topk: If true, keep only the quality-best ``k_hat`` lanes after conf/NMS.
        count_aware_min_k: Minimum dynamic lane count when count-aware top-k is enabled.
        count_aware_max_k: Maximum dynamic lane count when count-aware top-k is enabled.
        count_aware_length_norm: Visible-point count that saturates the count-aware length factor.
        count_aware_extra_margin: Extra hypotheses to keep above ``k_hat`` before capping by ``max_det``.
        count_mode: Count source for count-aware top-k. ``score_sum`` preserves the historical
            sum(sigmoid(pred_logits)) behavior with the configured k range; ``count_logits`` uses
            the query Count Head's fixed 2/3/4/5 class mapping; ``oracle_gt`` uses the supplied
            GT lane count for diagnostic-only evaluation.
        candidate_decode: If true, replace only prediction-short query geometry with the highest-scoring
            lateral candidate. This preserves query existence scores and is off by default.
        candidate_score_thr: Minimum selector probability required before candidate geometry replaces base geometry.
        candidate_short_min_points: Minimum predicted visible anchors for candidate-decode applicability.
        candidate_short_max_points: Maximum predicted visible anchors for candidate-decode applicability.
        pred_short_segment_points: Optional Q x S x K x 2 local short-segment proposals for one image.
        pred_short_segment_logits: Optional Q x S selector logits for local short-segment proposals.
        pred_short_segment_window_mask: Optional S x K validity mask for local short-segment proposals.
        segment_decode: If true, use the best valid local segment as a base-choice option. The base query
            has a fixed zero logit, so the segment must exceed ``segment_score_thr`` to replace it.
        segment_score_thr: Minimum local-segment probability required before replacing base geometry.
        segment_short_min_points: Minimum fixed-y window length allowed for segment selection.
        segment_short_max_points: Maximum fixed-y window length allowed for segment selection.
        segment_pred_valid_overlap_min: Optional minimum overlap between the segment window and base
            predicted-valid anchors before the segment can be selected.
        segment_score_as_lane_score: If true, a selected segment may raise the final lane score to
            ``max(base_probability, segment_probability)``. This is off by default.
        segment_rescue_base_miss_only: If true, only queries whose base probability is below
            ``score_thr`` may be replaced by a segment. This is off by default.
        pred_short_segment_choice_logits: Optional Q x (S+1) logits from the v12 unified selector.
            Class 0 is the base geometry and classes 1..S are the segment windows. When present,
            ``segment_decode`` uses this unified choice distribution instead of the legacy segment
            threshold path.
        full_lane_decode: If true, decode base queries and independent full-lane proposals as one set.
            This is prediction-only and default-off.

    Returns:
        A list of dictionaries with score, query index, normalized points, and optional pixel points.
    """
    if pred_logits.ndim == 2 and pred_logits.shape[-1] == 1:
        pred_logits = pred_logits.squeeze(-1)
    if pred_points.ndim != 3 or pred_points.shape[-1] != 2:
        raise ValueError(f"pred_points must have shape Q x K x 2, got {tuple(pred_points.shape)}.")
    if pred_logits.ndim != 1 or pred_logits.shape[0] != pred_points.shape[0]:
        raise ValueError(
            f"pred_logits must have shape Q and match pred_points Q, got {tuple(pred_logits.shape)} "
            f"vs {tuple(pred_points.shape)}."
        )
    if pred_valid_logits is not None:
        if pred_valid_logits.ndim == 3 and pred_valid_logits.shape[-1] == 1:
            pred_valid_logits = pred_valid_logits.squeeze(-1)
        if pred_valid_logits.shape != pred_points.shape[:2]:
            raise ValueError(
                "pred_valid_logits must have shape Q x K matching pred_points, "
                f"got {tuple(pred_valid_logits.shape)} vs {tuple(pred_points.shape[:2])}."
            )
    if full_lane_decode:
        if candidate_decode or segment_decode:
            raise ValueError("full_lane_decode is mutually exclusive with candidate_decode and segment_decode.")
        if count_aware_topk:
            raise ValueError("full_lane_decode is mutually exclusive with count_aware_topk.")
        required = (
            pred_full_lane_points,
            pred_full_lane_valid_logits,
            pred_full_lane_exist_logits,
            pred_full_lane_quality_logits,
            pred_full_lane_start_logits,
            pred_full_lane_end_logits,
        )
        if any(value is None for value in required):
            raise ValueError(
                "full_lane_decode requires pred_full_lane_points, pred_full_lane_valid_logits, "
                "pred_full_lane_exist_logits, pred_full_lane_quality_logits, "
                "pred_full_lane_start_logits, and pred_full_lane_end_logits."
            )
        return _decode_full_lane_proposals(
            pred_points,
            pred_logits,
            pred_valid_logits,
            pred_full_lane_points,
            pred_full_lane_valid_logits,
            pred_full_lane_exist_logits,
            pred_full_lane_quality_logits,
            pred_full_lane_start_logits,
            pred_full_lane_end_logits,
            image_shape=image_shape,
            score_thr=score_thr,
            point_valid_thr=point_valid_thr,
            min_points=min_points,
            max_det=max_det,
            nms_dist_px=nms_dist_px,
        )

    points = pred_points.detach().float().cpu().clamp(0.0, 1.0)
    scores = pred_logits.detach().float().cpu().sigmoid()
    short_candidate_indices = torch.full((points.shape[0],), -1, dtype=torch.long)
    short_candidate_scores = torch.zeros((points.shape[0],), dtype=torch.float32)
    short_segment_indices = torch.full((points.shape[0],), -1, dtype=torch.long)
    short_segment_scores = torch.zeros((points.shape[0],), dtype=torch.float32)
    segment_window_override = torch.ones(points.shape[:2], dtype=torch.bool)
    if candidate_decode and segment_decode:
        raise ValueError("candidate_decode and segment_decode are mutually exclusive query decode modes.")
    if segment_score_as_lane_score and not segment_decode:
        raise ValueError("segment_score_as_lane_score requires segment_decode=True.")
    if segment_rescue_base_miss_only and not segment_score_as_lane_score:
        raise ValueError("segment_rescue_base_miss_only requires segment_score_as_lane_score=True.")
    if candidate_decode:
        if count_aware_topk:
            raise ValueError("candidate_decode is mutually exclusive with count_aware_topk for query decode.")
        if pred_valid_logits is None:
            raise ValueError("candidate_decode requires pred_valid_logits for prediction-only short-query gating.")
        if pred_short_candidate_points is None or pred_short_candidate_logits is None:
            raise ValueError("candidate_decode requires pred_short_candidate_points and pred_short_candidate_logits.")
        candidate_points = pred_short_candidate_points.detach().float().cpu().clamp(0.0, 1.0)
        candidate_logits = pred_short_candidate_logits.detach().float().cpu()
        if candidate_points.ndim != 4 or candidate_points.shape[-1] != 2:
            raise ValueError(
                "pred_short_candidate_points must have shape Q x M x K x 2 for one image, "
                f"got {tuple(candidate_points.shape)}."
            )
        if candidate_points.shape[0] != points.shape[0] or candidate_points.shape[2] != points.shape[1]:
            raise ValueError(
                "pred_short_candidate_points Q,K must match pred_points, "
                f"got {tuple(candidate_points.shape)} vs {tuple(points.shape)}."
            )
        if candidate_logits.shape != candidate_points.shape[:2]:
            raise ValueError(
                "pred_short_candidate_logits must have shape Q x M matching candidate points, "
                f"got {tuple(candidate_logits.shape)} vs {tuple(candidate_points.shape[:2])}."
            )
        valid_scores_for_gate = pred_valid_logits.detach().float().cpu().sigmoid()
        if valid_scores_for_gate.shape != points.shape[:2]:
            raise ValueError(
                "pred_valid_logits must have shape Q x K before candidate_decode, "
                f"got {tuple(valid_scores_for_gate.shape)} vs {tuple(points.shape[:2])}."
            )
        visible_counts = (valid_scores_for_gate >= float(point_valid_thr)).sum(dim=1)
        short_gate = (visible_counts >= int(candidate_short_min_points)) & (
            visible_counts <= int(candidate_short_max_points)
        )
        candidate_prob = candidate_logits.sigmoid()
        best_scores, best_indices = candidate_prob.max(dim=1)
        apply_gate = short_gate & (best_scores >= float(candidate_score_thr))
        if bool(apply_gate.any()):
            query_ids = torch.nonzero(apply_gate, as_tuple=False).flatten()
            points[query_ids] = candidate_points[query_ids, best_indices[query_ids]]
            short_candidate_indices[query_ids] = best_indices[query_ids]
            short_candidate_scores[query_ids] = best_scores[query_ids]
    if segment_decode:
        if count_aware_topk:
            raise ValueError("segment_decode is mutually exclusive with count_aware_topk for query decode.")
        if pred_valid_logits is None:
            raise ValueError("segment_decode requires pred_valid_logits for prediction-only gating.")
        if (
            pred_short_segment_points is None
            or pred_short_segment_logits is None
            or pred_short_segment_window_mask is None
        ):
            raise ValueError(
                "segment_decode requires pred_short_segment_points, pred_short_segment_logits, "
                "and pred_short_segment_window_mask."
            )
        segment_points = pred_short_segment_points.detach().float().cpu().clamp(0.0, 1.0)
        segment_logits = pred_short_segment_logits.detach().float().cpu()
        segment_window_mask = pred_short_segment_window_mask.detach().bool().cpu()
        if segment_points.ndim != 4 or segment_points.shape[-1] != 2:
            raise ValueError(
                "pred_short_segment_points must have shape Q x S x K x 2 for one image, "
                f"got {tuple(segment_points.shape)}."
            )
        if segment_points.shape[0] != points.shape[0] or segment_points.shape[2] != points.shape[1]:
            raise ValueError(
                "pred_short_segment_points Q,K must match pred_points, "
                f"got {tuple(segment_points.shape)} vs {tuple(points.shape)}."
            )
        if segment_logits.shape != segment_points.shape[:2]:
            raise ValueError(
                "pred_short_segment_logits must have shape Q x S matching segment points, "
                f"got {tuple(segment_logits.shape)} vs {tuple(segment_points.shape[:2])}."
            )
        if segment_window_mask.ndim == 3 and segment_window_mask.shape[0] == 1:
            segment_window_mask = segment_window_mask[0]
        if segment_window_mask.ndim != 2 or segment_window_mask.shape != segment_points.shape[1:3]:
            raise ValueError(
                "pred_short_segment_window_mask must have shape S x K matching segment points, "
                f"got {tuple(segment_window_mask.shape)} vs {tuple(segment_points.shape[1:3])}."
            )
        segment_short_min_points = int(segment_short_min_points)
        segment_short_max_points = int(segment_short_max_points)
        segment_pred_valid_overlap_min = int(segment_pred_valid_overlap_min)
        if segment_short_min_points < 1 or segment_short_max_points < segment_short_min_points:
            raise ValueError(
                "segment short point bounds must satisfy 1 <= min <= max, "
                f"got {segment_short_min_points}/{segment_short_max_points}."
            )
        if segment_pred_valid_overlap_min < 0:
            raise ValueError(
                f"segment_pred_valid_overlap_min must be >= 0, got {segment_pred_valid_overlap_min}."
            )
        valid_scores_for_gate = pred_valid_logits.detach().float().cpu().sigmoid()
        if valid_scores_for_gate.shape != points.shape[:2]:
            raise ValueError(
                "pred_valid_logits must have shape Q x K before segment_decode, "
                f"got {tuple(valid_scores_for_gate.shape)} vs {tuple(points.shape[:2])}."
            )
        segment_lengths = segment_window_mask.sum(dim=1)
        segment_gate = (segment_lengths >= segment_short_min_points) & (
            segment_lengths <= segment_short_max_points
        )
        segment_gate = segment_gate.view(1, -1).expand(points.shape[0], -1)
        if segment_pred_valid_overlap_min > 0:
            query_valid = valid_scores_for_gate >= float(point_valid_thr)
            valid_overlap = (
                segment_window_mask.view(1, segment_window_mask.shape[0], segment_window_mask.shape[1])
                & query_valid.view(query_valid.shape[0], 1, query_valid.shape[1])
            ).sum(dim=2)
            segment_gate = segment_gate & (valid_overlap >= segment_pred_valid_overlap_min)
        if pred_short_segment_choice_logits is not None:
            choice_logits = pred_short_segment_choice_logits.detach().float().cpu()
            expected_choice_shape = (points.shape[0], segment_points.shape[1] + 1)
            if tuple(choice_logits.shape) != expected_choice_shape:
                raise ValueError(
                    "pred_short_segment_choice_logits must have shape Q x (S+1), "
                    f"got {tuple(choice_logits.shape)} vs {expected_choice_shape}."
                )
            choice_logits = choice_logits.clone()
            choice_logits[:, 1:] = choice_logits[:, 1:].masked_fill(~segment_gate, float("-inf"))
            choice_probs = torch.softmax(choice_logits, dim=1)
            best_choice_probs, best_choice = choice_probs.max(dim=1)
            apply_gate = (best_choice > 0) & torch.isfinite(best_choice_probs)
            if bool(apply_gate.any()):
                query_ids = torch.nonzero(apply_gate, as_tuple=False).flatten()
                selected_indices = best_choice[query_ids] - 1
                points[query_ids] = segment_points[query_ids, selected_indices]
                short_segment_indices[query_ids] = selected_indices
                short_segment_scores[query_ids] = best_choice_probs[query_ids]
                segment_window_override[query_ids] = segment_window_mask[selected_indices]
                # The unified selector is calibrated against the base class, so
                # its selected probability is the only segment score allowed to
                # rescue a low base existence score.
                scores[query_ids] = torch.maximum(scores[query_ids], best_choice_probs[query_ids])
        else:
            selection_logits = segment_logits.masked_fill(~segment_gate, float("-inf"))
            best_segment_logits, best_indices = selection_logits.max(dim=1)
            threshold_logit = _prob_threshold_to_logit_floor(segment_score_thr)
            apply_gate = torch.isfinite(best_segment_logits) & (best_segment_logits >= threshold_logit)
            if segment_rescue_base_miss_only:
                apply_gate = apply_gate & (scores < float(score_thr))
            if bool(apply_gate.any()):
                query_ids = torch.nonzero(apply_gate, as_tuple=False).flatten()
                selected_indices = best_indices[query_ids]
                points[query_ids] = segment_points[query_ids, selected_indices]
                short_segment_indices[query_ids] = selected_indices
                short_segment_scores[query_ids] = best_segment_logits[query_ids].sigmoid()
                segment_window_override[query_ids] = segment_window_mask[selected_indices]
                if segment_score_as_lane_score:
                    scores[query_ids] = torch.maximum(scores[query_ids], short_segment_scores[query_ids])
    count_aware_k = None
    count_aware_base_k = None
    if count_aware_topk:
        min_k, max_k, length_norm, extra_margin = _validate_count_aware_topk(
            count_aware_min_k,
            count_aware_max_k,
            count_aware_length_norm,
            count_aware_extra_margin,
        )
        count_mode = str(count_mode or "score_sum")
        if count_mode == "count_logits":
            if pred_count_logits is None:
                raise ValueError("count_mode='count_logits' requires pred_count_logits.")
            count_logits = pred_count_logits.detach().float().cpu().reshape(-1)
            count_logits_min_k = 2
            count_logits_max_k = 5
            expected = count_logits_max_k - count_logits_min_k + 1
            if count_logits.numel() != expected:
                raise ValueError(
                    f"pred_count_logits must have {expected} classes for range [{count_logits_min_k}, {count_logits_max_k}], "
                    f"got {count_logits.numel()}."
                )
            count_aware_k = int(count_logits.argmax().item()) + count_logits_min_k
        elif count_mode == "oracle_gt":
            if oracle_count is None:
                raise ValueError("count_mode='oracle_gt' requires oracle_count.")
            count_aware_k = int(oracle_count)
            if count_aware_k < 0:
                raise ValueError(f"oracle_count must be >= 0, got {count_aware_k}.")
        elif count_mode == "score_sum":
            count_aware_k = _count_aware_k_hat(float(scores.sum()), min_k=min_k, max_k=max_k)
        else:
            raise ValueError(f"Unsupported count_mode={count_mode!r}.")
        count_aware_base_k = int(count_aware_k)
        count_aware_k = int(count_aware_k) + int(extra_margin)
        if max_det is not None and int(max_det) > 0:
            count_aware_k = min(int(count_aware_k), int(max_det))
    else:
        length_norm = float(count_aware_length_norm)
        extra_margin = int(count_aware_extra_margin)
    point_valid_scores = pred_valid_logits.detach().float().cpu().sigmoid() if pred_valid_logits is not None else None
    query_indices = torch.arange(points.shape[0], dtype=torch.long)

    if min_points > points.shape[1]:
        return []

    keep = torch.nonzero(scores >= float(score_thr), as_tuple=False).flatten()
    if keep.numel() == 0:
        return []

    sorted_points = []
    sorted_valid_scores = []
    sorted_segment_window_overrides = []
    for i in keep:
        order_i = torch.argsort(points[i, :, 1], descending=True, stable=True)
        sorted_points.append(points[i][order_i])
        if point_valid_scores is not None:
            sorted_valid_scores.append(point_valid_scores[i][order_i])
        sorted_segment_window_overrides.append(segment_window_override[i][order_i])
    points = torch.stack(sorted_points, dim=0)
    point_valid_scores = torch.stack(sorted_valid_scores, dim=0) if sorted_valid_scores else None
    segment_window_override = torch.stack(sorted_segment_window_overrides, dim=0)
    scores = scores[keep]
    query_indices = query_indices[keep]
    short_candidate_indices = short_candidate_indices[keep]
    short_candidate_scores = short_candidate_scores[keep]
    short_segment_indices = short_segment_indices[keep]
    short_segment_scores = short_segment_scores[keep]
    if segment_decode and point_valid_scores is not None:
        point_valid_scores = point_valid_scores * segment_window_override.to(dtype=point_valid_scores.dtype)

    order = torch.argsort(scores, descending=True)
    if nms_dist_px > 0.0:
        if image_shape is None:
            raise ValueError("decode_gcs_predictions requires image_shape when nms_dist_px > 0.")
        sorted_points = points[order]
        sorted_scores = scores[order]
        sorted_queries = query_indices[order]
        sorted_candidate_indices = short_candidate_indices[order]
        sorted_candidate_scores = short_candidate_scores[order]
        sorted_segment_indices = short_segment_indices[order]
        sorted_segment_scores = short_segment_scores[order]
        sorted_valid_masks = None
        if point_valid_scores is not None:
            sorted_valid_masks = torch.stack(
                [
                    longest_contiguous_valid_mask(v >= float(point_valid_thr), min_points=min_points)
                    for v in point_valid_scores[order]
                ],
                dim=0,
            ).to(device=sorted_points.device)
        keep_sorted = lane_nms(
            sorted_points,
            sorted_scores,
            image_shape=image_shape,
            dist_thr_px=float(nms_dist_px),
            valid_masks=sorted_valid_masks,
        )
        points = sorted_points[keep_sorted]
        scores = sorted_scores[keep_sorted]
        query_indices = sorted_queries[keep_sorted]
        short_candidate_indices = sorted_candidate_indices[keep_sorted]
        short_candidate_scores = sorted_candidate_scores[keep_sorted]
        short_segment_indices = sorted_segment_indices[keep_sorted]
        short_segment_scores = sorted_segment_scores[keep_sorted]
        if point_valid_scores is not None:
            point_valid_scores = point_valid_scores[order][keep_sorted]
        order = torch.arange(scores.shape[0], dtype=torch.long)
    if valid_before_maxdet and point_valid_scores is not None:
        valid_masks = torch.stack(
            [
                longest_contiguous_valid_mask(v >= float(point_valid_thr), min_points=min_points)
                for v in point_valid_scores
            ],
            dim=0,
        )
        keep_valid = torch.nonzero(valid_masks.sum(dim=1) >= int(min_points), as_tuple=False).flatten()
        points = points[keep_valid]
        scores = scores[keep_valid]
        query_indices = query_indices[keep_valid]
        short_candidate_indices = short_candidate_indices[keep_valid]
        short_candidate_scores = short_candidate_scores[keep_valid]
        short_segment_indices = short_segment_indices[keep_valid]
        short_segment_scores = short_segment_scores[keep_valid]
        point_valid_scores = point_valid_scores[keep_valid]
        order = torch.argsort(scores, descending=True)
    if max_det is not None and max_det > 0:
        order = order[: int(max_det)]
    points = points[order]
    scores = scores[order]
    query_indices = query_indices[order]
    short_candidate_indices = short_candidate_indices[order]
    short_candidate_scores = short_candidate_scores[order]
    short_segment_indices = short_segment_indices[order]
    short_segment_scores = short_segment_scores[order]
    if point_valid_scores is not None:
        point_valid_scores = point_valid_scores[order]

    scale = None
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        scale = torch.tensor([w, h], dtype=points.dtype).view(1, 1, 2)

    lanes = []
    for lane_i, (lane_points, score, query_idx) in enumerate(zip(points, scores, query_indices)):
        lane_norm = lane_points.numpy().astype(np.float32)
        item = {
            "score": float(score),
            "query": int(query_idx),
            "points_norm": lane_norm,
            "count_mode": str(count_mode or "score_sum"),
            "decoded_count_k": int(count_aware_k) if count_aware_k is not None else -1,
            "decoded_count_base_k": int(count_aware_base_k) if count_aware_base_k is not None else -1,
            "count_aware_extra_margin": int(extra_margin),
        }
        cand_idx = int(short_candidate_indices[lane_i].item())
        if cand_idx >= 0:
            item["candidate_decode_applied"] = True
            item["short_candidate_index"] = cand_idx
            item["short_candidate_score"] = float(short_candidate_scores[lane_i].item())
        segment_idx = int(short_segment_indices[lane_i].item())
        if segment_idx >= 0:
            item["segment_decode_applied"] = True
            item["short_segment_index"] = segment_idx
            item["short_segment_score"] = float(short_segment_scores[lane_i].item())
        visible_mask = None
        if point_valid_scores is not None:
            visible_mask_t = longest_contiguous_valid_mask(
                point_valid_scores[lane_i] >= float(point_valid_thr),
                min_points=min_points,
            )
            if int(visible_mask_t.sum()) < int(min_points):
                continue
            visible_mask = visible_mask_t.numpy().astype(bool)
            item["point_valid_scores"] = point_valid_scores[lane_i].numpy().astype(np.float32)
            item["point_valid"] = visible_mask.astype(np.float32)
            item["visible_points_norm"] = lane_norm[visible_mask]
        if scale is not None:
            points_px = (lane_points.unsqueeze(0) * scale).squeeze(0).numpy().astype(np.float32)
            item["points"] = points_px
            if visible_mask is not None:
                item["visible_points"] = points_px[visible_mask]
        lanes.append(item)
    if count_aware_topk and count_aware_k is not None:
        lanes = _apply_count_aware_topk(lanes, k_hat=count_aware_k, length_norm=length_norm)
    return lanes


def draw_gcs_lanes(
    image: np.ndarray,
    lanes: list[dict],
    show_scores: bool = True,
    line_width: int = 2,
    point_radius: int = 3,
) -> np.ndarray:
    """Draw decoded GCS lane point sequences on a BGR image."""
    out = image.copy()
    h, w = out.shape[:2]
    for i, lane in enumerate(lanes):
        pts = lane.get("visible_points", lane.get("points"))
        if pts is None:
            pts_norm = lane.get("visible_points_norm", lane["points_norm"])
            pts = np.asarray(pts_norm, dtype=np.float32) * np.array([w, h], dtype=np.float32)
        pts = np.asarray(pts, dtype=np.float32)
        if "point_valid" in lane and "visible_points" not in lane and "visible_points_norm" not in lane:
            valid = np.asarray(lane["point_valid"], dtype=np.float32) > 0.5
            if valid.shape[0] == pts.shape[0]:
                pts = pts[valid]
        if pts.shape[0] < 2:
            continue

        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts_i = np.round(pts).astype(np.int32)
        color = GCS_LANE_COLORS[i % len(GCS_LANE_COLORS)]

        cv2.polylines(out, [pts_i], isClosed=False, color=color, thickness=line_width, lineType=cv2.LINE_AA)
        for x, y in pts_i:
            cv2.circle(out, (int(x), int(y)), point_radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        if show_scores:
            x0, y0 = pts_i[0]
            cv2.putText(
                out,
                f"{lane['score']:.2f}",
                (int(x0), max(12, int(y0) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return out


def save_gcs_lanes_txt(path: str | Path, lanes: list[dict], save_conf: bool = True) -> str:
    """Save decoded GCS lanes as normalized point sequences, one lane per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for lane in lanes:
        pts = np.asarray(lane.get("visible_points_norm", lane["points_norm"]), dtype=np.float32)
        if "point_valid" in lane and "visible_points_norm" not in lane:
            valid = np.asarray(lane["point_valid"], dtype=np.float32) > 0.5
            if valid.shape[0] == pts.shape[0]:
                pts = pts[valid]
        pts = pts.reshape(-1)
        values: list[float | int] = [int(lane.get("query", -1))]
        if save_conf:
            values.append(float(lane["score"]))
        values.extend(float(x) for x in pts)
        lines.append(" ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in values))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return str(path)
