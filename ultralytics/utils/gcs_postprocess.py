# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Post-processing helpers for GCS-YOLO-Lane structured lane predictions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

__all__ = (
    "decode_gcs_predictions",
    "decode_gcs_candidate_pool",
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


def decode_gcs_candidate_pool(
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None = None,
    proposal_points: torch.Tensor | None = None,
    proposal_logits: torch.Tensor | None = None,
    proposal_valid_logits: torch.Tensor | None = None,
    proposal_score_thr: float | None = None,
    proposal_point_valid_thr: float | None = None,
    **kwargs,
) -> list[dict]:
    """Decode base queries and optional short proposals as one capped candidate pool."""
    if proposal_points is None or proposal_logits is None:
        return decode_gcs_predictions(pred_points, pred_logits, pred_valid_logits=pred_valid_logits, **kwargs)
    if proposal_valid_logits is None:
        raise ValueError("proposal_valid_logits is required when proposal_points are provided.")
    base_count = int(pred_points.shape[0])
    base_score_thr = float(kwargs.get("score_thr", 0.5))
    base_point_valid_thr = float(kwargs.get("point_valid_thr", 0.5))
    proposal_score_thr = base_score_thr if proposal_score_thr is None else float(proposal_score_thr)
    proposal_point_valid_thr = (
        base_point_valid_thr if proposal_point_valid_thr is None else float(proposal_point_valid_thr)
    )
    if not 0.0 <= proposal_score_thr <= 1.0:
        raise ValueError(f"proposal_score_thr must be in [0, 1], got {proposal_score_thr}.")
    if not 0.0 <= proposal_point_valid_thr <= 1.0:
        raise ValueError(f"proposal_point_valid_thr must be in [0, 1], got {proposal_point_valid_thr}.")
    proposal_keep = proposal_logits.detach().float().sigmoid() >= proposal_score_thr
    proposal_points = proposal_points[proposal_keep]
    proposal_logits = proposal_logits[proposal_keep]
    proposal_valid_logits = proposal_valid_logits[proposal_keep]
    if proposal_valid_logits.numel() > 0 and proposal_point_valid_thr != base_point_valid_thr:
        eps = torch.finfo(proposal_valid_logits.dtype).eps
        base_prob = proposal_valid_logits.new_tensor(base_point_valid_thr).clamp(eps, 1.0 - eps)
        proposal_prob = proposal_valid_logits.new_tensor(proposal_point_valid_thr).clamp(eps, 1.0 - eps)
        proposal_valid_logits = proposal_valid_logits + torch.logit(base_prob) - torch.logit(proposal_prob)
    points = torch.cat((pred_points, proposal_points), dim=0)
    logits = torch.cat((pred_logits, proposal_logits), dim=0)
    if pred_valid_logits is None:
        base_valid = torch.zeros(
            (base_count, int(pred_points.shape[1])),
            dtype=proposal_valid_logits.dtype,
            device=proposal_valid_logits.device,
        )
        valid = torch.cat((base_valid, proposal_valid_logits), dim=0)
    else:
        valid = torch.cat((pred_valid_logits, proposal_valid_logits), dim=0)
    lanes = decode_gcs_predictions(points, logits, pred_valid_logits=valid, **kwargs)
    for lane in lanes:
        query = int(lane["query"])
        lane["candidate_source"] = "base" if query < base_count else "short_proposal"
        lane["query"] = query if query < base_count else query - base_count
    return lanes


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


def decode_gcs_predictions(
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None = None,
    pred_count_logits: torch.Tensor | None = None,
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

    points = pred_points.detach().float().cpu().clamp(0.0, 1.0)
    scores = pred_logits.detach().float().cpu().sigmoid()
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
    for i in keep:
        order_i = torch.argsort(points[i, :, 1], descending=True, stable=True)
        sorted_points.append(points[i][order_i])
        if point_valid_scores is not None:
            sorted_valid_scores.append(point_valid_scores[i][order_i])
    points = torch.stack(sorted_points, dim=0)
    point_valid_scores = torch.stack(sorted_valid_scores, dim=0) if sorted_valid_scores else None
    scores = scores[keep]
    query_indices = query_indices[keep]

    order = torch.argsort(scores, descending=True)
    if nms_dist_px > 0.0:
        if image_shape is None:
            raise ValueError("decode_gcs_predictions requires image_shape when nms_dist_px > 0.")
        sorted_points = points[order]
        sorted_scores = scores[order]
        sorted_queries = query_indices[order]
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
        point_valid_scores = point_valid_scores[keep_valid]
        order = torch.argsort(scores, descending=True)
    if max_det is not None and max_det > 0:
        order = order[: int(max_det)]
    points = points[order]
    scores = scores[order]
    query_indices = query_indices[order]
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
