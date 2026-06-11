# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Candidate and GT matching helpers for GCS lane-count diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class GCSLaneCandidate:
    """Unified pre/post-NMS lane-candidate record used by count diagnostics."""

    image_id: str
    query_idx: int
    points: Any
    valid_probs: Any
    exist_logit: float
    exist_score: float
    point_valid_mean: float
    point_valid_max: float
    valid_points: int
    lane_quality: float
    geometry_feat: dict[str, float] = field(default_factory=dict)
    pre_nms_rank: int = 0
    pre_nms_score: float = 0.0
    keep_after_nms: bool = False
    suppressed_by: int | None = None
    matched_gt_id: int = -1
    line_iou: float = 0.0
    official_lane_acc: float = 0.0
    source: str = "normal"

    def points_tensor(self) -> torch.Tensor:
        """Return normalized K x 2 points as a float tensor."""
        return torch.as_tensor(self.points, dtype=torch.float32)

    def valid_tensor(self, threshold: float = 0.5) -> torch.Tensor:
        """Return a K bool visibility mask."""
        valid = torch.as_tensor(self.valid_probs, dtype=torch.float32)
        return valid >= float(threshold)

    def to_json(self) -> dict:
        """Return a JSON-serializable candidate dictionary."""
        out = asdict(self)
        out["points"] = np.asarray(self.points, dtype=np.float32).round(6).tolist()
        out["valid_probs"] = np.asarray(self.valid_probs, dtype=np.float32).round(6).tolist()
        return out


def _as_points(points: Any) -> torch.Tensor:
    pts = torch.as_tensor(points, dtype=torch.float32)
    if pts.ndim != 2 or pts.shape[-1] != 2:
        raise ValueError(f"lane points must have shape K x 2, got {tuple(pts.shape)}.")
    return pts


def _as_valid(valid: Any, k: int) -> torch.Tensor:
    mask = torch.as_tensor(valid, dtype=torch.float32)
    if mask.ndim != 1 or mask.shape[0] != k:
        raise ValueError(f"lane valid mask must have shape K={k}, got {tuple(mask.shape)}.")
    return mask > 0.5


def lane_similarity(
    pred_points: Any,
    pred_valid: Any,
    gt_points: Any,
    gt_valid: Any,
    *,
    image_shape: tuple[int, int] = (544, 960),
    dist_thr_px: float = 20.0,
    min_overlap: int = 2,
) -> float:
    """Return a [0,1] fixed-y lane similarity based on shared visible-anchor x error."""
    pred = _as_points(pred_points)
    gt = _as_points(gt_points)
    if pred.shape != gt.shape:
        return 0.0
    pv = _as_valid(pred_valid, pred.shape[0])
    gv = _as_valid(gt_valid, gt.shape[0])
    keep = pv & gv
    if int(keep.sum().item()) < int(min_overlap):
        return 0.0
    width = float(image_shape[1])
    dx = (pred[keep, 0] - gt[keep, 0]).abs() * width
    score = (1.0 - dx / max(float(dist_thr_px), 1e-6)).clamp(min=0.0, max=1.0).mean()
    return float(score.item())


def match_candidates_to_gt(
    candidates: list[GCSLaneCandidate],
    gt_lanes: Any,
    gt_valid: Any,
    *,
    match_thr: float = 0.5,
    image_shape: tuple[int, int] = (544, 960),
    dist_thr_px: float = 20.0,
    min_overlap: int = 2,
) -> tuple[dict[int, int], dict[int, float]]:
    """Greedily match candidates to GT lanes with one-to-one GT assignment."""
    gt_points = torch.as_tensor(gt_lanes, dtype=torch.float32)
    gt_mask = torch.as_tensor(gt_valid, dtype=torch.float32)
    if gt_points.ndim != 3 or gt_points.shape[-1] != 2:
        raise ValueError(f"gt_lanes must have shape N x K x 2, got {tuple(gt_points.shape)}.")
    if gt_mask.shape != gt_points.shape[:2]:
        raise ValueError(f"gt_valid must have shape N x K, got {tuple(gt_mask.shape)} vs {tuple(gt_points.shape[:2])}.")

    scored: list[tuple[float, int, int]] = []
    best_score_by_candidate: dict[int, float] = {}
    for ci, cand in enumerate(candidates):
        best = 0.0
        for gi in range(gt_points.shape[0]):
            score = lane_similarity(
                cand.points,
                cand.valid_probs,
                gt_points[gi],
                gt_mask[gi],
                image_shape=image_shape,
                dist_thr_px=dist_thr_px,
                min_overlap=min_overlap,
            )
            best = max(best, score)
            if score >= float(match_thr):
                scored.append((score, ci, gi))
        best_score_by_candidate[ci] = best

    matched_candidates: dict[int, int] = {}
    matched_scores: dict[int, float] = {}
    used_gt: set[int] = set()
    for score, ci, gi in sorted(scored, key=lambda x: x[0], reverse=True):
        if ci in matched_candidates or gi in used_gt:
            continue
        matched_candidates[ci] = gi
        matched_scores[ci] = float(score)
        used_gt.add(gi)

    for ci, cand in enumerate(candidates):
        cand.matched_gt_id = int(matched_candidates.get(ci, -1))
        cand.line_iou = float(matched_scores.get(ci, best_score_by_candidate.get(ci, 0.0)))
        cand.official_lane_acc = cand.line_iou
    return matched_candidates, matched_scores
