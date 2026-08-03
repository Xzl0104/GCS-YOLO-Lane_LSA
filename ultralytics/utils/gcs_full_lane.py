"""Shared score and visibility helpers for full-lane proposals."""

from __future__ import annotations

import torch


def full_lane_interval_bounds(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ordered visible-interval indices from start/end logits."""
    if start_logits.shape != end_logits.shape or start_logits.ndim < 1:
        raise ValueError(
            "Full-lane interval logits must have matching shape [..., K], "
            f"got {tuple(start_logits.shape)} and {tuple(end_logits.shape)}."
        )
    start = start_logits.argmax(dim=-1)
    end = end_logits.argmax(dim=-1)
    return torch.minimum(start, end), torch.maximum(start, end)


def full_lane_interval_mask(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
) -> torch.Tensor:
    """Return a hard contiguous visible-interval mask with shape [..., K]."""
    start, end = full_lane_interval_bounds(start_logits, end_logits)
    point_idx = torch.arange(start_logits.shape[-1], device=start_logits.device)
    point_idx = point_idx.view(*([1] * start.ndim), -1)
    return (point_idx >= start.unsqueeze(-1)) & (point_idx <= end.unsqueeze(-1))


def full_lane_proposal_score_probability(
    exist_logits: torch.Tensor,
    quality_logits: torch.Tensor | None,
    valid_logits: torch.Tensor | None,
    start_logits: torch.Tensor | None = None,
    end_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the single full-lane proposal score used by matching and decode.

    The score is the product of existence probability, geometry-quality
    probability, and predicted visible probability.  Base queries can pass
    ``quality_logits=None`` to use an implicit quality probability of one.
    When an interval is available, visibility is averaged only inside the
    predicted contiguous interval.
    """
    if exist_logits.ndim < 1:
        raise ValueError(f"exist_logits must have at least one dimension, got {tuple(exist_logits.shape)}.")
    if quality_logits is not None and quality_logits.shape != exist_logits.shape:
        raise ValueError(
            f"quality_logits must match exist_logits, got {tuple(quality_logits.shape)} vs "
            f"{tuple(exist_logits.shape)}."
        )
    if valid_logits is None:
        visible_probability = torch.ones_like(exist_logits)
    else:
        if valid_logits.shape[:-1] != exist_logits.shape:
            raise ValueError(
                "valid_logits must have shape [..., K] with leading dimensions matching exist_logits, "
                f"got {tuple(valid_logits.shape)} vs {tuple(exist_logits.shape)}."
            )
        valid_probability = valid_logits.sigmoid()
        if start_logits is not None or end_logits is not None:
            if start_logits is None or end_logits is None:
                raise ValueError("start_logits and end_logits must be provided together.")
            interval = full_lane_interval_mask(start_logits, end_logits).to(dtype=valid_probability.dtype)
            visible_probability = (valid_probability * interval).sum(dim=-1) / interval.sum(dim=-1).clamp_min(1.0)
        else:
            visible_probability = valid_probability.mean(dim=-1)

    exist_probability = exist_logits.sigmoid()
    quality_probability = torch.ones_like(exist_probability) if quality_logits is None else quality_logits.sigmoid()
    return (exist_probability * quality_probability * visible_probability).clamp(0.0, 1.0)


def probability_to_logit(probability: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert a probability tensor into a finite logit for matcher costs."""
    return torch.logit(probability.clamp(float(eps), 1.0 - float(eps)))
