"""Shared point-regression losses for GCS lane heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.gcs_shape import normalize_imgsz


def aspect_weighted_l1_point_loss(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    mask: torch.Tensor,
    image_size: tuple[int, int] | list[int] = (544, 960),
    y_weight: float = 1.0,
    x_only: bool = False,
) -> torch.Tensor:
    """Aspect-weighted L1 point loss on normalized xy coordinates."""
    mask = mask.to(device=pred_points.device, dtype=torch.bool)
    if not bool(mask.any()):
        return pred_points.sum() * 0.0

    h, w = normalize_imgsz(image_size)
    base = float(max(h, w))
    scale = pred_points.new_tensor((float(w) / base, float(h) / base))
    diff = (pred_points - target_points).abs() * scale
    if x_only:
        return diff[..., 0][mask].mean()
    loss = diff[..., 0] + float(y_weight) * diff[..., 1]
    return loss[mask].mean()


def pixel_smooth_l1_point_loss(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    mask: torch.Tensor,
    image_size: tuple[int, int] | list[int] = (544, 960),
    beta: float = 1.0,
    y_weight: float = 1.0,
    x_only: bool = False,
) -> torch.Tensor:
    """SmoothL1 point loss after converting normalized xy coordinates to pixels."""
    mask = mask.to(device=pred_points.device, dtype=torch.bool)
    if not bool(mask.any()):
        return pred_points.sum() * 0.0

    h, w = normalize_imgsz(image_size)
    scale = pred_points.new_tensor((float(w), float(h)))
    pred_px = pred_points * scale
    target_px = target_points * scale
    loss = F.smooth_l1_loss(pred_px, target_px, beta=float(beta), reduction="none")
    if x_only:
        return loss[..., 0][mask].mean()
    loss = loss[..., 0] + float(y_weight) * loss[..., 1]
    return loss[mask].mean()


def normalized_smooth_l1_point_loss(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    mask: torch.Tensor,
    image_size: tuple[int, int] | list[int] = (544, 960),
    y_weight: float = 1.0,
    x_only: bool = False,
) -> torch.Tensor:
    """Legacy ordered-slot SmoothL1 on normalized xy coordinates."""
    mask = mask.to(device=pred_points.device, dtype=torch.bool)
    if not bool(mask.any()):
        return pred_points.sum() * 0.0

    h, w = normalize_imgsz(image_size)
    base = float(max(h, w))
    scale = pred_points.new_tensor((float(w) / base, float(h) / base))
    loss = F.smooth_l1_loss(pred_points, target_points, reduction="none") * scale
    if x_only:
        return loss[..., 0][mask].mean()
    loss = loss[..., 0] + float(y_weight) * loss[..., 1]
    return loss[mask].mean()
