# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Five-term query loss for GCS-YOLO-Lane structured lane predictions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.gcs_matcher import GCSHungarianMatcher
from ultralytics.utils.gcs_point_loss import aspect_weighted_l1_point_loss
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, normalize_imgsz


class GCSLoss(nn.Module):
    """Query-mode GCS lane loss with only existence, points, validity, curve, and visible line-IoU terms."""

    loss_names = (
        "exist_loss",
        "point_loss",
        "point_valid_loss",
        "curve_loss",
        "visible_line_iou_loss",
    )

    def __init__(self, model: Any = None, **kwargs: Any):
        """Initialize the five-term query loss from a model, args namespace, or plain dict."""
        super().__init__()
        # Keep a plain reference instead of registering the model as a child module.
        # GCSLaneModel already stores this loss as `criterion`; a normal Module
        # attribute would create a model -> criterion -> model cycle and break
        # state_dict()/EMA updates during training.
        object.__setattr__(self, "source", model)
        self.extra_args = dict(kwargs)
        self.image_size = normalize_imgsz(
            self._arg("gcs_imgsz", self._arg("image_shape", self._arg("imgsz", None))),
            dataset=self._arg("dataset", None),
        )

        self.exist_gain = float(self._arg("gcs_exist", 2.0) or 0.0)
        self.point_gain = float(self._arg("gcs_point", 15.0) or 0.0)
        self.point_valid_gain = float(self._arg("gcs_point_valid", 1.0) or 0.0)
        self.curve_gain = float(self._arg("gcs_curve", 0.1) or 0.0)
        self.line_iou_gain = float(self._arg("gcs_line_iou", 1.0) or 0.0)

        self.line_iou_width_px = float(self._arg("gcs_line_iou_width_px", 18.0) or 18.0)
        self.line_iou_temperature_px = float(self._arg("gcs_line_iou_temperature_px", 1.0) or 1.0)
        self.curve_alpha = float(self._arg("gcs_curve_alpha", 5.0) or 0.0)
        self.curve_weight_max = float(self._arg("gcs_curve_weight_max", 5.0) or 1.0)
        self.point_valid_unmatched_ignore = self._as_bool(self._arg("gcs_point_valid_unmatched_ignore", True), True)
        self.point_valid_unmatched_ignore_px = float(self._arg("gcs_point_valid_unmatched_ignore_px", 30.0) or 0.0)
        anchor_ignore_px = self._arg("gcs_point_valid_unmatched_ignore_anchor_px", None)
        self.point_valid_unmatched_ignore_anchor_px = (
            self.point_valid_unmatched_ignore_px if anchor_ignore_px is None else float(anchor_ignore_px or 0.0)
        )
        self.point_valid_unmatched_ignore_min_overlap = max(
            int(self._arg("gcs_point_valid_unmatched_ignore_min_overlap", 3) or 0),
            0,
        )

        self.matcher = GCSHungarianMatcher(
            cost_point=float(self._arg("gcs_cost_point", 5.0) or 0.0),
            cost_curve=float(self._arg("gcs_cost_curve", 0.05) or 0.0),
            cost_exist=float(self._arg("gcs_cost_exist", 0.1) or 0.0),
            image_size=self.image_size,
            min_overlap=int(self._arg("gcs_match_min_overlap", 2) or 0),
            max_x_dist=float(self._arg("gcs_match_max_x_dist", 0.0) or 0.0),
            match_gate_px=float(self._arg("gcs_match_gate_px", 160.0) or 0.0),
        )

    def _arg(self, name: str, default: Any = None) -> Any:
        """Resolve an argument from kwargs, dicts, namespaces, model args, or model yaml."""
        if name in self.extra_args:
            value = self.extra_args[name]
            return default if value is None else value
        for obj in (self.source, getattr(self.source, "args", None), getattr(self.source, "yaml", None)):
            if obj is None:
                continue
            if isinstance(obj, dict) and name in obj:
                value = obj[name]
                return default if value is None else value
            if hasattr(obj, name):
                value = getattr(obj, name)
                return default if value is None else value
        return default

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        """Parse bool-like config values without treating the string 'False' as truthy."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        return bool(value)

    @staticmethod
    def _require_preds(preds: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return required query tensors and validate their basic shape contract."""
        if not isinstance(preds, dict):
            raise TypeError(f"GCSLoss expects prediction dict output from GCSLaneHead, got {type(preds).__name__}.")
        missing = [name for name in ("pred_points", "pred_logits", "pred_valid_logits") if name not in preds]
        if missing:
            raise RuntimeError(f"GCSLoss missing required prediction tensor(s): {', '.join(missing)}.")

        pred_points = preds["pred_points"]
        pred_logits = preds["pred_logits"]
        pred_valid_logits = preds["pred_valid_logits"]
        if pred_valid_logits is None:
            raise RuntimeError("GCSLoss requires preds['pred_valid_logits']; point-valid supervision is part of the five-loss contract.")

        if pred_points.ndim != 4 or pred_points.shape[-1] != 2:
            raise ValueError(f"pred_points must have shape B x Q x K x 2, got {tuple(pred_points.shape)}.")
        if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
            pred_logits = pred_logits.squeeze(-1)
        if pred_logits.ndim != 2:
            raise ValueError(f"pred_logits must have shape B x Q, got {tuple(pred_logits.shape)}.")
        if pred_valid_logits.shape != pred_points.shape[:3]:
            raise ValueError(
                "pred_valid_logits must have shape B x Q x K matching pred_points, "
                f"got {tuple(pred_valid_logits.shape)} vs {tuple(pred_points.shape[:3])}."
            )
        if pred_points.shape[:2] != pred_logits.shape:
            raise ValueError(
                f"pred_points B,Q must match pred_logits, got {tuple(pred_points.shape[:2])} "
                f"vs {tuple(pred_logits.shape)}."
            )
        return pred_points, pred_logits, pred_valid_logits

    @staticmethod
    def _targets(batch: dict[str, Any], device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Extract per-image GT lanes and visible-anchor masks from a collated GCS batch."""
        gt_points = batch.get("lanes", batch.get("gt_lanes"))
        gt_valid = batch.get("lane_valid", batch.get("gt_lane_valid"))
        if gt_points is None or gt_valid is None:
            raise KeyError("GCSLoss requires batch['lanes']/['lane_valid'] or batch['gt_lanes']/['gt_lane_valid'].")

        if isinstance(gt_points, torch.Tensor):
            gt_points = [gt_points[i] for i in range(gt_points.shape[0])]
        if isinstance(gt_valid, torch.Tensor):
            gt_valid = [gt_valid[i] for i in range(gt_valid.shape[0])]
        if len(gt_points) != len(gt_valid):
            raise ValueError(f"GT lanes and lane_valid length mismatch: {len(gt_points)} vs {len(gt_valid)}.")
        return [x.to(device=device) for x in gt_points], [x.to(device=device) for x in gt_valid]

    @staticmethod
    def _zero_like(tensor: torch.Tensor) -> torch.Tensor:
        """Return a finite scalar zero connected to the provided tensor graph."""
        return tensor.sum() * 0.0

    def _matched_lane_tensors(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather matched prediction lanes, target lanes, and GT visible masks."""
        pred_chunks: list[torch.Tensor] = []
        gt_chunks: list[torch.Tensor] = []
        valid_chunks: list[torch.Tensor] = []
        device = pred_points.device
        dtype = pred_points.dtype

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            pred_chunks.append(pred_points[b, src_idx])
            gt_chunks.append(gt_points[b].to(device=device, dtype=dtype)[tgt_idx])
            valid_chunks.append(gt_valid[b].to(device=device, dtype=torch.bool)[tgt_idx])

        if not pred_chunks:
            empty_points = pred_points.new_zeros((0, pred_points.shape[2], 2))
            empty_valid = torch.zeros((0, pred_points.shape[2]), dtype=torch.bool, device=device)
            return empty_points, empty_points.clone(), empty_valid

        return torch.cat(pred_chunks, dim=0), torch.cat(gt_chunks, dim=0), torch.cat(valid_chunks, dim=0)

    def exist_loss(
        self,
        pred_logits: torch.Tensor,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Hard matched-query BCE: Hungarian-matched queries are 1, all remaining queries are 0."""
        if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
            pred_logits = pred_logits.squeeze(-1)
        target = torch.zeros_like(pred_logits, dtype=torch.float32)
        for b, (src_idx, _) in enumerate(indices):
            if src_idx.numel():
                target[b, src_idx.to(device=pred_logits.device, dtype=torch.long)] = 1.0
        return F.binary_cross_entropy_with_logits(pred_logits.float(), target, reduction="mean")

    def point_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Aspect-weighted L1 point regression on GT-visible anchors of matched lanes."""
        pred_match, gt_match, valid_match = self._matched_lane_tensors(pred_points, gt_points, gt_valid, indices)
        if pred_match.numel() == 0:
            return self._zero_like(pred_points)
        return aspect_weighted_l1_point_loss(
            pred_match.float(),
            gt_match.float(),
            valid_match,
            image_size=self.image_size,
        )

    def point_valid_loss(
        self,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Per-anchor visibility BCE while ignoring GT-close unmatched negative anchors."""
        target = torch.zeros_like(pred_valid_logits, dtype=torch.float32)
        device = pred_valid_logits.device
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            target[b, src_idx] = gt_valid[b].to(device=device, dtype=torch.float32)[tgt_idx]
        loss = F.binary_cross_entropy_with_logits(pred_valid_logits.float(), target, reduction="none")
        weight = self._point_valid_weight_mask(pred_points, gt_points, gt_valid, indices)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    @torch.no_grad()
    def _point_valid_weight_mask(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Build point-valid BCE weights, zeroing GT-close unmatched visible anchors only."""
        weights = torch.ones(pred_points.shape[:3], dtype=torch.float32, device=pred_points.device)
        if (
            not self.point_valid_unmatched_ignore
            or self.point_valid_unmatched_ignore_px <= 0.0
            or self.point_valid_unmatched_ignore_min_overlap <= 0
        ):
            return weights

        _, w = self.image_size
        q, k = pred_points.shape[1:3]
        mean_thr = float(self.point_valid_unmatched_ignore_px)
        anchor_thr = float(self.point_valid_unmatched_ignore_anchor_px)
        min_overlap = int(self.point_valid_unmatched_ignore_min_overlap)
        pred_points_detached = pred_points.detach()

        for b, (src_idx, _) in enumerate(indices):
            gp = gt_points[b].to(device=pred_points.device, dtype=pred_points_detached.dtype)
            gv = gt_valid[b].to(device=pred_points.device, dtype=torch.bool)
            if gp.numel() == 0 or gv.numel() == 0:
                continue
            if gp.ndim != 3 or gp.shape[-1] != 2:
                raise ValueError(f"Each GT lane tensor must have shape N x K x 2, got {tuple(gp.shape)}.")
            if gv.shape != gp.shape[:2]:
                raise ValueError(f"GT valid mask must match GT lane first two dims, got {tuple(gv.shape)} vs {tuple(gp.shape[:2])}.")

            visible_count = gv.sum(dim=1)
            keep_gt = visible_count >= min_overlap
            if not bool(keep_gt.any()):
                continue
            gp = gp[keep_gt]
            gv = gv[keep_gt]
            visible_count = visible_count[keep_gt]

            matched = torch.zeros((q,), dtype=torch.bool, device=pred_points.device)
            if src_idx.numel():
                matched[src_idx.to(device=pred_points.device, dtype=torch.long)] = True
            unmatched = ~matched
            if not bool(unmatched.any()):
                continue

            diff_x_px = (pred_points_detached[b, :, None, :, 0].float() - gp[None, :, :, 0].float()).abs() * float(w)
            valid_pair = gv[None].expand(q, -1, k)
            mean_x = (diff_x_px * valid_pair.float()).sum(dim=2) / visible_count[None].clamp_min(1).float()
            close_pair = unmatched[:, None] & (mean_x <= mean_thr)
            if not bool(close_pair.any()):
                continue

            close_anchor = (close_pair[:, :, None] & valid_pair & (diff_x_px <= anchor_thr)).any(dim=1)
            weights[b] = weights[b].masked_fill(close_anchor, 0.0)
        return weights

    def curve_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """GT-visible second-order curvature SmoothL1 on matched lanes in pixel space."""
        pred_match, gt_match, valid_match = self._matched_lane_tensors(pred_points, gt_points, gt_valid, indices)
        if pred_match.numel() == 0 or pred_match.shape[1] < 3:
            return self._zero_like(pred_points)

        h, w = self.image_size
        scale = pred_match.new_tensor((float(w), float(h))).view(1, 1, 2)
        pred_px = pred_match.float() * scale
        gt_px = gt_match.float() * scale
        pred_lap = pred_px[:, 2:] - 2.0 * pred_px[:, 1:-1] + pred_px[:, :-2]
        gt_lap = gt_px[:, 2:] - 2.0 * gt_px[:, 1:-1] + gt_px[:, :-2]
        curve_valid = valid_match[:, 2:] & valid_match[:, 1:-1] & valid_match[:, :-2]
        if not bool(curve_valid.any()):
            return self._zero_like(pred_points)

        loss = F.smooth_l1_loss(pred_lap, gt_lap, beta=1.0, reduction="none").sum(dim=-1)
        if self.curve_alpha > 0.0:
            gt_curve_mag = torch.linalg.vector_norm(gt_lap.detach(), dim=-1)
            weight = (1.0 + self.curve_alpha * gt_curve_mag).clamp(max=max(self.curve_weight_max, 1.0))
            loss = loss * weight
        return loss[curve_valid].mean()

    def visible_line_iou_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Differentiable visible lane-region IoU surrogate on GT-visible anchors of matched lanes."""
        pred_match, gt_match, valid_match = self._matched_lane_tensors(pred_points, gt_points, gt_valid, indices)
        if pred_match.numel() == 0:
            return self._zero_like(pred_points)

        h, w = self.image_size
        scale = pred_match.new_tensor((float(w), float(h))).view(1, 1, 2)
        diff_px = (pred_match.float() - gt_match.float()) * scale
        dist = torch.linalg.vector_norm(diff_px, dim=-1)
        temperature = max(float(self.line_iou_temperature_px), 1e-6)
        dist = torch.sqrt(dist.square() + temperature * temperature) - temperature

        width = max(float(self.line_iou_width_px), 1e-6)
        overlap = torch.clamp(width - dist, min=0.0)
        union = (width + dist).clamp_min(1e-6)
        loss = 1.0 - overlap / union

        valid = valid_match.to(dtype=loss.dtype)
        valid_count = valid.sum(dim=1)
        lane_loss = (loss * valid).sum(dim=1) / valid_count.clamp_min(1.0)
        has_visible = valid_count > 0
        if not bool(has_visible.any()):
            return self._zero_like(pred_points)
        return lane_loss[has_visible].mean()

    def forward(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the weighted five-term query loss and return detached raw loss items."""
        pred_points, pred_logits, pred_valid_logits = self._require_preds(preds)
        if "img" in batch:
            assert_gcs_image_tensor(batch["img"], self.image_size, name="img", context="GCSLoss.forward")
        gt_points, gt_valid = self._targets(batch, device=pred_points.device)
        if len(gt_points) != pred_points.shape[0]:
            raise ValueError(f"Batch target count {len(gt_points)} does not match predictions B={pred_points.shape[0]}.")

        indices = self.matcher(pred_points, pred_logits, gt_points, gt_valid)

        exist_loss = self.exist_loss(pred_logits, indices)
        point_loss = self.point_loss(pred_points, gt_points, gt_valid, indices)
        point_valid_loss = self.point_valid_loss(pred_points, pred_valid_logits, gt_points, gt_valid, indices)
        curve_loss = self.curve_loss(pred_points, gt_points, gt_valid, indices)
        visible_line_iou_loss = self.visible_line_iou_loss(pred_points, gt_points, gt_valid, indices)

        total = (
            self.exist_gain * exist_loss
            + self.point_gain * point_loss
            + self.point_valid_gain * point_valid_loss
            + self.curve_gain * curve_loss
            + self.line_iou_gain * visible_line_iou_loss
        )
        loss_items = torch.stack(
            (
                exist_loss.detach(),
                point_loss.detach(),
                point_valid_loss.detach(),
                curve_loss.detach(),
                visible_line_iou_loss.detach(),
            )
        )
        return total, loss_items
