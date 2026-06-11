# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Hungarian matching for GCS-YOLO-Lane structured lane predictions."""

from __future__ import annotations

import torch

from ultralytics.utils.gcs_shape import normalize_imgsz

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:  # pragma: no cover
    linear_sum_assignment = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


class GCSHungarianMatcher:
    """Match unordered predicted lane queries to unordered GT lanes.

    The GCS-YOLO-Lane matching cost is the weighted sum of normalized point
    distance and existence confidence cost.
    The point term is weighted by the real input image aspect ratio so
    normalized x/y coordinates approximate pixel-space errors.
    """

    def __init__(
        self,
        cost_point: float = 5.0,
        cost_exist: float = 0.1,
        image_size=None,
        min_overlap: int = 2,
        max_x_dist: float = 0.0,
        match_gate_px: float = 0.0,
        point_mode: str = "free",
    ):
        """Initialize the matching cost weights used by GCS-YOLO-Lane."""
        self.cost_point = float(cost_point)
        self.cost_exist = float(cost_exist)
        self.point_scale = self._point_scale(image_size)
        self.pixel_scale = self._pixel_scale(image_size)
        self.min_overlap = max(int(min_overlap), 0)
        self.max_x_dist = float(max_x_dist)
        self.match_gate_px = float(match_gate_px)
        point_mode = str(point_mode).lower()
        self.point_mode = "fixed_y" if point_mode in {"fixed-y", "fixedy"} else point_mode

    @staticmethod
    def _point_scale(image_size) -> tuple[float, float]:
        """Return normalized pixel-aspect weights for x/y point distances."""
        if image_size is None or image_size == "":
            return 1.0, 1.0
        h, w = normalize_imgsz(image_size)
        base = float(max(h, w))
        return float(w) / base, float(h) / base

    @staticmethod
    def _pixel_scale(image_size) -> tuple[float, float]:
        """Return x/y pixel scales for geometry gates on normalized points."""
        if image_size is None or image_size == "":
            return 1.0, 1.0
        h, w = normalize_imgsz(image_size)
        return float(w), float(h)

    def _scale_tensor(self, ref: torch.Tensor, dims: int) -> torch.Tensor:
        """Create a broadcastable x/y scale tensor on the reference device."""
        shape = (1,) * (dims - 1) + (2,)
        return ref.new_tensor(self.point_scale).view(shape)

    @staticmethod
    def _empty_indices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an empty prediction/target index pair on the requested device."""
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    def _point_cost(self, pred_points: torch.Tensor, gt_points: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
        """Compute Q x N aspect-weighted normalized L1 point distance cost."""
        if self.point_mode == "fixed_y":
            scale_x = pred_points.new_tensor(float(self.point_scale[0]))
            point_dist = (pred_points[:, None, :, 0] - gt_points[None, :, :, 0]).abs() * scale_x * gt_valid[None]
            valid_count = gt_valid.sum(dim=1).clamp_min(1.0)
            return point_dist.sum(dim=2) / valid_count[None, :]

        scale = self._scale_tensor(pred_points, dims=4)
        point_dist = (pred_points[:, None] - gt_points[None]).abs() * scale * gt_valid[None, :, :, None]
        valid_count = gt_valid.sum(dim=1).clamp_min(1.0)
        return point_dist.sum(dim=(2, 3)) / valid_count[None, :]

    def _gate_mask(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        gt_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q x N finite-match mask and GT columns whose geometry gate was relaxed."""
        valid_count = gt_valid.sum(dim=1)
        base_gate = torch.ones((pred_points.shape[0], gt_points.shape[0]), dtype=torch.bool, device=pred_points.device)
        if self.min_overlap > 0:
            base_gate = base_gate & (valid_count[None, :] >= int(self.min_overlap))
        gate = base_gate.clone()

        pixel_scale = pred_points.new_tensor(self.pixel_scale).view(1, 1, 1, 2)
        diff_px = (pred_points[:, None] - gt_points[None]) * pixel_scale
        valid = gt_valid[None, :, :, None]

        if self.max_x_dist > 0.0:
            x_dist = diff_px[..., 0].abs() * valid[..., 0]
            mean_x = x_dist.sum(dim=2) / valid_count[None, :].clamp_min(1.0)
            gate = gate & (mean_x <= float(self.max_x_dist))

        if self.match_gate_px > 0.0:
            if self.point_mode == "fixed_y":
                point_error = diff_px[..., 0].abs() * gt_valid[None]
            else:
                point_error = torch.norm(diff_px, dim=-1) * gt_valid[None]
            ape = point_error.sum(dim=2) / valid_count[None, :].clamp_min(1.0)
            gate = gate & (ape <= float(self.match_gate_px))

        eligible_gt = base_gate.any(dim=0)
        relaxed_gt = eligible_gt & ~gate.any(dim=0)
        if relaxed_gt.any():
            # Keep the label-validity gate, but do not let prediction-dependent geometry gates remove
            # every candidate for a GT lane. Otherwise the image becomes all-negative for structured losses.
            gate[:, relaxed_gt] = base_gate[:, relaxed_gt]

        return gate, relaxed_gt

    def cost_matrix(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_points: torch.Tensor,
        gt_valid: torch.Tensor,
        return_relaxed_gt: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Build the Q x N Hungarian cost matrix for one image."""
        cost_point = self._point_cost(pred_points, gt_points, gt_valid)
        cost_exist = -pred_logits.sigmoid()[:, None].expand_as(cost_point)
        cost = self.cost_point * cost_point + self.cost_exist * cost_exist
        gate, relaxed_gt = self._gate_mask(pred_points, gt_points, gt_valid)
        cost = cost.masked_fill(~gate, torch.inf)
        return (cost, relaxed_gt) if return_relaxed_gt else cost

    @torch.no_grad()
    def __call__(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_points: list[torch.Tensor] | tuple[torch.Tensor, ...],
        gt_valid: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return matched prediction and GT indices for each image.

        Args:
            pred_points: Predicted normalized lane points with shape B x Q x K x 2.
            pred_logits: Predicted lane existence logits with shape B x Q.
            gt_points: Per-image GT normalized lane points, each N_i x K x 2.
            gt_valid: Per-image GT point validity masks, each N_i x K.
        """
        if linear_sum_assignment is None:
            raise ImportError("scipy is required for GCS Hungarian matching.") from _SCIPY_IMPORT_ERROR
        if pred_points.ndim != 4 or pred_points.shape[-1] != 2:
            raise ValueError(f"pred_points must have shape B x Q x K x 2, got {tuple(pred_points.shape)}.")
        if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
            pred_logits = pred_logits.squeeze(-1)
        if pred_logits.ndim != 2:
            raise ValueError(f"pred_logits must have shape B x Q, got {tuple(pred_logits.shape)}.")
        if pred_points.shape[:2] != pred_logits.shape:
            raise ValueError(
                f"pred_points B,Q must match pred_logits, got {tuple(pred_points.shape[:2])} vs {tuple(pred_logits.shape)}."
            )
        if len(gt_points) != pred_points.shape[0] or len(gt_valid) != pred_points.shape[0]:
            raise ValueError("gt_points and gt_valid must contain one tensor per batch image.")

        device = pred_points.device
        dtype = pred_points.dtype
        indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        stats = {
            "images": 0,
            "images_with_gt": 0,
            "gt_lanes": 0,
            "matched_gt_lanes": 0,
            "no_match_images": 0,
            "relaxed_gt_lanes": 0,
        }

        for b in range(pred_points.shape[0]):
            stats["images"] += 1
            pp = pred_points[b]
            pl = pred_logits[b]
            gp = gt_points[b].to(device=device, dtype=dtype)
            gv = gt_valid[b].to(device=device, dtype=dtype)

            if gp.numel() == 0:
                indices.append(self._empty_indices(device))
                continue
            if gp.ndim != 3 or gp.shape[-1] != 2:
                raise ValueError(f"Each GT lane tensor must have shape N x K x 2, got {tuple(gp.shape)}.")
            if gv.shape != gp.shape[:2]:
                raise ValueError(f"GT valid mask must match GT lane first two dims, got {tuple(gv.shape)} vs {tuple(gp.shape[:2])}.")

            valid_lane = gv.sum(dim=1) >= 2
            if not valid_lane.any():
                indices.append(self._empty_indices(device))
                continue

            original_cols = torch.arange(gp.shape[0], device=device)[valid_lane]
            gp = gp[valid_lane]
            gv = gv[valid_lane]
            stats["images_with_gt"] += 1
            stats["gt_lanes"] += int(gp.shape[0])

            cost, relaxed_gt = self.cost_matrix(pp, pl, gp, gv, return_relaxed_gt=True)
            stats["relaxed_gt_lanes"] += int(relaxed_gt.sum().item())
            finite = torch.isfinite(cost)
            if not finite.any():
                stats["no_match_images"] += 1
                indices.append(self._empty_indices(device))
                continue
            large_cost = torch.nan_to_num(cost, posinf=1e9, neginf=1e9).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(large_cost)
            rows_all = torch.as_tensor(row_ind, dtype=torch.long, device=device)
            cols_all = torch.as_tensor(col_ind, dtype=torch.long, device=device)
            keep = finite[rows_all, cols_all]
            rows = rows_all[keep]
            cols = original_cols[cols_all[keep]]
            matched = int(cols.numel())
            stats["matched_gt_lanes"] += matched
            if matched == 0:
                stats["no_match_images"] += 1
            indices.append((rows, cols))

        if int(stats["gt_lanes"]) > 0:
            gt_lanes = int(stats["gt_lanes"])
            images_with_gt = max(int(stats["images_with_gt"]), 1)
            stats["matched_gt_ratio"] = float(stats["matched_gt_lanes"]) / gt_lanes
            stats["no_match_image_rate"] = float(stats["no_match_images"]) / images_with_gt
            stats["relaxed_gt_ratio"] = float(stats["relaxed_gt_lanes"]) / gt_lanes
        else:
            stats["matched_gt_ratio"] = 1.0
            stats["no_match_image_rate"] = 0.0
            stats["relaxed_gt_ratio"] = 0.0
        self.last_stats = stats
        return indices
