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
    distance, second-order curvature distance, and existence confidence cost.
    Point and curvature terms are weighted by the real input image aspect ratio
    so normalized x/y coordinates approximate pixel-space errors.
    """

    def __init__(
        self,
        cost_point: float = 5.0,
        cost_curve: float = 0.05,
        cost_exist: float = 0.1,
        image_size=None,
        min_overlap: int = 2,
        max_x_dist: float = 0.0,
        match_gate_px: float = 0.0,
        gt4_short_match_endpoint: float = 0.0,
        gt4_short_match_max_points: int = 20,
    ):
        """Initialize the three matching cost weights used by GCS-YOLO-Lane."""
        self.cost_point = float(cost_point)
        self.cost_curve = float(cost_curve)
        self.cost_exist = float(cost_exist)
        self.point_scale = self._point_scale(image_size)
        self.pixel_scale = self._pixel_scale(image_size)
        self.min_overlap = max(int(min_overlap), 0)
        self.max_x_dist = float(max_x_dist)
        self.match_gate_px = float(match_gate_px)
        self.gt4_short_match_endpoint = float(gt4_short_match_endpoint)
        self.gt4_short_match_max_points = max(int(gt4_short_match_max_points), 1)
        if self.gt4_short_match_endpoint < 0.0:
            raise ValueError(
                f"gcs_gt4_short_match_endpoint must be >= 0, got {self.gt4_short_match_endpoint}."
            )

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
        """Return x/y pixel scales for curvature matching on normalized points."""
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
        scale = self._scale_tensor(pred_points, dims=4)
        point_dist = (pred_points[:, None] - gt_points[None]).abs() * scale * gt_valid[None, :, :, None]
        valid_count = gt_valid.sum(dim=1).clamp_min(1.0)
        return point_dist.sum(dim=(2, 3)) / valid_count[None, :]

    def _gt4_short_endpoint_cost(
        self, pred_points: torch.Tensor, gt_points: torch.Tensor, gt_valid: torch.Tensor
    ) -> torch.Tensor:
        """Return Q x N normalized-x endpoint cost for short GT lanes in GT4 images."""
        if self.gt4_short_match_endpoint <= 0.0 or gt_points.shape[0] != 4:
            return pred_points.new_zeros((pred_points.shape[0], gt_points.shape[0]))

        valid_count = gt_valid.sum(dim=1)
        short_mask = valid_count <= float(self.gt4_short_match_max_points)
        if not short_mask.any():
            return pred_points.new_zeros((pred_points.shape[0], gt_points.shape[0]))

        # GCS fixed-y x coordinates are normalized to 0..1 in labels and head output,
        # so endpoint matching must not divide by image width again.
        pred_x = pred_points[..., 0]
        gt_x = gt_points[..., 0]
        costs = []
        for gt_idx in range(gt_points.shape[0]):
            if not bool(short_mask[gt_idx].item()):
                costs.append(pred_x.new_zeros(pred_x.shape[0]))
                continue
            valid_idx = torch.nonzero(gt_valid[gt_idx] > 0.5, as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                costs.append(pred_x.new_zeros(pred_x.shape[0]))
                continue
            first = valid_idx[0]
            last = valid_idx[-1]
            endpoint_error = (pred_x[:, first] - gt_x[gt_idx, first]).abs()
            endpoint_error = endpoint_error + (pred_x[:, last] - gt_x[gt_idx, last]).abs()
            costs.append(endpoint_error * 0.5)
        return torch.stack(costs, dim=1)

    def _curve_cost(self, pred_points: torch.Tensor, gt_points: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
        """Compute Q x N aspect-weighted L1 second-order curvature cost."""
        if pred_points.shape[1] < 3:
            return pred_points.new_zeros((pred_points.shape[0], gt_points.shape[0]))

        pred_curve = pred_points[:, 2:] - 2.0 * pred_points[:, 1:-1] + pred_points[:, :-2]
        gt_curve = gt_points[:, 2:] - 2.0 * gt_points[:, 1:-1] + gt_points[:, :-2]
        curve_valid = gt_valid[:, 2:] * gt_valid[:, 1:-1] * gt_valid[:, :-2]

        scale = pred_points.new_tensor(getattr(self, "pixel_scale", (1.0, 1.0))).view(1, 1, 1, 2)
        curve_dist = (pred_curve[:, None] - gt_curve[None]).abs() * scale * curve_valid[None, :, :, None]
        curve_count = curve_valid.sum(dim=1).clamp_min(1.0)
        return curve_dist.sum(dim=(2, 3)) / curve_count[None, :]

    def _gate_mask(self, pred_points: torch.Tensor, gt_points: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
        """Return Q x N finite-match mask for optional training-time geometry gates."""
        valid_count = gt_valid.sum(dim=1)
        gate = torch.ones((pred_points.shape[0], gt_points.shape[0]), dtype=torch.bool, device=pred_points.device)
        if self.min_overlap > 0:
            gate = gate & (valid_count[None, :] >= int(self.min_overlap))

        pixel_scale = pred_points.new_tensor(getattr(self, "pixel_scale", (1.0, 1.0))).view(1, 1, 1, 2)
        diff_px = (pred_points[:, None] - gt_points[None]) * pixel_scale
        valid = gt_valid[None, :, :, None]

        if self.max_x_dist > 0.0:
            x_dist = diff_px[..., 0].abs() * valid[..., 0]
            mean_x = x_dist.sum(dim=2) / valid_count[None, :].clamp_min(1.0)
            gate = gate & (mean_x <= float(self.max_x_dist))

        if self.match_gate_px > 0.0:
            point_error = torch.norm(diff_px, dim=-1) * gt_valid[None]
            ape = point_error.sum(dim=2) / valid_count[None, :].clamp_min(1.0)
            gate = gate & (ape <= float(self.match_gate_px))

        return gate

    def cost_matrix(self, pred_points: torch.Tensor, pred_logits: torch.Tensor, gt_points: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
        """Build the Q x N Hungarian cost matrix for one image."""
        cost_point = self._point_cost(pred_points, gt_points, gt_valid)
        endpoint_cost = self._gt4_short_endpoint_cost(pred_points, gt_points, gt_valid)
        cost_curve = self._curve_cost(pred_points, gt_points, gt_valid)
        cost_exist = -pred_logits.sigmoid()[:, None].expand_as(cost_point)
        cost = (
            self.cost_point * cost_point
            + self.gt4_short_match_endpoint * endpoint_cost
            + self.cost_curve * cost_curve
            + self.cost_exist * cost_exist
        )
        gate = self._gate_mask(pred_points, gt_points, gt_valid)
        return cost.masked_fill(~gate, torch.inf)

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

        for b in range(pred_points.shape[0]):
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

            cost = self.cost_matrix(pp, pl, gp, gv)
            finite = torch.isfinite(cost)
            if not finite.any():
                indices.append(self._empty_indices(device))
                continue
            large_cost = torch.nan_to_num(cost, posinf=1e9, neginf=1e9).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(large_cost)
            rows_all = torch.as_tensor(row_ind, dtype=torch.long, device=device)
            cols_all = torch.as_tensor(col_ind, dtype=torch.long, device=device)
            keep = finite[rows_all, cols_all]
            rows = rows_all[keep]
            cols = original_cols[cols_all[keep]]
            indices.append((rows, cols))

        return indices
