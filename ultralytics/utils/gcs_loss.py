# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions for GCS-YOLO-Lane structured lane training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.gcs_matcher import GCSHungarianMatcher
from ultralytics.utils.gcs_point_loss import aspect_weighted_l1_point_loss
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, assert_gcs_shape, normalize_imgsz


class GCSLoss(nn.Module):
    """Hungarian-matched loss for query-based structured lane predictions."""

    loss_names = (
        "exist_loss",
        "point_loss",
        "point_valid_loss",
        "smooth_loss",
        "curve_loss",
        "mask_loss",
        "edge_loss",
        "count_loss",
        "count_under5_loss",
        "count_boundary_loss",
        "short_side_geom_loss",
        "short_side_geom_count",
        "short_side_geom_gt4",
        "short_side_geom_gt5",
        "spurious_neg_loss",
        "spurious_negative_count",
        "spur_cand",
        "spur_prot",
        "spur_final",
        "spur_neg",
        "spur_cnt_gt3",
        "spur_cnt_gt4",
        "spur_cnt_gt5",
        "spur_neg_gt3",
        "spur_neg_gt4",
        "spur_neg_gt5",
        "count_score_mean",
        "gt5_short_pos_count",
        "gt5_short_pos_anchor_count",
        "gt5_short_point_valid_loss",
        "cnt_bound_5under",
        "cnt_score",
    )

    def __init__(
        self,
        model=None,
        lambda_exist: float | None = None,
        lambda_point: float | None = None,
        lambda_point_valid: float | None = None,
        lambda_smooth: float | None = None,
        lambda_curve: float | None = None,
        lambda_mask: float | None = None,
        lambda_edge: float | None = None,
        lambda_count: float | None = None,
        lambda_count_under5: float | None = None,
        lambda_count_boundary: float | None = None,
        lambda_short_side_geom: float | None = None,
        lambda_spurious_neg: float | None = None,
        count_under5_min_lanes: int | None = None,
        count_boundary_gt4_weight: float | None = None,
        count_boundary_gt5_weight: float | None = None,
        count_boundary_gt5_under_weight: float | None = None,
        count_boundary_margin34: float | None = None,
        count_boundary_margin45: float | None = None,
        short_side_geom_min_gt_lanes: int | None = None,
        short_side_geom_visible_max: int | None = None,
        short_side_geom_side_only: bool | None = None,
        short_side_geom_weight_gt4: float | None = None,
        short_side_geom_weight_gt5: float | None = None,
        spurious_neg_weight: float | None = None,
        spurious_gt3_weight: float | None = None,
        spurious_gt4_weight: float | None = None,
        spurious_gt5_weight: float | None = None,
        spurious_disable_gt5: bool | None = None,
        spurious_max_points: int | None = None,
        spurious_close_px: float | None = None,
        spurious_min_overlap: int | None = None,
        spurious_gt_protect: bool | None = None,
        spurious_gt_protect_px: float | None = None,
        spurious_gt_protect_min_overlap: int | None = None,
        spurious_gt_protect_min_gt_lanes: int | None = None,
        spurious_gt_protect_margin_px: float | None = None,
        spurious_gt_protect_mode: str | None = None,
        eval_point_valid_thr: float | None = None,
        gt5_short_visible_thr: int | None = None,
        gt5_short_point_valid_weight: float | None = None,
        curve_alpha: float | None = None,
        curve_weight_max: float | None = None,
        exist_pos_weight: float | None = None,
        exist_focal_gamma: float | None = None,
        exist_focal_alpha: float | None = None,
        point_valid_pos_weight_max: float | None = None,
        mask_pos_weight_max: float | None = None,
        edge_pos_weight_max: float | None = None,
        aux_dice_gain: float | None = None,
        exist_quality_alpha: float | None = None,
        exist_quality_mode: str | None = None,
        exist_quality_tau: float | None = None,
        exist_quality_floor: float | None = None,
        exist_quality_pos_px: float | None = None,
        exist_quality_neg_px: float | None = None,
        match_min_overlap: int | None = None,
        match_max_x_dist: float | None = None,
        match_gate_px: float | None = None,
        image_size=None,
    ):
        """Initialize GCS-YOLO-Lane loss weights and Hungarian matcher."""
        super().__init__()
        args = model if isinstance(model, dict) else getattr(model, "args", None)

        self.exist_gain = float(lambda_exist if lambda_exist is not None else self._arg(args, "gcs_exist", 2.0))
        self.point_gain = float(lambda_point if lambda_point is not None else self._arg(args, "gcs_point", 15.0))
        self.point_valid_gain = float(
            lambda_point_valid if lambda_point_valid is not None else self._arg(args, "gcs_point_valid", 1.0)
        )
        self.smooth_gain = float(lambda_smooth if lambda_smooth is not None else self._arg(args, "gcs_smooth", 0.05))
        self.curve_gain = float(lambda_curve if lambda_curve is not None else self._arg(args, "gcs_curve", 0.1))
        self.mask_gain = float(lambda_mask if lambda_mask is not None else self._arg(args, "gcs_mask", 0.2))
        self.edge_gain = float(lambda_edge if lambda_edge is not None else self._arg(args, "gcs_edge", 0.2))
        self.count_gain = float(lambda_count if lambda_count is not None else self._arg(args, "gcs_count", 0.0))
        self.count_under5_gain = float(
            lambda_count_under5
            if lambda_count_under5 is not None
            else self._arg(args, "gcs_count_under5", 0.0)
        )
        self.count_boundary_gain = float(
            lambda_count_boundary
            if lambda_count_boundary is not None
            else self._arg(args, "gcs_count_boundary", 0.0)
        )
        self.short_side_geom_gain = float(
            lambda_short_side_geom
            if lambda_short_side_geom is not None
            else self._arg(args, "gcs_short_side_geom", 0.0)
        )
        self.spurious_neg_gain = float(
            lambda_spurious_neg if lambda_spurious_neg is not None else self._arg(args, "gcs_spurious_neg", 0.0)
        )
        self.count_boundary_gt4_weight = float(
            count_boundary_gt4_weight
            if count_boundary_gt4_weight is not None
            else self._arg(args, "gcs_count_boundary_gt4_weight", 2.0)
        )
        self.count_boundary_gt5_weight = float(
            count_boundary_gt5_weight
            if count_boundary_gt5_weight is not None
            else self._arg(args, "gcs_count_boundary_gt5_weight", 1.5)
        )
        gt5_under_weight_arg = self._arg(args, "gcs_count_boundary_gt5_under_weight", None)
        self.count_boundary_gt5_under_weight = float(
            count_boundary_gt5_under_weight
            if count_boundary_gt5_under_weight is not None
            else (gt5_under_weight_arg if gt5_under_weight_arg is not None else self.count_boundary_gt5_weight)
        )
        self.count_boundary_margin34 = float(
            count_boundary_margin34
            if count_boundary_margin34 is not None
            else self._arg(args, "gcs_count_boundary_margin34", 0.35)
        )
        self.count_boundary_margin45 = float(
            count_boundary_margin45
            if count_boundary_margin45 is not None
            else self._arg(args, "gcs_count_boundary_margin45", 0.35)
        )
        self.short_side_geom_min_gt_lanes = int(
            short_side_geom_min_gt_lanes
            if short_side_geom_min_gt_lanes is not None
            else self._arg(args, "gcs_short_side_geom_min_gt_lanes", 4)
        )
        self.short_side_geom_visible_max = int(
            short_side_geom_visible_max
            if short_side_geom_visible_max is not None
            else self._arg(args, "gcs_short_side_geom_visible_max", 20)
        )
        self.short_side_geom_side_only = self._bool_arg(
            short_side_geom_side_only
            if short_side_geom_side_only is not None
            else self._arg(args, "gcs_short_side_geom_side_only", False)
        )
        self.short_side_geom_weight_gt4 = float(
            short_side_geom_weight_gt4
            if short_side_geom_weight_gt4 is not None
            else self._arg(args, "gcs_short_side_geom_weight_gt4", 1.0)
        )
        self.short_side_geom_weight_gt5 = float(
            short_side_geom_weight_gt5
            if short_side_geom_weight_gt5 is not None
            else self._arg(args, "gcs_short_side_geom_weight_gt5", 1.0)
        )
        self.spurious_neg_weight = float(
            spurious_neg_weight if spurious_neg_weight is not None else self._arg(args, "gcs_spurious_neg_weight", 1.0)
        )
        self.spurious_gt3_weight = float(
            spurious_gt3_weight
            if spurious_gt3_weight is not None
            else self._arg(args, "gcs_spurious_gt3_weight", 1.0)
        )
        self.spurious_gt4_weight = float(
            spurious_gt4_weight
            if spurious_gt4_weight is not None
            else self._arg(args, "gcs_spurious_gt4_weight", 1.0)
        )
        self.spurious_gt5_weight = float(
            spurious_gt5_weight
            if spurious_gt5_weight is not None
            else self._arg(args, "gcs_spurious_gt5_weight", 1.0)
        )
        self.spurious_disable_gt5 = self._bool_arg(
            spurious_disable_gt5
            if spurious_disable_gt5 is not None
            else self._arg(args, "gcs_spurious_disable_gt5", False)
        )
        self.spurious_max_points = int(
            spurious_max_points if spurious_max_points is not None else self._arg(args, "gcs_spurious_max_points", 12)
        )
        self.spurious_close_px = float(
            spurious_close_px if spurious_close_px is not None else self._arg(args, "gcs_spurious_close_px", 30.0)
        )
        self.spurious_min_overlap = int(
            spurious_min_overlap
            if spurious_min_overlap is not None
            else self._arg(args, "gcs_spurious_min_overlap", 3)
        )
        self.spurious_gt_protect = self._bool_arg(
            spurious_gt_protect
            if spurious_gt_protect is not None
            else self._arg(args, "gcs_spurious_gt_protect", False)
        )
        self.spurious_gt_protect_px = float(
            spurious_gt_protect_px
            if spurious_gt_protect_px is not None
            else self._arg(args, "gcs_spurious_gt_protect_px", 25.0)
        )
        self.spurious_gt_protect_min_overlap = int(
            spurious_gt_protect_min_overlap
            if spurious_gt_protect_min_overlap is not None
            else self._arg(args, "gcs_spurious_gt_protect_min_overlap", 3)
        )
        self.spurious_gt_protect_min_gt_lanes = int(
            spurious_gt_protect_min_gt_lanes
            if spurious_gt_protect_min_gt_lanes is not None
            else self._arg(args, "gcs_spurious_gt_protect_min_gt_lanes", 0)
        )
        self.spurious_gt_protect_margin_px = float(
            spurious_gt_protect_margin_px
            if spurious_gt_protect_margin_px is not None
            else self._arg(args, "gcs_spurious_gt_protect_margin_px", 5.0)
        )
        self.spurious_gt_protect_mode = str(
            spurious_gt_protect_mode
            if spurious_gt_protect_mode is not None
            else self._arg(args, "gcs_spurious_gt_protect_mode", "better_matched")
        )
        self.eval_point_valid_thr = float(
            eval_point_valid_thr
            if eval_point_valid_thr is not None
            else self._arg(args, "gcs_eval_point_valid_thr", 0.5)
        )
        self.gt5_short_visible_thr = int(
            gt5_short_visible_thr
            if gt5_short_visible_thr is not None
            else self._arg(args, "gcs_gt5_short_visible_thr", 0)
        )
        self.gt5_short_point_valid_weight = float(
            gt5_short_point_valid_weight
            if gt5_short_point_valid_weight is not None
            else self._arg(args, "gcs_gt5_short_point_valid_weight", 1.0)
        )
        self.count_under5_min_lanes = int(
            count_under5_min_lanes
            if count_under5_min_lanes is not None
            else self._arg(args, "gcs_count_under5_min_lanes", 5)
        )
        if self.count_under5_min_lanes < 1:
            raise ValueError(f"gcs_count_under5_min_lanes must be >= 1, got {self.count_under5_min_lanes}.")
        if self.count_boundary_margin34 < 0.0:
            raise ValueError(f"gcs_count_boundary_margin34 must be >= 0, got {self.count_boundary_margin34}.")
        if self.count_boundary_margin45 < 0.0:
            raise ValueError(f"gcs_count_boundary_margin45 must be >= 0, got {self.count_boundary_margin45}.")
        if self.short_side_geom_gain < 0.0:
            raise ValueError(f"gcs_short_side_geom must be >= 0, got {self.short_side_geom_gain}.")
        if self.short_side_geom_min_gt_lanes < 1:
            raise ValueError(
                "gcs_short_side_geom_min_gt_lanes must be >= 1, "
                f"got {self.short_side_geom_min_gt_lanes}."
            )
        if self.short_side_geom_visible_max < 0:
            raise ValueError(
                "gcs_short_side_geom_visible_max must be >= 0, "
                f"got {self.short_side_geom_visible_max}."
            )
        if self.short_side_geom_weight_gt4 < 0.0:
            raise ValueError(
                "gcs_short_side_geom_weight_gt4 must be >= 0, "
                f"got {self.short_side_geom_weight_gt4}."
            )
        if self.short_side_geom_weight_gt5 < 0.0:
            raise ValueError(
                "gcs_short_side_geom_weight_gt5 must be >= 0, "
                f"got {self.short_side_geom_weight_gt5}."
            )
        if self.spurious_neg_gain < 0.0:
            raise ValueError(f"gcs_spurious_neg must be >= 0, got {self.spurious_neg_gain}.")
        if self.spurious_neg_weight < 0.0:
            raise ValueError(f"gcs_spurious_neg_weight must be >= 0, got {self.spurious_neg_weight}.")
        if self.spurious_gt3_weight < 0.0:
            raise ValueError(f"gcs_spurious_gt3_weight must be >= 0, got {self.spurious_gt3_weight}.")
        if self.spurious_gt4_weight < 0.0:
            raise ValueError(f"gcs_spurious_gt4_weight must be >= 0, got {self.spurious_gt4_weight}.")
        if self.spurious_gt5_weight < 0.0:
            raise ValueError(f"gcs_spurious_gt5_weight must be >= 0, got {self.spurious_gt5_weight}.")
        if self.spurious_max_points < 2:
            raise ValueError(f"gcs_spurious_max_points must be >= 2, got {self.spurious_max_points}.")
        if self.spurious_close_px < 0.0:
            raise ValueError(f"gcs_spurious_close_px must be >= 0, got {self.spurious_close_px}.")
        if self.spurious_min_overlap < 1:
            raise ValueError(f"gcs_spurious_min_overlap must be >= 1, got {self.spurious_min_overlap}.")
        if self.spurious_gt_protect_px < 0.0:
            raise ValueError(
                f"gcs_spurious_gt_protect_px must be >= 0, got {self.spurious_gt_protect_px}."
            )
        if self.spurious_gt_protect_min_overlap < 1:
            raise ValueError(
                "gcs_spurious_gt_protect_min_overlap must be >= 1, "
                f"got {self.spurious_gt_protect_min_overlap}."
            )
        if self.spurious_gt_protect_min_gt_lanes < 0:
            raise ValueError(
                "gcs_spurious_gt_protect_min_gt_lanes must be >= 0, "
                f"got {self.spurious_gt_protect_min_gt_lanes}."
            )
        if self.spurious_gt_protect_margin_px < 0.0:
            raise ValueError(
                "gcs_spurious_gt_protect_margin_px must be >= 0, "
                f"got {self.spurious_gt_protect_margin_px}."
            )
        if self.spurious_gt_protect_mode != "better_matched":
            raise ValueError(
                "Only gcs_spurious_gt_protect_mode='better_matched' is supported, "
                f"got {self.spurious_gt_protect_mode!r}."
            )
        if not 0.0 <= self.eval_point_valid_thr <= 1.0:
            raise ValueError(f"gcs_eval_point_valid_thr must be in [0, 1], got {self.eval_point_valid_thr}.")
        if self.gt5_short_visible_thr < 0:
            raise ValueError(f"gcs_gt5_short_visible_thr must be >= 0, got {self.gt5_short_visible_thr}.")
        if self.gt5_short_point_valid_weight < 0.0:
            raise ValueError(
                "gcs_gt5_short_point_valid_weight must be >= 0, "
                f"got {self.gt5_short_point_valid_weight}."
            )
        self.curve_alpha = float(curve_alpha if curve_alpha is not None else self._arg(args, "gcs_curve_alpha", 5.0))
        self.curve_weight_max = float(
            curve_weight_max if curve_weight_max is not None else self._arg(args, "gcs_curve_weight_max", 5.0)
        )
        self.exist_pos_weight = float(
            exist_pos_weight if exist_pos_weight is not None else self._arg(args, "gcs_exist_pos_weight", 1.0)
        )
        self.exist_focal_gamma = float(
            exist_focal_gamma if exist_focal_gamma is not None else self._arg(args, "gcs_exist_focal_gamma", 0.0)
        )
        self.exist_focal_alpha = float(
            exist_focal_alpha if exist_focal_alpha is not None else self._arg(args, "gcs_exist_focal_alpha", -1.0)
        )
        self.point_valid_pos_weight_max = float(
            point_valid_pos_weight_max
            if point_valid_pos_weight_max is not None
            else self._arg(args, "gcs_point_valid_pos_weight_max", 10.0)
        )
        self.exist_quality_alpha = float(
            exist_quality_alpha if exist_quality_alpha is not None else self._arg(args, "gcs_exist_quality_alpha", 1.0)
        )
        self.exist_quality_mode = str(
            exist_quality_mode
            if exist_quality_mode is not None
            else self._arg(args, "gcs_exist_quality_mode", "linear")
        ).lower()
        self.exist_quality_tau = float(
            exist_quality_tau if exist_quality_tau is not None else self._arg(args, "gcs_exist_quality_tau", 25.0)
        )
        self.exist_quality_floor = float(
            exist_quality_floor if exist_quality_floor is not None else self._arg(args, "gcs_exist_quality_floor", 0.0)
        )
        self.exist_quality_pos_px = float(
            exist_quality_pos_px
            if exist_quality_pos_px is not None
            else self._arg(args, "gcs_exist_quality_pos_px", 10.0)
        )
        self.exist_quality_neg_px = float(
            exist_quality_neg_px
            if exist_quality_neg_px is not None
            else self._arg(args, "gcs_exist_quality_neg_px", 20.0)
        )
        if self.exist_quality_mode in {"exponential"}:
            self.exist_quality_mode = "exp"
        if self.exist_quality_mode not in {"linear", "exp"}:
            raise ValueError(f"Unsupported gcs_exist_quality_mode={self.exist_quality_mode!r}; use 'linear' or 'exp'.")
        if self.exist_quality_neg_px <= self.exist_quality_pos_px:
            raise ValueError(
                "gcs_exist_quality_neg_px must be greater than gcs_exist_quality_pos_px "
                f"({self.exist_quality_neg_px} <= {self.exist_quality_pos_px})."
            )
        self.mask_pos_weight_max = float(
            mask_pos_weight_max if mask_pos_weight_max is not None else self._arg(args, "gcs_mask_pos_weight_max", 20.0)
        )
        self.edge_pos_weight_max = float(
            edge_pos_weight_max if edge_pos_weight_max is not None else self._arg(args, "gcs_edge_pos_weight_max", 50.0)
        )
        self.aux_dice_gain = float(aux_dice_gain if aux_dice_gain is not None else self._arg(args, "gcs_aux_dice", 0.5))
        image_size = (
            image_size
            or self._arg(args, "gcs_imgsz", None)
            or self._arg(args, "image_shape", None)
            or getattr(model, "gcs_imgsz", None)
        )
        assert image_size is not None and image_size != "", (
            "GCSLoss requires an explicit rectangular image_size/gcs_imgsz. "
            "Do not let loss scaling fall back to scalar args.imgsz."
        )
        self.image_size = normalize_imgsz(image_size)
        self.register_buffer(
            "point_scale",
            torch.tensor(self._point_scale(self.image_size), dtype=torch.float32).view(1, 1, 2),
            persistent=False,
        )
        self.register_buffer(
            "pixel_scale",
            torch.tensor(self._pixel_scale(self.image_size), dtype=torch.float32).view(1, 1, 2),
            persistent=False,
        )

        self.matcher = GCSHungarianMatcher(
            cost_point=float(self._arg(args, "gcs_cost_point", 5.0)),
            cost_curve=float(self._arg(args, "gcs_cost_curve", 0.05)),
            cost_exist=float(self._arg(args, "gcs_cost_exist", 0.1)),
            image_size=self.image_size,
            min_overlap=int(match_min_overlap if match_min_overlap is not None else self._arg(args, "gcs_match_min_overlap", 2)),
            max_x_dist=float(match_max_x_dist if match_max_x_dist is not None else self._arg(args, "gcs_match_max_x_dist", 0.0)),
            match_gate_px=float(match_gate_px if match_gate_px is not None else self._arg(args, "gcs_match_gate_px", 160.0)),
        )

    @staticmethod
    def _arg(args, name: str, default):
        """Read a config value from an Ultralytics namespace or dict."""
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    @staticmethod
    def _bool_arg(value) -> bool:
        """Parse bool-like config values without treating the string 'False' as true."""
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _point_scale(image_size) -> tuple[float, float]:
        """Return normalized pixel-aspect weights for x/y point losses."""
        if image_size is None or image_size == "":
            return 1.0, 1.0
        h, w = normalize_imgsz(image_size)
        base = float(max(h, w))
        return float(w) / base, float(h) / base

    @staticmethod
    def _pixel_scale(image_size) -> tuple[float, float]:
        """Return x/y pixel scales for curvature losses on normalized points."""
        if image_size is None or image_size == "":
            return 1.0, 1.0
        h, w = normalize_imgsz(image_size)
        return float(w), float(h)

    def _scale_for(self, ref: torch.Tensor) -> torch.Tensor:
        """Return x/y point-loss weights on the same device and dtype as ref."""
        return self.point_scale.to(device=ref.device, dtype=ref.dtype)

    def _pixel_scale_for(self, ref: torch.Tensor) -> torch.Tensor:
        """Return x/y pixel scales on the same device and dtype as ref."""
        pixel_scale = getattr(self, "pixel_scale", None)
        if pixel_scale is None:
            pixel_scale = ref.new_ones((1, 1, 2))
        return pixel_scale.to(device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _zero_like(pred_points: torch.Tensor) -> torch.Tensor:
        """Return a differentiable scalar zero on the prediction device."""
        return pred_points.sum() * 0.0

    def _exist_quality_from_ape(self, ape: torch.Tensor) -> torch.Tensor:
        """Map matched lane APE in pixels to an existence target quality."""
        if self.exist_quality_mode == "exp":
            tau = max(float(self.exist_quality_tau), 1e-6)
            floor = min(max(float(self.exist_quality_floor), 0.0), 1.0)
            return torch.exp(-ape / tau).clamp(min=floor, max=1.0)

        pos_px = float(self.exist_quality_pos_px)
        neg_px = float(self.exist_quality_neg_px)
        quality = ((neg_px - ape) / max(neg_px - pos_px, 1e-6)).clamp(min=0.0, max=1.0)
        quality = torch.where(ape <= pos_px, torch.ones_like(quality), quality)
        quality = torch.where(ape >= neg_px, torch.zeros_like(quality), quality)
        return quality

    @staticmethod
    def _targets_from_batch(batch: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Support both local and reference GCS batch key names."""
        if "lanes" in batch and "lane_valid" in batch:
            return batch["lanes"], batch["lane_valid"]
        if "gt_lanes" in batch and "gt_lane_valid" in batch:
            return batch["gt_lanes"], batch["gt_lane_valid"]
        raise KeyError(
            "GCSLoss requires batch['lanes']/batch['lane_valid'] or batch['gt_lanes']/batch['gt_lane_valid']."
        )

    @staticmethod
    def _normalize_pred_shapes(preds: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Read and validate GCS head outputs."""
        if "pred_points" not in preds or "pred_logits" not in preds:
            raise KeyError("GCSLoss expects preds with 'pred_points' and 'pred_logits'.")

        pred_points = preds["pred_points"]
        pred_logits = preds["pred_logits"]
        if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
            pred_logits = pred_logits.squeeze(-1)
        if pred_points.ndim != 4 or pred_points.shape[-1] != 2:
            raise ValueError(f"pred_points must have shape B x Q x K x 2, got {tuple(pred_points.shape)}.")
        if pred_logits.ndim != 2:
            raise ValueError(f"pred_logits must have shape B x Q, got {tuple(pred_logits.shape)}.")
        if pred_points.shape[:2] != pred_logits.shape:
            raise ValueError(
                f"pred_points B,Q must match pred_logits, got {tuple(pred_points.shape[:2])} vs {tuple(pred_logits.shape)}."
            )
        return pred_points, pred_logits

    @staticmethod
    def _pred_valid_logits(preds: dict[str, torch.Tensor], pred_points: torch.Tensor) -> torch.Tensor | None:
        """Read optional per-point visibility logits from the GCS head output."""
        pred_valid_logits = preds.get("pred_valid_logits")
        if pred_valid_logits is None:
            return None
        if pred_valid_logits.ndim == 4 and pred_valid_logits.shape[-1] == 1:
            pred_valid_logits = pred_valid_logits.squeeze(-1)
        if pred_valid_logits.shape != pred_points.shape[:3]:
            raise ValueError(
                "pred_valid_logits must have shape B x Q x K matching pred_points, "
                f"got {tuple(pred_valid_logits.shape)} vs {tuple(pred_points.shape[:3])}."
            )
        return pred_valid_logits

    def exist_loss(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Quality-aware existence supervision for matched lane queries.

        Geometry alone is not enough for fixed-y lanes: a query can fit the GT x
        coordinates on visible anchors while marking many invalid anchors as
        visible, which renders a long false polyline. Fold point-visibility IoU
        into the existence target so such queries are not trained as confident
        positives until their visible segment is also correct.
        """
        target = torch.zeros_like(pred_logits)
        alpha = min(max(float(self.exist_quality_alpha), 0.0), 1.0)
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._pixel_scale_for(pred_points)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel():
                if alpha <= 0.0:
                    target[b, src_idx] = 1.0
                    continue

                pred = pred_points[b, src_idx].detach()
                target_points = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
                valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]
                point_error = torch.norm((pred - target_points) * scale, dim=-1)
                ape = (point_error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                quality = self._exist_quality_from_ape(ape)
                if pred_valid_logits is not None:
                    valid_prob = pred_valid_logits[b, src_idx].detach().sigmoid().to(dtype=dtype)
                    intersection = (valid_prob * valid).sum(dim=1)
                    union = valid_prob.sum(dim=1) + valid.sum(dim=1) - intersection
                    visible_quality = intersection / union.clamp_min(1e-6)
                    quality = quality * visible_quality.clamp(min=0.0, max=1.0)
                target[b, src_idx] = (1.0 - alpha) + alpha * quality.to(dtype=target.dtype)
        pos_weight = pred_logits.new_tensor(self.exist_pos_weight)
        loss = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight, reduction="none")
        gamma = max(float(self.exist_focal_gamma), 0.0)
        if gamma > 0.0:
            prob = pred_logits.sigmoid()
            focal_weight = (target - prob).abs().clamp(min=0.0, max=1.0).pow(gamma)
            focal_alpha = float(self.exist_focal_alpha)
            if 0.0 <= focal_alpha <= 1.0:
                alpha_t = focal_alpha * target + (1.0 - focal_alpha) * (1.0 - target)
                focal_weight = focal_weight * alpha_t
            loss = loss * focal_weight
        return loss.mean()

    def point_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Aspect-weighted L1 point loss on Hungarian-matched lane point sequences."""
        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pred = pred_points[b, src_idx]
            target = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
            valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]

            losses.append(
                aspect_weighted_l1_point_loss(
                    pred,
                    target,
                    valid > 0.5,
                    image_size=self.image_size,
                    y_weight=1.0,
                    x_only=False,
                )
            )

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    @staticmethod
    def _side_gt_indices_by_bottom_x(gt_points_b: torch.Tensor, gt_valid_b: torch.Tensor) -> set[int]:
        """Return leftmost/rightmost GT ids using each lane's bottom-most visible x."""
        valid_mask = gt_valid_b > 0.5
        lane_has_visible = valid_mask.any(dim=1)
        if not bool(lane_has_visible.any()):
            return set()
        lane_ids = lane_has_visible.nonzero(as_tuple=False).reshape(-1)
        bottom_idx = valid_mask.to(dtype=torch.long).argmax(dim=1)
        bottom_x = gt_points_b[lane_ids, bottom_idx[lane_ids], 0]
        left_id = int(lane_ids[bottom_x.argmin()].item())
        right_id = int(lane_ids[bottom_x.argmax()].item())
        return {left_id, right_id}

    def short_side_geom_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extra matched-lane geometry loss for short side GT lanes."""
        zero = pred_points.new_zeros(())
        if self.short_side_geom_gain == 0.0:
            return self._zero_like(pred_points), zero, zero, zero

        bsz = pred_points.shape[0]
        device, dtype = pred_points.device, pred_points.dtype
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=dtype).reshape(-1)
        if gt_lanes.numel() != bsz:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={bsz}.")

        losses: list[torch.Tensor] = []
        selected_count = 0
        selected_gt4 = 0
        selected_gt5 = 0
        min_gt_lanes = int(self.short_side_geom_min_gt_lanes)
        visible_max = int(self.short_side_geom_visible_max)

        for b, (src_idx, tgt_idx) in enumerate(indices):
            gt_count = int(round(float(gt_lanes[b].detach().item())))
            if gt_count < min_gt_lanes or src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            gt_points_b = gt_points[b].to(device=device, dtype=dtype)
            gt_valid_b = gt_valid[b].to(device=device, dtype=dtype)
            if gt_points_b.ndim != 3 or gt_points_b.shape[-1] != 2:
                raise ValueError(f"Each GT lane tensor must have shape N x K x 2, got {tuple(gt_points_b.shape)}.")
            if gt_valid_b.shape != gt_points_b.shape[:2]:
                raise ValueError(
                    f"GT valid mask must match GT lane first two dims, got {tuple(gt_valid_b.shape)} vs {tuple(gt_points_b.shape[:2])}."
                )

            side_ids = self._side_gt_indices_by_bottom_x(gt_points_b.detach(), gt_valid_b.detach())
            if self.short_side_geom_side_only and not side_ids:
                continue
            image_weight = 1.0
            if gt_count == 4:
                image_weight = float(self.short_side_geom_weight_gt4)
            elif gt_count >= 5:
                image_weight = float(self.short_side_geom_weight_gt5)

            for src, tgt in zip(src_idx, tgt_idx):
                tgt_i = int(tgt.item())
                if self.short_side_geom_side_only and tgt_i not in side_ids:
                    continue
                valid = gt_valid_b[tgt_i]
                visible = int((valid > 0.5).sum().item())
                if visible <= 0 or visible > visible_max:
                    continue
                pred = pred_points[b, src].unsqueeze(0)
                target = gt_points_b[tgt_i].unsqueeze(0)
                valid_mask = valid.unsqueeze(0) > 0.5
                loss = aspect_weighted_l1_point_loss(
                    pred,
                    target,
                    valid_mask,
                    image_size=self.image_size,
                    y_weight=1.0,
                    x_only=False,
                )
                losses.append(loss * image_weight)
                selected_count += 1
                if gt_count == 4:
                    selected_gt4 += 1
                elif gt_count >= 5:
                    selected_gt5 += 1

        loss = torch.stack(losses).mean() if losses else self._zero_like(pred_points)
        return (
            loss,
            pred_points.new_tensor(float(selected_count)),
            pred_points.new_tensor(float(selected_gt4)),
            pred_points.new_tensor(float(selected_gt5)),
        )

    def point_valid_loss(
        self,
        pred_valid_logits: torch.Tensor | None,
        pred_points: torch.Tensor,
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """BCE supervision for visible fixed-y anchors on matched lanes and zero target for unmatched queries."""
        if pred_valid_logits is None:
            loss = self._zero_like(pred_points)
            zero = pred_points.new_zeros(())
            return (loss, zero, zero, zero) if return_details else loss

        target = torch.zeros_like(pred_valid_logits)
        extra_weight = torch.ones_like(pred_valid_logits)
        gt5_short_boost_mask = torch.zeros_like(pred_valid_logits, dtype=torch.bool)
        gt5_short_pos_count = 0
        gt5_short_pos_anchor_count = 0
        if gt_lanes is not None:
            gt_lanes = torch.as_tensor(gt_lanes, device=pred_valid_logits.device, dtype=pred_valid_logits.dtype).reshape(-1)
            if gt_lanes.numel() != pred_valid_logits.shape[0]:
                raise ValueError(
                    f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={pred_valid_logits.shape[0]}."
                )
        rescue_enabled = gt_lanes is not None and self.gt5_short_visible_thr > 0
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=target.device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=target.device, dtype=torch.long)
            target_valid = gt_valid[b].to(device=target.device, dtype=target.dtype)[tgt_idx]
            target[b, src_idx] = target_valid
            if not rescue_enabled or int(round(float(gt_lanes[b].detach().item()))) != 5:
                continue

            visible_counts = target_valid.sum(dim=1)
            short_mask = visible_counts <= float(self.gt5_short_visible_thr)
            if not bool(short_mask.any()):
                continue
            boost_mask = short_mask[:, None] & target_valid.bool()
            gt5_short_pos_count += int(short_mask.sum().item())
            gt5_short_pos_anchor_count += int(boost_mask.sum().item())
            if bool(boost_mask.any()):
                local_weight = torch.ones_like(target_valid)
                local_weight[boost_mask] = float(self.gt5_short_point_valid_weight)
                local_boost_mask = torch.zeros_like(target_valid, dtype=torch.bool)
                local_boost_mask[boost_mask] = True
                extra_weight[b, src_idx] = local_weight
                gt5_short_boost_mask[b, src_idx] = local_boost_mask

        pos = target.sum().clamp_min(1.0)
        neg = (target.numel() - target.sum()).clamp_min(1.0)
        pos_weight = (neg / pos).clamp(min=1.0, max=float(self.point_valid_pos_weight_max)).to(pred_valid_logits)
        bce = F.binary_cross_entropy_with_logits(pred_valid_logits, target, pos_weight=pos_weight, reduction="none")
        loss = (bce * extra_weight).sum() / extra_weight.sum().clamp_min(1.0)
        if not return_details:
            return loss
        gt5_short_point_valid_loss = (
            bce[gt5_short_boost_mask].mean() if bool(gt5_short_boost_mask.any()) else self._zero_like(pred_points)
        )
        return (
            loss,
            pred_valid_logits.new_tensor(float(gt5_short_pos_count)),
            pred_valid_logits.new_tensor(float(gt5_short_pos_anchor_count)),
            gt5_short_point_valid_loss,
        )

    def smooth_loss(
        self,
        pred_points: torch.Tensor,
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Second-order smoothness regularization for matched predicted lanes."""
        if pred_points.shape[2] < 3:
            return self._zero_like(pred_points)

        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._scale_for(pred_points)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pred = pred_points[b, src_idx]
            valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]
            triplet_valid = valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]
            valid_sum = triplet_valid.sum().clamp_min(1.0)

            lap = (pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]) * scale
            loss = lap.abs().sum(dim=-1) * triplet_valid
            losses.append(loss.sum() / valid_sum)

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    def curve_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Adaptive curvature-aware loss with GT-curvature weighting."""
        if pred_points.shape[2] < 3:
            return self._zero_like(pred_points)

        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._pixel_scale_for(pred_points)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            pred = pred_points[b, src_idx]
            target = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
            valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]
            triplet_valid = valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]
            valid_sum = triplet_valid.sum().clamp_min(1.0)

            pred_px = pred * scale
            target_px = target * scale
            pred_lap = pred_px[:, 2:] - 2.0 * pred_px[:, 1:-1] + pred_px[:, :-2]
            gt_lap = target_px[:, 2:] - 2.0 * target_px[:, 1:-1] + target_px[:, :-2]
            gt_curve_mag = torch.norm(gt_lap.detach(), dim=-1)
            weight = (1.0 + self.curve_alpha * gt_curve_mag).clamp(max=self.curve_weight_max)

            loss = F.smooth_l1_loss(pred_lap, gt_lap, reduction="none").sum(dim=-1)
            loss = loss * triplet_valid * weight
            losses.append(loss.sum() / valid_sum)

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    def target_lane_count(self, pred_logits: torch.Tensor, batch: dict, gt_valid: list[torch.Tensor]) -> torch.Tensor:
        """Return one GT lane count per image on the prediction device."""
        num_lanes = batch.get("num_lanes")
        if num_lanes is not None:
            target = torch.as_tensor(num_lanes, device=pred_logits.device, dtype=pred_logits.dtype).reshape(-1)
            if target.numel() != pred_logits.shape[0]:
                raise ValueError(
                    f"batch['num_lanes'] must have one value per image, got {target.numel()} vs B={pred_logits.shape[0]}."
                )
        else:
            counts = []
            for valid in gt_valid:
                valid = valid.detach().to(device=pred_logits.device)
                if valid.ndim != 2:
                    raise ValueError(f"GT lane_valid must have shape N x K, got {tuple(valid.shape)}.")
                counts.append(int((valid.float().sum(dim=1) >= 2).sum().item()))
            target = pred_logits.new_tensor(counts, dtype=pred_logits.dtype)
        return target

    def count_losses(
        self, pred_logits: torch.Tensor, batch: dict, gt_valid: list[torch.Tensor], target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cardinality losses and count-score diagnostics."""
        pred_count = pred_logits.sigmoid().sum(dim=1)
        if target is None:
            target = self.target_lane_count(pred_logits, batch, gt_valid)
        count_loss = F.smooth_l1_loss(pred_count, target)

        under_gap = torch.relu(target - pred_count)
        mask = (target >= float(self.count_under5_min_lanes)).to(dtype=pred_logits.dtype)
        count_under5_loss = (mask * under_gap.pow(2)).sum() / mask.sum().clamp_min(1.0)
        count_boundary_loss, cnt_bound_5under = self.count_boundary_loss(pred_count, target, return_details=True)
        count_score_mean = pred_count.mean()
        return count_loss, count_under5_loss, count_boundary_loss, count_score_mean, cnt_bound_5under, count_score_mean

    def count_loss(self, pred_logits: torch.Tensor, batch: dict, gt_valid: list[torch.Tensor]) -> torch.Tensor:
        """Cardinality loss that aligns summed existence probability with GT lane count."""
        return self.count_losses(pred_logits, batch, gt_valid)[0]

    def count_boundary_loss(
        self, count_score: torch.Tensor, target: torch.Tensor, return_details: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Penalize adjacent-count boundary drift for GT3/GT4/GT5 lane counts."""

        def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            weight = mask.to(dtype=values.dtype)
            return (values * weight).sum() / weight.sum().clamp_min(1.0)

        target_count = target.round()
        gt3 = target_count == 3
        gt4 = target_count == 4
        gt5 = target_count == 5
        margin34 = float(self.count_boundary_margin34)
        margin45 = float(self.count_boundary_margin45)

        gt3_over = torch.relu(count_score - (3.0 + margin34)).pow(2)
        gt4_under = torch.relu((4.0 - margin34) - count_score).pow(2)
        gt4_over = torch.relu(count_score - (4.0 + margin45)).pow(2)
        gt5_under = torch.relu((5.0 - margin45) - count_score).pow(2)
        loss_5_under = masked_mean(gt5_under, gt5)
        loss = (
            masked_mean(gt3_over, gt3)
            + float(self.count_boundary_gt4_weight) * masked_mean(gt4_under + gt4_over, gt4)
            + float(self.count_boundary_gt5_under_weight) * loss_5_under
        )
        return (loss, loss_5_under) if return_details else loss

    def _spurious_x_scale(self, pred_points: torch.Tensor) -> torch.Tensor:
        """Return x-coordinate scale, avoiding double scaling if points are already pixel coordinates."""
        points = pred_points.detach()
        finite = points[torch.isfinite(points)]
        if finite.numel() and float(finite.abs().max().item()) > 2.0:
            return pred_points.new_tensor(1.0)
        return self._pixel_scale_for(pred_points).reshape(-1)[0]

    def _spurious_gt_group_and_weight(self, gt_count: int) -> tuple[int, float]:
        """Map GT lane count to diagnostic group and per-image spurious-negative weight."""
        if gt_count <= 3:
            return 3, float(self.spurious_gt3_weight)
        if gt_count == 4:
            return 4, float(self.spurious_gt4_weight)
        return 5, float(self.spurious_gt5_weight)

    def _spurious_candidate_gt_protected(
        self,
        points_b: torch.Tensor,
        valid_prob_b: torch.Tensor,
        uq: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        gt_points_b: torch.Tensor,
        gt_valid_b: torch.Tensor,
        x_scale: torch.Tensor,
        valid_thr: float,
    ) -> bool:
        """Return whether an unmatched duplicate-like candidate is close enough to GT to protect."""
        if not self.spurious_gt_protect or gt_points_b.numel() == 0:
            return False
        if gt_points_b.ndim != 3 or gt_points_b.shape[-1] != 2:
            raise ValueError(f"Each GT lane tensor must have shape N x K x 2, got {tuple(gt_points_b.shape)}.")
        if gt_valid_b.shape != gt_points_b.shape[:2]:
            raise ValueError(
                f"GT valid mask must match GT lane first two dims, got {tuple(gt_valid_b.shape)} vs {tuple(gt_points_b.shape[:2])}."
            )

        candidate_valid = valid_prob_b[uq] >= valid_thr
        min_overlap = int(self.spurious_gt_protect_min_overlap)
        protect_px = float(self.spurious_gt_protect_px)
        margin_px = float(self.spurious_gt_protect_margin_px)

        for gt_i in range(gt_points_b.shape[0]):
            lane_valid = gt_valid_b[gt_i] > 0.5
            candidate_overlap = candidate_valid & lane_valid
            if int(candidate_overlap.sum().item()) < min_overlap:
                continue
            candidate_dx = (points_b[uq, candidate_overlap, 0] - gt_points_b[gt_i, candidate_overlap, 0]).abs()
            candidate_dx = candidate_dx * x_scale
            candidate_dx_mean = float(candidate_dx.mean().item())
            if candidate_dx_mean > protect_px:
                continue

            matched_pos = (tgt_idx == int(gt_i)).nonzero(as_tuple=False).reshape(-1)
            if matched_pos.numel() == 0:
                return True

            mq = src_idx[matched_pos[0]]
            matched_valid = valid_prob_b[mq] >= valid_thr
            matched_overlap = matched_valid & lane_valid
            if int(matched_overlap.sum().item()) < min_overlap:
                return True

            matched_dx = (points_b[mq, matched_overlap, 0] - gt_points_b[gt_i, matched_overlap, 0]).abs()
            matched_dx = matched_dx * x_scale
            matched_dx_mean = float(matched_dx.mean().item())
            if candidate_dx_mean + margin_px < matched_dx_mean:
                return True

        return False

    def spurious_negative_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Extra BCE negative loss for short unmatched queries duplicating matched lanes."""
        zero = pred_logits.new_zeros(())
        if self.spurious_neg_gain == 0.0:
            return self._zero_like(pred_points), zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero
        if pred_valid_logits is None:
            raise ValueError(
                "gcs_spurious_neg requires preds['pred_valid_logits'] with shape B x Q x K; "
                "disable --gcs-spurious-neg or use a GCS head that emits per-point visibility logits."
            )

        bsz, num_queries, _, _ = pred_points.shape
        device = pred_logits.device
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=pred_logits.dtype).reshape(-1)
        if gt_lanes.numel() != bsz:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={bsz}.")
        valid_thr = float(self.eval_point_valid_thr)
        valid_prob = pred_valid_logits.detach().sigmoid()
        points = pred_points.detach()
        x_scale = self._spurious_x_scale(pred_points)
        selected_losses = []
        candidate_count = 0
        protected_count = 0
        final_count = 0
        group_counts = {3: 0, 4: 0, 5: 0}
        group_losses = {3: [], 4: [], 5: []}

        for b in range(bsz):
            gt_count = int(round(float(gt_lanes[b].detach().item())))
            is_gt5 = gt_count >= 5
            if is_gt5 and self.spurious_disable_gt5:
                continue
            group_key, image_weight = self._spurious_gt_group_and_weight(gt_count)
            src_idx, tgt_idx = indices[b]
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            matched_mask = torch.zeros(num_queries, dtype=torch.bool, device=device)
            matched_mask[src_idx] = True
            unmatched_idx = torch.arange(num_queries, device=device)[~matched_mask]
            if unmatched_idx.numel() == 0:
                continue
            gt_points_b = gt_points[b].detach().to(device=device, dtype=points.dtype)
            gt_valid_b = gt_valid[b].detach().to(device=device)

            for uq in unmatched_idx:
                uq_valid = valid_prob[b, uq] >= valid_thr
                uq_count = int(uq_valid.sum().item())
                if uq_count < 2 or uq_count > self.spurious_max_points:
                    continue

                is_duplicate = False
                for mq in src_idx:
                    mq_valid = valid_prob[b, mq] >= valid_thr
                    overlap = uq_valid & mq_valid
                    if int(overlap.sum().item()) < self.spurious_min_overlap:
                        continue
                    dx_px = (points[b, uq, overlap, 0] - points[b, mq, overlap, 0]).abs() * x_scale
                    if float(dx_px.mean().item()) <= self.spurious_close_px:
                        is_duplicate = True
                        break

                if is_duplicate:
                    candidate_count += 1
                    protect_enabled_for_gt_count = gt_count >= int(self.spurious_gt_protect_min_gt_lanes)
                    if protect_enabled_for_gt_count and self._spurious_candidate_gt_protected(
                        points[b],
                        valid_prob[b],
                        uq,
                        src_idx,
                        tgt_idx,
                        gt_points_b,
                        gt_valid_b,
                        x_scale,
                        valid_thr,
                    ):
                        protected_count += 1
                        continue
                    raw_loss = F.binary_cross_entropy_with_logits(
                        pred_logits[b, uq], pred_logits.new_zeros(()), reduction="none"
                    )
                    weighted_loss = raw_loss * image_weight
                    selected_losses.append(weighted_loss)
                    final_count += 1
                    group_counts[group_key] += 1
                    group_losses[group_key].append(weighted_loss)

        def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(values).mean() if values else zero

        loss = torch.stack(selected_losses).mean() if selected_losses else self._zero_like(pred_points)
        return (
            loss,
            pred_logits.new_tensor(float(final_count)),
            pred_logits.new_tensor(float(candidate_count)),
            pred_logits.new_tensor(float(protected_count)),
            pred_logits.new_tensor(float(final_count)),
            loss.detach(),
            pred_logits.new_tensor(float(group_counts[3])),
            pred_logits.new_tensor(float(group_counts[4])),
            pred_logits.new_tensor(float(group_counts[5])),
            mean_or_zero(group_losses[3]),
            mean_or_zero(group_losses[4]),
            mean_or_zero(group_losses[5]),
        )

    @staticmethod
    def _foreground_pos_weight(target: torch.Tensor, max_weight: float) -> torch.Tensor:
        """Return a capped foreground weight for sparse binary auxiliary targets."""
        target = target.float()
        pos = target.sum().clamp_min(1.0)
        neg = (target.numel() - target.sum()).clamp_min(1.0)
        return (neg / pos).clamp(min=1.0, max=float(max_weight))

    @staticmethod
    def _dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        """Soft Dice loss for sparse foreground auxiliary targets."""
        target = target.float()
        dims = tuple(range(1, prob.ndim))
        intersection = (prob * target).sum(dim=dims)
        denom = prob.sum(dim=dims) + target.sum(dim=dims)
        return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()

    def mask_aux_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Auxiliary semantic mask loss, where logits are B x 2 x H x W and target is 0/1."""
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        if target.ndim != 3:
            raise ValueError(f"semantic_mask target must have shape B x H x W, got {tuple(target.shape)}.")
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise ValueError(f"aux_mask_logits must have shape B x 2 x H x W, got {tuple(logits.shape)}.")
        assert tuple(target.shape[-2:]) == tuple(logits.shape[-2:]), (
            "GCSLoss.mask_aux_loss shape mismatch: semantic_mask and aux_mask_logits must already share H,W. "
            f"target={tuple(target.shape[-2:])}, logits={tuple(logits.shape[-2:])}."
        )
        target = target.long().clamp(0, 1)
        pos_weight = self._foreground_pos_weight(target, max_weight=self.mask_pos_weight_max).to(logits)
        class_weight = torch.stack((logits.new_tensor(1.0), pos_weight))
        ce = F.cross_entropy(logits, target, weight=class_weight)
        fg_prob = logits.softmax(dim=1)[:, 1]
        dice = self._dice_loss(fg_prob, target)
        return ce + self.aux_dice_gain * dice

    def edge_aux_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Auxiliary edge mask loss, where logits are B x 1 x H x W and target is 0/1."""
        if target.ndim == 3:
            target = target[:, None]
        if target.ndim != 4 or target.shape[1] != 1:
            raise ValueError(f"edge_mask target must have shape B x 1 x H x W, got {tuple(target.shape)}.")
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"aux_edge_logits must have shape B x 1 x H x W, got {tuple(logits.shape)}.")
        assert tuple(target.shape[-2:]) == tuple(logits.shape[-2:]), (
            "GCSLoss.edge_aux_loss shape mismatch: edge_mask and aux_edge_logits must already share H,W. "
            f"target={tuple(target.shape[-2:])}, logits={tuple(logits.shape[-2:])}."
        )
        target = target.float().clamp(0, 1)
        pos_weight = self._foreground_pos_weight(target, max_weight=self.edge_pos_weight_max).to(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        dice = self._dice_loss(logits.sigmoid(), target)
        return bce + self.aux_dice_gain * dice

    def forward(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total GCS lane loss and detached loss components."""
        pred_points, pred_logits = self._normalize_pred_shapes(preds)
        pred_valid_logits = self._pred_valid_logits(preds, pred_points)
        if "img" in batch:
            assert_gcs_image_tensor(batch["img"], self.image_size, name="batch['img']", context="GCSLoss.forward")
        if "semantic_mask" in batch:
            assert_gcs_shape(
                batch["semantic_mask"].shape[-2:],
                self.image_size,
                name="batch['semantic_mask']",
                context="GCSLoss.forward",
            )
        if "edge_mask" in batch:
            assert_gcs_shape(
                batch["edge_mask"].shape[-2:],
                self.image_size,
                name="batch['edge_mask']",
                context="GCSLoss.forward",
            )
        if "aux_mask_logits" in preds:
            assert_gcs_shape(
                preds["aux_mask_logits"].shape[-2:],
                self.image_size,
                name="preds['aux_mask_logits']",
                context="GCSLoss.forward",
            )
        if "aux_edge_logits" in preds:
            assert_gcs_shape(
                preds["aux_edge_logits"].shape[-2:],
                self.image_size,
                name="preds['aux_edge_logits']",
                context="GCSLoss.forward",
            )
        gt_points, gt_valid = self._targets_from_batch(batch)

        indices = self.matcher(pred_points, pred_logits, gt_points, gt_valid)
        exist_loss = self.exist_loss(pred_logits, pred_points, pred_valid_logits, gt_points, gt_valid, indices)
        point_loss = self.point_loss(pred_points, gt_points, gt_valid, indices)
        gt_lanes = self.target_lane_count(pred_logits, batch, gt_valid)
        (
            point_valid_loss,
            gt5_short_pos_count,
            gt5_short_pos_anchor_count,
            gt5_short_point_valid_loss,
        ) = self.point_valid_loss(pred_valid_logits, pred_points, gt_valid, indices, gt_lanes=gt_lanes, return_details=True)
        smooth_loss = self.smooth_loss(pred_points, gt_valid, indices)
        curve_loss = self.curve_loss(pred_points, gt_points, gt_valid, indices)
        (
            count_loss,
            count_under5_loss,
            count_boundary_loss,
            count_score_mean,
            cnt_bound_5under,
            cnt_score,
        ) = self.count_losses(
            pred_logits, batch, gt_valid, target=gt_lanes
        )
        (
            short_side_geom_loss,
            short_side_geom_count,
            short_side_geom_gt4,
            short_side_geom_gt5,
        ) = self.short_side_geom_loss(pred_points, gt_points, gt_valid, indices, gt_lanes)
        (
            spurious_neg_loss,
            spurious_negative_count,
            spur_cand,
            spur_prot,
            spur_final,
            spur_neg,
            spur_cnt_gt3,
            spur_cnt_gt4,
            spur_cnt_gt5,
            spur_neg_gt3,
            spur_neg_gt4,
            spur_neg_gt5,
        ) = self.spurious_negative_loss(
            pred_points, pred_logits, pred_valid_logits, indices, gt_lanes, gt_points, gt_valid
        )

        mask_loss = self._zero_like(pred_points)
        if "aux_mask_logits" in preds and "semantic_mask" in batch:
            target_mask = batch["semantic_mask"].to(device=pred_points.device)
            mask_loss = self.mask_aux_loss(preds["aux_mask_logits"], target_mask)

        edge_loss = self._zero_like(pred_points)
        if "aux_edge_logits" in preds and "edge_mask" in batch:
            target_edge = batch["edge_mask"].to(device=pred_points.device, dtype=pred_points.dtype)
            edge_loss = self.edge_aux_loss(preds["aux_edge_logits"], target_edge)

        total = (
            self.exist_gain * exist_loss
            + self.point_gain * point_loss
            + self.point_valid_gain * point_valid_loss
            + self.smooth_gain * smooth_loss
            + self.curve_gain * curve_loss
            + self.mask_gain * mask_loss
            + self.edge_gain * edge_loss
            + self.count_gain * count_loss
            + self.count_under5_gain * count_under5_loss
        )
        if self.count_boundary_gain != 0.0:
            total = total + self.count_boundary_gain * count_boundary_loss
        if self.short_side_geom_gain != 0.0:
            total = total + self.short_side_geom_gain * short_side_geom_loss
        if self.spurious_neg_gain != 0.0:
            total = total + self.spurious_neg_gain * self.spurious_neg_weight * spurious_neg_loss
        loss_items = torch.stack(
            (
                exist_loss.detach(),
                point_loss.detach(),
                point_valid_loss.detach(),
                smooth_loss.detach(),
                curve_loss.detach(),
                mask_loss.detach(),
                edge_loss.detach(),
                count_loss.detach(),
                count_under5_loss.detach(),
                count_boundary_loss.detach(),
                short_side_geom_loss.detach(),
                short_side_geom_count.detach(),
                short_side_geom_gt4.detach(),
                short_side_geom_gt5.detach(),
                spurious_neg_loss.detach(),
                spurious_negative_count.detach(),
                spur_cand.detach(),
                spur_prot.detach(),
                spur_final.detach(),
                spur_neg.detach(),
                spur_cnt_gt3.detach(),
                spur_cnt_gt4.detach(),
                spur_cnt_gt5.detach(),
                spur_neg_gt3.detach(),
                spur_neg_gt4.detach(),
                spur_neg_gt5.detach(),
                count_score_mean.detach(),
                gt5_short_pos_count.detach(),
                gt5_short_pos_anchor_count.detach(),
                gt5_short_point_valid_loss.detach(),
                cnt_bound_5under.detach(),
                cnt_score.detach(),
            )
        )
        return total, loss_items
