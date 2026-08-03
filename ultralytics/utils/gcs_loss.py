# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions for GCS-YOLO-Lane structured lane training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.gcs_matcher import GCSHungarianMatcher
from ultralytics.utils.gcs_point_loss import aspect_weighted_l1_point_loss
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, assert_gcs_shape, normalize_imgsz
from ultralytics.utils.gcs_full_lane import (
    full_lane_proposal_score_probability,
    probability_to_logit,
)


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
        "boundary_pseudo_neg_loss",
        "boundary_pseudo_count",
        "boundary_pseudo_score_mean",
        "short_candidate_loss",
        "short_candidate_score_loss",
        "short_candidate_pull_loss",
        "short_candidate_pos_count",
        "short_candidate_soft_count",
        "short_candidate_neg_count",
        "short_segment_loss",
        "short_segment_score_loss",
        "short_segment_point_loss",
        "short_segment_pos_count",
        "short_segment_soft_count",
        "short_segment_neg_count",
        "query_count_ce_loss",
        "query_count_acc",
        "query_count_pred_mean",
        "full_lane_proposal_loss",
        "full_lane_point_loss",
        "full_lane_valid_loss",
        "full_lane_interval_loss",
        "full_lane_exist_loss",
        "full_lane_quality_loss",
        "full_lane_match_count",
        "full_lane_unmatched_count",
        "dense_instance_loss",
        "dense_centerline_loss",
        "dense_endpoint_loss",
        "dense_embed_pull_loss",
        "dense_embed_push_loss",
        "dense_centerline_pos",
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
        lambda_spurious_neg: float | None = None,
        count_under5_min_lanes: int | None = None,
        count_boundary_gt4_weight: float | None = None,
        count_boundary_gt5_weight: float | None = None,
        count_boundary_gt5_under_weight: float | None = None,
        count_boundary_margin34: float | None = None,
        count_boundary_margin45: float | None = None,
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
        query_count_ce: float | None = None,
        query_count_min_lanes: int | None = None,
        query_count_max_lanes: int | None = None,
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
        self.spurious_neg_gain = float(
            lambda_spurious_neg if lambda_spurious_neg is not None else self._arg(args, "gcs_spurious_neg", 0.0)
        )
        self.query_count_ce_gain = float(
            query_count_ce if query_count_ce is not None else self._arg(args, "gcs_query_count_ce", 0.0)
        )
        self.query_count_min_lanes = int(
            query_count_min_lanes
            if query_count_min_lanes is not None
            else self._arg(args, "gcs_query_count_min_lanes", 2)
        )
        self.query_count_max_lanes = int(
            query_count_max_lanes
            if query_count_max_lanes is not None
            else self._arg(args, "gcs_query_count_max_lanes", 5)
        )
        self.query_count_classes = self.query_count_max_lanes - self.query_count_min_lanes + 1
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
        if self.query_count_min_lanes <= 0:
            raise ValueError(f"gcs_query_count_min_lanes must be > 0, got {self.query_count_min_lanes}.")
        if self.query_count_max_lanes < self.query_count_min_lanes:
            raise ValueError(
                "gcs_query_count_max_lanes must be >= gcs_query_count_min_lanes, "
                f"got {self.query_count_max_lanes} < {self.query_count_min_lanes}."
            )
        if (self.query_count_min_lanes, self.query_count_max_lanes) != (2, 5):
            raise ValueError(
                "query Count Head currently supports the fixed 2..5 lane-count contract, "
                f"got {self.query_count_min_lanes}..{self.query_count_max_lanes}."
            )
        if self.query_count_classes != 4:
            raise ValueError(f"gcs_query_count_classes must be 4 for 2..5 lanes, got {self.query_count_classes}.")
        if self.query_count_ce_gain < 0.0:
            raise ValueError(f"gcs_query_count_ce must be >= 0, got {self.query_count_ce_gain}.")
        if self.count_boundary_margin34 < 0.0:
            raise ValueError(f"gcs_count_boundary_margin34 must be >= 0, got {self.count_boundary_margin34}.")
        if self.count_boundary_margin45 < 0.0:
            raise ValueError(f"gcs_count_boundary_margin45 must be >= 0, got {self.count_boundary_margin45}.")
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
        self.short_geom_gain = float(self._arg(args, "gcs_short_geom", 0.0))
        self.short_geom_visible_thr = int(self._arg(args, "gcs_short_geom_visible_thr", 10))
        self.short_geom_gt4_weight = float(self._arg(args, "gcs_short_geom_gt4_weight", 1.0))
        self.short_geom_gt5_weight = float(self._arg(args, "gcs_short_geom_gt5_weight", 2.0))
        self.short_geom_max_weight = float(self._arg(args, "gcs_short_geom_max_weight", 3.0))
        self.short_geom_curve = float(self._arg(args, "gcs_short_geom_curve", 1.0))
        self.boundary_pseudo_neg_gain = float(self._arg(args, "gcs_boundary_pseudo_neg", 0.0))
        self.boundary_pseudo_visible_thr = int(self._arg(args, "gcs_boundary_pseudo_visible_thr", 10))
        self.boundary_pseudo_dist_thr = float(self._arg(args, "gcs_boundary_pseudo_dist_thr", 60.0))
        self.boundary_pseudo_valid_thr = float(self._arg(args, "gcs_boundary_pseudo_valid_thr", 0.5))
        self.boundary_pseudo_min_valid = int(self._arg(args, "gcs_boundary_pseudo_min_valid", 3))
        self.boundary_pseudo_gt_count = int(self._arg(args, "gcs_boundary_pseudo_gt_count", 5))
        self.boundary_pseudo_score_thr = float(self._arg(args, "gcs_boundary_pseudo_score_thr", 0.0))
        self.boundary_pseudo_envelope_margin_px = float(
            self._arg(args, "gcs_boundary_pseudo_envelope_margin_px", -1.0)
        )
        self.boundary_pseudo_envelope_ratio_thr = float(
            self._arg(args, "gcs_boundary_pseudo_envelope_ratio_thr", 0.75)
        )
        self.short_candidate_gain = float(self._arg(args, "gcs_short_candidate", 0.0))
        self.short_candidate_topk = int(self._arg(args, "gcs_short_candidate_topk", 4))
        self.short_candidate_visible_thr = int(self._arg(args, "gcs_short_candidate_visible_thr", 10))
        self.short_candidate_min_visible = int(self._arg(args, "gcs_short_candidate_min_visible", 2))
        self.short_candidate_pos_px = float(self._arg(args, "gcs_short_candidate_pos_px", 20.0))
        self.short_candidate_soft_px = float(self._arg(args, "gcs_short_candidate_soft_px", 40.0))
        self.short_candidate_tau = float(self._arg(args, "gcs_short_candidate_tau", 25.0))
        self.short_candidate_pull_weight = float(self._arg(args, "gcs_short_candidate_pull_weight", 0.05))
        self.short_candidate_gt4_weight = float(self._arg(args, "gcs_short_candidate_gt4_weight", 1.0))
        self.short_candidate_gt5_weight = float(self._arg(args, "gcs_short_candidate_gt5_weight", 1.5))
        self.short_candidate_neg_score_thr = float(self._arg(args, "gcs_short_candidate_neg_score_thr", 0.6))
        self.short_segment_gain = float(self._arg(args, "gcs_short_segment", 0.0))
        self.short_segment_topk = int(self._arg(args, "gcs_short_segment_topk", 8))
        self.short_segment_visible_thr = int(self._arg(args, "gcs_short_segment_visible_thr", 10))
        self.short_segment_min_visible = int(self._arg(args, "gcs_short_segment_min_visible", 3))
        self.short_segment_min_overlap = int(self._arg(args, "gcs_short_segment_min_overlap", 3))
        self.short_segment_pos_px = float(self._arg(args, "gcs_short_segment_pos_px", 20.0))
        self.short_segment_soft_px = float(self._arg(args, "gcs_short_segment_soft_px", 40.0))
        self.short_segment_tau = float(self._arg(args, "gcs_short_segment_tau", 25.0))
        self.short_segment_point_weight = float(self._arg(args, "gcs_short_segment_point_weight", 0.05))
        self.short_segment_gt4_weight = float(self._arg(args, "gcs_short_segment_gt4_weight", 1.0))
        self.short_segment_gt5_weight = float(self._arg(args, "gcs_short_segment_gt5_weight", 1.5))
        self.short_segment_neg_score_thr = float(self._arg(args, "gcs_short_segment_neg_score_thr", 0.6))
        self.short_segment_bce_weight = float(self._arg(args, "gcs_short_segment_bce_weight", 1.0))
        self.short_segment_listwise_weight = float(self._arg(args, "gcs_short_segment_listwise_weight", 0.0))
        self.short_segment_query_rank_weight = float(self._arg(args, "gcs_short_segment_query_rank_weight", 0.0))
        self.short_segment_base_choice_weight = float(
            self._arg(args, "gcs_short_segment_base_choice_weight", 0.0)
        )
        self.short_segment_replace_weight = float(self._arg(args, "gcs_short_segment_replace_weight", 0.0))
        self.short_segment_query_replace_weight = float(
            self._arg(args, "gcs_short_segment_query_replace_weight", 0.0)
        )
        self.short_segment_query_replace_neg_weight = float(
            self._arg(args, "gcs_short_segment_query_replace_neg_weight", 0.25)
        )
        self.short_segment_replace_margin_px = float(self._arg(args, "gcs_short_segment_replace_margin_px", 5.0))
        self.short_segment_dense_quality_weight = float(
            self._arg(args, "gcs_short_segment_dense_quality_weight", 0.0)
        )
        self.short_segment_dense_neg_weight = float(self._arg(args, "gcs_short_segment_dense_neg_weight", 0.05))
        self.short_segment_replace_dense_neg_weight = float(
            self._arg(args, "gcs_short_segment_replace_dense_neg_weight", 0.0)
        )
        self.short_segment_unified_choice_weight = float(
            self._arg(args, "gcs_short_segment_unified_choice_weight", 0.0)
        )
        self.short_segment_unified_choice_temperature = float(
            self._arg(args, "gcs_short_segment_unified_choice_temperature", 0.25)
        )
        self.short_segment_unified_choice_base_neg_weight = float(
            self._arg(args, "gcs_short_segment_unified_choice_base_neg_weight", 0.25)
        )
        self.short_segment_candidate_aware_assignment = self._bool_arg(
            self._arg(args, "gcs_short_segment_candidate_aware_assignment", False)
        )
        self.short_segment_listwise_all_candidates = self._bool_arg(
            self._arg(args, "gcs_short_segment_listwise_all_candidates", False)
        )
        self.short_segment_base_preserve = self._bool_arg(
            self._arg(args, "gcs_short_segment_base_preserve", True)
        )
        self.short_segment_matched_assignment = self._bool_arg(
            self._arg(args, "gcs_short_segment_matched_assignment", False)
        )
        self.short_segment_official_quality_target = self._bool_arg(
            self._arg(args, "gcs_short_segment_official_quality_target", False)
        )
        self.short_segment_official_pt_thresh = float(
            self._arg(args, "gcs_short_segment_official_pt_thresh", 0.85)
        )
        self.short_segment_base_valid_thr = float(
            self._arg(args, "gcs_short_segment_base_valid_thr", 0.6)
        )
        self.short_segment_base_choice_all_queries = self._bool_arg(
            self._arg(args, "gcs_short_segment_base_choice_all_queries", False)
        )
        self.short_segment_base_choice_neg_weight = float(
            self._arg(args, "gcs_short_segment_base_choice_neg_weight", 0.25)
        )
        self.full_lane_proposal_gain = float(self._arg(args, "gcs_full_lane_proposal", 0.0))
        self.full_lane_quality_tau = float(self._arg(args, "gcs_full_lane_quality_tau", 25.0))
        self.full_lane_unmatched_valid_weight = float(
            self._arg(args, "gcs_full_lane_unmatched_valid_weight", 0.1)
        )
        self.full_lane_unmatched_weight = float(
            self._arg(args, "gcs_full_lane_unmatched_weight", 1.0)
        )
        self.full_lane_aux_assignment = self._bool_arg(
            self._arg(args, "gcs_full_lane_aux_assignment", False)
        )
        self.full_lane_unified_matching = self._bool_arg(
            self._arg(args, "gcs_full_lane_unified_matching", True)
        )
        self.full_lane_aux_match_min_overlap = int(
            self._arg(args, "gcs_full_lane_aux_match_min_overlap", 2)
        )
        self.full_lane_aux_match_gate_px = float(
            self._arg(args, "gcs_full_lane_aux_match_gate_px", 0.0)
        )
        self.full_lane_hard_focus = self._bool_arg(
            self._arg(args, "gcs_full_lane_hard_focus", False)
        )
        self.full_lane_focus_hit_px = float(
            self._arg(args, "gcs_full_lane_focus_hit_px", 20.0)
        )
        self.full_lane_focus_base_valid_thr = float(
            self._arg(args, "gcs_full_lane_focus_base_valid_thr", 0.6)
        )
        self.full_lane_focus_base_min_coverage = float(
            self._arg(args, "gcs_full_lane_focus_base_min_coverage", 1.0)
        )
        self.full_lane_base_hit_weight = float(
            self._arg(args, "gcs_full_lane_base_hit_weight", 1.0)
        )
        self.full_lane_base_miss_weight = float(
            self._arg(args, "gcs_full_lane_base_miss_weight", 1.0)
        )
        self.full_lane_gt4_weight = float(self._arg(args, "gcs_full_lane_gt4_weight", 1.0))
        self.full_lane_gt5_weight = float(self._arg(args, "gcs_full_lane_gt5_weight", 1.0))
        self.full_lane_short_visible_thr = int(
            self._arg(args, "gcs_full_lane_short_visible_thr", 10)
        )
        self.full_lane_short_visible_weight = float(
            self._arg(args, "gcs_full_lane_short_visible_weight", 1.0)
        )
        self.dense_instance_gain = float(self._arg(args, "gcs_dense_instance", 0.0))
        self.dense_centerline_weight = float(self._arg(args, "gcs_dense_centerline_weight", 1.0))
        self.dense_endpoint_weight = float(self._arg(args, "gcs_dense_endpoint_weight", 1.0))
        self.dense_embed_pull_weight = float(self._arg(args, "gcs_dense_embed_pull_weight", 0.25))
        self.dense_embed_push_weight = float(self._arg(args, "gcs_dense_embed_push_weight", 0.25))
        self.dense_embed_margin = float(self._arg(args, "gcs_dense_embed_margin", 0.5))
        self.dense_sigma_px = float(self._arg(args, "gcs_dense_sigma_px", 3.0))
        self.dense_pos_weight_max = float(self._arg(args, "gcs_dense_pos_weight_max", 50.0))

        if self.short_geom_gain < 0.0:
            raise ValueError(f"gcs_short_geom must be >= 0, got {self.short_geom_gain}.")
        if self.short_geom_visible_thr < 0:
            raise ValueError(f"gcs_short_geom_visible_thr must be >= 0, got {self.short_geom_visible_thr}.")
        if self.short_geom_gt4_weight < 1.0:
            raise ValueError(f"gcs_short_geom_gt4_weight must be >= 1, got {self.short_geom_gt4_weight}.")
        if self.short_geom_gt5_weight < 1.0:
            raise ValueError(f"gcs_short_geom_gt5_weight must be >= 1, got {self.short_geom_gt5_weight}.")
        if self.short_geom_max_weight < 1.0:
            raise ValueError(f"gcs_short_geom_max_weight must be >= 1, got {self.short_geom_max_weight}.")
        if self.short_geom_curve < 0.0:
            raise ValueError(f"gcs_short_geom_curve must be >= 0, got {self.short_geom_curve}.")
        if self.boundary_pseudo_neg_gain < 0.0:
            raise ValueError("gcs_boundary_pseudo_neg must be >= 0.")
        if self.boundary_pseudo_visible_thr < 0:
            raise ValueError("gcs_boundary_pseudo_visible_thr must be >= 0.")
        if self.boundary_pseudo_dist_thr < 0.0:
            raise ValueError("gcs_boundary_pseudo_dist_thr must be >= 0.")
        if not (0.0 <= self.boundary_pseudo_valid_thr <= 1.0):
            raise ValueError("gcs_boundary_pseudo_valid_thr must be in [0, 1].")
        if self.boundary_pseudo_min_valid < 0:
            raise ValueError("gcs_boundary_pseudo_min_valid must be >= 0.")
        if self.boundary_pseudo_gt_count < 0:
            raise ValueError("gcs_boundary_pseudo_gt_count must be >= 0.")
        if self.boundary_pseudo_score_thr < 0.0:
            raise ValueError("gcs_boundary_pseudo_score_thr must be >= 0.")
        if self.boundary_pseudo_envelope_margin_px < -1.0:
            raise ValueError("gcs_boundary_pseudo_envelope_margin_px must be >= -1.0.")
        if not (0.0 <= self.boundary_pseudo_envelope_ratio_thr <= 1.0):
            raise ValueError("gcs_boundary_pseudo_envelope_ratio_thr must be in [0, 1].")
        if self.short_candidate_gain < 0.0:
            raise ValueError("gcs_short_candidate must be >= 0.")
        if self.short_candidate_topk < 1:
            raise ValueError("gcs_short_candidate_topk must be >= 1.")
        if self.short_candidate_visible_thr < 1:
            raise ValueError("gcs_short_candidate_visible_thr must be >= 1.")
        if self.short_candidate_min_visible < 1:
            raise ValueError("gcs_short_candidate_min_visible must be >= 1.")
        if self.short_candidate_pos_px < 0.0:
            raise ValueError("gcs_short_candidate_pos_px must be >= 0.")
        if self.short_candidate_soft_px < self.short_candidate_pos_px:
            raise ValueError("gcs_short_candidate_soft_px must be >= gcs_short_candidate_pos_px.")
        if self.short_candidate_tau <= 0.0:
            raise ValueError("gcs_short_candidate_tau must be > 0.")
        if self.short_candidate_pull_weight < 0.0:
            raise ValueError("gcs_short_candidate_pull_weight must be >= 0.")
        if self.short_candidate_gt4_weight < 0.0:
            raise ValueError("gcs_short_candidate_gt4_weight must be >= 0.")
        if self.short_candidate_gt5_weight < 0.0:
            raise ValueError("gcs_short_candidate_gt5_weight must be >= 0.")
        if not 0.0 <= self.short_candidate_neg_score_thr <= 1.0:
            raise ValueError("gcs_short_candidate_neg_score_thr must be in [0, 1].")
        if self.short_segment_gain < 0.0:
            raise ValueError("gcs_short_segment must be >= 0.")
        if self.short_segment_topk < 1:
            raise ValueError("gcs_short_segment_topk must be >= 1.")
        if self.short_segment_visible_thr < 1:
            raise ValueError("gcs_short_segment_visible_thr must be >= 1.")
        if self.short_segment_min_visible < 1:
            raise ValueError("gcs_short_segment_min_visible must be >= 1.")
        if self.short_segment_min_overlap < 1:
            raise ValueError("gcs_short_segment_min_overlap must be >= 1.")
        if self.short_segment_pos_px < 0.0:
            raise ValueError("gcs_short_segment_pos_px must be >= 0.")
        if self.short_segment_soft_px < self.short_segment_pos_px:
            raise ValueError("gcs_short_segment_soft_px must be >= gcs_short_segment_pos_px.")
        if self.short_segment_tau <= 0.0:
            raise ValueError("gcs_short_segment_tau must be > 0.")
        if self.short_segment_point_weight < 0.0:
            raise ValueError("gcs_short_segment_point_weight must be >= 0.")
        if self.short_segment_gt4_weight < 0.0:
            raise ValueError("gcs_short_segment_gt4_weight must be >= 0.")
        if self.short_segment_gt5_weight < 0.0:
            raise ValueError("gcs_short_segment_gt5_weight must be >= 0.")
        if not 0.0 <= self.short_segment_neg_score_thr <= 1.0:
            raise ValueError("gcs_short_segment_neg_score_thr must be in [0, 1].")
        if self.short_segment_bce_weight < 0.0:
            raise ValueError("gcs_short_segment_bce_weight must be >= 0.")
        if self.short_segment_listwise_weight < 0.0:
            raise ValueError("gcs_short_segment_listwise_weight must be >= 0.")
        if self.short_segment_query_rank_weight < 0.0:
            raise ValueError("gcs_short_segment_query_rank_weight must be >= 0.")
        if self.short_segment_base_choice_weight < 0.0:
            raise ValueError("gcs_short_segment_base_choice_weight must be >= 0.")
        if self.short_segment_replace_weight < 0.0:
            raise ValueError("gcs_short_segment_replace_weight must be >= 0.")
        if self.short_segment_query_replace_weight < 0.0:
            raise ValueError("gcs_short_segment_query_replace_weight must be >= 0.")
        if self.short_segment_query_replace_neg_weight < 0.0:
            raise ValueError("gcs_short_segment_query_replace_neg_weight must be >= 0.")
        if self.short_segment_replace_margin_px < 0.0:
            raise ValueError("gcs_short_segment_replace_margin_px must be >= 0.")
        if self.short_segment_dense_quality_weight < 0.0:
            raise ValueError("gcs_short_segment_dense_quality_weight must be >= 0.")
        if self.short_segment_dense_neg_weight < 0.0:
            raise ValueError("gcs_short_segment_dense_neg_weight must be >= 0.")
        if self.short_segment_replace_dense_neg_weight < 0.0:
            raise ValueError("gcs_short_segment_replace_dense_neg_weight must be >= 0.")
        if self.short_segment_unified_choice_weight < 0.0:
            raise ValueError("gcs_short_segment_unified_choice_weight must be >= 0.")
        if self.short_segment_unified_choice_temperature <= 0.0:
            raise ValueError("gcs_short_segment_unified_choice_temperature must be > 0.")
        if self.short_segment_unified_choice_base_neg_weight < 0.0:
            raise ValueError("gcs_short_segment_unified_choice_base_neg_weight must be >= 0.")
        if not 0.0 < self.short_segment_official_pt_thresh <= 1.0:
            raise ValueError("gcs_short_segment_official_pt_thresh must be in (0, 1].")
        if not 0.0 <= self.short_segment_base_valid_thr <= 1.0:
            raise ValueError("gcs_short_segment_base_valid_thr must be in [0, 1].")
        if self.short_segment_base_choice_neg_weight < 0.0:
            raise ValueError("gcs_short_segment_base_choice_neg_weight must be >= 0.")
        if self.full_lane_proposal_gain < 0.0:
            raise ValueError("gcs_full_lane_proposal must be >= 0.")
        if self.full_lane_quality_tau <= 0.0:
            raise ValueError("gcs_full_lane_quality_tau must be > 0.")
        if self.full_lane_unmatched_valid_weight < 0.0:
            raise ValueError("gcs_full_lane_unmatched_valid_weight must be >= 0.")
        if self.full_lane_unmatched_weight < 0.0:
            raise ValueError("gcs_full_lane_unmatched_weight must be >= 0.")
        if self.full_lane_aux_match_min_overlap < 1:
            raise ValueError("gcs_full_lane_aux_match_min_overlap must be >= 1.")
        if self.full_lane_aux_match_gate_px < 0.0:
            raise ValueError("gcs_full_lane_aux_match_gate_px must be >= 0.")
        if self.full_lane_focus_hit_px < 0.0:
            raise ValueError("gcs_full_lane_focus_hit_px must be >= 0.")
        if not 0.0 <= self.full_lane_focus_base_valid_thr <= 1.0:
            raise ValueError("gcs_full_lane_focus_base_valid_thr must be in [0, 1].")
        if not 0.0 <= self.full_lane_focus_base_min_coverage <= 1.0:
            raise ValueError("gcs_full_lane_focus_base_min_coverage must be in [0, 1].")
        if self.full_lane_base_hit_weight < 0.0:
            raise ValueError("gcs_full_lane_base_hit_weight must be >= 0.")
        if self.full_lane_base_miss_weight < 0.0:
            raise ValueError("gcs_full_lane_base_miss_weight must be >= 0.")
        if self.full_lane_gt4_weight < 0.0:
            raise ValueError("gcs_full_lane_gt4_weight must be >= 0.")
        if self.full_lane_gt5_weight < 0.0:
            raise ValueError("gcs_full_lane_gt5_weight must be >= 0.")
        if self.full_lane_short_visible_thr < 1:
            raise ValueError("gcs_full_lane_short_visible_thr must be >= 1.")
        if self.full_lane_short_visible_weight < 0.0:
            raise ValueError("gcs_full_lane_short_visible_weight must be >= 0.")
        if self.dense_instance_gain < 0.0:
            raise ValueError("gcs_dense_instance must be >= 0.")
        if self.dense_centerline_weight < 0.0 or self.dense_endpoint_weight < 0.0:
            raise ValueError("gcs_dense_centerline_weight and gcs_dense_endpoint_weight must be >= 0.")
        if self.dense_embed_pull_weight < 0.0 or self.dense_embed_push_weight < 0.0:
            raise ValueError("gcs_dense_embed_pull_weight and gcs_dense_embed_push_weight must be >= 0.")
        if self.dense_embed_margin <= 0.0:
            raise ValueError("gcs_dense_embed_margin must be > 0.")
        if self.dense_sigma_px <= 0.0:
            raise ValueError("gcs_dense_sigma_px must be > 0.")
        if self.dense_pos_weight_max < 1.0:
            raise ValueError("gcs_dense_pos_weight_max must be >= 1.")
        if self.full_lane_proposal_gain > 0.0 and not (
            self.full_lane_aux_assignment or self.full_lane_unified_matching
        ):
            raise ValueError(
                "gcs_full_lane_proposal > 0 requires either "
                "gcs_full_lane_aux_assignment=True or gcs_full_lane_unified_matching=True."
            )
        if self.short_candidate_gain > 0.0 and self.short_segment_gain > 0.0:
            raise ValueError(
                "gcs_short_candidate and gcs_short_segment are separate default-off probes; "
                "enable only one candidate/proposal loss in a run."
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
        self.full_lane_aux_matcher = GCSHungarianMatcher(
            cost_point=1.0,
            cost_curve=0.0,
            cost_exist=0.0,
            image_size=self.image_size,
            min_overlap=int(self.full_lane_aux_match_min_overlap),
            max_x_dist=0.0,
            match_gate_px=float(self.full_lane_aux_match_gate_px),
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

    @staticmethod
    def _official_point_threshold(
        target: torch.Tensor,
        valid: torch.Tensor,
        pixel_scale: torch.Tensor,
        pixel_thresh: float,
    ) -> torch.Tensor:
        """Return TuSimple-style x-error threshold adjusted by the GT lane angle."""
        valid_bool = valid > 0.5
        if int(valid_bool.sum().detach().cpu().item()) < 2:
            return target.new_tensor(float(pixel_thresh))
        target_px = target * pixel_scale.view(1, 2)
        x = target_px[valid_bool, 0]
        y = target_px[valid_bool, 1]
        y_centered = y - y.mean()
        denom = (y_centered.square()).sum()
        if bool((denom <= 1.0e-6).item()):
            return target.new_tensor(float(pixel_thresh))
        slope = (y_centered * (x - x.mean())).sum() / denom
        return target.new_tensor(float(pixel_thresh)) * torch.sqrt(1.0 + slope.square())

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

    def _short_geom_lane_weights(
        self,
        gt_valid_b: torch.Tensor,
        gt_count_b: torch.Tensor | int | float,
    ) -> torch.Tensor:
        """Return one geometry weight per GT lane."""
        device = gt_valid_b.device
        dtype = gt_valid_b.dtype
        n = int(gt_valid_b.shape[0])

        if n == 0:
            return torch.ones((0,), device=device, dtype=dtype)

        weights = torch.ones((n,), device=device, dtype=dtype)

        if float(self.short_geom_gain) <= 0.0:
            return weights

        gt_count = int(round(float(torch.as_tensor(gt_count_b).detach().cpu().item())))
        if gt_count == 4:
            target_weight = self.short_geom_gt4_weight
        elif gt_count == 5:
            target_weight = self.short_geom_gt5_weight
        else:
            return weights
        if float(target_weight) <= 1.0:
            return weights

        visible_counts = gt_valid_b.float().sum(dim=1)
        short_mask = visible_counts <= float(self.short_geom_visible_thr)

        boost = 1.0 + float(self.short_geom_gain) * (float(target_weight) - 1.0)
        weights[short_mask] = boost

        return weights.clamp(min=1.0, max=float(self.short_geom_max_weight))

    def point_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aspect-weighted L1 point loss with optional GT4/GT5 short-visible lane weighting."""
        short_geom_enabled = (
            float(self.short_geom_gain) > 0.0
            and int(self.short_geom_visible_thr) > 0
            and max(float(self.short_geom_gt4_weight), float(self.short_geom_gt5_weight)) > 1.0
        )
        if not short_geom_enabled:
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

        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._scale_for(pred_points).view(1, 1, 2)

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)

            pred = pred_points[b, src_idx]
            target_all = gt_points[b].to(device=device, dtype=dtype)
            valid_all = gt_valid[b].to(device=device, dtype=dtype)

            target = target_all[tgt_idx]
            valid = valid_all[tgt_idx]

            if gt_lanes is not None:
                all_lane_weights = self._short_geom_lane_weights(valid_all, gt_lanes[b])
                lane_weights = all_lane_weights[tgt_idx].to(device=device, dtype=dtype)
            else:
                lane_weights = torch.ones((pred.shape[0],), device=device, dtype=dtype)

            mask = (valid > 0.5).to(dtype=dtype)
            valid_counts = mask.sum(dim=1)
            active = valid_counts > 0
            boosted = lane_weights > 1.0 + 1e-6
            active_boosted = boosted & active
            if not bool(active_boosted.any()):
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
                continue

            point_err = ((pred - target).abs() * scale).sum(dim=-1)
            lane_loss = (point_err * mask).sum(dim=1) / valid_counts.clamp_min(1.0)

            lane_loss = lane_loss[active]
            lane_weights = lane_weights[active]
            losses.append((lane_loss * lane_weights).sum() / lane_weights.sum().clamp_min(1.0))

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

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
        gt_lanes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Adaptive curvature-aware loss with optional GT4/GT5 short-visible lane weighting."""
        if pred_points.shape[2] < 3:
            return self._zero_like(pred_points)

        short_curve_enabled = (
            float(self.short_geom_gain) > 0.0
            and float(self.short_geom_curve) > 0.0
            and int(self.short_geom_visible_thr) > 0
            and max(float(self.short_geom_gt4_weight), float(self.short_geom_gt5_weight)) > 1.0
        )
        if not short_curve_enabled:
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

        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._pixel_scale_for(pred_points)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)

            pred = pred_points[b, src_idx]
            target_all = gt_points[b].to(device=device, dtype=dtype)
            valid_all = gt_valid[b].to(device=device, dtype=dtype)

            target = target_all[tgt_idx]
            valid = valid_all[tgt_idx]

            triplet_valid = valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]

            pred_px = pred * scale
            target_px = target * scale

            pred_lap = pred_px[:, 2:] - 2.0 * pred_px[:, 1:-1] + pred_px[:, :-2]
            gt_lap = target_px[:, 2:] - 2.0 * target_px[:, 1:-1] + target_px[:, :-2]

            gt_curve_mag = torch.norm(gt_lap.detach(), dim=-1)
            curve_weight = (1.0 + self.curve_alpha * gt_curve_mag).clamp(max=self.curve_weight_max)

            loss = F.smooth_l1_loss(pred_lap, gt_lap, reduction="none").sum(dim=-1)
            loss = loss * triplet_valid * curve_weight
            triplet_counts = triplet_valid.sum(dim=1)
            active = triplet_counts > 0

            if gt_lanes is not None and float(self.short_geom_curve) > 0.0:
                all_lane_weights = self._short_geom_lane_weights(valid_all, gt_lanes[b])
                base_weights = all_lane_weights[tgt_idx].to(device=device, dtype=dtype)
                lane_weights = 1.0 + float(self.short_geom_curve) * (base_weights - 1.0)
            else:
                lane_weights = torch.ones((pred.shape[0],), device=device, dtype=dtype)

            boosted = lane_weights > 1.0 + 1e-6
            active_boosted = boosted & active
            if not bool(active_boosted.any()):
                valid_sum = triplet_valid.sum().clamp_min(1.0)
                losses.append(loss.sum() / valid_sum)
                continue

            lane_curve_loss = loss.sum(dim=1) / triplet_counts.clamp_min(1.0)
            lane_curve_loss = lane_curve_loss[active]
            lane_weights = lane_weights[active]
            losses.append((lane_curve_loss * lane_weights).sum() / lane_weights.sum().clamp_min(1.0))

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

    def query_count_ce_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict,
        gt_valid: list[torch.Tensor],
        target_count: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return query-mode explicit count CE and classifier diagnostics."""
        pred_count_logits = preds.get("pred_count_logits")
        pred_points = preds["pred_points"]

        if pred_count_logits is None:
            zero = self._zero_like(pred_points)
            return zero, zero, zero

        if pred_count_logits.ndim != 2:
            raise ValueError(f"pred_count_logits must have shape B x C, got {tuple(pred_count_logits.shape)}.")

        expected_classes = int(self.query_count_classes)
        if pred_count_logits.shape[1] != expected_classes:
            raise ValueError(
                f"pred_count_logits C must be {expected_classes}, got {pred_count_logits.shape[1]}."
            )

        if target_count is None:
            pred_logits = preds["pred_logits"]
            if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
                pred_logits = pred_logits.squeeze(-1)
            target_count = self.target_lane_count(pred_logits, batch, gt_valid)

        label = target_count.round().long() - int(self.query_count_min_lanes)
        label = label.clamp(0, expected_classes - 1)

        loss = F.cross_entropy(pred_count_logits, label)
        pred_cls = pred_count_logits.detach().argmax(dim=1)
        acc = (pred_cls == label).float().mean()
        pred_count_mean = (pred_cls.float() + float(self.query_count_min_lanes)).mean()

        return loss, acc, pred_count_mean

    def short_candidate_loss(
        self,
        preds: dict[str, torch.Tensor],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        gt_lanes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train the default-off lateral candidate geometry selector on short GT4/GT5 lanes only."""
        pred_points = preds["pred_points"]
        zero = self._zero_like(pred_points)
        zero_count = pred_points.new_zeros(())
        candidate_points = preds.get("pred_short_candidate_points")
        candidate_logits = preds.get("pred_short_candidate_logits")
        if float(self.short_candidate_gain) == 0.0:
            return zero, zero, zero, zero_count, zero_count, zero_count
        if candidate_points is None or candidate_logits is None:
            raise KeyError(
                "gcs_short_candidate > 0 requires a GCS head that emits "
                "pred_short_candidate_points and pred_short_candidate_logits."
            )
        if candidate_points.ndim != 5 or candidate_points.shape[-1] != 2:
            raise ValueError(
                "pred_short_candidate_points must have shape B x Q x M x K x 2, "
                f"got {tuple(candidate_points.shape)}."
            )
        if candidate_logits.shape != candidate_points.shape[:3]:
            raise ValueError(
                "pred_short_candidate_logits must have shape B x Q x M matching candidate points, "
                f"got {tuple(candidate_logits.shape)} vs {tuple(candidate_points.shape[:3])}."
            )
        if candidate_points.shape[0] != pred_points.shape[0] or candidate_points.shape[1] != pred_points.shape[1]:
            raise ValueError(
                "pred_short_candidate_points B,Q must match pred_points, "
                f"got {tuple(candidate_points.shape[:2])} vs {tuple(pred_points.shape[:2])}."
            )
        if candidate_points.shape[3] != pred_points.shape[2]:
            raise ValueError(
                "pred_short_candidate_points K must match pred_points, "
                f"got {candidate_points.shape[3]} vs {pred_points.shape[2]}."
            )
        zero = candidate_logits.sum() * 0.0

        device = pred_points.device
        dtype = pred_points.dtype
        scale = self._pixel_scale_for(pred_points).view(1, 1, 2)
        bsz, num_queries, num_candidates, num_points, _ = candidate_points.shape
        flat_count = int(num_queries * num_candidates)
        topk = min(int(self.short_candidate_topk), flat_count)
        pos_px = float(self.short_candidate_pos_px)
        soft_px = float(self.short_candidate_soft_px)
        tau = max(float(self.short_candidate_tau), 1e-6)
        min_visible = int(self.short_candidate_min_visible)
        visible_thr = int(self.short_candidate_visible_thr)

        score_losses: list[torch.Tensor] = []
        pull_losses: list[torch.Tensor] = []
        pos_count = 0
        soft_count = 0
        neg_count = 0

        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=dtype).reshape(-1)
        if gt_lanes.numel() != bsz:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={bsz}.")

        for b in range(bsz):
            gt_count = int(round(float(gt_lanes[b].detach().cpu().item())))
            if gt_count not in {4, 5}:
                continue
            lane_weight = float(self.short_candidate_gt4_weight if gt_count == 4 else self.short_candidate_gt5_weight)
            if lane_weight <= 0.0:
                continue

            target_points_b = gt_points[b].to(device=device, dtype=dtype)
            target_valid_b = gt_valid[b].to(device=device, dtype=dtype)
            if target_points_b.numel() == 0:
                continue
            visible_counts = target_valid_b.sum(dim=1)
            short_lane_mask = (visible_counts >= float(min_visible)) & (visible_counts <= float(visible_thr))
            short_lane_indices = torch.nonzero(short_lane_mask, as_tuple=False).flatten()
            if short_lane_indices.numel() == 0:
                continue

            candidates_b = candidate_points[b].reshape(flat_count, num_points, 2)
            logits_b = candidate_logits[b].reshape(flat_count)
            score_target = torch.zeros((flat_count,), device=device, dtype=dtype)
            score_weight = torch.zeros((flat_count,), device=device, dtype=dtype)
            min_ape = torch.full((flat_count,), float("inf"), device=device, dtype=dtype)

            for lane_idx in short_lane_indices.tolist():
                target = target_points_b[lane_idx]
                valid = target_valid_b[lane_idx].clamp(0.0, 1.0)
                valid_count = valid.sum().clamp_min(1.0)
                point_error = torch.norm((candidates_b.detach() - target.view(1, num_points, 2)) * scale, dim=-1)
                ape = (point_error * valid.view(1, num_points)).sum(dim=1) / valid_count
                min_ape = torch.minimum(min_ape, ape)
                top_vals, top_idx = torch.topk(ape, k=topk, largest=False)
                active = top_vals <= soft_px
                if not bool(active.any()):
                    continue

                selected = top_idx[active]
                selected_ape = top_vals[active]
                coverage = (valid_count / float(num_points)).clamp(0.0, 1.0)
                soft_target = torch.exp(-selected_ape / tau) * coverage
                target_score = torch.where(
                    selected_ape <= pos_px,
                    torch.ones_like(selected_ape),
                    soft_target,
                ).clamp(0.0, 1.0)
                score_target[selected] = torch.maximum(score_target[selected], target_score.to(dtype=dtype))
                score_weight[selected] = torch.maximum(
                    score_weight[selected],
                    score_weight.new_full((selected.numel(),), lane_weight),
                )
                pos_count += int((selected_ape <= pos_px).sum().detach().cpu().item())
                soft_count += int(((selected_ape > pos_px) & (selected_ape <= soft_px)).sum().detach().cpu().item())

                if float(self.short_candidate_pull_weight) > 0.0:
                    selected_candidates = candidates_b[selected]
                    target_px = target.view(1, num_points, 2) * scale
                    candidate_px = selected_candidates * scale
                    pull = F.smooth_l1_loss(
                        candidate_px,
                        target_px.expand_as(candidate_px),
                        reduction="none",
                    ).sum(dim=-1)
                    pull = (pull * valid.view(1, num_points)).sum(dim=1) / valid_count
                    pull_losses.append((pull * target_score.detach() * lane_weight).mean())

            neg_mask = (score_weight <= 0.0) & (min_ape > soft_px)
            if float(self.short_candidate_neg_score_thr) > 0.0:
                neg_mask = neg_mask & (logits_b.detach().sigmoid() >= float(self.short_candidate_neg_score_thr))
            if bool(neg_mask.any()):
                score_weight[neg_mask] = 1.0
                neg_count += int(neg_mask.sum().detach().cpu().item())

            active_weight = score_weight > 0.0
            if bool(active_weight.any()):
                bce = F.binary_cross_entropy_with_logits(logits_b, score_target, reduction="none")
                score_losses.append((bce * score_weight).sum() / score_weight.sum().clamp_min(1.0))

        score_loss = torch.stack(score_losses).mean() if score_losses else zero
        pull_loss = torch.stack(pull_losses).mean() if pull_losses else zero
        total = score_loss + float(self.short_candidate_pull_weight) * pull_loss
        return (
            total,
            score_loss,
            pull_loss,
            pred_points.new_tensor(float(pos_count)),
            pred_points.new_tensor(float(soft_count)),
            pred_points.new_tensor(float(neg_count)),
        )

    def short_segment_loss(
        self,
        preds: dict[str, torch.Tensor],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        gt_lanes: torch.Tensor,
        indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train default-off local short-window segment proposals on short GT4/GT5 lanes only."""
        pred_points = preds["pred_points"]
        zero = self._zero_like(pred_points)
        zero_count = pred_points.new_zeros(())
        pred_valid_logits = preds.get("pred_valid_logits")
        segment_points = preds.get("pred_short_segment_points")
        segment_logits = preds.get("pred_short_segment_logits")
        replace_logits = preds.get("pred_short_segment_replace_logits")
        query_replace_logits = preds.get("pred_short_segment_query_replace_logits")
        unified_choice_logits = preds.get("pred_short_segment_choice_logits")
        window_mask = preds.get("pred_short_segment_window_mask")
        if float(self.short_segment_gain) == 0.0:
            return zero, zero, zero, zero_count, zero_count, zero_count
        if segment_points is None or segment_logits is None or window_mask is None:
            raise KeyError(
                "gcs_short_segment > 0 requires a GCS head that emits pred_short_segment_points, "
                "pred_short_segment_logits, and pred_short_segment_window_mask."
            )
        if segment_points.ndim != 5 or segment_points.shape[-1] != 2:
            raise ValueError(
                "pred_short_segment_points must have shape B x Q x S x K x 2, "
                f"got {tuple(segment_points.shape)}."
            )
        if segment_logits.shape != segment_points.shape[:3]:
            raise ValueError(
                "pred_short_segment_logits must have shape B x Q x S matching segment points, "
                f"got {tuple(segment_logits.shape)} vs {tuple(segment_points.shape[:3])}."
            )
        if pred_valid_logits is not None and (
            pred_valid_logits.ndim != 3
            or tuple(pred_valid_logits.shape) != (segment_points.shape[0], segment_points.shape[1], segment_points.shape[3])
        ):
            raise ValueError(
                "pred_valid_logits must have shape B x Q x K when used by short-segment loss, "
                f"got {tuple(pred_valid_logits.shape)} vs "
                f"{(segment_points.shape[0], segment_points.shape[1], segment_points.shape[3])}."
            )
        if self.short_segment_official_quality_target and pred_valid_logits is None:
            raise KeyError(
                "gcs_short_segment_official_quality_target requires pred_valid_logits for base/no-replace quality."
            )
        if float(self.short_segment_replace_weight) > 0.0:
            if replace_logits is None:
                raise KeyError(
                    "gcs_short_segment_replace_weight > 0 requires pred_short_segment_replace_logits. "
                    "Use the local-segment-proposal-v5 YAML."
                )
            if replace_logits.shape != segment_points.shape[:3]:
                raise ValueError(
                    "pred_short_segment_replace_logits must have shape B x Q x S matching segment points, "
                    f"got {tuple(replace_logits.shape)} vs {tuple(segment_points.shape[:3])}."
                )
        elif replace_logits is not None and replace_logits.shape != segment_points.shape[:3]:
            raise ValueError(
                "pred_short_segment_replace_logits must have shape B x Q x S matching segment points, "
                f"got {tuple(replace_logits.shape)} vs {tuple(segment_points.shape[:3])}."
            )
        if float(self.short_segment_query_replace_weight) > 0.0:
            if query_replace_logits is None:
                raise KeyError(
                    "gcs_short_segment_query_replace_weight > 0 requires "
                    "pred_short_segment_query_replace_logits. Use the local-segment-proposal-v8 YAML."
                )
            if query_replace_logits.shape != segment_points.shape[:2]:
                raise ValueError(
                    "pred_short_segment_query_replace_logits must have shape B x Q matching segment points, "
                    f"got {tuple(query_replace_logits.shape)} vs {tuple(segment_points.shape[:2])}."
                )
        elif query_replace_logits is not None and query_replace_logits.shape != segment_points.shape[:2]:
            raise ValueError(
                "pred_short_segment_query_replace_logits must have shape B x Q matching segment points, "
                f"got {tuple(query_replace_logits.shape)} vs {tuple(segment_points.shape[:2])}."
            )
        if float(self.short_segment_unified_choice_weight) > 0.0:
            if unified_choice_logits is None:
                raise KeyError(
                    "gcs_short_segment_unified_choice_weight > 0 requires "
                    "pred_short_segment_choice_logits. Use the local-segment-proposal-v12 YAML."
                )
            expected_choice_shape = (segment_points.shape[0], segment_points.shape[1], segment_points.shape[2] + 1)
            if tuple(unified_choice_logits.shape) != expected_choice_shape:
                raise ValueError(
                    "pred_short_segment_choice_logits must have shape B x Q x (S+1), "
                    f"got {tuple(unified_choice_logits.shape)} vs {expected_choice_shape}."
                )
        elif unified_choice_logits is not None:
            expected_choice_shape = (segment_points.shape[0], segment_points.shape[1], segment_points.shape[2] + 1)
            if tuple(unified_choice_logits.shape) != expected_choice_shape:
                raise ValueError(
                    "pred_short_segment_choice_logits must have shape B x Q x (S+1), "
                    f"got {tuple(unified_choice_logits.shape)} vs {expected_choice_shape}."
                )
        if segment_points.shape[0] != pred_points.shape[0] or segment_points.shape[1] != pred_points.shape[1]:
            raise ValueError(
                "pred_short_segment_points B,Q must match pred_points, "
                f"got {tuple(segment_points.shape[:2])} vs {tuple(pred_points.shape[:2])}."
            )
        if segment_points.shape[3] != pred_points.shape[2]:
            raise ValueError(
                "pred_short_segment_points K must match pred_points, "
                f"got {segment_points.shape[3]} vs {pred_points.shape[2]}."
            )

        device = pred_points.device
        dtype = pred_points.dtype
        bsz, num_queries, num_segments, num_points, _ = segment_points.shape
        if window_mask.ndim == 2:
            if tuple(window_mask.shape) != (num_segments, num_points):
                raise ValueError(
                    "pred_short_segment_window_mask must have shape S x K or B x Q x S x K, "
                    f"got {tuple(window_mask.shape)} for S,K={(num_segments, num_points)}."
                )
            window_mask_bq = window_mask.to(device=device).bool().view(1, 1, num_segments, num_points)
            window_mask_bq = window_mask_bq.expand(bsz, num_queries, -1, -1)
        elif window_mask.ndim == 4:
            if tuple(window_mask.shape) != (bsz, num_queries, num_segments, num_points):
                raise ValueError(
                    "pred_short_segment_window_mask B,Q,S,K must match segment points, "
                    f"got {tuple(window_mask.shape)} vs {(bsz, num_queries, num_segments, num_points)}."
                )
            window_mask_bq = window_mask.to(device=device).bool()
        else:
            raise ValueError(
                "pred_short_segment_window_mask must have shape S x K or B x Q x S x K, "
                f"got {tuple(window_mask.shape)}."
            )

        zero = segment_logits.sum() * 0.0 + segment_points.sum() * 0.0
        if replace_logits is not None:
            zero = zero + replace_logits.sum() * 0.0
        if query_replace_logits is not None:
            zero = zero + query_replace_logits.sum() * 0.0
        if unified_choice_logits is not None:
            zero = zero + unified_choice_logits.sum() * 0.0
        flat_count = int(num_queries * num_segments)
        topk = min(int(self.short_segment_topk), flat_count)
        pos_px = float(self.short_segment_pos_px)
        soft_px = float(self.short_segment_soft_px)
        tau = max(float(self.short_segment_tau), 1e-6)
        min_visible = int(self.short_segment_min_visible)
        visible_thr = int(self.short_segment_visible_thr)
        min_overlap = int(self.short_segment_min_overlap)
        scale = self._pixel_scale_for(pred_points).view(1, 1, 2)
        pixel_scale = scale.reshape(-1)
        official_quality_target = bool(self.short_segment_official_quality_target)
        official_pt_thresh = float(self.short_segment_official_pt_thresh)
        base_valid_thr = float(self.short_segment_base_valid_thr)
        base_choice_all_queries = bool(self.short_segment_base_choice_all_queries)
        unified_choice_active = (
            float(self.short_segment_unified_choice_weight) > 0.0 and unified_choice_logits is not None
        )
        unified_choice_temperature = float(self.short_segment_unified_choice_temperature)
        unified_choice_base_neg_weight = float(self.short_segment_unified_choice_base_neg_weight)
        candidate_aware_assignment = bool(self.short_segment_candidate_aware_assignment)

        bce_losses: list[torch.Tensor] = []
        dense_quality_losses: list[torch.Tensor] = []
        listwise_losses: list[torch.Tensor] = []
        query_rank_losses: list[torch.Tensor] = []
        base_choice_losses: list[torch.Tensor] = []
        unified_choice_losses: list[torch.Tensor] = []
        replace_losses: list[torch.Tensor] = []
        query_replace_losses: list[torch.Tensor] = []
        point_losses: list[torch.Tensor] = []
        pos_count = 0
        soft_count = 0
        neg_count = 0

        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=dtype).reshape(-1)
        if gt_lanes.numel() != bsz:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={bsz}.")
        if self.short_segment_matched_assignment or float(self.short_segment_unified_choice_weight) > 0.0:
            if indices is None:
                raise ValueError(
                    "short-segment matched/unified choice supervision requires Hungarian matcher indices."
                )
            if len(indices) != bsz:
                raise ValueError(
                    "Hungarian matcher indices must contain one pair per image, "
                    f"got {len(indices)} vs B={bsz}."
                )

        for b in range(bsz):
            gt_count = int(round(float(gt_lanes[b].detach().cpu().item())))
            if gt_count not in {4, 5}:
                continue
            lane_weight = float(self.short_segment_gt4_weight if gt_count == 4 else self.short_segment_gt5_weight)
            if lane_weight <= 0.0:
                continue

            target_points_b = gt_points[b].to(device=device, dtype=dtype)
            target_valid_b = gt_valid[b].to(device=device, dtype=dtype)
            if target_points_b.numel() == 0:
                continue
            visible_counts = target_valid_b.sum(dim=1)
            short_lane_mask = (visible_counts >= float(min_visible)) & (visible_counts <= float(visible_thr))
            short_lane_indices = torch.nonzero(short_lane_mask, as_tuple=False).flatten()
            if short_lane_indices.numel() == 0:
                continue

            segments_b = segment_points[b].reshape(flat_count, num_points, 2)
            logits_b = segment_logits[b].reshape(flat_count)
            logits_qs_b = segment_logits[b]
            unified_choice_logits_b = unified_choice_logits[b] if unified_choice_active else None
            replace_logits_b = replace_logits[b].reshape(flat_count) if replace_logits is not None else None
            query_replace_logits_b = query_replace_logits[b] if query_replace_logits is not None else None
            masks_b = window_mask_bq[b].reshape(flat_count, num_points)
            base_valid_mask_b = None
            if official_quality_target:
                base_valid_mask_b = (
                    pred_valid_logits[b].detach().float().sigmoid() >= base_valid_thr
                )
            score_target = torch.zeros((flat_count,), device=device, dtype=dtype)
            score_weight = torch.zeros((flat_count,), device=device, dtype=dtype)
            min_ape = torch.full((flat_count,), float("inf"), device=device, dtype=dtype)
            min_ape_lane_weight = torch.zeros((flat_count,), device=device, dtype=dtype)
            official_quality = torch.zeros((flat_count,), device=device, dtype=dtype)
            official_quality_lane_weight = torch.zeros((flat_count,), device=device, dtype=dtype)
            official_quality_best_idx = torch.full((flat_count,), -1, device=device, dtype=torch.long)
            query_replace_target = torch.zeros((num_queries,), device=device, dtype=dtype)
            query_replace_weight = torch.zeros((num_queries,), device=device, dtype=dtype)
            query_replace_pos = torch.zeros((num_queries,), device=device, dtype=torch.bool)
            base_choice_target = torch.zeros((num_queries,), device=device, dtype=torch.long)
            base_choice_weight = torch.zeros((num_queries,), device=device, dtype=dtype)
            base_hit_any = torch.zeros((num_queries,), device=device, dtype=torch.bool)
            base_official_quality = torch.zeros((num_queries,), device=device, dtype=dtype)
            base_official_lane_weight = torch.zeros((num_queries,), device=device, dtype=dtype)
            matched_query_for_lane: dict[int, int] = {}
            if self.short_segment_matched_assignment or float(self.short_segment_unified_choice_weight) > 0.0:
                src_idx, tgt_idx = indices[b]
                src_idx = src_idx.to(device=device, dtype=torch.long).reshape(-1)
                tgt_idx = tgt_idx.to(device=device, dtype=torch.long).reshape(-1)
                if src_idx.numel() != tgt_idx.numel():
                    raise ValueError(
                        "Hungarian matcher source/target index lengths must match, "
                        f"got {src_idx.numel()} vs {tgt_idx.numel()}."
                    )
                for query_idx, lane_idx in zip(src_idx.tolist(), tgt_idx.tolist()):
                    if 0 <= int(query_idx) < num_queries and 0 <= int(lane_idx) < int(target_points_b.shape[0]):
                        matched_query_for_lane[int(lane_idx)] = int(query_idx)

            def update_query_replace_weight(mask: torch.Tensor, value: float) -> None:
                nonlocal query_replace_weight
                if not bool(mask.any()):
                    return
                count = int(mask.sum().detach().cpu().item())
                query_replace_weight[mask] = torch.maximum(
                    query_replace_weight[mask],
                    query_replace_weight.new_full((count,), float(value)),
                )

            def update_base_choice(query_idx: int, target_idx: int, value: float) -> None:
                if query_idx < 0 or query_idx >= num_queries:
                    return
                current = float(base_choice_weight[query_idx].detach().cpu().item())
                if float(value) >= current:
                    base_choice_target[query_idx] = int(target_idx)
                base_choice_weight[query_idx] = torch.maximum(
                    base_choice_weight[query_idx],
                    base_choice_weight.new_tensor(float(value)),
                )

            train_base_choice = float(self.short_segment_base_choice_weight) > 0.0
            train_query_replace = query_replace_logits_b is not None and float(self.short_segment_query_replace_weight) > 0.0
            if train_query_replace or (train_base_choice and not self.short_segment_matched_assignment):
                for all_lane_idx in range(int(target_points_b.shape[0])):
                    valid_all = target_valid_b[all_lane_idx].clamp(0.0, 1.0)
                    valid_count_all = valid_all.sum()
                    if float(valid_count_all.detach().cpu().item()) < float(min_overlap):
                        continue
                    target_all = target_points_b[all_lane_idx]
                    base_error_all = torch.norm(
                        (pred_points[b].detach() - target_all.view(1, num_points, 2)) * scale,
                        dim=-1,
                    )
                    base_ape_all = (base_error_all * valid_all.view(1, num_points)).sum(dim=1) / valid_count_all.clamp_min(
                        1.0
                    )
                    base_hit_any = base_hit_any | (base_ape_all <= pos_px)

            for lane_idx in short_lane_indices.tolist():
                target = target_points_b[lane_idx]
                valid = target_valid_b[lane_idx].clamp(0.0, 1.0)
                valid_bool = valid > 0.5
                valid_count = valid.sum().clamp_min(1.0)
                overlap_mask = masks_b & valid_bool.view(1, num_points)
                overlap_counts = overlap_mask.sum(dim=1)
                enough_overlap = overlap_counts >= min_overlap
                base_point_error = torch.norm(
                    (pred_points[b].detach() - target.view(1, num_points, 2)) * scale,
                    dim=-1,
                )
                base_ape = (base_point_error * valid.view(1, num_points)).sum(dim=1) / valid_count
                base_best_ape = base_ape.min()

                if official_quality_target:
                    official_threshold = self._official_point_threshold(
                        target,
                        valid,
                        pixel_scale,
                        pos_px,
                    )
                    official_valid_count = valid_bool.sum().to(dtype).clamp_min(1.0)
                    segment_x_error = (
                        (segments_b.detach()[..., 0] - target.view(1, num_points, 2)[..., 0]).abs()
                        * pixel_scale[0]
                    )
                    official_hits = (
                        (segment_x_error < official_threshold)
                        & overlap_mask
                    ).sum(dim=1).to(dtype)
                    official_quality_b = torch.where(
                        enough_overlap,
                        official_hits / official_valid_count,
                        torch.zeros_like(official_hits),
                    ).clamp(0.0, 1.0)
                    better_official = official_quality_b > official_quality
                    official_quality_lane_weight = torch.where(
                        better_official,
                        official_quality_lane_weight.new_full((flat_count,), lane_weight),
                        official_quality_lane_weight,
                    )
                    official_quality_best_idx = torch.where(
                        better_official,
                        torch.arange(flat_count, device=device, dtype=torch.long),
                        official_quality_best_idx,
                    )
                    official_quality = torch.maximum(official_quality, official_quality_b)

                    base_x_error = (
                        (pred_points[b].detach()[..., 0] - target.view(1, num_points, 2)[..., 0]).abs()
                        * pixel_scale[0]
                    )
                    base_overlap = base_valid_mask_b & valid_bool.view(1, num_points)
                    base_hits = (
                        (base_x_error < official_threshold) & base_overlap
                    ).sum(dim=1).to(dtype)
                    base_quality_b = (base_hits / official_valid_count).clamp(0.0, 1.0)
                    better_base = base_quality_b > base_official_quality
                    base_official_lane_weight = torch.where(
                        better_base,
                        base_official_lane_weight.new_full((num_queries,), lane_weight),
                        base_official_lane_weight,
                    )
                    base_official_quality = torch.maximum(base_official_quality, base_quality_b)
                    base_hit_any = base_hit_any | (base_quality_b >= official_pt_thresh)

                if not bool(enough_overlap.any()):
                    if train_base_choice and self.short_segment_matched_assignment and not base_choice_all_queries:
                        pos_query = matched_query_for_lane.get(int(lane_idx))
                        if pos_query is not None and bool(torch.isfinite(base_ape[pos_query]).item()):
                            choice_weight = lane_weight * (
                                2.0 if bool((base_ape[pos_query] <= pos_px).item()) else 1.0
                            )
                            update_base_choice(pos_query, 0, choice_weight)
                    continue

                point_error = torch.norm((segments_b.detach() - target.view(1, num_points, 2)) * scale, dim=-1)
                overlap_weight = overlap_mask.to(dtype=dtype)
                ape = (point_error * overlap_weight).sum(dim=1) / overlap_counts.to(dtype=dtype).clamp_min(1.0)
                ape = torch.where(enough_overlap, ape, torch.full_like(ape, float("inf")))
                better_min = ape < min_ape
                min_ape_lane_weight = torch.where(
                    better_min,
                    min_ape_lane_weight.new_full((flat_count,), lane_weight),
                    min_ape_lane_weight,
                )
                min_ape = torch.minimum(min_ape, ape)

                if float(self.short_segment_listwise_weight) > 0.0:
                    best_idx = int(torch.argmin(ape).detach().cpu().item())
                    if bool(self.short_segment_listwise_all_candidates):
                        masked_logits = logits_b.view(1, -1)
                    else:
                        masked_logits = logits_b.masked_fill(~enough_overlap, -1.0e4).view(1, -1)
                    listwise = F.cross_entropy(
                        masked_logits,
                        torch.tensor([best_idx], device=device, dtype=torch.long),
                    )
                    listwise_losses.append(listwise * lane_weight)

                query_best_ape, query_best_m = ape.view(num_queries, num_segments).min(dim=1)
                if float(self.short_segment_query_rank_weight) > 0.0:
                    rankable_queries = torch.nonzero(
                        torch.isfinite(query_best_ape) & (query_best_ape <= soft_px),
                        as_tuple=False,
                    ).flatten()
                    for query_idx in rankable_queries.tolist():
                        target_m = int(query_best_m[query_idx].detach().cpu().item())
                        query_rank = F.cross_entropy(
                            logits_qs_b[query_idx].view(1, -1),
                            torch.tensor([target_m], device=device, dtype=torch.long),
                        )
                        query_rank_losses.append(query_rank * lane_weight)

                if train_base_choice and not base_choice_all_queries and torch.isfinite(base_best_ape):
                    margin = float(self.short_segment_replace_margin_px)
                    if self.short_segment_matched_assignment:
                        pos_query = matched_query_for_lane.get(int(lane_idx))
                        if pos_query is not None and bool(torch.isfinite(base_ape[pos_query]).item()):
                            matched_base_ape = base_ape[pos_query]
                            matched_segment_ape = query_best_ape[pos_query]
                            target_idx = 0
                            choice_weight = lane_weight
                            if bool((matched_base_ape <= pos_px).item()):
                                # A base lane that already passes the 20 px gate is protected.
                                choice_weight = lane_weight * 2.0
                            elif bool(torch.isfinite(matched_segment_ape).item()) and bool(
                                (
                                    (matched_segment_ape <= soft_px)
                                    & (
                                        ((matched_segment_ape + margin) < matched_base_ape)
                                        | (matched_segment_ape <= pos_px)
                                    )
                                ).item()
                            ):
                                pos_segment = int(query_best_m[pos_query].detach().cpu().item())
                                target_idx = pos_segment + 1
                            update_base_choice(pos_query, target_idx, choice_weight)
                    else:
                        finite_query = torch.isfinite(query_best_ape)
                        improves_query = finite_query & (
                            ((query_best_ape + margin) < base_ape)
                            | ((base_ape > pos_px) & (query_best_ape <= pos_px))
                        )
                        improves_query = improves_query & (query_best_ape <= soft_px)
                        if bool(improves_query.any()):
                            pos_values = torch.where(
                                improves_query,
                                query_best_ape,
                                torch.full_like(query_best_ape, float("inf")),
                            )
                            pos_query = int(torch.argmin(pos_values).detach().cpu().item())
                            if not bool(base_hit_any[pos_query].item()):
                                pos_segment = int(query_best_m[pos_query].detach().cpu().item())
                                update_base_choice(pos_query, pos_segment + 1, lane_weight)

                if (
                    replace_logits_b is not None
                    and float(self.short_segment_replace_weight) > 0.0
                    and torch.isfinite(base_best_ape)
                ):
                    margin = float(self.short_segment_replace_margin_px)
                    improves = enough_overlap & (
                        ((ape + margin) < base_best_ape)
                        | ((base_best_ape > pos_px) & (ape <= pos_px))
                    )
                    replace_target = torch.zeros((flat_count,), device=device, dtype=dtype)
                    replace_weight = torch.zeros((flat_count,), device=device, dtype=dtype)

                    pos_budget = min(topk, int(improves.sum().detach().cpu().item()))
                    if pos_budget > 0:
                        pos_values = torch.where(improves, ape, torch.full_like(ape, float("inf")))
                        _, pos_idx = torch.topk(pos_values, k=pos_budget, largest=False)
                        replace_target[pos_idx] = 1.0
                        replace_weight[pos_idx] = lane_weight

                    if bool(self.short_segment_base_preserve) and bool((base_best_ape <= pos_px).item()):
                        preserve_negs = enough_overlap & (ape > pos_px)
                        if float(self.short_segment_replace_dense_neg_weight) > 0.0:
                            dense_preserve_negs = ape > pos_px
                            replace_weight[dense_preserve_negs] = torch.maximum(
                                replace_weight[dense_preserve_negs],
                                replace_weight.new_full(
                                    (int(dense_preserve_negs.sum().detach().cpu().item()),),
                                    lane_weight * float(self.short_segment_replace_dense_neg_weight),
                                ),
                            )
                        neg_budget = min(topk, int(preserve_negs.sum().detach().cpu().item()))
                        if neg_budget > 0:
                            neg_values = torch.where(
                                preserve_negs,
                                replace_logits_b.detach(),
                                torch.full_like(replace_logits_b.detach(), float("-inf")),
                            )
                            _, neg_idx = torch.topk(neg_values, k=neg_budget, largest=True)
                            replace_weight[neg_idx] = torch.maximum(
                                replace_weight[neg_idx],
                                replace_weight.new_full((neg_idx.numel(),), lane_weight * 2.0),
                            )

                    active_replace = replace_weight > 0.0
                    if bool(active_replace.any()):
                        replace_bce = F.binary_cross_entropy_with_logits(
                            replace_logits_b,
                            replace_target,
                            reduction="none",
                        )
                        replace_losses.append(
                            (replace_bce * replace_weight).sum() / replace_weight.sum().clamp_min(1.0)
                        )

                if (
                    query_replace_logits_b is not None
                    and float(self.short_segment_query_replace_weight) > 0.0
                    and torch.isfinite(base_best_ape)
                ):
                    margin = float(self.short_segment_replace_margin_px)
                    finite_query = torch.isfinite(query_best_ape)
                    if bool(self.short_segment_base_preserve) and bool((base_best_ape <= pos_px).item()):
                        duplicate_good = finite_query & (query_best_ape <= pos_px)
                        update_query_replace_weight(duplicate_good & ~query_replace_pos, lane_weight)
                    else:
                        improves_query = finite_query & (
                            ((query_best_ape + margin) < base_ape)
                            | ((base_ape > pos_px) & (query_best_ape <= pos_px))
                        )
                        improves_query = improves_query & (query_best_ape <= soft_px) & ~base_hit_any
                        if bool(improves_query.any()):
                            pos_values = torch.where(
                                improves_query,
                                query_best_ape,
                                torch.full_like(query_best_ape, float("inf")),
                            )
                            pos_query = int(torch.argmin(pos_values).detach().cpu().item())
                            query_replace_target[pos_query] = 1.0
                            query_replace_pos[pos_query] = True
                            update_query_replace_weight(
                                torch.arange(num_queries, device=device) == pos_query,
                                lane_weight,
                            )
                    negative_query = finite_query & ~query_replace_pos
                    update_query_replace_weight(
                        negative_query,
                        lane_weight * float(self.short_segment_query_replace_neg_weight),
                    )

                candidate_topk = min(topk, int(enough_overlap.sum().detach().cpu().item()))
                if candidate_topk <= 0:
                    continue
                top_vals, top_idx = torch.topk(ape, k=candidate_topk, largest=False)
                active = top_vals <= soft_px
                if not bool(active.any()):
                    continue

                selected = top_idx[active]
                selected_ape = top_vals[active]
                selected_overlap_counts = overlap_counts[selected].to(dtype=dtype)
                coverage = (selected_overlap_counts / valid_count).clamp(0.0, 1.0)
                soft_target = torch.exp(-selected_ape / tau) * coverage
                target_score = torch.where(
                    selected_ape <= pos_px,
                    torch.ones_like(selected_ape),
                    soft_target,
                ).clamp(0.0, 1.0)
                score_target[selected] = torch.maximum(score_target[selected], target_score.to(dtype=dtype))
                score_weight[selected] = torch.maximum(
                    score_weight[selected],
                    score_weight.new_full((selected.numel(),), lane_weight),
                )
                pos_count += int((selected_ape <= pos_px).sum().detach().cpu().item())
                soft_count += int(((selected_ape > pos_px) & (selected_ape <= soft_px)).sum().detach().cpu().item())

                if float(self.short_segment_point_weight) > 0.0:
                    selected_segments = segments_b[selected]
                    selected_mask = overlap_mask[selected].to(dtype=dtype)
                    target_px = target.view(1, num_points, 2) * scale
                    segment_px = selected_segments * scale
                    point = F.smooth_l1_loss(
                        segment_px,
                        target_px.expand_as(segment_px),
                        reduction="none",
                    ).sum(dim=-1)
                    point = (point * selected_mask).sum(dim=1) / selected_mask.sum(dim=1).clamp_min(1.0)
                    point_losses.append((point * target_score.detach() * lane_weight).mean())

            if unified_choice_active and unified_choice_logits_b is not None:
                # One per-query choice distribution whose class 0 is the
                # untouched base lane and classes 1..S are segment windows.
                # v13 can choose the positive query from raw candidate geometry
                # instead of forcing all short GT lanes through the frozen-base
                # Hungarian assignment.
                choice_target_dist = torch.zeros(
                    (num_queries, num_segments + 1),
                    device=device,
                    dtype=unified_choice_logits_b.dtype,
                )
                choice_target_dist[:, 0] = 1.0
                choice_weight = unified_choice_logits_b.new_full(
                    (num_queries,),
                    unified_choice_base_neg_weight,
                )
                choice_lane_target = torch.zeros((num_queries,), device=device, dtype=torch.bool)
                choice_protected = torch.zeros((num_queries,), device=device, dtype=torch.bool)
                segment_points_qs = segments_b.view(num_queries, num_segments, num_points, 2).detach()
                window_masks_qs = masks_b.view(num_queries, num_segments, num_points)
                base_points_q = pred_points[b].detach()
                if pred_valid_logits is not None:
                    base_valid_q = pred_valid_logits[b].detach().float().sigmoid() >= base_valid_thr
                else:
                    base_valid_q = torch.ones((num_queries, num_points), device=device, dtype=torch.bool)

                def set_choice_target(query_idx: int, target_dist: torch.Tensor, target_weight: float) -> None:
                    if query_idx < 0 or query_idx >= num_queries:
                        return
                    current = float(choice_weight[query_idx].detach().cpu().item())
                    if float(target_weight) >= current:
                        choice_target_dist[query_idx] = target_dist
                        choice_weight[query_idx] = float(target_weight)
                        choice_lane_target[query_idx] = True

                for lane_idx in short_lane_indices.tolist():
                    target = target_points_b[lane_idx]
                    valid = target_valid_b[lane_idx].clamp(0.0, 1.0)
                    valid_bool = valid > 0.5
                    valid_count = valid.sum().clamp_min(1.0)
                    valid_count_f = valid_count.to(dtype=dtype)

                    segment_error = torch.norm(
                        (segment_points_qs - target.view(1, 1, num_points, 2)) * scale,
                        dim=-1,
                    )
                    segment_overlap = window_masks_qs & valid_bool.view(1, 1, num_points)
                    overlap_counts = segment_overlap.sum(dim=-1)
                    enough_overlap = overlap_counts >= min_overlap
                    segment_ape = (
                        (segment_error * segment_overlap.to(dtype=segment_error.dtype)).sum(dim=-1)
                        / overlap_counts.clamp_min(1).to(dtype=segment_error.dtype)
                    )
                    segment_ape = torch.where(
                        enough_overlap,
                        segment_ape,
                        torch.full_like(segment_ape, float("inf")),
                    )
                    segment_coverage = (overlap_counts.to(dtype=dtype) / valid_count_f).clamp(0.0, 1.0)
                    segment_quality = torch.exp(
                        -torch.where(
                            torch.isfinite(segment_ape),
                            segment_ape,
                            torch.zeros_like(segment_ape),
                        )
                        / tau
                    ) * segment_coverage
                    segment_quality = torch.where(
                        torch.isfinite(segment_ape),
                        segment_quality,
                        torch.zeros_like(segment_quality),
                    ).clamp(0.0, 1.0)

                    base_error = torch.norm(
                        (base_points_q - target.view(1, num_points, 2)) * scale,
                        dim=-1,
                    )
                    base_ape = (
                        (base_error * valid.view(1, num_points)).sum(dim=-1) / valid_count
                    )
                    base_overlap = base_valid_q & valid_bool.view(1, num_points)
                    base_coverage = (
                        base_overlap.sum(dim=-1).to(dtype=dtype) / valid_count_f
                    ).clamp(0.0, 1.0)
                    base_quality = (
                        torch.exp(-base_ape / tau) * base_coverage
                    ).clamp(0.0, 1.0)

                    query_best_ape, query_best_m = segment_ape.min(dim=1)
                    lane_base_hit = bool((base_ape.min() <= pos_px).item())
                    selected_query = matched_query_for_lane.get(int(lane_idx))
                    selected_segment = None
                    use_hard_segment_target = False
                    if candidate_aware_assignment and not lane_base_hit:
                        finite_query = torch.isfinite(query_best_ape)
                        margin = float(self.short_segment_replace_margin_px)
                        useful_query = finite_query & (query_best_ape <= soft_px)
                        improves_query = useful_query & (
                            ((query_best_ape + margin) < base_ape)
                            | ((base_ape > pos_px) & (query_best_ape <= pos_px))
                        )
                        if bool(useful_query.any()):
                            choice_protected |= useful_query
                        if bool(improves_query.any()):
                            pos_values = torch.where(
                                improves_query,
                                query_best_ape,
                                torch.full_like(query_best_ape, float("inf")),
                            )
                            selected_query = int(torch.argmin(pos_values).detach().cpu().item())
                            selected_segment = int(query_best_m[selected_query].detach().cpu().item())
                            use_hard_segment_target = True

                    if selected_query is None:
                        continue

                    if use_hard_segment_target:
                        target_dist = torch.zeros(
                            (num_segments + 1,),
                            device=device,
                            dtype=unified_choice_logits_b.dtype,
                        )
                        target_dist[int(selected_segment) + 1] = 1.0
                        set_choice_target(selected_query, target_dist, lane_weight)
                        continue

                    query_segment_ape = segment_ape[selected_query]
                    query_segment_quality = segment_quality[selected_query]
                    base_hit = bool((base_ape[selected_query] <= pos_px).item())
                    candidate_hit = bool((query_segment_ape <= pos_px).any().item())
                    if lane_base_hit or base_hit or not candidate_hit:
                        target_dist = torch.zeros(
                            (num_segments + 1,),
                            device=device,
                            dtype=unified_choice_logits_b.dtype,
                        )
                        target_dist[0] = 1.0
                        target_weight = lane_weight * (2.0 if lane_base_hit or base_hit else 1.0)
                    else:
                        quality = torch.cat(
                            (
                                base_quality[selected_query].view(1),
                                query_segment_quality,
                            ),
                            dim=0,
                        ).to(dtype=unified_choice_logits_b.dtype)
                        target_dist = F.softmax(
                            torch.log(quality.clamp_min(1.0e-6)) / unified_choice_temperature,
                            dim=0,
                        )
                        target_weight = lane_weight
                    set_choice_target(selected_query, target_dist, target_weight)

                if candidate_aware_assignment and bool(choice_protected.any()):
                    ignore_base_neg = choice_protected & ~choice_lane_target
                    if bool(ignore_base_neg.any()):
                        choice_weight[ignore_base_neg] = 0.0

                active_choice = choice_weight > 0.0
                if bool(active_choice.any()):
                    choice_log_probs = F.log_softmax(unified_choice_logits_b, dim=-1)
                    choice_ce = -(choice_target_dist * choice_log_probs).sum(dim=-1)
                    unified_choice_losses.append(
                        (choice_ce * choice_weight).sum() / choice_weight.sum().clamp_min(1.0)
                    )

            if train_base_choice and base_choice_all_queries:
                if not official_quality_target:
                    raise ValueError(
                        "gcs_short_segment_base_choice_all_queries requires "
                        "gcs_short_segment_official_quality_target=True."
                    )
                best_quality_q, best_segment_q = official_quality.view(num_queries, num_segments).max(dim=1)
                best_weight_q = official_quality_lane_weight.view(num_queries, num_segments).gather(
                    1, best_segment_q.view(-1, 1)
                ).squeeze(1)
                base_hit_q = base_official_quality >= official_pt_thresh
                candidate_hit_q = best_quality_q >= official_pt_thresh
                candidate_better_q = (
                    candidate_hit_q
                    & ~base_hit_q
                    & (best_quality_q > base_official_quality)
                )
                base_choice_target.zero_()
                base_choice_target[candidate_better_q] = best_segment_q[candidate_better_q] + 1
                base_choice_weight.fill_(float(self.short_segment_base_choice_neg_weight))
                base_choice_weight[base_hit_q] = torch.maximum(
                    base_choice_weight[base_hit_q],
                    (base_official_lane_weight[base_hit_q] * 2.0).clamp_min(1.0),
                )
                base_choice_weight[candidate_better_q] = torch.maximum(
                    base_choice_weight[candidate_better_q],
                    best_weight_q[candidate_better_q].clamp_min(1.0),
                )

            if query_replace_logits_b is not None and float(self.short_segment_query_replace_weight) > 0.0:
                if bool(self.short_segment_base_preserve):
                    query_replace_target[base_hit_any] = 0.0
                    query_replace_pos[base_hit_any] = False
                    update_query_replace_weight(base_hit_any, 2.0)
                active_query_replace = query_replace_weight > 0.0
                if bool(active_query_replace.any()):
                    query_replace_bce = F.binary_cross_entropy_with_logits(
                        query_replace_logits_b,
                        query_replace_target,
                        reduction="none",
                    )
                    query_replace_losses.append(
                        (query_replace_bce * query_replace_weight).sum()
                        / query_replace_weight.sum().clamp_min(1.0)
                    )

            if train_base_choice:
                if (
                    bool(self.short_segment_base_preserve)
                    and not self.short_segment_matched_assignment
                    and not base_choice_all_queries
                ):
                    for query_idx in torch.nonzero(base_hit_any, as_tuple=False).flatten().tolist():
                        update_base_choice(int(query_idx), 0, 2.0)
                active_base_choice = base_choice_weight > 0.0
                if bool(active_base_choice.any()):
                    active_logits = logits_qs_b[active_base_choice]
                    base_logits = active_logits.new_zeros((active_logits.shape[0], 1))
                    choice_logits = torch.cat((base_logits, active_logits), dim=1)
                    choice_targets = base_choice_target[active_base_choice].clamp(0, num_segments)
                    choice_ce = F.cross_entropy(choice_logits, choice_targets, reduction="none")
                    choice_weight = base_choice_weight[active_base_choice]
                    base_choice_losses.append(
                        (choice_ce * choice_weight).sum() / choice_weight.sum().clamp_min(1.0)
                    )

            neg_mask = (score_weight <= 0.0) & (min_ape > soft_px)
            if float(self.short_segment_neg_score_thr) > 0.0:
                neg_mask = neg_mask & (logits_b.detach().sigmoid() >= float(self.short_segment_neg_score_thr))
            if bool(neg_mask.any()):
                score_weight[neg_mask] = 1.0
                neg_count += int(neg_mask.sum().detach().cpu().item())

            active_weight = score_weight > 0.0
            if float(self.short_segment_bce_weight) > 0.0 and bool(active_weight.any()):
                bce = F.binary_cross_entropy_with_logits(logits_b, score_target, reduction="none")
                bce_losses.append((bce * score_weight).sum() / score_weight.sum().clamp_min(1.0))
            if float(self.short_segment_dense_quality_weight) > 0.0:
                if official_quality_target:
                    dense_target = (official_quality >= official_pt_thresh).to(dtype=dtype)
                    dense_weight = torch.where(
                        dense_target > 0.0,
                        official_quality_lane_weight.clamp_min(1.0),
                        official_quality.new_full((flat_count,), float(self.short_segment_dense_neg_weight)),
                    )
                else:
                    finite = torch.isfinite(min_ape)
                    dense_soft = torch.exp(-torch.where(finite, min_ape, torch.zeros_like(min_ape)) / tau)
                    dense_target = torch.where(
                        finite & (min_ape <= pos_px),
                        torch.ones_like(min_ape),
                        torch.where(finite & (min_ape <= soft_px), dense_soft, torch.zeros_like(min_ape)),
                    ).clamp(0.0, 1.0)
                    dense_weight = torch.where(
                        dense_target > 0.0,
                        min_ape_lane_weight.clamp_min(1.0),
                        min_ape_lane_weight.new_full((flat_count,), float(self.short_segment_dense_neg_weight)),
                    )
                active_dense = dense_weight > 0.0
                if bool(active_dense.any()):
                    dense_bce = F.binary_cross_entropy_with_logits(logits_b, dense_target, reduction="none")
                    dense_quality_losses.append(
                        (dense_bce * dense_weight).sum() / dense_weight.sum().clamp_min(1.0)
                    )

        bce_loss = torch.stack(bce_losses).mean() if bce_losses else zero
        dense_quality_loss = torch.stack(dense_quality_losses).mean() if dense_quality_losses else zero
        listwise_loss = torch.stack(listwise_losses).mean() if listwise_losses else zero
        query_rank_loss = torch.stack(query_rank_losses).mean() if query_rank_losses else zero
        base_choice_loss = torch.stack(base_choice_losses).mean() if base_choice_losses else zero
        unified_choice_loss = torch.stack(unified_choice_losses).mean() if unified_choice_losses else zero
        replace_loss = torch.stack(replace_losses).mean() if replace_losses else zero
        query_replace_loss = torch.stack(query_replace_losses).mean() if query_replace_losses else zero
        score_loss = (
            float(self.short_segment_bce_weight) * bce_loss
            + float(self.short_segment_dense_quality_weight) * dense_quality_loss
            + float(self.short_segment_listwise_weight) * listwise_loss
            + float(self.short_segment_query_rank_weight) * query_rank_loss
            + float(self.short_segment_base_choice_weight) * base_choice_loss
            + float(self.short_segment_unified_choice_weight) * unified_choice_loss
            + float(self.short_segment_replace_weight) * replace_loss
            + float(self.short_segment_query_replace_weight) * query_replace_loss
        )
        point_loss = torch.stack(point_losses).mean() if point_losses else zero
        total = score_loss + float(self.short_segment_point_weight) * point_loss
        return (
            total,
            score_loss,
            point_loss,
            pred_points.new_tensor(float(pos_count)),
            pred_points.new_tensor(float(soft_count)),
            pred_points.new_tensor(float(neg_count)),
        )

    @staticmethod
    def _full_lane_outputs(
        preds: dict[str, torch.Tensor],
        pred_points: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ] | None:
        """Validate and return the independent full-lane proposal tensors."""
        names = (
            "pred_full_lane_points",
            "pred_full_lane_valid_logits",
            "pred_full_lane_exist_logits",
            "pred_full_lane_quality_logits",
            "pred_full_lane_start_logits",
            "pred_full_lane_end_logits",
        )
        if not all(name in preds for name in names):
            return None

        full_points = preds["pred_full_lane_points"]
        full_valid = preds["pred_full_lane_valid_logits"]
        full_exist = preds["pred_full_lane_exist_logits"]
        full_quality = preds["pred_full_lane_quality_logits"]
        full_start = preds["pred_full_lane_start_logits"]
        full_end = preds["pred_full_lane_end_logits"]
        if full_points.ndim != 4 or full_points.shape[-1] != 2:
            raise ValueError(
                "pred_full_lane_points must have shape B x P x K x 2, "
                f"got {tuple(full_points.shape)}."
            )
        expected_bpk = (pred_points.shape[0], full_points.shape[1], pred_points.shape[2])
        if tuple(full_valid.shape) != expected_bpk:
            raise ValueError(
                "pred_full_lane_valid_logits must have shape B x P x K, "
                f"got {tuple(full_valid.shape)} vs {expected_bpk}."
            )
        if tuple(full_exist.shape) != expected_bpk[:2]:
            raise ValueError(
                "pred_full_lane_exist_logits must have shape B x P, "
                f"got {tuple(full_exist.shape)} vs {expected_bpk[:2]}."
            )
        if tuple(full_quality.shape) != expected_bpk[:2]:
            raise ValueError(
                "pred_full_lane_quality_logits must have shape B x P, "
                f"got {tuple(full_quality.shape)} vs {expected_bpk[:2]}."
            )
        if tuple(full_start.shape) != expected_bpk or tuple(full_end.shape) != expected_bpk:
            raise ValueError(
                "pred_full_lane_start_logits and pred_full_lane_end_logits must have shape B x P x K, "
                f"got {tuple(full_start.shape)} and {tuple(full_end.shape)} vs {expected_bpk}."
            )
        if "pred_full_lane_score_logits" in preds and tuple(preds["pred_full_lane_score_logits"].shape) != expected_bpk[:2]:
            raise ValueError(
                "pred_full_lane_score_logits must have shape B x P, "
                f"got {tuple(preds['pred_full_lane_score_logits'].shape)} vs {expected_bpk[:2]}."
            )
        return full_points, full_valid, full_exist, full_quality, full_start, full_end

    def _full_lane_unified_indices(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        full_outputs: tuple[torch.Tensor, ...],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Match base queries and independent full proposals in one set."""
        full_points, full_valid, full_exist, full_quality, full_start, full_end = full_outputs
        base_score = full_lane_proposal_score_probability(
            pred_logits,
            quality_logits=None,
            valid_logits=pred_valid_logits,
        )
        full_score = full_lane_proposal_score_probability(
            full_exist,
            quality_logits=full_quality,
            valid_logits=full_valid,
            start_logits=full_start,
            end_logits=full_end,
        )
        combined_points = torch.cat((pred_points, full_points), dim=1)
        combined_score = torch.cat((base_score, full_score), dim=1)
        combined_logits = probability_to_logit(combined_score)
        return self.matcher(combined_points, combined_logits, gt_points, gt_valid)

    def _full_lane_aux_indices(
        self,
        full_outputs: tuple[torch.Tensor, ...],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Match only full-lane proposals to GT so the new head cannot be starved by base queries."""
        full_points = full_outputs[0]
        neutral_logits = full_points.new_zeros(full_points.shape[:2])
        return self.full_lane_aux_matcher(full_points, neutral_logits, gt_points, gt_valid)

    def _full_lane_focus_target_weights(
        self,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
    ) -> list[torch.Tensor] | None:
        """Build per-GT weights that focus full proposals on base-miss short GT4/GT5 lanes."""
        if not self.full_lane_hard_focus:
            return None
        device = pred_points.device
        dtype = pred_points.dtype
        pixel_scale = self._pixel_scale_for(pred_points).to(device=device, dtype=dtype).view(1, 1, 1, 2)
        hit_px = float(self.full_lane_focus_hit_px)
        valid_thr = float(self.full_lane_focus_base_valid_thr)
        min_coverage = float(self.full_lane_focus_base_min_coverage)
        weights: list[torch.Tensor] = []
        for b, (gt_points_b, gt_valid_b) in enumerate(zip(gt_points, gt_valid)):
            gt_points_b = gt_points_b.to(device=device, dtype=dtype)
            gt_valid_b = gt_valid_b.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            lane_count = int(gt_points_b.shape[0])
            if lane_count <= 0:
                weights.append(pred_points.new_zeros((0,)))
                continue
            visible_count = gt_valid_b.sum(dim=1)
            valid_denom = visible_count.clamp_min(1.0)
            base_points_b = pred_points[b].to(dtype=dtype)
            if base_points_b.numel():
                error = torch.norm((base_points_b[:, None] - gt_points_b[None]) * pixel_scale, dim=-1)
                ape = (error * gt_valid_b[None]).sum(dim=-1) / valid_denom[None]
                if pred_valid_logits is not None:
                    base_valid_b = pred_valid_logits[b].to(device=device, dtype=dtype).sigmoid()
                    valid_hit = (base_valid_b[:, None] >= valid_thr).to(dtype=dtype) * gt_valid_b[None]
                    coverage = valid_hit.sum(dim=-1) / valid_denom[None]
                    base_hit = ((ape <= hit_px) & (coverage >= min_coverage)).any(dim=0)
                else:
                    base_hit = (ape <= hit_px).any(dim=0)
            else:
                base_hit = torch.zeros((lane_count,), device=device, dtype=torch.bool)

            lane_weight = torch.where(
                base_hit,
                pred_points.new_tensor(float(self.full_lane_base_hit_weight)),
                pred_points.new_tensor(float(self.full_lane_base_miss_weight)),
            )
            if lane_count == 4:
                lane_weight = lane_weight * float(self.full_lane_gt4_weight)
            elif lane_count >= 5:
                lane_weight = lane_weight * float(self.full_lane_gt5_weight)
            short_mask = visible_count <= float(self.full_lane_short_visible_thr)
            lane_weight = lane_weight * torch.where(
                short_mask,
                pred_points.new_tensor(float(self.full_lane_short_visible_weight)),
                pred_points.new_tensor(1.0),
            )
            weights.append(lane_weight.to(device=device, dtype=dtype))
        return weights

    @staticmethod
    def _split_full_lane_indices(
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        base_query_count: int,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
        """Split unified set matches into base-query and full-proposal indices."""
        base_indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        full_indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        for src_idx, tgt_idx in indices:
            base_mask = src_idx < int(base_query_count)
            base_indices.append((src_idx[base_mask], tgt_idx[base_mask]))
            full_indices.append(
                (
                    src_idx[~base_mask] - int(base_query_count),
                    tgt_idx[~base_mask],
                )
            )
        return base_indices, full_indices

    def full_lane_proposal_loss(
        self,
        full_outputs: tuple[torch.Tensor, ...],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        full_indices: list[tuple[torch.Tensor, torch.Tensor]],
        unmatched_weight: float | None = None,
        target_weights: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Supervise complete full-lane proposals selected by unified matching."""
        full_points, full_valid, full_exist, full_quality, full_start, full_end = full_outputs
        device = full_points.device
        dtype = full_points.dtype
        batch_size, proposal_count, num_points, _ = full_points.shape
        unmatched_weight = float(self.full_lane_unmatched_weight if unmatched_weight is None else unmatched_weight)
        if len(full_indices) != batch_size:
            raise ValueError(
                "full-lane indices must contain one pair per image, "
                f"got {len(full_indices)} vs B={batch_size}."
            )
        if target_weights is not None and len(target_weights) != batch_size:
            raise ValueError(
                "full-lane target weights must contain one tensor per image, "
                f"got {len(target_weights)} vs B={batch_size}."
            )

        zero = full_points.sum() * 0.0
        point_losses: list[torch.Tensor] = []
        point_weights: list[torch.Tensor] = []
        valid_losses: list[torch.Tensor] = []
        interval_losses: list[torch.Tensor] = []
        interval_weights: list[torch.Tensor] = []
        exist_losses: list[torch.Tensor] = []
        quality_losses: list[torch.Tensor] = []
        matched_count = 0
        unmatched_count = 0
        pixel_scale = self._pixel_scale_for(full_points).view(1, 1, 2)
        tau = max(float(self.full_lane_quality_tau), 1.0e-6)

        for b in range(batch_size):
            src_idx, tgt_idx = full_indices[b]
            src_idx = src_idx.to(device=device, dtype=torch.long).reshape(-1)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long).reshape(-1)
            matched_mask = torch.zeros((proposal_count,), device=device, dtype=torch.bool)
            if src_idx.numel():
                matched_mask[src_idx] = True
            unmatched_count += int((~matched_mask).sum().detach().cpu().item())
            matched_count += int(src_idx.numel())

            exist_target = matched_mask.to(dtype=dtype)
            quality_target = torch.zeros((proposal_count,), device=device, dtype=dtype)
            valid_target = torch.zeros((proposal_count, num_points), device=device, dtype=dtype)
            valid_weight = torch.full(
                (proposal_count,),
                float(self.full_lane_unmatched_valid_weight) * unmatched_weight,
                device=device,
                dtype=dtype,
            )
            matched_target_weight = full_points.new_ones((int(src_idx.numel()),), dtype=dtype)
            if target_weights is not None and src_idx.numel():
                target_weight_b = target_weights[b].to(device=device, dtype=dtype).reshape(-1)
                if int(target_weight_b.numel()) != int(gt_points[b].shape[0]):
                    raise ValueError(
                        "full-lane target weight count must match GT lane count, "
                        f"got {int(target_weight_b.numel())} vs {int(gt_points[b].shape[0])}."
                    )
                matched_target_weight = target_weight_b[tgt_idx].clamp_min(0.0)
            if src_idx.numel():
                valid_weight[src_idx] = matched_target_weight

            if src_idx.numel():
                target_points = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
                target_valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx].clamp(0.0, 1.0)
                pred_points_b = full_points[b, src_idx]
                pred_valid_b = full_valid[b, src_idx].sigmoid()
                point_error = torch.norm((pred_points_b - target_points) * pixel_scale, dim=-1)
                ape = (point_error * target_valid).sum(dim=1) / target_valid.sum(dim=1).clamp_min(1.0)
                visible_coverage = (
                    (pred_valid_b * target_valid).sum(dim=1) / target_valid.sum(dim=1).clamp_min(1.0)
                ).clamp(0.0, 1.0)
                quality = (torch.exp(-ape / tau) * visible_coverage).clamp(0.0, 1.0)
                quality_target[src_idx] = quality.detach().to(dtype=dtype)
                # Existence answers whether this proposal is assigned to a
                # lane; geometry quality is supervised independently.
                exist_target[src_idx] = 1.0
                valid_target[src_idx] = target_valid

                for local_idx in range(int(src_idx.numel())):
                    valid_bool = target_valid[local_idx] > 0.5
                    if bool(valid_bool.any()):
                        point_losses.append(
                            aspect_weighted_l1_point_loss(
                                pred_points_b[local_idx],
                                target_points[local_idx],
                                valid_bool,
                                image_size=self.image_size,
                            )
                        )
                        point_weights.append(matched_target_weight[local_idx])
                        valid_indices = torch.nonzero(valid_bool, as_tuple=False).flatten()
                        start_target = valid_indices[0].view(1)
                        end_target = valid_indices[-1].view(1)
                        proposal_idx = src_idx[local_idx]
                        interval_losses.append(
                            F.cross_entropy(full_start[b, proposal_idx].view(1, -1), start_target)
                            + F.cross_entropy(full_end[b, proposal_idx].view(1, -1), end_target)
                        )
                        interval_weights.append(matched_target_weight[local_idx])

            proposal_weight = torch.full(
                (proposal_count,),
                unmatched_weight,
                device=device,
                dtype=dtype,
            )
            if src_idx.numel():
                proposal_weight[src_idx] = matched_target_weight
            exist_bce = F.binary_cross_entropy_with_logits(
                full_exist[b],
                exist_target,
                reduction="none",
            )
            quality_bce = F.binary_cross_entropy_with_logits(
                full_quality[b],
                quality_target,
                reduction="none",
            )
            exist_losses.append((exist_bce * proposal_weight).sum() / proposal_weight.sum().clamp_min(1.0))
            quality_losses.append((quality_bce * proposal_weight).sum() / proposal_weight.sum().clamp_min(1.0))
            valid_bce = F.binary_cross_entropy_with_logits(
                full_valid[b],
                valid_target,
                reduction="none",
            ).mean(dim=1)
            valid_losses.append((valid_bce * valid_weight).sum() / valid_weight.sum().clamp_min(1.0))

        if point_losses:
            point_loss_values = torch.stack(point_losses)
            point_weight_values = torch.stack(point_weights).to(device=device, dtype=dtype)
            point_loss = (point_loss_values * point_weight_values).sum() / point_weight_values.sum().clamp_min(1.0)
        else:
            point_loss = zero
        valid_loss = torch.stack(valid_losses).mean() if valid_losses else zero
        if interval_losses:
            interval_loss_values = torch.stack(interval_losses)
            interval_weight_values = torch.stack(interval_weights).to(device=device, dtype=dtype)
            interval_loss = (
                (interval_loss_values * interval_weight_values).sum()
                / interval_weight_values.sum().clamp_min(1.0)
            )
        else:
            interval_loss = zero
        exist_loss = torch.stack(exist_losses).mean() if exist_losses else zero
        quality_loss = torch.stack(quality_losses).mean() if quality_losses else zero
        total = point_loss + valid_loss + interval_loss + exist_loss + quality_loss
        return (
            total,
            point_loss,
            valid_loss,
            interval_loss,
            exist_loss,
            quality_loss,
            full_points.new_tensor(float(matched_count)),
            full_points.new_tensor(float(unmatched_count)),
        )

    @staticmethod
    def _dense_draw_gaussian(
        target: torch.Tensor,
        channel: int,
        x: float,
        y: float,
        sigma: float,
    ) -> None:
        """Draw one clipped Gaussian on a dense target map in-place."""
        height, width = int(target.shape[-2]), int(target.shape[-1])
        if height <= 0 or width <= 0 or not math.isfinite(float(x)) or not math.isfinite(float(y)):
            return
        radius = max(int(math.ceil(3.0 * float(sigma))), 1)
        center_x = int(round(float(x)))
        center_y = int(round(float(y)))
        x0 = max(center_x - radius, 0)
        x1 = min(center_x + radius, width - 1)
        y0 = max(center_y - radius, 0)
        y1 = min(center_y + radius, height - 1)
        if x0 > x1 or y0 > y1:
            return
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1 + 1, device=target.device, dtype=target.dtype),
            torch.arange(x0, x1 + 1, device=target.device, dtype=target.dtype),
            indexing="ij",
        )
        patch = torch.exp(-((xx - float(x)) ** 2 + (yy - float(y)) ** 2) / (2.0 * float(sigma) ** 2))
        current = target[channel, y0 : y1 + 1, x0 : x1 + 1]
        target[channel, y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(current, patch)

    @classmethod
    def _dense_draw_line(
        cls,
        target: torch.Tensor,
        channel: int,
        start: torch.Tensor,
        end: torch.Tensor,
        sigma: float,
    ) -> None:
        """Rasterize a line segment between two fixed-y lane anchors."""
        distance = float(torch.linalg.vector_norm(end - start).detach().cpu().item())
        steps = max(int(math.ceil(distance)) + 1, 2)
        values = torch.linspace(0.0, 1.0, steps=steps, device=target.device, dtype=target.dtype)
        for value in values:
            point = start.to(device=target.device, dtype=target.dtype) * (1.0 - value) + end.to(
                device=target.device, dtype=target.dtype
            ) * value
            cls._dense_draw_gaussian(target, channel, float(point[0].item()), float(point[1].item()), sigma)

    @classmethod
    def _dense_targets(
        cls,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        output_hw: tuple[int, int],
        sigma_px: float,
        image_size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rasterize centerline and endpoint evidence from normalized lane points."""
        output_h, output_w = int(output_hw[0]), int(output_hw[1])
        if output_h <= 0 or output_w <= 0:
            raise ValueError(f"Dense evidence output size must be positive, got {output_hw}.")
        image_h, image_w = normalize_imgsz(image_size)
        stride_x = float(image_w) / float(max(output_w - 1, 1))
        stride_y = float(image_h) / float(max(output_h - 1, 1))
        sigma_cells = max(float(sigma_px) / max((stride_x + stride_y) * 0.5, 1.0), 0.5)
        batch_size = len(gt_points)
        centerline = torch.zeros((batch_size, 1, output_h, output_w), device=gt_points[0].device if batch_size else "cpu")
        endpoints = torch.zeros_like(centerline).expand(batch_size, 2, output_h, output_w).clone()

        for batch_index, (points_b, valid_b) in enumerate(zip(gt_points, gt_valid)):
            points_b = points_b.to(device=centerline.device, dtype=torch.float32)
            valid_b = valid_b.to(device=centerline.device, dtype=torch.float32)
            if points_b.ndim != 3 or points_b.shape[-1] != 2:
                raise ValueError(f"Dense GT points must have shape N x K x 2, got {tuple(points_b.shape)}.")
            if tuple(valid_b.shape) != tuple(points_b.shape[:2]):
                raise ValueError(
                    "Dense GT validity must match point shape, "
                    f"got {tuple(valid_b.shape)} vs {tuple(points_b.shape[:2])}."
                )
            for lane_points, lane_valid in zip(points_b, valid_b):
                indices = torch.nonzero(lane_valid > 0.5, as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                coords = lane_points[indices].clone()
                coords[:, 0] *= float(max(output_w - 1, 1))
                coords[:, 1] *= float(max(output_h - 1, 1))
                for point in coords:
                    cls._dense_draw_gaussian(
                        centerline[batch_index],
                        0,
                        float(point[0].item()),
                        float(point[1].item()),
                        sigma_cells,
                    )
                for point_index in range(int(indices.numel()) - 1):
                    if int(indices[point_index + 1] - indices[point_index]) == 1:
                        cls._dense_draw_line(
                            centerline[batch_index],
                            0,
                            coords[point_index],
                            coords[point_index + 1],
                            sigma_cells,
                        )
                cls._dense_draw_gaussian(
                    endpoints[batch_index],
                    0,
                    float(coords[0, 0].item()),
                    float(coords[0, 1].item()),
                    sigma_cells,
                )
                cls._dense_draw_gaussian(
                    endpoints[batch_index],
                    1,
                    float(coords[-1, 0].item()),
                    float(coords[-1, 1].item()),
                    sigma_cells,
                )
        return centerline, endpoints

    def dense_instance_loss(
        self,
        preds: dict[str, torch.Tensor],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Train dense centerline, endpoint, and instance-embedding evidence."""
        centerline_logits = preds.get("pred_dense_centerline_logits")
        endpoint_logits = preds.get("pred_dense_endpoint_logits")
        instance_embed = preds.get("pred_dense_instance_embed")
        if centerline_logits is None or endpoint_logits is None or instance_embed is None:
            raise KeyError(
                "gcs_dense_instance > 0 requires pred_dense_centerline_logits, "
                "pred_dense_endpoint_logits, and pred_dense_instance_embed."
            )
        if centerline_logits.ndim != 4 or tuple(centerline_logits.shape[1:2]) != (1,):
            raise ValueError(
                "pred_dense_centerline_logits must have shape B x 1 x Hf x Wf, "
                f"got {tuple(centerline_logits.shape)}."
            )
        if endpoint_logits.ndim != 4 or tuple(endpoint_logits.shape[1:2]) != (2,):
            raise ValueError(
                "pred_dense_endpoint_logits must have shape B x 2 x Hf x Wf, "
                f"got {tuple(endpoint_logits.shape)}."
            )
        if instance_embed.ndim != 4 or instance_embed.shape[0] != centerline_logits.shape[0]:
            raise ValueError(
                "pred_dense_instance_embed must have shape B x D x Hf x Wf, "
                f"got {tuple(instance_embed.shape)}."
            )
        if tuple(endpoint_logits.shape[-2:]) != tuple(centerline_logits.shape[-2:]) or tuple(
            instance_embed.shape[-2:]
        ) != tuple(centerline_logits.shape[-2:]):
            raise ValueError("Dense evidence logits and embeddings must share Hf,Wf.")
        if len(gt_points) != int(centerline_logits.shape[0]) or len(gt_valid) != len(gt_points):
            raise ValueError(
                "Dense evidence targets must contain one GT point/valid tensor per batch image, "
                f"got points={len(gt_points)}, valid={len(gt_valid)}, batch={int(centerline_logits.shape[0])}."
            )

        center_target, endpoint_target = self._dense_targets(
            gt_points,
            gt_valid,
            tuple(centerline_logits.shape[-2:]),
            sigma_px=float(self.dense_sigma_px),
            image_size=self.image_size,
        )
        center_target = center_target.to(device=centerline_logits.device, dtype=centerline_logits.dtype)
        endpoint_target = endpoint_target.to(device=endpoint_logits.device, dtype=endpoint_logits.dtype)

        def sparse_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            positive = target.sum()
            negative = target.numel() - positive
            pos_weight = (negative / positive.clamp_min(1.0)).clamp(
                min=1.0,
                max=float(self.dense_pos_weight_max),
            )
            return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

        centerline_loss = sparse_bce(centerline_logits, center_target)
        endpoint_loss = sparse_bce(endpoint_logits, endpoint_target)

        normalized_embed = F.normalize(instance_embed, dim=1, eps=1.0e-6)
        pull_losses: list[torch.Tensor] = []
        push_losses: list[torch.Tensor] = []
        for batch_index, (points_b, valid_b) in enumerate(zip(gt_points, gt_valid)):
            points_b = points_b.to(device=normalized_embed.device, dtype=normalized_embed.dtype)
            valid_b = valid_b.to(device=normalized_embed.device)
            if points_b.numel() == 0:
                continue
            image_lane_means: list[torch.Tensor] = []
            for lane_points, lane_valid in zip(points_b, valid_b):
                lane_mask = lane_valid > 0.5
                if int(lane_mask.sum().item()) < 2:
                    continue
                coords = lane_points[lane_mask, :2].clamp(0.0, 1.0)
                grid = coords.mul(2.0).sub(1.0).view(1, -1, 1, 2)
                sampled = F.grid_sample(
                    normalized_embed[batch_index : batch_index + 1],
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                ).squeeze(0).squeeze(-1).transpose(0, 1)
                sampled = F.normalize(sampled, dim=1, eps=1.0e-6)
                mean_embedding = F.normalize(sampled.mean(dim=0, keepdim=True), dim=1, eps=1.0e-6)[0]
                pull_losses.append((1.0 - (sampled * mean_embedding.view(1, -1)).sum(dim=1)).mean())
                image_lane_means.append(mean_embedding)

            # Instance identity is only defined within one image. Do not push
            # lane embeddings from different batch images apart.
            if len(image_lane_means) >= 2:
                means = torch.stack(image_lane_means, dim=0)
                pair_losses = []
                for left in range(int(means.shape[0]) - 1):
                    distances = torch.linalg.vector_norm(means[left + 1 :] - means[left], dim=1)
                    pair_losses.append(F.relu(float(self.dense_embed_margin) - distances).square().mean())
                push_losses.append(torch.stack(pair_losses).mean())
        embed_push_loss = torch.stack(push_losses).mean() if push_losses else centerline_logits.sum() * 0.0
        embed_pull_loss = torch.stack(pull_losses).mean() if pull_losses else centerline_logits.sum() * 0.0
        dense_loss = (
            float(self.dense_centerline_weight) * centerline_loss
            + float(self.dense_endpoint_weight) * endpoint_loss
            + float(self.dense_embed_pull_weight) * embed_pull_loss
            + float(self.dense_embed_push_weight) * embed_push_loss
        )
        return (
            dense_loss,
            centerline_loss,
            endpoint_loss,
            embed_pull_loss,
            embed_push_loss,
            center_target.sum().detach(),
        )

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

    def _gt_lane_mean_x_px(
        self,
        gt_points_b: torch.Tensor,
        gt_valid_b: torch.Tensor,
        width: float,
    ) -> torch.Tensor:
        """Return mean visible x in pixels for every GT lane."""
        valid = gt_valid_b > 0.5
        x_px = gt_points_b[..., 0] * float(width)
        denom = valid.float().sum(dim=1).clamp_min(1.0)
        return (x_px * valid.float()).sum(dim=1) / denom

    def _query_outside_gt_envelope_ratio(
        self,
        pred_points_q: torch.Tensor,
        pred_visible_q: torch.Tensor,
        gt_points_b: torch.Tensor,
        gt_valid_b: torch.Tensor,
        width: float,
        *,
        side: str,
        margin: float,
    ) -> torch.Tensor:
        """Return outside-envelope ratio over query/GT common visible anchors."""
        device = pred_points_q.device
        dtype = pred_points_q.dtype
        q_visible = pred_visible_q > 0.5
        gt_visible = gt_valid_b > 0.5
        env_valid = q_visible & gt_visible.any(dim=0)
        if int(env_valid.sum().item()) < int(self.boundary_pseudo_min_valid):
            return torch.tensor(float("nan"), device=device, dtype=dtype)

        pred_x = pred_points_q[:, 0] * float(width)
        gt_x = gt_points_b[..., 0] * float(width)
        left_env_x = gt_x.masked_fill(~gt_visible, float("inf")).min(dim=0).values
        right_env_x = gt_x.masked_fill(~gt_visible, float("-inf")).max(dim=0).values

        if side == "left":
            outside = pred_x[env_valid] < (left_env_x[env_valid] - float(margin))
        elif side == "right":
            outside = pred_x[env_valid] > (right_env_x[env_valid] + float(margin))
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
        return outside.float().mean()

    def _query_to_gt_lane_dist_px(
        self,
        pred_points_q: torch.Tensor,
        pred_visible_q: torch.Tensor,
        gt_points_b: torch.Tensor,
        gt_valid_b: torch.Tensor,
        width: float,
    ) -> torch.Tensor:
        """Return mean x-distance from one predicted query lane to every GT lane."""
        device = pred_points_q.device
        dtype = pred_points_q.dtype
        pred_x_px = pred_points_q[:, 0] * float(width)
        gt_x_px = gt_points_b[..., 0] * float(width)

        dists = []
        for g in range(gt_points_b.shape[0]):
            common = (pred_visible_q > 0.5) & (gt_valid_b[g] > 0.5)
            if bool(common.any()):
                dist = (pred_x_px[common] - gt_x_px[g, common]).abs().mean()
            else:
                dist = torch.tensor(float("inf"), device=device, dtype=dtype)
            dists.append(dist)

        return torch.stack(dists) if dists else torch.empty((0,), device=device, dtype=dtype)

    def boundary_pseudo_neg_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extra negative BCE for short far-side pseudo lanes on unmatched queries."""
        if float(self.boundary_pseudo_neg_gain) <= 0.0:
            zero = self._zero_like(pred_points)
            return zero, zero, zero
        if pred_valid_logits is None:
            raise ValueError(
                "gcs_boundary_pseudo_neg requires preds['pred_valid_logits'] with shape B x Q x K; "
                "disable --gcs-boundary-pseudo-neg or use a GCS head that emits per-point visibility logits."
            )

        if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
            pred_logits_2d = pred_logits.squeeze(-1)
        else:
            pred_logits_2d = pred_logits
        if pred_logits_2d.ndim != 2:
            raise ValueError(f"pred_logits must be B x Q, got {tuple(pred_logits.shape)}.")

        device = pred_points.device
        dtype = pred_points.dtype
        bsz, num_queries = pred_logits_2d.shape
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=pred_logits_2d.dtype).reshape(-1)
        if gt_lanes.numel() != bsz:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={bsz}.")

        width = float(self._pixel_scale_for(pred_points).reshape(-1)[0].detach().cpu().item())
        valid_prob = pred_valid_logits.detach().sigmoid()
        exist_prob = pred_logits_2d.detach().sigmoid()
        points = pred_points.detach()

        losses = []
        pseudo_counts = []
        pseudo_score_means = []

        for b in range(bsz):
            gt_count_b = int(round(float(gt_lanes[b].detach().cpu().item())))
            if gt_count_b != int(self.boundary_pseudo_gt_count):
                continue

            gt_points_b = gt_points[b].detach().to(device=device, dtype=dtype)
            gt_valid_b = gt_valid[b].detach().to(device=device, dtype=dtype)
            if gt_points_b.numel() == 0 or gt_points_b.ndim != 3 or gt_points_b.shape[0] < 2:
                continue
            if gt_valid_b.shape != gt_points_b.shape[:2]:
                raise ValueError(
                    f"GT valid mask must match GT lane first two dims, got {tuple(gt_valid_b.shape)} vs "
                    f"{tuple(gt_points_b.shape[:2])}."
                )

            src_idx, _ = indices[b]
            matched = torch.zeros((num_queries,), device=device, dtype=torch.bool)
            if src_idx.numel() > 0:
                matched[src_idx.to(device=device, dtype=torch.long)] = True
            unmatched_idx = torch.nonzero(~matched, as_tuple=False).flatten()
            if unmatched_idx.numel() == 0:
                continue

            gt_mean_x = self._gt_lane_mean_x_px(gt_points_b, gt_valid_b, width)
            left_gt = int(torch.argmin(gt_mean_x).item())
            right_gt = int(torch.argmax(gt_mean_x).item())

            selected = []
            selected_scores = []
            for q in unmatched_idx.tolist():
                q_valid = valid_prob[b, q] > float(self.boundary_pseudo_valid_thr)
                visible_len = int(q_valid.sum().item())
                if visible_len < int(self.boundary_pseudo_min_valid):
                    continue
                if visible_len > int(self.boundary_pseudo_visible_thr):
                    continue
                if float(self.boundary_pseudo_score_thr) > 0.0:
                    if float(exist_prob[b, q].detach().cpu().item()) < float(self.boundary_pseudo_score_thr):
                        continue

                dists = self._query_to_gt_lane_dist_px(
                    points[b, q],
                    q_valid,
                    gt_points_b,
                    gt_valid_b,
                    width,
                )
                if dists.numel() == 0 or not bool(torch.isfinite(dists).any()):
                    continue

                nearest_gt = int(torch.argmin(dists).item())
                nearest_dist = float(dists[nearest_gt].detach().cpu().item())
                if nearest_gt not in (left_gt, right_gt):
                    continue
                if nearest_dist < float(self.boundary_pseudo_dist_thr):
                    continue
                if float(self.boundary_pseudo_envelope_margin_px) >= 0.0:
                    margin = float(self.boundary_pseudo_envelope_margin_px)
                    side = "left" if nearest_gt == left_gt else "right"
                    outside_ratio = self._query_outside_gt_envelope_ratio(
                        points[b, q],
                        q_valid,
                        gt_points_b,
                        gt_valid_b,
                        width,
                        side=side,
                        margin=margin,
                    )
                    if not bool(torch.isfinite(outside_ratio)):
                        continue
                    if float(outside_ratio.detach().cpu().item()) < float(self.boundary_pseudo_envelope_ratio_thr):
                        continue

                selected.append(q)
                selected_scores.append(exist_prob[b, q].to(device=device, dtype=dtype))

            if not selected:
                continue

            selected_idx = torch.tensor(selected, device=device, dtype=torch.long)
            logits = pred_logits_2d[b, selected_idx]
            loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="mean")
            losses.append(loss)
            pseudo_counts.append(torch.tensor(float(len(selected)), device=device, dtype=dtype))
            pseudo_score_means.append(torch.stack(selected_scores).mean())

        if not losses:
            zero = self._zero_like(pred_points)
            return zero, zero, zero

        return torch.stack(losses).mean(), torch.stack(pseudo_counts).sum(), torch.stack(pseudo_score_means).mean()

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
                    if self._spurious_candidate_gt_protected(
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

        base_indices = self.matcher(pred_points, pred_logits, gt_points, gt_valid)
        indices = base_indices
        full_outputs = self._full_lane_outputs(preds, pred_points)
        full_lane_loss = self._zero_like(pred_points)
        full_lane_point_loss = self._zero_like(pred_points)
        full_lane_valid_loss = self._zero_like(pred_points)
        full_lane_interval_loss = self._zero_like(pred_points)
        full_lane_exist_loss = self._zero_like(pred_points)
        full_lane_quality_loss = self._zero_like(pred_points)
        full_lane_match_count = self._zero_like(pred_points)
        full_lane_unmatched_count = self._zero_like(pred_points)
        if self.full_lane_proposal_gain > 0.0:
            if full_outputs is None:
                raise KeyError(
                    "gcs_full_lane_proposal > 0 requires an independent full-lane proposal head "
                    "that emits pred_full_lane_points/valid/exist/quality/start/end."
                )
            if pred_valid_logits is None:
                raise KeyError("gcs_full_lane_proposal > 0 requires base pred_valid_logits.")
            if self.full_lane_aux_assignment:
                full_indices = self._full_lane_aux_indices(full_outputs, gt_points, gt_valid)
                indices = base_indices
            elif self.full_lane_unified_matching:
                unified_indices = self._full_lane_unified_indices(
                    pred_points,
                    pred_logits,
                    pred_valid_logits,
                    full_outputs,
                    gt_points,
                    gt_valid,
                )
                base_indices, full_indices = self._split_full_lane_indices(
                    unified_indices,
                    base_query_count=int(pred_points.shape[1]),
                )
                indices = base_indices
            else:
                raise RuntimeError(
                    "gcs_full_lane_proposal > 0 reached loss without aux or unified assignment. "
                    "Enable gcs_full_lane_aux_assignment or gcs_full_lane_unified_matching."
                )
            full_lane_target_weights = self._full_lane_focus_target_weights(
                pred_points,
                pred_valid_logits,
                gt_points,
                gt_valid,
            )
            (
                full_lane_loss,
                full_lane_point_loss,
                full_lane_valid_loss,
                full_lane_interval_loss,
                full_lane_exist_loss,
                full_lane_quality_loss,
                full_lane_match_count,
                full_lane_unmatched_count,
            ) = self.full_lane_proposal_loss(
                full_outputs,
                gt_points,
                gt_valid,
                full_indices,
                unmatched_weight=float(self.full_lane_unmatched_weight),
                target_weights=full_lane_target_weights,
            )
        dense_instance_loss = self._zero_like(pred_points)
        dense_centerline_loss = self._zero_like(pred_points)
        dense_endpoint_loss = self._zero_like(pred_points)
        dense_embed_pull_loss = self._zero_like(pred_points)
        dense_embed_push_loss = self._zero_like(pred_points)
        dense_centerline_pos = self._zero_like(pred_points)
        if self.dense_instance_gain > 0.0:
            (
                dense_instance_loss,
                dense_centerline_loss,
                dense_endpoint_loss,
                dense_embed_pull_loss,
                dense_embed_push_loss,
                dense_centerline_pos,
            ) = self.dense_instance_loss(preds, gt_points, gt_valid)
        exist_loss = self.exist_loss(pred_logits, pred_points, pred_valid_logits, gt_points, gt_valid, indices)
        gt_lanes = self.target_lane_count(pred_logits, batch, gt_valid)
        point_loss = self.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=gt_lanes)
        (
            point_valid_loss,
            gt5_short_pos_count,
            gt5_short_pos_anchor_count,
            gt5_short_point_valid_loss,
        ) = self.point_valid_loss(pred_valid_logits, pred_points, gt_valid, indices, gt_lanes=gt_lanes, return_details=True)
        smooth_loss = self.smooth_loss(pred_points, gt_valid, indices)
        curve_loss = self.curve_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=gt_lanes)
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
        boundary_pseudo_neg_loss, boundary_pseudo_count, boundary_pseudo_score_mean = self.boundary_pseudo_neg_loss(
            pred_points,
            pred_logits,
            pred_valid_logits,
            gt_points,
            gt_valid,
            indices,
            gt_lanes,
        )
        query_count_ce_loss, query_count_acc, query_count_pred_mean = self.query_count_ce_loss(
            preds,
            batch,
            gt_valid,
            target_count=gt_lanes,
        )
        (
            short_candidate_loss,
            short_candidate_score_loss,
            short_candidate_pull_loss,
            short_candidate_pos_count,
            short_candidate_soft_count,
            short_candidate_neg_count,
        ) = self.short_candidate_loss(preds, gt_points, gt_valid, gt_lanes)
        (
            short_segment_loss,
            short_segment_score_loss,
            short_segment_point_loss,
            short_segment_pos_count,
            short_segment_soft_count,
            short_segment_neg_count,
        ) = self.short_segment_loss(preds, gt_points, gt_valid, gt_lanes, indices=indices)
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
        if self.full_lane_proposal_gain != 0.0:
            total = total + self.full_lane_proposal_gain * full_lane_loss
        if self.count_boundary_gain != 0.0:
            total = total + self.count_boundary_gain * count_boundary_loss
        if self.boundary_pseudo_neg_gain != 0.0:
            total = total + self.boundary_pseudo_neg_gain * boundary_pseudo_neg_loss
        if self.query_count_ce_gain != 0.0:
            total = total + self.query_count_ce_gain * query_count_ce_loss
        if self.short_candidate_gain != 0.0:
            total = total + self.short_candidate_gain * short_candidate_loss
        if self.short_segment_gain != 0.0:
            total = total + self.short_segment_gain * short_segment_loss
        if self.spurious_neg_gain != 0.0:
            total = total + self.spurious_neg_gain * self.spurious_neg_weight * spurious_neg_loss
        if self.dense_instance_gain != 0.0:
            total = total + self.dense_instance_gain * dense_instance_loss
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
                boundary_pseudo_neg_loss.detach(),
                boundary_pseudo_count.detach(),
                boundary_pseudo_score_mean.detach(),
                short_candidate_loss.detach(),
                short_candidate_score_loss.detach(),
                short_candidate_pull_loss.detach(),
                short_candidate_pos_count.detach(),
                short_candidate_soft_count.detach(),
                short_candidate_neg_count.detach(),
                short_segment_loss.detach(),
                short_segment_score_loss.detach(),
                short_segment_point_loss.detach(),
                short_segment_pos_count.detach(),
                short_segment_soft_count.detach(),
                short_segment_neg_count.detach(),
                query_count_ce_loss.detach(),
                query_count_acc.detach(),
                query_count_pred_mean.detach(),
                full_lane_loss.detach(),
                full_lane_point_loss.detach(),
                full_lane_valid_loss.detach(),
                full_lane_interval_loss.detach(),
                full_lane_exist_loss.detach(),
                full_lane_quality_loss.detach(),
                full_lane_match_count.detach(),
                full_lane_unmatched_count.detach(),
                dense_instance_loss.detach(),
                dense_centerline_loss.detach(),
                dense_endpoint_loss.detach(),
                dense_embed_pull_loss.detach(),
                dense_embed_push_loss.detach(),
                dense_centerline_pos.detach(),
            )
        )
        return total, loss_items
