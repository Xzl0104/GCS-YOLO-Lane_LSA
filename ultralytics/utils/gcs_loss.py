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
        "boundary_pseudo_candidate_count",
        "boundary_pseudo_protected_count",
        "query_count_ce_loss",
        "query_count_acc",
        "query_count_pred_mean",
        "query_quality_loss",
        "query_quality_pos_mean",
        "query_quality_pos_count",
        "query_extent_loss",
        "query_extent_start_acc",
        "query_extent_end_acc",
        "query_extent_iou",
        "query_extent_short_count",
        "short_local_refine_loss",
        "short_local_refine_count",
        "short_local_refine_coarse_ape",
        "short_local_refine_refined_ape",
        "short_local_refine_gain20",
        "short_local_refine_loss20",
        "short_local_refine_identity_count",
        "short_local_refine_pull_count",
        "short_candidate_loss",
        "short_candidate_count",
        "short_candidate_score_loss",
        "short_candidate_geometry_loss",
        "short_candidate_best_offset_px",
        "short_candidate_raw_hit20",
        "short_candidate_best_hit20",
        "role_contain_loss",
        "role_contain_exist_loss",
        "role_contain_valid_loss",
        "role_contain_count",
        "role_contain_gt3_count",
        "role_contain_gt4_gt5bank_count",
        "role_contain_gt4_extra_count",
        "role_contain_allowed_count",
        "q24_event_contain_loss",
        "q24_event_exist_loss",
        "q24_event_valid_loss",
        "q24_event_gt3_count",
        "q24_event_gt4_count",
        "q24_event_gt5_risk_count",
        "q24_event_gt5_risk_protected_count",
        "q24_event_clean_allowed_count",
        "q24_event_dynamic_count",
        "q24_event_dynamic_protected_count",
        "q24_event_score_loss",
        "q24_event_score_pos_count",
        "q24_event_score_prob_mean",
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
        query_quality: float | None = None,
        query_extent: float | None = None,
        query_extent_short_visible_thr: int | None = None,
        query_extent_short_weight: float | None = None,
        query_extent_gt_min_lanes: int | None = None,
        short_local_refine: float | None = None,
        short_local_refine_visible_thr: int | None = None,
        short_local_refine_gt_min_lanes: int | None = None,
        short_local_refine_beta_px: float | None = None,
        short_local_refine_max_delta_px: float | None = None,
        short_local_refine_identity_guard: bool | None = None,
        short_local_refine_identity_thr_px: float | None = None,
        short_local_refine_nearmiss_thr_px: float | None = None,
        short_local_refine_identity_weight: float | None = None,
        short_local_refine_nearmiss_weight: float | None = None,
        short_candidate: float | None = None,
        short_candidate_visible_thr: int | None = None,
        short_candidate_gt_min_lanes: int | None = None,
        short_candidate_beta_px: float | None = None,
        short_candidate_score_temperature: float | None = None,
        short_candidate_step_px: float | None = None,
        role_contain: float | None = None,
        role_contain_valid_weight: float | None = None,
        role_contain_matcher: bool | None = None,
        role_gt5_queries=None,
        role_gt4_queries=None,
        role_gt4_visible_thr: int | None = None,
        role_gt5_visible_thr: int | None = None,
        q24_event_contain: float | None = None,
        q24_event_valid_weight: float | None = None,
        q24_event_matcher: bool | None = None,
        q24_event_clean_gt5_queries=None,
        q24_event_risk_queries=None,
        q24_event_gt4_queries=None,
        q24_event_gt4_visible_thr: int | None = None,
        q24_event_gt5_visible_thr: int | None = None,
        q24_event_suppress_gt5_risk: bool | None = None,
        q24_event_gt5_risk_protect: bool | None = None,
        q24_event_gt5_risk_protect_short_visible_thr: int | None = None,
        q24_event_gt5_risk_protect_dist_px: float | None = None,
        q24_event_gt5_risk_protect_min_overlap: int | None = None,
        q24_event_dynamic: bool | None = None,
        q24_event_dynamic_queries=None,
        q24_event_dynamic_valid_thr: float | None = None,
        q24_event_dynamic_min_valid: int | None = None,
        q24_event_dynamic_max_visible: int | None = None,
        q24_event_dynamic_score_thr: float | None = None,
        q24_event_dynamic_protect: bool | None = None,
        q24_event_dynamic_protect_visible_thr: int | None = None,
        q24_event_dynamic_protect_dist_px: float | None = None,
        q24_event_dynamic_protect_min_overlap: int | None = None,
        q24_event_score_calib: float | None = None,
        q24_event_score_queries=None,
        q24_event_score_visible_thr: int | None = None,
        q24_event_score_valid_thr: float | None = None,
        q24_event_score_dist_px: float | None = None,
        q24_event_score_min_overlap: int | None = None,
        q24_event_score_target: float | None = None,
        boundary_pseudo_gt5_safe: bool | None = None,
        boundary_pseudo_protect_short_visible_thr: int | None = None,
        boundary_pseudo_protect_dist_px: float | None = None,
        boundary_pseudo_protect_min_overlap: int | None = None,
        boundary_pseudo_protect_queries=None,
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
        self.query_quality_gain = float(
            query_quality if query_quality is not None else self._arg(args, "gcs_query_quality", 0.0)
        )
        self.query_extent_gain = float(
            query_extent if query_extent is not None else self._arg(args, "gcs_query_extent", 0.0)
        )
        self.query_extent_short_visible_thr = int(
            query_extent_short_visible_thr
            if query_extent_short_visible_thr is not None
            else self._arg(args, "gcs_query_extent_short_visible_thr", 10)
        )
        self.query_extent_short_weight = float(
            query_extent_short_weight
            if query_extent_short_weight is not None
            else self._arg(args, "gcs_query_extent_short_weight", 2.0)
        )
        self.query_extent_gt_min_lanes = int(
            query_extent_gt_min_lanes
            if query_extent_gt_min_lanes is not None
            else self._arg(args, "gcs_query_extent_gt_min_lanes", 4)
        )
        self.short_local_refine_gain = float(
            short_local_refine
            if short_local_refine is not None
            else self._arg(args, "gcs_short_local_refine", 0.0)
        )
        self.short_local_refine_visible_thr = int(
            short_local_refine_visible_thr
            if short_local_refine_visible_thr is not None
            else self._arg(args, "gcs_short_local_refine_visible_thr", 10)
        )
        self.short_local_refine_gt_min_lanes = int(
            short_local_refine_gt_min_lanes
            if short_local_refine_gt_min_lanes is not None
            else self._arg(args, "gcs_short_local_refine_gt_min_lanes", 4)
        )
        self.short_local_refine_beta_px = float(
            short_local_refine_beta_px
            if short_local_refine_beta_px is not None
            else self._arg(args, "gcs_short_local_refine_beta_px", 5.0)
        )
        self.short_local_refine_max_delta_px = float(
            short_local_refine_max_delta_px
            if short_local_refine_max_delta_px is not None
            else self._arg(args, "gcs_short_local_refine_max_delta_px", 40.0)
        )
        self.short_local_refine_identity_guard = self._bool_arg(
            short_local_refine_identity_guard
            if short_local_refine_identity_guard is not None
            else self._arg(args, "gcs_short_local_refine_identity_guard", False)
        )
        self.short_local_refine_identity_thr_px = float(
            short_local_refine_identity_thr_px
            if short_local_refine_identity_thr_px is not None
            else self._arg(args, "gcs_short_local_refine_identity_thr_px", 20.0)
        )
        self.short_local_refine_nearmiss_thr_px = float(
            short_local_refine_nearmiss_thr_px
            if short_local_refine_nearmiss_thr_px is not None
            else self._arg(args, "gcs_short_local_refine_nearmiss_thr_px", 80.0)
        )
        self.short_local_refine_identity_weight = float(
            short_local_refine_identity_weight
            if short_local_refine_identity_weight is not None
            else self._arg(args, "gcs_short_local_refine_identity_weight", 1.0)
        )
        self.short_local_refine_nearmiss_weight = float(
            short_local_refine_nearmiss_weight
            if short_local_refine_nearmiss_weight is not None
            else self._arg(args, "gcs_short_local_refine_nearmiss_weight", 1.0)
        )
        self.short_candidate_gain = float(
            short_candidate if short_candidate is not None else self._arg(args, "gcs_short_candidate", 0.0)
        )
        self.short_candidate_visible_thr = int(
            short_candidate_visible_thr
            if short_candidate_visible_thr is not None
            else self._arg(args, "gcs_short_candidate_visible_thr", 10)
        )
        self.short_candidate_gt_min_lanes = int(
            short_candidate_gt_min_lanes
            if short_candidate_gt_min_lanes is not None
            else self._arg(args, "gcs_short_candidate_gt_min_lanes", 4)
        )
        self.short_candidate_beta_px = float(
            short_candidate_beta_px
            if short_candidate_beta_px is not None
            else self._arg(args, "gcs_short_candidate_beta_px", 3.0)
        )
        self.short_candidate_score_temperature = float(
            short_candidate_score_temperature
            if short_candidate_score_temperature is not None
            else self._arg(args, "gcs_short_candidate_score_temperature", 1.0)
        )
        self.short_candidate_step_px = float(
            short_candidate_step_px
            if short_candidate_step_px is not None
            else self._arg(args, "gcs_short_candidate_step_px", 20.0)
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
        self.role_contain_gain = float(
            role_contain if role_contain is not None else self._arg(args, "gcs_role_contain", 0.0)
        )
        self.role_contain_valid_weight = float(
            role_contain_valid_weight
            if role_contain_valid_weight is not None
            else self._arg(args, "gcs_role_contain_valid_weight", 0.25)
        )
        self.role_contain_matcher = self._bool_arg(
            role_contain_matcher
            if role_contain_matcher is not None
            else self._arg(args, "gcs_role_contain_matcher", False)
        )
        self.role_gt5_queries = self._parse_query_spec(
            role_gt5_queries if role_gt5_queries is not None else self._arg(args, "gcs_role_gt5_queries", "12-17")
        )
        self.role_gt4_queries = self._parse_query_spec(
            role_gt4_queries if role_gt4_queries is not None else self._arg(args, "gcs_role_gt4_queries", "18-23")
        )
        self.role_gt4_visible_thr = int(
            role_gt4_visible_thr
            if role_gt4_visible_thr is not None
            else self._arg(args, "gcs_role_gt4_visible_thr", 20)
        )
        self.role_gt5_visible_thr = int(
            role_gt5_visible_thr
            if role_gt5_visible_thr is not None
            else self._arg(args, "gcs_role_gt5_visible_thr", 20)
        )
        self.q24_event_contain_gain = float(
            q24_event_contain
            if q24_event_contain is not None
            else self._arg(args, "gcs_q24_event_contain", 0.0)
        )
        self.q24_event_valid_weight = float(
            q24_event_valid_weight
            if q24_event_valid_weight is not None
            else self._arg(args, "gcs_q24_event_valid_weight", 0.25)
        )
        self.q24_event_matcher = self._bool_arg(
            q24_event_matcher
            if q24_event_matcher is not None
            else self._arg(args, "gcs_q24_event_matcher", False)
        )
        self.q24_event_clean_gt5_queries = self._parse_query_spec(
            q24_event_clean_gt5_queries
            if q24_event_clean_gt5_queries is not None
            else self._arg(args, "gcs_q24_event_clean_gt5_queries", "")
        )
        self.q24_event_risk_queries = self._parse_query_spec(
            q24_event_risk_queries
            if q24_event_risk_queries is not None
            else self._arg(args, "gcs_q24_event_risk_queries", "")
        )
        self.q24_event_gt4_queries = self._parse_query_spec(
            q24_event_gt4_queries
            if q24_event_gt4_queries is not None
            else self._arg(args, "gcs_q24_event_gt4_queries", "")
        )
        self.q24_event_gt4_visible_thr = int(
            q24_event_gt4_visible_thr
            if q24_event_gt4_visible_thr is not None
            else self._arg(args, "gcs_q24_event_gt4_visible_thr", 20)
        )
        self.q24_event_gt5_visible_thr = int(
            q24_event_gt5_visible_thr
            if q24_event_gt5_visible_thr is not None
            else self._arg(args, "gcs_q24_event_gt5_visible_thr", 10)
        )
        self.q24_event_suppress_gt5_risk = self._bool_arg(
            q24_event_suppress_gt5_risk
            if q24_event_suppress_gt5_risk is not None
            else self._arg(args, "gcs_q24_event_suppress_gt5_risk", False)
        )
        self.q24_event_gt5_risk_protect = self._bool_arg(
            q24_event_gt5_risk_protect
            if q24_event_gt5_risk_protect is not None
            else self._arg(args, "gcs_q24_event_gt5_risk_protect", False)
        )
        self.q24_event_gt5_risk_protect_short_visible_thr = int(
            q24_event_gt5_risk_protect_short_visible_thr
            if q24_event_gt5_risk_protect_short_visible_thr is not None
            else self._arg(args, "gcs_q24_event_gt5_risk_protect_short_visible_thr", 10)
        )
        self.q24_event_gt5_risk_protect_dist_px = float(
            q24_event_gt5_risk_protect_dist_px
            if q24_event_gt5_risk_protect_dist_px is not None
            else self._arg(args, "gcs_q24_event_gt5_risk_protect_dist_px", 30.0)
        )
        self.q24_event_gt5_risk_protect_min_overlap = int(
            q24_event_gt5_risk_protect_min_overlap
            if q24_event_gt5_risk_protect_min_overlap is not None
            else self._arg(args, "gcs_q24_event_gt5_risk_protect_min_overlap", 3)
        )
        self.q24_event_dynamic = self._bool_arg(
            q24_event_dynamic
            if q24_event_dynamic is not None
            else self._arg(args, "gcs_q24_event_dynamic", False)
        )
        self.q24_event_dynamic_queries = self._parse_query_spec(
            q24_event_dynamic_queries
            if q24_event_dynamic_queries is not None
            else self._arg(args, "gcs_q24_event_dynamic_queries", "")
        )
        self.q24_event_dynamic_valid_thr = float(
            q24_event_dynamic_valid_thr
            if q24_event_dynamic_valid_thr is not None
            else self._arg(args, "gcs_q24_event_dynamic_valid_thr", 0.5)
        )
        self.q24_event_dynamic_min_valid = int(
            q24_event_dynamic_min_valid
            if q24_event_dynamic_min_valid is not None
            else self._arg(args, "gcs_q24_event_dynamic_min_valid", 3)
        )
        self.q24_event_dynamic_max_visible = int(
            q24_event_dynamic_max_visible
            if q24_event_dynamic_max_visible is not None
            else self._arg(args, "gcs_q24_event_dynamic_max_visible", 20)
        )
        self.q24_event_dynamic_score_thr = float(
            q24_event_dynamic_score_thr
            if q24_event_dynamic_score_thr is not None
            else self._arg(args, "gcs_q24_event_dynamic_score_thr", 0.0)
        )
        self.q24_event_dynamic_protect = self._bool_arg(
            q24_event_dynamic_protect
            if q24_event_dynamic_protect is not None
            else self._arg(args, "gcs_q24_event_dynamic_protect", False)
        )
        self.q24_event_dynamic_protect_visible_thr = int(
            q24_event_dynamic_protect_visible_thr
            if q24_event_dynamic_protect_visible_thr is not None
            else self._arg(args, "gcs_q24_event_dynamic_protect_visible_thr", 10)
        )
        self.q24_event_dynamic_protect_dist_px = float(
            q24_event_dynamic_protect_dist_px
            if q24_event_dynamic_protect_dist_px is not None
            else self._arg(args, "gcs_q24_event_dynamic_protect_dist_px", 20.0)
        )
        self.q24_event_dynamic_protect_min_overlap = int(
            q24_event_dynamic_protect_min_overlap
            if q24_event_dynamic_protect_min_overlap is not None
            else self._arg(args, "gcs_q24_event_dynamic_protect_min_overlap", 3)
        )
        self.q24_event_score_calib_gain = float(
            q24_event_score_calib
            if q24_event_score_calib is not None
            else self._arg(args, "gcs_q24_event_score_calib", 0.0)
        )
        self.q24_event_score_queries = self._parse_query_spec(
            q24_event_score_queries
            if q24_event_score_queries is not None
            else self._arg(args, "gcs_q24_event_score_queries", "")
        )
        self.q24_event_score_visible_thr = int(
            q24_event_score_visible_thr
            if q24_event_score_visible_thr is not None
            else self._arg(args, "gcs_q24_event_score_visible_thr", 10)
        )
        self.q24_event_score_valid_thr = float(
            q24_event_score_valid_thr
            if q24_event_score_valid_thr is not None
            else self._arg(args, "gcs_q24_event_score_valid_thr", 0.5)
        )
        self.q24_event_score_dist_px = float(
            q24_event_score_dist_px
            if q24_event_score_dist_px is not None
            else self._arg(args, "gcs_q24_event_score_dist_px", 40.0)
        )
        self.q24_event_score_min_overlap = int(
            q24_event_score_min_overlap
            if q24_event_score_min_overlap is not None
            else self._arg(args, "gcs_q24_event_score_min_overlap", 3)
        )
        self.q24_event_score_target = float(
            q24_event_score_target
            if q24_event_score_target is not None
            else self._arg(args, "gcs_q24_event_score_target", 0.75)
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
        if self.query_quality_gain < 0.0:
            raise ValueError(f"gcs_query_quality must be >= 0, got {self.query_quality_gain}.")
        if self.query_extent_gain < 0.0:
            raise ValueError(f"gcs_query_extent must be >= 0, got {self.query_extent_gain}.")
        if self.query_extent_short_visible_thr < 0:
            raise ValueError(
                "gcs_query_extent_short_visible_thr must be >= 0, "
                f"got {self.query_extent_short_visible_thr}."
            )
        if self.query_extent_short_weight < 0.0:
            raise ValueError(
                "gcs_query_extent_short_weight must be >= 0, "
                f"got {self.query_extent_short_weight}."
            )
        if self.query_extent_gt_min_lanes < 0:
            raise ValueError(
                "gcs_query_extent_gt_min_lanes must be >= 0, "
                f"got {self.query_extent_gt_min_lanes}."
            )
        if self.short_local_refine_gain < 0.0:
            raise ValueError(f"gcs_short_local_refine must be >= 0, got {self.short_local_refine_gain}.")
        if self.short_local_refine_visible_thr < 0:
            raise ValueError(
                "gcs_short_local_refine_visible_thr must be >= 0, "
                f"got {self.short_local_refine_visible_thr}."
            )
        if self.short_local_refine_gt_min_lanes < 0:
            raise ValueError(
                "gcs_short_local_refine_gt_min_lanes must be >= 0, "
                f"got {self.short_local_refine_gt_min_lanes}."
            )
        if self.short_local_refine_beta_px <= 0.0:
            raise ValueError(f"gcs_short_local_refine_beta_px must be > 0, got {self.short_local_refine_beta_px}.")
        if self.short_local_refine_max_delta_px <= 0.0:
            raise ValueError(
                f"gcs_short_local_refine_max_delta_px must be > 0, got {self.short_local_refine_max_delta_px}."
            )
        if self.short_local_refine_identity_thr_px < 0.0:
            raise ValueError(
                "gcs_short_local_refine_identity_thr_px must be >= 0, "
                f"got {self.short_local_refine_identity_thr_px}."
            )
        if self.short_local_refine_nearmiss_thr_px < self.short_local_refine_identity_thr_px:
            raise ValueError(
                "gcs_short_local_refine_nearmiss_thr_px must be >= "
                "gcs_short_local_refine_identity_thr_px, got "
                f"{self.short_local_refine_nearmiss_thr_px} < {self.short_local_refine_identity_thr_px}."
            )
        if self.short_local_refine_identity_weight < 0.0:
            raise ValueError(
                "gcs_short_local_refine_identity_weight must be >= 0, "
                f"got {self.short_local_refine_identity_weight}."
            )
        if self.short_local_refine_nearmiss_weight < 0.0:
            raise ValueError(
                "gcs_short_local_refine_nearmiss_weight must be >= 0, "
                f"got {self.short_local_refine_nearmiss_weight}."
            )
        if self.short_candidate_gain < 0.0:
            raise ValueError(f"gcs_short_candidate must be >= 0, got {self.short_candidate_gain}.")
        if self.short_candidate_visible_thr < 0:
            raise ValueError(
                f"gcs_short_candidate_visible_thr must be >= 0, got {self.short_candidate_visible_thr}."
            )
        if self.short_candidate_gt_min_lanes < 0:
            raise ValueError(
                f"gcs_short_candidate_gt_min_lanes must be >= 0, got {self.short_candidate_gt_min_lanes}."
            )
        if self.short_candidate_beta_px <= 0.0:
            raise ValueError(f"gcs_short_candidate_beta_px must be > 0, got {self.short_candidate_beta_px}.")
        if self.short_candidate_score_temperature <= 0.0:
            raise ValueError(
                "gcs_short_candidate_score_temperature must be > 0, "
                f"got {self.short_candidate_score_temperature}."
            )
        if self.short_candidate_step_px <= 0.0:
            raise ValueError(f"gcs_short_candidate_step_px must be > 0, got {self.short_candidate_step_px}.")
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
        if self.role_contain_gain < 0.0:
            raise ValueError(f"gcs_role_contain must be >= 0, got {self.role_contain_gain}.")
        if self.role_contain_valid_weight < 0.0:
            raise ValueError(
                "gcs_role_contain_valid_weight must be >= 0, "
                f"got {self.role_contain_valid_weight}."
            )
        if self.role_gt4_visible_thr < 0:
            raise ValueError(f"gcs_role_gt4_visible_thr must be >= 0, got {self.role_gt4_visible_thr}.")
        if self.role_gt5_visible_thr < 0:
            raise ValueError(f"gcs_role_gt5_visible_thr must be >= 0, got {self.role_gt5_visible_thr}.")
        role_overlap = set(self.role_gt5_queries) & set(self.role_gt4_queries)
        if role_overlap:
            raise ValueError(f"gcs_role_gt5_queries and gcs_role_gt4_queries must not overlap: {sorted(role_overlap)}.")
        if self.q24_event_contain_gain < 0.0:
            raise ValueError(f"gcs_q24_event_contain must be >= 0, got {self.q24_event_contain_gain}.")
        if self.q24_event_valid_weight < 0.0:
            raise ValueError(
                "gcs_q24_event_valid_weight must be >= 0, "
                f"got {self.q24_event_valid_weight}."
            )
        if self.q24_event_gt4_visible_thr < 0:
            raise ValueError(f"gcs_q24_event_gt4_visible_thr must be >= 0, got {self.q24_event_gt4_visible_thr}.")
        if self.q24_event_gt5_visible_thr < 0:
            raise ValueError(f"gcs_q24_event_gt5_visible_thr must be >= 0, got {self.q24_event_gt5_visible_thr}.")
        if self.q24_event_gt5_risk_protect_short_visible_thr < 0:
            raise ValueError(
                "gcs_q24_event_gt5_risk_protect_short_visible_thr must be >= 0, "
                f"got {self.q24_event_gt5_risk_protect_short_visible_thr}."
            )
        if self.q24_event_gt5_risk_protect_dist_px < 0.0:
            raise ValueError(
                "gcs_q24_event_gt5_risk_protect_dist_px must be >= 0, "
                f"got {self.q24_event_gt5_risk_protect_dist_px}."
            )
        if self.q24_event_gt5_risk_protect_min_overlap < 1:
            raise ValueError(
                "gcs_q24_event_gt5_risk_protect_min_overlap must be >= 1, "
                f"got {self.q24_event_gt5_risk_protect_min_overlap}."
            )
        if not (0.0 <= self.q24_event_dynamic_valid_thr <= 1.0):
            raise ValueError("gcs_q24_event_dynamic_valid_thr must be in [0, 1].")
        if self.q24_event_dynamic_min_valid < 0:
            raise ValueError("gcs_q24_event_dynamic_min_valid must be >= 0.")
        if self.q24_event_dynamic_max_visible < self.q24_event_dynamic_min_valid:
            raise ValueError(
                "gcs_q24_event_dynamic_max_visible must be >= gcs_q24_event_dynamic_min_valid "
                f"({self.q24_event_dynamic_max_visible} < {self.q24_event_dynamic_min_valid})."
            )
        if self.q24_event_dynamic_score_thr < 0.0:
            raise ValueError("gcs_q24_event_dynamic_score_thr must be >= 0.")
        if self.q24_event_dynamic_protect_visible_thr < 0:
            raise ValueError("gcs_q24_event_dynamic_protect_visible_thr must be >= 0.")
        if self.q24_event_dynamic_protect_dist_px < 0.0:
            raise ValueError("gcs_q24_event_dynamic_protect_dist_px must be >= 0.")
        if self.q24_event_dynamic_protect_min_overlap < 1:
            raise ValueError("gcs_q24_event_dynamic_protect_min_overlap must be >= 1.")
        if self.q24_event_dynamic and not self.q24_event_dynamic_queries:
            raise ValueError("gcs_q24_event_dynamic requires non-empty gcs_q24_event_dynamic_queries.")
        if self.q24_event_score_calib_gain < 0.0:
            raise ValueError("gcs_q24_event_score_calib must be >= 0.")
        if self.q24_event_score_visible_thr < 0:
            raise ValueError("gcs_q24_event_score_visible_thr must be >= 0.")
        if not (0.0 <= self.q24_event_score_valid_thr <= 1.0):
            raise ValueError("gcs_q24_event_score_valid_thr must be in [0, 1].")
        if self.q24_event_score_dist_px < 0.0:
            raise ValueError("gcs_q24_event_score_dist_px must be >= 0.")
        if self.q24_event_score_min_overlap < 1:
            raise ValueError("gcs_q24_event_score_min_overlap must be >= 1.")
        if not (0.0 <= self.q24_event_score_target <= 1.0):
            raise ValueError("gcs_q24_event_score_target must be in [0, 1].")
        if self.q24_event_score_calib_gain > 0.0 and not self.q24_event_score_queries:
            raise ValueError("gcs_q24_event_score_calib requires non-empty gcs_q24_event_score_queries.")
        event_sets = (
            ("clean_gt5", set(self.q24_event_clean_gt5_queries)),
            ("risk", set(self.q24_event_risk_queries)),
            ("gt4", set(self.q24_event_gt4_queries)),
        )
        for i, (name_i, set_i) in enumerate(event_sets):
            for name_j, set_j in event_sets[i + 1 :]:
                overlap = set_i & set_j
                if overlap:
                    raise ValueError(
                        f"gcs_q24_event_{name_i}_queries and gcs_q24_event_{name_j}_queries "
                        f"must not overlap: {sorted(overlap)}."
                    )
        if self._q24_event_enabled() and not any(values for _, values in event_sets):
            raise ValueError("Q24 event containment requires at least one clean, risk, or GT4 query set.")
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
        self.boundary_pseudo_gt5_safe = self._bool_arg(
            boundary_pseudo_gt5_safe
            if boundary_pseudo_gt5_safe is not None
            else self._arg(args, "gcs_boundary_pseudo_gt5_safe", False)
        )
        self.boundary_pseudo_protect_short_visible_thr = int(
            boundary_pseudo_protect_short_visible_thr
            if boundary_pseudo_protect_short_visible_thr is not None
            else self._arg(args, "gcs_boundary_pseudo_protect_short_visible_thr", 10)
        )
        self.boundary_pseudo_protect_dist_px = float(
            boundary_pseudo_protect_dist_px
            if boundary_pseudo_protect_dist_px is not None
            else self._arg(args, "gcs_boundary_pseudo_protect_dist_px", 40.0)
        )
        self.boundary_pseudo_protect_min_overlap = int(
            boundary_pseudo_protect_min_overlap
            if boundary_pseudo_protect_min_overlap is not None
            else self._arg(args, "gcs_boundary_pseudo_protect_min_overlap", 3)
        )
        self.boundary_pseudo_protect_queries = self._parse_query_spec(
            boundary_pseudo_protect_queries
            if boundary_pseudo_protect_queries is not None
            else self._arg(args, "gcs_boundary_pseudo_protect_queries", "")
        )

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
        if self.boundary_pseudo_gt5_safe and int(self.boundary_pseudo_gt_count) != 5:
            raise ValueError("gcs_boundary_pseudo_gt5_safe requires gcs_boundary_pseudo_gt_count=5.")
        if self.boundary_pseudo_gt5_safe and float(self.boundary_pseudo_envelope_margin_px) < 0.0:
            raise ValueError(
                "gcs_boundary_pseudo_gt5_safe requires gcs_boundary_pseudo_envelope_margin_px >= 0 "
                "so selected negatives are clear boundary-pseudo lanes."
            )
        if self.boundary_pseudo_protect_short_visible_thr < 0:
            raise ValueError("gcs_boundary_pseudo_protect_short_visible_thr must be >= 0.")
        if self.boundary_pseudo_protect_dist_px < 0.0:
            raise ValueError("gcs_boundary_pseudo_protect_dist_px must be >= 0.")
        if self.boundary_pseudo_protect_min_overlap < 1:
            raise ValueError("gcs_boundary_pseudo_protect_min_overlap must be >= 1.")

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
    def _parse_query_spec(value) -> tuple[int, ...]:
        """Parse a query list like '12-17,21' into sorted unique indices."""
        if value is None:
            return ()
        if isinstance(value, (list, tuple, set)):
            values = [int(v) for v in value]
        else:
            text = str(value).strip()
            if text.lower() in {"", "none", "false", "off"}:
                return ()
            values = []
            for raw_part in text.split(","):
                part = raw_part.strip()
                if not part:
                    continue
                if "-" in part:
                    start_s, end_s = part.split("-", 1)
                    start = int(start_s.strip())
                    end = int(end_s.strip())
                    step = 1 if end >= start else -1
                    values.extend(range(start, end + step, step))
                else:
                    values.append(int(part))
        if any(v < 0 for v in values):
            raise ValueError(f"Query indices must be non-negative, got {values}.")
        return tuple(sorted(set(values)))

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

    def _role_enabled(self) -> bool:
        """Return whether Q24 role-containment behavior is active."""
        return bool(float(self.role_contain_gain) > 0.0 or self.role_contain_matcher)

    def _role_query_tensors(self, device: torch.device, num_queries: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return GT5, GT4, and combined extra query tensors for role containment."""
        gt5_queries = torch.as_tensor(self.role_gt5_queries, device=device, dtype=torch.long)
        gt4_queries = torch.as_tensor(self.role_gt4_queries, device=device, dtype=torch.long)
        if gt5_queries.numel() == 0 and gt4_queries.numel() == 0:
            raise ValueError("Q24 role containment requires at least one configured extra query.")
        max_query = -1
        if gt5_queries.numel():
            max_query = max(max_query, int(gt5_queries.max().item()))
        if gt4_queries.numel():
            max_query = max(max_query, int(gt4_queries.max().item()))
        if max_query >= num_queries:
            raise ValueError(
                "Q24 role containment query range exceeds model query count: "
                f"max configured q{max_query}, model has Q={num_queries}."
            )
        extra_queries = torch.cat([gt5_queries, gt4_queries]) if gt5_queries.numel() and gt4_queries.numel() else (
            gt5_queries if gt5_queries.numel() else gt4_queries
        )
        return gt5_queries, gt4_queries, extra_queries.unique(sorted=True)

    def _role_allowed_masks(
        self,
        pred_points: torch.Tensor,
        gt_valid: list[torch.Tensor],
        gt_lanes: torch.Tensor,
    ) -> list[torch.Tensor] | None:
        """Build per-image Q x N masks for role-aware Hungarian matching."""
        if not self._role_enabled():
            return None

        device = pred_points.device
        num_queries = int(pred_points.shape[1])
        gt5_queries, gt4_queries, extra_queries = self._role_query_tensors(device, num_queries)
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=pred_points.dtype).reshape(-1)
        if gt_lanes.numel() != pred_points.shape[0]:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={pred_points.shape[0]}.")

        masks: list[torch.Tensor] = []
        for b, valid_b_raw in enumerate(gt_valid):
            valid_b = valid_b_raw.to(device=device, dtype=pred_points.dtype)
            if valid_b.ndim != 2:
                raise ValueError(f"GT lane_valid must have shape N x K, got {tuple(valid_b.shape)}.")
            mask = torch.ones((num_queries, valid_b.shape[0]), device=device, dtype=torch.bool)
            if valid_b.shape[0] == 0:
                masks.append(mask)
                continue

            gt_count = int(round(float(gt_lanes[b].detach().item())))
            visible_counts = valid_b.float().sum(dim=1)
            gt4_short = visible_counts <= float(self.role_gt4_visible_thr)
            gt5_short = visible_counts <= float(self.role_gt5_visible_thr)

            if gt_count <= 3:
                mask[extra_queries, :] = False
            elif gt_count == 4:
                if gt5_queries.numel():
                    mask[gt5_queries, :] = False
                if gt4_queries.numel():
                    mask[gt4_queries, :] = gt4_short[None, :]
            else:
                if gt5_queries.numel() and self.role_gt5_visible_thr > 0:
                    mask[gt5_queries, :] = gt5_short[None, :]
                if gt4_queries.numel():
                    mask[gt4_queries, :] = False
            masks.append(mask)
        return masks

    def role_containment_loss(
        self,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
        role_allowed_masks: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, ...]:
        """Suppress Q24 extra-bank queries when they violate their intended GT-count role."""
        zero = pred_logits.new_zeros(())
        if float(self.role_contain_gain) <= 0.0:
            return zero, zero, zero, zero, zero, zero, zero, zero
        if pred_valid_logits is None and float(self.role_contain_valid_weight) > 0.0:
            raise ValueError(
                "gcs_role_contain with gcs_role_contain_valid_weight > 0 requires pred_valid_logits."
            )

        device = pred_logits.device
        num_queries = int(pred_logits.shape[1])
        gt5_queries, gt4_queries, extra_queries = self._role_query_tensors(device, num_queries)
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=pred_logits.dtype).reshape(-1)
        if gt_lanes.numel() != pred_logits.shape[0]:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={pred_logits.shape[0]}.")
        if role_allowed_masks is None:
            raise ValueError("role_allowed_masks must be provided when gcs_role_contain is enabled.")

        exist_losses = []
        valid_losses = []
        total_count = 0
        gt3_count = 0
        gt4_gt5bank_count = 0
        gt4_extra_count = 0
        allowed_count = 0

        for b in range(pred_logits.shape[0]):
            gt_count = int(round(float(gt_lanes[b].detach().item())))
            selected = torch.zeros((num_queries,), device=device, dtype=torch.bool)

            if gt_count <= 3:
                selected[extra_queries] = True
                gt3_count += int(extra_queries.numel())
            elif gt_count == 4:
                if gt5_queries.numel():
                    selected[gt5_queries] = True
                    gt4_gt5bank_count += int(gt5_queries.numel())
                if gt4_queries.numel():
                    allowed_positive = torch.zeros((num_queries,), device=device, dtype=torch.bool)
                    src_idx, tgt_idx = indices[b]
                    if src_idx.numel():
                        src_idx = src_idx.to(device=device, dtype=torch.long)
                        tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
                        allowed_mask_b = role_allowed_masks[b].to(device=device, dtype=torch.bool)
                        for q, t in zip(src_idx.tolist(), tgt_idx.tolist()):
                            if 0 <= q < num_queries and 0 <= t < allowed_mask_b.shape[1] and bool(allowed_mask_b[q, t]):
                                allowed_positive[q] = True
                    allowed_gt4 = allowed_positive[gt4_queries]
                    if bool(allowed_gt4.any()):
                        allowed_count += int(allowed_gt4.sum().item())
                    suppressed_gt4 = gt4_queries[~allowed_gt4]
                    if suppressed_gt4.numel():
                        selected[suppressed_gt4] = True
                        gt4_extra_count += int(suppressed_gt4.numel())
            else:
                continue

            if not bool(selected.any()):
                continue

            selected_idx = torch.nonzero(selected, as_tuple=False).flatten()
            total_count += int(selected_idx.numel())
            exist_losses.append(
                F.binary_cross_entropy_with_logits(
                    pred_logits[b, selected_idx],
                    torch.zeros_like(pred_logits[b, selected_idx]),
                    reduction="mean",
                )
            )
            if pred_valid_logits is not None and float(self.role_contain_valid_weight) > 0.0:
                valid_losses.append(
                    F.binary_cross_entropy_with_logits(
                        pred_valid_logits[b, selected_idx],
                        torch.zeros_like(pred_valid_logits[b, selected_idx]),
                        reduction="mean",
                    )
                )

        exist_loss = torch.stack(exist_losses).mean() if exist_losses else zero
        valid_loss = torch.stack(valid_losses).mean() if valid_losses else zero
        loss = exist_loss + float(self.role_contain_valid_weight) * valid_loss
        return (
            loss,
            exist_loss,
            valid_loss,
            pred_logits.new_tensor(float(total_count)),
            pred_logits.new_tensor(float(gt3_count)),
            pred_logits.new_tensor(float(gt4_gt5bank_count)),
            pred_logits.new_tensor(float(gt4_extra_count)),
            pred_logits.new_tensor(float(allowed_count)),
        )

    def _q24_event_enabled(self) -> bool:
        """Return whether Q24 event-aware containment behavior is active."""
        return bool(float(self.q24_event_contain_gain) > 0.0 or self.q24_event_matcher)

    def _q24_event_query_tensors(
        self, device: torch.device, num_queries: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return clean-GT5, high-risk, GT4, and combined Q24 event query tensors."""
        clean_gt5 = torch.as_tensor(self.q24_event_clean_gt5_queries, device=device, dtype=torch.long)
        risk = torch.as_tensor(self.q24_event_risk_queries, device=device, dtype=torch.long)
        gt4 = torch.as_tensor(self.q24_event_gt4_queries, device=device, dtype=torch.long)
        groups = [x for x in (clean_gt5, risk, gt4) if x.numel()]
        if not groups:
            raise ValueError("Q24 event containment requires at least one configured query.")
        max_query = max(int(x.max().item()) for x in groups)
        if max_query >= num_queries:
            raise ValueError(
                "Q24 event containment query range exceeds model query count: "
                f"max configured q{max_query}, model has Q={num_queries}."
            )
        all_event = torch.cat(groups).unique(sorted=True)
        return clean_gt5, risk, gt4, all_event

    def _q24_event_allowed_masks(
        self,
        pred_points: torch.Tensor,
        gt_valid: list[torch.Tensor],
        gt_lanes: torch.Tensor,
    ) -> list[torch.Tensor] | None:
        """Build per-image Q x N masks for event-aware Q24 Hungarian matching."""
        if not self._q24_event_enabled():
            return None

        device = pred_points.device
        num_queries = int(pred_points.shape[1])
        clean_gt5, risk, gt4_queries, all_event = self._q24_event_query_tensors(device, num_queries)
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=pred_points.dtype).reshape(-1)
        if gt_lanes.numel() != pred_points.shape[0]:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={pred_points.shape[0]}.")

        masks: list[torch.Tensor] = []
        for b, valid_b_raw in enumerate(gt_valid):
            valid_b = valid_b_raw.to(device=device, dtype=pred_points.dtype)
            if valid_b.ndim != 2:
                raise ValueError(f"GT lane_valid must have shape N x K, got {tuple(valid_b.shape)}.")
            mask = torch.ones((num_queries, valid_b.shape[0]), device=device, dtype=torch.bool)
            if valid_b.shape[0] == 0:
                masks.append(mask)
                continue

            gt_count = int(round(float(gt_lanes[b].detach().item())))
            visible_counts = valid_b.float().sum(dim=1)
            gt4_short = visible_counts <= float(self.q24_event_gt4_visible_thr)
            gt5_short = visible_counts <= float(self.q24_event_gt5_visible_thr)

            if gt_count <= 3:
                mask[all_event, :] = False
            elif gt_count == 4:
                if clean_gt5.numel():
                    mask[clean_gt5, :] = False
                if risk.numel():
                    mask[risk, :] = False
                if gt4_queries.numel():
                    mask[gt4_queries, :] = gt4_short[None, :]
            else:
                if clean_gt5.numel():
                    mask[clean_gt5, :] = gt5_short[None, :] if self.q24_event_gt5_visible_thr > 0 else True
                if risk.numel():
                    mask[risk, :] = False
                if gt4_queries.numel():
                    mask[gt4_queries, :] = False
            masks.append(mask)
        return masks

    @staticmethod
    def _merge_allowed_masks(*mask_lists: list[torch.Tensor] | None) -> list[torch.Tensor] | None:
        """Combine optional matcher allow masks with logical AND."""
        active = [m for m in mask_lists if m is not None]
        if not active:
            return None
        merged = [m.clone() for m in active[0]]
        for masks in active[1:]:
            if len(masks) != len(merged):
                raise ValueError("Allowed-mask lists must have the same batch size.")
            for i, mask in enumerate(masks):
                if mask.shape != merged[i].shape:
                    raise ValueError(f"Allowed-mask shape mismatch: {tuple(mask.shape)} vs {tuple(merged[i].shape)}.")
                merged[i] = merged[i] & mask.to(device=merged[i].device, dtype=torch.bool)
        return merged

    def _q24_event_risk_protected_by_short_gt5(
        self,
        pred_points_q: torch.Tensor,
        pred_visible_q: torch.Tensor,
        gt_points_b: torch.Tensor,
        gt_valid_b: torch.Tensor,
        width: float,
    ) -> bool:
        """Return whether a high-risk query is close enough to a true short GT5 lane to skip extra negative pressure."""
        if not self.q24_event_gt5_risk_protect:
            return False
        short_thr = int(self.q24_event_gt5_risk_protect_short_visible_thr)
        if short_thr <= 0:
            return False

        q_visible = pred_visible_q if pred_visible_q.dtype == torch.bool else pred_visible_q > 0.5
        pred_x = pred_points_q[:, 0] * float(width)
        gt_x = gt_points_b[..., 0] * float(width)
        min_overlap = int(self.q24_event_gt5_risk_protect_min_overlap)
        protect_dist = float(self.q24_event_gt5_risk_protect_dist_px)

        for gt_i in range(gt_points_b.shape[0]):
            gt_visible = gt_valid_b[gt_i] > 0.5
            if int(gt_visible.sum().item()) > short_thr:
                continue
            common = q_visible & gt_visible
            if int(common.sum().item()) < min_overlap:
                continue
            mean_dx = (pred_x[common] - gt_x[gt_i, common]).abs().mean()
            if float(mean_dx.detach().cpu().item()) <= protect_dist:
                return True
        return False

    @staticmethod
    def _q24_event_gt_close(
        pred_points_q: torch.Tensor,
        pred_visible_q: torch.Tensor,
        gt_points_b: torch.Tensor,
        gt_valid_b: torch.Tensor,
        width: float,
        visible_thr: int,
        dist_px: float,
        min_overlap: int,
    ) -> bool:
        """Return whether a query is close enough to a visible GT lane to protect or calibrate it."""
        q_visible = pred_visible_q > 0.5
        pred_x = pred_points_q[:, 0] * float(width)
        gt_x = gt_points_b[..., 0] * float(width)

        for gt_i in range(gt_points_b.shape[0]):
            gt_visible = gt_valid_b[gt_i] > 0.5
            if visible_thr > 0 and int(gt_visible.sum().item()) > int(visible_thr):
                continue
            common = q_visible & gt_visible
            if int(common.sum().item()) < int(min_overlap):
                continue
            mean_dx = (pred_x[common] - gt_x[gt_i, common]).abs().mean()
            if float(mean_dx.detach().cpu().item()) <= float(dist_px):
                return True
        return False

    def q24_event_score_calibration_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        gt_lanes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raise existence scores for clean queries that are near true GT5 short lanes."""
        zero = self._zero_like(pred_points)
        if float(self.q24_event_score_calib_gain) <= 0.0:
            return zero, zero, zero
        if pred_valid_logits is None:
            raise ValueError("gcs_q24_event_score_calib requires pred_valid_logits.")

        if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
            pred_logits_2d = pred_logits.squeeze(-1)
        else:
            pred_logits_2d = pred_logits
        if pred_logits_2d.ndim != 2:
            raise ValueError(f"pred_logits must be B x Q, got {tuple(pred_logits.shape)}.")

        device = pred_logits_2d.device
        dtype = pred_points.dtype
        num_queries = int(pred_logits_2d.shape[1])
        score_queries = torch.as_tensor(self.q24_event_score_queries, device=device, dtype=torch.long)
        if score_queries.numel() and int(score_queries.max().item()) >= num_queries:
            raise ValueError(
                "Q24 event score-calibration query range exceeds model query count: "
                f"max configured q{int(score_queries.max().item())}, model has Q={num_queries}."
            )
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=pred_logits_2d.dtype).reshape(-1)
        if gt_lanes.numel() != pred_logits_2d.shape[0]:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={pred_logits_2d.shape[0]}.")

        width = float(self._pixel_scale_for(pred_points).reshape(-1)[0].detach().cpu().item())
        valid_prob = pred_valid_logits.detach().sigmoid()
        points = pred_points.detach()
        losses = []
        prob_sum = pred_logits_2d.new_tensor(0.0)
        pos_count = 0

        for b in range(pred_logits_2d.shape[0]):
            gt_count = int(round(float(gt_lanes[b].detach().item())))
            if gt_count < 5:
                continue
            gt_points_b = gt_points[b].detach().to(device=device, dtype=dtype)
            gt_valid_b = gt_valid[b].detach().to(device=device, dtype=dtype)
            for q in score_queries.tolist():
                if q < 0 or q >= num_queries:
                    continue
                close = self._q24_event_gt_close(
                    points[b, q],
                    valid_prob[b, q] > float(self.q24_event_score_valid_thr),
                    gt_points_b,
                    gt_valid_b,
                    width,
                    int(self.q24_event_score_visible_thr),
                    float(self.q24_event_score_dist_px),
                    int(self.q24_event_score_min_overlap),
                )
                if not close:
                    continue
                target = torch.full_like(pred_logits_2d[b, q], float(self.q24_event_score_target))
                losses.append(F.binary_cross_entropy_with_logits(pred_logits_2d[b, q], target, reduction="mean"))
                prob_sum = prob_sum + pred_logits_2d[b, q].detach().sigmoid()
                pos_count += 1

        score_loss = torch.stack(losses).mean() if losses else zero
        count_tensor = pred_logits_2d.new_tensor(float(pos_count))
        prob_mean = prob_sum / count_tensor.clamp_min(1.0)
        return score_loss, count_tensor, prob_mean

    def q24_event_containment_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        gt_lanes: torch.Tensor,
        event_allowed_masks: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, ...]:
        """Suppress event-mined Q24 high-risk queries while keeping clean GT5 carriers isolated."""
        zero = self._zero_like(pred_points)
        if float(self.q24_event_contain_gain) <= 0.0:
            return zero, zero, zero, zero, zero, zero, zero, zero, zero, zero
        if pred_valid_logits is None and float(self.q24_event_valid_weight) > 0.0:
            raise ValueError(
                "gcs_q24_event_contain with gcs_q24_event_valid_weight > 0 requires pred_valid_logits."
            )
        if self.q24_event_dynamic and pred_valid_logits is None:
            raise ValueError("gcs_q24_event_dynamic requires pred_valid_logits.")
        if event_allowed_masks is None:
            raise ValueError("event_allowed_masks must be provided when gcs_q24_event_contain is enabled.")

        device = pred_logits.device
        dtype = pred_points.dtype
        if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
            pred_logits_2d = pred_logits.squeeze(-1)
        else:
            pred_logits_2d = pred_logits
        if pred_logits_2d.ndim != 2:
            raise ValueError(f"pred_logits must be B x Q, got {tuple(pred_logits.shape)}.")

        num_queries = int(pred_logits_2d.shape[1])
        clean_gt5, risk, gt4_queries, all_event = self._q24_event_query_tensors(device, num_queries)
        dynamic_queries = torch.as_tensor(self.q24_event_dynamic_queries, device=device, dtype=torch.long)
        if dynamic_queries.numel() and int(dynamic_queries.max().item()) >= num_queries:
            raise ValueError(
                "Q24 event dynamic query range exceeds model query count: "
                f"max configured q{int(dynamic_queries.max().item())}, model has Q={num_queries}."
            )
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=pred_logits_2d.dtype).reshape(-1)
        if gt_lanes.numel() != pred_logits_2d.shape[0]:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={pred_logits_2d.shape[0]}.")

        width = float(self._pixel_scale_for(pred_points).reshape(-1)[0].detach().cpu().item())
        valid_prob = pred_valid_logits.detach().sigmoid() if pred_valid_logits is not None else None
        points = pred_points.detach()

        exist_losses = []
        valid_losses = []
        gt3_count = 0
        gt4_count = 0
        gt5_risk_count = 0
        gt5_risk_protected_count = 0
        clean_allowed_count = 0
        dynamic_count = 0
        dynamic_protected_count = 0

        for b in range(pred_logits_2d.shape[0]):
            gt_count = int(round(float(gt_lanes[b].detach().item())))
            selected = torch.zeros((num_queries,), device=device, dtype=torch.bool)
            allowed_positive = torch.zeros((num_queries,), device=device, dtype=torch.bool)
            src_idx, tgt_idx = indices[b]
            if src_idx.numel():
                src_idx = src_idx.to(device=device, dtype=torch.long)
                tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
                allowed_mask_b = event_allowed_masks[b].to(device=device, dtype=torch.bool)
                for q, t in zip(src_idx.tolist(), tgt_idx.tolist()):
                    if 0 <= q < num_queries and 0 <= t < allowed_mask_b.shape[1] and bool(allowed_mask_b[q, t]):
                        allowed_positive[q] = True

            if gt_count <= 3:
                selected[all_event] = True
                gt3_count += int(all_event.numel())
            elif gt_count == 4:
                if clean_gt5.numel():
                    selected[clean_gt5] = True
                if risk.numel():
                    selected[risk] = True
                if gt4_queries.numel():
                    allowed_gt4 = allowed_positive[gt4_queries]
                    suppressed_gt4 = gt4_queries[~allowed_gt4]
                    if suppressed_gt4.numel():
                        selected[suppressed_gt4] = True
                gt4_count += int(selected.sum().item())
            else:
                if clean_gt5.numel():
                    clean_allowed_count += int(allowed_positive[clean_gt5].sum().item())
                if risk.numel() and self.q24_event_suppress_gt5_risk:
                    selected[risk] = True
                    if valid_prob is not None and self.q24_event_gt5_risk_protect:
                        gt_points_b = gt_points[b].detach().to(device=device, dtype=dtype)
                        gt_valid_b = gt_valid[b].detach().to(device=device, dtype=dtype)
                        for q in risk.tolist():
                            if self._q24_event_risk_protected_by_short_gt5(
                                points[b, q],
                                valid_prob[b, q],
                                gt_points_b,
                                gt_valid_b,
                                width,
                            ):
                                selected[q] = False
                                gt5_risk_protected_count += 1
                    gt5_risk_count += int(selected[risk].sum().item())

            if self.q24_event_dynamic and dynamic_queries.numel() and valid_prob is not None:
                gt_points_b = gt_points[b].detach().to(device=device, dtype=dtype)
                gt_valid_b = gt_valid[b].detach().to(device=device, dtype=dtype)
                for q in dynamic_queries.tolist():
                    if q < 0 or q >= num_queries or bool(allowed_positive[q]) or bool(selected[q]):
                        continue
                    q_valid = valid_prob[b, q] > float(self.q24_event_dynamic_valid_thr)
                    visible_len = int(q_valid.sum().item())
                    if visible_len < int(self.q24_event_dynamic_min_valid):
                        continue
                    if visible_len > int(self.q24_event_dynamic_max_visible):
                        continue
                    if float(self.q24_event_dynamic_score_thr) > 0.0:
                        score = float(pred_logits_2d[b, q].detach().sigmoid().cpu().item())
                        if score < float(self.q24_event_dynamic_score_thr):
                            continue
                    if self.q24_event_dynamic_protect:
                        protected = self._q24_event_gt_close(
                            points[b, q],
                            q_valid,
                            gt_points_b,
                            gt_valid_b,
                            width,
                            int(self.q24_event_dynamic_protect_visible_thr),
                            float(self.q24_event_dynamic_protect_dist_px),
                            int(self.q24_event_dynamic_protect_min_overlap),
                        )
                        if protected:
                            dynamic_protected_count += 1
                            continue
                    selected[q] = True
                    dynamic_count += 1

            if not bool(selected.any()):
                continue

            selected_idx = torch.nonzero(selected, as_tuple=False).flatten()
            exist_losses.append(
                F.binary_cross_entropy_with_logits(
                    pred_logits_2d[b, selected_idx],
                    torch.zeros_like(pred_logits_2d[b, selected_idx]),
                    reduction="mean",
                )
            )
            if pred_valid_logits is not None and float(self.q24_event_valid_weight) > 0.0:
                valid_losses.append(
                    F.binary_cross_entropy_with_logits(
                        pred_valid_logits[b, selected_idx],
                        torch.zeros_like(pred_valid_logits[b, selected_idx]),
                        reduction="mean",
                    )
                )

        exist_loss = torch.stack(exist_losses).mean() if exist_losses else zero
        valid_loss = torch.stack(valid_losses).mean() if valid_losses else zero
        loss = exist_loss + float(self.q24_event_valid_weight) * valid_loss
        return (
            loss,
            exist_loss,
            valid_loss,
            pred_logits_2d.new_tensor(float(gt3_count)),
            pred_logits_2d.new_tensor(float(gt4_count)),
            pred_logits_2d.new_tensor(float(gt5_risk_count)),
            pred_logits_2d.new_tensor(float(gt5_risk_protected_count)),
            pred_logits_2d.new_tensor(float(clean_allowed_count)),
            pred_logits_2d.new_tensor(float(dynamic_count)),
            pred_logits_2d.new_tensor(float(dynamic_protected_count)),
        )

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

    def query_quality_loss(
        self,
        preds: dict[str, torch.Tensor],
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train an optional query-level quality/ranking head independently from count estimation."""
        pred_quality_logits = preds.get("pred_quality_logits")
        if pred_quality_logits is None:
            if float(self.query_quality_gain) > 0.0:
                raise ValueError(
                    "gcs_query_quality requires preds['pred_quality_logits']; use a query_quality_head model YAML "
                    "or set --gcs-query-quality 0.0."
                )
            zero = self._zero_like(pred_points)
            return zero, zero, zero

        if pred_quality_logits.ndim == 3 and pred_quality_logits.shape[-1] == 1:
            pred_quality_logits = pred_quality_logits.squeeze(-1)
        if pred_quality_logits.ndim != 2 or tuple(pred_quality_logits.shape) != tuple(pred_points.shape[:2]):
            raise ValueError(
                "pred_quality_logits must have shape B x Q matching pred_points, "
                f"got {tuple(pred_quality_logits.shape)} vs {tuple(pred_points.shape[:2])}."
            )

        target = torch.zeros_like(pred_quality_logits)
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._pixel_scale_for(pred_points)
        pos_targets: list[torch.Tensor] = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
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
            quality = quality.to(dtype=target.dtype)
            target[b, src_idx] = quality
            pos_targets.append(quality.detach())

        pos_weight = pred_quality_logits.new_tensor(self.exist_pos_weight)
        loss = F.binary_cross_entropy_with_logits(pred_quality_logits, target, pos_weight=pos_weight, reduction="mean")
        if pos_targets:
            pos_cat = torch.cat([x.reshape(-1) for x in pos_targets])
            pos_mean = pos_cat.mean().to(device=pred_points.device, dtype=pred_points.dtype)
            pos_count = pred_points.new_tensor(float(pos_cat.numel()))
        else:
            pos_mean = self._zero_like(pred_points)
            pos_count = self._zero_like(pred_points)
        return loss, pos_mean, pos_count

    def query_extent_loss(
        self,
        preds: dict[str, torch.Tensor],
        pred_points: torch.Tensor,
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train optional query-mode first/last visible anchor classifiers."""
        pred_start_logits = preds.get("pred_start_logits")
        pred_end_logits = preds.get("pred_end_logits")
        if pred_start_logits is None or pred_end_logits is None:
            if float(self.query_extent_gain) > 0.0:
                raise ValueError(
                    "gcs_query_extent requires preds['pred_start_logits'] and preds['pred_end_logits']; "
                    "use a query_extent_head model YAML or set --gcs-query-extent 0.0."
                )
            zero = self._zero_like(pred_points)
            return zero, zero, zero, zero, zero

        expected_shape = tuple(pred_points.shape[:3])
        if tuple(pred_start_logits.shape) != expected_shape:
            raise ValueError(
                "pred_start_logits must have shape B x Q x K matching pred_points, "
                f"got {tuple(pred_start_logits.shape)} vs {expected_shape}."
            )
        if tuple(pred_end_logits.shape) != expected_shape:
            raise ValueError(
                "pred_end_logits must have shape B x Q x K matching pred_points, "
                f"got {tuple(pred_end_logits.shape)} vs {expected_shape}."
            )

        device = pred_points.device
        dtype = pred_points.dtype
        loss_sum = pred_points.new_zeros(())
        weight_sum = pred_points.new_zeros(())
        start_correct = pred_points.new_zeros(())
        end_correct = pred_points.new_zeros(())
        interval_iou_sum = pred_points.new_zeros(())
        matched_count = 0
        short_count = 0

        gt_lanes_t = None
        if gt_lanes is not None:
            gt_lanes_t = torch.as_tensor(gt_lanes, device=device, dtype=dtype).reshape(-1)
            if gt_lanes_t.numel() != pred_points.shape[0]:
                raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes_t.numel()} vs B={pred_points.shape[0]}.")

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            target_valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]
            visible = target_valid > 0.5
            has_visible = visible.any(dim=1)
            if not bool(has_visible.any()):
                continue

            src_idx = src_idx[has_visible]
            visible = visible[has_visible]
            visible_float = visible.to(dtype=dtype)
            k = int(visible.shape[1])
            start_targets = visible_float.argmax(dim=1).long()
            end_targets = (k - 1 - torch.flip(visible_float, dims=[1]).argmax(dim=1)).long()

            start_logits = pred_start_logits[b, src_idx]
            end_logits = pred_end_logits[b, src_idx]
            start_loss = F.cross_entropy(start_logits, start_targets, reduction="none")
            end_loss = F.cross_entropy(end_logits, end_targets, reduction="none")

            lane_weights = torch.ones_like(start_loss, dtype=dtype)
            visible_counts = visible_float.sum(dim=1)
            gt_count = int(round(float(gt_lanes_t[b].detach().item()))) if gt_lanes_t is not None else 0
            short_mask = (visible_counts <= float(self.query_extent_short_visible_thr)) & (
                gt_count >= int(self.query_extent_gt_min_lanes)
            )
            if bool(short_mask.any()):
                lane_weights[short_mask] = lane_weights[short_mask] + float(self.query_extent_short_weight)
                short_count += int(short_mask.sum().item())

            loss_sum = loss_sum + ((start_loss + end_loss) * lane_weights).sum()
            weight_sum = weight_sum + lane_weights.sum()

            pred_start = start_logits.detach().argmax(dim=1)
            pred_end = end_logits.detach().argmax(dim=1)
            pred_lo = torch.minimum(pred_start, pred_end)
            pred_hi = torch.maximum(pred_start, pred_end)
            target_lo = torch.minimum(start_targets, end_targets)
            target_hi = torch.maximum(start_targets, end_targets)
            inter = (torch.minimum(pred_hi, target_hi) - torch.maximum(pred_lo, target_lo) + 1).clamp_min(0)
            union = (torch.maximum(pred_hi, target_hi) - torch.minimum(pred_lo, target_lo) + 1).clamp_min(1)
            interval_iou = inter.to(dtype=dtype) / union.to(dtype=dtype)

            start_correct = start_correct + (pred_start == start_targets).to(dtype=dtype).sum()
            end_correct = end_correct + (pred_end == end_targets).to(dtype=dtype).sum()
            interval_iou_sum = interval_iou_sum + interval_iou.sum()
            matched_count += int(start_targets.numel())

        if matched_count == 0:
            zero = self._zero_like(pred_points)
            return zero, zero, zero, zero, zero

        denom = weight_sum.clamp_min(1.0)
        count_t = pred_points.new_tensor(float(matched_count))
        return (
            loss_sum / denom,
            start_correct / count_t,
            end_correct / count_t,
            interval_iou_sum / count_t,
            pred_points.new_tensor(float(short_count)),
        )

    def short_local_refine_loss(
        self,
        preds: dict[str, torch.Tensor],
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Supervise optional auxiliary local x refinement on matched short GT4/GT5 lanes."""
        pred_short_refined_points = preds.get("pred_short_refined_points")
        if pred_short_refined_points is None:
            if float(self.short_local_refine_gain) > 0.0:
                raise ValueError(
                    "gcs_short_local_refine requires preds['pred_short_refined_points']; "
                    "use a query_short_local_refine_head model YAML or set --gcs-short-local-refine 0.0."
                )
            zero = self._zero_like(pred_points)
            return zero, zero, zero, zero, zero, zero, zero, zero
        if tuple(pred_short_refined_points.shape) != tuple(pred_points.shape):
            raise ValueError(
                "pred_short_refined_points must have shape B x Q x K x 2 matching pred_points, "
                f"got {tuple(pred_short_refined_points.shape)} vs {tuple(pred_points.shape)}."
            )
        pred_coarse_points = preds.get("pred_coarse_points")
        if isinstance(pred_coarse_points, torch.Tensor) and tuple(pred_coarse_points.shape) != tuple(pred_points.shape):
            raise ValueError(
                "pred_coarse_points must have shape B x Q x K x 2 matching pred_points when present, "
                f"got {tuple(pred_coarse_points.shape)} vs {tuple(pred_points.shape)}."
            )

        device, dtype = pred_points.device, pred_points.dtype
        gt_lanes_t = None
        if gt_lanes is not None:
            gt_lanes_t = torch.as_tensor(gt_lanes, device=device, dtype=dtype).reshape(-1)
            if gt_lanes_t.numel() != pred_points.shape[0]:
                raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes_t.numel()} vs B={pred_points.shape[0]}.")

        width = self._pixel_scale_for(pred_points).reshape(-1)[0].to(device=device, dtype=dtype)
        width_float = max(float(width.detach().item()), 1.0)
        beta_norm = max(float(self.short_local_refine_beta_px) / width_float, 1e-12)
        selected_losses: list[torch.Tensor] = []
        selected_weights: list[torch.Tensor] = []
        coarse_apes: list[torch.Tensor] = []
        refined_apes: list[torch.Tensor] = []
        identity_counts: list[torch.Tensor] = []
        pull_counts: list[torch.Tensor] = []
        identity_guard = bool(getattr(self, "short_local_refine_identity_guard", False))
        identity_thr = float(getattr(self, "short_local_refine_identity_thr_px", 20.0))
        nearmiss_thr = float(getattr(self, "short_local_refine_nearmiss_thr_px", 80.0))
        identity_weight = float(getattr(self, "short_local_refine_identity_weight", 1.0))
        nearmiss_weight = float(getattr(self, "short_local_refine_nearmiss_weight", 1.0))

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            target_all = gt_points[b].to(device=device, dtype=dtype)
            valid_all = gt_valid[b].to(device=device, dtype=dtype)
            target = target_all[tgt_idx]
            valid = valid_all[tgt_idx] > 0.5
            visible_counts = valid.to(dtype=dtype).sum(dim=1)
            if gt_lanes_t is None:
                gt_count = 0
            else:
                gt_count = int(round(float(gt_lanes_t[b].detach().item())))
            short_mask = (visible_counts <= float(self.short_local_refine_visible_thr)) & (
                gt_count >= int(self.short_local_refine_gt_min_lanes)
            )
            if not bool(short_mask.any()):
                continue

            src_short = src_idx[short_mask]
            target_short = target[short_mask]
            valid_short = valid[short_mask]
            refined_short = pred_short_refined_points[b, src_short]
            coarse_short = pred_coarse_points[b, src_short] if isinstance(pred_coarse_points, torch.Tensor) else pred_points[b, src_short]
            mask = valid_short.to(dtype=dtype)
            valid_den = mask.sum(dim=1).clamp_min(1.0)
            coarse_ape = ((coarse_short.detach()[..., 0] - target_short[..., 0]).abs() * width * mask).sum(dim=1) / valid_den

            if identity_guard:
                identity_mask = coarse_ape <= identity_thr
                pull_mask = (coarse_ape > identity_thr) & (coarse_ape <= nearmiss_thr)
                eligible_mask = identity_mask | pull_mask
                if not bool(eligible_mask.any()):
                    continue
                target_x = torch.where(identity_mask[:, None], coarse_short.detach()[..., 0], target_short[..., 0])
                lane_weight = torch.where(
                    identity_mask,
                    coarse_ape.new_tensor(identity_weight),
                    coarse_ape.new_tensor(nearmiss_weight),
                )
            else:
                identity_mask = torch.zeros_like(coarse_ape, dtype=torch.bool)
                pull_mask = torch.ones_like(coarse_ape, dtype=torch.bool)
                eligible_mask = pull_mask
                target_x = target_short[..., 0]
                lane_weight = coarse_ape.new_ones(coarse_ape.shape)

            refined_short = refined_short[eligible_mask]
            coarse_ape = coarse_ape[eligible_mask]
            refined_target_x = target_short[..., 0][eligible_mask]
            target_x = target_x[eligible_mask]
            mask = mask[eligible_mask]
            valid_den = valid_den[eligible_mask]
            lane_weight = lane_weight[eligible_mask]

            per_anchor = F.smooth_l1_loss(
                refined_short[..., 0],
                target_x,
                beta=beta_norm,
                reduction="none",
            )
            selected_losses.append((per_anchor * mask).sum(dim=1) / valid_den)
            selected_weights.append(lane_weight)

            refined_ape = ((refined_short.detach()[..., 0] - refined_target_x).abs() * width * mask).sum(dim=1) / valid_den
            coarse_apes.append(coarse_ape)
            refined_apes.append(refined_ape)
            identity_counts.append(identity_mask[eligible_mask].to(dtype=dtype).sum().reshape(1))
            pull_counts.append(pull_mask[eligible_mask].to(dtype=dtype).sum().reshape(1))

        if not selected_losses:
            zero = self._zero_like(pred_short_refined_points)
            return zero, zero, zero, zero, zero, zero, zero, zero

        loss_cat = torch.cat([x.reshape(-1) for x in selected_losses])
        weight_cat = torch.cat([x.reshape(-1) for x in selected_weights])
        loss = (loss_cat * weight_cat).sum() / weight_cat.sum().clamp_min(1.0)
        coarse_cat = torch.cat([x.reshape(-1) for x in coarse_apes])
        refined_cat = torch.cat([x.reshape(-1) for x in refined_apes])
        count_t = pred_points.new_tensor(float(refined_cat.numel()))
        gain20 = ((coarse_cat > 20.0) & (refined_cat <= 20.0)).to(dtype=dtype).sum() / count_t.clamp_min(1.0)
        loss20 = ((coarse_cat <= 20.0) & (refined_cat > 20.0)).to(dtype=dtype).sum() / count_t.clamp_min(1.0)
        identity_count_t = torch.cat(identity_counts).sum() if identity_counts else self._zero_like(pred_points)
        pull_count_t = torch.cat(pull_counts).sum() if pull_counts else self._zero_like(pred_points)
        return loss, count_t, coarse_cat.mean(), refined_cat.mean(), gain20, loss20, identity_count_t, pull_count_t

    def short_candidate_loss(
        self,
        preds: dict[str, torch.Tensor],
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train feature-conditioned scores for fixed lateral candidates around matched short lanes."""
        candidate_points = preds.get("pred_short_candidate_points")
        candidate_logits = preds.get("pred_short_candidate_logits")
        if candidate_points is None or candidate_logits is None:
            if float(self.short_candidate_gain) > 0.0:
                raise ValueError(
                    "gcs_short_candidate requires preds['pred_short_candidate_points'] and "
                    "preds['pred_short_candidate_logits']; use a query_short_candidate_head model YAML "
                    "or set --gcs-short-candidate 0.0."
                )
            zero = self._zero_like(pred_points)
            return zero, zero, zero, zero, zero, zero, zero

        bsz, num_queries, num_points, _ = pred_points.shape
        if candidate_points.ndim != 5 or candidate_points.shape[:2] != (bsz, num_queries) or candidate_points.shape[-1] != 2:
            raise ValueError(
                "pred_short_candidate_points must have shape B x Q x H x K x 2, got "
                f"{tuple(candidate_points.shape)}."
            )
        candidate_count = int(candidate_points.shape[2])
        if candidate_points.shape[3] != num_points:
            raise ValueError(
                "pred_short_candidate_points K dimension must match pred_points, got "
                f"{candidate_points.shape[3]} vs {num_points}."
            )
        if candidate_logits.shape != (bsz, num_queries, candidate_count):
            raise ValueError(
                "pred_short_candidate_logits must have shape B x Q x H matching candidate points, got "
                f"{tuple(candidate_logits.shape)} vs {(bsz, num_queries, candidate_count)}."
            )

        # Freeze-base candidate probes still need a differentiable zero on
        # batches without an eligible short GT4/GT5 lane.
        candidate_zero = candidate_logits.sum() * 0.0
        device, dtype = pred_points.device, pred_points.dtype
        width = self._pixel_scale_for(pred_points).reshape(-1)[0].to(device=device, dtype=dtype)
        width_float = max(float(width.detach().item()), 1.0)
        beta_px = float(self.short_candidate_beta_px)
        temperature = float(self.short_candidate_score_temperature)
        offsets_px = pred_points.new_tensor([0.0])
        if candidate_count > 1:
            step_px = float(self.short_candidate_step_px)
            half = (candidate_count - 1) // 2
            offsets = [0.0]
            for i in range(1, half + 1):
                offsets.extend((-i * step_px, i * step_px))
            offsets_px = pred_points.new_tensor(offsets)
        if offsets_px.numel() != candidate_count:
            offsets_px = pred_points.new_zeros((candidate_count,))

        selected_score_losses: list[torch.Tensor] = []
        selected_geometry_losses: list[torch.Tensor] = []
        best_offsets: list[torch.Tensor] = []
        raw_hits: list[torch.Tensor] = []
        best_hits: list[torch.Tensor] = []

        gt_lanes_t = None
        if gt_lanes is not None:
            gt_lanes_t = torch.as_tensor(gt_lanes, device=device, dtype=dtype).reshape(-1)
            if gt_lanes_t.numel() != bsz:
                raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes_t.numel()} vs B={bsz}.")

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            target_all = gt_points[b].to(device=device, dtype=dtype)
            valid_all = gt_valid[b].to(device=device, dtype=dtype)
            target = target_all[tgt_idx]
            valid = valid_all[tgt_idx] > 0.5
            visible_counts = valid.to(dtype=dtype).sum(dim=1)
            gt_count = int(round(float(gt_lanes_t[b].detach().item()))) if gt_lanes_t is not None else 0
            short_mask = (visible_counts <= float(self.short_candidate_visible_thr)) & (
                gt_count >= int(self.short_candidate_gt_min_lanes)
            )
            if not bool(short_mask.any()):
                continue

            src_short = src_idx[short_mask]
            target_short = target[short_mask]
            valid_short = valid[short_mask].to(dtype=dtype)
            candidate_short = candidate_points[b, src_short]
            candidate_x = candidate_short[..., 0]
            target_x = target_short[..., 0].unsqueeze(1)
            mask = valid_short.unsqueeze(1)
            valid_den = valid_short.sum(dim=1, keepdim=True).clamp_min(1.0)
            ape_px = ((candidate_x - target_x).abs() * width * mask).sum(dim=-1) / valid_den

            target_prob = F.softmax(-ape_px / beta_px, dim=1)
            logits = candidate_logits[b, src_short]
            predicted_prob = F.softmax(logits / temperature, dim=1)
            score_loss = -(target_prob * F.log_softmax(logits / temperature, dim=1)).sum(dim=1)
            geometry_loss = (predicted_prob * ape_px).sum(dim=1) / width_float
            selected_score_losses.append(score_loss)
            selected_geometry_losses.append(geometry_loss)

            raw_best_ape, raw_best_idx = ape_px.detach().min(dim=1)
            predicted_idx = logits.detach().argmax(dim=1)
            predicted_ape = ape_px.detach().gather(1, predicted_idx[:, None]).squeeze(1)
            best_offsets.append(offsets_px[raw_best_idx].abs().reshape(-1))
            raw_hits.append((raw_best_ape <= 20.0).to(dtype=dtype).reshape(-1))
            best_hits.append((predicted_ape <= 20.0).to(dtype=dtype).reshape(-1))

        if not selected_score_losses:
            return (
                candidate_zero,
                candidate_zero,
                candidate_zero,
                candidate_zero,
                candidate_zero,
                candidate_zero,
                candidate_zero,
            )

        score_cat = torch.cat([value.reshape(-1) for value in selected_score_losses])
        geometry_cat = torch.cat([value.reshape(-1) for value in selected_geometry_losses])
        offset_cat = torch.cat(best_offsets)
        raw_hit_cat = torch.cat(raw_hits)
        best_hit_cat = torch.cat(best_hits)
        count_t = pred_points.new_tensor(float(score_cat.numel()))
        score_loss = score_cat.mean()
        geometry_loss = geometry_cat.mean()
        return (
            score_loss + geometry_loss,
            count_t,
            score_loss,
            geometry_loss,
            offset_cat.mean(),
            raw_hit_cat.mean(),
            best_hit_cat.mean(),
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

    def _boundary_pseudo_protected_by_short_gt5(
        self,
        pred_points_q: torch.Tensor,
        pred_visible_q: torch.Tensor,
        gt_points_b: torch.Tensor,
        gt_valid_b: torch.Tensor,
        width: float,
        *,
        query_idx: int,
    ) -> bool:
        """Return whether a GT5 boundary-pseudo candidate is close to a true short GT5 lane."""
        if not self.boundary_pseudo_gt5_safe:
            return False
        if self.boundary_pseudo_protect_queries and int(query_idx) not in set(self.boundary_pseudo_protect_queries):
            return False
        short_thr = int(self.boundary_pseudo_protect_short_visible_thr)
        if short_thr <= 0:
            return False

        q_visible = pred_visible_q > 0.5
        pred_x = pred_points_q[:, 0] * float(width)
        gt_x = gt_points_b[..., 0] * float(width)
        min_overlap = int(self.boundary_pseudo_protect_min_overlap)
        protect_dist = float(self.boundary_pseudo_protect_dist_px)

        for gt_i in range(gt_points_b.shape[0]):
            gt_visible = gt_valid_b[gt_i] > 0.5
            if int(gt_visible.sum().item()) > short_thr:
                continue
            common = q_visible & gt_visible
            if int(common.sum().item()) < min_overlap:
                continue
            mean_dx = (pred_x[common] - gt_x[gt_i, common]).abs().mean()
            if float(mean_dx.detach().cpu().item()) <= protect_dist:
                return True
        return False

    def boundary_pseudo_neg_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extra negative BCE for short far-side pseudo lanes on unmatched queries."""
        if float(self.boundary_pseudo_neg_gain) <= 0.0:
            zero = self._zero_like(pred_points)
            return zero, zero, zero, zero, zero
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
        candidate_count = 0
        protected_count = 0

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
                candidate_count += 1

                if self._boundary_pseudo_protected_by_short_gt5(
                    points[b, q],
                    q_valid,
                    gt_points_b,
                    gt_valid_b,
                    width,
                    query_idx=int(q),
                ):
                    protected_count += 1
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
            return (
                zero,
                zero,
                zero,
                pred_points.new_tensor(float(candidate_count)),
                pred_points.new_tensor(float(protected_count)),
            )

        return (
            torch.stack(losses).mean(),
            torch.stack(pseudo_counts).sum(),
            torch.stack(pseudo_score_means).mean(),
            pred_points.new_tensor(float(candidate_count)),
            pred_points.new_tensor(float(protected_count)),
        )

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

        gt_lanes = self.target_lane_count(pred_logits, batch, gt_valid)
        matcher_points = pred_points
        if isinstance(preds.get("pred_coarse_points"), torch.Tensor):
            pred_coarse_points = preds["pred_coarse_points"]
            if tuple(pred_coarse_points.shape) != tuple(pred_points.shape):
                raise ValueError(
                    "pred_coarse_points must have shape B x Q x K x 2 matching pred_points, "
                    f"got {tuple(pred_coarse_points.shape)} vs {tuple(pred_points.shape)}."
                )

        role_allowed_masks = self._role_allowed_masks(matcher_points, gt_valid, gt_lanes)
        event_allowed_masks = self._q24_event_allowed_masks(matcher_points, gt_valid, gt_lanes)
        matcher_allowed_masks = self._merge_allowed_masks(
            role_allowed_masks if self.role_contain_matcher else None,
            event_allowed_masks if self.q24_event_matcher else None,
        )
        indices = self.matcher(
            matcher_points,
            pred_logits,
            gt_points,
            gt_valid,
            allowed_masks=matcher_allowed_masks,
        )
        exist_loss = self.exist_loss(pred_logits, pred_points, pred_valid_logits, gt_points, gt_valid, indices)
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
        (
            boundary_pseudo_neg_loss,
            boundary_pseudo_count,
            boundary_pseudo_score_mean,
            boundary_pseudo_candidate_count,
            boundary_pseudo_protected_count,
        ) = self.boundary_pseudo_neg_loss(
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
        query_quality_loss, query_quality_pos_mean, query_quality_pos_count = self.query_quality_loss(
            preds,
            pred_points,
            pred_valid_logits,
            gt_points,
            gt_valid,
            indices,
        )
        (
            query_extent_loss,
            query_extent_start_acc,
            query_extent_end_acc,
            query_extent_iou,
            query_extent_short_count,
        ) = self.query_extent_loss(
            preds,
            pred_points,
            gt_valid,
            indices,
            gt_lanes=gt_lanes,
        )
        (
            short_local_refine_loss,
            short_local_refine_count,
            short_local_refine_coarse_ape,
            short_local_refine_refined_ape,
            short_local_refine_gain20,
            short_local_refine_loss20,
            short_local_refine_identity_count,
            short_local_refine_pull_count,
        ) = self.short_local_refine_loss(
            preds,
            pred_points,
            gt_points,
            gt_valid,
            indices,
            gt_lanes=gt_lanes,
        )
        (
            short_candidate_loss,
            short_candidate_count,
            short_candidate_score_loss,
            short_candidate_geometry_loss,
            short_candidate_best_offset_px,
            short_candidate_raw_hit20,
            short_candidate_best_hit20,
        ) = self.short_candidate_loss(
            preds,
            pred_points,
            gt_points,
            gt_valid,
            indices,
            gt_lanes=gt_lanes,
        )
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
        (
            role_contain_loss,
            role_contain_exist_loss,
            role_contain_valid_loss,
            role_contain_count,
            role_contain_gt3_count,
            role_contain_gt4_gt5bank_count,
            role_contain_gt4_extra_count,
            role_contain_allowed_count,
        ) = self.role_containment_loss(
            pred_logits,
            pred_valid_logits,
            indices,
            gt_lanes,
            role_allowed_masks,
        )
        (
            q24_event_contain_loss,
            q24_event_exist_loss,
            q24_event_valid_loss,
            q24_event_gt3_count,
            q24_event_gt4_count,
            q24_event_gt5_risk_count,
            q24_event_gt5_risk_protected_count,
            q24_event_clean_allowed_count,
            q24_event_dynamic_count,
            q24_event_dynamic_protected_count,
        ) = self.q24_event_containment_loss(
            pred_points,
            pred_logits,
            pred_valid_logits,
            indices,
            gt_points,
            gt_valid,
            gt_lanes,
            event_allowed_masks,
        )
        (
            q24_event_score_loss,
            q24_event_score_pos_count,
            q24_event_score_prob_mean,
        ) = self.q24_event_score_calibration_loss(
            pred_points,
            pred_logits,
            pred_valid_logits,
            gt_points,
            gt_valid,
            gt_lanes,
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
        if self.boundary_pseudo_neg_gain != 0.0:
            total = total + self.boundary_pseudo_neg_gain * boundary_pseudo_neg_loss
        if self.query_count_ce_gain != 0.0:
            total = total + self.query_count_ce_gain * query_count_ce_loss
        if self.query_quality_gain != 0.0:
            total = total + self.query_quality_gain * query_quality_loss
        if self.query_extent_gain != 0.0:
            total = total + self.query_extent_gain * query_extent_loss
        if self.short_local_refine_gain != 0.0:
            total = total + self.short_local_refine_gain * short_local_refine_loss
        if self.short_candidate_gain != 0.0:
            total = total + self.short_candidate_gain * short_candidate_loss
        if self.spurious_neg_gain != 0.0:
            total = total + self.spurious_neg_gain * self.spurious_neg_weight * spurious_neg_loss
        if self.role_contain_gain != 0.0:
            total = total + self.role_contain_gain * role_contain_loss
        if self.q24_event_contain_gain != 0.0:
            total = total + self.q24_event_contain_gain * q24_event_contain_loss
        if self.q24_event_score_calib_gain != 0.0:
            total = total + self.q24_event_score_calib_gain * q24_event_score_loss
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
                boundary_pseudo_candidate_count.detach(),
                boundary_pseudo_protected_count.detach(),
                query_count_ce_loss.detach(),
                query_count_acc.detach(),
                query_count_pred_mean.detach(),
                query_quality_loss.detach(),
                query_quality_pos_mean.detach(),
                query_quality_pos_count.detach(),
                query_extent_loss.detach(),
                query_extent_start_acc.detach(),
                query_extent_end_acc.detach(),
                query_extent_iou.detach(),
                query_extent_short_count.detach(),
                short_local_refine_loss.detach(),
                short_local_refine_count.detach(),
                short_local_refine_coarse_ape.detach(),
                short_local_refine_refined_ape.detach(),
                short_local_refine_gain20.detach(),
                short_local_refine_loss20.detach(),
                short_local_refine_identity_count.detach(),
                short_local_refine_pull_count.detach(),
                short_candidate_loss.detach(),
                short_candidate_count.detach(),
                short_candidate_score_loss.detach(),
                short_candidate_geometry_loss.detach(),
                short_candidate_best_offset_px.detach(),
                short_candidate_raw_hit20.detach(),
                short_candidate_best_hit20.detach(),
                role_contain_loss.detach(),
                role_contain_exist_loss.detach(),
                role_contain_valid_loss.detach(),
                role_contain_count.detach(),
                role_contain_gt3_count.detach(),
                role_contain_gt4_gt5bank_count.detach(),
                role_contain_gt4_extra_count.detach(),
                role_contain_allowed_count.detach(),
                q24_event_contain_loss.detach(),
                q24_event_exist_loss.detach(),
                q24_event_valid_loss.detach(),
                q24_event_gt3_count.detach(),
                q24_event_gt4_count.detach(),
                q24_event_gt5_risk_count.detach(),
                q24_event_gt5_risk_protected_count.detach(),
                q24_event_clean_allowed_count.detach(),
                q24_event_dynamic_count.detach(),
                q24_event_dynamic_protected_count.detach(),
                q24_event_score_loss.detach(),
                q24_event_score_pos_count.detach(),
                q24_event_score_prob_mean.detach(),
            )
        )
        return total, loss_items
