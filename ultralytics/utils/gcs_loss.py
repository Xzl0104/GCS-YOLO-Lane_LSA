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


class GCSLoss(nn.Module):
    """Hungarian-matched loss for query-based structured lane predictions."""

    loss_names = (
        "exist_loss",
        "point_loss",
        "point_valid_loss",
        "smooth_loss",
        "curve_loss",
        "line_iou_loss",
        "line_iou_valid_preserve_loss",
        "line_iou_valid_preserve_count",
        "line_iou_valid_preserve_anchor_count",
        "line_iou_exist_survival_loss",
        "line_iou_exist_survival_count",
        "mask_loss",
        "edge_loss",
        "count_loss",
        "count_under5_loss",
        "count_boundary_loss",
        "spurious_neg_loss",
        "spurious_negative_count",
        "query_survival_rank_loss",
        "query_survival_rank_pair_count",
        "query_survival_rank_margin_mean",
        "query_valid_survival_loss",
        "query_valid_survival_count",
        "query_valid_survival_anchor_count",
        "query_valid_survival_dice_loss",
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
        "query_count_ce_loss",
        "query_count_acc",
        "query_count_pred_mean",
        "dense_instance_loss",
        "dense_centerline_loss",
        "dense_endpoint_loss",
        "dense_endpoint_peak_loss",
        "dense_endpoint_offset_loss",
        "dense_embed_pull_loss",
        "dense_embed_push_loss",
        "dense_centerline_pos",
        "dense_candidate_loss",
        "dense_candidate_quality_loss",
        "dense_candidate_replace_loss",
        "dense_candidate_quality_pos",
        "dense_candidate_replace_pos",
        "residual_proposal_loss",
        "residual_point_loss",
        "residual_row_loss",
        "residual_valid_loss",
        "residual_interval_loss",
        "residual_consistency_loss",
        "residual_positive_span_loss",
        "residual_identity_pull_loss",
        "residual_identity_push_loss",
        "residual_quality_loss",
        "residual_quality_rank_loss",
        "residual_quality_pos",
        "residual_identity_pair_count",
        "residual_exist_loss",
        "residual_target_count",
        "residual_match_count",
        "residual_full_hit20",
        "residual_replace_loss",
        "residual_replace_pos",
        "residual_replace_rank_loss",
        "residual_replace_listwise_loss",
        "residual_replace_action_pos",
        "residual_replace_action_acc",
    )
    lane_instance_loss_names = (
        "lane_instance_set_loss",
        "lane_instance_visibility_loss",
        "lane_instance_start_loss",
        "lane_instance_end_loss",
        "lane_instance_order_loss",
        "lane_instance_contiguity_loss",
        "lane_instance_positive_span_loss",
        "lane_instance_empty_loss",
        "lane_instance_geometry_quality_loss",
        "lane_instance_survival_loss",
        "lane_instance_duplicate_loss",
        "lane_instance_novelty_loss",
        "lane_instance_topology_loss",
        "lane_instance_identity_relation_loss",
        "lane_instance_set_noop_loss",
        "lane_instance_count_calibration_loss",
        "lane_instance_match_count",
    )
    lane_instance_progress_loss_names = (
        "lis_set",
        "lis_vis",
        "lis_start",
        "lis_end",
        "lis_order",
        "lis_contig",
        "lis_span",
        "lis_empty",
        "lis_quality",
        "lis_survive",
        "lis_dup",
        "lis_novel",
        "lis_topo",
        "lis_ident",
        "lis_noop",
        "lis_match",
    )

    @classmethod
    def lane_instance_enabled(cls, args=None) -> bool:
        """Return whether the default-off lane-instance loss vector is active."""
        value = args.get("gcs_lane_instance_set", 0.0) if isinstance(args, dict) else getattr(args, "gcs_lane_instance_set", 0.0)
        return float(value or 0.0) > 0.0

    @classmethod
    def active_loss_names(cls, args=None) -> tuple[str, ...]:
        """Return the query loss names emitted for the supplied runtime gains."""
        return cls.loss_names + (cls.lane_instance_loss_names if cls.lane_instance_enabled(args) else ())

    def __init__(
        self,
        model=None,
        lambda_exist: float | None = None,
        lambda_point: float | None = None,
        lambda_point_valid: float | None = None,
        lambda_smooth: float | None = None,
        lambda_curve: float | None = None,
        lambda_line_iou: float | None = None,
        line_iou_half_width_px: float | None = None,
        line_iou_short_min_points: int | None = None,
        lambda_line_iou_valid_preserve: float | None = None,
        lambda_line_iou_exist_survival: float | None = None,
        line_iou_valid_visible_thr: int | None = None,
        line_iou_valid_min_visible: int | None = None,
        line_iou_valid_max_ape_px: float | None = None,
        line_iou_valid_gt4_weight: float | None = None,
        line_iou_valid_gt5_weight: float | None = None,
        lambda_mask: float | None = None,
        lambda_edge: float | None = None,
        lambda_count: float | None = None,
        lambda_count_under5: float | None = None,
        lambda_count_boundary: float | None = None,
        lambda_spurious_neg: float | None = None,
        lambda_query_survival_rank: float | None = None,
        query_survival_rank_margin: float | None = None,
        query_survival_rank_max_ape_px: float | None = None,
        query_survival_rank_min_lanes: int | None = None,
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
        self.line_iou_gain = float(
            lambda_line_iou if lambda_line_iou is not None else self._arg(args, "gcs_line_iou", 0.0)
        )
        self.line_iou_half_width_px = float(
            line_iou_half_width_px
            if line_iou_half_width_px is not None
            else self._arg(args, "gcs_line_iou_half_width_px", 15.0)
        )
        self.line_iou_short_min_points = int(
            line_iou_short_min_points
            if line_iou_short_min_points is not None
            else self._arg(args, "gcs_line_iou_short_min_points", 3)
        )
        if self.line_iou_half_width_px <= 0.0:
            raise ValueError("gcs_line_iou_half_width_px must be positive.")
        if self.line_iou_short_min_points < 1:
            raise ValueError("gcs_line_iou_short_min_points must be at least 1.")
        self.line_iou_valid_preserve_gain = float(
            lambda_line_iou_valid_preserve
            if lambda_line_iou_valid_preserve is not None
            else self._arg(args, "gcs_line_iou_valid_preserve", 0.0)
        )
        self.line_iou_exist_survival_gain = float(
            lambda_line_iou_exist_survival
            if lambda_line_iou_exist_survival is not None
            else self._arg(args, "gcs_line_iou_exist_survival", 0.0)
        )
        self.line_iou_valid_visible_thr = int(
            line_iou_valid_visible_thr
            if line_iou_valid_visible_thr is not None
            else self._arg(args, "gcs_line_iou_valid_visible_thr", 10)
        )
        self.line_iou_valid_min_visible = int(
            line_iou_valid_min_visible
            if line_iou_valid_min_visible is not None
            else self._arg(args, "gcs_line_iou_valid_min_visible", 3)
        )
        self.line_iou_valid_max_ape_px = float(
            line_iou_valid_max_ape_px
            if line_iou_valid_max_ape_px is not None
            else self._arg(args, "gcs_line_iou_valid_max_ape_px", 30.0)
        )
        self.line_iou_valid_gt4_weight = float(
            line_iou_valid_gt4_weight
            if line_iou_valid_gt4_weight is not None
            else self._arg(args, "gcs_line_iou_valid_gt4_weight", 1.0)
        )
        self.line_iou_valid_gt5_weight = float(
            line_iou_valid_gt5_weight
            if line_iou_valid_gt5_weight is not None
            else self._arg(args, "gcs_line_iou_valid_gt5_weight", 1.0)
        )
        self.dense_instance_gain = float(self._arg(args, "gcs_dense_instance", 0.0))
        self.residual_proposal_gain = float(self._arg(args, "gcs_residual_proposal", 0.0))
        self.lane_instance_set_gain = float(self._arg(args, "gcs_lane_instance_set", 0.0))
        self.lane_instance_visibility_weight = float(self._arg(args, "gcs_lane_instance_visibility_weight", 1.0))
        self.lane_instance_endpoint_weight = float(self._arg(args, "gcs_lane_instance_endpoint_weight", 1.0))
        self.lane_instance_order_weight = float(self._arg(args, "gcs_lane_instance_order_weight", 0.25))
        self.lane_instance_contiguity_weight = float(self._arg(args, "gcs_lane_instance_contiguity_weight", 1.0))
        self.lane_instance_positive_span_weight = float(
            self._arg(args, "gcs_lane_instance_positive_span_weight", 0.25)
        )
        self.lane_instance_empty_weight = float(self._arg(args, "gcs_lane_instance_empty_weight", 1.0))
        self.lane_instance_geometry_quality_weight = float(
            self._arg(args, "gcs_lane_instance_geometry_quality_weight", 1.0)
        )
        self.lane_instance_survival_weight = float(self._arg(args, "gcs_lane_instance_survival_weight", 1.0))
        self.lane_instance_duplicate_weight = float(self._arg(args, "gcs_lane_instance_duplicate_weight", 0.5))
        self.lane_instance_novelty_weight = float(self._arg(args, "gcs_lane_instance_novelty_weight", 0.5))
        self.lane_instance_topology_weight = float(self._arg(args, "gcs_lane_instance_topology_weight", 0.5))
        self.lane_instance_identity_weight = float(self._arg(args, "gcs_lane_instance_identity_weight", 0.5))
        self.lane_instance_set_noop_weight = float(self._arg(args, "gcs_lane_instance_set_noop_weight", 0.5))
        self.lane_instance_count_weight = float(self._arg(args, "gcs_lane_instance_count_weight", 0.0))
        self.lane_instance_min_span = float(self._arg(args, "gcs_lane_instance_min_span", 2.0))
        self.lane_instance_duplicate_px = float(self._arg(args, "gcs_lane_instance_duplicate_px", 20.0))
        self.lane_instance_quality_tau_px = float(self._arg(args, "gcs_lane_instance_quality_tau_px", 20.0))
        self.lane_instance_identity_temperature = float(
            self._arg(args, "gcs_lane_instance_identity_temperature", 0.2)
        )
        self.lane_instance_set_margin = float(self._arg(args, "gcs_lane_instance_set_margin", 0.5))
        self.residual_point_weight = float(self._arg(args, "gcs_residual_point_weight", 5.0))
        self.residual_row_weight = float(self._arg(args, "gcs_residual_row_weight", 1.0))
        self.residual_valid_weight = float(self._arg(args, "gcs_residual_valid_weight", 1.0))
        self.residual_interval_weight = float(self._arg(args, "gcs_residual_interval_weight", 0.5))
        self.residual_interval_valid_consistency = float(
            self._arg(args, "gcs_residual_interval_valid_consistency", 0.0)
        )
        self.residual_positive_span = float(self._arg(args, "gcs_residual_positive_span", 0.0))
        self.residual_identity_pull_weight = float(self._arg(args, "gcs_residual_identity_pull_weight", 0.0))
        self.residual_identity_push_weight = float(self._arg(args, "gcs_residual_identity_push_weight", 0.0))
        self.residual_quality_weight = float(self._arg(args, "gcs_residual_quality_weight", 0.0))
        self.residual_quality_rank_weight = float(self._arg(args, "gcs_residual_quality_rank_weight", 0.0))
        self.residual_quality_rank_margin = float(self._arg(args, "gcs_residual_quality_rank_margin", 0.5))
        self.residual_quality_soft_px = float(self._arg(args, "gcs_residual_quality_soft_px", 0.0))
        self.residual_quality_rank_target_gap = float(
            self._arg(args, "gcs_residual_quality_rank_target_gap", 0.0)
        )
        self.residual_quality_hard_weight = float(self._arg(args, "gcs_residual_quality_hard_weight", 1.0))
        self.residual_identity_margin = float(self._arg(args, "gcs_residual_identity_margin", 0.2))
        self.residual_identity_assign_px = float(self._arg(args, "gcs_residual_identity_assign_px", 30.0))
        self.residual_quality_pos_weight_max = float(self._arg(args, "gcs_residual_quality_pos_weight_max", 20.0))
        self.residual_replace_weight = float(self._arg(args, "gcs_residual_replace_weight", 0.0))
        self.residual_replace_pos_weight_max = float(
            self._arg(args, "gcs_residual_replace_pos_weight_max", 50.0)
        )
        self.residual_replace_hit_px = float(self._arg(args, "gcs_residual_replace_hit_px", 20.0))
        self.residual_replace_soft_px = float(self._arg(args, "gcs_residual_replace_soft_px", 0.0))
        self.residual_replace_official_delta_scale = float(
            self._arg(args, "gcs_residual_replace_official_delta_scale", 0.0)
        )
        self.residual_replace_official_min_delta = float(
            self._arg(args, "gcs_residual_replace_official_min_delta", 0.001)
        )
        self.residual_replace_official_pixel_thr = float(
            self._arg(args, "gcs_residual_replace_official_pixel_thr", 20.0)
        )
        self.residual_replace_official_pt_thr = float(
            self._arg(args, "gcs_residual_replace_official_pt_thr", 0.85)
        )
        self.residual_replace_official_fp_weight = float(
            self._arg(args, "gcs_residual_replace_official_fp_weight", 0.02)
        )
        self.residual_replace_official_fn_weight = float(
            self._arg(args, "gcs_residual_replace_official_fn_weight", 0.02)
        )
        self.residual_replace_rank_weight = float(self._arg(args, "gcs_residual_replace_rank_weight", 0.0))
        self.residual_replace_listwise_weight = float(
            self._arg(args, "gcs_residual_replace_listwise_weight", 0.0)
        )
        self.residual_replace_listwise_positive_weight = float(
            self._arg(args, "gcs_residual_replace_listwise_positive_weight", 1.0)
        )
        self.residual_replace_listwise_mode = str(
            self._arg(args, "gcs_residual_replace_listwise_mode", "flat")
        ).lower()
        self.residual_replace_rank_margin = float(self._arg(args, "gcs_residual_replace_rank_margin", 0.5))
        self.residual_replace_rank_target_gap = float(
            self._arg(args, "gcs_residual_replace_rank_target_gap", 0.0)
        )
        self.residual_replace_valid_thr = float(self._arg(args, "gcs_residual_replace_valid_thr", 0.5))
        self.residual_replace_min_points = int(self._arg(args, "gcs_residual_replace_min_points", 4))
        self.residual_replace_max_det = int(self._arg(args, "gcs_residual_replace_max_det", 5))
        self.residual_exist_weight = float(self._arg(args, "gcs_residual_exist_weight", 1.0))
        self.residual_unmatched_weight = float(self._arg(args, "gcs_residual_unmatched_weight", 0.05))
        self.residual_min_lanes = int(self._arg(args, "gcs_residual_min_lanes", 4))
        self.residual_min_visible = int(self._arg(args, "gcs_residual_min_visible", 3))
        self.residual_max_visible = int(self._arg(args, "gcs_residual_max_visible", 10))
        self.residual_base_miss_px = float(self._arg(args, "gcs_residual_base_miss_px", 20.0))
        self.dense_centerline_weight = float(self._arg(args, "gcs_dense_centerline_weight", 1.0))
        self.dense_endpoint_weight = float(self._arg(args, "gcs_dense_endpoint_weight", 1.0))
        self.dense_embed_pull_weight = float(self._arg(args, "gcs_dense_embed_pull_weight", 0.25))
        self.dense_embed_push_weight = float(self._arg(args, "gcs_dense_embed_push_weight", 0.25))
        self.dense_embed_margin = float(self._arg(args, "gcs_dense_embed_margin", 0.5))
        self.dense_sigma_px = float(self._arg(args, "gcs_dense_sigma_px", 3.0))
        self.dense_pos_weight_max = float(self._arg(args, "gcs_dense_pos_weight_max", 50.0))
        self.dense_endpoint_pos_weight_max = float(
            self._arg(args, "gcs_dense_endpoint_pos_weight_max", 500.0)
        )
        self.dense_hard_short_weight = float(self._arg(args, "gcs_dense_hard_short_weight", 1.0))
        self.dense_hard_short_min_visible = int(self._arg(args, "gcs_dense_hard_short_min_visible", 3))
        self.dense_hard_short_visible_max = int(self._arg(args, "gcs_dense_hard_short_visible_max", 10))
        self.dense_hard_short_gt4_weight = float(self._arg(args, "gcs_dense_hard_short_gt4_weight", 1.0))
        self.dense_hard_short_gt5_weight = float(self._arg(args, "gcs_dense_hard_short_gt5_weight", 1.0))
        self.dense_endpoint_peak_weight = float(self._arg(args, "gcs_dense_endpoint_peak_weight", 0.0))
        self.dense_endpoint_peak_radius_px = float(self._arg(args, "gcs_dense_endpoint_peak_radius_px", 16.0))
        self.dense_endpoint_offset_weight = float(self._arg(args, "gcs_dense_endpoint_offset_weight", 0.0))
        self.dense_endpoint_offset_radius_px = float(self._arg(args, "gcs_dense_endpoint_offset_radius_px", 8.0))
        raw_endpoint_balance_mode = self._arg(args, "gcs_dense_endpoint_balance_mode", 0)
        if isinstance(raw_endpoint_balance_mode, str):
            endpoint_balance_mode = raw_endpoint_balance_mode.strip().lower()
        else:
            endpoint_balance_mode = {0: "support", 1: "mass"}.get(int(raw_endpoint_balance_mode), "")
        self.dense_endpoint_balance_mode = endpoint_balance_mode
        self.dense_candidate_gain = float(self._arg(args, "gcs_dense_candidate", 0.0))
        self.dense_candidate_quality_weight = float(
            self._arg(args, "gcs_dense_candidate_quality_weight", 1.0)
        )
        self.dense_candidate_replace_weight = float(
            self._arg(args, "gcs_dense_candidate_replace_weight", 1.0)
        )
        self.dense_candidate_quality_pos_weight_max = float(
            self._arg(args, "gcs_dense_candidate_quality_pos_weight_max", 50.0)
        )
        self.dense_candidate_replace_pos_weight_max = float(
            self._arg(args, "gcs_dense_candidate_replace_pos_weight_max", 100.0)
        )
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
        self.query_survival_rank_gain = float(
            lambda_query_survival_rank
            if lambda_query_survival_rank is not None
            else self._arg(args, "gcs_query_survival_rank", 0.0)
        )
        self.query_survival_rank_margin = float(
            query_survival_rank_margin
            if query_survival_rank_margin is not None
            else self._arg(args, "gcs_query_survival_rank_margin", 0.5)
        )
        self.query_survival_rank_max_ape_px = float(
            query_survival_rank_max_ape_px
            if query_survival_rank_max_ape_px is not None
            else self._arg(args, "gcs_query_survival_rank_max_ape_px", 20.0)
        )
        self.query_survival_rank_min_lanes = int(
            query_survival_rank_min_lanes
            if query_survival_rank_min_lanes is not None
            else self._arg(args, "gcs_query_survival_rank_min_lanes", 4)
        )
        self.query_valid_survival_gain = float(self._arg(args, "gcs_query_valid_survival", 0.0))
        self.query_valid_survival_hit_px = float(self._arg(args, "gcs_query_valid_survival_hit_px", 20.0))
        self.query_valid_survival_min_hit_ratio = float(
            self._arg(args, "gcs_query_valid_survival_min_hit_ratio", 0.85)
        )
        self.query_valid_survival_boundary_weight = float(
            self._arg(args, "gcs_query_valid_survival_boundary_weight", 4.0)
        )
        self.query_valid_survival_dice_weight = float(
            self._arg(args, "gcs_query_valid_survival_dice_weight", 1.0)
        )
        self.query_valid_survival_identity_weight = float(
            self._arg(args, "gcs_query_valid_survival_identity_weight", 0.0)
        )
        self.query_valid_survival_anchor_weight = float(
            self._arg(args, "gcs_query_valid_survival_anchor_weight", 1.0)
        )
        self.query_valid_interval_boundary_weight = float(
            self._arg(args, "gcs_query_valid_interval_boundary_weight", 0.0)
        )
        self.query_valid_interval_min_span = float(self._arg(args, "gcs_query_valid_interval_min_span", 4.0))
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
        if self.query_survival_rank_gain < 0.0:
            raise ValueError(f"gcs_query_survival_rank must be >= 0, got {self.query_survival_rank_gain}.")
        if self.query_survival_rank_margin < 0.0:
            raise ValueError(
                f"gcs_query_survival_rank_margin must be >= 0, got {self.query_survival_rank_margin}."
            )
        if self.query_survival_rank_max_ape_px <= 0.0:
            raise ValueError(
                "gcs_query_survival_rank_max_ape_px must be > 0, "
                f"got {self.query_survival_rank_max_ape_px}."
            )
        if self.query_survival_rank_min_lanes < 1:
            raise ValueError(
                f"gcs_query_survival_rank_min_lanes must be >= 1, got {self.query_survival_rank_min_lanes}."
            )
        if self.query_valid_survival_gain < 0.0:
            raise ValueError(f"gcs_query_valid_survival must be >= 0, got {self.query_valid_survival_gain}.")
        if self.query_valid_survival_hit_px <= 0.0:
            raise ValueError(
                "gcs_query_valid_survival_hit_px must be > 0, "
                f"got {self.query_valid_survival_hit_px}."
            )
        if not 0.0 < self.query_valid_survival_min_hit_ratio <= 1.0:
            raise ValueError(
                "gcs_query_valid_survival_min_hit_ratio must be in (0, 1], "
                f"got {self.query_valid_survival_min_hit_ratio}."
            )
        if self.query_valid_survival_boundary_weight < 1.0:
            raise ValueError(
                "gcs_query_valid_survival_boundary_weight must be >= 1, "
                f"got {self.query_valid_survival_boundary_weight}."
            )
        if self.query_valid_survival_dice_weight < 0.0:
            raise ValueError(
                "gcs_query_valid_survival_dice_weight must be >= 0, "
                f"got {self.query_valid_survival_dice_weight}."
            )
        if self.query_valid_survival_identity_weight < 0.0:
            raise ValueError(
                "gcs_query_valid_survival_identity_weight must be >= 0, "
                f"got {self.query_valid_survival_identity_weight}."
            )
        if self.query_valid_survival_anchor_weight < 0.0:
            raise ValueError(
                "gcs_query_valid_survival_anchor_weight must be >= 0, "
                f"got {self.query_valid_survival_anchor_weight}."
            )
        if self.query_valid_interval_boundary_weight < 0.0:
            raise ValueError(
                "gcs_query_valid_interval_boundary_weight must be >= 0, "
                f"got {self.query_valid_interval_boundary_weight}."
            )
        if self.query_valid_interval_min_span < 0.0:
            raise ValueError(
                "gcs_query_valid_interval_min_span must be >= 0, "
                f"got {self.query_valid_interval_min_span}."
            )
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
        if self.line_iou_valid_preserve_gain < 0.0:
            raise ValueError(
                "gcs_line_iou_valid_preserve must be >= 0, "
                f"got {self.line_iou_valid_preserve_gain}."
            )
        if self.line_iou_valid_preserve_gain != 0.0 and self.line_iou_gain == 0.0:
            raise ValueError("gcs_line_iou_valid_preserve requires nonzero gcs_line_iou.")
        if self.line_iou_exist_survival_gain < 0.0:
            raise ValueError(
                "gcs_line_iou_exist_survival must be >= 0, "
                f"got {self.line_iou_exist_survival_gain}."
            )
        if self.line_iou_exist_survival_gain != 0.0 and self.line_iou_gain == 0.0:
            raise ValueError("gcs_line_iou_exist_survival requires nonzero gcs_line_iou.")
        if self.line_iou_valid_visible_thr < 0:
            raise ValueError(
                "gcs_line_iou_valid_visible_thr must be >= 0, "
                f"got {self.line_iou_valid_visible_thr}."
            )
        if self.line_iou_valid_min_visible < 1:
            raise ValueError(
                "gcs_line_iou_valid_min_visible must be at least 1, "
                f"got {self.line_iou_valid_min_visible}."
            )
        if self.line_iou_valid_max_ape_px < 0.0:
            raise ValueError(
                "gcs_line_iou_valid_max_ape_px must be >= 0, "
                f"got {self.line_iou_valid_max_ape_px}."
            )
        if self.line_iou_valid_gt4_weight < 0.0 or self.line_iou_valid_gt5_weight < 0.0:
            raise ValueError(
                "gcs_line_iou_valid_gt4_weight and gcs_line_iou_valid_gt5_weight must be >= 0, "
                f"got {self.line_iou_valid_gt4_weight}, {self.line_iou_valid_gt5_weight}."
            )
        if self.dense_instance_gain < 0.0:
            raise ValueError("gcs_dense_instance must be >= 0.")
        if self.residual_proposal_gain < 0.0:
            raise ValueError("gcs_residual_proposal must be >= 0.")
        if self.lane_instance_set_gain < 0.0:
            raise ValueError("gcs_lane_instance_set must be >= 0.")
        lane_instance_weights = (
            self.lane_instance_visibility_weight,
            self.lane_instance_endpoint_weight,
            self.lane_instance_order_weight,
            self.lane_instance_contiguity_weight,
            self.lane_instance_positive_span_weight,
            self.lane_instance_empty_weight,
            self.lane_instance_geometry_quality_weight,
            self.lane_instance_survival_weight,
            self.lane_instance_duplicate_weight,
            self.lane_instance_novelty_weight,
            self.lane_instance_topology_weight,
            self.lane_instance_identity_weight,
            self.lane_instance_set_noop_weight,
            self.lane_instance_count_weight,
        )
        if any(weight < 0.0 for weight in lane_instance_weights):
            raise ValueError("gcs_lane_instance_*_weight values must be >= 0.")
        if self.lane_instance_min_span < 1.0:
            raise ValueError("gcs_lane_instance_min_span must be >= 1.")
        if self.lane_instance_duplicate_px <= 0.0 or self.lane_instance_quality_tau_px <= 0.0:
            raise ValueError("gcs_lane_instance_duplicate_px and quality_tau_px must be > 0.")
        if self.lane_instance_identity_temperature <= 0.0:
            raise ValueError("gcs_lane_instance_identity_temperature must be > 0.")
        if self.lane_instance_set_margin < 0.0:
            raise ValueError("gcs_lane_instance_set_margin must be >= 0.")
        if self.residual_replace_listwise_weight < 0.0:
            raise ValueError("gcs_residual_replace_listwise_weight must be >= 0.")
        if self.residual_replace_listwise_positive_weight <= 0.0:
            raise ValueError("gcs_residual_replace_listwise_positive_weight must be > 0.")
        if self.residual_replace_listwise_mode not in {"flat", "hierarchical"}:
            raise ValueError(
                "gcs_residual_replace_listwise_mode must be 'flat' or 'hierarchical', "
                f"got {self.residual_replace_listwise_mode!r}."
            )
        if self.residual_replace_listwise_weight > 0.0 and self.residual_replace_official_delta_scale <= 0.0:
            raise ValueError(
                "gcs_residual_replace_listwise_weight > 0 requires "
                "gcs_residual_replace_official_delta_scale > 0."
            )
        if self.residual_min_visible < 2 or self.residual_max_visible < self.residual_min_visible:
            raise ValueError("gcs_residual_min_visible/max_visible define an invalid visible-anchor range.")
        if self.residual_base_miss_px <= 0.0:
            raise ValueError("gcs_residual_base_miss_px must be positive.")
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
        if self.dense_endpoint_pos_weight_max < 1.0:
            raise ValueError("gcs_dense_endpoint_pos_weight_max must be >= 1.")
        if self.dense_hard_short_weight < 1.0:
            raise ValueError("gcs_dense_hard_short_weight must be >= 1.")
        if self.dense_hard_short_min_visible < 1:
            raise ValueError("gcs_dense_hard_short_min_visible must be >= 1.")
        if self.dense_hard_short_visible_max < self.dense_hard_short_min_visible:
            raise ValueError(
                "gcs_dense_hard_short_visible_max must be >= gcs_dense_hard_short_min_visible, "
                f"got {self.dense_hard_short_visible_max} < {self.dense_hard_short_min_visible}."
            )
        if self.dense_hard_short_gt4_weight < 0.0 or self.dense_hard_short_gt5_weight < 0.0:
            raise ValueError("gcs_dense_hard_short_gt4_weight and gcs_dense_hard_short_gt5_weight must be >= 0.")
        if self.dense_endpoint_peak_weight < 0.0:
            raise ValueError("gcs_dense_endpoint_peak_weight must be >= 0.")
        if self.dense_endpoint_peak_radius_px <= 0.0:
            raise ValueError("gcs_dense_endpoint_peak_radius_px must be > 0.")
        if self.dense_endpoint_offset_weight < 0.0:
            raise ValueError("gcs_dense_endpoint_offset_weight must be >= 0.")
        if self.dense_endpoint_offset_radius_px <= 0.0:
            raise ValueError("gcs_dense_endpoint_offset_radius_px must be > 0.")
        if self.dense_endpoint_balance_mode not in {"support", "mass"}:
            raise ValueError(
                "gcs_dense_endpoint_balance_mode must be 'support' or 'mass', "
                f"got {self.dense_endpoint_balance_mode!r}."
            )
        if self.dense_candidate_gain < 0.0:
            raise ValueError("gcs_dense_candidate must be >= 0.")
        if self.dense_candidate_quality_weight < 0.0 or self.dense_candidate_replace_weight < 0.0:
            raise ValueError("Dense candidate quality/replace weights must be >= 0.")
        if self.dense_candidate_quality_pos_weight_max < 1.0:
            raise ValueError("gcs_dense_candidate_quality_pos_weight_max must be >= 1.")
        if self.dense_candidate_replace_pos_weight_max < 1.0:
            raise ValueError("gcs_dense_candidate_replace_pos_weight_max must be >= 1.")
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
        self.residual_matcher = GCSHungarianMatcher(
            cost_point=5.0,
            cost_curve=0.05,
            cost_exist=0.1,
            image_size=self.image_size,
            min_overlap=self.residual_min_visible,
            max_x_dist=0.0,
            match_gate_px=0.0,
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

    def line_iou_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Masked fixed-y LineIoU over Hungarian-matched lanes."""
        lane_losses = []
        half_width = float(self.line_iou_half_width_px)
        min_points = int(self.line_iou_short_min_points)
        image_width = float(self.image_size[1])
        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=pred_points.device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=pred_points.device, dtype=torch.long)
            pred_x = pred_points[batch_index, src_idx, :, 0].float() * image_width
            target_x = gt_points[batch_index].to(device=pred_points.device, dtype=torch.float32)[tgt_idx, :, 0] * image_width
            valid = gt_valid[batch_index].to(device=pred_points.device)[tgt_idx] > 0.5
            valid_counts = valid.sum(dim=1)
            active = valid_counts >= min_points
            if not bool(active.any()):
                continue
            distance = (pred_x - target_x).abs()
            intersection = (2.0 * half_width - distance).clamp_min(0.0)
            union = 4.0 * half_width - intersection
            point_iou = intersection / union.clamp_min(1e-6)
            lane_iou = (point_iou * valid.float()).sum(dim=1) / valid_counts.clamp_min(1).float()
            lane_losses.append(1.0 - lane_iou[active])
        if not lane_losses:
            return self._zero_like(pred_points)
        return torch.cat(lane_losses).mean().to(dtype=pred_points.dtype)

    def _line_iou_short_active_matches(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor | None,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, float]]:
        """Select matched short GT4/GT5 lanes using the LineIoU valid-preserve gates."""
        if gt_lanes is None or self.line_iou_valid_visible_thr <= 0:
            return []

        gt_lanes = torch.as_tensor(gt_lanes, device=pred_points.device, dtype=pred_points.dtype).reshape(-1)
        if gt_lanes.numel() != pred_points.shape[0]:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={pred_points.shape[0]}.")

        matches: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        scale = self._pixel_scale_for(pred_points)
        max_ape = float(self.line_iou_valid_max_ape_px)

        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            gt_count = int(round(float(gt_lanes[batch_index].detach().cpu().item())))
            if gt_count == 4:
                count_weight = float(self.line_iou_valid_gt4_weight)
            elif gt_count == 5:
                count_weight = float(self.line_iou_valid_gt5_weight)
            else:
                continue
            if count_weight <= 0.0:
                continue

            src_idx = src_idx.to(device=pred_points.device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=pred_points.device, dtype=torch.long)
            valid = gt_valid[batch_index].to(device=pred_points.device, dtype=pred_points.dtype)[tgt_idx]
            visible_counts = valid.sum(dim=1)
            active = (visible_counts >= float(self.line_iou_valid_min_visible)) & (
                visible_counts <= float(self.line_iou_valid_visible_thr)
            )
            if not bool(active.any()):
                continue

            if max_ape > 0.0:
                pred = pred_points[batch_index, src_idx].detach()
                target = gt_points[batch_index].to(device=pred_points.device, dtype=pred_points.dtype)[tgt_idx]
                point_error = torch.norm((pred - target) * scale, dim=-1)
                ape = (point_error * valid).sum(dim=1) / visible_counts.clamp_min(1.0)
                active = active & (ape <= max_ape)
                if not bool(active.any()):
                    continue

            matches.append((batch_index, src_idx[active], valid[active], visible_counts[active], count_weight))

        return matches

    def line_iou_valid_preserve_loss(
        self,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Positive-only point-valid preservation for matched short GT4/GT5 lanes."""
        if pred_valid_logits is None:
            zero = self._zero_like(pred_points)
            return zero, zero, zero

        matches = self._line_iou_short_active_matches(pred_points, gt_points, gt_valid, indices, gt_lanes)
        if not matches:
            zero = self._zero_like(pred_points)
            return zero, zero, zero

        losses = []
        lane_weights = []
        lane_count = 0
        anchor_count = 0

        for batch_index, src_idx, valid_active, visible_counts_active, count_weight in matches:
            logits = pred_valid_logits[batch_index, src_idx]
            bce = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
            per_lane = (bce * valid_active).sum(dim=1) / visible_counts_active.clamp_min(1.0)
            losses.append(per_lane)
            lane_weights.append(torch.full_like(per_lane, count_weight))
            lane_count += int(per_lane.numel())
            anchor_count += int(valid_active.sum().detach().cpu().item())

        if not losses:
            zero = self._zero_like(pred_points)
            return zero, zero, zero

        loss_values = torch.cat(losses)
        weights = torch.cat(lane_weights).to(device=loss_values.device, dtype=loss_values.dtype)
        loss = (loss_values * weights).sum() / weights.sum().clamp_min(1.0)
        return (
            loss.to(dtype=pred_points.dtype),
            pred_points.new_tensor(float(lane_count)),
            pred_points.new_tensor(float(anchor_count)),
        )

    def line_iou_exist_survival_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Positive-only existence BCE for LineIoU-gated matched short GT4/GT5 lanes."""
        matches = self._line_iou_short_active_matches(pred_points, gt_points, gt_valid, indices, gt_lanes)
        if not matches:
            zero = self._zero_like(pred_points)
            return zero, zero

        losses = []
        lane_weights = []
        lane_count = 0
        for batch_index, src_idx, _valid_active, _visible_counts_active, count_weight in matches:
            logits = pred_logits[batch_index, src_idx]
            bce = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
            losses.append(bce)
            lane_weights.append(torch.full_like(bce, count_weight))
            lane_count += int(bce.numel())

        loss_values = torch.cat(losses)
        weights = torch.cat(lane_weights).to(device=loss_values.device, dtype=loss_values.dtype)
        loss = (loss_values * weights).sum() / weights.sum().clamp_min(1.0)
        return loss.to(dtype=pred_points.dtype), pred_points.new_tensor(float(lane_count))

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

    def query_survival_rank_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_base_logits: torch.Tensor | None,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rank the weakest accurate matched query above the strongest unmatched query."""
        zero = self._zero_like(pred_points)
        if self.query_survival_rank_gain == 0.0:
            return zero, zero, zero
        if pred_base_logits is None:
            raise KeyError("gcs_query_survival_rank > 0 requires preds['pred_base_logits'].")
        if pred_base_logits.shape != pred_logits.shape:
            raise ValueError(
                "pred_base_logits must match pred_logits for query-survival ranking, "
                f"got {tuple(pred_base_logits.shape)} vs {tuple(pred_logits.shape)}."
            )

        points = pred_points.detach()
        base_logits = pred_base_logits.detach()
        x_scale = self._spurious_x_scale(pred_points)
        gt_lanes = torch.as_tensor(gt_lanes, device=pred_logits.device).reshape(-1)
        losses = []
        margins = []
        num_queries = int(pred_logits.shape[1])

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if int(round(float(gt_lanes[batch_idx].detach().item()))) < self.query_survival_rank_min_lanes:
                continue
            if src_idx.numel() == 0:
                continue
            src_idx = src_idx.to(device=pred_logits.device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=pred_logits.device, dtype=torch.long)
            matched_mask = torch.zeros(num_queries, device=pred_logits.device, dtype=torch.bool)
            matched_mask[src_idx] = True
            unmatched_idx = torch.arange(num_queries, device=pred_logits.device)[~matched_mask]
            if unmatched_idx.numel() == 0:
                continue

            accurate_matches = []
            points_gt = gt_points[batch_idx].detach().to(device=points.device, dtype=points.dtype)
            valid_gt = gt_valid[batch_idx].detach().to(device=points.device)
            for query_idx, target_idx in zip(src_idx, tgt_idx):
                visible = valid_gt[target_idx] > 0.5
                if not bool(visible.any()):
                    continue
                ape_px = (
                    (points[batch_idx, query_idx, visible, 0] - points_gt[target_idx, visible, 0]).abs()
                    * x_scale
                ).mean()
                if float(ape_px.item()) <= self.query_survival_rank_max_ape_px:
                    accurate_matches.append(query_idx)
            if not accurate_matches:
                continue

            accurate_idx = torch.stack(accurate_matches)
            weakest_query = accurate_idx[base_logits[batch_idx, accurate_idx].argmin()]
            strongest_unmatched = unmatched_idx[base_logits[batch_idx, unmatched_idx].argmax()]
            observed_margin = pred_logits[batch_idx, weakest_query] - pred_logits[batch_idx, strongest_unmatched]
            losses.append(F.softplus(self.query_survival_rank_margin - observed_margin))
            margins.append(observed_margin.detach())

        if not losses:
            return zero, zero, zero
        return (
            torch.stack(losses).mean(),
            pred_logits.new_tensor(float(len(losses))),
            torch.stack(margins).mean(),
        )

    def query_valid_survival_loss(
        self,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        pred_base_valid_logits: torch.Tensor | None,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        pred_interval_bounds: torch.Tensor | None = None,
        pred_interval_base_bounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Correct visibility on prediction-only TuSimple-line-accuracy assignments."""
        zero = self._zero_like(pred_points)
        if self.query_valid_survival_gain == 0.0:
            return zero, zero, zero, zero
        if pred_valid_logits is None or pred_base_valid_logits is None:
            raise KeyError(
                "gcs_query_valid_survival > 0 requires pred_valid_logits and pred_base_valid_logits "
                "from the query-survival geometry YAML."
            )
        if pred_valid_logits.shape != pred_base_valid_logits.shape:
            raise ValueError(
                "pred_base_valid_logits must match pred_valid_logits for valid survival, "
                f"got {tuple(pred_base_valid_logits.shape)} vs {tuple(pred_valid_logits.shape)}."
            )

        points = pred_points.detach()
        x_scale = self._spurious_x_scale(pred_points)
        losses = []
        boundary_losses = []
        identity_losses = []
        dice_losses = []
        qualified_count = 0
        qualified_anchor_count = 0
        boundary_weight = float(self.query_valid_survival_boundary_weight)

        del indices
        for batch_idx in range(int(pred_points.shape[0])):
            target_points = gt_points[batch_idx].detach().to(device=points.device, dtype=points.dtype)
            target_valid = gt_valid[batch_idx].to(device=pred_valid_logits.device, dtype=pred_valid_logits.dtype)
            if target_points.numel() == 0:
                continue
            visible_mask = target_valid > 0.5
            valid_count = visible_mask.sum(dim=1).clamp_min(1.0)
            target_x_px = target_points[..., 0] * x_scale
            target_y_px = target_points[..., 1] * float(self.image_size[0])
            visible_float = visible_mask.to(dtype=points.dtype)
            y_mean = (target_y_px * visible_float).sum(dim=1) / valid_count
            x_mean = (target_x_px * visible_float).sum(dim=1) / valid_count
            centered_y = target_y_px - y_mean[:, None]
            centered_x = target_x_px - x_mean[:, None]
            slope_numerator = (centered_y * centered_x * visible_float).sum(dim=1)
            slope_denominator = (centered_y.square() * visible_float).sum(dim=1).clamp_min(1e-12)
            slope = torch.where(valid_count > 1.0, slope_numerator / slope_denominator, torch.zeros_like(valid_count))
            official_threshold = self.query_valid_survival_hit_px * torch.sqrt(1.0 + slope.square())
            x_error_px = (
                (points[batch_idx, :, None, :, 0] - target_points[None, :, :, 0]).abs()
                * x_scale
            )
            visible_hits = (x_error_px < official_threshold[None, :, None]) & visible_mask[None]
            invalid_count = (~visible_mask).sum(dim=1).to(dtype=points.dtype)
            line_accuracy = (visible_hits.sum(dim=2).to(dtype=points.dtype) + invalid_count[None]) / float(
                pred_points.shape[2]
            )
            ape_px = (x_error_px * visible_mask[None].to(dtype=points.dtype)).sum(dim=2) / valid_count[None]
            pair_score = line_accuracy - 1e-4 * ape_px
            flat_order = pair_score.flatten().argsort(descending=True)
            used_queries: set[int] = set()
            used_targets: set[int] = set()
            assignments: list[tuple[int, int]] = []
            target_count = int(target_points.shape[0])
            for flat_index in flat_order.tolist():
                query_index = int(flat_index // target_count)
                target_index = int(flat_index % target_count)
                if query_index in used_queries or target_index in used_targets:
                    continue
                if float(line_accuracy[query_index, target_index].item()) < self.query_valid_survival_min_hit_ratio:
                    continue
                assignments.append((query_index, target_index))
                used_queries.add(query_index)
                used_targets.add(target_index)
                if len(used_targets) == target_count:
                    break

            for query_index, target_index in assignments:
                query_idx = torch.tensor(query_index, device=pred_valid_logits.device, dtype=torch.long)
                target_idx = torch.tensor(target_index, device=pred_valid_logits.device, dtype=torch.long)
                visible = target_valid[target_idx] > 0.5
                logits = pred_valid_logits[batch_idx, query_idx]
                target = target_valid[target_idx]
                weights = torch.ones_like(target)
                visible_indices = torch.nonzero(visible, as_tuple=False).flatten()
                start = int(visible_indices[0].item())
                end = int(visible_indices[-1].item())
                for anchor in (start - 1, start, end, end + 1):
                    if 0 <= anchor < int(weights.numel()):
                        weights[anchor] = boundary_weight
                bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                bce_loss = (bce * weights).sum() / weights.sum().clamp_min(1.0)
                probability = logits.sigmoid()
                dice_numerator = 2.0 * (probability * target).sum() + 1.0
                dice_denominator = probability.sum() + target.sum() + 1.0
                dice_loss = 1.0 - dice_numerator / dice_denominator
                if self.query_valid_survival_anchor_weight > 0.0:
                    losses.append(bce_loss + self.query_valid_survival_dice_weight * dice_loss)
                dice_losses.append(dice_loss.detach())
                if self.query_valid_interval_boundary_weight > 0.0:
                    if pred_interval_bounds is None or pred_interval_base_bounds is None:
                        raise KeyError(
                            "gcs_query_valid_interval_boundary_weight > 0 requires interval bounds outputs."
                        )
                    predicted_bounds = pred_interval_bounds[batch_idx, query_idx]
                    base_bounds = pred_interval_base_bounds[batch_idx, query_idx].detach()
                    target_bounds = predicted_bounds.new_tensor((float(start), float(end)))
                    min_span = float(self.query_valid_interval_min_span)
                    if float((target_bounds[1] - target_bounds[0]).item()) < min_span:
                        target_center = target_bounds.mean()
                        target_start = (target_center - 0.5 * min_span).clamp(
                            0.0, float(pred_valid_logits.shape[-1] - 1) - min_span
                        )
                        target_bounds = torch.stack((target_start, target_start + min_span))
                    max_shift = 8.0
                    reachable_bounds = torch.maximum(
                        torch.minimum(target_bounds, base_bounds + max_shift),
                        base_bounds - max_shift,
                    ).clamp(0.0, float(pred_valid_logits.shape[-1] - 1))
                    boundary_losses.append(
                        F.smooth_l1_loss(predicted_bounds, reachable_bounds, reduction="mean", beta=1.0)
                        / max_shift
                    )
                qualified_count += 1
                qualified_anchor_count += int(visible.sum().item())

            if self.query_valid_survival_identity_weight > 0.0:
                unmatched_mask = torch.ones(
                    pred_valid_logits.shape[1], device=pred_valid_logits.device, dtype=torch.bool
                )
                if used_queries:
                    unmatched_mask[list(used_queries)] = False
                if unmatched_mask.any():
                    unmatched_delta = (
                        pred_valid_logits[batch_idx, unmatched_mask]
                        - pred_base_valid_logits[batch_idx, unmatched_mask].detach()
                    )
                    identity_losses.append(unmatched_delta.square().mean())

        if not losses and not boundary_losses and not identity_losses:
            return zero, zero, zero, zero
        correction_loss = torch.stack(losses).mean() if losses else zero
        if boundary_losses:
            correction_loss = correction_loss + self.query_valid_interval_boundary_weight * torch.stack(
                boundary_losses
            ).mean()
        if identity_losses:
            correction_loss = correction_loss + self.query_valid_survival_identity_weight * torch.stack(
                identity_losses
            ).mean()
        return (
            correction_loss,
            pred_valid_logits.new_tensor(float(qualified_count)),
            pred_valid_logits.new_tensor(float(qualified_anchor_count)),
            torch.stack(dice_losses).mean() if dice_losses else zero,
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

    @staticmethod
    def _dense_draw_gaussian(
        target: torch.Tensor,
        channel: int,
        x: float,
        y: float,
        sigma: float,
        value_scale: float = 1.0,
    ) -> None:
        """Draw one clipped Gaussian on a dense target map in-place."""
        centers = torch.tensor([[float(x), float(y)]], device=target.device, dtype=target.dtype)
        GCSLoss._dense_draw_gaussians(target, channel, centers, sigma, value_scale=value_scale)

    @staticmethod
    def _dense_draw_gaussians(
        target: torch.Tensor,
        channel: int,
        centers: torch.Tensor,
        sigma: float,
        value_scale: float = 1.0,
    ) -> None:
        """Draw many clipped Gaussians on one dense target channel in-place."""
        height, width = int(target.shape[-2]), int(target.shape[-1])
        if height <= 0 or width <= 0 or centers.numel() == 0:
            return
        radius = max(int(math.ceil(3.0 * float(sigma))), 1)
        centers = centers.to(device=target.device, dtype=target.dtype).reshape(-1, 2)
        finite = torch.isfinite(centers).all(dim=1)
        if not bool(finite.any()):
            return
        centers = centers[finite]
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=target.device, dtype=torch.long),
            torch.arange(-radius, radius + 1, device=target.device, dtype=torch.long),
            indexing="ij",
        )
        offsets_x = offsets_x.reshape(1, -1)
        offsets_y = offsets_y.reshape(1, -1)
        center_x = centers[:, 0:1]
        center_y = centers[:, 1:2]
        x_idx = center_x.round().to(torch.long) + offsets_x
        y_idx = center_y.round().to(torch.long) + offsets_y
        keep = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
        if not bool(keep.any()):
            return
        values = torch.exp(
            -(
                (x_idx.to(dtype=target.dtype) - center_x).square()
                + (y_idx.to(dtype=target.dtype) - center_y).square()
            )
            / (2.0 * float(sigma) ** 2)
        )
        flat_indices = (y_idx * width + x_idx)[keep]
        flat_values = values[keep] * float(value_scale)
        flat_target = target[channel].reshape(-1)
        if hasattr(flat_target, "scatter_reduce_"):
            flat_target.scatter_reduce_(0, flat_indices, flat_values, reduce="amax", include_self=True)
        else:
            for index, value in zip(flat_indices, flat_values):
                flat_target[index] = torch.maximum(flat_target[index], value)

    @staticmethod
    def _dense_hard_short_lane_weight(
        lane_count: int,
        visible_count: int,
        hard_short_weight: float = 1.0,
        min_visible: int = 3,
        visible_max: int = 10,
        gt4_weight: float = 1.0,
        gt5_weight: float = 1.0,
    ) -> float:
        if float(hard_short_weight) <= 1.0:
            return 1.0
        if int(visible_count) < int(min_visible) or int(visible_count) > int(visible_max):
            return 1.0
        if int(lane_count) == 4:
            return max(1.0, float(hard_short_weight) * float(gt4_weight))
        if int(lane_count) >= 5:
            return max(1.0, float(hard_short_weight) * float(gt5_weight))
        return 1.0

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
        start = start.to(device=target.device, dtype=target.dtype)
        end = end.to(device=target.device, dtype=target.dtype)
        for value in values:
            point = start * (1.0 - value) + end * value
            cls._dense_draw_gaussian(target, channel, float(point[0].item()), float(point[1].item()), sigma)

    @classmethod
    def _dense_targets(
        cls,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        output_hw: tuple[int, int],
        sigma_px: float,
        image_size,
        hard_short_weight: float = 1.0,
        hard_short_min_visible: int = 3,
        hard_short_visible_max: int = 10,
        hard_short_gt4_weight: float = 1.0,
        hard_short_gt5_weight: float = 1.0,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rasterize centerline and endpoint evidence from normalized lane points."""
        output_h, output_w = int(output_hw[0]), int(output_hw[1])
        if output_h <= 0 or output_w <= 0:
            raise ValueError(f"Dense evidence output size must be positive, got {output_hw}.")
        image_h, image_w = normalize_imgsz(image_size)
        stride_x = float(image_w) / float(max(output_w - 1, 1))
        stride_y = float(image_h) / float(max(output_h - 1, 1))
        sigma_cells = max(float(sigma_px) / max((stride_x + stride_y) * 0.5, 1.0), 0.5)
        batch_size = len(gt_points)
        device = gt_points[0].device if batch_size else torch.device("cpu")
        centerline = torch.zeros((batch_size, 1, output_h, output_w), device=device, dtype=torch.float32)
        endpoints = torch.zeros((batch_size, 2, output_h, output_w), device=device, dtype=torch.float32)
        centerline_weights = torch.ones_like(centerline) if return_weights else None
        endpoint_weights = torch.ones_like(endpoints) if return_weights else None

        with torch.no_grad():
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
                center_centers: list[torch.Tensor] = []
                endpoint_start_centers: list[torch.Tensor] = []
                endpoint_end_centers: list[torch.Tensor] = []
                lane_count = int(points_b.shape[0])
                for lane_points, lane_valid in zip(points_b, valid_b):
                    indices = torch.nonzero(lane_valid > 0.5, as_tuple=False).flatten()
                    if indices.numel() == 0:
                        continue
                    coords = lane_points[indices].clone()
                    coords[:, 0] *= float(max(output_w - 1, 1))
                    coords[:, 1] *= float(max(output_h - 1, 1))
                    lane_centers = [coords]
                    center_centers.append(coords)
                    endpoint_start_centers.append(coords[0:1])
                    endpoint_end_centers.append(coords[-1:])
                    for point_index in range(int(indices.numel()) - 1):
                        if int(indices[point_index + 1] - indices[point_index]) != 1:
                            continue
                        start = coords[point_index]
                        end = coords[point_index + 1]
                        distance = float(torch.linalg.vector_norm(end - start).detach().cpu().item())
                        steps = max(int(math.ceil(distance)) + 1, 2)
                        values = torch.linspace(0.0, 1.0, steps=steps, device=centerline.device, dtype=torch.float32)
                        line_points = start.view(1, 2) * (1.0 - values.view(-1, 1)) + end.view(1, 2) * values.view(-1, 1)
                        center_centers.append(line_points)
                        lane_centers.append(line_points)
                    lane_weight = cls._dense_hard_short_lane_weight(
                        lane_count,
                        int(indices.numel()),
                        hard_short_weight=hard_short_weight,
                        min_visible=hard_short_min_visible,
                        visible_max=hard_short_visible_max,
                        gt4_weight=hard_short_gt4_weight,
                        gt5_weight=hard_short_gt5_weight,
                    )
                    if lane_weight > 1.0 and centerline_weights is not None:
                        cls._dense_draw_gaussians(
                            centerline_weights[batch_index],
                            0,
                            torch.cat(lane_centers, dim=0),
                            sigma_cells,
                            value_scale=lane_weight,
                        )
                    if lane_weight > 1.0 and endpoint_weights is not None:
                        cls._dense_draw_gaussians(
                            endpoint_weights[batch_index],
                            0,
                            coords[0:1],
                            sigma_cells,
                            value_scale=lane_weight,
                        )
                        cls._dense_draw_gaussians(
                            endpoint_weights[batch_index],
                            1,
                            coords[-1:],
                            sigma_cells,
                            value_scale=lane_weight,
                        )
                if center_centers:
                    cls._dense_draw_gaussians(centerline[batch_index], 0, torch.cat(center_centers, dim=0), sigma_cells)
                if endpoint_start_centers:
                    cls._dense_draw_gaussians(
                        endpoints[batch_index],
                        0,
                        torch.cat(endpoint_start_centers, dim=0),
                        sigma_cells,
                    )
                    cls._dense_draw_gaussians(
                        endpoints[batch_index],
                        1,
                        torch.cat(endpoint_end_centers, dim=0),
                        sigma_cells,
                    )
        if return_weights:
            return centerline, endpoints, centerline_weights, endpoint_weights
        return centerline, endpoints

    @staticmethod
    def _dense_endpoint_pos_weight(
        target: torch.Tensor,
        max_weight: float,
        mode: str = "support",
    ) -> torch.Tensor:
        """Compute per-channel endpoint BCE balance from support or target mass."""
        if target.ndim != 4 or target.shape[1] != 2:
            raise ValueError(
                "Dense endpoint target must have shape B x 2 x H x W, "
                f"got {tuple(target.shape)}."
            )
        mode = str(mode).strip().lower()
        if mode not in {"support", "mass"}:
            raise ValueError(f"Unsupported dense endpoint balance mode {mode!r}.")
        target_float = target.float()
        if mode == "support":
            positive = (target_float > 0.0).sum(dim=(0, 2, 3), keepdim=True).to(dtype=target_float.dtype)
        else:
            positive = target_float.sum(dim=(0, 2, 3), keepdim=True)
        pixels_per_channel = int(target.shape[0]) * int(target.shape[2]) * int(target.shape[3])
        negative = target_float.new_tensor(float(pixels_per_channel)) - positive
        return (negative / positive.clamp_min(1.0)).clamp(min=1.0, max=float(max_weight))

    def _dense_endpoint_peak_loss(
        self,
        endpoint_logits: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
    ) -> torch.Tensor:
        """Local softmax endpoint-peak localization loss for dense endpoints."""
        if self.dense_endpoint_peak_weight <= 0.0:
            return endpoint_logits.sum() * 0.0
        if endpoint_logits.ndim != 4 or int(endpoint_logits.shape[1]) != 2:
            raise ValueError(
                "pred_dense_endpoint_logits must have shape B x 2 x Hf x Wf for endpoint peak loss, "
                f"got {tuple(endpoint_logits.shape)}."
            )
        output_h, output_w = int(endpoint_logits.shape[-2]), int(endpoint_logits.shape[-1])
        if output_h <= 0 or output_w <= 0:
            return endpoint_logits.sum() * 0.0
        image_h, image_w = normalize_imgsz(self.image_size)
        stride_x = float(image_w) / float(max(output_w - 1, 1))
        stride_y = float(image_h) / float(max(output_h - 1, 1))
        radius_cells = max(
            int(math.ceil(float(self.dense_endpoint_peak_radius_px) / max((stride_x + stride_y) * 0.5, 1.0))),
            1,
        )
        target_sigma = max(float(radius_cells) / 3.0, 0.75)
        losses: list[torch.Tensor] = []
        loss_weights: list[float] = []
        for batch_index, (points_b, valid_b) in enumerate(zip(gt_points, gt_valid)):
            points_b = points_b.to(device=endpoint_logits.device, dtype=torch.float32)
            valid_b = valid_b.to(device=endpoint_logits.device)
            if points_b.ndim != 3 or valid_b.shape != points_b.shape[:2]:
                raise ValueError(
                    "Dense endpoint peak GT points/valid shape mismatch: "
                    f"points={tuple(points_b.shape)}, valid={tuple(valid_b.shape)}."
                )
            for lane_points, lane_valid in zip(points_b, valid_b):
                indices = torch.nonzero(lane_valid > 0.5, as_tuple=False).flatten()
                if int(indices.numel()) == 0:
                    continue
                lane_weight = self._dense_hard_short_lane_weight(
                    int(points_b.shape[0]),
                    int(indices.numel()),
                    hard_short_weight=self.dense_hard_short_weight,
                    min_visible=self.dense_hard_short_min_visible,
                    visible_max=self.dense_hard_short_visible_max,
                    gt4_weight=self.dense_hard_short_gt4_weight,
                    gt5_weight=self.dense_hard_short_gt5_weight,
                )
                endpoint_indices = ((0, indices[0]), (1, indices[-1]))
                for channel, point_index in endpoint_indices:
                    point = lane_points[point_index, :2].clamp(0.0, 1.0)
                    center_x = point[0] * float(max(output_w - 1, 1))
                    center_y = point[1] * float(max(output_h - 1, 1))
                    x0 = max(int(torch.floor(center_x).item()) - radius_cells, 0)
                    x1 = min(int(torch.ceil(center_x).item()) + radius_cells, output_w - 1)
                    y0 = max(int(torch.floor(center_y).item()) - radius_cells, 0)
                    y1 = min(int(torch.ceil(center_y).item()) + radius_cells, output_h - 1)
                    crop = endpoint_logits[batch_index, int(channel), y0 : y1 + 1, x0 : x1 + 1].float().reshape(-1)
                    if crop.numel() == 0:
                        continue
                    yy, xx = torch.meshgrid(
                        torch.arange(y0, y1 + 1, device=endpoint_logits.device, dtype=torch.float32),
                        torch.arange(x0, x1 + 1, device=endpoint_logits.device, dtype=torch.float32),
                        indexing="ij",
                    )
                    dist2 = (xx.reshape(-1) - center_x).square() + (yy.reshape(-1) - center_y).square()
                    target = torch.exp(-dist2 / (2.0 * target_sigma * target_sigma))
                    target = target / target.sum().clamp_min(1.0e-12)
                    losses.append(-(target * F.log_softmax(crop, dim=0)).sum())
                    loss_weights.append(float(lane_weight))
        if not losses:
            return endpoint_logits.sum() * 0.0
        stacked = torch.stack(losses)
        weights = stacked.new_tensor(loss_weights)
        return (stacked * weights).sum().to(endpoint_logits) / weights.sum().clamp_min(1.0)

    def _dense_endpoint_offset_targets(
        self,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        output_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build endpoint-offset targets on visible lane support pixels."""
        output_h, output_w = int(output_hw[0]), int(output_hw[1])
        image_h, image_w = normalize_imgsz(self.image_size)
        stride_x = float(image_w) / float(max(output_w - 1, 1))
        stride_y = float(image_h) / float(max(output_h - 1, 1))
        radius_cells = max(
            int(math.ceil(float(self.dense_endpoint_offset_radius_px) / max((stride_x + stride_y) * 0.5, 1.0))),
            1,
        )
        batch_size = len(gt_points)
        device = gt_points[0].device if batch_size else torch.device("cpu")
        offsets = torch.zeros((batch_size, 4, output_h, output_w), device=device, dtype=torch.float32)
        weights = torch.zeros((batch_size, 1, output_h, output_w), device=device, dtype=torch.float32)

        with torch.no_grad():
            for batch_index, (points_b, valid_b) in enumerate(zip(gt_points, gt_valid)):
                points_b = points_b.to(device=device, dtype=torch.float32)
                valid_b = valid_b.to(device=device)
                if points_b.ndim != 3 or valid_b.shape != points_b.shape[:2]:
                    raise ValueError(
                        "Dense endpoint-offset GT points/valid shape mismatch: "
                        f"points={tuple(points_b.shape)}, valid={tuple(valid_b.shape)}."
                    )
                lane_count = int(points_b.shape[0])
                for lane_points, lane_valid in zip(points_b, valid_b):
                    indices = torch.nonzero(lane_valid > 0.5, as_tuple=False).flatten()
                    if int(indices.numel()) < 2:
                        continue
                    lane_weight = self._dense_hard_short_lane_weight(
                        lane_count,
                        int(indices.numel()),
                        hard_short_weight=self.dense_hard_short_weight,
                        min_visible=self.dense_hard_short_min_visible,
                        visible_max=self.dense_hard_short_visible_max,
                        gt4_weight=self.dense_hard_short_gt4_weight,
                        gt5_weight=self.dense_hard_short_gt5_weight,
                    )
                    coords = lane_points[indices, :2].clamp(0.0, 1.0).clone()
                    coords[:, 0] *= float(max(output_w - 1, 1))
                    coords[:, 1] *= float(max(output_h - 1, 1))
                    bottom = coords[0]
                    top = coords[-1]
                    for coord in coords:
                        cx = float(coord[0].item())
                        cy = float(coord[1].item())
                        x0 = max(int(math.floor(cx)) - radius_cells, 0)
                        x1 = min(int(math.ceil(cx)) + radius_cells, output_w - 1)
                        y0 = max(int(math.floor(cy)) - radius_cells, 0)
                        y1 = min(int(math.ceil(cy)) + radius_cells, output_h - 1)
                        if x1 < x0 or y1 < y0:
                            continue
                        yy, xx = torch.meshgrid(
                            torch.arange(y0, y1 + 1, device=device, dtype=torch.float32),
                            torch.arange(x0, x1 + 1, device=device, dtype=torch.float32),
                            indexing="ij",
                        )
                        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
                        local_weight = torch.exp(-dist2 / (2.0 * max(float(radius_cells) / 3.0, 0.75) ** 2))
                        local_weight = local_weight * float(lane_weight)
                        current_weight = weights[batch_index, 0, y0 : y1 + 1, x0 : x1 + 1]
                        update = local_weight > current_weight
                        if not bool(update.any()):
                            continue
                        target = torch.stack(
                            (
                                (bottom[0] - xx) / float(max(output_w - 1, 1)),
                                (bottom[1] - yy) / float(max(output_h - 1, 1)),
                                (top[0] - xx) / float(max(output_w - 1, 1)),
                                (top[1] - yy) / float(max(output_h - 1, 1)),
                            ),
                            dim=0,
                        )
                        crop_offsets = offsets[batch_index, :, y0 : y1 + 1, x0 : x1 + 1]
                        crop_offsets[:, update] = target[:, update]
                        current_weight[update] = local_weight[update]
        return offsets, weights

    def _dense_endpoint_offset_loss(
        self,
        preds: dict[str, torch.Tensor],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Regress same-lane bottom/top endpoint offsets from dense lane support."""
        if self.dense_endpoint_offset_weight <= 0.0:
            reference = preds.get("pred_dense_centerline_logits")
            if reference is None:
                return torch.tensor(0.0)
            return reference.sum() * 0.0
        pred_offsets = preds.get("pred_dense_endpoint_offsets")
        if pred_offsets is None:
            raise KeyError("gcs_dense_endpoint_offset_weight > 0 requires pred_dense_endpoint_offsets.")
        if pred_offsets.ndim != 4 or int(pred_offsets.shape[1]) != 4:
            raise ValueError(
                "pred_dense_endpoint_offsets must have shape B x 4 x Hf x Wf, "
                f"got {tuple(pred_offsets.shape)}."
            )
        if tuple(pred_offsets.shape[-2:]) != tuple(output_hw):
            raise ValueError("Dense endpoint offsets must share Hf,Wf with dense evidence maps.")
        target_offsets, target_weights = self._dense_endpoint_offset_targets(gt_points, gt_valid, output_hw)
        target_offsets = target_offsets.to(device=pred_offsets.device, dtype=pred_offsets.dtype)
        target_weights = target_weights.to(device=pred_offsets.device, dtype=pred_offsets.dtype)
        if float(target_weights.sum().detach().cpu().item()) <= 0.0:
            return pred_offsets.sum() * 0.0
        offset_error = F.smooth_l1_loss(pred_offsets.float(), target_offsets.float(), reduction="none")
        offset_error = offset_error.mean(dim=1, keepdim=True)
        return (offset_error * target_weights.float()).sum().to(pred_offsets) / target_weights.sum().clamp_min(1.0)

    def _dense_candidate_targets(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        output_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rasterize lane-quality and base-replacement targets at bottom endpoints."""
        output_h, output_w = int(output_hw[0]), int(output_hw[1])
        device = pred_points.device
        quality_target = torch.zeros((len(gt_points), 1, output_h, output_w), device=device, dtype=torch.float32)
        replace_target = torch.zeros_like(quality_target)
        image_h, image_w = normalize_imgsz(self.image_size)
        sigma_cells = max(
            float(self.dense_sigma_px)
            / max(
                (
                    float(image_w) / float(max(output_w - 1, 1))
                    + float(image_h) / float(max(output_h - 1, 1))
                )
                * 0.5,
                1.0,
            ),
            0.5,
        )
        pred_points = pred_points.detach().float()
        pred_logits = pred_logits.detach().float()
        pred_valid_logits = pred_valid_logits.detach().float()
        with torch.no_grad():
            for batch_index, (points_b, valid_b) in enumerate(zip(gt_points, gt_valid)):
                points_b = points_b.to(device=device, dtype=torch.float32)
                valid_b = valid_b.to(device=device, dtype=torch.float32)
                base_points_b = pred_points[batch_index]
                base_exist_b = pred_logits[batch_index].sigmoid() >= 0.001
                base_valid_b = pred_valid_logits[batch_index].sigmoid() >= 0.6
                for lane_points, lane_valid in zip(points_b, valid_b):
                    indices = torch.nonzero(lane_valid > 0.5, as_tuple=False).flatten()
                    if int(indices.numel()) == 0:
                        continue
                    coords = lane_points[indices].clone()
                    coords[:, 0] *= float(max(output_w - 1, 1))
                    coords[:, 1] *= float(max(output_h - 1, 1))
                    self._dense_draw_gaussian(
                        quality_target[batch_index],
                        0,
                        float(coords[0, 0].item()),
                        float(coords[0, 1].item()),
                        sigma_cells,
                    )
                    if int(indices.numel()) < 3:
                        continue
                    gt_x = lane_points[indices, 0] * float(image_w)
                    base_hit = False
                    for query_index in range(int(base_points_b.shape[0])):
                        if not bool(base_exist_b[query_index]):
                            continue
                        overlap = base_valid_b[query_index, indices]
                        if int(overlap.sum().item()) < 3:
                            continue
                        error = torch.abs(base_points_b[query_index, indices, 0] * float(image_w) - gt_x)
                        if bool(overlap.any()) and float(error[overlap].mean().item()) <= 20.0:
                            base_hit = True
                            break
                    if not base_hit:
                        self._dense_draw_gaussian(
                            replace_target[batch_index],
                            0,
                            float(coords[0, 0].item()),
                            float(coords[0, 1].item()),
                            sigma_cells,
                        )
        return quality_target, replace_target

    def dense_instance_loss(
        self,
        preds: dict[str, torch.Tensor],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Train dense centerline, endpoint, and image-local instance-embedding evidence."""
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

        center_target, endpoint_target, center_weight, endpoint_weight = self._dense_targets(
            gt_points,
            gt_valid,
            tuple(centerline_logits.shape[-2:]),
            sigma_px=float(self.dense_sigma_px),
            image_size=self.image_size,
            hard_short_weight=self.dense_hard_short_weight,
            hard_short_min_visible=self.dense_hard_short_min_visible,
            hard_short_visible_max=self.dense_hard_short_visible_max,
            hard_short_gt4_weight=self.dense_hard_short_gt4_weight,
            hard_short_gt5_weight=self.dense_hard_short_gt5_weight,
            return_weights=True,
        )
        center_target = center_target.to(device=centerline_logits.device, dtype=centerline_logits.dtype)
        endpoint_target = endpoint_target.to(device=endpoint_logits.device, dtype=endpoint_logits.dtype)
        center_weight = center_weight.to(device=centerline_logits.device, dtype=centerline_logits.dtype)
        endpoint_weight = endpoint_weight.to(device=endpoint_logits.device, dtype=endpoint_logits.dtype)

        def sparse_bce(
            logits: torch.Tensor,
            target: torch.Tensor,
            max_weight: float,
            loss_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            positive = target.sum()
            negative = target.numel() - positive
            pos_weight = (negative / positive.clamp_min(1.0)).clamp(
                min=1.0,
                max=float(max_weight),
            )
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
            if loss_weight is None:
                return loss.mean()
            return (loss * loss_weight).sum() / loss_weight.sum().clamp_min(1.0)

        centerline_loss = sparse_bce(centerline_logits, center_target, self.dense_pos_weight_max, center_weight)
        endpoint_pos_weight = self._dense_endpoint_pos_weight(
            endpoint_target,
            max_weight=self.dense_endpoint_pos_weight_max,
            mode=self.dense_endpoint_balance_mode,
        ).to(device=endpoint_logits.device, dtype=endpoint_logits.dtype)
        endpoint_loss_raw = F.binary_cross_entropy_with_logits(
            endpoint_logits,
            endpoint_target,
            pos_weight=endpoint_pos_weight,
            reduction="none",
        )
        endpoint_loss = (endpoint_loss_raw * endpoint_weight).sum() / endpoint_weight.sum().clamp_min(1.0)
        endpoint_peak_loss = self._dense_endpoint_peak_loss(endpoint_logits, gt_points, gt_valid)
        endpoint_offset_loss = self._dense_endpoint_offset_loss(
            preds,
            gt_points,
            gt_valid,
            tuple(centerline_logits.shape[-2:]),
        )

        normalized_embed = F.normalize(instance_embed, dim=1, eps=1.0e-6)
        pull_losses: list[torch.Tensor] = []
        pull_weights: list[float] = []
        push_losses: list[torch.Tensor] = []
        push_weights: list[float] = []
        for batch_index, (points_b, valid_b) in enumerate(zip(gt_points, gt_valid)):
            points_b = points_b.to(device=normalized_embed.device, dtype=normalized_embed.dtype)
            valid_b = valid_b.to(device=normalized_embed.device)
            if points_b.numel() == 0:
                continue
            image_lane_means: list[torch.Tensor] = []
            image_lane_weights: list[float] = []
            lane_count = int(points_b.shape[0])
            for lane_points, lane_valid in zip(points_b, valid_b):
                lane_mask = lane_valid > 0.5
                visible_count = int(lane_mask.sum().item())
                if visible_count < 2:
                    continue
                lane_weight = self._dense_hard_short_lane_weight(
                    lane_count,
                    visible_count,
                    hard_short_weight=self.dense_hard_short_weight,
                    min_visible=self.dense_hard_short_min_visible,
                    visible_max=self.dense_hard_short_visible_max,
                    gt4_weight=self.dense_hard_short_gt4_weight,
                    gt5_weight=self.dense_hard_short_gt5_weight,
                )
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
                pull_weights.append(float(lane_weight))
                image_lane_means.append(mean_embedding)
                image_lane_weights.append(float(lane_weight))
            if len(image_lane_means) >= 2:
                means = torch.stack(image_lane_means, dim=0)
                pair_losses = []
                for left in range(int(means.shape[0]) - 1):
                    distances = torch.linalg.vector_norm(means[left + 1 :] - means[left], dim=1)
                    pair_loss = F.relu(float(self.dense_embed_margin) - distances).square()
                    pair_weight = means.new_tensor(
                        [
                            0.5 * (float(image_lane_weights[left]) + float(image_lane_weights[right]))
                            for right in range(left + 1, len(image_lane_weights))
                        ]
                    )
                    pair_losses.append((pair_loss * pair_weight).sum() / pair_weight.sum().clamp_min(1.0))
                push_losses.append(torch.stack(pair_losses).mean())
                push_weights.append(max(float(weight) for weight in image_lane_weights))

        if pull_losses:
            pull_stack = torch.stack(pull_losses)
            pull_weight_tensor = pull_stack.new_tensor(pull_weights)
            embed_pull_loss = (pull_stack * pull_weight_tensor).sum() / pull_weight_tensor.sum().clamp_min(1.0)
        else:
            embed_pull_loss = centerline_logits.sum() * 0.0
        if push_losses:
            push_stack = torch.stack(push_losses)
            push_weight_tensor = push_stack.new_tensor(push_weights)
            embed_push_loss = (push_stack * push_weight_tensor).sum() / push_weight_tensor.sum().clamp_min(1.0)
        else:
            embed_push_loss = centerline_logits.sum() * 0.0
        dense_loss = (
            float(self.dense_centerline_weight) * centerline_loss
            + float(self.dense_endpoint_weight) * endpoint_loss
            + float(self.dense_endpoint_peak_weight) * endpoint_peak_loss
            + float(self.dense_endpoint_offset_weight) * endpoint_offset_loss
            + float(self.dense_embed_pull_weight) * embed_pull_loss
            + float(self.dense_embed_push_weight) * embed_push_loss
        )
        return (
            dense_loss,
            centerline_loss,
            endpoint_loss,
            endpoint_peak_loss,
            endpoint_offset_loss,
            embed_pull_loss,
            embed_push_loss,
            center_target.sum().detach(),
        )

    @staticmethod
    def _longest_contiguous_masks(mask: torch.Tensor, min_points: int) -> torch.Tensor:
        """Keep the longest true run for each N x K visibility mask."""
        if mask.ndim != 2:
            raise ValueError(f"visibility mask must be N x K, got {tuple(mask.shape)}.")
        output = torch.zeros_like(mask, dtype=torch.bool)
        for row_index in range(int(mask.shape[0])):
            row = mask[row_index].bool()
            padded = F.pad(row.to(dtype=torch.int8), (1, 1), value=0)
            difference = padded[1:] - padded[:-1]
            starts = torch.nonzero(difference == 1, as_tuple=False).flatten()
            ends = torch.nonzero(difference == -1, as_tuple=False).flatten()
            if starts.numel() == 0:
                continue
            lengths = ends - starts
            best = int(torch.argmax(lengths).item())
            length = int(lengths[best].item())
            if length >= int(min_points):
                start = int(starts[best].item())
                output[row_index, start : start + length] = True
        return output

    def _residual_official_set_score(
        self,
        pred_points: torch.Tensor,
        pred_valid: torch.Tensor,
        target_points: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return the fixed-y TuSimple official score proxy for one predicted set."""
        lane_count = int(target_points.shape[0])
        if lane_count <= 0:
            return pred_points.new_zeros(())
        image_height = float(self.image_size[0])
        image_width = float(self.image_size[1])
        gt_valid = target_valid.bool()
        gt_x_px = torch.where(
            gt_valid,
            target_points[..., 0].float() * image_width,
            target_points.new_full(target_points.shape[:2], -100.0, dtype=torch.float32),
        )
        gt_y_px = target_points[..., 1].float() * image_height
        valid_float = gt_valid.float()
        valid_count = valid_float.sum(dim=-1).clamp_min(1.0)
        mean_y = (gt_y_px * valid_float).sum(dim=-1) / valid_count
        mean_x = (gt_x_px * valid_float).sum(dim=-1) / valid_count
        centered_y = (gt_y_px - mean_y[:, None]) * valid_float
        centered_x = (gt_x_px - mean_x[:, None]) * valid_float
        slope = (centered_y * centered_x).sum(dim=-1) / centered_y.square().sum(dim=-1).clamp_min(1.0e-12)
        thresholds = float(self.residual_replace_official_pixel_thr) * torch.sqrt(1.0 + slope.square())

        if int(pred_points.shape[0]) == 0:
            max_accuracy = target_points.new_zeros(lane_count, dtype=torch.float32)
        else:
            pred_x_px = torch.where(
                pred_valid.bool(),
                pred_points[..., 0].float() * image_width,
                pred_points.new_full(pred_points.shape[:2], -100.0, dtype=torch.float32),
            )
            pair_accuracy = (
                (pred_x_px[:, None] - gt_x_px[None]).abs() < thresholds[None, :, None]
            ).float().mean(dim=-1)
            max_accuracy = pair_accuracy.max(dim=0).values

        matched = (max_accuracy >= float(self.residual_replace_official_pt_thr)).float()
        fn_count = float(lane_count) - matched.sum()
        if lane_count > 4 and float(fn_count.item()) > 0.0:
            fn_count = fn_count - 1.0
        score_sum = max_accuracy.sum()
        if lane_count > 4:
            score_sum = score_sum - max_accuracy.min()
        denominator = min(4.0, float(lane_count))
        accuracy = score_sum / denominator
        prediction_count = float(pred_points.shape[0])
        fp = (prediction_count - matched.sum()) / max(prediction_count, 1.0)
        fn = fn_count / denominator
        return (
            accuracy
            - float(self.residual_replace_official_fp_weight) * fp
            - float(self.residual_replace_official_fn_weight) * fn
        )

    def _residual_official_replace_targets(
        self,
        base_points: torch.Tensor,
        base_valid_probability: torch.Tensor,
        residual_points: torch.Tensor,
        residual_valid_probability: torch.Tensor,
        target_points: torch.Tensor,
        target_valid: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build proposal-victim targets from positive TuSimple official-score deltas."""
        target = residual_points.new_zeros((residual_points.shape[0], base_points.shape[0]))
        selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
        if selected_indices.numel() == 0:
            return target
        base_valid = self._longest_contiguous_masks(
            base_valid_probability >= self.residual_replace_valid_thr,
            self.residual_replace_min_points,
        )
        residual_valid = self._longest_contiguous_masks(
            residual_valid_probability >= self.residual_replace_valid_thr,
            self.residual_replace_min_points,
        )
        selected_points = base_points[selected_indices].detach()
        selected_valid = base_valid[selected_indices].detach()
        target_points = target_points.detach()
        target_valid = target_valid.detach()
        baseline_score = self._residual_official_set_score(
            selected_points,
            selected_valid,
            target_points,
            target_valid,
        )
        delta_scale = max(float(self.residual_replace_official_delta_scale), 1.0e-12)
        for victim_position, victim_index_tensor in enumerate(selected_indices):
            victim_index = int(victim_index_tensor.item())
            for proposal_index in range(int(residual_points.shape[0])):
                replaced_points = selected_points.clone()
                replaced_valid = selected_valid.clone()
                replaced_points[victim_position] = residual_points[proposal_index].detach()
                replaced_valid[victim_position] = residual_valid[proposal_index].detach()
                replacement_score = self._residual_official_set_score(
                    replaced_points,
                    replaced_valid,
                    target_points,
                    target_valid,
                )
                delta = replacement_score - baseline_score
                if float(delta.item()) > float(self.residual_replace_official_min_delta):
                    target[proposal_index, victim_index] = (delta / delta_scale).clamp(max=1.0)
        return target

    def residual_proposal_loss(
        self,
        preds,
        base_points,
        gt_points,
        gt_valid,
        gt_lanes,
        base_logits=None,
        base_valid_logits=None,
    ):
        """Train dense instance-channel proposals only on env30 raw20 base-miss GT4/GT5 short lanes."""
        required = (
            "pred_residual_points",
            "pred_residual_valid_logits",
            "pred_residual_exist_logits",
            "pred_residual_start_logits",
            "pred_residual_end_logits",
            "pred_residual_row_logits",
            "pred_residual_identity",
            "pred_residual_quality_logits",
        )
        missing = [name for name in required if name not in preds]
        if missing:
            raise KeyError(f"gcs_residual_proposal > 0 requires outputs: {missing}.")
        residual_points = preds["pred_residual_points"]
        residual_valid = preds["pred_residual_valid_logits"]
        residual_exist = preds["pred_residual_exist_logits"]
        residual_start = preds["pred_residual_start_logits"]
        residual_end = preds["pred_residual_end_logits"]
        residual_rows = preds["pred_residual_row_logits"]
        residual_identity = preds["pred_residual_identity"]
        residual_quality = preds["pred_residual_quality_logits"]
        residual_replace = preds.get("pred_residual_replace_logits")
        residual_noop = preds.get("pred_residual_noop_logit")
        replace_supervision = self.residual_replace_weight > 0.0 or self.residual_replace_listwise_weight > 0.0
        if replace_supervision:
            if residual_replace is None:
                raise KeyError("Residual replacement supervision requires pred_residual_replace_logits.")
            if base_logits is None or base_valid_logits is None:
                raise ValueError("Residual replacement supervision requires base logits and point-valid logits.")
        if self.residual_replace_listwise_weight > 0.0 and residual_noop is None:
            raise KeyError(
                "gcs_residual_replace_listwise_weight > 0 requires pred_residual_noop_logit."
            )
        zero = self._zero_like(residual_points)
        point_terms, row_terms, valid_terms, interval_terms, consistency_terms, positive_span_terms = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        identity_pull_terms, identity_push_terms, quality_terms, quality_rank_terms, exist_terms = (
            [],
            [],
            [],
            [],
            [],
        )
        replace_terms, replace_rank_terms, replace_listwise_terms = [], [], []
        replace_pos = zero.clone()
        replace_action_pos = zero.clone()
        replace_action_correct = zero.clone()
        replace_action_count = zero.clone()
        quality_pos = zero.clone()
        identity_pair_count = zero.clone()
        target_count = zero.clone()
        match_count = zero.clone()
        full_hit20 = zero.clone()
        image_width = float(self.image_size[1])
        row_width = int(residual_rows.shape[-1])

        for batch_index in range(residual_points.shape[0]):
            lane_count = int(gt_lanes[batch_index].item())
            if lane_count < self.residual_min_lanes:
                continue
            target_points_all = gt_points[batch_index].to(residual_points)
            target_valid_all = gt_valid[batch_index].to(device=residual_points.device, dtype=residual_points.dtype)
            if target_points_all.numel() == 0:
                continue
            visible_count = target_valid_all.sum(dim=1)
            short_mask = (visible_count >= self.residual_min_visible) & (visible_count <= self.residual_max_visible)
            point_error_px = (
                (base_points[batch_index, :, None, :, 0] - target_points_all[None, :, :, 0]).abs()
                * image_width
                * target_valid_all[None]
            )
            base_ape = point_error_px.sum(dim=-1) / visible_count.clamp_min(1.0)[None]
            residual_ape_all = (
                (residual_points[batch_index, :, None, :, 0] - target_points_all[None, :, :, 0]).abs()
                * image_width
                * target_valid_all[None]
            ).sum(dim=-1) / visible_count.clamp_min(1.0)[None]
            if self.residual_quality_soft_px > 0.0:
                all_valid = target_valid_all.bool()
                all_full_span = (
                    (residual_valid[batch_index].sigmoid()[:, None, :] >= 0.5) | ~all_valid[None]
                ).all(dim=-1)
                geometry_quality = (1.0 - residual_ape_all / self.residual_quality_soft_px).clamp(0.0, 1.0)
                soft_quality_target = (
                    geometry_quality * all_full_span.to(geometry_quality.dtype)
                ).max(dim=1).values.detach()
            else:
                soft_quality_target = None
            if replace_supervision:
                base_valid_probability = base_valid_logits[batch_index].sigmoid()
                base_decodable = (base_valid_probability >= self.residual_replace_valid_thr).sum(dim=-1)
                base_decodable = base_decodable >= self.residual_replace_min_points
                decodable_indices = torch.nonzero(base_decodable, as_tuple=False).flatten()
                selected_mask = torch.zeros_like(base_decodable)
                if decodable_indices.numel() > 0:
                    selected_count = min(self.residual_replace_max_det, int(decodable_indices.numel()))
                    selected_local = base_logits[batch_index, decodable_indices].topk(selected_count).indices
                    selected_mask[decodable_indices[selected_local]] = True

                gt_visible = target_valid_all.bool()
                base_full_span = (
                    (base_valid_probability[:, None, :] >= self.residual_replace_valid_thr) | ~gt_visible[None]
                ).all(dim=-1)
                selected_hit = (
                    base_full_span
                    & (base_ape <= self.residual_replace_hit_px)
                    & selected_mask[:, None]
                )
                missing_target = ~selected_hit.any(dim=0)
                unique_selected_hit = selected_hit.sum(dim=0) == 1
                protected_query = (selected_hit & unique_selected_hit[None]).any(dim=1)
                safe_victim = selected_mask & ~protected_query

                if self.residual_replace_official_delta_scale > 0.0:
                    with torch.no_grad():
                        replace_target_float = self._residual_official_replace_targets(
                            base_points[batch_index],
                            base_valid_probability,
                            residual_points[batch_index],
                            residual_valid[batch_index].sigmoid(),
                            target_points_all[:lane_count],
                            target_valid_all[:lane_count],
                            selected_mask,
                        )
                else:
                    residual_full_span = (
                        (residual_valid[batch_index].sigmoid()[:, None, :] >= self.residual_replace_valid_thr)
                        | ~gt_visible[None]
                    ).all(dim=-1)
                    if self.residual_replace_soft_px > 0.0:
                        residual_missing_quality = (
                            (1.0 - residual_ape_all / self.residual_replace_soft_px).clamp(0.0, 1.0)
                            * residual_full_span.to(residual_ape_all.dtype)
                            * missing_target[None].to(residual_ape_all.dtype)
                        ).max(dim=1).values
                    else:
                        residual_missing_quality = (
                            residual_full_span
                            & (residual_ape_all <= self.residual_replace_hit_px)
                            & missing_target[None]
                        ).any(dim=1).to(residual_ape_all.dtype)
                    replace_target_float = residual_missing_quality[:, None] * safe_victim[None].to(
                        residual_ape_all.dtype
                    )
                replace_mask = selected_mask[None].expand_as(replace_target_float)
                if replace_mask.any():
                    replace_target_float = replace_target_float.to(dtype=residual_replace.dtype)
                    positive = replace_target_float[replace_mask].sum()
                    if self.residual_replace_weight > 0.0:
                        negative = replace_mask.sum().to(dtype=positive.dtype) - positive
                        pos_weight = (negative / positive.clamp_min(1.0)).clamp(
                            min=1.0, max=self.residual_replace_pos_weight_max
                        )
                        replace_terms.append(
                            F.binary_cross_entropy_with_logits(
                                residual_replace[batch_index][replace_mask],
                                replace_target_float[replace_mask],
                                pos_weight=pos_weight,
                            )
                        )
                    replace_pos = replace_pos + positive
                    if self.residual_replace_rank_weight > 0.0:
                        selected_victims = torch.nonzero(selected_mask, as_tuple=False).flatten()
                        for victim_index in selected_victims:
                            victim_target = replace_target_float[:, victim_index]
                            target_difference = victim_target[:, None] - victim_target[None, :]
                            ordered_pairs = target_difference > self.residual_replace_rank_target_gap
                            if ordered_pairs.any():
                                victim_logits = residual_replace[batch_index, :, victim_index]
                                logit_difference = victim_logits[:, None] - victim_logits[None, :]
                                replace_rank_terms.append(
                                    F.relu(
                                        self.residual_replace_rank_margin - logit_difference[ordered_pairs]
                                    ).mean()
                                )
                    if self.residual_replace_listwise_weight > 0.0:
                        selected_victims = torch.nonzero(selected_mask, as_tuple=False).flatten()
                        action_logits = torch.cat(
                            (
                                residual_noop[batch_index].reshape(1),
                                residual_replace[batch_index, :, selected_victims].reshape(-1),
                            )
                        )
                        action_utility = replace_target_float[:, selected_victims].reshape(-1)
                        best_utility, best_action = action_utility.max(dim=0)
                        has_positive_action = best_utility > 0.0
                        action_target = torch.where(
                            has_positive_action,
                            best_action + 1,
                            best_action.new_zeros(()),
                        ).reshape(1)
                        if self.residual_replace_listwise_mode == "hierarchical":
                            replacement_logits = action_logits[1:]
                            presence_logit = replacement_logits.max() - action_logits[0]
                            presence_target = has_positive_action.to(dtype=presence_logit.dtype).reshape(1)
                            listwise_term = F.binary_cross_entropy_with_logits(
                                presence_logit.reshape(1),
                                presence_target,
                                pos_weight=presence_logit.new_tensor(self.residual_replace_listwise_positive_weight),
                            )
                            if bool(has_positive_action.item()):
                                listwise_term = listwise_term + F.cross_entropy(
                                    replacement_logits.reshape(1, -1),
                                    best_action.reshape(1),
                                )
                        else:
                            listwise_term = F.cross_entropy(action_logits.reshape(1, -1), action_target)
                            if bool(has_positive_action.item()):
                                listwise_term = listwise_term * self.residual_replace_listwise_positive_weight
                        replace_listwise_terms.append(listwise_term)
                        replace_action_pos = replace_action_pos + has_positive_action.to(replace_action_pos.dtype)
                        replace_action_correct = replace_action_correct + (
                            action_logits.argmax(dim=0) == action_target[0]
                        ).to(replace_action_correct.dtype)
                        replace_action_count = replace_action_count + 1.0
            nearest_ape, nearest_target = residual_ape_all.min(dim=1)
            assigned = nearest_ape <= self.residual_identity_assign_px
            pair_mask = torch.triu(torch.ones_like(nearest_ape[:, None] == nearest_ape[None], dtype=torch.bool), diagonal=1)
            assigned_pairs = assigned[:, None] & assigned[None, :] & pair_mask
            same_identity = assigned_pairs & (nearest_target[:, None] == nearest_target[None, :])
            different_identity = assigned_pairs & (nearest_target[:, None] != nearest_target[None, :])
            similarity = residual_identity[batch_index] @ residual_identity[batch_index].transpose(0, 1)
            if same_identity.any():
                identity_pull_terms.append((1.0 - similarity[same_identity]).mean())
                identity_pair_count = identity_pair_count + same_identity.sum()
            if different_identity.any():
                identity_push_terms.append(F.relu(similarity[different_identity] - self.residual_identity_margin).mean())
                identity_pair_count = identity_pair_count + different_identity.sum()
            target_indices = torch.nonzero(short_mask & (base_ape.min(dim=0).values > self.residual_base_miss_px), as_tuple=False).flatten()
            target_count = target_count + target_indices.numel()
            existence_target = torch.zeros_like(residual_exist[batch_index])
            validity_target = torch.zeros_like(residual_valid[batch_index])
            validity_weight = torch.full_like(validity_target, self.residual_unmatched_weight)
            quality_target = (
                torch.zeros_like(residual_quality[batch_index])
                if soft_quality_target is None
                else soft_quality_target.to(dtype=residual_quality.dtype)
            )
            quality_sample_weight = 1.0

            if target_indices.numel() > 0:
                target_points = target_points_all[target_indices]
                target_valid = target_valid_all[target_indices]
                src_idx, tgt_idx = self.residual_matcher(
                    residual_points[batch_index : batch_index + 1],
                    residual_exist[batch_index : batch_index + 1],
                    [target_points],
                    [target_valid],
                )[0]
                if src_idx.numel() > 0:
                    matched_points = target_points[tgt_idx]
                    matched_valid = target_valid[tgt_idx]
                    existence_target[src_idx] = 1.0
                    validity_target[src_idx] = matched_valid
                    validity_weight[src_idx] = 1.0
                    denom = matched_valid.sum().clamp_min(1.0)
                    point_terms.append(
                        ((residual_points[batch_index, src_idx, :, 0] - matched_points[:, :, 0]).abs() * matched_valid).sum()
                        / denom
                    )
                    target_x_index = (matched_points[:, :, 0] * float(row_width - 1)).round().long().clamp(0, row_width - 1)
                    matched_rows = residual_rows[batch_index, src_idx]
                    row_terms.append(F.cross_entropy(matched_rows[matched_valid.bool()], target_x_index[matched_valid.bool()]))
                    valid_bool = matched_valid.bool()
                    start_target = valid_bool.float().argmax(dim=1)
                    end_target = residual_points.shape[2] - 1 - valid_bool.flip(dims=(1,)).float().argmax(dim=1)
                    interval_terms.append(
                        0.5
                        * (
                            F.cross_entropy(residual_start[batch_index, src_idx], start_target)
                            + F.cross_entropy(residual_end[batch_index, src_idx], end_target)
                        )
                    )
                    matched_valid_logits = residual_valid[batch_index, src_idx]
                    start_prob = residual_start[batch_index, src_idx].softmax(dim=-1)
                    end_prob = residual_end[batch_index, src_idx].softmax(dim=-1)
                    interval_prob = start_prob.cumsum(dim=-1) * end_prob.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
                    consistency_terms.append(
                        F.binary_cross_entropy_with_logits(matched_valid_logits, interval_prob.detach())
                        + F.binary_cross_entropy_with_logits(
                            torch.logit(interval_prob.clamp(min=1e-6, max=1.0 - 1e-6)),
                            matched_valid_logits.sigmoid().detach(),
                        )
                    )
                    positive_span_terms.append(
                        F.binary_cross_entropy_with_logits(
                            matched_valid_logits[valid_bool],
                            torch.ones_like(matched_valid_logits[valid_bool]),
                        )
                    )
                    matched_ape = (
                        (residual_points[batch_index, src_idx, :, 0] - matched_points[:, :, 0]).abs()
                        * image_width
                        * matched_valid
                    ).sum(dim=1) / matched_valid.sum(dim=1).clamp_min(1.0)
                    match_count = match_count + src_idx.numel()
                    full_hit20 = full_hit20 + (matched_ape <= 20.0).sum()

                eligible_ape = residual_ape_all[:, target_indices]
                eligible_valid = target_valid_all[target_indices].bool()
                full_span = (
                    (residual_valid[batch_index].sigmoid()[:, None, :] >= 0.5) | ~eligible_valid[None]
                ).all(dim=-1)
                strict_hit = full_span & (eligible_ape <= 20.0)
                if soft_quality_target is None:
                    quality_target = strict_hit.any(dim=1).to(dtype=quality_target.dtype)
                else:
                    eligible_geometry_quality = (
                        1.0 - eligible_ape / self.residual_quality_soft_px
                    ).clamp(0.0, 1.0)
                    quality_target = (
                        eligible_geometry_quality * full_span.to(eligible_geometry_quality.dtype)
                    ).max(dim=1).values.detach()
                quality_sample_weight = self.residual_quality_hard_weight

            positive = quality_target.sum()
            negative = quality_target.numel() - positive
            pos_weight = (negative / positive.clamp_min(1.0)).clamp(min=1.0, max=self.residual_quality_pos_weight_max)
            quality_terms.append(
                F.binary_cross_entropy_with_logits(
                    residual_quality[batch_index], quality_target, pos_weight=pos_weight
                )
                * quality_sample_weight
            )
            if soft_quality_target is not None:
                target_difference = quality_target[:, None] - quality_target[None, :]
                ordered_pairs = target_difference > self.residual_quality_rank_target_gap
                if ordered_pairs.any():
                    logit_difference = residual_quality[batch_index][:, None] - residual_quality[batch_index][None, :]
                    quality_rank_terms.append(
                        F.relu(self.residual_quality_rank_margin - logit_difference[ordered_pairs]).mean()
                        * quality_sample_weight
                    )
            else:
                positive_logits = residual_quality[batch_index][quality_target > 0.5]
                negative_logits = residual_quality[batch_index][quality_target <= 0.5]
                if positive_logits.numel() > 0 and negative_logits.numel() > 0:
                    quality_rank_terms.append(
                        F.relu(
                            self.residual_quality_rank_margin
                            - positive_logits[:, None]
                            + negative_logits[None, :]
                        ).mean()
                        * quality_sample_weight
                    )
            quality_pos = quality_pos + positive

            exist_weight = torch.where(existence_target > 0.5, torch.ones_like(existence_target), self.residual_unmatched_weight)
            exist_terms.append(
                (F.binary_cross_entropy_with_logits(residual_exist[batch_index], existence_target, reduction="none") * exist_weight).sum()
                / exist_weight.sum().clamp_min(1.0)
            )
            valid_terms.append(
                (F.binary_cross_entropy_with_logits(residual_valid[batch_index], validity_target, reduction="none") * validity_weight).sum()
                / validity_weight.sum().clamp_min(1.0)
            )

        def mean_or_zero(values):
            return torch.stack(values).mean() if values else zero

        point_loss = mean_or_zero(point_terms)
        row_loss = mean_or_zero(row_terms)
        valid_loss = mean_or_zero(valid_terms)
        interval_loss = mean_or_zero(interval_terms)
        consistency_loss = mean_or_zero(consistency_terms)
        positive_span_loss = mean_or_zero(positive_span_terms)
        identity_pull_loss = mean_or_zero(identity_pull_terms)
        identity_push_loss = mean_or_zero(identity_push_terms)
        quality_loss = mean_or_zero(quality_terms)
        quality_rank_loss = mean_or_zero(quality_rank_terms)
        exist_loss = mean_or_zero(exist_terms)
        replace_loss = mean_or_zero(replace_terms)
        replace_rank_loss = mean_or_zero(replace_rank_terms)
        replace_listwise_loss = mean_or_zero(replace_listwise_terms)
        replace_action_acc = replace_action_correct / replace_action_count.clamp_min(1.0)
        total = (
            self.residual_point_weight * point_loss
            + self.residual_row_weight * row_loss
            + self.residual_valid_weight * valid_loss
            + self.residual_interval_weight * interval_loss
            + self.residual_interval_valid_consistency * consistency_loss
            + self.residual_positive_span * positive_span_loss
            + self.residual_identity_pull_weight * identity_pull_loss
            + self.residual_identity_push_weight * identity_push_loss
            + self.residual_quality_weight * quality_loss
            + self.residual_quality_rank_weight * quality_rank_loss
            + self.residual_exist_weight * exist_loss
            + self.residual_replace_weight * replace_loss
            + self.residual_replace_rank_weight * replace_rank_loss
            + self.residual_replace_listwise_weight * replace_listwise_loss
        )
        return (
            total,
            point_loss,
            row_loss,
            valid_loss,
            interval_loss,
            consistency_loss,
            positive_span_loss,
            identity_pull_loss,
            identity_push_loss,
            quality_loss,
            quality_rank_loss,
            quality_pos,
            identity_pair_count,
            exist_loss,
            target_count,
            match_count,
            full_hit20,
            replace_loss,
            replace_pos,
            replace_rank_loss,
            replace_listwise_loss,
            replace_action_pos,
            replace_action_acc,
        )

    def lane_instance_set_loss(
        self,
        preds: dict[str, torch.Tensor],
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, ...]:
        """Train the default-off lane-instance-set representation from matched lane targets."""
        required = (
            "pred_lane_instance_points",
            "pred_lane_instance_start_logits",
            "pred_lane_instance_end_logits",
            "pred_lane_instance_valid_logits",
            "pred_lane_instance_interval_start",
            "pred_lane_instance_interval_end",
            "pred_lane_instance_identity",
            "pred_lane_instance_pair_duplicate_logits",
            "pred_lane_instance_novelty_logits",
            "pred_lane_instance_left_logits",
            "pred_lane_instance_right_logits",
            "pred_lane_instance_geometry_quality_logits",
            "pred_lane_instance_survival_logits",
            "pred_lane_instance_empty_logit",
        )
        missing = [key for key in required if key not in preds]
        if missing:
            raise KeyError(f"gcs_lane_instance_set > 0 requires output keys: {missing}.")

        points = preds["pred_lane_instance_points"]
        start_logits = preds["pred_lane_instance_start_logits"]
        end_logits = preds["pred_lane_instance_end_logits"]
        valid_logits = preds["pred_lane_instance_valid_logits"]
        interval_start = preds["pred_lane_instance_interval_start"]
        interval_end = preds["pred_lane_instance_interval_end"]
        identity = preds["pred_lane_instance_identity"]
        duplicate_logits = preds["pred_lane_instance_pair_duplicate_logits"]
        novelty_logits = preds["pred_lane_instance_novelty_logits"]
        left_logits = preds["pred_lane_instance_left_logits"]
        right_logits = preds["pred_lane_instance_right_logits"]
        quality_logits = preds["pred_lane_instance_geometry_quality_logits"]
        survival_logits = preds["pred_lane_instance_survival_logits"]
        empty_logit = preds["pred_lane_instance_empty_logit"]
        bsz, queries, points_per_lane, _ = points.shape
        if start_logits.shape != (bsz, queries, points_per_lane) or end_logits.shape != start_logits.shape:
            raise ValueError("lane-instance start/end logits must have shape B x Q x K.")
        if valid_logits.shape != start_logits.shape or survival_logits.shape != (bsz, queries):
            raise ValueError("lane-instance valid/survival tensor shapes do not match B x Q x K/B x Q.")
        if duplicate_logits.shape != (bsz, queries, queries):
            raise ValueError("lane-instance duplicate logits must have shape B x Q x Q.")

        zero = self._zero_like(points)
        buckets: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "visibility",
                "start",
                "end",
                "order",
                "contiguity",
                "span",
                "empty",
                "quality",
                "survival",
                "duplicate",
                "novelty",
                "topology",
                "identity",
                "set_noop",
            )
        }
        target_survival = torch.zeros_like(survival_logits)
        target_quality = torch.zeros_like(quality_logits)
        target_novelty = torch.zeros_like(novelty_logits)
        empty_target = torch.zeros_like(empty_logit)
        count_calibration_values: list[torch.Tensor] = []
        anchor_index = torch.arange(points_per_lane, device=points.device, dtype=points.dtype)
        pixel_scale = self._pixel_scale_for(points)
        match_count = 0

        for batch_index, (query_index, target_index) in enumerate(indices):
            gt_points_b = gt_points[batch_index].to(device=points.device, dtype=points.dtype)
            gt_valid_b = gt_valid[batch_index].to(device=points.device, dtype=points.dtype)
            lane_count = int(gt_points_b.shape[0])
            count_calibration_values.append(
                F.smooth_l1_loss(
                    survival_logits[batch_index].sigmoid().sum(),
                    survival_logits.new_tensor(float(lane_count)),
                )
            )
            empty_target[batch_index] = float(lane_count == 0)
            if lane_count == 0:
                buckets["set_noop"].append(F.softplus(survival_logits[batch_index]).mean())
                continue

            distance_columns = []
            for lane_index in range(lane_count):
                valid = gt_valid_b[lane_index]
                error = torch.norm((points[batch_index] - gt_points_b[lane_index].unsqueeze(0)) * pixel_scale, dim=-1)
                distance_columns.append((error * valid.unsqueeze(0)).sum(dim=-1) / valid.sum().clamp_min(1.0))
            candidate_gt_distance = torch.stack(distance_columns, dim=-1)
            nearest_distance, nearest_gt = candidate_gt_distance.detach().min(dim=-1)
            close = nearest_distance <= float(self.lane_instance_duplicate_px)
            duplicate_target = (nearest_gt[:, None] == nearest_gt[None, :]) & close[:, None] & close[None, :]
            pair_mask = ~torch.eye(queries, device=points.device, dtype=torch.bool)
            buckets["duplicate"].append(
                F.binary_cross_entropy_with_logits(
                    duplicate_logits[batch_index][pair_mask],
                    duplicate_target[pair_mask].to(device=duplicate_logits.device, dtype=duplicate_logits.dtype),
                )
            )
            identity_cos = identity[batch_index] @ identity[batch_index].transpose(0, 1)
            buckets["identity"].append(
                F.binary_cross_entropy_with_logits(
                    identity_cos[pair_mask] / float(self.lane_instance_identity_temperature),
                    duplicate_target[pair_mask].to(device=identity_cos.device, dtype=identity_cos.dtype),
                )
            )

            if query_index.numel():
                match_count += int(query_index.numel())
                matched_valid = gt_valid_b[target_index]
                valid_bool = matched_valid > 0.5
                first = torch.where(valid_bool, anchor_index.view(1, -1), float(points_per_lane)).amin(dim=-1).long()
                last = torch.where(valid_bool, anchor_index.view(1, -1), -1.0).amax(dim=-1).long()
                contiguous_target = (
                    (anchor_index.view(1, -1) >= first.to(dtype=points.dtype).unsqueeze(-1))
                    & (anchor_index.view(1, -1) <= last.to(dtype=points.dtype).unsqueeze(-1))
                ).to(dtype=points.dtype)
                buckets["visibility"].append(
                    F.binary_cross_entropy_with_logits(
                        valid_logits[batch_index, query_index], matched_valid.to(dtype=valid_logits.dtype)
                    )
                )
                buckets["contiguity"].append(
                    F.binary_cross_entropy_with_logits(
                        valid_logits[batch_index, query_index], contiguous_target.to(dtype=valid_logits.dtype)
                    )
                )
                buckets["start"].append(F.cross_entropy(start_logits[batch_index, query_index], first))
                buckets["end"].append(F.cross_entropy(end_logits[batch_index, query_index], last))
                raw_start = (start_logits[batch_index, query_index].softmax(dim=-1) * anchor_index).sum(dim=-1)
                raw_end = (end_logits[batch_index, query_index].softmax(dim=-1) * anchor_index).sum(dim=-1)
                buckets["order"].append(F.relu(raw_start - raw_end).mean())
                target_span = (last - first).to(dtype=points.dtype)
                predicted_span = interval_end[batch_index, query_index] - interval_start[batch_index, query_index]
                buckets["span"].append(F.relu(torch.maximum(target_span, target_span.new_full(target_span.shape, self.lane_instance_min_span)) - predicted_span).mean())

                matched_points = points[batch_index, query_index]
                target_points = gt_points_b[target_index]
                point_error = torch.norm((matched_points - target_points) * pixel_scale, dim=-1)
                ape = (point_error * matched_valid).sum(dim=-1) / matched_valid.sum(dim=-1).clamp_min(1.0)
                quality_target = torch.exp(-ape.detach() / float(self.lane_instance_quality_tau_px)).clamp(0.0, 1.0)
                target_quality[batch_index, query_index] = quality_target.to(dtype=target_quality.dtype)
                target_survival[batch_index, query_index] = quality_target.clamp_min(0.25).to(dtype=target_survival.dtype)
                target_novelty[batch_index, query_index] = 1.0

                if query_index.numel() > 1:
                    matched_first = first
                    matched_bottom_x = target_points[torch.arange(target_points.shape[0], device=points.device), matched_first, 0]
                    topology_pair_mask = ~torch.eye(query_index.numel(), device=points.device, dtype=torch.bool)
                    left_target = matched_bottom_x[:, None] < matched_bottom_x[None, :]
                    matched_left = left_logits[batch_index][query_index][:, query_index]
                    matched_right = right_logits[batch_index][query_index][:, query_index]
                    topology_loss = F.binary_cross_entropy_with_logits(
                        matched_left[topology_pair_mask], left_target[topology_pair_mask].to(dtype=matched_left.dtype)
                    ) + F.binary_cross_entropy_with_logits(
                        matched_right[topology_pair_mask], (~left_target)[topology_pair_mask].to(dtype=matched_right.dtype)
                    )
                    buckets["topology"].append(0.5 * topology_loss)

                unmatched_mask = torch.ones(queries, device=points.device, dtype=torch.bool)
                unmatched_mask[query_index] = False
                if unmatched_mask.any():
                    weakest_match = survival_logits[batch_index, query_index].amin()
                    strongest_unmatched = survival_logits[batch_index, unmatched_mask].amax()
                    buckets["set_noop"].append(
                        F.relu(float(self.lane_instance_set_margin) - weakest_match + strongest_unmatched)
                    )

        buckets["empty"].append(F.binary_cross_entropy_with_logits(empty_logit, empty_target))
        buckets["quality"].append(F.binary_cross_entropy_with_logits(quality_logits, target_quality))
        buckets["survival"].append(F.binary_cross_entropy_with_logits(survival_logits, target_survival))
        buckets["novelty"].append(F.binary_cross_entropy_with_logits(novelty_logits, target_novelty))

        def mean_bucket(name: str) -> torch.Tensor:
            values = buckets[name]
            return torch.stack(values).mean() if values else zero

        visibility_loss = mean_bucket("visibility")
        start_loss = mean_bucket("start")
        end_loss = mean_bucket("end")
        order_loss = mean_bucket("order")
        contiguity_loss = mean_bucket("contiguity")
        positive_span_loss = mean_bucket("span")
        empty_loss = mean_bucket("empty")
        quality_loss = mean_bucket("quality")
        survival_loss = mean_bucket("survival")
        duplicate_loss = mean_bucket("duplicate")
        novelty_loss = mean_bucket("novelty")
        topology_loss = mean_bucket("topology")
        identity_loss = mean_bucket("identity")
        set_noop_loss = mean_bucket("set_noop")
        count_calibration_loss = torch.stack(count_calibration_values).mean() if count_calibration_values else zero
        total = (
            self.lane_instance_visibility_weight * visibility_loss
            + self.lane_instance_endpoint_weight * (start_loss + end_loss)
            + self.lane_instance_order_weight * order_loss
            + self.lane_instance_contiguity_weight * contiguity_loss
            + self.lane_instance_positive_span_weight * positive_span_loss
            + self.lane_instance_empty_weight * empty_loss
            + self.lane_instance_geometry_quality_weight * quality_loss
            + self.lane_instance_survival_weight * survival_loss
            + self.lane_instance_duplicate_weight * duplicate_loss
            + self.lane_instance_novelty_weight * novelty_loss
            + self.lane_instance_topology_weight * topology_loss
            + self.lane_instance_identity_weight * identity_loss
            + self.lane_instance_set_noop_weight * set_noop_loss
            + self.lane_instance_count_weight * count_calibration_loss
        )
        return (
            total,
            visibility_loss,
            start_loss,
            end_loss,
            order_loss,
            contiguity_loss,
            positive_span_loss,
            empty_loss,
            quality_loss,
            survival_loss,
            duplicate_loss,
            novelty_loss,
            topology_loss,
            identity_loss,
            set_noop_loss,
            count_calibration_loss,
            points.new_tensor(float(match_count)),
        )

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
        line_iou_loss = self._zero_like(pred_points)
        if self.line_iou_gain != 0.0:
            line_iou_loss = self.line_iou_loss(pred_points, gt_points, gt_valid, indices)
        (
            line_iou_valid_preserve_loss,
            line_iou_valid_preserve_count,
            line_iou_valid_preserve_anchor_count,
        ) = self.line_iou_valid_preserve_loss(pred_points, pred_valid_logits, gt_points, gt_valid, indices, gt_lanes)
        line_iou_exist_survival_loss = self._zero_like(pred_points)
        line_iou_exist_survival_count = self._zero_like(pred_points)
        if self.line_iou_exist_survival_gain != 0.0:
            (
                line_iou_exist_survival_loss,
                line_iou_exist_survival_count,
            ) = self.line_iou_exist_survival_loss(pred_points, pred_logits, gt_points, gt_valid, indices, gt_lanes)
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
        dense_instance_loss = self._zero_like(pred_points)
        dense_centerline_loss = self._zero_like(pred_points)
        dense_endpoint_loss = self._zero_like(pred_points)
        dense_endpoint_peak_loss = self._zero_like(pred_points)
        dense_endpoint_offset_loss = self._zero_like(pred_points)
        dense_embed_pull_loss = self._zero_like(pred_points)
        dense_embed_push_loss = self._zero_like(pred_points)
        dense_centerline_pos = self._zero_like(pred_points)
        dense_candidate_loss = self._zero_like(pred_points)
        dense_candidate_quality_loss = self._zero_like(pred_points)
        dense_candidate_replace_loss = self._zero_like(pred_points)
        dense_candidate_quality_pos = self._zero_like(pred_points)
        dense_candidate_replace_pos = self._zero_like(pred_points)
        residual_values = (self._zero_like(pred_points),) * 23
        lane_instance_values = (self._zero_like(pred_points),) * len(self.lane_instance_loss_names)
        if self.dense_instance_gain > 0.0:
            (
                dense_instance_loss,
                dense_centerline_loss,
                dense_endpoint_loss,
                dense_endpoint_peak_loss,
                dense_endpoint_offset_loss,
                dense_embed_pull_loss,
                dense_embed_push_loss,
                dense_centerline_pos,
            ) = self.dense_instance_loss(preds, gt_points, gt_valid)
        if self.dense_candidate_gain > 0.0:
            (
                dense_candidate_loss,
                dense_candidate_quality_loss,
                dense_candidate_replace_loss,
                dense_candidate_quality_pos,
                dense_candidate_replace_pos,
            ) = self.dense_candidate_loss(
                preds,
                pred_points,
                pred_logits,
                pred_valid_logits,
                gt_points,
                gt_valid,
            )
        if self.residual_proposal_gain > 0.0:
            residual_values = self.residual_proposal_loss(
                preds,
                pred_points,
                gt_points,
                gt_valid,
                gt_lanes,
                base_logits=pred_logits,
                base_valid_logits=pred_valid_logits,
            )
        if self.lane_instance_set_gain > 0.0:
            lane_instance_values = self.lane_instance_set_loss(preds, gt_points, gt_valid, indices)
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
            query_survival_rank_loss,
            query_survival_rank_pair_count,
            query_survival_rank_margin_mean,
        ) = self.query_survival_rank_loss(
            pred_points,
            pred_logits,
            preds.get("pred_base_logits"),
            indices,
            gt_lanes,
            gt_points,
            gt_valid,
        )
        (
            query_valid_survival_loss,
            query_valid_survival_count,
            query_valid_survival_anchor_count,
            query_valid_survival_dice_loss,
        ) = self.query_valid_survival_loss(
            pred_points,
            pred_valid_logits,
            preds.get("pred_base_valid_logits"),
            indices,
            gt_points,
            gt_valid,
            preds.get("pred_valid_interval_bounds"),
            preds.get("pred_valid_interval_base_bounds"),
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
        if self.line_iou_gain != 0.0:
            total = total + self.line_iou_gain * line_iou_loss
        if self.line_iou_valid_preserve_gain != 0.0:
            total = total + self.line_iou_valid_preserve_gain * line_iou_valid_preserve_loss
        if self.line_iou_exist_survival_gain != 0.0:
            total = total + self.line_iou_exist_survival_gain * line_iou_exist_survival_loss
        if self.boundary_pseudo_neg_gain != 0.0:
            total = total + self.boundary_pseudo_neg_gain * boundary_pseudo_neg_loss
        if self.query_count_ce_gain != 0.0:
            total = total + self.query_count_ce_gain * query_count_ce_loss
        if self.spurious_neg_gain != 0.0:
            total = total + self.spurious_neg_gain * self.spurious_neg_weight * spurious_neg_loss
        if self.query_survival_rank_gain != 0.0:
            total = total + self.query_survival_rank_gain * query_survival_rank_loss
        if self.query_valid_survival_gain != 0.0:
            total = total + self.query_valid_survival_gain * query_valid_survival_loss
        if self.dense_instance_gain != 0.0:
            total = total + self.dense_instance_gain * dense_instance_loss
        if self.dense_candidate_gain != 0.0:
            total = total + self.dense_candidate_gain * dense_candidate_loss
        if self.residual_proposal_gain != 0.0:
            total = total + self.residual_proposal_gain * residual_values[0]
        if self.lane_instance_set_gain != 0.0:
            total = total + self.lane_instance_set_gain * lane_instance_values[0]
        loss_items = torch.stack(
            (
                exist_loss.detach(),
                point_loss.detach(),
                point_valid_loss.detach(),
                smooth_loss.detach(),
                curve_loss.detach(),
                line_iou_loss.detach(),
                line_iou_valid_preserve_loss.detach(),
                line_iou_valid_preserve_count.detach(),
                line_iou_valid_preserve_anchor_count.detach(),
                line_iou_exist_survival_loss.detach(),
                line_iou_exist_survival_count.detach(),
                mask_loss.detach(),
                edge_loss.detach(),
                count_loss.detach(),
                count_under5_loss.detach(),
                count_boundary_loss.detach(),
                spurious_neg_loss.detach(),
                spurious_negative_count.detach(),
                query_survival_rank_loss.detach(),
                query_survival_rank_pair_count.detach(),
                query_survival_rank_margin_mean.detach(),
                query_valid_survival_loss.detach(),
                query_valid_survival_count.detach(),
                query_valid_survival_anchor_count.detach(),
                query_valid_survival_dice_loss.detach(),
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
                query_count_ce_loss.detach(),
                query_count_acc.detach(),
                query_count_pred_mean.detach(),
                dense_instance_loss.detach(),
                dense_centerline_loss.detach(),
                dense_endpoint_loss.detach(),
                dense_endpoint_peak_loss.detach(),
                dense_endpoint_offset_loss.detach(),
                dense_embed_pull_loss.detach(),
                dense_embed_push_loss.detach(),
                dense_centerline_pos.detach(),
                dense_candidate_loss.detach(),
                dense_candidate_quality_loss.detach(),
                dense_candidate_replace_loss.detach(),
                dense_candidate_quality_pos.detach(),
                dense_candidate_replace_pos.detach(),
                *(value.detach() for value in residual_values),
                *((value.detach() for value in lane_instance_values) if self.lane_instance_set_gain > 0.0 else ()),
            )
        )
        return total, loss_items

    def dense_candidate_loss(
        self,
        preds: dict[str, torch.Tensor],
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Train candidate quality and base-replacement probabilities."""
        quality_logits = preds.get("pred_dense_candidate_quality_logits")
        replace_logits = preds.get("pred_dense_candidate_replace_logits")
        if quality_logits is None or replace_logits is None:
            raise KeyError(
                "gcs_dense_candidate > 0 requires pred_dense_candidate_quality_logits and "
                "pred_dense_candidate_replace_logits."
            )
        if quality_logits.ndim != 4 or tuple(quality_logits.shape[1:2]) != (1,):
            raise ValueError(
                "pred_dense_candidate_quality_logits must have shape B x 1 x Hf x Wf, "
                f"got {tuple(quality_logits.shape)}."
            )
        if replace_logits.ndim != 4 or tuple(replace_logits.shape[1:2]) != (1,):
            raise ValueError(
                "pred_dense_candidate_replace_logits must have shape B x 1 x Hf x Wf, "
                f"got {tuple(replace_logits.shape)}."
            )
        if tuple(replace_logits.shape[-2:]) != tuple(quality_logits.shape[-2:]):
            raise ValueError("Dense candidate quality and replace maps must share Hf,Wf.")
        quality_target, replace_target = self._dense_candidate_targets(
            pred_points,
            pred_logits,
            pred_valid_logits,
            gt_points,
            gt_valid,
            tuple(quality_logits.shape[-2:]),
        )
        quality_target = quality_target.to(device=quality_logits.device, dtype=quality_logits.dtype)
        replace_target = replace_target.to(device=replace_logits.device, dtype=replace_logits.dtype)

        def sparse_bce(logits: torch.Tensor, target: torch.Tensor, max_weight: float) -> torch.Tensor:
            positive = target.sum()
            negative = target.numel() - positive
            pos_weight = (negative / positive.clamp_min(1.0)).clamp(min=1.0, max=float(max_weight))
            return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

        quality_loss = sparse_bce(
            quality_logits,
            quality_target,
            self.dense_candidate_quality_pos_weight_max,
        )
        replace_loss = sparse_bce(
            replace_logits,
            replace_target,
            self.dense_candidate_replace_pos_weight_max,
        )
        candidate_loss = (
            float(self.dense_candidate_quality_weight) * quality_loss
            + float(self.dense_candidate_replace_weight) * replace_loss
        )
        return (
            candidate_loss,
            quality_loss,
            replace_loss,
            quality_target.sum().detach(),
            replace_target.sum().detach(),
        )
