# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions for GCS-YOLO-Lane structured lane training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.gcs_count_contract import shortside_visible_decision
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
        "far_spur_loss",
        "far_spur_cand",
        "far_spur_neg",
        "far_spur_gt3",
        "far_spur_gt4",
        "far_spur_gt5",
        "far_spur_score",
        "far_spur_valid",
        "count_score_mean",
        "gt5_short_pos_count",
        "gt5_short_pos_anchor_count",
        "gt5_short_point_valid_loss",
        "cnt_bound_5under",
        "cnt_score",
        "short_raw_boost_count",
        "short_raw_boost_gt4",
        "short_raw_boost_gt5",
        "short_raw_missing",
        "shortside_selected_by_abs_count",
        "shortside_selected_by_median_count",
        "shortside_selected_total_count",
        "shortside_hungarian_rawmatch_candidate_count",
        "shortside_unmatched_raw_rescue_candidate_count",
        "shortside_rawmatch_candidate_total_count",
        "raw_rescue_candidate_count",
        "raw_rescue_final_count",
        "raw_rescue_conflict_count",
        "shortside_selected_hungarian_rawmatch",
        "shortside_selected_unmatched_rescue",
        "shortside_rescue_conflict",
        "shortside_missing_no_rawmatch",
        "shortside_nearest_was_duplicate_but_hungarian_boosted",
        "shortside_rescue_exist_loss",
        "shortside_rescue_point_loss",
        "shortside_rescue_valid_loss",
        "shortside_score_floor_loss",
        "shortside_ultra_point_loss",
        "shortside_ultra_valid_loss",
        "short_raw_boost_exist_target_mean",
        "short_raw_boost_exist_target_min",
        "short_raw_boost_exist_target_p25",
        "short_raw_boost_exist_target_p50",
        "short_raw_boost_exist_target_p75",
        "short_raw_boost_target_below_05_count",
        "short_raw_boost_target_below_07_count",
        "shortside_rawmatch_target_mean_before",
        "shortside_rawmatch_target_min_before",
        "shortside_rawmatch_target_p25_before",
        "shortside_rawmatch_target_p50_before",
        "shortside_rawmatch_target_below_05_count",
        "shortside_rawmatch_target_below_07_count",
        "shortside_target_floor_applied_count",
        "shortside_score_grad_down_count",
        "shortside_score_grad_up_count",
        "shortside_boost_only_mode",
        "shortside_protect_mode",
        "shortside_reliable_count",
        "shortside_reliable_selected_count",
        "shortside_ultra_seen_count",
        "shortside_ultra_enabled_count",
        "shortside_ultra_score_floor_count",
        "shortside_ultra_valid_count",
        "shortside_visible_lt2_skipped_count",
        "shortside_ultra_hungarian_base_boost_count",
        "shortside_ultra_unmatched_rescue_seen_count",
        "shortside_ultra_unmatched_rescue_enabled_count",
        "shortside_ultra_unmatched_rescue_skipped_count",
        "shortside_ultra_in_gt4_4to3_count",
        "shortside_ultra_in_gt5_5to4_count",
        "near_gt_ignored_count",
        "duplicate_like_rank_neg_count",
        "clear_far_rank_neg_count",
        "clear_far_boundary_count",
        "rank_neg_side_duplicate_like_count",
        "rank_neg_normal_duplicate_like_count",
        "rank_side_ambiguous_ignored_count",
        "rank_side_duplicate_rejected_better_than_true_count",
        "rank_pos_scope",
        "rank_pos_total",
        "rank_pos_hungarian_all",
        "rank_pos_shortside_matched",
        "rank_pos_shortside_rescue",
        "rank_pos_shortside_reliable",
        "rank_pos_shortside_ultra",
        "rank_pos_gt4gt5_matched",
        "rank_pos_all_matched",
        "rank_pos_hungarian_count",
        "rank_pos_unmatched_rescue_excluded_count",
        "rank_pos_unmatched_rescue_included_count",
        "rank_pos_conflict_excluded_count",
        "rank_neg_duplicate_like",
        "rank_neg_clear_far",
        "rank_neg_near_ignored",
        "base_exist_ignore_raw_rescue_count",
        "base_exist_ignore_rank_near_count",
        "base_exist_ignore_rank_side_count",
        "base_exist_ignore_farspur_near_count",
        "base_exist_ignore_farspur_side_count",
        "base_exist_ignore_duplicate_rank_only_count",
        "base_exist_ignore_near_count",
        "base_exist_ignore_side_ambiguous_count",
        "base_exist_ignore_duplicate_like_count",
        "base_ignore_farspur_near_count",
        "base_ignore_farspur_side_count",
        "base_ignore_duplicate_like_count",
        "farspur_aux_only_mode",
        "farspur_full_ignore_first_mode",
        "farspur_near_still_base_negative_count",
        "farspur_side_still_base_negative_count",
        "rank_pair_only_mode",
        "rank_full_duplicate_contract_mode",
        "duplicate_like_still_base_negative_count",
        "base_exist_negative_kept_clear_far_count",
        "base_exist_negative_kept_other_count",
        "legacy_spurious_active",
        "farspur_if_loss",
        "farspur_if_samples",
        "farspur_if_pos",
        "farspur_if_ignore",
        "farspur_if_clear",
        "farspur_if_near",
        "farspur_if_side",
        "rank_topk_loss",
        "rank_topk_samples",
        "rank_topk_pos",
        "rank_topk_neg",
        "rank_topk_clear",
        "rank_topk_dup",
        "rank_loss_noop_images",
        "rank_noop_because_no_pos",
        "rank_noop_because_no_neg",
        "rank_neg_duplicate_like_count",
        "rank_neg_clear_far_count",
        "rank_pos_shortside_reliable_count",
        "rank_pos_gt4gt5_matched_count",
        "rank_loss_noop_image_count",
        "rank_noop_because_no_pos_count",
        "rank_noop_because_no_neg_count",
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
        lambda_far_spurious_neg: float | None = None,
        lambda_rank_topk: float | None = None,
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
        far_spurious_min_gt_lanes: int | None = None,
        far_spurious_max_gt_lanes: int | None = None,
        far_spurious_min_valid: int | None = None,
        far_spurious_max_valid: int | None = None,
        far_spurious_far_px: float | None = None,
        far_spurious_protect_px: float | None = None,
        far_spurious_min_overlap: int | None = None,
        far_spurious_score_thr: float | None = None,
        far_spurious_gt3_weight: float | None = None,
        far_spurious_gt4_weight: float | None = None,
        far_spurious_gt5_weight: float | None = None,
        shortside_rawmatch_boost: float | None = None,
        shortside_rawmatch_dist_px: float | None = None,
        shortside_side_dist_px: float | None = None,
        shortside_max_valid_points: int | None = None,
        shortside_raw_rescue: bool | None = None,
        shortside_min_gt_lanes: int | None = None,
        shortside_min_valid_points: int | None = None,
        shortside_visible_max: int | None = None,
        shortside_use_median: bool | None = None,
        shortside_median_margin: float | None = None,
        shortside_rawmatch_px: float | None = None,
        shortside_side_rawmatch_px: float | None = None,
        shortside_exist_target_floor: float | None = None,
        shortside_score_floor_gain: float | None = None,
        shortside_score_target: float | None = None,
        shortside_ultra_enable: bool | None = None,
        shortside_ultra_min_valid_points: int | None = None,
        shortside_reliable_min_valid_points: int | None = None,
        shortside_ultra_score_floor_gain: float | None = None,
        shortside_ultra_point_gain: float | None = None,
        shortside_ultra_valid_gain: float | None = None,
        shortside_ultra_rank_pos: bool | None = None,
        shortside_rescue_exist_gain: float | None = None,
        shortside_rescue_point_gain: float | None = None,
        shortside_rescue_valid_gain: float | None = None,
        shortside_debug: bool | None = None,
        farspur_ignore_first: bool | None = None,
        farspur_weight: float | None = None,
        farspur_score_thr: float | None = None,
        farspur_near_dist_px: float | None = None,
        farspur_side_ignore_dist_px: float | None = None,
        farspur_clear_dist_px: float | None = None,
        farspur_min_valid_points: int | None = None,
        rank_margin: float | None = None,
        rank_max_negs: int | None = None,
        rank_gt_min_lanes: int | None = None,
        rank_focus_shortside: bool | None = None,
        rank_pos_scope: str | None = None,
        rank_include_unmatched_rescue_pos: bool | None = None,
        rank_dup_close_px: float | None = None,
        rank_dup_min_overlap: int | None = None,
        rank_side_duplicate_enable: bool | None = None,
        rank_side_dup_margin_px: float | None = None,
        rank_side_dup_min_overlap: int | None = None,
        rank_near_gt_ignore_px: float | None = None,
        rank_side_ignore_px: float | None = None,
        base_ignore_raw_rescue: bool | None = None,
        base_ignore_rank_near: bool | None = None,
        base_ignore_farspur_near: bool | None = None,
        base_ignore_duplicate_like: bool | None = None,
        rank_pair_reduction: str | None = None,
        allow_legacy_spurious_with_new_contract: bool | None = None,
        allow_gt3_count_contract_ablation: bool | None = None,
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
        self.far_spurious_neg_gain = float(
            lambda_far_spurious_neg
            if lambda_far_spurious_neg is not None
            else self._arg(args, "gcs_far_spurious_neg", 0.0)
        )
        self.farspur_weight = float(
            farspur_weight if farspur_weight is not None else self._arg(args, "gcs_farspur_weight", 0.0)
        )
        self.rank_topk_gain = float(
            lambda_rank_topk if lambda_rank_topk is not None else self._arg(args, "gcs_rank_topk_weight", 0.0)
        )
        self.shortside_rescue_exist_gain = float(
            shortside_rescue_exist_gain
            if shortside_rescue_exist_gain is not None
            else self._arg(args, "gcs_shortside_rescue_exist_gain", 0.0)
        )
        self.shortside_rescue_point_gain = float(
            shortside_rescue_point_gain
            if shortside_rescue_point_gain is not None
            else self._arg(args, "gcs_shortside_rescue_point_gain", 0.0)
        )
        self.shortside_rescue_valid_gain = float(
            shortside_rescue_valid_gain
            if shortside_rescue_valid_gain is not None
            else self._arg(args, "gcs_shortside_rescue_valid_gain", 0.0)
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
        self.far_spurious_min_gt_lanes = int(
            far_spurious_min_gt_lanes
            if far_spurious_min_gt_lanes is not None
            else self._arg(args, "gcs_far_spurious_min_gt_lanes", 3)
        )
        self.far_spurious_max_gt_lanes = int(
            far_spurious_max_gt_lanes
            if far_spurious_max_gt_lanes is not None
            else self._arg(args, "gcs_far_spurious_max_gt_lanes", 4)
        )
        self.far_spurious_min_valid = int(
            far_spurious_min_valid
            if far_spurious_min_valid is not None
            else self._arg(args, "gcs_far_spurious_min_valid", 2)
        )
        self.far_spurious_max_valid = int(
            far_spurious_max_valid
            if far_spurious_max_valid is not None
            else self._arg(args, "gcs_far_spurious_max_valid", 56)
        )
        self.far_spurious_far_px = float(
            far_spurious_far_px
            if far_spurious_far_px is not None
            else self._arg(args, "gcs_far_spurious_far_px", 40.0)
        )
        self.far_spurious_protect_px = float(
            far_spurious_protect_px
            if far_spurious_protect_px is not None
            else self._arg(args, "gcs_far_spurious_protect_px", 25.0)
        )
        self.far_spurious_min_overlap = int(
            far_spurious_min_overlap
            if far_spurious_min_overlap is not None
            else self._arg(args, "gcs_far_spurious_min_overlap", 3)
        )
        self.far_spurious_score_thr = float(
            far_spurious_score_thr
            if far_spurious_score_thr is not None
            else self._arg(args, "gcs_far_spurious_score_thr", 0.003)
        )
        self.far_spurious_gt3_weight = float(
            far_spurious_gt3_weight
            if far_spurious_gt3_weight is not None
            else self._arg(args, "gcs_far_spurious_gt3_weight", 1.0)
        )
        self.far_spurious_gt4_weight = float(
            far_spurious_gt4_weight
            if far_spurious_gt4_weight is not None
            else self._arg(args, "gcs_far_spurious_gt4_weight", 1.0)
        )
        self.far_spurious_gt5_weight = float(
            far_spurious_gt5_weight
            if far_spurious_gt5_weight is not None
            else self._arg(args, "gcs_far_spurious_gt5_weight", 0.0)
        )
        self.shortside_rawmatch_boost = float(
            shortside_rawmatch_boost
            if shortside_rawmatch_boost is not None
            else self._arg(args, "gcs_shortside_rawmatch_boost", 0.0)
        )
        shortside_rawmatch_explicit = (
            shortside_rawmatch_px
            if shortside_rawmatch_px is not None
            else self._cfg_value(
                args,
                "gcs_shortside_rawmatch_px",
                aliases=("gcs_shortside_rawmatch_dist_px",),
                default=None,
            )
        )
        if shortside_rawmatch_explicit is None:
            shortside_rawmatch_explicit = shortside_rawmatch_dist_px
        self.shortside_rawmatch_px = float(30.0 if shortside_rawmatch_explicit is None else shortside_rawmatch_explicit)

        shortside_side_rawmatch_explicit = (
            shortside_side_rawmatch_px
            if shortside_side_rawmatch_px is not None
            else self._cfg_value(
                args,
                "gcs_shortside_side_rawmatch_px",
                aliases=("gcs_shortside_side_dist_px",),
                default=None,
            )
        )
        if shortside_side_rawmatch_explicit is None:
            shortside_side_rawmatch_explicit = shortside_side_dist_px
        self.shortside_side_rawmatch_px = float(
            40.0 if shortside_side_rawmatch_explicit is None else shortside_side_rawmatch_explicit
        )
        self.shortside_rawmatch_dist_px = self.shortside_rawmatch_px
        self.shortside_side_dist_px = self.shortside_side_rawmatch_px
        self.shortside_exist_target_floor = float(
            shortside_exist_target_floor
            if shortside_exist_target_floor is not None
            else self._arg(args, "gcs_shortside_exist_target_floor", 0.0)
        )
        self.shortside_score_floor_gain = float(
            shortside_score_floor_gain
            if shortside_score_floor_gain is not None
            else self._arg(args, "gcs_shortside_score_floor_gain", 0.0)
        )
        self.shortside_score_target = float(
            shortside_score_target
            if shortside_score_target is not None
            else self._arg(args, "gcs_shortside_score_target", 0.8)
        )
        self.shortside_ultra_enable = self._bool_arg(
            shortside_ultra_enable
            if shortside_ultra_enable is not None
            else self._arg(args, "gcs_shortside_ultra_enable", False)
        )
        self.shortside_raw_rescue = self._bool_arg(
            shortside_raw_rescue
            if shortside_raw_rescue is not None
            else self._arg(args, "gcs_shortside_raw_rescue", False)
        )
        self.shortside_min_gt_lanes = int(
            shortside_min_gt_lanes
            if shortside_min_gt_lanes is not None
            else self._arg(args, "gcs_shortside_min_gt_lanes", 4)
        )
        self.shortside_min_valid_points = int(
            shortside_min_valid_points
            if shortside_min_valid_points is not None
            else self._arg(args, "gcs_shortside_min_valid_points", 6)
        )
        self.shortside_ultra_min_valid_points = int(
            shortside_ultra_min_valid_points
            if shortside_ultra_min_valid_points is not None
            else self._arg(args, "gcs_shortside_ultra_min_valid_points", 2)
        )
        reliable_min_value = (
            shortside_reliable_min_valid_points
            if shortside_reliable_min_valid_points is not None
            else self._arg(args, "gcs_shortside_reliable_min_valid_points", None)
        )
        if reliable_min_value is None:
            reliable_min_value = self.shortside_min_valid_points
        self.shortside_reliable_min_valid_points = int(reliable_min_value)
        self.shortside_ultra_score_floor_gain = float(
            shortside_ultra_score_floor_gain
            if shortside_ultra_score_floor_gain is not None
            else self._arg(args, "gcs_shortside_ultra_score_floor_gain", 0.0)
        )
        self.shortside_ultra_point_gain = float(
            shortside_ultra_point_gain
            if shortside_ultra_point_gain is not None
            else self._arg(args, "gcs_shortside_ultra_point_gain", 0.0)
        )
        self.shortside_ultra_valid_gain = float(
            shortside_ultra_valid_gain
            if shortside_ultra_valid_gain is not None
            else self._arg(args, "gcs_shortside_ultra_valid_gain", 0.0)
        )
        self.shortside_ultra_rank_pos = self._bool_arg(
            shortside_ultra_rank_pos
            if shortside_ultra_rank_pos is not None
            else self._arg(args, "gcs_shortside_ultra_rank_pos", False)
        )
        if self.shortside_ultra_min_valid_points < 1:
            raise ValueError(
                f"gcs_shortside_ultra_min_valid_points must be >= 1, got {self.shortside_ultra_min_valid_points}."
            )
        if self.shortside_reliable_min_valid_points < self.shortside_ultra_min_valid_points:
            raise ValueError(
                "gcs_shortside_reliable_min_valid_points must be >= gcs_shortside_ultra_min_valid_points "
                f"({self.shortside_reliable_min_valid_points} < {self.shortside_ultra_min_valid_points})."
            )
        shortside_visible_max_explicit = (
            shortside_visible_max
            if shortside_visible_max is not None
            else self._cfg_value(
                args,
                "gcs_shortside_visible_max",
                aliases=("gcs_shortside_max_valid_points",),
                default=None,
            )
        )
        if shortside_visible_max_explicit is None:
            shortside_visible_max_explicit = shortside_max_valid_points
        self.shortside_visible_max = int(42 if shortside_visible_max_explicit is None else shortside_visible_max_explicit)
        self.shortside_max_valid_points = self.shortside_visible_max
        self.shortside_use_median = self._bool_arg(
            shortside_use_median
            if shortside_use_median is not None
            else self._arg(args, "gcs_shortside_use_median", True)
        )
        self.shortside_median_margin = float(
            shortside_median_margin
            if shortside_median_margin is not None
            else self._arg(args, "gcs_shortside_median_margin", 0.0)
        )
        self.shortside_debug = self._bool_arg(
            shortside_debug if shortside_debug is not None else self._arg(args, "gcs_shortside_debug", False)
        )
        self.farspur_ignore_first = self._bool_arg(
            farspur_ignore_first if farspur_ignore_first is not None else self._arg(args, "gcs_farspur_ignore_first", False)
        )
        self.farspur_score_thr = float(
            farspur_score_thr if farspur_score_thr is not None else self._arg(args, "gcs_farspur_score_thr", 0.05)
        )
        self.farspur_near_dist_px = float(
            farspur_near_dist_px
            if farspur_near_dist_px is not None
            else self._arg(args, "gcs_farspur_near_dist_px", 40.0)
        )
        self.farspur_side_ignore_dist_px = float(
            farspur_side_ignore_dist_px
            if farspur_side_ignore_dist_px is not None
            else self._arg(args, "gcs_farspur_side_ignore_dist_px", 60.0)
        )
        self.farspur_clear_dist_px = float(
            farspur_clear_dist_px
            if farspur_clear_dist_px is not None
            else self._arg(args, "gcs_farspur_clear_dist_px", 80.0)
        )
        self.farspur_min_valid_points = int(
            farspur_min_valid_points
            if farspur_min_valid_points is not None
            else self._arg(args, "gcs_farspur_min_valid_points", 2)
        )
        self.rank_margin = float(
            rank_margin if rank_margin is not None else self._arg(args, "gcs_rank_margin", 0.05)
        )
        self.rank_max_negs = int(
            rank_max_negs if rank_max_negs is not None else self._arg(args, "gcs_rank_max_negs", 3)
        )
        self.rank_gt_min_lanes = int(
            rank_gt_min_lanes if rank_gt_min_lanes is not None else self._arg(args, "gcs_rank_gt_min_lanes", 4)
        )
        self.rank_focus_shortside = self._bool_arg(
            rank_focus_shortside
            if rank_focus_shortside is not None
            else self._arg(args, "gcs_rank_focus_shortside", True)
        )
        scope_value = rank_pos_scope if rank_pos_scope is not None else self._arg(args, "gcs_rank_pos_scope", None)
        if scope_value is None:
            scope_value = "shortside_reliable" if bool(self.rank_focus_shortside) else "all_matched"
        scope_value = str(scope_value).strip().lower()
        if not bool(self.rank_focus_shortside) and scope_value == "shortside_reliable":
            scope_value = "all_matched"
        valid_rank_scopes = {"shortside_reliable", "shortside_with_ultra", "gt4gt5_matched", "all_matched"}
        if scope_value not in valid_rank_scopes:
            raise ValueError(f"gcs_rank_pos_scope must be one of {sorted(valid_rank_scopes)}, got {scope_value!r}.")
        self.rank_pos_scope = scope_value
        self.rank_include_unmatched_rescue_pos = self._bool_arg(
            rank_include_unmatched_rescue_pos
            if rank_include_unmatched_rescue_pos is not None
            else self._arg(args, "gcs_rank_include_unmatched_rescue_pos", False)
        )
        self.rank_dup_close_px = float(
            rank_dup_close_px if rank_dup_close_px is not None else self._arg(args, "gcs_rank_dup_close_px", 30.0)
        )
        self.rank_dup_min_overlap = int(
            rank_dup_min_overlap
            if rank_dup_min_overlap is not None
            else self._arg(args, "gcs_rank_dup_min_overlap", 6)
        )
        self.rank_side_duplicate_enable = self._bool_arg(
            rank_side_duplicate_enable
            if rank_side_duplicate_enable is not None
            else self._arg(args, "gcs_rank_side_duplicate_enable", False)
        )
        self.rank_side_dup_margin_px = float(
            rank_side_dup_margin_px
            if rank_side_dup_margin_px is not None
            else self._arg(args, "gcs_rank_side_dup_margin_px", 5.0)
        )
        self.rank_side_dup_min_overlap = int(
            rank_side_dup_min_overlap
            if rank_side_dup_min_overlap is not None
            else self._arg(args, "gcs_rank_side_dup_min_overlap", 6)
        )
        self.rank_near_gt_ignore_px = float(
            rank_near_gt_ignore_px
            if rank_near_gt_ignore_px is not None
            else self._arg(args, "gcs_rank_near_gt_ignore_px", 40.0)
        )
        self.rank_side_ignore_px = float(
            rank_side_ignore_px if rank_side_ignore_px is not None else self._arg(args, "gcs_rank_side_ignore_px", 60.0)
        )
        self.base_ignore_raw_rescue = self._bool_arg(
            base_ignore_raw_rescue
            if base_ignore_raw_rescue is not None
            else self._arg(args, "gcs_base_ignore_raw_rescue", False)
        )
        self.base_ignore_rank_near = self._bool_arg(
            base_ignore_rank_near
            if base_ignore_rank_near is not None
            else self._arg(args, "gcs_base_ignore_rank_near", False)
        )
        self.base_ignore_farspur_near = self._bool_arg(
            base_ignore_farspur_near
            if base_ignore_farspur_near is not None
            else self._arg(args, "gcs_base_ignore_farspur_near", False)
        )
        self.base_ignore_duplicate_like = self._bool_arg(
            base_ignore_duplicate_like
            if base_ignore_duplicate_like is not None
            else self._arg(args, "gcs_base_ignore_duplicate_like", False)
        )
        reduction_value = (
            rank_pair_reduction
            if rank_pair_reduction is not None
            else self._arg(args, "gcs_rank_pair_reduction", "global_pair_mean")
        )
        self.rank_pair_reduction = str(reduction_value).strip().lower()
        valid_rank_reductions = {"global_pair_mean", "image_mean"}
        if self.rank_pair_reduction not in valid_rank_reductions:
            raise ValueError(
                "gcs_rank_pair_reduction must be one of "
                f"{sorted(valid_rank_reductions)}, got {self.rank_pair_reduction!r}."
            )
        self.allow_legacy_spurious_with_new_contract = self._bool_arg(
            allow_legacy_spurious_with_new_contract
            if allow_legacy_spurious_with_new_contract is not None
            else self._arg(args, "gcs_allow_legacy_spurious_with_new_contract", False)
        )
        self.allow_gt3_count_contract_ablation = self._bool_arg(
            allow_gt3_count_contract_ablation
            if allow_gt3_count_contract_ablation is not None
            else self._arg(args, "gcs_allow_gt3_count_contract_ablation", False)
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
        if self.far_spurious_neg_gain < 0.0:
            raise ValueError(f"gcs_far_spurious_neg must be >= 0, got {self.far_spurious_neg_gain}.")
        base_ignore_enabled = (
            bool(self.base_ignore_raw_rescue)
            or bool(self.base_ignore_rank_near)
            or bool(self.base_ignore_farspur_near)
            or bool(self.base_ignore_duplicate_like)
        )
        ignore_first_contract_enabled = (
            float(self.farspur_weight) > 0.0
            or float(self.rank_topk_gain) > 0.0
            or bool(self.shortside_raw_rescue)
            or base_ignore_enabled
        )
        min_gt_guard_contract_enabled = (
            ignore_first_contract_enabled
            or float(self.shortside_rawmatch_boost) > 0.0
            or float(self.shortside_score_floor_gain) > 0.0
            or bool(self.shortside_ultra_enable)
            or float(self.shortside_ultra_score_floor_gain) > 0.0
            or float(self.shortside_ultra_point_gain) > 0.0
            or float(self.shortside_ultra_valid_gain) > 0.0
        )
        self.ignore_first_contract_enabled = bool(ignore_first_contract_enabled)
        self.min_gt_guard_contract_enabled = bool(min_gt_guard_contract_enabled)
        if self.min_gt_guard_contract_enabled and (
            self.shortside_min_gt_lanes < 4 or self.rank_gt_min_lanes < 4
        ):
            message = (
                "GT4/GT5 count-contract losses require min_gt_lanes >= 4. "
                "Set gcs_allow_gt3_count_contract_ablation=True for explicit GT3 ablation."
            )
            if not self.allow_gt3_count_contract_ablation:
                raise ValueError(message)
            LOGGER.warning(message + " Proceeding with explicit GT3 ablation enabled.")
        if ignore_first_contract_enabled and self.spurious_neg_gain != 0.0:
            message = (
                "gcs_spurious_neg conflicts with the new ignore-first/ranking contract. "
                "Disable gcs_spurious_neg or explicitly set "
                "gcs_allow_legacy_spurious_with_new_contract=True."
            )
            if not self.allow_legacy_spurious_with_new_contract:
                raise ValueError(message)
            LOGGER.warning(
                message + " Proceeding only because gcs_allow_legacy_spurious_with_new_contract=True."
            )
        if ignore_first_contract_enabled and self.far_spurious_neg_gain != 0.0:
            message = (
                "gcs_far_spurious_neg conflicts with the new ignore-first/ranking contract. "
                "Disable gcs_far_spurious_neg or explicitly set "
                "gcs_allow_legacy_spurious_with_new_contract=True."
            )
            if not self.allow_legacy_spurious_with_new_contract:
                raise ValueError(message)
            LOGGER.warning(
                message + " Proceeding only because gcs_allow_legacy_spurious_with_new_contract=True."
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
        if self.far_spurious_min_gt_lanes < 0:
            raise ValueError(
                "gcs_far_spurious_min_gt_lanes must be >= 0, "
                f"got {self.far_spurious_min_gt_lanes}."
            )
        if self.far_spurious_max_gt_lanes < self.far_spurious_min_gt_lanes:
            raise ValueError(
                "gcs_far_spurious_max_gt_lanes must be >= gcs_far_spurious_min_gt_lanes "
                f"({self.far_spurious_max_gt_lanes} < {self.far_spurious_min_gt_lanes})."
            )
        if self.far_spurious_min_valid < 1:
            raise ValueError(f"gcs_far_spurious_min_valid must be >= 1, got {self.far_spurious_min_valid}.")
        if self.far_spurious_max_valid < self.far_spurious_min_valid:
            raise ValueError(
                "gcs_far_spurious_max_valid must be >= gcs_far_spurious_min_valid "
                f"({self.far_spurious_max_valid} < {self.far_spurious_min_valid})."
            )
        if self.far_spurious_far_px < 0.0:
            raise ValueError(f"gcs_far_spurious_far_px must be >= 0, got {self.far_spurious_far_px}.")
        if self.far_spurious_protect_px < 0.0:
            raise ValueError(f"gcs_far_spurious_protect_px must be >= 0, got {self.far_spurious_protect_px}.")
        if self.far_spurious_min_overlap < 1:
            raise ValueError(f"gcs_far_spurious_min_overlap must be >= 1, got {self.far_spurious_min_overlap}.")
        if not 0.0 <= self.far_spurious_score_thr <= 1.0:
            raise ValueError(
                f"gcs_far_spurious_score_thr must be in [0, 1], got {self.far_spurious_score_thr}."
            )
        if self.far_spurious_gt3_weight < 0.0:
            raise ValueError(f"gcs_far_spurious_gt3_weight must be >= 0, got {self.far_spurious_gt3_weight}.")
        if self.far_spurious_gt4_weight < 0.0:
            raise ValueError(f"gcs_far_spurious_gt4_weight must be >= 0, got {self.far_spurious_gt4_weight}.")
        if self.far_spurious_gt5_weight < 0.0:
            raise ValueError(f"gcs_far_spurious_gt5_weight must be >= 0, got {self.far_spurious_gt5_weight}.")
        if self.shortside_rescue_exist_gain < 0.0:
            raise ValueError(
                f"gcs_shortside_rescue_exist_gain must be >= 0, got {self.shortside_rescue_exist_gain}."
            )
        if self.shortside_rescue_point_gain < 0.0:
            raise ValueError(
                f"gcs_shortside_rescue_point_gain must be >= 0, got {self.shortside_rescue_point_gain}."
            )
        if self.shortside_rescue_valid_gain < 0.0:
            raise ValueError(
                f"gcs_shortside_rescue_valid_gain must be >= 0, got {self.shortside_rescue_valid_gain}."
            )
        if self.shortside_rawmatch_boost < 0.0:
            raise ValueError(
                f"gcs_shortside_rawmatch_boost must be >= 0, got {self.shortside_rawmatch_boost}."
            )
        if self.shortside_rawmatch_dist_px < 0.0:
            raise ValueError(
                f"gcs_shortside_rawmatch_dist_px must be >= 0, got {self.shortside_rawmatch_dist_px}."
            )
        if self.shortside_side_dist_px < 0.0:
            raise ValueError(f"gcs_shortside_side_dist_px must be >= 0, got {self.shortside_side_dist_px}.")
        if self.shortside_max_valid_points < 0:
            raise ValueError(
                f"gcs_shortside_max_valid_points must be >= 0, got {self.shortside_max_valid_points}."
            )
        if self.shortside_min_gt_lanes < 1:
            raise ValueError(f"gcs_shortside_min_gt_lanes must be >= 1, got {self.shortside_min_gt_lanes}.")
        if self.shortside_min_valid_points < 1:
            raise ValueError(
                f"gcs_shortside_min_valid_points must be >= 1, got {self.shortside_min_valid_points}."
            )
        if self.shortside_visible_max < self.shortside_min_valid_points:
            raise ValueError(
                "gcs_shortside_visible_max must be >= gcs_shortside_min_valid_points "
                f"({self.shortside_visible_max} < {self.shortside_min_valid_points})."
            )
        if self.farspur_weight < 0.0:
            raise ValueError(f"gcs_farspur_weight must be >= 0, got {self.farspur_weight}.")
        if self.farspur_weight > 0.0 and not self.farspur_ignore_first:
            raise ValueError("gcs_farspur_weight > 0 requires gcs_farspur_ignore_first=True.")
        if not 0.0 <= self.farspur_score_thr <= 1.0:
            raise ValueError(f"gcs_farspur_score_thr must be in [0, 1], got {self.farspur_score_thr}.")
        if self.farspur_near_dist_px < 0.0:
            raise ValueError(f"gcs_farspur_near_dist_px must be >= 0, got {self.farspur_near_dist_px}.")
        if self.farspur_side_ignore_dist_px < 0.0:
            raise ValueError(
                f"gcs_farspur_side_ignore_dist_px must be >= 0, got {self.farspur_side_ignore_dist_px}."
            )
        if self.farspur_clear_dist_px < self.farspur_near_dist_px:
            raise ValueError(
                "gcs_farspur_clear_dist_px must be >= gcs_farspur_near_dist_px "
                f"({self.farspur_clear_dist_px} < {self.farspur_near_dist_px})."
            )
        if self.farspur_min_valid_points < 1:
            raise ValueError(f"gcs_farspur_min_valid_points must be >= 1, got {self.farspur_min_valid_points}.")
        if self.rank_topk_gain < 0.0:
            raise ValueError(f"gcs_rank_topk_weight must be >= 0, got {self.rank_topk_gain}.")
        if self.rank_margin < 0.0:
            raise ValueError(f"gcs_rank_margin must be >= 0, got {self.rank_margin}.")
        if self.rank_max_negs < 1:
            raise ValueError(f"gcs_rank_max_negs must be >= 1, got {self.rank_max_negs}.")
        if self.rank_gt_min_lanes < 1:
            raise ValueError(f"gcs_rank_gt_min_lanes must be >= 1, got {self.rank_gt_min_lanes}.")
        if self.rank_dup_close_px < 0.0:
            raise ValueError(f"gcs_rank_dup_close_px must be >= 0, got {self.rank_dup_close_px}.")
        if self.rank_dup_min_overlap < 1:
            raise ValueError(f"gcs_rank_dup_min_overlap must be >= 1, got {self.rank_dup_min_overlap}.")
        if self.rank_side_dup_margin_px < 0.0:
            raise ValueError(f"gcs_rank_side_dup_margin_px must be >= 0, got {self.rank_side_dup_margin_px}.")
        if self.rank_side_dup_min_overlap < 1:
            raise ValueError(f"gcs_rank_side_dup_min_overlap must be >= 1, got {self.rank_side_dup_min_overlap}.")
        if self.rank_near_gt_ignore_px < 0.0:
            raise ValueError(f"gcs_rank_near_gt_ignore_px must be >= 0, got {self.rank_near_gt_ignore_px}.")
        if self.rank_side_ignore_px < 0.0:
            raise ValueError(f"gcs_rank_side_ignore_px must be >= 0, got {self.rank_side_ignore_px}.")
        if self.shortside_ultra_score_floor_gain < 0.0:
            raise ValueError(
                f"gcs_shortside_ultra_score_floor_gain must be >= 0, got {self.shortside_ultra_score_floor_gain}."
            )
        if self.shortside_ultra_point_gain < 0.0:
            raise ValueError(f"gcs_shortside_ultra_point_gain must be >= 0, got {self.shortside_ultra_point_gain}.")
        if self.shortside_ultra_valid_gain < 0.0:
            raise ValueError(f"gcs_shortside_ultra_valid_gain must be >= 0, got {self.shortside_ultra_valid_gain}.")
        if not 0.0 <= self.eval_point_valid_thr <= 1.0:
            raise ValueError(f"gcs_eval_point_valid_thr must be in [0, 1], got {self.eval_point_valid_thr}.")
        if self.gt5_short_visible_thr < 0:
            raise ValueError(f"gcs_gt5_short_visible_thr must be >= 0, got {self.gt5_short_visible_thr}.")
        if self.gt5_short_point_valid_weight < 0.0:
            raise ValueError(
                "gcs_gt5_short_point_valid_weight must be >= 0, "
                f"got {self.gt5_short_point_valid_weight}."
            )
        self._record_effective_count_contract(args)
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
    def _has_arg(args, name: str) -> bool:
        """Return whether a config object explicitly contains an attribute/key."""
        if isinstance(args, dict):
            return name in args and args.get(name) is not None
        return args is not None and hasattr(args, name) and getattr(args, name) is not None

    @classmethod
    def _cfg_value(cls, args, canonical: str, aliases: tuple[str, ...] = (), default=None):
        """Read config with canonical highest priority and legacy aliases as fallback only."""
        if cls._has_arg(args, canonical):
            value = cls._arg(args, canonical, default)
            for alias in aliases:
                if cls._has_arg(args, alias):
                    alias_value = cls._arg(args, alias, None)
                    try:
                        differs = float(alias_value) != float(value)
                    except (TypeError, ValueError):
                        differs = alias_value != value
                    if differs:
                        LOGGER.warning(
                            f"Both {canonical} and legacy alias {alias} are set with different values; "
                            f"using canonical {canonical}={value!r}."
                        )
            return value
        for alias in aliases:
            if cls._has_arg(args, alias):
                return cls._arg(args, alias, default)
        return default

    @staticmethod
    def _bool_arg(value) -> bool:
        """Parse bool-like config values without treating the string 'False' as true."""
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def effective_count_contract_config(self) -> dict[str, float | int | bool | str]:
        """Return the effective GT4/GT5 count-contract config for run metadata."""
        farspur_active = bool(self.farspur_ignore_first and self.farspur_weight > 0.0)
        rank_active = bool(self.rank_topk_gain > 0.0)
        return {
            "shortside_rawmatch_px": float(self.shortside_rawmatch_px),
            "shortside_side_rawmatch_px": float(self.shortside_side_rawmatch_px),
            "shortside_visible_max": int(self.shortside_visible_max),
            "shortside_min_valid_points": int(self.shortside_min_valid_points),
            "shortside_ultra_enable": bool(self.shortside_ultra_enable),
            "shortside_ultra_min_valid_points": int(self.shortside_ultra_min_valid_points),
            "shortside_reliable_min_valid_points": int(self.shortside_reliable_min_valid_points),
            "shortside_exist_target_floor": float(self.shortside_exist_target_floor),
            "shortside_score_floor_gain": float(self.shortside_score_floor_gain),
            "shortside_score_target": float(self.shortside_score_target),
            "shortside_ultra_score_floor_gain": float(self.shortside_ultra_score_floor_gain),
            "shortside_ultra_point_gain": float(self.shortside_ultra_point_gain),
            "shortside_ultra_valid_gain": float(self.shortside_ultra_valid_gain),
            "shortside_ultra_rank_pos": bool(self.shortside_ultra_rank_pos),
            "shortside_use_median": bool(self.shortside_use_median),
            "shortside_median_margin": float(self.shortside_median_margin),
            "rank_focus_shortside": bool(self.rank_focus_shortside),
            "rank_pos_scope": str(self.rank_pos_scope),
            "rank_pair_reduction": str(self.rank_pair_reduction),
            "rank_include_unmatched_rescue_pos": bool(self.rank_include_unmatched_rescue_pos),
            "rank_gt_min_lanes": int(self.rank_gt_min_lanes),
            "rank_side_duplicate_enable": bool(self.rank_side_duplicate_enable),
            "rank_side_dup_margin_px": float(self.rank_side_dup_margin_px),
            "rank_side_dup_min_overlap": int(self.rank_side_dup_min_overlap),
            "base_ignore_raw_rescue": bool(self.base_ignore_raw_rescue),
            "base_ignore_rank_near": bool(self.base_ignore_rank_near),
            "base_ignore_farspur_near": bool(self.base_ignore_farspur_near),
            "base_ignore_duplicate_like": bool(self.base_ignore_duplicate_like),
            "farspur_aux_only_mode": bool(farspur_active and not self.base_ignore_farspur_near),
            "farspur_full_ignore_first_mode": bool(farspur_active and self.base_ignore_farspur_near),
            "rank_pair_only_mode": bool(rank_active and not self.base_ignore_duplicate_like),
            "rank_full_duplicate_contract_mode": bool(rank_active and self.base_ignore_duplicate_like),
            "shortside_boost_only_mode": bool(
                self.shortside_rawmatch_boost > 0.0 and self.shortside_exist_target_floor <= 0.0
            ),
            "shortside_protect_mode": bool(
                self.shortside_rawmatch_boost > 0.0 and self.shortside_exist_target_floor > 0.0
            ),
            "allow_legacy_spurious_with_new_contract": bool(self.allow_legacy_spurious_with_new_contract),
            "allow_gt3_count_contract_ablation": bool(self.allow_gt3_count_contract_ablation),
            "legacy_spurious_active": bool(self.spurious_neg_gain != 0.0 or self.far_spurious_neg_gain != 0.0),
        }

    def _record_effective_count_contract(self, args) -> None:
        """Attach and log effective count-contract values for reproducibility."""
        config = self.effective_count_contract_config()
        if isinstance(args, dict):
            args["gcs_effective_count_contract"] = config
        elif args is not None:
            setattr(args, "gcs_effective_count_contract", config)
        LOGGER.info(
            "GCS count-contract effective config: "
            f"shortside_rawmatch_px={config['shortside_rawmatch_px']}, "
            f"shortside_side_rawmatch_px={config['shortside_side_rawmatch_px']}, "
            f"shortside_visible_max={config['shortside_visible_max']}, "
            f"shortside_min_valid_points={config['shortside_min_valid_points']}, "
            f"shortside_ultra_enable={str(config['shortside_ultra_enable']).lower()}, "
            f"shortside_ultra_min_valid_points={config['shortside_ultra_min_valid_points']}, "
            f"shortside_reliable_min_valid_points={config['shortside_reliable_min_valid_points']}, "
            f"shortside_exist_target_floor={config['shortside_exist_target_floor']}, "
            f"shortside_score_floor_gain={config['shortside_score_floor_gain']}, "
            f"shortside_score_target={config['shortside_score_target']}, "
            f"shortside_ultra_score_floor_gain={config['shortside_ultra_score_floor_gain']}, "
            f"shortside_ultra_point_gain={config['shortside_ultra_point_gain']}, "
            f"shortside_ultra_valid_gain={config['shortside_ultra_valid_gain']}, "
            f"shortside_ultra_rank_pos={str(config['shortside_ultra_rank_pos']).lower()}, "
            f"shortside_use_median={str(config['shortside_use_median']).lower()}, "
            f"shortside_median_margin={config['shortside_median_margin']}, "
            f"rank_focus_shortside={str(config['rank_focus_shortside']).lower()}, "
            f"rank_pos_scope={config['rank_pos_scope']}, "
            f"rank_pair_reduction={config['rank_pair_reduction']}, "
            f"rank_include_unmatched_rescue_pos={str(config['rank_include_unmatched_rescue_pos']).lower()}, "
            f"rank_gt_min_lanes={config['rank_gt_min_lanes']}, "
            f"rank_side_duplicate_enable={str(config['rank_side_duplicate_enable']).lower()}, "
            f"rank_side_dup_margin_px={config['rank_side_dup_margin_px']}, "
            f"rank_side_dup_min_overlap={config['rank_side_dup_min_overlap']}, "
            f"base_ignore_raw_rescue={str(config['base_ignore_raw_rescue']).lower()}, "
            f"base_ignore_rank_near={str(config['base_ignore_rank_near']).lower()}, "
            f"base_ignore_farspur_near={str(config['base_ignore_farspur_near']).lower()}, "
            f"base_ignore_duplicate_like={str(config['base_ignore_duplicate_like']).lower()}, "
            f"farspur_aux_only_mode={str(config['farspur_aux_only_mode']).lower()}, "
            f"farspur_full_ignore_first_mode={str(config['farspur_full_ignore_first_mode']).lower()}, "
            f"rank_pair_only_mode={str(config['rank_pair_only_mode']).lower()}, "
            f"rank_full_duplicate_contract_mode={str(config['rank_full_duplicate_contract_mode']).lower()}, "
            f"shortside_boost_only_mode={str(config['shortside_boost_only_mode']).lower()}, "
            f"shortside_protect_mode={str(config['shortside_protect_mode']).lower()}, "
            f"allow_gt3_count_contract_ablation={str(config['allow_gt3_count_contract_ablation']).lower()}, "
            f"legacy_spurious_active={str(config['legacy_spurious_active']).lower()}"
        )
        if config["farspur_aux_only_mode"]:
            LOGGER.warning(
                "This is farspur_aux_only, not full ignore-first; near/side queries are still base BCE negatives."
            )
        if config["rank_pair_only_mode"]:
            LOGGER.warning("This is rank_pair_only; duplicate-like queries are still base BCE negatives.")
        if self.rank_topk_gain > 0.0 and self.rank_pos_scope == "shortside_reliable":
            LOGGER.warning(
                "rank_topk is enabled with gcs_rank_pos_scope=shortside_reliable. "
                "This is NOT all GT4/GT5 matched positives. "
                "Use gcs_rank_pos_scope=gt4gt5_matched for full GT4/GT5 matched ranking."
            )

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

    def _build_exist_target(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        target_floor_mask: torch.Tensor | None = None,
        target_floor: float = 0.0,
    ) -> torch.Tensor:
        """Build quality-aware existence targets, with an optional mask-restricted floor."""
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
        if target_floor_mask is not None and float(target_floor) > 0.0:
            if target_floor_mask.shape != pred_logits.shape:
                raise ValueError(
                    "target_floor_mask must have shape B x Q matching pred_logits, "
                    f"got {tuple(target_floor_mask.shape)} vs {tuple(pred_logits.shape)}."
                )
            floor_mask = target_floor_mask.to(device=target.device, dtype=torch.bool)
            floor_value = target.new_tensor(float(target_floor)).clamp(min=0.0, max=1.0)
            target = torch.where(floor_mask, torch.maximum(target, floor_value), target)
        return target

    @staticmethod
    def _selected_target_stats(target: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        """Return compact distribution stats for selected existence targets."""
        zero = target.new_zeros(())
        if mask is None:
            return (zero, zero, zero, zero, zero, zero, zero)
        if mask.shape != target.shape:
            raise ValueError(f"mask must have shape {tuple(target.shape)}, got {tuple(mask.shape)}.")
        values = target[mask.to(device=target.device, dtype=torch.bool)]
        if values.numel() == 0:
            return (zero, zero, zero, zero, zero, zero, zero)
        values = values.detach().float()
        q = torch.quantile(values, values.new_tensor([0.25, 0.50, 0.75]))
        return (
            values.mean().to(device=target.device, dtype=target.dtype),
            values.min().to(device=target.device, dtype=target.dtype),
            q[0].to(device=target.device, dtype=target.dtype),
            q[1].to(device=target.device, dtype=target.dtype),
            q[2].to(device=target.device, dtype=target.dtype),
            (values < 0.5).sum().to(device=target.device, dtype=target.dtype),
            (values < 0.7).sum().to(device=target.device, dtype=target.dtype),
        )

    def exist_loss(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        query_weight: torch.Tensor | None = None,
        ignore_mask: torch.Tensor | None = None,
        target_floor_mask: torch.Tensor | None = None,
        target_floor: float = 0.0,
        return_target: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Quality-aware existence supervision for matched lane queries.

        Geometry alone is not enough for fixed-y lanes: a query can fit the GT x
        coordinates on visible anchors while marking many invalid anchors as
        visible, which renders a long false polyline. Fold point-visibility IoU
        into the existence target so such queries are not trained as confident
        positives until their visible segment is also correct.
        """
        target = self._build_exist_target(
            pred_logits,
            pred_points,
            pred_valid_logits,
            gt_points,
            gt_valid,
            indices,
            target_floor_mask=target_floor_mask,
            target_floor=target_floor,
        )
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
        if query_weight is not None:
            if query_weight.shape != pred_logits.shape:
                raise ValueError(
                    f"query_weight must have shape B x Q matching pred_logits, got {tuple(query_weight.shape)} "
                    f"vs {tuple(pred_logits.shape)}."
                )
            loss = loss * query_weight.to(device=loss.device, dtype=loss.dtype)
        if ignore_mask is not None:
            if ignore_mask.shape != pred_logits.shape:
                raise ValueError(
                    f"ignore_mask must have shape B x Q matching pred_logits, got {tuple(ignore_mask.shape)} "
                    f"vs {tuple(pred_logits.shape)}."
            )
            keep = ~ignore_mask.to(device=loss.device, dtype=torch.bool)
            result = loss[keep].mean() if bool(keep.any()) else self._zero_like(pred_logits)
            return (result, target) if return_target else result
        result = loss.mean()
        return (result, target) if return_target else result

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

    @staticmethod
    def _gt_side_ranks_by_lower_x(gt_points_b: torch.Tensor, gt_valid_b: torch.Tensor) -> dict[int, int]:
        """Return GT side ranks using lower-half mean x, falling back to any visible anchor."""
        valid_mask = gt_valid_b > 0.5
        lane_has_visible = valid_mask.any(dim=1)
        if not bool(lane_has_visible.any()):
            return {}
        k = int(gt_points_b.shape[1])
        # K56 fixed-y is bottom-to-top: index 0 is bottom y=710, index K-1 is top y=160.
        # Therefore lower/bottom half is indices < K//2.
        bottom_half = torch.arange(k, device=gt_points_b.device) < (k // 2)
        lower_valid = valid_mask & bottom_half.view(1, -1)
        use_valid = torch.where(lower_valid.any(dim=1, keepdim=True), lower_valid, valid_mask)
        lane_ids = lane_has_visible.nonzero(as_tuple=False).reshape(-1)
        valid = use_valid[lane_ids].to(dtype=gt_points_b.dtype)
        x_sum = (gt_points_b[lane_ids, :, 0] * valid).sum(dim=1)
        count = valid.sum(dim=1).clamp_min(1.0)
        order_x = x_sum / count
        ordered_lane_ids = lane_ids[torch.argsort(order_x, stable=True)]
        return {int(lane_id.item()): int(rank) for rank, lane_id in enumerate(ordered_lane_ids)}

    @staticmethod
    def _raw_lane_dx_px(pred_lane: torch.Tensor, gt_lane: torch.Tensor, gt_valid: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
        """Mean absolute x-distance in pixels on GT-visible fixed-y anchors."""
        valid = gt_valid > 0.5
        if not bool(valid.any()):
            return pred_lane.new_tensor(float("inf"))
        return ((pred_lane[valid, 0] - gt_lane[valid, 0]).abs() * x_scale).mean()

    def _eligible_shortside_gt_ids(
        self,
        gt_points_b: torch.Tensor,
        gt_valid_b: torch.Tensor,
        gt_count: int,
    ) -> tuple[dict[int, int], set[int], list[int], dict[str, int]]:
        """Return side GT ids eligible for shortside raw-match contracts."""
        side_ranks = self._gt_side_ranks_by_lower_x(gt_points_b.detach(), gt_valid_b.detach())
        if gt_count < int(self.shortside_min_gt_lanes) or not side_ranks:
            return side_ranks, set(), [], {"abs": 0, "median": 0, "total": 0, "reliable": 0, "ultra": 0, "lt2": 0}
        side_rank_values = {0, max(side_ranks.values())}
        visible_counts = (gt_valid_b.detach() > 0.5).sum(dim=1)
        visible_count_list = [int(v.item()) for v in visible_counts.reshape(-1)]
        eligible: list[int] = []
        stats = {"abs": 0, "median": 0, "total": 0, "reliable": 0, "ultra": 0, "lt2": 0}
        for gt_i, rank in side_ranks.items():
            if rank not in side_rank_values:
                continue
            visible = int(visible_counts[gt_i].item())
            decision = shortside_visible_decision(
                visible,
                visible_count_list,
                min_valid_points=int(self.shortside_min_valid_points),
                ultra_min_valid_points=int(self.shortside_ultra_min_valid_points),
                reliable_min_valid_points=int(self.shortside_reliable_min_valid_points),
                visible_max=int(self.shortside_visible_max),
                use_median=bool(self.shortside_use_median),
                median_margin=float(self.shortside_median_margin),
            )
            if decision.skipped_visible_lt2:
                stats["lt2"] += 1
            if decision.eligible:
                eligible.append(int(gt_i))
                stats["total"] += 1
                if decision.is_reliable:
                    stats["reliable"] += 1
                elif decision.is_ultra_short:
                    stats["ultra"] += 1
                if decision.selected_by_abs:
                    stats["abs"] += 1
                if decision.selected_by_median:
                    stats["median"] += 1
        return side_ranks, side_rank_values, eligible, stats

    def build_count_contract_masks(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Build shared GT4/GT5 count-contract masks for rescue, ignore, far-spur, and rank losses."""
        bsz, num_queries, num_points, _ = pred_points.shape
        device, dtype = pred_points.device, pred_points.dtype
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=dtype).reshape(-1)
        if gt_lanes.numel() != bsz:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={bsz}.")

        hungarian_pos = torch.zeros((bsz, num_queries), dtype=torch.bool, device=device)
        hungarian_gt_idx = torch.full((bsz, num_queries), -1, dtype=torch.long, device=device)
        shortside_matched_pos = torch.zeros_like(hungarian_pos)
        shortside_rescue_pos = torch.zeros_like(hungarian_pos)
        shortside_rescue_conflict_pos = torch.zeros_like(hungarian_pos)
        shortside_rescue_gt_idx = torch.full_like(hungarian_gt_idx, -1)
        raw_rescue_pos = torch.zeros_like(hungarian_pos)
        raw_rescue_gt_idx = torch.full_like(hungarian_gt_idx, -1)
        shortside_boost_pos = torch.zeros_like(hungarian_pos)
        shortside_hungarian_rawmatch_pos = torch.zeros_like(hungarian_pos)
        shortside_unmatched_rescue_pos = torch.zeros_like(hungarian_pos)
        shortside_ultra_hungarian_pos = torch.zeros_like(hungarian_pos)
        shortside_ultra_pos = torch.zeros_like(hungarian_pos)
        shortside_ultra_valid_pos = torch.zeros_like(hungarian_pos)
        shortside_target_floor_pos = torch.zeros_like(hungarian_pos)
        shortside_score_floor_pos = torch.zeros_like(hungarian_pos)
        shortside_ultra_score_floor_pos = torch.zeros_like(hungarian_pos)
        shortside_ultra_gt_idx = torch.full_like(hungarian_gt_idx, -1)
        shortside_selected_reason_code = torch.zeros_like(hungarian_gt_idx)
        shortside_selected_gt_idx = torch.full_like(hungarian_gt_idx, -1)
        near_gt_corridor = torch.zeros_like(hungarian_pos)
        ambiguous_side_region = torch.zeros_like(hungarian_pos)
        clear_far_spurious = torch.zeros_like(hungarian_pos)
        farspur_near_gt_corridor = torch.zeros_like(hungarian_pos)
        farspur_ambiguous_side_region = torch.zeros_like(hungarian_pos)
        farspur_clear_far_spurious = torch.zeros_like(hungarian_pos)
        duplicate_like = torch.zeros_like(hungarian_pos)
        normal_duplicate_like = torch.zeros_like(hungarian_pos)
        side_duplicate_like = torch.zeros_like(hungarian_pos)
        duplicate_like_including_raw_rescue = torch.zeros_like(hungarian_pos)
        rank_side_ambiguous_ignored = torch.zeros_like(hungarian_pos)

        valid_prob = (
            pred_valid_logits.detach().sigmoid()
            if pred_valid_logits is not None
            else pred_points.new_zeros((bsz, num_queries, num_points))
        )
        score_prob = pred_logits.detach().sigmoid()
        points = pred_points.detach()
        x_scale = self._spurious_x_scale(pred_points)
        valid_thr = float(self.eval_point_valid_thr)

        shortside_hungarian_rawmatch_candidate_count = 0
        shortside_unmatched_raw_rescue_candidate_count = 0
        shortside_rawmatch_candidate_total_count = 0
        raw_rescue_candidate_count = 0
        raw_rescue_final_count = 0
        raw_rescue_conflict_count = 0
        short_raw_missing_count = 0
        shortside_selected_hungarian_rawmatch_count = 0
        shortside_selected_unmatched_rescue_count = 0
        shortside_rescue_conflict_count = 0
        shortside_missing_no_rawmatch_count = 0
        shortside_nearest_was_duplicate_but_hungarian_boosted_count = 0
        clear_far_boundary_count = 0
        shortside_selected_by_abs_count = 0
        shortside_selected_by_median_count = 0
        shortside_selected_total_count = 0
        shortside_reliable_count = 0
        shortside_reliable_selected_count = 0
        shortside_ultra_seen_count = 0
        shortside_ultra_enabled_count = 0
        shortside_ultra_score_floor_count = 0
        shortside_ultra_valid_count = 0
        shortside_visible_lt2_skipped_count = 0
        shortside_ultra_hungarian_base_boost_count = 0
        shortside_ultra_unmatched_rescue_seen_count = 0
        shortside_ultra_unmatched_rescue_enabled_count = 0
        shortside_ultra_unmatched_rescue_skipped_count = 0
        shortside_ultra_in_gt4_4to3_count = 0
        shortside_ultra_in_gt5_5to4_count = 0
        rank_side_duplicate_rejected_better_than_true_count = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            src_idx = src_idx.to(device=device, dtype=torch.long)
            tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            gt_count = int(round(float(gt_lanes[b].detach().item())))
            if src_idx.numel():
                hungarian_pos[b, src_idx] = True
                hungarian_gt_idx[b, src_idx] = tgt_idx

            gt_points_b = gt_points[b].detach().to(device=device, dtype=dtype)
            gt_valid_b = gt_valid[b].detach().to(device=device)
            if gt_points_b.ndim != 3 or gt_points_b.shape[-1] != 2:
                raise ValueError(f"Each GT lane tensor must have shape N x K x 2, got {tuple(gt_points_b.shape)}.")
            if gt_valid_b.shape != gt_points_b.shape[:2]:
                raise ValueError(
                    f"GT valid mask must match GT lane first two dims, got {tuple(gt_valid_b.shape)} vs {tuple(gt_points_b.shape[:2])}."
                )

            side_ranks, side_rank_values, eligible_shortside, shortside_stats = self._eligible_shortside_gt_ids(
                gt_points_b, gt_valid_b, gt_count
            )
            shortside_selected_by_abs_count += int(shortside_stats["abs"])
            shortside_selected_by_median_count += int(shortside_stats["median"])
            shortside_selected_total_count += int(shortside_stats["total"])
            shortside_reliable_count += int(shortside_stats["reliable"])
            shortside_ultra_seen_count += int(shortside_stats["ultra"])
            shortside_visible_lt2_skipped_count += int(shortside_stats["lt2"])
            eligible_shortside_set = set(eligible_shortside)
            visible_counts_b = (gt_valid_b.detach() > 0.5).sum(dim=1)
            ultra_shortside_set = {
                int(gt_i)
                for gt_i in eligible_shortside_set
                if int(visible_counts_b[int(gt_i)].item()) < int(self.shortside_reliable_min_valid_points)
            }
            reliable_shortside_set = eligible_shortside_set.difference(ultra_shortside_set)
            pred_count_rounded = int(round(float(score_prob[b].sum().detach().item())))
            visible_2_5_count = sum(1 for gt_i in ultra_shortside_set if int(visible_counts_b[int(gt_i)].item()) <= 5)
            if gt_count == 4 and pred_count_rounded == 3:
                shortside_ultra_in_gt4_4to3_count += visible_2_5_count
            elif gt_count >= 5 and pred_count_rounded == 4:
                shortside_ultra_in_gt5_5to4_count += visible_2_5_count
            if src_idx.numel() and gt_count >= int(self.rank_gt_min_lanes):
                for src_t, tgt_t in zip(src_idx, tgt_idx):
                    src = int(src_t.item())
                    gt_i = int(tgt_t.item())
                    if gt_i in reliable_shortside_set:
                        shortside_matched_pos[b, src] = True
                    elif gt_i in ultra_shortside_set:
                        shortside_ultra_hungarian_pos[b, src] = True
            shortside_candidates_by_q: dict[int, list[tuple[float, int, str]]] = {}
            shortside_ultra_candidates_by_q: dict[int, list[tuple[float, int, str]]] = {}
            for gt_i in eligible_shortside:
                is_reliable_shortside = int(gt_i) in reliable_shortside_set
                rank = side_ranks.get(gt_i)
                threshold_px = float(
                    self.shortside_side_rawmatch_px if rank in side_rank_values else self.shortside_rawmatch_px
                )
                distances = [
                    self._raw_lane_dx_px(points[b, q], gt_points_b[gt_i], gt_valid_b[gt_i], x_scale)
                    for q in range(num_queries)
                ]
                if not distances:
                    short_raw_missing_count += 1
                    shortside_missing_no_rawmatch_count += 1
                    continue
                dist_t = torch.stack(distances)
                finite_rawmatch = torch.isfinite(dist_t) & (dist_t <= threshold_px)
                if not bool(finite_rawmatch.any()):
                    short_raw_missing_count += 1
                    shortside_missing_no_rawmatch_count += 1
                    continue
                nearest_q = int(torch.argmin(dist_t).item())
                nearest_dist = dist_t[nearest_q]

                selected_q = -1
                selected_dist = pred_logits.new_tensor(float("inf"))
                selected_reason = ""
                same_gt_q = src_idx[tgt_idx == int(gt_i)] if src_idx.numel() else src_idx
                if same_gt_q.numel():
                    same_dist = dist_t[same_gt_q]
                    same_order = torch.argsort(same_dist)
                    q_t = same_gt_q[same_order[0]]
                    dist = same_dist[same_order[0]]
                    if torch.isfinite(dist) and float(dist.item()) <= threshold_px:
                        selected_q = int(q_t.item())
                        selected_dist = dist
                        selected_reason = "hungarian_rawmatch"
                        nearest_matched_gt = int(hungarian_gt_idx[b, nearest_q].item())
                        if (
                            nearest_q != selected_q
                            and nearest_matched_gt < 0
                            and torch.isfinite(nearest_dist)
                            and float(nearest_dist.item()) < float(selected_dist.item())
                            and float(nearest_dist.item()) <= threshold_px
                        ):
                            shortside_nearest_was_duplicate_but_hungarian_boosted_count += 1

                if selected_q < 0:
                    unmatched = ~hungarian_pos[b]
                    unmatched_candidates = finite_rawmatch & unmatched
                    if bool(unmatched_candidates.any()):
                        q_candidates = unmatched_candidates.nonzero(as_tuple=False).reshape(-1)
                        cand_dist = dist_t[q_candidates]
                        best_local = int(torch.argmin(cand_dist).item())
                        selected_q = int(q_candidates[best_local].item())
                        selected_dist = cand_dist[best_local]
                        selected_reason = "unmatched_raw_rescue"

                if selected_q < 0 or not selected_reason:
                    short_raw_missing_count += 1
                    shortside_missing_no_rawmatch_count += 1
                    continue
                target_candidates = shortside_candidates_by_q if is_reliable_shortside else shortside_ultra_candidates_by_q
                if is_reliable_shortside:
                    shortside_rawmatch_candidate_total_count += 1
                    if selected_reason == "hungarian_rawmatch":
                        shortside_hungarian_rawmatch_candidate_count += 1
                    elif selected_reason == "unmatched_raw_rescue":
                        shortside_unmatched_raw_rescue_candidate_count += 1
                        raw_rescue_candidate_count += 1
                target_candidates.setdefault(selected_q, []).append((float(selected_dist.item()), int(gt_i), selected_reason))

            for best_q, candidates in shortside_candidates_by_q.items():
                candidates.sort(key=lambda item: item[0])
                if len(candidates) > 1:
                    raw_rescue_conflict_count += len(candidates) - 1
                    shortside_rescue_conflict_count += len(candidates) - 1
                    shortside_rescue_conflict_pos[b, best_q] = True
                _, gt_i, selected_reason = candidates[0]
                if selected_reason == "hungarian_rawmatch":
                    shortside_boost_pos[b, best_q] = True
                    shortside_hungarian_rawmatch_pos[b, best_q] = True
                    shortside_target_floor_pos[b, best_q] = True
                    shortside_score_floor_pos[b, best_q] = True
                    shortside_selected_reason_code[b, best_q] = 1
                    shortside_selected_gt_idx[b, best_q] = int(gt_i)
                    shortside_selected_hungarian_rawmatch_count += 1
                    shortside_reliable_selected_count += 1
                elif selected_reason == "unmatched_raw_rescue":
                    shortside_unmatched_rescue_pos[b, best_q] = True
                    shortside_selected_reason_code[b, best_q] = 2
                    shortside_selected_gt_idx[b, best_q] = int(gt_i)
                    shortside_selected_unmatched_rescue_count += 1
                    shortside_reliable_selected_count += 1
                    if self.shortside_raw_rescue:
                        shortside_rescue_pos[b, best_q] = True
                        shortside_rescue_gt_idx[b, best_q] = int(gt_i)
                        raw_rescue_pos[b, best_q] = True
                        raw_rescue_gt_idx[b, best_q] = int(gt_i)
                        shortside_target_floor_pos[b, best_q] = True
                        shortside_score_floor_pos[b, best_q] = True
                        raw_rescue_final_count += 1
                    else:
                        short_raw_missing_count += 1
                else:
                    short_raw_missing_count += 1
                    shortside_missing_no_rawmatch_count += 1

            for best_q, candidates in shortside_ultra_candidates_by_q.items():
                if bool(shortside_target_floor_pos[b, best_q]):
                    continue
                candidates.sort(key=lambda item: item[0])
                if len(candidates) > 1:
                    raw_rescue_conflict_count += len(candidates) - 1
                    shortside_rescue_conflict_count += len(candidates) - 1
                    shortside_rescue_conflict_pos[b, best_q] = True
                _, gt_i, selected_reason = candidates[0]
                if selected_reason == "hungarian_rawmatch":
                    shortside_boost_pos[b, best_q] = True
                    shortside_hungarian_rawmatch_pos[b, best_q] = True
                    shortside_target_floor_pos[b, best_q] = True
                    shortside_score_floor_pos[b, best_q] = True
                    shortside_ultra_hungarian_pos[b, best_q] = True
                    shortside_selected_reason_code[b, best_q] = 1
                    shortside_selected_gt_idx[b, best_q] = int(gt_i)
                    shortside_selected_hungarian_rawmatch_count += 1
                    shortside_ultra_hungarian_base_boost_count += 1
                    continue
                shortside_ultra_unmatched_rescue_seen_count += 1
                if bool(self.shortside_ultra_enable):
                    shortside_ultra_pos[b, best_q] = True
                    shortside_ultra_gt_idx[b, best_q] = int(gt_i)
                    shortside_selected_reason_code[b, best_q] = 2
                    shortside_selected_gt_idx[b, best_q] = int(gt_i)
                    shortside_ultra_enabled_count += 1
                    shortside_ultra_unmatched_rescue_enabled_count += 1
                    if float(self.shortside_ultra_score_floor_gain) > 0.0:
                        shortside_ultra_score_floor_pos[b, best_q] = True
                        shortside_ultra_score_floor_count += 1
                    if float(self.shortside_ultra_valid_gain) > 0.0:
                        shortside_ultra_valid_pos[b, best_q] = True
                        shortside_ultra_valid_count += 1
                else:
                    shortside_ultra_unmatched_rescue_skipped_count += 1

            unmatched_idx = torch.arange(num_queries, device=device)[~hungarian_pos[b]]
            side_ids = {gt_i for gt_i, rank in side_ranks.items() if rank in side_rank_values}
            rank_bottom_thr = float(self.rank_near_gt_ignore_px) + 10.0
            farspur_bottom_thr = float(self.farspur_near_dist_px) + 10.0
            for uq_t in unmatched_idx:
                uq = int(uq_t.item())
                uq_valid = valid_prob[b, uq] >= valid_thr
                valid_count = self._longest_true_run(uq_valid)
                min_dist = float("inf")
                min_bottom_dx = float("inf")
                min_side_dist = float("inf")
                for gt_i in range(int(gt_points_b.shape[0])):
                    lane_valid = gt_valid_b[gt_i] > 0.5
                    if not bool(lane_valid.any()):
                        continue
                    raw_dx = (points[b, uq, lane_valid, 0] - gt_points_b[gt_i, lane_valid, 0]).abs() * x_scale
                    raw_mean = float(raw_dx.mean().item()) if raw_dx.numel() else float("inf")
                    common = uq_valid & lane_valid
                    if bool(common.any()):
                        common_dx = (points[b, uq, common, 0] - gt_points_b[gt_i, common, 0]).abs() * x_scale
                        common_mean = float(common_dx.mean().item())
                    else:
                        common_mean = float("inf")
                    lane_dist = min(raw_mean, common_mean)
                    min_dist = min(min_dist, lane_dist)
                    bottom_idx = int(lane_valid.nonzero(as_tuple=False).reshape(-1)[0].item())
                    bottom_dx = float(((points[b, uq, bottom_idx, 0] - gt_points_b[gt_i, bottom_idx, 0]).abs() * x_scale).item())
                    min_bottom_dx = min(min_bottom_dx, bottom_dx)
                    if gt_i in side_ids:
                        min_side_dist = min(min_side_dist, lane_dist, bottom_dx)

                rank_near = min_dist <= float(self.rank_near_gt_ignore_px) or min_bottom_dx <= rank_bottom_thr
                rank_side_ambiguous = min_side_dist <= float(self.rank_side_ignore_px)
                farspur_near = min_dist <= float(self.farspur_near_dist_px) or min_bottom_dx <= farspur_bottom_thr
                farspur_side_ambiguous = min_side_dist <= float(self.farspur_side_ignore_dist_px)
                near_gt_corridor[b, uq] = bool(rank_near)
                ambiguous_side_region[b, uq] = bool(rank_side_ambiguous)
                farspur_near_gt_corridor[b, uq] = bool(farspur_near)
                farspur_ambiguous_side_region[b, uq] = bool(farspur_side_ambiguous)

                score_ok = float(score_prob[b, uq].item()) >= float(self.farspur_score_thr)
                valid_ok = int(valid_count) >= int(self.farspur_min_valid_points)
                clear_dist_px = float(self.farspur_clear_dist_px)
                at_clear_far_boundary = abs(min_dist - clear_dist_px) <= max(1e-6, abs(clear_dist_px) * 1e-6)
                if (
                    score_ok
                    and valid_ok
                    and at_clear_far_boundary
                    and not bool(raw_rescue_pos[b, uq])
                    and (
                        (min_bottom_dx > rank_bottom_thr and not rank_near and not rank_side_ambiguous)
                        or (min_bottom_dx > farspur_bottom_thr and not farspur_near and not farspur_side_ambiguous)
                    )
                ):
                    clear_far_boundary_count += 1
                clear_far_spurious[b, uq] = bool(
                    score_ok
                    and valid_ok
                    and min_dist > clear_dist_px
                    and min_bottom_dx > rank_bottom_thr
                    and not rank_near
                    and not rank_side_ambiguous
                    and not bool(raw_rescue_pos[b, uq])
                )
                farspur_clear_far_spurious[b, uq] = bool(
                    score_ok
                    and valid_ok
                    and min_dist > clear_dist_px
                    and min_bottom_dx > farspur_bottom_thr
                    and not farspur_near
                    and not farspur_side_ambiguous
                    and not bool(raw_rescue_pos[b, uq])
                )

                normal_close_to_matched = False
                normal_sufficient_overlap = False
                side_close_to_matched = False
                side_sufficient_overlap = False
                side_duplicate_is_worse = False
                side_duplicate_rejected = False
                for mq_t in src_idx:
                    mq = int(mq_t.item())
                    mq_valid = valid_prob[b, mq] >= valid_thr
                    overlap = uq_valid & mq_valid
                    overlap_count = int(overlap.sum().item())
                    if overlap_count < min(int(self.rank_dup_min_overlap), int(self.rank_side_dup_min_overlap)):
                        continue
                    dx_px = (points[b, uq, overlap, 0] - points[b, mq, overlap, 0]).abs() * x_scale
                    close_enough = float(dx_px.mean().item()) <= float(self.rank_dup_close_px)
                    if overlap_count >= int(self.rank_dup_min_overlap) and close_enough:
                        normal_sufficient_overlap = True
                        normal_close_to_matched = True
                    if rank_side_ambiguous and overlap_count >= int(self.rank_side_dup_min_overlap) and close_enough:
                        side_sufficient_overlap = True
                        side_close_to_matched = True
                        matched_gt = int(hungarian_gt_idx[b, mq].item())
                        if matched_gt not in side_ids:
                            continue
                        if self.rank_pos_scope == "shortside_reliable":
                            has_true_pos = bool(shortside_hungarian_rawmatch_pos[b, mq])
                        elif self.rank_pos_scope == "shortside_with_ultra":
                            has_true_pos = bool(
                                shortside_hungarian_rawmatch_pos[b, mq]
                                or (bool(self.shortside_ultra_rank_pos) and shortside_ultra_hungarian_pos[b, mq])
                            )
                        else:
                            has_true_pos = bool(hungarian_pos[b, mq])
                        if not has_true_pos:
                            continue
                        dup_dist = self._raw_lane_dx_px(points[b, uq], gt_points_b[matched_gt], gt_valid_b[matched_gt], x_scale)
                        true_dist = self._raw_lane_dx_px(points[b, mq], gt_points_b[matched_gt], gt_valid_b[matched_gt], x_scale)
                        if float(dup_dist.item()) >= float(true_dist.item()) + float(self.rank_side_dup_margin_px):
                            side_duplicate_is_worse = True
                            break
                        side_duplicate_rejected = True
                normal_is_duplicate = bool(
                    normal_close_to_matched
                    and normal_sufficient_overlap
                    and not bool(raw_rescue_pos[b, uq])
                    and not rank_side_ambiguous
                )
                raw_rescue_duplicate_like = bool(
                    normal_close_to_matched
                    and normal_sufficient_overlap
                    and not rank_side_ambiguous
                )
                raw_rescue_side_duplicate_like = bool(
                    side_close_to_matched
                    and side_sufficient_overlap
                    and rank_side_ambiguous
                )
                side_is_duplicate = bool(
                    self.rank_side_duplicate_enable
                    and side_close_to_matched
                    and side_sufficient_overlap
                    and rank_side_ambiguous
                    and not bool(raw_rescue_pos[b, uq])
                    and side_duplicate_is_worse
                )
                normal_duplicate_like[b, uq] = normal_is_duplicate
                side_duplicate_like[b, uq] = side_is_duplicate
                duplicate_like[b, uq] = normal_is_duplicate or side_is_duplicate
                duplicate_like_including_raw_rescue[b, uq] = raw_rescue_duplicate_like or raw_rescue_side_duplicate_like
                if rank_side_ambiguous and not side_is_duplicate and not bool(raw_rescue_pos[b, uq]):
                    rank_side_ambiguous_ignored[b, uq] = True
                if side_duplicate_rejected and not side_is_duplicate:
                    rank_side_duplicate_rejected_better_than_true_count += 1

        raw_rescue_unmatched = raw_rescue_pos & ~hungarian_pos
        rank_pos_conflict_excluded = raw_rescue_unmatched & (
            shortside_rescue_conflict_pos
            | duplicate_like_including_raw_rescue
            | clear_far_spurious
        )
        unmatched_rescue_rank_candidate = raw_rescue_unmatched & ~rank_pos_conflict_excluded
        unmatched_rescue_rank_included = (
            unmatched_rescue_rank_candidate
            if bool(self.rank_include_unmatched_rescue_pos)
            else torch.zeros_like(hungarian_pos)
        )
        rank_pos_unmatched_rescue_excluded = raw_rescue_unmatched & ~unmatched_rescue_rank_included
        shortside_reliable_rank_pos = shortside_matched_pos | unmatched_rescue_rank_included
        shortside_ultra_rank_pos = (
            shortside_ultra_hungarian_pos if bool(self.shortside_ultra_rank_pos) else torch.zeros_like(hungarian_pos)
        )
        gt4gt5_mask = (gt_lanes >= 4).view(-1, 1)
        gt4gt5_matched_pos = hungarian_pos & gt4gt5_mask
        all_matched_pos = hungarian_pos
        if self.rank_pos_scope == "shortside_reliable":
            rank_pos = shortside_reliable_rank_pos
        elif self.rank_pos_scope == "shortside_with_ultra":
            rank_pos = shortside_reliable_rank_pos | shortside_ultra_rank_pos
        elif self.rank_pos_scope == "gt4gt5_matched":
            rank_pos = gt4gt5_matched_pos
        elif self.rank_pos_scope == "all_matched":
            rank_pos = all_matched_pos
        else:
            raise AssertionError(f"Unexpected gcs_rank_pos_scope={self.rank_pos_scope!r}.")
        rank_pos_shortside_rescue_only = unmatched_rescue_rank_included
        rank_neg = duplicate_like | clear_far_spurious
        unmatched = ~hungarian_pos
        farspur_active = bool(self.farspur_ignore_first and float(self.farspur_weight) > 0.0)
        rank_effective = bool(float(self.rank_topk_gain) > 0.0)
        clear_far_final = (
            farspur_clear_far_spurious
            & ~near_gt_corridor
            & ~ambiguous_side_region
            & ~farspur_near_gt_corridor
            & ~farspur_ambiguous_side_region
            & ~raw_rescue_unmatched
        )
        rank_base_scope = (gt_lanes >= float(self.rank_gt_min_lanes)).view(-1, 1)
        base_ignore_raw_rescue_mask = raw_rescue_unmatched if bool(self.base_ignore_raw_rescue) else torch.zeros_like(hungarian_pos)
        base_ignore_rank_near_mask = (
            (near_gt_corridor | ambiguous_side_region) & rank_base_scope
            if bool(self.base_ignore_rank_near)
            else torch.zeros_like(hungarian_pos)
        )
        base_ignore_duplicate_like_mask = (
            duplicate_like & rank_base_scope
            if rank_effective and bool(self.base_ignore_duplicate_like)
            else torch.zeros_like(hungarian_pos)
        )
        base_ignore_farspur_near_mask = (
            farspur_near_gt_corridor | farspur_ambiguous_side_region
            if farspur_active and bool(self.base_ignore_farspur_near)
            else torch.zeros_like(hungarian_pos)
        )
        exist_ignore = (
            base_ignore_raw_rescue_mask
            | base_ignore_rank_near_mask
            | base_ignore_duplicate_like_mask
            | base_ignore_farspur_near_mask
        )
        exist_ignore = exist_ignore & unmatched & ~clear_far_final
        point_valid_ignore = (
            exist_ignore[:, :, None].expand(-1, -1, num_points).clone() if pred_valid_logits is not None else None
        )
        base_exist_negative_kept = unmatched & ~exist_ignore
        base_exist_negative_kept_clear_far = base_exist_negative_kept & clear_far_final
        base_exist_negative_kept_other = base_exist_negative_kept & ~clear_far_final
        farspur_near_still_base_negative = unmatched & farspur_near_gt_corridor & ~exist_ignore
        farspur_side_still_base_negative = unmatched & farspur_ambiguous_side_region & ~exist_ignore
        duplicate_like_rank_only = duplicate_like & ~clear_far_final
        duplicate_like_still_base_negative = unmatched & duplicate_like_rank_only & ~exist_ignore
        return {
            "hungarian_pos": hungarian_pos,
            "hungarian_gt_idx": hungarian_gt_idx,
            "shortside_matched_pos": shortside_matched_pos,
            "shortside_rescue_pos": shortside_rescue_pos,
            "shortside_rescue_conflict_pos": shortside_rescue_conflict_pos,
            "shortside_rescue_gt_idx": shortside_rescue_gt_idx,
            "raw_rescue_pos": raw_rescue_pos,
            "raw_rescue_gt_idx": raw_rescue_gt_idx,
            "shortside_boost_pos": shortside_boost_pos,
            "shortside_hungarian_rawmatch_pos": shortside_hungarian_rawmatch_pos,
            "shortside_unmatched_rescue_pos": shortside_unmatched_rescue_pos,
            "shortside_rawmatch_pos": shortside_hungarian_rawmatch_pos | shortside_unmatched_rescue_pos,
            "shortside_ultra_hungarian_pos": shortside_ultra_hungarian_pos,
            "shortside_ultra_pos": shortside_ultra_pos,
            "shortside_ultra_valid_pos": shortside_ultra_valid_pos,
            "shortside_target_floor_pos": shortside_target_floor_pos,
            "shortside_score_floor_pos": shortside_score_floor_pos,
            "shortside_ultra_score_floor_pos": shortside_ultra_score_floor_pos,
            "shortside_ultra_gt_idx": shortside_ultra_gt_idx,
            "shortside_selected_reason_code": shortside_selected_reason_code,
            "shortside_selected_gt_idx": shortside_selected_gt_idx,
            "near_gt_corridor": near_gt_corridor,
            "ambiguous_side_region": ambiguous_side_region,
            "clear_far_spurious": clear_far_spurious,
            "farspur_near_gt_corridor": farspur_near_gt_corridor,
            "farspur_ambiguous_side_region": farspur_ambiguous_side_region,
            "farspur_clear_far_spurious": farspur_clear_far_spurious,
            "clear_far_final": clear_far_final,
            "duplicate_like": duplicate_like,
            "normal_duplicate_like": normal_duplicate_like,
            "side_duplicate_like": side_duplicate_like,
            "rank_side_ambiguous_ignored": rank_side_ambiguous_ignored,
            "rank_pos": rank_pos,
            "rank_neg": rank_neg,
            "exist_ignore": exist_ignore,
            "point_valid_ignore": point_valid_ignore,
            "shortside_hungarian_rawmatch_candidate_count": pred_logits.new_tensor(
                float(shortside_hungarian_rawmatch_candidate_count)
            ),
            "shortside_unmatched_raw_rescue_candidate_count": pred_logits.new_tensor(
                float(shortside_unmatched_raw_rescue_candidate_count)
            ),
            "shortside_rawmatch_candidate_total_count": pred_logits.new_tensor(
                float(shortside_rawmatch_candidate_total_count)
            ),
            "raw_rescue_candidate_count": pred_logits.new_tensor(float(raw_rescue_candidate_count)),
            "raw_rescue_final_count": pred_logits.new_tensor(float(raw_rescue_final_count)),
            "raw_rescue_conflict_count": pred_logits.new_tensor(float(raw_rescue_conflict_count)),
            "short_raw_missing_count": pred_logits.new_tensor(float(short_raw_missing_count)),
            "shortside_selected_hungarian_rawmatch_count": pred_logits.new_tensor(
                float(shortside_selected_hungarian_rawmatch_count)
            ),
            "shortside_selected_unmatched_rescue_count": pred_logits.new_tensor(
                float(shortside_selected_unmatched_rescue_count)
            ),
            "shortside_rescue_conflict_count": pred_logits.new_tensor(float(shortside_rescue_conflict_count)),
            "shortside_missing_no_rawmatch_count": pred_logits.new_tensor(float(shortside_missing_no_rawmatch_count)),
            "shortside_nearest_was_duplicate_but_hungarian_boosted_count": pred_logits.new_tensor(
                float(shortside_nearest_was_duplicate_but_hungarian_boosted_count)
            ),
            "shortside_selected_by_abs_count": pred_logits.new_tensor(float(shortside_selected_by_abs_count)),
            "shortside_selected_by_median_count": pred_logits.new_tensor(float(shortside_selected_by_median_count)),
            "shortside_selected_total_count": pred_logits.new_tensor(float(shortside_selected_total_count)),
            "shortside_reliable_count": pred_logits.new_tensor(float(shortside_reliable_count)),
            "shortside_reliable_selected_count": pred_logits.new_tensor(float(shortside_reliable_selected_count)),
            "shortside_ultra_seen_count": pred_logits.new_tensor(float(shortside_ultra_seen_count)),
            "shortside_ultra_short_count": pred_logits.new_tensor(float(shortside_ultra_seen_count)),
            "shortside_ultra_enabled_count": pred_logits.new_tensor(float(shortside_ultra_enabled_count)),
            "shortside_ultra_score_floor_count": pred_logits.new_tensor(float(shortside_ultra_score_floor_count)),
            "shortside_ultra_valid_count": pred_logits.new_tensor(float(shortside_ultra_valid_count)),
            "shortside_visible_lt2_skipped_count": pred_logits.new_tensor(float(shortside_visible_lt2_skipped_count)),
            "shortside_skipped_visible_lt2_count": pred_logits.new_tensor(float(shortside_visible_lt2_skipped_count)),
            "shortside_ultra_hungarian_base_boost_count": pred_logits.new_tensor(
                float(shortside_ultra_hungarian_base_boost_count)
            ),
            "shortside_ultra_unmatched_rescue_seen_count": pred_logits.new_tensor(
                float(shortside_ultra_unmatched_rescue_seen_count)
            ),
            "shortside_ultra_unmatched_rescue_enabled_count": pred_logits.new_tensor(
                float(shortside_ultra_unmatched_rescue_enabled_count)
            ),
            "shortside_ultra_unmatched_rescue_skipped_count": pred_logits.new_tensor(
                float(shortside_ultra_unmatched_rescue_skipped_count)
            ),
            "shortside_ultra_in_gt4_4to3_count": pred_logits.new_tensor(float(shortside_ultra_in_gt4_4to3_count)),
            "shortside_visible_2_5_in_gt4_4to3_count": pred_logits.new_tensor(
                float(shortside_ultra_in_gt4_4to3_count)
            ),
            "shortside_ultra_in_gt5_5to4_count": pred_logits.new_tensor(float(shortside_ultra_in_gt5_5to4_count)),
            "shortside_visible_2_5_in_gt5_5to4_count": pred_logits.new_tensor(
                float(shortside_ultra_in_gt5_5to4_count)
            ),
            "near_gt_ignore_count": pred_logits.new_tensor(float(near_gt_corridor.sum().item())),
            "duplicate_like_rank_neg_count": pred_logits.new_tensor(float(duplicate_like.sum().item())),
            "clear_far_rank_neg_count": pred_logits.new_tensor(float(clear_far_spurious.sum().item())),
            "clear_far_boundary_count": pred_logits.new_tensor(float(clear_far_boundary_count)),
            "rank_neg_side_duplicate_like_count": pred_logits.new_tensor(float(side_duplicate_like.sum().item())),
            "rank_neg_normal_duplicate_like_count": pred_logits.new_tensor(float(normal_duplicate_like.sum().item())),
            "rank_side_ambiguous_ignored_count": pred_logits.new_tensor(float(rank_side_ambiguous_ignored.sum().item())),
            "rank_side_duplicate_rejected_better_than_true_count": pred_logits.new_tensor(
                float(rank_side_duplicate_rejected_better_than_true_count)
            ),
            "rank_pos_scope_id": pred_logits.new_tensor(
                float(
                    {
                        "shortside_reliable": 0,
                        "shortside_with_ultra": 1,
                        "gt4gt5_matched": 2,
                        "all_matched": 3,
                    }[self.rank_pos_scope]
                )
            ),
            "rank_pos_total_count": pred_logits.new_tensor(float(rank_pos.sum().item())),
            "rank_pos_hungarian_all_count": pred_logits.new_tensor(float(hungarian_pos.sum().item())),
            "rank_pos_shortside_matched_count": pred_logits.new_tensor(float(shortside_matched_pos.sum().item())),
            "rank_pos_shortside_rescue_count": pred_logits.new_tensor(float(rank_pos_shortside_rescue_only.sum().item())),
            "rank_pos_shortside_reliable_count": pred_logits.new_tensor(float(shortside_reliable_rank_pos.sum().item())),
            "rank_pos_shortside_ultra_count": pred_logits.new_tensor(float((rank_pos & shortside_ultra_rank_pos).sum().item())),
            "rank_pos_gt4gt5_matched_count": pred_logits.new_tensor(float(gt4gt5_matched_pos.sum().item())),
            "rank_pos_all_matched_count": pred_logits.new_tensor(float(all_matched_pos.sum().item())),
            "rank_pos_hungarian_count": pred_logits.new_tensor(float((rank_pos & hungarian_pos).sum().item())),
            "rank_pos_unmatched_rescue_excluded_count": pred_logits.new_tensor(
                float(rank_pos_unmatched_rescue_excluded.sum().item())
            ),
            "rank_pos_unmatched_rescue_included_count": pred_logits.new_tensor(
                float(unmatched_rescue_rank_included.sum().item())
            ),
            "rank_pos_conflict_excluded_count": pred_logits.new_tensor(
                float(rank_pos_conflict_excluded.sum().item())
            ),
            "rank_neg_duplicate_like_count": pred_logits.new_tensor(float(duplicate_like.sum().item())),
            "rank_neg_clear_far_count": pred_logits.new_tensor(float(clear_far_spurious.sum().item())),
            "rank_neg_near_ignored_count": pred_logits.new_tensor(float(near_gt_corridor.sum().item())),
            "base_exist_ignore_raw_rescue_count": pred_logits.new_tensor(
                float((exist_ignore & base_ignore_raw_rescue_mask).sum().item())
            ),
            "base_exist_ignore_rank_near_count": pred_logits.new_tensor(
                float((exist_ignore & near_gt_corridor & base_ignore_rank_near_mask).sum().item())
            ),
            "base_exist_ignore_rank_side_count": pred_logits.new_tensor(
                float((exist_ignore & ambiguous_side_region & base_ignore_rank_near_mask).sum().item())
            ),
            "base_exist_ignore_farspur_near_count": pred_logits.new_tensor(
                float((exist_ignore & farspur_near_gt_corridor & base_ignore_farspur_near_mask).sum().item())
            ),
            "base_exist_ignore_farspur_side_count": pred_logits.new_tensor(
                float((exist_ignore & farspur_ambiguous_side_region & base_ignore_farspur_near_mask).sum().item())
            ),
            "base_exist_ignore_duplicate_rank_only_count": pred_logits.new_tensor(
                float((exist_ignore & base_ignore_duplicate_like_mask).sum().item())
            ),
            "base_exist_ignore_near_count": pred_logits.new_tensor(
                float((exist_ignore & (near_gt_corridor | farspur_near_gt_corridor)).sum().item())
            ),
            "base_exist_ignore_side_ambiguous_count": pred_logits.new_tensor(
                float((exist_ignore & (ambiguous_side_region | farspur_ambiguous_side_region)).sum().item())
            ),
            "base_exist_ignore_duplicate_like_count": pred_logits.new_tensor(
                float((exist_ignore & duplicate_like).sum().item())
            ),
            "base_ignore_farspur_near_count": pred_logits.new_tensor(
                float((exist_ignore & farspur_near_gt_corridor & base_ignore_farspur_near_mask).sum().item())
            ),
            "base_ignore_farspur_side_count": pred_logits.new_tensor(
                float((exist_ignore & farspur_ambiguous_side_region & base_ignore_farspur_near_mask).sum().item())
            ),
            "base_ignore_duplicate_like_count": pred_logits.new_tensor(
                float((exist_ignore & duplicate_like).sum().item())
            ),
            "farspur_near_still_base_negative_count": pred_logits.new_tensor(
                float(farspur_near_still_base_negative.sum().item())
            ),
            "farspur_side_still_base_negative_count": pred_logits.new_tensor(
                float(farspur_side_still_base_negative.sum().item())
            ),
            "duplicate_like_still_base_negative_count": pred_logits.new_tensor(
                float(duplicate_like_still_base_negative.sum().item())
            ),
            "base_exist_negative_kept_clear_far_count": pred_logits.new_tensor(
                float((base_exist_negative_kept & clear_far_final).sum().item())
            ),
            "base_exist_negative_kept_other_count": pred_logits.new_tensor(
                float(base_exist_negative_kept_other.sum().item())
            ),
        }

    def shortside_rawmatch_boost_masks(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
        contract_masks: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-query/anchor weights for matched raw-close short side GT lanes."""
        query_weight = torch.ones_like(pred_logits)
        valid_weight = torch.ones_like(pred_valid_logits) if pred_valid_logits is not None else None
        zero = pred_logits.new_zeros(())
        if self.shortside_rawmatch_boost == 0.0 and not self.shortside_debug:
            return query_weight, valid_weight, zero, zero, zero, zero
        if contract_masks is None:
            contract_masks = self.build_count_contract_masks(
                pred_points, pred_logits, pred_valid_logits, gt_points, gt_valid, indices, gt_lanes
            )

        bsz, _, _, _ = pred_points.shape
        device, dtype = pred_points.device, pred_points.dtype
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=dtype).reshape(-1)
        if gt_lanes.numel() != bsz:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={bsz}.")

        boost = float(self.shortside_rawmatch_boost)
        exist_mul = 1.0 + boost
        valid_mul = 1.0 + 0.8 * boost
        boosted_count = 0
        boosted_gt4 = 0
        boosted_gt5 = 0
        boost_pos = contract_masks["shortside_boost_pos"]
        boost_gt_idx = contract_masks["shortside_selected_gt_idx"]
        assert isinstance(boost_pos, torch.Tensor) and isinstance(boost_gt_idx, torch.Tensor)
        for b in range(bsz):
            gt_count = int(round(float(gt_lanes[b].detach().item())))
            gt_points_b = gt_points[b].to(device=device, dtype=dtype)
            gt_valid_b = gt_valid[b].to(device=device, dtype=dtype)
            for src in boost_pos[b].nonzero(as_tuple=False).reshape(-1).to(device=device, dtype=torch.long):
                gt_i = int(boost_gt_idx[b, src].item())
                if gt_i < 0:
                    continue
                if boost > 0.0:
                    query_weight[b, src] = query_weight[b, src] * exist_mul
                    if valid_weight is not None:
                        visible_mask = gt_valid_b[gt_i] > 0.5
                        valid_weight[b, src, visible_mask] = valid_weight[b, src, visible_mask] * valid_mul
                boosted_count += 1
                if gt_count == 4:
                    boosted_gt4 += 1
                elif gt_count >= 5:
                    boosted_gt5 += 1

        return (
            query_weight,
            valid_weight,
            pred_logits.new_tensor(float(boosted_count)),
            pred_logits.new_tensor(float(boosted_gt4)),
            pred_logits.new_tensor(float(boosted_gt5)),
            contract_masks["short_raw_missing_count"].detach()
            if isinstance(contract_masks["short_raw_missing_count"], torch.Tensor)
            else zero,
        )

    def shortside_score_floor_loss(
        self,
        pred_logits: torch.Tensor,
        contract_masks: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        """Optional independent score-floor BCE for selected shortside queries."""
        losses: list[torch.Tensor] = []
        target = pred_logits.new_tensor(float(self.shortside_score_target)).clamp(min=0.0, max=1.0)
        score_floor_pos = contract_masks["shortside_score_floor_pos"]
        ultra_score_floor_pos = contract_masks["shortside_ultra_score_floor_pos"]
        assert isinstance(score_floor_pos, torch.Tensor) and isinstance(ultra_score_floor_pos, torch.Tensor)

        if self.shortside_score_floor_gain != 0.0 and bool(score_floor_pos.any()):
            losses.append(
                float(self.shortside_score_floor_gain)
                * F.binary_cross_entropy_with_logits(
                    pred_logits[score_floor_pos],
                    target.expand_as(pred_logits[score_floor_pos]),
                    reduction="mean",
                )
            )
        if self.shortside_ultra_score_floor_gain != 0.0 and bool(ultra_score_floor_pos.any()):
            losses.append(
                float(self.shortside_ultra_score_floor_gain)
                * F.binary_cross_entropy_with_logits(
                    pred_logits[ultra_score_floor_pos],
                    target.expand_as(pred_logits[ultra_score_floor_pos]),
                    reduction="mean",
                )
            )
        return torch.stack(losses).sum() if losses else self._zero_like(pred_logits)

    def shortside_ultra_point_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        contract_masks: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        """Optional point auxiliary for ultra-short selected shortside queries."""
        if self.shortside_ultra_point_gain == 0.0 or not bool(self.shortside_ultra_enable):
            return self._zero_like(pred_points)
        ultra_pos = contract_masks["shortside_ultra_pos"]
        ultra_gt_idx = contract_masks["shortside_ultra_gt_idx"]
        assert isinstance(ultra_pos, torch.Tensor) and isinstance(ultra_gt_idx, torch.Tensor)
        if not bool(ultra_pos.any()):
            return self._zero_like(pred_points)

        losses: list[torch.Tensor] = []
        device, dtype = pred_points.device, pred_points.dtype
        for b in range(pred_points.shape[0]):
            gt_points_b = gt_points[b].to(device=device, dtype=dtype)
            gt_valid_b = gt_valid[b].to(device=device, dtype=dtype)
            for q_t in ultra_pos[b].nonzero(as_tuple=False).reshape(-1).to(device=device, dtype=torch.long):
                gt_i = int(ultra_gt_idx[b, q_t].item())
                if gt_i < 0:
                    continue
                losses.append(
                    aspect_weighted_l1_point_loss(
                        pred_points[b, q_t].unsqueeze(0),
                        gt_points_b[gt_i].unsqueeze(0),
                        (gt_valid_b[gt_i].unsqueeze(0) > 0.5),
                        image_size=self.image_size,
                        y_weight=1.0,
                        x_only=False,
                    )
                )
        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    def shortside_ultra_valid_loss(
        self,
        pred_valid_logits: torch.Tensor | None,
        gt_valid: list[torch.Tensor],
        contract_masks: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        """Optional light point-valid auxiliary for ultra-short selected shortside queries."""
        if self.shortside_ultra_valid_gain == 0.0 or not bool(self.shortside_ultra_enable):
            if pred_valid_logits is not None:
                return self._zero_like(pred_valid_logits)
            zero_ref = next((v for v in gt_valid if isinstance(v, torch.Tensor)), None)
            if zero_ref is not None:
                return self._zero_like(zero_ref)
            return torch.zeros((), device=self.point_scale.device)
        if pred_valid_logits is None:
            raise ValueError(
                "gcs_shortside_ultra_valid_gain requires preds['pred_valid_logits'] with shape B x Q x K."
            )
        ultra_pos = contract_masks["shortside_ultra_valid_pos"]
        ultra_gt_idx = contract_masks["shortside_ultra_gt_idx"]
        assert isinstance(ultra_pos, torch.Tensor) and isinstance(ultra_gt_idx, torch.Tensor)
        if not bool(ultra_pos.any()):
            return self._zero_like(pred_valid_logits)

        losses: list[torch.Tensor] = []
        device, dtype = pred_valid_logits.device, pred_valid_logits.dtype
        for b in range(pred_valid_logits.shape[0]):
            gt_valid_b = gt_valid[b].to(device=device, dtype=dtype)
            for q_t in ultra_pos[b].nonzero(as_tuple=False).reshape(-1).to(device=device, dtype=torch.long):
                gt_i = int(ultra_gt_idx[b, q_t].item())
                if gt_i < 0:
                    continue
                losses.append(
                    F.binary_cross_entropy_with_logits(
                        pred_valid_logits[b, q_t],
                        gt_valid_b[gt_i],
                        reduction="mean",
                    )
                )
        return torch.stack(losses).mean() if losses else self._zero_like(pred_valid_logits)

    def shortside_rescue_aux_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        contract_masks: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Independent positive supervision for unmatched raw-rescue shortside queries."""
        raw_rescue_pos = contract_masks["raw_rescue_pos"]
        raw_rescue_gt_idx = contract_masks["raw_rescue_gt_idx"]
        assert isinstance(raw_rescue_pos, torch.Tensor) and isinstance(raw_rescue_gt_idx, torch.Tensor)
        if not bool(raw_rescue_pos.any()):
            zero = self._zero_like(pred_points)
            return zero, zero, zero

        exist_losses: list[torch.Tensor] = []
        point_losses: list[torch.Tensor] = []
        valid_losses: list[torch.Tensor] = []
        device, dtype = pred_points.device, pred_points.dtype
        for b in range(pred_points.shape[0]):
            gt_points_b = gt_points[b].to(device=device, dtype=dtype)
            gt_valid_b = gt_valid[b].to(device=device, dtype=dtype)
            for q_t in raw_rescue_pos[b].nonzero(as_tuple=False).reshape(-1).to(device=device, dtype=torch.long):
                gt_i = int(raw_rescue_gt_idx[b, q_t].item())
                if gt_i < 0:
                    continue
                exist_losses.append(
                    F.binary_cross_entropy_with_logits(pred_logits[b, q_t], pred_logits.new_ones(()), reduction="none")
                )
                point_losses.append(
                    aspect_weighted_l1_point_loss(
                        pred_points[b, q_t].unsqueeze(0),
                        gt_points_b[gt_i].unsqueeze(0),
                        (gt_valid_b[gt_i].unsqueeze(0) > 0.5),
                        image_size=self.image_size,
                        y_weight=1.0,
                        x_only=False,
                    )
                )
                if pred_valid_logits is not None:
                    valid_losses.append(
                        F.binary_cross_entropy_with_logits(
                            pred_valid_logits[b, q_t],
                            gt_valid_b[gt_i].to(device=pred_valid_logits.device, dtype=pred_valid_logits.dtype),
                            reduction="mean",
                        )
                    )

        zero = self._zero_like(pred_points)
        exist_loss = torch.stack(exist_losses).mean() if exist_losses else zero
        point_loss = torch.stack(point_losses).mean() if point_losses else zero
        valid_loss = torch.stack(valid_losses).mean() if valid_losses else zero
        return exist_loss, point_loss, valid_loss

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
        positive_extra_weight: torch.Tensor | None = None,
        ignore_mask: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """BCE supervision for visible fixed-y anchors on matched lanes and zero target for unmatched queries."""
        if pred_valid_logits is None:
            loss = self._zero_like(pred_points)
            zero = pred_points.new_zeros(())
            return (loss, zero, zero, zero) if return_details else loss

        target = torch.zeros_like(pred_valid_logits)
        if positive_extra_weight is not None:
            if positive_extra_weight.shape != pred_valid_logits.shape:
                raise ValueError(
                    "positive_extra_weight must have shape B x Q x K matching pred_valid_logits, "
                    f"got {tuple(positive_extra_weight.shape)} vs {tuple(pred_valid_logits.shape)}."
                )
            extra_weight = positive_extra_weight.to(device=pred_valid_logits.device, dtype=pred_valid_logits.dtype)
        else:
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
                extra_weight[b, src_idx] = extra_weight[b, src_idx] * local_weight
                gt5_short_boost_mask[b, src_idx] = local_boost_mask

        pos = target.sum().clamp_min(1.0)
        neg = (target.numel() - target.sum()).clamp_min(1.0)
        pos_weight = (neg / pos).clamp(min=1.0, max=float(self.point_valid_pos_weight_max)).to(pred_valid_logits)
        bce = F.binary_cross_entropy_with_logits(pred_valid_logits, target, pos_weight=pos_weight, reduction="none")
        if ignore_mask is not None:
            if ignore_mask.shape != pred_valid_logits.shape:
                raise ValueError(
                    "ignore_mask must have shape B x Q x K matching pred_valid_logits, "
                    f"got {tuple(ignore_mask.shape)} vs {tuple(pred_valid_logits.shape)}."
                )
            active_weight = extra_weight * (~ignore_mask.to(device=pred_valid_logits.device, dtype=torch.bool)).to(
                dtype=extra_weight.dtype
            )
        else:
            active_weight = extra_weight
        loss = (bce * active_weight).sum() / active_weight.sum().clamp_min(1.0)
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

    def _far_spurious_gt_group_and_weight(self, gt_count: int) -> tuple[int, float]:
        """Map GT lane count to diagnostic group and per-image far-spurious-negative weight."""
        if gt_count <= 3:
            return 3, float(self.far_spurious_gt3_weight)
        if gt_count == 4:
            return 4, float(self.far_spurious_gt4_weight)
        return 5, float(self.far_spurious_gt5_weight)

    @staticmethod
    def _longest_true_run(mask: torch.Tensor) -> int:
        """Return the longest contiguous True run in a 1D boolean mask."""
        best = 0
        current = 0
        for value in mask.detach().bool().flatten().tolist():
            if value:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

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
        ignore_mask: torch.Tensor | None = None,
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
        if ignore_mask is not None:
            if ignore_mask.shape != pred_logits.shape:
                raise ValueError(
                    f"ignore_mask must have shape B x Q matching pred_logits, got {tuple(ignore_mask.shape)} "
                    f"vs {tuple(pred_logits.shape)}."
                )
            ignore_mask = ignore_mask.to(device=device, dtype=torch.bool)
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
                if ignore_mask is not None and bool(ignore_mask[b, uq]):
                    continue
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

    def far_spurious_negative_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        ignore_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Extra BCE negative loss for far unmatched queries with enough score and valid anchors."""
        zero = pred_logits.new_zeros(())
        if self.far_spurious_neg_gain == 0.0:
            return self._zero_like(pred_points), zero, zero, zero, zero, zero, zero, zero
        if pred_valid_logits is None:
            raise ValueError(
                "gcs_far_spurious_neg requires preds['pred_valid_logits'] with shape B x Q x K; "
                "disable --gcs-far-spurious-neg or use a GCS head that emits per-point visibility logits."
            )

        bsz, num_queries, _, _ = pred_points.shape
        device = pred_logits.device
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=pred_logits.dtype).reshape(-1)
        if gt_lanes.numel() != bsz:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={bsz}.")

        valid_thr = float(self.eval_point_valid_thr)
        valid_prob = pred_valid_logits.detach().sigmoid()
        score_prob = pred_logits.detach().sigmoid()
        points = pred_points.detach()
        x_scale = self._spurious_x_scale(pred_points)
        min_gt_lanes = int(self.far_spurious_min_gt_lanes)
        max_gt_lanes = int(self.far_spurious_max_gt_lanes)
        min_valid = int(self.far_spurious_min_valid)
        max_valid = int(self.far_spurious_max_valid)
        min_overlap = int(self.far_spurious_min_overlap)
        score_thr = float(self.far_spurious_score_thr)
        protect_px = float(self.far_spurious_protect_px)
        far_px = float(self.far_spurious_far_px)
        if ignore_mask is not None:
            if ignore_mask.shape != pred_logits.shape:
                raise ValueError(
                    f"ignore_mask must have shape B x Q matching pred_logits, got {tuple(ignore_mask.shape)} "
                    f"vs {tuple(pred_logits.shape)}."
                )
            ignore_mask = ignore_mask.to(device=device, dtype=torch.bool)

        selected_losses = []
        selected_scores = []
        selected_valid_counts = []
        candidate_count = 0
        final_count = 0
        group_counts = {3: 0, 4: 0, 5: 0}

        for b in range(bsz):
            gt_count = int(round(float(gt_lanes[b].detach().item())))
            if gt_count < min_gt_lanes or gt_count > max_gt_lanes:
                continue
            group_key, image_weight = self._far_spurious_gt_group_and_weight(gt_count)
            src_idx, _ = indices[b]
            src_idx = src_idx.to(device=device, dtype=torch.long)
            matched_mask = torch.zeros(num_queries, dtype=torch.bool, device=device)
            if src_idx.numel():
                matched_mask[src_idx] = True
            unmatched_idx = torch.arange(num_queries, device=device)[~matched_mask]
            if unmatched_idx.numel() == 0:
                continue

            gt_points_b = gt_points[b].detach().to(device=device, dtype=points.dtype)
            gt_valid_b = gt_valid[b].detach().to(device=device)
            if gt_points_b.ndim != 3 or gt_points_b.shape[-1] != 2:
                raise ValueError(f"Each GT lane tensor must have shape N x K x 2, got {tuple(gt_points_b.shape)}.")
            if gt_valid_b.shape != gt_points_b.shape[:2]:
                raise ValueError(
                    f"GT valid mask must match GT lane first two dims, got {tuple(gt_valid_b.shape)} vs {tuple(gt_points_b.shape[:2])}."
                )

            for uq in unmatched_idx:
                if ignore_mask is not None and bool(ignore_mask[b, uq]):
                    continue
                score = score_prob[b, uq]
                if float(score.item()) < score_thr:
                    continue
                uq_valid = valid_prob[b, uq] >= valid_thr
                valid_count = self._longest_true_run(uq_valid)
                if valid_count < min_valid or valid_count > max_valid:
                    continue

                candidate_count += 1
                distances = []
                protected = False
                for gt_i in range(gt_points_b.shape[0]):
                    lane_valid = gt_valid_b[gt_i] > 0.5
                    overlap = uq_valid & lane_valid
                    if int(overlap.sum().item()) < min_overlap:
                        continue
                    dx_px = (points[b, uq, overlap, 0] - gt_points_b[gt_i, overlap, 0]).abs() * x_scale
                    mean_dx = float(dx_px.mean().item())
                    if mean_dx <= protect_px:
                        protected = True
                        break
                    distances.append(mean_dx)
                if protected or not distances:
                    continue
                if not all(distance > far_px for distance in distances):
                    continue

                raw_loss = F.binary_cross_entropy_with_logits(
                    pred_logits[b, uq], pred_logits.new_zeros(()), reduction="none"
                )
                selected_losses.append(raw_loss * image_weight)
                selected_scores.append(score)
                selected_valid_counts.append(pred_logits.new_tensor(float(valid_count)))
                final_count += 1
                group_counts[group_key] += 1

        loss = torch.stack(selected_losses).mean() if selected_losses else self._zero_like(pred_points)
        mean_score = torch.stack(selected_scores).mean() if selected_scores else zero
        mean_valid = torch.stack(selected_valid_counts).mean() if selected_valid_counts else zero
        return (
            loss,
            pred_logits.new_tensor(float(candidate_count)),
            pred_logits.new_tensor(float(final_count)),
            pred_logits.new_tensor(float(group_counts[3])),
            pred_logits.new_tensor(float(group_counts[4])),
            pred_logits.new_tensor(float(group_counts[5])),
            mean_score,
            mean_valid,
        )

    def farspur_ignore_first_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        contract_masks: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """BCE only on clear far-spurious unmatched queries after ignore-zone classification."""
        zero = pred_logits.new_zeros(())
        if self.farspur_weight == 0.0:
            return self._zero_like(pred_points), zero, zero, zero, zero, zero, zero
        if pred_valid_logits is None:
            raise ValueError(
                "gcs_farspur_weight requires preds['pred_valid_logits'] with shape B x Q x K; "
                "disable --gcs-farspur-weight or use a GCS head that emits per-point visibility logits."
            )

        bsz, _, _, _ = pred_points.shape
        device = pred_logits.device
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=pred_logits.dtype).reshape(-1)
        if gt_lanes.numel() != bsz:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={bsz}.")

        if contract_masks is None:
            contract_masks = self.build_count_contract_masks(
                pred_points, pred_logits, pred_valid_logits, gt_points, gt_valid, indices, gt_lanes
            )
        clear_far_mask = contract_masks.get("farspur_clear_far_spurious", contract_masks["clear_far_spurious"])
        near_mask = contract_masks.get("farspur_near_gt_corridor", contract_masks["near_gt_corridor"])
        side_mask = contract_masks.get("farspur_ambiguous_side_region", contract_masks["ambiguous_side_region"])
        hungarian_pos = contract_masks["hungarian_pos"]
        assert isinstance(clear_far_mask, torch.Tensor)
        assert isinstance(near_mask, torch.Tensor)
        assert isinstance(side_mask, torch.Tensor)
        assert isinstance(hungarian_pos, torch.Tensor)
        losses: list[torch.Tensor] = []
        sample_count = 0
        positive_count = 0
        ignore_count = 0
        clear_count = 0
        near_count = 0
        side_count = 0

        for b, (src_idx, _) in enumerate(indices):
            gt_count = int(round(float(gt_lanes[b].detach().item())))
            if gt_count <= 0:
                continue
            sample_count += 1
            positive_count += int(hungarian_pos[b].sum().item())
            ignore_count += int((near_mask[b] | side_mask[b]).sum().item())
            near_count += int(near_mask[b].sum().item())
            side_count += int(side_mask[b].sum().item())
            for uq_t in clear_far_mask[b].nonzero(as_tuple=False).reshape(-1):
                uq = int(uq_t.item())
                losses.append(
                    F.binary_cross_entropy_with_logits(pred_logits[b, uq], pred_logits.new_zeros(()), reduction="none")
                )
                clear_count += 1

        loss = torch.stack(losses).mean() if losses else self._zero_like(pred_points)
        return (
            loss,
            pred_logits.new_tensor(float(sample_count)),
            pred_logits.new_tensor(float(positive_count)),
            pred_logits.new_tensor(float(ignore_count)),
            pred_logits.new_tensor(float(clear_count)),
            pred_logits.new_tensor(float(near_count)),
            pred_logits.new_tensor(float(side_count)),
        )

    def rank_topk_loss(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_lanes: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        contract_masks: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Margin ranking loss that aligns matched true query scores above clear-far/duplicate scores."""
        zero = pred_logits.new_zeros(())
        if self.rank_topk_gain == 0.0:
            return self._zero_like(pred_points), zero, zero, zero, zero, zero, zero, zero, zero
        if pred_valid_logits is None:
            raise ValueError(
                "gcs_rank_topk_weight requires preds['pred_valid_logits'] with shape B x Q x K; "
                "disable --gcs-rank-topk-weight or use a GCS head that emits per-point visibility logits."
            )

        bsz, _, _, _ = pred_points.shape
        device = pred_logits.device
        gt_lanes = torch.as_tensor(gt_lanes, device=device, dtype=pred_logits.dtype).reshape(-1)
        if gt_lanes.numel() != bsz:
            raise ValueError(f"gt_lanes must have one value per image, got {gt_lanes.numel()} vs B={bsz}.")

        if contract_masks is None:
            contract_masks = self.build_count_contract_masks(
                pred_points, pred_logits, pred_valid_logits, gt_points, gt_valid, indices, gt_lanes
            )
        rank_pos_mask = contract_masks["rank_pos"]
        rank_neg_mask = contract_masks["rank_neg"]
        clear_far_mask = contract_masks["clear_far_spurious"]
        duplicate_mask = contract_masks["duplicate_like"]
        assert isinstance(rank_pos_mask, torch.Tensor)
        assert isinstance(rank_neg_mask, torch.Tensor)
        assert isinstance(clear_far_mask, torch.Tensor)
        assert isinstance(duplicate_mask, torch.Tensor)
        score_prob_detached = pred_logits.detach().sigmoid()
        lane_score = pred_logits.sigmoid()
        margin = float(self.rank_margin)
        max_negs = int(self.rank_max_negs)
        image_losses: list[torch.Tensor] = []
        pair_losses: list[torch.Tensor] = []
        sample_count = 0
        positive_count = 0
        negative_count = 0
        clear_count = 0
        duplicate_count = 0
        noop_images = 0
        noop_no_pos = 0
        noop_no_neg = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            gt_count = int(round(float(gt_lanes[b].detach().item())))
            if gt_count < int(self.rank_gt_min_lanes):
                continue
            pos_t = rank_pos_mask[b].nonzero(as_tuple=False).reshape(-1).to(device=device, dtype=torch.long)
            neg_t_all = rank_neg_mask[b].nonzero(as_tuple=False).reshape(-1).to(device=device, dtype=torch.long)
            missing_pos = pos_t.numel() == 0
            missing_neg = neg_t_all.numel() == 0
            if missing_pos or missing_neg:
                noop_images += 1
                noop_no_pos += int(missing_pos)
                noop_no_neg += int(missing_neg)
                continue
            order = torch.argsort(score_prob_detached[b, neg_t_all], descending=True)
            neg_t = neg_t_all[order[:max_negs]]
            pair_loss = torch.relu(margin - lane_score[b, pos_t].reshape(-1, 1) + lane_score[b, neg_t].reshape(1, -1))
            if self.rank_pair_reduction == "image_mean":
                image_losses.append(pair_loss.mean())
            else:
                pair_losses.append(pair_loss.reshape(-1))
            sample_count += 1
            positive_count += int(pos_t.numel())
            negative_count += int(neg_t.numel())
            clear_count += int(clear_far_mask[b, neg_t].sum().item())
            duplicate_count += int((duplicate_mask[b, neg_t] & ~clear_far_mask[b, neg_t]).sum().item())

        if self.rank_pair_reduction == "image_mean":
            loss = torch.stack(image_losses).mean() if image_losses else self._zero_like(pred_points)
        else:
            loss = torch.cat(pair_losses).mean() if pair_losses else self._zero_like(pred_points)
        return (
            loss,
            pred_logits.new_tensor(float(sample_count)),
            pred_logits.new_tensor(float(positive_count)),
            pred_logits.new_tensor(float(negative_count)),
            pred_logits.new_tensor(float(clear_count)),
            pred_logits.new_tensor(float(duplicate_count)),
            pred_logits.new_tensor(float(noop_images)),
            pred_logits.new_tensor(float(noop_no_pos)),
            pred_logits.new_tensor(float(noop_no_neg)),
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
        gt_lanes = self.target_lane_count(pred_logits, batch, gt_valid)
        contract_masks = self.build_count_contract_masks(
            pred_points, pred_logits, pred_valid_logits, gt_points, gt_valid, indices, gt_lanes
        )
        (
            short_raw_query_weight,
            short_raw_valid_weight,
            short_raw_boost_count,
            short_raw_boost_gt4,
            short_raw_boost_gt5,
            short_raw_missing,
        ) = self.shortside_rawmatch_boost_masks(
            pred_points,
            pred_logits,
            pred_valid_logits,
            gt_points,
            gt_valid,
            indices,
            gt_lanes,
            contract_masks=contract_masks,
        )
        (
            shortside_rescue_exist_loss,
            shortside_rescue_point_loss,
            shortside_rescue_valid_loss,
        ) = self.shortside_rescue_aux_loss(
            pred_points, pred_logits, pred_valid_logits, gt_points, gt_valid, contract_masks
        )
        shortside_score_floor_loss = self.shortside_score_floor_loss(pred_logits, contract_masks)
        shortside_ultra_point_loss = self.shortside_ultra_point_loss(pred_points, gt_points, gt_valid, contract_masks)
        shortside_ultra_valid_loss = self.shortside_ultra_valid_loss(pred_valid_logits, gt_valid, contract_masks)
        shortside_target_floor_pos = contract_masks["shortside_target_floor_pos"]
        assert isinstance(shortside_target_floor_pos, torch.Tensor)
        target_floor_enabled = bool(float(self.shortside_exist_target_floor) > 0.0)
        target_floor_mask = shortside_target_floor_pos if target_floor_enabled else None
        target_stats_mask = (
            shortside_target_floor_pos
            if target_floor_enabled or self.shortside_debug or self.shortside_rawmatch_boost != 0.0
            else None
        )
        exist_target_before_floor = self._build_exist_target(
            pred_logits,
            pred_points,
            pred_valid_logits,
            gt_points,
            gt_valid,
            indices,
        )
        (
            shortside_rawmatch_target_mean_before,
            shortside_rawmatch_target_min_before,
            shortside_rawmatch_target_p25_before,
            shortside_rawmatch_target_p50_before,
            _shortside_rawmatch_target_p75_before,
            shortside_rawmatch_target_below_05_count,
            shortside_rawmatch_target_below_07_count,
        ) = self._selected_target_stats(exist_target_before_floor, target_stats_mask)
        if target_floor_mask is not None:
            floor_mask = target_floor_mask.to(device=exist_target_before_floor.device, dtype=torch.bool)
            floor_value = exist_target_before_floor.new_tensor(float(self.shortside_exist_target_floor)).clamp(
                min=0.0, max=1.0
            )
            shortside_target_floor_applied_count = (
                (floor_mask & (exist_target_before_floor < floor_value))
                .sum()
                .to(device=exist_target_before_floor.device, dtype=exist_target_before_floor.dtype)
            )
        else:
            shortside_target_floor_applied_count = pred_logits.new_zeros(())
        exist_ignore_mask = contract_masks["exist_ignore"]
        if target_floor_mask is not None and isinstance(exist_ignore_mask, torch.Tensor):
            exist_ignore_mask = exist_ignore_mask & ~target_floor_mask
        exist_loss, exist_target = self.exist_loss(
            pred_logits,
            pred_points,
            pred_valid_logits,
            gt_points,
            gt_valid,
            indices,
            query_weight=short_raw_query_weight,
            ignore_mask=exist_ignore_mask,
            target_floor_mask=target_floor_mask,
            target_floor=float(self.shortside_exist_target_floor),
            return_target=True,
        )
        if target_stats_mask is not None:
            shortside_grad_mask = target_stats_mask.to(device=pred_logits.device, dtype=torch.bool)
            score_prob_detached = pred_logits.detach().sigmoid()
            exist_target_detached = exist_target.detach()
            shortside_score_grad_down_count = (
                (shortside_grad_mask & (score_prob_detached > exist_target_detached))
                .sum()
                .to(device=pred_logits.device, dtype=pred_logits.dtype)
            )
            shortside_score_grad_up_count = (
                (shortside_grad_mask & (score_prob_detached < exist_target_detached))
                .sum()
                .to(device=pred_logits.device, dtype=pred_logits.dtype)
            )
        else:
            shortside_score_grad_down_count = pred_logits.new_zeros(())
            shortside_score_grad_up_count = pred_logits.new_zeros(())
        (
            short_raw_boost_exist_target_mean,
            short_raw_boost_exist_target_min,
            short_raw_boost_exist_target_p25,
            short_raw_boost_exist_target_p50,
            short_raw_boost_exist_target_p75,
            short_raw_boost_target_below_05_count,
            short_raw_boost_target_below_07_count,
        ) = self._selected_target_stats(
            exist_target_before_floor,
            target_stats_mask,
        )
        point_loss = self.point_loss(pred_points, gt_points, gt_valid, indices)
        (
            point_valid_loss,
            gt5_short_pos_count,
            gt5_short_pos_anchor_count,
            gt5_short_point_valid_loss,
        ) = self.point_valid_loss(
            pred_valid_logits,
            pred_points,
            gt_valid,
            indices,
            gt_lanes=gt_lanes,
            positive_extra_weight=short_raw_valid_weight,
            ignore_mask=contract_masks["point_valid_ignore"],
            return_details=True,
        )
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
            pred_points,
            pred_logits,
            pred_valid_logits,
            indices,
            gt_lanes,
            gt_points,
            gt_valid,
            ignore_mask=contract_masks["exist_ignore"],
        )
        (
            far_spur_loss,
            far_spur_cand,
            far_spur_neg,
            far_spur_gt3,
            far_spur_gt4,
            far_spur_gt5,
            far_spur_score,
            far_spur_valid,
        ) = self.far_spurious_negative_loss(
            pred_points,
            pred_logits,
            pred_valid_logits,
            indices,
            gt_lanes,
            gt_points,
            gt_valid,
            ignore_mask=contract_masks["exist_ignore"],
        )
        (
            farspur_if_loss,
            farspur_if_samples,
            farspur_if_pos,
            farspur_if_ignore,
            farspur_if_clear,
            farspur_if_near,
            farspur_if_side,
        ) = self.farspur_ignore_first_loss(
            pred_points,
            pred_logits,
            pred_valid_logits,
            indices,
            gt_lanes,
            gt_points,
            gt_valid,
            contract_masks=contract_masks,
        )
        (
            rank_topk_loss,
            rank_topk_samples,
            rank_topk_pos,
            rank_topk_neg,
            rank_topk_clear,
            rank_topk_dup,
            rank_loss_noop_images,
            rank_noop_because_no_pos,
            rank_noop_because_no_neg,
        ) = self.rank_topk_loss(
            pred_points,
            pred_logits,
            pred_valid_logits,
            indices,
            gt_lanes,
            gt_points,
            gt_valid,
            contract_masks=contract_masks,
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
        if self.far_spurious_neg_gain != 0.0:
            total = total + self.far_spurious_neg_gain * far_spur_loss
        if self.farspur_weight != 0.0:
            total = total + self.farspur_weight * farspur_if_loss
        if self.rank_topk_gain != 0.0:
            total = total + self.rank_topk_gain * rank_topk_loss
        if self.shortside_rescue_exist_gain != 0.0:
            total = total + self.shortside_rescue_exist_gain * shortside_rescue_exist_loss
        if self.shortside_rescue_point_gain != 0.0:
            total = total + self.shortside_rescue_point_gain * shortside_rescue_point_loss
        if self.shortside_rescue_valid_gain != 0.0:
            total = total + self.shortside_rescue_valid_gain * shortside_rescue_valid_loss
        if self.shortside_score_floor_gain != 0.0 or self.shortside_ultra_score_floor_gain != 0.0:
            total = total + shortside_score_floor_loss
        if self.shortside_ultra_point_gain != 0.0:
            total = total + self.shortside_ultra_point_gain * shortside_ultra_point_loss
        if self.shortside_ultra_valid_gain != 0.0:
            total = total + self.shortside_ultra_valid_gain * shortside_ultra_valid_loss
        shortside_hungarian_rawmatch_candidate_count = contract_masks[
            "shortside_hungarian_rawmatch_candidate_count"
        ]
        shortside_unmatched_raw_rescue_candidate_count = contract_masks[
            "shortside_unmatched_raw_rescue_candidate_count"
        ]
        shortside_rawmatch_candidate_total_count = contract_masks["shortside_rawmatch_candidate_total_count"]
        raw_rescue_candidate_count = contract_masks["raw_rescue_candidate_count"]
        raw_rescue_final_count = contract_masks["raw_rescue_final_count"]
        raw_rescue_conflict_count = contract_masks["raw_rescue_conflict_count"]
        shortside_selected_hungarian_rawmatch_count = contract_masks["shortside_selected_hungarian_rawmatch_count"]
        shortside_selected_unmatched_rescue_count = contract_masks["shortside_selected_unmatched_rescue_count"]
        shortside_rescue_conflict_count = contract_masks["shortside_rescue_conflict_count"]
        shortside_missing_no_rawmatch_count = contract_masks["shortside_missing_no_rawmatch_count"]
        shortside_nearest_was_duplicate_but_hungarian_boosted_count = contract_masks[
            "shortside_nearest_was_duplicate_but_hungarian_boosted_count"
        ]
        shortside_selected_by_abs_count = contract_masks["shortside_selected_by_abs_count"]
        shortside_selected_by_median_count = contract_masks["shortside_selected_by_median_count"]
        shortside_selected_total_count = contract_masks["shortside_selected_total_count"]
        shortside_reliable_count = contract_masks["shortside_reliable_count"]
        shortside_reliable_selected_count = contract_masks["shortside_reliable_selected_count"]
        shortside_ultra_seen_count = contract_masks["shortside_ultra_seen_count"]
        shortside_ultra_enabled_count = contract_masks["shortside_ultra_enabled_count"]
        shortside_ultra_score_floor_count = contract_masks["shortside_ultra_score_floor_count"]
        shortside_ultra_valid_count = contract_masks["shortside_ultra_valid_count"]
        shortside_visible_lt2_skipped_count = contract_masks["shortside_visible_lt2_skipped_count"]
        shortside_ultra_hungarian_base_boost_count = contract_masks["shortside_ultra_hungarian_base_boost_count"]
        shortside_ultra_unmatched_rescue_seen_count = contract_masks["shortside_ultra_unmatched_rescue_seen_count"]
        shortside_ultra_unmatched_rescue_enabled_count = contract_masks[
            "shortside_ultra_unmatched_rescue_enabled_count"
        ]
        shortside_ultra_unmatched_rescue_skipped_count = contract_masks[
            "shortside_ultra_unmatched_rescue_skipped_count"
        ]
        shortside_ultra_in_gt4_4to3_count = contract_masks["shortside_ultra_in_gt4_4to3_count"]
        shortside_ultra_in_gt5_5to4_count = contract_masks["shortside_ultra_in_gt5_5to4_count"]
        near_gt_ignore_count = contract_masks["near_gt_ignore_count"]
        duplicate_like_rank_neg_count = contract_masks["duplicate_like_rank_neg_count"]
        clear_far_rank_neg_count = contract_masks["clear_far_rank_neg_count"]
        clear_far_boundary_count = contract_masks["clear_far_boundary_count"]
        rank_neg_side_duplicate_like_count = contract_masks["rank_neg_side_duplicate_like_count"]
        rank_neg_normal_duplicate_like_count = contract_masks["rank_neg_normal_duplicate_like_count"]
        rank_side_ambiguous_ignored_count = contract_masks["rank_side_ambiguous_ignored_count"]
        rank_side_duplicate_rejected_better_than_true_count = contract_masks[
            "rank_side_duplicate_rejected_better_than_true_count"
        ]
        rank_pos_scope_id = contract_masks["rank_pos_scope_id"]
        rank_pos_total_count = contract_masks["rank_pos_total_count"]
        rank_pos_hungarian_all_count = contract_masks["rank_pos_hungarian_all_count"]
        rank_pos_shortside_matched_count = contract_masks["rank_pos_shortside_matched_count"]
        rank_pos_shortside_rescue_count = contract_masks["rank_pos_shortside_rescue_count"]
        rank_pos_shortside_reliable_count = contract_masks["rank_pos_shortside_reliable_count"]
        rank_pos_shortside_ultra_count = contract_masks["rank_pos_shortside_ultra_count"]
        rank_pos_gt4gt5_matched_count = contract_masks["rank_pos_gt4gt5_matched_count"]
        rank_pos_all_matched_count = contract_masks["rank_pos_all_matched_count"]
        rank_pos_hungarian_count = contract_masks["rank_pos_hungarian_count"]
        rank_pos_unmatched_rescue_excluded_count = contract_masks["rank_pos_unmatched_rescue_excluded_count"]
        rank_pos_unmatched_rescue_included_count = contract_masks["rank_pos_unmatched_rescue_included_count"]
        rank_pos_conflict_excluded_count = contract_masks["rank_pos_conflict_excluded_count"]
        rank_neg_duplicate_like_count = contract_masks["rank_neg_duplicate_like_count"]
        rank_neg_clear_far_count = contract_masks["rank_neg_clear_far_count"]
        rank_neg_near_ignored_count = contract_masks["rank_neg_near_ignored_count"]
        base_exist_ignore_raw_rescue_count = contract_masks["base_exist_ignore_raw_rescue_count"]
        base_exist_ignore_near_count = contract_masks["base_exist_ignore_near_count"]
        base_exist_ignore_side_ambiguous_count = contract_masks["base_exist_ignore_side_ambiguous_count"]
        base_exist_ignore_duplicate_like_count = contract_masks["base_exist_ignore_duplicate_like_count"]
        base_exist_negative_kept_clear_far_count = contract_masks["base_exist_negative_kept_clear_far_count"]
        base_exist_negative_kept_other_count = contract_masks["base_exist_negative_kept_other_count"]
        base_exist_ignore_rank_near_count = contract_masks["base_exist_ignore_rank_near_count"]
        base_exist_ignore_rank_side_count = contract_masks["base_exist_ignore_rank_side_count"]
        base_exist_ignore_farspur_near_count = contract_masks["base_exist_ignore_farspur_near_count"]
        base_exist_ignore_farspur_side_count = contract_masks["base_exist_ignore_farspur_side_count"]
        base_exist_ignore_duplicate_rank_only_count = contract_masks[
            "base_exist_ignore_duplicate_rank_only_count"
        ]
        base_ignore_farspur_near_count = contract_masks["base_ignore_farspur_near_count"]
        base_ignore_farspur_side_count = contract_masks["base_ignore_farspur_side_count"]
        base_ignore_duplicate_like_count = contract_masks["base_ignore_duplicate_like_count"]
        farspur_near_still_base_negative_count = contract_masks["farspur_near_still_base_negative_count"]
        farspur_side_still_base_negative_count = contract_masks["farspur_side_still_base_negative_count"]
        duplicate_like_still_base_negative_count = contract_masks["duplicate_like_still_base_negative_count"]
        farspur_active = bool(self.farspur_ignore_first and self.farspur_weight > 0.0)
        rank_effective = bool(self.rank_topk_gain > 0.0)
        farspur_aux_only_mode = pred_logits.new_tensor(
            float(farspur_active and not bool(self.base_ignore_farspur_near))
        )
        farspur_full_ignore_first_mode = pred_logits.new_tensor(
            float(farspur_active and bool(self.base_ignore_farspur_near))
        )
        rank_pair_only_mode = pred_logits.new_tensor(
            float(rank_effective and not bool(self.base_ignore_duplicate_like))
        )
        rank_full_duplicate_contract_mode = pred_logits.new_tensor(
            float(rank_effective and bool(self.base_ignore_duplicate_like))
        )
        shortside_boost_only_mode = pred_logits.new_tensor(
            float(self.shortside_rawmatch_boost > 0.0 and self.shortside_exist_target_floor <= 0.0)
        )
        shortside_protect_mode = pred_logits.new_tensor(
            float(self.shortside_rawmatch_boost > 0.0 and self.shortside_exist_target_floor > 0.0)
        )
        legacy_spurious_active = pred_logits.new_tensor(
            float(self.spurious_neg_gain != 0.0 or self.far_spurious_neg_gain != 0.0)
        )
        assert isinstance(shortside_hungarian_rawmatch_candidate_count, torch.Tensor)
        assert isinstance(shortside_unmatched_raw_rescue_candidate_count, torch.Tensor)
        assert isinstance(shortside_rawmatch_candidate_total_count, torch.Tensor)
        assert isinstance(raw_rescue_candidate_count, torch.Tensor)
        assert isinstance(raw_rescue_final_count, torch.Tensor)
        assert isinstance(raw_rescue_conflict_count, torch.Tensor)
        assert isinstance(shortside_selected_hungarian_rawmatch_count, torch.Tensor)
        assert isinstance(shortside_selected_unmatched_rescue_count, torch.Tensor)
        assert isinstance(shortside_rescue_conflict_count, torch.Tensor)
        assert isinstance(shortside_missing_no_rawmatch_count, torch.Tensor)
        assert isinstance(shortside_nearest_was_duplicate_but_hungarian_boosted_count, torch.Tensor)
        assert isinstance(shortside_selected_by_abs_count, torch.Tensor)
        assert isinstance(shortside_selected_by_median_count, torch.Tensor)
        assert isinstance(shortside_selected_total_count, torch.Tensor)
        assert isinstance(shortside_reliable_count, torch.Tensor)
        assert isinstance(shortside_reliable_selected_count, torch.Tensor)
        assert isinstance(shortside_ultra_seen_count, torch.Tensor)
        assert isinstance(shortside_ultra_enabled_count, torch.Tensor)
        assert isinstance(shortside_ultra_score_floor_count, torch.Tensor)
        assert isinstance(shortside_ultra_valid_count, torch.Tensor)
        assert isinstance(shortside_visible_lt2_skipped_count, torch.Tensor)
        assert isinstance(shortside_ultra_hungarian_base_boost_count, torch.Tensor)
        assert isinstance(shortside_ultra_unmatched_rescue_seen_count, torch.Tensor)
        assert isinstance(shortside_ultra_unmatched_rescue_enabled_count, torch.Tensor)
        assert isinstance(shortside_ultra_unmatched_rescue_skipped_count, torch.Tensor)
        assert isinstance(shortside_ultra_in_gt4_4to3_count, torch.Tensor)
        assert isinstance(shortside_ultra_in_gt5_5to4_count, torch.Tensor)
        assert isinstance(near_gt_ignore_count, torch.Tensor)
        assert isinstance(duplicate_like_rank_neg_count, torch.Tensor)
        assert isinstance(clear_far_rank_neg_count, torch.Tensor)
        assert isinstance(clear_far_boundary_count, torch.Tensor)
        assert isinstance(rank_neg_side_duplicate_like_count, torch.Tensor)
        assert isinstance(rank_neg_normal_duplicate_like_count, torch.Tensor)
        assert isinstance(rank_side_ambiguous_ignored_count, torch.Tensor)
        assert isinstance(rank_side_duplicate_rejected_better_than_true_count, torch.Tensor)
        assert isinstance(rank_pos_scope_id, torch.Tensor)
        assert isinstance(rank_pos_total_count, torch.Tensor)
        assert isinstance(rank_pos_hungarian_all_count, torch.Tensor)
        assert isinstance(rank_pos_shortside_matched_count, torch.Tensor)
        assert isinstance(rank_pos_shortside_rescue_count, torch.Tensor)
        assert isinstance(rank_pos_shortside_reliable_count, torch.Tensor)
        assert isinstance(rank_pos_shortside_ultra_count, torch.Tensor)
        assert isinstance(rank_pos_gt4gt5_matched_count, torch.Tensor)
        assert isinstance(rank_pos_all_matched_count, torch.Tensor)
        assert isinstance(rank_pos_hungarian_count, torch.Tensor)
        assert isinstance(rank_pos_unmatched_rescue_excluded_count, torch.Tensor)
        assert isinstance(rank_pos_unmatched_rescue_included_count, torch.Tensor)
        assert isinstance(rank_pos_conflict_excluded_count, torch.Tensor)
        assert isinstance(rank_neg_duplicate_like_count, torch.Tensor)
        assert isinstance(rank_neg_clear_far_count, torch.Tensor)
        assert isinstance(rank_neg_near_ignored_count, torch.Tensor)
        assert isinstance(base_exist_ignore_raw_rescue_count, torch.Tensor)
        assert isinstance(base_exist_ignore_rank_near_count, torch.Tensor)
        assert isinstance(base_exist_ignore_rank_side_count, torch.Tensor)
        assert isinstance(base_exist_ignore_farspur_near_count, torch.Tensor)
        assert isinstance(base_exist_ignore_farspur_side_count, torch.Tensor)
        assert isinstance(base_exist_ignore_duplicate_rank_only_count, torch.Tensor)
        assert isinstance(base_ignore_farspur_near_count, torch.Tensor)
        assert isinstance(base_ignore_farspur_side_count, torch.Tensor)
        assert isinstance(base_ignore_duplicate_like_count, torch.Tensor)
        assert isinstance(farspur_near_still_base_negative_count, torch.Tensor)
        assert isinstance(farspur_side_still_base_negative_count, torch.Tensor)
        assert isinstance(duplicate_like_still_base_negative_count, torch.Tensor)
        assert isinstance(base_exist_ignore_near_count, torch.Tensor)
        assert isinstance(base_exist_ignore_side_ambiguous_count, torch.Tensor)
        assert isinstance(base_exist_ignore_duplicate_like_count, torch.Tensor)
        assert isinstance(base_exist_negative_kept_clear_far_count, torch.Tensor)
        assert isinstance(base_exist_negative_kept_other_count, torch.Tensor)
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
                far_spur_loss.detach(),
                far_spur_cand.detach(),
                far_spur_neg.detach(),
                far_spur_gt3.detach(),
                far_spur_gt4.detach(),
                far_spur_gt5.detach(),
                far_spur_score.detach(),
                far_spur_valid.detach(),
                count_score_mean.detach(),
                gt5_short_pos_count.detach(),
                gt5_short_pos_anchor_count.detach(),
                gt5_short_point_valid_loss.detach(),
                cnt_bound_5under.detach(),
                cnt_score.detach(),
                short_raw_boost_count.detach(),
                short_raw_boost_gt4.detach(),
                short_raw_boost_gt5.detach(),
                short_raw_missing.detach(),
                shortside_selected_by_abs_count.detach(),
                shortside_selected_by_median_count.detach(),
                shortside_selected_total_count.detach(),
                shortside_hungarian_rawmatch_candidate_count.detach(),
                shortside_unmatched_raw_rescue_candidate_count.detach(),
                shortside_rawmatch_candidate_total_count.detach(),
                raw_rescue_candidate_count.detach(),
                raw_rescue_final_count.detach(),
                raw_rescue_conflict_count.detach(),
                shortside_selected_hungarian_rawmatch_count.detach(),
                shortside_selected_unmatched_rescue_count.detach(),
                shortside_rescue_conflict_count.detach(),
                shortside_missing_no_rawmatch_count.detach(),
                shortside_nearest_was_duplicate_but_hungarian_boosted_count.detach(),
                shortside_rescue_exist_loss.detach(),
                shortside_rescue_point_loss.detach(),
                shortside_rescue_valid_loss.detach(),
                shortside_score_floor_loss.detach(),
                shortside_ultra_point_loss.detach(),
                shortside_ultra_valid_loss.detach(),
                short_raw_boost_exist_target_mean.detach(),
                short_raw_boost_exist_target_min.detach(),
                short_raw_boost_exist_target_p25.detach(),
                short_raw_boost_exist_target_p50.detach(),
                short_raw_boost_exist_target_p75.detach(),
                short_raw_boost_target_below_05_count.detach(),
                short_raw_boost_target_below_07_count.detach(),
                shortside_rawmatch_target_mean_before.detach(),
                shortside_rawmatch_target_min_before.detach(),
                shortside_rawmatch_target_p25_before.detach(),
                shortside_rawmatch_target_p50_before.detach(),
                shortside_rawmatch_target_below_05_count.detach(),
                shortside_rawmatch_target_below_07_count.detach(),
                shortside_target_floor_applied_count.detach(),
                shortside_score_grad_down_count.detach(),
                shortside_score_grad_up_count.detach(),
                shortside_boost_only_mode.detach(),
                shortside_protect_mode.detach(),
                shortside_reliable_count.detach(),
                shortside_reliable_selected_count.detach(),
                shortside_ultra_seen_count.detach(),
                shortside_ultra_enabled_count.detach(),
                shortside_ultra_score_floor_count.detach(),
                shortside_ultra_valid_count.detach(),
                shortside_visible_lt2_skipped_count.detach(),
                shortside_ultra_hungarian_base_boost_count.detach(),
                shortside_ultra_unmatched_rescue_seen_count.detach(),
                shortside_ultra_unmatched_rescue_enabled_count.detach(),
                shortside_ultra_unmatched_rescue_skipped_count.detach(),
                shortside_ultra_in_gt4_4to3_count.detach(),
                shortside_ultra_in_gt5_5to4_count.detach(),
                near_gt_ignore_count.detach(),
                duplicate_like_rank_neg_count.detach(),
                clear_far_rank_neg_count.detach(),
                clear_far_boundary_count.detach(),
                rank_neg_side_duplicate_like_count.detach(),
                rank_neg_normal_duplicate_like_count.detach(),
                rank_side_ambiguous_ignored_count.detach(),
                rank_side_duplicate_rejected_better_than_true_count.detach(),
                rank_pos_scope_id.detach(),
                rank_pos_total_count.detach(),
                rank_pos_hungarian_all_count.detach(),
                rank_pos_shortside_matched_count.detach(),
                rank_pos_shortside_rescue_count.detach(),
                rank_pos_shortside_reliable_count.detach(),
                rank_pos_shortside_ultra_count.detach(),
                rank_pos_gt4gt5_matched_count.detach(),
                rank_pos_all_matched_count.detach(),
                rank_pos_hungarian_count.detach(),
                rank_pos_unmatched_rescue_excluded_count.detach(),
                rank_pos_unmatched_rescue_included_count.detach(),
                rank_pos_conflict_excluded_count.detach(),
                rank_neg_duplicate_like_count.detach(),
                rank_neg_clear_far_count.detach(),
                rank_neg_near_ignored_count.detach(),
                base_exist_ignore_raw_rescue_count.detach(),
                base_exist_ignore_rank_near_count.detach(),
                base_exist_ignore_rank_side_count.detach(),
                base_exist_ignore_farspur_near_count.detach(),
                base_exist_ignore_farspur_side_count.detach(),
                base_exist_ignore_duplicate_rank_only_count.detach(),
                base_exist_ignore_near_count.detach(),
                base_exist_ignore_side_ambiguous_count.detach(),
                base_exist_ignore_duplicate_like_count.detach(),
                base_ignore_farspur_near_count.detach(),
                base_ignore_farspur_side_count.detach(),
                base_ignore_duplicate_like_count.detach(),
                farspur_aux_only_mode.detach(),
                farspur_full_ignore_first_mode.detach(),
                farspur_near_still_base_negative_count.detach(),
                farspur_side_still_base_negative_count.detach(),
                rank_pair_only_mode.detach(),
                rank_full_duplicate_contract_mode.detach(),
                duplicate_like_still_base_negative_count.detach(),
                base_exist_negative_kept_clear_far_count.detach(),
                base_exist_negative_kept_other_count.detach(),
                legacy_spurious_active.detach(),
                farspur_if_loss.detach(),
                farspur_if_samples.detach(),
                farspur_if_pos.detach(),
                farspur_if_ignore.detach(),
                farspur_if_clear.detach(),
                farspur_if_near.detach(),
                farspur_if_side.detach(),
                rank_topk_loss.detach(),
                rank_topk_samples.detach(),
                rank_topk_pos.detach(),
                rank_topk_neg.detach(),
                rank_topk_clear.detach(),
                rank_topk_dup.detach(),
                rank_loss_noop_images.detach(),
                rank_noop_because_no_pos.detach(),
                rank_noop_because_no_neg.detach(),
                rank_neg_duplicate_like_count.detach(),
                rank_neg_clear_far_count.detach(),
                rank_pos_shortside_reliable_count.detach(),
                rank_pos_gt4gt5_matched_count.detach(),
                rank_loss_noop_images.detach(),
                rank_noop_because_no_pos.detach(),
                rank_noop_because_no_neg.detach(),
            )
        )
        return total, loss_items
