# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions for GCS-YOLO-Lane structured lane training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.gcs_matcher import GCSHungarianMatcher
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, assert_gcs_shape, normalize_imgsz


class GCSLoss(nn.Module):
    """Hungarian-matched loss for query-based structured lane predictions."""

    loss_names = (
        "exist_loss",
        "point_loss",
        "lane_balanced_point_loss",
        "gt4_lane_balanced_point_loss",
        "point_valid_loss",
        "short_valid_recall_loss",
        "smooth_loss",
        "curve_loss",
        "mask_loss",
        "edge_loss",
        "count_loss",
        "count_under5_loss",
        "duplicate_margin_loss",
        "spurious_margin_loss",
        "far_spurious_survival_loss",
        "gt5_rank_consistency_loss",
        "gt3_extra_survival_loss",
    )

    def __init__(
        self,
        model=None,
        lambda_exist: float | None = None,
        lambda_point: float | None = None,
        lambda_point_valid: float | None = None,
        lane_balanced_point_gain: float | None = None,
        gt4_lane_balanced_point_gain: float | None = None,
        gt4_lane_balanced_topk: int | None = None,
        gt4_lane_balanced_max_mult: float | None = None,
        short_valid_recall_gain: float | None = None,
        short_valid_max_visible: int | None = None,
        short_valid_min_visible: int | None = None,
        short_valid_max_ape_px: float | None = None,
        short_valid_min_visible_iou: float | None = None,
        lambda_smooth: float | None = None,
        lambda_curve: float | None = None,
        lambda_mask: float | None = None,
        lambda_edge: float | None = None,
        lambda_count: float | None = None,
        lambda_count_under5: float | None = None,
        duplicate_margin_gain: float | None = None,
        duplicate_margin_logit: float | None = None,
        duplicate_gt_count: int | None = None,
        duplicate_short_visible_max: int | None = None,
        duplicate_min_overlap: int | None = None,
        duplicate_min_visible_iou: float | None = None,
        duplicate_pos_ape_px: float | None = None,
        duplicate_neg_ape_px: float | None = None,
        duplicate_ape_gap_px: float | None = None,
        duplicate_max_pairs_per_gt: int | None = None,
        spurious_margin_gain: float | None = None,
        spurious_margin_logit: float | None = None,
        spurious_gt_counts=None,
        spurious_pos_ape_px: float | None = None,
        spurious_pos_min_visible_iou: float | None = None,
        spurious_neg_min_ape_px: float | None = None,
        spurious_neg_max_visible_iou: float | None = None,
        spurious_duplicate_ape_px: float | None = None,
        spurious_duplicate_visible_iou: float | None = None,
        spurious_max_pairs_per_image: int | None = None,
        far_spurious_survival_gain: float | None = None,
        far_spurious_gt_counts=None,
        far_spurious_score_thr: float | None = None,
        far_spurious_min_score: float | None = None,
        far_spurious_min_ape_px: float | None = None,
        far_spurious_max_visible_iou: float | None = None,
        far_spurious_point_valid_thr: float | None = None,
        far_spurious_min_visible_run: int | None = None,
        far_spurious_max_neg_per_image: int | None = None,
        far_spurious_loss_type: str | None = None,
        gt5_rank_consistency_gain: float | None = None,
        gt5_rank_margin_logit: float | None = None,
        gt5_rank_min_qminus_score: float | None = None,
        gt5_rank_max_pairs_per_image: int | None = None,
        gt3_extra_survival_gain: float | None = None,
        gt3_extra_margin_logit: float | None = None,
        gt3_extra_topk: int | None = None,
        count_under5_min_lanes: int | None = None,
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
        self.lane_balanced_point_gain = float(
            lane_balanced_point_gain
            if lane_balanced_point_gain is not None
            else self._arg(args, "gcs_lane_balanced_point", 0.0)
        )
        self.gt4_lane_balanced_point_gain = float(
            gt4_lane_balanced_point_gain
            if gt4_lane_balanced_point_gain is not None
            else self._arg(args, "gcs_gt4_lane_balanced_point", 0.0)
        )
        self.gt4_lane_balanced_topk = int(
            gt4_lane_balanced_topk
            if gt4_lane_balanced_topk is not None
            else self._arg(args, "gcs_gt4_lane_balanced_topk", 1)
        )
        self.gt4_lane_balanced_max_mult = float(
            gt4_lane_balanced_max_mult
            if gt4_lane_balanced_max_mult is not None
            else self._arg(args, "gcs_gt4_lane_balanced_max_mult", 2.0)
        )
        self.short_valid_recall_gain = float(
            short_valid_recall_gain
            if short_valid_recall_gain is not None
            else self._arg(args, "gcs_short_valid_recall", 0.0)
        )
        self.short_valid_max_visible = int(
            short_valid_max_visible
            if short_valid_max_visible is not None
            else self._arg(args, "gcs_short_valid_max_visible", 20)
        )
        self.short_valid_min_visible = int(
            short_valid_min_visible
            if short_valid_min_visible is not None
            else self._arg(args, "gcs_short_valid_min_visible", 4)
        )
        self.short_valid_max_ape_px = float(
            short_valid_max_ape_px
            if short_valid_max_ape_px is not None
            else self._arg(args, "gcs_short_valid_max_ape_px", 40.0)
        )
        self.short_valid_min_visible_iou = float(
            short_valid_min_visible_iou
            if short_valid_min_visible_iou is not None
            else self._arg(args, "gcs_short_valid_min_visible_iou", 0.3)
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
        self.duplicate_margin_gain = float(
            duplicate_margin_gain
            if duplicate_margin_gain is not None
            else self._arg(args, "gcs_duplicate_margin", 0.0)
        )
        self.duplicate_margin_logit = float(
            duplicate_margin_logit
            if duplicate_margin_logit is not None
            else self._arg(args, "gcs_duplicate_margin_logit", 1.0)
        )
        self.duplicate_gt_count = int(
            duplicate_gt_count
            if duplicate_gt_count is not None
            else self._arg(args, "gcs_duplicate_gt_count", 4)
        )
        self.duplicate_short_visible_max = int(
            duplicate_short_visible_max
            if duplicate_short_visible_max is not None
            else self._arg(args, "gcs_duplicate_short_visible_max", 20)
        )
        self.duplicate_min_overlap = int(
            duplicate_min_overlap
            if duplicate_min_overlap is not None
            else self._arg(args, "gcs_duplicate_min_overlap", 2)
        )
        self.duplicate_min_visible_iou = float(
            duplicate_min_visible_iou
            if duplicate_min_visible_iou is not None
            else self._arg(args, "gcs_duplicate_min_visible_iou", 0.4)
        )
        self.duplicate_pos_ape_px = float(
            duplicate_pos_ape_px
            if duplicate_pos_ape_px is not None
            else self._arg(args, "gcs_duplicate_pos_ape_px", 20.0)
        )
        self.duplicate_neg_ape_px = float(
            duplicate_neg_ape_px
            if duplicate_neg_ape_px is not None
            else self._arg(args, "gcs_duplicate_neg_ape_px", 120.0)
        )
        self.duplicate_ape_gap_px = float(
            duplicate_ape_gap_px
            if duplicate_ape_gap_px is not None
            else self._arg(args, "gcs_duplicate_ape_gap_px", 5.0)
        )
        self.duplicate_max_pairs_per_gt = int(
            duplicate_max_pairs_per_gt
            if duplicate_max_pairs_per_gt is not None
            else self._arg(args, "gcs_duplicate_max_pairs_per_gt", 2)
        )
        self.spurious_margin_gain = float(
            spurious_margin_gain
            if spurious_margin_gain is not None
            else self._arg(args, "gcs_spurious_margin", 0.0)
        )
        self.spurious_margin_logit = float(
            spurious_margin_logit
            if spurious_margin_logit is not None
            else self._arg(args, "gcs_spurious_margin_logit", 1.0)
        )
        self.spurious_gt_counts = self._parse_int_set(
            spurious_gt_counts
            if spurious_gt_counts is not None
            else self._arg(args, "gcs_spurious_gt_counts", (3, 4)),
            default=(3, 4),
        )
        self.spurious_pos_ape_px = float(
            spurious_pos_ape_px
            if spurious_pos_ape_px is not None
            else self._arg(args, "gcs_spurious_pos_ape_px", 20.0)
        )
        self.spurious_pos_min_visible_iou = float(
            spurious_pos_min_visible_iou
            if spurious_pos_min_visible_iou is not None
            else self._arg(args, "gcs_spurious_pos_min_visible_iou", 0.4)
        )
        self.spurious_neg_min_ape_px = float(
            spurious_neg_min_ape_px
            if spurious_neg_min_ape_px is not None
            else self._arg(args, "gcs_spurious_neg_min_ape_px", 50.0)
        )
        self.spurious_neg_max_visible_iou = float(
            spurious_neg_max_visible_iou
            if spurious_neg_max_visible_iou is not None
            else self._arg(args, "gcs_spurious_neg_max_visible_iou", 0.2)
        )
        self.spurious_duplicate_ape_px = float(
            spurious_duplicate_ape_px
            if spurious_duplicate_ape_px is not None
            else self._arg(args, "gcs_spurious_duplicate_ape_px", 50.0)
        )
        self.spurious_duplicate_visible_iou = float(
            spurious_duplicate_visible_iou
            if spurious_duplicate_visible_iou is not None
            else self._arg(args, "gcs_spurious_duplicate_visible_iou", 0.4)
        )
        self.spurious_max_pairs_per_image = int(
            spurious_max_pairs_per_image
            if spurious_max_pairs_per_image is not None
            else self._arg(args, "gcs_spurious_max_pairs_per_image", 4)
        )
        self.far_spurious_survival_gain = float(
            far_spurious_survival_gain
            if far_spurious_survival_gain is not None
            else self._arg(args, "gcs_far_spurious_survival", 0.0)
        )
        self.far_spurious_gt_counts = self._parse_int_set(
            far_spurious_gt_counts
            if far_spurious_gt_counts is not None
            else self._arg(args, "gcs_far_spurious_gt_counts", (3, 4)),
            default=(3, 4),
        )
        self.far_spurious_score_thr = float(
            far_spurious_score_thr
            if far_spurious_score_thr is not None
            else self._arg(args, "gcs_far_spurious_score_thr", 0.03)
        )
        self.far_spurious_min_score = float(
            far_spurious_min_score
            if far_spurious_min_score is not None
            else self._arg(args, "gcs_far_spurious_min_score", 0.03)
        )
        self.far_spurious_min_ape_px = float(
            far_spurious_min_ape_px
            if far_spurious_min_ape_px is not None
            else self._arg(args, "gcs_far_spurious_min_ape_px", 50.0)
        )
        self.far_spurious_max_visible_iou = float(
            far_spurious_max_visible_iou
            if far_spurious_max_visible_iou is not None
            else self._arg(args, "gcs_far_spurious_max_visible_iou", 0.2)
        )
        self.far_spurious_point_valid_thr = float(
            far_spurious_point_valid_thr
            if far_spurious_point_valid_thr is not None
            else self._arg(args, "gcs_far_spurious_point_valid_thr", 0.5)
        )
        self.far_spurious_min_visible_run = int(
            far_spurious_min_visible_run
            if far_spurious_min_visible_run is not None
            else self._arg(args, "gcs_far_spurious_min_visible_run", 5)
        )
        self.far_spurious_max_neg_per_image = int(
            far_spurious_max_neg_per_image
            if far_spurious_max_neg_per_image is not None
            else self._arg(args, "gcs_far_spurious_max_neg_per_image", 1)
        )
        self.far_spurious_loss_type = str(
            far_spurious_loss_type
            if far_spurious_loss_type is not None
            else self._arg(args, "gcs_far_spurious_loss_type", "relu")
        ).lower()
        self.gt5_rank_consistency_gain = float(
            gt5_rank_consistency_gain
            if gt5_rank_consistency_gain is not None
            else self._arg(args, "gcs_gt5_rank_consistency", 0.0)
        )
        self.gt5_rank_margin_logit = float(
            gt5_rank_margin_logit
            if gt5_rank_margin_logit is not None
            else self._arg(args, "gcs_gt5_rank_margin_logit", 0.5)
        )
        self.gt5_rank_min_qminus_score = float(
            gt5_rank_min_qminus_score
            if gt5_rank_min_qminus_score is not None
            else self._arg(args, "gcs_gt5_rank_min_qminus_score", 0.02)
        )
        self.gt5_rank_max_pairs_per_image = int(
            gt5_rank_max_pairs_per_image
            if gt5_rank_max_pairs_per_image is not None
            else self._arg(args, "gcs_gt5_rank_max_pairs_per_image", 1)
        )
        self.gt3_extra_survival_gain = float(
            gt3_extra_survival_gain
            if gt3_extra_survival_gain is not None
            else self._arg(args, "gcs_gt3_extra_survival", 0.0)
        )
        self.gt3_extra_margin_logit = float(
            gt3_extra_margin_logit
            if gt3_extra_margin_logit is not None
            else self._arg(args, "gcs_gt3_extra_margin_logit", 0.05)
        )
        self.gt3_extra_topk = int(
            gt3_extra_topk
            if gt3_extra_topk is not None
            else self._arg(args, "gcs_gt3_extra_topk", 1)
        )
        if self.duplicate_margin_gain < 0.0:
            raise ValueError(f"gcs_duplicate_margin must be >= 0, got {self.duplicate_margin_gain}.")
        if self.duplicate_margin_logit < 0.0:
            raise ValueError(f"gcs_duplicate_margin_logit must be >= 0, got {self.duplicate_margin_logit}.")
        if self.duplicate_gt_count < 1:
            raise ValueError(f"gcs_duplicate_gt_count must be >= 1, got {self.duplicate_gt_count}.")
        if self.duplicate_short_visible_max < 1:
            raise ValueError(
                f"gcs_duplicate_short_visible_max must be >= 1, got {self.duplicate_short_visible_max}."
            )
        if self.duplicate_min_overlap < 0:
            raise ValueError(f"gcs_duplicate_min_overlap must be >= 0, got {self.duplicate_min_overlap}.")
        if not (0.0 <= self.duplicate_min_visible_iou <= 1.0):
            raise ValueError(
                "gcs_duplicate_min_visible_iou must be in [0, 1], "
                f"got {self.duplicate_min_visible_iou}."
            )
        if self.duplicate_pos_ape_px < 0.0:
            raise ValueError(f"gcs_duplicate_pos_ape_px must be >= 0, got {self.duplicate_pos_ape_px}.")
        if self.duplicate_neg_ape_px <= 0.0:
            raise ValueError(f"gcs_duplicate_neg_ape_px must be > 0, got {self.duplicate_neg_ape_px}.")
        if self.duplicate_ape_gap_px < 0.0:
            raise ValueError(f"gcs_duplicate_ape_gap_px must be >= 0, got {self.duplicate_ape_gap_px}.")
        if self.duplicate_max_pairs_per_gt < 1:
            raise ValueError(
                f"gcs_duplicate_max_pairs_per_gt must be >= 1, got {self.duplicate_max_pairs_per_gt}."
            )
        if self.spurious_margin_gain < 0.0:
            raise ValueError(f"gcs_spurious_margin must be >= 0, got {self.spurious_margin_gain}.")
        if self.spurious_margin_logit < 0.0:
            raise ValueError(f"gcs_spurious_margin_logit must be >= 0, got {self.spurious_margin_logit}.")
        if not self.spurious_gt_counts or min(self.spurious_gt_counts) < 1:
            raise ValueError(f"gcs_spurious_gt_counts must contain positive lane counts, got {self.spurious_gt_counts}.")
        if self.spurious_pos_ape_px < 0.0:
            raise ValueError(f"gcs_spurious_pos_ape_px must be >= 0, got {self.spurious_pos_ape_px}.")
        if not (0.0 <= self.spurious_pos_min_visible_iou <= 1.0):
            raise ValueError(
                "gcs_spurious_pos_min_visible_iou must be in [0, 1], "
                f"got {self.spurious_pos_min_visible_iou}."
            )
        if self.spurious_neg_min_ape_px <= 0.0:
            raise ValueError(f"gcs_spurious_neg_min_ape_px must be > 0, got {self.spurious_neg_min_ape_px}.")
        if not (0.0 <= self.spurious_neg_max_visible_iou <= 1.0):
            raise ValueError(
                "gcs_spurious_neg_max_visible_iou must be in [0, 1], "
                f"got {self.spurious_neg_max_visible_iou}."
            )
        if self.spurious_duplicate_ape_px < 0.0:
            raise ValueError(
                f"gcs_spurious_duplicate_ape_px must be >= 0, got {self.spurious_duplicate_ape_px}."
            )
        if not (0.0 <= self.spurious_duplicate_visible_iou <= 1.0):
            raise ValueError(
                "gcs_spurious_duplicate_visible_iou must be in [0, 1], "
                f"got {self.spurious_duplicate_visible_iou}."
            )
        if self.spurious_max_pairs_per_image < 1:
            raise ValueError(
                f"gcs_spurious_max_pairs_per_image must be >= 1, got {self.spurious_max_pairs_per_image}."
            )
        if self.far_spurious_survival_gain < 0.0:
            raise ValueError(
                f"gcs_far_spurious_survival must be >= 0, got {self.far_spurious_survival_gain}."
            )
        if not self.far_spurious_gt_counts or min(self.far_spurious_gt_counts) < 1:
            raise ValueError(
                "gcs_far_spurious_gt_counts must contain positive lane counts, "
                f"got {self.far_spurious_gt_counts}."
            )
        if not (0.0 < self.far_spurious_score_thr < 1.0):
            raise ValueError(
                f"gcs_far_spurious_score_thr must be in (0, 1), got {self.far_spurious_score_thr}."
            )
        if not (0.0 <= self.far_spurious_min_score < 1.0):
            raise ValueError(
                f"gcs_far_spurious_min_score must be in [0, 1), got {self.far_spurious_min_score}."
            )
        if self.far_spurious_min_ape_px <= 0.0:
            raise ValueError(f"gcs_far_spurious_min_ape_px must be > 0, got {self.far_spurious_min_ape_px}.")
        if not (0.0 <= self.far_spurious_max_visible_iou <= 1.0):
            raise ValueError(
                "gcs_far_spurious_max_visible_iou must be in [0, 1], "
                f"got {self.far_spurious_max_visible_iou}."
            )
        if not (0.0 <= self.far_spurious_point_valid_thr <= 1.0):
            raise ValueError(
                "gcs_far_spurious_point_valid_thr must be in [0, 1], "
                f"got {self.far_spurious_point_valid_thr}."
            )
        if self.far_spurious_min_visible_run < 1:
            raise ValueError(
                f"gcs_far_spurious_min_visible_run must be >= 1, got {self.far_spurious_min_visible_run}."
            )
        if self.far_spurious_max_neg_per_image < 1:
            raise ValueError(
                f"gcs_far_spurious_max_neg_per_image must be >= 1, got {self.far_spurious_max_neg_per_image}."
            )
        if self.far_spurious_loss_type not in {"relu", "softplus"}:
            raise ValueError(
                "gcs_far_spurious_loss_type must be either 'relu' or 'softplus', "
                f"got {self.far_spurious_loss_type!r}."
            )
        if self.gt5_rank_consistency_gain < 0.0:
            raise ValueError(
                f"gcs_gt5_rank_consistency must be >= 0, got {self.gt5_rank_consistency_gain}."
            )
        if self.gt5_rank_margin_logit < 0.0:
            raise ValueError(f"gcs_gt5_rank_margin_logit must be >= 0, got {self.gt5_rank_margin_logit}.")
        if not (0.0 <= self.gt5_rank_min_qminus_score < 1.0):
            raise ValueError(
                f"gcs_gt5_rank_min_qminus_score must be in [0, 1), got {self.gt5_rank_min_qminus_score}."
            )
        if self.gt5_rank_max_pairs_per_image < 1:
            raise ValueError(
                f"gcs_gt5_rank_max_pairs_per_image must be >= 1, got {self.gt5_rank_max_pairs_per_image}."
            )
        if self.gt3_extra_survival_gain < 0.0:
            raise ValueError(f"gcs_gt3_extra_survival must be >= 0, got {self.gt3_extra_survival_gain}.")
        if self.gt3_extra_margin_logit < 0.0:
            raise ValueError(f"gcs_gt3_extra_margin_logit must be >= 0, got {self.gt3_extra_margin_logit}.")
        if self.gt3_extra_topk < 1:
            raise ValueError(f"gcs_gt3_extra_topk must be >= 1, got {self.gt3_extra_topk}.")
        if self.lane_balanced_point_gain < 0.0:
            raise ValueError(f"gcs_lane_balanced_point must be >= 0, got {self.lane_balanced_point_gain}.")
        if self.gt4_lane_balanced_point_gain < 0.0:
            raise ValueError(
                f"gcs_gt4_lane_balanced_point must be >= 0, got {self.gt4_lane_balanced_point_gain}."
            )
        if self.gt4_lane_balanced_topk < 1:
            raise ValueError(f"gcs_gt4_lane_balanced_topk must be >= 1, got {self.gt4_lane_balanced_topk}.")
        if self.gt4_lane_balanced_max_mult < 1.0:
            raise ValueError(
                f"gcs_gt4_lane_balanced_max_mult must be >= 1.0, got {self.gt4_lane_balanced_max_mult}."
            )
        if self.short_valid_recall_gain < 0.0:
            raise ValueError(f"gcs_short_valid_recall must be >= 0, got {self.short_valid_recall_gain}.")
        if self.short_valid_min_visible < 1:
            raise ValueError(f"gcs_short_valid_min_visible must be >= 1, got {self.short_valid_min_visible}.")
        if self.short_valid_max_visible < self.short_valid_min_visible:
            raise ValueError(
                "gcs_short_valid_max_visible must be >= gcs_short_valid_min_visible "
                f"({self.short_valid_max_visible} < {self.short_valid_min_visible})."
            )
        if self.short_valid_max_ape_px < 0.0:
            raise ValueError(f"gcs_short_valid_max_ape_px must be >= 0, got {self.short_valid_max_ape_px}.")
        if not (0.0 <= self.short_valid_min_visible_iou <= 1.0):
            raise ValueError(
                "gcs_short_valid_min_visible_iou must be in [0, 1], "
                f"got {self.short_valid_min_visible_iou}."
            )
        self.count_under5_min_lanes = int(
            count_under5_min_lanes
            if count_under5_min_lanes is not None
            else self._arg(args, "gcs_count_under5_min_lanes", 5)
        )
        if self.count_under5_min_lanes < 1:
            raise ValueError(f"gcs_count_under5_min_lanes must be >= 1, got {self.count_under5_min_lanes}.")
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
    def _parse_int_set(value, default) -> frozenset[int]:
        """Parse a lane-count selector from a scalar, sequence, or comma-separated string."""
        if value is None or value == "":
            value = default
        if isinstance(value, str):
            text = value.strip()
            for ch in "[](){}":
                text = text.replace(ch, " ")
            values = [int(x) for x in text.replace(",", " ").split()]
        elif isinstance(value, (list, tuple, set, frozenset)):
            values = [int(x) for x in value]
        else:
            values = [int(value)]
        return frozenset(values)

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
        scale = self._scale_for(pred_points)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pred = pred_points[b, src_idx]
            target = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
            valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]

            loss = ((pred - target).abs() * scale).sum(dim=-1)
            loss = loss * valid
            losses.append(loss.sum() / valid.sum().clamp_min(1.0))

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    def lane_balanced_point_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Matched point loss averaged per lane before averaging across lanes."""
        if self.lane_balanced_point_gain <= 0.0:
            return self._zero_like(pred_points)

        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._scale_for(pred_points)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pred = pred_points[b, src_idx]
            target = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
            valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]
            visible = valid.sum(dim=1)
            has_visible = visible > 0.0
            if not has_visible.any():
                continue

            loss = ((pred - target).abs() * scale).sum(dim=-1) * valid
            lane_loss = loss.sum(dim=1) / visible.clamp_min(1.0)
            losses.append(lane_loss[has_visible])

        return torch.cat(losses).mean() if losses else self._zero_like(pred_points)

    def gt4_lane_balanced_point_loss(
        self,
        pred_points: torch.Tensor,
        batch: dict,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Reweight the weakest matched point lane inside GT4 images without changing average lane weight."""
        if self.gt4_lane_balanced_point_gain <= 0.0:
            return self._zero_like(pred_points)

        dummy_logits = pred_points.new_zeros((pred_points.shape[0], pred_points.shape[1]))
        target_counts = self.target_lane_count(dummy_logits, batch, gt_valid).detach()
        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._scale_for(pred_points)
        topk_cfg = int(self.gt4_lane_balanced_topk)
        max_mult = float(self.gt4_lane_balanced_max_mult)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if int(round(float(target_counts[b].item()))) != 4:
                continue
            if src_idx.numel() < 4:
                continue

            pred = pred_points[b, src_idx]
            target = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
            valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]
            visible = valid.sum(dim=1)
            has_visible = visible > 0.0
            if int(has_visible.sum().item()) < 4:
                continue

            loss = ((pred - target).abs() * scale).sum(dim=-1) * valid
            lane_loss = loss.sum(dim=1) / visible.clamp_min(1.0)
            lane_loss = lane_loss[has_visible]
            topk = min(topk_cfg, int(lane_loss.numel()))
            weak_idx = torch.topk(lane_loss.detach(), k=topk, largest=True).indices
            lane_weights = torch.ones_like(lane_loss)
            lane_weights[weak_idx] = max_mult
            lane_weights = lane_weights / lane_weights.mean().clamp_min(1e-6)

            baseline = lane_loss.mean()
            weighted = (lane_loss * lane_weights).mean()
            losses.append(weighted - baseline)

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    def point_valid_loss(
        self,
        pred_valid_logits: torch.Tensor | None,
        pred_points: torch.Tensor,
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """BCE supervision for visible fixed-y anchors on matched lanes and zero target for unmatched queries."""
        if pred_valid_logits is None:
            return self._zero_like(pred_points)

        target = torch.zeros_like(pred_valid_logits)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target[b, src_idx] = gt_valid[b].to(device=target.device, dtype=target.dtype)[tgt_idx]

        pos = target.sum().clamp_min(1.0)
        neg = (target.numel() - target.sum()).clamp_min(1.0)
        pos_weight = (neg / pos).clamp(min=1.0, max=float(self.point_valid_pos_weight_max)).to(pred_valid_logits)
        return F.binary_cross_entropy_with_logits(pred_valid_logits, target, pos_weight=pos_weight)

    def short_valid_recall_loss(
        self,
        pred_valid_logits: torch.Tensor | None,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Positive-only visibility BCE on geometrically plausible matched short GT lanes."""
        if self.short_valid_recall_gain <= 0.0 or pred_valid_logits is None:
            return self._zero_like(pred_points)

        losses = []
        device = pred_valid_logits.device
        dtype = pred_valid_logits.dtype
        min_visible = float(self.short_valid_min_visible)
        max_visible = float(self.short_valid_max_visible)
        max_ape = float(self.short_valid_max_ape_px)
        min_iou = float(self.short_valid_min_visible_iou)
        scale = self._pixel_scale_for(pred_points)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]
            visible = valid.sum(dim=1)
            short_mask = (visible >= min_visible) & (visible <= max_visible)
            if not short_mask.any():
                continue

            pred = pred_points[b, src_idx].detach()
            target_points = gt_points[b].to(device=pred_points.device, dtype=pred_points.dtype)[tgt_idx]
            valid_points = valid.to(dtype=pred_points.dtype)
            point_error = torch.norm((pred - target_points) * scale, dim=-1)
            ape = (point_error * valid_points).sum(dim=1) / visible.to(dtype=pred_points.dtype).clamp_min(1.0)

            valid_prob = pred_valid_logits[b, src_idx].detach().sigmoid().to(dtype=dtype)
            intersection = (valid_prob * valid).sum(dim=1)
            union = valid_prob.sum(dim=1) + visible - intersection
            visible_iou = intersection / union.clamp_min(1e-6)

            lane_mask = short_mask & (ape.to(dtype=dtype) <= max_ape) & (visible_iou >= min_iou)
            if not lane_mask.any():
                continue

            logits = pred_valid_logits[b, src_idx[lane_mask]]
            valid = valid[lane_mask]
            visible = visible[lane_mask]
            bce = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
            lane_loss = (bce * valid).sum(dim=1) / visible.clamp_min(1.0)
            losses.append(lane_loss)

        return torch.cat(losses).mean() if losses else self._zero_like(pred_points)

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
        self, pred_logits: torch.Tensor, batch: dict, gt_valid: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return base cardinality loss and targeted undercount loss for dense-lane samples."""
        pred_count = pred_logits.sigmoid().sum(dim=1)
        target = self.target_lane_count(pred_logits, batch, gt_valid)
        count_loss = F.smooth_l1_loss(pred_count, target)

        under_gap = torch.relu(target - pred_count)
        mask = (target >= float(self.count_under5_min_lanes)).to(dtype=pred_logits.dtype)
        count_under5_loss = (mask * under_gap.pow(2)).sum() / mask.sum().clamp_min(1.0)
        return count_loss, count_under5_loss

    def count_loss(self, pred_logits: torch.Tensor, batch: dict, gt_valid: list[torch.Tensor]) -> torch.Tensor:
        """Cardinality loss that aligns summed existence probability with GT lane count."""
        return self.count_losses(pred_logits, batch, gt_valid)[0]

    def duplicate_margin_loss(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        batch: dict,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Penalize unmatched duplicate-like queries near a reliably matched short GT lane."""
        if self.duplicate_margin_gain <= 0.0 or pred_valid_logits is None:
            return self._zero_like(pred_points)

        num_lanes = batch.get("num_lanes")
        if num_lanes is not None:
            num_lanes = torch.as_tensor(num_lanes, device=pred_logits.device).reshape(-1)
            if num_lanes.numel() != pred_logits.shape[0]:
                raise ValueError(
                    f"batch['num_lanes'] must have one value per image, got {num_lanes.numel()} "
                    f"vs B={pred_logits.shape[0]}."
                )

        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._pixel_scale_for(pred_points)
        min_iou = float(self.duplicate_min_visible_iou)
        min_overlap = int(self.duplicate_min_overlap)
        max_pairs = int(self.duplicate_max_pairs_per_gt)
        margin_logit = float(self.duplicate_margin_logit)

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            if num_lanes is not None:
                image_lane_count = int(num_lanes[b].item())
            else:
                valid_b = gt_valid[b].detach().to(device=device)
                image_lane_count = int((valid_b.float().sum(dim=1) >= 2).sum().item())
            if image_lane_count != self.duplicate_gt_count:
                continue

            points_b = pred_points[b].detach()
            logits_b = pred_logits[b]
            valid_prob_b = pred_valid_logits[b].detach().sigmoid().to(device=device, dtype=dtype)
            gt_points_b = gt_points[b].to(device=device, dtype=dtype)
            gt_valid_b = gt_valid[b].to(device=device, dtype=dtype)
            if gt_points_b.numel() == 0:
                continue
            valid_counts_all = gt_valid_b.sum(dim=1)
            valid_gt_mask = valid_counts_all >= 2
            if not valid_gt_mask.any():
                continue

            all_diff_px = (points_b[:, None] - gt_points_b[None]) * scale
            all_point_error = torch.norm(all_diff_px, dim=-1)
            all_ape = (all_point_error * gt_valid_b[None]).sum(dim=2) / valid_counts_all[None].clamp_min(1.0)
            all_ape = all_ape.masked_fill(~valid_gt_mask[None], float("inf"))
            # Assign each unmatched query to its nearest valid GT before applying duplicate penalties.
            best_gt_for_query = torch.argmin(all_ape, dim=1)

            matched_mask = torch.zeros(pred_logits.shape[1], dtype=torch.bool, device=device)
            matched_mask[src_idx] = True
            unmatched_mask = ~matched_mask
            if not unmatched_mask.any():
                continue

            for q_pos, gt_j in zip(src_idx.tolist(), tgt_idx.tolist()):
                gt_mask = gt_valid_b[gt_j]
                visible_count = gt_mask.sum()
                if visible_count <= 0.0 or visible_count > float(self.duplicate_short_visible_max):
                    continue

                diff_px = (points_b - gt_points_b[gt_j]) * scale
                point_error = torch.norm(diff_px, dim=-1)
                ape = (point_error * gt_mask).sum(dim=1) / visible_count.clamp_min(1.0)

                intersection = (valid_prob_b * gt_mask).sum(dim=1)
                union = valid_prob_b.sum(dim=1) + visible_count - intersection
                visible_iou = intersection / union.clamp_min(1e-6)
                overlap = ((valid_prob_b > 0.5) & (gt_mask > 0.5)).sum(dim=1)

                pos_ape = ape[q_pos]
                pos_visible_iou = visible_iou[q_pos]
                if pos_ape > float(self.duplicate_pos_ape_px) or pos_visible_iou < min_iou:
                    continue

                neg_worse = (ape > pos_ape + float(self.duplicate_ape_gap_px)) | (
                    ape > float(self.duplicate_pos_ape_px)
                )
                neg_mask = (
                    unmatched_mask
                    & (overlap >= min_overlap)
                    & (visible_iou >= min_iou)
                    & neg_worse
                    & (ape <= float(self.duplicate_neg_ape_px))
                    & (best_gt_for_query == int(gt_j))
                )
                candidates = torch.nonzero(neg_mask, as_tuple=False).flatten()
                if candidates.numel() == 0:
                    continue
                if candidates.numel() > max_pairs:
                    order = torch.argsort(logits_b.detach()[candidates], descending=True)[:max_pairs]
                    candidates = candidates[order]
                for q_neg in candidates:
                    losses.append(F.softplus(logits_b[q_neg] - logits_b[q_pos] + margin_logit))

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    def spurious_margin_loss(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        batch: dict,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Rank far unmatched spurious queries below reliable matched queries in selected GT-count images."""
        if self.spurious_margin_gain <= 0.0 or pred_valid_logits is None:
            return self._zero_like(pred_points)

        num_lanes = batch.get("num_lanes")
        if num_lanes is not None:
            num_lanes = torch.as_tensor(num_lanes, device=pred_logits.device).reshape(-1)
            if num_lanes.numel() != pred_logits.shape[0]:
                raise ValueError(
                    f"batch['num_lanes'] must have one value per image, got {num_lanes.numel()} "
                    f"vs B={pred_logits.shape[0]}."
                )

        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._pixel_scale_for(pred_points)
        gt_counts = self.spurious_gt_counts
        max_pairs = int(self.spurious_max_pairs_per_image)
        margin_logit = float(self.spurious_margin_logit)

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue

            if num_lanes is not None:
                image_lane_count = int(num_lanes[b].item())
            else:
                valid_b = gt_valid[b].detach().to(device=device)
                image_lane_count = int((valid_b.float().sum(dim=1) >= 2).sum().item())
            if image_lane_count not in gt_counts:
                continue

            points_b = pred_points[b].detach()
            logits_b = pred_logits[b]
            valid_prob_b = pred_valid_logits[b].detach().sigmoid().to(device=device, dtype=dtype)
            gt_points_b = gt_points[b].to(device=device, dtype=dtype)
            gt_valid_b = gt_valid[b].to(device=device, dtype=dtype)
            if gt_points_b.numel() == 0:
                continue

            valid_counts_all = gt_valid_b.sum(dim=1)
            valid_gt_mask = valid_counts_all >= 2
            if not valid_gt_mask.any():
                continue

            all_diff_px = (points_b[:, None] - gt_points_b[None]) * scale
            all_point_error = torch.norm(all_diff_px, dim=-1)
            all_ape = (all_point_error * gt_valid_b[None]).sum(dim=2) / valid_counts_all[None].clamp_min(1.0)
            all_ape = all_ape.masked_fill(~valid_gt_mask[None], float("inf"))

            intersection = (valid_prob_b[:, None] * gt_valid_b[None]).sum(dim=2)
            union = valid_prob_b.sum(dim=1, keepdim=True) + valid_counts_all[None] - intersection
            all_visible_iou = intersection / union.clamp_min(1e-6)
            all_visible_iou = all_visible_iou.masked_fill(~valid_gt_mask[None], 0.0)

            best_ape = all_ape.min(dim=1).values
            best_visible_iou = all_visible_iou.max(dim=1).values

            matched_mask = torch.zeros(pred_logits.shape[1], dtype=torch.bool, device=device)
            matched_mask[src_idx] = True
            unmatched_mask = ~matched_mask
            if not unmatched_mask.any():
                continue

            reliable_pos = []
            for q_pos, gt_j in zip(src_idx.tolist(), tgt_idx.tolist()):
                if not bool(valid_gt_mask[gt_j].item()):
                    continue
                pos_ape = all_ape[q_pos, gt_j]
                pos_visible_iou = all_visible_iou[q_pos, gt_j]
                if (
                    pos_ape <= float(self.spurious_pos_ape_px)
                    and pos_visible_iou >= float(self.spurious_pos_min_visible_iou)
                ):
                    reliable_pos.append(q_pos)
            if not reliable_pos:
                continue
            reliable_pos = torch.as_tensor(reliable_pos, device=device, dtype=torch.long)

            far_spurious = (best_ape > float(self.spurious_neg_min_ape_px)) | (
                best_visible_iou < float(self.spurious_neg_max_visible_iou)
            )
            duplicate_like = (best_ape <= float(self.spurious_duplicate_ape_px)) & (
                best_visible_iou >= float(self.spurious_duplicate_visible_iou)
            )
            neg_mask = unmatched_mask & far_spurious & ~duplicate_like
            candidates = torch.nonzero(neg_mask, as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            if candidates.numel() > max_pairs:
                order = torch.argsort(logits_b.detach()[candidates], descending=True)[:max_pairs]
                candidates = candidates[order]

            pos_grid = reliable_pos[:, None].expand(-1, candidates.numel()).reshape(-1)
            neg_grid = candidates[None, :].expand(reliable_pos.numel(), -1).reshape(-1)
            if neg_grid.numel() > max_pairs:
                pair_scores = logits_b.detach()[neg_grid] - logits_b.detach()[pos_grid]
                order = torch.argsort(pair_scores, descending=True)[:max_pairs]
                pos_grid = pos_grid[order]
                neg_grid = neg_grid[order]
            for q_pos, q_neg in zip(pos_grid, neg_grid):
                losses.append(F.softplus(logits_b[q_neg] - logits_b[q_pos] + margin_logit))

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    @staticmethod
    def _longest_true_run(mask: torch.Tensor) -> torch.Tensor:
        """Return the longest contiguous true run for each row in a B x K mask."""
        current = torch.zeros(mask.shape[0], device=mask.device, dtype=torch.long)
        best = torch.zeros_like(current)
        for k in range(mask.shape[1]):
            current = torch.where(mask[:, k], current + 1, torch.zeros_like(current))
            best = torch.maximum(best, current)
        return best

    def far_spurious_survival_loss(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        batch: dict,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Suppress decode-risk far-spurious unmatched queries in GT3/GT4 images only."""
        if self.far_spurious_survival_gain <= 0.0 or pred_valid_logits is None:
            return self._zero_like(pred_points)

        num_lanes = batch.get("num_lanes")
        if num_lanes is not None:
            num_lanes = torch.as_tensor(num_lanes, device=pred_logits.device).reshape(-1)
            if num_lanes.numel() != pred_logits.shape[0]:
                raise ValueError(
                    f"batch['num_lanes'] must have one value per image, got {num_lanes.numel()} "
                    f"vs B={pred_logits.shape[0]}."
                )

        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._pixel_scale_for(pred_points)
        gt_counts = self.far_spurious_gt_counts
        max_neg = int(self.far_spurious_max_neg_per_image)
        score_thr = float(self.far_spurious_score_thr)
        target_logit = pred_logits.new_tensor(score_thr).clamp(1e-6, 1.0 - 1e-6).logit()

        for b, (src_idx, _) in enumerate(indices):
            if num_lanes is not None:
                image_lane_count = int(num_lanes[b].item())
            else:
                valid_b = gt_valid[b].detach().to(device=device)
                image_lane_count = int((valid_b.float().sum(dim=1) >= 2).sum().item())
            if image_lane_count not in gt_counts:
                continue

            points_b = pred_points[b].detach()
            logits_b = pred_logits[b]
            valid_prob_b = pred_valid_logits[b].detach().sigmoid().to(device=device, dtype=dtype)
            gt_points_b = gt_points[b].to(device=device, dtype=dtype)
            gt_valid_b = gt_valid[b].to(device=device, dtype=dtype)
            if gt_points_b.numel() == 0:
                continue

            valid_counts_all = gt_valid_b.sum(dim=1)
            valid_gt_mask = valid_counts_all >= 2
            if not valid_gt_mask.any():
                continue

            all_diff_px = (points_b[:, None] - gt_points_b[None]) * scale
            all_point_error = torch.norm(all_diff_px, dim=-1)
            all_ape = (all_point_error * gt_valid_b[None]).sum(dim=2) / valid_counts_all[None].clamp_min(1.0)
            all_ape = all_ape.masked_fill(~valid_gt_mask[None], float("inf"))

            intersection = (valid_prob_b[:, None] * gt_valid_b[None]).sum(dim=2)
            union = valid_prob_b.sum(dim=1, keepdim=True) + valid_counts_all[None] - intersection
            all_visible_iou = intersection / union.clamp_min(1e-6)
            all_visible_iou = all_visible_iou.masked_fill(~valid_gt_mask[None], 0.0)

            best_ape = all_ape.min(dim=1).values
            best_visible_iou = all_visible_iou.max(dim=1).values

            matched_mask = torch.zeros(pred_logits.shape[1], dtype=torch.bool, device=device)
            matched_mask[src_idx] = True
            unmatched_mask = ~matched_mask
            if not unmatched_mask.any():
                continue

            visible_run = self._longest_true_run(valid_prob_b >= float(self.far_spurious_point_valid_thr))
            far_spurious = (best_ape > float(self.far_spurious_min_ape_px)) | (
                best_visible_iou < float(self.far_spurious_max_visible_iou)
            )
            decode_risk = (logits_b.detach().sigmoid() > float(self.far_spurious_min_score)) & (
                visible_run >= int(self.far_spurious_min_visible_run)
            )
            neg_mask = unmatched_mask & far_spurious & decode_risk
            candidates = torch.nonzero(neg_mask, as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            if candidates.numel() > max_neg:
                order = torch.argsort(logits_b.detach()[candidates], descending=True)[:max_neg]
                candidates = candidates[order]

            if self.far_spurious_loss_type == "softplus":
                losses.extend(F.softplus(logits_b[q_neg] - target_logit) for q_neg in candidates)
            else:
                losses.extend(F.relu(logits_b[q_neg] - target_logit) for q_neg in candidates)

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    def gt5_rank_consistency_loss(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        batch: dict,
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Keep all five matched GT5 queries ranked above high-risk unmatched queries."""
        if self.gt5_rank_consistency_gain <= 0.0:
            return self._zero_like(pred_points)

        num_lanes = batch.get("num_lanes")
        if num_lanes is not None:
            num_lanes = torch.as_tensor(num_lanes, device=pred_logits.device).reshape(-1)
            if num_lanes.numel() != pred_logits.shape[0]:
                raise ValueError(
                    f"batch['num_lanes'] must have one value per image, got {num_lanes.numel()} "
                    f"vs B={pred_logits.shape[0]}."
                )

        losses = []
        device = pred_logits.device
        max_pairs = int(self.gt5_rank_max_pairs_per_image)
        min_qminus_score = float(self.gt5_rank_min_qminus_score)
        margin_logit = float(self.gt5_rank_margin_logit)

        for b, (src_idx, _) in enumerate(indices):
            if num_lanes is not None:
                image_lane_count = int(num_lanes[b].item())
            else:
                valid_b = gt_valid[b].detach().to(device=device)
                image_lane_count = int((valid_b.float().sum(dim=1) >= 2).sum().item())
            if image_lane_count != 5 or src_idx.numel() < 5:
                continue

            logits_b = pred_logits[b]
            matched_mask = torch.zeros(pred_logits.shape[1], dtype=torch.bool, device=device)
            matched_mask[src_idx] = True
            unmatched_mask = ~matched_mask
            if not unmatched_mask.any():
                continue

            matched_logits = logits_b[src_idx]
            weakest_pos = matched_logits[torch.argmin(matched_logits.detach())]
            candidates = torch.nonzero(unmatched_mask, as_tuple=False).flatten()
            score_mask = logits_b.detach()[candidates].sigmoid() >= min_qminus_score
            candidates = candidates[score_mask]
            if candidates.numel() == 0:
                continue
            order = torch.argsort(logits_b.detach()[candidates], descending=True)[:max_pairs]
            candidates = candidates[order]
            for q_neg in candidates:
                losses.append(F.softplus(logits_b[q_neg] - weakest_pos + margin_logit))

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    def gt3_extra_survival_loss(
        self,
        pred_logits: torch.Tensor,
        batch: dict,
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Rank the top unmatched GT3 query below the weakest matched query."""
        if self.gt3_extra_survival_gain <= 0.0:
            return pred_logits.sum() * 0.0

        target_counts = self.target_lane_count(pred_logits, batch, gt_valid).detach()
        losses = []
        device = pred_logits.device
        num_queries = pred_logits.shape[1]
        margin_logit = float(self.gt3_extra_margin_logit)
        topk = int(self.gt3_extra_topk)

        for b, (src_idx, _) in enumerate(indices):
            if int(target_counts[b].item()) != 3 or src_idx.numel() < 3:
                continue

            matched_mask = torch.zeros(num_queries, dtype=torch.bool, device=device)
            matched_mask[src_idx.to(device=device)] = True
            if not matched_mask.any() or matched_mask.all():
                continue

            pos_scores = pred_logits[b, matched_mask]
            neg_scores = pred_logits[b, ~matched_mask]
            if pos_scores.numel() == 0 or neg_scores.numel() == 0:
                continue

            min_pos_score = pos_scores.min()
            k = min(topk, neg_scores.numel())
            top_extra_scores = neg_scores.topk(k).values
            losses.append(F.relu(top_extra_scores - min_pos_score + margin_logit).mean())

        return torch.stack(losses).mean() if losses else pred_logits.sum() * 0.0

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
        lane_balanced_point_loss = self.lane_balanced_point_loss(pred_points, gt_points, gt_valid, indices)
        gt4_lane_balanced_point_loss = self.gt4_lane_balanced_point_loss(
            pred_points, batch, gt_points, gt_valid, indices
        )
        point_valid_loss = self.point_valid_loss(pred_valid_logits, pred_points, gt_valid, indices)
        short_valid_recall_loss = self.short_valid_recall_loss(
            pred_valid_logits, pred_points, gt_points, gt_valid, indices
        )
        smooth_loss = self.smooth_loss(pred_points, gt_valid, indices)
        curve_loss = self.curve_loss(pred_points, gt_points, gt_valid, indices)
        count_loss, count_under5_loss = self.count_losses(pred_logits, batch, gt_valid)
        duplicate_margin_loss = self.duplicate_margin_loss(
            pred_logits, pred_points, pred_valid_logits, batch, gt_points, gt_valid, indices
        )
        spurious_margin_loss = self.spurious_margin_loss(
            pred_logits, pred_points, pred_valid_logits, batch, gt_points, gt_valid, indices
        )
        far_spurious_survival_loss = self.far_spurious_survival_loss(
            pred_logits, pred_points, pred_valid_logits, batch, gt_points, gt_valid, indices
        )
        gt5_rank_consistency_loss = self.gt5_rank_consistency_loss(pred_logits, pred_points, batch, gt_valid, indices)
        gt3_extra_survival_loss = self.gt3_extra_survival_loss(pred_logits, batch, gt_valid, indices)

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
            + self.lane_balanced_point_gain * lane_balanced_point_loss
            + self.gt4_lane_balanced_point_gain * gt4_lane_balanced_point_loss
            + self.point_valid_gain * point_valid_loss
            + self.short_valid_recall_gain * short_valid_recall_loss
            + self.smooth_gain * smooth_loss
            + self.curve_gain * curve_loss
            + self.mask_gain * mask_loss
            + self.edge_gain * edge_loss
            + self.count_gain * count_loss
            + self.count_under5_gain * count_under5_loss
            + self.duplicate_margin_gain * duplicate_margin_loss
            + self.spurious_margin_gain * spurious_margin_loss
            + self.far_spurious_survival_gain * far_spurious_survival_loss
            + self.gt5_rank_consistency_gain * gt5_rank_consistency_loss
            + self.gt3_extra_survival_gain * gt3_extra_survival_loss
        )
        loss_items = torch.stack(
            (
                exist_loss.detach(),
                point_loss.detach(),
                lane_balanced_point_loss.detach(),
                gt4_lane_balanced_point_loss.detach(),
                point_valid_loss.detach(),
                short_valid_recall_loss.detach(),
                smooth_loss.detach(),
                curve_loss.detach(),
                mask_loss.detach(),
                edge_loss.detach(),
                count_loss.detach(),
                count_under5_loss.detach(),
                duplicate_margin_loss.detach(),
                spurious_margin_loss.detach(),
                far_spurious_survival_loss.detach(),
                gt5_rank_consistency_loss.detach(),
                gt3_extra_survival_loss.detach(),
            )
        )
        return total, loss_items
