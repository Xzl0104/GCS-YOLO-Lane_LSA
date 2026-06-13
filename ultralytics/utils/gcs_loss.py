# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions for GCS-YOLO-Lane structured lane training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.gcs_matcher import GCSHungarianMatcher
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, normalize_imgsz


class GCSLoss(nn.Module):
    """Hungarian-matched loss for query-based structured lane predictions."""

    loss_names = (
        "exist_loss",
        "point_loss",
        "point_valid_loss",
        "line_iou_loss",
        "count_cls_loss",
        "count_sum_loss",
        "quality_loss",
    )

    def __init__(
        self,
        model=None,
        lambda_exist: float | None = None,
        lambda_point: float | None = None,
        lambda_point_valid: float | None = None,
        lambda_line_iou: float | None = None,
        lambda_count_cls: float | None = None,
        lambda_count_sum: float | None = None,
        lambda_quality: float | None = None,
        line_iou_width_px: float | None = None,
        count_head_warmup_epochs: float | None = None,
        count_min_gt_points: int | None = None,
        count_boundary_gt5_pos_weight: float | None = None,
        quality_dist_thr_px: float | None = None,
        quality_neg_weight: float | None = None,
        quality_hard_negative_weight: float | None = None,
        quality_duplicate_negative_weight: float | None = None,
        quality_hard_negative_from_head: bool | str | None = None,
        exist_pos_weight: float | None = None,
        exist_focal_gamma: float | None = None,
        exist_focal_alpha: float | None = None,
        hard_negative_quality_thr: float | None = None,
        hard_negative_topk: int | None = None,
        hard_negative_exist_weight: float | None = None,
        duplicate_negative_exist_weight: float | None = None,
        duplicate_dist_thr_px: float | None = None,
        duplicate_iou_thr: float | None = None,
        point_valid_pos_weight_max: float | None = None,
        point_valid_gt5_pos_weight: float | None = None,
        point_valid_unmatched_weight: float | None = None,
        point_valid_hard_negative_weight: float | None = None,
        point_valid_duplicate_negative_weight: float | None = None,
        gt5_edge_loss_weight: float | None = None,
        candidate_gt5_edge_weight: float | None = None,
        point_valid_gt5_edge_continuity: float | None = None,
        point_valid_gt5_edge_continuity_thr: float | None = None,
        point_valid_gt5_edge_segment: float | None = None,
        point_valid_gt5_edge_segment_thr: float | None = None,
        point_valid_gt5_edge_segment_min_points: int | None = None,
        hard_loss_file: str | None = None,
        hard_loss_lane_counts: str | list[int] | tuple[int, ...] | set[int] | None = None,
        hard_edge_loss_weight_by_count: str | dict[int, float] | None = None,
        hard_edge_loss_terms: str | list[str] | tuple[str, ...] | None = None,
        hard_edge_only: bool | str | None = None,
        exist_margin: float | None = None,
        exist_pos_margin: float | None = None,
        exist_neg_margin: float | None = None,
        point_valid_neg: float | None = None,
        point_valid_neg_thr: float | None = None,
        point_invalid_x: float | None = None,
        exist_quality_alpha: float | None = None,
        exist_quality_mode: str | None = None,
        exist_quality_tau: float | None = None,
        exist_quality_floor: float | None = None,
        exist_quality_pos_px: float | None = None,
        exist_quality_neg_px: float | None = None,
        exist_quality_lane_iou_alpha: float | None = None,
        match_min_overlap: int | None = None,
        match_max_x_dist: float | None = None,
        match_gate_px: float | None = None,
        image_size=None,
    ):
        """Initialize GCS-YOLO-Lane loss weights and Hungarian matcher."""
        super().__init__()
        args = model if isinstance(model, dict) else getattr(model, "args", None)

        self.exist_gain = float(lambda_exist if lambda_exist is not None else self._arg(args, "gcs_exist", 1.0))
        self.point_gain = float(lambda_point if lambda_point is not None else self._arg(args, "gcs_point", 5.0))
        self.point_valid_gain = float(
            lambda_point_valid if lambda_point_valid is not None else self._arg(args, "gcs_point_valid", 0.5)
        )
        self.line_iou_gain = float(
            lambda_line_iou if lambda_line_iou is not None else self._arg(args, "gcs_line_iou", 0.3)
        )
        self.point_mode = self._infer_point_mode(model, args)
        self.count_cls_gain = float(
            lambda_count_cls if lambda_count_cls is not None else self._arg(args, "gcs_count_cls", 0.3)
        )
        self.count_sum_gain = float(
            lambda_count_sum if lambda_count_sum is not None else self._arg(args, "gcs_count_sum", 0.03)
        )
        self.count_sum_normalize = bool(self._arg(args, "gcs_count_sum_normalize", True))
        self.quality_gain = float(
            lambda_quality if lambda_quality is not None else self._arg(args, "gcs_quality", 0.4)
        )
        self.quality_dist_thr_px = float(
            quality_dist_thr_px
            if quality_dist_thr_px is not None
            else self._arg(args, "gcs_quality_dist_thr_px", 20.0)
        )
        self.quality_neg_weight = float(
            quality_neg_weight
            if quality_neg_weight is not None
            else self._arg(args, "gcs_quality_neg_weight", 0.5)
        )
        self.quality_hard_negative_weight = float(
            quality_hard_negative_weight
            if quality_hard_negative_weight is not None
            else self._arg(args, "gcs_quality_hard_negative_weight", 1.0)
        )
        self.quality_duplicate_negative_weight = float(
            quality_duplicate_negative_weight
            if quality_duplicate_negative_weight is not None
            else self._arg(args, "gcs_quality_duplicate_negative_weight", 1.5)
        )
        self.quality_hard_negative_from_head = self._parse_bool(
            quality_hard_negative_from_head
            if quality_hard_negative_from_head is not None
            else self._arg(args, "gcs_quality_hard_negative_from_head", False),
            default=False,
        )
        self.count_head_warmup_epochs = float(
            count_head_warmup_epochs
            if count_head_warmup_epochs is not None
            else self._arg(args, "gcs_count_head_warmup_epochs", 5.0)
        )
        self.count_min_gt_points = int(
            count_min_gt_points
            if count_min_gt_points is not None
            else self._arg(args, "gcs_count_min_gt_points", 1)
        )
        self.count_cls_weights = (
            float(self._arg(args, "gcs_count_cls_w2", 0.5)),
            float(self._arg(args, "gcs_count_cls_w3", 1.2)),
            float(self._arg(args, "gcs_count_cls_w4", 1.4)),
            float(self._arg(args, "gcs_count_cls_w5", 1.8)),
        )
        self.count_boundary_gain = float(self._arg(args, "gcs_count_boundary", 0.05))
        self.count_boundary_label_smoothing = float(self._arg(args, "gcs_count_boundary_label_smoothing", 0.05))
        self.count_boundary_gt5_pos_weight = float(
            count_boundary_gt5_pos_weight
            if count_boundary_gt5_pos_weight is not None
            else self._arg(args, "gcs_count_boundary_gt5_pos_weight", 1.15)
        )
        self.updates = 0
        self.exist_pos_weight = float(
            exist_pos_weight if exist_pos_weight is not None else self._arg(args, "gcs_exist_pos_weight", 1.0)
        )
        self.exist_focal_gamma = float(
            exist_focal_gamma if exist_focal_gamma is not None else self._arg(args, "gcs_exist_focal_gamma", 2.0)
        )
        self.exist_focal_alpha = float(
            exist_focal_alpha if exist_focal_alpha is not None else self._arg(args, "gcs_exist_focal_alpha", -1.0)
        )
        self.hard_negative_quality_thr = float(
            hard_negative_quality_thr
            if hard_negative_quality_thr is not None
            else self._arg(args, "gcs_hard_negative_quality_thr", 0.5)
        )
        self.hard_negative_topk = int(
            hard_negative_topk
            if hard_negative_topk is not None
            else self._arg(args, "gcs_hard_negative_topk", 2)
        )
        self.hard_negative_exist_weight = float(
            hard_negative_exist_weight
            if hard_negative_exist_weight is not None
            else self._arg(args, "gcs_hard_negative_exist_weight", 4.0)
        )
        self.duplicate_negative_exist_weight = float(
            duplicate_negative_exist_weight
            if duplicate_negative_exist_weight is not None
            else self._arg(args, "gcs_duplicate_negative_exist_weight", 4.0)
        )
        self.duplicate_dist_thr_px = float(
            duplicate_dist_thr_px
            if duplicate_dist_thr_px is not None
            else self._arg(args, "gcs_duplicate_dist_thr_px", 25.0)
        )
        self.duplicate_iou_thr = float(
            duplicate_iou_thr
            if duplicate_iou_thr is not None
            else self._arg(args, "gcs_duplicate_iou_thr", 0.30)
        )
        self.exist_margin_gain = float(
            exist_margin if exist_margin is not None else self._arg(args, "gcs_exist_margin", 0.5)
        )
        self.exist_pos_margin = float(
            exist_pos_margin if exist_pos_margin is not None else self._arg(args, "gcs_exist_pos_margin", 0.55)
        )
        self.exist_neg_margin = float(
            exist_neg_margin if exist_neg_margin is not None else self._arg(args, "gcs_exist_neg_margin", 0.20)
        )
        self.point_valid_pos_weight_max = float(
            point_valid_pos_weight_max
            if point_valid_pos_weight_max is not None
            else self._arg(args, "gcs_point_valid_pos_weight_max", 10.0)
        )
        self.point_valid_gt5_pos_weight = float(
            point_valid_gt5_pos_weight
            if point_valid_gt5_pos_weight is not None
            else self._arg(args, "gcs_point_valid_gt5_pos_weight", 2.0)
        )
        self.point_valid_unmatched_weight = float(
            point_valid_unmatched_weight
            if point_valid_unmatched_weight is not None
            else self._arg(args, "gcs_point_valid_unmatched_weight", 0.35)
        )
        self.point_valid_hard_negative_weight = float(
            point_valid_hard_negative_weight
            if point_valid_hard_negative_weight is not None
            else self._arg(args, "gcs_point_valid_hard_negative_weight", 1.25)
        )
        self.point_valid_duplicate_negative_weight = float(
            point_valid_duplicate_negative_weight
            if point_valid_duplicate_negative_weight is not None
            else self._arg(args, "gcs_point_valid_duplicate_negative_weight", 1.5)
        )
        self.gt5_edge_loss_weight = float(
            gt5_edge_loss_weight
            if gt5_edge_loss_weight is not None
            else self._arg(args, "gcs_gt5_edge_loss_weight", 1.15)
        )
        self.candidate_gt5_edge_weight = float(
            candidate_gt5_edge_weight
            if candidate_gt5_edge_weight is not None
            else self._arg(args, "gcs_candidate_gt5_edge_weight", 1.10)
        )
        self.point_valid_gt5_edge_continuity = float(
            point_valid_gt5_edge_continuity
            if point_valid_gt5_edge_continuity is not None
            else self._arg(args, "gcs_point_valid_gt5_edge_continuity", 0.05)
        )
        self.point_valid_gt5_edge_continuity_thr = float(
            point_valid_gt5_edge_continuity_thr
            if point_valid_gt5_edge_continuity_thr is not None
            else self._arg(args, "gcs_point_valid_gt5_edge_continuity_thr", 0.55)
        )
        self.point_valid_gt5_edge_segment = float(
            point_valid_gt5_edge_segment
            if point_valid_gt5_edge_segment is not None
            else self._arg(args, "gcs_point_valid_gt5_edge_segment", 0.0)
        )
        self.point_valid_gt5_edge_segment_thr = float(
            point_valid_gt5_edge_segment_thr
            if point_valid_gt5_edge_segment_thr is not None
            else self._arg(args, "gcs_point_valid_gt5_edge_segment_thr", 0.65)
        )
        self.point_valid_gt5_edge_segment_min_points = int(
            point_valid_gt5_edge_segment_min_points
            if point_valid_gt5_edge_segment_min_points is not None
            else self._arg(args, "gcs_point_valid_gt5_edge_segment_min_points", 5)
        )
        self.hard_edge_loss_weight_by_count = self._parse_count_weight_map(
            hard_edge_loss_weight_by_count
            if hard_edge_loss_weight_by_count is not None
            else self._arg(args, "gcs_hard_edge_loss_weight_by_count", "4:1.15,5:1.6"),
            "gcs_hard_edge_loss_weight_by_count",
        )
        self.hard_edge_loss_terms = self._parse_hard_edge_loss_terms(
            hard_edge_loss_terms
            if hard_edge_loss_terms is not None
            else self._arg(args, "gcs_hard_edge_loss_terms", "exist,point,point_valid,line_iou")
        )
        self.hard_edge_only = self._parse_bool(
            hard_edge_only if hard_edge_only is not None else self._arg(args, "gcs_hard_edge_only", True),
            default=True,
        )
        self.hard_loss_file = str(
            hard_loss_file if hard_loss_file is not None else self._arg(args, "gcs_hard_loss_file", "") or ""
        ).strip()
        self.hard_loss_ids = self._load_hard_loss_ids(self.hard_loss_file)
        self.hard_loss_path_like_ids = tuple(x for x in self.hard_loss_ids if "/" in x)
        self.hard_loss_lane_counts = self._parse_lane_count_filter(
            hard_loss_lane_counts
            if hard_loss_lane_counts is not None
            else self._arg(args, "gcs_hard_loss_lane_counts", "")
        )
        self.point_valid_neg_gain = float(
            point_valid_neg if point_valid_neg is not None else self._arg(args, "gcs_point_valid_neg", 0.25)
        )
        self.point_valid_neg_thr = float(
            point_valid_neg_thr
            if point_valid_neg_thr is not None
            else self._arg(args, "gcs_point_valid_neg_thr", 0.20)
        )
        self.point_invalid_x_gain = float(
            point_invalid_x if point_invalid_x is not None else self._arg(args, "gcs_point_invalid_x", 0.05)
        )
        nonnegative = {
            "gcs_exist_margin": self.exist_margin_gain,
            "gcs_point_valid_neg": self.point_valid_neg_gain,
            "gcs_point_invalid_x": self.point_invalid_x_gain,
            "gcs_count_cls": self.count_cls_gain,
            "gcs_count_boundary": self.count_boundary_gain,
            "gcs_count_sum": self.count_sum_gain,
            "gcs_quality": self.quality_gain,
            "gcs_quality_dist_thr_px": self.quality_dist_thr_px,
            "gcs_quality_hard_negative_weight": self.quality_hard_negative_weight,
            "gcs_quality_duplicate_negative_weight": self.quality_duplicate_negative_weight,
            "gcs_hard_negative_exist_weight": self.hard_negative_exist_weight,
            "gcs_duplicate_negative_exist_weight": self.duplicate_negative_exist_weight,
            "gcs_duplicate_dist_thr_px": self.duplicate_dist_thr_px,
            "gcs_point_valid_unmatched_weight": self.point_valid_unmatched_weight,
            "gcs_point_valid_hard_negative_weight": self.point_valid_hard_negative_weight,
            "gcs_point_valid_duplicate_negative_weight": self.point_valid_duplicate_negative_weight,
            "gcs_count_head_warmup_epochs": self.count_head_warmup_epochs,
            "gcs_point_valid_gt5_edge_continuity": self.point_valid_gt5_edge_continuity,
            "gcs_point_valid_gt5_edge_segment": self.point_valid_gt5_edge_segment,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}.")
        fractions = {
            "gcs_exist_pos_margin": self.exist_pos_margin,
            "gcs_exist_neg_margin": self.exist_neg_margin,
            "gcs_point_valid_neg_thr": self.point_valid_neg_thr,
            "gcs_quality_neg_weight": self.quality_neg_weight,
            "gcs_hard_negative_quality_thr": self.hard_negative_quality_thr,
            "gcs_count_boundary_label_smoothing": self.count_boundary_label_smoothing,
            "gcs_point_valid_gt5_edge_continuity_thr": self.point_valid_gt5_edge_continuity_thr,
            "gcs_point_valid_gt5_edge_segment_thr": self.point_valid_gt5_edge_segment_thr,
            "gcs_duplicate_iou_thr": self.duplicate_iou_thr,
        }
        for name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if self.point_valid_gt5_pos_weight < 1.0:
            raise ValueError(
                f"gcs_point_valid_gt5_pos_weight must be >= 1.0, got {self.point_valid_gt5_pos_weight}."
            )
        if self.gt5_edge_loss_weight < 1.0:
            raise ValueError(f"gcs_gt5_edge_loss_weight must be >= 1.0, got {self.gt5_edge_loss_weight}.")
        if self.count_boundary_gt5_pos_weight < 1.0:
            raise ValueError(
                "gcs_count_boundary_gt5_pos_weight must be >= 1.0, "
                f"got {self.count_boundary_gt5_pos_weight}."
            )
        if self.candidate_gt5_edge_weight < 1.0:
            raise ValueError(
                f"gcs_candidate_gt5_edge_weight must be >= 1.0, got {self.candidate_gt5_edge_weight}."
            )
        if self.exist_pos_margin < self.exist_neg_margin:
            raise ValueError(
                "gcs_exist_pos_margin must be >= gcs_exist_neg_margin "
                f"({self.exist_pos_margin} < {self.exist_neg_margin})."
            )
        if self.count_min_gt_points <= 0:
            raise ValueError(f"gcs_count_min_gt_points must be > 0, got {self.count_min_gt_points}.")
        if self.hard_negative_topk < 0:
            raise ValueError(f"gcs_hard_negative_topk must be >= 0, got {self.hard_negative_topk}.")
        if self.point_valid_gt5_edge_segment_min_points <= 0:
            raise ValueError(
                "gcs_point_valid_gt5_edge_segment_min_points must be > 0, "
                f"got {self.point_valid_gt5_edge_segment_min_points}."
            )
        self.line_iou_width_px = float(
            line_iou_width_px if line_iou_width_px is not None else self._arg(args, "gcs_line_iou_width_px", 15.0)
        )
        if self.line_iou_width_px <= 0.0:
            raise ValueError(f"gcs_line_iou_width_px must be > 0, got {self.line_iou_width_px}.")
        if self.quality_gain > 0.0 and self.quality_dist_thr_px <= 0.0:
            raise ValueError(f"gcs_quality_dist_thr_px must be > 0 when gcs_quality is enabled, got {self.quality_dist_thr_px}.")
        self.exist_quality_alpha = float(
            exist_quality_alpha if exist_quality_alpha is not None else self._arg(args, "gcs_exist_quality_alpha", 1.0)
        )
        self.exist_quality_lane_iou_alpha = float(
            exist_quality_lane_iou_alpha
            if exist_quality_lane_iou_alpha is not None
            else self._arg(args, "gcs_exist_quality_lane_iou_alpha", 1.0)
        )
        if not 0.0 <= self.exist_quality_lane_iou_alpha <= 1.0:
            raise ValueError(
                "gcs_exist_quality_lane_iou_alpha must be in [0, 1], "
                f"got {self.exist_quality_lane_iou_alpha}."
            )
        if self.point_mode != "fixed_y" and (
            self.line_iou_gain > 0.0 or self.exist_quality_lane_iou_alpha > 0.0 or self.quality_gain > 0.0
        ):
            raise ValueError(
                "Current GCS LineIoU implementation is fixed-y only because it compares horizontal strips at shared "
                "y anchors. Set gcs_line_iou=0.0, gcs_exist_quality_lane_iou_alpha=0.0, and gcs_quality=0.0 "
                "for free-point mode, or implement a free-point LineIoU that first resamples lanes onto common y anchors."
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
            else self._arg(args, "gcs_exist_quality_neg_px", 25.0)
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
            cost_exist=float(self._arg(args, "gcs_cost_exist", 0.1)),
            image_size=self.image_size,
            min_overlap=int(match_min_overlap if match_min_overlap is not None else self._arg(args, "gcs_match_min_overlap", 2)),
            max_x_dist=float(match_max_x_dist if match_max_x_dist is not None else self._arg(args, "gcs_match_max_x_dist", 0.0)),
            match_gate_px=float(match_gate_px if match_gate_px is not None else self._arg(args, "gcs_match_gate_px", 160.0)),
            point_mode=self.point_mode,
        )

    @staticmethod
    def _arg(args, name: str, default):
        """Read a config value from an Ultralytics namespace or dict."""
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    @staticmethod
    def _parse_bool(value: Any, default: bool = False) -> bool:
        """Parse bool-like config values from YAML, argparse, or dict overrides."""
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off", "none", ""}:
            return False
        raise ValueError(f"Expected a boolean value, got {value!r}.")

    @staticmethod
    def _normalize_hard_edge_term(term: Any) -> str:
        """Normalize hard-edge loss term names."""
        text = str(term or "").strip().lower().replace("-", "_")
        if text.endswith("_loss"):
            text = text[: -len("_loss")]
        aliases = {
            "valid": "point_valid",
            "visibility": "point_valid",
            "point_visibility": "point_valid",
            "lineiou": "line_iou",
            "iou": "line_iou",
        }
        return aliases.get(text, text)

    @classmethod
    def _parse_hard_edge_loss_terms(cls, value: Any) -> set[str]:
        """Parse the loss terms that receive hard-edge weighting."""
        if value is None or value is False:
            return set()
        if isinstance(value, (list, tuple, set)):
            tokens = [str(x) for x in value]
        else:
            text = str(value).strip()
            if not text or text.lower() in {"none", "false", "off", "no"}:
                return set()
            tokens = [x for x in re.split(r"[,;\s]+", text) if x]

        allowed = {"exist", "point", "point_valid", "line_iou", "quality"}
        terms = {cls._normalize_hard_edge_term(token) for token in tokens}
        invalid = sorted(terms - allowed)
        if invalid:
            raise ValueError(
                "gcs_hard_edge_loss_terms supports only exist, point, point_valid, line_iou, and quality; "
                f"got {invalid}."
            )
        return terms

    @staticmethod
    def _parse_count_weight_map(value: Any, name: str) -> dict[int, float]:
        """Parse per-GT-count hard-edge multipliers such as '4:1.15,5:1.6'."""
        if value is None or value is False:
            return {}
        if isinstance(value, dict):
            items = value.items()
        else:
            text = str(value).strip()
            if not text or text.lower() in {"none", "false", "off", "no"}:
                return {}
            pairs = []
            for token in re.split(r"[,;\s]+", text):
                if not token:
                    continue
                if ":" in token:
                    count, weight = token.split(":", 1)
                elif "=" in token:
                    count, weight = token.split("=", 1)
                else:
                    raise ValueError(f"{name} entries must use count:weight or count=weight, got {token!r}.")
                pairs.append((count, weight))
            items = pairs

        weights = {}
        for count, weight in items:
            count_i = int(str(count).strip())
            weight_f = float(str(weight).strip())
            if count_i <= 0:
                raise ValueError(f"{name} count keys must be positive, got {count_i}.")
            if weight_f < 1.0:
                raise ValueError(f"{name} multipliers must be >= 1.0, got {weight_f} for count {count_i}.")
            weights[count_i] = weight_f
        return weights

    @staticmethod
    def _resolve_manifest_path(file_arg: str) -> Path:
        """Resolve a manifest path relative to cwd first, then the project root."""
        path = Path(file_arg)
        if path.is_absolute():
            return path
        cwd_candidate = Path.cwd() / path
        if cwd_candidate.exists():
            return cwd_candidate
        return Path(__file__).resolve().parents[2] / path

    @staticmethod
    def _normalize_sample_id(value: Any) -> str:
        """Normalize image/label identifiers used by hard loss manifests."""
        return str(value).strip().strip("\"'").replace("\\", "/")

    @staticmethod
    def _collect_json_strings(value: Any) -> list[str]:
        """Collect string leaves from a permissive JSON hard-sample manifest."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            items: list[str] = []
            for v in value.values():
                items.extend(GCSLoss._collect_json_strings(v))
            return items
        if isinstance(value, (list, tuple)):
            items = []
            for v in value:
                items.extend(GCSLoss._collect_json_strings(v))
            return items
        return []

    @classmethod
    def _sample_id_variants(cls, value: Any) -> set[str]:
        """Return exact and stem variants for one sample identifier."""
        norm = cls._normalize_sample_id(value)
        if not norm:
            return set()
        variants = {norm, norm.lstrip("./")}
        # Path-like ids can end with non-unique frame names such as 20.jpg, so do not add a bare stem.
        if "/" not in norm:
            variants.add(Path(norm).stem)
        return variants

    def _load_hard_loss_ids(self, file_arg: str) -> set[str]:
        """Load optional hard-loss image/label identifiers from txt/json."""
        if not file_arg:
            return set()
        path = self._resolve_manifest_path(file_arg)
        if not path.exists():
            raise FileNotFoundError(f"gcs_hard_loss_file does not exist: {path}")

        if path.suffix.lower() == ".json":
            values = self._collect_json_strings(json.loads(path.read_text(encoding="utf-8")))
        else:
            values = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]

        ids: set[str] = set()
        for value in values:
            ids.update(self._sample_id_variants(value))
        if not ids:
            raise ValueError(f"gcs_hard_loss_file is empty or contains no usable sample ids: {path}")
        return ids

    @staticmethod
    def _parse_lane_count_filter(value: Any) -> set[int]:
        """Parse an optional GT lane-count filter such as '5' or '4,5'."""
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            tokens = [str(x).strip() for x in value]
        else:
            tokens = str(value).replace(",", " ").split()
        counts: set[int] = set()
        for token in tokens:
            if not token:
                continue
            try:
                count = int(token)
            except ValueError as exc:
                raise ValueError(f"gcs_hard_loss_lane_counts entries must be integers, got {token!r}.") from exc
            if count <= 0:
                raise ValueError(f"gcs_hard_loss_lane_counts entries must be > 0, got {count}.")
            counts.add(count)
        return counts

    def _lane_count_mask(
        self,
        gt_valid: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return a B-vector mask for allowed GT lane counts, or None when disabled."""
        if not self.hard_loss_lane_counts:
            return None
        if gt_valid is None:
            raise ValueError("gcs_hard_loss_lane_counts is set, but gt_valid is unavailable for lane-count filtering.")
        if isinstance(gt_valid, torch.Tensor):
            if gt_valid.ndim != 3 or gt_valid.shape[0] != batch_size:
                raise ValueError(
                    f"Tensor gt_valid must have shape B x N x K with B={batch_size}, got {tuple(gt_valid.shape)}."
                )
            lane_counts = (gt_valid.detach().to(device=device).float().sum(dim=-1) >= 2).sum(dim=-1)
        elif isinstance(gt_valid, (list, tuple)):
            if len(gt_valid) != batch_size:
                raise ValueError(f"gt_valid must contain one tensor per image, got {len(gt_valid)} vs B={batch_size}.")
            counts = []
            for valid in gt_valid:
                valid = torch.as_tensor(valid, device=device)
                if valid.ndim != 2:
                    raise ValueError(f"Each GT lane_valid must have shape N x K, got {tuple(valid.shape)}.")
                counts.append((valid.float().sum(dim=-1) >= 2).sum())
            lane_counts = torch.stack(counts) if counts else torch.empty(0, device=device, dtype=torch.long)
        else:
            raise TypeError(f"gt_valid must be a tensor, list, or tuple, got {type(gt_valid).__name__}.")
        allowed = torch.zeros(batch_size, device=device, dtype=torch.bool)
        for count in self.hard_loss_lane_counts:
            allowed |= lane_counts.eq(int(count))
        return allowed

    @staticmethod
    def _batch_string_values(value: Any, batch_size: int, key: str) -> list[str]:
        """Normalize an optional batch path field to one string per image."""
        if value is None:
            return [""] * batch_size
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = [str(x) for x in value]
        else:
            values = [str(x) for x in list(value)] if hasattr(value, "__iter__") else [str(value)]
        if len(values) != batch_size:
            raise ValueError(f"batch[{key!r}] must contain B={batch_size} entries, got {len(values)}.")
        return values

    def hard_loss_mask(
        self,
        batch: dict,
        batch_size: int,
        device: torch.device,
        gt_valid: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Return a B-vector mask for images matched by gcs_hard_loss_file."""
        mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if not self.hard_loss_ids:
            return mask
        if not any(key in batch for key in ("im_file", "path", "label_file")):
            raise KeyError(
                "gcs_hard_loss_file is set, but the batch has no im_file/path/label_file fields for manifest matching."
            )
        im_files = self._batch_string_values(batch.get("im_file", batch.get("path")), batch_size, "im_file")
        label_files = self._batch_string_values(batch.get("label_file"), batch_size, "label_file")
        for i, (im_file, label_file) in enumerate(zip(im_files, label_files)):
            candidates: set[str] = set()
            for value in (im_file, label_file):
                candidates.update(self._sample_id_variants(value))
            if candidates & self.hard_loss_ids:
                mask[i] = True
                continue
            if any(candidate.endswith(hard_id) for candidate in candidates for hard_id in self.hard_loss_path_like_ids):
                mask[i] = True
        lane_count_mask = self._lane_count_mask(gt_valid, batch_size, device)
        if lane_count_mask is not None:
            mask &= lane_count_mask
        return mask

    @staticmethod
    def _hard_mask_value(hard_loss_mask: torch.Tensor | None, index: int) -> bool:
        """Read one Python bool from an optional hard-loss mask."""
        if hard_loss_mask is None:
            return False
        return bool(hard_loss_mask[index].detach().item())

    def _hard_edge_loss_enabled_for(self, term: str | None, hard_image: bool) -> bool:
        """Return True when a hard image and term should receive hard-edge weighting."""
        if not hard_image or not self.hard_edge_loss_weight_by_count:
            return False
        return self._normalize_hard_edge_term(term) in self.hard_edge_loss_terms

    def _hard_edge_loss_multiplier(self, lane_count: int, term: str | None, hard_image: bool) -> float:
        """Return the count-specific hard-edge multiplier for one image and loss term."""
        if not self._hard_edge_loss_enabled_for(term, hard_image):
            return 1.0
        return float(self.hard_edge_loss_weight_by_count.get(int(lane_count), 1.0))

    @staticmethod
    def _normalize_point_mode(value) -> str:
        """Normalize free/fixed-y point mode aliases."""
        mode = str(value or "free").lower()
        return "fixed_y" if mode in {"fixed-y", "fixedy"} else mode

    @classmethod
    def _infer_point_mode(cls, model, args) -> str:
        """Infer whether GCS losses should supervise full xy points or fixed-y x only."""
        explicit = cls._arg(args, "gcs_point_mode", None)
        if explicit is not None:
            return cls._normalize_point_mode(explicit)

        for module in getattr(model, "modules", lambda: [])():
            if hasattr(module, "point_mode"):
                return cls._normalize_point_mode(getattr(module, "point_mode"))

        yaml = model if isinstance(model, dict) else getattr(model, "yaml", None)
        if isinstance(yaml, dict):
            for value in yaml.get("head", []):
                if not isinstance(value, (list, tuple)) or len(value) < 4:
                    continue
                module = str(value[2]).lower()
                if "gcslanehead" not in module:
                    continue
                args = value[3]
                if not isinstance(args, (list, tuple)):
                    continue
                for arg in args:
                    if str(arg).lower() in {"fixed_y", "fixed-y", "fixedy", "free"}:
                        return cls._normalize_point_mode(arg)
        return "free"

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
        """Return x/y pixel scales for quality-aware existence targets."""
        if image_size is None or image_size == "":
            return 1.0, 1.0
        h, w = normalize_imgsz(image_size)
        return float(w), float(h)

    def _scale_for(self, ref: torch.Tensor) -> torch.Tensor:
        """Return x/y point-loss weights on the same device and dtype as ref."""
        return self.point_scale.to(device=ref.device, dtype=ref.dtype)

    def _pixel_scale_for(self, ref: torch.Tensor) -> torch.Tensor:
        """Return x/y pixel scales on the same device and dtype as ref."""
        return self.pixel_scale.to(device=ref.device, dtype=ref.dtype)

    def _is_fixed_y(self) -> bool:
        """Return True when fixed-y labels make y a shared anchor, not a learned target."""
        return self.point_mode == "fixed_y"

    def update(self) -> None:
        """Advance the epoch counter used when no explicit batch epoch is provided."""
        self.updates = int(getattr(self, "updates", 0)) + 1

    @staticmethod
    def _zero_like(pred_points: torch.Tensor) -> torch.Tensor:
        """Return a differentiable scalar zero on the prediction device."""
        return pred_points.sum() * 0.0

    @staticmethod
    def _nearest_visible_x(target_lane: torch.Tensor, valid_lane: torch.Tensor) -> torch.Tensor | None:
        """Build pseudo x targets by copying each anchor's nearest visible GT endpoint/point."""
        visible = torch.where(valid_lane > 0.5)[0]
        if visible.numel() == 0:
            return None
        anchors = torch.arange(target_lane.shape[0], device=target_lane.device)
        nearest = visible[(anchors[:, None] - visible[None]).abs().argmin(dim=1)]
        return target_lane[nearest, 0].clamp(0.0, 1.0)

    @staticmethod
    def _matched_query_mask(
        pred_logits: torch.Tensor, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Return a B x Q mask for Hungarian-matched query slots."""
        matched = torch.zeros_like(pred_logits, dtype=torch.bool)
        for b, (src_idx, _) in enumerate(indices):
            if src_idx.numel():
                matched[b, src_idx] = True
        return matched

    @staticmethod
    def _matched_target_lookup(
        pred_logits: torch.Tensor, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Return a B x Q tensor with matched GT lane indices, or -1 for unmatched queries."""
        lookup = torch.full(pred_logits.shape, -1, device=pred_logits.device, dtype=torch.long)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel():
                lookup[b, src_idx.to(device=pred_logits.device, dtype=torch.long)] = tgt_idx.to(
                    device=pred_logits.device, dtype=torch.long
                )
        return lookup

    @staticmethod
    def _matched_target_edge_mask(
        gt_points_b: torch.Tensor,
        gt_valid_b: torch.Tensor,
        tgt_idx: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        min_lane_count: int = 4,
    ) -> tuple[torch.Tensor, int]:
        """Return matched left/right edge-lane mask and valid GT lane count for one image."""
        is_edge = torch.zeros(tgt_idx.shape[0], device=device, dtype=torch.bool)
        if tgt_idx.numel() == 0:
            return is_edge, 0
        valid = gt_valid_b.detach().to(device=device, dtype=dtype)
        points = gt_points_b.detach().to(device=device, dtype=dtype)
        if valid.ndim != 2 or points.ndim != 3 or points.shape[:2] != valid.shape:
            raise ValueError(
                "GT points/valid shapes must be N x K x 2 and N x K for edge lane weighting, "
                f"got {tuple(points.shape)} and {tuple(valid.shape)}."
            )
        lane_mask = valid.sum(dim=1) >= 2
        lane_count = int(lane_mask.sum().item())
        if lane_count < int(min_lane_count):
            return is_edge, lane_count
        visible_den = valid.sum(dim=1).clamp_min(1.0)
        mean_x = (points[..., 0] * valid).sum(dim=1) / visible_den
        mean_x = torch.where(lane_mask, mean_x, torch.full_like(mean_x, float("inf")))
        left = int(torch.argmin(mean_x).item())
        mean_x_right = torch.where(lane_mask, mean_x, torch.full_like(mean_x, float("-inf")))
        right = int(torch.argmax(mean_x_right).item())
        edge = torch.tensor([left, right], device=device, dtype=torch.long)
        matched_tgt = tgt_idx.to(device=device, dtype=torch.long)
        is_edge = (matched_tgt[:, None] == edge.view(1, -1)).any(dim=1)
        return is_edge, lane_count

    def _gt5_edge_query_mask(
        self,
        pred_logits: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Return B x Q mask for matched left/right edge lanes in images with at least 5 GT lanes."""
        mask = torch.zeros_like(pred_logits, dtype=torch.bool)
        device, dtype = pred_logits.device, pred_logits.dtype
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            is_edge, lane_count = self._matched_target_edge_mask(
                gt_points[b],
                gt_valid[b],
                tgt_idx,
                device=device,
                dtype=dtype,
                min_lane_count=5,
            )
            if lane_count >= 5:
                mask[b, src_idx.to(device=device, dtype=torch.long)] = is_edge
        return mask

    def _matched_target_weights(
        self,
        gt_points_b: torch.Tensor,
        gt_valid_b: torch.Tensor,
        tgt_idx: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        hard_image: bool = False,
        term: str | None = None,
    ) -> torch.Tensor:
        """Return per-matched-lane weights, boosting left/right edge lanes for dense-lane images."""
        weights = torch.ones(tgt_idx.shape[0], device=device, dtype=dtype)
        hard_enabled = self._hard_edge_loss_enabled_for(term, hard_image)
        candidate_enabled = float(self.candidate_gt5_edge_weight) > 1.0
        if tgt_idx.numel() == 0 or (
            float(self.gt5_edge_loss_weight) <= 1.0 and not hard_enabled and not candidate_enabled
        ):
            return weights
        is_edge, lane_count = self._matched_target_edge_mask(
            gt_points_b,
            gt_valid_b,
            tgt_idx,
            device=device,
            dtype=dtype,
            min_lane_count=4,
        )
        if lane_count < 4:
            return weights
        base_edge_multiplier = float(self.gt5_edge_loss_weight) if float(self.gt5_edge_loss_weight) > 1.0 else 1.0
        candidate_multiplier = (
            float(self.candidate_gt5_edge_weight)
            if lane_count >= 5 and float(self.candidate_gt5_edge_weight) > 1.0
            else 1.0
        )
        hard_multiplier = self._hard_edge_loss_multiplier(lane_count, term, hard_image)
        if base_edge_multiplier <= 1.0 and candidate_multiplier <= 1.0 and hard_multiplier <= 1.0:
            return weights
        if base_edge_multiplier > 1.0:
            weights = torch.where(is_edge, weights * base_edge_multiplier, weights)
        if candidate_multiplier > 1.0:
            weights = torch.where(is_edge, weights * candidate_multiplier, weights)
        if hard_multiplier > 1.0:
            hard_apply = is_edge if self.hard_edge_only else torch.ones_like(is_edge)
            weights = torch.where(hard_apply, weights * hard_multiplier, weights)
        return weights

    def _edge_query_weight_matrix(
        self,
        pred_logits: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        hard_loss_mask: torch.Tensor | None = None,
        term: str | None = None,
    ) -> torch.Tensor:
        """Return B x Q query weights for matched edge lanes; unmatched queries stay at weight 1."""
        weights = torch.ones_like(pred_logits)
        if (
            float(self.gt5_edge_loss_weight) <= 1.0
            and float(self.candidate_gt5_edge_weight) <= 1.0
            and (hard_loss_mask is None or not bool(hard_loss_mask.detach().any().item()))
        ):
            return weights
        device, dtype = pred_logits.device, pred_logits.dtype
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            lane_weights = self._matched_target_weights(
                gt_points[b],
                gt_valid[b],
                tgt_idx,
                device=device,
                dtype=dtype,
                hard_image=self._hard_mask_value(hard_loss_mask, b),
                term=term,
            )
            weights[b, src_idx.to(device=device, dtype=torch.long)] = lane_weights
        return weights

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

    def _matched_exist_quality(
        self,
        pred_points: torch.Tensor,
        target_points: torch.Tensor,
        valid: torch.Tensor,
        pred_valid_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build detached matched-query quality from geometry, point error, and visibility completeness."""
        scale = self._pixel_scale_for(pred_points)
        if self._is_fixed_y():
            point_error = (pred_points[..., 0] - target_points[..., 0]).abs() * scale[..., 0]
        else:
            point_error = torch.norm((pred_points - target_points) * scale, dim=-1)
        ape = (point_error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        point_quality = self._exist_quality_from_ape(ape)
        line_iou_quality = (
            self._line_iou(pred_points, target_points, valid).clamp(min=0.0, max=1.0)
            if self._is_fixed_y()
            else point_quality
        )

        if pred_valid_logits is None:
            valid_quality = torch.ones_like(point_quality)
        else:
            valid_prob = pred_valid_logits.detach().sigmoid().to(dtype=pred_points.dtype)
            intersection = (valid_prob * valid).sum(dim=1)
            union = valid_prob.sum(dim=1) + valid.sum(dim=1) - intersection
            valid_quality = (intersection / union.clamp_min(1e-6)).clamp(min=0.0, max=1.0)

        floor = max(0.5, min(max(float(self.exist_quality_floor), 0.0), 1.0))
        quality = 0.6 * line_iou_quality + 0.3 * point_quality + 0.1 * valid_quality
        return quality.clamp(min=floor, max=1.0).detach()

    @torch.no_grad()
    def negative_query_masks(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard-negative and duplicate-negative B x Q masks for unmatched queries."""
        unmatched = ~self._matched_query_mask(pred_logits, indices)
        if pred_valid_logits is None:
            valid_mean = torch.ones_like(pred_logits)
        else:
            valid_mean = pred_valid_logits.detach().sigmoid().mean(dim=-1)
        quality_pred = pred_logits.detach().sigmoid() * valid_mean
        hard_negative = unmatched & (quality_pred > float(self.hard_negative_quality_thr))

        topk = min(int(self.hard_negative_topk), int(pred_logits.shape[1]))
        if topk > 0:
            for b in range(pred_logits.shape[0]):
                unmatched_idx = torch.nonzero(unmatched[b], as_tuple=False).flatten()
                if unmatched_idx.numel() == 0:
                    continue
                count = min(topk, int(unmatched_idx.numel()))
                selected = quality_pred[b, unmatched_idx].topk(k=count, largest=True).indices
                hard_negative[b, unmatched_idx[selected]] = True

        duplicate_negative = torch.zeros_like(unmatched)
        device, dtype = pred_points.device, pred_points.dtype
        scale_x = self._pixel_scale_for(pred_points).to(device=device, dtype=dtype)[..., 0]
        for b in range(pred_points.shape[0]):
            query_idx = torch.nonzero(unmatched[b], as_tuple=False).flatten()
            target = gt_points[b].to(device=device, dtype=dtype)
            valid = gt_valid[b].to(device=device, dtype=dtype)
            if query_idx.numel() == 0 or target.numel() == 0:
                continue
            valid_lane = valid.sum(dim=1) >= max(int(self.matcher.min_overlap), 1)
            target = target[valid_lane]
            valid = valid[valid_lane]
            if target.numel() == 0:
                continue

            pred = pred_points[b, query_idx].detach()
            pred_pair = pred[:, None].expand(-1, target.shape[0], -1, -1)
            target_pair = target[None].expand(pred.shape[0], -1, -1, -1)
            valid_pair = valid[None].expand(pred.shape[0], -1, -1)
            if self._is_fixed_y():
                dist = (pred_pair[..., 0] - target_pair[..., 0]).abs() * scale_x
            else:
                scale = self._pixel_scale_for(pred_points).to(device=device, dtype=dtype)
                dist = torch.norm((pred_pair - target_pair) * scale, dim=-1)
            mean_dist = (dist * valid_pair).sum(dim=-1) / valid_pair.sum(dim=-1).clamp_min(1.0)
            min_dist = mean_dist.min(dim=1).values

            if self._is_fixed_y():
                pair_count = pred.shape[0] * target.shape[0]
                pair_iou = self._line_iou(
                    pred_pair.reshape(pair_count, pred.shape[1], 2),
                    target_pair.reshape(pair_count, target.shape[1], 2),
                    valid_pair.reshape(pair_count, valid.shape[1]),
                ).reshape(pred.shape[0], target.shape[0])
                max_iou = pair_iou.max(dim=1).values
            else:
                max_iou = torch.zeros_like(min_dist)
            is_duplicate = torch.zeros_like(min_dist, dtype=torch.bool)
            if float(self.duplicate_dist_thr_px) > 0.0:
                is_duplicate |= min_dist < float(self.duplicate_dist_thr_px)
            if float(self.duplicate_iou_thr) > 0.0:
                is_duplicate |= max_iou > float(self.duplicate_iou_thr)
            duplicate_negative[b, query_idx] = is_duplicate

        hard_negative |= duplicate_negative
        return hard_negative.detach(), duplicate_negative.detach()

    def _line_iou(self, pred_points: torch.Tensor, target_points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Approximate lane IoU by expanding each fixed row point to a horizontal strip in pixel space."""
        if not self._is_fixed_y():
            raise ValueError(
                "GCS LineIoU currently only supports fixed-y lanes with shared y anchors. "
                "Set gcs_line_iou=0.0 and gcs_exist_quality_lane_iou_alpha=0.0 for free-point mode."
            )
        if pred_points.shape != target_points.shape:
            raise ValueError(
                f"LineIoU pred/target shapes must match, got {tuple(pred_points.shape)} vs {tuple(target_points.shape)}."
            )
        if valid.shape != pred_points.shape[:2]:
            raise ValueError(f"LineIoU valid mask must match M x K, got {tuple(valid.shape)} vs {tuple(pred_points.shape[:2])}.")

        valid = valid.to(device=pred_points.device, dtype=pred_points.dtype)
        scale_x = self._pixel_scale_for(pred_points).to(device=pred_points.device, dtype=pred_points.dtype)[..., 0]
        half_width = pred_points.new_tensor(float(self.line_iou_width_px))
        pred_x = pred_points[..., 0] * scale_x
        target_x = target_points[..., 0] * scale_x

        pred_left, pred_right = pred_x - half_width, pred_x + half_width
        target_left, target_right = target_x - half_width, target_x + half_width
        inter = (torch.minimum(pred_right, target_right) - torch.maximum(pred_left, target_left)).clamp_min(0.0)
        union = (torch.maximum(pred_right, target_right) - torch.minimum(pred_left, target_left)).clamp_min(1e-6)
        inter = inter * valid
        union = union * valid
        return inter.sum(dim=1) / union.sum(dim=1).clamp_min(1e-6)

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

    def _pred_quality_logits(self, preds: dict[str, torch.Tensor], pred_points: torch.Tensor) -> torch.Tensor | None:
        """Read optional lane-quality logits from the GCS head output."""
        pred_quality_logits = preds.get("pred_quality_logits")
        if pred_quality_logits is None:
            if self.quality_gain > 0.0:
                raise ValueError(
                    "GCS lane Quality Head loss is enabled but pred_quality_logits is missing. "
                    "Use a GCSLaneHead with quality_mlp, or set gcs_quality=0.0 for a deliberate ablation."
                )
            return None
        if pred_quality_logits.ndim == 3 and pred_quality_logits.shape[-1] == 1:
            pred_quality_logits = pred_quality_logits.squeeze(-1)
        if pred_quality_logits.shape != pred_points.shape[:2]:
            raise ValueError(
                "pred_quality_logits must have shape B x Q matching pred_points, "
                f"got {tuple(pred_quality_logits.shape)} vs {tuple(pred_points.shape[:2])}."
            )
        return pred_quality_logits

    def exist_loss(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        pred_valid_logits: torch.Tensor | None,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        hard_loss_mask: torch.Tensor | None = None,
        hard_negative_mask: torch.Tensor | None = None,
        duplicate_negative_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Quality-aware existence supervision with focused unmatched-query mining."""
        target = torch.zeros_like(pred_logits)
        matched = self._matched_query_mask(pred_logits, indices)
        alpha = min(max(float(self.exist_quality_alpha), 0.0), 1.0)
        device, dtype = pred_points.device, pred_points.dtype
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel():
                if alpha <= 0.0:
                    target[b, src_idx] = 1.0
                    continue

                pred = pred_points[b, src_idx].detach()
                target_points = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
                valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]
                valid_logits = pred_valid_logits[b, src_idx] if pred_valid_logits is not None else None
                quality = self._matched_exist_quality(pred, target_points, valid, valid_logits)
                target_quality = ((1.0 - alpha) + alpha * quality.to(dtype=target.dtype)).detach()
                target[b, src_idx] = target_quality

        if hard_negative_mask is None or duplicate_negative_mask is None:
            hard_negative_mask, duplicate_negative_mask = self.negative_query_masks(
                pred_logits, pred_points, pred_valid_logits, gt_points, gt_valid, indices
            )
        negative_weight = torch.ones_like(pred_logits)
        negative_weight = torch.where(
            hard_negative_mask,
            negative_weight.new_tensor(self.hard_negative_exist_weight),
            negative_weight,
        )
        negative_weight = torch.where(
            duplicate_negative_mask,
            negative_weight.new_tensor(self.duplicate_negative_exist_weight),
            negative_weight,
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
        edge_weight = self._edge_query_weight_matrix(
            pred_logits, gt_points, gt_valid, indices, hard_loss_mask=hard_loss_mask, term="exist"
        )
        query_weight = torch.where(matched, edge_weight, negative_weight)
        loss = (loss * query_weight).mean()

        if self.exist_margin_gain > 0.0:
            prob = pred_logits.sigmoid()
            pos_weight = (target.detach() * matched.to(dtype=target.dtype)).clamp(min=0.0, max=1.0)
            if bool((pos_weight > 0.0).any()):
                pos_loss = (
                    torch.relu(pred_logits.new_tensor(self.exist_pos_margin) - prob).pow(2) * pos_weight
                ).sum() / pos_weight.sum().clamp_min(1.0)
            else:
                pos_loss = self._zero_like(pred_logits)

            neg_mask = ~matched
            if bool(neg_mask.any()):
                neg_penalty = torch.relu(prob - pred_logits.new_tensor(self.exist_neg_margin)).pow(2)
                neg_loss = (neg_penalty[neg_mask] * negative_weight[neg_mask]).sum() / neg_mask.sum().clamp_min(1)
            else:
                neg_loss = self._zero_like(pred_logits)
            loss = loss + self.exist_margin_gain * (pos_loss + neg_loss)
        return loss

    def point_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        pred_valid_logits: torch.Tensor | None = None,
        hard_loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aspect-weighted L1 point loss on Hungarian-matched lane point sequences."""
        losses = []
        invalid_x_losses = []
        device, dtype = pred_points.device, pred_points.dtype
        scale = self._scale_for(pred_points)
        beta = 1.0 / max(float(max(self.image_size)), 1.0)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pred = pred_points[b, src_idx]
            target = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
            valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]

            if self._is_fixed_y():
                loss = (pred[..., 0] - target[..., 0]).abs() * scale[..., 0]
            else:
                loss = ((pred - target).abs() * scale).sum(dim=-1)
            loss = loss * valid
            lane_loss = loss.sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            lane_weights = self._matched_target_weights(
                gt_points[b],
                gt_valid[b],
                tgt_idx,
                device=device,
                dtype=dtype,
                hard_image=self._hard_mask_value(hard_loss_mask, b),
                term="point",
            )
            losses.append((lane_loss * lane_weights).sum() / lane_weights.sum().clamp_min(1.0))

            if self._is_fixed_y() and self.point_invalid_x_gain > 0.0 and pred_valid_logits is not None:
                valid_prob = pred_valid_logits[b, src_idx].sigmoid().detach().to(dtype=dtype)
                invalid_mask = valid < 0.5
                if bool(invalid_mask.any()):
                    lane_losses = []
                    for lane_i in range(pred.shape[0]):
                        pseudo_x = self._nearest_visible_x(target[lane_i], valid[lane_i])
                        if pseudo_x is None:
                            continue
                        lane_invalid = invalid_mask[lane_i]
                        if not bool(lane_invalid.any()):
                            continue
                        scale_x = scale[..., 0].reshape(())
                        pred_x = pred[lane_i, :, 0] * scale_x
                        pseudo_x = pseudo_x.to(device=device, dtype=dtype) * scale_x
                        invalid_delta = F.smooth_l1_loss(pred_x, pseudo_x, reduction="none", beta=beta)
                        lane_weight = valid_prob[lane_i] * lane_invalid.to(dtype=dtype)
                        lane_den = lane_invalid.sum().to(dtype=dtype).clamp_min(1.0)
                        lane_losses.append(((invalid_delta * lane_weight).sum() / lane_den) * lane_weights[lane_i])
                    if lane_losses:
                        invalid_x_losses.append(torch.stack(lane_losses).sum() / lane_weights.sum().clamp_min(1.0))

        visible_loss = torch.stack(losses).mean() if losses else self._zero_like(pred_points)
        if self.point_invalid_x_gain <= 0.0 or not invalid_x_losses:
            return visible_loss
        invalid_x_loss = torch.stack(invalid_x_losses).mean()
        return visible_loss + self.point_invalid_x_gain * invalid_x_loss

    def line_iou_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        hard_loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Whole-lane LineIoU loss on Hungarian-matched lane point sequences."""
        losses = []
        device, dtype = pred_points.device, pred_points.dtype
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pred = pred_points[b, src_idx]
            target = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
            valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]
            iou = self._line_iou(pred, target, valid)
            lane_weights = self._matched_target_weights(
                gt_points[b],
                gt_valid[b],
                tgt_idx,
                device=device,
                dtype=dtype,
                hard_image=self._hard_mask_value(hard_loss_mask, b),
                term="line_iou",
            )
            lane_loss = 1.0 - iou
            losses.append((lane_loss * lane_weights).sum() / lane_weights.sum().clamp_min(1.0))

        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    def _quality_point_inlier_score(
        self,
        pred_points: torch.Tensor,
        target_points: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return official-threshold-style inlier ratio for matched fixed-y lanes."""
        valid = valid.to(device=pred_points.device, dtype=pred_points.dtype)
        scale_x = self._pixel_scale_for(pred_points).to(device=pred_points.device, dtype=pred_points.dtype)[..., 0]
        if self._is_fixed_y():
            dist = (pred_points[..., 0] - target_points[..., 0]).abs() * scale_x
        else:
            scale = self._pixel_scale_for(pred_points)
            dist = torch.norm((pred_points - target_points) * scale, dim=-1)
        inlier = (dist <= float(self.quality_dist_thr_px)).to(dtype=pred_points.dtype) * valid
        return (inlier.sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def build_quality_targets(
        self,
        pred_quality_logits: torch.Tensor,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Build continuous B x Q lane-quality targets after Hungarian matching."""
        target_quality = torch.zeros_like(pred_quality_logits)
        device, dtype = pred_points.device, pred_points.dtype
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pred = pred_points[b, src_idx].detach()
            target = gt_points[b].to(device=device, dtype=dtype)[tgt_idx]
            valid = gt_valid[b].to(device=device, dtype=dtype)[tgt_idx]
            point_score = self._quality_point_inlier_score(pred, target, valid)
            line_iou_score = self._line_iou(pred, target, valid).clamp(min=0.0, max=1.0)
            quality = (0.5 * point_score + 0.5 * line_iou_score).clamp(min=0.0, max=1.0)
            target_quality[b, src_idx.to(device=target_quality.device, dtype=torch.long)] = quality.to(
                device=target_quality.device, dtype=target_quality.dtype
            )
        return target_quality.detach()

    @torch.no_grad()
    def _quality_head_hard_negative_mask(
        self,
        pred_quality_logits: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mine unmatched quality negatives directly from the Quality Head confidence."""
        quality_prob = pred_quality_logits.detach().sigmoid()
        hard_negative = neg_mask & (quality_prob > float(self.hard_negative_quality_thr))

        topk = min(int(self.hard_negative_topk), int(pred_quality_logits.shape[1]))
        if topk > 0:
            for b in range(pred_quality_logits.shape[0]):
                unmatched_idx = torch.nonzero(neg_mask[b], as_tuple=False).flatten()
                if unmatched_idx.numel() == 0:
                    continue
                count = min(topk, int(unmatched_idx.numel()))
                selected = quality_prob[b, unmatched_idx].topk(k=count, largest=True).indices
                hard_negative[b, unmatched_idx[selected]] = True
        return hard_negative.detach()

    @staticmethod
    def _longest_true_segment_bounds(mask: torch.Tensor) -> tuple[int, int]:
        """Return [start, end) bounds for the longest contiguous true run in a 1D mask."""
        idx = torch.nonzero(mask.detach(), as_tuple=False).flatten().tolist()
        if not idx:
            return 0, 0

        best_start = start = int(idx[0])
        best_len = 1
        prev = int(idx[0])
        cur_len = 1
        for raw in idx[1:]:
            cur = int(raw)
            if cur == prev + 1:
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_start, best_len = start, cur_len
                start, cur_len = cur, 1
            prev = cur
        if cur_len > best_len:
            best_start, best_len = start, cur_len
        return best_start, best_start + best_len

    def quality_loss(
        self,
        pred_quality_logits: torch.Tensor | None,
        pred_points: torch.Tensor,
        gt_points: list[torch.Tensor],
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        hard_loss_mask: torch.Tensor | None = None,
        hard_negative_mask: torch.Tensor | None = None,
        duplicate_negative_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BCE loss for lane-level geometry quality with focused negative-query weights."""
        if pred_quality_logits is None:
            return self._zero_like(pred_points)
        target_quality = self.build_quality_targets(pred_quality_logits, pred_points, gt_points, gt_valid, indices)
        raw_loss = F.binary_cross_entropy_with_logits(pred_quality_logits, target_quality, reduction="none")

        matched_mask = self._matched_query_mask(pred_quality_logits, indices)
        unmatched_mask = ~matched_mask
        if bool(matched_mask.any()):
            pos_query_weight = self._edge_query_weight_matrix(
                pred_quality_logits,
                gt_points,
                gt_valid,
                indices,
                hard_loss_mask=hard_loss_mask,
                term="quality",
            )
            pos_loss = (raw_loss[matched_mask] * pos_query_weight[matched_mask]).sum()
            pos_loss = pos_loss / pos_query_weight[matched_mask].sum().clamp_min(1.0)
        else:
            pos_loss = self._zero_like(pred_points)
        if bool(unmatched_mask.any()):
            neg_weight = torch.full_like(raw_loss, float(self.quality_neg_weight))
            if self.quality_hard_negative_from_head:
                head_hard_negative_mask = self._quality_head_hard_negative_mask(pred_quality_logits, unmatched_mask)
                hard_negative_mask = (
                    head_hard_negative_mask
                    if hard_negative_mask is None
                    else (hard_negative_mask | head_hard_negative_mask)
                )
            if hard_negative_mask is not None:
                hard_negative_mask = hard_negative_mask & unmatched_mask
                neg_weight = torch.where(
                    hard_negative_mask,
                    neg_weight.new_tensor(self.quality_hard_negative_weight),
                    neg_weight,
                )
            if duplicate_negative_mask is not None:
                duplicate_negative_mask = duplicate_negative_mask & unmatched_mask
                neg_weight = torch.where(
                    duplicate_negative_mask,
                    neg_weight.new_tensor(self.quality_duplicate_negative_weight),
                    neg_weight,
                )
            neg_loss = (raw_loss[unmatched_mask] * neg_weight[unmatched_mask]).sum()
            neg_loss = neg_loss / unmatched_mask.sum().clamp_min(1)
        else:
            neg_loss = self._zero_like(pred_points)
        return pos_loss + neg_loss

    def point_valid_loss(
        self,
        pred_valid_logits: torch.Tensor | None,
        pred_points: torch.Tensor,
        gt_valid: list[torch.Tensor],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        gt_points: list[torch.Tensor] | None = None,
        hard_loss_mask: torch.Tensor | None = None,
        hard_negative_mask: torch.Tensor | None = None,
        duplicate_negative_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BCE supervision for visible fixed-y anchors on matched lanes and zero target for unmatched queries."""
        if pred_valid_logits is None:
            return self._zero_like(pred_points)

        target = torch.zeros_like(pred_valid_logits)
        matched = self._matched_query_mask(pred_valid_logits[..., 0], indices)
        gt5_image_mask = torch.zeros(pred_valid_logits.shape[0], device=pred_valid_logits.device, dtype=torch.bool)
        edge_weight = torch.ones_like(pred_valid_logits)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            lane_count = int(
                (gt_valid[b].detach().to(device=target.device).float().sum(dim=1) >= 2).sum().item()
            )
            if lane_count >= 5:
                gt5_image_mask[b] = True
            if src_idx.numel() == 0:
                continue
            target[b, src_idx] = gt_valid[b].to(device=target.device, dtype=target.dtype)[tgt_idx]
            if gt_points is not None:
                lane_weights = self._matched_target_weights(
                    gt_points[b],
                    gt_valid[b],
                    tgt_idx,
                    device=target.device,
                    dtype=target.dtype,
                    hard_image=self._hard_mask_value(hard_loss_mask, b),
                    term="point_valid",
                )
            else:
                lane_weights = torch.ones(src_idx.shape[0], device=target.device, dtype=target.dtype)
            edge_weight[b, src_idx] = lane_weights.view(-1, 1)

        query_weight = torch.ones_like(pred_valid_logits[..., 0])
        query_weight = torch.where(
            ~matched,
            query_weight.new_tensor(self.point_valid_unmatched_weight),
            query_weight,
        )
        if hard_negative_mask is not None:
            query_weight = torch.where(
                hard_negative_mask,
                query_weight.new_tensor(self.point_valid_hard_negative_weight),
                query_weight,
            )
        if duplicate_negative_mask is not None:
            query_weight = torch.where(
                duplicate_negative_mask,
                query_weight.new_tensor(self.point_valid_duplicate_negative_weight),
                query_weight,
            )
        anchor_weight = query_weight.unsqueeze(-1) * edge_weight
        pos = target.sum().clamp_min(1.0)
        neg = (target.numel() - target.sum()).clamp_min(1.0)
        pos_weight = (neg / pos).clamp(min=1.0, max=float(self.point_valid_pos_weight_max)).to(pred_valid_logits)
        loss_elem = F.binary_cross_entropy_with_logits(
            pred_valid_logits, target, pos_weight=pos_weight, reduction="none"
        ) * anchor_weight
        if self.point_valid_gt5_pos_weight > 1.0 and bool(gt5_image_mask.any()):
            weight = torch.ones_like(loss_elem)
            boost_mask = (target > 0.5) & gt5_image_mask.view(-1, 1, 1)
            weight = torch.where(boost_mask, weight * float(self.point_valid_gt5_pos_weight), weight)
            loss = (loss_elem * weight).mean()
        else:
            loss = loss_elem.mean()

        edge_query_mask = None
        if (
            self.point_valid_gt5_edge_continuity > 0.0
            and gt_points is not None
            and pred_valid_logits.shape[-1] > 1
        ):
            edge_query_mask = self._gt5_edge_query_mask(pred_valid_logits[..., 0], gt_points, gt_valid, indices)
            pair_mask = (target[..., :-1] > 0.5) & (target[..., 1:] > 0.5) & edge_query_mask.unsqueeze(-1)
            if bool(pair_mask.any()):
                valid_prob = pred_valid_logits.sigmoid()
                pair_prob = torch.minimum(valid_prob[..., :-1], valid_prob[..., 1:])
                threshold = pred_valid_logits.new_tensor(float(self.point_valid_gt5_edge_continuity_thr))
                continuity_penalty = torch.relu(threshold - pair_prob).pow(2)
                pair_weight = 0.5 * (anchor_weight[..., :-1] + anchor_weight[..., 1:])
                continuity_loss = (continuity_penalty[pair_mask] * pair_weight[pair_mask]).sum()
                continuity_loss = continuity_loss / pair_weight[pair_mask].sum().clamp_min(1.0)
                loss = loss + float(self.point_valid_gt5_edge_continuity) * continuity_loss

        if self.point_valid_gt5_edge_segment > 0.0 and gt_points is not None:
            if edge_query_mask is None:
                edge_query_mask = self._gt5_edge_query_mask(pred_valid_logits[..., 0], gt_points, gt_valid, indices)
            if bool(edge_query_mask.any()):
                valid_prob = pred_valid_logits.sigmoid()
                threshold = pred_valid_logits.new_tensor(float(self.point_valid_gt5_edge_segment_thr))
                min_points = int(self.point_valid_gt5_edge_segment_min_points)
                segment_loss = self._zero_like(pred_points)
                segment_weight_sum = pred_valid_logits.new_tensor(0.0)
                segment_count = 0
                for b, q in torch.nonzero(edge_query_mask, as_tuple=False).tolist():
                    start, end = self._longest_true_segment_bounds(target[b, q] > 0.5)
                    if end - start < min_points:
                        continue
                    segment_prob = valid_prob[b, q, start:end]
                    anchor_penalty = torch.relu(threshold - segment_prob).pow(2).mean()
                    mean_penalty = torch.relu(threshold - segment_prob.mean()).pow(2)
                    segment_weight = anchor_weight[b, q, start:end].mean().clamp_min(1e-6)
                    segment_loss = segment_loss + 0.5 * (anchor_penalty + mean_penalty) * segment_weight
                    segment_weight_sum = segment_weight_sum + segment_weight
                    segment_count += 1
                if segment_count > 0:
                    segment_loss = segment_loss / segment_weight_sum.clamp_min(1.0)
                    loss = loss + float(self.point_valid_gt5_edge_segment) * segment_loss

        if self.point_valid_neg_gain > 0.0:
            neg_anchor_mask = target < 0.5
            if bool(neg_anchor_mask.any()):
                valid_prob = pred_valid_logits.sigmoid()
                neg_penalty = torch.relu(valid_prob - self.point_valid_neg_thr).pow(2)
                neg_loss = (neg_penalty[neg_anchor_mask] * anchor_weight[neg_anchor_mask]).sum()
                neg_loss = neg_loss / neg_anchor_mask.sum().clamp_min(1)
                loss = loss + self.point_valid_neg_gain * neg_loss
        return loss

    def target_lane_count(self, pred_logits: torch.Tensor, batch: dict, gt_valid: list[torch.Tensor]) -> torch.Tensor:
        """Return one GT lane count per image, deriving the authoritative count from lane_valid."""
        counts = []
        for valid in gt_valid:
            valid = valid.detach().to(device=pred_logits.device)
            if valid.ndim != 2:
                raise ValueError(f"GT lane_valid must have shape N x K, got {tuple(valid.shape)}.")
            counts.append(int((valid.float().sum(dim=1) >= 2).sum().item()))
        if len(counts) != pred_logits.shape[0]:
            raise ValueError(f"gt_valid must contain one tensor per image, got {len(counts)} vs B={pred_logits.shape[0]}.")
        target = pred_logits.new_tensor(counts, dtype=pred_logits.dtype)

        num_lanes = batch.get("num_lanes")
        if num_lanes is not None:
            provided = torch.as_tensor(num_lanes, device=pred_logits.device).reshape(-1).long()
            derived = target.long()
            if provided.numel() != pred_logits.shape[0]:
                raise ValueError(
                    f"batch['num_lanes'] must have one value per image, got {provided.numel()} vs B={pred_logits.shape[0]}."
                )
            if not torch.equal(provided, derived):
                raise ValueError(
                    "batch['num_lanes'] mismatch: the loss derives lane count from lane_valid after filtering/augmentation, "
                    f"but got num_lanes={provided.detach().cpu().tolist()} and derived={derived.detach().cpu().tolist()}."
                )
        return target

    def count_head_targets(
        self, pred_count_logits: torch.Tensor, gt_valid: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build count-head targets from lane_valid, clamped to the 2/3/4/5 count space."""
        counts = []
        for valid in gt_valid:
            valid = valid.detach().to(device=pred_count_logits.device)
            if valid.ndim != 2:
                raise ValueError(f"GT lane_valid must have shape N x K, got {tuple(valid.shape)}.")
            counts.append(int((valid.float().sum(dim=1) >= int(self.count_min_gt_points)).sum().item()))
        if len(counts) != pred_count_logits.shape[0]:
            raise ValueError(
                f"gt_valid must contain one tensor per image, got {len(counts)} vs B={pred_count_logits.shape[0]}."
            )
        gt_count_raw = torch.tensor(counts, device=pred_count_logits.device, dtype=torch.long)
        gt_count = gt_count_raw.clamp(min=2, max=5)
        gt_count_cls = (gt_count - 2).long()
        return gt_count, gt_count_cls, gt_count_raw

    def count_head_loss(
        self, preds: dict[str, torch.Tensor], pred_points: torch.Tensor, gt_valid: list[torch.Tensor]
    ) -> torch.Tensor:
        """Return CE plus optional count>=4/count>=5 boundary BCE for the image-level Count Head."""
        pred_count_logits = preds.get("pred_count_logits")
        if pred_count_logits is None:
            if self.count_cls_gain > 0.0:
                raise ValueError(
                    "GCS Count Head CE is enabled but pred_count_logits is missing. "
                    "Use a GCSLaneHead with LaneCountHead, or set gcs_count_cls=0.0 for a deliberate ablation."
                )
            return self._zero_like(pred_points)
        if pred_count_logits.ndim != 2 or pred_count_logits.shape[1] != 4:
            raise ValueError(f"pred_count_logits must have shape B x 4, got {tuple(pred_count_logits.shape)}.")

        _, gt_count_cls, _ = self.count_head_targets(pred_count_logits, gt_valid)
        class_weight = torch.tensor(
            self.count_cls_weights, device=pred_count_logits.device, dtype=torch.float32
        )
        count_loss = F.cross_entropy(pred_count_logits.float(), gt_count_cls, weight=class_weight)
        if self.count_boundary_gain <= 0.0:
            return count_loss
        pred_count_boundary_logits = preds.get("pred_count_boundary_logits")
        if pred_count_boundary_logits is None:
            raise ValueError(
                "GCS Count Boundary BCE is enabled but pred_count_boundary_logits is missing. "
                "Use a GCSLaneHead with Count Boundary Sub-Head, or set gcs_count_boundary=0.0 for a deliberate ablation."
            )
        if pred_count_boundary_logits.ndim != 2 or pred_count_boundary_logits.shape != (pred_count_logits.shape[0], 2):
            raise ValueError(
                "pred_count_boundary_logits must have shape B x 2, "
                f"got {tuple(pred_count_boundary_logits.shape)} vs B={pred_count_logits.shape[0]}."
            )
        boundary_targets = self.count_boundary_targets(pred_count_boundary_logits, gt_valid)
        boundary_loss_elem = F.binary_cross_entropy_with_logits(
            pred_count_boundary_logits.float(), boundary_targets.float(), reduction="none"
        )
        if self.count_boundary_gt5_pos_weight > 1.0:
            boundary_weight = torch.ones_like(boundary_loss_elem)
            gt5_positive = boundary_targets[:, 1] > 0.5
            boundary_weight[:, 1] = torch.where(
                gt5_positive,
                boundary_weight[:, 1] * float(self.count_boundary_gt5_pos_weight),
                boundary_weight[:, 1],
            )
            boundary_loss = (boundary_loss_elem * boundary_weight).mean()
        else:
            boundary_loss = boundary_loss_elem.mean()
        return count_loss + self.count_boundary_gain * boundary_loss

    def count_boundary_targets(self, pred_count_boundary_logits: torch.Tensor, gt_valid: list[torch.Tensor]) -> torch.Tensor:
        """Build smoothed binary targets for count>=4 and count>=5 boundary logits."""
        counts = []
        for valid in gt_valid:
            valid = valid.detach().to(device=pred_count_boundary_logits.device)
            if valid.ndim != 2:
                raise ValueError(f"GT lane_valid must have shape N x K, got {tuple(valid.shape)}.")
            counts.append(int((valid.float().sum(dim=1) >= int(self.count_min_gt_points)).sum().item()))
        if len(counts) != pred_count_boundary_logits.shape[0]:
            raise ValueError(
                "gt_valid must contain one tensor per image, "
                f"got {len(counts)} vs B={pred_count_boundary_logits.shape[0]}."
            )
        gt_count_raw = torch.tensor(counts, device=pred_count_boundary_logits.device, dtype=torch.float32)
        targets = torch.stack((gt_count_raw.ge(4).float(), gt_count_raw.ge(5).float()), dim=1)
        smoothing = min(max(float(self.count_boundary_label_smoothing), 0.0), 1.0)
        if smoothing > 0.0:
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets.to(dtype=pred_count_boundary_logits.dtype)

    def count_sum_loss(
        self,
        pred_logits: torch.Tensor,
        batch: dict,
        gt_valid: list[torch.Tensor],
    ) -> torch.Tensor:
        """Smooth-L1 consistency between summed query existence and GT lane count."""
        target = self.target_lane_count(pred_logits, batch, gt_valid).to(dtype=pred_logits.dtype)
        exist_sum = pred_logits.sigmoid().sum(dim=1)
        loss = F.smooth_l1_loss(exist_sum, target, reduction="none")
        if self.count_sum_normalize:
            loss = loss / target.clamp_min(1.0)
        return loss.mean()

    @torch.no_grad()
    def count_head_metrics(
        self, preds: dict[str, torch.Tensor], pred_points: torch.Tensor, gt_valid: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """Return scalar count-head accuracy, confusion, confidence, and raw-target audit metrics."""
        pred_count_logits = preds.get("pred_count_logits")
        if pred_count_logits is None:
            return tuple(self._zero_like(pred_points).detach() for _ in range(17))
        gt_count, _, gt_count_raw = self.count_head_targets(pred_count_logits, gt_valid)
        prob = pred_count_logits.detach().float().softmax(dim=-1)
        pred_count = prob.argmax(dim=-1).to(dtype=torch.long) + 2
        correct = pred_count.eq(gt_count)
        values: list[torch.Tensor] = [correct.float().mean()]
        for count in (2, 3, 4, 5):
            mask = gt_count.eq(count)
            if bool(mask.any()):
                values.append(pred_count[mask].eq(count).float().mean())
            else:
                values.append(pred_count.new_tensor(0.0, dtype=torch.float32))
        for src, dst in ((2, 3), (3, 2), (3, 4), (3, 5), (4, 3), (4, 5), (5, 4), (5, 3)):
            mask = gt_count.eq(src)
            if bool(mask.any()):
                values.append(pred_count[mask].eq(dst).float().mean())
            else:
                values.append(pred_count.new_tensor(0.0, dtype=torch.float32))
        top2 = prob.topk(k=2, dim=-1).values
        values.append(prob.max(dim=-1).values.mean())
        values.append((top2[:, 0] - top2[:, 1]).mean())
        values.append(gt_count_raw.float().min())
        values.append(gt_count_raw.float().max())
        return tuple(v.to(device=pred_points.device, dtype=pred_points.dtype).detach() for v in values)

    def count_head_warmup_factor(self, batch: dict | None = None) -> float:
        """Return the explicit count-head warmup multiplier."""
        if self.count_cls_gain <= 0.0:
            return 0.0
        warmup_epochs = float(self.count_head_warmup_epochs)
        if warmup_epochs <= 0.0:
            return 1.0
        if batch is not None and "epoch" in batch:
            epoch_value = batch["epoch"]
            if isinstance(epoch_value, torch.Tensor):
                epoch = float(epoch_value.detach().reshape(-1)[0].item())
            else:
                epoch = float(epoch_value)
        elif self.training:
            epoch = float(getattr(self, "updates", 0))
        else:
            return 1.0
        return float(min(max(epoch / max(warmup_epochs, 1e-6), 0.0), 1.0))

    def match_diagnostics(self, pred_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return detached matcher coverage diagnostics as scalar tensors."""
        stats = getattr(self.matcher, "last_stats", {}) or {}
        device, dtype = pred_points.device, pred_points.dtype
        return (
            torch.tensor(float(stats.get("matched_gt_ratio", 1.0)), device=device, dtype=dtype),
            torch.tensor(float(stats.get("no_match_image_rate", 0.0)), device=device, dtype=dtype),
            torch.tensor(float(stats.get("relaxed_gt_ratio", 0.0)), device=device, dtype=dtype),
        )

    def forward(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total GCS lane loss and detached loss components."""
        pred_points, pred_logits = self._normalize_pred_shapes(preds)
        pred_valid_logits = self._pred_valid_logits(preds, pred_points)
        pred_quality_logits = self._pred_quality_logits(preds, pred_points)
        if "img" in batch:
            assert_gcs_image_tensor(batch["img"], self.image_size, name="batch['img']", context="GCSLoss.forward")
        gt_points, gt_valid = self._targets_from_batch(batch)

        indices = self.matcher(pred_points, pred_logits, gt_points, gt_valid)
        hard_negative_mask, duplicate_negative_mask = self.negative_query_masks(
            pred_logits, pred_points, pred_valid_logits, gt_points, gt_valid, indices
        )
        hard_loss_mask = self.hard_loss_mask(batch, pred_logits.shape[0], pred_logits.device, gt_valid=gt_valid)
        exist_loss = self.exist_loss(
            pred_logits,
            pred_points,
            pred_valid_logits,
            gt_points,
            gt_valid,
            indices,
            hard_loss_mask=hard_loss_mask,
            hard_negative_mask=hard_negative_mask,
            duplicate_negative_mask=duplicate_negative_mask,
        )
        point_loss = self.point_loss(
            pred_points, gt_points, gt_valid, indices, pred_valid_logits, hard_loss_mask=hard_loss_mask
        )
        point_valid_loss = self.point_valid_loss(
            pred_valid_logits,
            pred_points,
            gt_valid,
            indices,
            gt_points=gt_points,
            hard_loss_mask=hard_loss_mask,
            hard_negative_mask=hard_negative_mask,
            duplicate_negative_mask=duplicate_negative_mask,
        )
        line_iou_loss = (
            self.line_iou_loss(pred_points, gt_points, gt_valid, indices, hard_loss_mask=hard_loss_mask)
            if self.line_iou_gain > 0.0
            else self._zero_like(pred_points)
        )
        count_cls_loss = self.count_head_loss(preds, pred_points, gt_valid)
        count_sum_loss = self.count_sum_loss(pred_logits, batch, gt_valid)
        quality_loss = (
            self.quality_loss(
                pred_quality_logits,
                pred_points,
                gt_points,
                gt_valid,
                indices,
                hard_loss_mask=hard_loss_mask,
                hard_negative_mask=hard_negative_mask,
                duplicate_negative_mask=duplicate_negative_mask,
            )
            if self.quality_gain > 0.0
            else self._zero_like(pred_points)
        )

        total = (
            self.exist_gain * exist_loss
            + self.point_gain * point_loss
            + self.point_valid_gain * point_valid_loss
            + self.line_iou_gain * line_iou_loss
            + self.count_head_warmup_factor(batch) * self.count_cls_gain * count_cls_loss
            + self.count_sum_gain * count_sum_loss
            + self.quality_gain * quality_loss
        )
        loss_items = torch.stack(
            (
                exist_loss.detach(),
                point_loss.detach(),
                point_valid_loss.detach(),
                line_iou_loss.detach(),
                count_cls_loss.detach(),
                count_sum_loss.detach(),
                quality_loss.detach(),
            )
        )
        return total, loss_items
