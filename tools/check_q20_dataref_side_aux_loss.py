"""Synthetic contract check for Q20 dataref side-query auxiliary loss."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.gcs_loss import GCSLoss


def _make_lane(bottom_x: float, top_x: float, y: torch.Tensor, valid_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one normalized fixed-y lane and its valid mask."""
    lane = torch.zeros(y.numel(), 2, dtype=torch.float32)
    lane[:, 0] = torch.linspace(float(bottom_x), float(top_x), y.numel())
    lane[:, 1] = y
    valid = torch.zeros(y.numel(), dtype=torch.float32)
    valid[: int(valid_points)] = 1.0
    return lane, valid


def build_synthetic_batch() -> tuple[dict[str, torch.Tensor], dict]:
    """Build one GT4 image with left/right short side lanes and Q20 predictions."""
    batch_size, num_queries, num_points = 1, 20, 56
    y = torch.linspace(710 / 720, 160 / 720, num_points, dtype=torch.float32)

    lane_specs = (
        (0.25, 0.36, 12),
        (0.43, 0.47, 56),
        (0.56, 0.53, 56),
        (0.75, 0.64, 12),
    )
    lanes, valid = zip(*(_make_lane(bottom, top, y, valid_points) for bottom, top, valid_points in lane_specs))
    gt_lanes = torch.stack(lanes, dim=0)
    gt_valid = torch.stack(valid, dim=0)

    pred_points = torch.zeros(batch_size, num_queries, num_points, 2, dtype=torch.float32)
    pred_points[..., 1] = y.view(1, 1, num_points)
    for q in range(num_queries):
        pred_points[0, q, :, 0] = 0.05 + 0.90 * q / max(num_queries - 1, 1)

    pred_points[0, 0, :, 0] = (gt_lanes[0, :, 0] + 0.02).clamp(0.0, 1.0)
    pred_points[0, 8, :, 0] = gt_lanes[1, :, 0]
    pred_points[0, 11, :, 0] = gt_lanes[2, :, 0]
    pred_points[0, 19, :, 0] = (gt_lanes[3, :, 0] - 0.02).clamp(0.0, 1.0)

    pred_reference_x = torch.full((batch_size, num_queries, num_points), 0.5, dtype=torch.float32)
    pred_reference_x[0, 0] = gt_lanes[0, :, 0]
    pred_reference_x[0, 19] = gt_lanes[3, :, 0]

    preds = {
        "pred_points": pred_points,
        "pred_logits": torch.zeros(batch_size, num_queries, dtype=torch.float32),
        "pred_valid_logits": torch.zeros(batch_size, num_queries, num_points, dtype=torch.float32),
        "pred_reference_x": pred_reference_x,
        "reference_mode": "dataref",
        "is_dataref_reference": torch.tensor(1.0, dtype=torch.float32),
        "pred_count_logits": torch.zeros(batch_size, 3, dtype=torch.float32),
    }
    batch = {
        "img": torch.zeros(batch_size, 3, 544, 960, dtype=torch.float32),
        "lanes": [gt_lanes],
        "lane_valid": [gt_valid],
        "gt_lanes": [gt_lanes],
        "gt_lane_valid": [gt_valid],
        "num_lanes": torch.tensor([4], dtype=torch.long),
    }
    return preds, batch


def main() -> None:
    """Run the side-aux loss contract check."""
    preds, batch = build_synthetic_batch()
    loss_fn = GCSLoss(
        model={
            "gcs_imgsz": [544, 960],
            "gcs_dataref_side_aux": 0.25,
            "gcs_dataref_side_aux_point": 1.0,
            "gcs_dataref_side_aux_valid": 0.25,
            "gcs_dataref_side_aux_exist": 0.25,
            "gcs_dataref_side_max_points": 24,
            "gcs_dataref_side_ref_thr_px": 80.0,
            "gcs_dataref_side_gt_count": 4,
            "gcs_dataref_side_left_thr": 0.35,
            "gcs_dataref_side_right_thr": 0.65,
            "gcs_lane_balanced_valid_loss": True,
            "gcs_gt4_short_valid_lane_weight": 1.5,
            "gcs_gt4_short_valid_pos_weight": 1.15,
            "gcs_unmatched_valid_neg_weight": 0.75,
            "gcs_gt4_short_valid_recall": False,
            "gcs_gt4_short_valid_count_floor": False,
        },
        image_size=(544, 960),
    )
    total, loss_items = loss_fn(preds, batch)
    values = dict(zip(GCSLoss.loss_names, loss_items.detach().cpu().tolist()))

    if not torch.isfinite(total) or not torch.isfinite(loss_items).all():
        raise AssertionError("Loss contains NaN or Inf.")
    for key in (
        "dataref_side_aux_lanes",
        "dataref_side_aux_loss",
        "dataref_side_aux_point",
        "dataref_side_aux_valid",
        "dataref_side_aux_exist",
    ):
        if values[key] <= 0.0:
            raise AssertionError(f"{key} must be > 0, got {values[key]}.")
    if values["dataref_side_aux_refdist"] > 80.0:
        raise AssertionError(f"dataref_side_aux_refdist must be <= 80, got {values['dataref_side_aux_refdist']}.")
    for key in ("gt4_short_pred_valid_prob_mean", "gt4_short_pred_valid_sum_mean"):
        if abs(values[key]) > 1e-12:
            raise AssertionError(f"{key} must stay 0 while GT4 valid repair flags are disabled, got {values[key]}.")
    for key in ("valid_lb_gt4_short_pred_prob_mean", "valid_lb_gt4_short_pred_sum_mean"):
        if values[key] <= 0.0:
            raise AssertionError(f"{key} must be > 0 when lane-balanced valid loss is enabled, got {values[key]}.")

    print("OK: Q20 dataref side-query auxiliary loss contract is correct.")
    print(
        "dataref_side_aux:",
        {
            key: values[key]
            for key in (
                "dataref_side_aux_lanes",
                "dataref_side_aux_refdist",
                "dataref_side_aux_point",
                "dataref_side_aux_valid",
                "dataref_side_aux_exist",
            )
        },
    )


if __name__ == "__main__":
    main()
