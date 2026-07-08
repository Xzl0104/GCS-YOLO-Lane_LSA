from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402
from ultralytics.utils.gcs_point_loss import aspect_weighted_l1_point_loss  # noqa: E402


def _quadratic_x_with_laplacian_px(num_points: int, laplacian_px: float) -> torch.Tensor:
    idx = torch.arange(num_points, dtype=torch.float32)
    return (float(laplacian_px) / 960.0) * idx * (idx - 1.0) * 0.5


def test_short_geom_lane_weights_gt5_short() -> None:
    loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 1.0})
    gt_valid = torch.zeros(5, 56)
    gt_valid[0, :20] = 1
    gt_valid[1, :20] = 1
    gt_valid[2, :20] = 1
    gt_valid[3, :20] = 1
    gt_valid[4, :8] = 1

    weights = loss._short_geom_lane_weights(gt_valid, torch.tensor(5))
    if not weights[4] > 1.0:
        raise AssertionError(f"short GT5 lane should be boosted, got weights={weights.tolist()}.")
    if not weights[:4].eq(1.0).all():
        raise AssertionError(f"non-short GT5 lanes should stay 1.0, got weights={weights.tolist()}.")


def test_short_geom_lane_weights_non_gt5_no_boost() -> None:
    loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 1.0})
    gt_valid = torch.zeros(4, 56)
    gt_valid[0, :20] = 1
    gt_valid[1, :20] = 1
    gt_valid[2, :20] = 1
    gt_valid[3, :8] = 1

    weights = loss._short_geom_lane_weights(gt_valid, torch.tensor(4))
    if not weights.eq(1.0).all():
        raise AssertionError(f"non-GT5 image should not be boosted, got weights={weights.tolist()}.")


def test_short_geom_default_point_loss_matches_visible_point_average() -> None:
    loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 0.0})
    pred_points = torch.zeros(1, 2, 4, 2)
    gt_points = [torch.zeros(2, 4, 2)]
    gt_valid = [torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])]
    pred_points[0, 0, :, 0] = 0.1
    pred_points[0, 1, 0, 0] = 0.5
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    got = loss.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5]))
    expected = pred_points.new_tensor((0.1 * 4.0 + 0.5) / 5.0)
    if not torch.allclose(got, expected):
        raise AssertionError(f"default-off point_loss changed: got={float(got):.8f}, expected={float(expected):.8f}.")


def test_short_geom_default_point_loss_matches_old_path_with_no_valid_image() -> None:
    loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 0.0})
    pred_points = torch.zeros(2, 1, 4, 2)
    gt_points = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    gt_valid = [
        torch.zeros(1, 4),
        torch.ones(1, 4),
    ]
    pred_points[1, 0, :, 0] = 0.25
    indices = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([0]), torch.tensor([0])),
    ]

    got = loss.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5, 5]))
    expected_terms = []
    for b, (src_idx, tgt_idx) in enumerate(indices):
        expected_terms.append(
            aspect_weighted_l1_point_loss(
                pred_points[b, src_idx],
                gt_points[b][tgt_idx],
                gt_valid[b][tgt_idx] > 0.5,
                image_size=loss.image_size,
                y_weight=1.0,
                x_only=False,
            )
        )
    expected = torch.stack(expected_terms).mean()
    if not torch.allclose(got, expected):
        raise AssertionError(
            f"default-off point_loss old-path mismatch: got={float(got):.8f}, expected={float(expected):.8f}."
        )


def test_short_geom_enabled_point_loss_keeps_gt4_no_boost_old_aggregation() -> None:
    off_loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 0.0})
    on_loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_visible_thr": 8,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    pred_points = torch.zeros(1, 2, 4, 2)
    gt_points = [torch.zeros(4, 4, 2)]
    gt_valid_b = torch.zeros(4, 4)
    gt_valid_b[0, :] = 1.0
    gt_valid_b[3, 0] = 1.0
    gt_valid = [gt_valid_b]
    pred_points[0, 0, :, 0] = 0.1
    pred_points[0, 1, 0, 0] = 0.5
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 3]))]

    got_off = off_loss.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([4]))
    got_on = on_loss.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([4]))
    expected = pred_points.new_tensor(0.18)
    if not torch.allclose(got_off, expected, atol=1e-6):
        raise AssertionError(f"GT4 default-off point_loss fixture changed: got={float(got_off):.8f}.")
    if not torch.allclose(got_on, expected, atol=1e-6):
        raise AssertionError(
            "GT4 no-boost point_loss should keep old anchor aggregation: "
            f"got={float(got_on):.8f}, expected=0.18000000."
        )


def test_short_geom_enabled_point_loss_keeps_boosted_inactive_old_aggregation() -> None:
    off_loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 0.0})
    on_loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_visible_thr": 8,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    k = 12
    pred_points = torch.zeros(1, 3, k, 2)
    gt_points = [torch.zeros(5, k, 2)]
    gt_valid_b = torch.zeros(5, k)
    gt_valid_b[0, :] = 1.0
    gt_valid_b[3, :9] = 1.0
    gt_valid = [gt_valid_b]
    pred_points[0, 0, :, 0] = 0.1
    pred_points[0, 1, :9, 0] = 0.5
    pred_points[0, 2, :, 0] = 0.9
    indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 3, 4]))]

    got_off = off_loss.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5]))
    got_on = on_loss.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5]))
    expected = pred_points.new_tensor((12.0 * 0.1 + 9.0 * 0.5) / 21.0)
    lane_level_wrong = pred_points.new_tensor(0.30)
    if not torch.allclose(got_off, expected, atol=1e-6):
        raise AssertionError(f"GT5 boosted-inactive default-off point fixture changed: got={float(got_off):.8f}.")
    if torch.allclose(got_on, lane_level_wrong, atol=1e-6):
        raise AssertionError(f"point_loss switched to lane-level aggregation too early: got={float(got_on):.8f}.")
    if not torch.allclose(got_on, expected, atol=1e-6):
        raise AssertionError(
            "boosted but inactive point lane should keep old anchor aggregation: "
            f"got={float(got_on):.8f}, expected={float(expected):.8f}."
        )


def test_short_geom_point_loss_uses_lane_level_weighting() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_visible_thr": 8,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    k = 56
    pred_points = torch.zeros(1, 2, k, 2)
    gt_points = [torch.zeros(5, k, 2)]
    gt_valid_b = torch.zeros(5, k)
    gt_valid_b[0, :] = 1.0
    gt_valid_b[4, :8] = 1.0
    gt_valid = [gt_valid_b]
    pred_points[0, 0, :, 0] = 0.1
    pred_points[0, 1, :8, 0] = 0.5
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 4]))]

    got = loss.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5]))
    expected = pred_points.new_tensor((0.1 + 0.5 * 2.0) / 3.0)
    anchor_level = pred_points.new_tensor((56.0 * 0.1 + 8.0 * 0.5 * 2.0) / (56.0 + 8.0))
    if torch.allclose(got, anchor_level, atol=1e-6):
        raise AssertionError(f"point_loss used anchor-level weighting: got={float(got):.8f}.")
    if not torch.allclose(got, expected, atol=1e-6):
        raise AssertionError(
            f"enabled point_loss lane-level mismatch: got={float(got):.8f}, expected={float(expected):.8f}."
        )


def test_short_geom_enabled_point_loss_keeps_no_valid_zero_in_batch_mean() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_visible_thr": 8,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    pred_points = torch.zeros(2, 1, 4, 2)
    gt_points = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    gt_valid = [
        torch.zeros(1, 4),
        torch.ones(1, 4),
    ]
    pred_points[1, 0, :, 0] = 0.25
    indices = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([0]), torch.tensor([0])),
    ]

    got = loss.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5, 5]))
    expected = pred_points.new_tensor((0.0 + 0.25) / 2.0)
    if not torch.allclose(got, expected):
        raise AssertionError(
            "enabled point_loss should keep no-valid matched images in the batch mean: "
            f"got={float(got):.8f}, expected={float(expected):.8f}."
        )


def test_short_geom_default_curve_loss_keeps_no_triplet_zero_in_batch_mean() -> None:
    loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 0.0})
    pred_points = torch.zeros(2, 1, 4, 2)
    gt_points = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    gt_valid = [
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
    ]
    pred_points[1, 0, :, 0] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    indices = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([0]), torch.tensor([0])),
    ]

    got = loss.curve_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5, 5]))
    expected = pred_points.new_tensor(959.75)
    if not torch.allclose(got, expected, atol=1e-6):
        raise AssertionError(f"default-off curve_loss changed: got={float(got):.8f}, expected=959.75000000.")


def test_short_geom_enabled_curve_loss_keeps_gt4_no_boost_old_aggregation() -> None:
    off_loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 0.0})
    on_loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_curve": 1.0,
            "gcs_short_geom_visible_thr": 8,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    k = 58
    pred_points = torch.zeros(1, 2, k, 2)
    gt_points = [torch.zeros(4, k, 2)]
    gt_valid_b = torch.zeros(4, k)
    gt_valid_b[0, :] = 1.0
    gt_valid_b[3, :10] = 1.0
    gt_valid = [gt_valid_b]
    pred_points[0, 0, :, 0] = _quadratic_x_with_laplacian_px(k, 2496.0)
    pred_points[0, 1, :, 0] = _quadratic_x_with_laplacian_px(k, 7104.0)
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 3]))]

    got_off = off_loss.curve_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([4]))
    got_on = on_loss.curve_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([4]))
    expected = pred_points.new_tensor(3071.5)
    if not torch.allclose(got_off, expected, atol=1e-3):
        raise AssertionError(f"GT4 default-off curve_loss fixture changed: got={float(got_off):.8f}.")
    if not torch.allclose(got_on, expected, atol=1e-3):
        raise AssertionError(
            "GT4 no-boost curve_loss should keep old triplet aggregation: "
            f"got={float(got_on):.8f}, expected=3071.50000000."
        )


def test_short_geom_enabled_curve_loss_keeps_boosted_inactive_old_aggregation() -> None:
    off_loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 0.0})
    on_loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_curve": 1.0,
            "gcs_short_geom_visible_thr": 8,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    k = 58
    pred_points = torch.zeros(1, 3, k, 2)
    gt_points = [torch.zeros(5, k, 2)]
    gt_valid_b = torch.zeros(5, k)
    gt_valid_b[0, :] = 1.0
    gt_valid_b[3, :10] = 1.0
    gt_valid_b[4, :2] = 1.0
    gt_valid = [gt_valid_b]
    pred_points[0, 0, :, 0] = _quadratic_x_with_laplacian_px(k, 2496.0)
    pred_points[0, 1, :, 0] = _quadratic_x_with_laplacian_px(k, 7104.0)
    pred_points[0, 2, :, 0] = _quadratic_x_with_laplacian_px(k, 9600.0)
    indices = [(torch.tensor([0, 1, 2]), torch.tensor([0, 3, 4]))]

    got_off = off_loss.curve_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5]))
    got_on = on_loss.curve_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5]))
    expected = pred_points.new_tensor(3071.5)
    lane_level_wrong = pred_points.new_tensor(4799.5)
    if not torch.allclose(got_off, expected, atol=1e-3):
        raise AssertionError(f"GT5 boosted-inactive default-off curve fixture changed: got={float(got_off):.8f}.")
    if torch.allclose(got_on, lane_level_wrong, atol=1e-3):
        raise AssertionError(f"curve_loss switched to lane-level aggregation too early: got={float(got_on):.8f}.")
    if not torch.allclose(got_on, expected, atol=1e-3):
        raise AssertionError(
            "boosted but inactive curve lane should keep old triplet aggregation: "
            f"got={float(got_on):.8f}, expected={float(expected):.8f}."
        )


def test_short_geom_curve_loss_uses_lane_level_weighting() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_curve": 1.0,
            "gcs_short_geom_visible_thr": 10,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    k = 58
    pred_points = torch.zeros(1, 2, k, 2)
    gt_points = [torch.zeros(5, k, 2)]
    gt_valid_b = torch.zeros(5, k)
    gt_valid_b[0, :] = 1.0
    gt_valid_b[4, :10] = 1.0
    gt_valid = [gt_valid_b]
    pred_points[0, 0, :, 0] = _quadratic_x_with_laplacian_px(k, 0.2**0.5)
    pred_points[0, 1, :, 0] = _quadratic_x_with_laplacian_px(k, 1.0)
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 4]))]

    got = loss.curve_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5]))
    expected = pred_points.new_tensor((0.1 + 0.5 * 2.0) / 3.0)
    triplet_level = pred_points.new_tensor((56.0 * 0.1 + 8.0 * 0.5 * 2.0) / (56.0 + 8.0))
    if torch.allclose(got, triplet_level, atol=1e-5):
        raise AssertionError(f"curve_loss used triplet-level weighting: got={float(got):.8f}.")
    if not torch.allclose(got, expected, atol=1e-5):
        raise AssertionError(
            f"enabled curve_loss lane-level mismatch: got={float(got):.8f}, expected={float(expected):.8f}."
        )


def test_short_geom_enabled_losses_keep_gt5_no_short_old_aggregation() -> None:
    off_loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 0.0})
    on_loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_curve": 1.0,
            "gcs_short_geom_visible_thr": 8,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )

    point_pred = torch.zeros(1, 2, 20, 2)
    point_gt = [torch.zeros(5, 20, 2)]
    point_valid_b = torch.zeros(5, 20)
    point_valid_b[0, :18] = 1.0
    point_valid_b[1, :12] = 1.0
    point_valid_b[2, :12] = 1.0
    point_valid_b[3, :12] = 1.0
    point_valid_b[4, :9] = 1.0
    point_valid = [point_valid_b]
    point_pred[0, 0, :18, 0] = 0.1
    point_pred[0, 1, :9, 0] = 0.5
    point_indices = [(torch.tensor([0, 1]), torch.tensor([0, 4]))]
    point_off = off_loss.point_loss(point_pred, point_gt, point_valid, point_indices, gt_lanes=torch.tensor([5]))
    point_on = on_loss.point_loss(point_pred, point_gt, point_valid, point_indices, gt_lanes=torch.tensor([5]))
    if not torch.allclose(point_on, point_off, atol=1e-6):
        raise AssertionError(
            "GT5 no-short point_loss should keep old anchor aggregation: "
            f"enabled={float(point_on):.8f}, default={float(point_off):.8f}."
        )

    k = 20
    curve_pred = torch.zeros(1, 2, k, 2)
    curve_gt = [torch.zeros(5, k, 2)]
    curve_valid_b = torch.zeros(5, k)
    curve_valid_b[0, :] = 1.0
    curve_valid_b[1, :12] = 1.0
    curve_valid_b[2, :12] = 1.0
    curve_valid_b[3, :12] = 1.0
    curve_valid_b[4, :10] = 1.0
    curve_valid = [curve_valid_b]
    curve_pred[0, 0, :, 0] = _quadratic_x_with_laplacian_px(k, 64.0)
    curve_pred[0, 1, :, 0] = _quadratic_x_with_laplacian_px(k, 512.0)
    curve_indices = [(torch.tensor([0, 1]), torch.tensor([0, 4]))]
    curve_off = off_loss.curve_loss(curve_pred, curve_gt, curve_valid, curve_indices, gt_lanes=torch.tensor([5]))
    curve_on = on_loss.curve_loss(curve_pred, curve_gt, curve_valid, curve_indices, gt_lanes=torch.tensor([5]))
    if not torch.allclose(curve_on, curve_off, atol=1e-5):
        raise AssertionError(
            "GT5 no-short curve_loss should keep old triplet aggregation: "
            f"enabled={float(curve_on):.8f}, default={float(curve_off):.8f}."
        )


def test_short_geom_enabled_curve_loss_keeps_no_triplet_zero_in_batch_mean() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_curve": 1.0,
            "gcs_short_geom_visible_thr": 8,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    pred_points = torch.zeros(2, 1, 4, 2)
    gt_points = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    gt_valid = [
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
    ]
    pred_points[1, 0, :, 0] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    indices = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([0]), torch.tensor([0])),
    ]

    got = loss.curve_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5, 5]))
    expected = pred_points.new_tensor(959.75)
    if not torch.allclose(got, expected, atol=1e-6):
        raise AssertionError(
            "enabled curve_loss should keep no-triplet matched images in the batch mean: "
            f"got={float(got):.8f}, expected={float(expected):.8f}."
        )


def test_short_geom_forward_keeps_loss_items_stable() -> None:
    criterion = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 1.0})
    b, q, k, n = 1, 12, 56, 5
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    lane_x = torch.linspace(0.2, 0.8, n).view(n, 1).expand(n, k)
    lanes = torch.stack((lane_x, y.view(1, k).expand(n, k)), dim=-1).float()
    lane_valid = torch.ones(n, k)
    lane_valid[-1, 8:] = 0

    pred_points = torch.zeros(b, q, k, 2)
    pred_points[0, :n] = lanes
    pred_points[0, n:] = lanes[-1:].expand(q - n, k, 2)
    pred_points[0, n - 1, :8, 0] += 0.02
    preds = {
        "pred_points": pred_points,
        "pred_logits": torch.zeros(b, q),
        "pred_valid_logits": torch.full((b, q, k), 4.0),
    }
    batch = {
        "lanes": [lanes],
        "lane_valid": [lane_valid],
        "num_lanes": torch.tensor([5], dtype=torch.long),
    }

    total, items = criterion(preds, batch)
    if int(items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(f"loss item length changed: got={items.numel()}, expected={len(GCSLoss.loss_names)}.")
    if not torch.isfinite(total) or not torch.isfinite(items).all():
        raise AssertionError("short-geom forward produced non-finite loss values.")


def main() -> None:
    test_short_geom_lane_weights_gt5_short()
    test_short_geom_lane_weights_non_gt5_no_boost()
    test_short_geom_default_point_loss_matches_visible_point_average()
    test_short_geom_default_point_loss_matches_old_path_with_no_valid_image()
    test_short_geom_enabled_point_loss_keeps_gt4_no_boost_old_aggregation()
    test_short_geom_enabled_point_loss_keeps_boosted_inactive_old_aggregation()
    test_short_geom_point_loss_uses_lane_level_weighting()
    test_short_geom_enabled_point_loss_keeps_no_valid_zero_in_batch_mean()
    test_short_geom_default_curve_loss_keeps_no_triplet_zero_in_batch_mean()
    test_short_geom_enabled_curve_loss_keeps_gt4_no_boost_old_aggregation()
    test_short_geom_enabled_curve_loss_keeps_boosted_inactive_old_aggregation()
    test_short_geom_curve_loss_uses_lane_level_weighting()
    test_short_geom_enabled_losses_keep_gt5_no_short_old_aggregation()
    test_short_geom_enabled_curve_loss_keeps_no_triplet_zero_in_batch_mean()
    test_short_geom_forward_keeps_loss_items_stable()
    print(json.dumps({"status": "ok", "tests": 15}, indent=2))


if __name__ == "__main__":
    main()
