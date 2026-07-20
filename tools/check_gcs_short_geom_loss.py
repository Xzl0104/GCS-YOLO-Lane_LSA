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


def _constant_lane_fixture(x_values: list[float], k: int, visible_counts: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    gt_points = torch.zeros(len(x_values), k, 2)
    gt_valid = torch.zeros(len(x_values), k)
    for i, x in enumerate(x_values):
        gt_points[i, :, 0] = float(x)
        if visible_counts is None:
            gt_valid[i, :] = 1.0
        else:
            gt_valid[i, : int(visible_counts[i])] = 1.0
    return gt_points, gt_valid


def test_short_geom_lane_weights_gt5_short() -> None:
    loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 1.0})
    gt_valid = torch.zeros(5, 56)
    gt_valid[0, :20] = 1
    gt_valid[1, :20] = 1
    gt_valid[2, :20] = 1
    gt_valid[3, :20] = 1
    gt_valid[4, :8] = 1

    weights = loss._short_geom_lane_weights(gt_valid, torch.tensor(5))
    if not torch.allclose(weights[4], torch.tensor(2.0)):
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
        raise AssertionError(f"GT4 default weight should not boost lanes, got weights={weights.tolist()}.")


def test_short_geom_rejects_gt4_weight_below_one() -> None:
    try:
        GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 1.0, "gcs_short_geom_gt4_weight": 0.5})
    except ValueError as exc:
        if "gcs_short_geom_gt4_weight" not in str(exc):
            raise AssertionError(f"unexpected ValueError for gt4 weight: {exc}") from exc
        return
    raise AssertionError("gcs_short_geom_gt4_weight < 1.0 should raise ValueError.")


def test_short_geom_lane_weights_gt4_short_enabled() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_visible_thr": 20,
            "gcs_short_geom_gt4_weight": 1.5,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    gt_valid = torch.zeros(4, 56)
    gt_valid[0, :21] = 1
    gt_valid[1, :30] = 1
    gt_valid[2, :20] = 1
    gt_valid[3, :8] = 1

    weights = loss._short_geom_lane_weights(gt_valid, torch.tensor(4))
    expected = torch.tensor([1.0, 1.0, 1.5, 1.5])
    if not torch.allclose(weights.cpu(), expected):
        raise AssertionError(f"GT4 short lanes should use gt4 boost, got weights={weights.tolist()}.")


def test_short_geom_lane_weights_gt3_no_boost() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_visible_thr": 20,
            "gcs_short_geom_gt4_weight": 1.5,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    gt_valid = torch.zeros(3, 56)
    gt_valid[0, :8] = 1
    gt_valid[1, :20] = 1
    gt_valid[2, :21] = 1

    weights = loss._short_geom_lane_weights(gt_valid, torch.tensor(3))
    if not weights.eq(1.0).all():
        raise AssertionError(f"GT3 lanes should not be boosted, got weights={weights.tolist()}.")


def test_short_geom_lane_weights_visible_gt20_no_boost() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_visible_thr": 20,
            "gcs_short_geom_gt4_weight": 1.5,
            "gcs_short_geom_gt5_weight": 2.0,
        }
    )
    gt_valid = torch.zeros(5, 56)
    gt_valid[:, :21] = 1

    weights = loss._short_geom_lane_weights(gt_valid, torch.tensor(5))
    if not weights.eq(1.0).all():
        raise AssertionError(f"visible_count > 20 lanes should not be boosted, got weights={weights.tolist()}.")


def test_short_geom_lane_weights_max_cap() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 2.0,
            "gcs_short_geom_visible_thr": 20,
            "gcs_short_geom_gt4_weight": 3.0,
            "gcs_short_geom_max_weight": 2.5,
        }
    )
    gt_valid = torch.zeros(4, 56)
    gt_valid[3, :20] = 1

    weights = loss._short_geom_lane_weights(gt_valid, torch.tensor(4))
    if not torch.allclose(weights[3], torch.tensor(2.5)):
        raise AssertionError(f"short geometry max cap should apply, got weights={weights.tolist()}.")


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


def test_short_geom_point_loss_uses_gt4_lane_level_weighting() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_visible_thr": 20,
            "gcs_short_geom_gt4_weight": 1.5,
            "gcs_short_geom_gt5_weight": 1.0,
        }
    )
    k = 56
    pred_points = torch.zeros(1, 2, k, 2)
    gt_points = [torch.zeros(4, k, 2)]
    gt_valid_b = torch.zeros(4, k)
    gt_valid_b[0, :] = 1.0
    gt_valid_b[3, :20] = 1.0
    gt_valid = [gt_valid_b]
    pred_points[0, 0, :, 0] = 0.1
    pred_points[0, 1, :20, 0] = 0.5
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 3]))]

    got = loss.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([4]))
    expected = pred_points.new_tensor((0.1 + 0.5 * 1.5) / 2.5)
    anchor_level = pred_points.new_tensor((56.0 * 0.1 + 20.0 * 0.5 * 1.5) / (56.0 + 20.0))
    if torch.allclose(got, anchor_level, atol=1e-6):
        raise AssertionError(f"GT4 point_loss used anchor-level weighting: got={float(got):.8f}.")
    if not torch.allclose(got, expected, atol=1e-6):
        raise AssertionError(
            f"GT4 enabled point_loss lane-level mismatch: got={float(got):.8f}, expected={float(expected):.8f}."
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


def test_short_geom_curve_loss_uses_gt4_lane_level_weighting() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_curve": 1.0,
            "gcs_short_geom_visible_thr": 20,
            "gcs_short_geom_gt4_weight": 1.5,
            "gcs_short_geom_gt5_weight": 1.0,
        }
    )
    k = 58
    pred_points = torch.zeros(1, 2, k, 2)
    gt_points = [torch.zeros(4, k, 2)]
    gt_valid_b = torch.zeros(4, k)
    gt_valid_b[0, :] = 1.0
    gt_valid_b[3, :20] = 1.0
    gt_valid = [gt_valid_b]
    pred_points[0, 0, :, 0] = _quadratic_x_with_laplacian_px(k, 0.2**0.5)
    pred_points[0, 1, :, 0] = _quadratic_x_with_laplacian_px(k, 1.0)
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 3]))]

    got = loss.curve_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([4]))
    expected = pred_points.new_tensor((0.1 + 0.5 * 1.5) / 2.5)
    triplet_level = pred_points.new_tensor((56.0 * 0.1 + 18.0 * 0.5 * 1.5) / (56.0 + 18.0))
    if torch.allclose(got, triplet_level, atol=1e-5):
        raise AssertionError(f"GT4 curve_loss used triplet-level weighting: got={float(got):.8f}.")
    if not torch.allclose(got, expected, atol=1e-5):
        raise AssertionError(
            f"GT4 enabled curve_loss lane-level mismatch: got={float(got):.8f}, expected={float(expected):.8f}."
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


def test_short_geom_lane_weights_gt5_outer_and_focus() -> None:
    loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 1.0})
    gt_valid = torch.zeros(5, 56)
    gt_valid[0, :21] = 1
    gt_valid[1, :20] = 1
    gt_valid[2, :10] = 1
    gt_valid[3, :9] = 1
    gt_valid[4, :8] = 1

    weights = loss._short_geom_lane_weights(gt_valid, torch.tensor(5))
    expected = torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0])
    if not torch.allclose(weights.cpu(), expected):
        raise AssertionError(f"GT5 outer/focus weights changed: got={weights.tolist()}, expected={expected.tolist()}.")


def test_short_geom_lane_weights_gt4_outer_and_focus() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_gt4_weight": 1.5,
        }
    )
    gt_valid = torch.zeros(4, 56)
    gt_valid[0, :21] = 1
    gt_valid[1, :20] = 1
    gt_valid[2, :11] = 1
    gt_valid[3, :10] = 1

    weights = loss._short_geom_lane_weights(gt_valid, torch.tensor(4))
    expected = torch.tensor([1.0, 1.5, 1.5, 2.25])
    if not torch.allclose(weights.cpu(), expected, atol=1e-6):
        raise AssertionError(f"GT4 outer/focus weights changed: got={weights.tolist()}, expected={expected.tolist()}.")


def test_short_geom_lane_weights_focus_cap() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_gt4_weight": 1.5,
            "gcs_short_geom_focus_weight": 3.0,
            "gcs_short_geom_max_weight": 2.5,
        }
    )
    gt_valid = torch.zeros(4, 56)
    gt_valid[3, :10] = 1

    weights = loss._short_geom_lane_weights(gt_valid, torch.tensor(4))
    if not torch.allclose(weights[3], torch.tensor(2.5)):
        raise AssertionError(f"short-geom focus cap changed: got={weights.tolist()}.")


def test_short_geom_point_loss_two_threshold_weighting() -> None:
    loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 1.0})
    k = 21
    pred_points = torch.zeros(1, 2, k, 2)
    gt_points = [torch.zeros(5, k, 2)]
    gt_valid_b = torch.zeros(5, k)
    gt_valid_b[0, :21] = 1.0
    gt_valid_b[4, :8] = 1.0
    gt_valid = [gt_valid_b]
    pred_points[0, 0, :, 0] = 0.1
    pred_points[0, 1, :8, 0] = 0.5
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 4]))]

    got = loss.point_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5]))
    expected = pred_points.new_tensor((0.1 + 0.5 * 3.0) / 4.0)
    anchor_level = pred_points.new_tensor((21.0 * 0.1 + 8.0 * 0.5 * 3.0) / (21.0 + 8.0))
    if torch.allclose(got, anchor_level, atol=1e-6):
        raise AssertionError(f"point_loss used anchor-level weighting: got={float(got):.8f}.")
    if not torch.allclose(got, expected, atol=1e-6):
        raise AssertionError(f"point_loss two-threshold weighting changed: got={float(got):.8f}, expected={float(expected):.8f}.")


def test_short_geom_curve_loss_two_threshold_weighting() -> None:
    off_loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 0.0})
    on_loss = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_geom": 1.0})
    k = 21
    pred_points = torch.zeros(1, 2, k, 2)
    gt_points = [torch.zeros(5, k, 2)]
    gt_valid_b = torch.zeros(5, k)
    gt_valid_b[0, :21] = 1.0
    gt_valid_b[4, :8] = 1.0
    gt_valid = [gt_valid_b]
    pred_points[0, 0, :, 0] = _quadratic_x_with_laplacian_px(k, 2496.0)
    pred_points[0, 1, :, 0] = _quadratic_x_with_laplacian_px(k, 7104.0)
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 4]))]

    got = on_loss.curve_loss(pred_points, gt_points, gt_valid, indices, gt_lanes=torch.tensor([5]))
    lane0 = off_loss.curve_loss(
        pred_points[:, :1],
        [gt_points[0][:1]],
        [gt_valid_b[:1]],
        [(torch.tensor([0]), torch.tensor([0]))],
        gt_lanes=torch.tensor([5]),
    )
    lane1 = off_loss.curve_loss(
        pred_points[:, 1:2],
        [gt_points[0][4:5]],
        [gt_valid_b[4:5]],
        [(torch.tensor([0]), torch.tensor([0]))],
        gt_lanes=torch.tensor([5]),
    )
    expected = (lane0 + lane1 * 3.0) / 4.0
    if not torch.allclose(got, expected, atol=1e-5):
        raise AssertionError(f"curve_loss two-threshold weighting changed: got={float(got):.8f}, expected={float(expected):.8f}.")


def test_short_geom_point_valid_loss_generalizes_to_gt4() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_geom": 1.0,
            "gcs_short_geom_gt4_weight": 1.5,
        }
    )
    pred_valid_logits = torch.zeros(1, 2, 12)
    pred_points = torch.zeros(1, 2, 12, 2)
    gt_points = [torch.zeros(4, 12, 2)]
    gt_valid_b = torch.zeros(4, 12)
    gt_valid_b[0, :8] = 1.0
    gt_valid = [gt_valid_b]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    got, short_pos_count, short_pos_anchor_count, short_point_valid_loss = loss.point_valid_loss(
        pred_valid_logits,
        pred_points,
        gt_valid,
        indices,
        gt_lanes=torch.tensor([4]),
        return_details=True,
    )
    if int(short_pos_count.item()) != 1:
        raise AssertionError(f"GT4 short rescue count changed: got={float(short_pos_count):.8f}.")
    if int(short_pos_anchor_count.item()) != 8:
        raise AssertionError(f"GT4 short rescue anchor count changed: got={float(short_pos_anchor_count):.8f}.")
    if not torch.isfinite(got) or not torch.isfinite(short_point_valid_loss):
        raise AssertionError("GT4 short point-valid rescue produced non-finite loss.")


def test_boundary_pseudo_neg_loss_clear_far_and_permutation_invariance() -> None:
    loss = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_boundary_pseudo_neg": 1.0,
            "gcs_boundary_pseudo_gt_count": 3,
            "gcs_boundary_pseudo_visible_thr": 10,
            "gcs_boundary_pseudo_min_valid": 2,
            "gcs_boundary_pseudo_dist_thr": 60.0,
            "gcs_boundary_pseudo_score_thr": 0.0,
            "gcs_boundary_pseudo_envelope_margin_px": -1.0,
        }
    )
    k = 4
    gt_points_b, gt_valid_b = _constant_lane_fixture([0.10, 0.30, 0.60, 0.85], k)
    gt_points = [gt_points_b]
    gt_valid = [gt_valid_b]
    pred_points = torch.zeros(1, 3, k, 2)
    pred_logits = torch.full((1, 3), 5.0)
    pred_valid_logits = torch.full((1, 3, k), 5.0)
    pred_points[0, 0, :, 0] = 0.10
    pred_points[0, 1, :, 0] = 0.62
    pred_points[0, 2, :, 0] = 0.95
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    got = loss.boundary_pseudo_neg_loss(pred_points, pred_logits, pred_valid_logits, gt_points, gt_valid, indices, torch.tensor([4]))
    if int(got[1].item()) != 1:
        raise AssertionError(f"clear-far negative count changed: got={float(got[1]):.8f}.")

    perm = torch.tensor([2, 0, 1])
    permuted = loss.boundary_pseudo_neg_loss(
        pred_points[:, perm],
        pred_logits[:, perm],
        pred_valid_logits[:, perm],
        gt_points,
        gt_valid,
        [(torch.tensor([1]), torch.tensor([0]))],
        torch.tensor([4]),
    )
    if not torch.allclose(got[0], permuted[0], atol=1e-6):
        raise AssertionError(f"clear-far loss changed under query permutation: got={float(got[0]):.8f}, permuted={float(permuted[0]):.8f}.")
    if not torch.allclose(got[1], permuted[1], atol=1e-6):
        raise AssertionError(f"clear-far count changed under query permutation: got={float(got[1]):.8f}, permuted={float(permuted[1]):.8f}.")
    if not torch.allclose(got[2], permuted[2], atol=1e-6):
        raise AssertionError(f"clear-far score mean changed under query permutation: got={float(got[2]):.8f}, permuted={float(permuted[2]):.8f}.")


def test_boundary_pseudo_valid_neg_weight_targets_selected_valid_logits() -> None:
    base = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_boundary_pseudo_neg": 1.0,
            "gcs_boundary_pseudo_gt_count": 3,
            "gcs_boundary_pseudo_visible_thr": 10,
            "gcs_boundary_pseudo_min_valid": 2,
            "gcs_boundary_pseudo_dist_thr": 60.0,
            "gcs_boundary_pseudo_score_thr": 0.0,
            "gcs_boundary_pseudo_envelope_margin_px": -1.0,
            "gcs_boundary_pseudo_valid_neg_weight": 0.0,
        }
    )
    weighted = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_boundary_pseudo_neg": 1.0,
            "gcs_boundary_pseudo_gt_count": 3,
            "gcs_boundary_pseudo_visible_thr": 10,
            "gcs_boundary_pseudo_min_valid": 2,
            "gcs_boundary_pseudo_dist_thr": 60.0,
            "gcs_boundary_pseudo_score_thr": 0.0,
            "gcs_boundary_pseudo_envelope_margin_px": -1.0,
            "gcs_boundary_pseudo_valid_neg_weight": 0.25,
        }
    )
    k = 4
    gt_points_b, gt_valid_b = _constant_lane_fixture([0.10, 0.30, 0.60, 0.85], k)
    gt_points = [gt_points_b]
    gt_valid = [gt_valid_b]
    pred_points = torch.zeros(1, 3, k, 2)
    pred_logits = torch.full((1, 3), 5.0, requires_grad=True)
    pred_valid_logits = torch.full((1, 3, k), 5.0, requires_grad=True)
    pred_points[0, 0, :, 0] = 0.10
    pred_points[0, 1, :, 0] = 0.62
    pred_points[0, 2, :, 0] = 0.95
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    base_loss = base.boundary_pseudo_neg_loss(
        pred_points.detach(),
        pred_logits.detach(),
        pred_valid_logits.detach(),
        gt_points,
        gt_valid,
        indices,
        torch.tensor([4]),
    )[0]
    weighted_loss = weighted.boundary_pseudo_neg_loss(
        pred_points,
        pred_logits,
        pred_valid_logits,
        gt_points,
        gt_valid,
        indices,
        torch.tensor([4]),
    )[0]
    if float(weighted_loss.detach().item()) <= float(base_loss.detach().item()):
        raise AssertionError(
            f"boundary pseudo valid-neg weight did not increase selected loss: base={float(base_loss):.8f}, "
            f"weighted={float(weighted_loss):.8f}."
        )
    weighted_loss.backward()
    selected_grad = pred_valid_logits.grad[0, 2].abs().sum()
    unselected_grad = pred_valid_logits.grad[0, 1].abs().sum()
    if float(selected_grad.item()) <= 0.0:
        raise AssertionError("boundary pseudo valid-neg weight did not backprop through selected valid logits.")
    if float(unselected_grad.item()) != 0.0:
        raise AssertionError("boundary pseudo valid-neg weight leaked valid-logit gradients to an unselected query.")


def main() -> None:
    test_short_geom_lane_weights_gt5_outer_and_focus()
    test_short_geom_lane_weights_gt4_outer_and_focus()
    test_short_geom_lane_weights_focus_cap()
    test_short_geom_point_loss_two_threshold_weighting()
    test_short_geom_curve_loss_two_threshold_weighting()
    test_short_geom_point_valid_loss_generalizes_to_gt4()
    test_boundary_pseudo_neg_loss_clear_far_and_permutation_invariance()
    test_boundary_pseudo_valid_neg_weight_targets_selected_valid_logits()
    test_short_geom_forward_keeps_loss_items_stable()
    print(json.dumps({"status": "ok", "tests": 9}, indent=2))


if __name__ == "__main__":
    main()
