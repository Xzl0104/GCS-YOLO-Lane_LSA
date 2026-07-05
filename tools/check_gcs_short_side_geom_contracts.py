from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_gcs import parse_args  # noqa: E402
from ultralytics.cfg import CFG_BOOL_KEYS, CFG_FLOAT_KEYS, CFG_INT_KEYS  # noqa: E402
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.models.yolo.gcs_lane.val import DEFAULT_LOSS_GAINS, LOSS_GAIN_ARGS, LOSS_NAMES  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402


def _make_gt(num_lanes: int = 4, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Build simple fixed-y-like GT lanes with short side lanes."""
    if num_lanes == 4:
        xs = [0.10, 0.30, 0.60, 0.90]
        visible_counts = [2, 2, 5, 2]
    elif num_lanes == 5:
        xs = [0.10, 0.25, 0.45, 0.65, 0.90]
        visible_counts = [2, 2, 2, 2, 2]
    else:
        raise ValueError(f"unsupported synthetic lane count: {num_lanes}")

    gt = torch.zeros(num_lanes, k, 2)
    valid = torch.zeros(num_lanes, k)
    for lane_i, (x, visible) in enumerate(zip(xs, visible_counts)):
        gt[lane_i, :, 0] = x
        gt[lane_i, :, 1] = torch.linspace(0.9, 0.1, k)
        valid[lane_i, :visible] = 1.0
    return gt, valid


def _make_preds_from_gt(gt: torch.Tensor, valid: torch.Tensor, q: int = 8) -> tuple[dict[str, torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Build predictions close enough for deterministic synthetic matched-lane checks."""
    num_lanes, k = gt.shape[:2]
    pred_points = torch.zeros(1, q, k, 2)
    pred_logits = torch.full((1, q), -4.0)
    pred_valid_logits = torch.full((1, q, k), -4.0)
    for lane_i in range(num_lanes):
        pred_points[0, lane_i] = gt[lane_i]
        pred_logits[0, lane_i] = 4.0
        pred_valid_logits[0, lane_i, valid[lane_i] > 0.5] = 4.0

    # Unmatched query shaped like a side lane; it must not be selected by the loss.
    if q > num_lanes:
        pred_points[0, num_lanes] = gt[0]
        pred_logits[0, num_lanes] = 4.0
        pred_valid_logits[0, num_lanes, valid[0] > 0.5] = 4.0

    preds = {
        "pred_points": pred_points,
        "pred_logits": pred_logits,
        "pred_valid_logits": pred_valid_logits,
    }
    indices = [(torch.arange(num_lanes, dtype=torch.long), torch.arange(num_lanes, dtype=torch.long))]
    return preds, indices


def check_loss_names_and_gains() -> None:
    """Check train/val loss vectors and validation gain mapping stay aligned."""
    assert GCSLoss.loss_names == GCSLaneTrainer.loss_names
    assert GCSLoss.loss_names == LOSS_NAMES
    assert len(GCSLoss.loss_names) == len(GCSLaneTrainer.progress_loss_names)
    assert len(LOSS_NAMES) == len(LOSS_GAIN_ARGS) == len(DEFAULT_LOSS_GAINS)
    assert len(GCSLoss.loss_names) == len(LOSS_NAMES)

    idx = LOSS_NAMES.index("short_side_geom_loss")
    assert LOSS_GAIN_ARGS[idx] == "gcs_short_side_geom"
    assert LOSS_GAIN_ARGS[idx + 1 : idx + 4] == (None, None, None)
    assert DEFAULT_LOSS_GAINS[idx : idx + 4] == (0.0, 0.0, 0.0, 0.0)


def check_cli_and_config_defaults() -> None:
    """Check train CLI defaults, default.yaml values, and Ultralytics config typing."""
    expected = {
        "gcs_short_side_geom": 0.0,
        "gcs_short_side_geom_min_gt_lanes": 4,
        "gcs_short_side_geom_visible_max": 20,
        "gcs_short_side_geom_side_only": False,
        "gcs_short_side_geom_weight_gt4": 1.0,
        "gcs_short_side_geom_weight_gt5": 1.0,
    }
    args = parse_args([])
    for name, value in expected.items():
        assert getattr(args, name) == value, (name, getattr(args, name), value)

    defaults = yaml.safe_load((ROOT / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8"))
    for name, value in expected.items():
        assert defaults[name] == value, (name, defaults[name], value)

    for name in ("gcs_short_side_geom", "gcs_short_side_geom_weight_gt4", "gcs_short_side_geom_weight_gt5"):
        assert name in CFG_FLOAT_KEYS, name
    for name in ("gcs_short_side_geom_min_gt_lanes", "gcs_short_side_geom_visible_max"):
        assert name in CFG_INT_KEYS, name
    assert "gcs_short_side_geom_side_only" in CFG_BOOL_KEYS


def check_direct_loss_selection() -> None:
    """Check default-off, matched-only, visible-max, side-only, and GT-count weights."""
    gt4, valid4 = _make_gt(4)
    preds4, indices4 = _make_preds_from_gt(gt4, valid4)

    crit_off = GCSLoss({"gcs_imgsz": [544, 960]})
    loss, count, gt4_count, gt5_count = crit_off.short_side_geom_loss(
        preds4["pred_points"], [gt4], [valid4], indices4, torch.tensor([4.0])
    )
    assert float(loss) == 0.0
    assert (float(count), float(gt4_count), float(gt5_count)) == (0.0, 0.0, 0.0)

    crit_all_short = GCSLoss(
        {"gcs_imgsz": [544, 960], "gcs_short_side_geom": 1.0, "gcs_short_side_geom_visible_max": 3}
    )
    loss, count, gt4_count, gt5_count = crit_all_short.short_side_geom_loss(
        preds4["pred_points"], [gt4], [valid4], indices4, torch.tensor([4.0])
    )
    assert float(loss) == 0.0
    assert (float(count), float(gt4_count), float(gt5_count)) == (3.0, 3.0, 0.0)

    crit_side = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_side_geom": 1.0,
            "gcs_short_side_geom_visible_max": 3,
            "gcs_short_side_geom_side_only": True,
        }
    )
    loss, count, gt4_count, gt5_count = crit_side.short_side_geom_loss(
        preds4["pred_points"], [gt4], [valid4], indices4, torch.tensor([4.0])
    )
    assert float(loss) == 0.0
    assert (float(count), float(gt4_count), float(gt5_count)) == (2.0, 2.0, 0.0)

    shifted4 = {k: v.clone() for k, v in preds4.items()}
    shifted4["pred_points"][0, :4, :, 0] += 0.01
    crit_gt4_w1 = GCSLoss(
        {"gcs_imgsz": [544, 960], "gcs_short_side_geom": 1.0, "gcs_short_side_geom_visible_max": 3}
    )
    crit_gt4_w3 = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_side_geom": 1.0,
            "gcs_short_side_geom_visible_max": 3,
            "gcs_short_side_geom_weight_gt4": 3.0,
        }
    )
    loss_w1, *_ = crit_gt4_w1.short_side_geom_loss(shifted4["pred_points"], [gt4], [valid4], indices4, torch.tensor([4.0]))
    loss_w3, *_ = crit_gt4_w3.short_side_geom_loss(shifted4["pred_points"], [gt4], [valid4], indices4, torch.tensor([4.0]))
    assert abs(float(loss_w3) - 3.0 * float(loss_w1)) < 1e-6

    gt5, valid5 = _make_gt(5)
    preds5, indices5 = _make_preds_from_gt(gt5, valid5)
    shifted5 = {k: v.clone() for k, v in preds5.items()}
    shifted5["pred_points"][0, :5, :, 0] += 0.01
    crit_gt5_w1 = GCSLoss(
        {"gcs_imgsz": [544, 960], "gcs_short_side_geom": 1.0, "gcs_short_side_geom_visible_max": 3}
    )
    crit_gt5_w2 = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_side_geom": 1.0,
            "gcs_short_side_geom_visible_max": 3,
            "gcs_short_side_geom_weight_gt5": 2.0,
        }
    )
    loss_w1, count, gt4_count, gt5_count = crit_gt5_w1.short_side_geom_loss(
        shifted5["pred_points"], [gt5], [valid5], indices5, torch.tensor([5.0])
    )
    loss_w2, *_ = crit_gt5_w2.short_side_geom_loss(shifted5["pred_points"], [gt5], [valid5], indices5, torch.tensor([5.0]))
    assert (float(count), float(gt4_count), float(gt5_count)) == (5.0, 0.0, 5.0)
    assert abs(float(loss_w2) - 2.0 * float(loss_w1)) < 1e-6


def check_forward_total_gain() -> None:
    """Check full GCSLoss.forward keeps default-off identity and applies the configured gain."""
    gt, valid = _make_gt(4)
    preds, _ = _make_preds_from_gt(gt, valid)
    preds["pred_points"][0, 0, :, 0] += 0.01
    preds["pred_points"][0, 3, :, 0] -= 0.01
    batch = {"lanes": [gt], "lane_valid": [valid], "num_lanes": torch.tensor([4.0])}

    crit_off = GCSLoss({"gcs_imgsz": [544, 960]})
    total_off, items_off = crit_off(preds, batch)
    idx = GCSLoss.loss_names.index("short_side_geom_loss")
    assert float(items_off[idx]) == 0.0

    crit_on = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_side_geom": 0.5,
            "gcs_short_side_geom_visible_max": 3,
            "gcs_short_side_geom_side_only": True,
        }
    )
    total_on, items_on = crit_on(preds, batch)
    assert int(items_on.numel()) == len(GCSLoss.loss_names)
    assert float(items_on[idx]) > 0.0
    assert float(items_on[GCSLoss.loss_names.index("short_side_geom_count")]) == 2.0
    assert abs(float(total_on - total_off) - 0.5 * float(items_on[idx])) < 1e-5


def main() -> None:
    check_loss_names_and_gains()
    check_cli_and_config_defaults()
    check_direct_loss_selection()
    check_forward_total_gain()
    print("GCS short-side geometry loss contract checks passed.")


if __name__ == "__main__":
    main()
