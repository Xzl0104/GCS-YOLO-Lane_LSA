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
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.models.yolo.gcs_lane import val as gcs_val  # noqa: E402


def _fixture(gt_count: int = 5) -> tuple[dict[str, torch.Tensor], dict, list[tuple[torch.Tensor, torch.Tensor]]]:
    b, q, k, n = 1, 12, 56, gt_count
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    lane_x = torch.linspace(0.2, 0.8, n).view(n, 1).expand(n, k)
    lanes = torch.stack((lane_x, y.view(1, k).expand(n, k)), dim=-1).float()
    lane_valid = torch.ones(n, k)
    lane_valid[-1, 8:] = 0.0

    pred_points = torch.zeros(b, q, k, 2)
    pred_points[0, :n] = lanes
    pred_points[0, n:] = lanes[-1:].expand(q - n, k, 2)
    pred_logits = torch.full((b, q), -4.0)
    pred_logits[0, :n] = 0.0
    pred_valid_logits = torch.full((b, q, k), -3.0)
    pred_valid_logits[0, :n] = 0.0
    preds = {
        "pred_points": pred_points,
        "pred_logits": pred_logits,
        "pred_valid_logits": pred_valid_logits,
    }
    batch = {
        "lanes": [lanes],
        "lane_valid": [lane_valid],
        "num_lanes": torch.tensor([gt_count], dtype=torch.long),
    }
    indices = [(torch.arange(n), torch.arange(n))]
    return preds, batch, indices


def test_rejects_invalid_gain() -> None:
    try:
        GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_survival": -0.1})
    except ValueError as exc:
        if "gcs_short_survival" not in str(exc):
            raise AssertionError(f"unexpected ValueError: {exc}") from exc
        return
    raise AssertionError("negative gcs_short_survival should raise ValueError.")


def test_loss_name_contracts_match() -> None:
    if tuple(GCSLoss.loss_names) != tuple(GCSLaneTrainer.loss_names):
        raise AssertionError("GCSLoss and GCSLaneTrainer loss_names must match exactly.")
    if tuple(GCSLoss.loss_names) != tuple(gcs_val.LOSS_NAMES):
        raise AssertionError("GCSLoss and GCS validator LOSS_NAMES must match exactly.")
    if len(gcs_val.LOSS_NAMES) != len(gcs_val.LOSS_GAIN_ARGS):
        raise AssertionError("GCS validator LOSS_GAIN_ARGS length must match LOSS_NAMES.")
    if len(gcs_val.LOSS_NAMES) != len(gcs_val.DEFAULT_LOSS_GAINS):
        raise AssertionError("GCS validator DEFAULT_LOSS_GAINS length must match LOSS_NAMES.")
    idx = gcs_val.LOSS_NAMES.index("short_survival_loss")
    if gcs_val.LOSS_GAIN_ARGS[idx] != "gcs_short_survival":
        raise AssertionError("short_survival_loss must be weighted by gcs_short_survival in validation.")


def test_default_off_forward_loss_matches_old_total() -> None:
    preds, batch, _ = _fixture(gt_count=5)
    off = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_survival": 0.0})
    ref = GCSLoss({"gcs_imgsz": [544, 960]})

    total_off, items_off = off(preds, batch)
    total_ref, items_ref = ref(preds, batch)
    names = GCSLoss.loss_names
    if len(items_off) != len(names):
        raise AssertionError(f"loss item length mismatch: got={len(items_off)}, expected={len(names)}.")
    if not torch.allclose(total_off, total_ref):
        raise AssertionError(f"default-off total changed: got={float(total_off)}, expected={float(total_ref)}.")
    for name in (
        "short_survival_loss",
        "short_survival_exist_loss",
        "short_survival_valid_loss",
        "short_survival_pos_count",
        "short_survival_anchor_count",
    ):
        value = items_off[names.index(name)]
        if float(value) != 0.0:
            raise AssertionError(f"default-off {name} should be zero, got {float(value)}.")
    if not torch.isfinite(items_ref).all():
        raise AssertionError("default reference produced non-finite items.")


def test_enabled_gt5_short_survival_counts_and_loss() -> None:
    preds, batch, indices = _fixture(gt_count=5)
    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_survival": 0.5,
            "gcs_short_survival_visible_thr": 10,
            "gcs_short_survival_gt5_weight": 2.0,
        }
    )

    loss, exist_loss, valid_loss, pos_count, anchor_count = criterion.short_survival_loss(
        preds["pred_logits"],
        preds["pred_valid_logits"],
        preds["pred_points"],
        batch["lane_valid"],
        indices,
        gt_lanes=batch["num_lanes"],
    )
    if not (float(loss) > 0.0 and float(exist_loss) > 0.0 and float(valid_loss) > 0.0):
        raise AssertionError(
            "enabled GT5 short survival should produce positive losses, "
            f"got loss={float(loss)}, exist={float(exist_loss)}, valid={float(valid_loss)}."
        )
    if int(pos_count.item()) != 1:
        raise AssertionError(f"expected one short GT5 lane, got {int(pos_count.item())}.")
    if int(anchor_count.item()) != 8:
        raise AssertionError(f"expected 8 visible anchors on the short lane, got {int(anchor_count.item())}.")


def test_enabled_gt4_short_survival_counts_and_forward_items() -> None:
    preds, batch, _ = _fixture(gt_count=4)
    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_survival": 1.0,
            "gcs_short_survival_visible_thr": 10,
            "gcs_short_survival_gt4_weight": 1.5,
        }
    )

    total, items = criterion(preds, batch)
    names = GCSLoss.loss_names
    if not torch.isfinite(total) or not torch.isfinite(items).all():
        raise AssertionError("enabled GT4 short survival forward produced non-finite values.")
    if int(items[names.index("short_survival_pos_count")].item()) != 1:
        raise AssertionError("enabled GT4 short survival should count the one short GT4 lane.")
    if int(items[names.index("short_survival_anchor_count")].item()) != 8:
        raise AssertionError("enabled GT4 short survival should count 8 visible anchors.")
    if float(items[names.index("short_survival_loss")]) <= 0.0:
        raise AssertionError("enabled GT4 short survival loss should be positive.")


def test_non_gt4_gt5_ignored() -> None:
    preds, batch, indices = _fixture(gt_count=3)
    criterion = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_survival": 1.0})
    loss, exist_loss, valid_loss, pos_count, anchor_count = criterion.short_survival_loss(
        preds["pred_logits"],
        preds["pred_valid_logits"],
        preds["pred_points"],
        batch["lane_valid"],
        indices,
        gt_lanes=batch["num_lanes"],
    )
    if any(float(x) != 0.0 for x in (loss, exist_loss, valid_loss, pos_count, anchor_count)):
        raise AssertionError("GT3 images should not receive short survival loss.")


def main() -> None:
    test_rejects_invalid_gain()
    test_loss_name_contracts_match()
    test_default_off_forward_loss_matches_old_total()
    test_enabled_gt5_short_survival_counts_and_loss()
    test_enabled_gt4_short_survival_counts_and_forward_items()
    test_non_gt4_gt5_ignored()
    print(json.dumps({"status": "ok", "tests": 6}, indent=2))


if __name__ == "__main__":
    main()
