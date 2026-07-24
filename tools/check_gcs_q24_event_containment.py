from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.gcs_loss import GCSLoss


def _criterion(**overrides) -> GCSLoss:
    args = {
        "gcs_imgsz": [544, 960],
        "gcs_q24_event_contain": 0.4,
        "gcs_q24_event_valid_weight": 0.2,
        "gcs_q24_event_matcher": True,
        "gcs_q24_event_clean_gt5_queries": "12,15,20",
        "gcs_q24_event_risk_queries": "13,21,22,23",
        "gcs_q24_event_gt4_queries": "14,17,18,19",
        "gcs_q24_event_gt4_visible_thr": 20,
        "gcs_q24_event_gt5_visible_thr": 10,
        "gcs_q24_event_suppress_gt5_risk": True,
    }
    args.update(overrides)
    return GCSLoss(args)


def _lane_batch(num_lanes: int, *, short_last: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    k = 6
    y = torch.linspace(710.0 / 720.0, 660.0 / 720.0, k)
    xs = torch.linspace(0.15, 0.75, num_lanes).view(num_lanes, 1).expand(num_lanes, k)
    points = torch.stack((xs, y.view(1, k).expand(num_lanes, k)), dim=-1).float()
    valid = torch.ones(num_lanes, k)
    if short_last:
        valid[-1, 4:] = 0.0
    return points, valid


def _preds() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = 6
    y = torch.linspace(710.0 / 720.0, 660.0 / 720.0, k)
    pred_points = torch.zeros(1, 24, k, 2)
    for q in range(24):
        pred_points[0, q, :, 0] = 0.05 + 0.03 * q
        pred_points[0, q, :, 1] = y
    pred_logits = torch.zeros(1, 24)
    pred_valid_logits = torch.full((1, 24, k), 2.0)
    return pred_points, pred_logits, pred_valid_logits


def test_gt3_event_queries_are_suppressed() -> None:
    loss = _criterion()
    pred_points, pred_logits, pred_valid_logits = _preds()
    gt_points, gt_valid = _lane_batch(3)
    gt_lanes = torch.tensor([3])
    masks = loss._q24_event_allowed_masks(pred_points, [gt_valid], gt_lanes)
    outputs = loss.q24_event_containment_loss(
        pred_points,
        pred_logits,
        pred_valid_logits,
        [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))],
        [gt_points],
        [gt_valid],
        gt_lanes,
        masks,
    )
    event_loss, _exist, _valid, gt3_count, _gt4_count, _gt5_risk, _protected, _clean = outputs
    if int(gt3_count.item()) != 11:
        raise AssertionError(f"GT3 should suppress 11 event queries, got {gt3_count}.")
    if not float(event_loss.item()) > 0.0:
        raise AssertionError("GT3 event containment loss should be positive.")


def test_gt5_mask_isolates_clean_short_carriers() -> None:
    loss = _criterion()
    pred_points, _pred_logits, _pred_valid_logits = _preds()
    _gt_points, gt_valid = _lane_batch(5, short_last=True)
    masks = loss._q24_event_allowed_masks(pred_points, [gt_valid], torch.tensor([5]))
    mask = masks[0]
    short_gt = gt_valid.shape[0] - 1
    if not bool(mask[12, short_gt]) or not bool(mask[15, short_gt]) or not bool(mask[20, short_gt]):
        raise AssertionError("Clean GT5 event queries should be allowed to match the short GT5 lane.")
    if bool(mask[13, short_gt]) or bool(mask[21, short_gt]) or bool(mask[22, short_gt]) or bool(mask[23, short_gt]):
        raise AssertionError("High-risk event queries must not match true GT5 short lanes.")
    if bool(mask[14, short_gt]) or bool(mask[17, short_gt]) or bool(mask[18, short_gt]) or bool(mask[19, short_gt]):
        raise AssertionError("GT4 event queries must not match GT5 lanes.")


def test_gt5_risk_queries_are_counted() -> None:
    loss = _criterion()
    pred_points, pred_logits, pred_valid_logits = _preds()
    gt_points, gt_valid = _lane_batch(5, short_last=True)
    gt_lanes = torch.tensor([5])
    masks = loss._q24_event_allowed_masks(pred_points, [gt_valid], gt_lanes)
    outputs = loss.q24_event_containment_loss(
        pred_points,
        pred_logits,
        pred_valid_logits,
        [(torch.tensor([12]), torch.tensor([4]))],
        [gt_points],
        [gt_valid],
        gt_lanes,
        masks,
    )
    _event_loss, _exist, _valid, _gt3, _gt4, gt5_risk, protected, clean = outputs
    if int(gt5_risk.item()) != 4:
        raise AssertionError(f"GT5 should suppress 4 high-risk queries, got {gt5_risk}.")
    if int(protected.item()) != 0:
        raise AssertionError(f"Risk protection is disabled in this check, got protected={protected}.")
    if int(clean.item()) != 1:
        raise AssertionError(f"One clean GT5 query should be counted as allowed, got {clean}.")


def test_forward_loss_items_length_stable() -> None:
    loss = _criterion()
    pred_points, pred_logits, pred_valid_logits = _preds()
    gt_points, gt_valid = _lane_batch(5, short_last=True)
    total, items = loss(
        {
            "pred_points": pred_points,
            "pred_logits": pred_logits,
            "pred_valid_logits": pred_valid_logits,
        },
        {
            "lanes": [gt_points],
            "lane_valid": [gt_valid],
            "num_lanes": torch.tensor([5], dtype=torch.long),
        },
    )
    if int(items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(f"loss item length changed: got={items.numel()}, expected={len(GCSLoss.loss_names)}.")
    if not torch.isfinite(total) or not torch.isfinite(items).all():
        raise AssertionError("Q24 event containment forward produced non-finite values.")


def main() -> None:
    test_gt3_event_queries_are_suppressed()
    test_gt5_mask_isolates_clean_short_carriers()
    test_gt5_risk_queries_are_counted()
    test_forward_loss_items_length_stable()
    print(json.dumps({"status": "ok", "tests": 4}, indent=2))


if __name__ == "__main__":
    main()
