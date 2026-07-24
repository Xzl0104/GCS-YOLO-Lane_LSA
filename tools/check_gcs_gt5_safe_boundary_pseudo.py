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
        "gcs_boundary_pseudo_neg": 0.02,
        "gcs_boundary_pseudo_visible_thr": 10,
        "gcs_boundary_pseudo_dist_thr": 80.0,
        "gcs_boundary_pseudo_valid_thr": 0.5,
        "gcs_boundary_pseudo_min_valid": 3,
        "gcs_boundary_pseudo_gt_count": 5,
        "gcs_boundary_pseudo_score_thr": 0.0,
        "gcs_boundary_pseudo_envelope_margin_px": 30.0,
        "gcs_boundary_pseudo_envelope_ratio_thr": 0.75,
        "gcs_boundary_pseudo_gt5_safe": True,
        "gcs_boundary_pseudo_protect_short_visible_thr": 4,
        "gcs_boundary_pseudo_protect_dist_px": 40.0,
        "gcs_boundary_pseudo_protect_min_overlap": 3,
    }
    args.update(overrides)
    return GCSLoss(args)


def _gt5_batch() -> tuple[torch.Tensor, torch.Tensor]:
    k = 6
    y = torch.linspace(710.0 / 720.0, 660.0 / 720.0, k)
    xs = torch.tensor([0.15, 0.30, 0.45, 0.60, 0.75]).view(5, 1).expand(5, k)
    gt_points = torch.stack((xs, y.view(1, k).expand(5, k)), dim=-1).float()
    gt_valid = torch.ones(5, k)
    gt_valid[0, 4:] = 0.0
    return gt_points, gt_valid


def _preds(candidate_x: float, candidate_logit: float = -2.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = 6
    y = torch.linspace(710.0 / 720.0, 660.0 / 720.0, k)
    pred_points = torch.zeros(1, 2, k, 2)
    pred_points[0, 0, :, 0] = 0.15
    pred_points[0, 0, :, 1] = y
    pred_points[0, 1, :, 0] = float(candidate_x)
    pred_points[0, 1, :, 1] = y

    pred_logits = torch.tensor([[3.0, float(candidate_logit)]])
    pred_valid_logits = torch.full((1, 2, k), -4.0)
    pred_valid_logits[0, 1, :4] = 4.0
    return pred_points, pred_logits, pred_valid_logits


def test_clear_boundary_pseudo_is_selected() -> None:
    loss = _criterion()
    gt_points, gt_valid = _gt5_batch()
    pred_points, pred_logits, pred_valid_logits = _preds(candidate_x=0.0)
    outputs = loss.boundary_pseudo_neg_loss(
        pred_points,
        pred_logits,
        pred_valid_logits,
        [gt_points],
        [gt_valid],
        [(torch.tensor([0]), torch.tensor([0]))],
        torch.tensor([5]),
    )
    bneg_loss, selected, _score_mean, candidates, protected = outputs
    if not (float(bneg_loss) > 0.0 and int(selected.item()) == 1):
        raise AssertionError(f"clear boundary pseudo should be selected, got loss={bneg_loss}, selected={selected}.")
    if int(candidates.item()) != 1 or int(protected.item()) != 0:
        raise AssertionError(f"unexpected clear pseudo counts: candidates={candidates}, protected={protected}.")


def test_near_true_gt5_short_candidate_is_protected() -> None:
    loss = _criterion()
    gt_points, gt_valid = _gt5_batch()
    pred_points, pred_logits, pred_valid_logits = _preds(candidate_x=0.11)
    outputs = loss.boundary_pseudo_neg_loss(
        pred_points,
        pred_logits,
        pred_valid_logits,
        [gt_points],
        [gt_valid],
        [(torch.tensor([0]), torch.tensor([0]))],
        torch.tensor([5]),
    )
    bneg_loss, selected, _score_mean, candidates, protected = outputs
    if float(bneg_loss) != 0.0 or int(selected.item()) != 0:
        raise AssertionError(f"protected true-short candidate should not be selected, got loss={bneg_loss}, selected={selected}.")
    if int(candidates.item()) != 1 or int(protected.item()) != 1:
        raise AssertionError(f"unexpected protected counts: candidates={candidates}, protected={protected}.")


def test_safe_mode_requires_envelope_gate() -> None:
    try:
        _criterion(gcs_boundary_pseudo_envelope_margin_px=-1.0)
    except ValueError as exc:
        if "gcs_boundary_pseudo_gt5_safe requires" not in str(exc):
            raise
    else:
        raise AssertionError("gt5-safe boundary pseudo mode must require the envelope gate.")


def test_forward_loss_items_length_stable() -> None:
    loss = _criterion()
    gt_points, gt_valid = _gt5_batch()
    pred_points, pred_logits, pred_valid_logits = _preds(candidate_x=0.0)
    batch = {
        "lanes": [gt_points],
        "lane_valid": [gt_valid],
        "num_lanes": torch.tensor([5], dtype=torch.long),
    }
    total, items = loss(
        {
            "pred_points": pred_points,
            "pred_logits": pred_logits,
            "pred_valid_logits": pred_valid_logits,
        },
        batch,
    )
    if int(items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(f"loss item length changed: got={items.numel()}, expected={len(GCSLoss.loss_names)}.")
    if not torch.isfinite(total) or not torch.isfinite(items).all():
        raise AssertionError("GT5-safe boundary pseudo forward produced non-finite values.")


def main() -> None:
    test_clear_boundary_pseudo_is_selected()
    test_near_true_gt5_short_candidate_is_protected()
    test_safe_mode_requires_envelope_gate()
    test_forward_loss_items_length_stable()
    print(json.dumps({"status": "ok", "tests": 4}, indent=2))


if __name__ == "__main__":
    main()
