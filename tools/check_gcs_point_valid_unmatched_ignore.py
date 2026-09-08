from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402


def _case() -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
    gt_lane = torch.tensor(
        [
            [0.50, 0.90],
            [0.50, 0.80],
            [0.50, 0.70],
            [0.50, 0.60],
        ],
        dtype=torch.float32,
    )
    gt_valid = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    pred_points = torch.tensor(
        [
            [
                gt_lane.tolist(),
                [[0.52, 0.90], [0.52, 0.80], [0.52, 0.70], [0.52, 0.60]],
                [[0.90, 0.90], [0.90, 0.80], [0.90, 0.70], [0.90, 0.60]],
            ]
        ],
        dtype=torch.float32,
    )
    logits = torch.zeros((1, 3, 4), dtype=torch.float32)
    logits[0, 1, :3] = 10.0
    logits[0, 2, :3] = 10.0
    indices = [(torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long))]
    return pred_points, logits, [gt_lane.unsqueeze(0)], [gt_valid], indices


def main() -> None:
    pred_points, logits_base, gt_points, gt_valid, indices = _case()
    criterion = GCSLoss(
        {
            "gcs_imgsz": [100, 100],
            "gcs_point_valid_unmatched_ignore": True,
            "gcs_point_valid_unmatched_ignore_px": 30.0,
            "gcs_point_valid_unmatched_ignore_anchor_px": 30.0,
            "gcs_point_valid_unmatched_ignore_min_overlap": 3,
        }
    )
    weights = criterion._point_valid_weight_mask(pred_points, gt_points, gt_valid, indices)
    assert weights.shape == logits_base.shape
    assert torch.equal(weights[0, 0], torch.ones(4)), "matched query weights changed"
    assert torch.equal(weights[0, 1, :3], torch.zeros(3)), "GT-close unmatched visible anchors were not ignored"
    assert float(weights[0, 1, 3]) == 1.0, "invalid/non-GT-visible anchor should remain supervised"
    assert torch.equal(weights[0, 2], torch.ones(4)), "far-unmatched query should remain negative"

    logits_on = logits_base.clone().requires_grad_(True)
    loss_on = criterion.point_valid_loss(pred_points, logits_on, gt_points, gt_valid, indices)
    loss_on.backward()
    grad_on = logits_on.grad.detach()
    assert torch.equal(grad_on[0, 1, :3], torch.zeros(3)), "ignored close anchors still produced gradients"
    assert bool((grad_on[0, 2, :3] > 0).all()), "far-unmatched negative anchors lost gradients"
    assert float(grad_on[0, 1, 3]) > 0.0, "non-visible close-query anchor lost negative supervision"

    criterion_off = GCSLoss({"gcs_imgsz": [100, 100], "gcs_point_valid_unmatched_ignore": False})
    logits_off = logits_base.clone().requires_grad_(True)
    loss_off = criterion_off.point_valid_loss(pred_points, logits_off, gt_points, gt_valid, indices)
    loss_off.backward()
    assert bool((logits_off.grad[0, 1, :3] > 0).all()), "legacy path no longer supervises close unmatched anchors"
    assert float(loss_on.detach()) < float(loss_off.detach()), "ignore mask did not remove the close-unmatched penalty"

    preds = {
        "pred_points": pred_points.clone(),
        "pred_logits": torch.tensor([[10.0, -10.0, -10.0]], dtype=torch.float32),
        "pred_valid_logits": logits_base.clone(),
    }
    batch = {
        "img": torch.zeros((1, 3, 100, 100), dtype=torch.float32),
        "lanes": torch.stack(gt_points, dim=0),
        "lane_valid": torch.stack(gt_valid, dim=0),
    }
    total, items = criterion(preds, batch)
    assert torch.isfinite(total), "total loss is not finite"
    assert int(items.numel()) == len(GCSLoss.loss_names) == 5, "query loss item contract changed"
    print(
        {
            "status": "ok",
            "loss_on": round(float(loss_on.detach()), 6),
            "loss_off": round(float(loss_off.detach()), 6),
            "loss_items": int(items.numel()),
        }
    )


if __name__ == "__main__":
    main()
