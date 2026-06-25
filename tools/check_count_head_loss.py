"""Check explicit count-head logits and CE gradient flow."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.gcs_lane import GCSLaneHead
from ultralytics.utils.gcs_loss import GCSLoss


def main() -> None:
    """Assert count_mlp produces Bx3 logits and receives non-zero CE gradients."""
    torch.manual_seed(0)

    head = GCSLaneHead(
        c1=128,
        num_queries=18,
        num_points=56,
        num_decoder_layers=1,
        nhead=8,
        aux=False,
        point_mode="fixed_y",
        fixed_y_start=710 / 720,
        fixed_y_end=160 / 720,
    )

    features = [
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 128, 16, 16),
        torch.randn(2, 128, 8, 8),
        torch.randn(2, 128, 4, 4),
    ]
    output = head(features, orig_size=(544, 960))
    logits = output["pred_count_logits"]
    assert logits.shape == (2, 3), logits.shape
    assert torch.isfinite(logits).all()

    loss_fn = GCSLoss(model={"gcs_count_ce": 0.15}, image_size=(544, 960))
    batch = {"num_lanes": torch.tensor([3, 5])}
    loss, acc = loss_fn.count_ce_loss(output, output["pred_logits"], batch, gt_valid=[])
    assert loss > 0.0, loss
    (loss_fn.count_ce_gain * loss).backward()

    grad_sum = 0.0
    for name, param in head.named_parameters():
        if name.startswith("count_mlp") and param.grad is not None:
            grad_sum += float(param.grad.detach().abs().sum())

    assert grad_sum > 0.0, "count_mlp has no gradient"
    print("OK: count head CE has gradients.")
    print("pred_count_logits:", tuple(logits.shape))
    print("loss:", float(loss.detach()))
    print("count_ce_acc:", float(acc.detach()))
    print("count_mlp_grad_sum:", grad_sum)


if __name__ == "__main__":
    main()
