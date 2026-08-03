"""Smoke checks for the default-off dense instance/keypoint evidence path."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel  # noqa: E402
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402


DEFAULT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"
DENSE_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-dense-instance-proposal.yaml"


def _head_from_yaml(cfg: Path) -> GCSLaneHead:
    model = GCSLaneModel(str(cfg), nc=1, verbose=False)
    head = model.model[-1]
    if not isinstance(head, GCSLaneHead):
        raise AssertionError(f"{cfg} did not build a GCSLaneHead.")
    return head


def _model_from_yaml(cfg: Path) -> GCSLaneModel:
    return GCSLaneModel(str(cfg), nc=1, verbose=False)


def _features(head: GCSLaneHead, batch: int) -> list[torch.Tensor]:
    channels = int(head.c1)
    return [
        torch.randn(batch, channels, 32, 32),
        torch.randn(batch, channels, 16, 16),
        torch.randn(batch, channels, 8, 8),
        torch.randn(batch, channels, 4, 4),
    ]


def _fixed_y(batch: int, lanes: int, points: int = 56) -> torch.Tensor:
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, points)
    return y.view(1, 1, points).expand(batch, lanes, points)


def _synthetic_targets() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    points = 56
    y = _fixed_y(1, 2, points)[0]
    x = torch.stack((torch.full((points,), 0.25), torch.full((points,), 0.75)))
    return [torch.stack((x, y), dim=-1)], [torch.ones(2, points)]


def _single_lane_batch_targets() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    points = 56
    y = _fixed_y(1, 1, points)[0, 0]
    targets = []
    valid = []
    for x_value in (0.25, 0.75):
        x = torch.full((points,), x_value)
        targets.append(torch.stack((x, y), dim=-1).unsqueeze(0))
        valid.append(torch.ones(1, points))
    return targets, valid


@torch.inference_mode()
def check_default_contract() -> None:
    head = _head_from_yaml(DEFAULT_CFG).eval()
    output = head(_features(head, 1), orig_size=(544, 960))
    dense_keys = sorted(key for key in output if key.startswith("pred_dense_"))
    if dense_keys:
        raise AssertionError(f"default env30 head unexpectedly emitted dense outputs: {dense_keys}.")
    if tuple(output["pred_points"].shape) != (1, 12, 56, 2):
        raise AssertionError(f"default pred_points shape changed: {tuple(output['pred_points'].shape)}.")


@torch.inference_mode()
def check_dense_forward() -> None:
    head = _head_from_yaml(DENSE_CFG).eval()
    if not bool(getattr(head, "dense_instance_head", False)):
        raise AssertionError("dense YAML did not enable dense_instance_head.")
    output = head(_features(head, 2), orig_size=(544, 960))
    expected = {
        "pred_dense_centerline_logits": (2, 1, 32, 32),
        "pred_dense_endpoint_logits": (2, 2, 32, 32),
        "pred_dense_instance_embed": (2, 8, 32, 32),
    }
    for key, shape in expected.items():
        if tuple(output[key].shape) != shape:
            raise AssertionError(f"{key} shape mismatch: {tuple(output[key].shape)} != {shape}.")
    fixed_y = _fixed_y(2, 12)
    if float((output["pred_points"][..., 1] - fixed_y).abs().max()) > 1.0e-6:
        raise AssertionError("dense model changed the fixed-y 710..160 anchor contract.")


def check_dense_head_backward() -> None:
    head = _head_from_yaml(DENSE_CFG).train()
    output = head(_features(head, 1), orig_size=(544, 960))
    loss = (
        output["pred_dense_centerline_logits"].float().mean()
        + output["pred_dense_endpoint_logits"].float().mean()
        + output["pred_dense_instance_embed"].float().square().mean()
    )
    loss.backward()
    dense_grads = [
        (name, parameter.grad)
        for name, parameter in head.named_parameters()
        if name.startswith("dense_instance_")
    ]
    if not dense_grads:
        raise AssertionError("dense-instance parameters were not found.")
    missing = [name for name, grad in dense_grads if grad is None]
    if missing:
        raise AssertionError(f"dense-instance parameters without gradients: {missing[:8]}.")
    zero = [name for name, grad in dense_grads if float(grad.abs().sum()) <= 0.0]
    if zero:
        raise AssertionError(f"dense-instance parameters received zero gradients: {zero[:8]}.")


def check_dense_loss_backward() -> None:
    head = _head_from_yaml(DENSE_CFG).train()
    features = _features(head, 1)
    preds = head(features, orig_size=(544, 960))
    gt_points, gt_valid = _synthetic_targets()
    batch = {
        "img": torch.zeros(1, 3, 544, 960),
        "lanes": gt_points,
        "lane_valid": gt_valid,
    }
    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_dense_instance": 1.0,
            "gcs_dense_centerline_weight": 1.0,
            "gcs_dense_endpoint_weight": 1.0,
            "gcs_dense_embed_pull_weight": 0.25,
            "gcs_dense_embed_push_weight": 0.25,
            "gcs_mask": 0.0,
            "gcs_edge": 0.0,
        }
    )
    total, items = criterion(preds, batch)
    if not torch.isfinite(total):
        raise AssertionError("GCS dense total loss is non-finite.")
    if int(items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(
            f"GCS loss item count mismatch: {int(items.numel())} != {len(GCSLoss.loss_names)}."
        )
    total.backward()
    dense_names = {
        "dense_instance_stem",
        "dense_instance_centerline",
        "dense_instance_endpoint",
        "dense_instance_embed",
    }
    dense_params = [
        (name, parameter.grad)
        for name, parameter in head.named_parameters()
        if any(name.startswith(prefix) for prefix in dense_names)
    ]
    if not dense_params or any(grad is None for _, grad in dense_params):
        raise AssertionError("GCS dense loss did not backpropagate to every dense head module.")
    if any(not torch.isfinite(grad).all() or float(grad.abs().sum()) <= 0.0 for _, grad in dense_params):
        raise AssertionError("GCS dense loss produced a non-finite or zero dense-head gradient.")


def check_embedding_push_is_image_local() -> None:
    head = _head_from_yaml(DENSE_CFG).eval()
    preds = head(_features(head, 2), orig_size=(544, 960))
    gt_points, gt_valid = _single_lane_batch_targets()
    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_dense_instance": 1.0,
            "gcs_dense_centerline_weight": 1.0,
            "gcs_dense_endpoint_weight": 1.0,
            "gcs_dense_embed_pull_weight": 0.25,
            "gcs_dense_embed_push_weight": 0.25,
            "gcs_dense_embed_margin": 0.5,
            "gcs_mask": 0.0,
            "gcs_edge": 0.0,
        }
    )
    dense_terms = criterion.dense_instance_loss(preds, gt_points, gt_valid)
    embed_push_loss = dense_terms[4]
    if float(embed_push_loss.detach()) != 0.0:
        raise AssertionError(
            "Dense embedding push loss must be zero when each image contains only one lane; "
            "cross-image lane means must not be pushed apart."
        )


def check_dense_freeze_contract() -> None:
    model = _model_from_yaml(DENSE_CFG)
    trainer = GCSLaneTrainer.__new__(GCSLaneTrainer)
    trainer.args = SimpleNamespace(
        gcs_dense_freeze_base=True,
        gcs_dense_instance=1.0,
        gcs_short_candidate_freeze_base=False,
        gcs_short_segment_freeze_base=False,
        gcs_full_lane_freeze_base=False,
    )
    trainer.model = model
    trainer._apply_custom_freeze()

    for name, parameter in model.named_parameters():
        expected = "dense_instance_" in name
        if bool(parameter.requires_grad) != expected:
            raise AssertionError(
                f"dense freeze requires_grad mismatch for {name}: "
                f"{parameter.requires_grad} != {expected}."
            )

    trainer.freeze_layer_names = []
    trainer._model_train()
    for name, module in model.named_modules():
        expected_train = "dense_instance_" in name
        if bool(module.training) != expected_train:
            raise AssertionError(
                f"dense freeze module mode mismatch for {name or '<head>'}: "
                f"{module.training} != {expected_train}."
            )
        if isinstance(module, nn.BatchNorm2d) and not expected_train and module.training:
            raise AssertionError(f"Frozen base BatchNorm unexpectedly remained in train mode: {name}.")


def main() -> None:
    check_default_contract()
    check_dense_forward()
    check_dense_head_backward()
    check_dense_loss_backward()
    check_embedding_push_is_image_local()
    check_dense_freeze_contract()
    print("GCS dense instance checks passed.")


if __name__ == "__main__":
    main()
