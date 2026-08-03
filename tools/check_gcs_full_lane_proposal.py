"""Smoke-test the default-off independent full-lane proposal set decoder."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402


DEFAULT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"
FULL_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-full-lane-proposal.yaml"
FULL_KEYS = {
    "pred_full_lane_points",
    "pred_full_lane_valid_logits",
    "pred_full_lane_exist_logits",
    "pred_full_lane_quality_logits",
    "pred_full_lane_start_logits",
    "pred_full_lane_end_logits",
    "pred_full_lane_score_logits",
}


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(probability / (1.0 - probability))


def _head_from_yaml(cfg: Path) -> GCSLaneHead:
    model = GCSLaneModel(str(cfg), nc=1, verbose=False)
    head = model.model[-1]
    if not isinstance(head, GCSLaneHead):
        raise AssertionError(f"{cfg} did not build a GCSLaneHead.")
    return head


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


@torch.inference_mode()
def check_default_contract() -> None:
    head = _head_from_yaml(DEFAULT_CFG).eval()
    output = head(_features(head, batch=1), orig_size=(544, 960))
    unexpected = sorted(key for key in output if key in FULL_KEYS)
    if unexpected:
        raise AssertionError(f"default env30 head unexpectedly emitted full-lane outputs: {unexpected}.")
    if tuple(output["pred_points"].shape) != (1, 12, 56, 2):
        raise AssertionError(f"default pred_points shape changed: {tuple(output['pred_points'].shape)}.")
    if tuple(output["pred_logits"].shape) != (1, 12):
        raise AssertionError(f"default pred_logits shape changed: {tuple(output['pred_logits'].shape)}.")
    if tuple(output["pred_valid_logits"].shape) != (1, 12, 56):
        raise AssertionError(f"default pred_valid_logits shape changed: {tuple(output['pred_valid_logits'].shape)}.")


@torch.inference_mode()
def check_full_forward() -> None:
    head = _head_from_yaml(FULL_CFG).eval()
    if not bool(getattr(head, "full_lane_proposal_head", False)):
        raise AssertionError("full-lane YAML did not enable full_lane_proposal_head.")
    if int(getattr(head, "full_lane_proposal_count", 0)) != 8:
        raise AssertionError("full-lane YAML must create exactly 8 independent proposals.")
    output = head(_features(head, batch=2), orig_size=(544, 960))
    missing = FULL_KEYS.difference(output)
    if missing:
        raise AssertionError(f"full-lane output is missing keys: {sorted(missing)}.")
    expected = {
        "pred_full_lane_points": (2, 8, 56, 2),
        "pred_full_lane_valid_logits": (2, 8, 56),
        "pred_full_lane_exist_logits": (2, 8),
        "pred_full_lane_quality_logits": (2, 8),
        "pred_full_lane_start_logits": (2, 8, 56),
        "pred_full_lane_end_logits": (2, 8, 56),
        "pred_full_lane_score_logits": (2, 8),
    }
    for key, shape in expected.items():
        if tuple(output[key].shape) != shape:
            raise AssertionError(f"{key} shape mismatch: {tuple(output[key].shape)} != {shape}.")
    fixed_y = _fixed_y(batch=2, lanes=8)
    actual_y = output["pred_full_lane_points"][..., 1]
    if float((actual_y - fixed_y).abs().max()) > 1.0e-6:
        raise AssertionError("full-lane fixed-y anchors are not exactly 710,700,...,160.")


def check_full_decoder_backward() -> None:
    head = _head_from_yaml(FULL_CFG).train()
    output = head(_features(head, batch=1), orig_size=(544, 960))
    loss = sum(output[key].float().mean() for key in FULL_KEYS)
    loss.backward()
    full_grads = [
        (name, parameter.grad)
        for name, parameter in head.named_parameters()
        if name.startswith("full_lane_")
    ]
    if not full_grads:
        raise AssertionError("full-lane decoder parameters were not found.")
    missing = [name for name, grad in full_grads if grad is None]
    if missing:
        raise AssertionError(f"full-lane decoder parameters without gradients: {missing[:8]}.")
    zero = [name for name, grad in full_grads if float(grad.abs().sum()) <= 0.0]
    if zero:
        raise AssertionError(f"full-lane decoder parameters received zero gradients: {zero[:8]}.")


def _synthetic_full_outputs() -> tuple[torch.Tensor, ...]:
    batch, proposals, points = 1, 3, 56
    y = _fixed_y(batch=1, lanes=proposals, points=points)
    x = torch.zeros(batch, proposals, points)
    x[:, 0] = 0.205
    x[:, 1] = 0.795
    x[:, 2] = 0.50
    full_points = torch.stack((x, y), dim=-1).requires_grad_()
    full_valid = torch.full((batch, proposals, points), _logit(0.9)).requires_grad_()
    full_exist = torch.tensor([[4.0, 4.0, -2.0]], requires_grad=True)
    full_quality = torch.tensor([[2.0, 2.0, -1.0]], requires_grad=True)
    start = torch.full((batch, proposals, points), -4.0)
    end = torch.full((batch, proposals, points), -4.0)
    start[:, :, 0] = 4.0
    end[:, :, -1] = 4.0
    start.requires_grad_()
    end.requires_grad_()
    return full_points, full_valid, full_exist, full_quality, start, end


def _synthetic_gt() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    points = 56
    y = _fixed_y(batch=1, lanes=2, points=points)[0]
    x = torch.stack((torch.full((points,), 0.20), torch.full((points,), 0.80)), dim=0)
    gt_points = [torch.stack((x, y), dim=-1)]
    gt_valid = [torch.ones(2, points)]
    return gt_points, gt_valid


def check_unified_hungarian_and_loss() -> None:
    criterion = GCSLoss({"gcs_imgsz": [544, 960], "gcs_full_lane_proposal": 1.0})
    gt_points, gt_valid = _synthetic_gt()
    full_outputs = _synthetic_full_outputs()
    batch, queries, points = 1, 2, 56
    y = _fixed_y(batch=1, lanes=queries, points=points)
    base_x = torch.full((batch, queries, points), 0.5)
    base_points = torch.stack((base_x, y), dim=-1)
    base_logits = torch.zeros(batch, queries)
    base_valid = torch.full((batch, queries, points), _logit(0.5))

    unified = criterion._full_lane_unified_indices(
        base_points,
        base_logits,
        base_valid,
        full_outputs,
        gt_points,
        gt_valid,
    )
    src_idx, tgt_idx = unified[0]
    if int(torch.unique(src_idx).numel()) != int(src_idx.numel()):
        raise AssertionError("unified Hungarian matching reused a proposal source.")
    if int(torch.unique(tgt_idx).numel()) != int(tgt_idx.numel()):
        raise AssertionError("unified Hungarian matching assigned one GT lane more than once.")
    full_indices = criterion._split_full_lane_indices(unified, base_query_count=queries)[1]
    if int(full_indices[0][0].numel()) < 1:
        raise AssertionError("synthetic full proposals did not enter the unified matching set.")

    losses = criterion.full_lane_proposal_loss(full_outputs, gt_points, gt_valid, full_indices)
    total = losses[0]
    if not torch.isfinite(total):
        raise AssertionError("full-lane proposal loss is non-finite.")
    total.backward()
    for name, value in (
        ("full_points", full_outputs[0].grad),
        ("full_valid", full_outputs[1].grad),
        ("full_exist", full_outputs[2].grad),
        ("full_quality", full_outputs[3].grad),
    ):
        if value is None or not torch.isfinite(value).all() or float(value.abs().sum()) <= 0.0:
            raise AssertionError(f"{name} did not receive a finite non-zero full-lane loss gradient.")
    if float(losses[7]) <= 0.0:
        raise AssertionError("unmatched full proposal negative supervision was not exercised.")


def check_aux_assignment_and_warmup_loss() -> None:
    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_full_lane_proposal": 1.0,
            "gcs_full_lane_aux_assignment": True,
            "gcs_full_lane_unified_matching": False,
            "gcs_full_lane_unmatched_weight": 0.0,
            "gcs_full_lane_unmatched_valid_weight": 0.0,
        }
    )
    gt_points, gt_valid = _synthetic_gt()
    full_outputs = _synthetic_full_outputs()
    aux_indices = criterion._full_lane_aux_indices(full_outputs, gt_points, gt_valid)
    src_idx, tgt_idx = aux_indices[0]
    if int(src_idx.numel()) != 2 or int(tgt_idx.numel()) != 2:
        raise AssertionError(
            "proposal-only full-lane aux assignment must match every synthetic GT lane, "
            f"got src={src_idx.tolist()} tgt={tgt_idx.tolist()}."
        )

    losses = criterion.full_lane_proposal_loss(
        full_outputs,
        gt_points,
        gt_valid,
        aux_indices,
        unmatched_weight=0.0,
    )
    total, point_loss, valid_loss, interval_loss = losses[:4]
    if not torch.isfinite(total):
        raise AssertionError("aux full-lane warmup loss is non-finite.")
    if float(losses[6]) != 2.0:
        raise AssertionError(f"aux full-lane match count should be 2, got {float(losses[6])}.")
    if float(losses[7]) != 1.0:
        raise AssertionError(f"aux full-lane unmatched count should be 1, got {float(losses[7])}.")
    point_value = float(point_loss.detach())
    valid_value = float(valid_loss.detach())
    interval_value = float(interval_loss.detach())
    if point_value <= 0.0 or valid_value <= 0.0 or interval_value <= 0.0:
        raise AssertionError(
            "aux full-lane warmup must exercise positive point/valid/interval losses, "
            f"got point={point_value}, valid={valid_value}, interval={interval_value}."
        )
    total.backward()
    for name, value in (
        ("full_points", full_outputs[0].grad),
        ("full_valid", full_outputs[1].grad),
        ("full_exist", full_outputs[2].grad),
        ("full_quality", full_outputs[3].grad),
    ):
        if value is None or not torch.isfinite(value).all() or float(value.abs().sum()) <= 0.0:
            raise AssertionError(f"{name} did not receive a finite non-zero aux warmup gradient.")


def check_hard_focus_target_weights() -> None:
    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_full_lane_proposal": 1.0,
            "gcs_full_lane_aux_assignment": True,
            "gcs_full_lane_unified_matching": False,
            "gcs_full_lane_hard_focus": True,
            "gcs_full_lane_base_hit_weight": 0.05,
            "gcs_full_lane_base_miss_weight": 4.0,
            "gcs_full_lane_gt5_weight": 3.0,
            "gcs_full_lane_short_visible_thr": 10,
            "gcs_full_lane_short_visible_weight": 4.0,
            "gcs_full_lane_focus_base_valid_thr": 0.6,
            "gcs_full_lane_focus_base_min_coverage": 1.0,
        }
    )
    batch, queries, lanes, points = 1, 2, 5, 56
    y = _fixed_y(batch=batch, lanes=lanes, points=points)
    gt_x = torch.stack(
        (
            torch.full((points,), 0.20),
            torch.full((points,), 0.35),
            torch.full((points,), 0.50),
            torch.full((points,), 0.65),
            torch.full((points,), 0.80),
        )
    ).unsqueeze(0)
    gt_points = [torch.stack((gt_x[0], y[0]), dim=-1)]
    gt_valid_tensor = torch.ones(lanes, points)
    gt_valid_tensor[4, 10:] = 0.0
    gt_valid = [gt_valid_tensor]

    base_y = _fixed_y(batch=batch, lanes=queries, points=points)
    base_x = torch.stack((torch.full((points,), 0.20), torch.full((points,), 0.50))).unsqueeze(0)
    base_points = torch.stack((base_x, base_y), dim=-1)
    base_valid = torch.full((batch, queries, points), _logit(0.95))
    weights = criterion._full_lane_focus_target_weights(base_points, base_valid, gt_points, gt_valid)
    if weights is None:
        raise AssertionError("hard-focus target weights were not produced.")
    lane_weights = weights[0]
    if tuple(lane_weights.shape) != (5,):
        raise AssertionError(f"hard-focus weight shape mismatch: {tuple(lane_weights.shape)}.")
    expected_base_hit_gt5 = 0.05 * 3.0
    expected_base_miss_short_gt5 = 4.0 * 3.0 * 4.0
    if not torch.isclose(lane_weights[0], torch.tensor(expected_base_hit_gt5), atol=1e-6):
        raise AssertionError(f"base-hit GT5 lane weight changed: {float(lane_weights[0])}.")
    if not torch.isclose(lane_weights[4], torch.tensor(expected_base_miss_short_gt5), atol=1e-6):
        raise AssertionError(f"base-miss short GT5 lane weight changed: {float(lane_weights[4])}.")
    if not float(lane_weights[4]) > float(lane_weights[0]) * 100.0:
        raise AssertionError("hard-focus did not strongly prioritize the base-miss short GT5 lane.")


@torch.inference_mode()
def check_decode_contract() -> None:
    batch, queries, proposals, points = 1, 2, 3, 56
    y_base = _fixed_y(batch=1, lanes=queries, points=points)[0]
    y_full = _fixed_y(batch=1, lanes=proposals, points=points)[0]
    base_x = torch.stack((torch.full((points,), 0.2), torch.full((points,), 0.8)))
    full_x = torch.stack(
        (
            torch.full((points,), 0.2),
            torch.full((points,), 0.8),
            torch.full((points,), 0.5),
        )
    )
    base_points = torch.stack((base_x, y_base), dim=-1)
    full_points = torch.stack((full_x, y_full), dim=-1)
    valid = torch.full((queries, points), _logit(0.9))
    full_valid = torch.full((proposals, points), _logit(0.9))
    start = torch.full((proposals, points), -4.0)
    end = torch.full((proposals, points), -4.0)
    start[:, 0] = 4.0
    end[:, -1] = 4.0

    lanes = decode_gcs_predictions(
        base_points,
        torch.full((queries,), 2.0),
        pred_valid_logits=valid,
        full_lane_decode=True,
        pred_full_lane_points=full_points,
        pred_full_lane_valid_logits=full_valid,
        pred_full_lane_exist_logits=torch.tensor([2.0, 2.0, -2.0]),
        pred_full_lane_quality_logits=torch.tensor([2.0, 2.0, -1.0]),
        pred_full_lane_start_logits=start,
        pred_full_lane_end_logits=end,
        image_shape=(544, 960),
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=2,
        max_det=5,
        nms_dist_px=0.0,
    )
    if not lanes:
        raise AssertionError("full-lane decode returned no valid proposals.")
    if not any(item.get("source") == "full_lane_proposal" for item in lanes):
        raise AssertionError("full-lane decode did not return an independent proposal.")
    try:
        decode_gcs_predictions(
            base_points,
            torch.zeros(queries),
            pred_valid_logits=valid,
            full_lane_decode=True,
            count_aware_topk=True,
            pred_full_lane_points=full_points,
            pred_full_lane_valid_logits=full_valid,
            pred_full_lane_exist_logits=torch.zeros(proposals),
            pred_full_lane_quality_logits=torch.zeros(proposals),
            pred_full_lane_start_logits=start,
            pred_full_lane_end_logits=end,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("full-lane decode must reject count-aware top-k.")


def main() -> None:
    check_default_contract()
    check_full_forward()
    check_full_decoder_backward()
    check_unified_hungarian_and_loss()
    check_aux_assignment_and_warmup_loss()
    check_hard_focus_target_weights()
    check_decode_contract()
    print("GCS full-lane proposal checks passed.")


if __name__ == "__main__":
    main()
