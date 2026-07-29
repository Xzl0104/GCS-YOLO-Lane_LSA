"""Smoke-test the default-off GCS local short-segment proposal head and loss."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402


DEFAULT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"
SEGMENT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v4.yaml"


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _head_from_yaml(cfg: Path) -> GCSLaneHead:
    model = GCSLaneModel(str(cfg), nc=1, verbose=False)
    head = model.model[-1]
    if not isinstance(head, GCSLaneHead):
        raise AssertionError(f"{cfg} did not build a GCSLaneHead.")
    return head.eval()


def _head_features(head: GCSLaneHead, batch: int = 1) -> list[torch.Tensor]:
    c = int(head.c1)
    return [
        torch.randn(batch, c, 32, 32),
        torch.randn(batch, c, 16, 16),
        torch.randn(batch, c, 8, 8),
        torch.randn(batch, c, 4, 4),
    ]


@torch.inference_mode()
def check_yaml_forward() -> None:
    default_head = _head_from_yaml(DEFAULT_CFG)
    default_out = default_head(_head_features(default_head), orig_size=(544, 960))
    unexpected = [k for k in default_out if k.startswith("pred_short_segment")]
    if unexpected:
        raise AssertionError(f"default query YAML unexpectedly emitted short-segment outputs: {unexpected}.")

    segment_head = _head_from_yaml(SEGMENT_CFG)
    segment_out = segment_head(_head_features(segment_head), orig_size=(544, 960))
    required = {
        "pred_short_segment_points",
        "pred_short_segment_logits",
        "pred_short_segment_x",
        "pred_short_segment_starts",
        "pred_short_segment_ends",
        "pred_short_segment_window_mask",
    }
    missing = required.difference(segment_out)
    if missing:
        raise AssertionError(f"segment YAML missing output keys: {sorted(missing)}.")

    points = segment_out["pred_short_segment_points"]
    logits = segment_out["pred_short_segment_logits"]
    segment_x = segment_out["pred_short_segment_x"]
    starts = segment_out["pred_short_segment_starts"]
    ends = segment_out["pred_short_segment_ends"]
    mask = segment_out["pred_short_segment_window_mask"]
    expected_segments = sum(56 - length + 1 for length in range(3, 11))
    if tuple(points.shape) != (1, 12, expected_segments, 56, 2):
        raise AssertionError(f"segment point shape mismatch: {tuple(points.shape)}.")
    if tuple(logits.shape) != (1, 12, expected_segments):
        raise AssertionError(f"segment logit shape mismatch: {tuple(logits.shape)}.")
    if tuple(segment_x.shape) != (1, 12, expected_segments, 2):
        raise AssertionError(f"segment endpoint-x shape mismatch: {tuple(segment_x.shape)}.")
    if tuple(mask.shape) != (expected_segments, 56):
        raise AssertionError(f"segment window mask shape mismatch: {tuple(mask.shape)}.")
    lengths = ends - starts + 1
    if set(int(x) for x in lengths.tolist()) != set(range(3, 11)):
        raise AssertionError(f"segment window lengths mismatch: {sorted(set(lengths.tolist()))}.")
    if not torch.equal(mask.sum(dim=1).cpu(), lengths.cpu()):
        raise AssertionError("segment window masks do not match start/end lengths.")


def _make_short_segment_batch() -> dict:
    k = 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    xs = torch.tensor([0.4, 0.3, 0.6, 0.75], dtype=torch.float32).view(4, 1).expand(4, k)
    lanes = torch.stack((xs, y.view(1, k).expand(4, k)), dim=-1).float()
    valid = torch.zeros(4, k, dtype=torch.float32)
    valid[0, :5] = 1.0
    valid[1:, :16] = 1.0
    return {
        "lanes": [lanes],
        "lane_valid": [valid],
        "num_lanes": torch.tensor([4], dtype=torch.long),
    }


def _make_short_segment_preds() -> dict[str, torch.Tensor]:
    b, q, s, k = 1, 12, 2, 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k).view(1, 1, k).expand(b, q, k)
    x = torch.full((b, q, k), 0.5)
    pred_points = torch.stack((x, y), dim=-1).contiguous()
    segment_points = pred_points.unsqueeze(2).expand(b, q, s, k, 2).clone()
    segment_points[0, 0, 0, :5, 0] = 0.4
    segment_logits = torch.full((b, q, s), _logit(0.02), dtype=torch.float32)
    segment_logits[0, 0, 0] = _logit(0.5)
    window_mask = torch.zeros(s, k, dtype=torch.bool)
    window_mask[0, :5] = True
    window_mask[1, 10:15] = True
    return {
        "pred_points": pred_points,
        "pred_logits": torch.full((b, q), _logit(0.2), dtype=torch.float32),
        "pred_valid_logits": torch.full((b, q, k), _logit(0.7), dtype=torch.float32),
        "pred_short_segment_points": segment_points,
        "pred_short_segment_logits": segment_logits,
        "pred_short_segment_window_mask": window_mask,
    }


def _segment_criterion() -> GCSLoss:
    return GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_segment": 0.1,
            "gcs_short_segment_topk": 2,
            "gcs_short_segment_visible_thr": 10,
            "gcs_short_segment_min_visible": 3,
            "gcs_short_segment_min_overlap": 3,
            "gcs_short_segment_neg_score_thr": 0.6,
        }
    )


def check_loss() -> None:
    criterion = _segment_criterion()
    _, items = criterion(_make_short_segment_preds(), _make_short_segment_batch())
    if int(items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(f"GCSLoss item length mismatch: {items.numel()} vs {len(GCSLoss.loss_names)}.")
    if not torch.isfinite(items).all():
        raise AssertionError("GCSLoss produced non-finite short-segment items.")
    idx = {name: i for i, name in enumerate(GCSLoss.loss_names)}
    if float(items[idx["short_segment_pos_count"]]) <= 0.0:
        raise AssertionError("short-segment assignment did not produce a strong positive.")
    if float(items[idx["short_segment_loss"]]) <= 0.0:
        raise AssertionError("short-segment loss should be positive for the synthetic segment.")

    missing = _make_short_segment_preds()
    missing.pop("pred_short_segment_points")
    try:
        criterion(missing, _make_short_segment_batch())
    except KeyError:
        pass
    else:
        raise AssertionError("gcs_short_segment > 0 should require segment head outputs.")


def check_backward() -> None:
    preds = _make_short_segment_preds()
    segment_points = preds["pred_short_segment_points"].detach().clone()
    segment_points[0, 0, 0, :5, 0] = 0.405
    segment_points.requires_grad_(True)
    segment_logits = preds["pred_short_segment_logits"].detach().clone().requires_grad_(True)
    preds["pred_short_segment_points"] = segment_points
    preds["pred_short_segment_logits"] = segment_logits

    criterion = _segment_criterion()
    total, items = criterion(preds, _make_short_segment_batch())
    total.backward()
    if not torch.isfinite(total.detach()) or not torch.isfinite(items).all():
        raise AssertionError("short-segment backward smoke produced non-finite losses.")
    if segment_logits.grad is None or not torch.isfinite(segment_logits.grad).all():
        raise AssertionError("short-segment score logits did not receive finite gradients.")
    if float(segment_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment score logits received zero gradient.")
    if segment_points.grad is None or not torch.isfinite(segment_points.grad).all():
        raise AssertionError("short-segment points did not receive finite pull gradients.")
    if float(segment_points.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment points received zero pull gradient.")


def check_empty_segment_batch_has_grad() -> None:
    preds = _make_short_segment_preds()
    segment_logits = preds["pred_short_segment_logits"].detach().clone().requires_grad_(True)
    segment_points = preds["pred_short_segment_points"].detach().clone().requires_grad_(True)
    preds["pred_short_segment_logits"] = segment_logits
    preds["pred_short_segment_points"] = segment_points
    batch = _make_short_segment_batch()
    batch["num_lanes"] = torch.tensor([3], dtype=torch.long)
    total, items = _segment_criterion()(preds, batch)
    if not total.requires_grad:
        raise AssertionError("empty short-segment batch should keep a zero-gradient path to segment outputs.")
    total.backward()
    if segment_logits.grad is None or not torch.isfinite(segment_logits.grad).all():
        raise AssertionError("empty short-segment batch did not produce finite zero score gradients.")
    if segment_points.grad is None or not torch.isfinite(segment_points.grad).all():
        raise AssertionError("empty short-segment batch did not produce finite zero point gradients.")
    if float(segment_logits.grad.abs().sum()) != 0.0 or float(segment_points.grad.abs().sum()) != 0.0:
        raise AssertionError("empty short-segment batch should not update segment outputs.")
    if not torch.isfinite(items).all():
        raise AssertionError("empty short-segment batch produced non-finite loss items.")


def check_freeze_contract() -> None:
    model = GCSLaneModel(str(SEGMENT_CFG), nc=1, verbose=False)
    trainer = object.__new__(GCSLaneTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(
        gcs_short_candidate_freeze_base=False,
        gcs_short_candidate=0.0,
        gcs_short_segment_freeze_base=True,
        gcs_short_segment=0.1,
    )
    trainer.freeze_layer_names = []
    trainer._apply_custom_freeze()

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise AssertionError("short-segment freeze contract left no trainable parameters.")
    bad = [name for name in trainable if "short_segment_" not in name]
    if bad:
        raise AssertionError(f"short-segment freeze contract left base parameters trainable: {bad[:5]}.")
    if not any("short_segment_x_mlp" in name for name in trainable):
        raise AssertionError("short-segment endpoint MLP is not trainable under freeze contract.")

    optimizer = trainer.build_optimizer(model, name="AdamW", lr=0.001, momentum=0.9, decay=0.0, iterations=1)
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    frozen_in_optimizer = [
        name for name, param in model.named_parameters() if not param.requires_grad and id(param) in optimizer_param_ids
    ]
    if frozen_in_optimizer:
        raise AssertionError(f"frozen parameters entered optimizer groups: {frozen_in_optimizer[:5]}.")

    trainer._model_train()
    head = model.model[-1]
    if head.training:
        raise AssertionError("frozen base GCSLaneHead parent should stay eval to suppress base-path stat drift.")
    if not head.short_segment_x_mlp.training:
        raise AssertionError("short-segment endpoint MLP should stay in train mode.")
    bn_training = [name for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d) and module.training]
    if bn_training:
        raise AssertionError(f"frozen base BatchNorm modules stayed in train mode: {bn_training[:5]}.")


def main() -> None:
    check_yaml_forward()
    check_loss()
    check_backward()
    check_empty_segment_batch_has_grad()
    check_freeze_contract()
    print("GCS short-segment proposal head checks passed.")


if __name__ == "__main__":
    main()
