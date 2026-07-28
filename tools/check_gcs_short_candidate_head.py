"""Smoke-test the default-off GCS short lateral candidate head and loss."""

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

from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel  # noqa: E402
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402


DEFAULT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"
CANDIDATE_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-gated-candidate-v2.yaml"


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _head_from_yaml(cfg: Path) -> GCSLaneHead:
    model = GCSLaneModel(str(cfg), nc=1, verbose=False)
    head = model.model[-1]
    if not isinstance(head, GCSLaneHead):
        raise AssertionError(f"{cfg} did not build a GCSLaneHead.")
    return head.eval()


def _head_features(head: GCSLaneHead, batch: int = 2) -> list[torch.Tensor]:
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
    if "pred_short_candidate_points" in default_out or "pred_short_candidate_logits" in default_out:
        raise AssertionError("default query YAML unexpectedly emitted short candidate outputs.")

    candidate_head = _head_from_yaml(CANDIDATE_CFG)
    candidate_out = candidate_head(_head_features(candidate_head), orig_size=(544, 960))
    required = {"pred_short_candidate_points", "pred_short_candidate_logits", "pred_short_candidate_offsets_px"}
    missing = required.difference(candidate_out)
    if missing:
        raise AssertionError(f"candidate YAML missing output keys: {sorted(missing)}.")
    points = candidate_out["pred_points"]
    cand_points = candidate_out["pred_short_candidate_points"]
    cand_logits = candidate_out["pred_short_candidate_logits"]
    if tuple(cand_points.shape) != (2, 12, 7, 56, 2):
        raise AssertionError(f"candidate point shape mismatch: {tuple(cand_points.shape)}.")
    if tuple(cand_logits.shape) != (2, 12, 7):
        raise AssertionError(f"candidate logit shape mismatch: {tuple(cand_logits.shape)}.")
    offsets = candidate_out["pred_short_candidate_offsets_px"].detach().cpu().tolist()
    if offsets != [0.0, -20.0, 20.0, -40.0, 40.0, -60.0, 60.0]:
        raise AssertionError(f"candidate offsets mismatch: {offsets}.")
    expected = points.unsqueeze(2).expand_as(cand_points).clone()
    expected[..., 0] = (expected[..., 0] + candidate_out["pred_short_candidate_offsets_px"].view(1, 1, 7, 1) / 960.0).clamp(0.0, 1.0)
    if not torch.allclose(cand_points, expected, atol=1e-6):
        raise AssertionError("candidate points are not fixed lateral offsets from pred_points.")


def _make_short_candidate_batch() -> dict:
    k = 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    xs = torch.tensor([0.4, 0.3, 0.6, 0.75], dtype=torch.float32).view(4, 1).expand(4, k)
    lanes = torch.stack((xs, y.view(1, k).expand(4, k)), dim=-1).float()
    valid = torch.zeros(4, k, dtype=torch.float32)
    valid[0, :6] = 1.0
    valid[1:, :16] = 1.0
    return {
        "lanes": [lanes],
        "lane_valid": [valid],
        "num_lanes": torch.tensor([4], dtype=torch.long),
    }


def _make_short_candidate_preds() -> dict[str, torch.Tensor]:
    b, q, m, k = 1, 12, 7, 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k).view(1, 1, k).expand(b, q, k)
    x = torch.full((b, q, k), 0.5)
    pred_points = torch.stack((x, y), dim=-1).contiguous()
    candidate_points = pred_points.unsqueeze(2).expand(b, q, m, k, 2).clone()
    candidate_points[0, 0, 1, :, 0] = 0.4
    candidate_logits = torch.full((b, q, m), _logit(0.02), dtype=torch.float32)
    candidate_logits[0, 0, 1] = _logit(0.5)
    return {
        "pred_points": pred_points,
        "pred_logits": torch.full((b, q), _logit(0.2), dtype=torch.float32),
        "pred_valid_logits": torch.full((b, q, k), _logit(0.7), dtype=torch.float32),
        "pred_short_candidate_points": candidate_points,
        "pred_short_candidate_logits": candidate_logits,
    }


def check_loss() -> None:
    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_candidate": 0.1,
            "gcs_short_candidate_topk": 3,
            "gcs_short_candidate_visible_thr": 10,
            "gcs_short_candidate_neg_score_thr": 0.6,
        }
    )
    _, items = criterion(_make_short_candidate_preds(), _make_short_candidate_batch())
    if int(items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(f"GCSLoss item length mismatch: {items.numel()} vs {len(GCSLoss.loss_names)}.")
    if not torch.isfinite(items).all():
        raise AssertionError("GCSLoss produced non-finite short candidate items.")
    idx = {name: i for i, name in enumerate(GCSLoss.loss_names)}
    if float(items[idx["short_candidate_pos_count"]]) <= 0.0:
        raise AssertionError("short candidate assignment did not produce a strong positive.")
    if float(items[idx["short_candidate_loss"]]) <= 0.0:
        raise AssertionError("short candidate loss should be positive for the synthetic candidate.")

    missing = _make_short_candidate_preds()
    missing.pop("pred_short_candidate_points")
    try:
        criterion(missing, _make_short_candidate_batch())
    except KeyError:
        pass
    else:
        raise AssertionError("gcs_short_candidate > 0 should require candidate head outputs.")


def check_backward() -> None:
    preds = _make_short_candidate_preds()
    candidate_points = preds["pred_short_candidate_points"].detach().clone()
    candidate_points[0, 0, 1, :, 0] = 0.405
    candidate_points.requires_grad_(True)
    candidate_logits = preds["pred_short_candidate_logits"].detach().clone().requires_grad_(True)
    preds["pred_short_candidate_points"] = candidate_points
    preds["pred_short_candidate_logits"] = candidate_logits

    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_candidate": 0.1,
            "gcs_short_candidate_topk": 3,
            "gcs_short_candidate_visible_thr": 10,
            "gcs_short_candidate_neg_score_thr": 0.6,
        }
    )
    total, items = criterion(preds, _make_short_candidate_batch())
    total.backward()
    if not torch.isfinite(total.detach()) or not torch.isfinite(items).all():
        raise AssertionError("short candidate backward smoke produced non-finite losses.")
    if candidate_logits.grad is None or not torch.isfinite(candidate_logits.grad).all():
        raise AssertionError("short candidate score logits did not receive finite gradients.")
    if float(candidate_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short candidate score logits received zero gradient.")
    if candidate_points.grad is None or not torch.isfinite(candidate_points.grad).all():
        raise AssertionError("short candidate points did not receive finite pull gradients.")
    if float(candidate_points.grad.abs().sum()) <= 0.0:
        raise AssertionError("short candidate points received zero pull gradient.")


def check_empty_candidate_batch_has_grad() -> None:
    preds = _make_short_candidate_preds()
    candidate_logits = preds["pred_short_candidate_logits"].detach().clone().requires_grad_(True)
    preds["pred_short_candidate_logits"] = candidate_logits
    batch = _make_short_candidate_batch()
    batch["num_lanes"] = torch.tensor([3], dtype=torch.long)
    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_candidate": 0.1,
            "gcs_short_candidate_topk": 3,
            "gcs_short_candidate_visible_thr": 10,
            "gcs_short_candidate_neg_score_thr": 0.6,
        }
    )
    total, items = criterion(preds, batch)
    if not total.requires_grad:
        raise AssertionError("empty short-candidate batch should keep a zero-gradient path to candidate logits.")
    total.backward()
    if candidate_logits.grad is None or not torch.isfinite(candidate_logits.grad).all():
        raise AssertionError("empty short-candidate batch did not produce finite zero gradients.")
    if float(candidate_logits.grad.abs().sum()) != 0.0:
        raise AssertionError("empty short-candidate batch should not update candidate logits.")
    if not torch.isfinite(items).all():
        raise AssertionError("empty short-candidate batch produced non-finite loss items.")


def check_freeze_contract() -> None:
    model = GCSLaneModel(str(CANDIDATE_CFG), nc=1, verbose=False)
    trainer = object.__new__(GCSLaneTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(gcs_short_candidate_freeze_base=True, gcs_short_candidate=0.1)
    trainer.freeze_layer_names = []
    trainer._apply_custom_freeze()

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise AssertionError("short-candidate freeze contract left no trainable parameters.")
    bad = [name for name in trainable if "short_candidate_" not in name]
    if bad:
        raise AssertionError(f"short-candidate freeze contract left base parameters trainable: {bad[:5]}.")
    if not any("short_candidate_score_mlp" in name for name in trainable):
        raise AssertionError("short-candidate score MLP is not trainable under freeze contract.")

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
    missing_trainable = [
        name for name, param in model.named_parameters() if param.requires_grad and id(param) not in optimizer_param_ids
    ]
    if missing_trainable:
        raise AssertionError(f"trainable short-candidate parameters missing from optimizer: {missing_trainable[:5]}.")

    trainer._model_train()
    head = model.model[-1]
    if head.training:
        raise AssertionError("frozen base GCSLaneHead parent should stay eval to suppress base-path stat drift.")
    if not head.short_candidate_score_mlp.training:
        raise AssertionError("short-candidate score MLP should stay in train mode.")
    bn_training = [name for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d) and module.training]
    if bn_training:
        raise AssertionError(f"frozen base BatchNorm modules stayed in train mode: {bn_training[:5]}.")


def check_decode() -> None:
    q, m, k = 2, 3, 6
    y = torch.linspace(0.9, 0.4, k)
    pred_points = torch.stack(
        (
            torch.stack((torch.full((k,), 0.5), y), dim=1),
            torch.stack((torch.full((k,), 0.7), y), dim=1),
        ),
        dim=0,
    )
    pred_logits = torch.tensor([_logit(0.9), _logit(0.8)], dtype=torch.float32)
    pred_valid_logits = torch.full((q, k), _logit(0.9), dtype=torch.float32)
    candidate_points = pred_points.unsqueeze(1).expand(q, m, k, 2).clone()
    candidate_points[0, 1, :, 0] = 0.4
    candidate_logits = torch.full((q, m), _logit(0.01), dtype=torch.float32)
    candidate_logits[0, 1] = _logit(0.95)

    base_lanes = decode_gcs_predictions(
        pred_points,
        pred_logits,
        pred_valid_logits=pred_valid_logits,
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=2,
        max_det=2,
    )
    cand_lanes = decode_gcs_predictions(
        pred_points,
        pred_logits,
        pred_valid_logits=pred_valid_logits,
        pred_short_candidate_points=candidate_points,
        pred_short_candidate_logits=candidate_logits,
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=2,
        max_det=2,
        candidate_decode=True,
        candidate_score_thr=0.05,
        candidate_short_min_points=2,
        candidate_short_max_points=10,
    )
    base_q0 = next(x for x in base_lanes if int(x["query"]) == 0)
    cand_q0 = next(x for x in cand_lanes if int(x["query"]) == 0)
    if abs(float(base_q0["points_norm"][0, 0]) - 0.5) > 1e-6:
        raise AssertionError("base decode synthetic query did not start at x=0.5.")
    if abs(float(cand_q0["points_norm"][0, 0]) - 0.4) > 1e-6:
        raise AssertionError("candidate decode did not replace gated query geometry.")
    if int(cand_q0.get("short_candidate_index", -1)) != 1:
        raise AssertionError(f"candidate decode metadata mismatch: {cand_q0}.")

    try:
        decode_gcs_predictions(
            pred_points,
            pred_logits,
            pred_valid_logits=pred_valid_logits,
            pred_short_candidate_points=candidate_points,
            pred_short_candidate_logits=candidate_logits,
            score_thr=0.1,
            count_aware_topk=True,
            candidate_decode=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("candidate_decode should reject count_aware_topk.")


def main() -> None:
    check_yaml_forward()
    check_loss()
    check_backward()
    check_empty_candidate_batch_has_grad()
    check_freeze_contract()
    check_decode()
    print("GCS short candidate head checks passed.")


if __name__ == "__main__":
    main()
