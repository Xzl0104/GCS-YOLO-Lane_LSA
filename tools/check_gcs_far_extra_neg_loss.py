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

from ultralytics.models.yolo.gcs_lane import val as gcs_val  # noqa: E402
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402


def _fixture(
    *,
    gt_count: int = 4,
    far_query_x: float = 0.92,
    near_query_x: float = 0.22,
    match_far_query: bool = False,
) -> tuple[dict[str, torch.Tensor], dict, list[tuple[torch.Tensor, torch.Tensor]]]:
    b, q, k, n = 1, 12, 56, gt_count
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    lane_x = torch.linspace(0.2, 0.65, n).view(n, 1).expand(n, k)
    lanes = torch.stack((lane_x, y.view(1, k).expand(n, k)), dim=-1).float()
    lane_valid = torch.ones(n, k)

    pred_points = torch.zeros(b, q, k, 2)
    pred_points[0, :n] = lanes
    pred_points[0, n] = torch.stack((torch.full((k,), far_query_x), y), dim=-1)
    pred_points[0, n + 1] = torch.stack((torch.full((k,), near_query_x), y), dim=-1)
    pred_points[0, n + 2 :] = lanes[-1:].expand(q - n - 2, k, 2)

    pred_logits = torch.full((b, q), -4.0)
    pred_logits[0, :n] = 2.0
    pred_logits[0, n] = 1.5
    pred_logits[0, n + 1] = 1.5

    pred_valid_logits = torch.full((b, q, k), -4.0)
    pred_valid_logits[0, :n] = 4.0
    pred_valid_logits[0, n, :8] = 4.0
    pred_valid_logits[0, n + 1, :8] = 4.0

    src = torch.arange(n)
    tgt = torch.arange(n)
    if match_far_query:
        src = torch.cat((src, torch.tensor([n])))
        tgt = torch.cat((tgt, torch.tensor([0])))

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
    return preds, batch, [(src, tgt)]


def test_loss_name_contracts_match() -> None:
    if tuple(GCSLoss.loss_names) != tuple(GCSLaneTrainer.loss_names):
        raise AssertionError("GCSLoss and GCSLaneTrainer loss_names must match exactly.")
    if tuple(GCSLoss.loss_names) != tuple(gcs_val.LOSS_NAMES):
        raise AssertionError("GCSLoss and GCS validator LOSS_NAMES must match exactly.")
    if len(gcs_val.LOSS_NAMES) != len(gcs_val.LOSS_GAIN_ARGS):
        raise AssertionError("GCS validator LOSS_GAIN_ARGS length must match LOSS_NAMES.")
    if len(gcs_val.LOSS_NAMES) != len(gcs_val.DEFAULT_LOSS_GAINS):
        raise AssertionError("GCS validator DEFAULT_LOSS_GAINS length must match LOSS_NAMES.")
    idx = gcs_val.LOSS_NAMES.index("far_extra_neg_loss")
    if gcs_val.LOSS_GAIN_ARGS[idx] != "gcs_far_extra_neg":
        raise AssertionError("far_extra_neg_loss must be weighted by gcs_far_extra_neg in validation.")


def test_rejects_invalid_params() -> None:
    invalid = (
        ("gcs_far_extra_neg", -0.1),
        ("gcs_far_extra_min_gt_lanes", -1),
        ("gcs_far_extra_max_gt_lanes", 2),
        ("gcs_far_extra_min_valid", -1),
        ("gcs_far_extra_max_valid", 3),
        ("gcs_far_extra_valid_thr", 1.1),
        ("gcs_far_extra_score_thr", -0.1),
        ("gcs_far_extra_dist_thr", -1.0),
        ("gcs_far_extra_gt4_weight", -0.1),
    )
    for key, value in invalid:
        args = {"gcs_imgsz": [544, 960], "gcs_far_extra_min_gt_lanes": 3, "gcs_far_extra_min_valid": 4}
        args[key] = value
        try:
            GCSLoss(args)
        except ValueError as exc:
            if "gcs_far_extra" not in str(exc):
                raise AssertionError(f"unexpected ValueError for {key}: {exc}") from exc
            continue
        raise AssertionError(f"invalid {key}={value!r} should raise ValueError.")


def test_default_off_forward_loss_matches_old_total() -> None:
    preds, batch, _ = _fixture()
    off = GCSLoss({"gcs_imgsz": [544, 960], "gcs_far_extra_neg": 0.0})
    ref = GCSLoss({"gcs_imgsz": [544, 960]})

    total_off, items_off = off(preds, batch)
    total_ref, items_ref = ref(preds, batch)
    names = GCSLoss.loss_names
    if len(items_off) != len(names):
        raise AssertionError(f"loss item length mismatch: got={len(items_off)}, expected={len(names)}.")
    if not torch.allclose(total_off, total_ref):
        raise AssertionError(f"default-off total changed: got={float(total_off)}, expected={float(total_ref)}.")
    for name in ("far_extra_neg_loss", "far_extra_neg_count", "far_extra_score_mean"):
        value = items_off[names.index(name)]
        if float(value) != 0.0:
            raise AssertionError(f"default-off {name} should be zero, got {float(value)}.")
    if not torch.isfinite(items_ref).all():
        raise AssertionError("default reference produced non-finite items.")


def test_far_unmatched_query_is_selected() -> None:
    preds, batch, indices = _fixture()
    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_far_extra_neg": 0.1,
            "gcs_far_extra_min_valid": 4,
            "gcs_far_extra_max_valid": 24,
            "gcs_far_extra_valid_thr": 0.5,
            "gcs_far_extra_score_thr": 0.05,
            "gcs_far_extra_dist_thr": 80.0,
        }
    )
    loss, count, score = criterion.far_extra_neg_loss(
        preds["pred_points"],
        preds["pred_logits"],
        preds["pred_valid_logits"],
        batch["lanes"],
        batch["lane_valid"],
        indices,
        batch["num_lanes"],
    )
    if float(loss) <= 0.0:
        raise AssertionError(f"far unmatched query should produce positive loss, got {float(loss)}.")
    if int(count.item()) != 1:
        raise AssertionError(f"expected one selected far-extra query, got {int(count.item())}.")
    if not (0.0 < float(score) < 1.0):
        raise AssertionError(f"expected finite selected score mean, got {float(score)}.")


def test_near_query_is_protected_by_distance() -> None:
    preds, batch, indices = _fixture(far_query_x=0.23, near_query_x=0.24)
    criterion = GCSLoss({"gcs_imgsz": [544, 960], "gcs_far_extra_neg": 0.1, "gcs_far_extra_dist_thr": 80.0})
    loss, count, score = criterion.far_extra_neg_loss(
        preds["pred_points"],
        preds["pred_logits"],
        preds["pred_valid_logits"],
        batch["lanes"],
        batch["lane_valid"],
        indices,
        batch["num_lanes"],
    )
    if any(float(x) != 0.0 for x in (loss, count, score)):
        raise AssertionError("near-GT unmatched queries should not receive far-extra loss.")


def test_matched_query_is_not_selected() -> None:
    preds, batch, indices = _fixture(match_far_query=True)
    criterion = GCSLoss({"gcs_imgsz": [544, 960], "gcs_far_extra_neg": 0.1, "gcs_far_extra_dist_thr": 80.0})
    loss, count, _ = criterion.far_extra_neg_loss(
        preds["pred_points"],
        preds["pred_logits"],
        preds["pred_valid_logits"],
        batch["lanes"],
        batch["lane_valid"],
        indices,
        batch["num_lanes"],
    )
    if int(count.item()) != 0 or float(loss) != 0.0:
        raise AssertionError("matched queries should not receive far-extra loss.")


def main() -> None:
    test_loss_name_contracts_match()
    test_rejects_invalid_params()
    test_default_off_forward_loss_matches_old_total()
    test_far_unmatched_query_is_selected()
    test_near_query_is_protected_by_distance()
    test_matched_query_is_not_selected()
    print(json.dumps({"status": "ok", "tests": 6}, indent=2))


if __name__ == "__main__":
    main()
