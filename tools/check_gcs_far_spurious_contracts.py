from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_gcs import parse_args  # noqa: E402
from ultralytics.cfg import CFG_FLOAT_KEYS, CFG_INT_KEYS  # noqa: E402
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.models.yolo.gcs_lane.val import DEFAULT_LOSS_GAINS, LOSS_GAIN_ARGS, LOSS_NAMES  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402


def _make_gt(num_lanes: int, k: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Build simple fixed-y-like GT lanes for far-spurious checks."""
    if num_lanes == 3:
        xs = [0.10, 0.35, 0.60]
    elif num_lanes == 4:
        xs = [0.10, 0.30, 0.55, 0.75]
    elif num_lanes == 5:
        xs = [0.10, 0.30, 0.50, 0.70, 0.85]
    else:
        raise ValueError(f"unsupported synthetic lane count: {num_lanes}")

    gt = torch.zeros(num_lanes, k, 2)
    valid = torch.ones(num_lanes, k)
    for lane_i, x in enumerate(xs):
        gt[lane_i, :, 0] = x
        gt[lane_i, :, 1] = torch.linspace(0.95, 0.20, k)
    return gt, valid


def _make_preds(
    gt: torch.Tensor,
    valid: torch.Tensor,
    *,
    candidate_x: float = 0.95,
    candidate_logit: float = -2.0,
    q: int = 8,
) -> tuple[dict[str, torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Build predictions with one unmatched candidate after matched GT queries."""
    num_lanes, k = gt.shape[:2]
    pred_points = torch.zeros(1, q, k, 2)
    pred_logits = torch.full((1, q), -6.0)
    pred_valid_logits = torch.full((1, q, k), -6.0)

    for lane_i in range(num_lanes):
        pred_points[0, lane_i] = gt[lane_i]
        pred_logits[0, lane_i] = 6.0
        pred_valid_logits[0, lane_i, valid[lane_i] > 0.5] = 6.0

    candidate_q = num_lanes
    pred_points[0, candidate_q, :, 0] = candidate_x
    pred_points[0, candidate_q, :, 1] = gt[0, :, 1]
    pred_logits[0, candidate_q] = candidate_logit
    pred_valid_logits[0, candidate_q, :4] = 6.0

    preds = {
        "pred_points": pred_points,
        "pred_logits": pred_logits,
        "pred_valid_logits": pred_valid_logits,
    }
    indices = [(torch.arange(num_lanes, dtype=torch.long), torch.arange(num_lanes, dtype=torch.long))]
    return preds, indices


def _loss_args(**overrides) -> dict:
    args = {"gcs_imgsz": [544, 960]}
    args.update(overrides)
    return args


def check_loss_names_and_gains() -> None:
    """Check train/val loss vectors and validation gain mapping stay aligned."""
    assert GCSLoss.loss_names == GCSLaneTrainer.loss_names
    assert GCSLoss.loss_names == LOSS_NAMES
    assert len(GCSLoss.loss_names) == len(GCSLaneTrainer.progress_loss_names) == 40
    assert len(LOSS_NAMES) == len(LOSS_GAIN_ARGS) == len(DEFAULT_LOSS_GAINS) == 40

    idx = LOSS_NAMES.index("far_spur_loss")
    assert LOSS_GAIN_ARGS[idx] == "gcs_far_spurious_neg"
    assert LOSS_GAIN_ARGS[idx + 1 : idx + 8] == (None, None, None, None, None, None, None)
    assert DEFAULT_LOSS_GAINS[idx : idx + 8] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def check_cli_and_config_defaults() -> None:
    """Check train CLI defaults, default.yaml values, and Ultralytics config typing."""
    expected = {
        "gcs_far_spurious_neg": 0.0,
        "gcs_far_spurious_min_gt_lanes": 3,
        "gcs_far_spurious_max_gt_lanes": 4,
        "gcs_far_spurious_min_valid": 2,
        "gcs_far_spurious_max_valid": 56,
        "gcs_far_spurious_far_px": 40.0,
        "gcs_far_spurious_protect_px": 25.0,
        "gcs_far_spurious_min_overlap": 3,
        "gcs_far_spurious_score_thr": 0.003,
        "gcs_far_spurious_gt3_weight": 1.0,
        "gcs_far_spurious_gt4_weight": 1.0,
        "gcs_far_spurious_gt5_weight": 0.0,
    }
    args = parse_args([])
    for name, value in expected.items():
        assert getattr(args, name) == value, (name, getattr(args, name), value)

    defaults = yaml.safe_load((ROOT / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8"))
    for name, value in expected.items():
        assert defaults[name] == value, (name, defaults[name], value)

    for name in (
        "gcs_far_spurious_neg",
        "gcs_far_spurious_far_px",
        "gcs_far_spurious_protect_px",
        "gcs_far_spurious_score_thr",
        "gcs_far_spurious_gt3_weight",
        "gcs_far_spurious_gt4_weight",
        "gcs_far_spurious_gt5_weight",
    ):
        assert name in CFG_FLOAT_KEYS, name
    for name in (
        "gcs_far_spurious_min_gt_lanes",
        "gcs_far_spurious_max_gt_lanes",
        "gcs_far_spurious_min_valid",
        "gcs_far_spurious_max_valid",
        "gcs_far_spurious_min_overlap",
    ):
        assert name in CFG_INT_KEYS, name


def check_direct_far_selection() -> None:
    """Check disabled zero behavior, far selection, GT protection, valid runs, and GT5-safe default."""
    gt3, valid3 = _make_gt(3)
    preds3, indices3 = _make_preds(gt3, valid3, candidate_x=0.95)

    crit_off = GCSLoss(_loss_args())
    out = crit_off.far_spurious_negative_loss(
        preds3["pred_points"], preds3["pred_logits"], None, indices3, torch.tensor([3.0]), [gt3], [valid3]
    )
    assert all(float(x) == 0.0 for x in out)

    crit_on = GCSLoss(_loss_args(gcs_far_spurious_neg=1.0))
    loss, cand, neg, gt3_count, gt4_count, gt5_count, score, valid = crit_on.far_spurious_negative_loss(
        preds3["pred_points"],
        preds3["pred_logits"],
        preds3["pred_valid_logits"],
        indices3,
        torch.tensor([3.0]),
        [gt3],
        [valid3],
    )
    assert float(loss) > 0.0
    assert (float(cand), float(neg), float(gt3_count), float(gt4_count), float(gt5_count)) == (1.0, 1.0, 1.0, 0.0, 0.0)
    assert float(score) > 0.003
    assert float(valid) == 4.0

    protected_preds, protected_indices = _make_preds(gt3, valid3, candidate_x=0.35)
    loss, cand, neg, *_ = crit_on.far_spurious_negative_loss(
        protected_preds["pred_points"],
        protected_preds["pred_logits"],
        protected_preds["pred_valid_logits"],
        protected_indices,
        torch.tensor([3.0]),
        [gt3],
        [valid3],
    )
    assert float(loss) == 0.0
    assert (float(cand), float(neg)) == (1.0, 0.0)

    split_valid_preds, split_valid_indices = _make_preds(gt3, valid3, candidate_x=0.95)
    split_valid_preds["pred_valid_logits"][0, 3] = -6.0
    split_valid_preds["pred_valid_logits"][0, 3, [0, 1, 3, 4]] = 6.0
    crit_min3 = GCSLoss(_loss_args(gcs_far_spurious_neg=1.0, gcs_far_spurious_min_valid=3))
    loss, cand, neg, *_ = crit_min3.far_spurious_negative_loss(
        split_valid_preds["pred_points"],
        split_valid_preds["pred_logits"],
        split_valid_preds["pred_valid_logits"],
        split_valid_indices,
        torch.tensor([3.0]),
        [gt3],
        [valid3],
    )
    assert float(loss) == 0.0
    assert (float(cand), float(neg)) == (0.0, 0.0)

    gt5, valid5 = _make_gt(5)
    preds5, indices5 = _make_preds(gt5, valid5, candidate_x=0.95)
    crit_gt5_default_weight = GCSLoss(
        _loss_args(gcs_far_spurious_neg=1.0, gcs_far_spurious_max_gt_lanes=5)
    )
    loss, cand, neg, gt3_count, gt4_count, gt5_count, *_ = crit_gt5_default_weight.far_spurious_negative_loss(
        preds5["pred_points"],
        preds5["pred_logits"],
        preds5["pred_valid_logits"],
        indices5,
        torch.tensor([5.0]),
        [gt5],
        [valid5],
    )
    assert float(loss) == 0.0
    assert (float(cand), float(neg), float(gt3_count), float(gt4_count), float(gt5_count)) == (1.0, 1.0, 0.0, 0.0, 1.0)


def check_forward_total_gain() -> None:
    """Check full forward logs far-spurious items and adds only the configured far gain."""
    gt, valid = _make_gt(3)
    preds, _ = _make_preds(gt, valid, candidate_x=0.95)
    batch = {"lanes": [gt], "lane_valid": [valid], "num_lanes": torch.tensor([3.0])}

    crit_off = GCSLoss(_loss_args())
    total_off, items_off = crit_off(preds, batch)
    idx = GCSLoss.loss_names.index("far_spur_loss")
    assert int(items_off.numel()) == len(GCSLoss.loss_names)
    assert float(items_off[idx]) == 0.0

    crit_on = GCSLoss(_loss_args(gcs_far_spurious_neg=0.25))
    total_on, items_on = crit_on(preds, batch)
    assert int(items_on.numel()) == len(GCSLoss.loss_names)
    assert float(items_on[idx]) > 0.0
    assert float(items_on[GCSLoss.loss_names.index("far_spur_neg")]) == 1.0
    assert abs(float(total_on - total_off) - 0.25 * float(items_on[idx])) < 1e-5


def main() -> None:
    check_loss_names_and_gains()
    check_cli_and_config_defaults()
    check_direct_far_selection()
    check_forward_total_gain()
    print("GCS far spurious-negative loss contract checks passed.")


if __name__ == "__main__":
    main()
