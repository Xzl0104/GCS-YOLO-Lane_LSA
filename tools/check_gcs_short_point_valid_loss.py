from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel  # noqa: E402


EXPECTED_LOSS_NAMES = (
    "exist_loss",
    "point_loss",
    "point_valid_loss",
    "smooth_loss",
    "curve_loss",
    "mask_loss",
    "edge_loss",
    "count_loss",
    "count_under5_loss",
    "count_boundary_loss",
    "spurious_neg_loss",
    "spurious_negative_count",
    "spur_cand",
    "spur_prot",
    "spur_final",
    "spur_neg",
    "spur_cnt_gt3",
    "spur_cnt_gt4",
    "spur_cnt_gt5",
    "spur_neg_gt3",
    "spur_neg_gt4",
    "spur_neg_gt5",
    "count_score_mean",
    "gt5_short_pos_count",
    "gt5_short_pos_anchor_count",
    "gt5_short_point_valid_loss",
    "cnt_bound_5under",
    "cnt_score",
    "boundary_pseudo_neg_loss",
    "boundary_pseudo_count",
    "boundary_pseudo_score_mean",
    "query_count_ce_loss",
    "query_count_acc",
    "query_count_pred_mean",
    "gt4_short_pos_count",
    "gt4_short_pos_anchor_count",
    "gt4_short_point_valid_loss",
)


def _case_inputs(
    gt_counts: list[int],
    visible_counts: list[int],
    *,
    q: int = 2,
    k: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    if len(gt_counts) != len(visible_counts):
        raise AssertionError("gt_counts and visible_counts must have the same length.")
    b = len(gt_counts)
    pred_valid_logits = torch.linspace(-1.35, 1.75, steps=b * q * k, dtype=torch.float32).reshape(b, q, k)
    pred_points = torch.zeros(b, q, k, 2, dtype=torch.float32)
    gt_valid = []
    indices = []
    for gt_count, visible_count in zip(gt_counts, visible_counts):
        if visible_count > k:
            raise AssertionError(f"visible_count={visible_count} exceeds K={k}.")
        valid = torch.zeros(gt_count, k, dtype=torch.float32)
        valid[0, :visible_count] = 1.0
        gt_valid.append(valid)
        indices.append((torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)))
    return pred_valid_logits, pred_points, gt_valid, indices, torch.tensor(gt_counts, dtype=torch.long)


def _expected_point_valid_loss(
    pred_valid_logits: torch.Tensor,
    gt_counts: list[int],
    visible_counts: list[int],
    *,
    pos_weight_max: float = 10.0,
    gt4_visible_thr: int = 0,
    gt4_weight: float = 1.0,
    gt5_visible_thr: int = 0,
    gt5_weight: float = 1.0,
    weight_whole_matched_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.zeros_like(pred_valid_logits)
    extra_weight = torch.ones_like(pred_valid_logits)
    for b, (gt_count, visible_count) in enumerate(zip(gt_counts, visible_counts)):
        target[b, 0, :visible_count] = 1.0
        weight = 1.0
        if gt_count == 4 and gt4_visible_thr > 0 and visible_count <= gt4_visible_thr:
            weight = gt4_weight
        elif gt_count == 5 and gt5_visible_thr > 0 and visible_count <= gt5_visible_thr:
            weight = gt5_weight
        if weight != 1.0:
            if weight_whole_matched_query:
                extra_weight[b, 0, :] = float(weight)
            else:
                extra_weight[b, 0, :visible_count] = float(weight)

    pos = target.sum().clamp_min(1.0)
    neg = (target.numel() - target.sum()).clamp_min(1.0)
    pos_weight = (neg / pos).clamp(min=1.0, max=float(pos_weight_max)).to(pred_valid_logits)
    bce = F.binary_cross_entropy_with_logits(pred_valid_logits, target, pos_weight=pos_weight, reduction="none")
    loss = (bce * extra_weight).sum() / extra_weight.sum().clamp_min(1.0)
    return loss, bce, extra_weight


def _actual_point_valid(
    args: dict,
    gt_counts: list[int],
    visible_counts: list[int],
) -> tuple[
    GCSLoss,
    torch.Tensor,
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]:
    criterion = GCSLoss({"gcs_imgsz": [544, 960], **args})
    pred_valid_logits, pred_points, gt_valid, indices, gt_lanes = _case_inputs(gt_counts, visible_counts)
    details = criterion.point_valid_loss(
        pred_valid_logits,
        pred_points,
        gt_valid,
        indices,
        gt_lanes=gt_lanes,
        return_details=True,
    )
    return criterion, pred_valid_logits, details


def _assert_close(name: str, got: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-6) -> None:
    if not torch.allclose(got, expected, atol=atol, rtol=1e-6):
        raise AssertionError(f"{name}: got={float(got):.8f}, expected={float(expected):.8f}.")


def _assert_zero(name: str, value: torch.Tensor) -> None:
    if float(value.detach().cpu()) != 0.0:
        raise AssertionError(f"{name}: expected zero, got={float(value):.8f}.")


def _assert_zero_details(prefix: str, details: tuple[torch.Tensor, ...], start: int = 1) -> None:
    for offset, value in enumerate(details[start:], start=start):
        _assert_zero(f"{prefix} details[{offset}]", value)


def test_gt4_default_no_weight() -> None:
    criterion, logits, details = _actual_point_valid({}, [4], [10])
    got = details[0]
    expected, _, _ = _expected_point_valid_loss(
        logits,
        [4],
        [10],
        pos_weight_max=criterion.point_valid_pos_weight_max,
    )
    _assert_close("GT4 default point-valid loss", got, expected)
    _assert_zero_details("GT4 default", details)


def test_gt4_visible_le_10_weights_visible_anchors_only() -> None:
    args = {"gcs_gt4_short_visible_thr": 10, "gcs_gt4_short_point_valid_weight": 1.2}
    criterion, logits, details = _actual_point_valid(args, [4], [10])
    got = details[0]
    expected, _, _ = _expected_point_valid_loss(
        logits,
        [4],
        [10],
        pos_weight_max=criterion.point_valid_pos_weight_max,
        gt4_visible_thr=10,
        gt4_weight=1.2,
    )
    unweighted, _, _ = _expected_point_valid_loss(
        logits,
        [4],
        [10],
        pos_weight_max=criterion.point_valid_pos_weight_max,
    )
    whole_query_weighted, _, _ = _expected_point_valid_loss(
        logits,
        [4],
        [10],
        pos_weight_max=criterion.point_valid_pos_weight_max,
        gt4_visible_thr=10,
        gt4_weight=1.2,
        weight_whole_matched_query=True,
    )
    if torch.allclose(got, unweighted, atol=1e-6, rtol=1e-6):
        raise AssertionError("GT4 short point-valid rescue did not apply the configured positive weight.")
    if torch.allclose(got, whole_query_weighted, atol=1e-6, rtol=1e-6):
        raise AssertionError("GT4 short point-valid rescue weighted invisible anchors on the matched query.")
    _assert_close("GT4 visible<=10 point-valid loss", got, expected)
    _, bce, _ = _expected_point_valid_loss(
        logits,
        [4],
        [10],
        pos_weight_max=criterion.point_valid_pos_weight_max,
    )
    _assert_zero("GT4 visible<=10 gt5_short_pos_count", details[1])
    _assert_zero("GT4 visible<=10 gt5_short_pos_anchor_count", details[2])
    _assert_zero("GT4 visible<=10 gt5_short_point_valid_loss", details[3])
    _assert_close("GT4 rescued lane count", details[4], torch.tensor(1.0))
    _assert_close("GT4 rescued anchor count", details[5], torch.tensor(10.0))
    _assert_close("GT4 diagnostic BCE", details[6], bce[0, 0, :10].mean())


def test_gt4_visible_11_no_weight() -> None:
    args = {"gcs_gt4_short_visible_thr": 10, "gcs_gt4_short_point_valid_weight": 1.2}
    criterion, logits, details = _actual_point_valid(args, [4], [11])
    expected, _, _ = _expected_point_valid_loss(
        logits,
        [4],
        [11],
        pos_weight_max=criterion.point_valid_pos_weight_max,
    )
    _assert_close("GT4 visible=11 point-valid loss", details[0], expected)
    _assert_zero_details("GT4 visible=11", details)


def test_gt4_rescue_disabled_when_loss_eval_mode() -> None:
    args = {"gcs_gt4_short_visible_thr": 10, "gcs_gt4_short_point_valid_weight": 1.2}
    criterion = GCSLoss({"gcs_imgsz": [544, 960], **args})
    criterion.eval()
    pred_valid_logits, pred_points, gt_valid, indices, gt_lanes = _case_inputs([4], [10])
    details = criterion.point_valid_loss(
        pred_valid_logits,
        pred_points,
        gt_valid,
        indices,
        gt_lanes=gt_lanes,
        return_details=True,
    )
    expected, _, _ = _expected_point_valid_loss(
        pred_valid_logits,
        [4],
        [10],
        pos_weight_max=criterion.point_valid_pos_weight_max,
    )
    _assert_close("GT4 eval-mode point-valid loss", details[0], expected)
    _assert_zero_details("GT4 eval-mode", details)


def test_gt5_old_behavior_visible_le_10_unchanged() -> None:
    args = {"gcs_gt5_short_visible_thr": 10, "gcs_gt5_short_point_valid_weight": 1.2}
    criterion, logits, details = _actual_point_valid(args, [5], [10])
    got, gt5_pos_count, gt5_anchor_count, gt5_diag = details[:4]
    expected, bce, _ = _expected_point_valid_loss(
        logits,
        [5],
        [10],
        pos_weight_max=criterion.point_valid_pos_weight_max,
        gt5_visible_thr=10,
        gt5_weight=1.2,
    )
    _assert_close("GT5 visible<=10 point-valid loss", got, expected)
    _assert_close("GT5 rescued lane count", gt5_pos_count, torch.tensor(1.0))
    _assert_close("GT5 rescued anchor count", gt5_anchor_count, torch.tensor(10.0))
    _assert_close("GT5 diagnostic BCE", gt5_diag, bce[0, 0, :10].mean())
    _assert_zero("GT5 old behavior gt4_short_pos_count", details[4])
    _assert_zero("GT5 old behavior gt4_short_pos_anchor_count", details[5])
    _assert_zero("GT5 old behavior gt4_short_point_valid_loss", details[6])


def test_gt3_no_weight_even_when_gt4_gt5_enabled() -> None:
    args = {
        "gcs_gt4_short_visible_thr": 10,
        "gcs_gt4_short_point_valid_weight": 1.2,
        "gcs_gt5_short_visible_thr": 10,
        "gcs_gt5_short_point_valid_weight": 1.5,
    }
    criterion, logits, details = _actual_point_valid(args, [3], [10])
    expected, _, _ = _expected_point_valid_loss(
        logits,
        [3],
        [10],
        pos_weight_max=criterion.point_valid_pos_weight_max,
    )
    _assert_close("GT3 point-valid loss", details[0], expected)
    _assert_zero_details("GT3", details)


def test_gt4_gt5_enabled_branch_by_gt_count() -> None:
    args = {
        "gcs_gt4_short_visible_thr": 10,
        "gcs_gt4_short_point_valid_weight": 1.2,
        "gcs_gt5_short_visible_thr": 10,
        "gcs_gt5_short_point_valid_weight": 1.5,
    }
    criterion, logits, details = _actual_point_valid(args, [4, 5], [10, 10])
    got, gt5_pos_count, gt5_anchor_count, gt5_diag = details[:4]
    expected, bce, _ = _expected_point_valid_loss(
        logits,
        [4, 5],
        [10, 10],
        pos_weight_max=criterion.point_valid_pos_weight_max,
        gt4_visible_thr=10,
        gt4_weight=1.2,
        gt5_visible_thr=10,
        gt5_weight=1.5,
    )
    _assert_close("GT4/GT5 branch-specific point-valid loss", got, expected)
    _assert_close("GT4/GT5 gt5 rescued lane count", gt5_pos_count, torch.tensor(1.0))
    _assert_close("GT4/GT5 gt5 rescued anchor count", gt5_anchor_count, torch.tensor(10.0))
    _assert_close("GT4/GT5 gt5 diagnostic BCE", gt5_diag, bce[1, 0, :10].mean())
    _assert_close("GT4/GT5 gt4 rescued lane count", details[4], torch.tensor(1.0))
    _assert_close("GT4/GT5 gt4 rescued anchor count", details[5], torch.tensor(10.0))
    _assert_close("GT4/GT5 gt4 diagnostic BCE", details[6], bce[0, 0, :10].mean())


def _extract_tuple(path: Path, name: str, *, class_name: str | None = None) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name is not None:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                body = node.body
                break
        else:
            raise AssertionError(f"{path}: class {class_name!r} not found.")
    for node in body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = ast.literal_eval(node.value)
                    return tuple(value)
    raise AssertionError(f"{path}: tuple assignment {name!r} not found.")


def test_loss_log_item_names_and_forward_length_stable() -> None:
    if tuple(GCSLoss.loss_names) != EXPECTED_LOSS_NAMES:
        raise AssertionError(f"GCSLoss.loss_names changed: {tuple(GCSLoss.loss_names)!r}.")
    train_names = _extract_tuple(
        ROOT / "ultralytics/models/yolo/gcs_lane/train.py",
        "loss_names",
        class_name="GCSLaneTrainer",
    )
    val_names = _extract_tuple(ROOT / "ultralytics/models/yolo/gcs_lane/val.py", "LOSS_NAMES")
    if train_names != EXPECTED_LOSS_NAMES:
        raise AssertionError("GCSLaneTrainer.loss_names no longer matches the stable GCSLoss order.")
    if val_names != EXPECTED_LOSS_NAMES:
        raise AssertionError("GCSLaneValidator LOSS_NAMES no longer matches the stable GCSLoss order.")

    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_gt4_short_visible_thr": 10,
            "gcs_gt4_short_point_valid_weight": 1.2,
            "gcs_gt5_short_visible_thr": 10,
            "gcs_gt5_short_point_valid_weight": 1.5,
        }
    )
    b, q, k, n = 1, 12, 56, 4
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    lane_x = torch.linspace(0.2, 0.8, n).view(n, 1).expand(n, k)
    lanes = torch.stack((lane_x, y.view(1, k).expand(n, k)), dim=-1).float()
    lane_valid = torch.ones(n, k, dtype=torch.float32)
    lane_valid[-1, 10:] = 0.0
    pred_points = torch.zeros(b, q, k, 2, dtype=torch.float32)
    pred_points[0, :n] = lanes
    pred_points[0, n:] = lanes[-1:].expand(q - n, k, 2)
    preds = {
        "pred_points": pred_points,
        "pred_logits": torch.zeros(b, q, dtype=torch.float32),
        "pred_valid_logits": torch.zeros(b, q, k, dtype=torch.float32),
    }
    batch = {
        "lanes": [lanes],
        "lane_valid": [lane_valid],
        "num_lanes": torch.tensor([4], dtype=torch.long),
    }
    total, loss_items = criterion(preds, batch)
    if int(loss_items.numel()) != len(EXPECTED_LOSS_NAMES):
        raise AssertionError(
            f"loss item length changed: got={loss_items.numel()}, expected={len(EXPECTED_LOSS_NAMES)}."
        )
    if not torch.isfinite(total) or not torch.isfinite(loss_items).all():
        raise AssertionError("GCSLoss.forward produced non-finite values for short point-valid contract check.")


def test_short_point_valid_weights_reject_suppression_values() -> None:
    for key in ("gcs_gt4_short_point_valid_weight", "gcs_gt5_short_point_valid_weight"):
        try:
            GCSLoss({"gcs_imgsz": [544, 960], key: 0.5})
        except ValueError as exc:
            if "must be >= 1.0" not in str(exc):
                raise AssertionError(f"{key} raised the wrong validation error: {exc}") from exc
        else:
            raise AssertionError(f"{key}=0.5 should fail because short point-valid weights are rescue-only.")

    GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_gt4_short_point_valid_weight": 1.0,
            "gcs_gt5_short_point_valid_weight": 1.0,
        }
    )


def test_gcs_model_loss_syncs_cached_criterion_training_state() -> None:
    model = GCSLaneModel("ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml", verbose=False)
    model.args = {
        "gcs_imgsz": [544, 960],
        "gcs_gt4_short_visible_thr": 10,
        "gcs_gt4_short_point_valid_weight": 1.2,
    }
    b, q, k, n = 1, 12, 56, 4
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    lane_x = torch.linspace(0.2, 0.8, n).view(n, 1).expand(n, k)
    lanes = torch.stack((lane_x, y.view(1, k).expand(n, k)), dim=-1).float()
    lane_valid = torch.ones(n, k, dtype=torch.float32)
    lane_valid[-1, 10:] = 0.0
    pred_points = torch.zeros(b, q, k, 2, dtype=torch.float32)
    pred_points[0, :n] = lanes
    pred_points[0, n:] = lanes[-1:].expand(q - n, k, 2)
    preds = {
        "pred_points": pred_points,
        "pred_logits": torch.zeros(b, q, dtype=torch.float32),
        "pred_valid_logits": torch.zeros(b, q, k, dtype=torch.float32),
    }
    batch = {
        "lanes": [lanes],
        "lane_valid": [lane_valid],
        "num_lanes": torch.tensor([4], dtype=torch.long),
    }
    count_idx = GCSLoss.loss_names.index("gt4_short_pos_count")

    model.train()
    _, train_items = model.loss(batch, preds)
    if float(train_items[count_idx]) <= 0.0:
        raise AssertionError("GCSLaneModel train-mode loss did not enable GT4 short point-valid rescue.")

    model.eval()
    _, eval_items = model.loss(batch, preds)
    if float(eval_items[count_idx]) != 0.0:
        raise AssertionError(
            "GCSLaneModel eval-mode loss reused a training-mode cached criterion; "
            f"gt4_short_pos_count={float(eval_items[count_idx])}."
        )


TESTS = (
    test_gt4_default_no_weight,
    test_gt4_visible_le_10_weights_visible_anchors_only,
    test_gt4_visible_11_no_weight,
    test_gt4_rescue_disabled_when_loss_eval_mode,
    test_gt5_old_behavior_visible_le_10_unchanged,
    test_gt3_no_weight_even_when_gt4_gt5_enabled,
    test_gt4_gt5_enabled_branch_by_gt_count,
    test_loss_log_item_names_and_forward_length_stable,
    test_short_point_valid_weights_reject_suppression_values,
    test_gcs_model_loss_syncs_cached_criterion_training_state,
)


def main() -> None:
    failures = []
    for test in TESTS:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report all contract failures in one run.
            failures.append({"test": test.__name__, "error": str(exc)})
    if failures:
        print(json.dumps({"status": "failed", "passed": len(TESTS) - len(failures), "failures": failures}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"status": "ok", "tests": len(TESTS)}, indent=2))


if __name__ == "__main__":
    main()
