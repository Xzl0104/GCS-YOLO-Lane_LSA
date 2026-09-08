from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tools.train_gcs import apply_dataset_profile, assert_dataset_profile_contract, parse_args
from tools.check_culane_dataset import validate as validate_culane_dataset
from ultralytics.data.dataset_gcs import GCSLaneDataset, gcs_collate_fn
from ultralytics.nn.modules import GCSLaneHead
from ultralytics.nn.tasks import GCSLaneModel
from ultralytics.utils.gcs_fixed_y import (
    build_fixed_y_anchors,
    validate_fixed_y_contract,
    validate_training_fixed_y_desc,
)
from ultralytics.utils.gcs_loss import GCSLoss
from ultralytics.utils.gcs_matcher import GCSHungarianMatcher


ROOT = Path(__file__).resolve().parents[1]
CULANE_BASE_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-culane.yaml"
CULANE_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-culane-count.yaml"
TUSIMPLE_COUNT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count.yaml"


def _profile_args(argv: list[str]):
    return apply_dataset_profile(parse_args(argv))


def _head_from_yaml(path: Path) -> GCSLaneHead:
    model = GCSLaneModel(str(path), nc=1, verbose=False)
    head = model.model[-1]
    assert isinstance(head, GCSLaneHead)
    return head.eval()


def _head_features(head: GCSLaneHead, batch: int = 2) -> list[torch.Tensor]:
    return [
        torch.randn(batch, int(head.c1), 32, 32),
        torch.randn(batch, int(head.c1), 16, 16),
        torch.randn(batch, int(head.c1), 8, 8),
        torch.randn(batch, int(head.c1), 4, 4),
    ]


def _culane_empty_label(path: Path) -> None:
    anchors = build_fixed_y_anchors(590, 589, 39, 56)
    np.savez_compressed(
        path,
        lanes=np.zeros((0, 56, 2), dtype=np.float32),
        lane_valid=np.zeros((0, 56), dtype=np.float32),
        semantic_mask=np.zeros((384, 960), dtype=np.uint8),
        edge_mask=np.zeros((384, 960), dtype=np.float32),
        num_lanes=np.asarray([0], dtype=np.int64),
        point_mode=np.asarray("fixed_y"),
        fixed_y=anchors,
        fixed_y_original_h=np.asarray([590], dtype=np.int64),
        fixed_y_start_px=np.asarray([589], dtype=np.float32),
        fixed_y_end_px=np.asarray([39], dtype=np.float32),
    )


def _culane_lane_label(path: Path) -> None:
    anchors = build_fixed_y_anchors(590, 589, 39, 56)
    lanes = np.zeros((1, 56, 2), dtype=np.float32)
    lane_valid = np.zeros((1, 56), dtype=np.float32)
    valid_indices = np.asarray([0, 1, 2, 10, 20], dtype=np.int64)
    lanes[0, :, 1] = anchors
    lanes[0, valid_indices, 0] = np.asarray([0.2, 0.25, 0.3, 0.4, 0.5], dtype=np.float32)
    lane_valid[0, valid_indices] = 1.0
    np.savez_compressed(
        path,
        lanes=lanes,
        lane_valid=lane_valid,
        semantic_mask=np.zeros((384, 960), dtype=np.uint8),
        edge_mask=np.zeros((384, 960), dtype=np.float32),
        num_lanes=np.asarray([1], dtype=np.int64),
        point_mode=np.asarray("fixed_y"),
        fixed_y=anchors,
        fixed_y_original_h=np.asarray([590], dtype=np.int64),
        fixed_y_start_px=np.asarray([589], dtype=np.float32),
        fixed_y_end_px=np.asarray([39], dtype=np.float32),
    )


def _culane_empty_preds(batch: int = 1) -> dict[str, torch.Tensor]:
    k, q = 56, 12
    anchors = torch.linspace(589.0 / 590.0, 39.0 / 590.0, k).view(1, 1, k).expand(batch, q, k)
    xs = torch.full((batch, q, k), 0.5)
    return {
        "pred_points": torch.stack((xs, anchors), dim=-1).contiguous(),
        "pred_logits": torch.zeros(batch, q),
        "pred_valid_logits": torch.zeros(batch, q, k),
    }


def test_fixed_y_contracts_are_dataset_specific() -> None:
    culane = build_fixed_y_anchors(590, 589, 39, 56)
    tusimple = build_fixed_y_anchors(720, 710, 160, 56)

    assert validate_fixed_y_contract(culane, original_h=590, start_px=589, end_px=39, k=56) == "desc"
    assert validate_training_fixed_y_desc(tusimple) == "desc"
    with pytest.raises(ValueError, match="fixed-y contract violated"):
        validate_training_fixed_y_desc(culane)


def test_culane_profile_sets_0_to_4_lane_contract() -> None:
    args = _profile_args(["--dataset", "culane"])

    assert args.imgsz == [384, 960]
    assert args.gcs_mode == "query"
    assert (args.gcs_min_lanes, args.gcs_max_lanes, args.gcs_count_classes) == (0, 4, 5)
    assert (args.gcs_query_count_min_lanes, args.gcs_query_count_max_lanes) == (0, 4)
    assert args.gcs_culane_val is True
    assert_dataset_profile_contract(args)


def test_culane_profile_rejects_query_count_ce() -> None:
    with pytest.raises(ValueError, match="five-loss contract"):
        _profile_args(["--dataset", "culane", "--gcs-query-count-ce", "0.5"])


def test_culane_dataset_audit_allows_train_val_only_root(tmp_path: Path) -> None:
    """The final held-out test split is optional while train/val is being prepared."""
    report = validate_culane_dataset(
        SimpleNamespace(
            dataset_root=str(tmp_path),
            archive_root=None,
            source_lines_root=None,
            imgsz=(384, 960),
            num_points=56,
            max_files=0,
        )
    )

    assert report["error_count"] == 0
    assert report["test_category_audit"] == "skipped_no_test_labels"
    assert report["missing_test_categories"] == []


@pytest.mark.parametrize(
    "argv",
    [
        ["--dataset", "culane", "--gcs-mode", "ordered_slot"],
        [
            "--dataset",
            "culane",
            "--gcs-min-lanes",
            "2",
            "--gcs-max-lanes",
            "5",
            "--gcs-count-classes",
            "4",
        ],
        ["--dataset", "culane", "--gcs-culane-val", "false"],
    ],
)
def test_culane_profile_rejects_mixed_tusimple_arguments(argv: list[str]) -> None:
    with pytest.raises(ValueError, match="CULane"):
        assert_dataset_profile_contract(_profile_args(argv))


@torch.inference_mode()
def test_query_head_outputs_follow_dataset_contract() -> None:
    culane_base_head = _head_from_yaml(CULANE_BASE_CFG)
    culane_out = culane_base_head(_head_features(culane_base_head), orig_size=(384, 960))
    assert tuple(culane_out["pred_points"].shape) == (2, 12, 56, 2)
    assert "pred_count_logits" not in culane_out
    assert "aux_mask_logits" not in culane_out
    assert "aux_edge_logits" not in culane_out
    assert culane_base_head.aux is False
    assert culane_base_head.query_count_head is False
    assert culane_base_head.geometry_aware_exist is True
    assert culane_base_head.two_stage_refine is True
    assert culane_base_head.gated_multiscale is True
    assert culane_base_head.proposal_state_refine is True

    culane_head = _head_from_yaml(CULANE_CFG)
    culane_out = culane_head(_head_features(culane_head), orig_size=(384, 960))
    assert tuple(culane_out["pred_points"].shape) == (2, 12, 56, 2)
    assert tuple(culane_out["pred_count_logits"].shape) == (2, 5)
    assert "aux_mask_logits" not in culane_out
    assert "aux_edge_logits" not in culane_out
    assert culane_head.aux is False
    assert culane_head.geometry_aware_exist is True
    assert culane_head.two_stage_refine is True
    assert culane_head.gated_multiscale is True
    assert culane_head.proposal_state_refine is True

    tusimple_head = _head_from_yaml(TUSIMPLE_COUNT_CFG)
    tusimple_out = tusimple_head(_head_features(tusimple_head), orig_size=(544, 960))
    assert tuple(tusimple_out["pred_points"].shape) == (2, 12, 56, 2)
    assert tuple(tusimple_out["pred_count_logits"].shape) == (2, 4)


def test_culane_zero_lane_dataset_collate_matcher_and_loss_are_finite(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "val"
    label_dir = tmp_path / "labels_gcs" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image_path = image_dir / "empty.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((384, 960, 3), dtype=np.uint8))
    _culane_empty_label(label_dir / "empty.npz")

    dataset = GCSLaneDataset(image_dir=image_dir, label_dir=label_dir, imgsz=(384, 960), strict=True)
    sample = dataset[0]
    assert int(sample["num_lanes"]) == 0
    assert tuple(sample["lanes"].shape) == (0, 56, 2)

    batch = gcs_collate_fn([sample])
    assert batch["batch_idx"].dtype == torch.long
    assert batch["batch_idx"].numel() == 0

    matcher = GCSHungarianMatcher(image_size=(384, 960))
    pred_points = _culane_empty_preds()["pred_points"]
    pred_logits = _culane_empty_preds()["pred_logits"]
    pred_idx, target_idx = matcher(pred_points, pred_logits, batch["lanes"], batch["lane_valid"])[0]
    assert pred_idx.dtype == torch.long
    assert target_idx.dtype == torch.long
    assert pred_idx.numel() == target_idx.numel() == 0

    criterion = GCSLoss({"gcs_imgsz": [384, 960]})
    total, items = criterion(_culane_empty_preds(), batch)
    assert torch.isfinite(total)
    assert int(items.numel()) == len(GCSLoss.loss_names)
    assert torch.isfinite(items).all()


def test_culane_fixed_y_horizontal_flip_preserves_invalid_x_sentinel(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels_gcs" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image_path = image_dir / "lane.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((384, 960, 3), dtype=np.uint8))
    _culane_lane_label(label_dir / "lane.npz")

    dataset = GCSLaneDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        imgsz=(384, 960),
        strict=True,
        augment=True,
        fliplr=1.0,
    )
    lanes, lane_valid, semantic_mask, edge_mask = dataset._load_label(label_dir / "lane.npz")
    flipped_img, flipped_lanes, flipped_valid, _, _ = dataset._apply_geometric_augment(
        np.zeros((384, 960, 3), dtype=np.uint8),
        lanes,
        lane_valid,
        semantic_mask,
        edge_mask,
    )

    assert flipped_img.shape == (384, 960, 3)
    assert flipped_lanes.shape == lanes.shape
    assert np.array_equal(flipped_valid, lane_valid)
    assert np.allclose(flipped_lanes[..., 1], lanes[..., 1])
    assert np.allclose(flipped_lanes[flipped_valid > 0.5, 0], 1.0 - lanes[flipped_valid > 0.5, 0])
    assert np.allclose(flipped_lanes[flipped_valid <= 0.5, 0], 0.0)
    assert int((flipped_valid > 0.5).sum()) == int((lane_valid > 0.5).sum())
    GCSLaneDataset._normalize_lanes(
        flipped_lanes,
        flipped_valid,
        Path("<flip-test>"),
        point_mode="fixed_y",
        fixed_y_contract=dataset.fixed_y_contract,
    )


def test_explicit_fixed_y_contract_skips_dataset_wide_metadata_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images" / "val"
    label_dir = tmp_path / "labels_gcs" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "empty.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((384, 960, 3), dtype=np.uint8))
    _culane_empty_label(label_dir / "empty.npz")
    contract = {
        "fixed_y": build_fixed_y_anchors(590, 589, 39, 56),
        "fixed_y_original_h": 590,
        "fixed_y_start_px": 589,
        "fixed_y_end_px": 39,
        "num_points": 56,
    }

    def fail_scan(*args, **kwargs):
        raise AssertionError("explicit contract must not scan all label files")

    monkeypatch.setattr(GCSLaneDataset, "_detect_point_mode", fail_scan)
    monkeypatch.setattr(GCSLaneDataset, "_detect_fixed_y_contract", fail_scan)
    dataset = GCSLaneDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        imgsz=(384, 960),
        strict=True,
        point_mode="fixed_y",
        fixed_y_contract=contract,
    )
    assert int(dataset[0]["num_lanes"]) == 0


def test_explicit_fixed_y_contract_rejects_sample_metadata_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "val"
    label_dir = tmp_path / "labels_gcs" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "bad.jpg"
    label_path = label_dir / "bad.npz"
    assert cv2.imwrite(str(image_path), np.zeros((384, 960, 3), dtype=np.uint8))
    _culane_empty_label(label_path)
    with np.load(label_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["fixed_y_start_px"] = np.asarray([588], dtype=np.float32)
    np.savez_compressed(label_path, **arrays)
    dataset = GCSLaneDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        imgsz=(384, 960),
        strict=True,
        point_mode="fixed_y",
        fixed_y_contract={
            "fixed_y": build_fixed_y_anchors(590, 589, 39, 56),
            "fixed_y_original_h": 590,
            "fixed_y_start_px": 589,
            "fixed_y_end_px": 39,
            "num_points": 56,
        },
    )
    with pytest.raises(ValueError, match="fixed-y metadata disagrees"):
        dataset[0]


def test_five_loss_ignores_pred_count_logits_pollution(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "val"
    label_dir = tmp_path / "labels_gcs" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "empty.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((384, 960, 3), dtype=np.uint8))
    _culane_empty_label(label_dir / "empty.npz")

    dataset = GCSLaneDataset(image_dir=image_dir, label_dir=label_dir, imgsz=(384, 960), strict=True)
    batch = gcs_collate_fn([dataset[0]])
    preds = _culane_empty_preds()
    preds["pred_logits"].requires_grad_(True)
    preds["pred_valid_logits"].requires_grad_(True)
    pred_count_logits = torch.zeros((1, 5), dtype=torch.float32, requires_grad=True)
    preds["pred_count_logits"] = pred_count_logits

    total, items = GCSLoss({"gcs_imgsz": [384, 960]})(preds, batch)
    total.backward()

    assert torch.isfinite(total)
    assert int(items.numel()) == 5
    assert pred_count_logits.grad is None


def test_missing_point_valid_logits_fails_fast(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "val"
    label_dir = tmp_path / "labels_gcs" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "empty.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((384, 960, 3), dtype=np.uint8))
    _culane_empty_label(label_dir / "empty.npz")

    batch = gcs_collate_fn([GCSLaneDataset(image_dir=image_dir, label_dir=label_dir, imgsz=(384, 960), strict=True)[0]])
    preds = _culane_empty_preds()
    preds.pop("pred_valid_logits")
    with pytest.raises(RuntimeError, match="pred_valid_logits"):
        GCSLoss({"gcs_imgsz": [384, 960]})(preds, batch)


def test_visible_line_iou_loss_zero_for_exact_and_increases_with_offset() -> None:
    criterion = GCSLoss({"gcs_imgsz": [384, 960], "gcs_line_iou_width_px": 18.0})
    gt_points = [torch.zeros((1, 4, 2), dtype=torch.float32)]
    gt_valid = [torch.ones((1, 4), dtype=torch.float32)]
    gt_points[0][0, :, 0] = 0.5
    gt_points[0][0, :, 1] = torch.linspace(589.0 / 590.0, 39.0 / 590.0, 4)
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    exact = gt_points[0].view(1, 1, 4, 2).clone()
    exact_loss = criterion.visible_line_iou_loss(exact, gt_points, gt_valid, indices)

    offset = exact.clone().detach()
    offset[..., 0] += 0.01
    offset.requires_grad_(True)
    offset_loss = criterion.visible_line_iou_loss(offset, gt_points, gt_valid, indices)
    offset_loss.backward()

    assert float(exact_loss.detach()) < 1e-6
    assert float(offset_loss.detach()) > float(exact_loss.detach())
    assert offset.grad is not None
    assert float(offset.grad.abs().sum()) > 0.0


def test_five_loss_single_lane_forward_backward() -> None:
    criterion = GCSLoss({"gcs_imgsz": [384, 960]})
    k, q = 4, 2
    anchors = torch.linspace(589.0 / 590.0, 39.0 / 590.0, k)
    gt_lane = torch.zeros((1, k, 2), dtype=torch.float32)
    gt_lane[0, :, 0] = torch.linspace(0.45, 0.55, k)
    gt_lane[0, :, 1] = anchors
    gt_valid = torch.ones((1, k), dtype=torch.float32)

    pred_points = torch.zeros((1, q, k, 2), dtype=torch.float32)
    pred_points[0, 0] = gt_lane[0]
    pred_points[0, 0, :, 0] += 0.005
    pred_points[0, 1, :, 0] = 0.1
    pred_points[0, :, :, 1] = anchors.view(1, k)
    pred_points.requires_grad_(True)
    preds = {
        "pred_points": pred_points,
        "pred_logits": torch.zeros((1, q), dtype=torch.float32, requires_grad=True),
        "pred_valid_logits": torch.zeros((1, q, k), dtype=torch.float32, requires_grad=True),
    }
    batch = {"lanes": [gt_lane], "lane_valid": [gt_valid]}

    total, items = criterion(preds, batch)
    total.backward()

    assert torch.isfinite(total)
    assert int(items.numel()) == 5
    assert preds["pred_logits"].grad is not None
    assert preds["pred_valid_logits"].grad is not None
    assert pred_points.grad is not None
    assert float(pred_points.grad.abs().sum()) > 0.0


def test_visible_line_iou_loss_no_match_is_finite_zero() -> None:
    criterion = GCSLoss({"gcs_imgsz": [384, 960]})
    pred_points = torch.zeros((1, 1, 4, 2), dtype=torch.float32, requires_grad=True)
    gt_points = [torch.zeros((1, 4, 2), dtype=torch.float32)]
    gt_valid = [torch.ones((1, 4), dtype=torch.float32)]
    indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]

    loss = criterion.visible_line_iou_loss(pred_points, gt_points, gt_valid, indices)

    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
