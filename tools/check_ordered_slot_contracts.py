from __future__ import annotations

import argparse
import ast
import csv
import inspect
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.eval_gcs import build_eval_config, resolve_decode_mode as eval_resolve_decode_mode  # noqa: E402
from tools.train_gcs import dataset_defaults, maybe_switch_ordered_slot_model, parse_args as parse_train_gcs_args  # noqa: E402
from tools.eval_tusimple_official import (  # noqa: E402
    ORDERED_SLOT_QUERY_ONLY_DEFAULTS,
    evaluate_official,
    resolve_pred_json_decode_contract,
    resolve_decode_mode as official_resolve_decode_mode,
)
from tools.visualize_ordered_slot_targets import _save_rgb_image  # noqa: E402
from tools.sweep_gcs_conf import assert_legacy_query_conf_sweep_model, assert_legacy_query_conf_sweep_split  # noqa: E402
from tools.strip_gcs_head_ckpt import is_gcs_head_key  # noqa: E402
from tools.sweep_tusimple_official import build_combos, select_best, write_csv as write_official_sweep_csv  # noqa: E402
from tools.overfit_ordered_slot_20 import OVERFIT_THRESHOLDS, assert_overfit_passed, overfit_failures  # noqa: E402
from gcs_tools.official_selection import (  # noqa: E402
    OFFICIAL_BEST_SELECTION_KEYS,
    OFFICIAL_SELECTION_POLICY,
    SWEEP_SELECTION_KEYS,
    official_best_sort_key,
    official_best_selection_policy,
    sweep_selection_policy,
    sweep_sort_key,
)
from gcs_tools.tusimple_official_eval import (  # noqa: E402
    gcs_lanes_to_tusimple_lanes,
    official_gt_contract_summary,
    stable_gt_content_hash,
    stable_raw_file_hash,
    validate_canonical_val_gt,
)
from tools.infer_gcs import resolve_decode_mode as infer_resolve_decode_mode  # noqa: E402
from ultralytics.models.gcs.decode_summary import (  # noqa: E402
    build_ordered_slot_decode_summary,
    build_official_best_decode_cfg,
    ordered_slot_decode_runtime_config,
    ordered_slot_effective_decode,
    ordered_slot_order_diagnostics_summary,
    raise_for_ordered_slot_query_args,
    validate_decode_yaml_for_model,
)
from ultralytics.models.gcs.decode_ordered_slot import (  # noqa: E402
    decode_ordered_slot_predictions,
    decoded_bottom_idx_from_start_end_logits,
    validate_ordered_slot_pred_shapes,
)
from ultralytics.utils.gcs_fixed_y import (  # noqa: E402
    validate_fixed_y_anchors,
    validate_official_h_samples_asc,
    validate_tusimple_h_samples_asc,
    validate_training_fixed_y_desc,
)
from ultralytics.models.gcs.loss_ordered_slot import OrderedSlotGCSLoss  # noqa: E402
from ultralytics.models.gcs.slot_targets import build_ordered_lane_slots_single  # noqa: E402
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.models.yolo.gcs_lane.val import GCSLaneValidator  # noqa: E402
from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel  # noqa: E402


ORDERED_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q5-slot-k56.yaml"
TMP_DIR = ROOT / ".tmp" / "ordered_slot_contract_fix_v2"
TRACKED_FILES = (
    ".gitignore",
    "gcs_tools/__init__.py",
    "gcs_tools/canonical_tusimple_val_363_manifest.json",
    "gcs_tools/official_selection.py",
    "tools/audit_fixed_y_labels.py",
    "tools/check_ordered_slot_contracts.py",
    "tools/overfit_ordered_slot_20.py",
    "tools/strip_gcs_head_ckpt.py",
    "tools/train_gcs.py",
    "tools/visualize_ordered_slot_targets.py",
    "ultralytics/cfg/__init__.py",
    "ultralytics/cfg/default.yaml",
    "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q5-slot-k56.yaml",
    "ultralytics/models/gcs/__init__.py",
    "ultralytics/models/gcs/decode_ordered_slot.py",
    "ultralytics/models/gcs/decode_summary.py",
    "ultralytics/models/gcs/fixed_y.py",
    "ultralytics/utils/gcs_fixed_y.py",
    "ultralytics/models/gcs/loss_ordered_slot.py",
    "ultralytics/models/gcs/mode_utils.py",
    "ultralytics/models/gcs/slot_targets.py",
    "ultralytics/models/yolo/gcs_lane/train.py",
    "ultralytics/models/yolo/gcs_lane/val.py",
    "ultralytics/utils/gcs_point_loss.py",
)
REQUIRED_IMPORT_MODULES = (
    "gcs_tools.official_selection",
    "ultralytics.models.gcs.decode_summary",
    "ultralytics.utils.gcs_fixed_y",
    "ultralytics.utils.gcs_point_loss",
)


class DummyOrderedModel(nn.Module):
    """Small module carrying enough ordered-slot metadata for mode inference tests."""

    def __init__(self):
        super().__init__()
        self.gcs_mode = "ordered_slot"
        self.start_mlp = nn.Identity()
        self.end_mlp = nn.Identity()
        self.count_mlp = nn.Identity()


def _fresh_trainer() -> GCSLaneTrainer:
    return object.__new__(GCSLaneTrainer)


def _make_ordered_model() -> GCSLaneModel:
    return GCSLaneModel(str(ORDERED_CFG), nc=1, verbose=False)


def _reset_tmp_subdir(name: str) -> Path:
    path = TMP_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "test_canonical_val_manifest",
                "num_images": len(records),
                "raw_file_sha256": stable_raw_file_hash(records),
                "gt_content_sha256": stable_gt_content_hash(records),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _official_eval_args(
    tmp: Path,
    archive_root: Path,
    *,
    gt_json: Path | None,
    pred_json: Path | None = None,
    allow_noncanonical_gt: bool = False,
    decode_mode: str = "ordered_slot",
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="tusimple",
        archive_root=str(archive_root),
        split="val",
        gt_json=None if gt_json is None else str(gt_json),
        allow_noncanonical_gt=allow_noncanonical_gt,
        pred_json=None if pred_json is None else str(pred_json),
        weights=str(tmp / "unused.pt"),
        decode_mode=decode_mode,
        decode_yaml=None,
        imgsz=[544, 960],
        conf=0.25,
        point_valid_thr=0.5,
        nms_dist_px=18.0,
        min_points=6,
        max_det=8,
        count_aware_topk=False,
        count_aware_min_k=3,
        count_aware_max_k=5,
        count_aware_length_norm=12.0,
        max_images=0,
        warmup=0,
        device="cpu",
        half=False,
        runtime_ms=1.0,
        use_measured_runtime=False,
        save_dir=str(tmp / "out"),
        save_records=False,
        score_fp_weight=0.02,
        score_fn_weight=0.02,
    )


def _first_key(state: dict[str, torch.Tensor], marker: str) -> str:
    for key in state:
        if marker in key:
            return key
    raise AssertionError(f"No state_dict key contains {marker!r}.")


def _fixed_y_points(num_lanes: int, k: int = 56) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    points = torch.zeros(num_lanes, k, 2)
    valid = torch.ones(num_lanes, k)
    for lane_idx in range(num_lanes):
        points[lane_idx, :, 0] = 0.15 + 0.15 * lane_idx
        points[lane_idx, :, 1] = y
    return points, valid


def _base_preds(batch: int = 1, slots: int = 5, k: int = 56) -> dict[str, torch.Tensor]:
    preds = {
        "pred_points": torch.zeros(batch, slots, k, 2),
        "pred_count_logits": torch.zeros(batch, 4),
        "pred_start_logits": torch.zeros(batch, slots, k),
        "pred_end_logits": torch.zeros(batch, slots, k),
        "pred_exist_logits": torch.ones(batch, slots),
        "pred_valid_logits": torch.zeros(batch, slots, k),
    }
    preds["pred_start_logits"][:, :, 0] = 10.0
    preds["pred_end_logits"][:, :, 0] = 10.0
    return preds


def test_two_lane_target_and_count_acc() -> None:
    points, valid = _fixed_y_points(2)
    targets = build_ordered_lane_slots_single(points, valid, min_lanes=2, max_lanes=5)
    assert targets["slot_exist"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert int(targets["count_label"].item()) == 0

    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960], "gcs_count_ce": 1.0})
    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    batch = {"lanes": [points], "lane_valid": [valid], "img": torch.zeros(1, 3, 544, 960)}
    _, loss_items = criterion(preds, batch)
    idx = OrderedSlotGCSLoss.loss_names.index("slot_count_acc_2")
    assert float(loss_items[idx].item()) == 1.0


def test_slot_count_absent_class_logs_nan_and_totals() -> None:
    lanes = [_fixed_y_points(n) for n in (2, 3, 4)]
    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960], "gcs_count_ce": 1.0})
    preds = _base_preds(batch=3)
    for batch_idx, count_cls in enumerate((0, 1, 2)):
        preds["pred_count_logits"][batch_idx, count_cls] = 10.0
    batch = {
        "lanes": [points for points, _ in lanes],
        "lane_valid": [valid for _, valid in lanes],
        "img": torch.zeros(3, 3, 544, 960),
    }

    _, loss_items = criterion(preds, batch)
    names = OrderedSlotGCSLoss.loss_names
    total5 = loss_items[names.index("slot_count_total_5")]
    acc5 = loss_items[names.index("slot_count_acc_5")]
    assert float(total5.item()) == 0.0
    assert torch.isnan(acc5)

    logged = OrderedSlotGCSLoss.label_loss_items(loss_items, prefix="train")
    assert math.isnan(float(logged["train/slot_count_acc_5"]))
    assert float(logged["train/slot_count_total_5"]) == 0.0
    assert float(logged["train/slot_count_acc"]) == 1.0
    assert GCSLaneTrainer._format_progress_cell(float(logged["train/slot_count_acc_5"])).strip() == "NA"
    assert GCSLaneTrainer._format_metric_csv_value(logged["train/slot_count_acc_5"]) == "NA"


def test_slot_count_acc_aggregates_from_correct_total_and_skips_absent() -> None:
    names = OrderedSlotGCSLoss.loss_names
    values = torch.zeros(len(names))
    values[names.index("slot_count_acc_2")] = float("nan")
    values[names.index("slot_count_acc_3")] = float("nan")
    values[names.index("slot_count_acc_4")] = float("nan")
    values[names.index("slot_count_acc_5")] = float("nan")
    for count, correct, total in ((2, 2.0, 2.0), (3, 1.0, 2.0), (4, 3.0, 4.0), (5, 0.0, 0.0)):
        values[names.index(f"slot_count_correct_{count}")] = correct
        values[names.index(f"slot_count_total_{count}")] = total

    logged = OrderedSlotGCSLoss.label_loss_items(values, prefix="val")
    assert float(logged["val/slot_count_acc_2"]) == 1.0
    assert float(logged["val/slot_count_acc_3"]) == 0.5
    assert float(logged["val/slot_count_acc_4"]) == 0.75
    assert math.isnan(float(logged["val/slot_count_acc_5"]))
    assert float(logged["val/slot_count_acc"]) == 0.75


def test_ordered_slot_loss_defaults_match_default_yaml_and_do_not_silent_disable() -> None:
    cfg = yaml.safe_load((ROOT / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8"))
    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960]})

    assert cfg["gcs_count_ce"] == 1.0
    assert cfg["gcs_interval"] == 1.0
    assert cfg["gcs_order"] == 0.2
    assert cfg["gcs_gt_bottom_order"] == 1.0
    assert cfg["gcs_decoded_bottom_order"] == 1.0
    assert cfg["gcs_slot_gt_bottom_x"] == 0.0
    assert cfg["gcs_slot_gt_bottom_x_beta"] == 0.05
    assert cfg["gcs_slot_gt_bottom_x_detach_interval"] == 1
    assert cfg["gcs_slot_gt_bottom_x_soft"] == 0.0
    assert cfg["gcs_slot_gt_bottom_x_soft_tau"] == 0.5
    assert cfg["gcs_slot_gt_bottom_x_soft_beta"] == 0.05
    assert cfg["gcs_slot_start_index_l1"] == 0.0
    assert cfg["gcs_slot_start_index_l1_beta"] == 2.0
    assert cfg["gcs_bottom_order_margin_px"] == 2.0
    assert cfg["gcs_min_interval_points"] == 2
    assert cfg["gcs_ordered_point_loss"] == "normalized_smooth_l1"
    assert criterion.count_ce_gain == cfg["gcs_count_ce"]
    assert criterion.interval_gain == cfg["gcs_interval"]
    assert criterion.order_gain == cfg["gcs_order"]
    assert criterion.gt_bottom_order_gain == cfg["gcs_gt_bottom_order"]
    assert criterion.decoded_bottom_order_gain == cfg["gcs_decoded_bottom_order"]
    assert criterion.slot_gt_bottom_x_gain == cfg["gcs_slot_gt_bottom_x"]
    assert criterion.slot_gt_bottom_x_beta == cfg["gcs_slot_gt_bottom_x_beta"]
    assert criterion.slot_gt_bottom_x_detach_interval is True
    assert criterion.slot_gt_bottom_x_soft_gain == cfg["gcs_slot_gt_bottom_x_soft"]
    assert criterion.slot_gt_bottom_x_soft_tau == cfg["gcs_slot_gt_bottom_x_soft_tau"]
    assert criterion.slot_gt_bottom_x_soft_beta == cfg["gcs_slot_gt_bottom_x_soft_beta"]
    assert criterion.slot_start_index_l1_gain == cfg["gcs_slot_start_index_l1"]
    assert criterion.slot_start_index_l1_beta == cfg["gcs_slot_start_index_l1_beta"]
    assert criterion.bottom_order_margin_px == cfg["gcs_bottom_order_margin_px"]
    assert criterion.min_interval_points == cfg["gcs_min_interval_points"]
    assert criterion.loss_contract_summary()["gcs_ordered_point_loss"] == "normalized_smooth_l1"
    assert criterion.count_ce_gain > 0.0
    assert criterion.interval_gain > 0.0
    assert criterion.order_gain > 0.0
    assert criterion.gt_bottom_order_gain > 0.0
    assert criterion.decoded_bottom_order_gain > 0.0
    assert criterion.loss_contract_summary()["slot_gt_bottom_x_supervision_enabled"] is False
    assert criterion.loss_contract_summary()["slot_gt_bottom_x_soft_supervision_enabled"] is False
    assert criterion.loss_contract_summary()["slot_start_index_l1_supervision_enabled"] is False


def test_ordered_slot_loss_core_supervision_requires_explicit_ablation() -> None:
    for key in ("gcs_count_ce", "gcs_interval"):
        try:
            OrderedSlotGCSLoss({"gcs_imgsz": [544, 960], key: 0.0})
        except RuntimeError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f"ordered_slot must fail when {key}=0.0.")

    try:
        OrderedSlotGCSLoss({"gcs_imgsz": [544, 960], "gcs_order": 0.0})
    except RuntimeError as exc:
        assert "--gcs-allow-disable-order-loss" in str(exc)
    else:
        raise AssertionError("ordered_slot order loss cannot be disabled without explicit ablation flag.")

    for key in ("gcs_gt_bottom_order", "gcs_decoded_bottom_order"):
        try:
            OrderedSlotGCSLoss({"gcs_imgsz": [544, 960], key: 0.0})
        except RuntimeError as exc:
            assert key in str(exc)
            assert "--gcs-allow-disable-order-loss" in str(exc)
        else:
            raise AssertionError(f"ordered_slot {key} cannot be disabled without explicit ablation flag.")

    criterion = OrderedSlotGCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_order": 0.0,
            "gcs_gt_bottom_order": 0.0,
            "gcs_decoded_bottom_order": 0.0,
            "gcs_allow_disable_order_loss": True,
        }
    )
    assert criterion.order_gain == 0.0
    assert criterion.gt_bottom_order_gain == 0.0
    assert criterion.decoded_bottom_order_gain == 0.0
    assert criterion.loss_contract_summary()["order_supervision_enabled"] is False
    assert criterion.loss_contract_summary()["gt_bottom_order_supervision_enabled"] is False
    assert criterion.loss_contract_summary()["decoded_bottom_order_supervision_enabled"] is False


def test_trainer_ordered_slot_loss_contract_rejects_disabled_bottom_order_losses() -> None:
    for key in ("gcs_gt_bottom_order", "gcs_decoded_bottom_order"):
        trainer = _fresh_trainer()
        trainer.args = SimpleNamespace(
            gcs_count_ce=1.0,
            gcs_interval=1.0,
            gcs_order=0.2,
            gcs_gt_bottom_order=1.0,
            gcs_decoded_bottom_order=1.0,
            gcs_min_interval_points=2,
            gcs_bottom_order_margin_px=2.0,
            gcs_allow_disable_order_loss=False,
            gcs_ordered_point_loss="normalized_smooth_l1",
        )
        setattr(trainer.args, key, 0.0)
        try:
            trainer._ordered_slot_loss_contract_from_args()
        except RuntimeError as exc:
            assert key in str(exc)
            assert "--gcs-allow-disable-order-loss" in str(exc)
        else:
            raise AssertionError(f"trainer contract must reject {key}=0.0 without explicit ablation flag.")

    trainer = _fresh_trainer()
    trainer.args = SimpleNamespace(
        gcs_count_ce=1.0,
        gcs_interval=1.0,
        gcs_order=0.2,
        gcs_gt_bottom_order=0.0,
        gcs_decoded_bottom_order=0.0,
        gcs_min_interval_points=2,
        gcs_bottom_order_margin_px=2.0,
        gcs_allow_disable_order_loss=True,
        gcs_ordered_point_loss="normalized_smooth_l1",
    )
    contract = trainer._ordered_slot_loss_contract_from_args()
    assert contract["gt_bottom_order_supervision_enabled"] is False
    assert contract["decoded_bottom_order_supervision_enabled"] is False
    assert contract["gcs_slot_gt_bottom_x"] == 0.0
    assert contract["gcs_slot_gt_bottom_x_beta"] == 0.05
    assert contract["gcs_slot_gt_bottom_x_detach_interval"] is True
    assert contract["gcs_slot_gt_bottom_x_soft"] == 0.0
    assert contract["gcs_slot_gt_bottom_x_soft_tau"] == 0.5
    assert contract["gcs_slot_gt_bottom_x_soft_beta"] == 0.05
    assert contract["gcs_slot_start_index_l1"] == 0.0
    assert contract["gcs_slot_start_index_l1_beta"] == 2.0
    assert contract["slot_gt_bottom_x_supervision_enabled"] is False
    assert contract["slot_gt_bottom_x_soft_supervision_enabled"] is False
    assert contract["slot_start_index_l1_supervision_enabled"] is False
    assert contract["gcs_ordered_point_loss"] == "normalized_smooth_l1"


def test_ordered_slot_validator_loss_gains_include_bottom_x_terms() -> None:
    validator = object.__new__(GCSLaneValidator)
    validator.args = SimpleNamespace(
        gcs_mode="ordered_slot",
        gcs_point=11.0,
        gcs_exist=2.0,
        gcs_count_ce=3.0,
        gcs_interval=4.0,
        gcs_point_valid=5.0,
        gcs_order=6.0,
        gcs_gt_bottom_order=7.0,
        gcs_decoded_bottom_order=8.0,
        gcs_slot_gt_bottom_x=0.25,
        gcs_slot_gt_bottom_x_soft=0.5,
        gcs_slot_start_index_l1=0.75,
    )
    gains = validator._loss_gains(torch.device("cpu"))
    names = OrderedSlotGCSLoss.loss_names

    assert float(gains[names.index("slot_gt_bottom_x_loss")].item()) == 0.25
    assert float(gains[names.index("slot_gt_bottom_x_abs_err")].item()) == 0.0
    assert float(gains[names.index("slot_gt_bottom_x_abs_err_px")].item()) == 0.0
    assert float(gains[names.index("slot_gt_bottom_x_soft_loss")].item()) == 0.5
    assert float(gains[names.index("slot_gt_bottom_x_soft_abs_err_px")].item()) == 0.0
    assert float(gains[names.index("slot_start_index_l1_loss")].item()) == 0.75


def test_bottom_order_loss_catches_no_common_anchor_violation() -> None:
    k = 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    points = torch.zeros(2, k, 2)
    valid = torch.zeros(2, k)
    points[:, :, 1] = y
    points[0, :, 0] = 0.2
    points[1, :, 0] = 0.7
    valid[0, 0:3] = 1.0
    valid[1, 10:13] = 1.0

    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_points"][0, :, :, 1] = y.view(1, k)
    preds["pred_points"][0, 0, 0, 0] = 0.8
    preds["pred_points"][0, 1, 10, 0] = 0.3
    preds["pred_start_logits"].fill_(-10.0)
    preds["pred_end_logits"].fill_(-10.0)
    preds["pred_start_logits"][0, 0, 0] = 10.0
    preds["pred_end_logits"][0, 0, 2] = 10.0
    preds["pred_start_logits"][0, 1, 10] = 10.0
    preds["pred_end_logits"][0, 1, 12] = 10.0

    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960]})
    batch = {"lanes": [points], "lane_valid": [valid], "img": torch.zeros(1, 3, 544, 960)}
    _, loss_items = criterion(preds, batch)
    names = OrderedSlotGCSLoss.loss_names

    assert float(loss_items[names.index("slot_order_loss")].item()) == 0.0
    assert float(loss_items[names.index("slot_gt_bottom_order_loss")].item()) > 0.0
    assert float(loss_items[names.index("slot_decoded_bottom_order_loss")].item()) > 0.0
    assert float(loss_items[names.index("slot_order_common_violation_rate")].item()) == 0.0
    assert float(loss_items[names.index("slot_order_gt_bottom_violation_rate")].item()) == 1.0
    assert float(loss_items[names.index("slot_order_decoded_bottom_violation_rate")].item()) == 1.0
    assert float(loss_items[names.index("slot_order_decoded_bottom_pair_total")].item()) == 1.0
    assert float(loss_items[names.index("slot_order_decoded_bottom_violation_pairs")].item()) == 1.0


def test_decoded_bottom_idx_matches_repaired_interval_when_start_gt_end() -> None:
    b, slots, k = 1, 5, 56
    start_logits = torch.full((b, slots, k), -10.0)
    end_logits = torch.full((b, slots, k), -10.0)
    start_logits[0, 0, 30] = 10.0
    end_logits[0, 0, 10] = 10.0
    start_logits[0, 1, 20] = 10.0
    end_logits[0, 1, 40] = 10.0
    start_logits[0, 2, 30] = 10.0
    end_logits[0, 2, 30] = 10.0

    idx = decoded_bottom_idx_from_start_end_logits(start_logits, end_logits, min_interval_points=2)

    assert int(idx[0, 0].item()) == 10
    assert int(idx[0, 1].item()) == 20
    assert int(idx[0, 2].item()) == 29


def test_decoded_bottom_helper_sees_same_start_gt_end_violation_as_strict_decoder() -> None:
    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_start_logits"].fill_(-10.0)
    preds["pred_end_logits"].fill_(-10.0)
    preds["pred_start_logits"][0, 0, 30] = 10.0
    preds["pred_end_logits"][0, 0, 10] = 10.0
    preds["pred_start_logits"][0, 1, 20] = 10.0
    preds["pred_end_logits"][0, 1, 40] = 10.0
    preds["pred_points"][0, 0, 10, 0] = 0.80
    preds["pred_points"][0, 1, 20, 0] = 0.20

    try:
        decode_ordered_slot_predictions(preds, batch_index=0, output_order="slot", order_check="error")
    except AssertionError as exc:
        assert "ordered_slot order violation" in str(exc)
    else:
        raise AssertionError("strict ordered-slot decoder must reject repaired bottom-x reversal.")

    idx = decoded_bottom_idx_from_start_end_logits(
        preds["pred_start_logits"],
        preds["pred_end_logits"],
        min_interval_points=2,
    )
    bottom_x = preds["pred_points"][..., 0].gather(dim=2, index=idx.unsqueeze(-1)).squeeze(-1)
    assert abs(float(bottom_x[0, 0].item()) - 0.80) < 1e-6
    assert abs(float(bottom_x[0, 1].item()) - 0.20) < 1e-6
    assert float(bottom_x[0, 0].item()) > float(bottom_x[0, 1].item())


def test_decoded_bottom_loss_and_metric_use_repaired_start_end_interval() -> None:
    points, valid = _fixed_y_points(2)
    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_start_logits"].fill_(-10.0)
    preds["pred_end_logits"].fill_(-10.0)
    preds["pred_start_logits"][0, 0, 30] = 10.0
    preds["pred_end_logits"][0, 0, 10] = 10.0
    preds["pred_start_logits"][0, 1, 20] = 10.0
    preds["pred_end_logits"][0, 1, 40] = 10.0
    preds["pred_points"][0, 0, 10, 0] = 0.80
    preds["pred_points"][0, 0, 30, 0] = 0.10
    preds["pred_points"][0, 1, 20, 0] = 0.20

    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960]})
    batch = {"lanes": [points], "lane_valid": [valid], "img": torch.zeros(1, 3, 544, 960)}
    _, loss_items = criterion(preds, batch)
    names = OrderedSlotGCSLoss.loss_names

    assert float(loss_items[names.index("slot_decoded_bottom_order_loss")].item()) > 0.0
    assert float(loss_items[names.index("slot_order_decoded_bottom_violation_rate")].item()) == 1.0
    assert float(loss_items[names.index("slot_order_decoded_bottom_violation_pairs")].item()) == 1.0


def test_slot_gt_bottom_x_loss_uses_repaired_decoded_bottom_index() -> None:
    points, valid = _fixed_y_points(2)
    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_points"][0, 0, :, 0] = 0.15
    preds["pred_points"][0, 1, :, 0] = 0.30
    preds["pred_start_logits"].fill_(-10.0)
    preds["pred_end_logits"].fill_(-10.0)
    preds["pred_start_logits"][0, 0, 30] = 10.0
    preds["pred_end_logits"][0, 0, 10] = 10.0
    preds["pred_start_logits"][0, 1, 0] = 10.0
    preds["pred_end_logits"][0, 1, 2] = 10.0
    preds["pred_points"][0, 0, 10, 0] = 0.45
    preds["pred_points"][0, 0, 30, 0] = 0.15

    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960], "gcs_slot_gt_bottom_x": 1.0})
    batch = {"lanes": [points], "lane_valid": [valid], "img": torch.zeros(1, 3, 544, 960)}
    _, loss_items = criterion(preds, batch)
    names = OrderedSlotGCSLoss.loss_names

    assert float(loss_items[names.index("slot_gt_bottom_x_loss")].item()) > 0.10
    assert abs(float(loss_items[names.index("slot_gt_bottom_x_abs_err")].item()) - 0.15) < 1e-5
    assert abs(float(loss_items[names.index("slot_gt_bottom_x_abs_err_px")].item()) - 144.0) < 1e-4


def test_slot_gt_bottom_x_soft_loss_backprops_to_start_logits() -> None:
    criterion = OrderedSlotGCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_slot_gt_bottom_x_soft": 1.0,
            "gcs_slot_start_index_l1": 1.0,
        }
    )
    k = 56
    pred_points = torch.zeros(1, 5, k, 2)
    pred_points[..., 0] = torch.linspace(0.0, 1.0, k).view(1, 1, k)
    pred_start_logits = torch.zeros(1, 5, k, requires_grad=True)
    target_points = torch.zeros(1, 5, k, 2)
    target_start_labels = torch.zeros(1, 5, dtype=torch.long)
    slot_exist = torch.ones(1, 5)

    bottom_x_loss, bottom_x_abs_err_px, start_index_l1_loss = criterion._slot_gt_bottom_x_soft_loss(
        pred_points,
        pred_start_logits,
        target_points,
        target_start_labels,
        slot_exist,
    )
    total = bottom_x_loss + start_index_l1_loss
    total.backward()

    assert float(bottom_x_loss.item()) > 0.0
    assert float(bottom_x_abs_err_px.item()) > 0.0
    assert float(start_index_l1_loss.item()) > 0.0
    assert pred_start_logits.grad is not None
    assert float(pred_start_logits.grad.abs().sum().item()) > 0.0


def test_slot_gt_bottom_x_loss_reports_available_target_keys() -> None:
    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960], "gcs_slot_gt_bottom_x": 1.0})
    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0

    def broken_targets(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "slot_valid": torch.ones(1, 5, 56, device=device),
            "slot_exist": torch.ones(1, 5, device=device),
            "count_label": torch.zeros(1, dtype=torch.long, device=device),
            "start_labels": torch.zeros(1, 5, dtype=torch.long, device=device),
            "end_labels": torch.zeros(1, 5, dtype=torch.long, device=device),
            "repaired_noncontiguous_lanes": torch.zeros(1, device=device),
            "repaired_hole_points": torch.zeros(1, device=device),
        }

    criterion._target_slots = broken_targets
    try:
        criterion(preds, {"img": torch.zeros(1, 3, 544, 960)})
    except KeyError as exc:
        message = str(exc)
        assert "slot_gt_bottom_x_loss" in message
        assert "slot_points" in message
        assert "available keys" in message
        assert "slot_valid" in message
    else:
        raise AssertionError("slot_gt_bottom_x_loss must report available target keys when required targets are missing.")


def test_decode_count_two_lanes() -> None:
    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    lanes = decode_ordered_slot_predictions(preds, batch_index=0, min_interval_points=2)
    assert len(lanes) == 2


def test_ordered_slot_head_requires_five_slots_and_queries() -> None:
    for kwargs, expected in (
        ({"num_slots": 6, "num_queries": 6}, "num_slots=5"),
        ({"num_slots": 5, "num_queries": 6}, "num_queries=5"),
    ):
        try:
            GCSLaneHead(
                c1=16,
                num_points=56,
                num_decoder_layers=1,
                nhead=4,
                point_mode="fixed_y",
                gcs_mode="ordered_slot",
                min_lanes=2,
                max_lanes=5,
                count_classes=4,
                **kwargs,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("ordered_slot head must reject non-5 slot/query contracts.")


def test_ordered_slot_decode_shape_guard_rejects_bad_count_logits() -> None:
    preds = _base_preds()
    preds["pred_count_logits"] = torch.zeros(1, 5)
    try:
        decode_ordered_slot_predictions(preds, batch_index=0)
    except ValueError as exc:
        msg = str(exc)
        assert "pred_count_logits" in msg
        assert "B x 4" in msg
        assert "[2, 5]" in msg
    else:
        raise AssertionError("ordered_slot decoder must reject malformed pred_count_logits shape.")

    for key, value, expected in (
        ("pred_start_logits", torch.zeros(1, 5, 55), "pred_start_logits"),
        ("pred_end_logits", torch.zeros(1, 5, 55), "pred_end_logits"),
        ("pred_exist_logits", torch.zeros(1, 5, 1), "pred_exist_logits"),
    ):
        preds = _base_preds()
        preds[key] = value
        try:
            validate_ordered_slot_pred_shapes(preds, min_lanes=2, max_lanes=5)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"ordered_slot shape guard must reject malformed {key}.")

    preds = _base_preds(slots=4)
    try:
        validate_ordered_slot_pred_shapes(preds, min_lanes=2, max_lanes=5)
    except ValueError as exc:
        assert "S == max_lanes=5" in str(exc)
    else:
        raise AssertionError("ordered_slot shape guard must reject S < max_lanes.")

    preds = _base_preds(slots=6)
    try:
        validate_ordered_slot_pred_shapes(preds, min_lanes=2, max_lanes=5)
    except ValueError as exc:
        assert "S == max_lanes=5" in str(exc)
    else:
        raise AssertionError("ordered_slot shape guard must reject S > max_lanes.")

    preds = _base_preds()
    assert validate_ordered_slot_pred_shapes(preds, min_lanes=2, max_lanes=5) == (1, 5, 56)


def test_ordered_slot_decode_order_check_error() -> None:
    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_points"][0, 0, :, 0] = 0.8
    preds["pred_points"][0, 1, :, 0] = 0.3
    preds["pred_logits"] = preds["pred_exist_logits"]
    try:
        decode_ordered_slot_predictions(
            preds,
            batch_index=0,
            min_interval_points=2,
            order_check="error",
            output_order="slot",
        )
    except AssertionError as exc:
        assert "ordered_slot order violation" in str(exc)
    else:
        raise AssertionError("order_check=error must reject inverse slot order.")


def test_ordered_slot_decode_default_is_strict_slot_order() -> None:
    params = inspect.signature(decode_ordered_slot_predictions).parameters
    assert params["order_check"].default == "error"
    assert params["output_order"].default == "slot"

    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_points"][0, 0, :, 0] = 0.8
    preds["pred_points"][0, 1, :, 0] = 0.3
    try:
        decode_ordered_slot_predictions(preds, batch_index=0, min_interval_points=2)
    except AssertionError as exc:
        assert "ordered_slot order violation" in str(exc)
    else:
        raise AssertionError("decode_ordered_slot_predictions default must reject reversed ordered slots.")


def test_ordered_slot_decode_rejects_sort_with_error_policy() -> None:
    preds = _base_preds()
    try:
        decode_ordered_slot_predictions(preds, batch_index=0, output_order="left_to_right", order_check="error")
    except ValueError as exc:
        assert "output_order=left_to_right sorts lanes after decoding" in str(exc)
    else:
        raise AssertionError("left_to_right sorted export must not be combined with order_check=error.")


def test_official_ordered_slot_decode_reversed_slots_fail_fast() -> None:
    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_points"][0, 0, :, 0] = 0.8
    preds["pred_points"][0, 1, :, 0] = 0.3
    runtime_cfg = ordered_slot_decode_runtime_config(context="official_eval")

    try:
        decode_ordered_slot_predictions(
            preds,
            batch_index=0,
            min_interval_points=2,
            order_check=runtime_cfg["order_check"],
            output_order=runtime_cfg["output_order"],
        )
    except AssertionError as exc:
        assert "ordered_slot order violation" in str(exc)
    else:
        raise AssertionError("official ordered_slot decode must fail before runtime sorting can hide slot reversal.")


def test_ordered_slot_decode_left_to_right_sorts_and_preserves_slot() -> None:
    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_points"][0, 0, :, 0] = 0.8
    preds["pred_points"][0, 1, :, 0] = 0.3

    lanes, diagnostics = decode_ordered_slot_predictions(
        preds,
        batch_index=0,
        min_interval_points=2,
        order_check="warn",
        output_order="left_to_right",
        return_diagnostics=True,
    )

    assert [lane["slot"] for lane in lanes] == [1, 0]
    assert [lane["query"] for lane in lanes] == [1, 0]
    assert [round(float(lane["bottom_x_norm"]), 3) for lane in lanes] == [0.3, 0.8]
    assert diagnostics["order_violation_count"] == 1
    assert diagnostics["has_order_violation"] is True


def test_ordered_slot_decode_signature_has_no_exist_safety() -> None:
    params = inspect.signature(decode_ordered_slot_predictions).parameters
    assert "use_exist_safety" not in params
    assert "exist_thr" not in params


def test_decode_count_five_repairs_start_equals_end() -> None:
    preds = _base_preds()
    preds["pred_count_logits"][0, 3] = 10.0
    lanes = decode_ordered_slot_predictions(preds, batch_index=0, min_interval_points=2)
    assert len(lanes) == 5
    for lane in lanes:
        assert lane["end"] - lane["start"] + 1 >= 2


def test_decode_mode_mismatch_raises_everywhere() -> None:
    model = DummyOrderedModel()
    for resolver in (infer_resolve_decode_mode, eval_resolve_decode_mode, official_resolve_decode_mode):
        assert resolver("auto", model) == "ordered_slot"
        try:
            resolver("query", model)
        except RuntimeError as exc:
            assert "decode-mode mismatch" in str(exc)
        else:
            raise AssertionError(f"{resolver.__module__}.resolve_decode_mode should reject query for ordered model.")


def test_resume_disables_pretrained_and_loads_head_weights() -> None:
    source = _make_ordered_model()
    key = _first_key(source.state_dict(), ".count_mlp.")
    with torch.no_grad():
        dict(source.named_parameters())[key].fill_(0.375)
    ckpt_path = _reset_tmp_subdir("resume") / "ordered_resume.pt"
    source.args = {"gcs_mode": "ordered_slot", "data": "dummy.yaml", "gcs_imgsz": [544, 960], "pretrained": "yolo11s-seg.pt"}
    torch.save({"model": source, "train_args": source.args, "epoch": 0, "optimizer": None, "scaler": None}, ckpt_path)

    trainer = _fresh_trainer()
    trainer.model = str(ckpt_path)
    trainer.resume = True
    trainer.args = SimpleNamespace(
        resume=str(ckpt_path),
        pretrained="yolo11s-seg.pt",
        gcs_mode="ordered_slot",
        gcs_imgsz=[544, 960],
        gcs_allow_internal_best=True,
        imgsz=960,
        augmentations=None,
    )
    trainer.data = {"nc": 1, "channels": 3}
    trainer.save_dir = _reset_tmp_subdir("resume_args")

    trainer.setup_model()
    assert trainer.args.pretrained is False
    assert torch.allclose(trainer.model.state_dict()[key], source.state_dict()[key])


def test_noncontiguous_strict_and_repair_interp() -> None:
    points, valid = _fixed_y_points(2)
    valid[0, 10] = 0.0
    try:
        build_ordered_lane_slots_single(points, valid, min_lanes=2, max_lanes=5, contiguity_policy="strict")
    except ValueError as exc:
        assert "non-contiguous" in str(exc)
    else:
        raise AssertionError("strict contiguity policy should reject non-contiguous valid masks.")

    repaired = build_ordered_lane_slots_single(points, valid, min_lanes=2, max_lanes=5, contiguity_policy="repair_interp")
    assert int(repaired["repaired_noncontiguous_lanes"].item()) == 1
    assert int(repaired["repaired_hole_points"].item()) == 1
    assert float(repaired["slot_valid"][0, 10].item()) == 1.0


def test_fixed_y_wrong_k_fails_fast() -> None:
    bad = torch.linspace(710.0 / 720.0, 170.0 / 720.0, 55)
    try:
        validate_fixed_y_anchors(bad.numpy(), name="bad fixed_y")
    except ValueError as exc:
        assert "K mismatch" in str(exc)
    else:
        raise AssertionError("K55 fixed-y anchors should fail immediately.")


def test_ordered_point_loss_has_non_tiny_gradient_for_10px_error() -> None:
    b, slots, k = 1, 5, 56
    pred = torch.zeros(b, slots, k, 2, requires_grad=True)
    target = torch.zeros(b, slots, k, 2)
    with torch.no_grad():
        pred[..., 0] = 10.0 / 960.0
    slot_valid = torch.ones(b, slots, k)
    slot_exist = torch.ones(b, slots)

    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960], "gcs_ordered_point_loss": "aspect_l1"})
    loss = criterion._point_loss(pred, target, slot_valid, slot_exist)
    loss.backward()
    assert float(pred.grad.abs().mean().item()) > 1e-4


def test_ordered_point_loss_defaults_preserve_protocol_behavior() -> None:
    cfg = yaml.safe_load((ROOT / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8"))
    args = parse_train_gcs_args([])
    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960]})
    assert cfg["gcs_ordered_point_loss"] == "normalized_smooth_l1"
    assert cfg["gcs_slot_gt_bottom_x"] == 0.0
    assert cfg["gcs_slot_gt_bottom_x_beta"] == 0.05
    assert cfg["gcs_slot_gt_bottom_x_detach_interval"] == 1
    assert cfg["gcs_slot_gt_bottom_x_soft"] == 0.0
    assert cfg["gcs_slot_gt_bottom_x_soft_tau"] == 0.5
    assert cfg["gcs_slot_gt_bottom_x_soft_beta"] == 0.05
    assert cfg["gcs_slot_start_index_l1"] == 0.0
    assert cfg["gcs_slot_start_index_l1_beta"] == 2.0
    assert args.gcs_ordered_point_loss == "normalized_smooth_l1"
    assert args.gcs_slot_gt_bottom_x == 0.0
    assert args.gcs_slot_gt_bottom_x_beta == 0.05
    assert args.gcs_slot_gt_bottom_x_detach_interval == 1
    assert args.gcs_slot_gt_bottom_x_soft == 0.0
    assert args.gcs_slot_gt_bottom_x_soft_tau == 0.5
    assert args.gcs_slot_gt_bottom_x_soft_beta == 0.05
    assert args.gcs_slot_start_index_l1 == 0.0
    assert args.gcs_slot_start_index_l1_beta == 2.0
    assert criterion.ordered_point_loss == "normalized_smooth_l1"
    source = (ROOT / "tools/train_gcs.py").read_text(encoding="utf-8")
    assert '"gcs_slot_gt_bottom_x": args.gcs_slot_gt_bottom_x' in source
    assert '"gcs_slot_gt_bottom_x_beta": args.gcs_slot_gt_bottom_x_beta' in source
    assert '"gcs_slot_gt_bottom_x_detach_interval": args.gcs_slot_gt_bottom_x_detach_interval' in source
    assert '"gcs_slot_gt_bottom_x_soft": args.gcs_slot_gt_bottom_x_soft' in source
    assert '"gcs_slot_gt_bottom_x_soft_tau": args.gcs_slot_gt_bottom_x_soft_tau' in source
    assert '"gcs_slot_gt_bottom_x_soft_beta": args.gcs_slot_gt_bottom_x_soft_beta' in source
    assert '"gcs_slot_start_index_l1": args.gcs_slot_start_index_l1' in source
    assert '"gcs_slot_start_index_l1_beta": args.gcs_slot_start_index_l1_beta' in source


def test_aspect_l1_requires_explicit_ordered_point_loss_flag() -> None:
    args = parse_train_gcs_args(["--gcs-ordered-point-loss", "aspect_l1"])
    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960], "gcs_ordered_point_loss": args.gcs_ordered_point_loss})
    assert args.gcs_ordered_point_loss == "aspect_l1"
    assert criterion.ordered_point_loss == "aspect_l1"


def test_training_fixed_y_ascending_fails_and_official_ascending_passes() -> None:
    asc = torch.arange(160.0, 720.0, 10.0)
    try:
        validate_training_fixed_y_desc(asc.numpy(), name="ascending training fixed_y")
    except ValueError as exc:
        assert "Expected desc 710..160" in str(exc)
    else:
        raise AssertionError("Training fixed_y anchors in asc 160..710 order must fail.")

    assert validate_official_h_samples_asc(asc.numpy(), name="official h_samples") == "asc"


def test_tusimple_record_h_samples_accepts_canonical_subsets_for_export() -> None:
    partial = list(range(240, 720, 10))
    try:
        validate_official_h_samples_asc(partial, name="strict official h_samples")
    except ValueError as exc:
        assert "K mismatch" in str(exc)
    else:
        raise AssertionError("Strict K56 official h_samples validation must reject per-record K48 h_samples.")

    assert validate_tusimple_h_samples_asc(partial, name="record h_samples") == "asc"

    y_desc = torch.arange(710.0, 150.0, -10.0) / 720.0
    points = torch.zeros(56, 2)
    points[:, 0] = 0.5
    points[:, 1] = y_desc
    lane = {"points_norm": points.numpy(), "point_valid": np.ones((56,), dtype=np.float32)}
    converted = gcs_lanes_to_tusimple_lanes([lane], partial, image_shape=(720, 1280))
    assert len(converted) == 1
    assert len(converted[0]) == len(partial)
    assert all(x == 640 for x in converted[0])


def test_slot_target_removes_padding_before_fixed_y_validation() -> None:
    k = 56
    gt_points = torch.zeros(3, k, 2)
    gt_valid = torch.zeros(3, k)
    fixed_y = torch.arange(710.0, 150.0, -10.0) / 720.0
    for lane_idx in (1, 2):
        gt_points[lane_idx, :, 0] = 0.20 + 0.20 * lane_idx
        gt_points[lane_idx, :, 1] = fixed_y
        gt_valid[lane_idx, 10:40] = 1.0

    targets = build_ordered_lane_slots_single(gt_points, gt_valid, min_lanes=2, max_lanes=5)
    assert int(targets["num_lanes"].item()) == 2
    assert targets["slot_exist"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_ordered_slot_target_sorts_shuffled_gt_by_bottom_x() -> None:
    k = 56
    fixed_y = torch.arange(710.0, 150.0, -10.0) / 720.0
    gt_points = torch.zeros(3, k, 2)
    gt_valid = torch.zeros(3, k)
    # Input order is right, left, middle. Fixed-y is desc, so the bottom-most visible anchor is the lowest valid index.
    lane_specs = (
        (0.8, 4, 40),
        (0.2, 2, 30),
        (0.5, 6, 50),
    )
    for lane_idx, (bottom_x, start, end_exclusive) in enumerate(lane_specs):
        gt_points[lane_idx, :, 0] = bottom_x
        gt_points[lane_idx, :, 1] = fixed_y
        gt_valid[lane_idx, start:end_exclusive] = 1.0

    targets = build_ordered_lane_slots_single(gt_points, gt_valid, min_lanes=2, max_lanes=5)

    assert targets["sorted_gt_indices"][:3].tolist() == [1, 2, 0]
    bottom_x = [float(targets["slot_points"][slot, int(targets["start_labels"][slot]), 0]) for slot in range(3)]
    assert bottom_x == sorted(bottom_x)
    assert [round(x, 3) for x in bottom_x] == [0.2, 0.5, 0.8]
    assert targets["start_labels"][:3].tolist() == [2, 6, 4]
    assert targets["end_labels"][:3].tolist() == [29, 49, 39]


def test_ordered_slot_target_accepts_half_quantized_fixed_y() -> None:
    k = 56
    fixed_y = torch.arange(710.0, 150.0, -10.0).float() / 720.0
    gt_points = torch.zeros(2, k, 2, dtype=torch.float16)
    gt_valid = torch.zeros(2, k, dtype=torch.float32)
    for lane_idx in range(2):
        gt_points[lane_idx, :, 0] = 0.20 + 0.20 * lane_idx
        gt_points[lane_idx, :, 1] = fixed_y.half()
        gt_valid[lane_idx, 5:50] = 1.0

    targets = build_ordered_lane_slots_single(gt_points, gt_valid, min_lanes=2, max_lanes=5)
    assert int(targets["num_lanes"].item()) == 2
    assert targets["slot_points"].dtype == torch.float32
    assert torch.allclose(targets["slot_points"][0, :, 1], fixed_y, atol=1e-6)


def test_ordered_slot_loss_amp_half_preds_float32_gt_fixed_y() -> None:
    points, valid = _fixed_y_points(2)
    criterion = OrderedSlotGCSLoss({"gcs_imgsz": [544, 960], "gcs_count_ce": 1.0})
    preds = {key: value.half() for key, value in _base_preds().items()}
    preds["pred_count_logits"][0, 0] = 10.0
    batch = {"lanes": [points.float()], "lane_valid": [valid.float()], "img": torch.zeros(1, 3, 544, 960)}

    loss, loss_items = criterion(preds, batch)
    assert torch.isfinite(loss)
    for name, value in zip(OrderedSlotGCSLoss.loss_names, loss_items):
        if name.startswith("slot_count_acc_") and name != "slot_count_acc":
            continue
        assert torch.isfinite(value), name


def test_slot_target_training_fixed_y_ascending_fails() -> None:
    points, valid = _fixed_y_points(2)
    points[:, :, 1] = torch.arange(160.0, 720.0, 10.0) / 720.0
    try:
        build_ordered_lane_slots_single(points, valid, min_lanes=2, max_lanes=5)
    except ValueError as exc:
        assert "Expected desc 710..160" in str(exc)
    else:
        raise AssertionError("ordered_slot target construction must reject asc training fixed_y.")


def test_selection_policy_matches_sweep_sort_key() -> None:
    assert sweep_selection_policy()["ordered_keys"] == [dict(item) for item in SWEEP_SELECTION_KEYS]
    rows = [
        {
            "official_acc": 0.9,
            "official_score": 0.9,
            "official_FP": 0.1,
            "official_FN": 0.1,
            "count_acc_4": 0.6,
            "count_acc": 0.8,
            "count_acc_5": 0.8,
            "conf": 0.2,
            "nms_dist_px": 18.0,
            "point_valid_thr": 0.5,
            "max_det": 8,
            "min_points": 6,
        },
        {
            "official_acc": 0.9,
            "official_score": 0.9,
            "official_FP": 0.1,
            "official_FN": 0.1,
            "count_acc_4": 0.7,
            "count_acc": 0.7,
            "count_acc_5": 0.7,
            "conf": 0.25,
            "nms_dist_px": 50.0,
            "point_valid_thr": 0.55,
            "max_det": 5,
            "min_points": 4,
        },
    ]
    assert select_best(rows) == dict(max(rows, key=sweep_sort_key))


def test_official_selection_policy_matches_code_source() -> None:
    assert official_best_selection_policy() == OFFICIAL_SELECTION_POLICY
    assert OFFICIAL_SELECTION_POLICY["name"] == "official_best_v4"
    assert OFFICIAL_SELECTION_POLICY["ordered_keys"] == [dict(item) for item in OFFICIAL_BEST_SELECTION_KEYS]
    assert [item["key"] for item in OFFICIAL_SELECTION_POLICY["ordered_keys"]] == [
        "strict_order_valid",
        "ordered_slot_order_violations",
        "official_acc",
        "official_score",
        "official_FP",
        "official_FN",
        "count_acc_4",
        "count_acc",
        "count_acc_5",
        "epoch",
    ]
    assert [item["direction"] for item in OFFICIAL_SELECTION_POLICY["ordered_keys"]] == [
        "max",
        "min",
        "max",
        "max",
        "min",
        "min",
        "max",
        "max",
        "max",
        "earliest",
    ]


def _official_best_row(*, strict_order_valid: bool, violations: int, official_acc: float) -> dict:
    return {
        "strict_order_valid": strict_order_valid,
        "ordered_slot_order_violations": violations,
        "official_acc": official_acc,
        "official_score": official_acc,
        "official_FP": 0.02,
        "official_FN": 0.02,
        "count_acc_4": 0.9,
        "count_acc": 0.9,
        "count_acc_5": 0.9,
    }


def test_official_best_selection_prefers_strict_order_before_acc() -> None:
    valid_lower_acc = _official_best_row(strict_order_valid=True, violations=0, official_acc=0.90)
    invalid_higher_acc = _official_best_row(strict_order_valid=False, violations=0, official_acc=0.99)
    assert official_best_sort_key(valid_lower_acc, epoch=10) > official_best_sort_key(invalid_higher_acc, epoch=1)

    invalid_fewer_violations = _official_best_row(strict_order_valid=False, violations=2, official_acc=0.90)
    invalid_more_violations = _official_best_row(strict_order_valid=False, violations=3, official_acc=0.99)
    assert official_best_sort_key(invalid_fewer_violations, epoch=10) > official_best_sort_key(
        invalid_more_violations,
        epoch=1,
    )


def test_ordered_slot_summary_helpers_expose_schema_specific_decode() -> None:
    runtime_cfg = ordered_slot_decode_runtime_config(context="official_eval")
    effective = build_ordered_slot_decode_summary(
        min_lanes=2,
        max_lanes=5,
        num_slots=5,
        min_interval_points=2,
        order_margin_px=2.0,
        output_order=runtime_cfg["output_order"],
        order_check=runtime_cfg["order_check"],
    )
    decode_cfg = build_official_best_decode_cfg(
        {
            "decode_mode": "ordered_slot",
            "official_acc": 1.0,
            "official_score": 1.0,
            "official_FP": 0.0,
            "official_FN": 0.0,
        },
        model_mode="ordered_slot",
        args=SimpleNamespace(gcs_min_lanes=2, gcs_max_lanes=5, gcs_num_slots=5),
    )
    forbidden = {"conf", "point_valid_thr", "nms_dist_px", "max_det", "min_points", "count_aware_topk"}
    assert not (forbidden & set(decode_cfg))
    assert decode_cfg["schema"] == "ordered_slot_decode_v1"
    assert decode_cfg["decode_mode"] == "ordered_slot"
    assert decode_cfg["gcs_min_lanes"] == 2
    assert decode_cfg["gcs_max_lanes"] == 5
    assert decode_cfg["gcs_bottom_order_margin_px"] == 2.0
    assert effective["schema"] == "ordered_slot_decode_v1"
    assert effective["decode_mode"] == "ordered_slot"
    assert effective["count_source"] == "argmax(pred_count_logits)+gcs_min_lanes"
    assert effective["output_slots"] == "slot[0:num_lanes]"
    assert effective["gcs_bottom_order_margin_px"] == 2.0
    assert effective["output_order"] == "slot"
    assert effective["order_check"] == "error"
    assert effective["uses_runtime_sort"] is False
    assert runtime_cfg["uses_runtime_sort"] == effective["uses_runtime_sort"]
    assert effective["order_violation_policy"] == "fail_fast"
    assert effective["uses_nms"] is False
    assert effective["uses_conf_threshold"] is False
    assert effective["uses_max_det"] is False
    assert effective["query_decode_args"] == "not_applicable"
    assert ordered_slot_effective_decode() == build_ordered_slot_decode_summary()

    sorted_debug = build_ordered_slot_decode_summary(output_order="left_to_right", order_check="warn")
    assert sorted_debug["uses_runtime_sort"] is True
    assert sorted_debug["result_type"] == "postprocessed_sorted_export"
    assert sorted_debug["not_for_main_ordered_slot_claim"] is True

    warn_slot = build_ordered_slot_decode_summary(output_order="slot", order_check="warn")
    assert warn_slot["uses_runtime_sort"] is False
    assert warn_slot["order_violation_policy"] == "warn_only"
    assert warn_slot["result_type"] == "diagnostic_ordered_slot"
    assert warn_slot["not_for_main_ordered_slot_claim"] is True


def test_ordered_slot_runtime_config_contexts() -> None:
    strict_expected = {
        "order_check": "error",
        "output_order": "slot",
        "uses_runtime_sort": False,
        "order_violation_policy": "fail_fast",
        "result_type": "strict_ordered_slot",
        "not_for_main_ordered_slot_claim": False,
    }
    for context in ("official_eval", "official_sweep", "official_best", "eval_gcs", "val", "predict", "infer"):
        cfg = ordered_slot_decode_runtime_config(context=context)
        assert cfg == strict_expected

    training_official_expected = {
        "order_check": "warn",
        "output_order": "slot",
        "uses_runtime_sort": False,
        "order_violation_policy": "warn_only",
        "result_type": "training_official_best_candidate",
        "not_for_main_ordered_slot_claim": True,
    }
    assert ordered_slot_decode_runtime_config(context="training_official_best") == training_official_expected

    sorted_expected = {
        "order_check": "warn",
        "output_order": "left_to_right",
        "uses_runtime_sort": True,
        "order_violation_policy": "warn_then_sort_by_bottom_x",
        "result_type": "postprocessed_sorted_export",
        "not_for_main_ordered_slot_claim": True,
    }
    for context in ("debug_sorted_export", "debug", "visualize"):
        assert ordered_slot_decode_runtime_config(context=context) == {
            **sorted_expected,
        }
    internal_val_expected = {
        "order_check": "none",
        "output_order": "slot",
        "uses_runtime_sort": False,
        "order_violation_policy": "diagnostic_only",
        "result_type": "internal_training_val",
        "not_for_main_ordered_slot_claim": True,
    }
    for context in ("training_val", "internal_val"):
        assert ordered_slot_decode_runtime_config(context=context) == internal_val_expected
    assert ordered_slot_decode_runtime_config(context="overfit") == strict_expected
    try:
        ordered_slot_decode_runtime_config(context="unknown")
    except ValueError as exc:
        assert "Unknown ordered_slot decode context" in str(exc)
    else:
        raise AssertionError("unknown ordered_slot runtime context must fail instead of falling back to warn_only.")


def test_ordered_slot_order_diag_pred_json_not_available() -> None:
    diag = ordered_slot_order_diagnostics_summary(pred_json_mode=True, decode_stats=None)
    assert diag["ordered_slot_order_checked"] is False
    assert diag["ordered_slot_order_violations"] is None
    assert diag["ordered_slot_order_violation_images"] is None
    assert diag["ordered_slot_order_diagnostics"] == "not_available_from_pred_json"


def test_ordered_slot_order_diag_model_decode_available() -> None:
    diag = ordered_slot_order_diagnostics_summary(
        pred_json_mode=False,
        decode_stats={
            "ordered_slot_order_violations": 4,
            "ordered_slot_order_violation_images": 2,
        },
    )
    assert diag["ordered_slot_order_checked"] is True
    assert diag["ordered_slot_order_violations"] == 4
    assert diag["ordered_slot_order_violation_images"] == 2
    assert diag["ordered_slot_order_diagnostics"] == "available_from_decode"


def test_ordered_slot_official_eval_rejects_non_default_query_args() -> None:
    args = SimpleNamespace(**ORDERED_SLOT_QUERY_ONLY_DEFAULTS)
    args.conf = 0.123
    try:
        raise_for_ordered_slot_query_args(args, ORDERED_SLOT_QUERY_ONLY_DEFAULTS, context="test")
    except RuntimeError as exc:
        assert "ordered_slot decode does not use query decode args" in str(exc)
        assert "conf" in str(exc)
    else:
        raise AssertionError("ordered_slot official eval should reject non-default query-only args.")


def _training_official_sweep_args(mode: str, **overrides) -> argparse.Namespace:
    args = parse_train_gcs_args(["--gcs-mode", mode])
    args.gcs_imgsz = [544, 960]
    for key, value in overrides.items():
        setattr(args, key, value)
    trainer = GCSLaneTrainer.__new__(GCSLaneTrainer)
    trainer.args = args
    trainer.data = {}
    trainer.last = TMP_DIR / "weights" / "last.pt"
    trainer.device = torch.device("cpu")
    return trainer._official_sweep_args(TMP_DIR / "official_sweep")


def test_ordered_slot_training_official_sweep_rejects_query_only_args() -> None:
    for key, value in (
        ("gcs_official_confs", [0.777]),
        ("gcs_official_nms_dist_pxs", [77.0]),
    ):
        try:
            _training_official_sweep_args("ordered_slot", **{key: value})
        except RuntimeError as exc:
            assert "ordered_slot decode does not use query decode args" in str(exc)
            assert key in str(exc)
        else:
            raise AssertionError(f"ordered_slot training official_best should reject {key}.")

    args = _training_official_sweep_args("ordered_slot")
    assert args.confs == [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
    assert args.point_valid_thrs == [0.30, 0.35, 0.40, 0.45, 0.50]
    assert args.nms_dist_pxs == [18.0]
    assert args.max_dets == [8]
    assert args.min_points == [6]
    assert args.ordered_slot_runtime_context == "training_official_best"
    row = build_combos(args)[0]
    assert row["effective_decode"]["output_order"] == "slot"
    assert row["effective_decode"]["order_check"] == "warn"
    assert row["effective_decode"]["uses_runtime_sort"] is False
    assert row["effective_decode"]["not_for_main_ordered_slot_claim"] is True

    args = _training_official_sweep_args(
        "ordered_slot",
        gcs_min_interval_points=4,
        gcs_bottom_order_margin_px=7.5,
    )
    assert args.gcs_min_interval_points == 4
    assert args.gcs_bottom_order_margin_px == 7.5
    row = build_combos(args)[0]
    assert row["effective_decode"]["min_interval_points"] == 4
    assert row["effective_decode"]["gcs_bottom_order_margin_px"] == 7.5


def test_query_training_official_sweep_keeps_user_grid() -> None:
    args = _training_official_sweep_args(
        "query",
        gcs_official_confs=[0.777],
        gcs_official_nms_dist_pxs=[77.0],
    )
    assert args.confs == [0.777]
    assert args.nms_dist_pxs == [77.0]


def test_default_yaml_count_balanced_false() -> None:
    cfg = yaml.safe_load((ROOT / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8"))
    assert cfg["gcs_lane_count_balanced"] is False
    assert cfg["gcs_allow_internal_best"] is False


def test_train_gcs_count_balanced_default_false_and_explicit_true() -> None:
    args = parse_train_gcs_args([])
    assert args.gcs_lane_count_balanced is False
    args = parse_train_gcs_args(["--gcs-lane-count-balanced"])
    assert args.gcs_lane_count_balanced is True


def test_ordered_slot_without_official_best_requires_explicit_debug_escape() -> None:
    trainer = _fresh_trainer()
    trainer.args = SimpleNamespace(
        gcs_mode="ordered_slot",
        gcs_official_best=False,
        gcs_allow_internal_best=False,
    )
    trainer._warned_ordered_slot_without_official_best = False
    try:
        trainer._warn_if_ordered_slot_without_official_best()
    except RuntimeError as exc:
        assert "--gcs-official-best" in str(exc)
        assert "--gcs-allow-internal-best" in str(exc)
    else:
        raise AssertionError("ordered_slot formal training must require official_best checkpoint selection.")

    trainer.args.gcs_allow_internal_best = True
    trainer._warned_ordered_slot_without_official_best = False
    trainer._warn_if_ordered_slot_without_official_best()
    assert trainer._warned_ordered_slot_without_official_best is True


def test_train_gcs_defaults_match_current_contract_paths() -> None:
    args = parse_train_gcs_args([])
    defaults = dataset_defaults("tusimple")
    assert Path(args.model).as_posix() == "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"
    assert defaults["data"].as_posix() == "data/tusimple_gcs_fixed_y_960x544.yaml"


def test_ordered_slot_scale_positive_fails_fast() -> None:
    args = parse_train_gcs_args(["--gcs-mode", "ordered_slot", "--scale", "0.3"])
    try:
        maybe_switch_ordered_slot_model(args)
    except RuntimeError as exc:
        msg = str(exc)
        assert "--scale > 0" in msg
        assert "0/1" in msg
        assert "2/3/4/5 count contract" in msg
    else:
        raise AssertionError("ordered_slot with --scale 0.3 must fail fast until count-preserving scale exists.")


def test_query_mode_scale_behavior_is_unchanged() -> None:
    args = parse_train_gcs_args(["--gcs-mode", "query", "--scale", "0.3"])
    maybe_switch_ordered_slot_model(args)
    assert args.gcs_mode == "query"
    assert args.scale == 0.3


def test_ordered_slot_auto_switches_any_non_slot_model_yaml() -> None:
    args = SimpleNamespace(
        gcs_mode="ordered_slot",
        model="ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml",
        gcs_disable_auto_model_switch=False,
    )
    maybe_switch_ordered_slot_model(args)
    assert Path(args.model).name == "gcs-yolo-lane-s-q5-slot-k56.yaml"


def test_ordered_slot_auto_switch_can_be_disabled() -> None:
    args = SimpleNamespace(
        gcs_mode="ordered_slot",
        model="ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml",
        gcs_disable_auto_model_switch=True,
    )
    try:
        maybe_switch_ordered_slot_model(args)
    except RuntimeError as exc:
        assert "--gcs-mode ordered_slot requires an ordered_slot model yaml" in str(exc)
    else:
        raise AssertionError("--gcs-disable-auto-model-switch must fail on a non-slot model yaml.")


def test_generic_ordered_slot_query_yaml_mismatch_mentions_q5_slot_model() -> None:
    model = GCSLaneModel(str(ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"), nc=1, verbose=False)
    trainer = _fresh_trainer()
    trainer.args = SimpleNamespace(gcs_mode="ordered_slot", scale=0.0)
    try:
        trainer._assert_model_gcs_mode(model)
    except ValueError as exc:
        msg = str(exc)
        assert "generic YOLO entrypoint" in msg
        assert "model=ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q5-slot-k56.yaml" in msg
    else:
        raise AssertionError("generic ordered_slot run with query YAML must fail with q5-slot-k56.yaml guidance.")


def test_query_model_build_ignores_ordered_slot_overrides() -> None:
    cfg = yaml.safe_load((ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml").read_text(encoding="utf-8"))
    cfg.update(
        {
            "gcs_mode": "query",
            "gcs_num_slots": -1,
            "gcs_min_lanes": 5,
            "gcs_max_lanes": 2,
            "gcs_count_classes": 99,
        }
    )

    model = GCSLaneModel(cfg, nc=1, verbose=False)
    head = model.model[-1]
    assert isinstance(head, GCSLaneHead)
    assert head.gcs_mode == "query"
    assert not hasattr(head, "start_mlp")
    assert not hasattr(head, "end_mlp")
    assert not hasattr(head, "count_mlp")


def test_eval_gcs_ordered_summary_has_no_query_decode_keys() -> None:
    runtime_cfg = ordered_slot_decode_runtime_config(context="eval_gcs")
    config = build_eval_config(
        weights=ROOT / "dummy.pt",
        source=ROOT / "dummy_images",
        label_dir=None,
        imgsz=(544, 960),
        decode_mode="ordered_slot",
        conf=0.123,
        point_valid_thr=0.45,
        ape_thr=20.0,
        match_gate_px=None,
        max_x_dist=0.0,
        min_overlap=2,
        nms_dist_px=99.0,
        max_det=9,
        count_aware_topk=True,
        count_aware_min_k=2,
        count_aware_max_k=5,
        count_aware_length_norm=6.0,
        gcs_min_interval_points=4,
        gcs_bottom_order_margin_px=7.5,
        warmup=0,
        device="cpu",
        half=False,
    )
    forbidden = {"conf", "point_valid_thr", "nms_dist_px", "max_det", "min_points", "count_aware_topk"}
    assert config["schema"] == "ordered_slot_decode_v1"
    assert config["decode_mode"] == "ordered_slot"
    assert config["query_decode_args"] == "not_applicable"
    assert config["uses_nms"] is False
    assert config["uses_conf_threshold"] is False
    assert config["uses_max_det"] is False
    assert config["min_interval_points"] == 4
    assert config["gcs_bottom_order_margin_px"] == 7.5
    assert config["order_check"] == runtime_cfg["order_check"]
    assert config["output_order"] == runtime_cfg["output_order"]
    assert config["uses_runtime_sort"] == runtime_cfg["uses_runtime_sort"]
    assert not (forbidden & set(config))

    preds = _base_preds()
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_points"][0, 0, :, 0] = 0.8
    preds["pred_points"][0, 1, :, 0] = 0.3
    try:
        decode_ordered_slot_predictions(
            preds,
            batch_index=0,
            min_interval_points=2,
            order_check=runtime_cfg["order_check"],
            output_order=runtime_cfg["output_order"],
            return_diagnostics=True,
        )
    except AssertionError as exc:
        assert "ordered_slot order violation" in str(exc)
    else:
        raise AssertionError("eval_gcs ordered_slot strict slot output must fail on reversed slots.")

    debug_cfg = ordered_slot_decode_runtime_config(context="debug")
    lanes, diagnostics = decode_ordered_slot_predictions(
        preds,
        batch_index=0,
        min_interval_points=2,
        order_check=debug_cfg["order_check"],
        output_order=debug_cfg["output_order"],
        return_diagnostics=True,
    )
    debug_summary = build_ordered_slot_decode_summary(
        output_order=debug_cfg["output_order"],
        order_check=debug_cfg["order_check"],
    )
    assert diagnostics["output_order"] == "left_to_right"
    assert [lane["slot"] for lane in lanes] == [1, 0]
    assert debug_summary["result_type"] == "postprocessed_sorted_export"
    assert debug_summary["not_for_main_ordered_slot_claim"] is True


def test_eval_gcs_query_summary_keeps_query_decode_keys() -> None:
    config = build_eval_config(
        weights=ROOT / "dummy.pt",
        source=ROOT / "dummy_images",
        label_dir=None,
        imgsz=(544, 960),
        decode_mode="query",
        conf=0.123,
        point_valid_thr=0.45,
        ape_thr=20.0,
        match_gate_px=None,
        max_x_dist=0.0,
        min_overlap=2,
        nms_dist_px=99.0,
        max_det=9,
        count_aware_topk=True,
        count_aware_min_k=2,
        count_aware_max_k=5,
        count_aware_length_norm=6.0,
        warmup=0,
        device="cpu",
        half=False,
    )
    assert config["decode_mode"] == "query"
    assert config["conf"] == 0.123
    assert config["point_valid_thr"] == 0.45
    assert config["nms_dist_px"] == 99.0
    assert config["max_det"] == 9
    assert config["count_aware_topk"] is True


def test_pred_json_requires_explicit_decode_mode() -> None:
    args = SimpleNamespace(pred_json="pred.json", decode_yaml=None, decode_mode="auto")
    try:
        resolve_pred_json_decode_contract(args)
    except RuntimeError as exc:
        assert "--pred-json mode cannot use --decode-mode auto" in str(exc)
    else:
        raise AssertionError("--pred-json + --decode-mode auto without --decode-yaml must fail.")


def test_pred_json_ordered_slot_rejects_query_only_args() -> None:
    args = SimpleNamespace(
        pred_json="pred.json",
        decode_yaml=None,
        decode_mode="ordered_slot",
        conf=0.003,
        point_valid_thr=0.5,
        nms_dist_px=18.0,
        min_points=6,
        max_det=8,
        count_aware_topk=False,
        count_aware_min_k=3,
        count_aware_max_k=5,
        count_aware_length_norm=12.0,
    )
    try:
        resolve_pred_json_decode_contract(args)
    except RuntimeError as exc:
        assert "ordered_slot decode does not use query decode args" in str(exc)
        assert "conf" in str(exc)
    else:
        raise AssertionError("--pred-json ordered_slot must reject non-default query-only args.")


def test_official_val_requires_explicit_gt_json() -> None:
    tmp = _reset_tmp_subdir("official_val_requires_gt")
    archive_root = tmp / "archive" / "TUSimple"
    (archive_root / "train_set").mkdir(parents=True, exist_ok=True)
    (archive_root / "test_set").mkdir(parents=True, exist_ok=True)

    try:
        evaluate_official(_official_eval_args(tmp, archive_root, gt_json=None, pred_json=None))
    except RuntimeError as exc:
        assert "requires an explicit canonical 363-image --gt-json" in str(exc)
        assert "train_val.json" in str(exc)
        assert "label_data_0313.json" in str(exc)
    else:
        raise AssertionError("split=val official eval must not fallback when --gt-json is omitted.")


def test_official_val_rejects_noncanonical_gt_by_default() -> None:
    tmp = _reset_tmp_subdir("official_val_noncanonical_reject")
    archive_root = tmp / "archive" / "TUSimple"
    (archive_root / "train_set").mkdir(parents=True, exist_ok=True)
    (archive_root / "test_set").mkdir(parents=True, exist_ok=True)
    h_samples = list(range(160, 720, 10))
    gt_path = tmp / "gt_one.json"
    _write_jsonl(gt_path, [{"lanes": [[320 for _ in h_samples], [720 for _ in h_samples]], "h_samples": h_samples, "raw_file": "clips/0000/0.jpg"}])

    try:
        evaluate_official(_official_eval_args(tmp, archive_root, gt_json=gt_path, pred_json=None))
    except RuntimeError as exc:
        assert "Official val GT is not the canonical 363-image validation set" in str(exc)
        assert "got images=1" in str(exc)
        assert "raw_file_sha256" in str(exc)
    else:
        raise AssertionError("split=val official eval must reject non-363 GT unless allow-noncanonical is set.")


def test_official_val_rejects_wrong_363_raw_file_set() -> None:
    tmp = _reset_tmp_subdir("official_val_wrong_363")
    h_samples = list(range(160, 720, 10))
    lanes = [[320 for _ in h_samples], [720 for _ in h_samples]]
    gt_records = [
        {"lanes": lanes, "h_samples": h_samples, "raw_file": f"clips/wrong/{i:04d}/20.jpg"}
        for i in range(363)
    ]
    try:
        validate_canonical_val_gt(gt_records, allow_noncanonical_gt=False)
    except RuntimeError as exc:
        assert "Official val GT is not the canonical 363-image validation set" in str(exc)
        assert "got images=363" in str(exc)
        assert stable_raw_file_hash(gt_records) in str(exc)
    else:
        raise AssertionError("363 GT records with a wrong raw_file set must not be canonical.")

    summary = validate_canonical_val_gt(gt_records, allow_noncanonical_gt=True)
    assert summary["gt_contract"] == "noncanonical"
    assert summary["comparable_to_e1_spurious"] is False
    assert summary["gt_images"] == 363
    assert summary["gt_raw_file_sha256"] == stable_raw_file_hash(gt_records)
    assert summary["gt_content_sha256"] == stable_gt_content_hash(gt_records)


def test_official_val_rejects_same_raw_files_with_modified_gt_content() -> None:
    tmp = _reset_tmp_subdir("official_val_same_raw_changed_content")
    h_samples = list(range(160, 720, 10))
    lanes = [[320 for _ in h_samples], [720 for _ in h_samples]]
    gt_records = [
        {"lanes": [list(lane) for lane in lanes], "h_samples": h_samples, "raw_file": f"clips/canonical/{i:04d}/20.jpg"}
        for i in range(363)
    ]
    manifest_path = tmp / "canonical_manifest.json"
    _write_manifest(manifest_path, gt_records)

    modified_records = [
        {"lanes": [list(lane) for lane in record["lanes"]], "h_samples": list(record["h_samples"]), "raw_file": record["raw_file"]}
        for record in gt_records
    ]
    modified_records[0]["lanes"][0][0] = modified_records[0]["lanes"][0][0] + 1

    assert stable_raw_file_hash(modified_records) == stable_raw_file_hash(gt_records)
    assert stable_gt_content_hash(modified_records) != stable_gt_content_hash(gt_records)

    try:
        validate_canonical_val_gt(modified_records, manifest_path=manifest_path, allow_noncanonical_gt=False)
    except RuntimeError as exc:
        assert "Official val GT is not the canonical 363-image validation set" in str(exc)
        assert stable_raw_file_hash(modified_records) in str(exc)
        assert stable_gt_content_hash(modified_records) in str(exc)
        assert stable_gt_content_hash(gt_records) in str(exc)
    else:
        raise AssertionError("GT records with the canonical raw_file set but modified lanes must not be canonical.")

    summary = validate_canonical_val_gt(modified_records, manifest_path=manifest_path, allow_noncanonical_gt=True)
    assert summary["gt_contract"] == "noncanonical"
    assert summary["comparable_to_e1_spurious"] is False
    assert summary["gt_raw_file_sha256"] == stable_raw_file_hash(modified_records)
    assert summary["canonical_raw_file_sha256"] == stable_raw_file_hash(gt_records)
    assert summary["gt_content_sha256"] == stable_gt_content_hash(modified_records)
    assert summary["canonical_gt_content_sha256"] == stable_gt_content_hash(gt_records)


def test_official_val_canonical_manifest_hash_match_is_comparable() -> None:
    tmp = _reset_tmp_subdir("official_val_canonical_manifest_match")
    h_samples = list(range(160, 720, 10))
    lanes = [[320 for _ in h_samples], [720 for _ in h_samples]]
    gt_path = tmp / "gt_363.json"
    gt_records = [
        {"lanes": lanes, "h_samples": h_samples, "raw_file": f"clips/canonical/{i:04d}/20.jpg"}
        for i in range(363)
    ]
    manifest_path = tmp / "canonical_manifest.json"
    _write_jsonl(gt_path, gt_records)
    _write_manifest(manifest_path, gt_records)

    output = official_gt_contract_summary(
        split="val",
        gt_json=gt_path,
        gt_records=gt_records,
        manifest_path=manifest_path,
    )

    assert output["gt_images"] == 363
    assert output["gt_contract"] == "canonical_official_val_363"
    assert output["comparable_to_e1_spurious"] is True
    assert output["gt_raw_file_sha256"] == stable_raw_file_hash(gt_records)
    assert output["canonical_raw_file_sha256"] == stable_raw_file_hash(gt_records)
    assert output["gt_content_sha256"] == stable_gt_content_hash(gt_records)
    assert output["canonical_gt_content_sha256"] == stable_gt_content_hash(gt_records)


def test_pred_json_ordered_slot_summary_uses_ordered_contract() -> None:
    tmp = _reset_tmp_subdir("pred_json_ordered_slot")
    archive_root = tmp / "archive" / "TUSimple"
    (archive_root / "train_set").mkdir(parents=True, exist_ok=True)
    (archive_root / "test_set").mkdir(parents=True, exist_ok=True)
    h_samples = list(range(160, 720, 10))
    lanes = [[320 for _ in h_samples], [720 for _ in h_samples]]
    raw_file = "clips/0000/0.jpg"
    gt_path = tmp / "gt.json"
    pred_path = tmp / "pred.json"
    _write_jsonl(gt_path, [{"lanes": lanes, "h_samples": h_samples, "raw_file": raw_file}])
    _write_jsonl(
        pred_path,
        [{"lanes": lanes, "h_samples": h_samples, "raw_file": raw_file, "run_time": 1.0}],
    )

    output = evaluate_official(
        SimpleNamespace(
            dataset="tusimple",
            archive_root=str(archive_root),
            split="val",
            gt_json=str(gt_path),
            allow_noncanonical_gt=True,
            pred_json=str(pred_path),
            weights=str(tmp / "unused.pt"),
            decode_mode="ordered_slot",
            decode_yaml=None,
            imgsz=[544, 960],
            conf=0.25,
            point_valid_thr=0.5,
            nms_dist_px=18.0,
            min_points=6,
            max_det=8,
            count_aware_topk=False,
            count_aware_min_k=3,
            count_aware_max_k=5,
            count_aware_length_norm=12.0,
            max_images=0,
            warmup=0,
            device="cpu",
            half=False,
            runtime_ms=1.0,
            use_measured_runtime=False,
            save_dir=str(tmp / "out"),
            save_records=False,
            score_fp_weight=0.02,
            score_fn_weight=0.02,
        )
    )
    forbidden = {"conf", "point_valid_thr", "nms_dist_px", "max_det", "min_points", "count_aware_topk"}
    assert output["config"]["schema"] == "ordered_slot_decode_v1"
    assert output["effective_decode"]["schema"] == "ordered_slot_decode_v1"
    assert output["effective_decode"]["output_order"] == "slot"
    assert output["effective_decode"]["uses_runtime_sort"] is False
    assert output["effective_decode"]["order_violation_policy"] == "fail_fast"
    assert output["query_decode_args"] == "not_applicable"
    assert output["gt_contract"] == "noncanonical"
    assert output["comparable_to_e1_spurious"] is False
    assert output["ordered_slot_order_checked"] is False
    assert output["ordered_slot_order_violations"] is None
    assert output["ordered_slot_order_violation_images"] is None
    assert output["ordered_slot_order_diagnostics"] == "not_available_from_pred_json"
    assert not (forbidden & set(output["config"]))


def test_model_decode_ordered_slot_summary_uses_real_order_stats() -> None:
    tmp = _reset_tmp_subdir("model_decode_ordered_slot")
    archive_root = tmp / "archive" / "TUSimple"
    (archive_root / "train_set").mkdir(parents=True, exist_ok=True)
    (archive_root / "test_set").mkdir(parents=True, exist_ok=True)
    h_samples = list(range(160, 720, 10))
    lanes = [[320 for _ in h_samples], [720 for _ in h_samples]]
    raw_file = "clips/0000/0.jpg"
    gt_path = tmp / "gt.json"
    _write_jsonl(gt_path, [{"lanes": lanes, "h_samples": h_samples, "raw_file": raw_file}])

    def fake_generate_predictions(**kwargs):
        assert kwargs["decode_mode"] == "ordered_slot"
        return (
            [{"lanes": lanes, "h_samples": h_samples, "raw_file": raw_file, "run_time": 1.0}],
            {"avg_inference_ms": 1.0, "avg_postprocess_ms": 0.5, "avg_total_ms": 1.5},
            "ordered_slot",
            {"ordered_slot_order_violations": 4, "ordered_slot_order_violation_images": 2},
        )

    globals_dict = evaluate_official.__globals__
    original_generate_predictions = globals_dict["generate_predictions"]
    globals_dict["generate_predictions"] = fake_generate_predictions
    try:
        output = evaluate_official(
            SimpleNamespace(
                dataset="tusimple",
                archive_root=str(archive_root),
                split="val",
                gt_json=str(gt_path),
                allow_noncanonical_gt=True,
                pred_json=None,
                weights=str(tmp / "unused.pt"),
                decode_mode="ordered_slot",
                decode_yaml=None,
                imgsz=[544, 960],
                conf=0.25,
                point_valid_thr=0.5,
                nms_dist_px=18.0,
                min_points=6,
                max_det=8,
                count_aware_topk=False,
                count_aware_min_k=3,
                count_aware_max_k=5,
                count_aware_length_norm=12.0,
                max_images=0,
                warmup=0,
                device="cpu",
                half=False,
                runtime_ms=1.0,
                use_measured_runtime=False,
                save_dir=str(tmp / "out"),
                save_records=False,
                score_fp_weight=0.02,
                score_fn_weight=0.02,
            )
        )
    finally:
        globals_dict["generate_predictions"] = original_generate_predictions

    assert output["ordered_slot_order_checked"] is True
    assert output["ordered_slot_order_violations"] == 4
    assert output["ordered_slot_order_violation_images"] == 2
    assert output["ordered_slot_order_diagnostics"] == "available_from_decode"


def test_ordered_decode_yaml_rejects_query_keys() -> None:
    cfg = {
        "schema": "ordered_slot_decode_v1",
        "decode_mode": "ordered_slot",
        "gcs_min_lanes": 2,
        "gcs_max_lanes": 5,
        "gcs_num_slots": 5,
        "min_interval_points": 2,
        "interval_repair": "clamp_expand",
        "conf": 0.0,
        "max_det": 5,
    }
    try:
        validate_decode_yaml_for_model(cfg, model_mode="ordered_slot")
    except RuntimeError as exc:
        assert "query-only keys" in str(exc)
        assert "Regenerate official_best_decode.yaml" in str(exc)
    else:
        raise AssertionError("ordered_slot decode yaml with query-only keys should be rejected.")


def test_query_best_decode_yaml_schema_keeps_query_behavior() -> None:
    cfg = build_official_best_decode_cfg(
        {
            "decode_mode": "query",
            "conf": 0.003,
            "point_valid_thr": 0.5,
            "nms_dist_px": 50.0,
            "max_det": 6,
            "min_points": 2,
            "count_aware_topk": False,
            "count_aware_min_k": 3,
            "count_aware_max_k": 5,
            "count_aware_length_norm": 12.0,
        },
        model_mode="query",
    )
    validate_decode_yaml_for_model(cfg, model_mode="query")
    assert cfg["schema"] == "query_decode_v1"
    assert cfg["conf"] == 0.003
    assert cfg["max_det"] == 6


def test_ordered_slot_sweep_row_uses_not_applicable_query_decode_args() -> None:
    row = build_combos(SimpleNamespace(decode_mode="ordered_slot"))[0]
    query_only = {
        "conf",
        "point_valid_thr",
        "nms_dist_px",
        "max_det",
        "min_points",
        "count_aware_topk",
        "count_aware_min_k",
        "count_aware_max_k",
        "count_aware_length_norm",
    }
    assert row["decode_mode"] == "ordered_slot"
    assert row["decode_schema"] == "ordered_slot_decode_v1"
    assert row["effective_decode"]["schema"] == "ordered_slot_decode_v1"
    assert row["effective_decode"]["output_order"] == "slot"
    assert row["effective_decode"]["uses_runtime_sort"] is False
    assert row["effective_decode"]["order_violation_policy"] == "fail_fast"
    assert row["query_decode_args"] == "not_applicable"
    assert not (query_only & set(row))

    row = build_combos(
        SimpleNamespace(decode_mode="ordered_slot", gcs_min_interval_points=2, gcs_bottom_order_margin_px=2.0),
        decode_yaml_cfg={
            "schema": "ordered_slot_decode_v1",
            "decode_mode": "ordered_slot",
            "gcs_min_lanes": 2,
            "gcs_max_lanes": 5,
            "gcs_num_slots": 5,
            "min_interval_points": 4,
            "gcs_bottom_order_margin_px": 7.5,
        },
    )[0]
    assert row["effective_decode"]["min_interval_points"] == 4
    assert row["effective_decode"]["gcs_bottom_order_margin_px"] == 7.5


def test_ordered_slot_sweep_csv_keeps_empty_query_only_columns() -> None:
    tmp = _reset_tmp_subdir("ordered_slot_sweep_csv")
    row = build_combos(SimpleNamespace(decode_mode="ordered_slot"))[0]
    row.update(
        {
            "official_acc": 1.0,
            "official_FP": 0.0,
            "official_FN": 0.0,
            "official_score": 1.0,
            "count_acc": 1.0,
            "count_acc_2": 1.0,
            "images": 1,
            "pred_lanes_hist": {"2": 1},
            "gt_lanes_hist": {"2": 1},
            "count_confusion": {"2->2": 1},
        }
    )
    csv_path = tmp / "sweep.csv"
    write_official_sweep_csv(csv_path, [row])

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        csv_row = next(csv.DictReader(f))
    assert csv_row["decode_schema"] == "ordered_slot_decode_v1"
    assert csv_row["query_decode_args"] == "not_applicable"
    for key in (
        "conf",
        "point_valid_thr",
        "nms_dist_px",
        "max_det",
        "min_points",
        "count_aware_topk",
        "count_aware_min_k",
        "count_aware_max_k",
        "count_aware_length_norm",
    ):
        assert csv_row[key] == ""


def test_legacy_conf_sweep_rejects_ordered_slot_model() -> None:
    try:
        assert_legacy_query_conf_sweep_model(DummyOrderedModel())
    except RuntimeError as exc:
        msg = str(exc)
        assert "legacy query-threshold sweep tool" in msg
        assert "ordered_slot" in msg
        assert "decode_gcs_predictions" not in msg
    else:
        raise AssertionError("legacy sweep_gcs_conf.py must reject ordered_slot models.")


def test_legacy_conf_sweep_rejects_test_split() -> None:
    try:
        assert_legacy_query_conf_sweep_split("test")
    except RuntimeError as exc:
        msg = str(exc)
        assert "threshold search tool" in msg
        assert "validation data" in msg
        assert "test may only be evaluated once with a fixed decode config" in msg
    else:
        raise AssertionError("legacy sweep_gcs_conf.py must reject --split test.")


def test_strip_gcs_head_key_matches_top_level_heads_only() -> None:
    assert is_gcs_head_key("query_embed.weight")
    assert is_gcs_head_key("point_head.weight")
    assert is_gcs_head_key("exist_head.bias")
    assert is_gcs_head_key("valid_head.weight")
    assert is_gcs_head_key("start_head.weight")
    assert is_gcs_head_key("end_head.weight")
    assert is_gcs_head_key("count_head.weight")
    assert is_gcs_head_key("module.query_embed.weight")
    assert not is_gcs_head_key("backbone.conv.weight")
    state = {
        "query_embed.weight": torch.zeros(1),
        "point_head.weight": torch.zeros(1),
        "backbone.conv.weight": torch.ones(1),
    }
    removed = {key for key in state if is_gcs_head_key(key)}
    kept = {key for key in state if not is_gcs_head_key(key)}
    assert removed == {"query_embed.weight", "point_head.weight"}
    assert kept == {"backbone.conv.weight"}


def test_visualize_ordered_slot_targets_saves_rgb_as_bgr() -> None:
    out_path = _reset_tmp_subdir("target_rgb_save") / "red.jpg"
    image_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    image_rgb[:, :] = (255, 0, 0)
    assert _save_rgb_image(out_path, image_rgb)
    saved_bgr = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    assert saved_bgr is not None
    assert int(saved_bgr[0, 0, 2]) > 200
    assert int(saved_bgr[0, 0, 0]) < 50


def test_visualize_ordered_slot_targets_write_failure_raises() -> None:
    vis_globals = _save_rgb_image.__globals__
    original_imwrite = vis_globals["cv2"].imwrite
    vis_globals["cv2"].imwrite = lambda *args, **kwargs: False
    try:
        try:
            _save_rgb_image(_reset_tmp_subdir("target_write_fail") / "fail.jpg", np.zeros((4, 4, 3), dtype=np.uint8))
        except IOError as exc:
            assert "Failed to write ordered-slot target visualization" in str(exc)
        else:
            raise AssertionError("visualize_ordered_slot_targets must raise when cv2.imwrite fails.")
    finally:
        vis_globals["cv2"].imwrite = original_imwrite


def test_visualize_ordered_slot_targets_help_has_no_circular_import() -> None:
    result = subprocess.run(
        [sys.executable, "tools/visualize_ordered_slot_targets.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Visualize ordered-slot targets" in result.stdout


def test_overfit20_failure_metrics_raise() -> None:
    failing = {
        "slot_count_acc": 0.95,
        "slot_exist_acc": 1.0,
        "interval_start_mae": 0.0,
        "interval_end_mae": 0.0,
        "order_violation_rate": 0.0,
        "point_l1_px": 0.0,
    }
    try:
        assert_overfit_passed(failing, OVERFIT_THRESHOLDS)
    except AssertionError as exc:
        assert "slot_count_acc" in str(exc)
    else:
        raise AssertionError("assert_overfit_passed must fail when required overfit metrics miss thresholds.")


def test_overfit20_skips_absent_per_class_count_acc() -> None:
    metrics = {"slot_count_acc_5": float("nan"), "slot_count_total_5": 0.0}
    thresholds = {"slot_count_acc_5": {"op": ">=", "value": 1.0}}
    assert overfit_failures(metrics, thresholds) == []


def test_standalone_validator_syncs_ordered_mode() -> None:
    validator = GCSLaneValidator(args=SimpleNamespace())
    mode = validator._sync_gcs_mode_from_model(DummyOrderedModel())

    assert mode == "ordered_slot"
    assert validator.args.gcs_mode == "ordered_slot"
    assert validator._loss_names()[0].startswith("slot_")


def test_training_validator_order_violation_is_diagnostic_not_fatal() -> None:
    validator = GCSLaneValidator(args=SimpleNamespace(gcs_mode="ordered_slot"))
    validator.model = DummyOrderedModel()
    preds = _base_preds(batch=1)
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_points"][0, 0, :, 0] = 0.8
    preds["pred_points"][0, 1, :, 0] = 0.3
    preds["pred_logits"] = preds["pred_exist_logits"]
    points, valid = _fixed_y_points(2)
    batch = {
        "img": torch.zeros(1, 3, 544, 960),
        "lanes": [points],
        "lane_valid": [valid],
    }
    state = validator._empty_metric_state()

    validator._update_metric_state(state, preds, batch)

    assert state["images"] == 1
    assert state["ordered_slot_order_violations"] == 1
    assert state["ordered_slot_order_violation_images"] == 1
    metrics = validator._metric_results(state)
    assert metrics["val/ordered_slot_order_violations"] == 1.0
    assert metrics["val/ordered_slot_order_violation_images"] == 1.0
    assert metrics["val/ordered_slot_order_violation_rate"] == 1.0


def test_training_validator_passes_ordered_decode_contract_args() -> None:
    validator = GCSLaneValidator(
        args=SimpleNamespace(
            gcs_mode="ordered_slot",
            gcs_min_lanes=2,
            gcs_max_lanes=5,
            gcs_min_interval_points=4,
            gcs_bottom_order_margin_px=7.5,
        )
    )
    validator.model = DummyOrderedModel()
    preds = _base_preds(batch=1)
    preds["pred_count_logits"][0, 0] = 10.0
    preds["pred_logits"] = preds["pred_exist_logits"]
    points, valid = _fixed_y_points(2)
    batch = {
        "img": torch.zeros(1, 3, 544, 960),
        "lanes": [points],
        "lane_valid": [valid],
    }
    state = validator._empty_metric_state()

    captured: dict = {}

    def fake_decode(*args, **kwargs):
        captured.update(kwargs)
        return [], {"order_violation_count": 0, "has_order_violation": False}

    globals_dict = GCSLaneValidator._update_metric_state.__globals__
    original_decode = globals_dict["decode_ordered_slot_predictions"]
    globals_dict["decode_ordered_slot_predictions"] = fake_decode
    try:
        validator._update_metric_state(state, preds, batch)
    finally:
        globals_dict["decode_ordered_slot_predictions"] = original_decode

    assert captured["min_lanes"] == 2
    assert captured["max_lanes"] == 5
    assert captured["min_interval_points"] == 4
    assert captured["order_margin_px"] == 7.5
    assert captured["img_w"] == 960.0


def test_git_tracking_and_idea_ignore() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".idea/GCS-YOLO-Lane_LSA_5-25-3-k56.iml"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert ignored.returncode == 0, ".idea/*.iml must be ignored by .gitignore."

    missing = []
    for rel in TRACKED_FILES:
        if not (ROOT / rel).exists():
            missing.append(f"missing_on_disk: {rel}")
            continue
        result = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            missing.append(f"untracked: {rel}")
    assert not missing, f"ordered_slot delivery files are not tracked: {missing}"


def test_required_ordered_slot_imports_available() -> None:
    for module_name in REQUIRED_IMPORT_MODULES:
        importlib.import_module(module_name)


def test_train_gcs_override_keys_are_registered_in_default_yaml() -> None:
    source = (ROOT / "tools/train_gcs.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    override_keys: set[str] = set()

    class OverrideVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            if not isinstance(node.value, ast.Dict):
                return
            if not any(isinstance(target, ast.Name) and target.id == "overrides" for target in node.targets):
                return
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    override_keys.add(key.value)

    OverrideVisitor().visit(module)
    cfg = yaml.safe_load((ROOT / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8"))
    missing = sorted(key for key in override_keys if key not in cfg)
    assert not missing, f"tools/train_gcs.py passes unregistered YOLO override keys: {missing}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ordered-slot GCS contract fixes.")
    parser.add_argument("--tmp-dir", default=str(TMP_DIR), help="Workspace-local directory for temporary checkpoints.")
    parser.add_argument("--skip-git", action="store_true", help="Skip git tracking/ignore assertions.")
    return parser.parse_args()


def main() -> None:
    global TMP_DIR
    args = parse_args()
    TMP_DIR = Path(args.tmp_dir)
    if not TMP_DIR.is_absolute():
        TMP_DIR = (ROOT / TMP_DIR).resolve()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tests = [
        test_two_lane_target_and_count_acc,
        test_slot_count_absent_class_logs_nan_and_totals,
        test_slot_count_acc_aggregates_from_correct_total_and_skips_absent,
        test_ordered_slot_loss_defaults_match_default_yaml_and_do_not_silent_disable,
        test_ordered_slot_loss_core_supervision_requires_explicit_ablation,
        test_trainer_ordered_slot_loss_contract_rejects_disabled_bottom_order_losses,
        test_ordered_slot_validator_loss_gains_include_bottom_x_terms,
        test_bottom_order_loss_catches_no_common_anchor_violation,
        test_decoded_bottom_idx_matches_repaired_interval_when_start_gt_end,
        test_decoded_bottom_helper_sees_same_start_gt_end_violation_as_strict_decoder,
        test_decoded_bottom_loss_and_metric_use_repaired_start_end_interval,
        test_slot_gt_bottom_x_loss_uses_repaired_decoded_bottom_index,
        test_slot_gt_bottom_x_soft_loss_backprops_to_start_logits,
        test_slot_gt_bottom_x_loss_reports_available_target_keys,
        test_decode_count_two_lanes,
        test_ordered_slot_head_requires_five_slots_and_queries,
        test_ordered_slot_decode_shape_guard_rejects_bad_count_logits,
        test_ordered_slot_decode_order_check_error,
        test_ordered_slot_decode_default_is_strict_slot_order,
        test_ordered_slot_decode_rejects_sort_with_error_policy,
        test_official_ordered_slot_decode_reversed_slots_fail_fast,
        test_ordered_slot_decode_left_to_right_sorts_and_preserves_slot,
        test_ordered_slot_decode_signature_has_no_exist_safety,
        test_decode_count_five_repairs_start_equals_end,
        test_decode_mode_mismatch_raises_everywhere,
        test_resume_disables_pretrained_and_loads_head_weights,
        test_noncontiguous_strict_and_repair_interp,
        test_fixed_y_wrong_k_fails_fast,
        test_ordered_point_loss_has_non_tiny_gradient_for_10px_error,
        test_ordered_point_loss_defaults_preserve_protocol_behavior,
        test_aspect_l1_requires_explicit_ordered_point_loss_flag,
        test_training_fixed_y_ascending_fails_and_official_ascending_passes,
        test_tusimple_record_h_samples_accepts_canonical_subsets_for_export,
        test_slot_target_removes_padding_before_fixed_y_validation,
        test_ordered_slot_target_sorts_shuffled_gt_by_bottom_x,
        test_ordered_slot_target_accepts_half_quantized_fixed_y,
        test_ordered_slot_loss_amp_half_preds_float32_gt_fixed_y,
        test_slot_target_training_fixed_y_ascending_fails,
        test_selection_policy_matches_sweep_sort_key,
        test_official_selection_policy_matches_code_source,
        test_official_best_selection_prefers_strict_order_before_acc,
        test_ordered_slot_summary_helpers_expose_schema_specific_decode,
        test_ordered_slot_runtime_config_contexts,
        test_ordered_slot_order_diag_pred_json_not_available,
        test_ordered_slot_order_diag_model_decode_available,
        test_ordered_slot_official_eval_rejects_non_default_query_args,
        test_ordered_slot_training_official_sweep_rejects_query_only_args,
        test_query_training_official_sweep_keeps_user_grid,
        test_default_yaml_count_balanced_false,
        test_train_gcs_count_balanced_default_false_and_explicit_true,
        test_ordered_slot_without_official_best_requires_explicit_debug_escape,
        test_train_gcs_defaults_match_current_contract_paths,
        test_ordered_slot_scale_positive_fails_fast,
        test_query_mode_scale_behavior_is_unchanged,
        test_ordered_slot_auto_switches_any_non_slot_model_yaml,
        test_ordered_slot_auto_switch_can_be_disabled,
        test_generic_ordered_slot_query_yaml_mismatch_mentions_q5_slot_model,
        test_query_model_build_ignores_ordered_slot_overrides,
        test_eval_gcs_ordered_summary_has_no_query_decode_keys,
        test_eval_gcs_query_summary_keeps_query_decode_keys,
        test_pred_json_requires_explicit_decode_mode,
        test_pred_json_ordered_slot_rejects_query_only_args,
        test_official_val_requires_explicit_gt_json,
        test_official_val_rejects_noncanonical_gt_by_default,
        test_official_val_rejects_wrong_363_raw_file_set,
        test_official_val_rejects_same_raw_files_with_modified_gt_content,
        test_official_val_canonical_manifest_hash_match_is_comparable,
        test_pred_json_ordered_slot_summary_uses_ordered_contract,
        test_model_decode_ordered_slot_summary_uses_real_order_stats,
        test_ordered_decode_yaml_rejects_query_keys,
        test_query_best_decode_yaml_schema_keeps_query_behavior,
        test_ordered_slot_sweep_row_uses_not_applicable_query_decode_args,
        test_ordered_slot_sweep_csv_keeps_empty_query_only_columns,
        test_legacy_conf_sweep_rejects_ordered_slot_model,
        test_legacy_conf_sweep_rejects_test_split,
        test_strip_gcs_head_key_matches_top_level_heads_only,
        test_visualize_ordered_slot_targets_saves_rgb_as_bgr,
        test_visualize_ordered_slot_targets_write_failure_raises,
        test_visualize_ordered_slot_targets_help_has_no_circular_import,
        test_overfit20_failure_metrics_raise,
        test_overfit20_skips_absent_per_class_count_acc,
        test_standalone_validator_syncs_ordered_mode,
        test_training_validator_order_violation_is_diagnostic_not_fatal,
        test_training_validator_passes_ordered_decode_contract_args,
        test_required_ordered_slot_imports_available,
        test_train_gcs_override_keys_are_registered_in_default_yaml,
    ]
    if not args.skip_git:
        tests.append(test_git_tracking_and_idea_ignore)
    passed = []
    for test in tests:
        test()
        passed.append(test.__name__)
    print(json.dumps({"passed": passed}, indent=2))


if __name__ == "__main__":
    main()
