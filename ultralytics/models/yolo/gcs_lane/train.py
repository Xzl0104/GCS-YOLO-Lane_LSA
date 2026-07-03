# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Trainer for GCS-YOLO-Lane structured lane detection."""

from __future__ import annotations

import ast
import json
import math
import numpy as np
import random
import re
import shutil
from collections import Counter
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from gcs_tools.official_selection import OFFICIAL_SELECTION_POLICY, official_best_sort_key
from ultralytics.data import build_dataloader
from ultralytics.data.dataset_gcs import GCSLaneDataset, resize_gcs_masks
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.gcs_lane.val import GCSLaneValidator
from ultralytics.models.gcs.decode_summary import build_official_best_decode_cfg, raise_for_ordered_slot_query_args
from ultralytics.models.gcs.loss_ordered_slot import OrderedSlotGCSLoss
from ultralytics.models.gcs.mode_utils import (
    assert_ordered_slot_scale_contract,
    infer_gcs_mode_from_ckpt,
    infer_gcs_mode_from_model,
)
from ultralytics.nn.modules import GCSLaneHead, LaneBiFPN, LSEM
from ultralytics.nn.tasks import GCSLaneModel, load_checkpoint, torch_safe_load
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, RANK, ROOT, YAML
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, assert_gcs_shape, normalize_imgsz
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first


class GCSLaneTrainer(BaseTrainer):
    """Train GCS-YOLO-Lane with structured lane labels and GCSLoss."""

    loss_names = (
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
        "short_side_geom_loss",
        "short_side_geom_count",
        "short_side_geom_gt4",
        "short_side_geom_gt5",
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
        "far_spur_loss",
        "far_spur_cand",
        "far_spur_neg",
        "far_spur_gt3",
        "far_spur_gt4",
        "far_spur_gt5",
        "far_spur_score",
        "far_spur_valid",
        "count_score_mean",
        "gt5_short_pos_count",
        "gt5_short_pos_anchor_count",
        "gt5_short_point_valid_loss",
        "cnt_bound_5under",
        "cnt_score",
    )
    # Keep tqdm headers within BaseTrainer's 11-character progress columns.
    progress_loss_names = (
        "exist",
        "point",
        "pt_valid",
        "smooth",
        "curve",
        "mask",
        "edge",
        "count",
        "cnt_under5",
        "cnt_bound",
        "short_geom",
        "sg_count",
        "sg_gt4",
        "sg_gt5",
        "spur_loss",
        "spur_cnt",
        "spur_cand",
        "spur_prot",
        "spur_final",
        "spur_neg",
        "spcnt_g3",
        "spcnt_g4",
        "spcnt_g5",
        "spneg_g3",
        "spneg_g4",
        "spneg_g5",
        "fspur_loss",
        "fspur_cand",
        "fspur_neg",
        "fspur_gt3",
        "fspur_gt4",
        "fspur_gt5",
        "fspur_scr",
        "fspur_val",
        "cnt_score",
        "gt5s_cnt",
        "gt5s_anc",
        "gt5s_pv",
        "cnt5under",
        "cnt_score2",
    )
    # YOLO11 backbone -> GCS-YOLO-Lane backbone. LSEM is inserted after old
    # layers 4 and 6, so all later backbone layers must be shifted explicitly.
    yolo11_to_gcs_backbone = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 6,
        6: 7,
        7: 9,
        8: 10,
        9: 11,
        10: 12,
    }

    @staticmethod
    def _normalize_gcs_mode(mode: Any) -> str:
        """Normalize GCS mode spelling used by CLI, YAML, and head modules."""
        mode = str(mode or "query").lower()
        if mode in {"ordered-slot", "orderedslot"}:
            mode = "ordered_slot"
        if mode not in {"query", "ordered_slot"}:
            raise ValueError(f"gcs_mode must be 'query' or 'ordered_slot', got {mode!r}.")
        return mode

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize the GCS lane trainer."""
        overrides = dict(overrides or {})
        overrides["task"] = "gcs_lane"
        overrides.setdefault("model", str(ROOT / "cfg/models/gcs/gcs-yolo-lane-s.yaml"))
        overrides.setdefault("data", str(ROOT.parent / "data/tusimple_gcs_fixed_y_960x544.yaml"))
        # GCSLaneHead defaults to 8 lane queries. Four-image mosaic can raise TuSimple GT lanes to ~16,
        # leaving unmatched GT lanes outside the structured point loss, so keep mosaic off unless requested.
        overrides.setdefault("mosaic", 0.0)
        # Plain GCS training historically used no affine erasing/scale aug; keep direct trainer construction stable.
        overrides.setdefault("scale", 0.0)
        overrides.setdefault("erasing", 0.0)
        super().__init__(cfg, overrides, _callbacks)
        self._warned_ordered_slot_without_official_best = False
        assert_ordered_slot_scale_contract(self._gcs_mode(), getattr(self.args, "scale", 0.0))
        self.official_best = self.wdir / "official_best.pt"
        self.official_best_sweep = self.wdir / "official_best_sweep.json"
        self.official_best_decode = self.wdir / "official_best_decode.yaml"
        self._official_best_state = self._load_official_best_state()
        self._lock_gcs_shape_contract()
        self._set_loss_names_for_mode()
        self._warn_if_ordered_slot_without_official_best()

    def _gcs_mode(self) -> str:
        """Return normalized GCS training mode."""
        head = None
        model = getattr(self, "model", None)
        if model is not None and getattr(model, "model", None) is not None and len(model.model):
            head = model.model[-1]
        if isinstance(head, GCSLaneHead):
            return self._normalize_gcs_mode(getattr(head, "gcs_mode", "query"))
        return self._normalize_gcs_mode(getattr(self.args, "gcs_mode", "query"))

    def _set_args_gcs_mode(self, mode: str) -> None:
        """Persist normalized mode into args so loss, val, and official hooks agree."""
        if isinstance(self.args, dict):
            self.args["gcs_mode"] = mode
        else:
            self.args.gcs_mode = mode

    def _set_arg_value(self, name: str, value: Any) -> None:
        """Persist one derived metadata value into trainer args."""
        if isinstance(self.args, dict):
            self.args[name] = value
        else:
            setattr(self.args, name, value)

    def _get_arg_value(self, name: str, default: Any = None) -> Any:
        """Read one trainer arg from a dict or namespace."""
        return self.args.get(name, default) if isinstance(self.args, dict) else getattr(self.args, name, default)

    def _ordered_slot_loss_contract_from_args(self) -> dict[str, Any]:
        """Return and validate the effective ordered-slot loss supervision contract."""
        count_ce = float(self._get_arg_value("gcs_count_ce", 1.0))
        interval = float(self._get_arg_value("gcs_interval", 1.0))
        order = float(self._get_arg_value("gcs_order", 0.2))
        gt_bottom_order = float(self._get_arg_value("gcs_gt_bottom_order", 1.0))
        decoded_bottom_order = float(self._get_arg_value("gcs_decoded_bottom_order", 1.0))
        slot_gt_bottom_x = float(self._get_arg_value("gcs_slot_gt_bottom_x", 0.0))
        slot_gt_bottom_x_beta = float(self._get_arg_value("gcs_slot_gt_bottom_x_beta", 0.05))
        slot_gt_bottom_x_detach_interval = bool(int(self._get_arg_value("gcs_slot_gt_bottom_x_detach_interval", 1)))
        slot_gt_bottom_x_soft = float(self._get_arg_value("gcs_slot_gt_bottom_x_soft", 0.0))
        slot_gt_bottom_x_soft_tau = float(self._get_arg_value("gcs_slot_gt_bottom_x_soft_tau", 0.5))
        slot_gt_bottom_x_soft_beta = float(self._get_arg_value("gcs_slot_gt_bottom_x_soft_beta", 0.05))
        slot_start_index_l1 = float(self._get_arg_value("gcs_slot_start_index_l1", 0.0))
        slot_start_index_l1_beta = float(self._get_arg_value("gcs_slot_start_index_l1_beta", 2.0))
        min_interval_points = int(self._get_arg_value("gcs_min_interval_points", 2))
        bottom_order_margin_px = float(self._get_arg_value("gcs_bottom_order_margin_px", 2.0))
        allow_disable_order = bool(self._get_arg_value("gcs_allow_disable_order_loss", False))
        if slot_gt_bottom_x < 0.0:
            raise ValueError(f"gcs_slot_gt_bottom_x must be >= 0, got {slot_gt_bottom_x}.")
        if slot_gt_bottom_x_beta <= 0.0:
            raise ValueError(f"gcs_slot_gt_bottom_x_beta must be > 0, got {slot_gt_bottom_x_beta}.")
        if slot_gt_bottom_x_soft < 0.0:
            raise ValueError(f"gcs_slot_gt_bottom_x_soft must be >= 0, got {slot_gt_bottom_x_soft}.")
        if slot_gt_bottom_x_soft_tau <= 0.0:
            raise ValueError(f"gcs_slot_gt_bottom_x_soft_tau must be > 0, got {slot_gt_bottom_x_soft_tau}.")
        if slot_gt_bottom_x_soft_beta <= 0.0:
            raise ValueError(f"gcs_slot_gt_bottom_x_soft_beta must be > 0, got {slot_gt_bottom_x_soft_beta}.")
        if slot_start_index_l1 < 0.0:
            raise ValueError(f"gcs_slot_start_index_l1 must be >= 0, got {slot_start_index_l1}.")
        if slot_start_index_l1_beta <= 0.0:
            raise ValueError(f"gcs_slot_start_index_l1_beta must be > 0, got {slot_start_index_l1_beta}.")
        if count_ce <= 0.0:
            raise RuntimeError(
                "ordered_slot requires gcs_count_ce > 0. "
                "Count-class supervision cannot be disabled silently."
            )
        if interval <= 0.0:
            raise RuntimeError(
                "ordered_slot requires gcs_interval > 0. "
                "Visibility interval supervision cannot be disabled silently."
            )
        if order <= 0.0 and not allow_disable_order:
            raise RuntimeError(
                "ordered_slot order loss is disabled. "
                "Pass --gcs-allow-disable-order-loss only for an explicit ablation."
            )
        if gt_bottom_order <= 0.0 and not allow_disable_order:
            raise RuntimeError(
                "ordered_slot strict contract requires gcs_gt_bottom_order > 0. "
                "Pass --gcs-allow-disable-order-loss only for an explicit ablation."
            )
        if decoded_bottom_order <= 0.0 and not allow_disable_order:
            raise RuntimeError(
                "ordered_slot strict contract requires gcs_decoded_bottom_order > 0. "
                "Pass --gcs-allow-disable-order-loss only for an explicit ablation."
            )
        if min_interval_points <= 0:
            raise ValueError(f"gcs_min_interval_points must be > 0, got {min_interval_points}.")
        return {
            "gcs_point": float(self._get_arg_value("gcs_point", 15.0)),
            "gcs_exist": float(self._get_arg_value("gcs_exist", 2.0)),
            "gcs_point_valid": float(self._get_arg_value("gcs_point_valid", 1.0)),
            "gcs_count_ce": count_ce,
            "gcs_interval": interval,
            "gcs_order": order,
            "gcs_gt_bottom_order": gt_bottom_order,
            "gcs_decoded_bottom_order": decoded_bottom_order,
            "gcs_slot_gt_bottom_x": slot_gt_bottom_x,
            "gcs_slot_gt_bottom_x_beta": slot_gt_bottom_x_beta,
            "gcs_slot_gt_bottom_x_detach_interval": slot_gt_bottom_x_detach_interval,
            "gcs_slot_gt_bottom_x_soft": slot_gt_bottom_x_soft,
            "gcs_slot_gt_bottom_x_soft_tau": slot_gt_bottom_x_soft_tau,
            "gcs_slot_gt_bottom_x_soft_beta": slot_gt_bottom_x_soft_beta,
            "gcs_slot_start_index_l1": slot_start_index_l1,
            "gcs_slot_start_index_l1_beta": slot_start_index_l1_beta,
            "gcs_min_interval_points": min_interval_points,
            "gcs_ordered_point_loss": str(self._get_arg_value("gcs_ordered_point_loss", "normalized_smooth_l1")),
            "gcs_bottom_order_margin_px": bottom_order_margin_px,
            "gcs_allow_disable_order_loss": allow_disable_order,
            "count_supervision_enabled": count_ce > 0.0,
            "interval_supervision_enabled": interval > 0.0,
            "order_supervision_enabled": order > 0.0,
            "gt_bottom_order_supervision_enabled": gt_bottom_order > 0.0,
            "decoded_bottom_order_supervision_enabled": decoded_bottom_order > 0.0,
            "slot_gt_bottom_x_supervision_enabled": slot_gt_bottom_x > 0.0,
            "slot_gt_bottom_x_soft_supervision_enabled": slot_gt_bottom_x_soft > 0.0,
            "slot_start_index_l1_supervision_enabled": slot_start_index_l1 > 0.0,
            "slot4_exist_bce_weight": float(self._get_arg_value("gcs_slot_exist_w4", 1.0)),
            "slot5_exist_bce_weight": float(self._get_arg_value("gcs_slot_exist_w5", 1.0)),
            "slot_exist_weight_semantics": "BCE element weight applied to positive and negative targets",
        }

    def _record_ordered_slot_loss_contract(self) -> None:
        """Record effective ordered-slot supervision in args.yaml and checkpoints."""
        if self._gcs_mode() != "ordered_slot":
            return
        contract = self._ordered_slot_loss_contract_from_args()
        self._set_arg_value("ordered_slot_loss_contract", contract)
        self._set_arg_value(
            "ordered_slot_effective_loss_weights",
            {
                k: contract[k]
                for k in (
                    "gcs_point",
                    "gcs_exist",
                    "gcs_point_valid",
                    "gcs_count_ce",
                    "gcs_interval",
                    "gcs_order",
                    "gcs_gt_bottom_order",
                    "gcs_decoded_bottom_order",
                    "gcs_slot_gt_bottom_x",
                    "gcs_slot_gt_bottom_x_soft",
                    "gcs_slot_start_index_l1",
                )
            },
        )

    def _warn_if_ordered_slot_without_official_best(self) -> None:
        """Fail formal ordered-slot training unless checkpoint selection uses strict official-val."""
        if self._gcs_mode() != "ordered_slot" or bool(getattr(self, "_warned_ordered_slot_without_official_best", False)):
            return
        if not bool(self._get_arg_value("gcs_official_best", False)):
            if bool(self._get_arg_value("gcs_allow_internal_best", False)):
                LOGGER.warning(
                    "ordered_slot debug training is running with gcs_official_best=False because "
                    "gcs_allow_internal_best=True. weights/best.pt is internal validation best, not strict official best."
                )
                self._warned_ordered_slot_without_official_best = True
                return
            raise RuntimeError(
                "ordered_slot formal training requires --gcs-official-best. "
                "Otherwise weights/best.pt is only internal-val best, not strict official best. "
                "For debug runs, add --gcs-allow-internal-best."
            )

    @staticmethod
    def _log_ordered_slot_model_contract(model: nn.Module) -> None:
        """Log the model-side ordered-slot contract for reproducibility."""
        head = model.model[-1] if getattr(model, "model", None) is not None and len(model.model) else None
        if not isinstance(head, GCSLaneHead) or str(getattr(head, "gcs_mode", "query")) != "ordered_slot":
            return
        LOGGER.info(
            "ordered_slot model contract: "
            f"gcs_mode={getattr(head, 'gcs_mode', None)}, "
            f"num_queries={getattr(head, 'num_queries', None)}, "
            f"num_points={getattr(head, 'num_points', None)}, "
            f"num_slots={getattr(head, 'num_slots', None)}, "
            f"min_lanes={getattr(head, 'min_lanes', None)}, "
            f"max_lanes={getattr(head, 'max_lanes', None)}, "
            f"count_classes={getattr(head, 'count_classes', None)}, "
            f"point_mode={getattr(head, 'point_mode', None)}"
        )

    def _assert_model_gcs_mode(self, model: nn.Module) -> None:
        """Fail fast when CLI/YAML/head GCS modes disagree."""
        head = model.model[-1] if getattr(model, "model", None) is not None and len(model.model) else None
        if not isinstance(head, GCSLaneHead):
            return
        head_mode = self._normalize_gcs_mode(getattr(head, "gcs_mode", "query"))
        yaml_mode = self._normalize_gcs_mode(getattr(model, "yaml", {}).get("gcs_mode", head_mode))
        arg_mode = self._normalize_gcs_mode(getattr(self.args, "gcs_mode", yaml_mode))
        if yaml_mode != head_mode:
            raise ValueError(f"GCS YAML gcs_mode={yaml_mode!r} but GCSLaneHead.gcs_mode={head_mode!r}.")
        if arg_mode == "query" and yaml_mode == "ordered_slot":
            arg_mode = "ordered_slot"
            self._set_args_gcs_mode(arg_mode)
        if arg_mode != head_mode:
            raise ValueError(
                f"Requested gcs_mode={arg_mode!r} but model head is {head_mode!r}. "
                "The generic YOLO entrypoint uses TASK2MODEL['gcs_lane'] query YAML by default and does not "
                "auto-switch ordered_slot models. Use tools/train_gcs.py for auto-switching, or pass "
                "model=ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q5-slot-k56.yaml when using "
                "yolo task=gcs_lane gcs_mode=ordered_slot."
            )
        model.gcs_mode = head_mode
        assert_ordered_slot_scale_contract(head_mode, getattr(self.args, "scale", 0.0))
        self._log_ordered_slot_model_contract(model)

    def _set_loss_names_for_mode(self) -> None:
        """Select the loss vector labels that match the active GCS criterion."""
        if self._gcs_mode() == "ordered_slot":
            self.loss_names = OrderedSlotGCSLoss.loss_names
            self.progress_loss_names = OrderedSlotGCSLoss.progress_loss_names
        else:
            self.loss_names = self.__class__.loss_names
            self.progress_loss_names = self.__class__.progress_loss_names

    def get_dataset(self) -> dict[str, Any]:
        """Load a standard YAML but pair images with labels_gcs/*.npz at dataset time."""
        data = check_det_dataset(self.args.data)
        train_images = getattr(self.args, "train_images", None)
        val_images = getattr(self.args, "val_images", None)
        if train_images:
            data["train"] = str(train_images)
        if val_images:
            data["val"] = str(val_images)
        if self.args.single_cls:
            data["names"] = {0: "lane"}
            data["nc"] = 1
        return data

    def _resolve_gcs_imgsz(self) -> tuple[int, int]:
        """Resolve the real GCS input shape as (height, width)."""
        arg_shape = getattr(self.args, "gcs_imgsz", None)
        if arg_shape is not None and arg_shape != "" and arg_shape is not False:
            return normalize_imgsz(arg_shape)
        for key in ("gcs_imgsz", "image_shape"):
            if isinstance(getattr(self, "data", None), dict) and self.data.get(key) is not None:
                return normalize_imgsz(self.data[key])
        arg_imgsz = getattr(self.args, "imgsz", None)
        if isinstance(arg_imgsz, (list, tuple)) and len(arg_imgsz) > 1:
            return normalize_imgsz(arg_imgsz)
        if isinstance(arg_imgsz, str) and any(x in arg_imgsz.lower() for x in (",", "x", "[", "(")):
            return normalize_imgsz(arg_imgsz)
        raise AssertionError(
            "GCSLaneTrainer requires a real rectangular GCS shape from args.gcs_imgsz or data image_shape, "
            f"but only scalar args.imgsz={arg_imgsz!r} was available. Use [544, 960] for TuSimple or [384, 960] "
            "for CULane; scalar args.imgsz is only a YOLO engine long-side value."
        )

    def _save_shape_locked_args(self) -> None:
        """Rewrite args.yaml after locking GCS shape so the run metadata is not ambiguous."""
        if RANK not in {-1, 0}:
            return
        args_dict = vars(self.args).copy()
        if args_dict.get("augmentations") is not None:
            args_dict["augmentations"] = [repr(t) for t in args_dict["augmentations"]]
        YAML.save(self.save_dir / "args.yaml", args_dict)

    def _rewrite_args_yaml_after_gcs_mode_sync(self) -> None:
        """Rewrite args.yaml after the constructed head synchronizes args.gcs_mode."""
        self._save_shape_locked_args()
        if RANK in {-1, 0}:
            LOGGER.info(f"Rewrote args.yaml after GCS mode sync: gcs_mode={self._gcs_mode()}.")

    def _lock_gcs_shape_contract(self) -> None:
        """Store the rectangular GCS H,W contract on trainer args and fail on square-only configs."""
        shape = self._resolve_gcs_imgsz()
        assert shape[0] != shape[1], (
            f"GCSLaneTrainer resolved a square GCS shape H,W={shape}. "
            "This project requires rectangular GCS inputs: TuSimple [544, 960], CULane [384, 960]."
        )
        assert float(getattr(self.args, "multi_scale", 0.0) or 0.0) == 0.0, (
            "GCS structured labels use a fixed rectangular image contract. Set multi_scale=0.0 so image, "
            "point, mask, edge, and eval pixel scales all stay aligned."
        )
        self.gcs_imgsz = shape
        self.args.gcs_imgsz = [int(shape[0]), int(shape[1])]
        # Keep Ultralytics train internals on their required scalar long side while GCS uses gcs_imgsz.
        self.args.imgsz = max(int(shape[0]), int(shape[1]))
        self._save_shape_locked_args()

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build the GCS lane dataset."""
        fraction = self.args.fraction if mode == "train" else 1.0
        image_dir = getattr(self.args, f"{mode}_images", None) or img_path
        label_dir = getattr(self.args, f"{mode}_gcs_labels", None)
        augment = mode == "train"
        gcs_imgsz = self._resolve_gcs_imgsz()
        dataset = GCSLaneDataset(
            img_path=image_dir,
            imgsz=gcs_imgsz,
            fraction=fraction,
            label_dir=label_dir,
            augment=augment,
            hsv_h=self.args.hsv_h if augment else 0.0,
            hsv_s=self.args.hsv_s if augment else 0.0,
            hsv_v=self.args.hsv_v if augment else 0.0,
            fliplr=self.args.fliplr if augment else 0.0,
            flipud=self.args.flipud if augment else 0.0,
            scale=self.args.scale if augment else 0.0,
            erasing=self.args.erasing if augment else 0.0,
            mosaic=self.args.mosaic if augment else 0.0,
        )
        assert_gcs_shape(dataset.imgsz, gcs_imgsz, name=f"{mode} dataset.imgsz", context="GCSLaneTrainer.build_dataset")
        self._check_point_mode_contract(dataset, mode=mode)
        return dataset

    def _head_point_mode(self) -> str | None:
        """Return the GCS head point mode after the model has been constructed."""
        model = getattr(self, "model", None)
        if model is None:
            return None
        model = getattr(model, "module", model)
        for module in model.modules():
            if isinstance(module, GCSLaneHead):
                mode = str(getattr(module, "point_mode", "free")).lower()
                return "fixed_y" if mode in {"fixed-y", "fixedy"} else mode
        return None

    def _check_point_mode_contract(self, dataset: GCSLaneDataset, mode: str) -> None:
        """Fail fast when a fixed-y head is paired with free Kx2 labels, or vice versa."""
        head_mode = self._head_point_mode()
        if head_mode is None:
            return
        data_mode = str(getattr(dataset, "point_mode", "free")).lower()
        if data_mode in {"fixed-y", "fixedy"}:
            data_mode = "fixed_y"
        if data_mode != head_mode:
            raise ValueError(
                f"GCS point-mode mismatch for {mode}: model GCSLaneHead point_mode={head_mode!r}, "
                f"but labels under {dataset.label_files[0].parent} are point_mode={data_mode!r}. "
                "Use fixed-y labels generated with tools/convert_tusimple_to_gcs.py --point-mode fixed_y "
                "for the fixed-y x-only head."
            )

    @staticmethod
    def _label_lane_count(label_file: Path) -> int:
        """Read the number of valid GT lanes from one GCS npz label."""
        with np.load(label_file, allow_pickle=False) as data:
            if "num_lanes" in data:
                return int(np.asarray(data["num_lanes"]).reshape(-1)[0])
            if "lane_valid" in data:
                return int((data["lane_valid"].sum(axis=1) >= 2).sum())
            return int(data["lanes"].shape[0])

    @staticmethod
    def _array_scalar_str(value: np.ndarray) -> str:
        """Convert a scalar-like npz array to a Python string."""
        value = np.asarray(value)
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
        return str(value)

    @staticmethod
    def _parse_tusimple_date(raw_file: str, image_file: Path) -> str:
        """Parse TuSimple date id from raw_file first, then converted image paths."""
        raw = str(raw_file or "").lstrip("/").replace("\\", "/")
        parts = Path(raw).parts
        if len(parts) >= 2 and parts[0] == "clips":
            return parts[1]

        image_text = str(image_file).replace("\\", "/")
        for text in (raw, image_text):
            for part in Path(text).parts:
                if part in {"0313-1", "0313-2", "0531", "0601"}:
                    return part
            match = re.search(r"(0313-[12]|0531|0601)", text)
            if match:
                return match.group(1)
        return "unknown"

    @classmethod
    def _hard_sampling_meta(cls, label_file: Path, image_file: Path) -> tuple[int, float, str]:
        """Read GT lane count, shortest visible lane length, and TuSimple date for one sample."""
        with np.load(label_file, allow_pickle=False) as data:
            if "lanes" not in data:
                raise KeyError(f"{label_file} is missing required hard-sampling array 'lanes'.")
            if "lane_valid" not in data:
                raise KeyError(f"{label_file} is missing required hard-sampling array 'lane_valid'.")
            lanes = np.asarray(data["lanes"])
            lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
            if lane_valid.ndim != 2 or lanes.ndim < 1 or lane_valid.shape[0] != lanes.shape[0]:
                raise ValueError(f"{label_file}: hard-sampling lanes/lane_valid shape mismatch.")
            if "num_lanes" in data:
                gt_lanes = int(np.asarray(data["num_lanes"]).reshape(-1)[0])
            else:
                gt_lanes = int(lane_valid.shape[0])
            min_visible = float(lane_valid.sum(axis=1).min()) if lane_valid.shape[0] else 0.0
            raw_file = cls._array_scalar_str(data["raw_file"]) if "raw_file" in data else ""
        return gt_lanes, min_visible, cls._parse_tusimple_date(raw_file, image_file)

    def _lane_count_sampler(self, dataset: GCSLaneDataset) -> WeightedRandomSampler:
        """Build a replacement sampler that balances samples by GT lane count."""
        counts = [self._label_lane_count(Path(p)) for p in dataset.label_files]
        hist = Counter(counts)
        power = float(getattr(self.args, "gcs_lane_count_balance_power", 1.0))
        min_group = max(int(getattr(self.args, "gcs_lane_count_min_group", 50) or 0), 1)
        weights = torch.as_tensor([1.0 / (float(max(hist[c], min_group)) ** power) for c in counts], dtype=torch.double)
        LOGGER.info(
            "GCS lane-count balanced sampling enabled: "
            f"hist={dict(sorted(hist.items()))}, power={power:g}, min_group={min_group}, "
            f"samples_per_epoch={len(weights)}"
        )
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def _hard_sampling_sampler(self, dataset: GCSLaneDataset) -> WeightedRandomSampler:
        """Build a train-only sampler for short visible side-lane and 0601 samples."""
        date_0601_weight = float(getattr(self.args, "gcs_hard_date_0601_weight", 2.0))
        gt4_weight = float(getattr(self.args, "gcs_hard_gt4_le10_weight", 4.0))
        gt5_weight = float(getattr(self.args, "gcs_hard_gt5_le10_weight", 3.0))
        gt3_weight = float(getattr(self.args, "gcs_hard_gt3_le20_weight", 1.5))
        gt4_0313_2_weight = float(getattr(self.args, "gcs_hard_0313_2_gt4_le10_weight", 4.0))
        visible_thr = int(getattr(self.args, "gcs_hard_visible_thr", 10))
        gt3_visible_thr = int(getattr(self.args, "gcs_hard_gt3_visible_thr", 20))

        weights_cfg = {
            "gcs_hard_date_0601_weight": date_0601_weight,
            "gcs_hard_gt4_le10_weight": gt4_weight,
            "gcs_hard_gt5_le10_weight": gt5_weight,
            "gcs_hard_gt3_le20_weight": gt3_weight,
            "gcs_hard_0313_2_gt4_le10_weight": gt4_0313_2_weight,
        }
        bad_weights = {k: v for k, v in weights_cfg.items() if v <= 0.0 or not math.isfinite(v)}
        if bad_weights:
            raise ValueError(f"GCS hard sampling weights must be positive finite values, got {bad_weights}.")
        if visible_thr < 0 or gt3_visible_thr < 0:
            raise ValueError(
                f"gcs_hard_visible_thr and gcs_hard_gt3_visible_thr must be >= 0, "
                f"got {visible_thr} and {gt3_visible_thr}."
            )

        stats = Counter()
        sample_weights: list[float] = []
        for label_file, image_file in zip(dataset.label_files, dataset.im_files):
            gt_lanes, min_visible, date = self._hard_sampling_meta(Path(label_file), Path(image_file))
            weight = 1.0
            if date == "0601":
                weight *= date_0601_weight
                stats["date_0601"] += 1
            if gt_lanes == 4 and min_visible <= visible_thr:
                weight *= gt4_weight
                stats[f"gt4_le{visible_thr}"] += 1
            if gt_lanes == 5 and min_visible <= visible_thr:
                weight *= gt5_weight
                stats[f"gt5_le{visible_thr}"] += 1
            if gt_lanes == 3 and min_visible <= gt3_visible_thr:
                weight *= gt3_weight
                stats[f"gt3_le{gt3_visible_thr}"] += 1
            if date == "0313-2" and gt_lanes == 4 and min_visible <= visible_thr:
                weight *= gt4_0313_2_weight
                stats[f"0313-2_gt4_le{visible_thr}"] += 1
            sample_weights.append(weight)

        weights = torch.as_tensor(sample_weights, dtype=torch.double)
        LOGGER.info(
            "GCS hard_sampling enabled: "
            f"num_date_0601={stats['date_0601']}, "
            f"num_gt4_le{visible_thr}={stats[f'gt4_le{visible_thr}']}, "
            f"num_gt5_le{visible_thr}={stats[f'gt5_le{visible_thr}']}, "
            f"num_gt3_le{gt3_visible_thr}={stats[f'gt3_le{gt3_visible_thr}']}, "
            f"num_0313-2_gt4_le{visible_thr}={stats[f'0313-2_gt4_le{visible_thr}']}, "
            f"weight_min={float(weights.min()):.6g}, weight_mean={float(weights.mean()):.6g}, "
            f"weight_max={float(weights.max()):.6g}, samples_per_epoch={len(weights)}"
        )
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Create a dataloader for variable-lane GCS labels."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        if mode == "val" and not self.args.val:
            return None
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        sampler = None
        shuffle = mode == "train"
        hard_sampling = mode == "train" and bool(getattr(self.args, "gcs_hard_sampling", False))
        count_balanced = mode == "train" and bool(getattr(self.args, "gcs_lane_count_balanced", False))
        use_count_balanced = count_balanced and not hard_sampling and rank == -1
        if mode == "train":
            LOGGER.info(f"GCS lane-count-balanced sampling: {str(use_count_balanced).lower()}")
        if hard_sampling:
            if rank != -1:
                LOGGER.warning("GCS hard sampling is only enabled for single-process training.")
            else:
                if count_balanced:
                    LOGGER.info("GCS hard sampling overrides gcs_lane_count_balanced for this train dataloader.")
                sampler = self._hard_sampling_sampler(dataset)
                shuffle = False
        elif count_balanced:
            if rank != -1:
                LOGGER.warning("GCS lane-count balanced sampling is only enabled for single-process training.")
            else:
                sampler = self._lane_count_sampler(dataset)
                shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            sampler=sampler,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Move GCS batches to device and normalize images."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["lanes"] = [x.to(self.device, non_blocking=self.device.type == "cuda") for x in batch["lanes"]]
        batch["lane_valid"] = [x.to(self.device, non_blocking=self.device.type == "cuda") for x in batch["lane_valid"]]
        batch["gt_lanes"] = batch["lanes"]
        batch["gt_lane_valid"] = batch["lane_valid"]
        batch["edge_mask"] = batch["edge_mask"].float()
        batch["img"] = batch["img"].float() / 255
        gcs_imgsz = self._resolve_gcs_imgsz()
        assert_gcs_image_tensor(batch["img"], gcs_imgsz, name="batch['img']", context="GCSLaneTrainer.preprocess_batch")
        assert_gcs_shape(
            batch["semantic_mask"].shape[-2:],
            gcs_imgsz,
            name="batch['semantic_mask']",
            context="GCSLaneTrainer.preprocess_batch",
        )
        assert_gcs_shape(
            batch["edge_mask"].shape[-2:],
            gcs_imgsz,
            name="batch['edge_mask']",
            context="GCSLaneTrainer.preprocess_batch",
        )

        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            base_imgsz = max(self._resolve_gcs_imgsz())
            sz = (
                random.randrange(
                    max(self.stride, int(base_imgsz * (1.0 - self.args.multi_scale))),
                    int(base_imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                batch["img"] = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
                batch = resize_gcs_masks(batch, size=tuple(ns))
        return batch

    def set_model_attributes(self):
        """Attach GCS lane task attributes to the model."""
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.gcs_imgsz = self._resolve_gcs_imgsz()
        self.model.args = self.args
        self.model.task = "gcs_lane"

    def set_class_weights(self):
        """GCS lane training uses existence loss, not class-frequency weights."""
        return None

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a GCS lane model with GCSLoss wiring."""
        model = GCSLaneModel(cfg, nc=self.data["nc"], ch=self.data.get("channels", 3), verbose=verbose and RANK == -1)
        self._assert_model_gcs_mode(model)
        self._warn_if_ordered_slot_without_official_best()
        self._record_ordered_slot_loss_contract()
        self._rewrite_args_yaml_after_gcs_mode_sync()
        self._set_loss_names_for_mode()
        if weights is not None:
            self.load_gcs_pretrained(model, weights)
        return model

    @staticmethod
    def _checkpoint_payload(weights: str | Path | dict | nn.Module):
        """Load a checkpoint payload without requiring ckpt['model'] for state_dict-only files."""
        if isinstance(weights, (str, Path)):
            weights, _ = torch_safe_load(weights)
        return weights

    @classmethod
    def _state_dict_and_meta_from_weights(cls, weights: str | Path | dict | nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Extract a plain state_dict plus metadata from a path, checkpoint dict, raw state_dict, or module."""
        weights = cls._checkpoint_payload(weights)
        meta: dict[str, Any] = {}
        if isinstance(weights, dict):
            if isinstance(weights.get("meta"), dict):
                meta.update(weights["meta"])
            if isinstance(weights.get("train_args"), dict):
                meta.update({f"train_args_{k}": v for k, v in weights["train_args"].items()})
                if weights["train_args"].get("gcs_mode") is not None:
                    meta.setdefault("gcs_mode", weights["train_args"]["gcs_mode"])
            weights = weights.get("ema") or weights.get("model") or weights.get("state_dict") or weights

        if isinstance(weights, nn.Module):
            if getattr(weights, "gcs_mode", None) is not None:
                meta.setdefault("gcs_mode", getattr(weights, "gcs_mode"))
            if isinstance(getattr(weights, "yaml", None), dict) and weights.yaml.get("gcs_mode") is not None:
                meta.setdefault("gcs_mode", weights.yaml.get("gcs_mode"))
            args = getattr(weights, "args", None)
            if isinstance(args, dict) and args.get("gcs_mode") is not None:
                meta.setdefault("gcs_mode", args.get("gcs_mode"))
            elif getattr(args, "gcs_mode", None) is not None:
                meta.setdefault("gcs_mode", getattr(args, "gcs_mode"))
            for module in weights.modules():
                if isinstance(module, GCSLaneHead):
                    meta.setdefault("gcs_mode", getattr(module, "gcs_mode", None))
                    break
            state = weights.float().state_dict()
        elif isinstance(weights, dict):
            state = weights
        else:
            raise TypeError(f"Unsupported pretrained weights type for GCSLaneTrainer: {type(weights).__name__}")

        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
        return state, meta

    @classmethod
    def _state_dict_from_weights(cls, weights: str | Path | dict | nn.Module) -> dict[str, torch.Tensor]:
        """Extract a plain state_dict from a checkpoint path, checkpoint dict, or loaded module."""
        return cls._state_dict_and_meta_from_weights(weights)[0]

    @staticmethod
    def _gcs_module_prefixes(model: nn.Module) -> tuple[str, ...]:
        """Return parameter prefixes for GCS-specific modules that must stay randomly initialized."""
        prefixes = []
        for name, module in model.named_modules():
            if name and isinstance(module, (LSEM, LaneBiFPN, GCSLaneHead)):
                prefixes.append(f"{name}.")
        return tuple(prefixes)

    @staticmethod
    def _gcs_head_prefixes(model: nn.Module) -> tuple[str, ...]:
        """Return parameter prefixes for the GCS lane head only."""
        return tuple(f"{name}." for name, module in model.named_modules() if name and isinstance(module, GCSLaneHead))

    @staticmethod
    def _state_dict_has_gcs_modules(state: dict[str, torch.Tensor]) -> bool:
        """Return True when a checkpoint already contains GCS-specific module tensors."""
        markers = (
            ".lsa.",
            ".dilated_context.",
            ".level_embed",
            ".query_embed.",
            ".decoder.",
            ".point_mlp.",
            ".point_valid_mlp.",
            ".point_valid_refine_mlp.",
            ".exist_mlp.",
            ".aux_mask.",
            ".aux_edge.",
            ".p2_in.",
            ".fuse_p",
        )
        return any(any(marker in key for marker in markers) for key in state)

    @classmethod
    def _infer_source_gcs_mode(cls, state: dict[str, torch.Tensor], meta: dict[str, Any] | None = None) -> str:
        """Infer source checkpoint GCS mode from metadata or ordered-slot head keys."""
        meta = meta or {}
        for key in ("gcs_mode", "source_gcs_mode"):
            if meta.get(key) is not None:
                return cls._normalize_gcs_mode(meta[key])
        ordered_markers = (
            ".start_mlp.",
            ".end_mlp.",
            ".count_mlp.",
            ".start_head.",
            ".end_head.",
            ".count_head.",
        )
        if any(any(marker in key for marker in ordered_markers) for key in state):
            return "ordered_slot"
        return "query"

    @classmethod
    def remap_yolo11_backbone_to_gcs(cls, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remap a standard YOLO11 checkpoint backbone into the GCS-YOLO-Lane backbone.

        The final GCS network does not reuse the ordinary YOLO PAN/FPN or Detect
        head. It inherits only the YOLO11 backbone tensors and leaves LSEM,
        Lane-BiFPN, and GCSLaneHead initialized by the GCS model itself.
        """
        remapped = {}
        pattern = re.compile(r"^model\.(\d+)\.(.+)$")
        for key, value in state.items():
            match = pattern.match(key)
            if not match:
                continue
            old_idx = int(match.group(1))
            new_idx = cls.yolo11_to_gcs_backbone.get(old_idx)
            if new_idx is None:
                continue
            remapped[f"model.{new_idx}.{match.group(2)}"] = value
        return remapped

    def load_gcs_pretrained(self, model: nn.Module, weights: str | Path | dict | nn.Module) -> None:
        """Load pretrained weights according to the GCS-YOLO-Lane inheritance rule.

        For ordinary YOLO11 checkpoints, only the backbone is inherited and the
        shifted layer indices caused by inserted LSEM blocks are remapped:
        old 0-4 -> new 0-4, old 5-6 -> new 6-7, old 7-10 -> new 9-12.
        GCS-specific modules are never copied from an ordinary YOLO checkpoint.
        """
        source_state, source_meta = self._state_dict_and_meta_from_weights(weights)
        target_state = model.state_dict()
        source_is_gcs = self._state_dict_has_gcs_modules(source_state)
        candidate_state = source_state if source_is_gcs else self.remap_yolo11_backbone_to_gcs(source_state)
        target_head = model.model[-1] if getattr(model, "model", None) is not None and len(model.model) else None
        target_mode = self._normalize_gcs_mode(getattr(target_head, "gcs_mode", getattr(model, "gcs_mode", "query")))
        source_mode = self._infer_source_gcs_mode(source_state, source_meta) if source_is_gcs else "query"
        gcs_prefixes = self._gcs_head_prefixes(model) if source_is_gcs and source_mode != target_mode else (
            () if source_is_gcs else self._gcs_module_prefixes(model)
        )

        loadable = {}
        skipped_gcs = 0
        skipped_shape = 0
        skipped_missing = 0
        for key, value in candidate_state.items():
            if key not in target_state:
                skipped_missing += 1
                continue
            if key.startswith(gcs_prefixes):
                skipped_gcs += 1
                continue
            if value.shape != target_state[key].shape:
                skipped_shape += 1
                continue
            loadable[key] = value.to(dtype=target_state[key].dtype)

        model.load_state_dict(loadable, strict=False)
        source_kind = (
            f"GCS-{source_mode}-to-{target_mode}"
            if source_is_gcs
            else "YOLO11-backbone-remap"
        )
        LOGGER.info(
            f"GCS pretrained transfer ({source_kind}): loaded {len(loadable)}/{len(target_state)} tensors "
            f"(candidates={len(candidate_state)}, skipped_missing={skipped_missing}, "
            f"skipped_gcs={skipped_gcs}, skipped_shape={skipped_shape})"
        )
        if not loadable:
            LOGGER.warning(
                "No pretrained tensors were transferred. Check that the weight file is a YOLO11/YOLO11-seg "
                "checkpoint with the same scale as the GCS YAML, e.g. yolo11s-seg.pt for gcs-yolo-lane-s.yaml."
            )

    def setup_model(self):
        """Build the GCS model before applying flexible pretrained transfer."""
        if isinstance(self.model, torch.nn.Module):
            return

        cfg, weights = self.model, None
        ckpt = None
        if self.resume:
            resume_path = Path(getattr(self.args, "resume", self.model))
            weights, ckpt = load_checkpoint(resume_path)
            cfg = weights.yaml
            self.args.pretrained = False
            model = self.get_model(cfg=cfg, weights=None, verbose=RANK in {-1, 0})
            ckpt_mode = infer_gcs_mode_from_ckpt(ckpt)
            source_mode = infer_gcs_mode_from_model(weights)
            target_mode = infer_gcs_mode_from_model(model)
            if ckpt_mode != source_mode:
                raise RuntimeError(
                    f"Resume checkpoint mode metadata mismatch: ckpt={ckpt_mode!r}, model={source_mode!r}."
                )
            if source_mode != target_mode:
                raise RuntimeError(
                    f"Resume checkpoint mode {source_mode!r} does not match target model mode {target_mode!r}."
                )
            source_model = ckpt.get("ema") or ckpt.get("model")
            if source_model is None:
                raise RuntimeError(f"Resume checkpoint {resume_path} has no model or ema weights.")
            model.load_state_dict(source_model.float().state_dict(), strict=True)
            self.model = model
            self._set_args_gcs_mode(target_mode)
            LOGGER.info(f"Resume enabled: loaded model weights from {resume_path} and disabled args.pretrained.")
            return ckpt
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        if isinstance(self.args.pretrained, (str, Path)):
            weights = self.args.pretrained
        elif self.args.pretrained is False and not self.resume:
            weights = None
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK in {-1, 0})
        return ckpt

    def get_validator(self):
        """Return a loss-based validator for structured lane training."""
        self._set_loss_names_for_mode()
        return GCSLaneValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def validate(self):
        """Respect val=False even on the final epoch forced by BaseTrainer."""
        if not self.args.val:
            return {}, None
        return super().validate()

    def final_eval(self):
        """Strip checkpoints but skip final best.pt validation when val=False."""
        if self.args.val:
            return super().final_eval()

        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        LOGGER.info("Skipping final validation because val=False.")

    @staticmethod
    def _official_float_list(value: Any, default: tuple[float, ...]) -> list[float]:
        """Normalize list-like official sweep float args from YAML, CLI, or resume metadata."""
        if value is None or value == "":
            return [float(x) for x in default]
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                value = ast.literal_eval(text)
            else:
                value = [x for x in re.split(r"[\s,;]+", text) if x]
        if isinstance(value, (int, float)):
            value = [value]
        return [float(x) for x in value]

    @staticmethod
    def _official_int_list(value: Any, default: tuple[int, ...]) -> list[int]:
        """Normalize list-like official sweep int args from YAML, CLI, or resume metadata."""
        if value is None or value == "":
            return [int(x) for x in default]
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                value = ast.literal_eval(text)
            else:
                value = [x for x in re.split(r"[\s,;]+", text) if x]
        if isinstance(value, int):
            value = [value]
        return [int(x) for x in value]

    @staticmethod
    def _official_best_key(best: dict[str, Any], epoch: int) -> tuple:
        """Order official-val candidates by the project checkpoint-selection contract."""
        return official_best_sort_key(best, epoch)

    def _load_official_best_state(self) -> dict[str, Any] | None:
        """Restore official-best comparison state when resuming a run."""
        if not self.official_best_sweep.exists():
            return None
        try:
            data = json.loads(self.official_best_sweep.read_text(encoding="utf-8"))
            meta = data.get("official_best", {})
            best = meta.get("best") or data.get("best")
            epoch = int(meta.get("epoch", data.get("selected_epoch", 0)) or 0)
            if isinstance(best, dict):
                return {"best": best, "epoch": epoch, "key": self._official_best_key(best, epoch)}
        except Exception as exc:
            LOGGER.warning(f"Could not restore existing official_best state from {self.official_best_sweep}: {exc}")
        return None

    def _should_run_official_best(self) -> bool:
        """Return True when this epoch should run training-time official-val selection."""
        if not bool(getattr(self.args, "gcs_official_best", False)):
            return False
        interval = int(getattr(self.args, "gcs_official_interval", 5) or 0)
        if interval <= 0:
            raise ValueError(f"gcs_official_interval must be > 0 when gcs_official_best=True, got {interval}.")
        epoch_num = int(self.epoch) + 1
        return (epoch_num % interval == 0) or bool(getattr(self, "stop", False)) or epoch_num >= int(self.epochs)

    def _official_device_arg(self) -> str:
        """Return a stable device string for the in-process official sweep."""
        value = getattr(self.args, "device", None)
        if value is not None and str(value).strip() and str(value).strip().lower() != "none":
            return str(value)
        if self.device.type == "cuda":
            return str(0 if self.device.index is None else self.device.index)
        return str(self.device)

    def _official_sweep_args(self, save_dir: Path) -> SimpleNamespace:
        """Build the in-process official-val sweep args for the current checkpoint."""
        shape = self._resolve_gcs_imgsz()
        ordered_slot = self._gcs_mode() == "ordered_slot"
        official_query_defaults = {
            "gcs_official_confs": [0.005, 0.01, 0.02, 0.05, 0.1],
            "gcs_official_point_valid_thrs": [0.45, 0.5],
            "gcs_official_nms_dist_pxs": [0.0, 18.0, 30.0, 50.0],
            "gcs_official_max_dets": [5, 6, 8],
            "gcs_official_min_points": [4, 5, 6],
            "gcs_official_valid_before_maxdet": False,
        }
        if ordered_slot:
            sweep_arg_values = {
                name: getattr(self.args, name, default)
                for name, default in official_query_defaults.items()
            }
            raise_for_ordered_slot_query_args(
                sweep_arg_values,
                official_query_defaults,
                context="training-time official_best sweep",
            )
            confs = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
            point_valid_thrs = [0.30, 0.35, 0.40, 0.45, 0.50]
            nms_dist_pxs = [18.0]
            max_dets = [8]
            min_points = [6]
        else:
            confs = self._official_float_list(
                getattr(self.args, "gcs_official_confs", None), official_query_defaults["gcs_official_confs"]
            )
            point_valid_thrs = self._official_float_list(
                getattr(self.args, "gcs_official_point_valid_thrs", None),
                official_query_defaults["gcs_official_point_valid_thrs"],
            )
            nms_dist_pxs = self._official_float_list(
                getattr(self.args, "gcs_official_nms_dist_pxs", None),
                official_query_defaults["gcs_official_nms_dist_pxs"],
            )
            max_dets = self._official_int_list(
                getattr(self.args, "gcs_official_max_dets", None), official_query_defaults["gcs_official_max_dets"]
            )
            min_points = self._official_int_list(
                getattr(self.args, "gcs_official_min_points", None), official_query_defaults["gcs_official_min_points"]
            )
        return SimpleNamespace(
            dataset="tusimple",
            archive_root=str(getattr(self.args, "gcs_official_archive_root", "archive") or "archive"),
            split="val",
            gt_json=getattr(self.args, "gcs_official_gt_json", None),
            allow_noncanonical_gt=bool(getattr(self.args, "gcs_official_allow_noncanonical_gt", False)),
            weights=str(self.last),
            imgsz=[int(shape[0]), int(shape[1])],
            decode_mode="ordered_slot" if ordered_slot else "query",
            confs=confs,
            point_valid_thrs=point_valid_thrs,
            nms_dist_pxs=nms_dist_pxs,
            max_dets=max_dets,
            min_points=min_points,
            gcs_min_lanes=int(getattr(self.args, "gcs_min_lanes", 2)),
            gcs_max_lanes=int(getattr(self.args, "gcs_max_lanes", 5)),
            gcs_num_slots=int(getattr(self.args, "gcs_num_slots", 5)),
            gcs_min_interval_points=int(getattr(self.args, "gcs_min_interval_points", 2)),
            gcs_bottom_order_margin_px=float(getattr(self.args, "gcs_bottom_order_margin_px", 2.0)),
            valid_before_maxdet=bool(getattr(self.args, "gcs_official_valid_before_maxdet", False)),
            max_images=int(getattr(self.args, "gcs_official_max_images", 0) or 0),
            warmup=int(getattr(self.args, "gcs_official_warmup", 5) or 0),
            device=self._official_device_arg(),
            half=bool(getattr(self.args, "gcs_official_half", False)),
            runtime_ms=1.0,
            save_dir=str(save_dir),
            ordered_slot_runtime_context="training_official_best" if ordered_slot else "official_sweep",
            score_fp_weight=float(getattr(self.args, "gcs_official_score_fp_weight", 0.02) or 0.02),
            score_fn_weight=float(getattr(self.args, "gcs_official_score_fn_weight", 0.02) or 0.02),
        )

    def _write_official_best_artifacts(self, output: dict[str, Any], epoch_num: int, sweep_dir: Path) -> None:
        """Persist official-best checkpoint, sweep summary, and decode config."""
        best = dict(output["best"])
        summary = dict(output)
        meta = {
            "epoch": int(epoch_num),
            "checkpoint": str(self.official_best.resolve()),
            "source_checkpoint": str(self.last.resolve()),
            "sweep_dir": str(sweep_dir.resolve()),
            "selection_policy": OFFICIAL_SELECTION_POLICY,
            "sweep_selection_policy": output.get("selection_policy") or output.get("config", {}).get("selection_policy"),
            "best": best,
        }
        summary["official_best"] = meta
        summary["selection_policy"] = OFFICIAL_SELECTION_POLICY
        summary["official_selection_policy"] = OFFICIAL_SELECTION_POLICY
        summary["sweep_selection_policy"] = meta["sweep_selection_policy"]
        shutil.copy2(self.last, self.official_best)
        self.official_best_sweep.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        model_mode = str(best.get("decode_mode", self._gcs_mode()))
        decode_cfg = build_official_best_decode_cfg(best, model_mode=model_mode, args=self.args)
        YAML.save(
            self.official_best_decode,
            {
                "weights": str(self.official_best.resolve()),
                "source_epoch": int(epoch_num),
                "sweep_summary": str(self.official_best_sweep.resolve()),
                "selection_policy": meta["selection_policy"],
                "sweep_selection_policy": meta["sweep_selection_policy"],
                "decode": decode_cfg,
                "official_metrics": {
                    "official_acc": float(best["official_acc"]),
                    "official_score": float(best["official_score"]),
                    "official_FP": float(best["official_FP"]),
                    "official_FN": float(best["official_FN"]),
                    "strict_order_valid": bool(best.get("strict_order_valid", True)),
                    "ordered_slot_order_violations": int(best.get("ordered_slot_order_violations", 0)),
                    "count_acc": float(best.get("count_acc", 0.0)),
                    "count_acc_2": float(best.get("count_acc_2", 0.0)),
                    "count_acc_3": float(best.get("count_acc_3", 0.0)),
                    "count_acc_4": float(best.get("count_acc_4", 0.0)),
                    "count_acc_5": float(best.get("count_acc_5", 0.0)),
                },
            },
        )
        self._official_best_state = {"best": best, "epoch": int(epoch_num), "key": self._official_best_key(best, epoch_num)}

    def _maybe_update_official_best(self) -> None:
        """Run periodic official-val sweep and update official_best when the official metric improves."""
        if RANK not in {-1, 0} or not self._should_run_official_best():
            return
        if not self.last.exists():
            raise FileNotFoundError(f"Cannot run official-best selection because {self.last} does not exist.")

        from tools.sweep_tusimple_official import sweep

        epoch_num = int(self.epoch) + 1
        sweep_dir = self.save_dir / "official_sweeps" / f"epoch{epoch_num:03d}"
        LOGGER.info(f"Running TuSimple official-val sweep for checkpoint selection at epoch {epoch_num}...")
        output = sweep(self._official_sweep_args(sweep_dir))
        best = dict(output["best"])
        new_key = self._official_best_key(best, epoch_num)
        old_key = self._official_best_state.get("key") if self._official_best_state else None
        if old_key is None or new_key > old_key:
            self._write_official_best_artifacts(output, epoch_num=epoch_num, sweep_dir=sweep_dir)
            LOGGER.info(
                "Updated official_best.pt: "
                f"epoch={epoch_num}, official_acc={float(best['official_acc']):.6f}, "
                f"official_score={float(best['official_score']):.6f}, "
                f"FP={float(best['official_FP']):.6f}, FN={float(best['official_FN']):.6f}, "
                f"count_acc_4={float(best.get('count_acc_4', 0.0)):.6f}, "
                f"strict_order_valid={bool(best.get('strict_order_valid', True))}, "
                f"order_violations={int(best.get('ordered_slot_order_violations', 0))}"
            )
        else:
            current = self._official_best_state["best"]
            LOGGER.info(
                "Kept existing official_best.pt: "
                f"epoch={self._official_best_state['epoch']}, official_acc={float(current['official_acc']):.6f}, "
                f"new_epoch={epoch_num}, new_official_acc={float(best['official_acc']):.6f}"
            )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def save_model(self):
        """Save checkpoints with explicit rectangular GCS imgsz in train_args."""
        old_imgsz = self.args.imgsz
        shape = self._resolve_gcs_imgsz()
        self.args.gcs_imgsz = [int(shape[0]), int(shape[1])]
        self.args.imgsz = max(int(shape[0]), int(shape[1]))
        try:
            saved = bool(super().save_model())
        finally:
            self.args.imgsz = old_imgsz
        if saved:
            self._maybe_update_official_best()
        return saved

    def label_loss_items(self, loss_items: list[float] | torch.Tensor | None = None, prefix: str = "train"):
        """Return named GCS loss items for logging."""
        if tuple(self.loss_names) == tuple(OrderedSlotGCSLoss.loss_names):
            return OrderedSlotGCSLoss.label_loss_items(loss_items, prefix=prefix)
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        return dict(zip(keys, [round(float(x), 5) for x in loss_items]))

    def progress_string(self):
        """Return a progress header matching the GCS loss vector."""
        return ("\n" + "%11s" * (4 + len(self.progress_loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.progress_loss_names,
            "Lanes",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Skip generic YOLO box plotting for GCS lane batches."""
        return None

    def plot_training_labels(self):
        """Skip generic YOLO label plotting for GCS lane labels."""
        return None

    def auto_batch(self):
        """Estimate batch size using a small fixed lane count proxy."""
        with torch.no_grad():
            n = len(self.train_loader.dataset) if hasattr(self, "train_loader") else 1
        return super().auto_batch(max_num_obj=8, dataset_size=n)
