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

from ultralytics.data import build_dataloader
from ultralytics.data.dataset_gcs import GCSLaneDataset, resize_gcs_masks
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.gcs_lane.val import GCSLaneValidator
from ultralytics.nn.modules import GCSLaneHead, LaneBiFPN, LSEM
from ultralytics.nn.tasks import GCSLaneModel, load_checkpoint
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
        "spurious_neg_loss",
        "spurious_negative_count",
        "count_score_mean",
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
        "spur_neg",
        "spur_cnt",
        "cnt_score",
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

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize the GCS lane trainer."""
        overrides = dict(overrides or {})
        overrides["task"] = "gcs_lane"
        overrides.setdefault("model", str(ROOT / "cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml"))
        overrides.setdefault("data", str(ROOT.parent / "data/tusimple_gcs_fixed_y_k56_960x544.yaml"))
        # GCSLaneHead defaults to 8 lane queries. Four-image mosaic can raise TuSimple GT lanes to ~16,
        # leaving unmatched GT lanes outside the structured point loss, so keep mosaic off unless requested.
        overrides.setdefault("mosaic", 0.0)
        # Plain GCS training historically used no affine erasing/scale aug; keep direct trainer construction stable.
        overrides.setdefault("scale", 0.0)
        overrides.setdefault("erasing", 0.0)
        super().__init__(cfg, overrides, _callbacks)
        self.official_best = self.wdir / "official_best.pt"
        self.official_best_sweep = self.wdir / "official_best_sweep.json"
        self.official_best_decode = self.wdir / "official_best_decode.yaml"
        self._official_best_state = self._load_official_best_state()
        self._lock_gcs_shape_contract()

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
        if mode == "train" and bool(getattr(self.args, "gcs_hard_sampling", False)):
            if rank != -1:
                LOGGER.warning("GCS hard sampling is only enabled for single-process training.")
            else:
                if bool(getattr(self.args, "gcs_lane_count_balanced", False)):
                    LOGGER.info("GCS hard sampling overrides gcs_lane_count_balanced for this train dataloader.")
                sampler = self._hard_sampling_sampler(dataset)
                shuffle = False
        elif mode == "train" and bool(getattr(self.args, "gcs_lane_count_balanced", False)):
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
        if weights is not None:
            self.load_gcs_pretrained(model, weights)
        return model

    @staticmethod
    def _state_dict_from_weights(weights: str | Path | dict | nn.Module) -> dict[str, torch.Tensor]:
        """Extract a plain state_dict from a checkpoint path, checkpoint dict, or loaded module."""
        if isinstance(weights, (str, Path)):
            weights, _ = load_checkpoint(weights)

        if isinstance(weights, dict):
            weights = weights.get("ema") or weights.get("model") or weights.get("state_dict") or weights

        if isinstance(weights, nn.Module):
            state = weights.float().state_dict()
        elif isinstance(weights, dict):
            state = weights
        else:
            raise TypeError(f"Unsupported pretrained weights type for GCSLaneTrainer: {type(weights).__name__}")

        return {k[7:] if k.startswith("module.") else k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

    @staticmethod
    def _gcs_module_prefixes(model: nn.Module) -> tuple[str, ...]:
        """Return parameter prefixes for GCS-specific modules that must stay randomly initialized."""
        prefixes = []
        for name, module in model.named_modules():
            if name and isinstance(module, (LSEM, LaneBiFPN, GCSLaneHead)):
                prefixes.append(f"{name}.")
        return tuple(prefixes)

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
        source_state = self._state_dict_from_weights(weights)
        target_state = model.state_dict()
        source_is_gcs = self._state_dict_has_gcs_modules(source_state)
        candidate_state = source_state if source_is_gcs else self.remap_yolo11_backbone_to_gcs(source_state)
        gcs_prefixes = () if source_is_gcs else self._gcs_module_prefixes(model)

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
        source_kind = "GCS" if source_is_gcs else "YOLO11-backbone-remap"
        LOGGER.info(
            f"GCS pretrained transfer ({source_kind}): loaded {len(loadable)}/{len(target_state)} tensors "
            f"(candidates={len(candidate_state)}, skipped_missing={skipped_missing}, "
            f"skipped_gcs={skipped_gcs}, skipped_shape={skipped_shape})"
        )
        if not loadable:
            LOGGER.warning(
                "No pretrained tensors were transferred. Check that the weight file is a YOLO11/YOLO11-seg "
                "checkpoint with the same scale as the GCS YAML, e.g. yolo11s-seg.pt for gcs-yolo-lane-s-q12-k56.yaml."
            )

    def get_validator(self):
        """Return a loss-based validator for structured lane training."""
        self.loss_names = self.__class__.loss_names
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
    def _official_best_key(best: dict[str, Any], epoch: int) -> tuple[float, float, float, float, float, float, float, int]:
        """Order official-val candidates by the project checkpoint-selection contract."""

        def f(name: str, default: float = 0.0) -> float:
            value = best.get(name, default)
            return float(default if value is None else value)

        return (
            f("official_acc"),
            f("official_score"),
            -f("official_FP"),
            -f("official_FN"),
            f("count_acc_4"),
            f("count_acc"),
            f("count_acc_5"),
            -int(epoch),
        )

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
        return SimpleNamespace(
            dataset="tusimple",
            archive_root=str(getattr(self.args, "gcs_official_archive_root", "archive") or "archive"),
            split="val",
            gt_json=getattr(self.args, "gcs_official_gt_json", None),
            weights=str(self.last),
            imgsz=[int(shape[0]), int(shape[1])],
            confs=self._official_float_list(getattr(self.args, "gcs_official_confs", None), (0.005, 0.01, 0.02, 0.05, 0.1)),
            point_valid_thrs=self._official_float_list(getattr(self.args, "gcs_official_point_valid_thrs", None), (0.45, 0.5)),
            nms_dist_pxs=self._official_float_list(getattr(self.args, "gcs_official_nms_dist_pxs", None), (0.0, 18.0, 30.0, 50.0)),
            max_dets=self._official_int_list(getattr(self.args, "gcs_official_max_dets", None), (5, 6, 8)),
            min_points=self._official_int_list(getattr(self.args, "gcs_official_min_points", None), (4, 5, 6)),
            max_images=int(getattr(self.args, "gcs_official_max_images", 0) or 0),
            warmup=int(getattr(self.args, "gcs_official_warmup", 5) or 0),
            device=self._official_device_arg(),
            half=bool(getattr(self.args, "gcs_official_half", False)),
            runtime_ms=1.0,
            save_dir=str(save_dir),
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
            "selection_policy": [
                "max official_acc",
                "then max official_score",
                "then lower official_FP",
                "then lower official_FN",
                "then max count_acc_4",
            ],
            "best": best,
        }
        summary["official_best"] = meta
        shutil.copy2(self.last, self.official_best)
        self.official_best_sweep.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        YAML.save(
            self.official_best_decode,
            {
                "weights": str(self.official_best.resolve()),
                "source_epoch": int(epoch_num),
                "sweep_summary": str(self.official_best_sweep.resolve()),
                "selection_policy": meta["selection_policy"],
                "decode": {
                    "conf": float(best["conf"]),
                    "point_valid_thr": float(best["point_valid_thr"]),
                    "nms_dist_px": float(best["nms_dist_px"]),
                    "max_det": int(best["max_det"]),
                    "min_points": int(best["min_points"]),
                },
                "official_metrics": {
                    "official_acc": float(best["official_acc"]),
                    "official_score": float(best["official_score"]),
                    "official_FP": float(best["official_FP"]),
                    "official_FN": float(best["official_FN"]),
                    "count_acc": float(best.get("count_acc", 0.0)),
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
                f"count_acc_4={float(best.get('count_acc_4', 0.0)):.6f}"
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
