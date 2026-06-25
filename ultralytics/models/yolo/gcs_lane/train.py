# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Trainer for GCS-YOLO-Lane structured lane detection."""

from __future__ import annotations

import math
import numpy as np
import random
import re
from collections import Counter
from copy import copy
from pathlib import Path
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
        "lane_balanced_point_loss",
        "gt4_short_lane_loss",
        "gt4_lane_balanced_point_loss",
        "point_valid_loss",
        "short_valid_recall_loss",
        "gt4_short_valid_recall_loss",
        "gt4_short_valid_count_floor_loss",
        "valid_lb_gt4_short_count",
        "valid_lb_gt4_short_gt_points_mean",
        "valid_lb_gt4_short_pred_prob_mean",
        "valid_lb_gt4_short_pred_sum_mean",
        "unmatched_valid_neg_loss",
        "unmatched_valid_query_count",
        "unmatched_valid_prob_mean",
        "smooth_loss",
        "curve_loss",
        "mask_loss",
        "edge_loss",
        "count_loss",
        "count_under5_loss",
        "count_ce_loss",
        "count_ce_acc",
        "duplicate_margin_loss",
        "spurious_margin_loss",
        "far_spurious_survival_loss",
        "gt5_rank_consistency_loss",
        "gt3_extra_survival_loss",
        "gt4_short_lane_valid_points_mean",
        "gt4_short_lane_count",
        "gt4_short_valid_lane_count",
        "gt4_short_gt_valid_points_mean",
        "gt4_short_pred_valid_prob_mean",
        "gt4_short_pred_valid_sum_mean",
    )
    # Keep tqdm headers within BaseTrainer's 11-character progress columns.
    progress_loss_names = (
        "exist",
        "point",
        "lane_bal",
        "gt4_pt",
        "gt4_lbp",
        "pt_valid",
        "short_rec",
        "gt4_vrec",
        "gt4_vfloor",
        "vlb_gt4_n",
        "vlb_gtpts",
        "vlb_vprob",
        "vlb_vsum",
        "uvneg_loss",
        "uvneg_n",
        "uvneg_prob",
        "smooth",
        "curve",
        "mask",
        "edge",
        "count",
        "cnt_under5",
        "cnt_ce",
        "cnt_acc",
        "dup_margin",
        "spur_margin",
        "far_surv",
        "gt5_rank",
        "gt3_ext",
        "gt4_vmean",
        "gt4_n",
        "gt4_vn",
        "gt4_gtpts",
        "gt4_vprob",
        "gt4_vsum",
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

    def _lane_count_sampler(self, dataset: GCSLaneDataset, balance_lane_counts: bool = True) -> WeightedRandomSampler:
        """Build a replacement sampler that balances GT lane counts and can upweight GT4 samples."""
        counts = [self._label_lane_count(Path(p)) for p in dataset.label_files]
        hist = Counter(counts)
        power = float(getattr(self.args, "gcs_lane_count_balance_power", 1.0)) if balance_lane_counts else 0.0
        min_group = max(int(getattr(self.args, "gcs_lane_count_min_group", 50) or 0), 1)
        gt4_gain = float(getattr(self.args, "gcs_gt4_sample_gain", 1.0) or 1.0)
        if gt4_gain < 1.0 or gt4_gain > 2.0:
            raise ValueError(f"gcs_gt4_sample_gain must be in [1.0, 2.0], got {gt4_gain}.")
        weights = torch.as_tensor(
            [
                (1.0 / (float(max(hist[c], min_group)) ** power)) * (gt4_gain if int(c) == 4 else 1.0)
                for c in counts
            ],
            dtype=torch.double,
        )
        total_weight = float(weights.sum().item())
        effective_hist = {}
        for lane_count in sorted(hist):
            mask = torch.as_tensor([int(c) == int(lane_count) for c in counts], dtype=torch.bool)
            expected = float(weights[mask].sum().item()) / max(total_weight, 1e-12) * float(len(weights))
            effective_hist[int(lane_count)] = round(expected, 3)
        LOGGER.info(
            "GCS lane-count sampler enabled: "
            f"hist={dict(sorted(hist.items()))}, train_gt_count_hist_effective={effective_hist}, "
            f"balance={bool(balance_lane_counts)}, power={power:g}, min_group={min_group}, gt4_gain={gt4_gain:g}, "
            f"samples_per_epoch={len(weights)}"
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
        gt4_sample_gain = float(getattr(self.args, "gcs_gt4_sample_gain", 1.0) or 1.0)
        if mode == "train" and (gt4_sample_gain < 1.0 or gt4_sample_gain > 2.0):
            raise ValueError(f"gcs_gt4_sample_gain must be in [1.0, 2.0], got {gt4_sample_gain}.")
        use_sampler = bool(getattr(self.args, "gcs_lane_count_balanced", False)) or gt4_sample_gain != 1.0
        if mode == "train" and use_sampler:
            if rank != -1:
                LOGGER.warning("GCS lane-count/GT4 sampling is only enabled for single-process training.")
            else:
                sampler = self._lane_count_sampler(
                    dataset,
                    balance_lane_counts=bool(getattr(self.args, "gcs_lane_count_balanced", False)),
                )
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
            ".count_mlp.",
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

    def save_model(self):
        """Save checkpoints with explicit rectangular GCS imgsz in train_args."""
        old_imgsz = self.args.imgsz
        shape = self._resolve_gcs_imgsz()
        self.args.gcs_imgsz = [int(shape[0]), int(shape[1])]
        self.args.imgsz = max(int(shape[0]), int(shape[1]))
        try:
            return super().save_model()
        finally:
            self.args.imgsz = old_imgsz

    def label_loss_items(self, loss_items: list[float] | torch.Tensor | None = None, prefix: str = "train"):
        """Return named GCS loss items for logging."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        return dict(zip(keys, [round(float(x), 5) for x in loss_items]))

    def _progress_loss_names(self) -> tuple[str, ...]:
        """Return short progress labels, marking default-off experiment columns explicitly."""
        names = list(self.progress_loss_names)
        args = getattr(self, "args", None)

        def arg_float(name: str, default: float = 0.0) -> float:
            value = getattr(args, name, default) if args is not None else default
            try:
                return float(default if value is None else value)
            except (TypeError, ValueError):
                return default

        if arg_float("gcs_gt4_lane_balanced_point") <= 0.0 and "gt4_lbp" in names:
            names[names.index("gt4_lbp")] = "gt4lbp_off"
        if arg_float("gcs_short_valid_recall") <= 0.0 and "short_rec" in names:
            names[names.index("short_rec")] = "short_off"
        if not bool(getattr(args, "gcs_gt4_short_valid_recall", False)) and "gt4_vrec" in names:
            names[names.index("gt4_vrec")] = "gt4vrec_off"
        if not bool(getattr(args, "gcs_gt4_short_valid_count_floor", False)) and "gt4_vfloor" in names:
            names[names.index("gt4_vfloor")] = "gt4vf_off"
        return tuple(names)

    def progress_string(self):
        """Return a progress header matching the GCS loss vector."""
        progress_loss_names = self._progress_loss_names()
        return ("\n" + "%11s" * (4 + len(progress_loss_names))) % (
            "Epoch",
            "GPU_mem",
            *progress_loss_names,
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
