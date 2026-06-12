# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Trainer for GCS-YOLO-Lane structured lane detection."""

from __future__ import annotations

import json
import numpy as np
import re
import shutil
from collections import Counter
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Sampler, WeightedRandomSampler

from ultralytics.data import build_dataloader
from ultralytics.data.dataset_gcs import GCSLaneDataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.gcs_lane.val import GCSLaneValidator
from ultralytics.nn.modules import GCSLaneHead, LaneBiFPN, LaneFeatureProjection, LSEM
from ultralytics.nn.tasks import GCSLaneModel, load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, RANK, ROOT, YAML
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, assert_gcs_shape, normalize_imgsz
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first


GCS_MAINLINE_GROUP_SAMPLER_RATIOS = "2:0.01,3:0.29,4:0.42,5:0.28"
GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT = 1.0
GCS_MAINLINE_COUNT_SUM_GAIN = 0.03
GCS_MAINLINE_QUALITY_GAIN = 0.4
GCS_MAINLINE_QUALITY_NEG_WEIGHT = 0.5
GCS_MAINLINE_COUNT_CLS_WEIGHTS = (0.5, 1.2, 1.4, 1.8)
GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT = 2.0
GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT = 1.15
GCS_MAINLINE_COUNT_BOUNDARY_GAIN = 0.05
GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING = 0.05
GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT = 1.15
GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT = 1.10
GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY = 0.05
GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR = 0.55


def _parse_number_list(value: Any, cast=float) -> list:
    """Parse a comma/space separated CLI-style list into numbers."""
    if value is None or value is False:
        return []
    if isinstance(value, (list, tuple, set)):
        return [cast(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    return [cast(x) for x in re.split(r"[,;\s]+", text) if x]


def _parse_string_list(value: Any) -> list[str]:
    """Parse a comma/space separated CLI-style list into strings."""
    if value is None or value is False:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return [x for x in re.split(r"[,;\s]+", str(value).strip()) if x]


def apply_gt5_oversample_weight_to_ratios(ratios: dict[int, float], weight: float) -> dict[int, float]:
    """Return sampler ratios with an explicit GT=5 boost applied before normalization."""
    weight_f = float(weight)
    if weight_f <= 0.0:
        raise ValueError(f"gcs_gt5_oversample_weight must be > 0, got {weight_f}.")
    out = {int(k): float(v) for k, v in ratios.items()}
    if weight_f != 1.0:
        out[5] = out.get(5, 0.0) * weight_f if 5 in out else weight_f
    return out


class LaneCountGroupCycleSampler(Sampler[int]):
    """Lane-count target-ratio sampler that cycles each group without replacement before repeating."""

    def __init__(
        self,
        counts: list[int],
        ratios: dict[int, float],
        hard_hits: list[bool] | None = None,
        hard_boost: float = 1.0,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        """Create a grouped sampler for rare 4/5-lane exposure without blind replacement sampling."""
        if not counts:
            raise ValueError("LaneCountGroupCycleSampler requires at least one sample.")
        self.counts = [int(x) for x in counts]
        self.num_samples = int(num_samples or len(self.counts))
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {self.num_samples}.")
        self.ratios = {int(k): float(v) for k, v in ratios.items() if float(v) > 0.0}
        self.seed = int(seed)
        self._cycle = 0
        self.hard_hits = [bool(x) for x in hard_hits] if hard_hits is not None else [False] * len(self.counts)
        if len(self.hard_hits) != len(self.counts):
            raise ValueError("hard_hits length must match counts length.")
        self.hard_boost = max(float(hard_boost), 1.0)
        self.groups: dict[int, list[int]] = {}
        for i, count in enumerate(self.counts):
            self.groups.setdefault(count, []).append(i)

    def __len__(self) -> int:
        """Return the number of sampled indices per logical epoch."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set a deterministic base cycle for external distributed-style callers."""
        self._cycle = int(epoch)

    def _quota_ratios(self) -> dict[int, float]:
        """Normalize configured ratios over present lane-count groups and distribute any leftover naturally."""
        present = sorted(self.groups)
        configured = {c: r for c, r in self.ratios.items() if c in self.groups and r > 0.0}
        if not configured:
            equal = 1.0 / float(len(present))
            return {c: equal for c in present}

        total = sum(configured.values())
        if total > 1.0:
            return {c: r / total for c, r in configured.items()}

        out = dict(configured)
        leftover = max(0.0, 1.0 - total)
        if leftover > 0.0:
            fallback = [c for c in present if c not in out]
            if not fallback:
                fallback = present
            fallback_total = float(sum(len(self.groups[c]) for c in fallback))
            if fallback_total <= 0.0:
                extra = leftover / float(len(fallback))
                for c in fallback:
                    out[c] = out.get(c, 0.0) + extra
            else:
                for c in fallback:
                    out[c] = out.get(c, 0.0) + leftover * len(self.groups[c]) / fallback_total
        return {c: r for c, r in out.items() if r > 0.0 and c in self.groups}

    def quotas(self) -> dict[int, int]:
        """Return per-lane-count integer quotas for one sampler cycle."""
        ratios = self._quota_ratios()
        raw = {c: ratios[c] * self.num_samples for c in ratios}
        quotas = {c: int(np.floor(v)) for c, v in raw.items()}
        remainder = self.num_samples - sum(quotas.values())
        order = sorted(raw, key=lambda c: raw[c] - quotas[c], reverse=True)
        for c in order[: max(remainder, 0)]:
            quotas[c] += 1
        return {c: q for c, q in quotas.items() if q > 0}

    @staticmethod
    def _draw_cycle(indices: list[int], quota: int, generator: torch.Generator) -> list[int]:
        """Draw quota indices, exhausting shuffled full group cycles before any repeat."""
        if quota <= 0 or not indices:
            return []
        out: list[int] = []
        while len(out) < quota:
            perm = torch.randperm(len(indices), generator=generator).tolist()
            take = min(quota - len(out), len(indices))
            out.extend(indices[i] for i in perm[:take])
        return out

    def _draw_group(self, indices: list[int], quota: int, generator: torch.Generator) -> list[int]:
        """Draw one lane-count group, optionally allocating more slots to hard-file failures."""
        if quota <= 0:
            return []
        hard = [i for i in indices if self.hard_hits[i]]
        normal = [i for i in indices if not self.hard_hits[i]]
        if not hard or self.hard_boost <= 1.0:
            return self._draw_cycle(indices, quota, generator)
        if not normal:
            return self._draw_cycle(hard, quota, generator)

        hard_mass = len(hard) * self.hard_boost
        normal_mass = float(len(normal))
        hard_quota = int(round(quota * hard_mass / max(hard_mass + normal_mass, 1e-6)))
        hard_quota = min(max(hard_quota, 1), quota)
        out = self._draw_cycle(hard, hard_quota, generator)
        out.extend(self._draw_cycle(normal, quota - hard_quota, generator))
        perm = torch.randperm(len(out), generator=generator).tolist()
        return [out[i] for i in perm]

    def __iter__(self):
        """Yield one target-ratio epoch of shuffled indices."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._cycle)
        self._cycle += 1
        out: list[int] = []
        for count, quota in self.quotas().items():
            out.extend(self._draw_group(self.groups[count], quota, generator))
        if len(out) > self.num_samples:
            out = out[: self.num_samples]
        if len(out) < self.num_samples:
            all_indices = list(range(len(self.counts)))
            out.extend(self._draw_cycle(all_indices, self.num_samples - len(out), generator))
        perm = torch.randperm(len(out), generator=generator).tolist()
        return iter(out[i] for i in perm)


class GCSLaneTrainer(BaseTrainer):
    """Train GCS-YOLO-Lane with structured lane labels and GCSLoss."""

    loss_names = (
        "exist_loss",
        "point_loss",
        "point_valid_loss",
        "line_iou_loss",
        "count_cls_loss",
        "count_sum_loss",
        "quality_loss",
    )
    progress_loss_names = (
        "exist_loss",
        "point_loss",
        "point_valid_loss",
        "line_iou_loss",
        "count_cls_loss",
        "count_sum_loss",
        "quality_loss",
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
        overrides.setdefault("model", str(ROOT / "cfg/models/gcs/gcs-yolo-lane-s-q12.yaml"))
        overrides.setdefault("data", str(ROOT.parent / "data/tusimple_gcs_fixed_y_960x544.yaml"))
        # The current main GCS config uses 12 lane queries. Four-image mosaic can still raise TuSimple GT lanes
        # above the query budget, so keep mosaic off unless requested.
        overrides.setdefault("mosaic", 0.0)
        # Plain GCS training historically used no affine erasing/scale aug; keep direct trainer construction stable.
        overrides.setdefault("scale", 0.0)
        overrides.setdefault("erasing", 0.0)
        overrides.setdefault("auto_augment", None)
        overrides.setdefault("gcs_sampler_mode", "group_cycle")
        overrides.setdefault("gcs_group_sampler_ratios", GCS_MAINLINE_GROUP_SAMPLER_RATIOS)
        overrides.setdefault("gcs_gt5_oversample_weight", GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT)
        overrides.setdefault("gcs_count_sum", GCS_MAINLINE_COUNT_SUM_GAIN)
        overrides.setdefault("gcs_quality", GCS_MAINLINE_QUALITY_GAIN)
        overrides.setdefault("gcs_quality_neg_weight", GCS_MAINLINE_QUALITY_NEG_WEIGHT)
        for idx, weight in enumerate(GCS_MAINLINE_COUNT_CLS_WEIGHTS, start=2):
            overrides.setdefault(f"gcs_count_cls_w{idx}", weight)
        overrides.setdefault("gcs_point_valid_gt5_pos_weight", GCS_MAINLINE_POINT_VALID_GT5_POS_WEIGHT)
        overrides.setdefault("gcs_gt5_edge_loss_weight", GCS_MAINLINE_GT5_EDGE_LOSS_WEIGHT)
        overrides.setdefault("gcs_count_boundary", GCS_MAINLINE_COUNT_BOUNDARY_GAIN)
        overrides.setdefault("gcs_count_boundary_label_smoothing", GCS_MAINLINE_COUNT_BOUNDARY_LABEL_SMOOTHING)
        overrides.setdefault("gcs_count_boundary_gt5_pos_weight", GCS_MAINLINE_COUNT_BOUNDARY_GT5_POS_WEIGHT)
        overrides.setdefault("gcs_candidate_gt5_edge_weight", GCS_MAINLINE_CANDIDATE_GT5_EDGE_WEIGHT)
        overrides.setdefault("gcs_point_valid_gt5_edge_continuity", GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY)
        overrides.setdefault("gcs_point_valid_gt5_edge_continuity_thr", GCS_MAINLINE_POINT_VALID_GT5_EDGE_CONTINUITY_THR)
        overrides.setdefault("gcs_hard_sampling", False)
        overrides.setdefault("gcs_hard_lane_counts", "")
        overrides.setdefault("gcs_hard_sampling_boost_by_count", "")
        overrides.setdefault("gcs_gt5_extra_aug", True)
        for native_loss_gain in ("box", "cls", "dfl", "pose", "kobj", "rle", "angle"):
            overrides[native_loss_gain] = 0.0
        super().__init__(cfg, overrides, _callbacks)
        self._lock_gcs_shape_contract()
        self.gcs_official_best_fitness = None

    def get_dataset(self) -> dict[str, Any]:
        """Load a standard YAML but pair images with labels_gcs/*.npz at dataset time."""
        data = check_det_dataset(self.args.data)
        train_images = getattr(self.args, "train_images", None)
        val_images = getattr(self.args, "val_images", None)
        if train_images:
            data["train"] = str(train_images)
        if val_images:
            data["val"] = str(val_images)
        if bool(getattr(self.args, "gcs_train_include_val", False)):
            train_sources = data["train"] if isinstance(data["train"], list) else [data["train"]]
            val_sources = data.get("val", [])
            val_sources = val_sources if isinstance(val_sources, list) else [val_sources]
            data["train"] = [*train_sources, *[x for x in val_sources if x]]
            if bool(getattr(self.args, "val", True)):
                LOGGER.warning(
                    "gcs_train_include_val=True adds validation images to the training loader. "
                    "Use --no-val for final train+val fitting, or keep a separate official-val subset for model selection."
                )
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
            "point, point-visibility, and eval pixel scales all stay aligned."
        )
        self.gcs_imgsz = shape
        self.args.gcs_imgsz = [int(shape[0]), int(shape[1])]
        # Keep run metadata explicit. BaseTrainer will convert this to max(H,W) later only for YOLO internals.
        self.args.imgsz = [int(shape[0]), int(shape[1])]
        self._save_shape_locked_args()

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build the GCS lane dataset."""
        fraction = self.args.fraction if mode == "train" else 1.0
        image_dir = getattr(self.args, f"{mode}_images", None) or img_path
        label_dir = getattr(self.args, f"{mode}_gcs_labels", None)
        if mode == "train" and bool(getattr(self.args, "gcs_train_include_val", False)):
            val_labels = getattr(self.args, "val_gcs_labels", None)
            if label_dir and val_labels:
                label_dir = [label_dir, val_labels]
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
            translate=self.args.translate if augment else 0.0,
            scale=self.args.scale if augment else 0.0,
            erasing=self.args.erasing if augment else 0.0,
            mosaic=self.args.mosaic if augment else 0.0,
            gt5_extra_aug=bool(getattr(self.args, "gcs_gt5_extra_aug", False)) if augment else False,
            gt5_aug_min_lanes=int(getattr(self.args, "gcs_gt5_aug_min_lanes", 5)),
            gt5_erasing=float(getattr(self.args, "gcs_gt5_erasing", 0.0)) if augment else 0.0,
            gt5_blur=float(getattr(self.args, "gcs_gt5_blur", 0.0)) if augment else 0.0,
            gt5_noise=float(getattr(self.args, "gcs_gt5_noise", 0.0)) if augment else 0.0,
            gt5_shadow=float(getattr(self.args, "gcs_gt5_shadow", 0.0)) if augment else 0.0,
        )
        assert_gcs_shape(dataset.imgsz, gcs_imgsz, name=f"{mode} dataset.imgsz", context="GCSLaneTrainer.build_dataset")
        self._check_point_mode_contract(dataset, mode=mode)
        return dataset

    def _head_point_mode(self) -> str | None:
        """Return the GCS head point mode after the model has been constructed."""
        head = self._gcs_head()
        if head is None:
            return None
        mode = str(getattr(head, "point_mode", "free")).lower()
        return "fixed_y" if mode in {"fixed-y", "fixedy"} else mode

    def _gcs_head(self) -> GCSLaneHead | None:
        """Return the first GCS lane head after the model has been constructed."""
        model = getattr(self, "model", None)
        if model is None:
            return None
        model = getattr(model, "module", model)
        for module in model.modules():
            if isinstance(module, GCSLaneHead):
                return module
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
        if head_mode == "fixed_y" and getattr(dataset, "fixed_y_anchors", None) is not None:
            head = self._gcs_head()
            head_anchors = head.fixed_y_anchors.detach().cpu().numpy().astype(np.float32)
            data_anchors = np.asarray(dataset.fixed_y_anchors, dtype=np.float32)
            if head_anchors.shape != data_anchors.shape:
                raise ValueError(
                    f"GCS fixed-y anchor mismatch for {mode}: model anchors shape={head_anchors.shape}, "
                    f"label anchors shape={data_anchors.shape} under {dataset.label_files[0].parent}."
                )
            max_err = float(np.max(np.abs(head_anchors - data_anchors))) if head_anchors.size else 0.0
            if max_err > 5e-5:
                raise ValueError(
                    f"GCS fixed-y anchor mismatch for {mode}: model first/last="
                    f"({head_anchors[0]:.9f}, {head_anchors[-1]:.9f}) but labels first/last="
                    f"({data_anchors[0]:.9f}, {data_anchors[-1]:.9f}), max_err={max_err:.6g}. "
                    "Regenerate labels with tools/convert_tusimple_to_gcs.py using the same --fixed-y-start/--fixed-y-end."
                )

    @staticmethod
    def _label_lane_count(label_file: Path) -> int:
        """Read the number of valid GT lanes from one GCS npz label."""
        count, _ = GCSLaneTrainer._label_metadata(label_file)
        return count

    @staticmethod
    def _array_scalar_str(value: Any) -> str:
        """Convert a scalar numpy/string label field to a Python string."""
        arr = np.asarray(value)
        item = arr.reshape(-1)[0] if arr.shape else arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="ignore")
        return str(item)

    @staticmethod
    def _label_metadata(label_file: Path) -> tuple[int, str]:
        """Read sampler metadata from one GCS npz label."""
        with np.load(label_file, allow_pickle=False) as data:
            if "num_lanes" in data:
                count = int(np.asarray(data["num_lanes"]).reshape(-1)[0])
            elif "lane_valid" in data:
                count = int((data["lane_valid"].sum(axis=1) >= 2).sum())
            else:
                count = int(data["lanes"].shape[0])
            raw_file = GCSLaneTrainer._array_scalar_str(data["raw_file"]) if "raw_file" in data else ""
        return count, raw_file

    @staticmethod
    def _parse_hard_lane_counts(value: Any) -> set[int]:
        """Parse comma/space separated lane-count ids such as '4,5'."""
        if value is None or value is False:
            return set()
        if isinstance(value, int):
            return {int(value)}
        if isinstance(value, (list, tuple, set)):
            text = ",".join(str(x) for x in value)
        else:
            text = str(value)
        counts = set()
        for token in re.split(r"[,;\s]+", text.strip()):
            if not token or token.lower() in {"none", "false", "off", "no"}:
                continue
            counts.add(int(token))
        return counts

    @staticmethod
    def _parse_hard_boost_by_count(value: Any) -> dict[int, float]:
        """Parse final per-lane-count sampler multipliers such as '4:1.5,5:2.0'."""
        if value is None or value is False:
            return {}
        if isinstance(value, dict):
            items = value.items()
        else:
            text = str(value).strip()
            if not text or text.lower() in {"none", "false", "off", "no"}:
                return {}
            pairs = []
            for token in re.split(r"[,;\s]+", text):
                if not token:
                    continue
                if ":" in token:
                    count, boost = token.split(":", 1)
                elif "=" in token:
                    count, boost = token.split("=", 1)
                else:
                    raise ValueError(
                        "gcs_hard_sampling_boost_by_count entries must use count:boost or count=boost, "
                        f"got {token!r} from {value!r}."
                    )
                pairs.append((count, boost))
            items = pairs

        boosts = {}
        for count, boost in items:
            count_i = int(str(count).strip())
            boost_f = float(str(boost).strip())
            if boost_f <= 0.0:
                raise ValueError(
                    f"gcs_hard_sampling_boost_by_count multipliers must be > 0, got {boost_f} for count {count_i}."
                )
            boosts[count_i] = boost_f
        return boosts

    @staticmethod
    def _parse_group_sampler_ratios(value: Any) -> dict[int, float]:
        """Parse target lane-count ratios such as '3:0.25,4:0.40,5:0.35'."""
        if value is None or value is False:
            return {}
        if isinstance(value, dict):
            items = value.items()
        else:
            text = str(value).strip()
            if not text or text.lower() in {"none", "false", "off", "no"}:
                return {}
            pairs = []
            for token in re.split(r"[,;\s]+", text):
                if not token:
                    continue
                if ":" in token:
                    count, ratio = token.split(":", 1)
                elif "=" in token:
                    count, ratio = token.split("=", 1)
                else:
                    raise ValueError(
                        "gcs_group_sampler_ratios entries must use count:ratio or count=ratio, "
                        f"got {token!r} from {value!r}."
                    )
                pairs.append((count, ratio))
            items = pairs

        ratios = {}
        for count, ratio in items:
            count_i = int(str(count).strip())
            ratio_f = float(str(ratio).strip())
            if ratio_f < 0.0:
                raise ValueError(f"gcs_group_sampler_ratios values must be >= 0, got {ratio_f} for count {count_i}.")
            if ratio_f > 0.0:
                ratios[count_i] = ratio_f
        return ratios

    @staticmethod
    def _normalize_sample_id(value: Any) -> str:
        """Normalize image/raw-file identifiers used by hard sample manifests."""
        return str(value).strip().strip("\"'").replace("\\", "/")

    @staticmethod
    def _collect_json_strings(value: Any) -> list[str]:
        """Collect string leaves from a permissive JSON hard-sample manifest."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            items: list[str] = []
            for v in value.values():
                items.extend(GCSLaneTrainer._collect_json_strings(v))
            return items
        if isinstance(value, (list, tuple)):
            items = []
            for v in value:
                items.extend(GCSLaneTrainer._collect_json_strings(v))
            return items
        return []

    @staticmethod
    def _sample_id_variants(value: Any) -> set[str]:
        """Return exact and stem variants for one sample identifier."""
        norm = GCSLaneTrainer._normalize_sample_id(value)
        if not norm:
            return set()
        variants = {norm, norm.lstrip("./")}
        # Path-like TuSimple raw_file ids end with non-unique frame names such as 20.jpg.
        # Keep suffix matching for full paths, but do not add the bare stem for these ids.
        if "/" not in norm:
            stem = Path(norm).stem
            variants.add(stem)
        return variants

    def _load_hard_sample_ids(self) -> set[str]:
        """Load optional hard-sample image/raw-file identifiers from txt/json."""
        file_arg = str(getattr(self.args, "gcs_hard_sample_file", "") or "").strip()
        if not file_arg:
            return set()
        path = Path(file_arg)
        if not path.is_absolute():
            path = ROOT.parent / path
        if not path.exists():
            raise FileNotFoundError(f"gcs_hard_sample_file does not exist: {path}")

        if path.suffix.lower() == ".json":
            values = self._collect_json_strings(json.loads(path.read_text(encoding="utf-8")))
        else:
            values = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]

        ids: set[str] = set()
        for value in values:
            ids.update(self._sample_id_variants(value))
        if not ids:
            raise ValueError(f"gcs_hard_sample_file is empty or contains no usable sample ids: {path}")
        return ids

    def _sample_matches_hard_ids(self, img_file: Path, label_file: Path, raw_file: str, hard_ids: set[str]) -> bool:
        """Check whether a dataset sample is listed in the hard-sample manifest."""
        if not hard_ids:
            return False
        candidates: set[str] = set()
        for value in (img_file, label_file, img_file.as_posix(), label_file.as_posix(), img_file.stem, label_file.stem, raw_file):
            candidates.update(self._sample_id_variants(value))
        if candidates & hard_ids:
            return True
        path_like_ids = [x for x in hard_ids if "/" in x]
        return any(candidate.endswith(hard_id) for candidate in candidates for hard_id in path_like_ids)

    def _train_weighted_sampler(self, dataset: GCSLaneDataset) -> WeightedRandomSampler | None:
        """Build a replacement sampler for lane-count balancing and optional hard-sample exposure."""
        metadata = [self._label_metadata(Path(p)) for p in dataset.label_files]
        counts = [x[0] for x in metadata]
        raw_files = [x[1] for x in metadata]
        hist = Counter(counts)
        weights = torch.ones(len(counts), dtype=torch.double)

        lane_count_balanced = bool(getattr(self.args, "gcs_lane_count_balanced", False))
        power = float(getattr(self.args, "gcs_lane_count_balance_power", 1.0))
        min_group = max(int(getattr(self.args, "gcs_lane_count_min_group", 50) or 0), 1)
        if lane_count_balanced:
            weights = torch.as_tensor([1.0 / (float(max(hist[c], min_group)) ** power) for c in counts], dtype=torch.double)

        hard_sampling = bool(getattr(self.args, "gcs_hard_sampling", False))
        hard_lane_counts = self._parse_hard_lane_counts(getattr(self.args, "gcs_hard_lane_counts", "4,5"))
        hard_boost = float(getattr(self.args, "gcs_hard_sampling_boost", 1.0))
        if hard_boost <= 0.0:
            raise ValueError(f"gcs_hard_sampling_boost must be > 0, got {hard_boost}.")
        hard_boost_by_count = self._parse_hard_boost_by_count(
            getattr(self.args, "gcs_hard_sampling_boost_by_count", "")
        )
        hard_count_hits = [hard_sampling and (c in hard_lane_counts or c in hard_boost_by_count) for c in counts]
        if any(hard_count_hits):
            hit_weights = torch.as_tensor(
                [
                    (hard_boost_by_count.get(c, hard_boost) if hit else 1.0)
                    for c, hit in zip(counts, hard_count_hits)
                ],
                dtype=torch.double,
            )
            weights *= hit_weights

        hard_ids = self._load_hard_sample_ids()
        hard_sample_boost = float(getattr(self.args, "gcs_hard_sample_boost", 1.0))
        if hard_sample_boost <= 0.0:
            raise ValueError(f"gcs_hard_sample_boost must be > 0, got {hard_sample_boost}.")
        hard_file_hits = [
            self._sample_matches_hard_ids(Path(img), Path(label), raw, hard_ids)
            for img, label, raw in zip(dataset.im_files, dataset.label_files, raw_files)
        ]
        if hard_ids and not any(hard_file_hits):
            raise ValueError(
                "gcs_hard_sample_file did not match any training samples. "
                "If this manifest was built from clean official-val, it cannot affect a train-only loader; "
                "build the hard file from a train-split eval summary or use train+val only for a final no-val fit."
            )
        if any(hard_file_hits) and hard_sample_boost != 1.0:
            hit_weights = torch.as_tensor([hard_sample_boost if hit else 1.0 for hit in hard_file_hits], dtype=torch.double)
            weights *= hit_weights

        gt5_oversample_weight = float(
            getattr(self.args, "gcs_gt5_oversample_weight", GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT)
        )
        if gt5_oversample_weight <= 0.0:
            raise ValueError(f"gcs_gt5_oversample_weight must be > 0, got {gt5_oversample_weight}.")
        if gt5_oversample_weight != 1.0:
            weights *= torch.as_tensor(
                [gt5_oversample_weight if c == 5 else 1.0 for c in counts],
                dtype=torch.double,
            )

        if (
            not lane_count_balanced
            and not any(hard_count_hits)
            and not any(hard_file_hits)
            and gt5_oversample_weight == 1.0
        ):
            return None

        LOGGER.info(
            "GCS weighted sampling enabled: "
            f"hist={dict(sorted(hist.items()))}, lane_count_balanced={lane_count_balanced}, "
            f"power={power:g}, min_group={min_group}, hard_sampling={hard_sampling}, "
            f"hard_lane_counts={sorted(hard_lane_counts)}, hard_sampling_boost={hard_boost:g}, "
            f"hard_sampling_boost_by_count={dict(sorted(hard_boost_by_count.items()))}, "
            f"hard_count_samples={sum(hard_count_hits)}, hard_sample_file={getattr(self.args, 'gcs_hard_sample_file', '') or ''}, "
            f"hard_sample_boost={hard_sample_boost:g}, hard_file_samples={sum(hard_file_hits)}, "
            f"gt5_oversample_weight={gt5_oversample_weight:g}, "
            f"samples_per_epoch={len(weights)}"
        )
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def _train_group_cycle_sampler(self, dataset: GCSLaneDataset) -> LaneCountGroupCycleSampler | None:
        """Build the default no-replacement lane-count group-cycle sampler."""
        metadata = [self._label_metadata(Path(p)) for p in dataset.label_files]
        counts = [x[0] for x in metadata]
        raw_files = [x[1] for x in metadata]
        hist = Counter(counts)
        ratios = self._parse_group_sampler_ratios(getattr(self.args, "gcs_group_sampler_ratios", ""))
        if not ratios:
            ratios = {count: 1.0 for count in sorted(hist)}
        ratios = apply_gt5_oversample_weight_to_ratios(
            ratios,
            float(getattr(self.args, "gcs_gt5_oversample_weight", GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT)),
        )

        hard_ids = self._load_hard_sample_ids()
        hard_file_hits = [
            self._sample_matches_hard_ids(Path(img), Path(label), raw, hard_ids)
            for img, label, raw in zip(dataset.im_files, dataset.label_files, raw_files)
        ]
        if hard_ids and not any(hard_file_hits):
            raise ValueError(
                "gcs_hard_sample_file did not match any training samples. "
                "If this manifest was built from clean official-val, it cannot affect a train-only loader; "
                "build the hard file from a train-split eval summary or use train+val only for a final no-val fit."
            )
        hard_sample_boost = float(getattr(self.args, "gcs_hard_sample_boost", 1.0))
        if hard_sample_boost <= 0.0:
            raise ValueError(f"gcs_hard_sample_boost must be > 0, got {hard_sample_boost}.")

        sampler = LaneCountGroupCycleSampler(
            counts=counts,
            ratios=ratios,
            hard_hits=hard_file_hits,
            hard_boost=hard_sample_boost,
            num_samples=len(counts),
            seed=int(getattr(self.args, "seed", 0)),
        )
        LOGGER.info(
            "GCS group-cycle sampling enabled: "
            f"hist={dict(sorted(hist.items()))}, ratios={dict(sorted(ratios.items()))}, "
            f"quotas={dict(sorted(sampler.quotas().items()))}, hard_sample_file={getattr(self.args, 'gcs_hard_sample_file', '') or ''}, "
            f"hard_sample_boost={hard_sample_boost:g}, hard_file_samples={sum(hard_file_hits)}, "
            f"gt5_oversample_weight="
            f"{float(getattr(self.args, 'gcs_gt5_oversample_weight', GCS_MAINLINE_GT5_OVERSAMPLE_WEIGHT)):g}, "
            "group cycles are shuffled without replacement before repeating within each lane-count group."
        )
        return sampler

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Create a dataloader for variable-lane GCS labels."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        if mode == "val" and not self.args.val:
            return None
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        sampler = None
        shuffle = mode == "train"
        sampler_mode = str(getattr(self.args, "gcs_sampler_mode", "group_cycle") or "none").strip().lower()
        if sampler_mode in {"group-cycle", "groupcycle", "cycle"}:
            sampler_mode = "group_cycle"
        if sampler_mode not in {"group_cycle", "weighted", "none"}:
            raise ValueError(f"Unsupported gcs_sampler_mode={sampler_mode!r}; expected group_cycle, weighted, or none.")
        custom_sampling = sampler_mode != "none"
        if mode == "train" and custom_sampling:
            if rank != -1:
                LOGGER.warning("GCS custom sampling is only enabled for single-process training.")
            else:
                sampler = self._train_group_cycle_sampler(dataset) if sampler_mode == "group_cycle" else self._train_weighted_sampler(dataset)
                if sampler is not None:
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
        batch["img"] = batch["img"].float() / 255
        batch["epoch"] = int(getattr(self, "epoch", 0))
        gcs_imgsz = self._resolve_gcs_imgsz()
        assert_gcs_image_tensor(batch["img"], gcs_imgsz, name="batch['img']", context="GCSLaneTrainer.preprocess_batch")

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
            if name and isinstance(module, (LSEM, LaneBiFPN, LaneFeatureProjection, GCSLaneHead)):
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
        target_has_count_head = any(".count_head." in key or key.endswith(".count_head") for key in target_state)
        source_has_count_head = any(".count_head." in key or key.endswith(".count_head") for key in candidate_state)
        if source_is_gcs and target_has_count_head and not source_has_count_head:
            LOGGER.warning(
                "GCS warm-start checkpoint has no count_head.* tensors. The Count Head remains newly initialized; "
                "train or finetune it before reporting Count Head Top-K decode results."
            )
        obsolete_count_ordinal = sum((".count_" + "ord.") in key for key in candidate_state)
        if obsolete_count_ordinal:
            LOGGER.info(
                "Removed obsolete Count Head ordinal parameters due to loss cleanup; "
                f"skipping {obsolete_count_ordinal} checkpoint tensors."
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
        source_kind = "GCS" if source_is_gcs else "YOLO11-backbone-remap"
        LOGGER.info(
            f"GCS pretrained transfer ({source_kind}): loaded {len(loadable)}/{len(target_state)} tensors "
            f"(candidates={len(candidate_state)}, skipped_missing={skipped_missing}, "
            f"skipped_gcs={skipped_gcs}, skipped_shape={skipped_shape})"
        )
        if not loadable:
            LOGGER.warning(
                "No pretrained tensors were transferred. Check that the weight file is a YOLO11/YOLO11-seg "
                "checkpoint with the same scale as the GCS YAML, e.g. yolo11s-seg.pt for gcs-yolo-lane-s-q12.yaml."
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
    def _path_arg(value: Any) -> str | None:
        """Return a non-empty path string or None."""
        if value is None:
            return None
        text = str(value).strip()
        return None if not text or text.lower() in {"none", "false", "0"} else text

    def _official_best_enabled(self) -> bool:
        """Return True when training should maintain official_best.pt with a TuSimple official sweep."""
        return bool(getattr(self.args, "gcs_official_best", False))

    def _official_best_period(self) -> int:
        """Resolve the epoch interval used for official best selection."""
        explicit = int(getattr(self.args, "gcs_official_best_period", 0) or 0)
        if explicit > 0:
            return explicit
        return 10

    def _official_best_top_k(self) -> int:
        """Resolve how many official-val candidate checkpoints to preserve."""
        return max(1, int(getattr(self.args, "gcs_official_best_top_k", 1) or 1))

    def _load_official_best_fitness(self) -> None:
        """Restore the current official-best fitness when resuming an existing run."""
        if self.gcs_official_best_fitness is not None:
            return
        official_best = self.wdir / "official_best.pt"
        record_path = self.save_dir / "official_best_summary.json"
        if not official_best.exists() or not record_path.exists():
            return
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            best_fitness = record.get("best_fitness", None)
            if best_fitness is not None:
                self.gcs_official_best_fitness = float(best_fitness)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            LOGGER.warning("Failed to restore GCS official best fitness from official_best_summary.json.")

    @staticmethod
    def _official_best_candidate_is_better(
        current: dict[str, Any],
        previous: dict[str, Any] | None,
    ) -> bool:
        """Return True only when official Accuracy strictly improves across epochs."""
        if previous is None:
            return True
        return float(current.get("official_acc", 0.0)) > float(previous.get("official_acc", 0.0))

    @staticmethod
    def _official_topk_score(entry: dict[str, Any]) -> tuple[float, int]:
        """Sort official-val checkpoint candidates by official Accuracy, then earlier epoch."""
        candidate = entry.get("candidate", {})
        fitness = entry.get("candidate_fitness", None)
        if fitness is None and isinstance(candidate, dict):
            fitness = candidate.get("official_acc", 0.0)
        epoch = int(entry.get("candidate_epoch", entry.get("best_epoch", -1)) or -1)
        return (float(fitness or 0.0), -epoch)

    def _official_topk_checkpoint_path(self, epoch_num: int, fitness: float) -> Path:
        """Return the immutable Top-K checkpoint path for one official-val candidate."""
        safe_acc = f"{float(fitness):.6f}".replace(".", "p")
        return self.wdir / "official_topk" / f"epoch{int(epoch_num):04d}_acc{safe_acc}.pt"

    def _copy_official_topk_checkpoint(self, entry: dict[str, Any], source: Path) -> None:
        """Copy a candidate checkpoint into the official Top-K archive and record its relative path."""
        if not source.exists():
            return
        epoch_num = int(entry.get("candidate_epoch", entry.get("best_epoch", -1)) or -1)
        fitness = float(entry.get("candidate_fitness", entry.get("best_fitness", 0.0)) or 0.0)
        if epoch_num < 0:
            return
        target = self._official_topk_checkpoint_path(epoch_num, fitness)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entry["weights"] = target.relative_to(self.save_dir).as_posix()

    @staticmethod
    def _official_topk_entries_from_record(record: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Load preserved official-val Top-K entries from an existing summary record."""
        if not isinstance(record, dict):
            return []
        entries = record.get("official_top_k", [])
        if isinstance(entries, list) and entries:
            return [dict(x) for x in entries if isinstance(x, dict)]
        best = record.get("best")
        best_epoch = record.get("best_epoch", None)
        if isinstance(best, dict) and best_epoch is not None:
            return [
                {
                    "candidate_epoch": int(best_epoch),
                    "candidate_fitness": float(record.get("best_fitness", best.get("official_acc", 0.0))),
                    "candidate": best,
                    "selector": record.get("selector", {}),
                }
            ]
        return []

    def _update_official_topk(
        self,
        candidate_record: dict[str, Any],
        previous_record: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Update official-val Top-K metadata and archive retained checkpoint files."""
        if top_k <= 1:
            return []
        candidate_epoch = int(candidate_record["candidate_epoch"])
        entries = [
            entry
            for entry in self._official_topk_entries_from_record(previous_record)
            if int(entry.get("candidate_epoch", -1) or -1) != candidate_epoch
        ]
        entries.append(dict(candidate_record))
        entries = sorted(entries, key=self._official_topk_score, reverse=True)[:top_k]

        previous_best_source = self.wdir / "official_best.pt"
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank
            entry_epoch = int(entry.get("candidate_epoch", -1) or -1)
            if entry_epoch == candidate_epoch:
                self._copy_official_topk_checkpoint(entry, self.last)
            elif not entry.get("weights") and previous_best_source.exists():
                self._copy_official_topk_checkpoint(entry, previous_best_source)
        return entries

    def _run_official_best_sweep(self) -> None:
        """Run a TuSimple official sweep for the current checkpoint and update official_best.pt when it improves."""
        if not self._official_best_enabled() or RANK not in {-1, 0}:
            return
        from tools.sweep_tusimple_official import run_sweep, validate_official_sweep_split

        split = validate_official_sweep_split(
            getattr(self.args, "gcs_official_best_split", "val") or "val",
            context="Training official_best selection",
        )
        if not self.last.exists():
            return
        period = self._official_best_period()
        epoch_num = int(self.epoch) + 1
        gt_json = self._path_arg(getattr(self.args, "gcs_official_best_gt_json", None))
        if gt_json is None:
            raise ValueError(
                "GCS official best is enabled but gcs_official_best_gt_json is empty. "
                "Set --gcs-official-best-gt-json or disable --gcs-official-best."
            )
        if period <= 0:
            raise ValueError(
                "GCS official best is enabled and gcs_official_best_gt_json is set, but no positive sweep interval "
                "was configured. Set --gcs-official-best-period > 0."
            )
        if epoch_num % period != 0:
            return

        archive_root = self._path_arg(getattr(self.args, "gcs_official_best_archive_root", None)) or str(ROOT.parent / "archive")
        save_dir = self.save_dir / "official_best_sweeps" / f"epoch{epoch_num}"
        sweep_args = SimpleNamespace(
            dataset="tusimple",
            archive_root=archive_root,
            split=split,
            gt_json=gt_json,
            weights=str(self.last),
            imgsz=list(self._resolve_gcs_imgsz()),
            confs=_parse_number_list(
                getattr(self.args, "gcs_official_best_confs", "0.005 0.01 0.015 0.02 0.03 0.05 0.08 0.10"),
                float,
            ),
            point_valid_thrs=_parse_number_list(
                getattr(self.args, "gcs_official_best_point_valid_thrs", "0.20 0.25 0.30 0.35"), float
            ),
            nms_dist_pxs=_parse_number_list(getattr(self.args, "gcs_official_best_nms_dist_pxs", "18.0"), float),
            max_dets=_parse_number_list(getattr(self.args, "gcs_official_best_max_dets", "5"), int),
            min_points=_parse_number_list(getattr(self.args, "gcs_official_best_min_points", "6"), int),
            rank_min_points=_parse_string_list(self.args.gcs_official_best_rank_min_points),
            use_count_head_decode=bool(getattr(self.args, "gcs_use_count_head_decode", True)),
            count_head_temp=float(getattr(self.args, "gcs_count_head_temp", 1.0) or 1.0),
            candidate_min_points=int(getattr(self.args, "gcs_decode_candidate_min_points", 5) or 5),
            enable_rescue_candidate_pool=bool(getattr(self.args, "gcs_enable_rescue_candidate_pool", True)),
            rescue_candidate_conf=float(getattr(self.args, "gcs_decode_rescue_candidate_conf", 0.005) or 0.0),
            rescue_candidate_point_valid_thr=float(
                getattr(self.args, "gcs_decode_rescue_candidate_point_valid_thr", 0.08) or 0.0
            ),
            rescue_candidate_min_points=int(getattr(self.args, "gcs_decode_rescue_candidate_min_points", 4) or 4),
            final_min_points=int(getattr(self.args, "gcs_decode_final_min_points", 6) or 6),
            fifth_min_points=int(getattr(self.args, "gcs_decode_fifth_min_points", 5) or 5),
            line_nms_min_overlap=int(getattr(self.args, "gcs_line_nms_min_overlap", 6) or 6),
            line_nms_rescue_dist_px=float(getattr(self.args, "gcs_line_nms_rescue_dist_px", 30.0) or 0.0),
            quality_rescue_5th=bool(getattr(self.args, "gcs_quality_rescue_5th", True)),
            quality_rescue_count5_thr=float(getattr(self.args, "gcs_quality_rescue_count5_thr", 0.70)),
            quality_rescue_conf_thr=float(getattr(self.args, "gcs_quality_rescue_conf_thr", 0.03)),
            quality_rescue_mean_valid_thr=float(getattr(self.args, "gcs_quality_rescue_mean_valid_thr", 0.45)),
            quality_rescue_quality_thr=float(getattr(self.args, "gcs_quality_rescue_quality_thr", 0.55)),
            quality_rescue_min_points=int(getattr(self.args, "gcs_quality_rescue_min_points", 5) or 5),
            quality_rescue_dist_px=float(getattr(self.args, "gcs_quality_rescue_dist_px", 24.0) or 0.0),
            last_lane_rescue=bool(getattr(self.args, "gcs_last_lane_rescue", False)),
            last_lane_rescue_min_policy_count=int(
                getattr(self.args, "gcs_last_lane_rescue_min_policy_count", 4) or 4
            ),
            last_lane_rescue_conf_thr=getattr(self.args, "gcs_last_lane_rescue_conf_thr", None),
            last_lane_rescue_point_valid_thrs=_parse_number_list(
                getattr(self.args, "gcs_official_best_last_lane_rescue_point_valid_thrs", "0.08"), float
            ),
            last_lane_rescue_min_points=_parse_number_list(
                getattr(self.args, "gcs_official_best_last_lane_rescue_min_points", "4"), int
            ),
            last_lane_rescue_mean_valid_thrs=_parse_number_list(
                getattr(self.args, "gcs_official_best_last_lane_rescue_mean_valid_thrs", "0.40"), float
            ),
            last_lane_rescue_quality_thrs=_parse_number_list(
                getattr(self.args, "gcs_official_best_last_lane_rescue_quality_thrs", "0.50"), float
            ),
            last_lane_rescue_dist_pxs=_parse_number_list(
                getattr(self.args, "gcs_official_best_last_lane_rescue_dist_pxs", "24.0"), float
            ),
            edge_last_lane_rescue=bool(getattr(self.args, "gcs_edge_last_lane_rescue", False)),
            edge_rescue_conf_thr=float(getattr(self.args, "gcs_edge_rescue_conf_thr", 0.02)),
            edge_rescue_point_valid_thr=float(getattr(self.args, "gcs_edge_rescue_point_valid_thr", 0.06)),
            edge_rescue_min_points=int(getattr(self.args, "gcs_edge_rescue_min_points", 4) or 4),
            edge_rescue_mean_valid_thr=float(getattr(self.args, "gcs_edge_rescue_mean_valid_thr", 0.35)),
            edge_rescue_quality_thr=float(getattr(self.args, "gcs_edge_rescue_quality_thr", 0.45)),
            edge_rescue_outside_gap_px=float(getattr(self.args, "gcs_edge_rescue_outside_gap_px", 28.0)),
            edge_rescue_dist_px=float(getattr(self.args, "gcs_edge_rescue_dist_px", 24.0)),
            edge_rescue_min_policy_count=int(getattr(self.args, "gcs_edge_rescue_min_policy_count", 4) or 4),
            edge_count4_to5_upgrade=bool(getattr(self.args, "gcs_edge_count4_to5_upgrade", True)),
            edge_count4_to5_prob_margin=float(getattr(self.args, "gcs_edge_count4_to5_prob_margin", 0.20)),
            soft_count_decision=bool(getattr(self.args, "gcs_soft_count_decision", False)),
            soft_count_prob_margin=float(getattr(self.args, "gcs_soft_count_prob_margin", 0.08)),
            soft_count_quality_weight=float(getattr(self.args, "gcs_soft_count_quality_weight", 1.0)),
            soft_count_prior_weight=float(getattr(self.args, "gcs_soft_count_prior_weight", 0.5)),
            soft_count_duplicate_penalty=float(getattr(self.args, "gcs_soft_count_duplicate_penalty", 1.0)),
            soft_count_invalid_penalty=float(getattr(self.args, "gcs_soft_count_invalid_penalty", 1.0)),
            max_images=int(getattr(self.args, "gcs_official_best_max_images", 0) or 0),
            warmup=int(getattr(self.args, "gcs_official_best_warmup", 0) or 0),
            device=str(getattr(self.args, "device", "0")),
            half=bool(getattr(self.args, "gcs_official_best_half", False)),
            runtime_ms=1.0,
            save_dir=str(save_dir),
            baseline_fp=None,
            baseline_fn=None,
            fp_tol=0.01,
            fn_tol=0.01,
            score_fp_weight=float(getattr(self.args, "gcs_official_best_score_fp_weight", 0.02)),
            score_fn_weight=float(getattr(self.args, "gcs_official_best_score_fn_weight", 0.02)),
            count_acc3_weight=float(getattr(self.args, "gcs_official_best_count_acc3_weight", 0.0) or 0.0),
            count_acc4_weight=float(getattr(self.args, "gcs_official_best_count_acc4_weight", 0.006) or 0.0),
            count_acc5_weight=float(getattr(self.args, "gcs_official_best_count_acc5_weight", 0.004) or 0.0),
            rate_4_to_5_weight=float(getattr(self.args, "gcs_official_best_rate_4_to_5_weight", 0.004) or 0.0),
            rate_3_to_5_weight=float(getattr(self.args, "gcs_official_best_rate_3_to_5_weight", 0.0025) or 0.0),
            rate_4_to_3_weight=float(getattr(self.args, "gcs_official_best_rate_4_to_3_weight", 0.0015) or 0.0),
            rate_3_to_4_weight=float(getattr(self.args, "gcs_official_best_rate_3_to_4_weight", 0.001) or 0.0),
            rate_5_to_4_weight=float(getattr(self.args, "gcs_official_best_rate_5_to_4_weight", 0.0) or 0.0),
            min_count_acc3=float(getattr(self.args, "gcs_official_best_min_count_acc3", -1.0)),
            min_count_acc4=float(getattr(self.args, "gcs_official_best_min_count_acc4", -1.0)),
            min_count_acc5=float(getattr(self.args, "gcs_official_best_min_count_acc5", -1.0)),
            min_gt5_output5_rate=float(getattr(self.args, "gcs_official_best_min_gt5_output5_rate", 0.80)),
            max_gt5_count_head_under_rate=float(
                getattr(self.args, "gcs_official_best_max_gt5_count_head_under_rate", 0.15)
            ),
            max_gt5_valid_points_fail_rate=float(
                getattr(self.args, "gcs_official_best_max_gt5_valid_points_fail_rate", 0.10)
            ),
            max_rate_3_to_4=float(getattr(self.args, "gcs_official_best_max_rate_3_to_4", -1.0)),
            max_rate_3_to_5=float(getattr(self.args, "gcs_official_best_max_rate_3_to_5", -1.0)),
            max_rate_4_to_3=float(getattr(self.args, "gcs_official_best_max_rate_4_to_3", -1.0)),
            max_rate_4_to_5=float(getattr(self.args, "gcs_official_best_max_rate_4_to_5", -1.0)),
            max_rate_5_to_4=float(getattr(self.args, "gcs_official_best_max_rate_5_to_4", -1.0)),
            select_best_metric="official_acc",
        )
        if not sweep_args.confs or not sweep_args.point_valid_thrs:
            raise ValueError("GCS official best sweep requires non-empty conf and point-valid threshold lists.")

        LOGGER.info(f"Running GCS official best sweep at epoch {epoch_num} on {gt_json}")
        output = run_sweep(sweep_args)
        best = output["best"]
        fitness = float(best["official_acc"])
        score_fp_weight = float(sweep_args.score_fp_weight)
        score_fn_weight = float(sweep_args.score_fn_weight)
        count_acc3_weight = float(sweep_args.count_acc3_weight)
        count_acc5_weight = float(sweep_args.count_acc5_weight)
        count_acc4_weight = float(sweep_args.count_acc4_weight)
        rate_4_to_5_weight = float(sweep_args.rate_4_to_5_weight)
        rate_3_to_5_weight = float(sweep_args.rate_3_to_5_weight)
        rate_4_to_3_weight = float(sweep_args.rate_4_to_3_weight)
        rate_3_to_4_weight = float(sweep_args.rate_3_to_4_weight)
        rate_5_to_4_weight = float(sweep_args.rate_5_to_4_weight)
        selector = {
            "metric": "official_acc",
            "tie_breakers": [],
            "cross_epoch_tie_breakers": [],
            "selection_constraints_mode": "diagnostic_only",
            "diagnostic_official_score_formula": (
                f"official_acc - {score_fp_weight:g} * official_fp - {score_fn_weight:g} * official_fn"
            ),
            "diagnostic_balanced_score_formula": (
                f"official_score + {count_acc4_weight:g} * count_acc_4 "
                f"+ {count_acc5_weight:g} * count_acc_5 "
                f"+ {count_acc3_weight:g} * count_acc_3 "
                f"- {rate_4_to_5_weight:g} * rate_4_to_5 "
                f"- {rate_3_to_5_weight:g} * rate_3_to_5 "
                f"- {rate_4_to_3_weight:g} * rate_4_to_3 "
                f"- {rate_3_to_4_weight:g} * rate_3_to_4 "
                f"- {rate_5_to_4_weight:g} * rate_5_to_4"
            ),
            "score_fp_weight": score_fp_weight,
            "score_fn_weight": score_fn_weight,
            "count_acc3_weight": count_acc3_weight,
            "count_acc5_weight": count_acc5_weight,
            "count_acc4_weight": count_acc4_weight,
            "rate_4_to_5_weight": rate_4_to_5_weight,
            "rate_3_to_5_weight": rate_3_to_5_weight,
            "rate_4_to_3_weight": rate_4_to_3_weight,
            "rate_3_to_4_weight": rate_3_to_4_weight,
            "rate_5_to_4_weight": rate_5_to_4_weight,
            "selection_constraints": best.get("selection_constraints", {}),
            "period": period,
            "gt_json": gt_json,
        }
        record_path = self.save_dir / "official_best_summary.json"
        previous_record = None
        previous_best = None
        if record_path.exists():
            try:
                previous_record = json.loads(record_path.read_text(encoding="utf-8"))
                if not isinstance(previous_record, dict):
                    raise TypeError("official_best_summary.json must contain a JSON object")
                loaded_best = previous_record.get("best")
                if isinstance(loaded_best, dict):
                    previous_best = loaded_best
                elif previous_record.get("best_fitness", None) is not None:
                    previous_best = {"official_acc": float(previous_record["best_fitness"])}
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                previous_record = None
                previous_best = None
        record = {
            "best_epoch": epoch_num,
            "best_fitness": fitness,
            "best": best,
            "selector": selector,
        }
        candidate_record = {
            "candidate_epoch": epoch_num,
            "candidate_fitness": fitness,
            "candidate": best,
            "selector": selector,
        }
        top_k = self._official_best_top_k()
        official_top_k = self._update_official_topk(candidate_record, previous_record, top_k)
        self._load_official_best_fitness()
        if previous_best is None and self.gcs_official_best_fitness is not None:
            previous_best = {"official_acc": float(self.gcs_official_best_fitness)}
        should_update = self._official_best_candidate_is_better(best, previous_best)
        if should_update:
            self.gcs_official_best_fitness = fitness
            shutil.copy2(self.last, self.wdir / "official_best.pt")
            LOGGER.info(
                f"GCS official_best.pt updated at epoch {epoch_num}: official_acc={fitness:.6f}; "
                "weights/best.pt remains the ordinary val-F1 best."
            )
        else:
            if previous_record is not None:
                record = previous_record
                record["last_candidate"] = candidate_record
            else:
                record = {
                    "best_epoch": None,
                    "best_fitness": (
                        float(self.gcs_official_best_fitness) if self.gcs_official_best_fitness is not None else None
                    ),
                    "best": None,
                    "last_candidate": candidate_record,
                    "selector": selector,
                }
            LOGGER.info(
                f"GCS official best unchanged at epoch {epoch_num}: current={fitness:.6f}, "
                f"best={self.gcs_official_best_fitness:.6f}"
            )
        if top_k > 1:
            record["official_top_k_size"] = top_k
            record["official_top_k"] = official_top_k
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def save_model(self):
        """Save checkpoints with explicit rectangular GCS imgsz in train_args."""
        old_imgsz = self.args.imgsz
        shape = self._resolve_gcs_imgsz()
        self.args.gcs_imgsz = [int(shape[0]), int(shape[1])]
        self.args.imgsz = [int(shape[0]), int(shape[1])]
        try:
            saved = super().save_model()
            if saved:
                self._run_official_best_sweep()
            return saved
        finally:
            self.args.imgsz = old_imgsz

    def label_loss_items(self, loss_items: list[float] | torch.Tensor | None = None, prefix: str = "train"):
        """Return named GCS loss items for logging."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        return dict(zip(keys, [round(float(x), 5) for x in loss_items]))

    def progress_loss_items(self, loss_items):
        """Hide matcher diagnostics from the live training progress bar."""
        if loss_items is None:
            return loss_items
        return loss_items[: len(self.progress_loss_names)] if len(loss_items.shape) else loss_items

    def _progress_columns(self) -> tuple[str, ...]:
        """Return live progress column names in display order."""
        return ("Epoch", "GPU_mem", *self.progress_loss_names, "Lanes", "Size")

    def _progress_column_widths(self) -> tuple[int, ...]:
        """Return widths wide enough for full GCS progress headers and values."""
        return tuple(max(11, len(name) + 2) for name in self._progress_columns())

    def progress_string(self):
        """Return a progress header matching the GCS loss vector."""
        return "\n" + "".join(
            f"{name:>{width}s}" for name, width in zip(self._progress_columns(), self._progress_column_widths())
        )

    def format_progress_values(self, epoch, progress_loss_items, batch):
        """Return a GCS progress row aligned to the full loss-name header."""
        loss_length = progress_loss_items.shape[0] if len(progress_loss_items.shape) else 1
        loss_values = (
            progress_loss_items if loss_length > 1 else torch.unsqueeze(progress_loss_items, 0)
        )
        widths = self._progress_column_widths()
        values = [
            f"{epoch + 1}/{self.epochs}",
            f"{self._get_memory():.3g}G",
            *[float(x) for x in loss_values],
            batch["cls"].shape[0],
            batch["img"].shape[-1],
        ]
        parts = [f"{values[0]:>{widths[0]}s}", f"{values[1]:>{widths[1]}s}"]
        parts.extend(f"{value:>{width}.4g}" for value, width in zip(values[2:-2], widths[2:-2]))
        parts.extend(f"{value:>{width}.4g}" for value, width in zip(values[-2:], widths[-2:]))
        return "".join(parts)

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
        head = self._gcs_head()
        max_num_obj = int(getattr(head, "num_queries", 12) or 12)
        return super().auto_batch(max_num_obj=max_num_obj, dataset_size=n)
