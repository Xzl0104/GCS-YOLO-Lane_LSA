"""Ordered-slot loss for GCS-YOLO-Lane."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.gcs.slot_targets import build_ordered_lane_slots_batch
from ultralytics.utils.gcs_point_loss import (
    aspect_weighted_l1_point_loss,
    normalized_smooth_l1_point_loss,
    pixel_smooth_l1_point_loss,
)
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, assert_gcs_shape, normalize_imgsz


class OrderedSlotGCSLoss(nn.Module):
    """Deterministic left-to-right slot loss for ordered-slot GCS heads."""

    loss_names = (
        "slot_point_loss",
        "slot_exist_loss",
        "slot_count_ce_loss",
        "slot_interval_loss",
        "slot_valid_loss",
        "slot_order_loss",
        "slot_gt_bottom_order_loss",
        "slot_decoded_bottom_order_loss",
        "slot_count_acc",
        "slot_count_acc_2",
        "slot_count_acc_3",
        "slot_count_acc_4",
        "slot_count_acc_5",
        "slot_count_correct_2",
        "slot_count_correct_3",
        "slot_count_correct_4",
        "slot_count_correct_5",
        "slot_count_total_2",
        "slot_count_total_3",
        "slot_count_total_4",
        "slot_count_total_5",
        "slot_exist_acc",
        "slot4_exist_acc",
        "slot5_exist_acc",
        "interval_start_mae",
        "interval_end_mae",
        "order_violation_rate",
        "slot_order_common_violation_rate",
        "slot_order_gt_bottom_violation_rate",
        "slot_order_decoded_bottom_violation_rate",
        "slot_order_decoded_bottom_pair_total",
        "slot_order_decoded_bottom_violation_pairs",
        "point_l1_px",
        "repaired_noncontiguous_lanes",
        "repaired_hole_points",
    )
    progress_loss_names = (
        "slot_pt",
        "slot_ex",
        "cnt_ce",
        "interval",
        "valid",
        "order",
        "gt_bot_ord",
        "dec_bot_ord",
        "cnt_acc",
        "cnt_a2",
        "cnt_a3",
        "cnt_a4",
        "cnt_a5",
        "cnt_c2",
        "cnt_c3",
        "cnt_c4",
        "cnt_c5",
        "cnt_t2",
        "cnt_t3",
        "cnt_t4",
        "cnt_t5",
        "ex_acc",
        "s4_acc",
        "s5_acc",
        "st_mae",
        "end_mae",
        "ord_bad",
        "ord_common",
        "ord_gtbot",
        "ord_decbot",
        "dec_pairs",
        "dec_bad",
        "l1_px",
        "rep_lane",
        "rep_hole",
    )

    def __init__(
        self,
        model=None,
        k: int | None = None,
        num_slots: int | None = None,
        image_size=None,
        min_valid_points: int = 2,
    ):
        """Initialize ordered-slot objective from trainer/model args."""
        super().__init__()
        args = model if isinstance(model, dict) else getattr(model, "args", None)
        yaml = getattr(model, "yaml", {}) if model is not None else {}
        self.k = int(k or self._arg(args, "gcs_k", yaml.get("gcs_k", 56)))
        self.num_slots = int(num_slots or self._arg(args, "gcs_num_slots", yaml.get("gcs_num_slots", 5)))
        self.min_lanes = int(self._arg(args, "gcs_min_lanes", yaml.get("gcs_min_lanes", 2)))
        self.max_lanes = int(self._arg(args, "gcs_max_lanes", yaml.get("gcs_max_lanes", self.num_slots)))
        self.count_classes = int(
            self._arg(args, "gcs_count_classes", yaml.get("gcs_count_classes", self.max_lanes - self.min_lanes + 1))
        )
        self.contiguity_policy = str(
            self._arg(args, "gcs_contiguity_policy", yaml.get("gcs_contiguity_policy", "strict"))
        ).strip().lower()
        self.min_valid_points = int(min_valid_points)
        if self.num_slots != 5:
            raise ValueError(f"ordered_slot currently requires exactly 5 slots, got {self.num_slots}.")
        if self.k != 56:
            raise ValueError(f"ordered_slot fixed-y contract requires K=56, got {self.k}.")
        if (self.min_lanes, self.max_lanes) != (2, 5):
            raise ValueError(
                "ordered_slot_v2 supports 2/3/4/5 lane-count classes and requires "
                f"gcs_min_lanes=2, gcs_max_lanes=5. Got min_lanes={self.min_lanes}, "
                f"max_lanes={self.max_lanes}."
            )
        expected_count_classes = self.max_lanes - self.min_lanes + 1
        if self.count_classes != expected_count_classes:
            raise ValueError(
                f"ordered_slot count_classes must be max_lanes-min_lanes+1={expected_count_classes}, "
                f"got {self.count_classes}."
            )
        if self.contiguity_policy not in {"strict", "repair_interp"}:
            raise ValueError("ordered_slot gcs_contiguity_policy must be 'strict' or 'repair_interp'.")

        self.point_gain = float(self._arg(args, "gcs_point", 15.0))
        self.exist_gain = float(self._arg(args, "gcs_exist", 2.0))
        self.point_valid_gain = float(self._arg(args, "gcs_point_valid", 1.0))
        self.count_ce_gain = float(self._arg(args, "gcs_count_ce", 1.0))
        self.interval_gain = float(self._arg(args, "gcs_interval", 1.0))
        self.order_gain = float(self._arg(args, "gcs_order", 0.2))
        self.gt_bottom_order_gain = float(self._arg(args, "gcs_gt_bottom_order", 1.0))
        self.decoded_bottom_order_gain = float(self._arg(args, "gcs_decoded_bottom_order", 1.0))
        self.allow_disable_order_loss = self._bool_arg(self._arg(args, "gcs_allow_disable_order_loss", False))
        self.slot_exist_w4 = float(self._arg(args, "gcs_slot_exist_w4", 1.0))
        self.slot_exist_w5 = float(self._arg(args, "gcs_slot_exist_w5", 1.0))
        self.order_margin_px = float(self._arg(args, "gcs_order_margin_px", 5.0))
        self.bottom_order_margin_px = float(self._arg(args, "gcs_bottom_order_margin_px", 2.0))
        self.ordered_point_loss = str(self._arg(args, "gcs_ordered_point_loss", "normalized_smooth_l1")).strip().lower()
        self.point_y_weight = float(self._arg(args, "gcs_point_y_weight", 0.25))
        self.point_x_only = self._bool_arg(self._arg(args, "gcs_point_x_only", False))
        self.pixel_smoothl1_beta = float(self._arg(args, "gcs_pixel_smoothl1_beta", 1.0))
        if min(
            self.point_gain,
            self.exist_gain,
            self.point_valid_gain,
            self.count_ce_gain,
            self.interval_gain,
            self.order_gain,
            self.gt_bottom_order_gain,
            self.decoded_bottom_order_gain,
        ) < 0:
            raise ValueError("ordered_slot loss gains must be non-negative.")
        if self.count_ce_gain <= 0.0:
            raise RuntimeError(
                "ordered_slot requires gcs_count_ce > 0. "
                "Count-class supervision cannot be disabled silently."
            )
        if self.interval_gain <= 0.0:
            raise RuntimeError(
                "ordered_slot requires gcs_interval > 0. "
                "Visibility interval supervision cannot be disabled silently."
            )
        if self.order_gain <= 0.0 and not self.allow_disable_order_loss:
            raise RuntimeError(
                "ordered_slot order loss is disabled. "
                "Pass --gcs-allow-disable-order-loss only for an explicit ablation."
            )
        if min(self.slot_exist_w4, self.slot_exist_w5) <= 0:
            raise ValueError("gcs_slot_exist_w4 and gcs_slot_exist_w5 must be positive.")
        if self.ordered_point_loss not in {"aspect_l1", "pixel_smooth_l1", "normalized_smooth_l1"}:
            raise ValueError(
                "gcs_ordered_point_loss must be one of "
                "{'aspect_l1', 'pixel_smooth_l1', 'normalized_smooth_l1'}, "
                f"got {self.ordered_point_loss!r}."
            )
        if self.point_y_weight < 0.0:
            raise ValueError(f"gcs_point_y_weight must be >= 0, got {self.point_y_weight}.")
        if self.pixel_smoothl1_beta <= 0.0:
            raise ValueError(f"gcs_pixel_smoothl1_beta must be > 0, got {self.pixel_smoothl1_beta}.")

        image_size = (
            image_size
            or self._arg(args, "gcs_imgsz", None)
            or self._arg(args, "image_shape", None)
            or getattr(model, "gcs_imgsz", None)
            or yaml.get("gcs_imgsz")
            or yaml.get("image_shape")
            or yaml.get("imgsz")
        )
        if image_size is None or image_size == "":
            raise ValueError("OrderedSlotGCSLoss requires explicit image_size/gcs_imgsz.")
        self.image_size = normalize_imgsz(image_size)
        h, w = self.image_size
        self.register_buffer("point_scale", torch.tensor((float(w) / max(h, w), float(h) / max(h, w))).view(1, 1, 1, 2), persistent=False)
        self.register_buffer("pixel_scale", torch.tensor((float(w), float(h))).view(1, 1, 1, 2), persistent=False)

    def loss_contract_summary(self) -> dict[str, object]:
        """Return the effective ordered-slot loss supervision contract."""
        return {
            "gcs_point": float(self.point_gain),
            "gcs_exist": float(self.exist_gain),
            "gcs_point_valid": float(self.point_valid_gain),
            "gcs_count_ce": float(self.count_ce_gain),
            "gcs_interval": float(self.interval_gain),
            "gcs_order": float(self.order_gain),
            "gcs_gt_bottom_order": float(self.gt_bottom_order_gain),
            "gcs_decoded_bottom_order": float(self.decoded_bottom_order_gain),
            "gcs_order_margin_px": float(self.order_margin_px),
            "gcs_bottom_order_margin_px": float(self.bottom_order_margin_px),
            "gcs_allow_disable_order_loss": bool(self.allow_disable_order_loss),
            "count_supervision_enabled": self.count_ce_gain > 0.0,
            "interval_supervision_enabled": self.interval_gain > 0.0,
            "order_supervision_enabled": self.order_gain > 0.0,
            "gt_bottom_order_supervision_enabled": self.gt_bottom_order_gain > 0.0,
            "decoded_bottom_order_supervision_enabled": self.decoded_bottom_order_gain > 0.0,
            "slot4_exist_bce_weight": float(self.slot_exist_w4),
            "slot5_exist_bce_weight": float(self.slot_exist_w5),
            "slot_exist_weight_semantics": "BCE element weight applied to positive and negative targets",
        }

    @staticmethod
    def _arg(args, name: str, default):
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    @staticmethod
    def _bool_arg(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _zero_like(pred_points: torch.Tensor) -> torch.Tensor:
        return pred_points.sum() * 0.0

    @classmethod
    def label_loss_items(cls, loss_items: list[float] | torch.Tensor | None = None, prefix: str = "train"):
        """Return ordered-slot loss items, deriving per-count acc from accumulated correct/total."""
        keys = [f"{prefix}/{x}" for x in cls.loss_names]
        if loss_items is None:
            return keys

        values = [float(x.detach().cpu()) if isinstance(x, torch.Tensor) else float(x) for x in loss_items]
        raw = dict(zip(cls.loss_names, values))

        def fmt(value: float) -> float:
            return round(float(value), 5) if math.isfinite(float(value)) else float(value)

        out = {key: fmt(value) for key, value in zip(keys, values)}
        total_correct = 0.0
        total_count = 0.0
        for count in range(2, 6):
            correct = float(raw.get(f"slot_count_correct_{count}", 0.0))
            total = float(raw.get(f"slot_count_total_{count}", 0.0))
            total_correct += correct
            total_count += total
            acc_key = f"{prefix}/slot_count_acc_{count}"
            out[acc_key] = fmt(correct / total) if total > 0.0 else float("nan")
        if total_count > 0.0:
            out[f"{prefix}/slot_count_acc"] = fmt(total_correct / total_count)
        return out

    @staticmethod
    def _targets_from_batch(batch: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if "lanes" in batch and "lane_valid" in batch:
            return batch["lanes"], batch["lane_valid"]
        if "gt_lanes" in batch and "gt_lane_valid" in batch:
            return batch["gt_lanes"], batch["gt_lane_valid"]
        raise KeyError("OrderedSlotGCSLoss requires batch['lanes']/batch['lane_valid'].")

    def _target_slots(self, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
        gt_points, gt_valid = self._targets_from_batch(batch)
        gt_points = [x.to(device=device, dtype=torch.float32) for x in gt_points]
        gt_valid = [x.to(device=device, dtype=torch.float32) for x in gt_valid]
        return build_ordered_lane_slots_batch(
            gt_points,
            gt_valid,
            max_lanes=self.num_slots,
            min_lanes=self.min_lanes,
            min_valid_points=self.min_valid_points,
            contiguity_policy=self.contiguity_policy,
        )

    def _normalize_preds(self, preds: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        required = ("pred_points", "pred_valid_logits", "pred_start_logits", "pred_end_logits", "pred_count_logits")
        missing = [key for key in required if key not in preds]
        if missing:
            raise KeyError(f"OrderedSlotGCSLoss missing prediction keys: {missing}.")
        pred_points = preds["pred_points"]
        pred_exist_logits = preds.get("pred_exist_logits", preds.get("pred_logits"))
        if pred_exist_logits is None:
            raise KeyError("OrderedSlotGCSLoss requires pred_exist_logits or pred_logits.")
        pred_valid_logits = preds["pred_valid_logits"]
        pred_start_logits = preds["pred_start_logits"]
        pred_end_logits = preds["pred_end_logits"]
        pred_count_logits = preds["pred_count_logits"]
        if pred_exist_logits.ndim == 3 and pred_exist_logits.shape[-1] == 1:
            pred_exist_logits = pred_exist_logits.squeeze(-1)
        expected_points = (pred_points.shape[0], self.num_slots, self.k, 2)
        if tuple(pred_points.shape) != expected_points:
            raise ValueError(f"ordered_slot pred_points must be {expected_points}, got {tuple(pred_points.shape)}.")
        for name, tensor, expected in (
            ("pred_exist_logits", pred_exist_logits, pred_points.shape[:2]),
            ("pred_valid_logits", pred_valid_logits, pred_points.shape[:3]),
            ("pred_start_logits", pred_start_logits, pred_points.shape[:3]),
            ("pred_end_logits", pred_end_logits, pred_points.shape[:3]),
        ):
            if tuple(tensor.shape) != tuple(expected):
                raise ValueError(f"{name} shape must be {tuple(expected)}, got {tuple(tensor.shape)}.")
        if tuple(pred_count_logits.shape) != (pred_points.shape[0], self.count_classes):
            raise ValueError(
                f"pred_count_logits must have shape B x {self.count_classes}, got {tuple(pred_count_logits.shape)}."
            )
        return pred_points, pred_exist_logits, pred_valid_logits, pred_start_logits, pred_end_logits, pred_count_logits

    def _point_loss(self, pred_points, slot_points, slot_valid, slot_exist):
        mask = (slot_exist > 0.5).unsqueeze(-1) & (slot_valid > 0.5)
        if not bool(mask.any()):
            return self._zero_like(pred_points)
        if self.ordered_point_loss == "aspect_l1":
            return aspect_weighted_l1_point_loss(
                pred_points,
                slot_points,
                mask,
                image_size=self.image_size,
                y_weight=self.point_y_weight,
                x_only=self.point_x_only,
            )
        if self.ordered_point_loss == "pixel_smooth_l1":
            return pixel_smooth_l1_point_loss(
                pred_points,
                slot_points,
                mask,
                image_size=self.image_size,
                beta=self.pixel_smoothl1_beta,
                y_weight=self.point_y_weight,
                x_only=self.point_x_only,
            )
        return normalized_smooth_l1_point_loss(
            pred_points,
            slot_points,
            mask,
            image_size=self.image_size,
            y_weight=self.point_y_weight,
            x_only=self.point_x_only,
        )

    def _exist_loss(self, pred_exist_logits, slot_exist):
        weight = torch.ones_like(slot_exist)
        if self.num_slots >= 4:
            weight[:, 3] = float(self.slot_exist_w4)
        if self.num_slots >= 5:
            weight[:, 4] = float(self.slot_exist_w5)
        bce = F.binary_cross_entropy_with_logits(pred_exist_logits, slot_exist, reduction="none")
        return (bce * weight).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def _masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, slot_exist: torch.Tensor) -> torch.Tensor:
        pos = slot_exist > 0.5
        if not bool(pos.any()):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[pos], target[pos], reduction="mean")

    def _valid_aux_loss(self, pred_valid_logits, slot_valid, slot_exist):
        pos = slot_exist > 0.5
        if not bool(pos.any()):
            return self._zero_like(pred_valid_logits)
        return F.binary_cross_entropy_with_logits(pred_valid_logits[pos], slot_valid[pos].float(), reduction="mean")

    def _order_loss(self, pred_points, slot_valid, slot_exist):
        margin = float(self.order_margin_px) / float(self.image_size[1])
        losses = []
        for s in range(self.num_slots - 1):
            exist_pair = (slot_exist[:, s] > 0.5) & (slot_exist[:, s + 1] > 0.5)
            valid_pair = (slot_valid[:, s] > 0.5) & (slot_valid[:, s + 1] > 0.5) & exist_pair[:, None]
            if not bool(valid_pair.any()):
                continue
            violation = F.relu(pred_points[:, s, :, 0] - pred_points[:, s + 1, :, 0] + margin)
            losses.append(violation[valid_pair].mean())
        return torch.stack(losses).mean() if losses else self._zero_like(pred_points)

    @staticmethod
    def _adjacent_exist_pair_mask(slot_exist: torch.Tensor) -> torch.Tensor:
        return (slot_exist[:, :-1] > 0.5) & (slot_exist[:, 1:] > 0.5)

    @staticmethod
    def _gather_bottom_x(pred_points: torch.Tensor, bottom_idx: torch.Tensor) -> torch.Tensor:
        idx = bottom_idx.to(device=pred_points.device, dtype=torch.long).clamp(0, pred_points.shape[2] - 1)
        return pred_points[..., 0].gather(dim=2, index=idx.unsqueeze(-1)).squeeze(-1)

    def _gt_bottom_order_loss(self, pred_points, slot_exist, start_labels):
        pair_mask = self._adjacent_exist_pair_mask(slot_exist)
        if not bool(pair_mask.any()):
            return self._zero_like(pred_points)
        margin = float(self.bottom_order_margin_px) / float(self.image_size[1])
        bottom_x = self._gather_bottom_x(pred_points, start_labels)
        violation = F.relu(bottom_x[:, :-1] - bottom_x[:, 1:] + margin)
        return violation[pair_mask].mean()

    def _decoded_bottom_order_loss(self, pred_points, pred_start_logits, slot_exist):
        pair_mask = self._adjacent_exist_pair_mask(slot_exist)
        if not bool(pair_mask.any()):
            return self._zero_like(pred_points)
        margin = float(self.bottom_order_margin_px) / float(self.image_size[1])
        decoded_bottom_idx = pred_start_logits.detach().float().argmax(dim=-1)
        bottom_x = self._gather_bottom_x(pred_points, decoded_bottom_idx)
        violation = F.relu(bottom_x[:, :-1] - bottom_x[:, 1:] + margin)
        return violation[pair_mask].mean()

    @torch.no_grad()
    def _metrics(
        self,
        pred_points,
        pred_exist_logits,
        pred_count_logits,
        pred_start_logits,
        pred_end_logits,
        slot_points,
        slot_valid,
        slot_exist,
        count_label,
        start_labels,
        end_labels,
    ) -> tuple[torch.Tensor, ...]:
        pred_count = pred_count_logits.argmax(dim=-1) + self.min_lanes
        gt_count = count_label + self.min_lanes
        count_correct_mask = pred_count.eq(gt_count)
        slot_count_acc = count_correct_mask.float().mean()
        by_count_acc = []
        by_count_correct = []
        by_count_total = []
        for count in range(self.min_lanes, self.max_lanes + 1):
            mask = gt_count == count
            total = mask.sum().to(dtype=pred_points.dtype)
            correct = count_correct_mask[mask].float().sum().to(dtype=pred_points.dtype) if bool(mask.any()) else self._zero_like(pred_points)
            acc = correct / total if bool(mask.any()) else pred_points.new_tensor(float("nan"))
            by_count_acc.append(acc)
            by_count_correct.append(correct)
            by_count_total.append(total)

        pred_exist = pred_exist_logits.sigmoid() >= 0.5
        gt_exist = slot_exist > 0.5
        slot_exist_acc = pred_exist.eq(gt_exist).float().mean()
        slot4_exist_acc = pred_exist[:, 3].eq(gt_exist[:, 3]).float().mean() if self.num_slots >= 4 else self._zero_like(pred_points)
        slot5_exist_acc = pred_exist[:, 4].eq(gt_exist[:, 4]).float().mean() if self.num_slots >= 5 else self._zero_like(pred_points)

        pos = slot_exist > 0.5
        if bool(pos.any()):
            pred_start = pred_start_logits.argmax(dim=-1)
            pred_end = pred_end_logits.argmax(dim=-1)
            interval_start_mae = (pred_start[pos] - start_labels[pos]).abs().float().mean()
            interval_end_mae = (pred_end[pos] - end_labels[pos]).abs().float().mean()
            point_mask = pos.unsqueeze(-1) & (slot_valid > 0.5)
            if bool(point_mask.any()):
                point_l1_px = (pred_points[..., 0] - slot_points[..., 0]).abs()[point_mask].mean() * float(self.image_size[1])
            else:
                point_l1_px = self._zero_like(pred_points)
        else:
            interval_start_mae = self._zero_like(pred_points)
            interval_end_mae = self._zero_like(pred_points)
            point_l1_px = self._zero_like(pred_points)

        order_total = pred_points.new_zeros(())
        order_bad = pred_points.new_zeros(())
        for s in range(self.num_slots - 1):
            exist_pair = (slot_exist[:, s] > 0.5) & (slot_exist[:, s + 1] > 0.5)
            valid_pair = (slot_valid[:, s] > 0.5) & (slot_valid[:, s + 1] > 0.5) & exist_pair[:, None]
            if bool(valid_pair.any()):
                order_total = order_total + valid_pair.sum().to(dtype=pred_points.dtype)
                order_bad = order_bad + ((pred_points[:, s, :, 0] >= pred_points[:, s + 1, :, 0]) & valid_pair).sum().to(dtype=pred_points.dtype)
        order_violation_rate = order_bad / order_total.clamp_min(1.0)

        pair_mask = self._adjacent_exist_pair_mask(slot_exist)
        pair_total = pair_mask.sum().to(dtype=pred_points.dtype)
        gt_bottom_x = self._gather_bottom_x(pred_points, start_labels)
        gt_bottom_bad = ((gt_bottom_x[:, :-1] > gt_bottom_x[:, 1:]) & pair_mask).sum().to(dtype=pred_points.dtype)
        gt_bottom_violation_rate = gt_bottom_bad / pair_total.clamp_min(1.0)

        decoded_bottom_idx = pred_start_logits.detach().float().argmax(dim=-1)
        decoded_bottom_x = self._gather_bottom_x(pred_points, decoded_bottom_idx)
        decoded_bottom_bad = ((decoded_bottom_x[:, :-1] > decoded_bottom_x[:, 1:]) & pair_mask).sum().to(dtype=pred_points.dtype)
        decoded_bottom_violation_rate = decoded_bottom_bad / pair_total.clamp_min(1.0)
        return (
            slot_count_acc,
            by_count_acc[0],
            by_count_acc[1],
            by_count_acc[2],
            by_count_acc[3],
            by_count_correct[0],
            by_count_correct[1],
            by_count_correct[2],
            by_count_correct[3],
            by_count_total[0],
            by_count_total[1],
            by_count_total[2],
            by_count_total[3],
            slot_exist_acc,
            slot4_exist_acc,
            slot5_exist_acc,
            interval_start_mae,
            interval_end_mae,
            order_violation_rate,
            order_violation_rate,
            gt_bottom_violation_rate,
            decoded_bottom_violation_rate,
            pair_total,
            decoded_bottom_bad,
            point_l1_px,
        )

    def forward(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        pred_points, pred_exist_logits, pred_valid_logits, pred_start_logits, pred_end_logits, pred_count_logits = self._normalize_preds(preds)
        if "img" in batch:
            assert_gcs_image_tensor(batch["img"], self.image_size, name="batch['img']", context="OrderedSlotGCSLoss.forward")
        for key in ("semantic_mask", "edge_mask"):
            if key in batch:
                assert_gcs_shape(batch[key].shape[-2:], self.image_size, name=f"batch['{key}']", context="OrderedSlotGCSLoss.forward")

        pred_points = pred_points.float()
        pred_exist_logits = pred_exist_logits.float()
        pred_valid_logits = pred_valid_logits.float()
        pred_start_logits = pred_start_logits.float()
        pred_end_logits = pred_end_logits.float()
        pred_count_logits = pred_count_logits.float()

        targets = self._target_slots(batch, device=pred_points.device)
        slot_points = targets["slot_points"].to(device=pred_points.device, dtype=torch.float32)
        slot_valid = targets["slot_valid"].to(device=pred_points.device, dtype=torch.float32)
        slot_exist = targets["slot_exist"].to(device=pred_points.device, dtype=torch.float32)
        count_label = targets["count_label"].to(device=pred_points.device, dtype=torch.long)
        start_labels = targets["start_labels"].to(device=pred_points.device, dtype=torch.long)
        end_labels = targets["end_labels"].to(device=pred_points.device, dtype=torch.long)
        repaired_noncontiguous_lanes = targets["repaired_noncontiguous_lanes"].to(device=pred_points.device, dtype=torch.float32)
        repaired_hole_points = targets["repaired_hole_points"].to(device=pred_points.device, dtype=torch.float32)

        point_loss = self._point_loss(pred_points, slot_points, slot_valid, slot_exist)
        exist_loss = self._exist_loss(pred_exist_logits, slot_exist)
        count_loss = F.cross_entropy(pred_count_logits, count_label, reduction="mean")
        start_loss = self._masked_cross_entropy(pred_start_logits, start_labels, slot_exist)
        end_loss = self._masked_cross_entropy(pred_end_logits, end_labels, slot_exist)
        interval_loss = 0.5 * (start_loss + end_loss)
        valid_loss = self._valid_aux_loss(pred_valid_logits, slot_valid, slot_exist)
        order_loss = self._order_loss(pred_points, slot_valid, slot_exist)
        gt_bottom_order_loss = self._gt_bottom_order_loss(pred_points, slot_exist, start_labels)
        decoded_bottom_order_loss = self._decoded_bottom_order_loss(pred_points, pred_start_logits, slot_exist)

        total = (
            self.point_gain * point_loss
            + self.exist_gain * exist_loss
            + self.count_ce_gain * count_loss
            + self.interval_gain * interval_loss
            + self.point_valid_gain * valid_loss
            + self.order_gain * order_loss
            + self.gt_bottom_order_gain * gt_bottom_order_loss
            + self.decoded_bottom_order_gain * decoded_bottom_order_loss
        )
        metrics = self._metrics(
            pred_points,
            pred_exist_logits,
            pred_count_logits,
            pred_start_logits,
            pred_end_logits,
            slot_points,
            slot_valid,
            slot_exist,
            count_label,
            start_labels,
            end_labels,
        )
        loss_items = torch.stack(
            (
                point_loss.detach(),
                exist_loss.detach(),
                count_loss.detach(),
                interval_loss.detach(),
                valid_loss.detach(),
                order_loss.detach(),
                gt_bottom_order_loss.detach(),
                decoded_bottom_order_loss.detach(),
                *(x.detach() for x in metrics),
                repaired_noncontiguous_lanes.float().mean().detach(),
                repaired_hole_points.float().mean().detach(),
            )
        )
        return total, loss_items
