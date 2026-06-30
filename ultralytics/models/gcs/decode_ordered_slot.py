"""Deterministic decoder for ordered-slot GCS lane predictions."""

from __future__ import annotations

import logging

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def repair_interval(start: int, end: int, k: int, min_len: int = 2) -> tuple[int, int]:
    """Clamp and expand an ordered-slot interval without dropping its slot."""
    k = int(k)
    if k <= 0:
        raise ValueError(f"ordered_slot interval repair requires K > 0, got {k}.")
    min_len = max(1, min(int(min_len), k))
    start, end = int(start), int(end)
    if start > end:
        start, end = end, start
    start = max(0, min(k - 1, start))
    end = max(0, min(k - 1, end))
    if end - start + 1 >= min_len:
        return start, end

    center = (start + end) // 2
    new_start = center - (min_len // 2)
    new_end = new_start + min_len - 1
    if new_start < 0:
        new_start = 0
        new_end = min_len - 1
    if new_end >= k:
        new_end = k - 1
        new_start = k - min_len
    return max(0, new_start), min(k - 1, new_end)


def decoded_bottom_idx_from_start_end_logits(
    pred_start_logits: torch.Tensor,
    pred_end_logits: torch.Tensor,
    min_interval_points: int = 2,
) -> torch.Tensor:
    """Return the bottom index that strict ordered-slot decode uses after interval repair."""
    if pred_start_logits.shape != pred_end_logits.shape:
        raise ValueError(
            "pred_start_logits and pred_end_logits must have identical shape, "
            f"got {tuple(pred_start_logits.shape)} vs {tuple(pred_end_logits.shape)}."
        )
    if pred_start_logits.ndim != 3:
        raise ValueError(f"pred_start_logits must have shape B x S x K, got {tuple(pred_start_logits.shape)}.")
    k = int(pred_start_logits.shape[-1])
    if k <= 0:
        raise ValueError(f"ordered_slot interval repair requires K > 0, got {k}.")

    min_len = max(1, min(int(min_interval_points), k))
    start = pred_start_logits.detach().float().argmax(dim=-1).long()
    end = pred_end_logits.detach().float().argmax(dim=-1).long()

    lo = torch.minimum(start, end).clamp(0, k - 1)
    hi = torch.maximum(start, end).clamp(0, k - 1)
    length = hi - lo + 1
    need_expand = length < min_len

    center = (lo + hi) // 2
    new_start = center - (min_len // 2)
    new_end = new_start + min_len - 1

    left_overflow = new_start < 0
    new_start = torch.where(left_overflow, torch.zeros_like(new_start), new_start)
    new_end = torch.where(left_overflow, torch.full_like(new_end, min_len - 1), new_end)

    right_overflow = new_end >= k
    new_end = torch.where(right_overflow, torch.full_like(new_end, k - 1), new_end)
    new_start = torch.where(right_overflow, torch.full_like(new_start, k - min_len), new_start)

    repaired_start = torch.where(need_expand, new_start, lo)
    return repaired_start.clamp(0, k - 1).long()


def lane_bottom_x_from_interval(points_norm: np.ndarray, start: int, end: int) -> float:
    """Return the bottom-most normalized x for a repaired fixed-y interval."""
    bottom_idx = int(min(start, end))
    return float(points_norm[bottom_idx, 0])


def check_ordered_slot_lane_order(
    lane_infos: list[dict],
    order_margin_px: float = 0.0,
    img_w: float = 960.0,
) -> list[dict]:
    """Check that decoded ordered slots remain left-to-right by bottom x."""
    violations: list[dict] = []
    margin_norm = float(order_margin_px) / max(float(img_w), 1.0)
    for left, right in zip(lane_infos, lane_infos[1:]):
        left_x = float(left["bottom_x_norm"])
        right_x = float(right["bottom_x_norm"])
        if left_x + margin_norm > right_x:
            violations.append(
                {
                    "left_slot": int(left["slot"]),
                    "right_slot": int(right["slot"]),
                    "left_bottom_x": left_x,
                    "right_bottom_x": right_x,
                }
            )
    return violations


def validate_ordered_slot_pred_shapes(preds: dict[str, torch.Tensor], min_lanes: int = 2, max_lanes: int = 5) -> tuple[int, int, int]:
    """Validate ordered-slot decoder prediction tensor shapes and return B,S,K."""
    required = ("pred_points", "pred_count_logits", "pred_start_logits", "pred_end_logits")
    missing = [key for key in required if key not in preds]
    if missing:
        raise KeyError(f"ordered_slot decode requires missing prediction keys: {missing}.")

    min_lanes = int(min_lanes)
    max_lanes = int(max_lanes)
    if min_lanes <= 0 or max_lanes < min_lanes:
        raise ValueError(f"ordered_slot decode lane bounds must satisfy 0 < min_lanes <= max_lanes, got {min_lanes}/{max_lanes}.")

    pred_points = preds["pred_points"]
    if pred_points.ndim != 4 or pred_points.shape[-1] != 2:
        raise ValueError(f"pred_points must have shape B x S x K x 2, got {tuple(pred_points.shape)}.")
    bsz, slots, k = (int(pred_points.shape[0]), int(pred_points.shape[1]), int(pred_points.shape[2]))
    if slots != max_lanes:
        raise ValueError(
            f"ordered_slot decode requires pred_points slot dimension S == max_lanes={max_lanes}, got S={slots}."
        )

    expected_slot_shape = (bsz, slots, k)
    for name in ("pred_start_logits", "pred_end_logits"):
        tensor = preds[name]
        if tuple(tensor.shape) != expected_slot_shape:
            raise ValueError(f"{name} must have shape B x S x K = {expected_slot_shape}, got {tuple(tensor.shape)}.")

    count_classes = max_lanes - min_lanes + 1
    pred_count_logits = preds["pred_count_logits"]
    expected_count_shape = (bsz, count_classes)
    if tuple(pred_count_logits.shape) != expected_count_shape:
        raise ValueError(
            f"pred_count_logits must have shape B x {count_classes} = {expected_count_shape} "
            f"for lane range [{min_lanes}, {max_lanes}], got {tuple(pred_count_logits.shape)}."
        )

    pred_exist_logits = preds.get("pred_exist_logits", None)
    exist_name = "pred_exist_logits"
    if pred_exist_logits is None and "pred_logits" in preds:
        pred_exist_logits = preds["pred_logits"]
        exist_name = "pred_logits"
    if pred_exist_logits is not None and tuple(pred_exist_logits.shape) != (bsz, slots):
        raise ValueError(f"{exist_name} must have shape B x S = {(bsz, slots)}, got {tuple(pred_exist_logits.shape)}.")
    return bsz, slots, k


@torch.no_grad()
def decode_ordered_slot_predictions(
    preds: dict[str, torch.Tensor],
    batch_index: int = 0,
    image_shape: tuple[int, int] | None = None,
    min_lanes: int = 2,
    max_lanes: int = 5,
    min_interval_points: int = 2,
    order_check: str = "error",
    output_order: str = "slot",
    order_margin_px: float = 0.0,
    img_w: float = 960.0,
    return_diagnostics: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Decode one image from a batch using count logits and start/end intervals only."""
    batch_size, slots, k = validate_ordered_slot_pred_shapes(preds, min_lanes=min_lanes, max_lanes=max_lanes)
    pred_points = preds["pred_points"]
    pred_count_logits = preds["pred_count_logits"]
    pred_start_logits = preds["pred_start_logits"]
    pred_end_logits = preds["pred_end_logits"]
    pred_exist_logits = preds.get("pred_exist_logits", preds.get("pred_logits"))
    min_lanes = int(min_lanes)
    max_lanes = int(max_lanes)
    if min_lanes <= 0 or max_lanes < min_lanes:
        raise ValueError(f"ordered_slot decode lane bounds must satisfy 0 < min_lanes <= max_lanes, got {min_lanes}/{max_lanes}.")

    b = int(batch_index)
    if b < 0 or b >= batch_size:
        raise IndexError(f"batch_index {b} out of range for B={batch_size}.")

    count_cls = int(pred_count_logits[b].detach().float().argmax(dim=-1).item())
    num_lanes = int(count_cls + min_lanes)
    if not (min_lanes <= num_lanes <= max_lanes):
        raise ValueError(f"decoded num_lanes={num_lanes} outside [{min_lanes}, {max_lanes}].")
    if num_lanes > slots:
        raise ValueError(f"ordered_slot count requested {num_lanes} lanes but prediction has only {slots} slots.")
    order_check = str(order_check or "error").strip().lower()
    if order_check not in {"none", "warn", "error"}:
        raise ValueError(f"order_check must be one of none/warn/error, got {order_check!r}.")
    output_order = str(output_order or "slot").strip().lower()
    if output_order not in {"slot", "left_to_right"}:
        raise ValueError(f"output_order must be one of slot/left_to_right, got {output_order!r}.")
    if output_order == "left_to_right" and order_check == "error":
        raise ValueError(
            "output_order=left_to_right sorts lanes after decoding and should not be combined with "
            "order_check=error. Use output_order=slot for strict ordered-slot evaluation, or "
            "order_check=warn for debug sorted export."
        )
    scale = None
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        scale = np.array([w, h], dtype=np.float32).reshape(1, 2)
        img_w = float(w)

    lanes: list[dict] = []
    for slot in range(num_lanes):
        exist_prob = float(pred_exist_logits[b, slot].detach().float().sigmoid().item()) if pred_exist_logits is not None else 1.0

        start = int(pred_start_logits[b, slot].detach().float().argmax(dim=-1).item())
        end = int(pred_end_logits[b, slot].detach().float().argmax(dim=-1).item())
        start, end = repair_interval(start, end, k, min_interval_points)

        points_norm = pred_points[b, slot].detach().float().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
        bottom_x_norm = lane_bottom_x_from_interval(points_norm, start=start, end=end)
        valid = np.zeros((k,), dtype=np.float32)
        valid[start : end + 1] = 1.0
        item = {
            "score": exist_prob,
            "query": int(slot),
            "slot": int(slot),
            "start": int(start),
            "end": int(end),
            "bottom_x_norm": bottom_x_norm,
            "points_norm": points_norm,
            "point_valid": valid,
            "visible_points_norm": points_norm[valid > 0.5],
        }
        if scale is not None:
            points_px = points_norm * scale
            item["points"] = points_px.astype(np.float32)
            item["visible_points"] = points_px[valid > 0.5].astype(np.float32)
        lanes.append(item)
    violations = check_ordered_slot_lane_order(lanes, order_margin_px=order_margin_px, img_w=img_w)
    if violations and order_check != "none":
        msg = f"ordered_slot order violation: image={b}, violations={violations}"
        if order_check == "error":
            raise AssertionError(msg)
        LOGGER.warning(msg)
    if output_order == "left_to_right":
        lanes = sorted(lanes, key=lambda lane: (float(lane["bottom_x_norm"]), int(lane["slot"])))
    assert len(lanes) == num_lanes, (
        f"ordered-slot decode contract violated: expected {num_lanes} lanes, got {len(lanes)}."
    )
    if return_diagnostics:
        return lanes, {
            "output_order": output_order,
            "order_check": order_check,
            "order_violations": violations,
            "order_violation_count": len(violations),
            "has_order_violation": bool(violations),
        }
    return lanes
