"""Ordered lane-slot target construction for GCS-YOLO-Lane."""

from __future__ import annotations

from typing import Sequence

import torch

from ultralytics.utils.gcs_fixed_y import expected_training_fixed_y_desc, validate_training_fixed_y_desc


def lane_count_to_label(num_lanes: int, min_lanes: int = 2, max_lanes: int = 5) -> int:
    """Map a supported lane count to an ordered-slot count class."""
    num_lanes = int(num_lanes)
    min_lanes = int(min_lanes)
    max_lanes = int(max_lanes)
    if num_lanes < min_lanes or num_lanes > max_lanes:
        raise ValueError(
            f"Unsupported lane count {num_lanes}; ordered_slot_v2 supports {min_lanes}..{max_lanes} lanes."
        )
    return num_lanes - min_lanes


def _is_contiguous_valid(valid: torch.Tensor) -> bool:
    """Return True when valid points form one contiguous interval."""
    idx = torch.where(valid > 0)[0]
    if idx.numel() == 0:
        return True
    start = int(idx.min().item())
    end = int(idx.max().item())
    return int((valid[start : end + 1] > 0).sum().item()) == end - start + 1


def repair_lane_to_contiguous_interval(points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fill holes inside one fixed-y valid interval by linear x interpolation."""
    idx = torch.where(valid > 0)[0]
    if idx.numel() == 0:
        return points, valid, 0

    start = int(idx.min().item())
    end = int(idx.max().item())
    repaired_valid = valid.clone()
    repaired_points = points.clone()
    hole_points = 0

    for k in range(start, end + 1):
        if bool(valid[k] > 0):
            continue
        left = idx[idx < k]
        right = idx[idx > k]
        if left.numel() > 0 and right.numel() > 0:
            l = int(left.max().item())
            r = int(right.min().item())
            t = float(k - l) / float(max(r - l, 1))
            repaired_points[k, 0] = points[l, 0] * (1.0 - t) + points[r, 0] * t
            repaired_points[k, 1] = points[k, 1]
        elif left.numel() > 0:
            repaired_points[k] = points[int(left.max().item())]
        elif right.numel() > 0:
            repaired_points[k] = points[int(right.min().item())]
        repaired_valid[k] = 1
        hole_points += 1
    return repaired_points, repaired_valid, hole_points


def _normalize_contiguity_policy(policy: str) -> str:
    policy = str(policy or "strict").strip().lower()
    if policy not in {"strict", "repair_interp", "raw"}:
        raise ValueError(f"Unsupported ordered_slot contiguity_policy={policy!r}.")
    return policy


def _apply_contiguity_policy(
    gt_points: torch.Tensor,
    gt_valid: torch.Tensor,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate or repair non-contiguous masks before interval target construction."""
    policy = _normalize_contiguity_policy(policy)
    points = gt_points.clone()
    valid = gt_valid.clone()
    repaired_lanes = gt_valid.new_zeros((), dtype=torch.long)
    repaired_holes = gt_valid.new_zeros((), dtype=torch.long)

    for i in range(valid.shape[0]):
        if _is_contiguous_valid(valid[i]):
            continue
        if policy == "strict":
            raise ValueError(f"ordered_slot target lane {i} has a non-contiguous valid mask under strict policy.")
        if policy == "raw":
            raise ValueError("ordered_slot contiguity_policy='raw' is incompatible with interval targets.")
        points_i, valid_i, holes = repair_lane_to_contiguous_interval(points[i], valid[i])
        points[i] = points_i
        valid[i] = valid_i
        repaired_lanes = repaired_lanes + 1
        repaired_holes = repaired_holes + int(holes)
    return points, valid, repaired_lanes, repaired_holes

def _remove_padded_lanes(
    gt_points: torch.Tensor,
    gt_valid: torch.Tensor,
    min_valid_points: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop padded or invalid lanes and return kept original indices."""
    lane_valid_count = gt_valid.float().sum(dim=1)
    keep = lane_valid_count >= int(min_valid_points)
    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
    return gt_points[keep], gt_valid[keep], kept_indices


def _lane_sort_key_bottom_x(lane_points: torch.Tensor, lane_valid: torch.Tensor) -> torch.Tensor | None:
    """Return the x coordinate at the bottom-most visible point of one lane."""
    valid_idx = torch.where(lane_valid > 0)[0]
    if valid_idx.numel() == 0:
        return None
    visible_y = lane_points[valid_idx, 1]
    bottom_pos = torch.argmax(visible_y)
    bottom_idx = valid_idx[bottom_pos]
    return lane_points[bottom_idx, 0]


def build_ordered_lane_slots_single(
    gt_points: torch.Tensor,
    gt_valid: torch.Tensor,
    max_lanes: int = 5,
    min_lanes: int = 2,
    min_valid_points: int = 2,
    contiguity_policy: str = "strict",
) -> dict[str, torch.Tensor]:
    """Build compact left-to-right ordered slot targets for one image."""
    if gt_points.ndim != 3 or gt_points.shape[-1] != 2:
        raise ValueError(f"gt_points must have shape N x K x 2, got {tuple(gt_points.shape)}.")
    if gt_valid.ndim != 2:
        raise ValueError(f"gt_valid must have shape N x K, got {tuple(gt_valid.shape)}.")
    if gt_points.shape[:2] != gt_valid.shape:
        raise ValueError(f"gt_points {tuple(gt_points.shape)} and gt_valid {tuple(gt_valid.shape)} mismatch.")
    if max_lanes <= 0 or min_lanes <= 0 or min_lanes > max_lanes:
        raise ValueError(f"ordered_slot lane bounds must satisfy 0 < min_lanes <= max_lanes, got {min_lanes}/{max_lanes}.")

    device = gt_points.device
    dtype = torch.float32
    gt_points = gt_points.to(device=device, dtype=torch.float32)
    gt_valid = gt_valid.to(device=device, dtype=torch.float32)
    gt_points, gt_valid, kept_indices = _remove_padded_lanes(gt_points, gt_valid, min_valid_points=min_valid_points)
    if gt_points.shape[0] == 0:
        raise ValueError("ordered_slot target has no valid lanes after removing padding lanes.")
    if gt_points.shape[1] > 0:
        for i in range(gt_points.shape[0]):
            validate_training_fixed_y_desc(
                gt_points[i, :, 1].detach().float().cpu().numpy(),
                name=f"ordered_slot target fixed_y lane {i}",
            )
        fixed_y = torch.as_tensor(
            expected_training_fixed_y_desc(k=gt_points.shape[1]) / 720.0,
            device=device,
            dtype=torch.float32,
        )
        gt_points[:, :, 1] = fixed_y.view(1, -1)
    gt_points, gt_valid, repaired_lanes, repaired_holes = _apply_contiguity_policy(
        gt_points,
        gt_valid,
        contiguity_policy,
    )
    _, k, _ = gt_points.shape

    slot_points = torch.zeros(max_lanes, k, 2, device=device, dtype=dtype)
    slot_valid = torch.zeros(max_lanes, k, device=device, dtype=gt_valid.dtype)
    slot_interval_valid = torch.zeros(max_lanes, k, device=device, dtype=gt_valid.dtype)
    slot_point_valid = torch.zeros(max_lanes, k, device=device, dtype=gt_valid.dtype)
    slot_exist = torch.zeros(max_lanes, device=device, dtype=dtype)
    start_labels = torch.zeros(max_lanes, device=device, dtype=torch.long)
    end_labels = torch.zeros(max_lanes, device=device, dtype=torch.long)
    sorted_gt_indices = torch.full((max_lanes,), -1, device=device, dtype=torch.long)

    lane_infos: list[tuple[float, int, torch.Tensor]] = []
    for i in range(gt_points.shape[0]):
        valid_idx = torch.where(gt_valid[i] > 0)[0]
        if valid_idx.numel() < int(min_valid_points):
            continue
        x_ref = _lane_sort_key_bottom_x(gt_points[i], gt_valid[i])
        if x_ref is None:
            continue
        lane_infos.append((float(x_ref.detach().cpu().item()), i, kept_indices[i]))

    lane_infos.sort(key=lambda item: item[0])
    num_lanes = len(lane_infos)
    count_label_value = lane_count_to_label(num_lanes, min_lanes=min_lanes, max_lanes=max_lanes)
    lane_infos = lane_infos[:max_lanes]

    for slot_idx, (_, filtered_idx, original_idx) in enumerate(lane_infos):
        slot_points[slot_idx] = gt_points[filtered_idx]
        slot_valid[slot_idx] = gt_valid[filtered_idx]
        slot_interval_valid[slot_idx] = gt_valid[filtered_idx]
        slot_point_valid[slot_idx] = gt_valid[filtered_idx]
        slot_exist[slot_idx] = 1.0
        sorted_gt_indices[slot_idx] = original_idx.to(dtype=torch.long)

        valid_idx = torch.where(gt_valid[filtered_idx] > 0)[0]
        start_labels[slot_idx] = valid_idx.min()
        end_labels[slot_idx] = valid_idx.max()

    count_label = torch.tensor(count_label_value, device=device, dtype=torch.long)

    return {
        "slot_points": slot_points,
        "slot_valid": slot_valid,
        "slot_interval_valid": slot_interval_valid,
        "slot_point_valid": slot_point_valid,
        "slot_exist": slot_exist,
        "count_label": count_label,
        "start_labels": start_labels,
        "end_labels": end_labels,
        "num_lanes": torch.tensor(num_lanes, device=device, dtype=torch.long),
        "min_lanes": torch.tensor(min_lanes, device=device, dtype=torch.long),
        "max_lanes": torch.tensor(max_lanes, device=device, dtype=torch.long),
        "sorted_gt_indices": sorted_gt_indices,
        "repaired_noncontiguous_lanes": repaired_lanes.to(device=device, dtype=torch.long),
        "repaired_hole_points": repaired_holes.to(device=device, dtype=torch.long),
    }


def build_ordered_lane_slots_batch(
    gt_points: torch.Tensor | Sequence[torch.Tensor],
    gt_valid: torch.Tensor | Sequence[torch.Tensor],
    max_lanes: int = 5,
    min_lanes: int = 2,
    min_valid_points: int = 2,
    contiguity_policy: str = "strict",
) -> dict[str, torch.Tensor]:
    """Build ordered slot targets for a tensor batch or a list of per-image tensors."""
    results = []
    if isinstance(gt_points, (list, tuple)):
        if not isinstance(gt_valid, (list, tuple)):
            raise TypeError("gt_valid must be a list/tuple when gt_points is a list/tuple.")
        if len(gt_points) != len(gt_valid):
            raise ValueError(f"gt_points and gt_valid list lengths differ: {len(gt_points)} vs {len(gt_valid)}.")
        for points_i, valid_i in zip(gt_points, gt_valid):
            results.append(
                build_ordered_lane_slots_single(
                    points_i,
                    valid_i,
                    max_lanes=max_lanes,
                    min_lanes=min_lanes,
                    min_valid_points=min_valid_points,
                    contiguity_policy=contiguity_policy,
                )
            )
    else:
        if not isinstance(gt_valid, torch.Tensor):
            raise TypeError("gt_valid must be a tensor when gt_points is a tensor.")
        if gt_points.ndim != 4:
            raise ValueError(f"gt_points must have shape B x N x K x 2, got {tuple(gt_points.shape)}.")
        if gt_valid.ndim != 3:
            raise ValueError(f"gt_valid must have shape B x N x K, got {tuple(gt_valid.shape)}.")
        if gt_points.shape[:3] != gt_valid.shape:
            raise ValueError(f"gt_points {tuple(gt_points.shape)} and gt_valid {tuple(gt_valid.shape)} mismatch.")
        for b in range(gt_points.shape[0]):
            results.append(
                build_ordered_lane_slots_single(
                    gt_points[b],
                    gt_valid[b],
                    max_lanes=max_lanes,
                    min_lanes=min_lanes,
                    min_valid_points=min_valid_points,
                    contiguity_policy=contiguity_policy,
                )
            )

    if not results:
        raise ValueError("Cannot build ordered slot targets for an empty batch.")
    return {key: torch.stack([result[key] for result in results], dim=0) for key in results[0]}
