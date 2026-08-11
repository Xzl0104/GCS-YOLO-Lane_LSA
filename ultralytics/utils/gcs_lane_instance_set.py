"""Prediction-only lane-instance-set helpers for default-off GCS experiments."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

__all__ = (
    "LANE_INSTANCE_PREDICTION_KEYS",
    "decode_lane_instance_set_predictions",
    "interval_valid_logits_from_start_end",
    "resolve_lane_instance_decode_mode",
    "select_lane_instance_survivors",
)

LANE_INSTANCE_PREDICTION_KEYS = (
    "pred_lane_instance_points",
    "pred_lane_instance_start_logits",
    "pred_lane_instance_end_logits",
    "pred_lane_instance_valid_logits",
    "pred_lane_instance_interval_start",
    "pred_lane_instance_interval_end",
    "pred_lane_instance_identity",
    "pred_lane_instance_pair_duplicate_logits",
    "pred_lane_instance_novelty_logits",
    "pred_lane_instance_left_logits",
    "pred_lane_instance_right_logits",
    "pred_lane_instance_geometry_quality_logits",
    "pred_lane_instance_survival_logits",
    "pred_lane_instance_empty_logit",
)


def resolve_lane_instance_decode_mode(requested_mode, model) -> str:
    """Resolve query/ordered-slot/lane-instance-set without changing the model's gcs_mode contract."""
    requested = str(requested_mode or "auto").strip().lower().replace("-", "_")
    supports_lane_instance = any(
        bool(getattr(module, "lane_instance_set_decoder_head", False)) for module in model.modules()
    )
    if requested == "auto" and supports_lane_instance:
        return "lane_instance_set"
    if requested == "lane_instance_set":
        if not supports_lane_instance:
            raise RuntimeError("decode-mode lane_instance_set requires an enabled lane-instance-set model head.")
        return requested
    from ultralytics.models.gcs.mode_utils import resolve_decode_mode

    return resolve_decode_mode(requested, model)


def interval_valid_logits_from_start_end(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    sharpness: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive contiguous bottom-to-top valid logits from unordered start/end logits.

    Point index `0` is the bottom TuSimple anchor (`y=710`) and index `K-1` is
    the top anchor (`y=160`). The returned interval always uses
    `ordered_start <= ordered_end`; visibility at index `k` is positive when
    `ordered_start <= k <= ordered_end` and negative outside that interval.
    """
    if start_logits.shape != end_logits.shape:
        raise ValueError(
            "start_logits and end_logits must have identical shape, "
            f"got {tuple(start_logits.shape)} and {tuple(end_logits.shape)}."
        )
    if start_logits.ndim < 1:
        raise ValueError("start_logits must have at least one dimension.")
    if not float(sharpness) > 0.0:
        raise ValueError(f"sharpness must be positive, got {sharpness}.")

    k = int(start_logits.shape[-1])
    if k <= 0:
        raise ValueError("interval logits must have a non-empty point dimension.")
    work_dtype = torch.float32 if start_logits.dtype in {torch.float16, torch.bfloat16} else start_logits.dtype
    start_work = start_logits.to(dtype=work_dtype)
    end_work = end_logits.to(dtype=work_dtype)
    position = torch.arange(k, device=start_logits.device, dtype=work_dtype)

    start_probability = torch.softmax(start_work, dim=-1)
    end_probability = torch.softmax(end_work, dim=-1)
    start_index = (start_probability * position).sum(dim=-1)
    end_index = (end_probability * position).sum(dim=-1)
    ordered_start = torch.minimum(start_index, end_index)
    ordered_end = torch.maximum(start_index, end_index)

    field = torch.minimum(
        position.view(*([1] * (start_logits.ndim - 1)), k) - ordered_start.unsqueeze(-1) + 0.5,
        ordered_end.unsqueeze(-1) - position.view(*([1] * (start_logits.ndim - 1)), k) + 0.5,
    )
    valid_logits = field * float(sharpness)
    return (
        valid_logits.to(dtype=start_logits.dtype),
        ordered_start.to(dtype=start_logits.dtype),
        ordered_end.to(dtype=start_logits.dtype),
    )


def _lane_tensor(preds: dict[str, torch.Tensor], key: str, batch_index: int) -> torch.Tensor | None:
    value = preds.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim == 0:
        return value
    if key == "pred_lane_instance_empty_logit":
        return value.reshape(-1)[int(batch_index)]
    point_keys = {"pred_points", "pred_lane_instance_points"}
    qk_keys = {"pred_valid_logits", "pred_lane_instance_valid_logits"}
    vector_keys = {
        "pred_logits",
        "pred_lane_instance_survival_logits",
        "pred_lane_instance_geometry_quality_logits",
        "pred_lane_instance_novelty_logits",
    }
    pair_keys = {
        "pred_lane_instance_pair_duplicate_logits",
        "pred_lane_instance_left_logits",
        "pred_lane_instance_right_logits",
    }
    if key in point_keys and value.ndim == 3:
        return value
    if key in qk_keys and value.ndim == 2:
        return value
    if key in vector_keys and value.ndim == 1:
        return value
    if key in pair_keys and value.ndim == 2:
        return value
    return value[int(batch_index)]


def _required_lane_instance_keys_present(preds: dict[str, torch.Tensor]) -> bool:
    required = {
        "pred_lane_instance_points",
        "pred_lane_instance_valid_logits",
        "pred_lane_instance_survival_logits",
        "pred_lane_instance_pair_duplicate_logits",
    }
    return required.issubset(preds)


def _select_global_survivor_subset(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    duplicate_probability: torch.Tensor,
    duplicate_thr: float,
    max_survivors: int,
) -> tuple[list[int], float]:
    """Return the exact maximum-utility non-duplicate subset for the eligible candidates."""
    candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist()
    best_subset: tuple[int, ...] = ()
    best_utility = 0.0
    max_size = min(int(max_survivors), len(candidates))
    for subset_size in range(1, max_size + 1):
        for subset in combinations(candidates, subset_size):
            if any(
                float(duplicate_probability[left, right].item()) >= float(duplicate_thr)
                for left, right in combinations(subset, 2)
            ):
                continue
            utility = sum(float(scores[index].item()) for index in subset)
            if utility > best_utility + 1e-12 or (
                abs(utility - best_utility) <= 1e-12 and len(subset) > len(best_subset)
            ):
                best_subset = subset
                best_utility = utility
    ordered_subset = sorted(best_subset, key=lambda index: (-float(scores[index].item()), index))
    return ordered_subset, best_utility


def select_lane_instance_survivors(
    preds: dict[str, torch.Tensor],
    batch_index: int = 0,
    score_thr: float = 0.25,
    point_valid_thr: float = 0.5,
    min_points: int = 2,
    duplicate_thr: float = 0.65,
    max_det: int = 5,
    min_survivors: int = 2,
    allow_empty: bool = False,
    empty_thr: float = 0.75,
) -> dict[str, torch.Tensor | int | bool | float | str]:
    """Select valid lane-instance survivors without labels, count heads, or low-score backfill."""
    if not _required_lane_instance_keys_present(preds):
        return {
            "survivor_indices": torch.empty(0, dtype=torch.long),
            "survivor_scores": torch.empty(0, dtype=torch.float32),
            "decoded_count": 0,
            "no_op": True,
            "under_min": True,
            "valid_candidate_count": 0,
            "empty_selected": False,
        }
    if max_det <= 0:
        raise ValueError(f"max_det must be positive, got {max_det}.")
    if min_survivors < 0:
        raise ValueError(f"min_survivors must be non-negative, got {min_survivors}.")
    max_survivors = min(int(max_det), 5)
    min_survivors = min(int(min_survivors), max_survivors)
    if min_points <= 0:
        raise ValueError(f"min_points must be positive, got {min_points}.")

    survival_logits = _lane_tensor(preds, "pred_lane_instance_survival_logits", batch_index)
    duplicate_logits = _lane_tensor(preds, "pred_lane_instance_pair_duplicate_logits", batch_index)
    empty_logit = _lane_tensor(preds, "pred_lane_instance_empty_logit", batch_index)
    valid_logits = _lane_tensor(preds, "pred_lane_instance_valid_logits", batch_index)

    if survival_logits is None:
        raise ValueError("lane-instance survival utility logits are required.")
    survival_logits = survival_logits.reshape(-1)
    q = int(survival_logits.numel())
    if duplicate_logits is None or tuple(duplicate_logits.shape) != (q, q):
        raise ValueError(f"duplicate logits must have shape Q x Q, got {None if duplicate_logits is None else tuple(duplicate_logits.shape)}.")
    if valid_logits is None or valid_logits.ndim != 2 or int(valid_logits.shape[0]) != q:
        raise ValueError("lane-instance valid logits must have shape Q x K.")

    valid_masks = torch.stack(
        [_contiguous_mask(row.sigmoid() >= float(point_valid_thr), min_points=min_points) for row in valid_logits],
        dim=0,
    ).to(device=survival_logits.device)
    valid_candidate_mask = valid_masks.sum(dim=-1) >= int(min_points)
    valid_candidate_count = int(valid_candidate_mask.sum().item())

    scores = survival_logits.float().sigmoid()

    if empty_logit is not None and bool(allow_empty):
        empty_probability = float(empty_logit.float().sigmoid().item())
        valid_scores = scores[valid_candidate_mask]
        if empty_probability >= float(empty_thr) and (
            valid_scores.numel() == 0 or float(valid_scores.max().item()) < float(score_thr)
        ):
            return {
                "survivor_indices": torch.empty(0, dtype=torch.long, device=scores.device),
                "survivor_scores": torch.empty(0, dtype=scores.dtype, device=scores.device),
                "decoded_count": 0,
                "no_op": False,
                "under_min": True,
                "valid_candidate_count": valid_candidate_count,
                "empty_selected": True,
            }

    duplicate_prob = duplicate_logits.float().sigmoid()
    eligible_mask = valid_candidate_mask & (scores >= float(score_thr))
    keep, set_utility = _select_global_survivor_subset(
        scores,
        eligible_mask,
        duplicate_prob,
        duplicate_thr=float(duplicate_thr),
        max_survivors=max_survivors,
    )

    survivor_indices = torch.tensor(keep, dtype=torch.long, device=scores.device)
    survivor_scores = scores[survivor_indices] if survivor_indices.numel() else torch.empty(0, dtype=scores.dtype, device=scores.device)
    return {
        "survivor_indices": survivor_indices,
        "survivor_scores": survivor_scores,
        "decoded_count": int(survivor_indices.numel()),
        "no_op": False,
        "under_min": int(survivor_indices.numel()) < min_survivors,
        "valid_candidate_count": valid_candidate_count,
        "eligible_candidate_count": int(eligible_mask.sum().item()),
        "empty_selected": False,
        "selection_strategy": "exact_global_survival_duplicate",
        "set_utility": float(set_utility),
    }


def _contiguous_mask(mask: torch.Tensor, min_points: int) -> torch.Tensor:
    mask = mask.detach().bool().cpu()
    best_start = 0
    best_len = 0
    start = None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            length = i - start
            if length > best_len:
                best_start = start
                best_len = length
            start = None
    output = torch.zeros_like(mask, dtype=torch.bool)
    if best_len >= int(min_points):
        output[best_start : best_start + best_len] = True
    return output


def decode_lane_instance_set_predictions(
    preds: dict[str, torch.Tensor],
    batch_index: int = 0,
    image_shape: tuple[int, int] | None = None,
    score_thr: float = 0.25,
    point_valid_thr: float = 0.5,
    min_points: int = 2,
    max_det: int = 5,
    duplicate_thr: float = 0.65,
    min_survivors: int = 2,
    allow_empty: bool = False,
    empty_thr: float = 0.75,
    return_diagnostics: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Decode lane-instance-set survivors into lane dictionaries using predictions only."""
    selection = select_lane_instance_survivors(
        preds,
        batch_index=batch_index,
        score_thr=score_thr,
        point_valid_thr=point_valid_thr,
        min_points=min_points,
        duplicate_thr=duplicate_thr,
        max_det=max_det,
        min_survivors=min_survivors,
        allow_empty=allow_empty,
        empty_thr=empty_thr,
    )
    indices = selection["survivor_indices"]
    if not isinstance(indices, torch.Tensor) or indices.numel() == 0:
        diagnostics = {
            **selection,
            "decoded_count": 0,
            "max_det_safety_cap": min(int(max_det), 5),
        }
        return ([], diagnostics) if return_diagnostics else []

    points = _lane_tensor(preds, "pred_lane_instance_points", batch_index)
    valid_logits = _lane_tensor(preds, "pred_lane_instance_valid_logits", batch_index)
    if points is None or valid_logits is None:
        raise ValueError("lane-instance points and valid logits are required for decode.")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"lane-instance points must have shape Q x K x 2, got {tuple(points.shape)}.")
    if valid_logits.shape != points.shape[:2]:
        raise ValueError(f"valid logits must have shape Q x K, got {tuple(valid_logits.shape)}.")

    points_cpu = points.detach().float().cpu().clamp(0.0, 1.0)
    valid_scores_cpu = valid_logits.detach().float().cpu().sigmoid()
    scores_cpu = selection["survivor_scores"].detach().float().cpu()
    scale = None
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"image_shape must be positive H,W, got {image_shape}.")
        scale = torch.tensor([w, h], dtype=points_cpu.dtype).view(1, 2)

    lanes: list[dict] = []
    for lane_rank, index in enumerate(indices.detach().cpu().tolist()):
        lane_points = points_cpu[int(index)]
        order = torch.argsort(lane_points[:, 1], descending=True, stable=True)
        lane_points = lane_points[order]
        lane_valid_scores = valid_scores_cpu[int(index)][order]
        visible_mask = _contiguous_mask(lane_valid_scores >= float(point_valid_thr), min_points=min_points)
        if int(visible_mask.sum().item()) < int(min_points):
            continue
        lane_np = lane_points.numpy().astype(np.float32)
        item = {
            "score": float(scores_cpu[lane_rank].item()),
            "query": int(index),
            "points_norm": lane_np,
            "point_valid_scores": lane_valid_scores.numpy().astype(np.float32),
            "point_valid": visible_mask.numpy().astype(np.float32),
            "visible_points_norm": lane_np[visible_mask.numpy().astype(bool)],
            "decoded_count": -1,
            "count_source": "survivors",
            "max_det_safety_cap": min(int(max_det), 5),
        }
        if scale is not None:
            points_px = (lane_points * scale).numpy().astype(np.float32)
            item["points"] = points_px
            item["visible_points"] = points_px[visible_mask.numpy().astype(bool)]
        lanes.append(item)

    decoded_count = len(lanes)
    for lane in lanes:
        lane["decoded_count"] = int(decoded_count)
    diagnostics = {
        **selection,
        "decoded_count": int(decoded_count),
        "under_min": int(decoded_count) < int(min_survivors),
        "max_det_safety_cap": min(int(max_det), 5),
    }
    return (lanes, diagnostics) if return_diagnostics else lanes
