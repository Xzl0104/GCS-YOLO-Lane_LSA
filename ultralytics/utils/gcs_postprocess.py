# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Post-processing helpers for GCS-YOLO-Lane structured lane predictions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch

__all__ = (
    "GCS_DEFAULT_MAX_DET",
    "decode_gcs_predictions",
    "draw_gcs_lanes",
    "apply_count_policy",
    "build_count_calibration_config",
    "count_head_decode_meta",
    "count_aware_refill",
    "empty_decode_count_state",
    "lane_x_distance_px",
    "lane_mean_distance_px",
    "lane_nms",
    "predict_count_from_logits",
    "save_gcs_lanes_txt",
    "soft_count_decision",
    "summarize_decode_count_state",
    "sort_lane_bottom_to_top",
    "update_decode_count_state",
)


GCS_DEFAULT_MAX_DET = 5


GCS_LANE_COLORS = (
    (0, 255, 0),
    (0, 200, 255),
    (255, 120, 0),
    (255, 0, 180),
    (80, 180, 255),
    (180, 255, 80),
    (255, 80, 80),
    (180, 120, 255),
)


def sort_lane_bottom_to_top(points: torch.Tensor) -> torch.Tensor:
    """Sort one K x 2 lane by descending y, matching the GCS bottom-to-top point order."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"Expected one lane with shape K x 2, got {tuple(points.shape)}.")
    order = torch.argsort(points[:, 1], descending=True, stable=True)
    return points[order]


def longest_contiguous_valid_mask(mask: torch.Tensor, min_points: int = 2) -> torch.Tensor:
    """Keep only the longest continuous visible point run in bottom-to-top point order."""
    mask = mask.detach().bool().cpu()
    if mask.ndim != 1:
        raise ValueError(f"point visibility mask must be 1D, got {tuple(mask.shape)}.")
    best_start = best_len = 0
    start = None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            length = i - start
            if length > best_len:
                best_start, best_len = start, length
            start = None
    out = torch.zeros_like(mask, dtype=torch.bool)
    if best_len >= int(min_points):
        out[best_start : best_start + best_len] = True
    return out


def lane_x_distance_px(
    a: torch.Tensor,
    b: torch.Tensor,
    image_shape: tuple[int, int],
    min_overlap: int = 2,
    valid_a: torch.Tensor | None = None,
    valid_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mean absolute x-distance in pixels between two normalized K x 2 lanes.

    GCS lanes use a fixed point order, so Lane-NMS compares points at the same
    sample index and suppresses duplicates by lateral distance. For fixed-y
    lanes, optional visibility masks restrict the comparison to the common
    visible anchor run so invalid endpoints do not decide duplicate suppression.
    """
    if a.shape != b.shape or a.ndim != 2 or a.shape[-1] != 2:
        raise ValueError(f"Lane NMS expects matching K x 2 lanes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if valid_a is not None or valid_b is not None:
        if valid_a is None or valid_b is None:
            raise ValueError("Lane NMS visibility comparison requires both valid_a and valid_b.")
        valid_a = valid_a.detach().bool().to(device=a.device)
        valid_b = valid_b.detach().bool().to(device=a.device)
        if valid_a.shape != a.shape[:1] or valid_b.shape != b.shape[:1]:
            raise ValueError(f"Lane NMS valid masks must have shape K, got {tuple(valid_a.shape)} and {tuple(valid_b.shape)}.")
        keep = valid_a & valid_b
        if int(keep.sum()) < int(min_overlap):
            return a.new_tensor(float("inf"))
        a = a[keep]
        b = b[keep]
    elif a.shape[0] < int(min_overlap):
        return a.new_tensor(float("inf"))
    w = int(image_shape[1])
    return torch.mean(torch.abs(a[:, 0] - b[:, 0]) * float(w))


def lane_mean_distance_px(
    a: torch.Tensor,
    b: torch.Tensor,
    image_shape: tuple[int, int],
    min_overlap: int = 6,
    valid_a: torch.Tensor | None = None,
    valid_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward-compatible alias for the Lane-NMS lateral distance."""
    return lane_x_distance_px(
        a,
        b,
        image_shape=image_shape,
        min_overlap=min_overlap,
        valid_a=valid_a,
        valid_b=valid_b,
    )


def lane_nms(
    points: torch.Tensor,
    scores: torch.Tensor,
    image_shape: tuple[int, int],
    dist_thr_px: float,
    valid_masks: torch.Tensor | None = None,
    point_valid_scores: torch.Tensor | None = None,
    exist_scores: torch.Tensor | None = None,
    min_overlap: int = 6,
    return_suppressed: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[dict]]:
    """Greedily suppress duplicate lane predictions by mean x-distance."""
    if dist_thr_px <= 0.0 or points.shape[0] <= 1:
        keep_all = torch.arange(points.shape[0], dtype=torch.long, device=points.device)
        return (keep_all, []) if return_suppressed else keep_all
    if valid_masks is not None and valid_masks.shape != points.shape[:2]:
        raise ValueError(f"Lane-NMS valid_masks must have shape N x K, got {tuple(valid_masks.shape)} vs {tuple(points.shape[:2])}.")
    if point_valid_scores is not None and point_valid_scores.shape != points.shape[:2]:
        raise ValueError(
            "Lane-NMS point_valid_scores must have shape N x K, "
            f"got {tuple(point_valid_scores.shape)} vs {tuple(points.shape[:2])}."
        )
    if exist_scores is not None and exist_scores.shape[0] != points.shape[0]:
        raise ValueError(
            f"Lane-NMS exist_scores must have length N={points.shape[0]}, got {tuple(exist_scores.shape)}."
        )
    order = torch.argsort(scores, descending=True)
    keep: list[int] = []
    suppressed: list[dict] = []
    for idx in order.tolist():
        lane = points[idx]
        duplicate = False
        suppressor_idx = None
        suppressor_dist = None
        for kept_idx in keep:
            valid_a = valid_masks[idx] if valid_masks is not None else None
            valid_b = valid_masks[kept_idx] if valid_masks is not None else None
            distance = lane_x_distance_px(
                lane,
                points[kept_idx],
                image_shape,
                min_overlap=int(min_overlap),
                valid_a=valid_a,
                valid_b=valid_b,
            )
            if torch.isfinite(distance) and float(distance) <= float(dist_thr_px):
                duplicate = True
                suppressor_idx = int(kept_idx)
                suppressor_dist = float(distance)
                break
        if not duplicate:
            keep.append(int(idx))
        else:
            valid_count = None
            point_valid_mean = None
            if valid_masks is not None:
                mask = valid_masks[idx].detach().bool()
                valid_count = int(mask.sum().item())
                if point_valid_scores is not None and valid_count > 0:
                    point_valid_mean = float(point_valid_scores[idx].detach()[mask].float().mean().cpu())
                else:
                    point_valid_mean = float(mask.float().mean().cpu())
            elif point_valid_scores is not None:
                point_valid_mean = float(point_valid_scores[idx].detach().float().mean().cpu())
            exist_score = float(exist_scores[idx].detach().cpu()) if exist_scores is not None else float(scores[idx].detach().cpu())
            rank_score = float(scores[idx].detach().cpu())
            suppressed.append(
                {
                    "index": int(idx),
                    "candidate_id": int(idx),
                    "query_id": int(idx),
                    "score": rank_score,
                    "rank_score": rank_score,
                    "valid_points": valid_count,
                    "point_valid_mean": point_valid_mean,
                    "mean_point_valid": point_valid_mean,
                    "exist_score": exist_score,
                    "lane_conf": exist_score,
                    "quality": None,
                    "suppressed_by_index": suppressor_idx,
                    "suppressed_by": suppressor_idx,
                    "suppress_reason": "distance",
                    "distance_to_suppressor": suppressor_dist,
                }
            )
    keep_tensor = torch.tensor(keep, dtype=torch.long, device=points.device)
    return (keep_tensor, suppressed) if return_suppressed else keep_tensor


def build_count_calibration_config(
    mode: str | None = "none",
    default_count: int = 4,
    min_count: int = 3,
    max_count: int = 5,
    s4_low: float = 0.35,
    s5_high: float = 0.50,
    s5_rescue_low: float = 0.20,
    gap34_thr: float = 0.25,
    gap45_thr: float = 0.25,
    gap45_rescue_thr: float = 0.60,
    min_valid_points: int = 8,
    min_length: int = 8,
    min_mean_valid_score: float = 0.45,
) -> dict | None:
    """Return None for disabled count calibration; score-gap count calibration has been removed."""
    mode = "none" if mode is None else str(mode).lower()
    if mode in {"", "none", "off", "false", "0"}:
        return None
    raise ValueError("Score-gap count calibration has been removed. Use Count Head Top-K decode instead.")


def _normalize_count_calibration_config(config: str | Mapping | None) -> dict | None:
    """Normalize string/dict count calibration settings."""
    if config is None or config is False:
        return None
    if isinstance(config, str):
        return build_count_calibration_config(mode=config)
    if not isinstance(config, Mapping):
        raise TypeError(f"count_calibration must be a mapping, string, or None, got {type(config).__name__}.")
    if "mode" not in config:
        if len(config) == 0:
            return None
        raise ValueError("Score-gap count calibration has been removed. Use Count Head Top-K decode instead.")
    mode = str(config.get("mode", "none")).lower()
    if mode in {"", "none", "off", "false", "0"}:
        return None
    raise ValueError("Score-gap count calibration has been removed. Use Count Head Top-K decode instead.")


def _lane_quality(lane: dict) -> tuple[int, int, float]:
    """Return visible-count, fixed-y length, and mean visible-point score for one decoded lane."""
    points = np.asarray(lane.get("points_norm", []), dtype=np.float32)
    valid = None
    if "point_valid" in lane:
        valid = np.asarray(lane["point_valid"], dtype=np.float32) > 0.5
    elif "visible_points_norm" in lane:
        valid = np.ones(np.asarray(lane["visible_points_norm"]).shape[0], dtype=bool)
    if valid is None or valid.shape[0] != points.shape[0]:
        valid = np.ones(points.shape[0], dtype=bool)
    valid_count = int(valid.sum())
    length = valid_count

    valid_scores = lane.get("point_valid_scores")
    if valid_scores is None:
        mean_valid_score = 1.0 if valid_count > 0 else 0.0
    else:
        scores = np.asarray(valid_scores, dtype=np.float32)
        if scores.shape[0] == valid.shape[0] and valid_count > 0:
            mean_valid_score = float(scores[valid].mean())
        else:
            mean_valid_score = 0.0
    return valid_count, length, mean_valid_score


def _lane_exist_score(lane: dict) -> float:
    """Return the raw lane existence probability, preserving old ``score`` compatibility."""
    return float(lane.get("exist_score", lane.get("score", 0.0)))


def _lane_rank_score(lane: dict) -> float:
    """Return the decoded lane rank score, falling back to raw existence for old records."""
    return float(lane.get("rank_score", _lane_exist_score(lane)))


def _normalize_rank_min_points(rank_min_points: Mapping[int, int] | None) -> dict[int, int] | None:
    """Normalize optional per-selected-rank min_points overrides."""
    if not rank_min_points:
        return None
    out: dict[int, int] = {}
    for rank, min_points in dict(rank_min_points).items():
        rank_i = int(rank)
        min_points_i = int(min_points)
        if rank_i <= 0:
            raise ValueError(f"rank_min_points ranks must be >= 1, got {rank_i}.")
        if min_points_i <= 0:
            raise ValueError(f"rank_min_points values must be >= 1, got {min_points_i}.")
        out[rank_i] = min_points_i
    return out or None


def predict_count_from_logits(pred_count_logits: torch.Tensor, temperature: float = 1.0) -> tuple[int, np.ndarray, float]:
    """Predict K in {2,3,4,5} plus probability vector and top-1 margin from one count-head logit vector."""
    if pred_count_logits.ndim != 1 or pred_count_logits.numel() != 4:
        raise ValueError(f"pred_count_logits must have shape 4 for one image, got {tuple(pred_count_logits.shape)}.")
    logits = pred_count_logits.detach().float().cpu() / max(float(temperature), 1e-6)
    prob = torch.softmax(logits, dim=-1)
    idx = int(prob.argmax().item())
    top2 = prob.topk(k=2).values
    return idx + 2, prob.numpy().astype(np.float32), float((top2[0] - top2[1]).item())


def apply_count_policy(
    count: int,
    count_prob: np.ndarray | torch.Tensor | None = None,
    dataset_name: str | None = "tusimple",
    merge_tusimple_2_to_3: bool = True,
    min_count: int | None = None,
    max_count: int = 5,
) -> int:
    """Clamp Count Head K to the dataset output space without using score-gap rules."""
    del count_prob
    name = str(dataset_name or "").lower()
    k = int(count)
    if min_count is None:
        min_count = 3 if name == "tusimple" else 2
    if name == "tusimple" and merge_tusimple_2_to_3 and k <= 2:
        k = 3
    return int(max(int(min_count), min(int(max_count), k)))


def count_head_decode_meta(
    pred_count_logits: torch.Tensor | None,
    pred_count_boundary_logits: torch.Tensor | None = None,
    *,
    use_count_head_decode: bool = True,
    count_head_temperature: float = 1.0,
    dataset_name: str | None = "tusimple",
    count_head_min_count: int | None = None,
    count_head_max_count: int = 5,
    merge_tusimple_2_to_3: bool = True,
    max_det: int | None = None,
) -> dict | None:
    """Return Count Head policy K metadata, or fail if requested logits are missing.

    The count margin is diagnostic only. K is decided by Count Head softmax plus
    dataset clamp policy; no score-gap rule fallback is applied.
    """
    if not use_count_head_decode:
        return None
    if pred_count_logits is None:
        raise ValueError(
            "Count Head Top-K decode requested but pred_count_logits is missing. "
            "Old GCS checkpoints without trained count_head.* tensors are warm-start sources only; "
            "finetune/train Count Head before reporting Count Head Top-K results, or pass "
            "--no-count-head-decode for legacy non-Count-Head diagnostics."
        )
    k_head_raw, count_prob_raw, count_margin_raw = predict_count_from_logits(
        pred_count_logits,
        temperature=float(count_head_temperature),
    )
    count_prob = count_prob_raw
    k_head = k_head_raw
    count_boundary_prob: list[float] | None = None
    count_boundary_applied = False
    if pred_count_boundary_logits is not None:
        boundary_logits = pred_count_boundary_logits.detach().float().cpu().reshape(-1)
        if boundary_logits.numel() != 2:
            raise ValueError(
                "pred_count_boundary_logits must have shape 2 for one image, "
                f"got {tuple(pred_count_boundary_logits.shape)}."
            )
        boundary_prob_t = torch.sigmoid(boundary_logits)
        count_boundary_prob = [float(x) for x in boundary_prob_t.tolist()]
        if not torch.allclose(boundary_logits, torch.zeros_like(boundary_logits)):
            p_ge4, p_ge5 = boundary_prob_t.numpy().astype(np.float32).tolist()
            boundary_likelihood = np.asarray(
                [1.0 - p_ge4, 1.0 - p_ge4, p_ge4 * (1.0 - p_ge5), p_ge5], dtype=np.float32
            )
            adjusted = np.asarray(count_prob_raw, dtype=np.float32) * boundary_likelihood
            raw_idx = int(k_head_raw) - 2
            allowed = {raw_idx}
            if raw_idx > 0:
                allowed.add(raw_idx - 1)
            if raw_idx < 3:
                allowed.add(raw_idx + 1)
            masked = np.zeros_like(adjusted, dtype=np.float32)
            for idx in allowed:
                masked[int(idx)] = adjusted[int(idx)]
            if float(masked.sum()) > 0.0:
                count_prob = masked / float(masked.sum())
                k_head = int(count_prob.argmax()) + 2
                count_boundary_applied = int(k_head) != int(k_head_raw)
    top2 = np.sort(np.asarray(count_prob, dtype=np.float32))[-2:]
    count_margin = float(top2[-1] - top2[-2])
    target_count = apply_count_policy(
        k_head,
        count_prob=count_prob,
        dataset_name=dataset_name,
        merge_tusimple_2_to_3=merge_tusimple_2_to_3,
        min_count=count_head_min_count,
        max_count=count_head_max_count,
    )
    if max_det is not None and int(max_det) > 0:
        target_count = min(int(target_count), int(max_det))
    count_prob_list = [float(x) for x in count_prob.tolist()]
    count4_prob = float(count_prob[2]) if len(count_prob) > 2 else 0.0
    count5_prob = float(count_prob[3]) if len(count_prob) > 3 else 0.0
    return {
        "count_head_raw_count": int(k_head_raw),
        "count_head_calibrated_count": int(k_head),
        "count_head_policy_count": int(target_count),
        "effective_policy_count": int(target_count),
        "count_head_prob": count_prob_list,
        "count_head_raw_prob": [float(x) for x in count_prob_raw.tolist()],
        "count_head_calibrated_prob": count_prob_list,
        "count_boundary_prob": count_boundary_prob,
        "count_boundary_applied": bool(count_boundary_applied),
        "count4_prob": count4_prob,
        "count5_prob": count5_prob,
        "count5_margin": float(count5_prob - count4_prob),
        "count_head_margin": float(count_margin),
        "count_head_raw_margin": float(count_margin_raw),
    }


def _soft_duplicate_penalty(
    lanes: list[dict],
    image_shape: tuple[int, int] | None,
    duplicate_dist_px: float = 18.0,
    min_overlap: int = 6,
) -> float:
    """Return a small penalty for near-duplicate lanes in a proposed Top-K set."""
    if image_shape is None or len(lanes) <= 1:
        return 0.0
    penalty = 0.0
    for i in range(len(lanes)):
        for j in range(i + 1, len(lanes)):
            a = torch.from_numpy(np.asarray(lanes[i]["points_norm"], dtype=np.float32))
            b = torch.from_numpy(np.asarray(lanes[j]["points_norm"], dtype=np.float32))
            valid_a = torch.from_numpy(_lane_valid_mask_np(lanes[i]))
            valid_b = torch.from_numpy(_lane_valid_mask_np(lanes[j]))
            dist = lane_x_distance_px(
                a,
                b,
                image_shape=image_shape,
                min_overlap=int(min_overlap),
                valid_a=valid_a,
                valid_b=valid_b,
            )
            if torch.isfinite(dist) and float(dist) < float(duplicate_dist_px):
                penalty += 1.0 - float(dist) / max(float(duplicate_dist_px), 1e-6)
    return float(penalty)


def soft_count_decision(
    count_prob: np.ndarray | torch.Tensor | list[float],
    candidate_lanes: list[dict],
    *,
    image_shape: tuple[int, int] | None = None,
    max_count: int = 5,
    prob_margin: float = 0.08,
    quality_weight: float = 1.0,
    prior_weight: float = 0.5,
    duplicate_penalty: float = 1.0,
    invalid_penalty: float = 1.0,
    min_points: int = 5,
    duplicate_dist_px: float = 18.0,
    min_overlap: int = 6,
) -> dict:
    """Choose a count K by candidate quality when Count Head probabilities are close."""
    prob = np.asarray(count_prob, dtype=np.float64).reshape(-1)
    if prob.size != 4:
        raise ValueError(f"soft_count_decision expects four count probabilities, got shape {prob.shape}.")
    if not np.isfinite(prob).all():
        raise ValueError("soft_count_decision received non-finite count probabilities.")
    count_values = np.asarray([2, 3, 4, 5], dtype=np.int64)
    max_idx = int(prob.argmax())
    raw_count = int(count_values[max_idx])
    max_prob = float(prob[max_idx])
    margin = max(float(prob_margin), 0.0)
    candidate_counts = [
        int(k)
        for k, p in zip(count_values.tolist(), prob.tolist())
        if int(k) <= int(max_count) and max_prob - float(p) <= margin + 1e-12
    ]
    if raw_count not in candidate_counts and raw_count <= int(max_count):
        candidate_counts.append(raw_count)
    candidate_counts = sorted(set(candidate_counts))

    ranked = sorted(candidate_lanes, key=_lane_rank_score, reverse=True)
    scores_by_k: dict[int, float] = {}
    for k in candidate_counts:
        top = ranked[:k]
        quality_sum = sum(float(lane.get("quality_score", _lane_rank_score(lane))) for lane in top)
        prior = float(np.log(max(float(prob[k - 2]), 1e-12)))
        dup = _soft_duplicate_penalty(
            top,
            image_shape=image_shape,
            duplicate_dist_px=float(duplicate_dist_px),
            min_overlap=int(min_overlap),
        )
        short = max(0, k - len(ranked))
        invalid = float(short)
        if top:
            invalid += sum(max(0, int(min_points) - int(lane.get("valid_count", 0))) / max(int(min_points), 1) for lane in top)
        scores_by_k[k] = (
            float(quality_weight) * float(quality_sum)
            + float(prior_weight) * prior
            - float(duplicate_penalty) * float(dup)
            - float(invalid_penalty) * float(invalid)
        )
    soft_count = max(scores_by_k, key=lambda k: (scores_by_k[k], prob[k - 2], -abs(k - raw_count))) if scores_by_k else raw_count
    return {
        "pred_count_cls_raw": int(raw_count),
        "pred_count_cls_soft": int(soft_count),
        "soft_count_changed": bool(int(soft_count) != int(raw_count)),
        "soft_count_candidate_ks": [int(k) for k in candidate_counts],
        "soft_count_score_by_k": {str(int(k)): float(v) for k, v in sorted(scores_by_k.items())},
    }


def empty_decode_count_state() -> dict:
    """Create aggregate Count Head K vs final-output counters."""
    return {
        "decode_count_head_images": 0,
        "decode_count_head_k_sum": 0.0,
        "decode_final_pred_lanes_sum": 0.0,
        "decode_count_shortfall": 0,
        "decode_count_shortfall_sum": 0.0,
        "decode_count_head_k4_images": 0,
        "decode_count_head_k5_images": 0,
        "decode_k4_to_output5": 0,
        "decode_k5_to_output4": 0,
        "decode_candidate_pool_images": 0,
        "decode_candidate_pool_shortfall": 0,
        "decode_top5_suppressed_by_nms": 0,
    }


def update_decode_count_state(state: dict, count_head_meta: dict | None, final_pred_lanes: int) -> None:
    """Accumulate Count Head policy K and final decoded lane-count diagnostics."""
    if not count_head_meta:
        return
    k = int(count_head_meta["count_head_policy_count"])
    final_count = int(final_pred_lanes)
    shortfall = max(0, k - final_count)
    state["decode_count_head_images"] = int(state.get("decode_count_head_images", 0)) + 1
    state["decode_count_head_k_sum"] = float(state.get("decode_count_head_k_sum", 0.0)) + float(k)
    state["decode_final_pred_lanes_sum"] = float(state.get("decode_final_pred_lanes_sum", 0.0)) + float(final_count)
    state["decode_count_shortfall"] = int(state.get("decode_count_shortfall", 0)) + int(shortfall > 0)
    state["decode_count_shortfall_sum"] = float(state.get("decode_count_shortfall_sum", 0.0)) + float(shortfall)
    if k == 4:
        state["decode_count_head_k4_images"] = int(state.get("decode_count_head_k4_images", 0)) + 1
        state["decode_k4_to_output5"] = int(state.get("decode_k4_to_output5", 0)) + int(final_count == 5)
    if k == 5:
        state["decode_count_head_k5_images"] = int(state.get("decode_count_head_k5_images", 0)) + 1
        state["decode_k5_to_output4"] = int(state.get("decode_k5_to_output4", 0)) + int(final_count == 4)
    if "candidate_pool_shortfall" in count_head_meta:
        state["decode_candidate_pool_images"] = int(state.get("decode_candidate_pool_images", 0)) + 1
        state["decode_candidate_pool_shortfall"] = int(state.get("decode_candidate_pool_shortfall", 0)) + int(
            int(count_head_meta.get("candidate_pool_shortfall", 0) or 0) > 0
        )
        state["decode_top5_suppressed_by_nms"] = int(state.get("decode_top5_suppressed_by_nms", 0)) + int(
            bool(count_head_meta.get("top5_suppressed_by_nms", False))
        )


def summarize_decode_count_state(state: dict, prefix: str = "decode/") -> dict[str, float]:
    """Summarize aggregate Count Head K vs final-output diagnostics."""
    n = int(state.get("decode_count_head_images", 0))
    k4 = int(state.get("decode_count_head_k4_images", 0))
    k5 = int(state.get("decode_count_head_k5_images", 0))
    pool_n = int(state.get("decode_candidate_pool_images", 0))
    return {
        f"{prefix}count_head_images": float(n),
        f"{prefix}count_head_k": round(float(state.get("decode_count_head_k_sum", 0.0)) / max(n, 1), 6),
        f"{prefix}final_pred_lanes": round(float(state.get("decode_final_pred_lanes_sum", 0.0)) / max(n, 1), 6),
        f"{prefix}count_shortfall_rate": round(float(state.get("decode_count_shortfall", 0)) / max(n, 1), 6),
        f"{prefix}count_shortfall_mean": round(float(state.get("decode_count_shortfall_sum", 0.0)) / max(n, 1), 6),
        f"{prefix}k4_to_output5_rate": round(float(state.get("decode_k4_to_output5", 0)) / max(k4, 1), 6),
        f"{prefix}k5_to_output4_rate": round(float(state.get("decode_k5_to_output4", 0)) / max(k5, 1), 6),
        f"{prefix}candidate_pool_shortfall_rate": round(
            float(state.get("decode_candidate_pool_shortfall", 0)) / max(pool_n, 1), 6
        ),
        f"{prefix}top5_suppressed_by_nms_rate": round(
            float(state.get("decode_top5_suppressed_by_nms", 0)) / max(pool_n, 1), 6
        ),
    }


def _apply_rank_min_points(lanes: list[dict], default_min_points: int, rank_min_points: Mapping[int, int] | None) -> list[dict]:
    """Filter rank-sorted candidates with per-selected-rank min_points requirements."""
    cfg = _normalize_rank_min_points(rank_min_points)
    if cfg is None:
        return lanes
    selected: list[dict] = []
    for lane in sorted(lanes, key=_lane_rank_score, reverse=True):
        selected_rank = len(selected) + 1
        required = int(cfg.get(selected_rank, default_min_points))
        if int(lane.get("valid_count", 0)) < required:
            continue
        lane = dict(lane)
        lane["rank_selection_rank"] = int(selected_rank)
        lane["rank_min_points_required"] = int(required)
        selected.append(lane)
    return selected


def _required_min_points_for_rank(
    selected_rank: int,
    default_min_points: int,
    rank_min_points: Mapping[int, int] | None = None,
    fifth_min_points: int | None = None,
) -> int:
    """Return final visible-anchor requirement for a selected output rank."""
    cfg = _normalize_rank_min_points(rank_min_points)
    if cfg is not None and int(selected_rank) in cfg:
        return int(cfg[int(selected_rank)])
    if int(selected_rank) == 5 and fifth_min_points is not None:
        return int(fifth_min_points)
    return int(default_min_points)


def _lane_valid_mask_np(lane: dict) -> np.ndarray:
    """Return a boolean K-vector visibility mask for one decoded lane."""
    points = np.asarray(lane.get("points_norm", lane.get("points", [])), dtype=np.float32)
    valid = lane.get("point_valid")
    if valid is None:
        return np.ones((points.shape[0],), dtype=bool)
    valid = np.asarray(valid, dtype=np.float32).reshape(-1) > 0.5
    if valid.shape[0] != points.shape[0]:
        return np.ones((points.shape[0],), dtype=bool)
    return valid


def _lanes_too_close(lane_a: dict, lane_b: dict, image_shape: tuple[int, int], dist_thr_px: float, min_overlap: int) -> bool:
    """Return True when two decoded lanes overlap laterally within dist_thr_px."""
    if dist_thr_px <= 0.0:
        return False
    points_a = torch.from_numpy(np.asarray(lane_a["points_norm"], dtype=np.float32))
    points_b = torch.from_numpy(np.asarray(lane_b["points_norm"], dtype=np.float32))
    valid_a = torch.from_numpy(_lane_valid_mask_np(lane_a))
    valid_b = torch.from_numpy(_lane_valid_mask_np(lane_b))
    dist = lane_x_distance_px(points_a, points_b, image_shape, min_overlap=min_overlap, valid_a=valid_a, valid_b=valid_b)
    return bool(torch.isfinite(dist) and float(dist) <= float(dist_thr_px))


def _lane_min_distance_to_selected(lane: dict, selected: list[dict], image_shape: tuple[int, int], min_overlap: int) -> float:
    """Return the minimum finite lateral distance from a candidate to already selected lanes."""
    if not selected:
        return float("inf")
    points_a = torch.from_numpy(np.asarray(lane["points_norm"], dtype=np.float32))
    valid_a = torch.from_numpy(_lane_valid_mask_np(lane))
    distances = []
    for kept in selected:
        points_b = torch.from_numpy(np.asarray(kept["points_norm"], dtype=np.float32))
        valid_b = torch.from_numpy(_lane_valid_mask_np(kept))
        dist = lane_x_distance_px(
            points_a,
            points_b,
            image_shape,
            min_overlap=int(min_overlap),
            valid_a=valid_a,
            valid_b=valid_b,
        )
        if torch.isfinite(dist):
            distances.append(float(dist))
    return min(distances) if distances else float("inf")


def _select_topk_with_final_constraints(
    lanes: list[dict],
    target_count: int,
    default_min_points: int,
    *,
    rank_min_points: Mapping[int, int] | None = None,
    fifth_min_points: int | None = None,
) -> list[dict]:
    """Select rank-score Top-K lanes while enforcing final per-selected-rank visible-point floors."""
    selected: list[dict] = []
    for lane in sorted(lanes, key=_lane_rank_score, reverse=True):
        selected_rank = len(selected) + 1
        required = _required_min_points_for_rank(
            selected_rank,
            default_min_points,
            rank_min_points=rank_min_points,
            fifth_min_points=fifth_min_points,
        )
        if int(lane.get("valid_count", 0)) < int(required):
            continue
        lane = dict(lane)
        lane["rank_selection_rank"] = int(selected_rank)
        lane["rank_min_points_required"] = int(required)
        selected.append(lane)
        if len(selected) >= int(target_count):
            break
    return selected


def _rescue_missing_lanes(
    selected: list[dict],
    pre_nms_candidates: list[dict],
    target_count: int,
    image_shape: tuple[int, int] | None,
    default_min_points: int,
    *,
    rank_min_points: Mapping[int, int] | None = None,
    fifth_min_points: int | None = None,
    rescue_dist_px: float = 0.0,
    min_overlap: int = 6,
) -> list[dict]:
    """Fill a Count Head Top-K shortfall from pre-NMS candidates without fabricating lanes."""
    if len(selected) >= int(target_count):
        return selected
    selected_queries = {int(x.get("query", -1)) for x in selected}
    out = list(selected)
    for lane in sorted(pre_nms_candidates, key=_lane_rank_score, reverse=True):
        if int(lane.get("query", -1)) in selected_queries:
            continue
        selected_rank = len(out) + 1
        required = _required_min_points_for_rank(
            selected_rank,
            default_min_points,
            rank_min_points=rank_min_points,
            fifth_min_points=fifth_min_points,
        )
        if int(lane.get("valid_count", 0)) < int(required):
            continue
        if image_shape is not None and any(
            _lanes_too_close(lane, kept, image_shape, rescue_dist_px, min_overlap) for kept in out
        ):
            continue
        lane = dict(lane)
        lane["rank_selection_rank"] = int(selected_rank)
        lane["rank_min_points_required"] = int(required)
        lane["count_head_rescue"] = True
        lane["source"] = "rescue_refill"
        out.append(lane)
        selected_queries.add(int(lane.get("query", -1)))
        if len(out) >= int(target_count):
            break
    return out


def count_aware_refill(
    selected: list[dict],
    pre_nms_candidates: list[dict],
    target_count: int,
    image_shape: tuple[int, int] | None,
    default_min_points: int,
    *,
    rank_min_points: Mapping[int, int] | None = None,
    fifth_min_points: int | None = None,
    rescue_dist_px: float = 0.0,
    min_overlap: int = 6,
) -> list[dict]:
    """Public Count Head refill helper that only reuses real pre-NMS candidates."""
    return _rescue_missing_lanes(
        selected=selected,
        pre_nms_candidates=pre_nms_candidates,
        target_count=target_count,
        image_shape=image_shape,
        default_min_points=default_min_points,
        rank_min_points=rank_min_points,
        fifth_min_points=fifth_min_points,
        rescue_dist_px=rescue_dist_px,
        min_overlap=min_overlap,
    )



def _lane_points_px(lane: dict, image_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """Return one decoded lane's Kx2 points in pixels."""
    if "points" in lane:
        points = np.asarray(lane.get("points", []), dtype=np.float32)
    else:
        points = np.asarray(lane.get("points_norm", []), dtype=np.float32)
        if image_shape is None:
            return None
        points = points * np.asarray([float(image_shape[1]), float(image_shape[0])], dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        return None
    return points


def _lane_x_at_y_px(lane: dict, y_px: float, image_shape: tuple[int, int] | None = None) -> float | None:
    """Interpolate a visible lane's x at one y coordinate without extrapolation."""
    points = _lane_points_px(lane, image_shape=image_shape)
    if points is None:
        return None
    valid = _lane_valid_mask_np(lane)
    if valid.shape[0] != points.shape[0]:
        valid = np.ones((points.shape[0],), dtype=bool)
    finite = np.isfinite(points).all(axis=1)
    visible = points[valid & finite]
    if visible.shape[0] == 0:
        return None
    order = np.argsort(visible[:, 1], kind="stable")
    ys = visible[order, 1]
    xs = visible[order, 0]
    ys, unique_indices = np.unique(ys, return_index=True)
    xs = xs[unique_indices]
    if ys.shape[0] == 1:
        return float(xs[0]) if abs(float(y_px) - float(ys[0])) <= 1.0 else None
    if float(y_px) < float(ys[0]) or float(y_px) > float(ys[-1]):
        return None
    return float(np.interp(float(y_px), ys, xs))


def _edge_side_and_gap(
    lane: dict,
    selected: list[dict],
    image_shape: tuple[int, int] | None,
    outside_gap_px: float,
) -> tuple[str | None, float | None, float | None]:
    """Return edge side and gap using x positions compared at shared y coordinates."""
    points = _lane_points_px(lane, image_shape=image_shape)
    if points is None:
        return None, None, None
    valid = _lane_valid_mask_np(lane)
    if valid.shape[0] != points.shape[0]:
        valid = np.ones((points.shape[0],), dtype=bool)
    candidate_points = points[valid & np.isfinite(points).all(axis=1)][:6]
    if candidate_points.shape[0] == 0:
        return None, None, None

    left_gaps: list[float] = []
    right_gaps: list[float] = []
    candidate_xs: list[float] = []
    for candidate_x, candidate_y in candidate_points:
        selected_xs = [
            x
            for x in (
                _lane_x_at_y_px(item, float(candidate_y), image_shape=image_shape)
                for item in selected
            )
            if x is not None and np.isfinite(float(x))
        ]
        if not selected_xs:
            continue
        candidate_x = float(candidate_x)
        candidate_xs.append(candidate_x)
        left_gaps.append(float(min(selected_xs)) - candidate_x)
        right_gaps.append(candidate_x - float(max(selected_xs)))
    if not candidate_xs:
        return None, None, None

    left_gap = float(np.median(np.asarray(left_gaps, dtype=np.float32)))
    right_gap = float(np.median(np.asarray(right_gaps, dtype=np.float32)))
    candidate_x = float(np.median(np.asarray(candidate_xs, dtype=np.float32)))
    gap = max(left_gap, right_gap)
    if left_gap > float(outside_gap_px):
        return "left", left_gap, candidate_x
    if right_gap > float(outside_gap_px):
        return "right", right_gap, candidate_x
    return None, gap, candidate_x


def _find_edge_rescue_candidate(
    selected: list[dict],
    rescue_candidates: list[dict],
    image_shape: tuple[int, int] | None,
    *,
    selected_queries: set[int] | None = None,
    edge_conf_thr: float = 0.02,
    edge_mean_valid_thr: float = 0.35,
    edge_quality_thr: float = 0.45,
    edge_min_points: int = 4,
    edge_outside_gap_px: float = 28.0,
    edge_dist_px: float = 24.0,
    min_overlap: int = 6,
) -> tuple[dict | None, dict]:
    """Select the best outside-left/right real-query candidate for edge rescue."""
    meta = {
        "edge_last_lane_rescue_reason": "no_candidate",
        "edge_last_lane_rescue_candidate_quality": None,
        "edge_last_lane_rescue_candidate_valid_points": None,
        "edge_last_lane_rescue_candidate_min_dist": None,
        "edge_last_lane_rescue_candidate_side": None,
        "edge_last_lane_rescue_candidate_outside_gap_px": None,
    }
    if image_shape is None:
        meta["edge_last_lane_rescue_reason"] = "missing_image_shape"
        return None, meta
    if not selected:
        meta["edge_last_lane_rescue_reason"] = "no_selected_edge_reference"
        return None, meta
    selected_queries = set() if selected_queries is None else set(selected_queries)
    edge_candidates: list[dict] = []
    saw_edge = False
    for lane in rescue_candidates:
        if int(lane.get("query", -1)) in selected_queries:
            continue
        side, outside_gap, _ = _edge_side_and_gap(
            lane,
            selected,
            image_shape=image_shape,
            outside_gap_px=float(edge_outside_gap_px),
        )
        if side is None:
            continue
        saw_edge = True
        meta["edge_last_lane_rescue_candidate_side"] = side
        meta["edge_last_lane_rescue_candidate_outside_gap_px"] = None if outside_gap is None else float(outside_gap)
        meta["edge_last_lane_rescue_candidate_valid_points"] = int(lane.get("valid_count", 0))
        quality_score = lane.get("quality_score")
        meta["edge_last_lane_rescue_candidate_quality"] = None if quality_score is None else float(quality_score)
        if float(lane.get("exist_score", lane.get("score", 0.0))) < float(edge_conf_thr):
            meta["edge_last_lane_rescue_reason"] = "conf_low"
            continue
        if int(lane.get("valid_count", 0)) < int(edge_min_points):
            meta["edge_last_lane_rescue_reason"] = "valid_points_fail"
            continue
        if float(lane.get("mean_valid_score", 0.0)) < float(edge_mean_valid_thr):
            meta["edge_last_lane_rescue_reason"] = "mean_valid_low"
            continue
        if quality_score is None and float(edge_quality_thr) > 0.0:
            meta["edge_last_lane_rescue_reason"] = "missing_quality"
            continue
        if quality_score is not None and float(quality_score) < float(edge_quality_thr):
            meta["edge_last_lane_rescue_reason"] = "quality_too_low"
            continue
        min_dist = _lane_min_distance_to_selected(lane, selected, image_shape, min_overlap)
        meta["edge_last_lane_rescue_candidate_min_dist"] = None if not np.isfinite(min_dist) else float(min_dist)
        if min_dist < float(edge_dist_px):
            meta["edge_last_lane_rescue_reason"] = "distance_too_close"
            continue
        lane = dict(lane)
        lane["edge_rescue_side"] = side
        lane["edge_rescue_outside_gap_px"] = None if outside_gap is None else float(outside_gap)
        lane["edge_rescue_min_dist"] = None if not np.isfinite(min_dist) else float(min_dist)
        edge_score = (
            _lane_rank_score(lane)
            + 0.10 * min(float(outside_gap or 0.0) / 120.0, 1.0)
            + 0.03 * min(float(lane.get("valid_count", 0)) / 8.0, 1.0)
        )
        if side == "left":
            edge_score += 0.01
        lane["edge_rescue_score"] = float(edge_score)
        edge_candidates.append(lane)
    if not edge_candidates:
        if not saw_edge:
            meta["edge_last_lane_rescue_reason"] = "no_outside_edge_candidate"
        return None, meta
    edge_candidates.sort(key=lambda x: (float(x.get("edge_rescue_score", 0.0)), _lane_rank_score(x)), reverse=True)
    best = edge_candidates[0]
    meta["edge_last_lane_rescue_reason"] = "candidate_found"
    meta["edge_last_lane_rescue_candidate_quality"] = (
        None if best.get("quality_score") is None else float(best.get("quality_score"))
    )
    meta["edge_last_lane_rescue_candidate_valid_points"] = int(best.get("valid_count", 0))
    meta["edge_last_lane_rescue_candidate_min_dist"] = best.get("edge_rescue_min_dist")
    meta["edge_last_lane_rescue_candidate_side"] = best.get("edge_rescue_side")
    meta["edge_last_lane_rescue_candidate_outside_gap_px"] = best.get("edge_rescue_outside_gap_px")
    return best, meta


def _edge_last_lane_rescue(
    selected: list[dict],
    rescue_candidates: list[dict],
    target_count: int,
    image_shape: tuple[int, int] | None,
    *,
    min_policy_count: int = 4,
    edge_conf_thr: float = 0.02,
    edge_mean_valid_thr: float = 0.35,
    edge_quality_thr: float = 0.45,
    edge_min_points: int = 4,
    edge_outside_gap_px: float = 28.0,
    edge_dist_px: float = 24.0,
    min_overlap: int = 6,
) -> tuple[list[dict], dict]:
    """Prioritize rescuing an outside-left/right lane from real model candidates only."""
    target_count = int(target_count)
    meta = {
        "edge_last_lane_rescue_attempt_count": 0,
        "edge_last_lane_rescue_success_count": 0,
        "edge_last_lane_rescue_reason": "not_attempted",
        "edge_last_lane_rescue_candidate_quality": None,
        "edge_last_lane_rescue_candidate_valid_points": None,
        "edge_last_lane_rescue_candidate_min_dist": None,
        "edge_last_lane_rescue_candidate_side": None,
        "edge_last_lane_rescue_candidate_outside_gap_px": None,
    }
    if target_count < int(min_policy_count):
        meta["edge_last_lane_rescue_reason"] = "policy_count_low"
        return selected, meta
    if len(selected) >= target_count:
        meta["edge_last_lane_rescue_reason"] = "no_shortfall"
        return selected, meta
    meta["edge_last_lane_rescue_attempt_count"] = 1
    out = list(selected)
    selected_queries = {int(x.get("query", -1)) for x in out}
    while len(out) < target_count:
        picked, picked_meta = _find_edge_rescue_candidate(
            selected=out,
            rescue_candidates=rescue_candidates,
            image_shape=image_shape,
            selected_queries=selected_queries,
            edge_conf_thr=float(edge_conf_thr),
            edge_mean_valid_thr=float(edge_mean_valid_thr),
            edge_quality_thr=float(edge_quality_thr),
            edge_min_points=int(edge_min_points),
            edge_outside_gap_px=float(edge_outside_gap_px),
            edge_dist_px=float(edge_dist_px),
            min_overlap=int(min_overlap),
        )
        meta.update(picked_meta)
        if picked is None:
            break
        selected_rank = len(out) + 1
        picked = dict(picked)
        picked["rank_selection_rank"] = int(selected_rank)
        picked["rank_min_points_required"] = int(edge_min_points)
        picked["count_head_rescue"] = True
        picked["last_lane_rescue"] = True
        picked["edge_last_lane_rescue"] = True
        picked["source"] = "edge_last_lane_rescue"
        out.append(picked)
        selected_queries.add(int(picked.get("query", -1)))
        meta["edge_last_lane_rescue_success_count"] = int(meta["edge_last_lane_rescue_success_count"]) + 1
        meta["edge_last_lane_rescue_reason"] = "rescued"
        meta["edge_last_lane_rescue_candidate_quality"] = (
            None if picked.get("quality_score") is None else float(picked.get("quality_score"))
        )
        meta["edge_last_lane_rescue_candidate_valid_points"] = int(picked.get("valid_count", 0))
        meta["edge_last_lane_rescue_candidate_min_dist"] = picked.get("edge_rescue_min_dist")
        meta["edge_last_lane_rescue_candidate_side"] = picked.get("edge_rescue_side")
        meta["edge_last_lane_rescue_candidate_outside_gap_px"] = picked.get("edge_rescue_outside_gap_px")
    if len(out) < target_count and int(meta["edge_last_lane_rescue_success_count"]) > 0:
        meta["edge_last_lane_rescue_reason"] = "partial_rescue"
    return out, meta


def _edge_count4_to5_upgrade_eligible(
    selected: list[dict],
    rescue_candidates: list[dict],
    count_head_meta: dict | None,
    image_shape: tuple[int, int] | None,
    *,
    enabled: bool = True,
    prob_margin: float = 0.12,
    edge_conf_thr: float = 0.02,
    edge_mean_valid_thr: float = 0.35,
    edge_quality_thr: float = 0.45,
    edge_min_points: int = 4,
    edge_outside_gap_px: float = 28.0,
    edge_dist_px: float = 24.0,
    min_overlap: int = 6,
) -> tuple[bool, dict]:
    """Allow only Count Head K=4 -> effective K=5 when a valid outside-edge candidate exists."""
    meta = {
        "edge_count4_to5_upgrade": False,
        "edge_count4_to5_upgrade_reason": "not_attempted",
        "edge_count4_to5_prob_margin": float(prob_margin),
        "edge_count4_to5_p4": None,
        "edge_count4_to5_p5": None,
        "edge_count4_to5_candidate_side": None,
        "edge_count4_to5_candidate_valid_points": None,
        "edge_count4_to5_candidate_quality": None,
        "edge_count4_to5_candidate_min_dist": None,
    }
    if not enabled:
        meta["edge_count4_to5_upgrade_reason"] = "disabled"
        return False, meta
    if count_head_meta is None:
        meta["edge_count4_to5_upgrade_reason"] = "missing_count_head"
        return False, meta
    prob = np.asarray(count_head_meta.get("count_head_prob", []), dtype=np.float64).reshape(-1)
    if prob.size != 4:
        meta["edge_count4_to5_upgrade_reason"] = "missing_count_prob"
        return False, meta
    p4 = float(prob[2])
    p5 = float(prob[3])
    meta["edge_count4_to5_p4"] = p4
    meta["edge_count4_to5_p5"] = p5
    if int(count_head_meta.get("count_head_raw_count", count_head_meta.get("count_head_policy_count", 0))) != 4:
        meta["edge_count4_to5_upgrade_reason"] = "raw_count_not_4"
        return False, meta
    if (p4 - p5) > float(prob_margin):
        meta["edge_count4_to5_upgrade_reason"] = "p5_not_close_to_p4"
        return False, meta
    picked, picked_meta = _find_edge_rescue_candidate(
        selected=selected,
        rescue_candidates=rescue_candidates,
        image_shape=image_shape,
        selected_queries={int(x.get("query", -1)) for x in selected},
        edge_conf_thr=float(edge_conf_thr),
        edge_mean_valid_thr=float(edge_mean_valid_thr),
        edge_quality_thr=float(edge_quality_thr),
        edge_min_points=int(edge_min_points),
        edge_outside_gap_px=float(edge_outside_gap_px),
        edge_dist_px=float(edge_dist_px),
        min_overlap=int(min_overlap),
    )
    if picked is None:
        meta["edge_count4_to5_upgrade_reason"] = picked_meta.get("edge_last_lane_rescue_reason", "no_valid_edge_candidate")
        meta["edge_count4_to5_candidate_side"] = picked_meta.get("edge_last_lane_rescue_candidate_side")
        meta["edge_count4_to5_candidate_valid_points"] = picked_meta.get("edge_last_lane_rescue_candidate_valid_points")
        meta["edge_count4_to5_candidate_quality"] = picked_meta.get("edge_last_lane_rescue_candidate_quality")
        meta["edge_count4_to5_candidate_min_dist"] = picked_meta.get("edge_last_lane_rescue_candidate_min_dist")
        return False, meta
    meta["edge_count4_to5_upgrade"] = True
    meta["edge_count4_to5_upgrade_reason"] = "edge_candidate_found"
    meta["edge_count4_to5_candidate_side"] = picked.get("edge_rescue_side")
    meta["edge_count4_to5_candidate_valid_points"] = int(picked.get("valid_count", 0))
    meta["edge_count4_to5_candidate_quality"] = None if picked.get("quality_score") is None else float(picked.get("quality_score"))
    meta["edge_count4_to5_candidate_min_dist"] = picked.get("edge_rescue_min_dist")
    return True, meta

def _last_required_lane_rescue(
    selected: list[dict],
    rescue_candidates: list[dict],
    target_count: int,
    image_shape: tuple[int, int] | None,
    *,
    min_policy_count: int = 4,
    rescue_conf_thr: float = 0.0,
    rescue_mean_valid_thr: float = 0.40,
    rescue_quality_thr: float = 0.50,
    rescue_min_points: int = 4,
    rescue_dist_px: float = 24.0,
    min_overlap: int = 6,
) -> tuple[list[dict], dict]:
    """Fill the final required lane for high-count policies using only a gated weak candidate pool."""
    target_count = int(target_count)
    meta = {
        "last_lane_rescue_attempt_count": 0,
        "last_lane_rescue_success_count": 0,
        "last_lane_rescue_reason": "not_attempted",
        "last_lane_rescue_candidate_quality": None,
        "last_lane_rescue_candidate_valid_points": None,
        "last_lane_rescue_candidate_min_dist": None,
    }
    if target_count < int(min_policy_count):
        meta["last_lane_rescue_reason"] = "policy_count_low"
        return selected, meta
    if len(selected) >= target_count:
        meta["last_lane_rescue_reason"] = "no_shortfall"
        return selected, meta

    meta["last_lane_rescue_attempt_count"] = 1
    selected_queries = {int(x.get("query", -1)) for x in selected}
    out = list(selected)
    saw_candidate = False
    for lane in sorted(rescue_candidates, key=_lane_rank_score, reverse=True):
        if int(lane.get("query", -1)) in selected_queries:
            continue
        saw_candidate = True
        meta["last_lane_rescue_candidate_valid_points"] = int(lane.get("valid_count", 0))
        quality_score = lane.get("quality_score")
        meta["last_lane_rescue_candidate_quality"] = None if quality_score is None else float(quality_score)
        if float(lane.get("exist_score", lane.get("score", 0.0))) < float(rescue_conf_thr):
            meta["last_lane_rescue_reason"] = "conf_low"
            continue
        if int(lane.get("valid_count", 0)) < int(rescue_min_points):
            meta["last_lane_rescue_reason"] = "valid_points_fail"
            continue
        if float(lane.get("mean_valid_score", 0.0)) < float(rescue_mean_valid_thr):
            meta["last_lane_rescue_reason"] = "mean_valid_low"
            continue
        if quality_score is None and float(rescue_quality_thr) > 0.0:
            meta["last_lane_rescue_reason"] = "missing_quality"
            continue
        if quality_score is not None and float(quality_score) < float(rescue_quality_thr):
            meta["last_lane_rescue_reason"] = "quality_too_low"
            continue

        min_dist = float("inf")
        if image_shape is not None:
            min_dist = _lane_min_distance_to_selected(lane, out, image_shape, min_overlap)
            meta["last_lane_rescue_candidate_min_dist"] = None if not np.isfinite(min_dist) else float(min_dist)
            if min_dist < float(rescue_dist_px):
                meta["last_lane_rescue_reason"] = "distance_too_close"
                continue

        selected_rank = len(out) + 1
        lane = dict(lane)
        lane["rank_selection_rank"] = int(selected_rank)
        lane["rank_min_points_required"] = int(rescue_min_points)
        lane["count_head_rescue"] = True
        lane["last_lane_rescue"] = True
        lane["last_lane_rescue_min_dist"] = None if not np.isfinite(min_dist) else float(min_dist)
        lane["source"] = "last_lane_rescue"
        out.append(lane)
        selected_queries.add(int(lane.get("query", -1)))
        meta["last_lane_rescue_success_count"] = int(meta["last_lane_rescue_success_count"]) + 1
        meta["last_lane_rescue_reason"] = "rescued"
        meta["last_lane_rescue_candidate_quality"] = None if quality_score is None else float(quality_score)
        meta["last_lane_rescue_candidate_valid_points"] = int(lane.get("valid_count", 0))
        meta["last_lane_rescue_candidate_min_dist"] = None if not np.isfinite(min_dist) else float(min_dist)
        if len(out) >= target_count:
            break

    if not saw_candidate:
        meta["last_lane_rescue_reason"] = "no_candidate"
    elif len(out) < target_count and meta["last_lane_rescue_reason"] == "rescued":
        meta["last_lane_rescue_reason"] = "partial_rescue"
    return out, meta


def _quality_gated_rescue_5th(
    selected: list[dict],
    pre_nms_candidates: list[dict],
    count_head_meta: dict | None,
    image_shape: tuple[int, int] | None,
    *,
    count5_thr: float = 0.70,
    rescue_conf_thr: float = 0.03,
    rescue_mean_valid_thr: float = 0.45,
    rescue_quality_thr: float = 0.55,
    rescue_min_points: int = 5,
    rescue_dist_px: float = 24.0,
    min_overlap: int = 6,
) -> tuple[list[dict], dict]:
    """Conservatively add a fifth lane only when Count Head and lane quality agree."""
    meta = {
        "rescue_attempted": False,
        "rescue_success": False,
        "rescue_reason": "not_attempted",
        "rescue_candidate_quality": None,
        "rescue_candidate_valid_points": None,
        "rescue_candidate_min_dist": None,
    }
    if len(selected) != 4 or not count_head_meta:
        meta["rescue_reason"] = "not_four_lanes" if len(selected) != 4 else "missing_count_meta"
        return selected, meta
    count_prob = count_head_meta.get("count_head_prob")
    if not count_prob or len(count_prob) < 4:
        meta["rescue_reason"] = "missing_count_prob"
        return selected, meta
    count5_prob = float(count_prob[3])
    if count5_prob < float(count5_thr):
        meta["rescue_attempted"] = True
        meta["rescue_reason"] = "count5_prob_low"
        return selected, meta

    meta["rescue_attempted"] = True
    selected_queries = {int(x.get("query", -1)) for x in selected}
    saw_candidate = False
    for lane in sorted(pre_nms_candidates, key=_lane_rank_score, reverse=True):
        if int(lane.get("query", -1)) in selected_queries:
            continue
        saw_candidate = True
        quality_score = lane.get("quality_score")
        meta["rescue_candidate_quality"] = None if quality_score is None else float(quality_score)
        meta["rescue_candidate_valid_points"] = int(lane.get("valid_count", 0))
        if quality_score is None:
            meta["rescue_reason"] = "missing_quality"
            continue
        if float(lane.get("exist_score", lane.get("score", 0.0))) < float(rescue_conf_thr):
            meta["rescue_reason"] = "conf_low"
            continue
        if float(lane.get("mean_valid_score", 0.0)) < float(rescue_mean_valid_thr):
            meta["rescue_reason"] = "mean_valid_low"
            continue
        if float(quality_score) < float(rescue_quality_thr):
            meta["rescue_reason"] = "quality_too_low"
            continue
        if int(lane.get("valid_count", 0)) < int(rescue_min_points):
            meta["rescue_reason"] = "valid_points_fail"
            continue
        min_dist = float("inf")
        if image_shape is not None:
            min_dist = _lane_min_distance_to_selected(lane, selected, image_shape, min_overlap)
            meta["rescue_candidate_min_dist"] = None if not np.isfinite(min_dist) else float(min_dist)
            if min_dist < float(rescue_dist_px):
                meta["rescue_reason"] = "distance_too_close"
                continue
        lane = dict(lane)
        lane["rank_selection_rank"] = 5
        lane["rank_min_points_required"] = int(rescue_min_points)
        lane["count_head_rescue"] = True
        lane["quality_rescue_5th"] = True
        lane["quality_rescue_count5_prob"] = count5_prob
        lane["quality_rescue_min_dist"] = None if not np.isfinite(min_dist) else float(min_dist)
        meta["rescue_success"] = True
        meta["rescue_reason"] = "rescued"
        meta["rescue_candidate_quality"] = float(quality_score)
        meta["rescue_candidate_valid_points"] = int(lane.get("valid_count", 0))
        meta["rescue_candidate_min_dist"] = None if not np.isfinite(min_dist) else float(min_dist)
        return [*selected, lane], meta
    if not saw_candidate:
        meta["rescue_reason"] = "no_5th_candidate"
    return selected, meta


def _lane_bottom_x(lane: dict) -> float:
    """Return the bottom visible x coordinate used only for final left-to-right output ordering."""
    points = lane.get("visible_points_norm", lane.get("points_norm", []))
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 1:
        return float("inf")
    return float(points[0, 0])


def _lane_rank_quality(
    lane_points: torch.Tensor,
    exist_score: float,
    point_valid_scores: torch.Tensor | None,
    image_shape: tuple[int, int] | None,
    point_valid_thr: float,
    min_points: int,
    lane_quality_score: float | None = None,
) -> dict | None:
    """Return visibility-aware rank factors for one sorted lane candidate."""
    min_points = int(min_points)
    if point_valid_scores is not None:
        threshold_mask = point_valid_scores >= float(point_valid_thr)
        rank_valid_count = int(threshold_mask.sum())
        mean_valid_score = float(point_valid_scores.mean()) if point_valid_scores.numel() > 0 else 0.0
        visible_mask = longest_contiguous_valid_mask(threshold_mask, min_points=min_points)
        valid_count = int(visible_mask.sum())
        if valid_count < min_points:
            return None
        seg_idx = torch.nonzero(visible_mask, as_tuple=False).flatten()
    else:
        if int(lane_points.shape[0]) < min_points:
            return None
        visible_mask = None
        seg_idx = torch.arange(lane_points.shape[0], dtype=torch.long, device=lane_points.device)
        valid_count = int(seg_idx.numel())
        rank_valid_count = valid_count
        mean_valid_score = 1.0 if valid_count > 0 else 0.0

    total_points = max(int(lane_points.shape[0]), 1)
    valid_count_score = float(
        np.clip((float(rank_valid_count) - float(min_points) + 1.0) / float(total_points), 0.0, 1.0)
    )
    length_factor = min(1.0, float(valid_count) / 12.0)
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        scale = lane_points.new_tensor([float(w), float(h)]).view(1, 2)
        points_px = lane_points * scale
    else:
        points_px = lane_points
    x = points_px[seg_idx, 0] if valid_count > 0 else points_px.new_zeros((0,))

    if valid_count >= 3:
        ddx = torch.abs(x[2:] - 2.0 * x[1:-1] + x[:-2]).mean()
        smooth_factor = float(torch.exp(-ddx / 60.0))
    else:
        smooth_factor = 0.5

    if valid_count >= 2:
        max_dx = torch.abs(x[1:] - x[:-1]).max()
        jump_factor = float(torch.exp(-torch.clamp(max_dx - 80.0, min=0.0) / 80.0))
    else:
        jump_factor = 0.5

    geometry_rank_score = float(exist_score) * mean_valid_score * length_factor * smooth_factor * jump_factor
    rank_quality_score = float(exist_score) * mean_valid_score * valid_count_score
    quality_score = (
        None if lane_quality_score is None else float(np.clip(float(lane_quality_score), 0.0, 1.0))
    )
    return {
        "visible_mask": visible_mask,
        "valid_count": valid_count,
        "rank_valid_count": rank_valid_count,
        "mean_valid_score": mean_valid_score,
        "valid_count_score": valid_count_score,
        "length_factor": length_factor,
        "smooth_factor": smooth_factor,
        "jump_factor": jump_factor,
        "geometry_rank_score": geometry_rank_score,
        "rank_quality_score": rank_quality_score,
        "quality_score": quality_score,
        "rank_score": rank_quality_score,
        "rank_score_source": "exist_visibility",
    }


def _build_lane_candidates(
    points: torch.Tensor,
    scores: torch.Tensor,
    point_valid_scores: torch.Tensor | None,
    quality_scores: torch.Tensor | None,
    query_indices: torch.Tensor,
    image_shape: tuple[int, int] | None,
    score_thr: float,
    point_valid_thr: float,
    min_points: int,
) -> list[dict]:
    """Build rank-scored lane candidates from real model queries only."""
    keep = torch.nonzero(scores >= float(score_thr), as_tuple=False).flatten()
    if keep.numel() == 0:
        return []

    scale = None
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        scale = torch.tensor([w, h], dtype=points.dtype).view(1, 1, 2)

    lanes: list[dict] = []
    for raw_idx in keep.tolist():
        order_i = torch.argsort(points[raw_idx, :, 1], descending=True, stable=True)
        lane_points = points[raw_idx][order_i]
        valid_scores_i = point_valid_scores[raw_idx][order_i] if point_valid_scores is not None else None
        quality_score_i = float(quality_scores[raw_idx]) if quality_scores is not None else None
        exist_score = float(scores[raw_idx])
        rank_quality = _lane_rank_quality(
            lane_points,
            exist_score=exist_score,
            point_valid_scores=valid_scores_i,
            image_shape=image_shape,
            point_valid_thr=float(point_valid_thr),
            min_points=int(min_points),
            lane_quality_score=quality_score_i,
        )
        if rank_quality is None:
            continue

        query_idx = int(query_indices[raw_idx])
        lane_norm = lane_points.numpy().astype(np.float32)
        item = {
            "candidate_id": query_idx,
            "score": exist_score,
            "exist_score": exist_score,
            "rank_score": float(rank_quality["rank_score"]),
            "query": query_idx,
            "points_norm": lane_norm,
            "valid_count": int(rank_quality["valid_count"]),
            "rank_valid_count": int(rank_quality["rank_valid_count"]),
            "mean_valid_score": float(rank_quality["mean_valid_score"]),
            "valid_count_score": float(rank_quality["valid_count_score"]),
            "length_factor": float(rank_quality["length_factor"]),
            "smooth_factor": float(rank_quality["smooth_factor"]),
            "jump_factor": float(rank_quality["jump_factor"]),
            "geometry_rank_score": float(rank_quality["geometry_rank_score"]),
            "rank_quality_score": float(rank_quality["rank_quality_score"]),
            "rank_score_source": str(rank_quality["rank_score_source"]),
        }
        if rank_quality["quality_score"] is not None:
            item["quality_score"] = float(rank_quality["quality_score"])
            item["quality_head_score"] = float(rank_quality["quality_score"])
        visible_mask = None
        if point_valid_scores is not None:
            visible_mask_t = rank_quality["visible_mask"]
            visible_mask = visible_mask_t.numpy().astype(bool)
            item["point_valid_scores"] = valid_scores_i.numpy().astype(np.float32)
            item["point_valid"] = visible_mask.astype(np.float32)
            item["visible_points_norm"] = lane_norm[visible_mask]
        if scale is not None:
            points_px = (lane_points.unsqueeze(0) * scale).squeeze(0).numpy().astype(np.float32)
            item["points"] = points_px
            if visible_mask is not None:
                item["visible_points"] = points_px[visible_mask]
        lanes.append(item)
    lanes.sort(key=_lane_rank_score, reverse=True)
    return lanes


def _select_rescue_candidates(
    selected: list[dict],
    rescue_candidates: list[dict],
    target_count: int,
    image_shape: tuple[int, int] | None,
    *,
    rescue_dist_px: float = 0.0,
    min_overlap: int = 6,
) -> list[dict]:
    """Append weak rescue-pool candidates without duplicating queries or nearby lanes."""
    if len(selected) >= int(target_count):
        return selected
    selected_queries = {int(x.get("query", -1)) for x in selected}
    out = list(selected)
    for lane in sorted(rescue_candidates, key=_lane_rank_score, reverse=True):
        if int(lane.get("query", -1)) in selected_queries:
            continue
        if image_shape is not None and any(
            _lanes_too_close(lane, kept, image_shape, rescue_dist_px, min_overlap) for kept in out
        ):
            continue
        lane = dict(lane)
        lane["candidate_pool_rescue"] = True
        out.append(lane)
        selected_queries.add(int(lane.get("query", -1)))
        if len(out) >= int(target_count):
            break
    return out


def decide_lane_count_by_rule(lanes: list[dict], config: str | Mapping | None = None) -> int:
    """Deprecated score-gap count path. Kept only to fail loudly for stale callers."""
    cfg = _normalize_count_calibration_config(config)
    if cfg is None:
        return len(lanes)
    raise ValueError("Score-gap count calibration has been removed. Use Count Head Top-K decode instead.")


def apply_count_calibration(lanes: list[dict], config: str | Mapping | None = None) -> list[dict]:
    """Deprecated no-op unless a stale count config is passed, in which case fail loudly."""
    cfg = _normalize_count_calibration_config(config)
    if cfg is None:
        return lanes
    raise ValueError("Score-gap count calibration has been removed. Use Count Head Top-K decode instead.")


def decode_gcs_predictions(
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None = None,
    pred_count_logits: torch.Tensor | None = None,
    pred_count_boundary_logits: torch.Tensor | None = None,
    pred_quality_logits: torch.Tensor | None = None,
    image_shape: tuple[int, int] | None = None,
    score_thr: float = 0.5,
    point_valid_thr: float = 0.5,
    min_points: int = 2,
    max_det: int | None = None,
    nms_dist_px: float = 0.0,
    count_calibration: str | Mapping | None = None,
    rank_min_points: Mapping[int, int] | None = None,
    use_count_head_decode: bool = True,
    count_head_temperature: float = 1.0,
    dataset_name: str | None = "tusimple",
    count_head_min_count: int | None = None,
    count_head_max_count: int = 5,
    merge_tusimple_2_to_3: bool = True,
    candidate_score_thr: float | None = None,
    candidate_point_valid_thr: float | None = None,
    candidate_min_points: int | None = None,
    enable_rescue_candidate_pool: bool = True,
    rescue_candidate_score_thr: float | None = None,
    rescue_candidate_point_valid_thr: float | None = None,
    rescue_candidate_min_points: int | None = None,
    final_min_points: int | None = None,
    fifth_min_points: int | None = None,
    line_nms_rescue_dist_px: float | None = None,
    line_nms_min_overlap: int = 6,
    quality_rescue_5th: bool = True,
    quality_rescue_count5_thr: float = 0.70,
    quality_rescue_conf_thr: float = 0.03,
    quality_rescue_mean_valid_thr: float = 0.45,
    quality_rescue_quality_thr: float = 0.55,
    quality_rescue_min_points: int = 5,
    quality_rescue_dist_px: float = 24.0,
    last_lane_rescue: bool = False,
    last_lane_rescue_min_policy_count: int = 4,
    last_lane_rescue_conf_thr: float | None = None,
    last_lane_rescue_point_valid_thr: float | None = None,
    last_lane_rescue_min_points: int | None = None,
    last_lane_rescue_mean_valid_thr: float = 0.40,
    last_lane_rescue_quality_thr: float = 0.50,
    last_lane_rescue_dist_px: float = 24.0,
    edge_last_lane_rescue: bool = False,
    edge_rescue_conf_thr: float = 0.02,
    edge_rescue_point_valid_thr: float = 0.06,
    edge_rescue_min_points: int = 4,
    edge_rescue_mean_valid_thr: float = 0.35,
    edge_rescue_quality_thr: float = 0.45,
    edge_rescue_outside_gap_px: float = 28.0,
    edge_rescue_dist_px: float = 24.0,
    edge_rescue_min_policy_count: int = 4,
    edge_count4_to5_upgrade: bool = True,
    edge_count4_to5_prob_margin: float = 0.20,
    enable_soft_count_decision: bool = False,
    soft_count_prob_margin: float = 0.08,
    soft_count_quality_weight: float = 1.0,
    soft_count_prior_weight: float = 0.5,
    soft_count_duplicate_penalty: float = 1.0,
    soft_count_invalid_penalty: float = 1.0,
    return_meta: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Decode ``pred_points`` and ``pred_logits`` into ordered lane point sequences.

    Args:
        pred_points: Q x K x 2 normalized point predictions in the GCS training coordinate system.
        pred_logits: Q existence logits for the lane queries.
        pred_valid_logits: Optional Q x K visibility logits. When present, decoded lanes keep full K points
            for metrics but drawing/export uses the longest visible contiguous point run.
        pred_count_logits: Optional count-head logits for count=2/3/4/5. When present and enabled, this
            predicts the final K; query scores only decide which K lanes survive.
        pred_count_boundary_logits: Optional count>=4/count>=5 logits used only to calibrate image-level K.
        pred_quality_logits: Optional Q lane-quality logits retained for diagnostics and quality-gated rescue.
            Top-K ranking always uses ``exist * mean_point_valid * valid_count_score``.
        image_shape: Optional original image shape as (height, width). If provided, pixel points are added.
        score_thr: Existence probability threshold.
        point_valid_thr: Per-point visibility probability threshold.
        min_points: Default final number of visible anchors required to keep a lane.
        max_det: Optional maximum number of kept lanes after rank-score sorting.
        nms_dist_px: Optional duplicate-lane suppression threshold in pixels. 0 disables lane NMS.
        count_calibration: Deprecated. Non-none score-gap configs now raise; use Count Head Top-K decode.
        rank_min_points: Optional per-selected-rank min_points overrides. Unspecified ranks use ``min_points``.
            For example, ``{5: 5}`` keeps ranks 1-4 at ``min_points`` while allowing the selected 5th lane
            to have 5 contiguous visible anchors. Candidate ranking uses the smallest configured min_points.
        candidate_score_thr: Optional relaxed existence threshold for Count Head candidate-pool construction.
            If omitted, ``score_thr`` is used.
        candidate_point_valid_thr: Optional relaxed point-valid threshold for Count Head candidate-pool construction.
            If omitted, ``point_valid_thr`` is used.
        candidate_min_points: Optional relaxed candidate-pool visible-anchor floor. With Count Head decode enabled,
            the implicit default is ``min(final_min_points, 5)`` so the candidate pool can include a 5-point fifth
            lane. Final Top-K still uses ``final_min_points``/``fifth_min_points`` or ``rank_min_points``.
        enable_rescue_candidate_pool: If True, use a second, weaker real-query candidate pool only when the
            normal pool has fewer lanes than Count Head K.
        rescue_candidate_score_thr: Optional rescue-pool existence threshold. If omitted, candidate_score_thr is used.
        rescue_candidate_point_valid_thr: Optional rescue-pool point-valid threshold. If omitted,
            candidate_point_valid_thr is used.
        rescue_candidate_min_points: Optional rescue-pool visible-anchor floor. If omitted, candidate_min_points is used.
        final_min_points: Final visible-anchor floor for selected ranks without an override.
        fifth_min_points: Optional final floor for selected rank 5. With Count Head decode enabled, the implicit
            default is ``min(final_min_points, 5)``.
        quality_rescue_5th: With quality logits and Count Head policy K=5, hold the stable base selection at
            4 lanes and add the fifth only if Count Head P(5) and candidate quality gates pass. When
            ``last_lane_rescue`` is also enabled, K=5 shortfalls try last-lane rescue first and fall back to this
            quality-gated rescue if the output is still exactly 4 lanes.
        last_lane_rescue: If True and Count Head policy K is at least ``last_lane_rescue_min_policy_count``,
            fill final Top-K shortfalls from a separate weak candidate pool without lowering the global
            candidate point-valid threshold.
        last_lane_rescue_point_valid_thr: Point-valid threshold used only to build last-lane rescue candidates.
        edge_last_lane_rescue: If True, final shortfalls for K>=edge_rescue_min_policy_count first try
            to add a real outside-left/right edge candidate before ordinary last-lane refill.
        edge_count4_to5_upgrade: If True, Count Head raw K=4 can become an
            effective K=5 only when P4-P5 is small and a valid outside-edge candidate exists. This
            upgrade can activate edge rescue for the current decode even if ``edge_last_lane_rescue``
            was disabled by the caller. Ordinary ``last_lane_rescue`` also tries this edge branch first.
        enable_soft_count_decision: If True, adjacent close Count Head probabilities are re-scored by candidate
            quality, duplicate risk, and invalid/short-candidate penalties before final K selection.
        return_meta: If True, return ``(lanes, decode_meta)`` for sweep/diagnostic aggregation.

    Returns:
        A left-to-right list of selected lane dictionaries with raw exist score, visibility-aware rank score,
        query index, normalized points, and optional pixel points. Rank score is used for filtering, NMS,
        Top-K lane selection, and the max_det cap before the final output ordering step.
    """
    if pred_logits.ndim == 2 and pred_logits.shape[-1] == 1:
        pred_logits = pred_logits.squeeze(-1)
    _ = _normalize_count_calibration_config(count_calibration)
    if pred_points.ndim != 3 or pred_points.shape[-1] != 2:
        raise ValueError(f"pred_points must have shape Q x K x 2, got {tuple(pred_points.shape)}.")
    if pred_logits.ndim != 1 or pred_logits.shape[0] != pred_points.shape[0]:
        raise ValueError(
            f"pred_logits must have shape Q and match pred_points Q, got {tuple(pred_logits.shape)} "
            f"vs {tuple(pred_points.shape)}."
        )
    if pred_valid_logits is not None:
        if pred_valid_logits.ndim == 3 and pred_valid_logits.shape[-1] == 1:
            pred_valid_logits = pred_valid_logits.squeeze(-1)
        if pred_valid_logits.shape != pred_points.shape[:2]:
            raise ValueError(
                "pred_valid_logits must have shape Q x K matching pred_points, "
                f"got {tuple(pred_valid_logits.shape)} vs {tuple(pred_points.shape[:2])}."
            )
    if pred_quality_logits is not None:
        if pred_quality_logits.ndim == 2 and pred_quality_logits.shape[-1] == 1:
            pred_quality_logits = pred_quality_logits.squeeze(-1)
        if pred_quality_logits.ndim != 1 or pred_quality_logits.shape[0] != pred_points.shape[0]:
            raise ValueError(
                "pred_quality_logits must have shape Q matching pred_points, "
                f"got {tuple(pred_quality_logits.shape)} vs Q={pred_points.shape[0]}."
            )

    points = pred_points.detach().float().cpu().clamp(0.0, 1.0)
    scores = pred_logits.detach().float().cpu().sigmoid()
    point_valid_scores = pred_valid_logits.detach().float().cpu().sigmoid() if pred_valid_logits is not None else None
    quality_scores = pred_quality_logits.detach().float().cpu().sigmoid() if pred_quality_logits is not None else None
    query_indices = torch.arange(points.shape[0], dtype=torch.long)
    rank_min_points_cfg = _normalize_rank_min_points(rank_min_points)
    count_head_active = bool(use_count_head_decode)
    count_head_meta = count_head_decode_meta(
        pred_count_logits,
        pred_count_boundary_logits,
        use_count_head_decode=count_head_active,
        count_head_temperature=float(count_head_temperature),
        dataset_name=dataset_name,
        count_head_min_count=count_head_min_count,
        count_head_max_count=count_head_max_count,
        merge_tusimple_2_to_3=merge_tusimple_2_to_3,
        max_det=max_det,
    )
    candidate_score_thr_i = float(score_thr if candidate_score_thr is None else candidate_score_thr)
    candidate_point_valid_thr_i = float(
        point_valid_thr if candidate_point_valid_thr is None else candidate_point_valid_thr
    )
    line_nms_min_overlap_i = max(int(line_nms_min_overlap), 1)
    if candidate_score_thr_i < 0.0:
        raise ValueError(f"candidate_score_thr must be >= 0, got {candidate_score_thr_i}.")
    if not (0.0 <= candidate_point_valid_thr_i <= 1.0):
        raise ValueError(
            f"candidate_point_valid_thr must be in [0, 1], got {candidate_point_valid_thr_i}."
        )
    final_min_points_i = int(min_points if final_min_points is None else final_min_points)
    if final_min_points_i <= 0:
        raise ValueError(f"final_min_points must be > 0, got {final_min_points_i}.")
    if fifth_min_points is None and count_head_active:
        fifth_min_points_i = min(final_min_points_i, 5)
    else:
        fifth_min_points_i = None if fifth_min_points is None else int(fifth_min_points)
    if fifth_min_points_i is not None and fifth_min_points_i <= 0:
        raise ValueError(f"fifth_min_points must be > 0, got {fifth_min_points_i}.")
    if candidate_min_points is None and count_head_active:
        candidate_min_points_i = min(final_min_points_i, 5)
    else:
        candidate_min_points_i = int(final_min_points_i if candidate_min_points is None else candidate_min_points)
    if rank_min_points_cfg is not None:
        candidate_min_points_i = min([candidate_min_points_i, *rank_min_points_cfg.values()])
    if fifth_min_points_i is not None:
        candidate_min_points_i = min(candidate_min_points_i, fifth_min_points_i)

    if candidate_min_points_i <= 0:
        raise ValueError(f"candidate_min_points must be > 0, got {candidate_min_points_i}.")
    rescue_candidate_score_thr_i = float(
        candidate_score_thr_i if rescue_candidate_score_thr is None else rescue_candidate_score_thr
    )
    rescue_candidate_point_valid_thr_i = float(
        candidate_point_valid_thr_i
        if rescue_candidate_point_valid_thr is None
        else rescue_candidate_point_valid_thr
    )
    rescue_candidate_min_points_i = int(
        candidate_min_points_i if rescue_candidate_min_points is None else rescue_candidate_min_points
    )
    if rescue_candidate_score_thr_i < 0.0:
        raise ValueError(f"rescue_candidate_score_thr must be >= 0, got {rescue_candidate_score_thr_i}.")
    if not (0.0 <= rescue_candidate_point_valid_thr_i <= 1.0):
        raise ValueError(
            "rescue_candidate_point_valid_thr must be in [0, 1], "
            f"got {rescue_candidate_point_valid_thr_i}."
        )
    if rescue_candidate_min_points_i <= 0:
        raise ValueError(f"rescue_candidate_min_points must be > 0, got {rescue_candidate_min_points_i}.")
    if not (0.0 <= float(quality_rescue_count5_thr) <= 1.0):
        raise ValueError(f"quality_rescue_count5_thr must be in [0, 1], got {quality_rescue_count5_thr}.")
    if not (0.0 <= float(quality_rescue_conf_thr) <= 1.0):
        raise ValueError(f"quality_rescue_conf_thr must be in [0, 1], got {quality_rescue_conf_thr}.")
    if not (0.0 <= float(quality_rescue_mean_valid_thr) <= 1.0):
        raise ValueError(f"quality_rescue_mean_valid_thr must be in [0, 1], got {quality_rescue_mean_valid_thr}.")
    if not (0.0 <= float(quality_rescue_quality_thr) <= 1.0):
        raise ValueError(f"quality_rescue_quality_thr must be in [0, 1], got {quality_rescue_quality_thr}.")
    if int(quality_rescue_min_points) <= 0:
        raise ValueError(f"quality_rescue_min_points must be > 0, got {quality_rescue_min_points}.")
    if float(quality_rescue_dist_px) < 0.0:
        raise ValueError(f"quality_rescue_dist_px must be >= 0, got {quality_rescue_dist_px}.")
    last_lane_rescue_min_policy_count_i = max(int(last_lane_rescue_min_policy_count), 1)
    last_lane_rescue_conf_thr_i = float(
        rescue_candidate_score_thr_i if last_lane_rescue_conf_thr is None else last_lane_rescue_conf_thr
    )
    last_lane_rescue_point_valid_thr_i = float(
        rescue_candidate_point_valid_thr_i
        if last_lane_rescue_point_valid_thr is None
        else last_lane_rescue_point_valid_thr
    )
    last_lane_rescue_min_points_i = int(
        rescue_candidate_min_points_i if last_lane_rescue_min_points is None else last_lane_rescue_min_points
    )
    if not (0.0 <= last_lane_rescue_conf_thr_i <= 1.0):
        raise ValueError(f"last_lane_rescue_conf_thr must be in [0, 1], got {last_lane_rescue_conf_thr_i}.")
    if not (0.0 <= last_lane_rescue_point_valid_thr_i <= 1.0):
        raise ValueError(
            "last_lane_rescue_point_valid_thr must be in [0, 1], "
            f"got {last_lane_rescue_point_valid_thr_i}."
        )
    if last_lane_rescue_min_points_i <= 0:
        raise ValueError(f"last_lane_rescue_min_points must be > 0, got {last_lane_rescue_min_points_i}.")
    if not (0.0 <= float(last_lane_rescue_mean_valid_thr) <= 1.0):
        raise ValueError(f"last_lane_rescue_mean_valid_thr must be in [0, 1], got {last_lane_rescue_mean_valid_thr}.")
    if not (0.0 <= float(last_lane_rescue_quality_thr) <= 1.0):
        raise ValueError(f"last_lane_rescue_quality_thr must be in [0, 1], got {last_lane_rescue_quality_thr}.")
    if float(last_lane_rescue_dist_px) < 0.0:
        raise ValueError(f"last_lane_rescue_dist_px must be >= 0, got {last_lane_rescue_dist_px}.")
    edge_rescue_min_policy_count_i = max(int(edge_rescue_min_policy_count), 1)
    if not (0.0 <= float(edge_rescue_conf_thr) <= 1.0):
        raise ValueError(f"edge_rescue_conf_thr must be in [0, 1], got {edge_rescue_conf_thr}.")
    if not (0.0 <= float(edge_rescue_point_valid_thr) <= 1.0):
        raise ValueError(f"edge_rescue_point_valid_thr must be in [0, 1], got {edge_rescue_point_valid_thr}.")
    if int(edge_rescue_min_points) <= 0:
        raise ValueError(f"edge_rescue_min_points must be > 0, got {edge_rescue_min_points}.")
    if not (0.0 <= float(edge_rescue_mean_valid_thr) <= 1.0):
        raise ValueError(f"edge_rescue_mean_valid_thr must be in [0, 1], got {edge_rescue_mean_valid_thr}.")
    if not (0.0 <= float(edge_rescue_quality_thr) <= 1.0):
        raise ValueError(f"edge_rescue_quality_thr must be in [0, 1], got {edge_rescue_quality_thr}.")
    if float(edge_rescue_outside_gap_px) < 0.0:
        raise ValueError(f"edge_rescue_outside_gap_px must be >= 0, got {edge_rescue_outside_gap_px}.")
    if float(edge_rescue_dist_px) < 0.0:
        raise ValueError(f"edge_rescue_dist_px must be >= 0, got {edge_rescue_dist_px}.")
    if float(edge_count4_to5_prob_margin) < 0.0:
        raise ValueError(f"edge_count4_to5_prob_margin must be >= 0, got {edge_count4_to5_prob_margin}.")
    edge_last_lane_rescue_requested = bool(edge_last_lane_rescue)
    edge_last_lane_rescue_active = edge_last_lane_rescue_requested or bool(last_lane_rescue)

    policy_target_count = int(max_det) if max_det is not None and int(max_det) > 0 else int(points.shape[0])
    if count_head_active:
        policy_target_count = int(count_head_meta["count_head_policy_count"])
    policy_target_count = max(0, int(policy_target_count))
    max_allowed_count = int(max_det) if max_det is not None and int(max_det) > 0 else int(count_head_max_count)
    count5_prob_for_rescue = float(count_head_meta.get("count5_prob", 0.0)) if count_head_meta else 0.0
    quality_count5_upgrade_eligible = bool(
        quality_scores is not None
        and quality_rescue_5th
        and count_head_active
        and count_head_meta
        and policy_target_count == 4
        and max_allowed_count >= 5
        and count5_prob_for_rescue >= float(quality_rescue_count5_thr)
    )
    candidate_target_count = 5 if quality_count5_upgrade_eligible else policy_target_count

    decode_meta = {
        "candidate_count_raw": int(points.shape[0]),
        "candidate_count_normal": 0,
        "candidate_count_rescue": 0,
        "candidate_count_after_rescue": 0,
        "effective_policy_count": int(policy_target_count),
        "candidate_target_count": int(candidate_target_count),
        "quality_count5_upgrade_eligible": bool(quality_count5_upgrade_eligible),
        "quality_count5_upgrade_success": False,
        "candidate_pool_shortfall": int(max(0, candidate_target_count)),
        "candidate_pool_shortfall_before_rescue": int(max(0, candidate_target_count)),
        "candidate_pool_shortfall_after_rescue": int(max(0, candidate_target_count)),
        "normal_candidate_min_points": int(candidate_min_points_i),
        "rescue_candidate_min_points": int(rescue_candidate_min_points_i),
        "normal_candidate_score_thr": float(candidate_score_thr_i),
        "rescue_candidate_score_thr": float(rescue_candidate_score_thr_i),
        "normal_candidate_point_valid_thr": float(candidate_point_valid_thr_i),
        "rescue_candidate_point_valid_thr": float(rescue_candidate_point_valid_thr_i),
        "enable_rescue_candidate_pool": bool(enable_rescue_candidate_pool),
        "top5_candidate_exists_before_nms": False,
        "top5_candidate_index_before_nms": None,
        "top5_candidate_score_before_nms": None,
        "top5_candidate_quality_before_nms": None,
        "top5_candidate_valid_points_before_nms": None,
        "top5_suppressed_by_nms": False,
        "candidate_count_after_nms": 0,
        "nms_suppressed_count": 0,
        "quality_rank_active": True,
        "quality_rank_source": "exist_visibility",
        "quality_head_available": bool(quality_scores is not None),
        "quality_rescue_5th_enabled": bool(quality_scores is not None and quality_rescue_5th),
        "last_lane_rescue_enabled": bool(last_lane_rescue),
        "last_lane_rescue_min_policy_count": int(last_lane_rescue_min_policy_count_i),
        "last_lane_rescue_score_thr": float(last_lane_rescue_conf_thr_i),
        "last_lane_rescue_point_valid_thr": float(last_lane_rescue_point_valid_thr_i),
        "last_lane_rescue_min_points": int(last_lane_rescue_min_points_i),
        "last_lane_rescue_mean_valid_thr": float(last_lane_rescue_mean_valid_thr),
        "last_lane_rescue_quality_thr": float(last_lane_rescue_quality_thr),
        "last_lane_rescue_dist_px": float(last_lane_rescue_dist_px),
        "edge_last_lane_rescue_enabled": bool(edge_last_lane_rescue_active),
        "edge_last_lane_rescue_requested": bool(edge_last_lane_rescue_requested),
        "edge_last_lane_rescue_active": bool(edge_last_lane_rescue_active),
        "edge_rescue_score_thr": float(edge_rescue_conf_thr),
        "edge_rescue_point_valid_thr": float(edge_rescue_point_valid_thr),
        "edge_rescue_min_points": int(edge_rescue_min_points),
        "edge_rescue_mean_valid_thr": float(edge_rescue_mean_valid_thr),
        "edge_rescue_quality_thr": float(edge_rescue_quality_thr),
        "edge_rescue_outside_gap_px": float(edge_rescue_outside_gap_px),
        "edge_rescue_dist_px": float(edge_rescue_dist_px),
        "edge_rescue_min_policy_count": int(edge_rescue_min_policy_count_i),
        "edge_count4_to5_upgrade_enabled": bool(edge_count4_to5_upgrade),
        "edge_count4_to5_upgrade": False,
        "edge_count4_to5_upgrade_reason": "not_attempted",
        "edge_count4_to5_prob_margin": float(edge_count4_to5_prob_margin),
        "edge_count4_to5_p4": None,
        "edge_count4_to5_p5": None,
        "edge_count4_to5_candidate_side": None,
        "edge_count4_to5_candidate_valid_points": None,
        "edge_count4_to5_candidate_quality": None,
        "edge_count4_to5_candidate_min_dist": None,
        "edge_last_lane_rescue_attempt_count": 0,
        "edge_last_lane_rescue_success_count": 0,
        "edge_last_lane_rescue_reason": "not_attempted",
        "edge_last_lane_rescue_candidate_quality": None,
        "edge_last_lane_rescue_candidate_valid_points": None,
        "edge_last_lane_rescue_candidate_min_dist": None,
        "edge_last_lane_rescue_candidate_side": None,
        "edge_last_lane_rescue_candidate_outside_gap_px": None,
        "soft_count_decision_enabled": bool(enable_soft_count_decision),
        "pred_count_cls_raw": None,
        "pred_count_cls_soft": None,
        "soft_count_changed": False,
        "soft_count_score_by_k": {},
        "quality_rescue_attempt_count": 0,
        "quality_rescue_success_count": 0,
        "quality_rescue_fallback_after_last_lane": False,
        "rescue_attempted": False,
        "rescue_success": False,
        "rescue_reason": "not_attempted",
        "rescue_candidate_quality": None,
        "rescue_candidate_valid_points": None,
        "rescue_candidate_min_dist": None,
        "last_lane_rescue_attempt_count": 0,
        "last_lane_rescue_success_count": 0,
        "last_lane_rescue_reason": "not_attempted",
        "last_lane_rescue_candidate_quality": None,
        "last_lane_rescue_candidate_valid_points": None,
        "last_lane_rescue_candidate_min_dist": None,
    }
    if count_head_meta is not None:
        decode_meta.update(count_head_meta)

    if candidate_min_points_i > points.shape[1]:
        return ([], decode_meta) if return_meta else []
    if rescue_candidate_min_points_i > points.shape[1]:
        rescue_candidate_min_points_i = int(points.shape[1])
        decode_meta["rescue_candidate_min_points"] = int(rescue_candidate_min_points_i)

    normal_lanes = _build_lane_candidates(
        points,
        scores,
        point_valid_scores,
        quality_scores,
        query_indices,
        image_shape,
        score_thr=candidate_score_thr_i,
        point_valid_thr=candidate_point_valid_thr_i,
        min_points=candidate_min_points_i,
    )
    rescue_lanes = (
        _build_lane_candidates(
            points,
            scores,
            point_valid_scores,
            quality_scores,
            query_indices,
            image_shape,
            score_thr=rescue_candidate_score_thr_i,
            point_valid_thr=rescue_candidate_point_valid_thr_i,
            min_points=rescue_candidate_min_points_i,
        )
        if bool(enable_rescue_candidate_pool)
        else []
    )
    last_lane_lanes = (
        _build_lane_candidates(
            points,
            scores,
            point_valid_scores,
            quality_scores,
            query_indices,
            image_shape,
            score_thr=last_lane_rescue_conf_thr_i,
            point_valid_thr=last_lane_rescue_point_valid_thr_i,
            min_points=last_lane_rescue_min_points_i,
        )
        if bool(last_lane_rescue)
        else []
    )
    edge_lanes = (
        _build_lane_candidates(
            points,
            scores,
            point_valid_scores,
            quality_scores,
            query_indices,
            image_shape,
            score_thr=float(edge_rescue_conf_thr),
            point_valid_thr=float(edge_rescue_point_valid_thr),
            min_points=int(edge_rescue_min_points),
        )
        if edge_last_lane_rescue_active or (bool(edge_count4_to5_upgrade) and count_head_active)
        else []
    )
    rescue_dist = float(nms_dist_px if line_nms_rescue_dist_px is None else line_nms_rescue_dist_px)
    lanes = list(normal_lanes)
    candidate_pool_shortfall_before = max(0, candidate_target_count - len(lanes))
    if bool(enable_rescue_candidate_pool) and candidate_pool_shortfall_before > 0:
        lanes = _select_rescue_candidates(
            lanes,
            rescue_lanes,
            target_count=candidate_target_count,
            image_shape=image_shape,
            rescue_dist_px=rescue_dist,
            min_overlap=line_nms_min_overlap_i,
        )
    lanes.sort(key=_lane_rank_score, reverse=True)

    decode_meta.update(
        {
            "candidate_count_normal": int(len(normal_lanes)),
            "candidate_count_rescue": int(len(rescue_lanes)),
            "candidate_count_last_lane": int(len(last_lane_lanes)),
            "candidate_count_edge_rescue": int(len(edge_lanes)),
            "candidate_count_after_rescue": int(len(lanes)),
            "candidate_pool_shortfall_before_rescue": int(candidate_pool_shortfall_before),
            "candidate_pool_shortfall_after_rescue": int(max(0, candidate_target_count - len(lanes))),
            "candidate_pool_shortfall": int(max(0, candidate_target_count - len(lanes))),
        }
    )
    if not lanes or policy_target_count <= 0:
        return ([], decode_meta) if return_meta else []

    pre_nms_lanes = list(lanes)
    top5_query = None
    if len(pre_nms_lanes) >= 5:
        top5 = pre_nms_lanes[4]
        top5_query = int(top5.get("query", -1))
        decode_meta.update(
            {
                "top5_candidate_exists_before_nms": True,
                "top5_candidate_index_before_nms": top5_query,
                "top5_candidate_score_before_nms": float(_lane_rank_score(top5)),
                "top5_candidate_quality_before_nms": (
                    None if top5.get("quality_score") is None else float(top5.get("quality_score"))
                ),
                "top5_candidate_valid_points_before_nms": int(top5.get("valid_count", 0)),
            }
        )
    suppressed_info: list[dict] = []
    suppressed_lanes: list[dict] = []

    if nms_dist_px > 0.0:
        if image_shape is None:
            raise ValueError("decode_gcs_predictions requires image_shape when nms_dist_px > 0.")
        nms_points = torch.from_numpy(np.stack([x["points_norm"] for x in lanes], axis=0).astype(np.float32))
        nms_scores = torch.tensor([_lane_rank_score(x) for x in lanes], dtype=torch.float32)
        nms_exist_scores = torch.tensor([float(x.get("exist_score", x.get("score", 0.0))) for x in lanes], dtype=torch.float32)
        nms_valid_masks = None
        nms_point_valid_scores = None
        if point_valid_scores is not None:
            nms_valid_masks = torch.from_numpy(np.stack([x["point_valid"] for x in lanes], axis=0).astype(bool))
            nms_point_valid_scores = torch.from_numpy(
                np.stack([x["point_valid_scores"] for x in lanes], axis=0).astype(np.float32)
            )
        keep_nms, suppressed_info = lane_nms(
            nms_points,
            nms_scores,
            image_shape=image_shape,
            dist_thr_px=float(nms_dist_px),
            valid_masks=nms_valid_masks,
            point_valid_scores=nms_point_valid_scores,
            exist_scores=nms_exist_scores,
            min_overlap=line_nms_min_overlap_i,
            return_suppressed=True,
        )
        suppressed_indices = {int(x["index"]) for x in suppressed_info}
        suppressed_queries = {int(lanes[i].get("query", -1)) for i in suppressed_indices}
        decode_meta["top5_suppressed_by_nms"] = bool(top5_query is not None and top5_query in suppressed_queries)
        for item in suppressed_info:
            lane = dict(lanes[int(item["index"])])
            lane["nms_suppressed"] = True
            lane["nms_suppressed_by_index"] = item.get("suppressed_by_index")
            lane["nms_suppress_reason"] = item.get("suppress_reason")
            lane["nms_distance_to_suppressor"] = item.get("distance_to_suppressor")
            suppressed_lanes.append(lane)
        lanes = [lanes[int(i)] for i in keep_nms.tolist()]

    lanes.sort(key=_lane_rank_score, reverse=True)
    decode_meta["candidate_count_after_nms"] = int(len(lanes))
    decode_meta["nms_suppressed_count"] = int(len(suppressed_info))
    if count_head_active:
        count_head_meta = {
            **count_head_meta,
            "count_head_candidate_score_thr": float(candidate_score_thr_i),
            "count_head_candidate_point_valid_thr": float(candidate_point_valid_thr_i),
            "count_head_rescue_candidate_score_thr": float(rescue_candidate_score_thr_i),
            "count_head_rescue_candidate_point_valid_thr": float(rescue_candidate_point_valid_thr_i),
            "line_nms_min_overlap": int(line_nms_min_overlap_i),
        }
    if policy_target_count <= 0:
        return ([], decode_meta) if return_meta else []
    if bool(enable_soft_count_decision) and count_head_active and count_head_meta is not None:
        soft_meta = soft_count_decision(
            count_head_meta.get("count_head_prob", [0.0, 0.0, 0.0, 0.0]),
            pre_nms_lanes,
            image_shape=image_shape,
            max_count=max_allowed_count,
            prob_margin=float(soft_count_prob_margin),
            quality_weight=float(soft_count_quality_weight),
            prior_weight=float(soft_count_prior_weight),
            duplicate_penalty=float(soft_count_duplicate_penalty),
            invalid_penalty=float(soft_count_invalid_penalty),
            min_points=int(candidate_min_points_i),
            duplicate_dist_px=float(nms_dist_px if nms_dist_px > 0.0 else 18.0),
            min_overlap=line_nms_min_overlap_i,
        )
        soft_count = int(soft_meta["pred_count_cls_soft"])
        policy_target_count = max(0, min(int(soft_count), int(max_allowed_count)))
        count_head_meta["count_head_policy_count"] = int(policy_target_count)
        count_head_meta["effective_policy_count"] = int(policy_target_count)
        decode_meta.update(soft_meta)
        decode_meta["effective_policy_count"] = int(policy_target_count)
    if (
        bool(edge_count4_to5_upgrade)
        and count_head_active
        and count_head_meta is not None
        and int(policy_target_count) == 4
        and int(max_allowed_count) >= 5
    ):
        edge_upgrade_base = _select_topk_with_final_constraints(
            lanes,
            target_count=4,
            default_min_points=final_min_points_i,
            rank_min_points=rank_min_points_cfg,
            fifth_min_points=fifth_min_points_i,
        )
        edge_upgrade_ok, edge_upgrade_meta = _edge_count4_to5_upgrade_eligible(
            selected=edge_upgrade_base,
            rescue_candidates=edge_lanes,
            count_head_meta=count_head_meta,
            image_shape=image_shape,
            enabled=True,
            prob_margin=float(edge_count4_to5_prob_margin),
            edge_conf_thr=float(edge_rescue_conf_thr),
            edge_mean_valid_thr=float(edge_rescue_mean_valid_thr),
            edge_quality_thr=float(edge_rescue_quality_thr),
            edge_min_points=int(edge_rescue_min_points),
            edge_outside_gap_px=float(edge_rescue_outside_gap_px),
            edge_dist_px=float(edge_rescue_dist_px),
            min_overlap=line_nms_min_overlap_i,
        )
        decode_meta.update(edge_upgrade_meta)
        if edge_upgrade_ok:
            policy_target_count = 5
            edge_last_lane_rescue_active = True
            decode_meta["effective_policy_count"] = 5
            decode_meta["candidate_target_count"] = max(int(decode_meta.get("candidate_target_count", 0)), 5)
            decode_meta["edge_last_lane_rescue_active"] = True
            if count_head_meta is not None:
                count_head_meta["effective_policy_count"] = 5
    quality_rescue_available = bool(
        quality_scores is not None
        and quality_rescue_5th
        and count_head_active
        and count_head_meta
    )
    quality_rescue_active = bool(
        quality_rescue_available
        and ((policy_target_count == 5 and not (bool(last_lane_rescue) or edge_last_lane_rescue_active)) or quality_count5_upgrade_eligible)
    )
    base_target_count = 4 if quality_rescue_active else policy_target_count
    lanes = _select_topk_with_final_constraints(
        lanes,
        target_count=base_target_count,
        default_min_points=final_min_points_i,
        rank_min_points=rank_min_points_cfg,
        fifth_min_points=fifth_min_points_i,
    )
    if len(lanes) < base_target_count:
        if edge_last_lane_rescue_active and base_target_count >= edge_rescue_min_policy_count_i:
            lanes, edge_lane_meta = _edge_last_lane_rescue(
                selected=lanes,
                rescue_candidates=edge_lanes,
                target_count=base_target_count,
                image_shape=image_shape,
                min_policy_count=edge_rescue_min_policy_count_i,
                edge_conf_thr=float(edge_rescue_conf_thr),
                edge_mean_valid_thr=float(edge_rescue_mean_valid_thr),
                edge_quality_thr=float(edge_rescue_quality_thr),
                edge_min_points=int(edge_rescue_min_points),
                edge_outside_gap_px=float(edge_rescue_outside_gap_px),
                edge_dist_px=float(edge_rescue_dist_px),
                min_overlap=line_nms_min_overlap_i,
            )
            decode_meta.update(edge_lane_meta)
        if len(lanes) < base_target_count and bool(last_lane_rescue) and base_target_count >= last_lane_rescue_min_policy_count_i:
            lanes, last_lane_meta = _last_required_lane_rescue(
                selected=lanes,
                rescue_candidates=last_lane_lanes,
                target_count=base_target_count,
                image_shape=image_shape,
                min_policy_count=last_lane_rescue_min_policy_count_i,
                rescue_conf_thr=last_lane_rescue_conf_thr_i,
                rescue_mean_valid_thr=float(last_lane_rescue_mean_valid_thr),
                rescue_quality_thr=float(last_lane_rescue_quality_thr),
                rescue_min_points=last_lane_rescue_min_points_i,
                rescue_dist_px=float(last_lane_rescue_dist_px),
                min_overlap=line_nms_min_overlap_i,
            )
            decode_meta.update(last_lane_meta)
        if len(lanes) < base_target_count and not bool(last_lane_rescue):
            lanes = count_aware_refill(
                selected=lanes,
                pre_nms_candidates=suppressed_lanes if nms_dist_px > 0.0 else pre_nms_lanes,
                target_count=base_target_count,
                image_shape=image_shape,
                default_min_points=final_min_points_i,
                rank_min_points=rank_min_points_cfg,
                fifth_min_points=fifth_min_points_i,
                rescue_dist_px=rescue_dist,
                min_overlap=line_nms_min_overlap_i,
            )
    if quality_rescue_active:
        decode_meta["quality_rescue_attempt_count"] = int(len(lanes) == 4)
        before_quality_rescue = len(lanes)
        lanes, rescue_meta = _quality_gated_rescue_5th(
            lanes,
            pre_nms_candidates=pre_nms_lanes,
            count_head_meta=count_head_meta,
            image_shape=image_shape,
            count5_thr=float(quality_rescue_count5_thr),
            rescue_conf_thr=float(quality_rescue_conf_thr),
            rescue_mean_valid_thr=float(quality_rescue_mean_valid_thr),
            rescue_quality_thr=float(quality_rescue_quality_thr),
            rescue_min_points=int(quality_rescue_min_points),
            rescue_dist_px=float(quality_rescue_dist_px),
            min_overlap=line_nms_min_overlap_i,
        )
        decode_meta["quality_rescue_success_count"] = int(len(lanes) > before_quality_rescue)
        decode_meta.update(rescue_meta)
        if len(lanes) > before_quality_rescue:
            decode_meta["effective_policy_count"] = 5
            decode_meta["quality_count5_upgrade_success"] = bool(quality_count5_upgrade_eligible)
    elif len(lanes) < policy_target_count:
        if edge_last_lane_rescue_active and policy_target_count >= edge_rescue_min_policy_count_i:
            lanes, edge_lane_meta = _edge_last_lane_rescue(
                selected=lanes,
                rescue_candidates=edge_lanes,
                target_count=policy_target_count,
                image_shape=image_shape,
                min_policy_count=edge_rescue_min_policy_count_i,
                edge_conf_thr=float(edge_rescue_conf_thr),
                edge_mean_valid_thr=float(edge_rescue_mean_valid_thr),
                edge_quality_thr=float(edge_rescue_quality_thr),
                edge_min_points=int(edge_rescue_min_points),
                edge_outside_gap_px=float(edge_rescue_outside_gap_px),
                edge_dist_px=float(edge_rescue_dist_px),
                min_overlap=line_nms_min_overlap_i,
            )
            decode_meta.update(edge_lane_meta)
        if len(lanes) < policy_target_count and bool(last_lane_rescue) and policy_target_count >= last_lane_rescue_min_policy_count_i:
            lanes, last_lane_meta = _last_required_lane_rescue(
                selected=lanes,
                rescue_candidates=last_lane_lanes,
                target_count=policy_target_count,
                image_shape=image_shape,
                min_policy_count=last_lane_rescue_min_policy_count_i,
                rescue_conf_thr=last_lane_rescue_conf_thr_i,
                rescue_mean_valid_thr=float(last_lane_rescue_mean_valid_thr),
                rescue_quality_thr=float(last_lane_rescue_quality_thr),
                rescue_min_points=last_lane_rescue_min_points_i,
                rescue_dist_px=float(last_lane_rescue_dist_px),
                min_overlap=line_nms_min_overlap_i,
            )
            decode_meta.update(last_lane_meta)
        if len(lanes) < policy_target_count and not bool(last_lane_rescue):
            lanes = count_aware_refill(
                selected=lanes,
                pre_nms_candidates=suppressed_lanes if nms_dist_px > 0.0 else pre_nms_lanes,
                target_count=policy_target_count,
                image_shape=image_shape,
                default_min_points=final_min_points_i,
                rank_min_points=rank_min_points_cfg,
                fifth_min_points=fifth_min_points_i,
                rescue_dist_px=rescue_dist,
                min_overlap=line_nms_min_overlap_i,
            )
    quality_rescue_fallback_active = bool(
        quality_rescue_available
        and (bool(last_lane_rescue) or edge_last_lane_rescue_active)
        and not quality_rescue_active
        and policy_target_count == 5
        and len(lanes) == 4
    )
    if quality_rescue_fallback_active:
        before_quality_rescue = len(lanes)
        lanes, rescue_meta = _quality_gated_rescue_5th(
            lanes,
            pre_nms_candidates=pre_nms_lanes,
            count_head_meta=count_head_meta,
            image_shape=image_shape,
            count5_thr=float(quality_rescue_count5_thr),
            rescue_conf_thr=float(quality_rescue_conf_thr),
            rescue_mean_valid_thr=float(quality_rescue_mean_valid_thr),
            rescue_quality_thr=float(quality_rescue_quality_thr),
            rescue_min_points=int(quality_rescue_min_points),
            rescue_dist_px=float(quality_rescue_dist_px),
            min_overlap=line_nms_min_overlap_i,
        )
        decode_meta["quality_rescue_attempt_count"] = int(decode_meta.get("quality_rescue_attempt_count", 0)) + int(
            before_quality_rescue == 4
        )
        decode_meta["quality_rescue_success_count"] = int(decode_meta.get("quality_rescue_success_count", 0)) + int(
            len(lanes) > before_quality_rescue
        )
        decode_meta.update(rescue_meta)
        decode_meta["quality_rescue_fallback_after_last_lane"] = True
        if len(lanes) > before_quality_rescue:
            decode_meta["effective_policy_count"] = 5
    decode_meta["edge_last_lane_rescue_active"] = bool(edge_last_lane_rescue_active)
    effective_policy_count = int(decode_meta.get("effective_policy_count", policy_target_count))
    decode_meta["count_head_shortfall"] = int(max(0, effective_policy_count - len(lanes)))
    if count_head_meta is not None:
        for lane in lanes:
            lane.update(count_head_meta)
            lane["count_head_shortfall"] = int(decode_meta["count_head_shortfall"])
            lane.update(decode_meta)
    lanes.sort(key=_lane_bottom_x)
    return (lanes, decode_meta) if return_meta else lanes


def draw_gcs_lanes(
    image: np.ndarray,
    lanes: list[dict],
    show_scores: bool = True,
    line_width: int = 2,
    point_radius: int = 3,
) -> np.ndarray:
    """Draw decoded GCS lane point sequences on a BGR image."""
    out = image.copy()
    h, w = out.shape[:2]
    for i, lane in enumerate(lanes):
        pts = lane.get("visible_points", lane.get("points"))
        if pts is None:
            pts_norm = lane.get("visible_points_norm", lane["points_norm"])
            pts = np.asarray(pts_norm, dtype=np.float32) * np.array([w, h], dtype=np.float32)
        pts = np.asarray(pts, dtype=np.float32)
        if "point_valid" in lane and "visible_points" not in lane and "visible_points_norm" not in lane:
            valid = np.asarray(lane["point_valid"], dtype=np.float32) > 0.5
            if valid.shape[0] == pts.shape[0]:
                pts = pts[valid]
        if pts.shape[0] < 2:
            continue

        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts_i = np.round(pts).astype(np.int32)
        color = GCS_LANE_COLORS[i % len(GCS_LANE_COLORS)]

        cv2.polylines(out, [pts_i], isClosed=False, color=color, thickness=line_width, lineType=cv2.LINE_AA)
        for x, y in pts_i:
            cv2.circle(out, (int(x), int(y)), point_radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        if show_scores:
            x0, y0 = pts_i[0]
            cv2.putText(
                out,
                f"{lane['score']:.2f}",
                (int(x0), max(12, int(y0) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return out


def save_gcs_lanes_txt(path: str | Path, lanes: list[dict], save_conf: bool = True) -> str:
    """Save decoded GCS lanes as normalized point sequences, one lane per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for lane in lanes:
        pts = np.asarray(lane.get("visible_points_norm", lane["points_norm"]), dtype=np.float32)
        if "point_valid" in lane and "visible_points_norm" not in lane:
            valid = np.asarray(lane["point_valid"], dtype=np.float32) > 0.5
            if valid.shape[0] == pts.shape[0]:
                pts = pts[valid]
        pts = pts.reshape(-1)
        values: list[float | int] = [int(lane.get("query", -1))]
        if save_conf:
            values.append(float(lane["score"]))
        values.extend(float(x) for x in pts)
        lines.append(" ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in values))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return str(path)
