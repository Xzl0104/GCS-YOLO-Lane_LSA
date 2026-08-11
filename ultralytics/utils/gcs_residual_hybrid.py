"""Prediction-only residual hybrid replacement helpers for controlled diagnostics."""

from __future__ import annotations

import math

import torch


REQUIRED_KEYS = (
    "pred_points",
    "pred_logits",
    "pred_valid_logits",
    "pred_residual_points",
    "pred_residual_valid_logits",
    "pred_residual_exist_logits",
    "pred_residual_row_logits",
    "pred_residual_replace_logits",
)


def apply_residual_hybrid_replacement(
    preds: dict[str, torch.Tensor],
    *,
    point_valid_thr: float,
    min_points: int,
    max_det: int,
    promote_rank: int = 5,
    max_top_replace_gap: float = 0.09,
    max_boundary_replace_gap: float = 0.04,
    min_row_margin: float = 0.18,
    max_row_evidence: float = 3.0,
    min_row_support_advantage: float = 0.01,
    row_temperature: float = 0.1,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Apply one gated residual proposal as an addition or replacement.

    When fewer than ``max_det`` base lanes are decodable, the proposal fills an
    unused query slot with its own existence logit. Otherwise it replaces the
    predicted victim while preserving that victim score. Ground truth is never used.
    """
    missing = [key for key in REQUIRED_KEYS if key not in preds]
    if missing:
        raise KeyError(f"Residual hybrid replacement requires outputs: {missing}")
    if int(max_det) <= 0 or int(promote_rank) <= 0:
        return preds, {"fired": False, "reason": "disabled_rank_or_max_det"}
    if float(row_temperature) <= 0.0:
        raise ValueError("row_temperature must be positive.")

    base_points = preds["pred_points"]
    base_logits = preds["pred_logits"]
    base_valid_logits = preds["pred_valid_logits"]
    residual_points = preds["pred_residual_points"]
    residual_valid_logits = preds["pred_residual_valid_logits"]
    residual_exist_logits = preds["pred_residual_exist_logits"]
    residual_rows = preds["pred_residual_row_logits"]
    replace_logits = preds["pred_residual_replace_logits"]
    if base_points.shape[0] != 1:
        raise ValueError(f"Residual hybrid replacement expects batch size 1, got {base_points.shape[0]}.")

    base_scores = base_logits[0].float().sigmoid()
    base_valid = base_valid_logits[0].float().sigmoid()
    decodable = (base_valid >= float(point_valid_thr)).sum(dim=-1) >= int(min_points)
    decodable_indices = torch.nonzero(decodable, as_tuple=False).flatten()
    if decodable_indices.numel() == 0:
        return preds, {"fired": False, "reason": "no_decodable_base"}
    selected_count = min(int(max_det), int(decodable_indices.numel()))
    selected_order = torch.argsort(base_scores[decodable_indices], descending=True)
    selected_base = decodable_indices[selected_order[:selected_count]]

    pair_probability = replace_logits[0].float().sigmoid()
    selected_pair_probability = pair_probability[:, selected_base]
    replace_score, victim_position = selected_pair_probability.max(dim=1)
    replace_order = torch.argsort(replace_score, descending=True)

    row_probability = torch.softmax(residual_rows[0].float() / float(row_temperature), dim=-1)
    row_top2 = row_probability.topk(2, dim=-1).values
    row_margin = row_top2[..., 0] - row_top2[..., 1]
    row_evidence = torch.logsumexp(residual_rows[0].float(), dim=-1) - math.log(float(residual_rows.shape[-1]))
    residual_valid = residual_valid_logits[0].float().sigmoid() >= float(point_valid_thr)
    residual_valid_count = residual_valid.sum(dim=-1)
    valid_denominator = residual_valid_count.clamp(min=1).float()
    row_margin_mean = (row_margin * residual_valid).sum(dim=-1) / valid_denominator
    row_evidence_mean = (row_evidence * residual_valid).sum(dim=-1) / valid_denominator
    row_support_score = replace_score * row_margin_mean / (1.0 + torch.nn.functional.softplus(row_evidence_mean))

    candidate = int(torch.argmax(row_support_score).item())
    candidate_replace_position = torch.nonzero(replace_order == candidate, as_tuple=False).flatten()
    if candidate_replace_position.numel() != 1:
        raise RuntimeError("Residual candidate must occur exactly once in replacement order.")
    candidate_replace_rank = int(candidate_replace_position.item()) + 1
    effective_promote_rank = min(int(promote_rank), int(replace_order.numel()))
    diagnostics = {
        "fired": False,
        "reason": "candidate_already_kept",
        "candidate": candidate,
        "candidate_replace_rank": candidate_replace_rank,
        "promote_rank": effective_promote_rank,
        "selected_base_count": selected_count,
    }
    if candidate_replace_rank <= effective_promote_rank:
        return preds, diagnostics

    top = int(replace_order[0].item())
    boundary = int(replace_order[effective_promote_rank - 1].item())
    top_replace_gap = float((replace_score[top] - replace_score[candidate]).item())
    boundary_replace_gap = float((replace_score[boundary] - replace_score[candidate]).item())
    row_support_advantage = float((row_support_score[candidate] - row_support_score[boundary]).item())
    candidate_row_margin = float(row_margin_mean[candidate].item())
    candidate_row_evidence = float(row_evidence_mean[candidate].item())
    candidate_valid_count = int(residual_valid_count[candidate].item())
    diagnostics.update(
        {
            "reason": "gate_rejected",
            "top_replace_gap": top_replace_gap,
            "boundary_replace_gap": boundary_replace_gap,
            "row_margin": candidate_row_margin,
            "row_evidence": candidate_row_evidence,
            "row_support_advantage": row_support_advantage,
            "candidate_valid_count": candidate_valid_count,
        }
    )
    gate_pass = bool(
        candidate_valid_count >= int(min_points)
        and top_replace_gap <= float(max_top_replace_gap)
        and boundary_replace_gap <= float(max_boundary_replace_gap)
        and candidate_row_margin >= float(min_row_margin)
        and candidate_row_evidence <= float(max_row_evidence)
        and row_support_advantage >= float(min_row_support_advantage)
    )
    if not gate_pass:
        return preds, diagnostics

    transformed = dict(preds)
    transformed_points = base_points.clone()
    transformed_logits = base_logits.clone()
    transformed_valid_logits = base_valid_logits.clone()
    if selected_count < int(max_det):
        selected_mask = torch.zeros_like(base_scores, dtype=torch.bool)
        selected_mask[selected_base] = True
        available = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        slot = int(available[torch.argmax(base_scores[available])].item())
        transformed_logits[0, slot] = residual_exist_logits[0, candidate]
        action = "added"
    else:
        slot = int(selected_base[int(victim_position[candidate].item())].item())
        action = "replaced"
    transformed_points[0, slot] = residual_points[0, candidate]
    transformed_valid_logits[0, slot] = residual_valid_logits[0, candidate]
    transformed["pred_points"] = transformed_points
    transformed["pred_logits"] = transformed_logits
    transformed["pred_valid_logits"] = transformed_valid_logits
    diagnostics.update(
        {
            "fired": True,
            "reason": action,
            "action": action,
            "slot": slot,
            "slot_base_score": float(base_scores[slot].item()),
            "candidate_exist_score": float(residual_exist_logits[0, candidate].float().sigmoid().item()),
            "pair_probability": float(
                pair_probability[candidate, selected_base[int(victim_position[candidate].item())]].item()
            ),
        }
    )
    return transformed, diagnostics
