# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Count-aware candidate diagnostics for GCS-YOLO-Lane."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils.gcs_candidate_matching import GCSLaneCandidate, match_candidates_to_gt
from ultralytics.utils.gcs_postprocess import lane_nms, longest_contiguous_valid_mask


ERROR_TYPES = (
    "A_COUNT_HEAD_WRONG",
    "B_CANDIDATE_POOL_MISSING",
    "C_TRUE_LANE_RANK_LOW",
    "D_TRUE_LANE_VALID_POINTS_LOW",
    "E_TRUE_LANE_SUPPRESSED_BY_NMS",
    "F_FINAL_COUNT_OK_BUT_FALSE_OR_DUP",
    "OK",
    "UNKNOWN",
)


def _gt_count(gt_valid: Any) -> int:
    valid = torch.as_tensor(gt_valid, dtype=torch.float32)
    if valid.ndim != 2:
        raise ValueError(f"gt_valid must have shape N x K, got {tuple(valid.shape)}.")
    return int((valid.sum(dim=1) >= 2).sum().item())


def _count_probs(pred_count_logits: Any | None, pred_count_cls: int | None, gt_count: int) -> tuple[int, list[float]]:
    if pred_count_logits is None:
        cls = int(gt_count if pred_count_cls is None else pred_count_cls)
        probs = [0.0, 0.0, 0.0, 0.0]
        if 2 <= cls <= 5:
            probs[cls - 2] = 1.0
        return cls, probs
    logits = torch.as_tensor(pred_count_logits, dtype=torch.float32).reshape(-1)
    if logits.numel() != 4:
        raise ValueError(f"pred_count_logits must have four values, got {tuple(logits.shape)}.")
    prob = logits.softmax(dim=0).cpu().numpy().astype(float).tolist()
    return int(np.argmax(prob) + 2), [float(x) for x in prob]


def _ranked(candidates: list[GCSLaneCandidate]) -> list[GCSLaneCandidate]:
    return sorted(candidates, key=lambda c: (c.pre_nms_rank if c.pre_nms_rank > 0 else 10**9, -c.pre_nms_score))


def _recall_count(candidates: list[GCSLaneCandidate], limit: int | None, gt_count: int) -> int:
    selected = _ranked(candidates)
    if limit is not None:
        selected = selected[: int(limit)]
    matched = {int(c.matched_gt_id) for c in selected if int(c.matched_gt_id) >= 0}
    return min(len(matched), int(gt_count))


def _recall_fraction(candidates: list[GCSLaneCandidate], limit: int | None, gt_count: int) -> float:
    if gt_count <= 0:
        return 1.0
    return float(_recall_count(candidates, limit, gt_count)) / float(gt_count)


def _best_candidate_for_gt(candidates: list[GCSLaneCandidate], gt_id: int) -> GCSLaneCandidate | None:
    pool = [c for c in candidates if int(c.matched_gt_id) == int(gt_id)]
    if not pool:
        return None
    return max(pool, key=lambda c: (float(c.line_iou), -int(c.pre_nms_rank or 10**9), float(c.pre_nms_score)))


def _edge_gt_ids(gt_lanes: Any, gt_valid: Any) -> set[int]:
    points = torch.as_tensor(gt_lanes, dtype=torch.float32)
    valid = torch.as_tensor(gt_valid, dtype=torch.float32)
    lane_mask = valid.sum(dim=1) >= 2
    if int(lane_mask.sum().item()) < 4:
        return set()
    mean_x = (points[..., 0] * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    mean_x = torch.where(lane_mask, mean_x, torch.full_like(mean_x, float("inf")))
    left = int(torch.argmin(mean_x).item())
    mean_x_right = torch.where(lane_mask, mean_x, torch.full_like(mean_x, float("-inf")))
    right = int(torch.argmax(mean_x_right).item())
    return {left, right}


def _duplicate_or_false(final_candidates: list[GCSLaneCandidate]) -> tuple[bool, bool]:
    best_ids = [int(c.matched_gt_id) for c in final_candidates]
    has_false = any(x < 0 for x in best_ids)
    positives = [x for x in best_ids if x >= 0]
    has_duplicate = len(positives) != len(set(positives))
    return has_false, has_duplicate


def diagnose_count_errors(
    *,
    image_id: str,
    gt_lanes: Any,
    gt_valid: Any,
    candidates: list[GCSLaneCandidate],
    final_candidates: list[GCSLaneCandidate],
    pred_count_logits: Any | None = None,
    pred_count_cls: int | None = None,
    diagnostic_topk: int = 8,
    diagnostic_match_thr: float = 0.5,
    image_shape: tuple[int, int] = (544, 960),
    normal_min_points: int = 5,
) -> dict:
    """Classify one image into A/B/C/D/E/F count-error buckets."""
    gt_n = _gt_count(gt_valid)
    pred_cls, probs = _count_probs(pred_count_logits, pred_count_cls, gt_n)
    candidates = _ranked(candidates)
    match_candidates_to_gt(candidates, gt_lanes, gt_valid, match_thr=diagnostic_match_thr, image_shape=image_shape)
    match_candidates_to_gt(final_candidates, gt_lanes, gt_valid, match_thr=diagnostic_match_thr, image_shape=image_shape)

    final_matched = {int(c.matched_gt_id) for c in final_candidates if int(c.matched_gt_id) >= 0}
    missing_gt_ids = [i for i in range(gt_n) if i not in final_matched]
    final_count = len(final_candidates)
    fp_count = sum(1 for c in final_candidates if int(c.matched_gt_id) < 0)
    fn_count = max(0, gt_n - len(final_matched))
    has_false_lane, has_duplicate_lane = _duplicate_or_false(final_candidates)
    edge_missing = bool(_edge_gt_ids(gt_lanes, gt_valid) - final_matched)

    best_missing = [_best_candidate_for_gt(candidates, gi) for gi in missing_gt_ids]
    best_existing = [c for c in best_missing if c is not None]
    missing_exists = bool(best_existing)
    missing_best_rank = min((int(c.pre_nms_rank) for c in best_existing if int(c.pre_nms_rank) > 0), default=-1)
    missing_best_valid = max((int(c.valid_points) for c in best_existing), default=0)
    missing_best_score = max((float(c.pre_nms_score) for c in best_existing), default=0.0)
    missing_suppressed = any(c.suppressed_by is not None and int(c.suppressed_by) >= 0 for c in best_existing)

    recall_all_count = _recall_count(candidates, None, gt_n)
    recall_top_count = _recall_count(candidates, diagnostic_topk, gt_n)

    secondary: list[str] = []
    if pred_cls != gt_n:
        secondary.append("A_COUNT_HEAD_WRONG")
    if recall_all_count < gt_n:
        secondary.append("B_CANDIDATE_POOL_MISSING")
    if recall_all_count >= gt_n and recall_top_count < gt_n:
        secondary.append("C_TRUE_LANE_RANK_LOW")
    if missing_exists and missing_best_valid < int(normal_min_points):
        secondary.append("D_TRUE_LANE_VALID_POINTS_LOW")
    if missing_suppressed:
        secondary.append("E_TRUE_LANE_SUPPRESSED_BY_NMS")
    if final_count == gt_n and (has_false_lane or has_duplicate_lane or fn_count > 0):
        secondary.append("F_FINAL_COUNT_OK_BUT_FALSE_OR_DUP")

    if pred_cls != gt_n:
        primary = "A_COUNT_HEAD_WRONG"
    elif fn_count == 0 and fp_count == 0 and not has_duplicate_lane and final_count == gt_n:
        primary = "OK"
    elif recall_all_count < gt_n:
        primary = "B_CANDIDATE_POOL_MISSING"
    elif recall_top_count < gt_n:
        primary = "C_TRUE_LANE_RANK_LOW"
    elif missing_exists and missing_best_valid < int(normal_min_points):
        primary = "D_TRUE_LANE_VALID_POINTS_LOW"
    elif missing_suppressed:
        primary = "E_TRUE_LANE_SUPPRESSED_BY_NMS"
    elif final_count == gt_n and (has_false_lane or has_duplicate_lane or fn_count > 0):
        primary = "F_FINAL_COUNT_OK_BUT_FALSE_OR_DUP"
    else:
        primary = "UNKNOWN"
    if primary == "OK":
        secondary = ["OK"]
    elif not secondary:
        secondary = [primary]

    hard_reasons = []
    if gt_n == 5 and final_count < 5:
        hard_reasons.append("gt5_final_lt5")
    if gt_n >= 4 and edge_missing:
        hard_reasons.append("edge_lane_missing")
    if primary != "OK":
        hard_reasons.append(primary)

    row = {
        "image_id": str(image_id),
        "gt_count": int(gt_n),
        "pred_count_cls": int(pred_cls),
        "pred_count_prob_2": float(probs[0]),
        "pred_count_prob_3": float(probs[1]),
        "pred_count_prob_4": float(probs[2]),
        "pred_count_prob_5": float(probs[3]),
        "pre_nms_candidate_count": int(len(candidates)),
        "post_count_before_nms": int(sum(1 for c in candidates if c.keep_after_nms or c.suppressed_by is not None)),
        "post_count_after_nms": int(sum(1 for c in candidates if c.keep_after_nms)),
        "final_count": int(final_count),
        "count_error_primary": primary,
        "count_error_secondary": ";".join(dict.fromkeys(secondary)),
        "candidate_recall_at_5": _recall_fraction(candidates, 5, gt_n),
        "candidate_recall_at_6": _recall_fraction(candidates, 6, gt_n),
        "candidate_recall_at_8": _recall_fraction(candidates, 8, gt_n),
        "candidate_recall_all": _recall_fraction(candidates, None, gt_n),
        "missing_gt_ids": ";".join(str(x) for x in missing_gt_ids),
        "missing_gt_exists_pre_nms": int(missing_exists),
        "missing_gt_best_rank": int(missing_best_rank),
        "missing_gt_best_valid_points": int(missing_best_valid),
        "missing_gt_best_score": float(missing_best_score),
        "missing_gt_suppressed_by_nms": int(missing_suppressed),
        "has_false_lane": int(has_false_lane),
        "has_duplicate_lane": int(has_duplicate_lane),
        "fp_count": int(fp_count),
        "fn_count": int(fn_count),
        "edge_lane_missing": int(edge_missing),
        "hard_sample_reason": ";".join(dict.fromkeys(hard_reasons)),
        "top_candidates": [c.to_json() for c in candidates[: int(diagnostic_topk)]],
    }
    for i in range(1, int(diagnostic_topk) + 1):
        cand = candidates[i - 1] if i <= len(candidates) else None
        row.update(
            {
                f"top{i}_query_idx": -1 if cand is None else int(cand.query_idx),
                f"top{i}_score": 0.0 if cand is None else float(cand.pre_nms_score),
                f"top{i}_exist_score": 0.0 if cand is None else float(cand.exist_score),
                f"top{i}_valid_points": 0 if cand is None else int(cand.valid_points),
                f"top{i}_point_valid_mean": 0.0 if cand is None else float(cand.point_valid_mean),
                f"top{i}_matched_gt_id": -1 if cand is None else int(cand.matched_gt_id),
                f"top{i}_line_iou": 0.0 if cand is None else float(cand.line_iou),
                f"top{i}_official_lane_acc": 0.0 if cand is None else float(cand.official_lane_acc),
                f"top{i}_keep_after_nms": 0 if cand is None else int(bool(cand.keep_after_nms)),
                f"top{i}_suppressed_by": -1 if cand is None or cand.suppressed_by is None else int(cand.suppressed_by),
            }
        )
    return row


def build_candidates_from_predictions(
    *,
    image_id: str,
    pred_points: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_valid_logits: torch.Tensor | None = None,
    pred_quality_logits: torch.Tensor | None = None,
    image_shape: tuple[int, int] = (544, 960),
    normal_candidate_score_thr: float = 0.03,
    normal_point_valid_thr: float = 0.15,
    normal_min_points: int = 5,
    rescue_candidate_score_thr: float = 0.015,
    rescue_point_valid_thr: float = 0.08,
    rescue_min_points: int = 4,
    nms_dist_px: float = 18.0,
    line_nms_min_overlap: int = 6,
) -> list[GCSLaneCandidate]:
    """Build normal/rescue diagnostic candidates from raw model outputs."""
    points = pred_points.detach().float().cpu().clamp(0.0, 1.0)
    logits = pred_logits.detach().float().cpu().reshape(-1)
    exist = logits.sigmoid()
    valid_scores = (
        pred_valid_logits.detach().float().cpu().sigmoid()
        if pred_valid_logits is not None
        else torch.ones(points.shape[:2], dtype=torch.float32)
    )
    quality = pred_quality_logits.detach().float().cpu().sigmoid().reshape(-1) if pred_quality_logits is not None else None

    candidates: dict[int, GCSLaneCandidate] = {}
    for source, score_thr, valid_thr, min_points in (
        ("normal", normal_candidate_score_thr, normal_point_valid_thr, normal_min_points),
        ("rescue", rescue_candidate_score_thr, rescue_point_valid_thr, rescue_min_points),
    ):
        for q in range(points.shape[0]):
            if float(exist[q]) < float(score_thr):
                continue
            valid_mask = longest_contiguous_valid_mask(valid_scores[q] >= float(valid_thr), min_points=int(min_points))
            valid_points = int(valid_mask.sum().item())
            if valid_points < int(min_points):
                continue
            mean_valid = float(valid_scores[q][valid_mask].mean().item()) if valid_points else 0.0
            lane_quality = float(exist[q]) * float(np.sqrt(max(mean_valid, 0.0)))
            if quality is not None:
                lane_quality *= float(quality[q])
            if q in candidates and candidates[q].source == "normal":
                continue
            candidates[q] = GCSLaneCandidate(
                image_id=image_id,
                query_idx=int(q),
                points=points[q].numpy(),
                valid_probs=valid_scores[q].numpy(),
                exist_logit=float(logits[q]),
                exist_score=float(exist[q]),
                point_valid_mean=float(valid_scores[q].mean().item()),
                point_valid_max=float(valid_scores[q].max().item()),
                valid_points=valid_points,
                lane_quality=float(lane_quality),
                geometry_feat={"mean_valid_visible": float(mean_valid)},
                pre_nms_score=float(lane_quality),
                source=source,
            )
    ranked = sorted(candidates.values(), key=lambda c: c.pre_nms_score, reverse=True)
    for i, cand in enumerate(ranked, start=1):
        cand.pre_nms_rank = i
    if ranked and float(nms_dist_px) > 0.0:
        nms_points = torch.stack([c.points_tensor() for c in ranked], dim=0)
        nms_scores = torch.tensor([c.pre_nms_score for c in ranked], dtype=torch.float32)
        valid_masks = torch.stack([c.valid_tensor() for c in ranked], dim=0)
        keep, suppressed = lane_nms(
            nms_points,
            nms_scores,
            image_shape=image_shape,
            dist_thr_px=float(nms_dist_px),
            valid_masks=valid_masks,
            min_overlap=int(line_nms_min_overlap),
            return_suppressed=True,
        )
        keep_set = {int(i) for i in keep.tolist()}
        for i, cand in enumerate(ranked):
            cand.keep_after_nms = i in keep_set
        for item in suppressed:
            ranked[int(item["index"])].suppressed_by = int(item.get("suppressed_by_index", -1))
    else:
        for cand in ranked:
            cand.keep_after_nms = True
    return ranked


def summarize_count_diagnostics(rows: list[dict]) -> dict:
    """Aggregate per-image diagnostic rows into summary metrics."""
    n = len(rows)
    errors = Counter(str(r.get("count_error_primary", "UNKNOWN")) for r in rows)
    for name in ERROR_TYPES:
        errors.setdefault(name, 0)
    gt5_rows = [r for r in rows if int(r.get("gt_count", 0)) == 5]
    gt5_n = len(gt5_rows)
    fp_sum = sum(int(r.get("fp_count", 0)) for r in rows)
    fn_sum = sum(int(r.get("fn_count", 0)) for r in rows)
    gt_sum = sum(max(int(r.get("gt_count", 0)), 1) for r in rows)
    return {
        "num_images": int(n),
        "official_acc": round(float(sum(max(0, int(r.get("gt_count", 0)) - int(r.get("fn_count", 0))) for r in rows) / max(gt_sum, 1)), 6),
        "official_fp": round(float(fp_sum / max(gt_sum, 1)), 6),
        "official_fn": round(float(fn_sum / max(gt_sum, 1)), 6),
        "count_acc_all": round(float(sum(int(r.get("gt_count")) == int(r.get("pred_count_cls")) for r in rows) / max(n, 1)), 6),
        "count_acc_gt5": round(float(sum(int(r.get("pred_count_cls")) == 5 for r in gt5_rows) / max(gt5_n, 1)), 6),
        "gt5_output5_rate": round(float(sum(int(r.get("final_count", 0)) == 5 for r in gt5_rows) / max(gt5_n, 1)), 6),
        "candidate_recall_at_5_all": round(float(np.mean([float(r.get("candidate_recall_at_5", 0.0)) for r in rows])) if rows else 0.0, 6),
        "candidate_recall_at_6_all": round(float(np.mean([float(r.get("candidate_recall_at_6", 0.0)) for r in rows])) if rows else 0.0, 6),
        "candidate_recall_at_8_all": round(float(np.mean([float(r.get("candidate_recall_at_8", 0.0)) for r in rows])) if rows else 0.0, 6),
        "candidate_recall_at_8_gt5": round(float(np.mean([float(r.get("candidate_recall_at_8", 0.0)) for r in gt5_rows])) if gt5_rows else 0.0, 6),
        "nms_suppressed_true_lane_rate": round(float(sum(int(r.get("missing_gt_suppressed_by_nms", 0)) for r in rows) / max(n, 1)), 6),
        "edge_lane_missing_rate_gt5": round(float(sum(int(r.get("edge_lane_missing", 0)) for r in gt5_rows) / max(gt5_n, 1)), 6),
        "error_type_counts": {k: int(errors[k]) for k in ERROR_TYPES},
    }


def diagnostic_csv_fields(diagnostic_topk: int = 8) -> list[str]:
    """Return stable per-image CSV field order."""
    base = [
        "image_id",
        "gt_count",
        "pred_count_cls",
        "pred_count_prob_2",
        "pred_count_prob_3",
        "pred_count_prob_4",
        "pred_count_prob_5",
        "pre_nms_candidate_count",
        "post_count_before_nms",
        "post_count_after_nms",
        "final_count",
        "count_error_primary",
        "count_error_secondary",
        "candidate_recall_at_5",
        "candidate_recall_at_6",
        "candidate_recall_at_8",
        "candidate_recall_all",
        "missing_gt_ids",
        "missing_gt_exists_pre_nms",
        "missing_gt_best_rank",
        "missing_gt_best_valid_points",
        "missing_gt_best_score",
        "missing_gt_suppressed_by_nms",
        "has_false_lane",
        "has_duplicate_lane",
        "fp_count",
        "fn_count",
        "edge_lane_missing",
        "hard_sample_reason",
    ]
    for i in range(1, int(diagnostic_topk) + 1):
        base.extend(
            [
                f"top{i}_query_idx",
                f"top{i}_score",
                f"top{i}_exist_score",
                f"top{i}_valid_points",
                f"top{i}_point_valid_mean",
                f"top{i}_matched_gt_id",
                f"top{i}_line_iou",
                f"top{i}_official_lane_acc",
                f"top{i}_keep_after_nms",
                f"top{i}_suppressed_by",
            ]
        )
    return base


def write_count_diagnostics(
    rows: list[dict],
    out_dir: str | Path,
    *,
    diagnostic_topk: int = 8,
    write_hard_samples: bool = True,
) -> dict:
    """Write per_image.csv, per_image.jsonl, summary.json, and optional hard-sample manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = diagnostic_csv_fields(diagnostic_topk)
    with (out / "per_image.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with (out / "per_image.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize_count_diagnostics(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if write_hard_samples:
        with (out / "count_hard_samples.txt").open("w", encoding="utf-8") as f:
            for row in rows:
                reason = str(row.get("hard_sample_reason", ""))
                if reason:
                    f.write(
                        f"{row.get('image_id')}\t{reason}\t{row.get('gt_count')}\t"
                        f"{row.get('pred_count_cls')}\t{row.get('count_error_primary')}\n"
                    )
    return summary
