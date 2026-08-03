"""Smoke-test the default-off GCS local short-segment proposal head and loss."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402


DEFAULT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"
SEGMENT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v4.yaml"
SEGMENT_V5_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v5.yaml"
SEGMENT_V6_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v6.yaml"
SEGMENT_V7_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v7.yaml"
SEGMENT_V8_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v8.yaml"
SEGMENT_V9_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v9.yaml"
SEGMENT_V10_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v10.yaml"
SEGMENT_V12_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v12.yaml"
SEGMENT_V13_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v13.yaml"


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _head_from_yaml(cfg: Path) -> GCSLaneHead:
    model = GCSLaneModel(str(cfg), nc=1, verbose=False)
    head = model.model[-1]
    if not isinstance(head, GCSLaneHead):
        raise AssertionError(f"{cfg} did not build a GCSLaneHead.")
    return head.eval()


def _head_features(head: GCSLaneHead, batch: int = 1) -> list[torch.Tensor]:
    c = int(head.c1)
    return [
        torch.randn(batch, c, 32, 32),
        torch.randn(batch, c, 16, 16),
        torch.randn(batch, c, 8, 8),
        torch.randn(batch, c, 4, 4),
    ]


@torch.inference_mode()
def check_yaml_forward() -> None:
    default_head = _head_from_yaml(DEFAULT_CFG)
    default_out = default_head(_head_features(default_head), orig_size=(544, 960))
    unexpected = [k for k in default_out if k.startswith("pred_short_segment")]
    if unexpected:
        raise AssertionError(f"default query YAML unexpectedly emitted short-segment outputs: {unexpected}.")

    segment_head = _head_from_yaml(SEGMENT_CFG)
    segment_out = segment_head(_head_features(segment_head), orig_size=(544, 960))
    required = {
        "pred_short_segment_points",
        "pred_short_segment_logits",
        "pred_short_segment_x",
        "pred_short_segment_starts",
        "pred_short_segment_ends",
        "pred_short_segment_window_mask",
    }
    missing = required.difference(segment_out)
    if missing:
        raise AssertionError(f"segment YAML missing output keys: {sorted(missing)}.")
    if "pred_short_segment_replace_logits" in segment_out:
        raise AssertionError("v4 segment YAML must not emit pred_short_segment_replace_logits.")

    points = segment_out["pred_short_segment_points"]
    logits = segment_out["pred_short_segment_logits"]
    segment_x = segment_out["pred_short_segment_x"]
    starts = segment_out["pred_short_segment_starts"]
    ends = segment_out["pred_short_segment_ends"]
    mask = segment_out["pred_short_segment_window_mask"]
    expected_segments = sum(56 - length + 1 for length in range(3, 11))
    if tuple(points.shape) != (1, 12, expected_segments, 56, 2):
        raise AssertionError(f"segment point shape mismatch: {tuple(points.shape)}.")
    if tuple(logits.shape) != (1, 12, expected_segments):
        raise AssertionError(f"segment logit shape mismatch: {tuple(logits.shape)}.")
    if tuple(segment_x.shape) != (1, 12, expected_segments, 2):
        raise AssertionError(f"segment endpoint-x shape mismatch: {tuple(segment_x.shape)}.")
    if tuple(mask.shape) != (expected_segments, 56):
        raise AssertionError(f"segment window mask shape mismatch: {tuple(mask.shape)}.")
    lengths = ends - starts + 1
    if set(int(x) for x in lengths.tolist()) != set(range(3, 11)):
        raise AssertionError(f"segment window lengths mismatch: {sorted(set(lengths.tolist()))}.")
    if not torch.equal(mask.sum(dim=1).cpu(), lengths.cpu()):
        raise AssertionError("segment window masks do not match start/end lengths.")

    segment_v5_head = _head_from_yaml(SEGMENT_V5_CFG)
    segment_v5_out = segment_v5_head(_head_features(segment_v5_head), orig_size=(544, 960))
    missing_v5 = (required | {"pred_short_segment_replace_logits"}).difference(segment_v5_out)
    if missing_v5:
        raise AssertionError(f"segment v5 YAML missing output keys: {sorted(missing_v5)}.")
    replace_logits = segment_v5_out["pred_short_segment_replace_logits"]
    if tuple(replace_logits.shape) != (1, 12, expected_segments):
        raise AssertionError(f"segment replace-logit shape mismatch: {tuple(replace_logits.shape)}.")
    if getattr(segment_v5_head, "short_segment_local_evidence", False):
        raise AssertionError("v5 segment YAML must not enable proposal-local evidence.")

    segment_v6_head = _head_from_yaml(SEGMENT_V6_CFG)
    if not getattr(segment_v6_head, "short_segment_local_evidence", False):
        raise AssertionError("segment v6 YAML should enable proposal-local evidence.")
    segment_v6_out = segment_v6_head(_head_features(segment_v6_head), orig_size=(544, 960))
    missing_v6 = (required | {"pred_short_segment_replace_logits"}).difference(segment_v6_out)
    if missing_v6:
        raise AssertionError(f"segment v6 YAML missing output keys: {sorted(missing_v6)}.")
    if tuple(segment_v6_out["pred_short_segment_points"].shape) != (1, 12, expected_segments, 56, 2):
        raise AssertionError(f"segment v6 point shape mismatch: {tuple(segment_v6_out['pred_short_segment_points'].shape)}.")
    if tuple(segment_v6_out["pred_short_segment_replace_logits"].shape) != (1, 12, expected_segments):
        raise AssertionError(
            f"segment v6 replace-logit shape mismatch: {tuple(segment_v6_out['pred_short_segment_replace_logits'].shape)}."
        )

    segment_v7_head = _head_from_yaml(SEGMENT_V7_CFG)
    if not getattr(segment_v7_head, "short_segment_local_evidence", False):
        raise AssertionError("segment v7 YAML should enable proposal-local evidence.")
    segment_v7_out = segment_v7_head(_head_features(segment_v7_head), orig_size=(544, 960))
    missing_v7 = (required | {"pred_short_segment_replace_logits"}).difference(segment_v7_out)
    if missing_v7:
        raise AssertionError(f"segment v7 YAML missing output keys: {sorted(missing_v7)}.")
    if tuple(segment_v7_out["pred_short_segment_points"].shape) != (1, 12, expected_segments, 56, 2):
        raise AssertionError(f"segment v7 point shape mismatch: {tuple(segment_v7_out['pred_short_segment_points'].shape)}.")
    if tuple(segment_v7_out["pred_short_segment_replace_logits"].shape) != (1, 12, expected_segments):
        raise AssertionError(
            f"segment v7 replace-logit shape mismatch: {tuple(segment_v7_out['pred_short_segment_replace_logits'].shape)}."
        )

    segment_v8_head = _head_from_yaml(SEGMENT_V8_CFG)
    if not getattr(segment_v8_head, "short_segment_local_evidence", False):
        raise AssertionError("segment v8 YAML should enable proposal-local evidence.")
    if getattr(segment_v8_head, "short_segment_replace_head", False):
        raise AssertionError("segment v8 YAML should disable per-candidate replace logits.")
    if not getattr(segment_v8_head, "short_segment_query_replace_head", False):
        raise AssertionError("segment v8 YAML should enable query-level replace logits.")
    segment_v8_out = segment_v8_head(_head_features(segment_v8_head), orig_size=(544, 960))
    missing_v8 = (required | {"pred_short_segment_query_replace_logits"}).difference(segment_v8_out)
    if missing_v8:
        raise AssertionError(f"segment v8 YAML missing output keys: {sorted(missing_v8)}.")
    if "pred_short_segment_replace_logits" in segment_v8_out:
        raise AssertionError("segment v8 YAML must not emit pred_short_segment_replace_logits.")
    if tuple(segment_v8_out["pred_short_segment_points"].shape) != (1, 12, expected_segments, 56, 2):
        raise AssertionError(f"segment v8 point shape mismatch: {tuple(segment_v8_out['pred_short_segment_points'].shape)}.")
    if tuple(segment_v8_out["pred_short_segment_query_replace_logits"].shape) != (1, 12):
        raise AssertionError(
            "segment v8 query replace-logit shape mismatch: "
            f"{tuple(segment_v8_out['pred_short_segment_query_replace_logits'].shape)}."
        )

    segment_v9_head = _head_from_yaml(SEGMENT_V9_CFG)
    if not getattr(segment_v9_head, "short_segment_local_evidence", False):
        raise AssertionError("segment v9 YAML should enable proposal-local evidence.")
    if getattr(segment_v9_head, "short_segment_replace_head", False):
        raise AssertionError("segment v9 YAML should disable per-candidate replace logits.")
    if getattr(segment_v9_head, "short_segment_query_replace_head", False):
        raise AssertionError("segment v9 YAML should disable query-level replace logits.")
    segment_v9_out = segment_v9_head(_head_features(segment_v9_head), orig_size=(544, 960))
    missing_v9 = required.difference(segment_v9_out)
    if missing_v9:
        raise AssertionError(f"segment v9 YAML missing output keys: {sorted(missing_v9)}.")
    if "pred_short_segment_replace_logits" in segment_v9_out:
        raise AssertionError("segment v9 YAML must not emit pred_short_segment_replace_logits.")
    if "pred_short_segment_query_replace_logits" in segment_v9_out:
        raise AssertionError("segment v9 YAML must not emit pred_short_segment_query_replace_logits.")
    if tuple(segment_v9_out["pred_short_segment_points"].shape) != (1, 12, expected_segments, 56, 2):
        raise AssertionError(f"segment v9 point shape mismatch: {tuple(segment_v9_out['pred_short_segment_points'].shape)}.")

    segment_v10_head = _head_from_yaml(SEGMENT_V10_CFG)
    if not getattr(segment_v10_head, "short_segment_local_evidence", False):
        raise AssertionError("segment v10 YAML should enable proposal-local evidence.")
    if getattr(segment_v10_head, "short_segment_replace_head", False):
        raise AssertionError("segment v10 YAML should disable per-candidate replace logits.")
    if getattr(segment_v10_head, "short_segment_query_replace_head", False):
        raise AssertionError("segment v10 YAML should disable query-level replace logits.")
    segment_v10_out = segment_v10_head(_head_features(segment_v10_head), orig_size=(544, 960))
    missing_v10 = required.difference(segment_v10_out)
    if missing_v10:
        raise AssertionError(f"segment v10 YAML missing output keys: {sorted(missing_v10)}.")
    if "pred_short_segment_replace_logits" in segment_v10_out:
        raise AssertionError("segment v10 YAML must not emit pred_short_segment_replace_logits.")
    if "pred_short_segment_query_replace_logits" in segment_v10_out:
        raise AssertionError("segment v10 YAML must not emit pred_short_segment_query_replace_logits.")
    if tuple(segment_v10_out["pred_short_segment_points"].shape) != (1, 12, expected_segments, 56, 2):
        raise AssertionError(f"segment v10 point shape mismatch: {tuple(segment_v10_out['pred_short_segment_points'].shape)}.")

    segment_v12_head = _head_from_yaml(SEGMENT_V12_CFG)
    if not getattr(segment_v12_head, "short_segment_local_evidence", False):
        raise AssertionError("segment v12 YAML should enable proposal-local evidence.")
    if getattr(segment_v12_head, "short_segment_replace_head", False):
        raise AssertionError("segment v12 YAML should disable per-candidate replace logits.")
    if getattr(segment_v12_head, "short_segment_query_replace_head", False):
        raise AssertionError("segment v12 YAML should disable query-level replace logits.")
    if not getattr(segment_v12_head, "short_segment_unified_choice_head", False):
        raise AssertionError("segment v12 YAML should enable the unified base-plus-segment choice head.")
    segment_v12_out = segment_v12_head(_head_features(segment_v12_head), orig_size=(544, 960))
    missing_v12 = required | {"pred_short_segment_choice_logits"}
    missing_v12 = missing_v12.difference(segment_v12_out)
    if missing_v12:
        raise AssertionError(f"segment v12 YAML missing output keys: {sorted(missing_v12)}.")
    if "pred_short_segment_replace_logits" in segment_v12_out:
        raise AssertionError("segment v12 YAML must not emit pred_short_segment_replace_logits.")
    if "pred_short_segment_query_replace_logits" in segment_v12_out:
        raise AssertionError("segment v12 YAML must not emit pred_short_segment_query_replace_logits.")
    if tuple(segment_v12_out["pred_short_segment_points"].shape) != (1, 12, expected_segments, 56, 2):
        raise AssertionError(
            f"segment v12 point shape mismatch: {tuple(segment_v12_out['pred_short_segment_points'].shape)}."
        )
    if tuple(segment_v12_out["pred_short_segment_choice_logits"].shape) != (1, 12, expected_segments + 1):
        raise AssertionError(
            "segment v12 unified-choice shape mismatch: "
            f"{tuple(segment_v12_out['pred_short_segment_choice_logits'].shape)}."
        )

    segment_v13_head = _head_from_yaml(SEGMENT_V13_CFG)
    if not getattr(segment_v13_head, "short_segment_local_evidence", False):
        raise AssertionError("segment v13 YAML should enable proposal-local evidence.")
    if getattr(segment_v13_head, "short_segment_replace_head", False):
        raise AssertionError("segment v13 YAML should disable per-candidate replace logits.")
    if getattr(segment_v13_head, "short_segment_query_replace_head", False):
        raise AssertionError("segment v13 YAML should disable query-level replace logits.")
    if not getattr(segment_v13_head, "short_segment_unified_choice_head", False):
        raise AssertionError("segment v13 YAML should enable the unified base-plus-segment choice head.")
    segment_v13_out = segment_v13_head(_head_features(segment_v13_head), orig_size=(544, 960))
    missing_v13 = (required | {"pred_short_segment_choice_logits"}).difference(segment_v13_out)
    if missing_v13:
        raise AssertionError(f"segment v13 YAML missing output keys: {sorted(missing_v13)}.")
    if "pred_short_segment_replace_logits" in segment_v13_out:
        raise AssertionError("segment v13 YAML must not emit pred_short_segment_replace_logits.")
    if "pred_short_segment_query_replace_logits" in segment_v13_out:
        raise AssertionError("segment v13 YAML must not emit pred_short_segment_query_replace_logits.")
    if tuple(segment_v13_out["pred_short_segment_points"].shape) != (1, 12, expected_segments, 56, 2):
        raise AssertionError(
            f"segment v13 point shape mismatch: {tuple(segment_v13_out['pred_short_segment_points'].shape)}."
        )
    if tuple(segment_v13_out["pred_short_segment_choice_logits"].shape) != (1, 12, expected_segments + 1):
        raise AssertionError(
            "segment v13 unified-choice shape mismatch: "
            f"{tuple(segment_v13_out['pred_short_segment_choice_logits'].shape)}."
        )


def _make_short_segment_batch() -> dict:
    k = 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    xs = torch.tensor([0.4, 0.3, 0.6, 0.75], dtype=torch.float32).view(4, 1).expand(4, k)
    lanes = torch.stack((xs, y.view(1, k).expand(4, k)), dim=-1).float()
    valid = torch.zeros(4, k, dtype=torch.float32)
    valid[0, :5] = 1.0
    valid[1:, :16] = 1.0
    return {
        "lanes": [lanes],
        "lane_valid": [valid],
        "num_lanes": torch.tensor([4], dtype=torch.long),
    }


def _make_short_segment_preds(
    *,
    include_replace: bool = False,
    include_query_replace: bool = False,
    include_unified_choice: bool = False,
) -> dict[str, torch.Tensor]:
    b, q, s, k = 1, 12, 2, 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k).view(1, 1, k).expand(b, q, k)
    x = torch.full((b, q, k), 0.5)
    pred_points = torch.stack((x, y), dim=-1).contiguous()
    segment_points = pred_points.unsqueeze(2).expand(b, q, s, k, 2).clone()
    segment_points[0, 0, 0, :5, 0] = 0.4
    segment_logits = torch.full((b, q, s), _logit(0.02), dtype=torch.float32)
    segment_logits[0, 0, 0] = _logit(0.5)
    window_mask = torch.zeros(s, k, dtype=torch.bool)
    window_mask[0, :5] = True
    window_mask[1, 10:15] = True
    preds = {
        "pred_points": pred_points,
        "pred_logits": torch.full((b, q), _logit(0.2), dtype=torch.float32),
        "pred_valid_logits": torch.full((b, q, k), _logit(0.7), dtype=torch.float32),
        "pred_short_segment_points": segment_points,
        "pred_short_segment_logits": segment_logits,
        "pred_short_segment_window_mask": window_mask,
    }
    if include_replace:
        replace_logits = torch.full((b, q, s), _logit(0.02), dtype=torch.float32)
        replace_logits[0, 0, 0] = _logit(0.5)
        preds["pred_short_segment_replace_logits"] = replace_logits
    if include_query_replace:
        query_replace_logits = torch.full((b, q), _logit(0.02), dtype=torch.float32)
        query_replace_logits[0, 0] = _logit(0.5)
        preds["pred_short_segment_query_replace_logits"] = query_replace_logits
    if include_unified_choice:
        choice_logits = torch.zeros((b, q, s + 1), dtype=torch.float32)
        choice_logits[:, :, 1:] = -4.0
        preds["pred_short_segment_choice_logits"] = choice_logits
    return preds


def _segment_criterion() -> GCSLoss:
    return GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_segment": 0.1,
            "gcs_short_segment_topk": 2,
            "gcs_short_segment_visible_thr": 10,
            "gcs_short_segment_min_visible": 3,
            "gcs_short_segment_min_overlap": 3,
            "gcs_short_segment_neg_score_thr": 0.6,
        }
    )


def _segment_v5_criterion() -> GCSLoss:
    return GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_segment": 0.1,
            "gcs_short_segment_topk": 2,
            "gcs_short_segment_visible_thr": 10,
            "gcs_short_segment_min_visible": 3,
            "gcs_short_segment_min_overlap": 3,
            "gcs_short_segment_bce_weight": 0.0,
            "gcs_short_segment_listwise_weight": 1.0,
            "gcs_short_segment_replace_weight": 1.0,
            "gcs_short_segment_replace_margin_px": 5.0,
        }
    )


def _segment_v7_criterion() -> GCSLoss:
    return GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_segment": 0.1,
            "gcs_short_segment_topk": 2,
            "gcs_short_segment_visible_thr": 10,
            "gcs_short_segment_min_visible": 3,
            "gcs_short_segment_min_overlap": 3,
            "gcs_short_segment_bce_weight": 0.0,
            "gcs_short_segment_listwise_weight": 1.0,
            "gcs_short_segment_listwise_all_candidates": True,
            "gcs_short_segment_dense_quality_weight": 1.0,
            "gcs_short_segment_dense_neg_weight": 0.05,
            "gcs_short_segment_replace_weight": 1.0,
            "gcs_short_segment_replace_dense_neg_weight": 0.25,
            "gcs_short_segment_replace_margin_px": 5.0,
        }
    )


def _segment_v8_criterion() -> GCSLoss:
    return GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_segment": 0.1,
            "gcs_short_segment_topk": 2,
            "gcs_short_segment_visible_thr": 10,
            "gcs_short_segment_min_visible": 3,
            "gcs_short_segment_min_overlap": 3,
            "gcs_short_segment_bce_weight": 0.0,
            "gcs_short_segment_listwise_weight": 0.0,
            "gcs_short_segment_query_rank_weight": 1.0,
            "gcs_short_segment_dense_quality_weight": 1.0,
            "gcs_short_segment_dense_neg_weight": 0.05,
            "gcs_short_segment_replace_weight": 0.0,
            "gcs_short_segment_query_replace_weight": 1.0,
            "gcs_short_segment_query_replace_neg_weight": 0.25,
            "gcs_short_segment_replace_margin_px": 5.0,
        }
    )


def _segment_v9_criterion() -> GCSLoss:
    return GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_segment": 0.1,
            "gcs_short_segment_topk": 2,
            "gcs_short_segment_visible_thr": 10,
            "gcs_short_segment_min_visible": 3,
            "gcs_short_segment_min_overlap": 3,
            "gcs_short_segment_bce_weight": 0.0,
            "gcs_short_segment_listwise_weight": 0.0,
            "gcs_short_segment_query_rank_weight": 0.0,
            "gcs_short_segment_base_choice_weight": 1.0,
            "gcs_short_segment_dense_quality_weight": 1.0,
            "gcs_short_segment_dense_neg_weight": 0.05,
            "gcs_short_segment_replace_weight": 0.0,
            "gcs_short_segment_query_replace_weight": 0.0,
            "gcs_short_segment_replace_margin_px": 5.0,
        }
    )


def _segment_v10_criterion() -> GCSLoss:
    return GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_segment": 0.1,
            "gcs_short_segment_topk": 2,
            "gcs_short_segment_visible_thr": 10,
            "gcs_short_segment_min_visible": 3,
            "gcs_short_segment_min_overlap": 3,
            "gcs_short_segment_bce_weight": 0.0,
            "gcs_short_segment_base_choice_weight": 1.0,
            "gcs_short_segment_dense_quality_weight": 1.0,
            "gcs_short_segment_dense_neg_weight": 0.05,
            "gcs_short_segment_matched_assignment": True,
        }
    )


def _segment_v12_criterion() -> GCSLoss:
    return GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_segment": 0.1,
            "gcs_short_segment_topk": 2,
            "gcs_short_segment_visible_thr": 10,
            "gcs_short_segment_min_visible": 3,
            "gcs_short_segment_min_overlap": 3,
            "gcs_short_segment_point_weight": 0.0,
            "gcs_short_segment_bce_weight": 0.0,
            "gcs_short_segment_listwise_weight": 0.0,
            "gcs_short_segment_base_choice_weight": 0.0,
            "gcs_short_segment_replace_weight": 0.0,
            "gcs_short_segment_query_replace_weight": 0.0,
            "gcs_short_segment_dense_quality_weight": 0.0,
            "gcs_short_segment_unified_choice_weight": 1.0,
            "gcs_short_segment_unified_choice_temperature": 0.25,
            "gcs_short_segment_unified_choice_base_neg_weight": 0.25,
            "gcs_short_segment_matched_assignment": True,
        }
    )


def _segment_v13_criterion() -> GCSLoss:
    return GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_segment": 0.1,
            "gcs_short_segment_topk": 2,
            "gcs_short_segment_visible_thr": 10,
            "gcs_short_segment_min_visible": 3,
            "gcs_short_segment_min_overlap": 3,
            "gcs_short_segment_point_weight": 0.0,
            "gcs_short_segment_bce_weight": 0.0,
            "gcs_short_segment_listwise_weight": 0.0,
            "gcs_short_segment_base_choice_weight": 0.0,
            "gcs_short_segment_replace_weight": 0.0,
            "gcs_short_segment_query_replace_weight": 0.0,
            "gcs_short_segment_dense_quality_weight": 0.0,
            "gcs_short_segment_unified_choice_weight": 1.0,
            "gcs_short_segment_unified_choice_temperature": 0.25,
            "gcs_short_segment_unified_choice_base_neg_weight": 0.25,
            "gcs_short_segment_matched_assignment": True,
            "gcs_short_segment_candidate_aware_assignment": True,
        }
    )


def check_loss() -> None:
    criterion = _segment_criterion()
    _, items = criterion(_make_short_segment_preds(), _make_short_segment_batch())
    if int(items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(f"GCSLoss item length mismatch: {items.numel()} vs {len(GCSLoss.loss_names)}.")
    if not torch.isfinite(items).all():
        raise AssertionError("GCSLoss produced non-finite short-segment items.")
    idx = {name: i for i, name in enumerate(GCSLoss.loss_names)}
    if float(items[idx["short_segment_pos_count"]]) <= 0.0:
        raise AssertionError("short-segment assignment did not produce a strong positive.")
    if float(items[idx["short_segment_loss"]]) <= 0.0:
        raise AssertionError("short-segment loss should be positive for the synthetic segment.")

    missing = _make_short_segment_preds()
    missing.pop("pred_short_segment_points")
    try:
        criterion(missing, _make_short_segment_batch())
    except KeyError:
        pass
    else:
        raise AssertionError("gcs_short_segment > 0 should require segment head outputs.")


def check_backward() -> None:
    preds = _make_short_segment_preds()
    segment_points = preds["pred_short_segment_points"].detach().clone()
    segment_points[0, 0, 0, :5, 0] = 0.405
    segment_points.requires_grad_(True)
    segment_logits = preds["pred_short_segment_logits"].detach().clone().requires_grad_(True)
    preds["pred_short_segment_points"] = segment_points
    preds["pred_short_segment_logits"] = segment_logits

    criterion = _segment_criterion()
    total, items = criterion(preds, _make_short_segment_batch())
    total.backward()
    if not torch.isfinite(total.detach()) or not torch.isfinite(items).all():
        raise AssertionError("short-segment backward smoke produced non-finite losses.")
    if segment_logits.grad is None or not torch.isfinite(segment_logits.grad).all():
        raise AssertionError("short-segment score logits did not receive finite gradients.")
    if float(segment_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment score logits received zero gradient.")
    if segment_points.grad is None or not torch.isfinite(segment_points.grad).all():
        raise AssertionError("short-segment points did not receive finite pull gradients.")
    if float(segment_points.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment points received zero pull gradient.")


def check_v5_replace_backward() -> None:
    preds = _make_short_segment_preds(include_replace=True)
    segment_logits = preds["pred_short_segment_logits"].detach().clone().requires_grad_(True)
    replace_logits = preds["pred_short_segment_replace_logits"].detach().clone().requires_grad_(True)
    preds["pred_short_segment_logits"] = segment_logits
    preds["pred_short_segment_replace_logits"] = replace_logits

    total, items = _segment_v5_criterion()(preds, _make_short_segment_batch())
    total.backward()
    if not torch.isfinite(total.detach()) or not torch.isfinite(items).all():
        raise AssertionError("short-segment v5 backward smoke produced non-finite losses.")
    if segment_logits.grad is None or not torch.isfinite(segment_logits.grad).all():
        raise AssertionError("short-segment v5 listwise logits did not receive finite gradients.")
    if float(segment_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment v5 listwise logits received zero gradient.")
    if replace_logits.grad is None or not torch.isfinite(replace_logits.grad).all():
        raise AssertionError("short-segment v5 replace logits did not receive finite gradients.")
    if float(replace_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment v5 replace logits received zero gradient.")

    missing = _make_short_segment_preds(include_replace=False)
    try:
        _segment_v5_criterion()(missing, _make_short_segment_batch())
    except KeyError:
        pass
    else:
        raise AssertionError("replace-weighted v5 loss should require pred_short_segment_replace_logits.")


def check_v7_dense_quality_backward() -> None:
    preds = _make_short_segment_preds(include_replace=True)
    segment_logits = preds["pred_short_segment_logits"].detach().clone()
    segment_logits[0, 0, 1] = _logit(0.9)
    segment_logits.requires_grad_(True)
    replace_logits = preds["pred_short_segment_replace_logits"].detach().clone()
    replace_logits[0, 0, 1] = _logit(0.9)
    replace_logits.requires_grad_(True)
    preds["pred_short_segment_logits"] = segment_logits
    preds["pred_short_segment_replace_logits"] = replace_logits

    total, items = _segment_v7_criterion()(preds, _make_short_segment_batch())
    total.backward()
    if not torch.isfinite(total.detach()) or not torch.isfinite(items).all():
        raise AssertionError("short-segment v7 backward smoke produced non-finite losses.")
    if segment_logits.grad is None or not torch.isfinite(segment_logits.grad).all():
        raise AssertionError("short-segment v7 dense-quality logits did not receive finite gradients.")
    if float(segment_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment v7 dense-quality logits received zero gradient.")
    if replace_logits.grad is None or not torch.isfinite(replace_logits.grad).all():
        raise AssertionError("short-segment v7 dense replace logits did not receive finite gradients.")
    if float(replace_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment v7 dense replace logits received zero gradient.")


def check_v8_query_replace_backward() -> None:
    preds = _make_short_segment_preds(include_query_replace=True)
    segment_logits = preds["pred_short_segment_logits"].detach().clone()
    segment_logits[0, 0, 1] = _logit(0.9)
    segment_logits.requires_grad_(True)
    query_replace_logits = preds["pred_short_segment_query_replace_logits"].detach().clone().requires_grad_(True)
    preds["pred_short_segment_logits"] = segment_logits
    preds["pred_short_segment_query_replace_logits"] = query_replace_logits

    total, items = _segment_v8_criterion()(preds, _make_short_segment_batch())
    total.backward()
    if not torch.isfinite(total.detach()) or not torch.isfinite(items).all():
        raise AssertionError("short-segment v8 backward smoke produced non-finite losses.")
    if segment_logits.grad is None or not torch.isfinite(segment_logits.grad).all():
        raise AssertionError("short-segment v8 query-rank logits did not receive finite gradients.")
    if float(segment_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment v8 query-rank logits received zero gradient.")
    if query_replace_logits.grad is None or not torch.isfinite(query_replace_logits.grad).all():
        raise AssertionError("short-segment v8 query replace logits did not receive finite gradients.")
    if float(query_replace_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment v8 query replace logits received zero gradient.")


def check_v9_base_choice_backward() -> None:
    preds = _make_short_segment_preds()
    segment_logits = preds["pred_short_segment_logits"].detach().clone()
    segment_logits[0, 0, 1] = _logit(0.9)
    segment_logits.requires_grad_(True)
    preds["pred_short_segment_logits"] = segment_logits

    total, items = _segment_v9_criterion()(preds, _make_short_segment_batch())
    total.backward()
    if not torch.isfinite(total.detach()) or not torch.isfinite(items).all():
        raise AssertionError("short-segment v9 base-choice backward smoke produced non-finite losses.")
    if segment_logits.grad is None or not torch.isfinite(segment_logits.grad).all():
        raise AssertionError("short-segment v9 base-choice logits did not receive finite gradients.")
    if float(segment_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment v9 base-choice logits received zero gradient.")


def check_v10_matched_assignment_backward() -> None:
    preds = _make_short_segment_preds()
    segment_logits = preds["pred_short_segment_logits"].detach().clone()
    segment_logits[0, 0, 1] = _logit(0.9)
    segment_logits.requires_grad_(True)
    preds["pred_short_segment_logits"] = segment_logits
    criterion = _segment_v10_criterion()
    batch = _make_short_segment_batch()
    explicit_indices = [
        (
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
        )
    ]
    total, score_loss, point_loss, pos_count, soft_count, neg_count = criterion.short_segment_loss(
        preds,
        batch["lanes"],
        batch["lane_valid"],
        batch["num_lanes"],
        indices=explicit_indices,
    )
    total.backward()
    values = torch.stack((total, score_loss, point_loss, pos_count, soft_count, neg_count))
    if not torch.isfinite(total.detach()) or not torch.isfinite(values).all():
        raise AssertionError("short-segment v10 matched-assignment backward smoke produced non-finite losses.")
    if segment_logits.grad is None or not torch.isfinite(segment_logits.grad).all():
        raise AssertionError("short-segment v10 matched-assignment logits did not receive finite gradients.")
    if float(segment_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment v10 matched-assignment logits received zero gradient.")
    forward_total, forward_items = criterion(preds, batch)
    if not torch.isfinite(forward_total.detach()) or not torch.isfinite(forward_items).all():
        raise AssertionError("short-segment v10 forward path produced non-finite losses.")


def check_v12_unified_choice_backward() -> None:
    preds = _make_short_segment_preds(include_unified_choice=True)
    choice_logits = preds["pred_short_segment_choice_logits"].detach().clone()
    # The first matched query is a base miss but its first local segment is an exact
    # short-lane proposal. This must create a non-zero gradient on the unified choice.
    choice_logits[0, 0, 0] = 0.0
    choice_logits[0, 0, 1] = -1.0
    choice_logits.requires_grad_(True)
    preds["pred_short_segment_choice_logits"] = choice_logits
    batch = _make_short_segment_batch()
    explicit_indices = [
        (
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
        )
    ]
    total, score_loss, point_loss, pos_count, soft_count, neg_count = _segment_v12_criterion().short_segment_loss(
        preds,
        batch["lanes"],
        batch["lane_valid"],
        batch["num_lanes"],
        indices=explicit_indices,
    )
    total.backward()
    values = torch.stack((total, score_loss, point_loss, pos_count, soft_count, neg_count))
    if not torch.isfinite(total.detach()) or not torch.isfinite(values).all():
        raise AssertionError("short-segment v12 unified-choice backward smoke produced non-finite losses.")
    if float(score_loss.detach()) <= 0.0:
        raise AssertionError("short-segment v12 unified-choice score loss should be positive.")
    if choice_logits.grad is None or not torch.isfinite(choice_logits.grad).all():
        raise AssertionError("short-segment v12 unified-choice logits did not receive finite gradients.")
    if float(choice_logits.grad.abs().sum()) <= 0.0:
        raise AssertionError("short-segment v12 unified-choice logits received zero gradient.")
    forward_total, forward_items = _segment_v12_criterion()(preds, batch)
    if not torch.isfinite(forward_total.detach()) or not torch.isfinite(forward_items).all():
        raise AssertionError("short-segment v12 forward path produced non-finite losses.")


def check_v13_candidate_aware_assignment_backward() -> None:
    preds = _make_short_segment_preds(include_unified_choice=True)
    segment_points = preds["pred_short_segment_points"].detach().clone()
    # Hungarian query 0 is a base miss and has no good segment for lane 0.
    # Query 5 carries the exact short segment and must become the positive
    # unified-choice query when candidate-aware assignment is enabled.
    segment_points[0, 0, 0, :5, 0] = 0.5
    segment_points[0, 5, 0, :5, 0] = 0.4
    preds["pred_short_segment_points"] = segment_points
    choice_logits = preds["pred_short_segment_choice_logits"].detach().clone()
    choice_logits[:, :, 0] = 0.0
    choice_logits[:, :, 1:] = -2.0
    choice_logits.requires_grad_(True)
    preds["pred_short_segment_choice_logits"] = choice_logits
    batch = _make_short_segment_batch()
    explicit_indices = [
        (
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
        )
    ]
    total, score_loss, point_loss, pos_count, soft_count, neg_count = _segment_v13_criterion().short_segment_loss(
        preds,
        batch["lanes"],
        batch["lane_valid"],
        batch["num_lanes"],
        indices=explicit_indices,
    )
    total.backward()
    values = torch.stack((total, score_loss, point_loss, pos_count, soft_count, neg_count))
    if not torch.isfinite(total.detach()) or not torch.isfinite(values).all():
        raise AssertionError("short-segment v13 candidate-aware backward smoke produced non-finite losses.")
    if choice_logits.grad is None or not torch.isfinite(choice_logits.grad).all():
        raise AssertionError("short-segment v13 candidate-aware logits did not receive finite gradients.")
    selected_grad = float(choice_logits.grad[0, 5, 1].detach().cpu().item())
    if selected_grad >= 0.0:
        raise AssertionError(
            "short-segment v13 did not push up the raw candidate query's selected segment class; "
            f"grad={selected_grad}."
        )
    hungarian_segment_grad = float(choice_logits.grad[0, 0, 1].detach().cpu().item())
    if hungarian_segment_grad <= 0.0:
        raise AssertionError(
            "short-segment v13 should not train the wrong Hungarian query segment as the positive; "
            f"grad={hungarian_segment_grad}."
        )
    forward_total, forward_items = _segment_v13_criterion()(preds, batch)
    if not torch.isfinite(forward_total.detach()) or not torch.isfinite(forward_items).all():
        raise AssertionError("short-segment v13 forward path produced non-finite losses.")


def check_empty_segment_batch_has_grad() -> None:
    preds = _make_short_segment_preds()
    segment_logits = preds["pred_short_segment_logits"].detach().clone().requires_grad_(True)
    segment_points = preds["pred_short_segment_points"].detach().clone().requires_grad_(True)
    preds["pred_short_segment_logits"] = segment_logits
    preds["pred_short_segment_points"] = segment_points
    batch = _make_short_segment_batch()
    batch["num_lanes"] = torch.tensor([3], dtype=torch.long)
    total, items = _segment_criterion()(preds, batch)
    if not total.requires_grad:
        raise AssertionError("empty short-segment batch should keep a zero-gradient path to segment outputs.")
    total.backward()
    if segment_logits.grad is None or not torch.isfinite(segment_logits.grad).all():
        raise AssertionError("empty short-segment batch did not produce finite zero score gradients.")
    if segment_points.grad is None or not torch.isfinite(segment_points.grad).all():
        raise AssertionError("empty short-segment batch did not produce finite zero point gradients.")
    if float(segment_logits.grad.abs().sum()) != 0.0 or float(segment_points.grad.abs().sum()) != 0.0:
        raise AssertionError("empty short-segment batch should not update segment outputs.")
    if not torch.isfinite(items).all():
        raise AssertionError("empty short-segment batch produced non-finite loss items.")


def check_segment_decode() -> None:
    preds = _make_short_segment_preds()
    lanes = decode_gcs_predictions(
        preds["pred_points"][0],
        preds["pred_logits"][0],
        pred_valid_logits=preds["pred_valid_logits"][0],
        pred_short_segment_points=preds["pred_short_segment_points"][0],
        pred_short_segment_logits=preds["pred_short_segment_logits"][0],
        pred_short_segment_window_mask=preds["pred_short_segment_window_mask"],
        image_shape=(544, 960),
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=3,
        max_det=12,
        segment_decode=True,
        segment_score_thr=0.5,
        segment_short_min_points=3,
        segment_short_max_points=10,
    )
    selected = [lane for lane in lanes if int(lane["query"]) == 0]
    if len(selected) != 1:
        raise AssertionError(f"segment decode should keep query 0 exactly once, got {len(selected)}.")
    lane = selected[0]
    if not lane.get("segment_decode_applied", False):
        raise AssertionError("segment decode did not apply the selected base-choice segment.")
    if int(lane["short_segment_index"]) != 0:
        raise AssertionError(f"segment decode selected the wrong window: {lane['short_segment_index']}.")
    visible = torch.as_tensor(lane["point_valid"], dtype=torch.float32)
    if int(visible.sum()) != 5 or bool(visible[5:].any()):
        raise AssertionError("segment decode did not restrict visibility to the selected local window.")
    if not torch.allclose(torch.as_tensor(lane["points_norm"][:5, 0]), torch.full((5,), 0.4), atol=1e-6):
        raise AssertionError("segment decode did not replace the query geometry inside the selected window.")

    score_lanes = decode_gcs_predictions(
        preds["pred_points"][0],
        preds["pred_logits"][0],
        pred_valid_logits=preds["pred_valid_logits"][0],
        pred_short_segment_points=preds["pred_short_segment_points"][0],
        pred_short_segment_logits=preds["pred_short_segment_logits"][0],
        pred_short_segment_window_mask=preds["pred_short_segment_window_mask"],
        image_shape=(544, 960),
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=3,
        max_det=12,
        segment_decode=True,
        segment_score_thr=0.5,
        segment_short_min_points=3,
        segment_short_max_points=10,
        segment_score_as_lane_score=True,
    )
    score_selected = [lane for lane in score_lanes if int(lane["query"]) == 0]
    if len(score_selected) != 1 or abs(float(score_selected[0]["score"]) - 0.5) > 1e-6:
        raise AssertionError("segment score bridge did not raise the selected lane score.")

    rescue_lanes = decode_gcs_predictions(
        preds["pred_points"][0],
        preds["pred_logits"][0],
        pred_valid_logits=preds["pred_valid_logits"][0],
        pred_short_segment_points=preds["pred_short_segment_points"][0],
        pred_short_segment_logits=preds["pred_short_segment_logits"][0],
        pred_short_segment_window_mask=preds["pred_short_segment_window_mask"],
        image_shape=(544, 960),
        score_thr=0.3,
        point_valid_thr=0.5,
        min_points=3,
        max_det=12,
        segment_decode=True,
        segment_score_thr=0.5,
        segment_short_min_points=3,
        segment_short_max_points=10,
        segment_score_as_lane_score=True,
        segment_rescue_base_miss_only=True,
    )
    rescue_selected = [lane for lane in rescue_lanes if int(lane["query"]) == 0]
    if len(rescue_selected) != 1 or not rescue_selected[0].get("segment_decode_applied", False):
        raise AssertionError("base-miss-only segment rescue should apply below the base score threshold.")

    preserve_lanes = decode_gcs_predictions(
        preds["pred_points"][0],
        preds["pred_logits"][0],
        pred_valid_logits=preds["pred_valid_logits"][0],
        pred_short_segment_points=preds["pred_short_segment_points"][0],
        pred_short_segment_logits=preds["pred_short_segment_logits"][0],
        pred_short_segment_window_mask=preds["pred_short_segment_window_mask"],
        image_shape=(544, 960),
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=3,
        max_det=12,
        segment_decode=True,
        segment_score_thr=0.5,
        segment_short_min_points=3,
        segment_short_max_points=10,
        segment_score_as_lane_score=True,
        segment_rescue_base_miss_only=True,
    )
    preserve_selected = [lane for lane in preserve_lanes if int(lane["query"]) == 0]
    if len(preserve_selected) != 1 or preserve_selected[0].get("segment_decode_applied", False):
        raise AssertionError("base-miss-only segment rescue should preserve an already-kept base query.")

    conservative_lanes = decode_gcs_predictions(
        preds["pred_points"][0],
        preds["pred_logits"][0],
        pred_valid_logits=preds["pred_valid_logits"][0],
        pred_short_segment_points=preds["pred_short_segment_points"][0],
        pred_short_segment_logits=preds["pred_short_segment_logits"][0],
        pred_short_segment_window_mask=preds["pred_short_segment_window_mask"],
        image_shape=(544, 960),
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=3,
        max_det=12,
        segment_decode=True,
        segment_score_thr=0.6,
        segment_short_min_points=3,
        segment_short_max_points=10,
    )
    conservative = [lane for lane in conservative_lanes if int(lane["query"]) == 0]
    if len(conservative) != 1 or conservative[0].get("segment_decode_applied", False):
        raise AssertionError("segment score gate below threshold should preserve the base geometry.")

    try:
        decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_valid_logits=preds["pred_valid_logits"][0],
            pred_short_candidate_points=preds["pred_short_segment_points"][0],
            pred_short_candidate_logits=preds["pred_short_segment_logits"][0],
            pred_short_segment_points=preds["pred_short_segment_points"][0],
            pred_short_segment_logits=preds["pred_short_segment_logits"][0],
            pred_short_segment_window_mask=preds["pred_short_segment_window_mask"],
            candidate_decode=True,
            segment_decode=True,
        )
    except ValueError as exc:
        if "mutually exclusive" not in str(exc):
            raise AssertionError(f"unexpected candidate/segment conflict error: {exc}") from exc
    else:
        raise AssertionError("candidate_decode and segment_decode should be mutually exclusive.")

    try:
        decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_valid_logits=preds["pred_valid_logits"][0],
            pred_short_segment_points=preds["pred_short_segment_points"][0],
            pred_short_segment_logits=preds["pred_short_segment_logits"][0],
            pred_short_segment_window_mask=preds["pred_short_segment_window_mask"],
            segment_score_as_lane_score=True,
        )
    except ValueError as exc:
        if "requires segment_decode" not in str(exc):
            raise AssertionError(f"unexpected score bridge validation error: {exc}") from exc
    else:
        raise AssertionError("segment_score_as_lane_score should require segment_decode.")

    try:
        decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_valid_logits=preds["pred_valid_logits"][0],
            pred_short_segment_points=preds["pred_short_segment_points"][0],
            pred_short_segment_logits=preds["pred_short_segment_logits"][0],
            pred_short_segment_window_mask=preds["pred_short_segment_window_mask"],
            segment_decode=True,
            segment_rescue_base_miss_only=True,
        )
    except ValueError as exc:
        if "requires segment_score_as_lane_score" not in str(exc):
            raise AssertionError(f"unexpected base-miss-only validation error: {exc}") from exc
    else:
        raise AssertionError("segment_rescue_base_miss_only should require the score bridge.")


def check_v12_unified_choice_decode() -> None:
    preds = _make_short_segment_preds()
    choice_logits = torch.full((1, 12, 3), -4.0, dtype=torch.float32)
    choice_logits[:, :, 0] = 0.0
    choice_logits[0, 0, 1] = 2.0
    preds["pred_short_segment_choice_logits"] = choice_logits
    lanes = decode_gcs_predictions(
        preds["pred_points"][0],
        preds["pred_logits"][0],
        pred_valid_logits=preds["pred_valid_logits"][0],
        pred_short_segment_points=preds["pred_short_segment_points"][0],
        pred_short_segment_logits=preds["pred_short_segment_logits"][0],
        pred_short_segment_window_mask=preds["pred_short_segment_window_mask"],
        pred_short_segment_choice_logits=preds["pred_short_segment_choice_logits"][0],
        image_shape=(544, 960),
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=3,
        max_det=12,
        segment_decode=True,
        segment_score_thr=0.5,
        segment_short_min_points=3,
        segment_short_max_points=10,
    )
    selected = [lane for lane in lanes if int(lane["query"]) == 0]
    if len(selected) != 1:
        raise AssertionError(f"v12 unified decode should keep query 0 exactly once, got {len(selected)}.")
    lane = selected[0]
    if not lane.get("segment_decode_applied", False):
        raise AssertionError("v12 unified decode did not apply the selected segment.")
    if int(lane["short_segment_index"]) != 0:
        raise AssertionError(f"v12 unified decode selected the wrong window: {lane['short_segment_index']}.")
    visible = torch.as_tensor(lane["point_valid"], dtype=torch.float32)
    if int(visible.sum()) != 5 or bool(visible[5:].any()):
        raise AssertionError("v12 unified decode did not restrict visibility to the selected local window.")

    preserve_logits = choice_logits.clone()
    preserve_logits[0, 0, 0] = 3.0
    preserve_logits[0, 0, 1] = 0.0
    preserve = decode_gcs_predictions(
        preds["pred_points"][0],
        preds["pred_logits"][0],
        pred_valid_logits=preds["pred_valid_logits"][0],
        pred_short_segment_points=preds["pred_short_segment_points"][0],
        pred_short_segment_logits=preds["pred_short_segment_logits"][0],
        pred_short_segment_window_mask=preds["pred_short_segment_window_mask"],
        pred_short_segment_choice_logits=preserve_logits[0],
        image_shape=(544, 960),
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=3,
        max_det=12,
        segment_decode=True,
        segment_score_thr=0.5,
        segment_short_min_points=3,
        segment_short_max_points=10,
    )
    preserve_selected = [lane for lane in preserve if int(lane["query"]) == 0]
    if len(preserve_selected) != 1 or preserve_selected[0].get("segment_decode_applied", False):
        raise AssertionError("v12 unified decode should preserve base when class 0 wins.")


def check_freeze_contract() -> None:
    model = GCSLaneModel(str(SEGMENT_V6_CFG), nc=1, verbose=False)
    trainer = object.__new__(GCSLaneTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(
        gcs_short_candidate_freeze_base=False,
        gcs_short_candidate=0.0,
        gcs_short_segment_freeze_base=True,
        gcs_short_segment=0.1,
    )
    trainer.freeze_layer_names = []
    trainer._apply_custom_freeze()

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise AssertionError("short-segment freeze contract left no trainable parameters.")
    bad = [name for name in trainable if "short_segment_" not in name]
    if bad:
        raise AssertionError(f"short-segment freeze contract left base parameters trainable: {bad[:5]}.")
    if not any("short_segment_x_mlp" in name for name in trainable):
        raise AssertionError("short-segment endpoint MLP is not trainable under freeze contract.")
    if not any("short_segment_replace_mlp" in name for name in trainable):
        raise AssertionError("short-segment replace MLP is not trainable under freeze contract.")
    if not any("short_segment_local_image_proj" in name for name in trainable):
        raise AssertionError("short-segment local image evidence projection is not trainable under freeze contract.")
    if not any("short_segment_geometry_mlp" in name for name in trainable):
        raise AssertionError("short-segment geometry evidence MLP is not trainable under freeze contract.")

    optimizer = trainer.build_optimizer(model, name="AdamW", lr=0.001, momentum=0.9, decay=0.0, iterations=1)
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    frozen_in_optimizer = [
        name for name, param in model.named_parameters() if not param.requires_grad and id(param) in optimizer_param_ids
    ]
    if frozen_in_optimizer:
        raise AssertionError(f"frozen parameters entered optimizer groups: {frozen_in_optimizer[:5]}.")

    trainer._model_train()
    head = model.model[-1]
    if head.training:
        raise AssertionError("frozen base GCSLaneHead parent should stay eval to suppress base-path stat drift.")
    if not head.short_segment_x_mlp.training:
        raise AssertionError("short-segment endpoint MLP should stay in train mode.")
    if not head.short_segment_replace_mlp.training:
        raise AssertionError("short-segment replace MLP should stay in train mode.")
    if not head.short_segment_local_image_proj.training:
        raise AssertionError("short-segment local image evidence projection should stay in train mode.")
    if not head.short_segment_geometry_mlp.training:
        raise AssertionError("short-segment geometry evidence MLP should stay in train mode.")
    bn_training = [name for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d) and module.training]
    if bn_training:
        raise AssertionError(f"frozen base BatchNorm modules stayed in train mode: {bn_training[:5]}.")

    model_v8 = GCSLaneModel(str(SEGMENT_V8_CFG), nc=1, verbose=False)
    trainer_v8 = object.__new__(GCSLaneTrainer)
    trainer_v8.model = model_v8
    trainer_v8.args = SimpleNamespace(
        gcs_short_candidate_freeze_base=False,
        gcs_short_candidate=0.0,
        gcs_short_segment_freeze_base=True,
        gcs_short_segment=0.1,
    )
    trainer_v8.freeze_layer_names = []
    trainer_v8._apply_custom_freeze()
    trainable_v8 = [name for name, param in model_v8.named_parameters() if param.requires_grad]
    bad_v8 = [name for name in trainable_v8 if "short_segment_" not in name]
    if bad_v8:
        raise AssertionError(f"v8 freeze contract left base parameters trainable: {bad_v8[:5]}.")
    if not any("short_segment_query_replace_mlp" in name for name in trainable_v8):
        raise AssertionError("v8 query-level replace MLP is not trainable under freeze contract.")
    if any("short_segment_replace_mlp" in name for name in trainable_v8):
        raise AssertionError("v8 should not train a per-candidate replace MLP.")
    trainer_v8._model_train()
    head_v8 = model_v8.model[-1]
    if head_v8.training:
        raise AssertionError("v8 frozen base GCSLaneHead parent should stay eval to suppress base-path stat drift.")
    if not head_v8.short_segment_query_replace_mlp.training:
        raise AssertionError("v8 query-level replace MLP should stay in train mode.")

    model_v9 = GCSLaneModel(str(SEGMENT_V9_CFG), nc=1, verbose=False)
    trainer_v9 = object.__new__(GCSLaneTrainer)
    trainer_v9.model = model_v9
    trainer_v9.args = SimpleNamespace(
        gcs_short_candidate_freeze_base=False,
        gcs_short_candidate=0.0,
        gcs_short_segment_freeze_base=True,
        gcs_short_segment=0.1,
    )
    trainer_v9.freeze_layer_names = []
    trainer_v9._apply_custom_freeze()
    trainable_v9 = [name for name, param in model_v9.named_parameters() if param.requires_grad]
    bad_v9 = [name for name in trainable_v9 if "short_segment_" not in name]
    if bad_v9:
        raise AssertionError(f"v9 freeze contract left base parameters trainable: {bad_v9[:5]}.")
    if any("short_segment_replace_mlp" in name for name in trainable_v9):
        raise AssertionError("v9 should not train a per-candidate replace MLP.")
    if any("short_segment_query_replace_mlp" in name for name in trainable_v9):
        raise AssertionError("v9 should not train a query-level replace MLP.")
    if not any("short_segment_score_mlp" in name for name in trainable_v9):
        raise AssertionError("v9 score MLP is not trainable under freeze contract.")

    model_v12 = GCSLaneModel(str(SEGMENT_V12_CFG), nc=1, verbose=False)
    trainer_v12 = object.__new__(GCSLaneTrainer)
    trainer_v12.model = model_v12
    trainer_v12.args = SimpleNamespace(
        gcs_short_candidate_freeze_base=False,
        gcs_short_candidate=0.0,
        gcs_short_segment_freeze_base=True,
        gcs_short_segment=0.1,
    )
    trainer_v12.freeze_layer_names = []
    trainer_v12._apply_custom_freeze()
    trainable_v12 = [name for name, param in model_v12.named_parameters() if param.requires_grad]
    bad_v12 = [name for name in trainable_v12 if "short_segment_" not in name]
    if bad_v12:
        raise AssertionError(f"v12 freeze contract left base parameters trainable: {bad_v12[:5]}.")
    if not any("short_segment_unified_choice_mlp" in name for name in trainable_v12):
        raise AssertionError("v12 unified-choice MLP is not trainable under freeze contract.")
    if any("short_segment_replace_mlp" in name for name in trainable_v12):
        raise AssertionError("v12 should not train a per-candidate replace MLP.")
    if any("short_segment_query_replace_mlp" in name for name in trainable_v12):
        raise AssertionError("v12 should not train a query-level replace MLP.")
    trainer_v12._model_train()
    head_v12 = model_v12.model[-1]
    if head_v12.training:
        raise AssertionError("v12 frozen base GCSLaneHead parent should stay eval to suppress base-path stat drift.")
    if not head_v12.short_segment_unified_choice_mlp.training:
        raise AssertionError("v12 unified-choice MLP should stay in train mode.")

    model_v13 = GCSLaneModel(str(SEGMENT_V13_CFG), nc=1, verbose=False)
    trainer_v13 = object.__new__(GCSLaneTrainer)
    trainer_v13.model = model_v13
    trainer_v13.args = SimpleNamespace(
        gcs_short_candidate_freeze_base=False,
        gcs_short_candidate=0.0,
        gcs_short_segment_freeze_base=True,
        gcs_short_segment=0.1,
    )
    trainer_v13.freeze_layer_names = []
    trainer_v13._apply_custom_freeze()
    trainable_v13 = [name for name, param in model_v13.named_parameters() if param.requires_grad]
    bad_v13 = [name for name in trainable_v13 if "short_segment_" not in name]
    if bad_v13:
        raise AssertionError(f"v13 freeze contract left base parameters trainable: {bad_v13[:5]}.")
    if not any("short_segment_unified_choice_mlp" in name for name in trainable_v13):
        raise AssertionError("v13 unified-choice MLP is not trainable under freeze contract.")
    if any("short_segment_replace_mlp" in name for name in trainable_v13):
        raise AssertionError("v13 should not train a per-candidate replace MLP.")
    if any("short_segment_query_replace_mlp" in name for name in trainable_v13):
        raise AssertionError("v13 should not train a query-level replace MLP.")
    trainer_v13._model_train()
    head_v13 = model_v13.model[-1]
    if head_v13.training:
        raise AssertionError("v13 frozen base GCSLaneHead parent should stay eval to suppress base-path stat drift.")
    if not head_v13.short_segment_unified_choice_mlp.training:
        raise AssertionError("v13 unified-choice MLP should stay in train mode.")


def main() -> None:
    check_yaml_forward()
    check_loss()
    check_backward()
    check_v5_replace_backward()
    check_v7_dense_quality_backward()
    check_v8_query_replace_backward()
    check_v9_base_choice_backward()
    check_v10_matched_assignment_backward()
    check_v12_unified_choice_backward()
    check_v13_candidate_aware_assignment_backward()
    check_empty_segment_batch_has_grad()
    check_segment_decode()
    check_v12_unified_choice_decode()
    check_freeze_contract()
    print("GCS short-segment proposal head checks passed.")


if __name__ == "__main__":
    main()
