from __future__ import annotations

import pytest
import torch

from ultralytics.utils.gcs_postprocess import decode_gcs_predictions


def _synthetic_fifthness_case() -> dict[str, torch.Tensor]:
    q = 6
    k = 6
    points = torch.zeros((q, k, 2), dtype=torch.float32)
    points[:, :, 0] = torch.linspace(0.10, 0.60, q, dtype=torch.float32).view(q, 1)
    points[:, :, 1] = torch.linspace(0.98, 0.25, k, dtype=torch.float32)
    return {
        "pred_points": points,
        "pred_logits": torch.tensor([9.0, 8.8, 8.6, 8.4, 8.2, 7.8], dtype=torch.float32),
        "pred_valid_logits": torch.full((q, k), 9.0, dtype=torch.float32),
        "pred_count_logits": torch.tensor([-8.0, -8.0, -8.0, 8.0], dtype=torch.float32),
        "pred_quality_logits": torch.full((q,), 9.0, dtype=torch.float32),
    }


def _decode_queries(**overrides) -> list[int]:
    preds = _synthetic_fifthness_case()
    kwargs = {
        **preds,
        "image_shape": (720, 1280),
        "score_thr": 0.0,
        "point_valid_thr": 0.5,
        "min_points": 2,
        "max_det": 5,
        "nms_dist_px": 0.0,
        "candidate_score_thr": 0.0,
        "candidate_point_valid_thr": 0.5,
        "candidate_min_points": 2,
        "final_min_points": 2,
        "fifth_min_points": 2,
        "quality_rescue_5th": False,
    }
    kwargs.update(overrides)
    lanes = decode_gcs_predictions(**kwargs)
    return [int(lane["query"]) for lane in lanes]


def test_fifthness_logits_do_not_change_decode_when_disabled():
    baseline = _decode_queries()
    with_logits = _decode_queries(pred_fifthness_logits=torch.tensor([-8.0, -8.0, -8.0, -8.0, -8.0, 8.0]))

    assert baseline == [0, 1, 2, 3, 4]
    assert with_logits == baseline


def test_fifthness_decode_requires_logits_when_enabled():
    with pytest.raises(ValueError, match="use_fifthness_decode=True requires pred_fifthness_logits"):
        _decode_queries(use_fifthness_decode=True)


def test_fifthness_decode_accepts_q_by_one_logits():
    queries = _decode_queries(
        pred_fifthness_logits=torch.tensor([[-8.0], [-8.0], [-8.0], [-8.0], [-8.0], [8.0]]),
        use_fifthness_decode=True,
        fifthness_decode_thr=0.5,
        fifthness_decode_rank_weight=1.0,
    )

    assert queries == [0, 1, 2, 3, 5]


def test_fifthness_decode_only_changes_selected_rank_five():
    baseline = _decode_queries()
    reranked = _decode_queries(
        pred_fifthness_logits=torch.tensor([-8.0, -8.0, -8.0, -8.0, -8.0, 8.0]),
        use_fifthness_decode=True,
        fifthness_decode_thr=0.5,
        fifthness_decode_rank_weight=1.0,
    )

    assert baseline[:4] == reranked[:4] == [0, 1, 2, 3]
    assert baseline[4] == 4
    assert reranked[4] == 5


def test_fifthness_decode_rejects_wrong_query_shape():
    with pytest.raises(ValueError, match="pred_fifthness_logits must have shape Q"):
        _decode_queries(pred_fifthness_logits=torch.zeros(5), use_fifthness_decode=True)
