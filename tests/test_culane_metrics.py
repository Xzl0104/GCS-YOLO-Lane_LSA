from __future__ import annotations

import numpy as np

from tools import eval_gcs
from tools.summarize_culane_table import build_row


RAW_SHAPE = (590, 1640)


def _lane(x: float) -> dict:
    points = np.asarray([[x, 500.0], [x, 300.0]], dtype=np.float32)
    return {"points_norm": points / np.asarray([RAW_SHAPE[1], RAW_SHAPE[0]], dtype=np.float32)}


def _gt(x: float) -> list[tuple[float, float]]:
    return [(x, 500.0), (x, 300.0)]


def test_culane_iou_requires_strictly_greater_than_threshold(monkeypatch) -> None:
    pred_mask = np.asarray([[1, 1, 0, 0]], dtype=np.uint8)
    gt_mask = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    monkeypatch.setattr(
        eval_gcs,
        "culane_lane_mask",
        lambda lane, image_shape, lane_width: pred_mask if isinstance(lane, np.ndarray) else gt_mask,
    )
    metrics, matches = eval_gcs.match_culane_lanes(
        [_lane(100.0)],
        [_gt(100.0)],
        raw_shape=RAW_SHAPE,
        iou_threshold=0.5,
    )
    assert metrics["tp"] == 0
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert matches[0]["culane_iou"] == 0.5
    assert matches[0]["tp"] is False


def test_culane_iou_above_threshold_is_true_positive(monkeypatch) -> None:
    pred_mask = np.asarray([[1, 1, 0, 0]], dtype=np.uint8)
    gt_mask = np.asarray([[1, 1, 1, 0]], dtype=np.uint8)
    monkeypatch.setattr(
        eval_gcs,
        "culane_lane_mask",
        lambda lane, image_shape, lane_width: pred_mask if isinstance(lane, np.ndarray) else gt_mask,
    )
    metrics, _ = eval_gcs.match_culane_lanes(
        [_lane(100.0)],
        [_gt(100.0)],
        raw_shape=RAW_SHAPE,
        iou_threshold=0.5,
    )
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_culane_iou_counts_extra_and_missing_lanes(monkeypatch) -> None:
    def fake_mask(lane, image_shape, lane_width):
        x = float(lane[0, 0] if isinstance(lane, np.ndarray) else lane[0][0])
        mask = np.zeros((1, 10), dtype=np.uint8)
        if isinstance(lane, np.ndarray):
            start = 0 if x < 200.0 else 4
        else:
            start = 0 if x < 200.0 else 8
        mask[0, start : start + 2] = 1
        return mask

    monkeypatch.setattr(eval_gcs, "culane_lane_mask", fake_mask)
    metrics, _ = eval_gcs.match_culane_lanes(
        [_lane(100.0), _lane(300.0)],
        [_gt(100.0), _gt(500.0)],
        raw_shape=RAW_SHAPE,
        iou_threshold=0.5,
    )
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1


def test_culane_iou_empty_prediction_and_ground_truth(monkeypatch) -> None:
    metrics, matches = eval_gcs.match_culane_lanes([], [], raw_shape=RAW_SHAPE)
    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert matches == []


def test_culane_table_reports_crossroad_false_positive_count() -> None:
    def record(image: str, category: str, tp: int, fp: int, fn: int) -> dict:
        return {
            "image": image,
            "inference_ms": 10.0,
            "postprocess_ms": 5.0,
            "metrics": {"tp": tp, "fp": fp, "fn": fn},
        }

    records = [
        record("normal.jpg", "Normal", 9, 1, 1),
        record("cross.jpg", "Crossroad", 0, 7, 0),
    ]
    category_by_image = {"normal.jpg": "Normal", "cross.jpg": "Crossroad"}
    row, details = build_row(records, category_by_image, "test")
    assert row["Normal"] == "90.0"
    assert row["Crossroad"] == "7"
    assert row["Total"] == "66.7"
    assert details["crossroad_field"] == "fp"
