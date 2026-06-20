from __future__ import annotations

import pytest
import numpy as np

from tools.diagnose_tusimple_count_confusion import _label_row, build_diagnostics, date_id
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer


def _lane(visible: int, total: int = 56) -> list[int]:
    return [100] * int(visible) + [-2] * (int(total) - int(visible))


def _record(raw_file: str, visible_counts: list[int]) -> dict:
    return {
        "raw_file": raw_file,
        "h_samples": list(range(710, 150, -10)),
        "lanes": [_lane(n) for n in visible_counts],
    }


def test_count_confusion_groups_by_date_lane_count_and_min_visible_bucket() -> None:
    gt_records = [
        _record("clips/0531/clip_a/20.jpg", [46, 45, 12, 8]),
        _record("clips/0531/clip_b/20.jpg", [46, 45, 18, 15]),
        _record("clips/0601/clip_c/20.jpg", [46, 45, 18, 12, 9]),
        _record("clips/0313-1/clip_d/20.jpg", [46, 45, 18, 15]),
    ]
    pred_records = [
        _record("clips/0531/clip_a/20.jpg", [46, 45, 12]),
        _record("clips/0531/clip_b/20.jpg", [46, 45, 18, 15, 7]),
        _record("clips/0601/clip_c/20.jpg", [46, 45, 18, 12, 9]),
        _record("clips/0313-1/clip_d/20.jpg", [46, 45, 18, 15]),
    ]

    summary = build_diagnostics(gt_records, pred_records, bucket_edges=[10, 20])

    assert summary["overall"]["images"] == 4
    assert summary["overall"]["count_confusion"] == {
        "4->3": 1,
        "4->4": 1,
        "4->5": 1,
        "5->5": 1,
    }

    groups = {
        (row["date"], row["gt_count"], row["min_visible_bucket"]): row
        for row in summary["groups"]
    }
    assert groups[("0531", 4, "<= 10")]["count_confusion"] == {"4->3": 1}
    assert groups[("0531", 4, "11-20")]["count_confusion"] == {"4->5": 1}
    assert groups[("0601", 5, "<= 10")]["count_confusion"] == {"5->5": 1}
    assert groups[("0313-1", 4, "11-20")]["count_confusion"] == {"4->4": 1}


def test_missing_prediction_fails_by_default() -> None:
    gt_records = [_record("clips/0531/clip_a/20.jpg", [46, 45, 12, 8])]

    with pytest.raises(ValueError, match="Missing predictions"):
        build_diagnostics(gt_records, pred_records=[], bucket_edges=[10, 20])


def test_date_id_extracts_tusimple_domain() -> None:
    assert date_id("clips/0313-2/123/20.jpg") == "0313-2"
    assert date_id("clips/0601/123/20.jpg") == "0601"


def test_fixed_y_label_row_uses_shortest_visible_lane_bucket(tmp_path) -> None:
    label = tmp_path / "sample.npz"
    lane_valid = np.zeros((4, 56), dtype=np.float32)
    for i, visible in enumerate([46, 45, 12, 8]):
        lane_valid[i, :visible] = 1.0
    np.savez(
        label,
        lane_valid=lane_valid,
        raw_file=np.array("clips/0601/clip_a/20.jpg"),
        num_lanes=np.array([4], dtype=np.int64),
    )

    row = _label_row(label, pred_count=3, bucket_edges=[10, 20])

    assert row["date"] == "0601"
    assert row["gt_count"] == 4
    assert row["pred_count"] == 3
    assert row["min_visible_points"] == 8
    assert row["min_visible_bucket"] == "<= 10"


def test_label_sampling_stats_reads_gt4_short_lane(tmp_path) -> None:
    label = tmp_path / "sample.npz"
    lane_valid = np.zeros((4, 56), dtype=np.float32)
    for i, visible in enumerate([46, 45, 12, 8]):
        lane_valid[i, :visible] = 1.0
    np.savez(label, lane_valid=lane_valid, num_lanes=np.array([4], dtype=np.int64))

    assert GCSLaneTrainer._label_sampling_stats(label) == (4, 8)
