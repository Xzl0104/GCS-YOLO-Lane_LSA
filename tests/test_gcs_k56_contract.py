from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from gcs_tools.label_utils import fixed_y_anchors


ROOT = Path(__file__).resolve().parents[1]


def test_k56_fixed_y_anchors_match_tusimple_official_h_samples():
    anchors = fixed_y_anchors(num_points=56, y_start=710.0 / 720.0, y_end=160.0 / 720.0)
    expected = [y / 720.0 for y in range(710, 159, -10)]
    assert anchors.shape == (56,)
    assert len(expected) == 56
    assert np.allclose(anchors, expected, rtol=0.0, atol=1e-7)


def test_k56_data_and_model_contract_match():
    data = yaml.safe_load((ROOT / "data" / "tusimple_gcs_fixed_y_k56_960x544.yaml").read_text(encoding="utf-8"))
    model = yaml.safe_load(
        (ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q12-k56.yaml").read_text(
            encoding="utf-8"
        )
    )
    head_args = model["head"][-1][3]

    assert data["point_mode"] == "fixed_y"
    assert data["num_points"] == 56
    assert head_args[0] == 12
    assert head_args[1] == 56
    assert head_args[4] == "fixed_y"
    assert math.isclose(float(data["fixed_y"][0]), float(head_args[5]))
    assert math.isclose(float(data["fixed_y"][1]), float(head_args[6]))
