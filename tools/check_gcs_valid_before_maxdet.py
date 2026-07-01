from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402


def _logit(p: float) -> float:
    p = float(p)
    return math.log(p / (1.0 - p))


def main() -> None:
    # q_empty has the higher score but no visible run. q_true has lower score
    # and enough contiguous valid anchors to survive min_points.
    k = 7
    y = torch.linspace(0.9, 0.3, k)
    pred_points = torch.stack(
        (
            torch.stack((torch.full((k,), 0.25), y), dim=1),
            torch.stack((torch.full((k,), 0.65), y), dim=1),
        ),
        dim=0,
    )
    pred_logits = torch.tensor([_logit(0.20), _logit(0.07)], dtype=torch.float32)
    pred_valid_logits = torch.stack(
        (
            torch.full((k,), _logit(0.01), dtype=torch.float32),
            torch.full((k,), _logit(0.90), dtype=torch.float32),
        ),
        dim=0,
    )

    common = {
        "pred_points": pred_points,
        "pred_logits": pred_logits,
        "pred_valid_logits": pred_valid_logits,
        "score_thr": 0.005,
        "point_valid_thr": 0.5,
        "min_points": 2,
        "max_det": 1,
        "nms_dist_px": 0.0,
    }
    baseline = decode_gcs_predictions(**common)
    valid_first = decode_gcs_predictions(**common, valid_before_maxdet=True)

    if baseline:
        raise AssertionError(f"Expected old max_det-first path to return no lanes, got {baseline!r}.")
    if len(valid_first) != 1 or int(valid_first[0]["query"]) != 1:
        raise AssertionError(f"Expected valid-before-maxdet path to keep q_true only, got {valid_first!r}.")

    report = {
        "baseline_queries": [int(x["query"]) for x in baseline],
        "valid_before_maxdet_queries": [int(x["query"]) for x in valid_first],
        "valid_before_maxdet_scores": [round(float(x["score"]), 6) for x in valid_first],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
