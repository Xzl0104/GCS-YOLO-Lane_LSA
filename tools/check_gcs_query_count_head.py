from __future__ import annotations

import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.sweep_tusimple_official import build_combos  # noqa: E402
from ultralytics.models.gcs.decode_summary import validate_decode_yaml_for_model  # noqa: E402
from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402


DEFAULT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"
COUNT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count.yaml"
ORDERED_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q5-slot-k56.yaml"


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _head_from_yaml(cfg: Path) -> GCSLaneHead:
    model = GCSLaneModel(str(cfg), nc=1, verbose=False)
    head = model.model[-1]
    if not isinstance(head, GCSLaneHead):
        raise AssertionError(f"{cfg} did not build a GCSLaneHead.")
    return head.eval()


def _head_features(head: GCSLaneHead, batch: int = 2) -> list[torch.Tensor]:
    c = int(head.c1)
    return [
        torch.randn(batch, c, 32, 32),
        torch.randn(batch, c, 16, 16),
        torch.randn(batch, c, 8, 8),
        torch.randn(batch, c, 4, 4),
    ]


@torch.inference_mode()
def check_yaml_forward() -> None:
    count_head = _head_from_yaml(COUNT_CFG)
    count_out = count_head(_head_features(count_head), orig_size=(544, 960))
    if "pred_count_logits" not in count_out:
        raise AssertionError("query-count YAML did not emit pred_count_logits.")
    if tuple(count_out["pred_count_logits"].shape) != (2, 4):
        raise AssertionError(f"query-count pred_count_logits shape mismatch: {tuple(count_out['pred_count_logits'].shape)}.")

    default_head = _head_from_yaml(DEFAULT_CFG)
    default_out = default_head(_head_features(default_head), orig_size=(544, 960))
    if "pred_count_logits" in default_out:
        raise AssertionError("default query YAML unexpectedly emitted pred_count_logits.")

    ordered_head = _head_from_yaml(ORDERED_CFG)
    ordered_out = ordered_head(_head_features(ordered_head), orig_size=(544, 960))
    if "pred_count_logits" not in ordered_out:
        raise AssertionError("ordered-slot YAML lost pred_count_logits.")
    if getattr(ordered_head, "query_count_head", False):
        raise AssertionError("ordered-slot head must not depend on query_count_head.")
    if tuple(ordered_out["pred_count_logits"].shape) != (2, 4):
        raise AssertionError(f"ordered-slot pred_count_logits shape mismatch: {tuple(ordered_out['pred_count_logits'].shape)}.")


def _make_batch() -> dict:
    k = 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)

    def lanes(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.linspace(0.2, 0.8, n).view(n, 1).expand(n, k)
        pts = torch.stack((xs, y.view(1, k).expand(n, k)), dim=-1).float()
        valid = torch.ones(n, k, dtype=torch.float32)
        return pts, valid

    lanes0, valid0 = lanes(2)
    lanes1, valid1 = lanes(5)
    return {
        "lanes": [lanes0, lanes1],
        "lane_valid": [valid0, valid1],
        "num_lanes": torch.tensor([2, 5], dtype=torch.long),
    }


def _make_preds(include_count: bool) -> dict[str, torch.Tensor]:
    b, q, k = 2, 12, 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k).view(1, 1, k).expand(b, q, k)
    x = torch.linspace(0.05, 0.95, q).view(1, q, 1).expand(b, q, k)
    preds = {
        "pred_points": torch.stack((x, y), dim=-1).contiguous(),
        "pred_logits": torch.zeros(b, q),
        "pred_valid_logits": torch.full((b, q, k), 4.0),
    }
    if include_count:
        preds["pred_count_logits"] = torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            dtype=torch.float32,
        )
    return preds


def check_loss() -> None:
    criterion = GCSLoss({"gcs_imgsz": [544, 960], "gcs_query_count_ce": 0.5})
    batch = _make_batch()

    _, with_items = criterion(_make_preds(include_count=True), batch)
    if int(with_items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(f"GCSLoss item length mismatch: {with_items.numel()} vs {len(GCSLoss.loss_names)}.")
    if not torch.isfinite(with_items).all():
        raise AssertionError("GCSLoss produced non-finite items with pred_count_logits.")
    query_count_indices = [GCSLoss.loss_names.index(name) for name in (
        "query_count_ce_loss",
        "query_count_acc",
        "query_count_pred_mean",
    )]
    if float(with_items[query_count_indices[0]]) <= 0.0:
        raise AssertionError(
            "query_count_ce_loss should be positive with logits present, "
            f"got {float(with_items[query_count_indices[0]])}."
        )

    _, without_items = criterion(_make_preds(include_count=False), batch)
    if int(without_items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(f"GCSLoss no-count item length mismatch: {without_items.numel()} vs {len(GCSLoss.loss_names)}.")
    without_query_count = without_items[query_count_indices]
    if not torch.allclose(without_query_count, torch.zeros_like(without_query_count)):
        raise AssertionError(f"missing pred_count_logits should log zeros, got {without_query_count.tolist()}.")


def check_decode_count_modes() -> None:
    q, k = 5, 8
    y = torch.linspace(0.9, 0.2, k)
    xs = torch.linspace(0.1, 0.9, q)
    pred_points = torch.stack(
        [torch.stack((torch.full((k,), float(x)), y), dim=1) for x in xs],
        dim=0,
    )
    pred_logits = torch.full((q,), _logit(0.95), dtype=torch.float32)
    pred_valid_logits = torch.full((q, k), _logit(0.90), dtype=torch.float32)

    common = {
        "pred_points": pred_points,
        "pred_logits": pred_logits,
        "pred_valid_logits": pred_valid_logits,
        "score_thr": 0.05,
        "point_valid_thr": 0.5,
        "min_points": 2,
        "count_aware_topk": True,
        "count_aware_min_k": 2,
        "count_aware_max_k": 5,
        "count_aware_length_norm": 2,
    }
    default_lanes = decode_gcs_predictions(**common)
    score_sum_lanes = decode_gcs_predictions(**common, count_mode="score_sum")
    if len(default_lanes) != len(score_sum_lanes) or len(score_sum_lanes) != 5:
        raise AssertionError("score_sum count-aware behavior changed.")

    score_sum_logits = torch.tensor([_logit(0.90), _logit(0.85), _logit(0.80), _logit(0.75), _logit(0.10)])
    score_sum_common = dict(common)
    score_sum_common["pred_logits"] = score_sum_logits
    score_sum_base = decode_gcs_predictions(
        **score_sum_common,
        count_mode="score_sum",
    )
    score_sum_margin = decode_gcs_predictions(
        **score_sum_common,
        count_mode="score_sum",
        count_aware_extra_margin=1,
    )
    if len(score_sum_base) != 3 or len(score_sum_margin) != 4:
        raise AssertionError(
            f"score_sum extra margin should keep 3 -> 4 lanes, got {len(score_sum_base)} -> {len(score_sum_margin)}."
        )

    count_logits = torch.tensor([0.0, 5.0, 0.0, 0.0], dtype=torch.float32)
    count_logits_lanes = decode_gcs_predictions(
        **common,
        pred_count_logits=count_logits,
        count_mode="count_logits",
    )
    if len(count_logits_lanes) != 3:
        raise AssertionError(f"count_logits mode should keep k_hat=3 lanes, got {len(count_logits_lanes)}.")
    if any(int(x.get("decoded_count_k", -1)) != 3 for x in count_logits_lanes):
        raise AssertionError("decoded lanes did not record decoded_count_k=3.")

    count_logits_margin_lanes = decode_gcs_predictions(
        **common,
        pred_count_logits=count_logits,
        count_mode="count_logits",
        count_aware_extra_margin=1,
    )
    if len(count_logits_margin_lanes) != 4:
        raise AssertionError(f"count_logits margin=1 should keep k_hat+1=4 lanes, got {len(count_logits_margin_lanes)}.")
    if any(int(x.get("decoded_count_base_k", -1)) != 3 for x in count_logits_margin_lanes):
        raise AssertionError("decoded lanes did not record decoded_count_base_k=3 for count_logits margin.")
    if any(int(x.get("decoded_count_k", -1)) != 4 for x in count_logits_margin_lanes):
        raise AssertionError("decoded lanes did not record decoded_count_k=4 for count_logits margin.")

    count_logits_capped_lanes = decode_gcs_predictions(
        **common,
        pred_count_logits=count_logits,
        count_mode="count_logits",
        count_aware_extra_margin=3,
        max_det=4,
    )
    if len(count_logits_capped_lanes) != 4:
        raise AssertionError(f"count_logits margin must be capped by max_det=4, got {len(count_logits_capped_lanes)}.")
    if any(int(x.get("decoded_count_k", -1)) != 4 for x in count_logits_capped_lanes):
        raise AssertionError("decoded lanes did not record max_det-capped decoded_count_k=4.")

    try:
        decode_gcs_predictions(**common, count_mode="count_logits")
    except ValueError as exc:
        if "requires pred_count_logits" not in str(exc):
            raise
    else:
        raise AssertionError("count_mode='count_logits' must fail without pred_count_logits.")

    default_range_common = dict(common)
    default_range_common.pop("count_aware_min_k")
    default_range_common.pop("count_aware_max_k")
    default_range_lanes = decode_gcs_predictions(
        **default_range_common,
        pred_count_logits=count_logits,
        count_mode="count_logits",
    )
    if len(default_range_lanes) != 3:
        raise AssertionError(
            "count_logits mode should use the fixed 2..5 class mapping even when score_sum defaults are 3..5."
        )


def check_sweep_combos() -> None:
    combos = build_combos(
        Namespace(
            decode_mode="query",
            confs=[0.01],
            point_valid_thrs=[0.5],
            nms_dist_pxs=[0.0],
            max_dets=[5],
            min_points=[2],
            valid_before_maxdet=True,
            count_aware_topk=True,
            count_aware_min_k=2,
            count_aware_max_k=5,
            count_aware_length_norm=12.0,
            count_aware_extra_margins=[0, 1, 2],
            count_modes=["score_sum", "count_logits"],
        )
    )
    modes = {combo["count_mode"] for combo in combos}
    margins = {combo["count_aware_extra_margin"] for combo in combos}
    if modes != {"score_sum", "count_logits"} or margins != {0, 1, 2} or len(combos) != 6:
        raise AssertionError(f"sweep combos did not preserve both count modes: {combos!r}.")

    try:
        build_combos(
            Namespace(
                decode_mode="query",
                confs=[0.01],
                point_valid_thrs=[0.5],
                nms_dist_pxs=[0.0],
                max_dets=[5],
                min_points=[2],
                valid_before_maxdet=False,
                count_aware_topk=False,
                count_aware_min_k=2,
                count_aware_max_k=5,
                count_aware_length_norm=12.0,
                count_aware_extra_margins=[-1],
                count_modes=["score_sum"],
            )
        )
    except ValueError as exc:
        if "extra margins" not in str(exc):
            raise
    else:
        raise AssertionError("negative count-aware extra margins must fail even when count-aware top-k is off.")


def check_decode_yaml_compatibility() -> None:
    old_query_yaml = {
        "schema": "query_decode_v1",
        "decode_mode": "query",
        "conf": 0.003,
        "point_valid_thr": 0.5,
        "nms_dist_px": 0.0,
        "max_det": 6,
        "min_points": 5,
        "valid_before_maxdet": True,
        "count_aware_topk": False,
        "count_aware_min_k": 3,
        "count_aware_max_k": 5,
        "count_aware_length_norm": 12.0,
        "count_mode": "score_sum",
    }
    validate_decode_yaml_for_model(old_query_yaml, model_mode="query")

    bad_query_yaml = dict(old_query_yaml)
    bad_query_yaml["count_mode"] = "oracle_gt"
    try:
        validate_decode_yaml_for_model(bad_query_yaml, model_mode="query")
    except RuntimeError as exc:
        if "Use --oracle-count" not in str(exc):
            raise
    else:
        raise AssertionError("query decode yaml must reject diagnostic-only count_mode='oracle_gt'.")


def main() -> None:
    check_yaml_forward()
    check_loss()
    check_decode_count_modes()
    check_sweep_combos()
    check_decode_yaml_compatibility()
    print(json.dumps({"status": "ok", "loss_items": len(GCSLoss.loss_names)}, indent=2))


if __name__ == "__main__":
    main()
