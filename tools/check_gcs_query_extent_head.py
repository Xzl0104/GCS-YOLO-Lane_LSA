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

from gcs_tools.official_selection import extent_decode_priority, sweep_sort_key  # noqa: E402
from tools.diagnose_tusimple_query_extent_gate import _group_summary, _mode_summary, _raw_match20_strata  # noqa: E402
from tools.sweep_tusimple_official import build_combos  # noqa: E402
from ultralytics.models.gcs.decode_summary import validate_decode_yaml_for_model  # noqa: E402
from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402


DEFAULT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"
EXTENT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-extent.yaml"
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
    default_head = _head_from_yaml(DEFAULT_CFG)
    default_out = default_head(_head_features(default_head), orig_size=(544, 960))
    if "pred_start_logits" in default_out or "pred_end_logits" in default_out:
        raise AssertionError("default query YAML unexpectedly emitted pred_start_logits/pred_end_logits.")
    if "pred_count_logits" in default_out:
        raise AssertionError("default query YAML unexpectedly emitted pred_count_logits.")
    if "pred_quality_logits" in default_out:
        raise AssertionError("default query YAML unexpectedly emitted pred_quality_logits.")

    extent_head = _head_from_yaml(EXTENT_CFG)
    if not getattr(extent_head, "query_extent_head", False):
        raise AssertionError("extent YAML did not enable query_extent_head.")
    extent_out = extent_head(_head_features(extent_head), orig_size=(544, 960))
    if tuple(extent_out.get("pred_start_logits", torch.empty(0)).shape) != (2, 12, 56):
        raise AssertionError(f"pred_start_logits shape mismatch: {tuple(extent_out.get('pred_start_logits', torch.empty(0)).shape)}.")
    if tuple(extent_out.get("pred_end_logits", torch.empty(0)).shape) != (2, 12, 56):
        raise AssertionError(f"pred_end_logits shape mismatch: {tuple(extent_out.get('pred_end_logits', torch.empty(0)).shape)}.")
    if "pred_count_logits" in extent_out:
        raise AssertionError("extent YAML must not emit pred_count_logits.")
    if "pred_quality_logits" in extent_out:
        raise AssertionError("extent YAML must not emit pred_quality_logits.")

    ordered_head = _head_from_yaml(ORDERED_CFG)
    ordered_out = ordered_head(_head_features(ordered_head), orig_size=(544, 960))
    for key in ("pred_count_logits", "pred_start_logits", "pred_end_logits", "pred_exist_logits"):
        if key not in ordered_out:
            raise AssertionError(f"ordered-slot YAML lost {key}.")
    if getattr(ordered_head, "query_extent_head", False):
        raise AssertionError("ordered-slot head must not depend on query_extent_head.")


def _make_batch() -> dict:
    k = 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)

    lanes = []
    valids = []
    for n, start, end in ((2, 0, 55), (5, 3, 9)):
        xs = torch.linspace(0.2, 0.8, n).view(n, 1).expand(n, k)
        pts = torch.stack((xs, y.view(1, k).expand(n, k)), dim=-1).float()
        valid = torch.zeros(n, k, dtype=torch.float32)
        valid[:, start : end + 1] = 1.0
        lanes.append(pts)
        valids.append(valid)
    return {
        "lanes": lanes,
        "lane_valid": valids,
        "num_lanes": torch.tensor([2, 5], dtype=torch.long),
    }


def _make_preds(include_extent: bool) -> dict[str, torch.Tensor]:
    b, q, k = 2, 12, 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k).view(1, 1, k).expand(b, q, k)
    x = torch.linspace(0.05, 0.95, q).view(1, q, 1).expand(b, q, k)
    preds = {
        "pred_points": torch.stack((x, y), dim=-1).contiguous(),
        "pred_logits": torch.zeros(b, q),
        "pred_valid_logits": torch.full((b, q, k), 4.0),
    }
    if include_extent:
        preds["pred_start_logits"] = torch.zeros(b, q, k)
        preds["pred_end_logits"] = torch.zeros(b, q, k)
    return preds


def check_loss() -> None:
    batch = _make_batch()
    names = list(GCSLoss.loss_names)
    query_extent_idx = names.index("query_extent_loss")
    short_count_idx = names.index("query_extent_short_count")

    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_query_extent": 0.5,
            "gcs_query_extent_short_visible_thr": 10,
            "gcs_query_extent_short_weight": 2.0,
            "gcs_query_extent_gt_min_lanes": 4,
        }
    )
    _, items = criterion(_make_preds(include_extent=True), batch)
    if int(items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(f"GCSLoss item length mismatch: {items.numel()} vs {len(GCSLoss.loss_names)}.")
    if not torch.isfinite(items).all():
        raise AssertionError("GCSLoss produced non-finite items with query extent logits.")
    if float(items[query_extent_idx]) <= 0.0:
        raise AssertionError(f"query_extent_loss should be positive, got {float(items[query_extent_idx])}.")
    if float(items[short_count_idx]) <= 0.0:
        raise AssertionError(f"query_extent_short_count should be positive for GT5 short lane, got {float(items[short_count_idx])}.")

    try:
        criterion(_make_preds(include_extent=False), batch)
    except ValueError as exc:
        if "gcs_query_extent requires" not in str(exc):
            raise
    else:
        raise AssertionError("gcs_query_extent > 0 must fail when query extent logits are missing.")

    disabled = GCSLoss({"gcs_imgsz": [544, 960], "gcs_query_extent": 0.0})
    _, disabled_items = disabled(_make_preds(include_extent=False), batch)
    if float(disabled_items[query_extent_idx]) != 0.0:
        raise AssertionError("missing query extent logits with zero gain should log zero query_extent_loss.")


def check_decode_extent_modes() -> None:
    q, k = 1, 8
    y = torch.linspace(0.9, 0.2, k)
    pred_points = torch.stack((torch.full((q, k), 0.5), y.view(1, k).expand(q, k)), dim=-1)
    pred_logits = torch.full((q,), _logit(0.95), dtype=torch.float32)
    pred_valid_logits = torch.full((q, k), _logit(0.05), dtype=torch.float32)
    pred_valid_logits[0, 4:8] = _logit(0.95)
    pred_start_logits = torch.full((q, k), -10.0)
    pred_end_logits = torch.full((q, k), -10.0)
    pred_start_logits[0, 2] = 10.0
    pred_end_logits[0, 5] = 10.0

    interval = decode_gcs_predictions(
        pred_points,
        pred_logits,
        pred_valid_logits=pred_valid_logits,
        pred_start_logits=pred_start_logits,
        pred_end_logits=pred_end_logits,
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=2,
        extent_decode=True,
        extent_decode_mode="interval",
    )
    if len(interval) != 1 or int(sum(interval[0]["point_valid"])) != 4:
        raise AssertionError(f"interval extent decode should keep anchors 2..5, got {interval!r}.")
    if interval[0].get("extent_start_idx") != 2 or interval[0].get("extent_end_idx") != 5:
        raise AssertionError(f"extent diagnostic indexes not recorded correctly: {interval[0]!r}.")

    intersect = decode_gcs_predictions(
        pred_points,
        pred_logits,
        pred_valid_logits=pred_valid_logits,
        pred_start_logits=pred_start_logits,
        pred_end_logits=pred_end_logits,
        score_thr=0.1,
        point_valid_thr=0.5,
        min_points=2,
        extent_decode=True,
        extent_decode_mode="intersect",
    )
    if len(intersect) != 1 or int(sum(intersect[0]["point_valid"])) != 2:
        raise AssertionError(f"intersect extent decode should keep anchors 4..5, got {intersect!r}.")

    try:
        decode_gcs_predictions(
            pred_points,
            pred_logits,
            pred_valid_logits=pred_valid_logits,
            score_thr=0.1,
            extent_decode=True,
            extent_decode_mode="interval",
        )
    except ValueError as exc:
        if "requires pred_start_logits" not in str(exc):
            raise
    else:
        raise AssertionError("extent_decode must fail without pred_start_logits/pred_end_logits.")


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
            extent_decode_modes=["none", "interval", "intersect"],
            count_aware_topk=False,
            count_aware_min_k=2,
            count_aware_max_k=5,
            count_aware_length_norm=12.0,
            count_aware_extra_margins=[0],
            count_modes=["score_sum"],
        )
    )
    modes = {combo["extent_decode_mode"] for combo in combos}
    enabled = {combo["extent_decode_mode"]: combo["extent_decode"] for combo in combos}
    if modes != {"none", "interval", "intersect"} or enabled.get("none") is not False:
        raise AssertionError(f"sweep combos did not preserve extent modes: {combos!r}.")
    if enabled.get("interval") is not True or enabled.get("intersect") is not True:
        raise AssertionError(f"extent sweep modes must enable extent_decode: {combos!r}.")
    priorities = {combo["extent_decode_mode"]: combo.get("extent_decode_priority") for combo in combos}
    if priorities != {"none": 0, "intersect": 1, "interval": 2}:
        raise AssertionError(f"extent sweep priorities must be none < intersect < interval, got {priorities!r}.")
    tied_rows = [
        {
            "official_acc": 0.97,
            "official_score": 0.95,
            "official_FP": 0.01,
            "official_FN": 0.02,
            "count_acc_4": 0.9,
            "count_acc": 0.9,
            "count_acc_5": 0.9,
            "extent_decode_mode": mode,
            "extent_decode_priority": extent_decode_priority(mode),
            "conf": 0.01,
            "nms_dist_px": 0.0,
            "point_valid_thr": 0.5,
            "max_det": 5,
            "min_points": 2,
        }
        for mode in ("interval", "intersect", "none")
    ]
    best_tie = max(tied_rows, key=sweep_sort_key)
    if best_tie["extent_decode_mode"] != "none":
        raise AssertionError(f"official sweep exact ties must prefer no extent decode, got {best_tie!r}.")

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
                extent_decode_modes=["bad"],
                count_aware_topk=False,
                count_aware_min_k=2,
                count_aware_max_k=5,
                count_aware_length_norm=12.0,
                count_aware_extra_margins=[0],
                count_modes=["score_sum"],
            )
        )
    except ValueError as exc:
        if "Unsupported extent decode modes" not in str(exc):
            raise
    else:
        raise AssertionError("unsupported extent decode modes must fail.")


def check_decode_yaml_compatibility() -> None:
    query_yaml = {
        "schema": "query_decode_v1",
        "decode_mode": "query",
        "conf": 0.003,
        "point_valid_thr": 0.5,
        "nms_dist_px": 0.0,
        "max_det": 6,
        "min_points": 5,
        "valid_before_maxdet": True,
        "extent_decode": True,
        "extent_decode_mode": "interval",
        "count_aware_topk": False,
        "count_aware_min_k": 3,
        "count_aware_max_k": 5,
        "count_aware_length_norm": 12.0,
        "count_mode": "score_sum",
    }
    validate_decode_yaml_for_model(query_yaml, model_mode="query")

    bad_query_yaml = dict(query_yaml)
    bad_query_yaml["extent_decode"] = False
    bad_query_yaml["extent_decode_mode"] = "interval"
    try:
        validate_decode_yaml_for_model(bad_query_yaml, model_mode="query")
    except RuntimeError as exc:
        if "extent_decode_mode must be 'none'" not in str(exc):
            raise
    else:
        raise AssertionError("query decode yaml must reject extent_decode=False with interval mode.")


def check_extent_gate_raw_geometry_strata() -> None:
    rows = [
        {
            "raw_has_match_20px": True,
            "extent_start_abs_err": 0,
            "extent_end_abs_err": 1,
            "extent_interval_iou": 0.75,
            "final_best_ape_px_interval": 18.0,
            "final_has_match_20px_interval": True,
            "final_best_ape_px_intersect": 28.0,
            "final_has_match_20px_intersect": False,
            "final_best_ape_px_none": 38.0,
            "final_has_match_20px_none": False,
            "gt_count": 4,
        },
        {
            "raw_has_match_20px": False,
            "extent_start_abs_err": 9,
            "extent_end_abs_err": 8,
            "extent_interval_iou": 0.1,
            "final_best_ape_px_interval": 45.0,
            "final_has_match_20px_interval": False,
            "final_best_ape_px_intersect": 25.0,
            "final_has_match_20px_intersect": False,
            "final_best_ape_px_none": 55.0,
            "final_has_match_20px_none": False,
            "gt_count": 4,
        },
    ]
    mixed = _group_summary(rows, "interval", endpoint_tol=1, match_thr_px=20.0)
    strata = _raw_match20_strata(rows, "interval", endpoint_tol=1, match_thr_px=20.0)
    if mixed["endpoint_start_acc_1"] != 0.5:
        raise AssertionError(f"mixed endpoint summary should include both raw hit/miss rows, got {mixed!r}.")
    raw_hit = strata["raw_has_match20_true"]
    raw_miss = strata["raw_has_match20_false"]
    if raw_hit["count"] != 1 or raw_hit["endpoint_start_acc_1"] != 1.0 or raw_hit["endpoint_end_acc_1"] != 1.0:
        raise AssertionError(f"raw-hit endpoint stratum is wrong: {raw_hit!r}.")
    if raw_miss["count"] != 1 or raw_miss["endpoint_start_acc_1"] != 0.0 or raw_miss["endpoint_end_acc_1"] != 0.0:
        raise AssertionError(f"raw-miss endpoint stratum is wrong: {raw_miss!r}.")

    by_group = {"short_gt4": rows, "short_gt5": []}
    image_rows = [{"gt_count": 4, "pred_count_interval": 4, "pred_count_intersect": 5, "pred_count_none": 3}]
    gt_hist = {4: 1}
    extra_by_gt_mode = {"interval": {3: 0, 4: 0}, "intersect": {3: 0, 4: 1}, "none": {3: 0, 4: 0}}
    interval_summary = _mode_summary(
        mode="interval",
        by_group=by_group,
        lane_rows=rows,
        image_rows=image_rows,
        gt_hist=gt_hist,
        extra_by_gt_mode=extra_by_gt_mode,
        endpoint_tol=1,
        match_thr_px=20.0,
    )
    intersect_summary = _mode_summary(
        mode="intersect",
        by_group=by_group,
        lane_rows=rows,
        image_rows=image_rows,
        gt_hist=gt_hist,
        extra_by_gt_mode=extra_by_gt_mode,
        endpoint_tol=1,
        match_thr_px=20.0,
    )
    if interval_summary["short_gt4_has_match20"] == intersect_summary["short_gt4_has_match20"]:
        raise AssertionError("per-mode extent gate summary must preserve different interval/intersect outcomes.")
    if interval_summary["count_acc_4"] != 1.0 or intersect_summary["count_acc_4"] != 0.0:
        raise AssertionError(
            f"per-mode count summaries are wrong: interval={interval_summary!r}, intersect={intersect_summary!r}."
        )


def check_resume_override_whitelist() -> None:
    text = (ROOT / "ultralytics/engine/trainer.py").read_text(encoding="utf-8")
    expected = [
        "gcs_query_extent",
        "gcs_query_extent_short_visible_thr",
        "gcs_query_extent_short_weight",
        "gcs_query_extent_gt_min_lanes",
        "gcs_extent_decode",
        "gcs_extent_decode_mode",
        "gcs_official_extent_decode_modes",
        "gcs_official_count_modes",
        "gcs_official_count_aware_topk",
        "gcs_official_count_aware_min_k",
        "gcs_official_count_aware_max_k",
        "gcs_official_count_aware_length_norm",
        "gcs_official_count_aware_extra_margins",
    ]
    missing = [key for key in expected if f'"{key}"' not in text]
    if missing:
        raise AssertionError(f"resume override whitelist is missing query extent args: {missing}.")


def check_full_protocol_script() -> None:
    text = (ROOT / "scripts/run_query_extent_env30_full_protocol_v1.sh").read_text(encoding="utf-8")
    expected = [
        'REQUIRED_MODEL="ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-extent.yaml"',
        '[[ "${MODEL}" != "${REQUIRED_MODEL}" ]]',
        "RUN_TESTS=0 bash scripts/run_query_extent_env30_probe40_v1.sh",
        "--decode-yaml",
        "--split test",
        "--allow-test-oracle",
        "gcs_official_extent_decode_modes",
        "official_best_sweep.json",
        "prediction_cache/manifest.json",
        "official_best_v4",
        "GCS_QUERY_EXTENT",
        "OFFICIAL_EXTENT_DECODE_MODES",
        "SWEEP_EXTENT_DECODE_MODES",
        "COUNT_AWARE_TOPK",
        '"paired_test_requested_by_user": True',
        '"do_not_compare_test_to_select_checkpoint": True',
        "run_test_eval official_best",
        "run_test_eval best",
        "run_test_diagnostics_for_checkpoint official_best",
        "run_test_diagnostics_for_checkpoint best",
    ]
    missing = [needle for needle in expected if needle not in text]
    if missing:
        raise AssertionError(f"full protocol script is missing expected query extent contract text: {missing}.")
    for label in ("official_best", "best"):
        eval_pos = text.find(f"run_test_eval {label}")
        diag_pos = text.find(f"run_test_diagnostics_for_checkpoint {label}")
        if eval_pos < 0 or diag_pos < 0 or eval_pos > diag_pos:
            raise AssertionError(f"full protocol script must run TEST eval before TEST diagnostics for {label}.")
    parent_text = (ROOT / "scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh").read_text(
        encoding="utf-8"
    )
    if "--count-aware-extra-margin" not in parent_text:
        raise AssertionError("parent protocol TEST path must preserve selected count_aware_extra_margin.")


def main() -> None:
    check_yaml_forward()
    check_loss()
    check_decode_extent_modes()
    check_sweep_combos()
    check_decode_yaml_compatibility()
    check_extent_gate_raw_geometry_strata()
    check_resume_override_whitelist()
    check_full_protocol_script()
    print(json.dumps({"status": "ok", "loss_items": len(GCSLoss.loss_names)}, indent=2))


if __name__ == "__main__":
    main()
