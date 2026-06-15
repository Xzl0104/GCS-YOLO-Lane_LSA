from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    DEFAULT_OFFICIAL_SCORE_FN_WEIGHT,
    DEFAULT_OFFICIAL_SCORE_FP_WEIGHT,
    TuSimpleOfficialLaneEval,
    default_tusimple_gt_json,
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    normalize_tusimple_gt_record,
    official_metric_score,
    read_tusimple_json_lines,
    tusimple_image_path,
)
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.utils.gcs_postprocess import (  # noqa: E402
    GCS_DEFAULT_MAX_DET,
    count_head_decode_meta,
    decode_gcs_predictions,
    empty_decode_count_state,
    summarize_decode_count_state,
    update_decode_count_state,
)
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = (
    ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_full" / "weights" / "best.pt"
)


def validate_official_sweep_split(split: str, *, context: str = "TuSimple official sweep") -> str:
    """Reject test-set parameter search and return the normalized split."""
    normalized = str(split).strip().lower()
    if normalized == "test":
        raise ValueError(
            f"{context} cannot use --split test for threshold or postprocess selection. "
            "Use --split val for sweeps, then use tools/eval_tusimple_official.py --split test "
            "for one-shot final test evaluation."
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep GCS-YOLO-Lane decode thresholds with TuSimple official Accuracy/FP/FN."
    )
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="TuSimple archive split.")
    parser.add_argument("--gt-json", default=None, help="Official TuSimple GT json-lines file.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS inference shape as H W.")
    parser.add_argument(
        "--confs",
        nargs="+",
        type=float,
        default=[0.05, 0.08, 0.10, 0.12, 0.15],
        help="Candidate-pool existence thresholds to sweep.",
    )
    parser.add_argument(
        "--point-valid-thrs",
        nargs="+",
        type=float,
        default=[0.20, 0.25, 0.30, 0.35],
        help="Candidate-pool per-point visibility thresholds to sweep.",
    )
    parser.add_argument(
        "--nms-dist-pxs",
        nargs="+",
        type=float,
        default=[18.0],
        help="Lane-NMS distances in original-image pixels to sweep.",
    )
    parser.add_argument(
        "--max-dets",
        nargs="+",
        type=int,
        default=[GCS_DEFAULT_MAX_DET],
        help="max_det values to sweep.",
    )
    parser.add_argument(
        "--min-points",
        nargs="+",
        type=int,
        default=[6],
        help="Minimum decoded visible anchors required to keep a lane.",
    )
    parser.add_argument(
        "--rank-min-points",
        nargs="+",
        default=["none"],
        help=(
            "Optional per-selected-rank min_points overrides to sweep, e.g. 'none' '5:5' '5:4'. "
            "Unspecified ranks use the combo's min_points."
        ),
    )
    count_head_group = parser.add_mutually_exclusive_group()
    count_head_group.add_argument("--use-count-head-decode", dest="use_count_head_decode", action="store_true", help="Use explicit Count Head K for final Top-K lane selection.")
    count_head_group.add_argument("--no-count-head-decode", dest="use_count_head_decode", action="store_false", help="Disable Count Head K and use max-det rank selection.")
    parser.set_defaults(use_count_head_decode=True)
    parser.add_argument("--count-head-temp", type=float, default=1.0, help="Temperature for Count Head count=2/3/4/5 softmax.")
    parser.add_argument("--candidate-min-points", type=int, default=5, help="Relaxed candidate-pool visible-anchor floor before final Top-K.")
    rescue_group = parser.add_mutually_exclusive_group()
    rescue_group.add_argument("--enable-rescue-candidate-pool", dest="enable_rescue_candidate_pool", action="store_true", help="Use weaker real-query candidates only when Count Head K exceeds the normal candidate pool.")
    rescue_group.add_argument("--no-enable-rescue-candidate-pool", dest="enable_rescue_candidate_pool", action="store_false", help="Disable the weaker rescue candidate pool.")
    parser.set_defaults(enable_rescue_candidate_pool=True)
    parser.add_argument("--rescue-candidate-conf", type=float, default=0.005, help="Rescue candidate-pool existence threshold for Count Head Top-K decode.")
    parser.add_argument("--rescue-candidate-point-valid-thr", type=float, default=0.08, help="Rescue candidate-pool point-valid threshold for Count Head Top-K decode.")
    parser.add_argument("--rescue-candidate-min-points", type=int, default=4, help="Rescue candidate-pool visible-anchor floor before final Top-K.")
    parser.add_argument("--final-min-points", type=int, default=6, help="Final visible-anchor floor for selected ranks 1-4.")
    parser.add_argument("--fifth-min-points", type=int, default=5, help="Final visible-anchor floor for selected rank 5.")
    parser.add_argument("--line-nms-min-overlap", type=int, default=6, help="Minimum shared visible anchors for lane-NMS duplicate suppression.")
    parser.add_argument("--line-nms-rescue-dist-px", type=float, default=30.0, help="Duplicate distance used when rescuing lanes from pre-NMS candidates.")
    parser.add_argument("--quality-rescue-5th", action=argparse.BooleanOptionalAction, default=True, help="Enable quality-gated fifth-lane rescue when pred_quality_logits are present.")
    parser.add_argument("--quality-rescue-count5-thr", type=float, default=0.70)
    parser.add_argument("--quality-rescue-conf-thr", type=float, default=0.03)
    parser.add_argument("--quality-rescue-mean-valid-thr", type=float, default=0.45)
    parser.add_argument("--quality-rescue-quality-thr", type=float, default=0.55)
    parser.add_argument("--quality-rescue-min-points", type=int, default=5)
    parser.add_argument("--quality-rescue-dist-px", type=float, default=24.0)
    parser.add_argument("--last-lane-rescue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--last-lane-rescue-min-policy-count", type=int, default=4)
    parser.add_argument("--last-lane-rescue-conf-thr", type=float, default=None)
    parser.add_argument("--last-lane-rescue-point-valid-thrs", nargs="+", type=float, default=[0.08])
    parser.add_argument("--last-lane-rescue-min-points", nargs="+", type=int, default=[4])
    parser.add_argument("--last-lane-rescue-mean-valid-thrs", nargs="+", type=float, default=[0.40])
    parser.add_argument("--last-lane-rescue-quality-thrs", nargs="+", type=float, default=[0.50])
    parser.add_argument("--last-lane-rescue-dist-pxs", nargs="+", type=float, default=[24.0])
    parser.add_argument("--edge-last-lane-rescue", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge-rescue-conf-thr", type=float, default=0.02)
    parser.add_argument("--edge-rescue-point-valid-thr", type=float, default=0.06)
    parser.add_argument("--edge-rescue-min-points", type=int, default=4)
    parser.add_argument("--edge-rescue-mean-valid-thr", type=float, default=0.35)
    parser.add_argument("--edge-rescue-quality-thr", type=float, default=0.45)
    parser.add_argument("--edge-rescue-outside-gap-px", type=float, default=28.0)
    parser.add_argument("--edge-rescue-dist-px", type=float, default=24.0)
    parser.add_argument("--edge-rescue-min-policy-count", type=int, default=4)
    parser.add_argument("--edge-count4-to5-upgrade", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-count4-to5-prob-margin", type=float, default=0.20)
    parser.add_argument("--soft-count-decision", action="store_true", help="Choose K by candidate quality when Count Head probabilities are close.")
    parser.add_argument("--soft-count-prob-margin", type=float, default=0.08)
    parser.add_argument("--soft-count-quality-weight", type=float, default=1.0)
    parser.add_argument("--soft-count-prior-weight", type=float, default=0.5)
    parser.add_argument("--soft-count-duplicate-penalty", type=float, default=1.0)
    parser.add_argument("--soft-count-invalid-penalty", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of GT records. 0 means all.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of untimed warmup forwards.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--runtime-ms", type=float, default=1.0, help="Constant TuSimple run_time in ms.")
    parser.add_argument("--save-dir", default=None, help="Output directory. Defaults under the weight run directory.")
    parser.add_argument("--baseline-fp", type=float, default=None, help="Optional baseline FP for soft diagnostic comparison.")
    parser.add_argument("--baseline-fn", type=float, default=None, help="Optional baseline FN for soft diagnostic comparison.")
    parser.add_argument("--fp-tol", type=float, default=0.01, help="Soft diagnostic FP tolerance over --baseline-fp.")
    parser.add_argument("--fn-tol", type=float, default=0.01, help="Soft diagnostic FN tolerance over --baseline-fn.")
    parser.add_argument(
        "--score-fp-weight",
        type=float,
        default=DEFAULT_OFFICIAL_SCORE_FP_WEIGHT,
        help="FP penalty in official_score = official_acc - w_fp * official_fp - w_fn * official_fn.",
    )
    parser.add_argument(
        "--score-fn-weight",
        type=float,
        default=DEFAULT_OFFICIAL_SCORE_FN_WEIGHT,
        help="FN penalty in official_score = official_acc - w_fp * official_fp - w_fn * official_fn.",
    )
    parser.add_argument(
        "--count-acc3-weight",
        type=float,
        default=0.0,
        help="Optional count-accuracy bonus for GT=3 images in official_count_score.",
    )
    parser.add_argument(
        "--count-acc4-weight",
        type=float,
        default=0.006,
        help="Optional count-accuracy bonus for GT=4 images in official_count_score.",
    )
    parser.add_argument(
        "--count-acc5-weight",
        type=float,
        default=0.004,
        help="Optional count-accuracy bonus for GT=5 images in official_count_score.",
    )
    parser.add_argument(
        "--rate-4-to-5-weight",
        type=float,
        default=0.004,
        help="Penalty weight subtracted for GT=4 images decoded as 5 lanes.",
    )
    parser.add_argument(
        "--rate-3-to-5-weight",
        type=float,
        default=0.0025,
        help="Penalty weight subtracted for GT=3 images decoded as 5 lanes.",
    )
    parser.add_argument(
        "--rate-4-to-3-weight",
        type=float,
        default=0.0015,
        help="Penalty weight subtracted for GT=4 images decoded as 3 lanes.",
    )
    parser.add_argument(
        "--rate-3-to-4-weight",
        type=float,
        default=0.001,
        help="Penalty weight subtracted for GT=3 images decoded as 4 lanes.",
    )
    parser.add_argument(
        "--rate-5-to-4-weight",
        type=float,
        default=0.0,
        help="Optional cautious penalty weight subtracted for GT=5 images decoded as 4 lanes.",
    )
    parser.add_argument(
        "--min-count-acc3",
        type=float,
        default=None,
        help="Optional soft diagnostic floor for GT=3 count accuracy. Negative disables.",
    )
    parser.add_argument(
        "--min-count-acc4",
        type=float,
        default=None,
        help="Optional soft diagnostic floor for GT=4 count accuracy. Negative disables.",
    )
    parser.add_argument(
        "--min-count-acc5",
        type=float,
        default=None,
        help="Optional soft diagnostic floor for GT=5 count accuracy. Negative disables.",
    )
    parser.add_argument(
        "--min-gt5-output5-rate",
        type=float,
        default=None,
        help="Optional soft diagnostic floor for GT=5 images whose final decoded output has 5 lanes. Negative disables.",
    )
    parser.add_argument(
        "--max-gt5-count-head-under-rate",
        type=float,
        default=None,
        help="Optional soft diagnostic ceiling for GT=5 images where Count Head policy K is below 5. Negative disables.",
    )
    parser.add_argument(
        "--max-gt5-valid-points-fail-rate",
        type=float,
        default=None,
        help="Optional soft diagnostic ceiling for GT=5 images where Count Head K=5 but final output has fewer than 5 lanes. Negative disables.",
    )
    parser.add_argument(
        "--max-rate-3-to-4",
        type=float,
        default=None,
        help="Optional soft diagnostic ceiling for GT=3 decoded as 4 lanes. Negative disables.",
    )
    parser.add_argument(
        "--max-rate-3-to-5",
        type=float,
        default=None,
        help="Optional soft diagnostic ceiling for GT=3 decoded as 5 lanes. Negative disables.",
    )
    parser.add_argument(
        "--max-rate-4-to-3",
        type=float,
        default=None,
        help="Optional soft diagnostic ceiling for GT=4 decoded as 3 lanes. Negative disables.",
    )
    parser.add_argument(
        "--max-rate-4-to-5",
        type=float,
        default=None,
        help="Optional soft diagnostic ceiling for GT=4 decoded as 5 lanes. Negative disables.",
    )
    parser.add_argument(
        "--max-rate-5-to-4",
        type=float,
        default=None,
        help="Optional soft diagnostic ceiling for GT=5 decoded as 4 lanes. Negative disables.",
    )
    parser.add_argument(
        "--select-best-metric",
        default="official_acc",
        choices=("official_acc", "official_score", "official_count_score", "balanced_score"),
        help="Deprecated compatibility option. Best-row selection always uses official_acc only.",
    )
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fmt_float_for_path(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _fmt_int_values_for_path(name: str, values: list[int]) -> str:
    values = sorted({int(x) for x in values})
    if len(values) == 1:
        return f"{name}{values[0]}"
    return f"{name}{values[0]}-{values[-1]}x{len(values)}"


def parse_rank_min_points(value: str | dict | None) -> dict[int, int] | None:
    """Parse rank min_points overrides like '5:5' or '1-4:6,5:5'."""
    if value is None:
        return None
    if isinstance(value, dict):
        out = {int(k): int(v) for k, v in value.items()}
        return out or None
    text = str(value).strip()
    if text.lower() in {"", "none", "off", "false", "0"}:
        return None
    out: dict[int, int] = {}
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        sep = ":" if ":" in chunk else "=" if "=" in chunk else None
        if sep is None:
            raise ValueError(f"Invalid rank_min_points item {chunk!r}; expected rank:min_points.")
        rank_text, min_points_text = (x.strip().lower().removeprefix("rank") for x in chunk.split(sep, 1))
        min_points = int(min_points_text)
        if "-" in rank_text:
            start_text, end_text = (x.strip().removeprefix("rank") for x in rank_text.split("-", 1))
            ranks = range(int(start_text), int(end_text) + 1)
        else:
            ranks = [int(rank_text)]
        for rank in ranks:
            if rank <= 0 or min_points <= 0:
                raise ValueError(f"rank_min_points values must be positive, got rank={rank}, min_points={min_points}.")
            out[int(rank)] = int(min_points)
    return out or None


def _rank_min_points_tag(config: dict[int, int] | None) -> str:
    if not config:
        return "none"
    return "rankmin" + "-".join(f"r{int(k)}p{int(v)}" for k, v in sorted(config.items()))


def _weight_run_dir(weights: str | Path) -> Path | None:
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def resolve_save_dir(
    save_dir: str | Path | None,
    weights: str | Path,
    split: str,
    max_images: int,
    min_points: list[int],
    rank_min_points: list[str],
) -> Path:
    if save_dir is not None and str(save_dir).strip():
        return Path(save_dir)
    tag = f"official_sweep_{split}_{_fmt_int_values_for_path('minp', min_points)}"
    rank_tags = sorted({_rank_min_points_tag(parse_rank_min_points(x)) for x in rank_min_points})
    if rank_tags != ["none"]:
        tag += f"_rankmin{len(rank_tags)}"
    if max_images and max_images > 0:
        tag += f"_maximg{int(max_images)}"
    run_dir = _weight_run_dir(weights)
    if run_dir is not None:
        return run_dir / "tusimple_official_sweep" / tag
    return ROOT / "runs" / "gcs_lane" / "tusimple_official_sweep" / Path(weights).stem / tag


def sweep_combinations(args: argparse.Namespace) -> list[dict]:
    combos = []
    rank_min_points_options = [parse_rank_min_points(x) for x in args.rank_min_points]
    for (
        conf,
        pvalid,
        nms,
        max_det,
        min_points,
        rank_min_points,
        last_lane_pvalid,
        last_lane_min_points,
        last_lane_mean_valid,
        last_lane_quality,
        last_lane_dist,
    ) in product(
        args.confs,
        args.point_valid_thrs,
        args.nms_dist_pxs,
        args.max_dets,
        args.min_points,
        rank_min_points_options,
        args.last_lane_rescue_point_valid_thrs,
        args.last_lane_rescue_min_points,
        args.last_lane_rescue_mean_valid_thrs,
        args.last_lane_rescue_quality_thrs,
        args.last_lane_rescue_dist_pxs,
    ):
        combos.append(
            {
                "conf": float(conf),
                "point_valid_thr": float(pvalid),
                "nms_dist_px": float(nms),
                "max_det": int(max_det),
                "min_points": int(min_points),
                "rank_min_points": rank_min_points,
                "rank_min_points_tag": _rank_min_points_tag(rank_min_points),
                "last_lane_rescue_point_valid_thr": float(last_lane_pvalid),
                "last_lane_rescue_min_points": int(last_lane_min_points),
                "last_lane_rescue_mean_valid_thr": float(last_lane_mean_valid),
                "last_lane_rescue_quality_thr": float(last_lane_quality),
                "last_lane_rescue_dist_px": float(last_lane_dist),
                "edge_last_lane_rescue": bool(getattr(args, "edge_last_lane_rescue", False)),
                "edge_rescue_conf_thr": float(getattr(args, "edge_rescue_conf_thr", 0.02)),
                "edge_rescue_point_valid_thr": float(getattr(args, "edge_rescue_point_valid_thr", 0.06)),
                "edge_rescue_min_points": int(getattr(args, "edge_rescue_min_points", 4)),
                "edge_rescue_mean_valid_thr": float(getattr(args, "edge_rescue_mean_valid_thr", 0.35)),
                "edge_rescue_quality_thr": float(getattr(args, "edge_rescue_quality_thr", 0.45)),
                "edge_rescue_outside_gap_px": float(getattr(args, "edge_rescue_outside_gap_px", 28.0)),
                "edge_rescue_dist_px": float(getattr(args, "edge_rescue_dist_px", 24.0)),
                "edge_rescue_min_policy_count": int(getattr(args, "edge_rescue_min_policy_count", 4)),
                "edge_count4_to5_upgrade": bool(getattr(args, "edge_count4_to5_upgrade", True)),
                "edge_count4_to5_prob_margin": float(getattr(args, "edge_count4_to5_prob_margin", 0.20)),
            }
        )
    return combos


def empty_state() -> dict:
    return {
        "images": 0,
        "accuracy_sum": 0.0,
        "fp_sum": 0.0,
        "fn_sum": 0.0,
        "pred_lanes_hist": Counter(),
        "gt_lanes_hist": Counter(),
        "gt_pred_lanes_hist": Counter(),
        "gt5_count_head_under": 0,
        "gt5_count_head_k5_output_lt5": 0,
        "candidate_pool_shortfall": 0,
        "gt5_candidate_pool_shortfall": 0,
        "top5_suppressed_by_nms": 0,
        "gt5_top5_suppressed_by_nms": 0,
        "all_pred_quality_sum": 0.0,
        "all_pred_quality_count": 0,
        "matched_pred_quality_sum": 0.0,
        "matched_pred_quality_count": 0,
        "unmatched_pred_quality_sum": 0.0,
        "unmatched_pred_quality_count": 0,
        "rank5_quality_gt5_k4_sum": 0.0,
        "rank5_quality_gt5_k4_count": 0,
        "rescue_attempt_count": 0,
        "rescue_success_count": 0,
        "rescue_tp_count": 0,
        "rescue_fp_count": 0,
        **empty_decode_count_state(),
    }


def matched_tusimple_prediction_indices(tusimple_lanes: list[list[int]], gt: dict) -> set[int]:
    """Return greedily matched prediction indices under the TuSimple official line threshold."""
    gt = normalize_tusimple_gt_record(gt)
    gt_lanes = [list(x) for x in gt["lanes"]]
    y_samples = list(gt["h_samples"])
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, x_gts in enumerate(gt_lanes):
        angle = TuSimpleOfficialLaneEval.get_angle(x_gts, y_samples)
        thresh = TuSimpleOfficialLaneEval.pixel_thresh / max(float(np.cos(angle)), 1e-12)
        for pred_idx, x_preds in enumerate(tusimple_lanes):
            acc = TuSimpleOfficialLaneEval.line_accuracy(x_preds, x_gts, thresh)
            if float(acc) >= float(TuSimpleOfficialLaneEval.pt_thresh):
                candidates.append((float(acc), int(gt_idx), int(pred_idx)))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for _, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
    return used_pred


def _lane_quality_score(lane: dict | None) -> float | None:
    if not lane:
        return None
    value = lane.get("quality_score")
    return None if value is None else float(value)


def update_state(
    state: dict,
    tusimple_lanes: list[list[int]],
    gt: dict,
    runtime_ms: float,
    count_head_meta: dict | None = None,
    decoded_lanes: list[dict] | None = None,
    raw_quality_scores: torch.Tensor | None = None,
) -> None:
    gt = normalize_tusimple_gt_record(gt)
    acc, fp, fn = TuSimpleOfficialLaneEval.bench(
        pred=tusimple_lanes,
        gt=[list(x) for x in gt["lanes"]],
        y_samples=list(gt["h_samples"]),
        running_time=float(runtime_ms),
    )
    gt_count = int(len(gt["lanes"]))
    pred_count = int(len(tusimple_lanes))
    state["images"] += 1
    state["accuracy_sum"] += float(acc)
    state["fp_sum"] += float(fp)
    state["fn_sum"] += float(fn)
    state["pred_lanes_hist"][pred_count] += 1
    state["gt_lanes_hist"][gt_count] += 1
    state["gt_pred_lanes_hist"][(gt_count, pred_count)] += 1
    if raw_quality_scores is not None:
        quality = raw_quality_scores.detach().float().cpu().reshape(-1)
        state["all_pred_quality_sum"] = float(state.get("all_pred_quality_sum", 0.0)) + float(quality.sum().item())
        state["all_pred_quality_count"] = int(state.get("all_pred_quality_count", 0)) + int(quality.numel())
    if count_head_meta and gt_count == 5:
        policy_count = int(count_head_meta["count_head_policy_count"])
        count5_upgrade_eligible = bool(count_head_meta.get("quality_count5_upgrade_eligible", False))
        state["gt5_count_head_under"] = int(state.get("gt5_count_head_under", 0)) + int(
            policy_count < 5 and not count5_upgrade_eligible
        )
        state["gt5_count_head_k5_output_lt5"] = int(state.get("gt5_count_head_k5_output_lt5", 0)) + int(
            (policy_count >= 5 or count5_upgrade_eligible) and pred_count < 5
        )
        if policy_count == 4 and count_head_meta.get("top5_candidate_quality_before_nms") is not None:
            state["rank5_quality_gt5_k4_sum"] = float(state.get("rank5_quality_gt5_k4_sum", 0.0)) + float(
                count_head_meta.get("top5_candidate_quality_before_nms")
            )
            state["rank5_quality_gt5_k4_count"] = int(state.get("rank5_quality_gt5_k4_count", 0)) + 1
    if count_head_meta:
        pool_shortfall = int(int(count_head_meta.get("candidate_pool_shortfall", 0) or 0) > 0)
        nms_suppressed = int(bool(count_head_meta.get("top5_suppressed_by_nms", False)))
        state["candidate_pool_shortfall"] = int(state.get("candidate_pool_shortfall", 0)) + pool_shortfall
        state["top5_suppressed_by_nms"] = int(state.get("top5_suppressed_by_nms", 0)) + nms_suppressed
        state["rescue_attempt_count"] = int(state.get("rescue_attempt_count", 0)) + int(
            count_head_meta.get("quality_rescue_attempt_count", 0) or 0
        )
        state["rescue_success_count"] = int(state.get("rescue_success_count", 0)) + int(
            count_head_meta.get("quality_rescue_success_count", 0) or 0
        )
        state["last_lane_rescue_attempt_count"] = int(state.get("last_lane_rescue_attempt_count", 0)) + int(
            count_head_meta.get("last_lane_rescue_attempt_count", 0) or 0
        )
        state["last_lane_rescue_success_count"] = int(state.get("last_lane_rescue_success_count", 0)) + int(
            count_head_meta.get("last_lane_rescue_success_count", 0) or 0
        )
        state["edge_last_lane_rescue_attempt_count"] = int(state.get("edge_last_lane_rescue_attempt_count", 0)) + int(
            count_head_meta.get("edge_last_lane_rescue_attempt_count", 0) or 0
        )
        state["edge_last_lane_rescue_success_count"] = int(state.get("edge_last_lane_rescue_success_count", 0)) + int(
            count_head_meta.get("edge_last_lane_rescue_success_count", 0) or 0
        )
        state["edge_count4_to5_upgrade_count"] = int(state.get("edge_count4_to5_upgrade_count", 0)) + int(
            bool(count_head_meta.get("edge_count4_to5_upgrade", False))
        )
        if gt_count == 5:
            state["gt5_candidate_pool_shortfall"] = int(state.get("gt5_candidate_pool_shortfall", 0)) + pool_shortfall
            state["gt5_top5_suppressed_by_nms"] = int(state.get("gt5_top5_suppressed_by_nms", 0)) + nms_suppressed
    if decoded_lanes is not None:
        matched_pred = matched_tusimple_prediction_indices(tusimple_lanes, gt)
        aligned = list(decoded_lanes)[: len(tusimple_lanes)]
        for pred_idx, lane in enumerate(aligned):
            quality = _lane_quality_score(lane)
            if quality is None:
                continue
            if pred_idx in matched_pred:
                state["matched_pred_quality_sum"] = float(state.get("matched_pred_quality_sum", 0.0)) + quality
                state["matched_pred_quality_count"] = int(state.get("matched_pred_quality_count", 0)) + 1
            else:
                state["unmatched_pred_quality_sum"] = float(state.get("unmatched_pred_quality_sum", 0.0)) + quality
                state["unmatched_pred_quality_count"] = int(state.get("unmatched_pred_quality_count", 0)) + 1
            if bool(lane.get("quality_rescue_5th", False)):
                if pred_idx in matched_pred:
                    state["rescue_tp_count"] = int(state.get("rescue_tp_count", 0)) + 1
                else:
                    state["rescue_fp_count"] = int(state.get("rescue_fp_count", 0)) + 1
            if bool(lane.get("last_lane_rescue", False)) and not bool(lane.get("edge_last_lane_rescue", False)):
                if pred_idx in matched_pred:
                    state["last_lane_rescue_tp_count"] = int(state.get("last_lane_rescue_tp_count", 0)) + 1
                else:
                    state["last_lane_rescue_fp_count"] = int(state.get("last_lane_rescue_fp_count", 0)) + 1
            if bool(lane.get("edge_last_lane_rescue", False)):
                if pred_idx in matched_pred:
                    state["edge_last_lane_rescue_tp_count"] = int(state.get("edge_last_lane_rescue_tp_count", 0)) + 1
                else:
                    state["edge_last_lane_rescue_fp_count"] = int(state.get("edge_last_lane_rescue_fp_count", 0)) + 1
    update_decode_count_state(state, count_head_meta, pred_count)


def summarize_state(
    combo: dict,
    state: dict,
    score_fp_weight: float = DEFAULT_OFFICIAL_SCORE_FP_WEIGHT,
    score_fn_weight: float = DEFAULT_OFFICIAL_SCORE_FN_WEIGHT,
    count_acc3_weight: float = 0.0,
    count_acc4_weight: float = 0.0,
    count_acc5_weight: float = 0.0,
    rate_4_to_5_weight: float = 0.0,
    rate_3_to_5_weight: float = 0.0,
    rate_4_to_3_weight: float = 0.0,
    rate_3_to_4_weight: float = 0.0,
    rate_5_to_4_weight: float = 0.0,
) -> dict:
    n = max(int(state["images"]), 1)
    def mean_metric(sum_key: str, count_key: str) -> float:
        return float(state.get(sum_key, 0.0)) / max(int(state.get(count_key, 0)), 1)

    official_acc = float(state["accuracy_sum"]) / n
    official_fp = float(state["fp_sum"]) / n
    official_fn = float(state["fn_sum"]) / n
    gt3 = int(state["gt_lanes_hist"].get(3, 0))
    gt4 = int(state["gt_lanes_hist"].get(4, 0))
    gt5 = int(state["gt_lanes_hist"].get(5, 0))
    count_acc_3 = float(state["gt_pred_lanes_hist"].get((3, 3), 0)) / gt3 if gt3 else 0.0
    count_acc_4 = float(state["gt_pred_lanes_hist"].get((4, 4), 0)) / gt4 if gt4 else 0.0
    count_acc_5 = float(state["gt_pred_lanes_hist"].get((5, 5), 0)) / gt5 if gt5 else 0.0
    rate_3_to_4 = float(state["gt_pred_lanes_hist"].get((3, 4), 0)) / gt3 if gt3 else 0.0
    rate_3_to_5 = float(state["gt_pred_lanes_hist"].get((3, 5), 0)) / gt3 if gt3 else 0.0
    rate_4_to_3 = float(state["gt_pred_lanes_hist"].get((4, 3), 0)) / gt4 if gt4 else 0.0
    rate_4_to_5 = float(state["gt_pred_lanes_hist"].get((4, 5), 0)) / gt4 if gt4 else 0.0
    rate_5_to_4 = float(state["gt_pred_lanes_hist"].get((5, 4), 0)) / gt5 if gt5 else 0.0
    gt5_output5_rate = count_acc_5
    gt5_count_head_under_rate = float(state.get("gt5_count_head_under", 0)) / gt5 if gt5 else 0.0
    gt5_valid_points_fail_rate = float(state.get("gt5_count_head_k5_output_lt5", 0)) / gt5 if gt5 else 0.0
    candidate_pool_shortfall_rate = float(state.get("candidate_pool_shortfall", 0)) / n
    top5_suppressed_by_nms_rate = float(state.get("top5_suppressed_by_nms", 0)) / n
    gt5_candidate_pool_shortfall_rate = float(state.get("gt5_candidate_pool_shortfall", 0)) / gt5 if gt5 else 0.0
    gt5_top5_suppressed_by_nms_rate = float(state.get("gt5_top5_suppressed_by_nms", 0)) / gt5 if gt5 else 0.0
    all_pred_quality_mean = mean_metric("all_pred_quality_sum", "all_pred_quality_count")
    matched_pred_quality_mean = mean_metric("matched_pred_quality_sum", "matched_pred_quality_count")
    unmatched_pred_quality_mean = mean_metric("unmatched_pred_quality_sum", "unmatched_pred_quality_count")
    rank5_quality_mean_on_gt5_k4 = mean_metric("rank5_quality_gt5_k4_sum", "rank5_quality_gt5_k4_count")
    rescue_attempt_count = int(state.get("rescue_attempt_count", 0))
    rescue_success_count = int(state.get("rescue_success_count", 0))
    rescue_tp_count = int(state.get("rescue_tp_count", 0))
    rescue_fp_count = int(state.get("rescue_fp_count", 0))
    rescue_pred_count = rescue_tp_count + rescue_fp_count
    rescue_precision = float(rescue_tp_count) / max(rescue_pred_count, 1)
    last_lane_rescue_attempt_count = int(state.get("last_lane_rescue_attempt_count", 0))
    last_lane_rescue_success_count = int(state.get("last_lane_rescue_success_count", 0))
    last_lane_rescue_tp_count = int(state.get("last_lane_rescue_tp_count", 0))
    last_lane_rescue_fp_count = int(state.get("last_lane_rescue_fp_count", 0))
    last_lane_rescue_pred_count = last_lane_rescue_tp_count + last_lane_rescue_fp_count
    last_lane_rescue_precision = float(last_lane_rescue_tp_count) / max(last_lane_rescue_pred_count, 1)
    edge_last_lane_rescue_attempt_count = int(state.get("edge_last_lane_rescue_attempt_count", 0))
    edge_last_lane_rescue_success_count = int(state.get("edge_last_lane_rescue_success_count", 0))
    edge_last_lane_rescue_tp_count = int(state.get("edge_last_lane_rescue_tp_count", 0))
    edge_last_lane_rescue_fp_count = int(state.get("edge_last_lane_rescue_fp_count", 0))
    edge_last_lane_rescue_pred_count = edge_last_lane_rescue_tp_count + edge_last_lane_rescue_fp_count
    edge_last_lane_rescue_precision = float(edge_last_lane_rescue_tp_count) / max(edge_last_lane_rescue_pred_count, 1)
    edge_count4_to5_upgrade_count = int(state.get("edge_count4_to5_upgrade_count", 0))
    official_score = official_metric_score(
        official_acc,
        official_fp,
        official_fn,
        fp_weight=score_fp_weight,
        fn_weight=score_fn_weight,
    )
    official_count_score = (
        official_score
        + float(count_acc4_weight) * count_acc_4
        + float(count_acc5_weight) * count_acc_5
        + float(count_acc3_weight) * count_acc_3
        - float(rate_4_to_5_weight) * rate_4_to_5
        - float(rate_3_to_5_weight) * rate_3_to_5
        - float(rate_4_to_3_weight) * rate_4_to_3
        - float(rate_3_to_4_weight) * rate_3_to_4
        - float(rate_5_to_4_weight) * rate_5_to_4
    )
    row = {
        "conf": round(float(combo["conf"]), 6),
        "point_valid_thr": round(float(combo["point_valid_thr"]), 6),
        "nms_dist_px": round(float(combo["nms_dist_px"]), 6),
        "max_det": int(combo["max_det"]),
        "min_points": int(combo["min_points"]),
        "rank_min_points": combo["rank_min_points_tag"],
        "official_acc": round(official_acc, 6),
        "official_fp": round(official_fp, 6),
        "official_fn": round(official_fn, 6),
        "official_score": round(official_score, 6),
        "count_acc_3": round(count_acc_3, 6),
        "count_acc_4": round(count_acc_4, 6),
        "count_acc_5": round(count_acc_5, 6),
        "rate_3_to_4": round(rate_3_to_4, 6),
        "rate_3_to_5": round(rate_3_to_5, 6),
        "rate_4_to_3": round(rate_4_to_3, 6),
        "rate_4_to_5": round(rate_4_to_5, 6),
        "rate_5_to_4": round(rate_5_to_4, 6),
        "gt5_output5_rate": round(gt5_output5_rate, 6),
        "gt5_count_head_under_rate": round(gt5_count_head_under_rate, 6),
        "gt5_valid_points_fail_rate": round(gt5_valid_points_fail_rate, 6),
        "candidate_pool_shortfall_rate": round(candidate_pool_shortfall_rate, 6),
        "gt5_candidate_pool_shortfall_rate": round(gt5_candidate_pool_shortfall_rate, 6),
        "top5_suppressed_by_nms_rate": round(top5_suppressed_by_nms_rate, 6),
        "gt5_top5_suppressed_by_nms_rate": round(gt5_top5_suppressed_by_nms_rate, 6),
        "all_pred_quality_mean": round(all_pred_quality_mean, 6),
        "matched_pred_quality_mean": round(matched_pred_quality_mean, 6),
        "unmatched_pred_quality_mean": round(unmatched_pred_quality_mean, 6),
        "TP_quality_mean": round(matched_pred_quality_mean, 6),
        "FP_quality_mean": round(unmatched_pred_quality_mean, 6),
        "rank5_quality_mean_on_gt5_k4": round(rank5_quality_mean_on_gt5_k4, 6),
        "rescue_attempt_count": rescue_attempt_count,
        "rescue_success_count": rescue_success_count,
        "rescue_tp_count": rescue_tp_count,
        "rescue_fp_count": rescue_fp_count,
        "rescue_precision": round(rescue_precision, 6),
        "last_lane_rescue_point_valid_thr": round(float(combo.get("last_lane_rescue_point_valid_thr", 0.08)), 6),
        "last_lane_rescue_min_points": int(combo.get("last_lane_rescue_min_points", 4)),
        "last_lane_rescue_mean_valid_thr": round(float(combo.get("last_lane_rescue_mean_valid_thr", 0.40)), 6),
        "last_lane_rescue_quality_thr": round(float(combo.get("last_lane_rescue_quality_thr", 0.50)), 6),
        "last_lane_rescue_dist_px": round(float(combo.get("last_lane_rescue_dist_px", 24.0)), 6),
        "last_lane_rescue_attempt_count": last_lane_rescue_attempt_count,
        "last_lane_rescue_success_count": last_lane_rescue_success_count,
        "last_lane_rescue_tp_count": last_lane_rescue_tp_count,
        "last_lane_rescue_fp_count": last_lane_rescue_fp_count,
        "last_lane_rescue_precision": round(last_lane_rescue_precision, 6),
        "edge_last_lane_rescue_enabled": bool(combo.get("edge_last_lane_rescue", False)),
        "edge_last_lane_rescue_attempt_count": edge_last_lane_rescue_attempt_count,
        "edge_last_lane_rescue_success_count": edge_last_lane_rescue_success_count,
        "edge_last_lane_rescue_tp_count": edge_last_lane_rescue_tp_count,
        "edge_last_lane_rescue_fp_count": edge_last_lane_rescue_fp_count,
        "edge_last_lane_rescue_precision": round(edge_last_lane_rescue_precision, 6),
        "edge_count4_to5_upgrade_count": edge_count4_to5_upgrade_count,
        "balanced_score": round(official_count_score, 6),
        "official_count_score": round(official_count_score, 6),
        "images": int(state["images"]),
        "pred_lanes_hist": {str(k): int(v) for k, v in sorted(state["pred_lanes_hist"].items())},
        "gt_lanes_hist": {str(k): int(v) for k, v in sorted(state["gt_lanes_hist"].items())},
        "gt_pred_lanes_hist": {
            f"{gt}->{pred}": int(v) for (gt, pred), v in sorted(state["gt_pred_lanes_hist"].items())
        },
        "rank_min_points_config": combo["rank_min_points"],
    }
    row.update(summarize_decode_count_state(state, prefix="decode/"))
    return row


def _active_selection_constraints(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Return active soft diagnostic thresholds from CLI/Namespace values."""
    constraints: dict[str, dict[str, float]] = {"min": {}, "max": {}}
    for attr, metric in (
        ("min_count_acc3", "count_acc_3"),
        ("min_count_acc4", "count_acc_4"),
        ("min_count_acc5", "count_acc_5"),
        ("min_gt5_output5_rate", "gt5_output5_rate"),
    ):
        value = getattr(args, attr, None)
        if value is not None and float(value) >= 0.0:
            constraints["min"][metric] = float(value)
    for attr, metric in (
        ("max_rate_3_to_4", "rate_3_to_4"),
        ("max_rate_3_to_5", "rate_3_to_5"),
        ("max_rate_4_to_3", "rate_4_to_3"),
        ("max_rate_4_to_5", "rate_4_to_5"),
        ("max_rate_5_to_4", "rate_5_to_4"),
        ("max_gt5_count_head_under_rate", "gt5_count_head_under_rate"),
        ("max_gt5_valid_points_fail_rate", "gt5_valid_points_fail_rate"),
    ):
        value = getattr(args, attr, None)
        if value is not None and float(value) >= 0.0:
            constraints["max"][metric] = float(value)
    baseline_fp = getattr(args, "baseline_fp", None)
    baseline_fn = getattr(args, "baseline_fn", None)
    if baseline_fp is not None:
        constraints["max"]["official_fp"] = float(baseline_fp) + float(getattr(args, "fp_tol", 0.01))
    if baseline_fn is not None:
        constraints["max"]["official_fn"] = float(baseline_fn) + float(getattr(args, "fn_tol", 0.01))
    return {kind: values for kind, values in constraints.items() if values}


def _selection_constraint_violations(row: dict, constraints: dict[str, dict[str, float]]) -> list[str]:
    """Return human-readable soft diagnostic threshold violations for one sweep row."""
    violations = []
    for metric, threshold in constraints.get("min", {}).items():
        value = float(row.get(metric, 0.0))
        if value < float(threshold):
            violations.append(f"{metric}={value:.6g} < {threshold:.6g}")
    for metric, threshold in constraints.get("max", {}).items():
        value = float(row.get(metric, 0.0))
        if value > float(threshold):
            violations.append(f"{metric}={value:.6g} > {threshold:.6g}")
    return violations


def _best_sort_key(row: dict) -> tuple[float]:
    """Order sweep rows by official Accuracy only."""
    return (float(row["official_acc"]),)


def select_best(rows: list[dict], args: argparse.Namespace) -> dict:
    """Select the first row with maximum official Accuracy; thresholds are diagnostic only."""
    if not rows:
        raise ValueError("Cannot select an official best row from an empty sweep.")
    constraints = _active_selection_constraints(args)
    best = dict(max(rows, key=lambda r: float(r["official_acc"])))
    violations = _selection_constraint_violations(best, constraints)
    best["selection_constraints"] = constraints
    best["selection_constraints_mode"] = "diagnostic_only"
    best["selection_constraints_satisfied"] = not violations
    best["selection_constraint_violations"] = violations
    return best


@torch.inference_mode()
def run_sweep(args: argparse.Namespace) -> dict:
    args.split = validate_official_sweep_split(args.split)
    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_path = Path(args.gt_json) if args.gt_json else default_tusimple_gt_json(archive_root, split=args.split)
    gt_records = read_tusimple_json_lines(gt_path)
    if args.max_images and args.max_images > 0:
        gt_records = gt_records[: int(args.max_images)]
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=imgsz)
    combos = sweep_combinations(args)
    states = [empty_state() for _ in combos]

    if args.warmup > 0 and gt_records:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz=imgsz, device=device, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device)

    total_infer_s = 0.0
    total_post_s = 0.0
    for gt in gt_records:
        raw_file = str(gt["raw_file"])
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        original_shape = (int(img.shape[0]), int(img.shape[1]))
        tensor = preprocess_image(img, imgsz=imgsz, device=device, half=args.half)

        _sync_if_cuda(device)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device)
        total_infer_s += time.perf_counter() - t0

        pred_valid = preds.get("pred_valid_logits")
        pred_count = preds.get("pred_count_logits")
        pred_count_boundary = preds.get("pred_count_boundary_logits")
        pred_quality = preds.get("pred_quality_logits")
        pred_quality_scores = pred_quality[0].detach().float().sigmoid().cpu() if pred_quality is not None else None
        t1 = time.perf_counter()
        for combo, state in zip(combos, states):
            decoded, decode_meta = decode_gcs_predictions(
                preds["pred_points"][0],
                preds["pred_logits"][0],
                pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
                pred_count_logits=pred_count[0] if pred_count is not None else None,
                pred_count_boundary_logits=pred_count_boundary[0] if pred_count_boundary is not None else None,
                pred_quality_logits=pred_quality[0] if pred_quality is not None else None,
                image_shape=original_shape,
                score_thr=combo["conf"],
                point_valid_thr=combo["point_valid_thr"],
                min_points=combo["min_points"],
                max_det=combo["max_det"],
                nms_dist_px=combo["nms_dist_px"],
                count_calibration=None,
                rank_min_points=combo["rank_min_points"],
                use_count_head_decode=bool(args.use_count_head_decode),
                count_head_temperature=float(args.count_head_temp),
                dataset_name="tusimple",
                candidate_score_thr=combo["conf"],
                candidate_point_valid_thr=combo["point_valid_thr"],
                candidate_min_points=int(args.candidate_min_points),
                enable_rescue_candidate_pool=bool(args.enable_rescue_candidate_pool),
                rescue_candidate_score_thr=float(args.rescue_candidate_conf),
                rescue_candidate_point_valid_thr=float(args.rescue_candidate_point_valid_thr),
                rescue_candidate_min_points=int(args.rescue_candidate_min_points),
                final_min_points=int(args.final_min_points),
                fifth_min_points=int(args.fifth_min_points),
                line_nms_min_overlap=int(args.line_nms_min_overlap),
                line_nms_rescue_dist_px=float(args.line_nms_rescue_dist_px),
                quality_rescue_5th=bool(args.quality_rescue_5th),
                quality_rescue_count5_thr=float(args.quality_rescue_count5_thr),
                quality_rescue_conf_thr=float(args.quality_rescue_conf_thr),
                quality_rescue_mean_valid_thr=float(args.quality_rescue_mean_valid_thr),
                quality_rescue_quality_thr=float(args.quality_rescue_quality_thr),
                quality_rescue_min_points=int(args.quality_rescue_min_points),
                quality_rescue_dist_px=float(args.quality_rescue_dist_px),
                last_lane_rescue=bool(args.last_lane_rescue),
                last_lane_rescue_min_policy_count=int(args.last_lane_rescue_min_policy_count),
                last_lane_rescue_conf_thr=args.last_lane_rescue_conf_thr,
                last_lane_rescue_point_valid_thr=float(combo["last_lane_rescue_point_valid_thr"]),
                last_lane_rescue_min_points=int(combo["last_lane_rescue_min_points"]),
                last_lane_rescue_mean_valid_thr=float(combo["last_lane_rescue_mean_valid_thr"]),
                last_lane_rescue_quality_thr=float(combo["last_lane_rescue_quality_thr"]),
                last_lane_rescue_dist_px=float(combo["last_lane_rescue_dist_px"]),
                edge_last_lane_rescue=bool(getattr(args, "edge_last_lane_rescue", False)),
                edge_rescue_conf_thr=float(getattr(args, "edge_rescue_conf_thr", 0.02)),
                edge_rescue_point_valid_thr=float(getattr(args, "edge_rescue_point_valid_thr", 0.06)),
                edge_rescue_min_points=int(getattr(args, "edge_rescue_min_points", 4)),
                edge_rescue_mean_valid_thr=float(getattr(args, "edge_rescue_mean_valid_thr", 0.35)),
                edge_rescue_quality_thr=float(getattr(args, "edge_rescue_quality_thr", 0.45)),
                edge_rescue_outside_gap_px=float(getattr(args, "edge_rescue_outside_gap_px", 28.0)),
                edge_rescue_dist_px=float(getattr(args, "edge_rescue_dist_px", 24.0)),
                edge_rescue_min_policy_count=int(getattr(args, "edge_rescue_min_policy_count", 4)),
                edge_count4_to5_upgrade=bool(getattr(args, "edge_count4_to5_upgrade", True)),
                edge_count4_to5_prob_margin=float(getattr(args, "edge_count4_to5_prob_margin", 0.20)),
                enable_soft_count_decision=bool(args.soft_count_decision),
                soft_count_prob_margin=float(args.soft_count_prob_margin),
                soft_count_quality_weight=float(args.soft_count_quality_weight),
                soft_count_prior_weight=float(args.soft_count_prior_weight),
                soft_count_duplicate_penalty=float(args.soft_count_duplicate_penalty),
                soft_count_invalid_penalty=float(args.soft_count_invalid_penalty),
                return_meta=True,
            )
            tusimple_lanes = gcs_lanes_to_tusimple_lanes(
                decoded,
                h_samples=list(gt["h_samples"]),
                image_shape=original_shape,
            )
            update_state(
                state,
                tusimple_lanes,
                gt,
                runtime_ms=args.runtime_ms,
                count_head_meta=decode_meta,
                decoded_lanes=decoded,
                raw_quality_scores=pred_quality_scores,
            )
        total_post_s += time.perf_counter() - t1

    rows = [
        summarize_state(
            combo,
            state,
            score_fp_weight=args.score_fp_weight,
            score_fn_weight=args.score_fn_weight,
            count_acc3_weight=float(getattr(args, "count_acc3_weight", 0.0)),
            count_acc4_weight=float(getattr(args, "count_acc4_weight", 0.0)),
            count_acc5_weight=float(getattr(args, "count_acc5_weight", 0.0)),
            rate_4_to_5_weight=float(getattr(args, "rate_4_to_5_weight", 0.0)),
            rate_3_to_5_weight=float(getattr(args, "rate_3_to_5_weight", 0.0)),
            rate_4_to_3_weight=float(getattr(args, "rate_4_to_3_weight", 0.0)),
            rate_3_to_4_weight=float(getattr(args, "rate_3_to_4_weight", 0.0)),
            rate_5_to_4_weight=float(getattr(args, "rate_5_to_4_weight", 0.0)),
        )
        for combo, state in zip(combos, states)
    ]
    rows.sort(key=_best_sort_key, reverse=True)
    best = select_best(rows, args)
    n = max(len(gt_records), 1)
    save_dir = resolve_save_dir(
        args.save_dir,
        args.weights,
        args.split,
        args.max_images,
        args.min_points,
        args.rank_min_points,
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "tusimple_official_sweep.csv"
    fieldnames = [
        "conf",
        "point_valid_thr",
        "nms_dist_px",
        "max_det",
        "min_points",
        "rank_min_points",
        "official_acc",
        "official_fp",
        "official_fn",
        "official_score",
        "count_acc_3",
        "count_acc_4",
        "count_acc_5",
        "rate_3_to_4",
        "rate_3_to_5",
        "rate_4_to_3",
        "rate_4_to_5",
        "rate_5_to_4",
        "gt5_output5_rate",
        "gt5_count_head_under_rate",
        "gt5_valid_points_fail_rate",
        "candidate_pool_shortfall_rate",
        "gt5_candidate_pool_shortfall_rate",
        "top5_suppressed_by_nms_rate",
        "gt5_top5_suppressed_by_nms_rate",
        "all_pred_quality_mean",
        "matched_pred_quality_mean",
        "unmatched_pred_quality_mean",
        "TP_quality_mean",
        "FP_quality_mean",
        "rank5_quality_mean_on_gt5_k4",
        "rescue_attempt_count",
        "rescue_success_count",
        "rescue_tp_count",
        "rescue_fp_count",
        "rescue_precision",
        "last_lane_rescue_point_valid_thr",
        "last_lane_rescue_min_points",
        "last_lane_rescue_mean_valid_thr",
        "last_lane_rescue_quality_thr",
        "last_lane_rescue_dist_px",
        "last_lane_rescue_attempt_count",
        "last_lane_rescue_success_count",
        "last_lane_rescue_tp_count",
        "last_lane_rescue_fp_count",
        "last_lane_rescue_precision",
        "balanced_score",
        "official_count_score",
        "images",
        "decode/count_head_k",
        "decode/final_pred_lanes",
        "decode/count_shortfall_rate",
        "decode/candidate_pool_shortfall_rate",
        "decode/top5_suppressed_by_nms_rate",
        "decode/k5_to_output4_rate",
        "decode/k4_to_output5_rate",
        "pred_lanes_hist",
        "gt_lanes_hist",
        "gt_pred_lanes_hist",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], separators=(",", ":")) if key.endswith("_hist") else row[key]
                    for key in fieldnames
                }
            )

    output = {
        "best": best,
        "results": rows,
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "archive_root": str(archive_root.resolve()),
            "split": args.split,
            "gt_json": str(gt_path.resolve()),
            "save_dir": str(save_dir.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "confs": [float(x) for x in args.confs],
            "point_valid_thrs": [float(x) for x in args.point_valid_thrs],
            "nms_dist_pxs": [float(x) for x in args.nms_dist_pxs],
            "max_dets": [int(x) for x in args.max_dets],
            "min_points": [int(x) for x in args.min_points],
            "rank_min_points": list(args.rank_min_points),
            "count_policy": "count_head_topk_no_rule_calibration",
            "use_count_head_decode": bool(args.use_count_head_decode),
            "count_head_temperature": float(args.count_head_temp),
            "candidate_min_points": int(args.candidate_min_points),
            "enable_rescue_candidate_pool": bool(args.enable_rescue_candidate_pool),
            "rescue_candidate_conf": float(args.rescue_candidate_conf),
            "rescue_candidate_point_valid_thr": float(args.rescue_candidate_point_valid_thr),
            "rescue_candidate_min_points": int(args.rescue_candidate_min_points),
            "final_min_points": int(args.final_min_points),
            "fifth_min_points": int(args.fifth_min_points),
            "line_nms_min_overlap": int(args.line_nms_min_overlap),
            "line_nms_rescue_dist_px": float(args.line_nms_rescue_dist_px),
            "quality_rescue_5th": bool(args.quality_rescue_5th),
            "quality_rescue_count5_thr": float(args.quality_rescue_count5_thr),
            "quality_rescue_conf_thr": float(args.quality_rescue_conf_thr),
            "quality_rescue_mean_valid_thr": float(args.quality_rescue_mean_valid_thr),
            "quality_rescue_quality_thr": float(args.quality_rescue_quality_thr),
            "quality_rescue_min_points": int(args.quality_rescue_min_points),
            "quality_rescue_dist_px": float(args.quality_rescue_dist_px),
            "last_lane_rescue": bool(args.last_lane_rescue),
            "last_lane_rescue_min_policy_count": int(args.last_lane_rescue_min_policy_count),
            "last_lane_rescue_conf_thr": (
                None if args.last_lane_rescue_conf_thr is None else float(args.last_lane_rescue_conf_thr)
            ),
            "last_lane_rescue_point_valid_thrs": [float(x) for x in args.last_lane_rescue_point_valid_thrs],
            "last_lane_rescue_min_points": [int(x) for x in args.last_lane_rescue_min_points],
            "last_lane_rescue_mean_valid_thrs": [float(x) for x in args.last_lane_rescue_mean_valid_thrs],
            "last_lane_rescue_quality_thrs": [float(x) for x in args.last_lane_rescue_quality_thrs],
            "last_lane_rescue_dist_pxs": [float(x) for x in args.last_lane_rescue_dist_pxs],
            "soft_count_decision": bool(args.soft_count_decision),
            "soft_count_prob_margin": float(args.soft_count_prob_margin),
            "soft_count_quality_weight": float(args.soft_count_quality_weight),
            "soft_count_prior_weight": float(args.soft_count_prior_weight),
            "soft_count_duplicate_penalty": float(args.soft_count_duplicate_penalty),
            "soft_count_invalid_penalty": float(args.soft_count_invalid_penalty),
            "runtime_ms": float(args.runtime_ms),
            "best_metric": "official_acc",
            "requested_best_metric": str(getattr(args, "select_best_metric", "official_acc")),
            "selection_constraints_mode": "diagnostic_only",
            "official_score_formula": "official_acc - score_fp_weight * official_fp - score_fn_weight * official_fn",
            "official_count_score_formula": (
                "official_score + count_acc4_weight * count_acc_4 + count_acc5_weight * count_acc_5 "
                "+ count_acc3_weight * count_acc_3 - rate_4_to_5_weight * rate_4_to_5 "
                "- rate_3_to_5_weight * rate_3_to_5 - rate_4_to_3_weight * rate_4_to_3 "
                "- rate_3_to_4_weight * rate_3_to_4 - rate_5_to_4_weight * rate_5_to_4"
            ),
            "score_fp_weight": float(args.score_fp_weight),
            "score_fn_weight": float(args.score_fn_weight),
            "count_acc3_weight": float(getattr(args, "count_acc3_weight", 0.0)),
            "count_acc4_weight": float(getattr(args, "count_acc4_weight", 0.0)),
            "count_acc5_weight": float(getattr(args, "count_acc5_weight", 0.0)),
            "rate_4_to_5_weight": float(getattr(args, "rate_4_to_5_weight", 0.0)),
            "rate_3_to_5_weight": float(getattr(args, "rate_3_to_5_weight", 0.0)),
            "rate_4_to_3_weight": float(getattr(args, "rate_4_to_3_weight", 0.0)),
            "rate_3_to_4_weight": float(getattr(args, "rate_3_to_4_weight", 0.0)),
            "rate_5_to_4_weight": float(getattr(args, "rate_5_to_4_weight", 0.0)),
            "selection_constraints": _active_selection_constraints(args),
            "max_images": int(args.max_images),
            "device": str(args.device),
            "half": bool(args.half),
            "avg_inference_ms": round(total_infer_s * 1000.0 / n, 4),
            "avg_sweep_postprocess_ms": round(total_post_s * 1000.0 / n, 4),
        },
    }
    (save_dir / "tusimple_official_sweep_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"swept {len(rows)} combinations on {len(gt_records)} images")
    print(f"saved to: {save_dir.resolve()}")
    return output


def main() -> None:
    args = parse_args()
    try:
        args.split = validate_official_sweep_split(args.split)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    for max_det in sorted({int(x) for x in args.max_dets}):
        warn_max_det_mismatch(args.weights, max_det=max_det, context="TuSimple official sweep")
    run_sweep(args)


if __name__ == "__main__":
    main()
