from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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
    write_tusimple_predictions,
)
from tools.infer_gcs import (  # noqa: E402
    count_calibration_from_args,
    count_head_decode_kwargs_from_args,
    load_gcs_model,
    preprocess_image,
    warn_max_det_mismatch,
)
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.gcs_postprocess import (  # noqa: E402
    GCS_DEFAULT_MAX_DET,
    count_head_decode_meta,
    decode_gcs_predictions,
    empty_decode_count_state,
    summarize_decode_count_state,
    update_decode_count_state,
)
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = (
    ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_full" / "weights" / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GCS-YOLO-Lane with the TuSimple official Accuracy/FP/FN protocol."
    )
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="test", choices=("test", "train", "val"), help="TuSimple archive split.")
    parser.add_argument("--gt-json", default=None, help="Official TuSimple GT json-lines file. Defaults to archive test labels.")
    parser.add_argument("--pred-json", default=None, help="Evaluate an existing TuSimple-format prediction json-lines file.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt used when --pred-json is not set.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS inference shape as H W. Defaults: TuSimple 544 960.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Lane existence confidence threshold.")
    parser.add_argument(
        "--point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold for fixed-y lane decoding.",
    )
    parser.add_argument("--nms-dist-px", type=float, default=18.0, help="Lane-NMS distance in original-image pixels. 0 disables.")
    parser.add_argument(
        "--max-det",
        type=int,
        default=GCS_DEFAULT_MAX_DET,
        help="Maximum decoded lane queries kept before official evaluation.",
    )
    parser.add_argument("--min-points", type=int, default=6, help="Minimum decoded visible anchors required to keep a lane.")
    parser.add_argument(
        "--rank-min-points",
        default="",
        help=(
            "Optional per-selected-rank min_points overrides, e.g. '5:5' or '1-4:6,5:5'. "
            "Ranks not listed use --min-points."
        ),
    )
    count_head_group = parser.add_mutually_exclusive_group()
    count_head_group.add_argument("--use-count-head-decode", dest="use_count_head_decode", action="store_true", help="Use explicit Count Head K for final Top-K lane selection.")
    count_head_group.add_argument("--no-count-head-decode", dest="use_count_head_decode", action="store_false", help="Disable Count Head K and use max-det rank selection.")
    parser.set_defaults(use_count_head_decode=True)
    parser.add_argument("--count-head-temp", type=float, default=1.0, help="Temperature for Count Head count=2/3/4/5 softmax.")
    parser.add_argument("--candidate-conf", type=float, default=0.05, help="Relaxed candidate-pool existence threshold for Count Head Top-K decode.")
    parser.add_argument("--candidate-point-valid-thr", type=float, default=0.20, help="Relaxed candidate-pool point-valid threshold for Count Head Top-K decode.")
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
    parser.add_argument("--last-lane-rescue-point-valid-thr", type=float, default=0.08)
    parser.add_argument("--last-lane-rescue-min-points", type=int, default=4)
    parser.add_argument("--last-lane-rescue-mean-valid-thr", type=float, default=0.40)
    parser.add_argument("--last-lane-rescue-quality-thr", type=float, default=0.50)
    parser.add_argument("--last-lane-rescue-dist-px", type=float, default=24.0)
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
    parser.add_argument(
        "--runtime-ms",
        type=float,
        default=1.0,
        help="Constant run_time written to TuSimple predictions unless --use-measured-runtime is set.",
    )
    parser.add_argument(
        "--use-measured-runtime",
        action="store_true",
        help="Use measured inference+postprocess time as official run_time, including the >200 ms penalty.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help=(
            "Output directory. Defaults to a parameter-specific folder under the weight run directory, "
            "or next to --pred-json when evaluating existing predictions."
        ),
    )
    parser.add_argument("--save-records", action="store_true", help="Save per-image official metric records.")
    parser.add_argument(
        "--score-fp-weight",
        type=float,
        default=DEFAULT_OFFICIAL_SCORE_FP_WEIGHT,
        help="FP penalty in official_score = Accuracy - w_fp * FP - w_fn * FN.",
    )
    parser.add_argument(
        "--score-fn-weight",
        type=float,
        default=DEFAULT_OFFICIAL_SCORE_FN_WEIGHT,
        help="FN penalty in official_score = Accuracy - w_fp * FP - w_fn * FN.",
    )
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_prediction_counts(pred_records: list[dict], gt_records: list[dict]) -> dict:
    gt_by_raw = {str(x["raw_file"]): normalize_tusimple_gt_record(x) for x in gt_records}
    pred_counts = [len(x["lanes"]) for x in pred_records]
    gt_counts = [len(gt_by_raw[str(x["raw_file"])]["lanes"]) for x in pred_records if str(x["raw_file"]) in gt_by_raw]
    return {
        "pred_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(pred_counts).items())},
        "gt_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(gt_counts).items())},
        "gt_pred_lanes_hist": {
            f"{gt}->{pred}": int(count)
            for (gt, pred), count in sorted(Counter(zip(gt_counts, pred_counts)).items())
        },
    }


def _fmt_float_for_path(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _count_calibration_tag(config: dict | None) -> str | None:
    """Deprecated: score-gap count calibration has been removed."""
    if not config:
        return None
    raise ValueError("Score-gap count calibration has been removed. Use Count Head Top-K decode instead.")


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
            raise ValueError(f"Invalid --rank-min-points item {chunk!r}; expected rank:min_points.")
        rank_text, min_points_text = (x.strip().lower().removeprefix("rank") for x in chunk.split(sep, 1))
        min_points = int(min_points_text)
        if "-" in rank_text:
            start_text, end_text = (x.strip().removeprefix("rank") for x in rank_text.split("-", 1))
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid --rank-min-points range {rank_text!r}.")
            ranks = range(start, end + 1)
        else:
            ranks = [int(rank_text)]
        for rank in ranks:
            if rank <= 0 or min_points <= 0:
                raise ValueError(f"--rank-min-points values must be positive, got rank={rank}, min_points={min_points}.")
            out[int(rank)] = int(min_points)
    return out or None


def _rank_min_points_tag(config: dict[int, int] | None) -> str | None:
    if not config:
        return None
    return "rankmin" + "-".join(f"r{int(k)}p{int(v)}" for k, v in sorted(config.items()))


def _weight_run_dir(weights: str | Path) -> Path | None:
    """Return runs/gcs_lane/<name> for a conventional .../weights/best.pt path."""
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def resolve_save_dir(
    save_dir: str | Path | None,
    weights: str | Path,
    pred_json: str | Path | None,
    split: str,
    conf: float,
    point_valid_thr: float,
    nms_dist_px: float,
    max_det: int,
    min_points: int = 6,
    max_images: int = 0,
    use_measured_runtime: bool = False,
    count_calibration: dict | None = None,
    rank_min_points: dict[int, int] | None = None,
    candidate_score_thr: float | None = None,
    candidate_point_valid_thr: float | None = None,
) -> Path:
    """Resolve the result directory without mixing different runs or threshold settings."""
    if save_dir is not None and str(save_dir).strip():
        return Path(save_dir)

    tag_parts = [
        f"official_{split}",
        f"conf{_fmt_float_for_path(conf)}",
        f"pvalid{_fmt_float_for_path(point_valid_thr)}",
        f"candconf{_fmt_float_for_path(conf if candidate_score_thr is None else candidate_score_thr)}",
        f"candpvalid{_fmt_float_for_path(point_valid_thr if candidate_point_valid_thr is None else candidate_point_valid_thr)}",
        f"nms{_fmt_float_for_path(nms_dist_px)}",
        f"maxdet{int(max_det)}",
        f"minp{int(min_points)}",
    ]
    calibration_tag = _count_calibration_tag(count_calibration)
    if calibration_tag:
        tag_parts.append(calibration_tag)
    rank_min_tag = _rank_min_points_tag(rank_min_points)
    if rank_min_tag:
        tag_parts.append(rank_min_tag)
    if max_images and max_images > 0:
        tag_parts.append(f"maximg{int(max_images)}")
    if use_measured_runtime:
        tag_parts.append("measured_runtime")
    tag = "_".join(tag_parts)

    if pred_json:
        return Path(pred_json).parent / "tusimple_official_eval" / tag

    run_dir = _weight_run_dir(weights)
    if run_dir is not None:
        return run_dir / "tusimple_official_eval" / tag

    return ROOT / "runs" / "gcs_lane" / "tusimple_official_eval" / Path(weights).stem / tag


@torch.inference_mode()
def predict_tusimple_records(
    weights: str | Path,
    gt_records: list[dict],
    archive_root: str | Path,
    split: str,
    imgsz: tuple[int, int],
    conf: float,
    point_valid_thr: float,
    nms_dist_px: float,
    max_det: int,
    min_points: int,
    max_images: int,
    warmup: int,
    device: str,
    half: bool,
    runtime_ms: float,
    use_measured_runtime: bool,
    count_calibration: dict | None,
    rank_min_points: dict[int, int] | None,
    use_count_head_decode: bool = True,
    count_head_temperature: float = 1.0,
    dataset_name: str = "tusimple",
    candidate_score_thr: float = 0.05,
    candidate_point_valid_thr: float = 0.20,
    candidate_min_points: int = 5,
    enable_rescue_candidate_pool: bool = True,
    rescue_candidate_score_thr: float = 0.005,
    rescue_candidate_point_valid_thr: float = 0.08,
    rescue_candidate_min_points: int = 4,
    final_min_points: int = 6,
    fifth_min_points: int = 5,
    line_nms_min_overlap: int = 6,
    line_nms_rescue_dist_px: float = 30.0,
    quality_rescue_5th: bool = True,
    quality_rescue_count5_thr: float = 0.70,
    quality_rescue_conf_thr: float = 0.03,
    quality_rescue_mean_valid_thr: float = 0.45,
    quality_rescue_quality_thr: float = 0.55,
    quality_rescue_min_points: int = 5,
    quality_rescue_dist_px: float = 24.0,
    last_lane_rescue: bool = False,
    last_lane_rescue_min_policy_count: int = 4,
    last_lane_rescue_conf_thr: float | None = None,
    last_lane_rescue_point_valid_thr: float = 0.08,
    last_lane_rescue_min_points: int = 4,
    last_lane_rescue_mean_valid_thr: float = 0.40,
    last_lane_rescue_quality_thr: float = 0.50,
    last_lane_rescue_dist_px: float = 24.0,
    edge_last_lane_rescue: bool = False,
    edge_rescue_conf_thr: float = 0.02,
    edge_rescue_point_valid_thr: float = 0.06,
    edge_rescue_min_points: int = 4,
    edge_rescue_mean_valid_thr: float = 0.35,
    edge_rescue_quality_thr: float = 0.45,
    edge_rescue_outside_gap_px: float = 28.0,
    edge_rescue_dist_px: float = 24.0,
    edge_rescue_min_policy_count: int = 4,
    edge_count4_to5_upgrade: bool = True,
    edge_count4_to5_prob_margin: float = 0.20,
    enable_soft_count_decision: bool = False,
    soft_count_prob_margin: float = 0.08,
    soft_count_quality_weight: float = 1.0,
    soft_count_prior_weight: float = 0.5,
    soft_count_duplicate_penalty: float = 1.0,
    soft_count_invalid_penalty: float = 1.0,
) -> tuple[list[dict], dict]:
    """Run GCS inference and export predictions in official TuSimple json-line format."""
    records = gt_records[: int(max_images)] if max_images and max_images > 0 else gt_records
    device_obj = select_device(device, verbose=False)
    model = load_gcs_model(weights, device=device_obj, half=half, gcs_imgsz=imgsz)

    if warmup > 0 and records:
        warm_path = tusimple_image_path(archive_root, str(records[0]["raw_file"]), split=split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz=imgsz, device=device_obj, half=half)
        for _ in range(int(warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    predictions: list[dict] = []
    total_infer_s = 0.0
    total_post_s = 0.0
    measured_times_ms: list[float] = []
    decode_count_state = empty_decode_count_state()

    for gt in records:
        raw_file = str(gt["raw_file"])
        image_path = tusimple_image_path(archive_root, raw_file, split=split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        original_shape = (int(img.shape[0]), int(img.shape[1]))

        tensor = preprocess_image(img, imgsz=imgsz, device=device_obj, half=half)
        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        infer_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred_valid = preds.get("pred_valid_logits")
        pred_count = preds.get("pred_count_logits")
        pred_count_boundary = preds.get("pred_count_boundary_logits")
        pred_quality = preds.get("pred_quality_logits")
        decoded, decode_meta = decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
            pred_count_logits=pred_count[0] if pred_count is not None else None,
            pred_count_boundary_logits=pred_count_boundary[0] if pred_count_boundary is not None else None,
            pred_quality_logits=pred_quality[0] if pred_quality is not None else None,
            image_shape=original_shape,
            score_thr=conf,
            point_valid_thr=point_valid_thr,
            min_points=min_points,
            max_det=max_det,
            nms_dist_px=nms_dist_px,
            count_calibration=count_calibration,
            rank_min_points=rank_min_points,
            use_count_head_decode=use_count_head_decode,
            count_head_temperature=count_head_temperature,
            dataset_name=dataset_name,
            candidate_score_thr=candidate_score_thr,
            candidate_point_valid_thr=candidate_point_valid_thr,
            candidate_min_points=candidate_min_points,
            enable_rescue_candidate_pool=enable_rescue_candidate_pool,
            rescue_candidate_score_thr=rescue_candidate_score_thr,
            rescue_candidate_point_valid_thr=rescue_candidate_point_valid_thr,
            rescue_candidate_min_points=rescue_candidate_min_points,
            final_min_points=final_min_points,
            fifth_min_points=fifth_min_points,
            line_nms_min_overlap=line_nms_min_overlap,
            line_nms_rescue_dist_px=line_nms_rescue_dist_px,
            quality_rescue_5th=quality_rescue_5th,
            quality_rescue_count5_thr=quality_rescue_count5_thr,
            quality_rescue_conf_thr=quality_rescue_conf_thr,
            quality_rescue_mean_valid_thr=quality_rescue_mean_valid_thr,
            quality_rescue_quality_thr=quality_rescue_quality_thr,
            quality_rescue_min_points=quality_rescue_min_points,
            quality_rescue_dist_px=quality_rescue_dist_px,
            last_lane_rescue=last_lane_rescue,
            last_lane_rescue_min_policy_count=last_lane_rescue_min_policy_count,
            last_lane_rescue_conf_thr=last_lane_rescue_conf_thr,
            last_lane_rescue_point_valid_thr=last_lane_rescue_point_valid_thr,
            last_lane_rescue_min_points=last_lane_rescue_min_points,
            last_lane_rescue_mean_valid_thr=last_lane_rescue_mean_valid_thr,
            last_lane_rescue_quality_thr=last_lane_rescue_quality_thr,
            last_lane_rescue_dist_px=last_lane_rescue_dist_px,
            edge_last_lane_rescue=edge_last_lane_rescue,
            edge_rescue_conf_thr=edge_rescue_conf_thr,
            edge_rescue_point_valid_thr=edge_rescue_point_valid_thr,
            edge_rescue_min_points=edge_rescue_min_points,
            edge_rescue_mean_valid_thr=edge_rescue_mean_valid_thr,
            edge_rescue_quality_thr=edge_rescue_quality_thr,
            edge_rescue_outside_gap_px=edge_rescue_outside_gap_px,
            edge_rescue_dist_px=edge_rescue_dist_px,
            edge_rescue_min_policy_count=edge_rescue_min_policy_count,
            edge_count4_to5_upgrade=edge_count4_to5_upgrade,
            edge_count4_to5_prob_margin=edge_count4_to5_prob_margin,
            enable_soft_count_decision=enable_soft_count_decision,
            soft_count_prob_margin=soft_count_prob_margin,
            soft_count_quality_weight=soft_count_quality_weight,
            soft_count_prior_weight=soft_count_prior_weight,
            soft_count_duplicate_penalty=soft_count_duplicate_penalty,
            soft_count_invalid_penalty=soft_count_invalid_penalty,
            return_meta=True,
        )
        tusimple_lanes = gcs_lanes_to_tusimple_lanes(
            decoded,
            h_samples=list(gt["h_samples"]),
            image_shape=original_shape,
        )
        update_decode_count_state(decode_count_state, decode_meta, len(tusimple_lanes))
        post_s = time.perf_counter() - t1

        measured_ms = (infer_s + post_s) * 1000.0
        predictions.append(
            {
                "raw_file": raw_file,
                "lanes": tusimple_lanes,
                "run_time": round(float(measured_ms if use_measured_runtime else runtime_ms), 3),
            }
        )
        total_infer_s += infer_s
        total_post_s += post_s
        measured_times_ms.append(float(measured_ms))

    n = max(len(records), 1)
    timing = {
        "images": len(records),
        "avg_inference_ms": round(total_infer_s * 1000.0 / n, 4),
        "avg_postprocess_ms": round(total_post_s * 1000.0 / n, 4),
        "avg_measured_runtime_ms": round(float(np.mean(measured_times_ms)), 4) if measured_times_ms else None,
        "max_measured_runtime_ms": round(float(np.max(measured_times_ms)), 4) if measured_times_ms else None,
        "official_runtime_source": "measured" if use_measured_runtime else "constant",
        "official_runtime_ms": None if use_measured_runtime else float(runtime_ms),
    }
    timing.update(summarize_decode_count_state(decode_count_state, prefix="decode/"))
    return predictions, timing


def evaluate_tusimple_official(
    weights: str | Path = DEFAULT_WEIGHTS,
    archive_root: str | Path = DEFAULT_ARCHIVE,
    split: str = "test",
    gt_json: str | Path | None = None,
    pred_json: str | Path | None = None,
    imgsz: int | tuple[int, int] | list[int] = (544, 960),
    conf: float = 0.25,
    point_valid_thr: float = 0.5,
    nms_dist_px: float = 0.0,
    max_det: int = GCS_DEFAULT_MAX_DET,
    min_points: int = 6,
    max_images: int = 0,
    warmup: int = 20,
    device: str = "0",
    half: bool = False,
    runtime_ms: float = 1.0,
    use_measured_runtime: bool = False,
    save_dir: str | Path | None = None,
    save_records: bool = False,
    count_calibration: dict | None = None,
    rank_min_points: dict[int, int] | str | None = None,
    use_count_head_decode: bool = True,
    count_head_temperature: float = 1.0,
    dataset_name: str = "tusimple",
    candidate_score_thr: float = 0.05,
    candidate_point_valid_thr: float = 0.20,
    candidate_min_points: int = 5,
    enable_rescue_candidate_pool: bool = True,
    rescue_candidate_score_thr: float = 0.005,
    rescue_candidate_point_valid_thr: float = 0.08,
    rescue_candidate_min_points: int = 4,
    final_min_points: int = 6,
    fifth_min_points: int = 5,
    line_nms_min_overlap: int = 6,
    line_nms_rescue_dist_px: float = 30.0,
    quality_rescue_5th: bool = True,
    quality_rescue_count5_thr: float = 0.70,
    quality_rescue_conf_thr: float = 0.03,
    quality_rescue_mean_valid_thr: float = 0.45,
    quality_rescue_quality_thr: float = 0.55,
    quality_rescue_min_points: int = 5,
    quality_rescue_dist_px: float = 24.0,
    last_lane_rescue: bool = False,
    last_lane_rescue_min_policy_count: int = 4,
    last_lane_rescue_conf_thr: float | None = None,
    last_lane_rescue_point_valid_thr: float = 0.08,
    last_lane_rescue_min_points: int = 4,
    last_lane_rescue_mean_valid_thr: float = 0.40,
    last_lane_rescue_quality_thr: float = 0.50,
    last_lane_rescue_dist_px: float = 24.0,
    edge_last_lane_rescue: bool = False,
    edge_rescue_conf_thr: float = 0.02,
    edge_rescue_point_valid_thr: float = 0.06,
    edge_rescue_min_points: int = 4,
    edge_rescue_mean_valid_thr: float = 0.35,
    edge_rescue_quality_thr: float = 0.45,
    edge_rescue_outside_gap_px: float = 28.0,
    edge_rescue_dist_px: float = 24.0,
    edge_rescue_min_policy_count: int = 4,
    edge_count4_to5_upgrade: bool = True,
    edge_count4_to5_prob_margin: float = 0.20,
    enable_soft_count_decision: bool = False,
    soft_count_prob_margin: float = 0.08,
    soft_count_quality_weight: float = 1.0,
    soft_count_prior_weight: float = 0.5,
    soft_count_duplicate_penalty: float = 1.0,
    soft_count_invalid_penalty: float = 1.0,
    score_fp_weight: float = DEFAULT_OFFICIAL_SCORE_FP_WEIGHT,
    score_fn_weight: float = DEFAULT_OFFICIAL_SCORE_FN_WEIGHT,
) -> dict:
    """Evaluate either a GCS checkpoint or an existing prediction file with TuSimple official metrics."""
    archive_root = find_tusimple_archive_root(archive_root)
    gt_path = Path(gt_json) if gt_json else default_tusimple_gt_json(archive_root, split=split)
    gt_records = read_tusimple_json_lines(gt_path)
    if max_images and max_images > 0:
        gt_records = gt_records[: int(max_images)]

    imgsz = normalize_imgsz(imgsz, dataset="tusimple")
    rank_min_points = parse_rank_min_points(rank_min_points)
    save_dir = resolve_save_dir(
        save_dir=save_dir,
        weights=weights,
        pred_json=pred_json,
        split=split,
        conf=conf,
        point_valid_thr=point_valid_thr,
        nms_dist_px=nms_dist_px,
        max_det=max_det,
        min_points=min_points,
        max_images=max_images,
        use_measured_runtime=use_measured_runtime,
        count_calibration=count_calibration,
        rank_min_points=rank_min_points,
        candidate_score_thr=candidate_score_thr,
        candidate_point_valid_thr=candidate_point_valid_thr,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_out_path = save_dir / "tusimple_predictions.json"
    input_pred_json_path = Path(pred_json).resolve() if pred_json else None

    if pred_json:
        pred_records = read_tusimple_json_lines(pred_json)
        if max_images and max_images > 0:
            wanted = {str(x["raw_file"]) for x in gt_records}
            pred_records = [x for x in pred_records if str(x.get("raw_file")) in wanted]
        timing = None
        write_tusimple_predictions(pred_out_path, pred_records)
    else:
        pred_records, timing = predict_tusimple_records(
            weights=weights,
            gt_records=gt_records,
            archive_root=archive_root,
            split=split,
            imgsz=imgsz,
            conf=conf,
            point_valid_thr=point_valid_thr,
            nms_dist_px=nms_dist_px,
            max_det=max_det,
            min_points=min_points,
            max_images=0,
            warmup=warmup,
            device=device,
            half=half,
            runtime_ms=runtime_ms,
            use_measured_runtime=use_measured_runtime,
            count_calibration=count_calibration,
            rank_min_points=rank_min_points,
            use_count_head_decode=use_count_head_decode,
            count_head_temperature=count_head_temperature,
            dataset_name=dataset_name,
            candidate_score_thr=candidate_score_thr,
            candidate_point_valid_thr=candidate_point_valid_thr,
            candidate_min_points=candidate_min_points,
            enable_rescue_candidate_pool=enable_rescue_candidate_pool,
            rescue_candidate_score_thr=rescue_candidate_score_thr,
            rescue_candidate_point_valid_thr=rescue_candidate_point_valid_thr,
            rescue_candidate_min_points=rescue_candidate_min_points,
            final_min_points=final_min_points,
            fifth_min_points=fifth_min_points,
            line_nms_min_overlap=line_nms_min_overlap,
            line_nms_rescue_dist_px=line_nms_rescue_dist_px,
            quality_rescue_5th=quality_rescue_5th,
            quality_rescue_count5_thr=quality_rescue_count5_thr,
            quality_rescue_conf_thr=quality_rescue_conf_thr,
            quality_rescue_mean_valid_thr=quality_rescue_mean_valid_thr,
            quality_rescue_quality_thr=quality_rescue_quality_thr,
            quality_rescue_min_points=quality_rescue_min_points,
            quality_rescue_dist_px=quality_rescue_dist_px,
            last_lane_rescue=last_lane_rescue,
            last_lane_rescue_min_policy_count=last_lane_rescue_min_policy_count,
            last_lane_rescue_conf_thr=last_lane_rescue_conf_thr,
            last_lane_rescue_point_valid_thr=last_lane_rescue_point_valid_thr,
            last_lane_rescue_min_points=last_lane_rescue_min_points,
            last_lane_rescue_mean_valid_thr=last_lane_rescue_mean_valid_thr,
            last_lane_rescue_quality_thr=last_lane_rescue_quality_thr,
            last_lane_rescue_dist_px=last_lane_rescue_dist_px,
            edge_last_lane_rescue=edge_last_lane_rescue,
            edge_rescue_conf_thr=edge_rescue_conf_thr,
            edge_rescue_point_valid_thr=edge_rescue_point_valid_thr,
            edge_rescue_min_points=edge_rescue_min_points,
            edge_rescue_mean_valid_thr=edge_rescue_mean_valid_thr,
            edge_rescue_quality_thr=edge_rescue_quality_thr,
            edge_rescue_outside_gap_px=edge_rescue_outside_gap_px,
            edge_rescue_dist_px=edge_rescue_dist_px,
            edge_rescue_min_policy_count=edge_rescue_min_policy_count,
            edge_count4_to5_upgrade=edge_count4_to5_upgrade,
            edge_count4_to5_prob_margin=edge_count4_to5_prob_margin,
            enable_soft_count_decision=enable_soft_count_decision,
            soft_count_prob_margin=soft_count_prob_margin,
            soft_count_quality_weight=soft_count_quality_weight,
            soft_count_prior_weight=soft_count_prior_weight,
            soft_count_duplicate_penalty=soft_count_duplicate_penalty,
            soft_count_invalid_penalty=soft_count_invalid_penalty,
        )
        write_tusimple_predictions(pred_out_path, pred_records)

    result, per_image = TuSimpleOfficialLaneEval.bench_records(
        pred_records,
        gt_records,
        strict_length=True,
        return_records=save_records,
    )
    summary = result.as_dict()
    summary["official_score"] = round(
        official_metric_score(
            result.accuracy,
            result.fp,
            result.fn,
            fp_weight=score_fp_weight,
            fn_weight=score_fn_weight,
        ),
        6,
    )
    summary.update(_summarize_prediction_counts(pred_records, gt_records))
    if timing:
        summary.update(timing)

    output = {
        "summary": summary,
        "config": {
            "weights": None if pred_json else str(Path(weights).resolve()),
            "archive_root": str(archive_root.resolve()),
            "split": split,
            "gt_json": str(gt_path.resolve()),
            "input_pred_json": None if input_pred_json_path is None else str(input_pred_json_path),
            "pred_json": str(pred_out_path.resolve()),
            "save_dir": str(save_dir.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(conf),
            "point_valid_thr": float(point_valid_thr),
            "nms_dist_px": float(nms_dist_px),
            "max_det": int(max_det),
            "min_points": int(min_points),
            "rank_min_points": rank_min_points,
            "use_count_head_decode": bool(use_count_head_decode),
            "count_head_temperature": float(count_head_temperature),
            "candidate_score_thr": float(candidate_score_thr),
            "candidate_point_valid_thr": float(candidate_point_valid_thr),
            "candidate_min_points": int(candidate_min_points),
            "enable_rescue_candidate_pool": bool(enable_rescue_candidate_pool),
            "rescue_candidate_score_thr": float(rescue_candidate_score_thr),
            "rescue_candidate_point_valid_thr": float(rescue_candidate_point_valid_thr),
            "rescue_candidate_min_points": int(rescue_candidate_min_points),
            "final_min_points": int(final_min_points),
            "fifth_min_points": int(fifth_min_points),
            "line_nms_min_overlap": int(line_nms_min_overlap),
            "line_nms_rescue_dist_px": float(line_nms_rescue_dist_px),
            "quality_rescue_5th": bool(quality_rescue_5th),
            "quality_rescue_count5_thr": float(quality_rescue_count5_thr),
            "quality_rescue_conf_thr": float(quality_rescue_conf_thr),
            "quality_rescue_mean_valid_thr": float(quality_rescue_mean_valid_thr),
            "quality_rescue_quality_thr": float(quality_rescue_quality_thr),
            "quality_rescue_min_points": int(quality_rescue_min_points),
            "quality_rescue_dist_px": float(quality_rescue_dist_px),
            "last_lane_rescue": bool(last_lane_rescue),
            "last_lane_rescue_min_policy_count": int(last_lane_rescue_min_policy_count),
            "last_lane_rescue_conf_thr": None if last_lane_rescue_conf_thr is None else float(last_lane_rescue_conf_thr),
            "last_lane_rescue_point_valid_thr": float(last_lane_rescue_point_valid_thr),
            "last_lane_rescue_min_points": int(last_lane_rescue_min_points),
            "last_lane_rescue_mean_valid_thr": float(last_lane_rescue_mean_valid_thr),
            "last_lane_rescue_quality_thr": float(last_lane_rescue_quality_thr),
            "last_lane_rescue_dist_px": float(last_lane_rescue_dist_px),
            "edge_last_lane_rescue": bool(edge_last_lane_rescue),
            "edge_rescue_conf_thr": float(edge_rescue_conf_thr),
            "edge_rescue_point_valid_thr": float(edge_rescue_point_valid_thr),
            "edge_rescue_min_points": int(edge_rescue_min_points),
            "edge_rescue_mean_valid_thr": float(edge_rescue_mean_valid_thr),
            "edge_rescue_quality_thr": float(edge_rescue_quality_thr),
            "edge_rescue_outside_gap_px": float(edge_rescue_outside_gap_px),
            "edge_rescue_dist_px": float(edge_rescue_dist_px),
            "edge_rescue_min_policy_count": int(edge_rescue_min_policy_count),
            "edge_count4_to5_upgrade": bool(edge_count4_to5_upgrade),
            "edge_count4_to5_prob_margin": float(edge_count4_to5_prob_margin),
            "soft_count_decision": bool(enable_soft_count_decision),
            "soft_count_prob_margin": float(soft_count_prob_margin),
            "soft_count_quality_weight": float(soft_count_quality_weight),
            "soft_count_prior_weight": float(soft_count_prior_weight),
            "soft_count_duplicate_penalty": float(soft_count_duplicate_penalty),
            "soft_count_invalid_penalty": float(soft_count_invalid_penalty),
            "runtime_ms": float(runtime_ms),
            "use_measured_runtime": bool(use_measured_runtime),
            "official_score_formula": "Accuracy - score_fp_weight * FP - score_fn_weight * FN",
            "score_fp_weight": float(score_fp_weight),
            "score_fn_weight": float(score_fn_weight),
            "device": str(device),
            "half": bool(half),
        },
    }
    if save_records:
        output["records"] = per_image
    (save_dir / "tusimple_official_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if pred_json:
        print(f"evaluated predictions: {input_pred_json_path}")
    else:
        print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved to: {save_dir.resolve()}")
    return output


def main() -> None:
    args = parse_args()
    if not args.pred_json:
        warn_max_det_mismatch(args.weights, max_det=args.max_det, context="TuSimple official eval")
    evaluate_tusimple_official(
        weights=args.weights,
        archive_root=args.archive_root,
        split=args.split,
        gt_json=args.gt_json,
        pred_json=args.pred_json,
        imgsz=normalize_imgsz(args.imgsz, dataset=args.dataset),
        conf=args.conf,
        point_valid_thr=args.point_valid_thr,
        nms_dist_px=args.nms_dist_px,
        max_det=args.max_det,
        min_points=args.min_points,
        max_images=args.max_images,
        warmup=args.warmup,
        device=args.device,
        half=args.half,
        runtime_ms=args.runtime_ms,
        use_measured_runtime=args.use_measured_runtime,
        save_dir=args.save_dir,
        save_records=args.save_records,
        count_calibration=count_calibration_from_args(args),
        rank_min_points=parse_rank_min_points(args.rank_min_points),
        **count_head_decode_kwargs_from_args(args, dataset_name=args.dataset),
        score_fp_weight=args.score_fp_weight,
        score_fn_weight=args.score_fn_weight,
    )


if __name__ == "__main__":
    main()
