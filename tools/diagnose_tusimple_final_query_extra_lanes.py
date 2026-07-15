from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    TuSimpleOfficialLaneEval,
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    official_gt_contract_summary,
    read_tusimple_json_lines,
    resolve_tusimple_gt_json,
    tusimple_image_path,
    valid_tusimple_lanes,
)
from tools.diagnose_tusimple_raw_q12_filters import (  # noqa: E402
    _apply_query_decode_yaml,
    _csv_value,
    _default_gt_json,
    _lane_position_map,
    _limit_records,
    _parse_date_session,
    _percentile,
    _sync_if_cuda,
    _valid_gt_lanes_with_ids,
    _visible_count,
    _write_csv,
)
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.models.gcs.decode_summary import load_decode_yaml, validate_decode_yaml_for_model  # noqa: E402
from ultralytics.models.gcs.mode_utils import resolve_decode_mode  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = ROOT / "runs/gcs_lane/query_alpha05_env30_q12_ultrashort_dataref_geom1_full_protocol_v1/weights/official_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose final decoded query ids for TuSimple over-count images. "
            "This is a reporting-only GT audit and does not change official metrics or decode."
        )
    )
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="TuSimple archive split.")
    parser.add_argument("--gt-json", default=None, help="Official TuSimple GT json-lines file.")
    parser.add_argument(
        "--allow-noncanonical-gt",
        action="store_true",
        help="Allow non-363 split=val GT for diagnostics; summary marks it incomparable.",
    )
    parser.add_argument(
        "--allow-test-oracle",
        action="store_true",
        help="Required for split=test because this tool uses GT after decode for diagnosis.",
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Query GCS checkpoint .pt or YAML.")
    parser.add_argument("--decode-yaml", default=None, help="Optional query official_best_decode.yaml.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.001, help="Lane existence confidence threshold.")
    parser.add_argument("--point-valid-thr", type=float, default=0.55, help="Per-point visibility threshold.")
    parser.add_argument("--nms-dist-px", type=float, default=18.0, help="Lane-NMS distance in original-image pixels.")
    parser.add_argument("--max-det", type=int, default=6, help="Final decode max_det.")
    parser.add_argument("--min-points", type=int, default=2, help="Minimum visible anchors required to keep a lane.")
    parser.add_argument("--valid-before-maxdet", action="store_true", help="Apply point-valid/min_points before max_det.")
    parser.add_argument("--count-aware-topk", action="store_true", help="Use count-aware top-k if evaluating that decode.")
    parser.add_argument("--count-aware-min-k", type=int, default=3)
    parser.add_argument("--count-aware-max-k", type=int, default=5)
    parser.add_argument("--count-aware-length-norm", type=float, default=12.0)
    parser.add_argument("--count-aware-extra-margin", type=int, default=0)
    parser.add_argument("--count-mode", choices=("score_sum", "count_logits"), default="score_sum")
    parser.add_argument("--match-min-overlap", type=int, default=3, help="Minimum overlapping h-samples for APE.")
    parser.add_argument("--extra-match-ape-px", type=float, default=20.0, help="APE gate used for diagnostic matches.")
    parser.add_argument("--duplicate-dist-px", type=float, default=30.0, help="Nearest-pred distance for duplicate-like extras.")
    parser.add_argument("--far-gt-px", type=float, default=50.0, help="Nearest-GT distance for far spurious extras.")
    parser.add_argument("--focus-queries", default="4,5,6,8,11", help="Comma-separated queries highlighted in summary.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of GT records. 0 means all.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of untimed warmup forwards.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--save-dir", default=None, help="Output directory.")
    return parser.parse_args()


def _weight_run_dir(weights: str | Path) -> Path | None:
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir is not None and str(args.save_dir).strip():
        return Path(args.save_dir)
    run_dir = _weight_run_dir(args.weights)
    tag = (
        f"final_query_extra_{args.split}_conf{float(args.conf):.4g}_"
        f"pvalid{float(args.point_valid_thr):.4g}_nms{float(args.nms_dist_px):.4g}_"
        f"maxdet{int(args.max_det)}_minp{int(args.min_points)}"
    ).replace(".", "p")
    if bool(args.valid_before_maxdet):
        tag += "_validbeforemaxdet"
    if run_dir is not None:
        return run_dir / tag
    return ROOT / "runs" / "gcs_lane" / "final_query_extra" / Path(args.weights).stem / tag


def _gt_lane_count(record: dict) -> int:
    return int(len(valid_tusimple_lanes(record.get("lanes", []))))


def _focus_queries(value: str) -> list[int]:
    out = []
    for part in str(value or "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, ndigits)


def _mean(values: list[float]) -> float | None:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _median(values: list[float]) -> float | None:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    if not vals:
        return None
    return float(median(vals))


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: (str(kv[0])))}


def _nested_counter_dict(counter: Counter) -> dict[str, int]:
    return {f"{k[0]}->{k[1]}": int(v) for k, v in sorted(counter.items(), key=lambda kv: (kv[0][0], kv[0][1]))}


def _lane_xs_for_official(lane: dict, h_samples: list[float], image_shape: tuple[int, int]) -> list[int] | None:
    lanes = gcs_lanes_to_tusimple_lanes([lane], h_samples, image_shape=image_shape)
    if not lanes:
        return None
    return [int(x) for x in lanes[0]]


def _mean_abs_x_px(pred_xs: list[float] | np.ndarray, gt_xs: list[float] | np.ndarray, min_overlap: int) -> tuple[float, int]:
    pred = np.asarray(pred_xs, dtype=np.float32)
    gt = np.asarray(gt_xs, dtype=np.float32)
    valid = (pred >= 0.0) & (gt >= 0.0) & np.isfinite(pred) & np.isfinite(gt)
    overlap = int(valid.sum())
    if overlap < int(min_overlap):
        return float("inf"), overlap
    return float(np.mean(np.abs(pred[valid] - gt[valid]))), overlap


def _official_line_acc(pred_xs: list[float], gt_xs: list[float], h_samples: list[float]) -> float:
    angle = TuSimpleOfficialLaneEval.get_angle(gt_xs, h_samples)
    thresh = TuSimpleOfficialLaneEval.pixel_thresh / max(float(np.cos(angle)), 1e-12)
    return float(TuSimpleOfficialLaneEval.line_accuracy(pred_xs, gt_xs, thresh))


def _assign_pred_to_gt(
    pred_xs_list: list[list[int]],
    gt_lanes: list[tuple[int, list[float]]],
    h_samples: list[float],
    min_overlap: int,
) -> tuple[dict[int, int], dict[int, int], dict[tuple[int, int], dict]]:
    pair_metrics: dict[tuple[int, int], dict] = {}
    candidates = []
    for pred_i, pred_xs in enumerate(pred_xs_list):
        for gt_list_i, (gt_lane_id, gt_lane) in enumerate(gt_lanes):
            ape, overlap = _mean_abs_x_px(pred_xs, gt_lane, min_overlap=min_overlap)
            acc = _official_line_acc(pred_xs, gt_lane, h_samples)
            pair_metrics[(pred_i, gt_list_i)] = {
                "gt_lane_id": int(gt_lane_id),
                "ape_px": ape,
                "overlap": int(overlap),
                "official_line_acc": float(acc),
            }
            if acc >= float(TuSimpleOfficialLaneEval.pt_thresh):
                candidates.append((float(acc), -float(ape) if math.isfinite(ape) else -1e9, pred_i, gt_list_i))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for _, _, pred_i, gt_list_i in candidates:
        if pred_i in pred_to_gt or gt_list_i in gt_to_pred:
            continue
        pred_to_gt[int(pred_i)] = int(gt_list_i)
        gt_to_pred[int(gt_list_i)] = int(pred_i)
    return pred_to_gt, gt_to_pred, pair_metrics


def _nearest_gt(
    pred_xs: list[int],
    gt_lanes: list[tuple[int, list[float]]],
    h_samples: list[float],
    min_overlap: int,
) -> dict:
    best = {
        "gt_list_index": "",
        "gt_lane_id": "",
        "ape_px": float("inf"),
        "overlap": 0,
        "official_line_acc": 0.0,
    }
    for gt_list_i, (gt_lane_id, gt_lane) in enumerate(gt_lanes):
        ape, overlap = _mean_abs_x_px(pred_xs, gt_lane, min_overlap=min_overlap)
        acc = _official_line_acc(pred_xs, gt_lane, h_samples)
        if best["gt_list_index"] == "" or (math.isfinite(ape), -ape, acc) > (
            math.isfinite(float(best["ape_px"])),
            -float(best["ape_px"]),
            float(best["official_line_acc"]),
        ):
            best = {
                "gt_list_index": int(gt_list_i),
                "gt_lane_id": int(gt_lane_id),
                "ape_px": float(ape),
                "overlap": int(overlap),
                "official_line_acc": float(acc),
            }
    return best


def _nearest_pred(pred_i: int, pred_xs_list: list[list[int]], min_overlap: int) -> dict:
    best = {"pred_index": "", "dist_px": float("inf"), "overlap": 0}
    for other_i, other_xs in enumerate(pred_xs_list):
        if int(other_i) == int(pred_i):
            continue
        dist, overlap = _mean_abs_x_px(pred_xs_list[pred_i], other_xs, min_overlap=min_overlap)
        if best["pred_index"] == "" or dist < float(best["dist_px"]):
            best = {"pred_index": int(other_i), "dist_px": float(dist), "overlap": int(overlap)}
    return best


def _lane_valid_stats(lane: dict, point_valid_thr: float) -> tuple[int, int]:
    valid_len = 0
    valid_count = 0
    if "point_valid" in lane:
        valid_len = int(np.asarray(lane["point_valid"], dtype=np.float32).sum())
    elif "visible_points_norm" in lane:
        valid_len = int(np.asarray(lane["visible_points_norm"], dtype=np.float32).shape[0])
    elif "points_norm" in lane:
        valid_len = int(np.asarray(lane["points_norm"], dtype=np.float32).shape[0])
    if "point_valid_scores" in lane:
        valid_count = int((np.asarray(lane["point_valid_scores"], dtype=np.float32) >= float(point_valid_thr)).sum())
    else:
        valid_count = valid_len
    return valid_len, valid_count


def _query_stats(rows: list[dict], query_id: int) -> dict:
    q_rows = [row for row in rows if int(row.get("query_id", -1)) == int(query_id)]
    extra = [row for row in q_rows if bool(row.get("extra_topk_candidate"))]
    unmatched = [row for row in q_rows if bool(row.get("over_image")) and not bool(row.get("greedy_matched_gt"))]
    duplicate = [row for row in extra if bool(row.get("duplicate_like_extra"))]
    far = [row for row in extra if bool(row.get("nearest_gt_far_extra"))]
    return {
        "final_lanes": int(len(q_rows)),
        "over_image_final_lanes": int(sum(1 for row in q_rows if bool(row.get("over_image")))),
        "greedy_unmatched_over_lanes": int(len(unmatched)),
        "extra_topk_lanes": int(len(extra)),
        "duplicate_like_extra_lanes": int(len(duplicate)),
        "nearest_gt_far_extra_lanes": int(len(far)),
        "score_mean": _round_or_none(_mean([float(row["score"]) for row in q_rows])),
        "extra_score_mean": _round_or_none(_mean([float(row["score"]) for row in extra])),
        "extra_nearest_gt_ape_p50": _round_or_none(_median([float(row["nearest_gt_ape_px"]) for row in extra])),
        "extra_nearest_gt_ape_p90": _round_or_none(_percentile([float(row["nearest_gt_ape_px"]) for row in extra], 90.0) if extra else None),
    }


def _group_extra_hist(rows: list[dict], gt_count: int, final_count: int | None = None) -> Counter:
    out = Counter()
    for row in rows:
        if int(row.get("gt_count", -1)) != int(gt_count):
            continue
        if final_count is not None and int(row.get("final_count", -1)) != int(final_count):
            continue
        if bool(row.get("extra_topk_candidate")):
            out[int(row["query_id"])] += 1
    return out


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if str(args.split).lower() == "test" and not bool(args.allow_test_oracle):
        raise ValueError(
            "This diagnostic uses TEST GT after decode to attribute extra lanes. "
            "Pass --allow-test-oracle only for reporting-only debugging, never for selection."
        )

    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_json = _default_gt_json(args)
    gt_path = resolve_tusimple_gt_json(archive_root, split=args.split, gt_json=gt_json)
    gt_records = _limit_records(read_tusimple_json_lines(gt_path), args.max_images)
    if not gt_records:
        raise ValueError(f"No TuSimple GT records found in {gt_path}.")
    gt_contract = official_gt_contract_summary(
        split=args.split,
        gt_json=gt_path,
        gt_records=gt_records,
        allow_noncanonical_gt=bool(args.allow_noncanonical_gt),
    )
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)

    device_obj = select_device(args.device)
    model = load_gcs_model(args.weights, device=device_obj, half=args.half, gcs_imgsz=imgsz)
    if args.decode_yaml:
        _, decode_yaml_cfg = load_decode_yaml(args.decode_yaml)
        decode_mode = resolve_decode_mode(decode_yaml_cfg.get("decode_mode"), model)
        validate_decode_yaml_for_model(decode_yaml_cfg, model_mode=decode_mode)
        if decode_mode != "query":
            raise ValueError(f"Final-query extra-lane diagnostic supports query decode only, got {decode_mode!r}.")
        _apply_query_decode_yaml(args, decode_yaml_cfg)
        args.count_aware_topk = bool(decode_yaml_cfg.get("count_aware_topk", False))
        args.count_aware_min_k = int(decode_yaml_cfg.get("count_aware_min_k", args.count_aware_min_k))
        args.count_aware_max_k = int(decode_yaml_cfg.get("count_aware_max_k", args.count_aware_max_k))
        args.count_aware_length_norm = float(decode_yaml_cfg.get("count_aware_length_norm", args.count_aware_length_norm))
        args.count_aware_extra_margin = int(decode_yaml_cfg.get("count_aware_extra_margin", args.count_aware_extra_margin))
        args.count_mode = str(decode_yaml_cfg.get("count_mode", args.count_mode))
    else:
        decode_mode = resolve_decode_mode("auto", model)
        if decode_mode != "query":
            raise ValueError(f"Final-query extra-lane diagnostic supports query decode only, got {decode_mode!r}.")

    warn_max_det_mismatch(args.weights, max_det=args.max_det, context="TuSimple final-query extra-lane diagnostic")

    if args.warmup > 0 and gt_records:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device_obj, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    save_dir = resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    lane_rows: list[dict] = []
    image_rows: list[dict] = []
    infer_time_s = 0.0
    post_time_s = 0.0

    for record in gt_records:
        raw_file = str(record["raw_file"])
        date, session = _parse_date_session(raw_file)
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        original_shape = img.shape[:2]
        h_samples = [float(x) for x in record["h_samples"]]
        tensor = preprocess_image(img, imgsz, device=device_obj, half=args.half)

        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        t1 = time.perf_counter()

        pred_valid = preds.get("pred_valid_logits")
        pred_count_logits = preds.get("pred_count_logits")
        lanes = decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
            pred_count_logits=pred_count_logits[0] if pred_count_logits is not None else None,
            image_shape=original_shape,
            score_thr=float(args.conf),
            point_valid_thr=float(args.point_valid_thr),
            min_points=int(args.min_points),
            max_det=int(args.max_det),
            nms_dist_px=float(args.nms_dist_px),
            valid_before_maxdet=bool(args.valid_before_maxdet),
            count_aware_topk=bool(args.count_aware_topk),
            count_aware_min_k=int(args.count_aware_min_k),
            count_aware_max_k=int(args.count_aware_max_k),
            count_aware_length_norm=float(args.count_aware_length_norm),
            count_aware_extra_margin=int(args.count_aware_extra_margin),
            count_mode=str(args.count_mode),
        )

        exported_lanes = []
        pred_xs_list: list[list[int]] = []
        for lane in lanes:
            lane_xs = _lane_xs_for_official(lane, h_samples, original_shape)
            if lane_xs is None:
                continue
            exported_lanes.append(lane)
            pred_xs_list.append(lane_xs)

        gt_lanes = _valid_gt_lanes_with_ids(record)
        gt_count = _gt_lane_count(record)
        final_count = int(len(exported_lanes))
        over_image = final_count > gt_count
        under_image = final_count < gt_count
        pred_to_gt, _, pair_metrics = _assign_pred_to_gt(
            pred_xs_list,
            gt_lanes,
            h_samples,
            min_overlap=int(args.match_min_overlap),
        )
        lane_positions = _lane_position_map(record)
        pred_records_lanes = [list(xs) for xs in pred_xs_list]
        image_acc, image_fp, image_fn = TuSimpleOfficialLaneEval.bench(
            pred_records_lanes,
            [list(x) for _, x in gt_lanes],
            h_samples,
            running_time=1.0,
        )

        image_lane_rows = []
        for pred_i, (lane, lane_xs) in enumerate(zip(exported_lanes, pred_xs_list)):
            query_id = int(lane["query"])
            valid_len, valid_count = _lane_valid_stats(lane, float(args.point_valid_thr))
            nearest_gt = _nearest_gt(lane_xs, gt_lanes, h_samples, min_overlap=int(args.match_min_overlap))
            nearest_pred = _nearest_pred(pred_i, pred_xs_list, min_overlap=2)
            matched_gt_list_i = pred_to_gt.get(int(pred_i), "")
            matched_gt_lane_id = ""
            matched_ape = ""
            matched_acc = ""
            matched_overlap = ""
            matched_lane_order = ""
            matched_side_group = ""
            matched_visible = ""
            if matched_gt_list_i != "":
                gt_lane_id, gt_lane = gt_lanes[int(matched_gt_list_i)]
                pm = pair_metrics.get((int(pred_i), int(matched_gt_list_i)), {})
                pos = lane_positions.get(int(gt_lane_id), {})
                matched_gt_lane_id = int(gt_lane_id)
                matched_ape = pm.get("ape_px", "")
                matched_acc = pm.get("official_line_acc", "")
                matched_overlap = pm.get("overlap", "")
                matched_lane_order = pos.get("lane_order", "")
                matched_side_group = pos.get("side_group", "")
                matched_visible = _visible_count(gt_lane)

            nearest_gt_lane_id = nearest_gt.get("gt_lane_id", "")
            nearest_pos = lane_positions.get(int(nearest_gt_lane_id), {}) if nearest_gt_lane_id != "" else {}
            row = {
                "raw_file": raw_file,
                "date": date,
                "session": session,
                "gt_count": int(gt_count),
                "final_count": int(final_count),
                "over_image": bool(over_image),
                "under_image": bool(under_image),
                "pred_index_score_order": int(pred_i),
                "query_id": int(query_id),
                "score": round(float(lane["score"]), 8),
                "valid_len": int(valid_len),
                "valid_count_at_thr": int(valid_count),
                "greedy_matched_gt": bool(matched_gt_list_i != ""),
                "matched_gt_lane_id": matched_gt_lane_id,
                "matched_gt_ape_px": matched_ape,
                "matched_gt_official_line_acc": matched_acc,
                "matched_gt_overlap": matched_overlap,
                "matched_gt_lane_order": matched_lane_order,
                "matched_gt_side_group": matched_side_group,
                "matched_gt_visible_points": matched_visible,
                "nearest_gt_lane_id": nearest_gt_lane_id,
                "nearest_gt_ape_px": nearest_gt.get("ape_px", float("inf")),
                "nearest_gt_overlap": nearest_gt.get("overlap", 0),
                "nearest_gt_official_line_acc": nearest_gt.get("official_line_acc", 0.0),
                "nearest_gt_lane_order": nearest_pos.get("lane_order", ""),
                "nearest_gt_side_group": nearest_pos.get("side_group", ""),
                "nearest_pred_index": nearest_pred.get("pred_index", ""),
                "nearest_pred_query_id": ""
                if nearest_pred.get("pred_index", "") == ""
                else int(exported_lanes[int(nearest_pred["pred_index"])]["query"]),
                "nearest_pred_dist_px": nearest_pred.get("dist_px", float("inf")),
                "nearest_pred_overlap": nearest_pred.get("overlap", 0),
                "image_official_acc": round(float(image_acc), 6),
                "image_official_fp": round(float(image_fp), 6),
                "image_official_fn": round(float(image_fn), 6),
            }
            image_lane_rows.append(row)

        extra_quota = max(0, final_count - gt_count)
        ranked_extra = sorted(
            range(len(image_lane_rows)),
            key=lambda idx: (
                bool(image_lane_rows[idx]["greedy_matched_gt"]),
                -float(image_lane_rows[idx]["nearest_gt_ape_px"])
                if math.isfinite(float(image_lane_rows[idx]["nearest_gt_ape_px"]))
                else -1e9,
                float(image_lane_rows[idx]["nearest_gt_official_line_acc"]),
                -float(image_lane_rows[idx]["score"]),
                int(image_lane_rows[idx]["query_id"]),
            ),
        )
        extra_topk = set(ranked_extra[:extra_quota]) if over_image and extra_quota > 0 else set()
        for idx, row in enumerate(image_lane_rows):
            greedy_unmatched_extra = bool(over_image and not bool(row["greedy_matched_gt"]))
            extra_topk_candidate = bool(idx in extra_topk)
            nearest_pred_dist = float(row["nearest_pred_dist_px"])
            nearest_gt_ape = float(row["nearest_gt_ape_px"])
            row["greedy_unmatched_extra"] = greedy_unmatched_extra
            row["extra_topk_candidate"] = extra_topk_candidate
            row["duplicate_like_extra"] = bool(extra_topk_candidate and nearest_pred_dist <= float(args.duplicate_dist_px))
            row["nearest_gt_far_extra"] = bool(extra_topk_candidate and nearest_gt_ape >= float(args.far_gt_px))
            lane_rows.append(row)

        image_rows.append(
            {
                "raw_file": raw_file,
                "date": date,
                "session": session,
                "gt_count": int(gt_count),
                "final_count": int(final_count),
                "over_image": bool(over_image),
                "under_image": bool(under_image),
                "extra_quota": int(extra_quota),
                "final_queries": ";".join(str(int(lane["query"])) for lane in exported_lanes),
                "extra_topk_queries": ";".join(str(image_lane_rows[i]["query_id"]) for i in sorted(extra_topk)),
                "greedy_unmatched_queries": ";".join(
                    str(row["query_id"]) for row in image_lane_rows if bool(row.get("greedy_unmatched_extra"))
                ),
                "image_official_acc": round(float(image_acc), 6),
                "image_official_fp": round(float(image_fp), 6),
                "image_official_fn": round(float(image_fn), 6),
            }
        )

        t2 = time.perf_counter()
        infer_time_s += t1 - t0
        post_time_s += t2 - t1

    lane_fields = [
        "raw_file",
        "date",
        "session",
        "gt_count",
        "final_count",
        "over_image",
        "under_image",
        "pred_index_score_order",
        "query_id",
        "score",
        "valid_len",
        "valid_count_at_thr",
        "greedy_matched_gt",
        "greedy_unmatched_extra",
        "extra_topk_candidate",
        "duplicate_like_extra",
        "nearest_gt_far_extra",
        "matched_gt_lane_id",
        "matched_gt_ape_px",
        "matched_gt_official_line_acc",
        "matched_gt_overlap",
        "matched_gt_lane_order",
        "matched_gt_side_group",
        "matched_gt_visible_points",
        "nearest_gt_lane_id",
        "nearest_gt_ape_px",
        "nearest_gt_overlap",
        "nearest_gt_official_line_acc",
        "nearest_gt_lane_order",
        "nearest_gt_side_group",
        "nearest_pred_index",
        "nearest_pred_query_id",
        "nearest_pred_dist_px",
        "nearest_pred_overlap",
        "image_official_acc",
        "image_official_fp",
        "image_official_fn",
    ]
    image_fields = [
        "raw_file",
        "date",
        "session",
        "gt_count",
        "final_count",
        "over_image",
        "under_image",
        "extra_quota",
        "final_queries",
        "extra_topk_queries",
        "greedy_unmatched_queries",
        "image_official_acc",
        "image_official_fp",
        "image_official_fn",
    ]
    _write_csv(save_dir / "final_lane_trace.csv", lane_rows, lane_fields)
    _write_csv(save_dir / "extra_image_trace.csv", [row for row in image_rows if bool(row["over_image"])], image_fields)
    _write_csv(save_dir / "per_image_final_query_trace.csv", image_rows, image_fields)

    final_query_hist_all = Counter(int(row["query_id"]) for row in lane_rows)
    final_query_hist_over = Counter(int(row["query_id"]) for row in lane_rows if bool(row["over_image"]))
    final_query_hist_extra = Counter(int(row["query_id"]) for row in lane_rows if bool(row["extra_topk_candidate"]))
    greedy_unmatched_extra = Counter(int(row["query_id"]) for row in lane_rows if bool(row["greedy_unmatched_extra"]))
    duplicate_extra = Counter(int(row["query_id"]) for row in lane_rows if bool(row["duplicate_like_extra"]))
    far_extra = Counter(int(row["query_id"]) for row in lane_rows if bool(row["nearest_gt_far_extra"]))
    confusion = Counter((int(row["gt_count"]), int(row["final_count"])) for row in image_rows)
    over_by_gt = Counter(int(row["gt_count"]) for row in image_rows if bool(row["over_image"]))
    under_by_gt = Counter(int(row["gt_count"]) for row in image_rows if bool(row["under_image"]))
    focus = _focus_queries(args.focus_queries)

    extra_rows = [row for row in lane_rows if bool(row["extra_topk_candidate"])]
    unmatched_rows = [row for row in lane_rows if bool(row["greedy_unmatched_extra"])]
    summary = {
        "schema": "gcs_final_query_extra_summary_v1",
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "decode_yaml": None if not args.decode_yaml else str(Path(args.decode_yaml).resolve()),
            "archive_root": str(archive_root.resolve()),
            "split": str(args.split),
            "gt_json": str(gt_path.resolve()),
            "save_dir": str(save_dir.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "decode_mode": "query",
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "valid_before_maxdet": bool(args.valid_before_maxdet),
            "count_aware_topk": bool(args.count_aware_topk),
            "count_aware_min_k": int(args.count_aware_min_k),
            "count_aware_max_k": int(args.count_aware_max_k),
            "count_aware_length_norm": float(args.count_aware_length_norm),
            "count_aware_extra_margin": int(args.count_aware_extra_margin),
            "count_mode": str(args.count_mode),
            "match_min_overlap": int(args.match_min_overlap),
            "extra_match_ape_px": float(args.extra_match_ape_px),
            "duplicate_dist_px": float(args.duplicate_dist_px),
            "far_gt_px": float(args.far_gt_px),
            "max_images": int(args.max_images),
            "device": str(args.device),
            "half": bool(args.half),
            "diagnostic_only": True,
            "uses_gt_for_decode": False,
            "uses_gt_for_post_decode_audit": True,
        },
        "images": int(len(image_rows)),
        "over_images": int(sum(1 for row in image_rows if bool(row["over_image"]))),
        "under_images": int(sum(1 for row in image_rows if bool(row["under_image"]))),
        "over_images_by_gt_count": _counter_dict(over_by_gt),
        "under_images_by_gt_count": _counter_dict(under_by_gt),
        "final_count_confusion": _nested_counter_dict(confusion),
        "final_lanes": int(len(lane_rows)),
        "extra_topk_lanes": int(len(extra_rows)),
        "greedy_unmatched_over_lanes": int(len(unmatched_rows)),
        "extra_topk_score_mean": _round_or_none(_mean([float(row["score"]) for row in extra_rows])),
        "extra_topk_nearest_gt_ape_p50": _round_or_none(_median([float(row["nearest_gt_ape_px"]) for row in extra_rows])),
        "extra_topk_nearest_gt_ape_p90": _round_or_none(
            _percentile([float(row["nearest_gt_ape_px"]) for row in extra_rows], 90.0) if extra_rows else None
        ),
        "extra_topk_nearest_pred_dist_p50": _round_or_none(
            _median([float(row["nearest_pred_dist_px"]) for row in extra_rows])
        ),
        "extra_topk_nearest_pred_dist_p90": _round_or_none(
            _percentile([float(row["nearest_pred_dist_px"]) for row in extra_rows], 90.0) if extra_rows else None
        ),
        "final_query_hist_all": _counter_dict(final_query_hist_all),
        "final_query_hist_over_images": _counter_dict(final_query_hist_over),
        "final_query_hist_extra_candidate": _counter_dict(final_query_hist_extra),
        "greedy_unmatched_extra_by_query": _counter_dict(greedy_unmatched_extra),
        "duplicate_like_extra_by_query": _counter_dict(duplicate_extra),
        "nearest_gt_far_extra_by_query": _counter_dict(far_extra),
        "gt4_to5_extra_query_hist": _counter_dict(_group_extra_hist(lane_rows, gt_count=4, final_count=5)),
        "gt4_to6_extra_query_hist": _counter_dict(_group_extra_hist(lane_rows, gt_count=4, final_count=6)),
        "gt5_to6_extra_query_hist": _counter_dict(_group_extra_hist(lane_rows, gt_count=5, final_count=6)),
        "focus_query_stats": {str(q): _query_stats(lane_rows, q) for q in focus},
        "outputs": {
            "final_lane_trace": str((save_dir / "final_lane_trace.csv").resolve()),
            "extra_image_trace": str((save_dir / "extra_image_trace.csv").resolve()),
            "per_image_final_query_trace": str((save_dir / "per_image_final_query_trace.csv").resolve()),
        },
        "timing": {
            "avg_inference_ms": round(infer_time_s * 1000.0 / max(len(image_rows), 1), 4),
            "avg_postprocess_ms": round(post_time_s * 1000.0 / max(len(image_rows), 1), 4),
            "avg_total_ms": round((infer_time_s + post_time_s) * 1000.0 / max(len(image_rows), 1), 4),
        },
        **gt_contract,
    }
    (save_dir / "final_query_extra_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "images": summary["images"],
                "over_images": summary["over_images"],
                "under_images": summary["under_images"],
                "final_query_hist_extra_candidate": summary["final_query_hist_extra_candidate"],
                "gt5_to6_extra_query_hist": summary["gt5_to6_extra_query_hist"],
                "gt4_to5_extra_query_hist": summary["gt4_to5_extra_query_hist"],
            },
            indent=2,
        )
    )
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
