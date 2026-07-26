from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    find_tusimple_archive_root,
    official_gt_contract_summary,
    read_tusimple_json_lines,
    resolve_tusimple_gt_json,
    tusimple_image_path,
    valid_tusimple_lanes,
)
from tools.diagnose_tusimple_raw_q12_filters import (  # noqa: E402
    _best_ape,
    _interp_lane_xs,
    _lane_position_map,
    _parse_date_session,
    _query_arrays,
    _valid_gt_lanes_with_ids,
    _visible_count,
)
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.models.gcs.decode_summary import load_decode_yaml, validate_decode_yaml_for_model  # noqa: E402
from ultralytics.models.gcs.mode_utils import resolve_decode_mode  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_VAL_GT = (
    ROOT
    / "runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/"
    / "tusimple_official_val_363_folder_aware_seed20260602.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic gate for query extent start/end heads on TuSimple.")
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="TuSimple archive split.")
    parser.add_argument("--gt-json", default=None, help="Official TuSimple GT json-lines file.")
    parser.add_argument("--allow-noncanonical-gt", action="store_true", help="Allow non-363 split=val GT for diagnostics.")
    parser.add_argument("--allow-test-oracle", action="store_true", help="Allow test split for predeclared audits only.")
    parser.add_argument("--weights", required=True, help="Query extent checkpoint .pt or YAML.")
    parser.add_argument("--decode-yaml", default=None, help="Optional query official_best_decode.yaml.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.003)
    parser.add_argument("--point-valid-thr", type=float, default=0.6)
    parser.add_argument("--nms-dist-px", type=float, default=0.0)
    parser.add_argument("--max-det", type=int, default=6)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--valid-before-maxdet", action="store_true")
    parser.add_argument("--count-aware-topk", action="store_true")
    parser.add_argument("--count-aware-min-k", type=int, default=2)
    parser.add_argument("--count-aware-max-k", type=int, default=5)
    parser.add_argument("--count-aware-length-norm", type=float, default=12.0)
    parser.add_argument("--count-aware-extra-margin", type=int, default=0)
    parser.add_argument("--count-mode", choices=("score_sum", "count_logits"), default="score_sum")
    parser.add_argument(
        "--extent-decode-modes",
        nargs="+",
        default=["interval", "intersect"],
        help="Extent decode modes to diagnose. 'none' is always included as point-valid baseline.",
    )
    parser.add_argument("--match-thr-px", type=float, default=20.0)
    parser.add_argument("--match-min-overlap", type=int, default=3)
    parser.add_argument("--short-visible-max", type=int, default=10)
    parser.add_argument("--endpoint-acc-tol", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def _default_gt_json(args: argparse.Namespace) -> str | None:
    if args.gt_json:
        return args.gt_json
    if str(args.split).lower() == "val" and DEFAULT_VAL_GT.exists():
        return str(DEFAULT_VAL_GT)
    return None


def _limit_records(records: list[dict], max_images: int) -> list[dict]:
    return records[: int(max_images)] if max_images and max_images > 0 else records


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _weight_run_dir(weights: str | Path) -> Path | None:
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def _resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir is not None and str(args.save_dir).strip():
        return Path(args.save_dir)
    run_dir = _weight_run_dir(args.weights)
    tag = f"query_extent_gate_{args.split}_conf{float(args.conf):.4g}_pvalid{float(args.point_valid_thr):.4g}"
    tag = tag.replace(".", "p")
    if run_dir is not None:
        return run_dir / tag
    return ROOT / "runs" / "gcs_lane" / "query_extent_gate" / Path(args.weights).stem / tag


def _apply_query_decode_yaml(args: argparse.Namespace, decode_yaml_cfg: dict) -> None:
    args.conf = float(decode_yaml_cfg["conf"])
    args.point_valid_thr = float(decode_yaml_cfg["point_valid_thr"])
    args.nms_dist_px = float(decode_yaml_cfg["nms_dist_px"])
    args.max_det = int(decode_yaml_cfg["max_det"])
    args.min_points = int(decode_yaml_cfg["min_points"])
    args.valid_before_maxdet = bool(decode_yaml_cfg.get("valid_before_maxdet", False))
    args.count_aware_topk = bool(decode_yaml_cfg.get("count_aware_topk", False))
    args.count_aware_min_k = int(decode_yaml_cfg.get("count_aware_min_k", args.count_aware_min_k))
    args.count_aware_max_k = int(decode_yaml_cfg.get("count_aware_max_k", args.count_aware_max_k))
    args.count_aware_length_norm = float(decode_yaml_cfg.get("count_aware_length_norm", args.count_aware_length_norm))
    args.count_aware_extra_margin = int(decode_yaml_cfg.get("count_aware_extra_margin", args.count_aware_extra_margin))
    args.count_mode = str(decode_yaml_cfg.get("count_mode", args.count_mode))


def _extent_modes(values: list[str]) -> list[str]:
    modes = ["none"]
    for value in values:
        mode = str(value or "none").strip().lower()
        if mode in {"off", "false", "0"}:
            mode = "none"
        if mode not in {"none", "interval", "intersect"}:
            raise ValueError(f"Unsupported extent decode mode {value!r}.")
        if mode not in modes:
            modes.append(mode)
    return modes


def _fixed_y_extent_target(gt_lane: list[float], h_samples: list[float]) -> tuple[int, int]:
    order = sorted(range(len(h_samples)), key=lambda i: float(h_samples[i]), reverse=True)
    visible = [rank for rank, sample_idx in enumerate(order) if float(gt_lane[sample_idx]) >= 0.0]
    if not visible:
        raise ValueError("GT lane has no visible h-samples.")
    return int(min(visible)), int(max(visible))


def _interval_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    lo_a, hi_a = min(int(a0), int(a1)), max(int(a0), int(a1))
    lo_b, hi_b = min(int(b0), int(b1)), max(int(b0), int(b1))
    inter = max(0, min(hi_a, hi_b) - max(lo_a, lo_b) + 1)
    union = max(1, max(hi_a, hi_b) - min(lo_a, lo_b) + 1)
    return float(inter / union)


def _decode_lanes(preds: dict, args: argparse.Namespace, image_shape: tuple[int, int], mode: str) -> list[dict]:
    pred_valid_logits = preds.get("pred_valid_logits")
    pred_count_logits = preds.get("pred_count_logits")
    pred_quality_logits = preds.get("pred_quality_logits")
    pred_start_logits = preds.get("pred_start_logits")
    pred_end_logits = preds.get("pred_end_logits")
    return decode_gcs_predictions(
        preds["pred_points"][0],
        preds["pred_logits"][0],
        pred_quality_logits=pred_quality_logits[0] if pred_quality_logits is not None else None,
        pred_valid_logits=pred_valid_logits[0] if pred_valid_logits is not None else None,
        pred_count_logits=pred_count_logits[0] if pred_count_logits is not None else None,
        pred_start_logits=pred_start_logits[0] if pred_start_logits is not None else None,
        pred_end_logits=pred_end_logits[0] if pred_end_logits is not None else None,
        image_shape=image_shape,
        score_thr=float(args.conf),
        point_valid_thr=float(args.point_valid_thr),
        min_points=int(args.min_points),
        max_det=int(args.max_det),
        nms_dist_px=float(args.nms_dist_px),
        valid_before_maxdet=bool(args.valid_before_maxdet),
        extent_decode=mode != "none",
        extent_decode_mode=mode,
        count_aware_topk=bool(args.count_aware_topk),
        count_aware_min_k=int(args.count_aware_min_k),
        count_aware_max_k=int(args.count_aware_max_k),
        count_aware_length_norm=float(args.count_aware_length_norm),
        count_aware_extra_margin=int(args.count_aware_extra_margin),
        count_mode=str(args.count_mode),
    )


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else round(float(mean(values)), 6)


def _percentile(values: list[float], q: float) -> float | None:
    finite = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not finite:
        return None
    if len(finite) == 1:
        return round(float(finite[0]), 6)
    pos = (len(finite) - 1) * float(q) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return round(float(finite[lo]), 6)
    frac = pos - lo
    return round(float(finite[lo] * (1.0 - frac) + finite[hi] * frac), 6)


def _finite_value(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_row_values(rows: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _finite_value(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _endpoint_acc(rows: list[dict], key: str, endpoint_tol: int) -> float:
    correct = 0
    for row in rows:
        value = _finite_value(row.get(key))
        if value is not None and value <= float(endpoint_tol):
            correct += 1
    return round(float(correct) / max(len(rows), 1), 6)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_summary(rows: list[dict], mode: str, endpoint_tol: int, match_thr_px: float) -> dict:
    if not rows:
        return {
            "count": 0,
            "has_match20": None,
            "best_ape_p90": None,
            "endpoint_start_acc_1": None,
            "endpoint_end_acc_1": None,
            "interval_iou": None,
        }
    ape_key = f"final_best_ape_px_{mode}"
    match_key = f"final_has_match_{int(match_thr_px)}px_{mode}" if float(match_thr_px).is_integer() else f"final_has_match_{match_thr_px}px_{mode}"
    apes = _finite_row_values(rows, ape_key)
    return {
        "count": int(len(rows)),
        "has_match20": round(sum(1 for row in rows if bool(row.get(match_key, False))) / max(len(rows), 1), 6),
        "best_ape_p90": _percentile(apes, 90.0),
        "endpoint_start_acc_1": _endpoint_acc(rows, "extent_start_abs_err", endpoint_tol),
        "endpoint_end_acc_1": _endpoint_acc(rows, "extent_end_abs_err", endpoint_tol),
        "interval_iou": _safe_mean(_finite_row_values(rows, "extent_interval_iou")),
    }


def _raw_match20_strata(rows: list[dict], mode: str, endpoint_tol: int, match_thr_px: float) -> dict:
    """Split endpoint diagnostics by whether the raw-best query was already a 20px geometry hit."""
    return {
        "raw_has_match20_true": _group_summary(
            [row for row in rows if bool(row.get("raw_has_match_20px", False))],
            mode,
            endpoint_tol,
            match_thr_px,
        ),
        "raw_has_match20_false": _group_summary(
            [row for row in rows if not bool(row.get("raw_has_match_20px", False))],
            mode,
            endpoint_tol,
            match_thr_px,
        ),
    }


def _mode_summary(
    *,
    mode: str,
    by_group: dict[str, list[dict]],
    lane_rows: list[dict],
    image_rows: list[dict],
    gt_hist: Counter,
    extra_by_gt_mode: dict[str, Counter],
    endpoint_tol: int,
    match_thr_px: float,
) -> dict:
    """Return the complete gate summary for one extent decode mode."""
    group_summary = {
        name: _group_summary(rows, mode, endpoint_tol, match_thr_px)
        for name, rows in by_group.items()
    }
    raw_match20_strata = {
        name: _raw_match20_strata(rows, mode, endpoint_tol, match_thr_px)
        for name, rows in by_group.items()
    }
    for name, strata in raw_match20_strata.items():
        group_summary[name]["raw_geometry_strata"] = strata

    final_survival = [
        1.0 if _finite_value(row.get(f"final_best_ape_px_{mode}")) is not None else 0.0
        for row in lane_rows
    ]

    def count_acc(gt_count: int) -> float:
        return round(
            sum(
                1
                for row in image_rows
                if int(row["gt_count"]) == int(gt_count)
                and int(row.get(f"pred_count_{mode}", -1)) == int(gt_count)
            )
            / max(int(gt_hist.get(int(gt_count), 0)), 1),
            6,
        )

    return {
        "short_gt4": group_summary["short_gt4"],
        "short_gt5": group_summary["short_gt5"],
        "short_gt4_raw_geometry_strata": raw_match20_strata["short_gt4"],
        "short_gt5_raw_geometry_strata": raw_match20_strata["short_gt5"],
        "short_gt4_has_match20": group_summary["short_gt4"]["has_match20"],
        "short_gt5_has_match20": group_summary["short_gt5"]["has_match20"],
        "short_gt4_best_ape_p90": group_summary["short_gt4"]["best_ape_p90"],
        "short_gt5_best_ape_p90": group_summary["short_gt5"]["best_ape_p90"],
        "short_gt4_endpoint_start_acc_1": group_summary["short_gt4"]["endpoint_start_acc_1"],
        "short_gt4_endpoint_end_acc_1": group_summary["short_gt4"]["endpoint_end_acc_1"],
        "short_gt5_endpoint_start_acc_1": group_summary["short_gt5"]["endpoint_start_acc_1"],
        "short_gt5_endpoint_end_acc_1": group_summary["short_gt5"]["endpoint_end_acc_1"],
        "short_gt4_interval_iou": group_summary["short_gt4"]["interval_iou"],
        "short_gt5_interval_iou": group_summary["short_gt5"]["interval_iou"],
        "final_survival": _safe_mean(final_survival),
        "normal_gt3_extra_rate": round(float(extra_by_gt_mode[mode][3]) / max(int(gt_hist.get(3, 0)), 1), 6),
        "normal_gt4_extra_rate": round(float(extra_by_gt_mode[mode][4]) / max(int(gt_hist.get(4, 0)), 1), 6),
        "count_acc_4": count_acc(4),
        "count_acc_5": count_acc(5),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if str(args.split).lower() == "test" and not bool(args.allow_test_oracle):
        raise ValueError(
            "Query extent gate uses GT to judge endpoint and raw/final match quality and is blocked on --split test by default."
        )

    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_path = resolve_tusimple_gt_json(archive_root, split=args.split, gt_json=_default_gt_json(args))
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
            raise ValueError(f"Query extent gate supports query decode only, got decode_mode={decode_mode!r}.")
        _apply_query_decode_yaml(args, decode_yaml_cfg)
    else:
        decode_mode = resolve_decode_mode("auto", model)
        if decode_mode != "query":
            raise ValueError(f"Query extent gate supports query decode only, got decode_mode={decode_mode!r}.")

    modes = _extent_modes(args.extent_decode_modes)
    primary_mode = next((mode for mode in modes if mode != "none"), "none")

    warn_max_det_mismatch(args.weights, max_det=args.max_det, context="TuSimple query extent gate")

    if args.warmup > 0:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device_obj, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    save_dir = _resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    lane_rows: list[dict] = []
    image_rows: list[dict] = []
    pred_count_by_mode: dict[str, Counter] = {mode: Counter() for mode in modes}
    extra_by_gt_mode: dict[str, Counter] = {mode: Counter() for mode in modes}
    infer_time_s = 0.0
    post_time_s = 0.0
    match_key = f"final_has_match_{int(args.match_thr_px)}px" if float(args.match_thr_px).is_integer() else f"final_has_match_{args.match_thr_px}px"

    for record in gt_records:
        raw_file = str(record["raw_file"])
        date, session = _parse_date_session(raw_file)
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_shape = img.shape[:2]
        h_samples = [float(x) for x in record["h_samples"]]
        tensor = preprocess_image(img, imgsz, device=device_obj, half=args.half)

        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        t1 = time.perf_counter()
        infer_time_s += t1 - t0

        if "pred_start_logits" not in preds or "pred_end_logits" not in preds:
            raise ValueError("Query extent gate requires model outputs pred_start_logits and pred_end_logits.")
        start_idx = preds["pred_start_logits"][0].detach().float().cpu().argmax(dim=1).long()
        end_idx = preds["pred_end_logits"][0].detach().float().cpu().argmax(dim=1).long()
        pred_lo = torch.minimum(start_idx, end_idx)
        pred_hi = torch.maximum(start_idx, end_idx)

        t2 = time.perf_counter()
        decoded_by_mode = {mode: _decode_lanes(preds, args, image_shape=image_shape, mode=mode) for mode in modes}
        post_time_s += time.perf_counter() - t2

        gt_lanes = _valid_gt_lanes_with_ids(record)
        gt_count = int(len(valid_tusimple_lanes(record.get("lanes", []))))
        lane_positions = _lane_position_map(record)
        raw_queries = _query_arrays(preds, image_shape=image_shape, h_samples=h_samples, min_points=args.min_points)
        pred_raw_xs = [q["pred_xs"] for q in raw_queries]
        final_xs_by_mode = {
            mode: [
                _interp_lane_xs(np.asarray(lane["points_norm"], dtype=np.float32), h_samples, image_shape)
                for lane in lanes
            ]
            for mode, lanes in decoded_by_mode.items()
        }

        image_row = {
            "raw_file": raw_file,
            "date": date,
            "session": session,
            "gt_count": gt_count,
        }
        for mode, lanes in decoded_by_mode.items():
            pred_count = int(len(lanes))
            image_row[f"pred_count_{mode}"] = pred_count
            pred_count_by_mode[mode][pred_count] += 1
            if pred_count > gt_count:
                extra_by_gt_mode[mode][gt_count] += 1
        image_rows.append(image_row)

        for gt_lane_id, gt_lane in gt_lanes:
            visible = _visible_count(gt_lane)
            target_start, target_end = _fixed_y_extent_target(gt_lane, h_samples)
            best_ape, best_query_list_idx, best_overlap = _best_ape(pred_raw_xs, gt_lane, min_overlap=args.match_min_overlap)
            best_query = raw_queries[best_query_list_idx] if best_query_list_idx >= 0 else None
            query_id = -1 if best_query is None else int(best_query["query"])
            pred_start = -1 if query_id < 0 else int(pred_lo[query_id].item())
            pred_end = -1 if query_id < 0 else int(pred_hi[query_id].item())
            row = {
                "raw_file": raw_file,
                "date": date,
                "session": session,
                "gt_count": gt_count,
                "gt_lane_id": int(gt_lane_id),
                "lane_order": lane_positions.get(int(gt_lane_id), {}).get("lane_order", ""),
                "lane_position": lane_positions.get(int(gt_lane_id), {}).get("lane_position", ""),
                "visible_points_gt": int(visible),
                "short_visible_lane": bool(visible <= int(args.short_visible_max)),
                "raw_best_query_id": query_id,
                "raw_best_ape_px": best_ape,
                "raw_best_overlap": int(best_overlap),
                "raw_has_match_20px": bool(best_ape <= 20.0),
                "extent_gt_start_idx": int(target_start),
                "extent_gt_end_idx": int(target_end),
                "extent_pred_start_idx": pred_start,
                "extent_pred_end_idx": pred_end,
                "extent_start_abs_err": abs(pred_start - int(target_start)) if query_id >= 0 else "",
                "extent_end_abs_err": abs(pred_end - int(target_end)) if query_id >= 0 else "",
                "extent_interval_iou": _interval_iou(pred_start, pred_end, target_start, target_end) if query_id >= 0 else "",
            }
            for mode in modes:
                final_best_ape, final_best_idx, final_best_overlap = _best_ape(
                    final_xs_by_mode[mode],
                    gt_lane,
                    min_overlap=args.match_min_overlap,
                )
                row[f"final_best_ape_px_{mode}"] = final_best_ape
                row[f"final_best_lane_index_{mode}"] = int(final_best_idx)
                row[f"final_best_overlap_{mode}"] = int(final_best_overlap)
                row[f"{match_key}_{mode}"] = bool(final_best_ape <= float(args.match_thr_px))
            lane_rows.append(row)

    gt_hist = Counter(int(row["gt_count"]) for row in image_rows)
    by_group: dict[str, list[dict]] = {
        "short_gt4": [
            row for row in lane_rows if int(row["gt_count"]) == 4 and bool(row["short_visible_lane"])
        ],
        "short_gt5": [
            row for row in lane_rows if int(row["gt_count"]) == 5 and bool(row["short_visible_lane"])
        ],
    }
    by_extent_mode = {
        mode: _mode_summary(
            mode=mode,
            by_group=by_group,
            lane_rows=lane_rows,
            image_rows=image_rows,
            gt_hist=gt_hist,
            extra_by_gt_mode=extra_by_gt_mode,
            endpoint_tol=int(args.endpoint_acc_tol),
            match_thr_px=float(args.match_thr_px),
        )
        for mode in modes
    }
    primary_summary = by_extent_mode[primary_mode]
    none_summary = by_extent_mode.get("none", primary_summary)

    summary = {
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "decode_yaml": None if args.decode_yaml is None else str(Path(args.decode_yaml).resolve()),
            "archive_root": str(archive_root),
            "gt_json": str(gt_path),
            "split": str(args.split),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "imgsz_display": shape_str(imgsz),
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "valid_before_maxdet": bool(args.valid_before_maxdet),
            "extent_decode_modes": modes,
            "primary_extent_mode": primary_mode,
            "match_thr_px": float(args.match_thr_px),
            "match_min_overlap": int(args.match_min_overlap),
            "short_visible_max": int(args.short_visible_max),
            "endpoint_acc_tol": int(args.endpoint_acc_tol),
            "endpoint_metric_note": (
                "Endpoint metrics are computed on the raw-best query. Use raw_geometry_strata.raw_has_match20_true "
                "to inspect extent endpoint quality after raw geometry already has a 20px candidate; "
                "raw_has_match20_false mixes endpoint errors with raw geometry misses."
            ),
            "gt_contract": gt_contract,
        },
        "images": int(len(image_rows)),
        "gt_lanes": int(len(lane_rows)),
        "gt_count_hist": {str(k): int(v) for k, v in sorted(gt_hist.items())},
        "pred_count_hist": {
            mode: {str(k): int(v) for k, v in sorted(counter.items())}
            for mode, counter in pred_count_by_mode.items()
        },
        "by_extent_mode": by_extent_mode,
        "short_gt4": primary_summary["short_gt4"],
        "short_gt5": primary_summary["short_gt5"],
        "short_gt4_raw_geometry_strata": primary_summary["short_gt4_raw_geometry_strata"],
        "short_gt5_raw_geometry_strata": primary_summary["short_gt5_raw_geometry_strata"],
        "short_gt4_has_match20": primary_summary["short_gt4_has_match20"],
        "short_gt5_has_match20": primary_summary["short_gt5_has_match20"],
        "short_gt4_best_ape_p90": primary_summary["short_gt4_best_ape_p90"],
        "short_gt5_best_ape_p90": primary_summary["short_gt5_best_ape_p90"],
        "short_gt4_endpoint_start_acc_1": primary_summary["short_gt4_endpoint_start_acc_1"],
        "short_gt4_endpoint_end_acc_1": primary_summary["short_gt4_endpoint_end_acc_1"],
        "short_gt5_endpoint_start_acc_1": primary_summary["short_gt5_endpoint_start_acc_1"],
        "short_gt5_endpoint_end_acc_1": primary_summary["short_gt5_endpoint_end_acc_1"],
        "short_gt4_interval_iou": primary_summary["short_gt4_interval_iou"],
        "short_gt5_interval_iou": primary_summary["short_gt5_interval_iou"],
        "after_extent_survival": primary_summary["final_survival"],
        "after_point_valid_survival": none_summary["final_survival"],
        "normal_gt3_extra_rate": primary_summary["normal_gt3_extra_rate"],
        "normal_gt4_extra_rate": primary_summary["normal_gt4_extra_rate"],
        "count_acc_4": primary_summary["count_acc_4"],
        "count_acc_5": primary_summary["count_acc_5"],
        "avg_inference_ms": round(float(infer_time_s) * 1000.0 / max(len(image_rows), 1), 4),
        "avg_postprocess_ms": round(float(post_time_s) * 1000.0 / max(len(image_rows), 1), 4),
        "artifacts": {
            "lane_rows_csv": str((save_dir / "lane_extent_rows.csv").resolve()),
            "image_rows_csv": str((save_dir / "image_extent_rows.csv").resolve()),
        },
    }

    _write_csv(save_dir / "lane_extent_rows.csv", lane_rows)
    _write_csv(save_dir / "image_extent_rows.csv", image_rows)
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
