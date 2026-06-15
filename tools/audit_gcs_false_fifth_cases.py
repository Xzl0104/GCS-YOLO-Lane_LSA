from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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
    default_tusimple_gt_json,
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    normalize_tusimple_gt_record,
    read_tusimple_json_lines,
    tusimple_image_path,
    validate_tusimple_selection_source,
)
from tools.eval_tusimple_official import parse_rank_min_points  # noqa: E402
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from tools.sweep_tusimple_official import matched_tusimple_prediction_indices  # noqa: E402
from ultralytics.utils.gcs_postprocess import GCS_DEFAULT_MAX_DET, decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = (
    ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4" / "weights" / "official_best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Official-val-only case audit for GT4 false fifth lanes and GT5 true fifth lanes. "
            "This diagnostic does not require pred_fifthness_logits and is not for selection."
        )
    )
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Must be val.")
    parser.add_argument("--gt-json", default=None, help="Official-val TuSimple GT json-lines file.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS inference shape as H W.")
    parser.add_argument("--device", default="", help="Torch device string.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional smoke cap; do not use for decisions.")
    parser.add_argument("--conf", type=float, default=0.005, help="Lane existence threshold.")
    parser.add_argument("--point-valid-thr", type=float, default=0.35, help="Per-point visibility threshold.")
    parser.add_argument("--nms-dist-px", type=float, default=18.0, help="Lane-NMS distance in original-image pixels.")
    parser.add_argument("--max-det", type=int, default=GCS_DEFAULT_MAX_DET, help="Maximum decoded lanes.")
    parser.add_argument("--min-points", type=int, default=6, help="Legacy/default visible-anchor floor.")
    parser.add_argument("--rank-min-points", default="", help="Optional per-selected-rank min_points override.")
    parser.add_argument("--candidate-conf", type=float, default=0.005)
    parser.add_argument("--candidate-point-valid-thr", type=float, default=0.35)
    parser.add_argument("--candidate-min-points", type=int, default=5)
    parser.add_argument("--final-min-points", type=int, default=6)
    parser.add_argument("--fifth-min-points", type=int, default=5)
    parser.add_argument("--count-head-temp", type=float, default=1.0)
    count_head = parser.add_mutually_exclusive_group()
    count_head.add_argument("--use-count-head-decode", dest="use_count_head_decode", action="store_true")
    count_head.add_argument("--no-count-head-decode", dest="use_count_head_decode", action="store_false")
    parser.set_defaults(use_count_head_decode=True)
    rescue_pool = parser.add_mutually_exclusive_group()
    rescue_pool.add_argument("--enable-rescue-candidate-pool", dest="enable_rescue_candidate_pool", action="store_true")
    rescue_pool.add_argument("--no-enable-rescue-candidate-pool", dest="enable_rescue_candidate_pool", action="store_false")
    parser.set_defaults(enable_rescue_candidate_pool=True)
    parser.add_argument("--rescue-candidate-conf", type=float, default=0.005)
    parser.add_argument("--rescue-candidate-point-valid-thr", type=float, default=0.08)
    parser.add_argument("--rescue-candidate-min-points", type=int, default=4)
    parser.add_argument("--line-nms-min-overlap", type=int, default=6)
    parser.add_argument("--line-nms-rescue-dist-px", type=float, default=30.0)
    parser.add_argument("--quality-rescue-5th", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-rescue-count5-thr", type=float, default=0.70)
    parser.add_argument("--quality-rescue-conf-thr", type=float, default=0.03)
    parser.add_argument("--quality-rescue-mean-valid-thr", type=float, default=0.45)
    parser.add_argument("--quality-rescue-quality-thr", type=float, default=0.55)
    parser.add_argument("--quality-rescue-min-points", type=int, default=5)
    parser.add_argument("--quality-rescue-dist-px", type=float, default=24.0)
    parser.add_argument("--use-fifthness-decode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fifthness-decode-thr", type=float, default=0.0)
    parser.add_argument("--fifthness-decode-rank-weight", type=float, default=1.0)
    parser.add_argument("--save-dir", default=None, help="Output directory for CSV and summary JSON.")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _score(source: dict | None, key: str) -> float | None:
    if source is None or source.get(key) is None:
        return None
    try:
        return float(source[key])
    except (TypeError, ValueError):
        return None


def _count_prob_value(source: dict | None, count: int) -> float | None:
    if source is None:
        return None
    prob = source.get("count_head_prob")
    if prob is None:
        return None
    try:
        return float(list(prob)[int(count) - 2])
    except (IndexError, TypeError, ValueError):
        return None


def _visible_points_px(lane: dict | None) -> np.ndarray:
    if lane is None:
        return np.zeros((0, 2), dtype=np.float32)
    points = lane.get("visible_points")
    if points is None:
        points = lane.get("points")
    arr = np.asarray(points if points is not None else [], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    finite = np.isfinite(arr[:, :2]).all(axis=1)
    return arr[finite, :2]


def _x_at_y(lane: dict, y_px: float) -> float | None:
    points = _visible_points_px(lane)
    if points.shape[0] == 0:
        return None
    order = np.argsort(points[:, 1], kind="stable")
    ys = points[order, 1]
    xs = points[order, 0]
    ys, unique_idx = np.unique(ys, return_index=True)
    xs = xs[unique_idx]
    if ys.shape[0] == 1:
        return float(xs[0]) if abs(float(y_px) - float(ys[0])) <= 1.0 else None
    if float(y_px) < float(ys[0]) or float(y_px) > float(ys[-1]):
        return None
    return float(np.interp(float(y_px), ys, xs))


def _outside_edge_stats(lane: dict | None, selected_top4: list[dict]) -> tuple[str | None, float | None, float | None]:
    points = _visible_points_px(lane)
    if points.shape[0] == 0 or not selected_top4:
        return None, None, None
    left_gaps: list[float] = []
    right_gaps: list[float] = []
    candidate_xs: list[float] = []
    for candidate_x, candidate_y in points[:6]:
        selected_xs = [
            x
            for x in (_x_at_y(item, float(candidate_y)) for item in selected_top4)
            if x is not None and np.isfinite(float(x))
        ]
        if not selected_xs:
            continue
        candidate_x = float(candidate_x)
        candidate_xs.append(candidate_x)
        left_gaps.append(float(min(selected_xs)) - candidate_x)
        right_gaps.append(candidate_x - float(max(selected_xs)))
    if not candidate_xs:
        return None, None, None
    left_gap = float(median(left_gaps))
    right_gap = float(median(right_gaps))
    candidate_x = float(median(candidate_xs))
    if left_gap > 0.0 and left_gap >= right_gap:
        return "left", left_gap, candidate_x
    if right_gap > 0.0 and right_gap > left_gap:
        return "right", right_gap, candidate_x
    return "inside", max(left_gap, right_gap), candidate_x


def _ranked_selected_lanes(decoded: list[dict]) -> list[dict]:
    return sorted(
        [lane for lane in decoded if lane.get("rank_selection_rank") is not None],
        key=lambda lane: int(lane.get("rank_selection_rank", 999)),
    )


def _selected_rank5(decoded: list[dict]) -> tuple[int | None, dict | None]:
    for index, lane in enumerate(decoded):
        if int(lane.get("rank_selection_rank", 0)) == 5:
            return index, lane
    return (4, decoded[4]) if len(decoded) >= 5 else (None, None)


def _lane_row(
    *,
    raw_file: str,
    gt_count: int,
    pred_count: int,
    group: str,
    pred_index: int | None,
    lane: dict | None,
    matched: bool | None,
    selected_top4: list[dict],
) -> dict:
    side, outside_gap, candidate_x = _outside_edge_stats(lane, selected_top4)
    return {
        "raw_file": raw_file,
        "gt_count": int(gt_count),
        "pred_count": int(pred_count),
        "group": group,
        "pred_index": pred_index,
        "query": None if lane is None else lane.get("query"),
        "rank_selection_rank": None if lane is None else lane.get("rank_selection_rank"),
        "matched": None if matched is None else int(bool(matched)),
        "rank_score": _score(lane, "rank_score"),
        "quality_score": _score(lane, "quality_score"),
        "exist_score": _score(lane, "exist_score"),
        "fifthness_score": _score(lane, "fifthness_score"),
        "valid_count": None if lane is None else lane.get("valid_count"),
        "rank_valid_count": None if lane is None else lane.get("rank_valid_count"),
        "mean_valid_score": _score(lane, "mean_valid_score"),
        "mean_valid_score_all": _score(lane, "mean_valid_score_all"),
        "valid_count_score": _score(lane, "valid_count_score"),
        "source": None if lane is None else lane.get("source", ""),
        "quality_rescue_5th": None if lane is None else int(bool(lane.get("quality_rescue_5th", False))),
        "count_head_policy_count": None if lane is None else lane.get("count_head_policy_count"),
        "effective_policy_count": None if lane is None else lane.get("effective_policy_count"),
        "count_head_prob_4": _count_prob_value(lane, 4),
        "count_head_prob_5": _count_prob_value(lane, 5),
        "count_head_margin": _score(lane, "count_head_margin"),
        "count5_margin": _score(lane, "count5_margin"),
        "edge_side": side,
        "outside_gap_px": outside_gap,
        "candidate_median_x_px": candidate_x,
    }


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 6),
        "median": round(float(np.quantile(arr, 0.50)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _summarize_rows(rows: list[dict], fields: list[str]) -> dict:
    out: dict[str, dict] = {}
    groups = sorted({str(row["group"]) for row in rows})
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        out[group] = {"rows": int(len(group_rows))}
        for field in fields:
            values = [float(row[field]) for row in group_rows if row.get(field) is not None]
            out[group][field] = _summary(values)
    return out


def _resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    weights = Path(args.weights)
    run_dir = weights.resolve().parents[1] if weights.name.endswith(".pt") else None
    tag = (
        f"false_fifth_case_audit_{args.split}"
        f"_conf{str(args.conf).replace('.', 'p')}"
        f"_pv{str(args.point_valid_thr).replace('.', 'p')}"
        f"_final{args.final_min_points}_fifth{args.fifth_min_points}"
    )
    if run_dir is not None and run_dir.name == "weights":
        return run_dir.parent / tag
    if run_dir is not None:
        return run_dir / tag
    return ROOT / "runs" / "gcs_lane" / tag


def run(args: argparse.Namespace) -> dict:
    split = validate_tusimple_selection_source(args.split, gt_json=args.gt_json, context="False fifth case audit")
    if split != "val":
        raise ValueError("False fifth case audit is official-val-only. Use --split val.")

    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_json = Path(args.gt_json) if args.gt_json else default_tusimple_gt_json(archive_root, split=split)
    gt_records = [normalize_tusimple_gt_record(x) for x in read_tusimple_json_lines(gt_json)]
    validate_tusimple_selection_source(
        split,
        gt_json=gt_json,
        gt_records=gt_records,
        archive_root=archive_root,
        context="False fifth case audit",
    )
    if args.max_images and args.max_images > 0:
        gt_records = gt_records[: int(args.max_images)]

    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    device = select_device(args.device, verbose=False)
    warn_max_det_mismatch(args.weights, max_det=args.max_det, context="False fifth case audit")
    model = load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    rank_min_points = parse_rank_min_points(args.rank_min_points)

    image_rows: list[dict] = []
    lane_rows: list[dict] = []
    gt4_to_5 = 0
    gt5_to_4 = 0
    gt4_total = 0
    gt5_total = 0

    for index, gt in enumerate(gt_records, start=1):
        raw_file = str(gt["raw_file"])
        image_path = tusimple_image_path(archive_root, raw_file, split=split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        original_shape = (int(img.shape[0]), int(img.shape[1]))
        tensor = preprocess_image(img, imgsz=imgsz, device=device, half=bool(args.half))
        with torch.no_grad():
            preds = model(tensor)

        pred_valid = preds.get("pred_valid_logits")
        pred_count = preds.get("pred_count_logits")
        pred_count_boundary = preds.get("pred_count_boundary_logits")
        pred_quality = preds.get("pred_quality_logits")
        pred_fifthness = preds.get("pred_fifthness_logits")
        decoded, decode_meta = decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
            pred_count_logits=pred_count[0] if pred_count is not None else None,
            pred_count_boundary_logits=pred_count_boundary[0] if pred_count_boundary is not None else None,
            pred_quality_logits=pred_quality[0] if pred_quality is not None else None,
            pred_fifthness_logits=pred_fifthness[0] if pred_fifthness is not None else None,
            image_shape=original_shape,
            score_thr=float(args.conf),
            point_valid_thr=float(args.point_valid_thr),
            min_points=int(args.min_points),
            max_det=int(args.max_det),
            nms_dist_px=float(args.nms_dist_px),
            count_calibration=None,
            rank_min_points=rank_min_points,
            use_count_head_decode=bool(args.use_count_head_decode),
            count_head_temperature=float(args.count_head_temp),
            dataset_name="tusimple",
            candidate_score_thr=float(args.candidate_conf),
            candidate_point_valid_thr=float(args.candidate_point_valid_thr),
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
            use_fifthness_decode=bool(args.use_fifthness_decode),
            fifthness_decode_thr=float(args.fifthness_decode_thr),
            fifthness_decode_rank_weight=float(args.fifthness_decode_rank_weight),
            return_meta=True,
        )
        tusimple_lanes = gcs_lanes_to_tusimple_lanes(decoded, h_samples=list(gt["h_samples"]), image_shape=original_shape)
        matched = matched_tusimple_prediction_indices(tusimple_lanes, gt)
        gt_count = int(len(gt["lanes"]))
        pred_count_out = int(len(tusimple_lanes))
        acc, fp, fn = TuSimpleOfficialLaneEval.bench(
            pred=tusimple_lanes,
            gt=[list(x) for x in gt["lanes"]],
            y_samples=list(gt["h_samples"]),
            running_time=1.0,
        )
        if gt_count == 4:
            gt4_total += 1
            gt4_to_5 += int(pred_count_out == 5)
        if gt_count == 5:
            gt5_total += 1
            gt5_to_4 += int(pred_count_out == 4)

        rank_sorted = _ranked_selected_lanes(decoded)
        selected_top4 = rank_sorted[:4]
        selected_rank5_index, selected_rank5 = _selected_rank5(decoded)
        selected_rank5_matched = None if selected_rank5_index is None else selected_rank5_index in matched

        image_rows.append(
            {
                "raw_file": raw_file,
                "gt_count": gt_count,
                "pred_count": pred_count_out,
                "accuracy": round(float(acc), 6),
                "fp": round(float(fp), 6),
                "fn": round(float(fn), 6),
                "matched_pred_count": int(len(matched)),
                "selected_rank5_query": None if selected_rank5 is None else selected_rank5.get("query"),
                "selected_rank5_matched": None if selected_rank5_matched is None else int(bool(selected_rank5_matched)),
                "selected_rank5_quality": _score(selected_rank5, "quality_score"),
                "selected_rank5_rank_score": _score(selected_rank5, "rank_score"),
                "selected_rank5_exist": _score(selected_rank5, "exist_score"),
                "selected_rank5_fifthness": _score(selected_rank5, "fifthness_score"),
                "selected_rank5_valid_count": None if selected_rank5 is None else selected_rank5.get("valid_count"),
                "top5_candidate_score_before_nms": decode_meta.get("top5_candidate_score_before_nms"),
                "top5_candidate_quality_before_nms": decode_meta.get("top5_candidate_quality_before_nms"),
                "top5_candidate_valid_points_before_nms": decode_meta.get("top5_candidate_valid_points_before_nms"),
                "effective_policy_count": decode_meta.get("effective_policy_count"),
                "count_head_policy_count": decode_meta.get("count_head_policy_count"),
                "count_head_prob_4": _count_prob_value(decode_meta, 4),
                "count_head_prob_5": _count_prob_value(decode_meta, 5),
                "count_head_margin": decode_meta.get("count_head_margin"),
                "count5_margin": decode_meta.get("count5_margin"),
                "candidate_count_normal": decode_meta.get("candidate_count_normal"),
                "candidate_count_after_nms": decode_meta.get("candidate_count_after_nms"),
                "candidate_pool_shortfall": decode_meta.get("candidate_pool_shortfall"),
                "top5_suppressed_by_nms": int(bool(decode_meta.get("top5_suppressed_by_nms", False))),
                "quality_rescue_success_count": decode_meta.get("quality_rescue_success_count"),
            }
        )

        if gt_count == 4 and pred_count_out == 5:
            if selected_rank5 is not None:
                lane_rows.append(
                    _lane_row(
                        raw_file=raw_file,
                        gt_count=gt_count,
                        pred_count=pred_count_out,
                        group="gt4_selected_rank5_output5",
                        pred_index=selected_rank5_index,
                        lane=selected_rank5,
                        matched=selected_rank5_matched,
                        selected_top4=selected_top4,
                    )
                )
            for pred_idx, lane in enumerate(decoded[: len(tusimple_lanes)]):
                if pred_idx not in matched:
                    lane_rows.append(
                        _lane_row(
                            raw_file=raw_file,
                            gt_count=gt_count,
                            pred_count=pred_count_out,
                            group="gt4_unmatched_output5",
                            pred_index=pred_idx,
                            lane=lane,
                            matched=False,
                            selected_top4=selected_top4,
                        )
                    )

        if gt_count == 5 and selected_rank5 is not None:
            group = "gt5_selected_rank5_output5" if pred_count_out == 5 else "gt5_selected_rank5_not_output5"
            lane_rows.append(
                _lane_row(
                    raw_file=raw_file,
                    gt_count=gt_count,
                    pred_count=pred_count_out,
                    group=group,
                    pred_index=selected_rank5_index,
                    lane=selected_rank5,
                    matched=selected_rank5_matched,
                    selected_top4=selected_top4,
                )
            )
            if pred_count_out == 5 and bool(selected_rank5_matched):
                lane_rows.append(
                    _lane_row(
                        raw_file=raw_file,
                        gt_count=gt_count,
                        pred_count=pred_count_out,
                        group="gt5_selected_rank5_matched_output5",
                        pred_index=selected_rank5_index,
                        lane=selected_rank5,
                        matched=True,
                        selected_top4=selected_top4,
                    )
                )

        if index % 100 == 0 or index == len(gt_records):
            print(f"processed {index}/{len(gt_records)}")

    summary_fields = [
        "rank_score",
        "quality_score",
        "exist_score",
        "fifthness_score",
        "valid_count",
        "mean_valid_score",
        "mean_valid_score_all",
        "count_head_prob_4",
        "count_head_prob_5",
        "count_head_margin",
        "count5_margin",
        "outside_gap_px",
    ]
    summary = {
        "config": {
            "weights": str(args.weights),
            "split": split,
            "gt_json": str(gt_json),
            "archive_root": str(archive_root),
            "imgsz": list(imgsz),
            "max_images": int(args.max_images),
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "candidate_min_points": int(args.candidate_min_points),
            "final_min_points": int(args.final_min_points),
            "fifth_min_points": int(args.fifth_min_points),
            "official_val_only": True,
            "not_for_selection": True,
        },
        "images": int(len(gt_records)),
        "gt4_total": int(gt4_total),
        "gt5_total": int(gt5_total),
        "gt4_to_5": int(gt4_to_5),
        "gt4_to_5_rate": round(float(gt4_to_5) / max(gt4_total, 1), 6),
        "gt5_to_4": int(gt5_to_4),
        "gt5_to_4_rate": round(float(gt5_to_4) / max(gt5_total, 1), 6),
        "lane_group_summaries": _summarize_rows(lane_rows, summary_fields),
        "decision_hint": (
            "Diagnostic only. Compare gt4_unmatched_output5 against gt5_selected_rank5_matched_output5. "
            "A next training hypothesis should lower GT4 P5/margin and false fifth rank/quality without "
            "lowering GT5 true fifth evidence or using test."
        ),
    }

    save_dir = _resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    lane_fields = [
        "raw_file",
        "gt_count",
        "pred_count",
        "group",
        "pred_index",
        "query",
        "rank_selection_rank",
        "matched",
        "rank_score",
        "quality_score",
        "exist_score",
        "fifthness_score",
        "valid_count",
        "rank_valid_count",
        "mean_valid_score",
        "mean_valid_score_all",
        "valid_count_score",
        "source",
        "quality_rescue_5th",
        "count_head_policy_count",
        "effective_policy_count",
        "count_head_prob_4",
        "count_head_prob_5",
        "count_head_margin",
        "count5_margin",
        "edge_side",
        "outside_gap_px",
        "candidate_median_x_px",
    ]
    image_fields = [
        "raw_file",
        "gt_count",
        "pred_count",
        "accuracy",
        "fp",
        "fn",
        "matched_pred_count",
        "selected_rank5_query",
        "selected_rank5_matched",
        "selected_rank5_quality",
        "selected_rank5_rank_score",
        "selected_rank5_exist",
        "selected_rank5_fifthness",
        "selected_rank5_valid_count",
        "top5_candidate_score_before_nms",
        "top5_candidate_quality_before_nms",
        "top5_candidate_valid_points_before_nms",
        "effective_policy_count",
        "count_head_policy_count",
        "count_head_prob_4",
        "count_head_prob_5",
        "count_head_margin",
        "count5_margin",
        "candidate_count_normal",
        "candidate_count_after_nms",
        "candidate_pool_shortfall",
        "top5_suppressed_by_nms",
        "quality_rescue_success_count",
    ]
    _write_csv(save_dir / "false_fifth_case_lanes.csv", lane_rows, lane_fields)
    _write_csv(save_dir / "false_fifth_case_images.csv", image_rows, image_fields)
    (save_dir / "false_fifth_case_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved {save_dir / 'false_fifth_case_summary.json'}")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
