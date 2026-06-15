from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

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
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = (
    ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_q12_k56_fifthness_gt5neg_ft8_seed1_b32w4" / "weights" / "official_best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Official-val-only audit of optional fifthness scores for GT4 false fifth candidates "
            "and GT5 selected fifth candidates."
        )
    )
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Must be val.")
    parser.add_argument("--gt-json", default=None, help="Official-val TuSimple GT json-lines file.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Fifthness-enabled GCS checkpoint .pt.")
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
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.50, 0.70, 0.85],
        help="Fifthness thresholds summarized against false/true selected fifth samples.",
    )
    parser.add_argument("--save-dir", default=None, help="Output directory for CSV and summary JSON.")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _score(lane: dict | None, key: str) -> float | None:
    if lane is None or lane.get(key) is None:
        return None
    try:
        return float(lane[key])
    except (TypeError, ValueError):
        return None


def _score_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 6),
        "q05": round(float(np.quantile(arr, 0.05)), 6),
        "q10": round(float(np.quantile(arr, 0.10)), 6),
        "q25": round(float(np.quantile(arr, 0.25)), 6),
        "median": round(float(np.quantile(arr, 0.50)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "q75": round(float(np.quantile(arr, 0.75)), 6),
        "q90": round(float(np.quantile(arr, 0.90)), 6),
        "q95": round(float(np.quantile(arr, 0.95)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _pairwise_auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg))
        wins += 0.5 * float(np.sum(value == neg))
    return round(float(wins / (float(pos.size) * float(neg.size))), 6)


def _threshold_rows(true_scores: list[float], false_scores: list[float], thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        thr = float(threshold)
        true_keep = sum(float(x) >= thr for x in true_scores)
        false_keep = sum(float(x) >= thr for x in false_scores)
        rows.append(
            {
                "threshold": round(thr, 6),
                "true_keep": int(true_keep),
                "true_total": int(len(true_scores)),
                "true_keep_rate": None if not true_scores else round(float(true_keep) / len(true_scores), 6),
                "false_keep": int(false_keep),
                "false_total": int(len(false_scores)),
                "false_keep_rate": None if not false_scores else round(float(false_keep) / len(false_scores), 6),
                "false_suppressed": int(len(false_scores) - false_keep),
                "false_suppressed_rate": None if not false_scores else round(1.0 - float(false_keep) / len(false_scores), 6),
            }
        )
    return rows


def _lane_row(
    *,
    raw_file: str,
    gt_count: int,
    pred_count: int,
    group: str,
    pred_index: int | None,
    lane: dict | None,
    matched: bool | None,
) -> dict:
    return {
        "raw_file": raw_file,
        "gt_count": int(gt_count),
        "pred_count": int(pred_count),
        "group": group,
        "pred_index": pred_index,
        "query": None if lane is None else lane.get("query"),
        "rank_selection_rank": None if lane is None else lane.get("rank_selection_rank"),
        "matched": None if matched is None else int(bool(matched)),
        "fifthness_score": _score(lane, "fifthness_score"),
        "rank_score": _score(lane, "rank_score"),
        "quality_score": _score(lane, "quality_score"),
        "exist_score": _score(lane, "exist_score"),
        "valid_count": None if lane is None else lane.get("valid_count"),
        "source": None if lane is None else lane.get("source", ""),
        "quality_rescue_5th": None if lane is None else int(bool(lane.get("quality_rescue_5th", False))),
        "count_head_policy_count": None if lane is None else lane.get("count_head_policy_count"),
        "effective_policy_count": None if lane is None else lane.get("effective_policy_count"),
    }


def _resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    run_dir = Path(args.weights).resolve().parents[1] if Path(args.weights).name.endswith(".pt") else None
    tag = (
        f"fifthness_score_audit_{args.split}"
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
    split = validate_tusimple_selection_source(
        args.split,
        gt_json=args.gt_json,
        context="Fifthness score audit",
    )
    if split != "val":
        raise ValueError("Fifthness score audit is official-val-only. Use --split val.")

    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_json = Path(args.gt_json) if args.gt_json else default_tusimple_gt_json(archive_root, split=split)
    gt_records = [normalize_tusimple_gt_record(x) for x in read_tusimple_json_lines(gt_json)]
    validate_tusimple_selection_source(
        split,
        gt_json=gt_json,
        gt_records=gt_records,
        archive_root=archive_root,
        context="Fifthness score audit",
    )
    if args.max_images and args.max_images > 0:
        gt_records = gt_records[: int(args.max_images)]

    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    device = select_device(args.device, verbose=False)
    warn_max_det_mismatch(args.weights, max_det=args.max_det, context="Fifthness score audit")
    model = load_gcs_model(args.weights, device=device, half=bool(args.half), gcs_imgsz=imgsz)
    rank_min_points = parse_rank_min_points(args.rank_min_points)

    rows: list[dict] = []
    image_rows: list[dict] = []
    score_groups: dict[str, list[float]] = {
        "gt4_unmatched_output5": [],
        "gt4_selected_rank5_output5": [],
        "gt5_selected_rank5_matched_output5": [],
        "gt5_selected_rank5_output5": [],
        "gt5_pre_nms_top5": [],
        "gt4_pre_nms_top5": [],
    }

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
        if "pred_fifthness_logits" not in preds:
            raise ValueError("Fifthness score audit requires a model that emits pred_fifthness_logits.")

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
        tusimple_lanes = gcs_lanes_to_tusimple_lanes(
            decoded,
            h_samples=list(gt["h_samples"]),
            image_shape=original_shape,
        )
        matched = matched_tusimple_prediction_indices(tusimple_lanes, gt)
        gt_count = int(len(gt["lanes"]))
        pred_count = int(len(tusimple_lanes))
        acc, fp, fn = TuSimpleOfficialLaneEval.bench(
            pred=tusimple_lanes,
            gt=[list(x) for x in gt["lanes"]],
            y_samples=list(gt["h_samples"]),
            running_time=1.0,
        )

        selected_rank5 = None
        selected_rank5_index = None
        for pred_idx, lane in enumerate(decoded[: len(tusimple_lanes)]):
            if int(lane.get("rank_selection_rank", 0)) == 5:
                selected_rank5 = lane
                selected_rank5_index = int(pred_idx)
                break
        if selected_rank5 is None and len(decoded) >= 5:
            selected_rank5 = decoded[4]
            selected_rank5_index = 4

        image_rows.append(
            {
                "raw_file": raw_file,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "accuracy": round(float(acc), 6),
                "fp": round(float(fp), 6),
                "fn": round(float(fn), 6),
                "matched_pred_count": int(len(matched)),
                "selected_rank5_query": None if selected_rank5 is None else selected_rank5.get("query"),
                "selected_rank5_matched": None
                if selected_rank5_index is None
                else int(selected_rank5_index in matched),
                "selected_rank5_fifthness": _score(selected_rank5, "fifthness_score"),
                "top5_candidate_fifthness_before_nms": decode_meta.get("top5_candidate_fifthness_before_nms"),
                "effective_policy_count": decode_meta.get("effective_policy_count"),
                "count_head_policy_count": decode_meta.get("count_head_policy_count"),
            }
        )

        top5_score = decode_meta.get("top5_candidate_fifthness_before_nms")
        if top5_score is not None and gt_count in {4, 5}:
            score_groups[f"gt{gt_count}_pre_nms_top5"].append(float(top5_score))

        if gt_count == 4 and pred_count == 5:
            if selected_rank5 is not None:
                rows.append(
                    _lane_row(
                        raw_file=raw_file,
                        gt_count=gt_count,
                        pred_count=pred_count,
                        group="gt4_selected_rank5_output5",
                        pred_index=selected_rank5_index,
                        lane=selected_rank5,
                        matched=None if selected_rank5_index is None else selected_rank5_index in matched,
                    )
                )
                fifthness = _score(selected_rank5, "fifthness_score")
                if fifthness is not None:
                    score_groups["gt4_selected_rank5_output5"].append(fifthness)
            for pred_idx, lane in enumerate(decoded[: len(tusimple_lanes)]):
                if pred_idx not in matched:
                    rows.append(
                        _lane_row(
                            raw_file=raw_file,
                            gt_count=gt_count,
                            pred_count=pred_count,
                            group="gt4_unmatched_output5",
                            pred_index=pred_idx,
                            lane=lane,
                            matched=False,
                        )
                    )
                    fifthness = _score(lane, "fifthness_score")
                    if fifthness is not None:
                        score_groups["gt4_unmatched_output5"].append(fifthness)

        if gt_count == 5 and pred_count == 5 and selected_rank5 is not None:
            is_matched = selected_rank5_index in matched if selected_rank5_index is not None else False
            rows.append(
                _lane_row(
                    raw_file=raw_file,
                    gt_count=gt_count,
                    pred_count=pred_count,
                    group="gt5_selected_rank5_output5",
                    pred_index=selected_rank5_index,
                    lane=selected_rank5,
                    matched=is_matched,
                )
            )
            fifthness = _score(selected_rank5, "fifthness_score")
            if fifthness is not None:
                score_groups["gt5_selected_rank5_output5"].append(fifthness)
                if is_matched:
                    score_groups["gt5_selected_rank5_matched_output5"].append(fifthness)
                    rows.append(
                        _lane_row(
                            raw_file=raw_file,
                            gt_count=gt_count,
                            pred_count=pred_count,
                            group="gt5_selected_rank5_matched_output5",
                            pred_index=selected_rank5_index,
                            lane=selected_rank5,
                            matched=True,
                        )
                    )

        if index % 100 == 0 or index == len(gt_records):
            print(f"processed {index}/{len(gt_records)}")

    false_scores = score_groups["gt4_unmatched_output5"]
    true_scores = score_groups["gt5_selected_rank5_matched_output5"]
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
            "use_fifthness_decode": bool(args.use_fifthness_decode),
            "fifthness_decode_thr": float(args.fifthness_decode_thr),
            "fifthness_decode_rank_weight": float(args.fifthness_decode_rank_weight),
            "official_val_only": True,
            "not_for_selection": True,
        },
        "images": int(len(gt_records)),
        "score_groups": {name: _score_summary(values) for name, values in score_groups.items()},
        "threshold_curve": _threshold_rows(true_scores, false_scores, args.thresholds),
        "pairwise_auc_true_gt5_rank5_matched_vs_false_gt4_unmatched": _pairwise_auc(true_scores, false_scores),
        "decision_hint": (
            "Use this as a diagnostic only. A useful training/decode hypothesis should show high true_keep_rate "
            "and low false_keep_rate at the same threshold before launching another training gate."
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
        "fifthness_score",
        "rank_score",
        "quality_score",
        "exist_score",
        "valid_count",
        "source",
        "quality_rescue_5th",
        "count_head_policy_count",
        "effective_policy_count",
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
        "selected_rank5_fifthness",
        "top5_candidate_fifthness_before_nms",
        "effective_policy_count",
        "count_head_policy_count",
    ]
    _write_csv(save_dir / "fifthness_score_audit_lanes.csv", rows, lane_fields)
    _write_csv(save_dir / "fifthness_score_audit_images.csv", image_rows, image_fields)
    (save_dir / "fifthness_score_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved {save_dir / 'fifthness_score_audit_summary.json'}")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary["score_groups"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["threshold_curve"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
