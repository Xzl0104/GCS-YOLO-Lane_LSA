from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import cv2
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
    official_metric_score,
    read_tusimple_json_lines,
    tusimple_image_path,
)
from gcs_tools.tusimple_split_guard import reject_tusimple_test_search_gt_json  # noqa: E402
from tools.eval_tusimple_official import (  # noqa: E402
    COUNT_HEAD_LOGIT_KEYS,
    _count_diagnostics,
    _count_head_prediction,
    _parse_allowed_counts,
)
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03" / "weights" / "best.pt"


def validate_search_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized == "test":
        raise ValueError(
            "TuSimple official threshold search cannot use --split test. "
            "Use --split val for sweeps, then run tools/eval_tusimple_official.py --split test once with the selected row."
        )
    if normalized not in {"train", "val"}:
        raise ValueError(f"Unsupported TuSimple sweep split: {split!r}")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep GCS-YOLO-Lane thresholds with the TuSimple official metric.")
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="TuSimple archive split for search.")
    parser.add_argument("--gt-json", default=None, help="Official TuSimple GT json-lines file for the search split.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS inference shape as H W. Defaults: TuSimple 544 960.",
    )
    parser.add_argument(
        "--confs",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25],
        help="Lane existence thresholds to sweep.",
    )
    parser.add_argument(
        "--point-valid-thrs",
        nargs="+",
        type=float,
        default=[0.30, 0.35, 0.40, 0.45, 0.50],
        help="Per-point visibility thresholds to sweep.",
    )
    parser.add_argument(
        "--nms-dist-pxs",
        nargs="+",
        type=float,
        default=[18.0],
        help="Lane-NMS distances in original-image pixels to sweep. 0 disables NMS.",
    )
    parser.add_argument("--max-dets", nargs="+", type=int, default=[8], help="max_det values to sweep.")
    parser.add_argument("--min-points", nargs="+", type=int, default=[6], help="Minimum visible-anchor floors to sweep.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of GT records. 0 means all.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of untimed warmup forwards.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--runtime-ms", type=float, default=1.0, help="Constant TuSimple run_time in ms.")
    parser.add_argument("--save-dir", default=None, help="Output directory. Defaults under the checkpoint run directory.")
    parser.add_argument(
        "--score-fp-weight",
        type=float,
        default=DEFAULT_OFFICIAL_SCORE_FP_WEIGHT,
        help="FP penalty in official_score = official_acc - w_fp * FP - w_fn * FN.",
    )
    parser.add_argument(
        "--score-fn-weight",
        type=float,
        default=DEFAULT_OFFICIAL_SCORE_FN_WEIGHT,
        help="FN penalty in official_score = official_acc - w_fp * FP - w_fn * FN.",
    )
    parser.add_argument(
        "--count-guided-topk",
        action="store_true",
        help="Also sweep count-head guided top-K rows for every base threshold combo. Default off.",
    )
    parser.add_argument(
        "--count-guided-min-probs",
        nargs="+",
        type=float,
        default=[0.0],
        help="Count-head probability thresholds for count-guided topK rows.",
    )
    parser.add_argument(
        "--count-guided-allowed-counts",
        default="3,4,5",
        help="Comma-separated count-head counts allowed for count-guided topK. Default: 3,4,5.",
    )
    parser.add_argument(
        "--count-guided-allow-unsupported-fallback",
        action="store_true",
        help="Allow count-guided rows to fall back to normal lanes if no count-head logits are present.",
    )
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _weight_run_dir(weights: str | Path) -> Path | None:
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def resolve_save_dir(save_dir: str | Path | None, weights: str | Path, split: str) -> Path:
    if save_dir is not None and str(save_dir).strip():
        return Path(save_dir)
    run_dir = _weight_run_dir(weights)
    if run_dir is not None:
        return run_dir / f"official_sweep_{split}"
    return ROOT / "runs" / "gcs_lane" / "tusimple_official_sweep" / Path(weights).stem / f"official_sweep_{split}"


def _limit_records(records: list[dict], max_images: int) -> list[dict]:
    if max_images and max_images > 0:
        return records[: int(max_images)]
    return records


def _base_combo_key(combo: dict) -> tuple[float, float, float, int, int]:
    return (
        float(combo["conf"]),
        float(combo["point_valid_thr"]),
        float(combo["nms_dist_px"]),
        int(combo["max_det"]),
        int(combo["min_points"]),
    )


def _combo_key(combo: dict) -> tuple[float, float, float, int, int, str, float]:
    min_prob = combo.get("count_guided_min_prob")
    return (*_base_combo_key(combo), str(combo["mode"]), -1.0 if min_prob is None else float(min_prob))


def build_base_combos(args: argparse.Namespace) -> list[dict]:
    combos: list[dict] = []
    for conf, point_valid_thr, nms_dist_px, max_det, min_points in product(
        sorted({float(x) for x in args.confs}),
        sorted({float(x) for x in args.point_valid_thrs}),
        sorted({float(x) for x in args.nms_dist_pxs}),
        sorted({int(x) for x in args.max_dets}),
        sorted({int(x) for x in args.min_points}),
    ):
        if point_valid_thr < 0.0 or point_valid_thr > 1.0:
            raise ValueError(f"point-valid thresholds must be in [0, 1], got {point_valid_thr}.")
        if nms_dist_px < 0.0:
            raise ValueError(f"nms-dist-pxs must be >= 0, got {nms_dist_px}.")
        if max_det <= 0 or min_points <= 0:
            raise ValueError(f"max_det and min_points must be positive, got max_det={max_det}, min_points={min_points}.")
        combos.append(
            {
                "conf": conf,
                "point_valid_thr": point_valid_thr,
                "nms_dist_px": nms_dist_px,
                "max_det": max_det,
                "min_points": min_points,
            }
        )
    if not combos:
        raise ValueError("No threshold combinations to evaluate.")
    return combos


def build_combos(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """Return base threshold combos and expanded evaluation combos."""
    base_combos = build_base_combos(args)
    eval_combos: list[dict] = []
    for combo in base_combos:
        eval_combos.append({**combo, "mode": "normal", "count_guided_min_prob": None})
        if args.count_guided_topk:
            min_probs = sorted({float(x) for x in args.count_guided_min_probs})
            for min_prob in min_probs:
                if not 0.0 <= min_prob <= 1.0:
                    raise ValueError(f"--count-guided-min-probs must be in [0, 1], got {min_prob}.")
                eval_combos.append({**combo, "mode": "count_guided_topk", "count_guided_min_prob": min_prob})
    return base_combos, eval_combos


def select_best(rows: list[dict]) -> dict:
    """Pick the official-Accuracy best row with deterministic tie-breakers."""
    return dict(
        max(
            rows,
            key=lambda r: (
                float(r["official_acc"]),
                float(r["official_score"]),
                -float(r["official_FP"]),
                -float(r["official_FN"]),
                float(r.get("count_acc", 0.0)),
                0 if r.get("mode") == "normal" else -1,
                -float(r["conf"]),
                -float(r["nms_dist_px"]),
                -float(r["point_valid_thr"]),
                -int(r["max_det"]),
                -int(r["min_points"]),
                -float(-1.0 if r.get("count_guided_min_prob") is None else r.get("count_guided_min_prob")),
            ),
        )
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "mode",
        "conf",
        "point_valid_thr",
        "nms_dist_px",
        "max_det",
        "min_points",
        "count_guided_min_prob",
        "count_head_key",
        "count_guided_applied_images",
        "count_guided_fallback_images",
        "count_guided_unsupported_images",
        "count_guided_short_candidate_images",
        "official_acc",
        "official_FP",
        "official_FN",
        "official_score",
        "count_acc",
        "count_acc_2",
        "count_acc_3",
        "count_acc_4",
        "count_acc_5",
        "images",
        "pred_lanes_hist",
        "gt_lanes_hist",
        "count_confusion",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("pred_lanes_hist", "gt_lanes_hist", "count_confusion"):
                out[key] = json.dumps(out.get(key, {}), sort_keys=True, separators=(",", ":"))
            writer.writerow(out)


def _empty_count_guided_stats(requested: bool, min_prob: float | None = None) -> dict:
    return {
        "requested": bool(requested),
        "min_prob": None if min_prob is None else float(min_prob),
        "count_head_key": None,
        "applied_images": 0,
        "fallback_images": 0,
        "unsupported_images": 0,
        "short_candidate_images": 0,
    }


def _apply_count_guided_topk(
    lanes: list[dict],
    count_head_pred: dict,
    min_prob: float,
) -> tuple[list[dict], dict]:
    """Apply count-head topK to a decoded lane list and return row stats."""
    stats = _empty_count_guided_stats(requested=True, min_prob=min_prob)
    stats["count_head_key"] = count_head_pred.get("head_key")
    if float(count_head_pred["prob"]) >= float(min_prob):
        keep_n = max(int(count_head_pred["pred_count"]), 0)
        stats["applied_images"] = 1
        if len(lanes) < keep_n:
            stats["short_candidate_images"] = 1
        return lanes[:keep_n], stats
    stats["fallback_images"] = 1
    return lanes, stats


@torch.inference_mode()
def sweep(args: argparse.Namespace) -> dict:
    args.split = validate_search_split(args.split)
    reject_tusimple_test_search_gt_json(args.gt_json, context="TuSimple official sweep")
    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_path = Path(args.gt_json) if args.gt_json else default_tusimple_gt_json(archive_root, split=args.split)
    gt_records = _limit_records(read_tusimple_json_lines(gt_path), args.max_images)
    if not gt_records:
        raise ValueError(f"No TuSimple GT records found in {gt_path}")

    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    allowed_counts = _parse_allowed_counts(args.count_guided_allowed_counts)
    base_combos, eval_combos = build_combos(args)
    for max_det in sorted({int(c["max_det"]) for c in base_combos}):
        warn_max_det_mismatch(args.weights, max_det=max_det, context="TuSimple official sweep")

    device_obj = select_device(args.device)
    model = load_gcs_model(args.weights, device=device_obj, half=args.half, gcs_imgsz=imgsz)
    if args.warmup > 0 and gt_records:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device_obj, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    combo_by_key = {_combo_key(combo): combo for combo in eval_combos}
    combo_records = {key: [] for key in combo_by_key}
    combo_guided_stats = {
        key: _empty_count_guided_stats(
            requested=combo["mode"] == "count_guided_topk",
            min_prob=combo.get("count_guided_min_prob"),
        )
        for key, combo in combo_by_key.items()
    }
    infer_time_s = 0.0
    post_time_s = 0.0
    for record in gt_records:
        raw_file = str(record["raw_file"])
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        original_shape = img.shape[:2]
        tensor = preprocess_image(img, imgsz, device=device_obj, half=args.half)

        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        t1 = time.perf_counter()

        count_head_pred = None
        count_head_reason = None
        if args.count_guided_topk:
            count_head_pred, count_head_reason = _count_head_prediction(preds, allowed_counts, batch_index=0)
            if count_head_pred is None and not args.count_guided_allow_unsupported_fallback:
                raise RuntimeError(
                    "count-guided sweep requested but no supported count-head logits were found. "
                    f"Reason: {count_head_reason}. Searched keys: {', '.join(COUNT_HEAD_LOGIT_KEYS)}. "
                    "Pass --count-guided-allow-unsupported-fallback only for diagnostic fallback rows."
                )

        pred_valid = preds.get("pred_valid_logits")
        for combo in base_combos:
            lanes = decode_gcs_predictions(
                preds["pred_points"][0],
                preds["pred_logits"][0],
                pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
                image_shape=original_shape,
                score_thr=combo["conf"],
                point_valid_thr=combo["point_valid_thr"],
                min_points=combo["min_points"],
                max_det=combo["max_det"],
                nms_dist_px=combo["nms_dist_px"],
            )
            normal_combo = {**combo, "mode": "normal", "count_guided_min_prob": None}
            tusimple_lanes = gcs_lanes_to_tusimple_lanes(lanes, record["h_samples"], image_shape=original_shape)
            combo_records[_combo_key(normal_combo)].append(
                {
                    "lanes": tusimple_lanes,
                    "h_samples": record["h_samples"],
                    "raw_file": raw_file,
                    "run_time": float(args.runtime_ms),
                }
            )

            if args.count_guided_topk:
                for min_prob in sorted({float(x) for x in args.count_guided_min_probs}):
                    guided_combo = {**combo, "mode": "count_guided_topk", "count_guided_min_prob": min_prob}
                    guided_key = _combo_key(guided_combo)
                    if count_head_pred is None:
                        guided_lanes = lanes
                        image_stats = _empty_count_guided_stats(requested=True, min_prob=min_prob)
                        image_stats["unsupported_images"] = 1
                    else:
                        guided_lanes, image_stats = _apply_count_guided_topk(lanes, count_head_pred, min_prob)
                    for stat_key in (
                        "applied_images",
                        "fallback_images",
                        "unsupported_images",
                        "short_candidate_images",
                    ):
                        combo_guided_stats[guided_key][stat_key] += int(image_stats[stat_key])
                    if image_stats.get("count_head_key"):
                        combo_guided_stats[guided_key]["count_head_key"] = image_stats["count_head_key"]
                    guided_tusimple = gcs_lanes_to_tusimple_lanes(
                        guided_lanes,
                        record["h_samples"],
                        image_shape=original_shape,
                    )
                    combo_records[guided_key].append(
                        {
                            "lanes": guided_tusimple,
                            "h_samples": record["h_samples"],
                            "raw_file": raw_file,
                            "run_time": float(args.runtime_ms),
                        }
                    )
        post_time_s += time.perf_counter() - t1
        infer_time_s += t1 - t0

    rows: list[dict] = []
    for combo in eval_combos:
        key = _combo_key(combo)
        pred_records = combo_records[key]
        result, _ = TuSimpleOfficialLaneEval.bench_records(pred_records, gt_records, strict_length=True, return_records=False)
        metrics = result.as_dict()
        guided_stats = combo_guided_stats[key]
        row = {
            **combo,
            "count_head_key": guided_stats.get("count_head_key"),
            "count_guided_applied_images": int(guided_stats["applied_images"]),
            "count_guided_fallback_images": int(guided_stats["fallback_images"]),
            "count_guided_unsupported_images": int(guided_stats["unsupported_images"]),
            "count_guided_short_candidate_images": int(guided_stats["short_candidate_images"]),
            "official_acc": metrics["Accuracy"],
            "official_FP": metrics["FP"],
            "official_FN": metrics["FN"],
            "images": metrics["images"],
        }
        row["official_score"] = round(
            official_metric_score(
                row["official_acc"],
                row["official_FP"],
                row["official_FN"],
                fp_weight=args.score_fp_weight,
                fn_weight=args.score_fn_weight,
            ),
            6,
        )
        row.update(_count_diagnostics(pred_records, gt_records))
        rows.append(row)

    rows = sorted(
        rows,
        key=lambda r: (
            r["conf"],
            r["point_valid_thr"],
            r["nms_dist_px"],
            r["max_det"],
            r["min_points"],
            0 if r["mode"] == "normal" else 1,
            -1.0 if r.get("count_guided_min_prob") is None else r["count_guided_min_prob"],
        ),
    )
    best = select_best(rows)
    save_dir = resolve_save_dir(args.save_dir, args.weights, args.split)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_csv(save_dir / "tusimple_official_sweep.csv", rows)

    n = max(len(gt_records), 1)
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
            "confs": [float(x) for x in sorted({float(x) for x in args.confs})],
            "point_valid_thrs": [float(x) for x in sorted({float(x) for x in args.point_valid_thrs})],
            "nms_dist_pxs": [float(x) for x in sorted({float(x) for x in args.nms_dist_pxs})],
            "max_dets": [int(x) for x in sorted({int(x) for x in args.max_dets})],
            "min_points": [int(x) for x in sorted({int(x) for x in args.min_points})],
            "count_guided_topk": bool(args.count_guided_topk),
            "count_guided_min_probs": [float(x) for x in sorted({float(x) for x in args.count_guided_min_probs})],
            "count_guided_allowed_counts": [int(x) for x in allowed_counts],
            "count_guided_allow_unsupported_fallback": bool(args.count_guided_allow_unsupported_fallback),
            "count_head_logit_keys": list(COUNT_HEAD_LOGIT_KEYS),
            "base_combo_count": len(base_combos),
            "eval_combo_count": len(eval_combos),
            "runtime_ms": float(args.runtime_ms),
            "max_images": int(args.max_images),
            "device": str(args.device),
            "half": bool(args.half),
            "best_metric": "official_acc",
            "score_fp_weight": float(args.score_fp_weight),
            "score_fn_weight": float(args.score_fn_weight),
        },
        "timing": {
            "avg_inference_ms": round(infer_time_s * 1000.0 / n, 4),
            "avg_sweep_postprocess_ms": round(post_time_s * 1000.0 / n, 4),
        },
    }
    (save_dir / "tusimple_official_sweep_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"swept {len(rows)} combinations on {len(gt_records)} images")
    print(f"saved to: {save_dir.resolve()}")
    return output


def main() -> None:
    sweep(parse_args())


if __name__ == "__main__":
    main()
