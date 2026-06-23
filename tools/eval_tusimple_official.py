from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
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
    write_tusimple_predictions,
)
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03" / "weights" / "best.pt"
COUNT_TOPK_CANDIDATE_POOL = {
    "conf": 0.005,
    "point_valid_thr": 0.5,
    "nms_dist_px": 0.0,
    "max_det": 8,
    "min_points": 6,
}
COUNT_HEAD_LOGIT_KEYS = (
    "pred_count_logits",
    "count_logits",
    "pred_lane_count_logits",
    "lane_count_logits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GCS-YOLO-Lane with the TuSimple official metric.")
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="test", choices=("test", "train", "val"), help="TuSimple archive split.")
    parser.add_argument("--gt-json", default=None, help="Official TuSimple GT json-lines file.")
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
    parser.add_argument("--min-points", type=int, default=6, help="Minimum visible anchors required to keep a lane.")
    parser.add_argument("--max-det", type=int, default=8, help="Maximum decoded lane queries kept before official evaluation.")
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
    parser.add_argument("--save-dir", default=None, help="Output directory. Defaults under the checkpoint run directory.")
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
    parser.add_argument(
        "--oracle-count-topk",
        action="store_true",
        help="Diagnostic only: keep top-K decoded candidates using the GT lane count. Not valid for selection/submission.",
    )
    parser.add_argument(
        "--dump-count-head-stats",
        action="store_true",
        help="If the model output contains count-head logits, report count-head accuracy and confusion.",
    )
    parser.add_argument(
        "--count-guided-topk",
        action="store_true",
        help="Use count-head prediction to keep top predicted-count decoded candidates. Default off.",
    )
    parser.add_argument(
        "--count-guided-min-prob",
        type=float,
        default=0.0,
        help="Minimum allowed-count probability needed before count-guided topK changes the decoded lane count.",
    )
    parser.add_argument(
        "--count-guided-allowed-counts",
        default="3,4,5",
        help="Comma-separated count-head counts allowed for count-guided topK. Default: 3,4,5.",
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


def resolve_save_dir(
    save_dir: str | Path | None,
    weights: str | Path,
    split: str,
    conf: float,
    point_valid_thr: float,
    nms_dist_px: float,
    max_det: int,
    min_points: int,
) -> Path:
    if save_dir is not None and str(save_dir).strip():
        return Path(save_dir)
    tag = (
        f"official_{split}_conf{float(conf):.4g}_pvalid{float(point_valid_thr):.4g}_"
        f"nms{float(nms_dist_px):.4g}_maxdet{int(max_det)}_minp{int(min_points)}"
    ).replace(".", "p")
    run_dir = _weight_run_dir(weights)
    if run_dir is not None:
        return run_dir / tag
    return ROOT / "runs" / "gcs_lane" / "tusimple_official_eval" / Path(weights).stem / tag


def _limit_records(records: list[dict], max_images: int) -> list[dict]:
    if max_images and max_images > 0:
        return records[: int(max_images)]
    return records


def _parse_allowed_counts(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]
        counts = [int(x) for x in parts]
    else:
        counts = [int(x) for x in value]
    counts = sorted(set(counts))
    if not counts:
        raise ValueError("--count-guided-allowed-counts must contain at least one integer count.")
    if any(x < 0 for x in counts):
        raise ValueError(f"count-guided allowed counts must be non-negative, got {counts}.")
    return tuple(counts)


def _validate_count_topk_candidate_pool(args: argparse.Namespace) -> None:
    if not (args.oracle_count_topk or args.count_guided_topk or args.dump_count_head_stats):
        return
    if args.count_guided_topk and not (0.0 <= float(args.count_guided_min_prob) <= 1.0):
        raise ValueError(f"--count-guided-min-prob must be in [0, 1], got {args.count_guided_min_prob}.")
    checks = {
        "conf": float(args.conf),
        "point_valid_thr": float(args.point_valid_thr),
        "nms_dist_px": float(args.nms_dist_px),
        "max_det": int(args.max_det),
        "min_points": int(args.min_points),
    }
    mismatches = []
    for key, expected in COUNT_TOPK_CANDIDATE_POOL.items():
        actual = checks[key]
        if isinstance(expected, float):
            ok = abs(float(actual) - float(expected)) <= 1e-12
        else:
            ok = int(actual) == int(expected)
        if not ok:
            mismatches.append(f"{key}={actual!r} expected {expected!r}")
    if mismatches:
        expected_text = ", ".join(f"{k}={v}" for k, v in COUNT_TOPK_CANDIDATE_POOL.items())
        raise ValueError(
            "count topK diagnostics must use the fixed candidate pool "
            f"({expected_text}); mismatches: {', '.join(mismatches)}."
        )


def _gt_lane_count(record: dict) -> int:
    return sum(1 for lane in record.get("lanes", []) if any(float(x) >= 0.0 for x in lane))


def _count_pairs(pred_records: list[dict], gt_records: list[dict]) -> list[tuple[int, int]]:
    gt_by_raw = {str(x["raw_file"]): x for x in gt_records}
    pairs: list[tuple[int, int]] = []
    for pred in pred_records:
        gt = gt_by_raw[str(pred["raw_file"])]
        pairs.append((_gt_lane_count(gt), len(pred.get("lanes", []))))
    return pairs


def _count_diagnostics(pred_records: list[dict], gt_records: list[dict]) -> dict:
    pairs = _count_pairs(pred_records, gt_records)
    n = max(len(pairs), 1)
    correct = sum(1 for gt, pred in pairs if gt == pred)
    by_gt: dict[str, float] = {}
    for gt_count in sorted({gt for gt, _ in pairs}):
        rows = [pred for gt, pred in pairs if gt == gt_count]
        by_gt[f"count_acc_{gt_count}"] = round(sum(1 for pred in rows if pred == gt_count) / max(len(rows), 1), 6)
    return {
        "count_acc": round(correct / n, 6),
        **by_gt,
        "pred_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(pred for _, pred in pairs).items())},
        "gt_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(gt for gt, _ in pairs).items())},
        "count_confusion": {f"{gt}->{pred}": int(v) for (gt, pred), v in sorted(Counter(pairs).items())},
    }


def _gt4_focus(confusion: dict[str, int]) -> dict[str, int]:
    return {key: int(confusion.get(key, 0)) for key in ("4->3", "4->4", "4->5", "4->6")}


def _official_metrics_for_records(
    pred_records: list[dict],
    gt_records: list[dict],
    score_fp_weight: float,
    score_fn_weight: float,
    return_records: bool,
) -> tuple[dict, list[dict]]:
    result, per_image = TuSimpleOfficialLaneEval.bench_records(
        pred_records,
        gt_records,
        strict_length=True,
        return_records=return_records,
    )
    metrics = result.as_dict()
    metrics["official_acc"] = metrics.pop("Accuracy")
    metrics["official_FP"] = metrics.pop("FP")
    metrics["official_FN"] = metrics.pop("FN")
    metrics["official_score"] = round(
        official_metric_score(
            metrics["official_acc"],
            metrics["official_FP"],
            metrics["official_FN"],
            fp_weight=score_fp_weight,
            fn_weight=score_fn_weight,
        ),
        6,
    )
    metrics.update(_count_diagnostics(pred_records, gt_records))
    return metrics, per_image


def _confusion_rows(confusion: dict[str, int], mode: str) -> list[dict]:
    rows = []
    for pair, count in sorted(confusion.items(), key=lambda x: tuple(int(v) for v in x[0].split("->"))):
        gt_count, pred_count = (int(x) for x in pair.split("->"))
        rows.append(
            {
                "mode": mode,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "images": int(count),
            }
        )
    return rows


def _write_count_confusion_csv(path: Path, mode_outputs: dict[str, dict], count_head_stats: dict | None = None) -> None:
    rows = []
    for mode, item in mode_outputs.items():
        rows.extend(_confusion_rows(item["metrics"].get("count_confusion", {}), mode=mode))
    if count_head_stats and count_head_stats.get("supported"):
        rows.extend(_confusion_rows(count_head_stats.get("count_head_confusion", {}), mode="count_head"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "gt_count", "pred_count", "images"])
        writer.writeheader()
        writer.writerows(rows)


def _count_head_prediction(
    preds: dict,
    allowed_counts: tuple[int, ...],
    batch_index: int = 0,
) -> tuple[dict | None, str | None]:
    for key in COUNT_HEAD_LOGIT_KEYS:
        if key not in preds:
            continue
        logits = preds[key]
        if not isinstance(logits, torch.Tensor):
            return None, f"{key} is not a torch.Tensor"
        logits = logits.detach().float().cpu()
        while logits.ndim > 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        if logits.ndim == 1:
            logits_b = logits
        elif logits.ndim == 2:
            if batch_index >= logits.shape[0]:
                return None, f"{key} batch_index={batch_index} is out of range for shape {tuple(logits.shape)}"
            logits_b = logits[batch_index]
        else:
            return None, f"{key} must have shape C or B x C, got {tuple(logits.shape)}"
        if logits_b.ndim != 1 or logits_b.numel() == 0:
            return None, f"{key} must have a non-empty class dimension, got {tuple(logits_b.shape)}"

        probs = torch.softmax(logits_b, dim=-1)
        if int(probs.numel()) == len(allowed_counts):
            class_counts = list(allowed_counts)
        else:
            class_counts = list(range(int(probs.numel())))
        allowed_indices = [i for i, count in enumerate(class_counts) if int(count) in set(allowed_counts)]
        if not allowed_indices:
            return None, f"{key} has classes {class_counts}, none in allowed counts {list(allowed_counts)}"

        allowed_probs = probs[torch.tensor(allowed_indices, dtype=torch.long)]
        selected_allowed = int(torch.argmax(allowed_probs).item())
        selected_idx = int(allowed_indices[selected_allowed])
        overall_idx = int(torch.argmax(probs).item())
        return (
            {
                "head_key": key,
                "pred_count": int(class_counts[selected_idx]),
                "prob": float(probs[selected_idx].item()),
                "overall_pred_count": int(class_counts[overall_idx]),
                "overall_prob": float(probs[overall_idx].item()),
                "class_counts": [int(x) for x in class_counts],
                "allowed_counts": [int(x) for x in allowed_counts],
            },
            None,
        )
    return None, f"no count-head logits found; searched keys: {', '.join(COUNT_HEAD_LOGIT_KEYS)}"


def _count_head_stats(rows: list[dict], unsupported_reason: str | None) -> dict:
    if not rows:
        return {
            "supported": False,
            "reason": unsupported_reason or "no count-head rows were collected",
            "searched_keys": list(COUNT_HEAD_LOGIT_KEYS),
        }
    pairs = [(int(row["gt_count"]), int(row["pred_count"])) for row in rows]
    n = max(len(pairs), 1)
    correct = sum(1 for gt, pred in pairs if gt == pred)
    out = {
        "supported": True,
        "head_key": rows[0].get("head_key"),
        "count_head_acc": round(correct / n, 6),
        "count_head_confusion": {f"{gt}->{pred}": int(v) for (gt, pred), v in sorted(Counter(pairs).items())},
        "allowed_counts": rows[0].get("allowed_counts"),
        "class_counts": rows[0].get("class_counts"),
    }
    for gt_count in sorted({gt for gt, _ in pairs}):
        bucket = [pred for gt, pred in pairs if gt == gt_count]
        out[f"count_head_acc_{gt_count}"] = round(sum(1 for pred in bucket if pred == gt_count) / max(len(bucket), 1), 6)
    return out


@torch.inference_mode()
def generate_predictions(
    weights: str | Path,
    archive_root: str | Path,
    split: str,
    gt_records: list[dict],
    imgsz: tuple[int, int],
    conf: float,
    point_valid_thr: float,
    nms_dist_px: float,
    max_det: int,
    min_points: int,
    runtime_ms: float,
    use_measured_runtime: bool,
    warmup: int,
    device: str,
    half: bool,
    oracle_count_topk: bool = False,
    count_guided_topk: bool = False,
    count_guided_min_prob: float = 0.0,
    count_guided_allowed_counts: tuple[int, ...] = (3, 4, 5),
    dump_count_head_stats: bool = False,
) -> tuple[dict[str, list[dict]], dict, dict]:
    device_obj = select_device(device)
    model = load_gcs_model(weights, device=device_obj, half=half, gcs_imgsz=imgsz)

    if warmup > 0 and gt_records:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device_obj, half=half)
        for _ in range(int(warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    pred_records_by_mode: dict[str, list[dict]] = {"normal": []}
    if oracle_count_topk:
        pred_records_by_mode["oracle_count_topk"] = []
    if count_guided_topk:
        pred_records_by_mode["count_guided_topk"] = []
    count_head_rows: list[dict] = []
    count_head_unsupported_reason: str | None = None
    need_count_head_global = bool(dump_count_head_stats or count_guided_topk)
    guided_stats = {
        "requested": bool(count_guided_topk),
        "supported": None,
        "applied_images": 0,
        "fallback_images": 0,
        "unsupported_images": 0,
        "short_candidate_images": 0,
        "min_prob": float(count_guided_min_prob),
        "allowed_counts": [int(x) for x in count_guided_allowed_counts],
    }
    infer_time_s = 0.0
    post_time_s = 0.0
    for record in gt_records:
        raw_file = str(record["raw_file"])
        image_path = tusimple_image_path(archive_root, raw_file, split=split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        original_shape = img.shape[:2]
        tensor = preprocess_image(img, imgsz, device=device_obj, half=half)

        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        t1 = time.perf_counter()

        pred_valid = preds.get("pred_valid_logits")
        lanes = decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
            image_shape=original_shape,
            score_thr=conf,
            point_valid_thr=point_valid_thr,
            min_points=min_points,
            max_det=max_det,
            nms_dist_px=nms_dist_px,
        )
        gt_count = _gt_lane_count(record)
        lanes_by_mode: dict[str, list[dict]] = {"normal": lanes}
        count_head_pred = None
        if need_count_head_global:
            count_head_pred, reason = _count_head_prediction(preds, count_guided_allowed_counts, batch_index=0)
            if count_head_pred is None:
                count_head_unsupported_reason = reason
                if count_guided_topk:
                    guided_stats["unsupported_images"] += 1
                    lanes_by_mode["count_guided_topk"] = lanes
            else:
                count_head_rows.append(
                    {
                        "raw_file": raw_file,
                        "gt_count": int(gt_count),
                        "pred_count": int(count_head_pred["pred_count"]),
                        "prob": float(count_head_pred["prob"]),
                        "overall_pred_count": int(count_head_pred["overall_pred_count"]),
                        "overall_prob": float(count_head_pred["overall_prob"]),
                        "head_key": count_head_pred["head_key"],
                        "allowed_counts": count_head_pred["allowed_counts"],
                        "class_counts": count_head_pred["class_counts"],
                    }
                )
                if count_guided_topk:
                    if float(count_head_pred["prob"]) >= float(count_guided_min_prob):
                        keep_n = max(int(count_head_pred["pred_count"]), 0)
                        guided_lanes = lanes[:keep_n]
                        guided_stats["applied_images"] += 1
                        if len(lanes) < keep_n:
                            guided_stats["short_candidate_images"] += 1
                    else:
                        guided_lanes = lanes
                        guided_stats["fallback_images"] += 1
                    lanes_by_mode["count_guided_topk"] = guided_lanes

        if oracle_count_topk:
            lanes_by_mode["oracle_count_topk"] = lanes[: max(int(gt_count), 0)]

        converted_by_mode = {
            mode: gcs_lanes_to_tusimple_lanes(mode_lanes, record["h_samples"], image_shape=original_shape)
            for mode, mode_lanes in lanes_by_mode.items()
        }
        t2 = time.perf_counter()
        infer_time_s += t1 - t0
        post_time_s += t2 - t1
        run_time = (t2 - t0) * 1000.0 if use_measured_runtime else float(runtime_ms)
        for mode, tusimple_lanes in converted_by_mode.items():
            pred_records_by_mode[mode].append(
                {
                    "lanes": tusimple_lanes,
                    "h_samples": record["h_samples"],
                    "raw_file": raw_file,
                    "run_time": run_time,
                }
            )

    n = max(len(gt_records), 1)
    timing = {
        "avg_inference_ms": round(infer_time_s * 1000.0 / n, 4),
        "avg_postprocess_ms": round(post_time_s * 1000.0 / n, 4),
        "avg_total_ms": round((infer_time_s + post_time_s) * 1000.0 / n, 4),
    }
    count_head = _count_head_stats(count_head_rows, count_head_unsupported_reason) if need_count_head_global else None
    if count_guided_topk:
        guided_stats["supported"] = bool(count_head and count_head.get("supported"))
        if not guided_stats["supported"]:
            guided_stats["reason"] = count_head.get("reason") if count_head else count_head_unsupported_reason
    diagnostics = {
        "count_head": count_head,
        "count_guided": guided_stats if count_guided_topk else None,
    }
    return pred_records_by_mode, timing, diagnostics


def evaluate_official(args: argparse.Namespace) -> dict:
    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_path = Path(args.gt_json) if args.gt_json else default_tusimple_gt_json(archive_root, split=args.split)
    gt_records = _limit_records(read_tusimple_json_lines(gt_path), args.max_images)
    if not gt_records:
        raise ValueError(f"No TuSimple GT records found in {gt_path}")

    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    save_dir = resolve_save_dir(
        args.save_dir,
        args.weights,
        args.split,
        args.conf,
        args.point_valid_thr,
        args.nms_dist_px,
        args.max_det,
        args.min_points,
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    allowed_counts = _parse_allowed_counts(args.count_guided_allowed_counts)
    _validate_count_topk_candidate_pool(args)
    advanced_flags = bool(args.oracle_count_topk or args.count_guided_topk or args.dump_count_head_stats)
    if args.pred_json and advanced_flags:
        raise ValueError(
            "--pred-json cannot be combined with oracle/count-head/count-guided diagnostics because raw model outputs "
            "and lane scores are required."
        )

    timing = {"avg_inference_ms": None, "avg_postprocess_ms": None, "avg_total_ms": None}
    diagnostics = {"count_head": None, "count_guided": None}
    pred_json = args.pred_json
    pred_json_by_mode: dict[str, str] = {}
    if pred_json:
        pred_path = Path(pred_json)
        pred_records_by_mode = {"normal": _limit_records(read_tusimple_json_lines(pred_path), args.max_images)}
        pred_json_by_mode["normal"] = str(pred_path.resolve())
    else:
        warn_max_det_mismatch(args.weights, max_det=args.max_det, context="TuSimple official eval")
        pred_records_by_mode, timing, diagnostics = generate_predictions(
            weights=args.weights,
            archive_root=archive_root,
            split=args.split,
            gt_records=gt_records,
            imgsz=imgsz,
            conf=args.conf,
            point_valid_thr=args.point_valid_thr,
            nms_dist_px=args.nms_dist_px,
            max_det=args.max_det,
            min_points=args.min_points,
            runtime_ms=args.runtime_ms,
            use_measured_runtime=args.use_measured_runtime,
            warmup=args.warmup,
            device=args.device,
            half=args.half,
            oracle_count_topk=args.oracle_count_topk,
            count_guided_topk=args.count_guided_topk,
            count_guided_min_prob=args.count_guided_min_prob,
            count_guided_allowed_counts=allowed_counts,
            dump_count_head_stats=args.dump_count_head_stats,
        )
        for mode, records in pred_records_by_mode.items():
            filename = "tusimple_predictions.json" if mode == "normal" else f"tusimple_predictions_{mode}.json"
            mode_pred_path = save_dir / filename
            write_tusimple_predictions(mode_pred_path, records)
            pred_json_by_mode[mode] = str(mode_pred_path.resolve())

    mode_outputs: dict[str, dict] = {}
    records_by_mode: dict[str, list[dict]] = {}
    for mode, records in pred_records_by_mode.items():
        metrics, per_image = _official_metrics_for_records(
            records,
            gt_records,
            score_fp_weight=args.score_fp_weight,
            score_fn_weight=args.score_fn_weight,
            return_records=args.save_records,
        )
        mode_outputs[mode] = {
            "metrics": metrics,
            "gt4_focus": _gt4_focus(metrics.get("count_confusion", {})),
            "diagnostic_only": mode == "oracle_count_topk",
        }
        if args.save_records:
            records_by_mode[mode] = per_image

    metrics = mode_outputs["normal"]["metrics"]

    output = {
        "metrics": metrics,
        "gt4_focus": mode_outputs["normal"]["gt4_focus"],
        "modes": mode_outputs,
        "count_head": diagnostics.get("count_head"),
        "count_guided": diagnostics.get("count_guided"),
        "config": {
            "weights": None if pred_json else str(Path(args.weights).resolve()),
            "pred_json": pred_json_by_mode["normal"],
            "pred_json_by_mode": pred_json_by_mode,
            "archive_root": str(archive_root.resolve()),
            "split": args.split,
            "gt_json": str(gt_path.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "runtime_ms": float(args.runtime_ms),
            "use_measured_runtime": bool(args.use_measured_runtime),
            "max_images": int(args.max_images),
            "device": str(args.device),
            "half": bool(args.half),
            "save_dir": str(save_dir.resolve()),
            "score_fp_weight": float(args.score_fp_weight),
            "score_fn_weight": float(args.score_fn_weight),
            "oracle_count_topk": bool(args.oracle_count_topk),
            "oracle_count_topk_diagnostic_only": bool(args.oracle_count_topk),
            "dump_count_head_stats": bool(args.dump_count_head_stats),
            "count_guided_topk": bool(args.count_guided_topk),
            "count_guided_min_prob": float(args.count_guided_min_prob),
            "count_guided_allowed_counts": [int(x) for x in allowed_counts],
            "count_topk_candidate_pool": dict(COUNT_TOPK_CANDIDATE_POOL),
        },
        "timing": timing,
    }
    if args.save_records:
        output["records"] = records_by_mode.get("normal", [])
        output["records_by_mode"] = records_by_mode

    (save_dir / "tusimple_official_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (save_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_count_confusion_csv(save_dir / "count_confusion.csv", mode_outputs, diagnostics.get("count_head"))
    print(json.dumps({mode: item["metrics"] for mode, item in mode_outputs.items()}, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved to: {save_dir.resolve()}")
    return output


def main() -> None:
    evaluate_official(parse_args())


if __name__ == "__main__":
    main()
