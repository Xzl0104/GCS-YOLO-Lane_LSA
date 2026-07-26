from __future__ import annotations

import argparse
import csv
import itertools
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
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    official_gt_contract_summary,
    official_metric_score,
    read_tusimple_json_lines,
    resolve_tusimple_gt_json,
    tusimple_image_path,
    valid_tusimple_lanes,
    write_tusimple_predictions,
)
from tools.eval_tusimple_official import _count_diagnostics  # noqa: E402
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.models.gcs.decode_summary import load_decode_yaml, validate_decode_yaml_for_model  # noqa: E402
from ultralytics.models.gcs.mode_utils import resolve_decode_mode  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = ROOT / "runs/gcs_lane/query_count_head_ce05_v1/weights/official_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose query ranking with three TuSimple official-val controls: "
            "current ranking/current count, count_logits k/current ranking, and GT count/oracle ranking."
        )
    )
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="TuSimple archive split.")
    parser.add_argument("--gt-json", default=None, help="Official TuSimple GT json-lines file.")
    parser.add_argument(
        "--allow-noncanonical-gt",
        action="store_true",
        help="Allow non-363 split=val GT for diagnostics; summary marks it incomparable with E1/spurious.",
    )
    parser.add_argument(
        "--allow-test-oracle",
        action="store_true",
        help="Explicit escape hatch for test-only audits. Do not use test oracle output for selection or tuning.",
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Query GCS checkpoint .pt or YAML.")
    parser.add_argument("--decode-yaml", default=None, help="Optional query official_best_decode.yaml.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.005, help="Lane existence confidence threshold.")
    parser.add_argument("--point-valid-thr", type=float, default=0.5, help="Per-point visibility threshold.")
    parser.add_argument("--nms-dist-px", type=float, default=0.0, help="Lane-NMS distance in original-image pixels.")
    parser.add_argument("--max-det", type=int, default=5, help="Current decode max_det.")
    parser.add_argument("--min-points", type=int, default=2, help="Minimum visible anchors required to keep a lane.")
    parser.add_argument("--valid-before-maxdet", action="store_true", help="Apply point-valid/min_points before max_det.")
    parser.add_argument("--extent-decode", action="store_true", help="Use query start/end extent logits for current and pool decode.")
    parser.add_argument(
        "--extent-decode-mode",
        choices=("none", "interval", "intersect"),
        default="interval",
        help="Query extent decode mode when --extent-decode is enabled.",
    )
    parser.add_argument(
        "--pool-max-det",
        type=int,
        default=12,
        help="Candidate-pool max_det before count/rank choices. Use 0 for no explicit pool cap.",
    )
    parser.add_argument(
        "--current-count-aware-topk",
        action="store_true",
        help="Let the current baseline use existing count-aware top-k controls.",
    )
    parser.add_argument("--current-count-aware-min-k", type=int, default=3)
    parser.add_argument("--current-count-aware-max-k", type=int, default=5)
    parser.add_argument("--current-count-aware-length-norm", type=float, default=12.0)
    parser.add_argument(
        "--current-count-mode",
        choices=("score_sum", "count_logits"),
        default="score_sum",
        help="Count source only for the current baseline when --current-count-aware-topk is set.",
    )
    parser.add_argument(
        "--require-count-logits",
        action="store_true",
        help="Fail if the checkpoint does not emit query pred_count_logits.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of GT records. 0 means all.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of untimed warmup forwards.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--runtime-ms", type=float, default=1.0, help="Constant TuSimple run_time in ms.")
    parser.add_argument("--save-dir", default=None, help="Output directory.")
    parser.add_argument("--save-records", action="store_true", help="Store per-image official records in summary.")
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
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _limit_records(records: list[dict], max_images: int) -> list[dict]:
    if max_images and max_images > 0:
        return records[: int(max_images)]
    return records


def _weight_run_dir(weights: str | Path) -> Path | None:
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir is not None and str(args.save_dir).strip():
        return Path(args.save_dir)
    run_dir = _weight_run_dir(args.weights)
    tag = f"oracle_rank_{args.split}"
    if run_dir is not None:
        return run_dir / tag
    return ROOT / "runs" / "gcs_lane" / "oracle_rank" / Path(args.weights).stem / tag


def _apply_query_decode_yaml(args: argparse.Namespace, decode_yaml_cfg: dict) -> None:
    args.conf = float(decode_yaml_cfg["conf"])
    args.point_valid_thr = float(decode_yaml_cfg["point_valid_thr"])
    args.nms_dist_px = float(decode_yaml_cfg["nms_dist_px"])
    args.max_det = int(decode_yaml_cfg["max_det"])
    args.min_points = int(decode_yaml_cfg["min_points"])
    args.valid_before_maxdet = bool(decode_yaml_cfg.get("valid_before_maxdet", False))
    args.extent_decode = bool(decode_yaml_cfg.get("extent_decode", False))
    args.extent_decode_mode = str(decode_yaml_cfg.get("extent_decode_mode", "none") or "none")
    args.current_count_aware_topk = bool(decode_yaml_cfg.get("count_aware_topk", False))
    args.current_count_aware_min_k = int(decode_yaml_cfg.get("count_aware_min_k", 3))
    args.current_count_aware_max_k = int(decode_yaml_cfg.get("count_aware_max_k", 5))
    args.current_count_aware_length_norm = float(decode_yaml_cfg.get("count_aware_length_norm", 12.0))
    args.current_count_mode = str(decode_yaml_cfg.get("count_mode", "score_sum"))


def _count_logits_k(pred_count_logits: torch.Tensor | None) -> int | None:
    if pred_count_logits is None:
        return None
    count_logits = pred_count_logits.detach().float().cpu().reshape(-1)
    if count_logits.numel() != 4:
        raise ValueError(f"query pred_count_logits must have 4 classes for 2..5 lanes, got {count_logits.numel()}.")
    return int(count_logits.argmax().item()) + 2


def _official_image_metrics(pred_lanes: list[list[int]], gt_record: dict, runtime_ms: float) -> dict:
    acc, fp, fn = TuSimpleOfficialLaneEval.bench(
        pred=pred_lanes,
        gt=[list(x) for x in gt_record["lanes"]],
        y_samples=list(gt_record["h_samples"]),
        running_time=runtime_ms,
    )
    return {
        "Accuracy": round(float(acc), 6),
        "FP": round(float(fp), 6),
        "FN": round(float(fn), 6),
    }


def _oracle_rank_indices(
    candidate_lanes: list[list[int]],
    gt_record: dict,
    k: int,
    runtime_ms: float,
) -> tuple[list[int], dict]:
    k = int(k)
    if k <= 0 or not candidate_lanes:
        metrics = _official_image_metrics([], gt_record, runtime_ms=runtime_ms)
        return [], metrics
    if len(candidate_lanes) <= k:
        indices = list(range(len(candidate_lanes)))
        metrics = _official_image_metrics([candidate_lanes[i] for i in indices], gt_record, runtime_ms=runtime_ms)
        return indices, metrics

    best_indices: tuple[int, ...] | None = None
    best_metrics: dict | None = None
    best_key: tuple[float, float, float, int, int] | None = None
    for combo in itertools.combinations(range(len(candidate_lanes)), k):
        metrics = _official_image_metrics([candidate_lanes[i] for i in combo], gt_record, runtime_ms=runtime_ms)
        # Optimize official per-image Accuracy. Ties prefer lower FP/FN, then the current-rank earlier lanes.
        key = (
            float(metrics["Accuracy"]),
            -float(metrics["FP"]),
            -float(metrics["FN"]),
            -int(sum(combo)),
            -int(max(combo)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_indices = combo
            best_metrics = metrics
    assert best_indices is not None and best_metrics is not None
    return list(best_indices), best_metrics


def _make_pred_record(raw_file: str, h_samples: list[int] | list[float], lanes: list[list[int]], runtime_ms: float) -> dict:
    return {"lanes": lanes, "h_samples": h_samples, "raw_file": raw_file, "run_time": float(runtime_ms)}


def _metric_block(
    pred_records: list[dict],
    gt_records: list[dict],
    *,
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


def _write_count_confusion(path: Path, modes: dict[str, dict]) -> None:
    rows = []
    for mode, block in modes.items():
        confusion = block.get("metrics", {}).get("count_confusion", {})
        for key, value in sorted(confusion.items()):
            gt_count, pred_count = key.split("->", 1)
            rows.append(
                {
                    "mode": mode,
                    "gt_count": int(gt_count),
                    "pred_count": int(pred_count),
                    "images": int(value),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "gt_count", "pred_count", "images"])
        writer.writeheader()
        writer.writerows(rows)


def _write_per_image_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "raw_file",
        "gt_count",
        "candidate_count",
        "current_count",
        "count_logits_k",
        "count_logits_supported",
        "oracle_k",
        "current_acc",
        "count_logits_current_rank_acc",
        "gt_count_oracle_rank_acc",
        "oracle_minus_current_acc",
        "oracle_minus_count_logits_acc",
        "current_fp",
        "count_logits_current_rank_fp",
        "gt_count_oracle_rank_fp",
        "current_fn",
        "count_logits_current_rank_fn",
        "gt_count_oracle_rank_fn",
        "oracle_indices",
        "oracle_queries",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if str(args.split).lower() == "test" and not bool(args.allow_test_oracle):
        raise ValueError(
            "oracle-rank uses GT during lane selection and is blocked on --split test by default. "
            "Run it on official-val for model decisions, or pass --allow-test-oracle only for a predeclared audit."
        )
    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_path = resolve_tusimple_gt_json(archive_root, split=args.split, gt_json=args.gt_json)
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
    decode_yaml_cfg = None
    if args.decode_yaml:
        _, decode_yaml_cfg = load_decode_yaml(args.decode_yaml)
        decode_mode = resolve_decode_mode(decode_yaml_cfg.get("decode_mode"), model)
        validate_decode_yaml_for_model(decode_yaml_cfg, model_mode=decode_mode)
        if decode_mode != "query":
            raise ValueError(f"oracle-rank diagnostic supports query decode only, got decode_mode={decode_mode!r}.")
        _apply_query_decode_yaml(args, decode_yaml_cfg)
    else:
        decode_mode = resolve_decode_mode("auto", model)
        if decode_mode != "query":
            raise ValueError(f"oracle-rank diagnostic supports query decode only, got decode_mode={decode_mode!r}.")

    warn_max_det_mismatch(args.weights, max_det=args.max_det, context="TuSimple oracle-rank diagnostic")
    if args.pool_max_det > 0:
        warn_max_det_mismatch(args.weights, max_det=args.pool_max_det, context="TuSimple oracle-rank candidate pool")

    if args.warmup > 0 and gt_records:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device_obj, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    pred_by_mode: dict[str, list[dict]] = {
        "current_ranking_current_count": [],
        "gt_count_oracle_rank": [],
    }
    count_logits_pred_records: list[dict] = []
    per_image_rows: list[dict] = []
    count_logits_pairs: list[tuple[int, int]] = []
    pool_short_gt_images = 0
    count_logits_supported_images = 0
    count_logits_missing_images = 0
    infer_time_s = 0.0
    post_time_s = 0.0

    save_dir = resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

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

        pred_valid = preds.get("pred_valid_logits")
        pred_count_logits = preds.get("pred_count_logits")
        pred_count_logits_0 = pred_count_logits[0] if pred_count_logits is not None else None
        pred_quality_logits = preds.get("pred_quality_logits")
        pred_quality_logits_0 = pred_quality_logits[0] if pred_quality_logits is not None else None
        pred_start_logits = preds.get("pred_start_logits")
        pred_start_logits_0 = pred_start_logits[0] if pred_start_logits is not None else None
        pred_end_logits = preds.get("pred_end_logits")
        pred_end_logits_0 = pred_end_logits[0] if pred_end_logits is not None else None
        count_logits_k = _count_logits_k(pred_count_logits_0)
        gt_count = len(valid_tusimple_lanes(record.get("lanes", [])))

        current_lanes = decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_quality_logits=pred_quality_logits_0,
            pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
            pred_count_logits=pred_count_logits_0,
            pred_start_logits=pred_start_logits_0,
            pred_end_logits=pred_end_logits_0,
            image_shape=original_shape,
            score_thr=args.conf,
            point_valid_thr=args.point_valid_thr,
            min_points=args.min_points,
            max_det=args.max_det,
            nms_dist_px=args.nms_dist_px,
            valid_before_maxdet=args.valid_before_maxdet,
            extent_decode=args.extent_decode,
            extent_decode_mode=args.extent_decode_mode,
            count_aware_topk=args.current_count_aware_topk,
            count_aware_min_k=args.current_count_aware_min_k,
            count_aware_max_k=args.current_count_aware_max_k,
            count_aware_length_norm=args.current_count_aware_length_norm,
            count_mode=args.current_count_mode,
        )
        pool_lanes = decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_quality_logits=pred_quality_logits_0,
            pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
            pred_count_logits=pred_count_logits_0,
            pred_start_logits=pred_start_logits_0,
            pred_end_logits=pred_end_logits_0,
            image_shape=original_shape,
            score_thr=args.conf,
            point_valid_thr=args.point_valid_thr,
            min_points=args.min_points,
            max_det=None if int(args.pool_max_det) <= 0 else int(args.pool_max_det),
            nms_dist_px=args.nms_dist_px,
            valid_before_maxdet=True,
            extent_decode=args.extent_decode,
            extent_decode_mode=args.extent_decode_mode,
            count_aware_topk=False,
        )
        current_tusimple = gcs_lanes_to_tusimple_lanes(current_lanes, record["h_samples"], image_shape=original_shape)
        # Keep lane/query metadata aligned after TuSimple export drops lanes that no longer contain two valid h-samples.
        pool_pairs: list[tuple[dict, list[int]]] = []
        for lane in pool_lanes:
            converted = gcs_lanes_to_tusimple_lanes([lane], record["h_samples"], image_shape=original_shape)
            if converted:
                pool_pairs.append((lane, converted[0]))
        pool_lanes = [x[0] for x in pool_pairs]
        pool_tusimple = [x[1] for x in pool_pairs]
        if len(pool_tusimple) < gt_count:
            pool_short_gt_images += 1

        pred_by_mode["current_ranking_current_count"].append(
            _make_pred_record(raw_file, record["h_samples"], current_tusimple, args.runtime_ms)
        )

        count_logits_metrics = {"Accuracy": None, "FP": None, "FN": None}
        if count_logits_k is None:
            count_logits_missing_images += 1
            count_logits_tusimple: list[list[int]] = []
        else:
            count_logits_supported_images += 1
            count_logits_pairs.append((gt_count, int(count_logits_k)))
            count_logits_tusimple = pool_tusimple[: max(int(count_logits_k), 0)]
            count_logits_metrics = _official_image_metrics(count_logits_tusimple, record, runtime_ms=args.runtime_ms)
        count_logits_pred_records.append(_make_pred_record(raw_file, record["h_samples"], count_logits_tusimple, args.runtime_ms))

        oracle_indices, oracle_metrics = _oracle_rank_indices(
            pool_tusimple,
            record,
            k=gt_count,
            runtime_ms=args.runtime_ms,
        )
        oracle_tusimple = [pool_tusimple[i] for i in oracle_indices]
        pred_by_mode["gt_count_oracle_rank"].append(
            _make_pred_record(raw_file, record["h_samples"], oracle_tusimple, args.runtime_ms)
        )

        current_metrics = _official_image_metrics(current_tusimple, record, runtime_ms=args.runtime_ms)
        count_logits_acc = count_logits_metrics["Accuracy"]
        per_image_rows.append(
            {
                "raw_file": raw_file,
                "gt_count": int(gt_count),
                "candidate_count": int(len(pool_tusimple)),
                "current_count": int(len(current_tusimple)),
                "count_logits_k": "" if count_logits_k is None else int(count_logits_k),
                "count_logits_supported": count_logits_k is not None,
                "oracle_k": int(gt_count),
                "current_acc": current_metrics["Accuracy"],
                "count_logits_current_rank_acc": "" if count_logits_acc is None else count_logits_acc,
                "gt_count_oracle_rank_acc": oracle_metrics["Accuracy"],
                "oracle_minus_current_acc": round(float(oracle_metrics["Accuracy"]) - float(current_metrics["Accuracy"]), 6),
                "oracle_minus_count_logits_acc": ""
                if count_logits_acc is None
                else round(float(oracle_metrics["Accuracy"]) - float(count_logits_acc), 6),
                "current_fp": current_metrics["FP"],
                "count_logits_current_rank_fp": count_logits_metrics["FP"] if count_logits_k is not None else "",
                "gt_count_oracle_rank_fp": oracle_metrics["FP"],
                "current_fn": current_metrics["FN"],
                "count_logits_current_rank_fn": count_logits_metrics["FN"] if count_logits_k is not None else "",
                "gt_count_oracle_rank_fn": oracle_metrics["FN"],
                "oracle_indices": ";".join(str(int(i)) for i in oracle_indices),
                "oracle_queries": ";".join(str(int(pool_lanes[i].get("query", -1))) for i in oracle_indices),
            }
        )

        post_time_s += time.perf_counter() - t1
        infer_time_s += t1 - t0

    if args.require_count_logits and count_logits_missing_images:
        raise RuntimeError(
            "Checkpoint did not emit pred_count_logits for all images: "
            f"missing={count_logits_missing_images}, supported={count_logits_supported_images}."
        )
    if count_logits_supported_images:
        pred_by_mode["count_logits_k_current_rank"] = count_logits_pred_records

    modes: dict[str, dict] = {}
    records_by_mode: dict[str, list[dict]] = {}
    mode_order = [
        "current_ranking_current_count",
        "count_logits_k_current_rank",
        "gt_count_oracle_rank",
    ]
    for mode in mode_order:
        if mode not in pred_by_mode:
            continue
        pred_records = pred_by_mode[mode]
        metrics, records = _metric_block(
            pred_records,
            gt_records,
            score_fp_weight=args.score_fp_weight,
            score_fn_weight=args.score_fn_weight,
            return_records=args.save_records,
        )
        pred_path = save_dir / f"tusimple_predictions_{mode}.json"
        write_tusimple_predictions(pred_path, pred_records)
        modes[mode] = {
            "metrics": metrics,
            "pred_json": str(pred_path.resolve()),
            "diagnostic_only": mode != "current_ranking_current_count",
        }
        records_by_mode[mode] = records

    deltas = {}
    current_metrics = modes["current_ranking_current_count"]["metrics"]
    oracle_metrics = modes["gt_count_oracle_rank"]["metrics"]
    deltas["gt_count_oracle_rank_minus_current"] = {
        "official_acc": round(float(oracle_metrics["official_acc"]) - float(current_metrics["official_acc"]), 6),
        "official_FP": round(float(oracle_metrics["official_FP"]) - float(current_metrics["official_FP"]), 6),
        "official_FN": round(float(oracle_metrics["official_FN"]) - float(current_metrics["official_FN"]), 6),
        "official_score": round(float(oracle_metrics["official_score"]) - float(current_metrics["official_score"]), 6),
    }
    if "count_logits_k_current_rank" in modes:
        logits_metrics = modes["count_logits_k_current_rank"]["metrics"]
        deltas["gt_count_oracle_rank_minus_count_logits_current_rank"] = {
            "official_acc": round(float(oracle_metrics["official_acc"]) - float(logits_metrics["official_acc"]), 6),
            "official_FP": round(float(oracle_metrics["official_FP"]) - float(logits_metrics["official_FP"]), 6),
            "official_FN": round(float(oracle_metrics["official_FN"]) - float(logits_metrics["official_FN"]), 6),
            "official_score": round(float(oracle_metrics["official_score"]) - float(logits_metrics["official_score"]), 6),
        }

    count_logits_summary = {
        "supported_images": int(count_logits_supported_images),
        "missing_images": int(count_logits_missing_images),
        "required": bool(args.require_count_logits),
    }
    if count_logits_pairs:
        n = max(len(count_logits_pairs), 1)
        count_logits_summary.update(
            {
                "count_acc": round(
                    sum(1 for gt, pred in count_logits_pairs if int(gt) == int(pred)) / n,
                    6,
                ),
                "count_confusion": {
                    f"{gt}->{pred}": int(v)
                    for (gt, pred), v in sorted(Counter((int(gt), int(pred)) for gt, pred in count_logits_pairs).items())
                },
            }
        )

    n = max(len(gt_records), 1)
    config = {
        "weights": str(Path(args.weights).resolve()),
        "decode_yaml": None if not args.decode_yaml else str(Path(args.decode_yaml).resolve()),
        "archive_root": str(archive_root.resolve()),
        "split": args.split,
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
        "extent_decode": bool(args.extent_decode),
        "extent_decode_mode": str(args.extent_decode_mode),
        "pool_max_det": None if int(args.pool_max_det) <= 0 else int(args.pool_max_det),
        "candidate_pool": "query decode after conf, Lane-NMS, point-valid/min_points filtering, before count/rank choice",
        "current_count_aware_topk": bool(args.current_count_aware_topk),
        "current_count_mode": str(args.current_count_mode),
        "runtime_ms": float(args.runtime_ms),
        "max_images": int(args.max_images),
        "device": str(args.device),
        "half": bool(args.half),
        "score_fp_weight": float(args.score_fp_weight),
        "score_fn_weight": float(args.score_fn_weight),
        **gt_contract,
    }
    output = {
        "modes": modes,
        "deltas": deltas,
        "count_logits": count_logits_summary,
        "candidate_pool": {
            "pool_short_gt_images": int(pool_short_gt_images),
            "pool_max_det": None if int(args.pool_max_det) <= 0 else int(args.pool_max_det),
        },
        "config": config,
        **gt_contract,
        "timing": {
            "avg_inference_ms": round(infer_time_s * 1000.0 / n, 4),
            "avg_postprocess_ms": round(post_time_s * 1000.0 / n, 4),
            "avg_total_ms": round((infer_time_s + post_time_s) * 1000.0 / n, 4),
        },
    }
    if args.save_records:
        output["records"] = records_by_mode

    (save_dir / "oracle_rank_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_count_confusion(save_dir / "count_confusion.csv", modes)
    _write_per_image_csv(save_dir / "oracle_rank_per_image.csv", per_image_rows)

    compact = {
        "modes": {mode: block["metrics"] for mode, block in modes.items()},
        "deltas": deltas,
        "count_logits": count_logits_summary,
        "save_dir": str(save_dir.resolve()),
    }
    print(json.dumps(compact, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
