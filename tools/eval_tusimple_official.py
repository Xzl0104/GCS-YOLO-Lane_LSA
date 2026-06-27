from __future__ import annotations

import argparse
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
    parser.add_argument("--count-aware-topk", action="store_true", help="Use count_score to keep only the quality-best dynamic lane count.")
    parser.add_argument("--count-aware-min-k", type=int, default=3, help="Minimum k_hat for --count-aware-topk.")
    parser.add_argument("--count-aware-max-k", type=int, default=5, help="Maximum k_hat for --count-aware-topk.")
    parser.add_argument("--count-aware-length-norm", type=float, default=12.0, help="Visible-point count that saturates count-aware length quality.")
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
    count_aware_topk: bool = False,
) -> Path:
    if save_dir is not None and str(save_dir).strip():
        return Path(save_dir)
    count_tag = "_catopk" if count_aware_topk else ""
    tag = (
        f"official_{split}_conf{float(conf):.4g}_pvalid{float(point_valid_thr):.4g}_"
        f"nms{float(nms_dist_px):.4g}_maxdet{int(max_det)}_minp{int(min_points)}{count_tag}"
    ).replace(".", "p")
    run_dir = _weight_run_dir(weights)
    if run_dir is not None:
        return run_dir / tag
    return ROOT / "runs" / "gcs_lane" / "tusimple_official_eval" / Path(weights).stem / tag


def _limit_records(records: list[dict], max_images: int) -> list[dict]:
    if max_images and max_images > 0:
        return records[: int(max_images)]
    return records


def _count_diagnostics(pred_records: list[dict], gt_records: list[dict]) -> dict:
    gt_by_raw = {str(x["raw_file"]): x for x in gt_records}
    pairs: list[tuple[int, int]] = []
    for pred in pred_records:
        gt = gt_by_raw[str(pred["raw_file"])]
        gt_count = sum(1 for lane in gt.get("lanes", []) if any(float(x) >= 0.0 for x in lane))
        pairs.append((gt_count, len(pred.get("lanes", []))))
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
    count_aware_topk: bool = False,
    count_aware_min_k: int = 3,
    count_aware_max_k: int = 5,
    count_aware_length_norm: float = 12.0,
) -> tuple[list[dict], dict]:
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

    pred_records: list[dict] = []
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
            count_aware_topk=count_aware_topk,
            count_aware_min_k=count_aware_min_k,
            count_aware_max_k=count_aware_max_k,
            count_aware_length_norm=count_aware_length_norm,
        )
        tusimple_lanes = gcs_lanes_to_tusimple_lanes(lanes, record["h_samples"], image_shape=original_shape)
        t2 = time.perf_counter()
        infer_time_s += t1 - t0
        post_time_s += t2 - t1
        run_time = (t2 - t0) * 1000.0 if use_measured_runtime else float(runtime_ms)
        pred_records.append({"lanes": tusimple_lanes, "h_samples": record["h_samples"], "raw_file": raw_file, "run_time": run_time})

    n = max(len(gt_records), 1)
    timing = {
        "avg_inference_ms": round(infer_time_s * 1000.0 / n, 4),
        "avg_postprocess_ms": round(post_time_s * 1000.0 / n, 4),
        "avg_total_ms": round((infer_time_s + post_time_s) * 1000.0 / n, 4),
    }
    return pred_records, timing


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
        count_aware_topk=args.count_aware_topk,
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    timing = {"avg_inference_ms": None, "avg_postprocess_ms": None, "avg_total_ms": None}
    pred_json = args.pred_json
    if pred_json:
        pred_path = Path(pred_json)
        pred_records = _limit_records(read_tusimple_json_lines(pred_path), args.max_images)
    else:
        warn_max_det_mismatch(args.weights, max_det=args.max_det, context="TuSimple official eval")
        pred_records, timing = generate_predictions(
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
            count_aware_topk=args.count_aware_topk,
            count_aware_min_k=args.count_aware_min_k,
            count_aware_max_k=args.count_aware_max_k,
            count_aware_length_norm=args.count_aware_length_norm,
            runtime_ms=args.runtime_ms,
            use_measured_runtime=args.use_measured_runtime,
            warmup=args.warmup,
            device=args.device,
            half=args.half,
        )
        pred_path = save_dir / "tusimple_predictions.json"
        write_tusimple_predictions(pred_path, pred_records)

    result, per_image = TuSimpleOfficialLaneEval.bench_records(
        pred_records,
        gt_records,
        strict_length=True,
        return_records=args.save_records,
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
            fp_weight=args.score_fp_weight,
            fn_weight=args.score_fn_weight,
        ),
        6,
    )
    metrics.update(_count_diagnostics(pred_records, gt_records))

    output = {
        "metrics": metrics,
        "config": {
            "weights": None if pred_json else str(Path(args.weights).resolve()),
            "pred_json": str(Path(pred_json).resolve()) if pred_json else str((save_dir / "tusimple_predictions.json").resolve()),
            "archive_root": str(archive_root.resolve()),
            "split": args.split,
            "gt_json": str(gt_path.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "count_aware_topk": bool(args.count_aware_topk),
            "count_aware_min_k": int(args.count_aware_min_k),
            "count_aware_max_k": int(args.count_aware_max_k),
            "count_aware_length_norm": float(args.count_aware_length_norm),
            "runtime_ms": float(args.runtime_ms),
            "use_measured_runtime": bool(args.use_measured_runtime),
            "max_images": int(args.max_images),
            "device": str(args.device),
            "half": bool(args.half),
            "save_dir": str(save_dir.resolve()),
            "score_fp_weight": float(args.score_fp_weight),
            "score_fn_weight": float(args.score_fn_weight),
        },
        "timing": timing,
    }
    if args.save_records:
        output["records"] = per_image

    (save_dir / "tusimple_official_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved to: {save_dir.resolve()}")
    return output


def main() -> None:
    evaluate_official(parse_args())


if __name__ == "__main__":
    main()
