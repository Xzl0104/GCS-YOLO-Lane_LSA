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
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.eval_gcs import label_path_for_image, load_gcs_label, match_lanes
from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.gcs.mode_utils import infer_gcs_mode_from_model
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, assert_gcs_shape, normalize_imgsz, shape_str
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions
from ultralytics.utils.torch_utils import select_device


DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_refquery_e220" / "weights" / "best.pt"


def assert_legacy_query_conf_sweep_model(model: torch.nn.Module) -> str:
    """Reject ordered-slot models before this legacy query-threshold sweep runs."""
    model_mode = infer_gcs_mode_from_model(model)
    if model_mode == "ordered_slot":
        raise RuntimeError(
            "tools/sweep_gcs_conf.py is a legacy query-threshold sweep tool. "
            "It must not be used with ordered_slot checkpoints because ordered_slot does not use "
            "conf/NMS/topk decoding. Use tools/sweep_tusimple_official_cached.py or "
            "tools/eval_tusimple_official.py with --decode-mode auto instead."
        )
    return model_mode


def assert_legacy_query_conf_sweep_split(split: str) -> str:
    """Reject test-set threshold search in this legacy query sweep tool."""
    normalized = str(split or "").strip().lower()
    if normalized == "test":
        raise RuntimeError(
            "tools/sweep_gcs_conf.py is a threshold search tool and must only sweep validation data. "
            "Use --split val for threshold selection; test may only be evaluated once with a fixed decode config."
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep GCS lane existence confidence on validation data.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--data", default=None, help="GCS data yaml. Defaults to local fixed-y TuSimple yaml when present.")
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="Dataset split to sweep when --source is not provided.",
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt.")
    parser.add_argument("--source", default=None, help="Validation image directory/list. Defaults to data yaml val.")
    parser.add_argument("--labels", default=None, help="Validation labels_gcs directory. Empty means infer from images.")
    parser.add_argument("--test-source", default=None, help="Optional test image directory/list for --run-test.")
    parser.add_argument("--test-labels", default=None, help="Optional test labels_gcs directory for --run-test.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS inference shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument(
        "--conf-start",
        type=float,
        default=0.10,
        help="First confidence threshold in the sweep.",
    )
    parser.add_argument("--conf-end", type=float, default=0.80, help="Last confidence threshold in the sweep.")
    parser.add_argument("--conf-step", type=float, default=0.05, help="Confidence threshold step.")
    parser.add_argument(
        "--confs",
        nargs="+",
        type=float,
        default=None,
        help="Explicit confidence thresholds. Overrides --conf-start/--conf-end/--conf-step.",
    )
    parser.add_argument("--ape-thr", type=float, default=20.0, help="APE threshold in pixels for TP matching.")
    parser.add_argument("--match-gate-px", type=float, default=None, help="Strict eval APE gate in pixels. Defaults to --ape-thr.")
    parser.add_argument("--max-x-dist", type=float, default=0.0, help="Optional strict eval mean x-distance gate in pixels. 0 disables.")
    parser.add_argument("--min-overlap", type=int, default=2, help="Minimum valid overlapping GT points required for eval matching.")
    parser.add_argument("--nms-dist-px", type=float, default=0.0, help="Optional lane duplicate suppression distance in pixels. 0 disables.")
    parser.add_argument(
        "--nms-dist-pxs",
        nargs="+",
        type=float,
        default=None,
        help="Explicit Lane-NMS thresholds in pixels. Overrides --nms-dist-px and sweeps conf x NMS.",
    )
    parser.add_argument(
        "--point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold for fixed-y lane decoding.",
    )
    parser.add_argument(
        "--point-valid-thrs",
        nargs="+",
        type=float,
        default=None,
        help="Explicit per-point visibility thresholds. Overrides --point-valid-thr and sweeps conf x NMS x point-valid.",
    )
    parser.add_argument("--max-det", type=int, default=8, help="Maximum decoded lane queries per image.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit validation images. 0 means all.")
    parser.add_argument("--test-max-images", type=int, default=0, help="Limit test images for --run-test. 0 means all.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of untimed warmup forwards.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--save-dir", default="runs/gcs_lane/conf_sweep", help="Output directory.")
    parser.add_argument(
        "--select-by",
        default="f1",
        choices=("f1", "fitness"),
        help="Criterion for best conf. fitness = f1 - 0.001*lane_count_mae - 0.0001*ape_mean_px.",
    )
    parser.add_argument("--run-test", action="store_true", help="Also evaluate test split using the best val conf.")
    parser.add_argument("--save-records", action="store_true", help="Save per-image records for each threshold.")
    return parser.parse_args()


def default_data_yaml(dataset: str) -> Path:
    """Prefer the fixed-y TuSimple yaml used by current experiments when it exists."""
    candidates = [
        ROOT / "data" / f"{dataset}_gcs_fixed_y_k56_960x544.yaml",
        ROOT / "data" / f"{dataset}_gcs_stratified_960x544.yaml",
        ROOT / "data" / f"{dataset}_gcs.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def resolve_dataset(args: argparse.Namespace) -> dict:
    """Resolve image split paths from a GCS data yaml."""
    data_path = Path(args.data) if args.data else default_data_yaml(args.dataset)
    if not data_path.exists():
        raise FileNotFoundError(f"GCS data yaml not found: {data_path}")
    data = check_det_dataset(str(data_path))
    data["yaml_file"] = str(data_path)
    return data


def labels_from_source(source: str | Path | None) -> Path | None:
    """Map an images/<split> directory to labels_gcs/<split> when possible."""
    if source is None:
        return None
    path = Path(source)
    parts = list(path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels_gcs"
        return Path(*parts)
    return None


def conf_values(args: argparse.Namespace) -> list[float]:
    """Build the confidence grid with stable decimal formatting."""
    if args.confs:
        values = args.confs
    else:
        n = int(round((args.conf_end - args.conf_start) / args.conf_step))
        values = [args.conf_start + i * args.conf_step for i in range(n + 1)]
    values = sorted({round(float(x), 6) for x in values})
    if not values:
        raise ValueError("No confidence thresholds to evaluate.")
    return values


def nms_values(args: argparse.Namespace) -> list[float]:
    """Build the Lane-NMS threshold grid."""
    values = args.nms_dist_pxs if args.nms_dist_pxs is not None else [args.nms_dist_px]
    values = sorted({round(float(x), 6) for x in values})
    if not values:
        raise ValueError("No Lane-NMS thresholds to evaluate.")
    if any(x < 0.0 for x in values):
        raise ValueError(f"Lane-NMS thresholds must be >= 0, got {values}.")
    return values


def point_valid_values(args: argparse.Namespace) -> list[float]:
    """Build the fixed-y point visibility threshold grid."""
    values = args.point_valid_thrs if args.point_valid_thrs is not None else [args.point_valid_thr]
    values = sorted({round(float(x), 6) for x in values})
    if not values:
        raise ValueError("No point-valid thresholds to evaluate.")
    if any(x < 0.0 or x > 1.0 for x in values):
        raise ValueError(f"point-valid thresholds must be in [0, 1], got {values}.")
    return values


def empty_state(conf: float, nms_dist_px: float = 0.0, point_valid_thr: float = 0.5) -> dict:
    """Create mutable metric state for one confidence threshold."""
    return {
        "conf": float(conf),
        "nms_dist_px": float(nms_dist_px),
        "point_valid_thr": float(point_valid_thr),
        "images": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "apes": [],
        "apes_matched_all": [],
        "apes_fp_matched": [],
        "curvature_error": [],
        "lane_count_abs_error": 0,
        "pred_lanes": [],
        "gt_lanes": [],
        "records": [],
    }


def update_state(
    state: dict,
    image_path: Path,
    gt_lanes: np.ndarray,
    gt_valid: np.ndarray,
    pred_lanes: list[dict],
    image_shape: tuple[int, int],
    ape_thr: float,
    match_gate_px: float | None,
    max_x_dist: float,
    min_overlap: int,
    save_records: bool,
) -> None:
    """Match one image and update one threshold's metric state."""
    metrics, matches = match_lanes(
        pred_lanes,
        gt_lanes,
        gt_valid,
        image_shape,
        ape_thr=ape_thr,
        match_gate_px=match_gate_px,
        max_x_dist=max_x_dist,
        min_overlap=min_overlap,
    )
    pred_count = len(pred_lanes)
    gt_count = int(gt_lanes.shape[0])
    state["images"] += 1
    state["tp"] += int(metrics["tp"])
    state["fp"] += int(metrics["fp"])
    state["fn"] += int(metrics["fn"])
    state["apes"].extend(float(x) for x in metrics.get("ape_tp", metrics.get("ape", [])))
    state["apes_matched_all"].extend(float(x) for x in metrics.get("ape_matched_all", []))
    state["apes_fp_matched"].extend(float(x) for x in metrics.get("ape_fp_matched", []))
    state["curvature_error"].extend(float(x) for x in metrics["curvature_error"])
    state["lane_count_abs_error"] += abs(pred_count - gt_count)
    state["pred_lanes"].append(pred_count)
    state["gt_lanes"].append(gt_count)
    if save_records:
        state["records"].append(
            {
                "image": str(image_path.resolve()),
                "pred_lanes": pred_count,
                "gt_lanes": gt_count,
                "metrics": metrics,
                "matches": matches,
            }
        )


def summarize_state(state: dict, ape_thr: float) -> dict:
    """Convert accumulated sweep state into JSON/CSV-friendly metrics."""
    tp, fp, fn = int(state["tp"]), int(state["fp"]), int(state["fn"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    images = max(int(state["images"]), 1)
    apes = state["apes"]
    apes_all = state["apes_matched_all"]
    apes_fp = state["apes_fp_matched"]
    curve = state["curvature_error"]
    pred_counts = [int(x) for x in state["pred_lanes"]]
    gt_counts = [int(x) for x in state["gt_lanes"]]
    ape_mean = None if not apes else round(float(np.mean(apes)), 4)
    lane_count_mae = round(float(state["lane_count_abs_error"]) / images, 6)
    fitness = f1 - 0.001 * lane_count_mae - 0.0001 * (0.0 if ape_mean is None else ape_mean)
    return {
        "conf": round(float(state["conf"]), 6),
        "nms_dist_px": round(float(state.get("nms_dist_px", 0.0)), 6),
        "point_valid_thr": round(float(state.get("point_valid_thr", 0.5)), 6),
        "images": int(state["images"]),
        "ape_threshold_px": float(ape_thr),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "fitness": round(float(fitness), 6),
        "lane_count_mae": lane_count_mae,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ape_mean_px": ape_mean,
        "ape_median_px": None if not apes else round(float(np.median(apes)), 4),
        "ape_tp_mean_px": ape_mean,
        "ape_matched_all_mean_px": None if not apes_all else round(float(np.mean(apes_all)), 4),
        "ape_fp_matched_mean_px": None if not apes_fp else round(float(np.mean(apes_fp)), 4),
        "ape_all_matched_mean_px": None if not apes_all else round(float(np.mean(apes_all)), 4),
        "fp_matched_ape_mean_px": None if not apes_fp else round(float(np.mean(apes_fp)), 4),
        "curvature_error_mean_px": None if not curve else round(float(np.mean(curve)), 4),
        "pred_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(pred_counts).items())},
        "gt_lanes_hist": {str(k): int(v) for k, v in sorted(Counter(gt_counts).items())},
        "gt_pred_lanes_hist": {
            f"{gt}->{pred}": int(count)
            for (gt, pred), count in sorted(Counter(zip(gt_counts, pred_counts)).items())
        },
    }


def select_best(rows: list[dict], select_by: str) -> dict:
    """Pick the best validation threshold with deterministic tie-breakers."""
    if select_by == "fitness":
        return max(
            rows,
            key=lambda x: (
                float(x["fitness"]),
                float(x["f1"]),
                -float(x["lane_count_mae"]),
                float(x["conf"]),
                -float(x.get("nms_dist_px", 0.0)),
                float(x.get("point_valid_thr", 0.5)),
            ),
        )
    return max(
        rows,
        key=lambda x: (
            float(x["f1"]),
            -float(x["lane_count_mae"]),
            float(x["precision"]),
            float(x["conf"]),
            -float(x.get("nms_dist_px", 0.0)),
            float(x.get("point_valid_thr", 0.5)),
        ),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write compact sweep metrics to CSV."""
    fields = [
        "point_valid_thr",
        "nms_dist_px",
        "conf",
        "images",
        "precision",
        "recall",
        "f1",
        "fitness",
        "lane_count_mae",
        "tp",
        "fp",
        "fn",
        "ape_mean_px",
        "ape_matched_all_mean_px",
        "ape_fp_matched_mean_px",
        "ape_all_matched_mean_px",
        "fp_matched_ape_mean_px",
        "ape_median_px",
        "curvature_error_mean_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_conf_grid(
    model: torch.nn.Module,
    source: str | Path,
    labels: str | Path | None,
    imgsz: tuple[int, int],
    confs: list[float],
    ape_thr: float,
    max_det: int,
    max_images: int,
    device: torch.device,
    half: bool,
    save_records: bool,
    match_gate_px: float | None,
    max_x_dist: float,
    min_overlap: int,
    nms_dist_pxs: list[float],
    point_valid_thrs: list[float],
) -> dict:
    """Run one forward pass per image and evaluate all confidence thresholds."""
    images = collect_images(source, max_images=max_images)
    label_dir = None if labels is None or str(labels).strip() == "" else Path(labels)
    states = {
        (point_valid_thr, nms, conf): empty_state(conf, nms, point_valid_thr)
        for point_valid_thr in point_valid_thrs
        for nms in nms_dist_pxs
        for conf in confs
    }
    total_infer = 0.0
    total_post = 0.0

    for image_path in images:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        assert_gcs_shape(img.shape[:2], imgsz, name="sweep image", context=f"sweep_gcs_conf({image_path})")
        gt_lanes, gt_valid = load_gcs_label(label_path_for_image(image_path, label_dir))
        tensor = preprocess_image(img, imgsz=imgsz, device=device, half=half)

        _sync_if_cuda(device)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device)
        total_infer += time.perf_counter() - t0

        t1 = time.perf_counter()
        pred_valid = preds.get("pred_valid_logits")
        for point_valid_thr in point_valid_thrs:
            for nms_dist_px in nms_dist_pxs:
                for conf in confs:
                    pred_lanes = decode_gcs_predictions(
                        preds["pred_points"][0],
                        preds["pred_logits"][0],
                        pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
                        image_shape=img.shape[:2],
                        score_thr=conf,
                        point_valid_thr=point_valid_thr,
                        max_det=max_det,
                        nms_dist_px=nms_dist_px,
                    )
                    update_state(
                        states[(point_valid_thr, nms_dist_px, conf)],
                        image_path,
                        gt_lanes,
                        gt_valid,
                        pred_lanes,
                        img.shape[:2],
                        ape_thr,
                        match_gate_px,
                        max_x_dist,
                        min_overlap,
                        save_records,
                    )
        total_post += time.perf_counter() - t1

    rows = [
        summarize_state(states[(point_valid_thr, nms, conf)], ape_thr=ape_thr)
        for point_valid_thr in point_valid_thrs
        for nms in nms_dist_pxs
        for conf in confs
    ]
    return {
        "rows": rows,
        "records_by_conf": {
            f"pvalid={point_valid_thr:.6g},nms={nms:.6g},conf={conf:.6g}": states[
                (point_valid_thr, nms, conf)
            ]["records"]
            for point_valid_thr in point_valid_thrs
            for nms in nms_dist_pxs
            for conf in confs
        }
        if save_records
        else None,
        "timing": {
            "images": len(images),
            "avg_inference_ms": round(total_infer * 1000.0 / max(len(images), 1), 4),
            "avg_sweep_postprocess_ms": round(total_post * 1000.0 / max(len(images), 1), 4),
        },
    }


def print_rows(rows: list[dict]) -> None:
    """Print a compact threshold table."""
    print("pvalid  nms_px  conf  precision  recall     f1  lane_count_mae  pred_lanes_hist")
    for row in rows:
        print(
            f"{row['point_valid_thr']:.2f}  {row['nms_dist_px']:.1f}  {row['conf']:.2f}  "
            f"{row['precision']:.6f}  {row['recall']:.6f}  {row['f1']:.6f}  "
            f"{row['lane_count_mae']:.6f}  {row['pred_lanes_hist']}"
        )


def main() -> None:
    args = parse_args()
    args.split = assert_legacy_query_conf_sweep_split(args.split)
    data = resolve_dataset(args)
    imgsz = normalize_imgsz(args.imgsz or data.get("gcs_imgsz") or data.get("image_shape"), dataset=args.dataset)
    confs = conf_values(args)
    nms_dist_pxs = nms_values(args)
    point_valid_thrs = point_valid_values(args)
    source = args.source or data.get(args.split)
    labels = args.labels or labels_from_source(source)
    if not source:
        raise ValueError(f"{args.split!r} source is required via --source or data yaml {args.split}.")

    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=imgsz)
    assert_legacy_query_conf_sweep_model(model)

    if args.warmup > 0:
        warm_images = collect_images(source, max_images=1)
        warm_img = cv2.imread(str(warm_images[0]), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_images[0]}")
        warm_tensor = preprocess_image(warm_img, imgsz=imgsz, device=device, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device)

    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"sweeping confs: {', '.join(f'{x:.2f}' for x in confs)}")
    print(f"sweeping Lane-NMS px: {', '.join(f'{x:.1f}' for x in nms_dist_pxs)}")
    print(f"sweeping point-valid thresholds: {', '.join(f'{x:.2f}' for x in point_valid_thrs)}")
    val_result = evaluate_conf_grid(
        model=model,
        source=source,
        labels=labels,
        imgsz=imgsz,
        confs=confs,
        ape_thr=args.ape_thr,
        max_det=args.max_det,
        max_images=args.max_images,
        device=device,
        half=args.half,
        save_records=args.save_records,
        match_gate_px=args.match_gate_px,
        max_x_dist=args.max_x_dist,
        min_overlap=args.min_overlap,
        nms_dist_pxs=nms_dist_pxs,
        point_valid_thrs=point_valid_thrs,
    )
    rows = val_result["rows"]
    best = select_best(rows, args.select_by)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    write_csv(save_dir / "conf_sweep_val.csv", rows)
    output = {
        "best": best,
        "selection": {"select_by": args.select_by},
        "config": {
            "weights": str(Path(args.weights).resolve()),
            "data": data.get("yaml_file"),
            "split": args.split,
            "source": str(Path(source).resolve()),
            "labels": None if labels is None else str(Path(labels).resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "confs": confs,
            "nms_dist_pxs": nms_dist_pxs,
            "point_valid_thrs": point_valid_thrs,
            "ape_threshold_px": float(args.ape_thr),
            "match_gate_px": float(args.ape_thr if args.match_gate_px is None else args.match_gate_px),
            "max_x_dist": float(args.max_x_dist),
            "min_overlap": int(args.min_overlap),
            "max_det": int(args.max_det),
            "device": str(args.device),
            "half": bool(args.half),
        },
        "val": rows,
        "timing": val_result["timing"],
    }
    if args.save_records:
        output["val_records_by_conf"] = val_result["records_by_conf"]

    if args.run_test:
        test_source = args.test_source or data.get("test")
        if not test_source:
            raise ValueError("--run-test requested but no test split is available. Pass --test-source.")
        test_labels = args.test_labels or labels_from_source(test_source)
        test_result = evaluate_conf_grid(
            model=model,
            source=test_source,
            labels=test_labels,
            imgsz=imgsz,
            confs=[float(best["conf"])],
            ape_thr=args.ape_thr,
            max_det=args.max_det,
            max_images=args.test_max_images,
            device=device,
            half=args.half,
            save_records=args.save_records,
            match_gate_px=args.match_gate_px,
            max_x_dist=args.max_x_dist,
            min_overlap=args.min_overlap,
            nms_dist_pxs=[float(best["nms_dist_px"])],
            point_valid_thrs=[float(best["point_valid_thr"])],
        )
        test_rows = test_result["rows"]
        write_csv(save_dir / "conf_sweep_test_best.csv", test_rows)
        output["test_at_best_conf"] = test_rows[0]
        output["test_config"] = {
            "source": str(Path(test_source).resolve()),
            "labels": None if test_labels is None else str(Path(test_labels).resolve()),
            "conf": float(best["conf"]),
            "nms_dist_px": float(best["nms_dist_px"]),
            "point_valid_thr": float(best["point_valid_thr"]),
            "max_images": int(args.test_max_images),
        }
        if args.save_records:
            output["test_records_by_conf"] = test_result["records_by_conf"]

    (save_dir / "conf_sweep_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print_rows(rows)
    print(
        f"best by {args.select_by}: conf={best['conf']:.2f} nms={best['nms_dist_px']:.1f} "
        f"pvalid={best['point_valid_thr']:.2f}  "
        f"f1={best['f1']:.6f}  lane_count_mae={best['lane_count_mae']:.6f}"
    )
    if args.run_test:
        test = output["test_at_best_conf"]
        print(
            f"test @ conf={best['conf']:.2f}, nms={best['nms_dist_px']:.1f}, "
            f"pvalid={best['point_valid_thr']:.2f}: "
            f"f1={test['f1']:.6f} lane_count_mae={test['lane_count_mae']:.6f}"
        )
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
