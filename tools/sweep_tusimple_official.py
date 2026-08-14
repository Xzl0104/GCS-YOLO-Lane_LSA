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
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    official_gt_contract_summary,
    official_metric_score,
    read_tusimple_json_lines,
    resolve_official_output_shape,
    resolve_tusimple_gt_json,
    tusimple_image_path,
)
from gcs_tools.official_selection import sweep_selection_policy, sweep_sort_key  # noqa: E402
from gcs_tools.tusimple_split_guard import reject_tusimple_test_search_gt_json  # noqa: E402
from tools.eval_tusimple_official import _count_diagnostics  # noqa: E402
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.models.gcs.decode_summary import (  # noqa: E402
    ORDERED_SLOT_DECODE_SCHEMA,
    QUERY_DECODE_SCHEMA,
    build_ordered_slot_decode_summary,
    load_decode_yaml,
    ordered_slot_decode_params,
    ordered_slot_decode_runtime_config,
    ordered_slot_order_diagnostics_summary,
    raise_for_ordered_slot_query_args,
    validate_decode_yaml_for_model,
)
from ultralytics.models.gcs.decode_ordered_slot import decode_ordered_slot_predictions  # noqa: E402
from ultralytics.models.gcs.mode_utils import resolve_decode_mode  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03" / "weights" / "best.pt"
ORDERED_SLOT_QUERY_ONLY_DEFAULTS = {
    "confs": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25],
    "point_valid_thrs": [0.30, 0.35, 0.40, 0.45, 0.50],
    "nms_dist_pxs": [18.0],
    "max_dets": [8],
    "min_points": [6],
    "valid_before_maxdet": False,
    "count_aware_topk": False,
    "count_aware_min_k": 3,
    "count_aware_max_k": 5,
    "count_aware_length_norm": 12.0,
    "count_aware_extra_margins": [0],
    "count_modes": ["score_sum"],
}
QUERY_ONLY_ROW_KEYS = (
    "conf",
    "point_valid_thr",
    "nms_dist_px",
    "max_det",
    "min_points",
    "count_aware_topk",
    "count_aware_min_k",
    "count_aware_max_k",
    "count_aware_length_norm",
    "count_aware_extra_margin",
    "count_mode",
)


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
    parser.add_argument(
        "--allow-noncanonical-gt",
        action="store_true",
        help="Allow non-363 split=val GT for diagnostics; summary marks it incomparable with E1/spurious.",
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt.")
    parser.add_argument("--decode-mode", choices=("auto", "query", "ordered_slot"), default="auto", help="Decode path for official sweep.")
    parser.add_argument("--decode-yaml", default=None, help="Schema-validated official_best_decode.yaml to reproduce a decode.")
    parser.add_argument(
        "--official-output-shape",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="TuSimple official prediction coordinate shape as H W. Defaults to 720 1280 for TuSimple.",
    )
    parser.add_argument("--gcs-min-lanes", type=int, default=2, help="ordered_slot minimum supported lane count.")
    parser.add_argument("--gcs-max-lanes", type=int, default=5, help="ordered_slot maximum supported lane count.")
    parser.add_argument("--gcs-num-slots", type=int, default=5, help="ordered_slot slot count.")
    parser.add_argument(
        "--gcs-min-interval-points",
        type=int,
        default=2,
        help="ordered_slot minimum decoded start/end interval length.",
    )
    parser.add_argument(
        "--gcs-bottom-order-margin-px",
        type=float,
        default=2.0,
        help="ordered_slot bottom-x left-to-right order margin in pixels.",
    )
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
    parser.add_argument("--valid-before-maxdet", action="store_true", help="Filter point-valid/min_points failures before max_det truncation.")
    parser.add_argument("--count-aware-topk", action="store_true", help="Use count_score to keep only the quality-best dynamic lane count.")
    parser.add_argument("--count-aware-min-k", type=int, default=3, help="Minimum k_hat for --count-aware-topk.")
    parser.add_argument("--count-aware-max-k", type=int, default=5, help="Maximum k_hat for --count-aware-topk.")
    parser.add_argument("--count-aware-length-norm", type=float, default=12.0, help="Visible-point count that saturates count-aware length quality.")
    parser.add_argument(
        "--count-aware-extra-margins",
        nargs="+",
        type=int,
        default=[0],
        help="Extra lanes to keep above k_hat for count-aware top-k, capped by max_det.",
    )
    parser.add_argument(
        "--count-modes",
        nargs="+",
        choices=("score_sum", "count_logits"),
        default=["score_sum"],
        help="Count source for query count-aware top-k sweep.",
    )
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
    count_aware_topk: bool = False,
    count_aware_extra_margins: list[int] | tuple[int, ...] | None = None,
    valid_before_maxdet: bool = False,
    decode_mode: str = "auto",
) -> Path:
    if save_dir is not None and str(save_dir).strip():
        return Path(save_dir)
    if str(decode_mode) == "ordered_slot":
        suffix = "_ordered_slot"
    else:
        suffix = ""
        if count_aware_topk:
            suffix += "_count_aware_topk"
            margins = sorted({int(x) for x in (count_aware_extra_margins or [0])})
            if margins != [0]:
                suffix += "_extra_margin"
        if valid_before_maxdet:
            suffix += "_valid_before_maxdet"
    run_dir = _weight_run_dir(weights)
    if run_dir is not None:
        return run_dir / f"official_sweep_{split}{suffix}"
    return ROOT / "runs" / "gcs_lane" / "tusimple_official_sweep" / Path(weights).stem / f"official_sweep_{split}{suffix}"


def _limit_records(records: list[dict], max_images: int) -> list[dict]:
    if max_images and max_images > 0:
        return records[: int(max_images)]
    return records


def _combo_key(combo: dict) -> tuple:
    if combo.get("decode_mode") == "ordered_slot":
        return ("ordered_slot",)
    return (
        float(combo["conf"]),
        float(combo["point_valid_thr"]),
        float(combo["nms_dist_px"]),
        int(combo["max_det"]),
        int(combo["min_points"]),
        str(combo.get("count_mode", "score_sum")),
        int(combo.get("count_aware_extra_margin", 0)),
    )


def _ordered_slot_runtime_context(args: argparse.Namespace) -> str:
    return str(getattr(args, "ordered_slot_runtime_context", "official_sweep") or "official_sweep")


def build_combos(args: argparse.Namespace, decode_yaml_cfg: dict | None = None) -> list[dict]:
    combos: list[dict] = []
    decode_mode = str(getattr(args, "decode_mode", "query"))
    if decode_mode == "ordered_slot":
        runtime_cfg = ordered_slot_decode_runtime_config(context=_ordered_slot_runtime_context(args))
        ordered_params = ordered_slot_decode_params(args, decode_yaml_cfg)
        effective_decode = build_ordered_slot_decode_summary(
            min_lanes=ordered_params["min_lanes"],
            max_lanes=ordered_params["max_lanes"],
            num_slots=ordered_params["num_slots"],
            min_interval_points=ordered_params["min_interval_points"],
            order_margin_px=ordered_params["order_margin_px"],
            output_order=runtime_cfg["output_order"],
            order_check=runtime_cfg["order_check"],
        )
        return [
            {
                "decode_mode": "ordered_slot",
                "decode_schema": effective_decode["schema"],
                "effective_decode": effective_decode,
                "query_decode_args": effective_decode["query_decode_args"],
            }
        ]
    count_aware_topk = bool(getattr(args, "count_aware_topk", False))
    count_aware_min_k = int(getattr(args, "count_aware_min_k", 3))
    count_aware_max_k = int(getattr(args, "count_aware_max_k", 5))
    count_aware_length_norm = float(getattr(args, "count_aware_length_norm", 12.0))
    count_aware_extra_margins = sorted({int(x) for x in getattr(args, "count_aware_extra_margins", [0])})
    count_modes = sorted({str(x) for x in getattr(args, "count_modes", ["score_sum"])})
    valid_before_maxdet = bool(getattr(args, "valid_before_maxdet", False))
    if any(int(x) < 0 for x in count_aware_extra_margins):
        raise ValueError(f"count-aware extra margins must be >= 0, got {count_aware_extra_margins}.")
    if count_aware_topk:
        if count_aware_min_k < 0 or count_aware_max_k < 0 or count_aware_min_k > count_aware_max_k:
            raise ValueError(
                "count-aware k bounds must satisfy 0 <= min_k <= max_k, "
                f"got min_k={count_aware_min_k}, max_k={count_aware_max_k}."
            )
        if count_aware_length_norm <= 0.0:
            raise ValueError(f"count-aware length norm must be > 0, got {count_aware_length_norm}.")
    else:
        count_aware_extra_margins = [0]
    for conf, point_valid_thr, nms_dist_px, max_det, min_points, count_mode, count_aware_extra_margin in product(
        sorted({float(x) for x in args.confs}),
        sorted({float(x) for x in args.point_valid_thrs}),
        sorted({float(x) for x in args.nms_dist_pxs}),
        sorted({int(x) for x in args.max_dets}),
        sorted({int(x) for x in args.min_points}),
        count_modes,
        count_aware_extra_margins,
    ):
        if point_valid_thr < 0.0 or point_valid_thr > 1.0:
            raise ValueError(f"point-valid thresholds must be in [0, 1], got {point_valid_thr}.")
        if nms_dist_px < 0.0:
            raise ValueError(f"nms-dist-pxs must be >= 0, got {nms_dist_px}.")
        if max_det <= 0 or min_points <= 0:
            raise ValueError(f"max_det and min_points must be positive, got max_det={max_det}, min_points={min_points}.")
        combos.append(
            {
                "decode_mode": "query",
                "decode_schema": QUERY_DECODE_SCHEMA,
                "conf": conf,
                "point_valid_thr": point_valid_thr,
                "nms_dist_px": nms_dist_px,
                "max_det": max_det,
                "min_points": min_points,
                "valid_before_maxdet": valid_before_maxdet,
                "count_aware_topk": count_aware_topk,
                "count_aware_min_k": count_aware_min_k,
                "count_aware_max_k": count_aware_max_k,
                "count_aware_length_norm": count_aware_length_norm,
                "count_aware_extra_margin": int(count_aware_extra_margin),
                "count_mode": count_mode,
            }
        )
    if not combos:
        raise ValueError("No threshold combinations to evaluate.")
    return combos


def select_best(rows: list[dict]) -> dict:
    """Pick the official-Accuracy best row with the project checkpoint-selection tie-breakers."""
    return dict(max(rows, key=sweep_sort_key))


def _row_sort_key(row: dict) -> tuple:
    if row.get("decode_mode") == "ordered_slot":
        return ("ordered_slot", 0.0, 0.0, 0, 0)
    return (
        float(row["conf"]),
        float(row["point_valid_thr"]),
        float(row["nms_dist_px"]),
        int(row["max_det"]),
        int(row["min_points"]),
        str(row.get("count_mode", "score_sum")),
        int(row.get("count_aware_extra_margin", 0)),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "decode_mode",
        "decode_schema",
        "effective_decode",
        "query_decode_args",
        "conf",
        "point_valid_thr",
        "nms_dist_px",
        "max_det",
        "min_points",
        "valid_before_maxdet",
        "count_aware_topk",
        "count_aware_min_k",
        "count_aware_max_k",
        "count_aware_length_norm",
        "count_aware_extra_margin",
        "count_mode",
        "strict_order_valid",
        "ordered_slot_order_violations",
        "ordered_slot_order_violation_images",
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
            if out.get("decode_mode") == "ordered_slot":
                out["decode_schema"] = ORDERED_SLOT_DECODE_SCHEMA
                out["query_decode_args"] = "not_applicable"
                for key in QUERY_ONLY_ROW_KEYS:
                    out[key] = ""
            for key in ("pred_lanes_hist", "gt_lanes_hist", "count_confusion"):
                out[key] = json.dumps(out.get(key, {}), sort_keys=True, separators=(",", ":"))
            if isinstance(out.get("effective_decode"), dict):
                out["effective_decode"] = json.dumps(out["effective_decode"], sort_keys=True, separators=(",", ":"))
            writer.writerow(out)


@torch.inference_mode()
def sweep(args: argparse.Namespace) -> dict:
    args.split = validate_search_split(args.split)
    reject_tusimple_test_search_gt_json(args.gt_json, context="TuSimple official sweep")
    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_path = resolve_tusimple_gt_json(archive_root, split=args.split, gt_json=args.gt_json)
    gt_records = _limit_records(read_tusimple_json_lines(gt_path), args.max_images)
    if not gt_records:
        raise ValueError(f"No TuSimple GT records found in {gt_path}")
    gt_contract = official_gt_contract_summary(
        split=args.split,
        gt_json=gt_path,
        gt_records=gt_records,
        allow_noncanonical_gt=bool(getattr(args, "allow_noncanonical_gt", False)),
    )

    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    official_output_shape = resolve_official_output_shape(
        getattr(args, "official_output_shape", None),
        dataset=args.dataset,
    )
    device_obj = select_device(args.device)
    model = load_gcs_model(args.weights, device=device_obj, half=args.half, gcs_imgsz=imgsz)
    decode_yaml_cfg = None
    if getattr(args, "decode_yaml", None):
        _, decode_yaml_cfg = load_decode_yaml(args.decode_yaml)
        args.decode_mode = resolve_decode_mode(decode_yaml_cfg.get("decode_mode"), model)
        validate_decode_yaml_for_model(decode_yaml_cfg, model_mode=args.decode_mode)
        if args.decode_mode == "query":
            args.confs = [float(decode_yaml_cfg["conf"])]
            args.point_valid_thrs = [float(decode_yaml_cfg["point_valid_thr"])]
            args.nms_dist_pxs = [float(decode_yaml_cfg["nms_dist_px"])]
            args.max_dets = [int(decode_yaml_cfg["max_det"])]
            args.min_points = [int(decode_yaml_cfg["min_points"])]
            args.valid_before_maxdet = bool(decode_yaml_cfg.get("valid_before_maxdet", False))
            args.count_aware_topk = bool(decode_yaml_cfg["count_aware_topk"])
            args.count_aware_min_k = int(decode_yaml_cfg["count_aware_min_k"])
            args.count_aware_max_k = int(decode_yaml_cfg["count_aware_max_k"])
            args.count_aware_length_norm = float(decode_yaml_cfg["count_aware_length_norm"])
            args.count_aware_extra_margins = [int(decode_yaml_cfg.get("count_aware_extra_margin", 0))]
            args.count_modes = [str(decode_yaml_cfg.get("count_mode", "score_sum"))]
    else:
        args.decode_mode = resolve_decode_mode(getattr(args, "decode_mode", "auto"), model)
    if str(getattr(args, "decode_mode", "query")) == "ordered_slot":
        raise_for_ordered_slot_query_args(args, ORDERED_SLOT_QUERY_ONLY_DEFAULTS, context="TuSimple official sweep")
    combos = build_combos(args, decode_yaml_cfg=decode_yaml_cfg)
    if str(getattr(args, "decode_mode", "query")) != "ordered_slot":
        for max_det in sorted({int(c["max_det"]) for c in combos}):
            warn_max_det_mismatch(args.weights, max_det=max_det, context="TuSimple official sweep")
    if args.warmup > 0 and gt_records:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device_obj, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    combo_records = {_combo_key(combo): [] for combo in combos}
    combo_order_stats = {
        _combo_key(combo): {"ordered_slot_order_violations": 0, "ordered_slot_order_violation_images": 0}
        for combo in combos
    }
    ordered_slot_runtime_cfg = ordered_slot_decode_runtime_config(context=_ordered_slot_runtime_context(args))
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

        for combo in combos:
            if combo["decode_mode"] == "ordered_slot":
                ordered_params = ordered_slot_decode_params(args, decode_yaml_cfg)
                lanes, order_diag = decode_ordered_slot_predictions(
                    preds,
                    batch_index=0,
                    image_shape=original_shape,
                    min_lanes=ordered_params["min_lanes"],
                    max_lanes=ordered_params["max_lanes"],
                    min_interval_points=ordered_params["min_interval_points"],
                    order_margin_px=ordered_params["order_margin_px"],
                    img_w=float(original_shape[1]),
                    order_check=ordered_slot_runtime_cfg["order_check"],
                    output_order=ordered_slot_runtime_cfg["output_order"],
                    return_diagnostics=True,
                )
                stats = combo_order_stats[_combo_key(combo)]
                stats["ordered_slot_order_violations"] += int(order_diag["order_violation_count"])
                stats["ordered_slot_order_violation_images"] += int(order_diag["has_order_violation"])
            else:
                pred_valid = preds.get("pred_valid_logits")
                pred_count_logits = preds.get("pred_count_logits")
                lanes = decode_gcs_predictions(
                    preds["pred_points"][0],
                    preds["pred_logits"][0],
                    pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
                    pred_count_logits=pred_count_logits[0] if pred_count_logits is not None else None,
                    image_shape=original_shape,
                    score_thr=combo["conf"],
                    point_valid_thr=combo["point_valid_thr"],
                    min_points=combo["min_points"],
                    max_det=combo["max_det"],
                    nms_dist_px=combo["nms_dist_px"],
                    valid_before_maxdet=combo["valid_before_maxdet"],
                    count_aware_topk=combo["count_aware_topk"],
                    count_aware_min_k=combo["count_aware_min_k"],
                    count_aware_max_k=combo["count_aware_max_k"],
                    count_aware_length_norm=combo["count_aware_length_norm"],
                    count_aware_extra_margin=combo["count_aware_extra_margin"],
                    count_mode=combo["count_mode"],
                )
            tusimple_lanes = gcs_lanes_to_tusimple_lanes(lanes, record["h_samples"], image_shape=official_output_shape)
            combo_records[_combo_key(combo)].append(
                {
                    "lanes": tusimple_lanes,
                    "h_samples": record["h_samples"],
                    "raw_file": raw_file,
                    "run_time": float(args.runtime_ms),
                }
            )
        post_time_s += time.perf_counter() - t1
        infer_time_s += t1 - t0

    rows: list[dict] = []
    for combo in combos:
        pred_records = combo_records[_combo_key(combo)]
        result, _ = TuSimpleOfficialLaneEval.bench_records(pred_records, gt_records, strict_length=True, return_records=False)
        metrics = result.as_dict()
        row = {
            **combo,
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
        if combo["decode_mode"] == "ordered_slot":
            row.update(combo_order_stats[_combo_key(combo)])
            row["strict_order_valid"] = int(row["ordered_slot_order_violations"]) == 0
        rows.append(row)

    rows = sorted(rows, key=_row_sort_key)
    best = select_best(rows)
    effective_count_aware_extra_margins = sorted(
        {int(combo.get("count_aware_extra_margin", 0)) for combo in combos if combo.get("decode_mode") == "query"}
    )
    save_dir = resolve_save_dir(
        args.save_dir,
        args.weights,
        args.split,
        count_aware_topk=bool(getattr(args, "count_aware_topk", False)),
        count_aware_extra_margins=effective_count_aware_extra_margins,
        valid_before_maxdet=bool(getattr(args, "valid_before_maxdet", False)),
        decode_mode=str(getattr(args, "decode_mode", "query")),
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    write_csv(save_dir / "tusimple_official_sweep.csv", rows)

    n = max(len(gt_records), 1)
    config = {
        "weights": str(Path(args.weights).resolve()),
        "archive_root": str(archive_root.resolve()),
        "split": args.split,
        "gt_json": str(gt_path.resolve()),
        "save_dir": str(save_dir.resolve()),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "official_output_shape": [int(official_output_shape[0]), int(official_output_shape[1])],
        "decode_mode": str(getattr(args, "decode_mode", "query")),
        "runtime_ms": float(args.runtime_ms),
        "max_images": int(args.max_images),
        "device": str(args.device),
        "half": bool(args.half),
        "best_metric": "official_acc",
        "selection_policy": sweep_selection_policy(),
        "score_fp_weight": float(args.score_fp_weight),
        "score_fn_weight": float(args.score_fn_weight),
        **gt_contract,
    }
    if str(getattr(args, "decode_mode", "query")) == "ordered_slot":
        ordered_slot_runtime_cfg = ordered_slot_decode_runtime_config(context=_ordered_slot_runtime_context(args))
        ordered_params = ordered_slot_decode_params(args, decode_yaml_cfg)
        effective_decode = build_ordered_slot_decode_summary(
            min_lanes=ordered_params["min_lanes"],
            max_lanes=ordered_params["max_lanes"],
            num_slots=ordered_params["num_slots"],
            min_interval_points=ordered_params["min_interval_points"],
            order_margin_px=ordered_params["order_margin_px"],
            output_order=ordered_slot_runtime_cfg["output_order"],
            order_check=ordered_slot_runtime_cfg["order_check"],
        )
        config.update(
            {
                "schema": effective_decode["schema"],
                "gcs_min_lanes": effective_decode["gcs_min_lanes"],
                "gcs_max_lanes": effective_decode["gcs_max_lanes"],
                "gcs_num_slots": effective_decode["gcs_num_slots"],
                "min_interval_points": effective_decode["min_interval_points"],
                "gcs_bottom_order_margin_px": effective_decode["gcs_bottom_order_margin_px"],
                "interval_repair": effective_decode["interval_repair"],
                "output_order": effective_decode["output_order"],
                "order_check": effective_decode["order_check"],
                "uses_runtime_sort": effective_decode["uses_runtime_sort"],
                "order_violation_policy": effective_decode["order_violation_policy"],
                "result_type": effective_decode["result_type"],
                "not_for_main_ordered_slot_claim": effective_decode["not_for_main_ordered_slot_claim"],
                "effective_decode": effective_decode,
                "query_decode_args": effective_decode["query_decode_args"],
            }
        )
    else:
        config.update(
            {
                "schema": "query_decode_v1",
                "confs": [float(x) for x in sorted({float(x) for x in args.confs})],
                "point_valid_thrs": [float(x) for x in sorted({float(x) for x in args.point_valid_thrs})],
                "nms_dist_pxs": [float(x) for x in sorted({float(x) for x in args.nms_dist_pxs})],
                "max_dets": [int(x) for x in sorted({int(x) for x in args.max_dets})],
                "min_points": [int(x) for x in sorted({int(x) for x in args.min_points})],
                "valid_before_maxdet": bool(getattr(args, "valid_before_maxdet", False)),
                "count_aware_topk": bool(getattr(args, "count_aware_topk", False)),
                "count_aware_min_k": int(getattr(args, "count_aware_min_k", 3)),
                "count_aware_max_k": int(getattr(args, "count_aware_max_k", 5)),
                "count_aware_length_norm": float(getattr(args, "count_aware_length_norm", 12.0)),
                "count_aware_extra_margins": effective_count_aware_extra_margins,
                "count_modes": [str(x) for x in sorted({str(x) for x in getattr(args, "count_modes", ["score_sum"])})],
            }
        )

    output = {
        "best": best,
        "results": rows,
        "selection_policy": sweep_selection_policy(),
        "config": config,
        **gt_contract,
        "timing": {
            "avg_inference_ms": round(infer_time_s * 1000.0 / n, 4),
            "avg_sweep_postprocess_ms": round(post_time_s * 1000.0 / n, 4),
        },
    }
    if str(getattr(args, "decode_mode", "query")) == "ordered_slot":
        output["effective_decode"] = config["effective_decode"]
        output["query_decode_args"] = "not_applicable"
        output.update(
            ordered_slot_order_diagnostics_summary(
                pred_json_mode=False,
                decode_stats=best,
            )
        )
    (save_dir / "tusimple_official_sweep_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"official output shape H,W={official_output_shape}")
    print(f"swept {len(rows)} combinations on {len(gt_records)} images")
    print(f"saved to: {save_dir.resolve()}")
    return output


def main() -> None:
    sweep(parse_args())


if __name__ == "__main__":
    main()
