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
    find_tusimple_archive_root,
    gcs_lanes_to_tusimple_lanes,
    official_gt_contract_summary,
    official_metric_score,
    read_tusimple_json_lines,
    resolve_tusimple_gt_json,
    tusimple_image_path,
    write_tusimple_predictions,
)
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.models.gcs.decode_summary import (  # noqa: E402
    ORDERED_SLOT_QUERY_DECODE_DEFAULTS,
    build_ordered_slot_decode_summary,
    guard_no_query_decode_args_for_ordered_slot,
    load_decode_yaml,
    ordered_slot_decode_params,
    ordered_slot_decode_runtime_config,
    ordered_slot_order_diagnostics_summary,
    validate_decode_yaml_for_model,
)
from ultralytics.models.gcs.decode_ordered_slot import decode_ordered_slot_predictions  # noqa: E402
from ultralytics.models.gcs.mode_utils import resolve_decode_mode  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03" / "weights" / "best.pt"
ORDERED_SLOT_QUERY_ONLY_DEFAULTS = ORDERED_SLOT_QUERY_DECODE_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GCS-YOLO-Lane with the TuSimple official metric.")
    parser.add_argument("--dataset", default="tusimple", choices=("tusimple",))
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--split", default="test", choices=("test", "train", "val"), help="TuSimple archive split.")
    parser.add_argument("--gt-json", default=None, help="Official TuSimple GT json-lines file.")
    parser.add_argument(
        "--allow-noncanonical-gt",
        action="store_true",
        help="Allow non-363 split=val GT for diagnostics; summary marks it incomparable with E1/spurious.",
    )
    parser.add_argument("--pred-json", default=None, help="Evaluate an existing TuSimple-format prediction json-lines file.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt used when --pred-json is not set.")
    parser.add_argument("--decode-mode", choices=("auto", "query", "ordered_slot"), default="auto", help="Decode path for official eval.")
    parser.add_argument("--decode-yaml", default=None, help="Schema-validated official_best_decode.yaml to use for final eval.")
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
    parser.add_argument("--valid-before-maxdet", action="store_true", help="Filter point-valid/min_points failures before max_det truncation.")
    parser.add_argument("--extent-decode", action="store_true", help="Use query start/end extent logits for query visibility.")
    parser.add_argument(
        "--extent-decode-mode",
        choices=("none", "interval", "intersect"),
        default="interval",
        help="Query extent visibility mode. 'none' preserves point-valid decode.",
    )
    parser.add_argument(
        "--candidate-decode",
        action="store_true",
        help="Use lateral candidate hypotheses only for prediction-gated short-lane queries.",
    )
    parser.add_argument(
        "--candidate-short-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply candidate selection only when predicted visible-anchor count is in the configured short-lane range.",
    )
    parser.add_argument("--candidate-gate-valid-thr", type=float, default=0.5)
    parser.add_argument("--candidate-gate-min-visible", type=int, default=2)
    parser.add_argument("--candidate-gate-max-visible", type=int, default=10)
    parser.add_argument(
        "--candidate-preserve-base-score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the original query score after candidate selection.",
    )
    parser.add_argument("--count-aware-topk", action="store_true", help="Use count_score to keep only the quality-best dynamic lane count.")
    parser.add_argument("--count-aware-min-k", type=int, default=3, help="Minimum k_hat for --count-aware-topk.")
    parser.add_argument("--count-aware-max-k", type=int, default=5, help="Maximum k_hat for --count-aware-topk.")
    parser.add_argument("--count-aware-length-norm", type=float, default=12.0, help="Visible-point count that saturates count-aware length quality.")
    parser.add_argument(
        "--count-aware-extra-margin",
        type=int,
        default=0,
        help="Extra lanes to keep above k_hat for count-aware top-k, capped by --max-det.",
    )
    parser.add_argument(
        "--count-mode",
        choices=("score_sum", "count_logits"),
        default="score_sum",
        help="Count source for query count-aware top-k.",
    )
    parser.add_argument(
        "--oracle-count",
        action="store_true",
        help="Diagnostic only: force query count-aware top-k k_hat to the GT lane count for each image.",
    )
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
    count_aware_extra_margin: int = 0,
    count_aware_topk: bool = False,
    valid_before_maxdet: bool = False,
    extent_decode: bool = False,
    extent_decode_mode: str = "interval",
    oracle_count: bool = False,
    candidate_decode: bool = False,
    candidate_short_gate: bool = True,
    decode_mode: str = "auto",
) -> Path:
    if save_dir is not None and str(save_dir).strip():
        return Path(save_dir)
    if str(decode_mode) == "ordered_slot":
        tag = f"official_{split}_ordered_slot"
    else:
        count_tag = "_catopk" if count_aware_topk else ""
        if count_aware_topk and int(count_aware_extra_margin) > 0:
            count_tag += f"_extra{int(count_aware_extra_margin)}"
        oracle_tag = "_oraclecount" if oracle_count else ""
        valid_tag = "_validbeforemaxdet" if valid_before_maxdet else ""
        extent_mode = str(extent_decode_mode or "none").strip().lower()
        extent_tag = f"_extent{extent_mode}" if bool(extent_decode) and extent_mode != "none" else ""
        candidate_tag = "_candidate_gated" if bool(candidate_decode) and bool(candidate_short_gate) else "_candidate" if bool(candidate_decode) else ""
        tag = (
            f"official_{split}_conf{float(conf):.4g}_pvalid{float(point_valid_thr):.4g}_"
            f"nms{float(nms_dist_px):.4g}_maxdet{int(max_det)}_minp{int(min_points)}"
            f"{count_tag}{oracle_tag}{valid_tag}{extent_tag}{candidate_tag}"
        ).replace(".", "p")
    run_dir = _weight_run_dir(weights)
    if run_dir is not None:
        return run_dir / tag
    return ROOT / "runs" / "gcs_lane" / "tusimple_official_eval" / Path(weights).stem / tag


def _limit_records(records: list[dict], max_images: int) -> list[dict]:
    if max_images and max_images > 0:
        return records[: int(max_images)]
    return records


def _normalize_decode_mode_name(decode_mode: str | None) -> str:
    mode = str(decode_mode or "").strip().lower()
    if mode in {"ordered-slot", "orderedslot"}:
        mode = "ordered_slot"
    return mode


def _apply_query_decode_yaml(args: argparse.Namespace, decode_yaml_cfg: dict) -> None:
    args.conf = float(decode_yaml_cfg["conf"])
    args.point_valid_thr = float(decode_yaml_cfg["point_valid_thr"])
    args.nms_dist_px = float(decode_yaml_cfg["nms_dist_px"])
    args.max_det = int(decode_yaml_cfg["max_det"])
    args.min_points = int(decode_yaml_cfg["min_points"])
    args.valid_before_maxdet = bool(decode_yaml_cfg.get("valid_before_maxdet", False))
    args.extent_decode = bool(decode_yaml_cfg.get("extent_decode", False))
    args.extent_decode_mode = str(decode_yaml_cfg.get("extent_decode_mode", "none") or "none")
    args.candidate_short_gate = bool(decode_yaml_cfg.get("candidate_short_gate", True))
    args.candidate_gate_valid_thr = float(decode_yaml_cfg.get("candidate_gate_valid_thr", 0.5))
    args.candidate_gate_min_visible = int(decode_yaml_cfg.get("candidate_gate_min_visible", 2))
    args.candidate_gate_max_visible = int(decode_yaml_cfg.get("candidate_gate_max_visible", 10))
    args.candidate_preserve_base_score = bool(decode_yaml_cfg.get("candidate_preserve_base_score", True))
    args.count_aware_topk = bool(decode_yaml_cfg["count_aware_topk"])
    args.count_aware_min_k = int(decode_yaml_cfg["count_aware_min_k"])
    args.count_aware_max_k = int(decode_yaml_cfg["count_aware_max_k"])
    args.count_aware_length_norm = float(decode_yaml_cfg["count_aware_length_norm"])
    args.count_aware_extra_margin = int(decode_yaml_cfg.get("count_aware_extra_margin", 0))
    args.count_mode = str(decode_yaml_cfg.get("count_mode", "score_sum"))


def _normalize_extent_decode_args(extent_decode: bool, extent_decode_mode: str) -> tuple[bool, str]:
    """Normalize disabled extent decode so ordered-slot guards see default query args."""
    enabled = bool(extent_decode)
    mode = str(extent_decode_mode or "interval").strip().lower()
    if mode in {"off", "false", "0"}:
        mode = "none"
    if not enabled:
        return False, "none"
    return True, mode


def resolve_pred_json_decode_contract(args: argparse.Namespace) -> tuple[str, dict | None] | None:
    """Resolve the explicit decode contract for --pred-json mode."""
    if not getattr(args, "pred_json", None):
        return None
    args.extent_decode, args.extent_decode_mode = _normalize_extent_decode_args(
        getattr(args, "extent_decode", False),
        getattr(args, "extent_decode_mode", "interval"),
    )

    if getattr(args, "decode_yaml", None):
        _, decode_yaml_cfg = load_decode_yaml(args.decode_yaml)
        yaml_decode_mode = _normalize_decode_mode_name(decode_yaml_cfg.get("decode_mode"))
        if yaml_decode_mode not in {"query", "ordered_slot"}:
            raise RuntimeError(
                f"Invalid decode_yaml decode_mode={yaml_decode_mode!r}; expected query or ordered_slot."
            )
        requested_decode_mode = _normalize_decode_mode_name(getattr(args, "decode_mode", "auto"))
        if requested_decode_mode not in {"auto", "query", "ordered_slot"}:
            raise RuntimeError(f"Invalid --decode-mode {requested_decode_mode!r} for --pred-json.")
        active_decode_mode = yaml_decode_mode if requested_decode_mode == "auto" else requested_decode_mode
        validate_decode_yaml_for_model(decode_yaml_cfg, model_mode=active_decode_mode)
        if active_decode_mode == "ordered_slot":
            guard_no_query_decode_args_for_ordered_slot(args, context="TuSimple official eval --pred-json")
        return active_decode_mode, decode_yaml_cfg

    decode_mode = _normalize_decode_mode_name(getattr(args, "decode_mode", "auto"))
    if decode_mode == "auto":
        raise RuntimeError(
            "--pred-json mode cannot use --decode-mode auto because no model is loaded. "
            "Please pass --decode-mode query, --decode-mode ordered_slot, or --decode-yaml official_best_decode.yaml."
        )
    if decode_mode not in {"query", "ordered_slot"}:
        raise RuntimeError(f"Invalid --decode-mode {decode_mode!r} for --pred-json.")
    if decode_mode == "ordered_slot":
        guard_no_query_decode_args_for_ordered_slot(args, context="TuSimple official eval --pred-json")
    return decode_mode, None


def _count_diagnostics(pred_records: list[dict], gt_records: list[dict]) -> dict:
    gt_by_raw = {str(x["raw_file"]): x for x in gt_records}
    pairs: list[tuple[int, int]] = []
    for pred in pred_records:
        gt = gt_by_raw[str(pred["raw_file"])]
        gt_count = _gt_lane_count(gt)
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


def _gt_lane_count(record: dict) -> int:
    return sum(1 for lane in record.get("lanes", []) if any(float(x) >= 0.0 for x in lane))


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
    count_aware_extra_margin: int = 0,
    count_mode: str = "score_sum",
    oracle_count: bool = False,
    valid_before_maxdet: bool = False,
    extent_decode: bool = False,
    extent_decode_mode: str = "interval",
    candidate_decode: bool = False,
    candidate_short_gate: bool = True,
    candidate_gate_valid_thr: float = 0.5,
    candidate_gate_min_visible: int = 2,
    candidate_gate_max_visible: int = 10,
    candidate_preserve_base_score: bool = True,
    decode_mode: str = "auto",
    decode_yaml_cfg: dict | None = None,
    gcs_min_lanes: int = 2,
    gcs_max_lanes: int = 5,
    gcs_num_slots: int = 5,
    gcs_min_interval_points: int = 2,
    gcs_bottom_order_margin_px: float = 2.0,
    query_decode_defaults: dict | None = None,
) -> tuple[list[dict], dict, str, dict]:
    device_obj = select_device(device)
    model = load_gcs_model(weights, device=device_obj, half=half, gcs_imgsz=imgsz)
    if decode_yaml_cfg is not None:
        decode_mode = resolve_decode_mode(decode_yaml_cfg.get("decode_mode"), model)
        validate_decode_yaml_for_model(decode_yaml_cfg, model_mode=decode_mode)
        if decode_mode == "query":
            conf = float(decode_yaml_cfg["conf"])
            point_valid_thr = float(decode_yaml_cfg["point_valid_thr"])
            nms_dist_px = float(decode_yaml_cfg["nms_dist_px"])
            max_det = int(decode_yaml_cfg["max_det"])
            min_points = int(decode_yaml_cfg["min_points"])
            valid_before_maxdet = bool(decode_yaml_cfg.get("valid_before_maxdet", False))
            extent_decode = bool(decode_yaml_cfg.get("extent_decode", False))
            extent_decode_mode = str(decode_yaml_cfg.get("extent_decode_mode", "none") or "none")
            candidate_decode = bool(decode_yaml_cfg.get("candidate_decode", False))
            count_aware_topk = bool(decode_yaml_cfg["count_aware_topk"])
            count_aware_min_k = int(decode_yaml_cfg["count_aware_min_k"])
            count_aware_max_k = int(decode_yaml_cfg["count_aware_max_k"])
            count_aware_length_norm = float(decode_yaml_cfg["count_aware_length_norm"])
            count_aware_extra_margin = int(decode_yaml_cfg.get("count_aware_extra_margin", 0))
            count_mode = str(decode_yaml_cfg.get("count_mode", "score_sum"))
    else:
        decode_mode = resolve_decode_mode(decode_mode, model)
    if oracle_count:
        if str(decode_mode) == "ordered_slot":
            raise RuntimeError("--oracle-count is query-decode diagnostic only and is not valid for ordered_slot.")
        count_aware_topk = True
        count_mode = "oracle_gt"
    extent_decode, extent_decode_mode = _normalize_extent_decode_args(extent_decode, extent_decode_mode)
    if str(decode_mode) == "ordered_slot" and query_decode_defaults is not None:
        guard_no_query_decode_args_for_ordered_slot(
            {
                "conf": conf,
                "point_valid_thr": point_valid_thr,
                "nms_dist_px": nms_dist_px,
                "min_points": min_points,
                "max_det": max_det,
                "valid_before_maxdet": valid_before_maxdet,
                "extent_decode": extent_decode,
                "extent_decode_mode": extent_decode_mode,
                "candidate_decode": candidate_decode,
                "count_aware_topk": count_aware_topk,
                "count_aware_min_k": count_aware_min_k,
                "count_aware_max_k": count_aware_max_k,
                "count_aware_length_norm": count_aware_length_norm,
                "count_aware_extra_margin": count_aware_extra_margin,
                "count_mode": count_mode,
            },
            context="TuSimple official eval",
            defaults=query_decode_defaults,
        )

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
    ordered_slot_runtime_cfg = ordered_slot_decode_runtime_config(context="official_eval")
    ordered_slot_order_stats = {
        "ordered_slot_order_violations": 0,
        "ordered_slot_order_violation_images": 0,
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

        if str(decode_mode) == "ordered_slot":
            ordered_params = ordered_slot_decode_params(
                {
                    "gcs_min_lanes": gcs_min_lanes,
                    "gcs_max_lanes": gcs_max_lanes,
                    "gcs_num_slots": gcs_num_slots,
                    "gcs_min_interval_points": gcs_min_interval_points,
                    "gcs_bottom_order_margin_px": gcs_bottom_order_margin_px,
                },
                decode_yaml_cfg,
            )
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
            ordered_slot_order_stats["ordered_slot_order_violations"] += int(order_diag["order_violation_count"])
            ordered_slot_order_stats["ordered_slot_order_violation_images"] += int(order_diag["has_order_violation"])
        else:
            pred_valid = preds.get("pred_valid_logits")
            pred_count_logits = preds.get("pred_count_logits")
            pred_quality_logits = preds.get("pred_quality_logits")
            pred_start_logits = preds.get("pred_start_logits")
            pred_end_logits = preds.get("pred_end_logits")
            pred_short_candidate_points = preds.get("pred_short_candidate_points")
            pred_short_candidate_logits = preds.get("pred_short_candidate_logits")
            lanes = decode_gcs_predictions(
                preds["pred_points"][0],
                preds["pred_logits"][0],
                pred_quality_logits=pred_quality_logits[0] if pred_quality_logits is not None else None,
                pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
                pred_count_logits=pred_count_logits[0] if pred_count_logits is not None else None,
                pred_start_logits=pred_start_logits[0] if pred_start_logits is not None else None,
                pred_end_logits=pred_end_logits[0] if pred_end_logits is not None else None,
                pred_short_candidate_points=(
                    pred_short_candidate_points[0] if pred_short_candidate_points is not None else None
                ),
                pred_short_candidate_logits=(
                    pred_short_candidate_logits[0] if pred_short_candidate_logits is not None else None
                ),
                oracle_count=_gt_lane_count(record) if oracle_count else None,
                image_shape=original_shape,
                score_thr=conf,
                point_valid_thr=point_valid_thr,
                min_points=min_points,
                max_det=max_det,
                nms_dist_px=nms_dist_px,
                valid_before_maxdet=valid_before_maxdet,
                extent_decode=extent_decode,
                extent_decode_mode=extent_decode_mode,
                candidate_decode=candidate_decode,
                candidate_short_gate=candidate_short_gate,
                candidate_gate_valid_thr=candidate_gate_valid_thr,
                candidate_gate_min_visible=candidate_gate_min_visible,
                candidate_gate_max_visible=candidate_gate_max_visible,
                candidate_preserve_base_score=candidate_preserve_base_score,
                count_aware_topk=count_aware_topk,
                count_aware_min_k=count_aware_min_k,
                count_aware_max_k=count_aware_max_k,
                count_aware_length_norm=count_aware_length_norm,
                count_aware_extra_margin=count_aware_extra_margin,
                count_mode=count_mode,
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
    return pred_records, timing, decode_mode, ordered_slot_order_stats


def evaluate_official(args: argparse.Namespace) -> dict:
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
    timing = {"avg_inference_ms": None, "avg_postprocess_ms": None, "avg_total_ms": None}
    ordered_slot_order_stats: dict = {}
    pred_json = args.pred_json
    active_decode_mode = str(args.decode_mode)
    decode_yaml_cfg = None
    if pred_json:
        resolved = resolve_pred_json_decode_contract(args)
        assert resolved is not None
        active_decode_mode, decode_yaml_cfg = resolved
        if decode_yaml_cfg is not None and active_decode_mode == "query":
            _apply_query_decode_yaml(args, decode_yaml_cfg)
    elif getattr(args, "decode_yaml", None):
        _, decode_yaml_cfg = load_decode_yaml(args.decode_yaml)
        yaml_decode_mode = str(decode_yaml_cfg.get("decode_mode", ""))
        args.decode_mode = yaml_decode_mode
        if yaml_decode_mode == "query":
            _apply_query_decode_yaml(args, decode_yaml_cfg)
    if pred_json:
        if bool(getattr(args, "oracle_count", False)):
            raise RuntimeError("--oracle-count requires model inference and cannot be used with --pred-json.")
        pred_path = Path(pred_json)
        pred_records = _limit_records(read_tusimple_json_lines(pred_path), args.max_images)
    else:
        query_conf = float(getattr(args, "conf", 0.25))
        query_point_valid_thr = float(getattr(args, "point_valid_thr", 0.5))
        query_nms_dist_px = float(getattr(args, "nms_dist_px", 18.0))
        query_max_det = int(getattr(args, "max_det", 8))
        query_min_points = int(getattr(args, "min_points", 6))
        query_valid_before_maxdet = bool(getattr(args, "valid_before_maxdet", False))
        query_extent_decode = bool(getattr(args, "extent_decode", False))
        query_extent_decode_mode = str(getattr(args, "extent_decode_mode", "interval") or "interval")
        query_candidate_decode = bool(getattr(args, "candidate_decode", False))
        query_candidate_short_gate = bool(getattr(args, "candidate_short_gate", True))
        query_candidate_gate_valid_thr = float(getattr(args, "candidate_gate_valid_thr", 0.5))
        query_candidate_gate_min_visible = int(getattr(args, "candidate_gate_min_visible", 2))
        query_candidate_gate_max_visible = int(getattr(args, "candidate_gate_max_visible", 10))
        query_candidate_preserve_base_score = bool(getattr(args, "candidate_preserve_base_score", True))
        query_count_aware_topk = bool(getattr(args, "count_aware_topk", False))
        query_count_aware_min_k = int(getattr(args, "count_aware_min_k", 3))
        query_count_aware_max_k = int(getattr(args, "count_aware_max_k", 5))
        query_count_aware_length_norm = float(getattr(args, "count_aware_length_norm", 12.0))
        query_count_aware_extra_margin = int(getattr(args, "count_aware_extra_margin", 0))
        query_count_mode = str(getattr(args, "count_mode", "score_sum"))
        query_oracle_count = bool(getattr(args, "oracle_count", False))
        if query_oracle_count:
            query_count_aware_topk = True
            query_count_mode = "oracle_gt"
        query_extent_decode, query_extent_decode_mode = _normalize_extent_decode_args(
            query_extent_decode,
            query_extent_decode_mode,
        )
        pred_records, timing, active_decode_mode, ordered_slot_order_stats = generate_predictions(
            weights=args.weights,
            archive_root=archive_root,
            split=args.split,
            gt_records=gt_records,
            imgsz=imgsz,
            conf=query_conf,
            point_valid_thr=query_point_valid_thr,
            nms_dist_px=query_nms_dist_px,
            max_det=query_max_det,
            min_points=query_min_points,
            valid_before_maxdet=query_valid_before_maxdet,
            extent_decode=query_extent_decode,
            extent_decode_mode=query_extent_decode_mode,
            candidate_decode=query_candidate_decode,
            candidate_short_gate=query_candidate_short_gate,
            candidate_gate_valid_thr=query_candidate_gate_valid_thr,
            candidate_gate_min_visible=query_candidate_gate_min_visible,
            candidate_gate_max_visible=query_candidate_gate_max_visible,
            candidate_preserve_base_score=query_candidate_preserve_base_score,
            count_aware_topk=query_count_aware_topk,
            count_aware_min_k=query_count_aware_min_k,
            count_aware_max_k=query_count_aware_max_k,
            count_aware_length_norm=query_count_aware_length_norm,
            count_aware_extra_margin=query_count_aware_extra_margin,
            count_mode=query_count_mode,
            oracle_count=query_oracle_count,
            decode_mode=args.decode_mode,
            decode_yaml_cfg=decode_yaml_cfg,
            gcs_min_lanes=int(getattr(args, "gcs_min_lanes", 2)),
            gcs_max_lanes=int(getattr(args, "gcs_max_lanes", 5)),
            gcs_num_slots=int(getattr(args, "gcs_num_slots", 5)),
            gcs_min_interval_points=int(getattr(args, "gcs_min_interval_points", 2)),
            gcs_bottom_order_margin_px=float(getattr(args, "gcs_bottom_order_margin_px", 2.0)),
            runtime_ms=args.runtime_ms,
            use_measured_runtime=args.use_measured_runtime,
            warmup=args.warmup,
            device=args.device,
            half=args.half,
            query_decode_defaults=ORDERED_SLOT_QUERY_ONLY_DEFAULTS,
        )
        if active_decode_mode != "ordered_slot":
            warn_max_det_mismatch(args.weights, max_det=query_max_det, context="TuSimple official eval")

    query_conf = float(getattr(args, "conf", 0.25))
    query_point_valid_thr = float(getattr(args, "point_valid_thr", 0.5))
    query_nms_dist_px = float(getattr(args, "nms_dist_px", 18.0))
    query_max_det = int(getattr(args, "max_det", 8))
    query_min_points = int(getattr(args, "min_points", 6))
    query_valid_before_maxdet = bool(getattr(args, "valid_before_maxdet", False))
    query_extent_decode = bool(getattr(args, "extent_decode", False))
    query_extent_decode_mode = str(getattr(args, "extent_decode_mode", "interval") or "interval")
    query_candidate_decode = bool(getattr(args, "candidate_decode", False))
    query_candidate_short_gate = bool(getattr(args, "candidate_short_gate", True))
    query_candidate_gate_valid_thr = float(getattr(args, "candidate_gate_valid_thr", 0.5))
    query_candidate_gate_min_visible = int(getattr(args, "candidate_gate_min_visible", 2))
    query_candidate_gate_max_visible = int(getattr(args, "candidate_gate_max_visible", 10))
    query_candidate_preserve_base_score = bool(getattr(args, "candidate_preserve_base_score", True))
    query_count_aware_topk = bool(getattr(args, "count_aware_topk", False))
    query_count_aware_min_k = int(getattr(args, "count_aware_min_k", 3))
    query_count_aware_max_k = int(getattr(args, "count_aware_max_k", 5))
    query_count_aware_length_norm = float(getattr(args, "count_aware_length_norm", 12.0))
    query_count_aware_extra_margin = int(getattr(args, "count_aware_extra_margin", 0))
    query_count_mode = str(getattr(args, "count_mode", "score_sum"))
    query_oracle_count = bool(getattr(args, "oracle_count", False))
    if query_oracle_count:
        query_count_aware_topk = True
        query_count_mode = "oracle_gt"
    query_extent_decode, query_extent_decode_mode = _normalize_extent_decode_args(
        query_extent_decode,
        query_extent_decode_mode,
    )

    save_dir = resolve_save_dir(
        args.save_dir,
        args.weights,
        args.split,
        query_conf,
        query_point_valid_thr,
        query_nms_dist_px,
        query_max_det,
        query_min_points,
        count_aware_extra_margin=query_count_aware_extra_margin,
        count_aware_topk=query_count_aware_topk,
        valid_before_maxdet=query_valid_before_maxdet,
        extent_decode=query_extent_decode,
        extent_decode_mode=query_extent_decode_mode,
        candidate_decode=query_candidate_decode,
        candidate_short_gate=query_candidate_short_gate,
        oracle_count=query_oracle_count,
        decode_mode=active_decode_mode,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    if not pred_json:
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

    config = {
        "weights": None if pred_json else str(Path(args.weights).resolve()),
        "pred_json": str(Path(pred_json).resolve()) if pred_json else str((save_dir / "tusimple_predictions.json").resolve()),
        "archive_root": str(archive_root.resolve()),
        "split": args.split,
        "gt_json": str(gt_path.resolve()),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "decode_mode": str(active_decode_mode),
        "runtime_ms": float(args.runtime_ms),
        "use_measured_runtime": bool(args.use_measured_runtime),
        "max_images": int(args.max_images),
        "device": str(args.device),
        "half": bool(args.half),
        "save_dir": str(save_dir.resolve()),
        "score_fp_weight": float(args.score_fp_weight),
        "score_fn_weight": float(args.score_fn_weight),
        **gt_contract,
    }
    if str(active_decode_mode) == "ordered_slot":
        ordered_slot_runtime_cfg = ordered_slot_decode_runtime_config(context="official_eval")
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
                "conf": query_conf,
                "point_valid_thr": query_point_valid_thr,
                "nms_dist_px": query_nms_dist_px,
                "max_det": query_max_det,
                "min_points": query_min_points,
                "valid_before_maxdet": query_valid_before_maxdet,
                "extent_decode": query_extent_decode,
                "extent_decode_mode": query_extent_decode_mode,
                "candidate_decode": query_candidate_decode,
                "candidate_short_gate": query_candidate_short_gate,
                "candidate_gate_valid_thr": query_candidate_gate_valid_thr,
                "candidate_gate_min_visible": query_candidate_gate_min_visible,
                "candidate_gate_max_visible": query_candidate_gate_max_visible,
                "candidate_preserve_base_score": query_candidate_preserve_base_score,
                "count_aware_topk": query_count_aware_topk,
                "count_aware_min_k": query_count_aware_min_k,
                "count_aware_max_k": query_count_aware_max_k,
                "count_aware_length_norm": query_count_aware_length_norm,
                "count_aware_extra_margin": query_count_aware_extra_margin,
                "count_mode": query_count_mode,
                "oracle_count": query_oracle_count,
                "uses_gt_count_for_decode": query_oracle_count,
                "diagnostic_only": query_oracle_count,
            }
        )

    output = {
        "metrics": metrics,
        "config": config,
        "timing": timing,
        **gt_contract,
    }
    if str(active_decode_mode) == "ordered_slot":
        output["effective_decode"] = config["effective_decode"]
        output["query_decode_args"] = "not_applicable"
        output.update(
            ordered_slot_order_diagnostics_summary(
                pred_json_mode=bool(pred_json),
                decode_stats=ordered_slot_order_stats,
            )
        )
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
