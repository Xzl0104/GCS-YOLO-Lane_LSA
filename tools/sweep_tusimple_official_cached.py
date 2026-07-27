from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

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
    stable_gt_content_hash,
    stable_raw_file_hash,
    tusimple_image_path,
)
from gcs_tools.official_selection import sweep_selection_policy  # noqa: E402
from gcs_tools.tusimple_split_guard import reject_tusimple_test_search_gt_json  # noqa: E402
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from tools.sweep_tusimple_official import (  # noqa: E402
    DEFAULT_ARCHIVE,
    DEFAULT_WEIGHTS,
    ORDERED_SLOT_QUERY_ONLY_DEFAULTS,
    build_combos,
    resolve_save_dir as resolve_base_save_dir,
    select_best,
    validate_search_split,
    write_csv,
)
from ultralytics.models.gcs.decode_ordered_slot import decode_ordered_slot_predictions  # noqa: E402
from ultralytics.models.gcs.decode_summary import (  # noqa: E402
    ORDERED_SLOT_DECODE_SCHEMA,
    build_ordered_slot_decode_summary,
    load_decode_yaml,
    ordered_slot_decode_params,
    ordered_slot_decode_runtime_config,
    ordered_slot_order_diagnostics_summary,
    raise_for_ordered_slot_query_args,
    validate_decode_yaml_for_model,
)
from ultralytics.models.gcs.mode_utils import resolve_decode_mode  # noqa: E402
from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


CACHE_SCHEMA = "gcs_tusimple_official_prediction_cache_v1"
CACHE_FILE = "predictions.pt"
QUERY_ROW_KEYS = (
    "conf",
    "point_valid_thr",
    "nms_dist_px",
    "max_det",
    "min_points",
    "query_points_source",
    "candidate_decode",
    "extent_decode",
    "extent_decode_mode",
    "extent_decode_priority",
    "count_aware_topk",
    "count_aware_min_k",
    "count_aware_max_k",
    "count_aware_length_norm",
    "count_aware_extra_margin",
    "count_mode",
)
PREDICTION_KEYS = (
    "pred_points",
    "pred_logits",
    "pred_coarse_points",
    "pred_short_refined_points",
    "pred_short_refine_delta_logits",
    "pred_short_refine_delta_norm",
    "pred_quality_logits",
    "pred_valid_logits",
    "pred_count_logits",
    "pred_start_logits",
    "pred_end_logits",
    "pred_short_candidate_points",
    "pred_short_candidate_logits",
    "pred_exist_logits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cached TuSimple official threshold sweep for GCS-YOLO-Lane."
    )
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
    parser.add_argument(
        "--query-points-source",
        choices=("main", "short_refined"),
        default="main",
        help="Query geometry source. 'main' is the default official path; 'short_refined' is a prediction-only v3 ablation.",
    )
    parser.add_argument(
        "--candidate-decode",
        action="store_true",
        help="Flatten fixed lateral candidate hypotheses into the query decode pool.",
    )
    parser.add_argument("--decode-yaml", default=None, help="Schema-validated official_best_decode.yaml to reproduce a decode.")
    parser.add_argument("--gcs-min-lanes", type=int, default=2, help="ordered_slot minimum supported lane count.")
    parser.add_argument("--gcs-max-lanes", type=int, default=5, help="ordered_slot maximum supported lane count.")
    parser.add_argument("--gcs-num-slots", type=int, default=5, help="ordered_slot slot count.")
    parser.add_argument("--gcs-min-interval-points", type=int, default=2, help="ordered_slot minimum decoded start/end interval length.")
    parser.add_argument("--gcs-bottom-order-margin-px", type=float, default=2.0, help="ordered_slot bottom-x order margin in pixels.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="GCS inference shape as H W. Defaults: TuSimple 544 960.")
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
    parser.add_argument(
        "--extent-decode-modes",
        nargs="+",
        choices=("none", "interval", "intersect"),
        default=["none"],
        help="Query extent visibility modes to sweep. 'none' preserves point-valid decode.",
    )
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
    parser.add_argument("--warmup", type=int, default=20, help="Number of untimed warmup forwards when building cache.")
    parser.add_argument("--device", default="0", help="Inference device used only when building cache, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA when building cache.")
    parser.add_argument("--runtime-ms", type=float, default=1.0, help="Constant TuSimple run_time in ms.")
    parser.add_argument("--save-dir", default=None, help="Output directory. Defaults under the checkpoint run directory.")
    parser.add_argument("--cache-dir", default=None, help="Prediction cache directory. Defaults under the checkpoint run directory.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Regenerate the prediction cache before sweeping.")
    parser.add_argument("--cache-only", action="store_true", help="Build or validate the cache, then exit without sweeping.")
    parser.add_argument("--sweep-only", action="store_true", help="Require an existing valid cache; do not run model inference.")
    parser.add_argument(
        "--allow-cache-weight-mismatch",
        action="store_true",
        help="Allow reusing a cache whose recorded checkpoint path/mtime/size differs from --weights.",
    )
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


def resolve_cache_dir(cache_dir: str | Path | None, weights: str | Path, split: str) -> Path:
    if cache_dir is not None and str(cache_dir).strip():
        return Path(cache_dir)
    run_dir = _weight_run_dir(weights)
    if run_dir is not None:
        return run_dir / f"official_pred_cache_{split}"
    return ROOT / "runs" / "gcs_lane" / "tusimple_official_prediction_cache" / Path(weights).stem / str(split)


def resolve_cached_save_dir(
    args: argparse.Namespace,
    effective_margins: list[int],
    effective_extent_decode_modes: list[str],
    decode_mode: str,
    query_points_source: str = "main",
    candidate_decode: bool = False,
) -> Path:
    if args.save_dir is not None and str(args.save_dir).strip():
        return Path(args.save_dir)
    base = resolve_base_save_dir(
        None,
        args.weights,
        args.split,
        count_aware_topk=bool(getattr(args, "count_aware_topk", False)),
        count_aware_extra_margins=effective_margins,
        valid_before_maxdet=bool(getattr(args, "valid_before_maxdet", False)),
        extent_decode_modes=effective_extent_decode_modes,
        decode_mode=decode_mode,
    )
    suffix = "_cached"
    if str(query_points_source) == "short_refined":
        suffix += "_points_short_refined"
    if bool(candidate_decode):
        suffix += "_candidate"
    return base.with_name(f"{base.name}{suffix}")


def _normalize_decode_mode_name(decode_mode: str | None) -> str:
    mode = str(decode_mode or "").strip().lower()
    if mode in {"ordered-slot", "orderedslot"}:
        mode = "ordered_slot"
    return mode


def _ordered_slot_runtime_context(args: argparse.Namespace) -> str:
    return str(getattr(args, "ordered_slot_runtime_context", "official_sweep") or "official_sweep")


def _weights_fingerprint(weights: str | Path) -> dict[str, Any]:
    path = Path(weights)
    out: dict[str, Any] = {"path": str(path.resolve())}
    if path.exists():
        stat = path.stat()
        out.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    else:
        out.update({"size": None, "mtime_ns": None})
    return out


def _patch_legacy_gcs_head_attrs(model: torch.nn.Module) -> int:
    """Patch pre-gcs_mode checkpoints so they keep their historical query behavior."""
    patched = 0
    for module in model.modules():
        if not isinstance(module, GCSLaneHead) or hasattr(module, "gcs_mode"):
            continue
        if all(hasattr(module, name) for name in ("count_mlp", "start_mlp", "end_mlp")):
            module.gcs_mode = "ordered_slot"
        elif all(hasattr(module, name) for name in ("count_head", "start_head", "end_head")):
            module.gcs_mode = "ordered_slot"
        else:
            module.gcs_mode = "query"
        patched += 1
    return patched


def _apply_query_decode_yaml(args: argparse.Namespace, decode_yaml_cfg: dict[str, Any]) -> None:
    args.confs = [float(decode_yaml_cfg["conf"])]
    args.point_valid_thrs = [float(decode_yaml_cfg["point_valid_thr"])]
    args.nms_dist_pxs = [float(decode_yaml_cfg["nms_dist_px"])]
    args.max_dets = [int(decode_yaml_cfg["max_det"])]
    args.min_points = [int(decode_yaml_cfg["min_points"])]
    args.valid_before_maxdet = bool(decode_yaml_cfg.get("valid_before_maxdet", False))
    extent_mode = str(decode_yaml_cfg.get("extent_decode_mode", "none") or "none")
    args.extent_decode_modes = [extent_mode if bool(decode_yaml_cfg.get("extent_decode", False)) else "none"]
    args.count_aware_topk = bool(decode_yaml_cfg["count_aware_topk"])
    args.count_aware_min_k = int(decode_yaml_cfg["count_aware_min_k"])
    args.count_aware_max_k = int(decode_yaml_cfg["count_aware_max_k"])
    args.count_aware_length_norm = float(decode_yaml_cfg["count_aware_length_norm"])
    args.count_aware_extra_margins = [int(decode_yaml_cfg.get("count_aware_extra_margin", 0))]
    args.count_modes = [str(decode_yaml_cfg.get("count_mode", "score_sum"))]


def _load_decode_yaml_for_sweep(args: argparse.Namespace) -> dict[str, Any] | None:
    if not getattr(args, "decode_yaml", None):
        return None
    _, decode_yaml_cfg = load_decode_yaml(args.decode_yaml)
    yaml_mode = _normalize_decode_mode_name(decode_yaml_cfg.get("decode_mode"))
    requested = _normalize_decode_mode_name(getattr(args, "decode_mode", "auto"))
    if requested not in {"auto", yaml_mode}:
        raise RuntimeError(f"decode yaml mismatch: decode_mode={yaml_mode!r}, requested={requested!r}.")
    args.decode_mode = yaml_mode
    if yaml_mode == "query":
        validate_decode_yaml_for_model(decode_yaml_cfg, model_mode="query")
        _apply_query_decode_yaml(args, decode_yaml_cfg)
    elif yaml_mode == "ordered_slot":
        validate_decode_yaml_for_model(decode_yaml_cfg, model_mode="ordered_slot")
    else:
        raise RuntimeError(f"Unsupported decode yaml mode: {yaml_mode!r}.")
    return decode_yaml_cfg


def _cache_manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def _cache_prediction_path(cache_dir: Path) -> Path:
    return cache_dir / CACHE_FILE


def _read_manifest(cache_dir: Path) -> dict[str, Any] | None:
    path = _cache_manifest_path(cache_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cache_mismatch_reasons(
    manifest: dict[str, Any] | None,
    *,
    args: argparse.Namespace,
    cache_dir: Path,
    archive_root: Path,
    gt_records: list[dict],
    gt_path: Path,
    imgsz: tuple[int, int],
    decode_mode: str | None,
    require_query_extent_logits: bool = False,
    require_query_short_refined_points: bool = False,
    require_query_short_candidate_outputs: bool = False,
) -> list[str]:
    if manifest is None:
        return ["missing manifest"]
    reasons: list[str] = []
    if manifest.get("schema") != CACHE_SCHEMA:
        reasons.append(f"schema {manifest.get('schema')!r} != {CACHE_SCHEMA!r}")
    if not _cache_prediction_path(cache_dir).exists():
        reasons.append("missing prediction file")
    if str(Path(str(manifest.get("archive_root", ""))).resolve()) != str(archive_root.resolve()):
        reasons.append("archive_root path differs")
    if str(manifest.get("split")) != str(args.split):
        reasons.append(f"split {manifest.get('split')!r} != {args.split!r}")
    if [int(x) for x in manifest.get("imgsz", [])] != [int(imgsz[0]), int(imgsz[1])]:
        reasons.append(f"imgsz {manifest.get('imgsz')} != {[int(imgsz[0]), int(imgsz[1])]}")
    if int(manifest.get("max_images", 0) or 0) != int(args.max_images or 0):
        reasons.append(f"max_images {manifest.get('max_images')} != {int(args.max_images or 0)}")
    if int(manifest.get("gt_images", -1)) != int(len(gt_records)):
        reasons.append(f"gt_images {manifest.get('gt_images')} != {len(gt_records)}")
    if str(Path(str(manifest.get("gt_json", ""))).resolve()) != str(gt_path.resolve()):
        reasons.append("gt_json path differs")
    if manifest.get("gt_raw_file_sha256") != stable_raw_file_hash(gt_records):
        reasons.append("gt raw_file hash differs")
    if manifest.get("gt_content_sha256") != stable_gt_content_hash(gt_records):
        reasons.append("gt content hash differs")
    if bool(manifest.get("half", False)) != bool(args.half):
        reasons.append(f"half {manifest.get('half')} != {bool(args.half)}")
    if decode_mode and decode_mode != "auto" and manifest.get("decode_mode") != decode_mode:
        reasons.append(f"decode_mode {manifest.get('decode_mode')!r} != {decode_mode!r}")
    if bool(require_query_extent_logits):
        prediction_keys = set(manifest.get("prediction_keys") or [])
        missing_extent = sorted({"pred_start_logits", "pred_end_logits"} - prediction_keys)
        if missing_extent:
            reasons.append(f"prediction cache missing query extent logits: {missing_extent}")
    if bool(require_query_short_refined_points):
        prediction_keys = set(manifest.get("prediction_keys") or [])
        if "pred_short_refined_points" not in prediction_keys:
            reasons.append("prediction cache missing pred_short_refined_points")
    if bool(require_query_short_candidate_outputs):
        prediction_keys = set(manifest.get("prediction_keys") or [])
        missing_candidates = sorted(
            {"pred_short_candidate_points", "pred_short_candidate_logits"} - prediction_keys
        )
        if missing_candidates:
            reasons.append(f"prediction cache missing short candidate outputs: {missing_candidates}")
    recorded_weights = manifest.get("weights", {})
    current_weights = _weights_fingerprint(args.weights)
    if not bool(getattr(args, "allow_cache_weight_mismatch", False)):
        for key in ("path", "size", "mtime_ns"):
            if recorded_weights.get(key) != current_weights.get(key):
                reasons.append(f"weights {key} differs")
                break
    return reasons


def _requires_query_extent_logits(args: argparse.Namespace, decode_yaml_cfg: dict[str, Any] | None) -> bool:
    """Return whether the requested query sweep needs cached start/end extent logits."""
    if decode_yaml_cfg is not None:
        return bool(decode_yaml_cfg.get("extent_decode", False)) and str(
            decode_yaml_cfg.get("extent_decode_mode", "none") or "none"
        ).strip().lower() != "none"
    modes = {str(x or "none").strip().lower() for x in getattr(args, "extent_decode_modes", ["none"])}
    return bool(modes - {"none", "off", "false", "0"})


def _normalize_query_points_source(points_source: str | None) -> str:
    source = str(points_source or "main").strip().lower()
    if source not in {"main", "short_refined"}:
        raise ValueError(f"Unsupported query_points_source={points_source!r}; use 'main' or 'short_refined'.")
    return source


def _requires_query_short_refined_points(args: argparse.Namespace) -> bool:
    return _normalize_query_points_source(getattr(args, "query_points_source", "main")) == "short_refined"


def _requires_query_short_candidate_outputs(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "candidate_decode", False))


def _tensor_for_cache(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.detach()
    if out.ndim > 0 and int(out.shape[0]) == 1:
        out = out[0]
    return out.float().cpu().contiguous()


@torch.inference_mode()
def build_prediction_cache(
    *,
    args: argparse.Namespace,
    cache_dir: Path,
    archive_root: Path,
    gt_path: Path,
    gt_records: list[dict],
    gt_contract: dict[str, Any],
    imgsz: tuple[int, int],
    decode_yaml_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    device_obj = select_device(args.device)
    model = load_gcs_model(args.weights, device=device_obj, half=args.half, gcs_imgsz=imgsz)
    legacy_head_patches = _patch_legacy_gcs_head_attrs(model)
    model_mode = resolve_decode_mode(getattr(args, "decode_mode", "auto"), model)
    if decode_yaml_cfg is not None:
        validate_decode_yaml_for_model(decode_yaml_cfg, model_mode=model_mode)
    args.decode_mode = model_mode

    if args.warmup > 0 and gt_records:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device_obj, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    entries: list[dict[str, Any]] = []
    infer_time_s = 0.0
    io_time_s = 0.0
    for idx, record in enumerate(gt_records):
        raw_file = str(record["raw_file"])
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        io0 = time.perf_counter()
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        io_time_s += time.perf_counter() - io0
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        original_shape = [int(img.shape[0]), int(img.shape[1])]
        tensor = preprocess_image(img, imgsz, device=device_obj, half=args.half)

        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        t1 = time.perf_counter()
        infer_time_s += t1 - t0

        cached_preds = {}
        for key in PREDICTION_KEYS:
            value = preds.get(key) if isinstance(preds, dict) else None
            if isinstance(value, torch.Tensor):
                cached_preds[key] = _tensor_for_cache(value)
        entries.append(
            {
                "index": int(idx),
                "raw_file": raw_file,
                "h_samples": [int(x) for x in record["h_samples"]],
                "image_shape": original_shape,
                "predictions": cached_preds,
            }
        )

    pred_tmp = cache_dir / f"{CACHE_FILE}.tmp"
    torch.save(entries, pred_tmp)
    pred_tmp.replace(_cache_prediction_path(cache_dir))

    n = max(len(gt_records), 1)
    manifest = {
        "schema": CACHE_SCHEMA,
        "cache_dir": str(cache_dir.resolve()),
        "prediction_file": CACHE_FILE,
        "created_unix_time": time.time(),
        "weights": _weights_fingerprint(args.weights),
        "archive_root": str(archive_root.resolve()),
        "split": str(args.split),
        "gt_json": str(gt_path.resolve()),
        "gt_images": int(len(gt_records)),
        "gt_raw_file_sha256": stable_raw_file_hash(gt_records),
        "gt_content_sha256": stable_gt_content_hash(gt_records),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "decode_mode": str(model_mode),
        "max_images": int(args.max_images or 0),
        "half": bool(args.half),
        "stored_tensor_dtype": "float32",
        "legacy_gcs_head_attr_patches": int(legacy_head_patches),
        "prediction_keys": sorted({k for item in entries for k in item["predictions"]}),
        "timing": {
            "avg_image_read_ms": round(io_time_s * 1000.0 / n, 4),
            "avg_inference_ms": round(infer_time_s * 1000.0 / n, 4),
        },
        **gt_contract,
    }
    manifest_tmp = cache_dir / "manifest.tmp.json"
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_tmp.replace(_cache_manifest_path(cache_dir))
    return manifest


def load_prediction_cache(cache_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_manifest(cache_dir)
    if manifest is None:
        raise FileNotFoundError(f"Missing prediction cache manifest: {_cache_manifest_path(cache_dir)}")
    pred_path = _cache_prediction_path(cache_dir)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction cache tensor file: {pred_path}")
    entries = torch.load(pred_path, map_location="cpu")
    if not isinstance(entries, list):
        raise RuntimeError(f"Prediction cache must contain a list, got {type(entries).__name__}.")
    return manifest, entries


def validate_prediction_cache_entries(entries: list[dict[str, Any]], gt_records: list[dict]) -> None:
    """Fail fast if a cache tensor file was swapped under a valid manifest."""
    if len(entries) != len(gt_records):
        raise RuntimeError(f"Prediction cache entry count {len(entries)} != GT records {len(gt_records)}.")
    for idx, (entry, record) in enumerate(zip(entries, gt_records)):
        cached_raw = str(entry.get("raw_file", ""))
        expected_raw = str(record.get("raw_file", ""))
        if cached_raw != expected_raw:
            raise RuntimeError(f"Prediction cache raw_file mismatch at index {idx}: {cached_raw!r} != {expected_raw!r}.")
        cached_h = [int(x) for x in entry.get("h_samples", [])]
        expected_h = [int(x) for x in record.get("h_samples", [])]
        if cached_h != expected_h:
            raise RuntimeError(f"Prediction cache h_samples mismatch at index {idx} for {expected_raw}.")


def _sigmoid_np(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _longest_contiguous_valid_mask_np(mask: np.ndarray, min_points: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    best_start = 0
    best_len = 0
    start = None
    for i, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            length = i - start
            if length > best_len:
                best_start, best_len = start, length
            start = None
    out = np.zeros(mask.shape, dtype=bool)
    if best_len >= int(min_points):
        out[best_start : best_start + best_len] = True
    return out


def _query_lane_to_tusimple(
    points_norm: np.ndarray,
    valid_mask: np.ndarray | None,
    h_samples: list[int],
    image_shape: tuple[int, int],
) -> list[int] | None:
    lane_points = np.asarray(points_norm, dtype=np.float32)
    if valid_mask is not None:
        lane_points = lane_points[np.asarray(valid_mask, dtype=bool)]
    if lane_points.shape[0] < 2:
        return None

    h, w = int(image_shape[0]), int(image_shape[1])
    points_px = lane_points * np.array([w, h], dtype=np.float32).reshape(1, 2)
    points_px = points_px[np.argsort(points_px[:, 1], kind="stable")]

    ys = points_px[:, 1]
    xs = points_px[:, 0]
    unique_ys, unique_idx = np.unique(np.round(ys, decimals=4), return_index=True)
    order = np.argsort(unique_ys, kind="stable")
    xs = xs[unique_idx][order].astype(np.float32)
    ys = ys[unique_idx][order].astype(np.float32)
    if xs.shape[0] < 2:
        return None

    pred_x = np.interp(np.asarray(h_samples, dtype=np.float32), ys, xs, left=np.nan, right=np.nan)
    lane_xs: list[int] = []
    for x in pred_x:
        if not np.isfinite(x):
            lane_xs.append(-2)
            continue
        xi = int(round(float(x)))
        lane_xs.append(xi if 0 <= xi < w else -2)
    return lane_xs if sum(1 for x in lane_xs if x >= 0) >= 2 else None


class CachedQueryPrediction:
    """In-memory query prediction cache with per-threshold decode precomputation."""

    def __init__(self, item: dict[str, Any]):
        self.raw_file = str(item["raw_file"])
        self.h_samples = [int(x) for x in item["h_samples"]]
        self.image_shape = (int(item["image_shape"][0]), int(item["image_shape"][1]))
        preds = item["predictions"]
        if "pred_points" not in preds or "pred_logits" not in preds:
            raise KeyError(f"Query prediction cache for {self.raw_file} lacks pred_points/pred_logits.")

        points = preds["pred_points"].float().cpu().numpy().astype(np.float32)
        logits = preds["pred_logits"].float().cpu().numpy().astype(np.float32)
        if logits.ndim == 2 and logits.shape[-1] == 1:
            logits = np.squeeze(logits, axis=-1)
        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError(f"pred_points must have Q x K x 2 for {self.raw_file}, got {points.shape}.")
        if logits.ndim != 1 or logits.shape[0] != points.shape[0]:
            raise ValueError(f"pred_logits must have Q for {self.raw_file}, got {logits.shape} vs {points.shape}.")

        quality_logits = None
        pred_quality_logits = preds.get("pred_quality_logits")
        if isinstance(pred_quality_logits, torch.Tensor):
            quality_logits = pred_quality_logits.float().cpu().numpy().astype(np.float32)
            if quality_logits.ndim == 2 and quality_logits.shape[-1] == 1:
                quality_logits = np.squeeze(quality_logits, axis=-1)
            if quality_logits.ndim != 1 or quality_logits.shape[0] != points.shape[0]:
                raise ValueError(
                    f"pred_quality_logits must have Q for {self.raw_file}, got {quality_logits.shape} vs {points.shape}."
                )

        points = np.clip(points, 0.0, 1.0)
        order = np.argsort(-points[:, :, 1], axis=1, kind="stable")
        self.points = np.take_along_axis(points, order[:, :, None], axis=1).astype(np.float32)
        self.candidate_points: np.ndarray | None = None
        self.candidate_logits: np.ndarray | None = None
        self.candidate_scores: np.ndarray | None = None
        self.candidate_selected_indices: np.ndarray | None = None
        pred_short_candidate_points = preds.get("pred_short_candidate_points")
        pred_short_candidate_logits = preds.get("pred_short_candidate_logits")
        if isinstance(pred_short_candidate_points, torch.Tensor) or isinstance(pred_short_candidate_logits, torch.Tensor):
            if not isinstance(pred_short_candidate_points, torch.Tensor) or not isinstance(pred_short_candidate_logits, torch.Tensor):
                raise ValueError(
                    f"Short candidate cache for {self.raw_file} requires both candidate point and score tensors."
                )
            candidate_points = pred_short_candidate_points.float().cpu().numpy().astype(np.float32)
            candidate_logits = pred_short_candidate_logits.float().cpu().numpy().astype(np.float32)
            if candidate_points.ndim != 4 or candidate_points.shape[:1] != points.shape[:1] or candidate_points.shape[2:] != points.shape[1:]:
                raise ValueError(
                    f"pred_short_candidate_points must have Q x H x K x 2 for {self.raw_file}, "
                    f"got {candidate_points.shape} vs Q={points.shape[0]}, K={points.shape[1]}."
                )
            if candidate_logits.shape != candidate_points.shape[:2]:
                raise ValueError(
                    f"pred_short_candidate_logits must have Q x H for {self.raw_file}, "
                    f"got {candidate_logits.shape} vs {candidate_points.shape[:2]}."
                )
            candidate_points = np.clip(candidate_points, 0.0, 1.0)
            candidate_order = np.broadcast_to(
                order[:, None, :, None],
                (candidate_points.shape[0], candidate_points.shape[1], candidate_points.shape[2], 2),
            )
            self.candidate_points = np.take_along_axis(candidate_points, candidate_order, axis=2).astype(np.float32)
            self.candidate_logits = candidate_logits
        self.short_refined_points: np.ndarray | None = None
        pred_short_refined_points = preds.get("pred_short_refined_points")
        if isinstance(pred_short_refined_points, torch.Tensor):
            short_refined_points = pred_short_refined_points.float().cpu().numpy().astype(np.float32)
            if tuple(short_refined_points.shape) != tuple(points.shape):
                raise ValueError(
                    f"pred_short_refined_points shape mismatch for {self.raw_file}: "
                    f"{short_refined_points.shape} vs {points.shape}."
                )
            short_refined_points = np.clip(short_refined_points, 0.0, 1.0)
            if not np.allclose(short_refined_points[:, :, 1], points[:, :, 1], atol=1e-4, rtol=0.0):
                raise ValueError(
                    f"pred_short_refined_points must preserve fixed-y coordinates for {self.raw_file}."
                )
            self.short_refined_points = np.take_along_axis(
                short_refined_points,
                order[:, :, None],
                axis=1,
            ).astype(np.float32)
        self.score_source = "quality_logits" if quality_logits is not None else "pred_logits"
        base_score_logits = quality_logits if quality_logits is not None else logits
        self.scores = _sigmoid_np(base_score_logits)
        self.query_indices = np.arange(self.points.shape[0], dtype=np.int64)
        self.k = int(self.points.shape[1])
        if self.candidate_points is not None and self.candidate_logits is not None:
            self.candidate_scores = _sigmoid_np(
                np.asarray(base_score_logits[:, None] + self.candidate_logits, dtype=np.float32)
            )
            self.candidate_selected_indices = np.argmax(self.candidate_logits, axis=1).astype(np.int64)

        pred_valid = preds.get("pred_valid_logits")
        self.valid_scores: np.ndarray | None = None
        if isinstance(pred_valid, torch.Tensor):
            valid_logits = pred_valid.float().cpu().numpy().astype(np.float32)
            if valid_logits.ndim == 3 and valid_logits.shape[-1] == 1:
                valid_logits = np.squeeze(valid_logits, axis=-1)
            if tuple(valid_logits.shape) != tuple(self.points.shape[:2]):
                raise ValueError(f"pred_valid_logits shape mismatch for {self.raw_file}: {valid_logits.shape} vs {self.points.shape[:2]}.")
            valid_scores = _sigmoid_np(valid_logits)
            self.valid_scores = np.take_along_axis(valid_scores, order, axis=1).astype(np.float32)

        pred_start_logits = preds.get("pred_start_logits")
        pred_end_logits = preds.get("pred_end_logits")
        self.extent_masks: np.ndarray | None = None
        if isinstance(pred_start_logits, torch.Tensor) or isinstance(pred_end_logits, torch.Tensor):
            if not isinstance(pred_start_logits, torch.Tensor) or not isinstance(pred_end_logits, torch.Tensor):
                raise ValueError(f"Query extent cache for {self.raw_file} requires both pred_start_logits and pred_end_logits.")
            start_logits = pred_start_logits.float().cpu().numpy().astype(np.float32)
            end_logits = pred_end_logits.float().cpu().numpy().astype(np.float32)
            if tuple(start_logits.shape) != tuple(points.shape[:2]):
                raise ValueError(f"pred_start_logits shape mismatch for {self.raw_file}: {start_logits.shape} vs {points.shape[:2]}.")
            if tuple(end_logits.shape) != tuple(points.shape[:2]):
                raise ValueError(f"pred_end_logits shape mismatch for {self.raw_file}: {end_logits.shape} vs {points.shape[:2]}.")
            start_idx = np.argmax(start_logits, axis=1).astype(np.int64)
            end_idx = np.argmax(end_logits, axis=1).astype(np.int64)
            lo = np.minimum(start_idx, end_idx)
            hi = np.maximum(start_idx, end_idx)
            anchor_idx = np.arange(points.shape[1], dtype=np.int64).reshape(1, -1)
            extent_masks = (anchor_idx >= lo.reshape(-1, 1)) & (anchor_idx <= hi.reshape(-1, 1))
            self.extent_masks = np.take_along_axis(extent_masks, order, axis=1).astype(bool)

        pred_count_logits = preds.get("pred_count_logits")
        self.pred_count_logits: np.ndarray | None = None
        if isinstance(pred_count_logits, torch.Tensor):
            self.pred_count_logits = pred_count_logits.float().cpu().numpy().astype(np.float32).reshape(-1)
        self._valid_cache: dict[tuple[float, int, float, str, str, bool], dict[str, Any]] = {}

    def _points_for_source(self, points_source: str | None, candidate_decode: bool = False) -> np.ndarray:
        if bool(candidate_decode):
            if _normalize_query_points_source(points_source) != "main":
                raise ValueError("candidate_decode supports only query_points_source='main'.")
            if self.candidate_points is None:
                raise ValueError(
                    f"candidate_decode requires pred_short_candidate_points in the cache for {self.raw_file}."
                )
            return self.candidate_points[
                np.arange(self.candidate_points.shape[0], dtype=np.int64),
                self.candidate_selected_indices,
            ]
        source = _normalize_query_points_source(points_source)
        if source == "main":
            return self.points
        if self.short_refined_points is None:
            raise ValueError(
                f"query_points_source='short_refined' requires pred_short_refined_points in the cache "
                f"for {self.raw_file}."
            )
        return self.short_refined_points

    @staticmethod
    def _normalize_extent_mode(extent_mode: str | None) -> str:
        mode = str(extent_mode or "none").strip().lower()
        if mode in {"none", "off", "false", "0"}:
            return "none"
        if mode not in {"interval", "intersect"}:
            raise ValueError(f"Unsupported extent_decode_mode={extent_mode!r}; use 'none', 'interval', or 'intersect'.")
        return mode

    def _valid_context(
        self,
        point_valid_thr: float,
        min_points: int,
        length_norm: float,
        extent_mode: str | None = "none",
        points_source: str = "main",
        candidate_decode: bool = False,
    ) -> dict[str, Any]:
        extent_mode = self._normalize_extent_mode(extent_mode)
        points_source = _normalize_query_points_source(points_source)
        if bool(candidate_decode) and extent_mode != "none":
            raise ValueError("candidate_decode cannot be combined with extent_decode.")
        points = self._points_for_source(points_source, candidate_decode=candidate_decode)
        valid_scores = self.valid_scores
        scores = self.scores
        if bool(candidate_decode):
            if self.candidate_scores is None:
                raise ValueError(f"candidate_decode requires candidate logits in the cache for {self.raw_file}.")
            scores = self.candidate_scores[
                np.arange(self.candidate_scores.shape[0], dtype=np.int64),
                self.candidate_selected_indices,
            ]
        key = (
            round(float(point_valid_thr), 12),
            int(min_points),
            float(length_norm),
            extent_mode,
            points_source,
            bool(candidate_decode),
        )
        if key in self._valid_cache:
            return self._valid_cache[key]

        uses_visibility_mask = False
        if int(min_points) > self.k:
            masks = np.zeros((points.shape[0], self.k), dtype=bool)
            uses_visibility_mask = True
        elif extent_mode != "none":
            if self.extent_masks is None:
                raise ValueError("extent decode requires pred_start_logits and pred_end_logits in the prediction cache.")
            masks = self.extent_masks.copy()
            uses_visibility_mask = True
            if extent_mode == "intersect":
                if self.valid_scores is None:
                    raise ValueError("extent_decode_mode='intersect' requires pred_valid_logits in the prediction cache.")
                valid_runs = np.stack(
                    [
                        _longest_contiguous_valid_mask_np(row >= float(point_valid_thr), min_points=int(min_points))
                        for row in self.valid_scores
                    ],
                    axis=0,
                )
                masks = masks & valid_runs
        elif valid_scores is None:
            masks = np.ones((points.shape[0], self.k), dtype=bool)
        else:
            masks = np.stack(
                [
                    _longest_contiguous_valid_mask_np(row >= float(point_valid_thr), min_points=int(min_points))
                    for row in valid_scores
                ],
                axis=0,
            )
            uses_visibility_mask = True
        valid_counts = masks.sum(axis=1).astype(np.int32)
        tusimple_lanes = [
            _query_lane_to_tusimple(
                points[i],
                masks[i] if uses_visibility_mask else None,
                self.h_samples,
                self.image_shape,
            )
            for i in range(points.shape[0])
        ]

        qualities = np.zeros((points.shape[0],), dtype=np.float32)
        for i in range(points.shape[0]):
            visible_count = int(valid_counts[i]) if uses_visibility_mask else int(self.k)
            mean_valid = 1.0
            if valid_scores is not None and uses_visibility_mask:
                mean_valid = float(valid_scores[i][masks[i]].mean()) if visible_count > 0 else 0.0
            length_factor = min(float(visible_count) / float(length_norm), 1.0)
            qualities[i] = float(scores[i] * mean_valid * length_factor)

        context = {
            "masks": masks,
            "valid_counts": valid_counts,
            "tusimple_lanes": tusimple_lanes,
            "qualities": qualities,
            "uses_visibility_mask": uses_visibility_mask,
        }
        self._valid_cache[key] = context
        return context

    def count_aware_k(self, combo: dict[str, Any]) -> int | None:
        if not bool(combo.get("count_aware_topk", False)):
            return None
        if bool(combo.get("candidate_decode", False)):
            raise ValueError("candidate_decode cannot be combined with count_aware_topk.")
        min_k = int(combo.get("count_aware_min_k", 3))
        max_k = int(combo.get("count_aware_max_k", 5))
        length_norm = float(combo.get("count_aware_length_norm", 12.0))
        extra_margin = int(combo.get("count_aware_extra_margin", 0))
        if min_k < 0 or max_k < 0 or min_k > max_k:
            raise ValueError(f"count-aware k bounds must satisfy 0 <= min_k <= max_k, got {min_k}/{max_k}.")
        if length_norm <= 0.0:
            raise ValueError(f"count-aware length norm must be > 0, got {length_norm}.")
        if extra_margin < 0:
            raise ValueError(f"count-aware extra margin must be >= 0, got {extra_margin}.")

        count_mode = str(combo.get("count_mode", "score_sum") or "score_sum")
        if count_mode == "score_sum":
            base_k = max(min_k, min(max_k, int(round(float(self.scores.sum())))))
        elif count_mode == "count_logits":
            if self.pred_count_logits is None:
                raise ValueError("count_mode='count_logits' requires pred_count_logits in the cache.")
            if self.pred_count_logits.size != 4:
                raise ValueError(f"pred_count_logits must have 4 classes for 2..5 lanes, got {self.pred_count_logits.size}.")
            base_k = int(np.argmax(self.pred_count_logits)) + 2
        else:
            raise ValueError(f"Unsupported count_mode={count_mode!r}.")
        k_hat = int(base_k) + extra_margin
        max_det = int(combo.get("max_det", 0) or 0)
        if max_det > 0:
            k_hat = min(k_hat, max_det)
        return k_hat

    def _lane_nms(
        self,
        query_ids: np.ndarray,
        scores: np.ndarray,
        masks: np.ndarray | None,
        dist_thr_px: float,
        points: np.ndarray,
    ) -> np.ndarray:
        if float(dist_thr_px) <= 0.0 or query_ids.size <= 1:
            return np.arange(query_ids.size, dtype=np.int64)
        order = np.argsort(-scores, kind="stable")
        keep_positions: list[int] = []
        w = float(self.image_shape[1])
        for pos in order.tolist():
            query_id = int(query_ids[pos])
            duplicate = False
            for kept_pos in keep_positions:
                kept_query = int(query_ids[kept_pos])
                if masks is None:
                    overlap = slice(None)
                    overlap_count = self.k
                else:
                    overlap_mask = masks[query_id] & masks[kept_query]
                    overlap_count = int(overlap_mask.sum())
                    overlap = overlap_mask
                if overlap_count < 2:
                    dist = float("inf")
                else:
                    dist = float(np.mean(np.abs(points[query_id, overlap, 0] - points[kept_query, overlap, 0]) * w))
                if dist <= float(dist_thr_px):
                    duplicate = True
                    break
            if not duplicate:
                keep_positions.append(int(pos))
        return np.asarray(keep_positions, dtype=np.int64)

    def decode_tusimple_lanes(self, combo: dict[str, Any]) -> list[list[int]]:
        min_points = int(combo["min_points"])
        if min_points > self.k:
            return []
        points_source = _normalize_query_points_source(combo.get("query_points_source", "main"))
        candidate_decode = bool(combo.get("candidate_decode", False))
        if candidate_decode and points_source != "main":
            raise ValueError("candidate_decode supports only query_points_source='main'.")
        if candidate_decode and str(combo.get("extent_decode_mode", "none")) != "none":
            raise ValueError("candidate_decode cannot be combined with extent_decode.")
        points = self._points_for_source(points_source, candidate_decode=candidate_decode)
        context = self._valid_context(
            point_valid_thr=float(combo["point_valid_thr"]),
            min_points=min_points,
            length_norm=float(combo.get("count_aware_length_norm", 12.0)),
            extent_mode=str(combo.get("extent_decode_mode", "none")),
            points_source=points_source,
            candidate_decode=candidate_decode,
        )
        masks = context["masks"]
        valid_counts = context["valid_counts"]
        lanes_by_query = context["tusimple_lanes"]
        qualities = context["qualities"]
        uses_visibility_mask = bool(context["uses_visibility_mask"])

        score_source = (
            self.candidate_scores[
                np.arange(self.candidate_scores.shape[0], dtype=np.int64),
                self.candidate_selected_indices,
            ]
            if candidate_decode and self.candidate_scores is not None
            else self.scores
        )
        if score_source is None:
            raise ValueError(f"candidate_decode requires candidate scores in the cache for {self.raw_file}.")
        query_ids = np.flatnonzero(score_source >= float(combo["conf"])).astype(np.int64)
        if query_ids.size == 0:
            return []
        scores = score_source[query_ids]
        order = np.argsort(-scores, kind="stable")

        if float(combo["nms_dist_px"]) > 0.0:
            sorted_query_ids = query_ids[order]
            sorted_scores = scores[order]
            keep_sorted = self._lane_nms(
                sorted_query_ids,
                sorted_scores,
                masks if uses_visibility_mask else None,
                dist_thr_px=float(combo["nms_dist_px"]),
                points=points,
            )
            query_ids = sorted_query_ids[keep_sorted]
            scores = sorted_scores[keep_sorted]
            order = np.arange(query_ids.size, dtype=np.int64)

        if bool(combo.get("valid_before_maxdet", False)) and uses_visibility_mask:
            keep_valid = valid_counts[query_ids] >= min_points
            query_ids = query_ids[keep_valid]
            scores = scores[keep_valid]
            order = np.argsort(-scores, kind="stable")

        max_det = int(combo["max_det"])
        if max_det > 0:
            order = order[:max_det]
        query_ids = query_ids[order]

        final_queries: list[int] = []
        final_lanes: list[list[int]] = []
        for query_id in query_ids.tolist():
            if uses_visibility_mask and int(valid_counts[query_id]) < min_points:
                continue
            lane = lanes_by_query[query_id]
            if lane is None:
                continue
            final_queries.append(int(query_id))
            final_lanes.append(list(lane))

        k_hat = self.count_aware_k(combo)
        if k_hat is not None and len(final_lanes) > int(k_hat):
            top = sorted(range(len(final_lanes)), key=lambda i: (-float(qualities[final_queries[i]]), i))[: int(k_hat)]
            keep = set(top)
            final_lanes = [lane for i, lane in enumerate(final_lanes) if i in keep]
        return final_lanes


def _gt_lane_count(record: dict) -> int:
    return sum(1 for lane in record.get("lanes", []) if any(float(x) >= 0.0 for x in lane))


def _count_diagnostics_from_pairs(pairs: Counter[tuple[int, int]]) -> dict[str, Any]:
    total = max(sum(pairs.values()), 1)
    correct = sum(count for (gt, pred), count in pairs.items() if gt == pred)
    by_gt: dict[str, float] = {}
    for gt_count in sorted({gt for gt, _ in pairs}):
        rows_total = sum(count for (gt, _), count in pairs.items() if gt == gt_count)
        rows_correct = sum(count for (gt, pred), count in pairs.items() if gt == gt_count and pred == gt_count)
        by_gt[f"count_acc_{gt_count}"] = round(rows_correct / max(rows_total, 1), 6)
    pred_hist = Counter()
    gt_hist = Counter()
    for (gt, pred), count in pairs.items():
        gt_hist[gt] += count
        pred_hist[pred] += count
    return {
        "count_acc": round(correct / total, 6),
        **by_gt,
        "pred_lanes_hist": {str(k): int(v) for k, v in sorted(pred_hist.items())},
        "gt_lanes_hist": {str(k): int(v) for k, v in sorted(gt_hist.items())},
        "count_confusion": {f"{gt}->{pred}": int(v) for (gt, pred), v in sorted(pairs.items())},
    }


def _new_state() -> dict[str, Any]:
    return {"accuracy": 0.0, "fp": 0.0, "fn": 0.0, "images": 0, "pairs": Counter()}


def _update_state(state: dict[str, Any], pred_lanes: list[list[int]], gt_record: dict, runtime_ms: float) -> None:
    acc, fp, fn = TuSimpleOfficialLaneEval.bench(
        pred=[list(x) for x in pred_lanes],
        gt=[list(x) for x in gt_record.get("lanes", [])],
        y_samples=list(gt_record["h_samples"]),
        running_time=float(runtime_ms),
    )
    state["accuracy"] += float(acc)
    state["fp"] += float(fp)
    state["fn"] += float(fn)
    state["images"] += 1
    state["pairs"].update([(_gt_lane_count(gt_record), len(pred_lanes))])


def _state_to_row(combo: dict[str, Any], state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    images = max(int(state["images"]), 1)
    row = {
        **combo,
        "official_acc": round(float(state["accuracy"]) / images, 6),
        "official_FP": round(float(state["fp"]) / images, 6),
        "official_FN": round(float(state["fn"]) / images, 6),
        "images": int(state["images"]),
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
    row.update(_count_diagnostics_from_pairs(state["pairs"]))
    return row


def _combo_key(combo: dict[str, Any]) -> tuple[Any, ...]:
    if combo.get("decode_mode") == "ordered_slot":
        return ("ordered_slot",)
    return (
        float(combo["conf"]),
        float(combo["point_valid_thr"]),
        float(combo["nms_dist_px"]),
        int(combo["max_det"]),
        int(combo["min_points"]),
        str(combo.get("query_points_source", "main")),
        bool(combo.get("candidate_decode", False)),
        str(combo.get("extent_decode_mode", "none")),
        str(combo.get("count_mode", "score_sum")),
        int(combo.get("count_aware_extra_margin", 0)),
    )


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if row.get("decode_mode") == "ordered_slot":
        return ("ordered_slot", 0.0, 0.0, 0, 0)
    return (
        float(row["conf"]),
        float(row["point_valid_thr"]),
        float(row["nms_dist_px"]),
        int(row["max_det"]),
        int(row["min_points"]),
        str(row.get("query_points_source", "main")),
        bool(row.get("candidate_decode", False)),
        str(row.get("extent_decode_mode", "none")),
        str(row.get("count_mode", "score_sum")),
        int(row.get("count_aware_extra_margin", 0)),
    )


def _batched_pred_dict(preds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in preds.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.unsqueeze(0) if value.ndim > 0 else value.reshape(1)
    return out


def cached_sweep_query(
    *,
    entries: list[dict[str, Any]],
    gt_records: list[dict],
    combos: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cached_predictions = [CachedQueryPrediction(item) for item in entries]
    gt_by_raw = {str(record["raw_file"]): record for record in gt_records}
    states = {_combo_key(combo): _new_state() for combo in combos}

    t0 = time.perf_counter()
    for pred in cached_predictions:
        gt_record = gt_by_raw[pred.raw_file]
        for combo in combos:
            pred_lanes = pred.decode_tusimple_lanes(combo)
            _update_state(states[_combo_key(combo)], pred_lanes, gt_record, runtime_ms=float(args.runtime_ms))
    elapsed = time.perf_counter() - t0

    rows = [_state_to_row(combo, states[_combo_key(combo)], args) for combo in combos]
    return sorted(rows, key=_row_sort_key), {"cached_decode_eval_s": elapsed}


def cached_sweep_ordered_slot(
    *,
    entries: list[dict[str, Any]],
    gt_records: list[dict],
    combos: list[dict[str, Any]],
    args: argparse.Namespace,
    decode_yaml_cfg: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if len(combos) != 1:
        raise RuntimeError(f"ordered_slot cached sweep expects one combo, got {len(combos)}.")
    combo = combos[0]
    gt_by_raw = {str(record["raw_file"]): record for record in gt_records}
    state = _new_state()
    order_stats = {"ordered_slot_order_violations": 0, "ordered_slot_order_violation_images": 0}
    ordered_slot_runtime_cfg = ordered_slot_decode_runtime_config(context=_ordered_slot_runtime_context(args))
    ordered_params = ordered_slot_decode_params(args, decode_yaml_cfg)

    t0 = time.perf_counter()
    for item in entries:
        raw_file = str(item["raw_file"])
        preds = _batched_pred_dict(item["predictions"])
        image_shape = (int(item["image_shape"][0]), int(item["image_shape"][1]))
        lanes, order_diag = decode_ordered_slot_predictions(
            preds,
            batch_index=0,
            image_shape=image_shape,
            min_lanes=ordered_params["min_lanes"],
            max_lanes=ordered_params["max_lanes"],
            min_interval_points=ordered_params["min_interval_points"],
            order_margin_px=ordered_params["order_margin_px"],
            img_w=float(image_shape[1]),
            order_check=ordered_slot_runtime_cfg["order_check"],
            output_order=ordered_slot_runtime_cfg["output_order"],
            return_diagnostics=True,
        )
        order_stats["ordered_slot_order_violations"] += int(order_diag["order_violation_count"])
        order_stats["ordered_slot_order_violation_images"] += int(order_diag["has_order_violation"])
        tusimple_lanes = gcs_lanes_to_tusimple_lanes(lanes, item["h_samples"], image_shape=image_shape)
        _update_state(state, tusimple_lanes, gt_by_raw[raw_file], runtime_ms=float(args.runtime_ms))
    elapsed = time.perf_counter() - t0

    row = _state_to_row(combo, state, args)
    row.update(order_stats)
    row["strict_order_valid"] = int(row["ordered_slot_order_violations"]) == 0
    return [row], {"cached_decode_eval_s": elapsed}


def _config_for_summary(
    *,
    args: argparse.Namespace,
    cache_dir: Path,
    manifest: dict[str, Any],
    save_dir: Path,
    archive_root: Path,
    gt_path: Path,
    gt_contract: dict[str, Any],
    imgsz: tuple[int, int],
    decode_mode: str,
    effective_margins: list[int],
    effective_extent_decode_modes: list[str],
    decode_yaml_cfg: dict[str, Any] | None,
    query_points_source: str,
) -> dict[str, Any]:
    config = {
        "weights": str(Path(args.weights).resolve()),
        "archive_root": str(archive_root.resolve()),
        "split": args.split,
        "gt_json": str(gt_path.resolve()),
        "save_dir": str(save_dir.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "cache_schema": CACHE_SCHEMA,
        "cache_prediction_file": str(_cache_prediction_path(cache_dir).resolve()),
        "cache_weights": manifest.get("weights", {}),
        "cache_created_unix_time": manifest.get("created_unix_time"),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "decode_mode": str(decode_mode),
        "query_points_source": str(query_points_source),
        "candidate_decode": bool(getattr(args, "candidate_decode", False)),
        "runtime_ms": float(args.runtime_ms),
        "max_images": int(args.max_images),
        "device": "cache",
        "half": bool(manifest.get("half", False)),
        "best_metric": "official_acc",
        "selection_policy": sweep_selection_policy(),
        "score_fp_weight": float(args.score_fp_weight),
        "score_fn_weight": float(args.score_fn_weight),
        **gt_contract,
    }
    if str(decode_mode) == "ordered_slot":
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
        config.update(
            {
                "schema": ORDERED_SLOT_DECODE_SCHEMA,
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
                "extent_decode_modes": [str(x) for x in effective_extent_decode_modes],
                "count_aware_topk": bool(getattr(args, "count_aware_topk", False)),
                "count_aware_min_k": int(getattr(args, "count_aware_min_k", 3)),
                "count_aware_max_k": int(getattr(args, "count_aware_max_k", 5)),
                "count_aware_length_norm": float(getattr(args, "count_aware_length_norm", 12.0)),
                "count_aware_extra_margins": effective_margins,
                "count_modes": [str(x) for x in sorted({str(x) for x in getattr(args, "count_modes", ["score_sum"])})],
                "query_cache_precompute": "valid_masks_tusimple_lanes_count_aware_quality",
            }
        )
    return config


def sweep(args: argparse.Namespace) -> dict[str, Any]:
    args.split = validate_search_split(args.split)
    reject_tusimple_test_search_gt_json(args.gt_json, context="TuSimple cached official sweep")
    cache_only = bool(getattr(args, "cache_only", False))
    sweep_only = bool(getattr(args, "sweep_only", False))
    rebuild_cache = bool(getattr(args, "rebuild_cache", False))
    if cache_only and sweep_only:
        raise ValueError("--cache-only and --sweep-only are mutually exclusive.")

    decode_yaml_cfg = _load_decode_yaml_for_sweep(args)
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
    cache_dir = resolve_cache_dir(getattr(args, "cache_dir", None), args.weights, args.split)
    require_query_extent_logits = _requires_query_extent_logits(args, decode_yaml_cfg)
    query_points_source = _normalize_query_points_source(getattr(args, "query_points_source", "main"))
    require_query_short_refined_points = _requires_query_short_refined_points(args)
    candidate_decode = bool(getattr(args, "candidate_decode", False))
    require_query_short_candidate_outputs = _requires_query_short_candidate_outputs(args)
    if candidate_decode:
        if query_points_source != "main":
            raise ValueError("candidate_decode supports only query_points_source='main'.")
        if bool(getattr(args, "count_aware_topk", False)):
            raise ValueError("candidate_decode cannot be combined with count-aware top-k.")
        extent_modes = {
            str(x or "none").strip().lower() for x in getattr(args, "extent_decode_modes", ["none"])
        }
        if extent_modes - {"none", "off", "false", "0"}:
            raise ValueError("candidate_decode cannot be combined with extent decode.")

    requested_mode = _normalize_decode_mode_name(getattr(args, "decode_mode", "auto"))
    manifest = None if rebuild_cache else _read_manifest(cache_dir)
    mismatch_reasons = _cache_mismatch_reasons(
        manifest,
        args=args,
        cache_dir=cache_dir,
        archive_root=archive_root,
        gt_records=gt_records,
        gt_path=gt_path,
        imgsz=imgsz,
        decode_mode=None if requested_mode == "auto" else requested_mode,
        require_query_extent_logits=require_query_extent_logits,
        require_query_short_refined_points=require_query_short_refined_points,
        require_query_short_candidate_outputs=require_query_short_candidate_outputs,
    )
    cache_rebuilt = False
    if mismatch_reasons:
        if sweep_only:
            raise RuntimeError(f"Prediction cache is not usable for this sweep: {mismatch_reasons}")
        manifest = build_prediction_cache(
            args=args,
            cache_dir=cache_dir,
            archive_root=archive_root,
            gt_path=gt_path,
            gt_records=gt_records,
            gt_contract=gt_contract,
            imgsz=imgsz,
            decode_yaml_cfg=decode_yaml_cfg,
        )
        cache_rebuilt = True
        mismatch_reasons = _cache_mismatch_reasons(
            manifest,
            args=args,
            cache_dir=cache_dir,
            archive_root=archive_root,
            gt_records=gt_records,
            gt_path=gt_path,
            imgsz=imgsz,
            decode_mode=None if requested_mode == "auto" else requested_mode,
            require_query_extent_logits=require_query_extent_logits,
            require_query_short_refined_points=require_query_short_refined_points,
            require_query_short_candidate_outputs=require_query_short_candidate_outputs,
        )
        if mismatch_reasons:
            raise RuntimeError(f"Prediction cache rebuilt but is still not usable for this sweep: {mismatch_reasons}")
        mismatch_reasons = []
    assert manifest is not None
    args.decode_mode = str(manifest["decode_mode"]) if requested_mode == "auto" else requested_mode
    if str(args.decode_mode) == "ordered_slot" and query_points_source != "main":
        raise ValueError("query_points_source='short_refined' is supported only for query decode.")

    if cache_only:
        print(json.dumps({"cache_dir": str(cache_dir.resolve()), "rebuilt": cache_rebuilt, "manifest": manifest}, indent=2))
        return {"cache_dir": str(cache_dir.resolve()), "rebuilt": cache_rebuilt, "manifest": manifest}

    if str(args.decode_mode) == "ordered_slot":
        raise_for_ordered_slot_query_args(args, ORDERED_SLOT_QUERY_ONLY_DEFAULTS, context="cached TuSimple official sweep")
    combos = build_combos(args, decode_yaml_cfg=decode_yaml_cfg)
    for combo in combos:
        if combo.get("decode_mode") == "query":
            combo["query_points_source"] = query_points_source
            combo["candidate_decode"] = candidate_decode
    if str(args.decode_mode) != "ordered_slot":
        for max_det in sorted({int(c["max_det"]) for c in combos}):
            warn_max_det_mismatch(args.weights, max_det=max_det, context="cached TuSimple official sweep")

    load_t0 = time.perf_counter()
    manifest, entries = load_prediction_cache(cache_dir)
    validate_prediction_cache_entries(entries, gt_records)
    cache_load_s = time.perf_counter() - load_t0

    if str(args.decode_mode) == "ordered_slot":
        rows, timing = cached_sweep_ordered_slot(
            entries=entries,
            gt_records=gt_records,
            combos=combos,
            args=args,
            decode_yaml_cfg=decode_yaml_cfg,
        )
    else:
        rows, timing = cached_sweep_query(entries=entries, gt_records=gt_records, combos=combos, args=args)
    rows = sorted(rows, key=_row_sort_key)
    best = select_best(rows)

    effective_margins = sorted(
        {int(combo.get("count_aware_extra_margin", 0)) for combo in combos if combo.get("decode_mode") == "query"}
    )
    effective_extent_decode_modes = sorted(
        {str(combo.get("extent_decode_mode", "none")) for combo in combos if combo.get("decode_mode") == "query"}
    )
    save_dir = resolve_cached_save_dir(
        args,
        effective_margins,
        effective_extent_decode_modes,
        str(args.decode_mode),
        query_points_source=query_points_source,
        candidate_decode=candidate_decode,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    write_csv(save_dir / "tusimple_official_sweep.csv", rows)

    config = _config_for_summary(
        args=args,
        cache_dir=cache_dir,
        manifest=manifest,
        save_dir=save_dir,
        archive_root=archive_root,
        gt_path=gt_path,
        gt_contract=gt_contract,
        imgsz=imgsz,
        decode_mode=str(args.decode_mode),
        effective_margins=effective_margins,
        effective_extent_decode_modes=effective_extent_decode_modes,
        decode_yaml_cfg=decode_yaml_cfg,
        query_points_source=query_points_source,
    )
    config.update(
        {
            "cache_rebuilt": bool(cache_rebuilt),
            "cache_mismatch_reasons": mismatch_reasons,
        }
    )

    n = max(len(gt_records), 1)
    build_timing = manifest.get("timing", {})
    selection_policy = sweep_selection_policy()
    output = {
        "best": best,
        "results": rows,
        "selection_policy": selection_policy,
        "config": config,
        **gt_contract,
        "timing": {
            "avg_cache_build_inference_ms": build_timing.get("avg_inference_ms"),
            "avg_cache_build_image_read_ms": build_timing.get("avg_image_read_ms"),
            "avg_cache_load_ms": round(cache_load_s * 1000.0 / n, 4),
            "avg_cached_sweep_ms": round(float(timing["cached_decode_eval_s"]) * 1000.0 / n, 4),
            "cache_rebuilt": bool(cache_rebuilt),
        },
    }
    if str(args.decode_mode) == "ordered_slot":
        output["effective_decode"] = config["effective_decode"]
        output["query_decode_args"] = "not_applicable"
        output.update(ordered_slot_order_diagnostics_summary(pred_json_mode=False, decode_stats=best))

    (save_dir / "tusimple_official_sweep_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"cached sweep used cache: {cache_dir.resolve()}")
    print(f"swept {len(rows)} combinations on {len(gt_records)} images")
    print(f"saved to: {save_dir.resolve()}")
    return output


def main() -> None:
    sweep(parse_args())


if __name__ == "__main__":
    main()
