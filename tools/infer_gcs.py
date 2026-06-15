from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.nn.tasks import GCSLaneModel, load_checkpoint
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, assert_gcs_image_tensor, normalize_imgsz, shape_str
from ultralytics.utils.gcs_postprocess import (
    GCS_DEFAULT_MAX_DET,
    decode_gcs_predictions,
    draw_gcs_lanes,
    save_gcs_lanes_txt,
)
from ultralytics.utils.torch_utils import select_device


DEFAULT_WEIGHTS = ROOT / "runs" / "gcs_lane" / "overfit20" / "weights" / "best.pt"


def weight_run_dir(weights: str | Path) -> Path | None:
    """Return the parent run directory for a standard runs/.../weights/best.pt path."""
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def read_run_gcs_eval_max_det(weights: str | Path) -> int | None:
    """Read gcs_eval_max_det from the checkpoint run args.yaml when available."""
    run_dir = weight_run_dir(weights)
    if run_dir is None:
        return None
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return None
    for line in args_path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text.startswith("gcs_eval_max_det:"):
            continue
        value = text.split(":", 1)[1].strip().strip("'\"")
        if value.lower() in {"", "none", "null", "~"}:
            return None
        return int(float(value))
    return None


def warn_max_det_mismatch(weights: str | Path, max_det: int, context: str) -> None:
    """Warn when evaluation keeps a different number of lanes than train-time validation."""
    train_max_det = read_run_gcs_eval_max_det(weights)
    if train_max_det is None or int(max_det) == int(train_max_det):
        return
    print(
        f"WARNING: {context} max_det={int(max_det)} differs from train-time gcs_eval_max_det={train_max_det} "
        f"in {weight_run_dir(weights) / 'args.yaml'}. Keep train-time val, sweeps, and official test on one "
        "max_det policy for comparable FP/FN.",
        file=sys.stderr,
    )


def dataset_defaults(dataset: str) -> dict[str, Path]:
    """Return conventional inference paths for a converted GCS dataset."""
    root = ROOT / "datasets" / ("tusimple_fixed_y_960x544" if dataset.lower() == "tusimple" else dataset.lower())
    return {"source": root / "images" / "val"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GCS-YOLO-Lane inference and structured lane post-processing.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="GCS checkpoint .pt, or a GCS yaml for smoke tests.")
    parser.add_argument("--source", default=None, help="Image file, image directory, or txt list.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="GCS inference shape as H W. Defaults: TuSimple 544 960, CULane 384 960.",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Lane existence confidence threshold.")
    parser.add_argument(
        "--point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold for fixed-y lane decoding.",
    )
    parser.add_argument("--min-points", type=int, default=6, help="Minimum decoded visible anchors required to keep a lane.")
    parser.add_argument("--nms-dist-px", type=float, default=18.0, help="Lane duplicate suppression distance in pixels. 0 disables.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument(
        "--max-det",
        type=int,
        default=GCS_DEFAULT_MAX_DET,
        help="Maximum lane queries to keep after rank-score sorting.",
    )
    count_head_group = parser.add_mutually_exclusive_group()
    count_head_group.add_argument("--use-count-head-decode", dest="use_count_head_decode", action="store_true", help="Use explicit Count Head K for final Top-K lane selection.")
    count_head_group.add_argument(
        "--no-count-head-decode",
        dest="use_count_head_decode",
        action="store_false",
        help="Disable Count Head K for legacy diagnostics and use max-det rank selection.",
    )
    parser.set_defaults(use_count_head_decode=True)
    parser.add_argument("--count-head-temp", type=float, default=1.0, help="Temperature for Count Head count=2/3/4/5 softmax.")
    parser.add_argument("--candidate-conf", type=float, default=0.05, help="Relaxed candidate-pool existence threshold for Count Head Top-K decode.")
    parser.add_argument("--candidate-point-valid-thr", type=float, default=0.20, help="Relaxed candidate-pool point-valid threshold for Count Head Top-K decode.")
    parser.add_argument("--candidate-min-points", type=int, default=5, help="Relaxed candidate-pool visible-anchor floor before final Top-K.")
    rescue_group = parser.add_mutually_exclusive_group()
    rescue_group.add_argument("--enable-rescue-candidate-pool", dest="enable_rescue_candidate_pool", action="store_true", help="Use weaker real-query candidates only when Count Head K exceeds the normal candidate pool.")
    rescue_group.add_argument("--no-enable-rescue-candidate-pool", dest="enable_rescue_candidate_pool", action="store_false", help="Disable the weaker rescue candidate pool.")
    parser.set_defaults(enable_rescue_candidate_pool=True)
    parser.add_argument("--rescue-candidate-conf", type=float, default=0.005, help="Rescue candidate-pool existence threshold for Count Head Top-K decode.")
    parser.add_argument("--rescue-candidate-point-valid-thr", type=float, default=0.08, help="Rescue candidate-pool point-valid threshold for Count Head Top-K decode.")
    parser.add_argument("--rescue-candidate-min-points", type=int, default=4, help="Rescue candidate-pool visible-anchor floor before final Top-K.")
    parser.add_argument("--final-min-points", type=int, default=6, help="Final visible-anchor floor for selected ranks 1-4.")
    parser.add_argument("--fifth-min-points", type=int, default=5, help="Final visible-anchor floor for selected rank 5.")
    parser.add_argument("--line-nms-min-overlap", type=int, default=6, help="Minimum shared visible anchors for lane-NMS duplicate suppression.")
    parser.add_argument("--line-nms-rescue-dist-px", type=float, default=30.0, help="Duplicate distance used when rescuing lanes from pre-NMS candidates.")
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument("--quality-rescue-5th", dest="quality_rescue_5th", action="store_true", help="Enable quality-gated fifth-lane rescue when pred_quality_logits are present.")
    quality_group.add_argument("--no-quality-rescue-5th", dest="quality_rescue_5th", action="store_false", help="Disable quality-gated fifth-lane rescue.")
    parser.set_defaults(quality_rescue_5th=True)
    parser.add_argument("--quality-rescue-count5-thr", type=float, default=0.70, help="Minimum Count Head P(count=5) for quality-gated fifth-lane rescue.")
    parser.add_argument("--quality-rescue-conf-thr", type=float, default=0.03, help="Minimum lane existence probability for quality-gated fifth-lane rescue.")
    parser.add_argument("--quality-rescue-mean-valid-thr", type=float, default=0.45, help="Minimum mean point-valid probability for quality-gated fifth-lane rescue.")
    parser.add_argument("--quality-rescue-quality-thr", type=float, default=0.55, help="Minimum Quality Head probability for quality-gated fifth-lane rescue.")
    parser.add_argument("--quality-rescue-min-points", type=int, default=5, help="Minimum visible anchors for quality-gated fifth-lane rescue.")
    parser.add_argument("--quality-rescue-dist-px", type=float, default=24.0, help="Minimum distance to existing lanes for quality-gated fifth-lane rescue.")
    parser.add_argument("--edge-last-lane-rescue", action=argparse.BooleanOptionalAction, default=False, help="Prioritize outside-left/right real-query candidates for final lane shortfalls.")
    parser.add_argument("--edge-rescue-conf-thr", type=float, default=0.02)
    parser.add_argument("--edge-rescue-point-valid-thr", type=float, default=0.06)
    parser.add_argument("--edge-rescue-min-points", type=int, default=4)
    parser.add_argument("--edge-rescue-mean-valid-thr", type=float, default=0.35)
    parser.add_argument("--edge-rescue-quality-thr", type=float, default=0.45)
    parser.add_argument("--edge-rescue-outside-gap-px", type=float, default=28.0)
    parser.add_argument("--edge-rescue-dist-px", type=float, default=24.0)
    parser.add_argument("--edge-rescue-min-policy-count", type=int, default=4)
    parser.add_argument("--edge-count4-to5-upgrade", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-count4-to5-prob-margin", type=float, default=0.20)
    parser.add_argument("--soft-count-decision", action="store_true", help="Choose K by candidate quality when Count Head probabilities are close.")
    parser.add_argument("--soft-count-prob-margin", type=float, default=0.08)
    parser.add_argument("--soft-count-quality-weight", type=float, default=1.0)
    parser.add_argument("--soft-count-prior-weight", type=float, default=0.5)
    parser.add_argument("--soft-count-duplicate-penalty", type=float, default=1.0)
    parser.add_argument("--soft-count-invalid-penalty", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images. 0 means all images.")
    parser.add_argument("--save-dir", default="runs/gcs_lane/infer", help="Directory for rendered images and labels.")
    parser.add_argument("--no-save-img", action="store_true", help="Do not save rendered lane images.")
    parser.add_argument("--save-txt", action="store_true", help="Save normalized lane point txt files.")
    parser.add_argument("--save-json", action="store_true", help="Save decoded lanes to predictions.json.")
    parser.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Preserve relative image paths under the save directory to avoid filename collisions.",
    )
    parser.add_argument("--line-width", type=int, default=2, help="Polyline width for rendered lane images.")
    return parser.parse_args()


def count_calibration_from_args(args: argparse.Namespace) -> dict | None:
    """Compatibility shim: score-gap count calibration has been removed."""
    if str(getattr(args, "count_calibration", "none")).lower() != "none":
        raise ValueError("Score-gap count calibration has been removed. Use Count Head Top-K decode instead.")
    return None


def count_head_decode_kwargs_from_args(args: argparse.Namespace, dataset_name: str = "tusimple") -> dict:
    """Build common Count Head Top-K decode keyword arguments from CLI or trainer args."""
    return {
        "use_count_head_decode": bool(
            getattr(args, "use_count_head_decode", getattr(args, "gcs_use_count_head_decode", True))
        ),
        "count_head_temperature": float(getattr(args, "count_head_temp", getattr(args, "gcs_count_head_temp", 1.0))),
        "dataset_name": str(dataset_name),
        "candidate_score_thr": float(getattr(args, "candidate_conf", getattr(args, "gcs_decode_candidate_conf", 0.05))),
        "candidate_point_valid_thr": float(
            getattr(args, "candidate_point_valid_thr", getattr(args, "gcs_decode_candidate_point_valid_thr", 0.20))
        ),
        "candidate_min_points": int(getattr(args, "candidate_min_points", getattr(args, "gcs_decode_candidate_min_points", 5))),
        "enable_rescue_candidate_pool": bool(
            getattr(args, "enable_rescue_candidate_pool", getattr(args, "gcs_enable_rescue_candidate_pool", True))
        ),
        "rescue_candidate_score_thr": float(
            getattr(args, "rescue_candidate_conf", getattr(args, "gcs_decode_rescue_candidate_conf", 0.005))
        ),
        "rescue_candidate_point_valid_thr": float(
            getattr(
                args,
                "rescue_candidate_point_valid_thr",
                getattr(args, "gcs_decode_rescue_candidate_point_valid_thr", 0.08),
            )
        ),
        "rescue_candidate_min_points": int(
            getattr(args, "rescue_candidate_min_points", getattr(args, "gcs_decode_rescue_candidate_min_points", 4))
        ),
        "final_min_points": int(getattr(args, "final_min_points", getattr(args, "gcs_decode_final_min_points", 6))),
        "fifth_min_points": int(getattr(args, "fifth_min_points", getattr(args, "gcs_decode_fifth_min_points", 5))),
        "line_nms_min_overlap": int(getattr(args, "line_nms_min_overlap", getattr(args, "gcs_line_nms_min_overlap", 6))),
        "line_nms_rescue_dist_px": float(
            getattr(args, "line_nms_rescue_dist_px", getattr(args, "gcs_line_nms_rescue_dist_px", 30.0))
        ),
        "quality_rescue_5th": bool(
            getattr(args, "quality_rescue_5th", getattr(args, "gcs_quality_rescue_5th", True))
        ),
        "quality_rescue_count5_thr": float(
            getattr(args, "quality_rescue_count5_thr", getattr(args, "gcs_quality_rescue_count5_thr", 0.70))
        ),
        "quality_rescue_conf_thr": float(
            getattr(args, "quality_rescue_conf_thr", getattr(args, "gcs_quality_rescue_conf_thr", 0.03))
        ),
        "quality_rescue_mean_valid_thr": float(
            getattr(
                args,
                "quality_rescue_mean_valid_thr",
                getattr(args, "gcs_quality_rescue_mean_valid_thr", 0.45),
            )
        ),
        "quality_rescue_quality_thr": float(
            getattr(args, "quality_rescue_quality_thr", getattr(args, "gcs_quality_rescue_quality_thr", 0.55))
        ),
        "quality_rescue_min_points": int(
            getattr(args, "quality_rescue_min_points", getattr(args, "gcs_quality_rescue_min_points", 5))
        ),
        "quality_rescue_dist_px": float(
            getattr(args, "quality_rescue_dist_px", getattr(args, "gcs_quality_rescue_dist_px", 24.0))
        ),
        "last_lane_rescue": bool(
            getattr(args, "last_lane_rescue", getattr(args, "gcs_last_lane_rescue", False))
        ),
        "last_lane_rescue_min_policy_count": int(
            getattr(
                args,
                "last_lane_rescue_min_policy_count",
                getattr(args, "gcs_last_lane_rescue_min_policy_count", 4),
            )
        ),
        "last_lane_rescue_conf_thr": getattr(
            args,
            "last_lane_rescue_conf_thr",
            getattr(args, "gcs_last_lane_rescue_conf_thr", None),
        ),
        "last_lane_rescue_point_valid_thr": float(
            getattr(
                args,
                "last_lane_rescue_point_valid_thr",
                getattr(args, "gcs_last_lane_rescue_point_valid_thr", 0.08),
            )
        ),
        "last_lane_rescue_min_points": int(
            getattr(
                args,
                "last_lane_rescue_min_points",
                getattr(args, "gcs_last_lane_rescue_min_points", 4),
            )
        ),
        "last_lane_rescue_mean_valid_thr": float(
            getattr(
                args,
                "last_lane_rescue_mean_valid_thr",
                getattr(args, "gcs_last_lane_rescue_mean_valid_thr", 0.40),
            )
        ),
        "last_lane_rescue_quality_thr": float(
            getattr(
                args,
                "last_lane_rescue_quality_thr",
                getattr(args, "gcs_last_lane_rescue_quality_thr", 0.50),
            )
        ),
        "last_lane_rescue_dist_px": float(
            getattr(
                args,
                "last_lane_rescue_dist_px",
                getattr(args, "gcs_last_lane_rescue_dist_px", 24.0),
            )
        ),
        "edge_last_lane_rescue": bool(
            getattr(args, "edge_last_lane_rescue", getattr(args, "gcs_edge_last_lane_rescue", False))
        ),
        "edge_rescue_conf_thr": float(
            getattr(args, "edge_rescue_conf_thr", getattr(args, "gcs_edge_rescue_conf_thr", 0.02))
        ),
        "edge_rescue_point_valid_thr": float(
            getattr(args, "edge_rescue_point_valid_thr", getattr(args, "gcs_edge_rescue_point_valid_thr", 0.06))
        ),
        "edge_rescue_min_points": int(
            getattr(args, "edge_rescue_min_points", getattr(args, "gcs_edge_rescue_min_points", 4))
        ),
        "edge_rescue_mean_valid_thr": float(
            getattr(args, "edge_rescue_mean_valid_thr", getattr(args, "gcs_edge_rescue_mean_valid_thr", 0.35))
        ),
        "edge_rescue_quality_thr": float(
            getattr(args, "edge_rescue_quality_thr", getattr(args, "gcs_edge_rescue_quality_thr", 0.45))
        ),
        "edge_rescue_outside_gap_px": float(
            getattr(args, "edge_rescue_outside_gap_px", getattr(args, "gcs_edge_rescue_outside_gap_px", 28.0))
        ),
        "edge_rescue_dist_px": float(
            getattr(args, "edge_rescue_dist_px", getattr(args, "gcs_edge_rescue_dist_px", 24.0))
        ),
        "edge_rescue_min_policy_count": int(
            getattr(args, "edge_rescue_min_policy_count", getattr(args, "gcs_edge_rescue_min_policy_count", 4))
        ),
        "edge_count4_to5_upgrade": bool(
            getattr(args, "edge_count4_to5_upgrade", getattr(args, "gcs_edge_count4_to5_upgrade", True))
        ),
        "edge_count4_to5_prob_margin": float(
            getattr(args, "edge_count4_to5_prob_margin", getattr(args, "gcs_edge_count4_to5_prob_margin", 0.20))
        ),
        "enable_soft_count_decision": bool(
            getattr(args, "soft_count_decision", getattr(args, "gcs_soft_count_decision", False))
        ),
        "soft_count_prob_margin": float(
            getattr(args, "soft_count_prob_margin", getattr(args, "gcs_soft_count_prob_margin", 0.08))
        ),
        "soft_count_quality_weight": float(
            getattr(args, "soft_count_quality_weight", getattr(args, "gcs_soft_count_quality_weight", 1.0))
        ),
        "soft_count_prior_weight": float(
            getattr(args, "soft_count_prior_weight", getattr(args, "gcs_soft_count_prior_weight", 0.5))
        ),
        "soft_count_duplicate_penalty": float(
            getattr(args, "soft_count_duplicate_penalty", getattr(args, "gcs_soft_count_duplicate_penalty", 1.0))
        ),
        "soft_count_invalid_penalty": float(
            getattr(args, "soft_count_invalid_penalty", getattr(args, "gcs_soft_count_invalid_penalty", 1.0))
        ),
    }


def collect_images(source: str | Path, max_images: int = 0) -> list[Path]:
    """Collect image paths from a file, directory, or txt list."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".txt":
        files = []
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            p = Path(line)
            files.append(p if p.is_absolute() else (source.parent / p))
    elif source.is_file() and source.suffix[1:].lower() in IMG_FORMATS:
        files = [source]
    elif source.is_dir():
        files = sorted(p for p in source.rglob("*.*") if p.suffix[1:].lower() in IMG_FORMATS)
    else:
        raise FileNotFoundError(f"Unsupported source path: {source}")

    files = [p.resolve() for p in files]
    if max_images and max_images > 0:
        files = files[: int(max_images)]
    if not files:
        raise FileNotFoundError(f"No images found in {source}")
    return files


def infer_output_root(source: str | Path, images: list[Path]) -> Path:
    """Infer the root used to preserve image-relative output paths."""
    source = Path(source)
    if source.is_dir():
        return source.resolve()
    if source.is_file() and source.suffix.lower() == ".txt":
        parents = [str(p.parent.resolve()) for p in images]
        return Path(os.path.commonpath(parents)).resolve()
    if source.is_file():
        return source.parent.resolve()
    return Path(os.path.commonpath([str(p.parent.resolve()) for p in images])).resolve()


def output_relative_path(img_path: Path, output_root: Path | None, preserve_paths: bool) -> Path:
    """Return an output path relative to the image/label save root."""
    if not preserve_paths or output_root is None:
        return Path(img_path.name)
    try:
        rel = img_path.resolve().relative_to(output_root.resolve())
    except ValueError:
        return Path(img_path.name)
    return rel


def _model_arg_value(model: torch.nn.Module, name: str):
    args = getattr(model, "args", None)
    if isinstance(args, dict):
        return args.get(name)
    return getattr(args, name, None)


def _set_model_gcs_imgsz(model: torch.nn.Module, imgsz: tuple[int, int]) -> None:
    existing = _model_arg_value(model, "gcs_imgsz") or _model_arg_value(model, "image_shape")
    if existing is not None and existing != "":
        existing = normalize_imgsz(existing)
        assert existing == imgsz, (
            f"GCS checkpoint shape H,W={existing} does not match requested inference shape H,W={imgsz}. "
            "Use the same gcs_imgsz for training, inference, and evaluation."
        )
    model.gcs_imgsz = imgsz
    if isinstance(getattr(model, "args", None), dict):
        model.args["gcs_imgsz"] = [int(imgsz[0]), int(imgsz[1])]
    elif getattr(model, "args", None) is not None:
        model.args.gcs_imgsz = [int(imgsz[0]), int(imgsz[1])]


def load_gcs_model(
    weights: str | Path,
    device: torch.device,
    half: bool = False,
    gcs_imgsz: tuple[int, int] | None = None,
) -> torch.nn.Module:
    """Load a trained GCS checkpoint, or construct a yaml model for smoke testing."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"GCS weights not found: {weights}")

    if weights.suffix.lower() in {".yaml", ".yml"}:
        model = GCSLaneModel(str(weights), nc=1, verbose=False).to(device).eval()
    else:
        model, _ = load_checkpoint(weights, device=device, fuse=False)
        model = model.to(device).eval()

    if getattr(model, "task", None) != "gcs_lane":
        raise ValueError(f"Expected a GCS lane model, got task={getattr(model, 'task', None)!r}.")
    if gcs_imgsz is not None:
        _set_model_gcs_imgsz(model, normalize_imgsz(gcs_imgsz))

    if half:
        if device.type != "cuda":
            raise ValueError("--half requires a CUDA device.")
        model.half()
    return model


def preprocess_image(
    img_bgr: np.ndarray,
    imgsz: int | tuple[int, int] | list[int],
    device: torch.device,
    half: bool,
) -> torch.Tensor:
    """Resize to the GCS training coordinate system and convert to BCHW tensor."""
    img_h, img_w = normalize_imgsz(imgsz)
    resized = cv2.resize(img_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(device)
    tensor = tensor.half() if half else tensor.float()
    tensor = tensor.unsqueeze(0) / 255.0
    assert_gcs_image_tensor(tensor, (img_h, img_w), name="preprocessed inference tensor", context="infer_gcs.preprocess_image")
    return tensor


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _json_lane(lane: dict) -> dict:
    item = {
        "query": int(lane["query"]),
        "score": round(float(lane["score"]), 6),
        "exist_score": round(float(lane.get("exist_score", lane["score"])), 6),
        "rank_score": round(float(lane.get("rank_score", lane["score"])), 6),
        "points_norm": np.asarray(lane["points_norm"], dtype=float).round(6).tolist(),
        "points": np.asarray(lane["points"], dtype=float).round(2).tolist(),
    }
    for key in (
        "valid_count",
        "mean_valid_score",
        "mean_valid_score_all",
        "length_factor",
        "valid_count_score",
        "anchor_valid_count_score",
        "smooth_factor",
        "jump_factor",
        "count_head_raw_count",
        "count_head_policy_count",
        "count_head_margin",
        "count_head_shortfall",
    ):
        if key in lane:
            item[key] = round(float(lane[key]), 6)
    if "point_valid" in lane:
        item["point_valid"] = np.asarray(lane["point_valid"], dtype=float).round(3).tolist()
    if "point_valid_scores" in lane:
        item["point_valid_scores"] = np.asarray(lane["point_valid_scores"], dtype=float).round(4).tolist()
    if "visible_points_norm" in lane:
        item["visible_points_norm"] = np.asarray(lane["visible_points_norm"], dtype=float).round(6).tolist()
    if "visible_points" in lane:
        item["visible_points"] = np.asarray(lane["visible_points"], dtype=float).round(2).tolist()
    return item


@torch.inference_mode()
def run_inference(
    weights: str | Path,
    source: str | Path,
    save_dir: str | Path = "runs/gcs_lane/infer",
    imgsz: int | tuple[int, int] | list[int] = (544, 960),
    conf: float = 0.2,
    point_valid_thr: float = 0.5,
    min_points: int = 6,
    nms_dist_px: float = 0.0,
    device: str = "0",
    half: bool = False,
    max_det: int = GCS_DEFAULT_MAX_DET,
    max_images: int = 0,
    save_img: bool = True,
    save_txt: bool = False,
    save_json: bool = False,
    line_width: int = 2,
    count_calibration: dict | None = None,
    use_count_head_decode: bool = True,
    count_head_temperature: float = 1.0,
    dataset_name: str = "tusimple",
    candidate_score_thr: float = 0.05,
    candidate_point_valid_thr: float = 0.20,
    candidate_min_points: int = 5,
    enable_rescue_candidate_pool: bool = True,
    rescue_candidate_score_thr: float = 0.005,
    rescue_candidate_point_valid_thr: float = 0.08,
    rescue_candidate_min_points: int = 4,
    final_min_points: int = 6,
    fifth_min_points: int = 5,
    line_nms_min_overlap: int = 6,
    line_nms_rescue_dist_px: float = 30.0,
    quality_rescue_5th: bool = True,
    quality_rescue_count5_thr: float = 0.70,
    quality_rescue_conf_thr: float = 0.03,
    quality_rescue_mean_valid_thr: float = 0.45,
    quality_rescue_quality_thr: float = 0.55,
    quality_rescue_min_points: int = 5,
    quality_rescue_dist_px: float = 24.0,
    edge_last_lane_rescue: bool = False,
    edge_rescue_conf_thr: float = 0.02,
    edge_rescue_point_valid_thr: float = 0.06,
    edge_rescue_min_points: int = 4,
    edge_rescue_mean_valid_thr: float = 0.35,
    edge_rescue_quality_thr: float = 0.45,
    edge_rescue_outside_gap_px: float = 28.0,
    edge_rescue_dist_px: float = 24.0,
    edge_rescue_min_policy_count: int = 4,
    edge_count4_to5_upgrade: bool = True,
    edge_count4_to5_prob_margin: float = 0.20,
    enable_soft_count_decision: bool = False,
    soft_count_prob_margin: float = 0.08,
    soft_count_quality_weight: float = 1.0,
    soft_count_prior_weight: float = 0.5,
    soft_count_duplicate_penalty: float = 1.0,
    soft_count_invalid_penalty: float = 1.0,
    preserve_paths: bool = False,
) -> list[dict]:
    """Run GCS-YOLO-Lane inference, decode ordered lanes, and optionally save visual outputs."""
    imgsz = normalize_imgsz(imgsz)
    warn_max_det_mismatch(weights, max_det=max_det, context="inference")
    device_obj = select_device(device, verbose=False)
    model = load_gcs_model(weights, device=device_obj, half=half, gcs_imgsz=imgsz)
    images = collect_images(source, max_images=max_images)
    output_root = infer_output_root(source, images) if preserve_paths else None
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")

    save_dir = Path(save_dir)
    image_dir = save_dir / "images"
    label_dir = save_dir / "labels"
    if save_img:
        image_dir.mkdir(parents=True, exist_ok=True)
    if save_txt:
        label_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_infer = 0.0
    total_post = 0.0
    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        tensor = preprocess_image(img, imgsz=imgsz, device=device_obj, half=half)

        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        infer_s = time.perf_counter() - t0

        if not isinstance(preds, dict) or "pred_points" not in preds or "pred_logits" not in preds:
            raise ValueError("GCS inference expects model outputs with pred_points and pred_logits.")

        t1 = time.perf_counter()
        pred_valid = preds.get("pred_valid_logits")
        pred_count = preds.get("pred_count_logits")
        pred_count_boundary = preds.get("pred_count_boundary_logits")
        pred_quality = preds.get("pred_quality_logits")
        lanes = decode_gcs_predictions(
            preds["pred_points"][0],
            preds["pred_logits"][0],
            pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
            pred_count_logits=pred_count[0] if pred_count is not None else None,
            pred_count_boundary_logits=pred_count_boundary[0] if pred_count_boundary is not None else None,
            pred_quality_logits=pred_quality[0] if pred_quality is not None else None,
            image_shape=img.shape[:2],
            score_thr=conf,
            point_valid_thr=point_valid_thr,
            min_points=min_points,
            max_det=max_det,
            nms_dist_px=nms_dist_px,
            count_calibration=count_calibration,
            use_count_head_decode=use_count_head_decode,
            count_head_temperature=count_head_temperature,
            dataset_name=dataset_name,
            candidate_score_thr=candidate_score_thr,
            candidate_point_valid_thr=candidate_point_valid_thr,
            candidate_min_points=candidate_min_points,
            enable_rescue_candidate_pool=enable_rescue_candidate_pool,
            rescue_candidate_score_thr=rescue_candidate_score_thr,
            rescue_candidate_point_valid_thr=rescue_candidate_point_valid_thr,
            rescue_candidate_min_points=rescue_candidate_min_points,
            final_min_points=final_min_points,
            fifth_min_points=fifth_min_points,
            line_nms_min_overlap=line_nms_min_overlap,
            line_nms_rescue_dist_px=line_nms_rescue_dist_px,
            quality_rescue_5th=quality_rescue_5th,
            quality_rescue_count5_thr=quality_rescue_count5_thr,
            quality_rescue_conf_thr=quality_rescue_conf_thr,
            quality_rescue_mean_valid_thr=quality_rescue_mean_valid_thr,
            quality_rescue_quality_thr=quality_rescue_quality_thr,
            quality_rescue_min_points=quality_rescue_min_points,
            quality_rescue_dist_px=quality_rescue_dist_px,
            edge_last_lane_rescue=edge_last_lane_rescue,
            edge_rescue_conf_thr=edge_rescue_conf_thr,
            edge_rescue_point_valid_thr=edge_rescue_point_valid_thr,
            edge_rescue_min_points=edge_rescue_min_points,
            edge_rescue_mean_valid_thr=edge_rescue_mean_valid_thr,
            edge_rescue_quality_thr=edge_rescue_quality_thr,
            edge_rescue_outside_gap_px=edge_rescue_outside_gap_px,
            edge_rescue_dist_px=edge_rescue_dist_px,
            edge_rescue_min_policy_count=edge_rescue_min_policy_count,
            edge_count4_to5_upgrade=edge_count4_to5_upgrade,
            edge_count4_to5_prob_margin=edge_count4_to5_prob_margin,
            enable_soft_count_decision=enable_soft_count_decision,
            soft_count_prob_margin=soft_count_prob_margin,
            soft_count_quality_weight=soft_count_quality_weight,
            soft_count_prior_weight=soft_count_prior_weight,
            soft_count_duplicate_penalty=soft_count_duplicate_penalty,
            soft_count_invalid_penalty=soft_count_invalid_penalty,
        )
        post_s = time.perf_counter() - t1
        total_infer += infer_s
        total_post += post_s

        if save_img:
            vis = draw_gcs_lanes(img, lanes, show_scores=True, line_width=line_width)
            out_img_path = image_dir / output_relative_path(img_path, output_root, preserve_paths)
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_path), vis)
        if save_txt:
            out_label_path = label_dir / output_relative_path(img_path, output_root, preserve_paths).with_suffix(".txt")
            out_label_path.parent.mkdir(parents=True, exist_ok=True)
            save_gcs_lanes_txt(out_label_path, lanes, save_conf=True)

        records.append(
            {
                "image": str(img_path),
                "height": int(img.shape[0]),
                "width": int(img.shape[1]),
                "num_lanes": len(lanes),
                "inference_ms": round(infer_s * 1000.0, 3),
                "postprocess_ms": round(post_s * 1000.0, 3),
                "lanes": [_json_lane(x) for x in lanes],
            }
        )

    if save_json:
        (save_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    n = max(len(records), 1)
    fps = n / max(total_infer + total_post, 1e-9)
    print(f"images: {len(records)}")
    print(f"avg inference: {total_infer * 1000.0 / n:.2f} ms/image")
    print(f"avg postprocess: {total_post * 1000.0 / n:.2f} ms/image")
    print(f"fps(infer+post): {fps:.2f}")
    print(f"saved to: {save_dir.resolve()}")
    return records


def main() -> None:
    args = parse_args()
    imgsz = normalize_imgsz(args.imgsz, dataset=args.dataset)
    defaults = dataset_defaults(args.dataset)
    run_inference(
        weights=args.weights,
        source=args.source or defaults["source"],
        save_dir=args.save_dir,
        imgsz=imgsz,
        conf=args.conf,
        point_valid_thr=args.point_valid_thr,
        min_points=args.min_points,
        nms_dist_px=args.nms_dist_px,
        device=args.device,
        half=args.half,
        max_det=args.max_det,
        max_images=args.max_images,
        save_img=not args.no_save_img,
        save_txt=args.save_txt,
        save_json=args.save_json,
        preserve_paths=args.preserve_paths,
        line_width=args.line_width,
        count_calibration=count_calibration_from_args(args),
        **count_head_decode_kwargs_from_args(args, dataset_name=args.dataset),
    )


if __name__ == "__main__":
    main()
