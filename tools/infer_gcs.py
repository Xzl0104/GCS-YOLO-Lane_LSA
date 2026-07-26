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
from ultralytics.models.gcs.decode_ordered_slot import decode_ordered_slot_predictions
from ultralytics.models.gcs.decode_summary import ordered_slot_decode_params, ordered_slot_decode_runtime_config
from ultralytics.models.gcs.mode_utils import resolve_decode_mode
from ultralytics.nn.modules import GCSLaneHead
from ultralytics.nn.tasks import GCSLaneModel, load_checkpoint
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, assert_gcs_image_tensor, normalize_imgsz, shape_str
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions, draw_gcs_lanes, save_gcs_lanes_txt
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
    run_dir = weight_run_dir(weights)
    args_path = run_dir / "args.yaml" if run_dir is not None else "args.yaml"
    print(
        f"WARNING: {context} max_det={int(max_det)} differs from train-time gcs_eval_max_det={train_max_det} "
        f"in {args_path}. Keep train-time val, sweeps, and official test on one max_det policy for comparable FP/FN.",
        file=sys.stderr,
    )


def dataset_defaults(dataset: str) -> dict[str, Path]:
    """Return conventional inference paths for a converted GCS dataset."""
    root = ROOT / "datasets" / ("tusimple_fixed_y_k56_960x544" if dataset.lower() == "tusimple" else dataset.lower())
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
    parser.add_argument("--decode-mode", choices=("auto", "query", "ordered_slot"), default="auto", help="Decode path.")
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
        "--point-valid-thr",
        type=float,
        default=0.5,
        help="Per-point visibility threshold for fixed-y lane decoding.",
    )
    parser.add_argument("--nms-dist-px", type=float, default=50.0, help="Optional lane duplicate suppression distance in pixels. 0 disables.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--max-det", type=int, default=8, help="Maximum lane queries to keep after score sorting.")
    parser.add_argument("--min-points", type=int, default=2, help="Minimum visible anchors required to keep a lane.")
    parser.add_argument("--valid-before-maxdet", action="store_true", help="Filter point-valid/min_points failures before max_det truncation.")
    parser.add_argument("--extent-decode", action="store_true", help="Use query start/end extent logits for query-mode visibility.")
    parser.add_argument(
        "--extent-decode-mode",
        choices=("none", "interval", "intersect"),
        default="interval",
        help="Query extent decode mode. 'interval' uses extent logits directly; 'intersect' also requires point-valid survival.",
    )
    parser.add_argument("--count-aware-topk", action="store_true", help="Use count_score to keep only the quality-best dynamic lane count.")
    parser.add_argument("--count-aware-min-k", type=int, default=3, help="Minimum k_hat for --count-aware-topk.")
    parser.add_argument("--count-aware-max-k", type=int, default=5, help="Maximum k_hat for --count-aware-topk.")
    parser.add_argument("--count-aware-length-norm", type=float, default=12.0, help="Visible-point count that saturates count-aware length quality.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images. 0 means all images.")
    parser.add_argument("--save-dir", default="runs/gcs_lane/infer", help="Directory for rendered images and labels.")
    parser.add_argument("--no-save-img", action="store_true", help="Do not save rendered lane images.")
    parser.add_argument("--save-txt", action="store_true", help="Save normalized lane point txt files.")
    parser.add_argument("--save-json", action="store_true", help="Save decoded lanes to predictions.json.")
    parser.add_argument("--line-width", type=int, default=2, help="Polyline width for rendered lane images.")
    return parser.parse_args()


def collect_images(source: str | Path, max_images: int = 0) -> list[Path]:
    """Collect image paths from a file, directory, or txt list."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".txt":
        files = []
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
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

    for module in model.modules():
        if isinstance(module, GCSLaneHead):
            module.return_aux = False

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


def _model_gcs_mode(model: torch.nn.Module) -> str:
    """Return the GCS head mode, falling back to model metadata."""
    for module in model.modules():
        if isinstance(module, GCSLaneHead):
            return str(getattr(module, "gcs_mode", "query"))
    return str(getattr(model, "gcs_mode", "query"))


def _json_lane(lane: dict) -> dict:
    item = {
        "query": int(lane["query"]),
        "score": round(float(lane["score"]), 6),
        "points_norm": np.asarray(lane["points_norm"], dtype=float).round(6).tolist(),
        "points": np.asarray(lane["points"], dtype=float).round(2).tolist(),
    }
    if "point_valid" in lane:
        item["point_valid"] = np.asarray(lane["point_valid"], dtype=float).round(3).tolist()
    if "point_valid_scores" in lane:
        item["point_valid_scores"] = np.asarray(lane["point_valid_scores"], dtype=float).round(4).tolist()
    if "visible_points_norm" in lane:
        item["visible_points_norm"] = np.asarray(lane["visible_points_norm"], dtype=float).round(6).tolist()
    if "visible_points" in lane:
        item["visible_points"] = np.asarray(lane["visible_points"], dtype=float).round(2).tolist()
    if "count_aware_quality" in lane:
        item["count_aware_quality"] = round(float(lane["count_aware_quality"]), 6)
    if "extent_start_idx" in lane:
        item["extent_start_idx"] = int(lane["extent_start_idx"])
    if "extent_end_idx" in lane:
        item["extent_end_idx"] = int(lane["extent_end_idx"])
    if "extent_visible_source" in lane:
        item["extent_visible_source"] = str(lane["extent_visible_source"])
    return item


@torch.inference_mode()
def run_inference(
    weights: str | Path,
    source: str | Path,
    save_dir: str | Path = "runs/gcs_lane/infer",
    imgsz: int | tuple[int, int] | list[int] = (544, 960),
    conf: float = 0.2,
    point_valid_thr: float = 0.5,
    nms_dist_px: float = 50.0,
    device: str = "0",
    half: bool = False,
    max_det: int = 8,
    min_points: int = 2,
    valid_before_maxdet: bool = False,
    extent_decode: bool = False,
    extent_decode_mode: str = "interval",
    count_aware_topk: bool = False,
    count_aware_min_k: int = 3,
    count_aware_max_k: int = 5,
    count_aware_length_norm: float = 12.0,
    gcs_min_lanes: int = 2,
    gcs_max_lanes: int = 5,
    gcs_num_slots: int = 5,
    gcs_min_interval_points: int = 2,
    gcs_bottom_order_margin_px: float = 2.0,
    decode_mode: str = "auto",
    max_images: int = 0,
    save_img: bool = True,
    save_txt: bool = False,
    save_json: bool = False,
    line_width: int = 2,
) -> list[dict]:
    """Run GCS-YOLO-Lane inference, decode ordered lanes, and optionally save visual outputs."""
    imgsz = normalize_imgsz(imgsz)
    device_obj = select_device(device, verbose=False)
    model = load_gcs_model(weights, device=device_obj, half=half, gcs_imgsz=imgsz)
    active_decode_mode = resolve_decode_mode(decode_mode, model)
    ordered_slot_runtime_cfg = (
        ordered_slot_decode_runtime_config(context="infer") if active_decode_mode == "ordered_slot" else None
    )
    ordered_slot_params = (
        ordered_slot_decode_params(
            {
                "gcs_min_lanes": gcs_min_lanes,
                "gcs_max_lanes": gcs_max_lanes,
                "gcs_num_slots": gcs_num_slots,
                "gcs_min_interval_points": gcs_min_interval_points,
                "gcs_bottom_order_margin_px": gcs_bottom_order_margin_px,
            }
        )
        if active_decode_mode == "ordered_slot"
        else None
    )
    images = collect_images(source, max_images=max_images)
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
        if active_decode_mode == "ordered_slot":
            lanes = decode_ordered_slot_predictions(
                preds,
                batch_index=0,
                image_shape=img.shape[:2],
                min_lanes=ordered_slot_params["min_lanes"],
                max_lanes=ordered_slot_params["max_lanes"],
                min_interval_points=ordered_slot_params["min_interval_points"],
                order_margin_px=ordered_slot_params["order_margin_px"],
                img_w=float(img.shape[1]),
                order_check=ordered_slot_runtime_cfg["order_check"],
                output_order=ordered_slot_runtime_cfg["output_order"],
            )
        else:
            pred_valid = preds.get("pred_valid_logits")
            pred_quality_logits = preds.get("pred_quality_logits")
            pred_start_logits = preds.get("pred_start_logits")
            pred_end_logits = preds.get("pred_end_logits")
            lanes = decode_gcs_predictions(
                preds["pred_points"][0],
                preds["pred_logits"][0],
                pred_quality_logits=pred_quality_logits[0] if pred_quality_logits is not None else None,
                pred_valid_logits=pred_valid[0] if pred_valid is not None else None,
                pred_start_logits=pred_start_logits[0] if pred_start_logits is not None else None,
                pred_end_logits=pred_end_logits[0] if pred_end_logits is not None else None,
                image_shape=img.shape[:2],
                score_thr=conf,
                point_valid_thr=point_valid_thr,
                min_points=min_points,
                max_det=max_det,
                nms_dist_px=nms_dist_px,
                valid_before_maxdet=valid_before_maxdet,
                extent_decode=extent_decode,
                extent_decode_mode=extent_decode_mode,
                count_aware_topk=count_aware_topk,
                count_aware_min_k=count_aware_min_k,
                count_aware_max_k=count_aware_max_k,
                count_aware_length_norm=count_aware_length_norm,
            )
        post_s = time.perf_counter() - t1
        total_infer += infer_s
        total_post += post_s

        if save_img:
            vis = draw_gcs_lanes(img, lanes, show_scores=True, line_width=line_width)
            cv2.imwrite(str(image_dir / img_path.name), vis)
        if save_txt:
            save_gcs_lanes_txt(label_dir / f"{img_path.stem}.txt", lanes, save_conf=True)

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

    n = max(len(records), 1)
    fps = n / max(total_infer + total_post, 1e-9)
    predictions_json = save_dir / "predictions.json" if save_json else None
    if predictions_json is not None:
        predictions_json.write_text(json.dumps(records, indent=2), encoding="utf-8")

    summary = {
        "config": {
            "weights": str(Path(weights).resolve()),
            "source": str(Path(source).resolve()),
            "save_dir": str(save_dir.resolve()),
            "imgsz": [int(imgsz[0]), int(imgsz[1])],
            "decode_mode": str(active_decode_mode),
            "conf": float(conf),
            "point_valid_thr": float(point_valid_thr),
            "nms_dist_px": float(nms_dist_px),
            "max_det": int(max_det),
            "min_points": int(min_points),
            "valid_before_maxdet": bool(valid_before_maxdet),
            "extent_decode": bool(extent_decode),
            "extent_decode_mode": str(extent_decode_mode),
            "count_aware_topk": bool(count_aware_topk),
            "count_aware_min_k": int(count_aware_min_k),
            "count_aware_max_k": int(count_aware_max_k),
            "count_aware_length_norm": float(count_aware_length_norm),
            "gcs_min_lanes": int(gcs_min_lanes),
            "gcs_max_lanes": int(gcs_max_lanes),
            "gcs_num_slots": int(gcs_num_slots),
            "gcs_min_interval_points": int(gcs_min_interval_points),
            "gcs_bottom_order_margin_px": float(gcs_bottom_order_margin_px),
            "max_images": int(max_images),
            "device": str(device),
            "half": bool(half),
            "save_img": bool(save_img),
            "save_txt": bool(save_txt),
            "save_json": bool(save_json),
            "line_width": int(line_width),
        },
        "outputs": {
            "predictions_json": str(predictions_json.resolve()) if predictions_json is not None else None,
            "image_dir": str(image_dir.resolve()) if save_img else None,
            "label_dir": str(label_dir.resolve()) if save_txt else None,
        },
        "metrics": {
            "images": len(records),
            "avg_inference_ms": round(total_infer * 1000.0 / n, 3),
            "avg_postprocess_ms": round(total_post * 1000.0 / n, 3),
            "fps_infer_post": round(fps, 3),
        },
    }
    (save_dir / "infer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

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
        nms_dist_px=args.nms_dist_px,
        device=args.device,
        half=args.half,
        max_det=args.max_det,
        min_points=args.min_points,
        valid_before_maxdet=args.valid_before_maxdet,
        extent_decode=args.extent_decode,
        extent_decode_mode=args.extent_decode_mode,
        count_aware_topk=args.count_aware_topk,
        count_aware_min_k=args.count_aware_min_k,
        count_aware_max_k=args.count_aware_max_k,
        count_aware_length_norm=args.count_aware_length_norm,
        gcs_min_lanes=args.gcs_min_lanes,
        gcs_max_lanes=args.gcs_max_lanes,
        gcs_num_slots=args.gcs_num_slots,
        gcs_min_interval_points=args.gcs_min_interval_points,
        gcs_bottom_order_margin_px=args.gcs_bottom_order_margin_px,
        decode_mode=args.decode_mode,
        max_images=args.max_images,
        save_img=not args.no_save_img,
        save_txt=args.save_txt,
        save_json=args.save_json,
        line_width=args.line_width,
    )


if __name__ == "__main__":
    main()
