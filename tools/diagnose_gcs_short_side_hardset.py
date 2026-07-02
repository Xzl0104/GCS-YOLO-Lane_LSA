from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.eval_gcs import label_path_for_image, load_gcs_label, pair_geometry  # noqa: E402
from tools.infer_gcs import load_gcs_model, preprocess_image  # noqa: E402
from ultralytics.utils.gcs_fixed_y import validate_training_fixed_y_desc  # noqa: E402
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions, lane_nms, longest_contiguous_valid_mask  # noqa: E402
from ultralytics.utils.gcs_shape import assert_gcs_shape, normalize_imgsz  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_DATASET_ROOT = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-diagnose a fixed short-side hard lane set with any GCS checkpoint."
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--hardset-jsonl", required=True, help="records.jsonl produced by build_gcs_short_side_hardset.py.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.003)
    parser.add_argument("--point-valid-thr", type=float, default=0.6)
    parser.add_argument("--nms-dist-px", type=float, default=18.0)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--valid-before-maxdet", action="store_true")
    parser.add_argument("--min-overlap", type=int, default=3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-records", type=int, default=0, help="Limit hardset records. 0 means all.")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--save-dir", required=True)
    return parser.parse_args()


def clean_float(value: float | int | np.floating | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return round(value_f, ndigits)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value_f = float(value)
        return None if not np.isfinite(value_f) else value_f
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default, allow_nan=False), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=json_default, allow_nan=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "gt_idx" not in record:
                raise KeyError(f"{path}:{line_no} missing gt_idx")
            records.append(record)
    return records


def lane_signature(lane: np.ndarray, valid: np.ndarray) -> str:
    payload = bytearray()
    payload.extend(np.asarray(lane, dtype=np.float32).tobytes())
    payload.extend(np.asarray(valid > 0.5, dtype=np.uint8).tobytes())
    return hashlib.sha256(bytes(payload)).hexdigest()


def array_scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.reshape(-1)[0])


def load_label_meta(label_path: Path) -> dict:
    meta = {"raw_file": "", "image_shape": None}
    if not label_path.exists():
        return meta
    with np.load(label_path, allow_pickle=False) as data:
        if "raw_file" in data.files:
            meta["raw_file"] = array_scalar_str(data["raw_file"])
        if "image_shape" in data.files:
            shape = np.asarray(data["image_shape"]).reshape(-1)
            if shape.size >= 2:
                meta["image_shape"] = [int(shape[0]), int(shape[1])]
        if "fixed_y" not in data.files:
            raise KeyError(f"{label_path} missing fixed_y anchors required by the K56 fixed-y contract.")
        validate_training_fixed_y_desc(data["fixed_y"], name=f"{label_path}: fixed_y")
    return meta


def visible_bucket(n: int) -> str:
    n = int(n)
    if n <= 10:
        return "<=10"
    if n <= 20:
        return "11-20"
    if n <= 30:
        return "21-30"
    return ">30"


def gt_bottom_x(lane, valid):
    idx = np.where(valid > 0.5)[0]
    if idx.size == 0:
        return float("inf")
    return float(lane[idx[0], 0])


def is_side_lane(gt_lanes, gt_valid, gt_idx):
    bottom_xs = [gt_bottom_x(l, v) for l, v in zip(gt_lanes, gt_valid)]
    order = np.argsort(bottom_xs)
    return gt_idx == int(order[0]) or gt_idx == int(order[-1])


def _image_rel(image_path: Path, dataset_root: Path, split: str) -> str:
    try:
        return image_path.resolve().relative_to((dataset_root / "images" / split).resolve()).as_posix()
    except ValueError:
        return image_path.name


def record_id_for(split: str, image_rel: str, gt_idx: int) -> str:
    normalized_image_rel = str(image_rel).replace("\\", "/")
    return f"{split}:{normalized_image_rel}:{int(gt_idx)}"


def query_stage_trace(
    pred_points_t: torch.Tensor,
    pred_logits_t: torch.Tensor,
    pred_valid_t: torch.Tensor | None,
    image_shape: tuple[int, int],
    *,
    conf: float,
    point_valid_thr: float,
    nms_dist_px: float,
    max_det: int,
    min_points: int,
    valid_before_maxdet: bool,
) -> dict:
    if pred_logits_t.ndim == 2 and pred_logits_t.shape[-1] == 1:
        pred_logits_t = pred_logits_t.squeeze(-1)
    points = pred_points_t.detach().float().cpu().clamp(0.0, 1.0)
    scores = pred_logits_t.detach().float().cpu().sigmoid()
    valid_scores = pred_valid_t.detach().float().cpu().sigmoid() if pred_valid_t is not None else None
    query_indices = torch.arange(points.shape[0], dtype=torch.long)

    keep = torch.nonzero(scores >= float(conf), as_tuple=False).flatten()
    stage = {
        "score": set(int(x) for x in keep.tolist()),
        "nms": set(),
        "valid": set(),
        "maxdet": set(),
        "final": set(),
        "rank_after_valid": {},
        "rank_by_score": {},
        "valid_count": {},
    }
    order_all = torch.argsort(scores, descending=True)
    for rank, q in enumerate(order_all.tolist(), start=1):
        stage["rank_by_score"][int(q)] = int(rank)

    for q in range(points.shape[0]):
        if valid_scores is None:
            stage["valid_count"][int(q)] = int(points.shape[1])
        else:
            mask = longest_contiguous_valid_mask(valid_scores[q] >= float(point_valid_thr), min_points=min_points)
            stage["valid_count"][int(q)] = int(mask.sum())

    if keep.numel() == 0:
        return stage

    sorted_points = []
    sorted_valid = []
    for i in keep:
        order_i = torch.argsort(points[i, :, 1], descending=True, stable=True)
        sorted_points.append(points[i][order_i])
        if valid_scores is not None:
            sorted_valid.append(valid_scores[i][order_i])
    points = torch.stack(sorted_points, dim=0)
    scores_kept = scores[keep]
    query_indices = query_indices[keep]
    valid_scores_kept = torch.stack(sorted_valid, dim=0) if sorted_valid else None

    order = torch.argsort(scores_kept, descending=True)
    if nms_dist_px > 0.0:
        sorted_points = points[order]
        sorted_scores = scores_kept[order]
        sorted_queries = query_indices[order]
        sorted_valid_scores = valid_scores_kept[order] if valid_scores_kept is not None else None
        sorted_valid_masks = None
        if sorted_valid_scores is not None:
            sorted_valid_masks = torch.stack(
                [
                    longest_contiguous_valid_mask(v >= float(point_valid_thr), min_points=min_points)
                    for v in sorted_valid_scores
                ],
                dim=0,
            )
        keep_sorted = lane_nms(
            sorted_points,
            sorted_scores,
            image_shape=image_shape,
            dist_thr_px=float(nms_dist_px),
            valid_masks=sorted_valid_masks,
        )
        points = sorted_points[keep_sorted]
        scores_kept = sorted_scores[keep_sorted]
        query_indices = sorted_queries[keep_sorted]
        valid_scores_kept = sorted_valid_scores[keep_sorted] if sorted_valid_scores is not None else None
        order = torch.arange(scores_kept.shape[0], dtype=torch.long)

    stage["nms"] = set(int(x) for x in query_indices.tolist())

    if valid_before_maxdet and valid_scores_kept is not None:
        valid_masks = torch.stack(
            [
                longest_contiguous_valid_mask(v >= float(point_valid_thr), min_points=min_points)
                for v in valid_scores_kept
            ],
            dim=0,
        )
        keep_valid = torch.nonzero(valid_masks.sum(dim=1) >= int(min_points), as_tuple=False).flatten()
        points = points[keep_valid]
        scores_kept = scores_kept[keep_valid]
        query_indices = query_indices[keep_valid]
        valid_scores_kept = valid_scores_kept[keep_valid]
        order = torch.argsort(scores_kept, descending=True)

    stage["valid"] = set(int(x) for x in query_indices.tolist())
    ranked_queries = query_indices[torch.argsort(scores_kept, descending=True)].tolist() if query_indices.numel() else []
    for rank, q in enumerate(ranked_queries, start=1):
        stage["rank_after_valid"][int(q)] = int(rank)

    if max_det is not None and max_det > 0:
        order = order[: int(max_det)]
    query_indices = query_indices[order]
    valid_scores_final = valid_scores_kept[order] if valid_scores_kept is not None else None
    stage["maxdet"] = set(int(x) for x in query_indices.tolist())

    if valid_scores_final is None:
        stage["final"] = set(int(x) for x in query_indices.tolist())
    else:
        final = []
        for lane_i, q in enumerate(query_indices.tolist()):
            mask = longest_contiguous_valid_mask(
                valid_scores_final[lane_i] >= float(point_valid_thr),
                min_points=min_points,
            )
            if int(mask.sum()) >= int(min_points):
                final.append(int(q))
        stage["final"] = set(final)
    return stage


def sorted_prediction_arrays(
    pred_points_t: torch.Tensor,
    pred_valid_t: torch.Tensor | None,
) -> tuple[np.ndarray, np.ndarray]:
    points_t = pred_points_t.detach().float().cpu().clamp(0.0, 1.0)
    valid_scores_t = pred_valid_t.detach().float().cpu().sigmoid() if pred_valid_t is not None else None
    sorted_points = []
    sorted_valid = []
    for q in range(points_t.shape[0]):
        order_q = torch.argsort(points_t[q, :, 1], descending=True, stable=True)
        sorted_points.append(points_t[q][order_q].numpy().astype(np.float32))
        if valid_scores_t is None:
            sorted_valid.append(np.ones((points_t.shape[1],), dtype=np.float32))
        else:
            sorted_valid.append(valid_scores_t[q][order_q].numpy().astype(np.float32))
    return np.stack(sorted_points, axis=0), np.stack(sorted_valid, axis=0)


def pack_pair(idx: int, ape: np.ndarray, mean_x: np.ndarray, overlap: np.ndarray) -> dict:
    if idx < 0:
        return {"query": -1, "ape_px": None, "mean_x_px": None, "overlap": 0}
    return {
        "query": int(idx),
        "ape_px": clean_float(ape[idx, 0]),
        "mean_x_px": clean_float(mean_x[idx, 0]),
        "overlap": int(overlap[idx, 0]),
    }


def best_idx_by_mean_x(mean_x: np.ndarray, overlap: np.ndarray, *, min_overlap: int) -> int:
    if mean_x.size == 0:
        return -1
    costs = mean_x[:, 0].astype(np.float32)
    overlaps = overlap[:, 0].astype(np.int32)
    gated = (overlaps >= int(min_overlap)) & np.isfinite(costs)
    if gated.any():
        return int(np.argmin(np.where(gated, costs, np.inf)))
    finite = np.isfinite(costs)
    return int(np.argmin(np.where(finite, costs, np.inf))) if finite.any() else -1


def best_gt_query(
    pred_points: np.ndarray,
    valid_scores: np.ndarray,
    gt_lanes: np.ndarray,
    gt_valid: np.ndarray,
    image_shape: tuple[int, int],
    gt_idx: int,
    *,
    point_valid_thr: float,
    min_points: int,
    min_overlap: int,
) -> dict:
    h, w = int(image_shape[0]), int(image_shape[1])
    scale = np.array([w, h], dtype=np.float32)
    ape_raw, mean_x_raw, overlap_raw = pair_geometry(pred_points, gt_lanes[[gt_idx]], gt_valid[[gt_idx]], scale)
    raw_idx = best_idx_by_mean_x(mean_x_raw, overlap_raw, min_overlap=min_overlap)
    nearest_raw_idx = best_idx_by_mean_x(mean_x_raw, overlap_raw, min_overlap=0)

    valid_masks = []
    for q in range(pred_points.shape[0]):
        mask_t = longest_contiguous_valid_mask(
            torch.from_numpy(valid_scores[q]) >= float(point_valid_thr),
            min_points=min_points,
        )
        valid_masks.append(mask_t.numpy().astype(np.float32))
    pred_valid = np.stack(valid_masks, axis=0) if valid_masks else np.zeros(pred_points.shape[:2], dtype=np.float32)
    ape_valid, mean_x_valid, overlap_valid = pair_geometry(
        pred_points,
        gt_lanes[[gt_idx]],
        gt_valid[[gt_idx]],
        scale,
        pred_valid=pred_valid,
    )
    valid_idx = best_idx_by_mean_x(mean_x_valid, overlap_valid, min_overlap=min_overlap)

    out = pack_pair(raw_idx, ape_raw, mean_x_raw, overlap_raw)
    out["nearest_raw"] = pack_pair(nearest_raw_idx, ape_raw, mean_x_raw, overlap_raw)
    out["valid_best"] = pack_pair(valid_idx, ape_valid, mean_x_valid, overlap_valid)
    return out


def final_best_gt_lane(
    decoded_lanes: list[dict],
    gt_lanes: np.ndarray,
    gt_valid: np.ndarray,
    image_shape: tuple[int, int],
    gt_idx: int,
    *,
    min_overlap: int,
) -> dict:
    if not decoded_lanes:
        return {"query": -1, "ape_px": None, "mean_x_px": None, "overlap": 0, "score": None}
    pred = np.stack([np.asarray(lane["points_norm"], dtype=np.float32) for lane in decoded_lanes], axis=0)
    pred_valid = np.stack(
        [
            np.asarray(lane.get("point_valid", np.ones(pred.shape[1])), dtype=np.float32)
            for lane in decoded_lanes
        ],
        axis=0,
    )
    h, w = int(image_shape[0]), int(image_shape[1])
    scale = np.array([w, h], dtype=np.float32)
    ape, mean_x, overlap = pair_geometry(pred, gt_lanes[[gt_idx]], gt_valid[[gt_idx]], scale, pred_valid=pred_valid)
    idx = best_idx_by_mean_x(mean_x, overlap, min_overlap=min_overlap)
    packed = pack_pair(idx, ape, mean_x, overlap)
    packed["pred_index"] = int(idx)
    if idx >= 0:
        packed["query"] = int(decoded_lanes[idx].get("query", -1))
        packed["score"] = clean_float(decoded_lanes[idx].get("score"), 8)
    else:
        packed["score"] = None
    return packed


def raw_match(record: dict, *, thr_px: float, min_overlap: int) -> bool:
    mean_x = record.get("best_raw_mean_x_px")
    return mean_x is not None and int(record.get("best_raw_overlap", 0)) >= int(min_overlap) and float(mean_x) <= float(thr_px)


def point_valid_match(record: dict, *, thr_px: float, min_overlap: int) -> bool:
    mean_x = record.get("valid_best_mean_x_px")
    return mean_x is not None and int(record.get("valid_best_overlap", 0)) >= int(min_overlap) and float(mean_x) <= float(thr_px)


def final_match(record: dict, *, thr_px: float, min_overlap: int) -> bool:
    mean_x = record.get("final_best_mean_x_px")
    return mean_x is not None and int(record.get("final_best_overlap", 0)) >= int(min_overlap) and float(mean_x) <= float(thr_px)


def classify_drop_reason(record: dict, *, conf: float, min_overlap: int) -> str:
    q = int(record.get("best_raw_query", -1))
    if final_match(record, thr_px=20.0, min_overlap=min_overlap):
        return "recovered_final_decode"
    if not raw_match(record, thr_px=20.0, min_overlap=min_overlap):
        return "raw_geometry_bad"
    if q < 0 or record.get("best_score") is None or float(record["best_score"]) < float(conf):
        return "low_exist_score"
    if not int(record.get("in_nms", 0)):
        return "nms_suppressed"
    if int(record.get("best_valid_count", 0)) < int(record.get("min_points", 0)):
        return "low_point_valid"
    if not point_valid_match(record, thr_px=20.0, min_overlap=min_overlap):
        return "valid_geometry_bad"
    if not int(record.get("in_valid", 0)):
        return "valid_before_filtered"
    if not int(record.get("in_maxdet", 0)):
        return "rank_maxdet"
    if not int(record.get("in_final", 0)):
        return "final_min_points_filtered"
    return "final_geometry_bad"


def validate_query_outputs(
    pred_points_t: torch.Tensor,
    pred_logits_t: torch.Tensor,
    pred_valid_t: torch.Tensor | None,
    *,
    image_path: Path,
) -> None:
    if tuple(pred_points_t.shape) != (12, 56, 2):
        raise ValueError(
            f"{image_path}: expected query-mode pred_points shape (12, 56, 2), got {tuple(pred_points_t.shape)}. "
            "This short-side hardset diagnostic is comparable only for the active Q12/K56 query contract."
        )
    logits_shape = tuple(pred_logits_t.shape)
    if logits_shape not in {(12,), (12, 1)}:
        raise ValueError(f"{image_path}: expected pred_logits shape (12,) or (12,1), got {logits_shape}.")
    if pred_valid_t is None:
        raise KeyError(f"{image_path}: missing pred_valid_logits required for valid-stage diagnosis.")
    if tuple(pred_valid_t.shape) != (12, 56):
        raise ValueError(f"{image_path}: expected pred_valid_logits shape (12, 56), got {tuple(pred_valid_t.shape)}.")


def verify_hardset_record(base: dict, current: dict, *, tolerance: float = 1e-5) -> None:
    if not base:
        return
    key = current.get("hardset_record_id") or current.get("record_id")
    for field in ("split", "image_rel", "raw_file", "gt_lanes", "gt_idx", "visible_count", "lane_signature"):
        if field in base and base[field] != current.get(field):
            raise ValueError(
                f"Hardset record drift for {key}: {field} changed from {base[field]!r} to {current.get(field)!r}."
            )
    if "gt_bottom_x" in base:
        old = float(base["gt_bottom_x"])
        new = float(current["gt_bottom_x"])
        if abs(old - new) > float(tolerance):
            raise ValueError(f"Hardset record drift for {key}: gt_bottom_x changed from {old} to {new}.")


def diagnose_one_image(
    model: torch.nn.Module,
    image_path: Path,
    labels_dir: Path,
    *,
    split: str,
    dataset_root: Path,
    imgsz: tuple[int, int],
    device: torch.device,
    half: bool,
    args: argparse.Namespace,
    gt_indices: list[int] | None = None,
    base_records_by_gt: dict[int, dict] | None = None,
) -> list[dict]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    assert_gcs_shape(img.shape[:2], imgsz, name="diagnostic image", context=f"{image_path}")

    label_path = label_path_for_image(image_path, labels_dir)
    gt_lanes, gt_valid = load_gcs_label(label_path)
    meta = load_label_meta(label_path)
    gt_count = int(gt_lanes.shape[0])

    tensor = preprocess_image(img, imgsz=imgsz, device=device, half=half)
    preds = model(tensor)
    if "pred_points" not in preds or "pred_logits" not in preds:
        raise KeyError("Short-side raw-query diagnostic requires query-mode pred_points and pred_logits outputs.")
    pred_points_t = preds["pred_points"][0].detach().float()
    pred_logits_t = preds["pred_logits"][0].detach().float()
    pred_valid_t = preds.get("pred_valid_logits")
    pred_valid_t = pred_valid_t[0].detach().float() if pred_valid_t is not None else None
    validate_query_outputs(pred_points_t, pred_logits_t, pred_valid_t, image_path=image_path)
    if pred_logits_t.ndim == 2 and pred_logits_t.shape[-1] == 1:
        pred_logits_t = pred_logits_t.squeeze(-1)

    stage = query_stage_trace(
        pred_points_t,
        pred_logits_t,
        pred_valid_t,
        img.shape[:2],
        conf=args.conf,
        point_valid_thr=args.point_valid_thr,
        nms_dist_px=args.nms_dist_px,
        max_det=args.max_det,
        min_points=args.min_points,
        valid_before_maxdet=bool(args.valid_before_maxdet),
    )
    decoded_lanes = decode_gcs_predictions(
        pred_points_t,
        pred_logits_t,
        pred_valid_logits=pred_valid_t,
        image_shape=img.shape[:2],
        score_thr=args.conf,
        point_valid_thr=args.point_valid_thr,
        min_points=args.min_points,
        max_det=args.max_det,
        nms_dist_px=args.nms_dist_px,
        valid_before_maxdet=bool(args.valid_before_maxdet),
    )
    decoded_final = {int(lane.get("query", -1)) for lane in decoded_lanes}
    if decoded_final != stage["final"]:
        raise AssertionError(
            f"{image_path}: traced final queries {sorted(stage['final'])} "
            f"do not match decode_gcs_predictions queries {sorted(decoded_final)}"
        )
    scores = pred_logits_t.detach().float().cpu().sigmoid().numpy().astype(np.float32)
    pred_points, valid_scores = sorted_prediction_arrays(pred_points_t, pred_valid_t)
    image_rel = _image_rel(image_path, dataset_root, split)

    if gt_indices is None:
        gt_indices = list(range(gt_count))
    records = []
    for gt_idx in gt_indices:
        gt_idx = int(gt_idx)
        if gt_idx < 0 or gt_idx >= gt_count:
            raise IndexError(f"{image_path}: gt_idx={gt_idx} outside GT lane count {gt_count}")

        best = best_gt_query(
            pred_points,
            valid_scores,
            gt_lanes,
            gt_valid,
            img.shape[:2],
            gt_idx,
            point_valid_thr=args.point_valid_thr,
            min_points=args.min_points,
            min_overlap=args.min_overlap,
        )
        valid_best = best["valid_best"]
        nearest_raw = best["nearest_raw"]
        final_best = final_best_gt_lane(
            decoded_lanes,
            gt_lanes,
            gt_valid,
            img.shape[:2],
            gt_idx,
            min_overlap=args.min_overlap,
        )
        q = int(best["query"])
        base = (base_records_by_gt or {}).get(gt_idx, {})
        computed_record_id = record_id_for(split, image_rel, gt_idx)
        record_id = str(base.get("record_id", computed_record_id))
        bottom_x = gt_bottom_x(gt_lanes[gt_idx], gt_valid[gt_idx])
        record = {
            "record_id": record_id,
            "hardset_record_id": str(base.get("record_id", record_id)) if base else record_id,
            "split": split,
            "image": str(image_path),
            "image_rel": image_rel,
            "label": str(label_path),
            "raw_file": meta.get("raw_file", ""),
            "image_shape": [int(img.shape[0]), int(img.shape[1])],
            "gt_lanes": gt_count,
            "gt_idx": gt_idx,
            "gt_bottom_x": clean_float(bottom_x, 8),
            "gt_bottom_x_px": clean_float(bottom_x * float(img.shape[1]), 4),
            "visible_count": int(gt_valid[gt_idx].sum()),
            "visible_bucket": visible_bucket(int(gt_valid[gt_idx].sum())),
            "side_lane": bool(is_side_lane(gt_lanes, gt_valid, gt_idx)),
            "lane_signature": lane_signature(gt_lanes[gt_idx], gt_valid[gt_idx]),
            "best_raw_query": q,
            "best_raw_mean_x_px": best["mean_x_px"],
            "best_raw_ape_px": best["ape_px"],
            "best_raw_overlap": int(best["overlap"]),
            "nearest_raw_query": int(nearest_raw["query"]),
            "nearest_raw_mean_x_px": nearest_raw["mean_x_px"],
            "nearest_raw_ape_px": nearest_raw["ape_px"],
            "nearest_raw_overlap": int(nearest_raw["overlap"]),
            "valid_best_query": int(valid_best["query"]),
            "valid_best_mean_x_px": valid_best["mean_x_px"],
            "valid_best_ape_px": valid_best["ape_px"],
            "valid_best_overlap": int(valid_best["overlap"]),
            "final_best_query": int(final_best["query"]),
            "final_best_pred_index": int(final_best.get("pred_index", -1)),
            "final_best_mean_x_px": final_best["mean_x_px"],
            "final_best_ape_px": final_best["ape_px"],
            "final_best_overlap": int(final_best["overlap"]),
            "final_best_score": final_best["score"],
            "best_score": clean_float(scores[q], 8) if q >= 0 else None,
            "best_rank_by_score": stage["rank_by_score"].get(q),
            "best_rank_after_valid": stage["rank_after_valid"].get(q),
            "best_valid_count": stage["valid_count"].get(q),
            "in_score": int(q in stage["score"]) if q >= 0 else 0,
            "in_nms": int(q in stage["nms"]) if q >= 0 else 0,
            "in_valid": int(q in stage["valid"]) if q >= 0 else 0,
            "in_maxdet": int(q in stage["maxdet"]) if q >= 0 else 0,
            "in_final": int(q in stage["final"]) if q >= 0 else 0,
            "conf": float(args.conf),
            "point_valid_thr": float(args.point_valid_thr),
            "nms_dist_px": float(args.nms_dist_px),
            "max_det": int(args.max_det),
            "min_points": int(args.min_points),
            "valid_before_maxdet": bool(args.valid_before_maxdet),
        }
        record["best_raw_mean_x_px"] = clean_float(record["best_raw_mean_x_px"])
        record["best_raw_ape_px"] = clean_float(record["best_raw_ape_px"])
        record["nearest_raw_mean_x_px"] = clean_float(record["nearest_raw_mean_x_px"])
        record["nearest_raw_ape_px"] = clean_float(record["nearest_raw_ape_px"])
        record["valid_best_mean_x_px"] = clean_float(record["valid_best_mean_x_px"])
        record["valid_best_ape_px"] = clean_float(record["valid_best_ape_px"])
        record["final_best_mean_x_px"] = clean_float(record["final_best_mean_x_px"])
        record["final_best_ape_px"] = clean_float(record["final_best_ape_px"])
        verify_hardset_record(base, record)
        record["drop_reason"] = classify_drop_reason(record, conf=args.conf, min_overlap=args.min_overlap)
        records.append(record)
    return records


def summarize_subset(records: list[dict], *, min_overlap: int, include_groups: bool = False) -> dict:
    n = len(records)

    def count_if(fn) -> int:
        return sum(1 for record in records if fn(record))

    raw20 = count_if(lambda r: raw_match(r, thr_px=20.0, min_overlap=min_overlap))
    raw30 = count_if(lambda r: raw_match(r, thr_px=30.0, min_overlap=min_overlap))
    after_score = count_if(lambda r: raw_match(r, thr_px=20.0, min_overlap=min_overlap) and bool(r.get("in_score")))
    after_valid = count_if(lambda r: point_valid_match(r, thr_px=20.0, min_overlap=min_overlap))
    after_nms = count_if(lambda r: raw_match(r, thr_px=20.0, min_overlap=min_overlap) and bool(r.get("in_nms")))
    final_decode = count_if(lambda r: final_match(r, thr_px=20.0, min_overlap=min_overlap))
    final_query_survival = count_if(
        lambda r: raw_match(r, thr_px=20.0, min_overlap=min_overlap) and bool(r.get("in_final"))
    )

    def rate(value: int) -> float | None:
        return None if n == 0 else round(float(value) / float(n), 6)

    summary = {
        "hard_lanes": int(n),
        "raw_match_recall_20px": rate(raw20),
        "raw_match_20px_count": int(raw20),
        "raw_match_recall_30px": rate(raw30),
        "raw_match_30px_count": int(raw30),
        "after_score_recall": rate(after_score),
        "after_score_count": int(after_score),
        "after_point_valid_recall": rate(after_valid),
        "after_point_valid_count": int(after_valid),
        "after_nms_recall": rate(after_nms),
        "after_nms_count": int(after_nms),
        "final_decode_recall": rate(final_decode),
        "final_decode_count": int(final_decode),
        "final_query_survival_recall": rate(final_query_survival),
        "final_query_survival_count": int(final_query_survival),
        "drop_reason_hist": {str(k): int(v) for k, v in sorted(Counter(r.get("drop_reason", "") for r in records).items())},
    }
    if include_groups:
        by_gt = defaultdict(list)
        by_vis = defaultdict(list)
        by_split = defaultdict(list)
        for record in records:
            by_gt[str(record.get("gt_lanes"))].append(record)
            by_vis[str(record.get("visible_bucket", visible_bucket(record.get("visible_count", 0))))].append(record)
            by_split[str(record.get("split", ""))].append(record)
        summary["by_gt_count"] = {
            key: summarize_subset(group, min_overlap=min_overlap, include_groups=False) for key, group in sorted(by_gt.items())
        }
        summary["by_visible_bucket"] = {
            key: summarize_subset(group, min_overlap=min_overlap, include_groups=False) for key, group in sorted(by_vis.items())
        }
        summary["by_split"] = {
            key: summarize_subset(group, min_overlap=min_overlap, include_groups=False) for key, group in sorted(by_split.items())
        }
    return summary


def build_config(args: argparse.Namespace, imgsz: tuple[int, int]) -> dict:
    return {
        "weights": str(Path(args.weights)),
        "dataset_root": str(Path(args.dataset_root)),
        "imgsz": [int(imgsz[0]), int(imgsz[1])],
        "conf": float(args.conf),
        "point_valid_thr": float(args.point_valid_thr),
        "nms_dist_px": float(args.nms_dist_px),
        "max_det": int(args.max_det),
        "min_points": int(args.min_points),
        "valid_before_maxdet": bool(args.valid_before_maxdet),
        "min_overlap": int(args.min_overlap),
        "device": str(args.device),
        "half": bool(args.half),
    }


def resolve_image_from_record(record: dict, dataset_root: Path) -> Path:
    split = str(record.get("split", ""))
    image_rel = record.get("image_rel")
    if split and image_rel:
        candidate = dataset_root / "images" / split / str(image_rel)
        if candidate.exists():
            return candidate
    image_value = record.get("image")
    if image_value:
        candidate = Path(image_value)
        if candidate.exists():
            return candidate
        if split:
            fallback = dataset_root / "images" / split / candidate.name
            if fallback.exists():
                return fallback
    raise FileNotFoundError(f"Cannot resolve hardset image for record_id={record.get('record_id')}")


def warmup_model(model: torch.nn.Module, image_path: Path, imgsz: tuple[int, int], device: torch.device, half: bool, warmup: int) -> None:
    if warmup <= 0:
        return
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read warmup image: {image_path}")
    tensor = preprocess_image(img, imgsz=imgsz, device=device, half=half)
    for _ in range(int(warmup)):
        _ = model(tensor)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    t0 = time.time()
    imgsz = normalize_imgsz(args.imgsz, dataset="tusimple")
    dataset_root = Path(args.dataset_root)
    hardset_records = load_jsonl(args.hardset_jsonl)
    if args.max_records and args.max_records > 0:
        hardset_records = hardset_records[: int(args.max_records)]
    if not hardset_records:
        raise ValueError(f"No hardset records found in {args.hardset_jsonl}")

    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=imgsz)

    first_image = resolve_image_from_record(hardset_records[0], dataset_root)
    warmup_model(model, first_image, imgsz, device, args.half, args.warmup)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in hardset_records:
        split = str(record.get("split", ""))
        image_key = str(record.get("image_rel") or record.get("image") or "")
        grouped[(split, image_key)].append(record)

    out_by_id: dict[str, dict] = {}
    for group_i, ((split, _), group) in enumerate(grouped.items(), start=1):
        image_path = resolve_image_from_record(group[0], dataset_root)
        labels_dir = dataset_root / "labels_gcs" / split
        base_by_gt = {int(record["gt_idx"]): record for record in group}
        image_records = diagnose_one_image(
            model,
            image_path,
            labels_dir,
            split=split,
            dataset_root=dataset_root,
            imgsz=imgsz,
            device=device,
            half=args.half,
            args=args,
            gt_indices=sorted(base_by_gt),
            base_records_by_gt=base_by_gt,
        )
        for record in image_records:
            out_by_id[str(record["hardset_record_id"])] = record
        if group_i % 200 == 0 or group_i == len(grouped):
            print(f"diagnosed images: {group_i}/{len(grouped)}")

    output_records = []
    for base in hardset_records:
        key = str(base.get("record_id", record_id_for(base.get("split", ""), base.get("image_rel", base.get("image", "")), base["gt_idx"])))
        if key not in out_by_id:
            raise KeyError(f"Missing diagnostic output for hardset record {key}")
        output_records.append(out_by_id[key])

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_subset(output_records, min_overlap=args.min_overlap, include_groups=True)
    summary.update(
        {
            "config": build_config(args, imgsz),
            "hardset_jsonl": str(Path(args.hardset_jsonl)),
            "images": int(len(grouped)),
            "runtime_sec": round(float(time.time() - t0), 3),
            "contains_test": any(str(record.get("split", "")).lower() == "test" for record in hardset_records),
        }
    )
    summary["selection_eligible"] = not bool(summary["contains_test"])
    summary["not_for_tuning"] = bool(summary["contains_test"])
    write_jsonl(save_dir / "records.jsonl", output_records)
    write_json(save_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=json_default, allow_nan=False))
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
