from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import (  # noqa: E402
    find_tusimple_archive_root,
    official_gt_contract_summary,
    read_tusimple_json_lines,
    resolve_tusimple_gt_json,
    tusimple_image_path,
    valid_tusimple_lanes,
)
from tools.infer_gcs import load_gcs_model, preprocess_image, warn_max_det_mismatch  # noqa: E402
from ultralytics.models.gcs.decode_summary import load_decode_yaml, validate_decode_yaml_for_model  # noqa: E402
from ultralytics.models.gcs.mode_utils import resolve_decode_mode  # noqa: E402
from ultralytics.utils.gcs_postprocess import (  # noqa: E402
    lane_nms,
    longest_contiguous_valid_mask,
)
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_VAL_GT = (
    ROOT
    / "runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/"
    / "tusimple_official_val_363_folder_aware_seed20260602.json"
)
DEFAULT_WEIGHTS = ROOT / "runs/gcs_lane/query_count_head_ce05_v1/weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Raw Q=12 TuSimple diagnostic for separating geometry recall from "
            "score, point-valid/min_points, NMS, pool cap, and final decode filtering."
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
    parser.add_argument("--conf", type=float, default=0.003, help="Lane existence confidence threshold.")
    parser.add_argument("--point-valid-thr", type=float, default=0.6, help="Per-point visibility threshold.")
    parser.add_argument("--nms-dist-px", type=float, default=0.0, help="Lane-NMS distance in original-image pixels.")
    parser.add_argument("--max-det", type=int, default=6, help="Final decode max_det.")
    parser.add_argument("--min-points", type=int, default=5, help="Minimum visible anchors required to keep a lane.")
    parser.add_argument("--valid-before-maxdet", action="store_true", help="Apply point-valid/min_points before max_det.")
    parser.add_argument("--pool-max-det", type=int, default=12, help="Candidate pool cap after filters. 0 disables cap.")
    parser.add_argument("--match-thr-px", type=float, default=20.0, help="APE threshold used for missing_gt_lane_ids.")
    parser.add_argument("--match-min-overlap", type=int, default=3, help="Minimum shared GT-visible samples for APE match.")
    parser.add_argument("--raw-match-thrs", nargs="+", type=float, default=[20.0, 30.0, 40.0])
    parser.add_argument("--short-visible-max", type=int, default=10, help="short-visible lane bucket threshold.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of GT records. 0 means all.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of untimed warmup forwards.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--save-dir", default=None, help="Output directory.")
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _limit_records(records: list[dict], max_images: int) -> list[dict]:
    if max_images and max_images > 0:
        return records[: int(max_images)]
    return records


def _default_gt_json(args: argparse.Namespace) -> str | None:
    if args.gt_json:
        return args.gt_json
    if str(args.split).lower() == "val" and DEFAULT_VAL_GT.exists():
        return str(DEFAULT_VAL_GT)
    return None


def _weight_run_dir(weights: str | Path) -> Path | None:
    path = Path(weights)
    if path.name.lower().endswith((".pt", ".pth")) and path.parent.name == "weights":
        return path.parent.parent
    return None


def resolve_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir is not None and str(args.save_dir).strip():
        return Path(args.save_dir)
    run_dir = _weight_run_dir(args.weights)
    tag = (
        f"raw_q12_filters_{args.split}_conf{float(args.conf):.4g}_"
        f"pvalid{float(args.point_valid_thr):.4g}_nms{float(args.nms_dist_px):.4g}_"
        f"maxdet{int(args.max_det)}_minp{int(args.min_points)}"
    ).replace(".", "p")
    if bool(args.valid_before_maxdet):
        tag += "_validbeforemaxdet"
    if run_dir is not None:
        return run_dir / tag
    return ROOT / "runs" / "gcs_lane" / "raw_q12_filters" / Path(args.weights).stem / tag


def _apply_query_decode_yaml(args: argparse.Namespace, decode_yaml_cfg: dict) -> None:
    args.conf = float(decode_yaml_cfg["conf"])
    args.point_valid_thr = float(decode_yaml_cfg["point_valid_thr"])
    args.nms_dist_px = float(decode_yaml_cfg["nms_dist_px"])
    args.max_det = int(decode_yaml_cfg["max_det"])
    args.min_points = int(decode_yaml_cfg["min_points"])
    args.valid_before_maxdet = bool(decode_yaml_cfg.get("valid_before_maxdet", False))


def _safe_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _csv_value(value) -> str | int | float | bool:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf"
    if isinstance(value, (list, tuple)):
        return ";".join(str(_csv_value(x)) for x in value)
    return value


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})


def _valid_gt_lanes_with_ids(record: dict) -> list[tuple[int, list[float]]]:
    out = []
    for idx, lane in enumerate(record.get("lanes", [])):
        if any(float(x) >= 0.0 for x in lane):
            out.append((int(idx), [float(x) for x in lane]))
    return out


def _visible_count(lane: list[float]) -> int:
    return int(sum(1 for x in lane if float(x) >= 0.0))


def _bottom_x(lane: list[float], h_samples: list[float]) -> float:
    best_y = -float("inf")
    best_x = float("nan")
    for x, y in zip(lane, h_samples):
        if float(x) >= 0.0 and float(y) > best_y:
            best_y = float(y)
            best_x = float(x)
    return best_x


def _lane_position_map(record: dict) -> dict[int, dict]:
    h_samples = [float(x) for x in record["h_samples"]]
    lanes = _valid_gt_lanes_with_ids(record)
    ordered = sorted(lanes, key=lambda item: _bottom_x(item[1], h_samples))
    n = len(ordered)
    out: dict[int, dict] = {}
    for rank, (lane_id, lane) in enumerate(ordered):
        if rank == 0:
            position = "leftmost"
        elif rank == n - 1:
            position = "rightmost"
        else:
            position = "center"
        if n <= 1:
            side = "single"
        elif rank == 0:
            side = "left_side"
        elif rank == n - 1:
            side = "right_side"
        else:
            side = "center"
        out[int(lane_id)] = {
            "lane_order": int(rank),
            "lane_position": position,
            "side_group": side,
            "bottom_x": _bottom_x(lane, h_samples),
        }
    return out


def _parse_date_session(raw_file: str) -> tuple[str, str]:
    parts = raw_file.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "clips":
        return parts[1], parts[2]
    return "", ""


def _visible_bucket(visible: int, short_visible_max: int) -> str:
    visible = int(visible)
    if visible <= int(short_visible_max):
        return f"<= {int(short_visible_max)}"
    if visible <= 20:
        return "11..20"
    return ">20"


def _interp_lane_xs(points_norm: np.ndarray, h_samples: list[float], image_shape: tuple[int, int]) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    points = np.asarray(points_norm, dtype=np.float32).reshape(-1, 2).copy()
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * float(w)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * float(h)
    order = np.argsort(points[:, 1], kind="stable")
    points = points[order]
    ys = points[:, 1]
    xs = points[:, 0]
    unique_ys, unique_idx = np.unique(np.round(ys, decimals=4), return_index=True)
    if unique_ys.shape[0] < 2:
        return np.full(len(h_samples), np.nan, dtype=np.float32)
    unique_xs = xs[unique_idx]
    return np.interp(np.asarray(h_samples, dtype=np.float32), unique_ys, unique_xs, left=np.nan, right=np.nan).astype(np.float32)


def _ape_px(pred_xs: np.ndarray, gt_xs: list[float], min_overlap: int) -> tuple[float, int]:
    gt = np.asarray(gt_xs, dtype=np.float32)
    pred = np.asarray(pred_xs, dtype=np.float32)
    valid = (gt >= 0.0) & np.isfinite(pred)
    overlap = int(valid.sum())
    if overlap < int(min_overlap):
        return float("inf"), overlap
    return float(np.mean(np.abs(pred[valid] - gt[valid]))), overlap


def _valid_len(valid_scores_desc: torch.Tensor | None, thr: float, min_points: int) -> int:
    if valid_scores_desc is None:
        return 0
    mask = longest_contiguous_valid_mask(valid_scores_desc >= float(thr), min_points=min_points)
    return int(mask.sum().item())


def _valid_count(valid_scores_desc: torch.Tensor | None, thr: float) -> int:
    if valid_scores_desc is None:
        return 0
    return int((valid_scores_desc >= float(thr)).sum().item())


def _valid_recall_at_gt(
    points_norm: np.ndarray,
    valid_scores: np.ndarray | None,
    h_samples: list[float],
    gt_lane: list[float],
    image_shape: tuple[int, int],
    thr: float,
) -> float:
    if valid_scores is None:
        return 0.0
    h, _ = int(image_shape[0]), int(image_shape[1])
    pts = np.asarray(points_norm, dtype=np.float32).reshape(-1, 2).copy()
    ys = np.clip(pts[:, 1], 0.0, 1.0) * float(h)
    order = np.argsort(ys, kind="stable")
    ys = ys[order]
    probs = np.asarray(valid_scores, dtype=np.float32).reshape(-1)[order]
    unique_ys, unique_idx = np.unique(np.round(ys, decimals=4), return_index=True)
    if unique_ys.shape[0] < 2:
        return 0.0
    gt = np.asarray(gt_lane, dtype=np.float32)
    valid_gt = gt >= 0.0
    if int(valid_gt.sum()) == 0:
        return 0.0
    interp = np.interp(np.asarray(h_samples, dtype=np.float32), unique_ys, probs[unique_idx], left=np.nan, right=np.nan)
    hit = (interp[valid_gt] >= float(thr)) & np.isfinite(interp[valid_gt])
    return float(hit.sum() / max(int(valid_gt.sum()), 1))


def _query_arrays(preds: dict, image_shape: tuple[int, int], h_samples: list[float], min_points: int) -> list[dict]:
    pred_points = preds["pred_points"][0].detach().float().cpu().clamp(0.0, 1.0)
    pred_logits = preds["pred_logits"][0].detach().float().cpu()
    if pred_logits.ndim == 2 and pred_logits.shape[-1] == 1:
        pred_logits = pred_logits.squeeze(-1)
    pred_valid_logits = preds.get("pred_valid_logits")
    pred_valid_scores = None
    if pred_valid_logits is not None:
        pred_valid_scores = pred_valid_logits[0].detach().float().cpu().sigmoid()

    rows = []
    for q in range(int(pred_points.shape[0])):
        points_q = pred_points[q]
        order_desc = torch.argsort(points_q[:, 1], descending=True, stable=True)
        valid_desc = pred_valid_scores[q][order_desc] if pred_valid_scores is not None else None
        points_norm = points_q.numpy().astype(np.float32)
        valid_scores = None if pred_valid_scores is None else pred_valid_scores[q].numpy().astype(np.float32)
        rows.append(
            {
                "query": int(q),
                "score": float(pred_logits[q].sigmoid().item()),
                "points_norm": points_norm,
                "pred_xs": _interp_lane_xs(points_norm, h_samples, image_shape),
                "valid_scores": valid_scores,
                "valid_len_0.5": _valid_len(valid_desc, 0.5, min_points=min_points),
                "valid_len_0.6": _valid_len(valid_desc, 0.6, min_points=min_points),
                "valid_count_0.5": _valid_count(valid_desc, 0.5),
                "valid_count_0.6": _valid_count(valid_desc, 0.6),
            }
        )
    return rows


def _trace_decode(
    preds: dict,
    *,
    image_shape: tuple[int, int],
    score_thr: float,
    point_valid_thr: float,
    min_points: int,
    max_det: int | None,
    nms_dist_px: float,
    valid_before_maxdet: bool,
) -> tuple[list[dict], dict]:
    pred_points = preds["pred_points"][0]
    pred_logits = preds["pred_logits"][0]
    pred_valid_logits = preds.get("pred_valid_logits")
    if pred_valid_logits is not None:
        pred_valid_logits = pred_valid_logits[0]
    if pred_logits.ndim == 2 and pred_logits.shape[-1] == 1:
        pred_logits = pred_logits.squeeze(-1)
    points = pred_points.detach().float().cpu().clamp(0.0, 1.0)
    scores = pred_logits.detach().float().cpu().sigmoid()
    point_valid_scores = pred_valid_logits.detach().float().cpu().sigmoid() if pred_valid_logits is not None else None
    query_indices = torch.arange(points.shape[0], dtype=torch.long)

    trace = {
        "raw_query_count": int(points.shape[0]),
        "after_score_count": 0,
        "after_point_valid_min_points_count": 0,
        "after_nms_count": 0,
        "after_pre_maxdet_valid_count": 0,
        "after_pool_max_det_count": 0,
        "final_count_before_export": 0,
    }
    if min_points > points.shape[1]:
        return [], trace

    keep = torch.nonzero(scores >= float(score_thr), as_tuple=False).flatten()
    trace["after_score_count"] = int(keep.numel())
    if keep.numel() == 0:
        return [], trace

    sorted_points = []
    sorted_valid_scores = []
    for i in keep:
        order_i = torch.argsort(points[i, :, 1], descending=True, stable=True)
        sorted_points.append(points[i][order_i])
        if point_valid_scores is not None:
            sorted_valid_scores.append(point_valid_scores[i][order_i])
    points = torch.stack(sorted_points, dim=0)
    point_valid_scores = torch.stack(sorted_valid_scores, dim=0) if sorted_valid_scores else None
    scores = scores[keep]
    query_indices = query_indices[keep]

    if point_valid_scores is not None:
        score_valid_masks = torch.stack(
            [
                longest_contiguous_valid_mask(v >= float(point_valid_thr), min_points=min_points)
                for v in point_valid_scores
            ],
            dim=0,
        )
        trace["after_point_valid_min_points_count"] = int((score_valid_masks.sum(dim=1) >= int(min_points)).sum().item())
    else:
        trace["after_point_valid_min_points_count"] = int(scores.shape[0])

    order = torch.argsort(scores, descending=True)
    if nms_dist_px > 0.0:
        sorted_points_nms = points[order]
        sorted_scores = scores[order]
        sorted_queries = query_indices[order]
        sorted_valid_masks = None
        if point_valid_scores is not None:
            sorted_valid_masks = torch.stack(
                [
                    longest_contiguous_valid_mask(v >= float(point_valid_thr), min_points=min_points)
                    for v in point_valid_scores[order]
                ],
                dim=0,
            ).to(device=sorted_points_nms.device)
        keep_sorted = lane_nms(
            sorted_points_nms,
            sorted_scores,
            image_shape=image_shape,
            dist_thr_px=float(nms_dist_px),
            valid_masks=sorted_valid_masks,
        )
        points = sorted_points_nms[keep_sorted]
        scores = sorted_scores[keep_sorted]
        query_indices = sorted_queries[keep_sorted]
        if point_valid_scores is not None:
            point_valid_scores = point_valid_scores[order][keep_sorted]
        order = torch.arange(scores.shape[0], dtype=torch.long)
    trace["after_nms_count"] = int(scores.shape[0])

    if valid_before_maxdet and point_valid_scores is not None:
        valid_masks = torch.stack(
            [
                longest_contiguous_valid_mask(v >= float(point_valid_thr), min_points=min_points)
                for v in point_valid_scores
            ],
            dim=0,
        )
        keep_valid = torch.nonzero(valid_masks.sum(dim=1) >= int(min_points), as_tuple=False).flatten()
        points = points[keep_valid]
        scores = scores[keep_valid]
        query_indices = query_indices[keep_valid]
        point_valid_scores = point_valid_scores[keep_valid]
        order = torch.argsort(scores, descending=True)
    trace["after_pre_maxdet_valid_count"] = int(scores.shape[0])

    if max_det is not None and max_det > 0:
        order = order[: int(max_det)]
    points = points[order]
    scores = scores[order]
    query_indices = query_indices[order]
    if point_valid_scores is not None:
        point_valid_scores = point_valid_scores[order]
    trace["after_pool_max_det_count"] = int(points.shape[0])

    scale = torch.tensor([float(image_shape[1]), float(image_shape[0])], dtype=points.dtype).view(1, 1, 2)
    lanes = []
    for lane_i, (lane_points, score, query_idx) in enumerate(zip(points, scores, query_indices)):
        lane_norm = lane_points.numpy().astype(np.float32)
        item = {
            "score": float(score),
            "query": int(query_idx),
            "points_norm": lane_norm,
            "points": (lane_points.unsqueeze(0) * scale).squeeze(0).numpy().astype(np.float32),
        }
        if point_valid_scores is not None:
            visible_mask_t = longest_contiguous_valid_mask(
                point_valid_scores[lane_i] >= float(point_valid_thr),
                min_points=min_points,
            )
            if int(visible_mask_t.sum()) < int(min_points):
                continue
            visible_mask = visible_mask_t.numpy().astype(bool)
            item["point_valid_scores"] = point_valid_scores[lane_i].numpy().astype(np.float32)
            item["point_valid"] = visible_mask.astype(np.float32)
            item["visible_points_norm"] = lane_norm[visible_mask]
            item["visible_points"] = item["points"][visible_mask]
        lanes.append(item)
    trace["final_count_before_export"] = int(len(lanes))
    return lanes, trace


def _best_ape(pred_xs_list: list[np.ndarray], gt_lane: list[float], min_overlap: int) -> tuple[float, int, int]:
    best_ape = float("inf")
    best_idx = -1
    best_overlap = 0
    for idx, pred_xs in enumerate(pred_xs_list):
        ape, overlap = _ape_px(pred_xs, gt_lane, min_overlap=min_overlap)
        if best_idx < 0 or ape < best_ape:
            best_ape = ape
            best_idx = idx
            best_overlap = overlap
    return best_ape, best_idx, best_overlap


def _percentile(values: list[float], pct: float) -> float | None:
    finite = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * float(pct) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    return finite[lo] * (hi - pos) + finite[hi] * (pos - lo)


def _summarize_groups(rows: list[dict], group_fields: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(field, "") for field in group_fields)].append(row)

    out = []
    for key, items in sorted(buckets.items(), key=lambda kv: kv[0]):
        apes = [float(x["raw_best_ape_px"]) for x in items if x.get("raw_best_ape_px") not in ("", None)]
        finite_apes = [x for x in apes if math.isfinite(x)]
        record = {field: key[i] for i, field in enumerate(group_fields)}
        record.update(
            {
                "lanes": int(len(items)),
                "best_ape_px_mean": None if not finite_apes else round(float(sum(finite_apes) / len(finite_apes)), 6),
                "best_ape_px_median": None if not finite_apes else round(float(median(finite_apes)), 6),
                "best_ape_px_p90": None if not finite_apes else round(float(_percentile(finite_apes, 90.0)), 6),
                "has_match_20px": round(
                    sum(1 for x in items if bool(x.get("raw_has_match_20px"))) / max(len(items), 1),
                    6,
                ),
                "has_match_30px": round(
                    sum(1 for x in items if bool(x.get("raw_has_match_30px"))) / max(len(items), 1),
                    6,
                ),
                "has_match_40px": round(
                    sum(1 for x in items if bool(x.get("raw_has_match_40px"))) / max(len(items), 1),
                    6,
                ),
            }
        )
        if any("pred_valid_points@0.6" in x for x in items):
            vals05 = [float(x["pred_valid_points@0.5"]) for x in items if str(x.get("pred_valid_points@0.5", "")) != ""]
            vals06 = [float(x["pred_valid_points@0.6"]) for x in items if str(x.get("pred_valid_points@0.6", "")) != ""]
            rec05 = [float(x["point_valid_recall@0.5"]) for x in items if str(x.get("point_valid_recall@0.5", "")) != ""]
            rec06 = [float(x["point_valid_recall@0.6"]) for x in items if str(x.get("point_valid_recall@0.6", "")) != ""]
            vis = [float(x["visible_points_gt"]) for x in items]
            record.update(
                {
                    "visible_points_gt_mean": round(sum(vis) / max(len(vis), 1), 6),
                    "pred_valid_points@0.5_mean": round(sum(vals05) / max(len(vals05), 1), 6),
                    "pred_valid_points@0.6_mean": round(sum(vals06) / max(len(vals06), 1), 6),
                    "point_valid_recall@0.5_mean": round(sum(rec05) / max(len(rec05), 1), 6),
                    "point_valid_recall@0.6_mean": round(sum(rec06) / max(len(rec06), 1), 6),
                    "pred_valid_points@0.6_lt_min_points": int(
                        sum(1 for x in items if float(x["pred_valid_points@0.6"]) < float(x["min_points"]))
                    ),
                }
            )
        out.append(record)
    return out


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if str(args.split).lower() == "test" and not bool(args.allow_test_oracle):
        raise ValueError(
            "Raw Q12 diagnostic uses GT to judge candidate quality and is blocked on --split test by default. "
            "Run it on official-val for model decisions, or pass --allow-test-oracle only for a predeclared audit."
        )

    archive_root = find_tusimple_archive_root(args.archive_root)
    gt_json = _default_gt_json(args)
    gt_path = resolve_tusimple_gt_json(archive_root, split=args.split, gt_json=gt_json)
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
    if args.decode_yaml:
        _, decode_yaml_cfg = load_decode_yaml(args.decode_yaml)
        decode_mode = resolve_decode_mode(decode_yaml_cfg.get("decode_mode"), model)
        validate_decode_yaml_for_model(decode_yaml_cfg, model_mode=decode_mode)
        if decode_mode != "query":
            raise ValueError(f"Raw Q12 diagnostic supports query decode only, got decode_mode={decode_mode!r}.")
        _apply_query_decode_yaml(args, decode_yaml_cfg)
    else:
        decode_mode = resolve_decode_mode("auto", model)
        if decode_mode != "query":
            raise ValueError(f"Raw Q12 diagnostic supports query decode only, got decode_mode={decode_mode!r}.")

    warn_max_det_mismatch(args.weights, max_det=args.max_det, context="TuSimple raw Q12 diagnostic")
    if int(args.pool_max_det) > 0:
        warn_max_det_mismatch(args.weights, max_det=args.pool_max_det, context="TuSimple raw Q12 diagnostic pool")

    if args.warmup > 0 and gt_records:
        warm_path = tusimple_image_path(archive_root, str(gt_records[0]["raw_file"]), split=args.split)
        warm_img = cv2.imread(str(warm_path), cv2.IMREAD_COLOR)
        if warm_img is None:
            raise FileNotFoundError(f"Failed to read warmup image: {warm_path}")
        warm_tensor = preprocess_image(warm_img, imgsz, device=device_obj, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(warm_tensor)
        _sync_if_cuda(device_obj)

    save_dir = resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    lane_rows: list[dict] = []
    query_rows: list[dict] = []
    image_rows: list[dict] = []
    hard_rows: list[dict] = []
    infer_time_s = 0.0
    post_time_s = 0.0
    match_thrs = sorted({float(x) for x in args.raw_match_thrs} | {20.0, 30.0, 40.0})

    for record in gt_records:
        raw_file = str(record["raw_file"])
        date, session = _parse_date_session(raw_file)
        image_path = tusimple_image_path(archive_root, raw_file, split=args.split)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        original_shape = img.shape[:2]
        h_samples = [float(x) for x in record["h_samples"]]
        tensor = preprocess_image(img, imgsz, device=device_obj, half=args.half)

        _sync_if_cuda(device_obj)
        t0 = time.perf_counter()
        preds = model(tensor)
        _sync_if_cuda(device_obj)
        t1 = time.perf_counter()

        raw_queries = _query_arrays(preds, image_shape=original_shape, h_samples=h_samples, min_points=args.min_points)
        pool_lanes, pool_trace = _trace_decode(
            preds,
            image_shape=original_shape,
            score_thr=args.conf,
            point_valid_thr=args.point_valid_thr,
            min_points=args.min_points,
            max_det=None if int(args.pool_max_det) <= 0 else int(args.pool_max_det),
            nms_dist_px=args.nms_dist_px,
            valid_before_maxdet=True,
        )
        final_lanes, final_trace = _trace_decode(
            preds,
            image_shape=original_shape,
            score_thr=args.conf,
            point_valid_thr=args.point_valid_thr,
            min_points=args.min_points,
            max_det=args.max_det,
            nms_dist_px=args.nms_dist_px,
            valid_before_maxdet=args.valid_before_maxdet,
        )

        gt_lanes = _valid_gt_lanes_with_ids(record)
        gt_count = int(len(valid_tusimple_lanes(record.get("lanes", []))))
        lane_positions = _lane_position_map(record)
        pred_raw_xs = [q["pred_xs"] for q in raw_queries]
        final_pred_xs = [
            _interp_lane_xs(np.asarray(lane["points_norm"], dtype=np.float32), h_samples, original_shape)
            for lane in final_lanes
        ]

        missing_ids = []
        missing_visible = []
        missing_raw_apes = []
        missing_decoded_apes = []

        for gt_lane_id, gt_lane in gt_lanes:
            best_ape, best_query_list_idx, best_overlap = _best_ape(pred_raw_xs, gt_lane, min_overlap=args.match_min_overlap)
            best_query = raw_queries[best_query_list_idx] if best_query_list_idx >= 0 else None
            decoded_best_ape, decoded_best_idx, decoded_best_overlap = _best_ape(
                final_pred_xs,
                gt_lane,
                min_overlap=args.match_min_overlap,
            )
            visible = _visible_count(gt_lane)
            pos = lane_positions.get(int(gt_lane_id), {})
            visible_bucket = _visible_bucket(visible, args.short_visible_max)
            row = {
                "raw_file": raw_file,
                "date": date,
                "session": session,
                "gt_count": gt_count,
                "gt_lane_id": int(gt_lane_id),
                "lane_order": pos.get("lane_order", ""),
                "lane_position": pos.get("lane_position", ""),
                "side_group": pos.get("side_group", ""),
                "visible_points_gt": int(visible),
                "visible_bucket": visible_bucket,
                "short_visible_lane": bool(visible <= int(args.short_visible_max)),
                "raw_best_ape_px": best_ape,
                "raw_best_query_id": "" if best_query is None else int(best_query["query"]),
                "raw_best_exist_score": "" if best_query is None else round(float(best_query["score"]), 8),
                "raw_best_overlap": int(best_overlap),
                "raw_best_valid_len@0.5": "" if best_query is None else int(best_query["valid_len_0.5"]),
                "raw_best_valid_len@0.6": "" if best_query is None else int(best_query["valid_len_0.6"]),
                "raw_best_valid_count@0.5": "" if best_query is None else int(best_query["valid_count_0.5"]),
                "raw_best_valid_count@0.6": "" if best_query is None else int(best_query["valid_count_0.6"]),
                "pred_valid_points@0.5": "" if best_query is None else int(best_query["valid_len_0.5"]),
                "pred_valid_points@0.6": "" if best_query is None else int(best_query["valid_len_0.6"]),
                "point_valid_recall@0.5": ""
                if best_query is None
                else _valid_recall_at_gt(
                    best_query["points_norm"],
                    best_query["valid_scores"],
                    h_samples,
                    gt_lane,
                    original_shape,
                    0.5,
                ),
                "point_valid_recall@0.6": ""
                if best_query is None
                else _valid_recall_at_gt(
                    best_query["points_norm"],
                    best_query["valid_scores"],
                    h_samples,
                    gt_lane,
                    original_shape,
                    0.6,
                ),
                "best_decoded_ape_px": decoded_best_ape,
                "best_decoded_lane_index": decoded_best_idx,
                "best_decoded_overlap": decoded_best_overlap,
                "missing_at_match_thr": bool(decoded_best_ape > float(args.match_thr_px)),
                "min_points": int(args.min_points),
            }
            for thr in match_thrs:
                key = str(int(thr)) if float(thr).is_integer() else str(thr).replace(".", "p")
                row[f"raw_has_match_{key}px"] = bool(best_ape <= float(thr))
            lane_rows.append(row)

            if bool(row["missing_at_match_thr"]):
                missing_ids.append(int(gt_lane_id))
                missing_visible.append(int(visible))
                missing_raw_apes.append(best_ape)
                missing_decoded_apes.append(decoded_best_ape)

            for query in raw_queries:
                ape, overlap = _ape_px(query["pred_xs"], gt_lane, min_overlap=args.match_min_overlap)
                query_rows.append(
                    {
                        "raw_file": raw_file,
                        "date": date,
                        "session": session,
                        "gt_count": gt_count,
                        "gt_lane_id": int(gt_lane_id),
                        "lane_position": pos.get("lane_position", ""),
                        "side_group": pos.get("side_group", ""),
                        "visible_points_gt": int(visible),
                        "query": int(query["query"]),
                        "ape_px": ape,
                        "overlap": int(overlap),
                        "exist_score": round(float(query["score"]), 8),
                        "valid_len@0.5": int(query["valid_len_0.5"]),
                        "valid_len@0.6": int(query["valid_len_0.6"]),
                        "valid_count@0.5": int(query["valid_count_0.5"]),
                        "valid_count@0.6": int(query["valid_count_0.6"]),
                    }
                )

        image_row = {
            "raw_file": raw_file,
            "date": date,
            "session": session,
            "gt_count": gt_count,
            "raw_query_count": int(pool_trace["raw_query_count"]),
            "after_score_count": int(pool_trace["after_score_count"]),
            "after_point_valid_min_points_count": int(pool_trace["after_point_valid_min_points_count"]),
            "after_nms_count": int(pool_trace["after_nms_count"]),
            "after_pre_maxdet_valid_count": int(pool_trace["after_pre_maxdet_valid_count"]),
            "after_pool_max_det_count": int(pool_trace["after_pool_max_det_count"]),
            "final_count": int(final_trace["final_count_before_export"]),
            "missing_gt_lane_ids": missing_ids,
            "missing_gt_lane_visible_points": missing_visible,
            "best_raw_ape_per_missing_gt": [round(float(x), 6) if math.isfinite(float(x)) else "inf" for x in missing_raw_apes],
            "best_decoded_ape_per_missing_gt": [
                round(float(x), 6) if math.isfinite(float(x)) else "inf" for x in missing_decoded_apes
            ],
        }
        image_rows.append(image_row)
        if int(image_row["after_pool_max_det_count"]) < int(gt_count):
            hard_rows.append(image_row)

        t2 = time.perf_counter()
        infer_time_s += t1 - t0
        post_time_s += t2 - t1

    lane_fields = [
        "raw_file",
        "date",
        "session",
        "gt_count",
        "gt_lane_id",
        "lane_order",
        "lane_position",
        "side_group",
        "visible_points_gt",
        "visible_bucket",
        "short_visible_lane",
        "raw_best_ape_px",
        "raw_best_query_id",
        "raw_best_exist_score",
        "raw_best_overlap",
        "raw_best_valid_len@0.5",
        "raw_best_valid_len@0.6",
        "raw_best_valid_count@0.5",
        "raw_best_valid_count@0.6",
        "raw_has_match_20px",
        "raw_has_match_30px",
        "raw_has_match_40px",
        "pred_valid_points@0.5",
        "pred_valid_points@0.6",
        "point_valid_recall@0.5",
        "point_valid_recall@0.6",
        "best_decoded_ape_px",
        "best_decoded_lane_index",
        "best_decoded_overlap",
        "missing_at_match_thr",
        "min_points",
    ]
    image_fields = [
        "raw_file",
        "date",
        "session",
        "gt_count",
        "raw_query_count",
        "after_score_count",
        "after_point_valid_min_points_count",
        "after_nms_count",
        "after_pre_maxdet_valid_count",
        "after_pool_max_det_count",
        "final_count",
        "missing_gt_lane_ids",
        "missing_gt_lane_visible_points",
        "best_raw_ape_per_missing_gt",
        "best_decoded_ape_per_missing_gt",
    ]
    query_fields = [
        "raw_file",
        "date",
        "session",
        "gt_count",
        "gt_lane_id",
        "lane_position",
        "side_group",
        "visible_points_gt",
        "query",
        "ape_px",
        "overlap",
        "exist_score",
        "valid_len@0.5",
        "valid_len@0.6",
        "valid_count@0.5",
        "valid_count@0.6",
    ]
    _write_csv(save_dir / "raw_gt_lane_diagnostics.csv", lane_rows, lane_fields)
    _write_csv(save_dir / "raw_query_gt_ape_long.csv", query_rows, query_fields)
    _write_csv(save_dir / "per_image_filter_trace.csv", image_rows, image_fields)
    _write_csv(save_dir / "hard_images_candidate_short.csv", hard_rows, image_fields)

    point_valid_groups = _summarize_groups(lane_rows, ["gt_count", "lane_position"])
    point_valid_groups += _summarize_groups(lane_rows, ["gt_count", "side_group", "visible_bucket"])
    raw_ape_groups = []
    raw_ape_groups += _summarize_groups(lane_rows, ["gt_count"])
    raw_ape_groups += _summarize_groups(lane_rows, ["gt_count", "short_visible_lane"])
    raw_ape_groups += _summarize_groups(lane_rows, ["gt_count", "lane_position"])
    raw_ape_groups += _summarize_groups(lane_rows, ["date", "session"])
    raw_ape_groups += _summarize_groups(lane_rows, ["date", "gt_count", "visible_bucket"])

    group_fields = [
        "gt_count",
        "lane_position",
        "side_group",
        "visible_bucket",
        "short_visible_lane",
        "date",
        "session",
        "lanes",
        "visible_points_gt_mean",
        "pred_valid_points@0.5_mean",
        "pred_valid_points@0.6_mean",
        "point_valid_recall@0.5_mean",
        "point_valid_recall@0.6_mean",
        "pred_valid_points@0.6_lt_min_points",
        "best_ape_px_mean",
        "best_ape_px_median",
        "best_ape_px_p90",
        "has_match_20px",
        "has_match_30px",
        "has_match_40px",
    ]
    _write_csv(save_dir / "point_valid_group_summary.csv", point_valid_groups, group_fields)
    _write_csv(save_dir / "raw_ape_group_summary.csv", raw_ape_groups, group_fields)

    apes = [float(row["raw_best_ape_px"]) for row in lane_rows if math.isfinite(float(row["raw_best_ape_px"]))]
    n_lanes = max(len(lane_rows), 1)
    summary = {
        "config": {
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
            "pool_max_det": None if int(args.pool_max_det) <= 0 else int(args.pool_max_det),
            "match_thr_px": float(args.match_thr_px),
            "match_min_overlap": int(args.match_min_overlap),
            "short_visible_max": int(args.short_visible_max),
            "max_images": int(args.max_images),
            "device": str(args.device),
            "half": bool(args.half),
        },
        "raw_q12": {
            "gt_lanes": int(len(lane_rows)),
            "best_ape_px_mean": None if not apes else round(float(sum(apes) / len(apes)), 6),
            "best_ape_px_median": None if not apes else round(float(median(apes)), 6),
            "best_ape_px_p90": None if not apes else round(float(_percentile(apes, 90.0)), 6),
            "has_match_20px": round(sum(1 for x in lane_rows if bool(x.get("raw_has_match_20px"))) / n_lanes, 6),
            "has_match_30px": round(sum(1 for x in lane_rows if bool(x.get("raw_has_match_30px"))) / n_lanes, 6),
            "has_match_40px": round(sum(1 for x in lane_rows if bool(x.get("raw_has_match_40px"))) / n_lanes, 6),
        },
        "filter_trace": {
            "images": int(len(image_rows)),
            "candidate_count_lt_gt_count_images": int(len(hard_rows)),
            "final_count_lt_gt_count_images": int(sum(1 for x in image_rows if int(x["final_count"]) < int(x["gt_count"]))),
        },
        "outputs": {
            "raw_gt_lane_diagnostics": str((save_dir / "raw_gt_lane_diagnostics.csv").resolve()),
            "raw_query_gt_ape_long": str((save_dir / "raw_query_gt_ape_long.csv").resolve()),
            "per_image_filter_trace": str((save_dir / "per_image_filter_trace.csv").resolve()),
            "hard_images_candidate_short": str((save_dir / "hard_images_candidate_short.csv").resolve()),
            "point_valid_group_summary": str((save_dir / "point_valid_group_summary.csv").resolve()),
            "raw_ape_group_summary": str((save_dir / "raw_ape_group_summary.csv").resolve()),
        },
        "timing": {
            "avg_inference_ms": round(infer_time_s * 1000.0 / max(len(image_rows), 1), 4),
            "avg_postprocess_ms": round(post_time_s * 1000.0 / max(len(image_rows), 1), 4),
            "avg_total_ms": round((infer_time_s + post_time_s) * 1000.0 / max(len(image_rows), 1), 4),
        },
        **gt_contract,
    }
    (save_dir / "raw_q12_filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["raw_q12"] | summary["filter_trace"], indent=2))
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
