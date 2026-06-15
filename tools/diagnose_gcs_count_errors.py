from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tools.eval_gcs import (  # noqa: E402
    assert_label_fixed_y_compatible,
    dataset_defaults,
    default_data_yaml,
    label_path_for_image,
    labels_from_source,
    load_gcs_label,
    model_fixed_y_anchors,
    resolve_dataset,
)
from tools.infer_gcs import collect_images, load_gcs_model, preprocess_image  # noqa: E402
from ultralytics.utils.gcs_candidate_matching import GCSLaneCandidate  # noqa: E402
from ultralytics.utils.gcs_count_diagnostics import (  # noqa: E402
    build_candidates_from_predictions,
    diagnose_count_errors,
    write_count_diagnostics,
)
from ultralytics.utils.gcs_postprocess import GCS_DEFAULT_MAX_DET, decode_gcs_predictions  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose GCS lane count errors with A/B/C/D/E/F buckets.")
    parser.add_argument("--dataset", default="tusimple")
    parser.add_argument("--data", default=str(default_data_yaml("tusimple")))
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--labels", default="")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960])
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--out", "--diagnostic-out", dest="out", default="runs/gcs_lane/count_diagnostics")
    parser.add_argument("--diagnostic-topk", type=int, default=8)
    parser.add_argument("--diagnostic-match-thr", type=float, default=0.5)
    parser.add_argument("--diagnostic-official-acc-thr", type=float, default=0.85)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--point-valid-thr", type=float, default=0.5)
    parser.add_argument("--nms-dist-px", type=float, default=18.0)
    parser.add_argument("--max-det", type=int, default=GCS_DEFAULT_MAX_DET)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--normal-candidate-score-thr", type=float, default=0.03)
    parser.add_argument("--normal-point-valid-thr", type=float, default=0.15)
    parser.add_argument("--normal-min-points", type=int, default=5)
    parser.add_argument("--rescue-candidate-score-thr", type=float, default=0.015)
    parser.add_argument("--rescue-point-valid-thr", type=float, default=0.08)
    parser.add_argument("--rescue-min-points", type=int, default=4)
    parser.add_argument("--rescue-geometry-distinct-thr", type=float, default=0.65)
    parser.add_argument("--line-nms-min-overlap", type=int, default=6)
    parser.add_argument("--line-nms-rescue-dist-px", type=float, default=30.0)
    parser.add_argument("--enable-count-aware-refill", action="store_true")
    parser.add_argument("--refill-max-extra", type=int, default=2)
    parser.add_argument("--refill-require-distinct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refill-allow-nms-suppressed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--soft-count-decision", action="store_true")
    parser.add_argument("--soft-count-prob-margin", type=float, default=0.08)
    parser.add_argument("--soft-count-quality-weight", type=float, default=1.0)
    parser.add_argument("--soft-count-prior-weight", type=float, default=0.5)
    parser.add_argument("--soft-count-duplicate-penalty", type=float, default=1.0)
    parser.add_argument("--soft-count-invalid-penalty", type=float, default=1.0)
    parser.add_argument("--write-hard-samples", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _candidate_from_decoded(image_id: str, lane: dict) -> GCSLaneCandidate:
    points = np.asarray(lane.get("points_norm"), dtype=np.float32)
    valid_probs = np.asarray(lane.get("point_valid_scores", lane.get("point_valid", np.ones(points.shape[0]))), dtype=np.float32)
    valid_count = int(lane.get("valid_count", int((valid_probs > 0.5).sum())))
    return GCSLaneCandidate(
        image_id=image_id,
        query_idx=int(lane.get("query", lane.get("candidate_id", -1))),
        points=points,
        valid_probs=valid_probs,
        exist_logit=0.0,
        exist_score=float(lane.get("exist_score", lane.get("score", 0.0))),
        point_valid_mean=float(valid_probs.mean()) if valid_probs.size else 0.0,
        point_valid_max=float(valid_probs.max()) if valid_probs.size else 0.0,
        valid_points=valid_count,
        lane_quality=float(lane.get("quality_score", lane.get("rank_score", lane.get("score", 0.0)))),
        pre_nms_rank=int(lane.get("rank_selection_rank", 0)),
        pre_nms_score=float(lane.get("rank_score", lane.get("score", 0.0))),
        keep_after_nms=True,
        source=str(lane.get("source", "final")),
    )


def final_candidates_from_decoded(image_id: str, decoded: list[dict], candidates: list[GCSLaneCandidate]) -> list[GCSLaneCandidate]:
    by_query = {int(c.query_idx): c for c in candidates}
    final: list[GCSLaneCandidate] = []
    for lane in decoded:
        q = int(lane.get("query", lane.get("candidate_id", -1)))
        if q in by_query:
            cand = by_query[q]
            final.append(
                GCSLaneCandidate(
                    **{
                        **cand.__dict__,
                        "keep_after_nms": True,
                        "source": str(lane.get("source", cand.source)),
                    }
                )
            )
        else:
            final.append(_candidate_from_decoded(image_id, lane))
    return final


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    data = resolve_dataset(args.data, args.dataset) if args.data else None
    defaults = dataset_defaults(args.dataset, split=args.split)
    source = Path(args.source) if args.source else Path(data.get(args.split) if data else defaults["source"])
    labels = Path(args.labels) if args.labels else Path(labels_from_source(source) or defaults["labels"])
    imgsz = normalize_imgsz(args.imgsz or (data.get("gcs_imgsz") if data else None), dataset=args.dataset)
    device = select_device(args.device, verbose=False)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=imgsz)
    expected_fixed_y = model_fixed_y_anchors(model)
    images = collect_images(source, max_images=args.max_images)
    print(f"GCS count diagnostics: {len(images)} images, input {shape_str(imgsz)} (H,W={imgsz})")

    if args.warmup > 0 and images:
        img0 = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        tensor0 = preprocess_image(img0, imgsz=imgsz, device=device, half=args.half)
        for _ in range(int(args.warmup)):
            _ = model(tensor0)

    rows = []
    for idx, image_path in enumerate(images, start=1):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        label_path = label_path_for_image(image_path, labels)
        assert_label_fixed_y_compatible(label_path, expected_fixed_y, image_shape=img.shape[:2])
        gt_lanes, gt_valid = load_gcs_label(label_path)
        tensor = preprocess_image(img, imgsz=imgsz, device=device, half=args.half)
        preds = model(tensor)

        pred_valid = preds.get("pred_valid_logits")
        pred_quality = preds.get("pred_quality_logits")
        pred_count = preds.get("pred_count_logits")
        pred_count_boundary = preds.get("pred_count_boundary_logits")
        pred_points = preds["pred_points"][0].detach().float()
        pred_logits = preds["pred_logits"][0].detach().float()
        pred_valid_0 = pred_valid[0].detach().float() if pred_valid is not None else None
        pred_quality_0 = pred_quality[0].detach().float() if pred_quality is not None else None
        pred_count_0 = pred_count[0].detach().float() if pred_count is not None else None
        pred_count_boundary_0 = (
            pred_count_boundary[0].detach().float() if pred_count_boundary is not None else None
        )

        image_id = str(image_path)
        candidates = build_candidates_from_predictions(
            image_id=image_id,
            pred_points=pred_points,
            pred_logits=pred_logits,
            pred_valid_logits=pred_valid_0,
            pred_quality_logits=pred_quality_0,
            image_shape=img.shape[:2],
            normal_candidate_score_thr=args.normal_candidate_score_thr,
            normal_point_valid_thr=args.normal_point_valid_thr,
            normal_min_points=args.normal_min_points,
            rescue_candidate_score_thr=args.rescue_candidate_score_thr,
            rescue_point_valid_thr=args.rescue_point_valid_thr,
            rescue_min_points=args.rescue_min_points,
            nms_dist_px=args.nms_dist_px,
            line_nms_min_overlap=args.line_nms_min_overlap,
        )
        decoded, decode_meta = decode_gcs_predictions(
            pred_points,
            pred_logits,
            pred_valid_logits=pred_valid_0,
            pred_count_logits=pred_count_0,
            pred_count_boundary_logits=pred_count_boundary_0,
            pred_quality_logits=pred_quality_0,
            image_shape=img.shape[:2],
            score_thr=args.conf,
            point_valid_thr=args.point_valid_thr,
            min_points=args.min_points,
            max_det=args.max_det,
            nms_dist_px=args.nms_dist_px,
            use_count_head_decode=pred_count_0 is not None,
            candidate_score_thr=args.normal_candidate_score_thr,
            candidate_point_valid_thr=args.normal_point_valid_thr,
            candidate_min_points=args.normal_min_points,
            enable_rescue_candidate_pool=bool(args.enable_count_aware_refill),
            rescue_candidate_score_thr=args.rescue_candidate_score_thr,
            rescue_candidate_point_valid_thr=args.rescue_point_valid_thr,
            rescue_candidate_min_points=args.rescue_min_points,
            final_min_points=args.min_points,
            fifth_min_points=min(args.min_points, 5),
            line_nms_min_overlap=args.line_nms_min_overlap,
            line_nms_rescue_dist_px=args.line_nms_rescue_dist_px,
            enable_soft_count_decision=bool(args.soft_count_decision),
            soft_count_prob_margin=args.soft_count_prob_margin,
            soft_count_quality_weight=args.soft_count_quality_weight,
            soft_count_prior_weight=args.soft_count_prior_weight,
            soft_count_duplicate_penalty=args.soft_count_duplicate_penalty,
            soft_count_invalid_penalty=args.soft_count_invalid_penalty,
            return_meta=True,
        )
        final = final_candidates_from_decoded(image_id, decoded, candidates)
        row = diagnose_count_errors(
            image_id=image_id,
            gt_lanes=gt_lanes,
            gt_valid=gt_valid,
            candidates=candidates,
            final_candidates=final,
            pred_count_logits=pred_count_0,
            diagnostic_topk=args.diagnostic_topk,
            diagnostic_match_thr=args.diagnostic_match_thr,
            image_shape=img.shape[:2],
            normal_min_points=args.normal_min_points,
        )
        row.update({f"decode_{k}": v for k, v in decode_meta.items() if isinstance(v, (int, float, bool, str))})
        rows.append(row)
        if idx % 100 == 0 or idx == len(images):
            print(f"processed {idx}/{len(images)}")

    summary = write_count_diagnostics(
        rows,
        args.out,
        diagnostic_topk=args.diagnostic_topk,
        write_hard_samples=bool(args.write_hard_samples),
    )
    print(json.dumps(summary, indent=2))
    print(f"saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
