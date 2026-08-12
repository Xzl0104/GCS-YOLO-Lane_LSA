"""Audit short-proposal outputs against TuSimple GT without changing official decode."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from gcs_tools.tusimple_official_eval import read_tusimple_json_lines, tusimple_image_path, valid_tusimple_lanes
from tools.infer_gcs import load_gcs_model, preprocess_image
from ultralytics.utils.torch_utils import select_device


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--archive-root", required=True)
    p.add_argument("--gt-json", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--imgsz", nargs=2, type=int, default=(544, 960))
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true")
    p.add_argument("--hit-px", type=float, default=20.0)
    p.add_argument("--hit40-px", type=float, default=40.0)
    p.add_argument("--valid-thr", type=float, default=0.5)
    p.add_argument("--short-visible-thr", type=int, default=10)
    p.add_argument("--max-images", type=int, default=0)
    return p.parse_args()


def ape_matrix(pred_x, gt_x, valid):
    if not valid.any():
        return np.full(pred_x.shape[0], np.inf, dtype=np.float32)
    return np.abs(pred_x[:, valid] - gt_x[valid][None]).mean(axis=1)


def main():
    args = args_parser()
    records = read_tusimple_json_lines(args.gt_json)
    if args.max_images > 0:
        records = records[:args.max_images]
    device = select_device(args.device)
    model = load_gcs_model(args.weights, device=device, half=args.half, gcs_imgsz=tuple(args.imgsz))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    totals = {k: 0 for k in ("images", "lanes", "short_lanes", "gt5_short", "base_hit20", "base_hit40", "proposal_hit20", "proposal_hit40", "proposal_valid_hit20", "proposal_on_base_miss20", "combined_oracle20", "combined_oracle40")}
    image_rows, lane_rows = [], []
    duplicate_images = capacity_overflow_images = 0
    duplicate_pairs = proposal_pairs = 0
    with torch.inference_mode():
        for record in records:
            image_path = tusimple_image_path(args.archive_root, record["raw_file"], split=args.split)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            outputs = model(preprocess_image(image, args.imgsz, device, args.half))
            required = ("pred_short_proposal_points", "pred_short_proposal_logits", "pred_short_proposal_valid_logits")
            missing = [key for key in required if key not in outputs]
            if missing:
                raise KeyError(f"Checkpoint lacks proposal outputs: {missing}")
            base_x = outputs["pred_points"][0, :, :, 0].float().cpu().numpy() * image.shape[1]
            proposal_x = outputs["pred_short_proposal_points"][0, :, :, 0].float().cpu().numpy() * image.shape[1]
            proposal_valid = outputs["pred_short_proposal_valid_logits"][0].float().sigmoid().cpu().numpy()
            proposal_score = outputs["pred_short_proposal_logits"][0].float().sigmoid().cpu().numpy()
            model_y = outputs["pred_points"][0, 0, :, 1].float().cpu().numpy() * image.shape[0]
            lanes = valid_tusimple_lanes(record.get("lanes", []))
            h_samples = [int(v) for v in record.get("h_samples", [])]
            image_row = {"raw_file": record["raw_file"], "gt_count": len(lanes), "short_gt_count": 0, "base_miss_short": 0, "proposal_hit_short": 0, "combined_oracle_hit": 0}
            image_combined_hits = 0
            valid_candidate_count = 0
            lane_apes = []
            for lane_index, lane in enumerate(lanes):
                lane_by_y = {int(y): float(x) for y, x in zip(h_samples, lane)}
                gt_x = np.asarray([lane_by_y.get(int(y), -2.0) for y in model_y], dtype=np.float32)
                valid = gt_x >= 0.0
                visible = int(valid.sum())
                if visible < 2:
                    continue
                totals["lanes"] += 1
                is_short = visible <= args.short_visible_thr
                is_gt5_short = len(lanes) == 5 and is_short
                totals["short_lanes"] += int(is_short)
                totals["gt5_short"] += int(is_gt5_short)
                base_ape = ape_matrix(base_x, gt_x, valid)
                prop_ape = ape_matrix(proposal_x, gt_x, valid)
                base20 = bool(base_ape.min() <= args.hit_px)
                base40 = bool(base_ape.min() <= args.hit40_px)
                prop20 = bool(prop_ape.min() <= args.hit_px)
                prop40 = bool(prop_ape.min() <= args.hit40_px)
                prop_valid20 = bool(((prop_ape <= args.hit_px) & ((proposal_valid[:, valid] >= args.valid_thr).all(axis=1))).any())
                if is_short:
                    totals["base_hit20"] += int(base20); totals["base_hit40"] += int(base40)
                    totals["proposal_hit20"] += int(prop20); totals["proposal_hit40"] += int(prop40)
                    totals["proposal_valid_hit20"] += int(prop_valid20)
                    totals["proposal_on_base_miss20"] += int((not base20) and prop20)
                    image_row["short_gt_count"] += 1
                    image_row["base_miss_short"] += int(not base20)
                    image_row["proposal_hit_short"] += int(prop20)
                lane_apes.append(np.concatenate([base_ape, prop_ape]))
                lane_rows.append({"raw_file": record["raw_file"], "lane_index": lane_index, "gt_count": len(lanes), "visible_points": visible, "short": is_short, "gt5_short": is_gt5_short, "base_ape20": float(base_ape.min()), "proposal_ape20": float(prop_ape.min()), "proposal_hit20": prop20, "proposal_hit40": prop40, "proposal_valid_hit20": prop_valid20, "proposal_scores": proposal_score.round(6).tolist()})
            if image_row["short_gt_count"]:
                short_apes = [r for r in lane_rows if r["raw_file"] == record["raw_file"] and r["short"]]
                base_miss = [r for r in short_apes if r["base_ape20"] > args.hit_px]
                if base_miss:
                    image_row["proposal_rescues"] = sum(r["proposal_hit20"] for r in base_miss)
            candidate_scores = np.concatenate([outputs["pred_logits"][0].float().sigmoid().cpu().numpy(), proposal_score])
            candidate_count = int((candidate_scores >= 0.2).sum())
            valid_candidate_count = candidate_count
            capacity_overflow_images += int(len(lanes) == 5 and candidate_count > 5)
            p = proposal_x.shape[0]
            for left in range(p):
                for right in range(left + 1, p):
                    overlap = np.ones(proposal_x.shape[1], dtype=bool)
                    d = float(np.abs(proposal_x[left, overlap] - proposal_x[right, overlap]).mean())
                    proposal_pairs += 1; duplicate_pairs += int(d <= args.hit_px)
            duplicate_images += int(proposal_pairs and duplicate_pairs / proposal_pairs > 0.25)
            image_rows.append({**image_row, "proposal_candidate_count_conf02": valid_candidate_count, "candidate_overflow_gt5": bool(len(lanes) == 5 and candidate_count > 5)})
    short = totals["short_lanes"] or 1
    gt5 = totals["gt5_short"] or 1
    summary = {"weights": str(Path(args.weights).resolve()), "gt_json": str(Path(args.gt_json).resolve()), "images": len(records), "totals": totals, "short_rates": {"base_hit20": totals["base_hit20"] / short, "proposal_hit20": totals["proposal_hit20"] / short, "proposal_hit40": totals["proposal_hit40"] / short, "proposal_valid_hit20": totals["proposal_valid_hit20"] / short, "proposal_rescue20_on_base_miss": totals["proposal_on_base_miss20"] / max(1, totals["short_lanes"] - totals["base_hit20"])}, "gt5_short_rates": {"count": totals["gt5_short"], "proposal_hit20": sum(r["proposal_hit20"] for r in lane_rows if r["gt5_short"]) / gt5, "proposal_hit40": sum(r["proposal_hit40"] for r in lane_rows if r["gt5_short"]) / gt5, "proposal_valid_hit20": sum(r["proposal_valid_hit20"] for r in lane_rows if r["gt5_short"]) / gt5}, "duplicate": {"proposal_pairs": proposal_pairs, "duplicate_pairs_ape20": duplicate_pairs, "duplicate_pair_rate": duplicate_pairs / max(1, proposal_pairs), "duplicate_images_over25pct": duplicate_images}, "capacity": {"gt5_images_with_base_plus_proposal_conf02_over5": capacity_overflow_images}}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (save_dir / "images.json").write_text(json.dumps(image_rows, indent=2), encoding="utf-8")
    (save_dir / "lanes.json").write_text(json.dumps(lane_rows, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
