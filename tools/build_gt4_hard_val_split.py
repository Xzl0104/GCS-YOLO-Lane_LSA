from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from diagnose_gt4_missing_lane_raw_queries import run_diagnostic  # noqa: E402
from ultralytics.utils.gcs_shape import normalize_imgsz, shape_str  # noqa: E402


DEFAULT_LABEL_DIR = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544" / "labels_gcs" / "train"
DEFAULT_ARCHIVE = ROOT / "archive"
DEFAULT_OUT_FILE = ROOT / "data" / "tusimple_gt4_hard_val.txt"
DEFAULT_SAVE_DIR = ROOT / "runs" / "gcs_lane" / "gt4_hard_val_build"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an internal GT4-hard validation sample list from train labels by running the "
            "GT4 missing-lane raw-query diagnostic on GT4 train images."
        )
    )
    parser.add_argument("--label-dir", default=str(DEFAULT_LABEL_DIR), help="GCS labels_gcs/train directory.")
    parser.add_argument("--weights", required=True, help="GCS checkpoint .pt used for the diagnostic.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE), help="Path to archive/ or archive/TUSimple.")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_FILE), help="Output txt file of hard sample raw_file ids.")
    parser.add_argument("--summary-json", default=None, help="Output summary JSON. Defaults beside --out-file.")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR), help="Directory for diagnostic artifacts.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[544, 960], help="GCS inference shape as H W.")
    parser.add_argument("--conf", type=float, default=0.005, help="Normal decode lane existence threshold.")
    parser.add_argument("--point-valid-thr", type=float, default=0.5, help="Normal decode point-valid threshold.")
    parser.add_argument("--nms-dist-px", type=float, default=0.0, help="Normal decode Lane-NMS threshold in original-image px.")
    parser.add_argument("--max-det", type=int, default=8, help="Normal decode max kept lanes.")
    parser.add_argument("--min-points", type=int, default=6, help="Normal decode minimum visible points.")
    parser.add_argument("--only-count-pair", default="4->3", help="Hard-count pair to select. Default: 4->3.")
    parser.add_argument("--match-overlap", type=int, default=3, help="Minimum common h_samples for matching.")
    parser.add_argument("--match-x-thr", type=float, default=20.0, help="Max mean absolute x error in px for matching.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit GT4 source records before diagnostic. 0 means all.")
    parser.add_argument("--hard-limit", type=int, default=0, help="Limit written hard samples after diagnostic. 0 means all.")
    parser.add_argument("--original-height", type=int, default=720, help="Original TuSimple image height for h_samples.")
    parser.add_argument("--original-width", type=int, default=1280, help="Original TuSimple image width for lane x values.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    parser.add_argument("--warmup", type=int, default=0, help="Number of warmup forwards on the first processed image.")
    parser.add_argument("--vis-limit", type=int, default=50, help="Save visualizations for the first N hard images.")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization output.")
    return parser.parse_args()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def scalar_str(value: np.ndarray) -> str:
    item = np.asarray(value).reshape(-1)[0].item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def label_lane_count(data: np.lib.npyio.NpzFile) -> int:
    if "num_lanes" in data:
        return int(np.asarray(data["num_lanes"]).reshape(-1)[0])
    lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
    return int((lane_valid.sum(axis=1) >= 2).sum())


def fixed_y_for_label(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "fixed_y" in data:
        return np.asarray(data["fixed_y"], dtype=np.float32).reshape(-1)
    lanes = np.asarray(data["lanes"], dtype=np.float32)
    if lanes.ndim == 3 and lanes.shape[0] > 0:
        return lanes[0, :, 1].astype(np.float32)
    raise KeyError("fixed_y labels must include fixed_y or non-empty lanes with y anchors.")


def label_to_tusimple_record(label_file: Path, original_width: int, original_height: int) -> dict | None:
    with np.load(label_file, allow_pickle=False) as data:
        if label_lane_count(data) != 4:
            return None
        if "raw_file" not in data:
            raise KeyError(f"{label_file} is missing raw_file; cannot build TuSimple train diagnostic records.")
        lanes = np.asarray(data["lanes"], dtype=np.float32)
        lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
        fixed_y = fixed_y_for_label(data)
        raw_file = scalar_str(data["raw_file"])

    h_samples = [int(round(float(y) * float(original_height))) for y in fixed_y]
    official_lanes: list[list[float]] = []
    for lane, valid in zip(lanes, lane_valid):
        xs = []
        for point, is_valid in zip(lane, valid):
            xs.append(round(float(point[0]) * float(original_width), 3) if float(is_valid) > 0.5 else -2)
        if any(float(x) >= 0.0 for x in xs):
            official_lanes.append(xs)
    if len(official_lanes) != 4:
        return None
    return {"lanes": official_lanes, "h_samples": h_samples, "raw_file": raw_file}


def build_gt_json(label_dir: Path, out_path: Path, original_width: int, original_height: int) -> tuple[list[dict], dict]:
    records = []
    label_files = sorted(label_dir.glob("*.npz"))
    hist = Counter()
    for label_file in label_files:
        with np.load(label_file, allow_pickle=False) as data:
            hist[label_lane_count(data)] += 1
        record = label_to_tusimple_record(label_file, original_width=original_width, original_height=original_height)
        if record is not None:
            records.append(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return records, {"labels": len(label_files), "lane_count_hist": {str(k): int(v) for k, v in sorted(hist.items())}}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_txt(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def write_json(path: Path, value: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_hard_outputs(args: argparse.Namespace, diagnostic_summary: dict) -> dict:
    artifacts = diagnostic_summary.get("artifacts", {})
    image_rows = read_csv_rows(Path(artifacts.get("per_image_summary_csv", "")))
    missing_rows = read_csv_rows(Path(artifacts.get("per_missing_lane_csv", "")))
    missing_by_raw: dict[str, list[dict]] = defaultdict(list)
    for row in missing_rows:
        missing_by_raw[str(row.get("raw_file", ""))].append(row)

    hard_rows = []
    for row in image_rows:
        raw_file = str(row.get("raw_file", ""))
        reasons = Counter(str(x.get("drop_reason", "")) for x in missing_by_raw.get(raw_file, []))
        hard_rows.append(
            {
                "raw_file": raw_file,
                "image_path": row.get("image_path"),
                "gt_count": int(row.get("gt_count") or 0),
                "pred_count": int(row.get("pred_count") or 0),
                "count_pair": row.get("count_pair"),
                "missing_gt_lanes": int(row.get("missing_gt_lanes") or 0),
                "matched_gt_lanes": int(row.get("matched_gt_lanes") or 0),
                "missing_gt_lane_ids": row.get("missing_gt_lane_ids"),
                "drop_reason_histogram": {str(k): int(v) for k, v in sorted(reasons.items()) if k},
            }
        )

    if int(args.hard_limit) > 0:
        hard_rows = hard_rows[: int(args.hard_limit)]

    out_file = resolve_repo_path(args.out_file)
    hard_json = out_file.with_suffix(".json")
    write_txt(out_file, [row["raw_file"] for row in hard_rows])
    write_json(hard_json, hard_rows)

    summary_json = resolve_repo_path(args.summary_json) if args.summary_json else out_file.parent / "gt4_hard_val_summary.json"
    summary = {
        "purpose": "internal GT4-hard validation list; do not feed these samples back into training",
        "hard_sample_count": int(len(hard_rows)),
        "hard_raw_files_txt": str(out_file.resolve()),
        "hard_samples_json": str(hard_json.resolve()),
        "diagnostic_summary_json": artifacts.get("summary_json"),
        "diagnostic": {
            "selected_images": diagnostic_summary.get("selected_images"),
            "total_missing_gt_lanes": diagnostic_summary.get("total_missing_gt_lanes"),
            "drop_reason_histogram": diagnostic_summary.get("drop_reason_histogram"),
            "stage_recall": diagnostic_summary.get("stage_recall"),
        },
        "config": {
            "label_dir": str(resolve_repo_path(args.label_dir).resolve()),
            "weights": str(resolve_repo_path(args.weights).resolve()),
            "archive_root": str(resolve_repo_path(args.archive_root).resolve()),
            "out_file": str(out_file.resolve()),
            "summary_json": str(summary_json.resolve()),
            "only_count_pair": str(args.only_count_pair),
            "hard_limit": int(args.hard_limit),
        },
    }
    write_json(summary_json, summary)
    return summary


def main() -> None:
    args = parse_args()
    label_dir = resolve_repo_path(args.label_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label dir: {label_dir}")

    imgsz = normalize_imgsz(args.imgsz, dataset="tusimple")
    save_dir = resolve_repo_path(args.save_dir)
    gt_json = save_dir / "gt4_train_records.json"
    records, source_summary = build_gt_json(
        label_dir,
        gt_json,
        original_width=int(args.original_width),
        original_height=int(args.original_height),
    )
    if not records:
        raise ValueError(f"No GT4 train records could be built from {label_dir}.")

    diag_args = argparse.Namespace(
        weights=str(resolve_repo_path(args.weights)),
        split="train",
        archive_root=str(resolve_repo_path(args.archive_root)),
        gt_json=str(gt_json),
        imgsz=[int(imgsz[0]), int(imgsz[1])],
        conf=float(args.conf),
        point_valid_thr=float(args.point_valid_thr),
        nms_dist_px=float(args.nms_dist_px),
        max_det=int(args.max_det),
        min_points=int(args.min_points),
        only_count_pair=args.only_count_pair,
        max_images=int(args.max_images),
        save_dir=str(save_dir / "diagnostic"),
        match_overlap=int(args.match_overlap),
        match_x_thr=float(args.match_x_thr),
        device=str(args.device),
        half=bool(args.half),
        warmup=int(args.warmup),
        vis_limit=int(args.vis_limit),
        no_vis=bool(args.no_vis),
    )
    print(f"GCS input shape: {shape_str(imgsz)} (W x H), stored as H,W={imgsz}")
    print(f"built GT4 train diagnostic records: {len(records)} from {label_dir}")
    diagnostic_summary = run_diagnostic(diag_args)
    summary = build_hard_outputs(args, diagnostic_summary)
    summary["source"] = {
        **source_summary,
        "gt4_records_json": str(gt_json.resolve()),
        "gt4_records": int(len(records)),
    }
    summary_json = resolve_repo_path(args.summary_json) if args.summary_json else resolve_repo_path(args.out_file).parent / "gt4_hard_val_summary.json"
    write_json(summary_json, summary)
    print(json.dumps({"hard_sample_count": summary["hard_sample_count"], "summary_json": str(summary_json.resolve())}, indent=2))


if __name__ == "__main__":
    main()
