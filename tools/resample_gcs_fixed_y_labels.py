from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample fixed-y GCS labels to another shared y-anchor range.")
    parser.add_argument("--src", required=True, help="Source labels_gcs split directory.")
    parser.add_argument("--dst", required=True, help="Destination labels_gcs split directory.")
    parser.add_argument("--fixed-y-start", type=float, required=True, help="New bottom fixed-y anchor.")
    parser.add_argument("--fixed-y-end", type=float, default=0.25, help="New top fixed-y anchor.")
    parser.add_argument("--num-points", type=int, default=32, help="Number of fixed-y anchors.")
    return parser.parse_args()


def resample_lane(lane: np.ndarray, valid: np.ndarray, target_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    new_lane = np.zeros((target_y.shape[0], 2), dtype=np.float32)
    new_lane[:, 1] = target_y
    new_valid = np.zeros((target_y.shape[0],), dtype=np.float32)

    mask = valid > 0.5
    if int(mask.sum()) < 2:
        return new_lane, new_valid

    pts = lane[mask].astype(np.float32)
    order = np.argsort(pts[:, 1], kind="stable")
    y_asc = pts[order, 1]
    x_asc = pts[order, 0]

    unique_y, unique_idx = np.unique(np.round(y_asc, decimals=8), return_index=True)
    y_asc = y_asc[unique_idx]
    x_asc = x_asc[unique_idx]
    if y_asc.shape[0] < 2:
        return new_lane, new_valid

    lo = float(y_asc.min())
    hi = float(y_asc.max())
    in_range = (target_y >= lo - 1e-6) & (target_y <= hi + 1e-6)
    if int(in_range.sum()) < 2:
        return new_lane, new_valid

    new_lane[in_range, 0] = np.interp(target_y[in_range], y_asc, x_asc).astype(np.float32)
    new_valid[in_range] = 1.0
    return new_lane, new_valid


def convert_one(src_path: Path, dst_path: Path, target_y: np.ndarray) -> bool:
    with np.load(src_path, allow_pickle=False) as data:
        keep_keys = {"lanes", "lane_valid", "num_lanes", "point_mode", "fixed_y", "num_points", "raw_file", "image_shape"}
        payload = {k: data[k] for k in data.files if k in keep_keys}

    old_num_lanes = int(np.asarray(payload["num_lanes"]).reshape(-1)[0]) if "num_lanes" in payload else 0
    lanes = np.asarray(payload["lanes"], dtype=np.float32)
    valid = np.asarray(payload["lane_valid"], dtype=np.float32)
    new_lanes = np.zeros((lanes.shape[0], target_y.shape[0], 2), dtype=np.float32)
    new_valid = np.zeros((lanes.shape[0], target_y.shape[0]), dtype=np.float32)
    for i in range(lanes.shape[0]):
        new_lanes[i], new_valid[i] = resample_lane(lanes[i], valid[i], target_y)

    keep = new_valid.sum(axis=1) >= 2
    new_lanes = new_lanes[keep]
    new_valid = new_valid[keep]

    payload["lanes"] = new_lanes.astype(np.float32)
    payload["lane_valid"] = new_valid.astype(np.float32)
    payload["num_lanes"] = np.array([int(new_lanes.shape[0])], dtype=np.int64)
    payload["point_mode"] = np.array("fixed_y")
    payload["fixed_y"] = target_y.astype(np.float32)
    payload["num_points"] = np.array([int(target_y.shape[0])], dtype=np.int32)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, **payload)
    return bool(int(payload["num_lanes"][0]) != old_num_lanes)


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    target_y = np.linspace(float(args.fixed_y_start), float(args.fixed_y_end), int(args.num_points), dtype=np.float32)
    files = sorted(src.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz labels found under {src}")
    changed_counts = 0
    for src_path in tqdm(files, desc="resample fixed_y labels"):
        if convert_one(src_path, dst / src_path.name, target_y):
            changed_counts += 1
    print(
        {
            "src": str(src),
            "dst": str(dst),
            "files": len(files),
            "changed_num_lanes": changed_counts,
            "fixed_y_start": float(target_y[0]),
            "fixed_y_end": float(target_y[-1]),
            "num_points": int(target_y.shape[0]),
        }
    )


if __name__ == "__main__":
    main()
