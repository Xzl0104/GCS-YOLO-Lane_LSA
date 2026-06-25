"""Build Q20 data-driven point-reference templates from missing-lane rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath

import numpy as np

from q20_dataref_common import deduplicate_rows, lane_key, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_Y = np.linspace(710 / 720, 160 / 720, 56, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    """Parse template-builder arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-jsonl", nargs="*", type=Path, default=[], help="JSONL files with x[56] and valid[56].")
    parser.add_argument(
        "--missing-csv",
        nargs="*",
        type=Path,
        default=[],
        help="Diagnostic missing-lane CSV files. Requires --label-dir to recover x[56]/valid[56].",
    )
    parser.add_argument("--label-dir", type=Path, default=None, help="labels_gcs split dir for --missing-csv.")
    parser.add_argument("--label-source-prefix", default="train", choices=("train", "test"), help="Converted label prefix.")
    parser.add_argument("--out-json", required=True, type=Path, help="Output q20_dataref_templates.json.")
    parser.add_argument("--out-py", required=True, type=Path, help="Output pasteable Python constant.")
    parser.add_argument("--image-width", type=float, default=1280.0, help="Original image width for pixel x inputs.")
    parser.add_argument("--left-thr", type=float, default=0.25, help="Use lanes with bottom_x <= this as left hard lanes.")
    parser.add_argument("--right-thr", type=float, default=0.75, help="Use lanes with bottom_x >= this as right hard lanes.")
    parser.add_argument("--side-templates-per-side", type=int, default=4, help="Side templates per side.")
    parser.add_argument(
        "--allow-duplicate-weighting",
        action="store_true",
        help="Debug only: keep duplicate source lanes so repeated rows can weight clustering.",
    )
    parser.add_argument(
        "--require-no-duplicates",
        action="store_true",
        help="Fail when duplicate source lanes are present in the input.",
    )
    parser.add_argument(
        "--min-unique-per-side",
        type=int,
        default=4,
        help="Minimum unique source lanes required for each side before clustering.",
    )
    parser.add_argument("--seed", type=int, default=0, help="K-means seed.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve repo-relative paths."""
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    """Return stable repo-relative paths when possible."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def count_unique_lanes(items: list[dict], image_width: float) -> int:
    """Count unique source lanes among already classified side items."""
    return len({lane_key(item, image_width=image_width) for item in items})


def npz_scalar_str(value) -> str:
    """Return a scalar npz value as text."""
    item = np.asarray(value).reshape(-1)[0].item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def sample_id_from_raw_file(raw_file: str, source_prefix: str) -> str:
    """Mirror the common TuSimple converted-label sample id."""
    rel = raw_file.lstrip("/").replace("\\", "/")
    parts = PurePosixPath(rel).parts
    if len(parts) >= 4 and parts[0] == "clips":
        day, clip, frame = parts[1], parts[2], Path(parts[3]).stem
        return f"{source_prefix}_{day}_{clip}_{frame}"
    raise ValueError(f"Cannot infer label sample_id from raw_file={raw_file!r}.")


def build_label_index(label_dir: Path) -> dict[str, Path]:
    """Build raw_file -> label path index as a fallback for nonstandard names."""
    index = {}
    for label_path in sorted(label_dir.glob("*.npz")):
        with np.load(label_path, allow_pickle=False) as data:
            if "raw_file" in data.files:
                index[npz_scalar_str(data["raw_file"])] = label_path
    return index


def label_path_for_raw(raw_file: str, label_dir: Path, source_prefix: str, index: dict[str, Path] | None) -> Path:
    """Resolve a converted GCS label path for a raw_file."""
    try:
        sample_id = sample_id_from_raw_file(raw_file, source_prefix)
        label_path = label_dir / f"{sample_id}.npz"
        if label_path.exists():
            return label_path
    except ValueError:
        label_path = label_dir / "<unknown>.npz"
    if index is not None and raw_file in index:
        return index[raw_file]
    raise FileNotFoundError(f"Missing label for raw_file={raw_file!r}: tried {label_path}")


def load_csv_rows(path: Path, label_dir: Path, source_prefix: str) -> list[dict]:
    """Load diagnostic CSV rows and recover x/valid from converted labels."""
    rows = []
    label_index: dict[str, Path] | None = None
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_file = str(row.get("raw_file", ""))
            if not raw_file:
                raise KeyError(f"{path} row is missing raw_file.")
            if label_index is None:
                try:
                    label_path = label_path_for_raw(raw_file, label_dir, source_prefix, index=None)
                except (FileNotFoundError, ValueError):
                    label_index = build_label_index(label_dir)
                    label_path = label_path_for_raw(raw_file, label_dir, source_prefix, index=label_index)
            else:
                label_path = label_path_for_raw(raw_file, label_dir, source_prefix, index=label_index)

            lane_id = int(row.get("gt_lane_id") or row.get("lane_id") or -1)
            with np.load(label_path, allow_pickle=False) as data:
                lanes = np.asarray(data["lanes"], dtype=np.float32)
                lane_valid = np.asarray(data["lane_valid"], dtype=np.float32)
                fixed_y = np.asarray(data["fixed_y"] if "fixed_y" in data.files else lanes[0, :, 1], dtype=np.float32)
            fixed_y = fixed_y.reshape(-1)
            if lane_id < 0 or lane_id >= lanes.shape[0]:
                raise IndexError(f"{raw_file}: lane_id={lane_id} outside label lanes shape {lanes.shape}.")
            if lanes.shape[1] != 56:
                raise ValueError(f"{label_path}: expected K=56, got lanes shape {lanes.shape}.")
            if not np.allclose(fixed_y, EXPECTED_Y, atol=1e-6):
                raise ValueError(f"{label_path}: fixed_y anchors do not match 710/720 -> 160/720 K56.")

            rows.append(
                {
                    "raw_file": raw_file,
                    "lane_id": lane_id,
                    "x": lanes[lane_id, :, 0].astype(float).tolist(),
                    "valid": (lane_valid[lane_id].astype(np.float32) > 0.5).astype(int).tolist(),
                    "drop_reason": str(row.get("drop_reason") or row.get("new_status") or row.get("old_drop_reason") or ""),
                    "source": path.name,
                }
            )
    return rows


def normalize_x(x, image_width: float) -> np.ndarray:
    """Normalize pixel or normalized x values into [0, 1], keeping invalid as nan."""
    arr = np.asarray(x, dtype=np.float32)
    arr = np.where(arr < 0, np.nan, arr)
    if np.isfinite(arr).any() and float(np.nanmax(arr)) > 2.0:
        arr = arr / float(image_width)
    return arr


def complete_curve(x: np.ndarray, valid) -> np.ndarray | None:
    """Fill valid-span gaps by interpolation and extend endpoints outside the visible span."""
    valid_arr = np.asarray(valid, dtype=bool) & np.isfinite(x)
    if x.shape[0] != 56:
        raise ValueError(f"x must have length 56, got {x.shape[0]}")
    if valid_arr.shape[0] != 56:
        raise ValueError(f"valid must have length 56, got {valid_arr.shape[0]}")
    if int(valid_arr.sum()) < 2:
        return None

    idx = np.arange(56, dtype=np.float32)
    valid_idx = np.where(valid_arr)[0]
    valid_x = x[valid_idx]
    full = np.interp(idx, valid_idx.astype(np.float32), valid_x).astype(np.float32)
    full[: valid_idx[0]] = valid_x[0]
    full[valid_idx[-1] + 1 :] = valid_x[-1]
    return np.clip(full, 0.001, 0.999).astype(np.float32)


def feature_from_curve(curve: np.ndarray, valid) -> np.ndarray:
    """Build a small robust feature vector for low-sample clustering."""
    valid_arr = np.asarray(valid, dtype=bool)
    valid_idx = np.where(valid_arr)[0]
    first = int(valid_idx[0])
    last = int(valid_idx[-1])
    mid = int((first + last) // 2)
    return np.array(
        [
            curve[first],
            curve[min(first + 5, 55)],
            curve[min(first + 12, 55)],
            curve[mid],
            curve[max(last - 12, 0)],
            curve[max(last - 5, 0)],
            curve[last],
            first / 55.0,
            last / 55.0,
            float(valid_arr.sum()) / 56.0,
        ],
        dtype=np.float32,
    )


def kmeans_numpy(features: np.ndarray, k: int, seed: int, iters: int = 100) -> np.ndarray:
    """Run deterministic small-sample k-means."""
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got {features.shape}.")
    if features.shape[0] < k:
        raise RuntimeError(f"Need at least {k} samples, got {features.shape[0]}.")

    rng = np.random.default_rng(seed)
    order = np.argsort(features[:, 0])
    init_pos = np.linspace(0, len(order) - 1, k).round().astype(int)
    centers = features[order[init_pos]].copy()

    for _ in range(iters):
        dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            if np.any(labels == j):
                new_centers[j] = features[labels == j].mean(axis=0)
            else:
                new_centers[j] = features[rng.integers(0, features.shape[0])]
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers

    dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return dist.argmin(axis=1)


def make_side_templates(items: list[dict], side: str, k: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    """Cluster one side and return median templates plus stats."""
    if len(items) < k:
        raise RuntimeError(f"Need at least {k} {side} side lanes, got {len(items)}.")
    curves = np.stack([item["curve"] for item in items], axis=0).astype(np.float32)
    features = np.stack([item["feature"] for item in items], axis=0).astype(np.float32)
    labels = kmeans_numpy(features, k=k, seed=seed)

    templates = []
    stats = []
    for cluster_id in range(k):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            raise RuntimeError(f"Empty cluster {cluster_id} for side={side}.")
        template = np.median(curves[idx], axis=0).astype(np.float32)
        templates.append(template)
        stats.append(
            {
                "side": side,
                "cluster": int(cluster_id),
                "count": int(len(idx)),
                "bottom_x": float(template[0]),
                "mid_x": float(template[28]),
                "top_x": float(template[-1]),
                "members": [
                    {
                        "raw_file": items[int(i)].get("raw_file", ""),
                        "lane_id": items[int(i)].get("lane_id"),
                        "drop_reason": items[int(i)].get("drop_reason", ""),
                        "source": items[int(i)].get("source", ""),
                    }
                    for i in idx[:20]
                ],
            }
        )

    templates_arr = np.stack(templates, axis=0)
    order = np.argsort(templates_arr[:, 0])
    templates_arr = templates_arr[order]
    stats = [stats[int(i)] for i in order]
    return templates_arr, stats


def build_normal_templates() -> np.ndarray:
    """Build the unchanged 12-query normal reference bank."""
    normal_bottom = np.linspace(0.05, 0.95, 12, dtype=np.float32)
    normal_top = 0.5 + (normal_bottom - 0.5) * 0.25
    t = np.linspace(0.0, 1.0, 56, dtype=np.float32)
    return (normal_bottom[:, None] * (1.0 - t[None]) + normal_top[:, None] * t[None]).astype(np.float32)


def main() -> None:
    """Build and save Q20 dataref templates."""
    args = parse_args()
    if args.allow_duplicate_weighting and args.require_no_duplicates:
        raise ValueError("--allow-duplicate-weighting and --require-no-duplicates cannot be used together.")
    if int(args.min_unique_per_side) < int(args.side_templates_per_side):
        raise ValueError("--min-unique-per-side must be >= --side-templates-per-side.")

    jsonl_paths = [resolve_path(path) for path in args.missing_jsonl]
    csv_paths = [resolve_path(path) for path in args.missing_csv]
    if not jsonl_paths and not csv_paths:
        raise ValueError("Pass at least one --missing-jsonl or --missing-csv file.")
    if csv_paths and args.label_dir is None:
        raise ValueError("--label-dir is required with --missing-csv.")
    label_dir = resolve_path(args.label_dir) if args.label_dir is not None else None

    rows: list[dict] = []
    for path in jsonl_paths:
        rows.extend(load_jsonl(path))
    for path in csv_paths:
        assert label_dir is not None
        rows.extend(load_csv_rows(path, label_dir=label_dir, source_prefix=str(args.label_source_prefix)))
    if not rows:
        raise ValueError("No missing-lane rows loaded.")

    rows, dedup_report = deduplicate_rows(
        rows,
        image_width=float(args.image_width),
        allow_duplicate_weighting=bool(args.allow_duplicate_weighting),
        require_no_duplicates=bool(args.require_no_duplicates),
    )

    left_items = []
    right_items = []
    skipped = {"bad_shape": 0, "too_few_valid": 0, "not_side": 0}
    for row in rows:
        try:
            x = normalize_x(row["x"], image_width=float(args.image_width))
            valid = np.asarray(row["valid"], dtype=bool) & np.isfinite(x)
            curve = complete_curve(x, valid)
        except Exception:
            skipped["bad_shape"] += 1
            continue
        if curve is None:
            skipped["too_few_valid"] += 1
            continue

        valid_idx = np.where(valid)[0]
        bottom_x = float(curve[int(valid_idx[0])])
        item = {k: v for k, v in row.items() if k not in {"curve", "feature"}}
        item["curve"] = curve
        item["feature"] = feature_from_curve(curve, valid)
        item["bottom_x"] = bottom_x
        if bottom_x <= float(args.left_thr):
            left_items.append(item)
        elif bottom_x >= float(args.right_thr):
            right_items.append(item)
        else:
            skipped["not_side"] += 1

    k = int(args.side_templates_per_side)
    min_unique = int(args.min_unique_per_side)
    num_left_unique = count_unique_lanes(left_items, image_width=float(args.image_width))
    num_right_unique = count_unique_lanes(right_items, image_width=float(args.image_width))
    if num_left_unique < min_unique:
        raise RuntimeError(
            f"Need at least {min_unique} unique left side lanes before clustering, got {num_left_unique}. "
            "Add train-derived hard lanes instead of relying on duplicate weighting."
        )
    if num_right_unique < min_unique:
        raise RuntimeError(
            f"Need at least {min_unique} unique right side lanes before clustering, got {num_right_unique}. "
            "Add train-derived hard lanes instead of relying on duplicate weighting."
        )

    left_templates, left_stats = make_side_templates(left_items, side="left", k=k, seed=int(args.seed))
    right_templates, right_stats = make_side_templates(right_items, side="right", k=k, seed=int(args.seed) + 17)
    normal_templates = build_normal_templates()
    templates = np.concatenate([left_templates, normal_templates, right_templates], axis=0).astype(np.float32)

    if templates.shape != (20, 56):
        raise AssertionError(f"Q20 dataref templates must have shape (20, 56), got {templates.shape}.")
    if not np.isfinite(templates).all():
        raise ValueError("Q20 dataref templates contain non-finite values.")
    if float(templates.min()) <= 0.0 or float(templates.max()) >= 1.0:
        raise ValueError(f"Q20 dataref templates must be inside (0,1), got {templates.min()}..{templates.max()}.")

    out = {
        "templates": templates.tolist(),
        "source_files": [display_path(path) for path in [*jsonl_paths, *csv_paths]],
        "num_source_rows": int(dedup_report["input_rows"]),
        "num_rows_after_dedup": int(len(rows)),
        "num_left_items": int(len(left_items)),
        "num_right_items": int(len(right_items)),
        "num_left_unique_lanes": int(num_left_unique),
        "num_right_unique_lanes": int(num_right_unique),
        "allow_duplicate_weighting": bool(args.allow_duplicate_weighting),
        "dedup_report": dedup_report,
        "skipped": skipped,
        "left_threshold": float(args.left_thr),
        "right_threshold": float(args.right_thr),
        "left_stats": left_stats,
        "right_stats": right_stats,
        "normal_bank": {
            "bottom_x": normal_templates[:, 0].astype(float).tolist(),
            "top_x": normal_templates[:, -1].astype(float).tolist(),
        },
    }

    out_json = resolve_path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    rounded = np.round(templates, 6).tolist()
    py = (
        "# Auto-generated by tools/build_q20_dataref_templates.py.\n"
        "# Paste Q20_DATAREF_X into ultralytics/nn/modules/gcs_lane.py when updating the built-in bank.\n"
        "Q20_DATAREF_X = (\n"
    )
    for row in rounded:
        py += "    " + repr(row) + ",\n"
    py += ")\n"

    out_py = resolve_path(args.out_py)
    out_py.parent.mkdir(parents=True, exist_ok=True)
    out_py.write_text(py, encoding="utf-8")

    printable = {key: value for key, value in out.items() if key != "templates"}
    print(json.dumps(printable, indent=2))
    print(f"saved json: {out_json}")
    print(f"saved py: {out_py}")


if __name__ == "__main__":
    main()
