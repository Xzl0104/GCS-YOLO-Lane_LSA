# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Generate fixed-y query x priors from original TuSimple JSON labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_LABEL_CANDIDATES = (
    ("archive/TUSimple/train_set/label_data_0313.json", "archive/TUSimple/train_set/label_data_0531.json", "archive/TUSimple/train_set/label_data_0601.json"),
    ("data/tusimple/label_data_0313.json", "data/tusimple/label_data_0531.json", "data/tusimple/label_data_0601.json"),
)


def _default_label_files() -> list[Path]:
    """Return the first complete default TuSimple train-label set found on disk."""
    for group in DEFAULT_LABEL_CANDIDATES:
        paths = [Path(x) for x in group]
        if all(path.exists() for path in paths):
            return paths
    return [Path(x) for x in DEFAULT_LABEL_CANDIDATES[0]]


def _fixed_y_pixels(num_points: int) -> np.ndarray:
    """Return bottom-to-top TuSimple fixed-y anchors, K56 by default."""
    if int(num_points) == 56:
        return np.arange(710.0, 150.0, -10.0, dtype=np.float32)
    return np.linspace(710.0, 160.0, int(num_points), dtype=np.float32)


def _sample_lane_x(lane: list[float], h_samples: list[float] | None, num_points: int, img_w: float) -> np.ndarray | None:
    """Sample one TuSimple lane to normalized bottom-to-top fixed-y x coordinates."""
    x = np.asarray(lane, dtype=np.float32)
    valid = x >= 0.0
    if int(valid.sum()) < 3:
        return None

    if h_samples is not None and len(h_samples) == len(lane):
        y = np.asarray(h_samples, dtype=np.float32)
        valid_y = y[valid]
        valid_x = x[valid] / float(img_w)
        order = np.argsort(valid_y, kind="stable")
        fixed_y = _fixed_y_pixels(num_points)
        sampled = np.interp(fixed_y, valid_y[order], valid_x[order])
    else:
        valid_idx = np.flatnonzero(valid).astype(np.float32)
        valid_x = x[valid] / float(img_w)
        # No h_samples metadata: approximate the same bottom-to-top order.
        sample_idx = np.linspace(float(len(lane) - 1), 0.0, int(num_points), dtype=np.float32)
        sampled = np.interp(sample_idx, valid_idx, valid_x)

    return np.clip(sampled, 0.01, 0.99).astype(np.float32)


def collect_lane_priors(label_files: list[Path], num_points: int, img_w: float, short_visible_thr: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect all lanes plus a GT4/GT5-or-short emphasized pool."""
    all_lanes: list[np.ndarray] = []
    hard_lanes: list[np.ndarray] = []

    for label_file in label_files:
        if not label_file.exists():
            print(f"Warning: {label_file} not found, skipping.")
            continue
        with label_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                lanes = item.get("lanes", [])
                h_samples = item.get("h_samples")
                valid_count = sum(1 for lane in lanes if any(float(x) >= 0.0 for x in lane))
                for lane in lanes:
                    sampled = _sample_lane_x(lane, h_samples, num_points=num_points, img_w=img_w)
                    if sampled is None:
                        continue
                    all_lanes.append(sampled)
                    visible = int(np.sum(np.asarray(lane, dtype=np.float32) >= 0.0))
                    if valid_count >= 4 or visible <= int(short_visible_thr):
                        hard_lanes.append(sampled)

    if not all_lanes:
        raise RuntimeError("No valid lanes were collected from the provided TuSimple JSON labels.")
    if not hard_lanes:
        print("Warning: no GT4/GT5-or-short lanes found; reusing the full lane pool for hard priors.")
        hard_lanes = list(all_lanes)
    return np.asarray(all_lanes, dtype=np.float32), np.asarray(hard_lanes, dtype=np.float32)


def cluster_priors(all_lanes: np.ndarray, hard_lanes: np.ndarray, num_queries: int, seed: int) -> np.ndarray:
    """Cluster six global and six hard-lane priors, then sort by bottom x."""
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover
        raise ImportError("tools/generate_query_priors.py requires scikit-learn for KMeans.") from exc

    if int(num_queries) % 2 != 0:
        raise ValueError(f"num_queries must be even for main/hard split clustering, got {num_queries}.")
    main_k = int(num_queries) // 2
    hard_k = int(num_queries) - main_k
    if all_lanes.shape[0] < main_k:
        raise ValueError(f"Need at least {main_k} valid lanes for main clustering, got {all_lanes.shape[0]}.")
    if hard_lanes.shape[0] < hard_k:
        raise ValueError(f"Need at least {hard_k} hard lanes for hard clustering, got {hard_lanes.shape[0]}.")

    main = KMeans(n_clusters=main_k, random_state=int(seed), n_init=15).fit(all_lanes).cluster_centers_
    hard = KMeans(n_clusters=hard_k, random_state=int(seed), n_init=15).fit(hard_lanes).cluster_centers_
    priors = np.vstack((main, hard)).astype(np.float32)
    return priors[np.argsort(priors[:, 0], kind="stable")]


def generate_and_save_priors(
    label_files: list[Path],
    output_path: Path,
    num_queries: int = 12,
    num_points: int = 56,
    img_w: float = 1280.0,
    short_visible_thr: int = 15,
    seed: int = 42,
) -> torch.Tensor:
    """Generate and save normalized x priors with shape Q x K."""
    all_lanes, hard_lanes = collect_lane_priors(
        label_files=label_files,
        num_points=int(num_points),
        img_w=float(img_w),
        short_visible_thr=int(short_visible_thr),
    )
    print(f"Total valid lanes: {len(all_lanes)}, GT4/GT5-or-short lanes: {len(hard_lanes)}")

    priors = cluster_priors(all_lanes=all_lanes, hard_lanes=hard_lanes, num_queries=int(num_queries), seed=int(seed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.as_tensor(priors, dtype=torch.float32)
    torch.save(tensor, output_path)
    print(f"Saved priors to {output_path}, shape: {tuple(tensor.shape)}")
    return tensor


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-files", nargs="+", type=Path, default=_default_label_files())
    parser.add_argument("--output", type=Path, default=Path("data/query_priors_q12_k56.pt"))
    parser.add_argument("--num-queries", type=int, default=12)
    parser.add_argument("--num-points", type=int, default=56)
    parser.add_argument("--img-w", type=float, default=1280.0)
    parser.add_argument("--short-visible-thr", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    generate_and_save_priors(
        label_files=list(args.label_files),
        output_path=args.output,
        num_queries=args.num_queries,
        num_points=args.num_points,
        img_w=args.img_w,
        short_visible_thr=args.short_visible_thr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
