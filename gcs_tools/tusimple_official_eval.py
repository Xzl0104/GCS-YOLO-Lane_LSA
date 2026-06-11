from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


TUSIMPLE_ORIGINAL_SHAPE = (720, 1280)
DEFAULT_OFFICIAL_SCORE_FP_WEIGHT = 0.02
DEFAULT_OFFICIAL_SCORE_FN_WEIGHT = 0.02


def official_metric_score(
    accuracy: float,
    fp: float,
    fn: float,
    fp_weight: float = DEFAULT_OFFICIAL_SCORE_FP_WEIGHT,
    fn_weight: float = DEFAULT_OFFICIAL_SCORE_FN_WEIGHT,
) -> float:
    """Return a single TuSimple selection score that penalizes FP and FN."""
    return float(accuracy) - float(fp_weight) * float(fp) - float(fn_weight) * float(fn)


@dataclass(frozen=True)
class TuSimpleEvalResult:
    """Aggregated TuSimple official lane detection metrics."""

    accuracy: float
    fp: float
    fn: float
    images: int

    def as_dict(self) -> dict:
        return {
            "Accuracy": round(float(self.accuracy), 6),
            "FP": round(float(self.fp), 6),
            "FN": round(float(self.fn), 6),
            "images": int(self.images),
        }


def read_tusimple_json_lines(path: str | Path) -> list[dict]:
    """Read a TuSimple json-lines label or prediction file."""
    path = Path(path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON line: {exc}") from exc
    return records


def is_valid_tusimple_lane(lane: Iterable[float]) -> bool:
    """Return True when a TuSimple lane contains at least one labeled x coordinate."""
    return any(float(x) >= 0.0 for x in lane)


def valid_tusimple_lanes(lanes: Iterable[Iterable[float]]) -> list[list[float]]:
    """Drop all-negative placeholder lanes from TuSimple GT records."""
    return [list(lane) for lane in lanes if is_valid_tusimple_lane(lane)]


def normalize_tusimple_gt_record(record: dict) -> dict:
    """Return a GT record with TuSimple all-negative placeholder lanes removed."""
    out = dict(record)
    out["lanes"] = valid_tusimple_lanes(record.get("lanes", []))
    return out


def _linear_regression_slope(y: np.ndarray, x: np.ndarray) -> float:
    """Return sklearn LinearRegression.coef_ compatible slope for x = k*y + b."""
    y = y.astype(np.float64)
    x = x.astype(np.float64)
    y_centered = y - y.mean()
    denom = float(np.sum(y_centered * y_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(y_centered * (x - x.mean())) / denom)


class TuSimpleOfficialLaneEval:
    """Python 3 implementation of the TuSimple official lane.py evaluator."""

    pixel_thresh = 20.0
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs: Iterable[float], y_samples: Iterable[float]) -> float:
        xs_arr = np.asarray(list(xs), dtype=np.float64)
        ys_arr = np.asarray(list(y_samples), dtype=np.float64)
        valid = xs_arr >= 0.0
        xs_valid = xs_arr[valid]
        ys_valid = ys_arr[valid]
        if xs_valid.shape[0] <= 1:
            return 0.0
        return float(np.arctan(_linear_regression_slope(ys_valid, xs_valid)))

    @staticmethod
    def line_accuracy(pred: Iterable[float], gt: Iterable[float], thresh: float) -> float:
        pred_arr = np.asarray([p if p >= 0 else -100 for p in pred], dtype=np.float64)
        gt_arr = np.asarray([g if g >= 0 else -100 for g in gt], dtype=np.float64)
        if gt_arr.shape[0] == 0:
            return 0.0
        return float(np.sum(np.where(np.abs(pred_arr - gt_arr) < float(thresh), 1.0, 0.0)) / gt_arr.shape[0])

    @classmethod
    def bench(
        cls,
        pred: list[list[float]],
        gt: list[list[float]],
        y_samples: list[float],
        running_time: float,
    ) -> tuple[float, float, float]:
        """Return official per-image (accuracy, fp, fn)."""
        if any(len(p) != len(y_samples) for p in pred):
            raise ValueError("Format of lanes error: every predicted lane must match len(h_samples).")
        gt = valid_tusimple_lanes(gt)
        if float(running_time) > 200.0 or len(gt) + 2 < len(pred):
            return 0.0, 0.0, 1.0

        angles = [cls.get_angle(x_gts, y_samples) for x_gts in gt]
        threshs = [cls.pixel_thresh / max(float(np.cos(angle)), 1e-12) for angle in angles]
        line_accs: list[float] = []
        matched = 0.0
        fn = 0.0

        for x_gts, thresh in zip(gt, threshs):
            accs = [cls.line_accuracy(x_preds, x_gts, thresh) for x_preds in pred]
            max_acc = float(np.max(accs)) if accs else 0.0
            if max_acc < cls.pt_thresh:
                fn += 1.0
            else:
                matched += 1.0
            line_accs.append(max_acc)

        fp = float(len(pred)) - matched
        if len(gt) > 4 and fn > 0.0:
            fn -= 1.0
        score_sum = float(sum(line_accs))
        if len(gt) > 4 and line_accs:
            score_sum -= float(min(line_accs))
        denom = max(min(4.0, float(len(gt))), 1.0)
        return score_sum / denom, fp / float(len(pred)) if pred else 0.0, fn / denom

    @classmethod
    def bench_records(
        cls,
        pred_records: Iterable[dict],
        gt_records: Iterable[dict],
        strict_length: bool = True,
        return_records: bool = False,
    ) -> tuple[TuSimpleEvalResult, list[dict]]:
        """Evaluate TuSimple-format predictions against TuSimple-format GT records."""
        preds = list(pred_records)
        gts = {str(item["raw_file"]): normalize_tusimple_gt_record(item) for item in gt_records}
        if strict_length and len(preds) != len(gts):
            raise ValueError("We do not get the predictions of all the test tasks.")

        accuracy = 0.0
        fp = 0.0
        fn = 0.0
        per_image: list[dict] = []

        for pred in preds:
            if "raw_file" not in pred or "lanes" not in pred or "run_time" not in pred:
                raise ValueError("raw_file or lanes or run_time not in some predictions.")
            raw_file = str(pred["raw_file"])
            if raw_file not in gts:
                raise ValueError(f"Prediction raw_file does not exist in GT: {raw_file}")

            gt = gts[raw_file]
            image_acc, image_fp, image_fn = cls.bench(
                pred=[list(x) for x in pred["lanes"]],
                gt=[list(x) for x in gt["lanes"]],
                y_samples=list(gt["h_samples"]),
                running_time=float(pred["run_time"]),
            )
            accuracy += image_acc
            fp += image_fp
            fn += image_fn
            if return_records:
                per_image.append(
                    {
                        "raw_file": raw_file,
                        "Accuracy": round(float(image_acc), 6),
                        "FP": round(float(image_fp), 6),
                        "FN": round(float(image_fn), 6),
                        "pred_lanes": int(len(pred["lanes"])),
                        "gt_lanes": int(len(gt["lanes"])),
                        "run_time": float(pred["run_time"]),
                    }
                )

        num = max(len(gts), 1)
        result = TuSimpleEvalResult(accuracy=accuracy / num, fp=fp / num, fn=fn / num, images=len(gts))
        return result, per_image


def find_tusimple_archive_root(path: str | Path) -> Path:
    """Resolve either archive/ or archive/TUSimple to the TuSimple dataset root."""
    root = Path(path)
    if (root / "train_set").exists() and (root / "test_set").exists():
        return root
    candidate = root / "TUSimple"
    if (candidate / "train_set").exists() and (candidate / "test_set").exists():
        return candidate
    raise FileNotFoundError(f"Cannot find TuSimple train_set/test_set under {root}")


def default_tusimple_gt_json(archive_root: str | Path, split: str = "test") -> Path:
    """Return a conventional TuSimple GT json path inside the local archive."""
    root = find_tusimple_archive_root(archive_root)
    split = split.lower()
    if split == "test":
        candidates = [
            root / "test_label.json",
            root.parent / "test_label_new.json",
            root / "train_set" / "seg_label" / "test.json",
        ]
    elif split in {"train", "val"}:
        candidates = [
            root / "train_set" / "seg_label" / "train_val.json",
            root / "train_set" / "label_data_0313.json",
        ]
    else:
        raise ValueError(f"Unsupported TuSimple split: {split!r}")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No TuSimple {split} GT json found under {root}")


def tusimple_image_path(archive_root: str | Path, raw_file: str, split: str = "test") -> Path:
    """Resolve a TuSimple raw_file to a frame path in archive/TUSimple."""
    root = find_tusimple_archive_root(archive_root)
    rel = raw_file.lstrip("/").replace("\\", "/")
    direct = root / rel
    if direct.exists():
        return direct
    preferred = root / ("test_set" if split.lower() == "test" else "train_set") / rel
    if preferred.exists():
        return preferred
    for split_dir in ("test_set", "train_set"):
        candidate = root / split_dir / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find TuSimple image for raw_file={raw_file!r} under {root}")


def _lane_points_for_official(lane: dict, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return visible lane x/y points in original TuSimple pixel coordinates."""
    points_norm = np.asarray(lane["points_norm"], dtype=np.float32)
    if points_norm.ndim != 2 or points_norm.shape[-1] != 2:
        raise ValueError(f"points_norm must have shape K x 2, got {points_norm.shape}")

    if "point_valid" in lane:
        valid = np.asarray(lane["point_valid"], dtype=np.float32) > 0.5
    elif "visible_points_norm" in lane:
        points_norm = np.asarray(lane["visible_points_norm"], dtype=np.float32)
        valid = np.ones(points_norm.shape[0], dtype=bool)
    else:
        valid = np.ones(points_norm.shape[0], dtype=bool)

    points_norm = points_norm[valid]
    if points_norm.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    h, w = int(image_shape[0]), int(image_shape[1])
    points = points_norm * np.array([w, h], dtype=np.float32).reshape(1, 2)
    points = points[np.argsort(points[:, 1], kind="stable")]

    ys = points[:, 1]
    xs = points[:, 0]
    unique_ys, unique_idx = np.unique(np.round(ys, decimals=4), return_index=True)
    order = np.argsort(unique_ys, kind="stable")
    return xs[unique_idx][order].astype(np.float32), ys[unique_idx][order].astype(np.float32)


def gcs_lanes_to_tusimple_lanes(
    lanes: list[dict],
    h_samples: list[int] | list[float],
    image_shape: tuple[int, int] = TUSIMPLE_ORIGINAL_SHAPE,
) -> list[list[int]]:
    """Convert decoded GCS lanes to TuSimple official x-at-h_samples lanes."""
    h, w = int(image_shape[0]), int(image_shape[1])
    sample_y = np.asarray(h_samples, dtype=np.float32)
    tusimple_lanes: list[list[int]] = []

    for lane in lanes:
        xs, ys = _lane_points_for_official(lane, image_shape=(h, w))
        if xs.shape[0] < 2:
            continue
        pred_x = np.interp(sample_y, ys, xs, left=np.nan, right=np.nan)
        lane_xs: list[int] = []
        for x in pred_x:
            if not np.isfinite(x):
                lane_xs.append(-2)
                continue
            xi = int(round(float(x)))
            lane_xs.append(xi if 0 <= xi < w else -2)
        if sum(1 for x in lane_xs if x >= 0) >= 2:
            tusimple_lanes.append(lane_xs)
    return tusimple_lanes


def write_tusimple_predictions(path: str | Path, records: Iterable[dict]) -> None:
    """Write TuSimple prediction records as JSON lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
