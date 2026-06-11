from __future__ import annotations

import argparse
import csv
from pathlib import Path
import yaml


MAX_METRICS = {"val/precision", "val/recall", "val/f1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GCS-YOLO-Lane training results.csv.")
    parser.add_argument("csv", help="Path to a GCS training results.csv file.")
    parser.add_argument("--args-yaml", default=None, help="Optional args.yaml for the run. Defaults to sibling args.yaml.")
    parser.add_argument("--late-window", type=int, default=20, help="Epochs used to inspect late-training trends.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                key = key.strip()
                text = (value or "").strip()
                if not key or not text:
                    continue
                parsed[key] = float(text)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def best_row(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    fn = max if key in MAX_METRICS else min
    return fn(rows, key=lambda row: row[key])


def trend(rows: list[dict[str, float]], key: str, window: int) -> float:
    if len(rows) < 2:
        return 0.0
    w = max(1, min(int(window), len(rows) // 2))
    early = sum(row[key] for row in rows[:w]) / w
    late = sum(row[key] for row in rows[-w:]) / w
    return late - early


def print_metric_summary(rows: list[dict[str, float]]) -> None:
    keys = [
        "train/exist_loss",
        "train/point_loss",
        "train/point_valid_loss",
        "train/line_iou_loss",
        "train/count_cls_loss",
        "train/quality_loss",
        "val/exist_loss",
        "val/point_loss",
        "val/point_valid_loss",
        "val/line_iou_loss",
        "val/count_cls_loss",
        "val/quality_loss",
        "val/precision",
        "val/recall",
        "val/f1",
        "val/ape_mean_px",
        "val/lane_count_mae",
        "val/total_loss",
    ]
    print("Metric Summary")
    for key in keys:
        if key not in rows[0]:
            continue
        values = [row[key] for row in rows]
        best = best_row(rows, key)
        print(
            f"{key}: first={values[0]:.6g} last={values[-1]:.6g} "
            f"min={min(values):.6g} max={max(values):.6g} "
            f"best_epoch={int(best['epoch'])} best={best[key]:.6g}"
        )


def print_diagnosis(rows: list[dict[str, float]], late_window: int, gains: dict[str, float]) -> None:
    last = rows[-1]
    best_f1 = best_row(rows, "val/f1") if "val/f1" in last else None
    print("\nDiagnosis")
    if best_f1:
        print(f"best_f1={best_f1['val/f1']:.6g} at epoch {int(best_f1['epoch'])}; last_f1={last['val/f1']:.6g}")

    if "val/ape_mean_px" in last and "val/f1" in last:
        ape_trend = trend(rows, "val/ape_mean_px", late_window)
        f1_trend = trend(rows, "val/f1", late_window)
        if ape_trend < -1.0 and f1_trend > 0.01:
            print("- Geometry is still improving late in training; train longer or slow the LR decay.")
        if last["val/ape_mean_px"] > 20.0:
            print("- Mean APE is still above 20px, so geometry is the main bottleneck.")

    if "val/exist_loss" in last and "val/point_loss" in last:
        exist_contrib = last["val/exist_loss"] * gains.get("exist", 0.5)
        point_contrib = last["val/point_loss"] * gains.get("point", 15.0)
        if exist_contrib > point_contrib:
            print("- Existence calibration contributes more loss than point geometry; lower gcs_exist/pos_weight or tune conf.")

    if "val/lane_count_mae" in last and last["val/lane_count_mae"] > 0.3:
        print("- Lane-count MAE is high enough to cause FP/FN; inspect existence scores and sampling balance.")

    if "train/point_loss" in last and "val/point_loss" in last:
        gap = last["val/point_loss"] - last["train/point_loss"]
        if gap > 0.02:
            print("- Point-loss train/val gap is large; likely domain split or overfitting.")
        elif gap > 0.005:
            print("- Point-loss train/val gap is moderate; more data balance and augmentation may help.")


def print_late_rows(rows: list[dict[str, float]], count: int) -> None:
    print(f"\nLast {min(count, len(rows))} Epochs")
    for row in rows[-count:]:
        fields = [
            f"epoch={int(row['epoch'])}",
            f"f1={row.get('val/f1', 0.0):.4f}",
            f"P={row.get('val/precision', 0.0):.4f}",
            f"R={row.get('val/recall', 0.0):.4f}",
            f"exist={row.get('val/exist_loss', 0.0):.4f}",
            f"point={row.get('val/point_loss', 0.0):.4f}",
            f"pvalid={row.get('val/point_valid_loss', 0.0):.4f}",
            f"line_iou={row.get('val/line_iou_loss', 0.0):.4f}",
            f"quality={row.get('val/quality_loss', 0.0):.4f}",
            f"ape={row.get('val/ape_mean_px', 0.0):.2f}",
            f"count={row.get('val/lane_count_mae', 0.0):.3f}",
        ]
        print(" ".join(fields))


def load_loss_gains(csv_path: Path, args_yaml: str | None) -> dict[str, float]:
    """Load loss gains from args.yaml, falling back to current training defaults."""
    gains = {
        "exist": 1.0,
        "point": 5.0,
        "point_valid": 0.5,
        "line_iou": 0.3,
        "count_cls": 0.3,
        "quality": 0.3,
    }
    path = Path(args_yaml) if args_yaml else csv_path.with_name("args.yaml")
    if not path.exists():
        return gains
    args = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping = {
        "exist": "gcs_exist",
        "point": "gcs_point",
        "point_valid": "gcs_point_valid",
        "line_iou": "gcs_line_iou",
        "count_cls": "gcs_count_cls",
        "quality": "gcs_quality",
    }
    for key, arg_name in mapping.items():
        if isinstance(arg_name, tuple):
            primary, legacy = arg_name
            if primary in args:
                gains[key] = float(args[primary])
            elif legacy in args:
                gains[key] = float(args[legacy])
        elif arg_name in args:
            gains[key] = float(args[arg_name])
    return gains


def print_weighted_last(rows: list[dict[str, float]], gains: dict[str, float]) -> None:
    """Print last-epoch weighted validation loss contributions."""
    last = rows[-1]
    parts = {}
    for name, gain in gains.items():
        key = f"val/{name}_loss"
        if key in last:
            parts[name] = gain * last[key]
    total = sum(parts.values())
    if total <= 0:
        return
    print("\nLast Epoch Weighted Val Loss Shares")
    for name, value in parts.items():
        print(f"{name}: value={value:.6g} share={value / total:.3f}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    rows = load_rows(csv_path)
    gains = load_loss_gains(csv_path, args.args_yaml)
    print(f"rows={len(rows)} file={csv_path.resolve()}")
    print_metric_summary(rows)
    print_diagnosis(rows, late_window=args.late_window, gains=gains)
    print_weighted_last(rows, gains)
    print_late_rows(rows, count=min(args.late_window, 20))


if __name__ == "__main__":
    main()
