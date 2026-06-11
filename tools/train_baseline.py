from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics import YOLO
from ultralytics.utils.gcs_shape import DATASET_IMAGE_SHAPES, normalize_imgsz, shape_str, trainer_imgsz


def dataset_defaults(dataset: str) -> dict[str, Path]:
    """Return conventional YOLO baseline paths for a converted lane dataset."""
    name = dataset.lower()
    return {
        "data": ROOT / "data" / f"{name}_yolo.yaml",
        "name": f"yolo11s_seg_{name}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO11 segmentation baseline on converted TuSimple labels.")
    parser.add_argument("--dataset", default="tusimple", choices=sorted(DATASET_IMAGE_SHAPES))
    parser.add_argument("--data", default=None, help="Ultralytics dataset yaml.")
    parser.add_argument("--weights", default="yolo11s-seg.pt", help="YOLO11 segmentation checkpoint or yaml.")
    parser.add_argument("--imgsz", type=int, default=None, help="Long-side training size. Defaults to dataset width 960.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/baseline")
    parser.add_argument("--name", default=None)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--lrf", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--nbs", type=int, default=0, help="Nominal batch size. 0 uses --batch, matching GCS training.")
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--rect", action=argparse.BooleanOptionalAction, default=True, help="Keep rectangular lane image shapes.")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_weights(value: str) -> str:
    if "://" in value:
        return value
    path = Path(value)
    candidate = path if path.is_absolute() else ROOT / path
    if candidate.exists():
        return str(candidate)
    if path.suffix == ".pt":
        raise FileNotFoundError(f"weights not found: {candidate}")
    return value


def main() -> None:
    args = parse_args()
    defaults = dataset_defaults(args.dataset)
    image_shape = normalize_imgsz(None, dataset=args.dataset)
    imgsz = int(args.imgsz or trainer_imgsz(image_shape))
    data = resolve_project_path(args.data or defaults["data"])
    project = resolve_project_path(args.project)
    weights = resolve_weights(args.weights)

    print(f"project root: {ROOT}")
    print(f"weights: {weights}")
    print(f"dataset shape: {shape_str(image_shape)} (W x H); YOLO rect long side={imgsz}")
    print(f"runs project: {project}")

    model = YOLO(weights)
    model.train(
        data=str(data),
        imgsz=imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name or defaults["name"],
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        nbs=args.nbs if args.nbs > 0 else args.batch,
        mosaic=args.mosaic,
        rect=args.rect,
    )


if __name__ == "__main__":
    main()
