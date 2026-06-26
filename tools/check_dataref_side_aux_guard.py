"""Smoke checks for Q20 dataref side-aux guardrails."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from check_q20_dataref_side_aux_loss import build_synthetic_batch
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer
from ultralytics.nn.tasks import GCSLaneModel
from ultralytics.utils.gcs_loss import GCSLoss


HINT = "Use gcs-yolo-lane-s-q20-k56-dataref.yaml or disable --gcs-dataref-side-aux"
CFG_ROOT = ROOT / "ultralytics" / "cfg" / "models" / "gcs"
CFG_DATAREF = CFG_ROOT / "gcs-yolo-lane-s-q20-k56-dataref.yaml"
CFG_SIDEGEOM = CFG_ROOT / "gcs-yolo-lane-s-q20-k56-sidegeom.yaml"
CFG_Q18 = CFG_ROOT / "gcs-yolo-lane-s-q18-k56-side.yaml"
CFG_Q12 = CFG_ROOT / "gcs-yolo-lane-s.yaml"


def _build_model(cfg: Path) -> GCSLaneModel:
    """Build a GCS model for guard checks."""
    if not cfg.exists():
        raise FileNotFoundError(f"Missing model config: {cfg}")
    return GCSLaneModel(str(cfg), nc=1, ch=3, verbose=False)


def _trainer_guard(model: GCSLaneModel) -> None:
    """Run the dataref side-aux trainer setup guard on a model."""
    fake_trainer = SimpleNamespace(args=SimpleNamespace(gcs_dataref_side_aux=0.25))
    GCSLaneTrainer._validate_dataref_side_aux_model(fake_trainer, model)


def _expect_guard_failure(label: str, model: GCSLaneModel) -> None:
    """Assert that the trainer guard rejects an invalid model."""
    try:
        _trainer_guard(model)
    except ValueError as exc:
        if HINT not in str(exc):
            raise AssertionError(f"{label} guard error did not include the expected hint: {exc}") from exc
        return
    raise AssertionError(f"{label} must fail when gcs_dataref_side_aux > 0.")


def _loss_fn() -> GCSLoss:
    """Build a loss instance with dataref side aux enabled."""
    return GCSLoss(
        model={
            "gcs_imgsz": [544, 960],
            "gcs_dataref_side_aux": 0.25,
            "gcs_dataref_side_aux_point": 1.0,
            "gcs_dataref_side_aux_valid": 0.25,
            "gcs_dataref_side_aux_exist": 0.25,
            "gcs_dataref_side_max_points": 24,
            "gcs_dataref_side_ref_thr_px": 80.0,
            "gcs_dataref_side_gt_count": 4,
            "gcs_dataref_side_left_thr": 0.35,
            "gcs_dataref_side_right_thr": 0.65,
        },
        image_size=(544, 960),
    )


def check_trainer_setup_guard() -> None:
    """Check YAML/head-level fail-fast behavior before training batches."""
    _trainer_guard(_build_model(CFG_DATAREF))
    _expect_guard_failure("Q20-sidegeom", _build_model(CFG_SIDEGEOM))
    _expect_guard_failure("Q18", _build_model(CFG_Q18))
    _expect_guard_failure("Q12", _build_model(CFG_Q12))


def check_loss_metadata_guard() -> None:
    """Check loss-level metadata guard for dataref versus sidegeom outputs."""
    preds, batch = build_synthetic_batch()
    loss_fn = _loss_fn()
    total, loss_items = loss_fn(preds, batch)
    if not torch.isfinite(total) or not torch.isfinite(loss_items).all():
        raise AssertionError("Q20-dataref loss guard pass produced NaN or Inf.")

    sidegeom_preds = dict(preds)
    sidegeom_preds["reference_mode"] = "sidegeom"
    sidegeom_preds["is_dataref_reference"] = torch.tensor(0.0, dtype=torch.float32)
    _expect_loss_failure("Q20-sidegeom metadata", loss_fn, sidegeom_preds, batch)

    q12_like_preds = dict(preds)
    for key in ("pred_points", "pred_logits", "pred_valid_logits", "pred_reference_x"):
        q12_like_preds[key] = preds[key][:, :12].clone()
    q12_like_preds["reference_mode"] = "dataref"
    q12_like_preds["is_dataref_reference"] = torch.tensor(1.0, dtype=torch.float32)
    _expect_loss_failure("Q12-like dataref metadata", loss_fn, q12_like_preds, batch)

    missing_ref_preds = dict(preds)
    missing_ref_preds.pop("pred_reference_x")
    _expect_loss_failure("missing pred_reference_x", loss_fn, missing_ref_preds, batch)


def _expect_loss_failure(label: str, loss_fn: GCSLoss, preds: dict, batch: dict) -> None:
    """Assert that loss guard rejects invalid dataref side-aux outputs."""
    try:
        loss_fn(preds, batch)
    except ValueError as exc:
        if HINT not in str(exc):
            raise AssertionError(f"{label} loss guard error did not include the expected hint: {exc}") from exc
        return
    raise AssertionError(f"{label} must fail in loss guard when gcs_dataref_side_aux > 0.")


def main() -> None:
    """Run all dataref side-aux guard smoke checks."""
    check_trainer_setup_guard()
    check_loss_metadata_guard()
    print("OK: dataref side-aux guards reject sidegeom/Q18/Q12 and allow Q20-dataref.")


if __name__ == "__main__":
    main()
