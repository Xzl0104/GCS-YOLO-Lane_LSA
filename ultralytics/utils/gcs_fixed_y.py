"""Fixed-y anchor contract checks for TuSimple K56 labels and decoders."""

from __future__ import annotations

import numpy as np


def expected_tusimple_h_samples(k: int = 56) -> np.ndarray:
    """Return canonical TuSimple K56 h_samples in ascending image-y order."""
    expected = np.arange(160, 720, 10, dtype=np.float32)
    if int(k) != int(expected.shape[0]):
        raise ValueError(f"TuSimple fixed-y contract requires K=56, got requested K={k}.")
    return expected


def expected_training_fixed_y_desc(k: int = 56) -> np.ndarray:
    """Return canonical TuSimple K56 training anchors in descending image-y order."""
    return expected_tusimple_h_samples(k=k)[::-1].copy()


def _to_pixel_anchors(fixed_y, original_h: int = 720, k: int = 56, name: str = "fixed_y") -> np.ndarray:
    """Normalize fixed-y-like values to pixel-space anchors and check shape/finite values."""
    anchors = np.asarray(fixed_y, dtype=np.float32).reshape(-1)
    if int(anchors.shape[0]) != int(k):
        raise ValueError(f"{name}: K mismatch, got {anchors.shape[0]}, expected {k}.")
    if not np.isfinite(anchors).all():
        raise ValueError(f"{name}: fixed-y anchors contain NaN or Inf.")
    return anchors * float(original_h) if float(np.max(np.abs(anchors))) <= 2.0 else anchors


def _allclose_allowing_fp16_roundoff(anchors_px: np.ndarray, expected: np.ndarray, atol: float) -> bool:
    """Accept canonical anchors that were quantized through normalized fp16 values."""
    if bool(np.allclose(anchors_px, expected, atol=atol)):
        return True
    return bool(np.max(np.abs(anchors_px - expected)) <= 0.25)


def validate_training_fixed_y_desc(
    fixed_y,
    original_h: int = 720,
    k: int = 56,
    atol: float = 1e-3,
    name: str = "training fixed_y",
) -> str:
    """Validate training labels against the active desc 710..160 fixed-y contract."""
    anchors_px = _to_pixel_anchors(fixed_y, original_h=original_h, k=k, name=name)
    expected = expected_training_fixed_y_desc(k=k)
    if not _allclose_allowing_fp16_roundoff(anchors_px, expected, atol=atol):
        raise ValueError(
            f"{name}: training fixed-y contract violated. "
            f"Expected desc 710..160 step -10, got first/last={anchors_px[0]:.6g}/{anchors_px[-1]:.6g}."
        )
    return "desc"


def validate_official_h_samples_asc(
    h_samples,
    k: int = 56,
    atol: float = 1e-3,
    name: str = "TuSimple official h_samples",
) -> str:
    """Validate TuSimple official h_samples against asc 160..710 order."""
    samples = np.asarray(h_samples, dtype=np.float32).reshape(-1)
    if int(samples.shape[0]) != int(k):
        raise ValueError(f"{name}: K mismatch, got {samples.shape[0]}, expected {k}.")
    if not np.isfinite(samples).all():
        raise ValueError(f"{name}: h_samples contain NaN or Inf.")
    expected = expected_tusimple_h_samples(k=k)
    if not _allclose_allowing_fp16_roundoff(samples, expected, atol=atol):
        raise ValueError(
            f"{name}: official h_samples contract violated. "
            f"Expected asc 160..710 step 10, got first/last={samples[0]:.6g}/{samples[-1]:.6g}."
        )
    return "asc"


def validate_fixed_y_anchors(
    fixed_y,
    original_h: int = 720,
    k: int = 56,
    atol: float = 1e-3,
    name: str = "fixed_y",
) -> str:
    """Validate K56 anchors against TuSimple h_samples and return asc/desc direction."""
    anchors_px = _to_pixel_anchors(fixed_y, original_h=original_h, k=k, name=name)
    expected = expected_tusimple_h_samples(k=k)
    ok_asc = _allclose_allowing_fp16_roundoff(anchors_px, expected, atol=atol)
    ok_desc = _allclose_allowing_fp16_roundoff(anchors_px, expected[::-1], atol=atol)
    if not (ok_asc or ok_desc):
        raise ValueError(
            f"{name}: fixed-y anchors mismatch. got first/last={anchors_px[0]:.6g}/{anchors_px[-1]:.6g}, "
            "expected 160..710 or 710..160 step 10."
        )
    return "asc" if ok_asc else "desc"
