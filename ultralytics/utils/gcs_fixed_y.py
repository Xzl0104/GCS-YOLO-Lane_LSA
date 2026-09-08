"""Shared fixed-y anchor contracts for structured lane labels and decoders."""

from __future__ import annotations

import numpy as np


def build_fixed_y_anchors(
    original_h: int,
    start_px: float,
    end_px: float,
    k: int,
) -> np.ndarray:
    """Build normalized bottom-to-top anchors from an explicit pixel-space contract."""
    original_h = int(original_h)
    k = int(k)
    start_px = float(start_px)
    end_px = float(end_px)
    if original_h <= 1:
        raise ValueError(f"fixed-y original_h must be > 1, got {original_h}.")
    if k < 2:
        raise ValueError(f"fixed-y K must be >= 2, got {k}.")
    if not (0.0 <= end_px < start_px < float(original_h)):
        raise ValueError(
            "fixed-y pixel anchors must satisfy 0 <= end_px < start_px < original_h, "
            f"got end_px={end_px}, start_px={start_px}, original_h={original_h}."
        )
    return (np.linspace(start_px, end_px, k, dtype=np.float32) / float(original_h)).astype(np.float32)


def build_fixed_y_contract(
    original_h: int,
    start_px: float,
    end_px: float,
    k: int,
) -> dict[str, object]:
    """Build and validate the explicit fixed-y dataset contract."""
    original_h = int(original_h)
    start_px = float(start_px)
    end_px = float(end_px)
    k = int(k)
    anchors = build_fixed_y_anchors(original_h, start_px, end_px, k)
    validate_fixed_y_contract(
        anchors,
        original_h=original_h,
        start_px=start_px,
        end_px=end_px,
        k=k,
        name="fixed_y contract",
    )
    return {
        "fixed_y": anchors,
        "fixed_y_original_h": original_h,
        "fixed_y_start_px": start_px,
        "fixed_y_end_px": end_px,
        "num_points": k,
        "legacy_fallback": False,
    }


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


def validate_fixed_y_contract(
    fixed_y,
    *,
    original_h: int,
    start_px: float,
    end_px: float,
    k: int,
    atol: float = 1e-3,
    name: str = "fixed_y",
) -> str:
    """Validate one explicit descending fixed-y anchor contract."""
    anchors_px = _to_pixel_anchors(fixed_y, original_h=original_h, k=k, name=name)
    expected = np.linspace(float(start_px), float(end_px), int(k), dtype=np.float32)
    if not _allclose_allowing_fp16_roundoff(anchors_px, expected, atol=atol):
        raise ValueError(
            f"{name}: fixed-y contract violated. Expected descending pixel anchors "
            f"{float(start_px):.6g}..{float(end_px):.6g} with K={int(k)}, "
            f"got first/last={anchors_px[0]:.6g}/{anchors_px[-1]:.6g}."
        )
    if not bool(np.all(np.diff(anchors_px) < 0.0)):
        raise ValueError(f"{name}: fixed-y anchors must be strictly descending bottom-to-top.")
    return "desc"


def validate_training_fixed_y_desc(
    fixed_y,
    original_h: int = 720,
    k: int = 56,
    atol: float = 1e-3,
    name: str = "training fixed_y",
) -> str:
    """Validate training labels against the active desc 710..160 fixed-y contract."""
    if int(original_h) != 720:
        raise ValueError(f"{name}: TuSimple training fixed-y contract requires original_h=720, got {original_h}.")
    return validate_fixed_y_contract(
        fixed_y,
        original_h=original_h,
        start_px=710.0,
        end_px=160.0,
        k=k,
        atol=atol,
        name=name,
    )


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


def validate_tusimple_h_samples_asc(
    h_samples,
    k: int = 56,
    atol: float = 1e-3,
    name: str = "TuSimple record h_samples",
) -> str:
    """Validate one TuSimple record's asc h_samples against a canonical K56 slice."""
    samples = np.asarray(h_samples, dtype=np.float32).reshape(-1)
    if int(samples.shape[0]) <= 0:
        raise ValueError(f"{name}: h_samples must not be empty.")
    if int(samples.shape[0]) > int(k):
        raise ValueError(f"{name}: K mismatch, got {samples.shape[0]}, expected at most {k}.")
    if not np.isfinite(samples).all():
        raise ValueError(f"{name}: h_samples contain NaN or Inf.")

    expected = expected_tusimple_h_samples(k=k)
    n = int(samples.shape[0])
    for start in range(0, int(expected.shape[0]) - n + 1):
        if _allclose_allowing_fp16_roundoff(samples, expected[start : start + n], atol=atol):
            return "asc"
    raise ValueError(
        f"{name}: h_samples contract violated. Expected an ascending contiguous subset of "
        f"160..710 step 10, got first/last={samples[0]:.6g}/{samples[-1]:.6g} and K={n}."
    )


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
