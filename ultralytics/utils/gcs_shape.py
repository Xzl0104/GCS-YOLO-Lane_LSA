# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Image-shape helpers for GCS-YOLO-Lane."""

from __future__ import annotations

import ast
import re
from typing import Any

DATASET_IMAGE_SHAPES: dict[str, tuple[int, int]] = {
    # Stored as (height, width). TuSimple is commonly written as 960x544 (W x H).
    "tusimple": (544, 960),
    # Stored as (height, width). CULane is commonly written as 960x384 (W x H).
    "culane": (384, 960),
}


def _parse_shape_string(value: str) -> list[int]:
    """Parse strings such as '544,960', '[544, 960]', or '960x544' into integer parts."""
    text = value.strip().lower()
    if not text:
        return []
    if text.startswith("[") or text.startswith("("):
        parsed = ast.literal_eval(text)
        if isinstance(parsed, int):
            return [int(parsed)]
        return [int(x) for x in parsed]
    if "x" in text:
        left, right = [int(x) for x in re.split(r"\s*x\s*", text, maxsplit=1)]
        # Human dataset specs are usually W x H, but the rest of this project uses H, W.
        return [right, left]
    return [int(x) for x in re.split(r"[\s,]+", text) if x]


def normalize_imgsz(
    imgsz: Any = None,
    dataset: str | None = None,
    default: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Normalize image-size inputs to ``(height, width)``.

    Args:
        imgsz: ``None``, int square size, two-item H/W sequence, or a shape string.
        dataset: Optional dataset preset name. Used when ``imgsz`` is omitted.
        default: Optional fallback shape before the dataset preset is consulted.

    Returns:
        A positive ``(height, width)`` tuple.
    """
    if imgsz is None or imgsz == "":
        if default is not None:
            shape = default
        elif dataset is not None and str(dataset).lower() in DATASET_IMAGE_SHAPES:
            shape = DATASET_IMAGE_SHAPES[str(dataset).lower()]
        else:
            shape = DATASET_IMAGE_SHAPES["tusimple"]
    elif isinstance(imgsz, int):
        shape = (int(imgsz), int(imgsz))
    elif isinstance(imgsz, str):
        parts = _parse_shape_string(imgsz)
        shape = (parts[0], parts[0]) if len(parts) == 1 else (parts[0], parts[1])
    elif isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 0:
            return normalize_imgsz(None, dataset=dataset, default=default)
        shape = (int(imgsz[0]), int(imgsz[0])) if len(imgsz) == 1 else (int(imgsz[0]), int(imgsz[1]))
    else:
        raise TypeError(f"Unsupported image size type: {type(imgsz).__name__}")

    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Image shape must be positive, got {(h, w)}")
    return h, w


def assert_gcs_shape(
    actual: tuple[int, int] | list[int],
    expected: Any,
    name: str,
    context: str,
) -> tuple[int, int]:
    """Assert that a tensor/image/mask H,W shape matches the configured GCS shape."""
    actual_hw = (int(actual[0]), int(actual[1]))
    expected_hw = normalize_imgsz(expected)
    assert actual_hw == expected_hw, (
        f"{context}: {name} shape must match GCS gcs_imgsz/image_shape. "
        f"expected H,W={expected_hw} ({shape_str(expected_hw)}), got H,W={actual_hw} ({shape_str(actual_hw)}). "
        "Do not use scalar args.imgsz as a square GCS image size."
    )
    return actual_hw


def assert_gcs_image_tensor(
    tensor: Any,
    expected: Any,
    name: str,
    context: str,
) -> tuple[int, int]:
    """Assert a BCHW/CHW tensor-like object's trailing H,W against the configured GCS shape."""
    shape = getattr(tensor, "shape", None)
    assert shape is not None and len(shape) >= 2, f"{context}: {name} must have a tensor-like shape, got {type(tensor)}."
    return assert_gcs_shape((int(shape[-2]), int(shape[-1])), expected, name=name, context=context)


def trainer_imgsz(shape: tuple[int, int]) -> int:
    """Return the square side length Ultralytics internals need while GCS uses ``shape``."""
    return max(int(shape[0]), int(shape[1]))


def shape_str(shape: tuple[int, int]) -> str:
    """Return a compact W x H display string."""
    h, w = int(shape[0]), int(shape[1])
    return f"{w}x{h}"
