from __future__ import annotations

import cv2
import numpy as np

TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM = 710.0 / 720.0


def _is_valid_number(value: float) -> bool:
    """Return True for finite numeric coordinates."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def sort_lane_bottom_to_top(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort lane points by image y from large to small."""
    clean = [
        (float(x), float(y))
        for x, y in points
        if _is_valid_number(x) and _is_valid_number(y) and float(x) >= 0.0 and float(y) >= 0.0
    ]
    return sorted(clean, key=lambda p: p[1], reverse=True)


def remove_duplicate_y(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep one point for each rounded y coordinate."""
    out: list[tuple[float, float]] = []
    used: set[int] = set()
    for x, y in points:
        key = int(round(y))
        if key in used:
            continue
        used.add(key)
        out.append((float(x), float(y)))
    return out


def resample_polyline(points: list[tuple[float, float]], num_points: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Resample a bottom-to-top polyline to a fixed number of points."""
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    points = remove_duplicate_y(sort_lane_bottom_to_top(points))
    if len(points) < 2:
        return (
            np.zeros((num_points, 2), dtype=np.float32),
            np.zeros((num_points,), dtype=np.float32),
        )

    pts = np.asarray(points, dtype=np.float32)
    diff = pts[1:] - pts[:-1]
    seg_len = np.sqrt((diff * diff).sum(axis=1))
    arc = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = float(arc[-1])
    if total_len < 1e-6:
        return (
            np.zeros((num_points, 2), dtype=np.float32),
            np.zeros((num_points,), dtype=np.float32),
        )

    target_arc = np.linspace(0.0, total_len, num_points)
    xs = np.interp(target_arc, arc, pts[:, 0])
    ys = np.interp(target_arc, arc, pts[:, 1])
    sampled = np.stack((xs, ys), axis=1).astype(np.float32)
    valid = np.ones((num_points,), dtype=np.float32)
    return sampled, valid


def fixed_y_anchors(num_points: int = 32, y_start: float = TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM, y_end: float = 0.25) -> np.ndarray:
    """Return bottom-to-top normalized y anchors for fixed-y x-only lane labels."""
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    if not (0.0 <= float(y_end) < float(y_start) <= 1.0):
        raise ValueError(f"Expected 0 <= y_end < y_start <= 1, got y_start={y_start}, y_end={y_end}")
    return np.linspace(float(y_start), float(y_end), int(num_points), dtype=np.float32)


def sample_polyline_fixed_y(
    points: list[tuple[float, float]],
    img_h: int,
    img_w: int,
    num_points: int = 32,
    y_start: float = TUSIMPLE_OFFICIAL_BOTTOM_Y_NORM,
    y_end: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one lane at fixed normalized y anchors and predict only x at those anchors.

    The returned lane keeps all K anchor rows. Invalid y anchors retain their
    y coordinate with x=0 and valid=0 so fixed-y losses compare matching rows.
    """
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"Image size must be positive, got h={img_h}, w={img_w}")

    anchors = fixed_y_anchors(num_points=num_points, y_start=y_start, y_end=y_end)
    sampled = np.zeros((num_points, 2), dtype=np.float32)
    sampled[:, 1] = anchors
    valid = np.zeros((num_points,), dtype=np.float32)

    lane = remove_duplicate_y(sort_lane_bottom_to_top(points))
    if len(lane) < 2:
        return sampled, valid, anchors

    pts = np.asarray(lane, dtype=np.float32)
    ys_desc = pts[:, 1]
    xs_desc = pts[:, 0]
    order = np.argsort(ys_desc)
    ys = ys_desc[order]
    xs = xs_desc[order]
    if float(ys[-1] - ys[0]) < 1e-6:
        return sampled, valid, anchors

    anchor_y_px = anchors * float(img_h)
    in_range = (anchor_y_px >= ys[0] - 1e-3) & (anchor_y_px <= ys[-1] + 1e-3)
    if not in_range.any():
        return sampled, valid, anchors

    interp_x = np.interp(anchor_y_px[in_range], ys, xs)
    valid_x = (interp_x >= 0.0) & (interp_x <= float(img_w - 1))
    in_range_indices = np.flatnonzero(in_range)
    keep_indices = in_range_indices[valid_x]
    sampled[keep_indices, 0] = np.clip(interp_x[valid_x] / float(img_w), 0.0, 1.0).astype(np.float32)
    valid[keep_indices] = 1.0
    return sampled, valid, anchors


def clip_lane_to_image(
    lane: list[tuple[float, float]],
    h: int,
    w: int,
) -> list[tuple[float, float]]:
    """Clip pixel-space lane coordinates to image bounds while preserving point order."""
    if h <= 0 or w <= 0:
        raise ValueError(f"Image size must be positive, got h={h}, w={w}")

    clipped: list[tuple[float, float]] = []
    for x, y in sort_lane_bottom_to_top(lane):
        clipped.append((float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))))
    return clipped


def resize_lanes(
    lanes: list[list[tuple[float, float]]],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[list[tuple[float, float]]]:
    """Scale lanes from source image size to destination image size."""
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"Invalid resize shape: src=({src_w}, {src_h}), dst=({dst_w}, {dst_h})")

    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out: list[list[tuple[float, float]]] = []
    for lane in lanes:
        scaled = [(float(x) * sx, float(y) * sy) for x, y in lane if x >= 0 and y >= 0]
        if len(scaled) >= 2:
            out.append(sort_lane_bottom_to_top(scaled))
    return out


def lane_to_points(lane_xs: list[float], h_samples: list[float]) -> list[tuple[float, float]]:
    """Convert one TuSimple lane x list and shared h_samples to valid points."""
    points: list[tuple[float, float]] = []
    for x, y in zip(lane_xs, h_samples):
        if _is_valid_number(x) and _is_valid_number(y) and float(x) >= 0.0 and float(y) >= 0.0:
            points.append((float(x), float(y)))
    return points


def points_to_mask(
    points: list[tuple[float, float]],
    h: int,
    w: int,
    line_width: int = 12,
) -> np.ndarray:
    """Rasterize one lane centerline to a binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    points = clip_lane_to_image(points, h=h, w=w)
    if len(points) < 2:
        return mask
    pts = np.asarray(points, dtype=np.int32)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=line_width)
    return mask


def mask_to_yolo_segments(mask: np.ndarray, class_id: int = 0) -> list[str]:
    """Convert a binary instance mask to YOLO segmentation label lines."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue
        coords = approx.reshape(-1, 2)
        if coords.shape[0] < 3:
            continue

        values: list[float] = []
        for x, y in coords:
            values.append(float(np.clip(x / w, 0.0, 1.0)))
            values.append(float(np.clip(y / h, 0.0, 1.0)))

        if len(values) >= 6:
            lines.append(str(class_id) + " " + " ".join(f"{v:.6f}" for v in values))
    return lines
