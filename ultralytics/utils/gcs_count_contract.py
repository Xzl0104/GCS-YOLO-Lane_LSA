# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Shared helpers for GCS GT4/GT5 count-contract diagnostics and losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ShortsideVisibleDecision:
    """Decision details for short-side weak-visible lane selection."""

    eligible: bool
    selected_by_abs: bool
    selected_by_median: bool
    visible: int
    image_visible_median: float | None
    is_ultra_short: bool = False
    is_reliable: bool = False
    skipped_visible_lt2: bool = False


def shortside_image_visible_median(
    visible_counts: Iterable[int | float],
    *,
    min_valid_points: int,
) -> float | None:
    """Return the standard image-level GT visible-count median after the reliable-lane filter."""
    min_valid_points = int(min_valid_points)
    counts = sorted(float(v) for v in visible_counts if int(v) >= min_valid_points)
    if not counts:
        return None
    mid = len(counts) // 2
    if len(counts) % 2 == 1:
        return float(counts[mid])
    return float(0.5 * (counts[mid - 1] + counts[mid]))


def shortside_visible_decision(
    visible: int | float,
    visible_counts: Iterable[int | float],
    *,
    min_valid_points: int = 6,
    ultra_min_valid_points: int = 2,
    reliable_min_valid_points: int | None = None,
    visible_max: int = 42,
    use_median: bool = True,
    median_margin: float = 0.0,
) -> ShortsideVisibleDecision:
    """Return whether one GT lane is weak-visible under the shared shortside contract.

    The absolute branch keeps legacy behavior. The median branch selects lanes that
    are not absolutely short but are shorter than the current image's filtered GT
    visible-count median.
    """
    visible_i = int(visible)
    reliable_min_i = int(reliable_min_valid_points if reliable_min_valid_points is not None else min_valid_points)
    ultra_min_i = int(ultra_min_valid_points)
    image_median = shortside_image_visible_median(visible_counts, min_valid_points=reliable_min_i)
    if visible_i < ultra_min_i:
        return ShortsideVisibleDecision(
            eligible=False,
            selected_by_abs=False,
            selected_by_median=False,
            visible=visible_i,
            image_visible_median=image_median,
            skipped_visible_lt2=True,
        )

    selected_by_abs = visible_i <= int(visible_max)
    selected_by_median = False
    if bool(use_median) and image_median is not None:
        selected_by_median = float(visible_i) < float(image_median) - float(median_margin)
    is_reliable = visible_i >= reliable_min_i

    return ShortsideVisibleDecision(
        eligible=bool(selected_by_abs or selected_by_median),
        selected_by_abs=bool(selected_by_abs),
        selected_by_median=bool(selected_by_median and not selected_by_abs),
        visible=visible_i,
        image_visible_median=image_median,
        is_ultra_short=bool(not is_reliable),
        is_reliable=bool(is_reliable),
    )
