from __future__ import annotations

from pathlib import Path


def _norm_path_text(path: str | Path) -> str:
    return Path(path).as_posix().lower()


def is_likely_tusimple_test_gt_json(path: str | Path) -> bool:
    """Return True for explicit TuSimple test-label paths that must not drive search tools."""
    text = _norm_path_text(path)
    name = Path(path).name.lower()
    return (
        name in {"test_label.json", "test_label_new.json", "test.json"}
        or "test_label" in name
        or "/test_label" in text
        or "/seg_label/test.json" in text
    )


def reject_tusimple_test_search_gt_json(
    gt_json: str | Path | None,
    *,
    context: str,
    allow: bool = False,
) -> None:
    """Reject explicit test GT json paths for threshold, postprocess, and diagnostic search tools."""
    if gt_json is None or allow:
        return
    if is_likely_tusimple_test_gt_json(gt_json):
        raise ValueError(
            f"{context} cannot use an explicit TuSimple test GT json ({gt_json}) for search or diagnostics. "
            "Use train/val GT for selection, and reserve test labels for one-shot final evaluation."
        )
