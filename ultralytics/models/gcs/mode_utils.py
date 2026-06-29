"""GCS mode inference and decode-mode contract helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def normalize_gcs_mode(mode: Any) -> str:
    """Normalize query/ordered-slot spelling."""
    value = str(mode or "query").strip().lower()
    if value in {"ordered-slot", "orderedslot"}:
        value = "ordered_slot"
    if value not in {"query", "ordered_slot"}:
        raise RuntimeError(f"Unsupported GCS mode {value!r}; expected 'query' or 'ordered_slot'.")
    return value


def assert_ordered_slot_scale_contract(gcs_mode: Any, scale: Any) -> None:
    """Fail fast until ordered-slot has count-preserving scale augmentation."""
    mode = normalize_gcs_mode(gcs_mode)
    scale_value = float(scale or 0.0)
    if mode == "ordered_slot" and scale_value > 0.0:
        raise RuntimeError(
            "ordered_slot with --scale > 0 is disabled in ordered_slot_runtime_stability_fix_v1: "
            "scale fixed-y resampling may drop the lane count to 0/1, which violates the "
            "ordered_slot 2/3/4/5 count contract. Use --scale 0.0 until a count-preserving "
            "augmentation path is explicitly implemented."
        )


def infer_gcs_mode_from_model(model: nn.Module) -> str:
    """Infer the actual GCS mode from a loaded model and reject mixed heads."""
    modes: list[str] = []
    if hasattr(model, "modules"):
        for module in model.modules():
            if hasattr(module, "gcs_mode"):
                modes.append(normalize_gcs_mode(getattr(module, "gcs_mode")))
    unique = sorted(set(modes))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(f"Multiple gcs_mode values found in one model: {unique}.")

    if hasattr(model, "modules"):
        for module in model.modules():
            if all(hasattr(module, name) for name in ("count_mlp", "start_mlp", "end_mlp")):
                return "ordered_slot"
            if all(hasattr(module, name) for name in ("count_head", "start_head", "end_head")):
                return "ordered_slot"
    return normalize_gcs_mode(getattr(model, "gcs_mode", "query"))


def _state_dict_has_ordered_slot(state: dict[str, torch.Tensor]) -> bool:
    ordered_markers = (
        ".start_mlp.",
        ".end_mlp.",
        ".count_mlp.",
        ".start_head.",
        ".end_head.",
        ".count_head.",
    )
    return any(any(marker in key for marker in ordered_markers) for key in state)


def infer_gcs_mode_from_ckpt(ckpt: dict | nn.Module | str | Path) -> str:
    """Infer GCS mode from a checkpoint payload, module, or checkpoint path."""
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location="cpu")
    if isinstance(ckpt, nn.Module):
        return infer_gcs_mode_from_model(ckpt)
    if not isinstance(ckpt, dict):
        return "query"

    meta = ckpt.get("meta") if isinstance(ckpt.get("meta"), dict) else {}
    train_args = ckpt.get("train_args") if isinstance(ckpt.get("train_args"), dict) else {}
    for source in (meta, train_args, ckpt):
        for key in ("gcs_mode", "source_gcs_mode"):
            if isinstance(source, dict) and source.get(key) is not None:
                return normalize_gcs_mode(source[key])

    model = ckpt.get("ema") or ckpt.get("model")
    if isinstance(model, nn.Module):
        return infer_gcs_mode_from_model(model)

    state = ckpt.get("state_dict") if isinstance(ckpt.get("state_dict"), dict) else ckpt
    tensors = {k: v for k, v in state.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}
    return "ordered_slot" if _state_dict_has_ordered_slot(tensors) else "query"


def resolve_decode_mode(args_decode_mode: str | None, model: nn.Module) -> str:
    """Resolve --decode-mode and reject explicit mismatches against model mode."""
    requested = str(args_decode_mode or "auto").strip().lower()
    if requested in {"ordered-slot", "orderedslot"}:
        requested = "ordered_slot"
    if requested not in {"auto", "query", "ordered_slot"}:
        raise RuntimeError(f"Unsupported decode-mode {requested!r}.")
    model_mode = infer_gcs_mode_from_model(model)
    if requested == "auto":
        return model_mode
    if requested != model_mode:
        raise RuntimeError(
            f"decode-mode mismatch: requested {requested!r}, but model gcs_mode is {model_mode!r}. "
            "Use --decode-mode auto or the matching mode."
        )
    return requested
