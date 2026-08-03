"""Decode-summary and decode-yaml helpers for GCS eval scripts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


QUERY_DECODE_SCHEMA = "query_decode_v1"
ORDERED_SLOT_DECODE_SCHEMA = "ordered_slot_decode_v1"
QUERY_DECODE_KEYS = frozenset(
    {
        "conf",
        "point_valid_thr",
        "nms_dist_px",
        "max_det",
        "min_points",
        "valid_before_maxdet",
        "count_aware_topk",
        "count_aware_min_k",
        "count_aware_max_k",
        "count_aware_length_norm",
        "count_aware_extra_margin",
        "count_mode",
        "full_lane_decode",
    }
)
QUERY_SWEEP_DECODE_KEYS = frozenset(
    {"confs", "point_valid_thrs", "nms_dist_pxs", "max_dets", "count_aware_extra_margins", "count_modes"}
)
ORDERED_SLOT_QUERY_ONLY_KEYS = QUERY_DECODE_KEYS | QUERY_SWEEP_DECODE_KEYS
ORDERED_SLOT_QUERY_DECODE_DEFAULTS = {
    "conf": 0.25,
    "point_valid_thr": 0.5,
    "nms_dist_px": 18.0,
    "min_points": 6,
    "max_det": 8,
    "valid_before_maxdet": False,
    "count_aware_topk": False,
    "count_aware_min_k": 3,
    "count_aware_max_k": 5,
    "count_aware_length_norm": 12.0,
    "count_aware_extra_margin": 0,
    "count_mode": "score_sum",
}


def ordered_slot_decode_runtime_config(context: str) -> dict[str, Any]:
    """Return the ordered-slot runtime order policy shared by decode calls and summaries."""
    normalized = str(context or "").strip().lower().replace("-", "_")
    strict_contexts = {
        "official_eval",
        "official_sweep",
        "official_best",
        "eval_gcs",
        "val",
        "predict",
        "infer",
        "overfit",
        "contract",
    }
    internal_val_contexts = {"training_val", "internal_val"}
    training_candidate_contexts = {"training_official_best", "official_best_candidate"}
    sorted_export_contexts = {"debug_sorted_export", "debug", "visualize"}
    if normalized in strict_contexts:
        return {
            "order_check": "error",
            "output_order": "slot",
            "uses_runtime_sort": False,
            "order_violation_policy": "fail_fast",
            "result_type": "strict_ordered_slot",
            "not_for_main_ordered_slot_claim": False,
        }
    if normalized in internal_val_contexts:
        return {
            "order_check": "none",
            "output_order": "slot",
            "uses_runtime_sort": False,
            "order_violation_policy": "diagnostic_only",
            "result_type": "internal_training_val",
            "not_for_main_ordered_slot_claim": True,
        }
    if normalized in training_candidate_contexts:
        return {
            "order_check": "warn",
            "output_order": "slot",
            "uses_runtime_sort": False,
            "order_violation_policy": "warn_only",
            "result_type": "training_official_best_candidate",
            "not_for_main_ordered_slot_claim": True,
        }
    if normalized in sorted_export_contexts:
        return {
            "order_check": "warn",
            "output_order": "left_to_right",
            "uses_runtime_sort": True,
            "order_violation_policy": "warn_then_sort_by_bottom_x",
            "result_type": "postprocessed_sorted_export",
            "not_for_main_ordered_slot_claim": True,
        }
    raise ValueError(f"Unknown ordered_slot decode context: {context}")


def ordered_slot_order_diagnostics_summary(
    *,
    pred_json_mode: bool,
    decode_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ordered-slot order diagnostics without treating unavailable checks as zero violations."""
    if pred_json_mode:
        return {
            "ordered_slot_order_checked": False,
            "ordered_slot_order_violations": None,
            "ordered_slot_order_violation_images": None,
            "ordered_slot_order_diagnostics": "not_available_from_pred_json",
        }

    stats = decode_stats or {}
    return {
        "ordered_slot_order_checked": True,
        "ordered_slot_order_violations": int(stats.get("ordered_slot_order_violations", 0)),
        "ordered_slot_order_violation_images": int(stats.get("ordered_slot_order_violation_images", 0)),
        "ordered_slot_order_diagnostics": "available_from_decode",
    }


def _arg(args: Any, name: str, default: Any) -> Any:
    if args is None:
        return default
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def build_ordered_slot_decode_summary(
    min_lanes: int = 2,
    max_lanes: int = 5,
    num_slots: int = 5,
    min_interval_points: int = 2,
    order_margin_px: float = 2.0,
    output_order: str = "slot",
    order_check: str = "error",
) -> dict[str, Any]:
    """Return the complete effective ordered-slot decode contract for summaries."""
    min_lanes = int(min_lanes)
    max_lanes = int(max_lanes)
    num_slots = int(num_slots)
    min_interval_points = int(min_interval_points)
    order_margin_px = float(order_margin_px)
    output_order = str(output_order or "slot").strip().lower()
    order_check = str(order_check or "error").strip().lower()
    uses_runtime_sort = output_order == "left_to_right"
    if uses_runtime_sort and order_check == "warn":
        order_violation_policy = "warn_then_sort_by_bottom_x"
    elif order_check == "error":
        order_violation_policy = "fail_fast"
    elif order_check == "warn":
        order_violation_policy = "warn_only"
    else:
        order_violation_policy = str(order_check)
    summary = {
        "schema": ORDERED_SLOT_DECODE_SCHEMA,
        "decode_mode": "ordered_slot",
        "gcs_min_lanes": min_lanes,
        "gcs_max_lanes": max_lanes,
        "gcs_num_slots": num_slots,
        "min_interval_points": min_interval_points,
        "gcs_bottom_order_margin_px": order_margin_px,
        "interval_repair": "clamp_expand",
        "count_source": "argmax(pred_count_logits)+gcs_min_lanes",
        "output_slots": "slot[0:num_lanes]",
        "output_order": output_order,
        "order_check": order_check,
        "uses_runtime_sort": uses_runtime_sort,
        "order_violation_policy": order_violation_policy,
        "visibility_source": "repaired_start_end_interval",
        "uses_nms": False,
        "uses_conf_threshold": False,
        "uses_point_valid_threshold": False,
        "uses_max_det": False,
        "uses_min_points_filter": False,
        "query_decode_args": "not_applicable",
    }
    if uses_runtime_sort:
        summary["result_type"] = "postprocessed_sorted_export"
        summary["not_for_main_ordered_slot_claim"] = True
    elif order_check == "error":
        summary["result_type"] = "strict_ordered_slot"
        summary["not_for_main_ordered_slot_claim"] = False
    else:
        summary["result_type"] = "diagnostic_ordered_slot"
        summary["not_for_main_ordered_slot_claim"] = True
    return summary


def ordered_slot_effective_decode(
    min_lanes: int = 2,
    max_lanes: int = 5,
    num_slots: int = 5,
    min_interval_points: int = 2,
    order_margin_px: float = 2.0,
    output_order: str = "slot",
    order_check: str = "error",
) -> dict[str, Any]:
    """Return the effective ordered-slot decode contract for summaries."""
    return build_ordered_slot_decode_summary(
        min_lanes=min_lanes,
        max_lanes=max_lanes,
        num_slots=num_slots,
        min_interval_points=min_interval_points,
        order_margin_px=order_margin_px,
        output_order=output_order,
        order_check=order_check,
    )


def ordered_slot_decode_params(args: Any = None, decode_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ordered-slot decode parameters from a decode yaml first, then runtime args."""
    cfg = decode_cfg or {}

    def value(*cfg_keys: str, arg_name: str, default: Any) -> Any:
        for key in cfg_keys:
            if key in cfg:
                return cfg[key]
        return _arg(args, arg_name, default)

    return {
        "min_lanes": int(value("gcs_min_lanes", arg_name="gcs_min_lanes", default=2)),
        "max_lanes": int(value("gcs_max_lanes", arg_name="gcs_max_lanes", default=5)),
        "num_slots": int(value("gcs_num_slots", arg_name="gcs_num_slots", default=5)),
        "min_interval_points": int(
            value("min_interval_points", "gcs_min_interval_points", arg_name="gcs_min_interval_points", default=2)
        ),
        "order_margin_px": float(
            value("gcs_bottom_order_margin_px", "order_margin_px", arg_name="gcs_bottom_order_margin_px", default=2.0)
        ),
    }


def ordered_slot_decode_cfg(args: Any = None) -> dict[str, Any]:
    """Build the official ordered-slot decode-yaml schema."""
    params = ordered_slot_decode_params(args)
    runtime_cfg = ordered_slot_decode_runtime_config(context="official_best")
    cfg = {
        "schema": ORDERED_SLOT_DECODE_SCHEMA,
        "decode_mode": "ordered_slot",
        "gcs_min_lanes": params["min_lanes"],
        "gcs_max_lanes": params["max_lanes"],
        "gcs_num_slots": params["num_slots"],
        "min_interval_points": params["min_interval_points"],
        "gcs_bottom_order_margin_px": params["order_margin_px"],
        "interval_repair": "clamp_expand",
        "output_order": runtime_cfg["output_order"],
        "order_check": runtime_cfg["order_check"],
        "uses_runtime_sort": runtime_cfg["uses_runtime_sort"],
        "order_violation_policy": runtime_cfg["order_violation_policy"],
        "effective_decode": build_ordered_slot_decode_summary(
            min_lanes=params["min_lanes"],
            max_lanes=params["max_lanes"],
            num_slots=params["num_slots"],
            min_interval_points=params["min_interval_points"],
            order_margin_px=params["order_margin_px"],
            output_order=runtime_cfg["output_order"],
            order_check=runtime_cfg["order_check"],
        ),
    }
    validate_decode_yaml_for_model(cfg, model_mode="ordered_slot")
    return cfg


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def query_decode_cfg(
    best_row: Mapping[str, Any],
    valid_before_maxdet: Any = None,
    full_lane_decode: Any = None,
) -> dict[str, Any]:
    """Build the official query decode-yaml schema from a selected sweep row."""
    if valid_before_maxdet is None:
        valid_before_maxdet = best_row.get("valid_before_maxdet", False)
    if full_lane_decode is None:
        full_lane_decode = best_row.get("full_lane_decode", False)
    cfg = {
        "schema": QUERY_DECODE_SCHEMA,
        "decode_mode": "query",
        "conf": float(best_row["conf"]),
        "point_valid_thr": float(best_row["point_valid_thr"]),
        "nms_dist_px": float(best_row["nms_dist_px"]),
        "max_det": int(best_row["max_det"]),
        "min_points": int(best_row["min_points"]),
        "valid_before_maxdet": _bool_value(valid_before_maxdet),
        "count_aware_topk": _bool_value(best_row.get("count_aware_topk", False)),
        "count_aware_min_k": int(best_row.get("count_aware_min_k", 3) or 3),
        "count_aware_max_k": int(best_row.get("count_aware_max_k", 5) or 5),
        "count_aware_length_norm": float(best_row.get("count_aware_length_norm", 12.0) or 12.0),
        "count_aware_extra_margin": int(best_row.get("count_aware_extra_margin", 0) or 0),
        "count_mode": str(best_row.get("count_mode", "score_sum") or "score_sum"),
        "full_lane_decode": _bool_value(full_lane_decode),
    }
    validate_decode_yaml_for_model(cfg, model_mode="query")
    return cfg


def build_official_best_decode_cfg(best_row: Mapping[str, Any], model_mode: str, args: Any = None) -> dict[str, Any]:
    """Build a schema-specific decode config for official_best_decode.yaml."""
    model_mode = str(model_mode or "query").strip().lower()
    if model_mode in {"ordered-slot", "orderedslot"}:
        model_mode = "ordered_slot"
    if model_mode == "ordered_slot":
        return ordered_slot_decode_cfg(args)
    if model_mode == "query":
        valid_before_maxdet = best_row.get(
            "valid_before_maxdet",
            _arg(args, "gcs_official_valid_before_maxdet", False),
        )
        full_lane_decode = best_row.get(
            "full_lane_decode",
            _arg(args, "gcs_full_lane_decode", False),
        )
        return query_decode_cfg(
            best_row,
            valid_before_maxdet=valid_before_maxdet,
            full_lane_decode=full_lane_decode,
        )
    raise ValueError(f"Unknown model_mode={model_mode!r}.")


def extract_decode_cfg(yaml_data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the actual decode schema from either a plain cfg or full official-best artifact."""
    decode = yaml_data.get("decode")
    if isinstance(decode, Mapping):
        return dict(decode)
    return dict(yaml_data)


def validate_decode_yaml_for_model(decode_cfg: Mapping[str, Any], model_mode: str) -> None:
    """Validate a decode-yaml schema against the active model mode."""
    model_mode = str(model_mode or "query").strip().lower()
    if model_mode in {"ordered-slot", "orderedslot"}:
        model_mode = "ordered_slot"
    decode_mode = str(decode_cfg.get("decode_mode", "")).strip().lower()
    if decode_mode in {"ordered-slot", "orderedslot"}:
        decode_mode = "ordered_slot"
    schema = decode_cfg.get("schema")

    if decode_mode != model_mode:
        raise RuntimeError(f"decode yaml mismatch: decode_mode={decode_mode!r}, model_mode={model_mode!r}.")

    if model_mode == "ordered_slot":
        bad_keys = sorted(ORDERED_SLOT_QUERY_ONLY_KEYS & set(decode_cfg))
        if bad_keys:
            raise RuntimeError(
                "Invalid ordered_slot decode yaml: contains query-only keys "
                f"{bad_keys}. Regenerate official_best_decode.yaml with ordered_slot_decode_v1 schema."
            )
        if schema != ORDERED_SLOT_DECODE_SCHEMA:
            raise RuntimeError(f"Invalid ordered_slot schema={schema!r}. Expected {ORDERED_SLOT_DECODE_SCHEMA}.")
        min_lanes = int(decode_cfg.get("gcs_min_lanes", 0))
        max_lanes = int(decode_cfg.get("gcs_max_lanes", 0))
        num_slots = int(decode_cfg.get("gcs_num_slots", 0))
        if (min_lanes, max_lanes, num_slots) != (2, 5, 5):
            raise RuntimeError(
                "Invalid ordered_slot decode yaml: ordered_slot_v2 supports 2/3/4/5 lanes "
                f"with gcs_min_lanes=2, gcs_max_lanes=5, gcs_num_slots=5; got "
                f"{min_lanes}/{max_lanes}/{num_slots}."
            )
        return

    if model_mode == "query":
        if schema != QUERY_DECODE_SCHEMA:
            raise RuntimeError(f"Invalid query schema={schema!r}. Expected {QUERY_DECODE_SCHEMA}.")
        required = QUERY_DECODE_KEYS - {
            "valid_before_maxdet",
            "count_mode",
            "count_aware_extra_margin",
            "full_lane_decode",
        }
        missing = sorted(required.difference(decode_cfg))
        if missing:
            raise RuntimeError(f"Invalid query decode yaml: missing keys {missing}.")
        count_mode = str(decode_cfg.get("count_mode", "score_sum") or "score_sum")
        if count_mode not in {"score_sum", "count_logits"}:
            raise RuntimeError(
                f"Invalid query decode yaml: count_mode={count_mode!r}. "
                "Supported formal count modes are 'score_sum' and 'count_logits'. "
                "Use --oracle-count for diagnostic GT-count decode."
            )
        return

    raise RuntimeError(f"Unsupported model_mode={model_mode!r} for decode yaml validation.")


def load_decode_yaml(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a decode yaml and return (full_yaml, decode_cfg)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise RuntimeError(f"Decode yaml {path} must contain a mapping.")
    root = dict(data)
    return root, extract_decode_cfg(root)


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return False
        if len(actual) != len(expected):
            return False
        return all(_same_value(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, float):
        return abs(float(actual) - float(expected)) <= 1e-12
    return actual == expected


def non_default_query_decode_args(args: Any, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Return query-only args whose values differ from parser defaults."""
    changed: dict[str, Any] = {}
    for name, default in defaults.items():
        actual = args.get(name, default) if isinstance(args, Mapping) else getattr(args, name, default)
        if not _same_value(actual, default):
            changed[name] = actual
    return changed


def guard_no_query_decode_args_for_ordered_slot(
    args: Any,
    context: str,
    defaults: Mapping[str, Any] | None = None,
) -> None:
    """Fail when ordered_slot receives non-default query-only decode arguments."""
    changed = non_default_query_decode_args(args, defaults or ORDERED_SLOT_QUERY_DECODE_DEFAULTS)
    if changed:
        names = ", ".join(sorted(changed))
        raise RuntimeError(
            f"ordered_slot decode does not use query decode args: {names}. "
            f"Remove them for {context} or use --decode-mode query with a query model."
        )


def raise_for_ordered_slot_query_args(args: Any, defaults: Mapping[str, Any], context: str) -> None:
    """Fail when ordered_slot official eval receives non-default query-only decode args."""
    guard_no_query_decode_args_for_ordered_slot(args, context=context, defaults=defaults)
