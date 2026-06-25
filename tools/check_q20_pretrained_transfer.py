"""Check pretrained transfer into the Q20 side-geometry K56 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer
from ultralytics.nn.modules.gcs_lane import GCSLaneHead
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.nn.tasks import GCSLaneModel


DEFAULT_CFG = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q20-k56-sidegeom.yaml"
DEFAULT_Q18_WEIGHTS = (
    ROOT
    / "runs"
    / "gcs_lane"
    / "gcs_yolo_lane_s_q18_k56_side_gt4endpoint_validneg_countce_v1"
    / "weights"
    / "best.pt"
)


def parse_args() -> argparse.Namespace:
    """Parse checkpoint transfer check arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="Q20 model YAML.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_Q18_WEIGHTS, help="Source GCS checkpoint.")
    parser.add_argument(
        "--source-kind",
        choices=("q18", "q12", "gt4pt025"),
        required=True,
        help="Source checkpoint family used to validate expected Q-shape mismatches.",
    )
    parser.add_argument("--report-json", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def is_query_embed_key(key: str) -> bool:
    """Return True for query embedding tensors."""
    return key.endswith("query_embed.weight")


def is_point_reference_key(key: str) -> bool:
    """Return True for query point-reference buffers."""
    return key.endswith("point_reference_logits")


def is_count_mlp(key: str) -> bool:
    """Return True for explicit count-head tensors."""
    return "count_mlp." in key


def expected_source_q(source_kind: str) -> int:
    """Return the required source query count for a checkpoint family."""
    if source_kind == "q18":
        return 18
    if source_kind in {"q12", "gt4pt025"}:
        return 12
    raise ValueError(f"Unsupported source_kind={source_kind!r}")


def is_allowed_shape_mismatch(item: dict, source_kind: str) -> tuple[bool, str]:
    """Return whether a shape mismatch is expected for source Q -> Q20 transfer."""
    key = str(item["key"])
    source_shape = list(item["source_shape"])
    target_shape = list(item["target_shape"])
    src_q = expected_source_q(source_kind)
    tgt_q = 20

    if key.endswith("query_embed.weight") and len(source_shape) == len(target_shape) == 2:
        if source_shape[0] == src_q and target_shape[0] == tgt_q and source_shape[1] == target_shape[1]:
            return True, f"expected query_embed Q{src_q}->Q{tgt_q}"
        return (
            False,
            f"unexpected query_embed shape for source_kind={source_kind}: "
            f"{source_shape} -> {target_shape}; expected [{src_q}, C] -> [20, C]",
        )

    if is_point_reference_key(key):
        if (
            len(source_shape) == 2
            and len(target_shape) == 2
            and source_shape[0] == src_q
            and target_shape[0] == tgt_q
            and source_shape[1] == 56
            and target_shape[1] == 56
        ):
            return True, f"expected point_reference_logits Q{src_q}->Q{tgt_q}"
        return (
            False,
            f"unexpected point_reference_logits shape for source_kind={source_kind}: "
            f"{source_shape} -> {target_shape}; expected [{src_q}, 56] -> [20, 56]",
        )

    return False, f"unexpected shape mismatch key: {key}"


def shape_record(key: str, source_shape: torch.Size, target_shape: torch.Size) -> dict:
    """Build a JSON-serializable shape mismatch record."""
    return {"key": key, "source_shape": list(source_shape), "target_shape": list(target_shape)}


def write_report(path: Path | None, report: dict) -> None:
    """Optionally write the transfer report."""
    if path is None:
        return
    report_path = path if path.is_absolute() else ROOT / path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def add_point_reference_buffers(state: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Add non-persistent point_reference_logits buffers to a state dict copy."""
    out = dict(state)
    for name, module in model.named_modules():
        if isinstance(module, GCSLaneHead) and hasattr(module, "point_reference_logits"):
            key = f"{name}.point_reference_logits" if name else "point_reference_logits"
            out[key] = module.point_reference_logits.detach().cpu()
    return out


def state_dict_from_weights_with_refs(weights: Path) -> dict[str, torch.Tensor]:
    """Extract a checkpoint state dict and include point_reference_logits buffers."""
    model, _ = load_checkpoint(weights)
    state = GCSLaneTrainer._state_dict_from_weights(model)
    return add_point_reference_buffers(state, model)


def main() -> None:
    """Load a source checkpoint into the Q20 target and assert transfer health."""
    args = parse_args()
    cfg = args.cfg if args.cfg.is_absolute() else ROOT / args.cfg
    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    if not cfg.exists():
        raise FileNotFoundError(f"Missing Q20 model YAML: {cfg}")
    if not weights.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights}")

    model = GCSLaneModel(str(cfg), nc=1, ch=3, verbose=False)
    target_state = add_point_reference_buffers(model.state_dict(), model)
    source_state = state_dict_from_weights_with_refs(weights)
    if not GCSLaneTrainer._state_dict_has_gcs_modules(source_state):
        raise RuntimeError(f"Expected a GCS checkpoint, got a non-GCS checkpoint: {weights}")

    loadable: dict[str, torch.Tensor] = {}
    source_missing_in_target: list[str] = []
    skipped_shape: list[dict] = []

    for key, value in source_state.items():
        if key not in target_state:
            source_missing_in_target.append(key)
            continue
        if value.shape != target_state[key].shape:
            skipped_shape.append(shape_record(key, value.shape, target_state[key].shape))
            continue
        loadable[key] = value.to(dtype=target_state[key].dtype)

    model.load_state_dict(loadable, strict=False)

    target_missing_from_source = sorted(key for key in target_state if key not in source_state)
    target_count_keys = sorted(key for key in target_state if is_count_mlp(key))
    count_loaded = sorted(key for key in target_count_keys if key in loadable)
    count_missing = sorted(set(target_count_keys) - set(count_loaded))
    allowed_target_missing = []
    if args.source_kind in {"q12", "gt4pt025"}:
        allowed_target_missing = [key for key in target_missing_from_source if is_count_mlp(key)]

    allowed_shape = []
    unexpected_shape = []
    for item in skipped_shape:
        ok, reason = is_allowed_shape_mismatch(item, str(args.source_kind))
        item = {**item, "reason": reason}
        if ok:
            allowed_shape.append(item)
        else:
            unexpected_shape.append(item)

    unexpected_target_missing = sorted(set(target_missing_from_source) - set(allowed_target_missing))
    if args.source_kind == "q18":
        unexpected_count_missing = count_missing
    else:
        unexpected_count_missing = []

    report = {
        "ok": True,
        "cfg": str(cfg),
        "weights": str(weights),
        "source_kind": str(args.source_kind),
        "source_tensors": len(source_state),
        "target_tensors": len(target_state),
        "loaded_count": len(loadable),
        "loaded_count_mlp": count_loaded,
        "skipped_shape_count": len(skipped_shape),
        "skipped_shape": skipped_shape,
        "allowed_shape": allowed_shape,
        "source_missing_in_target_count": len(source_missing_in_target),
        "source_missing_in_target": sorted(source_missing_in_target),
        "target_missing_from_source_count": len(target_missing_from_source),
        "target_missing_from_source": target_missing_from_source,
        "allowed_target_missing": allowed_target_missing,
        "count_mlp_target_keys": target_count_keys,
        "count_mlp_loaded": count_loaded,
        "count_mlp_missing_from_source": count_missing,
        "unexpected_shape": unexpected_shape,
        "unexpected_target_missing": unexpected_target_missing,
        "unexpected_count_mlp_missing": unexpected_count_missing,
        "explanation": {
            "allowed_shape_mismatch": "query_embed.weight and point_reference_logits must match source_kind Q -> Q20 exactly.",
            "q18_count_mlp_rule": "For source-kind=q18, every count_mlp.* tensor must load.",
            "q12_count_mlp_rule": "For source-kind=q12/gt4pt025, count_mlp.* may be missing from the source checkpoint.",
        },
    }

    if len(loadable) == 0:
        report["ok"] = False
        write_report(args.report_json, report)
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError("No tensors transferred from checkpoint.")
    if unexpected_shape:
        report["ok"] = False
        write_report(args.report_json, report)
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError(f"Unexpected shape mismatches found: {unexpected_shape}")
    if unexpected_target_missing:
        report["ok"] = False
        write_report(args.report_json, report)
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError(f"Unexpected target tensors missing from source checkpoint: {unexpected_target_missing}")
    if unexpected_count_missing:
        report["ok"] = False
        write_report(args.report_json, report)
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError(f"Q18 source did not load count_mlp.* tensors: {unexpected_count_missing}")
    if args.source_kind == "q18" and not count_loaded:
        report["ok"] = False
        write_report(args.report_json, report)
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError("source_kind=q18 expected count_mlp.* tensors to load, but none were loaded.")

    write_report(args.report_json, report)
    print(json.dumps(report, indent=2))
    print("OK: Q20 pretrained transfer contract is valid.")


if __name__ == "__main__":
    main()
