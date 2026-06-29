from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HEAD_MARKERS = (
    ".level_embed",
    ".query_embed.",
    ".decoder.",
    ".point_mlp.",
    ".point_valid_mlp.",
    ".point_embed.",
    ".point_coord_mlp.",
    ".point_refine_norm.",
    ".point_image_norm.",
    ".point_refine_mlp.",
    ".point_valid_refine_mlp.",
    ".exist_mlp.",
    ".start_mlp.",
    ".end_mlp.",
    ".count_mlp.",
    ".point_head.",
    ".exist_head.",
    ".valid_head.",
    ".start_head.",
    ".end_head.",
    ".count_head.",
    ".aux_mask.",
    ".aux_edge.",
    ".point_reference_logits",
    ".fixed_y_anchors",
)
HEAD_PREFIXES = (
    "level_embed",
    "query_embed.",
    "decoder.",
    "point_mlp.",
    "point_valid_mlp.",
    "point_embed.",
    "point_coord_mlp.",
    "point_refine_norm.",
    "point_image_norm.",
    "point_refine_mlp.",
    "point_valid_refine_mlp.",
    "exist_mlp.",
    "start_mlp.",
    "end_mlp.",
    "count_mlp.",
    "aux_mask.",
    "aux_edge.",
    "point_reference_logits",
    "fixed_y_anchors",
    "point_head.",
    "exist_head.",
    "valid_head.",
    "start_head.",
    "end_head.",
    "count_head.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip old GCS lane head tensors from a checkpoint.")
    parser.add_argument("--input", "-i", required=True, help="Source best.pt/last.pt checkpoint.")
    parser.add_argument("--output", "-o", required=True, help="Output checkpoint containing non-head tensors.")
    parser.add_argument("--metadata-json", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def state_dict_from_checkpoint(ckpt) -> dict[str, torch.Tensor]:
    source = ckpt
    if isinstance(ckpt, dict):
        source = ckpt.get("ema") or ckpt.get("model") or ckpt.get("state_dict") or ckpt
    if isinstance(source, nn.Module):
        return source.float().state_dict()
    if isinstance(source, dict):
        return {k: v for k, v in source.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}
    raise TypeError(f"Unsupported checkpoint payload: {type(source).__name__}")


def is_gcs_head_key(key: str) -> bool:
    key = key[7:] if key.startswith("module.") else key
    return key.startswith(HEAD_PREFIXES) or any(marker in key for marker in HEAD_MARKERS)


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        raise FileNotFoundError(src)
    ckpt = torch_load(src)
    state = state_dict_from_checkpoint(ckpt)
    kept = {k: v.detach().cpu() for k, v in state.items() if not is_gcs_head_key(k)}
    dropped = sorted(k for k in state if is_gcs_head_key(k))
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": kept,
            "meta": {
                "source": str(src.resolve()),
                "kept_tensors": len(kept),
                "dropped_tensors": len(dropped),
                "removed_head_tensors": len(dropped),
                "purpose": "GCS ordered_slot transfer without old query head tensors",
            },
        },
        dst,
    )
    report = {
        "input": str(src.resolve()),
        "output": str(dst.resolve()),
        "kept_tensors": len(kept),
        "dropped_tensors": len(dropped),
        "removed_head_tensors": len(dropped),
        "dropped_keys": dropped,
        "removed_head_keys": dropped,
    }
    if args.metadata_json:
        report_path = Path(args.metadata_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"dropped_keys", "removed_head_keys"}}, indent=2))


if __name__ == "__main__":
    main()
