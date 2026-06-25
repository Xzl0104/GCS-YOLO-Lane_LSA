"""Check Q12-to-Q18 pretrained transfer for the side-dense K56 model."""

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
from ultralytics.nn.tasks import GCSLaneModel


DEFAULT_CFG = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q18-k56-side.yaml"
DEFAULT_WEIGHTS = (
    ROOT
    / "runs"
    / "gcs_lane"
    / "gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025"
    / "weights"
    / "best.pt"
)
GCS_KEY_MARKERS = (
    ".lsa.",
    ".dilated_context.",
    ".level_embed",
    ".query_embed.",
    ".decoder.",
    ".point_mlp.",
    ".point_valid_mlp.",
    ".point_valid_refine_mlp.",
    ".exist_mlp.",
    ".count_mlp.",
    ".aux_mask.",
    ".aux_edge.",
    ".p2_in.",
    ".fuse_p",
)


def parse_args() -> argparse.Namespace:
    """Parse checkpoint transfer check arguments."""
    parser = argparse.ArgumentParser(description="Check Q18 model transfer from a Q12 GCS checkpoint.")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="Q18 model YAML.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Q12 GCS checkpoint to transfer.")
    parser.add_argument(
        "--max-skipped-shape",
        type=int,
        default=8,
        help="Maximum allowed shape-mismatched tensors; expected value is small for Q12->Q18.",
    )
    parser.add_argument(
        "--min-gcs-loaded-ratio",
        type=float,
        default=0.75,
        help="Minimum loaded ratio among GCS-specific checkpoint tensors that exist in the target model.",
    )
    return parser.parse_args()


def is_gcs_key(key: str) -> bool:
    """Return True when a state_dict key belongs to GCS-specific modules."""
    return any(marker in key for marker in GCS_KEY_MARKERS)


def is_allowed_q_shape_skip(key: str, source_shape: torch.Size, target_shape: torch.Size) -> bool:
    """Return True for expected Q-dimension mismatches during Q12->Q18 transfer."""
    if key.endswith("query_embed.weight") and len(source_shape) == len(target_shape) == 2:
        return source_shape[0] != target_shape[0] and source_shape[1:] == target_shape[1:]
    if "point_reference_logits" in key:
        return True
    return False


def is_allowed_random_init(key: str) -> bool:
    """Return True for target tensors that are intentionally new in the Q18 countguard model."""
    return ".count_mlp." in key


def shape_skip_record(key: str, source_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> dict:
    """Build a JSON-serializable shape mismatch record."""
    return {"key": key, "source_shape": list(source_shape), "target_shape": list(target_shape)}


def main() -> None:
    """Load a Q12 checkpoint into the Q18 target and assert transfer health."""
    args = parse_args()
    cfg = args.cfg if args.cfg.is_absolute() else ROOT / args.cfg
    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    if not cfg.exists():
        raise FileNotFoundError(f"Missing Q18 model YAML: {cfg}")
    if not weights.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights}")

    model = GCSLaneModel(str(cfg), nc=1, ch=3, verbose=False)
    target_state = model.state_dict()
    source_state = GCSLaneTrainer._state_dict_from_weights(weights)
    source_is_gcs = GCSLaneTrainer._state_dict_has_gcs_modules(source_state)
    if not source_is_gcs:
        raise RuntimeError(f"Expected a GCS checkpoint, got a non-GCS checkpoint: {weights}")

    loadable: dict[str, torch.Tensor] = {}
    source_missing_in_target: list[str] = []
    skipped_shape: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    gcs_candidates = 0
    gcs_loaded = 0

    for key, value in source_state.items():
        if key not in target_state:
            source_missing_in_target.append(key)
            continue
        if is_gcs_key(key):
            gcs_candidates += 1
        if value.shape != target_state[key].shape:
            skipped_shape.append((key, tuple(value.shape), tuple(target_state[key].shape)))
            continue
        loadable[key] = value.to(dtype=target_state[key].dtype)
        if is_gcs_key(key):
            gcs_loaded += 1

    model.load_state_dict(loadable, strict=False)

    target_missing_from_source = sorted(key for key in target_state if key not in source_state)
    allowed_random_init = [key for key in target_missing_from_source if is_allowed_random_init(key)]
    unexpected_target_missing = [key for key in target_missing_from_source if not is_allowed_random_init(key)]
    unexpected_shape = [
        shape_skip_record(key, source_shape, target_shape)
        for key, source_shape, target_shape in skipped_shape
        if not is_allowed_q_shape_skip(key, torch.Size(source_shape), torch.Size(target_shape))
    ]
    gcs_loaded_ratio = float(gcs_loaded / max(gcs_candidates, 1))

    report = {
        "ok": True,
        "cfg": str(cfg),
        "weights": str(weights),
        "loaded_count": len(loadable),
        "target_count": len(target_state),
        "gcs_loaded_count": gcs_loaded,
        "gcs_candidate_count": gcs_candidates,
        "gcs_loaded_ratio": round(gcs_loaded_ratio, 6),
        "skipped_shape": [shape_skip_record(key, src, dst) for key, src, dst in skipped_shape],
        "source_missing_in_target": sorted(source_missing_in_target),
        "target_missing_from_source": target_missing_from_source,
        "allowed_random_init": allowed_random_init,
        "unexpected_shape": unexpected_shape,
        "unexpected_target_missing": unexpected_target_missing,
        "explanation": {
            "allowed_shape_mismatch": "Only query_embed.weight and point_reference_logits are allowed Q-shape skips.",
            "allowed_random_init": "count_mlp.* missing from the source checkpoint is expected and remains randomly initialized.",
        },
    }

    if len(loadable) == 0:
        report["ok"] = False
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError("No tensors transferred from checkpoint.")
    if len(skipped_shape) > int(args.max_skipped_shape):
        report["ok"] = False
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError(f"Too many shape-skipped tensors: {len(skipped_shape)} > {args.max_skipped_shape}")
    if unexpected_shape:
        report["ok"] = False
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError(f"Unexpected shape mismatches found: {unexpected_shape}")
    if unexpected_target_missing:
        report["ok"] = False
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError(f"Unexpected target tensors missing from source checkpoint: {unexpected_target_missing}")
    if gcs_loaded == 0 or gcs_loaded_ratio < float(args.min_gcs_loaded_ratio):
        report["ok"] = False
        print(json.dumps(report, indent=2), flush=True)
        raise RuntimeError(
            f"Insufficient GCS transfer: loaded {gcs_loaded}/{gcs_candidates} "
            f"({gcs_loaded_ratio:.3f}) < {args.min_gcs_loaded_ratio:.3f}"
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
