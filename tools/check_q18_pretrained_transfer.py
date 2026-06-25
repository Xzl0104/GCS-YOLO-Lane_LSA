"""Check Q12-to-Q18 pretrained transfer for the side-dense K56 model."""

from __future__ import annotations

import argparse
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
    if len(source_shape) == len(target_shape) and any(s == 12 and t == 18 for s, t in zip(source_shape, target_shape)):
        return True
    return False


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
    skipped_missing: list[str] = []
    skipped_shape: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    gcs_candidates = 0
    gcs_loaded = 0

    for key, value in source_state.items():
        if key not in target_state:
            skipped_missing.append(key)
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

    disallowed_shape = [
        (key, source_shape, target_shape)
        for key, source_shape, target_shape in skipped_shape
        if not is_allowed_q_shape_skip(key, torch.Size(source_shape), torch.Size(target_shape))
    ]
    gcs_loaded_ratio = float(gcs_loaded / max(gcs_candidates, 1))

    print("OK: Q18 pretrained transfer dry-run completed.")
    print(f"weights: {weights}")
    print(f"loaded: {len(loadable)}/{len(target_state)}")
    print(f"gcs_loaded: {gcs_loaded}/{gcs_candidates} ({gcs_loaded_ratio:.3f})")
    print(f"skipped_missing: {len(skipped_missing)}")
    print(f"skipped_shape: {len(skipped_shape)}")
    for key, source_shape, target_shape in skipped_shape:
        print(f"shape_skip: {key}: {source_shape} -> {target_shape}")

    if len(loadable) == 0:
        raise RuntimeError("No tensors transferred from checkpoint.")
    if len(skipped_shape) > int(args.max_skipped_shape):
        raise RuntimeError(f"Too many shape-skipped tensors: {len(skipped_shape)} > {args.max_skipped_shape}")
    if disallowed_shape:
        raise RuntimeError(f"Non-Q shape mismatches found: {disallowed_shape}")
    if gcs_loaded == 0 or gcs_loaded_ratio < float(args.min_gcs_loaded_ratio):
        raise RuntimeError(
            f"Insufficient GCS transfer: loaded {gcs_loaded}/{gcs_candidates} "
            f"({gcs_loaded_ratio:.3f}) < {args.min_gcs_loaded_ratio:.3f}"
        )


if __name__ == "__main__":
    main()
