"""Contract checks for the default-off query short local x-refine head."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics.models.yolo.gcs_lane import val as gcs_val  # noqa: E402
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.nn.modules import GCSLaneHead  # noqa: E402
from ultralytics.nn.tasks import GCSLaneModel  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402


DEFAULT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml"
REFINE_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-short-local-refine.yaml"


def _head_from_yaml(cfg: Path) -> GCSLaneHead:
    model = GCSLaneModel(str(cfg), nc=1, verbose=False)
    head = model.model[-1]
    if not isinstance(head, GCSLaneHead):
        raise AssertionError(f"{cfg} did not build a GCSLaneHead.")
    return head.eval()


def _head_features(head: GCSLaneHead, batch: int = 2) -> list[torch.Tensor]:
    c = int(head.c1)
    return [
        torch.randn(batch, c, 32, 32),
        torch.randn(batch, c, 16, 16),
        torch.randn(batch, c, 8, 8),
        torch.randn(batch, c, 4, 4),
    ]


@torch.inference_mode()
def check_yaml_forward() -> None:
    default_head = _head_from_yaml(DEFAULT_CFG)
    default_out = default_head(_head_features(default_head), orig_size=(544, 960))
    if getattr(default_head, "query_short_local_refine_head", False):
        raise AssertionError("default query YAML unexpectedly enabled query_short_local_refine_head.")
    if (
        "pred_coarse_points" in default_out
        or "pred_short_refined_points" in default_out
        or "pred_short_refine_delta_logits" in default_out
        or "pred_short_refine_delta_norm" in default_out
    ):
        raise AssertionError("default query YAML unexpectedly emitted short local refine diagnostics.")

    refine_head = _head_from_yaml(REFINE_CFG)
    if not getattr(refine_head, "query_short_local_refine_head", False):
        raise AssertionError("short local refine YAML did not enable query_short_local_refine_head.")
    out = refine_head(_head_features(refine_head), orig_size=(544, 960))
    if tuple(out["pred_points"].shape) != (2, 12, 56, 2):
        raise AssertionError(f"pred_points shape mismatch: {tuple(out['pred_points'].shape)}.")
    if tuple(out.get("pred_coarse_points", torch.empty(0)).shape) != (2, 12, 56, 2):
        raise AssertionError(f"pred_coarse_points shape mismatch: {tuple(out.get('pred_coarse_points', torch.empty(0)).shape)}.")
    if tuple(out.get("pred_short_refined_points", torch.empty(0)).shape) != (2, 12, 56, 2):
        raise AssertionError(
            "pred_short_refined_points shape mismatch: "
            f"{tuple(out.get('pred_short_refined_points', torch.empty(0)).shape)}."
        )
    if tuple(out.get("pred_short_refine_delta_logits", torch.empty(0)).shape) != (2, 12, 56):
        raise AssertionError(
            "pred_short_refine_delta_logits shape mismatch: "
            f"{tuple(out.get('pred_short_refine_delta_logits', torch.empty(0)).shape)}."
        )
    if tuple(out.get("pred_short_refine_delta_norm", torch.empty(0)).shape) != (2, 12, 56):
        raise AssertionError(
            "pred_short_refine_delta_norm shape mismatch: "
            f"{tuple(out.get('pred_short_refine_delta_norm', torch.empty(0)).shape)}."
        )
    y_err = (out["pred_points"][..., 1] - out["pred_coarse_points"][..., 1]).abs().max()
    if float(y_err) > 1e-6:
        raise AssertionError(f"short local x-refine must not change fixed-y anchors, max y error={float(y_err):.6g}.")
    aux_y_err = (out["pred_short_refined_points"][..., 1] - out["pred_points"][..., 1]).abs().max()
    if float(aux_y_err) > 1e-6:
        raise AssertionError(f"auxiliary short local x-refine changed fixed-y anchors, max y error={float(aux_y_err):.6g}.")
    init_delta = (out["pred_short_refined_points"][..., 0] - out["pred_points"][..., 0]).abs().max()
    if float(init_delta) > 1e-8:
        raise AssertionError(f"zero-initialized short local x-refine must start as identity, max delta={float(init_delta):.6g}.")
    max_delta_norm = float(out["pred_short_refine_delta_norm"].abs().max())
    max_allowed_norm = float(getattr(refine_head, "short_local_refine_max_delta_px", 40.0)) / 960.0
    if max_delta_norm > max_allowed_norm + 1e-6:
        raise AssertionError(
            f"pred_short_refine_delta_norm exceeds max_delta_px contract: {max_delta_norm:.6g} > {max_allowed_norm:.6g}."
        )
    refine_head.query_short_local_refine_mlp[-1].bias.fill_(100.0)
    saturated = refine_head(_head_features(refine_head), orig_size=(544, 960))
    saturated_delta = float(saturated["pred_short_refine_delta_norm"].abs().max())
    if saturated_delta > max_allowed_norm + 1e-6:
        raise AssertionError(
            f"saturated pred_short_refine_delta_norm exceeds max_delta_px contract: "
            f"{saturated_delta:.6g} > {max_allowed_norm:.6g}."
        )


def _make_batch() -> dict:
    k = 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    gt2_x = torch.linspace(0.2, 0.8, 2).view(2, 1).expand(2, k)
    q_base = torch.linspace(0.05, 0.95, 12)
    gt5_x = (q_base[torch.tensor([1, 3, 5, 7, 9])] + 0.03).view(5, 1).expand(5, k)
    lanes = [
        torch.stack((gt2_x, y.view(1, k).expand(2, k)), dim=-1).float(),
        torch.stack((gt5_x, y.view(1, k).expand(5, k)), dim=-1).float(),
    ]
    valids = [
        torch.ones(2, k, dtype=torch.float32),
        torch.zeros(5, k, dtype=torch.float32),
    ]
    valids[1][:, 3:10] = 1.0
    return {
        "lanes": lanes,
        "lane_valid": valids,
        "num_lanes": torch.tensor([2, 5], dtype=torch.long),
    }


def _make_preds(include_coarse: bool) -> dict[str, torch.Tensor]:
    b, q, k = 2, 12, 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k).view(1, 1, k).expand(b, q, k)
    x = torch.linspace(0.05, 0.95, q).view(1, q, 1).expand(b, q, k)
    main = torch.stack(((x + 0.05).clamp(0.0, 1.0), y), dim=-1).contiguous()
    short_refined = torch.stack(((x + 0.02).clamp(0.0, 1.0), y), dim=-1).contiguous()
    preds = {
        "pred_points": main,
        "pred_logits": torch.zeros(b, q),
        "pred_valid_logits": torch.full((b, q, k), 4.0),
    }
    if include_coarse:
        preds["pred_coarse_points"] = main
        preds["pred_short_refined_points"] = short_refined
        preds["pred_short_refine_delta_logits"] = torch.zeros(b, q, k)
        preds["pred_short_refine_delta_norm"] = torch.zeros(b, q, k)
    return preds


def _make_coarse_match_batch() -> dict:
    k = 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k)
    x = torch.full((1, k), 0.25)
    pts = torch.stack((x, y.view(1, k)), dim=-1).float()
    valid = torch.zeros(1, k, dtype=torch.float32)
    valid[:, 3:10] = 1.0
    return {
        "lanes": [pts],
        "lane_valid": [valid],
        "num_lanes": torch.tensor([5], dtype=torch.long),
    }


def _make_coarse_match_preds() -> dict[str, torch.Tensor]:
    b, q, k = 1, 12, 56
    y = torch.linspace(710.0 / 720.0, 160.0 / 720.0, k).view(1, 1, k).expand(b, q, k)
    coarse_x = torch.full((b, q, k), 0.90)
    main_x = torch.full((b, q, k), 0.90)
    short_refined_x = torch.full((b, q, k), 0.90)
    coarse_x[:, 0, :] = 0.25
    main_x[:, 1, :] = 0.25
    short_refined_x[:, 0, :] = 0.90
    short_refined_x[:, 1, :] = 0.26
    return {
        "pred_points": torch.stack((main_x, y), dim=-1).contiguous(),
        "pred_coarse_points": torch.stack((coarse_x, y), dim=-1).contiguous(),
        "pred_short_refined_points": torch.stack((short_refined_x, y), dim=-1).contiguous(),
        "pred_logits": torch.zeros(b, q),
        "pred_valid_logits": torch.full((b, q, k), 4.0),
        "pred_short_refine_delta_logits": torch.zeros(b, q, k),
        "pred_short_refine_delta_norm": torch.zeros(b, q, k),
    }


def check_loss() -> None:
    names = list(GCSLoss.loss_names)
    refine_idx = names.index("short_local_refine_loss")
    count_idx = names.index("short_local_refine_count")
    coarse_idx = names.index("short_local_refine_coarse_ape")
    refined_idx = names.index("short_local_refine_refined_ape")

    criterion = GCSLoss(
        {
            "gcs_imgsz": [544, 960],
            "gcs_short_local_refine": 0.5,
            "gcs_short_local_refine_visible_thr": 10,
            "gcs_short_local_refine_gt_min_lanes": 4,
            "gcs_short_local_refine_beta_px": 5.0,
            "gcs_short_local_refine_max_delta_px": 40.0,
        }
    )
    _, items = criterion(_make_preds(include_coarse=True), _make_batch())
    if int(items.numel()) != len(GCSLoss.loss_names):
        raise AssertionError(f"GCSLoss item length mismatch: {items.numel()} vs {len(GCSLoss.loss_names)}.")
    if not torch.isfinite(items).all():
        raise AssertionError("GCSLoss produced non-finite items with short local refine diagnostics.")
    if float(items[refine_idx]) <= 0.0:
        raise AssertionError(f"short_local_refine_loss should be positive, got {float(items[refine_idx])}.")
    if float(items[count_idx]) <= 0.0:
        raise AssertionError(f"short_local_refine_count should be positive for GT5 short lanes, got {float(items[count_idx])}.")
    if float(items[coarse_idx]) <= float(items[refined_idx]):
        raise AssertionError("coarse APE should be larger than refined APE in the synthetic check.")
    if float(items[refine_idx]) >= 0.1:
        raise AssertionError(
            "short_local_refine_loss should be normalized, not pixel-scale; "
            f"got {float(items[refine_idx]):.6g}."
        )

    _, coarse_match_items = criterion(_make_coarse_match_preds(), _make_coarse_match_batch())
    if float(coarse_match_items[refine_idx]) >= 0.01:
        raise AssertionError(
            "short-local-refine matching must use main pred_points, not pred_coarse_points; "
            f"got refine loss {float(coarse_match_items[refine_idx]):.6g}."
        )

    try:
        criterion(_make_preds(include_coarse=False), _make_batch())
    except ValueError as exc:
        if "gcs_short_local_refine requires" not in str(exc):
            raise
    else:
        raise AssertionError("gcs_short_local_refine > 0 must fail when pred_short_refined_points is missing.")

    disabled = GCSLoss({"gcs_imgsz": [544, 960], "gcs_short_local_refine": 0.0})
    _, disabled_items = disabled(_make_preds(include_coarse=False), _make_batch())
    if float(disabled_items[refine_idx]) != 0.0:
        raise AssertionError("missing pred_short_refined_points with zero gain should log zero short_local_refine_loss.")


def check_loss_name_contract() -> None:
    if tuple(GCSLoss.loss_names) != tuple(GCSLaneTrainer.loss_names):
        raise AssertionError("GCSLoss.loss_names and GCSLaneTrainer.loss_names diverged.")
    if tuple(GCSLoss.loss_names) != tuple(gcs_val.LOSS_NAMES):
        raise AssertionError("GCSLoss.loss_names and GCSLaneValidator LOSS_NAMES diverged.")
    if len(gcs_val.LOSS_GAIN_ARGS) != len(gcs_val.DEFAULT_LOSS_GAINS):
        raise AssertionError(
            f"Validation loss gain length mismatch: {len(gcs_val.LOSS_GAIN_ARGS)} vs {len(gcs_val.DEFAULT_LOSS_GAINS)}."
        )


def main() -> None:
    check_yaml_forward()
    check_loss()
    check_loss_name_contract()
    print("GCS short local refine head checks passed.")


if __name__ == "__main__":
    main()
