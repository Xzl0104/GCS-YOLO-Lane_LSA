# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Validation loop for GCS-YOLO-Lane."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ultralytics.data import build_dataloader
from ultralytics.data.dataset_gcs import GCSLaneDataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules import GCSLaneHead
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import ROOT
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, assert_gcs_shape, normalize_imgsz
from ultralytics.utils.gcs_postprocess import (
    GCS_DEFAULT_MAX_DET,
    count_head_decode_meta,
    decode_gcs_predictions,
    empty_decode_count_state,
    summarize_decode_count_state,
    update_decode_count_state,
)
from ultralytics.utils.torch_utils import select_device


LOSS_NAMES = (
    "exist_loss",
    "point_loss",
    "point_valid_loss",
    "line_iou_loss",
    "count_cls_loss",
    "count_sum_loss",
    "quality_loss",
)
LOSS_GAIN_ARGS = (
    "gcs_exist",
    "gcs_point",
    "gcs_point_valid",
    "gcs_line_iou",
    "gcs_count_cls",
    "gcs_count_sum",
    "gcs_quality",
)
DEFAULT_LOSS_GAINS = (
    1.0,
    5.0,
    0.5,
    0.3,
    0.3,
    0.02,
    0.3,
)
METRIC_NAMES = (
    "precision",
    "recall",
    "f1",
    "ape_mean_px",
    "ape_median_px",
    "ape_tp_mean_px",
    "ape_matched_all_mean_px",
    "ape_fp_matched_mean_px",
    "ape_all_matched_mean_px",
    "fp_matched_ape_mean_px",
    "lane_count_mae",
    "fp_per_image",
    "fn_per_image",
    "decode/count_head_k",
    "decode/final_pred_lanes",
    "decode/count_shortfall_rate",
    "decode/k5_to_output4_rate",
    "decode/k4_to_output5_rate",
    "tp",
    "fp",
    "fn",
)


class _GCSLaneMetrics:
    """Minimal metric container compatible with BaseTrainer bookkeeping."""

    def __init__(self, results: dict[str, float] | None = None):
        """Store the latest GCS validation losses for YOLO(...).val() callers."""
        self._results = dict(results or {})

    @property
    def keys(self) -> list[str]:
        """Return extra metric keys used by BaseTrainer header setup."""
        return [f"val/{x}" for x in METRIC_NAMES]

    @property
    def results_dict(self) -> dict[str, float]:
        """Return validation losses and fitness in the same style as Ultralytics metrics."""
        return dict(self._results)

    @property
    def fitness(self) -> float:
        """Return the structure-metric fitness value."""
        return float(self._results.get("fitness", 0.0))

    def update(self, results: dict[str, float]) -> None:
        """Replace stored validation results."""
        self._results = dict(results)

    def __getitem__(self, key: str) -> float:
        """Allow dictionary-style access to validation results."""
        return self._results[key]

    def get(self, key: str, default=None):
        """Allow dictionary-style optional access to validation results."""
        return self._results.get(key, default)

    def items(self):
        """Return stored result items."""
        return self._results.items()

    def __repr__(self) -> str:
        """Show useful values when YOLO(...).val() is printed interactively."""
        return f"{self.__class__.__name__}({self._results})"


class GCSLaneValidator:
    """Validation helper that reports GCS losses and structure-aware lane metrics."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize the validator with the same signature as other YOLO validators."""
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.args = args
        self.callbacks = _callbacks
        self.metrics = _GCSLaneMetrics()

    @staticmethod
    def _arg(args, name: str, default=None):
        """Read an argument from a namespace-like object or dict."""
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    @staticmethod
    def _path_value(value):
        """Normalize optional path arguments from CLI/config values."""
        if value is None:
            return None
        text = str(value).strip()
        return None if text == "" or text.lower() in {"none", "false"} else text

    def _build_dataloader(self):
        """Build a validation dataloader for standalone YOLO(...).val() calls."""
        if self.args is None:
            raise ValueError("GCSLaneValidator requires args or an explicit dataloader.")

        data_path = self._path_value(self._arg(self.args, "data", None)) or str(ROOT.parent / "data/tusimple_gcs_fixed_y_960x544.yaml")
        data = check_det_dataset(data_path)
        image_dir = self._path_value(self._arg(self.args, "val_images", None)) or data.get("val") or data.get("test")
        label_dir = self._path_value(self._arg(self.args, "val_gcs_labels", None))
        if not image_dir:
            raise ValueError("GCS validation requires a val image directory via data yaml or val_images=...")

        gcs_imgsz = self._arg(self.args, "gcs_imgsz", None)
        if gcs_imgsz is None:
            arg_imgsz = self._arg(self.args, "imgsz", None)
            data_shape = data.get("gcs_imgsz") or data.get("image_shape")
            if data_shape is not None:
                gcs_imgsz = data_shape
            elif isinstance(arg_imgsz, (list, tuple)) and len(arg_imgsz) > 1:
                gcs_imgsz = arg_imgsz
            elif isinstance(arg_imgsz, str) and any(x in arg_imgsz.lower() for x in (",", "x", "[", "(")):
                gcs_imgsz = arg_imgsz
            else:
                raise AssertionError(
                    "GCS validation requires args.gcs_imgsz or data image_shape as H,W. "
                    f"Scalar args.imgsz={arg_imgsz!r} would imply a square image."
                )
        imgsz = normalize_imgsz(gcs_imgsz)
        assert imgsz[0] != imgsz[1], f"GCS validation resolved square H,W={imgsz}; expected rectangular GCS input."
        batch = max(int(self._arg(self.args, "batch", 1) or 1), 1)
        workers = max(int(self._arg(self.args, "workers", 0) or 0), 0)
        dataset = GCSLaneDataset(img_path=image_dir, imgsz=imgsz, label_dir=label_dir, strict=True)
        assert_gcs_shape(dataset.imgsz, imgsz, name="validation dataset.imgsz", context="GCSLaneValidator._build_dataloader")
        return build_dataloader(dataset, batch=batch, workers=workers, shuffle=False, rank=-1, drop_last=False)

    @staticmethod
    def _model_fixed_y_anchors(model) -> np.ndarray | None:
        """Return fixed-y anchors from the model head, or None for free-point heads."""
        model = getattr(model, "module", model)
        for module in model.modules():
            if isinstance(module, GCSLaneHead):
                mode = str(getattr(module, "point_mode", "free")).lower()
                if mode in {"fixed-y", "fixedy"}:
                    mode = "fixed_y"
                if mode != "fixed_y":
                    return None
                if all(hasattr(module, name) for name in ("fixed_y_start", "fixed_y_end", "num_points")):
                    return np.linspace(
                        float(module.fixed_y_start),
                        float(module.fixed_y_end),
                        int(module.num_points),
                        dtype=np.float32,
                    ).reshape(-1)
                return module.fixed_y_anchors.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
        return None

    @classmethod
    def _check_fixed_y_contract(cls, model, dataset) -> None:
        """Fail fast when validation labels use fixed-y anchors different from the model head."""
        expected = cls._model_fixed_y_anchors(model)
        if expected is None:
            return
        anchors = getattr(dataset, "fixed_y_anchors", None)
        if anchors is None:
            raise ValueError("GCS validation model uses fixed_y anchors but dataset labels are not fixed_y.")
        anchors = np.asarray(anchors, dtype=np.float32).reshape(-1)
        if anchors.shape != expected.shape:
            raise ValueError(f"GCS validation fixed_y anchor shape mismatch: labels={anchors.shape}, model={expected.shape}.")
        max_err = float(np.max(np.abs(anchors - expected))) if anchors.size else 0.0
        if max_err > 5e-5:
            raise ValueError(
                "GCS validation fixed_y anchor mismatch: "
                f"labels first/last=({anchors[0]:.9f}, {anchors[-1]:.9f}) but model first/last="
                f"({expected[0]:.9f}, {expected[-1]:.9f}), max_err={max_err:.6g}. "
                "Regenerate labels with the same fixed-y range used by the model."
            )

    @staticmethod
    def _preprocess_batch(batch: dict, device: torch.device, image_size: tuple[int, int]) -> dict:
        """Move a GCS validation batch to device and normalize images without training-time resizing."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=device.type == "cuda")
        batch["lanes"] = [x.to(device, non_blocking=device.type == "cuda") for x in batch["lanes"]]
        batch["lane_valid"] = [x.to(device, non_blocking=device.type == "cuda") for x in batch["lane_valid"]]
        batch["img"] = batch["img"].float() / 255
        assert_gcs_image_tensor(batch["img"], image_size, name="batch['img']", context="GCSLaneValidator._preprocess_batch")
        return batch

    @staticmethod
    def _label_loss_items(loss_items: torch.Tensor, prefix: str = "val") -> dict[str, float]:
        """Return named GCS loss items for validation logging."""
        return dict(zip((f"{prefix}/{x}" for x in LOSS_NAMES), (round(float(x), 5) for x in loss_items)))

    def _loss_gains(
        self,
        device: torch.device,
        count_head_warmup_factor: float = 1.0,
    ) -> torch.Tensor:
        """Return validation loss gains matching the training objective."""
        gains = [float(self._loss_gain_arg(name, default)) for name, default in zip(LOSS_GAIN_ARGS, DEFAULT_LOSS_GAINS)]
        gains[LOSS_NAMES.index("count_cls_loss")] *= float(count_head_warmup_factor)
        return torch.tensor(gains, device=device, dtype=torch.float32)

    def _loss_gain_arg(self, name, default: float):
        """Read a validation loss gain."""
        if name is None:
            return default
        if isinstance(name, tuple):
            primary, legacy = name
            primary_value = self._arg(self.args, primary, None)
            legacy_value = self._arg(self.args, legacy, None)
            if legacy_value is not None and (primary_value is None or primary_value == default):
                return legacy_value
            return default if primary_value is None else primary_value
        return self._arg(self.args, name, default)

    def _eval_conf(self) -> float:
        """Return the existence threshold used for validation metrics."""
        value = self._arg(self.args, "gcs_eval_conf", None)
        if value is None:
            value = self._arg(self.args, "conf", None)
        return 0.6 if value is None else float(value)

    def _eval_ape_thr(self) -> float:
        """Return the APE threshold in pixels used to count true positives."""
        return float(self._arg(self.args, "gcs_eval_ape_thr", 20.0))

    def _eval_match_gate_px(self) -> float:
        """Return strict validation APE gate; default equals the TP threshold."""
        value = self._arg(self.args, "gcs_eval_match_gate_px", None)
        return self._eval_ape_thr() if value is None else float(value)

    def _eval_max_x_dist(self) -> float:
        """Return optional strict validation mean x-distance gate in pixels."""
        return float(self._arg(self.args, "gcs_eval_max_x_dist", 0.0) or 0.0)

    def _eval_min_overlap(self) -> int:
        """Return minimum valid GT points required for validation matching."""
        return int(self._arg(self.args, "gcs_eval_min_overlap", 6) or 0)

    def _eval_min_points(self) -> int:
        """Return minimum decoded visible points required to keep a validation prediction."""
        return int(self._arg(self.args, "gcs_eval_min_points", 6) or 0)

    def _eval_min_gt_cover_ratio(self) -> float:
        """Return minimum fraction of a GT lane covered by the prediction/GT overlap."""
        return float(self._arg(self.args, "gcs_eval_min_gt_cover_ratio", 0.3) or 0.0)

    def _eval_min_pred_cover_ratio(self) -> float:
        """Return minimum fraction of a predicted lane covered by the prediction/GT overlap."""
        return float(self._arg(self.args, "gcs_eval_min_pred_cover_ratio", 0.3) or 0.0)

    def _eval_nms_dist_px(self) -> float:
        """Return optional lane NMS distance used during validation decoding."""
        return float(self._arg(self.args, "gcs_eval_nms_dist_px", 18.0) or 0.0)

    def _eval_point_valid_thr(self) -> float:
        """Return the per-point visibility threshold used during validation decoding."""
        value = self._arg(self.args, "gcs_eval_point_valid_thr", None)
        if value is None:
            value = self._arg(self.args, "point_valid_thr", None)
        return 0.5 if value is None else float(value)

    def _eval_max_det(self) -> int:
        """Return the maximum number of decoded lanes retained per image."""
        return int(self._arg(self.args, "gcs_eval_max_det", GCS_DEFAULT_MAX_DET))

    def _use_count_head_decode(self) -> bool:
        """Return whether validation decode should use explicit Count Head K when present."""
        return bool(self._arg(self.args, "gcs_use_count_head_decode", True))

    def _count_head_temperature(self) -> float:
        """Return Count Head softmax temperature for decode."""
        return float(self._arg(self.args, "gcs_count_head_temp", 1.0) or 1.0)

    def _decode_candidate_min_points(self) -> int:
        """Return relaxed candidate-pool visible-anchor floor."""
        return int(self._arg(self.args, "gcs_decode_candidate_min_points", 5) or 5)

    def _enable_rescue_candidate_pool(self) -> bool:
        """Return whether weak real-query rescue candidates may fill Count Head pool shortfalls."""
        return bool(self._arg(self.args, "gcs_enable_rescue_candidate_pool", True))

    def _decode_candidate_conf(self) -> float:
        """Return relaxed candidate-pool existence threshold for Count Head decode."""
        return float(self._arg(self.args, "gcs_decode_candidate_conf", 0.05) or 0.0)

    def _decode_candidate_point_valid_thr(self) -> float:
        """Return relaxed candidate-pool point-valid threshold for Count Head decode."""
        return float(self._arg(self.args, "gcs_decode_candidate_point_valid_thr", 0.20) or 0.0)

    def _decode_rescue_candidate_conf(self) -> float:
        """Return weaker rescue-pool existence threshold for Count Head decode."""
        return float(self._arg(self.args, "gcs_decode_rescue_candidate_conf", 0.005) or 0.0)

    def _decode_rescue_candidate_point_valid_thr(self) -> float:
        """Return weaker rescue-pool point-valid threshold for Count Head decode."""
        return float(self._arg(self.args, "gcs_decode_rescue_candidate_point_valid_thr", 0.08) or 0.0)

    def _decode_rescue_candidate_min_points(self) -> int:
        """Return weaker rescue-pool visible-anchor floor."""
        return int(self._arg(self.args, "gcs_decode_rescue_candidate_min_points", 4) or 4)

    def _decode_final_min_points(self) -> int:
        """Return final visible-anchor floor for selected ranks 1-4."""
        return int(self._arg(self.args, "gcs_decode_final_min_points", 6) or 6)

    def _decode_fifth_min_points(self) -> int:
        """Return final visible-anchor floor for selected rank 5."""
        return int(self._arg(self.args, "gcs_decode_fifth_min_points", 5) or 5)

    def _line_nms_rescue_dist_px(self) -> float:
        """Return duplicate distance used for pre-NMS rescue."""
        return float(self._arg(self.args, "gcs_line_nms_rescue_dist_px", 30.0) or 0.0)

    def _line_nms_min_overlap(self) -> int:
        """Return minimum shared visible anchors for lane-NMS duplicate suppression."""
        return max(int(self._arg(self.args, "gcs_line_nms_min_overlap", 6) or 6), 1)

    def _quality_rescue_5th(self) -> bool:
        """Return whether quality-gated fifth-lane rescue is enabled."""
        return bool(self._arg(self.args, "gcs_quality_rescue_5th", True))

    def _quality_rescue_count5_thr(self) -> float:
        """Return Count Head P(5) threshold for quality-gated fifth-lane rescue."""
        return float(self._arg(self.args, "gcs_quality_rescue_count5_thr", 0.70))

    def _quality_rescue_conf_thr(self) -> float:
        """Return lane existence threshold for quality-gated fifth-lane rescue."""
        return float(self._arg(self.args, "gcs_quality_rescue_conf_thr", 0.03))

    def _quality_rescue_mean_valid_thr(self) -> float:
        """Return mean point-valid threshold for quality-gated fifth-lane rescue."""
        return float(self._arg(self.args, "gcs_quality_rescue_mean_valid_thr", 0.45))

    def _quality_rescue_quality_thr(self) -> float:
        """Return lane quality threshold for quality-gated fifth-lane rescue."""
        return float(self._arg(self.args, "gcs_quality_rescue_quality_thr", 0.55))

    def _quality_rescue_min_points(self) -> int:
        """Return visible-anchor floor for quality-gated fifth-lane rescue."""
        return int(self._arg(self.args, "gcs_quality_rescue_min_points", 5) or 5)

    def _quality_rescue_dist_px(self) -> float:
        """Return duplicate distance threshold for quality-gated fifth-lane rescue."""
        return float(self._arg(self.args, "gcs_quality_rescue_dist_px", 24.0) or 0.0)

    def _last_lane_rescue(self) -> bool:
        """Return whether last-required-lane rescue is enabled for high-count policies."""
        return bool(self._arg(self.args, "gcs_last_lane_rescue", False))

    def _last_lane_rescue_min_policy_count(self) -> int:
        """Return minimum Count Head policy K required for last-lane rescue."""
        return max(int(self._arg(self.args, "gcs_last_lane_rescue_min_policy_count", 4) or 4), 1)

    def _last_lane_rescue_conf_thr(self) -> float | None:
        """Return optional existence threshold for last-lane rescue."""
        value = self._arg(self.args, "gcs_last_lane_rescue_conf_thr", None)
        return None if value is None else float(value)

    def _last_lane_rescue_point_valid_thr(self) -> float:
        """Return point-valid threshold used only for last-lane rescue candidates."""
        return float(self._arg(self.args, "gcs_last_lane_rescue_point_valid_thr", 0.08) or 0.0)

    def _last_lane_rescue_min_points(self) -> int:
        """Return visible-anchor floor for last-lane rescue candidates."""
        return int(self._arg(self.args, "gcs_last_lane_rescue_min_points", 4) or 4)

    def _last_lane_rescue_mean_valid_thr(self) -> float:
        """Return mean point-valid threshold for last-lane rescue."""
        return float(self._arg(self.args, "gcs_last_lane_rescue_mean_valid_thr", 0.40))

    def _last_lane_rescue_quality_thr(self) -> float:
        """Return Quality Head threshold for last-lane rescue."""
        return float(self._arg(self.args, "gcs_last_lane_rescue_quality_thr", 0.50))

    def _last_lane_rescue_dist_px(self) -> float:
        """Return duplicate distance threshold for last-lane rescue."""
        return float(self._arg(self.args, "gcs_last_lane_rescue_dist_px", 24.0) or 0.0)

    def _edge_last_lane_rescue(self) -> bool:
        """Return whether outside-edge last-lane rescue is enabled."""
        return bool(self._arg(self.args, "gcs_edge_last_lane_rescue", False))

    def _edge_rescue_conf_thr(self) -> float:
        return float(self._arg(self.args, "gcs_edge_rescue_conf_thr", 0.02))

    def _edge_rescue_point_valid_thr(self) -> float:
        return float(self._arg(self.args, "gcs_edge_rescue_point_valid_thr", 0.06))

    def _edge_rescue_min_points(self) -> int:
        return int(self._arg(self.args, "gcs_edge_rescue_min_points", 4) or 4)

    def _edge_rescue_mean_valid_thr(self) -> float:
        return float(self._arg(self.args, "gcs_edge_rescue_mean_valid_thr", 0.35))

    def _edge_rescue_quality_thr(self) -> float:
        return float(self._arg(self.args, "gcs_edge_rescue_quality_thr", 0.45))

    def _edge_rescue_outside_gap_px(self) -> float:
        return float(self._arg(self.args, "gcs_edge_rescue_outside_gap_px", 28.0) or 0.0)

    def _edge_rescue_dist_px(self) -> float:
        return float(self._arg(self.args, "gcs_edge_rescue_dist_px", 24.0) or 0.0)

    def _edge_rescue_min_policy_count(self) -> int:
        return max(int(self._arg(self.args, "gcs_edge_rescue_min_policy_count", 4) or 4), 1)

    def _edge_count4_to5_upgrade(self) -> bool:
        return bool(self._arg(self.args, "gcs_edge_count4_to5_upgrade", True))

    def _edge_count4_to5_prob_margin(self) -> float:
        return float(self._arg(self.args, "gcs_edge_count4_to5_prob_margin", 0.20))

    def _soft_count_decision(self) -> bool:
        """Return whether close Count Head probabilities should use candidate-quality count selection."""
        return bool(self._arg(self.args, "gcs_soft_count_decision", False))

    def _soft_count_prob_margin(self) -> float:
        return float(self._arg(self.args, "gcs_soft_count_prob_margin", 0.08))

    def _soft_count_quality_weight(self) -> float:
        return float(self._arg(self.args, "gcs_soft_count_quality_weight", 1.0))

    def _soft_count_prior_weight(self) -> float:
        return float(self._arg(self.args, "gcs_soft_count_prior_weight", 0.5))

    def _soft_count_duplicate_penalty(self) -> float:
        return float(self._arg(self.args, "gcs_soft_count_duplicate_penalty", 1.0))

    def _soft_count_invalid_penalty(self) -> float:
        return float(self._arg(self.args, "gcs_soft_count_invalid_penalty", 1.0))

    def _count_head_warmup_factor(self, trainer=None) -> float:
        """Return the validation count-head warmup factor for the current training epoch."""
        count_cls_gain = float(self._loss_gain_arg("gcs_count_cls", DEFAULT_LOSS_GAINS[LOSS_NAMES.index("count_cls_loss")]))
        if count_cls_gain <= 0.0:
            return 0.0
        warmup_epochs = float(self._arg(self.args, "gcs_count_head_warmup_epochs", 5.0) or 0.0)
        if warmup_epochs <= 0.0:
            return 1.0
        if trainer is not None and hasattr(trainer, "epoch"):
            epoch = float(trainer.epoch)
        else:
            epoch_arg = self._arg(self.args, "epoch", None)
            if epoch_arg is None:
                return 1.0
            epoch = float(epoch_arg)
        return float(min(max(epoch / max(warmup_epochs, 1e-6), 0.0), 1.0))

    @staticmethod
    def _empty_metric_state() -> dict:
        """Create mutable validation metric accumulators."""
        return {
            "images": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "apes_tp": [],
            "apes_matched_all": [],
            "apes_fp_matched": [],
            "lane_count_abs_error": 0.0,
            **empty_decode_count_state(),
        }

    @staticmethod
    def _valid_gt_lanes(lanes: torch.Tensor, valid: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Convert one GT lane tensor to clipped normalized numpy arrays."""
        lanes_np = lanes.detach().float().cpu().numpy().astype(np.float32)
        valid_np = (valid.detach().float().cpu().numpy() > 0.5).astype(np.float32)
        if lanes_np.ndim != 3 or lanes_np.shape[-1] != 2:
            raise ValueError(f"GT lanes must have shape N x K x 2, got {lanes_np.shape}.")
        if valid_np.shape != lanes_np.shape[:2]:
            raise ValueError(f"GT valid mask must match lanes, got {valid_np.shape} vs {lanes_np.shape[:2]}.")
        keep = valid_np.sum(axis=1) >= 2
        return np.clip(lanes_np[keep], 0.0, 1.0), valid_np[keep]

    @staticmethod
    def _lane_ape_px(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, scale: np.ndarray) -> float:
        """Return average point error in pixels for one predicted/GT lane pair."""
        mask = valid > 0.5
        if int(mask.sum()) < 2:
            return float("inf")
        return float(np.linalg.norm((pred[mask] - gt[mask]) * scale, axis=-1).mean())

    @staticmethod
    def _pair_geometry(pred: np.ndarray, gt_lanes: np.ndarray, gt_valid: np.ndarray, scale: np.ndarray, pred_valid=None):
        """Return pairwise APE, mean x-distance, and overlap counts."""
        n_pred, n_gt = int(pred.shape[0]), int(gt_lanes.shape[0])
        if n_pred == 0 or n_gt == 0:
            shape = (n_pred, n_gt)
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.int32)
        valid = (gt_valid > 0.5).astype(np.float32)
        if pred_valid is None:
            pred_valid = np.ones(pred.shape[:2], dtype=np.float32)
        pred_valid = (pred_valid > 0.5).astype(np.float32)
        if pred_valid.shape != pred.shape[:2]:
            raise ValueError(f"pred_valid shape {pred_valid.shape} must match pred point dims {pred.shape[:2]}.")
        overlap_mask = pred_valid[:, None, :] * valid[None]
        overlap_per_pair = overlap_mask.sum(axis=2).astype(np.int32)
        denom = np.maximum(overlap_per_pair.astype(np.float32), 1.0)
        diff_px = (pred[:, None] - gt_lanes[None]) * scale.reshape(1, 1, 1, 2)
        point_error = np.linalg.norm(diff_px, axis=-1)
        ape = (point_error * overlap_mask).sum(axis=2) / denom
        mean_x = (np.abs(diff_px[..., 0]) * overlap_mask).sum(axis=2) / denom
        ape = np.where(overlap_per_pair > 0, ape, np.inf)
        mean_x = np.where(overlap_per_pair > 0, mean_x, np.inf)
        overlap = overlap_per_pair.copy()
        return ape.astype(np.float32), mean_x.astype(np.float32), overlap

    @staticmethod
    def _gated_assignment(cost: np.ndarray, gate: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Run Hungarian assignment and drop pairs that fail gate/finite checks."""
        if cost.size == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        finite = np.isfinite(cost)
        if gate is not None:
            finite = finite & gate
        if not finite.any():
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        rows, cols = linear_sum_assignment(np.where(finite, cost, 1e9))
        keep = finite[rows, cols]
        return rows[keep].astype(np.int64), cols[keep].astype(np.int64)

    @classmethod
    def _match_lanes(
        cls,
        pred_lanes: list[dict],
        gt_lanes: np.ndarray,
        gt_valid: np.ndarray,
        image_shape: tuple[int, int],
        ape_thr: float,
        match_gate_px: float,
        max_x_dist: float,
        min_overlap: int,
        min_gt_cover_ratio: float,
        min_pred_cover_ratio: float,
    ) -> tuple[int, int, int, list[float], list[float], list[float]]:
        """Strictly match decoded predictions to GT lanes and return split APE diagnostics."""
        h, w = int(image_shape[0]), int(image_shape[1])
        scale = np.array([w, h], dtype=np.float32)
        pred = (
            np.stack([np.asarray(x["points_norm"], dtype=np.float32) for x in pred_lanes], axis=0)
            if pred_lanes
            else np.zeros((0, gt_lanes.shape[1] if gt_lanes.ndim == 3 else 0, 2), dtype=np.float32)
        )
        pred_valid = (
            np.stack(
                [
                    np.asarray(x.get("point_valid", np.ones(np.asarray(x["points_norm"]).shape[0])), dtype=np.float32)
                    for x in pred_lanes
                ],
                axis=0,
            )
            if pred_lanes
            else np.zeros((0, gt_lanes.shape[1] if gt_lanes.ndim == 3 else 0), dtype=np.float32)
        )
        n_pred = int(pred.shape[0])
        n_gt = int(gt_lanes.shape[0])
        if n_pred == 0 or n_gt == 0:
            return 0, n_pred, n_gt, [], [], []

        ape, mean_x, overlap = cls._pair_geometry(pred, gt_lanes, gt_valid, scale, pred_valid=pred_valid)
        diagnostic_rows, diagnostic_cols = cls._gated_assignment(ape)
        gate = overlap >= max(int(min_overlap), 0)
        gt_visible = np.maximum((gt_valid > 0.5).sum(axis=1).astype(np.float32), 1.0)
        pred_visible = np.maximum((pred_valid > 0.5).sum(axis=1).astype(np.float32), 1.0)
        if min_gt_cover_ratio > 0.0:
            gate = gate & ((overlap.astype(np.float32) / gt_visible.reshape(1, -1)) >= float(min_gt_cover_ratio))
        if min_pred_cover_ratio > 0.0:
            gate = gate & ((overlap.astype(np.float32) / pred_visible.reshape(-1, 1)) >= float(min_pred_cover_ratio))
        if max_x_dist > 0.0:
            gate = gate & (mean_x <= float(max_x_dist))
        if match_gate_px > 0.0:
            gate = gate & (ape <= float(match_gate_px))
        rows, cols = cls._gated_assignment(ape, gate=gate)

        strict_pairs = {(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())}
        apes_tp = [float(ape[r, c]) for r, c in zip(rows, cols) if float(ape[r, c]) < float(ape_thr)]
        apes_all = [float(ape[r, c]) for r, c in zip(diagnostic_rows, diagnostic_cols)]
        apes_fp = [
            float(ape[r, c])
            for r, c in zip(diagnostic_rows, diagnostic_cols)
            if (int(r), int(c)) not in strict_pairs or float(ape[r, c]) >= float(ape_thr)
        ]
        tp = len(apes_tp)
        return tp, n_pred - tp, n_gt - tp, apes_tp, apes_all, apes_fp

    def _update_metric_state(self, state: dict, preds: dict[str, torch.Tensor], batch: dict) -> None:
        """Decode validation predictions and accumulate structured lane metrics."""
        pred_points = preds["pred_points"].detach()
        pred_logits = preds["pred_logits"].detach()
        pred_valid_logits = preds.get("pred_valid_logits")
        if pred_valid_logits is not None:
            pred_valid_logits = pred_valid_logits.detach()
        pred_count_logits = preds.get("pred_count_logits")
        if pred_count_logits is not None:
            pred_count_logits = pred_count_logits.detach()
        pred_count_boundary_logits = preds.get("pred_count_boundary_logits")
        if pred_count_boundary_logits is not None:
            pred_count_boundary_logits = pred_count_boundary_logits.detach()
        pred_quality_logits = preds.get("pred_quality_logits")
        if pred_quality_logits is not None:
            pred_quality_logits = pred_quality_logits.detach()
        if pred_logits.ndim == 3 and pred_logits.shape[-1] == 1:
            pred_logits = pred_logits.squeeze(-1)
        h, w = int(batch["img"].shape[-2]), int(batch["img"].shape[-1])
        conf = self._eval_conf()
        ape_thr = self._eval_ape_thr()
        match_gate_px = self._eval_match_gate_px()
        max_x_dist = self._eval_max_x_dist()
        min_overlap = self._eval_min_overlap()
        min_points = self._eval_min_points()
        min_gt_cover_ratio = self._eval_min_gt_cover_ratio()
        min_pred_cover_ratio = self._eval_min_pred_cover_ratio()
        nms_dist_px = self._eval_nms_dist_px()
        point_valid_thr = self._eval_point_valid_thr()
        max_det = self._eval_max_det()

        for i, (gt_lanes_t, gt_valid_t) in enumerate(zip(batch["lanes"], batch["lane_valid"])):
            pred_lanes, decode_meta = decode_gcs_predictions(
                pred_points[i],
                pred_logits[i],
                pred_valid_logits=pred_valid_logits[i] if pred_valid_logits is not None else None,
                pred_count_logits=pred_count_logits[i] if pred_count_logits is not None else None,
                pred_count_boundary_logits=(
                    pred_count_boundary_logits[i] if pred_count_boundary_logits is not None else None
                ),
                pred_quality_logits=pred_quality_logits[i] if pred_quality_logits is not None else None,
                image_shape=(h, w),
                score_thr=conf,
                point_valid_thr=point_valid_thr,
                min_points=min_points,
                max_det=max_det,
                nms_dist_px=nms_dist_px,
                use_count_head_decode=self._use_count_head_decode(),
                count_head_temperature=self._count_head_temperature(),
                dataset_name="tusimple",
                candidate_score_thr=self._decode_candidate_conf(),
                candidate_point_valid_thr=self._decode_candidate_point_valid_thr(),
                candidate_min_points=self._decode_candidate_min_points(),
                enable_rescue_candidate_pool=self._enable_rescue_candidate_pool(),
                rescue_candidate_score_thr=self._decode_rescue_candidate_conf(),
                rescue_candidate_point_valid_thr=self._decode_rescue_candidate_point_valid_thr(),
                rescue_candidate_min_points=self._decode_rescue_candidate_min_points(),
                final_min_points=self._decode_final_min_points(),
                fifth_min_points=self._decode_fifth_min_points(),
                line_nms_min_overlap=self._line_nms_min_overlap(),
                line_nms_rescue_dist_px=self._line_nms_rescue_dist_px(),
                quality_rescue_5th=self._quality_rescue_5th(),
                quality_rescue_count5_thr=self._quality_rescue_count5_thr(),
                quality_rescue_conf_thr=self._quality_rescue_conf_thr(),
                quality_rescue_mean_valid_thr=self._quality_rescue_mean_valid_thr(),
                quality_rescue_quality_thr=self._quality_rescue_quality_thr(),
                quality_rescue_min_points=self._quality_rescue_min_points(),
                quality_rescue_dist_px=self._quality_rescue_dist_px(),
                last_lane_rescue=self._last_lane_rescue(),
                last_lane_rescue_min_policy_count=self._last_lane_rescue_min_policy_count(),
                last_lane_rescue_conf_thr=self._last_lane_rescue_conf_thr(),
                last_lane_rescue_point_valid_thr=self._last_lane_rescue_point_valid_thr(),
                last_lane_rescue_min_points=self._last_lane_rescue_min_points(),
                last_lane_rescue_mean_valid_thr=self._last_lane_rescue_mean_valid_thr(),
                last_lane_rescue_quality_thr=self._last_lane_rescue_quality_thr(),
                last_lane_rescue_dist_px=self._last_lane_rescue_dist_px(),
                edge_last_lane_rescue=self._edge_last_lane_rescue(),
                edge_rescue_conf_thr=self._edge_rescue_conf_thr(),
                edge_rescue_point_valid_thr=self._edge_rescue_point_valid_thr(),
                edge_rescue_min_points=self._edge_rescue_min_points(),
                edge_rescue_mean_valid_thr=self._edge_rescue_mean_valid_thr(),
                edge_rescue_quality_thr=self._edge_rescue_quality_thr(),
                edge_rescue_outside_gap_px=self._edge_rescue_outside_gap_px(),
                edge_rescue_dist_px=self._edge_rescue_dist_px(),
                edge_rescue_min_policy_count=self._edge_rescue_min_policy_count(),
                edge_count4_to5_upgrade=self._edge_count4_to5_upgrade(),
                edge_count4_to5_prob_margin=self._edge_count4_to5_prob_margin(),
                enable_soft_count_decision=self._soft_count_decision(),
                soft_count_prob_margin=self._soft_count_prob_margin(),
                soft_count_quality_weight=self._soft_count_quality_weight(),
                soft_count_prior_weight=self._soft_count_prior_weight(),
                soft_count_duplicate_penalty=self._soft_count_duplicate_penalty(),
                soft_count_invalid_penalty=self._soft_count_invalid_penalty(),
                return_meta=True,
            )
            gt_lanes, gt_valid = self._valid_gt_lanes(gt_lanes_t, gt_valid_t)
            tp, fp, fn, apes_tp, apes_all, apes_fp = self._match_lanes(
                pred_lanes,
                gt_lanes,
                gt_valid,
                (h, w),
                ape_thr,
                match_gate_px,
                max_x_dist,
                min_overlap,
                min_gt_cover_ratio,
                min_pred_cover_ratio,
            )
            state["images"] += 1
            state["tp"] += tp
            state["fp"] += fp
            state["fn"] += fn
            state["apes_tp"].extend(apes_tp)
            state["apes_matched_all"].extend(apes_all)
            state["apes_fp_matched"].extend(apes_fp)
            state["lane_count_abs_error"] += abs(len(pred_lanes) - int(gt_lanes.shape[0]))
            update_decode_count_state(state, decode_meta, len(pred_lanes))

    @staticmethod
    def _metric_results(state: dict) -> dict[str, float]:
        """Summarize structured lane validation metrics."""
        tp, fp, fn = int(state["tp"]), int(state["fp"]), int(state["fn"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        apes = state["apes_tp"]
        apes_all = state["apes_matched_all"]
        apes_fp = state["apes_fp_matched"]
        images = max(int(state["images"]), 1)
        results = {
            "val/precision": round(float(precision), 6),
            "val/recall": round(float(recall), 6),
            "val/f1": round(float(f1), 6),
            "val/ape_mean_px": round(float(np.mean(apes)), 4) if apes else 0.0,
            "val/ape_median_px": round(float(np.median(apes)), 4) if apes else 0.0,
            "val/ape_tp_mean_px": round(float(np.mean(apes)), 4) if apes else 0.0,
            "val/ape_matched_all_mean_px": round(float(np.mean(apes_all)), 4) if apes_all else 0.0,
            "val/ape_fp_matched_mean_px": round(float(np.mean(apes_fp)), 4) if apes_fp else 0.0,
            "val/ape_all_matched_mean_px": round(float(np.mean(apes_all)), 4) if apes_all else 0.0,
            "val/fp_matched_ape_mean_px": round(float(np.mean(apes_fp)), 4) if apes_fp else 0.0,
            "val/lane_count_mae": round(float(state["lane_count_abs_error"]) / images, 6),
            "val/fp_per_image": round(float(fp) / images, 6),
            "val/fn_per_image": round(float(fn) / images, 6),
            "val/tp": float(tp),
            "val/fp": float(fp),
            "val/fn": float(fn),
        }
        results.update(summarize_decode_count_state(state, prefix="val/decode/"))
        return results

    @staticmethod
    def _fitness_from_results(results: dict[str, float]) -> float:
        """Use F1 as the main checkpoint fitness with small APE/count tie-breakers."""
        f1 = float(results.get("val/f1", 0.0))
        ape = float(results.get("val/ape_mean_px", 0.0))
        lane_count_mae = float(results.get("val/lane_count_mae", 0.0))
        return f1 - 0.0001 * ape - 0.001 * lane_count_mae

    @torch.no_grad()
    def __call__(self, trainer=None, model=None):
        """Run validation and return loss components plus structure-metric fitness."""
        if self.dataloader is None:
            self.dataloader = self._build_dataloader()

        if trainer is not None:
            model = trainer.ema.ema if trainer.ema else trainer.model
            device = trainer.device
        elif model is not None:
            device = select_device(getattr(self.args, "device", None), verbose=False)
            if isinstance(model, (str, Path)):
                model, _ = load_checkpoint(model, device=device)
            else:
                model = model.to(device)
        else:
            raise ValueError("GCSLaneValidator requires a trainer or model to validate.")

        was_training = model.training
        image_size = tuple(getattr(getattr(self.dataloader, "dataset", None), "imgsz", None) or self._arg(self.args, "gcs_imgsz", None))
        image_size = normalize_imgsz(image_size)
        model.gcs_imgsz = image_size
        if getattr(model, "args", None) is not None:
            if isinstance(model.args, dict):
                model.args["gcs_imgsz"] = [int(image_size[0]), int(image_size[1])]
            else:
                model.args.gcs_imgsz = [int(image_size[0]), int(image_size[1])]
        self._check_fixed_y_contract(model, self.dataloader.dataset)
        model.eval()

        loss_sum = torch.zeros(len(LOSS_NAMES), device=device)
        metric_state = self._empty_metric_state()
        batches = 0
        try:
            for batch in self.dataloader:
                batch = self._preprocess_batch(batch, device, image_size)
                preds = model(batch["img"])
                _, items = model.loss(batch, preds)
                loss_sum += items.detach()
                self._update_metric_state(metric_state, preds, batch)
                batches += 1
        finally:
            if was_training:
                model.train()

        mean_loss = loss_sum / max(batches, 1)
        count_head_warmup_factor = self._count_head_warmup_factor(trainer)
        static_loss = (mean_loss * self._loss_gains(device, count_head_warmup_factor=1.0)).sum()
        weighted_loss = (
            mean_loss
            * self._loss_gains(
                device,
                count_head_warmup_factor=count_head_warmup_factor,
            )
        ).sum()
        results = self._label_loss_items(mean_loss.cpu(), prefix="val")
        results.update(self._metric_results(metric_state))
        count_cls_idx = LOSS_NAMES.index("count_cls_loss")
        count_cls_gain = float(self._loss_gain_arg("gcs_count_cls", DEFAULT_LOSS_GAINS[count_cls_idx]))
        results["val/count_cls_loss_weighted"] = round(
            float((mean_loss[count_cls_idx] * count_cls_gain * count_head_warmup_factor).cpu()), 5
        )
        results["val/count_head_warmup_factor"] = round(float(count_head_warmup_factor), 6)
        results["val/total_loss_static"] = round(float(static_loss.cpu()), 5)
        results["val/total_loss_weighted"] = round(float(weighted_loss.cpu()), 5)
        results["val/total_loss"] = round(float(weighted_loss.cpu()), 5)
        results["fitness"] = self._fitness_from_results(results)
        self.metrics.update(results)
        return results
