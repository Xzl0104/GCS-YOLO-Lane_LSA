# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Prediction post-processing for GCS-YOLO-Lane."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.models.gcs.decode_ordered_slot import decode_ordered_slot_predictions
from ultralytics.models.gcs.decode_summary import ordered_slot_decode_params, ordered_slot_decode_runtime_config
from ultralytics.nn.modules import GCSLaneHead
from ultralytics.utils import ops
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, assert_gcs_shape, normalize_imgsz
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions, draw_gcs_lanes, save_gcs_lanes_txt


class GCSLaneResults(Results):
    """Results object for structured GCS lane point sequences."""

    def __init__(self, orig_img, path, names, lanes: list[dict]):
        """Create a result with decoded GCS lanes and tensor convenience views."""
        super().__init__(orig_img, path=path, names=names)
        self.gcs_lanes = lanes
        if lanes:
            self.lanes = torch.from_numpy(np.stack([x["points"] for x in lanes], axis=0).astype("float32"))
            self.lanes_normalized = torch.from_numpy(
                np.stack([x["points_norm"] for x in lanes], axis=0).astype("float32")
            )
            self.lane_scores = torch.tensor([x["score"] for x in lanes], dtype=torch.float32)
            self.lane_queries = torch.tensor([x["query"] for x in lanes], dtype=torch.long)
        else:
            self.lanes = torch.zeros((0, 0, 2), dtype=torch.float32)
            self.lanes_normalized = torch.zeros((0, 0, 2), dtype=torch.float32)
            self.lane_scores = torch.zeros((0,), dtype=torch.float32)
            self.lane_queries = torch.zeros((0,), dtype=torch.long)
        self._keys = (*self._keys, "lanes")

    def verbose(self) -> str:
        """Return a concise lane-count summary."""
        n_lanes = len(self.lanes)
        return f"{n_lanes} lane{'s' * (n_lanes != 1)}, "

    def save_txt(self, txt_file: str | Path, save_conf: bool = False) -> str:
        """Save normalized structured lane point sequences."""
        return save_gcs_lanes_txt(txt_file, self.gcs_lanes, save_conf=save_conf)

    def plot(self, conf: bool = True, line_width: float | None = None, img: np.ndarray | None = None, **kwargs):
        """Plot GCS lanes instead of YOLO boxes or masks."""
        canvas = self.orig_img if img is None else img
        line_width = int(line_width or 2)
        plotted = draw_gcs_lanes(canvas, self.gcs_lanes, show_scores=conf, line_width=line_width)
        if kwargs.get("save") and kwargs.get("filename"):
            cv2.imwrite(str(kwargs["filename"]), plotted)
        return plotted


class GCSLanePredictor(BasePredictor):
    """Decode GCS lane point sequences from pred_points and pred_logits."""

    @staticmethod
    def _arg_value(args, name: str):
        """Read one predictor argument from a namespace-like object or dict."""
        if isinstance(args, dict):
            return args.get(name)
        return getattr(args, name, None)

    def _model_yaml_value(self, name: str):
        """Read a GCS shape field from the underlying model YAML when available."""
        model = getattr(self, "model", None)
        for obj in (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "model", None)):
            yaml = getattr(obj, "yaml", None)
            if isinstance(yaml, dict) and yaml.get(name) is not None:
                return yaml[name]
        return None

    @staticmethod
    def _normalize_gcs_mode(mode) -> str:
        """Normalize GCS mode spelling."""
        mode = str(mode or "query").lower()
        if mode in {"ordered-slot", "orderedslot"}:
            mode = "ordered_slot"
        if mode not in {"query", "ordered_slot"}:
            raise ValueError(f"gcs_mode must be 'query' or 'ordered_slot', got {mode!r}.")
        return mode

    def _model_gcs_mode(self) -> str:
        """Infer the loaded GCS head mode without relying on output-key collisions."""
        model = getattr(self, "model", None)
        for obj in (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "model", None)):
            if obj is None or not hasattr(obj, "modules"):
                continue
            modes = [
                self._normalize_gcs_mode(getattr(module, "gcs_mode", "query"))
                for module in obj.modules()
                if isinstance(module, GCSLaneHead)
            ]
            unique = sorted(set(modes))
            if len(unique) == 1:
                return unique[0]
            if len(unique) > 1:
                raise RuntimeError(f"Multiple GCS modes found in model: {unique}.")
        return self._normalize_gcs_mode(self._arg_value(self.args, "gcs_mode") or getattr(model, "gcs_mode", "query"))

    @staticmethod
    def _normalize_explicit_gcs_imgsz(value, source: str) -> tuple[int, int]:
        """Normalize an explicit GCS H,W shape and reject square-only fallbacks."""
        shape = normalize_imgsz(value)
        if int(shape[0]) == int(shape[1]):
            raise AssertionError(
                f"GCSLanePredictor resolved square {source} H,W={shape}. "
                "Use explicit rectangular GCS size [544, 960] for TuSimple or [384, 960] for CULane."
            )
        return shape

    @staticmethod
    def _normalize_optional_rect(value):
        """Return a non-square H,W shape, or None when the value is absent/square."""
        if value is None or value == "":
            return None
        shape = normalize_imgsz(value)
        return shape if int(shape[0]) != int(shape[1]) else None

    def _resolve_gcs_imgsz(self) -> tuple[int, int]:
        """Resolve the H,W inference shape without silently falling back to 640x640."""
        for source, value in (
            ("args.gcs_imgsz", self._arg_value(self.args, "gcs_imgsz")),
            ("args.image_shape", self._arg_value(self.args, "image_shape")),
        ):
            if value is not None and value != "":
                return self._normalize_explicit_gcs_imgsz(value, source)

        for value in (getattr(self, "imgsz", None), self._arg_value(self.args, "imgsz")):
            shape = self._normalize_optional_rect(value) if isinstance(value, (list, tuple, str)) else None
            if shape is not None:
                return shape

        for source, value in (
            ("model.yaml.gcs_imgsz", self._model_yaml_value("gcs_imgsz")),
            ("model.yaml.image_shape", self._model_yaml_value("image_shape")),
            ("model.yaml.imgsz", self._model_yaml_value("imgsz")),
        ):
            if value is not None and value != "":
                return self._normalize_explicit_gcs_imgsz(value, source)
        return normalize_imgsz((544, 960))

    def pre_transform(self, im: list) -> list:
        """Resize images to the same training coordinate system used by GCS labels."""
        h, w = self._resolve_gcs_imgsz()
        if getattr(self, "model", None) is not None and hasattr(self.model, "model"):
            setattr(self.model.model, "gcs_imgsz", (h, w))
        resized = [cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in im]
        for i, item in enumerate(resized):
            assert_gcs_shape(item.shape[:2], (h, w), name=f"pre_transform image[{i}]", context="GCSLanePredictor")
        return resized

    def postprocess(self, preds: dict[str, torch.Tensor], img: torch.Tensor, orig_imgs, **kwargs):
        """Attach decoded lane point sequences to standard Results objects."""
        assert_gcs_image_tensor(img, self._resolve_gcs_imgsz(), name="predictor input tensor", context="GCSLanePredictor.postprocess")
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if not isinstance(preds, dict) or "pred_points" not in preds or "pred_logits" not in preds:
            raise ValueError("GCSLanePredictor expects model outputs with 'pred_points' and 'pred_logits'.")

        points = preds["pred_points"].detach()
        logits = preds["pred_logits"].detach()
        quality_logits = preds.get("pred_quality_logits")
        if quality_logits is not None:
            quality_logits = quality_logits.detach()
        valid_logits = preds.get("pred_valid_logits")
        if valid_logits is not None:
            valid_logits = valid_logits.detach()
        start_logits = preds.get("pred_start_logits")
        if start_logits is not None:
            start_logits = start_logits.detach()
        end_logits = preds.get("pred_end_logits")
        if end_logits is not None:
            end_logits = end_logits.detach()
        short_candidate_points = preds.get("pred_short_candidate_points")
        if short_candidate_points is not None:
            short_candidate_points = short_candidate_points.detach()
        short_candidate_logits = preds.get("pred_short_candidate_logits")
        if short_candidate_logits is not None:
            short_candidate_logits = short_candidate_logits.detach()
        conf = 0.25 if self.args.conf is None else float(self.args.conf)
        max_det = int(self.args.max_det) if getattr(self.args, "max_det", None) else None
        nms_dist_px = float(getattr(self.args, "gcs_eval_nms_dist_px", 0.0) or 0.0)
        point_valid_thr = getattr(self.args, "gcs_eval_point_valid_thr", None)
        if point_valid_thr is None:
            point_valid_thr = getattr(self.args, "point_valid_thr", 0.5)
        point_valid_thr = float(point_valid_thr)
        extent_decode = bool(self._arg_value(self.args, "gcs_extent_decode") or False)
        extent_decode_mode = str(self._arg_value(self.args, "gcs_extent_decode_mode") or "interval")
        candidate_decode = bool(self._arg_value(self.args, "gcs_candidate_decode") or False)
        candidate_short_gate = bool(self._arg_value(self.args, "gcs_candidate_short_gate") if hasattr(self.args, "gcs_candidate_short_gate") else True)
        candidate_gate_valid_thr = float(self._arg_value(self.args, "gcs_candidate_gate_valid_thr") or 0.5)
        candidate_gate_min_visible = int(self._arg_value(self.args, "gcs_candidate_gate_min_visible") or 2)
        candidate_gate_max_visible = int(self._arg_value(self.args, "gcs_candidate_gate_max_visible") or 10)
        candidate_preserve_base_score = bool(
            self._arg_value(self.args, "gcs_candidate_preserve_base_score")
            if hasattr(self.args, "gcs_candidate_preserve_base_score")
            else True
        )

        results = []
        if valid_logits is None:
            valid_iter = [None] * int(points.shape[0])
        else:
            valid_iter = list(valid_logits)
        if quality_logits is None:
            quality_iter = [None] * int(points.shape[0])
        else:
            quality_iter = list(quality_logits)
        if start_logits is None:
            start_iter = [None] * int(points.shape[0])
        else:
            start_iter = list(start_logits)
        if end_logits is None:
            end_iter = [None] * int(points.shape[0])
        else:
            end_iter = list(end_logits)
        if short_candidate_points is None:
            candidate_points_iter = [None] * int(points.shape[0])
        else:
            candidate_points_iter = list(short_candidate_points)
        if short_candidate_logits is None:
            candidate_logits_iter = [None] * int(points.shape[0])
        else:
            candidate_logits_iter = list(short_candidate_logits)

        ordered_slot = self._model_gcs_mode() == "ordered_slot"
        ordered_slot_runtime_cfg = ordered_slot_decode_runtime_config(context="predict") if ordered_slot else None
        ordered_slot_params = ordered_slot_decode_params(self.args) if ordered_slot else None
        for batch_i, (
            lane_points,
            lane_logits,
            lane_quality_logits,
            lane_valid_logits,
            lane_start_logits,
            lane_end_logits,
            lane_candidate_points,
            lane_candidate_logits,
            orig_img,
            img_path,
        ) in enumerate(
            zip(
                points,
                logits,
                quality_iter,
                valid_iter,
                start_iter,
                end_iter,
                candidate_points_iter,
                candidate_logits_iter,
                orig_imgs,
                self.batch[0],
            )
        ):
            if ordered_slot:
                lanes = decode_ordered_slot_predictions(
                    preds,
                    batch_index=batch_i,
                    image_shape=orig_img.shape[:2],
                    min_lanes=ordered_slot_params["min_lanes"],
                    max_lanes=ordered_slot_params["max_lanes"],
                    min_interval_points=ordered_slot_params["min_interval_points"],
                    order_margin_px=ordered_slot_params["order_margin_px"],
                    img_w=float(orig_img.shape[1]),
                    order_check=ordered_slot_runtime_cfg["order_check"],
                    output_order=ordered_slot_runtime_cfg["output_order"],
                )
            else:
                lanes = decode_gcs_predictions(
                    lane_points,
                    lane_logits,
                    pred_quality_logits=lane_quality_logits,
                    pred_valid_logits=lane_valid_logits,
                    pred_start_logits=lane_start_logits,
                    pred_end_logits=lane_end_logits,
                    pred_short_candidate_points=lane_candidate_points,
                    pred_short_candidate_logits=lane_candidate_logits,
                    image_shape=orig_img.shape[:2],
                    score_thr=conf,
                    point_valid_thr=point_valid_thr,
                    max_det=max_det,
                    nms_dist_px=nms_dist_px,
                    extent_decode=extent_decode,
                    extent_decode_mode=extent_decode_mode,
                    candidate_decode=candidate_decode,
                    candidate_short_gate=candidate_short_gate,
                    candidate_gate_valid_thr=candidate_gate_valid_thr,
                    candidate_gate_min_visible=candidate_gate_min_visible,
                    candidate_gate_max_visible=candidate_gate_max_visible,
                    candidate_preserve_base_score=candidate_preserve_base_score,
                )
            result = GCSLaneResults(orig_img, path=img_path, names=self.model.names, lanes=lanes)
            if not lanes:
                k = int(lane_points.shape[1])
                result.lanes = torch.zeros((0, k, 2), dtype=torch.float32)
                result.lanes_normalized = torch.zeros((0, k, 2), dtype=torch.float32)
            results.append(result)
        return results

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        """Write GCS lane results without using YOLO box/mask plotting or NMS."""
        string = ""
        if len(im.shape) == 3:
            im = im[None]
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])

        result = self.results[i]
        result.save_dir = str(self.save_dir)
        lanes = getattr(result, "gcs_lanes", [])
        n_lanes = len(lanes)
        string += f"{n_lanes} lane{'s' * (n_lanes != 1)}, {result.speed['inference']:.1f}ms"

        if self.args.save or self.args.show:
            self.plotted_img = draw_gcs_lanes(
                result.orig_img,
                lanes,
                show_scores=self.args.show_conf,
                line_width=self.args.line_width or 2,
            )
        if self.args.save_txt:
            save_gcs_lanes_txt(f"{self.txt_path}.txt", lanes, save_conf=self.args.save_conf)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)

        return string
