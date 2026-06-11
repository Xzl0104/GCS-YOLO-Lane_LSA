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
from ultralytics.utils import ops
from ultralytics.utils.gcs_shape import assert_gcs_image_tensor, assert_gcs_shape, normalize_imgsz
from ultralytics.utils.gcs_postprocess import GCS_DEFAULT_MAX_DET, decode_gcs_predictions, draw_gcs_lanes, save_gcs_lanes_txt


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
            self.lane_exist_scores = torch.tensor([x.get("exist_score", x["score"]) for x in lanes], dtype=torch.float32)
            self.lane_rank_scores = torch.tensor([x.get("rank_score", x["score"]) for x in lanes], dtype=torch.float32)
            self.lane_queries = torch.tensor([x["query"] for x in lanes], dtype=torch.long)
        else:
            self.lanes = torch.zeros((0, 0, 2), dtype=torch.float32)
            self.lanes_normalized = torch.zeros((0, 0, 2), dtype=torch.float32)
            self.lane_scores = torch.zeros((0,), dtype=torch.float32)
            self.lane_exist_scores = torch.zeros((0,), dtype=torch.float32)
            self.lane_rank_scores = torch.zeros((0,), dtype=torch.float32)
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
        valid_logits = preds.get("pred_valid_logits")
        if valid_logits is not None:
            valid_logits = valid_logits.detach()
        count_logits = preds.get("pred_count_logits")
        if count_logits is not None:
            count_logits = count_logits.detach()
        count_boundary_logits = preds.get("pred_count_boundary_logits")
        if count_boundary_logits is not None:
            count_boundary_logits = count_boundary_logits.detach()
        quality_logits = preds.get("pred_quality_logits")
        if quality_logits is not None:
            quality_logits = quality_logits.detach()
        conf = 0.25 if self.args.conf is None else float(self.args.conf)
        generic_max_det = getattr(self.args, "max_det", None)
        if generic_max_det is not None and int(generic_max_det) != 300:
            max_det = int(generic_max_det)
        else:
            max_det = int(getattr(self.args, "gcs_eval_max_det", GCS_DEFAULT_MAX_DET) or GCS_DEFAULT_MAX_DET)
        nms_dist_px = float(getattr(self.args, "gcs_eval_nms_dist_px", 18.0) or 0.0)
        point_valid_thr = getattr(self.args, "gcs_eval_point_valid_thr", None)
        if point_valid_thr is None:
            point_valid_thr = getattr(self.args, "point_valid_thr", 0.5)
        point_valid_thr = float(point_valid_thr)
        min_points = int(getattr(self.args, "gcs_eval_min_points", 6) or 0)
        use_count_head_decode = bool(getattr(self.args, "gcs_use_count_head_decode", True))
        count_head_temperature = float(getattr(self.args, "gcs_count_head_temp", 1.0) or 1.0)
        candidate_score_thr = float(getattr(self.args, "gcs_decode_candidate_conf", 0.05) or 0.0)
        candidate_point_valid_thr = float(getattr(self.args, "gcs_decode_candidate_point_valid_thr", 0.20) or 0.0)
        candidate_min_points = int(getattr(self.args, "gcs_decode_candidate_min_points", 5) or 5)
        final_min_points = int(getattr(self.args, "gcs_decode_final_min_points", 6) or 6)
        fifth_min_points = int(getattr(self.args, "gcs_decode_fifth_min_points", 5) or 5)
        line_nms_min_overlap = max(int(getattr(self.args, "gcs_line_nms_min_overlap", 6) or 6), 1)
        line_nms_rescue_dist_px = float(getattr(self.args, "gcs_line_nms_rescue_dist_px", 30.0) or 0.0)
        quality_rescue_5th = bool(getattr(self.args, "gcs_quality_rescue_5th", True))
        quality_rescue_count5_thr = float(getattr(self.args, "gcs_quality_rescue_count5_thr", 0.70))
        quality_rescue_conf_thr = float(getattr(self.args, "gcs_quality_rescue_conf_thr", 0.03))
        quality_rescue_mean_valid_thr = float(getattr(self.args, "gcs_quality_rescue_mean_valid_thr", 0.45))
        quality_rescue_quality_thr = float(getattr(self.args, "gcs_quality_rescue_quality_thr", 0.55))
        quality_rescue_min_points = int(getattr(self.args, "gcs_quality_rescue_min_points", 5) or 5)
        quality_rescue_dist_px = float(getattr(self.args, "gcs_quality_rescue_dist_px", 24.0) or 0.0)

        results = []
        if valid_logits is None:
            valid_iter = [None] * int(points.shape[0])
        else:
            valid_iter = list(valid_logits)
        if count_logits is None:
            count_iter = [None] * int(points.shape[0])
        else:
            count_iter = list(count_logits)
        if count_boundary_logits is None:
            count_boundary_iter = [None] * int(points.shape[0])
        else:
            count_boundary_iter = list(count_boundary_logits)
        if quality_logits is None:
            quality_iter = [None] * int(points.shape[0])
        else:
            quality_iter = list(quality_logits)

        for (
            lane_points,
            lane_logits,
            lane_valid_logits,
            lane_count_logits,
            lane_count_boundary_logits,
            lane_quality_logits,
            orig_img,
            img_path,
        ) in zip(
            points, logits, valid_iter, count_iter, count_boundary_iter, quality_iter, orig_imgs, self.batch[0]
        ):
            lanes = decode_gcs_predictions(
                lane_points,
                lane_logits,
                pred_valid_logits=lane_valid_logits,
                pred_count_logits=lane_count_logits,
                pred_count_boundary_logits=lane_count_boundary_logits,
                pred_quality_logits=lane_quality_logits,
                image_shape=orig_img.shape[:2],
                score_thr=conf,
                point_valid_thr=point_valid_thr,
                min_points=min_points,
                max_det=max_det,
                nms_dist_px=nms_dist_px,
                use_count_head_decode=use_count_head_decode,
                count_head_temperature=count_head_temperature,
                dataset_name="tusimple",
                candidate_score_thr=candidate_score_thr,
                candidate_point_valid_thr=candidate_point_valid_thr,
                candidate_min_points=candidate_min_points,
                final_min_points=final_min_points,
                fifth_min_points=fifth_min_points,
                line_nms_min_overlap=line_nms_min_overlap,
                line_nms_rescue_dist_px=line_nms_rescue_dist_px,
                quality_rescue_5th=quality_rescue_5th,
                quality_rescue_count5_thr=quality_rescue_count5_thr,
                quality_rescue_conf_thr=quality_rescue_conf_thr,
                quality_rescue_mean_valid_thr=quality_rescue_mean_valid_thr,
                quality_rescue_quality_thr=quality_rescue_quality_thr,
                quality_rescue_min_points=quality_rescue_min_points,
                quality_rescue_dist_px=quality_rescue_dist_px,
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
