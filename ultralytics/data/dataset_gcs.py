# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Dataset utilities for GCS-YOLO-Lane structured lane labels."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils.gcs_shape import assert_gcs_shape, normalize_imgsz

__all__ = ("GCSLaneDataset", "gcs_collate_fn")


class GCSLaneDataset(Dataset):
    """Load images paired with GCS structured lane labels.

    The GCS-YOLO-Lane target contract is:
        lanes: normalized N x K x 2 point sequences, ordered bottom-to-top
        lane_valid: N x K point-valid masks

    Lane counts are intentionally variable per image. The collate function
    stacks only images, while keeping lane tensors as per-image lists for Hungarian matching.
    """

    def __init__(
        self,
        img_path: str | Path | list[str | Path] | None = None,
        imgsz: int | tuple[int, int] | list[int] = (544, 960),
        fraction: float = 1.0,
        image_dir: str | Path | list[str | Path] | None = None,
        label_dir: str | Path | list[str | Path] | None = None,
        img_size: int | tuple[int, int] | list[int] | None = None,
        strict: bool = True,
        augment: bool = False,
        hsv_h: float = 0.0,
        hsv_s: float = 0.0,
        hsv_v: float = 0.0,
        fliplr: float = 0.0,
        flipud: float = 0.0,
        translate: float = 0.0,
        scale: float = 0.0,
        erasing: float = 0.0,
        mosaic: float = 0.0,
        gt5_extra_aug: bool = False,
        gt5_aug_min_lanes: int = 5,
        gt5_erasing: float = 0.0,
        gt5_blur: float = 0.0,
        gt5_noise: float = 0.0,
        gt5_shadow: float = 0.0,
    ):
        """Create a GCS lane dataset from Ultralytics or reference-style arguments.

        Args:
            img_path: Ultralytics image directory/file/list path.
            imgsz: Training image size used by Ultralytics.
            fraction: Optional dataset fraction for quick overfit/debug runs.
            image_dir: Reference implementation alias for ``img_path``.
            label_dir: Optional explicit ``labels_gcs/<split>`` directory.
            img_size: Reference implementation alias for ``imgsz``.
            strict: If True, fail during construction when any image misses a label.
            augment: Enable GCS-safe image/point/mask augmentations.
            hsv_h: Random hue gain.
            hsv_s: Random saturation gain.
            hsv_v: Random value gain.
            fliplr: Horizontal flip probability.
            flipud: Vertical flip probability.
            translate: Random translation gain as a fraction of image width/height.
            scale: Random center scaling gain, e.g. 0.3 samples a scale factor from 0.7 to 1.3.
            erasing: Random erasing probability applied to image pixels only.
            mosaic: Four-image mosaic probability.
            gt5_extra_aug: Apply extra photometric multi-view augmentation to images with at least gt5_aug_min_lanes.
            gt5_aug_min_lanes: Minimum GT lane count that receives extra photometric augmentation.
            gt5_erasing: Additional erasing probability for gt5_extra_aug samples.
            gt5_blur: Random blur probability for gt5_extra_aug samples.
            gt5_noise: Random Gaussian noise probability for gt5_extra_aug samples.
            gt5_shadow: Random shadow probability for gt5_extra_aug samples.
        """
        source = img_path if img_path is not None else image_dir
        if source is None:
            raise ValueError("GCSLaneDataset requires img_path or image_dir.")
        if not (0.0 < float(fraction) <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}.")

        size = imgsz if img_size is None else img_size
        self.imgsz = self._parse_imgsz(size)
        self.img_h, self.img_w = self.imgsz
        self.img_size = self.imgsz
        self.image_dir = source
        if isinstance(label_dir, (list, tuple)):
            self.label_dir = [Path(x) for x in label_dir]
        else:
            self.label_dir = Path(label_dir) if label_dir is not None else None
        self.strict = bool(strict)
        self.augment = bool(augment)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.fliplr = float(fliplr)
        self.flipud = float(flipud)
        self.translate = float(translate)
        self.scale = float(scale)
        self.erasing = float(erasing)
        self.mosaic_prob = float(mosaic)
        self.mosaic = self.augment and self.mosaic_prob > 0.0
        self.gt5_extra_aug = bool(gt5_extra_aug)
        self.gt5_aug_min_lanes = max(int(gt5_aug_min_lanes), 1)
        self.gt5_erasing = float(gt5_erasing)
        self.gt5_blur = float(gt5_blur)
        self.gt5_noise = float(gt5_noise)
        self.gt5_shadow = float(gt5_shadow)
        if self.translate < 0.0 or self.translate > 1.0:
            raise ValueError(f"translate must be in [0, 1], got {self.translate}.")
        if self.scale < 0.0 or self.scale > 1.0:
            raise ValueError(f"scale must be in [0, 1], got {self.scale}.")
        if self.erasing < 0.0 or self.erasing > 1.0:
            raise ValueError(f"erasing must be in [0, 1], got {self.erasing}.")
        for name, value in (
            ("gt5_erasing", self.gt5_erasing),
            ("gt5_blur", self.gt5_blur),
            ("gt5_noise", self.gt5_noise),
            ("gt5_shadow", self.gt5_shadow),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

        self.im_files = self._find_images(source)
        if 0.0 < fraction < 1.0:
            self.im_files = self.im_files[: max(1, round(len(self.im_files) * fraction))]
        if not self.im_files:
            raise FileNotFoundError(f"No images found in {source}")

        self.label_files = [self._label_path(p) for p in self.im_files]
        self.image_paths = self.im_files
        self.label_paths = self.label_files
        self._check_label_pairs()
        self.point_mode = self._detect_point_mode()
        self.fixed_y_anchors = self._detect_fixed_y_anchors()
        if self.point_mode == "fixed_y" and self.mosaic:
            raise ValueError("fixed_y GCS labels do not support mosaic augmentation; set mosaic=0.0.")
        if self.point_mode == "fixed_y" and self.flipud > 0.0:
            raise ValueError("fixed_y GCS labels do not support vertical flip; set flipud=0.0.")

    @staticmethod
    def _parse_imgsz(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
        """Normalize image-size arguments to (height, width)."""
        is_scalar = isinstance(imgsz, int) or (
            isinstance(imgsz, (list, tuple)) and len(imgsz) == 1
        ) or (isinstance(imgsz, str) and imgsz.strip().isdigit())
        assert not is_scalar, (
            "GCSLaneDataset requires a rectangular H,W image shape such as [544, 960] for TuSimple "
            "or [384, 960] for CULane. A scalar imgsz would create a square 960x960 contract."
        )
        return normalize_imgsz(imgsz)

    @staticmethod
    def _find_images(img_path: str | Path | list[str | Path]) -> list[Path]:
        """Collect image files from common Ultralytics dataset path forms."""
        if isinstance(img_path, (list, tuple)):
            files: list[Path] = []
            for p in img_path:
                files.extend(GCSLaneDataset._find_images(p))
            return sorted(files)

        path = Path(img_path)
        if path.is_file() and path.suffix.lower() == ".txt":
            files = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                item = Path(line)
                if item.is_absolute() or item.exists():
                    files.append(item)
                else:
                    files.append(path.parent / item)
            return sorted(files)
        if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS:
            return [path]
        if path.is_dir():
            return sorted(p for p in path.rglob("*.*") if p.suffix[1:].lower() in IMG_FORMATS)
        raise FileNotFoundError(f"Image path does not exist or is unsupported: {path}")

    def _label_path(self, img_file: Path) -> Path:
        """Map images/<split>/name.jpg to labels_gcs/<split>/name.npz."""
        if self.label_dir is not None:
            if isinstance(self.label_dir, list):
                for label_dir in self.label_dir:
                    candidate = label_dir / f"{img_file.stem}.npz"
                    if candidate.exists():
                        return candidate
                return self.label_dir[0] / f"{img_file.stem}.npz"
            return self.label_dir / f"{img_file.stem}.npz"

        parts = list(img_file.parts)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels_gcs"
            return Path(*parts).with_suffix(".npz")
        return img_file.parent.parent / "labels_gcs" / img_file.parent.name / f"{img_file.stem}.npz"

    def _check_label_pairs(self) -> None:
        """Fail fast for missing labels, because GCS training needs one npz per image."""
        missing = [str(p) for p in self.label_files if not p.exists()]
        if not missing:
            return
        if self.strict:
            shown = "\n".join(missing[:5])
            extra = f"\n... and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise FileNotFoundError(f"Missing GCS labels for {len(missing)} image(s):\n{shown}{extra}")
        keep = [(im, lb) for im, lb in zip(self.im_files, self.label_files) if lb.exists()]
        self.im_files = [x[0] for x in keep]
        self.label_files = [x[1] for x in keep]
        if not self.im_files:
            raise FileNotFoundError("No image/labels_gcs npz pairs remain after filtering missing labels.")

    @staticmethod
    def _normalize_point_mode(mode: str) -> str:
        """Normalize supported GCS point-mode metadata values."""
        mode = str(mode).lower()
        if mode in {"fixed-y", "fixedy"}:
            mode = "fixed_y"
        if mode not in {"free", "fixed_y"}:
            raise ValueError(f"Unsupported point_mode={mode!r}.")
        return mode

    def _detect_point_mode(self) -> str:
        """Read and validate the dataset-wide point mode from label metadata."""
        if not self.label_files:
            return "free"

        modes: dict[str, list[Path]] = {}
        for label_file in self.label_files:
            if not label_file.exists():
                continue
            with np.load(label_file, allow_pickle=False) as data:
                raw_mode = str(np.asarray(data["point_mode"]).item()) if "point_mode" in data else "free"
            mode = self._normalize_point_mode(raw_mode)
            modes.setdefault(mode, []).append(label_file)

        if not modes:
            return "free"
        if len(modes) > 1:
            detail = {mode: str(paths[0]) for mode, paths in modes.items()}
            raise ValueError(f"Mixed GCS point_mode values in one dataset are not supported: {detail}")
        return next(iter(modes))

    def _detect_fixed_y_anchors(self) -> np.ndarray | None:
        """Read shared fixed-y anchors for fixed-y labels, if the dataset uses them."""
        if getattr(self, "point_mode", "free") != "fixed_y":
            return None
        expected = None
        for label_file in self.label_files:
            if not label_file.exists():
                continue
            with np.load(label_file, allow_pickle=False) as data:
                if "fixed_y" in data:
                    anchors = np.asarray(data["fixed_y"], dtype=np.float32).reshape(-1)
                elif "lanes" in data and data["lanes"].ndim == 3 and data["lanes"].shape[0] > 0:
                    anchors = np.asarray(data["lanes"][0, :, 1], dtype=np.float32).reshape(-1)
                else:
                    continue
            if anchors.size < 2:
                raise ValueError(f"{label_file}: fixed_y anchors must contain at least two points.")
            if not np.all(np.diff(anchors) < 0.0):
                raise ValueError(f"{label_file}: fixed_y anchors must be strictly descending from bottom to top.")
            if anchors.min() < -1e-4 or anchors.max() > 1.0 + 1e-4:
                raise ValueError(f"{label_file}: fixed_y anchors must be normalized to [0, 1].")
            anchors = np.clip(anchors, 0.0, 1.0).astype(np.float32)
            if expected is None:
                expected = anchors
                continue
            if anchors.shape != expected.shape:
                raise ValueError(
                    f"{label_file}: fixed_y anchor shape {anchors.shape} does not match dataset anchor shape {expected.shape}."
                )
            max_err = float(np.max(np.abs(anchors - expected))) if anchors.size else 0.0
            if max_err > 5e-5:
                raise ValueError(
                    f"{label_file}: fixed_y anchors differ from the dataset contract, max_err={max_err:.6g}. "
                    f"dataset first/last=({expected[0]:.9f}, {expected[-1]:.9f}), "
                    f"label first/last=({anchors[0]:.9f}, {anchors[-1]:.9f})."
                )
        return expected

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.im_files)

    @staticmethod
    def _require_label_keys(data: Any, label_file: Path) -> None:
        """Check that the compressed label contains all arrays used by GCS-YOLO-Lane."""
        required = {"lanes", "lane_valid"}
        missing = sorted(k for k in required if k not in data)
        if missing:
            raise KeyError(f"{label_file} is missing required GCS arrays: {missing}")

    @staticmethod
    def _normalize_lanes(
        lanes: np.ndarray,
        lane_valid: np.ndarray,
        label_file: Path,
        tol: float = 1e-4,
        num_lanes: np.ndarray | None = None,
        point_mode: str = "free",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate normalized lanes and enforce bottom-to-top point order."""
        point_mode = str(point_mode).lower()
        if point_mode in {"fixed-y", "fixedy"}:
            point_mode = "fixed_y"
        lanes = np.asarray(lanes, dtype=np.float32)
        lane_valid = np.asarray(lane_valid, dtype=np.float32)
        if num_lanes is not None:
            expected_lanes = int(np.asarray(num_lanes).reshape(-1)[0])
            if expected_lanes != lanes.shape[0]:
                raise ValueError(f"{label_file}: num_lanes={expected_lanes} does not match lanes N={lanes.shape[0]}.")

        if lanes.ndim != 3 or lanes.shape[-1] != 2:
            raise ValueError(f"{label_file}: lanes must have shape N x K x 2, got {lanes.shape}.")
        if lane_valid.shape != lanes.shape[:2]:
            raise ValueError(f"{label_file}: lane_valid shape {lane_valid.shape} must match lanes {lanes.shape[:2]}.")
        if not np.isfinite(lanes).all():
            raise ValueError(f"{label_file}: lanes contain NaN or Inf coordinates.")
        if not np.isfinite(lane_valid).all():
            raise ValueError(f"{label_file}: lane_valid contains NaN or Inf values.")

        lane_valid = (lane_valid > 0.5).astype(np.float32)
        valid_coords = lanes[lane_valid > 0.5]
        if valid_coords.size:
            coord_min = float(valid_coords.min())
            coord_max = float(valid_coords.max())
            if coord_min < -tol or coord_max > 1.0 + tol:
                raise ValueError(
                    f"{label_file}: lane coordinates must be normalized to [0, 1], "
                    f"got min={coord_min:.6f}, max={coord_max:.6f}."
                )
        lanes = np.clip(lanes, 0.0, 1.0)

        keep = lane_valid.sum(axis=1) >= 2
        lanes = lanes[keep]
        lane_valid = lane_valid[keep]
        if lanes.shape[0] == 0:
            return lanes.astype(np.float32), lane_valid.astype(np.float32)

        if point_mode == "fixed_y":
            for i, (lane, valid) in enumerate(zip(lanes, lane_valid)):
                ys = lane[valid > 0.5, 1]
                if ys.shape[0] >= 2 and not np.all(np.diff(ys) <= 1e-6):
                    raise ValueError(f"{label_file}: fixed_y lane {i} valid y anchors must be bottom-to-top.")
            return lanes.astype(np.float32), lane_valid.astype(np.float32)

        ordered_lanes = np.zeros_like(lanes, dtype=np.float32)
        ordered_valid = np.zeros_like(lane_valid, dtype=np.float32)
        for i, (lane, valid) in enumerate(zip(lanes, lane_valid)):
            pts = lane[valid > 0.5]
            if pts.shape[0] < 2:
                continue
            order = np.argsort(-pts[:, 1], kind="stable")
            pts = pts[order]
            n = pts.shape[0]
            ordered_lanes[i, :n] = pts
            ordered_valid[i, :n] = 1.0

        keep = ordered_valid.sum(axis=1) >= 2
        return ordered_lanes[keep].astype(np.float32), ordered_valid[keep].astype(np.float32)

    def _load_label(self, label_file: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load and validate one GCS npz label."""
        with np.load(label_file, allow_pickle=False) as data:
            self._require_label_keys(data, label_file)
            lanes = data["lanes"]
            lane_valid = data["lane_valid"]
            num_lanes = data["num_lanes"] if "num_lanes" in data else None
            point_mode = str(np.asarray(data["point_mode"]).item()) if "point_mode" in data else "free"

        lanes, lane_valid = self._normalize_lanes(
            lanes,
            lane_valid,
            label_file,
            num_lanes=num_lanes,
            point_mode=point_mode,
        )
        mode = self._normalize_point_mode(point_mode)
        if mode == "fixed_y" and self.fixed_y_anchors is not None and lanes.shape[0]:
            expected_y = self.fixed_y_anchors.reshape(1, -1)
            if lanes.shape[1] != expected_y.shape[1]:
                raise ValueError(
                    f"{label_file}: fixed_y lane K={lanes.shape[1]} does not match dataset anchors K={expected_y.shape[1]}."
                )
            y_err = np.abs(lanes[..., 1] - expected_y) * (lane_valid > 0.5)
            max_err = float(y_err.max()) if y_err.size else 0.0
            if max_err > 5e-5:
                raise ValueError(
                    f"{label_file}: fixed_y lane y coordinates do not match dataset anchors, max_err={max_err:.6g}."
                )
        return lanes, lane_valid

    def _load_resized_sample(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, Path, tuple[int, int]]:
        """Load one image/label pair and resize the image to the configured GCS training shape."""
        img_file = self.im_files[index]
        label_file = self.label_files[index]
        if not label_file.exists():
            raise FileNotFoundError(f"Missing GCS label for {img_file}: {label_file}")

        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_file}")
        h0, w0 = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != self.imgsz:
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        assert_gcs_shape(img.shape[:2], self.imgsz, name="image", context=f"GCSLaneDataset({img_file})")

        lanes, lane_valid = self._load_label(label_file)
        return img, lanes, lane_valid, img_file, label_file, (h0, w0)

    @staticmethod
    def _empty_lane_arrays(num_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Return empty lane arrays with the current fixed point count."""
        return (
            np.zeros((0, num_points, 2), dtype=np.float32),
            np.zeros((0, num_points), dtype=np.float32),
        )

    def _load_mosaic(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, Path, tuple[int, int]]:
        """Create a GCS-safe four-tile mosaic and remap normalized lane points."""
        out_h, out_w = self.imgsz
        half_h = out_h // 2
        half_w = out_w // 2
        tiles = (
            (0, 0, half_w, half_h),
            (half_w, 0, out_w, half_h),
            (0, half_h, half_w, out_h),
            (half_w, half_h, out_w, out_h),
        )
        if len(self.im_files) >= 4:
            pool = [i for i in range(len(self.im_files)) if i != index]
            indices = [index, *random.sample(pool, k=3)]
        else:
            indices = [index, *random.choices(range(len(self.im_files)), k=3)]

        out_img = np.full((out_h, out_w, 3), 114, dtype=np.uint8)
        lane_parts: list[np.ndarray] = []
        valid_parts: list[np.ndarray] = []
        num_points = 0
        first_img_file: Path | None = None
        first_label_file: Path | None = None
        first_shape = self.imgsz

        for tile, sample_index in zip(tiles, indices):
            x1, y1, x2, y2 = tile
            tile_w, tile_h = x2 - x1, y2 - y1
            img, lanes, lane_valid, img_file, label_file, ori_shape = self._load_resized_sample(sample_index)
            if first_img_file is None:
                first_img_file, first_label_file, first_shape = img_file, label_file, ori_shape
            if lanes.ndim == 3:
                num_points = lanes.shape[1]

            out_img[y1:y2, x1:x2] = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

            if lanes.shape[0]:
                remapped = lanes.copy()
                remapped[..., 0] = (remapped[..., 0] * tile_w + x1) / float(out_w)
                remapped[..., 1] = (remapped[..., 1] * tile_h + y1) / float(out_h)
                lane_parts.append(remapped)
                valid_parts.append(lane_valid.copy())

        if lane_parts:
            lanes = np.concatenate(lane_parts, axis=0).astype(np.float32)
            lane_valid = np.concatenate(valid_parts, axis=0).astype(np.float32)
            lanes, lane_valid = self._normalize_lanes(
                lanes,
                lane_valid,
                Path("<mosaic>"),
                point_mode=getattr(self, "point_mode", "free"),
            )
        else:
            lanes, lane_valid = self._empty_lane_arrays(num_points)

        assert_gcs_shape(out_img.shape[:2], self.imgsz, name="mosaic image", context="GCSLaneDataset._load_mosaic")
        return out_img, lanes, lane_valid, first_img_file, first_label_file, first_shape

    def close_mosaic(self, hyp: dict | None = None) -> None:
        """Disable mosaic augmentation for late training epochs."""
        self.mosaic = False
        self.mosaic_prob = 0.0

    def _augment_hsv(self, img: np.ndarray) -> np.ndarray:
        """Apply Ultralytics-style HSV jitter to an RGB image."""
        if not (self.hsv_h or self.hsv_s or self.hsv_v):
            return img
        r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v]
        x = np.arange(0, 256, dtype=np.float32)
        lut_hue = ((x + r[0] * 180) % 180).astype(np.uint8)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(np.uint8)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(np.uint8)
        lut_sat[0] = 0

        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _affine_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply a 2x3 affine matrix to Nx2 pixel points."""
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        return np.concatenate([points.astype(np.float32), ones], axis=1) @ matrix.T

    def _transform_free_lanes_affine(
        self,
        lanes: np.ndarray,
        lane_valid: np.ndarray,
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform free-mode normalized lane points with an affine matrix."""
        out = np.zeros_like(lanes, dtype=np.float32)
        out_valid = np.zeros_like(lane_valid, dtype=np.float32)
        for i, (lane, valid) in enumerate(zip(lanes, lane_valid)):
            mask = valid > 0.5
            if mask.sum() < 2:
                continue
            pts = lane[mask].copy()
            pts[:, 0] *= self.img_w
            pts[:, 1] *= self.img_h
            transformed = self._affine_points(pts, matrix)
            keep = (
                (transformed[:, 0] >= 0.0)
                & (transformed[:, 0] <= self.img_w - 1.0)
                & (transformed[:, 1] >= 0.0)
                & (transformed[:, 1] <= self.img_h - 1.0)
            )
            if keep.sum() < 2:
                continue
            transformed = transformed[keep]
            n = transformed.shape[0]
            out[i, :n, 0] = np.clip(transformed[:, 0] / float(self.img_w), 0.0, 1.0)
            out[i, :n, 1] = np.clip(transformed[:, 1] / float(self.img_h), 0.0, 1.0)
            out_valid[i, :n] = 1.0
        return self._normalize_lanes(out, out_valid, Path("<scale>"), point_mode="free")

    def _transform_fixed_y_lanes_affine(
        self,
        lanes: np.ndarray,
        lane_valid: np.ndarray,
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform lanes, then resample x back onto the shared fixed-y anchors."""
        anchors = self.fixed_y_anchors
        if anchors is None:
            anchors = lanes[0, :, 1].astype(np.float32) if lanes.shape[0] else None
        if anchors is None or anchors.shape[0] != lanes.shape[1]:
            raise ValueError("fixed_y scale augmentation requires label fixed_y anchors matching lanes N x K x 2.")

        k = anchors.shape[0]
        target_y_px = anchors.astype(np.float32) * float(self.img_h)
        out = np.zeros_like(lanes, dtype=np.float32)
        out[..., 1] = anchors.reshape(1, k)
        out_valid = np.zeros_like(lane_valid, dtype=np.float32)

        for i, (lane, valid) in enumerate(zip(lanes, lane_valid)):
            valid_indices = np.flatnonzero(valid > 0.5)
            if valid_indices.shape[0] < 2:
                continue
            breaks = np.where(np.diff(valid_indices) > 1)[0] + 1
            for run in np.split(valid_indices, breaks):
                if run.shape[0] < 2:
                    continue
                pts = lane[run].copy()
                pts[:, 0] *= self.img_w
                pts[:, 1] *= self.img_h
                transformed = self._affine_points(pts, matrix)
                order = np.argsort(transformed[:, 1], kind="stable")
                ys = transformed[order, 1]
                xs = transformed[order, 0]

                unique_ys, unique_idx = np.unique(ys, return_index=True)
                if unique_ys.shape[0] < 2:
                    continue
                xs = xs[unique_idx]
                in_y_range = (target_y_px >= unique_ys.min()) & (target_y_px <= unique_ys.max())
                if in_y_range.sum() < 2:
                    continue
                sampled_x = np.interp(target_y_px[in_y_range], unique_ys, xs).astype(np.float32)
                x_valid = (sampled_x >= 0.0) & (sampled_x <= self.img_w - 1.0)
                anchor_indices = np.flatnonzero(in_y_range)
                if x_valid.sum() < 2:
                    continue
                kept_indices = anchor_indices[x_valid]
                out[i, kept_indices, 0] = np.clip(sampled_x[x_valid] / float(self.img_w), 0.0, 1.0)
                out_valid[i, kept_indices] = 1.0

        return self._normalize_lanes(out, out_valid, Path("<scale>"), point_mode="fixed_y")

    def _transform_lanes_affine(
        self,
        lanes: np.ndarray,
        lane_valid: np.ndarray,
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply affine geometry to lane labels while respecting free/fixed-y point contracts."""
        if not lanes.shape[0]:
            return lanes, lane_valid
        if getattr(self, "point_mode", "free") == "fixed_y":
            return self._transform_fixed_y_lanes_affine(lanes, lane_valid, matrix)
        return self._transform_free_lanes_affine(lanes, lane_valid, matrix)

    def _apply_affine_augment(
        self,
        img: np.ndarray,
        lanes: np.ndarray,
        lane_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply GCS-safe scale and translation augmentation to image and structured lane labels."""
        if self.scale <= 0.0 and self.translate <= 0.0:
            return img, lanes, lane_valid

        factor = max(random.uniform(1.0 - self.scale, 1.0 + self.scale), 1e-3) if self.scale > 0.0 else 1.0
        shift_x = random.uniform(-self.translate, self.translate) * float(self.img_w) if self.translate > 0.0 else 0.0
        shift_y = random.uniform(-self.translate, self.translate) * float(self.img_h) if self.translate > 0.0 else 0.0
        if abs(factor - 1.0) < 1e-3 and abs(shift_x) < 1e-3 and abs(shift_y) < 1e-3:
            return img, lanes, lane_valid

        cx = (self.img_w - 1.0) * 0.5
        cy = (self.img_h - 1.0) * 0.5
        matrix = np.array(
            [[factor, 0.0, (1.0 - factor) * cx + shift_x], [0.0, factor, (1.0 - factor) * cy + shift_y]],
            dtype=np.float32,
        )
        img = cv2.warpAffine(
            img,
            matrix,
            (self.img_w, self.img_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )
        lanes, lane_valid = self._transform_lanes_affine(lanes, lane_valid, matrix)
        return img, lanes, lane_valid

    def _apply_random_erasing(self, img: np.ndarray, probability: float | None = None) -> np.ndarray:
        """Randomly erase one image rectangle while leaving structured labels unchanged."""
        prob = self.erasing if probability is None else float(probability)
        if prob <= 0.0 or random.random() >= prob:
            return img

        h, w = img.shape[:2]
        area = float(h * w)
        for _ in range(10):
            target_area = random.uniform(0.02, 0.20) * area
            aspect = random.uniform(0.3, 3.3)
            erase_h = int(round((target_area / aspect) ** 0.5))
            erase_w = int(round((target_area * aspect) ** 0.5))
            if 0 < erase_h < h and 0 < erase_w < w:
                y1 = random.randint(0, h - erase_h)
                x1 = random.randint(0, w - erase_w)
                img = img.copy()
                img[y1 : y1 + erase_h, x1 : x1 + erase_w] = np.array([114, 114, 114], dtype=np.uint8)
                break
        return img

    @staticmethod
    def _apply_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
        """Apply small RGB Gaussian noise while preserving uint8 image format."""
        noise = np.random.normal(0.0, float(sigma), img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _apply_random_shadow(self, img: np.ndarray) -> np.ndarray:
        """Apply a broad synthetic shadow band to diversify rare 5-lane lighting."""
        h, w = img.shape[:2]
        y1 = random.randint(0, max(h - 1, 0))
        y2 = random.randint(0, max(h - 1, 0))
        band = random.randint(max(12, h // 12), max(13, h // 4))
        polygon = np.array(
            [
                [0, np.clip(y1 - band, 0, h - 1)],
                [w - 1, np.clip(y2 - band, 0, h - 1)],
                [w - 1, np.clip(y2 + band, 0, h - 1)],
                [0, np.clip(y1 + band, 0, h - 1)],
            ],
            dtype=np.int32,
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        factor = random.uniform(0.55, 0.85)
        out = img.astype(np.float32)
        out[mask > 0] *= factor
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_gt5_extra_photometric(self, img: np.ndarray) -> np.ndarray:
        """Apply extra image-only augmentation to rare >=5-lane training samples."""
        if self.gt5_blur > 0.0 and random.random() < self.gt5_blur:
            kernel = random.choice((3, 5))
            img = cv2.GaussianBlur(img, (kernel, kernel), 0)
        if self.gt5_noise > 0.0 and random.random() < self.gt5_noise:
            img = self._apply_gaussian_noise(img, sigma=random.uniform(3.0, 10.0))
        if self.gt5_shadow > 0.0 and random.random() < self.gt5_shadow:
            img = self._apply_random_shadow(img)
        if self.gt5_erasing > 0.0:
            img = self._apply_random_erasing(img, probability=self.gt5_erasing)
        return img

    def _apply_geometric_augment(
        self,
        img: np.ndarray,
        lanes: np.ndarray,
        lane_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random scale/flips while keeping GCS points synchronized."""
        img, lanes, lane_valid = self._apply_affine_augment(
            img,
            lanes,
            lane_valid,
        )

        if self.fliplr > 0.0 and random.random() < self.fliplr:
            img = np.ascontiguousarray(img[:, ::-1])
            if lanes.shape[0]:
                lanes = lanes.copy()
                lanes[..., 0] = 1.0 - lanes[..., 0]

        if self.flipud > 0.0 and random.random() < self.flipud:
            img = np.ascontiguousarray(img[::-1])
            if lanes.shape[0]:
                lanes = lanes.copy()
                lanes[..., 1] = 1.0 - lanes[..., 1]

        if lanes.shape[0]:
            lanes, lane_valid = self._normalize_lanes(
                lanes,
                lane_valid,
                Path("<augment>"),
                point_mode=getattr(self, "point_mode", "free"),
            )
        return img, lanes, lane_valid

    def _apply_augmentations(
        self,
        img: np.ndarray,
        lanes: np.ndarray,
        lane_valid: np.ndarray,
        gt5_extra: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply photometric and geometric GCS training augmentations."""
        img = self._augment_hsv(img)
        if gt5_extra:
            img = self._apply_gt5_extra_photometric(img)
        img, lanes, lane_valid = self._apply_geometric_augment(
            img,
            lanes,
            lane_valid,
        )
        img = self._apply_random_erasing(img)
        return img, lanes, lane_valid

    def __getitem__(self, index: int) -> dict:
        """Load one image and its structured lane targets."""
        if self.mosaic and random.random() < self.mosaic_prob:
            img, lanes, lane_valid, img_file, label_file, ori_shape = self._load_mosaic(index)
        else:
            img, lanes, lane_valid, img_file, label_file, ori_shape = self._load_resized_sample(index)

        if self.augment:
            gt5_extra = self.gt5_extra_aug and lanes.shape[0] >= self.gt5_aug_min_lanes
            img, lanes, lane_valid = self._apply_augmentations(
                img,
                lanes,
                lane_valid,
                gt5_extra=gt5_extra,
            )

        assert_gcs_shape(img.shape[:2], self.imgsz, name="image", context=f"GCSLaneDataset.__getitem__({img_file})")
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        lanes_t = torch.from_numpy(lanes)
        lane_valid_t = torch.from_numpy(lane_valid)
        return {
            "img": torch.from_numpy(img),
            "im_file": str(img_file),
            "path": str(img_file),
            "label_file": str(label_file),
            "ori_shape": ori_shape,
            "resized_shape": self.imgsz,
            "lanes": lanes_t,
            "lane_valid": lane_valid_t,
            "gt_lanes": lanes_t,
            "gt_lane_valid": lane_valid_t,
            "num_lanes": torch.tensor(lanes.shape[0], dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate variable-lane targets while stacking only image tensors."""
        lanes = [b["lanes"] for b in batch]
        lane_valid = [b["lane_valid"] for b in batch]
        num_lanes = torch.stack([b["num_lanes"] for b in batch])
        total_lanes = int(num_lanes.sum().item())

        return {
            "img": torch.stack([b["img"] for b in batch], dim=0),
            "im_file": [b["im_file"] for b in batch],
            "path": [b["path"] for b in batch],
            "label_file": [b["label_file"] for b in batch],
            "ori_shape": [b["ori_shape"] for b in batch],
            "resized_shape": [b["resized_shape"] for b in batch],
            "lanes": lanes,
            "lane_valid": lane_valid,
            "gt_lanes": lanes,
            "gt_lane_valid": lane_valid,
            "num_lanes": num_lanes,
            "cls": torch.zeros((total_lanes, 1), dtype=torch.float32),
            "batch_idx": torch.cat(
                [torch.full((int(n),), i, dtype=torch.long) for i, n in enumerate(num_lanes.tolist())], dim=0
            )
            if total_lanes
            else torch.zeros((0,), dtype=torch.long),
        }


def gcs_collate_fn(batch: list[dict]) -> dict:
    """Reference-compatible GCS collate function."""
    return GCSLaneDataset.collate_fn(batch)
