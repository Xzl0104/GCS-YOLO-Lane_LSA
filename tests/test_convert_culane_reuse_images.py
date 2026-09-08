from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from tools.check_culane_dataset import inspect_label
from tools.convert_culane_to_gcs import convert, output_stem
from ultralytics.data.utils import check_det_dataset


def test_converter_rebuilds_fixed_y_labels_from_lines_with_reused_images(tmp_path: Path) -> None:
    raw_file = "driver_1_30frame/01010101_0001.MP4/00000.jpg"
    archive_root = tmp_path / "annotations"
    reuse_root = tmp_path / "reused_images"
    output_root = tmp_path / "converted"

    list_path = archive_root / "list" / "train.txt"
    list_path.parent.mkdir(parents=True)
    list_path.write_text(f"/{raw_file}\n", encoding="utf-8")
    source_lines = archive_root / Path(raw_file).with_suffix(".lines.txt")
    source_lines.parent.mkdir(parents=True)
    source_lines.write_text("100 589 110 400 120 200 130 39\n", encoding="utf-8")

    stem = output_stem(raw_file)
    reuse_image = reuse_root / "train" / f"{stem}.jpg"
    reuse_image.parent.mkdir(parents=True)
    assert cv2.imwrite(str(reuse_image), np.full((384, 960, 3), 127, dtype=np.uint8))

    summary = convert(
        argparse.Namespace(
            archive_root=str(archive_root),
            output_root=str(output_root),
            imgsz=(384, 960),
            raw_imgsz=(590, 1640),
            num_points=56,
            fixed_y_start_px=589.0,
            fixed_y_end_px=39.0,
            line_width=8,
            splits=["train"],
            overwrite=False,
            verify_image_shapes=False,
            reuse_images_root=str(reuse_root),
            reuse_image_mode="hardlink",
        )
    )

    output_image = output_root / "images" / "train" / f"{stem}.jpg"
    output_label = output_root / "labels_gcs" / "train" / f"{stem}.npz"
    assert summary["image_source"] == "reused_preprocessed"
    assert output_image.is_file()
    assert output_label.is_file()
    assert os.path.samefile(reuse_image, output_image)
    dataset_yaml = output_root / "dataset.yaml"
    assert f"path: {json.dumps(output_root.resolve().as_posix())}" in dataset_yaml.read_text(encoding="utf-8")
    data = check_det_dataset(str(dataset_yaml))
    assert data["path"] == output_root.resolve()
    assert data["train"] == str(output_root / "images" / "train")
    assert data["val"] == str(output_root / "images" / "val")

    with np.load(output_label, allow_pickle=False) as label:
        assert str(label["point_mode"]) == "fixed_y"
        assert tuple(label["lanes"].shape) == (1, 56, 2)
        assert tuple(label["lane_valid"].shape) == (1, 56)
        assert int(label["num_lanes"][0]) == 1
        assert np.allclose(label["lanes"][..., 1], label["fixed_y"].reshape(1, -1))
        assert str(label["source_image"]) == raw_file
        assert str(label["source_lines"]) == raw_file.replace(".jpg", ".lines.txt")

    errors = inspect_label(
        output_label,
        output_image,
        expected_shape=(384, 960),
        num_points=56,
        archive_root=None,
        source_lines_root=archive_root,
    )
    assert errors == []
