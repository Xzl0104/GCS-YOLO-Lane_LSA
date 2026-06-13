from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gcs_tools.label_utils import fixed_y_anchors
from tools import check_tusimple_fixed_y_label_oracle as oracle
from tools import rebuild_tusimple_fixed_y_k56_from_reference_split as builder
from tools import train_gcs


ROOT = Path(__file__).resolve().parents[1]


def test_k56_fixed_y_anchors_match_tusimple_official_h_samples():
    anchors = fixed_y_anchors(num_points=56, y_start=710.0 / 720.0, y_end=160.0 / 720.0)
    expected = [y / 720.0 for y in range(710, 159, -10)]
    assert anchors.shape == (56,)
    assert len(expected) == 56
    assert np.allclose(anchors, expected, rtol=0.0, atol=1e-7)


def test_k56_data_and_model_contract_match():
    data = yaml.safe_load((ROOT / "data" / "tusimple_gcs_fixed_y_k56_960x544.yaml").read_text(encoding="utf-8"))
    model = yaml.safe_load(
        (ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q12-k56.yaml").read_text(
            encoding="utf-8"
        )
    )
    head_args = model["head"][-1][3]

    assert data["point_mode"] == "fixed_y"
    assert data["num_points"] == 56
    assert head_args[0] == 12
    assert head_args[1] == 56
    assert head_args[4] == "fixed_y"
    assert math.isclose(float(data["fixed_y"][0]), float(head_args[5]))
    assert math.isclose(float(data["fixed_y"][1]), float(head_args[6]))


def test_k56_train_command_infers_k56_label_dirs_from_data_yaml():
    labels = train_gcs.infer_gcs_label_dirs_from_data(ROOT / "data" / "tusimple_gcs_fixed_y_k56_960x544.yaml")

    assert labels["train"] == str(ROOT / "datasets" / "tusimple_fixed_y_k56_960x544" / "labels_gcs" / "train")
    assert labels["val"] == str(ROOT / "datasets" / "tusimple_fixed_y_k56_960x544" / "labels_gcs" / "val")


def test_train_command_label_inference_does_not_follow_image_symlink(tmp_path):
    legacy_train = tmp_path / "legacy" / "images" / "train"
    legacy_val = tmp_path / "legacy" / "images" / "val"
    legacy_train.mkdir(parents=True)
    legacy_val.mkdir(parents=True)
    k56_images = tmp_path / "k56" / "images"
    k56_images.mkdir(parents=True)
    try:
        (k56_images / "train").symlink_to(legacy_train, target_is_directory=True)
        (k56_images / "val").symlink_to(legacy_val, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"train: {k56_images / 'train'}\nval: {k56_images / 'val'}\nnames:\n  0: lane\n",
        encoding="utf-8",
    )

    labels = train_gcs.infer_gcs_label_dirs_from_data(data_yaml)

    assert labels["train"] == str(tmp_path / "k56" / "labels_gcs" / "train")
    assert labels["val"] == str(tmp_path / "k56" / "labels_gcs" / "val")


def test_train_command_does_not_pair_yaml_labels_with_cli_image_override():
    assert train_gcs.choose_gcs_label_dir("explicit_labels", "override_images", "yaml_labels") == "explicit_labels"
    assert train_gcs.choose_gcs_label_dir(None, "override_images", "yaml_labels") is None
    assert train_gcs.choose_gcs_label_dir(None, None, "yaml_labels") == "yaml_labels"


def test_train_command_does_not_infer_split_label_dir_for_manifest_or_list_entries(tmp_path):
    manifest = tmp_path / "train.txt"
    manifest.write_text("images/train/0001.jpg\n", encoding="utf-8")
    val_images = tmp_path / "k56" / "images" / "val"
    val_images.mkdir(parents=True)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"train: {manifest}",
                "val:",
                f"  - {val_images}",
                "names:",
                "  0: lane",
                "",
            ]
        ),
        encoding="utf-8",
    )

    labels = train_gcs.infer_gcs_label_dirs_from_data(data_yaml)

    assert "train" not in labels
    assert "val" not in labels


def test_k56_builder_rejects_split_raw_file_overlap():
    split_samples = {
        "train": [SimpleNamespace(raw_file="clips/0601/0001/20.jpg")],
        "val": [SimpleNamespace(raw_file="clips/0601/0002/20.jpg")],
        "test": [SimpleNamespace(raw_file="clips/0601/0001/20.jpg")],
    }

    with pytest.raises(ValueError, match="raw_file overlap"):
        builder.assert_disjoint_raw_file_splits(split_samples)


def test_k56_label_oracle_requires_explicit_test_gt():
    args = SimpleNamespace(label_split="test", gt_json=None, archive_root="archive")

    with pytest.raises(SystemExit, match="explicit --gt-json"):
        oracle.resolve_gt_json(args)


def test_k56_label_oracle_defaults_to_official_val_gt():
    args = SimpleNamespace(label_split="val", gt_json=None, archive_root="archive")

    assert oracle.resolve_gt_json(args) == oracle.DEFAULT_VAL_GT_JSON
