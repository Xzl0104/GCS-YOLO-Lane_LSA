from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

from gcs_tools.label_utils import fixed_y_anchors
from tools import analyze_gcs_errors
from tools import analyze_gcs_oracle
from tools import analyze_gcs_results_csv
from tools import build_gcs_hard_samples_from_eval as hard_samples
from tools import build_tusimple_official_folder_balanced_subset as folder_subset_builder
from tools import build_tusimple_official_subset as subset_builder
from tools import check_model
from tools import check_tusimple_fixed_y_label_oracle as oracle
from tools import diagnose_gcs_gt5 as gt5_diag
from tools import eval_gcs
from tools import rebuild_tusimple_fixed_y_k56_from_reference_split as builder
from tools import sweep_gcs_conf
from tools import sweep_tusimple_official
from tools import train_gcs
from ultralytics.cfg import TASK2DATA, TASK2MODEL
from ultralytics.data.dataset_gcs import GCSLaneDataset
from ultralytics.models.yolo.gcs_lane.train import GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT
from ultralytics.utils.gcs_loss import GCSLoss


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


def test_check_model_enforces_named_candidate_output_flags():
    check_model.enforce_candidate_yaml_contract(
        "gcs-yolo-lane-s-q12-k56-survival", SimpleNamespace(survival_head_enabled=True, decoder_aux_outputs=False)
    )
    check_model.enforce_candidate_yaml_contract(
        "gcs-yolo-lane-s-q12-k56-decoder-aux", SimpleNamespace(survival_head_enabled=False, decoder_aux_outputs=True)
    )
    with pytest.raises(RuntimeError, match="survival_head_enabled"):
        check_model.enforce_candidate_yaml_contract(
            "gcs-yolo-lane-s-q12-k56-survival",
            SimpleNamespace(survival_head_enabled=False, decoder_aux_outputs=False),
        )
    with pytest.raises(RuntimeError, match="decoder_aux_outputs"):
        check_model.enforce_candidate_yaml_contract(
            "gcs-yolo-lane-s-q12-k56-decoder-aux",
            SimpleNamespace(survival_head_enabled=False, decoder_aux_outputs=False),
        )


def test_train_cli_rejects_named_candidate_without_matching_loss_gain():
    with pytest.raises(SystemExit, match="--gcs-survival > 0"):
        train_gcs.validate_named_candidate_training_args(
            "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-survival.yaml",
            SimpleNamespace(gcs_survival=0.0, gcs_decoder_aux=0.0),
        )
    with pytest.raises(SystemExit, match="--gcs-decoder-aux > 0"):
        train_gcs.validate_named_candidate_training_args(
            "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-decoder-aux.yaml",
            SimpleNamespace(gcs_survival=0.0, gcs_decoder_aux=0.0),
        )
    train_gcs.validate_named_candidate_training_args(
        "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-survival.yaml",
        SimpleNamespace(gcs_survival=0.2, gcs_decoder_aux=0.0),
    )
    train_gcs.validate_named_candidate_training_args(
        "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-decoder-aux.yaml",
        SimpleNamespace(gcs_survival=0.0, gcs_decoder_aux=0.1),
    )


def test_gcs_mainline_defaults_point_to_k56():
    assert train_gcs.DEFAULT_MODEL.name == "gcs-yolo-lane-s-q12-k56.yaml"
    defaults = train_gcs.dataset_defaults("tusimple")
    assert defaults["data"].name == "tusimple_gcs_fixed_y_k56_960x544.yaml"
    assert defaults["train_images"].as_posix().endswith("datasets/tusimple_fixed_y_k56_960x544/images/train")
    assert str(TASK2DATA["gcs_lane"]).replace("\\", "/").endswith("data/tusimple_gcs_fixed_y_k56_960x544.yaml")
    assert str(TASK2MODEL["gcs_lane"]).replace("\\", "/").endswith(
        "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml"
    )


def test_generic_tusimple_aliases_are_k56():
    for filename in (
        "tusimple_gcs_fixed_y_960x544.yaml",
        "tusimple_gcs.yaml",
        "tusimple_gcs_stratified_960x544.yaml",
    ):
        data = yaml.safe_load((ROOT / "data" / filename).read_text(encoding="utf-8"))
        assert "tusimple_fixed_y_k56_960x544" in data["train"]
        assert data["point_mode"] == "fixed_y"
        assert data["num_points"] == 56
        assert math.isclose(float(data["fixed_y"][1]), 160.0 / 720.0)

    for filename in ("tusimple_yolo.yaml", "tusimple_yolo_stratified_960x544.yaml"):
        data = yaml.safe_load((ROOT / "data" / filename).read_text(encoding="utf-8"))
        assert "tusimple_fixed_y_k56_960x544" in data["train"]


def test_test_protection_defaults_do_not_target_test_split(monkeypatch):
    assert analyze_gcs_errors.DEFAULT_SOURCE.as_posix().endswith("datasets/tusimple_fixed_y_k56_960x544/images/val")
    assert analyze_gcs_errors.DEFAULT_LABELS.as_posix().endswith("datasets/tusimple_fixed_y_k56_960x544/labels_gcs/val")
    assert analyze_gcs_oracle.DEFAULT_SOURCE.as_posix().endswith("datasets/tusimple_fixed_y_k56_960x544/images/val")
    assert analyze_gcs_oracle.DEFAULT_LABELS.as_posix().endswith("datasets/tusimple_fixed_y_k56_960x544/labels_gcs/val")

    test_source = ROOT / "datasets" / "tusimple_fixed_y_k56_960x544" / "images" / "test"
    with pytest.raises(ValueError, match="test split"):
        analyze_gcs_errors.reject_test_diagnostics(test_source, field="--source", allow=False)
    analyze_gcs_errors.reject_test_diagnostics(test_source, field="--source", allow=True)

    with pytest.raises(ValueError, match="test evaluation"):
        sweep_gcs_conf.reject_test_source(test_source, field="--source")
    monkeypatch.setattr("sys.argv", ["sweep_gcs_conf.py", "--split", "test"])
    with pytest.raises(SystemExit):
        sweep_gcs_conf.parse_args()


def test_subset_builders_reject_test_reference_by_default():
    test_json = ROOT / "archive" / "TUSimple" / "test_label.json"
    for module in (subset_builder, folder_subset_builder):
        with pytest.raises(ValueError, match="official test labels"):
            module.reject_test_reference(test_json, test_json, allow=False)
        module.reject_test_reference(test_json, test_json, allow=True)


def test_k56_experimental_model_variants_keep_q12_k56_contract():
    variants = {
        "gcs-yolo-lane-s-q12.yaml": {"decoder_layers": 3, "bifpn_channels": 128},
        "gcs-yolo-lane-s-q12-no-lsem.yaml": {"decoder_layers": 3, "bifpn_channels": 128},
        "gcs-yolo-lane-s-q12-proj-no-bifpn.yaml": {
            "decoder_layers": 3,
            "bifpn_channels": 128,
            "projection_only": True,
        },
        "gcs-yolo-lane-s-q12-k56-dec4.yaml": {"decoder_layers": 4, "bifpn_channels": 128},
        "gcs-yolo-lane-s-q12-k56-bifpn192.yaml": {"decoder_layers": 3, "bifpn_channels": 192},
        "gcs-yolo-lane-s-q12-k56-bifpn256.yaml": {"decoder_layers": 3, "bifpn_channels": 256},
        "gcs-yolo-lane-s-q12-k56-cqcalib.yaml": {
            "decoder_layers": 3,
            "bifpn_channels": 128,
            "count_quality_calib_dim": 64,
        },
        "gcs-yolo-lane-s-q12-k56-dec4-cqcalib.yaml": {
            "decoder_layers": 4,
            "bifpn_channels": 128,
            "count_quality_calib_dim": 64,
        },
        "gcs-yolo-lane-s-q12-k56-strip-p23.yaml": {
            "decoder_layers": 3,
            "bifpn_channels": 128,
            "strip_levels": "p2,p3",
        },
        "gcs-yolo-lane-s-q12-k56-dec4-bifpn256-cqcalib-strip-p23.yaml": {
            "decoder_layers": 4,
            "bifpn_channels": 256,
            "count_quality_calib_dim": 64,
            "strip_levels": "p2,p3",
        },
        "gcs-yolo-lane-s-q12-k56-survival.yaml": {
            "decoder_layers": 3,
            "bifpn_channels": 128,
            "survival_head": True,
            "decoder_aux_loss": False,
        },
        "gcs-yolo-lane-s-q12-k56-decoder-aux.yaml": {
            "decoder_layers": 3,
            "bifpn_channels": 128,
            "survival_head": False,
            "decoder_aux_loss": True,
        },
    }

    for filename, expected in variants.items():
        model = yaml.safe_load(
            (ROOT / "ultralytics" / "cfg" / "models" / "gcs" / filename).read_text(encoding="utf-8")
        )
        bifpn_layers = [layer for layer in model["head"] if layer[2] == "LaneBiFPN"]
        projection_layers = [layer for layer in model["head"] if layer[2] == "LaneFeatureProjection"]
        gcs_layers = [layer for layer in model["head"] if layer[2] == "GCSLaneHead"]
        strip_layers = [layer for layer in model["head"] if layer[2] == "LaneStripPyramidAttention"]
        if expected.get("projection_only"):
            assert bifpn_layers == []
            assert len(projection_layers) == 1
            assert projection_layers[0][3][0] == expected["bifpn_channels"]
        else:
            assert len(bifpn_layers) == 1
            assert bifpn_layers[0][3][0] == expected["bifpn_channels"]
        assert len(gcs_layers) == 1

        head_args = gcs_layers[0][3]
        assert head_args[:4] == [12, 56, expected["decoder_layers"], 8]
        assert head_args[4] == "fixed_y"
        assert math.isclose(float(head_args[5]), 710.0 / 720.0)
        assert math.isclose(float(head_args[6]), 160.0 / 720.0)
        expected_calib = expected.get("count_quality_calib_dim", 0)
        if len(head_args) > 7:
            assert head_args[7] == expected_calib
        else:
            assert expected_calib == 0
        expected_survival = expected.get("survival_head", False)
        expected_decoder_aux = expected.get("decoder_aux_loss", False)
        if len(head_args) > 8:
            assert bool(head_args[8]) is expected_survival
            assert bool(head_args[9]) is expected_decoder_aux
        else:
            assert expected_survival is False
            assert expected_decoder_aux is False
        if "strip_levels" in expected:
            assert len(strip_layers) == 1
            assert strip_layers[0][3][0] == expected["strip_levels"]
        else:
            assert strip_layers == []


def test_k56_viscountsum_hardsample_preset_is_train_failure_scope():
    transitions = hard_samples.parse_transitions(
        hard_samples.PRESET_TRANSITIONS["k56_viscountsum_hardsamples"]
    )
    assert transitions == {(4, 5), (5, 2), (5, 3), (5, 4)}
    assert hard_samples.parse_list("train") == ["train"]
    assert hard_samples.is_test_summary({"config": {"split": "test"}}, ROOT / "runs" / "summary.json")
    assert not hard_samples.is_test_summary({"config": {"split": "train"}}, ROOT / "runs" / "summary.json")

    args = SimpleNamespace(
        preset="k56_viscountsum_hardsamples",
        allow_non_train_preset=False,
        require_target_match=True,
        dataset_root="datasets/tusimple_fixed_y_k56_960x544",
    )
    with pytest.raises(SystemExit, match="requires.*split='train'"):
        hard_samples.validate_preset_scope(args, {"config": {"split": "val"}}, ROOT / "runs" / "summary.json")

    no_match = SimpleNamespace(**{**vars(args), "require_target_match": False})
    with pytest.raises(SystemExit, match="requires --require-target-match"):
        hard_samples.validate_preset_scope(no_match, {"config": {"split": "train"}}, ROOT / "runs" / "summary.json")

    bad_target = SimpleNamespace(**{**vars(args), "target_splits": "val"})
    with pytest.raises(SystemExit, match="requires --target-splits train"):
        hard_samples.validate_preset_scope(bad_target, {"config": {"split": "train"}}, ROOT / "runs" / "summary.json")

    allow_analysis = SimpleNamespace(**{**vars(args), "allow_non_train_preset": True, "require_target_match": False})
    hard_samples.validate_preset_scope(allow_analysis, {"config": {"split": "val"}}, ROOT / "runs" / "summary.json")

    with pytest.raises(SystemExit, match="every exported failure"):
        hard_samples.validate_preset_target_audit(
            args,
            [{"raw_file": "clips/train/missing.jpg"}],
            {"unmatched_unique_samples": 1},
        )


def test_k56_fifth_gate_hardsample_preset_collects_only_requested_failures():
    transitions = hard_samples.parse_transitions(hard_samples.PRESET_TRANSITIONS["k56_fifth_gate_hardsamples"])
    preset = "k56_fifth_gate_hardsamples"

    assert hard_samples.record_matches_preset(
        {"gt_lanes": 4, "pred_lanes": 5}, preset=preset, transitions=transitions
    ) == (True, "4->5")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 5, "pred_lanes": 4}, preset=preset, transitions=transitions
    ) == (True, "5->4")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 5, "pred_lanes": None, "final_pred_lanes": 4},
        preset=preset,
        transitions=transitions,
    ) == (True, "5->4")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 5, "pred_lanes": 5, "final_pred_lanes": 4},
        preset=preset,
        transitions=transitions,
    ) == (True, "5->4")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 4, "pred_lanes": 5, "final_pred_lanes": 4},
        preset=preset,
        transitions=transitions,
    ) == (False, "")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 5, "pred_lanes": 5, "decode_count_head_k": 4},
        preset=preset,
        transitions=transitions,
    ) == (True, "5->count_head_klt5")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 5, "decode_count_head_k": 4},
        preset=preset,
        transitions=transitions,
    ) == (True, "5->count_head_klt5")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 5, "final_pred_lanes": 5, "count_head_policy_count": 4},
        preset=preset,
        transitions=transitions,
    ) == (True, "5->count_head_klt5")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 5, "pred_lanes": 5, "decode_count_head_k": 5},
        preset=preset,
        transitions=transitions,
    ) == (False, "")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 5, "pred_lanes": 5, "effective_policy_count": 4},
        preset=preset,
        transitions=transitions,
    ) == (False, "")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 5, "pred_lanes": 5, "count_head_meta": {"effective_policy_count": 4}},
        preset=preset,
        transitions=transitions,
    ) == (False, "")
    assert hard_samples.record_matches_preset(
        {"gt_lanes": 3, "pred_lanes": 5}, preset=preset, transitions=transitions
    ) == (False, "")


def test_k56_fifth_gate_hardsample_builder_writes_train_sidecar(monkeypatch, tmp_path):
    summary_path = tmp_path / "train_summary.json"
    output = tmp_path / "out" / "fifth_gate.txt"
    dataset_root = tmp_path / "dataset"
    label_dir = dataset_root / "labels_gcs" / "train"
    label_dir.mkdir(parents=True)
    exported = [
        "clips/train/gt4_false/20.jpg",
        "clips/train/gt5_miss/20.jpg",
        "clips/train/gt5_count_under/20.jpg",
        "clips/train/gt5_count_under_no_pred/20.jpg",
    ]
    for idx, raw_file in enumerate(exported):
        np.savez(label_dir / f"sample{idx}.npz", raw_file=np.array(raw_file))
    summary_path.write_text(
        json.dumps(
            {
                "config": {"split": "train"},
                "records": [
                    {"raw_file": exported[0], "gt_lanes": 4, "pred_lanes": 5},
                    {"raw_file": exported[1], "gt_lanes": 5, "pred_lanes": 4},
                    {"raw_file": exported[2], "gt_lanes": 5, "pred_lanes": 5, "decode_count_head_k": 4},
                    {"raw_file": exported[3], "gt_lanes": 5, "decode_count_head_k": 4},
                    {"raw_file": "clips/train/clean_gt5/20.jpg", "gt_lanes": 5, "pred_lanes": 5, "decode_count_head_k": 5},
                    {"raw_file": "clips/train/gt3_false/20.jpg", "gt_lanes": 3, "pred_lanes": 5},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_gcs_hard_samples_from_eval.py",
            "--eval-summary",
            str(summary_path),
            "--preset",
            "k56_fifth_gate_hardsamples",
            "--dataset-root",
            str(dataset_root),
            "--target-splits",
            "train",
            "--require-target-match",
            "--output",
            str(output),
        ],
    )

    hard_samples.main()

    assert output.read_text(encoding="utf-8").splitlines() == exported
    sidecar = json.loads(output.with_suffix(output.suffix + ".summary.json").read_text(encoding="utf-8"))
    assert sidecar["preset"] == "k56_fifth_gate_hardsamples"
    assert sidecar["source_split"] == "train"
    assert sidecar["target_splits"] == ["train"]
    assert sidecar["require_target_match"] is True
    assert sidecar["analysis_only"] is False
    assert sidecar["target_match_audit"]["raw_file_only"] is True
    assert sidecar["target_match_audit"]["unmatched_unique_samples"] == 0
    assert sidecar["transition_counts"] == {"4->5": 1, "5->4": 1, "5->count_head_klt5": 2}


def test_k56_hardsample_preset_does_not_write_failed_target_audit(monkeypatch, tmp_path):
    summary_path = tmp_path / "train_summary.json"
    output = tmp_path / "out" / "hard.txt"
    dataset_root = tmp_path / "dataset"
    label_dir = dataset_root / "labels_gcs" / "train"
    image_dir = dataset_root / "images" / "train"
    label_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    np.savez(label_dir / "present.npz", raw_file=np.array("clips/train/present.jpg"))
    summary_path.write_text(
        json.dumps(
            {
                "config": {"split": "train"},
                "records": [{"raw_file": "clips/train/missing.jpg", "gt_lanes": 4, "pred_lanes": 5}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_gcs_hard_samples_from_eval.py",
            "--eval-summary",
            str(summary_path),
            "--preset",
            "k56_viscountsum_hardsamples",
            "--dataset-root",
            str(dataset_root),
            "--target-splits",
            "train",
            "--require-target-match",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit, match="Exported failures do not match|every exported failure"):
        hard_samples.main()

    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".summary.json").exists()


def test_k56_hardsample_raw_file_manifest_hits_dataset_batch(monkeypatch, tmp_path):
    summary_path = tmp_path / "train_summary.json"
    output = tmp_path / "out" / "hard.txt"
    dataset_root = tmp_path / "dataset"
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels_gcs" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    cv2.imwrite(str(image_dir / "present.jpg"), np.zeros((8, 16, 3), dtype=np.uint8))
    y = np.linspace(0.9, 0.2, 6, dtype=np.float32)
    lanes = np.stack(
        [
            np.stack((np.full_like(y, x), y), axis=-1)
            for x in (0.2, 0.4, 0.6, 0.8)
        ],
        axis=0,
    )
    lane_valid = np.ones((4, 6), dtype=np.float32)
    raw_file = "clips/train/present/20.jpg"
    np.savez(
        label_dir / "present.npz",
        lanes=lanes,
        lane_valid=lane_valid,
        num_lanes=np.array(4),
        point_mode=np.array("free"),
        raw_file=np.array(raw_file),
    )
    summary_path.write_text(
        json.dumps(
            {
                "config": {"split": "train"},
                "records": [{"raw_file": raw_file, "gt_lanes": 4, "pred_lanes": 5}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_gcs_hard_samples_from_eval.py",
            "--eval-summary",
            str(summary_path),
            "--preset",
            "k56_viscountsum_hardsamples",
            "--dataset-root",
            str(dataset_root),
            "--target-splits",
            "train",
            "--require-target-match",
            "--output",
            str(output),
        ],
    )

    hard_samples.main()

    assert output.read_text(encoding="utf-8").strip() == raw_file
    sidecar = json.loads(output.with_suffix(output.suffix + ".summary.json").read_text(encoding="utf-8"))
    assert sidecar["builder"] == "build_gcs_hard_samples_from_eval.py"
    assert sidecar["preset"] == "k56_viscountsum_hardsamples"
    assert sidecar["source_split"] == "train"
    assert sidecar["target_splits"] == ["train"]
    assert sidecar["require_target_match"] is True
    assert sidecar["analysis_only"] is False
    assert sidecar["test_summary_allowed"] is False
    assert sidecar["target_match_audit"]["raw_file_only"] is True
    assert sidecar["target_match_audit"]["unmatched_unique_samples"] == 0
    dataset = GCSLaneDataset(img_path=image_dir, label_dir=label_dir, imgsz=[8, 16], augment=False)
    batch = GCSLaneDataset.collate_fn([dataset[0]])
    criterion = GCSLoss(
        model={
            "gcs_point_mode": "free",
            "gcs_imgsz": [8, 16],
            "gcs_line_iou": 0.0,
            "gcs_quality": 0.0,
            "gcs_exist_quality_lane_iou_alpha": 0.0,
            "gcs_hard_loss_file": str(output),
        }
    )

    mask = criterion.hard_loss_mask(batch, 1, torch.device("cpu"), gt_valid=batch["lane_valid"])

    assert batch["raw_file"] == [raw_file]
    assert mask.tolist() == [True]


def test_eval_gcs_reads_raw_file_from_npz_label(tmp_path):
    label_path = tmp_path / "sample.npz"
    np.savez(label_path, raw_file=np.array(b"clips/train/present/20.jpg"))

    assert eval_gcs.label_raw_file(label_path) == "clips/train/present/20.jpg"


def test_k56_hardsample_builder_recovers_raw_file_from_eval_label_path(monkeypatch, tmp_path):
    summary_path = tmp_path / "train_summary.json"
    output = tmp_path / "out" / "hard.txt"
    dataset_root = tmp_path / "dataset"
    label_dir = dataset_root / "labels_gcs" / "train"
    image_dir = dataset_root / "images" / "train"
    label_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    raw_file = "clips/train/present/20.jpg"
    label_path = label_dir / "present.npz"
    np.savez(label_path, raw_file=np.array(raw_file))
    summary_path.write_text(
        json.dumps(
            {
                "config": {"split": "train"},
                "records": [
                    {
                        "image": str(image_dir / "present.jpg"),
                        "label": str(label_path),
                        "gt_lanes": 4,
                        "pred_lanes": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_gcs_hard_samples_from_eval.py",
            "--eval-summary",
            str(summary_path),
            "--preset",
            "k56_viscountsum_hardsamples",
            "--dataset-root",
            str(dataset_root),
            "--target-splits",
            "train",
            "--require-target-match",
            "--output",
            str(output),
        ],
    )

    hard_samples.main()

    assert output.read_text(encoding="utf-8").strip() == raw_file
    sidecar = json.loads(output.with_suffix(output.suffix + ".summary.json").read_text(encoding="utf-8"))
    assert sidecar["target_match_audit"]["raw_file_only"] is True
    assert sidecar["target_match_audit"]["unmatched_unique_samples"] == 0


def test_k56_hardsample_builder_refuses_overwrite(monkeypatch, tmp_path):
    summary_path = tmp_path / "train_summary.json"
    output = tmp_path / "hard.txt"
    dataset_root = tmp_path / "dataset"
    label_dir = dataset_root / "labels_gcs" / "train"
    label_dir.mkdir(parents=True)
    raw_file = "clips/train/present/20.jpg"
    np.savez(label_dir / "present.npz", raw_file=np.array(raw_file))
    summary_path.write_text(
        json.dumps(
            {
                "config": {"split": "train"},
                "records": [{"raw_file": raw_file, "gt_lanes": 4, "pred_lanes": 5}],
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "build_gcs_hard_samples_from_eval.py",
        "--eval-summary",
        str(summary_path),
        "--preset",
        "k56_viscountsum_hardsamples",
        "--dataset-root",
        str(dataset_root),
        "--target-splits",
        "train",
        "--require-target-match",
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", argv)
    hard_samples.main()
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        hard_samples.main()


def test_k56_hardsample_preset_requires_raw_file_for_target_audit(monkeypatch, tmp_path):
    summary_path = tmp_path / "train_summary.json"
    output = tmp_path / "hard.txt"
    dataset_root = tmp_path / "dataset"
    label_dir = dataset_root / "labels_gcs" / "train"
    image_dir = dataset_root / "images" / "train"
    label_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    np.savez(label_dir / "present.npz", raw_file=np.array("clips/train/present/20.jpg"))
    summary_path.write_text(
        json.dumps(
            {
                "config": {"split": "train"},
                "records": [{"image": str(image_dir / "present.jpg"), "gt_lanes": 4, "pred_lanes": 5}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_gcs_hard_samples_from_eval.py",
            "--eval-summary",
            str(summary_path),
            "--preset",
            "k56_viscountsum_hardsamples",
            "--dataset-root",
            str(dataset_root),
            "--target-splits",
            "train",
            "--require-target-match",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit, match="path-like raw_file"):
        hard_samples.main()

    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".summary.json").exists()


def test_train_rejects_analysis_only_hard_manifest_sidecar(tmp_path):
    manifest = tmp_path / "hard.txt"
    manifest.write_text("clips/train/present/20.jpg\n", encoding="utf-8")
    manifest.with_suffix(manifest.suffix + ".summary.json").write_text(
        json.dumps(
            {
                "builder": "build_gcs_hard_samples_from_eval.py",
                "preset": "k56_viscountsum_hardsamples",
                "source_split": "val",
                "target_splits": ["train"],
                "require_target_match": True,
                "analysis_only": True,
                "test_summary_allowed": False,
                "target_match_audit": {"raw_file_only": True, "unmatched_unique_samples": 0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="analysis_only"):
        train_gcs.validate_hard_manifest_sidecar(manifest, flag="--gcs-hard-loss-file")


def test_train_requires_hard_loss_file_for_hard_weighted_visible_count_sum():
    args = SimpleNamespace(
        gcs_visible_count_sum=0.2,
        gcs_visible_count_sum_hard_weight=2.0,
        gcs_hard_loss_file="",
        gcs_hard_sample_file="",
    )
    with pytest.raises(SystemExit, match="requires --gcs-hard-loss-file"):
        train_gcs.validate_training_hard_manifest_args(args)


def test_train_requires_builder_sidecar_for_hard_weighted_visible_count_sum(tmp_path):
    manifest = tmp_path / "hard.txt"
    manifest.write_text("clips/train/present/20.jpg\n", encoding="utf-8")
    args = SimpleNamespace(
        gcs_visible_count_sum=0.2,
        gcs_visible_count_sum_hard_weight=2.0,
        gcs_hard_loss_file=str(manifest),
        gcs_hard_sample_file="",
    )

    with pytest.raises(SystemExit, match="requires a build_gcs_hard_samples_from_eval.py sidecar"):
        train_gcs.validate_training_hard_manifest_args(args)


def test_train_rejects_wrong_builder_preset_for_hard_weighted_visible_count_sum(tmp_path):
    manifest = tmp_path / "hard.txt"
    manifest.write_text("clips/train/present/20.jpg\n", encoding="utf-8")
    manifest.with_suffix(manifest.suffix + ".summary.json").write_text(
        json.dumps(
            {
                "builder": "build_gcs_hard_samples_from_eval.py",
                "preset": "",
                "source_split": "train",
                "target_splits": ["train"],
                "require_target_match": True,
                "analysis_only": False,
                "test_summary_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        gcs_visible_count_sum=0.2,
        gcs_visible_count_sum_hard_weight=2.0,
        gcs_hard_loss_file=str(manifest),
        gcs_hard_sample_file="",
    )

    with pytest.raises(SystemExit, match="sidecar preset must be k56_viscountsum_hardsamples"):
        train_gcs.validate_training_hard_manifest_args(args)


def test_train_rejects_zero_matched_builder_sidecar_for_hard_weighted_visible_count_sum(tmp_path):
    manifest = tmp_path / "hard.txt"
    manifest.write_text("", encoding="utf-8")
    manifest.with_suffix(manifest.suffix + ".summary.json").write_text(
        json.dumps(
            {
                "builder": "build_gcs_hard_samples_from_eval.py",
                "preset": "k56_viscountsum_hardsamples",
                "source_split": "train",
                "target_splits": ["train"],
                "require_target_match": True,
                "analysis_only": False,
                "test_summary_allowed": False,
                "target_match_audit": {
                    "raw_file_only": True,
                    "matched_unique_samples": 0,
                    "unmatched_unique_samples": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        gcs_visible_count_sum=0.2,
        gcs_visible_count_sum_hard_weight=2.0,
        gcs_hard_loss_file=str(manifest),
        gcs_hard_sample_file="",
    )

    with pytest.raises(SystemExit, match="no matched train hard samples"):
        train_gcs.validate_training_hard_manifest_args(args)


def test_train_treats_zero_hard_weight_as_weighted_visible_count_sum(tmp_path):
    manifest = tmp_path / "hard.txt"
    manifest.write_text("clips/train/present/20.jpg\n", encoding="utf-8")
    args = SimpleNamespace(
        gcs_visible_count_sum=0.2,
        gcs_visible_count_sum_hard_weight=0.0,
        gcs_hard_loss_file=str(manifest),
        gcs_hard_sample_file="",
    )

    with pytest.raises(SystemExit, match="requires a build_gcs_hard_samples_from_eval.py sidecar"):
        train_gcs.validate_training_hard_manifest_args(args)


def test_train_requires_fifth_gate_manifest_for_count_conditioned_survival(tmp_path):
    manifest = tmp_path / "hard.txt"
    manifest.write_text("clips/train/present/20.jpg\n", encoding="utf-8")
    args = SimpleNamespace(
        gcs_visible_count_sum=0.0,
        gcs_visible_count_sum_hard_weight=1.0,
        gcs_survival=0.2,
        gcs_survival_target_mode="count_conditioned_fifth",
        gcs_count_boundary_hard_margin_gain=0.0,
        gcs_hard_loss_file="",
        gcs_hard_sample_file="",
    )
    with pytest.raises(SystemExit, match="k56_fifth_gate_hardsamples"):
        train_gcs.validate_training_hard_manifest_args(args)

    args.gcs_hard_loss_file = str(manifest)
    with pytest.raises(SystemExit, match="requires a build_gcs_hard_samples_from_eval.py sidecar"):
        train_gcs.validate_training_hard_manifest_args(args)

    manifest.with_suffix(manifest.suffix + ".summary.json").write_text(
        json.dumps(
            {
                "builder": "build_gcs_hard_samples_from_eval.py",
                "preset": "k56_viscountsum_hardsamples",
                "source_split": "train",
                "target_splits": ["train"],
                "require_target_match": True,
                "analysis_only": False,
                "test_summary_allowed": False,
                "target_match_audit": {
                    "raw_file_only": True,
                    "matched_unique_samples": 1,
                    "unmatched_unique_samples": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="sidecar preset must be k56_fifth_gate_hardsamples"):
        train_gcs.validate_training_hard_manifest_args(args)

    manifest.with_suffix(manifest.suffix + ".summary.json").write_text(
        json.dumps(
            {
                "builder": "build_gcs_hard_samples_from_eval.py",
                "preset": "k56_fifth_gate_hardsamples",
                "source_split": "train",
                "target_splits": ["train"],
                "require_target_match": True,
                "analysis_only": False,
                "test_summary_allowed": False,
                "target_match_audit": {
                    "raw_file_only": True,
                    "matched_unique_samples": 1,
                    "unmatched_unique_samples": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="Disable unrelated hard-edge weighting"):
        train_gcs.validate_training_hard_manifest_args(args)

    args.gcs_hard_edge_loss_weight_by_count = "none"
    train_gcs.validate_training_hard_manifest_args(args)


def test_train_requires_fifth_gate_manifest_for_count_boundary_hard_margin():
    args = SimpleNamespace(
        gcs_visible_count_sum=0.0,
        gcs_visible_count_sum_hard_weight=1.0,
        gcs_survival=0.0,
        gcs_survival_target_mode="matched",
        gcs_count_boundary_hard_margin_gain=0.2,
        gcs_hard_loss_file="",
        gcs_hard_sample_file="",
    )
    with pytest.raises(SystemExit, match="k56_fifth_gate_hardsamples"):
        train_gcs.validate_training_hard_manifest_args(args)


def test_train_rejects_default_hard_edge_weighting_for_fifth_gate_manifest(tmp_path):
    manifest = tmp_path / "hard.txt"
    manifest.write_text("clips/train/present/20.jpg\n", encoding="utf-8")
    manifest.with_suffix(manifest.suffix + ".summary.json").write_text(
        json.dumps(
            {
                "builder": "build_gcs_hard_samples_from_eval.py",
                "preset": "k56_fifth_gate_hardsamples",
                "source_split": "train",
                "target_splits": ["train"],
                "require_target_match": True,
                "analysis_only": False,
                "test_summary_allowed": False,
                "target_match_audit": {
                    "raw_file_only": True,
                    "matched_unique_samples": 1,
                    "unmatched_unique_samples": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        gcs_visible_count_sum=0.0,
        gcs_visible_count_sum_hard_weight=1.0,
        gcs_survival=0.2,
        gcs_survival_target_mode="count_conditioned_fifth",
        gcs_count_boundary_hard_margin_gain=0.0,
        gcs_hard_loss_file=str(manifest),
        gcs_hard_sample_file="",
        gcs_hard_edge_loss_weight_by_count="4:1.15,5:1.6",
        gcs_hard_edge_loss_terms="exist,point,point_valid,line_iou",
    )

    with pytest.raises(SystemExit, match="Disable unrelated hard-edge weighting"):
        train_gcs.validate_training_hard_manifest_args(args)

    args.gcs_hard_edge_loss_terms = "none"
    train_gcs.validate_training_hard_manifest_args(args)


def test_official_search_tools_reject_explicit_test_gt_json(monkeypatch):
    test_json = ROOT / "archive" / "TUSimple" / "test_label.json"
    with pytest.raises(ValueError, match="test GT json"):
        sweep_tusimple_official.validate_official_sweep_gt_json(test_json)

    monkeypatch.setattr("sys.argv", ["diagnose_gcs_gt5.py", "--split", "val", "--gt-json", str(test_json)])
    with pytest.raises(SystemExit, match="test GT json"):
        gt5_diag.parse_args()


def test_results_csv_summary_lists_current_gcs_loss_contract(capsys):
    row = {"epoch": 1.0}
    for prefix in ("train", "val"):
        for name in GCSLoss.loss_names:
            row[f"{prefix}/{name}"] = 0.1

    analyze_gcs_results_csv.print_metric_summary([row])

    out = capsys.readouterr().out
    for name in GCSLoss.loss_names:
        assert f"train/{name}:" in out
        assert f"val/{name}:" in out


def test_gt5_diagnosis_splits_survival_and_quality_gate_failures():
    args = SimpleNamespace(quality_rescue_quality_thr=0.55, exist_low_thr=0.1, s5_low_thr=0.1)
    survival_meta = {
        "count_head_policy_count": 5,
        "candidate_pool_shortfall": 0,
        "top5_candidate_gate_score_before_nms": 0.1,
        "top5_candidate_gate_source_before_nms": "survival",
        "top5_suppressed_by_nms": False,
    }
    quality_meta = {
        **survival_meta,
        "top5_candidate_gate_source_before_nms": "quality",
    }
    rank5 = {"valid_points": 6, "exist_score": 0.9, "rank_score": 0.8}

    assert gt5_diag.top5_gate_drop_reason(survival_meta, args) == "survival_too_low"
    assert gt5_diag.top5_gate_drop_reason(quality_meta, args) == "quality_too_low"
    assert gt5_diag.gt5_output_drop_reason(4, rank5, "postprocess", args, 5, survival_meta) == "survival_too_low"
    assert gt5_diag.gt5_output_drop_reason(4, rank5, "postprocess", args, 5, quality_meta) == "quality_too_low"


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


def test_k56_builder_uses_official_val_json_as_split_manifest(tmp_path):
    val_json = tmp_path / "val.json"
    val_json.write_text(
        '{"raw_file":"clips/0601/0001/20.jpg","lanes":[],"h_samples":[]}\n'
        '{"raw_file":"clips/0601/0002/20.jpg","lanes":[],"h_samples":[]}\n',
        encoding="utf-8",
    )

    mapping = builder.val_gt_raw_file_split(val_json)

    assert mapping == {
        "clips/0601/0001/20.jpg": "val",
        "clips/0601/0002/20.jpg": "val",
    }


def test_k56_label_oracle_requires_explicit_test_gt():
    args = SimpleNamespace(label_split="test", gt_json=None, archive_root="archive")

    with pytest.raises(SystemExit, match="explicit --gt-json"):
        oracle.resolve_gt_json(args)


def test_k56_label_oracle_defaults_to_official_val_gt():
    args = SimpleNamespace(label_split="val", gt_json=None, archive_root="archive")

    assert oracle.resolve_gt_json(args) == oracle.DEFAULT_VAL_GT_JSON


def test_k56_gt5_edge_segment_support_targets_edge_lanes_only():
    assert GCS_MAINLINE_POINT_VALID_GT5_EDGE_SEGMENT == 0.0

    anchors = torch.tensor(
        fixed_y_anchors(num_points=56, y_start=710.0 / 720.0, y_end=160.0 / 720.0),
        dtype=torch.float32,
    )
    lane_x = torch.tensor([0.1, 0.25, 0.4, 0.55, 0.7], dtype=torch.float32)
    lanes = torch.zeros(5, 56, 2, dtype=torch.float32)
    lanes[..., 0] = lane_x[:, None]
    lanes[..., 1] = anchors[None, :]
    valid = torch.zeros(5, 56, dtype=torch.float32)
    visible_start, visible_end = 18, 26
    valid[:, visible_start:visible_end] = 1.0
    pred_points = lanes.unsqueeze(0).clone()
    indices = [(torch.arange(5), torch.arange(5))]
    common = {
        "gcs_point_mode": "fixed_y",
        "gcs_imgsz": [544, 960],
        "gcs_gt5_edge_loss_weight": 1.0,
        "gcs_candidate_gt5_edge_weight": 1.0,
        "gcs_point_valid_gt5_pos_weight": 1.0,
        "gcs_point_valid_unmatched_weight": 1.0,
        "gcs_point_valid_gt5_edge_continuity": 0.0,
    }
    base = GCSLoss(model={**common, "gcs_point_valid_gt5_edge_segment": 0.0})
    segment = GCSLoss(
        model={
            **common,
            "gcs_point_valid_gt5_edge_segment": 0.5,
            "gcs_point_valid_gt5_edge_segment_thr": 0.8,
            "gcs_point_valid_gt5_edge_segment_min_points": 5,
        }
    )

    edge_logits = torch.full((1, 5, 56), 4.0)
    edge_logits[0, 0, visible_start:visible_end] = -3.0
    edge_logits[0, 4, visible_start:visible_end] = -3.0
    edge_logits = edge_logits.requires_grad_()
    edge_base_loss = base.point_valid_loss(edge_logits, pred_points, [valid], indices, gt_points=[lanes])
    edge_segment_loss = segment.point_valid_loss(edge_logits, pred_points, [valid], indices, gt_points=[lanes])
    assert edge_segment_loss > edge_base_loss
    edge_segment_loss.backward()
    assert edge_logits.grad is not None

    middle_logits = torch.full((1, 5, 56), 4.0)
    middle_logits[0, 2, visible_start:visible_end] = -3.0
    middle_base_loss = base.point_valid_loss(middle_logits, pred_points, [valid], indices, gt_points=[lanes])
    middle_segment_loss = segment.point_valid_loss(middle_logits, pred_points, [valid], indices, gt_points=[lanes])
    assert torch.isclose(middle_segment_loss, middle_base_loss)

    lanes4 = torch.zeros(4, 56, 2, dtype=torch.float32)
    lanes4[..., 0] = torch.tensor([0.1, 0.25, 0.55, 0.7], dtype=torch.float32)[:, None]
    lanes4[..., 1] = anchors[None, :]
    valid4 = torch.zeros(4, 56, dtype=torch.float32)
    valid4[:, visible_start:visible_end] = 1.0
    pred_points4 = lanes4.unsqueeze(0).clone()
    indices4 = [(torch.arange(4), torch.arange(4))]
    gt4_logits = torch.full((1, 4, 56), 4.0)
    gt4_logits[0, 0, visible_start:visible_end] = -3.0
    gt4_logits[0, 3, visible_start:visible_end] = -3.0
    gt4_base_loss = base.point_valid_loss(gt4_logits, pred_points4, [valid4], indices4, gt_points=[lanes4])
    gt4_segment_loss = segment.point_valid_loss(gt4_logits, pred_points4, [valid4], indices4, gt_points=[lanes4])
    assert torch.isclose(gt4_segment_loss, gt4_base_loss)
