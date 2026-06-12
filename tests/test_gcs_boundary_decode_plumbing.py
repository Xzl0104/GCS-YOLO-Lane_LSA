from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_tusimple_official, sweep_tusimple_official
from ultralytics.models.yolo.gcs_lane import train as gcs_train
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer
from ultralytics.utils.gcs_postprocess import decode_gcs_predictions


def _fake_predictions(boundary_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    points = torch.zeros((1, 5, 6, 2), dtype=torch.float32)
    points[0, :, :, 0] = torch.linspace(0.1, 0.9, 5).view(5, 1)
    points[0, :, :, 1] = torch.linspace(0.98, 0.25, 6)
    return {
        "pred_points": points,
        "pred_logits": torch.full((1, 5), 8.0),
        "pred_valid_logits": torch.full((1, 5, 6), 8.0),
        "pred_count_logits": torch.tensor([[-10.0, -10.0, 5.0, 4.0]]),
        "pred_count_boundary_logits": boundary_logits.view(1, 2),
        "pred_quality_logits": torch.full((1, 5), 8.0),
    }


@contextmanager
def _patched_official_inference(module, tmp_path: Path, boundary_logits: torch.Tensor, captured: list):
    predictions = _fake_predictions(boundary_logits)

    class FakeModel:
        def __call__(self, tensor):
            return predictions

    def capture_decode(*args, **kwargs):
        captured.append(kwargs["pred_count_boundary_logits"].detach().clone())
        return [], {"count_head_policy_count": 4}

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(module, "select_device", return_value=torch.device("cpu")))
        stack.enter_context(mock.patch.object(module, "load_gcs_model", return_value=FakeModel()))
        stack.enter_context(
            mock.patch.object(module, "preprocess_image", return_value=torch.zeros((1, 3, 544, 960)))
        )
        stack.enter_context(mock.patch.object(module, "_sync_if_cuda", return_value=None))
        stack.enter_context(mock.patch.object(module, "tusimple_image_path", return_value=tmp_path / "image.jpg"))
        stack.enter_context(
            mock.patch.object(module.cv2, "imread", return_value=np.zeros((720, 1280, 3), dtype=np.uint8))
        )
        stack.enter_context(mock.patch.object(module, "gcs_lanes_to_tusimple_lanes", return_value=[]))
        stack.enter_context(mock.patch.object(module, "decode_gcs_predictions", side_effect=capture_decode))
        yield


def _make_official_best_trainer(tmp_path: Path, *, top_k: int = 2) -> tuple[GCSLaneTrainer, Path, Path]:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(exist_ok=True)
    last = weights_dir / "last.pt"

    trainer = object.__new__(GCSLaneTrainer)
    trainer.args = SimpleNamespace(
        gcs_official_best=True,
        gcs_official_best_split="val",
        gcs_official_best_period=1,
        gcs_official_best_top_k=top_k,
        gcs_official_best_gt_json=str(tmp_path / "official_val.json"),
        gcs_official_best_archive_root=str(tmp_path),
        gcs_official_best_confs="0.005",
        gcs_official_best_point_valid_thrs="0.15",
        gcs_official_best_nms_dist_pxs="18.0",
        gcs_official_best_max_dets="5",
        gcs_official_best_min_points="6",
        gcs_official_best_rank_min_points="none",
    )
    trainer.save_dir = tmp_path
    trainer.wdir = weights_dir
    trainer.last = last
    trainer.gcs_official_best_fitness = None
    trainer._resolve_gcs_imgsz = lambda: (544, 960)
    return trainer, weights_dir, last


class CountBoundaryDecodePlumbingTest(unittest.TestCase):
    def test_decode_uses_count_boundary_logits_for_policy_count(self):
        predictions = _fake_predictions(torch.tensor([5.0, 5.0]))
        common = {
            "pred_points": predictions["pred_points"][0],
            "pred_logits": predictions["pred_logits"][0],
            "pred_valid_logits": predictions["pred_valid_logits"][0],
            "pred_count_logits": predictions["pred_count_logits"][0],
            "image_shape": (720, 1280),
            "score_thr": 0.0,
            "point_valid_thr": 0.5,
            "min_points": 2,
            "max_det": 5,
            "nms_dist_px": 0.0,
            "quality_rescue_5th": False,
        }

        raw_lanes = decode_gcs_predictions(**common)
        calibrated_lanes = decode_gcs_predictions(
            pred_count_boundary_logits=predictions["pred_count_boundary_logits"][0],
            **common,
        )

        self.assertEqual(len(raw_lanes), 4)
        self.assertEqual(len(calibrated_lanes), 5)

    def test_official_eval_forwards_count_boundary_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            boundary_logits = torch.tensor([2.5, -1.5])
            captured = []
            with _patched_official_inference(
                eval_tusimple_official,
                tmp_path,
                boundary_logits,
                captured,
            ):
                eval_tusimple_official.predict_tusimple_records(
                    weights="fake.pt",
                    gt_records=[{"raw_file": "clips/example.jpg", "h_samples": [160, 170]}],
                    archive_root=tmp_path,
                    split="test",
                    imgsz=(544, 960),
                    conf=0.05,
                    point_valid_thr=0.20,
                    nms_dist_px=18.0,
                    max_det=5,
                    min_points=6,
                    max_images=0,
                    warmup=0,
                    device="cpu",
                    half=False,
                    runtime_ms=1.0,
                    use_measured_runtime=False,
                    count_calibration=None,
                    rank_min_points=None,
                )

            self.assertEqual(len(captured), 1)
            self.assertTrue(torch.equal(captured[0], boundary_logits))

    def test_official_sweep_defaults_to_val_and_forwards_count_boundary_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with mock.patch.object(sys, "argv", ["sweep_tusimple_official.py"]):
                args = sweep_tusimple_official.parse_args()
            self.assertEqual(args.split, "val")

            args.archive_root = str(tmp_path)
            args.gt_json = str(tmp_path / "official_val.json")
            args.weights = "fake.pt"
            args.imgsz = [544, 960]
            args.confs = [0.05]
            args.point_valid_thrs = [0.20]
            args.nms_dist_pxs = [18.0]
            args.max_dets = [5]
            args.min_points = [6]
            args.rank_min_points = ["none"]
            args.max_images = 1
            args.warmup = 0
            args.device = "cpu"
            args.half = False
            args.save_dir = str(tmp_path / "sweep")

            boundary_logits = torch.tensor([-2.0, 3.0])
            captured = []

            def update_state(state, *args, **kwargs):
                state["images"] += 1

            with _patched_official_inference(
                sweep_tusimple_official,
                tmp_path,
                boundary_logits,
                captured,
            ):
                with ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(sweep_tusimple_official, "find_tusimple_archive_root", return_value=tmp_path)
                    )
                    stack.enter_context(
                        mock.patch.object(
                            sweep_tusimple_official,
                            "read_tusimple_json_lines",
                            return_value=[
                                {"raw_file": "clips/example.jpg", "h_samples": [160, 170], "lanes": []}
                            ],
                        )
                    )
                    stack.enter_context(mock.patch.object(sweep_tusimple_official, "update_state", update_state))
                    sweep_tusimple_official.run_sweep(args)

            self.assertEqual(len(captured), 1)
            self.assertTrue(torch.equal(captured[0], boundary_logits))

    def test_official_sweep_and_training_reject_test_selection(self):
        with self.assertRaisesRegex(ValueError, "tools/eval_tusimple_official.py --split test"):
            sweep_tusimple_official.validate_official_sweep_split("test")

        trainer = object.__new__(GCSLaneTrainer)
        trainer.args = SimpleNamespace(gcs_official_best=True, gcs_official_best_split="test")
        with mock.patch.object(gcs_train, "RANK", -1):
            with self.assertRaisesRegex(ValueError, "Training official_best selection"):
                trainer._run_official_best_sweep()

    def test_training_official_best_topk_preserves_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trainer, weights_dir, last = _make_official_best_trainer(tmp_path, top_k=2)

            sweep_results = iter((0.95, 0.96, 0.955, 0.96))

            def fake_run_sweep(_args):
                acc = next(sweep_results)
                return {
                    "best": {
                        "official_acc": acc,
                        "official_fp": 0.04,
                        "official_fn": 0.03,
                        "rank_min_points": "none",
                    }
                }

            with mock.patch.object(gcs_train, "RANK", -1), mock.patch.object(
                sweep_tusimple_official, "run_sweep", side_effect=fake_run_sweep
            ):
                for epoch, content in enumerate((b"epoch1", b"epoch2", b"epoch3", b"epoch4")):
                    trainer.epoch = epoch
                    last.write_bytes(content)
                    trainer._run_official_best_sweep()

            record = json.loads((tmp_path / "official_best_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(record["best_epoch"], 2)
            self.assertEqual(record["best_fitness"], 0.96)
            self.assertEqual((weights_dir / "official_best.pt").read_bytes(), b"epoch2")
            self.assertEqual(record["official_top_k_size"], 2)
            self.assertEqual([entry["candidate_epoch"] for entry in record["official_top_k"]], [2, 4])
            self.assertEqual([entry["rank"] for entry in record["official_top_k"]], [1, 2])
            top1 = tmp_path / record["official_top_k"][0]["weights"]
            top2 = tmp_path / record["official_top_k"][1]["weights"]
            self.assertEqual(top1.read_bytes(), b"epoch2")
            self.assertEqual(top2.read_bytes(), b"epoch4")

    def test_training_official_best_topk_migrates_legacy_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trainer, weights_dir, last = _make_official_best_trainer(tmp_path, top_k=2)
            (weights_dir / "official_best.pt").write_bytes(b"legacy-best")
            (tmp_path / "official_best_summary.json").write_text(
                json.dumps(
                    {
                        "best_epoch": 6,
                        "best_fitness": 0.954,
                        "best": {"official_acc": 0.954, "rank_min_points": "none"},
                        "selector": {"metric": "official_acc"},
                    }
                ),
                encoding="utf-8",
            )
            last.write_bytes(b"new-candidate")
            trainer.epoch = 6

            def fake_run_sweep(_args):
                return {
                    "best": {
                        "official_acc": 0.953,
                        "official_fp": 0.04,
                        "official_fn": 0.03,
                        "rank_min_points": "none",
                    }
                }

            with mock.patch.object(gcs_train, "RANK", -1), mock.patch.object(
                sweep_tusimple_official, "run_sweep", side_effect=fake_run_sweep
            ):
                trainer._run_official_best_sweep()

            record = json.loads((tmp_path / "official_best_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(record["best_epoch"], 6)
            self.assertEqual(record["best_fitness"], 0.954)
            self.assertEqual((weights_dir / "official_best.pt").read_bytes(), b"legacy-best")
            self.assertEqual([entry["candidate_epoch"] for entry in record["official_top_k"]], [6, 7])
            migrated = tmp_path / record["official_top_k"][0]["weights"]
            current = tmp_path / record["official_top_k"][1]["weights"]
            self.assertEqual(migrated.read_bytes(), b"legacy-best")
            self.assertEqual(current.read_bytes(), b"new-candidate")


if __name__ == "__main__":
    unittest.main()
