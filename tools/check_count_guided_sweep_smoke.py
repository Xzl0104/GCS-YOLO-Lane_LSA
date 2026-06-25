"""Smoke-check count-guided TuSimple official sweep row expansion."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcs_tools.tusimple_official_eval import DEFAULT_OFFICIAL_SCORE_FN_WEIGHT, DEFAULT_OFFICIAL_SCORE_FP_WEIGHT
from tools import sweep_tusimple_official as sweep_mod


class FakeBenchResult:
    """Minimal official-eval result object for sweep smoke tests."""

    def __init__(self, images: int):
        self.images = int(images)

    def as_dict(self) -> dict:
        return {"Accuracy": 0.99, "FP": 0.01, "FN": 0.02, "images": self.images}


class FakeTuSimpleOfficialLaneEval:
    """Minimal bench_records provider with the same call shape as the real evaluator."""

    @staticmethod
    def bench_records(pred_records, gt_records, strict_length=True, return_records=False):
        return FakeBenchResult(len(gt_records)), []


class FakeModel:
    """Model stub that exposes pred_count_logits and counts forwards."""

    def __init__(self):
        self.calls = 0

    def __call__(self, tensor):
        self.calls += 1
        return {
            "pred_points": torch.zeros(1, 6, 56, 2),
            "pred_logits": torch.ones(1, 6),
            "pred_valid_logits": torch.ones(1, 6, 56),
            # Classes map to allowed counts 3/4/5; count=4 has high confidence.
            "pred_count_logits": torch.tensor([[0.1, 2.0, 0.1]], dtype=torch.float32),
        }


def fake_records() -> list[dict]:
    """Return two GT4 records so count-guided topK can improve count diagnostics."""
    h_samples = list(range(710, 150, -10))
    lane = [100 for _ in h_samples]
    return [
        {
            "raw_file": f"clips/0000/{idx}/20.jpg",
            "h_samples": h_samples,
            "lanes": [lane, lane, lane, lane],
        }
        for idx in range(2)
    ]


def make_args(save_dir: Path) -> argparse.Namespace:
    """Build a 4-base-combo, 8-count-guided-row sweep namespace."""
    return argparse.Namespace(
        dataset="tusimple",
        archive_root=str(ROOT / "archive"),
        split="val",
        gt_json=str(save_dir / "fake_gt.json"),
        weights=str(save_dir / "fake.pt"),
        imgsz=[544, 960],
        confs=[0.01, 0.02],
        point_valid_thrs=[0.5],
        nms_dist_pxs=[0.0, 18.0],
        max_dets=[5],
        min_points=[4],
        max_images=0,
        warmup=0,
        device="cpu",
        half=False,
        runtime_ms=1.0,
        save_dir=str(save_dir),
        score_fp_weight=DEFAULT_OFFICIAL_SCORE_FP_WEIGHT,
        score_fn_weight=DEFAULT_OFFICIAL_SCORE_FN_WEIGHT,
        count_guided_topk=True,
        count_guided_min_probs=[0.60, 0.65],
        count_guided_allowed_counts="3,4,5",
        count_guided_allow_unsupported_fallback=False,
    )


def main() -> None:
    """Run the synthetic count-guided sweep and assert row/stat contracts."""
    save_dir = ROOT / ".tmp" / "count_guided_sweep_smoke"
    resolved_save_dir = save_dir.resolve()
    tmp_root = (ROOT / ".tmp").resolve()
    if tmp_root not in resolved_save_dir.parents:
        raise RuntimeError(f"Refusing to clean smoke directory outside .tmp: {resolved_save_dir}")
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = FakeModel()
    records = fake_records()
    sweep_mod.find_tusimple_archive_root = lambda path: Path(path)
    sweep_mod.default_tusimple_gt_json = lambda archive_root, split: save_dir / "fake_gt.json"
    sweep_mod.read_tusimple_json_lines = lambda path: list(records)
    sweep_mod.tusimple_image_path = lambda archive_root, raw_file, split: save_dir / raw_file
    sweep_mod.cv2.imread = lambda path, flags: np.zeros((544, 960, 3), dtype=np.uint8)
    sweep_mod.select_device = lambda device: torch.device("cpu")
    sweep_mod.load_gcs_model = lambda weights, device, half, gcs_imgsz: model
    sweep_mod.preprocess_image = lambda img, imgsz, device, half: torch.zeros(1, 3, int(imgsz[0]), int(imgsz[1]))
    sweep_mod.warn_max_det_mismatch = lambda *args, **kwargs: None
    sweep_mod.decode_gcs_predictions = lambda *args, **kwargs: [{"score": float(i)} for i in range(5)]
    sweep_mod.gcs_lanes_to_tusimple_lanes = lambda lanes, h_samples, image_shape: [
        [100 for _ in h_samples] for _ in lanes
    ]
    sweep_mod.TuSimpleOfficialLaneEval = FakeTuSimpleOfficialLaneEval

    output = sweep_mod.sweep(make_args(save_dir))
    rows = output["results"]
    normal_rows = [row for row in rows if row["mode"] == "normal"]
    guided_rows = [row for row in rows if row["mode"] == "count_guided_topk"]

    assert len(normal_rows) == 4, len(normal_rows)
    assert len(guided_rows) == 8, len(guided_rows)
    assert model.calls == len(records), model.calls
    assert all(row["count_guided_unsupported_images"] == 0 for row in guided_rows)
    assert all(row["count_guided_applied_images"] == len(records) for row in guided_rows)
    assert all(row["count_head_key"] == "pred_count_logits" for row in guided_rows)
    assert output["best"]["mode"] in {"normal", "count_guided_topk"}
    assert "count_guided_min_prob" in output["best"]

    print("OK: count-guided official sweep smoke passed.")
    print("normal_rows:", len(normal_rows))
    print("count_guided_rows:", len(guided_rows))
    print("forwards:", model.calls)
    print("best_mode:", output["best"]["mode"])


if __name__ == "__main__":
    main()
