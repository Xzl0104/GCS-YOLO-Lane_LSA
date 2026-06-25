"""Check Q20 dataref point-reference reset through the real GCS pretrained loader."""

from __future__ import annotations

import sys
import io
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer
from ultralytics.nn.modules.gcs_lane import GCSLaneHead, Q20_DATAREF_X
from ultralytics.nn.tasks import GCSLaneModel


CFG = ROOT / "ultralytics" / "cfg" / "models" / "gcs" / "gcs-yolo-lane-s-q20-k56-dataref.yaml"


def state_with_point_reference(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return model.state_dict() plus the non-persistent point_reference_logits buffer."""
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for name, module in model.named_modules():
        if isinstance(module, GCSLaneHead) and hasattr(module, "point_reference_logits"):
            key = f"{name}.point_reference_logits" if name else "point_reference_logits"
            state[key] = module.point_reference_logits.detach().cpu().clone()
    return state


def find_single_ref_key(state: dict[str, torch.Tensor]) -> str:
    """Find the single GCS point_reference_logits key."""
    ref_keys = sorted(key for key in state if key.endswith("point_reference_logits"))
    if len(ref_keys) != 1:
        raise AssertionError(f"Expected exactly one point_reference_logits key, got {ref_keys}")
    return ref_keys[0]


def find_probe_key(state: dict[str, torch.Tensor], ref_key: str) -> str:
    """Choose a normal floating tensor that should load from the fake checkpoint."""
    for key, value in state.items():
        if key == ref_key:
            continue
        if value.is_floating_point() and value.numel() > 0 and value.shape:
            return key
    raise AssertionError("No loadable floating probe tensor found.")


def write_minimal_data_yaml(tmp_dir: Path) -> Path:
    """Create a minimal dataset YAML so GCSLaneTrainer can be constructed."""
    data_root = tmp_dir / "dataset"
    (data_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (data_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    data_yaml = tmp_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {data_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "image_shape: [544, 960]",
                "point_mode: fixed_y",
                "num_points: 56",
                "fixed_y: [0.9861111111111112, 0.2222222222222222]",
                "names:",
                "  0: lane",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def build_loader_trainer(tmp_dir: Path) -> GCSLaneTrainer:
    """Construct a real GCSLaneTrainer configured for this method-level loader test."""
    data_yaml = write_minimal_data_yaml(tmp_dir)
    return GCSLaneTrainer(
        overrides={
            "model": str(CFG),
            "data": str(data_yaml),
            "device": "cpu",
            "batch": 1,
            "epochs": 1,
            "workers": 0,
            "imgsz": [544, 960],
            "gcs_imgsz": [544, 960],
            "val": False,
            "save": False,
            "plots": False,
            "pretrained": False,
            "project": str(tmp_dir / "runs"),
            "name": "loader_reset",
            "exist_ok": True,
            "verbose": False,
            "reset_point_reference": True,
        }
    )


def main() -> None:
    """Assert reset_point_reference skips only point_reference_logits through the real loader."""
    tmp_parent = ROOT / ".tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)

    model = GCSLaneModel(str(CFG), nc=1, ch=3, verbose=False)
    initial_state = state_with_point_reference(model)
    ref_key = find_single_ref_key(initial_state)
    probe_key = find_probe_key(initial_state, ref_key=ref_key)

    initialized_ref = torch.sigmoid(initial_state[ref_key].detach().cpu())
    expected_ref = torch.tensor(Q20_DATAREF_X, dtype=torch.float32)
    if not torch.allclose(initialized_ref, expected_ref, atol=2e-6):
        raise AssertionError((initialized_ref[:, 0], expected_ref[:, 0]))

    fake_state = {key: value.clone() for key, value in initial_state.items()}
    fake_state[ref_key] = torch.full_like(fake_state[ref_key], 3.0)
    if torch.allclose(torch.sigmoid(fake_state[ref_key]), expected_ref, atol=2e-6):
        raise AssertionError("Fake checkpoint point_reference_logits must differ from Q20_DATAREF_X.")
    fake_state[probe_key] = torch.full_like(fake_state[probe_key], 0.12345)

    tmp_dir = tmp_parent / "q20_dataref_reset_check"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tmp_dir / "fake_state_dict.pt"
    buffer = io.BytesIO()
    torch.save({"state_dict": fake_state}, buffer)
    ckpt_path.write_bytes(buffer.getvalue())

    trainer = build_loader_trainer(tmp_dir)
    loaded = trainer.load_gcs_pretrained(model, ckpt_path)
    if loaded is not None:
        model = loaded

    loaded_state = state_with_point_reference(model)
    loaded_ref = torch.sigmoid(loaded_state[ref_key].detach().cpu())
    loaded_probe = loaded_state[probe_key].detach().cpu()
    expected_probe = torch.full_like(loaded_probe, 0.12345)

    if not torch.allclose(loaded_ref, expected_ref, atol=2e-6):
        raise AssertionError("point_reference_logits was overwritten despite reset_point_reference=True.")
    if not torch.allclose(loaded_probe, expected_probe, atol=1e-6):
        raise AssertionError(f"Probe tensor {probe_key} was not loaded from the fake checkpoint.")

    print("OK: real GCSLaneTrainer.load_gcs_pretrained preserves Q20_DATAREF_X while loading other tensors.")
    print("point_reference_logits key:", ref_key)
    print("probe key:", probe_key)


if __name__ == "__main__":
    main()
