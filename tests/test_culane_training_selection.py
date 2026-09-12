from __future__ import annotations

from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer


def test_culane_val_epoch_interval_and_final_epoch() -> None:
    assert not GCSLaneTrainer._culane_val_epoch_due(epoch=0, epochs=10, interval=5)
    assert GCSLaneTrainer._culane_val_epoch_due(epoch=4, epochs=10, interval=5)
    assert not GCSLaneTrainer._culane_val_epoch_due(epoch=5, epochs=10, interval=5)
    assert GCSLaneTrainer._culane_val_epoch_due(epoch=9, epochs=10, interval=5)
    assert GCSLaneTrainer._culane_val_epoch_due(epoch=1, epochs=10, interval=5, stop=True)


def test_culane_val_best_key_prefers_f1_then_error_counts() -> None:
    base = {"f1": 0.8, "fp": 2, "fn": 1, "ape_mean_px": 2.0}
    assert GCSLaneTrainer._culane_val_best_key({**base, "f1": 0.9}, 1) > GCSLaneTrainer._culane_val_best_key(base, 1)
    assert GCSLaneTrainer._culane_val_best_key({**base, "fp": 1}, 1) > GCSLaneTrainer._culane_val_best_key(base, 1)
    assert GCSLaneTrainer._culane_val_best_key({**base, "fn": 0}, 1) > GCSLaneTrainer._culane_val_best_key(base, 1)
    assert GCSLaneTrainer._culane_val_best_key({**base, "ape_mean_px": 1.0}, 1) > GCSLaneTrainer._culane_val_best_key(base, 1)


def test_culane_val_best_key_prefers_newer_epoch_on_exact_tie() -> None:
    summary = {"f1": 0.8, "fp": 2, "fn": 1, "ape_mean_px": 2.0}
    assert GCSLaneTrainer._culane_val_best_key(summary, 2) > GCSLaneTrainer._culane_val_best_key(summary, 1)


def test_external_culane_fitness_refreshes_every_epoch() -> None:
    trainer = GCSLaneTrainer.__new__(GCSLaneTrainer)
    values = {
        "gcs_culane_val": True,
        "gcs_culane_external_fitness": True,
        "gcs_culane_val_interval": 5,
    }
    trainer._get_arg_value = lambda name, default=None: values.get(name, default)
    trainer._is_culane_training = lambda: True
    assert trainer._should_run_culane_val() is True
