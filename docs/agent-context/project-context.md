# Project Context

GCS-YOLO-Lane modifies YOLO11 into a structured lane detection network.

The model is expected to output lane instances as ordered 2D point sequences, not ordinary segmentation masks. The current research target is clean TuSimple official Accuracy under a reproducible and leakage-free protocol.

## Main Files

- Default model on this branch: `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`
- Compatibility model for old q12-k56 records: `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml`
- Data config: `data/tusimple_gcs_fixed_y_960x544.yaml`
- Compatibility data config for old q12-k56 records: `data/tusimple_gcs_fixed_y_k56_960x544.yaml`
- Training entry: `tools/train_gcs.py`
- Inference entry: `tools/infer_gcs.py`
- Custom GCS eval: `tools/eval_gcs.py`
- TuSimple official eval: `tools/eval_tusimple_official.py`
- TuSimple official sweep: `tools/sweep_tusimple_official.py`
- Model shape check: `tools/check_model.py`

## Current Branch Direction

This branch imports the historical `5-25-3.zip` algorithm and is now the current K56 mainline.

Only the explicit TuSimple contract was changed from the legacy `Q=8/K=32/fixed_y=[0.98,0.25]` setup to the current `Q=12/K=56/fixed_y=710/720 -> 160/720` setup, plus the branch-local default-off `count_boundary_loss`, train-only `gcs_hard_sampling`, default-off E3-lite `gcs_spurious_neg`, default-off `gcs_short_side_geom`, default-off `gcs_far_spurious_neg`, and training-time `official_best` protocol additions. The 5-25-3 algorithm body is intentionally not upgraded to the later Count Head, Quality Head, Survival Head, or near-miss machinery.

Previous q12-k56 experiment documentation remains historical context. Do not delete it, and do not read it as the active algorithm unless it is explicitly marked as a legacy run record.

## Data Summary

Current TuSimple fixed-y data root on this branch:

```text
datasets/tusimple_fixed_y_k56_960x544
```

Current split sizes:

```text
train: 3263
val:   363
test:  2782
```

The official-val subset is aligned with the current validation split and must stay separate from test-driven tuning.

The K56 labels keep the same split sizes and align fixed-y anchors exactly to TuSimple official h-samples `710..160`, descending by `10` pixels.

Official TuSimple Accuracy evaluation needs the original TuSimple archive layout, not only the fixed-y converted dataset. This branch includes `tools/eval_tusimple_official.py`, `tools/sweep_tusimple_official.py`, and `gcs_tools/tusimple_official_eval.py` for official-val evaluation and threshold sweeps. It also includes the explicit 2026-06-27 training-time `official_best` checkpoint-selection hook. Use official-val for checkpoint, threshold, and postprocess selection, and use test only once for the selected candidate.

Required test archive shape:

```text
archive/TUSimple/test_label.json
archive/TUSimple/test_set/clips/<date>/<clip>/<frame>.jpg
archive/TUSimple/train_set/
```

A minimal test-only archive may include only the 2,782 images referenced by `test_label.json`; it does not need the full TuSimple `test_set/clips` frame dump for final test evaluation.

## Collaboration Model

This branch is a source import of `5-25-3.zip`. Agent/Skill configuration is local Codex workspace context, not part of the server-side algorithm payload. Use multi-agent review only when explicitly requested, and keep write ownership in the main worktree unless separate worktrees and disjoint ownership are explicit.
