# Project Context

GCS-YOLO-Lane modifies YOLO11 into a structured lane detection network.

The model is expected to output lane instances as ordered 2D point sequences, not ordinary segmentation masks. The current research target is clean TuSimple official Accuracy under a reproducible and leakage-free protocol.

## Main Files

- Default model on this branch: `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml`
- Legacy alias on this branch: `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`
- Data config: `data/tusimple_gcs_fixed_y_k56_960x544.yaml`
- Training entry: `tools/train_gcs.py`
- Inference entry: `tools/infer_gcs.py`
- Custom GCS eval: `tools/eval_gcs.py`
- Model shape check: `tools/check_model.py`

## Current Branch Direction

This branch imports the historical `5-25-3.zip` algorithm as a separate K56-compatible branch.

Only the explicit TuSimple contract was changed from the legacy `Q=8/K=32/fixed_y=[0.98,0.25]` setup to the current `Q=12/K=56/fixed_y=710/720 -> 160/720` setup. The 5-25-3 algorithm body is intentionally not upgraded to the later Count Head, Count Boundary, Quality Head, Survival Head, or near-miss machinery.

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

The K56 labels keep the same split sizes and align fixed-y anchors exactly to TuSimple official h-samples `710..160` at step `10`.

Official TuSimple Accuracy evaluation needs the original TuSimple archive layout, not only the fixed-y converted dataset. This 5-25-3 branch does not include the later mainline official evaluation helpers, so generate predictions with this branch and run official-val evaluation from a compatible evaluation checkout when needed.

Required test archive shape:

```text
archive/TUSimple/test_label.json
archive/TUSimple/test_set/clips/<date>/<clip>/<frame>.jpg
archive/TUSimple/train_set/
```

A minimal test-only archive may include only the 2,782 images referenced by `test_label.json`; it does not need the full TuSimple `test_set/clips` frame dump for final test evaluation.

## Collaboration Model

This branch is a source import of `5-25-3.zip`; the zip did not include project `.codex/`, `.agents/`, `scripts/`, or `tests/` directories. Use the current Codex runtime tools for any multi-agent review, and keep write ownership in the main worktree unless separate worktrees and disjoint ownership are explicit.
