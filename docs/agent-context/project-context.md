# Project Context

GCS-YOLO-Lane modifies YOLO11 into a structured lane detection network.

The model is expected to output lane instances as ordered 2D point sequences, not ordinary segmentation masks. The current research target is clean TuSimple official Accuracy under a reproducible and leakage-free protocol.

## Main Files

- Default model: `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml`
- Legacy Q=8 model: `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`
- Data config: `data/tusimple_gcs_fixed_y_960x544.yaml`
- Training entry: `tools/train_gcs.py`
- Inference entry: `tools/infer_gcs.py`
- Custom GCS eval: `tools/eval_gcs.py`
- TuSimple official eval: `tools/eval_tusimple_official.py`
- Official sweep: `tools/sweep_tusimple_official.py`
- GT5 diagnosis: `tools/diagnose_gcs_gt5.py`
- Loss cleanup check: `scripts/verify_loss_cleanup.py`
- Count/decode contract check: `tools/check_gcs_count_head_topk_contract.py`
- Decode meta check: `tools/check_gcs_decode_meta_contract.py`
- Algorithm contract check: `tools/check_gcs_algorithm_contract.py`

## Current Default Direction

The current default line uses Q=12, fixed-y labels, Count Head with Count Boundary calibration, Quality Head, candidate-aware decode, strict official-val selection, and protected test usage.

The default line is not a research ban. Old or removed mechanisms can return as controlled experimental candidates when they are explicit, configurable, traceable, and evaluated on official-val without test leakage.

## Data Summary

Current TuSimple fixed-y data root:

```text
datasets/tusimple_fixed_y_960x544
```

Current split sizes:

```text
train: 3263
val:   363
test:  2782
```

The official-val subset is aligned with the current validation split and must stay separate from test-driven tuning.

## Collaboration Model

Project Agents and Skills are configured under `.codex/` and `.agents/skills/`. Use read-only Agents for exploration, review, experiment analysis, documentation research, and security review. Use only one writable Agent in the main worktree unless separate worktrees and disjoint ownership are explicit.
