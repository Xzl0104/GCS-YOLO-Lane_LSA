# Project Context

GCS-YOLO-Lane modifies YOLO11 into a structured lane detection network.

The model is expected to output lane instances as ordered 2D point sequences, not ordinary segmentation masks. The current research target is clean TuSimple official Accuracy under a reproducible and leakage-free protocol.

## Main Files

- Default model: `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml`
- Data config: `data/tusimple_gcs_fixed_y_k56_960x544.yaml`
- Training entry: `tools/train_gcs.py`
- Inference entry: `tools/infer_gcs.py`
- Custom GCS eval: `tools/eval_gcs.py`
- TuSimple official eval: `tools/eval_tusimple_official.py`
- Official sweep: `tools/sweep_tusimple_official.py`
- GT5 diagnosis: `tools/diagnose_gcs_gt5.py`
- TuSimple h-sample endpoint audit: `tools/analyze_tusimple_hsample_endpoints.py`
- Loss cleanup check: `scripts/verify_loss_cleanup.py`
- Count/decode contract check: `tools/check_gcs_count_head_topk_contract.py`
- Decode meta check: `tools/check_gcs_decode_meta_contract.py`
- Algorithm contract check: `tools/check_gcs_algorithm_contract.py`
- Label split/order/fixed-y contract check: `tools/check_gcs_label_order_split.py`

## Current Default Direction

The current default line uses Q=12/K56 fixed-y labels aligned to TuSimple official h-samples, Count Head with Count Boundary calibration, Quality Head, candidate-aware decode, strict official-val selection, and protected test usage.

The default line is not a research ban. Old or removed mechanisms can return as controlled experimental candidates when they are explicit, configurable, traceable, and evaluated on official-val without test leakage.

## Data Summary

Current TuSimple fixed-y data root:

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

The K56 labels keep the same split sizes and align fixed-y anchors exactly to TuSimple official h-samples `710..160` at step `10`. K56 is now the sole active code/config path; older K32 records are retained only as historical evidence.

Use `tools/check_gcs_label_order_split.py --expect-fixed-y 56,710/720,160/720` to verify exact K56 fixed-y artifacts. Use `tools/analyze_tusimple_hsample_endpoints.py` to audit raw TuSimple h-sample endpoint and ultra-short-lane coverage before changing label or decode semantics.

Official TuSimple Accuracy evaluation needs the original TuSimple archive layout, not only the fixed-y converted dataset.

Required test archive shape:

```text
archive/TUSimple/test_label.json
archive/TUSimple/test_set/clips/<date>/<clip>/<frame>.jpg
archive/TUSimple/train_set/
```

`tools/eval_tusimple_official.py --split test` resolves `raw_file` entries from `test_label.json` against this archive root. A minimal test-only archive may include only the 2,782 images referenced by `test_label.json`; it does not need the full TuSimple `test_set/clips` frame dump for final test evaluation.

## Collaboration Model

Project Agents and Skills are configured under `.codex/` and `.agents/skills/`. Use read-only Agents for exploration, review, experiment analysis, documentation research, and security review. Use only one writable Agent in the main worktree unless separate worktrees and disjoint ownership are explicit.
