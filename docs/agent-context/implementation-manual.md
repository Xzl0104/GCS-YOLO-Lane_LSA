# Implementation Manual

This branch is the current mainline source import of `5-25-3.zip` with a K56 TuSimple contract adaptation.

## Implementation Rules

- Keep the 5-25-3 algorithm body unchanged unless a future task explicitly asks for an algorithm change.
- Only change code/config needed for Q=12/K=56, `fixed_y_start=710/720`, `fixed_y_end=160/720`, `--imgsz 544 960`, and the K56 data/model YAMLs.
- Do not import later mainline Count Head, Quality Head, Survival Head, near-miss, or K56 candidate machinery.
- The only active Count Boundary mechanism is the 2026-06-27 user-requested, default-off `count_boundary_loss` on `sum(sigmoid(pred_logits))`; keep it separate from Count Head and decode changes.
- Keep the explicit 2026-06-27 `official_best` hook limited to official-val checkpoint/decode selection; it must not change model outputs, loss terms, training labels, or official metrics.
- Do not track `datasets/`, generated runs, checkpoints, caches, or converted labels in Git.

Active source/config is rolled back to commit `b6535f641` (`Fix GCS training
progress header alignment`). Post-`b6535f641` mechanisms such as Count Head,
Q18/Q20/dataref, duplicate/spurious/ranking losses, lane-balanced or
valid-repair objectives, side-aux checks, and their diagnostic helpers are
legacy records only and are not available in the current code unless a future
task explicitly restores them. The branch-local `count_boundary_loss` added on
2026-06-27 is an explicit exception requested by the user and remains
default-disabled.

## Main Files

Default active paths:

```text
data/tusimple_gcs_fixed_y_960x544.yaml
ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
```

Compatibility paths for old q12-k56 records:

```text
data/tusimple_gcs_fixed_y_k56_960x544.yaml
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
```

Shared implementation files:

```text
gcs_tools/label_utils.py
tools/convert_tusimple_to_gcs.py
tools/train_gcs.py
tools/eval_tusimple_official.py
tools/sweep_tusimple_official.py
tools/check_model.py
ultralytics/nn/modules/gcs_lane.py
ultralytics/models/yolo/gcs_lane/train.py
ultralytics/engine/trainer.py
```

## Expected Output

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

## Validation Order

1. Compile changed Python files.
2. Check YAML contract values.
3. Run `tools/check_model.py` with `--imgsz 544 960`.
4. Check fixed-y anchors are exactly `710..160` step `-10`.
5. If a K56 dataset root is available, run label order/split checks against that root.

For changes to training-time official checkpoint selection, also compile:

```text
tools/train_gcs.py
tools/sweep_tusimple_official.py
ultralytics/models/yolo/gcs_lane/train.py
ultralytics/engine/trainer.py
ultralytics/cfg/__init__.py
```

## Agent Tooling

Agent/Skill configuration is local Codex workspace context, not part of the server-side algorithm payload for this branch. Do not require agent setup checks on the remote training server. These workflow rules do not change the 5-25-3 algorithm body or activate later mainline Count/Quality/Boundary behavior.
