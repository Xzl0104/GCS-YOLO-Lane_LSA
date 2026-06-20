# Implementation Manual

This branch is the current mainline source import of `5-25-3.zip` with a K56 TuSimple contract adaptation.

## Implementation Rules

- Keep the 5-25-3 algorithm body unchanged unless a future task explicitly asks for an algorithm change.
- Only change code/config needed for Q=12/K=56, `fixed_y_start=710/720`, `fixed_y_end=160/720`, `--imgsz 544 960`, and the K56 data/model YAMLs.
- Do not import later mainline Count Head, Count Boundary, Quality Head, Survival Head, near-miss, official-best, or K56 candidate machinery.
- Do not track `datasets/`, generated runs, checkpoints, caches, or converted labels in Git.

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

## Agent Tooling

This branch tracks `.codex/`, `.agents/`, and the repository wrapper scripts:

```text
scripts/check_gcs_agent_setup.py
scripts/gcs_spawn_adapter.py
scripts/gcs_spawn_payload.py
```

After Agent, Skill, context, or delegation-policy edits, run:

```bash
python scripts/check_gcs_agent_setup.py
```

These tools are synchronized for workflow parity only. They do not change the 5-25-3 algorithm body or activate later mainline Count/Quality/Boundary behavior.