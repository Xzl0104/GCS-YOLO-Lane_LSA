# Implementation Manual

This branch is the current mainline source import of `5-25-3.zip` with a K56 TuSimple contract adaptation.

## Implementation Rules

- Keep the 5-25-3 algorithm body unchanged unless a future task explicitly asks for an algorithm change.
- Only change code/config needed for Q=12/K=56, `fixed_y_start=710/720`, `fixed_y_end=160/720`, `--imgsz 544 960`, and the K56 data/model YAMLs.
- Do not import later mainline Count Boundary, Quality Head, Survival Head, near-miss, official-best, or K56 candidate machinery.
- The q18-k56-gt4-candidate-countguard task explicitly adds a branch-local Q18 side-dense config and explicit 3/4/5 count head with default-disabled `gcs_count_ce`.
- The q20 side-geometry follow-up explicitly adds a branch-local Q20 experiment config. It is not the default model and its 2026-06-26 hard diagnostic is rejected before official-val sweep.
- Do not track `datasets/`, generated runs, checkpoints, caches, or converted labels in Git.

Active source/config is based on rollback commit `50999d6af` plus
default-disabled `duplicate_margin_loss`, `spurious_margin_loss`,
`lane_balanced_point_loss`, `short_valid_recall_loss`,
`far_spurious_survival_loss`, `gt5_rank_consistency_loss`, and
`gt3_extra_survival_loss`, `gt4_lane_balanced_point_loss`, and train-only
`gcs_gt4_sample_gain` experiment knobs. It also includes default-off GT4
short-lane candidate-recall knobs:
`gcs_lane_balanced_point_loss`, `gcs_gt4_short_lane_weight`,
`gcs_gt4_short_lane_max_points`, `gcs_gt4_short_match_endpoint`, and
`gcs_gt4_short_match_max_points`. It further includes default-off GT4
short-lane valid repair knobs:
`gcs_lane_balanced_valid_loss`, `gcs_gt4_short_valid_lane_weight`,
`gcs_gt4_short_valid_pos_weight`, `gcs_unmatched_valid_neg_weight`,
`gcs_gt4_short_valid_recall`, `gcs_gt4_short_valid_recall_weight`,
`gcs_gt4_short_valid_max_points`,
`gcs_gt4_short_valid_count_floor`, `gcs_gt4_short_valid_count_floor_weight`,
`gcs_gt4_short_valid_count_floor_ratio`, and
`gcs_gt4_short_valid_count_floor_min`.
Other post-`50999d6af` experiment knobs such as `gcs_gt4_short_boost`,
`gcs_extra_exist`, and `gcs_short_exist_floor` are legacy records only and are
not available in the current code unless a future task explicitly restores them.

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
tools/diagnose_gt4_missing_lane_raw_queries.py
tools/diagnose_tusimple_extra_lanes.py
tools/build_gt4_hard_val_split.py
tools/check_model.py
tools/check_count_guided_sweep_smoke.py
tools/check_q18_dryrun_metrics_contract.py
tools/check_q20_contract.py
tools/check_q20_pretrained_transfer.py
tools/check_q20_reference_coverage.py
ultralytics/nn/modules/gcs_lane.py
```

## Expected Output

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
pred_count_logits: B x 3
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

## Validation Order

1. Compile changed Python files.
2. Check YAML contract values.
3. Run `tools/check_model.py` with `--imgsz 544 960`.
4. Check fixed-y anchors are exactly `710..160` step `-10`.
5. If a K56 dataset root is available, run label order/split checks against that root.
6. For q18 countguard validation, run `tools/check_count_guided_sweep_smoke.py`
   and `tools/check_q18_dryrun_metrics_contract.py` before remote official-val
   count-guided sweeps.
7. For Q20 side-geometry validation, run `tools/check_q20_contract.py` and
   hard GT4 raw-query diagnostics before any official-val sweep.

## Agent Tooling

Agent/Skill configuration is local Codex workspace context, not part of the server-side algorithm payload for this branch. Do not require agent setup checks on the remote training server. These workflow rules do not change the 5-25-3 algorithm body or activate later mainline Count/Quality/Boundary behavior.
