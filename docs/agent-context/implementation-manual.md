# Implementation Manual

This branch is the current mainline source import of `5-25-3.zip` with a K56 TuSimple contract adaptation.

## Implementation Rules

- Keep the 5-25-3 algorithm body unchanged unless a future task explicitly asks for an algorithm change.
- Only change code/config needed for Q=12/K=56, `fixed_y_start=710/720`, `fixed_y_end=160/720`, `--imgsz 544 960`, and the K56 data/model YAMLs.
- Do not import later mainline Count Head, Quality Head, Survival Head, near-miss, or K56 candidate machinery.
- The only active Count Boundary mechanism is the 2026-06-27 user-requested, default-off `count_boundary_loss` on `sum(sigmoid(pred_logits))`; keep it separate from Count Head and decode changes.
- The active train-only hard-sampling mechanism is the 2026-06-27 user-requested, default-off `gcs_hard_sampling`; keep it limited to the training dataloader and do not change labels, validation/test dataloaders, point loss, smooth loss, curve loss, decode, or official metrics.
- The active E3-lite spurious negative mechanism is the 2026-06-27 user-requested, default-off `gcs_spurious_neg`; keep it limited to an extra `GCSLoss` BCE term on selected unmatched short duplicate-like queries and do not change data sampling, matcher logic, point loss, smooth loss, curve loss, decode, or official metrics.
- The active matched short-side geometry mechanism is the 2026-07-02 user-requested, default-off `gcs_short_side_geom`; keep it limited to matched short visible GT lanes and do not change unmatched queries, matcher logic, decode, NMS, or official metrics.
- The active far-spurious negative mechanism is the 2026-07-03 user-requested, default-off `gcs_far_spurious_neg`; keep it limited to target-zero BCE on gated far unmatched queries, with GT5 pressure off by default unless explicitly ablated.
- The active ignore-first count-contract mechanisms are default-off and training-only: `gcs_farspur_weight`, `gcs_rank_topk_weight`, reliable shortside raw-match/rescue, ultra-short light score/valid protection, and side-duplicate ranking negatives. They must not change labels, matcher assignment, decode, NMS, or official metrics.
- `gcs_shortside_rawmatch_boost` must remain a pure loss-weight boost unless `gcs_shortside_exist_target_floor > 0` is explicitly set; do not implicitly couple the boost knob to target flooring.
- Base `exist_loss` and `point_valid_loss` ignore masks must be controlled only by explicit default-off flags: `gcs_base_ignore_raw_rescue`, `gcs_base_ignore_rank_near`, `gcs_base_ignore_farspur_near`, and `gcs_base_ignore_duplicate_like`. Enabling ranking, farspur, raw-rescue, or `gcs_farspur_ignore_first=True` must not implicitly change base BCE supervision. Farspur-derived base ignore is effective only when the farspur ignore-first loss is active (`gcs_farspur_ignore_first=True` and `gcs_farspur_weight > 0`) and `gcs_base_ignore_farspur_near=True`; `gcs_farspur_weight=0` must not protect base BCE through the farspur masks. Rank-derived base-ignore sources must stay scoped to `gt_lanes >= gcs_rank_gt_min_lanes`, and final clear-far queries must remain trainable BCE negatives.
- Ranking duplicate-like negatives must stay ranking-only. Do not add normal duplicate-like or side duplicate-like queries to the original BCE hard-negative path; only clear-far spurious queries may be BCE hard negatives under the ignore-first contract.
- `gcs_rank_pair_reduction=global_pair_mean` is the default ranking reduction. Keep `image_mean` available only as an explicit legacy-compatible ablation.
- `gcs_rank_pos_scope=gt4gt5_matched` and `gcs_rank_pos_scope=all_matched` must mean all Hungarian matched true-lane queries in their image scopes; they must not be narrowed by `gcs_shortside_ultra_rank_pos`.
- Ultra-short visible `2..5` side GT lanes must stay a separate tier. Do not silently apply strong point geometry or ranking-positive treatment unless `gcs_shortside_ultra_enable` and the relevant ultra gain/rank flags are explicitly enabled.
- Keep the explicit 2026-06-27 `official_best` hook limited to official-val checkpoint/decode selection; it must not change model outputs, loss terms, training labels, or official metrics.
- Keep shared fixed-y contract helpers in `ultralytics/utils/gcs_fixed_y.py` so dataset and standalone tools do not depend on `ultralytics.models`.
- Keep ordered-slot target construction in float32 under AMP. Do not cast GT points to prediction dtype before fixed-y validation or slot construction.
- Do not use `--scale > 0` with `gcs_mode=ordered_slot` until a count-preserving fixed-y scale augmentation fallback exists. Query-mode scale behavior is unchanged.
- Ordered-slot training-time official-best candidate sweeps must use slot order with `order_check=warn` and `uses_runtime_sort=false`, record `strict_order_valid` and `ordered_slot_order_violations`, and select strict-order-valid checkpoints first. Final ordered-slot official/eval/main `official_best` decode must use slot order with `order_check=error`; runtime sorting is allowed only for debug/visualization exports marked as not usable for main ordered-slot claims.
- `OrderedSlotGCSLoss` standalone defaults must keep `gcs_count_ce=1.0`, `gcs_interval=1.0`, `gcs_order=0.2`, `gcs_gt_bottom_order=1.0`, `gcs_decoded_bottom_order=1.0`, and `gcs_bottom_order_margin_px=2.0`. `gcs_order` is the common-anchor order loss only; GT-bottom and decoded-bottom order losses are separate terms that do not require common visible anchors. Do not let missing args silently disable count, interval, or order supervision.
- For `split=val`, official evaluation must use the canonical 363-image GT JSON by default. Noncanonical GT requires an explicit flag and must be marked not comparable to E1/spurious official-val evidence.
- Pass ordered-slot-only model args to `parse_model` only when `gcs_mode=ordered_slot`; query-mode construction must ignore those overrides.
- Do not track `datasets/`, generated runs, checkpoints, caches, or converted labels in Git.

Active source/config is rolled back to commit `b6535f641` (`Fix GCS training
progress header alignment`). Post-`b6535f641` mechanisms such as Count Head,
Q18/Q20/dataref, duplicate/spurious/ranking losses, lane-balanced or
valid-repair objectives, side-aux checks, and their diagnostic helpers are
legacy records only and are not available in the current code unless a future
task explicitly restores them. The branch-local `count_boundary_loss` added on
2026-06-27, the branch-local train-only `gcs_hard_sampling` added on
2026-06-27, the branch-local `gcs_spurious_neg` E3-lite loss added on
2026-06-27, the branch-local `gcs_short_side_geom` loss added on 2026-07-02,
the branch-local `gcs_far_spurious_neg` loss added on 2026-07-03, and the
branch-local ignore-first/ranking/shortside count-contract tools updated on
2026-07-05 are explicit exceptions requested by the user and remain
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
ultralytics/utils/gcs_fixed_y.py
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

Ordered-slot v2 adds the ordered deterministic slot outputs:

```text
pred_count_logits: B x 4
pred_start_logits: B x 5 x 56
pred_end_logits: B x 5 x 56
pred_exist_logits: B x 5
```

The ordered-slot count classes are formal 2/3/4/5 classes:
`count_label = num_lanes - 2`, and decode uses
`argmax(pred_count_logits) + 2`.

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
