# Implementation Manual

This branch is the current mainline source import of `5-25-3.zip` with a K56 TuSimple contract adaptation.

## Implementation Rules

- Keep the 5-25-3 algorithm body unchanged unless a future task explicitly asks for an algorithm change.
- Only change code/config needed for Q=12/K=56, `fixed_y_start=710/720`, `fixed_y_end=160/720`, `--imgsz 544 960`, and the K56 data/model YAMLs.
- Do not import later mainline Count Head, Quality Head, Survival Head, near-miss, or K56 candidate machinery. The only Count Head exception is the 2026-07-06 user-requested, default-off query Count Head ablation documented in `current-contracts.md`.
- Historical/default-off `count_boundary_loss` keys may remain for old command compatibility, but the active 2026-09-08 query five-loss `GCSLoss` must not compute, log, return, or backpropagate count-boundary terms.
- The active train-only hard-sampling mechanism is the 2026-06-27 user-requested, default-off `gcs_hard_sampling`; keep it limited to the training dataloader and do not change labels, validation/test dataloaders, point loss, smooth loss, curve loss, decode, or official metrics.
- Historical/default-off `gcs_spurious_neg` keys may remain for old E3-lite run interpretation, but the active 2026-09-08 query five-loss `GCSLoss` must not compute, log, return, or backpropagate spurious-negative terms.
- Post-`424ab1c86` short-side hardset diagnostics, `gcs_short_side_geom`, `gcs_far_spurious_neg`, `gcs_farspur_*`, `gcs_shortside_*`, `gcs_rank_*`, `gcs_base_ignore_*`, and count-contract diagnostic tooling are legacy records only. Do not reintroduce them into active code/config unless a future task explicitly asks for that algorithm/tooling change.
- Keep the explicit 2026-06-27 `official_best` hook limited to official-val checkpoint/decode selection; it must not change model outputs, loss terms, training labels, or official metrics.
- Keep shared fixed-y contract helpers in `ultralytics/utils/gcs_fixed_y.py` so dataset and standalone tools do not depend on `ultralytics.models`.
- Keep ordered-slot target construction in float32 under AMP. Do not cast GT points to prediction dtype before fixed-y validation or slot construction.
- Do not use `--scale > 0` with `gcs_mode=ordered_slot` until a count-preserving fixed-y scale augmentation fallback exists. Query-mode scale behavior is unchanged.
- Ordered-slot training-time official-best candidate sweeps must use slot order with `order_check=warn` and `uses_runtime_sort=false`, record `strict_order_valid` and `ordered_slot_order_violations`, and select strict-order-valid checkpoints first. Final ordered-slot official/eval/main `official_best` decode must use slot order with `order_check=error`; runtime sorting is allowed only for debug/visualization exports marked as not usable for main ordered-slot claims.
- `OrderedSlotGCSLoss` standalone defaults must keep `gcs_count_ce=1.0`, `gcs_interval=1.0`, `gcs_order=0.2`, `gcs_gt_bottom_order=1.0`, `gcs_decoded_bottom_order=1.0`, and `gcs_bottom_order_margin_px=2.0`. `gcs_order` is the common-anchor order loss only; GT-bottom and decoded-bottom order losses are separate terms that do not require common visible anchors. Do not let missing args silently disable count, interval, or order supervision.
- For `split=val`, official evaluation must use the canonical 363-image GT JSON by default. Noncanonical GT requires an explicit flag and must be marked not comparable to E1/spurious official-val evidence.
- Pass ordered-slot-only model args to `parse_model` only when `gcs_mode=ordered_slot`; query-mode construction must ignore those overrides.
- Do not track `datasets/`, generated runs, checkpoints, caches, or converted labels in Git.

Active source/config is rolled back to commit `424ab1c86` (`Add
valid-before-maxdet decode option`). Post-`424ab1c86` mechanisms such as
short-side hardset diagnostics, `gcs_short_side_geom`, `gcs_far_spurious_neg`,
ignore-first/ranking/shortside count-contract tooling, later mainline Count Head,
Q18/Q20/dataref, lane-balanced or valid-repair objectives, side-aux checks, and
their diagnostic helpers are legacy records only and are not available in the
current code unless a future task explicitly restores them. The train-only
`gcs_hard_sampling`, training-time `official_best`, ordered-slot protocol
tooling, and `valid_before_maxdet` decode option are inside the active rollback
boundary. Count-boundary and spurious-negative loss keys are legacy
compatibility only under the active query five-loss contract.

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
tools/sweep_tusimple_official_cached.py
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

The optional query Count Head ablation adds only:

```text
pred_count_logits: B x 4
```

for query models built from
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count.yaml`. The default
query YAML still emits no `pred_count_logits`.

## Validation Order

1. Compile changed Python files.
2. Check YAML contract values.
3. Run `tools/check_model.py` with `--imgsz 544 960`.
4. Check fixed-y anchors are exactly `710..160` step `-10`.
5. If a K56 dataset root is available, run label order/split checks against that root.

For changes to training-time official checkpoint selection, also compile:

```text
tools/train_gcs.py
tools/sweep_tusimple_official_cached.py
tools/sweep_tusimple_official.py
ultralytics/models/yolo/gcs_lane/train.py
ultralytics/engine/trainer.py
ultralytics/cfg/__init__.py
```

## Agent Tooling

Agent/Skill configuration is local Codex workspace context, not part of the server-side algorithm payload for this branch. Do not require agent setup checks on the remote training server. These workflow rules do not change the 5-25-3 algorithm body or activate later mainline Count/Quality/Boundary behavior.
