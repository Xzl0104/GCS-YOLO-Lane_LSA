# Implementation Manual

This branch is the current mainline source import of `5-25-3.zip` with a K56 TuSimple contract adaptation.

## Implementation Rules

- Keep the 5-25-3 algorithm body unchanged unless a future task explicitly asks for an algorithm change.
- Only change code/config needed for the active Q12/K56 env30 baseline,
  explicitly active default-off experiment YAMLs, `fixed_y_start=710/720`,
  `fixed_y_end=160/720`, `--imgsz 544 960`, and the K56 data/model YAMLs.
- Do not import later mainline Count Head, Quality Head, Survival Head, near-miss, or K56 candidate machinery. The Count Head exception is the 2026-07-06 user-requested, default-off query Count Head ablation documented in `current-contracts.md`; the candidate exception is the 2026-07-28 user-reopened, default-off Q12/env30 gated candidate v2 documented in `current-contracts.md`.
- The only active Count Boundary mechanism is the 2026-06-27 user-requested, default-off `count_boundary_loss` on `sum(sigmoid(pred_logits))`; keep it separate from Count Head and decode changes.
- The active train-only hard-sampling mechanism is the 2026-06-27 user-requested, default-off `gcs_hard_sampling`; keep it limited to the training dataloader and do not change labels, validation/test dataloaders, point loss, smooth loss, curve loss, decode, or official metrics.
- The active E3-lite spurious negative mechanism is the 2026-06-27 user-requested, default-off `gcs_spurious_neg`; keep it limited to an extra `GCSLoss` BCE term on selected unmatched short duplicate-like queries and do not change data sampling, matcher logic, point loss, smooth loss, curve loss, decode, or official metrics.
- Post-`86c8fb31c` experiment code, including staticref, near20, Q20/Q24, dual-head, query extent, short local-refine, lateral candidate, gated candidate, role/event containment, and candidate gate fixes, is rejected legacy state only. Do not reintroduce it into active code/config unless a future task explicitly asks for that algorithm/tooling change.
- Keep the explicit 2026-06-27 `official_best` hook limited to official-val checkpoint/decode selection; it must not change model outputs, loss terms, training labels, or official metrics.
- Keep shared fixed-y contract helpers in `ultralytics/utils/gcs_fixed_y.py` so dataset and standalone tools do not depend on `ultralytics.models`.
- Keep ordered-slot target construction in float32 under AMP. Do not cast GT points to prediction dtype before fixed-y validation or slot construction.
- Do not use `--scale > 0` with `gcs_mode=ordered_slot` until a count-preserving fixed-y scale augmentation fallback exists. Query-mode scale behavior is unchanged.
- Ordered-slot training-time official-best candidate sweeps must use slot order with `order_check=warn` and `uses_runtime_sort=false`, record `strict_order_valid` and `ordered_slot_order_violations`, and select strict-order-valid checkpoints first. Final ordered-slot official/eval/main `official_best` decode must use slot order with `order_check=error`; runtime sorting is allowed only for debug/visualization exports marked as not usable for main ordered-slot claims.
- `OrderedSlotGCSLoss` standalone defaults must keep `gcs_count_ce=1.0`, `gcs_interval=1.0`, `gcs_order=0.2`, `gcs_gt_bottom_order=1.0`, `gcs_decoded_bottom_order=1.0`, and `gcs_bottom_order_margin_px=2.0`. `gcs_order` is the common-anchor order loss only; GT-bottom and decoded-bottom order losses are separate terms that do not require common visible anchors. Do not let missing args silently disable count, interval, or order supervision.
- For `split=val`, official evaluation must use the canonical 363-image GT JSON by default. Noncanonical GT requires an explicit flag and must be marked not comparable to E1/spurious official-val evidence.
- Pass ordered-slot-only model args to `parse_model` only when `gcs_mode=ordered_slot`; query-mode construction must ignore those overrides.
- Do not track `datasets/`, generated runs, checkpoints, caches, or converted labels in Git.

Active source/config is restored to commit `86c8fb31c` (`Add GT4 GT5 weak
geometry rescue run`). Post-`86c8fb31c` mechanisms, scripts, YAMLs, tools,
reference banks, and launch protocols are rejected legacy records only and are
not available in the current code unless a future task explicitly restores
them. The active env30 boundary includes the branch-local `count_boundary_loss`,
train-only `gcs_hard_sampling`, E3-lite `gcs_spurious_neg`, training-time
`official_best`, ordered-slot protocol tooling, `valid_before_maxdet` decode
option, boundary-pseudo mask protocol tooling, GT5 short geometry weighting,
the default-off query Count Head ablation, and the default-off gated candidate
v2 experiment YAML/tooling.

## Main Files

Default active paths:

```text
data/tusimple_gcs_fixed_y_960x544.yaml
ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
```

Default-off candidate v2 path:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-gated-candidate-v2.yaml
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

The optional local short-segment proposal v4 YAML adds only:

```text
pred_short_segment_points: B x 12 x 404 x 56 x 2
pred_short_segment_logits: B x 12 x 404
pred_short_segment_x: B x 12 x 404 x 2
pred_short_segment_starts: 404
pred_short_segment_ends: 404
pred_short_segment_window_mask: 404 x 56
```

for query models built from
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v4.yaml`.
The default query YAML still emits no short-segment tensors and default decode
does not consume them.

The optional local short-segment selector/gate v5 YAML preserves the v4 outputs
and additionally emits:

```text
pred_short_segment_replace_logits: B x 12 x 404
```

for query models built from
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v5.yaml`.
The default query YAML and v4 YAML still emit no replace logits, and default
decode does not consume the v5 tensor.

The optional local short-segment proposal-local selector v6 YAML preserves the
same public output tensors as v5 and adds no new public prediction tensor. It
is enabled only by
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v6.yaml`.
Internally, the v6 GCS head sets `short_segment_local_evidence=True` so
score/replace logits receive local image samples and proposal geometry. The
default query YAML, v4 YAML, v5 YAML, and default decode remain unchanged.

The optional local short-segment dense-quality selector v7 YAML preserves the
same public output tensors as v6 and changes only the default-off loss/script
protocol for dense quality targets and combined-score hard diagnostics. It is
enabled only by
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v7.yaml`.

The optional local short-segment two-stage selector v8 YAML preserves the v6/v7
proposal geometry and candidate-quality score, disables the old per-candidate
replace head, and additionally emits:

```text
pred_short_segment_query_replace_logits: B x 12
```

It is enabled only by
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v8.yaml`.
Default decode does not consume the v8 tensor.

Legacy post-env30 record: the rejected query extent probe added:

```text
pred_start_logits: B x 12 x 56
pred_end_logits: B x 12 x 56
```

This YAML is not active after the 2026-07-28 env30 rollback. The default query
YAML still emits no query extent logits, and ordered-slot keeps its separate
`B x 5 x 56` interval contract.

Legacy post-env30 record: the rejected Q12 short local x-refine probes added:

```text
pred_coarse_points: B x 12 x 56 x 2
pred_short_refined_points: B x 12 x 56 x 2
pred_short_refine_delta_logits: B x 12 x 56  # v2 raw residual logits; v3 expected-offset proxy
pred_short_refine_delta_norm: B x 12 x 56
pred_short_refine_window_logits: B x 12 x 56 x 7  # v3 window-search YAML only
```

These YAMLs are not active after the 2026-07-28 env30 rollback. In those old
experiments, `pred_points` remained the env30 main geometry used by training
matching, normal losses, and normal decode. The default query YAML still emits
no short-local-refine diagnostic tensors.

The rejected v3 window-search variant was built from
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-short-local-refine-v3.yaml`.
It freezes the env30/base path through
`gcs_short_local_refine_freeze_base=True`, keeps non-refine BatchNorm
statistics fixed, trains only `query_short_local_refine_*`, and uses
identity-guarded supervision through
`gcs_short_local_refine_identity_guard=True`.

Legacy post-env30 record: the rejected Q12 dual-head probe added:

```text
pred_count_logits: B x 4
pred_quality_logits: B x 12
```

That YAML is not active after the 2026-07-28 env30 rollback.
`pred_count_logits` was for image-level 2/3/4/5 count CE and
`pred_quality_logits` was for query-level quality/ranking decode.

Legacy post-env30 record: the rejected protected static Q24 YAML changed only
query count:

```text
pred_points: B x 24 x 56 x 2
pred_logits: B x 24
pred_valid_logits: B x 24 x 56
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

This YAML and `data/gcs_reference_banks/q24_protected_static_env30_best.json`
are not active after the 2026-07-28 env30 rollback.

Legacy post-env30 record: the rejected Q24 role-containment probe added only
training-time constraints when explicitly enabled by `--gcs-role-contain` or
`--gcs-role-contain-matcher`. It is not active after the 2026-07-28 env30
rollback.

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
