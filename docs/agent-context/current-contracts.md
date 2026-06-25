# Current Contracts

This file records the active contracts for branch `codex/5-25-3-k56`.

## Branch Scope

This branch imports the historical `5-25-3.zip` algorithm and changes only the TuSimple fixed-y contract:

```text
legacy archive: Q=8, K=32, fixed_y=[0.98, 0.25]
this branch:    Q=12, K=56, fixed_y=[710/720, 160/720]
```

Do not silently import later mainline mechanisms such as Count Boundary, Quality Head, Survival Head, near-miss mining, or official-best checkpoint preservation into this branch unless a future task explicitly asks for that algorithm change.

The q18-k56-gt4-candidate-countguard task explicitly adds a branch-local Q18
side-dense model config and an explicit 3/4/5 `pred_count_logits` count head.
The count-head CE objective is controlled by `gcs_count_ce` and remains
default-disabled at `0.0`.

## Input Contract

TuSimple uses:

```bash
--imgsz 544 960
```

This is H,W order. Do not reverse it.

## Data Contract

Default data YAML:

```text
data/tusimple_gcs_fixed_y_960x544.yaml
```

`data/tusimple_gcs_fixed_y_k56_960x544.yaml` is a compatibility path for old q12-k56 records and has the same K56 contract.

Default data root:

```text
datasets/tusimple_fixed_y_k56_960x544
```

The K56 dataset must be rebuilt from original TuSimple JSON and images, not resampled from historical K32 labels.

## Model Contract

Default model:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
```

Compatibility configs also use the same K56 fixed-y contract:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-fixed-y.yaml
```

All default branch configs keep `Q=12`, `K=56`, fixed-y anchors
`710/720 -> 160/720`, and `--imgsz 544 960`.
The branch-local Q18 experiment config is:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q18-k56-side.yaml
```

It keeps `K=56`, fixed-y anchors `710/720 -> 160/720`, and `--imgsz 544 960`,
but sets `Q=18` with side-dense bottom query references.

Historical q12-k56 experiment docs are old records. Preserve them, but do not let them override the active 5-25-3 K56 mainline contract.

Active source/config is based on rollback commit `50999d6af` (`Document 5-25-3
K56 as mainline`) plus default-disabled experiment knobs for
`duplicate_margin_loss`, `spurious_margin_loss`, `lane_balanced_point_loss`,
`short_valid_recall_loss`, `far_spurious_survival_loss`,
`gt5_rank_consistency_loss`, `gt3_extra_survival_loss`, and
`gt4_lane_balanced_point_loss` experiment knobs, plus the train-only
`gcs_gt4_sample_gain` sampler knob. It also includes default-off GT4
short-lane candidate-recall knobs for replacing the base x point-loss
reduction with lane-balanced reduction and for adding GT4-short endpoint
matching cost:
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
Later mechanisms such as GT4 short-lane sampling, `extra_exist_loss`, short
matched existence floor, and count-confusion diagnostic tooling are preserved
only as legacy experiment conclusions in the docs. They are not active CLI,
loss, or tool contracts in this code state.

## Label Contract

```text
point_mode = fixed_y
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end = 160 / 720 = 0.2222222222222222
K = 56
```

The K56 anchors align exactly to TuSimple official h-samples from `710` down to `160`, descending by `10` pixels and normalized by original height `720`.

Expected fixed-y label fields:

```text
semantic_mask
edge_mask
lanes
lane_valid
num_lanes
point_mode
fixed_y
num_points
raw_file
image_shape
```

Do not mix old `0.98`/K32 fixed-y labels with this branch.

## Output Contract

This 5-25-3 branch model output must include:

```text
pred_points: B x Q x K x 2
pred_logits: B x Q
pred_valid_logits: B x Q x K
pred_count_logits: B x 3
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

With the default K56 model this means:

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
pred_count_logits: B x 3
```

## Loss Contract

Default logged loss items on this branch:

```text
exist_loss
point_loss
lane_balanced_point_loss
gt4_short_lane_loss
gt4_lane_balanced_point_loss
point_valid_loss
short_valid_recall_loss
gt4_short_valid_recall_loss
gt4_short_valid_count_floor_loss
valid_lb_gt4_short_count
valid_lb_gt4_short_gt_points_mean
valid_lb_gt4_short_pred_prob_mean
valid_lb_gt4_short_pred_sum_mean
unmatched_valid_neg_loss
unmatched_valid_query_count
unmatched_valid_prob_mean
smooth_loss
curve_loss
mask_loss
edge_loss
count_loss
count_under5_loss
count_ce_loss
count_ce_acc
duplicate_margin_loss
spurious_margin_loss
far_spurious_survival_loss
gt5_rank_consistency_loss
gt3_extra_survival_loss
gt4_short_lane_valid_points_mean
gt4_short_lane_count
gt4_short_valid_lane_count
gt4_short_gt_valid_points_mean
gt4_short_pred_valid_prob_mean
gt4_short_pred_valid_sum_mean
```

`duplicate_margin_loss`, `spurious_margin_loss`, `lane_balanced_point_loss`,
`short_valid_recall_loss`, `far_spurious_survival_loss`, and
`gt5_rank_consistency_loss`, `gt3_extra_survival_loss`, and
`gt4_lane_balanced_point_loss`, `gt4_short_valid_recall_loss`,
`gt4_short_valid_count_floor_loss`, and `count_ce_loss` are default-disabled
experimental log items.
With the default gains they contribute `0` to the training objective; enabling
any of them is an explicit experiment contract change.

`gcs_count_ce` is a default-disabled explicit count-head CE gain. When enabled,
`pred_count_logits` is trained as a three-class classifier for GT lane counts
`3`, `4`, and `5`, with target classes computed as
`clamp(num_lanes, 3, 5) - 3`. `count_ce_acc` is a diagnostic log item and does
not affect loss scaling.

For the q18-k56-gt4-candidate-countguard dry-run acceptance, check these
training progress fields:
`cnt_ce`, `cnt_acc`, `gt4_pt`, `gt4_lbp`, `uvneg_loss`, `uvneg_prob`,
`vlb_vprob`, and `vlb_vsum`. Do not require `gt4_vprob` or `gt4_vsum` to be
non-zero in that dry-run: those columns belong to the disabled
`gcs_gt4_short_valid_recall` / `gcs_gt4_short_valid_count_floor` paths. When
both of those flags are disabled, non-zero `gt4_vprob` or `gt4_vsum` should be
treated as a flag/verification issue.

`gcs_lane_balanced_point_loss` is a boolean switch, default `false`. When it is
`true`, the base `point_loss` uses x-only per-lane reduction instead of the
legacy valid-point pooled reduction. In GT4 images, lanes with visible points
`<= gcs_gt4_short_lane_max_points` are multiplied by
`gcs_gt4_short_lane_weight`. The three `gt4_short_lane_*` fields are diagnostic
training/validation log items and do not add independent objective terms.
In the training progress table, `lane_bal` reports the active lane-balanced
base point-loss value when `gcs_lane_balanced_point_loss=true`; `gt4_pt`
reports `gt4_short_lane_loss`; `gt4_lbp` reports the separate
`gt4_lane_balanced_point_loss` experiment item. When
`gcs_gt4_lane_balanced_point=0.0`, the progress header marks that column as
`gt4lbp_off`; when `gcs_short_valid_recall=0.0`, it marks the short valid
recall column as `short_off`. Those disabled columns are expected to be
zero and do not contribute to `total_loss`.

`gcs_lane_balanced_valid_loss` is a boolean switch, default `false`. When it is
`true`, the base `point_valid_loss` replaces the legacy global point-valid BCE
reduction with a matched-lane reduction: BCE is computed with
`reduction="none"`, each matched lane is averaged over all `K` anchors so
invalid anchors stay supervised, and then lanes are averaged. Matched lanes
from images whose true GT lane count is `4` and whose GT-visible count is
`<= gcs_gt4_short_valid_max_points` receive
`gcs_gt4_short_valid_lane_weight`; `gcs_gt4_short_valid_pos_weight` is an
optional positive-anchor multiplier inside this replacement loss. The
replacement also retains the legacy unmatched-query negative supervision:
queries not selected by the Hungarian matcher receive zero-target point-valid
BCE as `softplus(valid_logit)`, reduced by averaging over K anchors per query
and then over unmatched queries, with
`gcs_unmatched_valid_neg_weight=0.5` by default. The matched lane-balanced loss
and unmatched negative loss are combined inside `point_valid_loss` as
`(matched + weight * unmatched) / (1 + weight)`. This switch does not change
model structure, matching output shape, decoder, NMS, sampler, official
metrics, or test-selection rules. The `valid_lb_*` and `unmatched_valid_*`
loss items are diagnostic log fields for this replacement path.

`gcs_gt4_short_valid_recall` is a boolean switch, default `false`. When
enabled, it adds a positive-only `softplus(-valid_logit)` loss on GT-valid
points of Hungarian-matched lanes whose current image has true GT lane count
`4` and whose GT visible-point count is between `2` and
`gcs_gt4_short_valid_max_points` inclusive. `gcs_gt4_short_valid_count_floor`
is a separate default-off follow-up that applies a light hinge when the
predicted valid probability sum over the GT-valid region falls below
`min(valid_points, max(gcs_gt4_short_valid_count_floor_min,
gcs_gt4_short_valid_count_floor_ratio * valid_points))`. Neither option
changes model structure, matching output shape, decoder, NMS, sampler,
official metrics, or test-selection rules.

`gcs_gt4_short_match_endpoint` is default `0.0`; with the default value the
Hungarian matcher is unchanged. When enabled, it adds normalized-x endpoint
cost only for GT4 GT lanes whose visible-point count is
`<= gcs_gt4_short_match_max_points`.

## Decode And Evaluation Contract

Decode must use real query predictions only, must not use GT during inference, and must not fabricate lanes. Final output should be sorted from left to right by bottom visible x.

This branch includes `tools/eval_tusimple_official.py`,
`tools/sweep_tusimple_official.py`, `gcs_tools/tusimple_official_eval.py`,
and the self-contained train/val query-trace helper
`tools/diagnose_gt4_short_failure_queries.py`, plus diagnostic-only
`tools/diagnose_gt4_missing_lane_raw_queries.py` and
`tools/build_gt4_hard_val_split.py` for internal GT4-hard validation lists. It
does not include
`tools/diagnose_tusimple_count_confusion.py`, `tools/diagnose_gcs_gt5.py`,
training-time `official_best` checkpoint preservation, or later mainline
Count/Quality/Boundary diagnostics unless a future task explicitly ports them.
Use official-val for selection and test only once for final evaluation.

`tools/sweep_tusimple_official.py` supports normal official-val rows and
count-head guided top-K rows. With `--count-guided-topk`, every threshold base
combo emits one `mode=normal` row and one `mode=count_guided_topk` row per
`--count-guided-min-probs` value. Count-guided sweep rows require count-head
logits by default; `--count-guided-allow-unsupported-fallback` must be passed
explicitly to allow fallback rows when count logits are unavailable.
