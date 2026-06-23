# Current Contracts

This file records the active contracts for branch `codex/5-25-3-k56`.

## Branch Scope

This branch imports the historical `5-25-3.zip` algorithm and changes only the TuSimple fixed-y contract:

```text
legacy archive: Q=8, K=32, fixed_y=[0.98, 0.25]
this branch:    Q=12, K=56, fixed_y=[710/720, 160/720]
```

Do not silently import later mainline mechanisms such as Count Head, Count Boundary, Quality Head, Survival Head, near-miss mining, or official-best checkpoint preservation into this branch unless a future task explicitly asks for that algorithm change.

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

All branch configs keep `Q=12`, `K=56`, fixed-y anchors `710/720 -> 160/720`, and `--imgsz 544 960`.

Historical q12-k56 experiment docs are old records. Preserve them, but do not let them override the active 5-25-3 K56 mainline contract.

Active source/config is based on rollback commit `50999d6af` (`Document 5-25-3
K56 as mainline`) plus default-disabled experiment knobs for
`duplicate_margin_loss`, `spurious_margin_loss`, `lane_balanced_point_loss`, and
`short_valid_recall_loss`, plus the later default-disabled
`far_spurious_survival_loss` and `gt5_rank_consistency_loss` experiment knobs.
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
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

With the default K56 model this means:

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
```

## Loss Contract

Default logged loss items on this branch:

```text
exist_loss
point_loss
lane_balanced_point_loss
point_valid_loss
short_valid_recall_loss
smooth_loss
curve_loss
mask_loss
edge_loss
count_loss
count_under5_loss
duplicate_margin_loss
spurious_margin_loss
far_spurious_survival_loss
gt5_rank_consistency_loss
```

`duplicate_margin_loss`, `spurious_margin_loss`, `lane_balanced_point_loss`,
`short_valid_recall_loss`, `far_spurious_survival_loss`, and
`gt5_rank_consistency_loss` are default-disabled experimental log items. With
the default gains they contribute `0` to the training objective; enabling any of
them is an explicit experiment contract change.

## Decode And Evaluation Contract

Decode must use real query predictions only, must not use GT during inference, and must not fabricate lanes. Final output should be sorted from left to right by bottom visible x.

This branch includes `tools/eval_tusimple_official.py`, `tools/sweep_tusimple_official.py`, and `gcs_tools/tusimple_official_eval.py`. It does not include `tools/diagnose_tusimple_count_confusion.py`, `tools/diagnose_gcs_gt5.py`, training-time `official_best` checkpoint preservation, or later mainline Count/Quality/Boundary diagnostics unless a future task explicitly ports them. Use official-val for selection and test only once for final evaluation.
