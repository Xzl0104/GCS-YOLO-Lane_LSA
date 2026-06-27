# Current Contracts

This file records the active contracts for branch `codex/5-25-3-k56`.

## Branch Scope

This branch imports the historical `5-25-3.zip` algorithm and changes only the TuSimple fixed-y contract:

```text
legacy archive: Q=8, K=32, fixed_y=[0.98, 0.25]
this branch:    Q=12, K=56, fixed_y=[710/720, 160/720]
```

Do not silently import later mainline mechanisms such as Count Head, Quality Head, Survival Head, or near-miss mining into this branch unless a future task explicitly asks for that algorithm change. The branch now includes the 2026-06-27 user-requested, default-off `count_boundary_loss` for adjacent GT3/GT4/GT5 count-score boundaries; this is not a Count Head or decode change.

The branch also includes the 2026-06-27 user-requested, default-off `gcs_hard_sampling` train-only sampler for short-visible GT3/GT4/GT5 and 0601 samples. It changes only the training dataloader sampling frequency through `WeightedRandomSampler`; it does not change labels, validation/test dataloaders, point/smooth/curve losses, decode, or official metrics.

Training-time `official_best` checkpoint preservation is active as an explicit 2026-06-27 selection-protocol change. It preserves the 5-25-3 algorithm body and only changes how formal TuSimple checkpoints are selected.

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

Active source/config is rolled back to commit `b6535f641` (`Fix GCS training
progress header alignment`). Its algorithm contract remains the 5-25-3 K56
mainline: no later Count Head, Q18/Q20/dataref, duplicate/spurious/ranking,
lane-balanced, valid-repair, side-aux, legacy `gcs_gt4_short_boost` sampling,
`extra_exist_loss`, short matched existence floor, or count-confusion
diagnostic tooling is active. Later commits and notes are preserved only as
legacy experiment conclusions in the docs. They are not active CLI, loss,
model-output, tool, or config contracts in this code state.

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
point_valid_loss
smooth_loss
curve_loss
mask_loss
edge_loss
count_loss
count_under5_loss
count_boundary_loss
count_score_mean
```

`count_boundary_loss` is disabled by default through `gcs_count_boundary=0.0`.
When enabled, it applies to `sum(sigmoid(pred_logits))` with GT3 upper, GT4
lower/upper, and GT5 lower boundaries. `count_score_mean` is a log-only
diagnostic and is not part of the weighted training objective.

## Decode And Evaluation Contract

Decode must use real query predictions only, must not use GT during inference, and must not fabricate lanes. Final output should be sorted from left to right by bottom visible x.

The branch includes a default-off count-aware top-k postprocess ablation for
inference/evaluation only. When explicitly enabled with `--count-aware-topk`,
decode uses `sum(sigmoid(pred_logits))` to choose a dynamic final lane count
and keeps the quality-best post-conf, post-NMS lanes. This does not change
training, labels, losses, model outputs, official metrics, Count Head, Quality
Head, Survival Head, or default decode behavior.

This branch includes `tools/eval_tusimple_official.py`, `tools/sweep_tusimple_official.py`, `gcs_tools/tusimple_official_eval.py`, and explicit training-time `official_best` checkpoint preservation. It does not include `tools/diagnose_tusimple_count_confusion.py`, `tools/diagnose_gcs_gt5.py`, or later mainline Count/Quality/Boundary diagnostics unless a future task explicitly ports them. Use official-val for selection and test only once for final evaluation.

Formal TuSimple checkpoint selection must use `official_acc` first, then `official_score`, then lower `official_FP`, lower `official_FN`, and higher `count_acc_4`. Do not select the final checkpoint only by `val/total_loss`, internal `val/f1`, or generic `best.pt`.
