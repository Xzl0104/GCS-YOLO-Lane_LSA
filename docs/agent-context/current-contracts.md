# Current Contracts

This file records active GCS-YOLO-Lane contracts. It is the first reference for current behavior.

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

Default data root:

```text
datasets/tusimple_fixed_y_960x544
```

The test split must not participate in training split rebuilds, threshold search, checkpoint selection, or postprocess tuning.

## Model Contract

Default model:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml
```

Legacy Q=8 config is retained for historical reproduction, ablation, or controlled experimental candidates.

## Label Contract

Current label mode:

```text
point_mode = fixed_y
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end = 0.25
K = 32
```

Expected fixed-y label fields:

```text
lanes
lane_valid
num_lanes
point_mode
fixed_y
num_points
raw_file
image_shape
```

Do not silently mix old `0.98` fixed-y labels with current `710/720` labels.

## Output Contract

The model output must include:

```text
pred_points: B x Q x K x 2
pred_logits: B x Q
pred_valid_logits: B x Q x K
pred_quality_logits: B x Q
pred_count_logits: B x 4
pred_count_boundary_logits: B x 2
```

`pred_count_logits` is the image-level Count Head for classes count=2/3/4/5. `pred_quality_logits` is lane-level quality for training, diagnostics, and gated rescue behavior.
`pred_count_boundary_logits` is the count>=4/count>=5 boundary calibration sub-head used by the default Count Head loss/decode path.

## Loss Contract

Default mainline loss items:

```text
exist_loss
point_loss
point_valid_loss
line_iou_loss
count_cls_loss
count_sum_loss
quality_loss
```

Training, validation, and CSV loss logging should keep these items explicit.

## Mainline Count And Quality Defaults

Current conservative count-generalization defaults:

```text
gcs_count_sum = 0.03
gcs_quality = 0.4
gcs_quality_neg_weight = 0.5
gcs_count_cls_w2/w3/w4/w5 = 0.5/1.2/1.4/1.8
gcs_count_boundary_gt5_pos_weight = 1.15
gcs_point_valid_gt5_pos_weight = 2.0
gcs_gt5_edge_loss_weight = 1.15
gcs_candidate_gt5_edge_weight = 1.10
gcs_point_valid_gt5_edge_continuity = 0.05
gcs_point_valid_gt5_edge_continuity_thr = 0.55
gcs_gt5_oversample_weight = 1.0
gcs_group_sampler_ratios = 2:0.01,3:0.29,4:0.42,5:0.28
```

The GT5 candidate-quality knobs above are training-side only. They strengthen real matched query supervision inside the existing 7 loss items:

- `gcs_count_boundary_gt5_pos_weight` weights the Count Boundary `count>=5` positive target inside `count_cls_loss`.
- `gcs_candidate_gt5_edge_weight` weights matched left/right GT5 edge queries/lanes inside `exist_loss`, `point_loss`, `point_valid_loss`, `line_iou_loss`, and `quality_loss`; it is matched edge-query/lane weighting, not per-anchor positive-target-only weighting.
- `gcs_point_valid_gt5_edge_continuity` adds a small adjacent-anchor continuity penalty inside `point_valid_loss`.
- `gcs_hard_edge_loss_terms` defaults to `exist,point,point_valid,line_iou`. `quality` is also a supported explicit term for controlled experiments, but it is not in the default list.

They do not change decode, do not use GT during inference/decode, and do not fabricate lanes.

`gcs_soft_count_decision`, `gcs_last_lane_rescue`, and `gcs_edge_last_lane_rescue` remain default-off unless selected by official-val evidence.

Current default-off training-side experimental knobs:

```text
gcs_quality_hard_negative_from_head = False
gcs_point_valid_gt5_edge_segment = 0.0
gcs_point_valid_gt5_edge_segment_thr = 0.65
gcs_point_valid_gt5_edge_segment_min_points = 5
```

These are intended for controlled GT5 segment-quality experiments. They do not change decode, read GT during inference, fabricate lanes, or alter official metrics.

When `gcs_quality_hard_negative_from_head` is enabled, Quality Head hard negatives are mined from unmatched queries only. Hungarian-matched queries remain matched quality targets even when their current continuous quality target is `0.0`; they must not be reclassified as hard negatives.

## Experimental Loss Policy

The current 7-loss setup is a default baseline, not a permanent restriction.

Previously removed losses or modules may be restored as controlled experimental candidates if the goal is to improve official ACC.

When restoring or adding a module:

- make it explicit
- make it configurable
- make it traceable
- document whether it is baseline or experimental
- do not silently mix it into the default mainline
- define an official-val comparison plan

## Decode Contract

Default decode behavior:

- Count Head Top-K determines the final policy K after explicit count-boundary calibration when boundary logits are present.
- Candidate ranking must be explicit and traceable.
- Default candidate ranking uses:

```text
rank_score = exist_score * visible_segment_mean_valid * visible_support_score
```

where `visible_segment_mean_valid` is the mean point-valid probability on the longest contiguous visible segment that passes the active visible-anchor floor, and `visible_support_score = min(1, visible_segment_points / 12)`. `mean_valid_score_all` remains diagnostic-only metadata for the all-anchor mean. This avoids structurally suppressing short but reliable TuSimple edge lanes.
- Quality Head may be used for quality loss, diagnostics, and rescue gates.
- Quality Head should not silently override the intended ranking policy unless that is an explicit experimental candidate.
- Rescue may only use real query candidates.
- Rescue must not read GT.
- Rescue must not fabricate lanes.
- Final output should be sorted from left to right by bottom visible x.

## Evaluation Contract

Default selection:

```text
official_best.pt is selected by official_acc.
```

`weights/best.pt` remains the ordinary validation-fitness checkpoint. `weights/official_best.pt` is maintained only by TuSimple official-val `official_acc`.

`gcs_official_best_top_k` defaults to `1`. When set above `1`, training preserves the top-K official-val checkpoint candidates under `weights/official_topk/` and records them in `official_best_summary.json`. This is checkpoint preservation only; it does not change decode behavior, use diagnostics as selection tie-breakers, or allow test-driven selection.

Diagnostic metrics include:

```text
official_score
FP
FN
count_acc_3
count_acc_4
count_acc_5
gt5_output5_rate
gt5_count_head_under_rate
gt5_valid_points_fail_rate
candidate_pool_shortfall
rescue_precision
```

These are diagnostics unless explicitly promoted into a controlled experimental objective.

## Test Protection Contract

The test set must not be used for threshold search, rescue parameter search, soft-count search, rank-min-points search, final/fifth min-points search, NMS distance search, checkpoint selection, model design iteration, or loss-weight tuning.

`tools/sweep_tusimple_official.py` and `tools/diagnose_gcs_gt5.py` default to `--split val` and reject `--split test`. Training-time `official_best` selection also rejects `split=test`.

Test is only for one-shot final evaluation of a candidate already selected on official-val, using `tools/eval_tusimple_official.py --split test`.
