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

Active experimental K56 data YAML:

```text
data/tusimple_gcs_fixed_y_k56_960x544.yaml
```

Active experimental K56 data root:

```text
datasets/tusimple_fixed_y_k56_960x544
```

The K56 dataset must be rebuilt from original TuSimple JSON and images. Do not resample existing K32 labels into K56 labels.

Current K56 experiment status:

```text
K56 label oracle: 0.998256
K56 parent official_best: 0.959315 at epoch 152
K56 best val-only min-points row: 0.959750 at point_valid_thr=0.40, candidate_min_points=5, final_min_points=9, fifth_min_points=4
```

The min-points row is validation-selected and not promoted because it carries GT4-to-5 false fifth-lane risk (`count_acc_4=0.863636`). The rejected K56 gates are `gcs_yolo_lane_s_q12_k56_cqcalib_ft12_seed1_b32w4`, `gcs_yolo_lane_s_q12_k56_cqcalib_lr1e4_ft8_seed1_b32w4`, `gcs_yolo_lane_s_q12_k56_curveaux_ft8_seed1_b32w4`, `gcs_yolo_lane_s_q12_k56_lowfp_joint_ft8_seed1_b32w4`, `gcs_yolo_lane_s_q12_k56_countadj_lowmargin_ft8_seed1_b32w4`, and `gcs_yolo_lane_s_q12_k56_countff_supp_ft8_seed1_b32w4`; do not rerun or continue those exact recipes as the next path.

## Model Contract

Default model:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml
```

Legacy Q=8 config is retained for historical reproduction, ablation, or controlled experimental candidates.

Default-off K56 fifth-candidate verifier experiment:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-fifthness-v1.yaml
```

This model YAML is opt-in and enables an optional fifthness verifier head plus Count Head fifth-candidate evidence. It is not the K56 default. The first `gcs_yolo_lane_s_q12_k56_fifthness_v1_ft8_seed1_b32w4` short gate is rejected: best official-val was epoch 5 `0.959006`, below the K56 parent `0.959315`, with worse FP/FN and high GT4-to-5 pressure. The follow-up `gcs_yolo_lane_s_q12_k56_fifthness_gt5neg_ft8_seed1_b32w4` gate is also not promotable: best official-val was epoch 6 `0.959319`, only `+0.000004` over parent, but FP worsened to `0.047429` and `rate_4_to_5` rose to `0.121212`. A 2026-06-15 official-val fifthness score audit found useful but insufficient separation: GT5 selected rank-5 matched scores had median `0.960218`, while GT4 unmatched false-fifth scores had median `0.841094` and max `0.942222`; thresholds low enough to retain GT5 keep too many false fifths, and high thresholds reproduce the GT5 `5->4` tradeoff. The enhanced audit CSV also records Count Head `P4/P5/margin`; on the same server rerun, GT4 unmatched false-fifth samples had median `P5=0.996325` and median margin `0.992650`, indicating many false fifths are Count Head count=5 overconfidence cases. The follow-up default-K56 adjacent Count margin gate `gcs_yolo_lane_s_q12_k56_countadj_lowmargin_ft8_seed1_b32w4` was stopped after epoch 4 and is rejected: independent official-val reproduced `0.958843`, below parent `0.959315`, and `rate_4_to_5=0.090909` worsened versus parent `0.075758`. The candidate-specific Count false-fifth suppression gate `gcs_yolo_lane_s_q12_k56_countff_supp_ft8_seed1_b32w4` completed 8 epochs and is also not promotable: official_best epoch 7 reached `0.959496`, but FP worsened to `0.047062`, FN to `0.031680`, and GT5 `rate_5_to_4` to `0.175676`; the more balanced epoch 4 row was only `+0.000057` ACC over parent. Do not start full/e180 from these checkpoints.

Default-off K56 Count Head fifth-candidate evidence isolation experiment:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count5ev-v1.yaml
```

This model YAML is opt-in and enables Count Head fifth-candidate evidence without enabling the fifthness verifier head. It must preserve the normal six-output contract and must not emit `pred_fifthness_logits`. Its purpose is to isolate whether direct fifth-candidate evidence in the Count Head helps GT4/GT5 count calibration, because the earlier fifthness-v1 gates mixed Count Head evidence with an auxiliary fifthness output and fifthness training losses.
The first remote FT8 attempt `gcs_yolo_lane_s_q12_k56_count5ev_v1_ft8_seed1_b32w4` is incomplete and not promotable: the process stopped after epoch 3/8 while the server root filesystem had only about `687M` free. Partial official-val rows stayed below the K56 parent (`epoch1=0.958453`, `epoch2=0.957082`, `epoch3=0.958500` versus parent `0.959315`), and epoch 3 worsened `rate_4_to_5` to `0.090909`. Do not start full/e180 from this run; rerun the FT8 gate only after freeing server disk space.

## Label Contract

Current label mode:

```text
point_mode = fixed_y
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end = 0.25
K = 32
```

Active experimental K56 label mode:

```text
point_mode = fixed_y
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end = 160 / 720 = 0.2222222222222222
K = 56
```

The K56 anchors align exactly to TuSimple official h-samples from `710` down to `160` at step `10`, normalized by original height `720`.

Exact K56 label artifacts should pass:

```bash
python tools/check_gcs_label_order_split.py \
  --dataset-root datasets/tusimple_fixed_y_k56_960x544 \
  --expect-fixed-y 56,710/720,160/720
```

This check validates the fixed-y anchor contract directly; it is not a substitute for rebuilding labels from raw TuSimple JSON/images.

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

The opt-in `gcs-yolo-lane-s-q12-k56-fifthness-v1.yaml` experiment may additionally emit:

```text
pred_fifthness_logits: B x Q
```

Default K32 and K56 model configs must not emit `pred_fifthness_logits`; `tools/check_model.py` treats it as optional only when the head enables `use_fifthness`.
The opt-in `gcs-yolo-lane-s-q12-k56-count5ev-v1.yaml` config also must not emit `pred_fifthness_logits`; it only enables Count Head fifth-candidate evidence.

The optional fifthness logits are ignored by decode unless the explicit default-off decode switch is enabled. When enabled, fifthness may only gate or re-rank the selected fifth lane and fifth-lane rescue candidates; selected ranks 1-4 keep the default `exist * visible_segment_mean_valid * visible_support_score` ordering.

The candidate-aware Count Head uses the same short-lane visibility semantics as decode when building count evidence:

```text
visible_lane_quality = exist_score * visible_segment_mean_valid * visible_support_score
visible_support_score = min(1, visible_segment_points / 12)
```

`visible_segment_mean_valid` is computed on the longest contiguous segment with point-valid probability at least `0.5`. This visible-lane quality drives Count Head top-query selection and top4/top5 cardinality evidence so short but reliable TuSimple edge lanes are not structurally suppressed by the all-anchor mean. The all-anchor point-valid mean remains available as an auxiliary aggregate feature; it must not be used as the primary fifth-lane evidence.

## Loss Contract

Default mainline loss items:

```text
exist_loss
point_loss
point_valid_loss
line_iou_loss
curvature_loss
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

The GT5 candidate-quality knobs above are training-side only. They strengthen real matched query supervision inside the existing logged loss items:

- `gcs_count_boundary_gt5_pos_weight` weights the Count Boundary `count>=5` positive target inside `count_cls_loss`.
- `gcs_candidate_gt5_edge_weight` weights matched left/right GT5 edge queries/lanes inside `exist_loss`, `point_loss`, `point_valid_loss`, `line_iou_loss`, and `quality_loss`; it is matched edge-query/lane weighting, not per-anchor positive-target-only weighting.
- `gcs_point_valid_gt5_edge_continuity` adds a small adjacent-anchor continuity penalty inside `point_valid_loss`.
- `gcs_hard_edge_loss_terms` defaults to `exist,point,point_valid,line_iou`. `quality` is also a supported explicit term for controlled experiments, but it is not in the default list.

They do not change decode, do not use GT during inference/decode, and do not fabricate lanes.

`gcs_soft_count_decision`, `gcs_last_lane_rescue`, and `gcs_edge_last_lane_rescue` remain default-off unless selected by official-val evidence.

Current default-off training-side experimental knobs:

```text
gcs_count_adjacent_margin = 0.2
gcs_count_adjacent_margin_gain = 0.0
gcs_count_adjacent_margin_gt45_weight = 1.0
gcs_quality_gt5_edge_floor = 0.0
gcs_quality_hard_negative_from_head = False
gcs_hard_negative_visible_segment = False
gcs_hard_negative_visible_thr = 0.5
gcs_hard_negative_visible_support_points = 12.0
gcs_point_valid_gt5_edge_segment = 0.0
gcs_point_valid_gt5_edge_segment_thr = 0.65
gcs_point_valid_gt5_edge_segment_min_points = 5
gcs_geometry_curvature = 0.0
gcs_geometry_curvature_beta_px = 5.0
gcs_quality_pairwise = 0.0
gcs_quality_pairwise_margin = 0.2
gcs_fifthness = 0.0
gcs_fifthness_pairwise = 0.0
gcs_fifthness_margin = 0.2
gcs_fifthness_negative_topk = 2
gcs_fifthness_negative_score_thr = 0.1
gcs_fifthness_include_gt5_negatives = False
gcs_count_cumulative = 0.0
gcs_count_cumulative_label_smoothing = 0.0
```

Current default-off inference/decode experimental knobs:

```text
gcs_use_fifthness_decode = False
gcs_fifthness_decode_thr = 0.0
gcs_fifthness_decode_rank_weight = 1.0
```

When `gcs_use_fifthness_decode=True`, `decode_gcs_predictions()` requires `pred_fifthness_logits` and fails fast if the model does not emit them. Historical official-val selection summaries that do not record these fields are treated as selecting the old defaults (`False/0.0/1.0`) for final-test provenance checks.

`gcs_count_adjacent_margin_gain` enables a default-off training-side margin term inside `count_cls_loss` that pushes the GT count logit above neighboring count classes. It is intended for controlled GT3/GT4/GT5 calibration experiments and does not add a new logged loss item.

The removed `gcs_count_false_fifth_suppression*` source switches belonged only to the rejected `gcs_yolo_lane_s_q12_k56_countff_supp_ft8_seed1_b32w4` gate. Keep that run as audit history, but do not treat the switches as available current-code knobs.


`gcs_quality_gt5_edge_floor` is a default-off training-side candidate that floors matched Quality Head targets only for real left/right edge lanes in GT5 images. It is intended to test whether true short GT5 edge lanes are being assigned quality targets too low to survive quality-gated fifth-lane decode behavior.

The `gcs_quality_gt5_edge_floor`, `gcs_quality_hard_negative_from_head`, `gcs_hard_negative_visible_segment*`, and `gcs_point_valid_gt5_edge_segment*` knobs are intended for controlled GT5 quality experiments. They do not change decode, read GT during inference, fabricate lanes, or alter official metrics.

`gcs_geometry_curvature` is a default-off fixed-y geometry auxiliary candidate. When enabled, it adds `curvature_loss`, a SmoothL1 penalty on second-order x curvature for Hungarian-matched left/right edge lanes in GT>=5 images only. It is training-side only and does not change decode, read GT during inference, fabricate lanes, or alter official metrics.

The first K56 curvature gate `gcs_yolo_lane_s_q12_k56_curveaux_ft8_seed1_b32w4` is rejected: best official-val was `0.958732`, below the K56 parent `0.959315`. Keep the infrastructure default-off and do not rerun the exact `gcs_geometry_curvature=0.05` recipe as the next path.

`gcs_fifthness*` is a default-off fifth-candidate verifier objective. It supervises GT5 edge matched lanes as positives and competitive unmatched outside candidates as negatives, with an optional pairwise margin term. By default, competitive fifthness negatives are mined from GT3/GT4 images. The additional `gcs_fifthness_include_gt5_negatives` switch is default-off and, when enabled, also mines unmatched outside candidates in GT5 images so same-image false fifth candidates are explicitly ranked below true GT5 edge matches. Loss-side GT lane counting, Count Head targets, Count Boundary targets, Count Sum targets, hard-loss lane-count filters, and GT5 point-valid boosting all use `gcs_count_min_gt_points` consistently. Current generated K56 labels, dataset loading, label-oracle conversion, and TuSimple prediction conversion still require at least 2 valid anchors per lane, so full one-anchor lane support is not yet an active data/decode contract. It requires a fifthness-enabled model YAML; enabling the loss against a default model is an error.

`gcs_quality_pairwise*` is a default-off competitive ranking term for the existing Quality Head. It does not rewrite the current Quality target; it only adds a pairwise constraint between GT5 edge matches and competitive false fifth candidates.

`gcs_count_cumulative*` is a default-off ordinal-style cumulative count>=3/count>=4/count>=5 supervision term built from the existing `pred_count_logits: B x 4`. It preserves `pred_count_logits` and `pred_count_boundary_logits` output shapes and does not add a new Count Head contract.

For logging stability, these experimental terms stay folded into existing loss items instead of adding new CSV columns: count cumulative is inside `count_cls_loss`, Quality pairwise is inside `quality_loss` and is scaled by the existing `gcs_quality` gain, while fifthness is an independent auxiliary term reported through the existing `quality_loss` item. Analyze enabled runs with the exact CLI gains from `args.yaml`; do not infer subterm magnitudes from the 8-loss CSV alone.

When `gcs_quality_hard_negative_from_head` is enabled, Quality Head hard negatives are mined from unmatched queries only. Hungarian-matched queries remain matched quality targets even when their current continuous quality target is `0.0`; they must not be reclassified as hard negatives.

When `gcs_hard_negative_visible_segment` is enabled, the shared unmatched hard-negative mask uses the same longest-visible-segment support semantics as Count/decode evidence instead of the all-anchor point-valid mean. It still mines unmatched queries only; Hungarian-matched queries remain protected.

These knobs remain default-off after the 2026-06-13 `gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0` and `gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0` official-val gates; neither recipe is promoted to mainline defaults.

## Experimental Loss Policy

The current 8-loss logging setup is a default baseline, not a permanent restriction. `curvature_loss` is inert unless its gain is enabled.

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

`tools/sweep_tusimple_official.py` supports validation-only grids for `candidate_min_points`, `final_min_points`, and `fifth_min_points`. K56 must retune these on official-val because 56 anchors make each point correspond to one official 10px h-sample; K32 min-points semantics do not carry over unchanged.

`tools/sweep_tusimple_official.py` and `tools/diagnose_gcs_gt5.py` default to `--split val` and reject `--split test`. Training-time `official_best` selection also rejects `split=test`.
These selection/diagnostic paths also reject conventional TuSimple test GT json paths and explicit GT records that resolve to `test_set` images, so `--split val --gt-json <test labels>` cannot be used as a test-leakage bypass. Training-time `official_best` selection is val-only; do not use train or test split for checkpoint selection.

Test is only for one-shot final evaluation of a candidate already selected on official-val, using `tools/eval_tusimple_official.py --split test --selection-summary <official-val-summary>`. The final-test command must prove the weights and postprocess parameters match the selected official-val row. Extra user-requested test audits must pass `--diagnostic-only-test`, and their outputs are marked `diagnostic_only_test=true` and `not_for_selection=true`.

If the user explicitly requests extra test evaluations for audit purposes, label them diagnostic-only and do not use them to choose checkpoints, thresholds, postprocess settings, losses, model variants, or promotion. The 2026-06-14 user-requested K56 test audit is such a diagnostic-only exception; it does not create a final/promotable official-test claim for K56.
