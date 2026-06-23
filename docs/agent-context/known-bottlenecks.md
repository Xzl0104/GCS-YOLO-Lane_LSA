# Known Bottlenecks

This file applies to branch `codex/5-25-3-k56`.

## Branch Scope

The current mainline imports the historical `5-25-3.zip` algorithm and changes only the TuSimple fixed-y contract to Q=12/K=56 with official h-sample anchors.

Do not read mainline Count Head, Count Boundary, Quality Head, Survival Head, near-miss, or official-best bottlenecks as active branch behavior. Those mechanisms are not part of this 5-25-3 branch.

Active source/config is based on rollback commit `50999d6af` (`Document 5-25-3
K56 as mainline`) plus the default-disabled `duplicate_margin_loss`,
`spurious_margin_loss`, `lane_balanced_point_loss`,
`short_valid_recall_loss`, `far_spurious_survival_loss`, and
`gt5_rank_consistency_loss` experiment knobs. Bottleneck notes below that depend on
`tools/diagnose_tusimple_count_confusion.py`, `--gcs-gt4-short-*`,
`extra_exist_loss`, or `--gcs-short-exist-*` are legacy post-`50999d6af`
experiment conclusions only. They do not describe currently available code,
CLI flags, loss terms, or active selected candidates.

## Data And Geometry

- The active fixed-y anchors must be `710, 700, 690, ..., 160` normalized by original height `720`.
- Do not mix old `fixed_y=[0.98,0.25]` or K32 labels with this branch.
- Do not resample historical K32 labels into K56. Regenerate K56 labels from original TuSimple JSON and images.
- Preserve `--imgsz 544 960` in H,W order.
- The converted fixed-y dataset alone is not the full original TuSimple archive; official TuSimple evaluation still needs original raw-file image resolution/path context.

## Validation

- Local validation can check parser defaults, YAML contracts, fixed-y anchors, model output shape, and sample labels.
- Formal training and official-val evaluation should run on the remote CUDA server.
- This branch includes `tools/eval_tusimple_official.py` and `tools/sweep_tusimple_official.py` for official-val and final TuSimple test evaluation.
- The active rollback code does not include `tools/diagnose_tusimple_count_confusion.py`.
- It still does not include later mainline `diagnose_gcs_gt5.py`, training-time `official_best` checkpoint preservation, Count/Quality/Boundary diagnostics, Survival, or near-miss machinery.

## Legacy Post-50999 Official-Val Selection State

The 2026-06-21 `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03`
candidate was selected on official-val in a later post-`50999d6af` experiment
line. After the rollback, it is a legacy result, not the active code baseline:

```text
official-val363 ACC = 0.970851
official-val363 FP = 0.022084
official-val363 FN = 0.011708
official-val363 count_acc = 0.939394
decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
```

It beats the previous selected official-val ACC by `+0.000875`, mainly by
reducing FN. The tradeoff is worse count robustness:

```text
previous count03_under5_03 count_acc = 0.969697, count_acc_4 = 0.909091
gt4short15 count_acc = 0.939394, count_acc_4 = 0.848485
```

Before the later `dupmargin005` report, the 2026-06-20
`gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03` candidate was
the stronger final-test report:

```text
official-val363 ACC = 0.969976
final test ACC = 0.965459
```

The 2026-06-21 `gt4short15` final-test report is reporting-only and did not
beat that previous result:

```text
final test ACC = 0.965369
final test FP = 0.033309
final test FN = 0.029236
final test count_acc = 0.864486
final test count_acc_4 = 0.482906
final test count_acc_5 = 0.845343
```

Final-test count breakdowns are diagnostic/reporting-only, not tuning sources:

```text
count03_under5_03 count_acc_4 = 0.542735, confusion 4->3=113, 4->5=96, 4->6=5
gt4short15 count_acc_4 = 0.482906, confusion 4->3=118, 4->5=117, 4->6=7
```

Any threshold, max-det, min-points, checkpoint, or postprocess changes must be selected on official-val. Do not tune from the final test breakdown.

## 2026-06-20 Train/Val Count-Confusion Diagnostic

A train/val-only diagnostic was run on the remote server with the frozen `count03_under5_03` checkpoint and the official-val selected decode:

```text
weights = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03/weights/best.pt
decode  = conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
output  = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_count_confusion_train_val_by_visibility/summary.json
```

This diagnostic did not sweep test thresholds and is not a candidate promotion. It groups decoded lane-count confusion by split, source date, GT lane count, and shortest visible GT lane bucket.

Fixed-y validation split:

```text
images = 363
count_acc = 0.939394
GT3 count_acc = 0.953271
GT4 count_acc = 0.941667
GT5 count_acc = 0.866667
0601 date count_acc = 0.793103
0313-2|GT4|min_visible<=10: n=6, count_acc=0.166667, confusion 4->3=1, 4->4=1, 4->5=4
0601|GT4|min_visible=11..20: n=6, count_acc=0.333333, confusion 4->3=1, 4->4=2, 4->5=3
0601|GT5|min_visible<=10: n=11, count_acc=0.818182, confusion 5->4=2, 5->5=9
```

Fixed-y training split:

```text
images = 3263
count_acc = 0.963837
GT3 count_acc = 0.974699
GT4 count_acc = 0.959202
GT5 count_acc = 0.968610
0601 date count_acc = 0.905512
0313-2|GT4|min_visible<=10: n=70, count_acc=0.657143, confusion 4->3=7, 4->4=46, 4->5=17
0601|GT4|min_visible<=10: n=19, count_acc=0.368421, confusion 4->3=6, 4->4=7, 4->5=6
0601|GT4|min_visible=11..20: n=47, count_acc=0.680851, confusion 4->3=3, 4->4=32, 4->5=12
0601|GT5|min_visible<=10: n=143, count_acc=0.958042, confusion 5->4=6, 5->5=137
```

Integrated conclusion:

- Supported fact: the count weakness is reproducible on train/val without touching final test. The sharpest train/val failure is `GT4` with short side lanes, especially `0601|GT4|min_visible<=10`.
- Supported fact: `GT5` is comparatively robust on the fixed-y train split, so the next change should not focus only on dense-lane undercount.
- Hypothesis: the current `sum(sigmoid(pred_logits))` count loss plus `target>=5` undercount penalty does not provide enough targeted pressure for `GT4` short-lane undercount and overcount.
- Legacy next action at the time: use `tools/diagnose_tusimple_count_confusion.py` for reusable train/val `(date, lane_count, min_visible_points)` confusion, then run a train-only experiment with explicit `--gcs-gt4-short-boost` sampling for `GT4` short-side-lane samples. This is historical context only because the active rollback code no longer includes that diagnostic or sampler flag.

## 2026-06-20 GT4 Short-Lane Boost Result

The completed `gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03`
experiment is rejected for promotion because the 363-image official-val ACC did
not beat the baseline.

```text
A baseline official-val:
best = conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
official_acc = 0.969976
official_FP = 0.019559
official_FN = 0.014463
official_score = 0.969296
count_acc = 0.969697

B gt4short2 official-val:
best = conf=0.15, point_valid_thr=0.5, nms_dist_px=18.0, max_det=5, min_points=4
official_acc = 0.969726
official_FP = 0.017264
official_FN = 0.016299
official_score = 0.969055
count_acc = 0.977961
```

The sampler boost did hit the intended train/val diagnostic bottleneck under
the fixed decode `conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8,
min_points=5`:

```text
val GT4|min_visible<=10: A 6/11 -> B 8/11
val 0313-2|GT4|min_visible<=10: A 1/6 -> B 4/6
train GT4|min_visible<=10: A 84/124 -> B 99/124
train 0313-2|GT4|min_visible<=10: A 46/70 -> B 58/70
train 0601|GT4|min_visible<=10: A 7/19 -> B 13/19
train 0601|GT4|min_visible=11..20: A 32/47 -> B 37/47
```

But it also left or introduced collateral issues:

```text
val GT3 count_acc: A 0.953271 -> B 0.934579
val GT5 count_acc: A 0.866667 -> B 0.800000
official-val count_acc_5: A 0.986486 -> B 0.972973
official-val FN: A 0.014463 -> B 0.016299
```

Do not use
`gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03_official_val_sweep_maxdet8`
for promotion decisions: its summary points to
`archive/TUSimple/train_set/label_data_0313.json` with `images=2858`, not the
363-image official-val subset.

Smallest next experiment: keep the same short-GT4 diagnostic target but reduce
collateral FN/GT3/GT5 cost, for example `--gcs-gt4-short-boost 1.5` with
`--gcs-gt4-short-min-visible-max 10`. Select only on the 363-image official-val
sweep, then rerun the train/val count-confusion diagnostic before any final-test
reporting.

## 2026-06-21 GT4 Short-Lane Boost 1.5 Result

The completed `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03`
experiment was accepted as the official-val selected candidate in the legacy
post-`50999d6af` line because
it beats the previous baseline on the same 363-image official-val surface.

```text
A baseline official-val:
best = conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
official_acc = 0.969976
official_FP = 0.019559
official_FN = 0.014463
official_score = 0.969296
count_acc = 0.969697
count_acc_4 = 0.909091

C gt4short15 official-val:
best = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
official_acc = 0.970851
official_FP = 0.022084
official_FN = 0.011708
official_score = 0.970175
count_acc = 0.939394
count_acc_4 = 0.848485
```

Interpretation:

- Supported fact: `gt4short15` improves official ACC and official score on the
  valid 363-image official-val subset.
- Supported fact: the gain is not from better lane-count prediction. Count
  accuracy drops by `0.030303`, and GT4 count accuracy drops by `0.060606`.
- Supported fact: the metric gain comes from the FN reduction
  `0.014463 -> 0.011708`, while FP rises `0.019559 -> 0.022084`.
- Legacy decision at the time: keep `gt4short15` selected on official-val, but treat its final-test
  report as non-improving reporting evidence.

Completed train/val diagnostic:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_count_confusion_train_val_by_visibility/summary.json
decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4

train images = 3263
train count_acc = 0.952191
train GT3 count_acc = 0.963855
train GT4 count_acc = 0.945150
train GT5 count_acc = 0.977578
train 0313-2|GT4|min_visible<=10: n=70, count_acc=0.600000, confusion 4->3=3, 4->4=42, 4->5=23, 4->6=2
train 0601|GT4|min_visible<=10: n=19, count_acc=0.315789, confusion 4->3=7, 4->4=6, 4->5=6
train 0601|GT4|min_visible=11..20: n=47, count_acc=0.659574
train 0601|GT5|min_visible<=10: n=143, count_acc=0.965035

val images = 363
val count_acc = 0.920110
val GT3 count_acc = 0.925234
val GT4 count_acc = 0.929167
val GT5 count_acc = 0.733333
val 0313-2|GT4|min_visible<=10: n=6, count_acc=0.666667
val 0601|GT4|min_visible=11..20: n=6, count_acc=0.333333
val 0601|GT5|min_visible<=10: n=11, count_acc=0.727273
val 0531|GT4|min_visible=11..20: n=5, count_acc=0.200000
```

Completed final-test report:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
images = 2782
official_acc = 0.965369
official_FP = 0.033309
official_FN = 0.029236
official_score = 0.964118
count_acc = 0.864486
count_acc_3 = 0.974713
count_acc_4 = 0.482906
count_acc_5 = 0.845343
count_confusion includes 4->3=118, 4->4=226, 4->5=117, 4->6=7, 5->3=25, 5->4=48, 5->5=481, 5->6=15
```

Integrated conclusion:

- Supported fact: there is no protocol or execution blocker in the diagnostic; the selected decode runs cleanly on train/val and final test.
- Supported fact: the official-val gain did not transfer into a better final-test report. `gt4short15` final-test ACC is lower by `0.000090` than `count03_under5_03`.
- Supported fact: count robustness is worse on final test, especially `GT4` (`0.542735 -> 0.482906`), while `GT5` improves only modestly (`0.831283 -> 0.845343`).
- Bottleneck: the branch still fails on count stability around short or ambiguous `GT4` side lanes, with both undercount and overcount. The problem is not solved by weaker GT4 short-lane oversampling plus a looser selected decode.
- Smallest safe next action: do not tune on final test. Return to official-val and train/val diagnostics, and target a train-side count/visibility refinement that separates `GT4` false-fifth suppression from true `GT5` retention.

## 2026-06-21 gt4short15 Failure Trace and NMS Check

Before changing `count_under5_loss`, a train/val failure-trace diagnostic and a
363-image official-val NMS-only sweep were run for the selected `gt4short15`
decode.

Failure-trace artifacts:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_failure_trace_train_val/failure_trace_summary.json
records = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_failure_trace_train_val/failure_trace_records.csv
decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
```

Train split:

```text
failure_images = 156 / 3263 = 0.047809
direction = overcount 126, undercount 30
missing reasons = geometry_miss_short_gt 28, geometry_miss 7, low_score 15, low_score_short_gt 8, matched_but_unassigned 1
extra reasons = spurious_extra 102, duplicate_like_extra 62
drop counts on failure images = score_below_conf 1158, min_points_filtered 8, max_det_dropped 2
```

Validation split:

```text
failure_images = 29 / 363 = 0.079890
direction = overcount 19, undercount 10
missing reasons = low_score 4, low_score_short_gt 4, geometry_miss 3, geometry_miss_short_gt 2, min_points_visibility 1
extra reasons = spurious_extra 19, duplicate_like_extra 6
drop counts on failure images = score_below_conf 222, min_points_filtered 3
```

Interpretation:

- The main selected-decode count failure is overcount, not undercount.
- The extra-lane failure is mostly `spurious_extra`; `duplicate_like_extra` is
  present but not dominant.
- The selected decode has `nms_dist_px=0.0`, so `duplicate_like_extra` means
  close predictions that were not suppressed, not true NMS suppression mistakes.
- Missed lanes are mostly low score or geometry miss; validation has only one
  `min_points_visibility` missed-lane reason. This is not primarily a
  point-valid contiguous-span or `min_points` failure.

The controlled NMS sweep fixed `conf=0.15`, `point_valid_thr=0.5`, `max_det=6`,
`min_points=4`, used `--half`, and swept only `nms_dist_px` on the 363-image
official-val subset:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val363_nms_only_conf015_maxdet6_minp4_half/tusimple_official_sweep_summary.json
gt_json = runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json
```

Key rows:

```text
nms=0/2/4/6: official_acc=0.970851, FP=0.022084, FN=0.011708, count_acc=0.939394
nms=8..30:  official_acc=0.970728, FP=0.022222, FN=0.012397, count_acc=0.942149
nms=50:     official_acc=0.970728, FP=0.020845, FN=0.012397, count_acc=0.947658
nms=80:     official_acc=0.970679, FP=0.019238, FN=0.012397, count_acc=0.953168
```

Integrated conclusion:

- Supported fact: NMS improves count shape and reduces FP at large distances,
  but it raises FN and lowers official ACC on the valid 363-image selection
  surface.
- Decision: keep the selected postprocess at `nms_dist_px=0.0`; do not promote
  an NMS-only decode change.
- Bottleneck: the next train-side work should suppress high-score spurious or
  duplicate-like extra queries while preserving true short side lanes. This is
  an existence/geometry calibration problem more than a `min_points` problem.
- Legacy implemented next action: `extra_exist_loss` added a default-disabled
  high-score unmatched-query BCE penalty using only training Hungarian matches
  and detached query existence scores. It must not use NMS, `max_det`,
  `min_points`, decoded lane counts, or GT-count gating.
- Completed result: `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist005_count03_under5_03`
  did suppress extras enough to improve official-val FP and count accuracy, but
  it did not beat the official-val ACC gate. Best 363-image official-val row:
  `conf=0.005`, `point_valid_thr=0.5`, `nms_dist_px=18.0`, `max_det=6`,
  `min_points=5`, `official_acc=0.969603`, `FP=0.017815`, `FN=0.012856`,
  `count_acc=0.961433`, `count_acc_4=0.878788`, `count_acc_5=0.986486`.
  Reject it for promotion and keep final test closed.
- Completed follow-up: `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist0025_count03_under5_03`
  is worse across official-val ACC, FP, FN, and count. Best 363-image
  official-val row: `conf=0.05`, `point_valid_thr=0.5`, `nms_dist_px=0.0`,
  `max_det=6`, `min_points=4`, `official_acc=0.961513`, `FP=0.049633`,
  `FN=0.029844`, `count_acc=0.887052`, `count_acc_4=0.772727`,
  `count_acc_5=0.864865`. Its best count row reaches only
  `count_acc=0.928375` with `official_acc=0.961400`, still below the selected
  `gt4short15` count accuracy.
- Legacy updated decision: stop the `extra_exist_loss` gain-sweep family for now. The
  `0.05` run gave diagnostic FP/count improvement but missed ACC; the `0.025`
  run gives no useful diagnostic tradeoff. Return to short-lane score/geometry
  retention rather than stronger or weaker unmatched-query suppression.
- Legacy implemented follow-up: `GCSLoss.exist_loss()` had a default-disabled
  short matched existence floor. It applies only after Hungarian matching and
  only when the matched short GT lane passes APE and visible-IoU gates. Initial
  experiment settings are `--gcs-short-exist-floor 0.4`,
  `--gcs-short-exist-max-visible 20`,
  `--gcs-short-exist-floor-max-ape 20.0`, and
  `--gcs-short-exist-floor-min-iou 0.3`.
- Completed result: `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_shortexist04_count03_under5_03`
  is valid official-val evidence but rejected for promotion. Best 363-image
  official-val row: `conf=0.1`, `point_valid_thr=0.5`,
  `nms_dist_px=30.0`, `max_det=8`, `min_points=5`,
  `official_acc=0.968966`, `FP=0.019972`, `FN=0.014922`,
  `official_score=0.968268`, `count_acc=0.950413`,
  `count_acc_4=0.863636`, `count_acc_5=0.972973`. It improves FP and count
  accuracy relative to `gt4short15`, but misses the ACC gate `0.970851` by
  `0.001885` and raises FN by `0.003214`. No final test should be run.
- Legacy updated decision: do not promote `shortexist04`; keep `gt4short15` selected in that experiment line.
  The next smallest safe action is diagnostic rather than another immediate
  training knob: run a train/val failure trace for the `shortexist04` best
  decode if we need to confirm whether the floor reduced `low_score_short_gt`
  while shifting errors into FN or count tradeoffs.

## 2026-06-21 User-Requested Reporting-Only Test Batch

The user requested official test ACC for the recent experiments. The runs below
use each experiment's official-val selected decode and must not be used to tune
thresholds, checkpoints, or postprocess settings:

```text
run                         official_test_ACC  FP        FN        count_acc
count03_under5_03           0.965459           0.029439  0.026270  0.872753
gt4short2                   0.965206           0.026432  0.027079  0.884256
gt4short15                  0.965369           0.033309  0.029236  0.864486
extraexist005               0.965032           0.029661  0.027139  0.878864
shortexist04                0.964992           0.031381  0.028876  0.875988
extraexist0025              0.961330           0.044764  0.034508  0.838605
```

Integrated conclusion:

- Supported fact: within that reporting-only batch, none of the rejected experiments beat the existing
  `count03_under5_03` final-test report `ACC=0.965459`.
- Supported fact: `gt4short15` remained the official-val selected candidate in the legacy experiment line, but
  its final-test ACC `0.965369` is still below the previous report.
- Decision: keep final-test evidence reporting-only. Future changes must return
  to official-val and train/val diagnostics instead of using these test results
  for selection.

## 2026-06-22 count03_under5_00 Under5-Loss Ablation

The completed `gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00`
run set `gcs_count=0.3`, `gcs_count_under5=0.0`, and
`gcs_count_under5_min_lanes=5`. It is rejected for promotion because the
363-image official-val ACC is below both `count03_under5_03` and the later
legacy `gt4short15` gate.

```text
run = gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00_official_val_sweep
best = conf=0.08, point_valid_thr=0.5, nms_dist_px=50.0, max_det=6, min_points=6
official-val ACC = 0.968578
official-val FP = 0.022590
official-val FN = 0.016529
official-val official_score = 0.967796
official-val count_acc = 0.955923
official-val count_acc_3/4/5 = 0.973094 / 0.893939 / 0.959459
```

The user-requested one-shot official test used that official-val selected
decode and is reporting-only:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00_official_test_best_from_val/tusimple_official_summary.json
images = 2782
official_test ACC = 0.965118
FP = 0.031908
FN = 0.029745
official_score = 0.963885
count_acc = 0.875270
count_acc_2/3/4/5 = 0.400000 / 0.975862 / 0.566239 / 0.826011
```

Comparison to the previous `count03_under5_03` final-test report:

```text
count03_under5_03: ACC=0.965459, FP=0.029439, FN=0.026270, official_score=0.964345, count_acc=0.872753
count03_under5_00: ACC=0.965118, FP=0.031908, FN=0.029745, official_score=0.963885, count_acc=0.875270
delta: ACC=-0.000341, FP=+0.002469, FN=+0.003475, official_score=-0.000460, count_acc=+0.002517
```

Integrated conclusion:

- Supported fact: removing the under-5 count penalty slightly improves total
  final-test `count_acc`, mostly through better GT4 count accuracy versus
  `count03_under5_03`.
- Supported fact: the official-val selection metric is worse
  (`0.968578 < 0.969976` baseline and `< 0.970851` legacy `gt4short15`), and
  final-test ACC, FP, and FN are worse than `count03_under5_03`.
- Supported fact: GT5 count accuracy falls versus `count03_under5_03`
  (`0.831283 -> 0.826011` on final test), so the change is not a broad count
  robustness fix.
- Caveat: the test command used the official-val selected `max_det=6`, while
  the train args recorded `gcs_eval_max_det=8`. This is valid for reporting the
  selected decode but should be kept as a comparability note.
- Decision: do not promote `count03_under5_00`; do not tune thresholds,
  checkpoint choice, `max_det`, `min_points`, NMS, or loss weights from the
  final-test breakdown.

## 2026-06-22 dupmargin005 Duplicate-Margin Near-Miss

The completed `gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03`
run enabled the current default-disabled duplicate-margin experiment:

```text
gcs_duplicate_margin = 0.05
gcs_duplicate_margin_logit = 1.0
gcs_duplicate_gt_count = 4
gcs_duplicate_short_visible_max = 20
gcs_duplicate_min_overlap = 2
gcs_duplicate_min_visible_iou = 0.4
gcs_duplicate_pos_ape_px = 20.0
gcs_duplicate_neg_ape_px = 120.0
gcs_duplicate_ape_gap_px = 5.0
gcs_duplicate_max_pairs_per_gt = 2
```

Its 363-image official-val sweep selected:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_val_sweep
best = conf=0.05, point_valid_thr=0.45, nms_dist_px=0.0, max_det=6, min_points=6
official-val ACC = 0.970272
official-val FP = 0.024564
official-val FN = 0.016070
official-val official_score = 0.969459
official-val count_acc = 0.953168
official-val count_acc_3/4/5 = 0.950673 / 0.939394 / 0.972973
```

The user-requested one-shot official test used that official-val selected
decode and is reporting-only:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
images = 2782
official_test ACC = 0.965702
FP = 0.029493
FN = 0.027348
official_score = 0.964565
count_acc = 0.865924
count_acc_2/3/4/5 = 0.200000 / 0.971839 / 0.547009 / 0.810193
```

Integrated conclusion:

- Supported fact: `dupmargin005` beats the older `count03_under5_03`
  official-val ACC (`0.970272 > 0.969976`) and gives the strongest
  reporting-only final-test ACC recorded in this branch notes through
  2026-06-22.
- Supported fact: it still misses the stronger `gt4short15` official-val gate
  (`0.970272 < 0.970851`), so final test cannot promote it.
- Supported fact: compared with `gt4short15`, it improves final-test ACC, FP,
  FN, total count accuracy, and GT4 count accuracy.
- Supported fact: compared with `count03_under5_03`, it improves final-test ACC
  by only `+0.000243` while increasing FN and reducing total count accuracy.
  GT5 count accuracy drops from `0.831283` to `0.810193`, with more `5->4`
  errors.
- Decision: keep it as a rejected near-miss. The duplicate-margin idea is
  useful, but this setting has not solved count stability and is not the
  selected candidate under official-val protocol.
- Smallest safe next action: if continuing this line, run train/val-only
  failure traces for the selected decode `conf=0.05`, `point_valid_thr=0.45`,
  `nms_dist_px=0.0`, `max_det=6`, `min_points=6` and compare failure buckets
  against `count03_under5_03` and `gt4short15`. Do not tune from final test.

## 2026-06-23 spurmargin003 Spurious-Margin Rejection

The completed `gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03`
run enabled the current default-disabled spurious-margin experiment:

```text
gcs_spurious_margin = 0.03
gcs_spurious_margin_logit = 1.0
gcs_spurious_gt_counts = 3,4
gcs_spurious_pos_ape_px = 20.0
gcs_spurious_pos_min_visible_iou = 0.4
gcs_spurious_neg_min_ape_px = 50.0
gcs_spurious_neg_max_visible_iou = 0.2
gcs_spurious_duplicate_ape_px = 50.0
gcs_spurious_duplicate_visible_iou = 0.4
gcs_spurious_max_pairs_per_image = 4
```

Its 363-image official-val sweep selected:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03_official_val_sweep
best = conf=0.02, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=5
official-val ACC = 0.969316
official-val FP = 0.023095
official-val FN = 0.014922
official-val official_score = 0.968556
official-val count_acc = 0.958678
official-val count_acc_3/4/5 = 0.973094 / 0.878788 / 0.986486
count_confusion = 3->3=217, 3->4=6, 4->3=1, 4->4=58, 4->5=7, 5->4=1, 5->5=73
```

The user-requested one-shot official test used that official-val selected
decode and is reporting-only:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
images = 2782
official_test ACC = 0.964522
FP = 0.033591
FN = 0.028636
official_score = 0.963277
count_acc = 0.869878
count_acc_2/3/4/5 = 0.200000 / 0.964368 / 0.527778 / 0.868190
```

Integrated conclusion:

- Supported fact: `spurmargin003` misses the official-val ACC of all relevant
  comparators: `count03_under5_03` (`0.969976`), `dupmargin005` (`0.970272`),
  and `gt4short15` (`0.970851`).
- Supported fact: its reporting-only final-test ACC `0.964522` is also below
  `count03_under5_03` (`0.965459`), `gt4short15` (`0.965369`), and
  `dupmargin005` (`0.965702`).
- Supported fact: the spurious margin did improve GT5 count accuracy relative
  to `dupmargin005` on final test (`0.810193 -> 0.868190`), but this is not
  promotable because official-val ACC and final-test ACC both regress.
- Supported fact: GT4 remains unstable. Official-val `count_acc_4=0.878788`
  is below `dupmargin005` (`0.939394`), and final-test `count_acc_4=0.527778`
  is below `dupmargin005` (`0.547009`) and roughly in the old failure range.
- Caveat: the selected official-val decode uses `max_det=6`, while train args
  record `gcs_eval_max_det=8`. This is valid reporting from selected
  official-val decode, not a reason to tune final test.
- Decision: reject `spurmargin003`; do not continue by sweeping
  `gcs_spurious_margin` gains or retuning NMS/thresholds from test.

## 2026-06-23 shortpos Positive Short-Lane Loss Rejection

The completed `gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03`
run enabled the current default-disabled positive short-lane losses:

```text
gcs_lane_balanced_point = 3.0
gcs_short_valid_recall = 0.5
gcs_short_valid_max_visible = 20
gcs_short_valid_min_visible = 4
gcs_short_valid_max_ape_px = 40.0
gcs_short_valid_min_visible_iou = 0.3
gcs_duplicate_margin = 0.0
gcs_spurious_margin = 0.0
```

Its 363-image official-val sweep selected:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03_official_val_sweep
best = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=5
official-val ACC = 0.968144
official-val FP = 0.027319
official-val FN = 0.017218
official-val official_score = 0.967253
official-val count_acc = 0.947658
official-val count_acc_3/4/5 = 0.959641 / 0.878788 / 0.972973
count_confusion = 3->3=214, 3->4=9, 4->3=1, 4->4=58, 4->5=7, 5->4=1, 5->5=72, 5->6=1
```

The user-requested one-shot official test used that official-val selected
decode and is reporting-only:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
images = 2782
official_test ACC = 0.963067
FP = 0.038330
FN = 0.030763
official_score = 0.961685
count_acc = 0.861251
count_acc_2/3/4/5 = 0.400000 / 0.957471 / 0.547009 / 0.829525
pred_lanes_hist = 2=4, 3=1801, 4=362, 5=574, 6=41
gt_lanes_hist = 2=5, 3=1740, 4=468, 5=569
count_confusion = 2->2=2, 2->3=2, 2->4=1, 3->2=2, 3->3=1666, 3->4=63, 3->5=7, 3->6=2, 4->3=109, 4->4=256, 4->5=95, 4->6=8, 5->3=24, 5->4=42, 5->5=472, 5->6=31
```

Integrated conclusion:

- Supported fact: `shortpos` misses the official-val ACC of `count03_under5_03`
  (`0.969976`), `dupmargin005` (`0.970272`), `gt4short15` (`0.970851`),
  `spurmargin003` (`0.969316`), and `count03_under5_00` (`0.968578`).
- Supported fact: its reporting-only final-test ACC `0.963067` is also below
  all 2026-06-20 to 2026-06-23 comparators recorded in this branch notes.
- Supported fact: the selected official-val confidence is `0.005`, which is a
  strong score-calibration warning. The positive geometry/visibility terms did
  not translate into stable lane-existence confidence.
- Supported fact: final-test FP and FN both worsen versus `count03_under5_03`.
  This is not a clean recall/precision tradeoff; it is a broad degradation.
- Supported fact: count behavior remains unstable. GT4 count accuracy only
  ties `dupmargin005` on final test (`0.547009`), while GT3 count accuracy
  drops and GT5 produces many `5->6` extras (`31` cases).
- Caveat: the official-test decode uses `max_det=6`, while train args record
  `gcs_eval_max_det=8`. This is valid reporting from selected official-val
  decode, not a reason to retune test.
- Decision: reject `shortpos`; do not continue this exact loss setting or tune
  `conf`, `point_valid_thr`, `max_det`, `min_points`, NMS, or gains from the
  final-test report.

Pre-trace requested check, now completed below:

The `shortpos` selected decode was compared against `count03_under5_03` and
`dupmargin005` on failure buckets for `low_score`, `low_score_short_gt`,
`geometry_miss_short_gt`, `spurious_extra`, `duplicate_like_extra`, and
GT-count-specific confusion. The completed evidence says the bottleneck is not
solved by adding positive short-lane point/valid losses on top of the existing
objective.

Completed train/val-only follow-up:

```text
summary = runs/gcs_lane/shortpos_failure_trace_train_val_compare/summary.json
splits = train + val only
```

The follow-up confirms the rejection. Across train+val, `shortpos` lowers the
short-lane miss-like buckets but shifts the failure mass into extra-query
overcount:

```text
shortpos vs count03_under5_03:
  count_acc -0.018753, failure_images +68
  geometry_miss_short_gt -1, low_score_short_gt -12, min_points_visibility_short_gt -2
  spurious_extra +52, duplicate_like_extra +26, duplicate_like_extra_short_gt +33
  confusion deltas: 3->4 +30, 4->5 +33, 4->6 +13, 5->6 +4

shortpos vs dupmargin005:
  count_acc -0.016547, failure_images +60
  geometry_miss_short_gt -3, low_score_short_gt -4, min_points_visibility_short_gt -6
  spurious_extra +30, duplicate_like_extra +18, duplicate_like_extra_short_gt +33
  confusion deltas: 3->4 +22, 4->5 +38, 4->6 +4, 5->6 +4
```

Updated bottleneck:

The remaining blocker is not just short-lane point geometry or point-valid
recall. `shortpos` made short GT lanes easier to keep, but did not teach the
model to rank true matched lanes above surviving extras. The current bottleneck
is query existence/ranking calibration under short-lane retention pressure,
especially spurious and duplicate-like extras causing `3->4`, `4->5`, and
`4->6` overcount.

## 2026-06-23 farspur001 + gt5rank001 Rejection

The completed
`gcs_yolo_lane_s_tusimple_fixed_y_farspur001_gt5rank001_count03_under5_03`
run enabled two default-disabled score/ranking calibration terms on top of the
`count03_under5_03` baseline:

```text
gcs_far_spurious_survival = 0.01
gcs_far_spurious_gt_counts = 3,4
gcs_far_spurious_score_thr = 0.03
gcs_far_spurious_min_score = 0.03
gcs_far_spurious_min_ape_px = 50.0
gcs_far_spurious_max_visible_iou = 0.2
gcs_far_spurious_point_valid_thr = 0.5
gcs_far_spurious_min_visible_run = 5
gcs_far_spurious_max_neg_per_image = 1
gcs_far_spurious_loss_type = relu
gcs_gt5_rank_consistency = 0.01
gcs_gt5_rank_margin_logit = 0.5
gcs_gt5_rank_min_qminus_score = 0.02
gcs_gt5_rank_max_pairs_per_image = 1
```

Its 363-image official-val sweep selected:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_farspur001_gt5rank001_count03_under5_03_official_val_sweep
best = conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=8, min_points=4
official-val ACC = 0.965673
official-val FP = 0.033747
official-val FN = 0.022727
official-val official_score = 0.964544
official-val count_acc = 0.917355
official-val count_acc_3/4/5 = 0.964126 / 0.757576 / 0.918919
count_confusion = 3->3=215, 3->4=7, 3->5=1, 4->3=3, 4->4=50, 4->5=13, 5->3=1, 5->4=1, 5->5=68, 5->6=4
```

The sweep surface has no postprocess rescue point:

```text
best official_acc row = 0.965673, FP=0.035675, FN=0.022727, count_acc=0.909091
best official_score row = 0.965609, FP=0.028466, FN=0.022727, count_acc=0.944904
best count_acc row = 0.965490, FP=0.025207, FN=0.023416, count_acc=0.958678
minimum-FN row = 0.964452, FP=0.034114, FN=0.021120, count_acc=0.917355
minimum-FP row = 0.961070, FP=0.019697, FN=0.030303, count_acc=0.922865
```

Comparison against relevant official-val gates:

```text
gt4short15        ACC = 0.970851
dupmargin005      ACC = 0.970272
count03_under5_03 ACC = 0.969976
spurmargin003     ACC = 0.969316
count03_under5_00 ACC = 0.968578
shortpos          ACC = 0.968144
farspur001_gt5rank001 ACC = 0.965673
```

Integrated conclusion:

- Supported fact: this is a broad official-val regression, not a small
  postprocess miss. ACC falls by `0.004303` versus `count03_under5_03` and by
  `0.005178` versus `gt4short15`.
- Supported fact: the best selected confidence is `0.005`, which is a severe
  score-calibration warning. Raising confidence reduces FP but pushes FN even
  farther above every relevant comparator.
- Supported fact: FP and FN both worsen. This is not a clean precision/recall
  exchange and should not be rescued by NMS, `max_det`, `min_points`, or final
  test threshold tuning.
- Supported fact: GT4 count accuracy collapses to `0.757576` on the selected
  row. The run does not solve the known `GT4` short-side-lane count bottleneck.
- Likely cause: the two new losses fight the existing query calibration rather
  than fixing it. `far_spurious_survival_loss` applies an absolute q- logit
  ceiling for selected GT3/GT4 unmatched decode-risk queries, while
  `gt5_rank_consistency_loss` ranks the weakest GT5 matched q+ above top
  unmatched q- without a matched-lane geometry/visible-IoU quality gate. The
  combination can depress useful query scores and also raise weak matched GT5
  queries, producing unstable ranking instead of cleaner separation.
- Decision: reject `farspur001_gt5rank001`; do not run final test and do not
  continue by sweeping larger or smaller combined gains.

Next safe action:

The train/val query-trace diagnostic has been replaced by the self-contained
`tools/diagnose_gt4_short_failure_queries.py`, so it no longer imports missing
legacy `tools.diagnose_tusimple_count_confusion`. Run train/val-only traces for
`GT4` and `GT5` using each run's official-val selected decode and compare
`matched_qpos` versus `unmatched_qminus` score/APE/visible-IoU distributions
against `count03_under5_03`, `dupmargin005`, and `gt4short15`. Only after that
should an ablation be considered, and it should separate
`far_spurious_survival` from `gt5_rank_consistency` instead of combining them
again.

## 2026-06-24 dupmargin005 Official-Val Failure-Mode Compare

The requested comparison used only the 363-image TuSimple official-val subset.
No model was trained and final test stayed closed.

Artifacts:

```text
script = tools/compare_tusimple_failure_modes.py
summary = runs/gcs_lane/failure_compare/dupmargin005_compare/summary.json
per-image CSV = runs/gcs_lane/failure_compare/dupmargin005_compare/per_image_failures.csv
GT json = runs/gcs_lane/failure_compare/dupmargin005_compare/gt/tusimple_official_val_363_folder_aware_seed20260602.json
```

The three prediction files were regenerated with `tools/eval_tusimple_official.py`
using the official-val selected decodes:

```text
count03:      conf=0.05, point_valid_thr=0.5,  nms_dist_px=50.0, max_det=8, min_points=5
gt4short15:   conf=0.15, point_valid_thr=0.5,  nms_dist_px=0.0,  max_det=6, min_points=4
dupmargin005: conf=0.05, point_valid_thr=0.45, nms_dist_px=0.0,  max_det=6, min_points=6
```

Rerun official-val metrics:

```text
count03:      ACC=0.969976, FP=0.019559, FN=0.014463, count_acc=0.969697, count_acc_4=0.909091, count_acc_5=0.986486
gt4short15:   ACC=0.970837, FP=0.022084, FN=0.011708, count_acc=0.939394, count_acc_4=0.848485, count_acc_5=0.945946
dupmargin005: ACC=0.970246, FP=0.024564, FN=0.016070, count_acc=0.953168, count_acc_4=0.939394, count_acc_5=0.972973
```

The comparison script uses one-to-one lane matching with overlap `>=3` common
visible h-samples and mean absolute x error `<=20px`. Unmatched predictions
near any GT lane or already matched prediction under the same gate are
`duplicate_like_extra`; the rest are `spurious_extra`. Unmatched GT lanes with
`<=20` visible samples are `missed_short_gt`.

Count-failure-only buckets:

```text
count03:      failure_images=11, confusion={3->4:4, 4->3:1, 4->5:5, 5->4:1}, duplicate_like_extra=0, spurious_extra=15, missed_short_gt=5, missed_gt=3
gt4short15:   failure_images=22, confusion={3->4:8, 4->3:1, 4->5:8, 4->6:1, 5->4:3, 5->6:1}, duplicate_like_extra=1, spurious_extra=24, missed_short_gt=8, missed_gt=2
dupmargin005: failure_images=17, confusion={3->4:11, 4->3:1, 4->5:3, 5->4:2}, duplicate_like_extra=0, spurious_extra=18, missed_short_gt=5, missed_gt=2
```

Cross-run image sets:

```text
dupmargin_fixed_images = 4
  all are GT4 images where count03 was wrong and dupmargin005 count is correct

dupmargin_regressed_images = 10
  GT3=7, GT4=2, GT5=1
  dupmargin005 failure modes on these images: spurious_extra=10, duplicate_like_extra=0, missed_short_gt=3, missed_gt=1

gt4short15_correct_dupmargin_wrong_images = 10
  same GT-count shape as the dupmargin regression set: GT3=7, GT4=2, GT5=1
```

Integrated conclusion:

- Supported fact: this diagnostic does not show `dupmargin005` mainly reducing
  `duplicate_like_extra`. Under the strict official-lane comparison gate,
  duplicate-like extras are essentially absent (`0` for `count03`, `0` for
  `dupmargin005`, `1` for `gt4short15` on count-failure images).
- Supported fact: `dupmargin005` helps the GT4 count shape. It improves
  `count_acc_4` from `0.909091` to `0.939394` versus `count03` and from
  `0.848485` to `0.939394` versus `gt4short15`, mainly by reducing `4->5`
  overcount.
- Supported fact: it is not primarily a short-GT-lane casualty versus
  `count03`. Total `missed_short_gt` is lower (`33 -> 28`), and
  count-failure-only `missed_short_gt` stays flat (`5 -> 5`).
- Supported fact: the main regressions are extra `GT3` overcount and a small
  GT5 undercount increase. `3->4` grows from `4` to `11`, and `5->4` grows
  from `1` to `2` versus `count03`.
- Explanation for not beating `gt4short15`: `dupmargin005` has better count
  shape but worse official-val FN and ACC. The 10 images where `gt4short15` is
  count-correct and `dupmargin005` is not are mostly `GT3` spurious-extra
  overcounts, with one `GT5` `5->4` undercount.

Next safe action:

Do not tune thresholds or postprocess on final test. If continuing this line,
use train/val or official-val query traces to isolate why `dupmargin005`
creates more `GT3` extras while still slightly weakening `GT5` retention, then
consider a narrower score/ranking change only if it targets that specific
pattern without reopening final test.

## 2026-06-24 GT3 Extra-Survival Follow-Up

The smallest follow-up to `dupmargin005` is a default-disabled GT3-only query
ranking loss, not another duplicate-like suppression pass and not a positive
short-lane boost.

Implemented bottleneck target:

```text
loss item = gt3_extra_survival_loss
gain arg = --gcs-gt3-extra-survival, default 0.0
margin arg = --gcs-gt3-extra-margin-logit, default 0.05
top-k arg = --gcs-gt3-extra-topk, default 1
```

Rationale:

- Supported fact from the strict official-val failure compare:
  `dupmargin005` did not mainly reduce `duplicate_like_extra`; count-failure
  duplicate-like extras were `0` for both `count03` and `dupmargin005`.
- Supported fact: `dupmargin005` improved GT4 count shape, especially `4->5`
  overcount (`5 -> 3`) and `count_acc_4` (`0.909091 -> 0.939394` versus
  `count03`).
- Supported fact: the main regression was GT3 surplus/spurious overcount:
  `3->4` increased from `4` to `11`.
- Supported fact: `missed_short_gt` did not explain the regression; total
  `missed_short_gt` decreased from `33` to `28`, and count-failure-only
  `missed_short_gt` stayed at `5`.

The new loss therefore acts only when the internal GT lane-count target is
exactly `3`. It reuses the current training Hungarian `indices`, takes the
matched query logits as positives, takes unmatched query logits as negatives,
and applies a logit-space hinge:

```text
relu(top_unmatched_logit - min_matched_logit + margin)
```

Only the top `gcs_gt3_extra_topk` unmatched queries are penalized, with first
experiment settings `gain=0.03`, `margin=0.05`, and `topk=1`.

Risks to monitor on official-val:

- If GT5 `5->4` grows above `dupmargin005`'s `2`, try a smaller gain or a
  conservative variant with `min_matched_logit.detach()`.
- If GT4 `4->5` returns toward `count03`'s `5`, the loss is over-correcting
  query score calibration and should not be promoted.
- If GT3 `3->4` does not drop, the surplus query may be surviving through
  point-valid/min-points interactions rather than pure existence-logit ranking.
