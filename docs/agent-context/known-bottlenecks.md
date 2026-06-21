# Known Bottlenecks

This file applies to branch `codex/5-25-3-k56`.

## Branch Scope

The current mainline imports the historical `5-25-3.zip` algorithm and changes only the TuSimple fixed-y contract to Q=12/K=56 with official h-sample anchors.

Do not read mainline Count Head, Count Boundary, Quality Head, Survival Head, near-miss, or official-best bottlenecks as active branch behavior. Those mechanisms are not part of this 5-25-3 branch.

Active source/config is rolled back to commit `50999d6af` (`Document 5-25-3
K56 as mainline`). All bottleneck notes below that depend on
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

The 2026-06-20 `gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03`
candidate remains the stronger final-test report:

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

- Supported fact: none of the recent rejected experiments beats the existing
  `count03_under5_03` final-test report `ACC=0.965459`.
- Supported fact: `gt4short15` remained the official-val selected candidate in the legacy experiment line, but
  its final-test ACC `0.965369` is still below the previous report.
- Decision: keep final-test evidence reporting-only. Future changes must return
  to official-val and train/val diagnostics instead of using these test results
  for selection.
