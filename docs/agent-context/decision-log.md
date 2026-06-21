# Decision Log

This file records decisions for branch `codex/5-25-3-k56`.

## 2026-06-19: Import 5-25-3 as a Separate K56 Branch

Decision:

Import the historical `5-25-3.zip` GCS-YOLO-Lane algorithm as branch `codex/5-25-3-k56` and adapt only the explicit TuSimple fixed-y contract:

```text
legacy archive: Q=8, K=32, fixed_y=[0.98, 0.25]
this branch:    Q=12, K=56, fixed_y=[710/720, 160/720]
```

Use:

```text
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
data  = data/tusimple_gcs_fixed_y_k56_960x544.yaml
imgsz = 544 960
```

Why:

The user requested a separate branch for the `5-25-3.zip` algorithm while preserving the current K56/710 TuSimple contract.

Alternatives considered:

- Port later mainline Count/Quality/Survival/near-miss mechanisms into the branch.
- Keep the legacy Q8/K32 contract.
- Resample old K32 labels to K56.

Tradeoff:

The branch is intentionally narrower than current mainline. It preserves 5-25-3 algorithm behavior and does not include later mainline Count/Quality/Boundary/Survival machinery or training-time official-best checkpoint preservation. K56 labels must be regenerated from original TuSimple JSON and images.

Validation evidence:

Local checks performed during branch setup:

```text
python -m py_compile <changed GCS Python files>
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml --imgsz 544 960 --batch 1 --device cpu
YAML contract assertions for Q=12, K=56, fixed_y_start=710/720, fixed_y_end=160/720
fixed-y anchor assertion for 56 y pixels: 710,700,...,160
sample label split/order check against an external K56 dataset root
```

Mainline or experiment at the time:

Separate historical algorithm branch. This status was superseded by the 2026-06-20 mainline promotion below.

## 2026-06-20: Promote 5-25-3 K56 as Current Mainline

Decision:

Treat branch `codex/5-25-3-k56` as the current mainline for new K56 TuSimple work. Use the mainline aliases:

```text
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
data  = data/tusimple_gcs_fixed_y_960x544.yaml
imgsz = 544 960
```

The q12-k56-named config/data paths remain compatibility references for old experiment records:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
data/tusimple_gcs_fixed_y_k56_960x544.yaml
```

Why:

The user clarified that `5-25-3-k56` and the previous q12-k56 line are different algorithms, and the new 5-25-3 branch should be the active line going forward.

Preservation rule:

Do not delete previous q12-k56 experiment documentation. Keep it as historical experiment context, and do not let it override `docs/agent-context/current-contracts.md`.

## 2026-06-20: Keep Agent Workflow Local to Codex

Decision:

Document project Agent/Skill workflow rules for Codex collaboration while keeping the server-side branch payload focused on algorithm/runtime code.

Why:

The remote training server only needs code required for training, evaluation, diagnostics, model/config contracts, and data conversion. Agent setup files and wrapper scripts are local Codex workspace concerns and must not imply that later mainline Count/Quality/Boundary/Survival algorithm mechanisms are active.

Policy:

Permit assistant-originated runtime delegation through `multi_agent_v1.spawn_agent` only when the user explicitly asks for multi-Agent, delegation, subagents, or parallel agent work. If the tool is absent from the initial surface, the assistant must call `tool_search` with query `multi_agent_v1 spawn_agent` before declaring that the current API/tool surface does not expose runtime multi-Agent delegation. The assistant must not simulate Agent roles or describe local work as delegated output. Skill loading remains separate from delegation.

Close completed runtime agents after their results are integrated. Do not describe local Skill loading as delegated Agent output.

Validation evidence:

No server-side agent setup check is required for this branch.

Mainline or experiment:

Branch collaboration policy, not an algorithm promotion.

## 2026-06-20: Freeze count03_under5_03 Official-Val Selected Candidate

Decision:

Freeze `gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03` as the current evaluated candidate for this branch using `weights/best.pt` and the official-val selected decode:

```text
conf = 0.05
point_valid_thr = 0.5
nms_dist_px = 50.0
max_det = 8
min_points = 5
```

Official-val evidence:

```text
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_official_val363_sweep
grid = 1800 combinations over conf, point_valid_thr, nms_dist_px, max_det, and min_points
images = 363
official_acc = 0.969976
official_FP = 0.019559
official_FN = 0.014463
official_score = 0.969296
count_acc = 0.969697
```

Final test reporting evidence:

```text
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_official_test_best_from_val
images = 2782
official_acc = 0.965459
official_FP = 0.029439
official_FN = 0.026270
official_score = 0.964345
count_acc = 0.872753
avg_total_ms = 14.8576
```

Why:

The decode was selected on official-val, which satisfies the branch research-integrity rule for threshold and postprocess selection. The final test run used that selected configuration once and is reporting evidence only.

Rejected alternative:

The top `official_score` row was not promoted. It had a tiny score gain of about `0.000032`, but its official ACC was lower by about `0.000036`; official ACC remains the primary branch comparison metric.

Tradeoff:

This is an experiment result freeze, not an algorithm change. The branch still does not include training-time official-best checkpoint preservation, so `best.pt` provenance should be described as the source run checkpoint rather than an official-val checkpoint-selected artifact.

Remaining bottleneck:

The test count breakdown is diagnostic-only and must not drive tuning. It shows the largest weakness on 4-lane scenes:

```text
count_acc_4 = 0.542735
4->3 = 113
4->5 = 96
4->6 = 5
```

Mainline or experiment:

Experimental candidate accepted as the current reported branch result. Future changes must use official-val only for selection and keep test closed.

## 2026-06-20: Diagnose Count Confusion by Date and Short Visible Lanes

Decision:

Treat the `count03_under5_03` count weakness as a train/val reproducible `GT4` short-side-lane robustness problem, not as a test-threshold tuning problem.

Remote diagnostic evidence:

```text
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_count_confusion_train_val_by_visibility
summary   = summary.json
weights   = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03/weights/best.pt
decode    = conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
splits    = fixed-y train and val only
```

Key findings:

```text
val count_acc = 0.939394
val GT4 count_acc = 0.941667
val 0313-2|GT4|min_visible<=10: n=6, count_acc=0.166667
val 0601|GT4|min_visible=11..20: n=6, count_acc=0.333333

train count_acc = 0.963837
train GT4 count_acc = 0.959202
train 0313-2|GT4|min_visible<=10: n=70, count_acc=0.657143
train 0601|GT4|min_visible<=10: n=19, count_acc=0.368421
train 0601|GT4|min_visible=11..20: n=47, count_acc=0.680851
```

Why:

The same failure mode appears without using final test for selection: `GT4` scenes with short visible side lanes are unstable and show both undercount and overcount. `GT5` is comparatively stronger on the fixed-y train split, so a dense-lane-only undercount change is too narrow.

Rejected actions:

- Do not sweep or select thresholds on final test.
- Do not promote a new candidate from this diagnostic alone.
- Do not silently import later mainline Count/Quality/Survival machinery into this branch.

Recommended next action:

First make the `(date, lane_count, min_visible_points)` count-confusion diagnostic reusable on train/val. Then run the smallest train-side experiment that either upweights `GT4` short-side-lane samples or adds a GT4-aware count margin to the existing query-logit count objective. Select any follow-up candidate only on official-val.

Mainline or experiment:

Diagnostic-only branch evidence. No algorithm promotion and no test-tuned parameter change.

## 2026-06-20: Add Reusable Count Diagnostic and GT4 Short-Lane Sampler Boost

Decision:

Add a reusable train/val count-confusion diagnostic flow and an explicit train-time sampler option for `GT4` short visible side-lane samples.

Implementation:

```text
tools/diagnose_tusimple_count_confusion.py
  direct mode: --weights, --dataset-root, --splits train val
  grouping: date, GT lane count, shortest visible GT lane bucket

tools/train_gcs.py
ultralytics/models/yolo/gcs_lane/train.py
ultralytics/cfg/default.yaml
ultralytics/cfg/__init__.py
  new options:
    --gcs-gt4-short-boost
    --gcs-gt4-short-min-visible-max
```

Default behavior:

```text
gcs_gt4_short_boost = 1.0
gcs_gt4_short_min_visible_max = 10
```

Why:

The previous train/val diagnostic localized the count weakness to `GT4` scenes with short visible side lanes. A sampler multiplier is the smallest explicit train-side experiment that targets those samples without changing the model output contract, official metric, test protocol, or default branch behavior.

Recommended experiment:

```text
--gcs-gt4-short-boost 2.0
--gcs-gt4-short-min-visible-max 10
run name: gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03
```

Risk:

Oversampling short `GT4` cases may improve `4->3` undercount but could increase `4->5` overcount or reduce ordinary official-val ACC. Candidate selection must therefore remain official-val only, followed by train/val count-confusion diagnostics before any final-test reporting.

Mainline or experiment:

Branch-local experimental option. It does not import later mainline Count/Quality/Survival machinery and does not promote a new candidate.

## 2026-06-20: Reject gt4short2 as an Official-Val Promotion

Decision:

Do not promote `gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03`
to final test. Keep `count03_under5_03` as the current selected branch
candidate.

Official-val evidence:

```text
A baseline sweep:
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_official_val363_sweep
images = 363
best = conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
official_acc = 0.969976
official_FP = 0.019559
official_FN = 0.014463
official_score = 0.969296
count_acc = 0.969697

B gt4short2 sweep:
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03_official_val_sweep
images = 363
best = conf=0.15, point_valid_thr=0.5, nms_dist_px=18.0, max_det=5, min_points=4
official_acc = 0.969726
official_FP = 0.017264
official_FN = 0.016299
official_score = 0.969055
count_acc = 0.977961
```

Why:

B improved count accuracy and reduced official FP on the 363-image official-val
subset, but it did not improve the primary promotion metric:
`official_acc` dropped by `0.000250`, `official_score` dropped by `0.000241`,
and official FN rose by `0.001836`. The candidate is therefore diagnostic-only,
not promotable.

Train/val diagnostic evidence:

```text
decode = conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03_count_confusion_train_val_by_visibility/summary.json

val GT4|min_visible<=10: A 6/11 -> B 8/11, confusion A 4->3=1 4->4=6 4->5=4, B 4->3=0 4->4=8 4->5=3
val 0313-2|GT4|min_visible<=10: A 1/6 -> B 4/6, confusion A 4->3=1 4->4=1 4->5=4, B 4->3=0 4->4=4 4->5=2
val 0601|GT4|min_visible=11..20: unchanged at 2/6, confusion A/B 4->3=1 4->4=2 4->5=3
train GT4|min_visible<=10: A 84/124 -> B 99/124, confusion A 4->3=15 4->4=84 4->5=25, B 4->3=5 4->4=99 4->5=20
train 0313-2|GT4|min_visible<=10: A 46/70 -> B 58/70, confusion A 4->3=7 4->4=46 4->5=17, B 4->3=1 4->4=58 4->5=11
train 0601|GT4|min_visible<=10: A 7/19 -> B 13/19, confusion A 4->3=6 4->4=7 4->5=6, B 4->3=3 4->4=13 4->5=3
train 0601|GT4|min_visible=11..20: A 32/47 -> B 37/47, confusion A 4->3=3 4->4=32 4->5=12, B 4->3=2 4->4=37 4->5=7
```

Side effects:

```text
official-val count_acc_5: A 0.986486 -> B 0.972973
train/val diagnostic GT5: val 0.866667 -> 0.800000, train 0.968610 -> 0.982063
ordinary val GT3: 0.953271 -> 0.934579
ordinary val GT4: 0.941667 -> 0.945833
```

The target short-GT4 bottleneck improved under the fixed diagnostic decode, but
the improvement did not transfer into a higher official-val ACC. The side
effect is not an FP rise on the comparable 363-image sweep; the larger risk is
FN/GT5/GT3 degradation.

Invalid or non-comparable artifact:

```text
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03_official_val_sweep_maxdet8
gt_json = archive/TUSimple/train_set/label_data_0313.json
images = 2858
```

Despite the directory name, this is not the 363-image official-val subset and
must not be used for A/B promotion or threshold selection.

Recommended next action:

Run the next smallest train-side experiment against the 363-image official-val
gate: keep the short-GT4 focus, but reduce the collateral FN/GT3/GT5 cost. A
conservative next candidate is a weaker GT4 short-lane sampler multiplier, for
example `--gcs-gt4-short-boost 1.5` with the same
`--gcs-gt4-short-min-visible-max 10`, selected only on official-val and then
checked with the same train/val count-confusion diagnostic.

Mainline or experiment:

Rejected experimental candidate. No final-test run should be launched from this
candidate.

## 2026-06-21: Select gt4short15 on Official-Val

Decision:

Select `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03` as the
current official-val selected candidate. Do not treat the previous
`count03_under5_03` final-test result as evidence for this candidate.

Official-val evidence:

```text
A baseline sweep:
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_official_val363_sweep
images = 363
best = conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
official_acc = 0.969976
official_FP = 0.019559
official_FN = 0.014463
official_score = 0.969296
count_acc = 0.969697
count_acc_3 = 0.982063
count_acc_4 = 0.909091
count_acc_5 = 0.986486

C gt4short15 sweep:
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val_sweep
images = 363
best = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
official_acc = 0.970851
official_FP = 0.022084
official_FN = 0.011708
official_score = 0.970175
count_acc = 0.939394
count_acc_3 = 0.964126
count_acc_4 = 0.848485
count_acc_5 = 0.945946
```

Why:

The candidate improves the primary official-val metric by `+0.000875` and
official score by `+0.000879` on the same 363-image validation subset. The gain
comes from lower FN (`0.014463 -> 0.011708`) despite a higher FP
(`0.019559 -> 0.022084`).

Tradeoff:

The sampler multiplier `1.5` avoids the official-val FN regression seen in
`gt4short2`, but it does not solve the count bottleneck. It worsens total
count accuracy (`0.969697 -> 0.939394`) and GT4 count accuracy
(`0.909091 -> 0.848485`). This means the current selected candidate is a
metric-improving recall tradeoff, not a count-robustness fix.

Rejected or deferred actions:

- Do not tune thresholds on test.
- Do not reuse the old `count03_under5_03` final-test result for this candidate.
- Do not launch final-test reporting before rerunning train/val
  count-confusion diagnostics for `gt4short15`.

Recommended next action:

Run the branch-local train/val count-confusion diagnostic with the selected
`gt4short15` decode:

```text
weights = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03/weights/best.pt
decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
splits = train val
```

At selection time, the next protocol step was the train/val diagnostic followed
by one reporting-only final-test run if no blocker appeared. The follow-up
decision below records that completed diagnostic and final-test report.

Mainline or experiment:

Official-val selected experimental candidate. Final-test reporting is recorded
in the follow-up decision below.

## 2026-06-21: Report gt4short15 Final Test Without Final-Test Improvement

Decision:

Keep `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03` as the
official-val selected candidate, but do not describe it as a final-test
improvement. Its one-shot final-test report did not beat the previous
`count03_under5_03` final-test report.

Train/val diagnostic evidence:

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

Final-test reporting evidence:

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
count_confusion = 4->3=118, 4->4=226, 4->5=117, 4->6=7, 5->3=25, 5->4=48, 5->5=481, 5->6=15
```

Comparison with the previous final-test report:

```text
previous count03_under5_03 final-test official_acc = 0.965459
gt4short15 final-test official_acc = 0.965369
delta = -0.000090

previous count03_under5_03 final-test count_acc = 0.872753
gt4short15 final-test count_acc = 0.864486
delta = -0.008267

previous count03_under5_03 final-test count_acc_4 = 0.542735
gt4short15 final-test count_acc_4 = 0.482906
delta = -0.059829
```

Why:

The diagnostic did not expose a protocol blocker, so the one-shot final-test
report was allowed under the branch rules. The final-test result shows that
the official-val FN gain did not transfer into a better final-test ACC, and the
count bottleneck worsened on `GT4`.

Rejected actions:

- Do not tune thresholds, `max_det`, `min_points`, checkpoint choice, or
  postprocess settings from final test.
- Do not claim `gt4short15` as a final-test improvement.
- Do not import later Count/Quality/Survival machinery into this branch as an
  undocumented fix.

Recommended next action:

Treat `gt4short15` as official-val selected but final-test non-improving.
Future experiments should go back to official-val plus train/val diagnostics
and target the count/visibility mechanism around short or ambiguous `GT4`
lanes, especially separating false-fifth suppression from true `GT5` retention.

Mainline or experiment:

Official-val selected experimental candidate with a non-improving final-test
report. Final test remains closed for tuning.

## 2026-06-21: Reject NMS-Only Postprocess Change for gt4short15

Decision:

Keep the official-val selected `gt4short15` decode at `nms_dist_px=0.0`. Do not
promote an NMS-only postprocess change, and do not change `count_under5_loss`
until the train-side failure mode is targeted more precisely.

Failure-trace evidence:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_failure_trace_train_val/failure_trace_summary.json
decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4

train failure_images = 156 / 3263 = 0.047809
train direction = overcount 126, undercount 30
train extra reasons = spurious_extra 102, duplicate_like_extra 62
train missing reasons = geometry_miss_short_gt 28, geometry_miss 7, low_score 15, low_score_short_gt 8, matched_but_unassigned 1

val failure_images = 29 / 363 = 0.079890
val direction = overcount 19, undercount 10
val extra reasons = spurious_extra 19, duplicate_like_extra 6
val missing reasons = low_score 4, low_score_short_gt 4, geometry_miss 3, geometry_miss_short_gt 2, min_points_visibility 1
```

Why:

The failure trace separates the previous coarse `4->3/4->5` diagnosis into
actionable buckets. The selected decode's count failures are overcount-heavy,
mainly from extra predicted lanes. Undercount misses are more often low score
or geometry miss than point-valid contiguous-span collapse; validation has only
one `min_points_visibility` miss. Because the selected decode disables NMS,
`duplicate_like_extra` reports close duplicate-like predictions that are not
suppressed, not an NMS bug.

NMS-only official-val evidence:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val363_nms_only_conf015_maxdet6_minp4_half/tusimple_official_sweep_summary.json
gt_json = runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json
fixed = conf=0.15, point_valid_thr=0.5, max_det=6, min_points=4, half=True

nms=0/2/4/6: official_acc=0.970851, FP=0.022084, FN=0.011708, count_acc=0.939394
nms=8..30:  official_acc=0.970728, FP=0.022222, FN=0.012397, count_acc=0.942149
nms=50:     official_acc=0.970728, FP=0.020845, FN=0.012397, count_acc=0.947658
nms=80:     official_acc=0.970679, FP=0.019238, FN=0.012397, count_acc=0.953168
```

Rejected action:

Do not use NMS as the selected fix. It can improve count accuracy and FP at
larger distances, but it also raises FN and lowers the primary official-val
ACC on the 363-image selection surface.

Recommended next action:

Return to train-side work. The smallest defensible next experiment should
target high-score spurious or duplicate-like extra queries while preserving true
short side lanes, for example an existence/geometry calibration refinement.
Avoid a blind `count_under5_loss` change unless it directly addresses those
failure buckets. Select any follow-up only on official-val and keep final test
closed.

Mainline or experiment:

Diagnostic-only postprocess and failure-bucket evidence. No new selected decode
and no final-test tuning.

## 2026-06-21: Add Default-Disabled extra_exist_loss for High-Score Extra Queries

Decision:

Add an explicit `extra_exist_loss` item that penalizes high-score unmatched
queries after Hungarian matching, while leaving the existing `exist_loss`
semantics unchanged.

Implementation:

```text
ultralytics/utils/gcs_loss.py
  gcs_extra_exist = 0.0
  gcs_extra_exist_thr = 0.15
  hard_extra = unmatched query && detach(sigmoid(pred_logit)) >= threshold
  target = 0

tools/train_gcs.py
  --gcs-extra-exist
  --gcs-extra-exist-thr

ultralytics/models/yolo/gcs_lane/train.py
ultralytics/models/yolo/gcs_lane/val.py
ultralytics/cfg/default.yaml
ultralytics/cfg/__init__.py
tools/overfit_gcs_20.py
tools/analyze_gcs_results_csv.py
  sync loss names, gains, defaults, logging, helper summaries
```

Why:

The selected `gt4short15` failure trace shows overcount-heavy train/val
failures, with extra lanes mainly `spurious_extra` and some
`duplicate_like_extra`. The controlled official-val NMS sweep reduces some
overcount shape but raises FN and lowers official ACC. This makes a
train-side high-score extra-query penalty the smallest targeted next
experiment.

Rejected actions:

- Do not fold this term into `exist_loss`; it needs an independent log item so
  the experiment can show whether high-score extra queries are actually being
  suppressed.
- Do not penalize all unmatched queries; the existing `exist_loss` already
  supervises unmatched queries as target zero.
- Do not use NMS, `max_det`, `min_points`, decoded lane count, or GT-count
  gating inside the training loss.
- Do not touch `point_valid_loss`; this experiment is not a visible-span fix.

Validation evidence:

```text
python -m py_compile ultralytics/utils/gcs_loss.py ultralytics/models/yolo/gcs_lane/train.py ultralytics/models/yolo/gcs_lane/val.py ultralytics/cfg/__init__.py tools/train_gcs.py tools/overfit_gcs_20.py tools/analyze_gcs_results_csv.py
python tools/train_gcs.py --help
direct extra_exist_loss behavior check
GCSLoss.forward loss-vector/gain check
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml --imgsz 544 960 --batch 1 --device cpu
```

Recommended experiment:

Start conservatively:

```text
--gcs-extra-exist 0.05
--gcs-extra-exist-thr 0.15
```

Promote only if the 363-image official-val ACC is not below `0.970851`, FN is
not materially above `0.011708`, count confusion improves on `3->4`, `4->5`,
and `4->6`, and train/val failure traces reduce `spurious_extra` and
`duplicate_like_extra` without worsening short-lane miss buckets.

Mainline or experiment:

Branch-local experimental option. Defaults preserve baseline behavior and do
not import later Count/Quality/Survival/near-miss machinery.

## 2026-06-21: Reject extraexist005 as an Official-Val Promotion

Decision:

Do not promote
`gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist005_count03_under5_03`.
Keep `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03` as the
current official-val selected candidate and keep final test closed.

Official-val evidence:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist005_count03_under5_03_official_val_sweep
rows = 1800
images = 363
best = conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=5
official_acc = 0.969603
official_FP = 0.017815
official_FN = 0.012856
official_score = 0.968990
count_acc = 0.961433
count_acc_3 = 0.977578
count_acc_4 = 0.878788
count_acc_5 = 0.986486
count_confusion = 3->3=218, 3->4=5, 4->3=2, 4->4=58, 4->5=6, 5->4=1, 5->5=73
```

Comparison:

```text
current gt4short15 official-val ACC = 0.970851
extraexist005 official-val ACC      = 0.969603
delta                                 = -0.001248

previous count03_under5_03 official-val ACC = 0.969976
extraexist005 official-val ACC              = 0.969603
delta                                         = -0.000373

gt4short15 FP/FN/count_acc = 0.022084 / 0.011708 / 0.939394
extraexist005 FP/FN/count_acc = 0.017815 / 0.012856 / 0.961433
```

Interpretation:

- Supported fact: `extraexist005` does what the new loss was meant to do in
  one direction: it lowers FP and improves count accuracy relative to
  `gt4short15`.
- Supported fact: it does not satisfy the promotion gate. Official ACC is below
  both the current `gt4short15` selection and the previous `count03_under5_03`
  official-val result.
- Supported fact: the best row moves to a very low `conf=0.005`; at the old
  `gt4short15` decode (`conf=0.15`, `point_valid_thr=0.5`,
  `nms_dist_px=0.0`, `max_det=6`, `min_points=4`), ACC is only `0.968380` and
  FN rises to `0.016299`. This indicates query-score over-suppression, not a
  postprocess-only issue.
- Supported fact: the best count row reaches `count_acc=0.966942` and
  `count_acc_4=0.909091`, but its ACC is only `0.968080` to `0.968946`, so the
  count gain is diagnostic rather than promotable.

Rejected actions:

- Do not send `extraexist005` to final test.
- Do not tune final-test thresholds or checkpoint choice from this result.
- Do not increase `gcs_extra_exist` to `0.1`; the `0.05` run already suppresses
  scores too much.

Recommended next action:

Run one smaller-gain ablation that keeps the same mechanism but reduces score
collapse:

```text
--gcs-extra-exist 0.025
--gcs-extra-exist-thr 0.15
```

Select only on the same 363-image official-val surface. Promote only if
official ACC reaches or exceeds `0.970851` without raising FN above `0.011708`.
If it again improves count/FP but misses ACC, stop this loss family and return
to short-lane score/geometry retention rather than stronger extra suppression.

Mainline or experiment:

Rejected experimental candidate. The loss remains useful as a branch-local
diagnostic knob, but `0.05` is too strong for promotion.
