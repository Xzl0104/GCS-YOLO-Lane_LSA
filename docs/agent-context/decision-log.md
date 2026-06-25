# Decision Log

This file records decisions for branch `codex/5-25-3-k56`.

## 2026-06-21: Roll Active Code Back to 50999d6af

Decision:

Restore the active source/config state to commit `50999d6af` (`Document 5-25-3
K56 as mainline`).

Scope:

The active code baseline is the 5-25-3 K56 mainline contract at `50999d6af`
plus later default-disabled experiment knobs that are explicitly documented in
`docs/agent-context/current-contracts.md`. Other later commits and notes,
including reusable count diagnostics, GT4 short-lane sampling,
`extra_exist_loss`, short matched existence floor, and reporting-only test
batches, are retained below as legacy experiment conclusions only. They do not
describe currently available CLI flags, loss items, scripts, or active selected
candidates unless a future task explicitly restores those mechanisms.

Why:

The user requested the code rollback while keeping later experiment content in
the documentation as old conclusions.

Mainline or experiment:

Current mainline rollback decision. Later experiment sections remain historical
records.

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

## 2026-06-21: Reject extraexist0025 and Stop the extra_exist Gain Sweep

Decision:

Do not promote
`gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist0025_count03_under5_03`.
Keep `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03` as the
current official-val selected candidate and keep final test closed.

Official-val evidence:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist0025_count03_under5_03_official_val_sweep
rows = 1800
images = 363
gt_json = runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json
best = conf=0.05, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
official_acc = 0.961513
official_FP = 0.049633
official_FN = 0.029844
official_score = 0.959923
count_acc = 0.887052
count_acc_3 = 0.928251
count_acc_4 = 0.772727
count_acc_5 = 0.864865
count_confusion = 3->3=207, 3->4=14, 3->5=2, 4->3=3, 4->4=51, 4->5=12, 5->4=1, 5->5=64, 5->6=9
```

Comparison:

```text
current gt4short15 official-val ACC = 0.970851
extraexist0025 official-val ACC     = 0.961513
delta                                = -0.009338

previous count03_under5_03 official-val ACC = 0.969976
extraexist0025 official-val ACC             = 0.961513
delta                                        = -0.008463

extraexist005 official-val ACC  = 0.969603
extraexist0025 official-val ACC = 0.961513
delta                            = -0.008090

gt4short15 FP/FN/count_acc = 0.022084 / 0.011708 / 0.939394
extraexist0025 FP/FN/count_acc = 0.049633 / 0.029844 / 0.887052
```

Additional checks:

```text
old gt4short15 decode on extraexist0025:
  conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
  official_acc=0.961450, FP=0.045638, FN=0.029844, count_acc=0.909091

best count row:
  count_acc=0.928375, official_acc=0.961400, FP=0.043159, FN=0.030533
  count_acc_4=0.818182, count_acc_5=0.986486

training results.csv:
  rows=83, requested epochs=160
  val/f1 best epoch=43, best=0.972376, last=0.963891
  val/extra_exist_loss first=0.20858, last=0.63395, max=1.14812
```

Interpretation:

- Supported fact: `extraexist0025` is not a near miss. It regresses official ACC,
  FP, FN, and count accuracy far below `gt4short15`, `count03_under5_03`, and
  the rejected `extraexist005` run.
- Supported fact: unlike `extraexist005`, this smaller gain does not retain a
  useful diagnostic FP/count improvement. Its best count row still has lower
  count accuracy than the selected `gt4short15` row and much lower ACC.
- Supported fact: the sweep uses the correct 363-image official-val JSON and
  `best_metric=official_acc`, so this is valid rejection evidence rather than a
  split mismatch.
- Supported caveat: the run stopped with 83 `results.csv` rows despite
  `epochs=160`, so training completion differs from `extraexist005`. That
  caveat does not make the candidate promotable because the available
  official-val evidence is far below the gate and below all relevant baselines.

Rejected actions:

- Do not send `extraexist0025` to final test.
- Do not tune final-test thresholds or checkpoint choice from this result.
- Do not continue the `gcs_extra_exist` gain sweep by trying `0.0125`, `0.01`,
  or stronger values. The `0.05` run was a diagnostic FP/count tradeoff but
  missed ACC; the `0.025` run lost both ACC and count shape.

Recommended next action:

Stop this loss family for now and return to short-lane score/geometry retention
work. The next train-side mechanism should protect true short side lanes while
controlling extra queries, rather than applying a direct unmatched-query score
penalty. Selection must remain on the same 363-image official-val surface, with
final test closed unless a candidate beats `0.970851`.

Mainline or experiment:

Rejected experimental candidate. No final-test run and no selected decode change.

## 2026-06-21: Add Default-Disabled Short Matched Existence Floor

Decision:

Add an explicit `GCSLoss.exist_loss()` refinement that can floor the matched
existence target for short GT lanes only when geometry and visibility quality
are already acceptable.

Implementation:

```text
ultralytics/utils/gcs_loss.py
  gcs_short_exist_floor = 0.0
  gcs_short_exist_max_visible = 20
  gcs_short_exist_floor_max_ape = 20.0
  gcs_short_exist_floor_min_iou = 0.3

tools/train_gcs.py
ultralytics/cfg/default.yaml
ultralytics/cfg/__init__.py
  CLI/default/config typing for the four parameters above
```

Why:

The rejected `extra_exist_loss` sweep showed that direct unmatched-query
suppression can over-suppress scores. The next smallest train-side mechanism is
to protect true short matched lanes whose APE and visible IoU already indicate
a plausible short lane, so they are not trained to low existence targets because
of unstable short-span quality estimates.

Default behavior:

The feature is disabled by default with `gcs_short_exist_floor=0.0`, so baseline
training behavior is unchanged unless the experiment flag is explicitly set.
It does not add a new logged loss item and does not change model outputs,
decode, postprocess, official metrics, or the K56 fixed-y contract.

Validation evidence:

```text
python -m py_compile ultralytics/utils/gcs_loss.py ultralytics/cfg/__init__.py tools/train_gcs.py
python tools/train_gcs.py --help
direct short-exist-floor tensor behavior check
python -c "from ultralytics.cfg import DEFAULT_CFG_DICT, check_cfg; ..."
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml --imgsz 544 960 --batch 1 --device cpu
```

Recommended experiment:

```text
--gcs-short-exist-floor 0.4
--gcs-short-exist-max-visible 20
--gcs-short-exist-floor-max-ape 20.0
--gcs-short-exist-floor-min-iou 0.3
```

Promote only if the same 363-image official-val surface reaches or exceeds the
current `gt4short15` gate `ACC=0.970851` without materially raising
`FN=0.011708`, while reducing `GT4` short undercount and
`low_score_short_gt` without obvious `spurious_extra` or
`duplicate_like_extra` growth.

Mainline or experiment:

Branch-local experimental option. This implementation note was superseded by
the completed `shortexist04` official-val rejection recorded below.

## 2026-06-21: Reject shortexist04 as an Official-Val Promotion

Decision:

Do not promote
`gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_shortexist04_count03_under5_03`.
Keep `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03` as the
current official-val selected candidate and keep final test closed.

Official-val evidence:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_shortexist04_count03_under5_03_official_val_sweep
rows = 1800
images = 363
gt_json = runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json
best = conf=0.1, point_valid_thr=0.5, nms_dist_px=30.0, max_det=8, min_points=5
official_acc = 0.968966
official_FP = 0.019972
official_FN = 0.014922
official_score = 0.968268
count_acc = 0.950413
count_acc_3 = 0.968610
count_acc_4 = 0.863636
count_acc_5 = 0.972973
count_confusion = 3->2=1, 3->3=216, 3->4=6, 4->3=2, 4->4=57, 4->5=7, 5->4=2, 5->5=72
```

Comparison:

```text
current gt4short15 official-val ACC = 0.970851
shortexist04 official-val ACC       = 0.968966
delta                                = -0.001885

previous count03_under5_03 official-val ACC = 0.969976
shortexist04 official-val ACC               = 0.968966
delta                                         = -0.001010

extraexist005 official-val ACC = 0.969603
shortexist04 official-val ACC  = 0.968966
delta                           = -0.000637

gt4short15 FP/FN/count_acc = 0.022084 / 0.011708 / 0.939394
shortexist04 FP/FN/count_acc = 0.019972 / 0.014922 / 0.950413
```

Interpretation:

- Supported fact: `shortexist04` is valid 363-image official-val evidence, not a
  split mismatch. The sweep used `split=val`, the explicit 363-image official-val
  GT JSON, `imgsz=[544, 960]`, `half=True`, and `best_metric=official_acc`.
- Supported fact: no row in the 1800-row sweep reaches current `gt4short15`
  ACC `0.970851`, previous `count03_under5_03` ACC `0.969976`, or rejected
  `extraexist005` ACC `0.969603`.
- Supported fact: the floor improves FP and count accuracy relative to
  `gt4short15`, but the improvement is diagnostic-only because FN rises by
  `0.003214` and official ACC falls by `0.001885`.
- Supported caveat: the completed run's `args.yaml` records `amp: true`, while
  the original command template included `--no-amp`. This does not change the
  rejection, but any reproducibility note for this run must use the recorded
  args rather than the template.

Rejected actions:

- Do not send `shortexist04` to final test.
- Do not tune final-test thresholds, checkpoint choice, or postprocess from this
  result.
- Do not immediately raise the floor to `0.5` as a blind next step; this run
  already shows the mechanism can trade FP/count for higher FN.

Recommended next action:

If more evidence is needed before another train-side change, run the train/val
failure trace on `shortexist04` using its official-val best decode
`conf=0.1`, `point_valid_thr=0.5`, `nms_dist_px=30.0`, `max_det=8`,
`min_points=5`. Use it only to determine whether `low_score_short_gt` actually
fell and what failure bucket replaced it. Selection remains official-val only,
and final test stays closed.

Mainline or experiment:

Rejected experimental candidate. No final-test run and no selected decode
change.

## 2026-06-21: User-Requested Reporting-Only Final Test Batch

Decision:

Run official TuSimple test once for the recent experiment checkpoints using
each run's official-val selected decode, because the user explicitly requested
test ACC for the recent experiments. Treat these results as reporting-only
evidence, not as a threshold, checkpoint, or postprocess selection surface.

Official-test evidence:

```text
baseline count03_under5_03:
  decode = conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
  ACC=0.965459, FP=0.029439, FN=0.026270, official_score=0.964345, count_acc=0.872753

gt4short2:
  decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=18.0, max_det=5, min_points=4
  ACC=0.965206, FP=0.026432, FN=0.027079, official_score=0.964136, count_acc=0.884256

gt4short15:
  decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
  ACC=0.965369, FP=0.033309, FN=0.029236, official_score=0.964118, count_acc=0.864486

extraexist005:
  decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=5
  ACC=0.965032, FP=0.029661, FN=0.027139, official_score=0.963896, count_acc=0.878864

shortexist04:
  decode = conf=0.1, point_valid_thr=0.5, nms_dist_px=30.0, max_det=8, min_points=5
  ACC=0.964992, FP=0.031381, FN=0.028876, official_score=0.963787, count_acc=0.875988

extraexist0025:
  decode = conf=0.05, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
  ACC=0.961330, FP=0.044764, FN=0.034508, official_score=0.959745, count_acc=0.838605
```

Interpretation:

- Supported fact: within that reporting-only batch, no rejected experiment beat the previous
  `count03_under5_03` final-test ACC `0.965459`.
- Supported fact: `gt4short15` still has the best official-val ACC, but its
  test ACC `0.965369` is below the previous final-test report.
- Supported fact: `extraexist0025` is a clear final-test regression, matching
  its official-val failure.

Rejected actions:

- Do not use these final-test values to tune thresholds, `max_det`,
  `min_points`, NMS, checkpoint choice, or future loss settings.
- Do not promote any rejected experiment based on this reporting-only batch.

Mainline or experiment:

Reporting-only final-test evidence requested by the user. Official-val remains
the only allowed selection surface for future experiments.

## 2026-06-22: Reject count03_under5_00 as an Official-Val Promotion

Decision:

Do not promote
`gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00`. Treat its
official test result as reporting-only evidence from the official-val selected
decode, not as a selection surface.

Official-val evidence:

```text
run = gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00
args = gcs_count=0.3, gcs_count_under5=0.0, gcs_count_under5_min_lanes=5, amp=true, gcs_eval_max_det=8
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00_official_val_sweep
images = 363
best = conf=0.08, point_valid_thr=0.5, nms_dist_px=50.0, max_det=6, min_points=6
official_acc = 0.968578
official_FP = 0.022590
official_FN = 0.016529
official_score = 0.967796
count_acc = 0.955923
count_acc_3 = 0.973094
count_acc_4 = 0.893939
count_acc_5 = 0.959459
count_confusion = 3->3=217, 3->4=5, 3->5=1, 4->3=2, 4->4=59, 4->5=5, 5->4=3, 5->5=71
```

Official-test reporting evidence:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00_official_test_best_from_val/tusimple_official_summary.json
images = 2782
official_acc = 0.965118
official_FP = 0.031908
official_FN = 0.029745
official_score = 0.963885
count_acc = 0.875270
count_acc_2 = 0.400000
count_acc_3 = 0.975862
count_acc_4 = 0.566239
count_acc_5 = 0.826011
count_confusion = 2->2=2, 2->3=3, 3->2=3, 3->3=1698, 3->4=34, 3->5=5, 4->3=110, 4->4=265, 4->5=89, 4->6=4, 5->3=26, 5->4=60, 5->5=470, 5->6=13
```

Comparison:

```text
count03_under5_03 final-test:
  ACC=0.965459, FP=0.029439, FN=0.026270, official_score=0.964345, count_acc=0.872753

count03_under5_00 final-test:
  ACC=0.965118, FP=0.031908, FN=0.029745, official_score=0.963885, count_acc=0.875270

delta count03_under5_00 - count03_under5_03:
  ACC=-0.000341, FP=+0.002469, FN=+0.003475, official_score=-0.000460, count_acc=+0.002517
```

Interpretation:

- Supported fact: setting `gcs_count_under5=0.0` did not pass the promotion gate.
  Official-val ACC is below the previous `count03_under5_03` official-val result
  `0.969976` and the later legacy `gt4short15` gate `0.970851`.
- Supported fact: the final-test total `count_acc` improved slightly, and GT4
  final-test count accuracy improved versus `count03_under5_03`, but this came
  with lower ACC and higher FP/FN. GT5 count accuracy fell.
- Supported caveat: the official-val selected decode uses `max_det=6`, while the
  train args record `gcs_eval_max_det=8`. This follows the official-val
  selection protocol but should be recorded when comparing count behavior.

Rejected actions:

- Do not promote `count03_under5_00`.
- Do not tune final-test thresholds, `max_det`, `min_points`, NMS, checkpoint
  choice, or loss weights from this test breakdown.
- Do not treat the small final-test `count_acc` gain as a count-robustness fix
  without official-val ACC support.

Mainline or experiment:

Rejected experiment result and reporting-only final-test evidence. It does not
change the active rollback code contract.

## 2026-06-22: Add Default-Disabled Visibility-Aware Duplicate Margin Loss

Decision:

Add a default-disabled `duplicate_margin_loss` experiment knob for the next
short-GT4 overcount attempt.

Implementation:

```text
ultralytics/utils/gcs_loss.py
  duplicate_margin_loss log item
  gcs_duplicate_margin = 0.0 default-disabled gain
  pairwise softplus(logit(q-) - logit(q+) + margin) ranking loss
  best-GT ownership filter for q- candidates

tools/train_gcs.py
ultralytics/cfg/default.yaml
ultralytics/cfg/__init__.py
  CLI/default/config typing for gcs_duplicate_* parameters

ultralytics/models/yolo/gcs_lane/train.py
ultralytics/models/yolo/gcs_lane/val.py
  train/val loss item alignment
```

Why:

The earlier `extra_exist_loss` and short matched existence floor experiments
showed that broad score suppression can improve FP/count shape but usually pays
for it with higher FN or score collapse. This loss is narrower: it only
penalizes an unmatched query when the same GT already has a reliable matched
query and the unmatched query looks like a nearby duplicate of that GT. It does
not push down all unmatched queries.

P2 ownership fix:

The first implementation risk was that an unmatched query between adjacent
lanes could be selected independently as `q-` for multiple nearby GT lanes when
`gcs_duplicate_neg_ape_px=120`. The current implementation computes each
query's nearest valid GT by APE and requires `best_gt_for_query == gt_j` before
adding the duplicate-pair loss. This keeps the constraint aligned with the
intended rule: only suppress the duplicate-like query owned by the GT that has
already been explained by a reliable matched `q+`.

Default behavior:

The feature is disabled by default with `gcs_duplicate_margin=0.0`, so baseline
training behavior is unchanged unless the experiment flag is explicitly set.
It does not change model outputs, decode, postprocess, official metrics, K56
fixed-y anchors, or the `--imgsz 544 960` contract.

Recommended experiment:

```text
--gcs-duplicate-margin 0.05
--gcs-duplicate-margin-logit 1.0
--gcs-duplicate-gt-count 4
--gcs-duplicate-short-visible-max 20
--gcs-duplicate-min-overlap 2
--gcs-duplicate-min-visible-iou 0.4
--gcs-duplicate-pos-ape-px 20.0
--gcs-duplicate-neg-ape-px 120.0
--gcs-duplicate-ape-gap-px 5.0
--gcs-duplicate-max-pairs-per-gt 2
```

Validation evidence:

```text
python -m py_compile ultralytics/utils/gcs_loss.py tools/train_gcs.py ultralytics/models/yolo/gcs_lane/train.py ultralytics/models/yolo/gcs_lane/val.py ultralytics/cfg/__init__.py
python tools/train_gcs.py --help
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml --imgsz 544 960 --batch 1 --device cpu
direct duplicate_margin_loss tensor behavior check
best-GT ownership synthetic check for adjacent GT lanes
```

Mainline or experiment:

Branch-local experimental option. Defaults preserve baseline behavior and do
not import later Count/Quality/Survival/near-miss/official-best machinery.

## 2026-06-24: Add Default-Disabled GT3 Extra-Survival Loss

Decision:

Add a default-disabled `gt3_extra_survival_loss` experiment knob as the
smallest branch-safe follow-up to `dupmargin005`.

Implementation:

```text
ultralytics/utils/gcs_loss.py
  gt3_extra_survival_loss log item
  gcs_gt3_extra_survival = 0.0 default-disabled gain
  gcs_gt3_extra_margin_logit = 0.05
  gcs_gt3_extra_topk = 1
  applies only when target_lane_count(...) == 3
  reuses current Hungarian indices
  ranks top unmatched q- below weakest matched q+ in raw logit space

tools/train_gcs.py
ultralytics/cfg/default.yaml
ultralytics/cfg/__init__.py
  CLI/default/config typing for gcs_gt3_extra_* parameters

ultralytics/models/yolo/gcs_lane/train.py
ultralytics/models/yolo/gcs_lane/val.py
  train/val loss item alignment and progress short name gt3_ext
```

Why:

The 363-image official-val failure-mode compare showed that `dupmargin005`
is a useful near-miss but its main count regression is GT3 surplus/spurious
overcount, not duplicate-like extras or short-GT misses:

```text
count-failure duplicate_like_extra: count03=0, dupmargin005=0, gt4short15=1
GT4 count_acc: count03=0.909091 -> dupmargin005=0.939394
GT4 4->5: 5 -> 3
GT3 3->4: 4 -> 11
GT5 5->4: 1 -> 2
total missed_short_gt: 33 -> 28
count-failure-only missed_short_gt: 5 -> 5
```

The loss directly targets the high-score fourth unmatched query in GT3 images:

```text
relu(top_unmatched_logit - min_matched_logit + gcs_gt3_extra_margin_logit)
```

It does not use sigmoid, does not detach `min_matched_logit` in the first
version, and does not apply to `gt_count <= 3`, GT4, or GT5.

Default behavior:

The feature is disabled by default with `gcs_gt3_extra_survival=0.0`, so baseline
training behavior is unchanged unless the experiment flag is explicitly set.
It does not change model outputs, decoder, postprocess, official metrics, K56
fixed-y anchors, or the `--imgsz 544 960` contract.

Recommended first experiment:

```text
run name = gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03
--gcs-duplicate-margin 0.05
--gcs-gt3-extra-survival 0.03
--gcs-gt3-extra-margin-logit 0.05
--gcs-gt3-extra-topk 1
```

Promotion gate:

Use official-val only, with the `dupmargin005` selected decode
`conf=0.05`, `point_valid_thr=0.45`, `nms_dist_px=0.0`, `max_det=6`,
`min_points=6`. The run should clearly reduce GT3 `3->4` below `11`, keep GT4
`4->5` near `3`, keep `count_acc_4` near `0.939394`, avoid increasing GT5
`5->4` beyond `2`, and not raise `missed_short_gt`.

Risks:

Because the first version does not detach `min_matched_logit`, it can both
lower the extra q- and raise the weakest matched q+. If GT5 undercount or score
calibration worsens, the next conservative ablation is smaller gain or
`min_matched_logit.detach()`. Do not choose that from final test; diagnose on
train/val and official-val only.

Validation status:

Local code validation completed:

```text
python -m py_compile ultralytics/utils/gcs_loss.py tools/train_gcs.py ultralytics/models/yolo/gcs_lane/train.py ultralytics/models/yolo/gcs_lane/val.py ultralytics/cfg/__init__.py
python tools/train_gcs.py --help
python -c "from ultralytics.cfg import DEFAULT_CFG_DICT, check_cfg; check_cfg(DEFAULT_CFG_DICT.copy())"
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml --imgsz 544 960 --batch 1 --device cpu
direct synthetic GT3 forward/backward check: loss_names_len=16, gt3_extra_unweighted=0.45, total=0.0135 with gain 0.03, progress_has_gt3_ext=True
```

Formal training and official-val metrics completed; see the follow-up rejection
decision below.

Mainline or experiment:

Branch-local experimental option. Defaults preserve baseline behavior and do
not import later Count/Quality/Survival/near-miss/official-best machinery.

## 2026-06-24: Reject dupmargin005_gt3extra003 as an Official-Val Promotion

Decision:

Do not promote
`gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03`.
The GT3-only extra-survival loss improved the intended `3->4` failure relative
to `dupmargin005`, but it missed all relevant official-val ACC gates and hurt
GT4 count shape relative to its parent run.

Training evidence:

```text
run = gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03
args = epochs=160, batch=32, workers=8, amp=true, gcs_imgsz=544 960, gcs_duplicate_margin=0.05, gcs_gt3_extra_survival=0.03, gcs_gt3_extra_margin_logit=0.05, gcs_gt3_extra_topk=1
results.csv includes train/gt3_extra_survival_loss and val/gt3_extra_survival_loss; final rows are non-zero, so the loss participated in training and validation logging.
```

Official-val evidence:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_official_val_sweep
selected decode = conf=0.05, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
split = val
gt_json = runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json
images = 363
official_acc = 0.969665
official_FP = 0.020615
official_FN = 0.014004
official_score = 0.968973
count_acc = 0.961433
count_acc_3 = 0.968610
count_acc_4 = 0.909091
count_acc_5 = 0.986486
count_confusion = 3->3=216, 3->4=7, 4->3=2, 4->4=60, 4->5=4, 5->4=1, 5->5=73
```

The sweep summary's first `best` row uses `conf=0.005`, but the selected
`conf=0.05` row ties it on official ACC, FP, FN, official score, count
accuracy, and count confusion. The `conf=0.05` selected decode is therefore
official-val evidence, not final-test tuning.

Failure-mode evidence:

```text
summary = runs/gcs_lane/failure_compare/dupmargin005_gt3extra003_compare/summary.json
script = tools/compare_tusimple_failure_modes.py
surface = 363-image official-val subset
```

Count-failure-only buckets:

```text
count03:      failure_images=11, confusion={3->4:4, 4->3:1, 4->5:5, 5->4:1}, duplicate_like_extra=0, spurious_extra=15, missed_short_gt=5, missed_gt=3
gt4short15:   failure_images=22, confusion={3->4:8, 4->3:1, 4->5:8, 4->6:1, 5->4:3, 5->6:1}, duplicate_like_extra=1, spurious_extra=24, missed_short_gt=8, missed_gt=2
dupmargin005: failure_images=17, confusion={3->4:11, 4->3:1, 4->5:3, 5->4:2}, duplicate_like_extra=0, spurious_extra=18, missed_short_gt=5, missed_gt=2
gt3extra003:  failure_images=14, confusion={3->4:7, 4->3:2, 4->5:4, 5->4:1}, duplicate_like_extra=0, spurious_extra=16, missed_short_gt=5, missed_gt=3
```

All-image strict buckets:

```text
spurious_extra:       count03=44, gt4short15=45, dupmargin005=44, gt3extra003=50
duplicate_like_extra: count03=0,  gt4short15=1,  dupmargin005=0,  gt3extra003=0
missed_short_gt:      count03=33, gt4short15=27, dupmargin005=28, gt3extra003=36
```

Reporting-only final-test evidence:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_official_val_selected_decode_test/tusimple_official_summary.json
split = test
decode = conf=0.05, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
images = 2782
official_acc = 0.965077
official_FP = 0.030494
official_FN = 0.027528
official_score = 0.963917
count_acc = 0.875988
count_acc_3 = 0.976437
count_acc_4 = 0.538462
count_acc_5 = 0.852373
count_confusion includes 3->4=34, 4->5=93, 5->4=46
```

Comparison:

```text
official-val ACC:
  gt4short15        = 0.970851
  dupmargin005      = 0.970272
  count03_under5_03 = 0.969976
  gt3extra003       = 0.969665

official-val count shape:
  GT3 3->4: dupmargin005 11 -> gt3extra003 7
  GT4 4->5: dupmargin005 3  -> gt3extra003 4
  GT5 5->4: dupmargin005 2  -> gt3extra003 1
```

Interpretation:

- Supported fact: the target GT3 surplus query penalty helped, but not enough.
  `3->4` falls below `dupmargin005` and `gt4short15`, yet remains above
  `count03`.
- Supported fact: GT5 official-val count shape is not harmed and recovers to
  the `count03` `5->4=1` level.
- Supported fact: GT4 shape is worse than `dupmargin005`: `count_acc_4`
  `0.939394 -> 0.909091`, and `4->5` `3 -> 4`.
- Supported fact: all-image `missed_short_gt` worsens to `36`, so this is not
  a clean GT3-only fix.
- Supported fact: official-val ACC is below every relevant comparator, so the
  run is not promotable under the branch protocol.

Rejected actions:

- Do not promote from the reporting-only test result.
- Do not tune `conf`, `point_valid_thr`, NMS, `max_det`, `min_points`,
  checkpoint choice, or loss gain from final test.
- Do not continue with a blind larger `gcs_gt3_extra_survival` gain.

Recommended next action:

Keep final test closed for tuning. If the line is revisited, use train/val or
official-val diagnostics to check whether the non-detached
`min_matched_logit` side of the hinge is raising weak matched GT3 lanes and
increasing `missed_short_gt`. A smaller gain or
`min_matched_logit.detach()` is a possible official-val ablation only after
that diagnostic.

Mainline or experiment:

Rejected experimental candidate and reporting-only final-test evidence. It
does not change the active branch protocol.

## 2026-06-23: Reject farspur001 + gt5rank001 Combined Ranking Run

Decision:

Do not promote
`gcs_yolo_lane_s_tusimple_fixed_y_farspur001_gt5rank001_count03_under5_03`.
Do not run final test for this candidate. Treat the run as evidence that the
current combined far-spurious survival plus GT5 rank-consistency setting damages
query score/ranking calibration.

Official-val evidence:

```text
run = gcs_yolo_lane_s_tusimple_fixed_y_farspur001_gt5rank001_count03_under5_03
args = epochs=160, batch=32, workers=8, amp=true, gcs_count=0.3, gcs_count_under5=0.3, gcs_far_spurious_survival=0.01, gcs_gt5_rank_consistency=0.01
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_farspur001_gt5rank001_count03_under5_03_official_val_sweep
images = 363
best = conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=8, min_points=4
official_acc = 0.965673
official_FP = 0.033747
official_FN = 0.022727
official_score = 0.964544
count_acc = 0.917355
count_acc_3 = 0.964126
count_acc_4 = 0.757576
count_acc_5 = 0.918919
count_confusion = 3->3=215, 3->4=7, 3->5=1, 4->3=3, 4->4=50, 4->5=13, 5->3=1, 5->4=1, 5->5=68, 5->6=4
```

Sweep checks:

```text
best official_acc row = 0.965673, FP=0.035675, FN=0.022727, count_acc=0.909091
best official_score row = 0.965609, FP=0.028466, FN=0.022727, count_acc=0.944904
best count_acc row = 0.965490, FP=0.025207, FN=0.023416, count_acc=0.958678
minimum-FN row = 0.964452, FP=0.034114, FN=0.021120, count_acc=0.917355
minimum-FP row = 0.961070, FP=0.019697, FN=0.030303, count_acc=0.922865
```

Comparison:

```text
official-val ACC:
  gt4short15        = 0.970851
  dupmargin005      = 0.970272
  count03_under5_03 = 0.969976
  spurmargin003     = 0.969316
  count03_under5_00 = 0.968578
  shortpos          = 0.968144
  farspur001_gt5rank001 = 0.965673
```

Interpretation:

- Supported fact: the candidate is worse than every relevant official-val
  comparator by a large margin. It misses `count03_under5_03` by `0.004303`
  ACC and `gt4short15` by `0.005178`.
- Supported fact: the best selected confidence is `0.005`, matching the
  score-calibration warning pattern seen in other rejected runs. No row in the
  1800-row sweep provides an acceptable FP/FN tradeoff.
- Supported fact: both FP and FN are bad. The minimum-FN row still has
  `FN=0.021120`, far above `count03_under5_03` (`0.014463`) and `gt4short15`
  (`0.011708`). The minimum-FP row pushes FN to `0.030303`.
- Supported fact: GT4 count accuracy drops to `0.757576` on the selected row,
  so the known short/ambiguous GT4 count bottleneck is worse, not better.
- Likely cause: `far_spurious_survival_loss` is an absolute unmatched-q logit
  suppression term for selected GT3/GT4 images, while
  `gt5_rank_consistency_loss` ranks the weakest matched GT5 q+ above high-score
  unmatched q- without requiring that weakest q+ to pass geometry or visible-IoU
  quality. The combined pressure can suppress useful alternate queries and
  hard-rank weak GT5 matches, causing global score/ranking instability.

Rejected actions:

- Do not promote the run.
- Do not run final test.
- Do not tune final-test thresholds, NMS, `max_det`, `min_points`, checkpoint
  choice, or combined loss gains.
- Do not continue by blindly sweeping `gcs_far_spurious_survival` and
  `gcs_gt5_rank_consistency` together.

Recommended next action:

The train/val query-trace diagnostic is now self-contained in
`tools/diagnose_gt4_short_failure_queries.py` and no longer imports missing
legacy `tools.diagnose_tusimple_count_confusion`. Run train/val-only traces for
`GT4` and `GT5` with each run's official-val selected decode and compare
`matched_qpos` versus `unmatched_qminus` distributions against
`count03_under5_03`, `dupmargin005`, and `gt4short15`. If another experiment is
still justified, ablate one mechanism at a time:

```text
far_spurious_survival=0.01, gt5_rank_consistency=0.0
far_spurious_survival=0.0, gt5_rank_consistency=0.01 with geometry/visible-IoU q+ quality gates
```

Selection must remain on the 363-image official-val surface.

Mainline or experiment:

Rejected experimental candidate. It does not change the selected checkpoint,
decode policy, final-test status, or branch contract.

## 2026-06-23: Add Default-Disabled Positive Short-Lane Point/Visibility Losses

Decision:

Add two default-disabled positive short-lane supervision terms:
`lane_balanced_point_loss` and `short_valid_recall_loss`.

Implementation:

```text
ultralytics/utils/gcs_loss.py
  lane_balanced_point_loss log item
  gcs_lane_balanced_point = 0.0 default-disabled gain
  matched lane point error is averaged per lane, then across lanes

  short_valid_recall_loss log item
  gcs_short_valid_recall = 0.0 default-disabled gain
  gcs_short_valid_max_visible = 20
  gcs_short_valid_min_visible = 4
  gcs_short_valid_max_ape_px = 40.0
  gcs_short_valid_min_visible_iou = 0.3
  positive-only BCE on GT-visible anchors for matched short lanes whose matched
  q+ passes APE and visible-IoU gates

tools/train_gcs.py
ultralytics/cfg/default.yaml
ultralytics/cfg/__init__.py
  CLI/default/config typing for gcs_lane_balanced_point and gcs_short_valid_* parameters

ultralytics/models/yolo/gcs_lane/train.py
ultralytics/models/yolo/gcs_lane/val.py
  train/val loss item alignment
```

Why:

The follow-up target is true short-lane learning, not another unmatched-query
suppression loss. The standard `point_loss` normalizes over all visible matched
points in an image, so long lanes naturally dominate short side lanes. The
extra lane-balanced term gives each matched GT lane equal geometric weight when
enabled. The short-valid recall term only boosts GT-visible anchors for matched
short lanes whose matched q+ already has plausible geometry and visible-IoU,
and does not add extra penalties on invisible anchors.

Default behavior:

Both gains default to `0.0`, so baseline training behavior is unchanged unless
the experiment flags are explicitly set. The change does not alter model
outputs, decode, NMS, postprocess, official metrics, K56 fixed-y anchors, or
the `--imgsz 544 960` contract.

Recommended first experiment:

```text
--gcs-count 0.3
--gcs-count-under5 0.3
--gcs-duplicate-margin 0.0
--gcs-spurious-margin 0.0
--gcs-lane-balanced-point 3.0
--gcs-short-valid-recall 0.5
--gcs-short-valid-max-visible 20
--gcs-short-valid-min-visible 4
--gcs-short-valid-max-ape-px 40.0
--gcs-short-valid-min-visible-iou 0.3
```

Promotion gate:

Use official-val only. Focus on `geometry_miss_short_gt` and
`min_points_visibility_short_gt` in train/val failure traces, while requiring
official-val ACC/FN and GT4/GT5 count behavior to stay within the recorded
branch gates.

Mainline or experiment:

Branch-local experimental option. Defaults preserve baseline behavior and do
not import later Count/Quality/Survival/near-miss/official-best machinery.

## 2026-06-23: Reject shortpos Positive Short-Lane Loss Run

Decision:

Do not promote
`gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03`. Treat its
official test result as reporting-only evidence from the official-val selected
decode, not as a selection surface.

Official-val evidence:

```text
run = gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03
args = epochs=160, batch=32, workers=4, no_amp=true, gcs_count=0.3, gcs_count_under5=0.3, gcs_lane_balanced_point=3.0, gcs_short_valid_recall=0.5
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03_official_val_sweep
images = 363
best = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=5
official_acc = 0.968144
official_FP = 0.027319
official_FN = 0.017218
official_score = 0.967253
count_acc = 0.947658
count_acc_3 = 0.959641
count_acc_4 = 0.878788
count_acc_5 = 0.972973
count_confusion = 3->3=214, 3->4=9, 4->3=1, 4->4=58, 4->5=7, 5->4=1, 5->5=72, 5->6=1
```

Official-test reporting evidence:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=5
images = 2782
official_acc = 0.963067
official_FP = 0.038330
official_FN = 0.030763
official_score = 0.961685
count_acc = 0.861251
count_acc_2 = 0.400000
count_acc_3 = 0.957471
count_acc_4 = 0.547009
count_acc_5 = 0.829525
pred_lanes_hist = 2=4, 3=1801, 4=362, 5=574, 6=41
gt_lanes_hist = 2=5, 3=1740, 4=468, 5=569
count_confusion = 2->2=2, 2->3=2, 2->4=1, 3->2=2, 3->3=1666, 3->4=63, 3->5=7, 3->6=2, 4->3=109, 4->4=256, 4->5=95, 4->6=8, 5->3=24, 5->4=42, 5->5=472, 5->6=31
```

Comparison:

```text
official-val ACC:
  gt4short15        = 0.970851
  dupmargin005      = 0.970272
  count03_under5_03 = 0.969976
  spurmargin003     = 0.969316
  count03_under5_00 = 0.968578
  shortpos          = 0.968144

official-test reporting:
  dupmargin005:      ACC=0.965702, FP=0.029493, FN=0.027348, official_score=0.964565, count_acc=0.865924
  count03_under5_03: ACC=0.965459, FP=0.029439, FN=0.026270, official_score=0.964345, count_acc=0.872753
  gt4short15:        ACC=0.965369, FP=0.033309, FN=0.029236, official_score=0.964118, count_acc=0.864486
  count03_under5_00: ACC=0.965118, FP=0.031908, FN=0.029745, official_score=0.963885, count_acc=0.875270
  spurmargin003:     ACC=0.964522, FP=0.033591, FN=0.028636, official_score=0.963277, count_acc=0.869878
  shortpos:          ACC=0.963067, FP=0.038330, FN=0.030763, official_score=0.961685, count_acc=0.861251
```

Interpretation:

- Supported fact: `shortpos` failed before final-test reporting because its
  official-val ACC is below every relevant comparator and below the first-pass
  gate `ACC >= 0.970272`.
- Supported fact: the one-shot final-test report confirms broad degradation:
  ACC, FP, FN, official score, and total count accuracy are all worse than the
  older `count03_under5_03` report.
- Supported fact: the selected official-val confidence is `0.005`, which means
  the positive short-lane losses did not preserve usable query score
  calibration. This resembles score-collapse symptoms from earlier suppression
  experiments, even though the mechanism is positive-only.
- Supported fact: final-test count errors are mixed rather than targeted:
  GT3 has more overcount (`3->4=63`), GT4 still has both undercount and
  overcount (`4->3=109`, `4->5=95`, `4->6=8`), and GT5 has many extra-six
  cases (`5->6=31`).
- Caveat: the official-test decode uses `max_det=6` while the train args record
  `gcs_eval_max_det=8`; this is valid because `max_det=6` came from
  official-val selection, not from final-test tuning.

Rejected actions:

- Do not promote `shortpos`.
- Do not tune final-test thresholds, `max_det`, `min_points`, NMS, checkpoint
  choice, or loss gains from this test report.
- Do not continue this exact `lane_balanced_point=3.0` plus
  `short_valid_recall=0.5` setting. It did not solve the short-lane bottleneck
  and damaged score/count stability.

Pre-trace next action, now completed below:

Keep final test closed. The requested train/val-only failure trace for the
`shortpos` official-val selected decode was completed and compared directly
against `count03_under5_03` and `dupmargin005`; see the follow-up decision
below. It shows the regression comes from extra-query ranking/calibration more
than from the targeted short-lane retention buckets.

Mainline or experiment:

Rejected experimental candidate and reporting-only final-test evidence. It
does not change the active branch protocol.

## 2026-06-23: Complete shortpos Train/Val Failure Trace

Decision:

Use the train/val-only failure trace to reject the exact `shortpos` loss setting
as a useful follow-up. It reduces several short-lane miss-like buckets, but the
errors move into extra-query count failures and overall count accuracy drops.

Diagnostic evidence:

```text
summary = runs/gcs_lane/shortpos_failure_trace_train_val_compare/summary.json
splits = train + val only
dataset = datasets/tusimple_fixed_y_k56_960x544

count03_under5_03 decode = conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
dupmargin005 decode      = conf=0.05, point_valid_thr=0.45, nms_dist_px=0.0, max_det=6, min_points=6
shortpos decode          = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=5
```

All train+val comparison:

```text
count03_under5_03: count_acc=0.961390, failure_images=140
  missing: geometry_miss_short_gt=16, low_score_short_gt=12, min_points_visibility_short_gt=2
  extra:   spurious_extra=60, duplicate_like_extra=11, duplicate_like_extra_short_gt=52
  confusion: 3->4=25, 4->3=28, 4->5=75, 4->6=1, 5->4=9, 5->6=0

dupmargin005: count_acc=0.959184, failure_images=148
  missing: geometry_miss_short_gt=18, low_score_short_gt=4, min_points_visibility_short_gt=6
  extra:   spurious_extra=82, duplicate_like_extra=19, duplicate_like_extra_short_gt=52
  confusion: 3->4=33, 4->3=18, 4->5=70, 4->6=10, 5->4=13, 5->6=0

shortpos: count_acc=0.942637, failure_images=208
  missing: geometry_miss_short_gt=15, low_score_short_gt=0, min_points_visibility_short_gt=0
  extra:   spurious_extra=112, duplicate_like_extra=37, duplicate_like_extra_short_gt=85
  confusion: 3->4=55, 4->3=15, 4->5=108, 4->6=14, 5->4=4, 5->6=4
```

Integrated conclusion:

- Supported fact: `shortpos` did improve the targeted short-lane retention
  symptoms: `low_score_short_gt` and `min_points_visibility_short_gt` drop to
  zero, and `geometry_miss_short_gt` is slightly lower than both comparators.
- Supported fact: the improvement is not usable because count failures increase
  sharply. Versus `count03_under5_03`, `shortpos` adds `+68` failure images,
  `+52` `spurious_extra`, `+26` `duplicate_like_extra`, and `+33`
  `duplicate_like_extra_short_gt`.
- Supported fact: the transfer is mainly overcount: `3->4` grows by `+30`,
  `4->5` by `+33`, and `4->6` by `+13` versus `count03_under5_03`. Versus
  `dupmargin005`, `4->5` grows by `+38`.
- Hypothesis: the positive matched-lane geometry/valid terms make short lanes
  easier to retain but do not calibrate query existence or rank extras below
  true lanes. The selected `conf=0.005` is consistent with this score/ranking
  failure.
- Decision: do not continue this exact `gcs_lane_balanced_point=3.0` plus
  `gcs_short_valid_recall=0.5` line, and do not rescue it with test thresholds,
  NMS, `max_det`, or `min_points`.

Smallest safe next action:

Return to train-side score/ranking calibration only after isolating which
surviving extras are far spurious versus near-duplicate and whether the issue is
global logit calibration or assignment/ranking. Do not add another positive
short-lane retention loss on top of `shortpos`.

Mainline or experiment:

Diagnostic-only result for a rejected experimental candidate. It does not
promote a checkpoint, change decode, or reopen final test.

## 2026-06-23: Reject spurmargin003 as an Official-Val Promotion

Decision:

Do not promote
`gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03`. Treat its
official test result as reporting-only evidence from the official-val selected
decode, not as a selection surface.

Official-val evidence:

```text
run = gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03
args = epochs=160, batch=32, workers=4, amp=false, gcs_count=0.3, gcs_count_under5=0.3, gcs_spurious_margin=0.03
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03_official_val_sweep
images = 363
best = conf=0.02, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=5
official_acc = 0.969316
official_FP = 0.023095
official_FN = 0.014922
official_score = 0.968556
count_acc = 0.958678
count_acc_3 = 0.973094
count_acc_4 = 0.878788
count_acc_5 = 0.986486
count_confusion = 3->3=217, 3->4=6, 4->3=1, 4->4=58, 4->5=7, 5->4=1, 5->5=73
```

Official-test reporting evidence:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
decode = conf=0.02, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=5
images = 2782
official_acc = 0.964522
official_FP = 0.033591
official_FN = 0.028636
official_score = 0.963277
count_acc = 0.869878
count_acc_2 = 0.200000
count_acc_3 = 0.964368
count_acc_4 = 0.527778
count_acc_5 = 0.868190
pred_lanes_hist = 2=4, 3=1820, 4=346, 5=598, 6=14
gt_lanes_hist = 2=5, 3=1740, 4=468, 5=569
count_confusion = 2->2=1, 2->3=4, 3->2=3, 3->3=1678, 3->4=53, 3->5=5, 3->6=1, 4->3=118, 4->4=247, 4->5=99, 4->6=4, 5->3=20, 5->4=46, 5->5=494, 5->6=9
```

Comparison:

```text
official-val ACC:
  count03_under5_03 = 0.969976
  dupmargin005      = 0.970272
  gt4short15        = 0.970851
  spurmargin003     = 0.969316

official-test reporting:
  count03_under5_03: ACC=0.965459, FP=0.029439, FN=0.026270, official_score=0.964345, count_acc=0.872753
  gt4short15:        ACC=0.965369, FP=0.033309, FN=0.029236, official_score=0.964118, count_acc=0.864486
  dupmargin005:      ACC=0.965702, FP=0.029493, FN=0.027348, official_score=0.964565, count_acc=0.865924
  spurmargin003:     ACC=0.964522, FP=0.033591, FN=0.028636, official_score=0.963277, count_acc=0.869878
```

Interpretation:

- Supported fact: `spurmargin003` is not a near miss against the current
  promotion gate. It is below `gt4short15` by `0.001535` official-val ACC and
  below `dupmargin005` by `0.000956`.
- Supported fact: it is also below the older `count03_under5_03`
  official-val ACC by `0.000660`, so it does not even clear the previous
  baseline gate.
- Supported fact: the reporting-only final-test result is lower than all three
  relevant comparators on ACC and official score. Test therefore offers no
  protocol-valid reason to keep this run alive.
- Supported fact: the only favorable sign is GT5 test count accuracy
  (`0.868190`), which is higher than `dupmargin005` and the previous
  `count03_under5_03` report. This does not offset the official-val ACC
  regression and the weaker GT4 count behavior.
- Supported fact: the selected official-val confidence is low (`conf=0.02`),
  which is a score-calibration warning for this loss setting.
- Supported caveat: the official-test decode uses `max_det=6` while train args
  record `gcs_eval_max_det=8`; this is valid because `max_det=6` came from
  official-val selection, not final-test tuning.

Rejected actions:

- Do not promote `spurmargin003`.
- Do not tune final-test thresholds, `max_det`, `min_points`, NMS, checkpoint
  choice, or loss weights from this test report.
- Do not continue by sweeping `gcs_spurious_margin` gains unless a separate
  train/val diagnostic identifies a narrower failure mode. This run does not
  justify a blind gain sweep.

Recommended next action:

Keep final test closed. Return to train/val diagnostics if continuing this
line, with emphasis on why the intended GT3/GT4 spurious suppression did not
transfer to official-val ACC and why GT4 remains weak despite better GT5 count
shape. Specifically compare `4->3`/`4->5`, `spurious_extra`,
`duplicate_like_extra`, `low_score_short_gt`, and `geometry_miss_short_gt`
against `dupmargin005` and `count03_under5_03`. Selection must remain
official-val only.

Mainline or experiment:

Rejected experimental candidate and reporting-only final-test evidence. It
does not change the active branch protocol.

## 2026-06-22: Review Spurious Margin Loss Implementation

Decision:

Keep the current `spurious_margin_loss` implementation for the first
GT3/GT4-only spurious-extra experiment. The implementation review found no
blocking bug in the gating, pair selection, default behavior, or train/val loss
logging alignment.

Checked behavior:

```text
default-disabled gain:
  gcs_spurious_margin = 0.0 returns zero and preserves baseline behavior

GT-count gate:
  default gcs_spurious_gt_counts = [3, 4]
  GT3 synthetic sample triggers
  GT5 synthetic sample returns zero

q- gate:
  only unmatched queries can be selected
  far-spurious gate is best_ape > 50px OR best_visible_iou < 0.2
  near-GT duplicate-like exclusion is best_ape <= 50px AND best_visible_iou >= 0.4

ranking direction:
  softplus(logit(q-) - logit(q+) + margin)
```

Validation evidence:

```text
python -m py_compile ultralytics/utils/gcs_loss.py tools/train_gcs.py ultralytics/models/yolo/gcs_lane/train.py ultralytics/models/yolo/gcs_lane/val.py ultralytics/cfg/__init__.py
python tools/train_gcs.py --help
python -c "from ultralytics.cfg import DEFAULT_CFG_DICT, check_cfg; check_cfg(DEFAULT_CFG_DICT.copy())"
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml --imgsz 544 960 --batch 1 --device cpu
direct tensor checks for GT3 trigger, GT5 gate, default-disabled zero, near-GT duplicate exclusion, and GCSLoss.forward loss-vector length
```

Risk to watch:

The current `far_spurious` definition intentionally follows the requested OR
rule: `best_ape > 50px` OR `best_visible_iou < 0.2`. If official-val FN or GT5
`5->4` worsens in the next run, inspect whether queries with acceptable
geometry but poor point-valid overlap are being treated as spurious. Do not
change this from final-test evidence; diagnose it on train/val and official-val
only.

Mainline or experiment:

Implementation-review note for a default-disabled branch-local experiment
knob. It does not promote a candidate or change the active protocol.

## 2026-06-22: Report dupmargin005 Final Test and Reject Official-Val Promotion

Decision:

Do not promote
`gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03` as the current
selected candidate. It is valid official-val evidence and a useful near-miss,
but it does not beat the strongest official-val gate from `gt4short15`.

Official-val evidence:

```text
run = gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03
args = epochs=160, batch=32, workers=8, amp=true, gcs_count=0.3, gcs_count_under5=0.3, gcs_duplicate_margin=0.05
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_val_sweep
images = 363
best = conf=0.05, point_valid_thr=0.45, nms_dist_px=0.0, max_det=6, min_points=6
official_acc = 0.970272
official_FP = 0.024564
official_FN = 0.016070
official_score = 0.969459
count_acc = 0.953168
count_acc_3 = 0.950673
count_acc_4 = 0.939394
count_acc_5 = 0.972973
count_confusion = 3->3=212, 3->4=11, 4->3=1, 4->4=62, 4->5=3, 5->4=2, 5->5=72
```

The sweep summary selected the row above. There was an `official_acc` tie with
a lower-confidence row, but the selected row had better `official_score` and
better count accuracy.

Official-test reporting evidence:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
decode = conf=0.05, point_valid_thr=0.45, nms_dist_px=0.0, max_det=6, min_points=6
images = 2782
official_acc = 0.965702
official_FP = 0.029493
official_FN = 0.027348
official_score = 0.964565
count_acc = 0.865924
count_acc_2 = 0.200000
count_acc_3 = 0.971839
count_acc_4 = 0.547009
count_acc_5 = 0.810193
pred_lanes_hist = 2=3, 3=1837, 4=371, 5=557, 6=14
gt_lanes_hist = 2=5, 3=1740, 4=468, 5=569
count_confusion = 2->2=1, 2->3=3, 2->4=1, 3->2=2, 3->3=1691, 3->4=41, 3->5=6, 4->3=118, 4->4=256, 4->5=90, 4->6=4, 5->3=25, 5->4=73, 5->5=461, 5->6=10
```

Comparison:

```text
official-val ACC:
  count03_under5_03 = 0.969976
  gt4short15        = 0.970851
  dupmargin005      = 0.970272
  delta dupmargin005 - count03_under5_03 = +0.000296
  delta dupmargin005 - gt4short15        = -0.000579

official-test reporting:
  count03_under5_03: ACC=0.965459, FP=0.029439, FN=0.026270, official_score=0.964345, count_acc=0.872753
  gt4short15:        ACC=0.965369, FP=0.033309, FN=0.029236, official_score=0.964118, count_acc=0.864486
  count03_under5_00: ACC=0.965118, FP=0.031908, FN=0.029745, official_score=0.963885, count_acc=0.875270
  dupmargin005:      ACC=0.965702, FP=0.029493, FN=0.027348, official_score=0.964565, count_acc=0.865924
```

Interpretation:

- Supported fact: `dupmargin005` gives the strongest reporting-only final-test
  ACC among the runs recorded through 2026-06-22, but test is not a selection
  surface.
- Supported fact: the run clears the older `count03_under5_03` official-val ACC
  but misses the `gt4short15` official-val gate by `0.000579`, so it is not a
  promotion under the branch protocol.
- Supported fact: relative to `gt4short15`, the test report improves ACC, FP,
  FN, GT4 count accuracy, and total count accuracy.
- Supported fact: relative to `count03_under5_03`, the test report improves ACC
  and official score only slightly, while FN rises and total count accuracy
  falls. The largest count regression is GT5 (`0.831283 -> 0.810193`) with more
  `5->4` errors (`52 -> 73`).
- Supported caveat: the official-test decode uses `max_det=6` while the
  training args record `gcs_eval_max_det=8`; this is valid because `max_det=6`
  came from official-val selection, not from test tuning.

Rejected actions:

- Do not promote `dupmargin005` from its final-test ACC.
- Do not tune `conf`, `point_valid_thr`, NMS, `max_det`, `min_points`,
  checkpoint choice, or loss weights from this final-test report.
- Do not claim the duplicate/count problem is solved; count behavior remains
  mixed, especially on GT5.

Recommended next action:

Keep final test closed. If more work is needed, run train/val-only failure
traces for the `dupmargin005` official-val selected decode
`conf=0.05`, `point_valid_thr=0.45`, `nms_dist_px=0.0`, `max_det=6`,
`min_points=6`, then compare failure buckets against `count03_under5_03` and
`gt4short15`. Only propose a follow-up loss or sampler change if the train/val
evidence identifies a specific duplicate-margin failure mode.

Mainline or experiment:

Rejected near-miss experimental candidate. The final-test report is
reporting-only evidence requested by the user and does not alter the selected
candidate or branch protocol.

## 2026-06-22: Add Default-Disabled Spurious Margin Loss

Decision:

Add a default-disabled `spurious_margin_loss` experiment knob that ranks far
unmatched spurious queries below reliable matched queries only in selected
GT-count images.

Implementation:

```text
ultralytics/utils/gcs_loss.py
  spurious_margin_loss log item
  gcs_spurious_margin = 0.0 default-disabled gain
  pairwise softplus(logit(q-) - logit(q+) + margin) ranking loss
  enabled only for gcs_spurious_gt_counts, default [3, 4]
  reliable q+: matched, GT visible points >= 2, APE <= 20px, visible IoU >= 0.4
  spurious q-: unmatched, far from all GT by APE/visible-IoU gate, and not near-GT duplicate-like

tools/train_gcs.py
ultralytics/cfg/default.yaml
ultralytics/cfg/__init__.py
  CLI/default/config typing for gcs_spurious_* parameters

ultralytics/models/yolo/gcs_lane/train.py
ultralytics/models/yolo/gcs_lane/val.py
  train/val loss item alignment
```

Why:

The `dupmargin005` follow-up target is not all unmatched queries and not
duplicate-like near-GT queries. The specific train-side hypothesis is that
high-score far spurious extras in GT3/GT4 images should be ranked below reliable
matched lanes without directly forcing all unmatched query logits toward zero.

Default behavior:

The feature is disabled by default with `gcs_spurious_margin=0.0`, so baseline
training behavior is unchanged unless the experiment flag is explicitly set. It
does not change model outputs, decode, postprocess, official metrics, K56
fixed-y anchors, or the `--imgsz 544 960` contract. It does not apply to GT5 by
default.

Recommended first experiment:

```text
--gcs-spurious-margin 0.03
--gcs-spurious-margin-logit 1.0
--gcs-spurious-gt-counts 3,4
--gcs-spurious-pos-ape-px 20.0
--gcs-spurious-pos-min-visible-iou 0.4
--gcs-spurious-neg-min-ape-px 50.0
--gcs-spurious-neg-max-visible-iou 0.2
--gcs-spurious-duplicate-ape-px 50.0
--gcs-spurious-duplicate-visible-iou 0.4
--gcs-spurious-max-pairs-per-image 4
```

Promotion gate:

Use official-val only. A candidate must mainly reduce the train/val
`spurious_extra` counts for GT3 `3->4` and GT4 `4->5` without worsening GT5
`5->4` geometry/min-points buckets, and must reach at least the current
official-val gate `ACC >= 0.970851` before any final-test reporting.

Validation evidence:

```text
python -m py_compile ultralytics/utils/gcs_loss.py ultralytics/cfg/__init__.py tools/train_gcs.py ultralytics/models/yolo/gcs_lane/train.py ultralytics/models/yolo/gcs_lane/val.py
python tools/train_gcs.py --help
python -c "from ultralytics.cfg import DEFAULT_CFG_DICT, check_cfg; check_cfg(DEFAULT_CFG_DICT.copy())"
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml --imgsz 544 960 --batch 1 --device cpu
direct spurious_margin_loss tensor behavior checks for GT3 trigger, default-disabled zero, GT5 gate, and duplicate-like exclusion
direct GCSLoss.forward loss-vector alignment check
```

Mainline or experiment:

Branch-local experimental option. Defaults preserve baseline behavior and do
not import later Count/Quality/Survival/near-miss/official-best machinery.

## 2026-06-24: Add Count-Oracle and Optional Count-Guided Decode Diagnostics

Decision:

Stop treating the current final-test count failure as a threshold-search
problem. Add minimal official-eval diagnostics that can compare normal decode
against oracle-count topK and, only when a real count head is present, optional
count-guided topK.

Implementation:

```text
tools/eval_tusimple_official.py
  --oracle-count-topk
  --dump-count-head-stats
  --count-guided-topk
  --count-guided-min-prob
  --count-guided-allowed-counts
```

The new paths use the same normal decoded candidate pool:

```text
conf = 0.005
point_valid_thr = 0.5
nms_dist_px = 0.0
max_det = 8
min_points = 6
```

Default behavior:

All new behavior is default-off. Normal `tools/eval_tusimple_official.py`
decode remains unchanged when the new flags are not passed. The script still
writes the legacy `tusimple_official_summary.json`, and now also writes
`summary.json` plus `count_confusion.csv` for the evaluated mode set.

Protocol:

`--oracle-count-topk` is diagnostic-only. It uses GT lane count after normal
candidate generation and is not valid for formal submission, threshold
selection, checkpoint selection, or final-test tuning.

`--count-guided-topk` only changes validation/inference final lane count after
normal candidate generation. It does not change model structure, training loss,
candidate geometry, K56 labels, Q=12, or the `--imgsz 544 960` contract. The
first allowed count set is `3,4,5` to avoid forcing many `2` or `6` lane
outputs.

Count-head status:

The active 5-25-3 K56 model output contract does not include independent count
logits. The existing count loss is based on `sum(sigmoid(pred_logits))`.
Therefore `--dump-count-head-stats` reports `supported=false` for the current
model unless a future model output includes recognized count-logit keys such as
`count_logits` or `pred_count_logits`. If no count head is present,
`--count-guided-topk` records `supported=false` and leaves the decoded lanes on
the normal path rather than fabricating count guidance.

Why:

Reporting-only official-test evidence shows a severe `GT4` count-shape
collapse:

```text
official-test count_acc_4 = 0.538462
GT4 confusion: 4->3=118, 4->4=252, 4->5=93, 4->6=4
```

The comparable official-val row looks much healthier:

```text
official-val count_acc_4 = 0.909091
GT4 confusion: 4->3=2, 4->4=60, 4->5=4
```

The failure is bidirectional on `GT4`, so a single global threshold change is
unlikely to solve it: raising thresholds helps `4->5` but risks more `4->3`,
while lowering thresholds helps `4->3` but risks more `4->5`.

Next action:

Run normal decode, oracle-count topK, and count-head guided topK first on
official-val with the fixed candidate pool. If oracle-count topK materially
reduces both `4->3` and `4->5`, count decision/ranking is the likely bottleneck
and count-guided decode can be considered only if official-val does not
regress. If oracle-count topK does not improve `GT4`, stop count-guided decode
and move to GT4 candidate quality/ranking changes such as weak-positive or
extra-ranking losses.

Completed diagnostic result:

Official-val ran with the fixed candidate pool:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_count_oracle_official_val/summary.json

normal:
  ACC=0.969665, FP=0.020615, FN=0.014004, count_acc=0.961433
  count_acc_4=0.909091
  GT4 4->3=2, 4->4=60, 4->5=4, 4->6=0

oracle_count_topk:
  ACC=0.969550, FP=0.013958, FN=0.014004, count_acc=0.991736
  count_acc_4=0.969697
  GT4 4->3=2, 4->4=64, 4->5=0, 4->6=0
```

Reporting-only official-test ran with the same fixed candidate pool:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_count_oracle_official_test_reporting_only/summary.json

normal:
  ACC=0.965254, FP=0.030649, FN=0.026959, count_acc=0.875629
  count_acc_4=0.542735
  GT4 4->3=116, 4->4=254, 4->5=94, 4->6=4

oracle_count_topk:
  ACC=0.964709, FP=0.020681, FN=0.028726, count_acc=0.932423
  count_acc_4=0.752137
  GT4 4->3=116, 4->4=352, 4->5=0, 4->6=0
```

Interpretation:

Oracle-count topK fixes the GT4 overcount side (`4->5` and `4->6`) but leaves
the GT4 undercount side unchanged on official-test (`4->3=116`). It improves
test `count_acc_4`, but not enough to call the collapse a pure count-decision
problem. The missing fourth-lane cases still need better candidate quality,
score ranking, or retention before final keep-count can solve them.

The count-head diagnostic reports `supported=false` for the active model, so
count-guided topK currently falls back to normal decode and should not be the
next candidate. A future count-guided candidate requires a real count head and
official-val evidence that it improves GT4 without a material ACC/FN regression.

Validation evidence:

Local implementation checks:

```text
D:\miniconda3\envs\lsa_yolo\python.exe -m py_compile tools/eval_tusimple_official.py
D:\miniconda3\envs\lsa_yolo\python.exe tools/eval_tusimple_official.py --help
helper-level synthetic checks for count_confusion, GT4 focus, and count-head stats helpers
```

Mainline or experiment:

Diagnostic/evaluation tooling only. It does not promote a checkpoint, change
training, alter the default decode path, or reopen final-test threshold tuning.

## 2026-06-24: Add GT4 Missing-Lane Raw-Query Diagnostic

Decision:

Stop the count-guided topK direction for the current `GT4 4->3` bottleneck and
add a raw-query diagnostic focused on whether the missing fourth GT lane exists
in the `Q=12` candidate set before normal decode filters.

Implementation:

```text
tools/diagnose_gt4_missing_lane_raw_queries.py
```

The script accepts:

```text
--weights
--split val|test
--archive-root
--gt-json
--imgsz
--conf
--point-valid-thr
--nms-dist-px
--max-det
--min-points
--save-dir
--only-count-pair 4->3
--max-images
```

It writes:

```text
summary.json
per_missing_lane.csv
per_image_summary.csv
raw_queries.csv
vis/
```

The diagnostic stages are:

```text
stage0_raw_all_queries: all Q=12 queries, no conf/point-valid/min-points/NMS
stage1_after_point_valid: raw query with point_valid_thr mask only
stage2_after_min_points: point-valid plus min_points
stage3_after_conf: point-valid plus min_points plus conf
stage4_final_decode: normal decode output
```

Each missing GT lane records the best raw query by common visible official
`h_samples`, using overlap `>=3` and mean absolute x error `<=20px` as the
default match gate. The script records existence logit, score, valid-point
counts at `0.5/0.45/0.4/0.35/0.3`, stage survival flags, side/order, and
`drop_reason`.

Protocol:

This is diagnostic tooling only. It does not add a training loss, change model
structure, alter decoder defaults, change official metrics, or tune test
thresholds. The official-test run below is reporting-only.

Validation evidence:

Local checks:

```text
D:\miniconda3\envs\lsa_yolo\python.exe -m py_compile tools\diagnose_gt4_missing_lane_raw_queries.py
D:\miniconda3\envs\lsa_yolo\python.exe tools\diagnose_gt4_missing_lane_raw_queries.py --help
```

Remote smoke check on the first 50 test records:

```text
selected_images = 12
total_gt4_4to3_images = 12
total_missing_gt_lanes = 12
drop_reason = geometry_bad 9, low_point_valid 3
```

Full reporting-only official-test diagnostic:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_gt4_missing_raw_queries_official_test_reporting_only/summary.json
records = 2782
GT4 4->3 images = 116
missing GT lanes = 129
drop_reason = geometry_bad 95, low_point_valid 33, low_score 1
raw_match_recall = 34/129 = 0.263566
after_point_valid_recall = 1/129 = 0.007752
after_min_points_recall = 1/129 = 0.007752
after_conf_recall = 0/129 = 0.0
final_decode_recall = 0/129 = 0.0
visualizations_saved = 50
```

Official-val comparison on the 363-image official-val subset:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_gt4_missing_raw_queries_official_val/summary.json
records = 363
GT4 4->3 images = 2
missing GT lanes = 3
drop_reason = geometry_bad 2, low_point_valid 1
raw_match_recall = 1/3 = 0.333333
after_point_valid_recall = 0/3 = 0.0
after_min_points_recall = 0/3 = 0.0
after_conf_recall = 0/3 = 0.0
final_decode_recall = 0/3 = 0.0
```

Interpretation:

Oracle-count topK only fixes the GT4 overcount side (`4->5` and `4->6`). It
does not fix `4->3`, and the current model has no independent count head, so
count-guided topK falls back to normal decode and is not a candidate.

The raw-query diagnostic shows that the official-test `4->3` missing lanes are
primarily candidate-recall failures before final count selection. Most missing
lanes are `geometry_bad`; the second-largest group is `low_point_valid`.
Almost none reach the confidence/ranking stage, and `low_score` is only one
lane.

Next action:

Use official-val/train-val diagnostics for the next candidate. The mainline
should shift to `GT4` missing-lane candidate recall, with the first hypothesis
being GT4 lane-balanced point learning or GT4-focused sampling. If a narrowed
trace shows raw geometry exists but point-valid collapses, then consider a GT4
weak-lane valid recall loss. Do not continue by tuning count-guided decode,
GT3-extra hinge gains, final-test thresholds, `max_det`, `min_points`, or NMS.

Mainline or experiment:

Diagnostic/evaluation tooling only. It does not promote a checkpoint, add a
loss, change decode, alter official metrics, or reopen final-test tuning.

## 2026-06-24: Add GT4 Lane-Balanced Point Loss and GT4-Hard Validation Builder

Decision:

Stop count-guided topK and GT3 extra-survival hinge as the current mainline
direction for the `GT4 4->3` bottleneck. Add default-disabled GT4 candidate
recall tooling instead: a GT4-only weak-lane point reweighting loss, optional
GT4-focused train sampling, and an internal GT4-hard validation list builder.

Implementation:

```text
ultralytics/utils/gcs_loss.py
  gt4_lane_balanced_point_loss
  gcs_gt4_lane_balanced_point = 0.0 default-disabled gain
  gcs_gt4_lane_balanced_topk = 1
  gcs_gt4_lane_balanced_max_mult = 2.0

tools/train_gcs.py
ultralytics/cfg/default.yaml
ultralytics/cfg/__init__.py
ultralytics/models/yolo/gcs_lane/train.py
ultralytics/models/yolo/gcs_lane/val.py
  CLI/default/config typing, loss logging alignment, progress short name gt4_pt
  gcs_gt4_sample_gain = 1.0 default train-only sampler gain

tools/diagnose_gt4_missing_lane_raw_queries.py
  supports split=train and exposes run_diagnostic(args)

tools/build_gt4_hard_val_split.py
  builds an internal GT4-hard sample list from train labels
```

Why:

The raw-query diagnostic shows the official-test `GT4 4->3` missing-lane cases
are dominated by candidate recall failures before final count selection:

```text
missing GT lanes = 129
drop_reason = geometry_bad 95, low_point_valid 33, low_score 1
raw_match_recall = 34/129
after_point_valid_recall = 1/129
after_conf_recall = 0/129
```

This means score/ranking changes are not the main bottleneck. Oracle-count
topK fixes GT4 overcount but leaves the `4->3` side unchanged, and the active
model has no independent count logits for count-guided topK. The next
experiment should therefore target raw GT4 lane geometry first.

Default behavior:

All new training behavior is off by default:

```text
gcs_gt4_lane_balanced_point = 0.0
gcs_gt4_sample_gain = 1.0
```

The new loss only applies when the image GT lane count is exactly `4` and the
existing Hungarian matcher matched all four lanes. It reuses the current
matching result, selects the top point-loss matched lane, raises that lane's
point weight up to `gcs_gt4_lane_balanced_max_mult`, and normalizes the image's
matched-lane weights back to mean `1.0`. It does not alter existence targets,
query ranking, decoder behavior, official metrics, model structure, Q=12/K=56,
or `--imgsz 544 960`.

Protocol:

`tools/build_gt4_hard_val_split.py` creates diagnostic train-derived hard
sample lists and summaries only. Those hard-val samples must not be fed back
into training. Candidate selection remains official-val first; official-test
may only be used once as reporting-only evidence for an already selected
candidate.

Recommended first experiments:

```text
gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025
  --gcs-duplicate-margin 0.05
  --gcs-gt4-lane-balanced-point 0.25
  --gcs-gt4-lane-balanced-topk 1
  --gcs-gt4-lane-balanced-max-mult 2.0

gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt050
  --gcs-duplicate-margin 0.05
  --gcs-gt4-lane-balanced-point 0.50
  --gcs-gt4-lane-balanced-topk 1
  --gcs-gt4-lane-balanced-max-mult 2.0
```

Optional follow-up:

```text
--gcs-gt4-sample-gain 1.5
```

Use this only after comparing official-val and GT4-hard diagnostics from the
loss-only variants.

Mainline or experiment:

Branch-local default-disabled experiment and diagnostic tooling. No candidate
is promoted by this implementation.

## 2026-06-25: Select gt4pt025 on Official-Val

Decision:

Select `gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025` as the
official-val candidate at this point in the experiment chain. Reject
`gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt050` because the larger
GT4 lane-balanced point gain degrades the primary metric and count shape.

Evidence:

```text
selected run = gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025_official_val_sweep/tusimple_official_sweep_summary.json
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=5, min_points=5
official-val ACC = 0.970975
official-val FP = 0.017264
official-val FN = 0.012167
official-val official_score = 0.970386
official-val count_acc = 0.966942
official-val count_acc_3/4/5 = 0.968610 / 0.954545 / 0.972973
```

The selected run narrowly beats the previous `gt4short15` official-val ACC
(`0.970975` vs `0.970851`) and materially improves GT4 count accuracy
(`0.954545` vs `0.848485`). It also beats `dupmargin005` on official-val ACC,
official score, FP/FN balance, and GT4 count accuracy.

Fixed candidate-pool support:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025_official_val_fixed_pool/summary.json
fixed pool decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
official-val ACC = 0.970363
official-val official_score = 0.969751
official-val count_acc = 0.961433
official-val count_acc_4 = 0.954545
```

Rejected run:

```text
run = gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt050
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt050_official_val_sweep/tusimple_official_sweep_summary.json
best decode = conf=0.1, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=6
official-val ACC = 0.964381
official-val FP = 0.034389
official-val FN = 0.021120
official-val count_acc = 0.903581
official-val count_acc_4 = 0.803030
```

Risk:

The margin over `gt4short15` is small and the selected `conf=0.005` is a score
calibration warning. The train-derived GT4-hard diagnostic for `gt4pt025`
still has `final_decode_recall = 0/22`; missing lanes are dominated by
`geometry_bad` and `low_point_valid`, so the GT4-hard candidate-recall
bottleneck is not solved.

Protocol:

Before the later `v2_validbranch_neg05-3` result, the next allowed action was a
one-shot reporting-only official-test report for `gt4pt025` using the frozen
selected decode:

```text
conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=5, min_points=5
```

Do not use that final-test result for threshold, NMS, `max_det`, `min_points`,
checkpoint, or loss-gain selection. After `v2_validbranch_neg05-3`, the
immediate next action is the v2 extra-lane diagnostic and fine official-val
sweep, not a `gt4pt025` final-test run.

## 2026-06-25: Reject gt4shortrecall_lbpt_endpoint

Decision:

Reject `gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint` as a
promotion candidate. Keep
`gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025` as the then-current
official-val selected candidate.

Evidence:

```text
run = gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint_official_val_sweep/tusimple_official_sweep_summary.json
best decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=5, min_points=4
official-val ACC = 0.970301
official-val FP = 0.024288
official-val FN = 0.013085
official-val official_score = 0.969554
official-val count_acc = 0.950413
official-val count_acc_3/4/5 = 0.959641 / 0.878788 / 0.986486
count_confusion = 3->3=214, 3->4=8, 3->5=1, 4->4=58, 4->5=8, 5->4=1, 5->5=73
```

Comparison:

The then-current `gt4pt025` gate remained stronger:

```text
gt4pt025 ACC = 0.970975
gt4pt025 count_acc_4 = 0.954545
gt4pt025 GT4 confusion includes 4->3=1, 4->4=63, 4->5=2
```

`gt4shortrecall_lbpt_endpoint` removes `4->3` from the selected official-val
decode, but worsens GT4 overcount to `4->5=8`, raises FP, and lowers ACC by
`0.000674` versus the gate.

Diagnostic evidence:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint_gt4_missing_raw_queries_official_val/summary.json
selected_images = 1
total_gt4_4to3_images = 1
total_missing_gt_lanes = 1
drop_reason = geometry_bad 1
raw_match_recall = 0/1 = 0.0
after_point_valid/min_points/conf/final_decode = 0/1
```

The missing lane is
`clips/0601/1494453641541664519/20.jpg`, `gt_lane_id=3`,
`left_inner`, with `12` visible points. The best raw query has overlap `12`,
score `0.28886989`, but mean absolute x error `30.265432px`, so it fails the
strict `20px` geometry gate.

Why:

The experiment is valid official-val evidence, but it does not beat the
current official-val candidate and does not prove that GT4 missing-lane raw
geometry is fixed. The remaining diagnostic failure is geometry quality, not
score thresholding.

Protocol:

Do not run final test for this rejected run. Do not tune thresholds, NMS,
`max_det`, `min_points`, checkpoint choice, or loss gains from final test.
Do not continue by adding score/count/rank losses unless a separate diagnostic
shows raw geometry and point-valid already survive.

Recommended next action:

Keep `gt4pt025` as the selected candidate. If continuing GT4 candidate recall
work, first inspect matcher, label, and query geometry around the remaining
official-val `0601` left-inner short lane and compare against `gt4pt025`.

## 2026-06-25: v2_validbranch_neg05-3 Extra-Lane Decision

Decision:

Classify `v2_validbranch_neg05-3` as valid official-val evidence that exceeds
the previous `gt4pt025` gate, but do not treat it as a clean robustness
promotion yet. Freeze the current weights and move to extra-lane diagnostics
plus a fine decoder sweep on official-val. Keep final test closed.

Strict ACC-best official-val row:

```text
run = v2_validbranch_neg05-3
decode = conf=0.01, point_valid_thr=0.45, nms_dist_px=30, max_det=5, min_points=2/3
official-val ACC = 0.971029
previous gate = 0.970975
margin = +0.000054
```

Risk-reduced near-tie row:

```text
decode = conf=0.015, point_valid_thr=0.45, nms_dist_px=60, max_det=5, min_points=2/3
official-val ACC = 0.971016
official-val FP = 0.022498
official-val FN = 0.012167
official-val official_score = 0.970323
official-val count_acc_4 = 0.878788
count_confusion extras = 3->4=8, 3->5=1, 4->5=7
```

The near-tie row is only `0.000013` below strict ACC best, with a better
secondary risk profile in the supplied diagnostics. Do not describe it as the
ACC-best row; it is the safer analysis contender.

Supported bottleneck update:

```text
GT4: 4->3 = 1, 4->5 = 11 on the strict best family
GT3: 3->4 = 9, 3->5 = 3 on the strict best family
pred_lanes_hist has too many 5-lane predictions
```

This means the current residual error is extra-lane / over-count dominated,
not GT4 `4->3` under-count dominated. Wider NMS is informative: the supplied
extra-lane diagnostics report `duplicate_like/spurious=13/22` for `nms30`,
versus `4/22` for `nms50` and `4/18` for `conf=0.015,nms60`. That supports a
duplicate/near-duplicate component, while leaving true spurious extras as an
open risk.

Rejected next actions:

- Do not continue increasing valid recall.
- Do not enable a valid count floor.
- Do not lower `point_valid_thr` below `0.45`.
- Do not increase `max_det` above `5`.
- Do not add Count/Quality/Survival/near-miss machinery.
- Do not tune any threshold, NMS, checkpoint, or loss weight from final test.

Smallest safe next action:

Use the fixed current weights on the same 363-image official-val surface.
First save image lists, visualizations, and per-extra-lane rows for
`GT3->4`, `GT3->5`, `GT4->5`, and `GT5->4`. Each extra prediction row must
include score, valid point count, nearest-GT distance, nearest-pred distance,
and a duplicate-like versus spurious label. Then run the fine decoder sweep:

```text
conf: 0.008, 0.010, 0.012, 0.015, 0.020
point_valid_thr: 0.43, 0.45, 0.47, 0.50
nms_dist_px: 30, 36, 42, 50, 60
max_det: 5
min_points: 2, 3
```

Report `official_acc`, `official_score`, `FP`, `FN`, `count_acc_4`, `3->4`,
`3->5`, and `4->5`. Use official ACC as the primary selector; when rows are
within about `1e-5` to `2e-5`, prefer the row with higher official score,
lower FP, better `count_acc_4`, and fewer `3->5` / `4->5` over-counts.

Uncertainty:

The ACC margin is tiny. `point_valid_thr=0.45` is already lower than
`gt4pt025`, so additional valid recall pressure is likely to amplify extras.
The top `conf=0.01,nms30` row does not yet have the same full extra-lane
diagnostic detail as the supplied `conf=0.015,nms30/nms60` rows. GT5 `5->4`
must be watched when choosing wider NMS.
