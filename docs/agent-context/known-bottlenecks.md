# Known Bottlenecks

This file applies to branch `codex/5-25-3-k56`.

## Branch Scope

The current mainline imports the historical `5-25-3.zip` algorithm and changes the TuSimple fixed-y contract to Q=12/K=56 with official h-sample anchors. It also includes the 2026-06-27 user-requested default-off `count_boundary_loss` for GT3/GT4/GT5 adjacent count-score boundaries, default-off train-only `gcs_hard_sampling` for 0601 and short-visible GT3/GT4/GT5 samples, default-off E3-lite `gcs_spurious_neg` loss for short unmatched duplicate-like queries, training-time `official_best`, and the default-off `valid_before_maxdet` query decode option present at `424ab1c86`. Follow-up code before that boundary keeps the old default behavior while adding default-preserving GT-count spurious weights, default-off GT spurious candidate protection, and default-effectively-off GT5 short point-valid rescue controls.

Do not read mainline Count Head, Quality Head, Survival Head, near-miss, or old mainline official-best bottlenecks as active branch behavior. Those algorithm mechanisms are not part of this 5-25-3 branch. The only active Count Boundary behavior is the branch-local default-off `count_boundary_loss`, the only active hard sampler is the branch-local default-off train-only `gcs_hard_sampling`, the only active E3-lite spurious negative behavior is the branch-local default-off `gcs_spurious_neg` family described in `current-contracts.md`, and the only active official-best behavior is the explicit 2026-06-27 training-time official-val selection hook.

Active source/config is rolled back to commit `424ab1c86` (`Add
valid-before-maxdet decode option`). Bottleneck notes below that depend on
post-`424ab1c86` mechanisms such as short-side hardset diagnostics,
`gcs_short_side_geom`, `gcs_far_spurious_neg`, `gcs_farspur_*`,
`gcs_shortside_*`, `gcs_rank_*`, count-contract diagnostics,
Q18/Q20/dataref configs, Count Head, count-guided decode, side-aux,
GT4-hard diagnostics, `tools/diagnose_tusimple_count_confusion.py`,
`tools/diagnose_gcs_count_contract.py`, `--gcs-gt4-short-*`,
`extra_exist_loss`, or `--gcs-short-exist-*` are legacy experiment conclusions
only. They do not describe currently available code, CLI flags, loss terms,
diagnostic scripts, configs, model outputs, or active selected candidates.

## Data And Geometry

- The active fixed-y anchors must be `710, 700, 690, ..., 160` normalized by original height `720`.
- Do not mix old `fixed_y=[0.98,0.25]` or K32 labels with this branch.
- Do not resample historical K32 labels into K56. Regenerate K56 labels from original TuSimple JSON and images.
- Preserve `--imgsz 544 960` in H,W order.
- The converted fixed-y dataset alone is not the full original TuSimple archive; official TuSimple evaluation still needs original raw-file image resolution/path context.

## Validation

- Local validation can check parser defaults, YAML contracts, fixed-y anchors, model output shape, and sample labels.
- Formal training and official-val evaluation should run on the remote CUDA server.
- This branch includes `tools/eval_tusimple_official.py`, `tools/sweep_tusimple_official_cached.py`, and `tools/sweep_tusimple_official.py` for official-val and final TuSimple test evaluation. Current threshold sweeps should use the cached helper.
- The active rollback code does not include `tools/diagnose_tusimple_count_confusion.py`.
- It includes explicit training-time `official_best` checkpoint preservation for official-val selection.
- It does not include post-`424ab1c86` short-side hardset/count-contract diagnostics, later mainline `diagnose_gcs_gt5.py`, Count/Quality/Boundary diagnostics, Survival, or near-miss machinery.

## 2026-07-23 Env30 Staticref / Valid-Neg / q7-Only Rejection Records

The recent env30 follow-up family is closed as rejected evidence. These runs
and diagnostics are not promotable, should not drive TEST tuning, and should
not be relaunched as the next path without a new official-val/train gate.

### GT4/GT5 Staticref v2

`query_alpha05_env30_gt45staticref_v2` is rejected after full protocol.

```text
env30 official-val ACC/FP/FN =
  0.973330 / 0.015748 / 0.009642
v2 official-val ACC/FP/FN =
  0.971040 / 0.018825 / 0.012626

env30 reporting-only TEST ACC/FP/FN =
  0.966780 / 0.028732 / 0.023544
v2 reporting-only TEST ACC/FP/FN =
  0.966541 / 0.029601 / 0.022795
```

The v2 bank fixed the older severe six-lane extra-carrier failure from
`gt5staticref_v1`, but it did not improve the real target. It traded partial
TEST GT4 retention for GT3 over-count and GT5 under-count:

```text
TEST GT4 4->3: 98 -> 81
TEST GT3 3->4: 50 -> 64
TEST GT5 5->4: 43 -> 69
```

Raw-Q12 confirms there is no net geometry gain:

```text
official-val overall has_match_20px:
  env30 0.975441 -> v2 0.972371
train0601 GT4 short has_match_20px:
  env30 0.473684 -> v2 0.578947
train0601 GT5 short has_match_20px:
  env30 0.775956 -> v2 0.737705
```

Decision: reject v2. Do not select thresholds, weights, or TEST behavior from
this run.

### GT4-Only Valid-Neg w01

`query_alpha05_env30_gt45staticref_v2_gt4only_validneg_w01` is rejected.

```text
w01 official-val ACC/FP/FN =
  0.970965 / 0.024656 / 0.012626

w01 official_best reporting-only TEST ACC/FP/FN =
  0.966061 / 0.037233 / 0.024712
w01 best.pt reporting-only TEST ACC/FP/FN =
  0.965752 / 0.041175 / 0.024053
```

Raw-Q12 shows partial recovery but not a solution:

```text
official-val overall has_match_20px:
  env30 0.975441 -> w01 0.976209
train0601 GT4 short has_match_20px:
  env30 0.473684 -> w01 0.736842
train0601 GT5 short has_match_20px:
  env30 0.775956 -> w01 0.759563
train0601 GT5 short point_valid_recall@0.6:
  env30 0.937601 -> w01 0.935816
```

The count-shape tradeoff remains unacceptable:

```text
train0601 4->5:
  env30 7, v2 4, w01 11
w01 TEST count_confusion includes:
  4->5 = 75, 4->6 = 15, 5->4 = 50, 5->6 = 99
```

Decision: reject w01 and do not launch the `w00` no-valid-negative companion
as the immediate next full train.

### v3 Static-Bank Diagnostic

The no-training v3 bank diagnostic comparing env30, v2, and w01 is diagnostic
only and does not justify a full v3 train.

```text
train0601 GT4 short hit20:
  env30 9/19, v2 11/19, w01 14/19
train0601 GT5 short hit20:
  env30 142/183, v2 135/183, w01 139/183

train0601 GT4 short q5/q6/q7 best-query share:
  env30 2/19, v2 11/19, w01 10/19
train0601 GT5 short q5/q6/q7 best-query share:
  env30 3/183, v2 56/183, w01 58/183
```

The useful GT4 gains and harmful GT5 losses both concentrate around q7 and
center visible-count 9-10 lanes. This means the failure is not just
valid-negative strength; the q5/q6/q7 carrier assignment itself is unstable.

Decision: do not train v3 from the current v2 bank.

### q7-Only Staticref v3a

The q7-only static-reference counterfactual is rejected before training. TEST
was not used.

```text
official-val has_match_20px:
  env30 0.975441 -> q7only 0.975441
train0601 has_match_20px:
  env30 0.960828 -> q7only 0.960828
train0531 has_match_20px:
  env30 0.983900 -> q7only 0.983900

train0601 GT4 short hit20:
  env30 9/19 -> q7only 9/19
train0601 GT5 short hit20:
  env30 142/183 -> q7only 142/183

GT4_short_gain20_q7only = 0
GT5_short_lost20_q7only = 1
```

Decision: reject q7-only v3a. It controls carrier explosion but creates no new
GT4-short raw coverage, so it should not proceed to full training or TEST.

Integrated bottleneck:

The rejected family shows that env30's remaining gap is not a single decode
knob, count head, static reference prior, or broad positive/negative loss.
The bottleneck is coupled: short GT4/GT5 raw geometry coverage and true/extra
query separation move against each other. A future candidate must first prove
on official-val plus train0601/train0531 raw-Q12 that it improves GT4-short
coverage without reducing GT5-short hit20/point-valid and without moving
q0/q1/q3/q10/q11 true lanes into q5/q6/q7 extra carriers.

### Q12 Joint Static Gate Follow-Up

The stricter Q12 joint-static gate is also closed as rejected diagnostic
evidence. `protected_gt5` passes the GT5 static gate, raising val GT5 primary
match20 to `0.5625`, train0601 GT5 primary match20 to `0.2786885`, and val
GT5 impact match20 to `1.0`, but GT4 remains `0.0/0.0/0.0` on
val/train0601/train0531. The `require_gt4` variant still has
`strict_joint_pass_count=0` and `loose_joint_pass_count=0`; its maximum GT4
match20 is only `0.125` on val, `0.066667` on train0601, and `0.285714` on
train0531. The five-slot rebalanced search evaluated `6226` designs and found
`gate_pass=0`.

Interpretation: the current Q12 protected replacement space does not have
enough separated carriers for both GT5 short/weak-visible raw geometry and
GT4 hard short/weak-visible geometry. Do not train from this Q12 route. The
next geometry-capacity experiment should be a default-off Q16 or Q20
protected dual-bank static gate that preserves `q0..q11` behavior and adds
separate GT5 and GT4 hard-lane query banks before any formal training or
reporting-only TEST.

### Q20 Protected Dual-Bank Static Gate Follow-Up

The default-off Q20 protected dual-bank tooling was implemented and run as a
static gate only. TEST was not used. The search compared `5:3`, `4:4`, and
`6:2` GT5:GT4 allocations over the extra queries `q12..q19`.

```text
artifact = .tmp/q20_dualbank_static_gate_alloc/summary.json
strict_pass = false
loose_pass = true
selected_allocation = 6:2
strict_pass_rows = 0
loose_pass_rows = 2415
```

Evidence:

```text
Q12 base val GT5 primary match20 = 0.0625
Q20 6:2 val GT5 primary match20 = 0.500000
Q20 6:2 train0601 GT5 primary match20 = 0.377049
Q20 6:2 val GT5 impact match20 = 0.750000

Q12 base train0601 GT4 primary match20 = 0.000000
Q20 6:2 train0601 GT4 primary match20 = 0.133333
strict gate requires train0601 GT4 primary match20 >= 0.200000

Q20 6:2 val GT5 primary p90 = 86.680450
strict gate requires val GT5 primary p90 <= 85.000000
```

Interpretation:

Q20 changes the GT5 short/weak-visible lane as intended and carries no added
normal GT3/GT4 match20 risk in the selected row, but it still cannot satisfy
GT5 and GT4 strict geometry coverage at the same time. The old `4:4` fixed
search improved GT4 more but failed `val_gt5_primary_match20 >= 0.50`; the new
`6:2` allocation fixes that GT5 match20 threshold but leaves train0601 GT4
below the strict gate. This is a remaining geometry-capacity and prototype
diversity bottleneck, not a decode or TEST-threshold problem.

Decision:

Do not publish the Q20 failed bank as a training candidate and do not run a
Q20 training probe from this result. The next geometry-capacity action should
increase separated reference capacity or prototype diversity before training,
for example a default-off Q24 protected dual-bank/static gate or a Q20
composite/clustered GT4 bank that proves train0601 GT4 reaches the strict gate
without losing the Q20 `6:2` GT5 gains.

### Q24 Protected Static Gate Follow-Up

The default-off Q24 protected static tooling was implemented and validated as
the direct follow-up to Q20. TEST was not used. The gate protects `q0..q11`,
adds `q12..q23`, and requires GT5 coverage, GT4 coverage, and zero added
normal GT3/GT4 match20 risk across val/train0601/train0531.

```text
artifact = .tmp/q24_protected_static_gate_server/summary.json
bank = data/gcs_reference_banks/q24_protected_static_env30_best.json
strict_pass = true
loose_pass = true
selected_allocation = 6:6
```

Evidence:

```text
Q12 base val GT5 primary match20 = 0.062500
Q24 val GT5 primary match20 = 0.500000
Q24 train0601 GT5 primary match20 = 0.393443
Q24 val GT5 impact match20 = 0.750000

Q12 base train0601 GT4 primary match20 = 0.000000
Q24 train0601 GT4 primary match20 = 0.266667
Q24 val/train0531 GT4 primary match20 = 0.625000 / 0.714286

Q24 val/train0601/train0531 normal GT3/GT4 added match20 risk =
  0.0 / 0.0 / 0.0
```

Interpretation:

Q24 breaks the Q20 capacity/prototype-diversity bottleneck. The selected
`6:6` allocation keeps GT5 at the requested static gate level while raising
train0601 GT4 beyond the `0.20` threshold and avoiding added normal GT3/GT4
carrier risk. This is the first protected static bank in this line that
simultaneously satisfies the GT4 and GT5 raw-geometry gates.

Decision:

Q24 is allowed to enter a 20-40 epoch remote probe using the dedicated Q24 YAML
and the strict-passed bank. It is not a final candidate yet. The probe must use
training-time official-val selection and must keep TEST closed. The probe
should be judged by official-val ACC/FP/FN, GT4/GT5 count shape, Q24 raw
GT4/GT5 survival, and whether the extra queries become harmful normal-lane
carriers after training.

### Q24 Protected Static Probe40 Rejection

The Q24 remote probe
`query_alpha05_env30_q24_protected_static_probe40_v1` is complete and rejected
for full training. TEST was not used.

```text
training rows = 40
official_best source_epoch = 35
external sweep rows = 864
```

Primary official-val evidence:

```text
env30 official-val ACC/FP/FN =
  0.973330 / 0.015748 / 0.009642
Q24 probe official-val ACC/FP/FN =
  0.961733 / 0.068457 / 0.023186

env30 count_acc_3/4/5 =
  0.968610 / 0.969697 / 0.986486
Q24 probe count_acc_3/4/5 =
  0.825112 / 0.803030 / 0.986486
```

The official-val sweep has no rescue row:

```text
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows passing count_acc_4>=0.954545 and count_acc_5>=0.972973 = 0
```

The failure shape is high FP and GT3/GT4 overcount, not GT5 undercount:

```text
Q24 count_confusion:
  3->4 = 28
  3->5 = 11
  4->5 = 13
  5->4 = 1
  5->5 = 73

env30 count_confusion:
  3->4 = 7
  4->5 = 0
  5->4 = 1
  5->5 = 73
```

Raw diagnostics show the static-bank protection did not survive training:

```text
Q24 val raw overall has_match20 = 0.925556
env30 val raw overall has_match20 = 0.975441

Q24 val GT5 visible<=10 has_match20/30 = 0.264151 / 0.584906
env30 val GT5 visible<=10 has_match20/30 = 0.754717 / 0.811321

Q24 q12..q23 raw-best carrier drift:
  GT3 = 130/669 lanes, 118 hit20
  GT4 = 66/264 lanes, 56 hit20
  GT3/GT4 visible<=20 = 179/311 lanes
```

Final decoded extra-lane diagnostics on official-val show the extra lanes are
not mainly duplicate lanes removable by NMS:

```text
GT3->4 images = 28, duplicate = 0, spurious = 11, boundary_pseudo = 9
GT4->5 images = 13, duplicate = 0, spurious = 1, boundary_pseudo = 4
```

Decision:

Reject this Q24 probe as a full-training candidate. Do not extend it to
160/220 epochs and do not run TEST. The bottleneck is now extra-query role
containment and existence calibration after training, not raw static capacity
alone.

Undertraining audit:

Q24 was trained for only 40 epochs from `yolo11s-seg.pt`, so part of the
absolute ACC gap to env30's 220-epoch final checkpoint is expected. However,
the same-epoch raw-cache comparison shows the Q24 problem is not only lack of
epochs:

```text
env30 epoch040 raw val overall has_match20 = 0.935285
Q24  epoch040 raw val overall has_match20 = 0.935285

env30 epoch040 GT3GT4 visible<=20 has_match20 = 0.889251
Q24  epoch040 GT3GT4 visible<=20 has_match20 = 0.925081

env30 epoch040 GT5 visible<=10 has_match20 = 0.576923
Q24  epoch040 GT5 visible<=10 has_match20 = 0.307692

env30 epoch040 GT5 visible<=20 has_match20 = 0.815534
Q24  epoch040 GT5 visible<=20 has_match20 = 0.733010
```

Q24's early overall raw coverage is not worse than env30 at the same age; it
is misallocated. It improves GT3/GT4 short-visible raw matching while leaving
GT5 short-visible coverage far weaker. The extra-query carrier drift also
persists through training:

```text
Q24 val GT3GT4 visible<=20 q12..q23 raw-best rate:
  epoch005 0.596091
  epoch010 0.589577
  epoch020 0.596091
  epoch035 0.583062
  epoch040 0.586319
```

The official count shape mirrors the same issue. GT5 undercount is repaired
early, but GT3/GT4 overcount remains: `4->5` is `13` at epoch035 and rebounds
to `19` at epoch040.

Interpretation:

Undertraining is a contributor to the absolute ACC gap, but the Q24-specific
bottleneck is extra-query role containment and GT5 short-geometry allocation.
Full training as-is is not a high-confidence next step unless an intermediate
continuation probe shows this drift naturally reverses.

Smallest safe next action:

Run only an official-val-only continuation probe to 80 or 100 epochs if the
team wants to settle the undertraining hypothesis without code changes. Gate it
on FP, GT3/GT4 overcount, GT5 short raw coverage, and carrier drift. If that
gate fails, use a default-off Q24 role-containment probe that constrains
`q12..q23` to their intended GT5 and GT4 hard-lane roles. Do not try to rescue
the current artifact by changing only thresholds, NMS, `max_det`, or
`min_points`.

### Q24 Protected Static Cont100 Rejection

The 100-epoch continuation diagnostic
`query_alpha05_env30_q24_protected_static_cont100_v1` is complete and rejected
for full training. TEST was not used.

```text
official_best source_epoch = 100
official-val ACC/FP/FN = 0.964376 / 0.055096 / 0.016758
count_acc_3/4/5 = 0.860987 / 0.742424 / 0.986486
```

It improves over the 40-epoch probe but remains far below env30:

```text
env30 official-val ACC/FP/FN =
  0.973330 / 0.015748 / 0.009642
env30 count_acc_3/4/5 =
  0.968610 / 0.969697 / 0.986486
```

The official-val sweep has no rescue row:

```text
max ACC = 0.964376
min FP = 0.046511
max count_acc_4 = 0.803030
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows passing count_acc_4>=0.954545 and count_acc_5>=0.972973 = 0
```

The remaining failure is GT3/GT4 overcount:

```text
cont100 count_confusion:
  3->4 = 19
  3->5 = 12
  4->5 = 17
  5->4 = 1
  5->5 = 73
```

Raw geometry improves but remains below env30, especially on train0601 and GT5
short-visible lanes:

```text
val raw overall has_match20:
  env30 0.975441, cont100 0.948580
train0601 raw overall has_match20:
  env30 0.960828, cont100 0.893677

val GT5 visible<=10 has_match20:
  env30 0.754717, cont100 0.490566
train0601 GT5 visible<=10 has_match20:
  env30 0.775956, cont100 0.415301
```

Extra-query carrier drift persists:

```text
cont100 val GT3GT4 visible<=20 q12..q23 raw-best rate = 0.581994
cont100 train0601 GT3GT4 visible<=20 q12..q23 raw-best rate = 0.662100
cont100 train0531 GT3GT4 visible<=20 q12..q23 raw-best rate = 0.654867
```

Epoch-cache diagnostics show this is stable, not a transient:

```text
val GT3GT4 visible<=20 q12..q23 raw-best rate:
  epoch005 0.605863
  epoch020 0.589577
  epoch040 0.589577
  epoch060 0.589577
  epoch080 0.589577
  epoch090 0.589577
  epoch100 0.589577
```

Interpretation:

Longer training helps GT5 short raw coverage, so undertraining contributed to
the 40-epoch gap. But the 100-epoch run proves that Q24 does not naturally
recover into the env30 band: FP stays high, `count_acc_4` remains very low, and
extra queries keep acting as wrong-count carriers. The bottleneck is Q24
extra-query role containment and existence/valid calibration, not static
capacity or training length alone.

Smallest safe next action:

Reject Q24 full training as-is. The default-off Q24 role-containment probe has
now been implemented for this failure mode: it suppresses wrong-count
`q12..q23` positives on GT3/GT4 while protecting GT5 from extra containment
negative pressure. The next action is to run only the 20-40 epoch
official-val/train-side probe before any formal full training or TEST.

### Q24 Role-Containment Probe40 v2 Rejection

The role-containment diagnostic
`query_alpha05_env30_q24_role_containment_probe40_v2` is complete and rejected
for full training. TEST was not used.

```text
official_best source_epoch = 40
official-val ACC/FP/FN = 0.963152 / 0.051607 / 0.021350
count_acc_3/4/5 = 0.869955 / 0.803030 / 0.972973
count_confusion includes 3->4=24, 3->5=5, 4->5=11, 5->4=2
```

The 864-row official-val sweep has no rescue row:

```text
max ACC = 0.963152
min FP = 0.044949
min FN = 0.021350
max count_acc_4 = 0.818182
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows with count_acc_4>=0.90 and count_acc_5>=0.972973 = 0
```

This probe proves the containment mechanism is active, because the old
GT3/GT4 short-visible extra-query carrier rate drops sharply:

```text
role v2 val/train0601/train0531 GT3GT4 visible<=20 q12..q23 raw-best rate =
  0.057878 / 0.118721 / 0.082596
cont100 comparable band was about 0.58 / 0.66 / 0.65
```

But the cost is unacceptable raw geometry loss, especially GT5 short-visible:

```text
role v2 val/train0601/train0531 raw overall match20 =
  0.937836 / 0.864578 / 0.967800
cont100 val/train0601/train0531 raw overall match20 =
  0.948580 / 0.893677 / 0.964222
env30 val/train0601/train0531 raw overall match20 =
  0.975441 / 0.960828 / 0.983900

role v2 val/train0601 GT5 visible<=10 match20 =
  0.358491 / 0.262295
cont100 val/train0601 GT5 visible<=10 match20 =
  0.490566 / 0.415301
env30 val/train0601 GT5 visible<=10 match20 =
  0.754717 / 0.775956
```

Oracle-rank closes the decode-only rescue path:

```text
current selected decode ACC/FP/FN = 0.963152 / 0.051607 / 0.021350
GT-count oracle-rank ACC/FP/FN = 0.963287 / 0.021534 / 0.021350
permissive pool24 oracle-rank ACC/FP/FN = 0.960530 / 0.023324 / 0.022039
```

Final decoded extra-lane diagnostics also show the residual overcount is not
mostly duplicate lanes removable by NMS:

```text
GT3->4 images = 24, duplicate=4, spurious=9, boundary_pseudo=5, ambiguous=6
GT4->5 images = 11, duplicate=1, spurious=2, boundary_pseudo=6, ambiguous=2
```

Interpretation:

The current role-containment design moved the failure from broad extra-query
carrier drift to a harder raw-coverage failure. The hard partition is wrong for
the trained Q24 bank: useful true-GT5 short carriers such as `q21/q23` are in
the GT4 bank and are forbidden on GT5 images, while `q13/q16` from the GT5 bank
still appear in GT4 false-fifth outputs. The bottleneck is therefore the role
partition and exist/valid calibration under containment, not training length,
static capacity, threshold selection, NMS, count estimation, or ranking.

Smallest safe next action:

Do not continue this run and do not launch full training. If Q24 continues, run
only a new 20-40 epoch official-val/train-side probe with a corrected partition
or softer matcher: make the GT5 bank follow the actual GT5 short carriers
(`q13/q15/q21/q23` are the current evidence-backed core), narrow the GT4 bank,
keep strong GT3/GT4 containment, and keep GT5 images free of extra negative
pressure. The gate must require GT5 visible<=10 raw match20 recovery above the
role-v2/cont100 band, no rebound in `3->5`/`4->5`, lower FP, and
`count_acc_4 >= 0.90` before any full training.

### Q24 Role-Partition Probe40 v3 Rejection

The role-partition follow-up
`query_alpha05_env30_q24_role_partition_v3_probe40_v1` is complete and
rejected for full training. TEST was not used.

```text
official_best source_epoch = 40
role GT5 bank = q12,q13,q15,q16,q20,q21,q23
role GT4 bank = q14,q17,q18,q19,q22
official-val ACC/FP/FN = 0.962991 / 0.079844 / 0.023416
count_acc_3/4/5 = 0.878924 / 0.818182 / 0.094595
```

The 864-row official-val sweep has no promotable or decode-rescue row:

```text
max ACC = 0.962991
min FP = 0.048118
min FN = 0.023416
max count_acc_4 = 0.818182
max count_acc_5 = 1.0
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows with count_acc_4>=0.90 and count_acc_5>=0.972973 = 0
```

The selected official-val count shape is worse than role v2 and exposes a new
GT5 sixth-lane failure:

```text
pred_lanes_hist = 3:197, 4:77, 5:20, 6:69
count_confusion:
  3->3=196, 3->4=23, 3->5=4
  4->3=1, 4->4=54, 4->5=9, 4->6=2
  5->5=7, 5->6=67
```

Training-time official-val progression is not a promotion signal:

```text
epoch030 ACC/FP/FN = 0.958986 / 0.072727 / 0.025482, count_acc_5=1.0
epoch035 ACC/FP/FN = 0.960652 / 0.059183 / 0.027548, count_acc_5=1.0
epoch040 ACC/FP/FN = 0.962991 / 0.079844 / 0.023416, count_acc_5=0.094595
```

Raw diagnostics show the GT5 short geometry bottleneck worsened rather than
recovered:

```text
v3 val/train0601/train0531 raw overall match20 =
  0.937068 / 0.864018 / 0.974061
env30 val/train0601/train0531 raw overall match20 =
  0.975441 / 0.960828 / 0.983900

v3 val/train0601 GT5 visible<=10 match20 =
  0.188679 / 0.229508
role v2 val/train0601 GT5 visible<=10 match20 =
  0.358491 / 0.262295
cont100 val/train0601 GT5 visible<=10 match20 =
  0.490566 / 0.415301
```

The role partition did suppress the old broad GT3/GT4 extra-query raw-best
carrier drift, but this is not enough for official ACC:

```text
v3 val/train0601/train0531 GT3GT4 visible<=20 q12..q23 raw-best rate =
  0.093248 / 0.127854 / 0.079646
```

Final decoded GT5 overcount is not an NMS duplicate problem:

```text
GT5->6 images = 67
duplicate = 0
spurious = 4
boundary_pseudo = 36
ambiguous = 27
extra_best_gt_ape_mean = 85.590668
exist_score_mean = 0.191999
visible_length_mean = 9.044776

GT5->6 extra query histogram:
  q13=30, q21=12, q23=10, q11=7, q20=5, q1=2, q15=1
```

The same query IDs overlap with true GT5 short-lane raw carriers, so a manual
hard partition alone cannot solve the problem:

```text
true GT5 visible<=10 raw-best query histogram on val:
  q21=20, q13=10, q23=7, q15=6, q20=4

hit20 among those true GT5 visible<=10 raw-best lanes:
  q23=4, q13=2, q15=2, q20=1, q14=1
```

The event-mined per-query gate confirms there is no clean hard-partition query
that has enough true GT5 short value and low false-extra risk:

```text
artifact = .tmp/q24_role_partition_v3_review/event_mined_gate/per_query_event_gate.csv
TEST used = false

total val GT5->6 events = 67
total val GT3/GT4 false-extra events = 38
total val true GT5 visible<=10 raw-best events = 53
total train0601 true GT5 visible<=10 raw-best events = 183
no query with val GT5<=10 hit20 >= 4 and risk_events <= 2

high-risk queries:
  q13 gt5_bank: true hit20=2, near20-40=8, GT5->6=30, GT3/GT4 false-extra=1
  q23 gt5_bank: true hit20=4, near20-40=3, GT5->6=10, GT3/GT4 false-extra=3
  q21 gt5_bank: true hit20=0, near20-40=18, GT5->6=12, GT3/GT4 false-extra=1
  q22 gt4_bank: true hit20=0, near20-40=0, GT5->6=0, GT3/GT4 false-extra=11
  q0 protected: true hit20=0, near20-40=0, GT5->6=0, GT3/GT4 false-extra=11
```

Score calibration cannot separate true GT5 short lanes from fake GT5 sixth
lanes either:

```text
GT5->6 final extra exist_score mean/p50 = 0.191999 / 0.146697
true GT5 visible<=10 raw-best exist_score mean/p50 ~= 0.199561 / 0.157525
boundary_pseudo GT5->6 extras below score 0.2 = 35/36
train boundary_pseudo_count last10 mean ~= 0.081
```

Oracle/count diagnostics close the decode-only rescue path:

```text
current selected decode ACC/FP/FN = 0.963007 / 0.079844 / 0.023416
GT-count oracle-rank ACC/FP/FN = 0.963118 / 0.023508 / 0.023416
best count-safe sweep row ACC/FP/FN = 0.962843 / 0.051745 / 0.023416
```

Interpretation:

V3 proves the previous idea "put the observed GT5 carriers into the GT5 bank"
is insufficient. The same carriers, especially `q13/q21/q23`, are also the
dominant GT5 sixth-lane boundary/ambiguous carriers, while their scores overlap
with true short GT5 lanes. The boundary-pseudo negative path was enabled but is
too sparse and score-gated above many final low-score decoded extras. The root
cause is therefore a coupled event-level separation failure: true GT5
short-geometry rescue and GT5 boundary-pseudo suppression are not separable by
static hard query partition, simple confidence thresholds, NMS, max-det, count
oracle, or rank oracle.

Smallest safe next action:

Stop Q24 hard role-partition as-is. Do not full-train and do not TEST. Before
another 20-40 epoch probe, run or implement an official-val/train-side
event-mined gate that reports, per query, true GT5 short hit20/near/miss counts
against GT5->6 boundary-pseudo/ambiguous counts and GT3/GT4 false-extra counts.
Only then try a GT5-safe boundary-pseudo suppression that protects true GT5
short candidates, paired with a true-short geometry rescue. The gate must prove
GT5 visible<=10 raw match20 recovers above role v2/cont100 without reintroducing
GT3/GT4 overcount before any full training.

Implementation status:

The default-off GT5-safe boundary-pseudo probe and event-mined gate tooling are
now implemented. Use `scripts/run_q24_gt5safe_boundary_probe_v1.sh` for the
40-epoch probe and `scripts/run_q24_event_mined_gate_v1.sh` after training.
Do not enable hard role-partition matcher constraints for this default probe.
The next bottleneck test is whether low-score GT5->6 boundary-pseudo pressure
can reduce overcount while preserving true GT5 short raw geometry; it is not a
full-training or TEST candidate until that official-val/train-side gate passes.

Completed result:

`query_alpha05_env30_q24_gt5safe_boundary_probe40_v1` is complete and rejected
for longer/full training. TEST was not used. The official-val result is:

```text
ACC/FP/FN = 0.962156 / 0.089302 / 0.021120
count_acc_3/4/5 = 0.852018 / 0.712121 / 0.175676
count_confusion includes 3->4=22, 3->5=11, 4->5=10, 4->6=8, 5->6=61
```

The external 864-row official-val sweep has no rescue row:

```text
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows passing longer-training gate = 0
max count_acc_4 = 0.757576
```

The mechanism was active but insufficient:

```text
epoch040 train boundary_pseudo_candidate/protected =
  2.19608 / 0.63726
epoch040 val boundary_pseudo_candidate/protected =
  3.66667 / 1.5
```

Raw and event diagnostics show the actual bottleneck:

```text
val/train0601 GT5 visible<=10 raw match20 =
  0.301887 / 0.316940
val/train0601/train0531 GT3GT4 visible<=20 q12..q23 raw-best rate =
  0.581994 / 0.657534 / 0.654867
event gate total_val_gt5_to6 = 60
event gate total_val_gt34_false_extra = 51
```

Per-query evidence isolates the problem:

```text
q13: risk=25, true GT5<=10 hit20/near=5/8, GT5->6=22
q21: risk=21, true GT5<=10 hit20/near=1/14, GT5->6=19
q23: risk=22, true GT5<=10 hit20/near=1/6, GT5->6=9, false-extra=13
q15: the only clean high-true low-risk query, hit20/near=5/2, risk=2
```

Interpretation: this is not a simple training-length issue or a no-op
implementation. The GT5-safe path protected some near-true candidates, but the
dominant carriers remain mixed true-short and false-extra events, and GT3/GT4
extra-query carrier drift remains close to the failed cont100 band. Stop Q24
GT5-safe boundary-pseudo as-is. If Q24 continues, the next probe must change
the mechanism toward clean-carrier isolation plus event-aware containment for
`q13/q21/q23/q22`, and must pass another 20-40 epoch official-val/train-side
gate before any longer run.

Implementation status:

The default-off Q24 event-containment probe is implemented through
`--gcs-q24-event-*` flags and
`scripts/run_q24_event_containment_probe_v1.sh`. The first probe should keep
clean GT5 protection on `q12/q15/q20`, stop protecting high-risk
`q13/q21/q22/q23` in the boundary-pseudo loss, enable event-aware Hungarian
matching, and suppress high-risk event queries on GT3/GT4/GT5 with low gain.
This is not a full-training or TEST candidate until the new official-val and
event-mined gates pass.

## 2026-07-23 Env30 GT4 Near-20px Geometry Refine v1 Rejection

The `query_alpha05_env30_gt4_near20_geom_refine_v1` full training run is
complete and rejected. It is not promotable. TEST was not used.

Run and code:

```text
run = query_alpha05_env30_gt4_near20_geom_refine_v1
git = c6af4b2 / c6af4b27c Add env30 GT4 near20 geometry refine
training rows = 186
official_best source_epoch = 170
```

The candidate fails the primary official-val gate:

```text
env30 official-val ACC/FP/FN =
  0.973330 / 0.015748 / 0.009642
near20 v1 official-val ACC/FP/FN =
  0.968943 / 0.035373 / 0.015152
```

The selected `official_best` decode also has unsafe count shape:

```text
decode =
  conf 0.001, point_valid_thr 0.6, nms_dist_px 18,
  max_det 8, min_points 2, valid_before_maxdet true

count_acc_3/4/5 =
  0.932735 / 0.878788 / 0.824324

count_confusion includes:
  3->4 = 14
  3->5 = 1
  4->5 = 7
  5->6 = 9
  5->7 = 3
```

The epoch170 official sweep has no rescue row:

```text
sweep rows = 864
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows passing GT4/GT5 count-shape gate = 0
```

Raw-Q12 diagnostics show the proposed geometry refine did not fix the target
and instead damaged broad candidate coverage:

```text
official-val overall has_match_20px:
  env30 0.975441 -> near20 0.963162
train0601 overall has_match_20px:
  env30 0.960828 -> near20 0.925574
train0531 overall has_match_20px:
  env30 0.983900 -> near20 0.975850

train0601 GT4 short hit20:
  env30 9/19 -> near20 8/19
train0601 GT5 short hit20:
  env30 142/183 -> near20 108/183
```

The intended near-miss repair was too sparse to control training:

```text
train/gt4_near20_refine_count last20 mean = 0.104903 lanes/batch
```

Event-level conclusion from the env30 GT4-short miss set:

```text
env30 train0601 GT4-short miss lanes fixed = 2/10
env30 train0601 GT4-short hit lanes lost = 3
```

Decision:

Reject the near20 v1 loss route in its current form. It does not improve the
official target, does not improve the train0601 GT4-short raw gate, and causes
a much larger GT5-short raw coverage regression. Do not run TEST, do not tune
thresholds from this checkpoint, and do not continue by simply raising
`gcs_gt4_near20_geom_refine` or widening the APE window.

Smallest safe next action:

Do not launch another full near20 training run directly. If this direction is
reopened, first run a no-training activation/risk scan on env30 over narrow
query sets such as `0,10`, `0,8,10`, `0,1,8,10`, and `0,6,8,10`, with APE
windows `20-25` and `20-30`. A later training attempt needs evidence that it
can cover at least `5/10` train0601 GT4-short env30 misses, lose at most one
existing env30 GT4-short hit, and avoid any GT5-short risk increase before
it is allowed to train.

## 2026-07-11 Boundary Pseudo-Negative Mask Bottleneck

The completed B1 run
`query_alpha05_gt5short_geom_w2_bneg005_nocount_v1` is diagnostic-only and
must not be promoted.

```text
G1 official_best val:
ACC/FP/FN = 0.973071 / 0.014784 / 0.009642
GT4 4->3/4->5 = 1 / 0
GT5 5->4/5->5/5->6 = 1 / 69 / 4
GT5 visible<=10 raw has_match20/30 = 0.886792 / 0.924528
GT5 visible<=10 raw p90 APE = 19.372219 px

B1 official_best val:
ACC/FP/FN = 0.972100 / 0.012259 / 0.008724
GT4 4->3/4->5 = 1 / 1
GT5 5->4/5->5/5->6 = 2 / 72 / 0
GT5 visible<=10 raw has_match20/30 = 0.849057 / 0.924528
GT5 visible<=10 raw p90 APE = 25.399066 px

B1 train last20:
train/boundary_pseudo_count mean = 0.011274
val/boundary_pseudo_count mean = 0.500000
spurious_neg_loss/count = 0

B1-v2 protocol-miss result:
run = query_alpha05_gt5short_geom_w2_bneg005_nocount_v2
actual envelope margin = gcs_boundary_pseudo_envelope_margin_px -1.0
actual pseudo config = neg 0.05, dist_thr 60, min_valid 3, score_thr 0.0
official_best source_epoch = 180
official_best val ACC/FP/FN = 0.969401 / 0.024334 / 0.016070
official_best val GT4 4->3/4->5 = 1 / 5
official_best val GT5 5->4/5->5/5->6 = 2 / 72 / 0
max_det=6 equal-ACC rows show GT5 5->6 = 4..6, so selected 5->6=0 is capped
GT5 visible<=10 raw has_match20/30 = 0.754717 / 0.792453
GT5 visible<=10 raw p90 APE = 35.917850 px
best.pt val ACC/FP/FN = 0.966980 / 0.035537 / 0.017906
reporting-only official_best test ACC/FP/FN = 0.965566 / 0.036137 / 0.026869
reporting-only official_best test GT4 4->5 = 100
reporting-only official_best test GT5 5->4/5->5 = 54 / 497
train GT4->5 diagnostic over train label_data_0313/0531/0601 = 45 + 6 + 23 = 74

Mask-v2/envelope result:
run = query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1
actual pseudo config = neg 0.02, dist_thr 80, min_valid 4, score_thr 0.2,
  envelope_margin_px 30, envelope_ratio_thr 0.75
official_best source_epoch = 220
official_best val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
official_best val GT4 4->3/4->5 = 2 / 0
official_best val GT5 5->4/5->5/5->6 = 1 / 73 / 0
max_det=6 equal-ACC rows also keep GT5 5->6 = 0
GT5 visible<=10 raw has_match20/30 = 0.754717 / 0.811321
GT5 visible<=10 raw p90 APE = 36.550196 px
train GT4->5 diagnostic over train label_data_0313/0531/0601 = 29 + 2 + 7 = 38
reporting-only official_best test ACC/FP/FN = 0.966780 / 0.028732 / 0.023544
reporting-only official_best test GT4 4->5 = 90
reporting-only official_best test GT5 5->4/5->5 = 43 / 507
```

Supported bottleneck:

- The current pseudo-negative mask is too broad for true GT5 short/side
  candidates. It suppresses selected-row GT5 overcount but worsens raw short
  geometry and GT4/GT5 count shape.
- `query_alpha05_gt5short_geom_w2_bneg005_nocount_v2` is not evidence that
  the outside-envelope protection failed. The run did not enable the envelope
  protection at all (`gcs_boundary_pseudo_envelope_margin_px=-1.0`) and also
  kept the broader/harder B1-style settings (`0.05`, `60 px`, `min_valid=3`,
  `score_thr=0.0`) instead of the intended smaller mask-v2 settings.
- The v2 result fails every serious promotion gate versus G1: lower
  official-val ACC, worse FP/FN, worse GT4 `4->5`, worse GT5 `5->4`, no
  robust GT5 `5->6` gain once `max_det=6` is allowed, and much worse
  GT5-visible<=10 raw geometry.
- The true mask-v2/envelope run fixes the main selected official-val
  count-shape problem: it is slightly above G1 on ACC, keeps FN equal to G1,
  removes official-val GT4 `4->5`, and removes GT5 `5->6` even when
  `max_det=6` is allowed. It should be read as a real validation count-shape
  improvement over B1/v2.
- It still fails the short-geometry and train-count diagnostic gates. GT5
  visible<=10 raw p90 APE is worse than both G1 and B1, and train GT4->5 is
  still higher than the earlier 27->31 warning band. Therefore the result is
  not a clean final promotion despite the higher official-val ACC.
- The mask-v2 loss is almost inactive late in training
  (`val/boundary_pseudo_count=0` and train last20 near zero), so the result
  does not prove that adding more pseudo-negative pressure will help. The next
  bottleneck is protecting/repairing true short-lane geometry, not simply
  increasing boundary pseudo-negative gain.
- `GT5 5->6=0` on B1 official-val is not sufficient promotion evidence
  because the selected decode uses `max_det=5`; the B1 `best.pt` validation
  row with `max_det=6` still shows `GT5 5->6=8`.
- The diagnostic should be read as "unmatched is not enough to define a pseudo
  lane." A target-zero existence term needs a clear-far or outside-GT-envelope
  condition, not just Hungarian-unmatched status.

Smallest safe next action:

Do not run `gcs_boundary_pseudo_neg=0.1`. If continuing the line, shrink the
negative mask before increasing pressure: try a validation-only B1-mask-v2
with a smaller gain such as `0.02`, larger distance-to-all-GT threshold such
as `80 px`, stricter `min_valid=4`, optional `score_thr=0.2`, and continued
matched-query exclusion. Use
`scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh` with
its default `RUN_TESTS=0`; only enable reporting-only test after official-val,
raw-Q12 short-GT5 geometry, and train GT4->5 diagnostics pass. In parallel or
as the next branch, design a narrower matched GT4/GT5 short-positive rescue so
true short lanes are protected before more unmatched negatives are applied.

## 2026-07-11 G1 Count-Aware Extra-Margin Diagnostic

The G1 GT5-short-geometry artifact
`query_alpha05_gt5short_geom_w2_v1` remains the selected/reference surface for
this line. Count-aware extra-margin is useful as default-off decode tooling,
but the G1 `score_sum` proxy sweep is diagnostic-only and does not replace the
no-count-aware official-best decode.

```text
G1 artifact:
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
gcs_query_count_ce = 0.0
pred_count_logits / query Count Head keys = absent

fixed official-val decode:
weights = runs/gcs_lane/query_alpha05_gt5short_geom_w2_v1/weights/official_best.pt
conf = 0.003
point_valid_thr = 0.5
nms_dist_px = 0
max_det = 6
min_points = 5
valid_before_maxdet = true
```

Official-val comparison:

```text
no count-aware:
ACC/FP/FN = 0.973071 / 0.014784 / 0.009642
GT5 5->4/5->5/5->6 = 1 / 69 / 4

score_sum k:
ACC/FP/FN = 0.972632 / 0.007668 / 0.011019
GT5 5->4/5->5/5->6 = 8 / 66 / 0

score_sum k+1:
ACC/FP/FN = 0.973069 / 0.014325 / 0.009642
GT5 5->4/5->5/5->6 = 1 / 70 / 3

score_sum k+2:
ACC/FP/FN = 0.973069 / 0.014784 / 0.009642
GT5 5->4/5->5/5->6 = 1 / 69 / 4
```

Supported bottleneck:

- Hard count-aware top-k is still risky even after G1 geometry improves.
  `score_sum k` removes sixth-lane overcount but creates GT5 undercount and
  raises FN.
- `score_sum k+1` has the only useful shape signal: one fewer GT5 `5->6`,
  one more GT5 `5->5`, slightly lower FP, and unchanged FN. The official ACC
  is still a near tie below the no-count-aware baseline, so it is not a
  promotable decode.
- `score_sum k+2` adds no meaningful change over no-count-aware.
- The result does not answer whether a learned query Count Head is useful
  after G1, because this checkpoint has no `pred_count_logits`.

Smallest safe next action:

Do not run final test or reselect G1 decode from this proxy. If count-aware is
revisited, use a G1-compatible query-count checkpoint and run official-val-only
`count_logits k/k+1/k+2`, with gates on GT5 `5->6`, official ACC, FN, and FP.

## 2026-07-09 Exist-Quality Alpha Ablation Bottleneck

The completed global existence-target ablations are diagnostic-only. They show
that hardening matched existence targets raises GT5 raw scores, but global
relaxation does not solve the selected official-val candidate problem.

```text
E1 = query_exist_quality_alpha05_v1
gcs_exist_quality_alpha = 0.5
official-val ACC/FP/FN = 0.969860 / 0.024197 / 0.012856
count_acc_4/5 = 0.924242 / 0.891892
GT5 confusion includes 5->4=1, 5->5=66, 5->6=7

E2 = query_exist_quality_alpha00_v1
gcs_exist_quality_alpha = 0.0
official-val ACC/FP/FN = 0.967784 / 0.020845 / 0.015152
count_acc_4/5 = 0.954545 / 0.972973
GT5 confusion includes 5->4=2, 5->5=72
```

Raw Q12 diagnostics on the canonical 363-image official-val split:

```text
GT5 visible<=10 raw-best score mean:
alpha05 = 0.631605
alpha00 = 0.896660

GT5 fifth-lane raw-best score mean:
alpha05 = 0.928582
alpha00 = 0.997159

query 10/11 vs GT5 fifth-lane score mean:
alpha05 = 0.492548
alpha00 = 0.559448
```

Supported bottleneck:

- `gcs_exist_quality_alpha=1.0`-style quality suppression can depress true GT5
  short/side existence scores.
- Completely disabling quality-aware targets is too blunt: it raises scores
  but worsens official-val ACC and FN compared with alpha `0.5`, and it does
  not produce the expected "GT5 recall up, FP up" pattern on validation.
- E1's wider keygrid ACC-best row (`0.970634`) is not target-aligned because it
  increases FP and GT5 `5->6` while dropping `count_acc_5` to `0.824324`.

Smallest safe next action:

Use a selective relaxation rather than a global alpha change: target GT5
short-side or weak-visible matched positives, keep full official-val selection,
and add an explicit overcount guard so lifted scores do not become extra sixth
lanes.

## 2026-07-07 Query Count Head CE0.5 Non-Promotion

The completed `query_count_head_ce05_v1` run is diagnostic-only, not a
promoted candidate.

Training-time official-best selected:

```text
weights = runs/gcs_lane/query_count_head_ce05_v1/weights/official_best.pt
decode = runs/gcs_lane/query_count_head_ce05_v1/weights/official_best_decode.yaml
source_epoch = 191
conf = 0.005
point_valid_thr = 0.45
nms_dist_px = 0.0
max_det = 5
min_points = 2
valid_before_maxdet = true
count_aware_topk = false
count_mode = score_sum
official-val ACC = 0.968473
official-val FP = 0.015152
official-val FN = 0.011938
official-val count_acc = 0.931129
official-val count_acc_4 = 0.893939
official-val count_acc_5 = 0.891892
```

The competing `best.pt` post-train official-val sweep is weaker:

```text
sweep = runs/gcs_lane/query_count_head_ce05_v1_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
weights = runs/gcs_lane/query_count_head_ce05_v1/weights/best.pt
best decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=0.0, max_det=6, min_points=5, valid_before_maxdet=true, count_aware_topk=false, count_mode=score_sum
official-val ACC = 0.967679
official-val FP = 0.032553
official-val FN = 0.015152
official-val count_acc = 0.906336
official-val count_acc_4 = 0.893939
official-val count_acc_5 = 0.770270
```

The explicit `count_logits` count-aware diagnostic improves lane-count
accuracy but not the target official ACC/FN tradeoff:

```text
val sweep = runs/gcs_lane/query_count_head_ce05_v1_best_val_sweep_valid_before_maxdet_catopk_count_logits/tusimple_official_sweep_summary.json
test summary = runs/gcs_lane/query_count_head_ce05_v1_best_official_test_from_val_catopk_count_logits/tusimple_official_summary.json
decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=3, valid_before_maxdet=true, count_aware_topk=true, count_mode=count_logits
official-val ACC/FP/FN = 0.967562 / 0.019330 / 0.015152
official-val count_acc = 0.975207
official-val count_acc_4/5 = 0.954545 / 0.959459
official-test ACC/FP/FN = 0.964362 / 0.026582 / 0.028696
official-test count_acc = 0.889288
official-test count_acc_4/5 = 0.591880 / 0.847100
```

Reporting-only official test confirms no promotion signal:

```text
official_best test = runs/gcs_lane/query_count_head_ce05_v1_official_test_official_best_decode/tusimple_official_summary.json
official_best test ACC = 0.964862
official_best test FP = 0.033914
official_best test FN = 0.024473
official_best test count_acc = 0.874191
official_best test count_acc_4 = 0.617521
official_best test count_acc_5 = 0.855888

best.pt sweep-threshold test = runs/gcs_lane/query_count_head_ce05_v1_best_official_test_from_val_sweep_valid_before_maxdet_b/tusimple_official_summary.json
best.pt sweep-threshold test ACC = 0.964518
best.pt sweep-threshold test FP = 0.037653
best.pt sweep-threshold test FN = 0.026779
best.pt sweep-threshold test count_acc = 0.836089
best.pt sweep-threshold test count_acc_4 = 0.566239
best.pt sweep-threshold test count_acc_5 = 0.667838
```

User-requested oracle-count diagnostic on the same selected
`official_best.pt` and `official_best_decode.yaml` test surface:

```text
oracle-count summary = runs/gcs_lane/query_count_head_ce05_v1_official_test_official_best_decode_oracle_count/tusimple_official_summary.json
oracle change = query count-aware top-k uses k_hat = GT lane count
decode = conf=0.005, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_test_acc = 0.963987
official_test_FP = 0.022172
official_test_FN = 0.027318
official_test_count_acc = 0.938893
official_test_count_acc_4 = 0.814103
official_test_count_acc_5 = 0.855888
count_confusion = 2->2=5, 3->2=1, 3->3=1739, 4->3=87, 4->4=381, 5->3=13, 5->4=69, 5->5=487
```

Supported interpretation:

- Use `official_best.pt` plus `official_best_decode.yaml` for any reporting of
  this run; do not replace it with generic `best.pt` or the post-train
  `best.pt` sweep.
- The explicit query Count Head CE0.5 training signal did not close the main
  test-side gap. Its reporting-only official-test ACC is below the historical
  reporting-only E1 count-boundary and spurious-lite valid-before results
  around `0.965428` to `0.965483`.
- The generic `best.pt` sweep has a clear GT5 retention risk
  (`count_acc_5=0.770270` on official-val and `0.667838` on reporting-only
  test), so it should not be used as a fallback.
- The count-logits diagnostic raises official-val `count_acc` to `0.975207`,
  but official-val ACC stays below the selected score-sum surface and
  reporting-only test ACC/FN are worse than the Count Head official-best
  score-sum test.
- The oracle-count test is diagnostic-only because it uses GT during decode.
  It proves that perfect test-time lane count alone does not improve this
  artifact's official-test ACC: count accuracy improves by `+0.064702`, but ACC
  drops by `0.000875` and FN rises by `0.002845`. The unchanged GT4/GT5
  undercount (`4->3=87`, `5->3=13`, `5->4=69`) points away from Count Head as
  the dominant remaining bottleneck and toward missing/low-ranked/filtered true
  4th/5th-lane candidates.
- Do not tune thresholds, checkpoint choice, count mode, count-aware top-k,
  `max_det`, `min_points`, NMS, or loss gains from either official-test
  summary.

## 2026-07-09 shortside025_farspur005_rank002 Non-Promotion

The completed experiment named
`gcs_yolo_lane_s_q12_k56_shortside025_farspur005_rank002_v1` is now readable
from the remote server and is diagnostic-only, not a promoted candidate.

```text
training-time official_best:
source_epoch = 35
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=50.0, max_det=6, min_points=4, valid_before_maxdet=false
official-val ACC/FP/FN = 0.968554 / 0.035078 / 0.018595
count_acc_4/5 = 0.818182 / 0.972973
count_confusion includes 4->5=11, 5->4=1, 5->6=1

external official_best val sweep:
decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=18.0, max_det=6, min_points=4, valid_before_maxdet=true
official-val ACC/FP/FN = 0.969366 / 0.034573 / 0.016988
count_acc_4/5 = 0.803030 / 0.986486
count_confusion includes 3->4=13, 3->5=2, 4->5=12, 5->6=1

reporting-only test:
official-test ACC/FP/FN = 0.964823 / 0.041643 / 0.025971
count_acc_4/5 = 0.549145 / 0.852373
count_confusion includes 4->5=122, 5->4=30, 5->6=42
```

Supported bottleneck:

- The run keeps validation GT5 retention relatively high, but this comes with
  poor GT4 count shape and high FP.
- The reporting-only test confirms the validation warning: GT4 false-fifth
  and sixth-lane overcount remain severe.
- This does not justify continuing broad rank/farspur/shortside pressure. The
  remaining problem is still selective candidate quality/ranking for true
  4th/5th lanes under a guard against false extra lanes.

Smallest safe next action:

Do not tune thresholds, checkpoint choice, NMS, rank weight, farspur weight, or
shortside settings from this report. Use only as diagnostic evidence when
designing a narrower GT5 short/weak-visible rescue with explicit overcount
protection.

## 2026-06-28 Hard-Sampling Short0601 Tradeoff

The completed E2 hard-sampling run is diagnostic-only, not a promoted
candidate.

```text
E1 baseline:
run = gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1_official_val_sweep/tusimple_official_sweep_summary.json
official_acc = 0.971208
official_FP = 0.022957
official_FN = 0.014004
count_acc_4 = 0.848485
count_acc_5 = 0.972973
4->5 = 9
5->4 = 2

E2 short0601 hard sampling:
run = gcs_yolo_lane_s_q12_k56_boundary02_short0601_v1
gcs_hard_sampling = true
gcs_official_best = false
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_short0601_v1_official_val_sweep/tusimple_official_sweep_summary.json
best decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=2
official_acc = 0.969988
official_FP = 0.022544
official_FN = 0.018825
count_acc_4 = 0.969697
count_acc_5 = 0.959459
4->5 = 1
5->4 = 3
```

Supported interpretation:

- Hard sampling cleanly attacks the GT4 false-fifth pattern:
  `4->5` drops by `8`, and `count_acc_4` improves by `0.121212`.
- The official-val tradeoff is not acceptable for promotion:
  `official_acc` drops by `0.001220` and `official_FN` rises by `0.004821`.
- GT5 does not collapse (`5->4` rises only from `2` to `3`), so the remaining
  issue is mainly recall/geometry or score calibration rather than the
  spurious-lite GT5-safe problem.
- The run used a post-hoc official-val sweep over `weights/best.pt`
  (`gcs_official_best=false`), so it is weaker than a formal
  training-time-official-best run even before the metric rejection.

Decision: do not run final test for `boundary02_short0601_v1`. Keep later
E3-lite spurious-negative ablations initialized from E1, not from this E2
hard-sampling checkpoint. If revisited, diagnose the added FN before changing
sampling weights again.

## 2026-06-28 E3-Lite Spurious-Negative Tradeoff

The completed E3-lite spurious-negative official-val sweep is diagnostic-only,
not a promoted candidate.

```text
E1 baseline:
run = gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1_official_val_sweep/tusimple_official_sweep_summary.json
official_acc = 0.971208
official_FP = 0.022957
official_FN = 0.014004
count_acc_4 = 0.848485
count_acc_5 = 0.972973
4->5 = 9
5->4 = 2

E3-lite spurious:
run = gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1
gcs_spurious_neg = 0.1
gcs_hard_sampling = false
gcs_official_best = false
results.csv rows = 59 logged epochs
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1_official_val_sweep/tusimple_official_sweep_summary.json
sweep rows = 3360, count_aware_topk = false
official_acc = 0.971721
official_FP = 0.013866
official_FN = 0.015611
count_acc_4 = 0.924242
count_acc_5 = 0.878378
4->5 = 4
5->4 = 9
```

Supported interpretation:

- The spurious-negative loss suppresses the intended false-fifth pattern:
  `4->5` drops by `5`, `count_acc_4` improves by `0.075757`, and FP drops by
  `0.009091`.
- The cost is true GT5 retention: `5->4` increases by `7`, and
  `count_acc_5` drops by `0.094595`.
- This means short unmatched duplicate-like query suppression is useful, but
  the selection rule is not GT5-safe enough when true fifth lanes are short.
- Caveat: this run used a post-hoc official-val sweep over `weights/best.pt`;
  `args.yaml` has `gcs_official_best=false`, so it did not preserve a
  training-time `official_best.pt` / `official_best_decode.yaml`.
- Caveat: the selected metrics are tied across 36 sweep rows under the branch
  selection priority. Treat the recorded decode as a representative selected
  row, not a unique postprocess preference.

The next bottleneck is no longer whether spurious-negative suppression can
reduce GT4 over-count. It can. The bottleneck is preserving true GT5 short
side lanes while retaining the FP / `4->5` benefit. The original default-decode
`spurious_lite_v1` artifact was not promotable and should not have been sent to
test for selection. After the 2026-07-02 official-val-only
`valid_before_maxdet` counterfactual selected a stronger decode for the same
`best.pt`, the user requested a reporting-only official test. That test report
is recorded below and must not be used for tuning.

```text
spurious_lite + valid_before test:
summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1_official_test_best_from_val_valid_before_maxdet/tusimple_official_summary.json
decode = conf=0.005, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_acc = 0.965483
official_FP = 0.030847
official_FN = 0.027678
count_acc = 0.882818
count_acc_4 = 0.606838
count_acc_5 = 0.843585

E1 count-boundary test comparator:
summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1_official_test_best_from_val/tusimple_official_summary.json
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=2, valid_before_maxdet=false
official_acc = 0.965428
official_FP = 0.031368
official_FN = 0.026060
count_acc = 0.874551
count_acc_4 = 0.568376
count_acc_5 = 0.834798
```

The integrated reading is narrow: spurious-lite + valid-before is marginally
higher than E1 on this one reporting-only test comparison (`+0.000055` ACC)
and improves GT4 false-fifth count shape, but it still has broad final-test
count errors and higher FN. Keep test closed for any further threshold,
checkpoint, decode, or loss decisions. Future formal candidates should use
training-time `official_best` and return to official-val/train diagnostics.

The 2026-07-02 short-side-geometry follow-up
`gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1`
is also diagnostic-only, not promotable. It improved the canonical 363-image
official-val surface over the no-shortside same-line comparator:

```text
shortside external official-val ACC = 0.977032
no-shortside external official-val ACC = 0.974538
delta = +0.002494
shortside official-val FP/FN = 0.010882 / 0.010790
shortside official-val count_acc_4/5 = 0.954545 / 0.986486
```

But the same official-val-selected decode regressed reporting-only official
test:

```text
shortside test ACC = 0.963348
no-shortside test ACC = 0.964358
delta = -0.001010
shortside test FP/FN = 0.032171 / 0.031722
shortside test count_acc_4/5 = 0.617521 / 0.801406
no-shortside test count_acc_4/5 = 0.611111 / 0.836555
GT5 5->4 = 85 for shortside, 69 for no-shortside
```

Supported interpretation:

- The matched short-side geometry term primarily hit GT4 side lanes in
  training (`train/short_side_geom_gt4 ~= 32.83` versus
  `train/short_side_geom_gt5 ~= 3.41` over the last 20 recorded epochs), so
  the 363 gain does not prove GT5 retention generalization.
- The 363 official-val split has no `0530` images and all GT5 validation
  images are from `0601`; official test has `0530=1248` images, including
  difficult `0530|GT4` and `0530|GT5` groups. This makes the 363 count-shape
  gain too narrow to trust as a final-test improvement signal.
- The next bottleneck is not adding more matched geometry in isolation. It is
  preserving GT5 short-side lanes and count-score calibration across date/GT
  count groups while keeping the GT4/FP benefit. Do not tune from the
  reporting-only test result.

The 2026-07-03 far-spurious follow-up
`gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_farspur005_v1`
is also diagnostic-only, not promotable. It added GT3/GT4-only far unmatched
query pressure on top of the same GT5-only protect + GT4W15 line:

```text
farspur official-val ACC = 0.977758
farspur official-val FP/FN = 0.008219 / 0.008953
farspur official-val count_acc_4/5 = 0.954545 / 0.972973
farspur official-val count_confusion includes 4->5=0, 5->4=2
same-line no-shortside official-val ACC = 0.974538
```

But the official-val-selected decode regressed reporting-only official test
against the same-line no-shortside comparator:

```text
farspur test ACC = 0.963907
no-shortside test ACC = 0.964358
delta = -0.000451
farspur test FP/FN = 0.031860 / 0.031332
no-shortside test FP/FN = 0.030721 / 0.029415
farspur test count_acc_4/5 = 0.628205 / 0.815466
no-shortside test count_acc_4/5 = 0.611111 / 0.836555
GT4 4->5 = 65 for farspur, 75 for no-shortside
GT5 5->4 = 77 for farspur, 69 for no-shortside
```

Supported interpretation:

- Far-spur is useful diagnostically: it reduces GT4 false-fifth lanes on both
  official-val and reporting-only test.
- It is not a promoted candidate because the same test comparison loses ACC,
  FP, FN, and GT5 retention. The gain is a GT4 over-count reduction traded for
  broader GT5 undercount.
- Training logs show the loss was active early and then saturated on train:
  first-20 train `far_spur_neg` averaged about `2.56`, last-20 averaged about
  `0.024`, and `far_spur_gt5` stayed `0.0`. The remaining issue is
  generalization, not an inactive loss.
- Do not tune thresholds, NMS, checkpoint choice, or far-spur weights from the
  reporting-only test. If continuing, run train/val-only raw-candidate and
  post-NMS diagnostics for GT4/GT5 date/count groups before another ablation.
  Reporting-only test date/count breakdowns are risk descriptions only, not
  selection gates.

The follow-up combination
`gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom010_gt5off_farspur005_v1`
is also rejected. It combined reduced short-side geometry
(`gcs_short_side_geom=0.10`, `gcs_short_side_geom_weight_gt5=0.0`) with
GT3/GT4-only far-spur (`gcs_far_spurious_neg=0.05`,
`gcs_far_spurious_max_gt_lanes=4`, `gcs_far_spurious_gt5_weight=0.0`).

Protocol caveat:

- The first external sweep directory named
  `...shortsidegeom010_gt5off_farspur005_v1_official_best_val_sweep_valid_before_maxdet`
  used `weights/best.pt`, while the paired test used
  `weights/official_best.pt`. Treat that pair as protocol-mismatched.
- The corrected official-val sweep on `weights/official_best.pt` is
  `...shortsidegeom010_gt5off_farspur005_v1_official_bestpt_val_sweep_valid_before_maxdet`.

Corrected official-val evidence:

```text
selected decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official-val ACC = 0.976937
official-val FP/FN = 0.003489 / 0.008724
official-val count_acc_4/5 = 0.939394 / 0.878378
official-val count_confusion includes 4->5=0, 5->4=9
zero sweep rows had 5->4 <= 2 or count_acc_5 >= 0.972973
```

Protocol-correct reporting-only test evidence:

```text
test summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom010_gt5off_farspur005_v1_official_best_test_best_from_correct_val/tusimple_official_summary.json
test ACC = 0.963280
test FP/FN = 0.031087 / 0.032590
test count_acc_4/5 = 0.632479 / 0.773286
same-line no-shortside test ACC = 0.964358
same-line no-shortside count_acc_4/5 = 0.611111 / 0.836555
GT4 4->5 = 63 for combo, 75 for no-shortside
GT5 5->4 = 103 for combo, 69 for no-shortside
```

Supported interpretation:

- The combination gives the intended GT4 over-count signal: reporting-only
  test `4->5` improves from `75` to `63`, and `count_acc_4` improves from
  `0.611111` to `0.632479`.
- The cost is too high: GT5 undercount becomes the dominant failure, with
  `5->4` increasing to `103` and `count_acc_5` dropping to `0.773286`.
- The damage is concentrated in `0601|GT5`: reporting-only test `5->4`
  increases from `26` for no-shortside to `62` for the combination.
- The next bottleneck is not more short-side or far-spur pressure. It is
  preventing GT5 existence/rank suppression and remaining GT5-group
  duplicate-like spurious pressure before adding more GT4 over-count pressure.

The 2026-07-06 shortside boost-only fixed-threshold follow-up
`gcs_yolo_lane_s_q12_k56_ablate_shortside_boost_only_fixedthr_v1` is also
diagnostic-only. It used `gcs_shortside_rawmatch_boost=0.25` with
`gcs_shortside_exist_target_floor=0.0`, so it tested BCE weight boost without
the target-floor protection contract.

Official-val selected decode:

```text
val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_boost_only_fixedthr_v1_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
selected decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official-val ACC = 0.974445
official-val FP/FN = 0.007576 / 0.009642
official-val count_acc_4/5 = 0.954545 / 0.959459
official-val count_confusion includes 4->3=3, 4->5=0, 5->4=3, 5->5=71
```

Reporting-only test evidence:

```text
test summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_boost_only_fixedthr_v1_test_best_from_val_b/tusimple_official_summary.json
test ACC = 0.964144
test FP/FN = 0.028666 / 0.028696
test count_acc_4/5 = 0.630342 / 0.757469
spurious_lite + valid_before test ACC = 0.965483
spurious_lite + valid_before count_acc_4/5 = 0.606838 / 0.843585
GT4 4->5 = 67 for boost-only, 79 for spurious_lite + valid_before
GT5 5->4 = 111 for boost-only, 63 for spurious_lite + valid_before
```

Supported interpretation:

- Boost-only gives the intended GT4 over-count signal: reporting-only test
  `4->5` improves from `79` to `67`, and FP improves from `0.030847` to
  `0.028666`.
- The tradeoff is not acceptable: ACC drops to `0.964144`, FN worsens to
  `0.028696`, and GT5 retention collapses (`5->4: 63 -> 111`,
  `count_acc_5: 0.843585 -> 0.757469`).
- The result supports the loss-contract concern that BCE weight boost alone is
  not a protection mechanism when the quality-aware target remains low. Future
  shortside experiments must keep boost-only, floor-only, and boost+floor
  ablations distinct.
- Do not tune thresholds, checkpoint, NMS, decode, or loss weights from this
  reporting-only test. If continuing, diagnose on official-val/train-val why
  the selected `point_valid_thr=0.6` row is not GT5-safe enough, and predefine
  any future GT5 gate on official-val before another test.

The 2026-07-06 shortside protect follow-up
`gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2` is also
diagnostic-only. It used `gcs_shortside_rawmatch_boost=0.25` with
`gcs_shortside_exist_target_floor=0.7`, so it tested the intended boost+floor
protection contract.

Primary protocol caveat:

- The run has both `weights/best.pt` and `weights/official_best.pt` companion
  sweeps/tests. Use `weights/official_best.pt` as primary because the training
  args set `gcs_official_best=true`.

Official-val selected decode:

```text
val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
selected decode = conf=0.008, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official-val ACC = 0.972798
official-val FP/FN = 0.011433 / 0.012626
official-val count_acc_4/5 = 0.939394 / 0.959459
official-val count_confusion includes 4->3=3, 4->5=1, 5->4=3, 5->5=71
```

Reporting-only test evidence:

```text
test summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2_test_best_from_val_b/tusimple_official_summary.json
test ACC = 0.964281
test FP/FN = 0.029014 / 0.029385
test count_acc_4/5 = 0.621795 / 0.766257
spurious_lite + valid_before test ACC = 0.965483
spurious_lite + valid_before count_acc_4/5 = 0.606838 / 0.843585
GT4 4->3/4->5 = 112 / 65 for protect, 105 / 79 for spurious_lite + valid_before
GT5 5->4 = 105 for protect, 63 for spurious_lite + valid_before
```

Supported interpretation:

- The target floor does not solve the GT5 retention failure. Versus
  spurious-lite + valid-before, GT4 false-fifth improves (`4->5: 79 -> 65`),
  but GT5 undercount remains severe (`5->4: 63 -> 105`) and `count_acc_5`
  drops from `0.843585` to `0.766257`.
- Versus boost-only, official-val ACC, FP, FN, and `count_acc_4` are worse.
  The reporting-only test ACC gain is only `+0.000137` and does not offset the
  worse FP/FN and GT4 `4->3` shape.
- This rejects the simple "boost + target floor 0.7" protection setting. The
  remaining issue is not only low existence target on matched shortside lanes;
  selected rows can still under-retain true GT5 lanes, especially at
  `point_valid_thr=0.6`.
- Do not tune thresholds, checkpoint, NMS, decode, target floor, or loss
  weights from this reporting-only test. If continuing, run official-val or
  train-val-only error decomposition around the added `4->3` and `5->4`
  undercount before defining another ablation.

The 2026-07-06 GT4/GT5 ranking follow-ups
`gcs_yolo_lane_s_q12_k56_ablate_rank_pair_gt4gt5_002_v1` and
`gcs_yolo_lane_s_q12_k56_ablate_rank_full_dupignore_gt4gt5_002_v1` are also
diagnostic-only. Both used `gcs_rank_topk_weight=0.02`,
`gcs_rank_margin=0.05`, `gcs_rank_pos_scope=gt4gt5_matched`, and
`gcs_rank_pair_reduction=global_pair_mean`. The difference is intentional:
rank-pair kept duplicate-like queries as base BCE negatives, while the full
duplicate-ignore contract set `gcs_base_ignore_duplicate_like=True`.

Primary protocol note:

- Use `weights/official_best.pt` evidence as primary because both formal
  training runs enabled `gcs_official_best=true`. The companion best.pt sweep
  for rank-pair is lower on official-val and not the selected checkpoint.

Official-val selected decode:

```text
rank_pair val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_rank_pair_gt4gt5_002_v1_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
rank_pair decode = conf=0.01, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=3, valid_before_maxdet=true
rank_pair official-val ACC = 0.974062
rank_pair official-val FP/FN = 0.007576 / 0.010560
rank_pair official-val count_acc_4/5 = 0.939394 / 0.959459
rank_pair official-val count_confusion includes 4->3=4, 4->4=62, 4->5=0, 5->4=3, 5->5=71

rank_full val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_rank_full_dupignore_gt4gt5_002_v1_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
rank_full decode = conf=0.008, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
rank_full official-val ACC = 0.974208
rank_full official-val FP/FN = 0.007668 / 0.012167
rank_full official-val count_acc_4/5 = 0.909091 / 0.945946
rank_full official-val count_confusion includes 4->3=6, 4->4=60, 4->5=0, 5->4=4, 5->5=70
```

Reporting-only test evidence:

```text
rank_pair test summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_rank_pair_gt4gt5_002_v1_official_test_best_from_val_bb/tusimple_official_summary.json
rank_pair test ACC = 0.963617
rank_pair test FP/FN = 0.026725 / 0.030404
rank_pair test count_acc_4/5 = 0.632479 / 0.738137
rank_pair count_confusion includes 4->3=113, 4->5=58, 5->3=28, 5->4=121, 5->5=420

rank_full test summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_rank_full_dupignore_gt4gt5_002_v1_official_test_best_from_val_b/tusimple_official_summary.json
rank_full test ACC = 0.963831
rank_full test FP/FN = 0.028313 / 0.029625
rank_full test count_acc_4/5 = 0.632479 / 0.750439
rank_full count_confusion includes 4->3=110, 4->5=62, 5->3=27, 5->4=115, 5->5=427

spurious_lite + valid_before test ACC = 0.965483
spurious_lite + valid_before test FP/FN = 0.030847 / 0.027678
spurious_lite + valid_before count_acc_4/5 = 0.606838 / 0.843585
spurious_lite + valid_before count_confusion includes 4->3=105, 4->5=79, 5->3=26, 5->4=63, 5->5=480
```

Supported interpretation:

- Both ranking ablations improve the official-val surface, primarily by
  reducing false positives and GT4 false-fifth rows, but the 363 split does not
  predict final-test GT5 retention.
- Rank-pair improves reporting-only test GT4 `4->5` from `79` to `58`, and
  full duplicate-ignore improves it to `62`, but both trade this for a severe
  GT5 undercount regression: `5->4` grows from `63` to `121` for rank-pair and
  to `115` for full duplicate-ignore.
- Full duplicate-ignore is marginally better than rank-pair on reporting-only
  test ACC (`0.963831` vs `0.963617`) and GT5 retention (`0.750439` vs
  `0.738137`), but it remains below the spurious-lite + valid-before baseline
  and well below the `0.9655` target.
- The result rejects simple `rank_topk_weight=0.02` all-GT4/GT5 matched
  ranking, whether duplicate-like queries remain base BCE negatives or are
  protected by the full duplicate-ignore contract.
- Do not tune thresholds, checkpoint, NMS, duplicate-ignore policy, rank
  margin, or rank weight from these reporting-only test results. If continuing,
  keep analysis on official-val/train-val diagnostics and inspect why ranking
  rows that improve GT4 false-fifth produce GT5 under-retention before defining
  any smaller or GT5-gated ranking ablation.

The existing `gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt5safe_v1` artifact
has only `gcs_spurious_gt5_weight=0.25`, starts from E1 `best.pt`, and keeps
`gcs_hard_sampling=false`, but it did not apply the GT4-strong weight. It is
not the current GT4-strong + GT5-safe candidate.

GT5-safe official-val result:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt5safe_v1_official_val_sweep/tusimple_official_sweep_summary.json
best decode = conf=0.005, point_valid_thr=0.55, nms_dist_px=18.0, max_det=5, min_points=2
official_acc = 0.971777
official_FP = 0.022957
official_FN = 0.015611
count_acc_4 = 0.863636
count_acc_5 = 0.986486
count_confusion = 3->3=218, 3->4=5, 4->3=1, 4->4=57, 4->5=8, 5->4=1, 5->5=73
```

This run fixes the spurious-lite GT5 undercount (`5->4: 9 -> 1`) but gives
back most of the intended GT4 false-fifth suppression (`4->5: 4 -> 8`) and FP
benefit (`0.013866 -> 0.022957`). The sweep has no row that simultaneously
keeps `4->5 <= 4` and `5->4 <= 2`. This motivated the GT-count-weighted
follow-up below: keep the GT5-safe reduction, but add stronger GT4
spurious-negative pressure.

GT4-strong + GT5-safe official-val result:

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt4strong_gt5safe_v1
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt4strong_gt5safe_v1_official_val_sweep/tusimple_official_sweep_summary.json
weights = gt3 1.0, gt4 1.5, gt5 0.25
gcs_spurious_neg = 0.1
gcs_hard_sampling = false
gcs_official_best = false
results.csv rows = 43 epochs, nonfinite_count = 0
best decode = conf=0.003, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=2
official_acc = 0.970530
official_FP = 0.025666
official_FN = 0.016529
count_acc_4 = 0.848485
count_acc_5 = 0.986486
count_confusion = 3->3=216, 3->4=7, 4->3=1, 4->4=56, 4->5=9, 5->4=1, 5->5=73
```

This rejects the simple "GT4 weight 1.5 + GT5 weight 0.25" fix. GT5 remains
safe, but GT4 false-fifth suppression disappears: selected `4->5=9`, equal to
E1 and worse than both `spurious_lite_v1` (`4`) and `spurious_gt5safe_v1` (`8`).
The full 1512-row official-val sweep has zero rows that meet the requested
gate; even the best individual limits miss `official_acc`, FP, `count_acc_4`,
and `4->5` (`max_acc=0.970530`, `min_FP=0.020707`,
`max_count_acc_4=0.893939`, `min_4->5=6`).

The active bottleneck is therefore not solved by scalar GT4 spurious-negative
weighting alone. Before adding another train-side loss or changing decode,
inspect official-val `GT4 4->5` and `GT3 3->4` failures per image and verify
whether the GT4 unmatched-query spurious candidate set has enough coverage; the
late training logs show very sparse GT4 spurious candidates compared with GT5.

## Legacy Post-b653 Official-Val Selection State

The 2026-06-21 `gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03`
candidate was selected on official-val in a later post-`b6535f641` experiment
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

## Legacy 2026-06-20 Train/Val Count-Confusion Diagnostic

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

## Legacy 2026-06-20 GT4 Short-Lane Boost Result

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

## Legacy 2026-06-21 GT4 Short-Lane Boost 1.5 Result

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

## Legacy 2026-06-21 gt4short15 Failure Trace and NMS Check

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

## Legacy 2026-06-21 User-Requested Reporting-Only Test Batch

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

## Legacy 2026-06-22 count03_under5_00 Under5-Loss Ablation

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

## Legacy 2026-06-22 dupmargin005 Duplicate-Margin Near-Miss

The completed `gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03`
run enabled a post-`b6535f641` duplicate-margin experiment that is not present
in the active `424ab1c86` rollback code:

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
  selected candidate in that legacy official-val protocol.
- Smallest safe next action: if continuing this line, run train/val-only
  failure traces for the selected decode `conf=0.05`, `point_valid_thr=0.45`,
  `nms_dist_px=0.0`, `max_det=6`, `min_points=6` and compare failure buckets
  against `count03_under5_03` and `gt4short15`. Do not tune from final test.

## Legacy 2026-06-23 spurmargin003 Spurious-Margin Rejection

The completed `gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03`
run enabled a post-`b6535f641` spurious-margin experiment that is not present
in the active `424ab1c86` rollback code:

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

## Legacy 2026-06-23 shortpos Positive Short-Lane Loss Rejection

The completed `gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03`
run enabled post-`b6535f641` positive short-lane losses that are not present in
the active `424ab1c86` rollback code:

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

## Legacy 2026-06-23 farspur001 + gt5rank001 Rejection

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

## Legacy 2026-06-24 dupmargin005 Official-Val Failure-Mode Compare

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

## Legacy 2026-06-24 GT3 Extra-Survival Follow-Up

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

## Legacy 2026-06-24 Official-Test GT4 Count-Shape Collapse

The later reporting-only official-test result exposed a sharper bottleneck
than the official-val sweep suggested. The fixed candidate pool is:

```text
conf = 0.005
point_valid_thr = 0.5
nms_dist_px = 0.0
max_det = 8
min_points = 6
```

Reporting-only official-test evidence:

```text
images = 2782
official_acc = 0.965077
count_acc = 0.875988
count_acc_3 = 0.976437
count_acc_4 = 0.538462
count_acc_5 = 0.852373
GT4 confusion: 4->3=118, 4->4=252, 4->5=93, 4->6=4
```

The corresponding official-val selected row under the same candidate pool is
well behaved by comparison:

```text
official_acc = 0.969665
count_acc = 0.961433
count_acc_3 = 0.968610
count_acc_4 = 0.909091
count_acc_5 = 0.986486
GT4 confusion: 4->3=2, 4->4=60, 4->5=4
```

Integrated conclusion:

- Supported fact: the dominant final-test weakness is `GT4` count shape, not a
  single global threshold miss and not a GT3-only `3->4` failure.
- Supported fact: final-test `GT4` fails in both directions at once
  (`4->3=118` and `4->5=93`). Raising a global threshold may reduce `4->5`
  overcount but will likely worsen `4->3`; lowering it has the opposite risk.
- Supported fact: the val/test gap is large: official-val `GT4`
  `count_acc=0.909091`, while reporting-only official-test `GT4`
  `count_acc=0.538462`.
- Decision: stop blind threshold sweeps for this bottleneck. The next step is
  diagnostic first: count-oracle topK, then count-head guided topK only if a
  real count head exists and official-val does not regress.

New diagnostic tooling is available in `tools/eval_tusimple_official.py`:

```text
--oracle-count-topk
--dump-count-head-stats
--count-guided-topk
--count-guided-min-prob
--count-guided-allowed-counts
```

`--oracle-count-topk` is diagnostic-only and uses GT lane count after normal
decode candidate generation; it must never be used for formal submission or
test selection. The active 5-25-3 K56 output contract has no independent count
head logits, so `--dump-count-head-stats` reports `supported=false` unless a
future checkpoint/model output includes one of the recognized count-logit keys.

Priority order:

```text
1. Run normal decode vs oracle-count topK on official-val first.
2. If a count head exists and is stronger than decoded lane count on GT4, test
   count-head guided topK on official-val.
3. Use official-test only as reporting-only evidence for an already selected
   official-val candidate/diagnostic.
4. If oracle-count topK does not materially improve GT4, stop count-guided
   decode and move to GT4 candidate quality/ranking work, such as weak-positive
   retention or extra-ranking loss.
```

Completed diagnostic evidence:

```text
official-val artifact:
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_count_oracle_official_val/summary.json

normal:
  official_acc=0.969665, FP=0.020615, FN=0.014004, count_acc=0.961433
  count_acc_4=0.909091
  GT4: 4->3=2, 4->4=60, 4->5=4, 4->6=0
oracle_count_topk:
  official_acc=0.969550, FP=0.013958, FN=0.014004, count_acc=0.991736
  count_acc_4=0.969697
  GT4: 4->3=2, 4->4=64, 4->5=0, 4->6=0

reporting-only official-test artifact:
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_count_oracle_official_test_reporting_only/summary.json

normal:
  official_acc=0.965254, FP=0.030649, FN=0.026959, count_acc=0.875629
  count_acc_4=0.542735
  GT4: 4->3=116, 4->4=254, 4->5=94, 4->6=4
oracle_count_topk:
  official_acc=0.964709, FP=0.020681, FN=0.028726, count_acc=0.932423
  count_acc_4=0.752137
  GT4: 4->3=116, 4->4=352, 4->5=0, 4->6=0
```

Updated conclusion:

- Oracle-count topK confirms that the `4->5`/`4->6` overcount side is a final
  keep-count problem: those errors drop to zero on GT4.
- Oracle-count topK does not move the `4->3` undercount side at all on
  reporting-only official-test (`116 -> 116`). Therefore the severe GT4 test
  collapse is not primarily solved by count-guided final topK alone.
- The active model has no independent count head; count-head stats report
  `supported=false`, and count-guided topK falls back to normal decode.
- Next priority is GT4 candidate quality/ranking for the missing fourth lane.
  Count-guided decode should not be the next mainline candidate unless a future
  model adds a real count head and official-val evidence shows it fixes GT4
  without a material ACC/FN regression.

## Legacy 2026-06-24 GT4 4->3 Missing-Lane Raw-Query Diagnostic

The follow-up diagnostic is now `tools/diagnose_gt4_missing_lane_raw_queries.py`.
It uses the same normal decode candidate pool:

```text
conf = 0.005
point_valid_thr = 0.5
nms_dist_px = 0.0
max_det = 8
min_points = 6
match gate = overlap >= 3 official h-samples and mean_abs_x_error <= 20px
```

The script dumps all raw `Q=12` queries for each selected image to
`raw_queries.csv`, then writes `per_missing_lane.csv`,
`per_image_summary.csv`, `summary.json`, and visualizations for the first
selected images. It stages each missing GT lane through raw geometry,
point-valid, min-points, confidence, and final decode. It does not change
training, model structure, decoder defaults, official metrics, or selection
rules.

Reporting-only official-test result:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_gt4_missing_raw_queries_official_test_reporting_only/summary.json
records = 2782
GT4 4->3 images = 116
missing GT lanes = 129
drop_reason = geometry_bad 95, low_point_valid 33, low_score 1
stage recall on missing lanes:
  raw_match = 34/129 = 0.263566
  after_point_valid = 1/129 = 0.007752
  after_min_points = 1/129 = 0.007752
  after_conf = 0/129 = 0.0
  final_decode = 0/129 = 0.0
missing visible buckets:
  <=5: 12, 6..10: 64, 11..20: 44, 21..30: 3, >40: 6
missing side/order:
  left_inner 52, right_inner 41, right_outer 34, left_outer 2
```

Official-val comparison on the same 363-image official-val subset:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_gt4_missing_raw_queries_official_val/summary.json
records = 363
GT4 4->3 images = 2
missing GT lanes = 3
drop_reason = geometry_bad 2, low_point_valid 1
stage recall on missing lanes:
  raw_match = 1/3 = 0.333333
  after_point_valid = 0/3 = 0.0
  after_min_points = 0/3 = 0.0
  after_conf = 0/3 = 0.0
  final_decode = 0/3 = 0.0
```

Integrated conclusion:

- The official-test `GT4 4->3` count is confirmed at `116` under the fixed
  candidate pool. Because strict final matching finds multiple unmatched GT
  lanes in some `4->3` images, the per-lane missing total is `129`.
- The val/test gap is mostly frequency, not a different drop taxonomy. Both
  surfaces are dominated by `geometry_bad` followed by `low_point_valid`.
- The missing fourth lane usually does not have an acceptable raw query under
  the strict `20px` official-h-sample gate. When raw geometry exists, the next
  failure is usually point-valid; almost none reach the score/ranking stage.
- `low_score` is only `1/129`, so a weak-positive survival/ranking loss is not
  the main next step for this bottleneck.
- The next mainline should move from count-guided topK and GT3-extra hinge work
  to `GT4` missing-lane candidate recall: first inspect whether a GT4
  lane-balanced point objective or GT4-focused sampling improves raw geometry.
  If a narrowed train/val trace shows raw geometry already exists but
  point-valid collapses, then consider a GT4 weak-lane valid recall loss.

Implementation update:

The next default-disabled experiment knobs are now focused on GT4 missing-lane
candidate recall, not count-guided topK or GT3 extra-query ranking:

```text
loss item = gt4_lane_balanced_point_loss
gain arg = --gcs-gt4-lane-balanced-point, default 0.0
top-k arg = --gcs-gt4-lane-balanced-topk, default 1
max multiplier arg = --gcs-gt4-lane-balanced-max-mult, default 2.0
optional sampler arg = --gcs-gt4-sample-gain, default 1.0
```

The loss applies only to `gt_lane_count == 4` samples with all four GT lanes
matched by the existing Hungarian assignment. It finds the matched GT lane with
the largest point regression loss, upweights that weak lane, and normalizes
per-image lane weights back to mean `1.0`. It does not change existence
targets, query ranking, model structure, decoder, evaluation, Q/K, or fixed-y
labels.

`tools/build_gt4_hard_val_split.py` can build an internal GT4-hard validation
list from train labels by running
`tools/diagnose_gt4_missing_lane_raw_queries.py` on GT4 train samples with the
fixed candidate pool. The generated hard-val list is diagnostic-only and must
not be fed back into training. Promotion still uses official-val, with final
test reserved for reporting-only evidence after selection.

## Legacy 2026-06-24 dupmargin005_gt3extra003 Rejection

The completed
`gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03`
run enabled the narrow GT3 extra-survival loss on top of `dupmargin005`:

```text
gcs_duplicate_margin = 0.05
gcs_gt3_extra_survival = 0.03
gcs_gt3_extra_margin_logit = 0.05
gcs_gt3_extra_topk = 1
train log item = gt3_extra_survival_loss
```

Its official-val sweep is valid 363-image selection evidence:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_official_val_sweep
split = val
gt_json = runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json
best/tied decode = conf=0.05, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
official-val ACC = 0.969665
official-val FP = 0.020615
official-val FN = 0.014004
official-val official_score = 0.968973
official-val count_acc = 0.961433
official-val count_acc_3/4/5 = 0.968610 / 0.909091 / 0.986486
count_confusion = 3->3=216, 3->4=7, 4->3=2, 4->4=60, 4->5=4, 5->4=1, 5->5=73
```

The sweep file's first `best` row uses `conf=0.005`, but the selected
`conf=0.05` row has the same official ACC, FP, FN, official score, count
accuracy, and count confusion. The `conf=0.05` selected decode is therefore
still official-val evidence, not a final-test-derived choice.

Failure-mode compare was run after the two requested result directories were
reviewed, using the same strict official-val matcher as the earlier
`dupmargin005` comparison:

```text
summary = runs/gcs_lane/failure_compare/dupmargin005_gt3extra003_compare/summary.json
predictions = count03, gt4short15, dupmargin005, gt3extra003 official-val selected decodes
match gate = overlap >= 3 official h-samples and mean absolute x error <= 20px
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

The selected-decode final test is reporting-only and cannot promote the run:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_official_val_selected_decode_test/tusimple_official_summary.json
split = test
decode = conf=0.05, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
official_test ACC = 0.965077
FP = 0.030494
FN = 0.027528
official_score = 0.963917
count_acc = 0.875988
count_acc_3/4/5 = 0.976437 / 0.538462 / 0.852373
count_confusion includes 3->4=34, 4->5=93, 5->4=46
```

Integrated conclusion:

- Supported fact: the loss hit part of its direct target. GT3 `3->4` improves
  versus `dupmargin005` (`11 -> 7`) and is also lower than `gt4short15` (`8`).
- Supported fact: it does not restore the `count03` GT3 shape. GT3 `3->4`
  remains worse than `count03` (`7 > 4`), and all-image `missed_short_gt`
  worsens versus both `dupmargin005` (`28 -> 36`) and `count03` (`33 -> 36`).
- Supported fact: GT5 is not harmed on official-val. `5->4` improves from
  `dupmargin005`'s `2` to `1`, matching `count03`, and `count_acc_5` returns
  to `0.986486`.
- Supported fact: GT4 is harmed relative to the parent. `count_acc_4` drops
  from `dupmargin005`'s `0.939394` to `0.909091`, and `4->5` increases
  `3 -> 4`.
- Supported fact: the primary promotion metric fails. Official-val ACC
  `0.969665` is below `count03_under5_03` (`0.969976`), `dupmargin005`
  (`0.970272`), and `gt4short15` (`0.970851` / rerun `0.970837`).
- Decision: reject `dupmargin005_gt3extra003` as a promotion. Do not tune
  final-test thresholds, `max_det`, `min_points`, NMS, checkpoint choice, or
  loss gains from the reporting-only test.

Smallest safe next action:

Do not continue with a blind larger GT3-extra gain. If this line is revisited,
first inspect whether the `missed_short_gt` increase comes from raising the
weakest GT3 matched logit side of the hinge. A smaller gain or
`min_matched_logit.detach()` is only a train/val or official-val ablation, not
a final-test-driven change.

## Legacy 2026-06-25 GT4 Lane-Balanced Point Sweep Result

The two requested GT4 lane-balanced point official-val sweeps are complete.
Both runs build on `dupmargin005` and use the branch-local default-disabled
GT4 candidate-recall tooling:

```text
selected run = gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025
loss gain = gcs_gt4_lane_balanced_point 0.25
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025_official_val_sweep/tusimple_official_sweep_summary.json
selected decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=5, min_points=5
official-val ACC = 0.970975
official-val FP = 0.017264
official-val FN = 0.012167
official-val official_score = 0.970386
official-val count_acc = 0.966942
official-val count_acc_3/4/5 = 0.968610 / 0.954545 / 0.972973
count_confusion = 3->3=216, 3->4=7, 4->3=1, 4->4=63, 4->5=2, 5->4=2, 5->5=72
```

Compared with the previous `gt4short15` official-val candidate, `gt4pt025`
raises ACC by `+0.000124` and improves GT4 count accuracy from `0.848485` to
`0.954545`. Compared with `dupmargin005`, it also improves official-val ACC,
official score, FP/FN balance, and GT4 count accuracy.

The fixed candidate-pool diagnostic supports that the gain is not only a sweep
artifact:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025_official_val_fixed_pool/summary.json
fixed pool decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
official-val ACC = 0.970363
official-val FP = 0.018457
official-val FN = 0.012167
official-val official_score = 0.969751
official-val count_acc = 0.961433
official-val count_acc_3/4/5 = 0.959641 / 0.954545 / 0.972973
```

The larger point gain is rejected:

```text
rejected run = gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt050
loss gain = gcs_gt4_lane_balanced_point 0.50
sweep = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt050_official_val_sweep/tusimple_official_sweep_summary.json
best decode = conf=0.1, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=6
official-val ACC = 0.964381
official-val FP = 0.034389
official-val FN = 0.021120
official-val official_score = 0.963271
official-val count_acc = 0.903581
official-val count_acc_3/4/5 = 0.932735 / 0.803030 / 0.905405
```

The remaining bottleneck is still GT4-hard candidate recall. The train-derived
GT4-hard diagnostic for `gt4pt025` found `22` missing GT lanes over `19`
selected hard images:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025_gt4_hard_val_build/diagnostic/summary.json
drop_reason = geometry_bad 13, low_point_valid 9
raw_match_recall = 9/22 = 0.409091
after_point_valid_recall = 0/22 = 0.0
after_min_points_recall = 0/22 = 0.0
after_conf_recall = 0/22 = 0.0
final_decode_recall = 0/22 = 0.0
```

Decision at that point: select `gt4pt025` as the official-val candidate and
reject `gt4pt050`. Before `v2_validbranch_neg05-3`, that selected candidate
could proceed to exactly one reporting-only final-test run using the frozen
official-val decode above. After `v2_validbranch_neg05-3`, the immediate next
action is the v2 extra-lane diagnostic and fine official-val sweep. Do not tune
final-test thresholds, NMS, `max_det`, `min_points`, checkpoint choice, or loss
gain from any final-test report.

## Legacy 2026-06-25 GT4 Short-Lane Recall Endpoint Rejection

The completed `gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint`
run enabled the default-off base lane-balanced point reduction and GT4-short
endpoint matcher cost:

```text
gcs_lane_balanced_point_loss = true
gcs_gt4_short_lane_weight = 1.5
gcs_gt4_short_lane_max_points = 20
gcs_gt4_short_match_endpoint = 1.0
gcs_gt4_short_match_max_points = 20
gcs_duplicate_margin = 0.0
```

Official-val result:

```text
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

Integrated conclusion:

- Supported fact: this run is below the previous `gt4pt025` official-val gate
  (`ACC=0.970975`, `count_acc_4=0.954545`).
- Supported fact: the selected decode removes `4->3` on official-val, but
  worsens GT4 overcount (`4->5=8` versus `gt4pt025`'s `2`) and raises FP.
- Supported fact: the fixed-pool official-val raw-query diagnostic still has
  one missing GT4 lane, classified as `geometry_bad`, with best raw-query
  mean absolute x error `30.265432px` against a `20px` gate.
- Decision: reject `gt4shortrecall_lbpt_endpoint` as a promotion and do not
  run final test for it.

Smallest safe next action:

Do not add score/count/rank losses for this failed run. If this line is
revisited, isolate matcher/label/query candidate geometry around
`clips/0601/1494453641541664519/20.jpg`, and compare against `gt4pt025` before
launching another training change.

## Legacy 2026-06-25 v2_validbranch_neg05-3 Extra-Lane Bottleneck

The `v2_validbranch_neg05-3` run moved the later official-val bottleneck away
from GT4 missing-lane recall and toward extra-lane over-count in that legacy experiment line. It exceeds the
previous `gt4pt025` ACC gate, but the margin is narrow and the count-shape
evidence is not robust enough to justify final-test evaluation yet.

Strict ACC-best evidence:

```text
decode = conf=0.01, point_valid_thr=0.45, nms_dist_px=30, max_det=5, min_points=2/3
official-val ACC = 0.971029
previous gate = 0.970975
```

Current error shape:

```text
GT4: 4->3 = 1, 4->5 = 11
GT3: 3->4 = 9, 3->5 = 3
pred_lanes_hist: 5-lane predictions are overrepresented
```

This is no longer primarily the old GT4 `4->3` recall failure. The main risk
is that extra decoded lanes survive into `GT3` and `GT4` images. Therefore the
next work must not increase valid recall, must not lower `point_valid_thr`,
must not enable valid count floor, and must not increase `max_det`.

Wider NMS is the key current diagnostic signal. `nms_dist_px=50` is almost tied
on official ACC at `0.971016`, only `0.000013` below strict best, while
reducing FP and improving `official_score` and `count_acc_4`. The supplied
extra-lane diagnostics also show fewer duplicate-like/spurious extra lanes
under wider NMS:

```text
nms30 selected: duplicate_like/spurious = 13/22
nms50:          duplicate_like/spurious = 4/22
conf015_nms60: duplicate_like/spurious = 4/18
```

The cleanest supplied stability row is:

```text
decode = conf=0.015, point_valid_thr=0.45, nms_dist_px=60, max_det=5, min_points=2/3
official-val ACC = 0.971016
official-val FP = 0.022498
official-val FN = 0.012167
official-val official_score = 0.970323
official-val count_acc_4 = 0.878788
count extras = 3->4=8, 3->5=1, 4->5=7
```

Interpretation:

- The strict ACC-best row is `nms=30`; do not relabel the wider-NMS row as
  the primary-metric best.
- The wider-NMS near tie is a serious risk-reduced contender because it cuts
  extra-lane evidence with almost no ACC loss.
- Duplicate/near-duplicate extras are likely part of the issue, but spurious
  extras remain because wider NMS does not remove all over-count.
- GT5 `5->4` must be watched before selecting an aggressive NMS row.

Smallest safe next action:

Freeze the current weights and complete official-val-only extra-lane
diagnostics for `GT3->4`, `GT3->5`, `GT4->5`, and `GT5->4`. For every extra
predicted lane, record score, valid point count, nearest-GT distance,
nearest-pred distance, and duplicate-like versus spurious classification. Then
run the fine decoder sweep:

```text
conf: 0.008, 0.010, 0.012, 0.015, 0.020
point_valid_thr: 0.43, 0.45, 0.47, 0.50
nms_dist_px: 30, 36, 42, 50, 60
max_det: 5
min_points: 2, 3
```

Select only on official-val. Report `official_acc`, `official_score`, `FP`,
`FN`, `count_acc_4`, `3->4`, `3->5`, and `4->5`. Keep final test closed until
one decode is selected by the official-val table and the extra-lane diagnosis
does not reveal a hidden GT5 under-count tradeoff.

## Legacy 2026-06-26 Q18 GT4-Hard Raw Geometry Check

The Q18 run
`gcs_yolo_lane_s_q18_k56_side_gt4endpoint_validneg_countce_v1` was checked
with raw-query diagnostics before looking at official ACC. The diagnostic used
the old `gt4pt025` train-derived GT4-hard set as a fixed geometry target:

```text
hard set = data/tusimple_gt4_hard_val_gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025.txt
records = 19 images
old missing denominator = 22 GT lanes
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
```

Q18 does not solve the GT4-hard geometry bottleneck:

```text
current-missing artifact:
runs/gcs_lane/gcs_yolo_lane_s_q18_k56_side_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/diagnostic_all_current_missing/summary.json

current missing lanes = 20
drop_reason = geometry_bad 12, low_point_valid 8
raw_match_recall = 8/20 = 0.400000
after_point_valid_recall = 0/20 = 0.000000

fixed old-missing recovery artifact:
runs/gcs_lane/gcs_yolo_lane_s_q18_k56_side_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/fixed_old_missing_recovery/summary.json

raw_match_recall = 9/22 = 0.409091
after_point_valid_recall = 1/22 = 0.045455
final_decode_recall = 2/22 = 0.090909
new_status = geometry_bad 12, low_point_valid 8, recovered_final_decode 2
```

中文结论：

Q18 side-dense reference failed the candidate-coverage gate.

Evidence:

- On current-missing diagnostic, `raw_match_recall = 8/20 = 0.400000`.
- `geometry_bad` remains dominant: `12/20` missing lanes.
- `after_point_valid_recall = 0/20`.
- On fixed old-missing denominator from `gt4pt025`, Q18
  `raw_match_recall = 9/22 = 0.409091`, identical to the old `gt4pt025`
  hard-set raw recall.
- Q18 therefore does not improve raw candidate coverage and stays below the
  required `>= 0.55` gate.

Decision:

- Stop tuning Q18 loss.
- Do not change `gcs_gt4_short_valid_pos_weight` to `1.25`.
- Do not change `unmatched_valid_neg_weight` to `0.5`.
- Next branch must change reference geometry to Q20, preferably with side-slope
  template diversity rather than only adding two more bottom references.

Interpretation:

- Q18 raw geometry recall stays at the previous `gt4pt025` level
  (`9/22 = 0.409091`) and misses the target starting gate `>= 0.55`.
- `geometry_bad` remains the largest residual failure bucket.
- The small valid/final recovery signal does not justify tuning
  `gcs_gt4_short_valid_pos_weight` or `gcs_unmatched_valid_neg_weight`, because
  that follow-up is only appropriate when raw geometry improves first.

Smallest safe next action:

Treat Q18 side references as insufficient for the real short side-lane
geometry. The next geometry-recall experiment should move to Q20 instead of
continuing Q18 loss tuning. Keep final test closed and do not lower
`point_valid_thr` from this evidence.

## Legacy 2026-06-26 Q20 GT4-Hard Raw Geometry Check

The Q20 run
`gcs_yolo_lane_s_q20_k56_sidegeom_gt4endpoint_validneg_countce_v1` was checked
with the same hard diagnostic protocol before looking at official ACC. It used
the old `gt4pt025` train-derived GT4-hard set and the same decode as the Q18
check:

```text
hard set = data/tusimple_gt4_hard_val_gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025.txt
records = 19 images
old missing denominator = 22 GT lanes from gt4pt025
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
match gate = overlap >= 3 h-samples and mean_abs_x_error <= 20px
```

Fixed old-missing recovery diagnostic:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_sidegeom_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/fixed_old_missing_recovery/summary.json
raw_match_recall = 12/22 = 0.545455
after_point_valid_recall = 2/22 = 0.090909
after_min_points_recall = 2/22 = 0.090909
after_conf_recall = 2/22 = 0.090909
final_decode_recall = 2/22 = 0.090909
new_status = geometry_bad 10, low_point_valid 10, recovered_final_decode 2
```

Current-missing diagnostic:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_sidegeom_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/diagnostic_all_current_missing/summary.json
current missing lanes = 20
drop_reason = geometry_bad 10, low_point_valid 10
raw_match_recall = 10/20 = 0.500000
after_point_valid_recall = 0/20 = 0.000000
final_decode_recall = 0/20 = 0.000000
```

Interpretation:

- Q20 reaches only the fixed old-missing starting line
  (`12/22 = 0.545455`) and misses the true qualification line
  (`>= 13/22`, `after_point_valid >= 4/22`, and `final_decode > 2/22`).
- On the current-missing denominator, Q20 misses the required target:
  `raw_match_recall = 0.500000 < 0.55`, `geometry_bad = 10 > 7`, and
  `after_point_valid_recall = 0.0`.
- Geometry is slightly better than Q18 (`geometry_bad` drops from `12` to
  `10` on both fixed-old and current-missing summaries), but the reduction is
  not large enough to claim that side-geometry references fixed candidate
  coverage.
- Lowering the diagnostic point-valid threshold only recovers one current
  missing lane at permissive thresholds, while baseline
  `point_valid_thr=0.5` remains `0/20`; this does not satisfy the condition
  for valid-loss tuning because raw geometry did not pass first.

中文记录：

Q20 sidegeom 部分改善了 raw candidate coverage，但没有通过 hard diagnostic
promotion gate。

证据：

- fixed old-missing `raw_match_recall` 从 Q18 的 `9/22` 提升到 Q20 的
  `12/22`。
- 但它只达到起步线 `12/22`，没有达到真正通过线 `13/22`。
- `after_point_valid` 只有 `2/22`，低于要求的 `4/22`。
- `final_decode` 仍然是 `2/22`，没有优于 Q18。
- current-missing `raw_match_recall` 是 `10/20 = 0.500000`，低于目标
  `0.55`。
- current `geometry_bad` 仍为 `10`，高于目标 `<=7`。
- current `after_point_valid` 是 `0/20`。

决定：

- 不调 Q20 sidegeom loss。
- 不把 `gcs_gt4_short_valid_pos_weight` 改成 `1.25`。
- 不把 `unmatched_valid_neg_weight` 改成 `0.5`。
- 下一分支：从真实 GT4 hard/missing lanes 生成 Q20 data-driven reference
  bank。

Decision:

- Reject Q20 as a GT4-hard geometry fix.
- Do not run the official-val normal/count-guided sweep for this Q20 run.
- Do not change `gcs_gt4_short_valid_pos_weight` to `1.25`.
- Do not change `gcs_unmatched_valid_neg_weight` to `0.5`.
- Do not tune loss, threshold, NMS, `max_det`, `min_points`, checkpoint choice,
  or final-test behavior from this diagnostic.

Smallest safe next action:

Move to data-driven reference clustering. Start with the `10`
`geometry_bad` current-missing lanes from the Q20 hard diagnostic, compare each
failed lane against the nearest raw queries and reference templates, and
identify which side/order/slope shapes are still uncovered. The next geometry
candidate should derive query references from those failed GT4-hard lane shapes
instead of only adding side-geometry query slots. Keep official-val and final
test closed until a new reference design passes the hard raw-geometry gate
first.

## Legacy 2026-06-26 Q20-Dataref v1 Hard-Gate Failure

The first dataref training attempt
`gcs_yolo_lane_s_q20_k56_dataref_gt4endpoint_validneg_countce_v1` did not pass
the stricter Q20-dataref hard diagnostic gate.

Checkpoint audit:

```text
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_sidegeom_gt4endpoint_validneg_countce_v1/weights/best.pt
reset_point_reference = false
sidegeom best.pt point_reference_logits keys = 0
dataref best.pt point_reference_logits keys = 0
fresh dataref model point_reference_logits = non-persistent buffer
```

The recorded `reset_point_reference=false` is not considered the cause of this
failure. The configured dataref reference bank is not overwritten by normal GCS
checkpoint loading when `point_reference_logits` is absent from the checkpoint
`state_dict()`.

Diagnostic protocol:

```text
hard set = data/tusimple_gt4_hard_val_gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025.txt
records = 19 images
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
match gate = overlap >= 3 h-samples and mean_abs_x_error <= 20px
```

Fixed old-missing recovery:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_dataref_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/fixed_old_missing_recovery/summary.json
raw_match_recall = 12/22 = 0.545455
after_point_valid_recall = 3/22 = 0.136364
final_decode_recall = 2/22 = 0.090909
new_status = geometry_bad 10, low_point_valid 10, recovered_final_decode 2
```

Current-missing diagnostic:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_dataref_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/diagnostic_all_current_missing/summary.json
current missing lanes = 20
drop_reason = geometry_bad 10, low_point_valid 10
raw_match_recall = 10/20 = 0.500000
after_point_valid_recall = 1/20 = 0.050000
final_decode_recall = 0/20 = 0.000000
```

Interpretation:

- Raw geometry did not improve over Q20 sidegeom: fixed old stays at `12/22`,
  current stays at `10/20`, and `geometry_bad` stays at `10`.
- Point-valid survival moved only slightly, from sidegeom `2/22` to dataref
  `3/22` on fixed old and from `0/20` to `1/20` on current missing.
- This is not the condition for valid-loss tuning. The intended valid-loss
  follow-up requires raw geometry to rise and `geometry_bad` to drop first.

Smallest safe next action:

Do not run official-val or final test for this attempt. Relaunch only a clean
Q20-dataref follow-up if it changes reference-bank coverage or otherwise has a
clear reason to improve raw geometry; do not move to valid-loss tuning from
this result.

中文结论：

Q20-dataref v1 没有通过 hard diagnostic promotion gate。fixed old-missing
denominator=22 上，`raw_match_recall=12/22=0.545455` 低于 `14/22`，
`after_point_valid=3/22` 低于 `4/22`，`final_decode=2/22` 没有高于之前的
`2/22`，`geometry_bad=10` 高于 `<=7`。current-missing 上，
`raw_match_recall=10/20=0.500000` 低于 `>=0.55`，`geometry_bad=10` 高于
`<=7`，虽然 `after_point_valid=1/20`，但 `final_decode=0/20`。

决定：不提升 Q20-dataref v1，不在这个分支上调 valid loss，不把
`gcs_gt4_short_valid_pos_weight` 改成 `1.25`，不把
`unmatched_valid_neg_weight` 改成 `0.5`。下一步诊断 dataref 失败是因为
reference bank 本身不对，还是因为训练/加载后预测偏离了 reference。
