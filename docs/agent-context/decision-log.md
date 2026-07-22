# Decision Log

## 2026-07-22: Add env30 GT4 near-20px geometry refine follow-up

Decision:

Implement a default-off env30 follow-up that targets the current train0601
GT4-short bottleneck without changing reference banks, decode, count losses, or
valid-negative pressure.

Implementation:

```text
ultralytics/utils/gcs_loss.py
tools/train_gcs.py
ultralytics/cfg/default.yaml
scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
scripts/run_query_alpha05_env30_gt4_near20_geom_refine_v1.sh
```

New default-off loss/config:

```text
gcs_gt4_near20_geom_refine = 0.0
gcs_gt4_near20_visible_thr = 10
gcs_gt4_near20_lower_px = 20.0
gcs_gt4_near20_upper_px = 25.0
gcs_gt4_near20_min_overlap = 3
gcs_gt4_near20_allowed_queries = "0,10"
gcs_gt4_near20_geom_curve = 0.0
```

Why:

The env30 train0601 GT4-short miss decomposition showed `10/10` misses already
have a final decoded lane, but all decoded x-APE values remain above 20px.
There was no clean threshold, max-det, count, point-valid, or static-reference
rescue. The useful subset is the `20..25px` near-miss region, especially
matched q0/q10 candidates that are not the dominant q5/q6/q7 GT5-lost20
carriers.

The new loss only applies on GT4 images, short GT lanes, Hungarian-matched
same-GT queries from the allowed-query list, and current fixed-y mean x-APE
inside the configured pixel window. It adds geometry only; it does not boost
existence or point-valid targets.

Next action:

Run the wrapper on the remote server and keep TEST closed:

```bash
bash scripts/run_query_alpha05_env30_gt4_near20_geom_refine_v1.sh
```

Before any reporting-only TEST, require official-val ACC/FP/FN at least as good
as env30, GT4/GT5 count-shape safety, train0601 GT4-short miss reduction, and
no GT5-short raw/point-valid regression on train0601/train0531.

## 2026-07-22: Complete env30 GT4/GT5 staticref v3 bank diagnostic

Decision:

Do not launch v3 full training from the current staticref v2 bank, and do not
launch the `w00` no-valid-negative companion as the immediate next full train.
Keep env30 as the active reference. The next allowed action is a no-training
q7-only static/pretrain counterfactual.

Artifacts:

```text
.tmp/env30_gt45staticref_v3_bank_diagnostic/summary.md
.tmp/env30_gt45staticref_v3_bank_diagnostic/short_gt45_summary.csv
.tmp/env30_gt45staticref_v3_bank_diagnostic/lane_transition_events.csv
.tmp/env30_gt45staticref_v3_bank_diagnostic/query_carrier_summary.csv
.tmp/env30_gt45staticref_v3_bank_diagnostic/focus_query_quality.csv
.tmp/env30_gt45staticref_v3_bank_diagnostic/candidate_v3_recommendation.json
```

Evidence:

```text
train0601 GT4 short hit20:
  env30 9/19, v2 11/19, w01 14/19
train0601 GT5 short hit20:
  env30 142/183, v2 135/183, w01 139/183

train0601 GT4 short q5/q6/q7 best-query share:
  env30 2/19, v2 11/19, w01 10/19
train0601 GT5 short q5/q6/q7 best-query share:
  env30 3/183, v2 56/183, w01 58/183

v2 GT4 short gained20:
  3 lanes, all train0601 center; v2 q7 is best for 2/3.
v2 GT5 short lost20:
  10 lanes, all train0601; v2 q5/q6/q7 is best for 6/10.

w01 GT4 short gained20:
  7 lanes, val+train0601; w01 q7 is best for 3/7.
w01 GT5 short lost20:
  8 lanes, val+train0601; w01 q5/q6/q7 is best for 6/8.
```

Why:

The diagnostic separates the previous aggregate tradeoff into carrier-level
events. q7 is the only edited query with meaningful GT4-short gain, while q5
and q6 add little direct GT4-short 20px coverage. But q7 is also the dominant
GT5-short lost20 carrier: protected q10/q11/q1/q8 lanes often hand off to
q7/q6, and the GT4-gain and GT5-loss pools overlap on center visible 9-10
lanes. This means the staticref v2 bank is not just suffering from too much
valid-negative pressure; the q5/q6/q7 carrier assignment itself is unstable.

Next action:

Build and screen a q7-only static/pretrain counterfactual before any training:

```text
q5/q6: default Q12 reference
q7: only changed query
q7 visible span: derived from GT4-short gained20 events
outside q7 span: exact default reference
taper: 2 anchors
boundary clamp: no contact with 0.04/0.96 extremes
selection: official-val + train0601/train0531 raw-Q12 only
TEST: closed
```

Reject the staticref route if this q7-only counterfactual cannot improve
GT4-short raw hit20 while keeping GT5-short hit20 and point-valid at or above
env30 and preventing protected-query handoff into q5/q6/q7.

## 2026-07-22: Add q7-only staticref v3a diagnostic implementation

Decision:

Implement the q7-only static reference counterfactual as default-off diagnostic
tooling only. Do not change env30 defaults, do not launch training, and keep
TEST closed.

Implementation:

```text
data/gcs_reference_banks/q12_env30_gt45_static_q7only_v3a.json
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-env30-gt45staticref-q7only-v3a.yaml
scripts/run_query_alpha05_env30_gt45staticref_q7only_v3a_raw_diag.sh
tools/diagnose_tusimple_raw_q12_filters.py --model-cfg
tools/infer_gcs.py::load_gcs_model_from_cfg_and_weights
```

Why:

The existing raw-Q12 diagnostic could only load the pickled checkpoint model or
a YAML model without checkpoint tensors. That is insufficient for static-bank
counterfactuals because the q7-only reference must come from a new YAML while
the learned weights must come from env30 `official_best.pt`. The new
`--model-cfg` path builds the YAML model first, loads checkpoint tensors into
it, and leaves the non-persistent fixed-y reference buffer defined by the YAML.

q7-only bank policy:

```text
q5/q6 = default Q12 reference
q7 = only changed query
prototype source = train0601 q7-assisted GT4-short gained20 events only
official-val usage = none for prototype generation; official-val is screening only
span = fixed-y desc indices 36..44
taper = 2 anchors each side
x clip = [0.06, 0.94], with no v2 0.96 boundary contact
```

Next action:

Run the no-training raw-Q12 diagnostic on the server and compare against env30:

```bash
bash scripts/run_query_alpha05_env30_gt45staticref_q7only_v3a_raw_diag.sh
```

Do not consider full training unless official-val, train0601, and train0531
raw gates pass together.

## 2026-07-22: Reject q7-only staticref v3a at raw-Q12 gate

Decision:

Reject `query_alpha05_env30_gt45staticref_q7only_v3a` before full training.
Do not run TEST and do not use this as the next training candidate.

Protocol:

```text
run type = no-training raw-Q12 counterfactual
weights = env30 official_best.pt
model_cfg = gcs-yolo-lane-s-env30-gt45staticref-q7only-v3a.yaml
splits = official-val, train0601, train0531
TEST = not used
```

Artifacts:

```text
runs/gcs_lane/query_alpha05_env30_gt45staticref_q7only_v3a_official_best_val_raw_q12_counterfactual
runs/gcs_lane/query_alpha05_env30_gt45staticref_q7only_v3a_official_best_train0601_raw_q12_counterfactual
runs/gcs_lane/query_alpha05_env30_gt45staticref_q7only_v3a_official_best_train0531_raw_q12_counterfactual
.tmp/env30_gt45staticref_q7only_v3a_analysis/summary.json
```

Evidence:

```text
overall raw has_match_20px:
  val       env30 0.975441, q7only 0.975441
  train0601 env30 0.960828, q7only 0.960828
  train0531 env30 0.983900, q7only 0.983900

train0601 GT4 short hit20:
  env30 9/19, q7only 9/19
train0601 GT5 short hit20:
  env30 142/183, q7only 142/183

GT4_short_gain20_q7only = 0
GT5_short_lost20_q7only = 1
```

Why:

The q7-only bank controlled the old q5/q6/q7 carrier explosion but did not
create useful GT4-short raw coverage. It is almost env30-neutral: q7 usage does
not grow into the dangerous GT5-short pool, but it also does not reproduce the
GT4-short gains seen after actual v2/w01 training. Therefore the bottleneck is
not solved by changing the static q7 reference prior alone.

Next action:

Do not run full training for q7-only v3a. A future attempt must first show a
raw-Q12 gain above env30 on GT4 short without GT5 short regression. If no
pretraining counterfactual can pass that gate, close the staticref route.

## 2026-07-20: Add constrained env30 GT4/GT5 staticref v2 candidate

Decision:

Implement the user-requested next env30-family candidate as default-off code
and config only. Do not run TEST or claim improvement until the candidate passes
official-val plus train0601/train0531 gates.

Implementation:

```text
data/gcs_reference_banks/q12_env30_gt45_static_v2.json
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-env30-gt45staticref-v2.yaml
scripts/run_query_alpha05_env30_gt45staticref_v2.sh
gcs_boundary_pseudo_valid_neg_weight default = 0.0
```

Why:

`gt5staticref_v1` improved some raw geometry but created extra carriers because
the short-lane prototypes were extrapolated into invisible anchors as hard
boundary lines. The v2 bank changes only q5/q6/q7, leaves q2/q8 and
q0/q1/q3/q10/q11 at the default Q12 reference, keeps prototype influence local
to visible spans with four-anchor taper, and clamps away from hard
`0.001`/`0.999` edges. The paired `gcs_boundary_pseudo_valid_neg_weight`
extension lets selected clear-far unmatched queries lose point-valid survival,
instead of only lowering their existence score.

Gate:

Use official-val for checkpoint/decode selection. Before any reporting-only
TEST, require no official-val FP/FN/count-shape regression versus env30, short
GT4/GT5 raw match and point-valid improvement, no q0/q1/q3/q10/q11 carrier
growth, and leave-date-out train0601/train0531 count-shape stability.

## 2026-07-21: Reject env30 GT4/GT5 staticref v2 after full protocol

Decision:

Reject `query_alpha05_env30_gt45staticref_v2` as a promotion candidate. Keep
`query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1` as the active env30
reference. Do not tune thresholds, valid-negative weights, reference-bank
queries, checkpoint choice, or TEST behavior from this result.

Primary official-val evidence:

```text
env30 official_best val ACC/FP/FN =
  0.973330 / 0.015748 / 0.009642
v2 official_best val ACC/FP/FN =
  0.971040 / 0.018825 / 0.012626
v2 best.pt val ACC/FP/FN =
  0.968341 / 0.026676 / 0.017677
```

The post-train fine sweep did not expose a threshold rescue:

```text
v2 official_best sweep rows = 2520
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows passing GT4/GT5 count-shape gate = 0

v2 best.pt sweep rows = 2520
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows passing GT4/GT5 count-shape gate = 0
```

Reporting-only TEST evidence is also below env30 and must not be used for
selection:

```text
env30 TEST ACC/FP/FN =
  0.966780 / 0.028732 / 0.023544
v2 official_best TEST ACC/FP/FN =
  0.966541 / 0.029601 / 0.022795
v2 best.pt TEST ACC/FP/FN =
  0.966150 / 0.033926 / 0.024832
```

What improved:

Compared with rejected `gt5staticref_v1`, v2 fixed the severe six-lane extra
carrier failure:

```text
val 5->6: 10 -> 0
test 5->6: 71 -> 0
test 4->5: 116 -> 83
test FP: 0.042535 -> 0.029601
```

Against env30, v2 improved reporting-only TEST GT4 retention:

```text
GT4 4->3: 98 -> 81
GT4 4->4: 280 -> 304
count_acc_4: 0.598291 -> 0.649573
```

What regressed:

The GT4 gain is offset by GT3 over-count and GT5 under-count:

```text
official-val GT4/GT5 shape:
  env30 4->5=0, 5->4=1
  v2    4->5=2, 5->4=2

TEST GT3 over-count:
  3->4: 50 -> 64

TEST GT5 retention:
  5->4: 43 -> 69
  count_acc_5: 0.891037 -> 0.857645
```

Post-hoc official-val raw-Q12 diagnostics confirm v2 did not improve overall
candidate geometry:

```text
raw has_match_20px: 0.975441 -> 0.972371
raw mean APE px:    5.627210 -> 6.053190
raw p90 APE px:     10.042735 -> 10.759785

GT4 all has_match_20px: 0.954545 -> 0.950758
GT5 all has_match_20px: 0.962162 -> 0.956757
GT5 short has_match_20px: 0.754717 -> 0.773585
GT5 short point_valid_recall@0.6: 0.957233 -> 0.943396
```

Post-hoc train-side raw-Q12 diagnostics close the remaining ambiguity:

```text
train0601 overall:
  has_match_20px: 0.960828 -> 0.950196
  mean APE px:    6.609740 -> 7.269312
  final under-count images: 5 -> 8

train0601 GT4 short:
  has_match_20px: 0.473684 -> 0.578947
  mean APE px:    23.666 -> 21.842
  point_valid_recall@0.6: 0.854386 -> 0.880702

train0601 GT5 short:
  has_match_20px: 0.775956 -> 0.737705
  mean APE px:    13.745 -> 15.007
  point_valid_recall@0.6: 0.937601 -> 0.908242

train0531 overall:
  has_match_20px: 0.983900 -> 0.980322
  mean APE px:    5.177374 -> 5.694983
  final over-count images: 7 -> 12
```

Train-side count shape confirms the same tradeoff:

```text
train0601:
  env30: 3->4=1, 4->5=7, 5->4=3
  v2:    3->4=3, 4->5=4, 5->4=6

train0531:
  env30: 3->4=4, 4->5=2
  v2:    3->4=6, 4->5=6
```

Query attribution shows the static v2 changes moved many GT5 short lanes onto
the edited q5/q6/q7 carriers but not cleanly:

```text
train0601 GT5 short q5/q6/q7 best-query share:
  env30 3/183 -> v2 56/183
train0601 GT5 short 20px misses:
  env30 41/183 -> v2 48/183
train0601 GT5 short lost20 events:
  10 lanes, often q10/q11 -> q7 handoff
```

Why:

The constrained v2 reference bank successfully removed the old v1
visible-outside extrapolation failure, but the paired q5/q6/q7 static changes
plus GT4/GT5 clear-far valid-negative pressure did not create a net official-val
geometry gain. The small GT5-short raw-match improvement is not clean because
mean/p90 APE and point-valid survival worsen, and the official count shape
shows the true-lane/extra-query separation remains unstable.

Smallest safe next action:

Run official-val plus train0601/train0531 raw-Q12 diagnostics for env30 and v2
before any new training. If the train-side diagnostics confirm that the damage
comes from GT5 valid-negative pressure rather than the static bank itself, the
only justified ablation is a narrow official-val-only run that keeps the v2
static bank but scopes valid-negative pressure to GT4 only
(`gcs_boundary_pseudo_gt_count=4`,
`gcs_boundary_pseudo_max_gt_count=4`) or lowers
`gcs_boundary_pseudo_valid_neg_weight` to `0.1/0.0`. Reject that ablation
unless official-val ACC/FP/FN, GT4/GT5 count shape, raw GT4/GT5 match, and
train0601/train0531 robustness all improve together.

## 2026-07-21: Add GT4-only valid-negative follow-up wrapper

Decision:

Add a separate wrapper for the narrow v2 follow-up instead of changing the
rejected v2 script defaults.

Implementation:

```text
script =
  scripts/run_query_alpha05_env30_gt45staticref_v2_gt4only_validneg_w01.sh
base wrapper =
  scripts/run_query_alpha05_env30_gt45staticref_v2.sh
model =
  ultralytics/cfg/models/gcs/gcs-yolo-lane-s-env30-gt45staticref-v2.yaml
default run =
  query_alpha05_env30_gt45staticref_v2_gt4only_validneg_w01
```

Default settings:

```text
gcs_boundary_pseudo_gt_count = 4
gcs_boundary_pseudo_max_gt_count = 4
gcs_boundary_pseudo_valid_neg_weight = 0.1
RUN_TESTS = 0
OFFICIAL_INTERVAL = 5
```

The no-valid-negative companion is the same wrapper with:

```bash
BOUNDARY_PSEUDO_VALID_NEG_WEIGHT=0.0
RUN_NAME=query_alpha05_env30_gt45staticref_v2_gt4only_validneg_w00
```

Why:

The train0601/train0531 diagnostics show v2 improved GT4-short coverage while
hurting GT5-short coverage and GT3/GT4 over-count robustness. Scoping
boundary-pseudo valid-negative pressure to GT4 is the smallest follow-up that
tests whether the GT5 damage is caused by broad valid-negative pressure. This
is not a promotion and should keep TEST closed until official-val plus
train0601/train0531 gates pass.

## 2026-07-22: Reject GT4-only valid-negative w01 follow-up

Decision:

Reject `query_alpha05_env30_gt45staticref_v2_gt4only_validneg_w01` as a
promotion candidate. Do not run TEST and do not immediately launch the `w00`
no-valid-negative companion. Keep `env30` as the active reference.

Primary official-val evidence:

```text
env30 official_best val ACC/FP/FN =
  0.973330 / 0.015748 / 0.009642
v2 official_best val ACC/FP/FN =
  0.971040 / 0.018825 / 0.012626
w01 official_best val ACC/FP/FN =
  0.970965 / 0.024656 / 0.012626
w01 best.pt val ACC/FP/FN =
  0.968859 / 0.033425 / 0.013545
```

Post-train sweep gate:

```text
w01 official_best sweep rows = 2520
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows passing GT4/GT5 count-shape gate = 0

w01 best.pt sweep rows = 2520
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows passing GT4/GT5 count-shape gate = 0
```

User-requested reporting-only TEST was run after this rejection. It confirms
the candidate is below env30 and must not be used for tuning:

```text
env30 reporting-only TEST ACC/FP/FN =
  0.966780 / 0.028732 / 0.023544
w01 official_best TEST ACC/FP/FN =
  0.966061 / 0.037233 / 0.024712
w01 best.pt TEST ACC/FP/FN =
  0.965752 / 0.041175 / 0.024053

w01 official_best TEST count_acc_3/4/5 =
  0.963218 / 0.617521 / 0.706503
w01 official_best TEST count_confusion includes:
  4->5 = 75, 4->6 = 15, 5->4 = 50, 5->6 = 99
```

The official ACC-best row uses `max_det=6` and is not count-shape-safe:

```text
decode = conf 0.001, point_valid_thr 0.6, nms 0, max_det 6, min_points 4
w01 val count_confusion =
  3->3 217, 3->4 6,
  4->3 1, 4->4 64, 4->6 1,
  5->4 2, 5->5 57, 5->6 15
```

The best `max_det=5` row removes the GT5 false-sixth explosion but still fails
env30 ACC/FP/FN and count shape:

```text
w01 max_det=5 best val ACC/FP/FN =
  0.970928 / 0.017493 / 0.012626
count_confusion =
  3->3 217, 3->4 6,
  4->3 1, 4->4 64, 4->5 1,
  5->4 2, 5->5 72
```

Raw-Q12 diagnostics:

```text
official-val overall has_match_20px:
  env30 0.975441, v2 0.972371, w01 0.976209
official-val overall mean APE:
  env30 5.627210, v2 6.053190, w01 6.021849

train0601 overall has_match_20px:
  env30 0.960828, v2 0.950196, w01 0.956911
train0601 overall mean APE:
  env30 6.609740, v2 7.269312, w01 7.122588

train0531 overall has_match_20px:
  env30 0.983900, v2 0.980322, w01 0.982111
train0531 overall mean APE:
  env30 5.177374, v2 5.694983, w01 5.520983
```

Short-lane diagnostics:

```text
train0601 GT4 short has_match_20px:
  env30 0.473684, v2 0.578947, w01 0.736842
train0601 GT5 short has_match_20px:
  env30 0.775956, v2 0.737705, w01 0.759563
train0601 GT5 short point_valid_recall@0.6:
  env30 0.937601, v2 0.908242, w01 0.935816
```

Train-side count shape:

```text
train0601:
  env30: 3->4=1, 4->5=7, 5->4=3
  v2:    3->4=3, 4->5=4, 5->4=6
  w01:   3->4=1, 4->5=11, 5->4=4

train0531:
  env30: 3->4=4, 4->5=2
  v2:    3->4=6, 4->5=6
  w01:   3->4=6, 3->5=1, 4->5=3
```

Why:

`w01` proves that reducing/scoping valid-negative pressure partially restores
GT5-short valid survival and strongly improves GT4-short raw coverage, but it
does not solve the official target. The same q5/q6/q7 static bank remains a
large GT5-short carrier (`train0601 q5/q6/q7 share 58/183`) while GT5-short
20px misses stay worse than env30 (`41/183 -> 44/183`). At the count level,
the improvement flips back into GT4 over-count (`train0601 4->5: 7 -> 11`) and
official-val FP remains too high.

Smallest safe next action:

Do not spend TEST and do not launch `w00` as the immediate next training run.
The current evidence says the blocker is not simply "valid-neg too strong";
the v2 static reference bank itself creates unstable q5/q6/q7 GT5/GT4 carrier
assignment. The next useful step should be static-bank analysis or a new bank
that changes the q5/q6/q7 GT5-short carrier behavior before another full
training run. A candidate should be screened by raw-Q12 official-val plus
train0601/train0531 before TEST.

## 2026-07-17: Execute env30 short GT4/GT5 gate and reject before training

Decision:

Complete the user-requested env30 short GT4/GT5 robustness-gate execution and
stop before remote training. Keep `env30` as the active reference. Do not run
TEST, do not launch formal training, and do not tune decode, loss, sampler,
checkpoint, reference-bank, or model-structure settings from this result.

Artifacts:

```text
.tmp/env30_scheme_execution_20260717/gate_analysis.md
.tmp/env30_scheme_execution_20260717/gate_analysis.json
.tmp/env30_scheme_execution_20260717/summary.md
```

Result:

```text
TEST used = no
remote training launched = no
reason = static_gate_failed
static candidates evaluated = 6226
gate-pass candidates = 0
allowed replacement queries = q2,q5,q6,q7,q8
changed known-extra overlap in saved top rows = empty
```

The short GT4/GT5 gate includes official-val plus train0313/train0601/
train0531 diagnostics. Train0313 supplies a large GT4 short-lane surface, but
the training decision is still blocked by GT5 coverage on official-val and
train0601:

```text
best static row = gt5cl1_gt5r1_gt4c2_gt4r1
val GT5 impact p50/p90/match20 = 43.579 / 47.756 / 0.2500
val GT5 primary p50/p90/match20 = 51.595 / 97.903 / 0.0625
train0601 GT5 primary p50/p90/match20 = 49.012 / 92.705 / 0.0833
```

Top-50 gate failures:

```text
val_gt5_primary_p90<=85: 0/50
val_gt5_primary_match20>=0.3125: 0/50
train0601_gt5_primary_p90<=90: 0/50
train0601_gt5_primary_match20>=0.2333: 5/50
val_gt5_impact_match20>0.25: 0/50
changed_known_extra_overlap_empty: 50/50
```

Integrated conclusion:

This confirms the bottleneck is not a missing count-loss switch or decode
threshold. The constrained safe-query reference space avoids known extra
carriers but lacks enough GT5 raw-geometry coverage. The candidate family is
rejected before training.

## 2026-07-17: Roll active code back to env30 commit 86c8fb31c

Decision:

Use commit `86c8fb31c` (`Add GT4 GT5 weak geometry rescue run`) as the active
source/config boundary. The worktree code/config/scripts outside documentation
is restored to that commit. All commits and experiment lines after
`86c8fb31c` are rejected legacy records unless a future task explicitly
reopens them with a new official-val/train gate.

Active env30-family scripts at this boundary:

```text
scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh
scripts/run_query_alpha05_gt4gt5weak_geom_w15w2_env30_nocount_v1.sh
```

Post-86 changes now marked legacy/rejected:

```text
544aa142c Add tiered GT4 GT5 rescue protocol
dac2f6141 Add tiered rescue full protocol v2
6d602d699 Add Q12 ultrashort dataref v2 extra gate
post-86 router/selector/final-query extra diagnostics recorded in docs/.tmp
```

Why:

The later tiered, ultrashort dataref, final-query guard/gate, and non-oracle
router/selector experiments did not produce a promotable official-val/train
result over env30. They either failed official-val ACC/FP/FN, failed GT4/GT5
count-shape gates, had too little non-oracle coverage, or remained diagnostic
only. They must not be used for TEST tuning or as the next active baseline.

Validation target:

Non-document tracked content should match `86c8fb31c`; documentation should
preserve later evidence only as rejected legacy records.

## 2026-07-17: Reject count-change non-oracle selector after oracle and leave-date-out scan

Decision:

Close the current count-change selector line as a promotion path. Keep `env30`
as the active reference. Do not run TEST and do not tune count-risk thresholds,
aux features, classifier depth, or switch caps from this diagnostic.

Protocol:

```text
TEST used = no
candidate outputs = env30, GT4/GT5 weak, tiered-v2
default output = env30
offline GT use = label count-fix/count-risk rows only
selector features = prediction-derived base features, optional aux mask/edge
validation = train0601/train0531 leave-date-out before canonical official-val
artifacts = .tmp/count_change_signal_experiment/
selector rows = 3840
train-gate pass rows = 96
allow reporting-only TEST rows = 0
meaningful val gain rows = 0
```

Offline oracle:

```text
val all_count_fix oracle:
  ACC/FP/FN = 0.973384 / 0.014371 / 0.009642
  gain = +0.000055
  switches = 2
  GT4/GT5 target fixes = 0

val gt45_count_fix oracle:
  ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
  gain = +0.000000
  switches = 0

train0601 gt45_count_fix oracle:
  gain = +0.000054
  switches = 3
  fixes = 2 GT4 4->5, 1 GT5 5->4

train0531 gt45_count_fix oracle:
  gain = +0.000037
  switches = 2
  fixes = 2 GT4 4->5
```

Best selector rows:

```text
best train-gate row:
  base gt45_count_fix tree_d1_balanced
  benefit>=0.1 risk<=0.02 cap=0.02
  val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
  val switches = 0

best overall val row:
  base all_count_fix tree_d2_balanced
  benefit>=0.1 risk<=0.02 cap=0.02
  val ACC/FP/FN = 0.973496 / 0.017585 / 0.009642
  val switches = 7
  val count fixes = 2 all-count, 0 GT4/GT5
  failed train gate and val gate:
    train0531->train0601 ACC below env30, under_images 6 > 5,
    GT5 5->4 4 > 3
    val FP 0.017585 > 0.015748, over_images 10 > 7
```

Why:

The offline oracle answers the core question: the three frozen model outputs do
not contain enough GT4/GT5 count-changing repair capacity on canonical
official-val. The only val count-fix oracle gain comes from two GT3 `3->4` to
`3->3` fixes, worth just `+0.000055`; the GT4/GT5 target oracle has zero val
switches. Train0601/train0531 contain a few GT4/GT5 fixes, but they are only
2-3 images and are surrounded by much larger count-risk pools.

Integrated conclusion:

Reject this selector as a route to `official_test_acc >= 0.9700`. The
bottleneck is not just absence of a stable prediction-only separator; the
available three-model candidate pool has insufficient GT4/GT5 count-fix
headroom. Future work should not continue by tuning this router. It needs a
new candidate model/output that creates more true GT4/GT5 repair opportunities
before non-oracle selection can matter.

## 2026-07-17: High-confidence abstention switch is eligible only as a weak reporting candidate

Decision:

Keep `env30` as the active reference. The new high-confidence abstention switch
is the first non-empty three-model router variant that passes the current
train-side leave-date-out and canonical official-val gates, but it is not a
promotion because the official-val gain is too small to materially change the
0.9700 TEST objective.

Protocol:

```text
TEST used = no
default output = env30
alternate outputs = GT4/GT5 weak, tiered-v2
router action = choose one complete model output per image
feature surfaces = base prediction features, aux mask/edge features
scan rows = 44928
train gate pass rows = 10864
nontrivial train gate pass rows = 2162
allow reporting-only TEST rows = 342
artifacts = .tmp/model_router_abstention_experiment/
```

Selected reporting candidate:

```text
feature_set = base
target_margin = 0.01
config = binary:logreg_c0.5_balanced
prob_thr = 0.9  # 0.6..0.95 produced the same 7 val switches
gap_thr = 0.0
max_switch_frac = 0.02

env30 val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
switch val ACC/FP/FN = 0.973551 / 0.015748 / 0.009642
val gain = +0.000221
val switches = 7 / 363
val count shape = unchanged versus env30
  GT4 4->5 = 0
  GT5 5->4 = 1
  GT5 5->6 = 0
```

Train-side leave-date-out gate:

```text
train0601->train0531 ACC/FP/FN =
  0.974694 / 0.010894 / 0.007682
switches = 7
count-shape gate = pass

train0531->train0601 ACC/FP/FN =
  0.978154 / 0.014390 / 0.004675
switches = 8
count-shape gate = pass
```

Why:

The passing switch fixes only seven official-val line-accuracy micro-errors.
All seven selected images keep the same lane count and contribute zero FP/FN
delta; the whole val gain is `0.080357 / 363 = 0.000221`. Higher-val aux rows
reach up to `0.974443` on canonical val, but they fail the train-side and val
count-shape gates with GT4 false-fifth and GT5 false-sixth risks.

Integrated conclusion:

This candidate may be used for exactly one frozen reporting-only TEST run if
the user decides to spend the TEST evaluation, but it should not be described
as a solution for the `official_test_acc >= 0.9700` objective. Do not use TEST
to choose another abstention threshold, aux row, cap, or classifier.

## 2026-07-17: User-override TEST confirms whole-image router is not promotable

Decision:

Keep the 2026-07-16 whole-image router rejection. The user explicitly
overrode the closed TEST gate and requested one TEST run. Treat this result as
reporting-only evidence, not as a selector, threshold search, or promotion
gate.

Protocol:

```text
TEST run = yes, explicit user override
candidate outputs = env30, GT4/GT5 weak, tiered-v2
router = pair_logreg_c0.1_balanced
router training targets = train0601+train0531 per-image official-like winners
test GT use = final reporting metrics only
test images = 2782
artifacts = .tmp/model_router_experiment/test_override_*
```

Results:

```text
env30 TEST ACC/FP/FN = 0.966788 / 0.028732 / 0.023544
router TEST ACC/FP/FN = 0.966452 / 0.027265 / 0.025521
router selections = env30 963, GT4/GT5 weak 869, tiered-v2 950

env30 count shape:
  over/under = 151 / 163
  GT4 4->5 = 90
  GT5 5->4 = 43
  GT5 5->6 = 0

router count shape:
  over/under = 137 / 199
  GT4 4->5 = 69
  GT4 4->6 = 3
  GT5 5->4 = 71
  GT5 5->6 = 23
```

Why:

The router lowers FP and reduces some GT3/GT4 over-count cases, but the cost is
higher FN, lower official ACC, more under-count images, worse GT5 `5->4`, and
new GT5 `5->6` errors. This matches the failed official-val/train-side gate:
the router shifts count shape instead of producing a stable improvement.

Integrated conclusion:

The TEST result confirms the router should stay rejected. Do not use this TEST
run to tune thresholds, choose a different router row, or reopen reporting-only
TEST. Future work must return to train/official-val diagnostics only.

## 2026-07-16: Reject three-model whole-image router before TEST

Decision:

Reject the current whole-image model router before TEST. Keep `env30` as the
active reference. Do not promote per-image routing across `env30`,
`GT4/GT5 weak`, and `tiered-v2` unless both train-side leave-date-out gates
and canonical official-val pass.

Protocol:

```text
candidate outputs = env30, GT4/GT5 weak, tiered-v2
router action = choose one complete model output per image
forbidden operations = no lane-level averaging, no lane stitching
train targets = per-image official-like score from train0601/train0531 GT only
validation = train0601->train0531 and train0531->train0601 leave-date-out
TEST used = false
artifacts = .tmp/model_router_experiment/
```

Results:

```text
selected router = pair_logreg_c0.1_balanced
env30 official-val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
router official-val ACC/FP/FN = 0.973087 / 0.014555 / 0.009412
router official-val selections = env30 120, GT4/GT5 weak 140, tiered-v2 103

official-val gate issues =
  ACC=0.973087 < env30 0.973330
  GT4 4->5 = 2, required 0
  GT5 5->6 = 2, required 0

train0601->train0531 issue =
  GT4 4->5 = 3 > env30 2

train0531->train0601 issues =
  over_images = 22 > env30 8
  GT4 4->5 = 9 > env30 7
```

The frozen prediction caches did not include `aux_mask_logits` or
`aux_edge_logits`, so no synthetic mask/edge consistency feature was used.
A GT-only whole-model oracle remains diagnostic-only: it reaches higher
official-val ACC but still produces GT4 false-fifth and GT5 false-sixth
count-shape errors, so the available checkpoint complementarity is not enough
to satisfy the promotion gate.

Integrated conclusion:

This router is an invalid promotion candidate, not a test candidate. Do not run
reporting-only TEST for it. Reopen this path only with a new non-oracle,
inference-available, split-stable feature set that can separate train0601 GT4
false-fifth risk and official-val GT5 false-sixth risk without degrading
`env30` FP/FN or count-shape gates.

## 2026-07-16: Reject fixed three-checkpoint geometry consensus

Decision:

Reject the fixed geometry-consensus ensemble before TEST. Keep `env30` as the
active reference. Do not average final-lane geometry or use three-model median
lane count as a promotable decode rule.

Protocol:

```text
models = env30, GT4/GT5 weak, tiered GT4/GT5 + GT4PV v2
surfaces = canonical official-val 363, train0601 410, train0531 358
rules = geometry median with one/two supporting models; median model count
distance = 10/15/20/25/30 px
minimum shared samples = 4/6/8/10
non-baseline combinations = 100
TEST used = false
artifacts = .tmp/geometry_consensus_experiment/
```

Results:

```text
env30 official-val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
best fixed consensus ACC/FP/FN = 0.973330 / 0.014371 / 0.009642
best rule = median_count, distance 25-30 px, overlap 4-6
official-val changes = remove two GT3 false-fourth lanes
official-val ACC gain = 0.000000
official-val combinations with ACC above env30 = 0

best geometry-median official-val ACC = 0.970923
geometry-median ACC delta = -0.002407
geometry-median improved/worsened/unchanged images = 71 / 128 / 164
```

The count-only rule lowers FP but does not improve official ACC. On train0601
it also removes one GT5 lane (`5->5` to `5->4`) without changing official ACC,
showing that count correctness and official ACC are not interchangeable under
the TuSimple GT5 rule. Geometry averaging changes too many already-good lanes
and regresses official-val and train0601.

A GT-only diagnostic selected the best complete model output per image:

```text
official-val model-choice oracle ACC/FP/FN = 0.977181 / 0.016529 / 0.008035
train0601 oracle ACC/FP/FN = 0.981929 / 0.013780 / 0.003049
train0531 oracle ACC/FP/FN = 0.979541 / 0.008892 / 0.005121
```

This oracle is diagnostic-only and cannot be used in decode or model
selection. It proves that the checkpoints have real image-level
complementarity, but the tested fixed geometric/count rules cannot identify the
best model without GT. Reopen only with a train-only, leave-date-out validated
non-oracle model router that selects one complete model output per image and
beats env30 on canonical official-val without FP/FN or train-split regression.


This file records decisions for branch `codex/5-25-3-k56`.

## 2026-07-16: Close env30 raw-GT lane context gate as a training trigger

Decision:

Keep the env30 raw-GT lane context gate as diagnostic-only. Do not start
formal training, run TEST, tune decode thresholds/NMS/`max_det`, adjust
geometry/validity/positive-rescue loss gains, change sampling, or edit
reference banks from this diagnostic.

Artifact:

```text
summary =
  .tmp/env30_raw_gt_lane_context_gate/summary.md
script =
  .tmp/goal_9700_audit/diagnose_env30_raw_gt_lane_context_gate.py
feature table =
  .tmp/env30_raw_gt_lane_context_gate/raw_gt_lane_context_features.csv
rule table =
  .tmp/env30_raw_gt_lane_context_gate/raw_gt_lane_context_rules.csv
```

Evidence:

```text
scope = env30 official-val, train0601, train0531 raw GT diagnostics
TEST used = no
GT-lane rows = 4208

target counts:
  raw_no20 = 120
  valid_below_min = 46
  decoded_bad20 = 127
  residual_image_lane_failure = 47
  count_correct_residual_lane_failure = 30
  fn_like_lane_failure = 19

best raw/decoded geometry context:
  visible_le5
  raw_no20 precision / recall = 0.818 / 0.150
  decoded_bad20 precision / recall = 0.818 / 0.142
  selected lanes = 22
  selected lanes by split =
    train0531 0
    train0601 14
    official-val 8

official-impact mapping:
  residual_image_lane_failure best context = visible_le5
  precision / recall = 0.364 / 0.170
  count_correct_residual_lane_failure best context = visible_le5
  precision / recall = 0.273 / 0.200
  fn_like_lane_failure best context = GT4 visible_le20
  precision / recall = 0.029 / 0.474
```

Why:

- Ultra-short visible lanes are a real raw/decoded failure carrier, but the
  useful high-precision subset is tiny and absent from train0531.
- The same context does not isolate official-like residual image failures;
  most selected lanes are not residual-impact lanes.
- Broader lane-context rules such as visible<=10/20, side lanes, GT4, or GT5
  improve recall only by selecting hundreds of already-ok lanes.
- This does not provide a split-stable positive-rescue denominator or anchor/
  geometry/validity target with the required `>=90%` engineering confidence.

Next action:

Keep positive-rescue and simple lane-context weighting closed. Reopen only if a
new diagnostic supplies a train-available denominator that links raw/valid/
decoded lane failures to official-like residuals across official-val,
train0601, and train0531 without mostly selecting already-ok lanes.

## 2026-07-16: Close env30 prediction-only residual separator path

Decision:

Close direct image-level prediction-only residual separators as a training,
decode, sampler, loss, or reference-bank trigger for the active 0.9700
official-TEST goal. Do not start formal training, run TEST, tune decode
thresholds/NMS/`max_det`, adjust loss gains, change sampling, add
query-suppression logic, or edit reference banks from this diagnostic.

Artifact:

```text
summary =
  .tmp/env30_residual_prediction_separator/summary.md
script =
  .tmp/goal_9700_audit/diagnose_env30_residual_prediction_separator.py
feature table =
  .tmp/env30_residual_prediction_separator/image_prediction_features.csv
rule table =
  .tmp/env30_residual_prediction_separator/image_prediction_rule_scan.csv
```

Leakage check:

The first scan exposed a feature-selection bug: diagnostic columns such as
`image_official_*`, `final_extra_rows`, GT-lane failure counts, and missed-
Hungarian counts were initially eligible as candidate features. The final run
explicitly excludes those columns, plus GT counts and target labels, so the
reported rules use only full 12-query prediction-cache features and final
decoded lane prediction features.

Evidence:

```text
scope = env30 official-val, train0601, train0531
TEST used = no
image rows = 1131

target counts:
  official_like_residual = 99
  count_correct_residual = 70
  over_image = 22
  under_image = 9
  low_acc_lt095 = 62
  has_fp = 63
  has_fn = 25

best broad official_like_residual prediction-only rule with >=10 positives:
  final_score_min <= 0.378976
  precision / recall = 0.765 / 0.131

best over_image rule with >=5 positives:
  final_nearest_pred_dist_min <= 92.75
  precision / recall = 0.556 / 0.227

best under_image rule with >=5 positives:
  score_std <= 0.396334
  precision / recall = 0.238 / 0.556

best count_correct_residual rule with >=5 positives:
  valid_count06_top5 >= 16
  precision / recall = 0.625 / 0.071
  selected only train0601 rows
```

Why:

- After removing official/GT-after-decode diagnostic columns, no prediction-
  only rule reaches the `>=90%` engineering-confidence reopen gate.
- High-precision rules for broad residuals select only tiny tails, often
  `5-8` images, and have low recall.
- Over-image and under-image targets remain poorly separated by score-sum
  margin, final-score, final-validity, final-lane spacing, and query-
  composition features.
- The only clean-looking rules in the leaked first pass were direct
  diagnostic labels, not valid non-oracle selectors.
- This therefore closes direct image-level confidence/margin/separation as a
  current implementation direction.

Next action:

Keep env30 as the reference. Continue only with read-only official-val/train
diagnostics unless a new non-oracle, train-available, split-stable selector or
target-margin artifact separates residual rows from clean/background rows with
useful precision and recall across official-val, train0601, and train0531.

## 2026-07-16: Keep env30 official-like residual decomposition diagnostic-only

Decision:

Use the env30 official-like residual decomposition as attribution evidence for
the active 0.9700 official-TEST goal, but do not start training, run TEST,
tune decode thresholds/NMS/`max_det`, adjust loss gains, change sampling, add
positive-rescue or hard-negative logic, or edit reference banks from this
diagnostic.

Artifact:

```text
summary =
  .tmp/env30_official_like_residual_decomp/summary.md
script =
  .tmp/goal_9700_audit/diagnose_env30_official_like_residuals.py
image residual rows =
  .tmp/env30_official_like_residual_decomp/official_like_per_image_residuals.csv
lane failure rows =
  .tmp/env30_official_like_residual_decomp/official_like_lane_match_failures.csv
group summary =
  .tmp/env30_official_like_residual_decomp/official_like_group_summary.csv
lane bucket summary =
  .tmp/env30_official_like_residual_decomp/official_like_lane_bucket_summary.csv
```

Evidence:

```text
scope = env30 official-val, train0601, train0531
TEST used = no
image residual rows = 1131
lane/failure rows = 655
final-extra rows = 23
count-correct official-like residual images = 70
  val = 23
  train0601 = 27
  train0531 = 20
GT-side failure-like lanes on residual images = 181
under-image missed Hungarian-positive rows = 7
```

Split-level details:

```text
official-val:
  over/under = 7 / 3
  count-correct residual images = 23
  final extra rows = 7, all far
  residual-image lane buckets include low_final_line_acc<0.95 33 rows,
    raw_geometry_no20_or_ape_gt20 14 rows, no_final_gt_match 5 rows
  non-residual background has low_final_line_acc<0.95 96 rows

train0601:
  over/under = 8 / 5
  count-correct residual images = 27
  final extra rows = 8, all far
  residual-image lane buckets include low_final_line_acc<0.95 39 rows,
    raw_geometry_no20_or_ape_gt20 16 rows, no_final_gt_match 10 rows
  non-residual background has low_final_line_acc<0.95 169 rows

train0531:
  over/under = 7 / 1
  count-correct residual images = 20
  final extra rows = 8, far 5 + near-nonduplicate 2 + duplicate-like 1
  residual-image lane buckets include low_final_line_acc<0.95 43 rows,
    raw_geometry_no20_or_ape_gt20 8 rows, no_final_gt_match 4 rows
  non-residual background has low_final_line_acc<0.95 66 rows
```

Why:

- The diagnostic is an official-like attribution surface, not an official
  metric replacement.
- The clean-looking final-extra rows are sparse and split-specific. They do
  not provide a broad hard-negative or decode target.
- The broad GT-side signals are noisy. In particular,
  `low_final_line_acc<0.95` appears on residual images but also heavily in
  non-residual count-correct background rows, so a direct geometry,
  line-accuracy, or validity loss would hit many already acceptable lanes.
- Under-count evidence remains tiny: `7` under-image missed Hungarian-positive
  rows across q5/q6/q10/q11 from the prior count-confusion decomposition.
- The labels that look most discriminative (`official_like_residual`,
  final-extra class, final matching state) are GT/post-decode diagnostic
  attribution, not train-available inference selectors.
- A read-only `gcs_experiment_analyst` concluded with confidence `0.93` that
  this should remain diagnostic-only, and a read-only `gcs_integrator`
  concluded with confidence `0.94` that it should not trigger training, TEST,
  decode tuning, loss-gain changes, or reference edits.

Next action:

Keep env30 as the reference. Reopen implementation or training only with a
new non-oracle, train-available, split-stable selector or explicit
target-margin artifact that separates residual rows from the large background
pool across official-val, train0601, and train0531 while covering GT3/GT4
over-count and GT5 `5->4`/`5->6` risk.

## 2026-07-16: Keep env30 train-side count-confusion decomposition diagnostic-only

Decision:

Use the env30 train-side count-confusion decomposition as supporting evidence
for the existing robustness gate, but do not start training, run TEST, tune
decode thresholds/NMS/`max_det`, adjust loss gains, change sampling, or edit
reference banks from this diagnostic.

Artifact:

```text
summary =
  .tmp/env30_train_count_confusion_decomp/summary.md
script =
  .tmp/goal_9700_audit/diagnose_env30_train_count_confusion.py
image rows =
  .tmp/env30_train_count_confusion_decomp/image_count_confusion_rows.csv
image label summary =
  .tmp/env30_train_count_confusion_decomp/image_count_confusion_by_label.csv
final extra rows =
  .tmp/env30_train_count_confusion_decomp/final_extra_query_rows.csv
missed Hungarian-positive rows =
  .tmp/env30_train_count_confusion_decomp/missed_hungarian_positive_rows.csv
under missed Hungarian-positive rows =
  .tmp/env30_train_count_confusion_decomp/under_missed_hungarian_rows.csv
```

Evidence:

```text
scope = env30 official-val, train0601, train0531
TEST used = no
image rows = 1131
query rows = 13572
final extra rows = 23
missed Hungarian-positive rows = 18
under-image missed Hungarian-positive rows = 7

official-val:
  over/under = 7 / 3
  final extra = 7, all far, q0 4 + q11 3
  transition = 3->4 7
  under missed Hungarian positives = 3, q6/q10/q11, transition 4->3

train0601:
  over/under = 8 / 5
  final extra = 8, all far, q1 4 + q10 3 + q11 1
  transitions = 3->4 1, 4->5 7
  under missed Hungarian positives = 2, q5/q6, transition 4->3

train0531:
  over/under = 7 / 1
  final extra = 8, q0/q1/q9/q10/q11, far 5, duplicate 1
  transitions = 2->3 1, 3->4 5, 4->5 2
  under missed Hungarian positives = 2, q10/q11, transition 4->3
```

Why:

- The decomposition confirms the existing train-side robustness gate with a
  row-level query audit: official-val over-count, train0601 GT4 false fifth,
  train0531 mixed over-count, and GT4 under-count misses are different sparse
  surfaces.
- The extra-lane carriers shift by split (`q0/q11` on official-val versus
  `q1/q10/q11` on train0601 and mixed q0/q1/q9/q10/q11 on train0531).
- The under-count side is a small matched-positive survival set, not a broad
  count-only class.
- No non-oracle selector or target-margin artifact separates these rows from
  already-correct positives across official-val/train splits with the required
  `>=90%` engineering confidence.

Next action:

Keep this diagnostic as a gate/checklist input only. Reopen implementation or
training only with a new non-oracle separator or explicit train-side
target-margin artifact that covers the split-specific over-count and GT4
under-count surfaces without mostly selecting matched or near-ok positives.

## 2026-07-16: Close cross-feature composite separator path

Decision:

Close the current cross-feature composite separator as a direct implementation
or training trigger for the active 0.9700 official-TEST goal. Do not start
formal training, run TEST, tune decode thresholds/NMS/`max_det`, adjust loss
gains, change sampling, or edit reference banks from this evidence.

Artifact:

```text
cross-feature separator audit =
  .tmp/env30_cross_feature_separator/summary.md
feature table =
  .tmp/env30_cross_feature_separator/cross_feature_rows.csv
rule table =
  .tmp/env30_cross_feature_separator/cross_feature_rules.csv
script =
  .tmp/goal_9700_audit/diagnose_cross_feature_separator.py
raw-query gate audit =
  .tmp/env30_raw_query_cross_feature_gate/summary.md
raw-query gate script =
  .tmp/goal_9700_audit/diagnose_raw_query_cross_feature_gate.py
read-only gate agent conclusion =
  close as diagnostic-only with confidence 0.93; direct trainability
  confidence below 0.60
```

Evidence:

```text
scope = env30/v2 official-val, train0601, and train0531 final-kept rows
TEST used = no
rows = 8530
harm/safe/ambig = 113 / 8364 / 53

top rule =
  valid_mean <= 0.2 AND label_min_visible >= 15
selected = 23 final-kept rows over 16 unique raw_file values
harm/safe/ambig = 23 / 0 / 0
precision / recall = 1.000 / 0.204
dataset surfaces with harm hits = 6
selected row class =
  pool_greedy_matched_gt=false and nearest_gt_far50=true
selected GT counts =
  GT3 8, GT4 15, GT5 0
selected transitions =
  3->4 8, 4->4 1, 4->5 10, 4->6 4
GT5 5->4 / 5->6 coverage = none

broader GT5-reaching rules:
  precision drops to about 0.50-0.67 and includes safe/ambiguous hits
```

Follow-up raw-query gate:

```text
scope = env30 official-val, train0601, and train0531 full 12-query traces
TEST used = no
rows / unique images = 13572 / 1131
final_far_extra / final_extra rows = 20 / 23
hungarian_positive / final_matched_positive rows = 4199 / 4155

exact valid_mean rule replay = from cached pred_valid_logits

exact replay of final-kept top rule:
  rule = valid_mean_exact <= 0.2 AND label_min_visible >= 15
  selected = 5373
  final_far_extra / final_extra = 9 / 9
  hungarian_positive / final_matched_positive / near_or_good = 3 / 0 / 0
  nonfinal_dead = 5360
  final_extra_transition_hist = {"3->4": 3, "4->5": 6}

best available raw-query relative by final-far-extra precision:
  rule = query_id == 1 AND label_min_visible >= 15
  selected = 609
  final_far_extra / final_extra = 4 / 4
  nonfinal_dead = 605
  final_extra_transition_hist = {"4->5": 4}

low-valid label-context relative:
  rule = valid_len <= 10 AND label_min_visible >= 15
  selected = 5370
  final_far_extra / final_extra = 6 / 6
  hungarian_positive = 3
  nonfinal_dead = 5360
  final_extra_transition_hist = {"3->4": 2, "4->5": 4}
```

Why:

- The strongest rule is high precision only on the post-decode `final_kept`
  proxy table. It is a clean far-extra attribution after decode, not a
  selector over the full raw-query pool.
- `label_min_visible` is label-only context: it is training-available but
  invalid for inference/decode.
- `harm` and `safe` are GT-derived post-decode diagnostic labels, not official
  FP/FN and not valid threshold, checkpoint, or decode-selection targets.
- The rule does not cover the GT5 `5->4`/`5->6` surfaces needed for the current
  promotion gate.
- The raw-query follow-up confirms the available train-side relatives mostly
  select non-final dead queries and cover too little of the final-extra surface.
  The exact `valid_mean` replay also collides with Hungarian positives and has
  no GT5 `5->4`/`5->6` coverage. There is no demonstrated target-margin
  artifact showing that a loss, sampler, reference edit, or query suppression
  would improve ACC/FP/FN or count shape.

Next action:

Keep TEST and training closed. Reopen only with a materially different
official-val/train diagnostic, not this final-kept composite family, that
proves a non-oracle selector has near-zero matched/Hungarian-positive hits,
covers GT3/GT4 over-count and GT5 `5->4`/`5->6` risk, and reaches the
`>=90%` engineering-confidence gate.

## 2026-07-16: Close simple session-context separator path

Decision:

Close simple session/date/window metadata separation as a direct implementation
or training trigger for the active 0.9700 official-TEST goal. Do not start
formal training, run TEST, tune decode thresholds/NMS/`max_det`, adjust loss
gains, change sampling or augmentation, or edit reference banks from this
evidence.

Artifact:

```text
session-context separator audit =
  .tmp/env30_session_context_separator/summary.md
feature table =
  .tmp/env30_session_context_separator/session_context_features.csv
rule table =
  .tmp/env30_session_context_separator/session_context_rules.csv
split summary =
  .tmp/env30_session_context_separator/session_context_split_summary.csv
script =
  .tmp/goal_9700_audit/diagnose_session_context_separator.py
overlap audit =
  .tmp/env30_session_context_overlap/summary.md
overlap table =
  .tmp/env30_session_context_overlap/session_context_overlap_rules.csv
read-only gate agent conclusion =
  close as candidate direction with confidence 0.94; direct
  implementation/training confidence from current evidence <0.30
```

Evidence:

```text
scope = env30 official-val/train0601/train0531
TEST used = no

split over/under/count_mismatch/official_error/low_acc_lt095:
  val = 7 / 3 / 10 / 21 / 23 over 363 images
  train0601 = 8 / 5 / 13 / 28 / 17 over 410 images
  train0531 = 7 / 1 / 8 / 15 / 22 over 358 images

best low_acc_lt095 rule:
  train0531 10-image local session window
  selected = 10, pos/neg = 5 / 5
  precision / recall = 0.500000 / 0.080645

best official_error rule:
  val:0313-2 10-image local session window
  selected = 10, pos/neg = 4 / 6
  precision / recall = 0.400000 / 0.062500

best over_image and count_mismatch precision = 0.200000
best under_image precision = 0.100000

follow-up overlap:
  top low_acc_lt095 window official-impact tail images/lanes = 1 / 1
  top low_acc_lt095 window label harm/safe totals = 2 / 76
  top low_acc_lt095 window prediction harm/safe totals = 2 / 76
  max official-impact tail image overlap = 3 images in a val:0313-2 window
    with prediction harm/safe totals = 8 / 68
```

Why:

- The diagnostic uses only raw-file date/session metadata as train-available
  selectors; offline official/count labels are diagnostic targets only.
- The strongest rules are tiny local clusters in a single split, not
  split-stable selectors across official-val, train0601, and train0531.
- The rules do not identify a mechanism for GT4 false fifth, GT5
  `5->6`/`5->4`, FP/FN, valid-point survival, NMS, ranking, or reference
  coverage.
- The follow-up overlap audit confirms the local windows do not line up with
  a near-zero-safe prediction-shape, label-context, or official-impact-tail
  mechanism; the same windows include many safe/matched rows.
- The evidence is far below the split-stable `>=90%` engineering-confidence
  gate for a sampler, augmentation, loss, decode, or reference-bank change.

Next action:

Keep TEST and training closed. Reopen only if a non-metadata, non-oracle
mechanism inside these local windows proves high precision with near-zero safe
hits across official-val, train0601, and train0531.

## 2026-07-16: Close simple image-appearance separator path

Decision:

Close simple image-appearance separation as a direct implementation or training
trigger for the active 0.9700 official-TEST goal. Do not start formal training,
run TEST, tune decode thresholds/NMS/`max_det`, adjust loss gains, change
sampling or augmentation, or edit reference banks from this evidence.

Artifact:

```text
image-appearance separator audit =
  .tmp/env30_image_appearance_separator/summary.md
feature table =
  .tmp/env30_image_appearance_separator/image_appearance_features.csv
rule table =
  .tmp/env30_image_appearance_separator/image_appearance_rules.csv
split summary =
  .tmp/env30_image_appearance_separator/image_appearance_split_summary.csv
script =
  .tmp/goal_9700_audit/diagnose_image_appearance_separator.py
```

Evidence:

```text
scope = env30 official-val/train0601/train0531
TEST used = no
images analyzed = 1131
missing images = 0

split over/under/low_acc_lt095:
  val = 7 / 3 / 23 over 363 images
  train0601 = 8 / 5 / 17 over 410 images
  train0531 = 7 / 1 / 22 over 358 images

best over_image rule:
  value_mean >= 116.475967
  selected = 114, pos/neg = 6 / 108
  precision / recall = 0.052632 / 0.272727

best count_mismatch rule:
  sat_mean >= 0.188590
  selected = 114, pos/neg = 7 / 107
  precision / recall = 0.061404 / 0.225806

best official_error rule:
  sat_mean >= 0.188590
  selected = 114, pos/neg = 11 / 103
  precision / recall = 0.096491 / 0.171875

best unique official-val tail-image rule:
  gray_std <= 43.683938
  selected = 37, pos/neg = 5 / 32
  precision / recall = 0.135135 / 0.416667
```

Why:

- The diagnostic uses only input-pixel appearance features as selectors:
  brightness, saturation, value statistics, top/middle/bottom regional means,
  side/center contrast, dark/bright fractions, gradient density, and Laplacian
  blur variance.
- Offline official/count labels are used only as read-only diagnostic targets;
  no TEST data or GT-at-inference rule is involved.
- Simple global visual thresholds are far below the split-stable `>=90%`
  engineering-confidence gate and mostly select count-correct or low-risk
  images.
- The result does not identify a safe sampler, augmentation, loss, decode, or
  reference-bank change.

Next action:

Keep TEST and training closed. Reopen only with a materially stronger
non-oracle visual signal that proves high precision across official-val,
train0601, and train0531 without selecting mostly count-correct images.

## 2026-07-16: Remote state recheck does not reopen candidates

Decision:

Keep the active 0.9700 official-TEST goal open, but do not start training,
run TEST, tune decode thresholds/NMS/`max_det`, change checkpoint selection,
adjust loss gains, or edit reference banks from the current remote state.

Artifact:

```text
remote state recheck =
  .tmp/goal_9700_audit/remote_state_recheck_20260716.md
current status report =
  .tmp/goal_9700_audit/current_goal_status_20260716.md
```

Evidence:

```text
remote parsed official summary rows = 939
split classification = val 877, test 59, unknown 3
latest remote summary =
  runs/gcs_lane/query_alpha05_env30_q12_ultrashort_q1q3protected_dataref_geom1_extraguard_v1/official_sweeps/epoch015/tusimple_official_sweep_summary.json
latest remote summary ACC/FP/FN =
  0.949375 / 0.113682 / 0.047980
official-val rows with ACC >= env30 = 45
gate-like official-val rows = 4
best reporting-only TEST ACC/FP/FN =
  0.966962 / 0.030188 / 0.023994
env30 reporting-only TEST ACC/FP/FN =
  0.966780 / 0.028732 / 0.023544
```

Why:

- The remote row count matches the local remote index audit.
- The latest remote summary belongs to the already rejected q1/q3-protected
  diagnostic and is far below the env30 official-val gate.
- The gate-like official-val rows are still only the two env30 rows plus two
  legacy farspur intermediate epoch rows already closed by the selection-surface
  audit.
- No reporting-only TEST row reaches the required `official_test_acc >= 0.9700`,
  and TEST remains non-selectable evidence.

Next action:

Continue only with a new read-only official-val/train diagnostic if it can
produce a split-stable, train-available non-oracle separator or target-margin
artifact that does not mostly select matched/near-ok positives. Do not treat
the remote recheck as permission for implementation or formal training.

## 2026-07-16: Close current positive-side reopen path

Decision:

Close the current positive-side reopen path as a direct implementation or
training trigger for the active 0.9700 official-TEST goal. Do not start formal
training, run TEST, tune decode thresholds/NMS/`max_det`, adjust loss gains,
change sampling, add positive-rescue or hard-negative/query-suppression logic,
or edit reference banks from this evidence.

Artifact:

```text
positive-side reopen audit =
  .tmp/goal_9700_audit/positive_side_reopen_audit.md
positive denominator =
  .tmp/env30_positive_denominator/summary.md
positive selector counterfactual =
  .tmp/env30_positive_selector_counterfactual/summary.md
assignment suppression counterfactual =
  .tmp/env30_assignment_counterfactual/summary.md
short GT4/GT5 hardset coverage =
  .tmp/env30_short_gt45_hardset_coverage/summary.md
read-only gate agent conclusion =
  positive-side artifacts do not reopen implementation/training; confidence
  0.93 for closure and below 0.90 for any positive-side change
```

Evidence:

```text
positive denominator:
  val GT lanes = 1303, geom>20 = 26, line_acc<0.8 = 12,
    no Hungarian = 5, no final match = 17
  train0601 GT lanes = 1787, geom>20 = 57, line_acc<0.8 = 14,
    no Hungarian = 4, no final match = 26
  train0531 GT lanes = 1118, geom>20 = 11, line_acc<0.8 = 6,
    no Hungarian = 0, no final match = 10

positive selector examples:
  geom_gt20 covers 12/15 val official-impact tails but selects 26 val lanes,
    including 14 non-impact readouts, 15 final-matched positives, and
    21 Hungarian positives
  no_final_match covers 12/15 tails but depends on post-decode/final matching
  geom20_or_line_and_no_final covers 12/15 tails but is also a final-match
    diagnostic gate rather than a direct training label

assignment suppression examples:
  selected_q0q11 val removes 6/7 final extras with precision 0.750,
    but touches 2 final positives
  selected_q0q1q10q11 train0601 removes 8/8 extras with precision 0.571,
    but touches 1 final positive and 4 GT5 5->5 lanes
  querywide_q0q11 val removes 7/7 extras but touches 490 Hungarian positives,
    489 final positives, and 6/15 official-impact tail raw-best rows

short GT4/GT5 hardset:
  val GT4 visible<=20 = 106 lanes, hard_fail = 10, official-impact tails = 4
  val GT5 visible<=20 = 207 lanes, hard_fail = 17, official-impact tails = 5
  train0601 GT5 visible<=20 = 671 lanes, hard_fail = 62
  train0531 has no comparable GT5 short denominator in this diagnostic
```

Why:

- Positive-tail official-impact rows are sparse and mixed.
- Broad train-available selectors cover tails but also cover many non-impact or
  already-matched positives.
- Compact selectors often depend on final/decode attribution, which is not a
  clean training label.
- Assignment suppression rules remove sparse extras only by touching non-extra
  final lanes, GT5 `5->5` rows, or many Hungarian/final positives.
- The short GT4/GT5 hardset denominator exposes a real raw-geometry/validity
  bottleneck, but it lacks a concrete reference/query counterfactual that
  improves raw `has20`/APE without broadening known extra carriers.

Next action:

Keep TEST and training closed. Reopen only with a new official-val/train
diagnostic that supplies a train-available denominator not mostly already
matched/near-ok/non-impact, a near-zero-safe suppression selector, or a
reference/query counterfactual that improves short GT4/GT5 hardset coverage
without changing known extra-carrier queries and remains stable across
official-val, train0601, and train0531.

## 2026-07-16: Close simple label-context separator path

Decision:

Close simple label-context separation as a direct implementation or training
trigger for the active 0.9700 official-TEST goal. Do not start formal
training, run TEST, tune decode thresholds/NMS/`max_det`, adjust loss gains,
edit references, change sampling, or add label-context hard-negative logic
from this evidence.

Artifact:

```text
label-context separator audit =
  .tmp/goal_9700_audit/label_context_separator_audit.md
image feature table =
  .tmp/goal_9700_audit/label_context_image_features.csv
image rule table =
  .tmp/goal_9700_audit/label_context_image_rules.csv
row rule table =
  .tmp/goal_9700_audit/label_context_row_rules.csv
read-only gate agent conclusions =
  label context does not reopen implementation/training; confidence 0.93-0.94
  for closure and below 0.90 for any label-context change
```

Evidence:

```text
model-run image rows = 2262
unique raw_file values = 914
has_harm model-run image rows = 107
over model-run image rows = 130
under model-run image rows = 18

best supported image-level has_harm rule:
  label_gt_count == 4 AND label_min_visible <= 5
  selected = 10
  pos / neg = 6 / 4
  precision / recall = 0.600 / 0.056

best supported image-level over rule:
  label_gt_count == 5 AND label_short_le10 >= 2
  selected = 70
  pos / neg = 30 / 40
  precision / recall = 0.429 / 0.231

best supported row-level rule:
  label_gt_count == 5 AND query_id == 3 AND valid_len <= 5
  selected = 13
  proxy harm / safe / ambiguous = 11 / 2 / 0
  precision / recall = 0.846 / 0.097

GT5 q6 valid_len<=5:
  proxy harm / safe / ambiguous = 5 / 1 / 0
  precision / recall = 0.833 / 0.044

GT4 q10 valid_len<=8:
  proxy harm / safe / ambiguous = 5 / 0 / 2
  precision / recall = 0.714 / 0.044

zero-safe and zero-ambiguous row rules:
  maximum proxy harm selected = 3
  maximum recall = 0.0265
```

Why:

- The diagnostic uses official-val/train-side rows only; TEST is absent.
- Counts are model-run/split rows, not independent unique images.
- `harm` and `safe` are offline proxy labels inherited from prediction-layout
  diagnostics, not direct TuSimple official FP/FN measurements.
- Image-level label context is too broad and often model-specific; it cannot
  be used at inference and does not identify a decode/reference change.
- Row-level label/query/valid composites are more concentrated but either hit
  safe or ambiguous rows, or cover too few proxy errors to justify a sampler or
  loss under the split-stable `>=90%` engineering-confidence gate.
- Sparse `precision=1.0` rules exist, but they select only `1-3` harmed rows
  and are not actionable.

Next action:

Keep TEST and training closed. Reopen only with a training-available selector
that shows near-zero safe close/matched hits, materially higher recall on both
GT4 false-fifth and GT5 sixth-lane surfaces, and stability across
official-val, train0601, and train0531.

## 2026-07-16: Close simple GT-count/query context separator path

Decision:

Close simple GT-count/query context separation as a direct implementation or
training trigger for the active 0.9700 official-TEST goal. Do not start formal
training, run TEST, tune decode thresholds/NMS/`max_det`, adjust loss gains,
edit references, or add GT-count/query-conditioned hard-negative logic from
this evidence.

Artifact:

```text
GT-count/query context separator audit =
  .tmp/goal_9700_audit/gtquery_context_separator_audit.md
rule table =
  .tmp/goal_9700_audit/gtquery_context_separator_rules.csv
```

Evidence:

```text
final-kept rows = 8530
far-unmatched harm-proxy rows = 113
safe close-or-matched rows = 8364
ambiguous rows = 53

GT5 q3:
  selected = 73
  proxy harm / safe / ambiguous = 19 / 47 / 7
  precision / recall = 0.260 / 0.168

GT5 q3 valid_len<=5:
  selected = 13
  proxy harm / safe / ambiguous = 11 / 2 / 0
  precision / recall = 0.846 / 0.097

GT4 q1:
  selected = 23
  proxy harm / safe / ambiguous = 9 / 14 / 0
  precision / recall = 0.391 / 0.080

GT4 q10 valid_len<=8 AND nearest_overlap_dist>=0.05:
  selected = 5
  proxy harm / safe / ambiguous = 5 / 0 / 0
  precision / recall = 1.000 / 0.044
```

Why:

- GT-count/query context is a training-time signal only; it cannot be used in
  inference or decode.
- Query-wide or GT-count/query-wide pressure would hit many safe close/matched
  lanes.
- Low-valid composites improve precision but remain too sparse and still hit
  safe rows on the main GT5 q3 surface.
- The few perfect-proxy rules select too few rows to explain or fix the
  current GT4/GT5 count-shape gap and do not satisfy the split-stable
  `>=90%` engineering-confidence gate for a training loss.

Next action:

Keep TEST closed. Reopen only with a training-available selector that shows
near-zero safe close/matched hits, materially higher recall on both GT4
false-fifth and GT5 sixth-lane surfaces, and stability across official-val,
train0601, and train0531.

## 2026-07-16: Close simple prediction-layout separator path

Decision:

Close simple prediction-layout separation as a direct implementation or
training trigger for the active 0.9700 official-TEST goal. Do not start formal
training, run TEST, tune decode thresholds/NMS/`max_det`, adjust loss gains,
edit references, or add hard-negative/query-suppression logic from this
evidence.

Artifact:

```text
prediction-layout separator audit =
  .tmp/goal_9700_audit/pred_layout_separator_audit.md
feature table =
  .tmp/goal_9700_audit/pred_layout_separator_features.csv
rule table =
  .tmp/goal_9700_audit/pred_layout_separator_rules.csv
read-only gate agent conclusion =
  layout separator does not reopen implementation/training; confidence 0.92
  for closure and below 0.90 for any layout-based change
```

Evidence:

```text
final-kept rows = 8530
far-unmatched harm-proxy rows = 113
safe close-or-matched rows = 8364
ambiguous rows = 53

nearest_overlap_dist >= 0.08:
  selected = 8289
  proxy harm / safe / ambiguous = 64 / 8214 / 11
  precision / recall = 0.008 / 0.566

layout_is_edge == True:
  selected = 4524
  proxy harm / safe / ambiguous = 36 / 4473 / 15
  precision / recall = 0.008 / 0.319

valid_len <= 5 AND nearest_overlap_dist >= 0.1:
  selected = 10
  proxy harm / safe / ambiguous = 8 / 2 / 0
  precision / recall = 0.800 / 0.071

valid_len <= 5 AND nearest_overlap_dist >= 0.08:
  selected = 13
  proxy harm / safe / ambiguous = 10 / 3 / 0
  precision / recall = 0.769 / 0.088
```

Why:

- Relative predicted-lane geometry adds no clean separator beyond the already
  rejected per-lane score/valid/raw-shape features.
- Pure layout signals such as predicted edge/interior position, bottom spacing,
  nearest overlap distance, and layout width have extremely low precision when
  used broadly.
- Sparse composites with low valid length improve precision but have very low
  recall and still hit safe close/matched lanes.
- The useful-looking sparse rows are not enough to explain or fix the current
  GT4/GT5 count-shape gap, and they do not satisfy the split-stable
  `>=90%` engineering-confidence gate for training.

Next action:

Keep TEST closed. Reopen only with a genuinely new non-oracle,
training-available feature beyond per-lane score/valid/raw-shape and
predicted-lane layout, with near-zero safe close/matched hits and stability
across official-val, train0601, and train0531.

## 2026-07-16: Close simple prediction-shape separator path

Decision:

Close the current pure prediction-shape separator direction as a direct
implementation or training trigger for the active 0.9700 official-TEST goal.
Do not start formal training, run TEST, tune decode thresholds/NMS/`max_det`,
adjust loss gains, edit references, or add hard-negative/query-suppression
logic from this evidence.

Artifact:

```text
prediction-shape separator audit =
  .tmp/goal_9700_audit/pred_shape_separator_audit.md
feature table =
  .tmp/goal_9700_audit/pred_shape_separator_features.csv
rule table =
  .tmp/goal_9700_audit/pred_shape_separator_rules.csv
read-only gate agent conclusion =
  current pure prediction-shape separator is not worth continuing; confidence
  0.92 for closure and below 0.90 for any implementation/training change
```

Evidence:

```text
final-kept rows = 8530
far-unmatched harm-proxy rows = 113
safe close-or-matched rows = 8364
ambiguous rows = 53

valid_len <= 5:
  selected = 51
  proxy harm / safe / ambiguous = 34 / 12 / 5
  precision / recall = 0.667 / 0.301

best sparse rules with at least five proxy rows:
  max precision = 0.833
  recall range = 0.044-0.088
  still touches safe close/matched lanes

broader score/point-valid rules:
  recall can reach about 0.69-0.73 only around 0.30 precision
  safe close/matched rows dominate the selected population
```

Why:

- GT is used only after decode for offline labeling; far-unmatched proxy rows
  are not all guaranteed official harms because some are count-correct.
- The strongest low-valid-count and raw-shape rules are too sparse to explain
  the remaining count-shape gap and still touch safe close/matched lanes.
- Broader score, point-valid mean/count, span, x-range, and raw prediction
  geometry proxies select too many safe rows across env30/v2 official-val,
  train0601, and train0531.
- This reinforces the previous pre-maxdet cache margin diagnostic: the current
  evidence does not support a decode cap, score/valid filter, hard-negative
  loss, query suppression, or reference-bank edit.

Next action:

Keep TEST closed. Reopen only with a genuinely new non-oracle,
training-available separator that proves split-stable high precision, near-zero
safe close/matched hits, meaningful GT4 false-fifth and GT5 sixth-lane recall,
and no GT5 matched-lane removal across official-val, train0601, and train0531.

## 2026-07-16: Consolidate current 0.9700 goal status

Decision:

Keep the 0.9700 official-TEST goal active, but do not start training, TEST,
decode search, or an implementation change from the current evidence. The
latest integrated scan found no high-value unclosed diagnostic direction that
reaches the `>=90%` engineering-confidence gate for a trainable candidate.
Env30 remains the active reference.

Artifact:

```text
current goal status report =
  .tmp/goal_9700_audit/current_goal_status_20260716.md
remote official summary index audit =
  .tmp/goal_9700_audit/remote_official_summary_index_audit.md
official-val gate family audit =
  .tmp/goal_9700_audit/official_val_gate_family_audit.md
non-oracle final-lane rule scan =
  .tmp/goal_9700_audit/non_oracle_final_lane_rule_scan.md
non-oracle count counterfactual =
  .tmp/goal_9700_audit/non_oracle_count_counterfactual.md
non-oracle split-stability counterfactual =
  .tmp/goal_9700_audit/non_oracle_count_counterfactual_split_stability.md
pre-maxdet cache margin diagnostic =
  .tmp/goal_9700_audit/premax_cache_margin_diagnostic.md
selection-surface reopen audit =
  .tmp/goal_9700_audit/selection_surface_reopen_audit.md
train-side robustness gate audit =
  .tmp/goal_9700_audit/train_side_robustness_gate_audit.md
score-margin target audit =
  .tmp/goal_9700_audit/score_margin_target_audit.md
tiered-v2 tail-reopen audit =
  .tmp/goal_9700_audit/tiered_v2_tail_reopen_audit.md
read-only gate agent conclusion =
  no open high-value next diagnostic; current minimum useful deliverable is
  an integrated candidate map / reopen-gate report
```

Current reference:

```text
env30 official-val ACC/FP/FN =
  0.973330 / 0.015748 / 0.009642
env30 official-val count_acc_3/4/5 =
  0.968610 / 0.969697 / 0.986486
env30 GT4 4->5 = 0
env30 GT5 5->4/5->6 = 1 / 0

env30 reporting-only TEST ACC/FP/FN =
  0.966780 / 0.028732 / 0.023544
```

Closed or diagnostic-only directions:

```text
ultrashort dataref family:
  v1 manual1, geom1 full-protocol v1, v2, and q1/q3-protected variants
env30 count-restore
tiered GT4/GT5 + GT4PV v2
expanded final-extra guard scope
q0/q11/q1/q10 hard-negative or query suppression
positive-tail rescue / geometry / line-accuracy selectors
q3 matched-vs-extra suppression/static-removal
short GT4/GT5 reference-bank static designs including q2-protected and
  five-slot q2/q5/q6/q7/q8-only search
```

Why:

- The active env30 reference still misses the final TEST target:
  reporting-only `official_test_acc=0.966780 < 0.9700`.
- The highest listed reporting-only TEST result in the current goal status
  report is `0.966962`, also below target and non-selectable.
- A remote read-only scan found old/legacy official-val rows above env30, but
  no reporting-only TEST row reaches `0.9700`; those high-val rows do not
  reopen a candidate because they are legacy/rejected and TEST remains closed
  for selection.
- A follow-up official-val-only gate-family audit found `45 / 880` val rows
  with ACC at or above env30, but only four rows also met the current
  FP/FN/count-shape gate (`4->5=0`, `5->4<=1`, `5->6=0`). Two are env30; the
  two non-env30 rows are legacy farspur training-time epoch rows. The farspur
  official-best/external selected rows use higher official-val ACC but fail
  `5->4<=1`, and selecting a lower-ACC epoch row would be a policy change
  against the current `official_best_v4` / ACC-first sweep selection surface.
- A non-oracle final-lane/count counterfactual found that the best count-only
  rule is `rank>=5`, equivalent to removing the sixth score-ordered lane under
  `max_det=6`. It reduces v2 over-count in copied official-val/train0601
  traces, but it removes true matched lanes as well as extras and corresponds
  to a `max_det=5`-style decode restriction already rejected by the v2
  official-val sweep. Low-score/short-valid rules catch extras but create
  under-count.
- The split-stability extension confirms `rank>=5` is only active on v2
  six-lane outputs in these traces and does not affect env30 because env30 is
  already capped at five lanes. On v2 it still removes true GT5 matched lanes
  (`6` on official-val, `20` on train0601), so it is not a trainable
  matched-vs-extra separator.
- A fresh `official_best.pt` pre-maxdet cache diagnostic closes the remaining
  cache-margin shortcut on official-val and train-side traces. Env30
  official-val has `1307` post-valid premax rows, `1307` final-kept rows, and
  no `rank_eq5`, `rank_ge5`, or beyond-`max_det` candidates, so the selected
  `max_det=5` decode is not hiding an extra official-val candidate pool. V2
  official-val has `14` sixth-lane rows, but they mix far/near and matched
  cases (`8` nearest-GT far, `2` close20, `6` pool-greedy matched). The
  train-side extension does not rescue this separator: env30 train0601 already
  has `5` `rank>=5` rows, and v2 train0601 has `58` `rank>=5` rows with `22`
  pool-greedy matched rows; v2 train0531 has only `1` far `rank>=5` row.
  Low-score/short-valid rows still touch matched or final-kept candidates.
  Fresh v2 caches differ from older copied traces on `2` official-val, `7`
  train0601, and `1` train0531 images, so these are margin diagnostics only,
  not replacement official summaries.
- A focused selection-surface reopen audit confirms that the only non-env30
  gate-passing official-val rows are legacy farspur intermediate epochs. They
  are not a current candidate because the selected farspur reporting-only TEST
  rows are below target, the mechanism is legacy outside the active rollback
  scope, and choosing those lower-epoch rows would change checkpoint selection
  policy rather than reuse a protocol-selected candidate.
- A train-side robustness gate audit turns split-stability into explicit
  future-candidate gates. Env30 official-val extras are GT3-only (`3->4=7`,
  q0/q11), but env30 train0601 already has GT4 false-fifth extras (`4->5=7`,
  q1/q10/q11). Future candidates must not pass only the 363-image
  official-val row while worsening train0601/train0531 count shape.
- A score-margin target audit found that v2 train0601 GT5 `5->6` has a real
  low-margin pathology, but the signal is not safe enough for a new loss:
  env30 count-correct GT5 `5->5` rows also have low/negative far margins, and
  v2 train0601 GT4 `4->5` is not a low-margin class. Direct score-margin or
  ranking remains diagnostic-only.
- A tiered-v2 tail-reopen audit consolidates the direct tail-retention
  rejection: six raw/valid tail fixes are real, but only one has the correct
  final count under tiered-v2, and train0601 q0/q10 rows mix selected
  clear-far extras with protected, near, or Hungarian-matched rows.
- The best-looking follow-ups either fail official-val ACC/FP/FN, fail
  GT4/GT5 count-shape gates, or only improve a diagnostic proxy while
  introducing over-count/under-count regressions.
- Remaining reopen paths require a new non-oracle separator or explicit
  train-side target-margin artifact; continuing to slice the same current
  `.tmp` fields would mostly repeat already closed conclusions.

Next action:

Keep TEST closed. Do not train, tune thresholds/NMS/`max_det`, adjust loss
gains, or edit reference banks unless a future task provides new evidence or a
materially different diagnostic surface that can pass the reopen gates in
`.tmp/goal_9700_audit/current_goal_status_20260716.md`.

## 2026-07-16: Close five-slot q2/q5/q6/q7/q8 rebalanced static search

Decision:

Close the stricter five-slot q2/q5/q6/q7/q8 rebalanced reference-bank search
as diagnostic-only. It avoids q4/q9 and the high-risk q0/q1/q3/q10/q11 set,
but no enumerated static design reaches the reference-bank continuation gate.
Env30 remains the active reference: official-val
`ACC/FP/FN = 0.973330 / 0.015748 / 0.009642`.

Artifacts:

```text
summary =
  .tmp/env30_static_design_search/five_slot_rebalanced_summary.md
top50 csv =
  .tmp/env30_static_design_search/five_slot_rebalanced_summary.csv
gate-pass csv =
  .tmp/env30_static_design_search/five_slot_rebalanced_gate_pass_rows.csv
primary rows =
  .tmp/env30_static_design_search/five_slot_rebalanced_primary_rows.csv
scratch banks =
  .tmp/env30_static_design_search/five_slot_rebalanced_top1_bank.json
  .tmp/env30_static_design_search/five_slot_rebalanced_top2_bank.json
  .tmp/env30_static_design_search/five_slot_rebalanced_top3_bank.json
```

Static gate audit:

```text
evaluated five-slot candidates = 6226
gate-pass candidates = 0

gate required:
  val GT5 primary-hard p90 <= 85 and match20 >= 0.3125
  train0601 GT5 primary-hard p90 <= 90 and match20 >= 0.2333
  val GT5 impact-only match20 > 0.2500
  val GT4 primary-hard p90 <= 80.432
  changed known-extra overlap = empty
```

Best top50 evidence:

```text
best design = gt5cl1_gt5r1_gt4c2_gt4r1
changed high-risk overlap = empty
changed known-extra overlap = empty

val GT5 impact-only p50/p90/match20 =
  43.579 / 47.756 / 0.2500
val GT5 primary-hard p50/p90/match20 =
  51.595 / 97.903 / 0.0625
train0601 GT5 primary-hard p50/p90/match20 =
  49.012 / 92.705 / 0.0833
val GT4 impact-only p50/p90/match20 =
  40.560 / 50.481 / 0.2500
```

Why:

- Avoiding q4/q9 plus q0/q1/q3/q10/q11 removes the direct known extra-carrier
  overlap, but it also removes enough reference capacity that GT5 primary-hard
  coverage cannot recover to the q9-assisted static row.
- The best design improves val GT5 impact-only APE p50/p90, but its
  impact-only `match20` stays at the default `0.2500`; it does not create a
  new close-reference hit on the official-impact GT5 rows.
- Top50 candidates have val GT5 impact-only `match20 = 0.2500`, val GT5
  primary-hard `match20 <= 0.1250`, and train0601 GT5 primary-hard
  `match20 <= 0.2500`, below the continuation gate.
- Static reference geometry still does not prove trained existence score,
  point-valid survival, final count shape, official-val ACC, or TEST behavior.

Next action:

Do not train, run TEST, tune decode, change thresholds/NMS/`max_det`, alter
loss gains, or commit these scratch banks as candidates. Treat the current
q2/q5/q6/q7/q8 reference-bank line as closed unless a future task provides a
new non-oracle selector or a materially different train-only static design
that can pass the above gate without known extra-carrier overlap.

## 2026-07-16: Keep q2 protected-query static designs diagnostic-only

Decision:

Keep the q2 protected-query static design search diagnostic-only. Two stricter
Q12 static banks were generated to avoid changing high-risk q0/q1/q3/q10/q11:
`q2/q5/q6/q7/q8/q9` and `q2/q4/q5/q6/q7/q8`. Both improve static
nearest-reference geometry, but neither reaches the evidence bar for remote
training. Env30 remains the active reference: official-val
`ACC/FP/FN = 0.973330 / 0.015748 / 0.009642`.

Artifacts:

```text
summary =
  .tmp/env30_static_design_search/summary.md
coverage csv =
  .tmp/env30_static_design_search/static_design_coverage_summary.csv
primary rows =
  .tmp/env30_static_design_search/static_design_primary_rows.csv
q4/q9 historical trace summary =
  .tmp/env30_static_design_search/q4_q9_historical_extra_summary.csv
scratch banks =
  .tmp/env30_static_design_search/q2q5q6q7q8q9_bank.json
  .tmp/env30_static_design_search/q2q4q5q6q7q8_bank.json
```

Static evidence:

```text
default val GT5 primary hard p50/p90/match20 =
  154.365 / 218.186 / 0.0625

q2/q5/q6/q7/q8/q9 val GT5 primary hard =
  35.381 / 68.968 / 0.3125
q2/q5/q6/q7/q8/q9 train0601 GT5 primary hard =
  38.628 / 85.445 / 0.2333
q2/q5/q6/q7/q8/q9 val GT5 impact-only =
  58.137 / 94.982 / 0.2500
changed known-extra overlap = q9

q2/q5/q6/q7/q8 restore-q9 val GT5 primary hard =
  40.753 / 118.111 / 0.2500
q2/q5/q6/q7/q8 restore-q9 train0601 GT5 primary hard =
  38.628 / 95.146 / 0.2333
q2/q5/q6/q7/q8 restore-q9 val GT5 impact-only =
  58.137 / 94.982 / 0.2500
changed known-extra overlap = none

q2/q4/q5/q6/q7/q8 val GT5 primary hard =
  31.542 / 90.534 / 0.3750
q2/q4/q5/q6/q7/q8 train0601 GT5 primary hard =
  39.848 / 101.380 / 0.2333
q2/q4/q5/q6/q7/q8 val GT5 impact-only =
  52.851 / 94.982 / 0.2500
changed known-extra overlap = q4
```

Historical trace risk:

```text
available q4 final-lane traces:
  extra = 19, GT5 extra = 19

available q9 final-lane traces:
  extra = 8, GT5 extra = 0
```

Why:

- Both designs avoid changing the high-risk q0/q1/q3/q10/q11 set.
- The q9 design is safer than the q4 design for GT5 over-count based on
  historical final-lane traces, because q4 repeatedly became a GT5 extra
  carrier in v2/tiered diagnostics.
- Restoring q9 gives a stricter five-slot counterfactual with no changed known
  extra-carrier queries, but its val GT5 primary-hard p90 regresses from
  `68.968` to `118.111`, showing that avoiding both q4 and q9 costs static
  tail coverage under the current prototype set.
- Neither design improves val GT5 impact-only `match20` beyond the default
  `0.2500`; they mainly improve APE p50/p90.
- Static reference geometry does not prove trained existence score,
  point-valid survival, final lane-count shape, official-val ACC, or TEST
  behavior.
- Prior v2 and q1/q3-protected results already show that static coverage gains
  can turn into trained extra final-lane carriers.

Next action:

Do not train, run TEST, tune decode, change thresholds/NMS/`max_det`, alter
loss gains, or commit these scratch banks as candidates. If this line
continues, keep it read-only and search for a stricter five-slot or rebalanced
prototype design that avoids q4 and q9 while improving impact-only GT5 coverage
and recovering the lost GT5 primary-hard p90, then require official-val/train
gates before any TEST.

## 2026-07-16: Keep env30 short GT4/GT5 reference counterfactual diagnostic-only

Decision:

Keep the env30 short GT4/GT5 reference-coverage counterfactual
diagnostic-only. Existing ultrashort banks improve static nearest-reference
coverage on the hard denominator, but no existing bank provides enough evidence
to launch training. Env30 remains the active reference: official-val
`ACC/FP/FN = 0.973330 / 0.015748 / 0.009642`.

Artifacts:

```text
summary =
  .tmp/env30_short_gt45_reference_counterfactual/summary.md
summary csv =
  .tmp/env30_short_gt45_reference_counterfactual/reference_counterfactual_summary.csv
primary rows =
  .tmp/env30_short_gt45_reference_counterfactual/reference_counterfactual_primary_rows.csv
```

Evidence:

```text
default_q12 val GT4 primary hard p50/p90/match20 =
  136.972 / 153.686 / 0.1250
default_q12 val GT5 primary hard p50/p90/match20 =
  154.365 / 218.186 / 0.0625

q1q3protected_v1 val GT4 primary hard p50/p90/match20 =
  41.667 / 80.432 / 0.2500
q1q3protected_v1 val GT5 primary hard p50/p90/match20 =
  36.234 / 68.968 / 0.2500
q1q3protected_v1 train0601 GT5 primary hard p50/p90/match20 =
  37.883 / 85.445 / 0.2333
changed extra-overlap queries = q4,q10

ultrashort_v2 val/train0601 GT5 primary hard match20 =
  0.2500 / 0.2667
changed extra-overlap queries = q3,q4,q10

q1q3protected_v1 val impact-only:
  GT4 p50/p90/match20 = 22.000 / 44.195 / 0.5000
  GT5 p50/p90/match20 = 58.990 / 94.982 / 0.0000
```

Why:

- Static reference coverage improves on the primary hard rows, especially GT5.
- Static geometry evidence does not prove trained query scores, point-valid
  survival, or final count shape.
- The v2 and q1/q3-protected trained diagnostics already showed that static
  coverage gains can become extra final-lane carriers after training.
- Every improving bank changes or uses queries overlapping known extra-lane
  carriers. The impact-only GT5 subset also remains unsolved under the best
  protected static variants.

Next action:

Do not train, run TEST, tune decode, change thresholds/NMS/`max_det`, alter
loss gains, or edit references from this counterfactual. A future reference
line first needs a new static design that improves hard and impact-only GT5
coverage while avoiding known extra-carrier query overlap, then must pass
official-val/train gates before any TEST.

## 2026-07-16: Keep env30 short GT4/GT5 hardset coverage diagnostic-only

Decision:

Keep the env30 short GT4/GT5 hardset coverage matrix diagnostic-only. The
matrix confirms a real short GT4/GT5 raw-candidate hard denominator, but it is
split-imbalanced and too contaminated by already matched positives and known
extra-carrier queries to justify training or a reference/query change. Env30
remains the active reference: official-val
`ACC/FP/FN = 0.973330 / 0.015748 / 0.009642`.

Artifacts:

```text
summary =
  .tmp/env30_short_gt45_hardset_coverage/summary.md
coverage summary =
  .tmp/env30_short_gt45_hardset_coverage/short_gt45_coverage_summary.csv
primary hard rows =
  .tmp/env30_short_gt45_hardset_coverage/primary_short_gt45_hard_rows.csv
primary hard summary =
  .tmp/env30_short_gt45_hardset_coverage/primary_short_gt45_hard_summary.csv
cross-run summary =
  .tmp/env30_short_gt45_hardset_coverage/official_impact_gt45_cross_run_summary.csv
```

Protocol:

```text
splits = official-val, train0601, train0531
TEST = not used
primary hard row =
  gt_count in {4,5}
  visible_points_gt <= 20
  (raw_has_match_20px = false OR pred_valid_points@0.6 < min_points)
```

Evidence:

```text
official-val GT4 visible<=20:
  lanes = 106, hard_fail = 10, official-impact tails = 4

official-val GT5 visible<=20:
  lanes = 207, hard_fail = 17, official-impact tails = 5

train0601 GT4/GT5 visible<=20:
  GT4 hard_fail = 15 / 134
  GT5 hard_fail = 62 / 671

train0531 GT4/GT5 visible<=20:
  GT4 hard_fail = 7 / 68
  GT5 denominator absent in this available diagnostic
```

Primary hard row risk:

```text
val GT4 selected = 10
  official-impact = 4, non-impact = 6
  final-matched positives touched = 6
  query hist = q0 5, q10 2, q11 2, q6 1

val GT5 selected = 17
  official-impact = 5, non-impact = 12
  final-matched positives touched = 13
  query hist = q0 1, q1 1, q3 9, q7 1, q8 1, q10 2, q11 2

train0601 GT4 selected = 15
  final-matched positives touched = 13
  query hist = q0 2, q1 1, q5 1, q6 2, q7 1, q8 1, q9 1, q10 4, q11 2

train0601 GT5 selected = 61
  final-matched positives touched = 51
  query hist = q0 6, q1 8, q3 30, q5 1, q7 3, q8 2, q10 8, q11 3

train0531 GT4 selected = 7
  final-matched positives touched = 5
  query hist = q1 2, q4 1, q7 1, q10 1, q11 2
```

Why:

- The primary hard row covers all short GT4/GT5 official-impact tails on
  official-val (`9/9`), but selects `27` val rows; `18` are non-impact readouts.
- Most selected rows already have Hungarian/final positives. This is a
  coverage diagnostic, not a safe direct loss target.
- The selected query histograms overlap known extra carriers from env30 and
  tiered-v2 final-query diagnostics, especially q0/q1/q3/q10/q11.
- The surface is split-imbalanced: train0601 is GT5-heavy while train0531 has
  no GT5 short denominator in the available diagnostic.
- The failures include raw geometry/no20 gaps, not only point-valid survival,
  so a valid-only rescue is not supported.

Next action:

Do not train, run TEST, tune decode, change thresholds/NMS/`max_det`, alter
loss gains, or edit references from this matrix. A future hardset-driven
reference/query candidate first needs a read-only reference-coverage
counterfactual proving improved raw `has20`/APE on this denominator without
broadening GT3/GT4 extra-lane carriers or suppressing already matched
positives, then it must pass official-val gates before any TEST.

## 2026-07-16: Close env30 non-oracle selector table as diagnostic-only

Decision:

Keep env30 non-oracle selector work diagnostic-only. The requested
training-available selector table excludes final-match state, final decode
inclusion, post-decode extra attribution, nearest-GT far/near labels, and TEST,
but still does not produce a clean implementation target. Env30 remains the
active reference: official-val `ACC/FP/FN = 0.973330 / 0.015748 / 0.009642`.

Artifacts:

```text
summary =
  .tmp/env30_non_oracle_selector_table/summary.md
selector summary =
  .tmp/env30_non_oracle_selector_table/selector_summary.csv
split distribution =
  .tmp/env30_non_oracle_selector_table/selector_split_distribution.csv
sample rows =
  .tmp/env30_non_oracle_selector_table/selected_rows_sample.csv
```

Evidence:

```text
raw_no20:
  selected = 32
  tail covered = 13 / 15
  non-impact selected = 19
  final-matched / Hungarian positives touched = 21 / 27
  train0601 selected = 70, train0531 selected = 18

assign_ape20_or_line08:
  selected = 27
  tail covered = 13 / 15
  non-impact selected = 14
  final-matched / Hungarian positives touched = 15 / 22
  train0601 selected = 59, train0531 selected = 12

raw_no20_score_lt01:
  selected = 4
  tail covered = 4 / 15
  non-impact selected = 0
  final-matched / Hungarian positives touched = 0 / 4

raw_no20_valid_lt_min:
  selected = 5
  tail covered = 5 / 15
  non-impact selected = 0
  final-matched / Hungarian positives touched = 0 / 5

raw_no20_score_lt05:
  selected = 6
  tail covered = 6 / 15
  non-impact selected = 0
  final-matched / Hungarian positives touched = 1 / 6
  extra-image touched = 1

raw_score_lt01_or_valid_lt_min:
  selected = 14
  tail covered = 7 / 15
  non-impact selected = 7
  final-matched / Hungarian positives touched = 8 / 14

assign_line_lt08:
  selected = 12
  tail covered = 10 / 15
  non-impact selected = 2
  final-matched / Hungarian positives touched = 2 / 9

assign_ape20_line08:
  selected = 11
  tail covered = 9 / 15
  non-impact selected = 2
  final-matched / Hungarian positives touched = 2 / 8
```

Why:

- Broad raw/assignment geometry selectors cover many tails but also touch many
  non-impact and already matched positives.
- Strict score/valid selectors cover only `4/15`, `5/15`, or `6/15`
  official-impact tails. Broader score/valid combinations reach `7/15` only by
  reintroducing non-impact and final-matched contamination.
- Assignment line-accuracy reaches `10/15` tail coverage, but still touches
  non-impact, final-matched, and Hungarian positives with split-instability.
- Visible-count filters do not create split stability; train0601 and
  train0531 keep different GT-count and query surfaces.
- The current artifacts do not expose an explicit train target margin for
  quality/existence. Raw existence score and point-valid survival are only
  proxies and are not enough for a safe train label.

Next action:

Do not train, do not run TEST, and do not tune decode, thresholds, NMS,
checkpoint choice, weak-positive rescue weights, or global
`gcs_exist_quality_alpha` from this evidence. This closes the current
selective exist-quality / weak-visible matched-positive calibration path for
implementation. A future attempt must first produce a new non-oracle separator
or explicit train target-margin artifact, then pass official-val gates before
any TEST.

## 2026-07-16: Keep remaining env30 candidate directions diagnostic-only

Decision:

Keep the remaining env30 candidate directions diagnostic-only. The latest
split/date/GT-count stability audit and read-only exist-quality gate do not
support training, decode search, or TEST. Env30 remains the active reference:
official-val `ACC/FP/FN = 0.973330 / 0.015748 / 0.009642`.

Artifacts:

```text
distribution stability audit =
  .tmp/env30_distribution_stability_audit/summary.md
final-query distribution =
  .tmp/env30_distribution_stability_audit/final_query_distribution.csv
positive denominator distribution =
  .tmp/env30_distribution_stability_audit/positive_denominator_distribution.csv
official-impact tail distribution =
  .tmp/env30_distribution_stability_audit/official_impact_tail_distribution.json
selective exist-quality gate =
  read-only gcs_experiment_analyst result, no file edits
```

Evidence:

```text
official-val final-query extras:
  over/under = 7 / 3
  extra lanes = 7
  extra query hist = q0 4, q11 3
  extra GT hist = GT3 7
  extra class hist = clear_far 7

train0601 final-query extras:
  over/under = 8 / 5
  extra lanes = 8
  extra query hist = q1 4, q10 3, q11 1
  extra GT hist = GT3 1, GT4 7
  extra class hist = clear_far 8

train0531 final-query extras:
  over/under = 7 / 1
  extra lanes = 8
  extra query hist = q0 2, q1 1, q9 1, q10 1, q11 3
  extra GT hist = GT2 1, GT3 5, GT4 2
  extra class hist = clear_far 5, duplicate_like 1, near_gt_nonduplicate 2

positive denominator geom_gt20:
  val selected = 26, GT3/GT4/GT5 = 1 / 10 / 15
  train0601 selected = 57, GT4/GT5 = 9 / 48
  train0531 selected = 11, GT3/GT4 = 5 / 6

positive denominator no_final_match:
  val selected = 17, ok_or_near_ok = 5
  train0601 selected = 26, ok_or_near_ok = 12
  train0531 selected = 10, ok_or_near_ok = 5
```

Selective exist-quality gate:

```text
classification = diagnostic-only
promotable = no
confidence = about 90% that it should remain diagnostic-only now
reason =
  no stable train-side selector;
  global gcs_exist_quality_alpha evidence is below env30;
  raw-tail target disagrees with assignment-best metrics;
  no selective official-val candidate exists against env30
```

Why:

- Env30 final-extra carriers are not stable across splits. A query-specific
  rule from official-val q0/q11 would miss train0601 q1/q10/q11 and train0531
  q0/q1/q9/q10/q11 behavior.
- Positive-denominator failures are split-dependent: train0601 is GT5-heavy,
  while train0531 is GT3/GT4-only in this diagnostic surface.
- `no_final_match` includes `ok_or_near_ok` rows on every split and depends on
  decode-after-the-fact final match state, so it cannot be used as a direct
  train-side label.
- Selective exist-quality / weak-visible matched-positive calibration has no
  stable non-oracle selector and no official-val evidence that it can beat or
  match env30 without GT3/GT4 over-count or GT5 `5->6` risk.

Next action:

Do not train, do not run TEST, and do not tune thresholds, decode, NMS,
`max_det`, checkpoint choice, q3 suppression, global
`gcs_exist_quality_alpha`, or weak-positive rescue weights from this evidence.
The only approved follow-up is read-only: build a training-available
non-oracle selector table that excludes `no_final_match` and GT post-decode
attribution, adds visible-count/quality/existence margin fields when available,
and proves it does not mostly select already-final/near-ok positives before
any implementation proposal.

## 2026-07-16: Reject q1/q3-protected ultrashort dataref diagnostic

Decision:

Reject
`query_alpha05_env30_q12_ultrashort_q1q3protected_dataref_geom1_extraguard_v1`
before TEST. The run was a controlled default-off diagnostic that preserved
q1/q3 defaults and moved GT5 ultrashort prototypes to q4/q5, but it failed the
official-val early-stop gate. It was stopped with `TERM` after 18 complete
`results.csv` rows, while epoch 19 was running.

Protocol:

```text
run = query_alpha05_env30_q12_ultrashort_q1q3protected_dataref_geom1_extraguard_v1
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-ultrashort-dataref-q1q3protected-v1.yaml
bank = data/gcs_reference_banks/q12_ultrashort_env30_train_q1q3protected_v1.json
static audit = data/gcs_reference_banks/q12_ultrashort_env30_train_q1q3protected_v1_reference_audit.json
RUN_TESTS = 0
official-val GT = canonical 363-image official-val JSON
primary checkpoint = weights/official_best.pt
official_best source_epoch = 15
selected decode = conf 0.001, point_valid_thr 0.6, nms_dist_px 0,
  max_det 5, min_points 3, valid_before_maxdet true
TEST artifacts = none
```

Official-val evidence:

```text
epoch005 ACC/FP/FN = 0.932141 / 0.151010 / 0.086088
epoch010 ACC/FP/FN = 0.945669 / 0.141873 / 0.059917
epoch015 ACC/FP/FN = 0.949375 / 0.113682 / 0.047980

env30 reference ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
```

Epoch 15 selected count shape:

```text
count_acc_3/4/5 = 0.726457 / 0.681818 / 0.959459
count_confusion =
  3->3 162, 3->4 49, 3->5 12,
  4->4 45, 4->5 21,
  5->4 3, 5->5 71
```

Hidden `max_det=6` overcount at epoch 15:

```text
best max_det=6 row ACC/FP/FN = 0.947559 / 0.133976 / 0.047291
count_confusion includes 4->5 12, 4->6 9, 5->6 69
```

Final-query carrier diagnostics:

```text
selected max_det=5 diagnostic =
  runs/gcs_lane/query_alpha05_env30_q12_ultrashort_q1q3protected_dataref_geom1_extraguard_v1_official_best_val_final_query_extra_valid_before_maxdet
selected max_det=5 over/under images = 82 / 3
selected max_det=5 extra_topk/far/duplicate = 94 / 84 / 3 lanes
selected max_det=5 extra hist =
  q11 33, q6 18, q0 13, q5 7, q1 6, q9 6, q7 5, q8 5, q2 1

diagnostic max_det=6 carrier diagnostic =
  runs/gcs_lane/query_alpha05_env30_q12_ultrashort_q1q3protected_dataref_geom1_extraguard_v1_epoch015_val_final_query_extra_maxdet6_diag
diagnostic max_det=6 over/under images = 151 / 3
diagnostic max_det=6 extra_topk/far/duplicate = 173 / 156 / 6 lanes
diagnostic max_det=6 GT5 5->6 carrier hist =
  q1 22, q6 21, q5 13, q11 9, q8 3, q9 1

env30 same-protocol max_det=5 diagnostic over/under images = 7 / 3
env30 same-protocol max_det=5 extra_topk/far/duplicate = 7 / 7 / 0 lanes
```

Sweep rescue check:

```text
epoch015 sweep rows = 864
rows with ACC >= env30 0.973330 = 0
rows with FN <= env30 0.009642 = 0
rows with GT4 4->5 = 0 = 0
rows passing count-shape gate
  (4->5=0, 4->6=0, 5->4<=1, 5->6=0, FN<=0.009642) = 0
```

Why:

- Three consecutive official intervals stayed far below the env30 ACC/FN gate.
- FP stayed high, and the count shape never became acceptable.
- The epoch015 selected row avoids GT5 sixth lanes only by `max_det=5` and
  still creates severe GT4 false-fifth behavior.
- The `max_det=6` surface exposes the same sixth-lane overcount failure that
  rejected previous ultrashort dataref candidates.
- Final-query diagnostics show the trained extra lanes are mostly nearest-GT
  far and distributed across q11/q6/q0/q5/q1/q7/q8 under selected decode and
  q1/q6/q5/q11/q8/q9 for GT5 sixth lanes under `max_det=6`; this is not a
  q4/q5-only transfer and not primarily duplicate/NMS failure.
- Internal validation continued to improve through epoch18, but official-val
  is the selection/gating surface; internal `val/f1` cannot rescue a candidate
  with zero official sweep rows passing the gate.

Next action:

Do not continue this run, do not run TEST, and do not tune decode/loss/query
assignment from this failed line. Return to env30 as the reference. The next
safe action is broader bottleneck diagnosis from env30 and accepted/rejected
official-val evidence before proposing another implementation change.

## 2026-07-16: Cross-run rejected-family bottleneck synthesis

Decision:

Keep env30 as the active reference and do not launch another broad
dataref/count/tiered ablation from the currently rejected families. The
cross-run matrix was generated locally at
`.tmp/cross_run_bottleneck_matrix/matrix.md` using official-val summaries and
final-query diagnostics only. TEST remains closed for selection.

Key comparisons:

```text
env30 reference:
  ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
  GT4 4->5 = 0
  GT5 5->4/5->6 = 1 / 0
  final-query over/under = 7 / 3
  final-query extra_topk/far/duplicate = 7 / 7 / 0

ultrashort dataref v2:
  ACC/FP/FN = 0.972040 / 0.023691 / 0.011938
  GT4 4->5/4->6 = 2 / 1
  GT5 5->4/5->6 = 2 / 12
  final-query over/under = 22 / 3
  final-query extra_topk/far/duplicate = 23 / 13 / 0

q1/q3-protected dataref:
  ACC/FP/FN = 0.949375 / 0.113682 / 0.047980
  GT3 3->4/3->5 = 49 / 12
  GT4 4->5 = 21
  GT5 5->4 = 3
  final-query over/under = 82 / 3
  final-query extra_topk/far/duplicate = 94 / 84 / 3

count-restore env30:
  ACC/FP/FN = 0.944877 / 0.096740 / 0.063131
  GT4 4->5 = 10
  GT5 5->4 = 6

tiered GT4/GT5 + GT4PV v2:
  ACC/FP/FN = 0.973049 / 0.014141 / 0.008724
  GT4 4->5 = 2
  GT5 5->4/5->6 = 1 / 5
  reporting-only TEST was worse than env30
```

Why:

- No rejected family beats env30 on the complete official-val gate.
- Dataref/ultrashort variants mainly introduce extra-lane carriers and
  over-count, often nearest-GT far rather than simple duplicate-like lanes.
- Count restoration on env30 damages both primary official metrics and count
  shape.
- Tiered GT4/GT5 + GT4PV reduces official-val FP/FN but gives lower ACC and
  worse reporting-only TEST than env30.

Next action:

Do not propose an implementation change from this matrix. No line reaches
>=90% engineering confidence. The next safe action is diagnostic-only:
classify env30 official-val failure cases by raw geometry, point-valid
survival, final-query clear-far/near-GT/duplicate type, GT4/GT5 count transfer,
and date/session distribution before deciding whether a narrower candidate is
justified.

## 2026-07-16: Classify env30 residual official-val failures

Decision:

Keep env30 as reference and do not implement a new loss, decode, sampler, or
reference-bank change from the residual-failure classification alone. The
classification is diagnostic-only and uses official-val selected decode, raw
Q12 diagnostics, and a small visual overlay sample. TEST remains closed for
selection.

Artifacts:

```text
summary = .tmp/env30_failure_classification/summary.md
official-impact tail CSV =
  .tmp/env30_failure_classification/official_impact_tail_lanes.csv
visual overlays = .tmp/env30_tail_overlays/
```

Evidence:

```text
official-val images = 363
FP images = 21
FN images = 10
over/under images = 7 / 3
low image_acc <0.95 / <0.90 = 23 / 8

raw-Q12 GT lanes = 1303
finite raw best APE mean/p50/p90 =
  5.627210 / 4.254436 / 10.033714 px
no raw match within 20/30/40 px = 32 / 21 / 11
pred_valid_points@0.6 < min_points = 14
official-impact tail lanes = 15 / 41 tail lanes

final-query extra top-k lanes = 7
extra far / duplicate = 7 / 0
extra carriers = q0 4, q11 3
```

Why:

- Env30 residual count errors are small, but official FP/FN also include
  matched-lane geometry and point-valid survival tails.
- Raw-Q12 performance is generally strong; the useful signal is in a small
  official-impact tail subset, not in every no20 or valid<min lane.
- Visual overlays show mixed failure modes: full center/right lane misses,
  ultra-short GT lanes that do not always hurt official FP/FN, and a few
  clear-far q0/q11 over-counts.
- This does not support a direct GT5-only, count-only, broad dataref, or
  duplicate/NMS implementation change.

Next action:

If continuing, compare the 15 official-impact tail lanes across existing
env30, v2, tiered, and count-restore artifacts before designing any new
candidate. A future implementation needs evidence that it fixes these tails
without reintroducing the much larger over-count patterns already observed in
rejected runs.

## 2026-07-16: Compare env30 official-impact tails across rejected runs

Decision:

Keep env30 as the active reference and do not start a new implementation from
the tail cross-run comparison. The diagnostic compared the 15 env30
official-impact tail lanes against v2, tiered v2, count-restore, and
q1/q3-protected artifacts using official-val evidence only. TEST remains
closed for selection.

Artifacts:

```text
summary = .tmp/tail_cross_run/tail_cross_run_summary.md
json = .tmp/tail_cross_run/tail_cross_run_summary.json
per-lane CSV = .tmp/tail_cross_run/official_impact_tail_cross_run.csv
```

Evidence:

```text
env30:
  ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
  global no20 / valid<min = 32 / 14
  has20 rate / APE p90 = 0.975441 / 10.033714
  tail still bad = 15
  final-query over/under = 7 / 3

v2:
  ACC/FP/FN = 0.972040 / 0.023691 / 0.011938
  global no20 / valid<min = 43 / 6
  tail no20 fixed / valid fixed / both ok = 2 / 3 / 3
  tail still bad = 12
  final-query over/under = 22 / 3

tiered_v2:
  ACC/FP/FN = 0.973049 / 0.014141 / 0.008724
  global no20 / valid<min = 27 / 10
  tail no20 fixed / valid fixed / both ok = 3 / 5 / 5
  tail still bad = 10
  reporting-only TEST was worse than env30

countrestore:
  ACC/FP/FN = 0.944877 / 0.096740 / 0.063131
  global no20 / valid<min = 186 / 40
  tail still bad = 13

q1q3:
  ACC/FP/FN = 0.949375 / 0.113682 / 0.047980
  global no20 / valid<min = 163 / 19
  tail still bad = 13
  final-query over/under = 82 / 3
```

Why:

- No existing rejected run fixes the 15 official-impact env30 tail lanes
  without broader regression.
- v2 and tiered v2 provide weak diagnostic signal that some tail failures are
  addressable, but v2 creates over-count and tiered v2 does not pass the env30
  ACC gate.
- Count-restore and q1/q3-protected are ruled out as tail-fix carriers because
  their global raw geometry and official metrics degrade sharply.
- The evidence does not reach the >=90% engineering-confidence threshold for a
  new implementation change.

Next action:

Stay diagnostic-only. If this line continues, inspect the tiered_v2 tail fixes
against env30 at per-lane/per-query level and identify the specific mechanism
that repairs those tails before designing a narrower official-val-only
experiment. Do not run TEST and do not revive rejected families wholesale.

## 2026-07-16: Tiered v2 tail fixes are not count-safe

Decision:

Do not reuse the tiered GT4/GT5 + GT4PV rescue as an implementation candidate
or as a direct small-weight follow-up. A narrower per-lane check shows that its
official-impact tail repairs are real but mostly not count-safe.

Evidence:

```text
scope = 15 env30 official-impact tail lanes, official-val diagnostics only
tiered_v2 lane-level fixes = 6
fix flags = fix_valid 3, fix_no20+fix_valid 2, fix_no20 1
fixed GT counts = GT3 2, GT4 2, GT5 2
fixed visible buckets = <=10: 1, 11..20: 3, >20: 2

env30 final count transitions on the six fixed lanes:
  GT3 3->4: 2
  GT4 4->3: 2
  GT5 5->5: 2

tiered_v2 final count transitions on the six fixed lanes:
  GT3 3->3: 1
  GT3 3->4: 1
  GT4 4->5: 2
  GT5 5->6: 2
```

Global raw-Q12 official-val comparison:

```text
env30      no20 / valid<min / APE p90 = 32 / 14 / 10.033714
tiered_v2  no20 / valid<min / APE p90 = 27 / 10 / 9.512536

GT3 no20 / valid<min: env30 6 / 5, tiered_v2 5 / 0
GT4 no20 / valid<min: env30 12 / 4, tiered_v2 9 / 7
GT5 no20 / valid<min: env30 14 / 5, tiered_v2 13 / 3
```

Why:

- Tiered v2 improves some raw geometry and point-valid survival, so the tail
  signal is not noise.
- Most lane-level fixes either remain over-counted or become over-counted:
  the GT4 fixed lanes map to `4->5`, and the GT5 fixed lanes map to `5->6`.
- This explains the earlier run-level rejection: official-val FP/FN improve
  slightly, but ACC and count shape remain below env30, with GT4 false fifth
  and GT5 sixth-lane regressions.

Next action:

Stay diagnostic-only. If the tiered signal is revisited, first separate
tail-lane retention from extra-lane carrier survival in official-val raw and
final-query traces. A new training candidate would need an explicit gate that
keeps the raw/valid tail gains while preserving env30 count shape:
GT4 `4->5=0`, GT5 `5->6=0`, and GT5 `5->4<=1`.

## 2026-07-16: Reject tiered v2 final-query carrier shape

Decision:

Keep `query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2`
rejected. The official-val final-query carrier diagnostic confirms that the
tiered tail-retention signal is coupled to far extra-lane survival, not a
clean duplicate/NMS problem.

Protocol:

```text
split = official-val only
weights =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2/weights/official_best.pt
decode = conf 0.001, point_valid_thr 0.6, nms_dist_px 30,
  max_det 6, min_points 2, valid_before_maxdet true
remote diagnostic =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_official_best_val_final_query_extra_valid_before_maxdet
local copy =
  .tmp/tiered_v2_final_query_diag/
TEST = not used
```

Evidence:

```text
over/under images = 11 / 3
extra_topk_lanes = 11
extra_topk score mean = 0.466249
nearest-GT APE p50/p90 = 145.333328 / 248.409088 px
nearest-pred distance p50/p90 = 133.0 / 250.181824 px

extra query hist =
  q0 1, q1 2, q3 2, q4 2, q9 1, q10 2, q11 1
GT5 5->6 extra query hist =
  q3 2, q4 2, q10 1
GT4 4->5 extra query hist =
  q9 1, q10 1
duplicate-like extras = 0
nearest-GT far extras = 10
```

Gate check against env30:

```text
pass = false
official_val_over_images: candidate 11, baseline 7, limit 10
official_val_gt5_5to6_extra_zero: candidate 5, limit 0
official_val_gt4_4to5_extra_le_baseline: candidate 2, baseline 0
official_val_focus_queries_gt5_5to6_zero: candidate 2, limit 0
```

Why:

- The tail-retention signal is real but not count-safe.
- The final extra lanes are almost all nearest-GT far and not duplicate-like,
  so wider NMS or duplicate-only suppression is not the supported fix.
- GT5 sixth-lane carriers are q3/q4/q10, while GT4 false-fifth carriers are
  q9/q10; this is a distributed carrier problem rather than a single-query
  handoff.

Next action:

Do not tune from tiered v2 or run TEST. A future diagnostic should compare the
matched tail-retention lanes and the far extra carriers within the same
official-val images to find a separable feature before any training change.

Follow-up feature split:

```text
tiered_v2 matched fixed-tail final lanes:
  n = 6
  query hist = q0 2, q4 1, q10 1, q11 2
  score median/range = 0.638 / 0.208..0.980
  valid_len median/range = 13.5 / 9..47
  matched-GT APE median/range = 9.051 / 2.375..21.909 px

tiered_v2 far extra final lanes:
  n = 10
  query hist = q0 1, q1 2, q3 1, q4 2, q9 1, q10 2, q11 1
  score median/range = 0.350 / 0.196..0.899
  valid_len median/range = 9.5 / 5..44
  nearest-GT APE median/range = 151.333 / 55.400..274.167 px
```

This rules out a simple score threshold, valid-length threshold, or query-wide
suppression as the next implementation. The same query ids can be useful
tail-retention carriers and harmful far-extra carriers. The only strong split
observed here is GT-based nearest-lane distance, which can inform a train-only
diagnostic but cannot be used at inference or decode.

GT-oracle removal counterfactual:

```text
tiered_v2 baseline:
  over/under = 11 / 3
  3->4 = 4, 4->3 = 2, 4->5 = 2, 5->4 = 1, 5->6 = 5

remove only GT-labeled far extra final lanes:
  over/under = 1 / 3
  3->4 = 0, 4->3 = 2, 4->5 = 0, 5->4 = 1, 5->6 = 1

remove all GT-labeled extra_topk final lanes:
  over/under = 0 / 3
  3->4 = 0, 4->3 = 2, 4->5 = 0, 5->4 = 1, 5->6 = 0
```

The oracle removal preserves the six matched tail-retention lanes because they
are not labeled as extra. This makes train-only far-extra carrier suppression a
reasonable diagnostic axis, but still below the implementation threshold:
the separator is GT-based, prior broad final-extra guarding was insufficient
on v2/q1q3, and query ids overlap between matched tails and harmful extras.

## 2026-07-16: Train split confirms tiered far-extra scope gap

Decision:

Do not start a new training run yet. Train0601/train0531 diagnostics confirm a
real far-extra carrier signal for tiered v2, but the evidence still supports a
diagnostic-only expanded-scope hypothesis rather than a direct implementation
or gain bump.

Protocol:

```text
split = train0601 and train0531 only
weights =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2/weights/official_best.pt
decode = conf 0.001, point_valid_thr 0.6, nms_dist_px 30,
  max_det 6, min_points 2, valid_before_maxdet true
train0601 diagnostic =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_official_best_train0601_final_query_extra_valid_before_maxdet
train0531 diagnostic =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_official_best_train0531_final_query_extra_valid_before_maxdet
local copy =
  .tmp/tiered_v2_train_final_query_diag/
TEST = not used
```

Evidence:

```text
env30 train0601 over/under/extra = 8 / 5 / 8
v2 train0601 over/under/extra = 70 / 6 / 70
tiered_v2 train0601 over/under/extra = 21 / 5 / 24

tiered_v2 train0601:
  extra hist = q0 4, q3 5, q4 5, q6 1, q7 2, q8 1, q10 6
  far / duplicate / near-nondup extras = 16 / 3 / 7
  GT4 4->5 = 7, GT4 4->6 = 3, GT5 5->6 = 11

env30 train0531 over/under/extra = 7 / 1 / 7
v2 train0531 over/under/extra = 7 / 1 / 8
tiered_v2 train0531 over/under/extra = 4 / 0 / 4

tiered_v2 train0531:
  extra hist = q1 1, q9 1, q10 2
  far / duplicate / near-nondup extras = 4 / 0 / 0
  GT4 4->5 = 3, GT5 5->6 = 0
```

GT-oracle counterfactual:

```text
tiered_v2 train0601 baseline:
  over/under = 21 / 5
  4->5 = 7, 4->6 = 3, 5->6 = 11
remove far extras:
  over/under = 8 / 5
  4->5 = 3, 4->6 = 0, 5->6 = 5
remove far-or-duplicate extras:
  over/under = 7 / 5
  4->5 = 2, 4->6 = 0, 5->6 = 5
remove all extra_topk:
  over/under = 0 / 5
  4->5 = 0, 4->6 = 0, 5->6 = 0

tiered_v2 train0531 baseline:
  over/under = 4 / 0
remove far extras:
  over/under = 0 / 0
```

Guard-scope audit:

```text
current gcs_final_extra_guard_scope = q1,q3,q4,q5,q6,q7,q8

val far extras in/out current scope = 5 / 5
  outside scope = q0 1, q9 1, q10 2, q11 1

train0601 far extras in/out current scope = 8 / 8
  outside scope = q0 3, q10 5

train0531 far extras in/out current scope = 1 / 3
  outside scope = q9 1, q10 2
```

Existing final-extra-guard log evidence:

```text
v2 remote results.csv:
  train/final_extra_guard_count first/last/last10mean =
    17.7843 / 0.14706 / 0.192157
  val/final_extra_guard_count = 0 throughout

q1q3-protected results.csv:
  train/final_extra_guard_count first/last/last10mean =
    25.3529 / 0.85294 / 0.664707
  val/final_extra_guard_count = 0 throughout
```

Why:

- The scope gap is real: q10 is a repeated far-extra carrier and is outside
  the current train-only guard scope.
- The direction is still not clean enough for direct promotion. Existing
  `gcs_final_extra_guard=0.02` was active in v2/q1q3 and did not prevent
  official-val failure.
- Far/duplicate suppression is insufficient for train0601 because five GT5
  `5->6` near-nonduplicate extras remain after oracle far/duplicate removal.
- q0/q10/q11 can also be useful matched tail-retention queries, so query-wide
  suppression is invalid.

Next action:

Do not run TEST or launch a full candidate solely from this evidence. The next
safe step is a narrower diagnostic design for expanded-scope train-only guard
selection: estimate whether adding q0/q9/q10/q11 to the guard scope would
select only GT-far unmatched extras and protect matched/GT-close tails, then
gate any future training on official-val ACC, GT4 `4->5=0`, GT5 `5->6=0`,
and GT5 `5->4<=1`.

## 2026-07-16: Replay expanded final-extra guard scope

Decision:

Keep expanded `gcs_final_extra_guard_scope` as diagnostic-only. The replay
confirms that q0/q9/q10/q11 include missed clear-far far-extra candidates, but
train0601 also contains protected q10 candidates. This is not enough to launch
a new training run or increase the guard gain.

Protocol:

```text
weights =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2/weights/official_best.pt
diagnostic script =
  .tmp/diagnose_final_guard_scope.py
local artifacts =
  .tmp/tiered_v2_guard_scope_trainimgsz_diag/
remote val =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_guard_scope_val_trainimgsz_diag
remote train0601 =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_guard_scope_train0601_trainimgsz_diag
remote train0531 =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_guard_scope_train0531_trainimgsz_diag
TEST = not used
distance pixel scale = training --imgsz 544 960
```

Evidence:

```text
official-val outside scope q0/q9/q10/q11:
  candidate/selected/protected = 8 / 7 / 1
  selected queries = q0 1, q9 2, q10 2, q11 2
  protected queries = q9 1
  selected GT counts = GT3 2, GT4 4, GT5 1
  selected_clear_far / duplicate_like = 7 / 0

train0601 outside scope q0/q9/q10/q11:
  candidate/selected/protected = 10 / 6 / 3
  selected queries = q0 3, q10 3
  protected queries = q10 3
  selected GT counts = GT4 4, GT5 2
  selected_clear_far / duplicate_like = 6 / 0

train0531 outside scope q0/q9/q10/q11:
  candidate/selected/protected = 4 / 4 / 0
  selected queries = q9 1, q10 3
  selected GT counts = GT4 4
  selected_clear_far / duplicate_like = 4 / 0

Trace join against existing final-query extras:
  official-val final outside extras selected/protected/other = 5 / 0 / 0
  train0601 final outside extras selected/protected/other = 6 / 2 / 2
  train0531 final outside extras selected/protected/other = 3 / 0 / 0
```

Why:

- The scope-gap signal is stronger than before: every selected outside-scope
  replay candidate is clear-far, and q10 repeats across val, train0601, and
  train0531.
- The safety signal is still incomplete: train0601 has three q10 protected
  candidates, so expanded scope would touch a query that sometimes behaves
  like a true lane.
- The final-query join is mixed. Official-val outside extras are fully covered
  by selected replay rows, but train0601 includes two q10 final extras that are
  not selected under training scale, one q10 near/non-far candidate, and one q0
  final extra that is skipped as Hungarian-matched in the guard replay. Scope
  expansion therefore does not
  cover all train0601 over-count modes.
- This replay estimates training-loss candidate selection, not final official
  accuracy. It does not solve the earlier tiered-v2 issue that near-nonduplicate
  train0601 GT5 `5->6` extras remain after far/duplicate oracle removal.

Follow-up tail/extra separation check:

```text
tiered_v2 fixed official-impact tail lanes = 6
expanded-guard status for those six tail lanes =
  matched=True, candidate=False, selected=False, protected=False

same-image final extra lanes = 4
expanded-guard status for those four final extras =
  candidate=True, selected=True, protected=False, matched=False
extra rows =
  clips/0601/1494452925855079794/20.jpg q10
  clips/0601/1495485219551850010/20.jpg q3
  clips/0531/1492724773779246375/20.jpg q9
  clips/0313-2/580/20.jpg q0
```

Follow-up `clear_far_px` counterfactual:

```text
lowering clear_far_px from 50 to 40 captures exactly one additional
train0601 final outside extra:
  clips/0601/1494453811467016035/20.jpg q10 GT4
  guard nearest_gt_ape_px = 42.971340
  line_acc = 0.277778
  valid_len = 11
  final_far=True
```

These follow-ups strengthen the diagnostic value of expanded-scope guard
selection, but they do not change the decision. The tail/extra split is clean
on the six official-val fixed-tail examples, while train0601 still has
protected, near/non-far, and Hungarian-matched-skip outside extras. Lowering
the clear-far threshold fixes only one of those train0601 rows.

Train0601 residual blocker audit:

```text
outside final extras = 10
outside queries = q0 4, q10 6
classification after joining final-query trace to expanded guard replay:
  selected = 6
  protected = 2
  near/non-far candidate = 1
  Hungarian-matched skip = 1
```

The four not-selected rows are:

```text
protected:
  clips/0601/1494453653536229036/20.jpg q10 GT4 4->6
  line_acc=1.0

protected:
  clips/0601/1495492674588939817/20.jpg q10 GT5 5->6
  nearest_gt_ape=26.58, line_acc=0.7

near/non-far:
  clips/0601/1494453811467016035/20.jpg q10 GT4 4->6
  nearest_gt_ape=42.97 < clear_far_px 50

Hungarian-matched skip:
  clips/0601/1495492704577874815/20.jpg q0 GT4 4->5
  matched=True, candidate=False
```

GT-after-decode counterfactual:

```text
baseline:
  over/under = 21 / 5
  4->5 = 7, 4->6 = 3, 5->4 = 4, 5->6 = 11

remove selected outside:
  over/under = 15 / 5
  4->5 = 3, 4->6 = 3, 5->4 = 4, 5->6 = 9

remove all outside extras:
  over/under = 13 / 5
  4->5 = 4, 4->6 = 1, 5->4 = 4, 5->6 = 8

remove all extra_topk:
  over/under = 0 / 5
  4->5 = 0, 4->6 = 0, 5->4 = 4, 5->6 = 0
```

Integrated conclusion: expanded scope finds a real missed clear-far subset,
but it fails the train0601 gate as a training candidate. It cannot address the
protected q10 rows, the near/non-far q10 row, or the q0 Hungarian-matched-skip
row, and selected-outside suppression leaves substantial `4->6` and `5->6`.
The engineering confidence is high for diagnostic rejection of direct expanded
scope training, not for any new implementation.

All-extra alignment makes the rejection stronger:

```text
official-val extra_topk vs expanded guard:
  selected = 8
  protected = 3

train0601 extra_topk vs expanded guard:
  selected = 10
  protected = 7
  near/non-far candidate = 2
  matched skip = 3
  noncandidate = 2

train0531 extra_topk vs expanded guard:
  selected = 4
```

GT-after-decode all-extra counterfactual:

```text
official-val:
  baseline over/under = 11 / 3
  remove selected = 3 / 3
  remove selected+protected = 0 / 3

train0601:
  baseline over/under = 21 / 5
  remove selected = 12 / 5
  remove selected+protected = 7 / 5
  remaining after selected+protected removal:
    4->5 = 3, 5->4 = 4, 5->6 = 4

train0531:
  baseline over/under = 4 / 0
  remove selected = 0 / 0
```

Decision update: pause direct expanded `gcs_final_extra_guard_scope` training
as a next candidate. The candidate lacks a clean non-oracle negative set:
remaining train0601 extras include safety-protected, near/non-far,
Hungarian-matched-skip, and noncandidate cases. A future candidate would first
need assignment-level evidence that separates harmful final extras from matched
or GT-close positives without query-wide q0/q10 suppression.

Assignment-level follow-up:

```text
artifact =
  .tmp/tiered_v2_assignment_blockers/train0601/
records = 410
query rows = 4920
per-query/GT rows = 21444
```

The unresolved outside blockers split by mechanism:

```text
q10 protected, GT4 4->6:
  clips/0601/1494453653536229036/20.jpg
  protected by protect_acc
  nearest GT lane/order = 3 / 1, APE 56.93 px
  best line_acc = 1.0
  nearest matched query = q8, distance 45.27 px

q10 protected, GT5 5->6:
  clips/0601/1495492674588939817/20.jpg
  protected by protect_px
  nearest GT lane/order = 4 / 3, APE 26.58 px
  best line_acc = 0.70
  nearest matched query = q11, distance 43.46 px

q10 near/non-far, GT4 4->6:
  clips/0601/1494453811467016035/20.jpg
  nearest GT lane/order = 3 / 3
  train-scale APE = 42.97 px, final-trace APE = 53.78 px
  selected=False because clear_far=False and duplicate_like=False

q0 Hungarian-matched skip, GT4 4->5:
  clips/0601/1495492704577874815/20.jpg
  Hungarian matched q0 -> GT0/order0, APE 10.18 px, line_acc 0.8333
  final trace labels q0 as extra
  final trace instead marks q3 as matched to GT0 with infinite APE
```

Integrated conclusion:

- q10 protected rows are GT-close or line-accurate enough that suppressing
  them as hard negatives would violate the current guard's purpose.
- q10 near/non-far is a distance-scale boundary case, not a clean far-extra
  example.
- q0 is a matcher-positive/final-greedy conflict, so unmatched-query BCE cannot
  target it without contradicting the training assignment.
- These three mechanisms do not share a safe non-oracle separating feature.

Next action:

Do not run TEST and do not launch training from this line. If this line
continues, the next work must be read-only: inspect q10 protected counterpart
assignments, q10 train-scale versus final-scale distance disagreement, and q0
Hungarian assignment versus final greedy assignment. Until a non-oracle
separating feature is found, env30 remains the active reference.

The earlier original-TuSimple-scale replay artifacts are superseded by this
training-scale replay and should not be treated as exact training-loss
selection evidence.

## 2026-07-16: Reject env30 query-level hard-negative direction

Decision:

Do not launch a new env30 hard-negative training run from the q0/q11 final-extra
signal. The assignment-blocker follow-up shows that the official-val q0/q11
extra pattern is real but not a clean train-only target, and q0/q11 also carry
substantial positive and tail-lane signal.

Protocol:

```text
diagnostic schema = assignment_blocker_diag_v1
TEST = not used
weights =
  runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt

official-val artifact =
  .tmp/env30_assignment_blockers/val/
train0601 artifact =
  .tmp/env30_assignment_blockers/train0601/
train0531 artifact =
  .tmp/env30_assignment_blockers/train0531/

train0531 command =
  python .tmp/diagnose_assignment_blockers.py \
    --split train \
    --gt-json archive/TUSimple/train_set/label_data_0531.json \
    --weights runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt \
    --final-trace runs/gcs_lane/env30_official_best_train0531_final_query_extra_valid_before_maxdet/final_lane_trace.csv \
    --imgsz 544 960 --device 0 --half \
    --save-dir .tmp/env30_assignment_blockers/train0531
```

Evidence:

```text
official-val:
  final extras = 7
  query hist = q0 4, q11 3
  transition = GT3 3->4: 7
  selected clear-far = 6 / 7
  blocker = 1 near/non-far q0 row

train0601:
  final extras = 8
  query hist = q1 4, q10 3, q11 1
  transitions = GT3 3->4: 1, GT4 4->5: 7
  selected clear-far = 8 / 8

train0531:
  final extras = 7
  query hist = q0 2, q1 1, q10 1, q11 3
  transitions = 2->3: 1, GT3 3->4: 4, GT4 4->5: 2
  selected clear-far = 4 / 7
  blockers = protected q10, near/non-far q0, Hungarian-matched q11
```

Positive-role conflict:

```text
official-val q0/q11:
  final extras = 7
  Hungarian positives = 490
  final matched positives = 489

train0601 q0/q11:
  final extras = 1
  Hungarian positives = 677
  final matched positives = 670

train0531 q0/q11:
  final extras = 5
  Hungarian positives = 408
  final matched positives = 405

official-impact tail lanes:
  q0/q11 raw-best = 6 / 15
```

Why:

- Official-val q0/q11 extras are only a small GT3 `3->4` tail, while q0/q11
  are hundreds of matched positives across val/train splits.
- Train0601 does not reproduce a q0/q11 carrier pattern; its harmful extras
  are mostly q1/q10 and GT4 `4->5`.
- Train0531 exposes blocker cases that are unsafe as target-zero negatives:
  a q10 row protected by `protect_px`, a q0 near/non-far row with high
  line-accuracy, and a q11 row that is Hungarian matched to a GT lane with
  line accuracy `1.0` while final greedy attribution labels it extra.
- Simple score or valid-length suppression does not separate extras from
  positives; it would remove many matched positives before removing all
  q0/q11 extras.
- Existing v2/q1q3 final-extra guard evidence already showed that broad guard
  pressure can still fail official-val gates, so expanding scope from this
  sparse env30 residual is not justified.

Next action:

Keep env30 as the active reference. Do not change loss, sampler, decode,
query references, guard scope, guard gain, `clear_far_px`, NMS, or thresholds
from this diagnostic. If this line continues, first produce a read-only
counterfactual table for any proposed non-oracle separator, including
selected-final-extra precision, matched/Hungarian positives selected,
official-impact tail rows touched, GT3/GT4/GT5 transition effects, and
train0601/train0531 disagreement. Do not train until that evidence reaches the
project's `>=90%` engineering-confidence gate.

Counterfactual follow-up:

```text
artifact =
  .tmp/env30_assignment_counterfactual/summary.md
csv =
  .tmp/env30_assignment_counterfactual/rule_counterfactual_summary.csv
examples =
  .tmp/env30_assignment_counterfactual/rule_counterfactual_examples.csv
```

The follow-up rejects all tested non-oracle separator families:

```text
selected_any:
  val touched/final-extra/precision = 11 / 6 / 0.545455
  train0601 touched/final-extra/precision = 16 / 8 / 0.500000
  train0601 GT5 5->5 final lanes touched = 6
  train0531 touched/final-extra/precision = 6 / 4 / 0.666667

selected_q0q11:
  val touched/final-extra/precision = 8 / 6 / 0.750000
  train0601 touched/final-extra/precision = 3 / 1 / 0.333333
  train0531 touched/final-extra/precision = 4 / 3 / 0.750000

querywide_q0q11:
  val extras removed = 7 / 7
  val Hungarian/final positives touched = 490 / 489
  val official-impact tail raw-best rows touched = 6 / 15
  train0601 extras removed = 1 / 8
  train0601 Hungarian/final positives touched = 677 / 670
  train0531 extras removed = 5 / 7
  train0531 Hungarian/final positives touched = 408 / 405

q0q11_score_below_0.93:
  val extras removed = 7 / 7
  val Hungarian/final positives touched = 85 / 84
  train0601 extras removed = 1 / 8
  train0601 Hungarian/final positives touched = 125 / 118

q0q11_valid_len_below_24:
  val extras removed = 7 / 7
  val Hungarian/final positives touched = 470 / 469
  train0601 extras removed = 1 / 8
  train0601 Hungarian/final positives touched = 657 / 650
```

Integrated conclusion:

- `selected_any` is not precise enough for a train-side hard-negative loss and
  touches GT5 `5->5` final lanes on train0601.
- q0/q11 query-wide suppression, score suppression, and valid-length
  suppression all remove too many matched positives relative to the sparse
  official-val extra-lane benefit.
- None of the tested rules reaches the `>=90%` engineering-confidence gate.

Final decision for this line:

Close the current env30 query-level hard-negative path. Do not train, do not
alter decode, and do not run TEST from these diagnostics. Any future return to
hard-negative training must first introduce a new separator and pass a fresh
official-val/train counterfactual gate that has materially higher precision
and no matched-positive or official-impact-tail collision.

## 2026-07-16: Keep env30 positive-tail rescue diagnostic-only

Decision:

Do not start a positive-rescue training run from the 15 env30 official-impact
tail lanes. The new positive-side triage shows mixed raw geometry, official
line-accuracy, assignment-scale, and point-valid/existence survival failures,
not a clean GT-count-specific or query-specific training target.

Protocol:

```text
TEST = not used
tail source =
  .tmp/env30_failure_classification/official_impact_tail_lanes.csv
assignment trace =
  .tmp/env30_assignment_blockers/val/assignment_query_trace.csv
per-query/GT metrics =
  .tmp/env30_assignment_blockers/val/assignment_query_gt_metrics.csv
triage artifact =
  .tmp/env30_tail_positive_triage/summary.md
triage csv =
  .tmp/env30_tail_positive_triage/official_impact_tail_triage.csv
```

Evidence:

```text
official-impact tail lanes = 15
no20 lanes = 13
valid<min lanes = 7
raw-best query hist =
  q0 3, q1 2, q2 1, q6 1, q7 2, q9 2, q10 1, q11 3

triage classes =
  raw_geometry_moderate_ape: 4
  raw_geometry_plus_valid: 4
  ultra_visible_le2_geometry_ambiguous: 3
  exist_score_and_valid_survival: 2
  raw_geometry_large_ape_plus_valid: 1
  raw_geometry_large_ape: 1
```

Assignment cross-check:

```text
raw-best query Hungarian matched = 13 / 15
raw-best query final_in_decode = 8 / 15
raw-best query final_greedy_matched_gt = 5 / 15
no20 and Hungarian matched = 12 / 13
no20 but assignment-trace APE <= 20 px = 5
no20 but assignment-trace line_acc >= 0.8 = 5
```

Why:

- The tail lanes are not a clean "missing positive assignment" set. Most
  raw-best queries are already matched by the training assignment diagnostic.
- The `no20` flag partly disagrees with assignment-scale APE/line-accuracy,
  so a loss built directly from these lanes would risk optimizing a
  scale/metric artifact.
- The failures span GT3/GT4/GT5 and center/side positions, so a GT5-only,
  side-only, or q0/q11-only positive rescue is unsupported.
- Only a small subset looks like pure existence/point-valid survival. That is
  not enough to justify a training run without a broader train-side denominator
  and an official-val gate.

Next action:

Keep this path read-only. If revisited, first build train-side denominators for
the same triage classes and require a candidate gate that improves raw
geometry, line accuracy, and valid survival without worsening GT3/GT4
over-count or GT5 under-count. Do not run TEST, do not tune thresholds, and do
not introduce a positive-rescue loss from the 15-lane official-val tail alone.

Train/val denominator follow-up:

```text
artifact =
  .tmp/env30_positive_denominator/summary.md
rows =
  .tmp/env30_positive_denominator/gt_lane_denominator.csv
summary =
  .tmp/env30_positive_denominator/split_summary.csv
```

Assignment best-query denominator:

```text
official-val:
  GT lanes = 1303
  best-query APE >20 / >30 = 26 / 15
  line_acc <0.8 = 12
  no Hungarian / no final match = 5 / 17

train0601:
  GT lanes = 1787
  best-query APE >20 / >30 = 57 / 25
  line_acc <0.8 = 14
  no Hungarian / no final match = 4 / 26

train0531:
  GT lanes = 1118
  best-query APE >20 / >30 = 11 / 6
  line_acc <0.8 = 6
  no Hungarian / no final match = 0 / 10
```

Metric-mismatch follow-up:

```text
artifact =
  .tmp/env30_tail_metric_mismatch/summary.md
csv =
  .tmp/env30_tail_metric_mismatch/tail_metric_mismatch.csv

raw-best query != assignment best-APE query = 9 / 15
raw-best query != assignment best-line-acc query = 9 / 15
tail valid<min but assignment best-query valid_len >= 3 = 7 / 15
tail no20 but assignment best-query APE <= 20 px = 2 / 15
tail raw-best query Hungarian matched = 13 / 15
assignment best-query Hungarian matched = 11 / 15
```

Updated decision:

Close direct positive-rescue training from the current official-impact tail
diagnostics. The denominator is broad and split-dependent, and the raw-tail
diagnostic target often disagrees with the assignment best-query target. A
future positive-rescue candidate must first define a consistent train-side
selection rule and prove, with official-val/train diagnostics before training,
that it does not mostly select already matched/near-ok positives and does not
increase GT3/GT4 over-count or GT5 under-count risk.

Selector counterfactual follow-up:

```text
artifact =
  .tmp/env30_positive_selector_counterfactual/summary.md
summary =
  .tmp/env30_positive_selector_counterfactual/selector_summary.csv
split details =
  .tmp/env30_positive_selector_counterfactual/selector_split_details.csv
tail coverage =
  .tmp/env30_positive_selector_counterfactual/selector_tail_coverage_details.csv
```

Key evidence:

```text
geom_gt20:
  val selected = 26
  tail covered = 12 / 15
  non-impact selected = 14
  final-matched positives touched = 15
  Hungarian positives touched = 21

geom_gt30:
  val selected = 15
  tail covered = 10 / 15
  non-impact selected = 5
  final-matched positives touched = 5
  Hungarian positives touched = 10

line_acc_lt08:
  val selected = 12
  tail covered = 10 / 15
  non-impact selected = 2
  final-matched positives touched = 2
  Hungarian positives touched = 9

no_final_match:
  val selected = 17
  tail covered = 12 / 15
  non-impact selected = 5
  final-matched positives touched = 0
  Hungarian positives touched = 14

geom20_or_line_and_no_final:
  val selected = 12
  tail covered = 12 / 15
  non-impact selected = 0
  final-matched positives touched = 0
  Hungarian positives touched = 9
```

Interpretation:

- Geometry-only selectors are too broad or miss too many official-impact tails.
- The best-looking `no_final_match` combinations are post-decode diagnostics,
  not train-side labels. They depend on final decoded match state and should
  not be converted directly into a loss target or selection policy.
- The split surfaces remain unstable. `geom_gt20` is GT5-heavy on train0601
  but GT3/GT4-only on train0531, and `no_final_match` includes
  `ok_or_near_ok` rows.
- No selector reaches the `>=90%` engineering-confidence gate for a training,
  decode, or TEST action.

Decision:

Keep positive-tail rescue diagnostic-only. Do not train from `geom`,
`line_acc`, `no_final_match`, or their current combinations. The next safe
action is still read-only: find a non-oracle separator between assignment-level
positive tails and decode/final-match artifacts before any implementation.

## 2026-07-16: Complete v2 q3/q4/q6/q7 final-query trace audit

Decision:

Keep `query_alpha05_env30_q12_ultrashort_dataref_geom1_extraguard_v2`
rejected and do not run TEST. The completed read-only trace audit is enough to
explain the failure direction, but it is not enough to justify a new
implementation, decode, loss, reference-bank, or training change under the
90% engineering-confidence gate.

Audited artifacts:

```text
v2 official-val =
  runs/gcs_lane/query_alpha05_env30_q12_ultrashort_dataref_geom1_extraguard_v2_official_best_val_final_query_extra_valid_before_maxdet
v2 train0601 =
  runs/gcs_lane/query_alpha05_env30_q12_ultrashort_dataref_geom1_extraguard_v2_official_best_train0601_final_query_extra_valid_before_maxdet
v2 train0531 =
  runs/gcs_lane/query_alpha05_env30_q12_ultrashort_dataref_geom1_extraguard_v2_official_best_train0531_final_query_extra_valid_before_maxdet
env30 official-val =
  runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1_official_best_val_final_query_extra_diag
env30 train0601 =
  runs/gcs_lane/env30_official_best_train0601_final_query_extra_valid_before_maxdet
env30 train0531 =
  runs/gcs_lane/env30_official_best_train0531_final_query_extra_valid_before_maxdet
```

Trace grouping result:

```text
v2 official-val:
images/over/under/extra/focus(q3,q4,q6,q7) = 363 / 22 / 3 / 23 / 15
focus by query = q3 6, q4 3, q6 5, q7 1
focus transitions = 3->4 2, 4->6 2, 5->6 11
focus class = clear_far 8, near_gt_nonduplicate 7, duplicate_like 0
focus score p50 = 0.259
focus valid_len p50 = 7
focus nearest-GT APE p50 = 56.700 px
focus nearest-pred distance p50 = 77.167 px

env30 official-val:
images/over/under/extra/focus(q3,q4,q6,q7) = 363 / 7 / 3 / 7 / 0
extra query hist = q0 4, q11 3

v2 train0601:
images/over/under/extra/focus(q3,q4,q6,q7) = 410 / 70 / 6 / 70 / 57
focus by query = q3 19, q4 9, q6 15, q7 14
focus transitions = 3->4 1, 4->5 10, 5->6 46
focus class = clear_far 28, near_gt_nonduplicate 29, duplicate_like 0
focus score p50 = 0.240
focus valid_len p50 = 6
focus nearest-GT APE p50 = 43.733 px

env30 train0601:
images/over/under/extra/focus(q3,q4,q6,q7) = 410 / 8 / 5 / 8 / 0

v2 train0531:
images/over/under/extra/focus(q3,q4,q6,q7) = 358 / 7 / 1 / 8 / 2
focus by query = q6 1, q7 1
focus transitions = 4->6 2
focus class = clear_far 2, duplicate_like 0
focus nearest-pred distance p50 = 34.429 px

env30 train0531:
images/over/under/extra/focus(q3,q4,q6,q7) = 358 / 7 / 1 / 7 / 0
```

Official-val sweep rescue check:

```text
v2 sweep rows = 864
rows with ACC >= env30 0.973330 = 0
rows passing count-shape gate
  (4->5=0, 4->6=0, 5->4<=1, 5->6=0, FN<=0.009642) = 0

best v2 row with 5->6=0:
ACC/FP/FN = 0.971993 / 0.017355 / 0.011938
decode = conf 0.001, point_valid_thr 0.6, nms_dist_px 18,
  max_det 5, min_points 5, valid_before_maxdet true
count shape = 4->5 2, 4->6 0, 5->4 2, 5->6 0

env30 under the v2 selected decode:
ACC/FP/FN = 0.972989 / 0.017677 / 0.011019
count shape = 4->5 1, 4->6 0, 5->4 1, 5->6 0
```

GT-based post-hoc lane-removal counterfactual:

```text
v2 official-val baseline:
over/under = 22 / 3
confusion = 3->3 216, 3->4 7, 4->3 1, 4->4 62,
            4->5 2, 4->6 1, 5->4 2, 5->5 60, 5->6 12

v2 official-val after removing only q3/q4/q6/q7 clear-far extras:
over/under = 15 / 3
confusion = 3->3 217, 3->4 6, 4->3 1, 4->4 63,
            4->5 2, 5->4 2, 5->5 65, 5->6 7

v2 official-val after removing all q3/q4/q6/q7 extras:
over/under = 8 / 3
confusion = 3->3 218, 3->4 5, 4->3 1, 4->4 63,
            4->5 2, 5->4 2, 5->5 71, 5->6 1

v2 train0601 baseline:
over/under = 70 / 6
confusion includes 4->5 15, 5->4 4, 5->6 53

v2 train0601 after removing only q3/q4/q6/q7 clear-far extras:
over/under = 42 / 6
confusion includes 4->5 10, 5->4 4, 5->6 30

v2 train0601 after removing all q3/q4/q6/q7 extras:
over/under = 13 / 6
confusion includes 4->5 5, 5->4 4, 5->6 7

env30 official-val/train0601 after removing all clear-far extras:
over = 0 on both diagnostics
```

This counterfactual uses GT-based post-decode attribution and is diagnostic
only. It does not define a valid decode or TEST-time behavior.

Why:

- The v2 failure is concentrated in GT4/GT5 over-count paths from
  q3/q4/q6/q7, especially GT5 `5->6`.
- Env30 does not show q3/q4/q6/q7 focus-query extras on official-val,
  train0601, or train0531 under the matching final-query trace diagnostics.
- The focus extras are mixed: roughly half clear-far and half
  near-GT/non-duplicate on official-val and train0601. Therefore a blind
  stronger `gcs_final_extra_guard` could suppress true short-lane-like
  candidates.
- Removing only clear-far focus extras would not restore v2's official-val or
  train0601 count shape. Removing all focus extras gets closer but would also
  remove near-GT non-duplicate lanes, which is too close to true short-lane
  suppression risk for a blind guard.
- Decode/sweep rescue is ruled out for this candidate by the official-val
  grid: no row matches env30 ACC, and no row passes the count-shape gate.
- Recent env30 count-restore and tiered GT4/GT5 rescue lines are already
  rejected, so they should not be revived from this evidence.

Candidate direction map:

```text
1. Read-only per-query trace audit: completed; diagnostic-only evidence.
2. Reference assignment revision for q3/q4/q6/q7: plausible but not yet
   >=90% supported; needs a concrete bank-change hypothesis and static/train
   gate before training.
3. Stronger final_extra_guard: currently risky because near-GT/protected
   extras are substantial, and clear-far-only removal is insufficient.
4. Decode/NMS/max_det rescue: invalid for v2; official-val sweep already
   rejects it.
5. Count-loss or broad geometry rescue: unsupported by recent rejected runs.
```

Next action:

Do not start training and do not run TEST. The next safe action is a smaller
official-val/train diagnostic that evaluates a concrete static bank/assignment
counterfactual before training: it must reduce q3/q4/q6/q7 clear-far extras
without increasing near-GT non-duplicate suppression or GT5 undercount. Any
future candidate must first pass official-val ACC/FN/count-shape and train0601
q4/focus-query gates before a reporting-only TEST run.

## 2026-07-16: Complete q3/q6 cached survival and q3 static counterfactual

Decision:

Do not suppress q3/q6/q7 as whole queries, and do not remove the q3 GT5
prototype blindly. The new evidence narrows the problem to q3-style GT5
prototype disambiguation after training, not static coverage alone.

Artifacts:

```text
cached q3/q6 survival =
  .tmp/final_query_trace_analysis/v2_q36_survival_summary.json
q3-default static bank/audit =
  .tmp/final_query_trace_analysis/q12_ultrashort_v2_restore_q3_default_bank.json
  .tmp/final_query_trace_analysis/q12_ultrashort_v2_restore_q3_default_audit.json
q3q4-default static bank/audit =
  .tmp/final_query_trace_analysis/q12_ultrashort_v2_restore_q3q4_default_bank.json
  .tmp/final_query_trace_analysis/q12_ultrashort_v2_restore_q3q4_default_audit.json
```

Cached official-val survival under the selected v2 decode
(`conf=0.001`, `point_valid_thr=0.55`, `nms_dist_px=30`, `max_det=6`,
`min_points=2`, `valid_before_maxdet=true`):

```text
GT5 q3: above_conf 26, after_nms 26, after_valid/final 9
GT5 q3 final lanes: matched 3, extra 6
GT5 q3 extras: clear_far 3, near_gt_nonduplicate 3

GT5 q6: above_conf 20, after_nms 18, after_valid/final 13
GT5 q6 final lanes: matched 11, extra 2
GT5 q6 extras: clear_far 1, near_gt_nonduplicate 1

GT5 q4 final lanes: matched 5, extra 3
GT5 q7 final lanes: matched 7, extra 0
```

Median GT5 final-lane score/valid split:

```text
q3 matched:     score 0.278, valid_len 7
q3 clear extra: score 0.288, valid_len 5
q3 near extra:  score 0.228, valid_len 6

q6 matched:     score 0.603, valid_len 11
q6 clear extra: score 0.081, valid_len 4
q6 near extra:  score 0.418, valid_len 9
```

Static counterfactual:

```text
v2 committed static audit:
hard new p50/p90/match20 = 21.000 / 43.636 / 0.400
gate.pass = true

restore q3 default:
hard new p50/p90/match20 = 33.378 / 95.883 / 0.200
gate.pass = false

restore q3/q4 default:
hard new p50/p90/match20 = 143.527 / 173.568 / 0.200
gate.pass = false
```

Why:

- q6 and q7 are mostly true GT5 final lanes on official-val. Query-wide
  suppression would directly damage GT5 retention.
- q3 is the riskiest focus query: on GT5 it produces more extra final lanes
  than matched lanes, and its matched/extra scores and valid lengths overlap.
  A simple score or valid-count threshold is unlikely to separate them.
- q3 is also needed for static hard-target coverage. Restoring q3 to the
  default reference fails the static gate, so removing the prototype is not a
  viable standalone fix.
- NMS is not the main q3 failure stage: GT5 q3 remains 26 after NMS and 9
  after valid-before-maxdet/top6.

Next action:

Do not train another dataref bank from this evidence. The next useful
diagnostic is q3-specific: compare q3 matched versus q3 extra images in train
and official-val for geometry, GT side/order, and nearest true-lane context,
then decide whether a train-only disambiguation term or a different q3
prototype source is supportable. Without that, there is still no >=90%
confidence implementation direction.

Follow-up q3-specific diagnostic completed:

```text
artifact =
  .tmp/v2_q3_matched_extra_split/summary.md
rows =
  .tmp/v2_q3_matched_extra_split/q3_final_lane_rows.csv
summary =
  .tmp/v2_q3_matched_extra_split/q3_summary.csv
feature overlap =
  .tmp/v2_q3_matched_extra_split/q3_non_gt_feature_overlap.csv
```

GT5 q3 final-lane split:

```text
official-val:
  matched = 3
  clear-far extras = 3
  near-GT non-duplicate extras = 3
  score / valid_len / rank ranges overlap between matched and extras

train0601:
  matched = 10
  clear-far extras = 13
  near-GT non-duplicate extras = 4
  score / valid_len / rank ranges overlap between matched and extras

train0531:
  no GT5 q3 final rows in this trace
```

Updated decision:

This diagnostic still does not reach implementation confidence. q3 query-wide
suppression would remove true matched GT5 lanes; q3 score, valid length, and
final score rank do not separate matched from extra rows; and GT-based
far/near attribution is not a valid inference/decode rule. Do not start a new
q3 dataref, q3 suppression, score/valid/rank threshold, or train-only
disambiguation term from the current evidence.

## 2026-07-16: Add q1/q3-protected static bank as diagnostic candidate

Decision:

Keep q1/q3-protected dataref as the current best static diagnostic candidate,
but do not promote it to TEST and do not claim it is likely enough to train
without explicit official-val gates. It is a controlled answer to the q3
failure mode: preserve q1/q3 default query behavior and move the two GT5
ultrashort prototypes to q4/q5.

Scratch artifacts:

```text
q3-protected bank =
  .tmp/final_query_trace_analysis/q12_ultrashort_q3protected_static_bank.json
q3-protected summary/audit =
  .tmp/final_query_trace_analysis/q12_ultrashort_q3protected_static_summary.json
  .tmp/final_query_trace_analysis/q12_ultrashort_q3protected_static_coverage_audit.json

q1+q3-protected bank =
  .tmp/final_query_trace_analysis/q12_ultrashort_q1q3protected_static_bank.json
q1+q3-protected summary/audit =
  .tmp/final_query_trace_analysis/q12_ultrashort_q1q3protected_static_summary.json
  .tmp/final_query_trace_analysis/q12_ultrashort_q1q3protected_static_coverage_audit.json
```

Static results:

```text
v2 committed:
selected qset = q3/q4/q6/q7/q8/q10
hard new p50/p90/match20 = 21.000 / 43.636 / 0.400
normal delta p50/p90 = -4.046 / +0.557
gate.pass = true

q3-protected:
selected qset = q1/q4/q5/q7/q8/q10
hard new p50/p90/match20 = 22.222 / 46.000 / 0.333
normal delta p50/p90 = -4.324 / -1.483
gate.pass = true

q1+q3-protected:
selected qset = q4/q5/q6/q7/q8/q10
hard new p50/p90/match20 = 24.000 / 46.000 / 0.333
normal delta p50/p90 = -6.013 / -2.425
gate.pass = true
```

q1+q3-protected assignment:

```text
q4 = gt5_vle10_center_left, source clips/0601/1495485255533685610/20.jpg lane0
q5 = gt5_vle10_center_left, source clips/0601/1494452949844803933/20.jpg lane0
q6 = gt4_vle10_right_side, source clips/0313-1/6400/20.jpg lane3
q7 = gt4_vle10_center, source clips/0313-1/49920/20.jpg lane1
q8 = gt4_vle10_center, source clips/0601/1495058805523198969/20.jpg lane3
q10 = gt4_vle10_right_side, source clips/0313-2/480/20.jpg lane3
```

Why q1+q3 protection is stronger than the previous focus-penalty bank:

- Env30 q3 is a useful default query: official-val q3 final/matched/extra is
  `9/8/0`, and train0601 is `28/26/0`.
- V2 changed q3 into a GT5 prototype and turned it into official-val
  `9/3/6` and train0601 `29/10/19`.
- V1 had already shown q1 as a risky GT5 prototype carrier
  (`q1` official-val extra `6`), so the q3-protected-but-q1-replaced bank is
  not enough.
- q1+q3-protected still passes the static gate while preserving both default
  q1 and default q3.

Residual risk:

- The q1+q3-protected static assignment moves the GT5 prototypes to q4/q5.
  There is no trained run with exactly this assignment.
- Historical proxy over the selected qset `q4/q5/q6/q7/q8/q10` still shows
  extra risk in neighboring trained runs: geom1 v1 proxy extra `10`
  (`GT4=4`, `GT5=5`) and v2 proxy extra `10` (`GT4=2`, `GT5=6`).
- Therefore static coverage alone is not a promotion signal. A future run must
  be treated as an official-val/train diagnostic candidate with early stop
  gates, not as a likely final candidate.

Next action:

If launching one new controlled training diagnostic, prefer the q1+q3-protected
bank over q3-protected or focus-penalty banks. Before launch, create a reviewed
default-off YAML/script pair that points the model to the matching bank and
records the static audit hash. Run only on remote, keep `RUN_TESTS=0`, and gate
on official-val ACC/FN/count-shape plus train0601 final-query extras. If q4/q5
become the new GT5 sixth-lane sources, stop early and reject the line.

Implementation follow-up:

The q1+q3-protected candidate has been materialized as default-off local
artifacts:

```text
model =
  ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-ultrashort-dataref-q1q3protected-v1.yaml
bank =
  data/gcs_reference_banks/q12_ultrashort_env30_train_q1q3protected_v1.json
audit =
  data/gcs_reference_banks/q12_ultrashort_env30_train_q1q3protected_v1_reference_audit.json
script =
  scripts/run_query_alpha05_env30_q12_ultrashort_q1q3protected_dataref_geom1_extraguard_v1.sh
```

Local validation completed:

```text
tools/check_gcs_reference_bank_coverage.py -> gate.pass=true
tools/check_model.py --cfg ...q1q3protected-v1.yaml --imgsz 544 960 --batch 1 --device cpu
  pred_points = [1, 12, 56, 2]
  pred_logits = [1, 12]
  pred_valid_logits = [1, 12, 56]
  aux mask/edge present
  no pred_count_logits
python -m py_compile tools/check_gcs_final_query_extra_gate.py
bash -n scripts/run_query_alpha05_env30_q12_ultrashort_q1q3protected_dataref_geom1_extraguard_v1.sh
PREFLIGHT_ONLY=1 ENFORCE_PRETRAINED_GATE=0 ... -> local dry-run preflight passed
```

The local dry-run bypassed `ENFORCE_PRETRAINED_GATE` only because the Windows
workstation does not have `yolo11s-seg.pt` in the repo root. Remote preflight
must keep the default `ENFORCE_PRETRAINED_GATE=1` and prove the pretrained
weight exists before training. The script defaults remain `RUN_TESTS=0`,
`ENFORCE_REFERENCE_GATE=1`, `ENFORCE_PRETRAINED_GATE=1`, and
`ENFORCE_FINAL_QUERY_EXTRA_GATE=1`.

## 2026-07-16: Reject focus-query penalty static bank as training trigger

Decision:

Do not train the scratch focus-query penalty bank. It is useful as a static
counterfactual, but it does not clear the 90% engineering-confidence gate for
a new remote training run.

Scratch artifacts:

```text
bank =
  .tmp/final_query_trace_analysis/q12_ultrashort_focus_penalty_static_bank.json
build summary =
  .tmp/final_query_trace_analysis/q12_ultrashort_focus_penalty_static_summary.json
standalone coverage audit =
  .tmp/final_query_trace_analysis/q12_ultrashort_focus_penalty_static_coverage_audit.json
query_static_penalty = 3:1000,4:1000,6:1000,7:1000,10:2
selected qset = q1/q3/q5/q6/q8/q10
official-val static gate.pass = true
official-val hard new p50/p90/match20 = 21.000 / 43.636 / 0.400000
official-val normal delta p50/p90 = -3.255 / +1.668
```

Why:

- The penalty moved q4/q7 out of the selected replacement set, but q3/q6
  remained selected. In v2, q3/q6 account for most focus-query official-val and
  train0601 extras.
- The nearest historical trained neighbor, geom1 v1, already used a similar
  q1/q3/q5/q6/q7/q8 replacement family and still produced 23 official-val
  final-query extras, the same total as v2.
- Static coverage gates measure reference coverage, not trained final-lane
  survival. Both v1 and v2 passed static gates before failing trained
  official-val/count-shape gates.

Next action:

Do not promote this scratch bank into a script/YAML or remote training run. A
useful next diagnostic must target why q3/q6 survive as GT5/GT4 final extras
after training, or must identify a bank assignment that removes q3/q6 risk
while preserving official-val hard-target coverage and normal-lane regression.

## 2026-07-16: Reject env30 count-restore ablation at epoch20

Decision:

Stop and reject
`query_alpha05_gt5short_geom_w2_bneg002_env30_countrestore_v1` before TEST.
The run was a constrained official-val-only ablation on top of the env30
reference. It changed only the count-loss gains:

```text
gcs_count = 0.3
gcs_count_under5 = 0.3
gcs_count_boundary = 0.2
gcs_query_count_ce = 0.0
```

Protocol:

```text
run = query_alpha05_gt5short_geom_w2_bneg002_env30_countrestore_v1
base = env30 protocol with GT5 short geometry and boundary pseudo envelope
RUN_TESTS = none
official-val GT = canonical 363-image official-val JSON
process = TERM after 22 complete results.csv rows, while epoch23 was running
primary checkpoint = weights/official_best.pt
official_best source_epoch = 20
selected decode = conf 0.001, point_valid_thr 0.6, nms_dist_px 30,
  max_det 5, min_points 5, valid_before_maxdet true
TEST artifacts = none
```

Official-val evidence:

```text
countrestore epoch005 ACC/FP/FN = 0.894234 / 0.295225 / 0.198118
countrestore epoch010 ACC/FP/FN = 0.942089 / 0.120018 / 0.060606
countrestore epoch015 ACC/FP/FN = 0.944528 / 0.111203 / 0.064050
countrestore epoch020 ACC/FP/FN = 0.944877 / 0.096740 / 0.063131

env30 epoch020 ACC/FP/FN = 0.953702 / 0.077410 / 0.045684
env30 final official_best val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
```

Count-shape evidence:

```text
countrestore epoch020 selected row:
count_acc_3/4/5 = 0.865471 / 0.833333 / 0.918919
GT3 3->4/3->5 = 28 / 2
GT4 4->3/4->5 = 1 / 10
GT5 5->4/5->5 = 6 / 68

countrestore epoch020 best max_det=6 row:
ACC/FP/FN = 0.942369 / 0.108907 / 0.064279
GT4 4->5 = 12
GT5 5->4/5->6 = 4 / 44

countrestore epoch020 sweep rows:
rows = 864
rows with ACC >= 0.94 = 618
minimum GT4 4->5 among ACC >= 0.94 rows = 8
max_det=6 rows with ACC >= 0.94 = 192
minimum GT5 5->6 among those max_det=6 rows = 31

env30 final official_best val:
count_acc_3/4/5 = 0.968610 / 0.969697 / 0.986486
GT4 4->3/4->5 = 2 / 0
GT5 5->4/5->5 = 1 / 73
```

Why:

- The run stayed far below env30 on the primary official-val metrics through
  epoch20. It was `-0.008825` ACC behind env30 at the same epoch and
  `-0.028453` behind the final env30 reference.
- FP and FN were both worse than env30 at epoch20.
- The selected `max_det=5` row hides part of the overcount surface. The
  `max_det=6` rows still expose severe GT5 sixth-lane survival, and no
  high-ACC row removes GT4 false fifth lanes.
- The run is not a crash or nonfinite-loss failure. It is an experiment
  direction failure for count restoration on this env30 base.

Next action:

Do not run TEST, do not promote this checkpoint, and do not tune count-loss
weights from this failed line. Return to env30 as the reference. Because no
new code/config/training direction has >=90% support from this result, the
next safe action is diagnostic: inspect official-val/train final-query traces
or raw-Q12 short-lane geometry/validity evidence before proposing another
implementation change.

## 2026-07-16: Clarify final-query extra count semantics and env30 count/geo direction

Decision:

Keep the Q12 ultrashort dataref v1/v2 rejection conclusions, but correct the
quantity semantics used when interpreting final-query extra-lane diagnostics.
Do not treat `extra_topk_candidate`, `extra_topk_lanes`, or per-query extra
histograms as exact official false positives.

Metric semantics:

```text
official over-image count = images where decoded pred_lanes > gt_lanes
official_FP = TuSimple official metric false-positive rate
extra_topk_candidate / extra_topk_lanes = post-decode GT diagnostic attribution
nearest-query histogram = diagnostic source attribution, not official FP
```

This correction changes the explanation precision, not the experiment
decision. The v1/v2 failures remain real because both official-val and
reporting-only TEST evidence show worse count shape than the env30 reference.

Key reporting-only TEST comparison:

```text
geom1 v1 official_best TEST ACC/FP/FN = 0.966130 / 0.038845 / 0.024203
geom1 v1 official over_images = 286, under_images = 147
geom1 v1 diagnostic extra_topk = 316
geom1 v1 extra query hist = q6 78, q5 74, q4 59, q1 34,
                            q11 28, q3 19, q7 11, q8 8
geom1 v1 GT5 5->6 extra hist = q6 29, q1 25, q5 20,
                                q3 10, q8 8, q4 6, q11 5, q7 3
geom1 v1 GT4 4->5 extra hist = q4 45, q5 24, q6 19,
                                q11 8, q1 2, q7 1

env30 official_best TEST ACC/FP/FN = 0.966780 / 0.028732 / 0.023544
env30 official over_images = 151, under_images = 163
env30 diagnostic extra_topk = 160
env30 extra query hist = q10 61, q1 47, q11 18, q9 12,
                         q3 6, q4 5, q0 4, q8 4, q2 3
env30 GT5 5->6 = 0
```

Official-val final-query diagnostic comparison:

```text
geom1 v1 official-val over_images = 22, under_images = 2
geom1 v1 official-val diagnostic extra_topk = 23
geom1 v1 official-val GT5 5->6 extra = 13
geom1 v1 GT5 5->6 extra hist = q1 6, q3 2, q5 2, q8 2, q6 1

env30 official-val over_images = 7, under_images = 3
env30 official-val diagnostic extra_topk = 7
env30 official-val GT4 4->5 = 0
env30 official-val GT5 5->6 = 0
```

Env30 count/geo conclusion:

- The env30 reference already includes the narrow GT5 short geometry loss:
  `gcs_short_geom=1.0` and `gcs_short_geom_gt5_weight=2.0`.
- Therefore "turning on geo like env30" is not a missing switch for the
  current env30 reference. The failed ultrashort dataref geom runs also already
  restored the intended geometry pressure.
- Count is the only plausible small ablation to revisit from the env30
  reference, because env30 kept `gcs_count=0.0`,
  `gcs_count_under5=0.0`, `gcs_count_boundary=0.0`, and
  `gcs_query_count_ce=0.0`.
- Count restoration must be an official-val-only ablation. Do not choose it
  from TEST and do not add GT4 point-valid rescue or broader tiered GT4/GT5
  geometry at the same time.

If this line is revisited, use a constrained env30 count-restore ablation:

```text
base = env30 protocol
keep current GT5 geometry and env30 boundary mask
set gcs_count = 0.3
set gcs_count_under5 = 0.3
set gcs_count_boundary = 0.2
keep gcs_query_count_ce = 0.0
do not add GT4 point-valid rescue
do not broaden tiered GT4/GT5 geometry
RUN_TESTS = 0
```

Gate:

```text
official-val ACC >= env30 0.973330
official-val FN <= env30 0.009642
official-val GT4 4->5 = 0
official-val GT5 5->4 <= 1
official-val GT5 5->6 = 0 under max_det=6 rows
raw-Q12 short GT4/GT5 geometry and point-valid survival do not regress
```

Why:

The TEST target remains higher ACC, but TEST is only reporting/failure
localization after official-val selection. The actionable bottleneck is not a
single final-query extra count number; it is the combined pattern of higher
official over-image count, higher official FP, and worse GT4/GT5 count shape
when ultrashort dataref prototypes survive as extra lane sources.

## 2026-07-16: Reject Q12 ultrashort dataref v2 after final-query extra gate

Decision:

Do not promote
`query_alpha05_env30_q12_ultrashort_dataref_geom1_extraguard_v2`, and do not
run TEST for this candidate. The run trained normally and the static reference
gate passed, but the post-train official-val/train final-query extra gate
failed.

Protocol:

```text
run = query_alpha05_env30_q12_ultrashort_dataref_geom1_extraguard_v2
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-ultrashort-dataref-v2.yaml
bank = data/gcs_reference_banks/q12_ultrashort_env30_train_v2.json
training stopped at epoch 154
primary checkpoint = weights/official_best.pt
selected decode = conf 0.001, point_valid_thr 0.55, nms_dist_px 30,
  max_det 6, min_points 2, valid_before_maxdet true
```

Training health:

```text
results.csv rows = 154
nonfinite_count = 0
final_extra_guard train side active, validation side zero
TEST artifacts = none
```

Official-val evidence:

```text
v2 official_best val ACC/FP/FN = 0.972040 / 0.023691 / 0.011938
v2 count_acc_3/4/5 = 0.968610 / 0.939394 / 0.810811
v2 count_confusion = 3->3 216, 3->4 7,
                     4->3 1, 4->4 62, 4->5 2, 4->6 1,
                     5->4 2, 5->5 60, 5->6 12

env30 reference val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
env30 count_acc_3/4/5 = 0.968610 / 0.969697 / 0.986486
env30 count_confusion = 3->3 216, 3->4 7,
                        4->3 2, 4->4 64,
                        5->4 1, 5->5 73
```

The companion `weights/best.pt` post-train official-val sweep is much weaker:

```text
best.pt val ACC/FP/FN = 0.966947 / 0.053444 / 0.018136
best.pt count_acc_3/4/5 = 0.905830 / 0.803030 / 0.635135
best.pt count_confusion includes 4->5 9, 4->6 3, 5->6 26
```

Final-query extra gate result:

```text
gate = runs/gcs_lane/query_alpha05_env30_q12_ultrashort_dataref_geom1_extraguard_v2_final_query_extra_gate.json
gate.pass = false
official-val over_images = 22 vs env30 7, limit 10
official-val GT5 5->6 extra = 12, limit 0
official-val GT4 4->6 extra = 2, limit 0
official-val GT4 4->5 extra = 2 vs env30 0
official-val focus-query GT5 5->6 extra for q1/q3/q5/q6/q8 = 8, limit 0
official-val q5+q6 total extra = 5, limit 2
train0601 q4 far extra = 4 vs env30 0
```

Final-query extra summaries:

```text
candidate official-val over_images = 22, under_images = 3
candidate official-val extra_topk_lanes = 23
candidate official-val greedy_unmatched_over_lanes = 26
candidate official-val extra_topk_nearest_gt_ape_p50/p90 = 66.133331 / 243.465712

env30 official-val over_images = 7, under_images = 3
env30 official-val extra_topk_lanes = 7
env30 official-val greedy_unmatched_over_lanes = 7
```

Why:

- The v2 static bank improves reference coverage, but the trained candidate
  does not preserve env30's official-val count shape.
- The small `gcs_final_extra_guard=0.02` term is active and train-only, but it
  is not strong enough to stop new final-query extra sources.
- The failure is not a training instability or data issue: the run has no
  nonfinite loss values, the selected checkpoint is from the official-best
  protocol, and the final-query gate correctly prevented TEST.
- The main official-val failure sources are extra lanes from the replacement
  query set, especially q3/q4/q6/q7 in the final-query diagnostics, with q6
  and the focus-query set violating the gate.

Next action:

Stop this candidate at official-val/train diagnostics. Do not run TEST and do
not replace env30 with v2. The smallest safe follow-up is a read-only analysis
of the official-best `final_lane_trace.csv` and `extra_image_trace.csv` for
q3/q4/q6/q7 extra lanes, to decide whether the next attempt should revise the
reference assignment, reduce the risky q4/q6/q7 replacement pressure, or
change the train-only guard. Do not make a blind loss-weight increase without
that per-query trace evidence.

## 2026-07-15: Add Q12 ultrashort dataref v2 and pre-TEST final-query extra gate

Decision:

Add a default-off `Q12-dataref-ultrashort-v2` candidate that lets `q4` enter
the static reference assignment search, then protect TEST by requiring
official-val/train final-query extra gates before any reporting-only TEST run.

Implementation scope:

- `tools/build_gcs_ultrashort_reference_bank.py` now supports
  `--required-replace-queries` and `--query-static-penalty`; the default
  replace set is `1,3,4,5,6,7,8,10`, with `4:0,10:2` static penalties.
- New bank:
  `data/gcs_reference_banks/q12_ultrashort_env30_train_v2.json`.
- New canonical static audit:
  `data/gcs_reference_banks/q12_ultrashort_env30_train_v2_reference_audit.json`.
- Generated run-copy static audit:
  `runs/gcs_lane/q12_ultrashort_env30_train_v2_reference_audit.json`.
- New YAML:
  `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-ultrashort-dataref-v2.yaml`.
- New final-query extra gate checker:
  `tools/check_gcs_final_query_extra_gate.py`.
- New launch script:
  `scripts/run_query_alpha05_env30_q12_ultrashort_dataref_geom1_extraguard_v2.sh`.
- New default-off train-only loss term:
  `gcs_final_extra_guard`.

Static audit result:

```text
bank_sha256 = 873af15e0f4e25b7429390fad0730628d8411b5077df74367838521a83e372ba
selected replacement queries = q3/q4/q6/q7/q8/q10
hard target base p50/p90/match20 = 147.049 / 199.196 / 0.133333
hard target new  p50/p90/match20 = 21.000 / 43.636 / 0.400000
normal lane p50 delta = -4.046
normal lane p90 delta = +0.557
gate.pass = true
```

Protocol:

- v2a (`q4` optional) and v2b (`q4` required) converged to the same selected
  mapping and both passed the static gate.
- Prototype shapes still come only from train diagnostics and train TuSimple
  JSON; official-val is used only for static assignment/audit.
- The final-query extra checker reads only `final_query_extra_summary.json`
  from official-val/train diagnostics. It rejects TEST summaries by default
  because those diagnostics use GT after decode.
- `gcs_final_extra_guard` is training-only, default-off, and uses GT only
  inside loss construction. It selects only unmatched query logits, protects
  true-lane-like candidates before clear-far/duplicate selection, and must not
  be moved into inference or TEST decode.

Next action:

Run the v2 script remotely. Keep `RUN_TESTS=0` until the static gate,
official-best/best official-val sweeps, official-val final-query extra
diagnostics, train0601/train0531 q4 diagnostics, and
`tools/check_gcs_final_query_extra_gate.py` all pass. If the gate fails, do not
run TEST.

## 2026-07-15: Regenerate Q12 ultrashort dataref bank and pass static gate

Decision:

Add a default-off `Q12-dataref-ultrashort-v1` candidate that changes only Q12
fixed-y `point_reference_logits` initialization from a small controlled
reference-bank JSON. Do not train it unless the static official-val coverage
gate passes.

Implementation scope:

- New builder:
  `tools/build_gcs_ultrashort_reference_bank.py`.
- New audit:
  `tools/check_gcs_reference_bank_coverage.py`.
- New controlled bank:
  `data/gcs_reference_banks/q12_ultrashort_env30_train_v1.json`.
- New model YAML:
  `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-ultrashort-dataref.yaml`.
- New launch script:
  `scripts/run_query_alpha05_env30_q12_ultrashort_dataref_v1.sh`.
- `GCSLaneHead` accepts `reference_mode="default|ultrashort_dataref"` and
  `reference_bank_path` after `query_count_head`. The old default path remains
  unchanged.

Protocol:

- Prototype shapes come from train hard lanes only.
- Official-val diagnostics are static gate / assignment selection only; no
  official-val lane geometry is used as a prototype.
- Hard lane filtering uses `raw_has_match_20px is False` OR
  `raw_best_valid_len@0.6 < min_points` OR
  `point_valid_recall@0.6 < 0.75`.
- `raw_best_valid_count@0.6` remains diagnostic only; contiguous
  `raw_best_valid_len@0.6` is the survival field.
- The mode name is `ultrashort_dataref`; do not add a generic `dataref` alias
  because legacy Q20/dataref records have a different meaning.
- No loss, matcher, query-binding cost, point-valid rescue, decode, Count
  Head, Quality Head, Survival Head, or Q20/Q24 change is part of this
  candidate.

Static audit result after the first small-pool v1 attempt:

```text
audit = runs/gcs_lane/q12_ultrashort_env30_train_v1_val_reference_audit.json
hard target base p50/p90/match20 = 147.049 / 199.196 / 0.133333
hard target new  p50/p90/match20 = 37.100 / 50.767 / 0.333333
normal lane p50 delta = +3.502
gate.pass = false
```

That first bank was not trainable evidence. It used too small a medoid pool and
left hard p50/p90 outside the required static gate.

Regenerated static audit result:

```text
bank = data/gcs_reference_banks/q12_ultrashort_env30_train_v1.json
bank_sha256 = e2c7afaa367983619f8f499033ddf92f47c7bab01a35a52c40908d03f99485d8
audit = runs/gcs_lane/q12_ultrashort_env30_train_v1_val_reference_audit.json
hard target base p50/p90/match20 = 147.049 / 199.196 / 0.133333
hard target new  p50/p90/match20 = 20.125 / 43.636 / 0.466667
normal lane p50 delta = -2.400
normal lane p90 delta = +4.324
gate.pass = true
```

Why:

The builder now enumerates all train-hard prototype pairs for the fixed
`2/2/2` group allocation, shortlists official-val hard-gate candidates, then
fully audits normal-lane regression. Prototype shapes still come only from
train hard lanes. The selected mapping replaces `q1/q3/q5/q6/q7/q8` and keeps
`q0/q2/q9/q10/q11`; q10 was avoided because equivalent static-gate candidates
exist without changing it.

Next action:

This bank is eligible for the first Q12-dataref-ultrashort training run through
`scripts/run_query_alpha05_env30_q12_ultrashort_dataref_v1.sh`, which enforces
the tracked static audit hash, audit schema, canonical GT path, and gate before
launching training. The training run still must start from `yolo11s-seg.pt` and
must not enable loss, matcher, query binding, point-valid rescue, decode, Count
Head, Q20, or Q24 changes.

## 2026-07-15: Add experiment recommendation confidence gate

Decision:

After experiment-result analysis, future agents must recommend code/config,
training, decode, loss, sampler, or architecture changes only when the evidence
supports the direction with at least 90% engineering confidence.

Protocol:

- The confidence must be grounded in comparable experiment results,
  official-val evidence, train/val diagnostics, raw-candidate metrics,
  count-confusion tables, or failure traces.
- If confidence is below 90%, or the available experiment conclusion is
  insufficient to isolate the cause, the next recommendation must be a
  validation experiment or diagnostic, not an implementation change.
- The proposed validation must name the uncertainty it resolves, the metrics to
  inspect, the result that would justify the later modification, and the result
  that would reject that direction.

Why:

The experiment chains recorded through 2026-07-14 include many narrow tradeoffs
where a plausible next tweak can be wrong without targeted evidence. This rule
prevents blind loss-weight, decode, sampler, or architecture changes and keeps
every recommended modification tied to experiment support.

## 2026-07-14: Reject tiered GT4/GT5 + GT4PV full-protocol v2

Decision:

Do not promote
`query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2`.
The run used the intended env30 mask, tiered geometry rescue, and GT4/GT5
point-valid rescue parameters, but it still fails the official-val promotion
gate and does not improve reporting-only TEST ACC versus env30.

Protocol:

```text
run = query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2
primary checkpoint = weights/official_best.pt
selected val decode = conf 0.001, point_valid_thr 0.6, nms_dist_px 30,
  max_det 6, min_points 2, valid_before_maxdet true
```

Official-val evidence:

```text
v2 official_best val ACC/FP/FN = 0.973049 / 0.014141 / 0.008724
count_acc_3/4/5 = 0.982063 / 0.939394 / 0.918919
count_confusion = 3->3 219, 3->4 4,
                  4->3 2, 4->4 62, 4->5 2,
                  5->4 1, 5->5 68, 5->6 5

env30 official_best val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
count_acc_3/4/5 = 0.968610 / 0.969697 / 0.986486
count_confusion = 3->3 216, 3->4 7,
                  4->3 2, 4->4 64,
                  5->4 1, 5->5 73
```

Across the 864-row v2 official-val sweep, no row reaches env30's
`ACC=0.973330`. Rows with `ACC >= 0.973` exist, but none removes GT4 `4->5`;
the best `5->6=0` rows still keep GT4 `4->5=2` and stay below env30 ACC.

Reporting-only TEST evidence:

```text
v2 official_best test ACC/FP/FN = 0.966657 / 0.030799 / 0.024203
count_acc_3/4/5 = 0.967816 / 0.600427 / 0.775044
count_confusion includes 3->4 46, 3->5 6,
                         4->3 103, 4->5 76, 4->6 8,
                         5->3 15, 5->4 63, 5->6 50

env30 official_best test ACC/FP/FN = 0.966780 / 0.028732 / 0.023544
count_acc_3/4/5 = 0.964943 / 0.598291 / 0.891037
count_confusion includes 3->4 50, 3->5 8,
                         4->3 98, 4->5 90,
                         5->3 19, 5->4 43, 5->5 507
```

Why:

- v2 is not a parameter or script failure; the intended new parameters were
  active. It is an algorithm-direction failure for this loss-weighting line.
- The official-val row improves FP/FN slightly versus env30 but loses primary
  ACC and count shape. GT4 `4->5` regresses from `0` to `2`, and GT5 `5->6`
  regresses from `0` to `5`.
- TEST remains below the target `0.97`; the gap is not explained by threshold
  choice. The raw-Q12 diagnostic still shows weak short-visible true-lane
  candidates: TEST GT4 short p90 APE is `46.765210 px` with match20
  `0.456897`, and TEST GT5 short p90 APE is `53.893844 px` with match20
  `0.658996`.
- Candidate/filter trace reports `180` TEST images where the candidate count
  is already below GT count. This cannot be fixed by a different final
  `conf`, NMS, `max_det`, or `min_points` row.

Next action:

Use env30 as the reference and stop increasing GT4 rescue, GT4 point-valid,
GT5 mid-short rescue, or boundary pseudo-negative pressure. The next serious
direction should be hardset-driven candidate-geometry coverage for train/val
short GT4/GT5 lanes, not another small weight bump. Any new structure must
first improve raw candidate recall/APE and point-valid survival on official-val
diagnostics before a reporting-only TEST run.

## 2026-07-13: Reject tiered GT4/GT5 + GT4PV env30 full-protocol result

Decision:

Do not promote
`query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v1`,
and do not continue by increasing GT4/GT5 weak-positive weights. Use the
previous env30 mask-v2 result as the stronger reference for this line.

Protocol:

```text
run = query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v1
primary checkpoint = weights/official_best.pt
source_epoch = 202
val/test decode = conf 0.001, point_valid_thr 0.55, nms_dist_px 0,
  max_det 5, min_points 2, valid_before_maxdet true, count_mode score_sum
```

Official-val evidence:

```text
tiered official_best val ACC/FP/FN = 0.971182 / 0.023370 / 0.012397
count_acc_3/4/5 = 0.946188 / 0.939394 / 0.986486
count_confusion = 3->3 211, 3->4 12, 4->4 62, 4->5 4,
                  5->4 1, 5->5 73

env30 official_best val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
count_confusion = 3->3 216, 3->4 7, 4->3 2, 4->4 64,
                  5->4 1, 5->5 73
```

Reporting-only TEST evidence:

```text
tiered official_best test ACC/FP/FN = 0.966118 / 0.034280 / 0.023934
count_acc_3/4/5 = 0.958621 / 0.568376 / 0.887522
count_confusion includes 3->5 20, 4->3 87, 4->5 114,
                         5->3 17, 5->4 47, 5->5 505

env30 official_best test ACC/FP/FN = 0.966780 / 0.028732 / 0.023544
count_confusion includes 3->5 8, 4->3 98, 4->5 90,
                         5->3 19, 5->4 43, 5->5 507
```

Why:

- The run fails before TEST: no official_best val sweep row reaches
  `ACC >= 0.973`, and no row simultaneously matches env30's FP/FN ceiling.
- GT4 point-valid rescue plus stronger tiered geometry reduces some GT4
  undercount but turns it into GT3/GT4 overcount. On TEST, GT4 `4->5` grows
  from env30's `90` to `114`, and GT3 `3->5` grows from `8` to `20`.
- The intended short-GT5 geometry rescue did not work. Official-val
  GT5 visible<=10 raw p90 APE worsens from env30's `36.550196 px` to
  `51.347903 px`, with `has_match20` stuck at `0.754717`.
- TEST raw-Q12 diagnostics show the remaining bottleneck is still true short
  GT4/GT5 candidate quality, especially `0601|GT5` missing lanes and
  `0530/0601|GT4` short/weak lanes, while overcount pressure increased.

Next action:

Do not tune thresholds, max-det, min-points, NMS, or TEST settings from this
result. The next experiment should start from env30, remove GT4 point-valid
rescue, avoid GT4 mid/ultra score inflation, and use a smaller GT5-only
ultra-short geometry/valid rescue only if paired with an explicit official-val
overcount gate. A stronger direction is hardset-driven candidate-geometry
coverage for train/val short GT4/GT5 lanes rather than broader loss weights.

## 2026-07-12: Move from env30 boundary mask to GT4/GT5 weak-positive rescue

Decision:

Keep the env30 boundary-pseudo mask as the current boundary guard, but do not
increase or broaden boundary pseudo-negative pressure. At the time, this
decision selected a matched weak-positive rescue for true GT4/GT5
short-visible lanes, using point/curve geometry weighting plus a small GT5
point-valid rescue. That follow-up has since been rejected by the 2026-07-13
full-protocol result above.

Official-val evidence for env30:

```text
run = query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1
official_best val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
official_best val GT4 4->3/4->5 = 2 / 0
official_best val GT5 5->4/5->5/5->6 = 1 / 73 / 0
max_det=6 equal-ACC rows also keep GT5 5->6 = 0
GT5 visible<=10 raw has_match20/30 = 0.754717 / 0.811321
GT5 visible<=10 raw p90 APE = 36.550196 px
train GT4->5 over label_data_0313/0531/0601 = 29 + 2 + 7 = 38
```

Why:

- The env30 run is real evidence that the narrower outside-envelope mask fixes
  the selected official-val count-shape issue better than broad B1/v2.
- It is not clean promotion evidence because raw GT5 short geometry is weak
  and train GT4->5 remains high.
- The undercount diagnostic indicates missing GT4/GT5 lanes are often
  `geometry_bad`, `score_low`, or `valid_low` before final postprocess, so
  changing only `conf`, NMS, `max_det`, or `min_points` is unlikely to solve
  the target test-ACC gap.
- More pseudo-negative pressure risks suppressing true weak-visible side
  lanes before their geometry and validity are repaired.

Follow-up status:

The intended follow-up from this decision was implemented as
`scripts/run_query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_v1.sh` and
then run through the full protocol as
`query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v1`.
The 2026-07-13 decision above rejects that result. Do not repeat this tiered
GT4/GT5 + GT4 point-valid setup except for reproduction or audit.

The superseded weak-positive settings were:

```text
gcs_short_geom = 1.0
gcs_short_geom_tiered = true
gcs_short_geom_gt4_ultra_visible_thr = 10
gcs_short_geom_gt4_ultra_weight = 1.35
gcs_short_geom_gt4_mid_visible_thr = 20
gcs_short_geom_gt4_mid_weight = 1.0
gcs_short_geom_gt5_ultra_visible_thr = 10
gcs_short_geom_gt5_ultra_weight = 2.25
gcs_short_geom_gt5_mid_visible_thr = 20
gcs_short_geom_gt5_mid_weight = 1.5
gcs_short_geom_max_weight = 3.0
gcs_short_geom_curve = 1.0
gcs_gt4_short_visible_thr = 10
gcs_gt4_short_point_valid_weight = 1.2
gcs_gt5_short_visible_thr = 10
gcs_gt5_short_point_valid_weight = 1.5
```

The env30 boundary mask remains:

```text
gcs_boundary_pseudo_neg = 0.02
gcs_boundary_pseudo_dist_thr = 80
gcs_boundary_pseudo_min_valid = 4
gcs_boundary_pseudo_score_thr = 0.2
gcs_boundary_pseudo_envelope_margin_px = 30
gcs_boundary_pseudo_envelope_ratio_thr = 0.75
```

Promotion gates used for that follow-up:

- official-val ACC/FN must beat or tie env30;
- GT4 `4->5`, GT5 `5->4`, and GT5 `5->6` under `max_det=6` must not regress;
- GT5 visible<=10 raw geometry must improve versus p90 APE `36.550196 px`;
- train GT4->5 must drop below `38`;
- official test remains closed until official-val and train/val diagnostics
  pass, and any later test is reporting-only.
## 2026-07-19: Switch short rescue and clear-far guard to the exact training contract

Decision:

Make the short-lane positive rescue explicit as a two-threshold contract and
move the extra negative guard to clear-far unmatched queries.

Implementation scope:

- `gcs_short_geom` now uses an outer visible-anchor threshold plus a focus
  threshold, with an extra `gcs_short_geom_focus_weight` multiplier for the
  focus bucket.
- GT4 and GT5 keep separate short-rescue weights, and the same rescue weights
  are shared by `point_loss`, `curve_loss`, and the positive-anchor side of
  `point_valid_loss`.
- `boundary_pseudo_neg_loss` now selects unmatched clear-far queries by
  distance to all GT lanes instead of left/right boundary pseudo lanes, while
  keeping the optional envelope protection.
- Updated the training entry, default config, contract notes, and the short
  geom self-check script to match the new behavior.

Validation:

- `D:\miniconda3\envs\lsa_yolo\python.exe -m py_compile
  ultralytics\utils\gcs_loss.py tools\train_gcs.py
  tools\check_gcs_short_geom_loss.py`
- `D:\miniconda3\envs\lsa_yolo\python.exe tools\check_gcs_short_geom_loss.py`
- `D:\miniconda3\envs\lsa_yolo\python.exe tools\check_model.py --cfg
  ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml --imgsz 544 960 --batch 1
  --device cpu`

## 2026-07-10: Use cached official-val sweep for threshold selection

Decision:

Use `tools/sweep_tusimple_official_cached.py` for current official-val
threshold sweeps, including training-time `--gcs-official-best` selection.
Keep `tools/sweep_tusimple_official.py` as the direct compatibility helper and
shared implementation source for sweep grid construction, row writing, and
selection helpers.

Why:

- Cached sweep forwards each checkpoint once on the canonical official-val
  images, then evaluates all decode threshold combinations from saved
  predictions.
- This keeps threshold/checkpoint selection on official-val while avoiding
  repeated model inference for every threshold row.
- Training-time `official_best` now writes a per-epoch
  `official_sweeps/epoch*/prediction_cache/` so each epoch summary points to
  the cache generated from that epoch's `last.pt`.

Scope:

This is a selection/tooling protocol change only. It does not change the model,
losses, labels, decode formulas, official metrics, final-test policy, or
official-best selection priority.

## 2026-07-11: Keep G1 count-aware extra-margin as diagnostic only

Decision:

Add `count_aware_extra_margin` as a default-off query decode/sweep control,
but do not promote the G1 count-aware margin result and do not run final test
from it. The selected/reference G1 decode remains the no-count-aware
`official_best.pt` row.

Implementation scope:

```text
eval CLI = --count-aware-extra-margin
sweep CLI = --count-aware-extra-margins
decode rule = keep_k = min(k_hat + extra_margin, max_det)
default = 0
formal count modes = score_sum, count_logits
diagnostic GT-count mode = --oracle-count only
```

This is decode/evaluation tooling only. It does not change training, labels,
losses, model outputs, Count Head availability, official metrics, or the
default decode path. `oracle_gt` is not a public formal `--count-mode` or
decode-yaml value.

G1 artifact check:

```text
run = query_alpha05_gt5short_geom_w2_v1
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
gcs_query_count_ce = 0.0
checkpoint count-related keys = 0
```

Therefore `count_logits k/k+1/k+2` cannot run on the current G1 artifact. G1
can only test the `score_sum` count-aware proxy unless a query-count model is
trained.

Official-val-only evidence on the fixed G1 `official_best.pt` decode
(`conf=0.003`, `point_valid_thr=0.5`, `nms_dist_px=0`, `max_det=6`,
`min_points=5`, `valid_before_maxdet=true`):

```text
no count-aware baseline:
ACC/FP/FN = 0.973071 / 0.014784 / 0.009642
GT5 5->4/5->5/5->6 = 1 / 69 / 4

score_sum k, extra_margin=0:
ACC/FP/FN = 0.972632 / 0.007668 / 0.011019
GT5 5->4/5->5/5->6 = 8 / 66 / 0

score_sum k+1, extra_margin=1:
ACC/FP/FN = 0.973069 / 0.014325 / 0.009642
GT5 5->4/5->5/5->6 = 1 / 70 / 3

score_sum k+2, extra_margin=2:
ACC/FP/FN = 0.973069 / 0.014784 / 0.009642
GT5 5->4/5->5/5->6 = 1 / 69 / 4
```

Why:

- `score_sum k` is too hard; it lowers FP but raises FN and GT5 undercount.
- `score_sum k+1` is only a near tie with a one-image GT5 `5->6` reduction and
  slightly lower FP. It does not beat the no-count-aware baseline on official
  ACC.
- `score_sum k+2` collapses back to the baseline count shape.
- This proxy sweep does not establish that true `count_logits` is useful after
  G1 geometry; that requires a checkpoint emitting `pred_count_logits`.

Next action:

Do not run final test or reselect G1 decode from this proxy. If count-aware is
revisited, train/evaluate a query-count version of the G1 geometry setup and
run `count_logits k/k+1/k+2` on canonical official-val only.

## 2026-07-11: Reject boundary_pseudo_neg v2 protocol-miss result

Decision:

Do not promote `query_alpha05_gt5short_geom_w2_bneg005_nocount_v2`. Do not
treat it as evidence that outside-GT-envelope protection failed, because the
actual training args disabled the envelope guard:

```text
gcs_boundary_pseudo_envelope_margin_px = -1.0
gcs_boundary_pseudo_neg = 0.05
gcs_boundary_pseudo_dist_thr = 60
gcs_boundary_pseudo_min_valid = 3
gcs_boundary_pseudo_score_thr = 0.0
```

Official-val evidence:

```text
selected weights = weights/official_best.pt
source_epoch = 180
decode = conf=0.001, point_valid_thr=0.6, nms_dist_px=18.0, max_det=5,
         min_points=2, valid_before_maxdet=true, count_mode=score_sum
official-val ACC/FP/FN = 0.969401 / 0.024334 / 0.016070
count_acc_4/5 = 0.909091 / 0.972973
count_confusion = 3->3=216, 3->4=7, 4->3=1, 4->4=60, 4->5=5,
                  5->4=2, 5->5=72
```

Compared with G1 (`ACC=0.973071`, `FP=0.014784`, `FN=0.009642`,
`4->5=0`, `5->4=1`, `GT5 visible<=10 p90 APE=19.372219 px`), v2 is worse on
primary ACC, FP, FN, GT4 false-fifth, GT5 undercount, and short-GT5 raw
geometry. The v2 raw-Q12 diagnostic reports:

```text
GT5 visible<=10 lanes = 53
has_match20/30/40 = 0.754717 / 0.792453 / 0.886792
p90 APE = 35.917850 px
```

The selected row has `GT5 5->6=0`, but equal-ACC `max_det=6` rows show
`GT5 5->6=4..6`, so the apparent sixth-lane improvement is still tied to the
`max_det=5` cap and is not a robust training-side fix. The generic `best.pt`
surface is weaker (`ACC=0.966980`, `FP=0.035537`, `FN=0.017906`) and must not
replace `official_best.pt`.

Reporting-only official test is not selection input, but it confirms the
generalization risk:

```text
official_best test ACC/FP/FN = 0.965566 / 0.036137 / 0.026869
test count_acc_4/5 = 0.598291 / 0.873462
test count_confusion includes 4->5=100 and 5->4=54
```

Train-only diagnostic on the three TuSimple train JSON files, using the
selected official-best decode and `target_gt_count=4`, `target_pred_count=5`,
found:

```text
label_data_0313 GT4->5 images = 45
label_data_0531 GT4->5 images = 6
label_data_0601 GT4->5 images = 23
total = 74
```

Why:

- The intended mask-v2 command was not run. The envelope guard stayed disabled,
  and the mask remained broader/harder than the proposed `neg=0.02`,
  `dist_thr=80`, `min_valid=4`, `score_thr=0.2`, `envelope_margin=30` setup.
- The run fails every serious promotion gate: not close to G1 on official-val
  ACC/FP/FN, GT4 `4->5` grows, GT5 `5->4` grows, selected `5->6=0` is not
  robust under `max_det=6`, and GT5 visible<=10 geometry regresses badly.
- `gcs_gt5_short_visible_thr=0` kept GT5 short point-valid rescue off, so the
  broad pseudo-negative pressure was not paired with a compensating positive
  protection for true short fifth lanes.

Next action:

If the boundary-pseudo line continues, rerun the actual mask-v2/envelope
experiment with the envelope margin explicitly passed in the script/command:
`gcs_boundary_pseudo_neg=0.02`, `dist_thr=80`, `min_valid=4`,
`score_thr=0.2`, `envelope_margin_px=30`, and
`envelope_ratio_thr=0.75`. The local protocol entry is
`scripts/run_query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1.sh`;
it defaults to `RUN_TESTS=0`, so it should run training and cached
official-val sweeps first. Keep selection on canonical 363-image official-val,
run cached sweeps for both `official_best.pt` and `best.pt`, and require the
raw-Q12 short-GT5 geometry and train GT4->5 diagnostics to pass before enabling
`RUN_TESTS=1` for reporting-only test.

## 2026-07-11: Reject boundary_pseudo_neg B1 as promotion

Decision:

Do not promote `query_alpha05_gt5short_geom_w2_bneg005_nocount_v1`, and do
not continue directly to `gcs_boundary_pseudo_neg=0.1`. Keep G1
`query_alpha05_gt5short_geom_w2_v1` as the stronger validation reference for
this line. The B1 result is diagnostic evidence that the current boundary
pseudo-negative mask can suppress some extra GT5 lanes, but it is too broad
and hurts GT5 short-lane raw geometry.

Official-val evidence:

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
```

Reporting-only test evidence is not selection input, but it shows residual
generalization risk. B1 official-best test reaches `ACC=0.966277`,
`FP=0.028906`, and `FN=0.025431`, while GT5 still has `5->4=82` and
`5->5=469`. The companion B1 `best.pt` row is not the candidate: its
official-val `GT5 5->6=8`, `count_acc_5=0.864865`, and reporting-only test
`GT5 5->6=30` show a clear overcount risk.

Why:

- B1 improves validation FP/FN and removes selected-row `GT5 5->6`, but it
  lowers the primary official-val ACC versus G1.
- The selected-row `5->6=0` is partly tied to `max_det=5`; it is not enough
  evidence that the training-side pseudo-negative objective solved overcount.
- GT5 short raw geometry regresses: `has_match20` falls and p90 APE worsens
  by about `+6.03 px`.
- Train diagnostics do not show a useful GT4 benefit: train `GT4 4->3` stays
  `2`, while train `GT4 4->5` worsens from `27` to `31`.

Next action:

Do not use final test for tuning. If this line continues, either shrink the
pseudo-negative mask first (`gcs_boundary_pseudo_neg=0.02`, larger
distance-to-all-GT threshold such as `80 px`, stricter `min_valid=4`, and an
optional score threshold such as `0.2`) or move to a narrower matched
GT4/GT5 short-positive rescue. Any follow-up must use training-time
`official_best`, canonical 363-image official-val, and train/val hard
diagnostics before any reporting-only test.

## 2026-07-09: Keep Count Head default-off and stop count-first direction

Decision:

Do not promote the query Count Head line, the `count_logits` count-aware decode,
or `gcs_yolo_lane_s_q12_k56_shortside025_farspur005_rank002_v1` as the next
algorithm direction. Keep the query Count Head implementation only as a
default-off, reproducibility-preserving ablation. The next experiment should
target selective GT5 short/weak-visible matched-positive score relaxation with
an explicit overcount guard, not a global count head or global existence-alpha
change.

Integrated evidence:

```text
query_count_head_ce05_v1 official_best test:
ACC/FP/FN = 0.964862 / 0.033914 / 0.024473
count_acc_4/5 = 0.617521 / 0.855888

query_count_head best.pt + count_logits count-aware official-val:
ACC/FP/FN = 0.967562 / 0.019330 / 0.015152
count_acc = 0.975207
count_acc_4/5 = 0.954545 / 0.959459

query_count_head best.pt + count_logits count-aware reporting-only test:
ACC/FP/FN = 0.964362 / 0.026582 / 0.028696
count_acc_4/5 = 0.591880 / 0.847100

query_count_head official_best + oracle-count reporting-only test:
ACC/FP/FN = 0.963987 / 0.022172 / 0.027318
count_acc = 0.938893

shortside025_farspur005_rank002 official_best official-val:
ACC/FP/FN = 0.968554 / 0.035078 / 0.018595
count_acc_4/5 = 0.818182 / 0.972973

shortside025_farspur005_rank002 external official-best val sweep:
ACC/FP/FN = 0.969366 / 0.034573 / 0.016988
count_acc_4/5 = 0.803030 / 0.986486
count_confusion includes 4->5=12, 5->6=1

shortside025_farspur005_rank002 reporting-only test:
ACC/FP/FN = 0.964823 / 0.041643 / 0.025971
count_acc_4/5 = 0.549145 / 0.852373
count_confusion includes 4->5=122, 5->4=30, 5->6=42
```

Why:

- `count_logits` improves count accuracy on official-val, but it does not
  improve official ACC and it increases FN on reporting-only test versus the
  Count Head official-best score-sum decode.
- Oracle count improves count accuracy and FP but still lowers ACC and raises
  FN. This rules against lane-count estimation as the dominant bottleneck for
  the current artifact.
- `rank002` preserves some GT5 count on validation but fails GT4 and FP badly;
  the reporting-only test has severe GT4 false-fifth and sixth-lane behavior.
- The `alpha=0.0` experiment proves raw GT5 scores can be raised, but the
  global change loses official-val ACC/FN. The useful signal is selective
  score rescue, not global count selection.

Next action:

Do not run more final tests or tune thresholds from these reports. If the line
continues, run a validation-only selective relaxation experiment for GT5
short-side or weak-visible matched positives, paired with a guard against
extra sixth lanes and GT4 false-fifth growth. Use training-time
`official_best`, the canonical 363-image official-val split, and train/val
diagnostics before any reporting-only test.

## 2026-07-09: Reject global exist-quality-alpha relaxation as next path

Decision:

Do not promote either global query existence-target ablation as the next
algorithm direction. The experiment supports that hard labels
(`gcs_exist_quality_alpha=0.0`) raise GT5 raw existence scores, especially on
short-visible lanes, but it does not support the expected end-to-end pattern
that GT5 recall improves while FP rises. Across the shared 360 official-val
sweep keys, alpha `0.0` is consistently lower on official ACC and higher on
FN than alpha `0.5`, while FP is lower, not higher.

Experiment scope:

```text
E1 = query_exist_quality_alpha05_v1
gcs_exist_quality_alpha = 0.5
batch = 32
scale = 0.15
erasing = 0.10
mosaic = 0.0
gcs_count = 0.0
gcs_count_under5 = 0.0
gcs_count_boundary = 0.0
gcs_spurious_neg = 0.0
gcs_official_best = true
gcs_official_valid_before_maxdet = true

E2 = query_exist_quality_alpha00_v1
gcs_exist_quality_alpha = 0.0
same remaining training/loss scope as E1
```

Primary training-time official-best evidence:

```text
E1 official_best:
weights = runs/gcs_lane/query_exist_quality_alpha05_v1/weights/official_best.pt
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=6, valid_before_maxdet=true
source_epoch = 145
official-val ACC = 0.969860
official-val FP = 0.024197
official-val FN = 0.012856
count_acc_4 = 0.924242
count_acc_5 = 0.891892
count_confusion = 3->3=215, 3->4=8, 4->3=1, 4->4=61, 4->5=4, 5->4=1, 5->5=66, 5->6=7

E2 official_best:
weights = runs/gcs_lane/query_exist_quality_alpha00_v1/weights/official_best.pt
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=30.0, max_det=5, min_points=4, valid_before_maxdet=true
source_epoch = 197
official-val ACC = 0.967784
official-val FP = 0.020845
official-val FN = 0.015152
count_acc_4 = 0.954545
count_acc_5 = 0.972973
count_confusion = 3->3=217, 3->4=6, 4->3=1, 4->4=63, 4->5=2, 5->4=2, 5->5=72
```

Raw official-val diagnostic evidence:

```text
diagnostic script = tools/diagnose_tusimple_raw_q12_filters.py
split = canonical 363-image official-val
E1 output = runs/gcs_lane/query_exist_quality_alpha05_v1/raw_q12_filters_official_best_decode
E2 output = runs/gcs_lane/query_exist_quality_alpha00_v1/raw_q12_filters_official_best_decode

GT5 visible<=10 raw-best exist score mean:
E1 = 0.631605 over 53 lanes
E2 = 0.896660 over 53 lanes

GT5 fifth lane raw-best exist score mean:
E1 = 0.928582 over 74 lanes
E2 = 0.997159 over 74 lanes

query 10/11 against GT5 fifth lane score mean:
E1 = 0.492548 over 148 query-lane pairs
E2 = 0.559448 over 148 query-lane pairs

query 10/11 best-geometry score mean per GT5 fifth lane:
E1 = 0.729697 over 74 lanes
E2 = 0.876093 over 74 lanes
```

Interpretation:

- Supported: lowering/removing quality-aware existence suppression raises GT5
  raw scores. The strongest evidence is E2's higher short-visible GT5 raw-best
  score and higher query 10/11 score against the GT5 fifth lane.
- Not supported: global hard existence labels improve the selected candidate.
  E2 has worse official-val ACC and FN than E1, and the shared-key sweep
  comparison shows the same pattern rather than a selected-decode artifact.
- Not supported: fully closing quality-aware existence causes the expected FP
  explosion. E2's selected FP is lower than E1's, and across common sweep keys
  alpha `0.0` has lower FP on average.
- E1's wider post-train keygrid has an ACC-best row at `0.970634`, but this
  row worsens the target shape (`FP=0.026492`, `count_acc_5=0.824324`,
  `5->6=11`). It is not a better answer to the GT5/short-side question.

Next action:

Do not continue by globally setting `gcs_exist_quality_alpha=0.0`. If this line
continues, make it selective: apply a score/target relaxation only to GT5
short-side or weak-visible matched positives, and pair it with a guard that
prevents the extra GT5 score mass from turning into sixth-lane overcount.
Select only on official-val; no final-test tuning was used for this decision.

## 2026-07-07: Close query_count_head_ce05_v1 as diagnostic

Decision:

Do not promote `query_count_head_ce05_v1`. The valid selection surface favors
the training-time official-best artifact, and the post-train `best.pt` sweep is
weaker on official-val. Both official-test runs are reporting-only evidence and
must not drive threshold, checkpoint, NMS, `max_det`, `min_points`,
`valid_before_maxdet`, count-mode, or loss-gain changes.

Selected official-val artifact:

```text
run = query_count_head_ce05_v1
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
official-val score = 0.967931
official-val count_acc = 0.931129
official-val count_acc_4 = 0.893939
official-val count_acc_5 = 0.891892
```

Rejected comparison surface:

```text
best.pt val sweep = runs/gcs_lane/query_count_head_ce05_v1_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
weights = runs/gcs_lane/query_count_head_ce05_v1/weights/best.pt
conf = 0.003
point_valid_thr = 0.6
nms_dist_px = 0.0
max_det = 6
min_points = 5
valid_before_maxdet = true
count_aware_topk = false
count_mode = score_sum
official-val ACC = 0.967679
official-val FP = 0.032553
official-val FN = 0.015152
official-val score = 0.966725
official-val count_acc = 0.906336
official-val count_acc_4 = 0.893939
official-val count_acc_5 = 0.770270
```

Count-logits count-aware diagnostic on the generic `best.pt` surface:

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

Reporting-only official-test evidence:

```text
official_best test summary = runs/gcs_lane/query_count_head_ce05_v1_official_test_official_best_decode/tusimple_official_summary.json
official_best test ACC = 0.964862
official_best test FP = 0.033914
official_best test FN = 0.024473
official_best test score = 0.963694
official_best test count_acc = 0.874191
official_best test count_acc_2/3/4/5 = 0.200000 / 0.951149 / 0.617521 / 0.855888

best.pt sweep-threshold test summary = runs/gcs_lane/query_count_head_ce05_v1_best_official_test_from_val_sweep_valid_before_maxdet_b/tusimple_official_summary.json
best.pt sweep-threshold test ACC = 0.964518
best.pt sweep-threshold test FP = 0.037653
best.pt sweep-threshold test FN = 0.026779
best.pt sweep-threshold test score = 0.963229
best.pt sweep-threshold test count_acc = 0.836089
best.pt sweep-threshold test count_acc_2/3/4/5 = 0.200000 / 0.965517 / 0.566239 / 0.667838
```

Oracle-count diagnostic requested on the same official-test surface:

```text
oracle-count test summary = runs/gcs_lane/query_count_head_ce05_v1_official_test_official_best_decode_oracle_count/tusimple_official_summary.json
weights = runs/gcs_lane/query_count_head_ce05_v1/weights/official_best.pt
decode base = runs/gcs_lane/query_count_head_ce05_v1/weights/official_best_decode.yaml
oracle change = query count-aware top-k uses k_hat = GT lane count
conf = 0.005
point_valid_thr = 0.45
nms_dist_px = 0.0
max_det = 5
min_points = 2
valid_before_maxdet = true
count_aware_topk = true
count_mode = oracle_gt
official_test_acc = 0.963987
official_test_FP = 0.022172
official_test_FN = 0.027318
official_test_score = 0.962997
test_count_acc = 0.938893
test_count_acc_2/3/4/5 = 1.000000 / 0.999425 / 0.814103 / 0.855888
count_confusion = 2->2=5, 3->2=1, 3->3=1739, 4->3=87, 4->4=381, 5->3=13, 5->4=69, 5->5=487
```

Supported interpretation:

- `official_best.pt` is better than the generic `best.pt` sweep on the valid
  official-val selection surface and also on reporting-only official test.
- The explicit query Count Head CE0.5 line did not produce a promotable result;
  its reporting-only official-test ACC is below the stronger historical
  reporting-only records around `0.965428` to `0.965483`.
- The selected decode does not use count-aware top-k, so this result is not
  evidence for promoting `--count-aware-topk --count-mode count_logits`.
- The count-logits diagnostic improves official-val count accuracy, but its
  official-val ACC remains below the official-best score-sum surface and its
  reporting-only test ACC/FN are worse than the official-best score-sum test.
  This is diagnostic evidence, not a promotion path.
- The oracle-count diagnostic is not a formal score because it uses GT during
  decode. It sharply improves test count accuracy (`0.874191 -> 0.938893`) and
  lowers FP (`0.033914 -> 0.022172`), but official-test ACC drops
  (`0.964862 -> 0.963987`) and FN rises (`0.024473 -> 0.027318`). GT4/GT5
  undercount is not rescued (`4->3=87`, `5->3=13`, `5->4=69` remain). This
  rules out lane-count estimation as the dominant remaining test bottleneck
  for this artifact; the residual problem is candidate quality/ranking or
  point-valid filtering of true 4th/5th lanes.
- Keep the query Count Head implementation available as a default-off ablation,
  but close `query_count_head_ce05_v1` as diagnostic evidence unless a future
  official-val-selected experiment changes the conclusion.

## 2026-07-06: Add default-off query Count Head

Decision:

Add an explicit query-mode Count Head only as a user-requested, default-off
ablation. The default query YAML remains unchanged, ordered-slot keeps its
existing count/slot contract, and query Count Head decode is used only when
`--count-aware-topk --count-mode count_logits` is explicitly selected.

Implementation scope:

- `GCSLaneHead` accepts tail argument `query_count_head=False`; when enabled in
  query mode it creates `query_count_mlp` and emits `pred_count_logits: B x 4`.
- New YAML `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count.yaml`
  enables the optional query Count Head under the current Q12/K56 fixed-y
  contract.
- `GCSLoss` logs `query_count_ce_loss`, `query_count_acc`, and
  `query_count_pred_mean`; the CE term contributes only when
  `gcs_query_count_ce > 0`.
- Query official eval/sweep can use `count_logits` as the count-aware top-k
  source, while `score_sum` remains the default count source.

Validation:

Local Python compile passed for changed Python files. The focused
`tools/check_gcs_query_count_head.py` contract check passed with 31 loss items.
CPU model shape checks passed for default query, query-count, and ordered-slot
YAMLs with `--imgsz 544 960`. `tools/check_ordered_slot_contracts.py
--skip-git` and `tools/check_gcs_valid_before_maxdet.py` passed.

## 2026-07-06: Roll Active Code Back to 424ab1c86

Decision:

Restore active source/config to commit
`424ab1c869f0a02556d8b6b6a44c27e5585e47c0` (`Add valid-before-maxdet decode
option`). Documentation remains current but must treat every mechanism, tool,
command, and result introduced after this commit as legacy old state unless a
future task explicitly restores it.

Active boundary:

- Keep the 5-25-3 K56 contract, `count_boundary_loss`, train-only
  `gcs_hard_sampling`, E3-lite `gcs_spurious_neg`, training-time
  `official_best`, ordered-slot protocol tooling already present at the target
  commit, and the target commit's default-off `valid_before_maxdet` decode
  option.
- Remove active code/config from later commits by restoring the affected
  `tools/`, `ultralytics/cfg/`, `ultralytics/models/yolo/gcs_lane/`, and
  `ultralytics/utils/` paths to the target commit.

Legacy after this rollback:

```text
436fcb616 Add GCS short-side hardset diagnostics
f4c200bbe Add default-off short-side geometry loss
f7f0bcc07 Add far spurious negative loss
8357e1ad8 Add GCS count contract tooling
```

These post-`424ab1c86` changes are old state: short-side hardset diagnostics,
`gcs_short_side_geom`, `gcs_far_spurious_neg`, `gcs_farspur_*`,
`gcs_shortside_*`, `gcs_rank_*`, `gcs_base_ignore_*`, and
`tools/diagnose_gcs_count_contract.py` / count-contract tooling must not be
read as available active code.

Validation target:

- `git diff --name-status 424ab1c869f0a02556d8b6b6a44c27e5585e47c0 -- tools ultralytics/cfg ultralytics/models/yolo/gcs_lane ultralytics/utils`
  should be empty after the rollback.
- Run local Python compile on Python files changed relative to `HEAD` where
  practical.
- Run `tools/check_gcs_valid_before_maxdet.py` because the target boundary is
  the valid-before-maxdet commit.

## 2026-07-09: Reject shortside025_farspur005_rank002 after artifact sync

Decision:

Reject `gcs_yolo_lane_s_q12_k56_shortside025_farspur005_rank002_v1` as a
promotion path. The previously missing artifact package is now readable on the
remote server, and the exact run fails official-val before considering its
reporting-only test.

Artifacts checked:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_k56_shortside025_farspur005_rank002_v1/args.yaml
runs/gcs_lane/gcs_yolo_lane_s_q12_k56_shortside025_farspur005_rank002_v1/results.csv
runs/gcs_lane/gcs_yolo_lane_s_q12_k56_shortside025_farspur005_rank002_v1/weights/official_best_decode.yaml
runs/gcs_lane/gcs_yolo_lane_s_q12_k56_shortside025_farspur005_rank002_v1/weights/official_best_sweep.json
runs/gcs_lane/gcs_yolo_lane_s_q12_k56_shortside025_farspur005_rank002_v1_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_k56_shortside025_farspur005_rank002_v1_official_test_best_from_val_b/tusimple_official_summary.json
```

Evidence:

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

reporting-only test from the external official-val decode:
official-test ACC/FP/FN = 0.964823 / 0.041643 / 0.025971
count_acc_4/5 = 0.549145 / 0.852373
count_confusion includes 4->5=122, 5->4=30, 5->6=42
```

Why:

- The run preserves GT5 better than the failed rank-pair variants on
  official-val, but it fails GT4 count and FP badly.
- The external validation sweep does not repair the issue: `4->5` remains
  worse than the spurious-lite/E1 direction, and FP remains high.
- The reporting-only test confirms the validation warning with severe GT4
  false-fifth and sixth-lane behavior.

Next action:

Do not tune thresholds, checkpoint choice, NMS, rank weight, farspur weight, or
shortside settings from this report. Treat it as another diagnostic showing
that broad GT4/GT5 ranking/shortside pressure can move count shape but does not
solve the candidate-quality and overcount tradeoff.

## 2026-07-06: Reject GT4/GT5 ranking pair and duplicate-ignore ablations

Decision:

Do not promote or continue
`gcs_yolo_lane_s_q12_k56_ablate_rank_pair_gt4gt5_002_v1` or
`gcs_yolo_lane_s_q12_k56_ablate_rank_full_dupignore_gt4gt5_002_v1`. Treat both
as diagnostic-only evidence: GT4 false-fifth ranking pressure can reduce
`4->5`, but the current `gcs_rank_topk_weight=0.02` all-GT4/GT5 matched
ranking configurations cause unacceptable GT5 undercount on reporting-only
test.

Evidence:

Baseline `gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1` with
`valid_before_maxdet=true` fixed official-val selected decode:

```text
363-val ACC = 0.971816
363-val FP/FN = 0.019376 / 0.015611
363-val count_acc_4/5 = 0.924242 / 0.986486
363-val confusion includes 4->3=1, 4->5=4, 5->4=1, 5->5=73

reporting-only test ACC = 0.965483
test FP/FN = 0.030847 / 0.027678
test count_acc_4/5 = 0.606838 / 0.843585
test confusion includes 4->3=105, 4->5=79, 5->3=26, 5->4=63, 5->5=480
```

`rank_pair_only` used `gcs_rank_topk_weight=0.02`,
`gcs_rank_pos_scope=gt4gt5_matched`, and
`gcs_base_ignore_duplicate_like=False`:

```text
official_best_val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_rank_pair_gt4gt5_002_v1_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
selected decode = conf=0.01, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=3, valid_before_maxdet=true
363-val ACC = 0.974062
363-val FP/FN = 0.007576 / 0.010560
363-val count_acc_4/5 = 0.939394 / 0.959459
363-val confusion includes 4->3=4, 4->5=0, 5->4=3, 5->5=71

reporting-only test summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_rank_pair_gt4gt5_002_v1_official_test_best_from_val_bb/tusimple_official_summary.json
test ACC = 0.963617
test FP/FN = 0.026725 / 0.030404
test count_acc_4/5 = 0.632479 / 0.738137
test confusion includes 4->3=113, 4->5=58, 5->3=28, 5->4=121, 5->5=420
```

`rank_full_duplicate_contract` used `gcs_rank_topk_weight=0.02`,
`gcs_rank_pos_scope=gt4gt5_matched`, and
`gcs_base_ignore_duplicate_like=True`:

```text
official_best_val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_rank_full_dupignore_gt4gt5_002_v1_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
selected decode = conf=0.008, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
363-val ACC = 0.974208
363-val FP/FN = 0.007668 / 0.012167
363-val count_acc_4/5 = 0.909091 / 0.945946
363-val confusion includes 4->3=6, 4->5=0, 5->4=4, 5->5=70

reporting-only test summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_rank_full_dupignore_gt4gt5_002_v1_official_test_best_from_val_b/tusimple_official_summary.json
test ACC = 0.963831
test FP/FN = 0.028313 / 0.029625
test count_acc_4/5 = 0.632479 / 0.750439
test confusion includes 4->3=110, 4->5=62, 5->3=27, 5->4=115, 5->5=427
```

Across both 2520-row `official_best_val_sweep` grids, no row reaches the
baseline `count_acc_5=0.986486` or `5->4<=1`; the best observed sweep rows only
reach `max_count_acc_5=0.972973` and `min_5->4=2`.

Why:

- Both ranking variants improve official-val ACC/FP and reduce GT4 false-fifth
  pressure, but they do not preserve GT5 retention.
- Reporting-only test confirms the official-val warning: `5->4` worsens from
  `63` to `121` for rank-pair and to `115` for duplicate-ignore, while test
  ACC falls below both the spurious-lite baseline and the `0.9655` target.
- Full duplicate-ignore is slightly better than rank-pair on reporting-only
  test ACC and GT5 retention, but the gap is too small and still fails the
  main GT5 and ACC targets.
- This does not reject all future ranking research. These two runs also kept
  the count/boundary/spurious training terms off, so the conclusion is scoped
  to the tested `0.02` all-GT4/GT5 matched ranking configurations.

Next action:

Do not run more final tests, tune thresholds, reselect checkpoints, adjust NMS,
change duplicate-ignore policy, or change rank margin/weight from these
reporting-only test results. If ranking is revisited, first run
train/official-val-only GT5 undercount and rank-mask diagnostics, then
predefine a GT5-safe gate before any new training.

## 2026-07-06: Reject shortside_protect_floor07_v1-2

Decision:

Do not promote
`gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2`. Treat it as
diagnostic-only evidence that adding `gcs_shortside_exist_target_floor=0.7` to
the rawmatch boost does not fix the GT5 retention problem.

Primary protocol note:

Use the `weights/official_best.pt` path for this decision. A companion
`weights/best.pt` sweep/test exists, but it is not the primary comparable
candidate because this run was trained with `gcs_official_best=true`.

Official-val evidence:

```text
val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
weights = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2/weights/official_best.pt
selected_decode = conf=0.008, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_val_acc = 0.972798
official_val_FP = 0.011433
official_val_FN = 0.012626
official_val_count_acc = 0.972452
official_val_count_acc_4 = 0.939394
official_val_count_acc_5 = 0.959459
official_val_count_confusion includes 4->3=3, 4->5=1, 5->4=3, 5->5=71
```

Reporting-only test evidence:

```text
test_summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2_test_best_from_val_b/tusimple_official_summary.json
official_test_acc = 0.964281
official_test_FP = 0.029014
official_test_FN = 0.029385
test_count_acc = 0.873832
test_count_acc_4 = 0.621795
test_count_acc_5 = 0.766257
test_count_confusion includes 4->3=112, 4->5=65, 5->3=28, 5->4=105, 5->5=436
```

Why:

- Against `shortside_boost_only_fixedthr_v1`, official-val ACC, FP, FN, and
  `count_acc_4` are all worse; reporting-only test ACC is only `+0.000137`,
  while FP/FN and GT4 `4->3` are worse.
- Against the spurious-lite + valid-before reporting-only comparator, GT4
  `4->5` improves from `79` to `65`, but ACC drops from `0.965483` to
  `0.964281`, FN rises from `0.027678` to `0.029385`, `count_acc_5` drops from
  `0.843585` to `0.766257`, and GT5 `5->4` worsens from `63` to `105`.
- It misses the active promotion gates: `ACC >= 0.9655`, `5->4 <= 63`,
  `GT4 4->3 <= 105`, `FN <= 0.0277`, and `count_acc_5` near `0.843585`.
  It only satisfies the GT4 `4->5 <= 79` and FP gates.
- The floor did not convert the shortside protect ablation into a GT5-safe
  candidate. The selected `point_valid_thr=0.6` row still has a large GT5
  undercount problem.
- The reporting-only test result is not a selection surface. Do not use it to
  reselect `point_valid_thr`, NMS, checkpoint, decode, target floor, or loss
  weights.

Next action:

Keep this run closed as a single val-selected test report. If this line is
revisited, use official-val/train-val-only error decomposition for the added
GT4 `4->3` and GT5 `5->4` undercount, especially around the
`point_valid_thr=0.6` selected row and shortside target floor behavior. Do not
run another final test until a new candidate is selected by official-val.

## 2026-07-06: Reject shortside_boost_only_fixedthr_v1

Decision:

Do not promote `gcs_yolo_lane_s_q12_k56_ablate_shortside_boost_only_fixedthr_v1`.
Treat it as diagnostic-only evidence that shortside boost-only can reduce some
GT4 false-fifth errors, but does not protect GT5 retention.

Official-val evidence:

```text
val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_boost_only_fixedthr_v1_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
weights = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_boost_only_fixedthr_v1/weights/official_best.pt
selected_decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_val_acc = 0.974445
official_val_FP = 0.007576
official_val_FN = 0.009642
official_val_count_acc = 0.975207
official_val_count_acc_4 = 0.954545
official_val_count_acc_5 = 0.959459
official_val_count_confusion includes 4->3=3, 4->5=0, 5->4=3, 5->5=71
```

Reporting-only test evidence:

```text
test_summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_boost_only_fixedthr_v1_test_best_from_val_b/tusimple_official_summary.json
official_test_acc = 0.964144
official_test_FP = 0.028666
official_test_FN = 0.028696
test_count_acc = 0.873472
test_count_acc_4 = 0.630342
test_count_acc_5 = 0.757469
test_count_confusion includes 4->3=106, 4->5=67, 5->3=27, 5->4=111, 5->5=431
```

Why:

- Against the spurious-lite + valid-before reporting-only comparator,
  `4->5` improves from `79` to `67`, FP improves from `0.030847` to
  `0.028666`, and `count_acc_4` improves from `0.606838` to `0.630342`.
- The cost is unacceptable: ACC drops from `0.965483` to `0.964144`, FN rises
  from `0.027678` to `0.028696`, `count_acc_5` drops from `0.843585` to
  `0.757469`, and GT5 `5->4` worsens from `63` to `111`.
- It misses the active promotion gates: `ACC >= 0.9655`, `5->4 <= 63`, and
  `GT4 4->3 <= 105`. Only the `GT4 4->5 <= 79` gate is met.
- The failure is consistent with the shortside boost-only concern: increasing
  BCE weight without an explicit positive target floor can reduce some
  overcount but still amplify undercount pressure on weak true GT5 lanes.
- The reporting-only date/count breakdown is a risk description, not a
  selection surface. Do not use it to reselect `point_valid_thr`, NMS,
  checkpoint, decode, or loss weights.

Next action:

Keep this run closed as a single val-selected test report. Continue only with
official-val/train-val diagnostics or new pre-registered ablations such as
`shortside_floor_only`, `shortside_protect`, `rank_full_duplicate_contract`, or
`farspur_full_ignore_first`, selected on official-val before any future test.

## 2026-06-29: Implement ordered_slot_eval_contract_hardening_v1

Decision:

Harden ordered-slot official/eval protocol, standalone loss defaults, canonical
official-val GT handling, and low-risk logging/model-construction boundaries
without changing model structure, loss gain values, labels, or running a new
algorithm experiment.

Implementation scope:

- Ordered-slot `official_eval`, `official_sweep`, `official_best`, and
  `eval_gcs` now use `output_order=slot`, `order_check=error`,
  `uses_runtime_sort=false`, and `order_violation_policy=fail_fast`.
- Debug/visualization sorted exports remain available, but summaries must mark
  `result_type=postprocessed_sorted_export` and
  `not_for_main_ordered_slot_claim=true`.
- `OrderedSlotGCSLoss` standalone defaults now match `default.yaml`:
  `gcs_count_ce=1.0`, `gcs_interval=1.0`, and `gcs_order=0.1`.
  Nonpositive count/interval gains fail fast, and disabling order loss requires
  an explicit ablation flag.
- Ordered-slot training metadata records `ordered_slot_loss_contract` and the
  effective loss weights, including the fact that slot4/slot5 exist weights are
  BCE element weights for both positive and negative targets.
- For `split=val`, official eval requires canonical 363-image GT by default
  and refuses `train_val.json` / `label_data_0313.json` fallback as comparable
  official-val evidence. `--allow-noncanonical-gt` is explicit and marks the
  summary as not comparable to E1/spurious official-val.
- Query-mode model construction ignores ordered-slot-only overrides; those args
  are passed into `parse_model` only under `gcs_mode=ordered_slot`.
- Absent per-class `slot_count_acc_N` values display as `NA` and are skipped in
  aggregate progress/logging instead of appearing as `0` or raw `nan`.

Validation:

Local Python compile checks passed for changed Python files.
`tools/check_ordered_slot_contracts.py --skip-git` passed, covering strict
official runtime config, reverse-slot fail-fast, debug sorted-export marking,
loss defaults and ablation guard, canonical/noncanonical val GT handling,
query-mode construction isolation, and absent-class `NA` logging. CPU model
shape checks passed for both the default query YAML and ordered-slot Q5/K56
YAML with `--imgsz 544 960`.

## 2026-06-29: Implement ordered_slot_boundary_and_legacy_safety_fix_v1

Decision:

Harden ordered-slot boundary contracts and legacy tool safety without changing
training targets, loss terms, decoder strategy, runtime sorting, or official
metrics.

Implementation scope:

- `GCSLaneHead` now rejects ordered-slot configs unless both `num_queries` and
  `num_slots` are exactly `5`.
- `decode_ordered_slot.validate_ordered_slot_pred_shapes` requires
  `S == max_lanes` and keeps the `pred_count_logits` shape tied to
  `max_lanes - min_lanes + 1`.
- ordered-slot `--pred-json` official eval uses the shared query-only decode
  guard, so replay evidence cannot include non-default query thresholds.
- `tools/sweep_gcs_conf.py` rejects `--split test` because it is a legacy
  validation threshold-search tool.
- `tools/strip_gcs_head_ckpt.py` strips raw top-level head keys, and
  `tools/visualize_ordered_slot_targets.py` fails fast when `cv2.imwrite`
  fails.

Validation:

Targeted contract tests were added for `S=6`, non-5 head config,
pred-json query-arg pollution, test split sweep rejection, raw head key
stripping, and visualization write failure.

## 2026-06-29: Implement ordered_slot_order_diagnostics_summary_fix_v1

Decision:

Fix ordered-slot order diagnostics summary consistency without changing model
forward, loss terms, decode geometry, runtime sorting behavior, or official
metric calculation.

Implementation scope:

- In `tools/eval_tusimple_official.py`, `--pred-json` with
  `decode_mode=ordered_slot` now records
  `ordered_slot_order_checked=false`,
  `ordered_slot_order_violations=null`,
  `ordered_slot_order_violation_images=null`, and
  `ordered_slot_order_diagnostics=not_available_from_pred_json` because the
  existing prediction JSON cannot prove original slot order.
- Model-forward ordered-slot official eval/sweep paths record
  `ordered_slot_order_checked=true` and write violation counts only from
  `decode_ordered_slot_predictions(..., return_diagnostics=True)`.
- The runtime order policy recorded in this older entry was superseded by
  `ordered_slot_eval_contract_hardening_v1`; active ordered-slot official/eval
  now preserves slot order and fails fast on order violations.

Validation:

Targeted contract tests were added for pred-json unavailable diagnostics,
model decode diagnostics, and `eval_gcs` summary/runtime-config consistency.

## 2026-06-29: Implement ordered_slot_output_and_legacy_tool_contract_fix_v1

Decision:

Fix ordered-slot final output order, legacy query-sweep misuse protection, and
official sweep evidence schema without changing training, loss terms, Count
Head, count logits, or official metrics.

Implementation scope:

- Add configurable ordered-slot output ordering and keep each returned lane's
  original `slot` id. The earlier sorted official-export policy recorded here
  was superseded by `ordered_slot_eval_contract_hardening_v1`; sorted exports
  are now diagnostic/debug only.
- Accumulate ordered-slot order diagnostics in official eval/sweep summaries:
  `ordered_slot_order_violations` and
  `ordered_slot_order_violation_images`.
- Record ordered-slot effective decode as `ordered_slot_decode_v1`; the active
  effective decode contract is now slot order with no runtime sort.
- Make `tools/sweep_gcs_conf.py` fail fast on ordered-slot models because it is
  a legacy query conf/NMS/topk sweep tool.
- Remove query-only keys from ordered-slot official sweep JSON rows. CSV keeps
  fixed query-only columns only as empty cells and writes
  `decode_schema=ordered_slot_decode_v1` plus
  `query_decode_args=not_applicable`.

Validation:

Local Python compile checks passed for changed Python files.
`tools/check_ordered_slot_contracts.py --skip-git` passed, including synthetic
reverse-slot runtime sorting, `output_order=slot` plus `order_check=error`,
legacy conf-sweep rejection for ordered-slot models, ordered-slot JSON row key
cleanup, and ordered-slot CSV empty query-only columns.

## 2026-06-29: Implement ordered_slot_runtime_stability_fix_v1

Decision:

Fix ordered-slot runtime stability issues without changing algorithm targets,
loss gains, decoder behavior, official metrics, or the 2/3/4/5 count contract.

Implementation scope:

- Keep ordered-slot GT target construction in float32 before padding removal,
  fixed-y validation, canonical y-anchor snapping, and slot target creation.
- Compute ordered-slot losses and diagnostics with float32 logits/points so AMP
  half prediction tensors do not force GT fixed-y anchors through half
  precision.
- Move shared fixed-y contract utilities to
  `ultralytics/utils/gcs_fixed_y.py`, keep
  `ultralytics/models/gcs/fixed_y.py` only as a compatibility re-export, and
  update dataset, target, audit, model-check, and official-eval imports.
- Delay `GCSLaneDataset` import inside `tools/visualize_ordered_slot_targets.py`
  so `--help` works without loading the full training/model package chain.
- Fail fast for `--gcs-mode ordered_slot --scale > 0` because fixed-y scale
  resampling can reduce targets to unsupported 0/1 lanes. Query-mode scale
  behavior remains unchanged; count-preserving scale fallback is a separate
  future experiment.

Validation:

Local Python compile checks passed for changed Python files.
`tools/visualize_ordered_slot_targets.py --help` passed standalone.
`tools/check_ordered_slot_contracts.py --skip-git` passed, including AMP
half-pred/float32-GT target construction, half-quantized fixed-y tolerance,
standalone visualization help, ordered-slot scale fail-fast, and query-mode
scale preservation tests.

## 2026-06-29: Implement ordered_slot_training_protocol_fix_v1

Decision:

Fix ordered-slot/query comparability and eval-summary protocol issues without
changing model structure, loss definitions, or decoder behavior.

Implementation scope:

- Set `gcs_lane_count_balanced` default to `False` in `default.yaml` and
  `tools/train_gcs.py`; count-balanced sampling now requires explicit
  `--gcs-lane-count-balanced`.
- Keep Trainer sampling driven only by the explicit flag, log
  `GCS lane-count-balanced sampling: true/false`, and do not auto-enable it
  for ordered-slot mode.
- Add `maybe_switch_ordered_slot_model(args)` so `--gcs-mode ordered_slot`
  switches any non-slot GCS YAML to
  `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q5-slot-k56.yaml` unless
  `--gcs-disable-auto-model-switch` requests fail-fast behavior.
- Make `tools/eval_gcs.py` use `ordered_slot_decode_v1` summary fields for
  ordered-slot decode and omit query-only keys such as `conf`,
  `point_valid_thr`, `nms_dist_px`, `max_det`, `min_points`, and
  `count_aware_topk`.
- Keep ordered-slot point regression default at
  `gcs_ordered_point_loss=normalized_smooth_l1`; `aspect_l1` is available only
  through explicit `--gcs-ordered-point-loss aspect_l1` for a separate
  point-loss ablation.
- Make training-time ordered-slot `official_best` fail fast when non-default
  query-only sweep args are supplied, instead of silently replacing them with
  ordered-slot defaults.
- Align `tools/train_gcs.py` and direct trainer defaults with the canonical
  current-contract paths `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml` and
  `data/tusimple_gcs_fixed_y_960x544.yaml`.

Validation:

Local Python compile checks passed for changed Python files. The ordered-slot
contract script passed with `--skip-git`, including tests for count-balanced
defaults, parser behavior, ordered-slot auto-switch/fail-fast behavior,
ordered-slot eval summary key filtering, ordered-slot point-loss default
preservation, training-time official-sweep fail-fast behavior, canonical
train defaults, and query-mode summary/grid preservation.

## 2026-06-29: Formalize ordered_slot_v2 2/3/4/5 Count Contract

Decision:

Keep ordered-slot v2 as a 2/3/4/5 four-class count contract because TuSimple
contains 2-lane samples. Do not roll the head or decoder back to 3/4/5.

Implementation scope:

- `gcs_min_lanes=2`, `gcs_max_lanes=5`, and `gcs_count_classes=4` define
  ordered-slot count classification.
- `pred_count_logits` is `B x 4`; `count_label = num_lanes - 2`; decoder
  lane count is `argmax(pred_count_logits) + 2`.
- A 2-lane target is `slot_exist=[1,1,0,0,0]` with `count_label=0`.
- Ordered-slot logging reports `slot_count_acc_2`, `slot_count_acc_3`,
  `slot_count_acc_4`, and `slot_count_acc_5`.
- Training-time `official_best_decode.yaml` now writes a schema-specific
  `ordered_slot_decode_v1` decode block without query-only keys such as
  `conf`, `nms_dist_px`, `max_det`, or `min_points`.
- Official eval/sweep decode-yaml loading validates schema and rejects polluted
  ordered-slot yaml files with query-only keys.

Validation target:

Run `tools/check_ordered_slot_contracts.py --skip-git`, Python compile checks
for changed Python files, and `tools/audit_fixed_y_labels.py --data ...` when
the fixed-y dataset is locally available.

## 2026-06-29: Implement ordered_slot_contract_fix_v3

Decision:

Fix the ordered-slot third-batch high-risk contract issues without continuing
the spurious/gtprotect line and without changing the algorithm direction.

Implementation scope:

- Add explicit ordered-slot aspect-weighted L1 support via a shared point-loss
  helper. This must be enabled with
  `--gcs-ordered-point-loss aspect_l1`; the protocol default remains
  `normalized_smooth_l1` so ordered-slot training objective comparisons stay
  comparable.
- Split fixed-y validation into training desc `710..160` and official
  h-samples asc `160..710` APIs. Training dataset labels, ordered-slot targets,
  model checks, and label audit use the desc API; official TuSimple conversion
  uses the asc API.
- Build ordered-slot targets by removing padded/invalid lanes before fixed-y
  validation, so leading all-zero padding lanes do not trigger false contract
  failures.
- Centralize official sweep and training-time official-best selection key
  definitions, and write the full `selection_policy.ordered_keys` into
  summaries/artifacts.
- Write ordered-slot `effective_decode` and `query_decode_args=not_applicable`
  into official eval/sweep summaries, and reject non-default query-only decode
  args for ordered-slot official eval/sweep.

Validation:

Local compile checks passed for changed Python files. The ordered-slot contract
script passed with `--skip-git`, including tests for explicit aspect_l1 10px
point-loss gradient, ascending training fixed-y rejection, ascending official
h-samples acceptance, padding-before-fixed-y validation,
selection-policy/sort-key consistency,
ordered-slot summary helpers, and official eval query-arg rejection.
## 2026-07-02: Add Default-Off Valid-Before-MaxDet Decode Ablation

Decision:

Add a default-disabled query-mode decode option `valid_before_maxdet`, exposed
as `--valid-before-maxdet` in TuSimple official eval/sweep and inference
helpers. When enabled, decoded candidates are filtered by the existing
point-valid/min_points rule after confidence sorting and Lane-NMS but before
`max_det` truncation, then re-sorted by score before final truncation.

Why:

The old decode order could let a high-score query with zero valid anchors take
a `max_det` slot, then be removed by the later point-valid/min_points filter,
leaving a real lower-score lane unavailable. The new opt-in path removes such
invalid queries before slot allocation.

Scope:

This is inference/evaluation postprocess selection only. It does not change
training, labels, losses, model outputs, official metrics, Count Head, Quality
Head, Survival Head, or default decode behavior. The default remains
`valid_before_maxdet=false`, preserving old experiments and sweeps.

Validation target:

Use the 363-image official-val surface only for counterfactual evaluation.
Do not use test for selecting this option. The minimal synthetic check should
cover `q_empty score=0.20 valid_count=0`, `q_true score=0.07 valid_count=7`,
and `max_det=1`, where the old path returns no lane and the opt-in path keeps
`q_true`.

Official-val counterfactual result:

```text
date = 2026-07-02
weights = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1/weights/best.pt
gt_json = runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json
surface = official-val 363 only
focused_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1_official_val_focused_sweep_valid_before_maxdet/tusimple_official_sweep_summary.json
focused_sweep_rows = 288
best decode = conf=0.005, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_acc = 0.971816
official_score = 0.971116
official_FP = 0.019376
official_FN = 0.015611
count_acc = 0.975207
count_acc_3 = 0.986547
count_acc_4 = 0.924242
count_acc_5 = 0.986486
count_confusion = 3->3=220, 3->4=3, 4->3=1, 4->4=61, 4->5=4, 5->4=1, 5->5=73
```

Conclusion:

For the spurious-lite checkpoint, `valid_before_maxdet=true` fixes the GT5
decode truncation failure: `5->4` improves from `9` to `1` and `count_acc_5`
improves from `0.878378` to `0.986486`. Official ACC also increases slightly
from `0.971721` to `0.971816`. The tradeoff is a moderate FP rebound from
`0.013866` to `0.019376`, but it remains below the E1 FP value `0.022957`
while keeping spurious-lite's `4->5=4` benefit versus E1's `4->5=9`.

A focused local grid did not find a better row that both lowers FP below
`0.018` and keeps official ACC above E1: the best `FP<=0.018` row had
`official_acc=0.971127`, `official_FP=0.017769`, `5->4=2`, and is slightly
below E1 `official_acc=0.971208`. Treat the selected row above as the current
official-val decode candidate for this checkpoint, not as a global default.

## 2026-07-02: Reject GT-Aware Spurious-Protect V2 Training Run

Decision:

Do not promote `gcs_yolo_lane_s_q12_k56_boundary02_spurious_gtprotect_v2`.
Do not run final test for it. Treat it as diagnostic evidence that default
GT-aware spurious protection plus valid-before-maxdet can recover GT5, but in
this configuration it loses the GT3/GT4 FP benefit needed from spurious-lite.

Official-val evidence:

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_spurious_gtprotect_v2
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
gcs_spurious_neg = 0.1
gcs_spurious_gt3_weight = 1.0
gcs_spurious_gt4_weight = 1.0
gcs_spurious_gt5_weight = 1.0
gcs_spurious_gt_protect = true
gcs_gt5_short_visible_thr = 0
gcs_official_best = true
gcs_official_valid_before_maxdet = true
official_best = weights/official_best.pt
official_best_epoch = 40
official_best_decode = conf=0.005, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_acc = 0.968515
official_score = 0.967623
official_FP = 0.029201
official_FN = 0.015381
count_acc = 0.928375
count_acc_3 = 0.937220
count_acc_4 = 0.833333
count_acc_5 = 0.986486
count_confusion = 3->3=209, 3->4=12, 3->5=2, 4->3=2, 4->4=55, 4->5=9, 5->4=1, 5->5=73
```

Focused official-val sweep on `official_best.pt`:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_gtprotect_v2_official_best_focused_sweep_valid_before_maxdet/tusimple_official_sweep_summary.json
surface = official-val 363 only
rows = 480
grid = conf 0.005..0.25, point_valid_thr 0.45/0.50/0.55/0.60, nms 0/18/30/50, max_det=5, min_points=2/3/4, valid_before_maxdet=true
best decode = conf=0.02, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=4
official_acc = 0.968515
official_score = 0.967682
official_FP = 0.026263
official_FN = 0.015381
count_acc = 0.936639
count_acc_3 = 0.946188
count_acc_4 = 0.848485
count_acc_5 = 0.986486
count_confusion = 3->3=211, 3->4=10, 3->5=2, 4->3=3, 4->4=56, 4->5=7, 5->4=1, 5->5=73
```

Integrated conclusion:

- Supported fact: GT5 is recovered under valid-before-maxdet (`5->4=1`,
  `count_acc_5=0.986486`), so the original GT5 truncation symptom is not the
  remaining blocker.
- Supported fact: official ACC is far below E1 and spurious-lite
  (`0.968515` vs E1 `0.971208`, spurious-lite+valid-before `0.971816`).
- Supported fact: FP rebounds above both E1 and spurious-lite+valid-before.
  Even the focused sweep best has `official_FP=0.026263`, above E1 `0.022957`
  and above spurious-lite+valid-before `0.019376`.
- Supported fact: the GT4 benefit is mostly lost: focused-sweep best has
  `4->5=7`, worse than spurious-lite's `4`, and the training official-best
  row has `4->5=9`, back at the E1 level.
- Supported fact: the new FP is not only GT4 false fifth. GT3 over-count also
  increases in the focused-sweep best (`3->4=10`, `3->5=2`) versus E1
  (`3->4=7`) and spurious-lite+valid-before (`3->4=3`).
- Supported fact: no focused-sweep row satisfied both `4->5<=5` and
  `5->4<=2`. Lower-FP rows required higher thresholds and reintroduced GT5
  misses (`5->4=8..9`) while keeping official ACC around `0.966`.

Likely mechanism:

The protection/default-weight setup does not preserve the desired GT4 pressure.
Late training logs show only small GT4 spurious-negative selection while many
selected spurious negatives are GT5-group candidates. The protection count is
small relative to candidate count, so the run neither cleanly protects only
true GT5-like short candidates nor focuses enough negative pressure on GT4
false-fifth cases.

Next action:

Do not add exist rescue to this failed V2 line. The next smallest controlled
experiment should narrow the protection policy to GT5-or-denser samples or
increase GT4 pressure while keeping GT5 protection, for example:

```text
--gcs-spurious-neg 0.1
--gcs-spurious-gt4-weight 1.5
--gcs-spurious-gt5-weight 1.0
--gcs-spurious-gt-protect
--gcs-official-valid-before-maxdet
```

If code is changed, prefer an explicit `gcs_spurious_gt_protect_min_gt_lanes`
or equivalent gate so GT3/GT4 samples keep the original spurious-lite negative
behavior while only GT5-like true short candidates are protected.

## 2026-07-02: GT5-Only Protect + GT4W15 V1 Official-Val / Test Review

Decision:

Treat `gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt5only_gt4w15_v1` as a
strong official-val diagnostic candidate, but do not claim a final-test
improvement. The one-shot test report selected from official-val is below the
previous reporting-only final-test high-water mark, so any follow-up must return
to official-val/train diagnostics and must not tune from test.

Training/config evidence:

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt5only_gt4w15_v1
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
gcs_spurious_neg = 0.1
gcs_spurious_gt3_weight = 1.0
gcs_spurious_gt4_weight = 1.5
gcs_spurious_gt5_weight = 1.0
gcs_spurious_gt_protect = true
gcs_spurious_gt_protect_min_gt_lanes = 5
gcs_official_valid_before_maxdet = true
official_best = weights/official_best.pt
official_best_epoch = 123
official_best_decode = conf=0.005, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_best_acc = 0.974056
official_best_FP = 0.012580
official_best_FN = 0.011938
official_best_count_acc = 0.972452
official_best_count_acc_4 = 0.939394
official_best_count_acc_5 = 0.972973
official_best_confusion = 3->3=219, 3->4=4, 4->3=3, 4->4=62, 4->5=1, 5->4=2, 5->5=72
```

Post-hoc official-val sweep on `official_best.pt`:

```text
sweep = D:/gcsxxl/1/gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt5only_gt4w15_v1_official_best_val_sweep/tusimple_official_sweep_summary.json
surface = canonical official-val 363
rows = 1512
valid_before_maxdet = false
best decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=18.0, max_det=8, min_points=2
official_acc = 0.975185
official_score = 0.974711
official_FP = 0.012443
official_FN = 0.011249
count_acc = 0.972452
count_acc_3 = 0.982063
count_acc_4 = 0.939394
count_acc_5 = 0.972973
count_confusion = 3->3=219, 3->4=4, 4->3=3, 4->4=62, 4->5=1, 5->4=2, 5->5=72
```

Reporting-only final-test result selected from official-val:

```text
summary = D:/gcsxxl/1/gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt5only_gt4w15_v1_official_best_test_best_from_val/tusimple_official_summary.json
decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=18.0, max_det=8, min_points=2, valid_before_maxdet=false
official_acc = 0.963755
official_score = 0.962489
official_FP = 0.032213
official_FN = 0.031093
count_acc = 0.872753
count_acc_3 = 0.971264
count_acc_4 = 0.602564
count_acc_5 = 0.797891
count_confusion = 2->2=2, 2->3=3, 3->2=7, 3->3=1690, 3->4=40, 3->5=3, 4->2=1, 4->3=111, 4->4=282, 4->5=71, 4->6=3, 5->3=25, 5->4=73, 5->5=454, 5->6=17
```

Integrated conclusion:

- Supported fact: the official-val result is the strongest seen in this local
  review set: `official_acc=0.975185`, with both GT4 false-fifth and GT5
  undercount controlled (`4->5=1`, `5->4=2`).
- Supported fact: the training-time `official_best` row was already strong
  under the intended valid-before decode (`official_acc=0.974056`, `4->5=1`,
  `5->4=2`).
- Supported fact: the post-hoc val sweep's selected row changes decode away
  from the intended valid-before policy (`valid_before_maxdet=false`,
  `max_det=8`, `point_valid_thr=0.6`). This is valid official-val selection
  evidence, but it should be called out when comparing to the planned
  valid-before experiment.
- Supported fact: final-test reporting does not confirm the val gain:
  `official_acc=0.963755`, below the previous reporting-only
  `dupmargin005` test result `0.965702` and below older
  `count03_under5_03` `0.965459`.
- Supported fact: final-test count errors remain broad: `4->3=111`,
  `4->5=71`, `5->4=73`, `5->6=17`. Count accuracy matches the old
  `count03_under5_03` level but does not improve final-test ACC.
- Caveat: this run's `args.yaml` records `gcs_count=0.0` and
  `gcs_count_under5=0.0`; it fine-tunes from E1 but is not a pure
  count03/under5-continuation ablation. That may contribute to the val/test
  count generalization gap.

Next action:

Do not tune thresholds, `valid_before_maxdet`, `max_det`, NMS, checkpoint, or
loss weights from this final-test result. If continuing this line, rerun the
same GT5-only protect + GT4W15 idea with the E1 count losses preserved
(`gcs_count=0.3`, `gcs_count_under5=0.3`) and compare only on official-val
before any further test reporting. Also run a matched official-val sweep that
separates the intended valid-before decode from the post-hoc
`valid_before_maxdet=false` decode.

## 2026-07-02: Reject Short-Side Geometry GT5-Only + GT4W15 Follow-Up

Decision:

Do not promote
`gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1`.
Treat it as diagnostic evidence that matched short-side geometry supervision
can improve the 363-image official-val surface, but that improvement does not
generalize to official test. The test result remains reporting-only and must
not be used for threshold, checkpoint, decode, or loss tuning.

Training/config evidence:

```text
run output = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1
args name = gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1-3
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
gcs_count = 0.3
gcs_count_under5 = 0.3
gcs_count_boundary = 0.2
gcs_spurious_neg = 0.1
gcs_spurious_gt4_weight = 1.5
gcs_spurious_gt5_weight = 1.0
gcs_spurious_gt_protect = true
gcs_short_side_geom = 0.25
gcs_short_side_geom_side_only = true
gcs_short_side_geom_visible_max = 20
gcs_official_best = true
gcs_official_valid_before_maxdet = true
official_best_epoch = 143
```

Training-time official-best evidence:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1/weights/official_best_sweep.json
decode = conf=0.005, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_acc = 0.976719
official_score = 0.976298
official_FP = 0.011203
official_FN = 0.009871
count_acc = 0.972452
count_acc_3 = 0.977578
count_acc_4 = 0.939394
count_acc_5 = 0.986486
count_confusion = 3->3=218, 3->4=5, 4->3=3, 4->4=62, 4->5=1, 5->4=1, 5->5=73
```

External official-val sweep on `official_best.pt`:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1_official_best_val_sweep_valid_before_maxdet/tusimple_official_sweep_summary.json
surface = canonical official-val 363
rows = 1890
decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=4, valid_before_maxdet=true
official_acc = 0.977032
official_score = 0.976599
official_FP = 0.010882
official_FN = 0.010790
count_acc = 0.977961
count_acc_3 = 0.982063
count_acc_4 = 0.954545
count_acc_5 = 0.986486
count_confusion = 3->3=219, 3->4=4, 4->3=3, 4->4=63, 5->4=1, 5->5=73
```

Reporting-only final-test result selected from the external official-val row:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1_official_best_test_best_from_val/tusimple_official_summary.json
decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=4, valid_before_maxdet=true
official_acc = 0.963348
official_score = 0.962070
official_FP = 0.032171
official_FN = 0.031722
count_acc = 0.874191
count_acc_2 = 0.600000
count_acc_3 = 0.967816
count_acc_4 = 0.617521
count_acc_5 = 0.801406
count_confusion = 2->2=3, 2->3=2, 3->2=8, 3->3=1684, 3->4=43, 3->5=5, 4->3=107, 4->4=289, 4->5=72, 5->3=28, 5->4=85, 5->5=456
```

Same-line no-shortside comparator:

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_v1
external official-val ACC = 0.974538
external official-val FP/FN = 0.012856 / 0.012626
external official-val count_acc_4/5 = 0.924242 / 0.972973
reporting-only test ACC = 0.964358
reporting-only test FP/FN = 0.030721 / 0.029415
reporting-only test count_acc_4/5 = 0.611111 / 0.836555
test GT5 confusion = 5->3=24, 5->4=69, 5->5=476
```

Integrated conclusion:

- Supported fact: short-side geometry improves the 363-image official-val
  surface over the no-shortside same-line run: `official_acc +0.002494`,
  `official_FP -0.001974`, `official_FN -0.001836`, `count_acc_4 +0.030303`,
  and `count_acc_5 +0.013513`.
- Supported fact: the same comparison is negative on reporting-only official
  test: `official_acc -0.001010`, `official_FP +0.001450`,
  `official_FN +0.002307`, and `count_acc_5 -0.035149`.
- Supported fact: the official-test regression is mainly a GT5 retention
  failure. Versus the no-shortside comparator, test `5->4` grows from `69` to
  `85`, and `5->5` falls from `476` to `456`.
- Supported fact: the short-side geometry term mostly trains GT4 side lanes,
  not GT5. The last 20 recorded epochs average
  `train/short_side_geom_gt4 ~= 32.83` and
  `train/short_side_geom_gt5 ~= 3.41`; validation averages are
  `val/short_side_geom_gt4 = 64.0` and `val/short_side_geom_gt5 ~= 3.67`.
- Supported fact: the 363-image official-val split is not representative of
  the test date distribution. The 363 split contains `0313-1`, `0313-2`,
  `0531`, and `0601`, with no `0530`; all 74 GT5 validation images are from
  `0601`. Official test contains `0530=1248`, `0531=715`, and `0601=819`
  images, including `0530|GT4=233` and `0530|GT5=84`.
- Supported fact: the largest reporting-only test weak spots include
  `0530|GT4` (`ACC=0.941485`, predicted `3` lanes for `88/233` images) and
  `0530|GT5` (`ACC=0.942496`, predicted `3/4` lanes for `71/84` images).
- Hypothesis: the matched-only short-side geometry loss improved the
  side-lane geometry/count shape that is visible on the canonical 363 split,
  but did not improve global lane-count calibration or GT5 retention on the
  broader test distribution. The later selected epoch (`143` versus the
  no-shortside comparator's `75`) may also reflect stronger fitting to the
  official-val surface.

Next action:

Do not tune the external-sweep decode (`conf=0.003`,
`point_valid_thr=0.6`, `min_points=4`), checkpoint, NMS, or short-side loss
weights from the test report. If continuing this line, keep final test closed
and use official-val/train-val diagnostics to inspect GT5 short-side matched
geometry, point-valid survival, and count-score calibration, especially by
date and GT lane count. Any follow-up must prove on official-val/train-val
that it preserves GT5 while keeping the GT4/FP benefit.

## 2026-07-03: Reject Far-Spurious GT3/GT4 Follow-Up as Promotion

Decision:

Do not promote
`gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_farspur005_v1`.
Treat it as diagnostic evidence that far unmatched-query suppression can reduce
GT4 false-fifth lanes, but the reporting-only official test does not confirm a
candidate improvement and shows a GT5 retention tradeoff. The test result must
not be used to tune thresholds, NMS, checkpoint choice, or loss weights.

Training/config evidence:

```text
run = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_farspur005_v1
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
gcs_count = 0.3
gcs_count_under5 = 0.3
gcs_count_boundary = 0.2
gcs_spurious_neg = 0.1
gcs_spurious_gt4_weight = 1.5
gcs_spurious_gt5_weight = 1.0
gcs_spurious_gt_protect = true
gcs_spurious_gt_protect_min_gt_lanes = 5
gcs_far_spurious_neg = 0.05
gcs_far_spurious_min_gt_lanes = 3
gcs_far_spurious_max_gt_lanes = 4
gcs_far_spurious_gt3_weight = 1.0
gcs_far_spurious_gt4_weight = 1.0
gcs_far_spurious_gt5_weight = 0.0
gcs_official_best = true
gcs_official_valid_before_maxdet = true
official_best_epoch = 145
```

Training-time official-best evidence:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_farspur005_v1/weights/official_best_sweep.json
decode = conf=0.005, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_acc = 0.977374
official_score = 0.977004
official_FP = 0.009550
official_FN = 0.008953
count_acc = 0.975207
count_acc_3 = 0.986547
count_acc_4 = 0.939394
count_acc_5 = 0.972973
count_confusion = 3->3=220, 3->4=3, 4->3=3, 4->4=62, 4->5=1, 5->4=2, 5->5=72
```

External official-val sweep on `official_best.pt`:

```text
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_farspur005_v1_official_best_val_sweep_valid_before_maxdet/tusimple_official_sweep_summary.json
surface = canonical official-val 363
rows = 1890
decode = conf=0.003, point_valid_thr=0.575, nms_dist_px=50.0, max_det=5, min_points=2, valid_before_maxdet=true
official_acc = 0.977758
official_score = 0.977415
official_FP = 0.008219
official_FN = 0.008953
count_acc = 0.975207
count_acc_3 = 0.982063
count_acc_4 = 0.954545
count_acc_5 = 0.972973
count_confusion = 3->2=1, 3->3=219, 3->4=3, 4->3=3, 4->4=63, 5->4=2, 5->5=72
```

Reporting-only final-test result selected from the external official-val row:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_farspur005_v1_official_best_test_best_from_val/tusimple_official_summary.json
decode = conf=0.003, point_valid_thr=0.575, nms_dist_px=50.0, max_det=5, min_points=2, valid_before_maxdet=true
official_acc = 0.963907
official_score = 0.962643
official_FP = 0.031860
official_FN = 0.031332
count_acc = 0.882459
count_acc_2 = 0.400000
count_acc_3 = 0.974138
count_acc_4 = 0.628205
count_acc_5 = 0.815466
count_confusion = 2->2=2, 2->3=3, 3->2=5, 3->3=1695, 3->4=37, 3->5=3, 4->3=109, 4->4=294, 4->5=65, 5->3=28, 5->4=77, 5->5=464
```

Same-line comparators:

```text
no-shortside run = gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_v1
no-shortside official-val ACC = 0.974538
no-shortside official-val FP/FN = 0.012856 / 0.012626
no-shortside official-val count_acc_4/5 = 0.924242 / 0.972973
no-shortside reporting-only test ACC = 0.964358
no-shortside reporting-only test FP/FN = 0.030721 / 0.029415
no-shortside reporting-only test count_acc_4/5 = 0.611111 / 0.836555
no-shortside test count_confusion includes 4->5=75, 5->4=69, 5->5=476

shortsidegeom025 official-val ACC = 0.977032
shortsidegeom025 reporting-only test ACC = 0.963348
shortsidegeom025 test count_confusion includes 4->5=72, 5->4=85, 5->5=456
```

Integrated conclusion:

- Supported fact: the far-spurious loss was active and not a no-op. In
  `results.csv`, train `far_spur_neg` averaged about `2.56` over the first 20
  epochs and about `0.024` over the last 20 epochs; `far_spur_gt5` stayed
  `0.0`, matching the intended GT3/GT4-only pressure.
- Supported fact: on canonical official-val, the external selected row is very
  strong: `official_acc=0.977758`, `FP=0.008219`, `FN=0.008953`, with GT4
  false-fifth removed (`4->5=0`) and GT5 undercount held to `5->4=2`.
- Supported fact: the full 1890-row official-val sweep is broadly strong, not a
  single lucky row. A lower-FP row exists (`FP=0.005923`) but worsens count
  shape (`count_acc_4=0.924242`, `count_acc_5=0.959459`, `5->4=3`), so it is
  not a better selection.
- Supported fact: reporting-only official test is below the same-line
  no-shortside comparator: `official_acc -0.000451`, `FP +0.001139`,
  `FN +0.001917`.
- Supported fact: the test tradeoff is the intended GT4 benefit plus GT5
  collateral damage. Versus no-shortside, GT4 improves (`4->5: 75 -> 65`,
  `count_acc_4: 0.611111 -> 0.628205`), but GT5 retention worsens
  (`5->4: 69 -> 77`, `5->5: 476 -> 464`,
  `count_acc_5: 0.836555 -> 0.815466`).
- Hypothesis: far-spur suppresses the far extra-lane pattern visible on
  official-val and partly on test GT4, but it does not solve the broader
  date/count generalization gap. It likely shifts some borderline dense-lane
  examples toward undercount on the broader test distribution.

Next action:

Do not promote this run and do not tune from its final-test report. If this
line continues, keep final test closed and run a train/val-only diagnostic
against the same-line no-shortside comparator: compare raw unmatched candidates,
post-NMS survival, nearest-GT distance, and point-valid spans for GT4 and GT5
groups. Use the reporting-only test date/count breakdown only as a risk
description; first reproduce comparable GT4 over-count and GT5 under-count
failure modes on train/official-val before designing or selecting another
far-spur ablation. Any future far-spur ablation must prove GT5 retention on
official-val/train-val diagnostics while preserving the GT4 false-fifth
reduction.

## 2026-07-02: Integrate Reporting-Only Test for E1 and Spurious-Lite + Valid-Before

Decision:

Record the user-requested official test calculation for two existing
`weights/best.pt` artifacts as reporting-only evidence. Both test runs used
decode parameters selected on official-val before touching test. Do not use
these test summaries to tune thresholds, `valid_before_maxdet`, `max_det`, NMS,
checkpoint choice, loss weights, or count policy.

Compatibility note:

Both checkpoints are legacy query-mode pickles that lack the newer
`GCSLaneHead.gcs_mode` attribute. The server evaluation used
`.tmp/eval_official_legacy_query.py`, which wraps the normal
`tools/eval_tusimple_official.py` path and only patches missing
`GCSLaneHead.gcs_mode = "query"` after loading. It does not change weights,
decode parameters, predictions, labels, or official metrics.

E1 count-boundary test report:

```text
run = gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1
weights = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
official-val sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1_official_val_sweep/tusimple_official_sweep_summary.json
test summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1_official_test_best_from_val/tusimple_official_summary.json
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=2, valid_before_maxdet=false
images = 2782
official_acc = 0.965428
official_score = 0.964279
official_FP = 0.031368
official_FN = 0.026060
count_acc = 0.874551
count_acc_3 = 0.971839
count_acc_4 = 0.568376
count_acc_5 = 0.834798
count_confusion = 2->2=1, 2->3=2, 2->4=2, 3->2=2, 3->3=1691, 3->4=41, 3->5=6, 4->3=105, 4->4=266, 4->5=94, 4->6=3, 5->3=26, 5->4=46, 5->5=475, 5->6=22
```

Spurious-lite + valid-before test report:

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1
weights = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1/weights/best.pt
official-val focused sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1_official_val_focused_sweep_valid_before_maxdet/tusimple_official_sweep_summary.json
test summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1_official_test_best_from_val_valid_before_maxdet/tusimple_official_summary.json
decode = conf=0.005, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
images = 2782
official_acc = 0.965483
official_score = 0.964313
official_FP = 0.030847
official_FN = 0.027678
count_acc = 0.882818
count_acc_3 = 0.971839
count_acc_4 = 0.606838
count_acc_5 = 0.843585
count_confusion = 2->2=1, 2->3=3, 2->4=1, 3->2=2, 3->3=1691, 3->4=43, 3->5=4, 4->3=105, 4->4=284, 4->5=79, 5->3=26, 5->4=63, 5->5=480
```

Integrated conclusion:

- Supported fact: among these two user-requested active-branch `best.pt`
  reports, spurious-lite + valid-before is marginally higher on official test
  ACC (`0.965483` vs `0.965428`) and official score (`0.964313` vs
  `0.964279`).
- Supported fact: the spurious-lite + valid-before test result improves
  `count_acc` (`0.882818` vs `0.874551`) and GT4 count accuracy
  (`0.606838` vs `0.568376`) while reducing GT4 false fifth lanes
  (`4->5: 94 -> 79`).
- Supported fact: the tradeoff is FN: spurious-lite + valid-before has higher
  `official_FN` (`0.027678` vs `0.026060`) and more GT5 undercount to four
  lanes (`5->4: 46 -> 63`), even though total GT5 correct predictions improve
  slightly (`475 -> 480`) because `max_det=5` removes the E1 `5->6` cases.
- Supported fact: the test margin is very small (`+0.000055` ACC). Treat it as
  reporting evidence consistent with the official-val selected decode, not as
  a reason to search test thresholds or claim a broad algorithmic fix.
- Caveat: both artifacts were trained without training-time `official_best`
  preservation (`gcs_official_best=false`) and were selected by post-hoc
  official-val sweeps over `weights/best.pt`. Future formal candidates should
  use `weights/official_best.pt` plus `weights/official_best_decode.yaml`.

Smallest safe next action:

Keep test closed for further tuning. If this line continues, use official-val
and train/val diagnostics only, with training-time official-best enabled. The
next change should target the remaining GT4/GT5 count generalization gap
without selecting any new decode or checkpoint from these test reports.

## 2026-06-28: Reject E2 Short0601 Hard-Sampling Run

Decision:

Do not promote `gcs_yolo_lane_s_q12_k56_boundary02_short0601_v1` and do not
run final test for it. Treat it as diagnostic evidence that the branch-local
train-only hard sampler can reduce GT4 false-fifth over-count, but with too
large an FN / official-ACC cost.

Official-val evidence:

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_short0601_v1
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
gcs_count_boundary = 0.2
gcs_hard_sampling = true
gcs_official_best = false
results.csv rows = 64 epochs
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_short0601_v1_official_val_sweep/tusimple_official_sweep_summary.json
sweep rows = 3360
best decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=2
official_acc = 0.969988
official_score = 0.969161
official_FP = 0.022544
official_FN = 0.018825
count_acc = 0.966942
count_acc_3 = 0.968610
count_acc_4 = 0.969697
count_acc_5 = 0.959459
count_confusion = 3->3=216, 3->4=7, 4->3=1, 4->4=64, 4->5=1, 5->4=3, 5->5=71
```

Compared with E1
`gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1`:

```text
official_acc: 0.971208 -> 0.969988 (-0.001220)
official_score: 0.970469 -> 0.969161 (-0.001308)
official_FP: 0.022957 -> 0.022544 (-0.000413)
official_FN: 0.014004 -> 0.018825 (+0.004821)
count_acc_4: 0.848485 -> 0.969697 (+0.121212)
count_acc_5: 0.972973 -> 0.959459 (-0.013514)
4->5: 9 -> 1
5->4: 2 -> 3
```

Integrated conclusion:

- Supported fact: the hard sampler strongly improves the intended GT4 count
  shape: `4->5` drops from `9` to `1`, and `count_acc_4` rises from
  `0.848485` to `0.969697`.
- Supported fact: the primary official-val surface rejects the run:
  `official_acc` falls by `0.001220`, mostly because `official_FN` rises by
  `0.004821`.
- Supported fact: GT5 retention is only mildly worse than E1 (`5->4: 2 -> 3`),
  so this is not the same failure mode as spurious-lite's GT5 collapse.
- Caveat: this artifact used a post-hoc official-val sweep over
  `weights/best.pt`; `args.yaml` records `gcs_official_best=false`, so no
  training-time `official_best.pt` / `official_best_decode.yaml` was selected.

Smallest safe next action:

Do not tune thresholds or run final test from this checkpoint. Keep E3-lite
spurious-negative experiments initialized from E1, not from this E2 hard
sampling run. If the hard-sampling line is revisited, first diagnose why the
FN increase appears despite the cleaner GT4 count shape.

## 2026-06-28: Reject GT4-Strong + GT5-Safe Spurious Run

Decision:

Do not promote `gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt4strong_gt5safe_v1`
and do not run final test for it. Classify it as a rejected experimental
candidate, with limited diagnostic value for GT5 retention only.

Official-val evidence:

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt4strong_gt5safe_v1
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
gcs_spurious_neg = 0.1
gcs_spurious_gt3_weight = 1.0
gcs_spurious_gt4_weight = 1.5
gcs_spurious_gt5_weight = 0.25
gcs_spurious_disable_gt5 = false
gcs_hard_sampling = false
gcs_official_best = false
results.csv rows = 43 epochs
nonfinite_count = 0
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt4strong_gt5safe_v1_official_val_sweep/tusimple_official_sweep_summary.json
best decode = conf=0.003, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=2
official_acc = 0.970530
official_score = 0.969686
official_FP = 0.025666
official_FN = 0.016529
count_acc = 0.950413
count_acc_4 = 0.848485
count_acc_5 = 0.986486
count_confusion = 3->3=216, 3->4=7, 4->3=1, 4->4=56, 4->5=9, 5->4=1, 5->5=73
```

Gate result:

Across 1512 official-val sweep rows, no row satisfies the requested gate:

```text
required official_acc >= 0.9717; observed max = 0.970530
required official_FP <= 0.018; observed min = 0.020707
required count_acc_4 >= 0.90; observed max = 0.893939
required count_acc_5 >= 0.94; observed max = 0.986486
required 4->5 <= 5; observed min = 6
required 5->4 <= 4; observed min = 1
```

Integrated conclusion:

- Supported fact: the GT5-safe side is preserved: the selected row has
  `count_acc_5=0.986486` and `5->4=1`.
- Supported fact: the GT4-strong side did not transfer to official-val:
  selected `4->5=9`, matching E1 and worse than spurious-lite's `4->5=4`.
- Supported fact: the primary official metrics regress versus E1
  (`official_acc 0.971208 -> 0.970530`, `FP 0.022957 -> 0.025666`,
  `FN 0.014004 -> 0.016529`).
- Supported fact: the sweep cannot be rescued by threshold/NMS selection under
  the recorded search grid because even the best individual limits miss ACC,
  FP, `count_acc_4`, and `4->5` gates.
- Caveat: this artifact used a post-hoc sweep over `weights/best.pt`
  (`gcs_official_best=false`) and stopped with 43 result rows despite
  `epochs=60`. That weakens it as a formal run but does not change the reject
  decision, because official-val is already below all promotion gates.

Smallest safe next action:

Do not tune on test and do not run final test for this checkpoint. Run a
read-only official-val failure trace for `GT4 4->5` and `GT3 3->4` extra lanes
to check whether GT4 spurious candidates are being selected too rarely. The
training logs show low late GT4 candidate density
(`train/spur_cnt_gt4` around `0.03` to `0.04` per logged batch near the end),
so increasing only the GT4 weight may not help if candidate selection has too
little GT4 coverage.

## 2026-06-28: Add GT4-Strong + GT5-Safe Spurious Weighting

Decision:

Extend the default-off E3-lite `gcs_spurious_neg` loss from GT5-only weighting
to GT-count weighting:

```text
gcs_spurious_gt3_weight = 1.0
gcs_spurious_gt4_weight = 1.0
gcs_spurious_gt5_weight = 1.0
```

When `gcs_spurious_neg` is enabled, each image derives `gt_lanes` from
`batch["num_lanes"]` or from `lane_valid` without changing the dataloader. If
`gt_lanes >= 5` and `gcs_spurious_disable_gt5=true`, the image is skipped for
spurious-negative loss. Otherwise selected unmatched duplicate-like queries use
per-image weights:

```text
gt_lanes <= 3: gcs_spurious_gt3_weight
gt_lanes == 4: gcs_spurious_gt4_weight
gt_lanes >= 5: gcs_spurious_gt5_weight
```

The selected target-zero BCE terms are normalized by selected spurious-query
count, not by weight sum, so `gcs_spurious_gt4_weight=1.5` strengthens GT4
pressure and `gcs_spurious_gt5_weight=0.25` weakens GT5 pressure. Defaults all
remain `1.0`, preserving old spurious-lite behavior.

Why:

The completed spurious-lite run reduced FP and `4->5`, but misclassified true
GT5 short lanes as spurious (`5->4=9`). The later GT5-safe setting recovered
GT5 but let GT4 over-count rebound. The next experiment should keep the
existing unmatched-query selection logic while applying stronger GT4 negative
pressure and weaker GT5 negative pressure:

```text
--gcs-spurious-neg 0.1
--gcs-spurious-gt3-weight 1.0
--gcs-spurious-gt4-weight 1.5
--gcs-spurious-gt5-weight 0.25
```

Scope:

This is a loss-weighting and logging extension only. It does not change data
sampling, dataset labels, Hungarian matcher behavior, point/smooth/curve
losses, decode, NMS, or official metrics.

Selection target:

Select only on official-val. Compare `official_acc`, `official_FP`,
`official_FN`, `count_acc_4`, `count_acc_5`, `4->5`, and `5->4` against E1,
spurious-lite, and GT5-safe.

## 2026-06-28: Treat E3-Lite Spurious Negative as Diagnostic-Only

Decision:

Do not promote the completed E3-lite spurious-negative run
`gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1`. Continue the ablation
with the GT5-safe variant instead of running final test for the spurious-lite
checkpoint.

Official-val evidence:

```text
E1 baseline run:
  gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1
  sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1_official_val_sweep/tusimple_official_sweep_summary.json
  decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=2
  official_acc = 0.971208
  official_score = 0.970469
  official_FP = 0.022957
  official_FN = 0.014004
  count_acc = 0.947658
  count_acc_4 = 0.848485
  count_acc_5 = 0.972973
  count_confusion = 4->3=1, 4->4=56, 4->5=9, 5->4=2, 5->5=72

E3-lite spurious run:
  gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1
  sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1_official_val_sweep/tusimple_official_sweep_summary.json
  training log = 59 logged epochs in results.csv; gcs_official_best=false
  sweep rows = 3360, count_aware_topk=false
  decode = conf=0.005, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=2
  official_acc = 0.971721
  official_score = 0.971131
  official_FP = 0.013866
  official_FN = 0.015611
  count_acc = 0.953168
  count_acc_4 = 0.924242
  count_acc_5 = 0.878378
  count_confusion = 3->3=220, 3->4=3, 4->3=1, 4->4=61, 4->5=4, 5->4=9, 5->5=65
```

Integrated conclusion:

- Supported fact: spurious-lite improves official-val ACC by `+0.000513`,
  official score by `+0.000662`, and reduces official FP by `-0.009091`.
- Supported fact: it directly improves the target false-fifth pattern:
  `4->5` drops from `9` to `4`, and `count_acc_4` rises from `0.848485` to
  `0.924242`.
- Supported fact: the collateral GT5 regression is large: `5->4` rises from
  `2` to `9`, and `count_acc_5` drops from `0.972973` to `0.878378`.
- Supported fact: the run started from E1
  `runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt`
  and kept `gcs_hard_sampling=false`.
- Caveat: this completed run used a post-hoc official-val sweep over
  `weights/best.pt`; its `args.yaml` records `gcs_official_best=false`, and no
  `official_best.pt` / `official_best_decode.yaml` artifact was found.
- Caveat: the best metrics have a 36-row tie under the branch selection
  priority across `conf in {0.005, 0.01, 0.02}`, all swept NMS values, and all
  swept `max_det` values, with `point_valid_thr=0.45` and `min_points=2`.
  Treat the recorded decode as a representative selected row, not a unique
  postprocess preference.

Why:

The E3-lite mechanism validates the hypothesis that unmatched short
duplicate-like suppression can reduce GT4 false fifth lanes and FP. It also
shows that the same selection rule can treat a true short fifth lane as
spurious. The GT5 collateral cost is too large for promotion even though the
primary official-val ACC improves.

Next action:

Keep final test closed for `spurious_lite_v1`. Complete or relaunch the
GT5-safe follow-up from E1 `weights/best.pt` with:

```text
--gcs-spurious-neg 0.1
--gcs-spurious-gt5-weight 0.25
```

Then run the same 363-image official-val sweep. Promotion should require
official-val ACC not below E1, most of the FP / `4->5` benefit retained, and
`5->4` materially below the spurious-lite value of `9`.

Completed GT5-safe follow-up evidence:

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt5safe_v1
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
gcs_spurious_gt5_weight = 0.25
gcs_hard_sampling = false
results.csv rows = 52 epochs plus header
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt5safe_v1_official_val_sweep/tusimple_official_sweep_summary.json
best decode = conf=0.005, point_valid_thr=0.55, nms_dist_px=18.0, max_det=5, min_points=2
official_acc = 0.971777
official_score = 0.971006
official_FP = 0.022957
official_FN = 0.015611
count_acc = 0.958678
count_acc_4 = 0.863636
count_acc_5 = 0.986486
count_confusion = 3->3=218, 3->4=5, 4->3=1, 4->4=57, 4->5=8, 5->4=1, 5->5=73
```

This run recovers GT5 retention (`5->4=1`) but loses most of the spurious-lite
GT4 / FP benefit: `4->5` rebounds to `8`, `count_acc_4` is only `0.863636`,
and `official_FP` returns to the E1 value `0.022957`. Across the sweep there is
no row with both `4->5 <= 4` and `5->4 <= 2`. Treat it as diagnostic evidence
for GT-count weighting, not as a clean promotion.

## 2026-06-27: Add GT5-Safe Spurious Negative Control

Decision:

Extend the default-off E3-lite `gcs_spurious_neg` loss with GT5-safe controls:

```text
gcs_spurious_gt5_weight = 1.0
gcs_spurious_disable_gt5 = false
```

When `gcs_spurious_neg` is enabled, each image now derives `gt_lanes` from
`batch["num_lanes"]` or from `lane_valid` without changing the dataloader. If
`gt_lanes >= 5` and `gcs_spurious_disable_gt5=true`, the image is skipped for
spurious-negative loss. Otherwise GT5-or-denser images multiply their selected
spurious-negative terms by `gcs_spurious_gt5_weight`.

Why:

The E1 + spurious-lite result improved official-val ACC and reduced FP, but
`count_acc_5` regressed with `5->4=9`, indicating that the short true fifth
side lane can look like a short duplicate-like spurious negative. The next
GT5-safe run should keep the existing spurious-negative selection logic while
reducing collateral pressure on true GT5 lanes:

```text
--gcs-spurious-neg 0.1
--gcs-spurious-gt5-weight 0.25
```

Scope:

This is a loss-weighting and logging extension only. It does not change data
sampling, dataset labels, Hungarian matcher behavior, point/smooth/curve
losses, decode, NMS, or official metrics. The defaults preserve the original
spurious-lite behavior.

New diagnostics:

```text
spur_cnt_gt3
spur_cnt_gt4
spur_cnt_gt5
spur_neg_gt3
spur_neg_gt4
spur_neg_gt5
```

Select any candidate only on official-val. The target tradeoff is lower GT5
`5->4` than spurious-lite, official ACC not below E1, and no large FP rebound.

## 2026-06-27: Add Default-Off E3-Lite Spurious Negative Loss

Decision:

Add a default-disabled `gcs_spurious_neg` loss to `GCSLoss` for the E3-lite
follow-up from the E1 count-boundary checkpoint.

When enabled, the loss uses Hungarian matcher `src_idx` only to identify
matched queries, then treats all other queries as unmatched. An unmatched query
is selected as a spurious short duplicate when:

```text
pred_valid_count >= 2
pred_valid_count <= gcs_spurious_max_points
overlap with any matched query >= gcs_spurious_min_overlap
mean overlapping x distance <= gcs_spurious_close_px pixels
```

Selected unmatched queries receive an extra target-zero
`BCEWithLogits(pred_logits, 0)`. The weighted objective contribution is:

```text
gcs_spurious_neg * gcs_spurious_neg_weight * spurious_neg_loss
```

Defaults:

```text
gcs_spurious_neg = 0.0
gcs_spurious_neg_weight = 1.0
gcs_spurious_max_points = 12
gcs_spurious_close_px = 30.0
gcs_spurious_min_overlap = 3
```

Scope:

This is a branch-local loss/logging option only. It does not change data
sampling, dataset labels, matcher logic, point loss, smooth loss, curve loss,
decode, NMS, official metrics, Count Head, Quality Head, Survival Head, or
near-miss machinery. When `gcs_spurious_neg > 0`, `pred_valid_logits` is
required and missing per-point visibility logits must fail fast.

Experiment protocol:

Run E3-lite from the E1 count-boundary `best.pt`
(`gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1`), not from E2 hard
sampling or count-aware top-k results. Keep `gcs_hard_sampling` disabled.
Select checkpoint/decode on official-val and compare against E1 on
`official_acc`, `official_FP`, `official_FN`, `count_acc_4`, `count_acc_5`,
and `count_confusion` `4->5` / `5->4`.

## 2026-06-27: Add Default-Off Count-Aware Top-K Decode Ablation

Decision:

Add a default-disabled inference/evaluation postprocess option,
`count_aware_topk`, to the shared GCS decode helper and expose it through
`tools/infer_gcs.py`, `tools/eval_gcs.py`,
`tools/eval_tusimple_official.py`, and `tools/sweep_tusimple_official.py`.

When enabled, decode computes:

```text
count_score = sigmoid(pred_logits).sum()
k_hat = round(count_score)
k_hat = clip(k_hat, count_aware_min_k, count_aware_max_k)
```

Default bounds:

```text
count_aware_min_k = 3
count_aware_max_k = 5
count_aware_length_norm = 12
```

After ordinary confidence filtering and Lane-NMS, decoded candidate lanes are
ranked by:

```text
quality = exist_prob * mean(point_valid_prob_on_visible_segment) * length_factor
length_factor = min(visible_count / count_aware_length_norm, 1.0)
```

Only the quality-best `k_hat` lanes are kept. If the candidate count is below
`k_hat`, all candidates are retained.

Scope:

This is a postprocess ablation only. It does not change training, labels,
losses, model outputs, matcher behavior, official metrics, Count Head, Quality
Head, Survival Head, or near-miss machinery. The option is disabled by default,
so existing inference, eval, and official sweep behavior remains unchanged.

Validation target:

Compile changed Python files, run a synthetic decode check proving default-off
keeps the old lane count while count-aware top-k trims to dynamic `k_hat`, then
select any candidate only on official-val. The requested E3 official-val
normal/count-aware sweeps require the remote E3 `best.pt` checkpoint.

## 2026-06-27: Add Default-Off Train-Only Hard Sampling

Decision:

Add a default-disabled `gcs_hard_sampling` option that uses
`WeightedRandomSampler(replacement=True, num_samples=len(train_dataset))` for
the training dataloader only.

Weights:

```text
date == 0601: x2.0
GT4 and min_visible_points <= 10: x4.0
GT5 and min_visible_points <= 10: x3.0
GT3 and min_visible_points <= 20: x1.5
date == 0313-2 and GT4 and min_visible_points <= 10: x4.0
```

The sampler reads `lanes`, `lane_valid`, `num_lanes`, and optional `raw_file`
from each train label npz. It parses TuSimple date from `raw_file` first and
falls back to the image path. It logs hard-group counts plus weight
min/mean/max at training start.

Scope:

This is a train-dataloader sampling option only. It does not change labels,
validation/test dataloaders, point loss, smooth loss, curve loss, decode, model
outputs, or official metrics. It is not the legacy `gcs_gt4_short_boost`
sampler and it does not globally weight all GT4 samples.

Validation evidence:

Local validation covered Python compilation, CLI/config parsing, model shape,
a direct sampler unit check for hard-on/hard-off/val-loader behavior, and a
1-epoch synthetic 544x960 smoke with `gcs_count_boundary=0.2` and
`gcs_hard_sampling=True`. The smoke completed with finite train/val results and
logged hard-group statistics.

## 2026-06-27: Add Default-Off Count Boundary Loss

Decision:

Add a default-disabled `count_boundary_loss` to the existing GCS loss for
adjacent GT3/GT4/GT5 lane-count boundaries. The loss uses:

```text
count_score = sigmoid(pred_logits).sum(dim=1)
```

Boundaries:

```text
GT3: penalize count_score > 3.35
GT4: penalize count_score < 3.65 and count_score > 4.35
GT5: penalize count_score < 4.65
```

Defaults:

```text
gcs_count_boundary = 0.0
gcs_count_boundary_gt4_weight = 2.0
gcs_count_boundary_gt5_weight = 1.5
gcs_count_boundary_margin34 = 0.35
gcs_count_boundary_margin45 = 0.35
```

Why:

The user explicitly requested a targeted train-side loss for the current
`count03_under5_03` bottleneck: GT4 and GT5 adjacent count errors, especially
`GT4->5`, `GT4->3`, and `GT5->4`. The existing `gcs_count` and
`gcs_count_under5` losses constrain global count score but do not explicitly
shape these adjacent decision bands.

Scope:

This is a branch-local loss/logging option only. It does not add Count Head,
Quality Head, Survival Head, near-miss mining, matcher changes, point loss
changes, smooth loss changes, curve loss changes, or decode changes. The
default gain is `0.0`, so old experiments keep the previous weighted objective.

Validation evidence:

Local checks covered Python compilation, CLI/config parsing, model shape, a
loss-level default-off total-loss regression, and a 1-epoch synthetic
train/val smoke with `gcs_count_boundary=0.2`. The smoke wrote
`train/count_boundary_loss`, `val/count_boundary_loss`,
`train/count_score_mean`, and `val/count_score_mean` with finite values.

## 2026-06-27: Select Training Checkpoints by Official-Val Metric

Decision:

Add explicit training-time `official_best` checkpoint preservation for formal
TuSimple runs. The training hook runs a lightweight official-val sweep every 5
epochs and again on the final/early-stop epoch, then writes:

```text
weights/official_best.pt
weights/official_best_sweep.json
weights/official_best_decode.yaml
```

Selection priority:

```text
1. maximum official_acc
2. maximum official_score
3. lower official_FP
4. lower official_FN
5. higher count_acc_4
```

Why:

The user reported a concrete mismatch between internal validation surfaces and
the desired official selection surface:

```text
best val/f1:         epoch 91  = 0.971997
best val/total_loss: epoch 116 = 0.69639
last epoch 131:      val/f1    = 0.968382
```

This means selecting the final model only from `val/total_loss`, internal
`val/f1`, or generic `weights/best.pt` can choose the wrong checkpoint for
TuSimple reporting.

Scope:

This is a selection/protocol change only. It does not alter the 5-25-3 model
body, labels, losses, decode implementation, or official metric. It also does
not import Count Head, Quality Head, Survival Head, near-miss, or later
mainline K56 candidate machinery.

Validation target:

Compile `tools/train_gcs.py`, `tools/sweep_tusimple_official.py`,
`ultralytics/models/yolo/gcs_lane/train.py`, and `ultralytics/cfg/__init__.py`.
Full official-best behavior requires a remote TuSimple archive and should be
verified during the next formal training run, not with local 8GB-GPU training.

## 2026-06-27: Historical Rollback to b6535f641

Decision:

At that time, restore the source/config state to commit `b6535f641`
(`Fix GCS training progress header alignment`). This historical rollback is
superseded by the 2026-07-06 rollback to `424ab1c86`.

Scope:

The active code baseline at that time was the 5-25-3 K56 mainline at
`b6535f641`. Later
commits, including duplicate/spurious/ranking losses, GT3/GT4/GT5 follow-up
losses, Q18/Q20/dataref configs, Count Head, count-guided decode, side-aux
guards, GT4-hard diagnostics, paper-generation scripts, and their helper
checks, are retained below only as legacy experiment conclusions. They do not
describe currently available CLI flags, loss items, scripts, configs, model
outputs, or active selected candidates unless a future task explicitly restores
those mechanisms.

Why:

The user requested the code rollback while keeping later experiment content in
the documentation as old conclusions.

Mainline or experiment:

Superseded historical rollback decision. Later experiment sections remain
historical records.

## 2026-06-21: Roll Active Code Back to 50999d6af

Decision:

Restore the active source/config state to commit `50999d6af` (`Document 5-25-3
K56 as mainline`).

Scope:

The active code baseline at that point was the 5-25-3 K56 mainline contract at
`50999d6af`. Later commits and notes, including reusable count diagnostics, GT4
short-lane sampling, `extra_exist_loss`, short matched existence floor, and
reporting-only test batches, were retained below as legacy experiment
conclusions only. This decision is now superseded by the 2026-06-27
`b6535f641` rollback above.

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

## 2026-06-26: Reject Q18 as a GT4-Hard Geometry Fix

Decision:

Do not continue tuning the Q18 valid-loss weights from
`gcs_yolo_lane_s_q18_k56_side_gt4endpoint_validneg_countce_v1`. The first
diagnostic target was GT4-hard raw geometry, not official ACC, and Q18 did not
raise raw-query geometry recall on the fixed GT4-hard set.

中文记录：

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
- Next branch is Q20 side-geometry reference, preferably with side-slope
  template diversity rather than only adding two more bottom references.

Diagnostic protocol:

Use the train-derived `gt4pt025` GT4-hard set as a fixed input:

```text
hard set = data/tusimple_gt4_hard_val_gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025.txt
records = 19 images
old missing denominator = 22 GT lanes from gt4pt025
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
match gate = overlap >= 3 h-samples and mean_abs_x_error <= 20px
```

Project raw-query diagnostic over the 19 fixed hard images:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_q18_k56_side_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/diagnostic_all_current_missing/summary.json
current missing lanes = 20
drop_reason = geometry_bad 12, low_point_valid 8
raw_match_recall = 8/20 = 0.400000
after_point_valid_recall = 0/20 = 0.000000
final_decode_recall = 0/20 = 0.000000
```

Fixed old-missing recovery diagnostic, using the old `gt4pt025` 22 missing GT
lanes as the denominator:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_q18_k56_side_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/fixed_old_missing_recovery/summary.json
raw_match_recall = 9/22 = 0.409091
after_point_valid_recall = 1/22 = 0.045455
after_min_points_recall = 2/22 = 0.090909
after_conf_recall = 2/22 = 0.090909
final_decode_recall = 2/22 = 0.090909
new_status = geometry_bad 12, low_point_valid 8, recovered_final_decode 2
pred_count on 19 old hard images = 3 lanes: 16, 4 lanes: 3
```

Why:

The target gate was `raw_match_recall >= 0.55` as a starting point, ideally
`>= 0.65`. Q18 stayed at the previous `gt4pt025` fixed-hard raw recall
`9/22 = 0.409091`; the current-missing diagnostic is similarly low at
`8/20 = 0.400000`. `geometry_bad` remains the dominant residual failure
bucket. Although Q18 recovers two old missing lanes and slightly improves
point-valid survival on the fixed old-missing denominator, that is not enough
to show that the Q18 side references cover the real short side-lane geometry.

The attempted same-denominator `v2_validbranch_neg05-3` raw-query comparison
was blocked by a checkpoint/current-code contract mismatch:

```text
AttributeError: 'GCSLaneHead' object has no attribute 'count_mlp'
```

This does not affect the Q18 decision because Q18 already fails the raw
geometry gate.

Next action:

Keep the final-test path closed and do not tune ACC, `point_valid_thr`, NMS,
`max_det`, `min_points`, checkpoint choice, or loss gains from this diagnostic.
The next geometry-recall experiment should change the query coverage to Q20
side-geometry reference rather than continuing to tune the Q18 loss weights.
In particular, do not apply the valid-loss follow-up
(`gcs_gt4_short_valid_pos_weight=1.25` and
`gcs_unmatched_valid_neg_weight=0.5`) because that branch is reserved for the
case where raw geometry rises but point-valid does not.

## 2026-06-26: Reject Q20 as a GT4-Hard Geometry Fix

Decision:

Reject `gcs_yolo_lane_s_q20_k56_sidegeom_gt4endpoint_validneg_countce_v1` at
the hard diagnostic gate. Do not run the official-val normal/count-guided
sweep for this run, because the requested protocol says official ACC is only
checked after hard diagnostic passes.

Diagnostic protocol:

Use the same train-derived `gt4pt025` GT4-hard set as the Q18 check:

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
old missing GT lanes = 22
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
after_min_points_recall = 0/20 = 0.000000
after_conf_recall = 0/20 = 0.000000
final_decode_recall = 0/20 = 0.000000
```

Why:

Q20 only reaches the fixed old-missing starting line
`raw_match_recall >= 12/22`; it does not reach the true qualification gate
`raw_match_recall >= 13/22`, `after_point_valid_recall >= 4/22`, and
`final_decode_recall > 2/22`. The current-missing diagnostic is also below the
requested target: `raw_match_recall = 0.500000 < 0.55`,
`geometry_bad = 10 > 7`, and baseline `after_point_valid_recall = 0.0`.

Compared with Q18, Q20 reduces `geometry_bad` from `12` to `10`, but that is
not enough to show that side-geometry references solved candidate coverage.
The point-valid threshold sweep recovers at most one current missing lane at
lower thresholds, while the baseline valid stage stays at `0/20`; therefore
the valid-loss follow-up condition is not met.

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

Rejected actions:

- Do not run the Q20 official-val sweep from this checkpoint.
- Do not change `gcs_gt4_short_valid_pos_weight` to `1.25`.
- Do not change `gcs_unmatched_valid_neg_weight` to `0.5`.
- Do not tune loss, threshold, NMS, `max_det`, `min_points`, checkpoint choice,
  or final-test behavior from this diagnostic.

Next action:

Move from hand-designed Q20 side-geometry references to data-driven reference
clustering over the actual failed GT4-hard lane shapes. Start with the `10`
`geometry_bad` current-missing lanes, compare each failed lane against its
nearest raw queries and reference templates, and identify which side/order/slope
shapes are still uncovered. The next candidate should pass the hard
raw-geometry gate before any official-val sweep is run.

## 2026-06-26: Implement Q20-Dataref Reference Bank as Next Geometry Candidate

Decision:

Keep the Q20 sidegeom run rejected at the hard diagnostic gate and move the
next geometry experiment to Q20-dataref. Do not continue tuning Q20 sidegeom
loss weights.

Evidence:

```text
fixed old-missing:
raw_match_recall = 12/22 = 0.545455
after_point_valid = 2/22
final_decode = 2/22
geometry_bad = 10
low_point_valid = 10

current-missing:
raw_match_recall = 10/20 = 0.500000
geometry_bad = 10
after_point_valid = 0/20
```

Why:

Q20 sidegeom improved partial raw candidate coverage relative to Q18
(`9/22 -> 12/22` on the fixed old-missing denominator), but it only reached
the starting line and failed the promotion gate. It did not exceed Q18 at
final decode, and current-missing raw geometry stayed below the `0.55` target.

Rejected actions:

- Do not change `gcs_gt4_short_valid_pos_weight` to `1.25`.
- Do not change `gcs_unmatched_valid_neg_weight` to `0.5`.
- Do not enable `gcs_gt4_short_valid_recall`.
- Do not enable `gcs_gt4_short_valid_count_floor`.
- Do not run official-val or final-test selection from the rejected Q20
  sidegeom checkpoint.

Implementation direction:

Add an explicit Q20-dataref experiment config and `reference_mode="dataref"` in
`GCSLaneHead`. The dataref reference bank is:

```text
[4 left hard side templates] + [12 unchanged normal templates] + [4 right hard side templates]
```

The side templates are generated from true GT4-hard missing-lane shapes, with
valid spans interpolated and endpoints extended. A later checkpoint audit found
that `point_reference_logits` is a non-persistent buffer and is absent from the
audited sidegeom/dataref `best.pt` `state_dict()` values, so normal GCS
checkpoint loading does not overwrite the dataref initialization.

Promotion gate for the trained Q20-dataref candidate:

```text
fixed old raw_match_recall >= 14/22
fixed old after_point_valid >= 4/22
fixed old final_decode > 2/22
fixed old geometry_bad <= 7
current raw_match_recall >= 0.55
current geometry_bad <= 7
current after_point_valid > 0
```

Mainline or experiment:

Branch-local Q20 experiment only. The default Q12 model, Q18 config, and Q20
sidegeom config remain unchanged.

## 2026-06-26: Tighten Q20-Dataref Acceptance Chain

Decision:

Keep the Q20-dataref reference bank logic itself unchanged, but tighten the
contract checks around it before any formal training evidence is accepted.

Changes:

- `reference_mode="dataref"` is valid only for `num_queries=20` and
  `point_mode="fixed_y"`.
- `reference_mode="sidegeom"` is valid only for `num_queries=20`.
- The template builder deduplicates source lanes by default using normalized
  `raw_file + lane_id`, falling back to an x/valid signature when `lane_id` is
  missing.
- The builder reports duplicate counts and per-side unique lane counts, and it
  refuses to cluster when either side has fewer than four unique lanes by
  default.
- `--allow-duplicate-weighting` is debug-only. `--require-no-duplicates` is the
  clean-input check for formal sources.
- The reset-point-reference check must call the real
  `GCSLaneTrainer.load_gcs_pretrained()` path and prove that a normal probe
  tensor loads while `point_reference_logits` remains initialized from
  `Q20_DATAREF_X`.

Evidence caveat:

The current old/current missing-lane source has 42 rows but only 22 unique
source lanes, with 20 duplicate rows. Any reference-only coverage result from
that duplicate bank is debug evidence and must not be used as formal promotion
evidence. Formal Q20-dataref evidence requires a deduplicated rebuild or a
clean train-derived hard-lane source.

Rejected actions:

- Do not change the Q20 bottom bank/top-pull sidegeom logic.
- Do not change the Q18 reference branch.
- Do not tune loss weights or enable GT4 short-valid recall/count-floor knobs
  from the duplicate-bank reference-only evidence.

## 2026-06-26: Add Q20-Dataref Reference-Coverage Duplicate Guard

Decision:

Make duplicate-source-lane evidence a hard formal-mode failure in
`tools/check_q20_dataref_reference_coverage.py`.

Why:

The template builder already detected that the old/current diagnostic input has
42 rows but only 22 unique source lanes. The reference-only coverage checker
previously accumulated all 42 rows and printed a passing OK result, which could
mislabel duplicate-weighted debug evidence as formal promotion evidence.

Implementation:

- Add shared `tools/q20_dataref_common.py` for `raw_file + lane_id` lane keys,
  x/valid signature fallback, JSONL loading, dedup reports, and deduplication.
- Make both the builder and coverage checker use the same dedup logic.
- Coverage default mode now raises when duplicate rows exist.
- `--dedup-evidence` computes formal evidence on deduplicated rows.
- `--allow-duplicate-evidence` computes debug duplicate-weighted metrics, but
  forces `gate.formal_eligible=false` and `gate.passed=false`.
- Add `tools/check_q20_dataref_reference_coverage_duplicate_guard.py` to cover
  default failure, debug-only duplicate evidence, and deduplicated formal
  evidence.

Evidence:

On the current old/current diagnostic CSV inputs:

```text
default mode: RuntimeError with input_rows=42, unique_source_lanes=22, duplicate_rows=20
--allow-duplicate-evidence: evidence_mode=debug_duplicate_weighted, gate.debug_passed=true, gate.passed=false
--dedup-evidence: evidence_mode=formal_deduplicated, lanes=22, gate.passed=true
```

## 2026-06-26: Reject Q20-Dataref v1 Hard-Gate Attempt

Decision:

Do not run official-val for
`gcs_yolo_lane_s_q20_k56_dataref_gt4endpoint_validneg_countce_v1`. The hard
GT4 diagnostic did not pass the stricter Q20-dataref acceptance gate.

Run parameter audit:

```text
run = gcs_yolo_lane_s_q20_k56_dataref_gt4endpoint_validneg_countce_v1
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q20-k56-dataref.yaml
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_sidegeom_gt4endpoint_validneg_countce_v1/weights/best.pt
reset_point_reference = false
sidegeom best.pt point_reference_logits keys = 0
dataref best.pt point_reference_logits keys = 0
fresh dataref model point_reference_logits = non-persistent buffer
```

The recorded `reset_point_reference=false` is not considered the cause of this
failure. `point_reference_logits` is registered as a non-persistent buffer and
is absent from the audited sidegeom/dataref checkpoint `state_dict()` values,
so normal GCS pretrained loading does not overwrite the configured dataref
reference bank in this run.

Hard diagnostic protocol:

```text
hard set = data/tusimple_gt4_hard_val_gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025.txt
records = 19 images
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
match gate = overlap >= 3 h-samples and mean_abs_x_error <= 20px
```

Fixed old-missing recovery diagnostic:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_dataref_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/fixed_old_missing_recovery/summary.json
old missing GT lanes = 22
raw_match_recall = 12/22 = 0.545455
after_point_valid_recall = 3/22 = 0.136364
after_min_points_recall = 2/22 = 0.090909
after_conf_recall = 2/22 = 0.090909
final_decode_recall = 2/22 = 0.090909
new_status = geometry_bad 10, low_point_valid 10, recovered_final_decode 2
```

The fixed-old summary was derived from the dataref
`diagnostic_all_current_missing/raw_queries.csv`; the reconstruction method was
checked by reproducing the Q20 sidegeom fixed-old summary with zero row
mismatches.

Current-missing diagnostic:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_dataref_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/diagnostic_all_current_missing/summary.json
current missing lanes = 20
drop_reason = geometry_bad 10, low_point_valid 10
raw_match_recall = 10/20 = 0.500000
after_point_valid_recall = 1/20 = 0.050000
after_min_points_recall = 0/20 = 0.000000
after_conf_recall = 0/20 = 0.000000
final_decode_recall = 0/20 = 0.000000
```

Why:

The stricter Q20-dataref acceptance gate was:

```text
fixed old-missing:
raw_match_recall >= 14/22
after_point_valid_recall >= 4/22
final_decode_recall > 2/22
geometry_bad <= 7

current-missing:
raw_match_recall >= 0.55
geometry_bad <= 7
after_point_valid_recall > 0
```

This attempt fails fixed-old raw geometry (`12/22 < 14/22`), fixed-old
point-valid survival (`3/22 < 4/22`), fixed-old final decode (`2/22` is not
`> 2/22`), fixed-old geometry (`10 > 7`), current raw geometry
(`0.500000 < 0.55`), and current geometry (`10 > 7`). It only satisfies the
current `after_point_valid > 0` condition.

Integrated conclusion:

- Supported fact: this run does not improve raw geometry over Q20 sidegeom on
  either denominator. Fixed old stays at `12/22`; current stays at `10/20`;
  `geometry_bad` stays at `10`.
- Supported fact: point-valid survival shows a tiny signal
  (`fixed old 2/22 -> 3/22`, current `0/20 -> 1/20`), but the raw-geometry
  gate did not move.
- Decision: reject this attempt before official-val, keep final test closed,
  and do not tune valid-loss weights from this evidence.
- Next action: continue only with a Q20-dataref follow-up that has a clear path
  to improve raw geometry, such as reference-bank construction changes. Do not
  move to valid-loss tuning from this result.

## 2026-06-27: Add Hard Guard for Q20-Dataref Side-Aux

Decision:

`gcs_dataref_side_aux > 0` must hard-require the Q20 fixed-y dataref model
contract before training or loss computation proceeds:

```text
reference_mode = dataref
num_queries = 20
point_mode = fixed_y
pred_reference_x output present
```

Why:

`pred_reference_x` is now emitted for fixed-y heads with point reference logits,
including Q20-sidegeom and the default fixed-y heads. It is useful reference
metadata but is not a sufficient dataref signal. Without a separate guard,
enabling `--gcs-dataref-side-aux` on the Q20-sidegeom YAML could silently train
the side-query auxiliary objective on the wrong reference bank.

Implementation:

- `GCSLaneHead.forward()` emits `reference_mode` and scalar tensor
  `is_dataref_reference` alongside `pred_reference_x`.
- `GCSLoss._dataref_side_aux_loss()` returns zero only when the aux gain is
  disabled. When enabled, it raises unless the output metadata marks dataref,
  Q is exactly 20, and `pred_reference_x` exists.
- `GCSLaneTrainer.get_model()` checks the constructed `GCSLaneHead` before the
  first batch. Q20-sidegeom, Q18, and Q12 models must fail fast with
  instructions to use `gcs-yolo-lane-s-q20-k56-dataref.yaml` or disable
  `--gcs-dataref-side-aux`.

Validation target:

`tools/check_dataref_side_aux_guard.py` must pass for Q20-dataref and must
observe `ValueError` for Q20-sidegeom, Q18, and Q12 when
`gcs_dataref_side_aux=0.25`.

中文记录：

Q20-dataref v1 没有通过 hard diagnostic promotion gate。

证据：

fixed old-missing denominator=22：

- `raw_match_recall = 12/22 = 0.545455`，低于要求的 `14/22`。
- `after_point_valid = 3/22`，低于要求的 `4/22`。
- `final_decode = 2/22`，没有高于之前的 `2/22`。
- `geometry_bad = 10`，高于允许的 `<=7`。

current-missing：

- `raw_match_recall = 10/20 = 0.500000`，低于要求的 `>=0.55`。
- `geometry_bad = 10`，高于允许的 `<=7`。
- `after_point_valid = 1/20`，但 `final_decode = 0/20`。

决定：

- 不提升 Q20-dataref v1。
- 不在这个分支上调 valid loss。
- 不把 `gcs_gt4_short_valid_pos_weight` 改成 `1.25`。
- 不把 `unmatched_valid_neg_weight` 改成 `0.5`。
- 下一步诊断 dataref 失败是因为 reference bank 本身不对，还是因为训练/加载后预测偏离了 reference。

## 2026-06-27: Reject Q20-Dataref Refaux v1 at Hard Gate

Decision:

Do not run official-val or final test for
`gcs_yolo_lane_s_q20_k56_dataref_refaux_v1`. The run has useful refaux
training-log signals, but it fails the required GT4-hard raw-query gate.

Training-log and checkpoint audit:

```text
run = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_dataref_refaux_v1
epochs recorded = 26
train dataref_side_aux_lanes last10 mean = 20.598
train dataref_side_aux_refdist last10 mean = 30.7414
train dataref_side_aux_point first10 -> last10 = 0.0027 -> 0.001507
train dataref_side_aux_valid first10 -> last10 = 0.366196 -> 0.162195
train dataref_side_aux_exist first10 -> last10 = 1.95761 -> 1.04107
train unmatched_valid_prob_mean first10 -> last10 = 0.020455 -> 0.014513
val dataref_side_aux_lanes = 78.8333
val dataref_side_aux_refdist = 32.4967
val unmatched_valid_prob_mean first10 -> last10 = 0.036695 -> 0.029314
loss NaN check = clean
```

`tools/check_saved_dataref_reference.py` passed on `weights/best.pt`:

```text
reference_mode = dataref
num_queries = 20
num_points = 56
point_mode = fixed_y
point_reference_logits_requires_grad = false
max_abs_diff_vs_Q20_DATAREF_X = 8.471310138702393e-05
```

Hard diagnostic protocol:

```text
hard set = data/tusimple_gt4_hard_val_gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025.txt
records = 19 images
decode = conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
match gate = overlap >= 3 h-samples and mean_abs_x_error <= 20px
```

Fixed old-missing recovery:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_dataref_refaux_v1/gt4_hard_raw_query_fixed_gt4pt025/fixed_old_missing_recovery/summary.json
old missing GT lanes = 22
raw_match_recall = 11/22 = 0.500000
after_point_valid_recall = 5/22 = 0.227273
after_min_points_recall = 3/22 = 0.136364
after_conf_recall = 3/22 = 0.136364
final_decode_recall = 3/22 = 0.136364
new_status = geometry_bad 11, low_point_valid 7, low_min_points 1, recovered_final_decode 3
old22 categories = pred_close_valid_bad 7, ref_close_pred_bad 2, ref_bad 9, recovered_final_decode 3, pred_close_later_drop 1
```

Current-missing diagnostic:

```text
artifact = runs/gcs_lane/gcs_yolo_lane_s_q20_k56_dataref_refaux_v1/gt4_hard_raw_query_fixed_gt4pt025/diagnostic_all_current_missing/summary.json
current missing GT lanes = 19
drop_reason = geometry_bad 11, low_point_valid 7, low_min_points 1
raw_match_recall = 8/19 = 0.421053
after_point_valid_recall = 2/19 = 0.105263
after_min_points_recall = 0/19 = 0.000000
after_conf_recall = 0/19 = 0.000000
final_decode_recall = 0/19 = 0.000000
```

Why:

The gate required fixed old-missing `raw_match_recall >= 14/22`,
`after_point_valid >= 5/22`, `final_decode >= 4/22`, and `geometry_bad <= 7`.
This run only meets the fixed-old point-valid threshold. It regresses fixed-old
raw geometry from the previous Q20-dataref v1 `12/22` to `11/22`, improves
fixed-old final decode only from `2/22` to `3/22`, and worsens
`geometry_bad` from `10` to `11`.

The current-missing side is a stronger failure: raw recall drops below the
previous `10/20` baseline to `8/19`, `geometry_bad` is `11`, and
`final_decode` remains `0`.

Integrated conclusion:

- Supported fact: refaux remained active in the training logs and did not raise
  unmatched valid probability.
- Supported fact: the saved checkpoint still uses the intended Q20 dataref
  reference bank.
- Supported fact: the auxiliary signal did not improve hard raw geometry or
  final decode enough to justify official-val.
- Decision: reject `gcs_yolo_lane_s_q20_k56_dataref_refaux_v1` before
  official-val, keep final test closed, and do not tune decode thresholds from
  this checkpoint.
- Smallest safe next action: do not simply increase valid loss. If this line
  continues, use a geometry-focused follow-up that changes reference-bank
  coverage or strengthens the point/refaux pull; the `valid05` branch is not
  justified because raw geometry failed first.

## 2026-07-03: Reject shortside010_gt5off + farspur005 Combination

Decision:

Do not promote
`gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom010_gt5off_farspur005_v1`.
The combination improves GT4 false-fifth behavior, but it reintroduces the
GT5 undercount failure more strongly than either single component.

Protocol note:

The first external 363-image sweep directory named
`...shortsidegeom010_gt5off_farspur005_v1_official_best_val_sweep_valid_before_maxdet`
used `weights/best.pt`, while the paired test directory used
`weights/official_best.pt`. Treat that pair as protocol-mismatched evidence.
A corrected official-val sweep was run on `weights/official_best.pt`:

```text
corrected_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom010_gt5off_farspur005_v1_official_bestpt_val_sweep_valid_before_maxdet/tusimple_official_sweep_summary.json
selected_decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_val_acc = 0.976937
official_val_FP = 0.003489
official_val_FN = 0.008724
official_val_count_acc_4 = 0.939394
official_val_count_acc_5 = 0.878378
official_val_count_confusion includes 3->4=2, 4->5=0, 5->4=9
```

The protocol-correct reporting-only test for that corrected official-val
decode was also run:

```text
test_summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom010_gt5off_farspur005_v1_official_best_test_best_from_correct_val/tusimple_official_summary.json
official_test_acc = 0.963280
official_test_FP = 0.031087
official_test_FN = 0.032590
test_count_acc_4 = 0.632479
test_count_acc_5 = 0.773286
test_count_confusion includes 4->5=63, 5->4=103, 5->5=440
```

Why:

- Against the same-line no-shortside comparator, the combination improves
  reporting-only test GT4 `4->5` from `75` to `63` and `count_acc_4` from
  `0.611111` to `0.632479`.
- The cost is unacceptable GT5 retention loss: `5->4` worsens from `69` to
  `103`, and `count_acc_5` drops from `0.836555` to `0.773286`.
- The damage is concentrated on `0601|GT5`: `5->4` grows from `26` to `62`
  and `5->5` falls from `458` to `421`.
- The corrected official-val sweep has zero rows with `5->4 <= 2` or
  `count_acc_5 >= 0.972973`, so this checkpoint cannot be saved by choosing a
  different row inside the tested validation grid.
- Training logs confirm `gcs_far_spurious_gt5_weight=0.0` kept far-spur off
  GT5, and `gcs_short_side_geom_weight_gt5=0.0` did not remove all GT5-side
  pressure because duplicate-like `gcs_spurious_neg` still selected GT5-group
  candidates late in training.

Integrated conclusion:

The combination is diagnostic-only. Far-spur remains useful for GT4 overcount
pressure, but pairing it with even reduced short-side geometry does not solve
the generalization problem. Do not continue by increasing `short_side_geom` to
`0.125`. The next safe step should target GT5 existence/rank protection or
remove remaining GT5 duplicate-like spurious pressure before adding any more
GT4/short-side pressure.

## 2026-07-05: Fix ranking/shortside count-contract debug surfaces

Decision:

Keep the ignore-first/ranking/shortside count-contract line default-off, but
fix three contract gaps before any new ablation:

- split side-ambiguous unmatched queries into true-or-uncertain ignore versus
  worse side duplicate-like ranking negatives;
- keep visible `2..5` side GT as a separate ultra-short tier with optional
  light score-floor and valid rescue only;
- replace the too-narrow implicit ranking-positive bool with explicit
  `gcs_rank_pos_scope`.

Implementation scope:

- `gcs_rank_side_duplicate_enable=False` preserves the old conservative
  default. When enabled, `side_duplicate_like` can enter ranking negatives only
  if it is close/overlapping, side-ambiguous, not raw-rescue protected, tied to
  a selected true same-side/GT positive, and worse by
  `gcs_rank_side_dup_margin_px`.
- BCE hard-negative pressure remains clear-far only under the ignore-first
  contract. Normal duplicate-like and side duplicate-like queries are ranking
  negatives only.
- `gcs_shortside_ultra_enable=False`,
  `gcs_shortside_ultra_score_floor_gain=0.0`,
  `gcs_shortside_ultra_valid_gain=0.0`, and
  `gcs_shortside_ultra_rank_pos=False` keep ultra-short visible `2..5` lanes
  diagnostic-only by default. Strong ultra-short point geometry remains off
  through `gcs_shortside_ultra_point_gain=0.0`.
- `gcs_rank_pos_scope` supports `shortside_reliable`,
  `shortside_with_ultra`, `gt4gt5_matched`, and `all_matched`; the default is
  `shortside_reliable`.
- Training, validation, and `tools/diagnose_gcs_count_contract.py` now expose
  explicit debug counts for side duplicate negatives, normal duplicate
  negatives, side-ambiguous ignores, ultra-short selection/enabling, and
  ranking no-op reasons.

Validation target:

Use local compile and contract checks only. Formal training/official-val
evidence must run remotely and be selected on official-val.

## 2026-07-05: Harden count-contract positive and ignore semantics

Decision:

Fix five count-contract guard gaps while keeping the ignore-first/ranking and
shortside rescue line default-off:

- ranking positives are Hungarian true-lane positives by default; unmatched
  reliable raw-rescue queries are excluded unless the explicit
  `gcs_rank_include_unmatched_rescue_pos=True` ablation is set and the rescue
  query passes conflict/duplicate-like filtering;
- base `exist_loss` and `point_valid_loss` use the same ignore-first protection
  for unmatched raw-rescue, near-GT corridor, side-ambiguous, and duplicate-like
  rank-only queries whenever the new count-contract path is enabled, while
  clear-far spurious queries remain BCE negatives;
- visible `2..5` same-GT Hungarian shortside rawmatch lanes receive the base
  shortside positive boost/target floor, but unmatched ultra-short rescue stays
  gated by `gcs_shortside_ultra_enable`;
- `gcs_shortside_min_gt_lanes < 4` or `gcs_rank_gt_min_lanes < 4` now fails
  under the new count-contract path unless
  `gcs_allow_gt3_count_contract_ablation=True` is explicitly set;
- shortside image medians now use the standard median, so `[30, 40, 50, 56]`
  gives `45` instead of the lower median `40`.

Implementation scope:

- Added debug counts for unmatched-rescue ranking inclusion/exclusion, base BCE
  ignore source counts, clear-far negatives kept, and ultra-short matched versus
  unmatched rescue handling.
- Added CLI/config defaults for `gcs_rank_include_unmatched_rescue_pos=False`
  and `gcs_allow_gt3_count_contract_ablation=False`.
- Kept decode, matcher assignment, labels, official metrics, and default-off
  training objectives unchanged.

Validation target:

Run `python tools/check_gcs_count_contract_losses.py` and Python compile checks
for changed Python files locally. Formal training/official-val evidence remains
remote-only and official-val selected.

## 2026-07-05: Fix farspur/base-BCE ignore and shortside/ranking semantics

Decision:

Close five high-priority count-contract gaps before any new ablation:

- Farspur ignore-first near-GT corridor and ambiguous side-region masks now
  protect the original base `exist_loss` and `point_valid_loss`, not only the
  farspur auxiliary BCE term.
- Base BCE clear-far preservation now uses a final clear-far mask outside both
  rank and farspur near/side ignore zones and outside raw-rescue protection.
- `gcs_shortside_rawmatch_boost` no longer implicitly enables
  `gcs_shortside_exist_target_floor`; the floor defaults to `0.0` and only an
  explicit positive value changes existence targets.
- `shortside_rescue_pos` is unmatched-rescue-only; same-GT Hungarian rawmatch
  positives are tracked separately through `shortside_hungarian_rawmatch_pos`,
  and `shortside_rawmatch_pos` is the explicit union/debug mask.
- `gcs_rank_pos_scope=gt4gt5_matched` and `all_matched` now mean all Hungarian
  matched true-lane queries in their scopes, regardless of
  `gcs_shortside_ultra_rank_pos`.

Implementation scope:

- Added base-BCE debug counts for rank-near, rank-side, farspur-near,
  farspur-side, duplicate-rank-only ignores, and kept final clear-far
  negatives.
- Kept unmatched raw-rescue excluded from ranking positives by default; it can
  enter only through `gcs_rank_include_unmatched_rescue_pos=True` after existing
  conflict/duplicate filtering.
- Kept decode, matcher assignment, labels, official metrics, and default-off
  training objectives unchanged.

Validation:

- `D:\miniconda3\envs\lsa_yolo\python.exe -m py_compile
  ultralytics\utils\gcs_loss.py tools\check_gcs_count_contract_losses.py
  ultralytics\models\yolo\gcs_lane\train.py
  ultralytics\models\yolo\gcs_lane\val.py tools\train_gcs.py`
- `D:\miniconda3\envs\lsa_yolo\python.exe tools\check_gcs_count_contract_losses.py`
- `D:\miniconda3\envs\lsa_yolo\python.exe tools\check_model.py --cfg
  ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml --imgsz 544 960 --batch 1
  --device cpu`

## 2026-07-05: Isolate base-BCE ignore, target floor, and ranking reduction

Decision:

Fix the ablation-isolation errors in the ignore-first/ranking/shortside
count-contract line before any new training run.

Implementation scope:

- Added explicit default-off base BCE ignore controls:
  `gcs_base_ignore_raw_rescue`, `gcs_base_ignore_rank_near`,
  `gcs_base_ignore_farspur_near`, and
  `gcs_base_ignore_duplicate_like`.
- Enabling `gcs_rank_topk_weight`, `gcs_farspur_ignore_first`,
  `gcs_farspur_weight`, or `gcs_shortside_raw_rescue` no longer implicitly
  changes the original base `exist_loss` or `point_valid_loss`. Rank-derived
  base ignore is also scoped by `gcs_rank_gt_min_lanes`, preventing GT3 samples
  from receiving rank-side base ignore when the ranking loss itself skips them.
- `gcs_farspur_ignore_first=True` with `gcs_farspur_weight=0` is now
  diagnostics/classification only unless a separate `gcs_base_ignore_*`
  experiment explicitly enables base BCE protection.
- `gcs_shortside_exist_target_floor` now has before-floor target diagnostics
  and `shortside_target_floor_applied_count`, so boost-only, floor-only, and
  boost+floor ablations are distinguishable.
- `rank_topk_loss` now defaults to
  `gcs_rank_pair_reduction=global_pair_mean`; the old per-image-equal behavior
  remains available as `image_mean`.
- Split shortside rawmatch/rescue candidate diagnostics into
  `shortside_hungarian_rawmatch_candidate_count`,
  `shortside_unmatched_raw_rescue_candidate_count`, and
  `shortside_rawmatch_candidate_total_count`. The compatibility
  `raw_rescue_candidate_count` now means unmatched raw-rescue candidates.
- Kept `gcs_rank_pos_scope=shortside_reliable` as the conservative default,
  but training logs a warning when `gcs_rank_topk_weight > 0` uses that scope
  so the run is not mislabeled as all-GT4/GT5 matched ranking.

Validation target:

Run local py_compile, `tools/check_gcs_count_contract_losses.py`, and the CPU
model shape check. Formal training/official-val evidence remains remote-only
and must be selected on official-val.
