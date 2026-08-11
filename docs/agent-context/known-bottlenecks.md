# Known Bottlenecks

## 2026-08-11: Lane-instance-set addresses the right class of problem, but has no measured coverage yet

The lane-instance-set candidate is a structural response to the current
base-query bottleneck: hard short/outer GT4 and GT5 lanes need complete K56
instance hypotheses whose geometry, visibility, duplicate identity, topology,
and survivor score are learned as a set. This is closer to the remaining
problem than another scalar query threshold or residual replacement threshold.

However, the current evidence is only contractual. Shape checks and synthetic
decode/cache/loss tests prove that the code path is finite, prediction-only,
survivor-counted, default-off, and separated from the active env30 baseline.
They do not prove that the candidate learns useful lane instances or covers
the hard GT4/GT5 misses.

The unverified bottlenecks are:

```text
training convergence of the lane-instance-set heads
candidate coverage on canonical official-val and clean-val
train0313/train0601 cross-session hard-lane transfer
set-survival calibration against harmful unmatched survivors
duplicate/topology reliability for adjacent real lanes
multi-seed stability
official-val promotion-gate pass/fail
TEST behavior
```

The old rejected paths remain closed. Residual replacement/listwise, dense
ranker/assembler, full-lane proposal, local-segment/candidate, and
query-only survival/valid-residual routes are still negative evidence and must
not be cited as active lane-instance-set gains. The next useful evidence is a
short training smoke followed by non-test candidate coverage and clean-val
official-val gates. TEST remains closed.

## 2026-08-11: Point-valid ceiling is real, but free-form logits are unstable

The matched-query oracle proves that GT-correct visibility can raise
official-val ACC by roughly `+0.0106..+0.0176`, while score protection is a
no-op. This identifies point-valid survival as the largest measured ceiling,
but it does not prove that an unconstrained 56-logit residual is learnable.

The global and anchor-local residual heads both fail. The local head already
uses the strongest available point-local image evidence, and an unmatched
identity trust region reduces but does not remove the regression. Replacing
the fixed-20px visible-only assignment with TuSimple angle-adjusted
`line_accuracy` also fails. At `LR=1e-4`, the main failure is GT4 survival:
canonical `count_acc_4` falls from `0.969697` to `0.954545` and FN rises.

The root mismatch is structural. TuSimple fixed-y lane visibility is one
contiguous start/end interval, but the query head and residual optimize 56
independent binary logits. Small shared-logit changes cross the discrete
`point_valid_thr=0.60`, `min_points=4`, `valid_before_maxdet`, and top-5
boundaries, changing lane count and official matching discontinuously. BCE and
Dice on anchors therefore do not provide a count-safe surrogate for official
set utility.

Do not continue global/local valid-delta heads or loss-weight tuning. The next
smallest credible route is a default-off contiguous visibility-boundary head:
predict start/end, enforce order and interval contiguity, preserve an explicit
base/no-change action, and train only on official-line-accuracy-qualified base
queries. Gate it on canonical, clean, and train0601 before TEST.

## 2026-08-11: Residual retrieval is not official set utility

The Stage-2d..2g sequence identifies a deeper objective mismatch. A proposal
can be a strict full-span APE20 hit while having zero TuSimple official value,
because official line accuracy uses angle-adjusted per-anchor agreement,
invalid-anchor agreement, an `0.85` matched-line threshold, and special GT5
handling that drops the worst lane from accuracy and forgives one FN. The
clean GT5 candidate at `clips/0601/1495485009641442407/20.jpg` is the clearest
case: diagnostic base-miss APE exceeds 20 px, but the baseline prediction set
already scores `Accuracy=0.995536, FP=0, FN=0` on that image.

The proposal pool is nevertheless useful. Exhaustive GT-assisted official
set-update analysis finds positive replacement ceilings on 11/363 canonical,
11/363 clean, and 21/410 train0601 images. Canonical oracle replacement would
move ACC `0.973346 -> 0.973904` while reducing both FP and FN. The current
bottleneck is therefore not only geometry recall; it is prediction-only action
identity and no-op calibration.

Stage-2g trains the existing proposal-victim head on exact positive
TuSimple-official-score deltas, but sparse independent BCE/ranking collapses
to unsafe calibration. After 12 epochs, canonical threshold `0.5` gives
`0.972641 / 0.016437 / 0.011019`; threshold `0.25` gives
`0.972735 / 0.015060 / 0.010331`; thresholds `>=0.7` are no-ops. Clean and
train0601 show the same pattern. More epochs or another scalar threshold are
not justified.

The smallest credible next route is a listwise image-level action selector
with an explicit no-op class, trained on official utility deltas and audited
leave-session-out. It must distinguish which victim is safe, not merely whether
a proposal resembles a missing lane.

This file applies to branch `codex/5-25-3-k56`.

## 2026-08-11: Existing-query geometry adapters cannot recover missing instances

The frozen-env30 query-survival sequence now rejects both harder sampling and
the zero-initialized geometry adapter. Hard sampling plus pairwise ranking in
`query_survival_rank_hardsampling_env30_probe10_seed0_v5` leaves the
train0601 final-hit result unchanged from env30, while `10/19` GT4-short and
`41/183` GT5-short targets have no raw Q12 geometry within 20 px. Reweighting
or reranking cannot select a lane that is absent from the query pool.

The geometry adapter in
`query_survival_geometry_hardsampling_env30_probe10_seed0_v6` can change x and
point-valid outputs, but both clean-val checkpoints regress:

```text
env30 clean-val:
  ACC/FP/FN = 0.966132 / 0.034527 / 0.019743
  count_acc_4/count_acc_5 = 0.958333 / 1.000000

v6 epoch005:
  ACC/FP/FN = 0.965310 / 0.035399 / 0.022727
  count_acc_4/count_acc_5 = 0.937500 / 1.000000

v6 epoch010:
  ACC/FP/FN = 0.965162 / 0.031589 / 0.020432
  count_acc_4/count_acc_5 = 0.941667 / 1.000000
```

The adapter receives only the frozen query token, its existing curve,
point-valid probabilities, and existence score. It can deform an existing
hypothesis but does not introduce new spatial instance evidence. The failure
therefore strengthens the representation diagnosis: the next candidate must
create a missing-lane proposal from image-local evidence, then learn explicit
proposal-to-base identity and replacement utility. Do not continue with larger
query deltas, longer adapter training, more hard-sampling weight, seed repeats,
or TEST.

## 2026-08-10: Explicit pair relations still lack clean GT5 ranking signal

Stage-2b encodes the full K56 residual-to-base and residual-to-residual
differences, valid probabilities, base scores, learned pair tokens, and
attention-weighted contexts. This is stronger than the rejected independent
per-proposal Stage-2 representation, but it still leaves the only clean-val
GT5 short base-miss strict candidate outside top5 (`oracle=1/6`, quality
top5=`0/6`).

The failure is not caused by geometry drift, visibility drift, checkpoint
selection, or incomplete freezing: all 527 Stage-1b state keys are unchanged,
`best.pt` and `last.pt` give identical retrieval counts, and env30 remains
`TP/FP/FN=1317/54/41`. Canonical ranking improves, while ordinary train0601
GT5 quality top3 drops from existence `67/1194` to `62/1194`. The relation
features therefore learn split-specific ranking rather than a stable notion of
lane novelty and replacement value.

Do not add a set selector on these scores. The next representation must obtain
stronger supervised relational evidence, such as explicit assignment to the
nearest competing base lane plus topology/order and candidate-to-GT utility
targets, while remaining diagnostic-only until clean GT5 top5 is nonzero and
ordinary GT5 top3 is preserved.

## 2026-08-10: Per-proposal identity/quality cannot identify clean GT5 novelty

Stage-2 v1, fix2, and fix3 all fail the joint retrieval gate. Adding strict
quality targets, pairwise ranking, stronger identity negatives, diversity
selection, and distances to high-score base queries improved canonical and
ordinary train-side GT5 top3 retrieval, but never moved the only clean-val GT5
strict candidate into top5. Train0601 critical short retrieval also remained
below the frozen existence ranking.

The core remaining issue is relational: a proposal cannot be judged safely
from its own row evidence, visibility, embedding, scalar quality, and summary
distance-to-base features. The next admissible research step is not a set
selector trained on these failed scores. It must first build explicit
proposal-to-each-base and proposal-to-proposal relational tokens, then prove
safe retrieval on canonical, clean GT5, and train0601 while replacement stays
closed.

## 2026-08-10: Visibility is repaired; proposal identity/ranking is now limiting

`residual_proposal_visibility_stage1b_probe10_v1_fix2` closes the Stage-1
point-valid collapse without changing geometry or env30 metrics. Strict
full-span hit20 rose from `1/17` to `8/17` on canonical, `0/13` to `3/13` on
clean-val, and `5/49` to `25/49` on train0601. GT5 rose to `6/12`, `1/6`, and
`21/38`, respectively. These results meet every registered Stage-1b gate.

The remaining gap is candidate identity and ranking rather than visible-span
capacity. Strict any-proposal versus current top1 retrieval is `8 vs 5` on
canonical, `3 vs 1` on clean-val, and `25 vs 17` on train0601. Clean-val GT5
has one valid complete proposal but current top1 remains `0/6`. Do not respond
with more visibility tuning, longer Stage-1b training, replacement, or decode
thresholds. The next experiment should add identity/quality supervision and
measure safe top-k retrieval while keeping all accepted geometry and
visibility heads frozen.

## 2026-08-10: Residual proposal finds geometry but point-valid destroys complete-span recall

The default-off frozen-env30 probe `residual_proposal_env30_probe20_v1_fix1`
completed 20 epochs with TEST and official decode closed. The base query metrics
remained exactly frozen (`val TP/FP/FN=1317/54/41` for every epoch).

The segmentation-first proposal representation learned real new geometry:

```text
canonical short base-miss geometry hit20 = 9/17
  GT4 short base-miss = 2/4
  GT5 short base-miss = 6/12
clean-val short base-miss geometry hit20 = 4/13
  GT4 short base-miss = 3/6
  GT5 short base-miss = 1/6
train0601 short base-miss geometry hit20 = 27/49
  GT4 short base-miss = 4/10
  GT5 short base-miss = 22/38
```

Strict point-valid full-span coverage collapses to `1/17`, `0/13`, and
`5/49`, while the separately supervised interval head retains `6/17`, `2/13`,
and `17/49` on the same canonical, clean-val, and train0601 short base-miss
groups.

Interpretation: the new representation creates useful missing-lane geometry,
but residual visible-span calibration destroys most complete proposals,
especially leakage-free clean-val GT5. Do not add identity or replacement yet.
First make point-valid and interval retain the already-present geometry.

## 2026-08-10: Dense hard-short weighting improves oracle pool slightly but fails selection

The clean converted-val hard-short follow-up
`dense_endpoint_offset_hardshort_w4_probe20_v2_cleanval` completed with TEST
closed. It trained the dense branch, but official query decode still did not
consume dense outputs, so the clean-val promotion gate failed:

```text
env30 clean-val ACC/score/FP/FN = 0.966132 / 0.965047 / 0.034527 / 0.019743
hard-short v2 official_best    = 0.966146 / 0.965061 / 0.034527 / 0.019743
ACC delta = +0.000014, below the +0.000500 gate
count_acc_4/count_acc_5 unchanged = 0.958333 / 1.000000
```

The dense branch converged numerically, but this did not alter the frozen Q12
query path:

```text
train/dense_instance_loss = 1.12297 -> 0.45465
val/dense_instance_loss   = 0.69145 -> 0.50340
val TP/FP/FN              = 1317/54/41 unchanged
```

The corrected clean-val pairing/ranking audit shows the remaining bottleneck:

```text
base-miss lanes:                 dense_full_hit20 = 15/57
GT5 base-miss:                   dense_full_hit20 = 4/11
GT5 vis6-10 base-miss:           dense_full_hit20 = 1/4
GT5 base-miss top5 full-hit:     0/11
GT5 vis6-10 base-miss top5:      0/4
GT4/GT5 short base-miss top5:    0/12
all-lane top5 full-hit:          398/1358
```

Compared with the earlier dense clean-val reference, the candidate pool gained
only small GT5 base-miss oracle capacity (`3/11 -> 4/11`) and finally produced
one GT5 vis6-10 base-miss oracle candidate (`0/4 -> 1/4`), but top-k selection
remained zero for the critical hard lanes and all-lane top5 ranking regressed
(`428/1358 -> 398/1358`). Useful hard-lane candidates remain low-ranked
(`GT5 base-miss best-rank p50=68`, p90=179).

Decision:

```text
Do not continue dense hard-short weighting, top-k replacement, threshold
tuning, or longer training. The next credible route must change the lane
instance/proposal representation so hard GT4/GT5 short lanes are generated and
ranked as complete K56 lanes before decode is reopened.
```

## 2026-08-10: Clean-val dense pool has oracle headroom but no safe learned selector yet

The clean converted-val dense endpoint-offset diagnostic was run with TEST
closed:

```text
run = dense_endpoint_offset_env30_probe20_v1_fix1_offset_gate_cleanval_all_v1
gt = runs/gcs_lane/clean_selection_val_from_converted_split/labels.json
weights = dense_endpoint_offset_env30_probe20_v1_fix1/weights/last.pt
assembly = offset_rowdp, p128, row_topk=10, interval_topk=24, pool=1024
```

It confirms that the dense pool contains some non-test rescue capacity, but
not enough prediction-ranked capacity for promotion:

```text
clean-val base_miss lanes = 57
dense_full_hit20 = 13/57
forced_full_hit20 = 20/57
endpoint_support_both = 47/57

clean-val GT5 base_miss lanes = 11
dense_full_hit20 = 3/11
forced_full_hit20 = 5/11
endpoint_near_both = 0/11

clean-val GT5 vis_6_10 base_miss lanes = 4
dense_full_hit20 = 0/4
forced_full_hit20 = 1/4
endpoint_near_both = 0/4
```

The replacement/set oracle audit shows why add-only or top-k promotion is not
sufficient:

```text
run = dense_endpoint_offset_env30_probe20_v1_fix1_replacement_audit_cleanval_v1
base_nohit = 57
replacement_oracle = 25/57
current_top5_replacement_oracle = 2/57
strict_add_oracle_rescue = 18/57

GT5 base_nohit = 11
replacement_oracle = 4/11
current_top5_replacement_oracle = 0/11
```

An offline non-test ranker trained on train0601 GT5 hard rows plus train0313
GT4 hard rows did not solve the critical clean-val groups:

```text
run = dense_ranker_train0601_0313_to_cleanval_local_mlp_v1
train/eval overlap handling = --drop-overlap
feature_set = local

all lanes top5_full_hit20: current 428/1358 -> supervised 405/1358
base_nohit top5_full_hit20: current 2/57 -> supervised 7/57
GT5 base_nohit top5_full_hit20: current 0/11 -> supervised 0/11
GT5 vis_6_10 base_nohit top5_full_hit20: current 0/4 -> supervised 0/4
```

Interpretation:

Dense evidence has useful oracle capacity on clean-val, but the current
prediction-only candidate representation and local-feature ranker still cannot
rank the GT5 hard lanes that are most aligned with the raw20 TEST gap. A
ranker-only implementation would improve some mixed base-miss lanes while
regressing ordinary lane ranking and leaving GT5 base-miss unresolved.

Decision:

```text
Do not implement a dense ranker or dense replacement decoder from the current
candidate rows.
Do not continue dense endpoint/row-DP/maxdet/top-k threshold work.
The next candidate must change the proposal representation or training target
so GT4/GT5 short base-miss lanes become high-quality complete K56 proposals
before ranking/decode is reopened.
```

## 2026-08-10: Canonical official-val is not sufficient for promotion

Future candidates must not be promoted from canonical official-val alone. The
clean converted-val GT is a leakage-free 363-image validation surface:

```text
runs/gcs_lane/clean_selection_val_from_converted_split/labels.json
converted_train_overlap = 0
official_val_overlap = 36
official_val_train_overlap = 327
gt_contract = noncanonical
```

On this trusted clean-val surface, the previously attractive canonical gains do
not hold:

```text
env30 clean-val sweep:
  ACC/score/FP/FN = 0.966132 / 0.965047 / 0.034527 / 0.019743
  count_acc_4/count_acc_5 = 0.958333 / 1.000000

dense_endpoint_offset_env30_probe20_v1_fix1 clean-val frozen decode:
  ACC/score/FP/FN = 0.965437 / 0.964361 / 0.034068 / 0.019743
  count_acc_4/count_acc_5 = 0.937500 / 1.000000

query_env30_lineiou_w05_clean40_v1 clean-val sweep:
  ACC/score/FP/FN = 0.965375 / 0.964369 / 0.029614 / 0.020661
  count_acc_4/count_acc_5 = 0.983333 / 0.800000
```

The active bottleneck is therefore not "find a better canonical-val row"; it is
leakage-free GT4/GT5 short-lane survival and complete-lane proposal identity
under clean converted-val plus train-side hard gates.

## 2026-08-10: Dense candidate reranking has no canonical GT5 headroom

The full canonical official-val replacement audit closes the current dense
endpoint/candidate reranking route:

```text
run = dense_canonical_allval_replacement_audit_v1
lanes/images = 1303 / 363
base_nohit = 33
replacement_oracle = 6
current_top5_replacement_oracle = 0
GT4 base_nohit = 12, replacement_oracle = 2
GT5 base_nohit = 15, replacement_oracle = 0
```

Candidate ordering confirms there is no safe scalar score:

```text
current_score top5: all = 370/1303, base_nohit = 0/33
endpoint_margin_desc top5: all = 282/1303, base_nohit = 1/33
far_from_base_desc top5: all = 85/1303, base_nohit = 1/33
short_visible_asc top5: all = 23/1303, base_nohit = 1/33
```

A same-checkpoint, same-val-subset row-DP widening diagnostic also failed to
produce GT5 base-nohit candidates:

```text
run = dense_gt5_base_nohit_rowdp_wide_beam8_p4096_v1
GT5 base_nohit = 15
row_topk/interval_topk/state_beam/pool = 10 / 24 / 8 / 4096
dense_full_hit20 = 0/15
replacement_oracle = 0/15
forced_GT_endpoint_full_hit20 = 6/15
endpoint_support_both = 12/15
endpoint_near_both = 0/15
```

A new train-side GT4 hard diagnostic, using only `label_data_0313` images where
env30 already missed under the fixed `20px` gate, shows the same structural
shape for the GT4 failure class:

```text
run = dense_endpoint_peak_train0313_gt4_missing_candidate_diag_v1
audit = dense_endpoint_peak_train0313_gt4_missing_replacement_audit_v1
images = 91
GT4 base_nohit = 95
dense_full_hit20 = 10/95
forced_GT_endpoint_full_hit20 = 52/95
endpoint_support_both = 77/95
endpoint_near_both = 21/95
replacement_oracle = 11/95
current_top5_replacement_oracle = 1/95
```

Interpretation:

The main blocker is not only ranking. On canonical all-val, the current dense
candidate generator has `0/15` full replacement oracle on GT5 base-miss lanes.
No selector, MLP ranker, threshold, NMS, add-only decode, or wider row-DP pool
can recover lanes that are absent from the candidate pool. The forced-endpoint
oracle indicates some dense image evidence exists, but the prediction-only
endpoint/identity carrier fails to localize and associate it. Future work must
first create new GT4/GT5 short-lane proposals while preserving env30 solved
lanes.

## 2026-08-10: GT5-short rescue requires set-level replacement, not add-only

Dense/proposal diagnostics now show that GT5-short base-miss rescue is blocked
by set selection, not by the mere absence of candidates.

Non-TEST train-side GT5-short excluding canonical official-val:

```text
run = dense_train_gt5short_excl_canonical_add_set_oracle_v1
base_nohit_lanes = 33
oracle_candidate_matching = 19
set_oracle_rescue = 19
strict_add_oracle_rescue = 0
current_topk_base_nohit = 0
```

Canonical official-val GT5 base-nohit and train0601 hard diagnostics agree:

```text
canonical base_nohit = 15, set_oracle_rescue = 7, strict_add_oracle_rescue = 0
train0601 base_nohit = 16, set_oracle_rescue = 12, current_topk_base_nohit = 0
```

The larger train-side scalar supervised ranker diagnostic does not provide a
safe ranking solution:

```text
run = dense_ranker_train_gt5short_scalar_to_canonical_gt5nohit_v1
canonical base_nohit top5 full-hit: current 0/15, supervised 2/15
canonical all-GT5 top5 full-hit: current 19/70, supervised 7/70
```

Interpretation:

The next bottleneck is prediction-only set-level replacement/ranking. The
candidate pool can sometimes contain the missing lane, but current scores rank
those candidates far below safe top-k, add-only cannot help when the base
decode already emits the maximum lane count, and a simple supervised scalar
ranker trades hard-case rescue for ordinary GT5 regression. Do not continue
with dense threshold, add-only decode, or simple top-k candidate promotion.

## 2026-08-10: LineIoU-only gains are seed-unstable and damage GT5 survival

The repeat run `query_env30_lineiou_w05_clean100_retention_seed1_v2` was
stopped after epoch 40 with TEST closed. It did not reproduce the seed0
clean100 official-val improvement:

```text
env30 official-val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
seed1 epoch040 official_best ACC/FP/FN = 0.971654 / 0.012672 / 0.008724
seed1 count_acc_4/count_acc_5 = 0.909091 / 0.864865
```

The raw-Q12 diagnostic shows the mixed mechanism:

```text
GT4-short raw20: env30 1/8 -> seed1 3/8
GT4-short final missing: env30 7/8 -> seed1 5/8
GT5-short raw20: env30 40/53 -> seed1 38/53
GT5-short final missing: env30 13/53 -> seed1 15/53
GT5-normal raw20: env30 316/317 -> seed1 311/317
GT5-normal final missing: env30 2/317 -> seed1 7/317
```

Interpretation:

LineIoU-only supervision can move short-lane geometry, but it is not an
env30-preserving fix. The main bottleneck is now more precise: any useful next
route must rescue GT4/GT5 short raw-miss lanes without degrading GT5-normal
geometry, GT5-short survival, or GT4/GT5 lane count. Repeating the same
LineIoU-only recipe or extending it to 100/220 epochs is not supported by the
seed1 evidence.

## 2026-08-10: Ordered-slot count succeeds but geometry/identity collapses

`ordered_slot_env30_probe40_v1` is rejected as a promotion route. It completed
40 epochs and reached high diagnostic official-val count accuracy, but its
lane geometry/visibility quality remained far below env30:

```text
env30 official-val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
ordered-slot epoch040 ACC/FP/FN = 0.889913 / 0.262810 / 0.240358
ordered-slot epoch040 count_acc/count_acc_4/count_acc_5 =
  0.983471 / 0.984848 / 0.986486
strict_order_valid = false, ordered_slot_order_violations = 13
```

The diagnostic official-val run with order checking downgraded to `warn`
showed the main failure is not lane count. Count-correct images still averaged
`ACC=0.893006`, `FP=0.256256`, and `FN=0.234127`, and paired comparison with
env30 regressed `334/363` images while improving only `18/363`.

Interpretation:

The current ordered-slot path resets too much of the mature env30 query head:
only `404/519` tensors transfer from env30, and the slot geometry/interval/
visibility/identity heads must learn a new representation from scratch. The
count head can learn quickly, but the lane geometry and slot identity do not
approach env30 within the probe. Do not continue this route by adding epochs,
weakening strict order checks, runtime sorting, or running TEST.

Current target bottleneck remains the env30 TEST gap:

```text
valid raw20 TEST env30 ACC/FP/FN = 0.966784 / 0.028732 / 0.023544
count_acc_4/count_acc_5 = 0.598291 / 0.891037
test dates = 0530, 0531, 0601
train dates = 0313-1, 0313-2, 0531, 0601
GT4 short lanes: 116, raw has20 = 0.396552
GT5 short lanes: 478, raw has20 = 0.642259
GT4 official count confusion: 4->3 = 98, 4->5 = 90, 4->4 = 280
GT5 official count confusion: 5->3 = 19, 5->4 = 43, 5->5 = 507
```

The existing TEST diagnostic is for post-hoc root-cause analysis only, not
threshold or checkpoint selection. The next credible non-test route must
preserve env30 behavior on solved lanes and apply any geometry/proposal change
only to train-side hard GT4/GT5 short base-miss cases, with leave-date/session-
out and frozen hard-set gates before any final TEST.

## 2026-08-09: LineIoU raw geometry gain is blocked by GT4 count survival

The count-safe clean100 seed0 run completed but produced no promotable
official-val candidate:

```text
run = query_env30_lineiou_w05_countsafe_select_clean100_seed0_v1
eligible rows = 0 / 17280
official_best artifacts = missing by design
TEST = closed
```

The raw-Q12 diagnostic confirms that LineIoU improves geometry:

```text
env30 overall has20 = 0.976209
epoch100 overall has20 = 0.983883
env30 GT4-short has20 = 0.125000
epoch100 GT4-short has20 = 0.500000
env30 GT5-short has20 = 0.754717
epoch100 GT5-short has20 = 0.849057
```

The same run fails promotion because high-ACC rows repeatedly break GT4 lane
count survival:

```text
epoch080 natural:
  ACC/FP/FN = 0.974099 / 0.013223 / 0.009642
  count_acc_4 = 0.954545
  failure = GT4 4->5 overcount

epoch100 natural:
  ACC/FP/FN = 0.974249 / 0.013085 / 0.010331
  count_acc_4 = 0.924242
  failure = GT4 4->3 undercount and GT4 4->5 overcount
```

This is not a simple decode-threshold fix: every official-val sweep row failed
the pre-registered count-safe gates. The next allowed diagnostic is the
default-off `gcs_line_iou_geometry_only` probe, which freezes existence,
point-valid, decoder, backbone, auxiliary, dense, and query-count parameters
and trains only `point_mlp` plus `point_refine_mlp`. It must pass a 4-seed
count-safe 40-epoch official-val gate before any 100-epoch run, and TEST stays
closed.

## 2026-08-08: Dense replacement is blocked by candidate selection, not FP removal

`tools/analyze_gcs_dense_replacement_audit.py` was added as a diagnostic-only
audit for dense candidate replacement. It reads non-test candidate rows and,
when available, raw-Q12 query diagnostics to reconstruct whether a base FP can
be removed while a dense candidate rescues a base-miss GT lane. It does not
modify training, inference, official decode, or TEST status.

New evidence on canonical GT5 base-nohit hard cases:

```text
strict base-nohit lanes = 15
oracle full-hit candidate = 7/15
oracle-removable base FP = 14/15
replacement oracle = 7/15
current top1/top5/top128 full-hit = 0/15 / 0/15 / 0/15
current top1024 full-hit = 7/15
lowest_exist / lowest_valid_count / lowest_valid_len removal = 7/15 when an oracle candidate exists
```

Candidate-only train0601 hard evidence is consistent:

```text
strict base-nohit lanes = 16
oracle full-hit candidate = 12/16
current top1/top5/top128 full-hit = 0/16 / 0/16 / 0/16
current top1024 full-hit = 12/16
```

Simple prediction-only candidate heuristics expose the tradeoff rather than a
solution:

```text
canonical far_from_base_desc top5:
  GT5 base-nohit = 5/15
  all GT5        = 6/70

canonical short_visible_asc / short_y_span_asc top5:
  GT5 base-nohit = 4/15
  all GT5        = 5/70

train0601 hard far_from_base_desc top5 = 4/16
train0601 hard short_visible_asc / short_y_span_asc top5 = 5/16
```

Interpretation:

Removing the wrong base lane is not the main bottleneck for the canonical
base-miss GT5 cases. The limiting problem is selecting short, far-from-base,
lower-score dense candidates without damaging ordinary GT5 lanes. Do not
implement a small replacement decoder, tune dense thresholds, or run TEST from
this trace. A future dense/full-lane route must first solve prediction-only
proposal ranking under a pre-registered official-val gate.

## 2026-08-08: Dense local features do not solve safe candidate selection

The candidate-local ranker diagnostic adds prediction-only local path,
endpoint, embedding, span, slope, and nearest-base features to dense trace
candidate rows. It does not change training, inference, official decode, or
TEST status.

New non-test evidence:

```text
canonical GT5 base-nohit localfeat v2:
  current top1/top5 full-hit = 0/15 and 0/15
  any full-hit candidate = 7/15

local MLP train0601->canonical:
  GT5 base-nohit top1/top5 full-hit = 1/15 and 1/15
  all-GT5 top5 full-hit = 13/70

local linear train0601->canonical:
  GT5 base-nohit top1/top5 full-hit = 1/15 and 1/15
  all-GT5 top5 full-hit = 10/70

current all-GT5 top5 full-hit baseline on the same rows = 19/70
pre-registered base-nohit gate = top5 full-hit >= 4/15
```

The feature analysis is internally consistent but unfavorable for promotion:

```text
canonical base-nohit top wrong candidates:
  median visible anchors = 45
  median nearest-base distance = 3.59 px
  median score = 0.9646

canonical base-nohit first full-hit candidates:
  median visible anchors = 11
  median nearest-base distance = 34.13 px
  median score = 0.8207
```

Interpretation:

Useful base-miss candidates exist, but they look like short, lower-score,
far-from-base alternatives. A global reranker that favors them damages the
dominant ordinary GT5 lanes. This is a structural selection problem, not just
a missing local-feature problem. Do not continue by tuning ranker weights,
thresholds, NMS, `max_det`, or `min_points` on this dense trace path.

Recommended direction:

Stop dense trace/ranker promotion. A credible next route needs set-level
lane-instance prediction or another structured representation that can add
short outer lanes while preserving ordinary correct lanes under official-val
selection.

## 2026-08-07: Dense endpoint-peak improves pool capacity but not promotion gates

`dense_instance_endpoint_peak_probe20_v1` completed `20/20` epochs with
`gcs_dense_freeze_base=true`, `gcs_dense_endpoint_peak_weight=0.05`,
canonical official-val selection, and `RUN_TESTS=0`.

The endpoint/centerline/instance losses decreased, but official-val is still
the frozen Q12 query decode path:

```text
official_best epoch = 10
ACC/score/FP/FN = 0.973365 / 0.972857 / 0.015748 / 0.009642
count_acc_4/count_acc_5 = 0.969697 / 0.986486
decode = conf 0.001, point_valid_thr 0.6, nms 0, max_det 5, min_points 4
```

Dense tensors are not consumed by official decode, so this is not an official
ACC gain. The meaningful diagnostic evidence is mixed:

```text
canonical GT5 base no-hit p128/e1:
  endpoint-mass dense_full_hit20 = 4/15
  endpoint-peak dense_full_hit20 = 7/15
  forced_full_hit20 = 8/15
  endpoint_support_both = 12/15
  endpoint_near_both = 0/15
  current top1/top5 full-hit = 0/15 and 0/15

train0601 hard GT5 base no-hit p128/e1:
  endpoint-mass dense_full_hit20 = 0/16
  endpoint-peak dense_full_hit20 = 11/16
  forced_full_hit20 = 6/16
  endpoint_support_both = 15/16
  endpoint_near_both = 1/16
  current top1/top5 full-hit = 0/16 and 0/16
```

On canonical GT5 base no-hit lanes with any full-hit candidate, useful
candidates are still ranked far below long wrong/neighbor candidates:

```text
median first-full rank = 969
median first-full visible anchors = 11
median top-ranked wrong visible anchors = 45
median first-full nearest-base distance = 34.13 px
median top-ranked wrong nearest-base distance = 3.59 px
```

The larger train-side scalar-feature ranker diagnostic also failed promotion:
the best MLP fit improved train hard cases but reached only `3/15` canonical
GT5 base no-hit top5 full-hit and reduced ordinary all-GT5 top5 full-hit from
`13/70` to `12/70`.

Interpretation:

Endpoint-peak supervision creates more oracle candidates, but current
prediction-only endpoint peak localization, same-lane association, and ranking
are still not sufficient. Do not solve this by adding epochs, tuning official
decode thresholds, increasing `max_det`, or directly enabling dense decode.
The next dense route needs richer candidate-local representation or a stronger
same-lane association/fitting objective before a learned ranker or official
decode can be credible.

## 2026-08-07: Supervised dense ranker features are not enough yet

The existing default-off `gcs_dense_candidate` implementation is not a true
candidate-level ranker. It adds two dense pixel maps,
`pred_dense_candidate_quality_logits` and
`pred_dense_candidate_replace_logits`, and `GCSLoss._dense_candidate_targets`
rasterizes quality/replace targets only at GT bottom endpoints. It does not
score the traced K56 dense candidates that failed the p128/e1 hard gate.

A diagnostic-only supervised ranker tool was added:

```text
tools/analyze_gcs_dense_candidate_supervised_ranker.py
```

The tool reads non-test `dense_candidate_ranking_rows.csv`, trains a small
offline classifier on prediction-only candidate factors, and evaluates rank
metrics using GT labels only after inference. It refuses TEST rows and blocks
train/eval raw-file overlap unless explicitly told to drop overlapping train
rows.

Protocol findings:

```text
canonical p128/e1 hard-list official-parity recheck:
  base_valid_before_maxdet=false
  GT5 base no-hit top1/top5 full-hit = 0/15 and 0/15
  first full-hit rank mean = 801.0
  same as the previous base_valid_before_maxdet=true hard-list result

larger train-side GT5-short bank:
  run = dense_ranking_candidates_train_gt5short_excl_canonical_p128e1_officialdecode_v2
  processed_images = 106/106
  lane rows = 530
  candidate rows = 542,720
  TEST = closed

train-side GT5 base no-hit evidence:
  dense_full_hit20 = 16/33
  forced_full_hit20 = 18/33
  endpoint_support_both = 28/33
  correct_channel_both = 30/33
  endpoint_near_both = 3/33
```

Supervised ranker results:

```text
linear, target=candidate_full_hit20:
  eval GT5 base no-hit top5 full-hit = 1/15
  eval all GT5 top5 full-hit = 7/70, below current 13/70

linear, target=candidate_hit20:
  eval GT5 base no-hit top5 full-hit = 1/15
  eval all GT5 top5 full-hit = 6/70, below current 13/70

MLP hidden=32, target=candidate_full_hit20:
  train GT5 base no-hit top5 full-hit = 14/33
  eval GT5 base no-hit top5 full-hit = 3/15
  eval all GT5 top5 full-hit = 12/70, below current 13/70
```

Interpretation:

The failure is not just that the hand-written score formula is bad. A small
supervised model can partially fit the larger train-side hard subset, but the
best canonical official-val hard result is still only `3/15`, below the
pre-registered `>=4/15` gate, and it still hurts ordinary GT5 ranking
(`13/70 -> 12/70`). The available scalar candidate factors do not yet provide
a stable prediction-only ranking signal for the exact outer short-lane misses.

Decision:

Do not implement or train a model-level dense candidate rank head from these
features yet, do not enable dense decode, and do not run TEST. The next useful
dense action is not another decoder patch or scalar-ranker sweep. The
completed larger bank strengthens the diagnosis that the blocker is richer
candidate-local representation: endpoint peak localization, same-lane
association, and complete K56 fitting must be improved before a learned rank
or replace gate can be a credible official-val candidate.

## 2026-08-07: Dense candidate ranking needs supervised novelty-aware calibration

The dense endpoint mass run is complete but remains diagnostic-only. Its loss
curves show that dense evidence training works, but the evidence is not yet a
reliable complete-lane proposal system:

```text
run = dense_instance_endpoint_mass_probe20_v3_canonical
epochs = 20/20
TEST = closed
official_best = epoch 10
official-val ACC/FP/FN = 0.973365 / 0.015748 / 0.009642
env30 reference ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
```

The small official-val delta is not a dense decode gain because official query
decode does not consume dense tensors. The meaningful diagnostic evidence is:

```text
default trace, canonical GT5 base no-hit:
  dense_full_hit20 = 0/15
  forced_full_hit20 = 5/15
  endpoint_support_both = 13/15
  endpoint_near_both = 0/15

p128/e1 trace, canonical GT5 base no-hit:
  dense_full_hit20 = 4/15
  forced_full_hit20 = 7/15
  current top5_full_hit20 = 0/15
```

The new `candidate_min_base_dist_px` diagnostic explains part of the ranking
failure. On GT5 base no-hit lanes, current top-ranked wrong candidates are
usually near existing Q12/base lanes, while full-hit candidates are farther
from every base lane:

```text
current top median nearest-base distance ~= 5.02 px
first full-hit median nearest-base distance ~= 34.80 px
```

Adding bounded novelty plus short-visible preference can rescue the hard
subset offline:

```text
short_visible_x_novelty:
  GT5 base no-hit top1/top5 full-hit20 = 4/15
  GT5 visible 6..10 base no-hit = 3/10
```

But the same unsupervised rerank damages ordinary GT5 ranking:

```text
current all top5 full-hit20 = 13/70
short_visible_x_novelty all top5 full-hit20 = 5/70
```

Therefore base-distance novelty is a useful feature, not a safe decoder rule.
The next credible dense route is a supervised/calibrated candidate ranker and
replace gate with base-preserve constraints. Do not run TEST, do not extend the
same dense training, and do not promote a `p128/e1` decoder directly.

## 2026-08-03 Final Env30 Conclusion

The active implementation is restored to `86c8fb31c`, the env30 baseline.
All later experiments are rejected as active paths, including every candidate,
local-segment, full-lane, dense-instance, selector, decode, and diagnostic
follow-up through the pre-rollback HEAD `70d9ef0c`.

All lower post-env30 bottleneck sections are historical evidence only. Their
next-action, ready, or reopen wording is cancelled and must not be used to
schedule training, official-val sweeps, or TEST. A future route requires a
new explicit user decision starting from the env30 payload.

## 2026-08-01 v10 Formal Segment Decode Check

The v10 matched run has useful GT-conditioned hard coverage, but the first
prediction-only official decode check does not convert it into a formal gain.
The same `segment_best.pt` checkpoint was evaluated with and without the new
default-off `segment_decode` adapter.

At the env30 reference point
`conf=0.001, point_valid_thr=0.6, nms_dist_px=0, max_det=5, min_points=4`:

```text
base:    ACC/FP/FN = 0.973346 / 0.015748 / 0.009642
segment: ACC/FP/FN = 0.973117 / 0.015886 / 0.010560
```

The segment threshold sensitivity is diagnostic, not a promotion sweep:

```text
threshold 0.6: ACC/FP/FN = 0.973276 / 0.015197 / 0.009642
threshold 0.7: ACC/FP/FN = 0.973276 / 0.015197 / 0.009642
threshold 0.8: ACC/FP/FN = 0.973346 / 0.015197 / 0.009642
threshold 0.9: ACC/FP/FN = 0.973346 / 0.015748 / 0.009642
```

Interpretation: thresholds `0.5..0.7` allow harmful replacements,
`0.8` removes the ACC loss but worsens count retention, and `0.9` collapses
back to the base path. This separates two facts that must not be conflated:
the raw segment oracle has capacity, while the learned score does not yet
provide a reliable prediction-only replacement decision across ordinary
queries. Do not run TEST, add epochs, or keep tuning this decode gate. The
next useful change is selector calibration/negative applicability training
with an explicit no-replace outcome, followed by a fresh hard rank audit.

## 2026-08-01 v10 selector evidence

The v10 run `query_local_segment_env30_frozen_probe20_v10_b4w0s1` did not
carry `gcs_short_segment_matched_assignment=true` into `args.yaml`, so it
cannot validate the intended Hungarian matched query-to-GT target fix.

The run still isolates the remaining selector bottleneck:

```text
last.pt official-val short GT5 base/raw/selected = 40/52/43 of 53
last.pt train0601    short GT5 base/raw/selected = 142/179/151 of 183
last.pt official-val short GT4 base/raw/selected = 1/4/4 of 8
last.pt train0601    short GT4 base/raw/selected = 9/19/16 of 19
```

Base-hit loss is zero for the last checkpoint, so base preservation is no
longer the main failure in this run. Oracle rank-1 remains only `0/53` on
official-val and `8/179` on train0601 short GT5, and most raw oracle
headroom is still missed. The next valid experiment must first prove that
matched assignment is active, then measure whether query-local ranking
improves. TEST and formal short-segment decode remain closed.

## 2026-08-01 v10 matched-assignment result

The valid run
`query_local_segment_env30_frozen_probe20_v10_matched_b4w0s1` verified
`gcs_short_segment_matched_assignment=true` in `args.yaml`, froze the env30
base path, completed `20/20` epochs, and kept both candidate decode and TEST
closed.

The hard selected-gated result from the diagnostic-selected `last.pt` was:

```text
official-val short GT5: base/raw/selected = 40/52/47 of 53
train0601    short GT5: base/raw/selected = 142/179/156 of 183
official-val short GT4: base/raw/selected = 1/4/3 of 8
train0601    short GT4: base/raw/selected = 9/18/15 of 19
```

Compared with the invalid unmatched v10 run (`43/53`, `151/183`,
`4/8`, `16/19` selected-gated), matched assignment adds useful GT5 coverage
and preserves base-hit lanes almost completely. It also exceeds the frozen
env30 base on the main GT5 groups by `+7` and `+14` net selected hits.

The remaining failure is ranking calibration, not raw proposal geometry:
the exact oracle score is top-1 in only `2/53` official-val and `7/183`
train0601 short GT5 cases, with top-5 counts `9/53` and `30/183`.
The official-best checkpoint is still epoch 5 by base-only official ACC and
does not carry the hard-gate gain (`41/53` and `142/183` selected-gated).
Thus `segment_best.pt` is a hard-diagnostic checkpoint, not a promoted
official checkpoint.

Decision: the matched-assignment selector passes the diagnostic gate and is
ready for one official-val-only candidate-decode sweep using
`segment_best.pt`. Do not add epochs, tune TEST, or promote the selector
before that sweep.

## 2026-07-28 Q12/env30 Gated Candidate v2 Reopen

The user reopened candidate generation with a default-off v2 implementation.
The bottleneck remains the same: short GT4/GT5 lanes often have hypotheses in
the 20-40px band but not inside the official 20px gate.

The v2 implementation addresses the rejected probe's structural mismatches:

1. Candidate decode is prediction-only gated to short visible-anchor counts
   instead of applying candidate selection to every Q12 query.
2. Candidate scoring pools image features with predicted visibility weights
   instead of averaging all K=56 anchors.
3. Candidate selector scores do not alter the base query existence score.
4. Count-aware top-k and candidate decode are mutually exclusive.

This is not enough to claim accuracy improvement. The next evidence must come
from raw candidate coverage and official-val-only sweeps. TEST remains closed
until a candidate is selected by official-val.

### Hard official-GT candidate coverage diagnostic

The hard-denominator diagnostic
`tools/diagnose_gcs_short_candidate_hard_coverage.py` was added to measure the
old official-GT visible<=10 short-lane bottleneck directly from TuSimple
json-lines records and original image shapes. It is diagnostic only and does
not change training, decode, loss, or official metrics.

For `query_gated_candidate_env30_frozen_probe20_v2/weights/last.pt` and
`weights/official_best.pt`, the diagnostic reproduces the old denominators and
shows that selected candidate decode has not solved the bottleneck:

```text
official-val short GT5 total = 53
base/raw/selected_gated hit20 = 40/43/40
raw oracle gain20 = +3, selected_gated gain20 = +0

train0601 short GT5 total = 183
base/raw/selected_gated hit20 = 142/148/142
raw oracle gain20 = +6, selected_gated gain20 = +0

train0601 short GT4 total = 19
base/raw/selected_gated hit20 = 9/15/9
raw oracle gain20 = +6, selected_gated gain20 = +0
```

Interpretation: the fixed `[0, +/-20, +/-40, +/-60]` candidate pool has limited
oracle headroom and the learned/gated selector turns none of that headroom into
actual gated 20px hits on either checkpoint. Do not continue this run to TEST
or longer training unchanged. The next change should first improve candidate
generation and geometry-quality selection on official-val/train-side gates.

The follow-up implementation is `gated-candidate-dense13-v3`, which adds
intermediate `+/-10`, `+/-30`, and `+/-50` px offsets while preserving the same
frozen env30 base path. It must be judged by the same hard official-GT
denominators before candidate decode or TEST can be considered.

Pre-training dense13 raw-oracle diagnostics with env30 `official_best.pt`
showed that densifying the fixed offset grid alone is not enough:

```text
official-val short GT5: base/raw_candidate = 40/43
train0601 short GT5: base/raw_candidate = 142/148
train0601 short GT4: base/raw_candidate = 9/15
```

The mean/p90 APE improved, but hit20 did not improve over the 7-offset pool.
Therefore dense13 should not proceed to selector training under the hard gate.
The remaining bottleneck is not offset step size inside `+/-60px`; it is the
coarse carrier shape/position itself or the lack of an adaptive residual/new
proposal.

The 2026-07-29 affine-oracle follow-up extends the same hard diagnostic with
`x' = x + offset + slope * centered_y` candidates from the frozen env30 base
`pred_points`:

```text
offsets = [0, -20, +20, -40, +40, -60, +60]
slopes  = [0, -20, +20, -40, +40]
official-val short GT5: base/raw_affine = 40/43
train0601 short GT5: base/raw_affine = 142/148
train0601 short GT4: base/raw_affine = 9/16
```

Interpretation: adding a simple affine shape residual helps the train0601 GT4
hard subset by one lane over dense13, but it does not move either GT5 gate.
Therefore the current Q12 carrier bank is still the limiting factor for short
GT5. Do not train an affine selector or run TEST from this result. The next
candidate should add proposal/query capacity or otherwise generate genuinely
new short-lane carriers.

The same diagnostic now also supports stronger proposal-capacity oracles:
static template proposals and predicted-mask connected-component proposals.
Both were run on env30 `official_best.pt` and failed the short-GT5 gate:

```text
static96 official-val short GT5: base/raw = 40/41
static96 train0601 short GT5: base/raw = 142/143
static96 train0601 short GT4: base/raw = 9/9

mask official-val short GT5: base/raw = 40/40
mask train0601 short GT5: base/raw = 142/143
mask train0601 short GT4: base/raw = 9/12
```

The broader `static1875` official-val-only check still reached only `41/53`.
This rules out two easy proposal sources: dense image-independent static
templates and raw connected components from the existing auxiliary mask. The
remaining proposal-capacity direction must be image-conditioned and trained, or
data-mined from missed short-GT5 cases, before selector training is justified.

An additional local short-segment shape oracle explains why these routes fail.
It enumerates proposals that are valid only over a contiguous `3..10`
h-sample window and fit local top/bottom x inside that short window. With a
coarse 20px x grid, it reaches:

```text
official-val short GT5: 40/53 -> 52/53
train0601 short GT5: 142/183 -> 179/183
official-val short GT4: 1/8 -> 4/8
train0601 short GT4: 9/19 -> 19/19
```

The remaining GT5 misses are the visible `<3` cases that cannot satisfy the
diagnostic `match_min_overlap=3` gate. This means the missing capacity is not
another global lane template, lateral offset, or aux-mask component. It is a
short-window local segment proposal: explicit interval/start-end survival plus
local segment geometry for outer bottom-edge lanes whose x changes by hundreds
of pixels over only about `50..60px` of visible y.

The current implementation vehicle for this conclusion is the default-off v4
local short-segment proposal head:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v4.yaml
scripts/run_query_local_segment_env30_frozen_probe20_v4.sh
```

Treat v4 as a diagnostic/probe path until hard official-GT coverage confirms
raw-segment and selected-gated gains on official-val plus train0601. TEST
remains closed.

The completed v5 selector/gate probe confirms that the remaining failure is
the learned proposal selector, not local segment capacity:

```text
run = query_local_segment_env30_frozen_probe20_v5
status = rejected for promotion
TEST used = false

official_best official-val ACC/FP/FN =
  0.972601 / 0.017585 / 0.011019

segment_best selected-gated hard hit20:
  official-val short GT5 = 0/53
  official-val short GT4 = 0/8
  train0601 short GT5 = 0/183
  train0601 short GT4 = 0/19

raw local-segment hard hit20:
  official-val short GT5 = 52/53
  official-val short GT4 = 4/8
  train0601 short GT5 = 179/183
  train0601 short GT4 = 19/19
```

The selected windows collapse to non-oracle short windows, commonly candidate
index `41` with length `3`, while the raw oracle windows are mostly lower
image windows with starts around `35..44` and lengths `4..8`. This means the
score/replace heads do not have enough proposal-local evidence to rank
`Q x 404` windows. Do not continue v5 as-is, increase epochs, run TEST, or use
`segment_best.pt` as a candidate. The next selector must score proposals from
local window evidence and pass selected-gated hard diagnostics first.

The v6 implementation target for that conclusion is:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v6.yaml
scripts/run_query_local_segment_env30_frozen_probe20_v6.sh
```

It keeps the v5 hard-gate protocol but scores each proposal from local
start/mid/end image samples plus proposal geometry. Treat v6 as a diagnostic
selector probe until `segment_best_hard_gate.json` shows selected-gated gains;
TEST remains closed.

The completed v6 b8s1 probe is rejected for promotion:

```text
run = query_local_segment_env30_frozen_probe20_v6_b8s1
status = rejected for promotion
TEST used = false

official_best base-decode official-val ACC/FP/FN =
  0.972582 / 0.017585 / 0.011019

segment_best_hard_gate selected = last
selected-gated hit20:
  official-val short GT5 = 11/53
  official-val short GT4 = 3/8
  train0601 short GT5 = 32/183
  train0601 short GT4 = 9/19

raw local-segment hit20 on the same selected checkpoint:
  official-val short GT5 = 52/53
  official-val short GT4 = 4/8
  train0601 short GT5 = 179/183
  train0601 short GT4 = 19/19

base-to-selected-gated GT5 gain/loss20:
  official-val = +2 / -31
  train0601 = +5 / -115
```

Interpretation: proposal-local evidence helped relative to v5's all-zero
selected-gated result, but the selector still converts only a small fraction
of raw oracle capacity and damages many base-hit lanes. The current failure is
not raw segment geometry; it is proposal ranking and base-preserve replacement
calibration. Do not continue v6 with more epochs, do not run TEST, and do not
promote `segment_best.pt`.

The completed selector rank audit on `last.pt` localizes the v6 failure more
precisely:

```text
rank audit outputs:
  runs/gcs_lane/query_local_segment_env30_frozen_probe20_v6_b8s1_rank_audit_val_last/
  runs/gcs_lane/query_local_segment_env30_frozen_probe20_v6_b8s1_rank_audit_train0601_last/

official-val short GT5:
  base/raw/selected-gated hit20 = 40/52/11 out of 53
  oracle hard-gate eligible = 52/53
  oracle selected as query top1 = 2/53
  oracle combined-rank-in-query top1/top5/top20 = 2/3/19
  selected window lengths collapse mostly to length 3: 48/53

train0601 short GT5:
  base/raw/selected-gated hit20 = 142/179/32 out of 183
  oracle hard-gate eligible = 179/183
  oracle selected as query top1 = 6/183
  oracle combined-rank-in-query top1/top5/top20 = 6/22/73
  selected window lengths collapse mostly to length 3: 164/183
```

This proves the replace threshold is not the main blocker: the oracle proposal
usually passes the hard gate, but is not ranked first inside its own query. A
diagnostic-only predicted-valid-overlap selector mask improves selected-gated
GT5 coverage only partially:

```text
official-val short GT5 selected-gated: 11/53 -> 19/53
train0601 short GT5 selected-gated:    32/183 -> 51/183
```

It still remains far below the frozen env30 base (`40/53`, `142/183`) and
continues to damage base-hit lanes. The root cause is training/selection
misalignment: the listwise target ranks only overlapping GT-visible candidates,
while `gcs_short_segment_bce_weight=0.0` leaves non-overlap high-score windows
without direct negative pressure. The next selector must train the same
applicability/window-overlap condition used at selection time and keep an
explicit base-preserve no-replace gate.

The v7 implementation target for that conclusion is:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v7.yaml
scripts/run_query_local_segment_env30_frozen_probe20_v7.sh
```

It keeps the v6 proposal-local selector architecture but changes the training
target: all-candidate listwise ranking, dense geometry-quality BCE for every
`Q x 404` proposal, explicit non-overlap/low-overlap negatives, dense
base-preserve no-replace negatives, and hard diagnostic selection with the same
combined score+replace logit. Treat v7 as a diagnostic selector probe until
`segment_best_hard_gate.json` shows selected-gated short GT5 above the frozen
base on official-val and train0601, with base-hit loss close to zero. TEST
remains closed.

Completed v7 result:

```text
run = query_local_segment_env30_frozen_probe20_v7_b4w0s1
selected checkpoint = last.pt

official-val short GT5 base/raw/selected-gated = 40/52/20 out of 53
train0601 short GT5 base/raw/selected-gated = 142/179/77 out of 183
official-val short GT4 selected-gated = 2/8
train0601 short GT4 selected-gated = 9/19

base-to-selected-gated short GT5 gain/loss20:
  official-val = +1 / -21
  train0601 = +5 / -70

oracle combined-rank-all short GT5:
  official-val top1/top5/top10 = 0/5/6
  train0601 top1/top5/top10 = 0/8/17
```

The route is rejected. Dense negatives improved v6's `11/53, 32/183`
selected-gated result, but the selector still fails to rank the oracle
proposal first and the replace gate damages base-hit lanes. Increasing the
diagnostic threshold from `0.5` to `0.8` also fails (`18/53`, `70/183`) and
increases base-hit loss. The next route must model an explicit calibrated
no-replace decision or base-versus-proposal comparison; do not add epochs,
enable decode, or run TEST from v7.

The v8 implementation target for that conclusion is:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v8.yaml
scripts/run_query_local_segment_env30_frozen_probe20_v8.sh
```

It makes the selector explicitly two-stage. The candidate score head now trains
with per-query segment ranking and dense geometry-quality targets, while
`pred_short_segment_query_replace_logits: B x 12` handles the no-replace versus
replace decision. The old per-candidate replace head is disabled in v8 so hard
diagnostics cannot maximize `Q x 404` `score + replace` globally. The hard
diagnostic mode is `segment_selection_score_mode=query_replace`: choose the
best candidate inside each query by candidate score, then apply that query's
replace gate. Treat v8 as a diagnostic selector probe until selected-gated
short GT5 exceeds the frozen env30 base on official-val plus train0601 and
base-hit loss is near zero. TEST remains closed.

Completed v8 result:

```text
run = query_local_segment_env30_frozen_probe20_v8_b4w0s1
status = rejected for promotion
training rows recorded = 10, requested epochs = 20
official_best source = epoch 5
last.pt = best.pt = epoch 10
TEST used = false

base-decode official-val ACC/FP/FN:
  official_best epoch5 = 0.972582 / 0.017585 / 0.011019
  last/best epoch10   = 0.972501 / 0.017126 / 0.011019
  env30 baseline      = 0.973330 / 0.015748 / 0.009642

short GT5 base/raw/selected-gated hit20:
  last/best official-val = 40/51/40 out of 53
  last/best train0601    = 142/176/141 out of 183
  official_best val      = 40/52/40 out of 53
  official_best train0601= 142/179/142 out of 183

short GT4 base/raw/selected-gated hit20:
  last/best official-val = 1/4/1 out of 8
  last/best train0601    = 9/19/9 out of 19
  official_best val      = 1/4/1 out of 8
  official_best train0601= 9/19/8 out of 19
```

v8 protects the env30 base path much better than v7, but mostly by refusing
replacement. The raw local-segment proposal capacity is still strong, while
the selected-gated output stays at base or slightly below base. The remaining
bottleneck is therefore still selector/ranking/applicability calibration:
the oracle proposal rarely ranks first, and the query-level replace head does
not produce a useful positive replacement decision. Do not add epochs to this
exact v8 setup, enable short-segment decode, or run TEST.

The next selector should make `base` a first-class choice in the same
supervised decision as the 404 segment windows, for example a per-query
softmax over `[base, segment_0, ..., segment_403]`. Targets should choose
base when base hits the 20px gate and choose a segment only when base misses
and the segment clearly improves APE or hits 20px. Keep dense negative
pressure for non-overlap windows and verify with a small rank/gate probe
before any formal decode experiment.

The v9 implementation target for this conclusion is:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v9.yaml
scripts/run_query_local_segment_env30_frozen_probe20_v9.sh
```

v9 removes the separate replace-head decision from the selector path. The
model emits proposal-local short-segment logits only; the loss and hard
diagnostic treat the frozen env30 base as a zero-logit candidate in the same
per-query choice as the 404 segment proposals. This directly targets the v8
failure mode where `query_replace` learned to refuse replacement while the
segment ranker still failed to promote oracle windows. TEST remains closed.

## 2026-07-27 Q12/env30 Lateral Candidate Probe Rejection

The run `query_short_candidate_env30_probe20_fix1` is rejected. It trained
only the 33k-parameter candidate score head from the env30
`official_best.pt`; the base model and main `pred_points` path stayed frozen.

The decisive same-checkpoint official-val comparison is:

```text
candidate_decode=false:
  ACC/FP/FN = 0.973330 / 0.015748 / 0.009642

candidate_decode=true, epoch005:
  ACC/FP/FN = 0.946994 / 0.048714 / 0.040404

candidate_decode=true, epoch010:
  ACC/FP/FN = 0.893279 / 0.120202 / 0.101469
```

The normal decode result proves that the env30 carrier geometry was not
damaged. The candidate path itself is harmful:

```text
train short_candidate_best_hit20: 0.27377 -> about 0.81
val   short_candidate_best_hit20: 0.34524..0.38690
val   short_candidate_raw_hit20:  0.70238
```

The raw candidate pool contains useful hypotheses on validation, but the
learned score selects the wrong one. The implementation has two structural
distribution mismatches:

1. The loss supervises only matched short GT4/GT5 lanes, while decode applies
   candidate selection to every Q12 query. Normal lanes and unmatched queries
   therefore receive an out-of-distribution score perturbation.
2. Candidate features are averaged over all K=56 anchors without predicted
   visibility weighting, although the target geometry is computed only over
   visible anchors. For short lanes, most pooled features are irrelevant
   background.

The relative candidate logit is also added to the main query existence score,
which makes an unstable auxiliary score alter ordinary ranking. Do not
continue this run, enable TEST, or tune only `gcs_short_candidate`/learning
rate/epochs. The next experiment must isolate candidate selection from the
normal path, gate it with prediction-only short-lane applicability, and use
visibility-aware scoring before any new official-val probe.

## 2026-07-27 Q12/env30 Windowed Short Local X-Refine v3 Rejection

The 10-epoch `query_short_local_refine_env30_window_v3_probe10` run is
complete and rejected for promotion. It was initialized from the env30
official-best checkpoint, froze the base path, and trained only the
window-search auxiliary head. TEST was run once with the frozen official-val
decode for reporting only.

Official-val:

```text
official_best.pt ACC/FP/FN = 0.973365 / 0.015748 / 0.009642
best.pt          ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
```

Raw/refined short-lane gate:

```text
official-val short GT4: 1/8 -> 2/8
official-val short GT5: 40/53 -> 40/53
train0601 short GT4:    9/19 -> 9/19
train0601 short GT5:    142/183 -> 142/183
```

The useful signal is only continuous-error reduction. For example, train0601
short GT5 refined p90 APE moves from `36.315779px` to `32.862357px`, and the
30px/40px hit rates improve, but the 20px hit20 count does not change. The
identity guard prevents coarse-hit regressions, but it also means this probe
does not create a new candidate when the raw query is far from the GT lane.

Reporting-only TEST:

```text
official_best.pt ACC/FP/FN = 0.966680 / 0.027924 / 0.023724
best.pt          ACC/FP/FN = 0.966684 / 0.027995 / 0.023724
```

Compared with the env30 official-best reporting result
`0.966780 / 0.028732 / 0.023544`, v3 does not improve ACC, and GT5 retention
is lower (`count_acc_5=0.880492` versus `0.891037`). This is not a reason to
tune TEST. It confirms that the current auxiliary local residual path is not
yet a solution.

Integrated conclusion:

The v3 mechanism learns to smooth existing carriers, but the dominant
short-lane failure is still candidate coverage and 20px crossing, not
ranking or count selection. Do not continue this exact artifact to 20/100/220
epochs and do not enable refined decode from it. The smallest informative next
step is an official-val-only prediction-only refined decode/sweep. If that
does not improve formal ACC and short hit20, replace residual refinement with
a candidate-generation mechanism that can produce a new lateral hypothesis.

## 2026-07-28 Env30 Baseline Reset

The active source/config is restored to env30 commit
`86c8fb31cb4b48a53086be183478a95b0807753d` (`Add GT4 GT5 weak geometry rescue
run`). The worktree code, configs, scripts, tools, and tracked reference banks
must match that baseline outside documentation.

All commits after `86c8fb31c` are rejected experiment records unless a future
task explicitly reopens one with new official-val/train-side gates. This
closes env30 staticref/valid-neg/near20 follow-ups, Q20/Q24 protected-static
or dual-head routes, Q12 dual-head, query extent, short local-refine, lateral
candidate, gated candidate, Q24 role/event containment, and candidate gate
fixes as active paths. Their evidence remains below only to explain why they
must not be relaunched or used for TEST/threshold tuning.

## Branch Scope

The current mainline imports the historical `5-25-3.zip` algorithm and changes the TuSimple fixed-y contract to Q=12/K=56 with official h-sample anchors. It also includes the env30-baseline default-off `count_boundary_loss`, train-only `gcs_hard_sampling`, E3-lite `gcs_spurious_neg` family, training-time `official_best`, `valid_before_maxdet`, and default-off query Count Head ablation present at `86c8fb31c`.

Do not read mainline Count Head, Quality Head, Survival Head, near-miss, or old mainline official-best bottlenecks as active branch behavior. Those algorithm mechanisms are not part of this 5-25-3 branch. The only active Count Boundary behavior is the branch-local default-off `count_boundary_loss`, the only active hard sampler is the branch-local default-off train-only `gcs_hard_sampling`, the only active E3-lite spurious negative behavior is the branch-local default-off `gcs_spurious_neg` family described in `current-contracts.md`, and the only active official-best behavior is the explicit 2026-06-27 training-time official-val selection hook.

Active source/config is restored to commit `86c8fb31c` (`Add GT4 GT5 weak
geometry rescue run`). Bottleneck notes below that depend on post-`86c8fb31c`
mechanisms are rejected legacy experiment conclusions only. They do not
describe currently available code, CLI flags, loss terms, diagnostic scripts,
configs, model outputs, or active selected candidates unless a future task
explicitly reopens them.

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
- It does not include post-`86c8fb31c` staticref, Q20/Q24, dual-head, query extent, short local-refine, lateral-candidate, gated-candidate, later mainline Count/Quality/Boundary diagnostics, Survival, or near-miss machinery as active code.

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

### Q24 Event-Containment Probe40 v1 Rejection

The event-containment probe
`query_alpha05_env30_q24_event_containment_probe40_v1` is complete and
rejected for longer/full training. TEST was not used.

Primary official-val evidence:

```text
official_best source_epoch = 40
official-val ACC/FP/FN = 0.961976 / 0.076860 / 0.025253
count_acc_3/4/5 = 0.834081 / 0.772727 / 0.527027

env30 official-val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
env30 count_acc_3/4/5 = 0.968610 / 0.969697 / 0.986486
```

The official-val count confusion remains unsafe:

```text
3->4=33, 3->5=4
4->3=1, 4->5=11, 4->6=3
5->5=39, 5->6=35
```

The 864-row official-val sweep cannot rescue the checkpoint:

```text
rows with ACC >= env30 = 0
rows with FP/FN both <= env30 = 0
rows with count_acc_4 >= 0.90 = 0
max ACC = 0.961976
min FP = 0.058586
min FN = 0.025253
max count_acc_4 = 0.787879
```

The event path was active, so this is not an implementation no-op:

```text
epoch040 train q24_event_contain_loss = 0.00029
epoch040 val q24_event_contain_loss = 0.00050
epoch040 train q24_event_gt3_count = 90.1569
epoch040 train q24_event_gt4_count = 237.912
epoch040 train q24_event_gt5_risk_count = 8.62745
```

Raw/event diagnostics show why the result is still bad:

```text
val/train0601 GT5 visible<=10 raw match20 = 0.509434 / 0.486339
env30 val/train0601 GT5 visible<=10 raw match20 = 0.754717 / 0.775956

val/train0601/train0531 GT3GT4 visible<=20 q12..q23 raw-best rate =
  0.578778 / 0.634703 / 0.646018

event gate high-risk queries:
  q11 risk=26, q16 risk=17, q0 risk=12, q20 risk=12, q1 risk=11
```

The previous event query assignment was partly wrong. It successfully crushed
some earlier high-risk carriers (`q21/q22/q23`), but risk migrated to
uncovered `q16`, protected original queries `q0/q1/q11`, and the supposedly
clean `q20`. At the same time, clean true-GT5 short raw carriers such as
`q13/q15` had very low exist scores and therefore did not reliably survive
decode.

Decision:

Do not continue this artifact, do not full-train it, and do not run TEST. The
undertraining hypothesis is not enough to explain this failure: the 40-epoch
trend improves ACC/FN slowly, but `count_acc_4` never approaches the gate,
`count_acc_5` oscillates, no sweep row reaches an FP/FN-safe region, and the
prior 100-epoch Q24 continuation already showed the same extra-query carrier
drift does not naturally disappear.

Smallest safe next action:

Stop fixed query-ID hard partitioning as the main mechanism. The next Q24
probe, if continued, should be a new 20-40 epoch official-val/train-side
diagnostic with dynamic event-aware containment and score calibration:

```text
- suppress false-extra event candidates across all queries, including
  q0/q1/q11/q16/q20, only when they are not GT-close true positives
- boost/calibrate true GT5 short clean carriers such as q13/q15 when matched
  to real GT5 short lanes
- keep TEST closed
- require count_acc_4 >= 0.90, count_acc_5 >= 0.972973, FP/FN improvement,
  lower GT3/GT4 false-extra, lower GT5->6, and no GT5 short raw regression
```

Implementation status:

The default-off Q24 event-v2 probe is implemented through the new dynamic and
score-calibration flags plus `scripts/run_q24_event_v2_dynamic_score_probe_v1.sh`.
It keeps `q13/q15` as clean score-calibrated true-GT5-short carriers, moves
`q16` into the high-risk event set, removes `q20` from clean GT5, and watches
`q0/q1/q11/q20` through dynamic GT-close-protected containment. This is not a
longer/full-training candidate until the new 20-40 epoch official-val and
event-mined gates pass. Use `scripts/run_q24_event_v2_mined_gate_v1.sh` after
training so the diagnostic role labels match the v2 query assignment.

Completed result: `query_alpha05_env30_q24_event_v2_dynamic_score_probe40_v1`
fails the promotion gate and is rejected for longer/full training. TEST was
not used.

```text
official-val ACC/FP/FN =
  0.960033 / 0.090037 / 0.026630
count_acc_3/4/5 =
  0.865471 / 0.772727 / 0.148649
count_confusion includes:
  3->4=24, 3->5=6, 4->5=5, 4->6=9, 5->4=1, 5->6=62

official_best.pt sweep:
  rows = 864
  rows with ACC >= env30 = 0
  rows with FP/FN both <= env30 = 0
  rows with count_acc_4 >= 0.90 = 0
  max count_acc_4 = 0.772727
```

The training logs show the new code path was active:

```text
train q24_event_dynamic_count last10 mean = 4.652944
train q24_event_score_pos_count last10 mean = 0.684314
val q24_event_dynamic_count last10 mean = 7.066667
val q24_event_score_pos_count last10 mean = 1.900000
```

But the mechanism made the core tradeoff worse. It reduced GT3/GT4
short-visible `q12..q23` carrier rate to `0.083601 / 0.132420 / 0.106195` on
val/train0601/train0531, but GT5 visible<=10 raw match20 regressed to
`0.245283 / 0.229508` on val/train0601. The event-mined gate still found
`total_val_gt5_to6=61`, `total_val_gt34_false_extra=44`, and no high-true,
low-risk query. `q13/q15` received true-short score calibration, but their
true GT5 short scores overlap their GT5->6 scores, so score calibration cannot
separate true fifth lanes from pseudo sixth lanes in this design.

Conclusion: the current Q24 line has exhausted fixed partition, boundary-safe,
event-containment, and event-v2 dynamic-score probes. The failure is the
coupled event-role and score-separation bottleneck, not a no-op implementation
or only short training. Do not continue Q24 event-v2 or run TEST.

### Q12 Env30 Dual-Head Next Route

Integrated bottleneck after the Q24 rejection chain:

The strongest remaining hypothesis is no longer "add more queries and patch
their roles." The Q24 line proved that extra capacity creates an unstable
carrier problem: the channels that can reach GT5 short lanes also become
GT5 pseudo-sixth or GT3/GT4 false-extra carriers, and suppression damages
true GT5 short raw coverage. The more general structural problem is that
`pred_logits` has been overloaded as:

```text
1. existence/objectness supervision
2. lane quality and decode ranking score
3. score-sum lane-count estimator
```

These objectives conflict after `exist_loss` becomes quality-aware. A matched
lane with imperfect geometry or visible-IoU receives a lower quality target,
so `sigmoid(pred_logits)` is no longer a pure lane-count probability.
Optimizing score-sum count losses or Q24 event score patches on the same logit
therefore pulls against ranking and geometry quality.

Decision:

Implement the next probe as a default-off Q12/env30 dual-head candidate:

```text
pred_count_logits: image-level 2/3/4/5 count CE
pred_quality_logits: query-level lane quality/ranking score
pred_logits: original query existence/objectness and old-checkpoint fallback
decode: count_logits chooses k_hat; quality_logits ranks top-k
```

The new route intentionally disables the rejected score-sum count losses and
Q24 patch losses for this probe:

```text
gcs_count = 0.0
gcs_count_under5 = 0.0
gcs_count_boundary = 0.0
gcs_boundary_pseudo_neg = 0.0
gcs_role_contain = 0.0
gcs_q24_event_contain = 0.0
gcs_q24_event_score_calib = 0.0
```

This is not yet promotion evidence. The 40-epoch probe must pass
official-val and train-side gates before any longer/full training or TEST:

```text
count_acc_4 must recover away from the Q24 0.74-0.82 band
count_acc_5 must remain high without 5->6 rebound
FP/FN must move toward env30, not Q24 event-v2
quality top-k must choose true GT4/GT5 lanes rather than pseudo/extra lanes
```

Completed 40-epoch review:

```text
run = query_dualhead_quality_count_env30_probe40_v1
status = not promoted to full training; TEST closed
official_best source_epoch = 40
official-val ACC/FP/FN = 0.962494 / 0.025941 / 0.021579
count_acc_3/4/5 = 0.991031 / 0.954545 / 0.972973
```

What worked:

- `pred_count_logits` learned the image count quickly:
  `val/query_count_acc` reached `0.96233` by epoch040, and official-val
  `count_acc=0.980716`.
- The count shape is much safer than Q24 event/role probes:
  only `3->4=2`, `4->3=3`, and `5->4=2`.
- At the same epoch040 point, it is stronger than env30 official-val:
  `0.962494` versus env30 epoch040 `0.958436`.

What failed:

- The primary official-val gate is still far below env30 epoch220
  `0.973330 / 0.015748 / 0.009642`.
- The 864-row official-val sweep has zero rows at or above env30 ACC and zero
  rows with FP/FN both at or below env30.
- GT-count oracle-rank only raises ACC to `0.963281`, and removing the oracle
  pool cap gives the same result. This rules out count/rank/max-det truncation
  as the main 40-epoch bottleneck.
- Raw candidate geometry is weak: official-val `has_match20=0.933998` versus
  env30 final `0.976209`. The largest bucket regression is GT5
  visible<=10: `0.509434` versus env30 final `0.754717`.

Interpretation:

The 40-epoch result is not a full-training candidate, but it is not a Q24-style
mechanism rejection. The current bottleneck is undertrained or interfered
candidate geometry/valid quality after adding the dual auxiliary heads; count
estimation itself is no longer the bottleneck. The quality head has not yet
proven ranking value, because count/rank oracle controls can recover less than
`+0.001` ACC.

Smallest safe next action:

Run a 100-epoch official-val-only diagnostic of the same dual-head route. Do
not run TEST. Promote to 160/220 only if epoch100 beats the old query-count-only
`0.968473` official-val ACC, approaches env30 epoch100 `0.969213`, and raw
GT5 visible<=10 match20 recovers toward at least `0.65`. If that fails, stop
this exact dual-head loss/decode and change the mechanism: keep the count head,
but reduce or remove the quality-head training/decode dependency and re-check
whether restoring env30's geometry/objectness path recovers raw match20.

## Q12 Dual-Head Probe100 Final Review

The 100-epoch run
`query_dualhead_quality_count_env30_probe100_v1` is rejected for blind
220-epoch continuation. TEST was run once only as a reporting-only evaluation
using the frozen official-val decode; it is closed for this artifact.

Official-val:

```text
ACC/FP/FN = 0.966268 / 0.015335 / 0.015840
count_acc_3/4/5 = 0.991031 / 0.939394 / 0.972973
GT-count oracle-rank ACC = 0.966877
raw has_match20 = 0.959325
GT5 visible<=10 raw match20 = 0.754717
```

The training-length hypothesis is partially confirmed: from the 40-epoch
dual-head run to epoch100, official-val ACC rises from `0.962494` to
`0.966268`, raw `has_match20` rises from `0.933998` to `0.959325`, and GT5
short raw match20 rises from `0.509434` to `0.754717`. However, epoch100
still fails the continuation gate: it is below query-count-only `0.968473`,
below env30 epoch100 `0.969213`, and `count_acc_4=0.939394` is below
`0.954545`.

TEST reporting-only result:

```text
ACC/FP/FN = 0.963288 / 0.028481 / 0.031332
count_acc/count_acc_4/count_acc_5 = 0.877067 / 0.611111 / 0.776801
count confusion includes 4->3=119, 4->5=62, 5->4=105, 5->5=442
```

This exposes a large official-val-to-TEST count generalization gap. It is
consistent with the official-val GT4 weakness and shows that the current
count-aware dual-head calibration is not robust on GT4/GT5 images.

Integrated conclusion:

- Longer training improves geometry, especially GT5 short lanes.
- The current count/quality split does not yet produce a promotable model.
- Oracle rank adds only `+0.000609` ACC at epoch100, so count/ranking is no
  longer the dominant bottleneck.
- Do not spend the next full 220-epoch run on the unchanged dual-head path.
  Keep the count head, but first change the quality-head dependency or restore
  env30's geometry/objectness behavior, then run a new official-val probe.

## 2026-07-26 Q12 Count Head-only Review And Query Extent Direction

The Count Head-only 100-epoch probe
`query_count_head_ce025_env30_probe100_v1` is rejected for blind 220-epoch
continuation. It confirms count/rank is not the current bottleneck:

```text
official-val ACC/FP/FN = 0.968109 / 0.016667 / 0.016070
count_acc = 0.975207
count_acc_4 = 0.924242
oracle-rank ACC = 0.968244
oracle-rank gain = +0.000135
raw official-val has_match20 = 0.960860
```

The weak cases are short GT4/GT5 candidate geometry and endpoint/visibility
quality, not query count capacity:

```text
official-val short GT4 visible<=10:
  lanes = 8
  has_match20 = 0.125
  best_ape_px_p90 = 32.301349

official-val short GT5 visible<=10:
  lanes = 53
  has_match20 = 0.698113
  best_ape_px_p90 = 38.932954
```

Decision:

Do not continue patching Q24 extra-query carrier drift and do not add another
Count Head / Quality Head variant as the immediate next path. The next
controlled experiment is the default-off Q12/env30 query extent v1 probe,
which makes each query explicitly learn a continuous first/last visible fixed-y
interval on top of the env30 parent protocol (`gcs_short_geom=1.0`) and then
compares `none`, `interval`, and `intersect` decode modes on official-val
only. If endpoint/interval learning fails, investigate coarse
geometry/reference coverage before implementing extent-guided local refine.

## 2026-07-26 Q12/env30 Query Extent Probe40 Rejection

The 40-epoch default-off extent probe
`query_extent_env30_probe40_v1` is complete and rejected. TEST was not used.

```text
official_best ACC/FP/FN =
  0.966997 / 0.052020 / 0.019972
official_best count_acc_3/4/5 =
  0.856502 / 0.818182 / 0.986486

best.pt post-train sweep best ACC/FP/FN =
  0.967358 / 0.055326 / 0.023416
best.pt count_acc_4/count_acc_5 =
  0.772727 / 0.959459
```

Neither checkpoint passes the probe gate. The selected `official_best` decode
is `extent_decode=false`. In the post-train sweep, the best rows by extent mode
are:

```text
none      ACC 0.966997, FP 0.052020, FN 0.019972, count_acc_4 0.818182
intersect ACC 0.964296, FP 0.053627, FN 0.021579, count_acc_4 0.818182
interval  ACC 0.964134, FP 0.103489, FN 0.023186, count_acc_4 0.621212
```

`interval` is not a viable rescue path. At the selected decode settings it
changes the prediction count histogram from `3:191, 4:86, 5:86` to
`3:18, 4:133, 5:212`, raising false-extra rates to:

```text
normal_gt3_extra_rate = 0.919283
normal_gt4_extra_rate = 0.727273
```

The endpoint/extent diagnostics must be read with raw-geometry strata. When
short GT5 already has a 20px raw candidate, endpoint prediction is good:

```text
short GT5 raw-hit count = 23
endpoint_start_acc_1 = 0.956522
endpoint_end_acc_1 = 0.956522
interval_iou = 0.865525
best_ape_p90 = 19.172512
```

The failures are dominated by raw candidate absence or large raw APE:

```text
official-val short GT4 visible<=10:
  raw hit20 = 1/8
  finite p90 APE = 53.089px
  inf APE count = 4

official-val short GT5 visible<=10:
  raw hit20 = 23/53
  finite p90 APE = 100.806px
  inf APE count = 1

train0601 short GT4 visible<=10:
  raw hit20 = 6/19
  p90 APE = 58.760px

train0601 short GT5 visible<=10:
  raw hit20 = 71/183
  finite p90 APE = 95.287px
  inf APE count = 4
```

The worst short GT5 failures often keep high point-valid recall while the x
geometry is off by 90-150px, so point-valid survival is not the dominant
cause. Query assignment is also concentrated: official-val short GT5 raw-best
queries are mostly q0 (`40/53`) and q11 (`12/53`), with q11 hit20 only
`1/12`; train0601 short GT5 is q0 (`130/183`) and q11 (`46/183`), with q11
hit20 only `2/46`.

Conclusion:

Do not continue extent-only v1, do not enable extent decode, and do not build
extent-guided local refine v2 until raw geometry coverage improves. The next
experiment should mine short GT4/GT5 raw misses on official-val plus
train0601/train0531 and then test a default-off coarse geometry/reference
coverage change that improves raw 20px hit rate without increasing GT3/GT4
false-extra or reducing GT5-short coverage.

### Raw-Miss Mining After Extent Rejection

The server-side raw-miss mining pass was run under:

```text
artifact root =
  runs/gcs_lane/raw_miss_mining_20260726
summary =
  runs/gcs_lane/raw_miss_mining_20260726/failure_mining_summary.json
cases =
  runs/gcs_lane/raw_miss_mining_20260726/short_gt45_env30_vs_dataref_cases.csv
TEST used = false
```

It compared the env30 strong baseline against the existing Q12 ultrashort
dataref/reference variant on official-val, train0601, and train0531. Dataref
does not solve the bottleneck:

```text
official-val short GT4:
  env30 1/8 -> dataref 2/8
official-val short GT5:
  env30 40/53 -> dataref 38/53

train0601 short GT4:
  env30 9/19 -> dataref 8/19
train0601 short GT5:
  env30 142/183 -> dataref 127/183

train0531 short GT4:
  env30 2/3 -> dataref 0/3
```

Pairwise changes show a negative tradeoff rather than a root fix:

```text
val short GT4: gain 2, loss 1, both_miss 5
val short GT5: gain 4, loss 6, both_miss 9
train0601 short GT4: gain 3, loss 4, both_miss 7
train0601 short GT5: gain 10, loss 25, both_miss 31
train0531 short GT4: gain 0, loss 2, both_miss 1
```

The actionable env30 misses are often near the 20px gate rather than missing by
an entirely different reference:

```text
env30 val short GT5:
  hit20 40, near20_30 3, near30_40 6, far40_80 3, ultra_visible_le2 1
env30 train0601 short GT5:
  hit20 142, near20_30 8, near30_40 18, far40_80 11, ultra_visible_le2 4
env30 train0601 short GT4:
  hit20 9, near20_30 6, near30_40 1, far40_80 3
```

Conclusion: the next mechanism should preserve env30's Q12 carriers and add a
default-off coarse-to-fine local x-refinement path for short GT4/GT5 near-miss
geometry. Do not relaunch Q12 dataref/reference-only as the immediate next
experiment. Gate the next run on refined raw `has_match20` for short GT4/GT5,
GT3/GT4 false-extra, and official-val only; TEST remains closed.

2026-07-27 implementation note: the default-off v1 probe was implemented as
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-short-local-refine.yaml`
with `scripts/run_query_short_local_refine_env30_probe40_v1.sh` and
`scripts/run_query_short_local_refine_env30_gate_v1.sh`.

2026-07-27 v1 failure note: the first 40-epoch implementation is rejected
because it changed `pred_points` itself, used pixel-scale auxiliary loss, and
trained from scratch. That made the auxiliary branch dominate the main lane
geometry, causing official-val geometry/count collapse. This does not reject
the local x-refine idea. The corrected v2 path keeps main `pred_points` for
matcher/loss/decode, emits `pred_short_refined_points` only as an auxiliary
bounded output, uses normalized-x SmoothL1 with
`gcs_short_local_refine=0.02`, caps the residual by
`gcs_short_local_refine_max_delta_px=40.0`, and starts from the env30
`official_best.pt` for a 20-epoch official-val/train-side probe. TEST remains
closed.

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
in the active env30 rollback code:

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
in the active env30 rollback code:

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
the active env30 rollback code:

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

## v11 Selector Calibration Failure (2026-08-01)

The v11 frozen-base official-quality probe confirms that the remaining
bottleneck is the selector decision, not candidate geometry:

```text
official-val short GT5: base/raw/selected-gated = 40/52/40 of 53
train0601    short GT5: base/raw/selected-gated = 142/178/142 of 183
official-val short GT4: base/raw/selected-gated = 1/4/1 of 8
train0601    short GT4: base/raw/selected-gated = 9/18/9 of 19
```

The strict v11 reference-point base sweep was `0.973330` ACC, versus the
env30 reference `0.973346`; the frozen base is effectively unchanged.
Changing the hard diagnostic threshold from `0.5` to `0.05` changed nothing.
The oracle segment was still top-1/top-5 in only `2/5` official-val GT5
cases and `6/13` train0601 GT5 cases, with hard-gate top-1/top-5 equal to
`0/0`.

Interpretation: dense BCE on a hard `0.85` point-quality target and
image-level all-query base-choice CE are not equivalent to the inference
problem of selecting one segment inside a matched query. Cross-GT target
competition and loss of continuous APE ordering leave the usable segment
logits under-calibrated. Do not address this by adding epochs, lowering the
decode threshold, or enabling TEST.

The next selector experiment must first validate a matched per-query
`[base + 404 segments]` softmax/ranking audit with continuous quality and
explicit wrong-window negatives. It is not eligible for official decode until
selected-gated coverage exceeds the frozen base on both splits.

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
## v12 Unified Choice Assignment Failure (2026-08-01)

The v12 frozen-base probe completed `20/20` epochs, but its hard diagnostic
stayed at the frozen base:

```text
official-val GT5 base/raw/selected-gated = 40/52/40
train0601    GT5 base/raw/selected-gated = 142/179/142
official-val GT4 base/raw/selected-gated = 1/4/1
train0601    GT4 base/raw/selected-gated = 9/19/9
```

The raw head is still useful. Its mean APE is `2.84px` on official-val GT5
and `2.99px` on train0601 GT5, versus selected/base `13.00px` and `13.74px`.
The issue is not raw local-segment geometry or a decode threshold.

The v12 loss has a structural supervision mismatch:

1. Positive unified-choice targets use the frozen-base Hungarian
   `matched_query_for_lane`.
2. Every query receives base/no-replace pressure through
   `gcs_short_segment_unified_choice_base_neg_weight`.
3. Raw oracle proposals are often carried by another query. On GT5 rows, the
   raw candidate query differs from the base-best query in `12/52` official-val
   cases and `50/179` train0601 cases.
4. The useful unmatched query is consequently trained toward class `base`,
   so the unified selector selects base and cannot expose the raw headroom.

The v12 combined oracle rank remains weak (`top1/top5 = 6/6` of 52 on
official-val and `24/26` of 179 on train0601), so changing only the gate
threshold is insufficient.

Next smallest experiment: implement a candidate-aware assignment audit first.
For each short GT4/GT5 lane, match queries using the best valid segment quality
with a base-geometry fallback, then train the unified `[base + 404 segments]`
choice on that assignment. Apply base negatives only to queries that are not
near/useful for any short GT lane. Require query-rank top1/top5 improvement and
selected-gated gain on both official-val and train0601 before any decode.
TEST remains closed.

## v13 Candidate-Aware Assignment Still Fails (2026-08-02)

The v13 frozen-base candidate-aware assignment probe completed all `20/20`
epochs and enabled both `gcs_short_segment_candidate_aware_assignment` and
`gcs_short_segment_matched_assignment`. It did not convert raw local-segment
capacity into selected-gated coverage:

```text
official-val short GT5: base/raw/selected-gated = 40/52/40
train0601    short GT5: base/raw/selected-gated = 142/179/142
official-val short GT4: base/raw/selected-gated = 1/4/1
train0601    short GT4: base/raw/selected-gated = 9/19/9
```

The failure is not a hard-gate threshold or a base-destruction regression:
base-to-selected loss was zero everywhere, but base-to-selected gain was also
zero everywhere. The candidate-aware target was enabled, yet the selector
audit reported no selected or gated oracle query on the examined groups. The
raw candidate remained available, so the remaining failure is the choice
distribution itself.

The v13 configuration exposes the structural cause:

```text
unified_choice_weight = 1.0
unified_choice_base_neg_weight = 0.25
bce_weight = 0.0
dense_quality_weight = 0.0
```

The loss initializes every query with class 0 (`base`) and applies base
negative pressure broadly. Only a sparse candidate-aware target can overwrite
that class. This overloads class 0 with two different meanings:

1. preserve the base geometry for a query that represents a real lane;
2. no useful short-segment proposal exists for an unassigned query.

The model therefore learns a strong base prior instead of a calibrated
base-versus-segment decision. The `short_segment_neg_count` remained zero in
the v13 logs, so wrong/non-overlap segment windows did not receive explicit
dense negative pressure either.

Next smallest safe action: build v14 as a two-part selector protocol. First,
train dense continuous geometry quality for every segment window, including
non-overlap negatives. Second, apply a balanced base-versus-segment softmax
only to candidate-aware assigned/protected short-lane queries; leave unrelated
queries ignored by that softmax and use a separate applicability/reject target
if needed. The same quality score and applicability rule must be used in the
rank audit and later decode. Require official-val and train0601 selected-gated
gain before any segment decode or TEST.

## 2026-08-02: Fundamental correction to the local-segment interpretation

The v4-v13 raw local-segment numbers are partial-window geometry diagnostics,
not complete lane recall. In the v13 hard CSV, the best raw segment passed the
local `20px` APE gate on `52/53` official-val GT5 lanes and `179/183`
train0601 GT5 lanes, but covered every visible GT point in only `1/53` and
`5/183` cases. The corresponding GT4 full-visible coverage was `0/8` and
`1/19`.

This explains why selector variants could improve local oracle coverage but
could not improve official decode. The segment head uses the same base query
carrier and predicts a bounded residual inside a short window; it does not
generate a complete lane instance. Its loss and diagnostic accept
`overlap >= 3`, while the official metric evaluates the complete lane set.
Changing candidate offsets, ranking losses, replacement heads, or thresholds
cannot close this representation/metric gap.

The problem should be considered a task-definition failure: a complete
lane-set prediction problem was decomposed into partial patch selection plus
post-hoc replacement. The correct future experiment is a new
image-conditioned full-lane proposal decoder with unified set matching,
visibility, existence, and quality supervision. Keep local-segment decode
default-off and TEST closed until that proposal contract is verified by a
full-visible-span oracle and an official-val result.

## 2026-08-03: Full-lane proposal v1 receives no positive supervision

The first independent full-lane proposal probe
`full_lane_proposal_env30_probe20_v1` completed `20/20` epochs, but does not
prove that the new proposal decoder works.

Training evidence:

```text
train/full_lane_match_count = 0 for epochs 1..20
val/full_lane_match_count   = 0 for epochs 1..20
train/full_lane_point_loss  = 0
val/full_lane_point_loss    = 0
train/full_lane_interval_loss = 0
val/full_lane_interval_loss   = 0
```

The full head was enabled, but every full proposal was treated as unmatched.
The loss quickly became near-zero because it only learned negative
existence/visibility targets. The new proposals never learned complete lane
geometry, visibility intervals, or quality.

Same-checkpoint paired official-val decode at the best-row settings showed no
ACC gain from `--full-lane-decode`:

```text
checkpoint = full_lane_proposal_env30_probe20_v1/weights/official_best.pt
conf/point_valid_thr/nms/max_det = 0.003 / 0.6 / 0 / 6

base decode      ACC/FP/FN = 0.974316 / 0.013636 / 0.011938
full-lane decode ACC/FP/FN = 0.974316 / 0.012948 / 0.011938
```

The strict full-visible-span oracle confirms the head has no complete-lane
capacity after this training:

```text
official-val all lanes: full_proposal_oracle_full_span = 0/1303
official-val GT5:       full_proposal_oracle_full_span = 0/370
train0601 all lanes:    full_proposal_oracle_full_span = 0/1787
train0601 GT5:          full_proposal_oracle_full_span = 0/1195
```

Root cause:

The first v1 design put mature env30 base queries and randomly initialized
full proposals into one Hungarian matching problem from the start. The base
queries already have good geometry and high scores, while full proposals start
with low score biases and poor geometry. Base queries therefore take every GT
match. Once that happens, the full head receives only unmatched negative
targets, so it is pushed further away from ever becoming positive.

Next safe action:

Do not add epochs, tune full-lane thresholds, or run TEST from v1. The next
full-lane experiment must add an auxiliary full-proposal-positive training
stage or parallel proposal-only assignment so each GT lane supervises at least
one full proposal before unified base/full competition is enabled. Unmatched
negative pressure must be delayed or down-weighted during warmup, and a gate
must require `full_lane_match_count > 0`, nonzero point/interval losses, and
nonzero full-proposal full-span oracle before any official decode sweep.

## 2026-08-03: Full-lane proposal v2 warmup opens supervision but not GT5 geometry yet

The initial v2 remote launch stopped on the first batch because AMP exposed a
loss dtype mismatch: `quality_target` was half precision while the detached
full-lane quality target was float32. This was a code bug, not evidence
against the v2 training protocol. The fix casts the detached quality target to
the proposal tensor dtype before assignment.

After the fix, the one-epoch smoke run
`full_lane_proposal_env30_auxwarm_v2_smoke_b4w0_e1_fix1` completed with
`RUN_TESTS=0`. The v2 proposal-only assignment now gives full proposals real
positive supervision:

```text
train/full_lane_match_count = 15.2414
train/full_lane_point_loss = 0.14676
train/full_lane_interval_loss = 4.56165
val/full_lane_match_count = 29.5217
val/full_lane_point_loss = 0.08634
val/full_lane_interval_loss = 4.25919
```

Strict full-visible-span oracle on `last.pt`:

```text
official-val all full proposal full-span/full-hit = 423/8 of 1303
official-val GT5 full proposal full-span/full-hit = 107/0 of 370
train0601 all full proposal full-span/full-hit = 551/8 of 1787
train0601 GT5 full proposal full-span/full-hit = 353/4 of 1195
```

Interpretation:

v2 fixes the v1 match-starvation bottleneck: the full proposal head is no
longer trained only as negative. The remaining bottleneck is complete-lane
geometry accuracy, especially GT5. The proposal visibility span begins to
cover many GT5 lanes, but the full-span APE is still far above the 20px gate
after one epoch.

Next action:

Run a real multi-epoch v2 warmup diagnostic and judge it by loss trends plus
strict full-span oracle, not by TEST. Do not run a full-lane official decode
sweep until GT5 full-hit and union-oracle gains become nontrivial on both
official-val and train0601.

Follow-up E5-to-20 result:

Plain v2 warmup was continued from the E5 checkpoint for 15 additional epochs
in `full_lane_proposal_env30_auxwarm_v2_e5to20_gate_fix3`. This proves the
positive-supervision loop is not the remaining bottleneck: full-lane point,
valid, and interval losses all continued to decrease, and the full head
produced many more full-span/full-hit proposals than the smoke run.

However, this did not solve the target ACC bottleneck. Strict oracle on
`last.pt` found only tiny union gains over the frozen env30 base:

```text
official-val GT5 union gain = +1 of 370
train0601 GT5 union gain    = +3 of 1195
official-val gt5_vis_6_10 union gain = +0 of 49
train0601 gt5_vis_6_10 union gain    = +0 of 170
```

The hard-lane failure mode is now precise: the full proposals often cover the
complete visible span but are not geometrically accurate enough. On
`gt5_vis_6_10`, official-val has `44` full-span proposals but only `1`
full-hit; train0601 has `154` full-span proposals but only `1` full-hit.

This rejects "continue the same warmup" as a serious next step. The next
version must change the target distribution, not just add epochs: train the
full proposal head preferentially on base-miss short GT4/GT5 and visible-short
lanes, down-weight ordinary base-hit lanes that env30 already solves, and use
hard-stratum union gain as the first gate before any decode or TEST action.

The v3 hard-focus implementation tested that target-distribution change and
did not pass the gate. It increased train0601 `gt5_vis_6_10` union coverage
from `130/170` to `131/170`, but official-val stayed at `39/49`. Official-val
GT5 full proposal hits also fell from `50` in v2 to `35` in v3, even though
the hard-focus score rank improved on some groups.

This separates the remaining issue from ordinary positive assignment:
reweighting the same proposal-bank loss is insufficient to create accurate
new instances for the official hard misses. Do not continue v2/v3 with more
epochs or weight sweeps. Close this route and move to a dense
image-conditioned instance proposal representation, such as
lane-centerline/keypoint/endpoint evidence followed by instance association
and complete-lane fitting. The first gate for that new route must be a
strict full-visible-span oracle on the frozen env30 base-miss GT5 subset; no
selector or decode work should start before that oracle has real headroom.

## 2026-08-03: Dense Evidence Stage Is Not Yet a Lane Proposal

The new dense instance/keypoint path is intentionally only the first stage of
the replacement direction. It predicts centerline support, endpoints, and
image-local instance embeddings, but it does not yet produce complete lane
instances or official predictions. Dense support on GT coordinates is useful
to answer whether the image contains recoverable lane evidence; it is not an
upper bound for final TuSimple ACC until a prediction-only instance
association and complete K56 fitting step exists.

The dense path must therefore be judged in this order:

```text
1. centerline and endpoint loss are finite and decrease;
2. complete-visible-span evidence is nontrivial on official-val and train0601
   frozen env30 base-miss GT5;
3. same-image embedding separation is measurable without cross-image pushes;
4. only then implement prediction-only instance association and K56 fitting;
5. official-val decode selection precedes any single TEST run.
```

The dense YAML, loss, script, and diagnostic are default-off. The default
env30 Q12/K56 path remains the reference. Do not use the dense evidence
summary as an ACC claim, and do not tune a dense threshold on TEST. The
dense probe script must use `gcs_dense_freeze_base=1`; otherwise shared P2 and
query features can drift and the evidence result is not isolated from base
regression.

## 2026-08-06: Dense endpoint mass evidence does not yield full K56 proposals

The completed mass-balanced dense endpoint probe
`dense_instance_endpoint_mass_probe20_v3_canonical` is diagnostic-only and did
not open TEST. It kept `gcs_dense_freeze_base=true`, trained for `20/20`
epochs, and selected epoch `10` by canonical official-val:

```text
official-val ACC/FP/FN = 0.973365 / 0.015748 / 0.009642
count_acc_4/count_acc_5 = 0.969697 / 0.986486
decode = conf 0.001, point_valid_thr 0.6, nms 0, max_det 5, min_points 4
```

This is effectively the frozen env30 base path, because dense evidence is not
used by the official decoder. It is not an official ACC gain.

The dense evidence head learned useful support at GT coordinates:

```text
official-val base-miss dense_full_evidence = 324/351
official-val GT5 base-miss dense_full_evidence = 88/94
official-val GT5 visible 6..10 base-miss dense_full_evidence = 14/15
train0601 base-miss dense_full_evidence = 509/551
train0601 GT5 base-miss dense_full_evidence = 354/382
train0601 GT5 visible 6..10 base-miss dense_full_evidence = 72/78
```

However, prediction-only dense candidate assembly did not convert that support
into complete K56 proposals. The canonical official-val trace gate found:

```text
all base-miss lanes: candidate_oracle_full_hit20 = 5/32
GT4 base-miss:       candidate_oracle_full_hit20 = 3/12
GT5 base-miss:       candidate_oracle_full_hit20 = 0/15
GT5 visible 6..10 base-miss: candidate_oracle_full_hit20 = 0/13
score top1/top5 on GT5 base-miss = 0/15 and 0/15
```

The hard GT5 official-val failures are concentrated in `clips/0601` outer
lanes (`lane_index` 0 or 4). Their best trace candidates remain far outside
the 20px gate, commonly `33..267px` APE, and one visible-1 lane is
unmeasurable under the normal full-hit gate.

Row-DP and beam diagnostics on the representative hard sample
`clips/0601/1494452397586667021/20.jpg` improved partial local APE but still
failed complete visible geometry:

```text
rowdp top32/pool50000: partial APE 10.7964, full-visible APE 89.7510
beam2048:              partial APE 3.1688,  full-visible APE 44.0607
candidate_oracle_full_hit20 = 0
score top1/top5 full-hit20 = 0
```

## 2026-08-07: LineIoU improves short geometry but breaks count calibration

`query_env30_lineiou_w1_clean40_v1` completed its 40-epoch canonical
official-val probe with TEST closed. It produced a small selected official-val
ACC increase over env30 (`0.973565` vs `0.973330`) and lower FP/FN
(`0.013820/0.009412` vs `0.015748/0.009642`), but the gain is not promotable:
`count_acc_4` fell from `0.969697` to `0.939394`, `count_acc_5` fell from
`0.986486` to `0.972973`, and no row in the `6912`-row official-val sweep
passed the combined ACC/FP/FN/count4/count5 promotion gate.

The same-decode comparison preserves the diagnosis:

```text
env30 fixed decode:   ACC/FP/FN/count4 = 0.973330 / 0.015748 / 0.009642 / 0.969697
clean40 fixed decode: ACC/FP/FN/count4 = 0.973527 / 0.014601 / 0.009412 / 0.939394
```

Paired bootstrap CIs include zero for both fixed-decode and selected-decode
comparisons, so the small official-val ACC delta is not stable enough to
freeze for TEST.

The useful part of the signal is localized: the GT4-short hard set improved
from `F1=0.897959` to `0.949807` (`tp/fp/fn` `462/49/56 -> 492/26/26`).
The harmful part is official-val GT4/GT5 count calibration, especially
single-point or extremely short fourth lanes where slight valid/score changes
turn `4 -> 3`.

Implication:

Do not continue this checkpoint to long training and do not tune decode to
hide the count regression. If LineIoU is reopened, the next experiment must be
a single-variable `LineIoU + short valid/count preserve` probe with a
pre-registered gate requiring GT4/GT5 count retention, not another broad
geometry-only probe.

## 2026-08-07: Dense hard GT5 blocker is peak localization plus association

`tools/diagnose_gcs_dense_pairing_fitting.py` separates dense endpoint
support, endpoint peak localization, channel assignment, embedding
separability, forced GT-endpoint tracing, and complete K56 fitting. It is
diagnostic-only and uses GT only after inference.

Canonical official-val run:

```text
runs/gcs_lane/dense_pairing_fitting_canonical_val_v1
```

Key canonical GT5 base no-hit evidence:

```text
GT5 base no-hit20 lanes = 15
dense prediction-only full-hit20 = 0/15
forced GT-endpoint full-hit20 = 5/15
endpoint support both = 13/15
correct endpoint channel = 13/15
near endpoint peaks = 0/15
```

Train0601 hard-40 run:

```text
runs/gcs_lane/dense_pairing_fitting_train0601_hard40_v1
```

The same pattern held:

```text
GT5 base no-hit20 lanes = 16
dense prediction-only full-hit20 = 0/16
forced GT-endpoint full-hit20 = 4/16
endpoint support both = 15/16
correct endpoint channel = 16/16
near endpoint peaks = 0/16
```

Implication:

The dense endpoint mass probe learned broad endpoint evidence, but the hard
outer GT5 lanes fail because the peak selected by prediction-only decoding is
spatially displaced and because forced tracing still fails on most lanes.
Small same-lane embedding margins (`~0.008` canonical GT5 base no-hit,
`~0.016` train0601 hard GT5 base no-hit) indicate association ambiguity.

Do not spend more runs on dense threshold, maxdet, NMS, or long training.
Reopen dense only with a single-variable mechanism that directly improves
endpoint peak localization/sharpness or same-lane association, and require
prediction-only canonical GT5 base no-hit full-hit20 to improve from `0/15`
to at least `4/15` before any official-val promotion discussion.

Additional canonical GT5 no-hit ablations confirm that oracle coverage alone
is not enough. Expanding dense endpoint peaks and increasing embedding weight
created some oracle candidates but left score ranking unusable:

```text
p64/e0.35: dense_full = 0/15
p64/e1.0:  dense_full = 2/15
p128/e1.0: dense_full = 4/15
top1 full-hit = 0/15
top5 hit = 0/15
hit ranks for the four p128/e1 hits = 865, 920, 932, 965 of 1024
```

This closes pure decode-parameter promotion. The dense path would need a
separate candidate-ranking mechanism or ranking diagnostic before it can be a
credible official-val candidate.

The ranking-factor audit in
`runs/gcs_lane/dense_ranking_audit_canonical_gt5_nohit_p128e1_v1` explains the
ranking failure. On canonical GT5 base no-hit lanes, first-full candidates
have lower score factors than top1 wrong candidates across several weak
dimensions, especially length:

```text
top1 median score / first-full median score = 0.983674 / 0.846281
endpoint factor = 0.991813 / 0.973963
centerline factor = 0.993723 / 0.978699
embedding factor = 0.999072 / 0.987544
length factor = 1.000000 / 0.916667
visible count = 45 / 11
```

Because these factors are multiplicative, the current score systematically
prefers long high-confidence wrong/neighbor candidates over short outer
GT5 full-hit candidates. The immediate dense target is therefore not more
candidate generation; it is short-lane-aware candidate ranking with a gate of
canonical GT5 base no-hit top5 full-hit `0/15 -> >= 4/15` before any official
decode discussion.

Interpretation:

The endpoint mass fix addresses the earlier endpoint-target imbalance and
proves dense evidence is present, but the current trace/row-DP assembler cannot
turn that evidence into full-lane K56 geometry for the exact GT5 base-miss
lanes that limit TuSimple ACC. This is a representation/association/fitting
failure, not a reason to add epochs or tune official decode thresholds.

Decision:

Do not continue this dense assembler route to `100/220` epochs, do not run
official TEST, and do not tune `conf`, NMS, `max_det`, or `min_points` from
this result. The next dense experiment must first add a diagnostic that
separates endpoint localization, endpoint pairing, embedding clustering, and
complete K56 fitting on the 15 official-val GT5 base-miss lanes. Only if an
oracle endpoint/pairing diagnostic produces nonzero full-hit20 headroom should
a new supervised dense-to-K56 proposal head or prediction-only decoder be
implemented.

### Canonical repeat and frozen-EMA audit

The canonical official-val comparison was repeated twice per frozen checkpoint
with the selected decode fixed at `conf=0.001`, `point_valid_thr=0.6`,
`nms_dist_px=0`, `max_det=5`, and `min_points=4`:

```text
env30 repeats: ACC/score/FP/FN = 0.973330 / 0.972822 / 0.015748 / 0.009642
dense repeats: ACC/score/FP/FN = 0.973365 / 0.972857 / 0.015748 / 0.009642
count confusion: identical
```

The `+0.000035` dense-checkpoint delta is reproducible but is not dense
evidence being decoded. A checkpoint-state audit found that all shared FP16
and integer tensors match the env30 checkpoint exactly; the only differences
are all `264` shared FP32 normalization tensors. Applying the current
`ModelEMA` recurrence to the unchanged env30 FP32 state for exactly `1020`
optimizer updates (the epoch-10 checkpoint) reproduces every one of those
candidate FP32 tensors exactly. The observed official-val delta is therefore
an EMA numerical drift of a frozen base state, not a promotable dense-model
gain.

Any future frozen-base diagnostic must pin non-trainable state in both the
live model and `ModelEMA.ema`, assert that its optimizer contains only the
declared trainable branch, and record before/after hashes for all frozen
parameters and buffers. Until that protocol exists, do not use a frozen-head
checkpoint's base-only official metric to claim an algorithm improvement.

## 2026-08-06: Visibility-only probe does not improve official accuracy

`query_env30_visibility_only_probe20_v2` trained only the point-valid heads
for `20/20` epochs. Under the fixed post-hoc decode
`conf=0.001`, `point_valid_thr=0.6`, `nms=0`, `max_det=5`,
`min_points=4`, `valid_before_maxdet=false`, it fell below env30:

```text
ACC/score/FP/FN: 0.973330/0.972822/0.015748/0.009642
             -> 0.972924/0.972402/0.015060/0.011019
count_acc/count_acc_4: 0.972452/0.969697 -> 0.969697/0.939394
```

The point-valid loss decreased, but raw geometry and candidate capacity
remained unchanged. Lane-level recall at the official `0.6` threshold fell
from `0.979776` to `0.978830` overall, from `0.977048` to `0.974974` on GT5,
and from `0.957233` to `0.951572` on 53 GT5-short lanes. This is a
loss/metric-contract mismatch: the head calibrates masks but does not create
new lane geometry, while `min_points=4` can remove the only query that earns
official credit for a one-point GT lane.

The route is closed for promotion. A corrected frozen-state implementation
is protocol hygiene only; it is not evidence that longer visibility training
will improve TuSimple TEST.

## 2026-08-08: Current remaining bottleneck after LineIoU w0.5 clean40

The `query_env30_lineiou_w05_clean40_v1` probe gives the first clean
official-val gain over env30 among the reopened LineIoU variants:

```text
ACC +0.000402
FP  -0.002617
FN  -0.000230
count_acc_4 +0.015151
raw has20 +0.006140
```

Confirmed bottlenecks that remain:

- **Short GT4/GT5 raw geometry is still weak.** GT4-short has20 improved
  from `0.1250` to `0.3750`, but remains low; GT5-short improved from
  `0.7547` to `0.8491`, still far below non-short GT5 `0.9968`.
- **One- or two-point GT lanes are not reliably modeled by Q12/K56 query
  output.** Two final undercount images have missing GT lanes with one visible
  official point and raw APE `inf`, so no query candidate represents them.
- **Some residual FN are true geometry misses, not decode filtering.** The
  remaining canonical missing rows include long lanes with raw APE above
  20 px, e.g. visible-point counts `46`, `43`, and `25`.
- **Existence calibration trades recall for precision.** converted-val shows
  active bad queries decreased (`146 -> 117`) but inactive good geometry
  increased (`165 -> 381`), so w0.5 creates more geometric candidates than the
  existence head activates.
- **Official-val gain is small.** The run is promotable only to a 100-epoch
  probe, not to TEST or a 220-epoch formal run, until the gain survives longer
  training and the non-test diagnostics remain stable.

Current rejected directions remain rejected: dense endpoint/assembler decode,
dense p128/e1 decode-parameter promotion, visibility-only training, TEST-side
threshold tuning, and arbitrary NMS/maxdet/min_points search.

## 2026-08-10 Residual Set Selection Bottleneck

Residual geometry is no longer the only bottleneck. Horizontal-flip
consistency retrieves every known Stage-1b strict oracle candidate by top5 on
canonical, clean, and train0601, and improves ordinary train0601 GT5 top3
retrieval from `67/1194` to `76/1194`.

The remaining failure is prediction-only victim identity and set utility.
Top3 set oracle gains are `+15` canonical, `+15` clean, and `+43` train0601,
with FP proxy reductions, so replacement headroom is real. Learned selectors
recover only `0..+2` safe hits per split and remain far below oracle.

Selector v2 with set-context features and ternary ranking eliminated hit
harms but selected mostly add actions, increasing count error on all three
evaluation splits. A count-safe target then selected replacements but caused
`-3` canonical and `-9` train0601 hit regressions at the calibration-selected
threshold. Higher thresholds remove those harms only by collapsing canonical
and train0601 gain to zero.

Do not continue threshold tuning or connect the selector to decode. The next
root-cause experiment must learn a model-internal base-query
unique-contribution/victim-identity representation, while keeping env30 and
the validated residual proposal head frozen.

## 2026-08-10 Final Residual Selector Limitation

Training utility directly from official per-image metrics fixed the hit20 vs
official-ACC target mismatch. The four-selector ensemble improves both
canonical official-val and clean-val while reducing FP/FN, but it activates
only a few images:

```text
canonical: 2 actions, ACC 0.973346 -> 0.973494
clean:     3 actions, ACC 0.965470 -> 0.965692
```

The corrected final TEST reaches `0.966646`, which rounds to approximately
`0.97`, but the selector changes only one of 2782 images. Reconstructed frozen
base TEST ACC is `0.96664279`; selector gain is only `+0.00000321`.

The remaining dominant TEST bottleneck is not residual replacement coverage.
Count accuracy is weak for GT4/GT5 (`0.602564` and `0.885764`), with substantial
`4->3`, `4->5`, `5->3`, and `5->4` confusion. Future work must target the base
model's cross-session count/instance survival using train/official-val only;
the completed TEST must not be used for further tuning.

## 2026-08-11 Frozen query-survival probes expose a distribution-coverage bottleneck

Strictly frozen env30 query-survival probes changed existence logits without
changing `pred_points` or `pred_valid_logits`, but did not change clean-val
official metrics. Standard quality-aware BCE, targeted duplicate-negative BCE,
and weakest-matched versus strongest-unmatched pairwise ranking all produced
the same clean-val result:

```text
ACC/score/FP/FN = 0.966146 / 0.965061 / 0.034527 / 0.019743
count_acc_4/5 = 0.958333 / 1.000000
```

The pairwise probe is especially diagnostic: on GT4/GT5 training images, the
average accurate-matched minus strongest-unmatched base-logit margin is already
approximately `13`. Therefore the training distribution already satisfies the
generic survival ordering that the new loss requests.

The remaining failure is cross-session representation and hard-example
coverage for short/outer GT4/GT5 lanes. Do not spend more runs on existence
loss gain, spurious-negative group weights, pairwise margin, threshold, maxdet,
or count-aware top-k. A credible next route must target train0313/train0601
hard-session failures explicitly and transfer to clean-val before TEST remains
eligible for discussion.

## 2026-08-11 Residual listwise selectors expose the current structural bottleneck

The residual proposal path no longer lacks a better binary threshold. Three
different action formulations fail in complementary ways:

```text
flat listwise:          all no-op
positive-weighted flat: frequent harmful replacements
hierarchical listwise:  early over-trigger and FP regression
```

This is a sparse-action and representation problem, not a calibration knob.
Only `11/363` canonical and clean images and `21/410` train0601 images contain
a positive replacement action, while most visually plausible replacements
have zero or negative official utility. Presence and victim identity are not
separable enough in the current residual features to generalize across
sessions.

More importantly, the residual replacement oracle ceiling is too small:

```text
canonical +0.000557
clean     +0.000859
train0601 +0.000308
```

The remaining root problem is in the base query set. Hard short/outer GT4 and
GT5 lanes are often absent geometrically from the raw Q12 pool; when geometry
training does recover them, shared query features and coordinate-conditioned
point-valid refinement can disturb existence, visibility, and top-5 count
survival. The current Hungarian per-query losses do not directly optimize the
decoded set property that every geometrically correct matched lane must outrank
harmful unmatched queries while remaining point-valid.

Next evidence gate: calculate a train/official-val-only matched-query
set-survival oracle, split by GT4 `4->3`, GT4 `4->5`, and GT5 `5->4`. If the
ceiling is insufficient, return to short-lane geometry with explicit
score/valid preservation instead of adding another selector. TEST stays closed
for all selection and tuning.
