# Known Bottlenecks

## 2026-07-21 Env30 GT4/GT5 Staticref v2 Closure

The `query_alpha05_env30_gt45staticref_v2` full protocol is complete and
rejected. It is not promotable and should not be threshold-tuned from TEST.

Primary comparison:

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

The post-train fine sweeps for both `official_best.pt` and `best.pt` had zero
rows reaching env30 official-val ACC, zero rows with FP/FN both at least as
good as env30, and zero rows passing the GT4/GT5 count-shape gate. Training-time
official_best selected epoch 213, so this is not an early-checkpoint issue.

What the run proved:

- The v2 static bank solved the old `gt5staticref_v1` six-lane extra explosion
  (`val 5->6: 10 -> 0`, `test 5->6: 71 -> 0`).
- It partly improved reporting-only TEST GT4 retention
  (`4->3: 98 -> 81`, `count_acc_4: 0.598291 -> 0.649573`).
- It did not improve the real target because GT3 over-count and GT5 under-count
  rose (`test 3->4: 50 -> 64`, `test 5->4: 43 -> 69`,
  `count_acc_5: 0.891037 -> 0.857645`).

Post-hoc official-val raw-Q12 diagnostics show that v2 did not improve net raw
geometry:

```text
overall has_match_20px: 0.975441 -> 0.972371
overall mean APE px:    5.627210 -> 6.053190
GT4 all has_match_20px: 0.954545 -> 0.950758
GT5 all has_match_20px: 0.962162 -> 0.956757
GT5 short has_match_20px: 0.754717 -> 0.773585
GT5 short point_valid_recall@0.6: 0.957233 -> 0.943396
```

Post-hoc train-side raw-Q12 diagnostics confirm the same split instability:

```text
train0601 overall has_match_20px: 0.960828 -> 0.950196
train0601 overall mean APE px:    6.609740 -> 7.269312
train0601 final under-count images: 5 -> 8

train0601 GT4 short has_match_20px: 0.473684 -> 0.578947
train0601 GT5 short has_match_20px: 0.775956 -> 0.737705
train0601 GT5 short point_valid_recall@0.6: 0.937601 -> 0.908242

train0531 overall has_match_20px: 0.983900 -> 0.980322
train0531 overall mean APE px:    5.177374 -> 5.694983
train0531 final over-count images: 7 -> 12
```

Train-side count shape also moves in opposite directions:

```text
train0601:
  env30: 3->4=1, 4->5=7, 5->4=3
  v2:    3->4=3, 4->5=4, 5->4=6

train0531:
  env30: 3->4=4, 4->5=2
  v2:    3->4=6, 4->5=6
```

The edited q5/q6/q7 references become major GT5-short carriers on train0601
(`3/183 -> 56/183`), but 20px misses also rise (`41/183 -> 48/183`). The v2
bank therefore did not simply add missing short-lane coverage; it changed the
query carrier assignment and made GT5-short survival less stable.

Current bottleneck:

The candidate solved the old boundary-extrapolated extra-carrier failure, but
it did not solve the active env30 bottleneck: improving short GT4/GT5 coverage
without suppressing true GT5 lanes or creating GT3/GT4 extras. The observed
tradeoff is specifically "GT4 count retention up, GT5 true-lane survival down,
GT3 false-fourth up."

Smallest safe next action:

Do not run another blind train. First run official-val plus
train0601/train0531 raw-Q12 diagnostics comparing env30 and v2. Only if those
diagnostics show that GT5 damage comes from broad valid-negative pressure should
the next ablation be narrowed to GT4-only boundary-pseudo valid negatives
(`gcs_boundary_pseudo_gt_count=4`,
`gcs_boundary_pseudo_max_gt_count=4`) or a lower
`gcs_boundary_pseudo_valid_neg_weight` (`0.1` or `0.0`). Any follow-up must pass
official-val ACC/FP/FN, GT4/GT5 count shape, raw match/point-valid survival, and
train0601/train0531 robustness together before any reporting-only TEST.

## 2026-07-17 Env30 Short GT4/GT5 Gate Execution

The user-requested env30 short GT4/GT5 robustness-gate experiment is complete
and rejected before training. TEST was not used, remote training was not
launched, and no decode/loss/sampler/reference/checkpoint setting was selected.

Artifacts:

```text
.tmp/env30_scheme_execution_20260717/gate_analysis.md
.tmp/env30_scheme_execution_20260717/gate_analysis.json
.tmp/env30_scheme_execution_20260717/summary.md
```

Result:

```text
static candidates evaluated = 6226
gate-pass candidates = 0
allowed replacement queries = q2,q5,q6,q7,q8
changed known-extra overlap in saved top rows = empty
```

Best static row:

```text
design = gt5cl1_gt5r1_gt4c2_gt4r1
val GT5 impact p50/p90/match20 = 43.579 / 47.756 / 0.2500
val GT5 primary p50/p90/match20 = 51.595 / 97.903 / 0.0625
train0601 GT5 primary p50/p90/match20 = 49.012 / 92.705 / 0.0833
val GT4 primary p50/p90/match20 = 44.210 / 78.183 / 0.1250
```

The result isolates the failure: the safe five-query space avoids known
extra-carrier overlap but cannot meet GT5 primary/impact raw-geometry coverage
requirements. `val_gt5_primary_match20`, `val_gt5_impact_match20`, and
train0601 GT5 primary p90/match20 all stay below gate. This line remains
diagnostic-only. Do not train or run TEST from it.

## 2026-07-17 Active Rollback Boundary

Active source/config is rolled back to commit `86c8fb31c` (`Add GT4 GT5 weak
geometry rescue run`). Bottleneck notes for experiments after that commit are
retained only as rejected legacy evidence. In particular, tiered GT4/GT5
rescue, Q12 ultrashort dataref/reference banks, final-query extra guard/gate,
router/selector diagnostics, and related post-86 scripts/YAMLs/tools are not
active code after the rollback and must not be used as promotion candidates or
TEST-tuning paths unless a future task explicitly reopens them.

## 2026-07-17 Count-Change Selector Closure

The count-change oracle and non-oracle selector experiment is closed before
TEST. Across `env30`, GT4/GT5 weak, and tiered-v2 outputs, canonical
official-val has no GT4/GT5 count-fix headroom: the `gt45_count_fix` oracle
switches zero images and keeps ACC at `0.973330`. The broader all-count oracle
switches only two GT3 over-count images and improves official-val by only
`+0.000055`.

Train-side GT4/GT5 repair opportunities exist but are too sparse:
train0601 has three target fixes and train0531 has two. The risk pool is much
larger (`42` count-risk candidate rows on train0601 and `16` on val), including
GT4 false-fifth and GT5 false-sixth/under-count cases. The prediction-only
selector scan over base and aux features produced `3840` rows, with `96`
passing train gates, but zero rows passing both train-side and canonical
official-val gates with a non-empty reporting candidate.

Conclusion: do not continue by tuning count-risk thresholds, aux feature
weights, classifier type/depth, or switch caps on this three-model pool. A
useful count-changing router first needs a new candidate output that actually
creates materially more GT4/GT5 repair opportunities on official-val/train
without adding count-risk.

## 2026-07-17 High-Confidence Abstention Switch Bottleneck

The high-confidence three-model abstention switch reopens the whole-image
router line only as a weak reporting candidate, not as a solved path. The best
gate-passing row defaults to `env30` and switches only `7 / 363` canonical
official-val images to GT4/GT5 weak or tiered-v2. Its official-val ACC improves
from `0.973330` to `0.973551` (`+0.000221`) with unchanged FP/FN and unchanged
GT4/GT5 count shape.

This proves that a conservative non-oracle router can avoid the previous
router's count-shape collapse, but the useful coverage is too small. The
passing switches are count-preserving line-accuracy micro-fixes; they do not
address the dominant TEST gap, which remains GT4/GT5 count generalization and
FP/FN robustness. Higher-val aux rows are not promotable because they fail the
train-side leave-date-out gate and reintroduce GT4 false-fifth / GT5
false-sixth risk.

Keep `env30` as the active reference. The frozen abstention row is eligible
only for one reporting-only TEST if explicitly used; do not tune thresholds,
caps, aux features, classifiers, or model selection from TEST.

## 2026-07-16 Fixed Geometry-Consensus Closure

The three-checkpoint `env30`/GT4-GT5-weak/tiered-v2 geometry-consensus scan is
closed before TEST. Across 100 non-baseline official-val/train-only rules, no
rule improves canonical official-val ACC above `env30=0.973330`.

The best median-count rule removes two official-val GT3 extra lanes and lowers
FP from `0.015748` to `0.014371`, but ACC remains exactly `0.973330`. Geometry
median fusion is harmful: its best official-val ACC is `0.970923`, a
`-0.002407` regression, with 128 worsened images versus 71 improved images.

The GT-only whole-model choice oracle reaches `0.977181` official-val ACC, so
the models contain complementary predictions. The unresolved bottleneck is a
non-oracle routing signal, not lack of geometric diversity. Do not run TEST,
average lane coordinates, or promote median model count. A future reopen needs
a small train-only router with leave-date-out evidence on train0601/train0531
and a canonical official-val gain with no FP/FN or count-shape regression.


This file applies to branch `codex/5-25-3-k56`.

## Branch Scope

The current mainline imports the historical `5-25-3.zip` algorithm and changes the TuSimple fixed-y contract to Q=12/K=56 with official h-sample anchors. It also includes the 2026-06-27 user-requested default-off `count_boundary_loss` for GT3/GT4/GT5 adjacent count-score boundaries, default-off train-only `gcs_hard_sampling` for 0601 and short-visible GT3/GT4/GT5 samples, default-off E3-lite `gcs_spurious_neg` loss for short unmatched duplicate-like queries, training-time `official_best`, the default-off `valid_before_maxdet` query decode option, default-off query Count Head, default-off `gcs_short_geom`, GT5 short point-valid rescue, `gcs_boundary_pseudo_neg`, cached official sweeps, and the env30-family scripts present at `86c8fb31c`.

Do not read mainline Count Head, Quality Head, Survival Head, near-miss, or old mainline official-best bottlenecks as active branch behavior. Those algorithm mechanisms are not part of this 5-25-3 branch. The only active Count Boundary behavior is the branch-local default-off `count_boundary_loss`, the only active hard sampler is the branch-local default-off train-only `gcs_hard_sampling`, the only active E3-lite spurious negative behavior is the branch-local default-off `gcs_spurious_neg` family described in `current-contracts.md`, and the only active official-best behavior is the explicit 2026-06-27 training-time official-val selection hook.

Active source/config is rolled back to commit `86c8fb31c` (`Add GT4 GT5 weak
geometry rescue run`). Bottleneck notes below that depend on post-`86c8fb31c`
mechanisms such as tiered rescue, Q12 ultrashort dataref/reference banks,
final-query extra guard/gate, router/selector diagnostics, Q18/Q20/dataref
configs, count-guided decode, side-aux, GT4-hard diagnostics,
`tools/diagnose_tusimple_count_confusion.py`,
`tools/diagnose_gcs_count_contract.py`, `--gcs-gt4-short-*`,
`extra_exist_loss`, or `--gcs-short-exist-*` are rejected legacy experiment
conclusions only. They do not describe currently available code, CLI flags,
loss terms, diagnostic scripts, configs, model outputs, or active selected
candidates.

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
- It does not include post-`86c8fb31c` tiered rescue, Q12 ultrashort dataref/reference banks, final-query extra guard/gate, router/selector diagnostics, later mainline `diagnose_gcs_gt5.py`, Count/Quality/Boundary diagnostics, Survival, or near-miss machinery.

## 2026-07-16 Current 0.9700 Goal Status

The current 0.9700 official-TEST objective remains active and incomplete. The
latest integrated status report includes the phase acceptance checklist,
candidate map, and current reopen/promotion gate:

```text
.tmp/goal_9700_audit/current_goal_status_20260716.md
.tmp/goal_9700_audit/remote_official_summary_index_audit.md
.tmp/goal_9700_audit/official_val_gate_family_audit.md
.tmp/goal_9700_audit/non_oracle_final_lane_rule_scan.md
.tmp/goal_9700_audit/non_oracle_count_counterfactual.md
.tmp/goal_9700_audit/non_oracle_count_counterfactual_split_stability.md
.tmp/goal_9700_audit/premax_cache_margin_diagnostic.md
.tmp/goal_9700_audit/pred_shape_separator_audit.md
.tmp/goal_9700_audit/pred_layout_separator_audit.md
.tmp/goal_9700_audit/gtquery_context_separator_audit.md
.tmp/goal_9700_audit/label_context_separator_audit.md
.tmp/goal_9700_audit/selection_surface_reopen_audit.md
.tmp/goal_9700_audit/train_side_robustness_gate_audit.md
.tmp/env30_train_count_confusion_decomp/summary.md
.tmp/env30_official_like_residual_decomp/summary.md
.tmp/env30_residual_prediction_separator/summary.md
.tmp/env30_raw_gt_lane_context_gate/summary.md
.tmp/goal_9700_audit/score_margin_target_audit.md
.tmp/goal_9700_audit/tiered_v2_tail_reopen_audit.md
.tmp/goal_9700_audit/positive_side_reopen_audit.md
.tmp/goal_9700_audit/remote_state_recheck_20260716.md
.tmp/env30_distribution_stability_audit/summary.md
.tmp/env30_non_oracle_selector_table/summary.md
.tmp/env30_short_gt45_reference_counterfactual/summary.md
.tmp/env30_image_appearance_separator/summary.md
.tmp/env30_session_context_separator/summary.md
.tmp/env30_session_context_overlap/summary.md
.tmp/env30_cross_feature_separator/summary.md
.tmp/env30_raw_query_cross_feature_gate/summary.md
.tmp/exist_quality_alpha/
```

Summary:

```text
active reference = env30
env30 official-val ACC/FP/FN =
  0.973330 / 0.015748 / 0.009642
env30 official-val count_acc_3/4/5 =
  0.968610 / 0.969697 / 0.986486
env30 reporting-only TEST ACC/FP/FN =
  0.966780 / 0.028732 / 0.023544
goal status = not complete
```

The reporting-only TEST result is below the required `official_test_acc >=
0.9700` and cannot be used for tuning. Current official-val/train diagnostics
close the completed dataref family (v1 manual1, geom1 full-protocol v1, v2,
and q1/q3-protected), count-restore, tiered rescue, hard-negative,
positive-rescue, q3-disambiguation, and q2/static reference-bank directions as
trainable candidates. A read-only gate scan found no high-value unclosed
diagnostic direction that should be continued from the same evidence surface.
The companion remote index scan confirms that some old/legacy rows exceed
env30 on official-val, but no reporting-only TEST row reaches `0.9700`; those
high-val rows stay legacy/rejected and do not reopen candidate selection.
A live read-only remote recheck on 2026-07-16 parsed the same `939` official
summary rows. The latest remote summary remained the rejected q1/q3-protected
epoch015 sweep (`0.949375 / 0.113682 / 0.047980`), the official-val gate-like
row count stayed `4`, and the best reporting-only TEST row stayed
`0.966962 / 0.030188 / 0.023994`, below the required `0.9700`. It found no new
remote evidence surface to open.
The official-val-only gate-family audit narrows this further: `45 / 880`
official-val rows reach env30 ACC, but only four also pass the current
FP/FN/count-shape gate; two are env30 and the two non-env30 rows are lower-ACC
legacy farspur epoch rows, not the protocol-selected official-best/external
farspur selection.
The non-oracle final-lane/count counterfactual also closes a tempting decode
shortcut: the best count-only rule is `rank>=5`, equivalent to removing the
sixth score-ordered lane under `max_det=6`. It reduces v2 over-count in copied
official-val/train0601 traces, but removes true matched lanes as well as
extras and maps to a `max_det=5`-style restriction already rejected by the v2
official-val sweep. Low-score/short-valid rules catch more extras but create
under-count.
The split-stability extension confirms `rank>=5` is a v2 six-lane cap artifact:
it does not act on env30 under the selected five-lane cap, and on v2 it removes
true GT5 matched lanes (`6` on official-val, `20` on train0601), so it is not a
matched-vs-extra separator.
The fresh `official_best.pt` pre-maxdet cache diagnostic closes the remaining
cache-margin shortcut on both official-val and train-side traces. On
official-val, env30 has `1307` post-valid premax rows and `1307` final-kept
rows with `0` `rank_eq5`, `rank_ge5`, or beyond-`max_det` candidates, so its
selected `max_det=5` row is not hiding an extra post-valid candidate pool. V2
has `14` sixth-lane rows on the fresh official-val cache (`13` on GT5 images,
`8` nearest-GT far, `2` close20, `6` pool-greedy matched). On train0601,
env30 already has `5` `rank>=5` rows, and v2 has `58` `rank>=5` rows with `22`
pool-greedy matched rows; on train0531, v2 has only `1` far `rank>=5` row. V2
fresh train caches differ from older copied traces on `7` train0601 images and
`1` train0531 image, so these are margin diagnostics only. Low-score/short-
valid rows still include matched/final-kept candidates. This does not reopen
decode caps, score/valid filters, hard-negative losses, or reference-bank
edits.
The selection-surface reopen audit also closes the tempting lower-epoch
checkpoint shortcut: the only non-env30 gate-passing official-val rows are
legacy farspur intermediate epochs, the selected farspur reporting-only TEST
rows remain below target, and reselecting them would require both a legacy
mechanism reopen and a checkpoint-policy change.
The train-side robustness gate audit makes the split-stability requirement
explicit for future candidates. Env30 official-val extras are GT3-only
(`3->4=7`, q0/q11), while env30 train0601 already has GT4 false-fifth extras
(`4->5=7`, q1/q10/q11). A future candidate must therefore preserve both
official-val count shape and train0601/train0531 count-shape robustness before
any reporting-only TEST is allowed.
The env30 train-side count-confusion decomposition confirms this is not just a
summary-table artifact. Across env30 official-val/train0601/train0531 it reads
`1131` images and `13572` query rows, with only `23` final-extra rows and `7`
under-image missed Hungarian-positive rows. Official-val extras are all-far
`3->4` q0/q11 rows, train0601 extras are all-far and dominated by GT4
`4->5` q1/q10/q11 rows, and train0531 extras are spread across
q0/q1/q9/q10/q11 with mixed GT2/GT3/GT4 over-count. The under-count side is
sparse GT4 `4->3` matched-positive survival on q5/q6/q10/q11 rather than a
stable count-only target. This does not reopen sampler, loss, decode, or
reference changes.
The env30 official-like residual decomposition adds image-level and lane-level
official-like residual attribution on the same official-val/train0601/
train0531 surfaces. It reads `1131` images and `655` lane/failure rows, with
`23` final-extra rows, `70` count-correct official-like residual images, and
`181` GT-side failure-like lanes on residual images. The result is still
diagnostic-only: final extras are sparse, and the largest broad signal
(`low_final_line_acc<0.95`) is noisy because it also appears in non-residual
count-correct background rows (`96` val, `169` train0601, `66` train0531).
This does not provide a non-oracle, train-available, split-stable selector or
target-margin artifact, and it does not reopen hard-negative, positive-rescue,
geometry/validity loss, sampler, decode/NMS/`max_det`, or reference changes.
The env30 prediction-only residual separator audit closes the direct image-
level confidence/separator variant of the same line. It uses full 12-query
prediction-cache features and final decoded lane prediction features from
official-val/train0601/train0531, while explicitly excluding `image_official_*`,
GT counts, final-extra labels, GT-lane failure counts, missed-Hungarian counts,
and target labels after an initial leakage check. Across `1131` images, the
best broad official-like residual rule with at least ten positives reaches only
`precision=0.765` and `recall=0.131`; the best over-image rule with at least
five positives reaches `0.556 / 0.227`; and under-image rules either select a
tiny tail or have low precision. This does not justify training, TEST, decode
tuning, loss changes, sampling changes, or reference edits.
The env30 raw-GT lane context gate closes the simple train-available lane-
context positive-rescue variant. It reads `4208` GT-lane rows from official-val,
train0601, and train0531 raw diagnostics. Ultra-short lanes (`visible_le5`)
are real raw/decoded bad carriers (`precision=0.818`) but select only `22`
lanes, are absent from train0531, and map poorly to official-like residual
lane failures (`0.364 / 0.170`) and count-correct residual lane failures
(`0.273 / 0.200`). Broader visible<=10/20, side, GT4, and GT5 context rules
cover more failures only by selecting hundreds of already-ok lanes. This does
not justify positive rescue, anchor weighting, geometry/validity loss changes,
sampler changes, or reference edits.
The score-margin target audit closes direct score-margin/ranking as a current
implementation direction. V2 train0601 has a real GT5 `5->6` low-margin
pathology, but env30 count-correct GT5 `5->5` rows also include low/negative
far margins, and v2 train0601 GT4 `4->5` is not a low-margin class.
The tiered-v2 tail-reopen audit consolidates the same conclusion for direct
tail-retention reuse: six raw/valid tail fixes are real, but only one has the
correct final count, and train0601 q0/q10 traces mix harmful extras with
protected, near, or Hungarian-matched rows.
The prediction-shape separator audit closes the simple score/valid/raw-shape
path. It adds `pred_points`/`pred_logits`/`pred_valid_logits` features to the
same official-val/train final-kept rows and treats far-unmatched rows only as
an offline harm proxy. The best sparse low-valid rules still hit safe
close/matched lanes and have very low recall, while broader score/point-valid
rules select too many safe rows. This does not reopen score/valid filters,
hard-negative losses, query suppression, or reference-bank edits.
The prediction-layout separator audit closes the simple relative-lane geometry
path. It tests edge/interior position, bottom spacing, layout width, nearest
predicted-lane overlap distance, and low-valid composites. The best sparse
layout composite still selects safe rows and has low recall, while broad
layout rules have extremely low precision. This does not reopen layout-based
decode filters, hard-negative losses, query suppression, or reference-bank
edits.
The GT-count/query context separator audit closes simple training-time
GT-count/query-conditioned hard-negative logic. GT-count/query context focuses
some proxy errors better than pure prediction features, but query-wide pressure
would hit many safe close/matched lanes, and the best low-valid composites are
too sparse or still touch safe rows.
The label-context separator audit closes simple label-only training context as
a sampler or targeted-loss trigger. It uses official-val/train-side model-run
rows only and no TEST; `harm`/`safe` are offline proxy labels rather than
direct official FP/FN. Image-level GT count/visibility/short-lane/date rules
are too broad, while row-level label/query/valid composites either still hit
safe or ambiguous rows or cover too few proxy errors. Sparse zero-safe rules
exist but select at most three harmed rows and do not satisfy the split-stable
`>=90%` confidence gate.
The positive-side reopen audit closes direct positive-rescue, assignment-aware
suppression, and hardset-driven reference/query changes from the current
env30 official-val/train artifacts. Broad train-available positive selectors
cover some official-impact tails but also touch non-impact or already-positive
lanes; compact selectors depend on post-decode final matching; suppression
rules touch non-extra final lanes, GT5 `5->5` rows, or hundreds of positives;
and the short GT4/GT5 hardset denominator lacks a concrete split-stable
reference/query counterfactual that improves raw coverage without broadening
known extra carriers.
The image-appearance separator audit closes simple visual-domain rules as a
trainable trigger. It covers all `1131` env30 official-val/train0601/train0531
trace images using pixel-level brightness, saturation, edge, blur, and regional
contrast features only. The best `over_image` precision is `0.052632`, best
`count_mismatch` precision is `0.061404`, best `official_error` precision is
`0.096491`, and best unique official-val tail-image precision is `0.135135`.
These rules mostly select count-correct/low-risk images and do not justify a
sampler, augmentation, loss, decode, or reference change.
The session-context separator audit closes simple date/session/window metadata
rules as a trainable trigger. The strongest `low_acc_lt095` local window is
only `5 / 10` positives on train0531 (`precision=0.500000`,
`recall=0.080645`), the strongest `official_error` local window is only
`4 / 10` positives on val 0313-2 (`precision=0.400000`,
`recall=0.062500`), and the best over/count/under windows are weaker. These
windows are single-split local clusters, not stable official-val/train0601/
train0531 selectors, and do not justify a sampler, augmentation, loss, decode,
or reference change. The follow-up overlap audit confirms this is not hiding a
clean non-metadata mechanism: the top low-accuracy window has only `1`
official-impact tail image and `2 / 76` prediction harm/safe rows, while the
best tail-overlap val 0313-2 window reaches only `3` tail images and still has
`8 / 68` prediction harm/safe rows.
The cross-feature composite separator audit closes the current attempt to
combine prediction-shape/layout/query features with label context as a direct
training trigger. Its strongest rule, `valid_mean <= 0.2 AND
label_min_visible >= 15`, selects `23` final-kept proxy-harm rows with
`0` safe/ambiguous hits across six official-val/train surfaces, but those
rows are only `16` unique raw files, all are post-decode far-extra attribution
rows (`pool_greedy_matched_gt=false`, `nearest_gt_far50=true`), and the
selected transitions are only `3->4`, `4->4`, `4->5`, and `4->6`. It covers no
GT5 `5->4` or `5->6` risk. Because the rule depends on final-kept rows,
label-only context, and GT-derived `harm`/`safe` labels, it is a diagnostic
proxy rather than a train-ready non-oracle selector. Do not start training,
run TEST, tune decode/NMS/thresholds/`max_det`, adjust losses, change
sampling, or edit reference banks from this evidence.
The follow-up raw-query gate tests this before decode on env30 full 12-query
official-val/train traces (`13572` rows, `1131` images) and replays the exact
`valid_mean` condition from cached `pred_valid_logits`. It confirms the
final-kept composite does not become a trainable selector:
`valid_mean_exact <= 0.2 AND label_min_visible >= 15` selects `5373` raw-query
rows but only `9` final-far extras, with `5360` non-final dead queries and
`3` Hungarian positives. The best precision relative,
`query_id == 1 AND label_min_visible >= 15`, selects `609` rows but only `4`
final-far extras. Both surfaces are far below the `>=90%` confidence gate and
provide no GT5 `5->4`/`5->6` coverage.

Current rule:

- Do not train, run TEST, tune decode thresholds/NMS/`max_det`, adjust loss
  gains, or edit reference banks from the current evidence.
- Reopen a direction only with a new non-oracle separator or explicit
  train-side target-margin artifact that can prove it does not mostly select
  already-matched/near-ok positives and can protect GT3/GT4 over-count plus
  GT5 `5->6`/`5->4` gates.

## 2026-07-16 Five-Slot q2/q5/q6/q7/q8 Static Search Closure

The stricter five-slot q2/q5/q6/q7/q8 reference-bank counterfactual is closed
as diagnostic-only. It was designed to avoid q4/q9 and the high-risk
q0/q1/q3/q10/q11 set, using only train-diagnostic prototypes after excluding
official-val raw files. TEST was not used, no model was trained, and no decode
threshold was selected.

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

Static gate result:

```text
evaluated five-slot candidates = 6226
gate-pass candidates = 0

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

Interpretation:

- The no-known-extra-overlap constraint is cleaner but too capacity-limited:
  it cannot recover the q9-assisted GT5 primary-hard p90 or match20.
- The best row improves val GT5 impact-only APE p50/p90, but does not improve
  impact-only `match20` beyond the default `0.2500`.
- GT5 primary-hard and train0601 GT5 primary-hard match20 stay far below the
  continuation gate, so the result is not a training candidate.
- This reinforces the earlier q2 protected-query conclusion: q9-assisted
  designs are the strongest static direction but still not trainable, q4
  remains a direct GT5 over-count risk, and the q2/q5/q6/q7/q8-only line does
  not provide enough coverage.

Supported next action:

Do not train, run TEST, tune decode, change thresholds/NMS/`max_det`, alter
loss gains, or commit these scratch banks as candidates. A future reference
line needs a materially different non-oracle selector or train-only static
design before reopening this path.

## 2026-07-16 Cross-Run Bottleneck Matrix

The latest official-val-only synthesis keeps env30 as the reference and rejects
the currently completed dataref, count-restore, and tiered rescue families as
promotion candidates. The local working matrix is recorded at
`.tmp/cross_run_bottleneck_matrix/matrix.md`; it is diagnostic-only and does
not use TEST for selection.

```text
env30 reference:
  ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
  count_acc_3/4/5 = 0.968610 / 0.969697 / 0.986486
  GT4 4->3/4->5 = 2 / 0
  GT5 5->4/5->6 = 1 / 0
  final-query over/under = 7 / 3
  final-query extra_topk/far/duplicate = 7 / 7 / 0

ultrashort dataref v2:
  ACC/FP/FN = 0.972040 / 0.023691 / 0.011938
  delta vs env30 = -0.001290 ACC, +0.007943 FP, +0.002296 FN
  GT4 4->3/4->5/4->6 = 1 / 2 / 1
  GT5 5->4/5->6 = 2 / 12
  final-query over/under = 22 / 3
  final-query extra_topk/far/duplicate = 23 / 13 / 0

q1/q3-protected dataref:
  ACC/FP/FN = 0.949375 / 0.113682 / 0.047980
  delta vs env30 = -0.023955 ACC, +0.097934 FP, +0.038338 FN
  GT3 3->4/3->5 = 49 / 12
  GT4 4->5 = 21
  GT5 5->4 = 3
  final-query over/under = 82 / 3
  final-query extra_topk/far/duplicate = 94 / 84 / 3

count-restore env30:
  ACC/FP/FN = 0.944877 / 0.096740 / 0.063131
  delta vs env30 = -0.028453 ACC, +0.080992 FP, +0.053489 FN
  GT3 3->4/3->5 = 28 / 2
  GT4 4->3/4->5 = 1 / 10
  GT5 5->4 = 6

tiered GT4/GT5 + GT4PV v2:
  ACC/FP/FN = 0.973049 / 0.014141 / 0.008724
  delta vs env30 = -0.000281 ACC, -0.001607 FP, -0.000918 FN
  GT3 3->4 = 4
  GT4 4->3/4->5 = 2 / 2
  GT5 5->4/5->6 = 1 / 5
  reporting-only TEST was worse than env30
```

Supported bottleneck synthesis:

- Env30 remains the strongest current official-val reference.
- Ultrashort dataref variants are a rejected family unless a future static
  counterfactual can prevent trained extra-lane carriers. v2 is close on ACC
  but fails count-shape/final-query gates; q1/q3-protected collapses earlier
  with distributed clear-far over-count.
- Restoring score-sum count losses on top of env30 worsens both primary
  official metrics and count shape.
- Tiered GT4/GT5 + GT4PV trades FP/FN in the right direction on official-val
  but does not beat env30 ACC and is worse on reporting-only TEST, so it is not
  promotable.
- No implementation direction from this matrix has >=90% engineering
  confidence. The next useful step should remain diagnostic-only and focus on
  env30 failure cases rather than another broad dataref/count/tiered ablation.

## 2026-07-16 Env30 Residual Failure Classification

The env30 official-val residual errors are sparse and mixed. The local
diagnostic summary is `.tmp/env30_failure_classification/summary.md`, with
official-impact tail lanes in
`.tmp/env30_failure_classification/official_impact_tail_lanes.csv` and a small
visual overlay sample in `.tmp/env30_tail_overlays/`.

Per-image official-val summary:

```text
images = 363
FP images = 21
FN images = 10
over/under images = 7 / 3
image_acc < 0.95 / < 0.90 = 23 / 8

FP by GT count = GT3 13, GT4 4, GT5 4
FN by GT count = GT3 6, GT4 4
FP by date = 0601 7, 0313-2 5, 0531 5, 0313-1 4
FN by date = 0313-2 3, 0531 3, 0313-1 2, 0601 2
```

Final-lane and raw-Q12 tail summary:

```text
unmatched final lanes = 21
extra top-k lanes = 7
extra far / duplicate = 7 / 0
extra carriers = q0 4, q11 3

GT lanes in raw-Q12 diagnostics = 1303
finite raw best APE mean/p50/p90 =
  5.627210 / 4.254436 / 10.033714 px
raw best APE nonfinite lanes = 5
no raw match within 20/30/40 px = 32 / 21 / 11
pred_valid_points@0.6 < min_points = 14
missing_at_match_thr = 33

no20 by GT count = GT5 14, GT4 12, GT3 6
no20 by side = center 20, right_side 7, left_side 5
no20 by visible bucket = <=10 21, 11..20 7, >20 4
no20 image status = count_ok 26, under 5, over 1

tail lanes total = 41
official-impact tail lanes = 15
non-impact tail lanes = 26
official-impact no20 lanes = 13
official-impact valid<min lanes = 7
```

Visual spot-checks support a mixed residual class:

- Some under-count images miss a full center or right lane with large raw APE
  or zero point-valid survival, not just a final threshold issue.
- Some GT lanes with 1-2 visible points are counted as raw tail diagnostics but
  do not necessarily hurt official FP/FN; blindly upweighting all no20 lanes
  would add noise.
- The few env30 over-count lanes are clear-far q0/q11 extras, not
  duplicate-like NMS failures.

Supported next action:

Do not start another training run from this alone. If continuing diagnosis,
compare only the 15 official-impact tail lanes across existing env30, v2,
tiered, and count-restore artifacts to determine whether any rejected run
actually fixes these tails without introducing larger over-count. A direct
GT5-only, count-only, or broad dataref change is not supported.

## 2026-07-16 Env30 Official-Impact Tail Cross-Run Check

The 15 env30 official-impact tail lanes were compared across existing
official-val-only artifacts. The local summary is
`.tmp/tail_cross_run/tail_cross_run_summary.md`, with per-lane details in
`.tmp/tail_cross_run/official_impact_tail_cross_run.csv`. TEST remains closed
for all selection and tuning decisions.

Aggregate diagnostic comparison:

```text
env30:
  ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
  global no20 / valid<min = 32 / 14
  has20 rate / APE p90 = 0.975441 / 10.033714
  tail still bad = 15
  final-query over/under = 7 / 3

ultrashort dataref v2:
  ACC/FP/FN = 0.972040 / 0.023691 / 0.011938
  global no20 / valid<min = 43 / 6
  has20 rate / APE p90 = 0.966999 / 10.554546
  tail no20 fixed / valid fixed / both ok = 2 / 3 / 3
  tail still bad = 12
  final-query over/under = 22 / 3

tiered GT4/GT5 + GT4PV v2:
  ACC/FP/FN = 0.973049 / 0.014141 / 0.008724
  global no20 / valid<min = 27 / 10
  has20 rate / APE p90 = 0.979279 / 9.512536
  tail no20 fixed / valid fixed / both ok = 3 / 5 / 5
  tail still bad = 10
  reporting-only TEST was worse than env30

count-restore env30:
  ACC/FP/FN = 0.944877 / 0.096740 / 0.063131
  global no20 / valid<min = 186 / 40
  has20 rate / APE p90 = 0.857252 / 28.397345
  tail still bad = 13

q1/q3-protected dataref:
  ACC/FP/FN = 0.949375 / 0.113682 / 0.047980
  global no20 / valid<min = 163 / 19
  has20 rate / APE p90 = 0.874904 / 23.709969
  tail still bad = 13
  final-query over/under = 82 / 3
```

Supported bottleneck synthesis:

- v2 fixes a few env30 tails but worsens global raw match rate and creates
  broader over-count, so it is not a promotable tail-fix direction.
- Tiered v2 gives the closest raw geometry and fixes the most tail lanes among
  rejected runs, but its official-val ACC remains below env30 and its
  reporting-only TEST result was worse. It is diagnostic evidence, not a
  candidate.
- Count-restore and q1/q3-protected both globally degrade raw geometry and are
  not useful tail-fix carriers.
- The remaining env30 tail failures are not solved by any existing rejected
  family without reintroducing larger count or over-count regressions.

Smallest safe next action:

Do not implement a new loss, decode, count policy, or reference-bank change
from this cross-run check. If continuing, isolate why tiered v2 fixes five
official-impact tails while missing the env30 gate, using read-only per-lane
and per-query diagnostics before proposing any trainable change. No direction
from this evidence reaches the >=90% engineering-confidence threshold.

### Tiered v2 Tail-Fix Mechanism Split

A narrower lane-level check of the same 15 official-impact tails shows why the
tiered-v2 signal is not directly reusable as a candidate.

```text
tiered_v2 lane-level fixes among 15 env30 official-impact tails = 6
fix flags = fix_valid 3, fix_no20+fix_valid 2, fix_no20 1
fixed GT counts = GT3 2, GT4 2, GT5 2
fixed positions = center 4, leftmost 1, rightmost 1
fixed visible buckets = <=10: 1, 11..20: 3, >20: 2

env30 final count transitions on those six lanes:
  GT3 3->4: 2
  GT4 4->3: 2
  GT5 5->5: 2

tiered_v2 final count transitions on those six lanes:
  GT3 3->3: 1
  GT3 3->4: 1
  GT4 4->5: 2
  GT5 5->6: 2
```

The global raw-Q12 official-val shift is also mixed:

```text
env30      no20 / valid<min / APE p90 = 32 / 14 / 10.033714
tiered_v2  no20 / valid<min / APE p90 = 27 / 10 / 9.512536

env30 GT3 no20 / valid<min = 6 / 5
tiered GT3 no20 / valid<min = 5 / 0
env30 GT4 no20 / valid<min = 12 / 4
tiered GT4 no20 / valid<min = 9 / 7
env30 GT5 no20 / valid<min = 14 / 5
tiered GT5 no20 / valid<min = 13 / 3
```

Interpretation:

- Tiered v2 does provide a real raw-geometry and point-valid survival signal.
- The signal is not count-safe: only one of the six lane-level tail fixes
  corresponds to a correct final count; the rest are still over-count or turn
  into GT4/GT5 over-count.
- The GT4 split is especially risky: raw no20 improves, but GT4
  `valid<min` increases and final count creates `4->5`.
- The GT5 short-lane rescue improves some raw/valid tails but reintroduces
  `5->6`.

Therefore the next diagnostic should not be "reuse tiered rescue with smaller
weights" by default. It should first separate tail-lane retention from the
extra-lane carriers that create GT4 false fifth and GT5 sixth-lane outputs.

### Tiered v2 Final-Query Carrier Check

An official-val-only final-query extra diagnostic was run for the tiered-v2
`official_best.pt` using the already selected official-val decode
(`conf=0.001`, `point_valid_thr=0.6`, `nms_dist_px=30`, `max_det=6`,
`min_points=2`, `valid_before_maxdet=True`). This is a GT-after-decode
diagnostic only; it does not alter official metrics and does not use TEST.

```text
remote =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_official_best_val_final_query_extra_valid_before_maxdet
local =
  .tmp/tiered_v2_final_query_diag/

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

The final-query extra gate rejects tiered v2 against env30:

```text
official_val_over_images: candidate 11, baseline 7, limit 10
official_val_gt5_5to6_extra_zero: candidate 5, limit 0
official_val_gt4_4to5_extra_le_baseline: candidate 2, baseline 0
official_val_focus_queries_gt5_5to6_zero: candidate 2, limit 0
```

This confirms that the tiered-v2 failure is not mainly duplicate/NMS. The
extra carriers are mostly far and query-distributed, while the same run's tail
repairs come from raw/valid survival. A viable future candidate must preserve
the tail survival gain without allowing q3/q4/q9/q10/q11 or q0/q1 far carriers
to survive as extra final lanes.

Matched tail lanes and far extras are not separable by a simple score or valid
length threshold:

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

Four of the six fixed-tail images still have an extra final lane. The matched
tail queries and extra carrier queries overlap (`q0`, `q4`, `q10`, `q11`), so
query-wide suppression would damage the same tail-retention signal it tries to
protect. The only strong separator in this diagnostic is GT-based nearest-lane
distance, which is valid for training diagnostics but not a TEST-time decode
feature.

GT-based post-hoc removal shows why the next step should be diagnostic rather
than decode:

```text
tiered_v2 baseline final-query confusion:
  over/under = 11 / 3
  3->4 = 4, 4->3 = 2, 4->5 = 2, 5->4 = 1, 5->6 = 5

remove only GT-labeled far extra final lanes:
  over/under = 1 / 3
  3->4 = 0, 4->3 = 2, 4->5 = 0, 5->4 = 1, 5->6 = 1

remove all GT-labeled extra_topk final lanes:
  over/under = 0 / 3
  3->4 = 0, 4->3 = 2, 4->5 = 0, 5->4 = 1, 5->6 = 0
```

This oracle removal would keep the six matched tail-retention lanes because
they are not marked as extra. It supports a possible diagnostic axis around
train-only far-extra carrier suppression, but not a valid inference-time rule.
Any future implementation would need to prove a non-oracle training signal can
reduce the q3/q4/q9/q10 far extras without suppressing the overlapping matched
tail queries.

### Tiered v2 Train-Split Final-Query Check

Follow-up train-only diagnostics ran the same selected tiered-v2
`official_best.pt` decode on train0601 and train0531. These use train GT only
for diagnosis and do not affect official-val or TEST selection.

```text
train0601 remote =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_official_best_train0601_final_query_extra_valid_before_maxdet
train0531 remote =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_official_best_train0531_final_query_extra_valid_before_maxdet
local =
  .tmp/tiered_v2_train_final_query_diag/
```

Comparable train final-query summaries:

```text
env30 train0601:
  over/under/extra = 8 / 5 / 8
  extra hist = q1 4, q10 3, q11 1
  GT4 4->5 = 7, GT5 5->6 = 0

v2 train0601:
  over/under/extra = 70 / 6 / 70
  extra hist = q1 8, q10 4, q11 1, q3 19, q4 9, q6 15, q7 14
  GT4 4->5 = 15, GT5 5->6 = 53

tiered_v2 train0601:
  over/under/extra = 21 / 5 / 24
  extra hist = q0 4, q3 5, q4 5, q6 1, q7 2, q8 1, q10 6
  far / duplicate / near-nondup extras = 16 / 3 / 7
  GT4 4->5 = 7, GT4 4->6 = 3, GT5 5->6 = 11

env30 train0531:
  over/under/extra = 7 / 1 / 7
  GT4 4->5 = 2, GT5 5->6 = 0

v2 train0531:
  over/under/extra = 7 / 1 / 8
  GT4 4->5 = 3, GT5 5->6 = 0

tiered_v2 train0531:
  over/under/extra = 4 / 0 / 4
  extra hist = q1 1, q9 1, q10 2
  far / duplicate / near-nondup extras = 4 / 0 / 0
  GT4 4->5 = 3, GT5 5->6 = 0
```

GT-oracle train split removal:

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

Guard-scope counterfactual:

```text
current gcs_final_extra_guard_scope = q1,q3,q4,q5,q6,q7,q8

val far extras:
  in current scope = 5 (q1 2, q3 1, q4 2)
  outside scope = 5 (q0 1, q9 1, q10 2, q11 1)

train0601 far extras:
  in current scope = 8 (q3 3, q4 3, q6 1, q7 1)
  outside scope = 8 (q0 3, q10 5)

train0531 far extras:
  in current scope = 1 (q1 1)
  outside scope = 3 (q9 1, q10 2)
```

Existing guard-log evidence:

```text
v2 remote results.csv:
  final_extra_guard active train-only
  train/final_extra_guard_count first/last/last10mean =
    17.7843 / 0.14706 / 0.192157
  val/final_extra_guard_count = 0 throughout

q1q3-protected results.csv:
  train/final_extra_guard_count first/last/last10mean =
    25.3529 / 0.85294 / 0.664707
  val/final_extra_guard_count = 0 throughout
```

Synthesis:

- The expanded-scope hypothesis is plausible because the current guard scope
  misses q0/q9/q10/q11, and q10 is a repeated far-extra carrier on val,
  train0601, and train0531.
- It is not yet a training recommendation. The existing guard already fired
  during v2/q1q3 training and those runs still failed official-val gates.
- Far/duplicate suppression alone would not solve train0601: five GT5
  `5->6` near-nonduplicate extras remain after oracle removal of far/duplicate
  extras.
- A future diagnostic candidate would need to test whether expanded-scope
  guard selection can suppress q0/q9/q10/q11 far extras without damaging the
  same queries when they are matched tail-retention lanes. A simple gain bump
  or query-wide suppression remains unsupported.

### Tiered v2 Guard-Scope Replay

The expanded-scope hypothesis was replayed against the tiered-v2
`official_best.pt` using the training-time `final_extra_guard` candidate
selection logic, not inference-time decode. This is a GT-after-forward
diagnostic for loss-selection scope only; it is not official-val ACC evidence
and does not touch TEST.

```text
remote val =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_guard_scope_val_trainimgsz_diag
remote train0601 =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_guard_scope_train0601_trainimgsz_diag
remote train0531 =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2_guard_scope_train0531_trainimgsz_diag
local =
  .tmp/tiered_v2_guard_scope_trainimgsz_diag/

current scope = q1,q3,q4,q5,q6,q7,q8
expanded scope = q0,q1,q3,q4,q5,q6,q7,q8,q9,q10,q11
outside scope = q0,q9,q10,q11
distance pixel scale = training --imgsz 544 960
```

Replay counts:

```text
official-val:
  current candidate/selected/protected = 7 / 3 / 4
  expanded candidate/selected/protected = 15 / 10 / 5
  outside candidate/selected/protected = 8 / 7 / 1
  outside selected queries = q0 1, q9 2, q10 2, q11 2
  outside protected queries = q9 1
  outside selected GT counts = GT3 2, GT4 4, GT5 1

train0601:
  current candidate/selected/protected = 20 / 5 / 13
  expanded candidate/selected/protected = 30 / 11 / 16
  outside candidate/selected/protected = 10 / 6 / 3
  outside selected queries = q0 3, q10 3
  outside protected queries = q10 3
  outside selected GT counts = GT4 4, GT5 2

train0531:
  current candidate/selected/protected = 2 / 1 / 1
  expanded candidate/selected/protected = 6 / 5 / 1
  outside candidate/selected/protected = 4 / 4 / 0
  outside selected queries = q9 1, q10 3
  outside selected GT counts = GT4 4
```

All outside-scope selected candidates in this replay are classified as
clear-far and none are duplicate-like. This strengthens the claim that q0,
q9, q10, and q11 carry real far-extra signal missed by the current guard scope.
However, the training-scale replay also has outside-scope protected candidates
on official-val and train0601, including three q10 protected candidates on
train0601, so an expanded scope is not proven safe as a direct training
change. The correct interpretation is narrower: expanded scope is a plausible
diagnostic axis, but a training candidate still needs an official-val gate and
should not rely on a gain bump or query-wide suppression.

Cross-referencing the outside-scope replay trace with the existing final-query
extra trace by `(raw_file, query_id)` gives:

```text
official-val:
  guard outside selected = 7
  selected rows that are final outside extras = 5
  final outside extras covered by guard selected/protected/other = 5 / 0 / 0

train0601:
  guard outside selected = 6
  selected rows that are final outside extras = 6
  final outside extras covered by guard selected/protected/other = 6 / 2 / 2
  non-selected final outside extras =
    q10 GT4 clear-far but protected by line_acc=1.0,
    q10 GT4 clear-far but below clear-far threshold under train scale,
    q10 GT5 near non-far candidate,
    q0 GT4 final extra skipped as Hungarian-matched in guard replay

train0531:
  guard outside selected = 4
  selected rows that are final outside extras = 3
  final outside extras covered by guard selected/protected/other = 3 / 0 / 0
```

This strengthens the diagnostic value of expanded scope for official-val and
train0531, but train0601 remains the blocker: the same outside queries include
protected and near/non-candidate final extras, so scope expansion alone would
not explain or solve all tiered-v2 over-count behavior.

A matched-tail cross-check on the same official-val traces confirms that the
expanded guard can separate the six tiered-v2 fixed official-impact tail lanes
from the same-image far extras in this replay, but only under GT-aware
diagnostic logic:

```text
tiered_v2 fixed official-impact tail lanes from
  .tmp/tail_cross_run/official_impact_tail_cross_run.csv:
  n = 6
  expanded-guard status = matched skip for all 6
  semantics = matched=True, candidate=False, selected=False, protected=False
  queries = q0 2, q4 1, q10 1, q11 2

same six images, final-query extra lanes:
  n = 4
  expanded-guard status = selected for all 4
  all four are nearest-GT far and duplicate_like=False
  extras =
    clips/0601/1494452925855079794/20.jpg q10
    clips/0601/1495485219551850010/20.jpg q3
    clips/0531/1492724773779246375/20.jpg q9
    clips/0313-2/580/20.jpg q0
```

This is useful evidence that the training-time guard predicate has a real
non-oracle-like separation on these official-val examples: matched tail lanes
are not selected, while same-image far extras are selected. It is still not a
training recommendation because the broader train0601 trace has protected and
near/non-candidate q10/q0 failures.

A narrow clear-far threshold counterfactual on the train-image-scale replay
also does not clear the blocker:

```text
counterfactual = lower guard clear_far_px from 50 to 40
newly captured final outside extras = 1
row =
  clips/0601/1494453811467016035/20.jpg q10 GT4
  final_far=True, duplicate_like=False
  guard nearest_gt_ape_px = 42.971340
  line_acc = 0.277778
  valid_len = 11
```

Lowering `clear_far_px` to `40` would capture this one train0601 q10 final
far extra, but it would not address the protected `line_acc=1.0` q10 case, the
near/non-far q10 case, or the q0 case that final-query greedy attribution marks
as extra while the training-time Hungarian matcher marks it as matched and
skips guard candidate selection. Therefore threshold relaxation remains
diagnostic-only and is not sufficient evidence for a new training run.

A row-level train0601 blocker classification gives the hard gate for this
line:

```text
train0601 outside final extras = 10
outside queries = q0 4, q10 6
outside final far extras = 8
outside duplicate-like extras = 1

expanded guard coverage:
  selected = 6
  protected = 2
  near/non-far candidate = 1
  Hungarian-matched skip = 1
  other = 0

uncovered rows:
  clips/0601/1494453653536229036/20.jpg q10 GT4
    protected=True, line_acc=1.0, final transition 4->6
  clips/0601/1495492674588939817/20.jpg q10 GT5
    protected=True, nearest_gt_ape=26.58, final transition 5->6
  clips/0601/1494453811467016035/20.jpg q10 GT4
    candidate=True, selected=False, clear_far=False,
    nearest_gt_ape=42.97, final transition 4->6
  clips/0601/1495492704577874815/20.jpg q0 GT4
    matched=True, candidate=False, final transition 4->5
```

Removing only the outside extras that expanded replay would select is still
not enough under the GT-after-decode counterfactual:

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

This closes the expanded-scope gate for now. It identifies a real missed
clear-far subset but not a sufficient training candidate: selected outside
suppression leaves substantial GT4/GT5 over-count, and all-extra removal would
still leave GT5 under-count (`5->4=4`).

A broader all-extra alignment against the expanded guard replay strengthens
the rejection. The issue is not only outside-scope q0/q10:

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

GT-after-decode all-extra counterfactuals:

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

This pattern pauses direct expanded-guard training as the next candidate. The
guard can identify useful selected clear-far subsets on official-val and
train0531, but train0601 retains protected, near/non-far, matched-skip, and
noncandidate extras. Treating those as one target-zero population would collide
with positive protection and Hungarian matched positives. The next useful step
is assignment/visualization-level diagnosis of the protected q10 rows, the
q10 distance-scale boundary row, and the q0 Hungarian-vs-final assignment
conflict, not a scope expansion, guard-gain bump, or `clear_far_px` relaxation.

Assignment-level inspection was then run on the remote train0601 split using
tiered-v2 `official_best.pt` and copied locally to
`.tmp/tiered_v2_assignment_blockers/train0601/`. This diagnostic joins
training Hungarian assignment, expanded-guard selection, final-query greedy
attribution, and per-query/per-GT APE/line-accuracy.

```text
artifact =
  .tmp/tiered_v2_assignment_blockers/train0601/
records = 410
query rows = 4920
per-query/GT rows = 21444
weights =
  runs/gcs_lane/query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2/weights/official_best.pt
```

The four outside final extras not selected by expanded guard have different
mechanisms:

```text
clips/0601/1494453653536229036/20.jpg q10 GT4 4->6
  Hungarian matched = false
  protected by = protect_acc
  nearest GT = lane 3/order 1, APE 56.93 px
  best line-acc GT = lane 3/order 1, line_acc 1.0
  nearest matched query = q8, distance 45.27 px
  final trace marks q10 as far and duplicate-like extra

clips/0601/1495492674588939817/20.jpg q10 GT5 5->6
  Hungarian matched = false
  protected by = protect_px
  nearest GT = lane 4/order 3, APE 26.58 px
  best line_acc = 0.70
  nearest matched query = q11, distance 43.46 px
  final trace marks q10 as near/non-far extra

clips/0601/1494453811467016035/20.jpg q10 GT4 4->6
  Hungarian matched = false
  candidate=True but selected=False and protected=False
  nearest GT = lane 3/order 3, APE 42.97 px
  best line_acc = 0.2778
  nearest matched query = q6, distance 42.98 px
  final trace marks q10 as far extra because final-scale APE is 53.78 px

clips/0601/1495492704577874815/20.jpg q0 GT4 4->5
  Hungarian matched = true to GT lane 0/order 0
  nearest GT = lane 0/order 0, APE 10.18 px, line_acc 0.8333
  final trace marks q0 as extra, while q3 is final-greedy matched to GT0
    with infinite APE under the trace metrics
```

Interpretation:

- The q10 protected rows are not safe hard negatives: one is protected by high
  line accuracy to a GT lane, and one is protected by being within the
  `protect_px` corridor.
- The q10 near/non-far row is a scale/boundary disagreement: the training-scale
  guard sees `42.97 < clear_far_px 50`, while final trace labels it far at
  `53.78`.
- The q0 row is an assignment-disagreement case: training Hungarian treats q0
  as a positive for GT0, while final greedy attribution labels q0 as extra.
  That cannot be handled by any unmatched-query guard without damaging a
  matcher positive.

This confirms the expanded-guard line should stay paused. There is no shared
non-oracle hard-negative feature across these blockers. A future candidate
would need a new, assignment-aware diagnostic signal that separates harmful
final extras from GT-close or Hungarian-positive lanes before any
official-val-gated training run is justified.

Earlier replay artifacts under `.tmp/tiered_v2_guard_scope_diag/` used original
TuSimple image scale for distance thresholds and are superseded by the
train-image-scale artifacts above. They remain useful only as a rough
cross-check and should not be described as exact training-loss replay.

## 2026-07-16 Env30 Assignment-Blocker Check

The env30 reference was inspected with the same assignment-blocker diagnostic
used for the tiered-v2 guard analysis. This is official-val/train diagnosis
only; TEST remains closed and no training candidate is selected from it.

```text
artifacts =
  .tmp/env30_assignment_blockers/val/
  .tmp/env30_assignment_blockers/train0601/
  .tmp/env30_assignment_blockers/train0531/
weights =
  runs/gcs_lane/query_alpha05_gt5short_geom_w2_bneg002_env30_nocount_v1/weights/official_best.pt
```

Split summaries:

```text
official-val:
  records/query rows/per-query-GT rows = 363 / 4356 / 15636
  final extras = 7 images
  extra query hist = q0 4, q11 3
  count transition = GT3 3->4: 7
  selected clear-far train signal = 6 / 7
  near/non-far blocker = 1 / 7

train0601:
  records/query rows/per-query-GT rows = 410 / 4920 / 21444
  final extras = 8 images
  extra query hist = q1 4, q10 3, q11 1
  count transitions = GT3 3->4: 1, GT4 4->5: 7
  selected clear-far train signal = 8 / 8

train0531:
  records/query rows/per-query-GT rows = 358 / 4296 / 13416
  final extras = 7 images
  extra query hist = q0 2, q1 1, q10 1, q11 3
  count transitions = 2->3: 1, 3->4: 4, 4->5: 2
  selected clear-far train signal = 4 / 7
  protected = 1 / 7
  Hungarian-matched final-extra attribution = 1 / 7
  near/non-far = 3 / 7
```

The q0/q11 official-val extra signal is not a safe query-level suppression
target:

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
```

The 15 env30 official-impact tail lanes also use q0/q11 as raw-best queries:

```text
official-impact tail raw-best query hist includes q0 3, q11 3
```

Supported interpretation:

- Official-val q0/q11 extras are real but sparse: all seven are GT3 `3->4`
  over-counts and six are clean selected clear-far rows.
- The train splits do not reproduce a clean q0/q11-only pattern. Train0601
  extra carriers are mostly q1/q10 and GT4 `4->5`; train0531 mixes q0/q1/q10/q11
  with protected, near/non-far, and Hungarian-matched blocker cases.
- q0/q11 are heavily used as true-lane positives and also appear in
  official-impact tails. Query-wide suppression, score/valid-length threshold
  suppression, or a blind expanded final-extra guard would likely trade a few
  extra-lane removals for GT4/GT5 under-count or tail-lane damage.
- Train0601 selected hard-negative rows are not identical to harmful final
  extras. Half of the selected rows are non-extra rows, concentrated in GT5
  cases, so a train-only BCE target from this population risks suppressing
  true `5->5` lanes.
- Train0531 confirms the blocker pattern is not just missing guard scope:
  one q10 final extra is `protect_px`, one q11 final-extra attribution is
  Hungarian matched with GT line accuracy `1.0`, and one q0 extra is
  near/non-far with high line accuracy.

Decision:

Do not start a new env30 q0/q11 or q1/q10 hard-negative training run. Do not
add query-wide suppression, score/valid-length suppression, guard gain bumps,
`clear_far_px` relaxation, or expanded `gcs_final_extra_guard_scope` from this
evidence. The train-only hard-negative direction remains below the
`>=90%` engineering-confidence implementation threshold.

Smallest safe next action:

If this direction continues, keep it read-only and require a counterfactual
table before any training proposal. The table must report, for each proposed
non-oracle rule, selected final-extra precision, matched/Hungarian positives
selected, official-impact tail raw-best rows touched, GT3/GT4/GT5 transition
effects, and train0601/train0531 disagreement. A later candidate should be
rejected unless it separates harmful extras from GT-close or matched positives
without relying on TEST or GT at decode time.

Follow-up counterfactual table:

```text
artifact =
  .tmp/env30_assignment_counterfactual/summary.md
csv =
  .tmp/env30_assignment_counterfactual/rule_counterfactual_summary.csv
examples =
  .tmp/env30_assignment_counterfactual/rule_counterfactual_examples.csv
```

Key rows:

```text
selected_any:
  official-val touched/final-extra/precision = 11 / 6 / 0.545455
  official-val final positives touched = 2
  train0601 touched/final-extra/precision = 16 / 8 / 0.500000
  train0601 final positives touched = 1
  train0601 GT5 5->5 final lanes touched = 6
  train0531 touched/final-extra/precision = 6 / 4 / 0.666667

selected_q0q11:
  official-val touched/final-extra/precision = 8 / 6 / 0.750000
  train0601 touched/final-extra/precision = 3 / 1 / 0.333333
  train0531 touched/final-extra/precision = 4 / 3 / 0.750000

querywide_q0q11:
  official-val extras removed = 7 / 7
  official-val Hungarian/final positives touched = 490 / 489
  official-impact tail raw-best rows touched = 6 / 15
  train0601 extras removed = 1 / 8
  train0601 Hungarian/final positives touched = 677 / 670
  train0531 extras removed = 5 / 7
  train0531 Hungarian/final positives touched = 408 / 405

q0q11_score_below_0.93:
  official-val extras removed = 7 / 7
  official-val Hungarian/final positives touched = 85 / 84
  train0601 extras removed = 1 / 8
  train0601 Hungarian/final positives touched = 125 / 118

q0q11_valid_len_below_24:
  official-val extras removed = 7 / 7
  official-val Hungarian/final positives touched = 470 / 469
  train0601 extras removed = 1 / 8
  train0601 Hungarian/final positives touched = 657 / 650
```

Counterfactual conclusion:

- No tested rule removes the sparse official-val extras while keeping
  Hungarian/final positives untouched across train0601 and train0531.
- The best-looking train-side row, `selected_any`, still has only 50-67%
  final-extra precision and touches non-extra final lanes; train0601 also
  touches six GT5 `5->5` final lanes.
- Query-wide q0/q11 suppression removes official-val extras but is invalid:
  it touches hundreds of true-lane positives and six of the 15 official-impact
  tail raw-best rows.
- Score and valid-length suppression are also invalid. They must discard many
  q0/q11 positives to remove the seven official-val extras and still do not
  address train0601 q1/q10 GT4 `4->5` extras.

This closes the current env30 query-level hard-negative path. It remains a
diagnostic-only finding and does not justify training, decode changes, or TEST.

## 2026-07-16 Env30 Official-Impact Tail Positive Triage

After closing the query-level hard-negative path, the 15 env30
official-impact tail lanes were triaged from the positive side. This is
official-val diagnosis only and does not use TEST.

```text
artifact =
  .tmp/env30_tail_positive_triage/summary.md
csv =
  .tmp/env30_tail_positive_triage/official_impact_tail_triage.csv
```

Aggregate:

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

Assignment-trace cross-check:

```text
raw-best query Hungarian matched = 13 / 15
raw-best query final_in_decode = 8 / 15
raw-best query final_greedy_matched_gt = 5 / 15
no20 and Hungarian matched = 12 / 13
no20 but assignment-trace APE <= 20 px = 5
no20 but assignment-trace line_acc >= 0.8 = 5
```

Interpretation:

- The positive tail is not simply an unsupervised-query or absent-positive
  problem. Most raw-best queries are already Hungarian matched under the
  training assignment diagnostic.
- Some official-val `no20` rows become close or high-line-accuracy under the
  assignment trace, so the residual is partly a scale/metric/official-line
  accuracy disagreement rather than a clean training target.
- The positive failures are mixed across GT3/GT4/GT5 and center/side lanes.
  A GT5-only or side-only positive rescue is not supported.
- Ultra-short visible<=2 rows are high risk: they may have high scores and
  valid-point survival but no stable 20px official raw geometry, so blindly
  upweighting them can add noise.
- Pure point-valid/existence survival is sparse (`2` clear rows, plus mixed
  geometry+valid rows). It is a diagnostic axis, not enough for a direct
  training change without a broader train-side denominator and official-val
  gate.

Smallest safe next action:

Do not start positive-rescue training from these 15 lanes. If this direction
continues, build a train-side denominator for the same classes and require an
official-val gate that proves raw geometry, line accuracy, and point-valid
survival improve without increasing GT3/GT4 over-count or GT5 under-count.

Follow-up train/val denominator:

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

This denominator is broad enough to support more diagnosis, but it does not
define a safe target by itself. Most rows are `ok_or_near_ok`; the positive
tail is a small subset of mixed GT3/GT4/GT5 geometry and line-accuracy issues.
The split surfaces differ sharply: train0601 is GT5-heavy, while train0531 in
this diagnostic is GT3/GT4-heavy.

Metric-mismatch check:

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

Updated interpretation:

- The 15-lane official-impact positive tail is useful for localization, but it
  is not yet a stable training target.
- The raw-tail diagnostic's `raw_best_query_id`, `no20`, and `valid<min`
  signals often disagree with the assignment best-query view that training
  would actually optimize.
- A loss built directly from these raw-tail rows would risk chasing diagnostic
  mismatch rather than improving official-val geometry.

This closes direct positive-rescue training from the current tail diagnostics.
A future positive-rescue candidate must first define a consistent train-side
selection rule and show, before training, that the selected denominator is not
mostly already matched/near-ok and does not create GT3/GT4 over-count or GT5
under-count risk.

Follow-up selector counterfactual:

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

Key official-val selector rows:

```text
geom_gt20:
  val selected = 26
  official-impact tail covered = 12 / 15
  non-impact selected = 14
  final-matched positives touched = 15
  Hungarian positives touched = 21

geom_gt30:
  val selected = 15
  official-impact tail covered = 10 / 15
  non-impact selected = 5
  final-matched positives touched = 5
  Hungarian positives touched = 10

line_acc_lt08:
  val selected = 12
  official-impact tail covered = 10 / 15
  non-impact selected = 2
  final-matched positives touched = 2
  Hungarian positives touched = 9

no_final_match:
  val selected = 17
  official-impact tail covered = 12 / 15
  non-impact selected = 5
  final-matched positives touched = 0
  Hungarian positives touched = 14

geom20_or_line_and_no_final:
  val selected = 12
  official-impact tail covered = 12 / 15
  non-impact selected = 0
  final-matched positives touched = 0
  Hungarian positives touched = 9
```

Selector interpretation:

- The best-looking `no_final_match` combinations are decode-after-the-fact
  diagnostics, not stable train-side labels. They depend on final decoded match
  state and can couple a training objective to postprocess behavior.
- Geometry-only selectors are too broad: `geom_gt20` covers many tails but
  touches more non-impact and already-final-matched positive rows than it
  explains.
- Tighter geometry and line-accuracy selectors are cleaner but miss too many
  official-impact tails and still span GT3/GT4/GT5 rather than a stable
  class-specific failure.
- The train split surfaces remain inconsistent. For example, `geom_gt20`
  selects GT4/GT5-heavy rows on train0601 (`9/48`) but only GT3/GT4 rows on
  train0531 (`5/6`), while `no_final_match` includes `ok_or_near_ok` rows on
  val and train.

Integrated candidate map after this counterfactual:

1. Assignment/final-match positive-tail selector diagnostics remain useful,
   but implementation confidence stays below 50%; no positive-rescue training
   is approved.
2. q3-style GT5 prototype matched-vs-extra disambiguation remains a read-only
   diagnostic candidate. It must find a non-oracle separator before any new
   dataref/reference run.
3. Tiered-v2 tail-retention versus far-extra carrier separation remains a
   read-only diagnostic candidate. Simple score, valid length, and query-wide
   suppression are already unsupported.
4. Selective exist/quality calibration for matched weak positives has weak
   evidence only and must first prove it will not increase GT3/GT4 over-count
   or GT5 sixth-lane behavior.
5. Split/date/GT-count distribution auditing is a guardrail for every future
   candidate, not a direct implementation path.

Decision: no direction reaches the `>=90%` engineering-confidence gate for
training, decode search, or TEST. Keep TEST closed and continue only with
read-only official-val/train diagnostics until a non-oracle selector is proven.

Follow-up split/date/GT-count stability audit:

```text
artifact =
  .tmp/env30_distribution_stability_audit/summary.md
final-query distribution =
  .tmp/env30_distribution_stability_audit/final_query_distribution.csv
positive denominator distribution =
  .tmp/env30_distribution_stability_audit/positive_denominator_distribution.csv
official-impact tail distribution =
  .tmp/env30_distribution_stability_audit/official_impact_tail_distribution.json
```

Final-query extra distribution:

```text
official-val:
  images = 363
  GT count hist = GT3 223, GT4 66, GT5 74
  over/under = 7 / 3
  extra lanes = 7
  extra query hist = q0 4, q11 3
  extra GT hist = GT3 7
  extra class hist = clear_far 7

train0601:
  images = 410
  GT count hist = GT3 92, GT4 79, GT5 239
  over/under = 8 / 5
  extra lanes = 8
  extra query hist = q1 4, q10 3, q11 1
  extra GT hist = GT3 1, GT4 7
  extra class hist = clear_far 8

train0531:
  images = 358
  GT count hist = GT2 1, GT3 312, GT4 45
  over/under = 7 / 1
  extra lanes = 8
  extra query hist = q0 2, q1 1, q9 1, q10 1, q11 3
  extra GT hist = GT2 1, GT3 5, GT4 2
  extra class hist = clear_far 5, duplicate_like 1, near_gt_nonduplicate 2
```

Positive-denominator stability:

```text
geom_gt20:
  val selected = 26, GT3/GT4/GT5 = 1 / 10 / 15
  train0601 selected = 57, GT4/GT5 = 9 / 48
  train0531 selected = 11, GT3/GT4 = 5 / 6

no_final_match:
  val selected = 17, includes ok_or_near_ok = 5
  train0601 selected = 26, includes ok_or_near_ok = 12
  train0531 selected = 10, includes ok_or_near_ok = 5
```

Stability interpretation:

- Env30 final-extra carriers are split-dependent. Official-val is sparse
  q0/q11 GT3 clear-far; train0601 shifts to q1/q10/q11 and mostly GT4;
  train0531 includes q0/q1/q9/q10/q11 plus duplicate/near-nonduplicate cases.
- Positive-denominator geometry failures are GT5-heavy on train0601 but
  GT3/GT4-only on train0531. A GT5-only or GT4-only rescue would overfit one
  split surface.
- `no_final_match` selectors include `ok_or_near_ok` rows on every split, so
  final-match state remains a diagnostic artifact rather than a clean
  train-side label.
- Split/date/GT-count stability is therefore a guardrail for future
  candidates, not an implementation direction.

Selective exist-quality / weak-visible matched-positive gate:

```text
read-only agent gate = diagnostic-only, not trainable
global gcs_exist_quality_alpha evidence = weaker than env30
current artifacts lack a stable train-side non-oracle selector
confidence = about 90% that this direction should remain diagnostic-only
```

The smallest allowed follow-up for this line is a read-only, training-available
selector table that excludes `no_final_match` and GT post-decode attribution,
then adds train0601/train0531 visible-count bins and quality/existence
margin-like fields if those are available in current artifacts. It must prove
tail coverage without mostly touching already-final/near-ok positives and
without GT3/GT4 over-count or GT5 `5->6` risk before any training proposal.

Follow-up training-available non-oracle selector table:

```text
artifact =
  .tmp/env30_non_oracle_selector_table/summary.md
summary =
  .tmp/env30_non_oracle_selector_table/selector_summary.csv
split distribution =
  .tmp/env30_non_oracle_selector_table/selector_split_distribution.csv
sample rows =
  .tmp/env30_non_oracle_selector_table/selected_rows_sample.csv
```

The selector inputs are limited to fields available from labels, raw queries,
and assignment diagnostics: GT visible count, raw APE, raw existence score,
raw point-valid count, assignment APE, and assignment line accuracy. They
explicitly exclude `gt_has_final_match`, final decode inclusion, final count,
extra-topK attribution, nearest-GT far/near post-decode labels, and TEST.

Key official-val rows:

```text
raw_no20:
  selected = 32
  official-impact tail covered = 13 / 15
  non-impact selected = 19
  final-matched positives touched = 21
  Hungarian positives touched = 27
  split selected: val 32, train0601 70, train0531 18

assign_ape20_or_line08:
  selected = 27
  official-impact tail covered = 13 / 15
  non-impact selected = 14
  final-matched positives touched = 15
  Hungarian positives touched = 22
  split selected: val 27, train0601 59, train0531 12

raw_no20_score_lt01:
  selected = 4
  official-impact tail covered = 4 / 15
  non-impact selected = 0
  final-matched positives touched = 0
  Hungarian positives touched = 4
  split selected: val 4, train0601 5, train0531 4

raw_no20_valid_lt_min:
  selected = 5
  official-impact tail covered = 5 / 15
  non-impact selected = 0
  final-matched positives touched = 0
  Hungarian positives touched = 5
  split selected: val 5, train0601 8, train0531 5

raw_no20_score_lt05:
  selected = 6
  official-impact tail covered = 6 / 15
  non-impact selected = 0
  final-matched positives touched = 1
  Hungarian positives touched = 6
  extra-image touched = 1

raw_score_lt01_or_valid_lt_min:
  selected = 14
  official-impact tail covered = 7 / 15
  non-impact selected = 7
  final-matched positives touched = 8
  Hungarian positives touched = 14

assign_line_lt08:
  selected = 12
  official-impact tail covered = 10 / 15
  non-impact selected = 2
  final-matched positives touched = 2
  Hungarian positives touched = 9

assign_ape20_line08:
  selected = 11
  official-impact tail covered = 9 / 15
  non-impact selected = 2
  final-matched positives touched = 2
  Hungarian positives touched = 8
```

Selector-table interpretation:

- Broad raw/assignment geometry selectors cover many official-impact tails but
  also select many non-impact and already matched positives. They are not safe
  train labels.
- Strict weak-existence or valid-survival selectors cover only `4/15`,
  `5/15`, or `6/15` official-impact tails. Broader score/valid combinations
  reach `7/15` only by reintroducing non-impact and final-matched
  contamination.
- Assignment line-accuracy selectors can cover `10/15` tails, but they still
  touch non-impact, final-matched, and Hungarian positives, and their GT-count
  mix is unstable across train splits.
- Visible-count filters do not stabilize the surface. `raw_no20_visible_le20`
  still spans GT3/GT4/GT5 on official-val and shifts strongly toward GT5 on
  train0601.
- The current artifacts do not contain an explicit train target margin for
  quality/existence. Raw existence score and point-valid count are only proxies
  and do not define a safe selector by themselves.

Decision: this closes the current selective exist-quality /
weak-visible matched-positive calibration path for implementation. No
training-available non-oracle selector reaches the `>=90%`
engineering-confidence gate. Keep TEST and training closed for this line. A
future attempt would need new artifacts with explicit train target margins or
another non-oracle separator, followed by official-val gating before any TEST.

### Env30 Short GT4/GT5 Hardset Coverage Gate

A read-only hardset coverage matrix was generated to test whether the next
candidate-geometry direction has a stable short GT4/GT5 raw-candidate
denominator before any reference/query redesign.

```text
artifact =
  .tmp/env30_short_gt45_hardset_coverage/summary.md
coverage summary =
  .tmp/env30_short_gt45_hardset_coverage/short_gt45_coverage_summary.csv
primary hard rows =
  .tmp/env30_short_gt45_hardset_coverage/primary_short_gt45_hard_rows.csv
primary hard summary =
  .tmp/env30_short_gt45_hardset_coverage/primary_short_gt45_hard_summary.csv
cross-run GT4/GT5 tail summary =
  .tmp/env30_short_gt45_hardset_coverage/official_impact_gt45_cross_run_summary.csv
```

Scope:

```text
splits = official-val, train0601, train0531
TEST = not used
primary hard row =
  gt_count in {4,5}
  visible_points_gt <= 20
  (raw_has_match_20px = false OR pred_valid_points@0.6 < min_points)
```

Key coverage:

```text
official-val GT4 visible<=20:
  lanes = 106
  hard_fail = 10
  official-impact tails = 4

official-val GT5 visible<=20:
  lanes = 207
  hard_fail = 17
  official-impact tails = 5

train0601 GT4 visible<=20:
  lanes = 134
  hard_fail = 15

train0601 GT5 visible<=20:
  lanes = 671
  hard_fail = 62

train0531 GT4 visible<=20:
  lanes = 68
  hard_fail = 7

train0531 GT5 visible<=20:
  no denominator in this available diagnostic
```

Primary hard row risk matrix:

```text
val GT4:
  selected = 10
  official-impact = 4
  non-impact = 6
  final-matched positives touched = 6
  query hist = q0 5, q10 2, q11 2, q6 1

val GT5:
  selected = 17
  official-impact = 5
  non-impact = 12
  final-matched positives touched = 13
  query hist = q0 1, q1 1, q3 9, q7 1, q8 1, q10 2, q11 2

train0601 GT4:
  selected = 15
  final-matched positives touched = 13
  query hist = q0 2, q1 1, q5 1, q6 2, q7 1, q8 1, q9 1, q10 4, q11 2

train0601 GT5:
  selected = 61
  final-matched positives touched = 51
  query hist = q0 6, q1 8, q3 30, q5 1, q7 3, q8 2, q10 8, q11 3

train0531 GT4:
  selected = 7
  final-matched positives touched = 5
  query hist = q1 2, q4 1, q7 1, q10 1, q11 2
```

Interpretation:

- The primary hard row covers all short GT4/GT5 official-impact tails on
  official-val (`9/9`), but it selects `27` official-val rows, so `18` are
  non-impact readouts.
- Most selected rows are already Hungarian/final positives in the joined
  positive-denominator table. This makes the row useful for coverage diagnosis
  but unsafe as a direct train loss selector.
- The query distribution overlaps known extra-lane carriers: q0/q10/q11 appear
  on val/train hard rows; train0601 also includes q1/q10/q11; tiered-v2 extra
  failures overlap q0/q1/q3/q4/q9/q10/q11.
- The matrix exposes raw geometry/no20 failures, not just point-valid
  survival. Therefore a valid-only rescue is not the supported next step.
- The split surface is unstable. Train0601 has a large GT5 short denominator,
  while train0531 has no GT5 short denominator in this available diagnostic.

Decision:

This is a useful gate diagnostic but not a training candidate. It supports the
need for a concrete reference-coverage counterfactual before any hardset-driven
reference/query experiment. A future candidate must first show, without using
TEST, that proposed references improve raw `has20`/APE on this denominator
without broadening GT3/GT4 extra-lane carriers or suppressing the already
matched positives.

### Env30 Short GT4/GT5 Reference-Coverage Counterfactual

A static reference-bank counterfactual compared existing Q12 ultrashort banks
against the short GT4/GT5 hard denominator above. It uses env30
official-val/train raw-Q12 diagnostics and TuSimple GT only; no model is
trained, no decode threshold is selected, and TEST is not used.

```text
artifact =
  .tmp/env30_short_gt45_reference_counterfactual/summary.md
summary csv =
  .tmp/env30_short_gt45_reference_counterfactual/reference_counterfactual_summary.csv
primary rows =
  .tmp/env30_short_gt45_reference_counterfactual/reference_counterfactual_primary_rows.csv
```

Primary hard static coverage:

```text
default_q12 val GT4 p50/p90/match20 =
  136.972 / 153.686 / 0.1250
default_q12 val GT5 p50/p90/match20 =
  154.365 / 218.186 / 0.0625

q1q3protected_v1 val GT4 p50/p90/match20 =
  41.667 / 80.432 / 0.2500
q1q3protected_v1 val GT5 p50/p90/match20 =
  36.234 / 68.968 / 0.2500
q1q3protected_v1 train0601 GT5 p50/p90/match20 =
  37.883 / 85.445 / 0.2333
changed extra-overlap queries = q4,q10

ultrashort_v2 val GT5 p50/p90/match20 =
  36.234 / 81.555 / 0.2500
ultrashort_v2 train0601 GT5 p50/p90/match20 =
  32.193 / 101.192 / 0.2667
changed extra-overlap queries = q3,q4,q10
```

Impact-only coverage still has an important gap:

```text
default_q12 val GT4 impact-only p50/p90/match20 =
  77.859 / 143.480 / 0.2500
q1q3protected_v1 val GT4 impact-only p50/p90/match20 =
  22.000 / 44.195 / 0.5000

default_q12 val GT5 impact-only p50/p90/match20 =
  113.839 / 139.892 / 0.2500
q1q3protected_v1 val GT5 impact-only p50/p90/match20 =
  58.990 / 94.982 / 0.0000
```

Interpretation:

- Existing ultrashort banks improve static nearest-reference coverage on the
  primary hard denominator, especially GT5.
- The improvement is not a training signal by itself. The already completed
  v2 and q1/q3-protected runs both had static coverage evidence but failed
  trained official-val/count-shape gates by creating extra final-lane carriers.
- Every improving bank changes or uses queries that overlap known extra-lane
  carriers, including q0/q1/q3/q4/q10/q11 depending on the bank and split.
- The impact-only GT5 rows are not safely solved: the best q1/q3-protected and
  q3-protected static variants still have `match20=0.0000` on the four
  official-val GT5 impact-only rows.

Decision:

Keep this counterfactual diagnostic-only. No existing reference bank reaches
the `>=90%` engineering-confidence threshold for training. Do not train, run
TEST, tune decode, or edit references from this result. If this line continues,
the next action should be a new static design search constrained by
extra-carrier query risk and impact-only GT5 coverage, followed by
official-val/train gates before any remote formal training.

### q2 Protected-Query Static Design Search

Two stricter scratch banks were generated to test whether a safer static
assignment can avoid the high-risk q0/q1/q3/q10/q11 query changes that rejected
earlier ultrashort banks. This is still static reference geometry only: no
model is trained, no decode row is selected, and TEST is not used.

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

Static coverage:

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

Historical q4/q9 trace risk:

```text
q4 available final-lane traces:
  extra = 19, GT5 extra = 19

q9 available final-lane traces:
  extra = 8, GT5 extra = 0
```

Interpretation:

- The q9 design is the safer of the two with respect to GT5 over-count because
  q4 repeatedly became a GT5 extra carrier in v2/tiered diagnostics.
- Both designs still change a known extra-carrier query (`q9` or `q4`).
- Restoring q9 gives a stricter five-slot counterfactual with no changed known
  extra-carrier queries, but it raises val GT5 primary-hard p90 from `68.968`
  to `118.111`, so the current prototype set loses tail coverage when q4 and
  q9 are both protected.
- Neither design improves val GT5 impact-only `match20` beyond the default
  `0.2500`; they improve p50/p90 APE but not the impact-only match20 rate.
- Static coverage remains weaker evidence than trained official-val behavior.
  The v2 and q1/q3-protected runs already demonstrated that static reference
  gains can become extra final-lane carriers after training.

Decision:

Keep both q2 protected-query scratch designs and the q9-restored five-slot
counterfactual diagnostic-only. Do not train, run TEST, tune decode, change
thresholds/NMS/`max_det`, alter loss gains, or commit these scratch banks as
candidates. A future static search must avoid q4 and q9 as well as
q0/q1/q3/q10/q11, improve impact-only GT5 coverage, and recover the lost GT5
primary-hard p90 before any implementation or remote training proposal.

## 2026-07-16 Env30 Count-Restore Rejection

The official-val-only run
`query_alpha05_gt5short_geom_w2_bneg002_env30_countrestore_v1` is rejected and
was stopped early. It tested the smallest plausible count restoration on top
of env30:

```text
gcs_count = 0.3
gcs_count_under5 = 0.3
gcs_count_boundary = 0.2
gcs_query_count_ce = 0.0
```

Final available official-val selection:

```text
official_best source_epoch = 20
selected decode = conf 0.001, point_valid_thr 0.6, nms_dist_px 30,
  max_det 5, min_points 5, valid_before_maxdet true
ACC/FP/FN = 0.944877 / 0.096740 / 0.063131
count_acc_3/4/5 = 0.865471 / 0.833333 / 0.918919
GT4 4->5 = 10
GT5 5->4 = 6
```

Comparable references:

```text
env30 epoch020 ACC/FP/FN = 0.953702 / 0.077410 / 0.045684
env30 final ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
env30 final GT4 4->5 = 0
env30 final GT5 5->4 = 1
```

Hidden overcount remains visible under `max_det=6`:

```text
countrestore epoch020 best max_det=6 row ACC/FP/FN =
  0.942369 / 0.108907 / 0.064279
GT4 4->5 = 12
GT5 5->4/5->6 = 4 / 44

minimum GT5 5->6 among epoch020 max_det=6 rows with ACC >= 0.94 = 31
minimum GT4 4->5 among all epoch020 rows with ACC >= 0.94 = 8
```

Supported bottleneck:

- Restoring the old score-sum count losses on env30 does not recover official
  ACC and does not preserve the env30 count shape.
- The selected `max_det=5` row is not clean evidence of solving GT5 behavior:
  it undercounts GT5 (`5->4=6`) and `max_det=6` rows reveal many sixth-lane
  extras.
- The run was healthy enough to evaluate (`results.csv` has 22 complete epoch
  rows and no nonfinite strings), so this is not a runtime failure.

Smallest safe next action:

Do not continue this run, do not run TEST, and do not try another blind
count-gain tweak. Return to env30 as the reference. Further progress needs
diagnostics that distinguish final-query extra-lane sources from raw candidate
geometry/validity failures before any new implementation change.

## 2026-07-16 Final-Query Extra Count Semantics And Env30 Count/Geo Direction

Final-query extra diagnostics must be read as GT-based attribution after
decode, not as exact official FP counts.

```text
official over-image count = images where decoded pred_lanes > gt_lanes
official_FP = TuSimple official metric false-positive rate
extra_topk_candidate / extra_topk_lanes = approximate post-decode diagnostic
query histograms = likely extra-lane source attribution, not official FP
```

The corrected quantity semantics do not rescue the ultrashort dataref result:

```text
geom1 v1 TEST ACC/FP/FN = 0.966130 / 0.038845 / 0.024203
geom1 v1 TEST over_images/under_images = 286 / 147
geom1 v1 TEST diagnostic extra_topk = 316

env30 TEST ACC/FP/FN = 0.966780 / 0.028732 / 0.023544
env30 TEST over_images/under_images = 151 / 163
env30 TEST diagnostic extra_topk = 160
env30 TEST GT5 5->6 = 0
```

Official-val shows the same failure direction:

```text
geom1 v1 official-val over_images = 22, diagnostic extra_topk = 23,
  GT5 5->6 extra = 13
env30 official-val over_images = 7, diagnostic extra_topk = 7,
  GT4 4->5 = 0, GT5 5->6 = 0
```

The likely bottleneck is that ultrashort dataref prototypes improve static
coverage but become extra final-lane sources after training. This is different
from "Hungarian bound the wrong lane" and different from "one exact query count
equals official FP."

Env30 count/geo direction:

- Env30 already has GT5 short geometry active:
  `gcs_short_geom=1.0`, `gcs_short_geom_gt5_weight=2.0`.
- Re-enabling "geo like env30" is therefore not a new missing switch for the
  env30 reference; the failed ultrashort geom runs also restored that geometry
  pressure.
- Count losses were disabled in env30. A count-restore follow-up is plausible
  only as a constrained official-val ablation:
  `gcs_count=0.3`, `gcs_count_under5=0.3`,
  `gcs_count_boundary=0.2`, `gcs_query_count_ce=0.0`.
- Do not combine that ablation with GT4 point-valid rescue or broader tiered
  GT4/GT5 geometry. Gate it on official-val ACC/FN, GT4 `4->5=0`,
  GT5 `5->4<=1`, GT5 `5->6=0` under `max_det=6`, and no raw-Q12
  short-lane survival regression.

## 2026-07-16 Q12 Ultrashort Dataref v2 Bottleneck

The completed run
`query_alpha05_env30_q12_ultrashort_dataref_geom1_extraguard_v2` is rejected
before TEST. It confirms that the static q4-enabled reference-bank improvement
does not by itself solve the trained final-query extra-lane bottleneck.

```text
v2 official_best val:
ACC/FP/FN = 0.972040 / 0.023691 / 0.011938
count_acc_3/4/5 = 0.968610 / 0.939394 / 0.810811
GT4 4->3/4->4/4->5/4->6 = 1 / 62 / 2 / 1
GT5 5->4/5->5/5->6 = 2 / 60 / 12

env30 reference val:
ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
count_acc_3/4/5 = 0.968610 / 0.969697 / 0.986486
GT4 4->3/4->4/4->5 = 2 / 64 / 0
GT5 5->4/5->5/5->6 = 1 / 73 / 0

best.pt val:
ACC/FP/FN = 0.966947 / 0.053444 / 0.018136
count_acc_3/4/5 = 0.905830 / 0.803030 / 0.635135
GT4 extra = 4->5 9, 4->6 3
GT5 extra = 5->6 26
```

Final-query extra gate:

```text
gate.pass = false
official-val over_images = 22 vs env30 7, limit 10
official-val GT5 5->6 extra = 12, limit 0
official-val GT4 4->6 extra = 2, limit 0
official-val GT4 4->5 extra = 2 vs env30 0
official-val focus-query GT5 5->6 extra for q1/q3/q5/q6/q8 = 8, limit 0
official-val q5+q6 total extra = 5, limit 2
train0601 q4 far extra = 4 vs env30 0
```

Supported bottleneck:

- The run is healthy (`154` result rows, no nonfinite losses), so the failure
  is not a training crash, corrupted data, or official-best bookkeeping issue.
- `gcs_final_extra_guard=0.02` is active only on the train side and validation
  logs stay zero, matching the intended train-only loss contract. The loss is
  simply too weak or too broadly protected to prevent the observed extra query
  survival in this reference setup.
- q4 entering the reference assignment search is not sufficient. It passes the
  train q4 GT4 4->5/4->6 checks, but fails train0601 q4 far-extra protection
  and does not prevent official-val GT4/GT5 overcount.
- The official-val extra lanes are not just a decode-row artifact. The selected
  `max_det=6` row keeps `GT5 5->6=12`, while the env30 reference has
  `GT5 5->6=0` and no GT4 false fifth lanes under its selected protocol.
- `weights/best.pt` is not a fallback; it is substantially worse than
  `weights/official_best.pt` on ACC, FP, FN, and count shape.

Completed focus-query trace audit:

```text
v2 official-val focus q3/q4/q6/q7 extra = 15
  q3 6, q4 3, q6 5, q7 1
  transitions: 3->4 2, 4->6 2, 5->6 11
  class: clear_far 8, near_gt_nonduplicate 7, duplicate_like 0

env30 official-val focus q3/q4/q6/q7 extra = 0

v2 train0601 focus q3/q4/q6/q7 extra = 57
  q3 19, q4 9, q6 15, q7 14
  transitions: 3->4 1, 4->5 10, 5->6 46
  class: clear_far 28, near_gt_nonduplicate 29, duplicate_like 0

env30 train0601 focus q3/q4/q6/q7 extra = 0

v2 train0531 focus q3/q4/q6/q7 extra = 2
  q6 1, q7 1, both 4->6 clear_far

env30 train0531 focus q3/q4/q6/q7 extra = 0
```

The trace split supports the conclusion that v2 introduced new risky
focus-query final lanes rather than only moving the old env30 q0/q11 GT3
clear-far behavior. It also shows mixed failure classes. About half of the
focus extras are near-GT non-duplicate on official-val/train0601, so a blind
increase of `gcs_final_extra_guard` is not justified and may suppress true
short-lane-like candidates.

The v2 official-val sweep does not provide a decode rescue:

```text
rows = 864
rows with ACC >= env30 0.973330 = 0
rows passing count-shape gate
  (4->5=0, 4->6=0, 5->4<=1, 5->6=0, FN<=0.009642) = 0
best 5->6=0 row ACC/FP/FN = 0.971993 / 0.017355 / 0.011938
best 5->6=0 row still has 4->5=2 and 5->4=2
```

A GT-based post-hoc removal counterfactual further separates the failure
classes:

```text
v2 official-val baseline over/under = 22 / 3
v2 official-val removing only q3/q4/q6/q7 clear-far extras:
  over/under = 15 / 3, residual GT5 5->6 = 7
v2 official-val removing all q3/q4/q6/q7 extras:
  over/under = 8 / 3, residual GT5 5->6 = 1

v2 train0601 baseline over/under = 70 / 6
v2 train0601 removing only q3/q4/q6/q7 clear-far extras:
  over/under = 42 / 6, residual GT5 5->6 = 30
v2 train0601 removing all q3/q4/q6/q7 extras:
  over/under = 13 / 6, residual GT5 5->6 = 7

env30 official-val/train0601 removing all clear-far extras:
  over = 0 on both diagnostics
```

This is diagnostic-only because it uses GT-based post-decode attribution. The
supported conclusion is that clear-far suppression alone is insufficient for
v2, while suppressing every focus-query extra would also suppress near-GT
non-duplicate lanes and risks damaging true short GT5 retention.

Smallest safe next action:

Do not run TEST and do not promote v2. Do not start another training run from
this evidence. The next safe step is a concrete static bank/assignment
counterfactual on official-val/train traces, not a blind guard increase: it
must reduce q3/q4/q6/q7 clear-far extras without increasing near-GT
non-duplicate suppression or GT5 undercount before any new training change.

### Focus-Query Penalty Static Counterfactual

A scratch bank that penalized replacing q3/q4/q6/q7 passed the static coverage
gate but is rejected as a training trigger:

```text
bank = .tmp/final_query_trace_analysis/q12_ultrashort_focus_penalty_static_bank.json
query_static_penalty = 3:1000,4:1000,6:1000,7:1000,10:2
selected qset = q1/q3/q5/q6/q8/q10
official-val static gate.pass = true
hard new p50/p90/match20 = 21.000 / 43.636 / 0.400000
normal delta p50/p90 = -3.255 / +1.668
```

It only removes q4/q7 from the replacement set; q3/q6 remain selected and are
major extra-lane sources in v2. The closest trained neighbor, geom1 v1, used a
similar q1/q3/q5/q6/q7/q8 family and still produced 23 official-val
final-query extras. Therefore this static counterfactual does not justify a
new training run. The unresolved bottleneck is trained q3/q6 final-lane
survival, not merely static hard-lane reference coverage.

### q3/q6 Survival Split

The selected v2 official-val cache shows that focus-query suppression would be
too blunt:

```text
GT5 q3 after_valid/final = 9
GT5 q3 final matched/extra = 3 / 6
GT5 q3 extra clear/near = 3 / 3

GT5 q6 after_valid/final = 13
GT5 q6 final matched/extra = 11 / 2
GT5 q6 extra clear/near = 1 / 1

GT5 q7 final matched/extra = 7 / 0
```

q3 is the risky GT5 prototype, but its matched and extra lanes overlap in
score/valid length:

```text
q3 matched score/valid median = 0.278 / 7
q3 clear-extra score/valid median = 0.288 / 5
q3 near-extra score/valid median = 0.228 / 6
```

Static removal is also not available as a standalone fix:

```text
v2 static hard p50/p90/match20 = 21.000 / 43.636 / 0.400, gate.pass=true
restore q3 default = 33.378 / 95.883 / 0.200, gate.pass=false
restore q3/q4 default = 143.527 / 173.568 / 0.200, gate.pass=false
```

Therefore the bottleneck is not "remove q3" or "suppress all focus queries."
It is q3 GT5 prototype disambiguation after training: keep the static
hard-target coverage benefit while preventing q3 from becoming a sixth lane on
GT5 images.

Follow-up q3 matched-vs-extra split:

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
  score / valid_len / rank ranges all overlap between matched and extras

train0601:
  matched = 10
  clear-far extras = 13
  near-GT non-duplicate extras = 4
  score / valid_len / rank ranges all overlap between matched and extras

train0531:
  no GT5 q3 final rows in this trace
```

Updated interpretation:

- q3 cannot be suppressed query-wide: it carries true GT5 matched lanes and
  GT5 extra lanes in the same selected v2 checkpoint.
- Non-GT final-lane features tested here (`score`, `valid_len`, and final
  score-rank) do not separate q3 matched rows from q3 extra rows on
  official-val or train0601.
- GT-based attribution can label far/near extras after decode, but it is not a
  valid inference/decode rule and is not a trainable selector by itself.
- This closes direct q3 query suppression, q3 static prototype removal, or
  score/valid/rank thresholding as implementation actions. A future q3 path
  would need a new non-oracle train signal, not another dataref run from the
  current evidence.

### q1/q3-Protected Static Candidate

Protecting q3 alone is not enough because the selected bank moves the known
risky GT5 prototype pressure to q1. Protecting both q1 and q3 still passes the
static gate:

```text
q1+q3-protected selected qset = q4/q5/q6/q7/q8/q10
hard new p50/p90/match20 = 24.000 / 46.000 / 0.333
normal delta p50/p90 = -6.013 / -2.425
gate.pass = true
```

This candidate preserves the env30-safe default q1/q3 behavior:

```text
env30 official-val q3 final/matched/extra = 9 / 8 / 0
env30 train0601 q3 final/matched/extra = 28 / 26 / 0
v2 official-val q3 final/matched/extra = 9 / 3 / 6
v2 train0601 q3 final/matched/extra = 29 / 10 / 19
geom1 v1 q1 official-val extra = 6
```

The remaining risk is that q4/q5 may become the new GT5 extra-lane carriers.
Historical proxy over the q1+q3-protected selected qset still has extras in
neighboring trained runs:

```text
geom1 v1 proxy q4/q5/q6/q7/q8/q10 extra = 10
v2 proxy q4/q5/q6/q7/q8/q10 extra = 10
```

So q1+q3-protected is a diagnostic training candidate, not a promotion. Its
gate must require official-val ACC/FN/count-shape at least env30-level and
train0601 final-query extras not moving from q3/q1 into q4/q5.

The candidate has now been materialized as default-off local artifacts:

```text
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-ultrashort-dataref-q1q3protected-v1.yaml
bank = data/gcs_reference_banks/q12_ultrashort_env30_train_q1q3protected_v1.json
audit = data/gcs_reference_banks/q12_ultrashort_env30_train_q1q3protected_v1_reference_audit.json
script = scripts/run_query_alpha05_env30_q12_ultrashort_q1q3protected_dataref_geom1_extraguard_v1.sh
```

This does not change the risk interpretation. The remote training preflight
must keep `ENFORCE_PRETRAINED_GATE=1`, `ENFORCE_REFERENCE_GATE=1`, and
`RUN_TESTS=0`. The final-query extra gate must watch q4/q5, not the old v2
q5/q6 proxy. A q4/q5 sixth-lane shift is an early rejection signal.

### q1/q3-Protected Training Rejection

The remote diagnostic run
`query_alpha05_env30_q12_ultrashort_q1q3protected_dataref_geom1_extraguard_v1`
has now been rejected before TEST. It was stopped with `TERM` after 18 complete
`results.csv` rows, while epoch 19 was running.

Official-val progression:

```text
epoch005 ACC/FP/FN = 0.932141 / 0.151010 / 0.086088
  count_acc_3/4/5 = 0.717489 / 0.575758 / 0.959459
  GT4 4->5 = 25, GT5 5->4 = 3

epoch010 ACC/FP/FN = 0.945669 / 0.141873 / 0.059917
  count_acc_3/4/5 = 0.757848 / 0.727273 / 0.094595
  GT4 4->5/4->6 = 8 / 10, GT5 5->6 = 65

epoch015 ACC/FP/FN = 0.949375 / 0.113682 / 0.047980
  count_acc_3/4/5 = 0.726457 / 0.681818 / 0.959459
  GT4 4->5 = 21, GT5 5->4 = 3
```

Env30 reference gate:

```text
env30 final official-val ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
env30 final GT4 4->5 = 0
env30 final GT5 5->4/5->6 = 1 / 0
```

Epoch 15 sweep rescue is ruled out:

```text
sweep rows = 864
rows with ACC >= env30 0.973330 = 0
rows with FN <= env30 0.009642 = 0
rows with GT4 4->5 = 0 = 0
rows passing count-shape gate
  (4->5=0, 4->6=0, 5->4<=1, 5->6=0, FN<=0.009642) = 0

best max_det=6 row ACC/FP/FN = 0.947559 / 0.133976 / 0.047291
best max_det=6 row includes GT4 4->5/4->6 = 12 / 9
best max_det=6 row includes GT5 5->6 = 69
```

Supported bottleneck:

- Protecting q1/q3 preserves their default references but does not prevent the
  replacement family from producing unstable official-val count shape after
  training.
- The selected `max_det=5` epoch015 row hides the worst GT5 sixth-lane surface
  but creates unacceptable GT4 false fifth behavior (`4->5=21`) and high FP/FN.
- The `max_det=6` surface shows the same extra-lane failure class as the v2
  line, with large GT4 and GT5 overcount.
- Follow-up final-query diagnostics confirm this is a distributed clear-far
  over-count failure, not a q4/q5-only transfer. Under selected `max_det=5`,
  q1/q3-protected produced 82 over-count images versus env30's 7 on the same
  valid-before-maxdet diagnostic, with `84/94` extra top-k lanes marked
  nearest-GT far. Under diagnostic `max_det=6`, it produced 151 over-count
  images, `156/173` nearest-GT far extra top-k lanes, and GT5 `5->6` carriers
  spread over q1/q6/q5/q11/q8/q9.
- Internal `val/f1` reached `0.951327` by epoch18, but internal val cannot
  override three failed official-val intervals and a zero-row official sweep
  rescue check.

Smallest safe next action:

Do not continue this run, do not run TEST, and do not start another blind
reference-bank training run from this family. The completed carrier diagnostic
supports rejecting q1/q3-protected and returning to env30 as the reference.
No implementation change has >=90% support from this failed diagnostic alone;
the next useful work should be broader bottleneck diagnosis rather than another
blind ultrashort reference-bank swap.

## 2026-07-14 Tiered GT4/GT5 Full-Protocol v2 Bottleneck

The completed run
`query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v2`
is rejected as a promotion. It confirms that the full tiered GT4/GT5 rescue
line is not the right path to push TEST ACC above `0.97`.

```text
v2 official_best val:
ACC/FP/FN = 0.973049 / 0.014141 / 0.008724
count_acc_3/4/5 = 0.982063 / 0.939394 / 0.918919
GT4 4->3/4->5 = 2 / 2
GT5 5->4/5->5/5->6 = 1 / 68 / 5

env30 official_best val reference:
ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
count_acc_3/4/5 = 0.968610 / 0.969697 / 0.986486
GT4 4->3/4->5 = 2 / 0
GT5 5->4/5->5/5->6 = 1 / 73 / 0

v2 official_best reporting-only TEST:
ACC/FP/FN = 0.966657 / 0.030799 / 0.024203
count_acc_3/4/5 = 0.967816 / 0.600427 / 0.775044
GT4 4->3/4->5/4->6 = 103 / 76 / 8
GT5 5->3/5->4/5->5/5->6 = 15 / 63 / 441 / 50

env30 official_best reporting-only TEST reference:
ACC/FP/FN = 0.966780 / 0.028732 / 0.023544
count_acc_3/4/5 = 0.964943 / 0.598291 / 0.891037
GT4 4->3/4->5 = 98 / 90
GT5 5->3/5->4/5->5 = 19 / 43 / 507
```

Supported bottleneck:

- The intended v2 parameters were active: env30 boundary mask, tiered
  GT4/GT5 geometry rescue, GT4 point-valid rescue, and GT5 point-valid rescue.
  The failure is not a wrapper/argument mistake.
- The run cannot be rescued by validation threshold selection. Its cached
  official-val sweep has `864` rows, maximum `ACC=0.973049`, and zero rows at
  or above env30's `0.973330`. No `ACC>=0.973` row removes GT4 `4->5`.
- The official-val tradeoff is worse than env30 for count shape: GT4 false
  fifth lanes reappear and GT5 sixth-lane predictions reappear under
  `max_det=6`.
- TEST remains below `0.97` because short-visible GT4/GT5 candidate quality is
  still weak. Raw-Q12 TEST diagnostics report GT4 short p90 APE
  `46.765210 px`, match20 `0.456897`, point-valid recall@0.6 `0.411905`;
  GT5 short p90 APE `53.893844 px`, match20 `0.658996`,
  point-valid recall@0.6 `0.738704`.
- The filter trace reports `180` TEST images where candidate count is already
  below GT count. This is upstream of final decode and is not solved by
  changing `conf`, NMS, `max_det`, or `min_points`.

Smallest safe next action:

Return to env30 as the reference and stop this loss-weight escalation line.
Do not increase GT4 point-valid rescue, GT4 short-geometry weights, GT5
mid-short weights, or boundary pseudo-negative pressure. The next serious
experiment should change candidate geometry coverage for train/official-val
short GT4/GT5 lanes, for example through a hardset-driven reference/query
coverage redesign, and must pass raw candidate gates before any reporting-only
TEST run.

## 2026-07-13 Tiered GT4/GT5 Rescue Bottleneck

The completed full-protocol run
`query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v1`
is rejected as a promotion and should not be used as the next starting point.

```text
tiered official_best val:
ACC/FP/FN = 0.971182 / 0.023370 / 0.012397
GT3 3->4/3->5 = 12 / 0
GT4 4->3/4->5 = 0 / 4
GT5 5->4/5->5 = 1 / 73

tiered official_best reporting-only test:
ACC/FP/FN = 0.966118 / 0.034280 / 0.023934
GT3 3->4/3->5 = 48 / 20
GT4 4->3/4->5 = 87 / 114
GT5 5->3/5->4/5->5 = 17 / 47 / 505

env30 official_best reporting-only test reference:
ACC/FP/FN = 0.966780 / 0.028732 / 0.023544
GT3 3->4/3->5 = 50 / 8
GT4 4->3/4->5 = 98 / 90
GT5 5->3/5->4/5->5 = 19 / 43 / 507
```

Supported bottleneck:

- The tiered rescue fails on official-val before considering TEST. Across the
  864-row official_best val sweep, no row reaches `ACC >= 0.973`, and no row
  matches env30 on both FP and FN.
- The GT4/GT5 weak-positive rescue is too broad. It reduces some GT4
  undercount but increases false extra lanes; TEST GT4 `4->5` rises by `+24`
  versus env30 and GT3 `3->5` rises by `+12`.
- The intended GT5 short geometry improvement did not occur. Official-val
  GT5 visible<=10 raw p90 APE worsens to `51.347903 px` from env30's
  `36.550196 px`, while `has_match20` stays at `0.754717`.
- TEST raw-Q12 diagnostics show persistent candidate-quality failure for
  short-visible lanes: GT4 visible<=10 has p90 APE `61.844274 px` and
  point-valid recall@0.6 `0.379396`; GT5 visible<=10 has p90 APE
  `65.861592 px` and point-valid recall@0.6 `0.698599`.
- The TEST gap to `0.97` is not a pure count-estimation or decode-threshold
  problem. The model needs better raw candidate geometry/validity for true
  short GT4/GT5 lanes while reducing GT3/GT4 extra-lane pressure.

Smallest safe next action:

Return to env30 as the reference. Do not increase GT4 point-valid rescue,
GT4 short-geometry weights, or GT5 mid-short weights. The next trainable
candidate should either be a smaller GT5-only ultra-short rescue with GT4
rescue disabled, or a hardset-driven candidate-geometry coverage experiment
for train/val short GT4/GT5 lanes. Any follow-up must pass official-val gates
before another reporting-only TEST run.

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

Do not increase boundary pseudo-negative pressure. Keep the env30 mask-v2
settings as the boundary mask guard:

```text
gcs_boundary_pseudo_neg = 0.02
gcs_boundary_pseudo_dist_thr = 80
gcs_boundary_pseudo_min_valid = 4
gcs_boundary_pseudo_score_thr = 0.2
gcs_boundary_pseudo_envelope_margin_px = 30
gcs_boundary_pseudo_envelope_ratio_thr = 0.75
```

The previous recommendation to run the tiered GT4/GT5 + GT4 point-valid rescue
has been executed and rejected by the 2026-07-13 full-protocol result. Do not
use that rejected setup as the next starting point.

The current smallest safe action is to return to env30 as the reference and
avoid GT4 point-valid rescue, GT4 short-geometry inflation, and GT5 mid-short
inflation. If another training run is justified, use either a smaller GT5-only
ultra-short rescue with GT4 rescue disabled, or a hardset-driven
candidate-geometry coverage experiment for train/val short GT4/GT5 lanes.
Promotion gates remain: beat or tie env30 official-val ACC/FN, do not regress
GT4 `4->5`, GT5 `5->4`, or GT5 `5->6` under `max_det=6`, improve GT5
visible<=10 raw geometry versus p90 APE `36.550196 px`, and reduce train
GT4->5 below `38`. Keep official test closed until those official-val and
train/val diagnostics pass.

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
in the active `86c8fb31c` rollback code:

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
in the active `86c8fb31c` rollback code:

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
the active `86c8fb31c` rollback code:

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
