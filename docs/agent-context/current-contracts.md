# Current Contracts

## 2026-08-11 Lane-Instance-Set Controlled Candidate Contract

The active baseline remains env30 commit
`86c8fb31cb4b48a53086be183478a95b0807753d`; the default model
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`, Q12/K56 fixed-y contract,
5-25-3 algorithm body, aux mask/edge outputs, and official evaluation
protocol are unchanged.

The lane-instance-set work is a controlled, default-off candidate enabled only
by the independent YAML:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-lane-instance-set-decoder.yaml
```

This YAML keeps `gcs_mode=query` and `Q=12/K=56`, but enables the
lane-instance-set decoder head. It switches that experiment model's main
query outputs to the lane-instance branch:

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
```

The lane-instance branch is image-local and does not register, execute, or
consume the legacy Transformer query decoder, `base_points`, `base_logits`,
or independent `base_valid_logits`. Its Q=12 dimension is candidate capacity
from image-local P2/P3 lane seeds, not the legacy coarse-query carrier path.
Its compatibility `pred_valid_logits` is always the deterministic contiguous
field derived from predicted start/end distributions; the old free-form
point-valid head has no influence on lane-instance geometry, matching, loss,
survival utility, or decode.

and additionally emits:

```text
pred_lane_instance_points: B x 12 x 56 x 2
pred_lane_instance_start_logits: B x 12 x 56
pred_lane_instance_end_logits: B x 12 x 56
pred_lane_instance_valid_logits: B x 12 x 56
pred_lane_instance_interval_start: B x 12
pred_lane_instance_interval_end: B x 12
pred_lane_instance_identity: B x 12 x D  # D=16 in the current YAML
pred_lane_instance_pair_duplicate_logits: B x 12 x 12
pred_lane_instance_novelty_logits: B x 12
pred_lane_instance_left_logits: B x 12 x 12
pred_lane_instance_right_logits: B x 12 x 12
pred_lane_instance_geometry_quality_logits: B x 12
pred_lane_instance_survival_logits: B x 12
pred_lane_instance_empty_logit: B
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

The default query YAML must not emit `pred_lane_instance_*` tensors and must
not register lane-instance-set parameters. The lane-instance candidate must
not emit `pred_count_logits`; its decoded lane count is derived from selected
survivors only. Do not confuse `pred_lane_instance_survival_logits` with any
later mainline Survival Head.

For this candidate, `pred_lane_instance_survival_logits` is the single learned
decode utility. Its model-side features include geometry quality, continuous
visibility/span quality, candidate novelty, predicted duplicate risk, and
topology confidence. Quality/novelty/topology tensors remain auxiliary
training and diagnostic outputs; decode must not multiply, rerank, or make a
parallel keep/count decision from them. Decode performs exact global subset
optimization over the eligible Q=12 candidates, maximizing summed unified
survival utility subject to predicted duplicate conflicts and the TuSimple
five-lane safety cap, then derives count from the surviving lane list.

The prediction-only decode path is selected by:

```text
--decode-mode lane_instance_set
```

or by `--decode-mode auto` only when the loaded model actually has
`lane_instance_set_decoder_head=true`. Lane-instance-set decode uses only
model predictions. It accepts `conf`/`point_valid_thr`/`min_points`/`max_det`
plus the lane-instance parameters recorded in `commands.md`, suppresses
duplicates by predicted duplicate probability, caps `max_det` at TuSimple's
`5`, and never backfills or fabricates lanes when fewer than
`min_survivors` survive. `min_survivors` is diagnostic-only. `allow_empty`
defaults false for TuSimple experiment decode. `--oracle-count` and
count-aware top-k are forbidden with `lane_instance_set`.

The default-off composite loss is controlled by:

```text
gcs_lane_instance_set = 0.0
```

When enabled, it requires all `pred_lane_instance_*` tensors and logs only
diagnostic/default-off lane-instance terms such as
`lane_instance_set_loss`, visibility/start/end/contiguity/span/empty/quality/
survival/duplicate/novelty/topology/identity losses, `lane_instance_set_noop_loss`,
and `lane_instance_match_count`. Their presence is not official-val evidence.

Accepted evidence as of this entry is limited to code, shape, synthetic,
loss, decode, and cache contracts:

```text
tools/check_gcs_lane_instance_set_decoder.py = passed locally
tools/check_model.py --cfg default YAML --imgsz 544 960 = passed locally
tools/check_model.py --cfg lane-instance-set YAML --imgsz 544 960 = passed locally
```

This evidence verifies tensor shapes, fixed-y anchors, finite gradients,
real `GCSLoss` participation when `gcs_lane_instance_set > 0`, contiguous
start/end-derived validity, duplicate suppression, no low-score backfill,
absence of legacy Transformer/query/point-valid parameters in the candidate,
exact global unified-utility subset decode, cache replay key coverage, and
default-YAML non-emission. It is not ACC,
official-val, clean-val, cross-session, multi-seed, or TEST evidence.

Rejected evidence remains rejected: old residual replacement/listwise routes,
dense instance/ranker/assembler routes, full-lane proposal routes, local
segment/candidate routes, and query-only survival/valid residual routes do not
become active behavior because this YAML exists.

Unverified hypotheses are training convergence, learned candidate coverage,
set-survival calibration, clean-val transfer, train0313/train0601 or other
cross-session behavior, multi-seed stability, promotion-gate pass/fail, and
TEST. TEST is closed and this candidate has not reached any promotion gate.

## 2026-08-11 Anchor-local Query Valid Residual Rejected

The default-off query-valid-local ablation reuses the existing point-local
token contract (`query state + point index + coordinate + sampled image
feature`) and adds a zero-initialized `query_valid_local_delta_mlp`. With the
env30 checkpoint loaded, `pred_points`, `pred_logits`, and
`pred_valid_logits` remain bitwise unchanged. Frozen-base training exposes
only `33,025` trainable parameters. The optional unmatched-query identity
trust region and TuSimple-angle-adjusted `line_accuracy` assignment are also
default-off and do not change env30 behavior.

Four one-epoch fixed-decode official-val probes reject this representation:

```text
env30 baseline:
  canonical  0.973330 / 0.015748 / 0.009642
  clean      0.965456 / 0.034068 / 0.019743
  train0601  0.978208 / 0.014878 / 0.004675

local LR=1e-5:
  canonical  0.973335 / 0.015748 / 0.009642
  clean      0.965348 / 0.034757 / 0.020432
  train0601  0.978067 / 0.014878 / 0.004675

local LR=1e-4:
  canonical  0.968849 / 0.012902 / 0.010101
  clean      0.959131 / 0.029063 / 0.020432
  train0601  0.974800 / 0.010650 / 0.004268

local LR=1e-4, unmatched identity weight=4:
  canonical  0.972207 / 0.015748 / 0.010331
  clean      0.964482 / 0.034068 / 0.019743
  train0601  0.977729 / 0.014390 / 0.004675

official-line-accuracy match + identity weight=4:
  canonical  0.971584 / 0.015748 / 0.010331
  clean      0.963189 / 0.033517 / 0.019743
  train0601  0.977080 / 0.012602 / 0.003252
```

Do not extend these runs, repeat seeds, tune LR/identity weights, or run TEST.
The matched-query valid-protect oracle remains diagnostic evidence of a large
GT-assisted ceiling, but a shared free-form 56-anchor residual does not turn
that ceiling into stable prediction-only gains. The next admissible
representation must predict a contiguous visibility interval (start/end) with
an explicit no-change path, rather than independently perturbing all 56 valid
logits.

## 2026-08-11 Residual Stage-2h/2i/2j Listwise Sequence Rejected

The default-off residual replacement selector now supports an image-level
action space consisting of one no-op action plus every proposal-victim
replacement. The listwise YAML additionally emits:

```text
pred_residual_noop_logit: B
```

Stage-2h uses flat cross-entropy with exact TuSimple official-score-delta
oracle labels. The 12-epoch frozen run collapses to the majority no-op class:

```text
canonical: 363/363 no-op, 11/11 positive-action images missed
clean:     363/363 no-op, 11/11 positive-action images missed
train0601: 410/410 no-op, 21/21 positive-action images missed
```

Stage-2i sets the pre-registered positive-image weight to `32.0`. It removes
the no-op collapse but over-triggers harmful actions:

```text
canonical: ACC 0.973330 -> 0.972996, 40 actions, 5 positive, 14 harmful
clean:     ACC 0.965432 -> 0.963874, 33 actions, 3 positive, 13 harmful
train0601: ACC 0.978198 -> 0.977588, 108 actions, 10 positive, 46 harmful
```

Stage-2j separates action presence from conditional action identity with a
hierarchical presence BCE plus positive-image replacement CE. Its one-epoch
canonical-64 smoke already over-triggers and regresses:

```text
baseline     ACC/FP/FN = 0.971211 / 0.028906 / 0.015625
hierarchical ACC/FP/FN = 0.970443 / 0.032031 / 0.015625
```

Do not run the 12-epoch Stage-2j job, tune listwise positive weights or
no-op/action margins, sweep thresholds, repeat seeds, or evaluate TEST. The
residual replacement oracle ceiling is only `+0.000557` canonical,
`+0.000859` clean, and `+0.000308` train0601, so even a perfect selector is
not sufficient by itself to close the current TEST gap to `0.970000`.

The next admissible route is base-query set survival, not residual set update.
Before implementing another training head, measure a train/official-val-only
oracle that preserves geometrically official-match GT4/GT5 queries through
joint existence, point-valid/min-points, and top-5 competition against harmful
unmatched queries. The default env30 model, decode, Q12/K56 contract, and TEST
status remain unchanged.

## 2026-08-11 Residual Stage-2d..2g Replacement Sequence Rejected

The default-off residual replacement sequence is implemented without changing
the active env30 query contract or official decode. Stage-2d adds
`pred_residual_replace_logits: B x 8 x 12`; Stage-2f adds curve-sampled
`pred_residual_visual_token: B x 8 x 64`. All new losses, heads, diagnostics,
and decode experiments remain disabled by default.

Stage-2d/2e/2f improve train-side proposal ranking but do not generalize to the
clean GT5-short case. The prediction-only row-support score is the first signal
to move the only recoverable clean GT5-short proposal from replacement rank 7
to rank 1, but global row-support ordering regresses canonical and train0601
GT5 retrieval. A fixed rank-5 hybrid gate preserves canonical base-miss top5
`8/8`, raises clean base-miss top5 `3/4 -> 4/4`, and raises train0601
`23/25 -> 25/25`, but this APE20 retrieval gate is not aligned with the
TuSimple official metric.

Real official-set A/B with the Stage-2f checkpoint shows:

```text
canonical baseline ACC/FP/FN = 0.973346 / 0.015748 / 0.009642
fixed hybrid                  = 0.973346 / 0.016162 / 0.009642
clean baseline                = 0.965470 / 0.034068 / 0.019743
fixed hybrid                  = 0.965528 / 0.034894 / 0.019743
train0601 baseline            = 0.978198 / 0.014878 / 0.004675
fixed hybrid                  = 0.978187 / 0.016341 / 0.004675
```

The clean GT5 proposal previously treated as a strict APE20 rescue already has
per-image official `Accuracy=0.995536, FP=0, FN=0` before replacement, so its
replacement has zero official utility. Do not use strict APE20 retrieval as a
proxy for set-update promotion.

The diagnostic-only official-utility oracle evaluates every residual
proposal-victim replacement using GT only for analysis. It proves a real
prediction-set ceiling:

```text
canonical: ACC/FP/FN 0.973346/0.015748/0.009642
        -> oracle    0.973904/0.012902/0.007576, 11 positive replacements
clean:     0.965470/0.034068/0.019743
        -> 0.966344/0.029522/0.016758, 10 positive replacements + 1 addition
train0601: 0.978198/0.014878/0.004675
        -> 0.978506/0.012439/0.004065, 21 positive replacements
```

Stage-2g adds default-off TuSimple official-set-utility replacement targets via
`gcs_residual_replace_official_delta_scale` and related threshold/penalty
fields. The 12-epoch frozen run
`residual_proposal_replacement_stage2g_official_utility_probe12_v1` is rejected.
Every prediction-only replacement-score threshold from `0.01` through `0.99`
is at or below the no-replacement baseline on canonical, clean, and train0601;
for example canonical threshold `0.25` gives
`0.972735 / 0.015060 / 0.010331`. TEST was not run.

Active env30 behavior remains unchanged. The official-utility code and tools
are diagnostic/default-off evidence only and must not be connected to formal
decode or TEST without a new cross-session selector gate.

This file records the active contracts for branch `codex/5-25-3-k56`.

## 2026-08-10 Residual Proposal Stage-2b Relational Retrieval Rejected

The default-off Stage-2b implementation adds explicit residual-to-base and
residual-to-residual relation encoding before the existing identity and quality
heads. The added modules are `residual_relation_proposal_mlp`,
`residual_relation_base_mlp`, `residual_relation_base_pair_mlp`,
`residual_relation_proposal_pair_mlp`, and `residual_relation_norm`; the
diagnostic output `pred_residual_relation_token` has shape `B x 8 x 64`.
Official decode and the env30 Q12/K56 output contract remain unchanged.

The authoritative run is:

```text
residual_proposal_relational_stage2b_probe12_v1
```

It initialized from the accepted Stage-1b `last.pt`, trained only 26 newly
added relation/identity/quality tensors, and kept TEST, replacement, and
residual decode closed. The checkpoint audit found all 527 Stage-1b keys
bitwise unchanged, 26 added keys, no missing keys, and no freeze violations.
Validation remained `TP/FP/FN=1317/54/41` for every epoch.

The relational head improves canonical short base-miss quality top3 from
`7/17` to `8/17`, but the joint gate still fails:

```text
canonical short base-miss: oracle/exist top3/quality top3 = 8/17, 6/17, 8/17
clean GT5 short base-miss: oracle/exist top5/quality top5 = 1/6, 0/6, 0/6
train0601 short base-miss:  oracle/exist top3/quality top3 = 25/49, 22/49, 22/49
train0601 ordinary GT5:    exist top3/quality top3          = 67/1194, 62/1194
```

`best.pt` and `last.pt` produce identical registered retrieval results. Stage-2b
is rejected because it still cannot rank the clean GT5 strict candidate into
top5 and regresses ordinary train0601 GT5 quality top3. Stage-3 set replacement
and Stage-4 experimental decode remain unauthorized. TEST remains closed.

## 2026-08-10 Residual Proposal Stage-2 Identity/Quality Rejected

The default-off Stage-2 implementation adds `pred_residual_identity`,
`pred_residual_quality_logits`, and `pred_residual_base_novelty`, plus
identity pull/push, strict-rescue quality BCE, pairwise quality ranking, and
diagnostic diversity-aware top-k retrieval. It does not modify official decode
or baseline outputs when the residual YAML and flags are disabled.

Three 10-epoch probes were run from the accepted Stage-1b fix2 checkpoint:

```text
residual_proposal_identity_stage2_probe10_v1
residual_proposal_identity_stage2_probe10_v1_fix2
residual_proposal_identity_stage2_probe10_v1_fix3
```

All kept TEST, replacement, and official residual decode closed. The final
fix3 freeze audit preserved all 527 Stage-1b state keys exactly and added only
the eight identity/quality parameter tensors. Env30 validation remained
`TP/FP/FN=1317/54/41` for every epoch.

Stage-2 is rejected because the safe retrieval gate did not pass jointly.
Fix3 canonical short base-miss quality top3/top5 reached `7/17` and `8/17`,
but clean-val GT5 short base-miss stayed `0/6` even at top5, despite a strict
oracle of `1/6`. Train0601 short base-miss quality top3/top5 was `21/49` and
`22/49`, below the frozen existence ranking `22/49` and `23/49`. Identity
separation was also split-dependent rather than stable.

Therefore Stage-3 set replacement and Stage-4 experimental decode are not
authorized from these checkpoints. Reopening requires a new proposal-to-base
relational representation that places the clean GT5 strict candidate inside a
safe top-k without regressing train0601 critical retrieval or ordinary GT5.

## 2026-08-10 Env30 Residual Proposal Stage-1b Visibility Contract

The completed authoritative run is
`residual_proposal_visibility_stage1b_probe10_v1_fix2`, initialized from
`residual_proposal_env30_probe20_v1_fix1/weights/last.pt`. It enables the
default-off `gcs_residual_visibility_only` mode and trains exactly eight
parameter tensors belonging to `residual_proposal_valid_mlp`,
`residual_proposal_start_mlp`, and `residual_proposal_end_mlp`. Env30,
residual instance masks, K56 geometry, and residual existence remain frozen;
frozen state is excluded from EMA arithmetic updates. The final checkpoint
audit changed exactly those eight tensors and no other state key.

Stage-1b adds default-off matched-proposal losses controlled by
`gcs_residual_interval_valid_consistency` and
`gcs_residual_positive_span`. The consistency term aligns point-valid logits
with the differentiable probability that `start <= k <= end` in both gradient
directions. Positive-span preservation applies positive BCE only to matched
GT-valid anchors. Official decode, Q12/K56 baseline outputs, identity,
replacement, and TEST remain unchanged or disabled.

The strict point-valid full-span hit20 gate passed on all registered surfaces:

```text
canonical short base-miss = 8/17, GT5 = 6/12
clean-val short base-miss = 3/13, GT5 = 1/6
train0601 short base-miss = 25/49, GT5 = 21/38
```

Geometry remained exactly `9/17`, `4/13`, and `27/49`, while env30 validation
remained `TP/FP/FN=1317/54/41` for every epoch. This checkpoint passes the
Stage-1b representation gate but is diagnostic-only and is not consumed by
official decode. The next permitted phase is identity/quality-aware proposal
retrieval with decode and replacement still closed.

## 2026-08-10 Env30 Residual Proposal Stage-1 Diagnostic Reopen

The user explicitly requested a new default-off env30-preserving stage-1
diagnostic after reviewing the current root bottleneck. The only reopened
mechanism is the segmentation-first residual proposal head configured by:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-residual-proposal.yaml
```

It adds eight diagnostic proposals generated from proposal-specific P2 instance
responses and differentiable fixed-y row soft-argmax fitting. The added outputs
are `pred_residual_points`, `pred_residual_valid_logits`,
`pred_residual_exist_logits`, `pred_residual_start_logits`,
`pred_residual_end_logits`, `pred_residual_row_logits`, and
`pred_residual_mask_logits`. The env30 Q12/K56 outputs and official decode stay
unchanged. `gcs_residual_proposal=0.0` preserves baseline behavior.

The frozen-base training target is limited to GT4/GT5 short lanes with 3..10
visible anchors that the frozen env30 raw query set misses at 20px. GT is used
only for training target construction and diagnostics, never at inference or
decode. No residual proposal is consumed by official decode, and TEST remains
closed.

The completed probe `residual_proposal_env30_probe20_v1_fix1` proves partial
geometry capacity but fails the strict complete-proposal gate because
point-valid full-span prediction collapses. Do not proceed to identity,
replacement, official decode, or TEST from this checkpoint. The next allowed
step is a stage-1b visibility/interval-consistency probe that keeps the same
frozen env30 geometry representation and changes only residual visible-span
supervision.

## 2026-08-03 Env30 LineIoU Diagnostic Reopen

The user explicitly reopened one new env30-based diagnostic after the final
rollback: a default-off matched fixed-y masked LineIoU loss. This exception is
limited to `gcs_line_iou`, `gcs_line_iou_half_width_px`, and
`gcs_line_iou_short_min_points`, the default-off
`gcs_line_iou_valid_preserve` short GT4/GT5 point-valid preservation probe,
the default-off `gcs_line_iou_exist_survival` short GT4/GT5 matched-existence
survival probe, plus train-side stratified-fold and frozen GT4-short manifest
tooling. It does not reopen any rejected post-env30 head, query-count,
proposal, selector, refinement, or decode route.

The baseline contract remains unchanged when `gcs_line_iou=0.0`. When enabled,
LineIoU applies only to Hungarian-matched queries and GT-valid anchors, requires
at least `gcs_line_iou_short_min_points` valid anchors, and does not change the
matcher, existence targets, point-valid loss, Q12/K56 output contract, or
decode. The first probe must use official-val selection, keep TEST closed, and
must not be interpreted as an accepted gain before reproducible evidence.

When `gcs_line_iou_valid_preserve > 0`, it is allowed only with nonzero
`gcs_line_iou`. It adds positive-only point-valid BCE on Hungarian-matched
GT4/GT5 short lanes selected by GT-visible anchor count and detached matched
APE gates. It does not change the matcher, existence targets, normal
point-valid loss targets, Q12/K56 output contract, decode, NMS, or official
metrics. It is a mechanism probe to test whether LineIoU geometry gains can be
kept without reducing short-lane visibility survival; it is not an accepted
algorithm gain until official-val and train-side gates pass with TEST closed.

When `gcs_line_iou_exist_survival > 0`, it is allowed only with nonzero
`gcs_line_iou`. It reuses the same GT4/GT5 short-lane, GT-visible-count, and
detached matched-APE gates as `gcs_line_iou_valid_preserve`, but applies only
positive BCE to the Hungarian-matched query existence logit. It does not change
the matcher, normal existence targets, point-valid targets, Q12/K56 output
contract, decode, NMS, thresholds, or official metrics. It is a narrow
mechanism probe for the LineIoU w0.5 seed1/seed3 count-survival instability
and is not eligible for TEST or long training until the canonical seed rescue
gates pass.

When `gcs_line_iou_geometry_only=true`, it is allowed only with nonzero
`gcs_line_iou` and only when both `gcs_line_iou_valid_preserve` and
`gcs_line_iou_exist_survival` are disabled. It freezes every parameter except
the fixed-y geometry point heads `point_mlp` and `point_refine_mlp`, keeps
frozen modules in eval mode, excludes frozen state from EMA updates, and writes
`line_iou_geometry_only_frozen_state_audit.json`. This probe is intended to
test whether the confirmed LineIoU raw-geometry gain can be retained without
directly updating existence, point-valid, decoder, backbone, auxiliary, dense,
or query-count parameters. It does not change model outputs, labels, matcher,
loss targets, decode, official metrics, or TEST status. Because fixed-y
point-valid refinement is coordinate-conditioned, geometry changes may still
indirectly alter point-valid logits; the probe is therefore not a guarantee of
score invariance and must pass count-safe multi-seed official-val gates before
any TEST discussion.

## 2026-08-05 Env30 Visibility-Only Diagnostic Reopen

The user explicitly reopened one additional default-off mechanism diagnostic:
`gcs_visibility_only`. It is limited to query-mode fixed-y models initialized
from the mature env30 checkpoint. When enabled, only
`point_valid_mlp` and `point_valid_refine_mlp` parameters are trainable;
geometry, existence, decoder, query embeddings, auxiliary heads, matcher,
labels, Q12/K56 outputs, and decode remain unchanged. Frozen BatchNorm
statistics remain in evaluation mode, and the optimizer excludes frozen
parameters.

This is a mechanism probe only. It must use canonical official-val selection,
`RUN_TESTS=0`, and a fixed env30 decode. It is not promotable unless
official-val ACC is at least the mature env30 reference, FP does not increase
by more than `0.002`, FN decreases, GT4/GT5-short point-valid recall improves,
and undercount does not increase. A failed probe closes this route; it does not
authorize any rejected post-env30 proposal or decoder route.

The visibility-only probe must lock every non-visibility parameter and buffer
in both the live model and `ModelEMA.ema`. Its optimizer must contain only
`point_valid_mlp` and `point_valid_refine_mlp` parameters, and
`frozen_state_audit.json` must record identical live/EMA frozen-state SHA256
values at setup, checkpoint save, and official-best update. Its script fixes
the env30 decode to `conf=0.001`, `point_valid=0.60`, `nms=0`, `max_det=5`,
`min_points=4`, and `valid_before_maxdet=false`; environment variables must
not change these values.

## 2026-08-07 Env30 Dense Endpoint-Peak Diagnostic Reopen

The user explicitly reopened one default-off dense endpoint localization
mechanism after the dense endpoint-mass/ranker rejection:
`gcs_dense_endpoint_peak_weight` with
`gcs_dense_endpoint_peak_radius_px`. This exception is limited to the existing
dense instance/keypoint YAML and loss. It does not connect dense tensors to
official decode, does not change Q12/K56 labels, does not change the default
env30 query outputs, and does not reopen dense candidate/ranker/test
promotion.

When enabled, the loss adds local softmax supervision around each GT bottom and
top endpoint in the corresponding dense endpoint channel. GT is used only for
training targets. The loss is default-off (`0.0`) and is intended to test the
confirmed bottleneck that canonical GT5 base-miss lanes have endpoint support
but no near endpoint peaks. The first probe must keep `gcs_dense_freeze_base`
enabled, keep `RUN_TESTS=0`, use canonical official-val for selection, and
judge mechanism success by endpoint/pairing diagnostics before any official
decode discussion.

## 2026-08-03 Final Env30 Rollback

The active source/config/script/tool/reference-bank payload is restored to
commit `86c8fb31cb4b48a53086be183478a95b0807753d` (`Add GT4 GT5 weak geometry
rescue run`), the original env30 baseline.

Every pre-existing commit after `86c8fb31c` is rejected as an active
experiment. The only later exception is the narrowly scoped default-off
LineIoU diagnostic above. This rejection includes staticref and
valid-neg variants, near20 geometry refine, Q20/Q24 routes, dual-head and
query-extent work, short local-refine, lateral/gated candidates, local-segment
v4-v13, full-lane proposals, dense-instance proposals, selector/decode
changes, and their diagnostics.

Precedence: this section overrides every lower section that describes a
post-`86c8fb31c` mechanism as active, reopened, ready for a next experiment,
or available for launch. Those sections remain only as historical rejection
evidence. They do not define current CLI flags, YAMLs, model outputs, losses,
decode behavior, launch commands, or recommended next actions.

## 2026-08-01 v10 Matched-Assignment Follow-up

The v10 default-off selector path adds
`gcs_short_segment_matched_assignment`. When enabled, short-segment
base-choice targets must use the Hungarian query-to-GT indices instead of
cross-GT query protection. The intended launch path is:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v10.yaml
scripts/run_query_local_segment_env30_frozen_probe20_v10.sh
```

The completed run
`query_local_segment_env30_frozen_probe20_v10_b4w0s1` is not valid evidence
for the matched-assignment change: its `args.yaml` and trainer log do not
contain `gcs_short_segment_matched_assignment: true` or
`--gcs-short-segment-matched-assignment`. The server script was updated after
that run. Do not promote it as the v10 matched-assignment result.

The run is still useful as a base-choice diagnostic. With `last.pt`, hard
selected-gated coverage was:

```text
official-val short GT5: base/raw/selected = 40/52/43 of 53
train0601 short GT5:    base/raw/selected = 142/179/151 of 183
official-val short GT4: base/raw/selected = 1/4/4 of 8
train0601 short GT4:    base/raw/selected = 9/19/16 of 19
```

Base-hit loss was zero for these `last.pt` groups, but the oracle candidate
was still rank-1 in only `0/53` official-val and `8/179` train0601 short GT5
cases. The raw geometry headroom therefore remained mostly unconverted.
`TEST` stayed closed and `candidate_decode` stayed false.

The original v10 official sweep omitted the env30 baseline threshold
`conf=0.001`. A same-protocol official-val sweep of final `last.pt` at
`conf=0.001`, `point_valid_thr=0.6`, `nms_dist_px=0`, `max_det=5`,
`min_points=4` returned `ACC=0.973346`, `FP=0.015748`, `FN=0.009642`.
This matches the env30 baseline and shows that the frozen base path itself
did not materially regress. A valid v10 matched-assignment run must verify
the flag in `args.yaml` before training results are interpreted.

The valid matched-assignment run
`query_local_segment_env30_frozen_probe20_v10_matched_b4w0s1` completed all
`20/20` epochs with the required flags:

```text
gcs_short_segment_matched_assignment = true
gcs_short_segment_base_choice_weight = 1.0
gcs_short_segment_freeze_base = true
gcs_candidate_decode = false
TEST = closed
```

Under the matched run's official-val sweep grid, base-only selection was:

```text
env30 baseline: epoch-selected ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
matched epoch 5 official_best: 0.972568 / 0.017585 / 0.011019
matched epoch 20 last:          0.972501 / 0.017126 / 0.011019
```

The matched sweep grid omitted the env30 reference point
`conf=0.001, point_valid_thr=0.6`, so these rows are not a strict
same-protocol base-regression comparison. They show no formal ACC gain in
the matched run's own sweep and are not sufficient to promote the base path.

The hard GT diagnostic did improve when the hard-gate-selected
`segment_best.pt` (a copy of `last.pt`) was used:

```text
official-val short GT5: base/raw/selected-gated = 40/52/47 of 53
train0601    short GT5: base/raw/selected-gated = 142/179/156 of 183
official-val short GT4: base/raw/selected-gated = 1/4/3 of 8
train0601    short GT4: base/raw/selected-gated = 9/18/15 of 19
```

Base-hit losses for these four groups were `0`, `1`, `0`, and `0`.
The oracle exact-score rank is still low inside the matched query:
short GT5 oracle score top-1/top-5 is `2/53` and `9/53` on official-val,
and `7/183` and `30/183` on train0601. Therefore matched assignment fixes
part of the target-conflict problem and converts useful raw capacity, but
does not solve exact proposal ranking. The result is diagnostic-only until
an official-val decode sweep proves a formal gain. Keep both
`gcs_candidate_decode=false` and `segment_decode=false` by default and keep
TEST closed.

## 2026-08-01 v10 Segment Decode Sweep

The v10 local-segment tensors use a separate default-off evaluation path:

```text
--segment-decode
--segment-score-thr 0.5
--segment-short-min-points 3
--segment-short-max-points 10
--segment-pred-valid-overlap-min 0
```

`segment_decode` is prediction-only and mutually exclusive with the older
`candidate_decode` and count-aware top-k paths. It treats the frozen env30
base geometry as a fixed zero-logit option, selects the highest-scoring
length-valid `pred_short_segment_*` window per query, and replaces base only
when the segment probability reaches the configured threshold. Replaced
queries use the selected window mask for point-valid filtering; query
existence logits, Lane-NMS, and official metrics are otherwise unchanged.
The default query decode and training-time decode remain unchanged.

The previous candidate sweep command failed because it requested
`--candidate-decode` for a v10 checkpoint that emits
`pred_short_segment_points/logits/window_mask`, not
`pred_short_candidate_points/logits`. The evaluation adapter was corrected,
locally and remotely compiled, and covered by
`tools/check_gcs_short_segment_head.py`.

Same-checkpoint official-val evidence using
`segment_best.pt` and the env30 reference point
`conf=0.001, point_valid_thr=0.6, nms_dist_px=0, max_det=5, min_points=4`:

```text
segment_decode=false: ACC/FP/FN = 0.973346 / 0.015748 / 0.009642
segment_score_thr=0.5: ACC/FP/FN = 0.973117 / 0.015886 / 0.010560
segment_score_thr=0.6: ACC/FP/FN = 0.973276 / 0.015197 / 0.009642
segment_score_thr=0.7: ACC/FP/FN = 0.973276 / 0.015197 / 0.009642
segment_score_thr=0.8: ACC/FP/FN = 0.973346 / 0.015197 / 0.009642
segment_score_thr=0.9: ACC/FP/FN = 0.973346 / 0.015748 / 0.009642
```

The full official-val grid's best segment row is
`ACC/FP/FN=0.973117/0.015886/0.010560`, below the same-grid base row
`0.973346/0.014509/0.009642`. The `0.8` row ties ACC but lowers
`count_acc_5` from `0.986486` to `0.972973`; `0.9` is effectively the base
path. The raw/hard coverage is therefore not sufficient to promote formal
segment decode. Keep `segment_decode=false`, do not run TEST, and do not
continue threshold tuning on this checkpoint.

## Branch Scope

This branch imports the historical `5-25-3.zip` algorithm and changes only the TuSimple fixed-y contract:

```text
legacy archive: Q=8, K=32, fixed_y=[0.98, 0.25]
this branch:    Q=12, K=56, fixed_y=[710/720, 160/720]
```

Do not silently import later mainline mechanisms such as Count Head, Quality Head, Survival Head, or near-miss mining into this branch unless a future task explicitly asks for that algorithm change. The branch now includes the 2026-06-27 user-requested, default-off `count_boundary_loss` for adjacent GT3/GT4/GT5 count-score boundaries; this is not a Count Head or decode change.

## 2026-07-28 Env30 Baseline Boundary (Superseded Record)

Active source/config is restored to env30 commit
`86c8fb31cb4b48a53086be183478a95b0807753d` (`Add GT4 GT5 weak geometry rescue
run`). Outside documentation, tracked code/config/script/tool/reference-bank
content must match that env30 baseline.

All commits after `86c8fb31c` are rejected experiment records unless a future
task explicitly reopens one with new official-val/train-side gates. This
rejects post-env30 staticref/valid-neg/near20 follow-ups, Q20/Q24
protected-static or dual-head routes, Q12 dual-head, query extent, short
local-refine, lateral candidate, gated candidate, Q24 role/event containment,
and candidate gate fixes as active code. Historical sections below are kept
only to explain the rejection evidence; their scripts, YAMLs, tools, outputs,
and launch commands are not active branch behavior.

The 2026-07-06 user-requested query-mode explicit Count Head is active only as
a default-off optional query ablation. It is enabled only by the dedicated
query-count YAML and emits `pred_count_logits: B x 4` for the fixed 2/3/4/5
lane-count classes. The default query YAML still emits no `pred_count_logits`,
and ordered-slot keeps its existing count/slot logic unchanged.

## 2026-07-28 Rejected Historical Q12/env30 Gated Candidate v2

The user temporarily reopened the lateral candidate-generation route on
2026-07-28 with new gates. The 2026-08-03 final env30 rollback cancels that
reopening. Everything in this section is historical evidence only.

The default query YAML and default decode remain unchanged. The v2 head is
enabled only by:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-gated-candidate-v2.yaml
```

When enabled, the query head adds:

```text
pred_short_candidate_points: B x 12 x 7 x 56 x 2
pred_short_candidate_logits: B x 12 x 7
pred_short_candidate_offsets_px: 7
```

The seven fixed lateral offsets are:

```text
[0, -20, +20, -40, +40, -60, +60]
```

The 2026-07-28 dense-offset follow-up is a separate default-off v3 YAML:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-gated-candidate-dense13-v3.yaml
```

It keeps the same frozen-env30/base-path and candidate decode contract, but
raises the candidate count from `7` to `13`:

```text
pred_short_candidate_points: B x 12 x 13 x 56 x 2
pred_short_candidate_logits: B x 12 x 13
pred_short_candidate_offsets_px: 13
offsets = [0, -10, +10, -20, +20, -30, +30, -40, +40, -50, +50, -60, +60]
```

The dense13 route exists because the 7-offset hard official-GT diagnostic only
improved raw oracle coverage from `40/53` to `43/53` on official-val short GT5
and from `142/183` to `148/183` on train0601 short GT5, while selected-gated
coverage stayed at base. TEST remains closed.

Training is controlled by default-off `gcs_short_candidate`. Candidate
assignment is limited to short GT4/GT5 lanes. APE `<=20px` candidates are
strong positives, `20..40px` candidates are soft positives with
`exp(-APE/tau) * visible_coverage`, and only far candidates with high current
selector score are used as negatives. Logs add:

```text
short_candidate_loss
short_candidate_score_loss
short_candidate_pull_loss
short_candidate_pos_count
short_candidate_soft_count
short_candidate_neg_count
```

Candidate decode is controlled separately by default-off `gcs_candidate_decode`
or the eval/sweep `--candidate-decode` flag. It is prediction-only gated by
predicted visible-anchor count and preserves the base query existence score.
It replaces geometry only for gated short queries and is mutually exclusive
with count-aware top-k. TEST remains closed until raw candidate coverage and
official-val gates pass.

The 2026-07-28 frozen-env30 probe is enabled only by
`--gcs-short-candidate-freeze-base` with the gated-candidate-v2 YAML. It loads
the env30 `weights/official_best.pt`, freezes all non-`short_candidate_*`
parameters, keeps frozen base modules in eval mode so BatchNorm statistics do
not drift, and trains only the candidate selector parameters. Its normal
`candidate_decode=false` official-val sweep must match env30 before candidate
decode is considered. The launch script is:

```text
scripts/run_query_gated_candidate_env30_frozen_probe20_v2.sh
```

## 2026-07-29 User-Requested Local Short-Segment Proposal v4

The local short-segment proposal route is a new explicit default-off v4
implementation following the hard oracle that showed contiguous `3..10`
h-sample segment capacity can cover the remaining short-lane misses. It is
enabled only by:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v4.yaml
```

The default query YAML and default decode remain unchanged. When enabled, the
query head adds:

```text
pred_short_segment_points: B x 12 x 404 x 56 x 2
pred_short_segment_logits: B x 12 x 404
pred_short_segment_x: B x 12 x 404 x 2
pred_short_segment_starts: 404
pred_short_segment_ends: 404
pred_short_segment_window_mask: 404 x 56
```

The `404` segment windows enumerate every contiguous fixed-y window with
length `3..10` anchors. Each proposal is valid only inside its own window and
predicts local start/end x as a bounded residual around the frozen base query
geometry. The main `pred_points`, `pred_logits`, matcher, normal losses, and
default decode are unchanged unless the v4 YAML and `gcs_short_segment > 0`
are explicitly selected.

Training is controlled by default-off `gcs_short_segment`. Assignment is
limited to short GT4/GT5 lanes, uses window-overlap APE with
`gcs_short_segment_min_overlap=3`, and trains scores from geometry quality
instead of ordinary existence labels. Logs add:

```text
short_segment_loss
short_segment_score_loss
short_segment_point_loss
short_segment_pos_count
short_segment_soft_count
short_segment_neg_count
```

The frozen-env30 probe is enabled by
`--gcs-short-segment-freeze-base`, which freezes all non-`short_segment_*`
parameters and keeps frozen base modules in eval mode. TEST remains closed.
The diagnostic gate must use the hard official-GT denominators and compare
base/raw-segment/selected-gated coverage before any decode promotion.

The v5 selector/gate follow-up is a separate default-off YAML:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v5.yaml
```

It preserves the v4 raw short-segment geometry and additionally emits:

```text
pred_short_segment_replace_logits: B x 12 x 404
```

The default v4 BCE score loss remains controlled by
`gcs_short_segment_bce_weight=1.0`. The v5 probe can explicitly set
`gcs_short_segment_bce_weight=0.0`,
`gcs_short_segment_listwise_weight>0`, and
`gcs_short_segment_replace_weight>0` so the selector learns listwise ranking
over `Q x 404` proposals plus a base-preserve replace gate. The hard
diagnostic uses proposal-local segment length and replace score when
`pred_short_segment_replace_logits` is present; formal/default decode still
does not consume short-segment tensors. TEST remains closed.

The v6 proposal-local selector follow-up is enabled only by:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v6.yaml
```

It preserves the v5 tensor contract, including:

```text
pred_short_segment_replace_logits: B x 12 x 404
```

and adds no new public prediction tensor. Internally, the v6 head sets
`short_segment_local_evidence=True`, samples image features at each proposal's
local start/mid/end points, appends proposal geometry evidence, and uses that
proposal-local token for both segment score and replace logits. The default
query YAML, v4 YAML, v5 YAML, and formal/default decode remain unchanged.
TEST remains closed until selected-gated hard diagnostics pass.

The completed `query_local_segment_env30_frozen_probe20_v6_b8s1` run is a
rejected diagnostic record. Its best hard-gate checkpoint (`last`) reaches only
official-val/train0601 short GT5 selected-gated `11/53` and `32/183`, while raw
local-segment capacity remains `52/53` and `179/183`. Do not promote its
`segment_best.pt`, run TEST, or enable formal short-segment decode from this
result.

The v7 dense-quality selector follow-up is enabled only by:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v7.yaml
```

It preserves the v6 public tensor contract and still emits no new formal decode
tensor. The v7 change is in the default-off loss/script protocol:

```text
gcs_short_segment_listwise_all_candidates = true
gcs_short_segment_dense_quality_weight > 0
gcs_short_segment_dense_neg_weight > 0
gcs_short_segment_replace_dense_neg_weight > 0
```

This trains dense geometry-quality targets over all `Q x 404` proposals,
including non-overlap/low-overlap negatives, and applies stronger no-replace
pressure when the env30 base query already hits the 20px gate. The v7 hard
diagnostic can use `segment_selection_score_mode=combined`, meaning it gates on
the same `score + replace` logit used for segment selection. Default query
decode remains unchanged and TEST stays closed until selected-gated hard
diagnostics pass on official-val plus train0601.

The completed `query_local_segment_env30_frozen_probe20_v7_b4w0s1` is rejected.
Its selected hard-gate checkpoint is `last.pt` and reaches only:

```text
official-val short GT5 selected-gated = 20/53
train0601 short GT5 selected-gated = 77/183
official-val short GT4 selected-gated = 2/8
train0601 short GT4 selected-gated = 9/19
```

The corresponding raw candidate capacity remains `52/53` and `179/183` for
short GT5, while base-hit losses are `21` and `70`. A diagnostic-only threshold
increase from `0.5` to `0.8` still reaches only `18/53` and `70/183`, with
base-hit losses `22` and `74`. Do not promote `segment_best.pt`, run TEST, or
enable formal short-segment decode from this run.

The v8 explicit two-stage selector follow-up is enabled only by:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v8.yaml
```

It preserves the v6/v7 local short-segment proposal geometry and proposal-local
candidate-quality score, disables the old per-candidate replace head, and adds:

```text
pred_short_segment_query_replace_logits: B x 12
```

The v8 default-off loss/script protocol separates the two decisions:

```text
gcs_short_segment_query_rank_weight > 0
gcs_short_segment_query_replace_weight > 0
gcs_short_segment_replace_weight = 0
segment_selection_score_mode = query_replace
```

Candidate scores rank the `404` segment windows inside each query. The
query-level replace head then decides whether that query may replace the env30
base geometry. Base-hit queries are trained as hard no-replace, and replace
positives are allowed only when the base misses the 20px gate and the query's
best candidate hits or clearly lowers APE. Default query decode remains
unchanged and TEST stays closed until hard selected-gated diagnostics exceed
the frozen env30 base on official-val and train0601 with near-zero base-hit
loss.

The completed `query_local_segment_env30_frozen_probe20_v8_b4w0s1` run is
rejected. It recorded only epochs `1..10` from the requested 20 epochs; no TEST
was used. Its base-decode official-val result remains below env30:

```text
official_best epoch5 ACC/FP/FN = 0.972582 / 0.017585 / 0.011019
last/best epoch10 ACC/FP/FN   = 0.972501 / 0.017126 / 0.011019
env30 baseline ACC/FP/FN      = 0.973330 / 0.015748 / 0.009642
```

The hard selected-gated gate does not exceed the frozen base:

```text
last/best short GT5 base/raw/selected-gated:
  official-val = 40/51/40 out of 53
  train0601    = 142/176/141 out of 183

official_best short GT5 base/raw/selected-gated:
  official-val = 40/52/40 out of 53
  train0601    = 142/179/142 out of 183
```

Do not promote `official_best.pt`, `best.pt`, `last.pt`, or any v8 diagnostic
checkpoint from this run; do not run TEST or enable formal short-segment decode
from this result. Any follow-up must change the selector target/decision,
preferably by making the env30 base geometry a first-class candidate in the
same per-query choice as the short-segment proposals.

The v9 base-choice selector follow-up is enabled only by:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-local-segment-proposal-v9.yaml
```

It preserves the v6/v7/v8 local short-segment proposal geometry and
proposal-local candidate-quality score, but emits no per-candidate replace
logits and no query-level replace logits. It adds no new public prediction
tensor beyond the v4 short-segment proposal contract. The v9 default-off
loss/script protocol is:

```text
gcs_short_segment_base_choice_weight > 0
gcs_short_segment_query_rank_weight = 0
gcs_short_segment_replace_weight = 0
gcs_short_segment_query_replace_weight = 0
segment_selection_score_mode = base_choice
```

The supervised v9 decision is a per-query softmax over
`[base, segment_0, ..., segment_403]`, with a fixed zero logit for `base`.
Base-hit queries are trained to choose base. A segment is a positive choice
only when the base misses the 20px gate and the segment hits or clearly lowers
APE. Default query decode remains unchanged and TEST stays closed until hard
selected-gated diagnostics exceed the frozen env30 base on official-val and
train0601.

Legacy post-env30 record: the 2026-07-25 user-requested Q12/env30 dual-head
probe was rejected and its YAML/script are not active after the 2026-07-28
env30 rollback. It added `pred_count_logits: B x 4` for image-level lane count
and `pred_quality_logits: B x 12` for query-level quality/ranking during that
experiment only.

Legacy post-env30 record: the 2026-07-26 user-requested Q12/env30 query
extent probe is rejected and its YAML/scripts are not active after the
2026-07-28 env30 rollback. It added `pred_start_logits: B x 12 x 56` and
`pred_end_logits: B x 12 x 56` for first/last visible fixed-y anchors during
that experiment only.

The completed 40-epoch extent probe `query_extent_env30_probe40_v1` is
rejected. Its official-val selected decode keeps `extent_decode=false` and
reaches only:

```text
official_best ACC/FP/FN =
  0.966997 / 0.052020 / 0.019972
count_acc_3/4/5 =
  0.856502 / 0.818182 / 0.986486
```

Post-train official-val sweeps show that `none` is better than both
`intersect` and `interval`. `interval` strongly increases GT3/GT4 false-extra
rates, while raw-geometry-stratified diagnostics show the remaining short
GT4/GT5 failures are mostly missing 20px raw candidates rather than endpoint
classification failures. Do not extend this exact setup, enable extent decode,
run TEST for it, or build local-refine v2 on top of it. The next route must
target coarse geometry/reference coverage first.

Legacy post-env30 record: the 2026-07-27 Q12/env30 short-lane local x-refine
v3 probe is rejected and its YAML/scripts/tools are not active after the
2026-07-28 env30 rollback. v3 started from the env30 `weights/official_best.pt`,
froze the env30/base parameters, and trained only the
`query_short_local_refine_*` head during that experiment only.

When enabled, the v3 head performs feature-conditioned horizontal local
window search around detached main `pred_points`. The default v3 window is:

```text
offsets_px = [-60, -40, -20, 0, 20, 40, 60]
gcs_short_local_refine_max_delta_px = 60.0
gcs_short_local_refine_window_radius_px = 60.0
gcs_short_local_refine_window_step_px = 20.0
```

The v3 head emits the v2 auxiliary tensors plus:

```text
pred_short_refine_window_logits: B x 12 x 56 x 7
```

`pred_points` remains the env30 main geometry for Hungarian matching, normal
losses, point-valid refinement, and official decode. The optional
`short_local_refine_loss` supervises only `pred_short_refined_points`.
With `gcs_short_local_refine_identity_guard=True`, matched short GT4/GT5 lanes
with coarse APE `<= gcs_short_local_refine_identity_thr_px` are supervised to
stay near the coarse x position, lanes in
`(identity_thr_px, gcs_short_local_refine_nearmiss_thr_px]` are pulled toward
GT, and farther misses are skipped by this auxiliary loss. Logs add:

```text
short_local_refine_loss20
short_local_refine_identity_count
short_local_refine_pull_count
```

The recommended first v3 probe uses `EPOCHS=10`,
`gcs_short_local_refine=0.05`, `gcs_short_local_refine_beta_px=3.0`,
`gcs_short_local_refine_freeze_base=True`, and
`gcs_short_local_refine_identity_guard=True`. TEST remains closed until
official-val and train-side raw/refined geometry gates pass.

The completed `query_short_local_refine_env30_window_v3_probe10` is rejected
for promotion. The frozen main path remains at the env30 raw geometry level,
and the auxiliary branch improves continuous APE mostly inside the 20px gate
without materially increasing short GT4/GT5 hit20:

```text
official-val raw/refined:
  short GT4 1/8 -> 2/8, gain=1, loss=0
  short GT5 40/53 -> 40/53, gain=0, loss=0
train0601 raw/refined:
  short GT4 9/19 -> 9/19, gain=0, loss=0
  short GT5 142/183 -> 142/183, gain=0, loss=0
```

The refined auxiliary p90 APE improves at 30-40px thresholds, but the
dominant misses remain far or absent raw carriers. The official decode still
uses `pred_points`, so the auxiliary refined points do not change formal
official metrics. The post-train official-val results are:

```text
official_best.pt ACC/FP/FN = 0.973365 / 0.015748 / 0.009642
best.pt          ACC/FP/FN = 0.973330 / 0.015748 / 0.009642
```

Reporting-only TEST with the frozen official-val decode gives:

```text
official_best.pt ACC/FP/FN = 0.966680 / 0.027924 / 0.023724
best.pt          ACC/FP/FN = 0.966684 / 0.027995 / 0.023724
```

Do not continue this exact 10-epoch v3 artifact to longer training, enable
refined decode, or use it for another TEST. The next experiment must first
test a prediction-only refined decode on official-val and, if that still
does not cross the 20px gate, change candidate generation rather than only
increase the local residual capacity.

Legacy post-env30 record: the superseded 2026-07-27 Q12/env30 short-lane
coarse-to-fine local x-refine v2 probe is rejected and its YAML/scripts/tools
are not active after the 2026-07-28 env30 rollback. It is retained only for
old-run interpretation and must not be used as a current next experiment.
When enabled, the head
emits:

```text
pred_points: B x 12 x 56 x 2                   # env30 main path for matcher/loss/decode
pred_coarse_points: B x 12 x 56 x 2            # diagnostic alias of the main path for gate compatibility
pred_short_refined_points: B x 12 x 56 x 2     # auxiliary bounded local x-refine output
pred_short_refine_delta_logits: B x 12 x 56    # v2 raw residual logits; v3 expected-offset proxy for compatibility
pred_short_refine_delta_norm: B x 12 x 56      # bounded normalized x residual
```

The auxiliary head samples image features at detached main points, detaches the
resulting refinement tokens, and only changes normalized x by `tanh(delta) *
gcs_short_local_refine_max_delta_px / image_width`; fixed-y anchors must stay
unchanged. To preserve env30 Q12 carrier assignment, `GCSLoss` matches
Hungarian assignments on the main `pred_points`, and normal point/smooth/curve/
exist/valid losses plus official decode also use the main `pred_points`. The optional
`short_local_refine_loss` supervises only `pred_short_refined_points` with
normalized-x SmoothL1, selected by short GT4/GT5 criteria
`gcs_short_local_refine_visible_thr` and
`gcs_short_local_refine_gt_min_lanes`. Historical v2 probes used
`gcs_short_local_refine=0.02`, `gcs_short_local_refine_beta_px=5.0`, and
`gcs_short_local_refine_max_delta_px=40.0`. Do not launch v2 for the current
next step; use the v3 window/freeze/identity probe instead.

Legacy post-env30 record: the 2026-07-27 Q12/env30 lateral
candidate-generation probe is rejected and its YAML/scripts/tools are not
active after the 2026-07-28 env30 rollback. It kept the env30 main
`pred_points` path unchanged and added fixed-y lateral hypotheses around each
query during that experiment:

```text
pred_short_candidate_points: B x 12 x 7 x 56 x 2
pred_short_candidate_logits: B x 12 x 7
candidate offsets: [0, -20, +20, -40, +40, -60, +60] px
```

The candidate score head is the only trainable head in the first probe,
initialized at zero so the base checkpoint is unchanged before training.
The first probe selected one highest-scoring lateral hypothesis per base
query and added its candidate-relative logit to the base query logit before
ordinary confidence filtering, NMS, and max-det truncation. That score-coupled
decode is rejected and retained only as historical failure evidence below.
Candidate decode is mutually exclusive with query extent decode and
count-aware top-k. Count Head, Quality Head, query extent loss/decode, and
TEST remain disabled for the first probe.
Official-val selection must first verify raw candidate `has_match20` gains
without increasing GT3/GT4 false-extra behavior.

The completed
`query_short_candidate_env30_probe20_fix1` is rejected. The frozen env30
main path is intact: with the same checkpoint and the same official-val
decode, `candidate_decode=false` returns the env30 result
`ACC/FP/FN = 0.973330 / 0.015748 / 0.009642`, while
`candidate_decode=true` reaches only
`0.946994 / 0.048714 / 0.040404` at epoch 5 and
`0.893279 / 0.120202 / 0.101469` at epoch 10. The official-best candidate
checkpoint is therefore not a TEST candidate and must not be continued.

The failure is in the candidate score/decode contract, not base geometry:

- training `short_candidate_best_hit20` rises to about `0.81`, but official-val
  stays at `0.35..0.39` while raw candidate coverage is `0.70238`;
- the score head is trained only on matched short GT4/GT5 lanes but candidate
  decode selects a hypothesis for every Q12 query, including normal GT3/GT4
  lanes and unmatched queries;
- candidate score evidence is an unweighted mean over all 56 anchors, while
  the target APE uses only visible anchors, diluting short-lane evidence;
- the selected relative logit is added to the base query score, so an
  out-of-distribution candidate score can change ordinary lane ranking.

Any future candidate-generation follow-up must keep the normal decode path
unchanged by default, add an explicit prediction-only short-lane applicability
gate, and use visibility-aware anchor aggregation. Do not tune learning rate,
candidate loss gain, or epoch count before those contract changes.

Legacy post-env30 record: the gated candidate-generation follow-up is also
rejected and is not the current candidate-generation path. It kept the env30
`pred_points`, query existence/quality score, visibility mask, NMS, and
max-det path unchanged for non-applicable queries. Candidate selection was
applied only when the prediction-only count of
`sigmoid(pred_valid_logits) >= gcs_candidate_gate_valid_thr` lies in
`[gcs_candidate_gate_min_visible, gcs_candidate_gate_max_visible]`, using
the default range `[2, 10]`. Candidate logits are aggregated with those
predicted visibility probabilities over fixed-y anchors, and
`gcs_candidate_preserve_base_score=true` keeps the original query/quality
score after selection. The candidate branch is therefore a geometry
hypothesis selector, not a new existence/ranking score. The first gated
probe must be judged by raw candidate `has_match20` coverage and
GT3/GT4 false-extra risk before any formal promotion or TEST.

The completed 40-epoch dual-head probe
`query_dualhead_quality_count_env30_probe40_v1` is not promoted to full
training or TEST. It is also not closed as a dead mechanism, because it beats
the env30 same-age epoch040 official-val ACC while remaining far below the
env30 epoch220 checkpoint:

```text
dual-head epoch040 official-val ACC/FP/FN =
  0.962494 / 0.025941 / 0.021579
env30 epoch040 official-val ACC/FP/FN =
  0.958436 / 0.065702 / 0.031221
env30 epoch220 official-val ACC/FP/FN =
  0.973330 / 0.015748 / 0.009642
```

The count head works (`count_acc=0.980716`), but the current 40-epoch
checkpoint is capped by lane candidate geometry/valid quality: GT-count
oracle-rank only reaches `ACC=0.963281`, and official-val raw
`has_match_20px` is `0.933998` versus env30 final `0.976209`. The next action
is a 100-epoch official-val-only undertraining diagnostic, not TEST or a blind
full 220-epoch promotion.

The 100-epoch dual-head diagnostic
`query_dualhead_quality_count_env30_probe100_v1` is rejected for continuation
to 220 epochs as-is. It improves raw candidate coverage, but fails the
official-val promotion gate:

```text
official-val ACC/FP/FN =
  0.966268 / 0.015335 / 0.015840
count_acc_3/4/5 =
  0.991031 / 0.939394 / 0.972973
GT-count oracle-rank ACC =
  0.966877
raw has_match20 = 0.959325
GT5 visible<=10 raw match20 = 0.754717
```

A single reporting-only TEST was run with the frozen official-val decode. It
is not promotion or tuning evidence and must not be rerun for this candidate:

```text
TEST ACC/FP/FN =
  0.963288 / 0.028481 / 0.031332
TEST count_acc/count_acc_4/count_acc_5 =
  0.877067 / 0.611111 / 0.776801
TEST count confusion includes:
  4->3=119, 4->5=62, 5->4=105, 5->5=442
```

Decision: do not complete 220 epochs on the current dual-head loss/decode
unchanged. The next candidate must keep the count head but change the quality
head training/decode dependency or otherwise restore the env30 geometry and
objectness path, then pass a fresh official-val/train-side gate. TEST remains
closed for subsequent selection.

Legacy post-env30 record: the user-requested Q24 dual-head follow-up is
rejected and its YAML/script are not active after the 2026-07-28 env30
rollback. It emitted `pred_count_logits: B x 4` and
`pred_quality_logits: B x 24` during that experiment only.

The branch also includes the 2026-06-27 user-requested, default-off `gcs_hard_sampling` train-only sampler for short-visible GT3/GT4/GT5 and 0601 samples. It changes only the training dataloader sampling frequency through `WeightedRandomSampler`; it does not change labels, validation/test dataloaders, point/smooth/curve losses, decode, or official metrics.

The branch also includes the 2026-06-27 user-requested, default-off `gcs_spurious_neg` loss for E3-lite. It uses the training Hungarian matcher indices only to select unmatched short duplicate-like queries near matched queries, then adds an extra target-zero BCE on their `pred_logits`. The GT-count weighting extension keeps the old default behavior with `gcs_spurious_gt3_weight=1.0`, `gcs_spurious_gt4_weight=1.0`, `gcs_spurious_gt5_weight=1.0`, and `gcs_spurious_disable_gt5=False`, while allowing GT3-or-sparser, GT4, and GT5-or-denser samples to carry different spurious-negative weights. The 2026-06-28 `gcs_spurious_gt_protect` extension is also default-off and only removes GT-close candidate queries from this extra negative BCE. It does not change data sampling, dataset labels, matcher logic, point/smooth/curve losses, decode, NMS, or official metrics.

Active source/config is restored to commit
`86c8fb31cb4b48a53086be183478a95b0807753d` (`Add GT4 GT5 weak geometry rescue
run`). All later commits are old state for this rollback and their mechanisms,
tools, commands, logs, and experiment results are rejected legacy records only
unless a future task explicitly re-enables them with fresh official-val/train
gates.

Therefore `tools/build_gcs_short_side_hardset.py`,
`tools/diagnose_gcs_short_side_hardset.py`,
`tools/diagnose_gcs_count_contract.py`, `gcs_short_side_geom`,
`gcs_far_spurious_neg`, `gcs_farspur_*`, `gcs_shortside_*`, `gcs_rank_*`,
`gcs_base_ignore_*`, and the GT4/GT5 count-contract diagnostics are not part of
the active code/config state. Historical sections below that mention those
names are retained only to interpret old runs.

Recent env30 follow-up experiments after `86c8fb31c` are rejected records
unless a future task explicitly reopens them with new official-val/train gates:

```text
query_alpha05_env30_gt45staticref_v2:
  rejected, official-val ACC/FP/FN = 0.971040 / 0.018825 / 0.012626
query_alpha05_env30_gt45staticref_v2_gt4only_validneg_w01:
  rejected, official-val ACC/FP/FN = 0.970965 / 0.024656 / 0.012626
query_alpha05_env30_gt45staticref_q7only_v3a:
  rejected before training, no TEST, GT4_short_gain20_q7only = 0
query_alpha05_env30_gt4_near20_geom_refine_v1:
  rejected, official-val ACC/FP/FN = 0.968943 / 0.035373 / 0.015152
```

Do not relaunch these as the next path, run TEST for them, tune thresholds from
them, or treat their default-off mechanisms as active improvement claims.

Legacy post-env30 record: the branch previously contained default-off Q20/Q24
protected dual-bank static-gate tooling paths for the 2026-07-23 env30 GT4/GT5
bottleneck diagnosis. These files are not active after the 2026-07-28 env30
rollback:

```text
Q20 model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q20-k56-dualbank.yaml
Q24 model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q24-k56-protected-static.yaml
Q20/Q24 tools:
  tools/build_q20_dualbank_static_gate.py
  tools/build_q24_protected_static_gate.py
mode: reference_mode=dualbank
```

These are rejected records and do not change the active Q12 baseline. They
were enabled only by dedicated Q20/Q24 YAMLs plus generated reference-bank
JSONs.
The first twelve references must preserve the default Q12 linear references;
queries `q12..q19` are Q20 extras and `q12..q23` are Q24 extras. The static
gate may use canonical official-val diagnostics and train0601/train0531 raw
diagnostics only. It must not read TEST, tune TEST, or publish a training bank
unless the strict gate passes. The strict gate includes GT5 coverage, GT4
coverage, Q12 reference preservation, and no increase in normal GT3/GT4
match20 risk across val/train0601/train0531. Failed banks under `.tmp/` are
diagnostic artifacts only. `GCSLaneHead` must reject dual-bank JSON whose
`formal_gate_passed` is not `true`, whose `test_used` is not `false`, whose
source splits include TEST, whose GT5/GT4 extra-query partition is malformed,
whose normal-risk gate is missing/failed, whose fixed-y/protected-query
metadata is malformed, or whose `q0..q11` references do not preserve the
default Q12 linear references.

Historical note: the Q24 bank passed the strict static gate on both local and
remote validation and was published only as a default-off training candidate
before the route was rejected:

```text
bank: data/gcs_reference_banks/q24_protected_static_env30_best.json
server summary: .tmp/q24_protected_static_gate_server/summary.json
selected allocation: 6:6
TEST used: false
val GT5 primary match20/p90: 0.500000 / 79.271370
train0601 GT5 primary match20/p90: 0.393443 / 61.043999
val GT5 impact match20/p90: 0.750000 / 64.845001
val/train0601/train0531 GT4 primary match20:
  0.625000 / 0.266667 / 0.714286
normal GT3/GT4 added match20 risk:
  val 0.0, train0601 0.0, train0531 0.0
```

Decision: Q24 may enter a 20-40 epoch remote probe selected only on
official-val. It is not promoted to full training or TEST by the static gate
alone.

Completed follow-up: the 40-epoch Q24 probe
`query_alpha05_env30_q24_protected_static_probe40_v1` is rejected for full
training. Its official-val ACC/FP/FN is
`0.961733 / 0.068457 / 0.023186` versus env30
`0.973330 / 0.015748 / 0.009642`; the external 864-row official-val sweep has
zero rows at or above env30 ACC and zero rows with FP/FN both at or below
env30. TEST was not used and must remain closed for this artifact. The failure
is post-train extra-query carrier drift and existence calibration, not static
gate capacity alone.

The 100-epoch Q24 continuation diagnostic
`query_alpha05_env30_q24_protected_static_cont100_v1` is also rejected for
full training. Its official-val ACC/FP/FN is
`0.964376 / 0.055096 / 0.016758`, with `count_acc_4=0.742424` and count
confusion including `3->4=19`, `3->5=12`, and `4->5=17`. The official-val
sweep has zero rows at or above env30 ACC and zero rows with FP/FN both at or
below env30. TEST was not used. This closes the simple undertraining
hypothesis: longer Q24 training helps but does not remove the wrong-count
extra-query carrier bottleneck.

The branch now includes a default-off Q24 role-containment probe for this
specific failure mode. It is enabled only by explicit training args such as
`--gcs-role-contain > 0` and/or `--gcs-role-contain-matcher` on the dedicated
Q24 protected-static YAML. It adds no decode changes, no official metric
changes, and no TEST usage. The intended Q24 partition is:

```text
GT5 bank: q12..q17
GT4 bank: q18..q23
```

When enabled, role-aware matching forbids `q12..q23` from matching GT3 lanes,
forbids GT5-bank matches on GT4 images, allows GT4-bank matches only to
short/weak-visible GT4 lanes, allows GT5-bank matches only to short/weak-visible
GT5 lanes, and forbids GT4-bank matches on GT5 images. The auxiliary
role-containment BCE adds extra exist/valid negative pressure only on GT3/GT4
role violations; it intentionally skips additional GT5-image containment
pressure to protect true fifth-lane retention. This is a 20-40 epoch probe
mechanism only until official-val/train-side gates prove reduced FP/overcount
without GT5 regression.

The completed role-containment probe
`query_alpha05_env30_q24_role_containment_probe40_v2` is rejected for full
training. TEST was not used. It confirms the code path is active and suppresses
the old broad GT3/GT4 extra-query raw carrier drift, but it does not pass the
official-val gate and regresses true GT5 short raw geometry:

```text
official-val ACC/FP/FN = 0.963152 / 0.051607 / 0.021350
count_acc_3/4/5 = 0.869955 / 0.803030 / 0.972973
val/train0601 GT5 visible<=10 raw match20 = 0.358491 / 0.262295
official-val sweep rows with ACC>=env30 = 0
oracle-rank ACC upper check ~= 0.9633, so decode/count/rank rescue is not enough
```

Decision: do not full-train or TEST this artifact. Any next Q24 probe must
first change the role partition or soften matcher constraints, using only
official-val/train-side gates.

The completed role-partition follow-up
`query_alpha05_env30_q24_role_partition_v3_probe40_v1` is also rejected for
full training. TEST was not used. It widened the GT5 bank to the observed
short-lane carriers, but this reintroduced severe GT5 overcount while true GT5
short raw coverage stayed weak:

```text
official-val ACC/FP/FN = 0.962991 / 0.079844 / 0.023416
count_acc_3/4/5 = 0.878924 / 0.818182 / 0.094595
count_confusion includes 3->4=23, 3->5=4, 4->5=9, 4->6=2, 5->6=67
val/train0601 GT5 visible<=10 raw match20 = 0.188679 / 0.229508
official-val sweep rows with ACC>=env30 = 0
GT-count oracle-rank ACC upper check ~= 0.9631
```

Decision: do not full-train or TEST this artifact. The failure is not a simple
partition-labeling issue: the same query IDs that are nearest to true GT5
short lanes also generate GT5 sixth-lane boundary/ambiguous outputs, and their
scores overlap with true short-lane scores. Any next Q24 work must first
separate true GT5 short geometry rescue from GT5 boundary-pseudo suppression
with event-level diagnostics and GT-safe protection.

The branch now includes a default-off GT5-safe boundary-pseudo suppression
probe for the Q24 failure above. It is enabled only by explicit args:

```text
--gcs-boundary-pseudo-gt5-safe
--gcs-boundary-pseudo-score-thr 0.0
--gcs-boundary-pseudo-envelope-margin-px >= 0
--gcs-boundary-pseudo-protect-short-visible-thr <N>
--gcs-boundary-pseudo-protect-dist-px <PX>
--gcs-boundary-pseudo-protect-min-overlap <N>
--gcs-boundary-pseudo-protect-queries <optional query ids>
```

When enabled, the boundary-pseudo negative loss still applies only during
training, only to unmatched short candidate lanes on GT5 images, and only after
the clear boundary/envelope checks pass. Before a candidate can become a
target-zero boundary-pseudo negative, the GT5-safe guard protects candidates
that overlap a true short GT5 lane and are within the configured mean x-distance
window. It does not change labels, matcher assignment, decode, NMS, official
metrics, model outputs, or the default Q12/Q24 behavior. Diagnostics add
`boundary_pseudo_candidate_count` and `boundary_pseudo_protected_count` to the
loss log.

The branch also includes diagnostic-only event-mined gate tooling:

```text
tools/diagnose_q24_event_mined_gate.py
scripts/run_q24_event_mined_gate_v1.sh
```

These consume official-val/train-side raw/extra diagnostic CSVs and write
per-query true GT5 short hit/near/miss counts versus GT5->6 and GT3/GT4
false-extra risks. They must not use TEST for candidate selection.

The completed GT5-safe boundary-pseudo probe
`query_alpha05_env30_q24_gt5safe_boundary_probe40_v1` is rejected for
longer/full training. TEST was not used. The mechanism was active, but the gate
failed:

```text
official-val ACC/FP/FN = 0.962156 / 0.089302 / 0.021120
count_acc_3/4/5 = 0.852018 / 0.712121 / 0.175676
external official_best.pt sweep rows with ACC>=env30 = 0
external official_best.pt sweep rows passing longer-training gate = 0
val/train0601 GT5 visible<=10 raw match20 = 0.301887 / 0.316940
val GT5->6 = 60
val GT3/GT4 false-extra = 51
```

Decision: do not full-train or TEST this artifact. The result shows that
GT5-safe boundary-pseudo suppression alone does not solve Q24's event-level
query-role separation and existence/valid calibration bottleneck.

The branch now includes a default-off Q24 event-containment probe for the
follow-up mechanism. It is enabled only by explicit args such as:

```text
--gcs-q24-event-contain > 0
--gcs-q24-event-matcher
--gcs-q24-event-clean-gt5-queries 12,15,20
--gcs-q24-event-risk-queries 13,21,22,23
--gcs-q24-event-gt4-queries 14,17,18,19
--gcs-q24-event-suppress-gt5-risk
```

When enabled, event-aware matching allows clean GT5 queries only on short GT5
lanes, forbids high-risk event queries from matching GT5 lanes, and confines
GT4 event queries to short GT4 lanes. The auxiliary event-containment BCE adds
extra exist/valid target-zero pressure on selected event-role violations. It
does not change labels, decode, NMS, official metrics, model outputs, or the
default Q12/Q24 behavior. It is a 20-40 epoch probe mechanism only until
official-val/train-side gates pass.

The completed event-containment probe
`query_alpha05_env30_q24_event_containment_probe40_v1` is rejected for longer
or full training. TEST was not used. The mechanism was active, but the
official-val and event-mined gates failed:

```text
official-val ACC/FP/FN = 0.961976 / 0.076860 / 0.025253
count_acc_3/4/5 = 0.834081 / 0.772727 / 0.527027
count_confusion includes 3->4=33, 3->5=4, 4->5=11, 4->6=3, 5->6=35
external sweep rows with ACC>=env30 = 0
external sweep rows with FP/FN both <= env30 = 0
external sweep rows with count_acc_4>=0.90 = 0
```

The probe recovered some GT5 short raw geometry versus the previous GT5-safe
and role-containment probes, but it did not solve the Q24 bottleneck:

```text
val/train0601 GT5 visible<=10 raw match20 = 0.509434 / 0.486339
env30 val/train0601 GT5 visible<=10 raw match20 = 0.754717 / 0.775956
val/train0601/train0531 GT3GT4 visible<=20 q12..q23 raw-best rate =
  0.578778 / 0.634703 / 0.646018
```

The failure is event-role coverage and existence/valid calibration, not only
short training. Previously high-risk queries were suppressed, but risk migrated
to uncovered or protected carriers (`q11`, `q16`, `q20`, `q0`, `q1`), while
clean true-GT5 short carriers such as `q13` and `q15` kept too-low exist
scores. Do not continue this artifact or run TEST. Any next Q24 probe must use
dynamic event-aware containment and positive score calibration instead of a
fixed query-ID partition copied from the previous probe.

The branch now includes that default-off Q24 event-v2 probe mechanism. It is
enabled only by explicit args:

```text
--gcs-q24-event-dynamic
--gcs-q24-event-dynamic-queries <ids>
--gcs-q24-event-dynamic-protect
--gcs-q24-event-score-calib > 0
--gcs-q24-event-score-queries <ids>
```

Dynamic containment adds extra target-zero exist/valid pressure to residual
false-extra carrier queries only when they are unmatched, short-visible by
predicted-valid length, and not protected by GT-close overlap. Score
calibration adds a soft positive existence BCE only for configured clean
queries that are close to true short GT5 lanes. It does not change labels,
decode, NMS, official metrics, model outputs, or default Q12/Q24 behavior.
It is a 20-40 epoch official-val/train-side probe only until its gate passes.

The completed event-v2 probe
`query_alpha05_env30_q24_event_v2_dynamic_score_probe40_v1` is rejected for
longer or full training. TEST was not used. Its official-val ACC/FP/FN is
`0.960033 / 0.090037 / 0.026630`, with `count_acc_4=0.772727`,
`count_acc_5=0.148649`, and count confusion including `4->6=9` and
`5->6=62`. The external 864-row official-val sweep has zero rows at or above
env30 ACC, zero rows with FP/FN both at or below env30, and zero rows with
`count_acc_4 >= 0.90`. The mechanism was active, but GT5 visible<=10 raw
match20 regressed to `0.245283 / 0.229508` on val/train0601 while the
event-mined gate still reported `total_val_gt5_to6=61` and
`total_val_gt34_false_extra=44`. Do not continue, full-train, or TEST this
artifact.

Training-time `official_best` checkpoint preservation is active as an explicit 2026-06-27 selection-protocol change. It preserves the 5-25-3 algorithm body and only changes how formal TuSimple checkpoints are selected.

The 2026-06-29 `ordered_slot_training_protocol_fix_v1` change is a protocol
cleanup only. It does not change model structure, loss definitions, or decode:

- `gcs_lane_count_balanced` is default-off in `default.yaml` and
  `tools/train_gcs.py`; enable it only with explicit
  `--gcs-lane-count-balanced` for count-balanced training ablations.
- `--gcs-mode ordered_slot` auto-switches any non-slot GCS model YAML to
  `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q5-slot-k56.yaml` unless
  `--gcs-disable-auto-model-switch` is set, in which case a non-slot model YAML
  fails fast.
- `tools/eval_gcs.py` writes an `ordered_slot_decode_v1` summary for
  ordered-slot decode and does not record query-only decode keys such as
  `conf`, `point_valid_thr`, `nms_dist_px`, `max_det`, `min_points`, or
  `count_aware_topk`.
- `gcs_ordered_point_loss` defaults to `normalized_smooth_l1`, preserving the
  old ordered-slot training objective. `aspect_l1` remains available only via
  explicit `--gcs-ordered-point-loss aspect_l1` for a separate point-loss
  ablation.
- Training-time ordered-slot `official_best` rejects non-default query-only
  sweep args (`gcs_official_confs`, `gcs_official_point_valid_thrs`,
  `gcs_official_nms_dist_pxs`, `gcs_official_max_dets`, and
  `gcs_official_min_points`) instead of silently replacing them with
  ordered-slot defaults.
- `tools/train_gcs.py` defaults to canonical current-contract paths
  `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml` and
  `data/tusimple_gcs_fixed_y_960x544.yaml`; q12-k56-named paths remain
  compatibility aliases only.

The 2026-06-29 `ordered_slot_contract_fix_v3` change tightens ordered-slot
training/evaluation contracts without changing the algorithm direction:

- ordered-slot point regression supports explicit
  `gcs_ordered_point_loss=aspect_l1`, sharing the query loss scale and avoiding
  normalized SmoothL1 beta compression. It is not the protocol default;
  protocol-only training keeps `normalized_smooth_l1` unless a separate
  point-loss ablation explicitly opts in.
- training fixed-y labels and ordered-slot targets must validate against
  descending `710,700,...,160`; official TuSimple record `h_samples` validate
  separately against ascending contiguous subsets of the canonical K56
  `160,170,...,710` grid, such as 56-point `160..710` or 48-point `240..710`.
- ordered-slot target construction removes padded/invalid lanes before fixed-y
  validation.
- official sweep/training-time selection summaries must record the full
  ordered selection keys, not only the primary metrics.
- ordered-slot official eval/sweep summaries must record the effective decode
  and `query_decode_args=not_applicable`; non-default query-only decode
  arguments are rejected for ordered-slot official evaluation.
- ordered-slot v2 lane-count classification is the formal 2/3/4/5 contract:
  `gcs_min_lanes=2`, `gcs_max_lanes=5`, `gcs_count_classes=4`,
  `pred_count_logits` is `B x 4`, `count_label = num_lanes - 2`, and decode
  uses `argmax(pred_count_logits) + 2`. A 2-lane target must be
  `slot_exist=[1,1,0,0,0]` and `count_label=0`.
- `official_best_decode.yaml` is schema-specific. Query models use
  `query_decode_v1` and may record query decode arguments. Ordered-slot models
  use `ordered_slot_decode_v1` and must not record `conf`,
  `point_valid_thr`, `nms_dist_px`, `max_det`, `min_points`, or
  `count_aware_topk` as effective decode arguments.

The 2026-06-29 `ordered_slot_runtime_stability_fix_v1` change is a runtime
stability fix only. It does not change ordered-slot loss targets, loss gains,
decoder behavior, official metrics, or the 2/3/4/5 count contract:

- Ordered-slot target construction keeps GT lanes as float32 through padding
  removal, fixed-y validation, canonical fixed-y snapping, and slot target
  construction. Ordered-slot loss calculations use float32 logits/points under
  AMP so half prediction tensors do not half-quantize GT fixed-y anchors.
- Shared fixed-y validation helpers live under
  `ultralytics/utils/gcs_fixed_y.py`; dataset and standalone tools must not
  depend on `ultralytics.models` for fixed-y contract checks.
- `tools/visualize_ordered_slot_targets.py --help` must run standalone without
  importing `GCSLaneDataset` at module import time.
- `--gcs-mode ordered_slot --scale > 0` fails fast until a count-preserving
  fixed-y scale augmentation path is explicitly implemented. Center scale
  resampling can drop ordered-slot targets to unsupported 0/1 lane counts,
  violating the formal 2/3/4/5 count contract. Query-mode scale behavior is
  unchanged.

The 2026-06-29 `ordered_slot_output_and_legacy_tool_contract_fix_v1` change
introduced schema-specific ordered-slot decode metadata and protected legacy
query sweep tools. Its earlier warning-and-sort official policy is superseded
by `ordered_slot_eval_contract_hardening_v1` below:

- ordered-slot decode supports `output_order=slot|left_to_right`; only
  diagnostic/debug exports may use `left_to_right` postprocessing.
- `tools/sweep_gcs_conf.py` is a legacy query-threshold sweep tool and fails
  fast for ordered-slot checkpoints. Ordered-slot checkpoints must use
  `tools/sweep_tusimple_official_cached.py`,
  `tools/sweep_tusimple_official.py`, or `tools/eval_tusimple_official.py`
  with auto decode. Current threshold sweeps should prefer the cached helper.
- ordered-slot official sweep JSON rows must not contain query-only keys such
  as `conf`, `point_valid_thr`, `nms_dist_px`, `max_det`, `min_points`, or
  `count_aware_*`. CSV output may keep fixed query-only columns, but
  ordered-slot rows must leave those columns empty and write
  `decode_schema=ordered_slot_decode_v1` plus
  `query_decode_args=not_applicable`.

The 2026-06-29 `ordered_slot_order_diagnostics_summary_fix_v1` change remains
an evidence-consistency rule:

- `tools/eval_tusimple_official.py --pred-json --decode-mode ordered_slot`
  cannot re-run `decode_ordered_slot`, so it records
  `ordered_slot_order_checked=false`,
  `ordered_slot_order_violations=null`,
  `ordered_slot_order_violation_images=null`, and
  `ordered_slot_order_diagnostics=not_available_from_pred_json`.
- Ordered-slot model-forward official eval/sweep paths record
  `ordered_slot_order_checked=true` and write violation counts only from
  decoder diagnostics.

The 2026-06-29 `ordered_slot_eval_contract_hardening_v1` change is the active
ordered-slot official/eval contract. It changes protocol safety only; it does
not change model structure, loss gains, labels, checkpoint selection metrics,
or official metric formulas:

- ordered-slot `official_eval`, `official_sweep`, `official_best`, and
  `eval_gcs` use `output_order=slot`, `order_check=error`,
  `uses_runtime_sort=false`, and `order_violation_policy=fail_fast`.
- Ordered-slot main official results must prove that slots `0..num_lanes-1`
  directly represent lanes from left to right. A slot-order violation fails
  fast instead of being repaired at export time.
- Debug or visualization exports may request `output_order=left_to_right`, but
  their summaries must write `result_type=postprocessed_sorted_export` and
  `not_for_main_ordered_slot_claim=true`.
- `official_best_decode.yaml` and official summaries for ordered-slot decode
  must record `decode_schema=ordered_slot_decode_v1`,
  `output_order=slot`, `uses_runtime_sort=false`, and
  `order_violation_policy=fail_fast`.
- For `split=val`, official evaluation must use the canonical 363-image
  official-val GT JSON. It must not fall back to `train_val.json` or
  `label_data_0313.json` as comparable official-val evidence.
- Noncanonical val GT requires the explicit `--allow-noncanonical-gt` escape
  hatch, and summaries must write `gt_contract=noncanonical` and
  `comparable_to_e1_spurious=false`.
- Query-mode model construction ignores ordered-slot-only overrides such as
  slot lane bounds and count-class settings. Ordered-slot-only arguments are
  passed to `parse_model` only when `gcs_mode=ordered_slot`.

The 2026-06-30 `ordered_slot_bottom_order_loss_fix_v1` change closes the
training/objective gap between ordered-slot supervision and strict official
slot-order checking:

- `gcs_order` now means only the old common-anchor adjacent-slot order loss and
  defaults to `0.2`.
- `gcs_gt_bottom_order=1.0` adds GT-bottom order supervision. For each GT lane,
  bottom means its own bottom-most visible anchor under the descending
  fixed-y order `710,700,...,160`; adjacent lanes do not need common visible
  anchors.
- `gcs_decoded_bottom_order=1.0` adds decoded-bottom order supervision. It uses
  `argmax(pred_start_logits)` per slot to gather the predicted decoded bottom
  x and applies the adjacent-slot left-to-right margin without requiring common
  visible anchors.
- `gcs_bottom_order_margin_px=2.0` is the bottom-x margin for both bottom-order
  losses.
- Ordered-slot training logs
  `slot_order_common_violation_rate`,
  `slot_order_gt_bottom_violation_rate`,
  `slot_order_decoded_bottom_violation_rate`,
  `slot_order_decoded_bottom_pair_total`, and
  `slot_order_decoded_bottom_violation_pairs`.
- Training-time ordered-slot official-best candidate sweeps use
  `output_order=slot`, `order_check=warn`, and `uses_runtime_sort=false`, then
  record `strict_order_valid` and `ordered_slot_order_violations`. This avoids
  stopping training early while still selecting strict-order-usable checkpoints
  first.
- Final ordered-slot official evaluation remains strict:
  `output_order=slot`, `order_check=error`, and `uses_runtime_sort=false`.
  `output_order=left_to_right` remains diagnostic/export-only and must be
  marked `not_for_main_ordered_slot_claim=true`.

The 2026-06-29 `ordered_slot_metrics_and_shape_contract_fix_v1` change is a
logging, input-validation, and entrypoint-guidance fix only. It does not change
model structure, training targets, loss definitions, count logits, decoder
lane-count selection, runtime sorting, or official metrics:

- ordered-slot per-class count logs record
  `slot_count_correct_2/3/4/5` and `slot_count_total_2/3/4/5`. Per-class
  `slot_count_acc_2/3/4/5` is derived as accumulated `sum(correct)/sum(total)`;
  when a class is absent (`total=0`), the class accuracy is `nan`/NA and must
  be skipped by overfit-style gates instead of treated as zero.
- ordered-slot decode validates `pred_points`, `pred_start_logits`,
  `pred_end_logits`, `pred_count_logits`, and optional `pred_exist_logits`
  shapes before decoding, and fails fast with `ValueError` on shape mismatch.
- ordered-slot target construction keeps the direct contract that GT lanes are
  assigned to slots left-to-right by bottom-most visible x under the training
  fixed-y desc `710,700,...,160` order.
- `tools/train_gcs.py --gcs-mode ordered_slot` supports automatic switching to
  `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q5-slot-k56.yaml` unless
  `--gcs-disable-auto-model-switch` is set. The generic Ultralytics entrypoint
  (`yolo task=gcs_lane gcs_mode=ordered_slot`) uses `TASK2MODEL["gcs_lane"]`
  and does not guarantee this auto-switch; pass
  `model=ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q5-slot-k56.yaml`
  explicitly for ordered-slot runs through the generic entrypoint.

The 2026-06-29 `ordered_slot_boundary_and_legacy_safety_fix_v1` change is a
contract and tooling-safety fix only. It does not change training objectives,
loss terms, ordered-slot lane-count selection, runtime sorting, or official
metrics:

- `GCSLaneHead` requires ordered-slot `num_queries=5` and `num_slots=5`.
- ordered-slot decode requires prediction slot dimension `S == max_lanes`;
  malformed `pred_count_logits` must fail with a clear `ValueError`.
- `--pred-json` ordered-slot official eval rejects non-default query-only
  decode args and keeps `query_decode_args=not_applicable` in summaries.
- legacy `tools/sweep_gcs_conf.py` rejects `--split test`; it is a
  validation-only threshold search tool.
- `tools/strip_gcs_head_ckpt.py` strips raw top-level GCS head keys, and
  `tools/visualize_ordered_slot_targets.py` raises when image writing fails.

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

`data/tusimple_gcs_fixed_y_k56_960x544.yaml` is a compatibility path for old q12-k56 records and has the same K56 contract.

Default data root:

```text
datasets/tusimple_fixed_y_k56_960x544
```

The K56 dataset must be rebuilt from original TuSimple JSON and images, not resampled from historical K32 labels.

## Model Contract

Default model:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
```

Compatibility configs also use the same K56 fixed-y contract:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-fixed-y.yaml
```

Default and compatibility baseline configs keep `Q=12`, `K=56`, fixed-y
anchors `710/720 -> 160/720`, and `--imgsz 544 960`. Post-env30 non-Q12 query
configs such as Q20/Q24 protected dual-bank static-gate candidates are
rejected legacy records, not active branch configs.

Historical q12-k56 experiment docs are old records. Preserve them, but do not let them override the active 5-25-3 K56 mainline contract.

Active source/config is restored to commit `86c8fb31c` (`Add GT4 GT5 weak
geometry rescue run`). Its algorithm contract is the env30 K56 baseline: no
post-env30 Q20/Q24, dual-head, query extent, short local-refine, lateral
candidate, gated candidate, role/event containment, or later candidate gate
fixes are active. Later commits and notes are preserved only as rejected
legacy experiment conclusions in the docs. They are not active
CLI/config/loss/tool behavior.

## Label Contract

```text
point_mode = fixed_y
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end = 160 / 720 = 0.2222222222222222
K = 56
```

The K56 anchors align exactly to TuSimple official h-samples from `710` down to `160`, descending by `10` pixels and normalized by original height `720`.

Expected fixed-y label fields:

```text
semantic_mask
edge_mask
lanes
lane_valid
num_lanes
point_mode
fixed_y
num_points
raw_file
image_shape
```

Do not mix old `0.98`/K32 fixed-y labels with this branch.

## Output Contract

This 5-25-3 branch model output must include:

```text
pred_points: B x Q x K x 2
pred_logits: B x Q
pred_valid_logits: B x Q x K
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

With the default K56 model this means:

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
```

The optional query-count model
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count.yaml` additionally
emits:

```text
pred_count_logits: B x 4
```

This optional output is default-off and uses class mapping
`0->2`, `1->3`, `2->4`, `3->5` lanes.

The optional Q12 dual-head model
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-dualhead.yaml` additionally
emits:

```text
pred_count_logits: B x 4
pred_quality_logits: B x 12
```

This dual-head output is default-off. `pred_count_logits` is the image-level
2/3/4/5 count classifier. `pred_quality_logits` is the query-level lane
quality/ranking score. The original `pred_logits` remain the query
existence/objectness logits and the default fallback decode score for old
checkpoints.

The optional query extent model
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-extent.yaml` additionally
emits:

```text
pred_start_logits: B x 12 x 56
pred_end_logits: B x 12 x 56
```

These logits classify the first and last visible fixed-y anchor indexes in
the bottom-to-top K56 order (`710, 700, ..., 160`). They are default-off,
query-only, and independent of ordered-slot interval heads. This YAML must not
emit `pred_count_logits` or `pred_quality_logits`.

The optional Q12 short local x-refine model
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-short-local-refine.yaml`
additionally emits:

```text
pred_coarse_points: B x 12 x 56 x 2
pred_short_refined_points: B x 12 x 56 x 2
pred_short_refine_delta_logits: B x 12 x 56
pred_short_refine_delta_norm: B x 12 x 56
```

This output is default-off. The default Q12 YAML must not emit
`pred_coarse_points`, `pred_short_refined_points`,
`pred_short_refine_delta_logits`, or `pred_short_refine_delta_norm`.

The optional Q24 dual-head protected-static model
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q24-k56-dualhead.yaml` emits:

```text
pred_count_logits: B x 4
pred_quality_logits: B x 24
```

It preserves the Q24 protected static bank and is only a 40-epoch
official-val diagnostic until its official-val and train-side gates pass.

## Loss Contract

Default logged loss items on the active `424ab1c86` rollback state include:

```text
exist_loss
point_loss
point_valid_loss
smooth_loss
curve_loss
mask_loss
edge_loss
count_loss
count_under5_loss
count_boundary_loss
spurious_neg_loss
spurious_negative_count
spur_cand
spur_prot
spur_final
spur_neg
spur_cnt_gt3
spur_cnt_gt4
spur_cnt_gt5
spur_neg_gt3
spur_neg_gt4
spur_neg_gt5
count_score_mean
gt5_short_pos_count
gt5_short_pos_anchor_count
gt5_short_point_valid_loss
cnt_bound_5under
cnt_score
boundary_pseudo_neg_loss
boundary_pseudo_count
boundary_pseudo_score_mean
boundary_pseudo_candidate_count
boundary_pseudo_protected_count
query_count_ce_loss
query_count_acc
query_count_pred_mean
query_quality_loss
query_quality_pos_mean
query_quality_pos_count
query_extent_loss
query_extent_start_acc
query_extent_end_acc
query_extent_iou
query_extent_short_count
short_local_refine_loss
short_local_refine_count
short_local_refine_coarse_ape
short_local_refine_refined_ape
short_local_refine_gain20
```

`query_count_ce_loss` is active only when a query model emits
`pred_count_logits` and `gcs_query_count_ce > 0`; the default gain is `0.0`.
When logits are absent, the three query-count log items are zero for old-model
compatibility.

`query_quality_loss` is active only when a query model emits
`pred_quality_logits` and `gcs_query_quality > 0`; the default gain is `0.0`.
Its target follows the quality-aware geometry/visible-IoU target family used
for existence supervision, but it trains a separate decode ranking score. The
dual-head probe intentionally sets `gcs_count=0.0`, `gcs_count_under5=0.0`,
`gcs_count_boundary=0.0`, `gcs_boundary_pseudo_neg=0.0`,
`gcs_role_contain=0.0`, `gcs_q24_event_contain=0.0`, and
`gcs_q24_event_score_calib=0.0`, so count estimation and lane ranking are not
implemented by reusing `sum(sigmoid(pred_logits))` or the rejected Q24
event/role patches.

`short_local_refine_loss` is active only when a query model emits
`pred_short_refined_points` and `gcs_short_local_refine > 0`; the default gain
is `0.0`. The loss is normalized-x SmoothL1 with
`gcs_short_local_refine_beta_px / image_width` as beta. The auxiliary residual
is bounded by `gcs_short_local_refine_max_delta_px / image_width`, default
`40px`. When auxiliary refined points are absent or the gain is zero, the five
short-local-refine log items are zero for old-model compatibility.

Post-`424ab1c86` log items such as `short_side_geom_loss`, `far_spur_loss`,
`farspur_if_loss`, `rank_topk_loss`, `shortside_*`, `rank_*`,
`base_exist_ignore_*`, and `clear_far_boundary_count` are legacy records only
and are not emitted by the active rollback code.

For ordered-slot mode, standalone `OrderedSlotGCSLoss` defaults must match
`ultralytics/cfg/default.yaml`: `gcs_count_ce=1.0`,
`gcs_interval=1.0`, `gcs_order=0.2`,
`gcs_gt_bottom_order=1.0`, `gcs_decoded_bottom_order=1.0`, and
`gcs_bottom_order_margin_px=2.0`. `gcs_order` is only the common-anchor order
loss. The two bottom-order losses are separate ordered-slot loss terms and do
not require adjacent lanes to share visible anchors. Constructing the loss with
`gcs_count_ce <= 0` or `gcs_interval <= 0` is a contract error. Disabling the
common-anchor order loss requires an explicit ablation flag; it must not happen
through a silent fallback default. Training summaries must record
`ordered_slot_loss_contract` and the effective ordered-slot loss weights.

`gcs_slot_exist_w4` and `gcs_slot_exist_w5` are BCE element weights for
slot-existence targets. They weight both positive and negative BCE elements for
the affected slots; they are not recall-only `pos_weight` controls. Any
recall-oriented positive-only weighting is a separate future ablation.

Per-class ordered-slot count accuracy logs may use `nan`/null internally when
`slot_count_total_N == 0`, but user-facing progress and `results.csv` must show
`NA`. Aggregation uses accumulated `correct/total`; absent classes are skipped
instead of being logged as `0` or raw `nan`.

`count_boundary_loss` is disabled by default through `gcs_count_boundary=0.0`.
When enabled, it applies to `sum(sigmoid(pred_logits))` with GT3 upper, GT4
lower/upper, and GT5 lower boundaries. `gcs_count_boundary_gt5_under_weight`
defaults to `None`, which follows `gcs_count_boundary_gt5_weight`; setting it
allows only the GT5 undercount boundary weight to be changed. `cnt_bound_5under`
logs the unweighted GT5 undercount boundary term.

`boundary_pseudo_neg_loss` is disabled by default through
`gcs_boundary_pseudo_neg=0.0`. When enabled, it requires `pred_valid_logits`
and applies only to unmatched queries on images whose GT lane count equals
`gcs_boundary_pseudo_gt_count`. Candidate queries must have predicted-visible
anchor count in `[gcs_boundary_pseudo_min_valid, gcs_boundary_pseudo_visible_thr]`,
existence score at least `gcs_boundary_pseudo_score_thr`, nearest GT lane equal
to the leftmost or rightmost GT lane, and nearest-GT mean x distance at least
`gcs_boundary_pseudo_dist_thr`. `gcs_boundary_pseudo_envelope_margin_px=-1.0`
preserves the old mask. When set to a non-negative margin, the envelope gate is
computed on query/GT common-visible anchors. For each such anchor, the left and
right GT envelope is the minimum and maximum valid GT x at that fixed-y anchor.
A left-boundary candidate must have at least
`gcs_boundary_pseudo_envelope_ratio_thr` of common-visible anchors left of
`left_env_x - margin`; a right-boundary candidate must have at least that ratio
right of `right_env_x + margin`. This changes only the training loss candidate
mask; it does not change model outputs, matcher assignment, labels, decode,
NMS, or official metrics.

### Legacy Post-424 Loss And Diagnostic Records

The following loss and diagnostic descriptions are retained only for old
experiment interpretation. They were introduced after `424ab1c86` and are not
available in the active rollback code unless a future task explicitly restores
the corresponding commits.

Legacy post-`424ab1c86` record: `short_side_geom_loss` was disabled by default through
`gcs_short_side_geom=0.0`. When enabled, it adds an aspect-weighted L1 point
loss only on Hungarian-matched GT lanes for images with at least
`gcs_short_side_geom_min_gt_lanes` GT lanes. Matched lanes are selected only
when their visible anchor count is at or below
`gcs_short_side_geom_visible_max`; `gcs_short_side_geom_side_only=True`
further restricts them to the leftmost/rightmost GT lanes by bottom visible x.
GT4 and GT5-or-denser samples can be weighted with
`gcs_short_side_geom_weight_gt4` and `gcs_short_side_geom_weight_gt5`. This
does not change unmatched queries, matcher logic, data sampling, labels, point
visibility targets, decode, NMS, or official metrics. `short_side_geom_count`,
`short_side_geom_gt4`, and `short_side_geom_gt5` are diagnostics for the
selected matched lanes.

Legacy post-`424ab1c86` record: `gcs_shortside_rawmatch_boost=0.0` kept shortside raw-match positive weighting
disabled by default. The shared shortside raw-match contract considers only
images with at least `gcs_shortside_min_gt_lanes=4`, only leftmost/rightmost GT
lanes by K56 bottom/lower-half mean x, and only GT lanes passing the shared
weak-visible predicate:
`visible >= gcs_shortside_ultra_min_valid_points=2` and
(`visible <= gcs_shortside_visible_max=42` or
`visible < image_gt_visible_median - gcs_shortside_median_margin`). The median
branch is enabled by default with `gcs_shortside_use_median=True` and
`gcs_shortside_median_margin=0.0`; the standard median is computed from
current-image GT lanes after filtering out lanes below the reliable minimum
`gcs_shortside_reliable_min_valid_points=6`, so even lane counts average the
two middle values. Visible `<2` side GT is skipped
for training rescue and logged through `shortside_visible_lt2_skipped_count`.
Visible `2..5` side GT is ultra-short: it can enter diagnostic and optional
light unmatched-rescue score-floor supervision, but only a same-GT Hungarian
rawmatch query receives the base shortside positive boost/target floor by
default. Unmatched ultra-short raw-rescue remains default-off, does not receive
strong geometry rescue, and does not enter ranking positives by default.
Visible `>=6` side GT is reliable and can use the full shortside rawmatch/rescue
path. For each eligible side GT, raw geometry is compared against all Q queries using
canonical `gcs_shortside_rawmatch_px=30` pixels, or canonical
`gcs_shortside_side_rawmatch_px=40` pixels for side lanes. Legacy
aliases `gcs_shortside_rawmatch_dist_px`, `gcs_shortside_side_dist_px`, and
`gcs_shortside_max_valid_points` are fallback-only compatibility keys. If a
canonical key and a legacy alias both exist with different values, the
canonical key wins and training logs a warning. For each eligible side GT, the
same-GT Hungarian query has first priority: if it exists and is raw-close, the
reliable boost path multiplies that matched query existence BCE by
`1 + gcs_shortside_rawmatch_boost` and visible point-valid positive anchors by
`1 + 0.8 * gcs_shortside_rawmatch_boost`. `gcs_shortside_exist_target_floor`
is separate and defaults to `0.0`; only an explicit positive floor value raises
the quality-aware existence target for selected shortside rawmatch/rescue
queries. A nearer unmatched duplicate cannot block this matched-query boost.
Optional independent reliable score-floor BCE uses
`gcs_shortside_score_floor_gain=0.0` and `gcs_shortside_score_target=0.8`.
Only when no same-GT Hungarian query satisfies the raw threshold does reliable
unmatched raw-rescue selection consider raw-close unmatched queries and choose
the nearest one. Queries already Hungarian-matched to another GT are skipped,
not reassigned. Multiple side GT lanes choosing the same unmatched rescue query
keep only the nearest GT and count the rest as conflicts.

Legacy post-`424ab1c86` record: `gcs_shortside_raw_rescue=False` kept unmatched raw-rescue behavior disabled by
default. When explicitly enabled, an accepted unmatched best query becomes a
raw-rescue auxiliary positive. It is excluded from
rank/far-spur/spurious negative masks and can receive independent positive
supervision only through `gcs_shortside_rescue_exist_gain`,
`gcs_shortside_rescue_point_gain`, and `gcs_shortside_rescue_valid_gain`.
It is protected from the original base existence and point-valid BCE only when
the separate explicit base-ignore ablation
`gcs_base_ignore_raw_rescue=True` is also enabled.
All three rescue gains default to `0.0`; optional positive existence target
flooring requires an explicit `gcs_shortside_exist_target_floor > 0`, while
decode, NMS, matcher assignment, official metrics, labels, and data sampling
are unchanged. Ultra-short
visible `2..5` side GT lanes are a separate tier. They are seen and counted by
default, but `gcs_shortside_ultra_enable=False`,
`gcs_shortside_ultra_score_floor_gain=0.0`,
`gcs_shortside_ultra_valid_gain=0.0`, and
`gcs_shortside_ultra_rank_pos=False` keep unmatched ultra-short light
score-floor, unmatched ultra-short light valid rescue, and optional ultra
ranking-positive participation off unless explicitly ablated. Same-GT
Hungarian ultra-short rawmatch base boost is not blocked by
`gcs_shortside_ultra_enable=False`.
`gcs_shortside_ultra_point_gain=0.0` keeps strong geometry off by default.
`short_raw_boost_count`, `short_raw_boost_gt4`, `short_raw_boost_gt5`,
`short_raw_missing`, `shortside_hungarian_rawmatch_candidate_count`,
`shortside_unmatched_raw_rescue_candidate_count`,
`shortside_rawmatch_candidate_total_count`, `raw_rescue_candidate_count`,
`raw_rescue_final_count`, `raw_rescue_conflict_count`,
`shortside_selected_hungarian_rawmatch`,
`shortside_selected_unmatched_rescue`, `shortside_rescue_conflict`,
`shortside_missing_no_rawmatch`,
`shortside_nearest_was_duplicate_but_hungarian_boosted`,
`shortside_selected_by_abs_count`, `shortside_selected_by_median_count`,
`shortside_selected_total_count`,
`short_raw_boost_exist_target_mean`, `short_raw_boost_exist_target_min`,
`short_raw_boost_exist_target_p25`, `short_raw_boost_exist_target_p50`,
`short_raw_boost_exist_target_p75`,
`short_raw_boost_target_below_05_count`,
`short_raw_boost_target_below_07_count`,
`shortside_rawmatch_target_mean_before`,
`shortside_rawmatch_target_min_before`,
`shortside_rawmatch_target_p25_before`,
`shortside_rawmatch_target_p50_before`,
`shortside_rawmatch_target_below_05_count`,
`shortside_rawmatch_target_below_07_count`,
`shortside_target_floor_applied_count`, `shortside_reliable_count`,
`shortside_reliable_selected_count`, `shortside_ultra_seen_count`,
`shortside_ultra_enabled_count`, `shortside_ultra_score_floor_count`,
`shortside_ultra_valid_count`, `shortside_visible_lt2_skipped_count`,
`shortside_ultra_hungarian_base_boost_count`,
`shortside_ultra_unmatched_rescue_seen_count`,
`shortside_ultra_unmatched_rescue_enabled_count`,
`shortside_ultra_unmatched_rescue_skipped_count`,
`shortside_ultra_in_gt4_4to3_count`, and
`shortside_ultra_in_gt5_5to4_count` are diagnostics. Training and
`tools/diagnose_gcs_count_contract.py` use the same weak-visible helper.
`raw_rescue_candidate_count` is now the unmatched raw-rescue candidate count;
Hungarian raw-close matches are reported separately through
`shortside_hungarian_rawmatch_candidate_count`.

`gcs_gt5_short_visible_thr=0` and
`gcs_gt5_short_point_valid_weight=1.0` keep GT5 short point-valid rescue
effectively disabled by default. When enabled, the point-valid BCE keeps the
same global target structure, applies only on images with `GT lane count == 5`,
only on Hungarian-matched GT lanes whose visible anchor count is at or below
the threshold, and only multiplies the BCE weight for visible
`target_valid == 1` anchors. It does not change point regression, smooth,
curve, mask, edge, dataset, dataloader, matcher, decode, NMS, or official
metrics. `gt5_short_pos_count`, `gt5_short_pos_anchor_count`, and
`gt5_short_point_valid_loss` are diagnostics for the rescued anchors.

`gcs_short_geom=0.0` keeps the default query point/curve geometry losses
unchanged. When explicitly enabled, `gcs_short_geom` applies only inside
training loss calculation for Hungarian-matched images with `GT lane count == 4`
or `GT lane count == 5`, only to GT lanes whose visible anchor count is at or
below `gcs_short_geom_visible_thr`, and only by lane-level weighting inside
`point_loss` and `curve_loss`. The GT4 path is default-off through
`gcs_short_geom_gt4_weight=1.0`; GT5 keeps the existing
`gcs_short_geom_gt5_weight` behavior. It does not change model outputs, matcher
assignment, smooth loss, point-valid BCE targets, mask/edge losses, dataset,
dataloader, decode, NMS, official metrics, Count Head, or loss item count.
The geometry boost has no side-lane/order assumption.

`spurious_neg_loss` is disabled by default through `gcs_spurious_neg=0.0`.
When enabled, it requires `pred_valid_logits` and applies only to unmatched
queries with visible-anchor count in `[2, gcs_spurious_max_points]`, at least
`gcs_spurious_min_overlap` overlapping predicted-valid anchors with any matched
query, and mean overlapping x distance at most `gcs_spurious_close_px` pixels.
For images with `gt_lanes >= 5`, `gcs_spurious_disable_gt5=True` skips this loss
for the image and takes priority over `gcs_spurious_gt5_weight`. Otherwise the
image's selected spurious-negative terms are multiplied by
`gcs_spurious_gt3_weight` when `gt_lanes <= 3`, `gcs_spurious_gt4_weight` when
`gt_lanes == 4`, and `gcs_spurious_gt5_weight` when `gt_lanes >= 5`. The loss is
normalized by selected spurious-query count, not by weight sum, so weights above
or below `1.0` strengthen or weaken that GT group. The default all-`1.0`
setting preserves old E3-lite behavior. `spurious_negative_count`,
`spur_cnt_gt3`, `spur_cnt_gt4`, `spur_cnt_gt5`, `spur_neg_gt3`,
`spur_neg_gt4`, `spur_neg_gt5`, `count_score_mean`, `cnt_bound_5under`, and
`cnt_score` are log-only diagnostics and are not directly part of the weighted
training objective.

Legacy post-`424ab1c86` record: the `gcs_spurious_neg` BCE system was mutually exclusive with effective
new ignore-first/ranking/raw-rescue/base-ignore count-contract paths by
default. If any of `gcs_farspur_weight > 0`, `gcs_rank_topk_weight > 0`,
`gcs_shortside_raw_rescue=True`, or any `gcs_base_ignore_*` flag is active
while `gcs_spurious_neg != 0`, loss construction raises `ValueError` with the
message that `gcs_spurious_neg` conflicts with the new ignore-first/ranking
contract. A legacy reproduction can override this only by explicitly setting
`gcs_allow_legacy_spurious_with_new_contract=True`, which logs a strong warning
and records `legacy_spurious_active=True`.

`gcs_spurious_gt_protect=False` preserves the old E3-lite spurious-negative
selection. When enabled, each duplicate-like spurious candidate is compared
against all GT lanes when the image has at least
`gcs_spurious_gt_protect_min_gt_lanes` GT lanes; the default `0` preserves the
old all-count behavior. If the candidate overlaps a GT lane by at least
`gcs_spurious_gt_protect_min_overlap` anchors and has mean x distance at most
`gcs_spurious_gt_protect_px` pixels, `better_matched` mode protects it when the
GT lane has no matched query, the matched query has insufficient GT overlap, or
the candidate mean GT x distance plus `gcs_spurious_gt_protect_margin_px` is
lower than the matched query mean GT x distance. Protected candidates are
excluded only from the extra spurious target-zero BCE. `spur_cand` logs the
pre-protect candidate count, `spur_prot` logs protected candidates,
`spur_final` and `spurious_negative_count` log final selected negatives, and
`spur_neg` mirrors `spurious_neg_loss` for compact progress logging.

Legacy post-`424ab1c86` record: `far_spur_loss` was disabled by default through `gcs_far_spurious_neg=0.0`.
When enabled, it requires `pred_valid_logits` and applies only to unmatched
queries on images whose GT lane count is in
`[gcs_far_spurious_min_gt_lanes, gcs_far_spurious_max_gt_lanes]`. A candidate
must have `sigmoid(pred_logits) >= gcs_far_spurious_score_thr` and a longest
contiguous predicted-valid span in
`[gcs_far_spurious_min_valid, gcs_far_spurious_max_valid]`. If any GT lane
overlaps by at least `gcs_far_spurious_min_overlap` valid anchors and has mean
x distance at or below `gcs_far_spurious_protect_px`, the candidate is
protected. Otherwise, all overlapping GT-lane mean x distances must be greater
than `gcs_far_spurious_far_px` before target-zero BCE is applied to the unmatched
query logit. `gcs_far_spurious_gt3_weight`, `gcs_far_spurious_gt4_weight`, and
`gcs_far_spurious_gt5_weight` weight selected candidates by GT group. Defaults
target GT3/GT4 (`1.0`, `1.0`) and keep GT5 pressure off (`0.0`).
`far_spur_cand`, `far_spur_neg`, `far_spur_gt3`, `far_spur_gt4`,
`far_spur_gt5`, `far_spur_score`, and `far_spur_valid` are diagnostics for
selected far spurious candidates.

Legacy post-`424ab1c86` record: `farspur_if_loss` was disabled by default through `gcs_farspur_weight=0.0`.
When enabled with `gcs_farspur_ignore_first=True`, unmatched queries are first
classified into near-GT corridor, ambiguous side region, and clear-far
spurious groups using the `gcs_farspur_near_dist_px` and
`gcs_farspur_side_ignore_dist_px` thresholds. Near/ambiguous queries are
ignored for this BCE term. Only
queries with `sigmoid(pred_logits) >= gcs_farspur_score_thr`, longest
predicted-valid span at least `gcs_farspur_min_valid_points`, and distance
greater than `gcs_farspur_clear_dist_px` from all GT lanes receive target-zero
BCE. `farspur_if_samples`, `farspur_if_pos`, `farspur_if_ignore`,
`farspur_if_clear`, `farspur_if_near`, `farspur_if_side`, and
`clear_far_boundary_count` are diagnostics. A query exactly at
`gcs_farspur_clear_dist_px` is counted as a boundary diagnostic but is not a
clear-far spurious negative.

Legacy post-`424ab1c86` record: base existence BCE and point-valid BCE ignore masks were independent explicit
ablations. They default off through:
`gcs_base_ignore_raw_rescue=False`,
`gcs_base_ignore_rank_near=False`,
`gcs_base_ignore_farspur_near=False`, and
`gcs_base_ignore_duplicate_like=False`. Enabling `gcs_rank_topk_weight`,
`gcs_farspur_ignore_first`, `gcs_farspur_weight`, or
`gcs_shortside_raw_rescue` does not implicitly change the base BCE targets.
`gcs_farspur_ignore_first=True` with `gcs_farspur_weight=0` therefore only
classifies diagnostics and does not protect base BCE negatives; the
farspur-derived base ignore flag is effective only in the full ignore-first
contract where `gcs_farspur_ignore_first=True`, `gcs_farspur_weight > 0`, and
`gcs_base_ignore_farspur_near=True`. The rank-derived base-ignore
sources are additionally scoped to images with
`gt_lanes >= gcs_rank_gt_min_lanes`, so GT3 samples are not affected by the
default GT4/GT5 ranking contract. `gcs_base_ignore_duplicate_like=True` ignores
only duplicate-like rank negatives from base BCE; rank near-GT/side-ambiguous
base ignore still requires `gcs_base_ignore_rank_near=True`. Clear-far base
BCE negatives are preserved through `clear_far_final`, and point-valid BCE
expands the same explicit per-query ignore to all anchors only for the
selected base-ignore sources.
Diagnostics include `base_exist_ignore_raw_rescue_count`,
`base_exist_ignore_rank_near_count`, `base_exist_ignore_rank_side_count`,
`base_exist_ignore_farspur_near_count`,
`base_exist_ignore_farspur_side_count`,
`base_exist_ignore_duplicate_rank_only_count`,
`base_exist_ignore_near_count`, `base_exist_ignore_side_ambiguous_count`,
`base_exist_ignore_duplicate_like_count`,
`base_exist_negative_kept_clear_far_count`, and
`base_exist_negative_kept_other_count`.

Legacy post-`424ab1c86` record: `rank_topk_loss` was disabled by default through `gcs_rank_topk_weight=0.0`.
When enabled, it uses the same score as query decode,
`sigmoid(pred_logits)`, and applies
`mean(max(0, gcs_rank_margin - pos_score + neg_score))`. The default
`gcs_rank_pair_reduction=global_pair_mean` averages across all valid
positive-negative pairs in the batch. The legacy-compatible
`gcs_rank_pair_reduction=image_mean` first averages pairs inside each image
and then averages images. It applies only on images with at least
`gcs_rank_gt_min_lanes=4` GT lanes. Positive queries are selected by
`gcs_rank_pos_scope`, whose default is `shortside_reliable`.
Supported scopes are `shortside_reliable`, `shortside_with_ultra`,
`gt4gt5_matched`, and `all_matched`. The default `shortside_reliable` scope
uses reliable Hungarian shortside rawmatch positives only. `shortside_with_ultra`
only adds the Hungarian ultra tier when `gcs_shortside_ultra_rank_pos=True`;
unmatched ultra rescue still stays out of ranking positives by default.
`gt4gt5_matched` means all Hungarian matched true-lane queries on images with
GT lane count >= 4, and `all_matched` means all Hungarian matched true-lane
queries; neither scope is narrowed by `gcs_shortside_ultra_rank_pos`. If
`gcs_rank_pos_scope=gt4gt5_matched`, use a small first ablation weight such as
`gcs_rank_topk_weight=0.01` or `0.02` because the positive set is wider.
When `gcs_rank_topk_weight > 0` and the default
`gcs_rank_pos_scope=shortside_reliable` is used, training logs a warning that
the run is not all-GT4/GT5-matched-positive ranking.
Unmatched reliable raw-rescue queries are excluded from ranking positives by
default through `gcs_rank_include_unmatched_rescue_pos=False`. Enabling that
escape hatch only allows vetted unmatched reliable rescue queries that are not
conflicting and not duplicate-like to enter `rank_pos`; it is an explicit
ablation, not baseline behavior.
Negative queries are capped by `gcs_rank_max_negs` and can only come from
`clear_far_spurious | normal_duplicate_like | side_duplicate_like`.
`clear_far_spurious` must be unmatched, high-score, have enough
predicted-valid anchors, be greater than `gcs_farspur_clear_dist_px` from all
GT lanes, and be outside near/ambiguous ignore zones. Normal duplicate-like
queries are close to a matched query within `gcs_rank_dup_close_px`, overlap by
at least `gcs_rank_dup_min_overlap` predicted-valid anchors, are not
raw-rescue protected, and are not rank side-ambiguous. Side-ambiguous
true-or-uncertain queries stay ignored; side-ambiguous duplicate-like queries
enter ranking negatives only when `gcs_rank_side_duplicate_enable=True` and the
same side GT has a selected true positive that is better by
`gcs_rank_side_dup_margin_px`. Duplicate-like queries are not BCE hard
negatives through this loss. `rank_topk_samples`, `rank_topk_pos`,
`rank_topk_neg`, `rank_topk_clear`, `rank_topk_dup`,
`rank_loss_noop_images`, `rank_noop_because_no_pos`,
`rank_noop_because_no_neg`, `near_gt_ignored_count`,
`duplicate_like_rank_neg_count`, `clear_far_rank_neg_count`,
`rank_neg_side_duplicate_like_count`, `rank_neg_normal_duplicate_like_count`,
`rank_side_ambiguous_ignored_count`,
`rank_side_duplicate_rejected_better_than_true_count`, `rank_pos_scope`,
`rank_pos_total`, `rank_pos_shortside_reliable`,
`rank_pos_shortside_ultra`, `rank_pos_gt4gt5_matched`,
`rank_pos_all_matched`, `rank_pos_hungarian_count`,
`rank_pos_unmatched_rescue_excluded_count`,
`rank_pos_unmatched_rescue_included_count`,
`rank_pos_conflict_excluded_count`, `rank_neg_duplicate_like`,
`rank_neg_clear_far`, and `rank_neg_near_ignored` are diagnostics.

Legacy post-`424ab1c86` record: for the GT4/GT5 count-contract features, `gcs_shortside_min_gt_lanes` and
`gcs_rank_gt_min_lanes` must stay at least `4`. Setting either below `4`
raises by default with the message:
"GT4/GT5 count-contract losses require min_gt_lanes >= 4. Set
gcs_allow_gt3_count_contract_ablation=True for explicit GT3 ablation." The
escape hatch `gcs_allow_gt3_count_contract_ablation=True` is only for explicit
GT3 ablations and logs a strong warning.

## Decode And Evaluation Contract

Decode must use real predictions only, must not use GT during inference, and
must not fabricate lanes. Query-mode TuSimple exports sort final lanes from
left to right by bottom visible x. Ordered-slot official/eval exports preserve
slot order and fail fast on slot-order violations; only diagnostic sorted
exports may postprocess ordered-slot lanes left-to-right.

The branch includes a default-off count-aware top-k postprocess ablation for
inference/evaluation only. When explicitly enabled with `--count-aware-topk`,
decode uses `sum(sigmoid(pred_logits))` to choose a dynamic final lane count
and keeps the quality-best post-conf, post-NMS lanes. This does not change
training, labels, losses, model outputs, official metrics, Count Head, Quality
Head, Survival Head, or default decode behavior.

The query count-aware top-k ablation also supports a default-zero extra margin
for validation/evaluation sweeps. `--count-aware-extra-margin` in
`tools/eval_tusimple_official.py` and `--count-aware-extra-margins` in
`tools/sweep_tusimple_official.py` / `tools/sweep_tusimple_official_cached.py`
keep `keep_k = min(k_hat + extra_margin, max_det)` lanes after the normal
count-aware `k_hat` estimate. The default `0` preserves the old behavior.
This is decode-only and does not change training, labels, losses, model
outputs, Count Head availability, or official metric formulas.

For the optional query Count Head, query decode can explicitly use
`--count-mode count_logits` with `--count-aware-topk`. This uses
`argmax(pred_count_logits)+2` as the count-aware `k_hat` for the fixed 2..5
class range. The default `--count-mode score_sum` preserves the historical
score-sum count source. `count_logits` is valid only for checkpoints whose
query model actually emits `pred_count_logits`; default query checkpoints
without the query Count Head must use `score_sum` or leave count-aware top-k
disabled.

For the optional Q12 dual-head probe, query decode uses
`pred_quality_logits` as the thresholding, ranking, NMS-order, and
count-aware top-k quality score whenever that tensor is present. Old
checkpoints without `pred_quality_logits` fall back to `pred_logits`. The
recommended probe selection uses `--count-aware-topk --count-mode count_logits`
so `pred_count_logits` chooses `k_hat` and `pred_quality_logits` chooses the
top-k lanes. This is still official-val selection only and must not read TEST.

For diagnosis only, `tools/eval_tusimple_official.py --oracle-count` can force
query count-aware top-k to use `k_hat = GT lane count` internally. This mode
intentionally uses GT during decode, so its outputs are not formal official
results, cannot be used for threshold or checkpoint selection, and are only
valid for isolating whether lane-count estimation is the bottleneck. It must
remain explicit and default-off; `oracle_gt` is not a formal public
`--count-mode` or decode-yaml count mode.

The branch includes a default-off query-mode decode ablation
`valid_before_maxdet`. When explicitly enabled with
`--valid-before-maxdet`, decode applies the existing point-valid/min_points
filter after confidence sorting and Lane-NMS but before `max_det`
truncation, then re-sorts by lane score before truncating. This prevents
queries with too few valid anchors from occupying a final `max_det` slot. The
default remains `false`, preserving old sweep/eval behavior. This does not
change training, labels, losses, model outputs, official metrics, Count Head,
Quality Head, Survival Head, or default decode behavior.

The branch also includes a default-off query extent decode ablation. When
explicitly enabled with `--extent-decode`, `extent_decode_mode=interval`
builds a continuous visible mask from `argmax(pred_start_logits)` to
`argmax(pred_end_logits)`; `extent_decode_mode=intersect` intersects that
interval with the normal point-valid contiguous mask. Decode uses only model
predictions, never GT. Official-val sweeps may compare `none`, `interval`, and
`intersect`; final selection remains official-val only and TEST remains closed
until a candidate is selected. Official sweep selection policy
`official_sweep_v4` breaks exact metric ties by preferring `none`, then
`intersect`, then `interval`, so extent decode must earn a metric improvement
instead of winning by incidental row order.

When query `count-aware top-k` is enabled, both normal decode and cached sweep
still compute lane quality with mean `pred_valid_logits` probability over the
selected visible mask. Therefore `extent_decode_mode=interval` is independent
of point-valid only for survival/visibility, not for count-aware ranking. The
Q12/env30 extent-only probe keeps `COUNT_AWARE_TOPK=0` so the first gate tests
extent visibility without that ranking dependency.

`tools/diagnose_tusimple_query_extent_gate.py` keeps the historical flat
summary fields mapped to `primary_extent_mode` for gate compatibility, and
also writes `by_extent_mode` with the same short GT4/GT5, raw-geometry-strata,
survival, extra-rate, and count-accuracy diagnostics for every decoded mode.
Use `by_extent_mode` when comparing `interval` against `intersect`.

For the rejected `query_extent_env30_probe40_v1`, the raw-geometry strata are
the decisive reading: raw-hit short GT5 endpoints are accurate
(`start/end acc = 0.956522/0.956522`, interval IoU `0.865525`), but short GT5
raw misses are `30/53` and short GT4 raw hits are only `1/8`. Therefore do not
interpret the flat endpoint metrics as a pure extent-head failure, and do not
interpret interval decode as independent of query-carrier quality.

This branch includes `tools/eval_tusimple_official.py`,
`tools/sweep_tusimple_official_cached.py`, `tools/sweep_tusimple_official.py`,
`gcs_tools/tusimple_official_eval.py`, and explicit training-time
`official_best` checkpoint preservation. Current official-val threshold
selection, including training-time `official_best`, uses the cached sweep path
and records per-sweep prediction caches without changing decode, metrics,
model outputs, losses, labels, or checkpoint-selection priority. The
post-`424ab1c86` `tools/diagnose_gcs_count_contract.py` diagnostic is legacy
only and is not present in the active rollback code. The branch still does not include
`tools/diagnose_tusimple_count_confusion.py`, `tools/diagnose_gcs_gt5.py`, or
later mainline Count/Quality/Boundary diagnostics unless a future task
explicitly ports them. Use the canonical 363-image official-val GT for
selection. After every completed training run, evaluate TEST ACC using the
official-val-selected checkpoint and decode. TEST results are verification or
reporting evidence only and must not influence selection or tuning.

Formal TuSimple checkpoint selection uses the `OFFICIAL_SELECTION_POLICY`
defined in `gcs_tools/official_selection.py` (`official_best_v4`). The
training-time `official_best` tie-break order is:

1. `strict_order_valid`: true is better
2. `ordered_slot_order_violations`: lower is better
3. `official_acc`: higher is better
4. `official_score`: higher is better
5. `official_FP`: lower is better
6. `official_FN`: lower is better
7. `count_acc_4`: higher is better
8. `count_acc`: higher is better
9. `count_acc_5`: higher is better
10. `epoch`: earliest wins when all metrics above tie

Do not select the final checkpoint only by `val/total_loss`, internal
`val/f1`, or generic `best.pt`.

## 2026-08-01 v11 Official-Quality Dense Selector Result

The frozen-base v11 probe
`query_local_segment_env30_frozen_probe20_v11_official85_allq_b4w0s1`
completed 20 epochs with:

```text
gcs_short_segment_matched_assignment = true
gcs_short_segment_official_quality_target = true
gcs_short_segment_official_pt_thresh = 0.85
gcs_short_segment_base_choice_all_queries = true
gcs_short_segment_dense_quality_weight = 1.0
gcs_short_segment_dense_neg_weight = 1.0
gcs_short_segment_freeze_base = true
candidate_decode = false
segment_decode = false
TEST = closed
```

The training-time official sweep did not beat the env30 reference. Epoch 5
was selected with `official_acc/FP/FN = 0.972568/0.017585/0.011019`;
epoch 20 was `0.972501/0.017126/0.011019`. A strict reference-point
base sweep on the selected checkpoint
(`conf=0.001`, `point_valid_thr=0.6`, `nms_dist_px=0`, `max_det=5`,
`min_points=4`) returned `0.973330/0.015748/0.009642`, matching the frozen
env30 path within sweep rounding. The base path therefore did not regress.

Hard official-val coverage with `candidate-score-thr=0.5` was:

```text
short GT5: base/raw/selected-gated = 40/52/40 of 53
short GT4: base/raw/selected-gated = 1/4/1 of 8
```

Hard train0601 coverage was:

```text
short GT5: base/raw/selected-gated = 142/178/142 of 183
short GT4: base/raw/selected-gated = 9/18/9 of 19
```

The same official-val diagnostic at `candidate-score-thr=0.05` produced the
same selected-gated counts. Within-query oracle score top-1/top-5 was only
`2/5` of 53 official-val short GT5 lanes and `6/13` of 183 train0601 short
GT5 lanes; hard-gate oracle top-1/top-5 was `0/0` on both splits. The raw
segment geometry still has capacity, but v11 does not convert it into a
usable selector signal.

Decision: reject v11 as a diagnostic promotion. Keep all short-segment
decode paths default-off and do not run TEST. A future v12 must use a
matched per-query base-plus-segment choice target, continuous geometry
quality, explicit wrong-window negatives, and the exact same combined score
for training, audit, and decode. Do not continue v11 by adding epochs or
tuning its gate threshold.

## 2026-08-01 v12 Unified Choice Result

The default-off v12 selector adds:

```text
pred_short_segment_choice_logits: B x 12 x 405
```

where class `0` is base and classes `1..404` are short-segment windows. The
completed frozen-base probe was:

```text
query_local_segment_env30_frozen_probe20_v12_b4w0s1
```

It completed `20/20` epochs with matched assignment enabled, but hard
selected-gated coverage did not exceed base:

```text
official-val short GT5 = 40/53
train0601    short GT5 = 142/183
official-val short GT4 = 1/8
train0601    short GT4 = 9/19
```

The raw segment oracle remained `52/53`, `179/183`, `4/8`, and `19/19` in
the same order. Keep v12 default-off, keep both `segment_decode=false` and
`candidate_decode=false`, and keep TEST closed.

The v12 positive target currently follows the frozen-base Hungarian
query-to-GT assignment. This is not sufficient when a useful raw segment is
generated by a different query. A reopened v13 must use candidate-aware
query-to-GT assignment and conditional unmatched-query negatives before any
formal decode is considered.

## 2026-08-02 v13 Candidate-Aware Unified Choice Result

The default-off v13 selector keeps the v12
`pred_short_segment_choice_logits: B x 12 x 405` contract and enables:

```text
gcs_short_segment_candidate_aware_assignment = true
gcs_short_segment_matched_assignment = true
gcs_short_segment_unified_choice_weight = 1.0
gcs_short_segment_unified_choice_base_neg_weight = 0.25
gcs_short_segment_bce_weight = 0.0
gcs_short_segment_dense_quality_weight = 0.0
gcs_short_segment_freeze_base = true
candidate_decode = false
TEST = closed
```

The completed run was:

```text
query_local_segment_env30_frozen_probe20_v13_b4w0s1
```

Training completed `20/20` epochs. Training-time default-decode
`official_best.pt` stayed at epoch 5 with:

```text
official_acc/FP/FN = 0.972568 / 0.017585 / 0.011019
count_acc_4/count_acc_5 = 0.984848 / 0.986486
```

Epoch 10, 15, and 20 default-decode official sweeps did not improve this
candidate (`epoch20 = 0.972501 / 0.017126 / 0.011019`). This is below the
env30 reference-point baseline and is not promotion evidence.

Hard selected-gated diagnostics using `segment_selection_score_mode =
unified_choice` failed to convert raw local-segment capacity:

```text
last.pt:
  official-val short GT5 base/raw/selected-gated = 40/52/40 of 53
  train0601    short GT5 base/raw/selected-gated = 142/179/142 of 183
  official-val short GT4 base/raw/selected-gated = 1/4/1 of 8
  train0601    short GT4 base/raw/selected-gated = 9/19/9 of 19

best.pt:
  official-val short GT5 base/raw/selected-gated = 40/51/40 of 53
  train0601    short GT5 base/raw/selected-gated = 142/177/142 of 183
  official-val short GT4 base/raw/selected-gated = 1/4/1 of 8
  train0601    short GT4 base/raw/selected-gated = 9/19/9 of 19

official_best.pt:
  official-val short GT5 base/raw/selected-gated = 40/52/40 of 53
  train0601    short GT5 base/raw/selected-gated = 142/179/142 of 183
  official-val short GT4 base/raw/selected-gated = 1/4/1 of 8
  train0601    short GT4 base/raw/selected-gated = 9/19/9 of 19
```

Base-hit loss was zero, but selected-gated gain was also zero on every group.
The oracle raw candidates remained eligible (`val GT5 52/53`, `train0601 GT5
179/183` for last/official_best), yet `oracle_query_selected_count`,
`oracle_query_gate_count`, and `oracle_query_selected_and_gate_count` were
all zero. Score/combined top-1 remained very low, e.g. last.pt short GT5
`score top1/top5 = 1/4` on official-val and `1/17` on train0601.

Decision: reject v13 as a selector fix. Do not run TEST, do not run formal
segment decode, and do not continue this run by adding epochs. Candidate-aware
assignment alone did not solve the class-0/base prior in the unified choice
head. A future v14 must stop using a global default base class as both
"preserve this lane" and "no short-segment candidate for this query". Train
candidate geometry quality densely over all segment windows, train base-vs-
segment choice only on assigned/protected short-lane queries, and keep
unassigned normal queries out of the base-vs-segment softmax instead of using
them as default class-0 negatives.

## 2026-08-02 Independent Full-Lane Proposal Set Decoder

The local short-segment route is closed as a formal solution path because its
oracle only covers a partial lane window. The explicit replacement path is an
image-conditioned full-lane proposal set decoder enabled only by:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-full-lane-proposal.yaml
```

The full head emits eight independent proposals, each with:

```text
pred_full_lane_points: B x 8 x 56 x 2
pred_full_lane_valid_logits: B x 8 x 56
pred_full_lane_exist_logits: B x 8
pred_full_lane_quality_logits: B x 8
pred_full_lane_start_logits: B x 8 x 56
pred_full_lane_end_logits: B x 8 x 56
pred_full_lane_score_logits: B x 8
```

When `gcs_full_lane_proposal > 0`, base queries and full proposals are
concatenated before one global Hungarian matching. Matched full proposals
receive complete geometry, visibility, interval, existence, and quality
supervision; unmatched full proposals receive negative existence/visibility
supervision. The shared proposal score is:

```text
sigmoid(existence) * sigmoid(quality) * mean(sigmoid(valid) inside predicted interval)
```

The same score helper is used by matching, direct decode, and cached decode.
`gcs_full_lane_decode` / `--full-lane-decode` are prediction-only and
default-off, and are mutually exclusive with candidate, segment, and
count-aware top-k decode paths. The default env30 YAML and output contract are
unchanged.

The completed first probe was:

```text
full_lane_proposal_env30_probe20_v1
```

It completed `20/20` epochs with `gcs_full_lane_proposal=1.0` and
`gcs_full_lane_decode=true`, but is rejected as full-lane proposal evidence.
The new proposal head received no positive matches:

```text
train/full_lane_match_count = 0 for every epoch
val/full_lane_match_count   = 0 for every epoch
train/full_lane_point_loss  = 0
val/full_lane_point_loss    = 0
train/full_lane_interval_loss = 0
val/full_lane_interval_loss   = 0
```

The full proposal loss therefore collapsed to unmatched negative supervision.
The paired official-val decode check on the same `official_best.pt` and the
same row `conf=0.003, point_valid_thr=0.6, nms_dist_px=0, max_det=6` found:

```text
base decode:      ACC/FP/FN = 0.974316 / 0.013636 / 0.011938
full-lane decode: ACC/FP/FN = 0.974316 / 0.012948 / 0.011938
```

The strict full-visible-span oracle found no complete proposal capacity from
the new full-lane proposals:

```text
official-val all: full_proposal_oracle_full_span/full_hit = 0 / 0 of 1303 lanes
official-val GT5: full_proposal_oracle_full_span/full_hit = 0 / 0 of 370 lanes
train0601 all:    full_proposal_oracle_full_span/full_hit = 0 / 0 of 1787 lanes
train0601 GT5:    full_proposal_oracle_full_span/full_hit = 0 / 0 of 1195 lanes
```

Do not promote v1, do not add epochs, and do not run TEST from this result.
The next full-lane attempt must first fix the training target so full proposals
receive independent positive geometry/visibility/interval supervision before
they compete with the mature env30 base queries in unified Hungarian matching.
Promotion still requires a full-visible-span oracle and learned set-decoder
result on official-val and train-side diagnostics, with no FP/FN or
GT3/GT4/GT5 count regression. Partial-window overlap is not a promotion metric.

The v2 warmup implementation keeps the default behavior unchanged and adds
explicit default-off controls:

```text
gcs_full_lane_aux_assignment
gcs_full_lane_unified_matching
gcs_full_lane_unmatched_weight
gcs_full_lane_aux_match_min_overlap
gcs_full_lane_aux_match_gate_px
gcs_full_lane_freeze_base
```

The dedicated warmup command is:

```text
scripts/run_full_lane_proposal_env30_auxwarm_v2.sh
```

In the v2 warmup script, full proposals are matched to GT with proposal-only
Hungarian assignment, env30/base parameters are frozen, unified base/full
matching is disabled, and unmatched negative pressure defaults to zero. The
diagnostic checkpoint is `last.pt`, not the base-only `official_best.pt`.

v2 is not a promoted decode path. Before any full-lane official decode sweep,
the run must first show:

```text
full_lane_match_count > 0
full_lane_point_loss > 0 and decreasing
full_lane_interval_loss > 0 and decreasing
official-val full_proposal_oracle_full_span > 0
train0601 full_proposal_oracle_full_span > 0
TEST closed
```

The completed v2 E5 and E5-to-20 warmup diagnostics prove this first closure
gate is necessary but not sufficient. Continuing the E5 checkpoint for 15 more
epochs improved full proposal supervision losses and full-span/full-hit counts,
but did not produce meaningful target-lane union gain:

```text
run = full_lane_proposal_env30_auxwarm_v2_e5to20_gate_fix3
checkpoint = last.pt
TEST = closed

official-val all: base/full/union full-hit = 1071 / 237 / 1088 of 1303, gain +17
official-val GT5: base/full/union full-hit = 312 / 50 / 313 of 370, gain +1
train0601 all:    base/full/union full-hit = 1468 / 233 / 1477 of 1787, gain +9
train0601 GT5:    base/full/union full-hit = 987 / 125 / 990 of 1195, gain +3

official-val gt5_vis_6_10: base/full/union full-hit = 39 / 1 / 39 of 49, gain +0
train0601 gt5_vis_6_10:    base/full/union full-hit = 130 / 1 / 130 of 170, gain +0
```

The dominant failure is full-span geometry miss, not missing positive
assignment: official-val GT5 has `249` full-span proposals but only `50`
full-hits, and train0601 GT5 has `771` full-span proposals but only `125`
full-hits. For `gt5_vis_6_10`, almost every covered proposal still misses the
20px gate (`43/44` on official-val and `153/154` on train0601). Do not promote
v2, run TEST, or enter full-lane decode from this checkpoint. A future v3 must
focus proposal assignment/loss on base-miss short GT4/GT5 hard lanes and reduce
ordinary base-hit GT supervision so the new proposals do not spend capacity
copying lanes the frozen env30 base already solves.

The v3 hard-focus follow-up was completed and also rejected. It enabled
`gcs_full_lane_hard_focus=true` with base-hit/base-miss weights
`0.05/4.0`, GT4/GT5 weights `2.0/3.0`, and short-visible weight `4.0`.
The full proposal head was still trained with proposal-only assignment and the
base path remained frozen. Same-protocol strict oracle results were:

```text
official-val gt5_vis_6_10: base/full/union full-hit = 39 / 1 / 39 of 49
train0601 gt5_vis_6_10:    base/full/union full-hit = 130 / 4 / 131 of 170
official-val GT5: base/full/union full-hit = 312 / 35 / 313 of 370
train0601 GT5:    base/full/union full-hit = 987 / 104 / 995 of 1195
```

The hard-focus weighting improved train-side oracle score rank but did not
produce consistent geometry rescue on official-val; the official GT5 full-hit
count decreased from `50` in v2 to `35` in v3. Therefore v3 is not eligible
for full-lane decode or TEST. The full-proposal warmup path is closed as a
promotion route. Future work should use an image-conditioned
instance/keypoint/segmentation-first proposal representation that generates
new lane instances from dense evidence before fitting complete K56 lane
sequences, rather than continuing to reweight the same independent proposal
head.

## 2026-08-03 Dense Instance/Keypoint Evidence Probe

The next direction is an image-conditioned dense evidence path, enabled only
by:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-dense-instance-proposal.yaml
```

The default env30 YAML and default output contract are unchanged. The dense
YAML preserves the normal query outputs and adds:

```text
pred_dense_centerline_logits: B x 1 x H/4 x W/4
pred_dense_endpoint_logits: B x 2 x H/4 x W/4
pred_dense_instance_embed: B x D x H/4 x W/4
```

The dense loss rasterizes normalized fixed-y GT lanes into centerline and
top/bottom endpoint evidence. Instance embedding pull/push is image-local:
points from the same lane are pulled together, and different lanes in the
same image are pushed apart. Embeddings from different batch images are not
compared because lane identity is not shared across images.

This is currently a diagnostic/evidence stage only. It does not assemble
connected components into lane instances, replace query predictions, alter
official decode, or open TEST. The dedicated launch script is:

```text
scripts/run_dense_instance_proposal_env30_probe20_v1.sh
```

The launch script enables `gcs_dense_freeze_base` by default. In that mode
only `dense_instance_*` parameters train; shared backbone/query parameters
remain frozen and frozen modules, including their BatchNorm statistics, stay
in eval mode. The config/CLI default remains false so the behavior is
explicit and opt-in.

The strict prediction-only diagnostic is:

```text
tools/diagnose_gcs_dense_instance_oracle.py
```

It uses GT only after model forward to measure full visible-span centerline
support, endpoint support, embedding separation, and evidence on frozen
env30 base-miss lanes. It must not be interpreted as official ACC or as a
complete lane proposal oracle. Before any dense lane assembly or decode is
implemented, the evidence stage must show nontrivial complete-span support on
official-val and train0601 base-miss GT5, with no change to the default env30
path and `TEST` closed.

The 2026-08-06 endpoint-mass and row-DP follow-up is also diagnostic-only.
`dense_instance_endpoint_mass_probe20_v3_canonical` showed strong GT-coordinate
dense evidence but failed to construct complete K56 candidates for official-val
GT5 base-miss lanes (`candidate_oracle_full_hit20=0/15` under the trace gate;
row-DP/beam still failed full-visible hit20 on the audited hard sample). It
does not enable dense decode, does not alter default env30 behavior, and must
not be promoted to official TEST or long training. A future dense route must
first pass an endpoint/pairing/embedding/K56-fitting oracle gate on
official-val plus train0601 with TEST closed.

## 2026-08-10 Residual Proposal Stage-2/Stage-3 Contract

The env30-preserving residual proposal path remains diagnostic-only and
default-off. The frozen base decode, Q12/K56 outputs, and official metric path
are unchanged. `TEST` has not been used.

Stage-2 retrieval may use horizontal-flip consistency only inside diagnostic
tools. The pre-registered `scale_px=20` ranking passed the candidate retrieval
gate, but it is not part of official inference or decode.

Stage-3 tools are also offline diagnostics:

```text
tools/diagnose_gcs_residual_set_replacement.py
tools/train_gcs_residual_set_selector.py
```

The learned selector now supports prediction-only 49-dimensional set-context
features and three utility classes (`harmful`, `neutral`, `improve`) with
image-level pairwise ranking. This implementation is not promotable: neither
selector v1 nor selector v2 approaches the top3 set oracle while preserving
victim identity, FP, lane count, and zero-harm constraints across canonical,
clean, and train0601 splits. Do not connect it to official decode.

## 2026-08-10 Residual Official-Utility Selector Contract

The residual selector remains an explicit, default-off experimental decode;
the normal env30 decode and YAML behavior are unchanged. The residual head now
also exposes read-only internal diagnostic tokens:

```text
pred_residual_base_token: B x 12 x 64
pred_residual_base_visual_token: B x 12 x 128
pred_residual_victim_pair_token: B x 8 x 12 x 64
```

The selected policy uses four independently trained selector checkpoints,
averages their utility/unique probabilities, and applies the frozen rules:

```text
residual topk = 3
flip consistency scale = 20 px
utility threshold = 0.95
replace victim unique probability <= 0.25
unique override disabled
```

Selector utility is trained from per-image TuSimple official
`Accuracy/FP/FN`, not hit20 count. GT is used only to build training labels and
offline metrics; inference uses predictions and internal model tokens only.

TuSimple TEST images in the current archive are converted `960x544` images,
while `archive/TUSimple/test_label.json` remains in original `1280x720`
coordinates. Official TEST export must therefore use:

```text
--official-image-shape 720 1280
```

The earlier Stage-13 result without that option is invalid protocol evidence
and must never be cited as model accuracy.

## 2026-08-11 Default-off query-survival diagnostic contract

The query-survival YAML remains a default-off experimental diagnostic:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-query-survival.yaml
```

It emits the normal query outputs plus:

```text
pred_base_logits: B x 12
pred_survival_delta_logits: B x 12
```

Zero initialization keeps `pred_logits == pred_base_logits`. With
`gcs_query_survival_freeze_base=true`, only `query_survival_*` parameters are
trainable; frozen env30 tensors are excluded from EMA and frozen modules remain
in `eval()` mode.

The following loss is also default off:

```text
gcs_query_survival_rank = 0.0
gcs_query_survival_rank_margin = 0.5
gcs_query_survival_rank_max_ape_px = 20.0
gcs_query_survival_rank_min_lanes = 4
```

It trains only the weakest accurate matched query against the strongest
unmatched query. Its v4 clean-val probe was rejected because epoch5 and
epoch10 exactly matched the frozen clean-val decode metrics. This diagnostic
must not be treated as active env30 behavior or as promotion evidence.

The follow-up hard-sampling v5 and geometry-adapter v6 probes are also
rejected. v5 proves that harder sampling does not change train0601 survival
when the raw query pool lacks the target geometry. v6 adds zero-initialized
`pred_survival_point_delta` and `pred_survival_valid_delta`, but clean-val
regresses at both epoch5 and epoch10; its best epoch5 ACC is `0.965310` versus
the frozen env30 `0.966132`, with worse FP, FN, and `count_acc_4`.

These outputs remain default-off diagnostics only. Do not run v5/v6 on TEST,
repeat seeds, extend training, enlarge the point delta, or treat the geometry
adapter as an active branch mechanism. Future work must add missing-lane
spatial evidence rather than only deform or rerank existing frozen queries.
