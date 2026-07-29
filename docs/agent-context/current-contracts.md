# Current Contracts

This file records the active contracts for branch `codex/5-25-3-k56`.

## Branch Scope

This branch imports the historical `5-25-3.zip` algorithm and changes only the TuSimple fixed-y contract:

```text
legacy archive: Q=8, K=32, fixed_y=[0.98, 0.25]
this branch:    Q=12, K=56, fixed_y=[710/720, 160/720]
```

Do not silently import later mainline mechanisms such as Count Head, Quality Head, Survival Head, or near-miss mining into this branch unless a future task explicitly asks for that algorithm change. The branch now includes the 2026-06-27 user-requested, default-off `count_boundary_loss` for adjacent GT3/GT4/GT5 count-score boundaries; this is not a Count Head or decode change.

## 2026-07-28 Env30 Baseline Boundary

Active source/config is restored to env30 commit
`86c8fb31cb4b48a53086be183478a95b0807753d` (`Add GT4 GT5 weak geometry rescue
run`). Outside documentation and explicitly reopened default-off v2 candidate
files, tracked code/config/script/tool/reference-bank content must match that
env30 baseline.

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

## 2026-07-28 User-Reopened Q12/env30 Gated Candidate v2

The user explicitly reopened the lateral candidate-generation route on
2026-07-28 with new gates. This is a new default-off v2 implementation, not a
promotion or relaunch of the rejected 2026-07-27 candidate probe.

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
selection and test only once for final evaluation.

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
