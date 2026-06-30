# Current Contracts

This file records the active contracts for branch `codex/5-25-3-k56`.

## Branch Scope

This branch imports the historical `5-25-3.zip` algorithm and changes only the TuSimple fixed-y contract:

```text
legacy archive: Q=8, K=32, fixed_y=[0.98, 0.25]
this branch:    Q=12, K=56, fixed_y=[710/720, 160/720]
```

Do not silently import later mainline mechanisms such as Count Head, Quality Head, Survival Head, or near-miss mining into this branch unless a future task explicitly asks for that algorithm change. The branch now includes the 2026-06-27 user-requested, default-off `count_boundary_loss` for adjacent GT3/GT4/GT5 count-score boundaries; this is not a Count Head or decode change.

The branch also includes the 2026-06-27 user-requested, default-off `gcs_hard_sampling` train-only sampler for short-visible GT3/GT4/GT5 and 0601 samples. It changes only the training dataloader sampling frequency through `WeightedRandomSampler`; it does not change labels, validation/test dataloaders, point/smooth/curve losses, decode, or official metrics.

The branch also includes the 2026-06-27 user-requested, default-off `gcs_spurious_neg` loss for E3-lite. It uses the training Hungarian matcher indices only to select unmatched short duplicate-like queries near matched queries, then adds an extra target-zero BCE on their `pred_logits`. The GT-count weighting extension keeps the old default behavior with `gcs_spurious_gt3_weight=1.0`, `gcs_spurious_gt4_weight=1.0`, `gcs_spurious_gt5_weight=1.0`, and `gcs_spurious_disable_gt5=False`, while allowing GT3-or-sparser, GT4, and GT5-or-denser samples to carry different spurious-negative weights. The 2026-06-28 `gcs_spurious_gt_protect` extension is also default-off and only removes GT-close candidate queries from this extra negative BCE. It does not change data sampling, dataset labels, matcher logic, point/smooth/curve losses, decode, NMS, or official metrics.

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
  descending `710,700,...,160`; official TuSimple `h_samples` validate
  separately against ascending `160,170,...,710`.
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
  `tools/sweep_tusimple_official.py` or `tools/eval_tusimple_official.py` with
  auto decode.
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

All branch configs keep `Q=12`, `K=56`, fixed-y anchors `710/720 -> 160/720`, and `--imgsz 544 960`.

Historical q12-k56 experiment docs are old records. Preserve them, but do not let them override the active 5-25-3 K56 mainline contract.

Active source/config is rolled back to commit `b6535f641` (`Fix GCS training
progress header alignment`). Its algorithm contract remains the 5-25-3 K56
mainline: no later Count Head, Q18/Q20/dataref, duplicate/spurious/ranking,
lane-balanced, valid-repair, side-aux, legacy `gcs_gt4_short_boost` sampling,
`extra_exist_loss`, short matched existence floor, or count-confusion
diagnostic tooling is active. Later commits and notes are preserved only as
legacy experiment conclusions in the docs. They are not active CLI, loss,
model-output, tool, or config contracts in this code state. The explicit
default-off `gcs_spurious_neg` E3-lite loss above is branch-local and is not an
import of the later legacy spurious/ranking loss family.

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

## Loss Contract

Default logged loss items on this branch:

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
```

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

`gcs_spurious_gt_protect=False` preserves the old E3-lite spurious-negative
selection. When enabled, each duplicate-like spurious candidate is compared
against all GT lanes. If the candidate overlaps a GT lane by at least
`gcs_spurious_gt_protect_min_overlap` anchors and has mean x distance at most
`gcs_spurious_gt_protect_px` pixels, `better_matched` mode protects it when the
GT lane has no matched query, the matched query has insufficient GT overlap, or
the candidate mean GT x distance plus `gcs_spurious_gt_protect_margin_px` is
lower than the matched query mean GT x distance. Protected candidates are
excluded only from the extra spurious target-zero BCE. `spur_cand` logs the
pre-protect candidate count, `spur_prot` logs protected candidates,
`spur_final` and `spurious_negative_count` log final selected negatives, and
`spur_neg` mirrors `spurious_neg_loss` for compact progress logging.

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

This branch includes `tools/eval_tusimple_official.py`,
`tools/sweep_tusimple_official.py`, `gcs_tools/tusimple_official_eval.py`, and
explicit training-time `official_best` checkpoint preservation. It does not
include `tools/diagnose_tusimple_count_confusion.py`,
`tools/diagnose_gcs_gt5.py`, or later mainline Count/Quality/Boundary
diagnostics unless a future task explicitly ports them. Use the canonical
363-image official-val GT for selection and test only once for final
evaluation.

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
