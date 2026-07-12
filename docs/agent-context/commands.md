# Commands

This file records commands for branch `codex/5-25-3-k56`, which imports the historical `5-25-3.zip` algorithm and changes only the TuSimple K56 fixed-y contract.

All TuSimple commands must use:

```bash
--imgsz 544 960
```

This is H,W order.

## Branch Contract

```text
model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
data:  data/tusimple_gcs_fixed_y_960x544.yaml
root:  datasets/tusimple_fixed_y_k56_960x544
Q:     12
K:     56
fixed_y_start: 710 / 720 = 0.9861111111111112
fixed_y_end:   160 / 720 = 0.2222222222222222
```

The q12-k56-named model/data files are compatibility paths for old experiment records and keep the same K56 fixed-y contract. New commands should use the mainline paths above.

The 5-25-3 branch now includes explicit branch-local `--gcs-official-best` training-time checkpoint selection. It does not include later mainline Count/Quality/Survival, near-miss, or K56 candidate machinery. The only Count Head available on this branch is the 2026-07-06 user-requested, default-off query Count Head ablation enabled by `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count.yaml`. It also includes branch-local TuSimple official eval/sweep helpers: `tools/eval_tusimple_official.py`, `tools/sweep_tusimple_official_cached.py`, and `tools/sweep_tusimple_official.py`. New threshold scans should use the cached helper.

Active source/config is rolled back to commit `424ab1c86` (`Add
valid-before-maxdet decode option`). Sections below that mention
post-`424ab1c86` mechanisms such as short-side hardset diagnostics,
`gcs_short_side_geom`, `gcs_far_spurious_neg`, `gcs_farspur_*`,
`gcs_shortside_*`, `gcs_rank_*`, count-contract diagnostics, Q18/Q20/dataref
configs, Count Head, count-guided decode, side-aux, or GT4-hard diagnostics
are legacy experiment records only. They are not commands for the current code
state unless a future task explicitly restores those commits.

## Query Count Head CE0.5 Run

The default-off query Count Head ablation is launched through:

```bash
bash scripts/run_query_count_head_ce05_v1.sh
```

As of 2026-07-07 the script's default `RUN_NAME` is
`query_count_head_ce05_fixedbest_sweep2520_v1`. It trains the query-count YAML
with `--gcs-query-count-ce 0.5`, disables the old score-sum count losses, uses
training-time `--gcs-official-best` with fixed official-val decode
`conf=0.005`, `point_valid_thr=0.45`, `nms_dist_px=0`, `max_det=5`,
`min_points=2`, and then runs a post-train official-val sweep over
`weights/official_best.pt`. Keep all selection on official-val.

Completed predecessor result:

```text
run = query_count_head_ce05_v1
selected weights = runs/gcs_lane/query_count_head_ce05_v1/weights/official_best.pt
selected decode = runs/gcs_lane/query_count_head_ce05_v1/weights/official_best_decode.yaml
selected official-val ACC = 0.968473
selected official-val FP = 0.015152
selected official-val FN = 0.011938
selected official-val count_acc = 0.931129
```

The post-train `best.pt` val sweep
`runs/gcs_lane/query_count_head_ce05_v1_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json`
is weaker on official-val (`ACC=0.967679`, `FP=0.032553`,
`FN=0.015152`, `count_acc=0.906336`) and must not replace the training-time
official-best selection.

Reporting-only official-test summaries:

```text
official_best test = runs/gcs_lane/query_count_head_ce05_v1_official_test_official_best_decode/tusimple_official_summary.json
official_best test ACC = 0.964862
official_best test FP = 0.033914
official_best test FN = 0.024473
official_best test count_acc = 0.874191

best.pt sweep-threshold test = runs/gcs_lane/query_count_head_ce05_v1_best_official_test_from_val_sweep_valid_before_maxdet_b/tusimple_official_summary.json
best.pt sweep-threshold test ACC = 0.964518
best.pt sweep-threshold test FP = 0.037653
best.pt sweep-threshold test FN = 0.026779
best.pt sweep-threshold test count_acc = 0.836089
```

These test values are reporting-only. Do not use them for thresholds,
checkpoint choice, NMS, `max_det`, `min_points`, count mode, count-aware top-k,
or loss-gain tuning.

## Query Alpha05 GT4/GT5 Tiered Geometry Env30 Run

The next official-val-only experiment after the env30 boundary-mask diagnostic
is:

```bash
cd /root/GCS-YOLO-Lane_LSA_5-25-3-k56
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssh_lane
bash scripts/run_query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_v1.sh
```

The script default run name is:

```text
query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_v1
```

Its purpose is to keep the env30 boundary mask while rescuing matched weak
GT4/GT5 lanes. Do not use it as a test-tuning script. By default,
`RUN_TESTS=0`, and the script runs training-time `official_best`, post-train
cached official-val sweeps for both `weights/official_best.pt` and
`weights/best.pt`, plus validation raw-Q12 diagnostics.

When a candidate has already passed the official-val gates and the user
explicitly wants the reporting-only full protocol, use:

```bash
bash scripts/run_query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_full_protocol_v1.sh
```

This wrapper keeps the same tiered GT4/GT5 and env30 parameters, but defaults
`RUN_TESTS=1`, `RUN_DIAGNOSTICS=1`, and `RUN_TEST_DIAGNOSTICS=1`. It still
selects thresholds only from cached official-val sweeps; TEST ACC and TEST
raw-Q12 diagnostics are reporting-only and must not be used to choose
thresholds, checkpoints, decode settings, or loss gains.

Core weak-positive settings:

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

Retained env30 boundary-mask settings:

```text
gcs_boundary_pseudo_neg = 0.02
gcs_boundary_pseudo_dist_thr = 80
gcs_boundary_pseudo_min_valid = 4
gcs_boundary_pseudo_score_thr = 0.2
gcs_boundary_pseudo_envelope_margin_px = 30
gcs_boundary_pseudo_envelope_ratio_thr = 0.75
```

Primary gates before any reporting-only test:

```text
official-val ACC/FN >= env30 or tied within noise
GT4 4->5 must not regress
GT5 5->4 must not regress
GT5 5->6 must not regress under max_det=6
GT5 visible<=10 raw p90 APE must improve versus 36.550196 px
train GT4->5 must drop below 38
```

If reusing an existing run directory after training, use `RUN_TRAIN=0`. If a
sweep or diagnostic directory already exists and must be regenerated, set only
the relevant overwrite flag:

```bash
RUN_TRAIN=0 OVERWRITE_SWEEPS=1 OVERWRITE_DIAGNOSTICS=1 \
bash scripts/run_query_alpha05_gt4gt5_tiered_geom_gt4pv12_env30_nocount_v1.sh
```

The previous tiered GT5-only point-valid wrapper
`scripts/run_query_alpha05_gt4gt5_tiered_geom_env30_nocount_v1.sh`
remains available for reproducing the tiered-geometry + GT5PV setup without
GT4 point-valid rescue.

The previous non-tiered script
`scripts/run_query_alpha05_gt4gt5weak_geom_w15w2_env30_nocount_v1.sh`
remains available for reproducing the legacy single-threshold w15/w2 run.

## Full Remote Training

Use the remote RTX 4090 environment for formal training:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssh_lane
```

Run from the dedicated remote Git clone checked out to branch `codex/5-25-3-k56` or an exact commit from that branch:

```bash
python tools/train_gcs.py \
  --dataset tusimple \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --pretrained yolo11s-seg.pt \
  --imgsz 544 960 \
  --epochs 160 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --no-amp \
  --optimizer AdamW \
  --lr0 5e-4 \
  --lrf 0.05 \
  --cos-lr \
  --weight-decay 1e-4 \
  --warmup-epochs 3.0 \
  --warmup-bias-lr 0.0 \
  --patience 40 \
  --erasing 0.1 \
  --scale 0.3 \
  --gcs-exist 2.0 \
  --gcs-point 15.0 \
  --gcs-point-valid 1.0 \
  --gcs-smooth 0.05 \
  --gcs-curve 0.1 \
  --gcs-mask 0.2 \
  --gcs-edge 0.2 \
  --gcs-count 0.2 \
  --gcs-count-under5 0.3 \
  --gcs-count-under5-min-lanes 5 \
  --gcs-lane-count-balanced \
  --gcs-official-best \
  --gcs-official-interval 5 \
  --gcs-official-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-half \
  --project runs/gcs_lane \
  --name gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count02_under5_03
```

If `batch=32` OOMs on the target machine, reduce batch only for OOM/instability and record the change in the run notes.

`--no-amp` is included because the current remote run hit an Ultralytics AMP self-check failure while loading `yolo26n.pt`. If that server cache/checkpoint issue is fixed, AMP may be re-enabled only with a run note.

## Completed E2 Short0601 Hard-Sampling Result

The E2 hard-sampling run from the E1 count-boundary checkpoint is rejected for
promotion. It is useful only as diagnostic evidence.

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_short0601_v1
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
gcs_count_boundary = 0.2
gcs_hard_sampling = true
gcs_official_best = false
results.csv rows = 64 epochs
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_short0601_v1_official_val_sweep/tusimple_official_sweep_summary.json
best decode = conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=2
official-val ACC = 0.969988
official-val FP = 0.022544
official-val FN = 0.018825
official-val count_acc_4 = 0.969697
official-val count_acc_5 = 0.959459
count_confusion = 3->3=216, 3->4=7, 4->3=1, 4->4=64, 4->5=1, 5->4=3, 5->5=71
```

Against E1, `4->5` improves from `9` to `1` and `count_acc_4` improves from
`0.848485` to `0.969697`, but official ACC drops from `0.971208` to
`0.969988` because FN rises from `0.014004` to `0.018825`. Do not run final
test for this checkpoint and do not use it as the starting point for E3-lite
spurious-negative ablations.

## E3-Lite GT4-Strong + GT5-Safe Spurious Negative From E1

E3-lite GT-count-weighted spurious-negative experiments must initialize from
the E1 count-boundary checkpoint, not from E2 hard sampling or count-aware
top-k results. Keep hard sampling disabled.

For new formal runs, use the training-time official-best protocol below. If a
run was trained without `--gcs-official-best`, record it explicitly as a
post-hoc official-val sweep over that run's checkpoint.

```bash
python tools/train_gcs.py \
  --dataset tusimple \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --pretrained runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt \
  --imgsz 544 960 \
  --epochs 160 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --no-amp \
  --optimizer AdamW \
  --lr0 5e-4 \
  --lrf 0.05 \
  --cos-lr \
  --weight-decay 1e-4 \
  --warmup-epochs 3.0 \
  --warmup-bias-lr 0.0 \
  --patience 40 \
  --erasing 0.1 \
  --scale 0.3 \
  --gcs-exist 2.0 \
  --gcs-point 15.0 \
  --gcs-point-valid 1.0 \
  --gcs-smooth 0.05 \
  --gcs-curve 0.1 \
  --gcs-mask 0.2 \
  --gcs-edge 0.2 \
  --gcs-count 0.3 \
  --gcs-count-under5 0.3 \
  --gcs-count-under5-min-lanes 5 \
  --gcs-count-boundary 0.2 \
  --gcs-spurious-neg 0.1 \
  --gcs-spurious-neg-weight 1.0 \
  --gcs-spurious-gt3-weight 1.0 \
  --gcs-spurious-gt4-weight 1.5 \
  --gcs-spurious-gt5-weight 0.25 \
  --gcs-spurious-max-points 12 \
  --gcs-spurious-close-px 30.0 \
  --gcs-spurious-min-overlap 3 \
  --gcs-lane-count-balanced \
  --gcs-official-best \
  --gcs-official-interval 5 \
  --gcs-official-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-half \
  --project runs/gcs_lane \
  --name gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt4strong_gt5safe_v1
```

Select only on official-val. Compare against E1 on `official_acc`,
`official_FP`, `official_FN`, `count_acc_4`, `count_acc_5`, and count-confusion
`4->5` / `5->4`.

The default `gcs_spurious_gt3_weight=1.0`,
`gcs_spurious_gt4_weight=1.0`, and `gcs_spurious_gt5_weight=1.0` settings preserve the
original spurious-lite behavior. Use `--gcs-spurious-gt4-weight 1.5` and
`--gcs-spurious-gt5-weight 0.25` for the GT4-strong + GT5-safe follow-up that
presses GT4 false fifth lanes harder while reducing pressure on true short
fifth lanes. `--gcs-spurious-disable-gt5` is the harder ablation that skips the
spurious-negative loss entirely on `gt_lanes >= 5` samples.

Completed diagnostic result:

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
gcs_spurious_neg = 0.1
gcs_hard_sampling = false
gcs_official_best = false
results.csv rows = 59 logged epochs
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1_official_val_sweep/tusimple_official_sweep_summary.json
sweep rows = 3360, count_aware_topk = false
best decode = conf=0.005, point_valid_thr=0.45, nms_dist_px=0.0, max_det=5, min_points=2
official-val ACC = 0.971721
official-val FP = 0.013866
official-val FN = 0.015611
official-val count_acc_4 = 0.924242
official-val count_acc_5 = 0.878378
count_confusion = 3->3=220, 3->4=3, 4->3=1, 4->4=61, 4->5=4, 5->4=9, 5->5=65
```

Against E1
`gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1`, spurious-lite improves
official-val ACC by `+0.000513`, lowers FP by `0.009091`, and reduces GT4
`4->5` from `9` to `4`. It also worsens GT5 retention: `5->4` increases from
`2` to `9`, and `count_acc_5` falls from `0.972973` to `0.878378`.

Decision: do not promote `spurious_lite_v1` and do not run final test for it.
Use it as diagnostic evidence that the loss suppresses false fifth lanes but
needs GT5-safe weighting before selection. The selected metrics are shared by
36 sweep rows under the branch selection priority, so the concrete decode above
is one representative selected row rather than evidence that `nms_dist_px=0`
or `max_det=5` is uniquely better.

Completed GT5-safe follow-up result:

```text
run = gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt5safe_v1
pretrained = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt
gcs_spurious_neg = 0.1
gcs_spurious_gt5_weight = 0.25
gcs_hard_sampling = false
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt5safe_v1_official_val_sweep/tusimple_official_sweep_summary.json
best decode = conf=0.005, point_valid_thr=0.55, nms_dist_px=18.0, max_det=5, min_points=2
official-val ACC = 0.971777
official-val FP = 0.022957
official-val FN = 0.015611
official-val count_acc_4 = 0.863636
official-val count_acc_5 = 0.986486
count_confusion = 3->3=218, 3->4=5, 4->3=1, 4->4=57, 4->5=8, 5->4=1, 5->5=73
```

This GT5-safe run recovers GT5 (`5->4=1`, `count_acc_5=0.986486`) but does not
retain the spurious-lite GT4 / FP benefit: `4->5` rebounds to `8`, `count_acc_4`
is only `0.863636`, and FP returns to the E1 value. It is diagnostic-only.

Completed GT4-strong + GT5-safe follow-up result:

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
results.csv rows = 43 epochs, nonfinite_count = 0
sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_gt4strong_gt5safe_v1_official_val_sweep/tusimple_official_sweep_summary.json
best decode = conf=0.003, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=2
official-val ACC = 0.970530
official-val FP = 0.025666
official-val FN = 0.016529
official-val count_acc_4 = 0.848485
official-val count_acc_5 = 0.986486
count_confusion = 3->3=216, 3->4=7, 4->3=1, 4->4=56, 4->5=9, 5->4=1, 5->5=73
```

Reject this artifact. It preserves GT5 retention but fails the GT4/FP target:
the best selected row is below E1 on ACC, FP, and FN, and GT4 `4->5` stays at
the E1 value `9`. Across 1512 official-val sweep rows, no row satisfies the
user gate; the best individual limits were `max_acc=0.970530`,
`min_FP=0.020707`, `max_count_acc_4=0.893939`, `min_4->5=6`, and
`min_5->4=1`. Do not run final test for this checkpoint.

## Training-Time Official-Best Selection

Formal TuSimple training must not select the final checkpoint from `val/total_loss`, internal `val/f1`, or generic `weights/best.pt` alone. Use `--gcs-official-best` so training runs a cached official-val sweep every 5 epochs and again on the final/early-stop epoch.

The selected artifacts are:

```text
weights/official_best.pt
weights/official_best_sweep.json
weights/official_best_decode.yaml
official_sweeps/epoch*/tusimple_official_sweep_summary.json
official_sweeps/epoch*/prediction_cache/manifest.json
official_sweeps/epoch*/prediction_cache/predictions.pt
```

`official_best_decode.yaml` is schema-specific. Query checkpoints write
`query_decode_v1` with query decode arguments. Ordered-slot checkpoints write
`ordered_slot_decode_v1` only, with the 2/3/4/5 count contract and
`effective_decode`; ordered-slot official-best decode uses `output_order=slot`,
`uses_runtime_sort=false`, and `order_violation_policy=fail_fast`.
Ordered-slot decode yaml must not contain `conf`, `point_valid_thr`,
`nms_dist_px`, `max_det`, `min_points`, or `count_aware_topk`.

Selection priority:

1. maximum `official_acc`
2. if tied, maximum `official_score`
3. if tied, lower `official_FP`
4. if tied, lower `official_FN`
5. if tied, higher `count_acc_4`

The 2026-06-27 checkpoint-selection failure case that motivated this rule:

```text
best val/f1:         epoch 91  = 0.971997
best val/total_loss: epoch 116 = 0.69639
last epoch 131:      val/f1    = 0.968382
```

This shows why `weights/best.pt` and `val/total_loss` are insufficient selection surfaces. Use `weights/official_best.pt` plus the decode in `weights/official_best_decode.yaml` for final-test reporting.

## Legacy Positive Short-Lane Point/Visibility Experiment

This default-disabled experiment targets true short-lane learning, not
unmatched-query suppression. It keeps decode, NMS, official metrics, and final
test closed:

```bash
python tools/train_gcs.py \
  --dataset tusimple \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --pretrained yolo11s-seg.pt \
  --imgsz 544 960 \
  --epochs 160 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --no-amp \
  --optimizer AdamW \
  --lr0 5e-4 \
  --lrf 0.05 \
  --cos-lr \
  --weight-decay 1e-4 \
  --warmup-epochs 3.0 \
  --warmup-bias-lr 0.0 \
  --patience 40 \
  --erasing 0.1 \
  --scale 0.3 \
  --gcs-exist 2.0 \
  --gcs-point 15.0 \
  --gcs-point-valid 1.0 \
  --gcs-smooth 0.05 \
  --gcs-curve 0.1 \
  --gcs-mask 0.2 \
  --gcs-edge 0.2 \
  --gcs-count 0.3 \
  --gcs-count-under5 0.3 \
  --gcs-count-under5-min-lanes 5 \
  --gcs-lane-count-balanced \
  --gcs-lane-count-balance-power 1.0 \
  --gcs-lane-count-min-group 50 \
  --gcs-duplicate-margin 0.0 \
  --gcs-spurious-margin 0.0 \
  --gcs-lane-balanced-point 3.0 \
  --gcs-short-valid-recall 0.5 \
  --gcs-short-valid-max-visible 20 \
  --gcs-short-valid-min-visible 4 \
  --gcs-short-valid-max-ape-px 40.0 \
  --gcs-short-valid-min-visible-iou 0.3 \
  --project runs/gcs_lane \
  --name gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03
```

Promotion must use official-val only. Historical first-pass gates used for this
run family:

```text
ACC >= 0.970272, preferably near or above 0.970851
FN <= 0.016070
GT5 count_acc should not fall far below 0.972973
GT4 count_acc should not drop sharply
APE / matched geometry metrics should not regress
```

After the 2026-06-25 `v2_validbranch_neg05-3` result, the strict official-val
ACC gate is `0.971029`; the previous gate was `0.970975` from `gt4pt025`.

After training, run train/val failure traces and check that
`geometry_miss_short_gt` and `min_points_visibility_short_gt` decrease without
new `5->4` growth or score-threshold-only `4->5` behavior.

Completed result:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03_official_val_sweep
best: conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=5
official-val363: ACC=0.968144, FP=0.027319, FN=0.017218, official_score=0.967253, count_acc=0.947658
count_acc_3=0.959641, count_acc_4=0.878788, count_acc_5=0.972973
```

One-shot official test used only the official-val selected decode:

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.005 \
  --point-valid-thr 0.5 \
  --nms-dist-px 0.0 \
  --max-det 6 \
  --min-points 5 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03_official_test_best_from_val \
  --save-records
```

Result:

```text
summary: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
images=2782
ACC=0.963067
FP=0.038330
FN=0.030763
official_score=0.961685
count_acc=0.861251
count_acc_2=0.400000
count_acc_3=0.957471
count_acc_4=0.547009
count_acc_5=0.829525
pred_lanes_hist: 2=4, 3=1801, 4=362, 5=574, 6=41
gt_lanes_hist: 2=5, 3=1740, 4=468, 5=569
count_confusion: 2->2=2, 2->3=2, 2->4=1, 3->2=2, 3->3=1666, 3->4=63, 3->5=7, 3->6=2, 4->3=109, 4->4=256, 4->5=95, 4->6=8, 5->3=24, 5->4=42, 5->5=472, 5->6=31
```

Do not promote this run. It missed the official-val ACC gate before test
reporting and the reporting-only test result is below `count03_under5_03`,
`gt4short15`, `dupmargin005`, `spurmargin003`, and `count03_under5_00`. The
selected `conf=0.005` points to poor score calibration, while FP and FN both
increase. Do not tune thresholds or loss weights from this test report.

The required train/val-only failure trace was completed without using final
test for selection:

```text
summary: runs/gcs_lane/shortpos_failure_trace_train_val_compare/summary.json
splits: train + val
```

Key result: `shortpos` does reduce `low_score_short_gt` and
`min_points_visibility_short_gt` to zero across train+val, but it is a bad
tradeoff. Compared with `count03_under5_03`, it adds `+68` failure images,
`+52` `spurious_extra`, `+26` `duplicate_like_extra`, and `+33`
`duplicate_like_extra_short_gt`; compared with `dupmargin005`, it adds `+60`
failure images, `+30` `spurious_extra`, `+18` `duplicate_like_extra`, and
`+33` `duplicate_like_extra_short_gt`. Do not continue this exact positive
short-lane loss setting.

## Legacy Post-b653 Official-Val Selection

The 2026-06-21 `gt4short15` candidate was selected on official-val in a later
post-`50999d6af` experiment line. After the rollback, it is a legacy result,
not the active code baseline:

```text
source run:   runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03
weights:      weights/best.pt
val sweep:    runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val_sweep
decode:       conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
official-val: ACC=0.970851, FP=0.022084, FN=0.011708, count_acc=0.939394
```

This beats the previous selected official-val ACC by `+0.000875`, but it does so by lowering FN while worsening lane-count accuracy:

```text
previous count03_under5_03 official-val: ACC=0.969976, FP=0.019559, FN=0.014463, count_acc=0.969697, count_acc_4=0.909091
gt4short15 official-val:                ACC=0.970851, FP=0.022084, FN=0.011708, count_acc=0.939394, count_acc_4=0.848485
```

The required train/val count-confusion diagnostic and one-shot final-test report for `gt4short15` are complete:

```text
diagnostic: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_count_confusion_train_val_by_visibility/summary.json
final test: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json

train diagnostic count_acc = 0.952191
val diagnostic count_acc = 0.920110
val diagnostic GT3/GT4/GT5 count_acc = 0.925234 / 0.929167 / 0.733333

final test ACC = 0.965369
final test FP = 0.033309
final test FN = 0.029236
final test count_acc = 0.864486
final test count_acc_4 = 0.482906
final test count_acc_5 = 0.845343
```

This final-test report is reporting-only and did not improve over the previous `count03_under5_03` final-test ACC `0.965459`. Treat `gt4short15` as a legacy official-val-selected experiment, not as the active rollback baseline; do not claim a final-test improvement and do not tune from final test. The final-test run used `max_det=6` because that was the official-val selected postprocess, even though the train-time args recorded `gcs_eval_max_det=8`.

Two follow-up diagnostics were run before changing any loss:

```text
failure trace:
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_failure_trace_train_val/failure_trace_summary.json
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_failure_trace_train_val/failure_trace_records.csv

NMS-only 363 official-val sweep:
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val363_nms_only_conf015_maxdet6_minp4_half/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val363_nms_only_conf015_maxdet6_minp4_half/tusimple_official_sweep.csv
```

The NMS-only sweep must use the explicit 363-image official-val GT json. A
plain `--split val` sweep covers `images=2858` and is not the selection surface:

```bash
python tools/sweep_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --half \
  --confs 0.15 \
  --point-valid-thrs 0.5 \
  --max-dets 6 \
  --min-points 4 \
  --nms-dist-pxs 0 2 4 6 8 10 12 14 16 18 20 24 30 40 50 60 80 \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val363_nms_only_conf015_maxdet6_minp4_half
```

Result: keep `nms_dist_px=0.0`. `nms=0/2/4/6` match the selected row
(`official_acc=0.970851`, `FP=0.022084`, `FN=0.011708`,
`count_acc=0.939394`). From `nms=8` onward, NMS lowers some overcount but raises
FN to `0.012397` and reduces official ACC. At `nms=80`, count accuracy improves
to `0.953168`, but official ACC drops to `0.970679`. This is diagnostic-only
postprocess evidence, not a new selected decode.

## Legacy Extra Exist Suppression Loss Experiment

The high-score unmatched-query suppression loss was an explicit post-`50999d6af`
experimental option. It is not present in the active rollback code. Historical
defaults were:

```text
gcs_extra_exist = 0.0
gcs_extra_exist_thr = 0.15
```

It uses the training Hungarian matcher only: unmatched queries with detached
`sigmoid(pred_logits) >= gcs_extra_exist_thr` receive a target-zero BCE penalty.
Do not add NMS, `max_det`, `min_points`, decoded lane count, or GT-count gating
inside this training loss.

Completed experiments:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist005_count03_under5_03
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist005_count03_under5_03_official_val_sweep
best: conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=5
official-val ACC=0.969603, FP=0.017815, FN=0.012856, official_score=0.968990, count_acc=0.961433
count_acc_3=0.977578, count_acc_4=0.878788, count_acc_5=0.986486
```

Decision: do not promote `extraexist005`. It improved FP and count robustness
relative to `gt4short15`, but it missed the then-current official-val ACC gate
`0.970851` by `0.001248` and also stayed below the previous
`count03_under5_03` official-val ACC `0.969976`. Its best row requiring
`conf=0.005` shows the extra existence penalty over-suppressed query scores.

The smaller gain run also fails and should stop this loss-family sweep:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist0025_count03_under5_03
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist0025_count03_under5_03_official_val_sweep
best: conf=0.05, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
official-val ACC=0.961513, FP=0.049633, FN=0.029844, official_score=0.959923, count_acc=0.887052
count_acc_3=0.928251, count_acc_4=0.772727, count_acc_5=0.864865
```

At the then-current `gt4short15` decode (`conf=0.15`, `point_valid_thr=0.5`,
`nms_dist_px=0.0`, `max_det=6`, `min_points=4`), the `extraexist0025` run gets
only `ACC=0.961450`, `FP=0.045638`, `FN=0.029844`, `count_acc=0.909091`. Its
best count row reaches only `count_acc=0.928375` with `ACC=0.961400`, still
below then-current `gt4short15` count accuracy and far below the ACC gate. No row in
the 1800-row sweep reaches `extraexist005` ACC `0.969603`, previous
`count03_under5_03` ACC `0.969976`, or then-current `gt4short15` ACC `0.970851`.

Default protocol decision: do not promote `extraexist0025`, keep final test
closed unless explicitly requested for reporting-only evidence, and do not
continue sweeping `gcs_extra_exist` gains. Return to train-side short-lane
score/geometry retention work; select any new mechanism only on the same
363-image official-val surface.

## Legacy Short Matched Existence Floor Experiment

The short matched existence floor was an explicit post-`50999d6af`
experimental option inside `GCSLoss.exist_loss()`. It is not present in the
active rollback code. It did not add a new loss item. Historical defaults were:

```text
gcs_short_exist_floor = 0.0
gcs_short_exist_max_visible = 20
gcs_short_exist_floor_max_ape = 20.0
gcs_short_exist_floor_min_iou = 0.3
```

It applies only to Hungarian-matched GT lanes whose visible-anchor count is at
or below `gcs_short_exist_max_visible`, whose matched APE is at or below
`gcs_short_exist_floor_max_ape`, and whose matched visible IoU is at or above
`gcs_short_exist_floor_min_iou`. The target floor is applied after the existing
APE quality and visible-IoU quality are computed.

Completed first experiment:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_shortexist04_count03_under5_03
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_shortexist04_count03_under5_03_official_val_sweep
rows: 1800
images: 363
gt_json: runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json
best: conf=0.1, point_valid_thr=0.5, nms_dist_px=30.0, max_det=8, min_points=5
official-val ACC=0.968966, FP=0.019972, FN=0.014922, official_score=0.968268, count_acc=0.950413
count_acc_3=0.968610, count_acc_4=0.863636, count_acc_5=0.972973
```

Default protocol decision: do not promote `shortexist04`, keep final test
closed unless explicitly requested for reporting-only evidence, and keep
`gt4short15` as the then-current official-val selected candidate. The short matched
existence floor improved FP and count accuracy relative to `gt4short15`, but it
missed the official-val ACC gate `0.970851` by `0.001885` and raised FN from
`0.011708` to `0.014922`. No row in the 1800-row sweep reached current
`gt4short15` ACC, previous `count03_under5_03` ACC `0.969976`, or rejected
`extraexist005` ACC `0.969603`.

User-requested reporting-only final-test batch:

```text
baseline count03_under5_03: ACC=0.965459, FP=0.029439, FN=0.026270, count_acc=0.872753
gt4short2:                 ACC=0.965206, FP=0.026432, FN=0.027079, count_acc=0.884256
gt4short15:                ACC=0.965369, FP=0.033309, FN=0.029236, count_acc=0.864486
extraexist005:             ACC=0.965032, FP=0.029661, FN=0.027139, count_acc=0.878864
shortexist04:              ACC=0.964992, FP=0.031381, FN=0.028876, count_acc=0.875988
extraexist0025:            ACC=0.961330, FP=0.044764, FN=0.034508, count_acc=0.838605
```

These test runs are reporting-only results from each run's official-val
selected decode. They do not reopen test for threshold, checkpoint, or
postprocess selection and do not change the official-val promotion decisions.

Reproducibility note: the actual `args.yaml` for this completed run records
`amp: true`, while the template below includes `--no-amp`. Treat the recorded
run args as authoritative for the completed result.

Original experiment template:

```bash
python tools/train_gcs.py \
  --dataset tusimple \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --pretrained yolo11s-seg.pt \
  --imgsz 544 960 \
  --epochs 160 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --no-amp \
  --optimizer AdamW \
  --lr0 5e-4 \
  --lrf 0.05 \
  --cos-lr \
  --weight-decay 1e-4 \
  --warmup-epochs 3.0 \
  --warmup-bias-lr 0.0 \
  --patience 40 \
  --erasing 0.1 \
  --scale 0.3 \
  --gcs-exist 2.0 \
  --gcs-point 15.0 \
  --gcs-point-valid 1.0 \
  --gcs-smooth 0.05 \
  --gcs-curve 0.1 \
  --gcs-mask 0.2 \
  --gcs-edge 0.2 \
  --gcs-count 0.3 \
  --gcs-count-under5 0.3 \
  --gcs-count-under5-min-lanes 5 \
  --gcs-lane-count-balanced \
  --gcs-lane-count-balance-power 1.0 \
  --gcs-lane-count-min-group 50 \
  --gcs-gt4-short-boost 1.5 \
  --gcs-gt4-short-min-visible-max 10 \
  --gcs-short-exist-floor 0.4 \
  --gcs-short-exist-max-visible 20 \
  --gcs-short-exist-floor-max-ape 20.0 \
  --gcs-short-exist-floor-min-iou 0.3 \
  --project runs/gcs_lane \
  --name gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_shortexist04_count03_under5_03
```

The completed result was judged only on the same 363-image official-val
surface. It failed the then-current required gate `ACC >= 0.970851`, so final
test remained closed. If this mechanism is revisited, compare first against
the latest `v2_validbranch_neg05-3` gate (`ACC >= 0.971029`) and also track the
previous `gt4pt025` gate (`ACC >= 0.970975`), run a train/val failure trace
first to verify whether `low_score_short_gt` improved enough to justify a
narrower variant, and do not tune from final test.

## Legacy count_under5=0.0 Ablation and Reporting Test

The 2026-06-22 `count03_under5_00` official test was requested after its
363-image official-val sweep completed. It is a reporting-only rejected
ablation, not a new selected candidate:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00
train args: epochs=160, batch=32, amp=true, gcs_count=0.3, gcs_count_under5=0.0, gcs_count_under5_min_lanes=5, gcs_eval_max_det=8
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00_official_val_sweep
best: conf=0.08, point_valid_thr=0.5, nms_dist_px=50.0, max_det=6, min_points=6
official-val363: ACC=0.968578, FP=0.022590, FN=0.016529, official_score=0.967796, count_acc=0.955923
```

One-shot official test command:

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.08 \
  --point-valid-thr 0.5 \
  --nms-dist-px 50.0 \
  --max-det 6 \
  --min-points 6 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00_official_test_best_from_val \
  --save-records
```

Result:

```text
summary: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00_official_test_best_from_val/tusimple_official_summary.json
images=2782
ACC=0.965118
FP=0.031908
FN=0.029745
official_score=0.963885
count_acc=0.875270
count_acc_2=0.400000
count_acc_3=0.975862
count_acc_4=0.566239
count_acc_5=0.826011
pred_lanes_hist: 2=5, 3=1837, 4=359, 5=564, 6=17
gt_lanes_hist: 2=5, 3=1740, 4=468, 5=569
count_confusion: 2->2=2, 2->3=3, 3->2=3, 3->3=1698, 3->4=34, 3->5=5, 4->3=110, 4->4=265, 4->5=89, 4->6=4, 5->3=26, 5->4=60, 5->5=470, 5->6=13
```

Do not promote this ablation. It slightly improves final-test total
`count_acc` versus `count03_under5_03`, but lowers official-val ACC and
final-test ACC while raising FP/FN. The official-val best decode uses
`max_det=6` even though training args record `gcs_eval_max_det=8`; keep that as
a comparability caveat and do not retune from test.

## Legacy Duplicate Margin 0.05 Near-Miss and Reporting Test

The 2026-06-22 `dupmargin005` official test was requested after its 363-image
official-val sweep completed. It is a reporting-only near-miss, not a promoted
selected candidate:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03
train args: epochs=160, batch=32, workers=8, amp=true, gcs_count=0.3, gcs_count_under5=0.3, gcs_duplicate_margin=0.05
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_val_sweep
best: conf=0.05, point_valid_thr=0.45, nms_dist_px=0.0, max_det=6, min_points=6
official-val363: ACC=0.970272, FP=0.024564, FN=0.016070, official_score=0.969459, count_acc=0.953168
```

One-shot official test command, using only the official-val selected decode:

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.05 \
  --point-valid-thr 0.45 \
  --nms-dist-px 0.0 \
  --max-det 6 \
  --min-points 6 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_test_best_from_val \
  --save-records
```

Result:

```text
summary: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
images=2782
ACC=0.965702
FP=0.029493
FN=0.027348
official_score=0.964565
count_acc=0.865924
count_acc_2=0.200000
count_acc_3=0.971839
count_acc_4=0.547009
count_acc_5=0.810193
pred_lanes_hist: 2=3, 3=1837, 4=371, 5=557, 6=14
gt_lanes_hist: 2=5, 3=1740, 4=468, 5=569
count_confusion: 2->2=1, 2->3=3, 2->4=1, 3->2=2, 3->3=1691, 3->4=41, 3->5=6, 4->3=118, 4->4=256, 4->5=90, 4->6=4, 5->3=25, 5->4=73, 5->5=461, 5->6=10
```

Do not promote this run from final-test ACC. It beats the older
`count03_under5_03` official-val ACC but misses the stronger `gt4short15`
official-val gate `0.970851`. Final test is reporting-only and must not be
used for threshold, checkpoint, postprocess, or loss selection.

## Legacy GT4 Lane-Balanced Point 0.25 Candidate

The 2026-06-25 `dupmargin005_gt4pt025` official-val sweep is the previous
selection surface for this branch-local GT4 candidate-recall line:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025_official_val_sweep/tusimple_official_sweep_summary.json
selected decode: conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=5, min_points=5
official-val363: ACC=0.970975, FP=0.017264, FN=0.012167, official_score=0.970386, count_acc=0.966942
count_acc_3=0.968610, count_acc_4=0.954545, count_acc_5=0.972973
```

It is selected over `gt4short15` (`ACC=0.970851`) and `dupmargin005`
(`ACC=0.970272`). The companion `dupmargin005_gt4pt050` sweep is rejected:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt050
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt050_official_val_sweep/tusimple_official_sweep_summary.json
best decode: conf=0.1, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=6
official-val363: ACC=0.964381, FP=0.034389, FN=0.021120, official_score=0.963271, count_acc=0.903581
count_acc_4=0.803030
```

Historical one-shot official test command for the selected `gt4pt025` run,
using only the frozen official-val selected decode:

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.005 \
  --point-valid-thr 0.5 \
  --nms-dist-px 18.0 \
  --max-det 5 \
  --min-points 5 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025_official_test_best_from_val \
  --save-records
```

Do not run final test for `gt4pt050`. Do not use any selected `gt4pt025`
final-test result for threshold, checkpoint, postprocess, or loss-gain tuning.
After the `v2_validbranch_neg05-3` result, the immediate next action is v2
extra-lane diagnosis and a fine official-val sweep, not a `gt4pt025` final-test
run.

## Legacy GT4 Short-Lane Recall Lane-Balanced Endpoint Rejection

The 2026-06-25 `gt4shortrecall_lbpt_endpoint` experiment used post-`b6535f641`
GT4 short-lane candidate-recall knobs that are no longer part of the active
rollback code. It is rejected and must not replace the previous `gt4pt025`
official-val candidate:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint
weights: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint/weights/best.pt
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint_official_val_sweep/tusimple_official_sweep_summary.json
best decode: conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=5, min_points=4
official-val363: ACC=0.970301, FP=0.024288, FN=0.013085, official_score=0.969554, count_acc=0.950413
count_acc_3=0.959641, count_acc_4=0.878788, count_acc_5=0.986486
count_confusion: 3->3=214, 3->4=8, 3->5=1, 4->4=58, 4->5=8, 5->4=1, 5->5=73
```

Actual recorded training args include:

```text
gcs_lane_balanced_point_loss = true
gcs_gt4_short_lane_weight = 1.5
gcs_gt4_short_lane_max_points = 20
gcs_gt4_short_match_endpoint = 1.0
gcs_gt4_short_match_max_points = 20
gcs_duplicate_margin = 0.0
gcs_count = 0.3
gcs_count_under5 = 0.3
```

Training command template:

```bash
python tools/train_gcs.py \
  --dataset tusimple \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --pretrained yolo11s-seg.pt \
  --imgsz 544 960 \
  --epochs 160 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --optimizer AdamW \
  --lr0 5e-4 \
  --lrf 0.05 \
  --cos-lr \
  --weight-decay 1e-4 \
  --warmup-epochs 3.0 \
  --warmup-bias-lr 0.0 \
  --patience 40 \
  --erasing 0.1 \
  --scale 0.3 \
  --gcs-exist 2.0 \
  --gcs-point 15.0 \
  --gcs-point-valid 1.0 \
  --gcs-smooth 0.05 \
  --gcs-curve 0.1 \
  --gcs-mask 0.2 \
  --gcs-edge 0.2 \
  --gcs-count 0.3 \
  --gcs-count-under5 0.3 \
  --gcs-count-under5-min-lanes 5 \
  --gcs-lane-count-balanced \
  --gcs-lane-count-balance-power 1.0 \
  --gcs-lane-count-min-group 50 \
  --gcs-lane-balanced-point-loss \
  --gcs-gt4-short-lane-weight 1.5 \
  --gcs-gt4-short-lane-max-points 20 \
  --gcs-gt4-short-match-endpoint 1.0 \
  --gcs-gt4-short-match-max-points 20 \
  --project runs/gcs_lane \
  --name gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint
```

Official-val sweep command:

```bash
python tools/sweep_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --half \
  --confs 0.005 0.01 0.02 0.03 0.05 0.08 0.1 0.15 0.2 0.25 \
  --point-valid-thrs 0.3 0.35 0.4 0.45 0.5 \
  --nms-dist-pxs 0 18 30 50 \
  --max-dets 5 6 8 \
  --min-points 4 5 6 \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint_official_val_sweep
```

Official-val raw-query diagnostic:

```bash
python tools/diagnose_gt4_missing_lane_raw_queries.py \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint/weights/best.pt \
  --split val \
  --archive-root archive/TUSimple \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --imgsz 544 960 \
  --conf 0.005 \
  --point-valid-thr 0.5 \
  --nms-dist-px 0.0 \
  --max-det 8 \
  --min-points 6 \
  --device 0 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4shortrecall_lbpt_endpoint_gt4_missing_raw_queries_official_val
```

Diagnostic result:

```text
selected_images = 1
total_gt4_4to3_images = 1
total_missing_gt_lanes = 1
drop_reason = geometry_bad 1
raw_match_recall = 0/1 = 0.0
after_point_valid/min_points/conf/final_decode = 0/1
missing lane = clips/0601/1494453641541664519/20.jpg, gt_lane_id=3, side=left_inner, visible_points=12
best raw query = 10, mean_abs_x_error=30.265432, overlap_points=12, score=0.28886989
```

Decision: reject this run as a promotion. The selected decode removes
official-val `4->3`, but it worsens GT4 `4->5` to `8`, raises FP, lowers ACC
below the `gt4pt025` gate, and the fixed-pool diagnostic still identifies
geometry-bad candidate recall. Do not run final test for this rejected run.

## Legacy v2_validbranch_neg05-3 Extra-Lane Diagnostic and Fine Sweep

The 2026-06-25 `v2_validbranch_neg05-3` result is valid official-val evidence
only. It exceeds the previous `gt4pt025` gate but should not go to final test
until the extra-lane diagnostic and fine official-val sweep are complete.

Known rows:

```text
strict ACC best:
  conf=0.01, point_valid_thr=0.45, nms_dist_px=30, max_det=5, min_points=2/3
  official-val ACC=0.971029

risk-reduced near tie:
  conf=0.015, point_valid_thr=0.45, nms_dist_px=60, max_det=5, min_points=2/3
  official-val ACC=0.971016, FP=0.022498, FN=0.012167
  official_score=0.970323, count_acc_4=0.878788
  count extras: 3->4=8, 3->5=1, 4->5=7
```

Do not continue this line by increasing valid recall, enabling valid count
floor, lowering `point_valid_thr` below `0.45`, or increasing `max_det` above
`5`.

Extra-lane diagnostic command template. Replace `<weights.pt>` and
`<save-prefix>` with the actual remote run paths. This command depends on the
workspace helper `tools/diagnose_tusimple_extra_lanes.py`; if preparing a clean
server checkout, sync that helper together with the algorithm payload before
running the command.

```bash
python tools/diagnose_tusimple_extra_lanes.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights <weights.pt> \
  --imgsz 544 960 \
  --device 0 \
  --half \
  --conf 0.01 \
  --point-valid-thr 0.45 \
  --nms-dist-px 30 \
  --max-det 5 \
  --min-points 2 \
  --count-pairs 3->4 3->5 4->5 5->4 \
  --save-dir <save-prefix>_extra_lane_diag_conf001_pvalid045_nms30_minp2
```

Run the same diagnostic for the risk-reduced near-tie decode:

```bash
python tools/diagnose_tusimple_extra_lanes.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights <weights.pt> \
  --imgsz 544 960 \
  --device 0 \
  --half \
  --conf 0.015 \
  --point-valid-thr 0.45 \
  --nms-dist-px 60 \
  --max-det 5 \
  --min-points 2 \
  --count-pairs 3->4 3->5 4->5 5->4 \
  --save-dir <save-prefix>_extra_lane_diag_conf0015_pvalid045_nms60_minp2
```

Required diagnostic artifacts:

```text
image lists and visualizations for GT3->4, GT3->5, GT4->5, GT5->4
extra_lanes.csv with:
  score
  valid point count
  nearest GT overlap and mean_abs_x_error
  nearest pred overlap and mean_abs_x_error
  possible_duplicate / possible_spurious
```

Fine official-val decoder sweep command:

```bash
python tools/sweep_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights <weights.pt> \
  --imgsz 544 960 \
  --device 0 \
  --half \
  --confs 0.008 0.010 0.012 0.015 0.020 \
  --point-valid-thrs 0.43 0.45 0.47 0.50 \
  --nms-dist-pxs 30 36 42 50 60 \
  --max-dets 5 \
  --min-points 2 3 \
  --save-dir <save-prefix>_official_val_fine_decoder_sweep
```

Report these columns from the sweep and confusion output:

```text
official_acc
official_score
FP
FN
count_acc_4
3->4
3->5
4->5
```

Selection rule: official ACC remains primary. If rows are within about
`1e-5` to `2e-5`, prefer the row with higher official score, lower FP, better
`count_acc_4`, fewer `3->5` / `4->5`, and no new `GT5->4` regression. Do not
use final test for any part of this choice.

## Legacy GT3 Extra-Survival 0.03 Rejection and Reporting Test

The 2026-06-24 `dupmargin005_gt3extra003` run enabled the default-disabled GT3
extra-survival loss on top of `dupmargin005`:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03
train args: epochs=160, batch=32, workers=8, amp=true, gcs_count=0.3, gcs_count_under5=0.3, gcs_duplicate_margin=0.05, gcs_gt3_extra_survival=0.03, gcs_gt3_extra_margin_logit=0.05, gcs_gt3_extra_topk=1
training log: results.csv includes non-zero train/gt3_extra_survival_loss and val/gt3_extra_survival_loss
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_official_val_sweep
best/tied selected decode: conf=0.05, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
official-val363: ACC=0.969665, FP=0.020615, FN=0.014004, official_score=0.968973, count_acc=0.961433
count_acc_3=0.968610, count_acc_4=0.909091, count_acc_5=0.986486
count_confusion: 3->3=216, 3->4=7, 4->3=2, 4->4=60, 4->5=4, 5->4=1, 5->5=73
```

Official-val sweep command:

```bash
python tools/sweep_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --half \
  --confs 0.005 0.01 0.02 0.03 0.05 0.08 0.1 0.15 0.2 0.25 \
  --point-valid-thrs 0.3 0.35 0.4 0.45 0.5 \
  --nms-dist-pxs 0 18 30 50 \
  --max-dets 5 6 8 \
  --min-points 4 5 6 \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_official_val_sweep
```

The sweep summary's first `best` row is `conf=0.005`, but the selected
`conf=0.05` row has the same official ACC, FP, FN, official score, count
accuracy, and count confusion. The selected `conf=0.05` decode is therefore
official-val evidence, not final-test tuning.

Failure-mode compare:

```bash
python tools/compare_tusimple_failure_modes.py \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --pred count03=runs/gcs_lane/failure_compare/dupmargin005_compare/count03_val/tusimple_predictions.json \
  --pred gt4short15=runs/gcs_lane/failure_compare/dupmargin005_compare/gt4short15_val/tusimple_predictions.json \
  --pred dupmargin005=runs/gcs_lane/failure_compare/dupmargin005_compare/dupmargin005_val/tusimple_predictions.json \
  --pred gt3extra003=runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_official_val_selected_decode/tusimple_predictions.json \
  --out-dir runs/gcs_lane/failure_compare/dupmargin005_gt3extra003_compare \
  --baseline-run count03 \
  --dupmargin-run gt3extra003 \
  --gt4short-run gt4short15
```

Key failure-mode result:

```text
summary: runs/gcs_lane/failure_compare/dupmargin005_gt3extra003_compare/summary.json
count-failure-only:
  count03      failure_images=11, 3->4=4,  4->5=5, 5->4=1, duplicate_like_extra=0, spurious_extra=15, missed_short_gt=5
  dupmargin005 failure_images=17, 3->4=11, 4->5=3, 5->4=2, duplicate_like_extra=0, spurious_extra=18, missed_short_gt=5
  gt3extra003  failure_images=14, 3->4=7,  4->5=4, 5->4=1, duplicate_like_extra=0, spurious_extra=16, missed_short_gt=5
all-image missed_short_gt: count03=33, dupmargin005=28, gt3extra003=36
```

One-shot official test command, using only the official-val selected decode:

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.05 \
  --point-valid-thr 0.5 \
  --nms-dist-px 0.0 \
  --max-det 8 \
  --min-points 6 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_official_val_selected_decode_test \
  --save-records
```

Result:

```text
summary: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt3extra003_count03_under5_03_official_val_selected_decode_test/tusimple_official_summary.json
images=2782
ACC=0.965077
FP=0.030494
FN=0.027528
official_score=0.963917
count_acc=0.875988
count_acc_2=0.200000
count_acc_3=0.976437
count_acc_4=0.538462
count_acc_5=0.852373
count_confusion includes 3->4=34, 4->5=93, 5->4=46
```

Do not promote this run. It reduces `dupmargin005` GT3 `3->4` from `11` to
`7` and recovers GT5 `5->4` from `2` to `1`, but it misses the official-val
ACC of `count03_under5_03`, `dupmargin005`, and `gt4short15`; it also hurts
GT4 relative to `dupmargin005` (`count_acc_4=0.939394 -> 0.909091`,
`4->5=3 -> 4`) and raises all-image `missed_short_gt` to `36`. The official-test
result is reporting-only and must not be used for tuning.

## Legacy Spurious Margin 0.03 Rejection and Reporting Test

The 2026-06-23 `spurmargin003` official test was requested after its 363-image
official-val sweep completed. It is a reporting-only rejected follow-up, not a
promoted selected candidate:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03
train args: epochs=160, batch=32, workers=4, no_amp=true, gcs_count=0.3, gcs_count_under5=0.3, gcs_spurious_margin=0.03
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03_official_val_sweep
best: conf=0.02, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=5
official-val363: ACC=0.969316, FP=0.023095, FN=0.014922, official_score=0.968556, count_acc=0.958678
count_acc_3=0.973094, count_acc_4=0.878788, count_acc_5=0.986486
```

One-shot official test command, using only the official-val selected decode:

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.02 \
  --point-valid-thr 0.5 \
  --nms-dist-px 18.0 \
  --max-det 6 \
  --min-points 5 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03_official_test_best_from_val \
  --save-records
```

Result:

```text
summary: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
images=2782
ACC=0.964522
FP=0.033591
FN=0.028636
official_score=0.963277
count_acc=0.869878
count_acc_2=0.200000
count_acc_3=0.964368
count_acc_4=0.527778
count_acc_5=0.868190
pred_lanes_hist: 2=4, 3=1820, 4=346, 5=598, 6=14
gt_lanes_hist: 2=5, 3=1740, 4=468, 5=569
count_confusion: 2->2=1, 2->3=4, 3->2=3, 3->3=1678, 3->4=53, 3->5=5, 3->6=1, 4->3=118, 4->4=247, 4->5=99, 4->6=4, 5->3=20, 5->4=46, 5->5=494, 5->6=9
```

Do not promote this run. It misses the official-val ACC of
`count03_under5_03` (`0.969976`), `dupmargin005` (`0.970272`), and
`gt4short15` (`0.970851`). Its reporting-only test ACC also trails those three
comparators, so there is no protocol-valid promotion path. The official-test
decode uses `max_det=6` while the train args record `gcs_eval_max_det=8`; this
comes from official-val selection and is a comparability caveat, not a reason
to retune on test.

The previous 2026-06-20 final-test-reported candidate was:

```text
source run:   runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03
weights:      weights/best.pt
val sweep:    runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_official_val363_sweep
final test:   runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_official_test_best_from_val
decode:       conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
official-val: ACC=0.969976, FP=0.019559, FN=0.014463, count_acc=0.969697
final test:   ACC=0.965459, FP=0.029439, FN=0.026270, count_acc=0.872753
```

Use this test result only as final reporting evidence for the previous `count03_under5_03` candidate. Do not use it to tune thresholds, checkpoint choice, or postprocess settings.

## Legacy Train/Val Count-Confusion Diagnostic

The 2026-06-20 train/val diagnostic for `count03_under5_03` groups decoded lane-count confusion by date, GT lane count, and the shortest visible GT lane bucket. It is a legacy diagnostic; `tools/diagnose_tusimple_count_confusion.py` is not present in the active `424ab1c86` rollback code. It used the frozen official-val selected decode and did not touch final test:

```text
output: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_count_confusion_train_val_by_visibility/summary.json
decode: conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
splits: fixed-y train and val
```

Run the reusable direct fixed-y diagnostic from the remote clone:

```bash
python tools/diagnose_tusimple_count_confusion.py \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03/weights/best.pt \
  --dataset-root datasets/tusimple_fixed_y_k56_960x544 \
  --splits train val \
  --imgsz 544 960 \
  --conf 0.05 \
  --point-valid-thr 0.5 \
  --nms-dist-px 50.0 \
  --max-det 8 \
  --min-points 5 \
  --device 0 \
  --half \
  --save-json runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_count_confusion_train_val_by_visibility/summary.json \
  --save-csv runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_count_confusion_train_val_by_visibility/groups.csv \
  --topk 20
```

The helper also supports a JSON-only mode for already-generated train/val TuSimple-format predictions:

```bash
python tools/diagnose_tusimple_count_confusion.py \
  --split val \
  --gt-json <train-or-val-gt-json> \
  --pred-json <tusimple-format-predictions-json> \
  --save-json <diagnostic-summary-json> \
  --save-csv <diagnostic-groups-csv> \
  --topk 20
```

This diagnostic is for train/val bottleneck localization only. Do not use final test for count-policy, threshold, checkpoint, or postprocess selection.

## Legacy Train/Val Query-Trace Diagnostic

Use the self-contained query-trace helper when comparing matched q+ and
unmatched q- behavior on fixed-y train/val. It does not depend on legacy
`tools/diagnose_tusimple_count_confusion.py`.

For `count03_under5_03`:

```bash
python tools/diagnose_gt4_short_failure_queries.py \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03/weights/best.pt \
  --dataset-root datasets/tusimple_fixed_y_k56_960x544 \
  --splits train val \
  --imgsz 544 960 \
  --conf 0.05 \
  --point-valid-thr 0.5 \
  --nms-dist-px 50.0 \
  --max-det 8 \
  --min-points 5 \
  --gt-counts 4 5 \
  --short-visible-max 0 \
  --trace-scope candidates \
  --device 0 \
  --half \
  --save-dir runs/gcs_lane/query_trace_count03_under5_03_gt4_gt5_train_val
```

For `dupmargin005`:

```bash
python tools/diagnose_gt4_short_failure_queries.py \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03/weights/best.pt \
  --dataset-root datasets/tusimple_fixed_y_k56_960x544 \
  --splits train val \
  --imgsz 544 960 \
  --conf 0.05 \
  --point-valid-thr 0.45 \
  --nms-dist-px 0.0 \
  --max-det 6 \
  --min-points 6 \
  --gt-counts 4 5 \
  --short-visible-max 0 \
  --trace-scope candidates \
  --device 0 \
  --half \
  --save-dir runs/gcs_lane/query_trace_dupmargin005_gt4_gt5_train_val
```

For `gt4short15`:

```bash
python tools/diagnose_gt4_short_failure_queries.py \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03/weights/best.pt \
  --dataset-root datasets/tusimple_fixed_y_k56_960x544 \
  --splits train val \
  --imgsz 544 960 \
  --conf 0.15 \
  --point-valid-thr 0.5 \
  --nms-dist-px 0.0 \
  --max-det 6 \
  --min-points 4 \
  --gt-counts 4 5 \
  --short-visible-max 0 \
  --trace-scope candidates \
  --device 0 \
  --half \
  --save-dir runs/gcs_lane/query_trace_gt4short15_gt4_gt5_train_val
```

For `farspur001_gt5rank001`:

```bash
python tools/diagnose_gt4_short_failure_queries.py \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_farspur001_gt5rank001_count03_under5_03/weights/best.pt \
  --dataset-root datasets/tusimple_fixed_y_k56_960x544 \
  --splits train val \
  --imgsz 544 960 \
  --conf 0.005 \
  --point-valid-thr 0.5 \
  --nms-dist-px 18.0 \
  --max-det 8 \
  --min-points 4 \
  --gt-counts 4 5 \
  --short-visible-max 0 \
  --trace-scope candidates \
  --device 0 \
  --half \
  --save-dir runs/gcs_lane/query_trace_farspur001_gt5rank001_gt4_gt5_train_val
```

Use `--short-visible-max 20 --trace-scope failures` for the narrower legacy
short-lane failure trace. The main outputs are `summary.json`,
`role_summary.csv`, `images.csv`, and `query_trace.csv`.

## Legacy GT4 Short-Lane Weighted Training

The GT4 short-lane sampler boost was an explicit post-`b6535f641`
experimental option. It is not present in the active `424ab1c86` rollback code. Historical
defaults were:

```text
gcs_gt4_short_boost = 1.0
gcs_gt4_short_min_visible_max = 10
```

Completed diagnostic experiment, not a promoted next step:

```bash
python tools/train_gcs.py \
  --dataset tusimple \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --pretrained yolo11s-seg.pt \
  --imgsz 544 960 \
  --epochs 160 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --no-amp \
  --optimizer AdamW \
  --lr0 5e-4 \
  --lrf 0.05 \
  --cos-lr \
  --weight-decay 1e-4 \
  --warmup-epochs 3.0 \
  --warmup-bias-lr 0.0 \
  --patience 40 \
  --erasing 0.1 \
  --scale 0.3 \
  --gcs-exist 2.0 \
  --gcs-point 15.0 \
  --gcs-point-valid 1.0 \
  --gcs-smooth 0.05 \
  --gcs-curve 0.1 \
  --gcs-mask 0.2 \
  --gcs-edge 0.2 \
  --gcs-count 0.3 \
  --gcs-count-under5 0.3 \
  --gcs-count-under5-min-lanes 5 \
  --gcs-lane-count-balanced \
  --gcs-lane-count-balance-power 1.0 \
  --gcs-lane-count-min-group 50 \
  --gcs-gt4-short-boost 2.0 \
  --gcs-gt4-short-min-visible-max 10 \
  --project runs/gcs_lane \
  --name gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03
```

After training, select checkpoint/decode only on official-val. Then run the train/val count-confusion diagnostic above on the selected candidate before any final-test evaluation.

Result note:

```text
gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03 improved the
target train/val short-GT4 diagnostic groups, but did not beat the baseline on
363-image official-val ACC:

A baseline official_acc = 0.969976
B gt4short2 official_acc = 0.969726
```

Do not send this candidate to final test. The similarly named
`gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03_official_val_sweep_maxdet8`
artifact used `archive/TUSimple/train_set/label_data_0313.json` with
`images=2858`, so it is not the 363-image official-val selection surface.

Follow-up result:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03
sampler: --gcs-gt4-short-boost 1.5 --gcs-gt4-short-min-visible-max 10
val sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val_sweep
best: conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
official-val ACC=0.970851, FP=0.022084, FN=0.011708, official_score=0.970175, count_acc=0.939394
```

This became the official-val selected candidate in the legacy experiment line, but its count accuracy is worse
than both baseline and `gt4short2`. The train/val diagnostic below has been
run and should be used as the reproducible count-bottleneck check.

```bash
python tools/diagnose_tusimple_count_confusion.py \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03/weights/best.pt \
  --dataset-root datasets/tusimple_fixed_y_k56_960x544 \
  --splits train val \
  --imgsz 544 960 \
  --conf 0.15 \
  --point-valid-thr 0.5 \
  --nms-dist-px 0.0 \
  --max-det 6 \
  --min-points 4 \
  --device 0 \
  --half \
  --save-json runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_count_confusion_train_val_by_visibility/summary.json \
  --save-csv runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_count_confusion_train_val_by_visibility/groups.csv \
  --topk 20
```

The one-shot final-test report used the same official-val selected decode:

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.15 \
  --point-valid-thr 0.5 \
  --nms-dist-px 0.0 \
  --max-det 6 \
  --min-points 4 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_test_best_from_val \
  --save-records
```

Result:

```text
official_acc = 0.965369
official_FP = 0.033309
official_FN = 0.029236
official_score = 0.964118
count_acc = 0.864486
```

This is reporting evidence only. It is lower than the previous `count03_under5_03`
final-test ACC `0.965459`, so future changes must return to official-val and
train/val diagnostics rather than test threshold search.

## Label Rebuild

Regenerate K56 labels from original TuSimple JSON and images. Do not resample historical K32 labels.

For this 5-25-3 branch, `.npz` labels must include `semantic_mask` and `edge_mask` because the branch trains `mask_loss` and `edge_loss`. If the remote clone has `datasets` as a symlink to a shared K56 dataset that lacks those arrays, replace only this clone's symlink with a real local runtime directory before rebuilding:

```bash
cd /root/GCS-YOLO-Lane_LSA_5-25-3-k56
rm -f datasets
mkdir -p datasets
```

```bash
python tools/convert_tusimple_to_gcs.py \
  --archive-root archive/TUSimple \
  --output-root datasets/tusimple_fixed_y_k56_960x544 \
  --imgsz 544 960 \
  --point-mode fixed_y \
  --num-points 56 \
  --fixed-y-start 0.9861111111111112 \
  --fixed-y-end 0.2222222222222222
```

Quick label-field check:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
p = next(Path("datasets/tusimple_fixed_y_k56_960x544/labels_gcs/train").glob("*.npz"))
with np.load(p) as data:
    print(p.name, sorted(data.files))
    assert {"semantic_mask", "edge_mask", "lanes", "lane_valid"}.issubset(data.files)
PY
```

## Model Shape Check

```bash
python tools/check_model.py \
  --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --imgsz 544 960 \
  --batch 1
```

Expected key shapes:

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
aux_mask_logits: B x 2 x 544 x 960
aux_edge_logits: B x 1 x 544 x 960
```

## Dataset Checks

```bash
python tools/check_dataset.py
python tools/check_gcs_label_order_split.py --dataset-root datasets/tusimple_fixed_y_k56_960x544
```

## Custom GCS Evaluation

```bash
python tools/eval_gcs.py \
  --weights <weights.pt> \
  --source datasets/tusimple_fixed_y_k56_960x544/images/val \
  --labels datasets/tusimple_fixed_y_k56_960x544/labels_gcs/val \
  --imgsz 544 960
```

## TuSimple Official-Val Sweep

Use `val` for threshold and postprocess selection. The comparable official-val
surface is the canonical 363-image GT JSON; do not use `train_val.json` or
`label_data_0313.json` as replacement validation GT. Use the cached sweep
helper for current threshold scans:

```bash
python tools/sweep_tusimple_official_cached.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights <weights.pt> \
  --imgsz 544 960 \
  --device 0
```

If a noncanonical validation GT is intentionally used, pass
`--allow-noncanonical-gt` and treat the summary as
`comparable_to_e1_spurious=false`.

Default-off count-aware top-k postprocess ablation uses the same official-val
surface and must be selected on validation only:

```bash
python tools/sweep_tusimple_official_cached.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary_shortpv_spurious_v1/weights/best.pt \
  --imgsz 544 960 \
  --confs 0.005 0.01 0.02 0.03 0.05 0.08 0.10 0.15 0.20 0.25 \
  --point-valid-thrs 0.30 0.35 0.40 0.45 0.50 \
  --nms-dist-pxs 0 18 30 50 \
  --max-dets 5 6 8 \
  --min-points 5 6 8 \
  --device 0 \
  --half \
  --count-aware-topk \
  --count-aware-min-k 3 \
  --count-aware-max-k 5 \
  --count-aware-length-norm 12 \
  --count-aware-extra-margins 0 \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary_shortpv_spurious_v1_official_val363_sweep_countaware
```

Compare this only against the matching normal official-val sweep for the same
E3 `best.pt`, focusing on `official_acc`, `count_acc_4`, `count_acc_5`,
`official_FP`, and `official_FN`. The sweep summary records
`count_aware_topk` and `count_aware_extra_margins`.

For the 2026-07-11 G1 GT5-short-geometry follow-up, the validation-only
count-aware extra-margin proxy sweep used the fixed G1 `official_best.pt`
decode row and swept only `score_sum` margins:

```bash
python tools/sweep_tusimple_official_cached.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights runs/gcs_lane/query_alpha05_gt5short_geom_w2_v1/weights/official_best.pt \
  --imgsz 544 960 \
  --device 0 \
  --half \
  --confs 0.003 \
  --point-valid-thrs 0.5 \
  --nms-dist-pxs 0 \
  --max-dets 6 \
  --min-points 5 \
  --valid-before-maxdet \
  --count-aware-topk \
  --count-modes score_sum \
  --count-aware-extra-margins 0 1 2 \
  --save-dir runs/gcs_lane/query_alpha05_gt5short_geom_w2_v1_official_best_val_score_sum_catopk_extra_margin
```

This G1 checkpoint does not emit `pred_count_logits`, so
`--count-modes count_logits` is invalid for it. True `count_logits k/k+1/k+2`
requires a query-count checkpoint built from
`ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count.yaml`.

Default-off valid-before-maxdet query decode uses the same official-val
surface and must be selected on validation only:

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --half \
  --conf 0.005 \
  --point-valid-thr 0.45 \
  --nms-dist-px 0.0 \
  --max-det 5 \
  --min-points 2 \
  --valid-before-maxdet \
  --save-records
```

Compare against the matching default decode on official-val only, focusing on
`5->4`, `official_FP`, `4->5`, and `official_acc`. The summary/config records
`valid_before_maxdet`.

The 2026-07-02 user-requested reporting-only official test for this
official-val selected decode used a temporary legacy-query compatibility
wrapper because the old pickled `best.pt` lacks the newer
`GCSLaneHead.gcs_mode` attribute. The wrapper only patches missing
`gcs_mode="query"` after model loading; it does not change weights, decode,
labels, or official metrics. If source compatibility is fixed, the same args
can be run through `tools/eval_tusimple_official.py` directly.

```bash
python .tmp/eval_official_legacy_query.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.005 \
  --point-valid-thr 0.45 \
  --nms-dist-px 0.0 \
  --max-det 5 \
  --min-points 2 \
  --valid-before-maxdet \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1_official_test_best_from_val_valid_before_maxdet \
  --save-records
```

Result:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1_official_test_best_from_val_valid_before_maxdet/tusimple_official_summary.json
official_acc = 0.965483
official_score = 0.964313
official_FP = 0.030847
official_FN = 0.027678
count_acc = 0.882818
count_acc_3 = 0.971839
count_acc_4 = 0.606838
count_acc_5 = 0.843585
```

The matched E1 count-boundary `best.pt` reporting-only official test used the
same wrapper and the E1 official-val selected decode:

```bash
python .tmp/eval_official_legacy_query.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.005 \
  --point-valid-thr 0.5 \
  --nms-dist-px 0.0 \
  --max-det 6 \
  --min-points 2 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1_official_test_best_from_val \
  --save-records
```

Result:

```text
summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1_official_test_best_from_val/tusimple_official_summary.json
official_acc = 0.965428
official_score = 0.964279
official_FP = 0.031368
official_FN = 0.026060
count_acc = 0.874551
count_acc_3 = 0.971839
count_acc_4 = 0.568376
count_acc_5 = 0.834798
```

Do not tune thresholds, `valid_before_maxdet`, `max_det`, NMS, checkpoint, or
loss weights from either test report.

The later short-side-geometry follow-up used an official-val-selected decode
from the external sweep below and is also reporting-only:

```text
val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1_official_best_val_sweep_valid_before_maxdet/tusimple_official_sweep_summary.json
test_summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1_official_best_test_best_from_val/tusimple_official_summary.json
decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=4, valid_before_maxdet=true
official_val_acc = 0.977032
official_test_acc = 0.963348
official_test_FP = 0.032171
official_test_FN = 0.031722
test_count_acc_4 = 0.617521
test_count_acc_5 = 0.801406
```

This result should not be promoted despite the higher 363-image official-val
ACC. Compared with the no-shortside same-line run, reporting-only test ACC
drops by `0.001010` and GT5 undercount worsens (`5->4: 69 -> 85`). Keep test
closed for any further tuning.

The later far-spurious follow-up also used an official-val-selected decode and
is reporting-only:

```text
val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_farspur005_v1_official_best_val_sweep_valid_before_maxdet/tusimple_official_sweep_summary.json
test_summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_farspur005_v1_official_best_test_best_from_val/tusimple_official_summary.json
decode = conf=0.003, point_valid_thr=0.575, nms_dist_px=50.0, max_det=5, min_points=2, valid_before_maxdet=true
official_val_acc = 0.977758
official_val_FP = 0.008219
official_val_FN = 0.008953
official_val_count_acc_4 = 0.954545
official_val_count_acc_5 = 0.972973
official_val_count_confusion includes 4->5=0, 5->4=2
official_test_acc = 0.963907
official_test_FP = 0.031860
official_test_FN = 0.031332
test_count_acc_4 = 0.628205
test_count_acc_5 = 0.815466
test_count_confusion includes 4->5=65, 5->4=77, 5->5=464
```

Do not promote this run. Compared with the no-shortside same-line run, it
improves reporting-only test GT4 false-fifth behavior (`4->5: 75 -> 65`) but
regresses test ACC (`0.964358 -> 0.963907`), FP/FN, and GT5 retention
(`5->4: 69 -> 77`, `count_acc_5: 0.836555 -> 0.815466`). Keep test closed for
any further threshold, NMS, checkpoint, decode, or far-spur loss tuning.

The later shortside boost-only fixed-threshold follow-up used
`gcs_shortside_rawmatch_boost=0.25` and
`gcs_shortside_exist_target_floor=0.0`; it is also reporting-only:

```text
val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_boost_only_fixedthr_v1_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
test_summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_boost_only_fixedthr_v1_test_best_from_val_b/tusimple_official_summary.json
decode = conf=0.003, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_val_acc = 0.974445
official_val_FP = 0.007576
official_val_FN = 0.009642
official_val_count_acc_4 = 0.954545
official_val_count_acc_5 = 0.959459
official_test_acc = 0.964144
official_test_FP = 0.028666
official_test_FN = 0.028696
test_count_acc_4 = 0.630342
test_count_acc_5 = 0.757469
test_count_confusion includes 4->3=106, 4->5=67, 5->3=27, 5->4=111, 5->5=431
```

Do not promote this run. Compared with spurious-lite + valid-before, it
improves reporting-only test GT4 false-fifth behavior (`4->5: 79 -> 67`) and
FP (`0.030847 -> 0.028666`), but it misses the ACC target
(`0.964144 < 0.9655`) and severely regresses GT5 retention
(`5->4: 63 -> 111`, `count_acc_5: 0.843585 -> 0.757469`). Do not reselect
`point_valid_thr`, NMS, checkpoint, decode, or loss weights from this test.

The later shortside protect floor follow-up used
`gcs_shortside_rawmatch_boost=0.25` and
`gcs_shortside_exist_target_floor=0.7`; it is also reporting-only. Because the
run used `--gcs-official-best`, use the `weights/official_best.pt` sweep/test
as the primary comparable result; the companion `weights/best.pt` sweep/test is
not the main candidate.

```text
run = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2
val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2_official_best_val_sweep_valid_before_maxdet_b/tusimple_official_sweep_summary.json
test_summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2_test_best_from_val_b/tusimple_official_summary.json
weights = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_ablate_shortside_protect_floor07_v1-2/weights/official_best.pt
decode = conf=0.008, point_valid_thr=0.6, nms_dist_px=0.0, max_det=5, min_points=2, valid_before_maxdet=true
official_val_acc = 0.972798
official_val_FP = 0.011433
official_val_FN = 0.012626
official_val_count_acc_4 = 0.939394
official_val_count_acc_5 = 0.959459
official_test_acc = 0.964281
official_test_FP = 0.029014
official_test_FN = 0.029385
test_count_acc_4 = 0.621795
test_count_acc_5 = 0.766257
test_count_confusion includes 4->3=112, 4->5=65, 5->3=28, 5->4=105, 5->5=436
```

Do not promote this run. Compared with spurious-lite + valid-before, it
improves reporting-only test GT4 false-fifth behavior (`4->5: 79 -> 65`) but
misses ACC (`0.964281 < 0.9655`), worsens FN (`0.027678 -> 0.029385`), and
still severely regresses GT5 retention (`5->4: 63 -> 105`,
`count_acc_5: 0.843585 -> 0.766257`). Compared with boost-only, official-val
ACC/FP/FN/`count_acc_4` are worse, and the reporting-only test ACC gain is too
small to offset worse FP/FN and GT4 `4->3`. Do not reselect `point_valid_thr`,
NMS, checkpoint, decode, target floor, or loss weights from this test.

Legacy Q18/count-head guided sweeps used the same
official-val surface and kept normal/count-guided rows in one sweep table.
These flags are not available in the active `424ab1c86` rollback code:

```bash
python tools/sweep_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split val \
  --weights <weights.pt> \
  --imgsz 544 960 \
  --device 0 \
  --count-guided-topk \
  --count-guided-min-probs 0.60 0.65 \
  --count-guided-allowed-counts 3,4,5
```

Do not pass `--count-guided-allow-unsupported-fallback` for candidate
selection. It is only for diagnostics when intentionally checking fallback
behavior with checkpoints that lack count-head logits.

## Legacy Q20 GT4-Hard Diagnostic Gate

Before any Q20 official-val sweep, run the GT4-hard raw-query diagnostic on the
fixed `gt4pt025` hard set. The 2026-06-26 Q20 run failed this gate, so its
official sweep was intentionally not run:

```text
run = gcs_yolo_lane_s_q20_k56_sidegeom_gt4endpoint_validneg_countce_v1
fixed old-missing raw_match_recall = 12/22 = 0.545455
current-missing raw_match_recall = 10/20 = 0.500000
current geometry_bad = 10
current after_point_valid_recall = 0/20
```

The diagnostic decode is:

```text
conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
match gate = overlap >= 3 h-samples and mean_abs_x_error <= 20px
```

Current-missing command:

```bash
python tools/diagnose_gt4_missing_lane_raw_queries.py \
  --weights runs/gcs_lane/gcs_yolo_lane_s_q20_k56_sidegeom_gt4endpoint_validneg_countce_v1/weights/best.pt \
  --split train \
  --gt-json runs/gcs_lane/gcs_yolo_lane_s_q18_k56_side_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/gt4_hard_gt4pt025_19_records.jsonl \
  --imgsz 544 960 \
  --conf 0.005 \
  --point-valid-thr 0.5 \
  --nms-dist-px 0.0 \
  --max-det 8 \
  --min-points 6 \
  --only-count-pair all \
  --device 0 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_q20_k56_sidegeom_gt4endpoint_validneg_countce_v1/gt4_hard_raw_query_fixed_gt4pt025/diagnostic_all_current_missing
```

For Q20-dataref/reference-clustering follow-ups, do not run official-val sweep
unless the hard diagnostic first reaches the stricter dataref gate:

```text
fixed old-missing:
  raw_match_recall >= 14/22
  after_point_valid_recall >= 4/22
  final_decode_recall > 2/22
  geometry_bad <= 7

current missing:
  raw_match_recall >= 0.55
  geometry_bad <= 7
  after_point_valid_recall > 0
```

The 2026-06-26 Q20-dataref v1 attempt failed this gate and should not receive
an official-val sweep:

```text
run = gcs_yolo_lane_s_q20_k56_dataref_gt4endpoint_validneg_countce_v1
checkpoint audit = no point_reference_logits keys in sidegeom/dataref best.pt state_dict

fixed old-missing:
raw_match_recall = 12/22 = 0.545455
after_point_valid_recall = 3/22 = 0.136364
final_decode_recall = 2/22 = 0.090909
geometry_bad = 10

current missing:
raw_match_recall = 10/20 = 0.500000
geometry_bad = 10
after_point_valid_recall = 1/20 = 0.050000
final_decode_recall = 0/20 = 0.000000
```

The recorded `reset_point_reference=false` is not considered the cause of this
failure: `point_reference_logits` is a non-persistent buffer and was absent from
both audited checkpoints. The failure is the hard-gate result itself.

Only consider valid-loss follow-up when raw geometry clearly improves and
`geometry_bad` clearly drops, but point-valid survival remains low. Otherwise,
keep valid weights unchanged and move to data-driven reference clustering.

## TuSimple Final Test

Run test only once for a candidate already selected on official-val:

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights <weights.pt> \
  --imgsz 544 960 \
  --device 0 \
  --conf <selected-conf> \
  --point-valid-thr <selected-point-valid-thr> \
  --nms-dist-px <selected-nms-dist-px> \
  --max-det <selected-max-det>
```

## Inference

```bash
python tools/infer_gcs.py \
  --weights <weights.pt> \
  --source <images-or-list> \
  --imgsz 544 960
```

## Python Compile Check

For changed Python files:

```bash
python -m py_compile <changed-python-files>
```

## Known Validation Limitation

This branch was imported from `5-25-3.zip` and includes standalone TuSimple official eval/sweep helpers plus the explicit `--gcs-official-best` selection hook. Server-side sync only needs algorithm/runtime code. Mainline Count/Quality/Survival tests and agent setup checks are intentionally not part of the server payload unless a future task explicitly restores them.

## Official TuSimple Evaluation Helpers

Official-val threshold sweeps are allowed only on validation data:

```bash
python tools/sweep_tusimple_official_cached.py \
  --weights <weights.pt> \
  --split val \
  --archive-root archive/TUSimple \
  --imgsz 544 960
```

Final test evaluation is one-shot only after official-val selection:

```bash
python tools/eval_tusimple_official.py \
  --weights <selected-weights.pt> \
  --split test \
  --archive-root archive/TUSimple \
  --imgsz 544 960
```
