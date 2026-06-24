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

The 5-25-3 branch does not include later mainline `--gcs-official-best`, Count/Quality/Survival, near-miss, or training-time official-best machinery. It does include branch-local TuSimple official eval/sweep helpers: `tools/eval_tusimple_official.py` and `tools/sweep_tusimple_official.py`.

Active source/config is based on rollback commit `50999d6af` (`Document 5-25-3
K56 as mainline`) plus the default-disabled `duplicate_margin_loss`,
`spurious_margin_loss`, `lane_balanced_point_loss`,
`short_valid_recall_loss`, `far_spurious_survival_loss`, and
`gt5_rank_consistency_loss`, `gt3_extra_survival_loss`, and
`gt4_lane_balanced_point_loss` experiment knobs, plus train-only
`gcs_gt4_sample_gain`. Sections below that mention `gt4short*`,
`--gcs-gt4-short-*`, `--gcs-extra-exist`, `--gcs-short-exist-*`, or
`tools/diagnose_tusimple_count_confusion.py` are legacy post-`50999d6af`
experiment records only. They are not commands for the current code state
unless a future task explicitly restores those commits.

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
  --project runs/gcs_lane \
  --name gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count02_under5_03
```

If `batch=32` OOMs on the target machine, reduce batch only for OOM/instability and record the change in the run notes.

`--no-amp` is included because the current remote run hit an Ultralytics AMP self-check failure while loading `yolo26n.pt`. If that server cache/checkpoint issue is fixed, AMP may be re-enabled only with a run note.

## Positive Short-Lane Point/Visibility Experiment

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

For future candidates after the 2026-06-25 GT4 lane-balanced point sweep, the
current official-val ACC gate is `0.970975` from `gt4pt025`.

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

## Legacy Post-50999 Official-Val Selection

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
test remained closed. If this mechanism is revisited, compare against the
current `gt4pt025` gate (`ACC >= 0.970975`), run a train/val failure trace first
to verify whether `low_score_short_gt` improved enough to justify a narrower
variant, and do not tune from final test.

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

## Duplicate Margin 0.05 Near-Miss and Reporting Test

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

## GT4 Lane-Balanced Point 0.25 Selected Candidate

The 2026-06-25 `dupmargin005_gt4pt025` official-val sweep is the current
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

One-shot official test command for the selected `gt4pt025` run, using only the
frozen official-val selected decode:

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

Do not run final test for `gt4pt050`. Do not use the selected `gt4pt025`
final-test result for threshold, checkpoint, postprocess, or loss-gain tuning.

## GT3 Extra-Survival 0.03 Rejection and Reporting Test

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

## Spurious Margin 0.03 Rejection and Reporting Test

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

The 2026-06-20 train/val diagnostic for `count03_under5_03` groups decoded lane-count confusion by date, GT lane count, and the shortest visible GT lane bucket. It is a legacy post-`50999d6af` diagnostic; `tools/diagnose_tusimple_count_confusion.py` is not present in the active rollback code. It used the frozen official-val selected decode and did not touch final test:

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

## Train/Val Query-Trace Diagnostic

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

The GT4 short-lane sampler boost was an explicit post-`50999d6af`
experimental option. It is not present in the active rollback code. Historical
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

Use `val` for threshold and postprocess selection:

```bash
python tools/sweep_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split val \
  --weights <weights.pt> \
  --imgsz 544 960 \
  --device 0
```

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

This branch was imported from `5-25-3.zip` and includes standalone TuSimple official eval/sweep helpers. Server-side sync only needs algorithm/runtime code. Mainline Count/Quality/Survival tests, agent setup checks, and training-time official-best checkpoint preservation are intentionally not part of the server payload unless a future task explicitly restores them.

## Official TuSimple Evaluation Helpers

Official-val threshold sweeps are allowed only on validation data:

```bash
python tools/sweep_tusimple_official.py \
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
