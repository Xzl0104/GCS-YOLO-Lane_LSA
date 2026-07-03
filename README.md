# GCS-YOLO-Lane 5-25-3 K56 Mainline

This is the current GCS-YOLO-Lane mainline branch. It imports the historical `5-25-3.zip` algorithm and adapts only the TuSimple fixed-y contract.

Active source/config is rolled back to commit `b6535f641` (`Fix GCS training progress header alignment`). Its active algorithm contract remains the 5-25-3 K56 mainline with no later Count Head, Q18/Q20/dataref, lane-balanced, valid-repair, side-aux, or GT4-hard diagnostic mechanisms active. The explicit 2026-06-27 branch-local additions are exceptions: default-off `count_boundary_loss`, default-off train-only `gcs_hard_sampling`, default-off E3-lite `gcs_spurious_neg`, and training-time `official_best` checkpoint selection. Results and mechanisms from later commits are retained below as legacy experiment conclusions only; they do not describe currently available CLI flags, loss items, diagnostic scripts, model outputs, configs, or selected candidates unless explicitly listed as branch-local additions here.

## Contract

```text
model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
data:  data/tusimple_gcs_fixed_y_960x544.yaml
root:  datasets/tusimple_fixed_y_k56_960x544
Q:     12
K:     56
fixed_y_start: 710 / 720 = 0.9861111111111112
fixed_y_end:   160 / 720 = 0.2222222222222222
imgsz: 544 960
```

`--imgsz 544 960` is H,W order.

The 56 fixed-y anchors are TuSimple official h-samples `710, 700, 690, ..., 160`. K56 labels must be regenerated from original TuSimple JSON and images, not resampled from old K32 labels.

Compatibility paths `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml` and `data/tusimple_gcs_fixed_y_k56_960x544.yaml` keep the same K56 contract for old run records. New training commands should use the mainline paths above.

The 5-25-3 algorithm body is intentionally not upgraded to later mainline Count Head, Quality Head, Survival Head, or near-miss machinery. The only active Count Boundary, hard-sampling, spurious-negative, and official-best behavior is the explicit branch-local default-off/protocol work recorded in `docs/agent-context/current-contracts.md`.

## Current Branch Reporting-Only Test Evidence

On 2026-07-02, active-branch artifacts were evaluated on TuSimple official
test at the user's request. Each used decode
parameters selected on official-val first; these reports are not a threshold,
checkpoint, or postprocess selection surface.

```text
gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1 + valid_before_maxdet:
summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_spurious_lite_v1_official_test_best_from_val_valid_before_maxdet/tusimple_official_summary.json
official_test_ACC = 0.965483
FP = 0.030847
FN = 0.027678
count_acc = 0.882818

gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1:
summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_count03_under5_boundary02_v1_official_test_best_from_val/tusimple_official_summary.json
official_test_ACC = 0.965428
FP = 0.031368
FN = 0.026060
count_acc = 0.874551

gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1:
val_sweep = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1_official_best_val_sweep_valid_before_maxdet/tusimple_official_sweep_summary.json
test_summary = runs/gcs_lane/gcs_yolo_lane_s_q12_k56_boundary02_count03_under5_spurious_gt5only_gt4w15_shortsidegeom025_v1_official_best_test_best_from_val/tusimple_official_summary.json
official_val_ACC = 0.977032
official_test_ACC = 0.963348
FP = 0.032171
FN = 0.031722
count_acc = 0.874191
```

The integrated conclusion is narrow: spurious-lite with the official-val
selected `valid_before_maxdet=true` decode is marginally higher than the E1
count-boundary comparator on this reporting-only test pair, but the margin is
small (`+0.000055` ACC) and must not be used for further test-time tuning.
The later short-side-geometry run improves the 363-image official-val surface,
but its reporting-only test result is lower than the no-shortside same-line
comparator and exposes a GT5 undercount/generalization gap; do not promote it
or tune from test.

## Legacy Post-b653 Official-Val Evidence

The 2026-06-25 `v2_validbranch_neg05-3` result was a post-`b6535f641`
official-val ACC leader in the later experiment line, but after the rollback it
is a legacy result only:

```text
strict ACC-best decode: conf=0.01, point_valid_thr=0.45, nms_dist_px=30, max_det=5, min_points=2/3
official-val363 ACC=0.971029
previous gate from gt4pt025=0.970975
```

A risk-reduced near-tie row uses
`conf=0.015, point_valid_thr=0.45, nms_dist_px=60, max_det=5, min_points=2/3`
with `official-val363 ACC=0.971016`, `FP=0.022498`, `FN=0.012167`,
`official_score=0.970323`, and `count_acc_4=0.878788`.

The remaining error shape is extra-lane / over-count dominated
(`GT4 4->3=1` but `4->5=11`; `GT3 3->4=9`, `3->5=3`). The next action is
official-val-only extra-lane diagnosis and a fixed-weight fine decoder sweep.
Do not continue this line by increasing valid recall, enabling valid count
floor, lowering `point_valid_thr`, increasing `max_det`, or using final test
for threshold/postprocess selection.

## Legacy Post-b653 Hard-Diagnostic Evidence

The 2026-06-26 Q18 and Q20 side-reference checks were run before looking at
official ACC. Both used the fixed train-derived `gt4pt025` GT4-hard set:

```text
hard set: data/tusimple_gt4_hard_val_gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025.txt
records: 19 images
old missing denominator: 22 GT lanes
decode: conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=8, min_points=6
```

Q18 failed the candidate-coverage gate with fixed old-missing
`raw_match_recall=9/22=0.409091` and current-missing
`raw_match_recall=8/20=0.400000`.

Q20 also fails the hard gate. It reaches only the fixed old-missing starting
line (`raw_match_recall=12/22=0.545455`) and misses the true qualification
line (`raw_match_recall>=13/22`, `after_point_valid_recall>=4/22`, and
`final_decode_recall>2/22`). On current missing lanes it gets
`raw_match_recall=10/20=0.500000`, `geometry_bad=10`, and baseline
`after_point_valid_recall=0/20`.

Decision at the time: reject Q20 as a GT4-hard geometry fix, do not run its
official-val sweep, and do not tune valid-loss weights. After the
`b6535f641` rollback, Q18/Q20/dataref tooling is not part of the active code
surface.

## Legacy Post-b653 Official-Val Candidate

The 2026-06-25 `dupmargin005_gt4pt025` run is the previous official-val
selected candidate in the later experiment line. After the `b6535f641`
rollback, it is legacy evidence only and does not define an active selected
candidate or available loss/config surface:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025
weights: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025/weights/best.pt
official-val sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_gt4pt025_official_val_sweep
official-val decode: conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=5, min_points=5
official-val363: ACC=0.970975, FP=0.017264, FN=0.012167, official_score=0.970386, count_acc=0.966942
count_acc_3/4/5=0.968610 / 0.954545 / 0.972973
```

It narrowly beats the previous `gt4short15` official-val ACC by `+0.000124`
and materially improves GT4 count accuracy (`0.954545` vs `0.848485`). The
larger `dupmargin005_gt4pt050` run is rejected on official-val
(`ACC=0.964381`, `count_acc_4=0.803030`). Before `v2_validbranch_neg05-3`,
the selected `gt4pt025` final test could only be run once with the frozen
official-val decode above. After the `v2_validbranch_neg05-3` result, the
immediate next action is the v2 extra-lane diagnostic and fine official-val
sweep, not a `gt4pt025` final-test run.

The 2026-06-25 `gt4shortrecall_lbpt_endpoint` follow-up is rejected and does
not replace `gt4pt025`: official-val `ACC=0.970301`, `FP=0.024288`,
`FN=0.013085`, `count_acc_4=0.878788`. Its selected decode removes the
official-val `4->3` case, but worsens GT4 `4->5` to `8` and remains below the
`gt4pt025` gate.

## Legacy Evaluated Candidates

The 2026-06-21 `gt4short15` run was selected on official-val in a later
post-`50999d6af` experiment line. After the rollback, it is a legacy result,
not the active code baseline:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03
weights: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03/weights/best.pt
official-val sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val_sweep
official-val decode: conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4
official-val363: ACC=0.970851, FP=0.022084, FN=0.011708, count_acc=0.939394
```

It improved the 363-image official-val ACC over the previous selected candidate by `+0.000875`, mainly from lower FN. Its count accuracy was worse, especially GT4 count accuracy. The completed train/val diagnostic and one-shot final-test report confirm that this was an official-val recall gain, not a count-robustness fix.

The 2026-06-21 reporting-only final-test result for `gt4short15` is:

```text
final test artifact: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json
final test: ACC=0.965369, FP=0.033309, FN=0.029236, count_acc=0.864486
count_acc_4=0.482906, count_acc_5=0.845343
```

The previous 2026-06-20 final-test-reported candidate was:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03
weights: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03/weights/best.pt
official-val decode: conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
official-val363: ACC=0.969976, FP=0.019559, FN=0.014463
final test: ACC=0.965459, FP=0.029439, FN=0.026270
```

The 2026-06-22 `count03_under5_00` ablation is also a rejected legacy result,
not a promoted baseline:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00
weights: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00/weights/best.pt
official-val sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_00_official_val_sweep
official-val decode: conf=0.08, point_valid_thr=0.5, nms_dist_px=50.0, max_det=6, min_points=6
official-val363: ACC=0.968578, FP=0.022590, FN=0.016529, count_acc=0.955923
final test: ACC=0.965118, FP=0.031908, FN=0.029745, count_acc=0.875270
```

It set `gcs_count_under5=0.0` and slightly improved final-test total
`count_acc` versus `count03_under5_03`, but official-val ACC, final-test ACC,
FP, and FN were worse. Its final-test run used the official-val selected
`max_det=6`, while the train args recorded `gcs_eval_max_det=8`; keep that as a
comparability caveat, not a reason to tune test.

The 2026-06-22 `dupmargin005` run is a valid near-miss from the
post-`b6535f641` duplicate-margin experiment line, not a promoted selected
candidate or an active-code capability:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03
weights: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03/weights/best.pt
official-val sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_val_sweep
official-val decode: conf=0.05, point_valid_thr=0.45, nms_dist_px=0.0, max_det=6, min_points=6
official-val363: ACC=0.970272, FP=0.024564, FN=0.016070, count_acc=0.953168
final test: ACC=0.965702, FP=0.029493, FN=0.027348, count_acc=0.865924
```

It beats the older `count03_under5_03` official-val ACC and gives the strongest
reporting-only final-test ACC recorded through 2026-06-22, but it does not beat the stronger
`gt4short15` official-val gate `0.970851`. Final test is not a selection
surface, so do not promote it or tune thresholds from its test report.

The 2026-06-23 `spurmargin003` run is a rejected follow-up from the
post-`b6535f641` spurious-margin experiment line:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03
weights: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03/weights/best.pt
official-val sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_spurmargin003_count03_under5_03_official_val_sweep
official-val decode: conf=0.02, point_valid_thr=0.5, nms_dist_px=18.0, max_det=6, min_points=5
official-val363: ACC=0.969316, FP=0.023095, FN=0.014922, count_acc=0.958678
final test: ACC=0.964522, FP=0.033591, FN=0.028636, count_acc=0.869878
```

It did not beat `count03_under5_03`, `dupmargin005`, or `gt4short15` on
official-val ACC. Its reporting-only final-test ACC is also below those three
comparators, so it is not a promotion candidate.

The 2026-06-23 `shortpos` run is a rejected follow-up from the
post-`b6535f641` positive short-lane point/visibility experiment line:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03
weights: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03/weights/best.pt
official-val sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_shortpos_count03_under5_03_official_val_sweep
official-val decode: conf=0.005, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=5
official-val363: ACC=0.968144, FP=0.027319, FN=0.017218, count_acc=0.947658
final test: ACC=0.963067, FP=0.038330, FN=0.030763, count_acc=0.861251
```

It missed the official-val gate before final-test reporting and then produced
the weakest reporting-only final-test ACC among the 2026-06-20 to 2026-06-23
candidate reports. The selected confidence `0.005` is a score-calibration
warning, and the test confusion shows both extra-lane and missed-lane damage
rather than a clean short-lane recall gain.

The follow-up train/val-only failure trace confirms the mechanism-level
failure:

```text
trace summary: runs/gcs_lane/shortpos_failure_trace_train_val_compare/summary.json
shortpos vs count03_under5_03: count_acc -0.018753, failure_images +68, spurious_extra +52, duplicate_like_extra +26, duplicate_like_extra_short_gt +33
shortpos vs dupmargin005:      count_acc -0.016547, failure_images +60, spurious_extra +30, duplicate_like_extra +18, duplicate_like_extra_short_gt +33
```

`shortpos` reduced `low_score_short_gt` and
`min_points_visibility_short_gt`, but shifted the failure mass into spurious and
duplicate-like overcount. Do not continue this exact positive short-lane loss
setting.

The 2026-06-23 `farspur001_gt5rank001` run is a rejected combined
score/ranking calibration experiment:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_farspur001_gt5rank001_count03_under5_03
official-val sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_farspur001_gt5rank001_count03_under5_03_official_val_sweep
official-val decode: conf=0.005, point_valid_thr=0.5, nms_dist_px=18.0, max_det=8, min_points=4
official-val363: ACC=0.965673, FP=0.033747, FN=0.022727, count_acc=0.917355
```

No row in the 1800-row sweep provides a usable FP/FN tradeoff. The selected
confidence `0.005` is a severe score-calibration warning, and GT4 count accuracy
falls to `0.757576`. Do not run final test for this candidate and do not
continue by sweeping `gcs_far_spurious_survival` and
`gcs_gt5_rank_consistency` together. Fix the train/val query-trace diagnostic
first, then ablate one mechanism at a time if further evidence justifies it.

These decodes were selected on official-val only in their historical experiment
contexts. None is an active-code contract after the rollback, and neither
the `gt4short15` nor `count03_under5_00` final-test report beat the older
`count03_under5_03` final-test ACC `0.965459`. The later `dupmargin005`
reporting-only final-test ACC does beat it, but official-val still controls
selection; do not use final test for threshold, checkpoint, postprocess, or
loss tuning. The later `spurmargin003`, `shortpos`, and
`farspur001_gt5rank001` runs do not improve official-val ACC.

Legacy bottleneck evidence is documented in `docs/agent-context/known-bottlenecks.md`. The post-`b6535f641` train/val diagnostics localized the main count weakness to `GT4` scenes with short visible side lanes, not to a simple decode-threshold issue. The relevant remote diagnostic artifacts are:

```text
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_count_confusion_train_val_by_visibility/summary.json
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_count_confusion_train_val_by_visibility/summary.json
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_failure_trace_train_val/failure_trace_summary.json
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val363_nms_only_conf015_maxdet6_minp4_half/tusimple_official_sweep_summary.json
runs/gcs_lane/shortpos_failure_trace_train_val_compare/summary.json
```

The 2026-06-21 `gt4short15` diagnostics split the count failures into concrete
failure modes: validation failures are overcount-heavy (`19` overcount images
vs `10` undercount images), with extra lanes mainly `spurious_extra` plus some
`duplicate_like_extra`. The NMS-only official-val363 sweep kept
`conf=0.15`, `point_valid_thr=0.5`, `max_det=6`, and `min_points=4` fixed; it
did not find a safe NMS replacement for `nms_dist_px=0.0` because NMS starts
raising FN as soon as it removes duplicate-like predictions.

The high-score unmatched-query suppression runs are rejected legacy experiments.
Their `--gcs-extra-exist` flags and `extra_exist_loss` logging are not present
in the active `b6535f641` rollback code state.
The first run,
`gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist005_count03_under5_03`,
had best 363-image official-val
`ACC=0.969603`, `FP=0.017815`, `FN=0.012856`, `count_acc=0.961433` with
`conf=0.005`, `point_valid_thr=0.5`, `nms_dist_px=18.0`, `max_det=6`, and
`min_points=5`. It improved FP/count shape but missed the then-current
official-val ACC gate `0.970851`, and the very low selected confidence indicates score
over-suppression. The smaller
`gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_extraexist0025_count03_under5_03`
run was worse: best official-val `ACC=0.961513`, `FP=0.049633`,
`FN=0.029844`, `count_acc=0.887052` with `conf=0.05`,
`point_valid_thr=0.5`, `nms_dist_px=0.0`, `max_det=6`, and `min_points=4`.
No `extraexist0025` sweep row reached the previous `extraexist005`,
`count03_under5_03`, or then-current `gt4short15` official-val ACC. Do not continue
extra-exist gain sweeps or send these candidates to final test; return to
short-lane score/geometry retention work selected only on official-val.

The short matched existence floor experiment is a rejected legacy experiment.
Its `--gcs-short-exist-*` flags are not present in the active `b6535f641` rollback code
state. It used a then-experimental `GCSLoss.exist_loss()` floor to protect only
Hungarian-matched short GT lanes after the existing APE quality and visible-IoU
quality were computed:

```text
--gcs-short-exist-floor 0.4
--gcs-short-exist-max-visible 20
--gcs-short-exist-floor-max-ape 20.0
--gcs-short-exist-floor-min-iou 0.3
```

The completed 363-image official-val sweep was valid but diagnostic-only:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_shortexist04_count03_under5_03
sweep: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_shortexist04_count03_under5_03_official_val_sweep
best: conf=0.1, point_valid_thr=0.5, nms_dist_px=30.0, max_det=8, min_points=5
official-val ACC=0.968966, FP=0.019972, FN=0.014922, official_score=0.968268, count_acc=0.950413
count_acc_3=0.968610, count_acc_4=0.863636, count_acc_5=0.972973
```

It improved FP and count accuracy relative to the then-current `gt4short15`
candidate, but missed that official-val ACC gate `0.970851` by
`0.001885` and raises FN by `0.003214`. Do not send it to final test. Its
`args.yaml` records `amp: true`, unlike the original `--no-amp` template, so
reproducibility notes should use the recorded args rather than the template.

## Full Training

Run formal training on the remote CUDA server from a clone checked out to this branch:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssh_lane

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

If `batch=32` OOMs, reduce it only for OOM or instability and record the change in the run notes.

`--no-amp` is included because the current remote run hit Ultralytics AMP self-check loading `yolo26n.pt`. Remove it only after that server-side checkpoint/cache problem is fixed and record the change.

## Label Rebuild

For this 5-25-3 branch, K56 `.npz` labels must include `semantic_mask` and `edge_mask` because the branch trains `mask_loss` and `edge_loss`. If the remote clone has `datasets` as a symlink to a shared K56 dataset that lacks those arrays, replace only this clone's symlink with a real local runtime directory before rebuilding:

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

## Local Checks

```bash
python tools/check_model.py \
  --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --imgsz 544 960 \
  --batch 1 \
  --device cpu
```

Expected key shapes:

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
aux_mask_logits: B x 2 x 544 x 960
aux_edge_logits: B x 1 x 544 x 960
```

## Agent Tooling

Agent/Skill configuration is local Codex workspace context, not part of the server-side algorithm payload for this branch. Server synchronization only needs the training, evaluation, diagnostic, model, config, and data-conversion code required to run experiments.
