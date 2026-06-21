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
  --gcs-count 0.3 \
  --gcs-count-under5 0.3 \
  --gcs-count-under5-min-lanes 5 \
  --gcs-lane-count-balanced \
  --project runs/gcs_lane \
  --name gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03
```

If `batch=32` OOMs on the target machine, reduce batch only for OOM/instability and record the change in the run notes.

`--no-amp` is included because the current remote run hit an Ultralytics AMP self-check failure while loading `yolo26n.pt`. If that server cache/checkpoint issue is fixed, AMP may be re-enabled only with a run note.

## Current Official-Val Selection

The 2026-06-21 official-val selected candidate uses:

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

This final-test report is reporting-only and did not improve over the previous `count03_under5_03` final-test ACC `0.965459`. Keep `gt4short15` as official-val selected, but do not claim a final-test improvement and do not tune from final test. The final-test run used `max_det=6` because that was the official-val selected postprocess, even though the train-time args recorded `gcs_eval_max_det=8`.

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

## Extra Exist Suppression Loss Experiment

The high-score unmatched-query suppression loss is an explicit experimental
option. Defaults preserve baseline behavior:

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
relative to `gt4short15`, but it missed the current official-val ACC gate
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

At the current `gt4short15` decode (`conf=0.15`, `point_valid_thr=0.5`,
`nms_dist_px=0.0`, `max_det=6`, `min_points=4`), the `extraexist0025` run gets
only `ACC=0.961450`, `FP=0.045638`, `FN=0.029844`, `count_acc=0.909091`. Its
best count row reaches only `count_acc=0.928375` with `ACC=0.961400`, still
below current `gt4short15` count accuracy and far below the ACC gate. No row in
the 1800-row sweep reaches `extraexist005` ACC `0.969603`, previous
`count03_under5_03` ACC `0.969976`, or current `gt4short15` ACC `0.970851`.

Decision: do not promote `extraexist0025`, do not run final test, and do not
continue sweeping `gcs_extra_exist` gains. Return to train-side short-lane
score/geometry retention work; select any new mechanism only on the same
363-image official-val surface.

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

## Train/Val Count-Confusion Diagnostic

The 2026-06-20 train/val diagnostic for `count03_under5_03` groups decoded lane-count confusion by date, GT lane count, and the shortest visible GT lane bucket. It uses the frozen official-val selected decode and does not touch final test:

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

## GT4 Short-Lane Weighted Training

The GT4 short-lane sampler boost is an explicit experimental option. Defaults preserve baseline behavior:

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

This is now the official-val selected candidate, but its count accuracy is worse
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
