# GCS-YOLO-Lane 5-25-3 K56 Mainline

This is the current GCS-YOLO-Lane mainline branch. It imports the historical `5-25-3.zip` algorithm and adapts only the TuSimple fixed-y contract.

Active source/config is based on rollback commit `50999d6af` (`Document 5-25-3 K56 as mainline`) plus the default-disabled `duplicate_margin_loss` experiment knob added on 2026-06-22. Results and mechanisms from other later commits are retained below as legacy experiment conclusions only; they do not describe currently available CLI flags, loss items, diagnostic scripts, or selected candidates.

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

The 5-25-3 algorithm body is intentionally not upgraded to later mainline Count Head, Count Boundary, Quality Head, Survival Head, near-miss, or official-best checkpoint machinery.

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

The 2026-06-22 `dupmargin005` run is a valid near-miss for the current
default-disabled `duplicate_margin_loss` experiment, not a promoted selected
candidate:

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

These decodes were selected on official-val only in their historical experiment
contexts. None is a current active-code contract after the rollback, and neither
the `gt4short15` nor `count03_under5_00` final-test report beat the older
`count03_under5_03` final-test ACC `0.965459`. The later `dupmargin005`
reporting-only final-test ACC does beat it, but official-val still controls
selection; do not use final test for threshold, checkpoint, or postprocess
tuning.

Current bottleneck evidence is documented in `docs/agent-context/known-bottlenecks.md`. The train/val diagnostics localize the main count weakness to `GT4` scenes with short visible side lanes, not to a simple decode-threshold issue. The relevant remote diagnostic artifacts are:

```text
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_count_confusion_train_val_by_visibility/summary.json
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_count_confusion_train_val_by_visibility/summary.json
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_failure_trace_train_val/failure_trace_summary.json
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val363_nms_only_conf015_maxdet6_minp4_half/tusimple_official_sweep_summary.json
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
in the active `50999d6af` code state.
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
Its `--gcs-short-exist-*` flags are not present in the active `50999d6af` code
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
