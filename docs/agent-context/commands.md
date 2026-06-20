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

## Current Evaluated Candidate

The 2026-06-20 evaluated candidate uses:

```text
source run:   runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03
weights:      weights/best.pt
val sweep:    runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_official_val363_sweep
final test:   runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_official_test_best_from_val
decode:       conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
official-val: ACC=0.969976, FP=0.019559, FN=0.014463, count_acc=0.969697
final test:   ACC=0.965459, FP=0.029439, FN=0.026270, count_acc=0.872753
```

Use this test result only as final reporting evidence for the selected official-val candidate. Do not use it to tune thresholds, checkpoint choice, or postprocess settings.

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

Recommended next formal train-side experiment:

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

This branch was imported from `5-25-3.zip` and now has project `.codex/`, `.agents/`, agent helper scripts, and standalone TuSimple official eval/sweep helpers synchronized from the main repository. Mainline Count/Quality/Survival tests and training-time official-best checkpoint preservation are intentionally not part of this branch unless a future task explicitly restores them.

## Agent Setup Check

After Agent, Skill, context, or delegation-policy changes:

```bash
python scripts/check_gcs_agent_setup.py
```

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
