# GCS-YOLO-Lane 5-25-3 K56 Mainline

This is the current GCS-YOLO-Lane mainline branch. It imports the historical `5-25-3.zip` algorithm and adapts only the TuSimple fixed-y contract.

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

## Evaluated Candidate

The 2026-06-20 evaluated candidate is:

```text
run: gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03
weights: runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03/weights/best.pt
official-val decode: conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
official-val363: ACC=0.969976, FP=0.019559, FN=0.014463
final test: ACC=0.965459, FP=0.029439, FN=0.026270
```

The decode was selected on official-val only. The final test result is reporting evidence and must not be used for threshold, checkpoint, or postprocess tuning.

Current bottleneck evidence is documented in `docs/agent-context/known-bottlenecks.md`. The 2026-06-20 train/val diagnostic localizes the main count weakness to `GT4` scenes with short visible side lanes, not to a simple decode-threshold issue. The remote diagnostic artifact is:

```text
runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_count_confusion_train_val_by_visibility/summary.json
```

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
