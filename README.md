# GCS-YOLO-Lane 5-25-3 K56 Branch

This branch imports the historical `5-25-3.zip` GCS-YOLO-Lane algorithm as a separate Git branch and adapts only the TuSimple fixed-y contract.

## Contract

```text
model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
data:  data/tusimple_gcs_fixed_y_k56_960x544.yaml
root:  datasets/tusimple_fixed_y_k56_960x544
Q:     12
K:     56
fixed_y_start: 710 / 720 = 0.9861111111111112
fixed_y_end:   160 / 720 = 0.2222222222222222
imgsz: 544 960
```

`--imgsz 544 960` is H,W order.

The 56 fixed-y anchors are TuSimple official h-samples `710, 700, 690, ..., 160`. K56 labels must be regenerated from original TuSimple JSON and images, not resampled from old K32 labels.

The 5-25-3 algorithm body is intentionally not upgraded to later mainline Count Head, Count Boundary, Quality Head, Survival Head, near-miss, or official-best checkpoint machinery.

## Full Training

Run formal training on the remote CUDA server from a clone checked out to this branch:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssh_lane

python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml \
  --data data/tusimple_gcs_fixed_y_k56_960x544.yaml \
  --imgsz 544 960 \
  --name gcs_5_25_3_q12_k56_e220_seed1_b32w4 \
  --pretrained yolo11s-seg.pt \
  --epochs 220 \
  --batch 32 \
  --workers 4 \
  --seed 1
```

If `batch=32` OOMs, reduce it only for OOM or instability and record the change in the run notes.

## Label Rebuild

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

## Local Checks

```bash
python tools/check_model.py \
  --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml \
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
