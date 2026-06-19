# Commands

This file records commands for branch `codex/5-25-3-k56`, which imports the historical `5-25-3.zip` algorithm and changes only the TuSimple K56 fixed-y contract.

All TuSimple commands must use:

```bash
--imgsz 544 960
```

This is H,W order.

## Branch Contract

```text
model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
data:  data/tusimple_gcs_fixed_y_k56_960x544.yaml
root:  datasets/tusimple_fixed_y_k56_960x544
Q:     12
K:     56
fixed_y_start: 710 / 720 = 0.9861111111111112
fixed_y_end:   160 / 720 = 0.2222222222222222
```

The 5-25-3 branch does not include later mainline `--gcs-official-best`, Count/Quality/Survival, near-miss, or official-val helper machinery.

## Full Remote Training

Use the remote RTX 4090 environment for formal training:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssh_lane
```

Run from the dedicated remote Git clone checked out to branch `codex/5-25-3-k56` or an exact commit from that branch:

```bash
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

If `batch=32` OOMs on the target machine, reduce batch only for OOM/instability and record the change in the run notes.

## Label Rebuild

Regenerate K56 labels from original TuSimple JSON and images. Do not resample historical K32 labels:

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

## Model Shape Check

```bash
python tools/check_model.py \
  --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml \
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

This branch was imported from `5-25-3.zip`, which does not contain the current project `.codex/config.toml`, `scripts/`, `tests/`, or later mainline official-val helper scripts. Agent setup checks and mainline Count/Quality/Survival tests are intentionally not part of this branch unless a future task explicitly restores them.
