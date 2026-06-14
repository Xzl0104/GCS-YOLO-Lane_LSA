# GCS-YOLO-Lane

GCS-YOLO-Lane is a YOLO11-based structured lane detection project. It is not a standard YOLO segmentation setup: the model predicts lane instances as ordered 2D point sequences.

The current research target is clean TuSimple official Accuracy under a reproducible, leakage-free protocol.

## Current Mainline

- Default model: `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml`
- Default data config: `data/tusimple_gcs_fixed_y_960x544.yaml`
- TuSimple input size: `--imgsz 544 960` in H,W order
- Label mode: fixed-y, `K=32`, `fixed_y_start=710/720`, `fixed_y_end=0.25`
- Default split root: `datasets/tusimple_fixed_y_960x544`

The model output contract includes:

```text
pred_points: B x Q x K x 2
pred_logits: B x Q
pred_valid_logits: B x Q x K
pred_quality_logits: B x Q
pred_count_logits: B x 4
pred_count_boundary_logits: B x 2
```

`pred_count_logits` predicts image-level lane count classes 2/3/4/5. `pred_count_boundary_logits` calibrates count>=4 and count>=5 inside the default Count Head loss/decode path.

## Count Head Status

The current mainline uses conservative count-generalization defaults:

```text
gcs_count_sum = 0.03
gcs_quality = 0.4
gcs_quality_neg_weight = 0.5
gcs_count_cls_w2/w3/w4/w5 = 0.5/1.2/1.4/1.8
gcs_count_boundary_gt5_pos_weight = 1.15
gcs_point_valid_gt5_pos_weight = 2.0
gcs_gt5_edge_loss_weight = 1.15
gcs_candidate_gt5_edge_weight = 1.10
gcs_point_valid_gt5_edge_continuity = 0.05
gcs_point_valid_gt5_edge_continuity_thr = 0.55
gcs_gt5_oversample_weight = 1.0
gcs_group_sampler_ratios = 2:0.01,3:0.29,4:0.42,5:0.28
```

The GT5 candidate-quality knobs are training-side only. They add small extra supervision to real matched GT5 edge queries and adjacent visible point-valid anchors; they do not change decode or fabricate lanes.

`gcs_soft_count_decision`, `gcs_last_lane_rescue`, and `gcs_edge_last_lane_rescue` remain default-off. Select rescue, soft-count, thresholds, and checkpoints on official-val only.

## Current Experiment Status

The 2026-06-13 GT5-only gates inside the current `K=32` contract are not promotable:

```text
gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0
gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0
```

They reached independent official-val `0.953639` and `0.953587`, below the active references `0.954137` and `0.954782`.

For the `0.97` objective, the higher-level bottleneck is now the current `K=32` fixed-y representation and official-grid alignment. The official-val label oracle for the current `K=32` fixed-y contract is only `Accuracy=0.956249`, `FP=0`, `FN=0.003444`, leaving too little headroom over the current-code audit baseline `0.953756`.

The active next path is a separate `Q12-K56` official-h-sample-aligned experimental candidate:

```text
model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
data:  data/tusimple_gcs_fixed_y_k56_960x544.yaml
root:  datasets/tusimple_fixed_y_k56_960x544
K:     56, fixed-y anchors aligned to TuSimple h_samples 710..160 step 10
```

K56 labels are regenerated from original TuSimple JSON and images, not resampled from K32 labels. The K56 official-val label oracle is `Accuracy=0.998256`, `FN=0.001377`, `FP=-0.000689` on the 363-image official-val split. The current mainline remains `K=32`; do not silently mix `K=56` labels with existing `K=32` data or checkpoints.

The formal K56 baseline is running on the remote RTX 4090 24GB server as:

```text
gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4
batch=32
workers=4
```

Monitoring snapshot at `2026-06-14` after ordinary epoch 160 and official-val candidate epoch 160: the run is alive. GPU memory is about 17.9-18.6 GiB of 24.6 GiB, so the current `batch=32` server run should not be interrupted or changed mid-run. Official-best remains epoch 152 with `official_acc=0.959315`, above the current-code K32 audit `0.953756` by `+0.005559`, prior K56 epoch127 best `0.958484` by `+0.000831`, and legacy `0.959224` by `+0.000091`. Epoch 142 remains the best ordinary-val row so far (`val/f1=0.962083`), while epoch160 latest ordinary row is `val/f1=0.957822`; epoch160 official-val candidate is `0.959138`. Retained official Top-K is `152, 154, 140, 153, 155`. This is a useful K56 milestone, but K56 is not promoted and no test result is claimed because the run is still `160/180` and the legacy margin is small. Continue the healthy `batch=32` run to completion; do not launch Count/Quality calibration or final test until official-val selection is stable after the baseline completes.

Use the local RTX 4060 8GB workstation for smoke, contract, label/oracle, and model-shape checks only. Run formal training and official-val evaluation on the remote server.

Default-off training knobs remain available for controlled experiments:

```text
gcs_quality_gt5_edge_floor = 0.0
gcs_quality_hard_negative_from_head = False
gcs_hard_negative_visible_segment = False
gcs_hard_negative_visible_thr = 0.5
gcs_hard_negative_visible_support_points = 12.0
gcs_point_valid_gt5_edge_segment = 0.0
gcs_point_valid_gt5_edge_segment_thr = 0.65
gcs_point_valid_gt5_edge_segment_min_points = 5
```

When `gcs_quality_hard_negative_from_head` is enabled, Quality Head hard negatives are mined from unmatched queries only; matched queries remain matched quality targets even if their current continuous quality target is `0.0`.

The visible-segment hard-negative and GT5 edge Quality floor recipes remain default-off infrastructure only. Do not rerun the same recipes as the next gate unless checking reproducibility, and do not use test to rescue or tune them.

## Environment

Use the existing Windows CUDA conda environment:

```text
D:\miniconda3\envs\lsa_yolo
```

If activation is unavailable, call Python directly:

```powershell
D:\miniconda3\envs\lsa_yolo\python.exe
```

## Common Commands

Train:

```powershell
python tools/train_gcs.py --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --data data/tusimple_gcs_fixed_y_960x544.yaml --imgsz 544 960
```

Remote experiment commands live in `docs/agent-context/commands.md`. Keep longer training and official-val evaluation on the remote CUDA server with the `ssh_lane` conda environment.

Official-val sweep:

```powershell
python tools/sweep_tusimple_official.py --weights <weights.pt> --split val --imgsz 544 960
```

GT5 diagnosis on official-val:

```powershell
python tools/diagnose_gcs_gt5.py --weights <weights.pt> --split val --imgsz 544 960
```

Final test evaluation, only after selecting the candidate on official-val:

```powershell
python tools/eval_tusimple_official.py --weights <weights.pt> --split test --imgsz 544 960
```

## Validation

Run targeted checks before metric work:

```powershell
python scripts/verify_loss_cleanup.py
python tools/check_gcs_count_head_topk_contract.py
python tools/check_gcs_decode_meta_contract.py
python tools/check_gcs_algorithm_contract.py
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --imgsz 544 960
```

## Protocol Rules

- Use official-val for checkpoint, threshold, rescue, ranking, count-policy, and postprocess selection.
- Use test only once for final evaluation of an already selected candidate.
- Do not use GT during inference/decode.
- Do not fabricate lanes.
- Do not claim improvement without official-val evidence.

Detailed agent and experiment context lives in `docs/agent-context/`.
