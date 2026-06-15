# Commands

All TuSimple commands should use:

```bash
--imgsz 544 960
```

Use hardware-aware batch strategy:

```text
local RTX 4060 8GB: smoke/contract/oracle checks only, small batches
remote RTX 4090 24GB: formal training/evaluation, default Q12/K56 batch=32 workers=4
```

## Train

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --imgsz 544 960
```

## Train With Official-Val Checkpoint Preservation

Use this when ordinary `best.pt` is not reliable for TuSimple official Accuracy selection.

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --imgsz 544 960 \
  --gcs-official-best \
  --gcs-official-best-period 1 \
  --gcs-official-best-top-k 3 \
  --gcs-official-best-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-best-archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset
```

`official_best.pt` is selected by official-val `official_acc`. `gcs_official_best_top_k > 1` additionally preserves retained candidates under `weights/official_topk/` and records them in `official_best_summary.json`.

## Next Remote Official-Val Experiments

Do not launch another `K=32` GT5 quality/count fine-tune as the next main path: the visible-segment hard-negative and GT5 edge Quality floor gates have both completed and are not promotable, and the `K=32` label oracle does not leave enough geometry headroom for the `0.97` objective.

The current experimental K56 family/reference is the separate `Q12-K56` official-h-sample-aligned candidate:

```text
data:  data/tusimple_gcs_fixed_y_k56_960x544.yaml
model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
root:  datasets/tusimple_fixed_y_k56_960x544
K:     56, aligned to TuSimple h_samples 710..160 step 10
```

The K56 labels must be regenerated from original TuSimple JSON and images, not from existing K32 labels. The K56 official-val label oracle is `Accuracy=0.998256` on the 363-image official-val split.

When a new remote CUDA experiment is selected, run it from a dedicated Git clone checked out to the exact pushed commit SHA. Do not run training locally from Codex. Activate the remote CUDA environment first:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssh_lane
```

K56 label rebuild command:

```bash
python tools/rebuild_tusimple_fixed_y_k56_from_reference_split.py \
  --archive-root archive \
  --output-root datasets/tusimple_fixed_y_k56_960x544 \
  --reference-root datasets/tusimple_fixed_y_960x544
```

K56 label oracle command:

```bash
python tools/check_tusimple_fixed_y_label_oracle.py \
  --dataset-root datasets/tusimple_fixed_y_k56_960x544 \
  --label-split val \
  --archive-root archive
```

K56 one-anchor official-val impact diagnostic:

```bash
python tools/check_tusimple_fixed_y_label_oracle.py \
  --dataset-root datasets/tusimple_fixed_y_k56_960x544 \
  --label-split val \
  --archive-root archive \
  --diagnose-one-anchor-impact \
  --save-dir runs/gcs_lane/tusimple_fixed_y_k56_label_oracle_val_one_anchor_diag
```

This diagnostic is GT-assisted and official-val-only. It appends exact raw one-anchor GT lanes to a copy of the current K56 label-oracle predictions to quantify the representation/export upper bound. It is not model evidence, must not be run on test, and must not be used for checkpoint, threshold, postprocess, loss, or promotion choices.
The GT-assisted prediction artifact is named `one_anchor_gt_assisted_not_for_selection_predictions.json` and is written with a matching `one_anchor_gt_assisted_not_for_selection_protocol.json` sidecar.

K56 exact fixed-y artifact check:

```bash
python tools/check_gcs_label_order_split.py \
  --dataset-root datasets/tusimple_fixed_y_k56_960x544 \
  --expect-fixed-y 56,710/720,160/720
```

Raw TuSimple h-sample endpoint audit:

```bash
python tools/analyze_tusimple_hsample_endpoints.py \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --split val
```

K56 formal remote baseline command:

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml \
  --data data/tusimple_gcs_fixed_y_k56_960x544.yaml \
  --imgsz 544 960 \
  --name gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4 \
  --pretrained yolo11s-seg.pt \
  --epochs 180 \
  --batch 32 \
  --workers 4 \
  --seed 1 \
  --gcs-official-best \
  --gcs-official-best-period 1 \
  --gcs-official-best-top-k 5 \
  --gcs-official-best-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-best-archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset
```

Completed K56 remote baseline state:

```text
run: gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4
remote training HEAD: 9b9769b61f8f
remote final audit branch HEAD: 655c116
formal batch: 32
workers: 4
GPU memory during training: about 17.6-18.6 GiB / 24.6 GiB on RTX 4090 24GB
status at 2026-06-14 final audit: training completed naturally, process exited, GPU idle, results.csv has 180 rows
ordinary val final row: epoch=180, val/f1=0.955190, precision=0.951182, recall=0.959231, fp=64, fn=53, val/decode/k5_to_output4_rate=0.080000
ordinary val best row by val/f1: epoch=142, val/f1=0.962083, precision=0.958047, recall=0.966154, fp=55, fn=44, val/decode/k5_to_output4_rate=0.067568
official_best: epoch 152, official_acc=0.959315, FP=0.045225, FN=0.028466
official_best count/GT5 diagnostics: count_acc_3/4/5=0.928251/0.878788/0.851351, gt5_output5_rate=0.851351, gt5_count_head_under_rate=0.067568, gt5_valid_points_fail_rate=0.081081, gt5_candidate_pool_shortfall_rate=0.000000, gt5_top5_suppressed_by_nms_rate=0.000000, decode/k5_to_output4_rate=0.105263, rescue_precision=0.779412, rate_3_to_4=0.071749, rate_4_to_5=0.075758, rate_5_to_4=0.148649, matched/unmatched_quality_mean=0.913939/0.831922
note: the run-summary `gt5_valid_points_fail_rate=0.081081` above is an official_best/decode aggregate; the independent GT5 rank-diagnosis drop attribution below reports `valid_points_fail=0`, so keep the two diagnostic scopes distinct.
official_top_k retained epochs and ACC: 152=0.959315, 170=0.959247, 166=0.959244, 168=0.959217, 165=0.959215
latest official-val candidate: epoch=180, official_acc=0.959087, FP=0.048072, FN=0.030762, count_acc_3/4/5=0.928251/0.893939/0.864865, gt5_output5_rate=0.864865, gt5_count_head_under_rate=0.067568, gt5_valid_points_fail_rate=0.067568, gt5_candidate_pool_shortfall_rate=0.000000, gt5_top5_suppressed_by_nms_rate=0.000000, decode/k5_to_output4_rate=0.093333, matched/unmatched_quality_mean=0.923615/0.818245
independent official-val sweep: runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/analysis_official_best_val_sweep/tusimple_official_sweep_summary.json, 64 val combinations, best reproduces epoch152 official_acc=0.959315 at conf=0.005, point_valid_thr=0.35, nms_dist_px=18.0, max_det=5, min_points=6, rank_min_points=none
GT5 rank-diagnosis drop attribution: runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/analysis_official_best_gt5_diag_val/gt5_rank_diagnostics_summary.json, kept=63/74, count_head_under_predict=5, quality_too_low=6, candidate_pool_shortfall=0, GT5 NMS suppression=0, rank5_score_low=0, valid_points_fail=0
diagnostic top-k notes: epoch152 exceeds the current-code K32 audit 0.953756 by +0.005559, countboundary 0.954137 by +0.005178, old FT6 0.954782 by +0.004533, prior K56 epoch127 best 0.958484 by +0.000831, epoch115 by +0.001355, and legacy 0.959224 by +0.000091. It remains below the 0.97 objective and is not promoted because the legacy margin is tiny. The remaining blocker is not representation oracle, candidate supply, rank, or NMS; it is Count/Quality separation around GT5 5->4 and false fifth-lane pressure.
errors: final process exited; results.csv has no numeric NaN/Inf values across 180 rows; 181 run JSON files have no parse error and no numeric NaN/Inf values; a text-artifact scan of 363 files found no `--split test`, `split: test`, `split=test`, `test_label.json`, or `test_set` hits, and no `Traceback`, `RuntimeError`, `shape error`, or `shape mismatch` hits. args.yaml records split=val, gcs_official_best_split=val, imgsz=[544, 960], gcs_imgsz=[544, 960], and K56 data/model.
decision: K56 baseline is a stronger official-val reference, but not a final/promotable test candidate. Do not use test for selection. Do not rerun the rejected K56 Count/Quality gates below.
```

K56 min-points validation-only grid result from the parent `official_best.pt`:

```text
first grid best: official_acc=0.959475, FP=0.043618, FN=0.027548, conf=0.005, point_valid_thr=0.35, nms_dist_px=18, max_det=5, candidate_min_points=5, final_min_points=9, fifth_min_points=4
pv030_040 grid best: official_acc=0.959750, FP=0.043848, FN=0.028466, conf=0.005, point_valid_thr=0.40, nms_dist_px=18, max_det=5, candidate_min_points=5, final_min_points=9, fifth_min_points=4
risk: count_acc_4=0.863636 on the pv030_040 best row, so treat it as a validation-only postprocess candidate with GT4-to-5 false fifth-lane risk, not a promotion decision
```

K56 fifthness-v1 default-off candidate:

```text
model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-fifthness-v1.yaml
data:  data/tusimple_gcs_fixed_y_k56_960x544.yaml
goal:  reduce both GT5 5->4 drops and GT4/GT3 false fifth-lane pressure
```

This candidate is opt-in. The YAML enables the optional `pred_fifthness_logits: B x Q` head and Count Head fifth-candidate evidence; the training losses and fifthness decode remain default-off until non-zero gains or explicit decode switches are passed. Use official-val only for selection and do not use the diagnostic K56 test audit to choose gains. By default, fifthness negatives come from GT3/GT4 unmatched outside candidates; `--gcs-fifthness-include-gt5-negatives True` is an explicit follow-up switch for also mining GT5 same-image unmatched outside false fifth candidates.

K56 count5ev-v1 default-off Count Head evidence isolation candidate:

```text
model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count5ev-v1.yaml
data:  data/tusimple_gcs_fixed_y_k56_960x544.yaml
goal:  isolate Count Head fifth-candidate evidence without emitting `pred_fifthness_logits`
```

This candidate is opt-in. The YAML enables `use_count_fifth_evidence=True` while keeping `use_fifthness=False`, so it must preserve the normal six model outputs. Use it to test whether fifth-candidate evidence alone improves Count Head GT4/GT5 calibration before adding fifthness verifier losses or fifthness decode.

Local shape check before any remote run:

```bash
python tools/check_model.py \
  --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count5ev-v1.yaml \
  --imgsz 544 960 \
  --batch 1
```

Remote official-val short gate from the K56 epoch152 parent:

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count5ev-v1.yaml \
  --data data/tusimple_gcs_fixed_y_k56_960x544.yaml \
  --imgsz 544 960 \
  --name gcs_yolo_lane_s_q12_k56_count5ev_v1_ft8_seed1_b32w4 \
  --pretrained runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/weights/official_best.pt \
  --epochs 8 \
  --batch 32 \
  --workers 4 \
  --seed 1 \
  --lr0 0.00005 \
  --lrf 0.2 \
  --gcs-official-best \
  --gcs-official-best-period 1 \
  --gcs-official-best-top-k 5 \
  --gcs-official-best-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-best-archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset
```

Do not start full/e180 unless the short gate improves or at least matches the K56 parent while keeping FP controlled, lowering `rate_4_to_5`, and avoiding GT5 `rate_5_to_4` regression.

First remote attempt status:

```text
run: gcs_yolo_lane_s_q12_k56_count5ev_v1_ft8_seed1_b32w4
status: incomplete, stopped after epoch 3/8
environment risk: server root filesystem had about 687M free after the stop
partial official-val: epoch1=0.958453, epoch2=0.957082, epoch3=0.958500
decision: not promotable; do not start full/e180 from this run
```

Before rerunning this FT8 gate, free server disk space or move run artifacts off the nearly full root filesystem.

The 2026-06-15 reliability audit closed the missing inference loop: `pred_fifthness_logits` can now be consumed by decode with:

```text
--use-fifthness-decode
--fifthness-decode-thr <0.0-1.0>
--fifthness-decode-rank-weight <explicit-weight>
```

For trainer-owned official-val sweeps, use the corresponding `tools/train_gcs.py` flags:

```text
--gcs-use-fifthness-decode
--gcs-fifthness-decode-thr <0.0-1.0>
--gcs-fifthness-decode-rank-weight <explicit-weight>
```

These decode knobs affect only the selected fifth lane and fifth-lane rescue candidates. They require a model that emits `pred_fifthness_logits`; enabling them against the default K56 model must fail rather than silently fall back.

The first short gate is rejected:

```text
run: gcs_yolo_lane_s_q12_k56_fifthness_v1_ft8_seed1_b32w4
commit: 393345a7dcb014b9e6b54d807579463a896e3a79
status: completed 8/8 epochs on official-val only
best official-val: epoch 5, official_acc=0.959006, FP=0.046097, FN=0.029155
parent reference: official_acc=0.959315, FP=0.045225, FN=0.028466
diagnosis: count_acc_4=0.848485, count_acc_5=0.878378, rate_4_to_5=0.106061, rate_5_to_4=0.121622, gt5_output5_rate=0.878378, unmatched_quality_mean=0.859536
decision: not promotable; do not start full training from this recipe
implementation finding: the first gate did not explicitly train same-image GT5 false outside candidates below true GT5 edge matches. Keep the original GT3/GT4 negative contract as default and test this as a new explicit switch.
```

Local shape check before any remote run:

```bash
python tools/check_model.py \
  --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-fifthness-v1.yaml \
  --imgsz 544 960 \
  --batch 1
```

Remote official-val training template from the K56 epoch152 parent:

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-fifthness-v1.yaml \
  --data data/tusimple_gcs_fixed_y_k56_960x544.yaml \
  --imgsz 544 960 \
  --name gcs_yolo_lane_s_q12_k56_fifthness_v1_ft8_seed1_b32w4 \
  --pretrained runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/weights/official_best.pt \
  --epochs 8 \
  --batch 32 \
  --workers 4 \
  --seed 1 \
  --lr0 <explicit-lr0> \
  --lrf <explicit-lrf> \
  --gcs-fifthness <explicit-gain> \
  --gcs-fifthness-pairwise <explicit-gain> \
  --gcs-fifthness-include-gt5-negatives <True-or-False> \
  --gcs-quality-pairwise <explicit-gain> \
  --gcs-count-cumulative <explicit-gain> \
  --gcs-official-best \
  --gcs-official-best-period 1 \
  --gcs-official-best-top-k 5 \
  --gcs-official-best-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-best-archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset
```

The GT5-negative switch follow-up gate is rejected:

```text
run: gcs_yolo_lane_s_q12_k56_fifthness_gt5neg_ft8_seed1_b32w4
commit: 9a81d76b00535a6966840e808f08f13889ffdf95
status: completed 8/8 epochs on official-val only
best official-val: epoch 6, official_acc=0.959319, FP=0.047429, FN=0.027089
parent reference: official_acc=0.959315, FP=0.045225, FN=0.028466
diagnosis: count_acc_4=0.833333, count_acc_5=0.932432, rate_4_to_5=0.121212, rate_5_to_4=0.067568, gt5_output5_rate=0.932432, gt5_valid_points_fail_rate=0.013514
decision: not promotable; the tiny ACC delta is not meaningful because FP and GT4-to-5 pressure worsened
```

Decision after combining both short gates with the server evidence: do not start K56/fifthness full/e180 training from either exact recipe.

The first closed-loop official-val fifthness decode sweep of the `fifthness_gt5neg` official-best checkpoint is rejected:

```text
thr=0.30: official_acc=0.959319, FP=0.047429, FN=0.027089, rate_4_to_5=0.121212, rate_5_to_4=0.067568
thr=0.50: official_acc=0.959319, FP=0.047429, FN=0.027089, rate_4_to_5=0.121212, rate_5_to_4=0.067568
thr=0.70: official_acc=0.959220, FP=0.046465, FN=0.027778, rate_4_to_5=0.106061, rate_5_to_4=0.121622
thr=0.85: official_acc=0.959023, FP=0.042470, FN=0.028466, rate_4_to_5=0.030303, rate_5_to_4=0.162162
```

Do not rerun these exact threshold sweeps as the next path. The official-val-only fifthness score distribution/case audit has now been run and also does not justify full/e180.

Fifthness score distribution audit command:

```bash
python tools/audit_gcs_fifthness_scores.py \
  --weights runs/gcs_lane/gcs_yolo_lane_s_q12_k56_fifthness_gt5neg_ft8_seed1_b32w4/weights/official_best.pt \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset \
  --imgsz 544 960 \
  --conf 0.005 \
  --point-valid-thr 0.35 \
  --nms-dist-px 18 \
  --max-det 5 \
  --candidate-conf 0.005 \
  --candidate-point-valid-thr 0.35 \
  --candidate-min-points 5 \
  --final-min-points 6 \
  --fifth-min-points 5 \
  --thresholds 0.30 0.50 0.70 0.85 \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_q12_k56_fifthness_gt5neg_ft8_seed1_b32w4/analysis_official_best_val_fifthness_score_audit
```

Result:

```text
artifact: runs/gcs_lane/gcs_yolo_lane_s_q12_k56_fifthness_gt5neg_ft8_seed1_b32w4/analysis_official_best_val_fifthness_score_audit/fifthness_score_audit_summary.json
GT4 unmatched output5 false-fifth scores: count=8, median=0.841094, mean=0.826202, max=0.942222
GT5 selected rank5 matched output5 scores: count=58, median=0.960218, mean=0.931124, min=0.576185
GT4 unmatched output5 Count Head: median P4=0.003675, median P5=0.996325, median margin=0.992650
GT5 selected rank5 matched Count Head: median P4=0.0000009, median P5=0.999999, median margin=0.999998
threshold 0.30/0.50: true_keep=1.000000, false_keep=1.000000
threshold 0.70: true_keep=0.965517, false_keep=0.875000
threshold 0.85: true_keep=0.931034, false_keep=0.375000
decision: score separation is real but not clean enough for the joint objective, and the false fifth cases show Count Head P5 overconfidence; do not launch full/e180 from this checkpoint
```

Reference command shape for a future official-val-only fifthness decode sweep with a new hypothesis:

```bash
python tools/sweep_tusimple_official.py \
  --weights <fifthness-enabled-official-val-selected-weights.pt> \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset \
  --imgsz 544 960 \
  --confs 0.005 \
  --point-valid-thrs 0.35 0.40 \
  --nms-dist-pxs 18 \
  --max-dets 5 \
  --candidate-min-points 5 \
  --final-min-points 6 7 8 9 \
  --fifth-min-points 4 5 6 \
  --use-fifthness-decode \
  --fifthness-decode-thr <explicit-threshold> \
  --fifthness-decode-rank-weight <explicit-weight>
```

Short closed-loop training gate template, only after an official-val case audit identifies a new hypothesis that explicitly targets lower `rate_4_to_5`/FP without worsening GT5 `5->4`:

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-fifthness-v1.yaml \
  --data data/tusimple_gcs_fixed_y_k56_960x544.yaml \
  --imgsz 544 960 \
  --name <new-k56-fifthness-closed-loop-ft8-or-ft12-name> \
  --pretrained runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/weights/official_best.pt \
  --epochs 8 \
  --batch 32 \
  --workers 4 \
  --seed 1 \
  --lr0 <explicit-lr0> \
  --lrf <explicit-lrf> \
  --gcs-fifthness <explicit-gain> \
  --gcs-fifthness-pairwise <explicit-gain> \
  --gcs-quality-pairwise <explicit-gain> \
  --gcs-count-cumulative <explicit-gain> \
  --gcs-use-fifthness-decode \
  --gcs-fifthness-decode-thr <explicit-threshold> \
  --gcs-fifthness-decode-rank-weight <explicit-weight> \
  --gcs-official-best \
  --gcs-official-best-period 1 \
  --gcs-official-best-top-k 5 \
  --gcs-official-best-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-best-archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset
```

Count Head GT4-vs-GT5 low-margin short-gate template, only after the false-fifth case audit supports testing the existing adjacent-margin mechanism:

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml \
  --data data/tusimple_gcs_fixed_y_k56_960x544.yaml \
  --imgsz 544 960 \
  --name gcs_yolo_lane_s_q12_k56_countadj_lowmargin_ft8_seed1_b32w4 \
  --pretrained runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/weights/official_best.pt \
  --epochs 8 \
  --batch 32 \
  --workers 4 \
  --seed 1 \
  --lr0 0.00005 \
  --lrf 0.2 \
  --gcs-count-adjacent-margin 0.2 \
  --gcs-count-adjacent-margin-gain 0.05 \
  --gcs-count-adjacent-margin-gt45-weight 1.0 \
  --gcs-official-best \
  --gcs-official-best-period 1 \
  --gcs-official-best-top-k 5 \
  --gcs-official-best-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-best-archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset
```

This is not a full/e180 recipe. It tests one hypothesis only: whether a small adjacent Count Head margin can reduce GT4 false-fifth `P5/margin` without increasing FP or GT5 `5->4`. Do not mix in new fifthness, cumulative, or Quality changes in the same gate.

Result: rejected after epoch 4. Training ran cleanly with no NaN/shape/CUDA error, but independent official-val sweep reproduced best `official_acc=0.958843`, `FP=0.044399`, `FN=0.029155`, `count_acc_4=0.848485`, `count_acc_5=0.905405`, `rate_4_to_5=0.090909`, and `rate_5_to_4=0.094595`. This is below the K56 parent `0.959315` and worsens parent `rate_4_to_5=0.075758`, so do not continue this run to epoch 8 or full/e180. GT5 diagnosis kept 67/74 and attributed drops to `count_head_under_predict=4` and `quality_too_low=3`, with candidate-pool shortfall, valid-points failure, and GT5 NMS suppression at zero.

Removed Count Head false-fifth suppression gate:

`gcs_yolo_lane_s_q12_k56_countff_supp_ft8_seed1_b32w4` is rejected after the planned 8 epochs, and its source-side `--gcs-count-false-fifth-suppression` / `--gcs-count-false-fifth-margin` CLI switches have been removed. Training ran cleanly and retained Top-K official-val checkpoints. Official_best epoch 7 reached `official_acc=0.959496`, but FP worsened to `0.047062`, FN to `0.031680`, GT5 `rate_5_to_4` to `0.175676`, and `rate_4_to_5=0.075758` only matched the K56 parent. The more balanced epoch 4 row had `official_acc=0.959372`, `FP=0.044444`, `rate_4_to_5=0.060606`, and `rate_5_to_4=0.135135`, but its ACC margin was only `+0.000057` and FN was worse than the parent. Independent official-val reproduced epoch 7; GT5 diagnosis kept 61/74 and attributed drops to `count_head_under_predict=8` and `quality_too_low=5`. Do not rerun this exact gate or start full/e180 from it.

Command retained for reproducibility only:

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-fifthness-v1.yaml \
  --data data/tusimple_gcs_fixed_y_k56_960x544.yaml \
  --imgsz 544 960 \
  --name gcs_yolo_lane_s_q12_k56_fifthness_gt5neg_ft8_seed1_b32w4 \
  --pretrained runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/weights/official_best.pt \
  --epochs 8 \
  --batch 32 \
  --workers 4 \
  --seed 1 \
  --lr0 0.00005 \
  --lrf 0.2 \
  --gcs-fifthness 0.20 \
  --gcs-fifthness-pairwise 0.20 \
  --gcs-fifthness-margin 0.20 \
  --gcs-fifthness-negative-topk 2 \
  --gcs-fifthness-negative-score-thr 0.10 \
  --gcs-fifthness-include-gt5-negatives True \
  --gcs-quality-pairwise 0.15 \
  --gcs-quality-pairwise-margin 0.20 \
  --gcs-count-cumulative 0.03 \
  --gcs-count-cumulative-label-smoothing 0.00 \
  --gcs-official-best \
  --gcs-official-best-period 1 \
  --gcs-official-best-top-k 5 \
  --gcs-official-best-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-best-archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset
```

Rejected K56 Count/Quality gates from the epoch152 parent:

```text
run: gcs_yolo_lane_s_q12_k56_cqcalib_ft12_seed1_b32w4
commit: 655c116
status: stopped early after epoch 5 because official-val regressed
recipe: lr0=0.0005 default, count_cls_w3/w4/w5=1.30/1.50/1.95, count_boundary_gt5_pos_weight=1.20, quality_neg_weight=0.60, quality_hard_negative_from_head=True, hard_negative_visible_segment=True, candidate_gt5_edge_weight=1.15
best official-val: epoch 2, official_acc=0.953415, FP=0.052479, FN=0.042011
diagnosis: count_acc_5=0.689189, gt5_output5_rate=0.689189, gt5_count_head_under_rate=0.270270
decision: not promotable; do not rerun this aggressive recipe

run: gcs_yolo_lane_s_q12_k56_cqcalib_lr1e4_ft8_seed1_b32w4
commit: 655c116
status: stopped early after epoch 4 because official-val stayed below parent
recipe: lr0=0.0001, lrf=0.2, count_cls_w5=1.90, count_boundary_gt5_pos_weight=1.20, quality_neg_weight=0.55, quality_hard_negative_from_head=True, candidate_gt5_edge_weight=1.15
best official-val: epoch 1, official_acc=0.957787, FP=0.046465, FN=0.033976
diagnosis: count_acc_5=0.851351, gt5_output5_rate=0.851351, gt5_count_head_under_rate=0.108108
decision: not promotable; simple Count/Quality fine-tuning from the K56 parent is too destructive without a more surgical change

run: gcs_yolo_lane_s_q12_k56_lowfp_joint_ft8_seed1_b32w4
remote audit HEAD: bcc15c650169
status: completed 8-epoch low-FP-priority fine-tune, rejected after official-val diagnosis
recipe: lr0=0.00005, lrf=0.2, count_cls_w3/w4/w5=1.25/1.55/1.80, count_boundary_gt5_pos_weight=1.10, quality_neg_weight=0.55, duplicate-negative weights=1.75, no quality_hard_negative_from_head, no forced GT5 rescue
best official-val: epoch 1, official_acc=0.958999, FP=0.046970, FN=0.031910
delta vs K56 parent official_best 0.959315: official_acc=-0.000316, FP=+0.001745, FN=+0.003444
post-sweep best: 0.958999
GT5 diagnosis: kept=61/74, count_head_under_predict=8, quality_too_low=5, candidate_pool_shortfall=0, GT5 NMS suppression=0, rank5_score_low=0, valid_points_fail=0
decision: not promotable; stop this low-FP direction and do not continue with GT5 rescue
```

The K56 official-val evidence is a completed baseline result, not a mainline promotion and not a reason to use test or tune postprocess settings. The next useful K56 step should preserve the epoch152 FP/FN balance while reducing Count/Quality confusion around GT5 `5->4` drops and false fifth-lane pressure. The default-off `gcs_geometry_curvature` auxiliary loss and the low-FP joint fine-tune have already been tested and rejected as promotion candidates.

User-requested diagnostic-only K56 official-test audit:

| Row | Val selection source | Test Accuracy | FP | FN | official_score |
| --- | --- | ---: | ---: | ---: | ---: |
| parent default | parent official_best `0.959315` | `0.959429` | `0.033459` | `0.033609` | `0.958088` |
| parent minpoints `pv040 final9 fifth4` | val-only min-points row `0.959750` | `0.959131` | `0.032860` | `0.033968` | `0.957795` |
| `cqcalib_ft12` | rejected official-val `0.953415` | `0.953748` | `0.038048` | `0.044213` | `0.952102` |
| `cqcalib_lr1e4_ft8` | rejected official-val `0.957787` | `0.958869` | `0.033615` | `0.036395` | `0.957469` |
| `curveaux_ft8` | rejected official-val `0.958732` | `0.959706` | `0.034669` | `0.034268` | `0.958327` |
| `lowfp_joint_ft8` | rejected official-val `0.958999` | `0.959401` | `0.032758` | `0.034747` | `0.958051` |

These test rows were run only because the user explicitly requested them after the validation diagnostics. They are diagnostic-only and must not be used for checkpoint, threshold, min-points, postprocess, loss, or model promotion choices. In particular, `curveaux_ft8` has the highest test Accuracy in this table but remains rejected because its official-val `0.958732` is below the K56 parent `0.959315`.

Rejected K56 curvature auxiliary gate from the epoch152 parent:

```text
run: gcs_yolo_lane_s_q12_k56_curveaux_ft8_seed1_b32w4
commit: 82e832904637980ac69d890d63269bd9310296ef
status: stopped after epoch 4 because official-val stayed below parent
recipe: lr0=0.0001, lrf=0.2, gcs_geometry_curvature=0.05, gcs_geometry_curvature_beta_px=5.0
official-val by epoch: 1=0.957947, 2=0.956361, 3=0.958569, 4=0.958732
best official-val: epoch 4, official_acc=0.958732, FP=0.048439, FN=0.030992
parent reference: K56 epoch152 official_best official_acc=0.959315
delta vs parent: -0.000583
independent official-val sweep: runs/gcs_lane/gcs_yolo_lane_s_q12_k56_curveaux_ft8_seed1_b32w4/analysis_official_best_val_sweep/tusimple_official_sweep_summary.json, best reproduces epoch4 official_acc=0.958732 at conf=0.005, point_valid_thr=0.30, nms_dist_px=18.0, max_det=5, min_points=6, rank_min_points=none
GT5 diagnosis: runs/gcs_lane/gcs_yolo_lane_s_q12_k56_curveaux_ft8_seed1_b32w4/analysis_official_best_gt5_diag_val/gt5_rank_diagnostics_summary.json, kept=61/74, count_head_under_predict=9, quality_too_low=2, unknown_shortfall=2, candidate_pool_shortfall=0, GT5 NMS suppression=0, rank5_score_low=0, valid_points_fail=0
audit: results.csv has 4 rows and no numeric NaN/Inf; log/artifact scan found no Traceback, RuntimeError, shape error, shape mismatch, CUDA OOM, or test-split leakage keywords
decision: not promotable; do not rerun this exact curvature recipe as the next path
```

Command retained for reproducibility only:

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml \
  --data data/tusimple_gcs_fixed_y_k56_960x544.yaml \
  --imgsz 544 960 \
  --name gcs_yolo_lane_s_q12_k56_curveaux_ft8_seed1_b32w4 \
  --pretrained runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/weights/official_best.pt \
  --epochs 8 \
  --batch 32 \
  --workers 4 \
  --seed 1 \
  --lr0 0.0001 \
  --lrf 0.2 \
  --gcs-geometry-curvature 0.05 \
  --gcs-geometry-curvature-beta-px 5.0 \
  --gcs-official-best \
  --gcs-official-best-period 1 \
  --gcs-official-best-top-k 5 \
  --gcs-official-best-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-best-archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset
```

The 2026-06-13 official-val gates after the Count Head visible-segment evidence change are not promotable:

```text
run: gcs_yolo_lane_s_q12_cb_gt45_ft6_countvis_clean_seed1_b8w0
commit: ec9cf5f47
best official-val: 0.953415
reference countboundary baseline: 0.954137
reference old FT6: 0.954782

run: gcs_yolo_lane_s_q12_jointcount_adjmargin_countvis_ft12_seed1_b8w0
commit: 632634eb6
independent official-val: 0.953113
reference countboundary baseline: 0.954137
reference old FT6: 0.954782
reference clean count-visible FT6: 0.953415

run: gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0
commit: 4881bcebc
best epoch: 5
independent official-val: 0.953639
official FP/FN: 0.044674 / 0.036272
reference countboundary baseline: 0.954137
reference old FT6: 0.954782
reference clean count-visible FT6: 0.953415
reference adjacent margin gate: 0.953113
```

Remote audit artifacts for these rejected runs should stay tied to their run directories:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_countvis_clean_seed1_b8w0/args.yaml
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_countvis_clean_seed1_b8w0/results.csv
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_countvis_clean_seed1_b8w0/weights/official_best_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_countvis_clean_seed1_b8w0/<official-val-sweep-summary>
runs/gcs_lane/gcs_yolo_lane_s_q12_cb_gt45_ft6_countvis_clean_seed1_b8w0/<gt5-diagnostic-output>
runs/gcs_lane/gcs_yolo_lane_s_q12_jointcount_adjmargin_countvis_ft12_seed1_b8w0/args.yaml
runs/gcs_lane/gcs_yolo_lane_s_q12_jointcount_adjmargin_countvis_ft12_seed1_b8w0/results.csv
runs/gcs_lane/gcs_yolo_lane_s_q12_jointcount_adjmargin_countvis_ft12_seed1_b8w0/weights/official_best_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_jointcount_adjmargin_countvis_ft12_seed1_b8w0/<official-val-sweep-summary>
runs/gcs_lane/gcs_yolo_lane_s_q12_jointcount_adjmargin_countvis_ft12_seed1_b8w0/<gt5-diagnostic-output>
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/args.yaml
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/results.csv
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/official_best_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/weights/official_best.pt
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/weights/official_topk/
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/analysis_official_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_gt5segq_vishn_countvis_ft12_seed1_b8w0/analysis_official_best_gt5_diag_val/gt5_rank_diagnostics_summary.json
```

Do not rerun these rejected gates unless checking reproducibility. Do not use test to choose the next candidate.

Current-code audit baseline before the rejected GT5 edge Quality floor gate:

```text
weights: runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/weights/official_best.pt
official-val sweep: runs/gcs_lane/reliability_audit_20260613_baseline_current_default_val_sweep
official_acc: 0.953756
FP/FN: 0.046006 / 0.036961
GT5 diagnosis: runs/gcs_lane/reliability_audit_20260613_baseline_current_default_gt5_diag
GT5 kept: 49/74
GT5 failure counts: quality_too_low=14, count_head_under_predict=7, valid_points_fail=3, candidate_pool_shortfall=1
```

Rejected gate command, kept for reproducibility only:

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --imgsz 544 960 \
  --name gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0 \
  --pretrained runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/weights/official_best.pt \
  --epochs 12 \
  --batch 8 \
  --workers 0 \
  --seed 1 \
  --gcs-quality-gt5-edge-floor 0.65 \
  --gcs-official-best \
  --gcs-official-best-period 1 \
  --gcs-official-best-top-k 5 \
  --gcs-official-best-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-best-archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset
```

Result:

```text
run: gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0
commit: 7adbf03a6
knob: gcs_quality_gt5_edge_floor=0.65
training-time official_best epoch: 12
independent official-val: 0.953587
official FP/FN: 0.048990 / 0.035583
gt5_output5_rate: 0.716216
gt5_count_head_under_rate: 0.027027
gt5_valid_points_fail_rate: 0.256757
matched/unmatched quality mean: 0.857180 / 0.703256
GT5 diagnosis: kept=53/74, quality_too_low=16, count_head_under_predict=2, valid_points_fail=2, candidate_pool_shortfall=1
decision: not promotable; keep default-off
```

Remote/local audit artifacts:

```text
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/args.yaml
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/results.csv
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/official_best_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/weights/official_best.pt
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/weights/official_topk/
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/analysis_official_best_val_sweep/tusimple_official_sweep_summary.json
runs/gcs_lane/gcs_yolo_lane_s_q12_quality_gt5edgefloor_ft12_seed1_b8w0/analysis_official_best_gt5_diag_val/gt5_rank_diagnostics_summary.json
```

Do not rerun this exact `0.65` floor gate unless checking reproducibility. Do not use test to choose the next candidate.

For every run, fetch `args.yaml`, `results.csv`, `weights/official_best_summary.json`, retained `weights/official_topk/` metadata, independent official-val sweep summaries, and GT5 diagnostics. The minimum analysis fields are `official_acc`, FP, FN, `count_acc_3/4/5`, GT3/GT4/GT5 confusion rates, `gt5_output5_rate`, `gt5_count_head_under_rate`, `gt5_valid_points_fail_rate`, candidate shortfall, GT5 NMS, `decode/k5_to_output4_rate`, rank-score failure counts, visible valid-point distributions, and matched/unmatched quality means.

## Inference

```bash
python tools/infer_gcs.py \
  --weights <weights.pt> \
  --source <images-or-list> \
  --imgsz 544 960
```

## Custom GCS Evaluation

```bash
python tools/eval_gcs.py \
  --weights <weights.pt> \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --split val \
  --imgsz 544 960
```

## TuSimple Official Final Test Evaluation

```bash
python tools/eval_tusimple_official.py \
  --weights <weights.pt> \
  --archive-root archive \
  --split test \
  --selection-summary <official-val-selection-summary.json> \
  --imgsz 544 960
```

Use this only once for the final checkpoint and postprocess configuration selected on official-val. Do not iterate on its result. `--selection-summary` must point to the official-val selection artifact that chose the checkpoint and postprocess row, such as `runs/gcs_lane/<run>/official_best_summary.json` or `runs/gcs_lane/<run>/analysis_official_best_val_sweep/tusimple_official_sweep_summary.json`. The command rejects final-test parameters that do not match the selected official-val row.

For an explicitly user-requested test audit that is not promotable final evidence, mark it permanently as diagnostic-only:

```bash
python tools/eval_tusimple_official.py \
  --weights <weights.pt> \
  --archive-root archive \
  --split test \
  --diagnostic-only-test \
  --diagnostic-reason "user-requested audit; not for checkpoint/threshold/model selection" \
  --imgsz 544 960
```

Diagnostic-only test output must not be used for checkpoint, threshold, postprocess, loss, model, or promotion decisions.

The command needs original TuSimple test archive files, not only the fixed-y converted dataset. The minimum archive shape is:

```text
archive/TUSimple/test_label.json
archive/TUSimple/test_set/clips/<date>/<clip>/<frame>.jpg
archive/TUSimple/train_set/
```

`train_set/` may be an empty placeholder for test-only final evaluation, but `find_tusimple_archive_root()` requires both `train_set` and `test_set` directories to exist.

To prepare a minimal test-only archive from a full local TuSimple archive, copy only the 2,782 frames referenced by `archive/TUSimple/test_label.json` while preserving `raw_file` paths. Do not use this test archive for checkpoint, threshold, postprocess, or rescue selection.

Remote verification after extraction:

```bash
python - <<'PY'
from gcs_tools.tusimple_official_eval import (
    default_tusimple_gt_json,
    find_tusimple_archive_root,
    read_tusimple_json_lines,
    tusimple_image_path,
)

root = find_tusimple_archive_root("archive")
gt_path = default_tusimple_gt_json(root, split="test")
records = read_tusimple_json_lines(gt_path)
missing = []
for record in records:
    try:
        tusimple_image_path(root, record["raw_file"], split="test")
    except FileNotFoundError:
        missing.append(record["raw_file"])

print("archive_root", root)
print("gt_json", gt_path)
print("records", len(records))
print("missing", len(missing))
PY
```

## TuSimple Official-Val Sweep

```bash
python tools/sweep_tusimple_official.py \
  --weights <weights.pt> \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset \
  --imgsz 544 960
```

`tools/sweep_tusimple_official.py` defaults to validation, rejects `--split test`, and rejects explicit GT records that resolve to TuSimple `test_set` images. For official-val selection, pass the 363-image official-val `--gt-json` and matching `--archive-root` explicitly.

For the K56 official-h-sample branch, retune min-points on official-val because `candidate_min_points`, `final_min_points`, and `fifth_min_points` have different semantics with 56 anchors than they had with K32. Start with:

```bash
python tools/sweep_tusimple_official.py \
  --weights runs/gcs_lane/gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4/weights/official_best.pt \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset \
  --imgsz 544 960 \
  --confs 0.005 \
  --point-valid-thrs 0.35 \
  --nms-dist-pxs 18 \
  --max-dets 5 \
  --candidate-min-points 5 6 7 \
  --final-min-points 6 7 8 9 \
  --fifth-min-points 4 5 6 7
```

Then resweep the best rows at `point_valid_thr=0.30/0.35/0.40`. Track joint GT3/GT4/GT5 metrics, especially official ACC, FP/FN, `count_acc_3/4/5`, `rate_4_to_5`, `rate_5_to_4`, `gt5_output5_rate`, Count Head underprediction, valid-points failure, candidate shortfall, GT5 NMS, and matched/unmatched quality means.

## GT5 Official-Val Diagnosis

```bash
python tools/diagnose_gcs_gt5.py \
  --weights <weights.pt> \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset \
  --imgsz 544 960
```

Use this to separate Count Head underprediction from candidate-pool shortfall, valid-points failure, NMS suppression, rank-score failure, quality-gate failure, and final-output shortfall. The tool defaults to `--split val`, rejects `--split test`, and rejects explicit GT records that resolve to TuSimple `test_set` images; do not use test for diagnosis or tuning.

## Contract Checks

```bash
python scripts/verify_loss_cleanup.py
python tools/check_gcs_count_head_topk_contract.py
python tools/check_gcs_decode_meta_contract.py
python tools/check_gcs_algorithm_contract.py
```

## Model Shape Check

```bash
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml --imgsz 544 960
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-fifthness-v1.yaml --imgsz 544 960 --batch 1
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count5ev-v1.yaml --imgsz 544 960 --batch 1
```

## Head Dependency Check

```bash
python tools/check_gcs_head_dependency.py --weights <weights.pt> --imgsz 544 960
```

## Dataset Checks

```bash
python tools/check_dataset.py
python tools/check_gcs_label_order_split.py
```

## Agent Setup Check

```bash
python scripts/check_gcs_agent_setup.py
```

## Python Compile Check

For changed Python files:

```bash
python -m py_compile <changed-python-files>
```

## GitHub Sync

After any code change is implemented and the relevant local validation has passed, sync the published source to:

```text
https://github.com/Xzl0104/GCS-YOLO-Lane_LSA
```

The published repository should include root project instructions plus these project folders:

```text
AGENTS.md
README.md
data
gcs_tools
scripts
tests
tools
ultralytics
docs
```

Do not publish local training outputs, checkpoints, datasets, zip archives, caches, or Python bytecode.

Keep rollback possible:

- sync each validated code change as a normal Git commit
- do not use force-push or history rewrite for the published repository
- keep commit messages specific enough to identify the algorithm change
- roll back a bad change with `git revert <commit>` or by checking out an earlier commit SHA

Archive notes and summaries:

- treat each Git commit/push sync as a project archive point
- write a concise commit note based on the completed work, not a generic message
- include important validation results or remaining risk in the commit body when useful
- after every archive push, report a sync summary to the user with changed files, validation performed, commit SHA, GitHub push status, and any remaining unsynced or ignored local files that matter to the requested work

PR handoff:

- PR creation is separate from Git sync; do not claim a PR was opened just because a branch was pushed.
- If `gh` is unavailable or unauthenticated, report PR creation as blocked.
- When PR creation is blocked, include the branch name, pushed commit SHA, intended base branch if known, and a GitHub manual PR URL such as `https://github.com/Xzl0104/GCS-YOLO-Lane_LSA/pull/new/<branch>`.

## Remote Server Experiment Loop

Use this loop when local Codex changes need to be trained or evaluated on a remote CUDA server.

Do not commit private SSH hosts, usernames, ports, keys, or server-local absolute paths to the published repository. Keep those values as operator/session parameters.

Recommended order:

```text
1. implement the local code/config/doc change
2. run targeted local validation
3. commit and push the validated published source to GitHub
4. SSH to the remote server, preferably with `ssh gcs-ebcloud-lane`
5. update the remote Git clone with `git pull --ff-only` or checkout the exact pushed commit SHA
6. activate the remote `ssh_lane` conda environment
7. run the training/evaluation command from the remote repository root
8. keep TuSimple commands on `--imgsz 544 960`
9. run official-val sweep and diagnostics on validation only
10. fetch back run summaries, CSV/JSON metrics, logs, and diagnostic outputs for local analysis
```

If the server already has a non-Git project copy containing datasets, runs, or checkpoints, do not make `git pull` operate inside that directory and do not overwrite it blindly. Create or reuse a dedicated Git clone for the published source, then link or copy only the required local runtime artifacts such as `datasets/`, `archive/`, and pretrained weights.

Prefer fetching lightweight analysis artifacts first:

```text
runs/gcs_lane/<run>/args.yaml
runs/gcs_lane/<run>/results.csv
runs/gcs_lane/<run>/weights/official_best_summary.json
official-val sweep output files
GT5 diagnostic output files
```

Fetch checkpoints such as `official_best.pt`, `last.pt`, or `weights/official_topk/` only when local inference, re-sweep, or archival review needs them.

Analysis rules:

- choose checkpoints, thresholds, and postprocess settings from official-val only
- do not use test for iteration or parameter search
- do not use `tools/sweep_gcs_conf.py --run-test` in the research loop
- do not report `tools/eval_tusimple_official.py --pred-json` results as model evidence unless the prediction file is tied to the exact generation command, commit SHA, weights, and official-val selection record
- separate ordinary validation logs, official-val results, diagnostics, and final test evidence in summaries
- report the pushed commit SHA, remote run path, command, validation artifacts, and remaining risks after each remote experiment

## Experiment Candidate Validation Order

```text
1. py_compile changed files
2. relevant contract checks
3. model shape check if model changed
4. dataset checks if data changed
5. official-val sweep
6. official-val diagnostic analysis
7. decision-log update
8. one-shot final test with `tools/eval_tusimple_official.py --split test` only after candidate selection
```

Do not use full test to search thresholds, rescue parameters, count-policy parameters, ranking parameters, or checkpoint choices.
