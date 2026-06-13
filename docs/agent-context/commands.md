# Commands

All TuSimple commands should use:

```bash
--imgsz 544 960
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

## GT5 Segment-Quality Candidate Training

Use this controlled candidate after the 2026-06-13 `gcs_yolo_lane_s_q12_cb_gt45_ft8_visrank_qhard_seed1_b8w0` analysis. It targets GT5 fifth-lane valid support and Quality Head false-positive separation. Select checkpoints and thresholds on official-val only.

```bash
python tools/train_gcs.py \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --imgsz 544 960 \
  --pretrained runs/gcs_lane/gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1/weights/official_best.pt \
  --epochs 10 \
  --batch 8 \
  --workers 0 \
  --lr0 8e-5 \
  --lrf 0.2 \
  --cos_lr true \
  --gcs-quality-hard-negative-from-head \
  --gcs-quality 0.6 \
  --gcs-quality-neg-weight 0.8 \
  --gcs-quality-hard-negative-weight 3.0 \
  --gcs-quality-duplicate-negative-weight 4.0 \
  --gcs-point-valid-gt5-pos-weight 2.5 \
  --gcs-candidate-gt5-edge-weight 1.25 \
  --gcs-point-valid-gt5-edge-continuity 0.10 \
  --gcs-point-valid-gt5-edge-continuity-thr 0.65 \
  --gcs-point-valid-gt5-edge-segment 0.10 \
  --gcs-point-valid-gt5-edge-segment-thr 0.65 \
  --gcs-point-valid-gt5-edge-segment-min-points 5 \
  --gcs-hard-edge-loss-terms exist,point,point_valid,line_iou,quality \
  --gcs-official-best \
  --gcs-official-best-period 1 \
  --gcs-official-best-top-k 5 \
  --gcs-official-best-gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --gcs-official-best-archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset
```

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
  --split test \
  --imgsz 544 960
```

Use this only once for the final checkpoint and postprocess configuration selected on official-val. Do not iterate on its result.

## TuSimple Official-Val Sweep

```bash
python tools/sweep_tusimple_official.py \
  --weights <weights.pt> \
  --split val \
  --imgsz 544 960
```

`tools/sweep_tusimple_official.py` defaults to validation and rejects `--split test`.

## GT5 Official-Val Diagnosis

```bash
python tools/diagnose_gcs_gt5.py \
  --weights <weights.pt> \
  --split val \
  --imgsz 544 960
```

Use this to separate Count Head underprediction from candidate-pool shortfall, valid-points failure, NMS suppression, rank-score failure, quality-gate failure, and final-output shortfall. The tool defaults to `--split val` and rejects `--split test`; do not use test for diagnosis or tuning.

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

The published repository should contain only these project folders:

```text
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
- after the archive is pushed, report a work summary to the user with changed files, validation, commit SHA, and GitHub sync status

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
