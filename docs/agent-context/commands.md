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

## Next Remote Official-Val Experiments

No next remote training command is selected at this checkpoint. The matched GT5 edge Quality target floor gate has completed and is not promotable. Pause before launching another run; the next candidate needs a new hypothesis for preserving GT5 real-candidate quality without increasing GT4-to-5 false lanes.

When a new remote CUDA experiment is selected, run it from a dedicated Git clone checked out to the exact pushed commit SHA. Do not run training locally from Codex. Activate the remote CUDA environment first:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ssh_lane
```

The recent official-val gates after the Count Head visible-segment evidence change are not promotable:

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
  --imgsz 544 960
```

Use this only once for the final checkpoint and postprocess configuration selected on official-val. Do not iterate on its result.

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
