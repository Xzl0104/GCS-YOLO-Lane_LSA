# Decision Log

This file records decisions for branch `codex/5-25-3-k56`.

## 2026-06-19: Import 5-25-3 as a Separate K56 Branch

Decision:

Import the historical `5-25-3.zip` GCS-YOLO-Lane algorithm as branch `codex/5-25-3-k56` and adapt only the explicit TuSimple fixed-y contract:

```text
legacy archive: Q=8, K=32, fixed_y=[0.98, 0.25]
this branch:    Q=12, K=56, fixed_y=[710/720, 160/720]
```

Use:

```text
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
data  = data/tusimple_gcs_fixed_y_k56_960x544.yaml
imgsz = 544 960
```

Why:

The user requested a separate branch for the `5-25-3.zip` algorithm while preserving the current K56/710 TuSimple contract.

Alternatives considered:

- Port later mainline Count/Quality/Survival/near-miss mechanisms into the branch.
- Keep the legacy Q8/K32 contract.
- Resample old K32 labels to K56.

Tradeoff:

The branch is intentionally narrower than current mainline. It preserves 5-25-3 algorithm behavior and does not include later mainline Count/Quality/Boundary/Survival machinery or training-time official-best checkpoint preservation. K56 labels must be regenerated from original TuSimple JSON and images.

Validation evidence:

Local checks performed during branch setup:

```text
python -m py_compile <changed GCS Python files>
python tools/check_model.py --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml --imgsz 544 960 --batch 1 --device cpu
YAML contract assertions for Q=12, K=56, fixed_y_start=710/720, fixed_y_end=160/720
fixed-y anchor assertion for 56 y pixels: 710,700,...,160
sample label split/order check against an external K56 dataset root
```

Mainline or experiment at the time:

Separate historical algorithm branch. This status was superseded by the 2026-06-20 mainline promotion below.

## 2026-06-20: Promote 5-25-3 K56 as Current Mainline

Decision:

Treat branch `codex/5-25-3-k56` as the current mainline for new K56 TuSimple work. Use the mainline aliases:

```text
model = ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml
data  = data/tusimple_gcs_fixed_y_960x544.yaml
imgsz = 544 960
```

The q12-k56-named config/data paths remain compatibility references for old experiment records:

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
data/tusimple_gcs_fixed_y_k56_960x544.yaml
```

Why:

The user clarified that `5-25-3-k56` and the previous q12-k56 line are different algorithms, and the new 5-25-3 branch should be the active line going forward.

Preservation rule:

Do not delete previous q12-k56 experiment documentation. Keep it as historical experiment context, and do not let it override `docs/agent-context/current-contracts.md`.

## 2026-06-20: Keep Agent Workflow Local to Codex

Decision:

Document project Agent/Skill workflow rules for Codex collaboration while keeping the server-side branch payload focused on algorithm/runtime code.

Why:

The remote training server only needs code required for training, evaluation, diagnostics, model/config contracts, and data conversion. Agent setup files and wrapper scripts are local Codex workspace concerns and must not imply that later mainline Count/Quality/Boundary/Survival algorithm mechanisms are active.

Policy:

Permit assistant-originated runtime delegation through `multi_agent_v1.spawn_agent` only when the user explicitly asks for multi-Agent, delegation, subagents, or parallel agent work. If the tool is absent from the initial surface, the assistant must call `tool_search` with query `multi_agent_v1 spawn_agent` before declaring that the current API/tool surface does not expose runtime multi-Agent delegation. The assistant must not simulate Agent roles or describe local work as delegated output. Skill loading remains separate from delegation.

Close completed runtime agents after their results are integrated. Do not describe local Skill loading as delegated Agent output.

Validation evidence:

No server-side agent setup check is required for this branch.

Mainline or experiment:

Branch collaboration policy, not an algorithm promotion.

## 2026-06-20: Freeze count03_under5_03 Official-Val Selected Candidate

Decision:

Freeze `gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03` as the current evaluated candidate for this branch using `weights/best.pt` and the official-val selected decode:

```text
conf = 0.05
point_valid_thr = 0.5
nms_dist_px = 50.0
max_det = 8
min_points = 5
```

Official-val evidence:

```text
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_official_val363_sweep
grid = 1800 combinations over conf, point_valid_thr, nms_dist_px, max_det, and min_points
images = 363
official_acc = 0.969976
official_FP = 0.019559
official_FN = 0.014463
official_score = 0.969296
count_acc = 0.969697
```

Final test reporting evidence:

```text
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_official_test_best_from_val
images = 2782
official_acc = 0.965459
official_FP = 0.029439
official_FN = 0.026270
official_score = 0.964345
count_acc = 0.872753
avg_total_ms = 14.8576
```

Why:

The decode was selected on official-val, which satisfies the branch research-integrity rule for threshold and postprocess selection. The final test run used that selected configuration once and is reporting evidence only.

Rejected alternative:

The top `official_score` row was not promoted. It had a tiny score gain of about `0.000032`, but its official ACC was lower by about `0.000036`; official ACC remains the primary branch comparison metric.

Tradeoff:

This is an experiment result freeze, not an algorithm change. The branch still does not include training-time official-best checkpoint preservation, so `best.pt` provenance should be described as the source run checkpoint rather than an official-val checkpoint-selected artifact.

Remaining bottleneck:

The test count breakdown is diagnostic-only and must not drive tuning. It shows the largest weakness on 4-lane scenes:

```text
count_acc_4 = 0.542735
4->3 = 113
4->5 = 96
4->6 = 5
```

Mainline or experiment:

Experimental candidate accepted as the current reported branch result. Future changes must use official-val only for selection and keep test closed.

## 2026-06-20: Diagnose Count Confusion by Date and Short Visible Lanes

Decision:

Treat the `count03_under5_03` count weakness as a train/val reproducible `GT4` short-side-lane robustness problem, not as a test-threshold tuning problem.

Remote diagnostic evidence:

```text
directory = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03_count_confusion_train_val_by_visibility
summary   = summary.json
weights   = runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_visible_iou_count03_under5_03/weights/best.pt
decode    = conf=0.05, point_valid_thr=0.5, nms_dist_px=50.0, max_det=8, min_points=5
splits    = fixed-y train and val only
```

Key findings:

```text
val count_acc = 0.939394
val GT4 count_acc = 0.941667
val 0313-2|GT4|min_visible<=10: n=6, count_acc=0.166667
val 0601|GT4|min_visible=11..20: n=6, count_acc=0.333333

train count_acc = 0.963837
train GT4 count_acc = 0.959202
train 0313-2|GT4|min_visible<=10: n=70, count_acc=0.657143
train 0601|GT4|min_visible<=10: n=19, count_acc=0.368421
train 0601|GT4|min_visible=11..20: n=47, count_acc=0.680851
```

Why:

The same failure mode appears without using final test for selection: `GT4` scenes with short visible side lanes are unstable and show both undercount and overcount. `GT5` is comparatively stronger on the fixed-y train split, so a dense-lane-only undercount change is too narrow.

Rejected actions:

- Do not sweep or select thresholds on final test.
- Do not promote a new candidate from this diagnostic alone.
- Do not silently import later mainline Count/Quality/Survival machinery into this branch.

Recommended next action:

First make the `(date, lane_count, min_visible_points)` count-confusion diagnostic reusable on train/val. Then run the smallest train-side experiment that either upweights `GT4` short-side-lane samples or adds a GT4-aware count margin to the existing query-logit count objective. Select any follow-up candidate only on official-val.

Mainline or experiment:

Diagnostic-only branch evidence. No algorithm promotion and no test-tuned parameter change.

## 2026-06-20: Add Reusable Count Diagnostic and GT4 Short-Lane Sampler Boost

Decision:

Add a reusable train/val count-confusion diagnostic flow and an explicit train-time sampler option for `GT4` short visible side-lane samples.

Implementation:

```text
tools/diagnose_tusimple_count_confusion.py
  direct mode: --weights, --dataset-root, --splits train val
  grouping: date, GT lane count, shortest visible GT lane bucket

tools/train_gcs.py
ultralytics/models/yolo/gcs_lane/train.py
ultralytics/cfg/default.yaml
ultralytics/cfg/__init__.py
  new options:
    --gcs-gt4-short-boost
    --gcs-gt4-short-min-visible-max
```

Default behavior:

```text
gcs_gt4_short_boost = 1.0
gcs_gt4_short_min_visible_max = 10
```

Why:

The previous train/val diagnostic localized the count weakness to `GT4` scenes with short visible side lanes. A sampler multiplier is the smallest explicit train-side experiment that targets those samples without changing the model output contract, official metric, test protocol, or default branch behavior.

Recommended experiment:

```text
--gcs-gt4-short-boost 2.0
--gcs-gt4-short-min-visible-max 10
run name: gcs_yolo_lane_s_tusimple_fixed_y_gt4short2_count03_under5_03
```

Risk:

Oversampling short `GT4` cases may improve `4->3` undercount but could increase `4->5` overcount or reduce ordinary official-val ACC. Candidate selection must therefore remain official-val only, followed by train/val count-confusion diagnostics before any final-test reporting.

Mainline or experiment:

Branch-local experimental option. It does not import later mainline Count/Quality/Survival machinery and does not promote a new candidate.
