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

The current experimental K56 family/reference is a separate `Q12-K56` official-h-sample-aligned candidate:

```text
model: ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml
data:  data/tusimple_gcs_fixed_y_k56_960x544.yaml
root:  datasets/tusimple_fixed_y_k56_960x544
K:     56, fixed-y anchors aligned to TuSimple h_samples 710..160 step 10
```

K56 labels are regenerated from original TuSimple JSON and images, not resampled from K32 labels. The K56 official-val label oracle is `Accuracy=0.998256`, `FN=0.001377`, `FP=-0.000689` on the 363-image official-val split. The current mainline remains `K=32`; do not silently mix `K=56` labels with existing `K=32` data or checkpoints.

The raw official-val h-sample endpoint audit found 363 records, 1303 lanes, and 6 lanes with only 1-3 valid official h-samples. K32 had 20 zero-anchor lanes and 582 one-anchor lanes; K56 had 0 zero-anchor lanes, 3 one-anchor lanes, and no endpoint loss in the audit. The current generated K56 labels and official prediction conversion still keep only lanes with at least 2 valid anchors, so the 3 one-anchor official-val lanes remain a separate explicit experiment rather than an active training/default decode contract. Use `tools/analyze_tusimple_hsample_endpoints.py` for this raw-data audit and `tools/check_gcs_label_order_split.py --expect-fixed-y 56,710/720,160/720` for exact K56 artifact validation.

The completed formal K56 baseline ran on the remote RTX 4090 24GB server as:

```text
gcs_yolo_lane_s_q12_k56_offhs_e180_seed1_b32w4
batch=32
workers=4
```

The run completed on `2026-06-14` at 180/180 epochs with no NaN, shape error, traceback, or test-split leakage found in run artifacts. Independent official-val sweep of `weights/official_best.pt` reproduced the training-time selection: epoch 152, `official_acc=0.959315`, `FP=0.045225`, `FN=0.028466`, `official_score=0.957841`, using `conf=0.005`, `point_valid_thr=0.35`, `nms_dist_px=18.0`, `max_det=5`, `min_points=6`, and `rank_min_points=none`. This exceeds the current-code K32 audit `0.953756` by `+0.005559` and legacy `0.959224` by `+0.000091`, but it is not promoted because the margin is tiny and the 0.97 objective remains unmet. Final retained official Top-K is `152=0.959315`, `170=0.959247`, `166=0.959244`, `168=0.959217`, `165=0.959215`; ordinary val best remains epoch 142 with `val/f1=0.962083`.

Independent GT5 diagnosis on official-val found 63/74 GT5 images kept; remaining GT5 drops are `count_head_under_predict=5` and `quality_too_low=6`, with candidate-pool shortfall, GT5 NMS suppression, and rank-score-low all at zero. A K56 min-points official-val grid improved the best validation row to `0.959750` at `point_valid_thr=0.40`, `candidate_min_points=5`, `final_min_points=9`, and `fifth_min_points=4`, but this row is only a validation-selected postprocess candidate because `count_acc_4=0.863636` exposes GT4-to-5 false fifth-lane risk.

The K56 direct Count/Quality/low-FP fine-tune gates from the epoch152 parent are not promotable: `gcs_yolo_lane_s_q12_k56_cqcalib_ft12_seed1_b32w4` best `0.953415`, `gcs_yolo_lane_s_q12_k56_cqcalib_lr1e4_ft8_seed1_b32w4` best `0.957787`, and `gcs_yolo_lane_s_q12_k56_lowfp_joint_ft8_seed1_b32w4` best `0.958999`. Do not rerun those exact recipes as the next path, and do not continue them with GT5 rescue.

A default-off K56 fifth-candidate verifier candidate is implemented as `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-fifthness-v1.yaml`. It is opt-in only: the default K32/K56 models keep the existing six-output contract, while this YAML additionally emits `pred_fifthness_logits: B x Q` and enables Count Head fifth-candidate evidence. The associated losses and calibration terms (`gcs_fifthness*`, `gcs_quality_pairwise*`, `gcs_count_cumulative*`) default to `0.0`. Fifthness decode is also default-off: `gcs_use_fifthness_decode=False`, `gcs_fifthness_decode_thr=0.0`, and `gcs_fifthness_decode_rank_weight=1.0`.

A separate default-off isolation candidate is implemented as `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count5ev-v1.yaml`. It enables Count Head fifth-candidate evidence while keeping `pred_fifthness_logits` off, so it preserves the normal six-output contract. Use it to test Count Head fifth-candidate evidence separately from the fifthness verifier/head.

The first remote `count5ev-v1` FT8 attempt stopped after epoch 3/8 while the server root filesystem had only about `687M` free. Its partial official-val rows were below the K56 parent, so it is not a full/e180 candidate; rerun the short gate only after freeing server disk space.

The first short fifthness-v1 gate `gcs_yolo_lane_s_q12_k56_fifthness_v1_ft8_seed1_b32w4` is rejected: best official-val was `0.959006`, below the K56 parent `0.959315`, with worse FP/FN and high GT4-to-5 pressure. The follow-up `gcs_fifthness_include_gt5_negatives` switch is default-off and explicitly tests whether adding GT5 same-image unmatched outside negatives helps rank false fifth candidates below real GT5 edges. Its first gate `gcs_yolo_lane_s_q12_k56_fifthness_gt5neg_ft8_seed1_b32w4` is also not promotable: best official-val was `0.959319`, only `+0.000004` over parent, with worse FP and GT4-to-5 pressure.

The `pred_fifthness_logits` verifier is now wired into inference/evaluation only through the explicit fifthness decode switches. When enabled, it can gate or re-rank only the selected fifth lane and fifth-lane rescue candidates; ranks 1-4 keep the default visible-segment ranking. Enabling fifthness decode against a model that does not emit `pred_fifthness_logits` fails fast. This closes an implementation blind spot but is not official-val improvement evidence by itself, so do not start full/e180 from the rejected fifthness gates.

The first closed-loop official-val fifthness decode sweep of the `fifthness_gt5neg` checkpoint is also not promotable. Thresholds `0.30/0.50` reproduce the same `0.959319` ACC while keeping high FP and `rate_4_to_5=0.121212`; `0.70` drops below parent; `0.85` lowers false fifth pressure but drops ACC to `0.959023` and worsens GT5 `5->4`.

The follow-up official-val score audit at `runs/gcs_lane/gcs_yolo_lane_s_q12_k56_fifthness_gt5neg_ft8_seed1_b32w4/analysis_official_best_val_fifthness_score_audit/fifthness_score_audit_summary.json` shows why this is not a full/e180 candidate. GT5 selected-rank-5 matched samples have median fifthness `0.960218`, but GT4 false-fifth samples still reach `0.942222`. Thresholds `0.30/0.50` keep all true and false samples, `0.70` still keeps `87.5%` of false samples, and `0.85` starts suppressing false fifths while also cutting true GT5 fifth lanes. The audit CSV now also records Count Head `count_head_prob_4`, `count_head_prob_5`, and `count_head_margin`; the server rerun found GT4 false-fifth samples with median `P5=0.996325` and median margin `0.992650`, so the current failure is often Count Head overconfidence in count=5, not a pure fifthness threshold issue. Treat `tools/audit_gcs_fifthness_scores.py` as a val-only diagnostic, not a selection or test tool.

The follow-up Count Head adjacent-margin short gate `gcs_yolo_lane_s_q12_k56_countadj_lowmargin_ft8_seed1_b32w4` is rejected and should not be continued to full/e180. It was stopped after epoch 4: independent official-val sweep reproduced best `official_acc=0.958843`, `FP=0.044399`, `FN=0.029155`, below the K56 parent `0.959315`, and `rate_4_to_5=0.090909` worsened versus parent `0.075758`. GT5 `rate_5_to_4` improved to `0.094595`, but the joint objective requires reducing both GT5 `5->4` and GT4 false fifth pressure. The GT5 diagnosis still points to `count_head_under_predict=4` and `quality_too_low=3`, with candidate pool, valid-points, and NMS not dominant.

A narrower default-off Count Head candidate is now available for the next short gate: `gcs_count_false_fifth_suppression` with `gcs_count_false_fifth_margin`. It reuses the existing competitive unmatched outside-candidate mining and suppresses Count Head `count=5` only for GT3/GT4 images that contain a mined false fifth candidate. It stays inside `count_cls_loss`, does not add outputs or loss columns, does not require the fifthness head, and does not alter decode or official evaluation. This is intended to test the specific Count Head overconfidence found in the false-fifth audit, not to rerun the rejected broad adjacent-margin recipe.

A user-requested K56 official-test audit on `2026-06-14` is diagnostic-only, not a final/promotable test claim. The tested rows were: parent default `0.959429`, parent minpoints `pv=0.40/final=9/fifth=4` `0.959131`, `cqcalib_ft12` `0.953748`, `cqcalib_lr1e4_ft8` `0.958869`, `curveaux_ft8` `0.959706`, and `lowfp_joint_ft8` `0.959401`. These numbers must not be used to choose checkpoints, thresholds, postprocess settings, losses, or promotion decisions; `curveaux_ft8` remains rejected because its official-val `0.958732` is below the K56 parent `0.959315`.

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
gcs_geometry_curvature = 0.0
gcs_geometry_curvature_beta_px = 5.0
gcs_quality_pairwise = 0.0
gcs_quality_pairwise_margin = 0.2
gcs_fifthness = 0.0
gcs_fifthness_pairwise = 0.0
gcs_fifthness_margin = 0.2
gcs_fifthness_negative_topk = 2
gcs_fifthness_negative_score_thr = 0.1
gcs_fifthness_include_gt5_negatives = False
gcs_count_cumulative = 0.0
gcs_count_cumulative_label_smoothing = 0.0
gcs_count_false_fifth_suppression = 0.0
gcs_count_false_fifth_margin = 0.2
gcs_use_fifthness_decode = False
gcs_fifthness_decode_thr = 0.0
gcs_fifthness_decode_rank_weight = 1.0
```

When `gcs_quality_hard_negative_from_head` is enabled, Quality Head hard negatives are mined from unmatched queries only; matched queries remain matched quality targets even if their current continuous quality target is `0.0`.

The visible-segment hard-negative, GT5 edge Quality floor, and GT5 edge curvature recipes remain default-off infrastructure only. The first K56 curvature gate `gcs_yolo_lane_s_q12_k56_curveaux_ft8_seed1_b32w4` reached best official-val `0.958732`, below the K56 parent `0.959315`, so it is not promotable and should not be rerun as the next gate unless checking reproducibility. Do not use test to rescue or tune rejected recipes.

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
python tools/sweep_tusimple_official.py --weights <weights.pt> --split val --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json --archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset --imgsz 544 960
```

K56 min-points sweep support accepts lists for `--candidate-min-points`, `--final-min-points`, and `--fifth-min-points`; use official-val only for these grids.

GT5 diagnosis on official-val:

```powershell
python tools/diagnose_gcs_gt5.py --weights <weights.pt> --split val --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json --archive-root runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset --imgsz 544 960
```

Final test evaluation, only after selecting the candidate on official-val:

```powershell
python tools/eval_tusimple_official.py --weights <weights.pt> --split test --selection-summary <official-val-selection-summary.json> --imgsz 544 960
```

Extra test audits must use `--diagnostic-only-test`; those results are not valid for selection or promotion.

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
