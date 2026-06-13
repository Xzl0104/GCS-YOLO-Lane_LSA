# Known Bottlenecks

This file summarizes recurring technical and experiment bottlenecks for GCS-YOLO-Lane.

## Count And Candidate Bottlenecks

- GT4 and GT5 scenes are sensitive to Count Head underprediction and candidate-pool shortfall.
- A high Count Head K is not enough if real query candidates fail `min_points`, valid-point gates, or Lane-NMS.
- Track Count Head K and final output count separately; GT5 can fail at raw Count Head calibration or later in candidate validity/postprocess.
- Edge lanes can be short or partially visible; rescue must remain real-query-only and must not fabricate lanes.
- 2026-06-12 official-val sweeps for `gcs_yolo_lane_s_q12_e180_countboundary_rankfix_balgt45_v1` showed a flat threshold surface. The main bottleneck was Count Head / Count Boundary GT4/GT5 confusion, not existence threshold, candidate-pool size, valid-point gates, NMS, or rescue.
- A 6-epoch Count Boundary / GT4-GT5 short fine-tune from `official_best.pt` improved official-val Accuracy from `0.954137` to `0.954782` on `last.pt`, with lower FP/FN and lower GT5 underprediction. The same run's ordinary `best.pt` regressed GT5 output, so short fine-tunes need official-val checkpoint selection and Top-K preservation through `gcs_official_best_top_k`.
- 2026-06-12 official-val/test analysis for `gcs_yolo_lane_s_q12_e180_gt5calib_mainline_seed0` showed that the full GT5-calibration mainline recipe is not promotable: `official_best.pt` reached official-val Accuracy `0.953507`, below the prior countboundary baseline `0.954137` and the short fine-tune `0.954782`. Its ordinary `best.pt` had stronger GT5 output on val but worse FP/GT4-to-5 tradeoff and lower official-val Accuracy `0.951660`.
- In the same `gt5calib_mainline_seed0` run, GT5 diagnosis still localized failures to Count Head underprediction and fifth-lane valid-points quality (`official_best.pt`: 4/74 GT5 count-under, 3/74 valid-points fail; `best.pt`: 2/74 and 2/74). Candidate-pool shortfall and GT5 NMS suppression were near zero, so the next useful work is GT4/GT5 joint calibration and matched candidate quality, not more broad threshold/NMS sweeps.
- 2026-06-13 analysis of `gcs_yolo_lane_s_q12_cb_gt45_ft6_officialtopk_seed2_b8w0` showed that the old rank formula structurally suppressed short GT5 edge lanes: official-val `gt5_output5_rate=0.0`, `count_acc_5=0.0`, and GT5 diagnosis had `rank_score_low=48/74`. Changing rank to use longest-visible-segment mean valid plus 12-point support restored `gt5_output5_rate=0.594595` and reduced `rank_score_low` to `1/74` on the same checkpoint, but official-val Accuracy only reached `0.954186`.
- The remaining GT5 bottleneck after visible-segment rank is quality/count separation, not raw fifth-lane availability. On the same seed2 checkpoint with visible-segment rank, GT5 diagnosis still showed `quality_too_low=13/74` and `count_head_under_predict=12/74`. On the older strong FT6 `last.pt`, visible-segment rank reached `gt5_output5_rate=0.743243` but official-val Accuracy was `0.954474`, below its old selected row `0.954782`, because FP/GT4-to-5 cost rose. The next training step should increase hard GT5 edge `quality_loss` supervision and hard/duplicate quality-negative pressure, not just sweep thresholds.
- 2026-06-13 official-val sweeps for `gcs_yolo_lane_s_q12_cb_gt45_ft8_visrank_qhard_seed1_b8w0` showed the `qhard` fine-tune is not promotable: `best.pt` official-val Accuracy was `0.949357`, while `official_best.pt` independent resweep was `0.953179`, still below the prior `0.954137` countboundary baseline and the `0.954782` short fine-tune reference.
- In the same FT8 seed1 run, `official_best.pt` still failed GT5 fifth-lane survival after Count Head K selection: `gt5_valid_points_fail_rate=0.216216`, `decode/k5_to_output4_rate=0.259259`, and `gt5_output5_rate=0.729730`. Candidate-pool shortfall (`0.005510` overall, `0.027027` on GT5) and GT5 NMS suppression (`0.013514`) were low, so the actionable bottleneck is matched GT5 edge visible-segment support and false fifth-lane quality separation.
- The FT8 seed1 run also showed weak Quality Head separation on official-val: `matched_pred_quality_mean=0.864285` vs `unmatched_pred_quality_mean=0.748277`. The next controlled code path is `gcs_quality_hard_negative_from_head` plus `gcs_point_valid_gt5_edge_segment`, not another broad threshold/NMS/rescue sweep.
- The first GT5 segment-quality fine-tune `gcs_yolo_lane_s_q12_gt5segq_ft10_seed1_b8w0_v1` is not promotable. Its best independent official-val row was `last.pt` with `official_acc=0.953994`, below the prior `0.954137` countboundary baseline and `0.954782` FT6 reference. It improved Quality Head separation (`matched/unmatched=0.870587/0.716851`) and lowered official-val GT5 valid-points failure versus the FT8 `official_best.pt`, but GT5 output fell to `0.662162` and GT5 Count Head underprediction rose to `0.148649`. The root implementation issue found afterward was that Quality Head hard-negative mining used `target_quality == 0` as its negative mask, so matched lanes with zero current quality could be weighted as hard negatives. Hard-negative mining must be strictly unmatched-only before rerunning this recipe.

## Decode Bottlenecks

- Ranking must stay traceable. Quality Head can gate rescue and provide diagnostics, but should not silently replace the intended ranking policy.
- Lane-NMS distance and overlap rules strongly affect edge-lane recall and duplicate suppression.
- Count-aware refill and last-lane rescue are useful diagnostics only when their precision and failure modes are measured on official-val.
- `gcs_soft_count_decision`, `gcs_last_lane_rescue`, and `gcs_edge_last_lane_rescue` remain default-off until selected by official-val evidence.
- The 2026-06-12 `gcs_soft_count_decision` official-val sweeps did not improve Accuracy. On `official_best.pt`, Accuracy dropped from `0.954137` to `0.953481`; on ordinary `best.pt`, Accuracy held at `0.953319` but FP increased. Do not prioritize soft-count decode unless a new hypothesis changes its objective or guardrails.
- The `gt5calib_mainline_seed0` official-val sweeps again showed a mostly flat `conf` surface and no useful rescue effect: last-lane and edge-last-lane rescue attempts had zero successes, and `edge_count4_to5_upgrade` did not fire. Treat rescue/edge-upgrade evidence from this run as diagnostic only, not a reason to promote decode defaults.
- Candidate ranking now uses visible-segment mean valid instead of all-anchor mean valid. This fixes a root decode bias against short TuSimple edge lanes, but it can surface false fifth lanes when the Quality Head does not separate true and false fifth candidates.

## Data And Evaluation Bottlenecks

- Do not mix old `fixed_y_start=0.98` labels with the current `710/720` anchor.
- Do not reverse `--imgsz 544 960`; this silently breaks resize and coordinate assumptions.
- Do not use test for any parameter or checkpoint search.
- Keep train, val, official-val, and test roles separate when interpreting runs.
- Test evidence from 2026-06-12 is diagnostic only and must not be used to choose thresholds, checkpoints, or model settings. The actionable selection signal remains official-val.
- `gt5calib_mainline_seed0` had a val/test ranking mismatch: ordinary `best.pt` test Accuracy `0.955855` slightly exceeded `official_best.pt` test Accuracy `0.955466`, despite worse official-val Accuracy. This is a leakage risk if misused; do not select `best.pt` from this test comparison.
- `gcs_yolo_lane_s_q12_cb_gt45_ft8_visrank_qhard_seed1_b8w0` test results are diagnostic only: `best.pt` reached test Accuracy `0.954517` using its val-selected row, and `official_best.pt` reached `0.955443` using its val-selected row. This must not override official-val selection.
- `tools/diagnose_gcs_gt5.py` must be used on official-val for diagnosis. It defaults to `--split val` and rejects `--split test`; test remains final-only.

## Implementation Bottlenecks

- Loss item names must remain explicit and stable in logs.
- Old modules may return only as controlled candidates with flags and documentation.
- Contract checks are cheaper than full training and should run before metric work.
- Python helper scripts for delegation must normalize low-level spawn payloads at the final boundary.
- For unstable fine-tunes, enable `--gcs-official-best` with Top-K preservation so official-val candidate checkpoints are preserved before `last.pt` is overwritten by later epochs. Use at least `--gcs-official-best-top-k 3`; use `5` when short fine-tunes are unstable or expensive to rerun.
