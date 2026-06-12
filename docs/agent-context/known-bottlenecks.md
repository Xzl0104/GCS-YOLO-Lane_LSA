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

## Decode Bottlenecks

- Ranking must stay traceable. Quality Head can gate rescue and provide diagnostics, but should not silently replace the intended ranking policy.
- Lane-NMS distance and overlap rules strongly affect edge-lane recall and duplicate suppression.
- Count-aware refill and last-lane rescue are useful diagnostics only when their precision and failure modes are measured on official-val.
- `gcs_soft_count_decision`, `gcs_last_lane_rescue`, and `gcs_edge_last_lane_rescue` remain default-off until selected by official-val evidence.
- The 2026-06-12 `gcs_soft_count_decision` official-val sweeps did not improve Accuracy. On `official_best.pt`, Accuracy dropped from `0.954137` to `0.953481`; on ordinary `best.pt`, Accuracy held at `0.953319` but FP increased. Do not prioritize soft-count decode unless a new hypothesis changes its objective or guardrails.
- The `gt5calib_mainline_seed0` official-val sweeps again showed a mostly flat `conf` surface and no useful rescue effect: last-lane and edge-last-lane rescue attempts had zero successes, and `edge_count4_to5_upgrade` did not fire. Treat rescue/edge-upgrade evidence from this run as diagnostic only, not a reason to promote decode defaults.

## Data And Evaluation Bottlenecks

- Do not mix old `fixed_y_start=0.98` labels with the current `710/720` anchor.
- Do not reverse `--imgsz 544 960`; this silently breaks resize and coordinate assumptions.
- Do not use test for any parameter or checkpoint search.
- Keep train, val, official-val, and test roles separate when interpreting runs.
- Test evidence from 2026-06-12 is diagnostic only and must not be used to choose thresholds, checkpoints, or model settings. The actionable selection signal remains official-val.
- `gt5calib_mainline_seed0` had a val/test ranking mismatch: ordinary `best.pt` test Accuracy `0.955855` slightly exceeded `official_best.pt` test Accuracy `0.955466`, despite worse official-val Accuracy. This is a leakage risk if misused; do not select `best.pt` from this test comparison.

## Implementation Bottlenecks

- Loss item names must remain explicit and stable in logs.
- Old modules may return only as controlled candidates with flags and documentation.
- Contract checks are cheaper than full training and should run before metric work.
- Python helper scripts for delegation must normalize low-level spawn payloads at the final boundary.
- For unstable fine-tunes, enable `--gcs-official-best --gcs-official-best-top-k 3` so official-val candidate checkpoints are preserved before `last.pt` is overwritten by later epochs.
