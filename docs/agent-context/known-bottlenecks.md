# Known Bottlenecks

This file summarizes recurring technical and experiment bottlenecks for GCS-YOLO-Lane.

## Count And Candidate Bottlenecks

- GT4 and GT5 scenes are sensitive to Count Head underprediction and candidate-pool shortfall.
- A high Count Head K is not enough if real query candidates fail `min_points`, valid-point gates, or Lane-NMS.
- Track Count Head K and final output count separately; GT5 can fail at raw Count Head calibration or later in candidate validity/postprocess.
- Edge lanes can be short or partially visible; rescue must remain real-query-only and must not fabricate lanes.

## Decode Bottlenecks

- Ranking must stay traceable. Quality Head can gate rescue and provide diagnostics, but should not silently replace the intended ranking policy.
- Lane-NMS distance and overlap rules strongly affect edge-lane recall and duplicate suppression.
- Count-aware refill and last-lane rescue are useful diagnostics only when their precision and failure modes are measured on official-val.
- `gcs_soft_count_decision`, `gcs_last_lane_rescue`, and `gcs_edge_last_lane_rescue` remain default-off until selected by official-val evidence.

## Data And Evaluation Bottlenecks

- Do not mix old `fixed_y_start=0.98` labels with the current `710/720` anchor.
- Do not reverse `--imgsz 544 960`; this silently breaks resize and coordinate assumptions.
- Do not use test for any parameter or checkpoint search.
- Keep train, val, official-val, and test roles separate when interpreting runs.

## Implementation Bottlenecks

- Loss item names must remain explicit and stable in logs.
- Old modules may return only as controlled candidates with flags and documentation.
- Contract checks are cheaper than full training and should run before metric work.
- Python helper scripts for delegation must normalize low-level spawn payloads at the final boundary.
