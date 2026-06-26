# GCS-YOLO-Lane Paper Outline and Evidence Map

Legacy/non-contract draft note: this document was added after rollback target
`b6535f641` and is retained only as a writing outline. It does not define the
active algorithm, available CLI flags, loss items, selected candidates, or
experiment conclusions. Use `docs/agent-context/current-contracts.md` for the
active rollback contract; any post-`b6535f641` experiment content is legacy
only.

Title: **GCS-YOLO-Lane: A YOLO11-Based Structured Lane Detection Network**

Workflow: `ars-outline`, academic-paper `outline-only` mode. This file is a detailed outline and evidence map, not a full manuscript draft.

## Paper Configuration Record

| Parameter | Value |
|---|---|
| Paper type | IMRaD-style empirical computer vision paper |
| Discipline | Computer vision, autonomous driving perception, lane detection |
| Target length | 10,000 body words, excluding references and optional appendix |
| Body language | English |
| Companion language | Chinese version saved separately |
| Citation style | IEEE-style references recommended, because the target field is CS/CV |
| Existing materials | `docs/gcs-yolo-lane-paper-draft-non-experiment.zh.md`, `docs/project-summary-current.md`, `docs/agent-context/*`, project code/configs, local papers under `paper/` |
| Evidence status | Method and contract claims are locally supported; final performance claims are pending active-default official-val and one-shot test evidence |
| Integrity boundary | Do not tune on test. Use official-val for threshold, checkpoint, and postprocess selection. Use test only once for a selected candidate. |
| Scope boundary | This paper outline describes the active 5-25-3 K56 structured-lane branch. Later Count Head, Quality Head, Survival Head, near-miss, and official-best machinery are out of scope unless explicitly reintroduced. |

## Central Thesis

GCS-YOLO-Lane reformulates YOLO-style lane perception as structured lane instance prediction. Instead of treating lane detection as ordinary object detection or final mask segmentation, it predicts a set of lane queries, where each query owns lane existence, a fixed-y ordered point sequence, and per-point visibility. This design aligns the model output with the geometric object required by TuSimple-style evaluation and downstream lane reasoning while preserving a YOLO11-style feature extraction backbone.

## High-Level Contribution Plan

1. **Structured lane output**: Represent each lane as a visibility-aware sequence over `K=56` fixed-y anchors, with `Q=12` learned lane queries.
2. **YOLO11-based architecture adaptation**: Replace ordinary YOLO box or mask outputs with a structured GCS lane head on top of a YOLO11-style backbone.
3. **Line-aware feature processing**: Use LSEM to bias intermediate features toward long, thin, directionally coherent structures.
4. **Multi-scale lane feature fusion**: Use LaneBiFPN over P2, enhanced P3, enhanced P4, and P5 features to combine fine localization cues and semantic context.
5. **Set-based structured supervision**: Train lane queries with Hungarian matching, visibility-aware point supervision, curve/smoothness regularization, and auxiliary mask/edge supervision.
6. **Evaluation protocol discipline**: Define a leakage-free TuSimple fixed-y K56 evaluation protocol that separates official-val selection from one-shot final test reporting.

Performance superiority, state-of-the-art claims, and final comparative conclusions must remain pending until active-default results, verified baseline tables, and ablations are complete.

## Recommended 10,000-Word Structure

| Section | Target words | Purpose |
|---|---:|---|
| Abstract | 250, not counted | Summarize problem, method, evidence boundary, and main result once available |
| 1. Introduction | 1,200 | Motivate structured lane instance prediction and state contributions |
| 2. Related Work | 1,800 | Position GCS-YOLO-Lane against segmentation, row/anchor, query/curve, and YOLO-style perception methods |
| 3. Method | 2,600 | Describe representation, architecture, head, matching, losses, and decoding |
| 4. Experiments | 2,200 | Define dataset, protocol, baselines, main results, ablations, efficiency, and failure analysis |
| 5. Discussion | 1,200 | Interpret design tradeoffs and evidence limits |
| 6. Limitations and Future Work | 700 | State evidence, dataset, and robustness limitations |
| 7. Conclusion | 300 | Close with the validated structural contribution |
| Total body | 10,000 | Excludes references, declarations, and appendix |

## Detailed Outline

### Abstract (about 250 words, not counted)

**Purpose**: Provide a compact summary after experimental evidence is fixed.

**Content to include**:
- Problem: lane detection needs thin, continuous, partially visible instance-level geometry.
- Gap: boxes and final masks do not directly represent ordered lane instances; many methods require postprocessing or predefined anchors.
- Method: GCS-YOLO-Lane uses a YOLO11-style feature extractor, LSEM, LaneBiFPN, and a query-based fixed-y lane head.
- Outputs: `pred_points`, `pred_logits`, `pred_valid_logits`, auxiliary mask and edge logits.
- Evidence boundary: insert active-default official-val and final-test results only after protocol-valid evaluation.

**Evidence**: project contracts, model YAML, GCS head implementation, official evaluation helpers.

**Do not include yet**: "state-of-the-art", "outperforms", or exact ranking claims.

### 1. Introduction (about 1,200 words)

#### 1.1 Lane Detection as Structured Geometry (250 words)

**Purpose**: Establish why lane detection is not ordinary object detection.

**Content summary**:
- Lane markings are long, thin, spatially continuous, and often partially missing due to occlusion, glare, worn paint, or perspective.
- A useful lane detector must recover lane instances, point ordering, and visible spans, not only a compact bounding box or foreground pixels.
- Introduce the paper's framing: lane detection as structured instance prediction.

**Sources and evidence**:
- SCNN: long continuous lane structures need spatial relationship modeling.
- UFLD/UFLDv2: row-anchor formulations show the value of sparse fixed-row representation.
- LaneATT/CLRNet: lane localization benefits from global context plus detailed features.

**Transition**: Move from task structure to limitations of common representations.

#### 1.2 Representation Gap in Existing Pipelines (300 words)

**Purpose**: Explain why the paper is needed.

**Content summary**:
- Segmentation-based methods learn dense lane pixels but need grouping, clustering, curve fitting, or heuristic decoding to obtain lane instances.
- Row/anchor methods are efficient and strong, but their instance organization is tied to predefined anchors or row classifiers.
- Query/curve/sequence methods are closer to structured prediction, but the design space remains open for a YOLO11-based structured lane detector that directly predicts fixed-y point sequences.

**Sources and evidence**:
- SCNN, LaneNetInstance, LaneAF for segmentation and instance grouping.
- UFLD, UFLDv2, LaneATT, CLRNet for row/anchor/refinement methods.
- LSTR, BezierLaneNet, FastDraw, PolyLaneNet for query, curve, and sequence outputs.

**Transition**: Introduce GCS-YOLO-Lane as a structural redesign of YOLO-style output.

#### 1.3 GCS-YOLO-Lane Overview (300 words)

**Purpose**: State what the proposed network does.

**Content summary**:
- GCS-YOLO-Lane keeps a YOLO11-style feature extraction body but replaces ordinary detection/mask output with a structured lane head.
- The model predicts at most `Q=12` candidate lanes.
- Each lane is represented at `K=56` fixed-y anchors aligned with TuSimple official h-samples from `710` down to `160`.
- For each query, the model predicts lane existence, x coordinates at fixed-y anchors, and per-anchor visibility.

**Project evidence**:
- `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`
- `data/tusimple_gcs_fixed_y_960x544.yaml`
- `docs/agent-context/current-contracts.md`

**Transition**: Summarize architecture modules.

#### 1.4 Technical Contributions (250 words)

**Purpose**: Present concrete contribution bullets.

**Content summary**:
- Structured fixed-y representation and label conversion.
- LSEM for line-sensitive feature enhancement.
- LaneBiFPN for multi-scale fusion.
- Query-based GCS lane head with x-only fixed-y prediction and per-point visibility.
- Hungarian matching and visibility-aware loss design.
- Protocol-driven TuSimple evaluation workflow.

**Caution**: Contributions should be phrased as design and implementation contributions unless active-default results justify performance claims.

**Transition**: Close with paper organization.

#### 1.5 Paper Scope and Organization (100 words)

**Purpose**: Make the integrity boundary explicit.

**Content summary**:
- The paper focuses on the active 5-25-3 K56 branch.
- Later experimental mechanisms are not part of the mainline unless explicitly enabled in an experiment table.
- Test results are reporting-only after official-val selection.

### 2. Related Work (about 1,800 words)

#### 2.1 Segmentation-Based Lane Detection (350 words)

**Purpose**: Explain the dominant dense-prediction lineage.

**Content summary**:
- SCNN introduced spatial message passing for long continuous lane structures and showed the need for context beyond local convolutions.
- LaneNetInstance and LaneAF model lane instances through segmentation or affinity fields.
- Segmentation and dense supervision are useful, but final lane output usually requires clustering, grouping, fitting, or protocol conversion.

**Core sources**:
- Pan et al., SCNN, AAAI 2018.
- Neven et al., LaneNet instance segmentation, IV/arXiv 2018.
- Abualsaud et al., LaneAF, RA-L/ICRA 2021.

**Connection to GCS-YOLO-Lane**:
- GCS-YOLO-Lane keeps mask/edge branches only as auxiliary supervision, not as final lane output.

#### 2.2 Row-Based and Anchor-Based Lane Detection (400 words)

**Purpose**: Compare against efficient sparse-coordinate methods.

**Content summary**:
- UFLD reframes lane detection as row-based selecting with global features and structural loss.
- UFLDv2 extends this to hybrid row/column anchors and ordinal classification.
- LaneATT uses line anchors and attention-guided feature aggregation.
- CLRNet refines lane priors across feature levels and uses ROIGather and Line IoU to improve localization.

**Core sources**:
- Qin et al., UFLD, ECCV 2020.
- Qin et al., UFLDv2, TPAMI 2022.
- Tabelini et al., LaneATT, CVPR 2021.
- Zheng et al., CLRNet, CVPR 2022.

**Connection to GCS-YOLO-Lane**:
- GCS-YOLO-Lane shares the insight that sparse y-sampled lane coordinates are efficient, but assigns the whole fixed-y sequence to learned lane queries rather than static lane anchors.

#### 2.3 Query-Based, Curve-Based, and Sequence-Based Methods (400 words)

**Purpose**: Position the work among structured-output methods.

**Content summary**:
- LSTR uses transformer reasoning and Hungarian matching to predict lane shape parameters.
- BezierLaneNet models lanes as Bezier curves to obtain compact holistic geometry.
- PolyLaneNet and FastDraw explore polynomial or sequence-style alternatives.
- These methods reduce reliance on final dense masks, but curve parameterization, sequence generation, and anchor-free query design have different optimization and representation tradeoffs.

**Core sources**:
- Liu et al., LSTR, WACV 2021.
- Feng et al., BezierLaneNet, CVPR 2022.
- Tabelini et al., PolyLaneNet, ICPR 2020.
- Philion, FastDraw, CVPR 2019.

**Connection to GCS-YOLO-Lane**:
- GCS-YOLO-Lane is structured and query-based, but keeps a direct fixed-y point sequence rather than a global curve coefficient vector.

#### 2.4 YOLO-Style Driving Perception and Multi-Task Networks (250 words)

**Purpose**: Explain the YOLO architectural context without overstating novelty.

**Content summary**:
- YOLOP and YOLOPv2 show how YOLO-style shared encoders can support real-time driving perception with multiple decoders.
- Most YOLO-style lane branches are segmentation-oriented or multi-task perception components.
- GCS-YOLO-Lane differs by turning the YOLO-style backbone into a structured lane instance predictor.

**Core sources**:
- Wu et al., YOLOP, Machine Intelligence Research/arXiv.
- Han et al., YOLOPv2, arXiv 2022.
- Recent YOLO-based lane detection/segmentation papers may be cited only after verification.

#### 2.5 Datasets and Evaluation Protocols (250 words)

**Purpose**: Introduce benchmark context and justify TuSimple focus.

**Content summary**:
- TuSimple evaluates x coordinates at sampled horizontal positions and is compatible with fixed-y output.
- CULane and CurveLanes cover more diverse or curved scenarios and can be discussed as future cross-dataset validation targets.
- 3D lane and topology datasets such as OpenLane, OpenLane-V2, and ONCE-3DLanes are outside the current 2D TuSimple scope but show field direction.

**Core sources**:
- TuSimple benchmark metadata.
- CULane from SCNN.
- CurveLanes dataset.
- OpenLane/OpenLane-V2 for future work context.

#### 2.6 Positioning Summary (150 words)

**Purpose**: End the related work section with the exact gap.

**Content summary**:
- The field has strong segmentation, anchor/row, refinement, and curve/query methods.
- The specific gap addressed here is a YOLO11-based structured lane detector that outputs query-owned fixed-y point sequences with explicit visibility.

### 3. Method (about 2,600 words)

#### 3.1 Problem Formulation and Output Contract (300 words)

**Purpose**: Define the task and tensors.

**Content summary**:
- Input: RGB TuSimple image resized as `--imgsz 544 960` in H,W order.
- Output: a set of up to `Q=12` lane queries.
- Default K56 output shapes:
  - `pred_points: B x 12 x 56 x 2`
  - `pred_logits: B x 12`
  - `pred_valid_logits: B x 12 x 56`
  - `aux_mask_logits: B x 2 x H x W`
  - `aux_edge_logits: B x 1 x H x W`
- Explain that `pred_points[..., 1]` is fixed-y in fixed-y mode, while x is predicted.

**Evidence**:
- `docs/agent-context/current-contracts.md`
- `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`
- `ultralytics/nn/modules/gcs_lane.py`

#### 3.2 Fixed-Y Label Representation and Dataset Conversion (400 words)

**Purpose**: Explain how labels are constructed.

**Content summary**:
- K56 labels are regenerated from original TuSimple JSON and images, not resampled from K32.
- Anchors are normalized by original image height: `710/720` to `160/720`, step 10 in original y-pixel coordinates.
- Each GT lane is interpolated at the fixed y anchors.
- Valid anchors store normalized `(x, y)` and `lane_valid=1`; invalid anchors keep fixed y with `x=0` and `lane_valid=0`.
- The converted `.npz` includes `semantic_mask`, `edge_mask`, `lanes`, `lane_valid`, `num_lanes`, `point_mode`, `fixed_y`, `raw_file`, `image_shape`, and `num_points`.

**Evidence**:
- `tools/convert_tusimple_to_gcs.py`
- `gcs_tools/label_utils.py`
- `data/tusimple_gcs_fixed_y_960x544.yaml`

**Figure idea**:
- Figure 2: fixed-y sampling diagram showing 56 anchors from bottom to top.

#### 3.3 Overall Architecture (350 words)

**Purpose**: Give the complete model pipeline.

**Content summary**:
- YOLO11-style backbone extracts P2-P5 features.
- LSEM is inserted after P3 and P4.
- LaneBiFPN fuses P2, enhanced P3, enhanced P4, and P5.
- GCSLaneHead flattens fused features into multi-scale spatial tokens, adds position and level embeddings, and decodes lane queries.
- Auxiliary mask and edge branches supervise training features only.

**Evidence**:
- model YAML head and backbone definitions.
- `ultralytics/nn/modules/gcs_lane.py`.

**Figure idea**:
- Figure 1: architecture diagram with backbone, LSEM, LaneBiFPN, GCS head, and outputs.

#### 3.4 Line-Sensitive Feature Enhancement Module (350 words)

**Purpose**: Explain LSEM.

**Content summary**:
- LSEM includes LineStripAttention, coordinate reweighting, dilated context, residual connection, and activation.
- LineStripAttention uses horizontal and vertical strip depthwise convolutions and a direction gate.
- Coordinate reweighting preserves height/width spatial cues.
- Dilated context branch enlarges receptive field.

**Evidence**:
- `ultralytics/nn/modules/gcs_lane.py`, classes `CoordReweight`, `LineStripAttention`, and `LSEM`.
- Related-work support: SCNN and RESA highlight the importance of long-range structured lane context.

**Ablation requirement**:
- Compare with and without LSEM on active-default official-val.

#### 3.5 LaneBiFPN Multi-Scale Fusion (300 words)

**Purpose**: Explain why multi-scale fusion is needed.

**Content summary**:
- Fine features help point localization; high-level features help distinguish lane markings from similar road structures.
- LaneBiFPN aligns P2-P5 channels and uses bidirectional weighted fusion.
- The output provides a unified multi-scale representation to the query decoder.

**Evidence**:
- model YAML: `LaneBiFPN [128]` over `[2, 5, 8, 12]`.
- `LaneBiFPN` and `WeightedFusion` classes.
- Related-work support: CLRNet emphasizes high/low feature complementarity.

#### 3.6 Query-Based GCS Lane Head (450 words)

**Purpose**: Explain the structured decoder.

**Content summary**:
- `Q=12` learned lane queries attend to multi-scale spatial tokens through a transformer decoder.
- Current implementation uses 3 decoder layers and 8 attention heads.
- In fixed-y mode, the head predicts x logits for 56 anchors; y anchors come from a registered fixed-y buffer.
- A reference-logit mechanism initializes different query shapes.
- Image-conditioned refinement samples features at predicted fixed-y points to refine x logits and point-valid logits.
- The head predicts lane existence and per-point visibility.

**Evidence**:
- `GCSLaneHead` class.
- model YAML final layer.

**Figure idea**:
- Figure 3: query-owned fixed-y point sequence and visibility branch.

#### 3.7 Matching and Training Objective (450 words)

**Purpose**: Describe supervision.

**Content summary**:
- Hungarian matching assigns predicted queries to GT lanes per image.
- Matching cost combines point cost, curve cost, and existence cost.
- Losses include lane existence, visible-point geometry, point visibility, smoothness, curve regularization, auxiliary mask, and auxiliary edge loss.
- Logged but default-disabled experimental items should be described carefully: count loss, under-5 count loss, duplicate margin, spurious margin, lane-balanced point, and short-valid recall are not core claims unless enabled in a reported experiment.
- Auxiliary mask/edge branches support feature learning but are not final output.

**Evidence**:
- `ultralytics/utils/gcs_matcher.py`
- `ultralytics/utils/gcs_loss.py`
- `ultralytics/cfg/default.yaml`

#### 3.8 Decoding and TuSimple Conversion (350 words)

**Purpose**: Explain inference and metric conversion.

**Content summary**:
- Decode by query existence score, point-valid threshold, longest contiguous visible span, `min_points`, optional lane NMS, and `max_det`.
- Convert decoded GCS lanes to TuSimple h-samples with visible points and interpolation.
- Sort final lanes left-to-right by bottom x where needed.
- No GT may be used during inference or decode.

**Evidence**:
- `ultralytics/utils/gcs_postprocess.py`
- `gcs_tools/tusimple_official_eval.py`
- `tools/eval_tusimple_official.py`

### 4. Experiments (about 2,200 words)

This section is a plan. It must not be finalized until active-default training, official-val selection, one-shot final test, verified baselines, and ablations are available.

#### 4.1 Dataset and Protocol (350 words)

**Purpose**: Define the evaluation surface.

**Content summary**:
- Dataset: TuSimple fixed-y K56 converted dataset.
- Splits: train 3263, val 363, test 2782.
- Evaluation: official TuSimple Accuracy, FP, FN, and optionally project `official_score`.
- Selection: official-val only.
- Final test: one-shot reporting with selected decode.

**Evidence**:
- `docs/agent-context/project-context.md`
- `docs/agent-context/experiment-rules.md`
- `tools/sweep_tusimple_official.py`
- `tools/eval_tusimple_official.py`

#### 4.2 Implementation Details (250 words)

**Purpose**: Make reproduction possible.

**Content summary**:
- Model config, data config, image size.
- Training environment: remote RTX 4090 formal training, batch 32 starting point.
- Optimizer and schedule from the frozen active-default `args.yaml`.
- Pretraining source if used.

**Evidence needed**:
- Active-default run `args.yaml`.
- Formal training command.

#### 4.3 Main Results (450 words)

**Purpose**: Report protocol-valid performance only.

**Required tables**:
- Table 1: TuSimple official-val selected row.
- Table 2: one-shot official test report using the selected row.
- Table 3: comparison with verified baselines.

**Required fields**:
- `official_acc`, `official_FP`, `official_FN`, `official_score`, `count_acc`, `avg_total_ms`.
- selected `conf`, `point_valid_thr`, `nms_dist_px`, `max_det`, `min_points`.

**Current status**:
- Pending for active-default.
- Legacy or reporting-only results can be mentioned only as historical notes, not as primary claims.

#### 4.4 Baseline Comparison Plan (300 words)

**Purpose**: Define fair comparisons.

**Candidate baseline families**:
- Segmentation/context: SCNN, RESA, SAD.
- Row/anchor: UFLD, UFLDv2, LaneATT, CLRNet.
- Query/curve/sequence: LSTR, PolyLaneNet, BezierLaneNet.
- YOLO-style multi-task perception: YOLOP, YOLOPv2 as contextual references rather than direct TuSimple 2D lane baselines unless their comparable lane metrics are verified.

**Rules**:
- Use official reported metrics only after citation verification.
- Separate TuSimple, CULane, LLAMAS, and CurveLanes metrics.
- Do not mix FPS from different hardware without caveat.

#### 4.5 Ablation Study Plan (450 words)

**Purpose**: Tie method claims to evidence.

**Ablations to run or report**:
- Baseline head without LSEM.
- Baseline head without LaneBiFPN or with simpler feature fusion.
- GCS head without fixed-y refinement.
- Without point-validity branch.
- Without auxiliary mask/edge supervision.
- Alternative `K` or anchor range only if protocol permits and labels are regenerated.
- Count and margin experimental knobs only in a clearly labeled experiment section if enabled.

**Required caution**:
- Any ablation must use official-val for selection and must not use final test for tuning.

#### 4.6 Efficiency and Complexity (200 words)

**Purpose**: Make deployment relevance concrete.

**Content summary**:
- Report parameters, FLOPs/MACs if available, throughput, preprocessing, inference, decode, and total time.
- Compare only under matched or clearly stated hardware.

**Evidence needed**:
- Model profiling command output.
- Evaluation summary timing fields.

#### 4.7 Failure Analysis (200 words)

**Purpose**: Explain residual weaknesses honestly.

**Content summary**:
- Analyze lane-count errors, short visible lanes, low-score short GT, geometry misses, spurious extras, duplicate-like extras, and `GT4`/`GT5` confusion.
- Use train/val diagnostics and official-val only for investigative iteration.
- Keep final-test breakdown reporting-only.

**Evidence sources**:
- `docs/agent-context/known-bottlenecks.md`
- future active-default train/val failure traces.

### 5. Discussion (about 1,200 words)

#### 5.1 What Structured Output Changes (300 words)

**Purpose**: Interpret the representation contribution.

**Content summary**:
- The model output is a lane instance object, not a mask proxy.
- Query ownership binds existence, geometry, and visibility into one candidate.
- Fixed-y anchors simplify evaluation conversion while preserving point visibility.

#### 5.2 Tradeoff Against Segmentation and Row/Anchor Methods (300 words)

**Purpose**: Explain benefits and costs.

**Content summary**:
- Compared with segmentation, GCS-YOLO-Lane reduces dependence on post-hoc instance grouping.
- Compared with row/anchor methods, it uses learned queries to organize lane candidates.
- The cost is the need for stable query allocation, count calibration, and robust visibility prediction.

#### 5.3 Role of Auxiliary Dense Supervision (200 words)

**Purpose**: Clarify the role of mask/edge branches.

**Content summary**:
- Dense supervision is useful for feature learning.
- It does not define the final output.
- This distinction prevents the method from being mischaracterized as another segmentation pipeline.

#### 5.4 Evidence Boundary and Research Integrity (200 words)

**Purpose**: Prevent overclaiming.

**Content summary**:
- Official-val selects candidate and decode.
- Test is reported once.
- Historical near-miss or reporting-only runs cannot promote a method.

#### 5.5 Practical Implications (200 words)

**Purpose**: Explain possible downstream value.

**Content summary**:
- Ordered point sequences are easier to pass into lane tracking, planning, map alignment, and curve fitting than raw masks.
- Point visibility supports partial-lane reasoning under occlusion.

### 6. Limitations and Future Work (about 700 words)

#### 6.1 Experimental Evidence Not Yet Complete (150 words)

State that the paper cannot make final superiority claims before active-default results, verified baselines, and ablations.

#### 6.2 TuSimple-Centric Fixed-Y Design (150 words)

Discuss that the representation aligns well with TuSimple but needs CULane, CurveLanes, LLAMAS, or other datasets to demonstrate broader generalization.

#### 6.3 Lane-Count Stability (150 words)

Discuss known risk around short visible side lanes, GT4/GT5 count confusion, duplicate-like extras, and spurious extras.

#### 6.4 2D Scope (100 words)

State that 3D lane detection, topology reasoning, map learning, and temporal consistency are outside the current paper.

#### 6.5 Future Work (150 words)

Propose cross-dataset validation, stronger active-default ablations, temporal extension, 3D adaptation, and more principled query allocation/count calibration.

### 7. Conclusion (about 300 words)

**Purpose**: Close with the strongest supported claim.

**Content summary**:
- GCS-YOLO-Lane is a YOLO11-based structured lane detector.
- It directly predicts query-owned fixed-y point sequences with lane existence and point visibility.
- LSEM, LaneBiFPN, GCS head, Hungarian matching, and auxiliary dense supervision form the core method.
- Final empirical conclusion must be inserted only after active-default evaluation is complete.

## Evidence Map

### Evidence Status Legend

| Status | Meaning |
|---|---|
| Local-code verified | Supported by project code/config/docs in this repository |
| Local-literature supported | Supported by local markdown/PDF metadata under `paper/`, but citation details still need final verification |
| Pending experiment | Requires active-default official-val/test, ablation, timing, or failure-analysis output |
| Do not claim | Not supported under current evidence boundary |

### Method Claim Map

| Claim | Planned section | Evidence source | Status | Caution |
|---|---|---|---|---|
| The model predicts structured lane point sequences, not YOLO boxes or final masks. | 1.3, 3.1 | `gcs-yolo-lane-s.yaml`, `current-contracts.md`, `GCSLaneHead` output dict | Local-code verified | Auxiliary mask/edge branches exist, but are not final lane output. |
| TuSimple input uses `--imgsz 544 960` in H,W order. | 3.1, 4.1 | `AGENTS.md`, `current-contracts.md`, data YAML | Local-code verified | Do not reverse to W,H. |
| Default K56 contract uses `Q=12`, `K=56`, `fixed_y=710/720 -> 160/720`. | 1.3, 3.1, 3.2 | model YAML, data YAML, current contracts | Local-code verified | K56 labels must be regenerated from original TuSimple JSON/images. |
| Labels include masks, edge masks, lanes, lane_valid, point_mode, fixed_y, image shape, and raw_file. | 3.2 | `tools/convert_tusimple_to_gcs.py` | Local-code verified | Confirm exact fields in any released dataset artifact. |
| LSEM uses line-strip attention, direction gating, coordinate reweighting, dilated context, and residual activation. | 3.4 | `ultralytics/nn/modules/gcs_lane.py` | Local-code verified | Needs ablation before claiming performance contribution. |
| LaneBiFPN fuses P2, enhanced P3, enhanced P4, and P5. | 3.5 | model YAML, `LaneBiFPN` implementation | Local-code verified | Needs ablation for empirical contribution. |
| GCSLaneHead uses learned lane queries and a transformer decoder. | 3.6 | `GCSLaneHead` implementation | Local-code verified | Avoid claiming it is the first query-based lane detector. LSTR and related works exist. |
| Fixed-y mode predicts x only while y anchors are fixed. | 3.1, 3.6 | `GCSLaneHead`, fixed-y buffer construction | Local-code verified | Explain that output still has 2D points because y is restored from anchors. |
| Hungarian matching assigns predictions to GT lanes. | 3.7 | `gcs_matcher.py`, `gcs_loss.py` | Local-code verified | Compare to LSTR/DETR-style set prediction only with citations. |
| Decode uses query score, point-valid threshold, longest contiguous visible span, min points, optional NMS, and max_det. | 3.8, 4.1 | `gcs_postprocess.py` | Local-code verified | Postprocess parameters must be selected on official-val only. |
| TuSimple official-val sweep rejects `--split test` for threshold search. | 4.1, 5.4 | `tools/sweep_tusimple_official.py`, experiment rules | Local-code verified | Keep final test closed until selected row is fixed. |

### Literature Evidence Map

| Theme | Sources from local library | Use in paper | Status |
|---|---|---|---|
| Lane continuity and spatial context | SCNN, RESA, SAD | Motivation for line-sensitive and context-aware feature modules | Local-literature supported |
| Segmentation and instance grouping | LaneNetInstance, LaneAF, SCNN | Explain dense-output lineage and postprocessing needs | Local-literature supported |
| Row-based sparse representation | UFLD, UFLDv2, E2ELMD | Justify fixed-y/row-sampled output as efficient and benchmark-aligned | Local-literature supported |
| Anchor-based lane detection | LaneATT, CLRNet, Line-CNN references via CLRNet/LaneATT | Compare anchor ownership with learned query ownership | Local-literature supported |
| Query and transformer lane prediction | LSTR, Laneformer, LDTR, SparseLaneformer | Position GCS head among structured query-like methods | Local-literature supported |
| Curve-based lane output | PolyLaneNet, BezierLaneNet, BezierFormer | Contrast point-sequence output with global curve parameters | Local-literature supported |
| Sequence/keypoint alternatives | FastDraw, PINet, GANet, LanePtrNet | Discuss non-mask structured alternatives | Local-literature supported |
| YOLO-style driving perception | YOLOP, YOLOPv2, A-YOLOM, YOLOMH, Q-YOLOP | Frame YOLO-style shared-backbone perception context | Local-literature supported |
| Dataset scope | TuSimple, CULane, CurveLanes, LLAMAS, OpenLane, OpenLane-V2 | Explain current TuSimple focus and future cross-dataset directions | Local-literature supported |

### Experiment Evidence Map

| Required evidence | Planned table/figure | Required source artifact | Current status |
|---|---|---|---|
| Active-default training configuration | Table: Training details | active-default `args.yaml` | Pending experiment |
| Official-val selected row | Table: Selected validation decode | `tusimple_official_sweep_summary.json` from active-default run | Pending experiment |
| One-shot final test | Table: Official test result | `tusimple_official_summary.json` from selected row | Pending experiment |
| Baseline comparison | Table: TuSimple comparisons | Verified papers and metric extraction | Pending verification |
| LSEM contribution | Ablation table | active-default ablation run without LSEM | Pending experiment |
| LaneBiFPN contribution | Ablation table | active-default ablation with simpler fusion | Pending experiment |
| Query/visibility design | Ablation table | active-default variants of head/visibility branch | Pending experiment |
| Auxiliary mask/edge role | Ablation table | active-default without mask/edge supervision | Pending experiment |
| Failure modes | Figure/table: error taxonomy | train/val failure trace, official-val diagnostics | Pending experiment |
| Efficiency | Table: Params, FLOPs, FPS/time | model profiling and eval timing | Pending experiment |

### Claims That Must Not Be Made Yet

| Claim | Why not |
|---|---|
| GCS-YOLO-Lane is state-of-the-art on TuSimple. | Active-default official-val/test and verified baselines are not complete. |
| GCS-YOLO-Lane outperforms SCNN, UFLD, LaneATT, CLRNet, LSTR, or BezierLaneNet. | Requires fair, citation-verified baseline table and comparable metrics. |
| LSEM or LaneBiFPN improves accuracy. | Requires ablation evidence. |
| Default-disabled margin/count/short-lane losses are core method contributions. | They are experiment knobs unless enabled in a reported configuration. |
| Reporting-only final-test near-miss results can select a candidate. | Project rules allow selection only on official-val. |

## Recommended Figures and Tables

| ID | Type | Content | Evidence source |
|---|---|---|---|
| Fig. 1 | Architecture | YOLO11-style backbone, LSEM, LaneBiFPN, GCS head, outputs | model YAML and code |
| Fig. 2 | Representation | K56 fixed-y anchors and point visibility | data YAML, conversion code |
| Fig. 3 | Head detail | Learned lane queries and x-only fixed-y prediction | `GCSLaneHead` |
| Fig. 4 | Training | Hungarian matching and loss components | matcher/loss code |
| Table 1 | Method contract | Input/output/loss/data contracts | `current-contracts.md` |
| Table 2 | Literature matrix | Baseline families and representation types | `paper/papers_metadata.xlsx` |
| Table 3 | Main result | official-val and one-shot test | pending active-default artifacts |
| Table 4 | Ablation | LSEM, LaneBiFPN, head/visibility, auxiliary supervision | pending ablation artifacts |
| Table 5 | Failure analysis | count, short-lane, spurious/duplicate, visibility errors | pending diagnostics |

## Writing Notes for the Next Draft

- Use "structured lane detection" consistently.
- Use "fixed-y anchors" for K56 representation and reserve "row anchors" for prior literature.
- Use "auxiliary mask/edge supervision" instead of implying final segmentation output.
- Keep results language conditional until evidence is available.
- Cite local papers after final citation verification. Do not rely only on filenames.
- Keep all TuSimple commands in `--imgsz 544 960` H,W order.
