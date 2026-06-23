# GCS-YOLO-Lane: A YOLO11-Based Structured Lane Detection Network

Draft status: non-experimental manuscript sections only.

This draft intentionally leaves experimental results, numerical comparisons, and final performance claims unresolved. Citations are written as verification placeholders, for example `[REF: LSTR]`, and must be checked against the original papers before submission. The main experimental line is assumed to be the current active default configuration, not legacy `gt4short15` or reporting-only final-test runs.

## Mini-Outline

- **Story**: Lane detection should be treated as structured lane-instance prediction rather than as ordinary box detection or final mask segmentation.
- **Technical core**: GCS-YOLO-Lane adapts a YOLO11-style network into a fixed-y, query-based structured detector that outputs ordered lane point sequences.
- **Method modules**: structured fixed-y representation, line-sensitive feature enhancement, multi-scale LaneBiFPN fusion, query-based GCS lane head, Hungarian matching, and visibility-aware supervision.
- **Evidence boundary**: method and implementation details are supported by the project code and contracts; empirical superiority is pending active-default official-val/test evidence.
- **Submission boundary**: the draft can become a Q3/Q4 journal manuscript only after experiments, verified baselines, and citation metadata are completed.

## Abstract

Lane detection in road scenes requires instance-level geometric reasoning over elongated and partially visible markings. Conventional object-detection formulations are poorly aligned with this output structure, while segmentation-style formulations often require additional postprocessing to recover lane instances from pixel regions. This paper proposes **GCS-YOLO-Lane**, a YOLO11-based structured lane detection network that represents each lane as an ordered sequence of points on fixed vertical anchors and predicts a set of candidate lane instances through learnable lane queries. The network combines a YOLO11-style feature extractor with a line-sensitive enhancement module, a multi-scale LaneBiFPN, and a query-based lane head that jointly estimates lane existence, point coordinates, and point-level visibility. Training uses Hungarian matching between predicted queries and ground-truth lanes, with supervision over lane existence, point geometry, visibility, curve regularity, smoothness, and auxiliary mask/edge maps. The final empirical sentence is deferred until the active default configuration has been selected on official validation and evaluated once on the final TuSimple test split. This design positions GCS-YOLO-Lane as a structured lane instance detector rather than a segmentation mask generator.

**Keywords**: lane detection; structured prediction; YOLO11; query-based detection; fixed-y representation; TuSimple

## 1. Introduction

Lane detection is a core perception task for intelligent vehicles because lane markings define the spatial constraints that support localization, path planning, and driving-scene understanding. Unlike generic object detection, the desired output is not a rectangular region around a compact object. A lane marking is an elongated curve-like structure whose visible portion may be fragmented by occlusion, worn paint, shadows, or viewpoint changes. A lane detector therefore needs to recover lane instances, preserve their geometric order, and identify which parts of each lane are visible.

Many modern perception systems are built around detection boxes, segmentation masks, anchors, or row-wise classification. These formulations have clear engineering advantages, but they do not always express the lane-detection target directly. A segmentation model can produce lane pixels, but lane pixels must still be grouped into lane instances and converted into the sampled coordinates required by common benchmarks. Anchor-based and row-based methods can be efficient and competitive, especially under fixed sampling protocols, but their output is often tied to predefined anchors or row positions. Query-based and curve-based methods move closer to structured lane prediction, yet the design space remains open for YOLO-style architectures that directly emit lane instances as ordered point sequences.

This paper studies lane detection as structured instance prediction. The central idea is to represent each lane as a point sequence on a fixed set of vertical anchors. In the TuSimple setting used by this project, each lane is sampled at `K=56` fixed-y anchors corresponding to `710, 700, 690, ..., 160` in the original image coordinate system. Rather than predicting only a mask and recovering curves later, the model predicts candidate lane instances directly. Each predicted lane carries query-level existence confidence, per-anchor x coordinates, and point-level visibility. This formulation makes the final lane output explicit and keeps the representation aligned with the official fixed-y evaluation protocol.

We propose **GCS-YOLO-Lane**, a YOLO11-based structured lane detection network. The network keeps the practical feature-extraction strengths of a YOLO11-style backbone while replacing the ordinary detection or segmentation output with a query-based structured lane head. A line-sensitive enhancement module strengthens elongated lane cues through horizontal and vertical strip responses, direction gating, coordinate-aware reweighting, and dilated context. A LaneBiFPN fuses P2-P5 features into multi-scale spatial tokens. A GCS lane head then uses learnable lane queries to attend to these tokens and predict a bounded set of candidate lanes.

The method is designed around two observations. First, lane markings are not isolated local objects; they are long, directionally biased structures. Feature enhancement should therefore be sensitive to line-like spatial continuity rather than relying only on generic convolutional responses. Second, a lane instance is naturally represented as an ordered sequence with incomplete visibility. The network should model point locations and point visibility together, instead of assuming that every sampled position along a lane is observable.

Training follows a set-prediction view. A Hungarian matcher assigns predicted lane queries to ground-truth lanes using point geometry, curve consistency, and lane-existence cost. The loss then supervises query existence, visible-point regression, point-level visibility, curve regularity, smoothness, and auxiliary segmentation/edge maps. The auxiliary mask and edge branches are used as training signals for feature learning; they are not the final lane representation. The final representation remains the structured point sequence predicted by each lane query.

This paper makes the following contributions:

1. It formulates a YOLO11-based lane detector as a structured lane instance network that directly outputs ordered fixed-y point sequences.
2. It introduces a query-based GCS lane head that predicts lane existence, point coordinates, and point-level visibility for a fixed set of candidate lane instances.
3. It incorporates line-sensitive feature enhancement and multi-scale LaneBiFPN fusion to improve the spatial features used by the structured lane head.
4. It defines a training pipeline based on Hungarian matching and visibility-aware supervision, with auxiliary mask and edge branches used only as training support.

The experimental claims are intentionally not stated in this draft. They will be added only after the active default configuration has a complete official-validation selection, one-shot final-test report, verified baseline table, and ablation evidence.

## 2. Related Work

### 2.1 Segmentation-Based Lane Detection

Segmentation-based lane detection treats lane markings as pixel-level regions or affinity structures. A representative early line of work uses convolutional spatial message passing to enhance thin lane structures and then recovers lane instances through additional grouping or fitting procedures `[REF: SCNN]`. These methods are important because they recognize that lane markings require contextual reasoning beyond local edge detection. Their limitation for the present paper is representational: a segmentation mask is not yet a lane instance sequence. The mask must be converted into ordered lane coordinates, and the grouping step can introduce additional heuristics.

GCS-YOLO-Lane differs from this route by making the final lane representation explicit. The model still uses auxiliary mask and edge supervision during training, because dense supervision can improve feature learning. However, its inference output is not a segmentation map. Each lane query directly emits a structured lane instance with ordered fixed-y points and point-level visibility.

### 2.2 Anchor-Based and Row-Based Lane Detection

Anchor-based and row-based methods have become strong practical baselines for lane detection. Row-wise classification methods such as UFLD and UFLDv2 frame lane localization as selecting positions along predefined rows, which supports high-speed inference and compact output representations `[REF: UFLD]` `[REF: UFLDv2]`. Anchor-based methods such as LaneATT use lane anchors and attention mechanisms to model candidate lanes efficiently `[REF: LaneATT]`. Refinement-based systems such as CLRNet further improve lane localization through stronger feature aggregation and iterative refinement `[REF: CLRNet]`.

These methods are highly relevant to the TuSimple setting because TuSimple evaluates predictions on fixed horizontal samples. Their strength is that they exploit a structure close to the benchmark protocol. Their limitation, from the perspective of this paper, is that lane instances are often organized around predefined anchors, row classifiers, or proposal refinements rather than around a direct set of learned lane queries. GCS-YOLO-Lane keeps the fixed-y sampling compatibility but represents each candidate lane as a query-owned point sequence. This allows the model to predict lane existence, geometry, and point visibility within one structured output.

### 2.3 Query-Based, Curve-Based, and Sequence-Based Lane Detection

Recent work has moved lane detection closer to structured prediction. Query-based approaches such as LSTR formulate lane detection as predicting a set of lane structures with transformer-style reasoning `[REF: LSTR]`. Curve-based approaches such as BezierLaneNet represent lane geometry through parametric curves rather than dense masks `[REF: BezierLaneNet]`. Sequence-based approaches such as Lane2Seq treat lane output as an ordered sequence-generation problem `[REF: Lane2Seq]`. These methods are the closest conceptual comparison for GCS-YOLO-Lane because they share the goal of predicting structured lane instances rather than only pixel regions.

The distinction of GCS-YOLO-Lane is its combination of a YOLO11-style backbone, line-sensitive feature enhancement, multi-scale LaneBiFPN fusion, and a fixed-y query head. Instead of predicting only curve parameters or using a generic query formulation, the proposed head predicts x coordinates on fixed-y anchors and explicitly estimates point-level visibility. This design is intended to align structured lane prediction with the TuSimple fixed-y evaluation protocol while retaining a network organization familiar from YOLO-style perception systems.

### 2.4 Positioning of This Work

The closest comparison should therefore be made along two axes. Against anchor/row-based methods, GCS-YOLO-Lane should be evaluated for whether learned lane queries can remain competitive under a fixed-y protocol. Against query/curve/sequence methods, GCS-YOLO-Lane should be evaluated for whether its YOLO11-based architecture and visibility-aware fixed-y head provide a useful structured alternative. The final manuscript should not claim superiority over any method in this section until the comparison table has verified papers, evaluation settings, and official TuSimple metrics.

## 3. Method

### 3.1 Overview

GCS-YOLO-Lane treats lane detection as structured set prediction. Given an input image, the network predicts at most `Q=12` candidate lanes. Each candidate lane is represented by `K=56` ordered points on fixed-y anchors, a lane-existence logit, and a point-validity logit for each anchor. The default output contract is:

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

The model contains four main parts: a YOLO11-style feature extractor, line-sensitive enhancement modules, a LaneBiFPN multi-scale fusion neck, and a query-based GCS lane head. During training, predicted queries are matched to ground-truth lanes with Hungarian assignment. During inference, decoded lanes are obtained from query scores, point visibility, minimum visible-point constraints, optional lane NMS, and a maximum lane count.

### 3.2 Structured Fixed-Y Lane Representation

The fixed-y representation converts each lane into an ordered point sequence. For TuSimple, the y anchors are the official h-sample positions from `710` down to `160` with a step of `10` pixels, normalized by the original image height `720`. Therefore, the normalized fixed-y range is:

```text
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end   = 160 / 720 = 0.2222222222222222
K = 56
```

For each ground-truth lane, the label generation pipeline interpolates x coordinates at these fixed-y anchors. Anchors inside the visible lane range receive normalized `(x, y)` coordinates and a valid flag. Anchors without a valid lane point keep the fixed y coordinate but are marked invisible. This representation separates geometry from visibility: the y coordinate is fixed by the protocol, the x coordinate is predicted by the model, and `lane_valid` indicates which points should contribute to visible-lane supervision.

This representation has two practical advantages. First, it directly matches the fixed-y sampling format used by the TuSimple official evaluation, reducing the need for late-stage curve conversion. Second, it gives the model a stable output tensor for every candidate lane while still allowing partial visibility through per-point validity logits.

### 3.3 Line-Sensitive Feature Enhancement Module

Lane markings are thin, elongated, and directionally structured. A generic convolutional feature map can capture local texture, but it may not emphasize long horizontal or vertical continuity strongly enough for lane geometry. GCS-YOLO-Lane therefore inserts line-sensitive enhancement modules into the YOLO11-style backbone at the P3 and P4 feature stages.

The line-sensitive enhancement module contains four components. First, horizontal and vertical strip depthwise convolutions extract directional responses over elongated neighborhoods. Second, a direction gate computes a soft weighting between the horizontal and vertical responses, allowing the module to adapt to the dominant local structure. Third, coordinate-aware reweighting modulates feature responses along height and width so that spatial location remains explicit. Fourth, a dilated depthwise context branch expands the receptive field without discarding the original spatial resolution. The module output is added back through a residual connection.

The purpose of this module is not to replace the backbone, but to bias intermediate features toward line-like continuity. This is particularly relevant for lane markings because the signal is often narrow, partially visible, and embedded in visually complex road texture.

### 3.4 LaneBiFPN Multi-Scale Fusion

The lane head needs both fine spatial detail and high-level context. Fine features help localize lane points, while deeper features help distinguish lane instances from road markings, shadows, and background edges. GCS-YOLO-Lane uses LaneBiFPN to fuse P2, P3, P4, and P5 features before query decoding.

Each input level is first projected to a shared channel dimension. Weighted fusion is then applied in both top-down and bottom-up directions. The resulting multi-scale feature maps preserve spatial structure while providing a unified representation for the GCS lane head. These fused feature maps are flattened into spatial tokens with positional information and level embeddings. The query decoder can therefore attend to lane evidence across multiple scales rather than relying on a single feature resolution.

### 3.5 Query-Based GCS Lane Head

The GCS lane head uses `Q=12` learnable lane queries to predict a set of candidate lane instances. Each query is intended to represent one possible lane, but the set is unordered before decoding. The transformer decoder updates the query embeddings by attending to the multi-scale spatial tokens from LaneBiFPN. The current implementation uses three decoder layers and eight attention heads.

In fixed-y mode, the head predicts x coordinates for the fixed y anchors. The y coordinates are stored as fixed anchors, so the point predictor only needs to estimate horizontal displacement for each of the `K=56` positions. A point MLP produces per-anchor x logits, and a reference-logit mechanism provides query-specific initial geometric patterns. The head also performs image-conditioned refinement by sampling features at predicted point locations and updating the x logits with local visual evidence.

The head jointly predicts lane existence and point visibility. The existence branch outputs one logit per query, indicating whether the query should be decoded as a lane instance. The point-validity branch outputs one logit for every query-anchor pair, indicating whether the corresponding point is visible. This visibility prediction is essential because a lane may be present even when only part of its fixed-y sequence is observable.

Auxiliary mask and edge branches are attached for training. The mask branch predicts a two-channel lane-region map, and the edge branch predicts a one-channel boundary map. These branches provide dense supervision for feature learning but are not used as final lane outputs.

### 3.6 Matching and Training Objective

Training uses Hungarian matching to connect predicted lane queries with ground-truth lanes. For each image, a cost matrix is built between `Q` predicted queries and the ground-truth lanes. The matching cost combines point distance, curve consistency, and lane-existence confidence:

```text
cost = cost_point * point_cost
     + cost_curve * curve_cost
     + cost_exist * exist_cost
```

The point cost is computed on visible ground-truth anchors, using an aspect-aware distance between predicted and ground-truth points. The curve cost compares second-order geometric behavior over visible triplets. The existence cost encourages high-confidence queries to be matched to real lanes.

After assignment, the loss supervises several aspects of the structured output. The existence loss trains matched and unmatched queries at the lane-instance level. The point loss trains geometry on visible anchors. The point-validity loss trains the visibility branch so that the decoder can distinguish visible and invisible sampled positions. Smoothness and curve losses regularize the ordered point sequence. Auxiliary mask and edge losses provide dense supervision at the feature level. Count-related or margin-related losses should be described in the final manuscript only if they are part of the frozen active-default experiment configuration; this draft does not treat default-disabled experimental knobs as core contributions.

## 4. Experiments

This section is intentionally deferred.

The final experimental section must be written only after the active default configuration has been frozen and evaluated under the project integrity rules. The section should include:

1. dataset and protocol details for TuSimple fixed-y K56 evaluation;
2. the exact active-default training command and `args.yaml`;
3. official-validation sweep setup and selected decode;
4. one-shot official test result for the selected candidate;
5. comparison against verified query/curve/sequence, anchor/row-based, and segmentation-based baselines;
6. ablations for LSEM, LaneBiFPN, the query head or visibility branch, and auxiliary supervision;
7. failure analysis for lane-count stability, short visible lanes, and visibility errors.

No numerical table is included in this draft because the user has chosen not to use legacy `gt4short15` as the main result and because the active-default experimental evidence is not yet complete.

## 5. Discussion

GCS-YOLO-Lane is best understood as a representation-level redesign of a YOLO-style lane detector. The method does not merely add a lane-specific postprocessor to a segmentation model. It changes the model output into a set of structured lane instances, where each instance is an ordered fixed-y point sequence with lane-level existence and point-level visibility. This makes the output closer to the geometric object expected by lane benchmarks and downstream driving modules.

The design also clarifies the role of dense supervision. Auxiliary mask and edge outputs are useful because lane markings are thin and benefit from pixel-level training signals. However, dense supervision is not the same as dense output. In GCS-YOLO-Lane, mask and edge prediction support the shared visual features, while the final decoded lanes come from query-owned structured sequences.

Compared with anchor/row-based methods, the proposed model keeps compatibility with fixed-y sampling but changes the instance organization. A row-based method often predicts lane positions row by row under a predefined lane or row structure. GCS-YOLO-Lane predicts a set of lane queries, each of which owns the full fixed-y sequence. This makes lane existence, geometry, and visibility part of a single candidate-lane representation.

Compared with query/curve/sequence methods, the distinction is architectural and representational. GCS-YOLO-Lane combines a YOLO11-style backbone, line-sensitive enhancement, multi-scale feature fusion, and fixed-y query decoding. Its output is neither a dense mask nor only a global curve parameter. It is a visibility-aware point sequence aligned with the sampling protocol.

The final discussion should be revised after experiments. If the active default configuration shows strong official-val and final-test behavior, this section can discuss empirical support for the structured design. If the results remain weak or inconsistent, the manuscript should report the limitation honestly or return to experimentation before submission.

## 6. Limitations and Future Work

The first limitation is empirical incompleteness. This draft does not yet contain active-default official-val selection, one-shot final-test evidence, verified baseline comparisons, or ablations. Therefore, it is not ready for submission.

The second limitation is benchmark scope. The current representation is written around the TuSimple fixed-y protocol. This is appropriate for the project target, but cross-dataset validation will be needed to show whether the method generalizes to datasets with different lane shapes, camera setups, or annotation conventions.

The third limitation is lane-count stability. Project notes show that lane-count behavior, especially around short or ambiguous side lanes, is a recurring risk. The final paper should include a failure analysis that separates geometry misses, low-confidence short lanes, duplicate-like predictions, and spurious extra lanes.

Future work should verify the method against strong structured and anchor/row-based baselines, test the design on additional lane datasets, and study whether query allocation and point visibility can be made more robust under heavy occlusion, worn markings, and dense multi-lane scenes.

## 7. Conclusion

This paper proposes GCS-YOLO-Lane, a YOLO11-based structured lane detection network. The key idea is to represent each lane as an ordered fixed-y point sequence and to predict candidate lane instances through learnable queries. The architecture combines line-sensitive feature enhancement, multi-scale LaneBiFPN fusion, and a GCS lane head that jointly predicts existence, point geometry, and point-level visibility. Training uses Hungarian matching and visibility-aware supervision, with auxiliary mask and edge branches used to support feature learning.

The final conclusion must be completed after the active default experimental evidence is fixed. At the current stage, the defensible conclusion is architectural: GCS-YOLO-Lane provides a structured alternative to mask-style lane outputs by directly predicting lane instance point sequences.

## Declarations To Complete Before Submission

- **Data availability**: Pending exact dataset access statement and TuSimple license wording.
- **Code availability**: Pending decision on whether and how the project code will be released.
- **Ethics declaration**: Likely no human-subject experiment; confirm dataset and autonomous-driving data-use requirements.
- **Conflict of interest**: Pending author confirmation.
- **Funding**: Pending author confirmation.
- **Author contributions**: Pending author list and CRediT roles.
- **AI assistance disclosure**: This draft was prepared with AI assistance and must be disclosed according to the selected venue policy.

## Reference Placeholders

Do not submit this section as a final reference list. Verify every item through the original paper, DOI, publisher page, arXiv page, or official code repository.

- `[REF: SCNN]` Segmentation/spatial CNN lane detection reference.
- `[REF: UFLD]` Ultra Fast Lane Detection.
- `[REF: UFLDv2]` Ultra Fast Lane Detection V2.
- `[REF: LaneATT]` Anchor-based attention lane detection.
- `[REF: CLRNet]` Cross-layer refinement lane detection.
- `[REF: LSTR]` Lane shape/transformer/query structured lane detection.
- `[REF: BezierLaneNet]` Bezier curve-based lane detection.
- `[REF: Lane2Seq]` Sequence-based lane detection.

## Claim-Evidence Map

| Claim | Evidence Source | Status |
|---|---|---|
| GCS-YOLO-Lane outputs structured lane point sequences rather than final masks. | Project contracts and model output shape. | Supported by project code/docs |
| The method uses `Q=12` lane queries and `K=56` fixed-y anchors. | `current-contracts.md`, model config, project summary. | Supported by project code/docs |
| LSEM uses strip responses, direction gating, coordinate reweighting, and dilated context. | Project summary and implementation description. | Supported by project code/docs |
| LaneBiFPN fuses P2-P5 features for the lane head. | Project summary and model YAML. | Supported by project code/docs |
| Hungarian matching supervises query-to-lane assignment. | `GCSHungarianMatcher` description. | Supported by project code/docs |
| GCS-YOLO-Lane outperforms any baseline. | Requires active-default official-val/test and verified baseline table. | Needs evidence |
| The method is competitive for Q3/Q4 submission. | Requires completed results, ablations, and venue fit. | Needs evidence |
| Related-work distinctions for LSTR, Lane2Seq, BezierLaneNet, UFLD/UFLDv2, LaneATT, CLRNet, and SCNN. | Requires source verification. | Needs literature verification |

## Self-Review Checklist

- **Contribution**: The architectural contribution is clear; empirical contribution is not yet claimable.
- **Writing clarity**: Terms are stable: fixed-y anchors, lane queries, point visibility, LSEM, LaneBiFPN, GCS lane head.
- **Experimental strength**: Not assessable until active-default experiments are complete.
- **Evaluation completeness**: Missing verified baseline table, ablations, and failure analysis.
- **Method soundness**: The representation and output contract are concrete; final credibility depends on evidence and fair comparison.
