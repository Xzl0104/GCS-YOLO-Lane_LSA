# GCS-YOLO-Lane Research Paper Plan (English Version)

Updated on 2026-06-23

## 1. Recommended Paper Title

**GCS-YOLO-Lane: A YOLO11-Based Structured Lane Detection Network**

This title deliberately foregrounds the algorithmic contribution: a structured lane detection network. It does not frame the paper primarily as a TuSimple fixed-y reproducibility report.

## 2. Confirmed Writing Decisions

| Decision Item | Confirmed Direction |
|---|---|
| Main claim | Emphasize the new algorithmic structure |
| Core positioning | A YOLO11-based structured lane detection network named GCS-YOLO-Lane |
| Main experiment line | Use results from the current active default configuration |
| Legacy `gt4short15` | Do not use as the main paper result; keep only as historical/internal context |
| If final-test results do not exceed older reports | Continue experiments before writing the formal result section |
| Main related-work comparison | Query / structured / curve / sequence methods |
| Strong baselines | Anchor / row-based methods |
| Supplementary baselines | Segmentation-based methods |
| Target venue | A Q3/Q4 English journal, exact venue not yet selected |

## 3. Revised Research Question and Thesis

### Research Question

**How can a YOLO11-based network be redesigned for structured lane detection by representing each lane as an ordered point sequence and directly regressing a set of candidate lane instances through a query-based detection head?**

### Working Thesis Statement

GCS-YOLO-Lane adapts YOLO11 into a structured lane detection network: each lane is represented as an ordered fixed-y point sequence, and a query-based detection head directly predicts a set of candidate lane instances with lane-level existence and point-level visibility. Auxiliary mask and edge outputs are used for training supervision, not as the final lane representation.

## 4. Draft Contributions

1. **Structured lane representation**: Each lane is represented as an ordered point sequence on fixed-y anchors rather than as a detection box or final segmentation mask.
2. **Query-based structured detection head**: `Q=12` learnable lane queries directly predict candidate lane instances, including lane existence, `K=56` point locations, and point-wise visibility.
3. **Line-sensitive feature enhancement**: `LSEM` introduces horizontal/vertical strip responses, direction gating, coordinate reweighting, and dilated context into a YOLO11-style backbone.
4. **Multi-scale lane feature fusion**: `LaneBiFPN` fuses P2-P5 features to provide multi-scale spatial tokens for the structured lane head.
5. **Training and supervision design**: Hungarian matching connects queries with GT lanes, and training combines point regression, visibility, smoothness, curve, auxiliary mask, and edge supervision.

The reproducible TuSimple official-val/test protocol remains an important credibility support, but it should not be the title-level contribution.

## 5. Experimental Writing Boundaries

This paper plan now uses the active default configuration as the main result line. Therefore:

- The legacy `gt4short15` run must not be used as the selected main paper result.
- The reporting-only final-test ACC from `dupmargin005` must not be used for model selection.
- Legacy mechanisms such as `--gcs-gt4-short-*`, `extra_exist_loss`, and short matched existence floor should not be mixed into the active default algorithm claim.
- Formal writing should wait until official-val selection and one-shot final-test reporting are complete for the active default configuration.
- If the active default configuration is not strong enough for a Q3/Q4 submission, the next step should be additional experiments rather than writing around the old final-test gap as a limitation.

## 6. Related Work Comparison Framework

### 6.1 Main Comparison: Query / Structured / Curve / Sequence Methods

Purpose: Show that the core design of GCS-YOLO-Lane is a structured query point-sequence representation.

Candidate methods:

- LSTR
- Lane2Seq
- BezierLaneNet

Verification needed:

- Whether each method is query-based, sequence-based, curve-based, or structured prediction.
- Whether comparable TuSimple official val/test results are available.
- Whether input size, postprocessing, and evaluation protocol are comparable.

### 6.2 Strong Baselines: Anchor / Row-Based Methods

Purpose: Show that GCS-YOLO-Lane remains competitive under the TuSimple fixed-y sampling protocol.

Candidate methods:

- UFLD / UFLDv2
- LaneATT
- CLRNet

Verification needed:

- Whether TuSimple numbers are official test results.
- Whether `ACC/FP/FN` are reported.
- Whether the comparison is fair against a fixed-y point-sequence representation.

### 6.3 Supplementary Baselines: Segmentation-Based Methods

Purpose: Historical reference only, not the core argument.

Candidate method:

- SCNN

Verification needed:

- Whether it is appropriate as the representative segmentation-based reference.
- Whether it should appear in the main quantitative table or only in Related Work.

## 7. Proposed Paper Structure (about 6000 words, excluding references)

### Abstract (about 200-250 words)

Do not draft the full abstract yet. The abstract should wait until the active default configuration has final official-val and one-shot final-test results.

### 1. Introduction (about 850 words)

- Core argument: YOLO-style segmentation output is not the same as structured lane instance output; lanes are better modeled as ordered point sequences.
- Problem framing: Lane markings are elongated, continuous, instance-level structures with incomplete visibility, so mask outputs require additional postprocessing to form lane instances.
- Proposed answer: GCS-YOLO-Lane directly regresses candidate lane point sequences through a query-based head.
- Contribution emphasis: A structured network design, not merely an evaluation protocol.
- Risk: If related work already contains strong query/sequence methods, novelty must be narrowed to the combination of YOLO11-based structured network design, LSEM/LaneBiFPN, and the GCS head.

### 2. Related Work (about 1200 words)

Organize this section by argumentative role rather than listing methods chronologically:

1. Query / structured / curve / sequence lane detection: LSTR, Lane2Seq, BezierLaneNet, etc.
2. Anchor / row-based lane detection: UFLD/UFLDv2, LaneATT, CLRNet, etc.
3. Segmentation-based lane detection: SCNN as a historical route.
4. Gap statement: Existing methods either rely on segmentation/row-anchor formulations or do not fully combine structured lane output with a YOLO-style backbone and line-sensitive feature enhancement. GCS-YOLO-Lane adapts YOLO11 into a network that directly outputs lane point sequences.

All papers and metrics must be verified through source lookup before citation.

### 3. Method (about 1700 words)

Suggested subsections:

1. **Overall Architecture**: YOLO11-style backbone + LSEM + LaneBiFPN + GCSLaneHead.
2. **Structured Lane Representation**: Each lane is represented as an ordered point sequence on `K=56` fixed-y anchors.
3. **Line-Sensitive Enhancement Module**: Explain horizontal/vertical strip attention, direction gating, coordinate reweighting, and dilated context.
4. **Query-Based GCS Lane Head**: Explain `Q=12` lane queries, existence logits, point logits, visibility logits, and image-conditioned refinement.
5. **Training Objective and Matching**: Explain Hungarian matching, point/curve/existence costs, and point, visibility, smoothness, curve, mask, and edge losses.

Method-section anchor sentence:

> GCS-YOLO-Lane treats lane detection as structured instance prediction: each learnable query predicts one candidate lane as an ordered fixed-y point sequence with query-level existence and point-level visibility.

### 4. Experiments (about 1400 words)

This section should use the active default configuration as the only main line.

Before formal drafting, collect:

1. Training command and `args.yaml` for the active default configuration.
2. Official-val sweep for the active default configuration.
3. Official-val selected decode.
4. One-shot final-test result.
5. Verified comparison table against the candidate baselines.
6. Necessary ablations, preferably covering LSEM, LaneBiFPN, query head / visibility, and auxiliary supervision.

Do not place legacy `gt4short15` in the main result table as the selected result. It can be kept in internal notes or an appendix caveat only if clearly separated from the active default configuration.

### 5. Discussion (about 650 words)

The discussion should focus on whether the structured network design is justified:

- Compared with segmentation output, GCS directly produces lane instances and point-level visibility.
- Compared with anchor/row-based methods, GCS still fits the TuSimple fixed-y sampling setting but organizes lanes as query set prediction.
- Compared with sequence/curve/query methods, the distinction is the combination of YOLO11 adaptation, LSEM/LaneBiFPN, and the fixed-y query head.
- If active default results are not strong enough, the manuscript should return to experimentation rather than force a submission-ready conclusion.

### 6. Conclusion (about 300 words)

The conclusion should emphasize:

- GCS-YOLO-Lane as a YOLO11-based structured lane detection network.
- The shift from mask/box-style output to query-based ordered point-sequence prediction.
- Validation under the TuSimple fixed-y contract.
- Future work on count stability, cross-dataset validation, and stronger query/sequence baseline comparisons.

## 8. Next Experiment and Writing Gates

Before moving into full manuscript drafting, complete:

1. **Active default result confirmation**: Identify the run that truly represents the active default configuration and does not include legacy `gt4short15`-only parameters.
2. **Official-val selection**: Use official-val only for checkpoint, threshold, and postprocess selection.
3. **One-shot final-test reporting**: Run final test only once for the official-val-selected candidate.
4. **Baseline literature matrix**: Verify papers, years, metrics, and protocols for LSTR, Lane2Seq, BezierLaneNet, UFLD/UFLDv2, LaneATT, CLRNet, and SCNN.
5. **Venue shortlist**: Later verify Q3/Q4 journals against the current journal-ranking system you want to use. Do not hard-code candidate journals before checking current rankings.

## 9. Best Immediate Next Step

The next step should not be a full paper draft yet. Create two working files first:

1. `docs/gcs-yolo-lane-paper-experiment-gap.md`: what active default results are still missing and which historical results cannot be used.
2. `docs/gcs-yolo-lane-related-work-matrix.md`: WHY/HOW/WHAT, TuSimple metrics, comparability, and citation status for candidate baselines.

After these two files are complete, the Abstract, Introduction, and Method sections can be drafted with a much stronger foundation.
