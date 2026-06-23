# GCS-YOLO-Lane 论文详细大纲与证据地图

题目：**GCS-YOLO-Lane: A YOLO11-Based Structured Lane Detection Network**

工作流：`ars-outline`，academic-paper `outline-only` 模式。本文件是详细论文大纲与 evidence map，不是完整论文正文。

## 论文配置记录

| 参数 | 内容 |
|---|---|
| 论文类型 | IMRaD 风格计算机视觉实证论文 |
| 学科方向 | Computer Vision, autonomous driving perception, lane detection |
| 目标字数 | 正文 10000 英文字，参考文献和附录不计入 |
| 正文语言 | 英文 |
| 伴随版本 | 中文版与英文版分别保存 |
| 建议引用格式 | IEEE 风格，因目标领域为 CS/CV |
| 已有材料 | `docs/gcs-yolo-lane-paper-draft-non-experiment.zh.md`、`docs/project-summary-current.md`、`docs/agent-context/*`、项目代码/配置、`paper/` 本地文献 |
| 证据状态 | 方法和合同类 claim 已有本地代码/文档支撑；最终性能 claim 仍等待 active-default official-val 和一次性 final-test 证据 |
| 完整性边界 | 不在 test 上调参。threshold、checkpoint、postprocess 只用 official-val 选择。test 只用于已选 candidate 的一次性最终报告。 |
| 范围边界 | 本大纲描述 active 5-25-3 K56 structured-lane branch。后续 Count Head、Quality Head、Survival Head、near-miss、official-best 等机制不属于本文主线，除非未来明确重新引入。 |

## 中心论点

GCS-YOLO-Lane 将 YOLO-style 车道线感知重构为结构化 lane instance prediction。它不把车道线检测写成普通目标检测框，也不把最终输出写成 mask segmentation，而是预测一组 lane queries。每个 query 同时拥有 lane existence、fixed-y 有序点序列和逐点可见性。该设计让模型输出直接对齐 TuSimple-style evaluation 和下游 lane reasoning 所需的几何对象，同时保留 YOLO11-style backbone 的特征提取能力。

## 贡献规划

1. **结构化 lane 输出**：每条 lane 表示为 `K=56` fixed-y anchors 上的 visibility-aware sequence，并由 `Q=12` 个 learned lane queries 预测。
2. **YOLO11-based architecture adaptation**：将普通 YOLO box 或 mask 输出替换为结构化 GCS lane head。
3. **线敏感特征处理**：使用 LSEM 强化长、细、方向一致的 lane-like structure。
4. **多尺度 lane 特征融合**：使用 LaneBiFPN 融合 P2、增强后的 P3、增强后的 P4 和 P5。
5. **set-based structured supervision**：用 Hungarian matching、visibility-aware point supervision、curve/smoothness regularization 和 auxiliary mask/edge supervision 训练 lane queries。
6. **评估协议纪律**：定义不泄漏的 TuSimple fixed-y K56 evaluation protocol，严格区分 official-val selection 和 one-shot final test reporting。

性能优越性、SOTA、最终比较结论必须等待 active-default 结果、verified baseline table 和消融实验完成后再写。

## 10000 字正文结构建议

| 章节 | 目标字数 | 作用 |
|---|---:|---|
| Abstract | 250，不计入正文 | 总结问题、方法、证据边界和最终结果 |
| 1. Introduction | 1,200 | 引出 structured lane instance prediction 并列出贡献 |
| 2. Related Work | 1,800 | 对比 segmentation、row/anchor、query/curve、YOLO-style perception 方法 |
| 3. Method | 2,600 | 描述表示、架构、head、matching、loss 和 decode |
| 4. Experiments | 2,200 | 规划数据集、协议、baselines、主结果、消融、效率和失败分析 |
| 5. Discussion | 1,200 | 解释设计取舍和证据边界 |
| 6. Limitations and Future Work | 700 | 说明证据、数据集和鲁棒性局限 |
| 7. Conclusion | 300 | 以已验证的结构贡献收束 |
| 正文合计 | 10,000 | 不含 references、declarations 和 appendix |

## 详细大纲

### Abstract（约 250 words，不计入正文）

**目的**：在实验结果固定后给出紧凑摘要。

**应写内容**：
- 问题：lane detection 需要细长、连续、局部不可见的 instance-level geometry。
- 缺口：boxes 和最终 masks 不直接表示 ordered lane instances；许多方法依赖 postprocessing 或预设 anchors。
- 方法：GCS-YOLO-Lane 使用 YOLO11-style feature extractor、LSEM、LaneBiFPN 和 query-based fixed-y lane head。
- 输出：`pred_points`、`pred_logits`、`pred_valid_logits`、auxiliary mask/edge logits。
- 证据边界：active-default official-val 和 final-test 完成后再填入具体结果。

**证据**：project contracts、model YAML、GCS head 实现、official evaluation helpers。

**暂不写**：state-of-the-art、outperforms、具体排名。

### 1. Introduction（约 1,200 words）

#### 1.1 Lane Detection as Structured Geometry（250 words）

**目的**：说明 lane detection 不是普通 object detection。

**内容摘要**：
- 车道线细长、连续，并经常因遮挡、眩光、磨损或透视变化而局部不可见。
- 有用的 lane detector 不应只输出紧凑框或前景像素，而应恢复 lane instances、点序和可见范围。
- 引入本文视角：lane detection as structured instance prediction。

**证据来源**：
- SCNN：长连续 lane structure 需要 spatial relationship modeling。
- UFLD/UFLDv2：row-anchor formulation 证明 sparse fixed-row representation 的价值。
- LaneATT/CLRNet：lane localization 需要 global context 和 detailed features。

**过渡**：从任务结构转向现有表示的不足。

#### 1.2 Representation Gap in Existing Pipelines（300 words）

**目的**：解释为什么需要本文方法。

**内容摘要**：
- Segmentation-based 方法学习 dense lane pixels，但通常还需要 grouping、clustering、curve fitting 或 heuristic decoding 才能得到 lane instances。
- Row/anchor 方法高效且强，但 lane instance 组织仍与 predefined anchors 或 row classifiers 绑定。
- Query/curve/sequence 方法更接近 structured prediction，但如何构建 YOLO11-based structured lane detector 并直接输出 fixed-y point sequences 仍有空间。

**证据来源**：
- SCNN、LaneNetInstance、LaneAF 支撑 segmentation 和 instance grouping 线索。
- UFLD、UFLDv2、LaneATT、CLRNet 支撑 row/anchor/refinement 方法线索。
- LSTR、BezierLaneNet、FastDraw、PolyLaneNet 支撑 query、curve、sequence 输出线索。

**过渡**：引出 GCS-YOLO-Lane 作为 YOLO-style output 的结构化重构。

#### 1.3 GCS-YOLO-Lane Overview（300 words）

**目的**：说明本文网络做什么。

**内容摘要**：
- GCS-YOLO-Lane 保留 YOLO11-style feature extraction body，但用 structured lane head 替代普通 detection/mask output。
- 模型最多预测 `Q=12` 条 candidate lanes。
- 每条 lane 在 `K=56` 个 fixed-y anchors 上表示，对齐 TuSimple official h-samples `710` 到 `160`。
- 每个 query 输出 lane existence、fixed-y anchors 上的 x 坐标和 per-anchor visibility。

**项目证据**：
- `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`
- `data/tusimple_gcs_fixed_y_960x544.yaml`
- `docs/agent-context/current-contracts.md`

**过渡**：概述架构模块。

#### 1.4 Technical Contributions（250 words）

**目的**：给出具体贡献。

**内容摘要**：
- Structured fixed-y representation 和 label conversion。
- LSEM line-sensitive feature enhancement。
- LaneBiFPN multi-scale fusion。
- Query-based GCS lane head，包含 x-only fixed-y prediction 和 per-point visibility。
- Hungarian matching 与 visibility-aware loss design。
- Protocol-driven TuSimple evaluation workflow。

**注意**：在 active-default 结果完成前，贡献应写成 design and implementation contributions，避免性能优越性 claim。

#### 1.5 Paper Scope and Organization（100 words）

**目的**：明确 integrity boundary。

**内容摘要**：
- 本文聚焦 active 5-25-3 K56 branch。
- 后续实验机制除非在实验表中明确启用，否则不属于主线。
- test 结果只作为 official-val 选择后的 reporting-only 证据。

### 2. Related Work（约 1,800 words）

#### 2.1 Segmentation-Based Lane Detection（350 words）

**目的**：说明 dense-prediction 路线。

**内容摘要**：
- SCNN 提出 spatial message passing，用于长连续 lane structure。
- LaneNetInstance 和 LaneAF 通过 segmentation 或 affinity fields 建模 lane instances。
- Segmentation 和 dense supervision 很有用，但最终 lane output 通常需要 clustering、grouping、fitting 或 protocol conversion。

**核心来源**：
- Pan et al., SCNN, AAAI 2018。
- Neven et al., LaneNet instance segmentation, IV/arXiv 2018。
- Abualsaud et al., LaneAF, RA-L/ICRA 2021。

**与本文关系**：
- GCS-YOLO-Lane 只把 mask/edge branches 用作 auxiliary supervision，不把它们作为最终 lane output。

#### 2.2 Row-Based and Anchor-Based Lane Detection（400 words）

**目的**：对比高效 sparse-coordinate 方法。

**内容摘要**：
- UFLD 将 lane detection 改写为 row-based selecting，并使用 global features 和 structural loss。
- UFLDv2 扩展为 hybrid row/column anchors 和 ordinal classification。
- LaneATT 使用 line anchors 和 attention-guided feature aggregation。
- CLRNet 跨 feature levels refinement lane priors，并使用 ROIGather 和 Line IoU 改善 localization。

**核心来源**：
- Qin et al., UFLD, ECCV 2020。
- Qin et al., UFLDv2, TPAMI 2022。
- Tabelini et al., LaneATT, CVPR 2021。
- Zheng et al., CLRNet, CVPR 2022。

**与本文关系**：
- GCS-YOLO-Lane 共享 sparse y-sampled lane coordinates 的思想，但用 learned lane queries 组织完整 fixed-y sequence，而不是使用静态 lane anchors。

#### 2.3 Query-Based, Curve-Based, and Sequence-Based Methods（400 words）

**目的**：把本文定位到 structured-output 方法中。

**内容摘要**：
- LSTR 使用 transformer reasoning 和 Hungarian matching 预测 lane shape parameters。
- BezierLaneNet 用 Bezier curves 建模 compact holistic geometry。
- PolyLaneNet 和 FastDraw 探索 polynomial 或 sequence-style alternatives。
- 这些方法降低了对 final dense masks 的依赖，但 curve parameterization、sequence generation 和 anchor-free query design 各有优化与表示取舍。

**核心来源**：
- Liu et al., LSTR, WACV 2021。
- Feng et al., BezierLaneNet, CVPR 2022。
- Tabelini et al., PolyLaneNet, ICPR 2020。
- Philion, FastDraw, CVPR 2019。

**与本文关系**：
- GCS-YOLO-Lane 是 structured and query-based，但保留 direct fixed-y point sequence，而不是全局曲线系数向量。

#### 2.4 YOLO-Style Driving Perception and Multi-Task Networks（250 words）

**目的**：解释 YOLO architecture context，但不夸大 novelty。

**内容摘要**：
- YOLOP 和 YOLOPv2 展示了 YOLO-style shared encoders 支持 real-time driving perception 和多个 decoders 的可行性。
- 多数 YOLO-style lane branches 更偏 segmentation 或 multi-task perception components。
- GCS-YOLO-Lane 的区别是将 YOLO-style backbone 改造成 structured lane instance predictor。

**核心来源**：
- Wu et al., YOLOP, Machine Intelligence Research/arXiv。
- Han et al., YOLOPv2, arXiv 2022。
- 其他近期 YOLO-based lane detection/segmentation 论文需核验后再引用。

#### 2.5 Datasets and Evaluation Protocols（250 words）

**目的**：介绍 benchmark context 并解释为什么本文聚焦 TuSimple。

**内容摘要**：
- TuSimple 在 sampled horizontal positions 上评估 x 坐标，因此与 fixed-y output 高度兼容。
- CULane 和 CurveLanes 覆盖更多复杂或弯曲场景，可作为未来 cross-dataset validation。
- OpenLane、OpenLane-V2、ONCE-3DLanes 等 3D lane/topology 数据集不属于当前 2D TuSimple 论文范围，但可在 future work 中说明领域趋势。

#### 2.6 Positioning Summary（150 words）

**目的**：用准确 gap 收束相关工作。

**内容摘要**：
- 领域已有 segmentation、anchor/row、refinement、curve/query 等强方法。
- 本文针对的具体 gap 是：YOLO11-based structured lane detector，输出 query-owned fixed-y point sequences 和 explicit visibility。

### 3. Method（约 2,600 words）

#### 3.1 Problem Formulation and Output Contract（300 words）

**目的**：定义任务和张量。

**内容摘要**：
- 输入：RGB TuSimple image，resize 为 `--imgsz 544 960`，顺序为 H,W。
- 输出：最多 `Q=12` 个 lane queries。
- 默认 K56 输出形状：
  - `pred_points: B x 12 x 56 x 2`
  - `pred_logits: B x 12`
  - `pred_valid_logits: B x 12 x 56`
  - `aux_mask_logits: B x 2 x H x W`
  - `aux_edge_logits: B x 1 x H x W`
- 解释 fixed-y mode 下 `pred_points[..., 1]` 是固定 y，x 由模型预测。

**证据**：
- `docs/agent-context/current-contracts.md`
- `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`
- `ultralytics/nn/modules/gcs_lane.py`

#### 3.2 Fixed-Y Label Representation and Dataset Conversion（400 words）

**目的**：解释 label 如何构建。

**内容摘要**：
- K56 labels 由原始 TuSimple JSON 和 images 重新生成，不从 K32 resample。
- anchors 按原始图像高度归一化：`710/720` 到 `160/720`，原始 y 像素步长 10。
- 每条 GT lane 在 fixed-y anchors 上插值得到 x。
- 有效 anchor 保存 normalized `(x, y)` 且 `lane_valid=1`；无效 anchor 保留 fixed y，`x=0` 且 `lane_valid=0`。
- `.npz` 包含 `semantic_mask`、`edge_mask`、`lanes`、`lane_valid`、`num_lanes`、`point_mode`、`fixed_y`、`raw_file`、`image_shape`、`num_points`。

**证据**：
- `tools/convert_tusimple_to_gcs.py`
- `gcs_tools/label_utils.py`
- `data/tusimple_gcs_fixed_y_960x544.yaml`

**图建议**：
- Figure 2：从底到顶的 56 个 fixed-y anchors 示意。

#### 3.3 Overall Architecture（350 words）

**目的**：给出完整模型流程。

**内容摘要**：
- YOLO11-style backbone 提取 P2-P5 features。
- LSEM 插入 P3 和 P4 后。
- LaneBiFPN 融合 P2、enhanced P3、enhanced P4 和 P5。
- GCSLaneHead 将融合特征展平为 multi-scale spatial tokens，加入 position/level embeddings，并 decode lane queries。
- Auxiliary mask 和 edge branches 只用于训练监督。

**证据**：
- model YAML backbone/head definitions。
- `ultralytics/nn/modules/gcs_lane.py`。

**图建议**：
- Figure 1：backbone、LSEM、LaneBiFPN、GCS head 和 outputs 架构图。

#### 3.4 Line-Sensitive Feature Enhancement Module（350 words）

**目的**：解释 LSEM。

**内容摘要**：
- LSEM 包含 LineStripAttention、coordinate reweighting、dilated context、residual connection 和 activation。
- LineStripAttention 使用 horizontal/vertical strip depthwise convolutions 和 direction gate。
- Coordinate reweighting 保留 height/width spatial cues。
- Dilated context branch 扩大 receptive field。

**证据**：
- `ultralytics/nn/modules/gcs_lane.py` 中 `CoordReweight`、`LineStripAttention`、`LSEM`。
- 相关工作支持：SCNN 和 RESA 强调 long-range structured lane context 的重要性。

**消融要求**：
- active-default official-val 上比较 with/without LSEM。

#### 3.5 LaneBiFPN Multi-Scale Fusion（300 words）

**目的**：解释为什么需要 multi-scale fusion。

**内容摘要**：
- 细粒度 features 有助于 point localization；high-level features 有助于区分 lane markings 与相似道路结构。
- LaneBiFPN 对齐 P2-P5 channels，并使用 bidirectional weighted fusion。
- 输出统一多尺度表示，供 query decoder 使用。

**证据**：
- model YAML：`LaneBiFPN [128]` over `[2, 5, 8, 12]`。
- `LaneBiFPN` 和 `WeightedFusion` classes。
- 相关工作支持：CLRNet 强调 high/low feature complementarity。

#### 3.6 Query-Based GCS Lane Head（450 words）

**目的**：解释 structured decoder。

**内容摘要**：
- `Q=12` learned lane queries 通过 transformer decoder attend to multi-scale spatial tokens。
- 当前实现使用 3 decoder layers 和 8 attention heads。
- fixed-y mode 下，head 预测 56 个 anchors 的 x logits；y anchors 来自 registered fixed-y buffer。
- reference-logit mechanism 为不同 query 提供不同初始几何模式。
- image-conditioned refinement 在预测点位置采样特征，细化 x logits 和 point-valid logits。
- head 同时预测 lane existence 和 per-point visibility。

**证据**：
- `GCSLaneHead` class。
- model YAML final layer。

**图建议**：
- Figure 3：query-owned fixed-y point sequence 与 visibility branch。

#### 3.7 Matching and Training Objective（450 words）

**目的**：描述监督。

**内容摘要**：
- Hungarian matching 按图像把 predicted queries 分配给 GT lanes。
- Matching cost 包含 point cost、curve cost 和 existence cost。
- Losses 包括 lane existence、visible-point geometry、point visibility、smoothness、curve regularization、auxiliary mask、auxiliary edge。
- 已记录但默认关闭的实验项必须谨慎写：count loss、under-5 count loss、duplicate margin、spurious margin、lane-balanced point、short-valid recall 不是核心贡献，除非在某个实验配置中明确启用并报告。
- Auxiliary mask/edge branches 支持 feature learning，但不是最终输出。

**证据**：
- `ultralytics/utils/gcs_matcher.py`
- `ultralytics/utils/gcs_loss.py`
- `ultralytics/cfg/default.yaml`

#### 3.8 Decoding and TuSimple Conversion（350 words）

**目的**：解释 inference 和 metric conversion。

**内容摘要**：
- Decode 依据 query existence score、point-valid threshold、longest contiguous visible span、`min_points`、optional lane NMS 和 `max_det`。
- decoded GCS lanes 用 visible points 和 interpolation 转为 TuSimple h-samples。
- 最终 lanes 在需要时按 bottom x 从左到右排序。
- inference/decode 不得使用 GT。

**证据**：
- `ultralytics/utils/gcs_postprocess.py`
- `gcs_tools/tusimple_official_eval.py`
- `tools/eval_tusimple_official.py`

### 4. Experiments（约 2,200 words）

本章目前是计划，不应在 active-default training、official-val selection、one-shot final test、verified baselines 和 ablations 完成前定稿。

#### 4.1 Dataset and Protocol（350 words）

**目的**：定义 evaluation surface。

**内容摘要**：
- Dataset：TuSimple fixed-y K56 converted dataset。
- Splits：train 3263、val 363、test 2782。
- Evaluation：official TuSimple Accuracy、FP、FN，并可报告项目 `official_score`。
- Selection：只用 official-val。
- Final test：selected decode 的一次性 reporting。

**证据**：
- `docs/agent-context/project-context.md`
- `docs/agent-context/experiment-rules.md`
- `tools/sweep_tusimple_official.py`
- `tools/eval_tusimple_official.py`

#### 4.2 Implementation Details（250 words）

**目的**：保证可复现。

**内容摘要**：
- Model config、data config、image size。
- Training environment：formal training 使用 remote RTX 4090，batch 32 起步。
- Optimizer 和 schedule 来自冻结 active-default `args.yaml`。
- 如使用 pretrained weights，说明来源。

**所需证据**：
- Active-default run `args.yaml`。
- Formal training command。

#### 4.3 Main Results（450 words）

**目的**：只报告 protocol-valid performance。

**必需表格**：
- Table 1：TuSimple official-val selected row。
- Table 2：使用 selected row 的 one-shot official test report。
- Table 3：verified baselines comparison。

**必需字段**：
- `official_acc`、`official_FP`、`official_FN`、`official_score`、`count_acc`、`avg_total_ms`。
- selected `conf`、`point_valid_thr`、`nms_dist_px`、`max_det`、`min_points`。

**当前状态**：
- active-default 主结果待补。
- Legacy 或 reporting-only 结果只能作为历史说明，不能作为 primary claims。

#### 4.4 Baseline Comparison Plan（300 words）

**目的**：定义公平比较。

**候选 baseline families**：
- Segmentation/context：SCNN、RESA、SAD。
- Row/anchor：UFLD、UFLDv2、LaneATT、CLRNet。
- Query/curve/sequence：LSTR、PolyLaneNet、BezierLaneNet。
- YOLO-style multi-task perception：YOLOP、YOLOPv2 更适合作为 context references；除非能核验可比 TuSimple 2D lane metrics，否则不直接作为主 baseline。

**规则**：
- 只使用核验后的 official reported metrics。
- TuSimple、CULane、LLAMAS、CurveLanes 指标分开写。
- 不同硬件 FPS 不能直接混比，必须加 caveat。

#### 4.5 Ablation Study Plan（450 words）

**目的**：把方法 claim 和证据绑定。

**应运行或报告的消融**：
- Baseline head without LSEM。
- Baseline head without LaneBiFPN 或 simpler feature fusion。
- GCS head without fixed-y refinement。
- Without point-validity branch。
- Without auxiliary mask/edge supervision。
- Alternative `K` 或 anchor range 只有在协议允许且重新生成 labels 时才能做。
- Count/margin experimental knobs 只有在明确启用时，放入单独标注的实验小节。

**注意**：
- 所有 ablation 都必须用 official-val 选择，不得从 final test 调参。

#### 4.6 Efficiency and Complexity（200 words）

**目的**：说明部署相关性。

**内容摘要**：
- 报告参数量、FLOPs/MACs、throughput、preprocessing、inference、decode、total time。
- 只有硬件匹配或明确说明硬件差异时才做速度比较。

**所需证据**：
- Model profiling command output。
- Evaluation summary timing fields。

#### 4.7 Failure Analysis（200 words）

**目的**：诚实解释剩余弱点。

**内容摘要**：
- 分析 lane-count errors、short visible lanes、low-score short GT、geometry misses、spurious extras、duplicate-like extras、`GT4`/`GT5` confusion。
- 只用 train/val diagnostics 和 official-val 进行迭代分析。
- final-test breakdown 只能 reporting-only。

**证据来源**：
- `docs/agent-context/known-bottlenecks.md`
- 未来 active-default train/val failure traces。

### 5. Discussion（约 1,200 words）

#### 5.1 What Structured Output Changes（300 words）

**目的**：解释 representation contribution。

**内容摘要**：
- 模型输出是 lane instance object，而不是 mask proxy。
- Query ownership 将 existence、geometry 和 visibility 绑定到同一个 candidate。
- Fixed-y anchors 简化 evaluation conversion，同时保留 point visibility。

#### 5.2 Tradeoff Against Segmentation and Row/Anchor Methods（300 words）

**目的**：解释收益和代价。

**内容摘要**：
- 相比 segmentation，GCS-YOLO-Lane 降低对 post-hoc instance grouping 的依赖。
- 相比 row/anchor 方法，GCS-YOLO-Lane 使用 learned queries 组织 lane candidates。
- 代价是需要稳定的 query allocation、count calibration 和 robust visibility prediction。

#### 5.3 Role of Auxiliary Dense Supervision（200 words）

**目的**：澄清 mask/edge branches 的角色。

**内容摘要**：
- Dense supervision 对 feature learning 有用。
- 它不定义最终输出。
- 这个区分能避免把方法误写成普通 segmentation pipeline。

#### 5.4 Evidence Boundary and Research Integrity（200 words）

**目的**：防止过度 claim。

**内容摘要**：
- official-val 选择 candidate 和 decode。
- test 只报告一次。
- historical near-miss 或 reporting-only runs 不能用于 promote method。

#### 5.5 Practical Implications（200 words）

**目的**：说明下游价值。

**内容摘要**：
- Ordered point sequences 比 raw masks 更容易用于 lane tracking、planning、map alignment 和 curve fitting。
- Point visibility 支持遮挡场景中的 partial-lane reasoning。

### 6. Limitations and Future Work（约 700 words）

#### 6.1 Experimental Evidence Not Yet Complete（150 words）

说明 active-default results、verified baselines 和 ablations 未完成前，不能写最终 superiority claims。

#### 6.2 TuSimple-Centric Fixed-Y Design（150 words）

说明该表示与 TuSimple 高度对齐，但还需要 CULane、CurveLanes、LLAMAS 等数据集验证泛化。

#### 6.3 Lane-Count Stability（150 words）

讨论 short visible side lanes、GT4/GT5 count confusion、duplicate-like extras、spurious extras 等已知风险。

#### 6.4 2D Scope（100 words）

说明 3D lane detection、topology reasoning、map learning、temporal consistency 不属于当前论文范围。

#### 6.5 Future Work（150 words）

提出 cross-dataset validation、更强 active-default ablations、temporal extension、3D adaptation、更原则化的 query allocation/count calibration。

### 7. Conclusion（约 300 words）

**目的**：用最强可支持 claim 收束。

**内容摘要**：
- GCS-YOLO-Lane 是 YOLO11-based structured lane detector。
- 它直接预测 query-owned fixed-y point sequences，并带有 lane existence 和 point visibility。
- LSEM、LaneBiFPN、GCS head、Hungarian matching、auxiliary dense supervision 构成核心方法。
- 最终经验结论只能在 active-default evaluation 完成后填入。

## Evidence Map

### 证据状态说明

| 状态 | 含义 |
|---|---|
| Local-code verified | 由本仓库项目代码、配置或文档支撑 |
| Local-literature supported | 由 `paper/` 下本地 markdown/PDF metadata 支撑，但最终 citation details 仍需核验 |
| Pending experiment | 需要 active-default official-val/test、ablation、timing 或 failure-analysis output |
| Do not claim | 当前证据边界下不支持 |

### 方法 Claim Map

| Claim | 计划章节 | 证据来源 | 状态 | 注意 |
|---|---|---|---|---|
| 模型输出 structured lane point sequences，不是 YOLO boxes 或 final masks。 | 1.3, 3.1 | `gcs-yolo-lane-s.yaml`、`current-contracts.md`、`GCSLaneHead` output dict | Local-code verified | auxiliary mask/edge branches 存在，但不是最终 lane output。 |
| TuSimple 输入使用 `--imgsz 544 960`，顺序为 H,W。 | 3.1, 4.1 | `AGENTS.md`、`current-contracts.md`、data YAML | Local-code verified | 不得写成 W,H。 |
| 默认 K56 合同使用 `Q=12`、`K=56`、`fixed_y=710/720 -> 160/720`。 | 1.3, 3.1, 3.2 | model YAML、data YAML、current contracts | Local-code verified | K56 labels 必须由原始 TuSimple JSON/images 生成。 |
| Labels 包含 masks、edge masks、lanes、lane_valid、point_mode、fixed_y、image shape、raw_file。 | 3.2 | `tools/convert_tusimple_to_gcs.py` | Local-code verified | 公开数据 artifact 中需再次确认 exact fields。 |
| LSEM 使用 line-strip attention、direction gating、coordinate reweighting、dilated context 和 residual activation。 | 3.4 | `ultralytics/nn/modules/gcs_lane.py` | Local-code verified | 没有消融前不能 claim accuracy contribution。 |
| LaneBiFPN 融合 P2、enhanced P3、enhanced P4 和 P5。 | 3.5 | model YAML、`LaneBiFPN` implementation | Local-code verified | 需要消融支撑经验贡献。 |
| GCSLaneHead 使用 learned lane queries 和 transformer decoder。 | 3.6 | `GCSLaneHead` implementation | Local-code verified | 不能声称 first query-based lane detector；已有 LSTR 等相关工作。 |
| fixed-y mode 只预测 x，y anchors 固定。 | 3.1, 3.6 | `GCSLaneHead`、fixed-y buffer construction | Local-code verified | 解释输出仍是 2D points，因为 y 由 anchors 恢复。 |
| Hungarian matching 将 predictions 分配给 GT lanes。 | 3.7 | `gcs_matcher.py`、`gcs_loss.py` | Local-code verified | 与 LSTR/DETR-style set prediction 对比时要加引用。 |
| Decode 使用 query score、point-valid threshold、longest contiguous visible span、min points、optional NMS、max_det。 | 3.8, 4.1 | `gcs_postprocess.py` | Local-code verified | postprocess 参数只能由 official-val 选择。 |
| TuSimple official-val sweep 禁止 `--split test` 做 threshold search。 | 4.1, 5.4 | `tools/sweep_tusimple_official.py`、experiment rules | Local-code verified | selected row 固定前保持 final test closed。 |

### 文献 Evidence Map

| 主题 | 本地文献来源 | 论文用途 | 状态 |
|---|---|---|---|
| Lane continuity and spatial context | SCNN、RESA、SAD | 支撑 line-sensitive/context-aware feature modules 的动机 | Local-literature supported |
| Segmentation and instance grouping | LaneNetInstance、LaneAF、SCNN | 解释 dense-output lineage 和 postprocessing needs | Local-literature supported |
| Row-based sparse representation | UFLD、UFLDv2、E2ELMD | 说明 fixed-y/row-sampled output 高效且对齐 benchmark | Local-literature supported |
| Anchor-based lane detection | LaneATT、CLRNet、Line-CNN references via CLRNet/LaneATT | 对比 anchor ownership 与 learned query ownership | Local-literature supported |
| Query and transformer lane prediction | LSTR、Laneformer、LDTR、SparseLaneformer | 定位 GCS head 与 structured query-like methods 的关系 | Local-literature supported |
| Curve-based lane output | PolyLaneNet、BezierLaneNet、BezierFormer | 对比 point-sequence output 与 global curve parameters | Local-literature supported |
| Sequence/keypoint alternatives | FastDraw、PINet、GANet、LanePtrNet | 讨论非 mask 的 structured alternatives | Local-literature supported |
| YOLO-style driving perception | YOLOP、YOLOPv2、A-YOLOM、YOLOMH、Q-YOLOP | 提供 YOLO-style shared-backbone perception context | Local-literature supported |
| Dataset scope | TuSimple、CULane、CurveLanes、LLAMAS、OpenLane、OpenLane-V2 | 解释当前 TuSimple focus 与未来 cross-dataset directions | Local-literature supported |

### 实验证据地图

| 所需证据 | 计划表/图 | 所需 artifact | 当前状态 |
|---|---|---|---|
| Active-default training configuration | Training details table | active-default `args.yaml` | Pending experiment |
| Official-val selected row | Selected validation decode table | active-default run 的 `tusimple_official_sweep_summary.json` | Pending experiment |
| One-shot final test | Official test result table | selected row 对应 `tusimple_official_summary.json` | Pending experiment |
| Baseline comparison | TuSimple comparison table | 已核验论文和 metric extraction | Pending verification |
| LSEM contribution | Ablation table | without LSEM 的 active-default ablation run | Pending experiment |
| LaneBiFPN contribution | Ablation table | simpler fusion 的 active-default ablation | Pending experiment |
| Query/visibility design | Ablation table | head/visibility branch variants | Pending experiment |
| Auxiliary mask/edge role | Ablation table | without mask/edge supervision | Pending experiment |
| Failure modes | Error taxonomy figure/table | train/val failure trace、official-val diagnostics | Pending experiment |
| Efficiency | Params/FLOPs/FPS/time table | model profiling 和 eval timing | Pending experiment |

### 目前不能写的 Claim

| Claim | 原因 |
|---|---|
| GCS-YOLO-Lane 在 TuSimple 上达到 state-of-the-art。 | active-default official-val/test 和 verified baselines 未完成。 |
| GCS-YOLO-Lane 超过 SCNN、UFLD、LaneATT、CLRNet、LSTR 或 BezierLaneNet。 | 需要公平、citation-verified baseline table 和 comparable metrics。 |
| LSEM 或 LaneBiFPN 提升 accuracy。 | 需要 ablation evidence。 |
| 默认关闭的 margin/count/short-lane losses 是核心方法贡献。 | 它们是 experiment knobs，除非在 reported configuration 中启用。 |
| reporting-only final-test near-miss results 可以选择 candidate。 | 项目规则只允许 official-val selection。 |

## 推荐图表

| ID | 类型 | 内容 | 证据来源 |
|---|---|---|---|
| Fig. 1 | Architecture | YOLO11-style backbone、LSEM、LaneBiFPN、GCS head、outputs | model YAML and code |
| Fig. 2 | Representation | K56 fixed-y anchors and point visibility | data YAML, conversion code |
| Fig. 3 | Head detail | Learned lane queries and x-only fixed-y prediction | `GCSLaneHead` |
| Fig. 4 | Training | Hungarian matching and loss components | matcher/loss code |
| Table 1 | Method contract | Input/output/loss/data contracts | `current-contracts.md` |
| Table 2 | Literature matrix | Baseline families and representation types | `paper/papers_metadata.xlsx` |
| Table 3 | Main result | official-val and one-shot test | pending active-default artifacts |
| Table 4 | Ablation | LSEM、LaneBiFPN、head/visibility、auxiliary supervision | pending ablation artifacts |
| Table 5 | Failure analysis | count、short-lane、spurious/duplicate、visibility errors | pending diagnostics |

## 下一版正文写作注意事项

- 统一使用 "structured lane detection"。
- K56 表示使用 "fixed-y anchors"，把 "row anchors" 留给已有文献。
- 写 "auxiliary mask/edge supervision"，避免暗示最终输出是 segmentation。
- 结果语言在证据完成前保持条件式。
- 本地文献最终引用前必须核验，不只依赖文件名。
- 所有 TuSimple commands 保持 `--imgsz 544 960` 的 H,W 顺序。
