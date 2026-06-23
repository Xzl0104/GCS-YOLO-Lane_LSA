# GCS-YOLO-Lane：一种基于 YOLO11 的结构化车道线检测网络

草稿状态：仅完成非实验部分。

本文件是中文版本，用于和英文稿对应阅读。实验结果、数值对比、最终性能主张暂不写入。引用统一保留为 `[REF: ...]` 占位符，后续必须核验原论文、DOI、arXiv 页面或官方代码。主实验线按当前 active default configuration 处理，不使用 legacy `gt4short15` 或 reporting-only final-test 结果作为正文主结果。

## 迷你大纲

- **论文主线**：车道线检测应写成结构化 lane instance prediction，而不是普通检测框或最终分割 mask。
- **技术核心**：GCS-YOLO-Lane 将 YOLO11-style 网络改造成 fixed-y、query-based 的结构化检测器，直接输出有序车道线点序列。
- **方法模块**：fixed-y 结构化表示、LSEM 线敏感特征增强、LaneBiFPN 多尺度融合、GCS lane head、Hungarian matching 和 visibility-aware supervision。
- **证据边界**：方法与实现由项目代码和 contract 支撑；性能优越性必须等待 active-default official-val/test 结果。
- **投稿边界**：需要补齐实验、baseline 文献矩阵、消融和引用信息后，才能进入投稿版本。

## 摘要

道路场景中的车道线检测需要对细长、连续且经常局部不可见的标线进行实例级几何建模。普通目标检测框难以表达这种结构，分割式方法虽然能够预测车道线区域，但通常还需要额外后处理才能形成有序车道线实例。本文提出 **GCS-YOLO-Lane**，一种基于 YOLO11 的结构化车道线检测网络。该方法将每条车道线表示为固定 y 锚点上的有序点序列，并通过可学习 lane query 直接预测候选车道线集合。网络由 YOLO11-style 特征提取器、线敏感特征增强模块、多尺度 LaneBiFPN 和 query-based lane head 组成，同时预测 lane existence、点坐标和逐点可见性。训练阶段使用 Hungarian matching 连接预测 query 与真实车道线，并联合监督车道线存在性、点几何、可见性、曲线正则、平滑性以及辅助 mask/edge 分支。最终实验结论将在 active default configuration 完成 official-val 选择和一次性 final-test 报告后补入。当前版本的核心结论是结构设计层面的：GCS-YOLO-Lane 将车道线输出从分割区域转为结构化 lane instance 点序列。

**关键词**：车道线检测；结构化预测；YOLO11；query-based detection；fixed-y representation；TuSimple

## 1. 引言

车道线检测是智能车辆感知系统中的基础任务，因为车道线为定位、路径规划和道路场景理解提供了关键空间约束。与通用目标检测不同，车道线检测的目标输出不是围绕紧凑物体的矩形框。车道线是细长的曲线状结构，其可见部分可能受到遮挡、磨损、阴影和视角变化影响。因此，车道线检测器不仅需要判断车道线是否存在，还需要恢复车道线实例、保持点序顺序，并识别每条车道线上哪些位置实际可见。

现有视觉感知系统常采用检测框、分割 mask、anchor 或 row-wise classification 等表示方式。这些表示具有工程上的便利性，但并不总是直接对应车道线检测的最终目标。分割模型可以预测车道线像素，但还需要将像素分组成 lane instance，并转换为评测协议要求的采样坐标。anchor-based 和 row-based 方法在固定采样协议下通常高效且有竞争力，但输出形式往往依赖预设 anchor 或 row 结构。query-based、curve-based 和 sequence-based 方法更接近结构化预测，但如何将 YOLO-style 网络改造成直接输出车道线实例的结构化检测器，仍然是一个值得研究的方向。

本文将车道线检测表述为结构化实例预测。核心思想是把每条车道线表示为一组固定 y 锚点上的有序点序列。在本项目采用的 TuSimple 设置中，每条 lane 在 `K=56` 个 fixed-y anchor 上采样，这些 anchor 对应原始图像坐标中的 `710, 700, 690, ..., 160`。模型不再只输出 mask 后再恢复曲线，而是直接预测候选 lane instances。每条预测 lane 包含 query-level existence、每个 y anchor 上的 x 坐标以及逐点可见性。

本文提出 **GCS-YOLO-Lane**，一种基于 YOLO11 的结构化车道线检测网络。该网络保留 YOLO11-style backbone 的特征提取能力，但将普通检测或分割输出替换为 query-based structured lane head。LSEM 通过水平/垂直 strip response、direction gate、coordinate-aware reweighting 和 dilated context 强化线状结构特征；LaneBiFPN 融合 P2-P5 多尺度特征；GCS lane head 使用可学习 lane query 对多尺度空间 token 做注意力交互，并直接预测一组候选车道线。

该方法基于两个观察。第一，车道线不是局部紧凑目标，而是具有长距离连续性和方向偏置的细长结构，因此特征增强应显式关注线状空间连续性。第二，车道线实例天然是带有可见性缺失的有序序列，因此模型应同时预测点位置和点可见性，而不是默认每个采样位置都可见。

训练过程采用 set prediction 思路。Hungarian matcher 根据点几何、曲线一致性和 lane existence cost 将预测 query 分配给 GT lane。随后 loss 同时监督 lane existence、visible-point regression、point-level visibility、curve regularity、smoothness，以及辅助 segmentation/edge maps。辅助 mask 和 edge 分支只用于训练期特征监督，不作为最终 lane 表示。最终输出仍然是每个 lane query 预测的结构化点序列。

本文贡献如下：

1. 提出一种基于 YOLO11 的结构化车道线检测网络，直接输出 fixed-y 有序点序列。
2. 设计 query-based GCS lane head，用 `Q=12` 个 lane queries 预测候选车道线的 existence、`K=56` 点坐标和逐点可见性。
3. 引入 LSEM 线敏感特征增强和 LaneBiFPN 多尺度特征融合，为结构化 lane head 提供更适合细长结构的空间特征。
4. 使用 Hungarian matching 与 visibility-aware supervision 训练结构化 lane output，并将 auxiliary mask/edge 分支限定为训练辅助信号。

实验性能主张在本草稿中暂不写出。只有当 active default configuration 完成 official-val 选择、一次性 final-test 报告、verified baseline table 和 ablation evidence 后，才能补入正式结论。

## 2. 相关工作

### 2.1 Segmentation-Based Lane Detection

Segmentation-based 方法将车道线看作像素级区域或 affinity structure。早期代表方法通过卷积空间信息传播增强细长车道线结构，再通过额外分组或拟合过程恢复 lane instances `[REF: SCNN]`。这类方法的重要意义在于，它们认识到车道线需要比局部边缘更强的上下文建模。但对本文而言，其主要局限在于表示形式：segmentation mask 还不是有序 lane instance，仍需后处理才能得到最终坐标序列。

GCS-YOLO-Lane 与该路线的区别在于显式预测最终车道线表示。模型训练时仍使用辅助 mask 和 edge supervision，因为 dense supervision 有助于学习车道线特征；但推理输出不是 segmentation map，而是每个 lane query 直接产生的 fixed-y 点序列和逐点可见性。

### 2.2 Anchor-Based and Row-Based Lane Detection

Anchor-based 和 row-based 方法是当前车道线检测中的强基线。UFLD 和 UFLDv2 等 row-wise classification 方法将车道线定位转化为预定义行上的位置分类，从而获得较高推理效率和紧凑输出 `[REF: UFLD]` `[REF: UFLDv2]`。LaneATT 等 anchor-based 方法使用 lane anchors 和 attention 机制高效建模候选车道线 `[REF: LaneATT]`。CLRNet 等 refinement-based 方法通过更强的多层特征聚合和迭代精修提升定位质量 `[REF: CLRNet]`。

这些方法与 TuSimple 很相关，因为 TuSimple 本身在固定 horizontal samples 上评测预测结果。其优势是输出形式接近 benchmark 协议；但从本文角度看，lane instance 往往仍围绕预设 anchor、row classifier 或 proposal refinement 组织。GCS-YOLO-Lane 保留 fixed-y sampling 兼容性，同时将每条候选 lane 交给一个 learned query 表示，使 lane existence、geometry 和 visibility 共同属于同一个结构化候选实例。

### 2.3 Query-Based, Curve-Based, and Sequence-Based Lane Detection

近年的研究逐渐将车道线检测推向结构化预测。LSTR 等 query-based 方法使用 transformer-style reasoning 预测 lane structures `[REF: LSTR]`。BezierLaneNet 等 curve-based 方法用参数曲线表示车道线几何 `[REF: BezierLaneNet]`。Lane2Seq 等 sequence-based 方法将车道线输出建模为有序序列生成问题 `[REF: Lane2Seq]`。这些方法是 GCS-YOLO-Lane 的核心概念对比对象，因为它们同样试图摆脱单纯像素区域输出。

GCS-YOLO-Lane 的区别在于将 YOLO11-style backbone、LSEM、LaneBiFPN 和 fixed-y query head 组合到一个结构化车道线检测网络中。该方法不是只预测全局曲线参数，也不是输出 dense mask，而是在固定 y 锚点上直接预测 x 坐标和 point-level visibility。这使得结构化预测与 TuSimple fixed-y evaluation protocol 保持一致。

### 2.4 本文定位

因此，本文应从两条线进行对比。面对 anchor/row-based 方法，GCS-YOLO-Lane 需要证明 learned lane queries 在 fixed-y 协议下仍具备竞争力。面对 query/curve/sequence 方法，GCS-YOLO-Lane 需要说明其 YOLO11-based architecture、line-sensitive enhancement、LaneBiFPN 和 visibility-aware fixed-y head 的组合价值。在 baseline 表和引用核验完成之前，正文不应宣称超过任何具体方法。

## 3. 方法

### 3.1 总体结构

GCS-YOLO-Lane 将车道线检测视为 structured set prediction。给定输入图像，网络最多预测 `Q=12` 条候选车道线。每条候选 lane 表示为 `K=56` 个 fixed-y anchor 上的有序点序列，并包含一个 lane-existence logit 和每个 anchor 的 point-validity logit。默认输出契约为：

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

模型由四个主要部分组成：YOLO11-style feature extractor、LSEM、LaneBiFPN 和 query-based GCS lane head。训练阶段通过 Hungarian assignment 匹配 predicted queries 和 GT lanes；推理阶段根据 query score、point visibility、最少可见点约束、可选 lane NMS 和最大 lane 数解码最终车道线。

### 3.2 Structured Fixed-Y Lane Representation

fixed-y representation 将每条车道线转换为有序点序列。在 TuSimple 设置中，y anchors 使用官方 h-sample 位置，从 `710` 到 `160`，步长为 `10`，并按原始图像高度 `720` 归一化：

```text
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end   = 160 / 720 = 0.2222222222222222
K = 56
```

标签生成时，系统在这些 fixed-y anchors 上对每条 GT lane 插值得到 x 坐标。落在可见 lane 范围内的 anchor 保存归一化 `(x, y)` 并标记为 valid；不可见 anchor 保留固定 y 坐标，但 valid flag 为 0。这样，几何和可见性被明确分离：y 坐标由协议固定，x 坐标由模型预测，`lane_valid` 指示哪些点参与可见车道线监督。

该表示有两个优点。第一，它直接对齐 TuSimple official evaluation 的 fixed-y sampling，减少后处理转换。第二，它让每条候选 lane 都有稳定的输出张量，同时通过 point-level visibility 表达局部不可见。

### 3.3 Line-Sensitive Feature Enhancement Module

车道线通常细长、方向性强，并且容易被道路纹理、阴影和磨损干扰。普通卷积特征可以捕获局部纹理，但未必能充分强调长距离线状连续性。因此，GCS-YOLO-Lane 在 YOLO11-style backbone 的 P3 和 P4 阶段插入 LSEM。

LSEM 包含四个组成部分。首先，水平和垂直 strip depthwise convolution 在细长邻域中提取方向响应。其次，direction gate 对水平和垂直响应做 soft weighting，使模块能够根据局部结构自适应选择方向。第三，coordinate-aware reweighting 沿高度和宽度调制特征，使空间位置信息保持显式。第四，dilated depthwise context branch 扩大感受野。模块输出通过 residual connection 与输入相加。

该模块的作用不是替代 backbone，而是在中间特征中注入对线状连续性的偏置。对于车道线这种窄而长、局部可见且嵌入复杂道路纹理中的目标，这种偏置与任务结构更一致。

### 3.4 LaneBiFPN Multi-Scale Fusion

lane head 同时需要细粒度空间定位和高层语义上下文。浅层特征有助于定位点坐标，深层特征有助于区分真实车道线、道路边缘、阴影和背景纹理。GCS-YOLO-Lane 使用 LaneBiFPN 融合 P2、P3、P4 和 P5 特征。

每个输入层级先通过投影对齐到统一通道数，然后进行 top-down 和 bottom-up 双向加权融合。融合后的多尺度特征保留空间结构，并为 GCS lane head 提供统一表示。这些特征会被展平为空间 token，并叠加位置编码和 level embedding，使 query decoder 能够跨尺度关注车道线证据。

### 3.5 Query-Based GCS Lane Head

GCS lane head 使用 `Q=12` 个可学习 lane queries 预测候选车道线集合。每个 query 目标上代表一条可能的 lane，但在解码前整个集合是无序的。Transformer decoder 让 query embeddings 关注 LaneBiFPN 的多尺度空间 token。当前实现使用三层 decoder 和八个 attention heads。

在 fixed-y 模式下，head 只预测每个 fixed y anchor 上的 x 坐标。y 坐标由固定 anchors 提供，因此 point predictor 只需估计 `K=56` 个位置的水平坐标。point MLP 输出 per-anchor x logits，reference-logit mechanism 为不同 query 提供不同的初始几何模式。head 还会在预测点位置采样图像特征，对 x logits 做 image-conditioned refinement。

head 同时预测 lane existence 和 point visibility。existence branch 为每个 query 输出一个 logit，用于判断该 query 是否应解码为 lane instance。point-validity branch 为每个 query-anchor pair 输出一个 logit，用于判断该点是否可见。由于车道线可能只部分可见，逐点可见性是结构化 lane representation 的必要组成。

辅助 mask 和 edge 分支仅用于训练。mask branch 输出二通道 lane-region map，edge branch 输出一通道边界 map。它们提供 dense supervision，但不作为最终车道线输出。

### 3.6 Matching and Training Objective

训练使用 Hungarian matching 连接预测 query 和 GT lane。对每张图像，系统构建 `Q` 个预测 query 与 GT lanes 之间的 cost matrix。matching cost 由 point distance、curve consistency 和 lane-existence confidence 组成：

```text
cost = cost_point * point_cost
     + cost_curve * curve_cost
     + cost_exist * exist_cost
```

point cost 在 GT 可见 anchor 上计算预测点与 GT 点之间的距离；curve cost 比较可见 triplet 上的二阶几何行为；existence cost 鼓励高置信 query 匹配真实 lane。

匹配完成后，loss 同时监督多个结构化输出。existence loss 训练 matched 和 unmatched query 的 lane-level existence；point loss 在可见 anchor 上训练几何；point-validity loss 训练 visibility branch；smoothness 和 curve loss 正则化点序列；auxiliary mask 和 edge loss 为共享特征提供 dense supervision。count-related 或 margin-related loss 只有在最终冻结的 active-default experiment configuration 中实际启用时，才应写入正式方法细节；本草稿不把默认关闭的实验 knob 写成核心贡献。

## 4. 实验

本章暂缓完成。

正式实验部分只能在 active default configuration 完全冻结并按项目 integrity rules 评估后撰写。需要补齐：

1. TuSimple fixed-y K56 dataset 与 evaluation protocol；
2. active-default training command 和 `args.yaml`；
3. official-val sweep 设置和 selected decode；
4. selected candidate 的一次性 official test result；
5. 与 verified query/curve/sequence、anchor/row-based 和 segmentation-based baselines 的对比；
6. LSEM、LaneBiFPN、query head/visibility branch、auxiliary supervision 的消融；
7. lane-count stability、short visible lanes 和 visibility errors 的失败分析。

本草稿不包含任何数值表，因为用户已明确不把 legacy `gt4short15` 作为主结果，并且 active-default 证据尚未完成。

## 5. 讨论

GCS-YOLO-Lane 应被理解为对 YOLO-style lane detector 的表示层重构。该方法不是在 segmentation model 后面简单加一个 lane postprocessor，而是将模型输出改造成一组结构化 lane instances。每个实例由有序 fixed-y 点序列、lane-level existence 和 point-level visibility 组成，因此更接近车道线 benchmark 和下游驾驶模块真正需要的几何对象。

该设计也明确了 dense supervision 的角色。辅助 mask 和 edge 输出有助于训练，因为车道线细长且需要像素级监督；但 dense supervision 不等于 dense output。GCS-YOLO-Lane 中的 mask/edge 预测只支持共享视觉特征，最终 lane 仍来自 query-owned structured sequences。

相对于 anchor/row-based 方法，GCS-YOLO-Lane 保留 fixed-y sampling 兼容性，但改变了 lane instance 的组织方式。row-based 方法通常逐行预测位置；GCS-YOLO-Lane 则让每个 query 拥有完整 fixed-y 点序列，使 existence、geometry 和 visibility 属于同一候选 lane。

相对于 query/curve/sequence 方法，本文区别主要在结构组合和表示方式。GCS-YOLO-Lane 结合 YOLO11-style backbone、LSEM、LaneBiFPN 和 fixed-y query decoding。其输出既不是 dense mask，也不是单纯全局曲线参数，而是与采样协议对齐的 visibility-aware point sequence。

最终 Discussion 需要在实验完成后重写。如果 active default configuration 在 official-val 和 final-test 上表现足够强，可以讨论结构设计的实证支持；如果结果仍然不足或不稳定，应诚实写成局限，或先继续实验再进入投稿版本。

## 6. 局限与未来工作

第一项局限是实验证据尚不完整。当前草稿没有 active-default official-val selection、one-shot final-test evidence、verified baseline comparisons 和 ablations，因此不能直接投稿。

第二项局限是 benchmark 范围。当前表示围绕 TuSimple fixed-y protocol 设计，这适合本项目目标，但仍需要跨数据集验证，才能说明该方法是否能泛化到不同 lane shape、camera setup 和 annotation convention。

第三项局限是 lane-count stability。项目记录显示，短可见侧车道和模糊 GT4 场景中的 lane-count 行为是反复出现的风险。正式论文应加入失败分析，区分 geometry miss、low-confidence short lane、duplicate-like prediction 和 spurious extra lane。

未来工作应补齐强 baselines 与消融，在更多车道线数据集上验证结构设计，并研究 query allocation 和 point visibility 在遮挡、标线磨损及密集多车道场景中的鲁棒性。

## 7. 结论

本文提出 GCS-YOLO-Lane，一种基于 YOLO11 的结构化车道线检测网络。核心思想是将每条车道线表示为 fixed-y 有序点序列，并通过可学习 lane queries 预测候选 lane instances。网络结合 LSEM 线敏感特征增强、LaneBiFPN 多尺度融合和 GCS lane head，同时预测 existence、point geometry 和 point-level visibility。训练使用 Hungarian matching 与 visibility-aware supervision，辅助 mask/edge 分支用于支持特征学习。

最终结论必须等待 active default experimental evidence 固定后补全。当前阶段可以可靠成立的结论是结构层面的：GCS-YOLO-Lane 通过直接预测 lane instance point sequences，为 mask-style lane output 提供了一种结构化替代方案。

## 投稿前声明待补

- **Data availability**：待补 TuSimple 数据访问方式和 license 表述。
- **Code availability**：待确认项目代码是否公开以及公开方式。
- **Ethics declaration**：大概率不涉及人体实验，但需确认数据集和自动驾驶数据使用要求。
- **Conflict of interest**：待作者确认。
- **Funding**：待作者确认。
- **Author contributions**：待作者名单和 CRediT 角色。
- **AI assistance disclosure**：本草稿使用 AI 辅助撰写，投稿时应按目标期刊政策披露。

## 参考文献占位符

以下不是最终参考文献列表。每一项都必须通过原论文、DOI、出版社页面、arXiv 页面或官方代码仓库核验。

- `[REF: SCNN]` segmentation/spatial CNN lane detection reference。
- `[REF: UFLD]` Ultra Fast Lane Detection。
- `[REF: UFLDv2]` Ultra Fast Lane Detection V2。
- `[REF: LaneATT]` anchor-based attention lane detection。
- `[REF: CLRNet]` cross-layer refinement lane detection。
- `[REF: LSTR]` lane shape/transformer/query structured lane detection。
- `[REF: BezierLaneNet]` Bezier curve-based lane detection。
- `[REF: Lane2Seq]` sequence-based lane detection。

## Claim-Evidence Map

| Claim | Evidence Source | Status |
|---|---|---|
| GCS-YOLO-Lane 输出结构化 lane point sequences，而不是最终 mask。 | project contracts 和 model output shape。 | 已由项目代码/文档支持 |
| 方法使用 `Q=12` lane queries 和 `K=56` fixed-y anchors。 | `current-contracts.md`、model config、project summary。 | 已由项目代码/文档支持 |
| LSEM 包含 strip responses、direction gate、coordinate reweighting 和 dilated context。 | project summary 与实现说明。 | 已由项目代码/文档支持 |
| LaneBiFPN 融合 P2-P5 特征供 lane head 使用。 | project summary 与 model YAML。 | 已由项目代码/文档支持 |
| Hungarian matching 用于 query-to-lane assignment。 | `GCSHungarianMatcher` 说明。 | 已由项目代码/文档支持 |
| GCS-YOLO-Lane 超过某个 baseline。 | 需要 active-default official-val/test 和 verified baseline table。 | 需要证据 |
| 方法足以支撑 Q3/Q4 投稿。 | 需要结果、消融和期刊匹配。 | 需要证据 |
| LSTR、Lane2Seq、BezierLaneNet、UFLD/UFLDv2、LaneATT、CLRNet、SCNN 的具体相关工作表述。 | 需要核验原始文献。 | 需要文献核验 |

## 自审清单

- **Contribution**：结构贡献清晰；实验贡献暂不能 claim。
- **Writing clarity**：术语稳定，包括 fixed-y anchors、lane queries、point visibility、LSEM、LaneBiFPN、GCS lane head。
- **Experimental strength**：active-default 实验完成前无法判断。
- **Evaluation completeness**：缺 verified baseline table、ablations 和 failure analysis。
- **Method soundness**：representation 和 output contract 具体明确；最终可信度取决于证据和公平对比。
