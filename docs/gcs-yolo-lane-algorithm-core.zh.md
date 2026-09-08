# GCS-YOLO-Lane 算法核心内容

> 本文档面向论文方法章节、项目交接和后续实验记录。描述对象是当前 `5-25-3` 分支的 active default 主线：`YOLO11-style backbone + LSEM + LaneBiFPN + query-based GCS lane head + fixed-y K56 structured supervision`。
>
> 当前工作区包含其他未提交修改和历史实验产物。本文件只整理当前合同要求的默认算法，不把默认关闭的实验机制写成主线方法，也不包含未经 official-val 和 final-test 证据支持的性能结论。

## 1. 一句话概括

给定一幅道路图像，GCS-YOLO-Lane 使用 YOLO11-style backbone 提取多尺度特征，在 P3/P4 阶段通过 LSEM 强化线状结构，再由 LaneBiFPN 融合 P2-P5 特征，最后使用一组可学习 lane queries 直接预测候选车道线集合。每个候选 lane 同时包含 lane existence、fixed-y 有序点序列和逐点可见性；训练阶段使用 Hungarian matching 将无序预测 query 与 GT lane 对齐，并联合优化几何、可见性、曲线连续性和辅助 dense supervision。

## 2. 论文中的算法定位

本文应将 GCS-YOLO-Lane 表述为一种 **structured lane instance prediction** 方法，而不是普通的目标检测框架或最终分割输出。

其核心表示为：

1. 每条车道线在共享的 `K=56` 个 fixed-y anchors 上表示；
2. 每个 lane instance 由一个 query 独立拥有；
3. query 预测完整的 x 坐标序列、lane existence 和逐点 visibility；
4. Hungarian matching 解决 query 集合与 GT lane 集合之间的无序对应关系；
5. mask/edge 分支只提供训练辅助监督，不定义最终 lane 输出。

建议使用下面的一句话作为论文方法概述：

> GCS-YOLO-Lane treats lane detection as visibility-aware structured set prediction: each learned lane query predicts one candidate lane on a shared fixed-y grid, while line-sensitive feature enhancement, bidirectional multi-scale fusion, Hungarian matching and geometry-aware supervision support the structured output.

这句话只描述已由代码和当前合同确认的结构，不声称性能优越、首次提出或达到 state of the art。

## 3. 当前有效配置与符号

### 3.1 默认配置

| 项目 | 当前合同 |
|---|---|
| 默认模型 | `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml` |
| 默认数据 | `data/tusimple_gcs_fixed_y_960x544.yaml` |
| TuSimple 输入 | `--imgsz 544 960` |
| 输入张量空间尺寸 | `H=544, W=960` |
| 原始 TuSimple 高度 | `720` |
| query 数量 | `Q=12` |
| 每条 lane 的点数 | `K=56` |
| 点表示 | `point_mode=fixed_y` |
| decoder 层数 | `3` |
| attention heads | `8` |
| LaneBiFPN 输出通道 | `128` |

`--imgsz 544 960` 是 `H,W` 顺序。论文、脚本和实验记录中不能将其写成 `960 544`。

### 3.2 数学符号

设输入图像为：

\[
I \in \mathbb{R}^{3 \times H \times W}, \qquad H=544,\;W=960.
\]

第 \(i\) 条 GT lane 在 fixed-y 网格上的结构化标签写为：

\[
g_i=\{(x_{ik},y_k,v_{ik})\}_{k=0}^{K-1},
\]

其中：

- \(x_{ik}\in[0,1]\) 是归一化 x 坐标；
- \(y_k\) 是共享的归一化 fixed-y anchor；
- \(v_{ik}\in\{0,1\}\) 表示该 anchor 是否位于该 lane 的有效可见范围内。

第 \(q\) 个 prediction query 输出：

\[
\hat{g}_q=
\left(
\hat{s}_q,
\{(\hat{x}_{qk},y_k,\hat{v}_{qk})\}_{k=0}^{K-1}
\right),
\]

其中 \(\hat{s}_q\) 是 lane existence logit，\(\hat{v}_{qk}\) 是逐点 visibility logit。

## 4. Fixed-Y 结构化车道线表示

### 4.1 固定 y 合同

当前 TuSimple K56 合同使用原始图像坐标中的：

```text
710, 700, 690, ..., 160
```

共 `56` 个 anchor，按从图像底部到顶部的降序排列。归一化形式为：

```text
fixed_y_start = 710 / 720 = 0.9861111111111112
fixed_y_end   = 160 / 720 = 0.2222222222222222
K = 56
```

等价地，第 \(k\) 个 anchor 为：

\[
y_k=\frac{710-10k}{720},\qquad k=0,\ldots,55.
\]

这里的归一化分母是原始 TuSimple 图像高度 `720`。在 resized image 上使用该 anchor 时，其像素 y 坐标通过 \(y_k H\) 计算。这样可以保持标签与 TuSimple official `h_samples` 的几何对应关系。

### 4.2 标签生成流程

标签转换入口为 `tools/convert_tusimple_to_gcs.py`，固定 y 采样实现为 `gcs_tools/label_utils.py::sample_polyline_fixed_y()`。

对每条原始 TuSimple lane，标签生成过程为：

1. 读取原始 JSON 中的 `raw_file`、`h_samples` 和 lane x 坐标；
2. 将图像和 lane 坐标同步 resize 到 `H=544, W=960`；
3. 删除非法坐标，按 y 从底部到顶部排序，并去除重复 y；
4. 在固定 anchor 网格 `710..160` 上对 lane x 坐标进行插值；
5. 将 x 坐标除以输出图像宽度 `W`，得到归一化 x；
6. 有效 anchor 保存 `(x,y)` 并令 `lane_valid=1`；
7. 无效 anchor 保留固定 y、令 `x=0` 并令 `lane_valid=0`；
8. 有效点少于两个的 lane 被过滤；
9. 根据 resized lane centerlines 生成 `semantic_mask` 和 `edge_mask`；
10. 将结构化标签写入压缩 `.npz` 文件。

标签中的主要字段为：

```text
lanes
lane_valid
num_lanes
fixed_y
point_mode
semantic_mask
edge_mask
raw_file
image_shape
num_points
```

K56 标签必须从原始 TuSimple JSON 和图像重新生成，不能由历史 K32 标签再次插值获得。这样可以避免旧的 anchor 数量和旧的 fixed-y 范围污染当前合同。

### 4.3 该表示解决的问题

fixed-y 表示具有三个直接作用：

1. 将不规则的 lane polyline 转换为固定形状的 `K x 2` 张量；
2. 将 y 方向的采样协议固定下来，使模型主要学习横向 x 几何；
3. 使用 `lane_valid` 显式表达局部遮挡、截断和不可见端点。

因此，模型不需要为每个 query 同时自由回归所有 y 坐标，但仍能通过 visibility 分支表示一条 lane 只在局部区域可见的情况。

## 5. 网络总体结构

当前默认网络由以下路径组成：

```text
Input image
    -> YOLO11-style backbone
    -> P2, P3, P4, P5 feature maps
    -> LSEM on P3 and P4
    -> LaneBiFPN bidirectional multi-scale fusion
    -> spatial tokens + 2D position embedding + level embedding
    -> 3-layer Transformer decoder with Q=12 lane queries
    -> point geometry, point visibility, lane existence
    -> training: Hungarian matching + multi-term loss
    -> inference: visibility-aware lane decoding
```

默认 YAML 中的主干和 head 连接关系为：

```yaml
backbone:
  P2 -> C3k2
  P3 -> C3k2 -> LSEM
  P4 -> C3k2 -> LSEM
  P5 -> C3k2 -> SPPF -> C2PSA

head:
  [P2, enhanced_P3, enhanced_P4, P5]
    -> LaneBiFPN[128]
    -> GCSLaneHead[Q=12, K=56, decoder_layers=3, nhead=8, fixed_y]
```

## 6. Line-Sensitive Feature Enhancement Module

### 6.1 模块动机

车道线通常具有细长、连续和明显方向性的结构。道路纹理、阴影、车辆遮挡和磨损会破坏局部 appearance，因此仅依赖普通局部卷积可能无法稳定保留 lane-like response。

LSEM 的设计目标是在中层 feature map 中加入方向性和长程结构偏置。它被插入 P3 和 P4 阶段，使增强后的特征同时保留中层空间分辨率和较大的上下文范围。

### 6.2 模块结构

LSEM 由以下部分组成：

1. `LineStripAttention`；
2. depthwise dilated context convolution；
3. `1 x 1` output projection；
4. residual connection；
5. SiLU activation。

`LineStripAttention` 具体包含：

- 水平 `1 x k` depthwise strip convolution，默认 `k=9`；
- 垂直 `k x 1` depthwise strip convolution；
- 基于全局池化特征的 direction gate；
- 对水平和垂直响应做 softmax 加权；
- `1 x 1` fuse；
- `CoordReweight` 坐标感知重加权。

若输入特征为 \(X\)，水平和垂直分支可概括为：

\[
F_h=\operatorname{DWConv}_{1\times k}(X),\qquad
F_v=\operatorname{DWConv}_{k\times 1}(X).
\]

direction gate 产生两个归一化权重 \(a_h,a_v\)，然后：

\[
F_{\mathrm{dir}}=a_hF_h+a_vF_v.
\]

经过 `1 x 1` 融合和坐标重加权后，再通过 dilation=`2` 的 depthwise context branch，最后与输入做残差相加：

\[
F_{\mathrm{LSEM}}
\;=\;
\operatorname{SiLU}
\left(
\operatorname{Proj}
\left(
\operatorname{DConv}_{d=2}
\left(
\operatorname{CoordReweight}
\left(
\operatorname{Fuse}(F_{\mathrm{dir}})
\right)
\right)
\right)
+X
\right).
\]

### 6.3 论文中的边界

可以可靠写入论文的是 LSEM 的实际组成和设计动机，即它显式建模水平/垂直 strip response、坐标信息和扩张上下文。

在没有独立消融实验前，不应写成“LSEM 已证明带来某个准确率增益”，也不应把实现动机写成已经完成验证的性能结论。

## 7. LaneBiFPN 多尺度融合

### 7.1 模块动机

结构化 lane head 同时需要：

- 浅层特征提供精细的 x 定位；
- 深层特征提供道路场景语义和上下文；
- 不同尺度上的 lane evidence 进行统一交互。

因此，LSEM 增强后的 P3/P4 与 P2/P5 一起送入 LaneBiFPN。

### 7.2 双向融合过程

LaneBiFPN 首先通过 `1 x 1 Conv-BN-SiLU` 将 P2、P3、P4、P5 对齐到 `128` 个通道。

随后执行 top-down pathway：

```text
P5 -> P4 -> P3 -> P2
```

再执行 bottom-up pathway：

```text
P2 -> P3 -> P4 -> P5
```

每个融合节点使用 `WeightedFusion`。对于输入特征 \(F_1,\ldots,F_m\)，其融合形式为：

\[
\bar{w}_i=
\frac{\operatorname{ReLU}(w_i)}
{\sum_j\operatorname{ReLU}(w_j)+\epsilon},
\qquad
F_{\mathrm{fuse}}=\sum_i\bar{w}_iF_i.
\]

因此融合权重是可学习的、非负的并且经过归一化。融合后再使用 `Conv-BN-SiLU` 进行特征变换，最终输出四个统一通道的多尺度 feature maps：

```text
[P2_fused, P3_fused, P4_fused, P5_fused]
```

### 7.3 论文中的边界

可以写明 LaneBiFPN 如何对齐和双向融合 P2-P5，以及它为结构化 lane head 提供多尺度 token。

在没有 LaneBiFPN 消融前，不应直接声称该模块相对于普通 FPN 或单尺度输入已经取得确定的性能提升。

## 8. Query-Based GCS Lane Head

### 8.1 多尺度 token 编码

GCS lane head 接收 LaneBiFPN 的四个 feature maps。对每个尺度：

1. 将特征图展平为空间 token；
2. 加入二维 sine-cosine position embedding；
3. 加入可学习的 level embedding；
4. 将四个尺度的 token 沿空间维拼接。

该过程保留了 feature map 的空间位置和尺度来源，不使用全局 `1 x 1` pooled feature 替代空间 token。

### 8.2 Query decoder

head 使用 `Q=12` 个可学习 query embeddings，并通过三层 Transformer decoder 让 query 与多尺度空间 token 交互。默认 attention head 数量为 `8`，LaneBiFPN 和 decoder 的工作通道数为 `128`。

在解码前，query 集合没有固定的 lane 顺序。第 `q` 个 query 只表示一个候选 lane instance，具体对应哪个 GT lane 由训练时的 Hungarian assignment 决定。

### 8.3 Fixed-y 点预测

在 `point_mode=fixed_y` 下，point head 只预测每个 fixed-y anchor 上的 x logits。y 坐标由 registered `fixed_y_anchors` buffer 提供：

\[
\hat{x}_{qk}=\sigma(z_{qk}^{x}),\qquad
\hat{y}_{qk}=y_k.
\]

当前实现还包含 query-specific point references。其作用是为不同 query 提供不同的初始透视形状，使不同 query 在训练早期不必全部从同一条中心线开始。

粗 x logits 经过 image-conditioned refinement：

1. 根据粗 x 和 fixed-y 组成粗点坐标；
2. 在 P2-P4 特征上对预测位置进行 bilinear sampling；
3. 将 query state、point embedding、坐标 embedding 与 sampled image feature 拼接；
4. 预测 x-logit refinement delta；
5. 将 delta 加到粗 x logits 上。

该过程使最终 x 预测不仅依赖 query 的全局状态，也依赖预测点位置附近的图像特征。

### 8.4 Existence 和 point visibility

对于每个 query，`exist_mlp` 输出一个 lane-level existence logit：

```text
pred_logits: B x Q
```

对于每个 query 和 fixed-y anchor，`point_valid_mlp` 先产生粗 visibility logits，然后使用预测点位置采样的图像特征进行细化：

```text
pred_valid_logits: B x Q x K
```

这两个分支表达不同层次的判断：

- `pred_logits`：该 query 是否应被解释为一条 lane；
- `pred_valid_logits`：该 lane 在某一个 fixed-y anchor 上是否可见。

### 8.5 辅助 mask 和 edge 分支

head 可以从 P2 特征产生：

```text
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

这两个分支用于训练期 dense supervision 或显式请求 auxiliary output 时的诊断。它们不是最终 lane 表示，推理时的 lane 结果仍来自 query-owned structured point sequences。

## 9. 默认输出合同

默认 query 模型的核心输出为：

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
```

训练或显式 auxiliary output 模式下还包括：

```text
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

每个张量的含义为：

| 输出 | 含义 |
|---|---|
| `pred_points[..., 0]` | 归一化 x 坐标 |
| `pred_points[..., 1]` | fixed-y anchor，默认不是自由预测的 y |
| `pred_logits` | lane existence logits |
| `pred_valid_logits` | point-level visibility logits |
| `aux_mask_logits` | 训练辅助的 lane-region logits |
| `aux_edge_logits` | 训练辅助的 edge logits |

默认 query YAML 不输出 `pred_count_logits`。可选 query Count Head 只在：

```text
ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-count.yaml
```

中启用，并额外输出：

```text
pred_count_logits: B x 4
```

该 Count Head 是 default-off ablation，不属于默认算法核心。

## 10. Hungarian Matching

### 10.1 匹配对象

对于一个 batch 中的第 \(b\) 张图像，设预测 query 数为 \(Q\)，有效 GT lane 数为 \(N_b\)。匹配器建立一个：

\[
C^{(b)}\in\mathbb{R}^{Q\times N_b}
\]

的 query-to-GT cost matrix。

GT lane 中有效 anchor 少于两个的项不会参与有效匹配。

### 10.2 匹配代价

当前默认代价为：

\[
C_{q i}
=
\lambda_{\mathrm{point}}C^{\mathrm{point}}_{q i}
+
\lambda_{\mathrm{curve}}C^{\mathrm{curve}}_{q i}
+
\lambda_{\mathrm{exist}}C^{\mathrm{exist}}_{q i},
\]

其中默认：

```text
gcs_cost_point = 5.0
gcs_cost_curve = 0.05
gcs_cost_exist = 0.1
```

各项含义如下：

1. `point_cost`
   - 只在 GT valid anchors 上计算；
   - 使用 aspect-weighted normalized L1；
   - x/y 按真实输入图像宽高进行尺度调整。
2. `curve_cost`
   - 对连续三个 valid anchors 计算二阶差分；
   - 在像素尺度上比较预测和 GT 的局部曲率变化；
   - 只使用连续有效 triplets。
3. `exist_cost`
   - 使用 `-sigmoid(pred_logits)`；
   - 高 existence probability 的 query 更容易匹配 GT lane。

当前 matcher 还支持训练期几何门控：

```text
gcs_match_min_overlap = 2
gcs_match_max_x_dist = 0.0
gcs_match_gate_px = 160.0
```

其中 `gcs_match_max_x_dist=0.0` 表示关闭该 x-distance gate；`gcs_match_gate_px` 用于限制平均点误差过大的匹配候选。

### 10.3 Assignment

将 cost matrix 转换为 CPU numpy array 后，使用 `scipy.optimize.linear_sum_assignment` 求解最小代价二分匹配。返回结果是每张图像的：

```text
(matched_prediction_indices, matched_gt_indices)
```

该 assignment 只定义训练期的 query-to-lane 对应关系，不要求 query index 在不同图像之间保持固定语义。

## 11. 默认训练目标

### 11.1 Existence loss

所有未匹配 query 的 existence target 为 `0`。

匹配 query 默认使用 quality-aware target。首先根据预测点和 GT 点的平均像素误差 `APE` 计算几何质量。在当前 linear quality 模式下：

- `APE <= 10 px` 时，几何 quality 为 `1`；
- `APE >= 20 px` 时，几何 quality 为 `0`；
- 中间区间做线性插值。

如果存在 `pred_valid_logits`，还计算预测 visibility probability 与 GT `lane_valid` 之间的 soft IoU，并将其乘到几何 quality 上。最终匹配 query 的 target 可写为：

\[
t_q=(1-\alpha)+\alpha\,
q_{\mathrm{geom}}\,q_{\mathrm{visible}},
\]

其中默认：

```text
gcs_exist_quality_alpha = 1.0
```

最后对 `pred_logits` 使用 BCE with logits。Focal 调制在默认配置下关闭。

### 11.2 Point regression loss

对 Hungarian matched lanes，在 GT valid anchors 上计算 aspect-weighted L1 point loss：

\[
\mathcal{L}_{\mathrm{point}}
=
\frac{1}{\sum_k v_{ik}}
\sum_k v_{ik}
\left\|
(\hat{x}_{qk},\hat{y}_{qk})-(x_{ik},y_k)
\right\|_{\mathrm{aspect}}.
\]

在 fixed-y 模式下，预测 y 由合同直接给定，并与 GT fixed-y anchor 对齐，因此有效学习量主要集中在 x 坐标。

默认 gain：

```text
gcs_point = 15.0
```

### 11.3 Point visibility loss

对 `pred_valid_logits` 使用 BCE：

- matched query 的 target 是对应 GT lane 的 `lane_valid`；
- unmatched query 的 target 是全零；
- 正样本权重根据 batch 中正负样本比例计算，并受上限约束。

默认 gain：

```text
gcs_point_valid = 1.0
```

### 11.4 Smoothness loss

对连续有效的相邻三点计算二阶差分：

\[
\Delta^2\hat{p}_k
=
\hat{p}_{k+1}-2\hat{p}_k+\hat{p}_{k-1}.
\]

在 GT 连续可见的 triplets 上，对二阶差分的绝对值进行平滑正则，抑制结构化 lane 序列中的局部抖动。

默认 gain：

```text
gcs_smooth = 0.05
```

### 11.5 Curve loss

Curve loss 比较预测和 GT 的二阶曲率：

\[
\mathcal{L}_{\mathrm{curve}}
\propto
\operatorname{SmoothL1}
\left(
\Delta^2\hat{p}_k,\Delta^2p_k
\right).
\]

该项在像素尺度上计算，并根据 GT 曲率幅度进行自适应加权，使有明显弯曲结构的局部得到更有针对性的几何监督。

默认 gain：

```text
gcs_curve = 0.1
```

### 11.6 Auxiliary mask loss

`aux_mask_logits` 对应二分类 lane-region dense prediction。默认 mask loss 由 class-weighted cross entropy 和 Dice loss 组成。

默认 gain：

```text
gcs_mask = 0.2
```

该分支的功能是为共享 feature 提供 dense spatial supervision，不改变最终结构化 lane output contract。

### 11.7 Auxiliary edge loss

`aux_edge_logits` 对应一通道 edge prediction。默认 edge loss 由 weighted BCE 和 Dice loss 组成。

默认 gain：

```text
gcs_edge = 0.2
```

edge supervision 用于强化 lane boundary 或细线结构的空间证据，但推理输出仍由 query head 产生。

### 11.8 总损失

当前默认 active loss 可写为：

\[
\mathcal{L}
=
\lambda_e\mathcal{L}_{\mathrm{exist}}
+\lambda_p\mathcal{L}_{\mathrm{point}}
+\lambda_v\mathcal{L}_{\mathrm{valid}}
+\lambda_s\mathcal{L}_{\mathrm{smooth}}
+\lambda_c\mathcal{L}_{\mathrm{curve}}
+\lambda_m\mathcal{L}_{\mathrm{mask}}
+\lambda_{edge}\mathcal{L}_{\mathrm{edge}}.
\]

默认 gain 为：

| Loss | Config gain |
|---|---:|
| `exist_loss` | `2.0` |
| `point_loss` | `15.0` |
| `point_valid_loss` | `1.0` |
| `smooth_loss` | `0.05` |
| `curve_loss` | `0.1` |
| `mask_loss` | `0.2` |
| `edge_loss` | `0.2` |

因此，当前默认配置的展开形式为：

\[
\mathcal{L}
=
2.0\mathcal{L}_{\mathrm{exist}}
+15.0\mathcal{L}_{\mathrm{point}}
+1.0\mathcal{L}_{\mathrm{valid}}
+0.05\mathcal{L}_{\mathrm{smooth}}
+0.1\mathcal{L}_{\mathrm{curve}}
+0.2\mathcal{L}_{\mathrm{mask}}
+0.2\mathcal{L}_{\mathrm{edge}}.
\]

## 12. 推理解码

### 12.1 基本流程

推理入口为 `ultralytics/utils/gcs_postprocess.py::decode_gcs_predictions()`。对单张图像，输入为：

```text
pred_points: Q x K x 2
pred_logits: Q
pred_valid_logits: Q x K
```

默认 query decode 流程为：

1. 对 `pred_logits` 使用 sigmoid 得到 query existence score；
2. 根据 `score_thr` 过滤低分 query；
3. 按 descending y 对每条 lane 的点重新排序；
4. 对 `pred_valid_logits` 使用 sigmoid；
5. 根据 `point_valid_thr` 得到 point visibility mask；
6. 只保留最长的连续 visible segment；
7. 少于 `min_points` 个有效点的 lane 被过滤；
8. 按 lane score 降序排列；
9. 可选执行 lane NMS；
10. 可选在 `max_det` 截断前先过滤 visibility/min-points failures；
11. 根据 `max_det` 保留最终 lane hypotheses；
12. 输出 normalized points、visible points、score、query index 和可选 pixel points。

该过程说明当前方法不是“完全没有后处理”。后处理的作用是将网络输出的候选 query 转换为符合 benchmark 评测和可视化要求的 lane instances。

### 12.2 Lane NMS

Lane NMS 使用共享 fixed-y 点索引比较 lane 对之间的平均 x 像素距离：

\[
d(a,b)
=
\frac{1}{|\Omega|}
\sum_{k\in\Omega}
|a_{k,x}-b_{k,x}|W,
\]

其中 \(\Omega\) 是两个 lane 的共同 visible anchors。

若距离小于 `nms_dist_px`，低分 lane 被认为是 duplicate-like prediction 并被抑制。使用共同 visible anchors 可以避免无效端点影响重复 lane 判断。

### 12.3 TuSimple official 转换

为了计算 TuSimple official metrics：

1. 使用 decoded visible points；
2. 将 normalized coordinates 还原到图像坐标；
3. 插值到 TuSimple official `h_samples`；
4. 无法插值或越界的位置写成 `-2`；
5. 少于两个有效 x 的 lane 不进入官方预测；
6. 使用 TuSimple official metric 计算 `ACC`、`FP` 和 `FN`。

推理和 decode 不能使用 GT，不能 fabricated lanes。threshold、checkpoint 和 postprocess 的选择应通过 official-val 完成；test 只用于已选定候选的一次性最终评估。

## 13. 训练和推理伪代码

### 13.1 训练

```text
for each image I:
    features = YOLO11_style_backbone(I)
    features[P3] = LSEM(features[P3])
    features[P4] = LSEM(features[P4])
    pyramid = LaneBiFPN([P2, P3, P4, P5])

    tokens = flatten_with_2d_position_and_level(pyramid)
    query_states = TransformerDecoder(query_embeddings, tokens)

    pred_points = fixed_y_point_head(query_states, pyramid)
    pred_logits = existence_head(query_states)
    pred_valid_logits = visibility_head(query_states, pred_points, pyramid)
    aux_mask_logits, aux_edge_logits = auxiliary_heads(P2)

    matches = Hungarian(pred_points, pred_logits, gt_lanes, gt_valid)
    loss = existence_loss(
        pred_logits, pred_points, pred_valid_logits, matches
    )
    loss += point_loss(pred_points, gt_lanes, gt_valid, matches)
    loss += point_valid_loss(pred_valid_logits, gt_valid, matches)
    loss += smooth_loss(pred_points, gt_valid, matches)
    loss += curve_loss(pred_points, gt_lanes, gt_valid, matches)
    loss += auxiliary_mask_edge_loss(aux_mask_logits, aux_edge_logits)
    update_parameters(loss)
```

### 13.2 推理

```text
features = backbone(I)
features[P3] = LSEM(features[P3])
features[P4] = LSEM(features[P4])
pyramid = LaneBiFPN([P2, P3, P4, P5])

tokens = flatten_with_position_and_level(pyramid)
query_states = TransformerDecoder(query_embeddings, tokens)
pred_points = fixed_y_point_head(query_states, pyramid)
pred_logits = existence_head(query_states)
pred_valid_logits = visibility_head(query_states, pred_points, pyramid)

lanes = decode_gcs_predictions(
    pred_points,
    pred_logits,
    pred_valid_logits,
    score_thr,
    point_valid_thr,
    min_points,
    nms_dist_px,
    max_det,
)
```

## 14. 当前主线与可选机制边界

### 14.1 属于默认算法核心

以下内容可以直接写入默认方法章节：

- `fixed_y` K56 structured lane representation；
- `Q=12` query-based lane set prediction；
- YOLO11-style backbone；
- P3/P4 的 LSEM；
- P2-P5 的 LaneBiFPN；
- multi-scale spatial tokens、2D position embedding 和 level embedding；
- 3-layer Transformer decoder；
- query-level existence；
- point-level visibility；
- Hungarian matching；
- point、smooth、curve、mask、edge 等默认 active losses；
- visibility-aware query decoding；
- TuSimple official fixed-sample conversion。

### 14.2 不属于默认主线核心

以下机制在当前分支存在代码或配置入口，但默认关闭，不能作为默认算法贡献写入：

| 机制 | 当前状态 | 论文处理 |
|---|---|---|
| query Count Head | 只在专用 query-count YAML 启用 | 单独作为可选 ablation |
| `count_loss` | `gcs_count=0.0` | 不写入默认总损失 |
| `count_under5_loss` | `gcs_count_under5=0.0` | 不写入默认总损失 |
| `count_boundary_loss` | `gcs_count_boundary=0.0` | 作为分支局部可选实验 |
| `gcs_spurious_neg` | `0.0` | 不写入默认训练方法 |
| `boundary_pseudo_neg` | `0.0` | 不写入默认训练方法 |
| `gcs_short_geom` | `0.0` | 属于历史/可选短 lane 几何实验 |
| `gcs_hard_sampling` | default-off train-only sampler | 作为训练协议消融，不是模型结构 |
| ordered-slot | 独立模式 | 不与默认 query 主线混写 |
| later Count/Quality/Survival Head | 不属于当前 active rollback 主线 | 只能作为历史记录 |

当前 active source/config 以 `docs/agent-context/current-contracts.md` 为准。后续历史文档、旧 checkpoint 说明或旧实验脚本不能覆盖该合同。

## 15. 可直接用于论文的方法总结

下面的段落可作为论文 `Method Overview` 的基础版本，之后只需根据目标期刊调整语气和符号：

> We formulate lane detection as structured set prediction. Given an input image, GCS-YOLO-Lane predicts a fixed-size set of candidate lane instances using learnable lane queries. Each candidate is represented on a shared fixed-y grid by a sequence of normalized x coordinates and point-level visibility states, together with a lane-existence score. For the current TuSimple setting, the grid contains `K=56` anchors corresponding to the original image heights `710, 700, ..., 160`, normalized by the original image height `720`.
>
> The network follows a YOLO11-style feature extraction pipeline. We insert the Line-Sensitive Enhancement Module (LSEM) at the P3 and P4 stages to model directional strip responses, coordinate-aware feature reweighting and dilated context. The resulting P2-P5 features are projected to a common channel dimension and fused by LaneBiFPN through top-down and bottom-up paths with learnable normalized fusion weights. The fused feature maps are flattened into spatial tokens and augmented with two-dimensional sine-cosine position embeddings and level embeddings. A three-layer Transformer decoder then lets `Q=12` learnable lane queries attend to these tokens.
>
> In fixed-y mode, the lane head predicts x coordinates while the y coordinates are provided by the fixed-y contract. Point features sampled at the predicted positions are used to refine the x logits and point-visibility logits. The head therefore produces lane geometry, lane existence and point visibility in a common query-owned representation. Auxiliary semantic-mask and edge branches provide dense training supervision but are not used as the final lane representation.
>
> We use Hungarian matching to associate the unordered query predictions with GT lanes. The matching cost combines valid-anchor point distance, second-order curve distance and existence confidence. The training objective jointly supervises existence, point geometry, point visibility, second-order smoothness, curvature consistency and auxiliary mask/edge predictions. At inference, query scores and point visibility are thresholded, the longest contiguous visible segment is retained, duplicate lanes may be suppressed by fixed-y lateral distance, and the remaining lane sequences are converted to the TuSimple official sampling format.

## 16. Claim-Evidence Map

| 论文表述 | 当前证据 | 状态 |
|---|---|---|
| 方法将 lane detection 表述为 structured set prediction | `GCSLaneHead`、matcher 和 fixed-y 输出合同 | 已支持 |
| 每条 lane 使用 `K=56` fixed-y anchors | `current-contracts.md`、数据 YAML、标签转换和 head 校验 | 已支持 |
| 默认模型使用 `Q=12` queries | 默认模型 YAML 与 `GCSLaneHead` | 已支持 |
| LSEM 包含水平/垂直 strip、direction gate、coordinate reweighting 和 dilated context | `ultralytics/nn/modules/gcs_lane.py` | 已支持 |
| LaneBiFPN 双向融合 P2-P5 | `LaneBiFPN` 实现与默认模型 YAML | 已支持 |
| matcher 使用 point、curve 和 existence cost | `GCSHungarianMatcher` | 已支持 |
| point visibility 被显式监督并参与 decode | `pred_valid_logits`、`point_valid_loss` 和 `decode_gcs_predictions` | 已支持 |
| mask/edge 是训练辅助分支而非最终 lane output | head 输出逻辑和 loss 调用路径 | 已支持 |
| LSEM 或 LaneBiFPN 带来确定性能增益 | 需要对应 ablation | 需要证据 |
| 方法超过某个 baseline | 需要公平 baseline、official-val 和 final-test | 需要证据 |
| 方法达到 SOTA 或具有普适性 | 需要完整实验和跨数据集证据 | 需要证据 |

## 17. 当前不能直接写入论文的结论

在实验章节完成前，不能直接使用以下表述：

- “首次提出 query-based lane detection”；
- “完全 anchor-free”；
- “不需要后处理”；
- “显著优于所有现有方法”；
- “达到 state of the art”；
- “LSEM 必然提升准确率”；
- “LaneBiFPN 已证明解决短 lane 或重复 lane 问题”；
- “默认 Count Head 改善 lane-count prediction”；
- 将历史 `gt4short15`、旧分支结果或 reporting-only test 结果当作当前主线证据。

当前阶段可靠的结论是结构性结论：

> GCS-YOLO-Lane provides a code-verified YOLO11-style structured lane detector whose default output is a visibility-aware fixed-y lane point set, trained with Hungarian set matching and geometry-aware supervision.

## 18. 实现证据索引

| 内容 | 主要文件 |
|---|---|
| 默认模型结构 | `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml` |
| 数据合同 | `data/tusimple_gcs_fixed_y_960x544.yaml` |
| LSEM、LaneBiFPN、GCSLaneHead | `ultralytics/nn/modules/gcs_lane.py` |
| Hungarian matching | `ultralytics/utils/gcs_matcher.py` |
| 默认 loss | `ultralytics/utils/gcs_loss.py` |
| 推理解码与 lane NMS | `ultralytics/utils/gcs_postprocess.py` |
| TuSimple 标签转换 | `tools/convert_tusimple_to_gcs.py` |
| fixed-y 采样与 mask 生成 | `gcs_tools/label_utils.py` |
| active branch contract | `docs/agent-context/current-contracts.md` |

本文件只负责算法核心整理。实验命令、official-val checkpoint selection、baseline 表、消融表和最终 test 结果应分别写入论文实验文档，不应在这里提前填入未经验证的数字。
