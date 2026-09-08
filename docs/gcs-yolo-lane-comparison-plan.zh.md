# GCS-YOLO-Lane 分数据集对比方案

整理日期：2026-09-06。

状态：算法选择建议，不代表已复现、已有可比结果或已确定最终训练配置。
本文将经典与代表性方法按 TuSimple 和 CULane 分开；不构成 2026 年最新方法的完整检索。

## TuSimple

侧重点：固定采样位置的定位精度、车道实例预测、结构化表示和推理效率。
建议主表包含以下 10 个对照方法，再加入 GCS-YOLO-Lane。

| 方法 | 发表信息 | 对比作用 |
|---|---|---|
| SCNN | AAAI 2018 | 经典空间信息传播与分割基线 |
| ENet-SAD | ICCV 2019 | 轻量分割与自注意力蒸馏基线 |
| UFLD | ECCV 2020 | 经典逐行位置分类与实时检测基线 |
| UFLDv2 | TPAMI，2022 在线发表 | 混合锚点与序数分类的高效强基线 |
| RESA | AAAI 2021 | 方向性空间聚合与上下文建模对照 |
| LaneATT | CVPR 2021 | 线锚点与注意力机制对照 |
| LSTR | WACV 2021 | Transformer 集合预测与车道形状参数回归，关联 GCS 的 query 设计 |
| PolyLaneNet | ICPR 2020，论文集出版于 2021 | 早期多项式曲线回归，关联点序列与整体曲线表示的区别 |
| BezierLaneNet | CVPR 2022 | Bézier 曲线建模，关联结构化几何输出与效率 |
| CLRNet | CVPR 2022 | 跨层特征聚合和车道精修的精度强基线 |

首批实际复现的 6 个方法：SCNN、UFLDv2、LaneATT、LSTR、BezierLaneNet、CLRNet。
若 LSEM 是论文重点贡献，将 RESA 提前纳入首批复现。

可选补充：CondLaneNet、GANet、FOLOLane、PINet、LaneNet（Neven 等，IV 2018）。
CondLaneNet 和 GANet 也有 TuSimple 结果；将其列为补充是控制主表长度的选择，不是数据集不兼容。

主指标：官方 Accuracy、FP、FN。效率列报告参数量、FLOPs、延迟或 FPS。
额外 F1 必须说明定义及实现；项目内部几何 F1 和组合分数不能直接作为跨论文通用指标。

## CULane

侧重点：遮挡、拥挤、夜间、阴影、弯道和无车道线场景中的实例检测、几何定位与误检。
建议主表包含以下 10 个对照方法，再加入 GCS-YOLO-Lane。

| 方法 | 发表信息 | 对比作用 |
|---|---|---|
| SCNN | AAAI 2018 | 经典 CULane 分割与空间传播基线 |
| ENet-SAD | ICCV 2019 | 轻量网络与上下文蒸馏基线 |
| UFLD | ECCV 2020 | 经典高效逐行分类基线 |
| UFLDv2 | TPAMI，2022 在线发表 | 高效行列锚点表示强基线 |
| RESA | AAAI 2021 | 遮挡、弱纹理等场景的空间聚合对照 |
| LaneATT | CVPR 2021 | 线锚点定位与全局注意力对照 |
| CondLaneNet | ICCV 2021 | 条件卷积、车道实例生成与逐行定位对照 |
| GANet | CVPR 2022 | 关键点全局关联与实例分组对照 |
| CLRNet | CVPR 2022 | 多尺度特征利用和车道精修的精度强基线 |
| CLRerNet | WACV 2024 | 置信度学习与 LaneIoU 的较新强基线补充 |

首批实际复现的 6 个方法：SCNN、RESA、UFLDv2、LaneATT、CondLaneNet、CLRNet。
第二批优先 GANet、CLRerNet；后者属于较新补充，不归为早期经典方法。

可选补充：BezierLaneNet、FOLOLane、PINet、LaneAF。
BezierLaneNet 已有 CULane 评测；若强调曲线表示，优先将它加入主表。
LSTR 和 PolyLaneNet 的原始主要定量证据更偏 TuSimple；CULane 采用它们之前需核验对应实现与结果协议。

主指标：官方协议下的总体 F1，以及 Normal、Crowded、Dazzle、Shadow、No line、Arrow、Curve、Night 分场景 F1。
Cross/Crossroad 报告误检数量 FP，越低越好，不能当作 F1。
项目 `ape` 几何评测结果不能代替 CULane benchmark F1；参见 [CULane 评测说明](culane_evaluation.md)。

## 两套名单的复用

- 共同主表方法：SCNN、ENet-SAD、UFLD、UFLDv2、RESA、LaneATT、CLRNet，共 7 个。
- TuSimple 额外主表方法：LSTR、PolyLaneNet、BezierLaneNet，共 3 个。
- CULane 额外主表方法：CondLaneNet、GANet、CLRerNet，共 3 个。
- 两张完整主表合计涉及 13 种对照算法。每个数据集仍须分别训练或采用其对应权重，不能默认直接复用另一数据集的 checkpoint。

## 结果填写要求

1. 区分原论文结果、公开权重复测和自行训练结果；为每行记录来源、backbone、输入分辨率、预训练来源及训练数据。
2. 自行复现时明确 train/val/test 图像清单和标注转换；保留各方法原生输出表示，统一到相同官方评测格式。
3. 在验证集选择 checkpoint、阈值和后处理，再固定最终候选做测试；不以测试结果反向筛选算法配置。
4. 不强制其他方法使用 GCS 的内部 K56 标签或 query 数量。GCS 的 TuSimple 输入维持 `--imgsz 544 960`；其他方法的原生分辨率须注明。
5. 速度比较统一硬件、batch、数值精度和推理框架，说明预处理、前向、后处理的计时范围。原文 FPS 只能作为带来源的参考。
6. 整网对比用于检验整体竞争力；LSEM、LaneBiFPN、辅助监督等模块的贡献需通过相同骨干下的消融验证。

## 证据与待核验项

方法定位依据项目 `paper/classic/` 与 `paper/recent/` 中保存的原论文 Markdown 转写，以及综述中的方法和数据集对比。
公开仓库链接可用于后续核验，不代表已经安装或测试。

- [论文规划中的对比框架](gcs-yolo-lane-paper-plan.zh.md)
- [本地车道线检测综述](../paper/survey_dataset/MinerU_markdown_2024_MonocularLaneSurvey_Li_2068783662725160966.md)
- [LSTR 原论文转写](../paper/recent/MinerU_markdown_2021_LSTR_Liu_2068782531613646857.md)
- [BezierLaneNet 原论文转写](../paper/recent/MinerU_markdown_2022_BezierLaneNet_Feng_2068782531613646858.md)
- [GANet 原论文转写](../paper/recent/MinerU_markdown_2022_GANet_Wang_2068782531613646860.md)
- [CLRNet 作者仓库](https://github.com/Turoad/CLRNet)
- [CondLaneNet 作者仓库](https://github.com/aliyun/conditional-lane-detection)
- [CLRerNet 论文页面](https://openaccess.thecvf.com/content/WACV2024/html/Honda_CLRerNet_Improving_Confidence_of_Lane_Detection_With_LaneIoU_WACV_2024_paper.html)

本次未执行模型复现，也未逐项核对每个配置的最终测试数值。CLRerNet 的发表信息及定位有本地综述支撑；其具体复现配置须从原论文和作者实现核验。正式投稿前需额外检索与论文定位接近的 2025-2026 年方法。
