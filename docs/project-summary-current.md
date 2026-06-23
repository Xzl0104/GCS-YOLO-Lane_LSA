# 当前算法实现

## 算法大纲

当前项目实现的是 GCS-YOLO-Lane 结构化车道线检测：输入 TuSimple 图像，输出最多 `Q=12` 条 lane query；每条 lane 在固定的 `K=56` 个 y 锚点上预测 x 坐标、lane 存在性和逐点可见性。模型不以普通检测框作为主输出，也不以分割 mask 作为最终 lane 输出；分割 mask 和 edge mask 只作为训练期辅助监督。

当前主线配置与入口：

- 模型配置：`ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`
- 数据配置：`data/tusimple_gcs_fixed_y_960x544.yaml`
- 标签生成：`tools/convert_tusimple_to_gcs.py`
- 训练入口：`tools/train_gcs.py`
- 推理入口：`tools/infer_gcs.py`
- 自定义 GCS 评估：`tools/eval_gcs.py`
- TuSimple official 评估：`tools/eval_tusimple_official.py`
- TuSimple official-val sweep：`tools/sweep_tusimple_official.py`

核心数据与形状契约：

- TuSimple 输入尺寸固定为 `--imgsz 544 960`，顺序为 `H,W`
- 数据根目录：`datasets/tusimple_fixed_y_k56_960x544`
- split 数量：`train=3263`，`val=363`，`test=2782`
- `point_mode = fixed_y`
- `Q = 12`
- `K = 56`
- fixed-y anchors：`710, 700, 690, ..., 160`
- 归一化 fixed-y 范围：`fixed_y_start = 710 / 720 = 0.9861111111111112`，`fixed_y_end = 160 / 720 = 0.2222222222222222`
- K56 标签必须由原始 TuSimple JSON 和图像重新生成，不从旧 K32 标签重采样

端到端流程：

1. `tools/convert_tusimple_to_gcs.py` 读取原始 TuSimple JSON 和图像，将图像 resize 到 `544 x 960`，将每条 lane 插值到 56 个固定 y 锚点，并生成 `.npz` 标签。
2. `ultralytics/data/dataset_gcs.py::GCSLaneDataset` 读取 `images/<split>` 和 `labels_gcs/<split>/*.npz`，校验 fixed-y、lane 顺序、mask/edge 形状，并在 batch 中保留可变数量 lane 的 list 结构。
3. `tools/train_gcs.py` 构造 `ultralytics/models/yolo/gcs_lane/train.py::GCSLaneTrainer`，锁定矩形输入尺寸，构建 `ultralytics/nn/tasks.py::GCSLaneModel`，并按 GCS 任务接入 `GCSLoss`。
4. 模型使用 YOLO11-style backbone、P3/P4 的 `LSEM`、P2-P5 的 `LaneBiFPN` 和 query-based `GCSLaneHead` 输出结构化 lane。
5. 训练时 `ultralytics/utils/gcs_matcher.py::GCSHungarianMatcher` 先对 query 与 GT lane 做 Hungarian matching，随后 `ultralytics/utils/gcs_loss.py::GCSLoss` 计算存在性、点回归、可见性、曲率/平滑、辅助 mask/edge 和可选 count/margin 类 loss。
6. 推理时 `ultralytics/utils/gcs_postprocess.py::decode_gcs_predictions` 按 query score、逐点可见性、最长连续可见段、`min_points`、可选 lane NMS 和 `max_det` 解码 lane。
7. official 评估时 `gcs_tools/tusimple_official_eval.py::gcs_lanes_to_tusimple_lanes()` 将 fixed-y GCS 输出插值到 TuSimple `h_samples`，再用 TuSimple official metric 计算 `ACC/FP/FN`。

兼容路径仍存在，但新命令主路径以上述模型和数据配置为准：

- `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56.yaml`
- `ultralytics/cfg/models/gcs/gcs-yolo-lane-s-fixed-y.yaml`
- `data/tusimple_gcs_fixed_y_k56_960x544.yaml`

## 标签生成

标签生成入口为 `tools/convert_tusimple_to_gcs.py`。核心函数为：

- `build_gcs_arrays()`
- `convert_one()`
- `validate_gcs_arrays()`
- `gcs_tools/label_utils.py::sample_polyline_fixed_y()`
- `gcs_tools/label_utils.py::build_semantic_mask()`
- `gcs_tools/label_utils.py::build_edge_mask()`

生成流程：

1. 从 TuSimple archive 读取原始 `raw_file`、`h_samples` 和 lane x 坐标。
2. 通过 `resize_image_and_lanes()` 将图像与 lane 点同步缩放到 `H,W = 544,960`。
3. 每条 lane 先执行清洗：过滤非法/负坐标，按 y 从底到顶排序，并去掉重复 y。
4. `fixed_y_anchors()` 生成 56 个共享 y 锚点：`np.linspace(710/720, 160/720, 56)`。
5. `sample_polyline_fixed_y()` 将 fixed-y 锚点映射到 resize 后的像素 y 坐标，在 lane polyline 上按 y 插值得到 x；在 lane y 范围内且 x 位于图像内的 anchor 标为有效点。
6. 输出 lane 标签保持完整 `K=56` 行：有效 anchor 存储归一化 `(x,y)` 并令 `lane_valid=1`；无效 anchor 保留固定 y、`x=0`、`lane_valid=0`。
7. 保留有效点数小于 2 的 lane 会被丢弃。
8. 对保留下来的 lane 生成辅助 `semantic_mask` 和 `edge_mask`。
9. 每张图保存一个压缩 `.npz` 标签。

`.npz` 标签字段：

- `semantic_mask`: `H x W` 二值 lane 区域 mask
- `edge_mask`: `H x W` 二值 edge mask
- `lanes`: `N x 56 x 2`，归一化坐标，点序为 bottom-to-top
- `lane_valid`: `N x 56`，逐 anchor 可见性
- `num_lanes`: 当前图像有效 lane 数
- `point_mode`: `fixed_y`
- `fixed_y`: `56` 个固定 y 归一化锚点
- `raw_file`: 原始 TuSimple 相对路径
- `image_shape`: `[544, 960]`
- `num_points`: `56`

标签生成命令：

```bash
python tools/convert_tusimple_to_gcs.py \
  --archive-root archive/TUSimple \
  --output-root datasets/tusimple_fixed_y_k56_960x544 \
  --imgsz 544 960 \
  --point-mode fixed_y \
  --num-points 56 \
  --fixed-y-start 0.9861111111111112 \
  --fixed-y-end 0.2222222222222222
```

标签校验要求：

- `semantic_mask` 和 `edge_mask` 必须与训练图像同为 `544 x 960`
- `lanes` 必须是 `N x 56 x 2`
- `lane_valid` 必须是 `N x 56`
- fixed-y anchors 必须严格从大到小排列
- valid 点的 y 必须与 `fixed_y` 对齐
- lane 坐标必须为 `[0,1]` 归一化值

## 数据加载与训练入口

`GCSLaneDataset` 加载并校验固定 y 标签：

- 图像路径来自 `images/<split>`
- 标签路径来自 `labels_gcs/<split>/*.npz`
- `point_mode` 在整个 dataset 内必须一致
- fixed-y anchors 必须严格 descending 且在 `[0,1]`
- fixed-y 模式不支持 mosaic 和 vertical flip
- horizontal flip 会更新 x 为 `1 - x`
- scale augmentation 会先仿射变换 lane，再重新采样回共享 fixed-y anchors
- batch collate 时图像、`semantic_mask`、`edge_mask` 被 stack；`lanes` 与 `lane_valid` 保持 per-image list，用于 Hungarian matching

`GCSLaneTrainer` 的关键行为：

- `task` 固定为 `gcs_lane`
- `_lock_gcs_shape_contract()` 将真实 GCS 输入锁定为 `[544, 960]`，拒绝 square-only `imgsz`
- `build_dataset()` 构建 `GCSLaneDataset`
- `_check_point_mode_contract()` 校验模型 head 与标签 `point_mode` 一致
- `get_model()` 构建 `GCSLaneModel`
- `load_gcs_pretrained()` 从普通 YOLO11 checkpoint 只迁移 backbone 权重；由于 GCS backbone 插入了 LSEM，YOLO11 layer index 会显式 remap；`LSEM`、`LaneBiFPN` 和 `GCSLaneHead` 保持 GCS 初始化
- `--gcs-lane-count-balanced` 默认启用，按 GT lane count 做 inverse-frequency replacement sampling

正式 TuSimple 训练的主配置口径：

```bash
python tools/train_gcs.py \
  --dataset tusimple \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --pretrained yolo11s-seg.pt \
  --imgsz 544 960
```

## 模型结构

模型配置为 `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`。

Backbone：

- YOLO11s 风格 backbone
- `Conv/C3k2/SPPF/C2PSA` 构成主干特征提取
- P3 路径后插入 `LSEM`
- P4 路径后插入 `LSEM`

`LSEM` 位于 `ultralytics/nn/modules/gcs_lane.py`，由以下部分组成：

- `LineStripAttention`: 水平方向 strip depthwise conv 与垂直方向 strip depthwise conv
- `direction_gate`: 对水平/垂直 strip 响应做 softmax 加权
- `CoordReweight`: 分别沿 height/width 做 coordinate-aware reweighting
- `dilated_context`: depthwise dilated context conv
- residual connection: 输出与输入残差相加后激活

`LaneBiFPN`：

- 输入 P2、增强后的 P3、增强后的 P4、P5
- 每个尺度先通过 `1x1 ConvBNAct` 对齐到 `128` channels
- 使用 `WeightedFusion` 做非负归一化加权融合
- 包含 top-down 和 bottom-up 双向融合
- 输出 4 个统一通道的空间特征图，供 lane head flatten 成 token

`GCSLaneHead`：

- `num_queries = 12`
- `num_points = 56`
- Transformer decoder 层数为 `3`
- attention heads 为 `8`
- `point_mode = fixed_y`
- fixed-y 模式下 `point_dims = 1`，head 只预测每个 y anchor 的 x
- `fixed_y_anchors` 由 head buffer 保存，范围为 `710/720 -> 160/720`
- `query_embed` 是 12 个 learnable lane query
- P2-P5 特征被 flatten 成 spatial tokens，并叠加 2D sin-cos position embedding 与 level embedding
- Transformer decoder 让 lane query attend 到 P2-P5 空间 token
- `point_mlp` 输出每个 query 的 56 个 x logits
- `point_reference_logits` 为每个 query 提供不同的初始透视形状参考
- `_refine_fixed_y_logits()` 在预测点位置从图像特征采样 token，对 x logits 做 image-conditioned refinement
- `point_valid_mlp` 和 `_refine_fixed_y_valid_logits()` 输出逐 anchor 可见性 logits
- `exist_mlp` 输出每条 query 的 lane existence logit
- `aux_mask` 从 P2 输出 `B x 2 x H x W`
- `aux_edge` 从 P2 输出 `B x 1 x H x W`

模型 YAML 的 head 定义：

```yaml
head:
  - [[2, 5, 8, 12], 1, LaneBiFPN, [128]]
  - [-1, 1, GCSLaneHead, [12, 56, 3, 8, True, fixed_y, 0.9861111111111112, 0.2222222222222222]]
```

## 模型输出契约

默认 K56 模型输出：

```text
pred_points: B x 12 x 56 x 2
pred_logits: B x 12
pred_valid_logits: B x 12 x 56
aux_mask_logits: B x 2 x H x W
aux_edge_logits: B x 1 x H x W
```

fixed-y 模式下：

- `pred_points[..., 0]` 是模型预测的归一化 x
- `pred_points[..., 1]` 不是自由回归值，而是固定 y anchor
- `pred_logits` 是 query-level lane existence logits
- `pred_valid_logits` 是每个 query、每个 fixed-y anchor 的可见性 logits
- `aux_mask_logits` 和 `aux_edge_logits` 用于训练辅助监督

模型 shape 检查命令：

```bash
python tools/check_model.py \
  --cfg ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --imgsz 544 960 \
  --batch 1
```

期望辅助输出尺寸：

```text
aux_mask_logits: B x 2 x 544 x 960
aux_edge_logits: B x 1 x 544 x 960
```

## 匹配与 loss

匹配模块为 `ultralytics/utils/gcs_matcher.py::GCSHungarianMatcher`。它对每张图单独构造 `Q x N` cost matrix，其中 `Q=12` 为 query 数，`N` 为当前图 GT lane 数。

matching cost：

```text
cost = gcs_cost_point * point_cost
     + gcs_cost_curve * curve_cost
     + gcs_cost_exist * cost_exist
```

默认 matching 配置：

- `gcs_cost_point = 5.0`
- `gcs_cost_curve = 0.05`
- `gcs_cost_exist = 0.1`
- `gcs_match_min_overlap = 2`
- `gcs_match_max_x_dist = 0.0`
- `gcs_match_gate_px = 160.0`

matching cost 细节：

- `point_cost`: 在 GT 可见 anchor 上计算预测点与 GT 点的 aspect-weighted normalized L1 距离
- `curve_cost`: 在连续三点都可见的 anchor triplet 上比较二阶曲率差，按真实图像像素比例缩放
- `cost_exist`: `-sigmoid(pred_logits)`，存在性越高 cost 越低
- `match_gate_px`: 可选 APE 像素门控；默认 `160.0`
- Hungarian assignment 使用 `scipy.optimize.linear_sum_assignment`

loss 模块为 `ultralytics/utils/gcs_loss.py::GCSLoss`。`forward()` 流程：

1. 校验 `pred_points`、`pred_logits`、`pred_valid_logits` 与 batch 中 image/mask 的形状。
2. 从 batch 读取 `lanes` 和 `lane_valid`。
3. 调用 `GCSHungarianMatcher` 得到每张图的 `(query_idx, gt_idx)`。
4. 计算各 loss item。
5. 按 gain 加权求和，返回 `total` 和 detach 后的 loss item 向量。

当前记录的 loss items：

```text
exist_loss
point_loss
lane_balanced_point_loss
point_valid_loss
short_valid_recall_loss
smooth_loss
curve_loss
mask_loss
edge_loss
count_loss
count_under5_loss
duplicate_margin_loss
spurious_margin_loss
```

默认启用的基础 loss gain：

- `gcs_exist = 2.0`
- `gcs_point = 15.0`
- `gcs_point_valid = 1.0`
- `gcs_smooth = 0.05`
- `gcs_curve = 0.1`
- `gcs_mask = 0.2`
- `gcs_edge = 0.2`

默认关闭、仅在显式实验参数中启用的 loss gain：

- `gcs_lane_balanced_point = 0.0`
- `gcs_short_valid_recall = 0.0`
- `gcs_count = 0.0`
- `gcs_count_under5 = 0.0`
- `gcs_duplicate_margin = 0.0`
- `gcs_spurious_margin = 0.0`

各 loss 实现：

- `exist_loss`: 对所有 query 做 BCE。unmatched query 的 target 为 0；matched query 的 target 不是固定 1，而是由几何 APE 与 visible-IoU 得到的 quality target。默认 `gcs_exist_quality_alpha=1.0`，`linear` 模式下 APE 小于 `gcs_exist_quality_pos_px=10.0` 视为高质量，APE 大于 `gcs_exist_quality_neg_px=20.0` 视为低质量。
- `point_loss`: 对 Hungarian matched lane，在 GT 可见 anchor 上计算 aspect-weighted L1 点误差。
- `lane_balanced_point_loss`: matched point loss 先按 lane 平均、再跨 lane 平均；默认 gain 为 0。
- `point_valid_loss`: 对 `pred_valid_logits` 做 BCE。matched query 的 target 为 GT `lane_valid`；unmatched query 的 target 为全 0；正样本权重 capped by `gcs_point_valid_pos_weight_max=10.0`。
- `short_valid_recall_loss`: 对几何匹配合理的短 GT lane 做 positive-only visibility BCE；默认 gain 为 0。
- `smooth_loss`: 对 matched lane 的预测点做二阶差分平滑正则，只在 GT 连续可见 triplet 上计算。
- `curve_loss`: 对 matched lane 的预测二阶曲率与 GT 二阶曲率做 SmoothL1，并用 GT 曲率幅值自适应加权。
- `mask_loss`: 对 `aux_mask_logits` 和 `semantic_mask` 做 class-weighted cross entropy + Dice loss。
- `edge_loss`: 对 `aux_edge_logits` 和 `edge_mask` 做 weighted BCE + Dice loss。
- `count_loss`: 令 `sum(sigmoid(pred_logits))` 拟合 GT lane count；代码默认 gain 为 0，正式实验可通过 CLI 显式启用。
- `count_under5_loss`: 对 GT lane 数大于等于 `gcs_count_under5_min_lanes=5` 的样本，惩罚 `target_count - pred_count` 的 undercount gap；代码默认 gain 为 0，正式实验可通过 CLI 显式启用。
- `duplicate_margin_loss`: 对可靠 matched `q+` 与 duplicate-like unmatched `q-` 施加 pairwise logit margin；默认 gain 为 0。
- `spurious_margin_loss`: 对可靠 matched `q+` 与 far-spurious unmatched `q-` 施加 pairwise logit margin；默认 gain 为 0。

## 解码与推理

推理入口：

```bash
python tools/infer_gcs.py \
  --weights <weights.pt> \
  --source <images-or-list> \
  --imgsz 544 960
```

解码函数为 `ultralytics/utils/gcs_postprocess.py::decode_gcs_predictions()`。

解码流程：

1. 对 `pred_logits` 做 sigmoid 得到 query score。
2. 保留 `score >= score_thr` 的 query。
3. 对每条 lane 的 56 个点按 y 从底到顶排序。
4. 对 `pred_valid_logits` 做 sigmoid 得到逐 anchor 可见性概率。
5. 使用 `point_valid_thr` 得到可见 anchor mask。
6. `longest_contiguous_valid_mask()` 只保留最长连续可见 anchor 段。
7. 可见点数小于 `min_points` 的 lane 被过滤。
8. 以 score 降序排序。
9. 如 `nms_dist_px > 0`，执行 `lane_nms()`；lane NMS 使用共享 fixed-y 点序上的平均 x 像素距离，并可结合可见 anchor mask。
10. 按 `max_det` 截断保留 lane 数。
11. 输出每条 lane 的 `score`、`query`、`points_norm`、`point_valid`、`visible_points_norm`，并在提供 `image_shape` 时输出像素坐标。

TuSimple official 转换：

- `gcs_tools/tusimple_official_eval.py::_lane_points_for_official()` 只使用 visible points。
- `gcs_lanes_to_tusimple_lanes()` 将 GCS visible points 映射到原始 TuSimple 图像尺寸，再插值到官方 `h_samples`。
- 插值失败或越界的 x 写为 `-2`。
- 少于 2 个有效 x 的 lane 不进入 official prediction。

official-val sweep：

```bash
python tools/sweep_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split val \
  --weights <weights.pt> \
  --imgsz 544 960 \
  --device 0
```

`tools/sweep_tusimple_official.py` 的选择规则：

- 禁止 `--split test` 用于 threshold/postprocess 搜索
- 对每个 `conf`、`point_valid_thr`、`nms_dist_px`、`max_det`、`min_points` 组合生成 TuSimple prediction
- 使用 TuSimple official evaluator 得到 `official_acc`、`official_FP`、`official_FN`
- 主要按 `official_acc` 选 best row
- tie-breaker 依次包括 `official_score`、更低 FP、更低 FN、更高 `count_acc` 和 decode 参数

final test：

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights <selected-weights.pt> \
  --imgsz 544 960 \
  --device 0 \
  --conf <selected-conf> \
  --point-valid-thr <selected-point-valid-thr> \
  --nms-dist-px <selected-nms-dist-px> \
  --max-det <selected-max-det> \
  --min-points <selected-min-points>
```

# 最佳实验结果

## official-val 最优结果

当前项目记录中，按 official-val ACC 选择的最高结果为：

- run：`gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03`
- weights：`/root/GCS-YOLO-Lane_LSA_5-25-3-k56/runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03/weights/best.pt`
- train args：`/root/GCS-YOLO-Lane_LSA_5-25-3-k56/runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03/args.yaml`
- official-val sweep：`/root/GCS-YOLO-Lane_LSA_5-25-3-k56/runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val_sweep/tusimple_official_sweep_summary.json`
- official-val split：363 images
- selected decode：`conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4`
- 复现代码状态：该 run 使用包含 `--gcs-gt4-short-*` 实验参数的已完成实验代码状态；上文“当前算法实现”不把该参数作为当前默认算法配置。

official-val metrics：

```text
official_acc   = 0.970851
official_FP    = 0.022084
official_FN    = 0.011708
official_score = 0.970175
count_acc      = 0.939394
count_acc_3    = 0.964126
count_acc_4    = 0.848485
count_acc_5    = 0.945946
images         = 363
```

official-val lane count distribution：

```text
pred_lanes_hist = 3:216, 4:67, 5:78, 6:2
gt_lanes_hist   = 3:223, 4:66, 5:74
count_confusion = 3->3:215, 3->4:8,
                  4->3:1, 4->4:56, 4->5:8, 4->6:1,
                  5->4:3, 5->5:70, 5->6:1
```

关键训练配置：

```bash
python tools/train_gcs.py \
  --dataset tusimple \
  --model ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml \
  --data data/tusimple_gcs_fixed_y_960x544.yaml \
  --pretrained yolo11s-seg.pt \
  --imgsz 544 960 \
  --epochs 160 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --no-amp \
  --optimizer AdamW \
  --lr0 5e-4 \
  --lrf 0.05 \
  --cos-lr \
  --weight-decay 1e-4 \
  --warmup-epochs 3.0 \
  --warmup-bias-lr 0.0 \
  --patience 40 \
  --erasing 0.1 \
  --scale 0.3 \
  --gcs-exist 2.0 \
  --gcs-point 15.0 \
  --gcs-point-valid 1.0 \
  --gcs-smooth 0.05 \
  --gcs-curve 0.1 \
  --gcs-mask 0.2 \
  --gcs-edge 0.2 \
  --gcs-count 0.3 \
  --gcs-count-under5 0.3 \
  --gcs-count-under5-min-lanes 5 \
  --gcs-lane-count-balanced \
  --gcs-lane-count-balance-power 1.0 \
  --gcs-lane-count-min-group 50 \
  --gcs-gt4-short-boost 1.5 \
  --gcs-gt4-short-min-visible-max 10 \
  --project runs/gcs_lane \
  --name gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03
```

official-val selected row 复现命令：

```bash
python tools/sweep_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split val \
  --gt-json runs/gcs_lane/tusimple_official_val_363_folder_aware_seed20260602_subset/labels/tusimple_official_val_363_folder_aware_seed20260602.json \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --half \
  --confs 0.15 \
  --point-valid-thrs 0.5 \
  --nms-dist-pxs 0.0 \
  --max-dets 6 \
  --min-points 4 \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_val_sweep
```

对应 final-test 报告：

- summary：`/root/GCS-YOLO-Lane_LSA_5-25-3-k56/runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json`
- split：test
- images：`2782`
- selected decode：`conf=0.15, point_valid_thr=0.5, nms_dist_px=0.0, max_det=6, min_points=4`

final-test metrics：

```text
official_acc   = 0.965369
official_FP    = 0.033309
official_FN    = 0.029236
official_score = 0.964118
count_acc      = 0.864486
count_acc_3    = 0.974713
count_acc_4    = 0.482906
count_acc_5    = 0.845343
avg_total_ms   = 11.4977
```

official-test 复现命令：

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.15 \
  --point-valid-thr 0.5 \
  --nms-dist-px 0.0 \
  --max-det 6 \
  --min-points 4 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_gt4short15_count03_under5_03_official_test_best_from_val \
  --save-records
```

## reporting-only final-test 最高结果

当前项目记录中，reporting-only final-test ACC 最高的结果为：

- run：`gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03`
- weights：`/root/GCS-YOLO-Lane_LSA_5-25-3-k56/runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03/weights/best.pt`
- official-val sweep：`/root/GCS-YOLO-Lane_LSA_5-25-3-k56/runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_val_sweep/tusimple_official_sweep_summary.json`
- final-test summary：`/root/GCS-YOLO-Lane_LSA_5-25-3-k56/runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_test_best_from_val/tusimple_official_summary.json`
- official-val selected decode：`conf=0.05, point_valid_thr=0.45, nms_dist_px=0.0, max_det=6, min_points=6`
- 该结果不替代 official-val 最优选择口径

official-val metrics：

```text
official_acc   = 0.970272
official_FP    = 0.024564
official_FN    = 0.016070
official_score = 0.969459
count_acc      = 0.953168
count_acc_3    = 0.950673
count_acc_4    = 0.939394
count_acc_5    = 0.972973
images         = 363
```

official-val lane count distribution：

```text
pred_lanes_hist = 3:213, 4:75, 5:75
gt_lanes_hist   = 3:223, 4:66, 5:74
count_confusion = 3->3:212, 3->4:11,
                  4->3:1, 4->4:62, 4->5:3,
                  5->4:2, 5->5:72
```

关键训练配置：

```text
epochs = 160
batch = 32
workers = 8
amp = true
gcs_count = 0.3
gcs_count_under5 = 0.3
gcs_duplicate_margin = 0.05
```

official-test 命令：

```bash
python tools/eval_tusimple_official.py \
  --archive-root archive/TUSimple \
  --split test \
  --weights runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03/weights/best.pt \
  --imgsz 544 960 \
  --device 0 \
  --conf 0.05 \
  --point-valid-thr 0.45 \
  --nms-dist-px 0.0 \
  --max-det 6 \
  --min-points 6 \
  --half \
  --save-dir runs/gcs_lane/gcs_yolo_lane_s_tusimple_fixed_y_dupmargin005_count03_under5_03_official_test_best_from_val \
  --save-records
```

final-test metrics：

```text
images         = 2782
official_acc   = 0.965702
official_FP    = 0.029493
official_FN    = 0.027348
official_score = 0.964565
count_acc      = 0.865924
count_acc_2    = 0.200000
count_acc_3    = 0.971839
count_acc_4    = 0.547009
count_acc_5    = 0.810193
avg_total_ms   = 11.0472
```

final-test lane count distribution：

```text
pred_lanes_hist = 2:3, 3:1837, 4:371, 5:557, 6:14
gt_lanes_hist   = 2:5, 3:1740, 4:468, 5:569
count_confusion = 2->2:1, 2->3:3, 2->4:1,
                  3->2:2, 3->3:1691, 3->4:41, 3->5:6,
                  4->3:118, 4->4:256, 4->5:90, 4->6:4,
                  5->3:25, 5->4:73, 5->5:461, 5->6:10
```
