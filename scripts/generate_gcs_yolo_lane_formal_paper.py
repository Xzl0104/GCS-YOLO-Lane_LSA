from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "gcs_yolo_lane_paper"
FIG_DIR = OUT / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_gcs_yolo_lane_manuscript_package as previous_package  # noqa: E402


@dataclass(frozen=True)
class Block:
    kind: str
    value: str


def p(text: str) -> Block:
    return Block("p", " ".join(text.strip().split()))


def formula(text: str) -> Block:
    return Block("formula", text.strip())


def fig(key: str) -> Block:
    return Block("figure", key)


def table(key: str) -> Block:
    return Block("table", key)


CITATION_NUM = {entry.key: i + 1 for i, entry in enumerate(previous_package.BIB_ENTRIES)}


def render_cites(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        keys = [k.strip() for k in match.group(1).split(";")]
        nums = [str(CITATION_NUM[k]) for k in keys if k in CITATION_NUM]
        return "[" + ", ".join(nums) + "]" if nums else match.group(0)

    return re.sub(r"\[@([^\]]+)\]", repl, text)


FIGURES = {
    "overview": {
        "file": "figure_01_overview.png",
        "en": "Overall framework of GCS-YOLO-Lane. A YOLO11-style feature extractor is converted into a structured lane detector by line-sensitive feature enhancement, bidirectional feature fusion, and a query-based fixed-y lane head.",
        "zh": "GCS-YOLO-Lane 整体框架。YOLO11-style 特征提取器通过线敏感特征增强、双向特征融合和 query-based fixed-y lane head 被改造为结构化车道检测器。",
    },
    "fixed_y": {
        "file": "figure_02_fixed_y_formal.png",
        "en": "Fixed-y K56 lane representation. Each lane query owns x coordinates and visibility logits on the same 56 vertical anchors, while the y coordinates are shared and deterministic.",
        "zh": "Fixed-y K56 车道表示。每个 lane query 在同一组 56 个纵向 anchors 上拥有 x 坐标和 visibility logits，而 y 坐标由共享 anchor 决定。",
    },
    "head": {
        "file": "figure_03_query_head.png",
        "en": "Query-based GCS lane head. Multi-scale tokens are attended by learnable lane queries, and each query predicts lane existence, point coordinates, and point visibility.",
        "zh": "Query-based GCS lane head。多尺度 tokens 被可学习 lane queries 访问，每个 query 同时预测 lane existence、point coordinates 和 point visibility。",
    },
    "protocol": {
        "file": "figure_04_protocol_formal.png",
        "en": "Training and evaluation protocol. Training uses converted K56 labels and Hungarian assignment. Official validation selects checkpoint and decoding parameters; final test is used once for reporting.",
        "zh": "训练与评估协议。训练使用转换后的 K56 标签和 Hungarian assignment。Official validation 用于选择 checkpoint 与 decode 参数；final test 只用于一次性报告。",
    },
    "failure": {
        "file": "figure_05_failure_modes.png",
        "en": "Planned failure analysis categories for structured lane prediction. Count, visibility, geometry, duplicate, and spurious-query errors are separated for diagnosis.",
        "zh": "结构化车道预测的计划失败分析类别。诊断时将 count、visibility、geometry、duplicate 和 spurious-query errors 分开统计。",
    },
}


TABLES_EN = {
    "notation": (
        "Table 1. Notation used in GCS-YOLO-Lane.",
        ["Symbol", "Type", "Definition"],
        [
            ["B", "scalar", "Batch size"],
            ["Q", "scalar", "Number of lane queries, fixed to 12 in the default model"],
            ["K", "scalar", "Number of fixed-y anchors, fixed to 56 in the TuSimple contract"],
            ["I", "tensor", "Input image resized to 544 x 960 in H,W order"],
            ["F_l", "tensor", "Feature map at pyramid level l"],
            ["z_q", "tensor", "Feature vector of the q-th lane query"],
            ["P", "tensor", "Predicted point tensor with shape B x Q x K x 2"],
            ["s", "tensor", "Lane existence logits with shape B x Q"],
            ["v", "tensor", "Point-valid logits with shape B x Q x K"],
            ["A_k", "scalar", "The k-th fixed-y anchor normalized by the original TuSimple height"],
        ],
    ),
    "contract": (
        "Table 2. Active TuSimple K56 contract used in this manuscript.",
        ["Item", "Value"],
        [
            ["Model configuration", "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"],
            ["Data configuration", "data/tusimple_gcs_fixed_y_960x544.yaml"],
            ["Image size", "--imgsz 544 960, interpreted as H,W"],
            ["Dataset splits", "train 3263, validation 363, test 2782"],
            ["Lane queries", "Q = 12"],
            ["Fixed-y anchors", "K = 56, y = 710, 700, ..., 160"],
            ["Final lane outputs", "pred_points, pred_logits, pred_valid_logits"],
            ["Auxiliary outputs", "aux_mask_logits and aux_edge_logits for training only"],
        ],
    ),
    "environment": (
        "Table 3. Experimental implementation record template.",
        ["Setting", "Value to report", "Source"],
        [
            ["Runtime platform", "Remote RTX 4090 CUDA server", "run notes and environment record"],
            ["Python environment", "ssh_lane for formal runs; lsa_yolo for local validation", "project environment contract"],
            ["Input resolution", "544 x 960 in H,W order", "active TuSimple K56 contract"],
            ["Pretrained weights", "[PRETRAINED_SOURCE]", "frozen args.yaml"],
            ["Epochs and batch", "[EPOCHS], [BATCH]", "frozen args.yaml"],
            ["Optimizer and schedule", "[OPTIMIZER], [LR0], [LRF], [SCHEDULE]", "frozen args.yaml"],
            ["Augmentation", "[AUGMENTATION_POLICY]", "frozen args.yaml"],
            ["AMP and precision", "[AMP_STATE], [EVAL_PRECISION]", "run args and evaluation command"],
            ["Decode selection", "[CONF, PVT, NMS, MAX_DET, MIN_POINTS]", "official-validation sweep"],
        ],
    ),
    "results": (
        "Table 4. Placeholder template for protocol-valid TuSimple results.",
        ["Split", "Checkpoint", "conf", "pvt", "NMS", "max_det", "min_pts", "Acc", "FP", "FN"],
        [
            ["Official-val", "[SELECTED_CKPT]", "[VAL_CONF]", "[VAL_PVT]", "[VAL_NMS]", "[VAL_MAX_DET]", "[VAL_MIN_PTS]", "[VAL_ACC]", "[VAL_FP]", "[VAL_FN]"],
            ["Official-test", "same as val-selected", "same", "same", "same", "same", "same", "[TEST_ACC]", "[TEST_FP]", "[TEST_FN]"],
        ],
    ),
    "comparison": (
        "Table 5. Placeholder comparison with representative TuSimple methods.",
        ["Family", "Method", "Representation", "Acc", "FP", "FN", "Runtime", "Status"],
        [
            ["Segmentation", "SCNN / LaneNet / LaneAF", "dense mask or instance map", "[TBD]", "[TBD]", "[TBD]", "[TBD]", "verify citation and protocol"],
            ["Row/anchor", "UFLD / UFLDv2 / LaneATT / CLRNet", "row or line anchors", "[TBD]", "[TBD]", "[TBD]", "[TBD]", "fill from verified sources"],
            ["Curve/query", "LSTR / PolyLaneNet / BezierLaneNet", "curve or query output", "[TBD]", "[TBD]", "[TBD]", "[TBD]", "fill from verified sources"],
            ["YOLO-style", "YOLOP / YOLOPv2 / recent YOLO lane models", "multi-task or segmentation branch", "[TBD]", "[TBD]", "[TBD]", "[TBD]", "only if metrics are comparable"],
            ["Ours", "GCS-YOLO-Lane", "query-owned fixed-y point sequence", "[TBD]", "[TBD]", "[TBD]", "[TBD]", "pending active-default evidence"],
        ],
    ),
    "ablation": (
        "Table 6. Required ablation study plan.",
        ["Variant", "Changed component", "Acc", "FP", "FN", "Expected diagnosis"],
        [
            ["Full GCS-YOLO-Lane", "none", "[TBD]", "[TBD]", "[TBD]", "reference row"],
            ["w/o LSEM", "remove line-sensitive enhancement", "[TBD]", "[TBD]", "[TBD]", "tests long thin feature modeling"],
            ["w/o LaneBiFPN", "replace bidirectional fusion with simple fusion", "[TBD]", "[TBD]", "[TBD]", "tests multi-scale aggregation"],
            ["w/o fixed-y refinement", "disable point-location feature sampling", "[TBD]", "[TBD]", "[TBD]", "tests image-conditioned point refinement"],
            ["w/o point visibility", "remove point-valid branch", "[TBD]", "[TBD]", "[TBD]", "tests partial-lane modeling"],
            ["w/o mask/edge aux.", "remove dense auxiliary supervision", "[TBD]", "[TBD]", "[TBD]", "tests training-time dense cues"],
        ],
    ),
    "efficiency": (
        "Table 7. Placeholder efficiency report.",
        ["Model", "Params", "FLOPs/MACs", "preprocess", "network", "decode", "total", "Hardware"],
        [
            ["GCS-YOLO-Lane-S", "[PARAMS]", "[FLOPS]", "[PRE_MS]", "[NET_MS]", "[DEC_MS]", "[TOTAL_MS]", "RTX 4090, batch 1 or stated batch"],
            ["Ablated / baseline", "[PARAMS]", "[FLOPS]", "[PRE_MS]", "[NET_MS]", "[DEC_MS]", "[TOTAL_MS]", "same hardware only"],
        ],
    ),
    "failure": (
        "Table 8. Failure analysis checklist.",
        ["Error family", "Observable symptom", "Evidence to collect", "Selection rule"],
        [
            ["Count error", "predicted lane number differs from GT", "count confusion matrix on validation", "use official-val only"],
            ["Short-lane miss", "short visible side lane is dropped", "matched missing-lane traces", "do not tune on final test"],
            ["Geometry miss", "visible lane exists but x points drift", "average point error by anchor", "compare frozen variants"],
            ["Duplicate-like extra", "two queries explain one physical lane", "query distance and visibility overlap", "use validation diagnostics"],
            ["Spurious extra", "query predicts a lane with weak GT overlap", "low-overlap extra-lane report", "reject if official metrics regress"],
        ],
    ),
}


TABLES_ZH = {
    "notation": (
        "表 1. GCS-YOLO-Lane 使用的主要符号。",
        ["符号", "类型", "定义"],
        [
            ["B", "标量", "batch size"],
            ["Q", "标量", "lane query 数量，默认模型中固定为 12"],
            ["K", "标量", "fixed-y anchors 数量，TuSimple 契约中固定为 56"],
            ["I", "张量", "按 H,W 顺序 resize 到 544 x 960 的输入图像"],
            ["F_l", "张量", "第 l 个 pyramid level 的特征图"],
            ["z_q", "张量", "第 q 个 lane query 的特征向量"],
            ["P", "张量", "预测点张量，形状为 B x Q x K x 2"],
            ["s", "张量", "lane existence logits，形状为 B x Q"],
            ["v", "张量", "point-valid logits，形状为 B x Q x K"],
            ["A_k", "标量", "按 TuSimple 原始高度归一化后的第 k 个 fixed-y anchor"],
        ],
    ),
    "contract": (
        "表 2. 本文采用的当前 TuSimple K56 契约。",
        ["项目", "取值"],
        [
            ["模型配置", "ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml"],
            ["数据配置", "data/tusimple_gcs_fixed_y_960x544.yaml"],
            ["图像尺寸", "--imgsz 544 960，解释顺序为 H,W"],
            ["数据划分", "train 3263, validation 363, test 2782"],
            ["Lane queries", "Q = 12"],
            ["Fixed-y anchors", "K = 56, y = 710, 700, ..., 160"],
            ["最终车道输出", "pred_points, pred_logits, pred_valid_logits"],
            ["辅助输出", "aux_mask_logits 和 aux_edge_logits，仅用于训练监督"],
        ],
    ),
    "environment": (
        "表 3. 实验实现记录模板。",
        ["设置", "需要报告的取值", "来源"],
        [
            ["运行平台", "远程 RTX 4090 CUDA 服务器", "run notes 与环境记录"],
            ["Python 环境", "formal runs 使用 ssh_lane；local validation 使用 lsa_yolo", "项目环境契约"],
            ["输入分辨率", "544 x 960，H,W 顺序", "当前 TuSimple K56 契约"],
            ["预训练权重", "[PRETRAINED_SOURCE]", "冻结 args.yaml"],
            ["Epochs 与 batch", "[EPOCHS], [BATCH]", "冻结 args.yaml"],
            ["优化器与 schedule", "[OPTIMIZER], [LR0], [LRF], [SCHEDULE]", "冻结 args.yaml"],
            ["数据增强", "[AUGMENTATION_POLICY]", "冻结 args.yaml"],
            ["AMP 与 precision", "[AMP_STATE], [EVAL_PRECISION]", "run args 与 evaluation command"],
            ["Decode selection", "[CONF, PVT, NMS, MAX_DET, MIN_POINTS]", "official-validation sweep"],
        ],
    ),
    "results": (
        "表 4. TuSimple protocol-valid 结果占位模板。",
        ["Split", "Checkpoint", "conf", "pvt", "NMS", "max_det", "min_pts", "Acc", "FP", "FN"],
        [
            ["Official-val", "[SELECTED_CKPT]", "[VAL_CONF]", "[VAL_PVT]", "[VAL_NMS]", "[VAL_MAX_DET]", "[VAL_MIN_PTS]", "[VAL_ACC]", "[VAL_FP]", "[VAL_FN]"],
            ["Official-test", "与 val-selected 完全一致", "same", "same", "same", "same", "same", "[TEST_ACC]", "[TEST_FP]", "[TEST_FN]"],
        ],
    ),
    "comparison": (
        "表 5. 与 TuSimple 代表性方法比较的占位表。",
        ["类别", "方法", "表示方式", "Acc", "FP", "FN", "Runtime", "状态"],
        [
            ["Segmentation", "SCNN / LaneNet / LaneAF", "dense mask 或 instance map", "[TBD]", "[TBD]", "[TBD]", "[TBD]", "核对引用和协议"],
            ["Row/anchor", "UFLD / UFLDv2 / LaneATT / CLRNet", "row 或 line anchors", "[TBD]", "[TBD]", "[TBD]", "[TBD]", "从可靠来源填入"],
            ["Curve/query", "LSTR / PolyLaneNet / BezierLaneNet", "curve 或 query output", "[TBD]", "[TBD]", "[TBD]", "[TBD]", "从可靠来源填入"],
            ["YOLO-style", "YOLOP / YOLOPv2 / recent YOLO lane models", "multi-task 或 segmentation branch", "[TBD]", "[TBD]", "[TBD]", "[TBD]", "仅在指标可比时填写"],
            ["Ours", "GCS-YOLO-Lane", "query-owned fixed-y point sequence", "[TBD]", "[TBD]", "[TBD]", "[TBD]", "等待 active-default evidence"],
        ],
    ),
    "ablation": (
        "表 6. 必要消融实验计划。",
        ["Variant", "改变的组件", "Acc", "FP", "FN", "诊断目的"],
        [
            ["Full GCS-YOLO-Lane", "无", "[TBD]", "[TBD]", "[TBD]", "reference row"],
            ["w/o LSEM", "移除 line-sensitive enhancement", "[TBD]", "[TBD]", "[TBD]", "检验细长线特征建模"],
            ["w/o LaneBiFPN", "用简单融合替代双向融合", "[TBD]", "[TBD]", "[TBD]", "检验多尺度聚合"],
            ["w/o fixed-y refinement", "关闭点位置特征采样", "[TBD]", "[TBD]", "[TBD]", "检验 image-conditioned point refinement"],
            ["w/o point visibility", "移除 point-valid branch", "[TBD]", "[TBD]", "[TBD]", "检验部分车道建模"],
            ["w/o mask/edge aux.", "移除 dense auxiliary supervision", "[TBD]", "[TBD]", "[TBD]", "检验训练期 dense cues"],
        ],
    ),
    "efficiency": (
        "表 7. 效率报告占位表。",
        ["Model", "Params", "FLOPs/MACs", "preprocess", "network", "decode", "total", "Hardware"],
        [
            ["GCS-YOLO-Lane-S", "[PARAMS]", "[FLOPS]", "[PRE_MS]", "[NET_MS]", "[DEC_MS]", "[TOTAL_MS]", "RTX 4090, batch 1 或声明 batch"],
            ["Ablated / baseline", "[PARAMS]", "[FLOPS]", "[PRE_MS]", "[NET_MS]", "[DEC_MS]", "[TOTAL_MS]", "必须使用同一硬件"],
        ],
    ),
    "failure": (
        "表 8. 失败分析检查表。",
        ["错误类别", "可观察现象", "需要收集的证据", "选择规则"],
        [
            ["Count error", "预测车道数与 GT 不一致", "validation count confusion matrix", "只用 official-val"],
            ["Short-lane miss", "短可见侧车道被漏检", "matched missing-lane traces", "不得在 final test 调参"],
            ["Geometry miss", "可见车道存在但 x 点漂移", "按 anchor 的 average point error", "比较冻结 variants"],
            ["Duplicate-like extra", "两个 query 解释同一物理车道", "query distance 和 visibility overlap", "使用 validation diagnostics"],
            ["Spurious extra", "低 GT overlap 的额外 query", "low-overlap extra-lane report", "official metrics 回退则拒绝"],
        ],
    ),
}


TABLES_EN["reporting"] = (
    "Table 9. Reproducibility checklist before replacing placeholders.",
    ["Item", "Required artifact", "Acceptance rule"],
    [
        ["Training record", "frozen args.yaml, commit hash, dataset path, pretrained source", "all settings match the reported model"],
        ["Validation sweep", "official-val sweep summary with selected row", "checkpoint and decode selected only on official-val"],
        ["Final-test report", "single official-test summary using selected decode", "no test-time parameter search"],
        ["Baseline extraction", "paper/source, metric definition, input resolution, runtime hardware", "only comparable TuSimple metrics enter the main table"],
        ["Ablation record", "variant args.yaml and validation metrics", "one changed component per row whenever possible"],
        ["Efficiency profile", "params, FLOPs/MACs, preprocess, network, decode, total time", "same hardware and precision for comparable rows"],
        ["Failure analysis", "validation diagnostic traces and count confusion", "used for interpretation, not final-test selection"],
    ],
)

TABLES_EN["result_card"] = (
    "Table 10. Result-card template for the selected active-default candidate.",
    ["Field", "Value to insert", "Notes"],
    [
        ["Run name", "[RUN_NAME]", "must identify the frozen active-default run"],
        ["Checkpoint", "[BEST_OR_SELECTED_PT]", "selected on official validation"],
        ["Decode", "[CONF, PVT, NMS, MAX_DET, MIN_POINTS]", "copied unchanged to final test"],
        ["Official-val metrics", "[VAL_ACC, VAL_FP, VAL_FN]", "primary selection surface"],
        ["Official-test metrics", "[TEST_ACC, TEST_FP, TEST_FN]", "reported once after selection"],
        ["Count metrics", "[COUNT_ACC, PER_COUNT_ACC]", "secondary diagnostic, not official ranking"],
        ["Timing", "[PRE, NET, DEC, TOTAL_MS]", "state hardware, precision, batch size"],
    ],
)

TABLES_ZH["reporting"] = (
    "表 9. 替换占位符前的复现检查清单。",
    ["项目", "必需 artifact", "接受规则"],
    [
        ["训练记录", "冻结 args.yaml、commit hash、dataset path、pretrained source", "所有设置必须与报告模型一致"],
        ["Validation sweep", "包含 selected row 的 official-val sweep summary", "checkpoint 与 decode 只能由 official-val 选择"],
        ["Final-test report", "使用 selected decode 的单次 official-test summary", "不得在 test 上搜索参数"],
        ["Baseline extraction", "论文/来源、metric definition、input resolution、runtime hardware", "只有可比 TuSimple metrics 进入主表"],
        ["Ablation record", "variant args.yaml 与 validation metrics", "尽量每行只改变一个组件"],
        ["Efficiency profile", "params、FLOPs/MACs、preprocess、network、decode、total time", "可比行必须使用同一硬件与 precision"],
        ["Failure analysis", "validation diagnostic traces 与 count confusion", "用于解释，不用于 final-test selection"],
    ],
)

TABLES_ZH["result_card"] = (
    "表 10. Active-default selected candidate 结果卡模板。",
    ["字段", "待填入内容", "说明"],
    [
        ["Run name", "[RUN_NAME]", "必须标识冻结 active-default run"],
        ["Checkpoint", "[BEST_OR_SELECTED_PT]", "由 official validation 选择"],
        ["Decode", "[CONF, PVT, NMS, MAX_DET, MIN_POINTS]", "原样复制到 final test"],
        ["Official-val metrics", "[VAL_ACC, VAL_FP, VAL_FN]", "主要选择表面"],
        ["Official-test metrics", "[TEST_ACC, TEST_FP, TEST_FN]", "选择后只报告一次"],
        ["Count metrics", "[COUNT_ACC, PER_COUNT_ACC]", "辅助诊断，不作为 official ranking"],
        ["Timing", "[PRE, NET, DEC, TOTAL_MS]", "声明 hardware、precision、batch size"],
    ],
)


FORMULAS_EN = {
    "anchors": r"A_k = (710 - 10k) / 720,  k = 0, 1, ..., 55.",
    "outputs": r"P in R^{B x Q x K x 2},  s in R^{B x Q},  v in R^{B x Q x K}.",
    "matching": r"C(q,g) = lambda_p C_point(q,g) + lambda_c C_curve(q,g) + lambda_e C_exist(q,g).",
    "loss": r"L = L_exist + L_point + L_valid + lambda_s L_smooth + lambda_c L_curve + lambda_m L_mask + lambda_e L_edge.",
    "accuracy": r"Accuracy = sum_clip C_clip / sum_clip S_clip.",
    "f1": r"F1 = 2 x Precision x Recall / (Precision + Recall).",
}


FORMULAS_ZH = {
    "anchors": r"A_k = (710 - 10k) / 720,  k = 0, 1, ..., 55.",
    "outputs": r"P in R^{B x Q x K x 2},  s in R^{B x Q},  v in R^{B x Q x K}.",
    "matching": r"C(q,g) = lambda_p C_point(q,g) + lambda_c C_curve(q,g) + lambda_e C_exist(q,g).",
    "loss": r"L = L_exist + L_point + L_valid + lambda_s L_smooth + lambda_c L_curve + lambda_m L_mask + lambda_e L_edge.",
    "accuracy": r"Accuracy = sum_clip C_clip / sum_clip S_clip.",
    "f1": r"F1 = 2 x Precision x Recall / (Precision + Recall).",
}


EN_SECTIONS = [
    (
        "1 Introduction",
        [
            p("""Lane detection is a structured perception problem rather than a conventional object localization problem. A useful lane detector must recover thin and elongated lane markings, separate different lane instances, preserve the order of sampled points, and remain reliable when parts of a lane are occluded, worn, shadowed, or outside the visible field of view. These requirements make the output different from a bounding box and also different from an ungrouped foreground mask."""),
            p("""Early deep lane detectors often formulated the task as segmentation or instance segmentation, which is natural because lane markings occupy sparse pixels in the image [@pan2018scnn; @neven2018lanenet; @abualsaud2021laneaf]. However, dense masks are not the final object required by many lane benchmarks and downstream modules. They must be grouped, fitted, ordered, and converted into sampled coordinates. This conversion can be reliable in clean scenes, but it becomes a source of errors when markings are discontinuous, lane counts change, or several lane boundaries are close to one another."""),
            p("""Another influential line of work represents lanes by sparse coordinates on predefined rows or anchors. UFLD and UFLDv2 show that a lane can be detected efficiently by predicting positions on row or hybrid anchors instead of segmenting all pixels [@qin2020ufld; @qin2022ufldv2]. LaneATT and CLRNet further demonstrate that anchor priors, attention, and cross-level refinement are effective ways to exploit global context and local detail [@tabelini2021laneatt; @zheng2022clrnet]. These works motivate a key observation of this paper: lane detection benefits when the model output is already close to a lane object."""),
            p("""This paper proposes GCS-YOLO-Lane, a YOLO11-based structured lane detection network. The goal is to keep the efficient and deployment-friendly spirit of YOLO-style feature extraction while replacing ordinary box or final-mask outputs with query-owned lane instances. Each lane query predicts a lane existence score, a sequence of x coordinates on shared fixed-y anchors, and point-level visibility. The representation is therefore explicit: one query corresponds to one lane hypothesis, and the geometry of the lane is represented by an ordered point sequence rather than by a post-hoc mask-to-curve conversion."""),
            fig("overview"),
            p("""The current implementation is designed around the active TuSimple K56 contract of this project. The input image size is fixed to 544 x 960 in H,W order. The model predicts Q = 12 lane queries, and each query owns K = 56 anchors corresponding to original TuSimple y coordinates 710, 700, ..., 160. This fixed-y design is not a cosmetic detail. It aligns the network output with the horizontal samples used by the benchmark, and it gives each point a visibility logit so that the model can distinguish non-visible anchors from visible geometry."""),
            p("""The first contribution is a structured fixed-y representation for YOLO-style lane detection. Unlike a segmentation branch, the final output of GCS-YOLO-Lane is a set of lane hypotheses. Unlike a pure row classifier, each lane hypothesis is owned by a learned query and includes lane-level existence and point-level visibility. This makes lane count, point geometry, and partial visibility explicit training and inference targets."""),
            p("""The second contribution is an architecture that adapts a YOLO11-style backbone to structured lane output. The network uses line-sensitive feature enhancement modules to emphasize long thin structures, a LaneBiFPN to fuse P2-P5 multi-scale features, and a query-based GCS lane head to decode lane instances from spatial tokens. Auxiliary mask and edge branches are retained only as training supervision, not as final outputs."""),
            p("""The third contribution is a training and evaluation protocol that keeps method claims separated from pending empirical evidence. Training uses Hungarian assignment between queries and ground-truth lanes, point geometry and visibility losses, curve and smoothness regularization, and auxiliary dense supervision. Evaluation must use official validation for checkpoint and post-processing selection, while final test is reserved for one-shot reporting. Because the user request requires experimental results to remain placeholders, this manuscript deliberately avoids state-of-the-art or superiority claims until protocol-valid results are inserted."""),
        ],
    ),
    (
        "2 Related Work",
        [
            p("""Segmentation-based lane detection models learn dense foreground evidence and then recover lane instances from pixel-level predictions. SCNN introduced spatial message passing to propagate information along rows and columns, showing that lane detection requires spatial context beyond local convolution [@pan2018scnn]. LaneNet treated lane detection as an instance segmentation problem and separated binary lane segmentation from instance embedding [@neven2018lanenet]. LaneAF introduced affinity fields to associate lane pixels more robustly [@abualsaud2021laneaf]. These methods are important because they provide strong supervision for thin lane structures, but their final output still needs grouping, association, fitting, or coordinate conversion before benchmark evaluation."""),
            p("""Row-based and anchor-based methods reduce the dense prediction burden by directly predicting sparse lane locations. UFLD formulates lane detection as row-wise classification, which gives high efficiency by using global features and discrete row locations [@qin2020ufld]. UFLDv2 extends this direction with hybrid row and column anchors and ordinal classification, addressing the localization issues that arise when one anchor family does not fit all lane types [@qin2022ufldv2]. LaneATT uses lane anchors with attention-guided feature aggregation [@tabelini2021laneatt], and CLRNet refines lane priors across feature levels [@zheng2022clrnet]. GCS-YOLO-Lane follows the sparse-coordinate motivation but organizes each coordinate sequence through learned lane queries."""),
            p("""Curve-based, sequence-based, and query-based lane detectors aim to predict a more compact lane object than a dense mask. PolyLaneNet predicts polynomial coefficients for lane curves [@tabelini2020polylanenet], BezierLaneNet rethinks lane detection through Bezier curve modeling [@feng2022bezierlane], and FastDraw uses sequential prediction to handle long-tail lane cases [@philion2019fastdraw]. LSTR predicts lane shape through a transformer-based set formulation [@liu2021lstr]. These methods demonstrate that lane detection can be posed as structured output prediction. The distinction of GCS-YOLO-Lane is that it keeps a direct fixed-y point sequence with explicit point visibility rather than compressing the whole lane into a global curve parameter."""),
            p("""YOLO-style driving perception models show that shared detection backbones can support efficient multi-task perception. YOLOP and YOLOPv2 use YOLO-style designs for object detection, drivable-area segmentation, and lane-related perception [@wu2022yolop; @han2022yolopv2]. Recent YOLO-based lane papers further explore attention and segmentation heads for real-time lane or intrusion detection. GCS-YOLO-Lane is related to this family in backbone philosophy, but the final task head is different: the network is not only a segmentation add-on to a YOLO detector, but a structured lane instance predictor."""),
            p("""Dataset scope also matters. TuSimple evaluates x coordinates at fixed horizontal samples and is therefore naturally compatible with fixed-y outputs [@tusimpleBenchmark]. CULane, CurveLanes, ONCE-3DLanes, and OpenLane-V2 cover broader scenarios, stronger curvature, 3D geometry, or topology reasoning [@pan2018scnn; @xu2020curvelanes; @yan2022once3d; @wang2023openlanev2]. The present manuscript focuses on the active 2D TuSimple K56 branch. Cross-dataset and 3D generalization are important future work rather than evidence already provided by this draft."""),
        ],
    ),
    (
        "3 GCS-YOLO-Lane",
        [
            p("""This section describes the method in the same order as the forward and training pipeline. Section 3.1 defines the output contract and notation. Section 3.2 introduces the fixed-y K56 label representation. Section 3.3 gives the overall architecture. Sections 3.4 and 3.5 describe the line-sensitive and multi-scale feature modules. Section 3.6 presents the query-based lane head, and Sections 3.7 and 3.8 describe training and inference."""),
            table("notation"),
            p("""Problem formulation. Given an RGB image I resized to 544 x 960 in H,W order, the detector predicts a set of lane instances. The number of candidate instances is fixed to Q = 12. Each candidate owns K = 56 ordered point slots, a lane-existence logit, and K point-valid logits. The model emits the following tensors:"""),
            formula(FORMULAS_EN["outputs"]),
            p("""The first coordinate of each point is the predicted x location. In fixed-y mode, the second coordinate is not freely regressed; it is restored from a shared anchor buffer. This keeps the tensor interface two-dimensional while constraining the representation to the benchmark-aligned fixed-y grid."""),
            p("""Fixed-y label representation. The active TuSimple representation uses the 56 vertical anchors 710, 700, 690, ..., 160 in original image coordinates. After normalization by the original TuSimple height 720, the k-th anchor is defined as:"""),
            formula(FORMULAS_EN["anchors"]),
            fig("fixed_y"),
            p("""Labels are regenerated from original TuSimple JSON and images rather than resampled from historical K32 labels. For each annotated lane, invalid points are removed, points are ordered from bottom to top, duplicate y positions are discarded, and x is interpolated at each fixed-y anchor. Anchors inside the visible lane span receive normalized x and lane_valid = 1. Anchors outside the visible span keep the fixed y coordinate but set x = 0 and lane_valid = 0. This design separates the existence of a point slot from the visibility of lane evidence at that slot."""),
            p("""The fixed-y representation is chosen for protocol alignment rather than for convenience alone. TuSimple evaluates lanes on a fixed set of horizontal samples, so a detector that already predicts on this grid avoids an additional curve-to-sample conversion step. This does not mean that the method assumes every road marking is globally straight. The x coordinate remains image-conditioned at every anchor, and the visibility branch allows the visible support of a lane to be shorter than the full anchor set. The representation is therefore a sampled structured object: it keeps the benchmark-aligned y locations deterministic while leaving lateral geometry and visibility to the network."""),
            p("""The query capacity Q is deliberately larger than the usual visible lane count in TuSimple. This over-complete set lets the model represent uncertain or partially visible lanes without hard-coding the number of outputs for each image. The matching stage assigns only a subset of queries to ground-truth lanes, and the remaining queries are trained as non-existing lanes through the existence target. In inference, the score threshold, point-valid filtering, minimum visible-point rule, and optional NMS determine which query hypotheses survive. This design makes lane-count errors inspectable at the query level, which is useful for the failure analysis requested later in the experiment section."""),
            p("""Overall architecture. GCS-YOLO-Lane starts from a YOLO11-style backbone and extracts pyramid features P2, P3, P4, and P5. LSEM is inserted after the P3 and P4 stages to strengthen line-like structures. LaneBiFPN then aligns channels and performs bidirectional weighted fusion across P2-P5. The fused features are flattened into spatial tokens with positional and level embeddings, and a transformer decoder lets learned lane queries attend to the tokens. The GCS lane head predicts lane existence, fixed-y point coordinates, and point visibility. Auxiliary mask and edge outputs are attached to the shared feature path for training supervision."""),
            table("contract"),
            p("""Line-sensitive feature enhancement. Lane markings are long, thin, and directionally coherent, so a standard local convolution may underuse the shape prior of lane evidence. LSEM addresses this by combining horizontal and vertical strip depthwise convolutions, a direction gate, coordinate-aware reweighting, and dilated context. The strip convolutions capture elongated responses, the direction gate lets the module choose between horizontal and vertical evidence, and coordinate reweighting preserves spatial information along height and width. A residual connection keeps the enhanced feature compatible with the backbone stream."""),
            p("""LaneBiFPN multi-scale fusion. Fine spatial features are useful for point localization, while higher-level features help separate lane markings from road texture, shadows, guardrails, and neighboring lanes. LaneBiFPN receives P2, enhanced P3, enhanced P4, and P5 features, projects them into a common channel dimension, and fuses them through top-down and bottom-up weighted paths. This module is motivated by the same practical issue emphasized by refinement-based lane detectors: lane localization needs both boundary-level detail and semantic context."""),
            p("""Query-based GCS lane head. The lane head uses Q learnable lane queries and a transformer decoder with three decoder layers and eight attention heads in the default configuration. Each query attends to multi-scale spatial tokens and produces one lane hypothesis. In fixed-y mode, the head predicts x logits for the 56 anchors. Query-specific reference logits provide different initial geometric patterns, and image-conditioned refinement samples features around predicted point positions to update x logits and point-valid logits."""),
            fig("head"),
            p("""The head also predicts lane existence. A high existence score means that the query is likely to correspond to one physical lane, while point-valid logits decide which anchors of that lane are visible. This separation is important because a lane can be real but only partially visible. It is also important for suppressing duplicate or spurious queries: an unmatched query should be rejected at the lane level rather than decoded into a short false lane."""),
            p("""Matching and loss. During training, predictions are matched to ground-truth lanes by Hungarian assignment. The matching cost combines visible-point distance, curve consistency, and existence confidence:"""),
            formula(FORMULAS_EN["matching"]),
            p("""After assignment, the training objective supervises lane existence, point geometry on visible anchors, point visibility, smoothness, curve regularity, auxiliary mask prediction, and auxiliary edge prediction:"""),
            formula(FORMULAS_EN["loss"]),
            p("""The implementation also logs several default-disabled experimental items, including lane-balanced point loss, short-valid recall loss, count loss, under-5 count loss, duplicate-margin loss, and spurious-margin loss. These terms are not part of the main method claim unless they are explicitly enabled in a frozen experiment and reported as such. This distinction matters because the current branch is the 5-25-3 K56 mainline, not the later Count Head or Quality Head branch."""),
            p("""Auxiliary dense supervision is used as a training aid, not as an alternative prediction head at evaluation time. The mask branch encourages the shared features to retain lane-region evidence, and the edge branch encourages sensitivity to thin boundary-like structures. Because the final decoded lane is still produced from query-owned points and point-valid logits, these auxiliary losses should be interpreted as representation support rather than as a segmentation-based fallback. Any ablation that removes mask or edge supervision should therefore be evaluated with the same structured decoder and official-validation selection rule."""),
            p("""Network inference. At inference time, GCS-YOLO-Lane decodes real query predictions only. The decoder applies a query score threshold, a point-valid threshold, longest contiguous visible span selection, a minimum visible-point constraint, optional lane NMS based on shared-anchor x distances, and a maximum lane count. The resulting visible points are converted back to TuSimple h-samples by interpolation. The decoder must not use ground truth, fabricate lanes, or select thresholds on final test."""),
            fig("protocol"),
        ],
    ),
    (
        "4 Experiments",
        [
            p("""This section follows the organization of standard lane-detection papers: experimental setting, metrics, implementation details, main results, ablations, efficiency, and failure analysis. The numerical entries are intentionally placeholders because the current request asks for results to be filled later. The section is therefore a paper-ready experiment scaffold rather than an empirical claim."""),
            p("""Dataset. The active dataset is the TuSimple fixed-y K56 conversion with 3263 training images, 363 official-validation images, and 2782 test images. The conversion uses original TuSimple JSON and images and outputs labels with semantic masks, edge masks, fixed-y lane coordinates, point visibility, original file metadata, and image shape. The paper should not mix this dataset with historical K32 labels."""),
            p("""Evaluation metrics. The primary TuSimple metrics are Accuracy, FP, and FN. Accuracy is computed over sampled lane points as:"""),
            formula(FORMULAS_EN["accuracy"]),
            p("""For comparison tables that include methods or datasets using detection-style metrics, precision, recall, and F1 should be reported only when the protocol is clear. F1 is defined as:"""),
            formula(FORMULAS_EN["f1"]),
            p("""Implementation details. Formal training should be performed on the remote RTX 4090 environment, using the ssh_lane conda environment and batch 32 as the starting point for current Q12/K56 runs. The default model and data files are ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml and data/tusimple_gcs_fixed_y_960x544.yaml. The final paper must copy optimizer, schedule, AMP state, workers, augmentation, and epoch settings from the frozen active-default args.yaml rather than from a template."""),
            table("environment"),
            p("""Experimental implementation record. Table 3 follows the reporting style of applied YOLO lane papers: it separates the runtime environment, input contract, training recipe, precision mode, and decode-selection record. This table is intentionally a template because the final values must come from the frozen run artifacts. It should not be filled from memory, from a generic command example, or from a run that used a different K56 fixed-y contract."""),
            table("results"),
            p("""Main results. Table 4 is the only place where official-val selection and one-shot final-test numbers should be reported. The official-validation row must contain the selected checkpoint and decode parameters. The final-test row must use exactly the same checkpoint and decode parameters. If multiple final-test runs or test-tuned thresholds are used, the table is invalid for the main paper."""),
            table("comparison"),
            p("""Baseline comparison. Table 5 follows the comparison style of lane-detection papers such as UFLDv2: methods are grouped by representation family, and all metrics must be aligned to the same benchmark protocol. A fair table should distinguish original-paper metrics, reproduced metrics, and project-specific reruns. Runtime must state hardware and precision. Metrics from CULane, CurveLanes, LLAMAS, 3D lane benchmarks, or custom lane-intrusion datasets should not be mixed with TuSimple official metrics in the same numerical block."""),
            table("ablation"),
            p("""Ablation study. Table 6 defines the minimum evidence needed to support the method design. Removing LSEM tests whether line-sensitive feature processing contributes to lane evidence. Replacing LaneBiFPN tests the benefit of bidirectional multi-scale fusion. Disabling fixed-y refinement tests whether point-location feature sampling improves geometry. Removing point visibility tests whether explicit partial-lane modeling matters. Removing auxiliary mask/edge supervision tests whether dense training cues help a structured final output."""),
            p("""Efficiency. Since the proposed method uses a structured decoder after a YOLO-style backbone, efficiency must be decomposed into preprocessing, network forward, decoding, and total latency. A single FPS value is not sufficient because decoding and TuSimple conversion can contribute non-negligible time. Table 7 should be filled with model parameters, FLOPs or MACs, timing, precision mode, batch size, and hardware."""),
            table("efficiency"),
            p("""Failure analysis. Structured output makes failure modes more observable. In a mask pipeline, lane count errors can be hidden inside grouping or fitting. In GCS-YOLO-Lane, errors can be assigned to query existence, point visibility, geometry, duplicate-like queries, or spurious extra lanes. Table 8 and Figure 5 define the diagnostic categories that should be reported on validation data before final-test reporting."""),
            fig("failure"),
            table("failure"),
        ],
    ),
    (
        "5 Discussion",
        [
            p("""The main methodological change in GCS-YOLO-Lane is the output object. The network does not predict a box, and it does not depend on a final foreground mask as the lane instance. It predicts a set of lane hypotheses, each of which owns a lane score, a fixed-y point sequence, and point-level visibility. This makes the representation closer to the object that TuSimple-style evaluation consumes."""),
            p("""Compared with segmentation-based pipelines, the proposed representation reduces the burden of post-hoc grouping. Dense supervision remains useful for learning lane-sensitive features, but dense masks are no longer the final representation. This distinction should be clear to reviewers because auxiliary mask and edge branches may otherwise make the method look like another segmentation model."""),
            p("""Compared with row-anchor methods, the fixed-y anchors in GCS-YOLO-Lane are shared by all queries rather than tied to a fixed lane slot. Learned queries own lane instances and must learn to allocate themselves to scenes with different lane counts. This design can expose count errors more directly, but it also creates a calibration problem: queries must suppress extra lanes without missing short or partially visible true lanes."""),
            p("""Compared with curve-parameter methods, the fixed-y point sequence is less compact but more transparent. Each anchor has an x coordinate and a visibility logit, so local errors can be analyzed by anchor position. This is useful for diagnosing lower-lane drift, upper-lane invisibility, and short side-lane misses. The cost is that the representation is tied to the fixed-y sampling scheme and may need adaptation for datasets with different geometry."""),
            p("""The current evidence boundary is strict. The architecture and training design are described because they are supported by the project code and active contracts. Performance superiority, ablation causality, and final ranking are not claimed because the requested manuscript keeps experiments as placeholders. This conservative writing is not a weakness of the method description; it is the required condition for a paper that does not tune on final test or import legacy runs as active-default evidence."""),
        ],
    ),
    (
        "6 Reproducibility and Reporting Checklist",
        [
            p("""A complete lane-detection paper must make its empirical claims reproducible. In this manuscript, reproducibility is especially important because the method is tied to a precise fixed-y contract, and because small changes in threshold, point-validity filtering, NMS distance, maximum detections, and minimum visible points can change FP and FN. The final manuscript should therefore treat experiment reporting as a first-class part of the method rather than as a short afterthought."""),
            p("""The first record to freeze is the training record. A valid result must identify the exact model configuration, data configuration, dataset root, image size, pretrained source, optimizer, learning-rate schedule, batch size, AMP setting, number of workers, augmentation policy, epoch count, and random seed if available. The most reliable source for these details is the final args.yaml from the selected run. Template commands are useful for explanation, but they cannot replace the actual frozen training record."""),
            p("""The second record to freeze is the official-validation selection record. For each candidate checkpoint, the sweep should report the tested confidence thresholds, point-validity thresholds, NMS distances, maximum detections, minimum visible-point values, and resulting official metrics. The selected row must be chosen before final test is evaluated. This mirrors the discipline used in strong lane-detection papers: the reported test number is meaningful only when the selection rule is fixed in advance."""),
            p("""The third record is the one-shot final-test package. The final-test command must copy the selected checkpoint and decode parameters from official validation without modification. The output directory should contain the official summary, prediction records if saved, the command used to generate them, and the script version. If a second final-test run is made because of a formatting or path error, the manuscript should explain it; if a second run is made to improve metrics, the result should not be used for the main claim."""),
            table("reporting"),
            p("""Baseline extraction needs the same care. A comparison table should identify whether each number comes from the original paper, a public leaderboard, an official implementation, or a project-local rerun. It should record the input resolution and runtime hardware whenever speed is compared. Strong baselines such as SCNN, UFLD, UFLDv2, LaneATT, CLRNet, LSTR, and curve-based methods should not be reduced to a citation list; they should be reported under a protocol that readers can audit."""),
            p("""Ablation records should be tied to method claims. If the manuscript claims that LSEM improves long thin feature modeling, there must be a variant without LSEM under the same training and selection protocol. If it claims that LaneBiFPN improves multi-scale fusion, there must be a simpler-fusion variant. If it claims that point visibility helps partial-lane reasoning, there must be a visibility ablation. Without these records, the method can still be described, but the empirical contribution should remain conditional."""),
            p("""Efficiency reporting should separate model design from implementation overhead. The structured lane head introduces decoding and TuSimple conversion steps that are not captured by network forward time alone. The paper should therefore report preprocessing time, network time, decode time, and total time. If batch inference, half precision, TensorRT, CPU decoding, or different image loading behavior is used, the table must say so explicitly."""),
            table("result_card"),
            p("""Failure analysis should be reported from validation diagnostics before final-test reporting. The useful categories are count errors, short-lane misses, geometry drift, duplicate-like extra queries, spurious extra queries, and visibility failures. These categories are directly linked to the structured representation. They should be used to explain where the model fails and to motivate future work, not to tune final-test behavior."""),
            p("""The final paper should pass a simple claim-evidence audit. Every result claim in the abstract and introduction must point to one of the frozen artifacts listed in Table 9 or Table 10. Every method claim must point to code, configuration, or a controlled ablation. Every comparison claim must point to a verified baseline source. This audit is what turns the present method-complete draft into a submission-ready empirical paper."""),
        ],
    ),
    (
        "7 Conclusion",
        [
            p("""This paper presents GCS-YOLO-Lane, a YOLO11-based structured lane detection network for TuSimple fixed-y lane prediction. The key idea is to replace ordinary YOLO box or final-mask outputs with query-owned lane instances. Each query predicts lane existence, x coordinates on 56 fixed-y anchors, and point-level visibility. The architecture combines line-sensitive feature enhancement, LaneBiFPN multi-scale fusion, a query-based GCS lane head, Hungarian matching, visibility-aware losses, and auxiliary dense supervision."""),
            p("""The current manuscript should be read as a method-complete and experiment-placeholder draft. It gives the representation, architecture, training objective, inference procedure, evaluation protocol, and all tables needed for final reporting. Once active-default official-validation selection, one-shot final-test evaluation, verified baselines, ablations, efficiency measurements, and failure analysis are complete, the placeholder tables can be filled and the abstract and conclusion can be updated with supported empirical claims."""),
        ],
    ),
]


ZH_SECTIONS = [
    (
        "1 引言",
        [
            p("""车道线检测是一个结构化感知问题，而不是普通目标定位问题。一个可用的车道检测器需要恢复细长的道路标线，区分不同车道实例，保持采样点顺序，并且在遮挡、磨损、阴影或视野外缺失时仍能给出可靠预测。这些要求使它的输出既不同于 bounding box，也不同于未分组的 foreground mask。"""),
            p("""早期深度车道检测方法常把任务建模为 segmentation 或 instance segmentation，因为车道标线在图像中确实是稀疏像素结构 [@pan2018scnn; @neven2018lanenet; @abualsaud2021laneaf]。然而，dense masks 并不是很多 benchmark 和下游模块真正需要的最终对象。它们还需要 grouping、association、fitting、ordering 和 coordinate conversion。场景干净时这些转换可以工作，但在车道断裂、车道数量变化或相邻车道边界很近时，转换层本身会引入错误。"""),
            p("""另一类重要方法直接在预定义 rows 或 anchors 上预测稀疏车道坐标。UFLD 和 UFLDv2 证明，使用 row-wise 或 hybrid-anchor classification 可以高效表示 lane location [@qin2020ufld; @qin2022ufldv2]。LaneATT 和 CLRNet 进一步说明，anchor priors、attention 和 cross-level refinement 可以同时利用全局上下文与局部细节 [@tabelini2021laneatt; @zheng2022clrnet]。这些工作给本文提供了动机：当模型输出本身已经接近 lane object 时，车道检测会更直接。"""),
            p("""本文提出 GCS-YOLO-Lane，一种基于 YOLO11 的结构化车道检测网络。目标是在保留 YOLO-style feature extraction 高效性的同时，把 ordinary box 或 final-mask output 替换为 query-owned lane instances。每个 lane query 预测一个 lane existence score、一组共享 fixed-y anchors 上的 x 坐标，以及 point-level visibility。因此，表示是显式的：一个 query 对应一个 lane hypothesis，车道几何由有序点序列表达，而不是由 mask 后处理拟合得到。"""),
            fig("overview"),
            p("""当前实现围绕本项目 active TuSimple K56 contract 设计。输入图像固定为 544 x 960，顺序为 H,W。模型预测 Q = 12 个 lane queries，每个 query 拥有 K = 56 个 anchors，对应 TuSimple 原始 y 坐标 710, 700, ..., 160。该 fixed-y 设计并非表面细节。它使网络输出与 benchmark 的水平采样位置对齐，并为每个点提供 visibility logit，从而区分不可见 anchor 和可见几何。"""),
            p("""第一项贡献是用于 YOLO-style lane detection 的结构化 fixed-y representation。不同于 segmentation branch，GCS-YOLO-Lane 的最终输出是 lane hypotheses 集合。不同于 pure row classifier，每个 lane hypothesis 由 learned query 拥有，并同时包含 lane-level existence 和 point-level visibility。因此，lane count、point geometry 和 partial visibility 都成为显式训练与推理目标。"""),
            p("""第二项贡献是将 YOLO11-style backbone 改造为结构化 lane output 的网络结构。模型使用 line-sensitive feature enhancement modules 强化长而细的线状结构，使用 LaneBiFPN 融合 P2-P5 multi-scale features，并使用 query-based GCS lane head 从 spatial tokens 解码 lane instances。Auxiliary mask 和 edge branches 只作为训练监督保留，不作为最终输出。"""),
            p("""第三项贡献是将方法主张和待完成经验证据严格分开的训练与评估协议。训练使用 Hungarian assignment 匹配 queries 与 ground-truth lanes，结合 point geometry、visibility、curve/smoothness regularization 和 auxiliary dense supervision。评估必须使用 official validation 选择 checkpoint 和 post-processing，final test 只用于一次性报告。由于当前请求要求实验结果保留占位符，本文不会在证据完成前声称 state-of-the-art 或优于某个具体 baseline。"""),
        ],
    ),
    (
        "2 相关工作",
        [
            p("""Segmentation-based lane detection models 学习 dense foreground evidence，然后从 pixel-level predictions 中恢复 lane instances。SCNN 通过空间消息传递传播行列方向信息，说明 lane detection 需要超出局部卷积的 spatial context [@pan2018scnn]。LaneNet 将 lane detection 作为 instance segmentation，并分离 binary lane segmentation 与 instance embedding [@neven2018lanenet]。LaneAF 使用 affinity fields 更稳健地关联 lane pixels [@abualsaud2021laneaf]。这些方法重要在于它们为细长车道结构提供强监督，但最终输出仍需 grouping、association、fitting 或 coordinate conversion。"""),
            p("""Row-based 和 anchor-based 方法通过直接预测 sparse lane locations 降低 dense prediction 负担。UFLD 将 lane detection 表述为 row-wise classification，通过 global features 和 discrete row locations 获得高效率 [@qin2020ufld]。UFLDv2 使用 hybrid row/column anchors 和 ordinal classification 处理单一 anchor family 不适配所有车道类型的问题 [@qin2022ufldv2]。LaneATT 使用 line anchors 与 attention-guided feature aggregation [@tabelini2021laneatt]，CLRNet 跨特征层 refine lane priors [@zheng2022clrnet]。GCS-YOLO-Lane 继承 sparse-coordinate 思路，但每条 coordinate sequence 由 learned lane query 组织。"""),
            p("""Curve-based、sequence-based 和 query-based 方法尝试预测比 dense mask 更紧凑的 lane object。PolyLaneNet 预测车道多项式系数 [@tabelini2020polylanenet]，BezierLaneNet 用 Bezier curve modeling 重新思考 lane detection [@feng2022bezierlane]，FastDraw 使用 sequential prediction 处理 long-tail lane cases [@philion2019fastdraw]，LSTR 则通过 transformer-based set formulation 预测 lane shape [@liu2021lstr]。这些方法说明 lane detection 可以作为 structured output prediction。GCS-YOLO-Lane 的区别在于保留直接 fixed-y point sequence，并显式预测 point visibility，而不是把整条车道压缩成 global curve coefficients。"""),
            p("""YOLO-style driving perception models 说明共享 detection backbone 可以支持高效 multi-task perception。YOLOP 和 YOLOPv2 使用 YOLO-style 设计处理 object detection、drivable-area segmentation 与 lane-related perception [@wu2022yolop; @han2022yolopv2]。近年的 YOLO-based lane papers 也进一步探索 attention 和 segmentation heads 用于 real-time lane 或 intrusion detection。GCS-YOLO-Lane 与这一类工作的联系在于 backbone philosophy，但最终 task head 不同：它不是 YOLO detector 的 segmentation add-on，而是 structured lane instance predictor。"""),
            p("""Dataset scope 同样重要。TuSimple 在固定 horizontal samples 上评估 x 坐标，因此天然适合 fixed-y outputs [@tusimpleBenchmark]。CULane、CurveLanes、ONCE-3DLanes 和 OpenLane-V2 覆盖更复杂场景、强曲率、3D geometry 或 topology reasoning [@pan2018scnn; @xu2020curvelanes; @yan2022once3d; @wang2023openlanev2]。本文聚焦当前 active 2D TuSimple K56 branch。跨数据集和 3D 泛化是未来工作，而不是本文已经完成的经验证据。"""),
        ],
    ),
    (
        "3 GCS-YOLO-Lane 方法",
        [
            p("""本节按 forward 与 training pipeline 的顺序描述方法。3.1 节定义 output contract 和符号，3.2 节介绍 fixed-y K56 标签表示，3.3 节给出整体网络结构，3.4 和 3.5 节分别描述线敏感特征增强与多尺度融合，3.6 节介绍 query-based lane head，3.7 和 3.8 节描述训练与推理。"""),
            table("notation"),
            p("""问题定义。给定一张按 H,W 顺序 resize 到 544 x 960 的 RGB 图像 I，检测器输出一组 lane instances。候选实例数量固定为 Q = 12。每个候选实例拥有 K = 56 个有序 point slots、一个 lane-existence logit 和 K 个 point-valid logits。模型输出张量为："""),
            formula(FORMULAS_ZH["outputs"]),
            p("""每个点的第一个坐标是预测 x 位置。在 fixed-y mode 中，第二个坐标不是自由回归值，而是由共享 anchor buffer 恢复。这样既保持二维点的 tensor interface，又将表示约束到与 benchmark 对齐的 fixed-y grid。"""),
            p("""Fixed-y 标签表示。当前 TuSimple 表示使用原始图像坐标中的 56 个纵向 anchors：710, 700, 690, ..., 160。按 TuSimple 原始高度 720 归一化后，第 k 个 anchor 定义为："""),
            formula(FORMULAS_ZH["anchors"]),
            fig("fixed_y"),
            p("""K56 标签必须从原始 TuSimple JSON 和 images 重新生成，不能从历史 K32 labels 重采样。对于每条标注 lane，转换流程会移除非法点，按 bottom-to-top 排序，删除重复 y，并在每个 fixed-y anchor 上插值得到 x。位于可见 lane span 内的 anchors 存储归一化 x 并设置 lane_valid = 1。不可见 anchors 保留 fixed y 坐标，但设置 x = 0 和 lane_valid = 0。该设计将 point slot 的存在与该位置是否有可见 lane evidence 分开。"""),
            p("""Fixed-y representation 的选择首先是为了协议对齐，而不只是实现方便。TuSimple 在固定 horizontal samples 上评估 lane x 坐标，因此模型直接在这一 grid 上输出可以减少额外的 curve-to-sample conversion。该设计并不假设所有道路标线都是全局直线，因为每个 anchor 的 x 坐标仍由图像条件决定，visibility branch 也允许一条 lane 的可见支撑短于完整 anchor set。因此，该表示是 sampled structured object：y 位置由 benchmark-aligned anchors 决定，而 lateral geometry 与可见性由网络预测。"""),
            p("""Query capacity Q 被故意设置得大于 TuSimple 中通常可见的车道数。这个 over-complete set 让模型可以表达不确定或局部可见的 lanes，而不必为每张图硬编码输出数量。匹配阶段只把一部分 queries 分配给 ground-truth lanes，其余 queries 通过 existence target 学习为 non-existing lanes。推理阶段再由 score threshold、point-valid filtering、minimum visible-point rule 和 optional NMS 决定哪些 query hypotheses 保留。这样 lane-count errors 可以在 query level 被检查，便于实验部分后续做 failure analysis。"""),
            p("""整体结构。GCS-YOLO-Lane 从 YOLO11-style backbone 开始，提取 P2、P3、P4 和 P5 pyramid features。LSEM 插入在 P3 和 P4 之后以强化线状结构。LaneBiFPN 随后对 P2-P5 做通道对齐与双向加权融合。融合后的 features 被展开为带 positional 和 level embeddings 的 spatial tokens，transformer decoder 使 learned lane queries attend 到这些 tokens。GCS lane head 预测 lane existence、fixed-y point coordinates 和 point visibility。Auxiliary mask 与 edge outputs 连接到共享 feature path，只用于训练监督。"""),
            table("contract"),
            p("""线敏感特征增强。车道标线长、细且具有方向一致性，因此普通局部卷积可能不能充分利用 lane shape prior。LSEM 结合 horizontal/vertical strip depthwise convolutions、direction gate、coordinate-aware reweighting 和 dilated context。Strip convolutions 捕获拉长响应，direction gate 在水平与垂直证据之间选择，coordinate reweighting 保留 height/width 空间信息。Residual connection 保证增强后的 feature 与 backbone stream 兼容。"""),
            p("""LaneBiFPN 多尺度融合。细粒度空间特征有利于 point localization，高层语义特征有利于把 lane markings 与 road texture、shadows、guardrails 和 neighboring lanes 区分开。LaneBiFPN 接收 P2、enhanced P3、enhanced P4 和 P5 features，将其投影到统一 channel dimension，并通过 top-down 与 bottom-up weighted paths 融合。该模块回应了 refinement-based lane detectors 强调的实际问题：lane localization 同时需要边界级细节与语义上下文。"""),
            p("""Query-based GCS lane head。Lane head 使用 Q 个可学习 lane queries 和一个默认配置为 3 decoder layers、8 attention heads 的 transformer decoder。每个 query attend 到 multi-scale spatial tokens 并输出一个 lane hypothesis。在 fixed-y mode 中，head 为 56 个 anchors 预测 x logits。Query-specific reference logits 提供不同初始几何形态，image-conditioned refinement 在预测点附近采样 feature 以更新 x logits 和 point-valid logits。"""),
            fig("head"),
            p("""Head 同时预测 lane existence。高 existence score 表示该 query 很可能对应一条物理车道，而 point-valid logits 决定该 lane 的哪些 anchors 可见。这种分离很重要，因为一条真实车道可能只有部分可见。它也有助于抑制 duplicate 或 spurious queries：unmatched query 应在 lane level 被拒绝，而不是被解码成短 false lane。"""),
            p("""匹配与 loss。训练时，predictions 通过 Hungarian assignment 匹配到 ground-truth lanes。Matching cost 结合 visible-point distance、curve consistency 和 existence confidence："""),
            formula(FORMULAS_ZH["matching"]),
            p("""完成 assignment 后，训练目标监督 lane existence、visible anchors 上的 point geometry、point visibility、smoothness、curve regularity、auxiliary mask prediction 和 auxiliary edge prediction："""),
            formula(FORMULAS_ZH["loss"]),
            p("""实现中还会记录若干默认关闭的实验项，包括 lane-balanced point loss、short-valid recall loss、count loss、under-5 count loss、duplicate-margin loss 和 spurious-margin loss。除非这些项在冻结实验中显式启用并按实验项报告，否则它们不是 main method claim。这个区别很重要，因为当前分支是 5-25-3 K56 mainline，而不是 later Count Head 或 Quality Head branch。"""),
            p("""Auxiliary dense supervision 是训练辅助，而不是 evaluation-time alternative prediction head。Mask branch 促使共享特征保留 lane-region evidence，edge branch 促使网络对细薄边界状结构保持敏感。由于最终 decoded lane 仍然来自 query-owned points 与 point-valid logits，这些辅助 losses 应解释为 representation support，而不是 segmentation-based fallback。任何移除 mask 或 edge supervision 的消融都应使用相同 structured decoder 和 official-validation selection rule 评估。"""),
            p("""网络推理。Inference 时，GCS-YOLO-Lane 只解码真实 query predictions。Decoder 依次应用 query score threshold、point-valid threshold、longest contiguous visible span selection、minimum visible-point constraint、optional lane NMS 和 maximum lane count。得到的 visible points 再插值回 TuSimple h-samples。Decoder 不得使用 ground truth，不得伪造 lanes，也不得在 final test 上选择 thresholds。"""),
            fig("protocol"),
        ],
    ),
    (
        "4 实验",
        [
            p("""本节按照标准 lane-detection papers 的组织方式给出：experimental setting、metrics、implementation details、main results、ablations、efficiency 和 failure analysis。由于当前请求要求实验结果后续再填，所有数值项均保留占位符。本节是 paper-ready experiment scaffold，而不是经验结论。"""),
            p("""数据集。当前 active dataset 是 TuSimple fixed-y K56 conversion，包括 3263 张训练图像、363 张 official-validation 图像和 2782 张测试图像。转换流程使用原始 TuSimple JSON 和 images，输出 semantic masks、edge masks、fixed-y lane coordinates、point visibility、original file metadata 和 image shape。论文不得混用该数据集与历史 K32 labels。"""),
            p("""评估指标。主要 TuSimple metrics 是 Accuracy、FP 和 FN。Accuracy 按 sampled lane points 计算："""),
            formula(FORMULAS_ZH["accuracy"]),
            p("""如果比较表包含使用 detection-style metrics 的方法或数据集，precision、recall 和 F1 只能在协议清晰时报告。F1 定义为："""),
            formula(FORMULAS_ZH["f1"]),
            p("""实现细节。正式训练应在远程 RTX 4090 环境完成，使用 ssh_lane conda 环境，并以 batch 32 作为当前 Q12/K56 runs 的起点。默认 model 和 data files 分别为 ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml 与 data/tusimple_gcs_fixed_y_960x544.yaml。最终论文必须从冻结 active-default args.yaml 复制 optimizer、schedule、AMP 状态、workers、augmentation 和 epochs，而不是使用模板命令。"""),
            table("environment"),
            p("""实验实现记录。表 3 采用应用型 YOLO 车道检测论文常见的报告方式，把 runtime environment、input contract、training recipe、precision mode 和 decode-selection record 分开列出。该表当前仍是模板，因为最终取值必须来自冻结 run artifacts。不能凭记忆、通用命令示例或使用不同 K56 fixed-y contract 的 run 来填写。"""),
            table("results"),
            p("""主结果。表 4 是报告 official-val selection 和 one-shot final-test numbers 的唯一位置。Official-validation row 必须包含 selected checkpoint 和 decode parameters。Final-test row 必须使用完全相同的 checkpoint 和 decode parameters。如果使用多次 final-test runs 或 test-tuned thresholds，则该表不能进入主论文。"""),
            table("comparison"),
            p("""Baseline comparison。表 5 采用 UFLDv2 等 lane-detection papers 常见的比较方式：按 representation family 分组，并让所有 metrics 对齐到同一 benchmark protocol。公平表格应区分 original-paper metrics、reproduced metrics 和 project-specific reruns。Runtime 必须声明 hardware 和 precision。CULane、CurveLanes、LLAMAS、3D lane benchmarks 或 custom lane-intrusion datasets 的指标不应与 TuSimple official metrics 混在同一数值表中。"""),
            table("ablation"),
            p("""消融实验。表 6 定义支撑方法设计所需的最小证据。移除 LSEM 用于检验 line-sensitive feature processing 是否有贡献。替换 LaneBiFPN 用于检验 bidirectional multi-scale fusion。关闭 fixed-y refinement 用于检验 point-location feature sampling 是否改进几何。移除 point visibility 用于检验 explicit partial-lane modeling。移除 auxiliary mask/edge supervision 用于检验 dense training cues 是否帮助 structured final output。"""),
            p("""效率。由于该方法在 YOLO-style backbone 后使用 structured decoder，效率报告必须拆分为 preprocessing、network forward、decoding 和 total latency。单个 FPS 不足以说明问题，因为 decoding 和 TuSimple conversion 也可能占用不可忽略的时间。表 7 应填入 model parameters、FLOPs 或 MACs、timing、precision mode、batch size 和 hardware。"""),
            table("efficiency"),
            p("""失败分析。Structured output 让 failure modes 更可观察。在 mask pipeline 中，lane count errors 可能隐藏在 grouping 或 fitting 中。GCS-YOLO-Lane 中，错误可以归因到 query existence、point visibility、geometry、duplicate-like queries 或 spurious extra lanes。表 8 与图 5 定义了 final-test reporting 前应该在 validation data 上报告的诊断类别。"""),
            fig("failure"),
            table("failure"),
        ],
    ),
    (
        "5 讨论",
        [
            p("""GCS-YOLO-Lane 的主要方法变化是 output object。网络不预测 box，也不依赖 final foreground mask 作为 lane instance。它预测一组 lane hypotheses，每个 hypothesis 拥有 lane score、fixed-y point sequence 和 point-level visibility。这使表示更接近 TuSimple-style evaluation 实际消费的对象。"""),
            p("""与 segmentation-based pipelines 相比，所提出表示降低了 post-hoc grouping 的负担。Dense supervision 对学习 lane-sensitive features 仍然有用，但 dense masks 不再定义最终表示。这个区别需要向 reviewer 说明清楚，因为 auxiliary mask 和 edge branches 可能让方法被误解为另一种 segmentation model。"""),
            p("""与 row-anchor methods 相比，GCS-YOLO-Lane 中的 fixed-y anchors 由所有 queries 共享，而不是绑定到固定 lane slot。Learned queries 拥有 lane instances，并需要学习如何分配到不同 lane counts 的场景。这种设计可以更直接暴露 count errors，但也带来 calibration problem：queries 必须抑制 extra lanes，同时不能漏掉短车道或部分可见真车道。"""),
            p("""与 curve-parameter methods 相比，fixed-y point sequence 不够紧凑，但更透明。每个 anchor 有 x coordinate 和 visibility logit，因此局部错误可以按 anchor position 分析。这有助于诊断 lower-lane drift、upper-lane invisibility 和 short side-lane misses。代价是该表示与 fixed-y sampling scheme 绑定，对其他几何规则的数据集需要适配。"""),
            p("""当前证据边界是严格的。Architecture 和 training design 可以描述，因为它们由项目代码和 active contracts 支持。Performance superiority、ablation causality 和 final ranking 不作主张，因为当前稿件按要求保留 experiment placeholders。这种保守写法不是方法描述的弱点，而是避免 test tuning 或把 legacy runs 写成 active-default evidence 的必要条件。"""),
        ],
    ),
    (
        "6 复现与报告清单",
        [
            p("""一篇完整的车道检测论文必须让经验结论可以复现。在本文中，复现尤其重要，因为方法绑定了精确的 fixed-y contract，而且 confidence threshold、point-validity filtering、NMS distance、maximum detections 和 minimum visible points 的微小变化都可能影响 FP 和 FN。因此，最终论文应把 experiment reporting 视为方法的一部分，而不是附带说明。"""),
            p("""首先需要冻结的是训练记录。一个有效结果必须标识 exact model configuration、data configuration、dataset root、image size、pretrained source、optimizer、learning-rate schedule、batch size、AMP setting、workers、augmentation policy、epoch count，以及可用时的 random seed。这些细节最可靠的来源是 selected run 的最终 args.yaml。模板命令可以用于解释，但不能替代真实冻结的训练记录。"""),
            p("""其次需要冻结的是 official-validation selection record。对每个 candidate checkpoint，sweep 应报告测试过的 confidence thresholds、point-validity thresholds、NMS distances、maximum detections、minimum visible-point values 和对应 official metrics。Selected row 必须在 final test 评估前确定。这与强 lane-detection papers 的报告纪律一致：只有选择规则预先固定，test number 才有意义。"""),
            p("""第三个记录是 one-shot final-test package。Final-test command 必须原样复制 official validation 选择出的 checkpoint 和 decode parameters。输出目录应包含 official summary、必要时的 prediction records、生成它们的 command，以及 script version。如果因为格式或路径错误需要第二次 final-test run，论文应说明原因；如果第二次运行是为了改善指标，则该结果不应进入主结论。"""),
            table("reporting"),
            p("""Baseline extraction 也需要同样谨慎。比较表应说明每个数值来自 original paper、public leaderboard、official implementation 还是 project-local rerun。只要比较 speed，就应记录 input resolution 和 runtime hardware。SCNN、UFLD、UFLDv2、LaneATT、CLRNet、LSTR 以及 curve-based methods 等强 baseline 不应只是引用列表，而应以读者可审计的 protocol 报告。"""),
            p("""Ablation records 必须与方法主张绑定。如果论文声称 LSEM 改进 long thin feature modeling，就必须有同一训练与选择协议下的 without-LSEM variant。如果声称 LaneBiFPN 改进 multi-scale fusion，就必须有 simpler-fusion variant。如果声称 point visibility 帮助 partial-lane reasoning，就必须有 visibility ablation。没有这些记录时，方法仍可描述，但经验贡献应保持条件化。"""),
            p("""效率报告应区分 model design 与 implementation overhead。Structured lane head 引入了 decoding 和 TuSimple conversion，这些步骤不包含在 network forward time 中。因此论文应分别报告 preprocessing time、network time、decode time 和 total time。如果使用 batch inference、half precision、TensorRT、CPU decoding 或不同 image loading behavior，表格必须明确说明。"""),
            table("result_card"),
            p("""Failure analysis 应在 final-test reporting 前从 validation diagnostics 中报告。有效类别包括 count errors、short-lane misses、geometry drift、duplicate-like extra queries、spurious extra queries 和 visibility failures。这些类别与 structured representation 直接相关。它们应用于解释模型失败点和提出未来工作，而不是用于调整 final-test behavior。"""),
            p("""最终论文应通过一个简单的 claim-evidence audit。Abstract 和 Introduction 中的每个结果主张都必须指向表 9 或表 10 中的某个冻结 artifact。每个方法主张都必须指向 code、configuration 或 controlled ablation。每个比较主张都必须指向 verified baseline source。这个 audit 是把当前 method-complete draft 转换为 submission-ready empirical paper 的关键步骤。"""),
        ],
    ),
    (
        "7 结论",
        [
            p("""本文提出 GCS-YOLO-Lane，一种用于 TuSimple fixed-y lane prediction 的 YOLO11-based structured lane detection network。核心思想是用 query-owned lane instances 替代 ordinary YOLO box 或 final-mask outputs。每个 query 预测 lane existence、56 个 fixed-y anchors 上的 x coordinates 和 point-level visibility。模型结构结合 line-sensitive feature enhancement、LaneBiFPN multi-scale fusion、query-based GCS lane head、Hungarian matching、visibility-aware losses 和 auxiliary dense supervision。"""),
            p("""当前稿件应被视为 method-complete and experiment-placeholder draft。它给出了 representation、architecture、training objective、inference procedure、evaluation protocol，以及最终报告所需的所有 tables。待 active-default official-validation selection、one-shot final-test evaluation、verified baselines、ablations、efficiency measurements 和 failure analysis 完成后，可以填入占位表，并用受支持的经验结论更新 abstract 与 conclusion。"""),
        ],
    ),
]


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def try_register_font(name: str, path: str) -> None:
    if name in pdfmetrics.getRegisteredFontNames():
        return
    pth = Path(path)
    if pth.exists():
        pdfmetrics.registerFont(TTFont(name, str(pth)))


def register_fonts() -> None:
    try_register_font("TimesNewRoman", "C:/Windows/Fonts/times.ttf")
    try_register_font("TimesNewRoman-Bold", "C:/Windows/Fonts/timesbd.ttf")
    try_register_font("SimSun", "C:/Windows/Fonts/simsun.ttc")
    try_register_font("SimHei", "C:/Windows/Fonts/simhei.ttf")


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    pth = Path(path)
    if pth.exists():
        return ImageFont.truetype(str(pth), size)
    return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, xy, text: str, fill, outline, fnt, align="center") -> None:
    draw.rounded_rectangle(xy, radius=14, fill=fill, outline=outline, width=3)
    x0, y0, x1, y1 = xy
    lines = []
    words = text.split()
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] - draw.textbbox((0, 0), candidate, font=fnt)[0] > (x1 - x0 - 30):
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    total_h = len(lines) * (fnt.size + 8)
    y = y0 + (y1 - y0 - total_h) / 2
    for line in lines:
        w = draw.textbbox((0, 0), line, font=fnt)[2]
        x = x0 + (x1 - x0 - w) / 2 if align == "center" else x0 + 18
        draw.text((x, y), line, fill=(18, 31, 45), font=fnt)
        y += fnt.size + 8


def make_figures() -> None:
    fnt = font("C:/Windows/Fonts/arial.ttf", 26)
    bold = font("C:/Windows/Fonts/arialbd.ttf", 30)
    small = font("C:/Windows/Fonts/arial.ttf", 20)

    img = Image.new("RGB", (1700, 760), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "GCS-YOLO-Lane Overall Architecture", fill=(15, 23, 42), font=bold)
    blocks = [
        ((60, 155, 300, 295), "Input 544 x 960"),
        ((380, 95, 650, 215), "YOLO11-style Backbone"),
        ((380, 280, 650, 400), "LSEM on P3/P4"),
        ((730, 190, 1010, 330), "LaneBiFPN P2-P5"),
        ((1090, 95, 1390, 215), "Query Decoder"),
        ((1090, 280, 1390, 400), "GCS Lane Head"),
        ((1460, 80, 1640, 180), "existence"),
        ((1460, 230, 1640, 330), "fixed-y x"),
        ((1460, 380, 1640, 480), "visibility"),
    ]
    for xy, txt in blocks:
        draw_box(d, xy, txt, (239, 246, 255), (71, 114, 182), fnt)
    for a, b in [((300, 225), (380, 155)), ((300, 225), (380, 340)), ((650, 155), (730, 260)), ((650, 340), (730, 260)), ((1010, 260), (1090, 155)), ((1010, 260), (1090, 340)), ((1390, 340), (1460, 130)), ((1390, 340), (1460, 280)), ((1390, 340), (1460, 430))]:
        d.line([a, b], fill=(29, 78, 137), width=5)
    d.text((1080, 555), "Auxiliary mask / edge branches supervise features during training only.", fill=(71, 85, 105), font=small)
    img.save(FIG_DIR / FIGURES["overview"]["file"])

    img = Image.new("RGB", (1500, 820), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "Fixed-y K56 Representation", fill=(15, 23, 42), font=bold)
    road = [(360, 720), (650, 130), (920, 130), (1180, 720)]
    d.polygon(road, fill=(242, 246, 250), outline=(148, 163, 184))
    for k in range(0, 56, 5):
        y = 710 - 10 * k
        yy = 720 - (y - 160) / (710 - 160) * 590
        d.line([(405, yy), (1135, yy)], fill=(203, 213, 225), width=1)
    lane1 = [(500, 705), (555, 600), (610, 480), (665, 350), (720, 180)]
    lane2 = [(865, 705), (860, 580), (855, 445), (840, 305), (815, 175)]
    d.line(lane1, fill=(6, 125, 150), width=7)
    d.line(lane2, fill=(6, 125, 150), width=7)
    for x, y in lane1[::2] + lane2[::2]:
        d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(14, 165, 233))
    draw_box(d, (70, 250, 290, 350), "Shared y anchors", (240, 253, 250), (13, 148, 136), small)
    draw_box(d, (1210, 250, 1430, 350), "Query predicts x and visibility", (240, 253, 250), (13, 148, 136), small)
    d.text((490, 755), "K = 56 anchors: 710, 700, ..., 160", fill=(71, 85, 105), font=small)
    img.save(FIG_DIR / FIGURES["fixed_y"]["file"])

    img = Image.new("RGB", (1600, 780), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "Query-based GCS Lane Head", fill=(15, 23, 42), font=bold)
    draw_box(d, (70, 150, 310, 300), "P2-P5 spatial tokens + position embeddings", (239, 246, 255), (71, 114, 182), small)
    draw_box(d, (430, 150, 660, 300), "Q = 12 learned lane queries", (239, 246, 255), (71, 114, 182), small)
    draw_box(d, (770, 150, 1040, 300), "Transformer decoder", (239, 246, 255), (71, 114, 182), small)
    draw_box(d, (1160, 80, 1490, 175), "Lane existence logits", (240, 253, 250), (13, 148, 136), small)
    draw_box(d, (1160, 230, 1490, 325), "K fixed-y point x logits", (240, 253, 250), (13, 148, 136), small)
    draw_box(d, (1160, 380, 1490, 475), "K point-valid logits", (240, 253, 250), (13, 148, 136), small)
    for a, b in [((310, 225), (430, 225)), ((660, 225), (770, 225)), ((1040, 225), (1160, 130)), ((1040, 225), (1160, 275)), ((1040, 225), (1160, 425))]:
        d.line([a, b], fill=(29, 78, 137), width=5)
    d.text((420, 560), "One query = one lane hypothesis with geometry and visibility.", fill=(71, 85, 105), font=fnt)
    img.save(FIG_DIR / FIGURES["head"]["file"])

    img = Image.new("RGB", (1700, 760), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "Training and Evaluation Protocol", fill=(15, 23, 42), font=bold)
    steps = [
        ((70, 165, 330, 305), "Original TuSimple JSON + images"),
        ((430, 165, 690, 305), "Regenerate K56 fixed-y labels"),
        ((790, 165, 1050, 305), "Train with Hungarian matching"),
        ((1150, 165, 1410, 305), "Official-val selects decode"),
        ((1450, 165, 1660, 305), "Final test once"),
    ]
    for xy, txt in steps:
        draw_box(d, xy, txt, (239, 246, 255), (71, 114, 182), fnt)
    for a, b in [((330, 235), (430, 235)), ((690, 235), (790, 235)), ((1050, 235), (1150, 235)), ((1410, 235), (1450, 235))]:
        d.line([a, b], fill=(29, 78, 137), width=5)
    d.text((190, 500), "No test tuning. No GT during decode. No fabricated lanes. Report selected parameters.", fill=(185, 28, 28), font=fnt)
    img.save(FIG_DIR / FIGURES["protocol"]["file"])

    img = Image.new("RGB", (1500, 820), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 35), "Failure Analysis Categories", fill=(15, 23, 42), font=bold)
    categories = [
        ((70, 160, 360, 270), "Count error"),
        ((430, 160, 720, 270), "Short-lane miss"),
        ((790, 160, 1080, 270), "Geometry drift"),
        ((250, 390, 540, 500), "Duplicate-like extra"),
        ((850, 390, 1140, 500), "Spurious extra"),
    ]
    for xy, txt in categories:
        draw_box(d, xy, txt, (255, 247, 237), (234, 88, 12), fnt)
    d.text((245, 650), "Each category should be reported on validation diagnostics before final-test reporting.", fill=(71, 85, 105), font=fnt)
    img.save(FIG_DIR / FIGURES["failure"]["file"])


def abstract(lang: str) -> str:
    if lang == "en":
        return (
            "Lane detection requires the recovery of thin, continuous, and partially visible road markings as structured lane instances. "
            "Most segmentation-oriented pipelines learn useful dense evidence, but they still need grouping, fitting, or protocol-specific conversion before the result becomes an ordered lane object. "
            "This paper presents GCS-YOLO-Lane, a YOLO11-based structured lane detection network that predicts lane instances as query-owned fixed-y point sequences. "
            "The model uses line-sensitive feature enhancement, LaneBiFPN multi-scale fusion, and a query-based GCS lane head. "
            "Each of the Q=12 lane queries predicts lane existence, x coordinates on K=56 fixed-y anchors, and point-level visibility. "
            "The network is trained with Hungarian assignment, visibility-aware point supervision, curve and smoothness regularization, and auxiliary mask/edge supervision. "
            "Experiments are deliberately left as placeholders in this version: active-default official-validation selection, one-shot TuSimple final-test reporting, verified baselines, ablations, and efficiency measurements must be inserted before performance claims are made. "
            "Under this evidence boundary, the contribution of this draft is a complete structured YOLO-style formulation for fixed-y lane instance prediction."
        )
    return (
        "车道线检测需要把细长、连续且常常局部可见的道路标线恢复为结构化 lane instances。"
        "多数 segmentation-oriented pipelines 能学习有效的 dense evidence，但在结果成为有序 lane object 之前仍需要 grouping、fitting 或 protocol-specific conversion。"
        "本文提出 GCS-YOLO-Lane，一种 YOLO11-based structured lane detection network，它把车道实例预测为 query-owned fixed-y point sequences。"
        "模型使用 line-sensitive feature enhancement、LaneBiFPN multi-scale fusion 和 query-based GCS lane head。"
        "Q=12 个 lane queries 中，每个 query 都预测 lane existence、K=56 个 fixed-y anchors 上的 x 坐标和 point-level visibility。"
        "训练采用 Hungarian assignment、visibility-aware point supervision、curve/smoothness regularization 以及 auxiliary mask/edge supervision。"
        "本版本按要求将实验结果保留为占位符：active-default official-validation selection、one-shot TuSimple final-test reporting、verified baselines、ablations 和 efficiency measurements 都必须在性能结论前补齐。"
        "在这一证据边界下，本文贡献是给出完整的 structured YOLO-style fixed-y lane instance prediction 方案。"
    )


def styles(lang: str):
    sample = getSampleStyleSheet()
    if lang == "en":
        body_font = "TimesNewRoman" if "TimesNewRoman" in pdfmetrics.getRegisteredFontNames() else "Times-Roman"
        bold_font = "TimesNewRoman-Bold" if "TimesNewRoman-Bold" in pdfmetrics.getRegisteredFontNames() else "Times-Bold"
        body_size = 11.2
        leading = 20.6
        alignment = TA_JUSTIFY
    else:
        body_font = "SimSun" if "SimSun" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
        bold_font = "SimHei" if "SimHei" in pdfmetrics.getRegisteredFontNames() else body_font
        body_size = 11.0
        leading = 20.4
        alignment = TA_LEFT
    sample.add(ParagraphStyle("PaperTitle", parent=sample["Title"], fontName=bold_font, fontSize=18, leading=23, alignment=TA_CENTER, spaceAfter=8))
    sample.add(ParagraphStyle("Meta", parent=sample["Normal"], fontName=body_font, fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#667085")))
    sample.add(ParagraphStyle("Heading", parent=sample["Heading1"], fontName=bold_font, fontSize=13.2, leading=17, spaceBefore=12, spaceAfter=5))
    sample.add(ParagraphStyle("Body", parent=sample["BodyText"], fontName=body_font, fontSize=body_size, leading=leading, alignment=alignment, firstLineIndent=0.42 * cm, spaceAfter=5))
    sample.add(ParagraphStyle("Abstract", parent=sample["Body"], firstLineIndent=0, spaceAfter=8))
    sample.add(ParagraphStyle("Caption", parent=sample["BodyText"], fontName=body_font, fontSize=8.2, leading=10.8 if lang == "en" else 13.5, alignment=TA_LEFT, spaceBefore=3, spaceAfter=5))
    sample.add(ParagraphStyle("Formula", parent=sample["BodyText"], fontName="Courier", fontSize=8.2, leading=11, leftIndent=0.8 * cm, rightIndent=0.4 * cm, spaceBefore=3, spaceAfter=5))
    sample.add(ParagraphStyle("TableHead", parent=sample["BodyText"], fontName=bold_font, fontSize=7.4, leading=9.6 if lang == "en" else 12.5))
    sample.add(ParagraphStyle("TableCell", parent=sample["BodyText"], fontName=body_font, fontSize=7.2, leading=9.4 if lang == "en" else 12.2))
    sample.add(ParagraphStyle("Reference", parent=sample["BodyText"], fontName=body_font, fontSize=8.0, leading=10.1 if lang == "en" else 12.8, leftIndent=0.45 * cm, firstLineIndent=-0.45 * cm, spaceAfter=2.5))
    return sample


def markup(text: str, lang: str) -> str:
    text = html.escape(render_cites(text))
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text.replace("\n", "<br/>")


def table_story(key: str, lang: str, st):
    caption, header, rows = (TABLES_EN if lang == "en" else TABLES_ZH)[key]
    data = [[Paragraph(markup(cell, lang), st["TableHead"]) for cell in header]]
    for row in rows:
        data.append([Paragraph(markup(cell, lang), st["TableCell"]) for cell in row])
    width = 17.1 * cm
    col_widths = [width / len(header)] * len(header)
    tbl = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#98a2b3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [Paragraph(markup(caption, lang), st["Caption"]), tbl, Spacer(1, 7)]


def figure_story(key: str, lang: str, st):
    info = FIGURES[key]
    caption = info[lang]
    img = Image.open(FIG_DIR / info["file"])
    max_w = 15.0 * cm
    max_h = 6.4 * cm
    scale = min(max_w / img.width, max_h / img.height)
    flow = RLImage(str(FIG_DIR / info["file"]), width=img.width * scale, height=img.height * scale)
    fig_num = list(FIGURES).index(key) + 1
    prefix = f"Figure {fig_num}. " if lang == "en" else f"图 {fig_num}. "
    return [KeepTogether([flow, Paragraph(markup(prefix + caption, lang), st["Caption"])])]


def build_story(lang: str):
    st = styles(lang)
    story = [
        Paragraph("GCS-YOLO-Lane: A YOLO11-Based Structured Lane Detection Network" if lang == "en" else "GCS-YOLO-Lane：基于 YOLO11 的结构化车道线检测网络", st["PaperTitle"]),
        Paragraph("Author information: to be completed", st["Meta"]),
        Paragraph("Structured lane detection manuscript with placeholder experiments", st["Meta"]),
        Spacer(1, 10),
        Paragraph("Abstract" if lang == "en" else "摘要", st["Heading"]),
        Paragraph(markup(abstract(lang), lang), st["Abstract"]),
        Paragraph(markup("Index Terms - Lane detection, YOLO11, structured prediction, fixed-y representation, query-based detection, TuSimple." if lang == "en" else "关键词 - 车道线检测，YOLO11，结构化预测，fixed-y 表示，query-based detection，TuSimple。", lang), st["Caption"]),
    ]
    sections = EN_SECTIONS if lang == "en" else ZH_SECTIONS
    for title, blocks in sections:
        story.append(Paragraph(markup(title, lang), st["Heading"]))
        for block in blocks:
            if block.kind == "p":
                story.append(Paragraph(markup(block.value, lang), st["Body"]))
            elif block.kind == "formula":
                story.append(Paragraph(html.escape(block.value), st["Formula"]))
            elif block.kind == "figure":
                story.extend(figure_story(block.value, lang, st))
            elif block.kind == "table":
                story.extend(table_story(block.value, lang, st))
    story.append(Paragraph("Declarations" if lang == "en" else "声明", st["Heading"]))
    declarations = [
        "Data Availability: The manuscript is based on the TuSimple lane detection benchmark and a project-local fixed-y K56 conversion. Exact dataset access terms should be completed before submission.",
        "Code Availability: Code release status is pending author decision.",
        "Ethics Declaration: No human-subject experiment is introduced by this manuscript.",
        "Conflict of Interest: To be completed by the author(s).",
        "Funding: To be completed by the author(s).",
        "Author Contributions: To be completed using CRediT roles after the author list is finalized.",
        "AI Disclosure: This manuscript draft was prepared with AI assistance under author direction. The author(s) remain responsible for factual accuracy, citation verification, experiments, and final claims.",
    ]
    if lang != "en":
        declarations = [
            "数据可用性：本文基于 TuSimple lane detection benchmark 和项目本地 fixed-y K56 conversion。投稿前应补全数据访问条款。",
            "代码可用性：代码发布状态待作者决定。",
            "伦理声明：本文不引入人体实验。",
            "利益冲突：待作者确认。",
            "资助：待作者确认。",
            "作者贡献：作者列表确定后按 CRediT roles 补全。",
            "AI 使用披露：本文初稿在作者指令下由 AI 辅助准备。作者仍需对事实准确性、引用核验、实验和最终结论负责。",
        ]
    for item in declarations:
        story.append(Paragraph(markup(item, lang), st["Body"]))
    story.append(Paragraph("References" if lang == "en" else "参考文献", st["Heading"]))
    for i, entry in enumerate(previous_package.BIB_ENTRIES, 1):
        story.append(Paragraph(markup(f"[{i}] {entry.author}. {entry.title}. {entry.venue}, {entry.year}. {entry.url or ''}", lang), st["Reference"]))
    return story


def header_footer(canvas, doc, lang: str) -> None:
    canvas.saveState()
    canvas.setFont("Times-Roman", 7.5)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(2.0 * cm, 28.2 * cm, "GCS-YOLO-Lane formal manuscript draft")
    canvas.drawRightString(19.0 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(lang: str, filename: str) -> None:
    doc = SimpleDocTemplate(
        str(OUT / filename),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.9 * cm,
        bottomMargin=1.7 * cm,
    )
    doc.build(build_story(lang), onFirstPage=lambda c, d: header_footer(c, d, lang), onLaterPages=lambda c, d: header_footer(c, d, lang))


def markdown_table(key: str, lang: str) -> str:
    caption, header, rows = (TABLES_EN if lang == "en" else TABLES_ZH)[key]
    lines = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(lang: str) -> str:
    lines = [
        "# " + ("GCS-YOLO-Lane: A YOLO11-Based Structured Lane Detection Network" if lang == "en" else "GCS-YOLO-Lane：基于 YOLO11 的结构化车道线检测网络"),
        "",
        "Author information: to be completed",
        "",
        "## " + ("Abstract" if lang == "en" else "摘要"),
        "",
        render_cites(abstract(lang)),
        "",
        "**" + ("Index Terms" if lang == "en" else "关键词") + "** - " + ("Lane detection; YOLO11; structured prediction; fixed-y representation; query-based detection; TuSimple" if lang == "en" else "车道线检测；YOLO11；结构化预测；fixed-y 表示；query-based detection；TuSimple"),
        "",
    ]
    for title, blocks in (EN_SECTIONS if lang == "en" else ZH_SECTIONS):
        lines.extend([f"## {title}", ""])
        for block in blocks:
            if block.kind == "p":
                lines.extend([render_cites(block.value), ""])
            elif block.kind == "formula":
                lines.extend(["```text", block.value, "```", ""])
            elif block.kind == "figure":
                info = FIGURES[block.value]
                idx = list(FIGURES).index(block.value) + 1
                cap = (f"Figure {idx}. " if lang == "en" else f"图 {idx}. ") + info[lang]
                lines.extend([f"![{cap}](figures/{info['file']})", "", f"*{cap}*", ""])
            elif block.kind == "table":
                lines.extend([markdown_table(block.value, lang), ""])
    lines.extend(["## " + ("References" if lang == "en" else "参考文献"), ""])
    for i, entry in enumerate(previous_package.BIB_ENTRIES, 1):
        lines.append(f"[{i}] {entry.author}. {entry.title}. {entry.venue}, {entry.year}. {entry.url or ''}")
    return "\n".join(lines) + "\n"


def esc_tex(text: str) -> str:
    text = render_cites(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_table(key: str, lang: str) -> str:
    caption, header, rows = (TABLES_EN if lang == "en" else TABLES_ZH)[key]
    cols = "p{0.18\\linewidth}" * len(header)
    lines = [r"\begin{table}[htbp]", r"\caption{" + esc_tex(caption) + r"}", r"\scriptsize", r"\begin{tabular}{" + cols + r"}", r"\toprule"]
    lines.append(" & ".join(esc_tex(x) for x in header) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(esc_tex(x) for x in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def build_latex(lang: str) -> str:
    is_en = lang == "en"
    preamble = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=2cm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{hyperref}",
        r"\usepackage{xurl}",
    ]
    if not is_en:
        preamble.extend([r"\usepackage{fontspec}", r"\usepackage{xeCJK}", r"\setmainfont{Times New Roman}", r"\setCJKmainfont{SimSun}"])
    preamble.extend([
        r"\title{" + esc_tex("GCS-YOLO-Lane: A YOLO11-Based Structured Lane Detection Network" if is_en else "GCS-YOLO-Lane：基于 YOLO11 的结构化车道线检测网络") + r"}",
        r"\author{Author information: to be completed}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        esc_tex(abstract(lang)),
        r"\end{abstract}",
    ])
    body: list[str] = []
    for title, blocks in (EN_SECTIONS if is_en else ZH_SECTIONS):
        body.append(r"\section{" + esc_tex(title) + r"}")
        for block in blocks:
            if block.kind == "p":
                body.append(esc_tex(block.value) + "\n")
            elif block.kind == "formula":
                body.append(r"\begin{verbatim}" + "\n" + block.value + "\n" + r"\end{verbatim}")
            elif block.kind == "figure":
                info = FIGURES[block.value]
                idx = list(FIGURES).index(block.value) + 1
                cap = (f"Figure {idx}. " if is_en else f"图 {idx}. ") + info[lang]
                body.extend([r"\begin{figure}[htbp]", r"\centering", rf"\includegraphics[width=0.95\linewidth]{{figures/{Path(info['file']).stem}.png}}", r"\caption{" + esc_tex(cap) + r"}", r"\end{figure}"])
            elif block.kind == "table":
                body.append(latex_table(block.value, lang))
    body.extend([r"\bibliographystyle{IEEEtran}", r"\bibliography{references}", r"\end{document}"])
    return "\n\n".join(preamble + body) + "\n"


def copy_tex_figure_aliases() -> None:
    for info in FIGURES.values():
        src = FIG_DIR / info["file"]
        alias = FIG_DIR / f"{Path(info['file']).stem}.png"
        if src != alias:
            alias.write_bytes(src.read_bytes())


def main() -> None:
    ensure_dirs()
    register_fonts()
    make_figures()
    copy_tex_figure_aliases()
    (OUT / "references.bib").write_text(previous_package.bibtex(), encoding="utf-8")
    (OUT / "gcs_yolo_lane_formal_en.md").write_text(build_markdown("en"), encoding="utf-8")
    (OUT / "gcs_yolo_lane_formal_zh.md").write_text(build_markdown("zh"), encoding="utf-8")
    (OUT / "gcs_yolo_lane_formal_en.tex").write_text(build_latex("en"), encoding="utf-8")
    (OUT / "gcs_yolo_lane_formal_zh.tex").write_text(build_latex("zh"), encoding="utf-8")
    build_pdf("en", "gcs_yolo_lane_formal_en.pdf")
    build_pdf("zh", "gcs_yolo_lane_formal_zh.pdf")
    print(f"Generated formal manuscript package at {OUT}")


if __name__ == "__main__":
    main()
