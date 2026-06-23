from __future__ import annotations

import html
import re
import textwrap
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


@dataclass(frozen=True)
class BibEntry:
    key: str
    entry_type: str
    author: str
    title: str
    year: str
    venue: str
    url: str | None = None
    note: str | None = None


BIB_ENTRIES: list[BibEntry] = [
    BibEntry(
        "pan2018scnn",
        "inproceedings",
        "Xingang Pan and Jianping Shi and Ping Luo and Xiaogang Wang and Xiaoou Tang",
        "Spatial As Deep: Spatial CNN for Traffic Scene Understanding",
        "2018",
        "AAAI Conference on Artificial Intelligence",
        "https://cdn.aaai.org/ojs/12301/12301-13-15829-1-2-20201228.pdf",
    ),
    BibEntry(
        "neven2018lanenet",
        "inproceedings",
        "Davy Neven and Bert De Brabandere and Stamatios Georgoulis and Marc Proesmans and Luc Van Gool",
        "Towards End-to-End Lane Detection: an Instance Segmentation Approach",
        "2018",
        "IEEE Intelligent Vehicles Symposium",
        "https://arxiv.org/abs/1802.05591",
    ),
    BibEntry(
        "hou2019sad",
        "inproceedings",
        "Yuenan Hou and Zheng Ma and Chunxiao Liu and Chen Change Loy",
        "Learning Lightweight Lane Detection CNNs by Self Attention Distillation",
        "2019",
        "IEEE/CVF International Conference on Computer Vision",
        "https://openaccess.thecvf.com/content_ICCV_2019/html/Hou_Learning_Lightweight_Lane_Detection_CNNs_by_Self_Attention_Distillation_ICCV_2019_paper.html",
    ),
    BibEntry(
        "philion2019fastdraw",
        "inproceedings",
        "Jonah Philion",
        "FastDraw: Addressing the Long Tail of Lane Detection by Adapting a Sequential Prediction Network",
        "2019",
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "https://openaccess.thecvf.com/content_CVPR_2019/html/Philion_FastDraw_Addressing_the_Long_Tail_of_Lane_Detection_by_Adapting_CVPR_2019_paper.html",
    ),
    BibEntry(
        "yoo2020e2elmd",
        "inproceedings",
        "Seungwoo Yoo and Hee Seok Lee and Heesoo Myeong and Sungrack Yun and Hyoungwoo Park and Janghoon Cho and Dong Hoon Kim",
        "End-to-End Lane Marker Detection via Row-wise Classification",
        "2020",
        "CVPR Workshops / arXiv",
        "https://arxiv.org/abs/2005.08630",
    ),
    BibEntry(
        "qin2020ufld",
        "inproceedings",
        "Zequn Qin and Huanyu Wang and Xi Li",
        "Ultra Fast Structure-aware Deep Lane Detection",
        "2020",
        "European Conference on Computer Vision",
        "https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3326_ECCV_2020_paper.php",
    ),
    BibEntry(
        "tabelini2020polylanenet",
        "inproceedings",
        "Lucas Tabelini and Rodrigo Berriel and Thiago M. Paixao and Claudine Badue and Alberto F. De Souza and Thiago Oliveira-Santos",
        "PolyLaneNet: Lane Estimation via Deep Polynomial Regression",
        "2020",
        "International Conference on Pattern Recognition / arXiv",
        "https://arxiv.org/abs/2004.10924",
    ),
    BibEntry(
        "zheng2020resa",
        "inproceedings",
        "Tu Zheng and Hao Fang and Yi Zhang and Wenjian Tang and Zheng Yang and Haifeng Liu and Deng Cai",
        "RESA: Recurrent Feature-Shift Aggregator for Lane Detection",
        "2021",
        "AAAI Conference on Artificial Intelligence",
        "https://arxiv.org/abs/2008.13719",
    ),
    BibEntry(
        "tabelini2021laneatt",
        "inproceedings",
        "Lucas Tabelini and Rodrigo Berriel and Thiago M. Paixao and Claudine Badue and Alberto F. De Souza and Thiago Oliveira-Santos",
        "Keep Your Eyes on the Lane: Real-Time Attention-Guided Lane Detection",
        "2021",
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "https://openaccess.thecvf.com/content/CVPR2021/html/Tabelini_Keep_Your_Eyes_on_the_Lane_Real-Time_Attention-Guided_Lane_Detection_CVPR_2021_paper.html",
    ),
    BibEntry(
        "abualsaud2021laneaf",
        "article",
        "Hala Abualsaud and Sean Liu and David B. Lu and Duthon Quy Nguyen and Gordon Chance and Mohamad Aziz and Karim Alahari",
        "LaneAF: Robust Multi-Lane Detection with Affinity Fields",
        "2021",
        "IEEE Robotics and Automation Letters / ICRA",
        "https://arxiv.org/abs/2103.12040",
    ),
    BibEntry(
        "liu2021lstr",
        "inproceedings",
        "Ruijin Liu and Zejian Yuan and Tie Liu and Zhiliang Xiong",
        "End-to-end Lane Shape Prediction with Transformers",
        "2021",
        "IEEE/CVF Winter Conference on Applications of Computer Vision",
        "https://arxiv.org/abs/2011.04233",
    ),
    BibEntry(
        "qin2022ufldv2",
        "article",
        "Zequn Qin and Huanyu Wang and Xi Li",
        "Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification",
        "2022",
        "IEEE Transactions on Pattern Analysis and Machine Intelligence / arXiv",
        "https://arxiv.org/abs/2206.07389",
    ),
    BibEntry(
        "zheng2022clrnet",
        "inproceedings",
        "Tu Zheng and Hao Fang and Yi Zhang and Wenjian Tang and Zheng Yang and Haifeng Liu and Deng Cai",
        "CLRNet: Cross Layer Refinement Network for Lane Detection",
        "2022",
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_CLRNet_Cross_Layer_Refinement_Network_for_Lane_Detection_CVPR_2022_paper.html",
    ),
    BibEntry(
        "feng2022bezierlane",
        "inproceedings",
        "Zhengyang Feng and Shaohua Guo and Xin Tan and Ke Xu and Min Wang and Lizhuang Ma",
        "Rethinking Efficient Lane Detection via Curve Modeling",
        "2022",
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition / arXiv",
        "https://arxiv.org/abs/2203.02431",
    ),
    BibEntry(
        "han2022laneformer",
        "inproceedings",
        "Jin Han and Xiajun Deng and Xinyue Cai and Zhen Yang and Hang Xu and Chunjing Xu and Xiaodan Liang",
        "Laneformer: Object-Aware Row-Column Transformers for Lane Detection",
        "2022",
        "AAAI Conference on Artificial Intelligence",
        "https://arxiv.org/abs/2203.09830",
    ),
    BibEntry(
        "wang2022ganet",
        "inproceedings",
        "Jian Wang and Yuxuan Chen and Zhenhua Huang and Tianpeng Bao and Chenwei Zhang and Jiancheng Fang and Huaidong Zhang and Huchuan Lu",
        "A Keypoint-based Global Association Network for Lane Detection",
        "2022",
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "https://openaccess.thecvf.com/content/CVPR2022/html/Wang_A_Keypoint-Based_Global_Association_Network_for_Lane_Detection_CVPR_2022_paper.html",
    ),
    BibEntry(
        "wu2022yolop",
        "article",
        "Dong Wu and Manwen Liao and Weitian Zhang and Xinggang Wang and Xiang Bai and Wenqing Cheng and Wenyu Liu",
        "YOLOP: You Only Look Once for Panoptic Driving Perception",
        "2022",
        "Machine Intelligence Research / arXiv",
        "https://arxiv.org/abs/2108.11250",
    ),
    BibEntry(
        "han2022yolopv2",
        "misc",
        "Cheng Han and Qichao Zhao and Shuyi Zhang and Yinzi Chen and Zhenlin Zhang and Jinwei Yuan",
        "YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception",
        "2022",
        "arXiv preprint",
        "https://arxiv.org/abs/2208.11434",
    ),
    BibEntry(
        "xu2020curvelanes",
        "inproceedings",
        "Hang Xu and Shaoju Wang and Xinyue Cai and Wei Zhang and Xiaodan Liang and Zhenguo Li",
        "CurveLanes Dataset",
        "2020",
        "European Conference on Computer Vision",
        "https://github.com/SoulmateB/CurveLanes",
    ),
    BibEntry(
        "yan2022once3d",
        "inproceedings",
        "Feng Yan and Ming Nie and Xianpeng Cai and Jianping Shi and Hang Zhao and Ran Quan and Xiaofei He",
        "ONCE-3DLanes: Building Monocular 3D Lane Detection",
        "2022",
        "CVPR Workshops / arXiv",
        "https://arxiv.org/abs/2205.00301",
    ),
    BibEntry(
        "wang2023openlanev2",
        "inproceedings",
        "Huijie Wang and Tianyu Li and Yang Li and Li Chen and Chonghao Sima and Zhiding Yu and Yuning Chai and Mingming Sun and Hongyang Li",
        "OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping",
        "2023",
        "NeurIPS Datasets and Benchmarks / arXiv",
        "https://arxiv.org/abs/2304.10440",
    ),
    BibEntry(
        "wang2024ldtr",
        "misc",
        "Haotian Wang and Yuchen Liu and Zhenhua Guo and Guang Chen",
        "LDTR: Transformer-based Lane Detection with Anchor-chain Representation",
        "2024",
        "arXiv preprint",
        "https://arxiv.org/abs/2403.14354",
    ),
    BibEntry(
        "zhang2024sparselaneformer",
        "misc",
        "Yuan Zhang and Kai Sun and Weiming Zhang and Jun Li",
        "Sparse Laneformer",
        "2024",
        "arXiv preprint",
        "https://arxiv.org/abs/2404.07821",
    ),
    BibEntry(
        "li2024laneptrnet",
        "misc",
        "Xiaolong Li and Yao Chen and Bin Zhang",
        "LanePtrNet: Revisiting Point Voting for Lane Detection",
        "2024",
        "arXiv preprint",
        "https://arxiv.org/abs/2407.12147",
    ),
    BibEntry(
        "zhang2024bezierformer",
        "misc",
        "Zhen Zhang and Yu Zhang and Xiaoyang Wang and Jianping Shi",
        "BezierFormer: A Unified Architecture for 2D and 3D Lane Detection",
        "2024",
        "arXiv preprint",
        "https://arxiv.org/abs/2406.19603",
    ),
    BibEntry(
        "tusimpleBenchmark",
        "misc",
        "TuSimple",
        "TuSimple Lane Detection Benchmark",
        "2017",
        "Benchmark dataset and evaluation protocol",
        "https://github.com/TuSimple/tusimple-benchmark",
        "Used as the protocol reference for fixed horizontal lane samples.",
    ),
]

CITE_ORDER = [entry.key for entry in BIB_ENTRIES]
CITE_NUM = {key: i + 1 for i, key in enumerate(CITE_ORDER)}


def cite(keys: str | list[str], mode: str = "pdf") -> str:
    if isinstance(keys, str):
        key_list = [k.strip() for k in keys.split(",") if k.strip()]
    else:
        key_list = keys
    if mode == "tex":
        return r"\cite{" + ",".join(key_list) + "}"
    nums = [str(CITE_NUM[k]) for k in key_list]
    return "[" + ", ".join(nums) + "]"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(draw: ImageDraw.ImageDraw, xy, text, font, fill, outline="#333333") -> None:
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    x1, y1, x2, y2 = xy
    lines = wrap_text(text, font, x2 - x1 - 28)
    total_h = len(lines) * (font.size + 6)
    y = y1 + ((y2 - y1) - total_h) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((x1 + (x2 - x1 - (bbox[2] - bbox[0])) / 2, y), line, fill="#111111", font=font)
        y += font.size + 6


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.getlength(candidate) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def arrow(draw: ImageDraw.ImageDraw, start, end, color="#4a5568") -> None:
    draw.line([start, end], fill=color, width=5)
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) >= abs(dy):
        sign = 1 if dx >= 0 else -1
        pts = [(x2, y2), (x2 - sign * 22, y2 - 12), (x2 - sign * 22, y2 + 12)]
    else:
        sign = 1 if dy >= 0 else -1
        pts = [(x2, y2), (x2 - 12, y2 - sign * 22), (x2 + 12, y2 - sign * 22)]
    draw.polygon(pts, fill=color)


def make_figures() -> None:
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    font = ImageFont.truetype(str(font_path), 30)
    small = ImageFont.truetype(str(font_path), 24)
    tiny = ImageFont.truetype(str(font_path), 20)
    bold = ImageFont.truetype(str(bold_path), 38)

    img = Image.new("RGB", (1800, 1050), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 42), "Figure 1. GCS-YOLO-Lane architecture", fill="#111111", font=bold)
    d.text((70, 93), "YOLO11-style feature extraction is retained, but the terminal output is a structured lane set.", fill="#333333", font=small)
    boxes = [
        (80, 210, 340, 350, "Input image\n544 x 960", "#e6f4f1"),
        (430, 170, 720, 390, "YOLO11-style\nBackbone\nP2/P3/P4/P5", "#e9edf7"),
        (820, 150, 1070, 270, "LSEM on P3", "#fff4d6"),
        (820, 315, 1070, 435, "LSEM on P4", "#fff4d6"),
        (1160, 190, 1445, 390, "LaneBiFPN\nbidirectional\nmulti-scale fusion", "#eaf7df"),
        (1530, 170, 1760, 410, "GCS lane head\nQ=12, K=56", "#fbe7e4"),
    ]
    for xy in boxes:
        draw_box(d, xy[:4], xy[4], font, xy[5])
    arrow(d, (340, 280), (430, 280))
    arrow(d, (720, 260), (820, 210))
    arrow(d, (720, 310), (820, 375))
    arrow(d, (1070, 210), (1160, 260))
    arrow(d, (1070, 375), (1160, 320))
    arrow(d, (1445, 290), (1530, 290))
    out_boxes = [
        (1450, 590, 1730, 675, "pred_points: B x 12 x 56 x 2", "#f5f5f5"),
        (1450, 700, 1730, 785, "pred_logits: B x 12", "#f5f5f5"),
        (1450, 810, 1730, 895, "pred_valid_logits: B x 12 x 56", "#f5f5f5"),
        (1120, 670, 1390, 775, "aux mask / edge\ntraining signals", "#f5f5f5"),
    ]
    for xy in out_boxes:
        draw_box(d, xy[:4], xy[4], small, xy[5], "#8a8a8a")
    arrow(d, (1645, 410), (1585, 590))
    arrow(d, (1645, 410), (1585, 700))
    arrow(d, (1645, 410), (1585, 810))
    arrow(d, (1280, 390), (1280, 670))
    img.save(FIG_DIR / "figure_01_architecture.png", dpi=(220, 220))

    img = Image.new("RGB", (1600, 1100), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 42), "Figure 2. Fixed-y K56 lane representation", fill="#111111", font=bold)
    d.text((70, 93), "The y anchors are shared and fixed; the model predicts x and point visibility.", fill="#333333", font=small)
    road = [(430, 980), (1180, 980), (1010, 190), (600, 190)]
    d.polygon(road, fill="#eef2f6", outline="#667085")
    for lane_x in [620, 780, 960]:
        pts = []
        for i, y in enumerate(range(930, 240, -48)):
            offset = (i * 8) if lane_x < 800 else (-i * 5 if lane_x > 900 else i * 1)
            pts.append((lane_x + offset, y))
        d.line(pts, fill="#006d77", width=8)
    for idx, y in enumerate(range(930, 250, -48)):
        color = "#d0d5dd" if idx % 2 else "#98a2b3"
        d.line((520, y, 1090, y), fill=color, width=2)
    d.text((1160, 260), "K = 56 anchors", fill="#111111", font=font)
    d.text((1160, 310), "710, 700, ..., 160", fill="#111111", font=small)
    d.text((1160, 360), "normalized by original H=720", fill="#111111", font=small)
    draw_box(d, (125, 760, 390, 880), "valid point", small, "#e6f4f1")
    draw_box(d, (125, 900, 390, 1010), "invisible anchor", small, "#f5f5f5")
    d.ellipse((455, 884, 475, 904), fill="#006d77")
    d.line((475, 894, 560, 835), fill="#006d77", width=3)
    d.ellipse((455, 792, 475, 812), fill="#d0d5dd")
    img.save(FIG_DIR / "figure_02_fixed_y.png", dpi=(220, 220))

    img = Image.new("RGB", (1800, 1000), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 42), "Figure 3. Query-based GCS lane head", fill="#111111", font=bold)
    d.text((70, 93), "Each query owns lane existence, fixed-y geometry, and per-anchor visibility.", fill="#333333", font=small)
    draw_box(d, (80, 260, 340, 420), "Multi-scale tokens\nfrom LaneBiFPN", font, "#eaf7df")
    draw_box(d, (470, 180, 740, 320), "Learnable lane\nqueries Q=12", font, "#e9edf7")
    draw_box(d, (470, 420, 740, 560), "Transformer\ndecoder", font, "#fff4d6")
    draw_box(d, (875, 170, 1165, 300), "Existence\nMLP", font, "#fbe7e4")
    draw_box(d, (875, 380, 1165, 510), "Point x\nMLP + refinement", font, "#fbe7e4")
    draw_box(d, (875, 590, 1165, 720), "Point visibility\nMLP + refinement", font, "#fbe7e4")
    draw_box(d, (1300, 160, 1670, 285), "lane confidence\nB x 12", font, "#f5f5f5")
    draw_box(d, (1300, 375, 1670, 500), "fixed-y points\nB x 12 x 56 x 2", font, "#f5f5f5")
    draw_box(d, (1300, 590, 1670, 715), "visibility logits\nB x 12 x 56", font, "#f5f5f5")
    arrow(d, (340, 340), (470, 490))
    arrow(d, (605, 320), (605, 420))
    arrow(d, (740, 490), (875, 235))
    arrow(d, (740, 490), (875, 445))
    arrow(d, (740, 490), (875, 655))
    arrow(d, (1165, 235), (1300, 220))
    arrow(d, (1165, 445), (1300, 435))
    arrow(d, (1165, 655), (1300, 650))
    d.text((475, 795), "Fixed-y mode: y is restored from the shared anchor buffer; the head regresses x.", fill="#333333", font=small)
    img.save(FIG_DIR / "figure_03_head.png", dpi=(220, 220))

    img = Image.new("RGB", (1800, 980), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 42), "Figure 4. Training and evaluation protocol", fill="#111111", font=bold)
    d.text((70, 93), "Candidate and decode selection must be made on official validation; test is reported once.", fill="#333333", font=small)
    items = [
        (80, 230, 390, 370, "Original TuSimple\nJSON + images", "#e6f4f1"),
        (500, 230, 810, 370, "K56 fixed-y\nlabel conversion", "#e9edf7"),
        (920, 230, 1230, 370, "Hungarian\nquery-lane matching", "#fff4d6"),
        (1340, 230, 1650, 370, "Visibility-aware\ntraining losses", "#fbe7e4"),
        (500, 570, 810, 710, "Official-val\nsweep", "#eaf7df"),
        (920, 570, 1230, 710, "Selected decode\nfrozen", "#eaf7df"),
        (1340, 570, 1650, 710, "One-shot\nfinal test", "#f5f5f5"),
    ]
    for xy in items:
        draw_box(d, xy[:4], xy[4], font, xy[5])
    arrow(d, (390, 300), (500, 300))
    arrow(d, (810, 300), (920, 300))
    arrow(d, (1230, 300), (1340, 300))
    arrow(d, (1495, 370), (650, 570))
    arrow(d, (810, 640), (920, 640))
    arrow(d, (1230, 640), (1340, 640))
    d.text((90, 825), "Integrity rule: no test threshold search, no GT during inference, no fabricated lanes, no claim of improvement without official-val evidence.", fill="#333333", font=tiny)
    img.save(FIG_DIR / "figure_04_protocol.png", dpi=(220, 220))


def esc_tex(text: str) -> str:
    placeholders: dict[str, str] = {}

    def hold(match: re.Match[str]) -> str:
        key = f"@@CITE{len(placeholders)}@@"
        placeholders[key] = cite(match.group(1), "tex")
        return key

    text = re.sub(r"\[\[CITE:([^\]]+)\]\]", hold, text)
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
    escaped = "".join(replacements.get(ch, ch) for ch in text)
    escaped = escaped.replace("<= ", r"$\leq$ ").replace(">=", r"$\geq$")
    for key, value in placeholders.items():
        escaped = escaped.replace(key, value)
    escaped = re.sub(r"`([^`]+)`", r"\\texttt{\1}", escaped)
    return escaped


def render_cites(text: str) -> str:
    return re.sub(r"\[\[CITE:([^\]]+)\]\]", lambda m: cite(m.group(1), "pdf"), text)


EN_SECTIONS: list[tuple[str, list[tuple[str | None, list[str]]]]] = [
    (
        "Introduction",
        [
            (
                "Lane Detection as Structured Geometry",
                [
                    "Lane detection is a structured perception problem rather than a compact object localization problem. A useful detector must recover visible lane instances, preserve their geometric order, and express partial visibility under occlusion, worn paint, shadows, and perspective distortion. This output is different from the bounding boxes used by ordinary object detection and also different from final foreground masks, because downstream planning modules need lane geometry as ordered coordinates. The central premise of this paper is therefore simple: a lane detector should predict lane objects in the form in which they are consumed.",
                    "The literature has repeatedly shown that lane markings require contextual reasoning over long, thin structures. Spatial message passing in SCNN, self-attention distillation, recurrent feature aggregation, and keypoint association all address the fact that lane evidence is narrow but spatially extended [[CITE:pan2018scnn,hou2019sad,zheng2020resa,wang2022ganet]]. These works motivate feature mechanisms that are sensitive to continuity rather than only local texture. GCS-YOLO-Lane follows this motivation but places the final representation in an explicit query-owned point sequence.",
                    "The benchmark setting used in this project further strengthens the case for structured output. TuSimple evaluates lane predictions at fixed horizontal samples, so a method that directly predicts fixed-y point sequences can reduce late-stage conversion ambiguity [[CITE:tusimpleBenchmark]]. In the active project contract, each lane is represented at K=56 vertical anchors corresponding to original y coordinates 710, 700, 690, ..., 160, normalized by the original height 720. This contract is not an implementation detail; it defines the representation around which the model, labels, training losses, decoding rules, and official evaluation are organized.",
                ],
            ),
            (
                "Representation Gap in Existing Pipelines",
                [
                    "Segmentation-based lane detectors learn dense lane evidence, and this line remains important because dense supervision exposes thin visual structures to strong pixel-level gradients. LaneNet-style instance segmentation and affinity-field methods estimate lane pixels or pixel associations and then recover instances through clustering, affinity grouping, or fitting [[CITE:neven2018lanenet,abualsaud2021laneaf]]. Their strength is dense visual coverage. Their representational cost is that the final output still needs a conversion step from pixels into lane instances and sampled coordinates.",
                    "Row-based and anchor-based methods reduce this gap by predicting sparse lane coordinates under predefined structures. Row-wise classification methods and hybrid anchor methods are efficient and well aligned with sampled lane benchmarks [[CITE:yoo2020e2elmd,qin2020ufld,qin2022ufldv2]]. Attention-guided line anchors and cross-layer refinement further strengthen proposal quality and feature aggregation [[CITE:tabelini2021laneatt,zheng2022clrnet]]. These methods show that sparse structured prediction is a practical choice, but lane ownership is still largely tied to row classifiers, anchors, or proposal refinements rather than to learned lane queries.",
                    "Query-based, curve-based, and sequence-based methods move closer to the representation pursued here. LSTR predicts lane shapes with transformer reasoning, PolyLaneNet and BezierLaneNet model lane geometry through polynomial or Bezier representations, and FastDraw treats lane detection as a sequential drawing task [[CITE:liu2021lstr,tabelini2020polylanenet,feng2022bezierlane,philion2019fastdraw]]. These methods reduce dependence on dense mask postprocessing. GCS-YOLO-Lane is positioned in the same structured-output family, but it keeps a direct fixed-y point sequence and integrates it with a YOLO11-style feature extractor.",
                ],
            ),
            (
                "GCS-YOLO-Lane Overview",
                [
                    "GCS-YOLO-Lane adapts a YOLO11-style network into a structured lane instance detector. The architecture keeps the practical feature-extraction pattern of a YOLO-style backbone, inserts line-sensitive enhancement modules into intermediate stages, fuses P2-P5 features through LaneBiFPN, and replaces ordinary detection or mask output with a query-based GCS lane head. Each lane query predicts lane existence, x coordinates at fixed-y anchors, and per-anchor visibility. Auxiliary mask and edge outputs are retained only as training signals.",
                    "The active default output contract is concrete. For a batch of B images, the model emits `pred_points` with shape B x 12 x 56 x 2, `pred_logits` with shape B x 12, and `pred_valid_logits` with shape B x 12 x 56. The auxiliary outputs are `aux_mask_logits` with shape B x 2 x H x W and `aux_edge_logits` with shape B x 1 x H x W. In fixed-y mode, the y coordinate is restored from a shared anchor buffer and the head regresses x. This gives every candidate lane a stable tensor while preserving visibility through point-validity logits.",
                    "The method is designed around two linked observations. First, lane evidence is long, thin, and directionally biased, so intermediate feature maps should emphasize line-like continuity. Second, a lane instance is not merely a curve; it is a partially visible ordered sequence whose lane-level existence and point-level visibility must be modeled together. GCS-YOLO-Lane operationalizes these observations through LSEM, LaneBiFPN, a query decoder, Hungarian matching, and visibility-aware losses.",
                ],
            ),
            (
                "Contributions and Scope",
                [
                    "This paper makes four design contributions. First, it formulates a YOLO-style lane detector as a structured fixed-y set predictor with Q=12 learnable queries and K=56 ordered anchors. Second, it describes a line-sensitive feature pathway that combines strip responses, direction gating, coordinate reweighting, and dilated context. Third, it uses LaneBiFPN to fuse fine and semantic multi-scale features before query decoding. Fourth, it trains the structured output with Hungarian matching, point geometry, point visibility, curve and smoothness regularization, and auxiliary dense supervision.",
                    "The empirical claims in this manuscript are deliberately bounded. The current user request asks for experimental results to remain as placeholders, so the paper does not state state-of-the-art performance, superiority over specific baselines, or completed ablation gains. Tables in the Experiments section identify the exact values that must be inserted after active-default official-validation selection and one-shot final-test reporting. This boundary is essential because the project rules prohibit test tuning and require official validation to select thresholds, checkpoints, and postprocessing.",
                    "The paper is organized as follows. Section 2 positions GCS-YOLO-Lane among segmentation, row/anchor, curve/query, and YOLO-style driving perception methods. Section 3 describes the fixed-y representation, architecture, lane head, matching, losses, and decoding. Section 4 defines the planned evaluation protocol and uses placeholders for pending results. Sections 5 and 6 discuss the implications and limitations of the structured design, and Section 7 concludes with the strongest claim currently supported by code and project contracts.",
                ],
            ),
        ],
    ),
    (
        "Related Work",
        [
            (
                "Segmentation and Dense Lane Representations",
                [
                    "Segmentation-based lane detection treats lane markings as pixel-level regions or instance-aware embeddings. SCNN is a representative early system that used spatial message passing to propagate information across rows and columns, showing that thin lane structures benefit from non-local context [[CITE:pan2018scnn]]. LaneNet modeled lane detection through semantic segmentation and instance embedding [[CITE:neven2018lanenet]], while LaneAF used affinity fields to group lane pixels into instances [[CITE:abualsaud2021laneaf]]. These methods are important because they expose lane continuity to dense supervision.",
                    "The limitation of dense lane representations is not that they are weak visual learners. Rather, their final output is not yet the structured lane object required by fixed-sample evaluation or planning. Pixels must be grouped, fitted, ordered, and converted to sample coordinates. In clean scenes this conversion can be reliable, but under broken paint, shadows, nearby lane markings, and partial visibility, the conversion layer may introduce heuristic decisions. GCS-YOLO-Lane retains dense mask and edge branches only as auxiliary feature supervision, while its final output is a query-owned point sequence.",
                    "Dense-supervision methods also motivate why auxiliary branches remain useful in this project. Thin lane markings provide sparse coordinate targets, and auxiliary mask or edge supervision can enrich shared feature learning. The proposed design therefore does not reject segmentation information. It separates the role of dense supervision from the role of final output: dense maps help train the network, but decoded lanes come from structured queries.",
                ],
            ),
            (
                "Row-Based, Anchor-Based, and Refinement-Based Methods",
                [
                    "Row-based and anchor-based approaches are strong because they exploit the regular geometry of lane benchmarks. End-to-end row-wise marker detection and UFLD reformulate lane localization as sparse classification over predefined row locations, improving speed and reducing heavy postprocessing [[CITE:yoo2020e2elmd,qin2020ufld]]. UFLDv2 extends this idea with hybrid anchor driven ordinal classification, showing that anchor design can encode useful structural prior for lane localization [[CITE:qin2022ufldv2]]. These methods provide an important reference point for any fixed-y representation.",
                    "Anchor-based attention and cross-layer refinement methods further improve lane localization by attaching candidate lane structures to informative features. LaneATT uses attention-guided line anchors for real-time detection [[CITE:tabelini2021laneatt]], and CLRNet uses cross-layer refinement to gather both semantic and fine localization cues [[CITE:zheng2022clrnet]]. These designs emphasize the need to combine global context with spatial precision. GCS-YOLO-Lane adopts the same broad requirement through LaneBiFPN and a query decoder, but it changes the ownership unit from anchors to learned queries.",
                    "The distinction matters because lane ownership influences count calibration and visibility reasoning. In row or anchor systems, a lane may be represented by predefined slots or proposals; in GCS-YOLO-Lane, each learned query carries existence, geometry, and visibility. This does not automatically make the method superior. It creates a different design space in which query allocation, matching, and visibility supervision become central optimization questions.",
                ],
            ),
            (
                "Query, Curve, and Sequence Lane Prediction",
                [
                    "Structured lane prediction has been explored through transformer queries, curve parameterization, polynomial regression, keypoint association, and sequence generation. LSTR predicts lane shape with transformer-style reasoning and set prediction [[CITE:liu2021lstr]]. PolyLaneNet and BezierLaneNet represent lanes through compact curve parameters instead of dense masks [[CITE:tabelini2020polylanenet,feng2022bezierlane]]. FastDraw uses sequential prediction to address long-tail lane shapes [[CITE:philion2019fastdraw]], and GANet associates keypoints globally [[CITE:wang2022ganet]]. These methods are closest to GCS-YOLO-Lane in spirit because they predict lane structure directly.",
                    "More recent transformer and point-based variants continue this trend. Laneformer models object-aware row-column relationships [[CITE:han2022laneformer]], while LDTR, Sparse Laneformer, LanePtrNet, and BezierFormer explore anchor-chain, sparse-query, point-voting, and unified 2D/3D lane formulations [[CITE:wang2024ldtr,zhang2024sparselaneformer,li2024laneptrnet,zhang2024bezierformer]]. These works show that the field is moving beyond purely dense segmentation. The relevant question for this paper is therefore not whether structured lane prediction is possible, but how a YOLO-style backbone can be reshaped to emit fixed-y lane instances.",
                    "GCS-YOLO-Lane differs from curve-parameter methods by preserving direct sampled points. A compact curve can be elegant and efficient, but fixed-y point sequences align naturally with TuSimple-style evaluation and allow anchor-level visibility. The proposed head therefore predicts x at each fixed-y anchor and estimates whether that anchor is visible. This representation avoids converting global curve coefficients into sampled visible spans as the primary output mechanism.",
                ],
            ),
            (
                "YOLO-Style Driving Perception and Dataset Scope",
                [
                    "YOLO-style networks have been widely used in real-time driving perception because they offer efficient shared feature extraction and deployment-friendly inference. YOLOP and YOLOPv2 show how a YOLO-like encoder can support multi-task driving perception, including lane-related outputs, drivable area, and object detection [[CITE:wu2022yolop,han2022yolopv2]]. GCS-YOLO-Lane follows the engineering intuition that a YOLO-style feature body is useful, but it changes the lane branch from segmentation-oriented output to structured query prediction.",
                    "Dataset scope also shapes method design. TuSimple uses fixed horizontal sampling and is well matched to the K56 fixed-y contract [[CITE:tusimpleBenchmark]]. CULane, CurveLanes, ONCE-3DLanes, and OpenLane-V2 cover broader road layouts, curved lanes, 3D geometry, or topology reasoning [[CITE:pan2018scnn,xu2020curvelanes,yan2022once3d,wang2023openlanev2]]. The present manuscript focuses on the active TuSimple 2D fixed-y branch. Cross-dataset validation is treated as future work rather than as an already supported empirical claim.",
                    "In summary, prior work provides strong evidence for dense supervision, sparse row structures, anchor refinement, curve modeling, transformers, and YOLO-style driving backbones. The specific gap addressed here is narrower: a YOLO11-based structured detector that predicts query-owned fixed-y point sequences with explicit point visibility, trained under a leakage-free official-validation protocol.",
                ],
            ),
        ],
    ),
    (
        "Method",
        [
            (
                "Problem Formulation and Output Contract",
                [
                    "Given an RGB road image resized with the project contract `--imgsz 544 960` in H,W order, GCS-YOLO-Lane predicts a set of candidate lane instances. The maximum number of candidate queries is Q=12. Each candidate query owns K=56 ordered points, a lane-existence logit, and K point-validity logits. The final decoded lanes are obtained from query scores, point visibility, minimum visible-point constraints, optional lane NMS, and a maximum detection count selected on official validation.",
                    "The output tensors are fixed by the active branch contract. `pred_points` has shape B x 12 x 56 x 2, `pred_logits` has shape B x 12, and `pred_valid_logits` has shape B x 12 x 56. The auxiliary tensors are `aux_mask_logits` with shape B x 2 x H x W and `aux_edge_logits` with shape B x 1 x H x W. These auxiliary tensors are used for training supervision; they are not the final lane representation.",
                    "In fixed-y mode, the second coordinate of each predicted point is not freely regressed. The y coordinate comes from a registered fixed-y anchor buffer, while the head predicts the x coordinate. This x-only prediction is still emitted as 2D points so that downstream code and losses operate on consistent point tensors. The design gives the model a stable output grid without assuming that every anchor along a lane is visible.",
                ],
            ),
            (
                "Fixed-Y Label Representation",
                [
                    "The fixed-y representation converts each annotated TuSimple lane into a bottom-to-top sequence sampled at shared y anchors. The active branch uses the exact anchors 710, 700, 690, ..., 160 in original image coordinates, normalized by the original height 720. The normalized range is therefore 710/720 to 160/720 with K=56 points. The K56 labels must be regenerated from original TuSimple JSON and images rather than resampled from historical K32 labels.",
                    "For each ground-truth lane, the conversion routine filters invalid coordinates, orders the lane from bottom to top, removes duplicate y positions, and interpolates x at each fixed-y anchor. Anchors inside the valid lane span receive normalized x and the shared y coordinate. Anchors outside the visible lane span keep the fixed y value but are marked invisible through `lane_valid=0`. This separation lets the model learn geometry only where evidence exists while still producing a complete fixed-size tensor.",
                    "The label files also contain dense supervision arrays. Each `.npz` label includes `semantic_mask`, `edge_mask`, `lanes`, `lane_valid`, `num_lanes`, `point_mode`, `fixed_y`, `raw_file`, `image_shape`, and `num_points`. These fields make the representation auditable: the split, anchor order, mask shape, image shape, and point mode can be checked before training. The paper treats these checks as part of the method because the representation contract is a core contribution of this branch.",
                ],
            ),
            (
                "Overall Architecture",
                [
                    "Figure 1 summarizes the architecture. The model starts from a YOLO11-style backbone that produces P2, P3, P4, and P5 feature maps. P3 and P4 are passed through line-sensitive enhancement modules. The resulting P2, enhanced P3, enhanced P4, and P5 features are fused by LaneBiFPN into a set of aligned multi-scale feature maps. The GCS lane head then flattens these maps into spatial tokens, adds positional and level information, and decodes learned lane queries.",
                    "The architecture intentionally keeps the feature extractor familiar but changes the prediction head. A standard YOLO head would emit boxes, class scores, or segmentation masks. GCS-YOLO-Lane instead emits lane sequences. This substitution is not merely a postprocessor: it changes the training target, matching rule, tensor contract, and decoding interface. The auxiliary mask and edge branches are attached to support feature learning but do not define the predicted lane object.",
                    "The active model YAML defines the LaneBiFPN over feature indices corresponding to P2, enhanced P3, enhanced P4, and P5, and then attaches a `GCSLaneHead` configured as `[12, 56, 3, 8, True, fixed_y, 0.9861111111111112, 0.2222222222222222]`. This encodes the main hyperparameters: 12 queries, 56 anchors, 3 decoder layers, 8 attention heads, auxiliary outputs enabled, and the active fixed-y range.",
                ],
            ),
            (
                "Line-Sensitive Feature Enhancement",
                [
                    "The line-sensitive enhancement module is inserted after P3 and P4 because these stages balance spatial detail and semantic context. The module is motivated by a simple property of lane markings: the useful signal is thin but coherent across space. A generic convolutional response may detect local paint fragments, but lane detection benefits from features that preserve continuity along elongated structures.",
                    "LSEM contains four parts. A line-strip attention component applies horizontal and vertical strip depthwise convolutions, then uses a direction gate to weight these responses. A coordinate reweighting branch modulates the feature map along height and width so that spatial location remains explicit. A dilated context branch increases the receptive field without discarding resolution. The module output is connected through a residual path so that line enhancement augments rather than replaces the backbone features.",
                    "The expected advantage of LSEM is representational rather than yet empirically proven in this placeholder manuscript. It should make intermediate features more sensitive to long lane evidence, but the magnitude of its contribution must be reported only after an ablation under the active official-validation protocol. The Experiments section therefore includes a placeholder ablation row rather than a claimed numerical gain.",
                ],
            ),
            (
                "LaneBiFPN Multi-Scale Fusion",
                [
                    "Lane localization requires both high-resolution evidence and semantic discrimination. P2 and P3 preserve finer spatial detail, which is important for point regression. P4 and P5 provide broader context, which helps suppress road texture, shadows, and markings that resemble lanes. LaneBiFPN projects these levels to a shared channel dimension and applies bidirectional weighted fusion so that the lane head can attend across scales.",
                    "The fusion design follows the practical insight seen in refinement-based lane detectors: high-level context alone is insufficient for precise lane coordinates, while shallow features alone can be visually ambiguous. By presenting fused P2-P5 features as spatial tokens, the query decoder can draw from both local and global evidence. This is particularly useful for partially visible lanes, where local point evidence may be missing and query-level context must stabilize the prediction.",
                    "As with LSEM, the manuscript does not claim that LaneBiFPN improves performance until a matched ablation exists. The method section states the design and its motivation; the experiment plan states the required comparison against simpler fusion. This split keeps the paper honest while still documenting the implemented architecture.",
                ],
            ),
            (
                "Query-Based GCS Lane Head",
                [
                    "The GCS lane head uses 12 learnable queries as candidate lane slots. Each query attends to the multi-scale spatial tokens through a transformer decoder and produces three linked predictions: lane existence, fixed-y point geometry, and point visibility. This design makes lane ownership explicit. A query is not only a score; it is a structured lane hypothesis whose geometry and visibility are trained together.",
                    "In fixed-y mode, the point predictor produces x logits for 56 anchors. A query-specific reference-logit buffer supplies different initial geometric patterns, and image-conditioned refinement samples features at predicted point locations to refine the x logits. The y coordinate is then restored from the fixed-y anchor buffer. The same head also refines point-valid logits, allowing the model to mark which anchors along a lane are visible.",
                    "The output representation supports partial lanes naturally. A lane can have high existence confidence while only a subset of its anchors are visible. Conversely, an unmatched query can be suppressed at the existence level and should not decode into a final lane. This distinction is important for lane-count stability because false extra lanes, duplicate-like queries, and low-score short lanes are different failure modes and should not be collapsed into one mask-quality problem.",
                ],
            ),
            (
                "Matching, Losses, and Decoding",
                [
                    "Training uses Hungarian matching between predicted queries and ground-truth lanes. For each image, the matcher builds a Q x N cost matrix, where Q is the number of predicted queries and N is the number of ground-truth lanes. The cost combines visible-point distance, curve consistency, and existence confidence. This set-matching step prevents the model from relying on a fixed lane index and lets learned queries specialize through assignment.",
                    "After matching, the loss supervises lane existence, visible-point geometry, point validity, smoothness, curve regularity, auxiliary mask prediction, and auxiliary edge prediction. The current code also logs default-disabled experimental items such as lane-balanced point loss, short-valid recall loss, count losses, duplicate-margin loss, and spurious-margin loss. This paper treats those items carefully: they are not core method claims unless explicitly enabled in a frozen experiment.",
                    "Inference decodes only real query predictions. The decoder applies a query score threshold, point-validity threshold, longest contiguous visible span selection, a minimum visible-point requirement, optional lane NMS based on average x distance over shared anchors, and `max_det`. The project contract requires that decode use no ground truth, fabricate no lanes, and sort final lanes left to right by bottom visible x when needed. Thresholds and postprocess settings must be selected on official validation, not on final test.",
                ],
            ),
        ],
    ),
    (
        "Experiments",
        [
            (
                "Dataset and Protocol",
                [
                    "The experimental section is intentionally written as a protocol-complete placeholder. The active dataset is the TuSimple fixed-y K56 conversion with 3263 training images, 363 validation images, and 2782 test images. The input size is fixed to `--imgsz 544 960` in H,W order. The model and data configs are `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml` and `data/tusimple_gcs_fixed_y_960x544.yaml`.",
                    "The official TuSimple metrics are Accuracy, FP, and FN. The project may also report an internal `official_score`, lane-count accuracy, per-count accuracy, and timing fields, but these must be clearly separated from the official metric. Thresholds, checkpoint choice, NMS distance, maximum lane count, minimum visible points, and point-validity threshold must be selected on the official-validation surface. Final test is used once for the selected candidate.",
                    "This protocol is part of the paper contribution because it prevents leakage. The manuscript must not tune on test, use ground truth during inference or decode, fabricate lanes, silently change official metrics, or claim improvement without official-validation evidence. Any result table that violates these rules should be excluded from the main paper and documented only as a non-comparable historical artifact.",
                ],
            ),
            (
                "Implementation Details",
                [
                    "The default formal training recipe should be copied from the frozen active-default run once it exists. The expected setting uses the remote RTX 4090 environment, the `ssh_lane` conda environment, batch 32 as the starting point, AdamW optimization, cosine learning-rate decay, K56 fixed-y labels regenerated from original TuSimple JSON and images, and the default model/data paths. If batch size, AMP state, workers, or preprocessing differ, the paper must report the exact `args.yaml` rather than a template.",
                    "The result placeholders in Table 2 must be filled only after active-default official-validation selection. The paper should report the selected row, the selected decode parameters, and the one-shot final-test report using exactly those parameters. If an experiment uses a default-disabled loss knob, the table must name it as an experiment, not as the mainline method.",
                ],
            ),
            (
                "Main Results and Baselines",
                [
                    "The main comparison table should include representative segmentation, row/anchor, refinement, query, curve, and sequence methods. Candidate baselines include SCNN, SAD, UFLD, UFLDv2, LaneATT, CLRNet, LSTR, PolyLaneNet, BezierLaneNet, and related structured methods [[CITE:pan2018scnn,hou2019sad,qin2020ufld,qin2022ufldv2,tabelini2021laneatt,zheng2022clrnet,liu2021lstr,tabelini2020polylanenet,feng2022bezierlane]]. Each row must state whether the metric comes from the original paper, reimplementation, or project reproduction.",
                    "Because the current request asks for placeholders, the manuscript does not claim a rank. The table columns should include Accuracy, FP, FN, runtime or FPS when comparable, input resolution, and notes about evaluation protocol. Hardware-dependent speed numbers should not be mixed without caveat. Metrics from CULane, CurveLanes, LLAMAS, OpenLane, or 3D lane benchmarks should not be placed in the same TuSimple comparison table.",
                    "Ablation studies should directly correspond to the method claims. The required ablations include removing LSEM, replacing LaneBiFPN with simpler fusion, disabling fixed-y image-conditioned refinement, disabling the point-validity branch, and removing auxiliary mask/edge supervision. Optional count or margin losses should appear in a separate experiment block only when they are explicitly enabled and selected through official validation.",
                ],
            ),
            (
                "Failure Analysis and Efficiency",
                [
                    "The failure analysis should focus on lane-count stability, short visible side lanes, low-score short ground-truth lanes, geometry misses, duplicate-like extra lanes, spurious extra lanes, and GT4/GT5 confusion. These categories are motivated by project diagnostics, but the final manuscript should use active-default train/val diagnostics and official-validation evidence rather than reporting-only final-test breakdowns for selection.",
                    "Efficiency reporting should include parameters, FLOPs or MACs if available, preprocessing time, inference time, decode time, total time, GPU model, precision mode, and batch size. A single average latency number is insufficient because structured decoding and official conversion can be non-negligible. If the model is intended for deployment, the paper should report both network inference and end-to-end lane-output latency.",
                    "The experimental section should close with a brief evidence summary after results are filled in. If the active-default model improves official-validation and final-test behavior, the summary can state the supported gain. If the model is competitive but not superior, the summary should emphasize the structured-output contribution and identify the remaining bottleneck. If the evidence is weak, the paper should not be submitted without further experiments.",
                ],
            ),
        ],
    ),
    (
        "Discussion",
        [
            (
                "What Structured Output Changes",
                [
                    "The main conceptual change in GCS-YOLO-Lane is that the final lane object is represented explicitly. A decoded lane is not a box, a mask fragment, or a curve fitted after segmentation. It is a query-owned sequence of fixed-y points with a lane score and point-level visibility. This makes the output closer to the object required by evaluation and downstream reasoning.",
                    "This representation also changes where errors are exposed. In a mask pipeline, a lane-count error may be hidden inside grouping or fitting. In GCS-YOLO-Lane, count errors appear as query-existence, duplicate-like, spurious-extra, low-score, or visibility failures. These categories can be diagnosed at the training and decoding levels, which makes future improvement more targeted.",
                    "The design does not eliminate all postprocessing. Score thresholds, point-valid thresholds, minimum visible-point rules, NMS, and maximum detections remain part of decoding. The difference is that postprocessing operates on structured lane hypotheses rather than on raw pixels. This is a narrower and more auditable decision surface.",
                ],
            ),
            (
                "Tradeoffs Against Existing Paradigms",
                [
                    "Compared with segmentation methods, GCS-YOLO-Lane reduces dependence on post-hoc instance grouping but gives up the simplicity of dense foreground output. Compared with row/anchor methods, it keeps fixed-y sampling but uses learned queries for lane ownership. Compared with curve methods, it avoids compact global coefficients but stores a longer sequence. Each tradeoff is legitimate; the final value depends on empirical stability.",
                    "The query formulation introduces its own risks. Learned queries must allocate correctly across scenes with variable lane counts. They must suppress spurious extras without missing short true lanes. They must preserve GT4 and GT5 count behavior while still optimizing official Accuracy. These are not minor implementation details. They are central to whether the structured representation is useful in practice.",
                    "The auxiliary mask and edge branches should be interpreted as dense supervision rather than as segmentation outputs. This distinction matters for paper positioning. The method uses dense pixel targets because lane features are thin and benefit from local supervision, but the final lane prediction is query-structured. A reviewer should therefore evaluate the method as a structured detector with auxiliary dense training signals.",
                ],
            ),
            (
                "Research Integrity Boundary",
                [
                    "The project protocol separates validation-driven selection from final-test reporting. This is especially important for lane detection because small changes in confidence threshold, point-validity threshold, NMS distance, `max_det`, and `min_points` can change FP and FN. If these parameters are tuned on final test, the reported result no longer measures generalization. The paper therefore treats final test as a one-shot report after official-validation selection.",
                    "Historical experiment notes can inform discussion but should not override the active contract. Runs based on later or default-disabled mechanisms may be useful diagnostics, but they should not be described as the active mainline unless the corresponding code and configuration are explicitly restored. This manuscript therefore writes experiments as placeholders rather than importing legacy numbers into the main result table.",
                    "The practical implication is that the paper is currently method-complete but evidence-incomplete. The architecture, label contract, output contract, and training design can be described. Performance superiority, ablation causality, and venue-readiness require additional official-validation and final-test evidence. This conservative boundary improves the credibility of the manuscript.",
                ],
            ),
        ],
    ),
    (
        "Limitations and Future Work",
        [
            (
                None,
                [
                    "The first limitation is empirical incompleteness. This manuscript contains the full method description and protocol skeleton, but the active-default official-validation selection, one-shot final-test report, verified baseline table, and ablations remain placeholders. The paper should not be submitted until those values are filled and checked against the project integrity rules.",
                    "The second limitation is TuSimple-centric representation. Fixed-y K56 anchors align naturally with TuSimple h-samples, but other datasets may use different camera geometry, lane curvature, visibility distributions, or evaluation rules. Extending the method to CULane, CurveLanes, LLAMAS, OpenLane-style topology, or 3D lane datasets will require careful representation and metric adaptation [[CITE:xu2020curvelanes,yan2022once3d,wang2023openlanev2]].",
                    "The third limitation is lane-count stability. Query-based detection exposes lane count as a score and allocation problem. Short side lanes, ambiguous GT4 scenes, dense GT5 scenes, duplicate-like queries, and far spurious extras can all affect the final count. Future work should study query allocation, calibrated existence targets, visibility confidence, and training signals that separate false fifth lanes from true dense-lane retention.",
                    "The fourth limitation is temporal and 3D scope. The current branch operates on single 2D images. It does not model temporal consistency, 3D lane geometry, topology, map priors, or lane-line type semantics. These extensions are relevant for deployment but outside the current paper. A future version could use the fixed-y structured head as a 2D component inside temporal or 3D systems.",
                    "Finally, the method adds architectural complexity relative to simple row classifiers. LSEM, LaneBiFPN, a transformer decoder, visibility refinement, and auxiliary supervision all require ablation evidence to justify their cost. The strongest future version of this work will be one that reports not only main accuracy, but also which modules matter, where they fail, and whether the structured representation yields a net benefit under fair comparison.",
                ],
            )
        ],
    ),
    (
        "Conclusion",
        [
            (
                None,
                [
                    "This paper presents GCS-YOLO-Lane, a YOLO11-based structured lane detection network. The key idea is to predict lane instances directly as query-owned fixed-y point sequences with lane existence and point visibility, rather than treating lane detection as ordinary box detection or final mask segmentation. The implemented architecture combines line-sensitive feature enhancement, LaneBiFPN multi-scale fusion, a query-based GCS lane head, Hungarian matching, and visibility-aware supervision.",
                    "The present manuscript is intentionally conservative about results. It gives a complete method and evaluation protocol, but keeps experimental numbers as placeholders until the active default configuration is selected on official validation and evaluated once on final test. Under that boundary, the defensible conclusion is architectural: GCS-YOLO-Lane provides a structured YOLO-style alternative for fixed-y lane instance prediction, and its empirical contribution should be judged after the pending protocol-valid evidence is inserted.",
                ],
            )
        ],
    ),
]


ZH_SECTIONS: list[tuple[str, list[tuple[str | None, list[str]]]]] = [
    (
        "引言",
        [
            (
                "作为结构化几何问题的车道线检测",
                [
                    "车道线检测不是普通的紧凑目标定位问题，而是一个结构化感知问题。一个可用的车道线检测器需要恢复可见的车道实例，保持几何点序，并表达遮挡、磨损、阴影和透视变化下的局部可见性。这样的输出不同于普通目标检测中的矩形框，也不同于最终的前景 mask，因为下游规划模块真正需要的是按顺序组织的车道几何坐标。本文的核心前提是：车道线检测器应当以被下游消费的形式直接预测车道对象。",
                    "已有研究反复说明，车道线这种细长结构需要跨空间的上下文建模。SCNN 的空间消息传递、自注意力蒸馏、循环特征聚合以及关键点全局关联都在处理同一个事实：车道线证据很窄，但在空间上延展 [[CITE:pan2018scnn,hou2019sad,zheng2020resa,wang2022ganet]]。这些工作说明，中间特征不应只依赖局部纹理响应，还应强化线状连续性。GCS-YOLO-Lane 延续这一动机，但把最终表示放在 query 拥有的点序列中。",
                    "本项目采用的 TuSimple 设置进一步强化了结构化输出的必要性。TuSimple 在固定水平采样位置上评估车道预测，因此直接预测 fixed-y 点序列可以减少后期曲线转换和采样转换的歧义 [[CITE:tusimpleBenchmark]]。在当前项目契约中，每条车道由 K=56 个 y 锚点表示，对应原始图像坐标 710, 700, 690, ..., 160，并以原始高度 720 归一化。这个契约不是实现细节，而是模型、标签、loss、解码和 official 评估共同遵守的表示基础。",
                ],
            ),
            (
                "现有流水线中的表示缺口",
                [
                    "分割式车道线检测器学习密集车道证据，这一路线仍然重要，因为密集监督能给细长结构提供强像素级梯度。LaneNet 类实例分割和 affinity-field 方法会预测车道像素或像素关联，再通过聚类、关联分组或拟合恢复实例 [[CITE:neven2018lanenet,abualsaud2021laneaf]]。它们的优势是视觉覆盖充分；代价是最终输出还必须从像素转换为车道实例和采样坐标。",
                    "row-based 和 anchor-based 方法通过预定义结构预测稀疏坐标，缩小了这一表示缺口。Row-wise 分类与 UFLD 把车道定位改写为预定义行位置上的选择问题，从而提升速度并减少繁重后处理 [[CITE:yoo2020e2elmd,qin2020ufld]]。UFLDv2 进一步使用 hybrid anchor driven ordinal classification，说明 anchor 设计可以为车道定位编码有效结构先验 [[CITE:qin2022ufldv2]]。这些方法证明稀疏结构化预测很实用，但车道归属通常仍由 row classifier、anchor 或 proposal refinement 决定。",
                    "query-based、curve-based 和 sequence-based 方法更接近本文追求的结构化预测。LSTR 使用 transformer 推理预测 lane shape，PolyLaneNet 与 BezierLaneNet 通过多项式或 Bezier 表示建模车道几何，FastDraw 则把车道检测看作顺序绘制任务 [[CITE:liu2021lstr,tabelini2020polylanenet,feng2022bezierlane,philion2019fastdraw]]。GCS-YOLO-Lane 位于同一结构化输出谱系中，但保留直接 fixed-y 点序列，并把它与 YOLO11-style 特征提取器结合。",
                ],
            ),
            (
                "GCS-YOLO-Lane 概览",
                [
                    "GCS-YOLO-Lane 将 YOLO11-style 网络改造成结构化车道实例检测器。它保留 YOLO-style backbone 的实用特征提取方式，在中间层插入 line-sensitive enhancement module，通过 LaneBiFPN 融合 P2-P5 多尺度特征，并用 query-based GCS lane head 替代普通检测或 mask 输出。每个 lane query 同时预测车道存在性、fixed-y 锚点上的 x 坐标，以及逐锚点可见性。辅助 mask 和 edge 输出仅作为训练信号保留。",
                    "当前默认输出契约是明确的。对于 batch 大小为 B 的输入，模型输出形状为 B x 12 x 56 x 2 的 `pred_points`、B x 12 的 `pred_logits` 和 B x 12 x 56 的 `pred_valid_logits`。辅助输出包括 B x 2 x H x W 的 `aux_mask_logits` 和 B x 1 x H x W 的 `aux_edge_logits`。在 fixed-y 模式下，y 坐标来自共享 anchor buffer，head 只回归 x。这使每个候选车道具有稳定张量，同时通过 point-validity logits 保留局部可见性。",
                    "该方法围绕两个相关观察设计。第一，车道证据细长且方向性强，因此中间特征应强化线状连续性。第二，车道实例不只是曲线，而是带有局部可见性的有序序列，lane-level existence 与 point-level visibility 必须联合建模。GCS-YOLO-Lane 通过 LSEM、LaneBiFPN、query decoder、Hungarian matching 和 visibility-aware losses 将这些观察落到实现中。",
                ],
            ),
            (
                "贡献与范围",
                [
                    "本文给出四点设计贡献。第一，将 YOLO-style 车道检测器表述为结构化 fixed-y set predictor，使用 Q=12 个 learnable queries 和 K=56 个有序锚点。第二，描述包含 strip responses、direction gating、coordinate reweighting 和 dilated context 的 line-sensitive 特征路径。第三，在 query 解码前使用 LaneBiFPN 融合细节与语义多尺度特征。第四，用 Hungarian matching、点几何、点可见性、曲率/平滑正则以及辅助密集监督训练结构化输出。",
                    "本文对经验结论保持边界。当前请求要求实验结果先用占位符，因此本文不声称 state-of-the-art、不声称优于具体 baseline，也不声称完成的 ablation 增益。实验部分的表格只标出 active-default official-val selection 与 one-shot final-test report 后需要填入的数值。这一边界很重要，因为项目规则禁止 test tuning，并要求所有 threshold、checkpoint 和 postprocess 选择都来自 official-val。",
                    "本文结构如下。第二节将 GCS-YOLO-Lane 放在 segmentation、row/anchor、curve/query 与 YOLO-style driving perception 方法中定位。第三节描述 fixed-y 表示、模型结构、lane head、matching、loss 与 decoding。第四节定义计划中的评估协议并使用占位符。第五、六节讨论结构化设计的意义和局限，第七节给出当前由代码与项目契约支持的结论。",
                ],
            ),
        ],
    ),
    (
        "相关工作",
        [
            (
                "分割式与密集车道表示",
                [
                    "分割式车道线检测把车道标线看作像素级区域或实例嵌入。SCNN 使用跨行列的空间消息传递传播上下文，说明细长车道结构受益于非局部信息 [[CITE:pan2018scnn]]。LaneNet 将车道检测建模为语义分割与实例嵌入 [[CITE:neven2018lanenet]]，LaneAF 使用 affinity fields 将车道像素组合为实例 [[CITE:abualsaud2021laneaf]]。这些方法的重要性在于它们让车道连续性接受密集监督。",
                    "密集表示的限制不在于视觉学习能力弱，而在于最终输出还不是 fixed-sample 评估或规划模块需要的结构化车道对象。像素需要被分组、拟合、排序并转换为采样坐标。在简单场景中转换可以可靠，但在断裂标线、阴影、相邻车道干扰与局部遮挡下，转换层会引入额外启发式决策。GCS-YOLO-Lane 只把 mask 和 edge 分支作为辅助特征监督，最终输出来自结构化 queries。",
                    "密集监督方法也解释了为什么本项目仍保留辅助分支。车道标线很细，坐标监督相对稀疏，辅助 mask 或 edge 监督可以丰富共享特征学习。因此本文并不拒绝 segmentation 信息，而是区分 dense supervision 与 final output：密集图帮助训练网络，解码车道来自 query-owned structured sequences。",
                ],
            ),
            (
                "Row-Based、Anchor-Based 与 Refinement 方法",
                [
                    "Row-based 与 anchor-based 方法很强，因为它们利用了车道基准中的规则几何结构。End-to-end row-wise marker detection 与 UFLD 将车道定位改写为预定义 row 位置上的稀疏分类，从而提升速度并减少后处理 [[CITE:yoo2020e2elmd,qin2020ufld]]。UFLDv2 通过 hybrid anchor driven ordinal classification 扩展这一思路 [[CITE:qin2022ufldv2]]。这些工作是任何 fixed-y 表示必须对照的重要基线。",
                    "Anchor attention 和 cross-layer refinement 方法进一步通过候选车道结构绑定有用特征来提升定位。LaneATT 使用 attention-guided line anchors 实现实时检测 [[CITE:tabelini2021laneatt]]，CLRNet 使用 cross-layer refinement 聚合语义与细节线索 [[CITE:zheng2022clrnet]]。这些设计强调全局上下文与空间精度必须结合。GCS-YOLO-Lane 通过 LaneBiFPN 与 query decoder 满足同一要求，但把车道归属单元从 anchor 改为 learned queries。",
                    "这种区别很重要，因为车道归属方式会影响 count calibration 与 visibility reasoning。在 row 或 anchor 系统中，车道通常由预定义 slot 或 proposal 表示；在 GCS-YOLO-Lane 中，每个 learned query 同时携带 existence、geometry 和 visibility。这并不自动意味着方法更优，但它打开了 query allocation、matching 与 visibility supervision 共同决定性能的设计空间。",
                ],
            ),
            (
                "Query、Curve 与 Sequence 车道预测",
                [
                    "结构化车道预测已通过 transformer queries、curve parameterization、polynomial regression、keypoint association 与 sequence generation 等路线展开。LSTR 使用 transformer-style reasoning 与 set prediction 预测 lane shape [[CITE:liu2021lstr]]。PolyLaneNet 和 BezierLaneNet 使用紧凑曲线参数而不是密集 mask 表示车道 [[CITE:tabelini2020polylanenet,feng2022bezierlane]]。FastDraw 用顺序预测处理长尾车道形态 [[CITE:philion2019fastdraw]]，GANet 则进行关键点全局关联 [[CITE:wang2022ganet]]。这些工作与 GCS-YOLO-Lane 最接近，因为它们都直接预测车道结构。",
                    "较新的 transformer 与 point-based 变体继续推动这一趋势。Laneformer 建模 object-aware row-column relationships [[CITE:han2022laneformer]]，LDTR、Sparse Laneformer、LanePtrNet 和 BezierFormer 分别探索 anchor-chain、sparse-query、point-voting 以及统一 2D/3D lane formulation [[CITE:wang2024ldtr,zhang2024sparselaneformer,li2024laneptrnet,zhang2024bezierformer]]。这些工作说明领域正在超越纯密集分割。本文关注的问题不是结构化预测是否可行，而是如何把 YOLO-style backbone 改造成 fixed-y lane instance predictor。",
                    "GCS-YOLO-Lane 与曲线参数方法的区别在于它保留直接采样点。紧凑曲线优雅且高效，但 fixed-y point sequence 与 TuSimple-style 评估天然对齐，并允许 anchor-level visibility。本文 head 因此在每个 fixed-y anchor 上预测 x，并估计该 anchor 是否可见，避免把全局曲线系数转换成可见采样区间作为主要输出机制。",
                ],
            ),
            (
                "YOLO-Style Driving Perception 与数据集范围",
                [
                    "YOLO-style 网络在实时驾驶感知中广泛使用，因为它们具有高效共享特征提取和易部署推理的优势。YOLOP 与 YOLOPv2 说明 YOLO-like encoder 可以支持多任务 driving perception，包括车道相关输出、可行驶区域和目标检测 [[CITE:wu2022yolop,han2022yolopv2]]。GCS-YOLO-Lane 延续 YOLO-style 特征主体有用这一工程直觉，但把 lane branch 从 segmentation-oriented output 改为 structured query prediction。",
                    "数据集范围也塑造方法设计。TuSimple 使用固定水平采样，因而与 K56 fixed-y 契约匹配 [[CITE:tusimpleBenchmark]]。CULane、CurveLanes、ONCE-3DLanes 与 OpenLane-V2 覆盖更宽的道路布局、弯曲车道、3D 几何或拓扑推理 [[CITE:pan2018scnn,xu2020curvelanes,yan2022once3d,wang2023openlanev2]]。本文聚焦当前 TuSimple 2D fixed-y 分支，跨数据集验证被作为未来工作而不是已支持的经验结论。",
                    "综上，已有工作为 dense supervision、sparse row structures、anchor refinement、curve modeling、transformers 与 YOLO-style driving backbones 提供了充分背景。本文处理的具体缺口更窄：构建一个 YOLO11-based structured detector，使其直接预测 query-owned fixed-y point sequences，并显式建模 point visibility，同时遵守无泄漏 official-validation 协议。",
                ],
            ),
        ],
    ),
    (
        "方法",
        [
            (
                "问题定义与输出契约",
                [
                    "给定一张按项目契约 `--imgsz 544 960` 以 H,W 顺序 resize 的 RGB 道路图像，GCS-YOLO-Lane 预测一组候选车道实例。候选 query 的最大数量为 Q=12。每个 query 拥有 K=56 个有序点、一个 lane-existence logit 和 K 个 point-validity logits。最终解码车道由 query score、point visibility、最小可见点数、可选 lane NMS 和在 official-val 上选择的最大检测数量共同决定。",
                    "输出张量由当前分支契约固定。`pred_points` 的形状为 B x 12 x 56 x 2，`pred_logits` 为 B x 12，`pred_valid_logits` 为 B x 12 x 56。辅助张量包括 B x 2 x H x W 的 `aux_mask_logits` 和 B x 1 x H x W 的 `aux_edge_logits`。这些辅助张量只用于训练监督，不是最终车道表示。",
                    "在 fixed-y 模式中，每个预测点的第二个坐标不是自由回归值，而来自注册的 fixed-y anchor buffer；head 只预测 x 坐标。输出仍然保持 2D points，使下游代码和 loss 可以使用一致的点张量。该设计在不假设每个 anchor 都可见的情况下，为模型提供稳定输出网格。",
                ],
            ),
            (
                "Fixed-Y 标签表示",
                [
                    "fixed-y 表示把每条 TuSimple 标注车道转换为共享 y anchors 上自底向上的点序列。当前分支使用原始图像坐标 710, 700, 690, ..., 160 这 56 个 anchor，并以原始高度 720 归一化。因此 normalized range 为 710/720 到 160/720，K=56。K56 标签必须从原始 TuSimple JSON 与图像重新生成，不能从历史 K32 标签重采样。",
                    "对于每条 GT lane，转换流程会过滤非法坐标，按 bottom-to-top 排序，去除重复 y，并在每个 fixed-y anchor 上插值得到 x。位于有效 lane span 内的 anchor 存储归一化 x 和共享 y；不在可见 span 内的 anchor 保留 fixed y，但通过 `lane_valid=0` 标记为不可见。这样模型只在有证据的位置学习几何，同时仍保持固定大小张量。",
                    "标签文件还包含密集监督数组。每个 `.npz` 标签包括 `semantic_mask`、`edge_mask`、`lanes`、`lane_valid`、`num_lanes`、`point_mode`、`fixed_y`、`raw_file`、`image_shape` 和 `num_points`。这些字段让表示可以被审计：split、anchor order、mask shape、image shape 和 point mode 都可在训练前检查。本文把这些检查视为方法的一部分，因为表示契约是本分支的核心贡献。",
                ],
            ),
            (
                "整体架构",
                [
                    "图 1 总结了整体架构。模型从 YOLO11-style backbone 开始，产生 P2、P3、P4 和 P5 特征图。P3 与 P4 经过 line-sensitive enhancement modules。随后 P2、增强后的 P3、增强后的 P4 和 P5 由 LaneBiFPN 融合为通道对齐的多尺度特征图。GCS lane head 将这些特征图展开为空间 tokens，加入 position 与 level 信息，并解码 learnable lane queries。",
                    "该架构有意保留熟悉的特征提取主体，但改变预测头。标准 YOLO head 通常输出 boxes、class scores 或 segmentation masks；GCS-YOLO-Lane 输出 lane sequences。这不是简单后处理替换，而是改变了训练目标、matching 规则、张量契约和解码接口。辅助 mask/edge 分支支持特征学习，但不定义预测车道对象。",
                    "当前 model YAML 把 LaneBiFPN 接在对应 P2、增强 P3、增强 P4 和 P5 的特征索引上，再接入配置为 `[12, 56, 3, 8, True, fixed_y, 0.9861111111111112, 0.2222222222222222]` 的 `GCSLaneHead`。这编码了主超参数：12 个 queries、56 个 anchors、3 层 decoder、8 个 attention heads、启用 auxiliary outputs，以及 active fixed-y range。",
                ],
            ),
            (
                "Line-Sensitive Feature Enhancement",
                [
                    "line-sensitive enhancement module 插入在 P3 与 P4 之后，因为这两个 stage 在空间细节与语义上下文之间取得平衡。该模块的动机来自车道标线的基本属性：有效信号很细，但在空间上连续。普通卷积响应可以检测局部油漆片段，但车道检测更需要保持沿线状结构的连续性。",
                    "LSEM 包含四个部分。Line-strip attention 使用水平和垂直 strip depthwise convolutions，并通过 direction gate 加权这些响应。Coordinate reweighting 分支沿 height 与 width 调制特征，使空间位置保持显式。Dilated context 分支扩大感受野而不牺牲分辨率。模块输出通过 residual path 回加，使线状增强补充而不是替代 backbone features。",
                    "LSEM 的预期优势是表征层面的，在这份占位结果稿件中还不是经验结论。它应当让中间特征更敏感于长车道证据，但贡献大小必须在 active official-validation protocol 下通过 ablation 报告。因此实验部分给出占位 ablation 行，而不声称已有数值增益。",
                ],
            ),
            (
                "LaneBiFPN 多尺度融合",
                [
                    "车道定位同时需要高分辨率证据和语义辨别能力。P2 与 P3 保留更细空间信息，有利于点回归；P4 与 P5 提供更宽上下文，有助于抑制路面纹理、阴影和类似车道的干扰标记。LaneBiFPN 将这些层投影到共享通道维度，并进行 bidirectional weighted fusion，使 lane head 可以跨尺度注意。",
                    "这一融合设计对应 refinement-based lane detectors 中的实践洞察：只有高层上下文不足以精确定位，只有浅层细节又容易产生视觉歧义。通过把融合后的 P2-P5 特征作为 spatial tokens 提供给 query decoder，模型可以同时利用局部与全局证据。对于局部证据缺失的部分可见车道，这种上下文尤其重要。",
                    "与 LSEM 一样，本文在缺少匹配 ablation 前不声称 LaneBiFPN 提升了性能。方法部分陈述设计和动机，实验计划部分陈述与简单融合的必要比较。这种写法让论文在记录实现的同时，避免把未验证的因果贡献写成事实。",
                ],
            ),
            (
                "Query-Based GCS Lane Head",
                [
                    "GCS lane head 使用 12 个 learnable queries 作为候选车道 slots。每个 query 通过 transformer decoder attend 到多尺度空间 tokens，并产生三个相互关联的预测：lane existence、fixed-y point geometry 与 point visibility。该设计使车道归属显式化。一个 query 不只是一个 score，而是带有几何与可见性的结构化车道假设。",
                    "在 fixed-y 模式中，point predictor 为 56 个 anchors 产生 x logits。query-specific reference-logit buffer 提供不同初始几何形态，image-conditioned refinement 在预测点位置采样特征并 refine x logits。随后 y 坐标从 fixed-y anchor buffer 恢复。head 还会 refine point-valid logits，使模型能够标记每个 anchor 是否可见。",
                    "该输出表示天然支持部分车道。一条车道可以有较高 existence confidence，但只有部分 anchors 可见；相反，unmatched query 可以在 existence 层面被压低，不应解码为最终车道。这个区别对 lane-count stability 很重要，因为 false extra lanes、duplicate-like queries 和 low-score short lanes 是不同失败模式，不能都归结为 mask 质量问题。",
                ],
            ),
            (
                "Matching、Loss 与 Decoding",
                [
                    "训练使用 Hungarian matching 连接预测 queries 与 ground-truth lanes。对于每张图，matcher 构造 Q x N cost matrix，其中 Q 是预测 query 数，N 是当前图中的 GT lane 数。cost 结合 visible-point distance、curve consistency 和 existence confidence。这个 set-matching 步骤避免模型依赖固定 lane index，使 learnable queries 通过分配逐渐专门化。",
                    "匹配后，loss 监督 lane existence、visible-point geometry、point validity、smoothness、curve regularity、auxiliary mask prediction 和 auxiliary edge prediction。当前代码还记录默认关闭的实验项，如 lane-balanced point loss、short-valid recall loss、count losses、duplicate-margin loss 和 spurious-margin loss。本文谨慎处理这些项：除非在冻结实验中显式启用，否则它们不是核心方法贡献。",
                    "推理只解码真实 query predictions。decoder 应用 query score threshold、point-validity threshold、longest contiguous visible span selection、minimum visible-point requirement、基于共享 anchors 平均 x 距离的可选 lane NMS，以及 `max_det`。项目契约要求 decode 不使用 ground truth、不伪造 lanes，并在需要时按底部可见 x 从左到右排序。所有 threshold 与 postprocess 设置必须在 official-val 上选择，而不是在 final test 上选择。",
                ],
            ),
        ],
    ),
    (
        "实验",
        [
            (
                "数据集与协议",
                [
                    "本节被有意写成协议完整的占位实验部分。当前 active dataset 是 TuSimple fixed-y K56 conversion，包含 3263 张训练图像、363 张验证图像和 2782 张测试图像。输入尺寸固定为 `--imgsz 544 960`，顺序为 H,W。模型与数据配置分别为 `ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml` 和 `data/tusimple_gcs_fixed_y_960x544.yaml`。",
                    "Official TuSimple 指标包括 Accuracy、FP 和 FN。项目还可以报告内部 `official_score`、lane-count accuracy、per-count accuracy 和 timing 字段，但必须与 official metric 区分。Threshold、checkpoint choice、NMS distance、maximum lane count、minimum visible points 和 point-validity threshold 都必须在 official-val surface 上选择。Final test 只用于已选择 candidate 的一次报告。",
                    "该协议也是论文贡献的一部分，因为它防止 leakage。论文不得在 test 上调 threshold，不得在 inference 或 decode 中使用 GT，不得伪造 lanes，不得静默修改 official metrics，也不得在没有 official-val evidence 的情况下声称提升。任何违反这些规则的结果表都应排除在主论文之外，只能作为不可比历史记录。",
                ],
            ),
            (
                "实现细节",
                [
                    "默认正式训练 recipe 应在 active-default run 冻结后从真实 `args.yaml` 复制。预期设置使用远端 RTX 4090 环境、`ssh_lane` conda 环境、batch 32 作为起点、AdamW 优化、cosine learning-rate decay、从原始 TuSimple JSON 和图像重新生成的 K56 fixed-y labels，以及默认 model/data 路径。如果 batch size、AMP 状态、workers 或 preprocessing 不同，论文必须报告真实 `args.yaml`，而不是模板命令。",
                    "表 2 中的结果占位符只能在 active-default official-validation selection 完成后填入。论文应报告 selected row、selected decode parameters，以及使用完全相同参数的一次 final-test report。如果某个实验启用了默认关闭的 loss knob，表格必须把它命名为 experiment，而不是 mainline method。",
                ],
            ),
            (
                "主结果与基线",
                [
                    "主比较表应覆盖 segmentation、row/anchor、refinement、query、curve 与 sequence 方法。候选基线包括 SCNN、SAD、UFLD、UFLDv2、LaneATT、CLRNet、LSTR、PolyLaneNet、BezierLaneNet 以及相关 structured methods [[CITE:pan2018scnn,hou2019sad,qin2020ufld,qin2022ufldv2,tabelini2021laneatt,zheng2022clrnet,liu2021lstr,tabelini2020polylanenet,feng2022bezierlane]]。每一行都必须说明 metric 来自原论文、复现还是项目重新训练。",
                    "由于当前请求要求使用占位符，本文不声称排名。表格列应包括 Accuracy、FP、FN、可比时的 runtime 或 FPS、input resolution 以及 protocol notes。硬件相关速度数值不能无 caveat 混合。CULane、CurveLanes、LLAMAS、OpenLane 或 3D lane benchmark 的指标不应放入同一个 TuSimple comparison table。",
                    "Ablation studies 应直接对应方法主张。必要 ablation 包括移除 LSEM、用简单融合替换 LaneBiFPN、禁用 fixed-y image-conditioned refinement、禁用 point-validity branch，以及移除 auxiliary mask/edge supervision。可选 count 或 margin losses 只有在显式启用并通过 official-val 选择时，才应放入单独 experiment block。",
                ],
            ),
            (
                "失败分析与效率",
                [
                    "Failure analysis 应聚焦 lane-count stability、short visible side lanes、low-score short ground-truth lanes、geometry misses、duplicate-like extra lanes、spurious extra lanes 和 GT4/GT5 confusion。这些类别由项目诊断动机支持，但最终论文应使用 active-default train/val diagnostics 与 official-val evidence，而不是用 reporting-only final-test breakdown 做选择。",
                    "效率报告应包括参数量、FLOPs 或 MACs（如可得）、preprocessing time、inference time、decode time、total time、GPU 型号、precision mode 和 batch size。单一平均 latency 不足够，因为 structured decoding 与 official conversion 可能不可忽略。如果模型面向部署，论文应同时报告 network inference 与 end-to-end lane-output latency。",
                    "实验节应在填入结果后给出简短 evidence summary。如果 active-default model 在 official-val 与 final-test 上均表现更好，summary 可以陈述受支持的提升。如果模型只是竞争性但不占优，summary 应强调 structured-output contribution 并指出剩余 bottleneck。如果证据较弱，则应在补充实验前暂缓投稿。",
                ],
            ),
        ],
    ),
    (
        "讨论",
        [
            (
                "结构化输出改变了什么",
                [
                    "GCS-YOLO-Lane 的主要概念变化是显式表示最终车道对象。解码车道不是 box、mask fragment 或 segmentation 后拟合出的曲线，而是 query-owned fixed-y points sequence，并具有 lane score 与 point-level visibility。这使输出更接近评估与下游 reasoning 所需的对象。",
                    "这种表示也改变了错误暴露方式。在 mask pipeline 中，lane-count error 可能隐藏在 grouping 或 fitting 内；在 GCS-YOLO-Lane 中，count error 会表现为 query-existence、duplicate-like、spurious-extra、low-score 或 visibility failure。这些类别可以在训练和解码层面被诊断，从而让后续改进更有针对性。",
                    "该设计并不消除所有 postprocessing。Score thresholds、point-valid thresholds、minimum visible-point rules、NMS 和 maximum detections 仍是 decoding 的一部分。区别在于 postprocessing 操作的是结构化 lane hypotheses，而不是原始像素。这是更窄且更可审计的决策面。",
                ],
            ),
            (
                "相对现有范式的取舍",
                [
                    "与 segmentation methods 相比，GCS-YOLO-Lane 降低了对后验实例分组的依赖，但放弃了 dense foreground output 的简单性。与 row/anchor methods 相比，它保留 fixed-y sampling，但用 learned queries 组织 lane ownership。与 curve methods 相比，它避免紧凑全局系数，但存储更长序列。每种取舍都合理，最终价值取决于经验稳定性。",
                    "query formulation 也引入自身风险。Learned queries 必须在可变车道数场景中正确分配；必须在不漏掉 true short lanes 的情况下抑制 spurious extras；必须在优化 official Accuracy 的同时保持 GT4 与 GT5 count behavior。这些不是小实现细节，而是结构化表示是否实用的核心问题。",
                    "辅助 mask 和 edge 分支应被理解为 dense supervision，而不是 segmentation outputs。这一区分对论文定位很关键。方法使用密集像素目标，因为车道特征细长且受益于局部监督，但最终车道预测是 query-structured。审稿人因此应将该方法视为带有辅助密集训练信号的 structured detector。",
                ],
            ),
            (
                "研究诚信边界",
                [
                    "项目协议把 validation-driven selection 与 final-test reporting 分开。对车道检测尤其如此，因为 confidence threshold、point-validity threshold、NMS distance、`max_det` 和 `min_points` 的小变化都可能改变 FP 和 FN。如果这些参数在 final test 上调优，报告结果就不再测量泛化。因此本文把 final test 作为 official-val selection 之后的一次性报告。",
                    "历史实验记录可以为讨论提供背景，但不应覆盖 active contract。基于较晚机制或默认关闭机制的 runs 可能有诊断价值，但除非对应代码与配置被显式恢复，否则不能写成 active mainline。因此本文把实验结果写成占位符，而不是把 legacy numbers 导入主结果表。",
                    "实际含义是：当前论文 method-complete 但 evidence-incomplete。架构、标签契约、输出契约和训练设计可以完整描述。性能优势、ablation causality 和 venue-readiness 仍需要 official-val 与 final-test evidence。这一保守边界会提升稿件可信度。",
                ],
            ),
        ],
    ),
    (
        "局限与未来工作",
        [
            (
                None,
                [
                    "第一项局限是经验证据尚未完成。本文包含完整方法描述和协议骨架，但 active-default official-validation selection、one-shot final-test report、verified baseline table 与 ablations 仍是占位符。在这些数值填入并按项目 integrity rules 检查前，论文不应投稿。",
                    "第二项局限是表示以 TuSimple 为中心。Fixed-y K56 anchors 与 TuSimple h-samples 天然对齐，但其他数据集可能具有不同 camera geometry、lane curvature、visibility distributions 或 evaluation rules。将方法扩展到 CULane、CurveLanes、LLAMAS、OpenLane-style topology 或 3D lane datasets 时，需要谨慎调整表示和指标 [[CITE:xu2020curvelanes,yan2022once3d,wang2023openlanev2]]。",
                    "第三项局限是 lane-count stability。Query-based detection 将车道数暴露为 score 与 allocation 问题。Short side lanes、ambiguous GT4 scenes、dense GT5 scenes、duplicate-like queries 与 far spurious extras 都会影响最终 count。未来工作应研究 query allocation、calibrated existence targets、visibility confidence 以及能区分 false fifth lanes 与 true dense-lane retention 的训练信号。",
                    "第四项局限是时间与 3D 范围。当前分支只处理单帧 2D 图像，不建模 temporal consistency、3D lane geometry、topology、map priors 或 lane-line type semantics。这些扩展对部署重要，但超出本文范围。未来版本可以把 fixed-y structured head 作为 temporal 或 3D systems 中的 2D component。",
                    "最后，与简单 row classifier 相比，该方法增加了架构复杂度。LSEM、LaneBiFPN、transformer decoder、visibility refinement 与 auxiliary supervision 都需要 ablation evidence 来证明成本合理。最强版本的后续工作应不仅报告 main accuracy，还要报告哪些模块真正有用、它们在哪里失败，以及 structured representation 是否在公平比较下带来净收益。",
                ],
            )
        ],
    ),
    (
        "结论",
        [
            (
                None,
                [
                    "本文提出 GCS-YOLO-Lane，一种 YOLO11-based structured lane detection network。核心思想是直接把车道实例预测为 query-owned fixed-y point sequences，并同时输出 lane existence 与 point visibility，而不是把车道检测当作普通 box detection 或最终 mask segmentation。该实现结合 line-sensitive feature enhancement、LaneBiFPN multi-scale fusion、query-based GCS lane head、Hungarian matching 与 visibility-aware supervision。",
                    "当前稿件对结果保持保守。它给出完整方法与评估协议，但在 active default configuration 通过 official validation 选择并在 final test 上一次性报告前，所有实验数值都保持占位符。在这一边界下，目前可辩护的结论是架构性的：GCS-YOLO-Lane 为 fixed-y lane instance prediction 提供了一种结构化 YOLO-style 替代方案，其经验贡献应在后续填入 protocol-valid evidence 后再判断。",
                ],
            )
        ],
    ),
]


TABLES_EN = [
    (
        "Table 1. Active Contract of GCS-YOLO-Lane",
        ["Item", "Value"],
        [
            ["Input size", "`--imgsz 544 960` in H,W order"],
            ["Default model", "`ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`"],
            ["Default data", "`data/tusimple_gcs_fixed_y_960x544.yaml`"],
            ["Lane queries", "Q = 12"],
            ["Fixed-y anchors", "K = 56, y = 710, 700, ..., 160"],
            ["Final output", "`pred_points`, `pred_logits`, `pred_valid_logits`"],
            ["Auxiliary outputs", "`aux_mask_logits`, `aux_edge_logits` for training supervision"],
        ],
    ),
    (
        "Table 2. Main TuSimple Results to Be Filled After Active-Default Evaluation",
        ["Split", "Selected decode", "Accuracy", "FP", "FN", "Count acc.", "Status"],
        [
            ["Official-val", "[CONF / PVT / NMS / MAX_DET / MIN_POINTS]", "[VAL_ACC]", "[VAL_FP]", "[VAL_FN]", "[VAL_COUNT_ACC]", "Pending"],
            ["Official test", "Same as official-val selected decode", "[TEST_ACC]", "[TEST_FP]", "[TEST_FN]", "[TEST_COUNT_ACC]", "Pending one-shot report"],
        ],
    ),
    (
        "Table 3. Required Ablation Placeholders",
        ["Variant", "Purpose", "Accuracy", "FP", "FN", "Interpretation"],
        [
            ["Full GCS-YOLO-Lane", "Active default", "[TBD]", "[TBD]", "[TBD]", "Reference row"],
            ["Without LSEM", "Test line-sensitive enhancement", "[TBD]", "[TBD]", "[TBD]", "Pending"],
            ["Simpler feature fusion", "Test LaneBiFPN", "[TBD]", "[TBD]", "[TBD]", "Pending"],
            ["Without visibility branch", "Test point-valid prediction", "[TBD]", "[TBD]", "[TBD]", "Pending"],
            ["Without mask/edge aux.", "Test dense supervision", "[TBD]", "[TBD]", "[TBD]", "Pending"],
        ],
    ),
    (
        "Table 4. Failure Analysis Checklist",
        ["Failure family", "Diagnostic evidence to collect", "Selection rule"],
        [
            ["Low-score short lane", "Train/val failure trace and official-val miss analysis", "Do not use final test for threshold search"],
            ["Geometry miss", "Matched APE and visible-IoU buckets", "Use official-val for promotion"],
            ["Duplicate-like extra", "Query-pair distance and visibility overlap", "Compare before/after only on val"],
            ["Spurious extra", "Extra lane with low GT overlap", "Reject if ACC gate regresses"],
            ["GT4/GT5 count confusion", "Per-count confusion tables", "Report final test only once"],
        ],
    ),
]

TABLES_ZH = [
    (
        "表 1. GCS-YOLO-Lane 当前有效契约",
        ["项目", "取值"],
        [
            ["输入尺寸", "`--imgsz 544 960`，顺序为 H,W"],
            ["默认模型", "`ultralytics/cfg/models/gcs/gcs-yolo-lane-s.yaml`"],
            ["默认数据", "`data/tusimple_gcs_fixed_y_960x544.yaml`"],
            ["车道 query", "Q = 12"],
            ["Fixed-y 锚点", "K = 56, y = 710, 700, ..., 160"],
            ["最终输出", "`pred_points`, `pred_logits`, `pred_valid_logits`"],
            ["辅助输出", "`aux_mask_logits`, `aux_edge_logits`，仅用于训练监督"],
        ],
    ),
    (
        "表 2. Active-default 评估后待填入的 TuSimple 主结果",
        ["Split", "Selected decode", "Accuracy", "FP", "FN", "Count acc.", "状态"],
        [
            ["Official-val", "[CONF / PVT / NMS / MAX_DET / MIN_POINTS]", "[VAL_ACC]", "[VAL_FP]", "[VAL_FN]", "[VAL_COUNT_ACC]", "待完成"],
            ["Official test", "与 official-val selected decode 完全一致", "[TEST_ACC]", "[TEST_FP]", "[TEST_FN]", "[TEST_COUNT_ACC]", "待一次性报告"],
        ],
    ),
    (
        "表 3. 必要 ablation 占位符",
        ["Variant", "目的", "Accuracy", "FP", "FN", "解释"],
        [
            ["Full GCS-YOLO-Lane", "Active default", "[TBD]", "[TBD]", "[TBD]", "Reference row"],
            ["Without LSEM", "检验 line-sensitive enhancement", "[TBD]", "[TBD]", "[TBD]", "待完成"],
            ["Simpler feature fusion", "检验 LaneBiFPN", "[TBD]", "[TBD]", "[TBD]", "待完成"],
            ["Without visibility branch", "检验 point-valid prediction", "[TBD]", "[TBD]", "[TBD]", "待完成"],
            ["Without mask/edge aux.", "检验 dense supervision", "[TBD]", "[TBD]", "[TBD]", "待完成"],
        ],
    ),
    (
        "表 4. Failure analysis 检查清单",
        ["失败类型", "需要收集的诊断证据", "选择规则"],
        [
            ["低分短车道", "train/val failure trace 与 official-val miss analysis", "不得用 final test 搜 threshold"],
            ["几何漏检", "matched APE 与 visible-IoU 分桶", "只用 official-val 做 promotion"],
            ["duplicate-like extra", "query-pair distance 与 visibility overlap", "只在 val 上做前后比较"],
            ["spurious extra", "与 GT overlap 低的额外车道", "ACC gate 回退则拒绝"],
            ["GT4/GT5 count confusion", "per-count confusion tables", "final test 只报告一次"],
        ],
    ),
]


FIGURES_EN = [
    ("figure_01_architecture.png", "GCS-YOLO-Lane architecture. The YOLO11-style backbone is retained, while the terminal output is a query-owned structured lane set."),
    ("figure_02_fixed_y.png", "Fixed-y K56 representation. The y anchors are shared across lanes; the model predicts x coordinates and point visibility."),
    ("figure_03_head.png", "Query-based GCS lane head. Each query predicts lane existence, fixed-y points, and point-level visibility."),
    ("figure_04_protocol.png", "Training and evaluation protocol. Official validation selects the decode; final test is used only once for reporting."),
]

FIGURES_ZH = [
    ("figure_01_architecture.png", "GCS-YOLO-Lane 架构。模型保留 YOLO11-style backbone，但最终输出是 query-owned structured lane set。"),
    ("figure_02_fixed_y.png", "Fixed-y K56 表示。所有车道共享 y anchors；模型预测 x 坐标和 point visibility。"),
    ("figure_03_head.png", "Query-based GCS lane head。每个 query 同时预测 lane existence、fixed-y points 和 point-level visibility。"),
    ("figure_04_protocol.png", "训练与评估协议。Official validation 选择 decode；final test 仅一次性报告。"),
]


def abstract_en() -> str:
    return (
        "Lane detection requires instance-level geometric reasoning over thin, elongated, and partially visible road markings. "
        "Conventional object detectors do not directly express ordered lane geometry, while segmentation-oriented pipelines often require grouping or fitting before a lane can be evaluated as sampled coordinates. "
        "This paper presents GCS-YOLO-Lane, a YOLO11-based structured lane detection network that predicts lane instances as query-owned fixed-y point sequences. "
        "The model combines a YOLO11-style backbone, line-sensitive feature enhancement, LaneBiFPN multi-scale fusion, and a query-based GCS lane head that jointly estimates lane existence, x coordinates on K=56 fixed-y anchors, and point-level visibility for Q=12 candidate lanes. "
        "Training uses Hungarian assignment between lane queries and ground-truth lanes, with supervision over existence, visible-point geometry, point visibility, curve and smoothness regularization, and auxiliary mask/edge maps. "
        "The experimental section is intentionally left with placeholders until the active default configuration is selected on official validation and evaluated once on the TuSimple final test. "
        "Under this evidence boundary, the paper's current claim is architectural: GCS-YOLO-Lane provides a structured YOLO-style alternative to final-mask lane outputs by directly predicting visibility-aware fixed-y lane instance sequences."
    )


def abstract_zh() -> str:
    return (
        "车道线检测需要对细长、连续且常常局部可见的道路标线进行实例级几何推理。普通目标检测器不能直接表达有序车道几何，而分割式流水线通常还需要 grouping 或 fitting，才能把像素结果转换为可评估的采样坐标。"
        "本文提出 GCS-YOLO-Lane，一种 YOLO11-based structured lane detection network，它把车道实例预测为 query-owned fixed-y point sequences。"
        "模型结合 YOLO11-style backbone、line-sensitive feature enhancement、LaneBiFPN multi-scale fusion，以及 query-based GCS lane head；该 head 为 Q=12 条候选车道联合预测 lane existence、K=56 个 fixed-y anchors 上的 x 坐标和 point-level visibility。"
        "训练采用 Hungarian assignment 匹配 lane queries 与 ground-truth lanes，并对 existence、visible-point geometry、point visibility、curve/smoothness regularization 以及 auxiliary mask/edge maps 进行监督。"
        "实验部分按当前请求保留占位符，只有在 active default configuration 通过 official validation 选择并在 TuSimple final test 上一次性评估后才填入结果。"
        "在这一证据边界下，本文当前可主张的是架构贡献：GCS-YOLO-Lane 通过直接预测 visibility-aware fixed-y lane instance sequences，为 final-mask lane outputs 提供了一种结构化 YOLO-style 替代方案。"
    )


def expand_section(section_data, mode: str) -> str:
    lines: list[str] = []
    for section_title, subs in section_data:
        lines.append(f"## {section_title}")
        for subsection, paragraphs in subs:
            if subsection:
                lines.append(f"### {subsection}")
            for para in paragraphs:
                lines.append(render_cites(para))
            if section_title in {"Method", "方法"} and subsection in {"Overall Architecture", "整体架构"}:
                lines.append("[FIGURE:0]")
            if section_title in {"Method", "方法"} and subsection in {"Fixed-Y Label Representation", "Fixed-Y 标签表示"}:
                lines.append("[FIGURE:1]")
            if section_title in {"Method", "方法"} and subsection in {"Query-Based GCS Lane Head", "Query-Based GCS Lane Head"}:
                lines.append("[FIGURE:2]")
            if section_title in {"Experiments", "实验"} and subsection in {"Dataset and Protocol", "数据集与协议"}:
                lines.append("[FIGURE:3]")
                for table_idx in range(4):
                    lines.append(f"[TABLE:{table_idx}]")
    return "\n\n".join(lines)


def markdown_cell(text: str) -> str:
    return render_cites(text).replace("|", r"\|").replace("\n", "<br>")


def markdown_table(caption: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [
        f"**{caption}**",
        "",
        "| " + " | ".join(markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


FIGURE_DISPLAY_NUMBERS = {0: 2, 1: 1, 2: 3, 3: 4}


def markdown_figure(idx: int, lang: str) -> str:
    figures = FIGURES_EN if lang == "en" else FIGURES_ZH
    fn, caption = figures[idx]
    figure_number = FIGURE_DISPLAY_NUMBERS[idx]
    label = f"Figure {figure_number}" if lang == "en" else f"图 {figure_number}"
    full_caption = f"{label}. {caption}"
    return f"![{full_caption}](figures/{fn})\n\n*{full_caption}*"


def render_markdown_placeholders(text: str, lang: str) -> str:
    tables = TABLES_EN if lang == "en" else TABLES_ZH

    def replace_figure(match: re.Match[str]) -> str:
        return markdown_figure(int(match.group(1)), lang)

    def replace_table(match: re.Match[str]) -> str:
        caption, header, rows = tables[int(match.group(1))]
        return markdown_table(caption, header, rows)

    text = re.sub(r"\[FIGURE:(\d+)\]", replace_figure, text)
    text = re.sub(r"\[TABLE:(\d+)\]", replace_table, text)
    return text


def build_markdown(lang: str) -> str:
    title = "GCS-YOLO-Lane: A YOLO11-Based Structured Lane Detection Network"
    if lang == "en":
        sections = expand_section(EN_SECTIONS, "pdf")
        abstract = abstract_en()
        keywords = "lane detection; structured prediction; YOLO11; fixed-y representation; query-based detection; TuSimple"
        declarations = [
            "Data Availability: The manuscript is based on the TuSimple lane detection benchmark and a project-local fixed-y K56 conversion. Exact release terms and dataset access links must be completed before submission.",
            "Code Availability: Code release status is pending author decision. The current project paths are listed for reproducibility but should not be treated as public release URLs.",
            "Ethics Declaration: No human-subject experiment is introduced by this manuscript. Dataset use and autonomous-driving data terms should be verified before submission.",
            "Conflict of Interest: To be completed by the author(s).",
            "Funding: To be completed by the author(s).",
            "Author Contributions: To be completed using CRediT roles after the author list is finalized.",
            "AI Disclosure: This manuscript draft was prepared with AI assistance under author direction. The author(s) remain responsible for factual accuracy, citation verification, experiments, and final claims.",
        ]
        ref_heading = "References"
    else:
        sections = expand_section(ZH_SECTIONS, "pdf")
        abstract = abstract_zh()
        keywords = "车道线检测；结构化预测；YOLO11；fixed-y 表示；query-based detection；TuSimple"
        declarations = [
            "数据可用性：本文基于 TuSimple lane detection benchmark 和项目本地 fixed-y K56 转换。投稿前需要补充确切数据访问链接和授权表述。",
            "代码可用性：代码发布状态待作者决定。文中列出的项目路径用于复现实验，不应视为公开 release URL。",
            "伦理声明：本文不引入人体实验。投稿前仍需核对数据集使用条款和自动驾驶数据相关要求。",
            "利益冲突：待作者确认。",
            "资助：待作者确认。",
            "作者贡献：作者列表确定后按 CRediT roles 补充。",
            "AI 使用披露：本文初稿在作者指令下由 AI 辅助准备。作者仍需对事实准确性、引用核验、实验和最终结论负责。",
        ]
        ref_heading = "参考文献"
    sections = render_markdown_placeholders(sections, lang)
    lines = [
        f"# {title}",
        "",
        "Author information: to be completed",
        "",
        "## Abstract" if lang == "en" else "## 摘要",
        "",
        abstract,
        "",
        "**Keywords:** " + keywords if lang == "en" else "**关键词：**" + keywords,
        "",
        sections,
        "",
        "## Declarations" if lang == "en" else "## 声明",
        "",
    ]
    for item in declarations:
        lines.append("- " + item)
    lines.extend(["", f"## {ref_heading}", ""])
    for i, entry in enumerate(BIB_ENTRIES, 1):
        url = f" {entry.url}" if entry.url else ""
        lines.append(f"[{i}] {entry.author}. {entry.title}. {entry.venue}, {entry.year}.{url}")
    return "\n".join(lines) + "\n"


def make_table_story(caption: str, header: list[str], rows: list[list[str]], styles, lang: str):
    data = [[Paragraph(markup(cell, lang), styles["TableHead"]) for cell in header]]
    for row in rows:
        data.append([Paragraph(markup(cell, lang), styles["TableCell"]) for cell in row])
    width = 16.8 * cm
    col_widths = [width / len(header)] * len(header)
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#98a2b3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return [Paragraph(markup(caption, lang), styles["Caption"]), table, Spacer(1, 8)]


def try_register_font(name: str, path: str) -> bool:
    try:
        pdfmetrics.registerFont(TTFont(name, path))
        return True
    except Exception:
        return False


def register_fonts() -> None:
    try_register_font("TimesNewRoman", "C:/Windows/Fonts/times.ttf")
    try_register_font("TimesNewRoman-Bold", "C:/Windows/Fonts/timesbd.ttf")
    try_register_font("SimSun", "C:/Windows/Fonts/simsun.ttc")
    try_register_font("SimHei", "C:/Windows/Fonts/simhei.ttf")
    try_register_font("NotoSerifSC", "C:/Windows/Fonts/NotoSerifSC-VF.ttf")


def make_styles(lang: str):
    styles = getSampleStyleSheet()
    if lang == "en":
        body_font = "TimesNewRoman" if "TimesNewRoman" in pdfmetrics.getRegisteredFontNames() else "Times-Roman"
        bold_font = "TimesNewRoman-Bold" if "TimesNewRoman-Bold" in pdfmetrics.getRegisteredFontNames() else "Times-Bold"
        body_size, leading = 10.8, 17.0
    else:
        body_font = "SimSun" if "SimSun" in pdfmetrics.getRegisteredFontNames() else "NotoSerifSC"
        if body_font not in pdfmetrics.getRegisteredFontNames():
            body_font = "Helvetica"
        bold_font = "SimHei" if "SimHei" in pdfmetrics.getRegisteredFontNames() else body_font
        body_size, leading = 11.0, 20.6
    body_alignment = TA_JUSTIFY if lang == "en" else TA_LEFT
    styles.add(ParagraphStyle("PaperTitle", parent=styles["Title"], fontName=bold_font, fontSize=18, leading=24, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle("Meta", parent=styles["Normal"], fontName=body_font, fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#475467")))
    styles.add(ParagraphStyle("Heading", parent=styles["Heading1"], fontName=bold_font, fontSize=13.5, leading=18, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle("Subheading", parent=styles["Heading2"], fontName=bold_font, fontSize=11.5, leading=16, spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle("Body", parent=styles["BodyText"], fontName=body_font, fontSize=body_size, leading=leading, alignment=body_alignment, firstLineIndent=0.45 * cm, spaceAfter=6))
    styles.add(ParagraphStyle("Abstract", parent=styles["Body"], firstLineIndent=0, fontSize=body_size, leading=leading))
    styles.add(ParagraphStyle("Caption", parent=styles["BodyText"], fontName=body_font, fontSize=8.5, leading=11.5 if lang == "en" else 14, alignment=TA_LEFT, spaceBefore=4, spaceAfter=5))
    styles.add(ParagraphStyle("TableHead", parent=styles["BodyText"], fontName=bold_font, fontSize=7.8, leading=10.5 if lang == "en" else 13))
    styles.add(ParagraphStyle("TableCell", parent=styles["BodyText"], fontName=body_font, fontSize=7.6, leading=10 if lang == "en" else 13))
    styles.add(ParagraphStyle("Reference", parent=styles["BodyText"], fontName=body_font, fontSize=8.3, leading=10.5 if lang == "en" else 13, leftIndent=0.45 * cm, firstLineIndent=-0.45 * cm, spaceAfter=3))
    return styles


def markup(text: str, lang: str) -> str:
    text = html.escape(render_cites(text))
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = text.replace("\n", "<br/>")
    return text


def build_story(lang: str):
    tables = TABLES_EN if lang == "en" else TABLES_ZH
    figures = FIGURES_EN if lang == "en" else FIGURES_ZH
    section_data = EN_SECTIONS if lang == "en" else ZH_SECTIONS
    styles = make_styles(lang)
    story: list = []
    title = "GCS-YOLO-Lane: A YOLO11-Based Structured Lane Detection Network"
    story.append(Paragraph(markup(title, lang), styles["PaperTitle"]))
    story.append(Paragraph(markup("Author information: to be completed", lang), styles["Meta"]))
    story.append(Paragraph(markup("Structured lane detection manuscript with placeholder experiments", lang), styles["Meta"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Abstract" if lang == "en" else "摘要", styles["Heading"]))
    story.append(Paragraph(markup(abstract_en() if lang == "en" else abstract_zh(), lang), styles["Abstract"]))
    story.append(Paragraph(markup(("Keywords: lane detection; structured prediction; YOLO11; fixed-y representation; query-based detection; TuSimple") if lang == "en" else "关键词：车道线检测；结构化预测；YOLO11；fixed-y 表示；query-based detection；TuSimple", lang), styles["Abstract"]))
    fig_count = 0
    for section_title, subs in section_data:
        story.append(Paragraph(markup(section_title, lang), styles["Heading"]))
        for subsection, paragraphs in subs:
            if subsection:
                story.append(Paragraph(markup(subsection, lang), styles["Subheading"]))
            for para in paragraphs:
                story.append(Paragraph(markup(para, lang), styles["Body"]))
            if section_title in {"Method", "方法"} and subsection in {"Overall Architecture", "整体架构"}:
                add_figure(story, FIG_DIR / figures[0][0], f"Figure 2. {figures[0][1]}" if lang == "en" else f"图 2. {figures[0][1]}", styles, lang)
                fig_count += 1
            if section_title in {"Method", "方法"} and subsection in {"Fixed-Y Label Representation", "Fixed-Y 标签表示"}:
                add_figure(story, FIG_DIR / figures[1][0], f"Figure 1. {figures[1][1]}" if lang == "en" else f"图 1. {figures[1][1]}", styles, lang)
                fig_count += 1
            if section_title in {"Method", "方法"} and subsection in {"Query-Based GCS Lane Head", "Query-Based GCS Lane Head"}:
                add_figure(story, FIG_DIR / figures[2][0], f"Figure 3. {figures[2][1]}" if lang == "en" else f"图 3. {figures[2][1]}", styles, lang)
                fig_count += 1
            if section_title in {"Experiments", "实验"} and subsection in {"Dataset and Protocol", "数据集与协议"}:
                add_figure(story, FIG_DIR / figures[3][0], f"Figure 4. {figures[3][1]}" if lang == "en" else f"图 4. {figures[3][1]}", styles, lang)
                for caption, header, rows in tables:
                    story.extend(make_table_story(caption, header, rows, styles, lang))
    story.append(Paragraph("Declarations" if lang == "en" else "声明", styles["Heading"]))
    declarations = [
        "Data Availability: The manuscript is based on the TuSimple lane detection benchmark and a project-local fixed-y K56 conversion. Exact release terms and dataset access links must be completed before submission.",
        "Code Availability: Code release status is pending author decision. The current project paths are listed for reproducibility but should not be treated as public release URLs.",
        "Ethics Declaration: No human-subject experiment is introduced by this manuscript. Dataset use and autonomous-driving data terms should be verified before submission.",
        "Conflict of Interest: To be completed by the author(s).",
        "Funding: To be completed by the author(s).",
        "Author Contributions: To be completed using CRediT roles after the author list is finalized.",
        "AI Disclosure: This manuscript draft was prepared with AI assistance under author direction. The author(s) remain responsible for factual accuracy, citation verification, experiments, and final claims.",
    ]
    if lang == "zh":
        declarations = [
            "数据可用性：本文基于 TuSimple lane detection benchmark 和项目本地 fixed-y K56 转换。投稿前需要补充确切数据访问链接和授权表述。",
            "代码可用性：代码发布状态待作者决定。文中列出的项目路径用于复现实验，不应视为公开 release URL。",
            "伦理声明：本文不引入人体实验。投稿前仍需核对数据集使用条款和自动驾驶数据相关要求。",
            "利益冲突：待作者确认。",
            "资助：待作者确认。",
            "作者贡献：作者列表确定后按 CRediT roles 补充。",
            "AI 使用披露：本文初稿在作者指令下由 AI 辅助准备。作者仍需对事实准确性、引用核验、实验和最终结论负责。",
        ]
    for item in declarations:
        story.append(Paragraph(markup(item, lang), styles["Body"]))
    story.append(PageBreak())
    story.append(Paragraph("References" if lang == "en" else "参考文献", styles["Heading"]))
    for i, entry in enumerate(BIB_ENTRIES, 1):
        ref = f"[{i}] {entry.author}. {entry.title}. {entry.venue}, {entry.year}."
        if entry.url:
            ref += f" {entry.url}"
        story.append(Paragraph(markup(ref, lang), styles["Reference"]))
    return story


def add_figure(story: list, path: Path, caption: str, styles, lang: str) -> None:
    probe = RLImage(str(path))
    width = 15.4 * cm
    ratio = probe.imageHeight / probe.imageWidth
    height = width * ratio
    max_height = 8.9 * cm
    if height > max_height:
        height = max_height
        width = height / ratio
    img = RLImage(str(path), width=width, height=height)
    story.append(Spacer(1, 4))
    story.append(img)
    story.append(Paragraph(markup(caption, lang), styles["Caption"]))
    story.append(Spacer(1, 8))


def draw_page(canvas, doc, lang: str):
    canvas.saveState()
    font = "TimesNewRoman" if lang == "en" and "TimesNewRoman" in pdfmetrics.getRegisteredFontNames() else ("SimSun" if "SimSun" in pdfmetrics.getRegisteredFontNames() else "Helvetica")
    canvas.setFont(font, 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(doc.leftMargin, A4[1] - 1.15 * cm, "GCS-YOLO-Lane manuscript draft")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.85 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(lang: str, filename: str) -> None:
    story = build_story(lang)
    doc = SimpleDocTemplate(
        str(OUT / filename),
        pagesize=A4,
        rightMargin=1.85 * cm,
        leftMargin=1.85 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.55 * cm,
        title="GCS-YOLO-Lane manuscript",
        author="GCS-YOLO-Lane project",
        subject="Structured lane detection manuscript with placeholder experiments",
    )
    doc.build(story, onFirstPage=lambda c, d: draw_page(c, d, lang), onLaterPages=lambda c, d: draw_page(c, d, lang))


def bibtex() -> str:
    chunks = []
    for entry in BIB_ENTRIES:
        fields = {
            "author": entry.author,
            "title": entry.title,
            "year": entry.year,
        }
        if entry.entry_type == "article":
            fields["journal"] = entry.venue
        elif entry.entry_type == "inproceedings":
            fields["booktitle"] = entry.venue
        else:
            fields["howpublished"] = entry.venue
        if entry.url:
            fields["url"] = entry.url
        if entry.note:
            fields["note"] = entry.note
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items())
        chunks.append(f"@{entry.entry_type}{{{entry.key},\n{body}\n}}\n")
    return "\n".join(chunks)


def latex_table(caption: str, header: list[str], rows: list[list[str]], lang: str) -> str:
    cols = "p{0.22\\linewidth}" + "p{0.24\\linewidth}" * (len(header) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{" + esc_tex(caption) + r"}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + cols + r"}",
        r"\toprule",
        " & ".join(esc_tex(h) for h in header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(esc_tex(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def latex_figure(idx: int, lang: str) -> str:
    figures = FIGURES_EN if lang == "en" else FIGURES_ZH
    fn, cap = figures[idx]
    figure_number = FIGURE_DISPLAY_NUMBERS[idx]
    label = f"fig:{figure_number}"
    caption = f"Figure {figure_number}. {cap}" if lang == "en" else f"图 {figure_number}. {cap}"
    return "\n".join(
        [
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=0.95\linewidth]{{figures/{Path(fn).stem}.png}}",
            r"\caption{" + esc_tex(caption) + r"}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )


def build_latex(lang: str) -> str:
    is_en = lang == "en"
    title = "GCS-YOLO-Lane: A YOLO11-Based Structured Lane Detection Network"
    preamble = [
        r"\documentclass[11pt,a4paper]{article}" if is_en else r"\documentclass[11pt,a4paper]{ctexart}",
        r"\usepackage[margin=2.2cm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{float}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\usepackage[numbers,sort&compress]{natbib}",
        r"\usepackage{setspace}",
        r"\onehalfspacing",
    ]
    if is_en:
        preamble.extend([r"\usepackage{newtxtext,newtxmath}"])
    else:
        preamble.extend(
            [
                r"\usepackage{fontspec}",
                r"\usepackage{xeCJK}",
                r"\setmainfont{Times New Roman}",
                r"\setCJKmainfont{SimSun}",
                r"\setCJKsansfont{SimHei}",
            ]
        )
    preamble.extend(
        [
            r"\title{" + esc_tex(title) + r"}",
            r"\author{Author information: to be completed}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
            esc_tex(abstract_en() if is_en else abstract_zh()),
            r"\end{abstract}",
            r"\noindent\textbf{" + ("Keywords" if is_en else "关键词") + r":} "
            + esc_tex("lane detection; structured prediction; YOLO11; fixed-y representation; query-based detection; TuSimple" if is_en else "车道线检测；结构化预测；YOLO11；fixed-y 表示；query-based detection；TuSimple"),
        ]
    )
    body: list[str] = []
    section_data = EN_SECTIONS if is_en else ZH_SECTIONS
    tables = TABLES_EN if is_en else TABLES_ZH
    for section_title, subs in section_data:
        body.append(r"\section{" + esc_tex(section_title) + r"}")
        for subsection, paragraphs in subs:
            if subsection:
                body.append(r"\subsection{" + esc_tex(subsection) + r"}")
            for para in paragraphs:
                body.append(esc_tex(para) + "\n")
            if section_title in {"Method", "方法"} and subsection in {"Overall Architecture", "整体架构"}:
                body.append(latex_figure(0, lang))
            if section_title in {"Method", "方法"} and subsection in {"Fixed-Y Label Representation", "Fixed-Y 标签表示"}:
                body.append(latex_figure(1, lang))
            if section_title in {"Method", "方法"} and subsection in {"Query-Based GCS Lane Head", "Query-Based GCS Lane Head"}:
                body.append(latex_figure(2, lang))
            if section_title in {"Experiments", "实验"} and subsection in {"Dataset and Protocol", "数据集与协议"}:
                body.append(latex_figure(3, lang))
                for caption, header, rows in tables:
                    body.append(latex_table(caption, header, rows, lang))
    declarations = [
        "Data Availability: The manuscript is based on the TuSimple lane detection benchmark and a project-local fixed-y K56 conversion. Exact release terms and dataset access links must be completed before submission.",
        "Code Availability: Code release status is pending author decision. The current project paths are listed for reproducibility but should not be treated as public release URLs.",
        "Ethics Declaration: No human-subject experiment is introduced by this manuscript. Dataset use and autonomous-driving data terms should be verified before submission.",
        "Conflict of Interest: To be completed by the author(s).",
        "Funding: To be completed by the author(s).",
        "Author Contributions: To be completed using CRediT roles after the author list is finalized.",
        "AI Disclosure: This manuscript draft was prepared with AI assistance under author direction. The author(s) remain responsible for factual accuracy, citation verification, experiments, and final claims.",
    ]
    if not is_en:
        declarations = [
            "数据可用性：本文基于 TuSimple lane detection benchmark 和项目本地 fixed-y K56 转换。投稿前需要补充确切数据访问链接和授权表述。",
            "代码可用性：代码发布状态待作者决定。文中列出的项目路径用于复现实验，不应视为公开 release URL。",
            "伦理声明：本文不引入人体实验。投稿前仍需核对数据集使用条款和自动驾驶数据相关要求。",
            "利益冲突：待作者确认。",
            "资助：待作者确认。",
            "作者贡献：作者列表确定后按 CRediT roles 补充。",
            "AI 使用披露：本文初稿在作者指令下由 AI 辅助准备。作者仍需对事实准确性、引用核验、实验和最终结论负责。",
        ]
    body.append(r"\section{" + ("Declarations" if is_en else "声明") + r"}")
    for dec in declarations:
        body.append(esc_tex(dec) + "\n")
    body.extend([r"\bibliographystyle{IEEEtran}", r"\bibliography{references}", r"\end{document}"])
    return "\n\n".join(preamble + body) + "\n"


def copy_figure_aliases() -> None:
    for fn, _ in FIGURES_EN:
        src = FIG_DIR / fn
        alias = FIG_DIR / f"{Path(fn).stem}.png"
        if src != alias:
            alias.write_bytes(src.read_bytes())


def main() -> None:
    ensure_dirs()
    register_fonts()
    make_figures()
    copy_figure_aliases()
    (OUT / "references.bib").write_text(bibtex(), encoding="utf-8")
    (OUT / "gcs_yolo_lane_paper_en.md").write_text(build_markdown("en"), encoding="utf-8")
    (OUT / "gcs_yolo_lane_paper_zh.md").write_text(build_markdown("zh"), encoding="utf-8")
    (OUT / "gcs_yolo_lane_paper_en.tex").write_text(build_latex("en"), encoding="utf-8")
    (OUT / "gcs_yolo_lane_paper_zh.tex").write_text(build_latex("zh"), encoding="utf-8")
    build_pdf("en", "gcs_yolo_lane_paper_en.pdf")
    build_pdf("zh", "gcs_yolo_lane_paper_zh.pdf")
    print(f"Generated manuscript package at {OUT}")


if __name__ == "__main__":
    main()
