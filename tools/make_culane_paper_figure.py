from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CASES = [
    {
        "category": "Crowded",
        "output_image": "images/test/driver_100_30frame__05251517_0433.MP4__01140.jpg",
        "panel": "(a)",
        "description": "Vehicle and pedestrian occlusion",
        "caption": (
            "(a) Crowded. Dense traffic and pedestrian activity partially occlude "
            "the lane boundaries."
        ),
    },
    {
        "category": "Dazzle",
        "output_image": "images/test/driver_100_30frame__05251548_0439.MP4__00960.jpg",
        "panel": "(b)",
        "description": "Strong illumination and glare",
        "caption": (
            "(b) Dazzle. Strong sunlight and glare reduce the visibility of lane "
            "markings."
        ),
    },
    {
        "category": "Shadow",
        "output_image": "images/test/driver_100_30frame__05251517_0433.MP4__01590.jpg",
        "panel": "(c)",
        "description": "Cast shadows on the roadway",
        "caption": (
            "(c) Shadow. Cast shadows and severe illumination changes make lane "
            "markings difficult to distinguish."
        ),
    },
    {
        "category": "Curve",
        "output_image": "images/test/driver_100_30frame__05250541_0314.MP4__04140.jpg",
        "panel": "(d)",
        "description": "Large road curvature",
        "caption": (
            "(d) Curve. Large road curvature changes the apparent lane geometry "
            "and challenges long-range localization."
        ),
    },
]

LANE_COLORS = [
    (228, 26, 28),
    (55, 126, 184),
    (77, 175, 74),
    (152, 78, 163),
    (255, 127, 0),
    (166, 86, 40),
]

CAPTION = (
    "Figure X. Representative challenging scenarios in the CULane dataset. "
    "Different lane instances are marked with distinct colors. The examples "
    "illustrate vehicle and pedestrian occlusion, strong illumination, cast "
    "shadows, and large road curvature."
)

CAPTION_ZH = (
    "图1. CULane 数据集中的车道检测困难场景示例。不同车道采用不同颜色进行标注。"
    "受车辆和行人遮挡、强光眩光、树荫造成的光照变化以及道路大曲率等复杂条件影响，"
    "部分场景中的车道线仅保留少量甚至没有可用于车道检测的视觉线索。"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create a paper-style 2x2 CULane hard-case figure from real images and GT labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "datasets" / "culane_test_eval",
        help="Converted CULane dataset root containing images/test.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path(r"D:\BaiduNetdiskDownload\CULane"),
        help="Original CULane root containing raw images and .lines.txt files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "datasets" / "culane_test_eval" / "manifests" / "test.json",
        help="CULane test manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "culane_paper_hard_cases",
        help="Directory for the generated figure and metadata.",
    )
    parser.add_argument("--panel-width", type=int, default=1200)
    parser.add_argument("--lane-width", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\Arial Bold.ttf"),
    ) if bold else (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\Arial.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def parse_lines(path: Path) -> list[np.ndarray]:
    lanes: list[np.ndarray] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        values = [float(value) for value in raw_line.split()]
        if len(values) < 4 or len(values) % 2:
            continue
        points = np.asarray(list(zip(values[0::2], values[1::2])), dtype=np.float32)
        if len(points) >= 2:
            lanes.append(points)
    return lanes


def interpolate_lane(points: np.ndarray, samples: int = 120) -> np.ndarray:
    order = np.argsort(points[:, 1])
    points = points[order]
    y = points[:, 1]
    x = points[:, 0]
    unique_y, unique_indices = np.unique(y, return_index=True)
    unique_x = x[unique_indices]
    if len(unique_y) < 2:
        return points
    sampled_y = np.linspace(float(unique_y[0]), float(unique_y[-1]), samples, dtype=np.float32)
    sampled_x = np.interp(sampled_y, unique_y, unique_x).astype(np.float32)
    return np.column_stack((sampled_x, sampled_y))


def draw_lane_gt(
    image: np.ndarray,
    lanes: list[np.ndarray],
    raw_shape: tuple[int, int] = (590, 1640),
    thickness: int = 5,
) -> None:
    display_h, display_w = image.shape[:2]
    raw_h, raw_w = raw_shape
    scale = np.asarray([display_w / raw_w, display_h / raw_h], dtype=np.float32)
    for lane_index, lane in enumerate(lanes):
        points = interpolate_lane(lane) * scale
        points[:, 0] = np.clip(points[:, 0], 0, display_w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, display_h - 1)
        points_i = np.rint(points).astype(np.int32)
        color_rgb = LANE_COLORS[lane_index % len(LANE_COLORS)]
        color_bgr = tuple(reversed(color_rgb))
        cv2.polylines(
            image,
            [points_i],
            isClosed=False,
            color=(18, 18, 18),
            thickness=thickness + 4,
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(
            image,
            [points_i],
            isClosed=False,
            color=color_bgr,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def rounded_label(
    canvas: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    left, top = xy
    bbox = draw.textbbox((left, top), text, font=font)
    padding_x, padding_y = 14, 8
    draw.rounded_rectangle(
        (
            bbox[0] - padding_x,
            bbox[1] - padding_y,
            bbox[2] + padding_x,
            bbox[3] + padding_y,
        ),
        radius=6,
        fill=(255, 255, 255, 220),
        outline=(30, 30, 30, 220),
        width=2,
    )
    draw.text((left, top), text, font=font, fill=(20, 20, 20, 255))


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def save_individual_panel(
    panel: Image.Image,
    case: dict[str, Any],
    output_dir: Path,
    panel_width: int,
    panel_height: int,
    caption_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    label_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    dpi: int,
) -> dict[str, str]:
    """Save one panel with its own paper-style caption below the image."""
    margin_x = 48
    margin_top = 48
    caption_gap = 34
    line_height = 40
    line_gap = 12
    caption_text = str(case["caption"])
    probe = Image.new("RGB", (panel_width, 100), "white")
    probe_draw = ImageDraw.Draw(probe)
    caption_lines = wrap_text(probe_draw, caption_text, caption_font, panel_width)
    caption_height = len(caption_lines) * line_height + max(0, len(caption_lines) - 1) * line_gap
    canvas = Image.new(
        "RGB",
        (
            margin_x * 2 + panel_width,
            margin_top + panel_height + caption_gap + caption_height + 32,
        ),
        "white",
    )
    x = margin_x
    y = margin_top
    canvas.paste(panel, (x, y))
    ImageDraw.Draw(canvas).rectangle(
        (x, y, x + panel_width - 1, y + panel_height - 1),
        outline=(35, 35, 35),
        width=2,
    )
    rounded_label(canvas, (x + 18, y + 16), f"{case['panel']} {case['category']}", label_font)

    draw = ImageDraw.Draw(canvas)
    caption_y = y + panel_height + caption_gap
    for line in caption_lines:
        draw.text((margin_x, caption_y), line, font=caption_font, fill=(28, 28, 28))
        caption_y += line_height + line_gap

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"culane_{str(case['category']).lower()}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    canvas.save(png_path, dpi=(int(dpi), int(dpi)))
    canvas.save(pdf_path, "PDF", resolution=float(dpi))
    return {
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "caption": caption_text,
    }


def save_line_only_image(
    panel: Image.Image,
    case: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> str:
    """Save the image and colored GT lanes without any labels, borders, or captions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"culane_{str(case['category']).lower()}.png"
    panel.save(path, dpi=(int(dpi), int(dpi)))
    return str(path.resolve())


def create_figure(args: argparse.Namespace) -> dict[str, Any]:
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_output = {str(record["output_image"]): record for record in records}
    panel_width = int(args.panel_width)
    panel_height = round(panel_width * 384 / 960)
    gap = 32
    margin_x = 48
    margin_top = 48
    caption_gap = 34
    caption_font = load_font(30)
    label_font = load_font(32, bold=True)
    caption_line_gap = 12
    caption_max_width = panel_width * 2 + gap
    caption_probe = Image.new("RGB", (caption_max_width, 100), "white")
    caption_draw = ImageDraw.Draw(caption_probe)
    caption_lines = wrap_text(caption_draw, CAPTION, caption_font, caption_max_width)
    caption_height = len(caption_lines) * 40 + max(0, len(caption_lines) - 1) * caption_line_gap
    figure_width = margin_x * 2 + panel_width * 2 + gap
    figure_height = margin_top + panel_height * 2 + gap + caption_gap + caption_height + 36
    canvas = Image.new("RGB", (figure_width, figure_height), "white")
    metadata_cases: list[dict[str, Any]] = []
    individual_results: list[dict[str, str]] = []
    line_only_results: list[str] = []

    for index, case in enumerate(DEFAULT_CASES):
        record = by_output.get(case["output_image"])
        if record is None:
            raise KeyError(f"Missing case in manifest: {case['output_image']}")
        image_path = args.dataset_root / record["output_image"]
        raw_lines_path = args.archive_root / record["raw_lines"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        lanes = parse_lines(raw_lines_path)
        if image.shape[:2] != (384, 960):
            raise ValueError(f"Unexpected CULane display shape {image.shape[:2]} for {image_path}")
        draw_lane_gt(image, lanes, thickness=int(args.lane_width))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        panel = Image.fromarray(image).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        x = margin_x + (index % 2) * (panel_width + gap)
        y = margin_top + (index // 2) * (panel_height + gap)
        canvas.paste(panel, (x, y))
        ImageDraw.Draw(canvas).rectangle(
            (x, y, x + panel_width - 1, y + panel_height - 1),
            outline=(35, 35, 35),
            width=2,
        )
        rounded_label(canvas, (x + 18, y + 16), f"{case['panel']} {case['category']}", label_font)
        individual_result = save_individual_panel(
            panel=panel,
            case=case,
            output_dir=args.output_dir / "individual",
            panel_width=panel_width,
            panel_height=panel_height,
            caption_font=caption_font,
            label_font=label_font,
            dpi=int(args.dpi),
        )
        individual_results.append(individual_result)
        line_only_results.append(
            save_line_only_image(
                panel=panel,
                case=case,
                output_dir=args.output_dir / "line_only",
                dpi=int(args.dpi),
            )
        )
        metadata_cases.append(
            {
                "panel": case["panel"],
                "category": case["category"],
                "description": case["description"],
                "caption": case["caption"],
                "individual_png": individual_result["png"],
                "individual_pdf": individual_result["pdf"],
                "line_only_png": line_only_results[-1],
                "display_image": str(image_path.resolve()),
                "raw_file": str((args.archive_root / record["raw_file"]).resolve()),
                "raw_lines": str(raw_lines_path.resolve()),
                "num_lanes_manifest": int(record.get("num_lanes", 0)),
                "num_lanes_rendered": len(lanes),
            }
        )

    draw = ImageDraw.Draw(canvas)
    caption_y = margin_top + panel_height * 2 + gap + caption_gap
    for line in caption_lines:
        draw.text((margin_x, caption_y), line, font=caption_font, fill=(28, 28, 28))
        caption_y += 40 + caption_line_gap

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "culane_hard_cases_2x2.png"
    pdf_path = args.output_dir / "culane_hard_cases_2x2.pdf"
    metadata_path = args.output_dir / "culane_hard_cases_2x2_metadata.json"
    captions_path = args.output_dir / "individual_captions.txt"
    caption_zh_path = args.output_dir / "caption_zh.txt"
    canvas.save(png_path, dpi=(int(args.dpi), int(args.dpi)))
    canvas.save(pdf_path, "PDF", resolution=float(args.dpi))
    captions_path.write_text(
        "\n\n".join(result["caption"] for result in individual_results) + "\n",
        encoding="utf-8",
    )
    caption_zh_path.write_text(CAPTION_ZH + "\n", encoding="utf-8")
    metadata = {
        "figure": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "individual_captions": str(captions_path.resolve()),
        "caption_zh": str(caption_zh_path.resolve()),
        "individual_panels": individual_results,
        "line_only_images": line_only_results,
        "caption": CAPTION,
        "caption_zh_text": CAPTION_ZH,
        "dataset": "CULane",
        "split": "test",
        "display_shape_hw": [384, 960],
        "raw_shape_hw": [590, 1640],
        "lane_source": "original CULane .lines.txt ground-truth annotations",
        "cases": metadata_cases,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "individual_captions": str(captions_path.resolve()),
        "individual_panels": individual_results,
        "line_only_images": line_only_results,
        "caption_zh": str(caption_zh_path.resolve()),
        "size": list(canvas.size),
        "cases": metadata_cases,
    }


if __name__ == "__main__":
    result = create_figure(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=True))
