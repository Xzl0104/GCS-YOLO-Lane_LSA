from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


def inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="PaperTitle",
            parent=styles["Title"],
            fontName="Times-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PaperMeta",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PaperBullet",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            leftIndent=12,
            firstLineIndent=0,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PaperCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            leftIndent=6,
            rightIndent=6,
            spaceBefore=4,
            spaceAfter=8,
            backColor=colors.HexColor("#F4F4F4"),
        )
    )
    return styles


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = lines[i].strip().strip("|")
        cells = [cell.strip() for cell in row.split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            rows.append(cells)
        i += 1
    return rows, i


def table_flowable(rows: list[list[str]], styles) -> Table:
    data = [[Paragraph(inline_markup(cell), styles["Small"]) for cell in row] for row in rows]
    ncols = max(len(row) for row in rows)
    width = 17.2 * cm
    col_widths = [width / ncols for _ in range(ncols)]
    tbl = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Times-Roman", 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDEDED")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0B0B0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tbl


def flush_paragraph(buf: list[str], story: list, styles):
    if not buf:
        return
    text = " ".join(part.strip() for part in buf).strip()
    if text:
        story.append(Paragraph(inline_markup(text), styles["Body"]))
    buf.clear()


def markdown_to_story(markdown: str, styles) -> list:
    lines = markdown.splitlines()
    story: list = []
    paragraph: list[str] = []
    i = 0
    in_code = False
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["PaperCode"]))
                code_lines.clear()
                in_code = False
            else:
                flush_paragraph(paragraph, story, styles)
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            flush_paragraph(paragraph, story, styles)
            i += 1
            continue

        if stripped == "---":
            flush_paragraph(paragraph, story, styles)
            story.append(Spacer(1, 8))
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph(paragraph, story, styles)
            rows, next_i = parse_table(lines, i)
            if rows:
                story.append(table_flowable(rows, styles))
                story.append(Spacer(1, 8))
            i = next_i
            continue

        if stripped.startswith("# "):
            flush_paragraph(paragraph, story, styles)
            story.append(Paragraph(inline_markup(stripped[2:].strip()), styles["PaperTitle"]))
            story.append(Paragraph("Author information: to be completed", styles["PaperMeta"]))
            story.append(Paragraph("GCS-YOLO-Lane project manuscript draft", styles["PaperMeta"]))
            story.append(Spacer(1, 10))
            i += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph(paragraph, story, styles)
            if stripped.lower().startswith("## references"):
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(stripped[3:].strip()), styles["Section"]))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph(paragraph, story, styles)
            story.append(Paragraph(inline_markup(stripped[4:].strip()), styles["Subsection"]))
            i += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph(paragraph, story, styles)
            while i < len(lines) and lines[i].strip().startswith("- "):
                item_text = lines[i].strip()[2:].strip()
                story.append(Paragraph("- " + inline_markup(item_text), styles["PaperBullet"]))
                i += 1
            story.append(Spacer(1, 4))
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph(paragraph, story, styles)
    return story


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(doc.leftMargin, A4[1] - 1.2 * cm, "GCS-YOLO-Lane formatted draft")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.9 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(input_md: Path, output_pdf: Path):
    styles = make_styles()
    markdown = input_md.read_text(encoding="utf-8")
    story = markdown_to_story(markdown, styles)
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=1.9 * cm,
        leftMargin=1.9 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.7 * cm,
        title="GCS-YOLO-Lane Structured Lane Detection Draft",
        author="GCS-YOLO-Lane project",
        subject="Structured fixed-y lane detection under TuSimple protocol",
    )
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)


def main():
    parser = argparse.ArgumentParser(description="Generate the formatted GCS-YOLO-Lane paper draft PDF.")
    parser.add_argument("--input-md", type=Path, default=Path("output/pdf/gcs_yolo_lane_paper_draft.md"))
    parser.add_argument("--output-pdf", type=Path, default=Path("output/pdf/gcs_yolo_lane_paper_draft.pdf"))
    args = parser.parse_args()
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(args.input_md, args.output_pdf)
    print(args.output_pdf)


if __name__ == "__main__":
    main()
