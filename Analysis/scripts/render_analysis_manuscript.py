"""Render the current Markdown manuscript to an A4 layout proof, without analysis."""
from pathlib import Path
import html
import json
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from pypdf import PdfReader
import fitz

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Analysis/03_Analysis_3페이지용_압축본.md'
OUTPUT = ROOT / 'Analysis/03_Analysis_압축_배치확인.pdf'


def inline(text):
    # Local references appear with their readable labels in the printed proof.
    text = re.sub(r'\[([^\]]+)\]\((?:<[^>]+>|[^)]+)\)', r'\1', text)
    text = html.escape(text.replace('−', '-'))
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text


def main():
    pdfmetrics.registerFont(TTFont('Malgun', 'C:/Windows/Fonts/malgun.ttf'))
    pdfmetrics.registerFont(TTFont('MalgunBold', 'C:/Windows/Fonts/malgunbd.ttf'))
    pdfmetrics.registerFontFamily('Malgun', normal='Malgun', bold='MalgunBold', italic='Malgun', boldItalic='MalgunBold')
    body = ParagraphStyle('body', fontName='Malgun', fontSize=11, leading=17.6,
                          wordWrap='CJK', alignment=TA_LEFT, spaceAfter=7)
    title = ParagraphStyle('title', parent=body, fontName='MalgunBold', fontSize=16,
                           leading=22, spaceAfter=10, keepWithNext=True)
    heading = ParagraphStyle('heading', parent=body, fontName='MalgunBold', fontSize=12.5,
                             leading=19, spaceBefore=8, spaceAfter=7, keepWithNext=True)
    caption = ParagraphStyle('caption', parent=body, fontSize=9, leading=13.5, spaceAfter=7)
    reference = ParagraphStyle('reference', parent=caption, textColor=colors.HexColor('#475569'))
    tabletext = ParagraphStyle('tabletext', parent=body, fontSize=10, leading=14, spaceAfter=0)
    story = []
    lines = SOURCE.read_text(encoding='utf-8-sig').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), title))
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), heading))
        elif line.startswith('!['):
            target = re.search(r'\]\(([^)]+)\)', line).group(1).strip('<>')
            image_path = SOURCE.parent / target
            from PIL import Image as PILImage
            with PILImage.open(image_path) as image:
                width = 170*mm if 'summary' in target else 150*mm
                height = width*image.height/image.width
            story.append(Image(str(image_path), width=width, height=height))
            story.append(Spacer(1, 2*mm))
        elif line.startswith('|'):
            rows = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip())
                i += 1
            cells = [[Paragraph(inline(cell.strip()), tabletext) for cell in row.strip('|').split('|')]
                     for row in rows if not re.match(r'^\|[\s:|\-]+\|$', row)]
            table = Table(cells, colWidths=[45*mm, 55*mm, 70*mm], hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EDF3F5')),
                ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#78909C')),
                ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#D4DEE3')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.extend([table, Spacer(1, 2*mm)])
        elif line.startswith('*') and line.endswith('*') and not line.startswith('**'):
            story.append(Paragraph(inline(line[1:-1]), caption))
        elif line.startswith(('관련 Python:', '이미지·생성 코드:')):
            story.append(Paragraph(inline(line), reference))
        else:
            story.append(Paragraph(inline(line), body))

    page_bottoms = {}
    class LayoutProof(SimpleDocTemplate):
        def afterFlowable(self, flowable):
            if not isinstance(flowable, Spacer):
                page_bottoms[self.page] = round(self.frame._y/mm, 1)

    def footer(canvas, doc):
        canvas.setFont('Malgun', 8)
        canvas.setFillColor(colors.HexColor('#64748B'))
        canvas.drawRightString(A4[0]-20*mm, 12*mm, f'Analysis | {doc.page}')

    doc = LayoutProof(str(OUTPUT), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                      topMargin=18*mm, bottomMargin=18*mm, title='Analysis 압축 원고', author='')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(OUTPUT).pages)
    review = ROOT / 'tmp/pdfs/analysis_compact_review'
    review.mkdir(parents=True, exist_ok=True)
    with fitz.open(OUTPUT) as pdf:
        for n, page in enumerate(pdf, 1):
            page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False).save(review / f'page_{n}.png')
    occupied_last = round((A4[1]/mm-18-page_bottoms[pages])/(A4[1]/mm-36), 3)
    print(json.dumps({'pages': pages, 'font_size': 11, 'leading': 17.6,
                      'margins_mm': {'sides': 20, 'top_bottom': 18},
                      'last_page_used_fraction': occupied_last,
                      'page_bottoms_mm': page_bottoms, 'pdf': str(OUTPUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

