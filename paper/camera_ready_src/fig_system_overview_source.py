from pathlib import Path
import math

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "fig_system_overview.pdf"
W, H = 900, 342

pdfmetrics.registerFont(TTFont("Serif", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"))
pdfmetrics.registerFont(TTFont("Serif-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))

C = {
    "ink": (0.05, 0.06, 0.07),
    "muted": (0.30, 0.32, 0.34),
    "line": (0.18, 0.20, 0.22),
    "soft": (0.74, 0.76, 0.78),
    "blue": (0.16, 0.42, 0.70),
    "blue_fill": (0.89, 0.94, 0.98),
    "yellow": (1.00, 0.96, 0.68),
    "yellow_fill": (1.00, 0.98, 0.84),
    "peach": (0.98, 0.87, 0.80),
    "green": (0.82, 0.93, 0.78),
    "lav": (0.90, 0.86, 0.95),
    "orange": (0.90, 0.27, 0.06),
    "panel": (0.985, 0.985, 0.985),
    "paper": (1.0, 1.0, 1.0),
}


def yy(y):
    return H - y


def text(c, s, x, y, size=9, bold=False, center=False, right=False,
         color=C["ink"], serif=False):
    family = "Serif" if serif else "Sans"
    c.setFont(f"{family}-Bold" if bold else family, size)
    c.setFillColorRGB(*color)
    if center:
        c.drawCentredString(x, yy(y), s)
    elif right:
        c.drawRightString(x, yy(y), s)
    else:
        c.drawString(x, yy(y), s)


def box(c, x0, y0, x1, y1, fill=C["paper"], stroke=C["line"],
        width=1.0, dash=None, radius=0):
    c.saveState()
    if fill is not None:
        c.setFillColorRGB(*fill)
    if stroke is not None:
        c.setStrokeColorRGB(*stroke)
    c.setLineWidth(width)
    if dash:
        c.setDash(*dash)
    if radius:
        c.roundRect(x0, H - y1, x1 - x0, y1 - y0, radius,
                    stroke=int(stroke is not None), fill=int(fill is not None))
    else:
        c.rect(x0, H - y1, x1 - x0, y1 - y0,
               stroke=int(stroke is not None), fill=int(fill is not None))
    c.restoreState()


def line(c, x0, y0, x1, y1, color=C["line"], width=1.1, dash=None):
    c.saveState()
    c.setStrokeColorRGB(*color)
    c.setLineWidth(width)
    if dash:
        c.setDash(*dash)
    c.line(x0, yy(y0), x1, yy(y1))
    c.restoreState()


def arrow(c, x0, y0, x1, y1, color=C["line"], width=1.3, dash=None, head=6):
    line(c, x0, y0, x1, y1, color, width, dash)
    a = math.atan2(y1 - y0, x1 - x0)
    bx = x1 - head * math.cos(a)
    by = y1 - head * math.sin(a)
    half = head * 0.42
    px = half * math.sin(a)
    py = -half * math.cos(a)
    c.saveState()
    c.setFillColorRGB(*color)
    p = c.beginPath()
    p.moveTo(x1, yy(y1))
    p.lineTo(bx + px, yy(by + py))
    p.lineTo(bx - px, yy(by - py))
    p.close()
    c.drawPath(p, stroke=0, fill=1)
    c.restoreState()


def poly(c, pts, fill, stroke=C["line"], width=1.2):
    c.saveState()
    c.setFillColorRGB(*fill)
    c.setStrokeColorRGB(*stroke)
    c.setLineWidth(width)
    p = c.beginPath()
    p.moveTo(pts[0][0], yy(pts[0][1]))
    for x, y in pts[1:]:
        p.lineTo(x, yy(y))
    p.close()
    c.drawPath(p, stroke=1, fill=1)
    c.restoreState()


def snowflake(c, x, y, r=8, color=(0.16, 0.60, 0.95), width=1.4):
    for a in (0, math.pi / 3, 2 * math.pi / 3):
        dx, dy = r * math.cos(a), r * math.sin(a)
        line(c, x - dx, y - dy, x + dx, y + dy, color, width)
        for sign in (-1, 1):
            ex, ey = x + sign * dx, y + sign * dy
            base = a + (0 if sign == 1 else math.pi)
            for off in (-0.55, 0.55):
                bx = ex - 3.6 * math.cos(base + off)
                by = ey - 3.6 * math.sin(base + off)
                line(c, ex, ey, bx, by, color, width * 0.75)


def section_tag(c, x, y, label):
    box(c, x, y, x + 32, y + 14, fill=C["paper"], stroke=C["orange"],
        width=1.0, radius=2)
    text(c, label, x + 16, y + 10.2, 7.3, bold=True, center=True, color=C["orange"])


def road_frame(c, x, y, w=48, h=36, car_color=(0.95, 0.65, 0.12)):
    box(c, x, y, x + w, y + h, fill=(0.88, 0.91, 0.92), stroke=C["line"], width=0.8)
    # Perspective road and lane markings.
    poly(c, [(x + 4, y + h), (x + w * 0.42, y + 7), (x + w * 0.60, y + 7),
             (x + w - 4, y + h)], fill=(0.28, 0.30, 0.32), stroke=(0.28, 0.30, 0.32), width=0.3)
    line(c, x + w * 0.51, y + 11, x + w * 0.50, y + h - 3, C["paper"], 1.0, (3, 3))
    box(c, x + w * 0.33, y + h * 0.58, x + w * 0.45, y + h * 0.76,
        fill=car_color, stroke=C["line"], width=0.6, radius=1)


def question_card(c, x, y, w=102, h=40):
    box(c, x, y, x + w, y + h, fill=C["paper"], stroke=C["line"], width=0.9)
    text(c, "Q", x + 12, y + 17, 12, bold=True, color=C["blue"], serif=True)
    line(c, x + 28, y + 12, x + w - 8, y + 12, C["soft"], 1.2)
    line(c, x + 28, y + 21, x + w - 18, y + 21, C["soft"], 1.2)
    line(c, x + 10, y + 31, x + w - 28, y + 31, C["soft"], 1.2)


def stack(c, x, y, n=4, w=70, h=11):
    colors = [C["blue_fill"], C["peach"], C["green"], C["lav"]]
    for i in range(n):
        box(c, x + i * 3, y + i * 8, x + w + i * 3, y + h + i * 8,
            fill=colors[i % len(colors)], stroke=C["line"], width=0.7, radius=2)


def pill(c, x0, y0, x1, y1, label, fill):
    box(c, x0, y0, x1, y1, fill=fill, stroke=C["line"], width=0.9, radius=7)
    # Optical vertical center: ReportLab positions text by its baseline.
    text(c, label, (x0 + x1) / 2, (y0 + y1) / 2 + 3.8,
         11.2, bold=True, center=True, serif=True)


def control_block(c, x0, y0, x1, y1, title, detail, fill=C["paper"], tag=None):
    box(c, x0, y0, x1, y1, fill=fill, stroke=C["line"], width=0.8, radius=3)
    title_size = 7.2 if tag else 8.0
    text(c, title, x0 + 7, y0 + 12, title_size, bold=True, color=C["muted"])
    if isinstance(detail, (tuple, list)):
        text(c, detail[0], (x0 + x1) / 2, y0 + 27, 8.3,
             bold=True, center=True, serif=True)
        text(c, detail[1], (x0 + x1) / 2, y0 + 39, 8.3,
             bold=True, center=True, serif=True)
    else:
        text(c, detail, (x0 + x1) / 2, y0 + 28, 9.3,
             bold=True, center=True, serif=True)
    if tag:
        section_tag(c, x1 - 35, y0 + 3, tag)


def plain_module(c, x0, y0, x1, y1, title, detail, fill=C["paper"]):
    """Square-cornered module styled like a conventional paper schematic."""
    box(c, x0, y0, x1, y1, fill=fill, stroke=C["line"], width=0.9)
    text(c, title, x0 + 6, y0 + 11, 6.7, bold=True, color=C["muted"])
    if isinstance(detail, (tuple, list)):
        text(c, detail[0], (x0 + x1) / 2, y0 + 26, 8.2,
             bold=True, center=True, serif=True)
        text(c, detail[1], (x0 + x1) / 2, y0 + 38, 8.2,
             bold=True, center=True, serif=True)
    else:
        text(c, detail, (x0 + x1) / 2, y0 + 29, 8.7,
             bold=True, center=True, serif=True)


def analysis_marker(c, x, y, label):
    """Place a Section 6 marker outside a module so it cannot cover labels."""
    line(c, x - 16, y + 2, x, y + 2, C["orange"], 1.2)
    text(c, label, x, y, 7.3, bold=True, right=True, color=C["orange"])


def task_label(c, x, cy, label, color):
    box(c, x, cy - 15, x + 5, cy + 15, fill=color, stroke=None)
    text(c, label, x + 31, cy + 4, 9.5, bold=True, center=True, serif=True)


def document_artifact(c, x, cy, label, color, w=86, h=38):
    """Small document glyph replaces the former green rounded output card."""
    y0 = cy - h / 2
    fold = 10
    pts = [(x, y0), (x + w - fold, y0), (x + w, y0 + fold),
           (x + w, y0 + h), (x, y0 + h)]
    poly(c, pts, C["paper"], C["line"], 0.9)
    line(c, x + w - fold, y0, x + w - fold, y0 + fold, C["line"], 0.7)
    line(c, x + w - fold, y0 + fold, x + w, y0 + fold, C["line"], 0.7)
    line(c, x + 7, y0 + 8, x + 28, y0 + 8, color, 2.5)
    text(c, label, x + w / 2, y0 + 27, 7.5, bold=True, center=True, serif=True)


def draw_system(c):
    """Draw a sparse academic schematic rather than a card-style UI."""
    text(c, "KoreaDrive: one frozen VLM, three deterministic task programs",
         18, 22, 13.5, bold=True, serif=True)
    text(c, "one row is activated by the benchmark task ID",
         882, 22, 7.4, right=True, color=C["muted"])

    headers = [
        (216, "TASK"), (362, "PRE-VLM TASK PROGRAM"),
        (560, "SHARED FROZEN VLM"), (706, "POST-VLM CONTROL"),
        (837, "ARTIFACT"),
    ]
    for x, label in headers:
        text(c, label, x, 49, 6.9, bold=True, center=True, color=C["muted"])
    line(c, 252, 54, 475, 54, C["soft"], 0.7)

    # Benchmark content and its task ID are dispatched together.
    box(c, 18, 58, 151, 122, fill=C["paper"], stroke=C["line"], width=0.9)
    text(c, "Benchmark-provided item", 84.5, 71, 8.6, bold=True,
         center=True, serif=True)
    road_frame(c, 27, 77, 46, 28)
    question_card(c, 80, 76, 61, 30)
    text(c, "video", 50, 116, 6.8, bold=True, center=True, color=C["muted"])
    text(c, "instruction", 110.5, 116, 6.8, bold=True, center=True, color=C["muted"])
    text(c, "benchmark task ID", 84.5, 138, 8.1, bold=True,
         center=True, color=C["muted"])
    arrow(c, 84.5, 141, 84.5, 149, C["line"], 1.0, head=4)
    poly(c, [(31, 149), (142, 149), (149, 183), (24, 183)],
         C["panel"], C["line"], 1.0)
    text(c, "Task-control selection", 86.5, 171, 8.7,
         bold=True, center=True, serif=True)

    row_y = {"TAR": 88, "FETV": 165, "PSI-VQA": 242}
    task_colors = {"TAR": C["yellow"], "FETV": C["peach"], "PSI-VQA": C["lav"]}
    rows = [
        ("TAR", "MM:SS window", "exact / descriptive",
         ("token extraction", "BCQ vote"), "TAR CSV"),
        ("FETV", "uniform, max 16", "one-call 13 fields",
         ("schema validation", "description template"), "13-field JSON"),
        ("PSI-VQA", "real-fps, max 16", "per-subtask",
         ("type fallback", "temporal prior"), "PSI-VQA CSV"),
    ]

    # Thin separators and unboxed task names follow the reference-paper style.
    line(c, 181, 126, 884, 126, C["soft"], 0.65)
    line(c, 181, 203, 884, 203, C["soft"], 0.65)

    # One encoder-shaped backbone is shared by all three paths.
    poly(c, [(503, 59), (617, 67), (617, 270), (503, 278)],
         C["blue_fill"], C["line"], 1.2)
    snowflake(c, 517, 75, 7)
    text(c, "Qwen3-VL-8B-", 560, 116, 10.8, bold=True, center=True, serif=True)
    text(c, "Instruct", 560, 132, 10.8, bold=True, center=True, serif=True)
    text(c, "ONE SHARED", 560, 189, 7.8, bold=True, center=True, color=C["blue"])
    text(c, "FROZEN BACKBONE", 560, 203, 7.8, bold=True, center=True, color=C["blue"])
    text(c, "one checkpoint", 560, 229, 7.1, center=True, color=C["muted"])
    text(c, "no adapter or parameter update", 560, 260, 6.6,
         center=True, color=C["muted"])

    branch_x = 171
    line(c, 149, 166, branch_x, 166, C["line"], 1.2)
    line(c, branch_x, row_y["TAR"], branch_x, row_y["PSI-VQA"], C["line"], 1.2)

    for name, frame, prompt, post, contract in rows:
        cy = row_y[name]
        color = task_colors[name]
        arrow(c, branch_x, cy, 187, cy, C["line"], 1.1, head=4)
        task_label(c, 187, cy, name, color)
        arrow(c, 245, cy, 251, cy, C["line"], 0.9, head=4)
        plain_module(c, 251, cy - 21, 345, cy + 21,
                     "FRAME POLICY", frame, C["paper"])
        arrow(c, 345, cy, 365, cy, C["line"], 0.9, head=4)
        plain_module(c, 365, cy - 21, 476, cy + 21,
                     "PROMPT PROGRAM", prompt, C["yellow_fill"])
        arrow(c, 476, cy, 503, cy, C["line"], 1.0, head=5)
        arrow(c, 617, cy, 644, cy, C["line"], 1.0, head=5)
        plain_module(c, 644, cy - 21, 768, cy + 21,
                     "OUTPUT CONTROL", post, C["paper"])
        arrow(c, 768, cy, 792, cy, C["line"], 0.9, head=4)
        document_artifact(c, 792, cy, contract, color)

    # Section markers live outside modules; none can obscure a module title.
    analysis_marker(c, 345, row_y["TAR"] - 26, "§6.4")
    analysis_marker(c, 768, row_y["FETV"] - 26, "§6.3")
    analysis_marker(c, 476, row_y["PSI-VQA"] - 26, "§6.2")
    analysis_marker(c, 768, row_y["PSI-VQA"] - 26, "§6.1")

    # Minimal legend: color is semantic, not decorative.
    line(c, 18, 293, 882, 293, C["soft"], 0.65)
    arrow(c, 29, 316, 54, 316, C["line"], 1.1, head=5)
    text(c, "selected inference path", 62, 320, 7.0, color=C["muted"])
    snowflake(c, 219, 315, 5.5)
    text(c, "frozen", 231, 320, 7.0, color=C["muted"])
    box(c, 292, 309, 312, 321, fill=C["yellow_fill"], stroke=C["soft"], width=0.6)
    text(c, "KoreaDrive prompt program used in the submission",
         320, 319, 6.8, color=C["muted"])
    text(c, "§6", 699, 319, 7.2, bold=True, color=C["orange"])
    line(c, 698, 321, 719, 321, C["orange"], 1.2)
    text(c, "variable analyzed in Sec. 6", 727, 319, 6.8, color=C["muted"])


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(W, H), pageCompression=1)
    c.setTitle("KoreaDrive: one frozen backbone with deterministic task-specific control")
    c.setAuthor("KoreaDrive")
    draw_system(c)
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
