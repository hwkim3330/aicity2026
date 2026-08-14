from pathlib import Path
import math

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "fig_system_overview.pdf"
W, H = 900, 380

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
        width=1.0, dash=(3, 2), radius=2)
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
    text(c, label, (x0 + x1) / 2, y0 + 13.3, 9, bold=True, center=True, serif=True)


def control_block(c, x0, y0, x1, y1, title, detail, fill=C["paper"], tag=None):
    box(c, x0, y0, x1, y1, fill=fill, stroke=C["line"], width=0.8, radius=3)
    title_size = 6.3 if tag else 7.3
    text(c, title, x0 + 7, y0 + 12, title_size, bold=True, color=C["muted"])
    text(c, detail, (x0 + x1) / 2, y0 + 27, 8.1, bold=True, center=True, serif=True)
    if tag:
        section_tag(c, x1 - 35, y0 + 3, tag)


def draw_shared_inference(c):
    text(c, "1", 25, 30, 11, bold=True, center=True, color=C["paper"], serif=True)
    c.setFillColorRGB(*C["ink"])
    c.circle(25, yy(26), 10, stroke=0, fill=1)
    text(c, "1", 25, 30, 11, bold=True, center=True, color=C["paper"], serif=True)
    text(c, "Shared frozen video-language inference", 44, 31, 15, bold=True, serif=True)

    # Benchmark-provided multimodal input, grouped as one direct model input.
    box(c, 28, 48, 253, 145, fill=C["paper"], stroke=C["soft"], width=0.8, dash=(4, 3), radius=4)
    text(c, "Benchmark-provided input", 38, 62, 9.5, bold=True, serif=True)
    road_frame(c, 38, 69, 54, 40)
    road_frame(c, 48, 75, 54, 40, (0.85, 0.22, 0.15))
    road_frame(c, 58, 81, 54, 40, (0.20, 0.55, 0.85))
    text(c, "video clip", 75, 136, 8.2, bold=True, center=True)
    question_card(c, 128, 76, 108, 42)
    text(c, "instruction / question", 182, 136, 8.2, bold=True, center=True)

    # Direct path to the shared backbone.
    arrow(c, 253, 98, 322, 98, C["line"], 1.5)
    text(c, "direct model input", 288, 86, 7.2, center=True, color=C["muted"])

    # Backbone shown as the single visual center of gravity.
    poly(c, [(322, 63), (505, 72), (505, 131), (322, 140)], C["blue_fill"], C["line"], 1.3)
    snowflake(c, 340, 69, 8)
    text(c, "Qwen3-VL-8B-Instruct", 414, 91, 12.2, bold=True, center=True, serif=True)
    text(c, "shared frozen backbone", 414, 108, 9.2, bold=True, center=True, color=C["blue"])
    stack(c, 382, 116, n=3, w=66, h=7)

    # Draft output and submission program hand-off.
    arrow(c, 505, 101, 557, 101, C["line"], 1.5)
    box(c, 557, 78, 677, 124, fill=C["paper"], stroke=C["line"], width=1.0, radius=3)
    text(c, "Draft response", 617, 98, 11.2, bold=True, center=True, serif=True)
    text(c, "text / fields / interval", 617, 114, 7.4, center=True, color=C["muted"])
    arrow(c, 677, 101, 733, 101, C["line"], 1.5)
    box(c, 733, 69, 873, 133, fill=C["green"], stroke=C["line"], width=1.1, radius=3)
    text(c, "Benchmark-valid", 803, 91, 10.2, bold=True, center=True, serif=True)
    text(c, "submission artifact", 803, 108, 10.2, bold=True, center=True, serif=True)
    text(c, "CSV or 13-field JSON", 803, 123, 7.4, center=True, color=C["muted"])

    # Compact configuration bar and legend.
    box(c, 322, 146, 677, 168, fill=C["panel"], stroke=C["soft"], width=0.7)
    text(c, "bf16", 350, 160, 7.7, bold=True, center=True)
    text(c, "rev. 0c351dd0", 428, 160, 7.7, bold=True, center=True)
    text(c, "up to 16 frames", 526, 160, 7.7, bold=True, center=True)
    text(c, "151,200 px/frame", 623, 160, 7.7, bold=True, center=True)
    line(c, 380, 150, 380, 164, C["soft"], 0.7)
    line(c, 479, 150, 479, 164, C["soft"], 0.7)
    line(c, 574, 150, 574, 164, C["soft"], 0.7)
    snowflake(c, 727, 153, 6)
    text(c, "Frozen; no adapter or parameter update", 741, 157, 7.4, color=C["muted"])


def draw_task_controls(c):
    line(c, 20, 180, 880, 180, C["soft"], 0.8)
    c.setFillColorRGB(*C["ink"])
    c.circle(25, yy(201), 10, stroke=0, fill=1)
    text(c, "2", 25, 205, 11, bold=True, center=True, color=C["paper"], serif=True)
    text(c, "Deterministic task-specific control (not learned)", 44, 206, 15, bold=True, serif=True)

    # Selector and unmistakable three-way branch.
    text(c, "benchmark task ID", 37, 232, 8.2, bold=True, color=C["muted"])
    arrow(c, 88, 238, 88, 248, C["line"], 1.2)
    poly(c, [(36, 248), (140, 248), (152, 282), (24, 282)], C["peach"], C["line"], 1.1)
    text(c, "Task-control", 87, 264, 10.2, bold=True, center=True, serif=True)
    text(c, "selection", 87, 278, 10.2, bold=True, center=True, serif=True)

    branch_x = 171
    line(c, 152, 265, branch_x, 265, C["line"], 1.3)
    line(c, branch_x, 231, branch_x, 331, C["line"], 1.3)
    row_y = {"TAR": 231, "FETV": 281, "PSI-VQA": 331}
    fills = {"TAR": C["yellow_fill"], "FETV": C["peach"], "PSI-VQA": C["lav"]}

    rows = [
        ("TAR", "MM:SS window", "exact / descriptive", "token extraction + BCQ vote", "TAR CSV", "§6.4", None),
        ("FETV", "uniform, up to 16", "one-call 13 fields", "schema + description template", "13-field JSON", None, "§6.3"),
        ("PSI-VQA", "real-fps, up to 16", "per-subtask", "type fallback + temporal prior", "PSI-VQA CSV", None, "§6.1-2"),
    ]
    for name, frame, prompt, post, contract, frame_tag, post_tag in rows:
        cy = row_y[name]
        # A task lane contains the task's individual control modules.  The
        # nesting mirrors conventional architecture figures and avoids a
        # floating sequence of unrelated boxes.
        box(c, 184, cy - 22, 885, cy + 22, fill=(0.997, 0.997, 0.997),
            stroke=C["soft"], width=0.55, radius=3)
        arrow(c, branch_x, cy, 190, cy, C["line"], 1.2)
        pill(c, 190, cy - 18, 264, cy + 18, name, fills[name])
        arrow(c, 264, cy, 280, cy, C["line"], 1.0)
        control_block(c, 280, cy - 19, 394, cy + 19, "FRAME POLICY", frame, C["paper"], frame_tag)
        arrow(c, 394, cy, 410, cy, C["line"], 1.0)
        control_block(c, 410, cy - 19, 536, cy + 19, "PROMPT PROGRAM", prompt, C["yellow_fill"])
        arrow(c, 536, cy, 556, cy, C["line"], 1.0)

        # Repeated ports call the same frozen backbone drawn once above.
        box(c, 556, cy - 14, 620, cy + 14, fill=C["blue_fill"], stroke=C["blue"], width=0.9, radius=4)
        snowflake(c, 566, cy, 4.5)
        text(c, "frozen VLM", 597, cy + 4, 7.1, bold=True, center=True, color=C["blue"])
        arrow(c, 620, cy, 636, cy, C["line"], 1.0)
        control_block(c, 636, cy - 19, 788, cy + 19, "OUTPUT CONTROL", post, C["paper"], post_tag)
        arrow(c, 788, cy, 804, cy, C["line"], 1.0)
        box(c, 804, cy - 19, 880, cy + 19, fill=C["green"], stroke=C["line"], width=0.9, radius=3)
        text(c, contract, 842, cy + 4, 8.0, bold=True, center=True, serif=True)

    # Small, factual legend.
    box(c, 281, 362, 301, 373, fill=C["yellow_fill"], stroke=C["soft"], width=0.6)
    text(c, "KoreaDrive prompt program used in the submission", 308, 371, 6.8, color=C["muted"])
    box(c, 617, 362, 645, 374, fill=C["paper"], stroke=C["orange"], width=0.9, dash=(3, 2), radius=2)
    text(c, "§6", 631, 371, 7.1, bold=True, center=True, color=C["orange"])
    text(c, "variable analyzed in Sec. 6", 652, 371, 6.8, color=C["muted"])


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(W, H), pageCompression=1)
    c.setTitle("KoreaDrive: one frozen backbone with deterministic task-specific control")
    c.setAuthor("KoreaDrive")
    draw_shared_inference(c)
    draw_task_controls(c)
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
