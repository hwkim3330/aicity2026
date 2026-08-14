#!/usr/bin/env python3
"""Draw the violator annotation on the FETV case frame.

The raw frame shows a busy fisheye intersection and, on its own, argues nothing.
What the case is about is that one submitted version designated a violator and
another declined to, while every global field stayed identical. So the box marks
**what the system predicted**, not ground truth: the FETV labels for the scored
subset are not public, and the case study explicitly does not claim which of the
two records is right.

    python3 scripts/annotate_fetv_case.py
"""
from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

SRC = pathlib.Path("paper/camera_ready_src/frames")
FRAME = SRC / "fetv_019_004_t01_50.jpg"
OUT = SRC / "fetv_019_004_annotated.jpg"

# Located by inspection on the 900 px frame: the yellow saloon stopped across the
# hatched keep-clear box beside the crosswalk.
BOX = (147, 313, 245, 389)

ORANGE = (232, 119, 34)
INK = (17, 17, 17)


def font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if pathlib.Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main() -> None:
    im = Image.open(FRAME).convert("RGB")
    d = ImageDraw.Draw(im)

    x0, y0, x1, y1 = BOX
    # Corner ticks rather than a closed rectangle: the box says "this object",
    # and leaving the sides open keeps the vehicle's own outline visible.
    d.rectangle(BOX, outline=ORANGE, width=3)
    t = 14
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(cx, cy), (cx + dx * t, cy)], fill=ORANGE, width=6)
        d.line([(cx, cy), (cx, cy + dy * t)], fill=ORANGE, width=6)

    f = font(19)
    label = "predicted violator (v9–v11)"
    tw = d.textlength(label, font=f)
    lx, ly = x0, y1 + 10
    d.rectangle((lx - 5, ly - 4, lx + tw + 7, ly + 26), fill=ORANGE)
    d.text((lx + 1, ly), label, fill=(255, 255, 255), font=f)

    # State the other reading in the frame, so the image cannot be read as a
    # ground-truth assertion.
    f2 = font(17)
    note = "v8 predicted no_violation with na for every dependent field"
    nw = d.textlength(note, font=f2)
    ny = im.height - 34
    d.rectangle((10, ny - 6, 10 + nw + 12, ny + 24), fill=(255, 255, 255))
    d.text((16, ny), note, fill=INK, font=f2)

    im.save(OUT, quality=93, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
