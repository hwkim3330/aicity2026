"""Render one PNG per track and view — 16 images, each a complete table.

The combined PDF breaks tables wherever the page ends, which is fine to read but
awkward to drop into a slide or a mail. This gives one self-contained image per
board instead: sized to its own contents so nothing paginates, then trimmed to
the ink.

Route is HTML -> single huge PDF page (Chrome) -> PNG (pdftoppm) -> trim (Pillow),
because Chrome's --screenshot only captures the window rectangle and would clip
the long boards.
"""

from __future__ import annotations

import html
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

from build_final_leaderboard import CSS, TRACKS, columns, load, table_html

OUT = Path(__file__).resolve().parent / "images"
DPI = 150
PAD = 24  # px of white kept around the trimmed content


def chrome() -> str:
    for c in ("google-chrome", "chromium", "chromium-browser", "/snap/bin/chromium"):
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    sys.exit("no chrome/chromium found")


def render(track: int, view: str, tmp: Path, browser: str) -> Path | None:
    rows = load(view, track)
    if not rows:
        print(f"  track{track} {view}: no rows, skipped")
        return None
    name = TRACKS[track][0]
    ncol, nrow = len(columns(track)) + 2, len(rows)

    # Sized so the whole board lands on one PDF page. Generous on both axes;
    # the trim afterwards removes whatever is left over.
    w_mm, h_mm = 110 + ncol * 24, 70 + nrow * 6

    page = "\n".join([
        f"<title>Track {track} {view}</title>",
        CSS,
        # Images get pasted into light documents, so pin the light palette
        # rather than inheriting whatever the renderer defaults to.
        "<style>:root{color-scheme:light;--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;"
        "--mine:#fff3cd;--minefg:#7a5b00;--base:#f5f7fa}"
        f"@page{{size:{w_mm}mm {h_mm}mm;margin:8mm}}"
        "body{padding:0;font-size:11px}table{font-size:10px}"
        # The page grows to fit, so headers need not wrap here the way they do
        # in the paginated PDF -- mid-word breaks like "answer_descriptio/n"
        # only exist to save width that an image does not have to save.
        "th{max-width:none;white-space:nowrap}"
        "th,td{padding:3px 6px}h2{border:0;padding:0;margin:0 0 2px}</style>",
        f"<h2>Track {track} — {html.escape(name)}</h2>",
        f"<h3>{view} leaderboard — {nrow} teams — Korea Drive highlighted</h3>",
        table_html(view, track),
    ])
    src = tmp / f"t{track}_{view}.html"
    src.write_text(page, encoding="utf-8")
    pdf = tmp / f"t{track}_{view}.pdf"
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-sandbox",
                    f"--print-to-pdf={pdf}", "--no-pdf-header-footer",
                    f"file://{src}"], check=True, capture_output=True, timeout=180)
    subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-f", "1", "-l", "1",
                    str(pdf), str(tmp / f"t{track}_{view}")],
                   check=True, capture_output=True, timeout=180)

    raw = next(tmp.glob(f"t{track}_{view}-*.png"))
    im = Image.open(raw).convert("RGB")
    bbox = Image.new("RGB", im.size, (255, 255, 255))
    from PIL import ImageChops
    box = ImageChops.difference(im, bbox).getbbox()
    if box:
        l, t, r, b = box
        im = im.crop((max(0, l - PAD), max(0, t - PAD),
                      min(im.width, r + PAD), min(im.height, b + PAD)))
    dst = OUT / f"track{track}_{view}.png"
    im.save(dst, optimize=True)
    print(f"  {dst.name}: {im.width}x{im.height}, {nrow} rows, "
          f"{dst.stat().st_size / 1024:.0f} KB")
    return dst


def main() -> int:
    OUT.mkdir(exist_ok=True)
    tmp = OUT / ".work"
    tmp.mkdir(exist_ok=True)
    browser = chrome()
    made = 0
    for track in TRACKS:
        for view in ("general", "public"):
            if render(track, view, tmp, browser):
                made += 1
    shutil.rmtree(tmp)
    print(f"wrote {made} images to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
