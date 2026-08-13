"""Render the final AI City 2026 standings for all 8 tracks, both views.

The site loads each table from
`/aicity2026/submission/leaderboard/stats/{general|public}/{track}` and paginates
at 10 rows, so a screenshot of the page shows neither the full field nor Korea
Drive on the tracks where we rank low. This pulls every row of all sixteen tables
and renders them in full, with our row marked.

`general` is the full field; `public` is the subset that opted into the public
board. Both are kept because our rank differs between them on every track.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

TEAM_ID = 277
SRC = Path(__file__).resolve().parent / "raw"
OUT = Path(__file__).resolve().parent

TRACKS = {
    1: ("Multi-Camera 3D Perception (Sim2Real)", "hota"),
    2: ("Transportation Safety Understanding and Captioning (Sim2Real)", "s2"),
    3: ("Traffic Anomaly Reasoning (TAR)", "mean"),
    4: ("Text-Based Person Re-Identification (Sim2Real)", "map"),
    5: ("Generative Traffic Video Forecasting", "final"),
    6: ("Cross-City Object Detection (Milestone Project Hafnia)", "map"),
    7: ("Out-of-Domain Evaluation A (FETV)", "final_score"),
    8: ("Out-of-Domain Evaluation B (PSI-VQA)", "final"),
}


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, (int, float)):
        return f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
    return html.escape(str(v))


def load(view: str, track: int) -> list[dict]:
    path = SRC / f"{view}_{track}.json"
    return json.loads(path.read_text()) if path.is_file() else []


def columns(track: int) -> list[str]:
    """Every metric the portal reports for this track, primary first.

    Derived from the data rather than listed by hand: the tracks report between
    four and fifteen metrics each, and a hand-written list silently drops the
    ones it forgets. Ranking metric first, then the raw metrics, then the
    normalised `*_norm` restatements that tracks 5 and 8 carry.
    """
    keys: set[str] = set()
    for view in ("general", "public"):
        for r in load(view, track):
            keys |= set((r.get("score") or {}).keys())
    primary = TRACKS[track][1]
    keys.discard(primary)
    norm = sorted(k for k in keys if k.endswith("_norm"))
    rest = sorted(k for k in keys if not k.endswith("_norm"))
    return [primary] + rest + norm


def summary_rows() -> list[tuple]:
    out = []
    for t, (name, _) in TRACKS.items():
        cells = []
        for view in ("general", "public"):
            rows = load(view, t)
            mine = [r for r in rows if r.get("teamId") == TEAM_ID]
            cells.append((mine[0]["rank"] if mine else None, len(rows)))
        out.append((t, name, *cells))
    return out


def table_html(view: str, track: int) -> str:
    cols = columns(track)
    rows = load(view, track)
    if not rows:
        return f"<p class='empty'>no rows for {view} track {track}</p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for r in sorted(rows, key=lambda r: r.get("rank") or 10**9):
        sc = r.get("score") or {}
        mine = r.get("teamId") == TEAM_ID
        cls = " class='mine'" if mine else (" class='baseline'" if r.get("isBaseline") else "")
        cells = "".join(f"<td>{fmt(sc.get(c, r.get(c)))}</td>" for c in cols)
        tag = " <span class='tag'>ours</span>" if mine else (
            " <span class='tag base'>baseline</span>" if r.get("isBaseline") else "")
        body.append(f"<tr{cls}><td class='r'>{r.get('rank','')}</td>"
                    f"<td>{html.escape(str(r.get('teamName','')))}{tag}</td>{cells}</tr>")
    return (f"<table><thead><tr><th class='r'>#</th><th>team</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


CSS = "\n".join([
        "<style>",
        ":root{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--mine:#fff3cd;--minefg:#7a5b00;--base:#f5f7fa}",
        "@media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2a2f36;--mine:#3a3000;--minefg:#ffd866;--base:#1c2027}}",
        ":root[data-theme=dark]{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2a2f36;--mine:#3a3000;--minefg:#ffd866;--base:#1c2027}",
        ":root[data-theme=light]{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--mine:#fff3cd;--minefg:#7a5b00;--base:#f5f7fa}",
        "body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:28px}",
        "h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:30px 0 4px;border-top:1px solid var(--line);padding-top:18px}",
        "h3{font-size:13px;color:var(--mut);margin:14px 0 6px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}",
        "p.sub{color:var(--mut);margin:0 0 18px}",
        ".wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}",
        "table{border-collapse:collapse;width:100%;max-width:100%;font-variant-numeric:tabular-nums;margin-bottom:6px}",
        "th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}",
        "th:nth-child(2),td:nth-child(2){text-align:left}th{color:var(--mut);font-weight:600;font-size:12px}",
        # Metric names run to 24 characters while the values are six, so the
        # headers alone decide the width. Let them wrap; keep the numbers on
        # one line. Without this, track 7's fifteen columns overrun the page.
        "th{white-space:normal;overflow-wrap:anywhere;vertical-align:bottom;max-width:80px}",
        "td.r,th.r{width:34px;text-align:right;color:var(--mut)}",
        "tr.mine{background:var(--mine)}tr.mine td{color:var(--minefg);font-weight:700}",
        "tr.baseline{background:var(--base)}",
        ".tag{font-size:10px;background:var(--minefg);color:var(--bg);padding:1px 5px;border-radius:3px;margin-left:6px;vertical-align:1px}",
        ".tag.base{background:var(--mut)}",
        ".empty{color:var(--mut);font-style:italic}",
        # Track 7 reports fifteen metrics, so portrait A4 clips the widest tables.
        "@page{size:A4 landscape;margin:10mm}",
        "@media print{body{padding:0;font-size:9px}h2{font-size:12px;page-break-after:avoid}"
        "h3{page-break-after:avoid}table{font-size:8px}th,td{padding:2px 4px}"
        "tr{page-break-inside:avoid}.wrap{overflow:visible}}",
        "</style>",
])


def main() -> None:
    parts = [
        "<title>AI City Challenge 2026 — final standings, all tracks</title>",
        CSS,
        "<h1>AI City Challenge 2026 — final standings</h1>",
        "<p class='sub'>Team <strong>Korea Drive</strong> (id 277). Every row of all eight tracks, "
        "both the full <em>general</em> field and the opted-in <em>public</em> board. "
        "The live site paginates at ten rows, which hides our position on most tracks.</p>",
        "<h2>Where we placed</h2><div class='wrap'><table><thead><tr>"
        "<th class='r'>#</th><th>track</th><th>general</th><th>public</th></tr></thead><tbody>",
    ]
    for t, name, gen, pub in summary_rows():
        def cell(x):
            rank, total = x
            return f"<strong>{rank}</strong> / {total}" if rank else f"— / {total}"
        parts.append(f"<tr><td class='r'>{t}</td><td>{html.escape(name)}</td>"
                     f"<td>{cell(gen)}</td><td>{cell(pub)}</td></tr>")
    parts.append("</tbody></table></div>")

    for t, (name, _) in TRACKS.items():
        parts.append(f"<h2>Track {t} — {html.escape(name)}</h2>")
        for view in ("general", "public"):
            parts.append(f"<h3>{view}</h3><div class='wrap'>{table_html(view, t)}</div>")

    (OUT / "final_leaderboard.html").write_text("\n".join(parts), encoding="utf-8")

    md = ["# AI City Challenge 2026 — final standings",
          "",
          "Team **Korea Drive** (id 277). All rows, all eight tracks, both views.",
          "",
          "| track | general | public |", "| --- | ---: | ---: |"]
    for t, name, gen, pub in summary_rows():
        g = f"{gen[0]}/{gen[1]}" if gen[0] else f"—/{gen[1]}"
        p = f"{pub[0]}/{pub[1]}" if pub[0] else f"—/{pub[1]}"
        md.append(f"| {t}. {name} | {g} | {p} |")
    for t, (name, _) in TRACKS.items():
        cols = columns(t)
        md += ["", f"## Track {t} — {name}"]
        for view in ("general", "public"):
            rows = sorted(load(view, t), key=lambda r: r.get("rank") or 10**9)
            if not rows:
                continue
            md += ["", f"### {view}", "",
                   "| # | team | " + " | ".join(cols) + " |",
                   "| ---: | --- | " + " | ".join("---:" for _ in cols) + " |"]
            for r in rows:
                sc = r.get("score") or {}
                mark = " **(ours)**" if r.get("teamId") == TEAM_ID else (
                    " _(baseline)_" if r.get("isBaseline") else "")
                vals = " | ".join(fmt(sc.get(c, r.get(c))) for c in cols)
                md.append(f"| {r.get('rank','')} | {r.get('teamName','')}{mark} | {vals} |")
    (OUT / "final_leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT/'final_leaderboard.html'} and .md")


if __name__ == "__main__":
    main()
