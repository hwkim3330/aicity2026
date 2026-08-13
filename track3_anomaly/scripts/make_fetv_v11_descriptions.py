#!/usr/bin/env python3
"""Reconstruct the v10 -> v11 description rewrite, the last FETV step.

The script that produced v11 was never committed. What it did is recoverable
from the artifact: every row whose `answer_violation_type` is not
`no_violation` had its description replaced by a template filled from that
row's own structured fields, and the 107 `no_violation` rows were left as the
model wrote them. 36 jaywalking rows share one sentence skeleton; the other
violation classes share one each.

So v11 = v10 with 92 descriptions regenerated. `001_001.mp4` is the 93rd
violation row and did not change, because v10 already carried the exemplar
sentence from the FETV README, which is what the template produces anyway.

    python3 make_fetv_v11_descriptions.py --verify

Verifies against the shipped artifact. Same idea as
`make_psi_v7_openqa_prior.py`, which recovered the equivalent gap on PSI.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBS = HERE.parent / "submissions"

# "dark", "light" and "mixed" describe a shade rather than a hue, and the
# sentence reads "dark-colored car" but "yellow car".
SHADES = {"dark", "light", "mixed"}

CLAUSE = {
    "wrong_way": ("traveled against the flow of traffic", "wrong-way driving"),
    "red_light": ("proceeded through the intersection against a red traffic signal",
                  "red-light"),
    "lane_use_control": ("used a lane restricted to a different movement",
                         "lane-use-control"),
    "lane_discipline": ("drifted across the lane markings", "lane-discipline"),
    "uturn": ("made a prohibited U-turn at the intersection", "U-turn"),
}


def subject(color: str, violator: str) -> str:
    if color in ("na", "", None):
        return violator
    # A vehicle of mixed colour reads "multi-colored"; the clothing sentence
    # keeps "mixed-colored". Only the vehicle wording changes.
    if color == "mixed":
        return f"multi-colored {violator}"
    return f"{color}-colored {violator}" if color in SHADES else f"{color} {violator}"


def lane_phrase(i: str, f: str) -> str:
    """Lanes are often 'na' or unchanged; the sentence adapts rather than
    printing 'from lane na into lane na'."""
    if i in ("na", "", None) and f in ("na", "", None):
        return ""
    if f in ("na", "", None) or i == f:
        return f"traveling in lane {i}"
    return f"moving from lane {i} into lane {f}"


def build(row: dict) -> str:
    vt = row["answer_violation_type"]
    date, time = row["answer_date"], row["answer_time"]
    ipos = str(row["answer_initial_position"]).lower()
    fpos = str(row["answer_final_position"]).lower()
    inter = row["answer_intersection_type"]
    head = f"On {date} at {time}, "

    if vt == "jaywalking":
        color = row["answer_color"]
        who = row["answer_violator_type"]
        cl = f"{color}-colored" if color in SHADES else color
        return (f"{head}a {who} wearing {cl} clothing crossed the roadway outside "
                f"the marked crosswalk at a {inter}, jaywalking from the {ipos} of "
                f"the frame toward the {fpos} while vehicle traffic was present.")

    action, label = CLAUSE[vt]
    subj = subject(row["answer_color"], row["answer_violator_type"])
    lanes = lane_phrase(str(row["answer_initial_lane"]), str(row["answer_final_lane"]))
    mid = f"{action}, {lanes} as it" if lanes else f"{action} as it"
    return (f"{head}a traffic incident occurred at a {inter}. A {subj} {mid} "
            f"crossed from the {ipos} of the frame toward the {fpos} in a "
            f"{label} violation.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=SUBS / "fetv_submission_v10.json")
    ap.add_argument("--out", type=Path, default=SUBS / "reconstructed_fetv_v11.json")
    ap.add_argument("--verify", action="store_true",
                    help="compare against the shipped v11 instead of writing")
    args = ap.parse_args()

    rows = json.loads(args.base.read_text())
    for r in rows:
        if r["answer_violation_type"] != "no_violation":
            r["answer_description"] = build(r)

    if not args.verify:
        args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")
        return 0

    ship = {r["clip_name"]: r for r in json.loads((SUBS / "fetv_submission_v11.json").read_text())}
    viol = [r for r in rows if r["answer_violation_type"] != "no_violation"]
    bad = [r for r in viol if r["answer_description"] != ship[r["clip_name"]]["answer_description"]]
    print(f"violation rows: {len(viol)}   reconstructed exactly: {len(viol) - len(bad)}")
    for r in bad[:6]:
        c = r["clip_name"]
        print(f"\n  [{c}] {r['answer_violation_type']}")
        print(f"    shipped: {ship[c]['answer_description']}")
        print(f"    built  : {r['answer_description']}")
    if bad:
        print(f"\n  {len(bad)} mismatches")
        return 1
    # the no_violation rows must be untouched for the whole file to match
    same = all(r["answer_description"] == ship[r["clip_name"]]["answer_description"]
               for r in rows)
    print("all 200 descriptions match the shipped v11" if same
          else "violation rows match; some no_violation rows differ from v10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
