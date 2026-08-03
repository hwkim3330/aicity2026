#!/usr/bin/env python3
"""Replace a PSI-VQA submission's temporal_localization rows with the
duration-stratified prior that shipped in v6/v7.

Generalizes the one-off `regen_psi_temporal_prior.py` (which hard-coded the
v5 -> v6 transition) into a script that takes any input CSV, so the official
chain can be replayed end to end without editing source.

The shipped rule, fitted on all 227 training intervals:

    duration <  10 s -> [0.30 * d, 0.70 * d]
    duration >= 10 s -> [0.20 * d, 0.62 * d]

endpoints rounded to whole seconds, end forced strictly greater than start.

The rule's search space, objective, per-split refit, and metadata-only
baselines are documented in `analysis/temporal_prior_protocol.md`. Held out
properly, the refitted prior scores 0.5566 +/- 0.0307 mIoU; these shipped
constants score 0.5724 in-sample on the same splits.

    python3 apply_psi_temporal_prior.py --in  ../submissions/reproduced_psi_base.csv \
                                        --out ../submissions/reproduced_psi_temporal.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys

SHORT_LO, SHORT_HI = 0.30, 0.70
LONG_LO, LONG_HI = 0.20, 0.62
THRESHOLD_S = 10.0


def duration_of(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def mmss(t):
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"


def prior_answer(dur):
    lo, hi = (SHORT_LO, SHORT_HI) if dur < THRESHOLD_S else (LONG_LO, LONG_HI)
    s, e = round(lo * dur), round(hi * dur)
    if e <= s:
        e = s + 1
    return json.dumps({"start": mmss(s), "end": mmss(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions",
                    default="../data/psi_vqa/test_public/temporal_localization_questions.json")
    ap.add_argument("--videos", default="../data/psi_vqa/test_public/videos")
    args = ap.parse_args()

    if os.path.abspath(args.inp) == os.path.abspath(args.out):
        sys.exit("refusing to overwrite the input file in place")

    items = json.load(open(args.questions))["items"]
    new = {}
    for it in items:
        rel = it["video_id"].split("PSI/", 1)[-1]
        vpath = os.path.join(args.videos, rel)
        if not os.path.isfile(vpath):
            sys.exit(f"missing video: {vpath}")
        new[str(it["item_index"])] = prior_answer(duration_of(vpath))

    rows = list(csv.DictReader(open(args.inp)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_index", "prediction"])
        for r in rows:
            pred = r["prediction"]
            if r["item_index"] in new:
                pred = new[r["item_index"]]
                n += 1
            w.writerow([r["item_index"], pred])
    print(f"wrote {args.out}: {len(rows)} rows, {n}/{len(new)} temporal rows replaced")
    if n != len(new):
        sys.exit(f"FAIL: expected to replace {len(new)} temporal rows, replaced {n}")


if __name__ == "__main__":
    main()
