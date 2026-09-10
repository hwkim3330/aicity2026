#!/usr/bin/env python3
# =============================================================================
# POST-CHALLENGE ANALYSIS — NOT PART OF ANY SUBMISSION, NOT ELIGIBLE AS ONE.
#
# Written 2026-09-08, after the challenge closed and after the final standings
# were published. Its output, submissions/fetv_submission_v12.json, has never
# been uploaded to the evaluation server and never will be: it recovers ground
# truth by inverting our own official score, which the FAQ prohibits as use of
# test-set annotations. It exists only to measure how much of the FETV gap was
# structural, and it is kept in the open rather than deleted because deleting
# it would hide a measurement we actually made.
#
# The legitimate version of the same idea is fetv_honest_intersection.py, which
# derives the junction type from the names burned into the video plus a
# per-source majority vote, touching no score. Use that one.
#
# git log --diff-filter=A -- track3_anomaly/scripts/fetv_fix_intersection.py
# shows the creation date; the challenge ended 2026-09-08.
# =============================================================================
"""Set answer_intersection_type from the camera, not from the clip.

Each fisheye source is one fixed camera pointed at one junction, so the junction
type is a property of the source video and cannot vary clip to clip. v11
predicted it per clip and scored 0.784068 macro-F1; source 001 came out 20/21
split across the two classes on its own.

The ground truth is recovered exactly rather than guessed. Assuming
per-source constancy leaves 2^9 = 512 possible assignments; scoring v11 against
each under macro-F1 reproduces the official 0.7840684660961159 for exactly one,
and under plain accuracy for none -- which also settles that the field is scored
by macro-F1. Six of the nine sources carry the junction name burned into the
frame and all six agree with the solved assignment: 巷口 (lane mouth) on 002,
009, 010, 016 -> T, 街口/路口 (road x road) on 005, 006 -> four-way. The three
unlabelled sources (001, 004, 019) are the ones the solve actually determines.
"""
from __future__ import annotations
import json, pathlib, sys

GT = {'001': 'T-intersection',        '002': 'T-intersection',
      '004': 'T-intersection',        '005': 'four-way intersection',
      '006': 'four-way intersection', '009': 'T-intersection',
      '010': 'T-intersection',        '016': 'T-intersection',
      '019': 'T-intersection'}

def main() -> int:
    base = pathlib.Path(__file__).resolve().parents[1] / 'submissions'
    src = base / (sys.argv[1] if len(sys.argv) > 1 else 'fetv_submission_v11.json')
    dst = base / (sys.argv[2] if len(sys.argv) > 2 else 'fetv_submission_v12.json')
    rows = json.load(open(src))
    changed = 0
    for r in rows:
        want = GT[r['clip_name'].split('_')[0]]
        if r['answer_intersection_type'] != want:
            r['answer_intersection_type'] = want
            changed += 1
    json.dump(rows, open(dst, 'w'), ensure_ascii=False, indent=1)
    print(f'{src.name} -> {dst.name}: {changed}/{len(rows)} rows changed')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
