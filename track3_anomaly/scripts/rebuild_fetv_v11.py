#!/usr/bin/env python3
"""Rebuild the shipped FETV v11 from the archived v7, and name what cannot be derived.

The chain was not eleven model runs. Reading the artifact timestamps of 2026-07-11 --
v7 05:48, v8 10:54, v9 11:09, v10 16:41, v11 16:44 -- and diffing them:

  v7 -> v8   a second pass over 134 clips changed 56 rows. Its raw model output is
             archived in submissions/fetv_v8_secondpass_raw.json.
  v8 -> v9   57 rows reverted. v9 equals v7 on 199 of 200 rows: the second pass was
             tried and discarded. v8 is a dead end, not a stage.
  v9 -> v7   the single surviving difference is 001_001.mp4, set to the one ground-truth
             row the FETV documentation publishes as an example.
  v9 -> v10  15 no_violation rows get violator_type and colour filled in.
  v10 -> v11 92 descriptions rewritten by template from each row's own fields, already
             recovered as make_fetv_v11_descriptions.py.

So no model call is needed to go from v7 to v11. Four of the five steps are derivable.
The fifth is not, and this script does not pretend otherwise: the 15-row edit is carried
here as a literal table, because

  * 14 of its 15 (violator_type, colour) pairs are the model's own output in the first
    two submissions, recoverable -- but
  * 001_013.mp4's colour "dark" appears in no archived artifact at all, and
  * the choice of these 15 clips out of the 64 that qualify equally has no rule that
    survives testing. Edited and unedited candidates are statistically indistinguishable
    (mean cross-version agreement 4.00 vs 3.63, mean distinct answers 1.53 vs 1.55).

That is the honest boundary of the reconstruction, and it is why the artifact is not
reproducible: something outside the pipeline chose those rows.

    python3 rebuild_fetv_v11.py --verify
"""
from __future__ import annotations
import argparse, json, os, sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'submissions')

# The published FETV example row (also used as the anchor in fetv_gt_posterior200.py).
PUBLISHED_GT = {
    '001_001.mp4': dict(answer_time='06:04:50', answer_violation_type='jaywalking',
                        answer_violator_type='pedestrian', answer_color='mixed',
                        answer_initial_position='Bottom-Right',
                        answer_final_position='Middle-Right'),
}

# provenance: "v1/v2" = the pair appears verbatim in the first two submissions.
EDIT_V10 = {
    '001_003.mp4': ('car',        'light', 'v1/v2'),
    '001_008.mp4': ('pedestrian', 'light', 'v1/v2'),
    '001_013.mp4': ('pedestrian', 'dark',  'colour unattributed'),
    '001_016.mp4': ('motorcycle', 'mixed', 'v1/v2'),
    '001_020.mp4': ('bus',        'red',   'v1/v2'),
    '001_027.mp4': ('car',        'dark',  'v1/v2'),
    '002_008.mp4': ('motorcycle', 'red',   'v1/v2'),
    '002_012.mp4': ('pedestrian', 'dark',  'v1/v2'),
    '005_009.mp4': ('truck',      'dark',  'v1/v2'),
    '005_026.mp4': ('car',        'dark',  'v1/v2'),
    '006_003.mp4': ('motorcycle', 'dark',  'v1/v2'),
    '006_017.mp4': ('bus',        'blue',  'v1/v2'),
    '010_002.mp4': ('motorcycle', 'light', 'v1/v2'),
    '010_004.mp4': ('car',        'light', 'v1/v2'),
    '019_011.mp4': ('car',        'light', 'v1/v2'),
}


def load(name):
    return json.load(open(os.path.join(BASE, name)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--out', default='rebuilt_fetv_v11.json')
    args = ap.parse_args()

    rows = load('fetv_submission_v7.json')
    by = {r['clip_name']: r for r in rows}

    for clip, fields in PUBLISHED_GT.items():
        by[clip].update(fields)
    for clip, (vt, colour, _prov) in EDIT_V10.items():
        by[clip]['answer_violator_type'] = vt
        by[clip]['answer_color'] = colour

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_fetv_v11_descriptions import build as describe  # the recovered v10 -> v11 template
    # The template was applied to the 93 violation rows only; the 107 no_violation
    # descriptions carry through from v7 untouched.
    for r in rows:
        if str(r['answer_violation_type']).lower() != 'no_violation':
            r['answer_description'] = describe(r)

    target = {r['clip_name']: r for r in load('fetv_submission_v11.json')}
    exact = sum(1 for c in by if by[c] == target.get(c))
    print(f'rebuilt from v7: {exact}/{len(by)} rows identical to the shipped v11')
    if exact != len(by):
        for c in sorted(by):
            if by[c] != target.get(c):
                diff = [f for f in by[c] if by[c][f] != target.get(c, {}).get(f)]
                print(f'  {c}: {diff}')
    if not args.verify:
        json.dump(rows, open(os.path.join(BASE, args.out), 'w'), ensure_ascii=False, indent=1)
        print(f'wrote {args.out}')
    return 0 if exact == len(by) else 1


if __name__ == '__main__':
    raise SystemExit(main())
