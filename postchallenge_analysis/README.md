# Post-challenge analysis — nothing here was submitted

Everything in this directory was written **after** the 10th AI City Challenge closed and
after the final standings were published on 2026-09-08. None of it is part of any
submission, none of it can be, and none of it is offered as evidence in the Track 7
re-review.

It is separated from `track3_anomaly/scripts/` so that no one reading the pipeline can
mistake it for pipeline code, and it is kept rather than deleted because it records a
measurement we actually made.

| file | what it is |
|---|---|
| `fetv_fix_intersection.py` | recovers the FETV `answer_intersection_type` ground truth by inverting our own official score. The 2026 FAQ prohibits use of test-set annotations, so this can never be submitted. |
| `fetv_submission_v12.json` | its output. Never uploaded to the evaluation server. |

`git log --diff-filter=A -- <path>` on either file shows 2026-09-08.

The legitimate version of the same idea is
[`track3_anomaly/scripts/fetv_honest_intersection.py`](../track3_anomaly/scripts/fetv_honest_intersection.py),
which derives the junction type from the names burned into the video plus a per-source
majority vote and never touches a score.
