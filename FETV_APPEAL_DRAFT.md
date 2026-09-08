# FETV disqualification — assessment

Korea Drive (Team 277), Track 7 / FETV, 2026 AI City Challenge.
Written 2026-09-08 after the on-site announcement that Korea Drive was disqualified
from FETV for using a different model.

**Conclusion: the disqualification is most likely correct, and we should not contest
it.** The stated reason ("a different model") does not match the record, but the
underlying failure is real, is ours, and is already documented in this repository.
Only the characterization is worth correcting, and only if asked.

## Why the stated reason does not match the record

The rule (2026 FAQ Q2, repeated per leaderboard in Q7) is a unified-system
requirement: "teams may not submit separately tuned or specialized systems per task
type or per test set." It is not a model whitelist, so "a different model on FETV"
must mean *different from TAR*.

Three organizer-side records say otherwise:

* the portal carries `models_used: qwen3` on the scored TAR submission;
* *The 10th AI City Challenge* (arXiv 2608.17044) lists Team 277 in the Track 3 table
  at 0.4256 attributing **Qwen3**, and lists Korea Drive on FETV at 3/8 public
  (5/15 overall), 0.4634, with no ineligibility mark;
* the camera-ready abstract states a "frozen Qwen3-VL-8B-Instruct inference pipeline …
  no task-specific parameter update or adapter generated the official predictions,"
  and the final PDF contains zero occurrences of "Qwen2.5".

Every reproduce script in this repository also hardcodes or defaults to
`Qwen/Qwen3-VL-8B-Instruct`. No adapter weights are tracked.

## Why the disqualification is nonetheless sound

The summary paper describes verification as: "Award-candidate teams were required to
provide reproducible code and models." At FETV rank 3 we were inside that band, and
the repository handed over is the one cited in the paper.

**The FETV artifact does not reproduce byte-for-byte, and we knew.** Commit `84eb117`,
2026-08-13, "FETV does not reproduce, and the repository said it would":

* re-ran all 200 public clips with the revision pinned: **0 of 200 records matched**
  the shipped v11 (re-measured today: `answer_description` differs on 200/200,
  `answer_time` on 169/200, the categorical fields on 28-81 each);
* the artifact is **the last of an eleven-step chain, not one run** — v7→v8 rewrote 56
  rows, v8→v9 another 57, v9→v10 fifteen, v10→v11 ninety-two — while `REPRODUCE.md`
  listed a single command and an expected SHA256 beside it;
* only v10→v11 was recovered and verified exactly (`make_fetv_v11_descriptions.py
  --verify`). **The commands behind v9 and v10 remain unrecorded**, so those two steps
  cannot be reproduced by anyone, including us.

### But byte-identity is the wrong test, and the score-level evidence is better

The challenge summary paper states the metric: categorical attributes by macro-averaged
F1, `date` by exact match, and **`time` correct if within seven seconds of the ground
truth**. Under the metric rather than under `diff`:

| measurement | shipped v11 vs reproduction |
| --- | --- |
| records byte-identical | 0 / 200 |
| `answer_intersection_type` **actual score** (ground truth recovered, below) | 0.784068 vs **0.781971** |
| `answer_time` within the 7 s scoring tolerance | **162 / 200** — score-equivalent |
| other categorical fields, agreement | 60–72 % |

So on the one field whose ground truth we can establish exactly, the reproduction
scores within 0.002 of the submission, and four fifths of the timestamp differences
are inside the tolerance the metric actually applies. The honest statement is that the
pipeline reproduces the *result* closely wherever we can measure it, while the
*artifact* does not reproduce at all — and that two of the eleven steps are
unrecoverable, which is the part that cannot be argued away.

Commit `b6949ae` the same day ruled out five candidate causes for the residual
`answer_time` bias by measurement — run-to-run nondeterminism, the determinism pin,
a different clip encode, frame-sampling drift, and the few-shot exemplars — and named
no cause.

Unrecorded row-rewriting steps applied to one test set are not something we can defend
as the "task-specific prompts, parsing, and routing" the rule permits, because we
cannot say what they did.

### What the same investigation did establish

`answer_intersection_type` is a property of the camera, not the clip: each fisheye
source watches one fixed junction. Assuming per-source constancy leaves 512 possible
assignments, and scoring v11 against each reproduces our official
`0.7840684660961159` for **exactly one**, under macro-F1 and under no other metric.
Six of the nine sources carry the junction name burned into the frame, and all six
agree with the solved assignment (巷口 → T-intersection, 街口/路口 → four-way). The
scoring model is now verified for all eight teams to machine precision:
`final = (mean of the twelve categorical macro-F1 + description) / 2`.

Correcting that one field — `fetv_fix_intersection.py`, 41 of 200 rows — takes the
submission from the official **0.463436** to **0.472433**. It is recorded because it
is provable, not because it changes anything: the board is closed, first place was
0.489150, and none of the remaining gap is in a field whose ground truth we can
recover.

## Our own documentation problem, separately

`track3_anomaly/README.md` still read, until commit `9bc25e8` today, "**TAR baseline:**
`Qwen/Qwen2.5-VL-7B-Instruct`, 4-bit NF4 … Qwen3-VL-8B was later integrated for the
official FETV and PSI-VQA runs. The earlier TAR baseline retained Qwen2.5-VL-7B."
Read literally that asserts different models per test set. It described the July
exploratory baseline, was never updated when Qwen3-VL-8B became the TAR submission,
and stood in the cited public repository throughout the verification window. It is
the most plausible source of the "different model" wording.

The submitted manuscript carried the same misstatement. We found it ourselves and
published the correction in `a07a930` on **2026-08-03**, before results, in
`OFFICIAL_RESULTS.md` and `REPRODUCE.md` gap 2, and carried it into the camera-ready —
but missed the track README in that sweep.

## Recommendation

Do not file an appeal contesting the disqualification. If we write to the organizers
at all, ask only two things, and concede the rest:

1. Which finding the determination rested on, for our records.
2. That the characterization be corrected to reproducibility rather than model
   identity, if that is in fact what was found — the portal field, the summary paper
   and the camera-ready all attribute one backbone to Team 277.

And state plainly that we do not dispute the outcome: the FETV artifact is the end of
a chain whose middle steps we did not record, and we published that finding ourselves
on 2026-08-13.
