# The PSI-VQA artifact that was never submitted (found 2026-08-20)

The organizers re-opened the evaluation system on 2026-08-19 and doubled the
per-track submission count. That makes one dormant artifact worth revisiting.

## What it is

`track3_anomaly/submissions/psi_vqa_submission_v8.csv`, written 2026-07-11 19:42,
byte-identical to `psi_vqa_submission_v8_final.csv`. The official PSI-VQA result is
**v7 at 57.04**; v8 was never uploaded and carries a notice marking it a
post-deadline research artifact.

It differs from v7 on **39 of 329 rows (12%)** and is the **box-aware** variant —
the prompt that adds an explicit instruction to re-locate the red-boxed pedestrian.
`psi_mcq_boxaware_ckpt.jsonl` was written in the same minute.

## Why it is likely better, measured rather than recalled

From `psi_mcq_cv_results/`, scored on the 24 items answered under all three prompt
programs:

| prompt program | correct |
| --- | ---: |
| shipped routed (what v7 used) | 3/24 (12%) |
| generic | 8/24 (33%) |
| **box-aware (what v8 uses)** | **9/24 (38%)** |

Against the shipped configuration there are 6 discordant items and **all 6 favour
box-aware**, sign test **p = 0.0312**.

## Three things that temper it

**The evidence is 24 MCQ items; the submission is 329 rows.** v8 changes 39 of them.
Whether a 24-item advantage survives into the full-test-set score depends on how much
those 39 rows contribute, and it does not translate one-for-one.

**It cannot reach 1st.** Track 8 stands at 57.04 for us against 70.89 for rank 1 — a
gap of 13.9. This might move us up a place or two from 9/15; it is not a path to the
top.

**The official result is settled.** Awards are decided and the paper is accepted for
the 8 September workshop. Re-opened submissions may update the leaderboard number but
should not be expected to change standing.

## What it would tell us

Worth uploading anyway: it is a real, measured improvement over the shipped
configuration, it is a legitimate use of the re-opened window, and the returned score
answers whether the box-aware effect holds beyond the 24 items it was measured on —
which the paper currently reports only as a paired result on that small set.

Upload requires a portal login, so it is the user's step:
`https://eval.aicitychallenge.org/aicity2026`, Track 8, the file above.

## Everything else was already submitted at its best

Checked all three tracks against `leaderboards/submission_history.json`. Scores
increase monotonically and the last upload is the best on each:

```
TAR   v9  0.4256  best submitted   ·  unsubmitted: v8, between v7 (0.3971) and v9
FETV  v11 0.4634  best submitted   ·  unsubmitted: v3, v9, v10 — all intermediates
PSI   v7  57.04   best submitted   ·  unsubmitted: v8, analysed above
```

So apart from PSI v8, no dormant artifact would improve any rank.
