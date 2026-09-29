# Track 7 (FETV) review — resolved

**Outcome (2026-09-29).** The AI City Challenge Organizing Committee confirmed that
Team 277 used the same frozen `Qwen/Qwen3-VL-8B-Instruct` model across the relevant
tracks, without task-specific adapters or parameter updates, and that the submission
therefore satisfies the unified-system requirement. On the verified Track 7 score of
**0.4618**, Korea Drive is listed **third**, and the committee is correcting the
official Track 7 ranking accordingly.

The public leaderboard score of the same submission is 0.4634; 0.4618 is the
organizers' verified score. Quote whichever you mean, and say which it is.

## Timeline

| Date | Event |
|---|---|
| 2026-07-11 | FETV submission scored 0.4634 on the public board (3rd of 8) |
| 2026-09-08 | Final standings published without Team 277; told at the workshop the entry was removed for using a different model |
| 2026-09-09 | Team wrote to the committee with the model-identity evidence below |
| 2026-09-29 | Committee confirmed the unified system and reinstated the entry in third place, verified score 0.4618 |

## The evidence the review used

1. [`MODEL_EVIDENCE.md`](MODEL_EVIDENCE.md) — which model produced the FETV artifacts,
   and the one command that checks it. Re-running our published second pass regenerates
   the 2026-07-11 output byte-for-byte, including the raw model text for all 134 clips
   it queried; only the same weights can do that.
2. [`../../REPRODUCE.md`](../../REPRODUCE.md) — what reproduces and what does not. The
   scored FETV artifact does **not** reproduce from the clips end to end. That section
   says so, with the explanations ruled out by measurement and the one that remains.
3. [`EVIDENCE.md`](EVIDENCE.md) — our own audit written during the review, question by
   question, including the parts that go against us.
4. [`../../track3_anomaly/scripts/build_fetv_clean.sh`](../../track3_anomaly/scripts/build_fetv_clean.sh)
   — the pipeline rebuilt so it runs end to end from the public clips through committed
   code, with no uncommitted step and no manual edit.

## Two files in this repository were never submitted

`postchallenge_analysis/fetv_fix_intersection.py` and its output
`postchallenge_analysis/fetv_submission_v12.json` recover ground truth by inverting
our own leaderboard score, which the FAQ prohibits. They were written on **2026-09-08**,
after the final standings were published — `git log --diff-filter=A` on either path shows
it — and were excluded from every submission. They are kept visible rather than deleted
because they record a measurement we actually made. The legitimate counterpart, which
uses only the junction names burned into the video, is
`track3_anomaly/scripts/fetv_honest_intersection.py`.
