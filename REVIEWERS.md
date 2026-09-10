# Where to start, for anyone verifying Track 7

1. [`MODEL_EVIDENCE.md`](MODEL_EVIDENCE.md) — which model produced the FETV artifacts,
   and the one command that checks it. Re-running our published second pass regenerates
   the 2026-07-11 output byte-for-byte, including the raw model text for all 134 clips
   it queried; only the same weights can do that.
2. [`REPRODUCE.md`](REPRODUCE.md) — what reproduces and what does not. The scored FETV
   artifact does **not** reproduce from the clips. That section says so, with the nine
   explanations ruled out by measurement and the one that remains.
3. [`TRACK7_DQ_EVIDENCE.md`](TRACK7_DQ_EVIDENCE.md) — our own audit, including the parts
   that go against us.
4. [`track3_anomaly/scripts/build_fetv_clean.sh`](track3_anomaly/scripts/build_fetv_clean.sh)
   — the pipeline rebuilt so it runs end to end from the public clips through committed
   code, with no uncommitted step and no manual edit.

## Two files here were never submitted

`postchallenge_analysis/fetv_fix_intersection.py` and its output
`postchallenge_analysis/fetv_submission_v12.json` recover ground truth by inverting
our own leaderboard score, which the FAQ prohibits. They were written on **2026-09-08**,
after the final standings were published — `git log --diff-filter=A` on either path shows
it — and were excluded from every submission. They are kept visible rather than deleted
because they record a measurement we actually made. The legitimate counterpart, which
uses only the junction names burned into the video, is
`track3_anomaly/scripts/fetv_honest_intersection.py`.
