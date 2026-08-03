# `*.results.json` files are NOT scores

`fetv_submission_v11.results.json` and `fetv_submission_v10.results.json` report
`final_score: 1.0` with every per-field score at `1.0`. **That is not a result.**
These files were produced by scoring a submission against itself during
development of the local FETV harness, so every field trivially matches. They
carry no information about accuracy.

The real FETV result for `fetv_submission_v11.json` is:

| | |
|---|---|
| Official final score | **0.4634** (rank 3, full test set) |
| Description | 0.4238 |
| Categorical mean | 0.5031 |
| Weakest fields | violation type 0.1578, initial/final position 0.1239 / 0.1278 |

See [`../../OFFICIAL_RESULTS.md`](../../OFFICIAL_RESULTS.md) and
[`../../leaderboards/track7_fetv_final.json`](../../leaderboards/track7_fetv_final.json).

The `evaluation_subset` block inside those files is also inconsistent with
itself: it names `eval_subset_50.json` as the source while reporting
`selected_count: 100` of `total_count: 200`.

These files are retained only so this notice has something to point at. Do not
cite them, and do not let a reader mistake `1.0` for a score.
