# Leaderboard snapshots

Final full-test standings for the three Track 3 evaluations, kept as data so
the paper's rank claims carry a denominator.

| File | Evaluation | Korea Drive | Teams listed |
|---|---|---|---|
| `track3_tar_final.json` | Traffic Anomaly Reasoning | rank 24, 0.4256 | denominator not archived |
| `track7_fetv_final.json` | Fisheye Traffic Violation Understanding | rank 3, 0.4634 | 5 |
| `track8_psi_vqa_final.json` | Pedestrian Situated Intent VQA | rank 5, 57.0400 | 7 |

## Provenance and caveats

**These rows were transcribed from the submitted paper's Tables 3 and 5, not
exported from the evaluation system.** Two consequences:

1. **Whether the FETV and PSI tables are complete leaderboards or top-N
   excerpts is not established.** Both files carry
   `"completeness": "unverified"`. If they are excerpts, "rank 3 of 5" and
   "rank 5 of 7" understate the field and must not be written as denominators
   in the camera-ready.
2. **Team names on the evaluation system were updated after the paper was
   submitted** (per the acceptance notice of 2026-08-02). The names stored here
   are pre-update names. They must be re-exported before the camera-ready.

**The TAR denominator was never archived.** Rank 24 is recorded without a
scored-team count rather than filled in with a guess.

Two rows in the PSI snapshot (`SMART Lab` and `Team KODE`) carry identical BCQ
mF1 and OpenQA values (0.5796 / 0.5793) while differing on MCQ and final score.
That may be genuine or may be a transcription slip in the source table; it is
flagged in the file and should be checked against a fresh export.

## Refreshing these files

Export the final standings for each evaluation from
<https://eval.aicitychallenge.org/aicity2026/> while signed in as Team 277,
then update the `teams` array, set `"completeness": "complete"` if the export
is the full table, and record the export date in `snapshot_date`.
