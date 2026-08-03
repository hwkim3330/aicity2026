# Official AI City Challenge 2026 Results

Final leaderboard results, separated from local and post-deadline experiments.
Ranks are from the portal's **public** board; a separate `general` board exists
and gives different ranks (see [`leaderboards/`](leaderboards/)).

| Evaluation | Rank | Score | Official artifact |
|---|---|---|---|
| Track 3 TAR | **24 of 27** | 0.4256 | [`submission_qwen3vl8b_v9.csv`](track3_anomaly/submissions/submission_qwen3vl8b_v9.csv) |
| Track 7 FETV | **3 of 8** | 0.4634 | [`fetv_submission_v11.json`](track3_anomaly/submissions/fetv_submission_v11.json) |
| Track 8 PSI-VQA | **5 of 7** | 57.0400 | [`psi_vqa_submission_v7.csv`](track3_anomaly/submissions/psi_vqa_submission_v7.csv) |

FETV: description 0.4238, categorical mean 0.5031.
PSI-VQA: BCQ mF1 0.5045, Open QA Cue-F1 0.6019, MCQ 0.6044, temporal mIoU 0.5708.

All three artifacts were identified from the portal's Team 277 submission
pages, exported 2026-08-03 and stored in
[`leaderboards/submission_history.json`](leaderboards/submission_history.json).
Each is matched to its submission by the file mtime immediately preceding the
submission timestamp; the mapping is 1:1 on every track.

## The TAR backbone was Qwen3-VL-8B, not Qwen2.5-VL-7B

The portal records `models_used: qwen3` for the scored TAR submission. The only
Qwen2.5-VL 4-bit entry is `test` (General type, 0.3480, 2026-07-03), which was
never the scored submission.

**This contradicts the submitted paper**, whose abstract and Table 1 state that
the official TAR run used Qwen2.5-VL-7B with 4-bit inference. All three official
runs in fact used Qwen3-VL-8B-Instruct in bf16 through the same interface. The
paper's stated reason for declining a controlled cross-domain claim — differing
backbones and precision between TAR and the two out-of-domain runs — does not
hold. This must be corrected in the camera-ready.

## Research-only artifacts

`track3_anomaly/submissions/psi_vqa_submission_v8_final.csv` is a
**POST-DEADLINE RESEARCH ARTIFACT — NOT USED FOR THE OFFICIAL LEADERBOARD
RESULT**. `track3_anomaly/scripts/fetv_structured_pipeline.py` is a
**POST-CHALLENGE PROTOTYPE — NOT USED FOR THE OFFICIAL LEADERBOARD RESULT**.

`track3_anomaly/submissions/*.results.json` report `final_score: 1.0` because
they scored a submission against itself. They are not results; see the notice
file beside them.
