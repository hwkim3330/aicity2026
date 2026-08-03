# Leaderboard snapshots

Final full-test standings for the three Track 3 evaluations, exported from the
evaluation system on **2026-08-03** so the paper's rank claims carry a
denominator.

## The denominators

The portal keeps **two boards per track**, and they give different ranks:

| Evaluation | Public rank | General rank |
|---|---|---|
| Track 3 — TAR | **24 of 27** | 55 of 76 |
| Track 7 — FETV | **3 of 8** | 5 of 15 |
| Track 8 — PSI-VQA | **5 of 7** | 9 of 15 |

`public` counts only submissions marked Public; `general` counts every
submission. **The paper's ranks are the public board** — 24th, 3rd, and 5th all
reconcile exactly. Any rank reported without naming the board is ambiguous, so
both are stored in each file.

This settles the reviewers' question about whether the paper's tables were
complete: **Table 3 (FETV) is a top-5 excerpt of an 8-team board**, and
**Table 5 (PSI-VQA) is the complete 7-team board.**

Two other things the export confirms:

- The identical BCQ mF1 (0.5796) and Open QA (0.5793) values for SMART Lab and
  Team KODE in the paper's Table 5 are **genuine**, not a transcription slip.
  They differ on MCQ accuracy and temporal mIoU.
- Team names in the export match the paper's. The organizers' 2026-08-02 notice
  about updated team names does not appear to have changed any name relevant to
  these three boards, but the export postdates that notice, so these are the
  current names.

## Files

| File | Evaluation |
|---|---|
| `track3_tar_final.json` | Traffic Anomaly Reasoning |
| `track7_fetv_final.json` | Fisheye Traffic Violation Understanding |
| `track8_psi_vqa_final.json` | Pedestrian Situated Intent VQA |

Each holds both boards with every team's rank, id, final score, and full
component breakdown.

## Refreshing

The endpoint is public — no login required:

```bash
python3 scripts/fetch_leaderboards.py --snapshot-date $(date -I)
```

It reads
`https://eval.aicitychallenge.org/aicity2026/submission/leaderboard/stats/{public|general}/{3,7,8}`.
