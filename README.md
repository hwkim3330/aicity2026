# Korea Drive: AI City Challenge 2026

Research and submission code for Korea Drive (Team 277) across all eight
evaluation tracks of the 2026 AI City Challenge.

## Final results

| Track | Public rank | Primary score | Status |
|---:|---:|---:|---|
| 1 | 14 | 3D HOTA 5.4253 | Scored |
| 2 | 21 | S2 32.6404 | Scored |
| 3 | 24 | Mean 0.4256 | Scored |
| 4 | 27 | mAP 64.2005 | Scored |
| 5 | 10 | Final 70.3004 | Scored |
| 6 | - | - | Two submissions failed; no leaderboard result |
| 7 | **3** | **Final 0.4634** | Official podium finish |
| 8 | 5 | Final 57.0400 | Scored |

Ranks are the values returned for Team 277 by the final Public leaderboard
API. Track 6 had two before-deadline submissions, both marked `Failed`, and
therefore has no scored leaderboard row.

Track 8's official component scores were BCQ 0.5045, OpenQA 0.6019, MCQ
0.6044, and temporal mIoU 0.5708. The post-deadline candidate is retained
for research only and must not be represented as an official result.

Track 7's strongest fields were weather and lighting (1.0), date (1.0), time
(0.94), and intersection type (0.7841). Its main losses were violation type
(0.1578), positions (0.1239/0.1278), and lanes (0.1780/0.1694). See the
[`POSTMORTEM.md`](POSTMORTEM.md) for the redesign.

## Repository map

| Track | Task | Directory |
|---:|---|---|
| 1 | Multi-camera 3D perception | [`track1_3dperception`](track1_3dperception/) |
| 2 | Safety captioning and VQA | [`track2_captioning`](track2_captioning/) |
| 3 | Traffic anomaly reasoning | [`track3_anomaly`](track3_anomaly/) |
| 4 | Text-based person ReID | [`track4_reid`](track4_reid/) |
| 5 | Generative video forecasting | [`track5_forecasting`](track5_forecasting/) |
| 6 | Cross-city object detection | [`track6_detection`](track6_detection/) |
| 7 | FETV OOD evaluation | [`track3_anomaly`](track3_anomaly/) |
| 8 | PSI-VQA OOD evaluation | [`track3_anomaly`](track3_anomaly/) |

The repository excludes challenge datasets, model weights, virtual
environments, training caches, and large generated submissions. See
[`ARTIFACTS.md`](ARTIFACTS.md) for retained local artifacts and checksums.

## Reproducibility

Each track directory documents its own environment and commands. Track 3's
pipeline uses a unified Qwen-VL inference backend with task-specific prompts,
format parsing, temporal priors, and validation utilities. Public external
models and datasets must be cited and their licenses followed.

## Publication

A workshop-paper outline is available at [`paper/README.md`](paper/README.md).
It separates official full-test results from local cross-validation and
post-deadline analysis. The project report is published through GitHub Pages.
The detailed engineering retrospective is in [`POSTMORTEM.md`](POSTMORTEM.md).

## License and data

No challenge dataset is redistributed here. Dataset access remains subject to
the licenses of AI City Challenge, PSI-VQA, FETV, WTS, Hafnia, and their source
datasets. Code licensing should be confirmed before third-party reuse.
