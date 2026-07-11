# Korea Drive: AI City Challenge 2026

Research and submission code for Korea Drive (Team 277) across all eight
evaluation tracks of the 2026 AI City Challenge.

## Final highlights

| Evaluation | Result | Artifact |
|---|---:|---|
| Track 7, FETV out-of-domain | **3rd**, 0.4691 | `track3_anomaly/submissions/fetv_submission_v11.json` |
| Track 8, PSI-VQA out-of-domain | **5th**, 57.0400 | Official best described in `leaderboards.md` |
| Track 8 post-deadline candidate | Not submitted | `track3_anomaly/submissions/psi_vqa_submission_v8_final.csv` |

Track 8's official component scores were BCQ 0.5045, OpenQA 0.6019, MCQ
0.6044, and temporal mIoU 0.5708. The post-deadline candidate is retained
for research only and must not be represented as an official result.

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

## License and data

No challenge dataset is redistributed here. Dataset access remains subject to
the licenses of AI City Challenge, PSI-VQA, FETV, WTS, Hafnia, and their source
datasets. Code licensing should be confirmed before third-party reuse.
