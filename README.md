# Korea Drive: AI City Challenge 2026

Research and submission code for Korea Drive (Team 277) across all eight
evaluation tracks of the 2026 AI City Challenge.

## Final results

| Track | Public rank | Status |
|---:|---:|---|
| 1 | 14 | Scored |
| 2 | 21 | Scored |
| 3 | 24 | Scored |
| 4 | 27 | Scored |
| 5 | 10 | Scored |
| 6 | - | Hafnia benchmark completed; two portal evaluations failed |
| 7 | **3** | **Official podium finish** |
| 8 | 5 | Scored |

Ranks are the values returned for Team 277 by the final Public leaderboard
API. Track 6 had two before-deadline submissions, both marked `Failed`, and
therefore has no scored leaderboard row.

### Track 1 · Multi-camera 3D perception

| 3D HOTA | DetA | AssA | LocA | Online |
|---:|---:|---:|---:|---|
| 5.4253 | 5.1081 | 6.6976 | 47.6221 | No |

### Track 2 · Safety captioning and VQA

| S2 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 32.6404 | 0.1264 | 0.3102 | 0.3355 | 0.5108 | 44.7012 |

### Track 3 · Traffic anomaly reasoning

| Mean | BCQ | MCQ | BCQ OE F1 | MCQ OE F1 | Open QA F1 |
|---:|---:|---:|---:|---:|---:|
| 0.4256 | 0.5438 | 0.5875 | 0.4824 | 0.7664 | 0.3333 |

| Causal F1 | Scene F1 | Temporal desc. F1 | Summary F1 | Temporal mIoU* |
|---:|---:|---:|---:|---:|
| 0.2874 | 0.3282 | 0.1856 | 0.3160 | 0.1990 |

`*` Temporal localization was reported by the API but excluded from the final
Track 3 mean after the July 1 rule update.

### Track 4 · Text-based person ReID

| mAP | R@1 | R@5 | R@10 |
|---:|---:|---:|---:|
| 64.2005 | 51.1122 | 81.0415 | 88.0688 |

### Track 5 · Generative video forecasting

| Final | PSNR | SSIM | LPIPS | CLIP | FID | FVD |
|---:|---:|---:|---:|---:|---:|---:|
| 70.3004 | 19.3013 | 0.6198 | 0.2789 | 0.9590 | 49.7246 | 33.6228 |

### Track 6 · Cross-city object detection

The RF-DETR Hafnia experiment and hidden benchmark inference completed, but
both AI City portal evaluations were marked `Failed`; no mAP metrics exist.
The reconstructed file contained 275,159 detections for 14,814 of 14,925
benchmark images. It was a custom flat COCO-style list recovered from logs,
not a verified Hafnia-generated evaluator artifact. See the postmortem.

### Track 7 · FETV out-of-domain

| Final | Description | Categorical mean |
|---:|---:|---:|
| **0.4634** | 0.4238 | 0.5031 |

| Violation | Violator | Color | Start pos. | End pos. | Start lane | End lane |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1578 | 0.3127 | 0.2434 | 0.1239 | 0.1278 | 0.1780 | 0.1694 |

| Intersection | Weather | Light | Date | Time |
|---:|---:|---:|---:|---:|
| 0.7841 | 1.0000 | 1.0000 | 1.0000 | 0.9400 |

### Track 8 · PSI-VQA out-of-domain

| Final | BCQ Macro-F1 | BCQ Accuracy | Open QA Cue-F1 | MCQ Accuracy | Temporal mIoU |
|---:|---:|---:|---:|---:|---:|
| 57.0400 | 0.5045 | 0.5636 | 0.6019 | 0.6044 | 0.5708 |

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
