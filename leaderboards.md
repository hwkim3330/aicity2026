# 2026 AI City Challenge — Leaderboard Snapshot

Team: **Korea Drive** / Team ID: **277**
Challenge ends: 2026-07-11 05:00:00
Note: scores based on 50% subset of test set; only top-3 + own best submission shown per track.

Snapshot captured: 2026-07-03

---

## Track 1 — Multi-Camera 3D Perception (Sim2Real)

| Rank | Team ID | Online | 3D HOTA (%) | DetA (%) | AssA (%) | LocA (%) |
|---|---|---|---|---|---|---|
| 1 | 34 | Yes | 53.3272 | 44.6537 | 58.2813 | 76.2957 |
| 2 | 130 | Yes | 37.3901 | 33.4902 | 36.4531 | 67.6457 |
| 3 | 133 | Yes | 24.7885 | 21.4645 | 30.4676 | 66.0915 |

Our status: working baseline in `/home/kim/aicity2026/track1_3dperception` (YOLO COCO-proxy + IoU/Kalman 2D track + homography 3D backproject + distance-based MTMC fusion). Currently fine-tuning YOLO11n on warehouse classes (Person/Forklift/PalletTruck) to replace weak COCO proxies — see that folder's README.md for architecture/limitations.

---

## Track 2 — Transportation Safety Understanding and Captioning (Sim2Real)

| Rank | Team ID | Models Used | S2 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 47 | vljepa | 58.1040 | 0.2409 | 0.4311 | 0.4493 | 0.5323 | 86.8453 |
| 2 | 24 | Qwen | 56.9324 | 0.2593 | 0.4169 | 0.4518 | 0.5502 | 84.2875 |
| 3 | 266 | QWEN-3-VL-8B | 56.3631 | 0.2557 | 0.4254 | 0.4459 | 0.7391 | 82.7040 |
| 4 (baseline) | N/A | TrafficInternVL | 47.1620 | 0.2339 | 0.4028 | 0.4282 | 0.3980 | 66.7073 |

---

## Traffic Anomaly Reasoning (TAR)

| Rank | Team ID | Models Used | Mean | BCQ | MCQ | BCQ OE F1 | MCQ OE F1 | Open QA F1 | Causal Linkage F1 | Scene Desc F1 | Temporal Desc F1 | Summarization F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 13 | Qwen3-VL-8B | 0.6382 | 0.9500 | 0.9250 | 0.6154 | 0.9578 | 0.4592 | 0.5040 | 0.4057 | 0.4241 | 0.5028 |
| 2 | 264 | Qwen3-VL-8B | 0.6381 | 0.9000 | 0.9500 | 0.6517 | 0.9561 | 0.4827 | 0.4802 | 0.4399 | 0.4084 | 0.4742 |
| 3 | 10 | Qwen3-VL-8B | 0.6373 | 0.9500 | 0.9250 | 0.5991 | 0.9578 | 0.4598 | 0.4944 | 0.4091 | 0.4351 | 0.5057 |
| 4 (baseline) | N/A | Cosmos3-Super | 0.5748 | 0.8250 | 0.8500 | 0.6034 | 0.8734 | 0.4235 | 0.4086 | 0.4222 | 0.3475 | 0.4196 |
| 5 (baseline) | N/A | Cosmos3-Nano | 0.4528 | 0.7375 | 0.8250 | 0.5762 | 0.5174 | 0.3879 | 0.3129 | 0.2558 | 0.2481 | 0.2144 |
| 6 | N/A | Qwen3.5-27B | 0.4035 | 0.7750 | 0.8000 | 0.4375 | 0.5338 | 0.3250 | 0.2666 | 0.2135 | 0.1368 | 0.1433 |
| 7 (baseline) | N/A | Cosmos-Reason2-32B | 0.3895 | 0.6625 | 0.7750 | 0.4373 | 0.5549 | 0.3203 | 0.2734 | 0.1660 | 0.2079 | 0.1078 |
| 8 | N/A | Gemini-3.1-Pro-Preview | 0.3681 | 0.6875 | 0.7750 | 0.4364 | 0.3575 | 0.3095 | 0.2193 | 0.2433 | 0.1048 | 0.1796 |
| 9 (baseline) | N/A | Cosmos-Reason2-8B | 0.3559 | 0.4375 | 0.7250 | 0.4121 | 0.7325 | 0.3042 | 0.1311 | 0.1808 | 0.1973 | 0.0824 |
| 10 | N/A | Gemma-4-31B-It | 0.3510 | 0.6250 | 0.7500 | 0.4852 | 0.3670 | 0.2580 | 0.2839 | 0.1644 | 0.1392 | 0.0862 |

(showing 1–10 of 16 rows; remaining 6 not captured)

---

## Track 4 — Text-Based Person Re-Identification (Sim2Real)

| Rank | Team ID | mAP (%) | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|
| 1 | 233 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 2 | 256 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| 3 | 9 | 98.1165 | 96.8655 | 99.5956 | 99.7978 |

---

## Generative Traffic Video Forecasting

| Rank | Team ID | Final Score | PSNR | SSIM | LPIPS | CLIP | FID | FVD |
|---|---|---|---|---|---|---|---|---|
| 1 | 78 | 73.1661 | 18.9483 | 0.6092 | 0.2727 | 0.9359 | 31.1354 | 22.4018 |
| 2 | 47 | 72.8417 | 18.9851 | 0.6325 | 0.2838 | 0.9363 | 33.9932 | 24.5775 |
| 3 | 209 | 72.5842 | 18.5505 | 0.5782 | 0.2761 | 0.9398 | 30.0188 | 25.2304 |

---

## Cross-City Object Detection (Milestone Project Hafnia)

| Rank | Team ID | Models Used | mAP | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l | AR@1 | AR@10 | AR@100 | AR_s | AR_m | AR_l |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 34 | RFDETR-AiO | 0.4592 | 0.5930 | 0.4949 | 0.0848 | 0.3011 | 0.5894 | 0.4206 | 0.6980 | 0.7416 | 0.3364 | 0.6151 | 0.8414 |
| 2 | 245 | DETR | 0.4504 | 0.5841 | 0.4828 | 0.0860 | 0.2950 | 0.5786 | 0.4240 | 0.6968 | 0.7394 | 0.2703 | 0.5915 | 0.8411 |
| 3 | 265 | detr | 0.4333 | 0.5600 | 0.4634 | 0.0674 | 0.2619 | 0.5718 | 0.4017 | 0.6448 | 0.6728 | 0.1734 | 0.4840 | 0.8044 |
| 4 (baseline) | N/A | RF-DETR nano, 8 epoch | 0.2350 | 0.3416 | 0.2439 | 0.0181 | 0.0804 | 0.3503 | 0.3082 | 0.4478 | 0.4609 | 0.0422 | 0.1981 | 0.6116 |

---

## Traffic Violation Understanding (FETV) — Track 3 Out-of-Domain Evaluation

Final result (full test set, 2026-07-11): **Korea Drive (Team 277) finished
3rd**. The final submitted candidate is preserved as
`track3_anomaly/submissions/fetv_submission_v11.json` (0.4634). UWIPL_ETRI
finished 1st.

| Rank | Team ID | Team | Final Score | Description Score | Categorical Mean |
|---|---|---|---|---|---|
| 1 | 30 | UWIPL_ETRI | 0.4891 | 0.4171 | 0.5612 |
| 2 | 139 | MR-CAS | 0.4884 | 0.4411 | 0.5358 |
| **3** | **277** | **Korea Drive** | **0.4634** | **0.4238** | **0.5031** |
| 4 | 257 | MobilityAI | 0.4525 | 0.4087 | 0.4963 |
| 5 | 309 | GDG | 0.4209 | 0.3582 | 0.4835 |
| 6 | 122 | OptimAI | 0.4108 | 0.3918 | 0.4298 |
| 7 | 97 | SMART Lab | 0.3905 | 0.3412 | 0.4397 |
| 8 | 60 | Vision | 0.2921 | 0.3582 | 0.2261 |

Earlier public-leaderboard snapshot retained below for development history.

| Rank | Team ID | Final Score | Description Score | Categorical Mean |
|---|---|---|---|---|
| 1 | 30 | 0.4513 | 0.4100 | 0.4926 |
| 2 | 257 | 0.4316 | 0.3889 | 0.4743 |
| 3 | 218 | 0.4175 | 0.3493 | 0.4856 |

---

## PSI-VQA: Pedestrian Situated Intent VQA — Track 3 Out-of-Domain Evaluation

Final full-test leaderboard (2026-07-11):

| Rank | Team ID | Team | Final Score | BCQ Macro-F1 | BCQ Accuracy | Open QA Cue-F1 | MCQ Accuracy | Temporal mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 | 30 | UWIPL_ETRI | 70.6397 | 0.7084 | 0.7273 | 0.5833 | 0.7912 | 0.7427 |
| 2 | 257 | MobilityAI | 69.0698 | 0.6136 | 0.6182 | 0.6674 | 0.7912 | 0.6906 |
| 3 | 122 | OptimAI | 65.5476 | 0.6464 | 0.6727 | 0.5846 | 0.7253 | 0.6656 |
| 4 | 139 | MR-CAS | 64.4161 | 0.5934 | 0.6000 | 0.6389 | 0.7692 | 0.5751 |
| **5** | **277** | **Korea Drive** | **57.0400** | **0.5045** | **0.5636** | **0.6019** | **0.6044** | **0.5708** |
| 6 | 97 | SMART Lab | 54.2445 | 0.5796 | 0.5818 | 0.5793 | 0.6154 | 0.3955 |
| 7 | 127 | Team KODE | 53.1456 | 0.5796 | 0.5818 | 0.5793 | 0.5714 | 0.3955 |

Post-deadline analysis produced `psi_vqa_submission_v8_final.csv`, combining
the submitted temporal prior with a two-cue OpenQA prior and box-aware MCQ
inference. It was **not submitted and has no official score**. Based on
held-out training-video validation, its estimated score was 65--68.5, but
that estimate must not be treated as a leaderboard result.

Earlier snapshot retained below for development history.

Snapshot captured: 2026-07-07 (updated -- team rankings shifted since 07-03 snapshot)

| Rank | Team ID | Final Score | BCQ Macro-F1 | BCQ Accuracy | Open QA Cue-F1 | MCQ Accuracy | Temporal mIoU |
|---|---|---|---|---|---|---|---|
| 1 | 139 | 62.3452 | 0.5159 | 0.5185 | 0.6458 | 0.8043 | 0.5278 |
| 2 | 122 | 59.6801 | 0.5235 | 0.5556 | 0.5967 | 0.6522 | 0.6148 |
| 3 | 97 | 44.3121 | 0.4000 | 0.6667 | 0.5612 | 0.4348 | 0.3765 |
| **4 (us, 277)** | **277** | **41.3421** | **0.4631** | **0.4815** | **0.6001** | **0.5652** | **0.0253** |

**Key gap**: BCQ/OpenQA/MCQ are all within striking distance of 3rd place (0.46/0.60/0.57 vs 0.40/0.56/0.43), but **Temporal mIoU is 0.0253 vs competitors' 0.38-0.71** -- roughly 15-25x lower. This is almost certainly the single highest-value fix for Track8: our temporal localization output format/logic is likely producing near-zero-overlap windows (e.g. wrong units, wrong reference frame, or a formatting bug), not a genuine model-quality gap given how competitive the other four metrics are.
