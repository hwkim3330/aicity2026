# Result grounding used for v12

The task-wise summary and controlled-analysis claims were cross-checked against the public Korea Drive repository (`hwkim3330/aicity2026`).

## Official results
- TAR: 24 of 27, 0.4256, artifact `submission_qwen3vl8b_v9.csv`.
- FETV: 3 of 8, 0.4634, artifact `fetv_submission_v11.json`.
- PSI-VQA: 5 of 7, 57.0400, artifact `psi_vqa_submission_v7.csv`.

## Controlled analyses retained in the manuscript
- PSI temporal: Qwen3-VL official temporal configuration 0.4617 +/- 0.0150 held-out mIoU; duration-stratified metadata prior 0.5566 +/- 0.0307 under per-split held-out refitting.
- PSI MCQ prompt comparison: generic 8/24, shipped routed 3/24, box-aware routed 9/24.
- TAR timestamp-window A/B: 16/25 versus 15/25; treated as no demonstrated gain at this sample size.
- QLoRA: attempted during development, but the exact frozen control on the same sample was not preserved; therefore it is not presented as a controlled gain/loss.

Historical submission-to-submission score movements are not used to attribute causal improvements to a single prompt or module when multiple factors changed.
