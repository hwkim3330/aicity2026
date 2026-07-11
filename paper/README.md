# Paper Draft: Generalizing Traffic Video Reasoning Across Domains

## Proposed title

**A Unified Video-Language Pipeline for Out-of-Domain Traffic Violation and
Pedestrian-Intent Reasoning**

## Core claim

A task-routed but unified VLM pipeline can transfer from traffic anomaly
reasoning to fisheye violation understanding and egocentric pedestrian intent.
Statistical priors provide a strong, reproducible complement for structured
temporal outputs, while cue-aware prompting improves ambiguous-intent tasks.

## Evidence available

- Official FETV full-test result: 3rd place, 0.4634 (description 0.4238,
  categorical mean 0.5031).
- Official PSI-VQA full-test result: 5th place, 57.0400.
- PSI temporal mIoU improved from an early 0.0253 to the final 0.5708.
- Post-deadline OpenQA and MCQ experiments are local validation only.

## Suggested structure

1. Introduction and cross-domain motivation.
2. Unified Qwen-VL inference architecture and task routing.
3. FETV structured prediction, vocabulary normalization, and posterior study.
4. PSI-VQA red-box localization, cue prompting, and temporal prior.
5. Official results, ablations, failure analysis, and deadline-safe inference.
6. Reproducibility, licenses, limitations, and ethical considerations.

## Required work before submission

- Re-run all reported ablations from clean environments.
- Record exact model revisions, prompts, seeds, hardware, and runtime.
- Confirm that every method satisfies the Track 3 unified-system rule.
- Add qualitative examples using only redistributable or properly licensed media.
- Cite the AI City Challenge Track 3, FETV, PSI 2.0/PSI-VQA, and base-model papers.
- Never report `psi_vqa_submission_v8_final.csv` as an official submission.
