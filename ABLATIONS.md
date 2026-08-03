# Ablations and observations

Table 6 of the submitted paper mixed two kinds of evidence in one table. This
document separates them. **Section A** holds controlled experiments: one factor
varied, everything else held fixed, scored on labelled data with a stated
protocol. **Section B** holds submission-to-submission leaderboard movements,
which are not ablations and cannot attribute a change to any single component.

Nothing in Section B may be cited as evidence that a specific design choice
helped.

---

## A. Controlled local experiments

### A1. PSI temporal localization: what the score actually comes from

Same 227 labelled training clips, same 5 video-grouped splits, same seed. Every
parameterized method is refitted on each split's training half only and scored
on the held-out half. Full protocol, search spaces, and per-split fitted
parameters: [`track3_anomaly/analysis/temporal_prior_protocol.md`](track3_anomaly/analysis/temporal_prior_protocol.md).

| Method | Sees video? | Held-out mIoU |
|---|---|---:|
| Random interval | no | 0.2741 ± 0.0155 |
| **Qwen3-VL-8B, official temporal config** | **yes** | **0.4617 ± 0.0150** |
| Fixed centered window (fitted width) | no | 0.5200 ± 0.0299 |
| Global mean normalized interval | no | 0.5369 ± 0.0412 |
| VLM + duration prior, endpoint blend | yes | 0.5495 ± 0.0230 |
| **Duration-stratified prior (refit per split)** | **no** | **0.5566 ± 0.0307** |
| Shipped constants (fitted on all 227) | no | 0.5724 ± 0.0378 *(not held out)* |

Reproduce: `cd track3_anomaly/analysis && python3 temporal_prior_baselines.py --vlm-preds ../results/psi_temporal_vlm_train_preds.jsonl`

What it shows:

- The VLM is clearly above random, so it does extract real temporal signal.
- It is nonetheless beaten by a fixed centered window with one fitted
  parameter, and by every other metadata-only prior. On this benchmark the
  model's temporal output is worth less than the benchmark's duration
  statistics.
- Blending the VLM into the prior *hurts* (0.5495 vs 0.5566).
- Duration stratification contributes less than the framing in the paper
  implies: +0.0197 over a global mean interval, +0.0366 over a centered window.
  Most of the gain belongs to *any* metadata-only prior.
- The shipped constants score 0.5724 in-sample versus 0.5566 held out. The
  0.0158 gap is the optimism of fitting once on all 227 windows. The paper's
  0.553 figure is consistent with the honest held-out value, but its protocol
  was never stated.

### A2. PSI multiple choice: does task routing help?

Same held-out training items, same backbone, precision, frame budget, and
decoding. Only the prompt program varies. Items answered under all three
conditions form a paired subset of 24.

| Prompt program | Paired accuracy |
|---|---:|
| Generic Track 3 MCQ prompt (no routing) | 8/24 = 0.3333 |
| Shipped `psi_mcq` routed prompt | 3/24 = 0.1250 |
| Routed + explicit red-box re-location | 9/24 = 0.3750 |

Exact McNemar on discordant pairs:

| Comparison | discordant | p (two-sided) |
|---|---|---:|
| generic vs routed | 5–0 for generic | 0.0625 |
| routed vs box-aware | 0–6 for box-aware | 0.0312 |
| generic vs box-aware | 3–4 | 1.0000 |

Reproduce: `cd track3_anomaly/analysis && python3 prompt_ablation_psi_mcq.py`

What it shows: **task routing hurt PSI multiple choice.** The shipped
PSI-specific prompt lost to the generic shared prompt on every discordant item
(5–0), and adding an explicit instruction to re-locate the red-boxed pedestrian
recovered the loss (0–6, p = 0.031). The box-aware variant is statistically
indistinguishable from simply using the generic prompt. This is consistent with
the official PSI MCQ accuracy of 0.6044 being one of the system's weakest
components, and it is evidence *against* the paper's implicit claim that
routing was uniformly beneficial.

Caveat: 24 paired items is a small sample. The directions are informative; the
absolute accuracies are not precise.

### A3. TAR timestamp windowing

Same 25 timestamp-referencing multiple-choice items from the TAR training
split, same model and precision, in one process. The windowing function is
monkey-patched to a no-op for the control condition, so nothing else differs.

| Condition | Accuracy |
|---|---:|
| Frames sampled inside the question's `MM:SS` window | 16/25 = 0.640 |
| Uniform sampling over the whole clip | 15/25 = 0.600 |

Source: `track3_anomaly/eval_timewindow.log`, produced by
`track3_anomaly/scripts/eval_timewindow.py`.

What it shows: one item of difference. This is a clean A/B, but the effect is
not distinguishable from noise at n = 25 and should not be reported as a
demonstrated gain.

### A4. Open QA cue count

Video-grouped 5-fold CV over the PSI training split, greedy forward selection
of a fixed cue set per question direction, selected under the pessimistic
metric variant (one-to-one matching, macro averaging) and reported under all
four plausible variants of the unpublished official Cue-F1 implementation.

Held-out Cue-F1, means over 5 folds × 3 question directions
(`track3_anomaly/results/openqa_cue_count_cv.txt`):

| cues (k) | many-to-many + macro | many-to-many + micro | one-to-one + macro | one-to-one + micro |
|---:|---:|---:|---:|---:|
| 1 | 0.8002 | 0.8502 | 0.5819 | 0.6082 |
| **2** | 0.8504 | 0.8962 | **0.7816** | 0.8191 |
| 3 | 0.8559 | 0.9022 | 0.6618 | 0.6910 |
| 4 | 0.8394 | 0.8841 | 0.5660 | 0.5886 |

Reproduce: `cd track3_anomaly/scripts && python3 psi_openqa_prior_robust.py`

What it shows: k = 2 is the only cue count that stays strong under all four
variants. Under the permissive many-to-many variants k = 3 looks marginally
better, but under one-to-one matching it collapses from 0.7816 to 0.6618 —
extra cues start competing for the same ground-truth cue. Selection happens on
training folds only, so these held-out numbers are valid.

**They were also badly over-optimistic about the real metric.** The shipped
two-cue answers reached only 0.6019 official Open QA on the full test set, far
below the 0.78 this analysis projected. The local Cue-F1 was reimplemented from
the organizers' text description, not their actual scorer. Methodological
rigour — proper grouping, no leakage, pessimistic variant selection — protects
against overfitting noise; it does not protect against a wrong model of the
metric itself. This is the single most transferable lesson in the repository.

### A5. LoRA fine-tuning — abandoned, and not a clean comparison

A QLoRA adapter was trained on the released TAR annotations
(`track3_anomaly/lora_out_v1`, `scripts/build_finetune_data.py`,
`scripts/eval_lora.py`) and evaluated on a 77-item local proxy sample:

| Condition | Accuracy |
|---|---:|
| LoRA adapter | 49/77 = 0.636 (BCQ 22/40, MCQ 27/40) |
| Frozen backbone on the same sample | **not preserved in the repository logs** |

Source: `track3_anomaly/lora_eval_run.log`.

The adapter was judged worse than the frozen baseline at the time and no
fine-tuned checkpoint contributed to any submission. **The frozen-baseline
number for that exact sample was not kept**, so this row cannot be presented as
a controlled comparison. The nearest surviving frozen measurement
(`eval_fewshot_sc5.log`, 61/79 = 0.772) used few-shot prompting *and* 5-sample
self-consistency on a different sample, so it is not a valid control either.
Recorded here to document that parameter-efficient fine-tuning was attempted,
not to claim a measured margin.

---

## B. Uncontrolled submission-to-submission observations

These rows are historical leaderboard deltas, not controlled ablations.
Multiple configuration changes occurred simultaneously between submissions, so
improvements must not be attributed to any single component.

### B1. Rows carried over from the paper

| Comparison | Observation |
|---|---|
| Initial official temporal submission → final prior submission | 0.0253 → 0.5708 mIoU |
| Short BCQ output → reason-then-answer prompting (TAR) | 0.4625 → 0.5438 |
| Terse → fuller TAR summary prompting | 0.0993 → 0.3160 |

### B2. FETV submission sequence

Artifacts and dates are verifiable from the repository; portal scores are
recorded only where they were preserved.

| Artifact | Date | Portal score |
|---|---|---|
| `fetv_submission_v2_scored_0.3898.json` | 2026-07-06 | 0.3898 (encoded in the filename) |
| `fetv_submission_v3` … `v10` | 07-06 … 07-11 | not archived in this repository |
| `fetv_submission_v11.json` | 2026-07-11 | **0.4634 official full test, rank 3** |

The intermediate public-leaderboard scores were tracked during the challenge
but never written into the repository. They must be re-exported from the portal
before being cited.

### B3. PSI-VQA submission sequence

| Artifact | Date | Portal score |
|---|---|---|
| `psi_vqa_submission.csv` … `v6.csv` | 07-05 … 07-11 | not archived in this repository |
| `psi_vqa_submission_v7.csv` | 2026-07-11 | **57.0400 official full test, rank 5** |

`v7` differs from `v6` only in the 126 Open QA rows — see the
"What v7 actually is" section of [`REPRODUCE.md`](REPRODUCE.md).

### B4. Post-deadline, never submitted

| Artifact | Status |
|---|---|
| `psi_vqa_submission_v8_final.csv` | post-deadline research artifact; 39 MCQ rows differ from v7 |
| `scripts/fetv_structured_pipeline.py` | post-challenge prototype; needs an upstream detector/tracker/lane/OCR stack that was never built |

Neither has an official score, and neither may be presented as one.
