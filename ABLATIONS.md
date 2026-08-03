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

### B0. TAR submission sequence

Retrieved from the portal's Track 3 submission history on 2026-08-03. Full test
set throughout. `models_used` is the free-text field recorded at upload.

| Submission | `models_used` | Artifact | Submitted | Mean | BCQ | MCQ | Summary F1 |
|---|---|---|---|---:|---:|---:|---:|
| `test` (General) | `qwen25vl_4bit` | `submission_qwen25vl_4bit.csv` | 07-03 16:34 | 0.3480 | 0.4813 | 0.5000 | 0.1059 |
| `qwen` | `qwen` | `submission_qwen3vl8b_v2.csv` | 07-05 19:35 | 0.3414 | 0.5687 | 0.4875 | 0.2157 |
| `1` (General) | `qwen` | `submission_qwen3vl8b_v3.csv` | 07-06 09:41 | 0.3859 | 0.5687 | 0.4875 | 0.2157 |
| `QWEN` | `QWEN` | `submission_qwen3vl8b_v4.csv` | 07-06 15:32 | 0.3956 | 0.5687 | 0.5750 | 0.2157 |
| `QWEN3` | `QWEN3` | `submission_qwen3vl8b_v5.csv` | 07-07 12:57 | 0.3862 | 0.5687 | 0.4250 | 0.2204 |
| `qwen4` | `qwen4` | `submission_qwen3vl8b_v6_fewshot.csv` | 07-09 09:32 | 0.3944 | 0.5938 | 0.4875 | 0.2253 |
| `kwen7` | `kwen7` | `submission_qwen3vl8b_v7.csv` | 07-10 09:47 | 0.3971 | 0.5563 | 0.5375 | 0.2253 |
| **`9`** | **`qwen3`** | **`submission_qwen3vl8b_v9.csv`** | **07-11 16:14** | **0.4256** | 0.5437 | 0.5875 | **0.3160** |

`submission_qwen3vl8b_v8.csv` was never submitted.

**The scored TAR run used Qwen3-VL-8B, not Qwen2.5-VL-7B at 4-bit.** The
submitted paper states otherwise in its abstract and Table 1. The single
Qwen2.5-VL 4-bit entry is `test`, a General submission that scored 0.3480 eight
days before the deadline.

The summarization F1 jump from 0.2253 to 0.3160 between `kwen7` and `9` is the
largest single component move in the sequence and is what carries the mean from
0.3971 to 0.4256 — but the two submissions differ in more than the summary
prompt, so it cannot be attributed to that change alone.

### B1. Rows carried over from the paper

| Comparison | Observation |
|---|---|
| Initial official temporal submission → final prior submission | 0.0253 → 0.5708 mIoU |
| Short BCQ output → reason-then-answer prompting (TAR) | 0.4625 → 0.5438 |
| Terse → fuller TAR summary prompting | 0.0993 → 0.3160 |

### B2. FETV submission sequence

Retrieved from the portal's Track 7 submission history on 2026-08-03. Every row
is a full-test-set score. Artifacts are matched to submissions by timestamp.

| Submission | Artifact | Submitted | Final | Violation type F1 | Violator type F1 | Categorical mean |
|---|---|---|---:|---:|---:|---:|
| `tr` | `fetv_submission_v2.json` | 07-05 21:33 | 0.3907 | 0.1820 | 0.4011 | 0.3993 |
| `V4` | `fetv_submission_v4.json` | 07-08 16:36 | 0.3960 | 0.1288 | 0.2256 | 0.4252 |
| `5` | `fetv_submission_v5.json` | 07-09 10:08 | 0.4063 | 0.1628 | 0.2823 | 0.4351 |
| `6` | `fetv_submission_v6_fewshot.json` | 07-10 14:39 | 0.4238 | 0.1535 | 0.2595 | 0.4830 |
| `7` | `fetv_submission_v7.json` | 07-11 07:51 | 0.4584 | 0.1535 | 0.2595 | 0.4960 |
| `8` | `fetv_submission_v8.json` | 07-11 10:58 | 0.4616 | 0.1908 | 0.2613 | 0.5016 |
| **`11`** | **`fetv_submission_v11.json`** | **07-11 18:05** | **0.4634** | 0.1578 | 0.3127 | 0.5031 |

`v9` and `v10` exist in the repository but were never submitted. One earlier
attempt (`teat`, 07-05 19:36) failed on the server.

Two things this corrects. The filename
`fetv_submission_v2_scored_0.3898.json` encodes 0.3898; the portal recorded
0.3907 for that submission. And v8 was **not** a regression — an earlier note in
this repository claimed v8 fell to 0.4505 against v7 at 0.4621, but both figures
came from development notes rather than the portal. v8 improved on v7, and v11
improved again.

Note how little of the movement is attributable to any single change: between
`6` and `7` the violation-type and violator-type F1 are byte-identical while the
final score moves 0.0346, which is the description opener and time-field work,
not the cascade fields.

### B3. PSI-VQA submission sequence

Retrieved from the portal's Track 8 submission history on 2026-08-03. Full test
set throughout.

| Submission | Submitted | Final | BCQ mF1 | Open QA Cue-F1 | MCQ Acc | Temporal mIoU |
|---|---|---:|---:|---:|---:|---:|
| `1` | 07-06 09:42 | 44.5735 | 0.5528 | 0.5970 | 0.6044 | 0.0287 |
| `2` | 07-07 13:12 | 49.5260 | 0.5528 | 0.5970 | 0.6044 | 0.2268 |
| `3` (General) | 07-08 16:52 | 55.4135 | 0.5528 | 0.5970 | 0.6044 | 0.4623 |
| `4` | 07-09 11:28 | 52.8314 | 0.5045 | 0.5970 | 0.5495 | 0.4623 |
| `5` | 07-09 11:34 | 54.2050 | 0.5045 | 0.5970 | 0.6044 | 0.4623 |
| `6` | 07-11 07:53 | 56.9175 | 0.5045 | 0.5970 | 0.6044 | 0.5708 |
| **`7`** | **07-11 16:15** | **57.0400** | 0.5045 | **0.6019** | 0.6044 | 0.5708 |

`v7` differs from `v6` only in the 126 Open QA rows — see the "What v7 actually
is" section of [`REPRODUCE.md`](REPRODUCE.md) — and the portal confirms it: Cue-F1
moved 0.5970 → 0.6019 while every other component stayed identical. That is
**+0.12 final points**, against the +4.5 to +7.5 the cross-validation in A4
projected. It is the cleanest single-factor row in this document, and it is the
one that most contradicts its own local validation.

Self-consistency voting (`4`) cost 0.55 MCQ accuracy points and was reverted in
`5`. The temporal prior (`6`) is the largest single move: mIoU 0.4623 → 0.5708.

### B4. Post-deadline, never submitted

| Artifact | Status |
|---|---|
| `psi_vqa_submission_v8_final.csv` | post-deadline research artifact; 39 MCQ rows differ from v7 |
| `scripts/fetv_structured_pipeline.py` | post-challenge prototype; needs an upstream detector/tracker/lane/OCR stack that was never built |

Neither has an official score, and neither may be presented as one.
