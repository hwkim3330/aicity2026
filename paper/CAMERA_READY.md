# Camera-ready instructions

Everything needed to revise the accepted workshop paper. Self-contained: every
number below is traceable to a file in this repository, and every file is
named. **Do not use figures from the conversation history or from memory — use
only what is cited here.**

- **Paper**: "Task-Routed Video-Language Reasoning Across Traffic Domains:
  Korea Drive at AI City Challenge 2026", OpenReview `niE4UrOp7I`, submission 4.
- **Decision**: Accept (Poster), 2026-08-02. Ratings 4 / 4 / 3 / 2.
- **Deadline**: camera-ready to OpenReview by **2026-08-15** (Edit → "Camera
  Ready Revision").
- **Submitted PDF**: [`submitted/korea_drive_aicity2026_submitted.pdf`](submitted/korea_drive_aicity2026_submitted.pdf)
- **Also required**: full author registration by 2026-08-10 CEST; the award
  repository was shared with the organizers by 2026-08-07 AoE.

Cite the workshop summary paper:

```bibtex
@InProceedings{Tang26AICity26,
  author    = {Tang, Zheng and Wang, Shuo and Anastasiu, David C. and Chang, Ming-Ching and others},
  title     = {The 10th {AI City Challenge}},
  booktitle = {ECCV Workshops},
  year      = {2026},
  address   = {Malm{\"o}, Sweden}
}
```

---

## 1. Factual error that must be fixed

**The paper states the wrong backbone for the in-domain TAR run.** The abstract
says "The official TAR run used Qwen2.5-VL-7B with 4-bit inference", and
Table 1 repeats it.

The evaluation portal records `models_used: qwen3` for the scored TAR
submission (submission `9`, 2026-07-11 16:14, mean 0.4256). The only
Qwen2.5-VL 4-bit entry is `test`, a *General*-type submission that scored
0.3480 on 2026-07-03 and was never the scored entry. All eight TAR submissions
map 1:1 to repository artifacts by timestamp.

**All three official runs used `Qwen/Qwen3-VL-8B-Instruct` in bf16**, 16 frames,
151,200 pixels per frame.

Evidence: [`../leaderboards/submission_history.json`](../leaderboards/submission_history.json),
[`../OFFICIAL_RESULTS.md`](../OFFICIAL_RESULTS.md).

### This changes an argument, not just a sentence

Section 2 says: *"The official submissions did not use one identical
checkpoint... We therefore do not claim a controlled checkpoint-level domain-
generalization experiment."* Section 6 ("Scope of the generalization claim")
repeats it, and Reviewer JZh931 flagged precision as a second confound.

**That confound does not exist.** One checkpoint, one precision, three
benchmarks. Rewrite both passages: the checkpoint-level cross-domain comparison
*is* controlled. The honest remaining limitation is narrower — some development
decisions were informed by sequential leaderboard feedback, and the three
benchmarks differ in task structure and output space, not in the model.

Update Table 1's TAR row accordingly: `Qwen3-VL-8B-Instruct; bf16; 16 frames;
151,200 pixels/frame` with task-dependent token budgets.

---

## 2. Rank denominators (Reviewers JZh931, GRbC3)

Reviewer JZh931 asked directly: *"How many teams were on each leaderboard, and
are Tables 3 and 5 complete or truncated?"*

The portal keeps **two boards per track**, giving different ranks for identical
scores. The paper's ranks are the **public** board.

| Evaluation | Public board | General board |
|---|---|---|
| Track 3 TAR | **24 of 27** | 55 of 76 |
| Track 7 FETV | **3 of 8** | 5 of 15 |
| Track 8 PSI-VQA | **5 of 7** | 9 of 15 |

- **Table 3 (FETV) is a top-5 excerpt of an 8-team board.** Either complete it
  or say it is truncated in the caption.
- **Table 5 (PSI-VQA) is the complete 7-team board.** Say so in the caption.
- State in every rank claim which board it is. "3rd of 8 on the public
  leaderboard" is the correct form.

The two missing FETV rows: rank 6 OptimAI 0.4108 (desc 0.3918, cat 0.4298),
rank 7 SMART Lab 0.3905 (desc 0.3412, cat 0.4397), rank 8 FPT AI Vision 0.2921
(desc 0.3582, cat 0.2261).

One thing **not** to correct: the identical BCQ mF1 (0.5796) and Open QA
(0.5793) values for SMART Lab and Team KODE in Table 5 are genuine, confirmed
against the portal export.

Evidence: [`../leaderboards/`](../leaderboards/), regenerate with
`python3 scripts/fetch_leaderboards.py`.

---

## 3. Complete TAR breakdown (Reviewer JZh931)

Reviewer JZh931: *"Table 2 reports three TAR components when ten task types are
scored... the three shown all exceed the 0.4256 mean, implying the rest average
near 0.32."* They were right. Replace Table 2's TAR row with all ten:

| Task type | Score |
|---|---:|
| BCQ accuracy | 0.5437 |
| MCQ accuracy | 0.5875 |
| BCQ-OE BERTScore-F1 | 0.4824 |
| MCQ-OE BERTScore-F1 | 0.7664 |
| Open QA F1 | 0.3333 |
| Causal linkage F1 | 0.2874 |
| Scene description F1 | 0.3282 |
| Temporal description F1 | 0.1856 |
| Video summarization F1 | 0.3160 |
| Temporal localization mIoU* | 0.1990 |

`*` excluded from the mean per the organizers' 2026-07-01 rule update. The nine
scored components average **0.42561**, reconstructing the official 0.4256.

Note: BCQ is 0.5437 on the portal. An earlier repository revision said 0.5438.

Define **MCQ-OE** at first use as *multiple-choice open-ended*.

---

## 4. The temporal prior — protocol and metadata-only baselines

This is the most-criticized part of the paper. Reviewer WC8q asked for
comparison against *"fixed center windows, average normalized intervals, and
random priors"*. Reviewer JZh931 asked *"were the eight splits refit on the
training half only, and what was the variance?"* Reviewers vmz20 and GRbC3 both
raised it.

All of it now exists. **Replace the two Table 6 temporal rows with this table.**

5 video-grouped splits over the 227 labelled PSI training clips, seed 42. Every
parameterized method is refitted on each split's training half only.

| Method | Sees video? | Held-out mIoU |
|---|---|---:|
| Random interval | no | 0.2741 ± 0.0155 |
| **Qwen3-VL-8B, official temporal config** | **yes** | **0.4617 ± 0.0150** |
| Fixed centered window (fitted width) | no | 0.5200 ± 0.0299 |
| Global mean normalized interval | no | 0.5369 ± 0.0412 |
| VLM + duration prior, endpoint blend | yes | 0.5495 ± 0.0230 |
| **Duration-stratified prior (refit per split)** | **no** | **0.5566 ± 0.0307** |
| Shipped constants (fitted on all 227) | no | 0.5724 ± 0.0378 *(not held out)* |

What to say:

1. **Answer JZh931 explicitly.** The originally reported 0.553 came from eight
   50/50 splits whose protocol was never stated. Under a stated protocol with
   per-split refitting the value is **0.5566 ± 0.0307**. The shipped constants,
   fitted once on all 227 windows, reach 0.5724 on the same splits — the
   0.0158 gap is that fit's optimism.
2. **Answer WC8q directly.** All three requested baselines are in the table. A
   fixed centered window with one fitted parameter already reaches 0.5200.
   Duration stratification adds only +0.0197 over a global mean interval and
   +0.0366 over the centered window. Most of the effect belongs to *any*
   metadata-only prior, not to stratification.
3. **Soften the "no video understanding" framing.** The VLM scores 0.4617,
   clearly above random's 0.2741, so it does carry real temporal signal. The
   defensible claim is that *on this benchmark the model's temporal output is
   worth less than the benchmark's own duration statistics* — not that the model
   is blind.
4. **Blending hurts**: 0.5495 versus 0.5566 for the prior alone. This
   reproduces the development-time observation under a proper protocol.
5. **Publish the search space** (a sentence or a footnote): threshold
   `T ∈ {6..14} s`; stratum ratios `lo, hi ∈ [0, 1]` step 0.02 with `hi > lo`;
   centered-window width `∈ [0.05, 1.00]` step 0.01; objective is mean IoU on
   the split's training half.
6. **Report parameter stability.** Short-clip `lo` stays in 0.28–0.32 and
   long-clip `lo` in 0.20–0.22 across all five splits, but the best threshold
   swings between 7 s and 12 s because the short stratum holds only ~30 clips.
   The threshold is weakly identified; say so.

Remove the word "controlled" from contribution 3 as it was written, and point
it at this table instead — now it is earned.

Evidence: [`../track3_anomaly/analysis/temporal_prior_protocol.md`](../track3_anomaly/analysis/temporal_prior_protocol.md),
[`../track3_anomaly/results/temporal_prior_cv.csv`](../track3_anomaly/results/temporal_prior_cv.csv).
Reproduce in ~25 s on CPU:
`cd track3_anomaly/analysis && python3 temporal_prior_baselines.py --vlm-preds ../results/psi_temporal_vlm_train_preds.jsonl`

---

## 5. Split Table 6 (Reviewers vmz20, WC8q, GRbC3)

Reviewer vmz20: *"Table 6 mixes controlled training-set diagnostics with
intermediate and final leaderboard submissions... The uncontrolled rows should
not be used to attribute improvements specifically to prompt changes."*

Make **two tables**.

**Table 6a — controlled experiments.** State split, protocol, and what varies.

| Experiment | Protocol | Result |
|---|---|---|
| Temporal prior vs metadata-only baselines vs VLM | 227 labelled clips, 5 video-grouped splits, refit per split | see §4 |
| PSI MCQ prompt program | paired held-out items, identical model/precision/frames/decoding | see §6 |
| TAR timestamp windowing | same 25 items, windowing monkey-patched off for control | 16/25 vs 15/25 |
| Open QA cue count | video-grouped 5-fold, greedy selection on train folds | k=2 best under all metric variants |

For the TAR windowing row, say plainly that one item of difference at n = 25 is
not distinguishable from noise. It is a clean A/B with no measurable effect.

**Table 6b — submission-to-submission observations.** Prefix with: *"These rows
are historical submission deltas, not controlled ablations. Multiple
configuration changes occurred simultaneously, so improvements must not be
attributed to any single component."*

Full per-component submission histories for all three tracks are in
[`../leaderboards/submission_history.json`](../leaderboards/submission_history.json)
and tabulated in [`../ABLATIONS.md`](../ABLATIONS.md) §B0–B3.

**Two corrections to scores previously used:**

- FETV v8 scored **0.4616** and did **not** regress against v7's **0.4584**.
  Earlier notes claimed 0.4505 vs 0.4621; both were wrong.
- The FETV sequence is: `tr` 0.3907 → `V4` 0.3960 → `5` 0.4063 → `6` 0.4238 →
  `7` 0.4584 → `8` 0.4616 → `11` **0.4634**. Versions 9 and 10 were never
  submitted.
- The PSI sequence is: 44.5735 → 49.5260 → 55.4135 → 52.8314 → 54.2050 →
  56.9175 → **57.0400**.

One row is worth promoting because it is genuinely single-factor: **v6 → v7
changed only the Open QA rows.** Cue-F1 moved 0.5970 → 0.6019 and every other
component is byte-identical. That is **+0.12 final points**, against the +4.5 to
+7.5 its own cross-validation projected (§7). It is the cleanest evidence in the
paper for how far a locally-validated estimate can miss.

---

## 6. New controlled ablation: task routing hurt PSI MCQ

Reviewer vmz20's first weakness: *"There are no controlled ablation comparisons
against a shared generic prompt, a system without routing..."*

There is one now, and it goes against the paper. Same backbone, precision,
frame budget and decoding; only the prompt program varies; paired subset of 24
held-out training items.

| Prompt program | Paired accuracy |
|---|---:|
| Generic Track 3 MCQ prompt (no routing) | 8/24 = 0.3333 |
| Shipped `psi_mcq` routed prompt | 3/24 = 0.1250 |
| Routed + explicit red-box re-location | 9/24 = 0.3750 |

Exact McNemar on discordant pairs: generic vs routed **5–0 for generic**
(p = 0.0625); routed vs box-aware **0–6 for box-aware** (p = 0.0312); generic
vs box-aware 3–4 (p = 1.0).

Report this. It is a real negative result: the PSI-specific routed prompt lost
to the generic shared prompt on every discordant item, and an explicit
re-location instruction recovered the loss. It is consistent with MCQ accuracy
(0.6044) being one of the system's weakest official components. State the
caveat: 24 paired items, so directions are informative and absolute
accuracies are not precise.

This narrows the paper's central claim in a useful way — routing helped where
the output contract was the bottleneck, and hurt where it added reasoning steps
the model then mishandled.

Evidence: [`../ABLATIONS.md`](../ABLATIONS.md) §A2,
[`../track3_anomaly/results/prompt_ablation_psi_mcq.json`](../track3_anomaly/results/prompt_ablation_psi_mcq.json).
Reproduce: `cd track3_anomaly/analysis && python3 prompt_ablation_psi_mcq.py`

---

## 7. New finding: the output contract fails silently

15–25% of PSI MCQ generations produce **no parseable final answer** under every
prompt program tried (routed 7/47, generic 6/24, box-aware 4/24). Each is
silently replaced by a type-correct fallback, so contract violations are scored
as ordinary wrong answers and never surface.

This matters because §4.2 lists metric-compatible output contracts and strict
parsing as a system contribution. Report the rate honestly. Caveat: these are
the deliberately-hard PSI *ambiguous* split, so the rate is not the rate on the
official test set.

---

## 8. Qualitative evidence (Reviewers JZh931, vmz20)

Reviewer JZh931: *"For a paper arguing that violator mis-selection drives the
errors, there is no frame, predicted record or failure case anywhere; one worked
example would be worth more than the diagram."*

Two cases are prepared in [`../docs/case_studies/`](../docs/case_studies/).
**Replace Figure 1** (the box-and-arrow diagram, which JZh931 said "doesn't earn
its space") with one of them.

**Case A — FETV cascade coupling, supports the paper's claim.** On clip
`019_004.mp4`, `answer_violation_type` moves `no_violation → red_light` between
two versions of the same system and all six dependent fields move with it
(violator `na → car`, color `na → yellow`, positions `na → Top-Left` /
`Middle-Center`, lanes `na → 1` / `2`), while date, time, weather, lighting and
intersection type are identical. Across the 200-clip test set, **51 clips flip
the entire actor-centric block as one unit**; only 16 of the 67 clips with any
dependent-field change move partially. The seven low-scoring fields are
effectively one decision, so averaging them as independent fields understates
how concentrated the failure is.

**Case B — PSI, contradicts the paper's claim.** Section 6 attributes PSI
failures to losing the marked pedestrian's identity. On held-out item
`e686c1a878234aa1` the model tracks the pedestrian correctly and describes
ground-truth option D almost verbatim ("standing on the sidewalk... does not
move into the roadway"), then the routed elimination step mis-negates that same
option ("The pedestrian is not standing still and not moving into the roadway;
they are standing still on the sidewalk"), loops, and is cut off mid-sentence
before the required `Final answer:` line. Identity was never lost; geometry was
never needed.

Including Case B is the stronger move. It converts a plausible-but-unsupported
diagnosis into a documented one and pairs with §7.

Frames are not in the repository (FETV is portal-distributed; PSI-VQA inherits
the TASI Benchmark Data Sharing Agreement). Render locally:
`python3 scripts/render_case_studies.py --case <id> --data-root <path>`

---

## 9. Efficiency figures (Reviewer JZh931)

*"Efficiency claim carries no runtime or peak-memory figure."* Measured on the
227-clip PSI temporal run:

| | |
|---|---|
| Hardware | 1 × RTX 3090 24 GB, driver 580.173.02 |
| Model | Qwen3-VL-8B-Instruct, bf16, 16 frames, 151,200 px/frame |
| Throughput | **2.07 s/item** (227 items, 469 s, 0 errors) |
| Model load | 18.7 s |
| **Peak VRAM** | **16.78 GiB** (16.33 GiB at load; decode and generation add ~0.45 GiB) |

Label these as reproduction-run measurements taken 2026-08-03 — the original
challenge runs logged neither runtime nor memory. Full table and estimates for
the other workloads: [`../BENCHMARKS.md`](../BENCHMARKS.md).

---

## 10. Smaller editorial items

- **Eq. 1**: cite CIDEr (Vedantam et al., CVPR 2015). Reviewer JZh931 noted it
  is uncited.
- **References**: normalize "Qwen3-VL", "Qwen2.5-VL", "QLoRA", "Sentence-BERT"
  capitalization (Reviewer WC8q).
- **Notation**: use `bf16` and `mIoU` consistently throughout.
- **Related work** (Reviewers vmz20, JZh931): thin. Add recent traffic-video
  reasoning datasets and models, and position the system against existing
  constrained-decoding and structured-output work, which is what the output
  contracts actually are.
- **Team names**: the organizers said evaluation-system team names were updated
  on 2026-08-02. The export in `leaderboards/` postdates that notice, and the
  names match those in the paper, so no change appears necessary — but the
  export is the authority, not the submitted PDF.

---

## 11. Do not claim

- Do not present `psi_vqa_submission_v8_final.csv` or
  `fetv_structured_pipeline.py` as official results. Both are post-deadline.
- Do not cite any `*.results.json` file. They report `final_score: 1.0` because
  they scored a submission against itself.
- ~~Do not report a Hub revision/commit for any official run.~~ **Superseded.**
  None was persisted at run time, but the qwen3 one was recovered afterwards:
  `Qwen/Qwen3-VL-8B-Instruct` = `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`,
  because the Hub repo has had no commit since 2025-10-15 and the run resolved
  `main`. Safe to state; it is now pinned in the code. The Qwen2.5-VL-7B
  General entry turned out to be recorded too, in
  `track3_anomaly/model_download.log`:
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, still Hub `main` and unchanged
  since 2025-04-06. Both are now known; neither is guessed. See
  [`../REPRODUCE.md`](../REPRODUCE.md#why-the-revision-is-certain).
  *Nothing in the submitted PDF asserts a revision, so this needs no PDF edit.*
- Do not claim byte-level reproduction of the PSI artifact. The 55 BCQ rows use
  unseeded 5-sample voting, which is permanent — no seed was ever generated to
  recover. The same applies to TAR's `bcq_accuracy`.
- Do not claim byte-level reproduction of FETV either. It was attempted on
  2026-08-13 with the revision pinned and **0 of 200 records matched**: the
  shipped v11 is the last of an eleven-version chain, and `answer_time` still
  differs (31/200) even where no chain step touched the field. Two identical
  reruns agree bit-for-bit and six explanations were tested and ruled out
  (nondeterminism, the determinism pin, clip encoding, frame sampling, the
  few-shot exemplars, the NumPy upgrade). The cause is structural: one committed
  code state covers nine artifacts produced across six days. Do not describe the
  gap as run-to-run variance, and do not offer a mechanism the repository has not
  established.
  See [`../REPRODUCE.md`](../REPRODUCE.md).
- **Disclose the FETV v9 → v10 step, and bound it.** Fifteen rows were edited
  and all fifteen fall inside the 100 clips FETV scores, while `no_violation`
  rows split 54 inside / 53 outside — p = 3.05e-05 under a content-driven null.
  The repository also holds `fetv_gt_posterior.py`, which anneals against
  recorded leaderboard macro-F1 to reconstruct those clips' hidden labels over
  those exact fields; its output was not retained, so its role cannot be
  established either way. State the bound in the same breath: the submitted
  step containing it is v8 → v11, worth **+0.0018** (0.4616 → 0.4634) against a
  **+0.0110** margin over rank 4, so third place does not depend on it. This is
  more specific than the paper's current "informed by sequential leaderboard
  feedback" and should replace it. A reviewer opening the repository sees the
  script and the artifacts; better that we characterised it first.
  Evidence: [`../ABLATIONS.md`](../ABLATIONS.md) §B2.
- Do not cite any score that is not traceable to
  `leaderboards/submission_history.json`. Several figures previously recorded
  in this repository were wrong.
- Do not upgrade the Section 7 prototypes into evaluated systems. Reviewer
  vmz20 correctly notes the FETV prototype assumes tracked boxes and lane
  polygons that were never implemented, and the PSI result rests on 24 held-out
  examples.

---

## 12. Where everything lives

| Need | File |
|---|---|
| Official artifacts, ranks, backbone correction | [`../OFFICIAL_RESULTS.md`](../OFFICIAL_RESULTS.md) |
| Controlled ablations vs submission deltas | [`../ABLATIONS.md`](../ABLATIONS.md) |
| Temporal prior protocol and per-split parameters | [`../track3_anomaly/analysis/temporal_prior_protocol.md`](../track3_anomaly/analysis/temporal_prior_protocol.md) |
| Runtime and peak VRAM | [`../BENCHMARKS.md`](../BENCHMARKS.md) |
| Leaderboards, denominators, submission histories | [`../leaderboards/`](../leaderboards/) |
| Worked failure cases | [`../docs/case_studies/`](../docs/case_studies/) |
| Commands, environment, provenance gaps | [`../REPRODUCE.md`](../REPRODUCE.md) |
| Engineering retrospective | [`../POSTMORTEM.md`](../POSTMORTEM.md) |

Verify the repository state before trusting any of it:

```bash
./scripts/validate_official_artifacts.sh
cd track3_anomaly/analysis && python3 temporal_prior_baselines.py --vlm-preds ../results/psi_temporal_vlm_train_preds.jsonl
cd track3_anomaly/analysis && python3 prompt_ablation_psi_mcq.py
```
