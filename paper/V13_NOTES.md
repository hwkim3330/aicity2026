# v13 edits — find-and-replace against `camera_ready_src/main.tex`

Written against the Overleaf source pulled 2026-08-14 (v12, snapshot in
[`camera_ready_src/`](camera_ready_src/)). **Nothing here edits `main.tex`** —
the source is being revised in Overleaf and ChatGPT at the same time, so these
are quoted originals and replacements to apply there.

Line numbers are from the v12 snapshot and will drift; match on the quoted text.

---

## 1. The weight revision is no longer unknown — and this strengthens the paper

Three passages say the Hub revision was never recorded. That was true when
written and is not now: both were recovered on 2026-08-13 with evidence, and are
pinned in the code. See [`../REPRODUCE.md`](../REPRODUCE.md#why-the-revision-is-certain).

| Model | Revision | How it was recovered |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` (all three official runs) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | the Hub repo has had no commit since 2025-10-15, so `main` on the run date can only have been this; the local cache holds exactly that one snapshot |
| `Qwen/Qwen2.5-VL-7B-Instruct` (General TAR entry only) | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | logged at download time in `track3_anomaly/model_download.log`, 2026-07-02; still Hub `main`, unchanged since 2025-04-06 |

**§1 Introduction, line 43.** Replace

> The archived runs record the same model identifier, bf16 precision, and visual
> budget, but not the exact Hub revisions, so we present a capability audit
> rather than a causal domain-generalization study.

with

> The archived runs record the same model identifier, bf16 precision, and visual
> budget, and the Hub revision has since been recovered
> (\texttt{0c351dd0}), so the three benchmarks were scored under one checkpoint.
> We nonetheless present a capability audit rather than a causal
> domain-generalization study, because the benchmarks differ in task structure
> and output space and because some development decisions were informed by
> sequential leaderboard feedback.

The reason for the caveat moves from "we cannot show the checkpoints match" to
the two limitations that actually remain. Reviewer JZh931 raised precision as a
second confound; §1 of [`CAMERA_READY.md`](CAMERA_READY.md) already establishes
there is no backbone or precision confound, and this removes the third.

**§3 System, line 94.** Replace

> The exact Hugging Face weight revision was not recorded during the challenge;
> consequently, we do not claim byte-identical checkpoints or attribute
> cross-leaderboard score differences to domain shift alone.

with

> The Hugging Face weight revision was not pinned at run time but has been
> recovered (\texttt{0c351dd0}) and is now pinned in the released code, so the
> three artifacts share one checkpoint. We still do not attribute
> cross-leaderboard score differences to domain shift alone.

**§8 Reproducibility, line ~291.** Replace

> The exact Hub weight revision was not persisted for any official run.

with

> The Hub weight revision was not persisted at run time and was recovered
> afterwards (\texttt{0c351dd0}); it is pinned in the released code. Regenerating
> the FETV artifact from that code does not reproduce it, because the shipped
> file is the last of an eleven-version chain and only one code state was
> committed for the nine artifacts it contains. Against \texttt{v6\_fewshot},
> the single-pass artifact whose configuration matches, a fresh run matches
> 67.0\% of fields.

Also update the footnote: it cites commit `8dc5615f`, which now trails by
several commits. Use the hash at submission time.

---

## 2. Organizer baselines exist on TAR, and one beats us

Line 94 says the organizers "publish reference submissions" without numbers, and
§1 of the paper reports 0.4256 on TAR with nothing to compare it against. The
General board carries 13 rows flagged `isBaseline` on Track 3 — one per baseline
model, not 13 attempts — plus one on Track 2 and one on Track 6. FETV and PSI-VQA
have none, which is why the panel (a) / panel (b) asymmetry in Figure 1 is
correct for those two.

| Track | Baseline models | Best baseline | KoreaDrive |
|---|---:|---:|---:|
| 2 | 1 | 47.9186 S2 | 32.6404 |
| **3 TAR** | **13** | **0.5729 mean** | **0.4256** |
| 6 | 1 | 0.2324 mAP | not submitted |
| 7 FETV | none | — | 0.4634 |
| 8 PSI-VQA | none | — | 57.0400 |

Suggested sentence for §5 Official Results, in the neutral reporting register
requested for §6:

> On the in-domain TAR board the shared backbone reaches 0.4256, above 11 of the
> 13 organizer baseline models and 0.1472 below the strongest. The two
> out-of-domain boards carry no organizer baseline; the same checkpoint reaches
> 0.4634 (3rd of 8) on FETV and 57.04 (5th of 7) on PSI-VQA.

No editorialising in either direction, and the relative out-of-domain strength
follows from the numbers rather than from an adjective. Evidence:
`leaderboards/raw/general_{2,3,6}.json`, rows with `isBaseline` true.

---

## 3. Figure 1 — what the review asked for

Two requests: show the §6 setup parameters, and resolve the panel (a) labelling.

**Per-branch parameters.** Three branches off the Task Router, each carrying the
knobs that §6 actually varied:

| Branch | Frame policy | Prompt program | Post-processing | §6 knob |
|---|---|---|---|---|
| TAR | 16 frames, 151,200 px, `MM:SS` window extraction | exact-choice + reasoning, 5-sample vote on BCQ | vocabulary normalization | vote count, window extraction |
| FETV | 16 frames, 151,200 px | one call, full 13-field record, few-shot | field legalization, description template | few-shot on/off |
| PSI-VQA | 16 frames, real-fps metadata | per-task prompts, greedy | duration-stratified temporal prior $\hat I(d)$ | prior thresholds $T$, ratios lo/hi |

**Panel (a).** Keep "participant model/system" for FETV and PSI-VQA — accurate,
those boards have no baseline. Add a baseline box on the TAR branch only, since
there it would otherwise misstate the setting.

---

## 4. Smaller items from the same review

- **Abstract**: insert VLM into the frozen-pipeline sentence — "a frozen
  \qwen \emph{vision-language model (VLM)} inference pipeline".
- **FETV**: highlight the expansion at first use — Fisheye Traffic Violation
  Understanding (FETV).
- **§6 register**: §7 already carries the failure analysis, so §6 should report
  values without self-assessment. The routing result reads as a finding, not a
  confession, when both directions are stated: routing helped where the output
  contract was the bottleneck and hurt where it added reasoning steps the model
  then mishandled (generic 8/24, routed 3/24, box-aware 9/24).
- **Title**: the proposed *A Shared Inference Backbone for Multi-Benchmark
  Traffic-Video Reasoning* requires all three benchmarks. Dropping TAR would
  also answer Reviewer JZh931's request for the full ten-component TAR breakdown
  by deleting its subject — see [`CAMERA_READY.md`](CAMERA_READY.md) §3.
