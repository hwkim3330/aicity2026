# Camera-ready final verification — 2026-08-14

Checked against the Overleaf source pulled at 11:4x on 2026-08-14 (320 lines) and
against the compiled PDF text supplied separately. Repository evidence is commit
`d49b75b`.

---

## 0. The two versions have diverged, and Overleaf is behind

The PDF under review contains yesterday's corrections. **The Overleaf source
does not.** Greping the live `main.tex`:

| Content | In the PDF | In Overleaf `main.tex` |
|---|---|---|
| Revision `0c351dd…` | present | **0 occurrences** |
| 13 organizer baselines / `0.5729` | present | **0 occurrences** |
| Footnote commit | `65b444df` | `8dc5615f` |
| Figure 1 | redrawn with branches | old version, text overflowing |
| Abstract | revision recovered | *"but not the exact Hub revisions"* |

Compiling and submitting from Overleaf as it stands would drop every correction.
Resolve this before anything else on this list.

---

## 1. Must fix before camera-ready

**1.1 Title metadata does not match the title.** Three places disagree:

- `\title` (L28): *KoreaDrive: A System-Level Capability Analysis of Video-Language Understanding across TAR, FETV, and PSI-VQA at AI City Challenge 2026*
- `\titlerunning` (L29): *Task-Routed Video-Language Reasoning Across Traffic Domains*
- `pdftitle` (L16): *Task-Routed Video-Language Reasoning Across Traffic Domains*

The running head and the PDF metadata carry the previous title. `pdftitle` is
what appears in a reader's window title and in indexing; it should match `\title`.
A shortened `\titlerunning` is normal, but it should be a shortening of the
current title, e.g. *KoreaDrive: Capability Analysis across TAR, FETV, and PSI-VQA*.

**1.2 Section 7.3 contradicts Sections 1, 4 and 9** (L289):

> "The official artifacts record the same model identifier and precision, while
> the missing revision hashes prevent a stronger checkpoint-identity claim."

In the PDF version, Sections 1, 4 and 9 state the revision was recovered. This
sentence was not updated, so the paper asserts both that the revision is known
and that it is missing. Replace with:

> The official artifacts record the same model identifier, precision, and weight
> revision, so the three benchmarks were scored under one checkpoint.

**1.3 The footnote commit is stale** (L298): `8dc5615f` predates the leaderboard
export, the revision recovery and the baseline analysis that the text now cites.
Use the hash at submission time.

**1.4 Name the version pair** (L188): *"Between two late system versions"* is
vague and unverifiable by a reader. It is **v8 → v11**, both submitted to the
portal, and no adjacent pair produces 67/51 — a checker comparing neighbours will
conclude the claim is wrong. Write "Between submitted versions v8 and v11".

**1.5 `Capability Audit` survives in three places** (L43, L48 keywords, L313
conclusion) while the title now says *Capability Analysis*. The keyword list is
the most visible: it is indexed.

---

## 2. Recommended wording improvements

**2.1 Section 6.2** (L234): *"\Cref{tab:prompt} shows a negative result: the
generic Track 3 MCQ prompt beats the shipped PSI-specific prompt…"*

Report the measurement, then the interpretation:

> \Cref{tab:prompt} shows a task-dependent effect. The generic Track 3 MCQ prompt
> reaches 8/24 against 3/24 for the shipped routed prompt, and adding an explicit
> instruction to re-locate the red-boxed pedestrian raises the routed
> configuration to 9/24, winning all six discordant items ($p=0.0312$). Explicit
> target grounding therefore recovers the routed configuration rather than
> routing being harmful in itself.

Same numbers, same conclusion, and it states what was varied before what it means.

**2.2 Conclusion** (L313) leads with *"task routing offers a reusable interface,
not a uniformly beneficial reasoning method"* and then lists three things that
underperformed. The findings are worth keeping; the ordering makes the paper
close on its weakest note. Lead with what the analysis establishes — one frozen
backbone deployed across three benchmarks through inference-time control alone —
and keep the qualifications as the bounded claim.

---

## 3. Figure and caption issues

**3.1 Figure 1, Overleaf version: text overflows its boxes.** Verified by
rendering `fig_system_overview.pdf` at 130 dpi:

- *Required output format* — the final `t` touches the border
- *Frozen Qwen3-VL-8B* — spills outside the box
- *Contract + parser norm./calibration* — overflows on **both** sides
- *TAR / FETV / PSI-VQA* — floats under an arrow with nothing to attach to

**3.2 Figure 1, PDF version: configuration and results are mixed.** Boxes that
carry frame policies also carry `0.4256 | Public 24/27`, `Beats 11/13 organizer
baselines`, `Controlled A/B: 15/25 -> 16/25`, `Temporal: VLM 0.4617 > random
0.2741`. Those duplicate Tables 1–5 and Figures 2–3. The effect is that three
branches read as three systems, which contradicts the paper's central claim of
one shared backbone.

**Target structure.** Panel (a) the Track 3 evaluation interface, with TAR marked
in-domain and FETV/PSI-VQA as the two out-of-domain evaluations; the organizer
baseline annotation appears **once**, not per branch, since only TAR has one.
Panel (b) the same input, then one shared frozen backbone box drawn once
(`Qwen3-VL-8B-Instruct, bf16, 0c351dd0, ≤16 frames, 151,200 px/frame`), then
task-specific control selection branching to three control layers carrying
**configuration only**:

| Branch | Frame policy | Prompt program | Contract and post-processing |
|---|---|---|---|
| TAR | 16 frames, `MM:SS` question-window extraction | exact-choice with short reasoning; longer budgets for descriptive tasks | final-token extraction, vocabulary normalization, 5-sample vote on BCQ |
| FETV | 16 frames, uniform | one call for the complete 13-field record, few-shot | schema validation, field legalization, description template |
| PSI-VQA | 16 frames, real-fps metadata | per-subtask programs, greedy | letter extraction with fallback, duration-stratified prior $\hat I(d)$ |

Mark only the three components §6 varied — TAR question-window sampling, PSI MCQ
red-box re-localization, PSI temporal threshold and endpoints — with one
consistent cue, and say so in the caption. No scores in the figure.

**"Task-specific control selection" is more accurate than "Task Router."** The
implementation selects a prompt, frame policy, parser and calibrator; drawing it
as a router invites reading it as a learned component, which it is not.

**3.3 Figure 2 caption** (L193) states a conclusion the figure supports but does
not name the mechanism it motivates:

> Official FETV field-level results. Global and overlay-associated attributes
> score highest, whereas violator-centric geometry and violation fields score
> lower, motivating the coupling analysis in Sec. 7.1.

**3.4 Figure 3 caption** (L229): *"exceeds random intervals but trails simple
metadata-only priors"* is accurate and evaluative for a caption. Neutral form,
same numbers:

> The video-aware Qwen3-VL configuration improves substantially over random
> intervals, while fitted metadata priors give stronger localization on these
> held-out splits.

---

## 4. Terminology inconsistencies

`actor` appears on 11 lines, `violator` on 2, in five different compounds:
`actor-centric geometry`, `actor-dependent field`, `actor identity`, `actor
consistency`, `actor-record decision`, plus `actor selection` and
`actor-centered error`.

In FETV every one of these refers to the **violator** — the fields are
`answer_violator_type`, `answer_color`, `answer_initial_position`,
`answer_final_position`, `answer_initial_lane`, `answer_final_lane`, all
predicated on `answer_violation_type`. In PSI-VQA the referent is the **target
pedestrian**, marked by a red box. These are different entities and the paper
currently uses one word for both.

Suggested convention:

- FETV: **violator-centric** geometry, violator-dependent fields
- PSI-VQA: **target pedestrian**
- Cross-benchmark discussion: **target entity**, defined once at first use as
  *the violator in FETV or the target pedestrian in PSI-VQA*

This matters for §7.3, which currently says *"not every actor-centered error is
an identity failure"* while moving between the two benchmarks in one sentence.

---

## 5. Claims that remain too negative or too strong

**Too negative.** §6.2 *"negative result"*; Figure 3's *"trails"*; the conclusion
ordering (§2 above). Section 7 already carries the failure analysis, so §6
carrying the same register duplicates it.

**Too strong — none found.** Every quantitative claim checked (99 of them) is
supported. Two are worth watching:

- *"FETV is the strongest competitive result among our three Track 3
  leaderboards"* (L165). True on rank (3/8 vs 24/27 and 5/7), and the denominators
  differ, but the sentence says "competitive result", not "score", so it holds.
- *"exceeds 11 of 13 organizer baselines"* — correct, and the paper also states
  the strongest baseline is 0.5729 above ours. Keep both halves together; the
  first alone would mislead.

---

## 6. Internal contradictions

1. **Revision status** — §7.3 (L289) says the hashes are missing; §§1, 4, 9 in the
   PDF say they were recovered. *(item 1.2)*
2. **Title** — `\title` vs `\titlerunning` vs `pdftitle`. *(item 1.1)*
3. **Framing word** — title says *Capability Analysis*; abstract, keywords and
   conclusion say *capability audit*. *(item 1.5)*
4. **Figure 1 vs the text** — the figure's branch boxes imply three systems; the
   text and title claim one shared backbone. *(item 3.2)*
5. **Overleaf vs PDF** — the source lacks every correction the PDF contains.
   *(section 0)*

No new numbers were introduced anywhere in this document.

---

## 7. Added 2026-08-14, second pass — the same-backbone baseline

Refetched the live boards. Two things.

**`0.5748` is not on the board.** The name is right: the strongest TAR baseline
carries `data.models_used = "Cosmos3-Super"`, which is what the web UI shows. Its
score is `0.5728527671760983`, i.e. **0.5729**. Every numeric field of every row
on `general/3` was scanned and nothing equals 0.5748; the nearest value anywhere
is an unrelated row's `mcq_accuracy = 0.575`. Revert both occurrences.

**All 13 baselines are named, and one of them is our backbone.**

| Organizer baseline | TAR mean |
|---|---:|
| Cosmos3-Super | 0.5729 |
| Cosmos3-Nano | 0.4613 |
| **KoreaDrive** | **0.4256** |
| Qwen3.5-27B | 0.4084 |
| Cosmos-Reason2-32B | 0.3798 |
| Gemini-3.1-Pro-Preview | 0.3659 |
| Cosmos-Reason2-8B | 0.3598 |
| Gemma-4-31B-It | 0.3445 |
| **Qwen3-VL-8B-Instruct** | **0.3143** |
| Qwen3.5-9B (with reasoning) | 0.3063 |
| Qwen3.5-9B | 0.3063 |
| Gemma-4-31B-It (with reasoning) | 0.3062 |
| Qwen3.5-27B (with reasoning) | 0.2983 |
| Qwen3-VL-32B-Instruct | 0.2875 |

The organizers ran **the same checkpoint the paper uses**. `Qwen3-VL-8B-Instruct`
scores 0.3143 against KoreaDrive's 0.4256: **+0.1113 from the task-specific
control layer alone**, same model, same benchmark, same scorer. Their
`Qwen3-VL-32B-Instruct` scores 0.2875, so the 8B backbone with control also
exceeds their 32B run by **+0.1381**.

This is a controlled comparison of exactly the thing the paper claims to study,
and it is currently absent. It is much stronger than "exceeds 11 of 13
baselines", which mixes thirteen different models together. Suggested wording:

> On the TAR General board the organizers publish 13 baselines, including a run
> of the same backbone this system uses. KoreaDrive's 0.4256 exceeds the
> `Qwen3-VL-8B-Instruct` baseline (0.3143) by 0.1113 and the
> `Qwen3-VL-32B-Instruct` baseline (0.2875) by 0.1381, isolating the contribution
> of the inference-time control layer under an identical checkpoint. The
> strongest baseline, `Cosmos3-Super`, reaches 0.5729.

Keep the last sentence: the honest framing is that control adds 0.1113 over the
same model and that a stronger model still leads.

Evidence: `leaderboards/raw/general_3.json`, `data.models_used` on rows where
`isBaseline` is true; refetched 2026-08-14.

---

## 8. FETV has no entry in Section 6 — and one is available

Figure 1 marks §6.3 on TAR's frame policy and §6.2 / §6.1 on PSI's prompt program
and post-processing. FETV carries no marker, which is correct: §6 covers PSI
temporal calibration, PSI target grounding, TAR sampling and scorer sensitivity,
and the TAR adapter study. **The benchmark the paper places third of eight has no
controlled analysis at all.** The gap is real, not a drawing error.

It can be closed with data already in the repository, and the comparison is
stronger than the ones §6 currently reports because it runs on the **official
test set with the official scorer** rather than on 24–25 local items.

Between submitted artifacts `v6_fewshot` and `v7`, **all eleven categorical
fields are byte-identical** on the portal — violation type, violator type,
colour, both positions, both lanes, intersection type, weather and lighting.
Only three numbers moved:

| | v6\_fewshot | v7 |
|---|---:|---:|
| `date_accuracy` | 0.995 | 1.000 |
| `time_accuracy` | 0.79 | 0.94 |
| `description` | 0.3645 | 0.4209 |
| **final** | **0.4238** | **0.4584** |

Two changes produced this, both identifiable in the artifacts:

1. **Deterministic description formatting.** 0/200 v6 descriptions begin with
   `On <date> at <time>, `; 200/200 of v7's do. This is the opener enforcement in
   `fetv_submission.py`, i.e. the FETV *output / post-processing* block already
   drawn in Figure 1.
2. **Timestamp re-reading.** Only 37/200 `answer_time` values are unchanged.

Because the official score is the mean of the description score and the
categorical mean — verified, 0.4237 and 0.4585 reconstruct the reported 0.4238
and 0.4584 — the gain decomposes:

- description $0.3645 \to 0.4209$ contributes $+0.0282$
- categorical mean $0.483 \to 0.496$ contributes $+0.0065$, and since eleven of
  the thirteen fields are identical this comes entirely from date and time
- total $+0.0347$ against the reported $+0.0346$

So **81% of the official FETV improvement came from the output contract and the
timestamp field, with every categorical prediction held fixed.** That is exactly
the claim the paper makes about inference-time control, measured on the official
scorer.

Caveat to state: two components changed together, so this attributes the gain to
the pair rather than isolating either. It is a clean joint attribution with the
categorical half frozen, not a single-variable ablation — stronger than the other
historical comparisons in §6.3 and weaker than the paired PSI studies.

If this is added, mark FETV's *Output / post-processing* box in Figure 1 with the
matching section number so all three branches carry one.

---

## 9. Two review questions, answered from the data

### 9.1 "Is there evidence the prompt program contributed to the improvement?"

**No, and the one controlled prompt experiment in the paper points the other
way.** Every piece of evidence in the repository:

| Comparison | What varied | Result |
|---|---|---|
| §6.2 PSI MCQ, 24 paired items | **prompt program only** | shipped routed prompt **3/24**, generic no-routing prompt **8/24** |
| FETV `v5 → v6_fewshot`, official scorer | few-shot prompt **plus other changes** | 11 of 13 fields moved — confounded, nothing attributable |
| FETV `v6_fewshot → v7`, official scorer | output contract + timestamp | **+0.0346**, of which **+0.0282 (81%) is the description formatting** and +0.0065 is date/time; all eleven categorical fields byte-identical |
| §6.3 TAR | frame policy | 15/25 → 16/25 |

So: the only controlled test of a shipped prompt program found it **below** the
generic prompt, and the largest clean official gain on FETV came from the
**output contract**, not from prompting. The `v5 → v6_fewshot` step is the one
place a prompt change might have helped, and it moved eleven of thirteen fields
at once, so it cannot carry the claim.

The defensible statement is that **output contracts and post-processing carry the
measured gains, while prompt programs are the component the analysis found most
fragile** — the box-aware variant recovers PSI MCQ to 9/24 by adding explicit
grounding, which is again a change to what the prompt *specifies*, not evidence
that the shipped prompt program helped.

**This changes Figure 1.** Pale-yellow currently emphasises the prompt-program
rows as the proposed design. Nothing measured supports that emphasis. Either
move the highlight to the *Output / post-processing* rows, which is where the
evidence is, or drop the fill entirely and let the orange §6 markers carry the
figure. Keeping yellow on prompt programs asserts a contribution the paper
cannot show.

Do not fix this by adding a claim. If the intent is to present prompt programs
as a design contribution, the honest framing is that they are the interface
through which task specialization is expressed, with §6.2 showing that *what*
the prompt specifies — explicit target grounding — is what moves the score.

### 9.2 "Does 'official pipeline' mean organizer-provided?"

No, but the paper never says so. `official` appears **41 times in two different
senses**, undefined:

- **Ours**: official artifacts (7), official submissions (3), official runs (2),
  official pipeline (2), official predictions (1) — meaning *the runs we
  submitted and that were scored*.
- **The organizers'**: official mean (4), official scorer, official evaluator,
  official public board — meaning *defined by the challenge*.

In the Figure 1 caption the ambiguity is worst, because the same figure now also
carries organizer baselines. A reader can reasonably parse "prompt programs used
in the official pipeline" as "prompt programs the organizers supplied".

Fix in the caption: **"used in the submitted pipeline"** or **"in the scored
submissions"**. Elsewhere, define the term once at first use — *we write* official
*for the submissions that were scored on the evaluation server* — and keep
organizer-defined objects as *organizer scorer*, *organizer baseline*.
