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
