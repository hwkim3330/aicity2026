# A reviewer's read — what will be attacked, and what holds

Written 2026-08-14 against the current source. This is not the consistency check
in [`FINAL_CHECK.md`](FINAL_CHECK.md), which is done and clean. This asks the
other question: if a hostile ECCV reviewer or a sharp person at the poster went
after the argument, where would they push?

Most of this cannot be fixed before tomorrow, and should not be. It is for the
poster and the Q&A.

---

## The one real attack

**The organizing idea is the one thing never measured.**

The figure's title is *one frozen VLM, three deterministic task programs*. The
paper's structure says: a shared backbone plus task-specific control works across
three benchmarks. §6 then tests **components** — frame policy, target grounding,
temporal calibration, output contract — but never tests **routing itself**. There
is no run of "one generic program applied to all three benchmarks" to compare the
routed system against.

The closest thing in the paper points the wrong way: on PSI MCQ the shipped
routed prompt scores **3/24 against the generic prompt's 8/24**. The single
controlled test of routing found it worse than not routing.

A reviewer will say: *your paper is named after a design whose benefit you did not
measure, and your one measurement of it is negative.*

**The answer is already in the paper, and it should be the first thing said at the
poster:** this is a *capability analysis of a deployed system*, not a proposal
that routing helps. The abstract, §1 and the conclusion all avoid claiming routing
improves anything. That discipline is the paper's strongest feature, and it only
reads as a strength if stated before someone else frames it as a gap.

Do not let the poster's headline imply the opposite.

---

## Four more that will come up

**1 · The controlled analyses are small.** 24 paired items; 25 items; p-values of
0.0625 and 0.0312 computed from 5 and 6 discordant pairs. The paper hedges
correctly in §6, but the abstract leads with 3/24 → 9/24, and a reader who reaches
§6 finds the sample it rests on. Have the number ready: **six discordant pairs, all
six favouring the box-aware variant.** That is what p = 0.0312 means here.

**2 · The most quotable number is the least controlled.** The paper is unusually
rigorous about its own provenance — recovered weight revision, bit-stable reruns,
six eliminated explanations — and then the headline external comparison is
**+0.1113 over the organizer's `Qwen3-VL-8B-Instruct` baseline**, whose prompts,
frames, decoding and revision are all unknown. §5.2 says so explicitly and calls
it an *external same-model-label reference*. Expect someone to notice the
asymmetry anyway. The honest line: *it is the closest external reference that
exists, we cannot control it, and we say so in the text.*

**3 · FETV is the best result and the least analysed.** 3rd of 8, and its §6 entry
is a comparison between two submitted artifacts where the description formatting
and the timestamp field changed together. The paper labels it a joint association
rather than an ablation, which is right. But the strongest result has the weakest
experiment behind it, and the paper's own field-level analysis (violator-centric
geometry) is diagnostic rather than causal.

**4 · The reproducibility candor invites an obvious question.** §9 states that
re-running FETV matches `v6_fewshot` on 67.0% of fields, that the shipped artifact
is the end of an eleven-version chain, and that six explanations for the residual
were tested and ruled out with no cause established. A hostile reading: *you are
publishing a system you cannot regenerate.*

The defence is strong but must be ready, in this order: the scored artifact is
byte-identical to a Git object created on submission day and unchanged since; two
reruns of the current pipeline agree bit-for-bit on 13 fields across 200 clips;
the v10 → v11 step was recovered exactly; and what is missing is the intermediate
*code states*, which were never committed — not the artifacts, which were.

---

## What actually holds, and should lead

- **One checkpoint across three benchmarks, with the revision recovered and
  pinned.** Most challenge reports cannot say this. It makes the cross-benchmark
  comparison controlled at the level that matters.
- **Negative results published.** Routing hurt PSI MCQ; metadata priors beat the
  VLM on temporal localization. Papers that report these are believed about
  everything else.
- **An external check nobody arranged.** The organizers ran the same backbone and
  scored 0.3143 against 0.4256. Uncontrolled, but not chosen by us.
- **Artifact discipline.** SHA-sealed submissions, a recovered generation step, a
  scored-subset-targeted edit disclosed with its own p-value and bounded at
  +0.0018 against a +0.0110 margin. That last one is the paper disclosing
  something against its own interest, which is the single most credible thing in
  the package.

---

## If asked the hardest version

> *"You placed 24th of 27, 3rd of 8, and 5th of 7. What did the design buy?"*

The defensible answer is not about rank. It is: one frozen checkpoint reached all
three benchmarks with no training, and the analysis says where the remaining loss
sits — target grounding and violator-centric geometry, not semantic
understanding. The organizer's run of the same backbone at 0.3143 is the only
external evidence about what the control layer is worth, and it is worth stating
plainly along with its limits.

Do not argue the ranks. Argue what was isolated.
