# Reply to the 2026-08-14 review comments, and what is left

Checked against the Overleaf source as of 20:5x. Short version: **two of the three
comments are already resolved in the source, one is a judgement I agree with, and
two small items remain.**

---

## The three comments

### 1 · "7/47 routed, 6/24 generic, 4/24 — does the 47 need fixing?"

**Already fixed in the source.** §4.2 now reads:

> On the 24 items answered in all three conditions, 3/24 routed, 6/24 generic, and
> 4/24 box-aware outputs omitted a parseable final letter; across the routed
> condition's full 47-item run the rate is 7/47.

The comment was right, and it was not a typo. The routed prompt ran on 47 items;
generic and box-aware ran on the same 24; those 24 are a strict subset of the 47,
so the original sentence printed one full-run rate beside two paired-subset rates.
Verified from `track3_anomaly/psi_mcq_cv_results/`: the three-way intersection is
exactly 24, and restricted to it the routed condition has 3 unparseable outputs.
The paired accuracies of 3/24, 8/24 and 9/24 reproduce Table 5 exactly, so the
pairing itself was always sound.

Nothing further needed.

### 2 · "Remove the +0.1113 baseline sentence from the abstract"

**Done — and the reasoning is right.** The sentence is commented out at L41.

The concern was that naming one baseline invites *"so it does not beat the
others?"*. That holds, and there is a second reason to keep it out of the
abstract: it is the least controlled number in the paper. The board exposes model
labels only — not prompts, frames, decoding or revision — which §5.2 says
explicitly. A caveat that long does not survive an abstract, and the claim without
its caveat is worse than not making it.

It stays in §5.2, with the caveat, which is the right place. I had earlier
suggested promoting it to the abstract; that suggestion is withdrawn.

### 3 · "Keep the added sentence about §§7–9 in the abstract and introduction"

**Noted, and left alone.** I had earlier suggested trading that sentence for the
baseline result. Comment 2 removes the baseline result, so the trade is moot and
the sentence stays. It will not be raised again.

---

## What is left

### A · Delete the commented-out sentence rather than leaving it (L41)

The removed abstract sentence is still in the file as `%On the TAR General
board...`. That is invisible in the PDF but not in the source, and the LaTeX
source is submitted to Springer through Meteor. A commented-out claim about
beating an organizer baseline is a strange thing for a reader of the source to
find. Delete the line.

### B · The footnote commit is stale again

Currently `9050049`. It moves with every push; set it immediately before
generating the final PDF and do not touch the repository afterwards.

---

## Everything else checks out

Verified against the current source:

| | |
|---|---|
| citations | 15 cited / 15 defined, none dangling, none unused |
| cross-references | none dangling, no unreferenced labels |
| numbers | every figure traceable to the portal export or the artifacts |
| Figure 1 | §6 tags match the text — §6.2 on PSI's prompt program, §6.1 on its output control, §6.3 FETV output, §6.4 TAR frame policy; no text collisions |
| terminology | `official` reserved for organizer-defined objects; `capability audit`, `negative result`, `two late system versions` all gone |
| format | `eccv` in final mode, body ends p.14, references beyond — within the allowance |

Abstract is 216 words, which is comfortable for LNCS.

---

## One judgement call still open, unrelated to the comments

The paper does not mention that one intermediate FETV step modified only clips
inside the scored subset; its Limitations say only that *some development used
sequential leaderboard feedback*. `ABLATIONS.md` §B2 carries the full analysis
(p = 3.05e-05; worth +0.0018 against a +0.0110 margin over the next-ranked team,
so it changes no placement).

The repository is registered with the organizers and both the script and the
intermediate artifacts are visible in it. Saying it first costs two lines and is
worth more than the lines. But it is a disclosure decision, not an error, and the
paper is defensible either way — the generic sentence does point at it.

If it goes in, one clause after *"some development used sequential leaderboard
feedback"* is enough.
