# Camera-ready checklist

**Status 2026-08-14, late:** all four previous items are applied, verbatim and
correctly. Full re-verification of the updated source found **one** new problem
and nothing else.

- citations: 15 cited / 15 defined, none dangling, none unused
- cross-references: one dangling, see item 1 below
- numbers: **0 unverified** — every figure in the paper matches the portal export
  or the artifacts
- retired phrasing: `capability audit`, `negative result`, `two late system
  versions`, `not retained`, `missing revision`, `official pipeline` — all zero
  occurrences; `KoreaDrive System` appears once, as the §4 title, which is correct

---

## The one remaining item

### 1 · `fig:fetvcase` is never referenced

The new frame figure carries `\label{fig:fetvcase}` but no `\Cref` points at it.
An orphan float has no textual anchor: LaTeX places it wherever it fits and the
reader is never sent to it. §7.1 discusses exactly this clip, so the reference
belongs in its first sentence.

**Find**

```
In clip 019\_004.mp4, one system version predicts no violation
```

**Replace**

```
In clip 019\_004.mp4 (\Cref{fig:fetvcase}), one system version predicts no violation
```

### 2 · The footnote commit — do this last

Currently `7787a98`. Set it immediately before generating the final PDF.

---

## Previously listed, now applied — verified verbatim

| | |
|---|---|
| §3.1 | `BERTScore-F1 (\texttt{roberta-large}, rescaled with baseline)` |
| §3.2 | `IDF-weighted BERTScore (\texttt{microsoft/deberta-xlarge-mnli})` |
| §4.2 | `On the 24 items answered in all three conditions, 3/24 routed, 6/24 generic, and 4/24 box-aware …; across the routed condition's full 47-item run the rate is 7/47.` |
| footnote | moved from `d100b24` to `7787a98`; still needs the final value |

---

<details>
<summary>Earlier revision of this checklist</summary>

# Camera-ready checklist — regenerated against the Overleaf source of 2026-08-14 evening

Six of the eight earlier items are applied. **Three edits and one final step
remain.** Find-strings below were checked against the current
source, so they can be applied directly.

---

## Remaining

### 1 · §3.1 — name the TAR BERTScore configuration

**Find**

```
open-ended tasks are evaluated with BERTScore-F1~\cite{bertscore}
```

**Replace**

```
open-ended tasks are evaluated with BERTScore-F1 (\texttt{roberta-large}, rescaled with baseline)~\cite{bertscore}
```

### 2 · §3.2 — name the FETV BERTScore configuration

**Find**

```
The description metric combines normalized CIDEr~\cite{cider} and BERTScore;
```

**Replace**

```
The description metric combines normalized CIDEr~\cite{cider} and IDF-weighted BERTScore (\texttt{microsoft/deberta-xlarge-mnli});
```

**Why these two are worth the space.** BERTScore is the organizers' scorer, not
ours, and it carries most of the paper's headline numbers:

- **TAR: 7 of the 9 scored components are BERTScore** — 78% of that mean by
  weight, contributing 0.2999 of the reported 0.4256. The "strongest component,
  MCQ-OE at 0.7664" *is* a BERTScore value.
- **FETV: 25% of the final score.** `description = (CIDEr + BERTScore)/2` and
  `final = (categorical + description)/2`.
- **PSI-VQA: not used** — cue matching is SBERT.

The two benchmarks use different configurations and the paper names neither:
`deberta`, `roberta`, `idf`, `rescale` appear zero times. §9 already pins
`transformers==4.57.0` because BERTScore moves up to 0.02 absolute across
versions — a reader warned the metric is version-sensitive, but not told which
model produced the numbers, cannot act on that warning. Both additions are facts
about the organizers' scorers, so neither adds a claim about our system.

Sources: the AI City Track 3 page for TAR; `evaluate.py` in the FETV repository
for FETV.

### 4 · The footnote commit — do this last

Currently `d100b24`, already behind. Set it immediately before generating the
final PDF; it moves with every push.

**Find** `commit \texttt{d100b24}`

### 3 · §4.2 — put the three contract rates on one basis

Raised in review: only the routed condition has denominator 47. The clause
*"Using all available held-out generations per condition"* was added, which
explains why they differ but does not make them comparable — the 24 are a subset
of the 47, so the reader is asked to compare a full-run rate against two
paired-subset rates.

**Find**

```
Using all available held-out generations per condition, 7/47 routed, 6/24 generic, and 4/24 box-aware outputs omitted a parseable final letter.
```

**Replace**

```
On the 24 items answered in all three conditions, 3/24 routed, 6/24 generic, and 4/24 box-aware outputs omitted a parseable final letter; across the routed condition's full 47-item run the rate is 7/47.
```

This is one series on one basis, matching Table 5's denominators, with the 47-item
figure kept as context rather than dropped.

**Verified from `track3_anomaly/psi_mcq_cv_results/`:** `mcq_baseline.jsonl` holds
47 items, `mcq_generic.jsonl` and `mcq_boxaware.jsonl` hold the same 24, and those
24 are a strict subset of the 47 — the three-way intersection is exactly 24, so 23
routed items have no counterpart. Restricted to those 24 the routed condition has
**3** unparseable outputs. The paired accuracies are 3/24, 8/24 and 9/24, which
reproduce Table 5 exactly, so the pairing itself is sound.

The ordering does not change: the routed prompt has the lowest contract-failure
rate on either basis. The fix costs nothing and removes the discrepancy a reviewer
would otherwise have to resolve.

---

## Applied since the last list — verified in place

| | |
|---|---|
| §1 team name | now `KoreaDrive (Team 277; …)`; §4 title and Figure 1 keep *KoreaDrive System* |
| §1 baseline attribution | now credited to the General board, with the caveat that the export exposes only model labels — more accurate than what I proposed |
| Abstract MCQ ordering | now *"statistically indistinguishable from the 8/24 generic prompt"* |
| §7.3 clause | now *"is where stronger explicit structure would help most"* |
| Abstract contents sentence | replaced by the same-backbone baseline result |
| §4.2 denominators | clause added — see item 3, which finishes it |

**Also newly added and checked:** the FETV `v6_fewshot → v7` comparison in §6, the
TAR 32B baseline sentence in §5.2, and the annotated FETV frame as
`\Cref{fig:fetvcase}`. All thirteen new numbers match the values recomputed from
the portal export and the artifacts. The frame figure cites
`aicity2026track3,fisheye8k`, states that the box marks a prediction rather than
ground truth, and the worked-PSI caption is correctly narrowed to *PSI-VQA
frames*.

One correction to my own earlier reports: I wrote that **eleven** categorical
fields were byte-identical across `v6_fewshot → v7`. It is **ten** — date and time
accuracy are the two that changed, which is the point of the comparison. The paper
says ten and is right; `paper/FINAL_CHECK.md` §14 records the fix.

---

## Not to do

- Do not add a claim that the prompt program improved performance — no evidence
  exists, and the paper currently claims none, which is correct.
- Do not remove TAR.
- Do not introduce a number that is not in this file or in `FINAL_CHECK.md`.

## Page budget

16 pages against a 14-page main-text allowance with references beyond it. Inside,
without margin. If space is needed, §8 *Design Implications and Exploratory
Prototypes* reports nothing scored and Reviewer vmz20 warned against presenting
those prototypes as evaluated systems.

</details>
