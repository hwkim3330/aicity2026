# Camera-ready checklist — regenerated against the Overleaf source of 2026-08-14 evening

Six of the eight earlier items are applied. **Two edits and one final step remain**,
plus one optional improvement. Find-strings below were checked against the current
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

### 3 · The footnote commit — do this last

Currently `d100b24`, already behind. Set it immediately before generating the
final PDF; it moves with every push.

**Find** `commit \texttt{d100b24}`

---

## Optional

### §4.2 — the three contract rates still are not on one basis

The added clause *"Using all available held-out generations per condition"* fixes
the ambiguity and is a legitimate resolution. It explains why the denominators
differ; it does not make 7/47, 6/24 and 4/24 comparable, since the 24 are a
subset of the 47. On the paired 24 the routed condition is **3/24**, which is the
basis Table 5 uses. Adding that in parentheses would let a reader compare
directly:

> …7/47 routed, 6/24 generic, and 4/24 box-aware outputs omitted a parseable
> final letter (3/24 for the routed condition when restricted to the paired
> subset).

---

## Applied since the last list — verified in place

| | |
|---|---|
| §1 team name | now `KoreaDrive (Team 277; …)`; §4 title and Figure 1 keep *KoreaDrive System* |
| §1 baseline attribution | now credited to the General board, with the caveat that the export exposes only model labels — more accurate than what I proposed |
| Abstract MCQ ordering | now *"statistically indistinguishable from the 8/24 generic prompt"* |
| §7.3 clause | now *"is where stronger explicit structure would help most"* |
| Abstract contents sentence | replaced by the same-backbone baseline result |
| §4.2 denominators | clause added |

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
