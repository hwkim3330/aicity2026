# Camera-ready checklist — 2026-08-15 deadline

Eight edits, each with the exact string to find. Nothing here introduces a number
that has not been recomputed from the portal export or the artifacts. Reasoning
for each is in [`FINAL_CHECK.md`](FINAL_CHECK.md) at the section noted.

---

### 1 · §1, the team is `KoreaDrive`, the system is `the KoreaDrive System` — §11.1

The §4 title and Figure 1 keep *KoreaDrive System*; that rename was deliberate.
Only the introduction changes, because its parenthetical introduces the **team**
and the two table captions already define *KoreaDrive* as Team 277.

**Find** `This paper describes KoreaDrive System (Team 277;`
**Replace** `This paper describes KoreaDrive (Team 277;`

---

### 2 · §1 item 1, the baselines are on the General board — §11.2

The comparison is sound — the same submission scores 0.4256 on both boards — but
the sentence credits it to the public board, where no baseline exists.

**Find**

> the official public-board score is 0.4256 (24th of 27). For reference, the
> separate TAR General board includes 13 organizer baselines. KoreaDrive's
> public-board score is 0.1113 above the organizer baseline labeled
> Qwen3-VL-8B-Instruct (0.3143) and numerically exceeds 11 of the 13 baselines;
> the strongest organizer baseline, Cosmos3-Super, scores 0.5729.

**Replace**

> the official public-board score is 0.4256 (24th of 27). The same submission also
> appears on the TAR General board, where the organizers publish 13 baselines.
> There it is 0.1113 above their `Qwen3-VL-8B-Instruct` baseline (0.3143), which
> runs the same backbone as this system, and exceeds 11 of the 13; the strongest,
> `Cosmos3-Super`, reaches 0.5729.

---

### 3 · Abstract, do not order 9/24 above 8/24 — §11.3

§6.2 calls the two statistically indistinguishable, so the abstract must not
assert an ordering the body declines to.

**Find** `from 3/24 to 9/24, above the generic prompt at 8/24,`
**Replace** `from 3/24 to 9/24, matching the 8/24 generic prompt,`

---

### 4 · §4.2, put all three rates on the paired 24 — §13

Not a typo: the routed prompt ran on 47 items, generic and box-aware on the same
24, and those 24 are a subset of the 47. So `7/47` is a full-run rate printed
beside two paired-subset rates. Recomputed on the paired 24 the routed condition
is **3/24**. The ordering is unchanged either way, and this matches Table 5's
denominators.

**Find** `In the controlled PSI prompt study, 7/47 routed, 6/24 generic, and 4/24 box-aware generations omitted a parseable final letter.`
**Replace** `On the 24 paired items, 3/24 routed, 6/24 generic, and 4/24 box-aware generations omitted a parseable final letter.`

*(Keeping 47 instead is defensible but then needs a clause saying the routed
condition was run on a larger set and 47 is its full-run rate.)*

---

### 5 · §3.1 and §3.2, name the BERTScore configuration — §12.1

BERTScore is the organizers' scorer, not ours, and it carries **7 of the 9 scored
TAR components — 78% of that mean by weight, 0.2999 of our 0.4256** — plus
**25% of the FETV final score**. The two benchmarks use different setups, and the
paper names neither: `deberta`, `roberta`, `idf` and `rescale` appear zero times.
This matters because §9 already pins `transformers==4.57.0` on the ground that
BERTScore moves up to 0.02 absolute under 5.x — a reader warned the metric is
version-sensitive, but not told which model produced the numbers, cannot act on
the warning.

**§3.1 find** `evaluated with BERTScore-F1~\cite{bertscore}`
**§3.1 replace** `evaluated with BERTScore-F1 (\texttt{roberta-large}, rescaled with baseline)~\cite{bertscore}`

**§3.2 find** `combines normalized CIDEr~\cite{cider} and BERTScore`
**§3.2 replace** `combines normalized CIDEr~\cite{cider} and IDF-weighted BERTScore (\texttt{microsoft/deberta-xlarge-mnli})`

Sources: the AI City Track 3 page for TAR, `evaluate.py` in the FETV repository
for FETV. Both are facts about the organizers' scorers, so neither adds a claim
about our system. PSI-VQA does not use BERTScore — its cue matching is SBERT.

---

### 6 · §7.3, a clause with the wrong subject — §11.6

**Find** `FETV shows that violator-centric geometry motivates the need for stronger explicit structure`
**Replace** `FETV shows that violator-centric geometry is where explicit structure would help most`

---

### 7 · Abstract, trade a contents list for the strongest result — §11.5

The removed sentence lists the paper's own sections. The replacement is the only
controlled comparison in the paper where the organizers ran the same backbone.

**Find** `We further provide diagnostic case studies, exploratory design implications, and reproducibility and limitations analyses.`
**Replace** `On the TAR General board the organizers publish a baseline using the same backbone, which this system exceeds by 0.1113 under an identical checkpoint.`

---

### 8 · The footnote commit — §11.4

Currently `4e9d92a`, already several pushes behind. Set it **last**, immediately
before generating the final PDF; it moves with every commit.

**Find** `commit \texttt{4e9d92a}`

---

## Optional, only if the annotated frame is used

`camera_ready_src/frames/fetv_019_004_annotated.jpg` — the box marks the road user
versions v9–v11 designated as the violator, with a strip recording that v8
predicted `no_violation` for the same clip. It marks a prediction, not ground
truth; FETV labels for the scored subset are not public.

If it goes in: cite FETV and FishEye8K at the figure, and narrow the worked-PSI
caption's *"frames are not redistributed under the benchmark data terms"* to
PSI-VQA, which is the benchmark that actually carries the TASI agreement.
`frames/README.md` records what the licensing does and does not support.

## Not to do

- Do not add a claim that the prompt program improved performance. There is no
  evidence for it, the one controlled test found the shipped routed prompt below
  the generic one, and the paper currently claims nothing — which is correct.
- Do not remove TAR. A reviewer asked for the full component breakdown, the title
  names three benchmarks, and the same-backbone baseline is now the paper's
  clearest quantitative reference.

## Page budget

16 pages against a 14-page main-text allowance with references permitted beyond
it — inside, with no margin. If space is needed, §8 *Design Implications and
Exploratory Prototypes* is the candidate: it reports nothing scored, and Reviewer
vmz20 warned against presenting those prototypes as evaluated systems.
