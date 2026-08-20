# Poster content plan — AI City Challenge 2026, ECCV Workshop, 9 Sep, Malmö

Presenter: Yooseung Wang. **Board size confirmed 2026-08-19: 140 x 100 cm**
(landscape), per the organizing committee. Everything below already exists in the
paper or the repository — nothing new has to be produced except the layout.

Landscape at 140 x 100 confirms the **four-column** plan; the portrait fallback
noted earlier is no longer needed.

## Layout, to millimetres

Total 1400 x 1000 mm.

| band | height | contents |
| --- | ---: | --- |
| top margin | 40 mm | — |
| title band | 150 mm | title, authors, affiliation, board/score strip |
| gap | 30 mm | — |
| body | 700 mm | four columns |
| bottom margin | 80 mm | QR code sits bottom-right of column 4 |

Columns: side margins 50 mm each leaves 1300 mm. Three 30 mm gutters gives
**four columns of 302 mm**. Keep the gutters empty — nothing spans columns except
the title band.

Type sizes, set for a 1.5–2 m viewing distance:

| element | size |
| --- | ---: |
| title | 100 pt |
| authors / affiliation | 44 pt |
| section headers | 50 pt |
| body | 30 pt |
| table cells | 28 pt |
| captions and caveats | 24 pt |

**The caveats print at body size, not caption size**, wherever a caveat qualifies a
headline number — specifically the same-model-label caveat under the +0.1113 table
in 2.2. Shrinking a caveat below the claim it qualifies is how a poster overstates
what a paper says.

Figures are vector PDFs placed at column width (302 mm). The annotated FETV frame
is raster at 900 px, which is 76 mm at 300 dpi and 152 mm at 150 dpi — **below
column width**. Re-export it from the 1920x1920 original with
`scripts/annotate_fetv_case.py` before printing, or place it at 150 mm and accept
the white space rather than upscaling.

---

## The headline

> **One frozen Qwen3-VL-8B-Instruct, three traffic-video benchmarks, no training.**
> *What the task-specific control layer buys, and where it does not.*

**Do not** write anything implying task routing improved results. The paper never
claims it, the single controlled test of it is negative, and a poster that implies
otherwise hands a visitor the one attack that lands. See
[`REVIEWER_VIEW.md`](REVIEWER_VIEW.md).

Subtitle line, small: *KoreaDrive (Team 277) · TAR 0.4256 · FETV 0.4634, 3rd of 8
· PSI-VQA 57.04 · public leaderboards*

---

## Column 1 — the setting and the system

**1.1 Three benchmarks, one system** (≈60 words)
Track 3 evaluates traffic-video reasoning three ways: TAR (in-domain, CCTV
anomaly reasoning), FETV (out-of-domain, fisheye violation records), PSI-VQA
(out-of-domain, pedestrian intent). Different inputs, different output contracts,
independent scorers. KoreaDrive entered all three with one checkpoint.

**1.2 Figure 1, full width of the column** — reuse `fig_system_overview.pdf`
unchanged. It is the single most important object on the poster; 왕유승 is right
that most visitors will read the title, the abstract and this figure only.
Keep the §6 tags: they are what turns a block diagram into a claim about what was
tested.

**1.3 What is actually shared** (≈40 words, as a boxed callout)
`Qwen/Qwen3-VL-8B-Instruct` · bf16 · revision `0c351dd0` · ≤16 frames · 151,200
px/frame · **no adapter, no parameter update**. The revision was recovered after
the challenge and is pinned in the released code, so the three benchmarks are
scored under one checkpoint — not merely one model name.

---

## Column 2 — results, stated with their boards

**2.1 Table: official results** — reuse Table 1. Add one column: *board*. Every
rank on the poster must name its board; the portal keeps two with different
denominators for identical scores.

**2.2 The external check** (the strongest single item — give it space)

> The organizers published 13 baselines on the TAR **General** board. One of them
> runs **the same backbone** this system uses.
>
> | | TAR mean |
> |---|---:|
> | `Cosmos3-Super` (strongest baseline) | 0.5729 |
> | **KoreaDrive** | **0.4256** |
> | organizer `Qwen3-VL-8B-Instruct` | 0.3143 |
> | organizer `Qwen3-VL-32B-Instruct` | 0.2875 |
>
> **+0.1113** over the same backbone. **+0.1381** over their 32B run.

Print the caveat at the same size, not smaller: *the board exposes model labels
only — not prompts, frames, decoding or revision. An external same-model-label
reference, not a same-checkpoint control.*

**2.3 Figure 2 (FETV field scores)** — reuse `fig_fetv_fields.pdf`. It carries the
paper's clearest diagnostic on one glance: overlay and global attributes at
1.000/0.940, violator-centric geometry at 0.124–0.178.

---

## Column 3 — what the controlled analyses found

Head this column **"What we varied, and what happened"** — not "ablations", and
not "limitations".

**3.1 Target grounding, PSI MCQ** (24 paired items)

| Prompt program | Accuracy |
|---|---|
| generic Track 3 MCQ, no routing | 8/24 |
| shipped routed `psi_mcq` | 3/24 |
| routed + explicit red-box re-location | 9/24 |

One sentence beneath: *adding an explicit instruction to re-locate the red-boxed
pedestrian recovers the routed configuration; all six discordant pairs favour it
(p = 0.0312). It does not exceed the generic prompt.*

**And a second sentence, added 2026-08-20 — this must not be printed without it.**
The box-aware configuration was uploaded to the re-opened evaluation on 20 August
and scored **53.19 against the previous General submission's 55.41**. *The paired
advantage on 24 items did not survive to the full test set.*

That changes what this panel is for. It is no longer "a prompt fix that works",
it is **the cleanest negative result on the poster**: a statistically significant
paired win on a hand-scored subset, refuted by the only measurement that ranks
anything. Present it that way. A visitor who sees 9/24 with p = 0.0312 and no
follow-up will reasonably assume it generalised, and it did not.

It is also the third instance in this project — Track 1 ByteTrack and Track 4 rerank
regressed the same way — which makes it a point worth making rather than hiding:
**small-subset paired significance is not evidence about a leaderboard.**

**3.2 Temporal localization** — reuse `fig_psi_temporal.pdf`. Caption in one line:
*the VLM carries real temporal signal (0.4617 vs 0.2741 random), and fitted
metadata priors are stronger still (0.5566 held out).*

**3.3 Output contract, FETV** (the v6 → v7 comparison)
Ten structured fields byte-identical; only date, time and the description moved;
official score 0.4238 → 0.4584. **81% of the gain is the description formatting.**
Label it what it is: a joint association across two components, not a
single-factor ablation.

**3.4 One line on TAR frame policy**: question-window sampling, 15/25 → 16/25 on
matched items — too small to claim.

---

## Column 4 — evidence, and the honest column

**4.1 Figure: the FETV case** — reuse `frames/fetv_019_004_annotated.jpg` with the
same caption discipline as the paper: the box marks a **prediction**, not ground
truth. Beneath it, the coupling number: *between submitted v8 and v11, 51 clips
changed the whole violator-dependent block together.*

**4.2 A worked failure** — condense Figure 5 to four lines: correct evidence →
mis-negation → truncation → fallback `A`. The point is that the perception was
right and the contract broke.

**4.3 Provenance** (a short box, and do not shrink it)

- Scored artifacts byte-identical to Git objects created on submission day
- Two reruns of the current pipeline agree on 13 fields × 200 clips
- The `v10 → v11` generation step recovered exactly
- **What is missing is the intermediate code states, not the artifacts**
- One chain step edited only the scored clips — disclosed, p = 3.05e-05, worth
  +0.0018 against a +0.0110 margin

That last bullet is the most credible thing on the poster: reporting something
against your own interest is what makes the rest believed.

**But it is not currently in the paper.** `main.tex` has no mention of the scored
subset; its Limitations say only *"some development used sequential leaderboard
feedback"*, while `ABLATIONS.md` §B2 carries the full analysis. A poster that
discloses more than the paper invites *"why isn't this in the paper?"*

Two consistent options — pick one before printing:

1. **Add one clause to the paper's Limitations** (today, cheap). After
   *"some development used sequential leaderboard feedback"* add: *"one
   intermediate FETV step modified only clips in the scored subset; it is
   documented in the repository and is worth +0.0018 against a +0.0110 margin
   over the next-ranked team."* Then the poster bullet stands as written.
2. **Drop the bullet from the poster** and keep the generic sentence in both.
   The repository still carries the full record for anyone who looks.

Option 1 is stronger. The repository is registered with the organizers, the
script and the intermediate artifacts are visible in it, and the analysis is
already written — being the one to say it first is worth more than the two lines
it costs.

**4.4 QR code** → `github.com/hwkim3330/aicity2026`

---

## Held ready but not printed

For questions, not for the board:

- Why the ranks are what they are — argue what was isolated, not the placement
- FETV single-pass reproduction matches `v6_fewshot` on 67.0% of fields; six
  explanations for the residual tested and ruled out; cause not established
- PSI BCQ and TAR BCQ cannot reproduce byte-for-byte — the seeds were never
  generated, so there is nothing to recover
- BERTScore configurations differ by benchmark (`roberta-large` rescaled for TAR,
  IDF-weighted `deberta-xlarge-mnli` for FETV)

---

## Deadlines fixed by the committee email, 2026-08-19

Workshop is a half-day ECCV session on **8 September, 08:30-12:30**; badge pick-up
from 08:00 and there is usually a queue on day one. Presentations are strongly
expected on-site; a proxy presenter is the remedy if the author cannot attend, and
virtual is accepted only for visa or medical reasons.

Uploads go to the committee's Google Drive under the subfolder for our paper ID,
by **4 September, 23:59 AoE**:

| file | required? |
| --- | --- |
| `XX-paper.pdf` camera-ready | yes |
| `XX-slides.pdf` | orals only |
| `XX-oral.mp4`, 4 min | virtual presentations only |
| `XX-poster.pdf` | optional, recommended |

**Two things still need confirming before these can be produced:** our paper ID
(the `XX` in every filename), and whether we are assigned an oral. Orals are 4
minutes, strictly enforced, plus a group Q&A per track.

## Production notes

- Reuse the three paper figures as vector PDFs; do not rasterise.
- The annotated frame is 900 px — fine up to about 15 cm wide at 150 dpi. If the
  board is large, re-export from the 1920×1920 original with
  `scripts/annotate_fetv_case.py` after raising the downscale target.
- Cite FETV and FishEye8K under the frame, as the paper does.
- Every rank names its board. Every number on the poster appears in the paper.
