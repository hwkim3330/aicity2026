# Three edits required before printing, after the 2026-08-20 v8 result

Box-aware (routed + explicit red-box re-location) was uploaded to the re-opened
evaluation on 20 August and scored **53.19 against the previous General submission's
55.41**. The paired win on 24 items did not generalise. Three places currently say
otherwise.

---

## 1. Poster — PSI-VQA panel

**Currently:**
> Explicit re-location wins the six discordant pairs (p = 0.0312); at this sample
> size, 9/24 remains close to the 8/24 generic prompt.

**Replace with:**
> Explicit re-location wins all six discordant pairs (p = 0.0312) — but submitted to
> the full test set it scored **53.19 against 55.41**. A significant paired win on 24
> items did not predict the leaderboard.

The panel stops being "a prompt fix that works" and becomes the clearest negative
result on the board. That is a stronger thing to show, not a weaker one.

---

## 2. Slide 7 — PSI-VQA

**Currently:**
> All six discordant pairs favor explicit re-location (p = 0.0312), but the sample is
> still small.

**Replace with:**
> All six discordant pairs favor explicit re-location (p = 0.0312). Submitted to the
> full test set it scored 53.19 vs 55.41 — the paired win did not generalise.

---

## 3. Slide 9 (backup) — the one that matters most

**Currently, recommendation 01:**
> **PSI-VQA MCQ** — Add explicit red-box re-location, require a final letter early,
> reduce long elimination chains, and fail closed when the letter is absent.

This now recommends an action measured to be harmful. If the slide is shown during
Q&A it advises the audience to do the thing that lost 2.2 points.

**Replace with:**
> **PSI-VQA MCQ** — Red-box re-location was tried on the re-opened board and lost 2.2
> points (53.19 vs 55.41), so it is ruled out. What remains untested: require a final
> letter early, reduce long elimination chains, and fail closed when the letter is
> absent — all answer-contract fixes rather than prompt content.

---

## Why this is worth the edit rather than quietly dropping it

Three fixes in this project have now regressed a real score after a sound local
diagnosis — Track 1 ByteTrack, Track 4 rerank, and now this one. That pattern is
more interesting than any of the individual prompt results, and a workshop audience
building the same kind of system will recognise it. Presenting it deliberately is
better than being asked "did you check it on the full set?" and having to say yes,
and it was worse.

---

## 4. The colour, which text review missed

Rendering the poster and slides revealed a bigger problem than the wording.

**Orange is the "our result" colour** in this deck — the three headline scores, the
`v11 red_light` arrow, the winning `Duration prior` bar. And the
`Routed + red-box 9/24` bar is **orange**, and on slide 7 it is also the **longest**
bar. A viewer reads that as "this is what we found", and on a slide shown for thirty
seconds in a four-minute talk the grey caveat underneath will not register at all.

**Recolour that bar to grey** — the same treatment as `Shipped routed`. It was an
attempt, not a result. This matters more than the caption text, because the colour is
what gets read.

Same for **Takeaways 02** on the poster: `MCQ 3/24 → 9/24`. The arrow states an
improvement. Replace with `MCQ paired win 9/24 did not transfer (53.19 vs 55.41)`.

## 5. Keep slide 9's title

*"If we submit again: prioritize contract fixes, not broad rewrites"* — and the rule
of thumb below it, *"run one clean variant per task, report any improvement as
post-deadline evaluation"* — is exactly the protocol we followed on 20 August, and it
worked. It produced a clean, attributable negative result.

The irony is worth noticing: **the methodology the slide recommends is what refuted
the slide's own first recommendation.** The prompt-content fix failed; the three
remaining items are all answer-contract fixes and are untouched by this result. Fix
item 01, keep the title and the rule.
