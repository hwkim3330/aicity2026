
### The grounding sentence is worth nothing: ordering is the whole effect

Ran a third arm over the same 321 items, same seed: `answer_first_grounded` is
`answer_first` plus one sentence warning that the box is often small or distant
and that a nearer, more obvious pedestrian is usually *not* the subject.

| | accuracy | vs shipped | parse-fail |
| --- | ---: | --- | ---: |
| shipped | 0.2804 | — | 11.8% |
| answer_first | 0.4424 | 82W/30L, p<1e-5 | 0.0% |
| answer_first_grounded | 0.4424 | 83W/31L, p<1e-5 | 0.0% |

Head to head the two variants are **5W/5L, p=1.000** -- identical to four decimal
places and a coin flip item by item.

This refutes the hypothesis the sentence was written to test. `red_box_stats.py`
measured a real 5.6-point accuracy gap between items whose box covers ≥1% of the
frame and the rest (0.3008 vs 0.2447), and the poster's own 3/24 → 9/24 result
came from re-locating the target. I read that as an instruction-shaped gap. It is
not: telling the model the box may be small does not make it see a small box. The
correlation is perceptual, and prompting does not reach it.

What survives is narrow and solid -- **ordering alone, +0.162.** Nothing about
the content of the reasoning mattered; only whether the letter was emitted before
it or after it.
