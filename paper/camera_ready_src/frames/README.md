# Figure frames — source and terms

Three frames of `019_004.mp4` from the **FETV** public dataset, at t = 0.00,
1.50 and 3.00 s. This is the clip discussed in the paper's FETV coupling
analysis: one submitted version predicts `no_violation` with `na` for every
dependent field, a later one predicts `red_light`, a yellow car, Top-Left to
Middle-Center motion, and lanes 1 to 2, while date, time, weather, lighting and
intersection type stay identical.

The frames also show why the global fields score 1.000 while the
violator-centric ones do not: the date and time are burned into the overlay at
top left and the weather and lighting are properties of the whole scene, whereas
identifying which road user committed the violation requires resolving the
actual traffic situation under fisheye distortion.

## Source and attribution

- Dataset: FETV, <https://github.com/MoyoG/FETV> — 200 public fisheye
  intersection clips, distributed by the dataset authors through the Google
  Drive folder linked from that repository's README.
- Cite FETV and FishEye8K when a frame is reproduced.

## Terms — read before publishing

The FETV repository ships **no LICENSE file and no terms section**. What can be
said is that its README embeds three frames from the dataset as public examples,
so illustrating the dataset with an attributed frame is consistent with how its
authors present it. That is a norm, not a grant, and the decision to publish
rests with the corresponding author.

**PSI-VQA frames must not be used.** PSI-VQA inherits the TASI Benchmark Data
Sharing Agreement. TAR clips come from third-party source datasets with their own
terms. FETV is the only one of the three where this question is arguable at all.

## If a frame goes into the paper

The paper currently states, in the worked PSI case caption, that *frames are not
redistributed under the benchmark data terms*. That sentence has to be narrowed
to PSI-VQA, or it will contradict the figure.

## Regenerating

Not committed from the dataset directly — produced by

```bash
python3 scripts/render_case_studies.py --case fetv_violator_misselection \
    --data-root track3_anomaly/data/fetv/FETV_public_clips
```

then downscaled to 900 px and saved as JPEG at quality 92. The originals are
1920×1920 PNGs of about 4 MB each, which is far more than a two-column figure
can display. `docs/case_studies/rendered/` stays gitignored; these copies exist
because they are figure assets for a specific paper, not a dataset mirror.
