# AI City Challenge 2026 Postmortem

This document separates official results, observed failure modes, and proposed
follow-up experiments. Proposed fixes are hypotheses until re-evaluated on a
proper held-out set.

## Track 7: FETV

### Official result

Korea Drive finished 3rd with 0.4634:

| Component | Korea Drive | Winner | Difference |
|---|---:|---:|---:|
| Description | 0.4238 | 0.4171 | **+0.0067** |
| Categorical mean | 0.5031 | 0.5612 | **-0.0581** |
| Final | 0.4634 | 0.4891 | -0.0257 |

The final score is the mean of description and categorical scores. Matching
the winner's categorical mean while retaining our description score would
have yielded 0.4925 and first place.

### What failed

The pipeline asked one VLM pass to identify the violator and jointly predict
violation type, actor type, color, start/end position, start/end lane,
intersection, weather, lighting, date, and time. A wrong violator selection
therefore caused correlated errors across several fields.

Position and lane are geometric outputs, but the implementation largely
delegated them to language prompting. The lane convention depends on the
actor's direction of travel, while position uses a center-square 3x3 grid;
both require stable tracking and coordinate transforms. Macro-F1 also gives
rare violation classes substantial influence, making prior-heavy predictions
fragile.

The generated `*.results.json` files showing 1.0 were format/self-consistency
checks, not ground-truth evaluations. They could not guide field-level model
selection. Aggregate leaderboard feedback was insufficient to identify which
rows or fields were wrong, and posterior corrections from a few submissions
did not generalize reliably to position and lane.

The final server's field-level scores make the bottleneck explicit:

| Field | Score |
|---|---:|
| Violation type | 0.1578 |
| Violator type | 0.3127 |
| Color | 0.2434 |
| Initial / final position | 0.1239 / 0.1278 |
| Initial / final lane | 0.1780 / 0.1694 |
| Intersection type | 0.7841 |
| Weather / lighting | 1.0000 / 1.0000 |
| Date / time | 1.0000 / 0.9400 |

The metadata pipeline worked; actor-centric spatial reasoning did not.

### Redesign

1. Detect and track all candidate road users with an object detector and
   multi-object tracker.
2. Identify the violation interval and violator before predicting attributes.
3. Use OCR for date/time and deterministic parsing of the overlay.
4. Use lane segmentation, a vanishing-point model, and actor heading to assign
   direction-relative lane numbers.
5. Compute 3x3 positions directly from tracked coordinates in the prescribed
   center-square crop.
6. Use dedicated classifiers for color, weather, lighting, and intersection.
7. Ensemble trajectory rules with a VLM only for violation type.
8. Generate the description from the frozen structured record, preserving the
   already competitive language score.

An executable detector-agnostic implementation of these interfaces is in
`track3_anomaly/scripts/fetv_structured_pipeline.py`, with a sample scene and
synthetic geometry/schema tests. It is a post-challenge prototype and has no
official score.

## Track 6: Cross-City Object Detection

### What completed

The managed RF-DETR experiment succeeded on Hafnia. Hidden benchmark inference
produced 275,159 detections covering 14,814 image IDs, recovered byte-exactly
from 451 log chunks. The benchmark was documented as containing 14,925 images,
so 111 images had no recovered detection record.

### Why the portal submissions likely failed

The portal exposes only `Failed`, without an evaluator traceback, so the exact
cause cannot be proven. The strongest evidence points to a submission-contract
failure rather than model quality:

1. The official workflow says Hafnia generates the prediction artifact in the
   evaluator's required format. Because artifact retrieval was unavailable,
   we reconstructed a custom flat COCO-style JSON list from stdout logs.
2. The records included custom fields (`file_path`, `sample_index`, and
   `category_name`) and used a schema that was never validated against a
   successful Track 6 portal example.
3. The custom benchmark code and repository README explicitly left the real
   Hafnia submission schema as a login-dependent verification step.
4. 111 benchmark images had no records. This can be valid in ordinary COCO
   evaluation when an image has zero detections, but a strict challenge parser
   may require explicit image coverage.
5. The file was 64.5 MB with 275k rows. Size or evaluator resource limits are
   secondary possibilities, though the portal accepted the upload before
   marking evaluation failed.

The correct redesign is to test a minimal file against the portal early, use
the exact artifact emitted by Hafnia, validate schema and image coverage, and
retain a lower-confidence top-k export to reduce file size. Track 6 should
remain in the archive as an attempted track with no scored result, not be
presented as a leaderboard participation result.

## Track 8: PSI-VQA

### Official result

Korea Drive finished 5th with 57.0400:

| Component | Score |
|---|---:|
| BCQ Macro-F1 | 0.5045 |
| Open QA Cue-F1 | 0.6019 |
| MCQ Accuracy | 0.6044 |
| Temporal mIoU | 0.5708 |

OpenQA was competitive and temporal localization improved substantially from
an early 0.0253. BCQ and MCQ remained the main gaps.

### What failed

The target pedestrian is identified by a brief red box, but whole-video VLM
inference can attend to another pedestrian after the box disappears. That
identity error changes every downstream intent judgment. The initial pipeline
also lacked explicit representations of curb distance, road-relative motion,
body orientation, gaze, and stop/go transitions.

### Redesign

1. Detect the red annotation and initialize a target-specific tracker.
2. Crop the target with sufficient context rather than repeatedly passing the
   full frame.
3. Extract pose, optical flow, curb/road segmentation, and target trajectory.
4. Convert observations into facts such as stationary, road-parallel,
   approaching curb, entering roadway, crossing, yielding, and looking at the
   ego vehicle.
5. Answer BCQ and eliminate MCQ choices from those shared facts.
6. Retain the strong cue-style OpenQA format and fit temporal priors only on
   training data, with video-grouped cross-validation.

The post-deadline `psi_vqa_submission_v8_final.csv` is an experimental
artifact, not an official submission. Its local validation must not be
reported as leaderboard performance.

## Track 5: Generative Forecasting

### Official result

Korea Drive finished 10th with 70.3004. CLIP was 0.9590, the highest value in
the supplied top-ten final table, while FID (49.7246), FVD (33.6228), and SSIM
(0.6198) exposed visual-distribution and temporal-consistency weaknesses.

### Interpretation and redesign

The system preserved caption semantics but changed appearance and motion too
aggressively. A stronger solution would preserve static background and camera
geometry, generate only moving regions, share a latent motion trajectory
across frames, and train with optical-flow/temporal consistency objectives.
Candidate selection should balance LPIPS and temporal consistency instead of
optimizing CLIP alone. Any fine-tuning must use only permitted data and be
fully disclosed.

## Operational lesson

Model selection and submission are separate engineering systems. A future run
should freeze a verified candidate at least 30 minutes before the deadline,
upload it first, and continue experiments only in parallel. Checkpointed
inference and deadline-safe assemblers help only when the final upload itself
has an explicit owner and cutoff.

## Paper framing

The useful research story is not "a larger VLM solves traffic understanding."
It is that language generation transferred well, while spatial identity,
geometry, and temporal consistency required explicit structured modules. The
paper should validate that claim with clean video-grouped ablations and report
official results separately from post-deadline experiments.

## The PSI MCQ CV harness disagrees with the official score by 2x (2026-08-21)

Running `scripts/psi_mcq_cv.py` over all 321 labelled train MCQ items at the
shipped 16-frame budget scores **90/321 = 0.2804**. The official Track 8 MCQ
score is **0.6044**, which is exactly `55/91` -- simple accuracy over the 91
test items. Chance on a 4-option question is 0.25, so the harness says the
pipeline is barely above guessing on the same task it scored 0.60 on.

Everything checkable was checked and none of it explains the gap:

* **labels** -- each `answer` field's letter matches the same-lettered option in
  the question, text and all; the distribution is A 86 / B 80 / C 84 / D 71.
* **prompt path** -- no `--variant`, so it uses `prompts.py`'s `psi_mcq` suffix
  and calls `backend.answer(video, "psi_mcq", question)`, the same entry the
  official run used.
* **videos** -- 1280x720, ~6 s, 181 frames, zero missing for the sampled items.
* **metric** -- `55/91` lands on 0.6044 exactly, so the official number is plain
  accuracy and not a partial-credit variant.
* **distribution** -- both splits are entirely the `ambiguous` subset with 33.0%
  negative-polarity questions and four options each. If anything the test split
  is *easier*: mean pairwise Jaccard between options is 0.139 against train's
  0.163, so its distractors overlap the answer less.
* **polarity** -- CROSS 0.332 vs NOT 0.290 on resolved items, so negation
  handling is not the cause either.

91 items give a binomial standard error of 5.1 points, so sampling does not cover
a 32-point gap.

**What this invalidates.** Every past decision taken on this harness loses its
support, including the three stored runs in
`track3_anomaly/psi_mcq_cv_results/` (0.128, 0.375, 0.333 -- all at or below
chance) and the 24-item paired grounding study that the poster and the talk both
report. That study's conclusion already failed to transfer: it won all six
discordant pairs at p=0.0312 and then scored 53.19 against 55.41 on the real
board. This is the same harness telling the same kind of lie, and the leaderboard
already caught it once.

**What is not yet known** is which side is wrong. Either the train labels do not
match their videos, or the pipeline behaves differently on the train path. Until
that is settled, this harness cannot gate a submission, and the dense-frame
16-vs-32 comparison running today can be recorded but not acted on.

### Grounding difficulty does not explain it either (same day)

Reading three of the disagreements frame by frame, the model's description
matched the video better than the label did in all three -- in each case the red
box marked a small or distant subject while something more salient moved through
the frame. That suggested the train split simply has harder grounding, which
would have explained the gap *and* the poster's re-location finding at once.

Measured instead of assumed. `scripts/red_box_stats.py` isolates the drawn
overlay by colour on the first frame and reports its extent:

| | items | videos | box area, median | under 0.1% of frame |
| --- | ---: | ---: | ---: | ---: |
| train | 321 | 118 | **0.01770** | 16% |
| test | 91 | 42 | **0.01764** | 8% |

The boxes are the same size. Grounding difficulty is refuted as the explanation,
and with it the tidy story that the harness was merely measuring a harder split.

Two smaller things fell out. Questions are not one per video -- train runs 2.7
questions per video and test 2.2 -- so any per-video count taken as an item count
is wrong, which is how a first pass here produced a bogus "227 vs 25". And the
colour threshold finds a box on 69% of train videos against 29% of test, which is
as likely to be the threshold as the data and is not evidence of anything on its
own.

So the 2x gap is recorded as unexplained. Seven candidate causes are ruled out
above; three hand-read examples are not enough to conclude the labels are wrong,
and that remains the only surviving hypothesis.

### Box size matters and still does not close the gap (same day)

Splitting the 321 train items by the measured box size:

| subset | n | accuracy |
| --- | ---: | ---: |
| all | 321 | 0.2804 |
| box detected | 227 | 0.2775 |
| box not detected | 94 | 0.2872 |
| box >= 1% of frame | 133 | **0.3008** |
| box < 1% of frame | 94 | **0.2447** |

Grounding is real -- a larger box is worth 5.6 points -- but the best subset
available is 0.3008 against an official 0.6044, so it is a contributing factor
and not the explanation. The lower tail does not help either: test's 5th
percentile box (0.00008 of frame) is four times *smaller* than train's (0.00032),
on 25 videos.

One thing this does settle: the "227 labelled PSI clips" the earlier notes refer
to is the subset with a detectable red box, and it scores 0.2775 against the full
set's 0.2804. So the gap is not an artefact of which subset past runs used.

Eight candidate causes are now ruled out. Of five disagreements read frame by
frame, four had the model's description fitting the video better than the label
did -- but five cases cannot establish a labelling error across 321, and settling
it means a human reading a few dozen. Closing this thread here rather than
guessing further.

### Reading the disagreements by eye: the labels are mostly fine (2026-08-21)

The last surviving hypothesis was that the train labels do not match their
videos. Twelve disagreements were pulled up frame by frame -- five earlier plus a
random sample of the rest. The result argues against it.

| what the frames showed | count |
| --- | ---: |
| label correct, model simply wrong | 4 |
| subject too small or distant to judge at all | 5 |
| label appears to contradict the video | 3 |

The three that look like label errors share a feature with the five unjudgeable
ones: the red box marks a small, distant subject while a different, salient
pedestrian moves through the frame. In `video_0153_track_23` the label says the
pedestrian stood on the sidewalk while someone walks clearly across the road --
but the box is a few pixels near the centre, not that person. So those are the
grounding failure again, not mislabelling.

Where the subject is visible enough to assess, the label is usually right and the
model is wrong. `video_0112_track_13`: two pedestrians walking between parked
cars, label says walking along the road, model says standing still. Model wrong.
`video_0136_track_43`: a pedestrian in pink walking diagonally across a lot,
label says walking diagonally, model says still on the sidewalk. Model wrong.

So the harness is measuring something real: on this split the pipeline genuinely
performs near chance. **The 2x gap with the official 0.6044 remains unexplained**
and the labelling hypothesis is now the weakest of the nine considered, not the
strongest. What is left is a property of the two splits that none of the measured
comparisons -- box size, polarity, option overlap, question length, class balance
-- has captured.

### The gap was a comparison that never held (2026-08-21, resolved)

There is no 2x gap to explain. The two numbers were never produced by the same
configuration.

`leaderboards/submission_history.json` records MCQ accuracy per submission:

| # | submitted | mcq_accuracy | temporal_miou |
| ---: | --- | ---: | ---: |
| 1 | 07-06 09:42 | **0.6044** | 0.0287 |
| 2 | 07-07 13:12 | 0.6044 | 0.2268 |
| 3 | 07-08 16:52 | 0.6044 | 0.4623 |
| 4 | 07-09 11:28 | 0.5495 | 0.4623 |
| 5 | 07-09 11:34 | 0.6044 | 0.4623 |
| 6 | 07-11 07:53 | 0.6044 | 0.5708 |
| 7 | 07-11 16:15 | 0.6044 | 0.5708 |

MCQ is identical to four decimals across six of seven submissions while
`temporal_miou` climbs from 0.0287 to 0.5708. The MCQ rows were written once and
carried forward; the work went into temporal. So the 0.6044 was set on **07-06**.

`git log -S psi_mcq -- scripts/prompts.py` returns exactly one commit, `c441d15`
on 07-11 at 22:51, which *adds* the `psi_mcq` entry and removes nothing. The
prompt this harness runs did not exist in the repository when 0.6044 was scored,
and that commit lands six hours after the last submission.

So today's comparison put a 07-11 prompt on the train split against a pre-07-06
configuration's test score and called the difference unexplained. Nine hypotheses
were tested against a gap that was an artefact of pairing the wrong two numbers:
labels, prompt path, videos, metric definition, split distribution, polarity,
grounding, subset selection, and reading twelve disagreements by eye.

What stands from that work: the labels are sound, box size is worth 5.6 points,
and the model does perform near chance on the train split with the current
prompt. What falls: the claim that this harness disagrees with the official
score, and with it the doubt cast on every decision made through it -- including
the 24-item grounding study in the poster. That study still failed to transfer
(53.19 against 55.41), which was always the stronger evidence and does not
depend on any of this.

The lesson is narrow and worth keeping: before comparing a local number to a
leaderboard number, check that the leaderboard number was produced by the code
you are about to run. A submission history that repeats a metric unchanged is
the signal that it was not recomputed.

## The answer-first result arrives after every deadline it could have served

Chasing the "2x harness gap" ended in two places. The gap itself was a
comparison that never held (previous section). But the harness built to chase it
then produced a real finding: emitting the answer letter before the reasoning
instead of after it moves PSI MCQ from 0.2804 to 0.4424 over all 321 labelled
train items, 82W/30L, p < 1e-5, with parse failures going 11.8% → 0.0%. A third
arm showed the effect is ordering alone -- adding a grounding sentence changed
nothing (5W/5L, p=1.000). Details in
[`track3_anomaly/ABLATIONS.md`](track3_anomaly/ABLATIONS.md).

**It cannot be used.** Track 3/8 submissions closed 2026-07-11, the repository
went to NVIDIA on 08-07, and camera-ready was 08-15. There is no artifact left to
improve and no document left to put it in. The shipped MCQ rows score 0.6044 and
came from a configuration predating this repository, so this variant cannot even
be compared against what was actually submitted -- only against the current
`psi_mcq` prompt, which was never scored by the organizers.

Recording it because the mechanism is portable and the next VLM task with a token
cap and a parsed letter will hit the same wall: **if a parser needs one token out
of a capped generation, emit that token first.** 11.8% of items here ran past the
cap mid-elimination and fell through to a literal `"A"`. Doubling the budget to
640 tokens left that at 11.2% -- the model was not short of room, it declined to
stop. Ordering makes running long harmless; a larger budget does not.

The honest summary of the night's Track 3 work: one false alarm diagnosed and
closed, one genuine finding produced too late to matter, one hypothesis (box-size
grounding) refuted.
