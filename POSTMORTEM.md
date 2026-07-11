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
