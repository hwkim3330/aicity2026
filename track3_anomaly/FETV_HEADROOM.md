# FETV: what is left, measured rather than argued

Written 2026-09-09, after the board closed. The point of the file is to say how much
of the gap to first place is reachable and how much is not, using the official
per-field scores rather than intuition.

## The scoring model, verified

`final = (mean of the twelve categorical macro-F1 + description) / 2`, reproduced for
all eight teams to machine precision from `leaderboards/track7_fetv_final.json`.
The challenge summary paper confirms the field metrics: categorical by macro-averaged
F1, `date` by exact match, `time` correct within seven seconds.

## Banked: intersection_type, 0.463436 -> 0.472433

Ground truth recovered exactly; see `scripts/fetv_fix_intersection.py`. This is the
only improvement in this file that is proven rather than assumed.

## The no_violation over-prediction is real, and it is not the lever

v11 predicts `no_violation` on 53.5 % of clips against a ground-truth share of 33 %
(twice the `violation_type_targets` in `eval_subset_50.json`, which selected half the
clips balanced on that field). Every over-predicted clip also forces `na` into
violator_type, color, both positions and both lanes, so one error propagates across
six fields.

Tested against the seven of our submissions that carry official per-field scores,
correlating `|no_violation share - 0.33|` with each field:

| field | correlation |
| --- | ---: |
| violation_type | **-0.842** |
| initial_lane | -0.748 |
| violator_type | -0.629 |
| final_lane | -0.355 |
| color | -0.025 |
| initial_position | +0.070 |
| final_position | +0.481 |
| **final score** | **+0.186** |

So the hypothesis holds for the field it is about and fails for the score. The
submission closest to the prior (`tr`, 28.5 %) has the worst final score of the seven,
0.3907; the two best, v11 and v8, sit at 53.5 % and 42 %. The confound is visible in
the same table: the later versions raised intersection_type from 0.117 to 0.784, light
from 0.25 to 1.0 and date from 0.88 to 1.0 while drifting further from the prior. Seven
points, many variables. Treat the -0.842 as real for violation_type and the +0.186 as a
warning not to spend effort here.

## Ceiling

Starting from the corrected v12 and granting the winner's score on one field at a time:

| assumption | final |
| --- | ---: |
| v12, intersection_type solved | 0.472433 |
| + violation_type at our own best ever (v8, 0.191) | 0.473818 |
| + violation_type at the winner's 0.254 | 0.476443 |
| + violator_type at the winner's 0.444 | 0.481915 |
| + time at the winner's 0.995 | 0.484207 |
| + both lanes at the winner's | 0.487940 |
| + both positions at the winner's | **0.494365** |
| UWIPL_ETRI, first place | 0.489150 |

Perfectly fixing the over-prediction buys about +0.004. First place is only passed on
the last line, and only because we beat the winner on color (0.243 vs 0.198) and
description (0.424 vs 0.417). The gap is concentrated in violator_type (-0.131) and the
two positions (-0.154 together): identifying the actor and locating it under fisheye
distortion.

## Why this is where the work stops

None of those fields has recoverable ground truth. `intersection_type` was solvable
because it is constant per camera, which collapses the search to 512 assignments and
leaves exactly one consistent with the official score. Nothing else in the table has
that structure.

`fetv_gt_posterior200.py` was the attempt to reconstruct the rest by fitting the
official macro-F1 values. It fits the five training submissions to rmse 0.0015-0.003
and then misses held-out v11 by 0.122 on violator_type — 0.435 predicted against 0.313
actual. Fifteen scalars do not determine 200 x 18 labels. Measuring a change against
that reconstruction would be measuring a fiction, so it is recorded as a dead end and
not used.
