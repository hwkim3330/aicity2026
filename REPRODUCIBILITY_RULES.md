# Rules for the next challenge, derived from being removed from one

Written 2026-09-09. Korea Drive was removed from the FETV leaderboard after the
organizers' final code and reproducibility review. This file exists so the same thing
does not happen twice. Each rule names the specific failure that produced it.

## What the review actually was

The 2026 winners page publishes a **Verified Score** column for FETV alongside the
public score, and both listed teams moved: UWIPL_ETRI 0.4891 → 0.4629, MR-CAS
0.4884 → 0.4848, which reversed the award order. So the review is not a document
check — the organizers re-run the team's published code and score what comes out.

The FAQ states the purpose: verification exists "to ensure that no private data or
private pre-trained model was used for training and **the tasks were performed by
algorithms and not humans**," and that "using test data during training or for model
selection invalidates eligibility for awards."

Our own outcome fits that exactly. Our PSI-VQA artifact reproduces from the repository
on all 328 records and drew no sanction; our FETV artifact reproduces on none of 200
and was removed. Same team, same model, same paper — the difference is reproducibility.

## The rules

1. **One command, end to end, or it is not a submission.** The FETV artifact was the
   last of an eleven-step chain: v7→v8 rewrote 56 rows, v8→v9 another 57, v9→v10
   fifteen, v10→v11 ninety-two. `REPRODUCE.md` listed a single command and an expected
   SHA256 next to it, which was never true.

2. **A step that is not a committed script did not happen.** The commands behind v9 and
   v10 were never recorded and cannot be recovered. Unrecorded edits that rewrite answer
   rows are indistinguishable from a human editing the answers, which is the specific
   thing the verification is designed to detect.

3. **Verify the reproduction before submitting, not a month after.** Running
   `reproduce_*.sh` and diffing against the artifact takes one afternoon. We did it on
   2026-08-13, five weeks after the deadline, and it failed.

4. **Byte comparison is the check to run, but the metric is the check that matters.**
   The two are different: the FETV re-run matched 0 of 200 records yet scored 0.7820
   against 0.7841 on the one field with recoverable ground truth, with 162 of 200
   timestamps inside the metric's seven-second tolerance. Report both. Do not conclude
   "does not reproduce" from `diff` alone, and do not conclude "reproduces" from a close
   score alone.

5. **Never derive an answer from the leaderboard.** Solving
   `answer_intersection_type` by inverting our own official score recovers real ground
   truth for 87 clips and would raise the score from 0.4634 to 0.4724. It is also
   exactly what the FAQ prohibits — test set annotations, and model selection on test
   data. The same insight reached honestly, from junction names burned into the video
   plus a per-source majority vote, is worth 0.4697 and is submittable. Use that one.

6. **Sweep every document, not the ones you remember.** The TAR backbone error was
   found on 2026-08-03 and corrected in `OFFICIAL_RESULTS.md`, `REPRODUCE.md` and the
   camera-ready. `track3_anomaly/README.md` repeated it and was missed, and it stood in
   the repository the paper cites throughout the review window. `grep -rn` the wrong
   claim across the whole tree, not the files that come to mind.

7. **Pin the model revision at run time.** No Hub revision was persisted for any
   official run. One was recovered afterwards by argument rather than by record, which
   is not the same thing and cannot be checked by a third party.

8. **Check whether a declaration field exists before assuming you skipped it.** An
   earlier version of this file blamed us for leaving `models_used` empty on Track 7.
   That was wrong: the portal export shows 27 of 27 teams carrying the field on Track 3
   and **0 of 8 on Track 7, 0 of 7 on Track 8** — the FETV and PSI-VQA forms had no such
   field. The declaration duty sits in the technical report, which is where ours is.
   The rule that survives is the general one: verify a claim about your own record
   against the record before repeating it.

## The check to run before any future submission

```
run the documented reproduce command from a clean checkout
diff the output against the artifact, record the match rate
score both against whatever ground truth exists, record both numbers
confirm every row-modifying step is a committed script
confirm no field was derived from leaderboard feedback
```

None of that is expensive. All of it was skipped.
