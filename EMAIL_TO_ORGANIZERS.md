# Draft email — FETV eligibility determination

To: aicitychallenges@gmail.com
Subject: Team 277 (Korea Drive) — FETV determination, and the reproduction we can now provide

---

Dear AI City Challenge organizers,

Thank you for running the 10th challenge and for completing the code and reproducibility
reviews. We are writing about Team 277 (Korea Drive) and the FETV leaderboard, where we
were told at the workshop that our entry was removed for using a different model.

**We are not contesting the outcome.** We audited our own submission afterwards, found a
real gap, and would rather report it accurately than argue. Two things below: what we can
now reproduce, and what we cannot.

## What we can now reproduce

Our scored FETV artifact regenerates byte-for-byte from committed code:

```
cd track3_anomaly/scripts && python3 rebuild_fetv_v11.py --verify
# rebuilt from v7: 200/200 rows identical to the shipped v11
```

The output hashes to `39abdb0a8cca7a7fa18dbd31374ee353e032977df9928d54734a53e9ec43e835`,
the same SHA256 as the submitted file. The script and every input it reads are public at
https://github.com/hwkim3330/aicity2026.

This corrects our own earlier description of the pipeline. We had recorded it as an
eleven-stage chain including two model re-inference passes, and concluded it could not be
rebuilt. That was wrong. The intermediate that looked like a second inference stage was a
second pass we ran and then discarded — the artifact after it is identical to the one
before it on 199 of 200 rows — and everything from the first pass onward is deterministic
post-processing over archived files with no model call.

## What we cannot reproduce, stated plainly

Two gaps remain, and the script names both rather than papering over them.

1. **The first pass does not regenerate.** Re-running our published inference over the 200
   clips produces output that matches no archived version exactly. We have ruled out seven
   explanations by measurement — run-to-run nondeterminism, the determinism pin, clip
   encoding, frame sampling, the few-shot exemplars, the NumPy upgrade, the model weights
   (same cache blob hashes), and the prompt as committed. What remains is that the first
   pass was written at 05:48 on 2026-07-11 and the pipeline was committed at 22:51 the
   same day, so the code that produced it was an uncommitted working tree that no longer
   exists. That is our failure of record-keeping, not a claim of bad luck.

2. **Fifteen rows are a lookup table, not a derivation.** One step fills a violator type
   and colour into 15 `no_violation` rows. Fourteen of those fifteen values are our own
   model's output in our first two submissions and are traceable; one colour appears in no
   archived artifact; and the selection of those 15 clips out of 64 equally qualified ones
   follows no rule we can recover — edited and unedited candidates are statistically
   indistinguishable. Something outside the pipeline chose them, and we cannot say what.

We discovered the reproducibility problem ourselves on 2026-08-13 and published it in the
same public repository, under the commit title "FETV does not reproduce, and the
repository said it would," five weeks before the results. We should have caught it before
submitting.

## One point of fact about the model

All three of our scored runs — TAR, FETV and PSI-VQA — used a single frozen
`Qwen/Qwen3-VL-8B-Instruct` in bf16, with no adapter and no task-specific parameter
update. The evaluation portal carries `models_used: qwen3` on our scored TAR submission;
the challenge summary paper attributes "Qwen3" to Team 277 in the Track 3 table; our
camera-ready states a "frozen Qwen3-VL-8B-Instruct inference pipeline" throughout; and our
team remains listed on the PSI-VQA board, which a unified-system finding would not leave
standing.

Where the other impression could have come from is our own fault twice over. The
manuscript we submitted misstated the TAR backbone as Qwen2.5-VL-7B at 4-bit; we found
that ourselves on 2026-08-03, published the correction before results, and carried it into
the camera-ready. A track-level README in the repository repeated the same error and was
missed in that sweep; it is corrected now.

## Two questions

1. Which finding does the FETV determination rest on in your records — model identity,
   reproducibility, or something else? We would like our own record to be accurate.
2. For future challenges, is byte-exact regeneration of the submitted artifact from a
   single command the standard you apply to award candidates? We have adopted that
   internally and would rather match your requirement than guess at it.

Thank you for your time, and congratulations to MR-CAS and UWIPL_ETRI.

Best regards,

Hyunwoo Kim, Yooseung Wang, Pusik Park
Korea Electronics Technology Institute (KETI) — Team 277, Korea Drive
hwkim3@keti.re.kr
Repository: https://github.com/hwkim3330/aicity2026
