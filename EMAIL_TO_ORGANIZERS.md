# Draft email — FETV determination and our reproducibility audit

To: aicitychallenges@gmail.com (cc: the Track 7 organizer 왕유승 spoke with on site)
Subject: Team 277 (Korea Drive) — FETV determination: our reproducibility audit, and one question about the reason recorded

---

Dear AI City Challenge organizers,

Thank you for running the 10th challenge and for completing the code and reproducibility
reviews. We are writing about Team 277 (Korea Drive) and the FETV leaderboard, where we
were told at the workshop that our entry was removed because we had used a different
model.

Since the announcement we have audited our own submission end to end. **The artifact
does not reproduce**, and we are not asking you to reverse the decision on that point.
What we can now do is say exactly where it stops reproducing, which we could not do
before, and we have one question about the reason recorded against us.

## 1. What does regenerate, and from where — stated precisely

Our pipeline produced eleven intermediate files. Only one of them is a model run whose
output feeds the final artifact; the rest are post-processing. Starting from the fourth
intermediate, the scored submission regenerates byte-for-byte

```
git clone https://github.com/hwkim3330/aicity2026
cd aicity2026/track3_anomaly/scripts && python3 rebuild_fetv_v11.py --verify
# rebuilt from v7: 200/200 rows identical to the shipped v11
```

The rebuilt file hashes to
`39abdb0a8cca7a7fa18dbd31374ee353e032977df9928d54734a53e9ec43e835`, the same SHA256 as
the file we submitted. Every input it reads is in the public repository.

This also corrects our own earlier description of the pipeline. We had recorded it as an
eleven-stage chain including two model re-inference passes and had concluded it could
not be rebuilt. That was wrong: the stage that looked like a second inference pass was
one we ran and then discarded — the artifact after it is identical to the one before it
on 199 of 200 rows — and everything from the first pass onward is deterministic
post-processing over archived files, with no model call.

That is **not** a reproduction of our submission. It starts from an archived
intermediate, and two of the steps upstream of it are themselves unrecoverable. We say
so here because the SHA match above would otherwise read as more than it is.

## 2. The three points where it stops reproducing

**The model run does not regenerate.** Re-running our published inference over the 200
clips produces output matching no archived version exactly. We have ruled out seven
explanations by measurement — run-to-run nondeterminism, the determinism pin, clip
encoding, frame sampling, the few-shot exemplars, the NumPy upgrade, the model weights
(identical cache blob hashes), and the prompt as committed. What remains is that the
first pass ran at 05:48 on 2026-07-11 while the pipeline was committed at 22:51 the same
day, so the code that produced it was an uncommitted working tree that no longer exists.
That is a record-keeping failure on our side.

**The timestamp correction does not regenerate.** The step after the model run rewrote
163 of 200 timestamps. Those values appear in no earlier artifact, and reading the
clock burned into the video shows them falling four to six seconds after each clip's
first frame at no fixed offset. They were produced by a script that read the burned-in
clock and was never committed. The description rewrite in the same step *is* recovered
and matches 199 of 200 rows exactly.

**Fifteen rows are a lookup table, not a derivation.** A later step writes a violator
type and colour into 15 `no_violation` rows. Fourteen of those fifteen values are our
own model's output in our first two submissions and are traceable; one colour appears in
no archived artifact; and the selection of those 15 clips out of 64 equally qualified
ones follows no rule we can recover. We cannot account for how they were chosen.

We found the reproducibility problem ourselves on 2026-08-13 and published it in the same
public repository, under the commit title "FETV does not reproduce, and the repository
said it would," five weeks before results.

## 3. On the reason given

All three of our scored runs — TAR, FETV and PSI-VQA — used a single frozen
`Qwen/Qwen3-VL-8B-Instruct` in bf16, with no adapter and no task-specific parameter
update. Four independent records agree: the portal carries `models_used: qwen3` on our
scored TAR submission; the challenge summary paper attributes "Qwen3" to Team 277 in the
Track 3 table; our camera-ready states a "frozen Qwen3-VL-8B-Instruct inference pipeline"
throughout; and Team 277 remains listed on the PSI-VQA board, which a unified-system
finding would not leave standing.

We also trained one LoRA adapter on 2026-07-08 while exploring. It scored 0.636 on our
local proxy against 0.772 for the few-shot base model, so it was abandoned; it was never
used to produce any submission on any track, and the checkpoint no longer exists.

Where a different impression could have come from is our own fault. The manuscript we
submitted misstated the TAR backbone as Qwen2.5-VL-7B at 4-bit. We found that ourselves
on 2026-08-03, published the correction before results were released, and carried it
into the camera-ready, whose final PDF contains no occurrence of "Qwen2.5". A
track-level README in the repository repeated the error and was missed in that sweep; it
is corrected now.

## 4. What we are asking

We note that the published FETV table lists a Verified Score alongside the public score,
and that both listed teams moved in it — one by 0.026, which changed the award order.
That suggests divergence under re-running is normally handled by re-scoring. Our
questions follow from that:

1. Which finding does our determination rest on in your records — model identity,
   reproducibility, or something else? We would like our own record to be accurate,
   whichever it is.
2. If the finding was reproducibility, we accept it — section 2 is our own account of
   why, and we would only ask that the reason be recorded as reproducibility rather than
   model identity, since the four records above all show one backbone.
3. For future challenges, is byte-exact regeneration of the submitted artifact from a
   single command the standard applied to award candidates? We have adopted it
   internally and would rather match your requirement than guess at it.

Thank you for your time, and congratulations to MR-CAS and UWIPL_ETRI.

Best regards,

Hyunwoo Kim, Yooseung Wang, Pusik Park
Korea Electronics Technology Institute (KETI) — Team 277, Korea Drive
hwkim3@keti.re.kr
Repository: https://github.com/hwkim3330/aicity2026
