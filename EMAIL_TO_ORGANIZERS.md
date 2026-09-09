# Draft email — FETV eligibility determination

To: aicitychallenges@gmail.com
Subject: Team 277 (Korea Drive) — FETV eligibility determination, request for the recorded reason

---

Dear AI City Challenge organizers,

Thank you for running the 10th challenge and for completing the code and
reproducibility reviews. We are writing about Team 277 (Korea Drive) and the FETV
leaderboard, where we were told at the workshop that our entry was removed for using
a different model.

**We are not contesting the outcome.** We have since audited our own submission and
found a reproducibility gap that we consider sufficient on its own; it is described
below. We are asking for two things: the reason as it stands in your records, and
guidance on the standard we should meet next time.

**One point of fact, in case the characterization carries forward.** All three of our
scored runs — TAR, FETV and PSI-VQA — used a single frozen `Qwen/Qwen3-VL-8B-Instruct`
in bf16, with no adapter and no task-specific parameter update. Three records already
say so: the evaluation portal carries `models_used: qwen3` on our scored TAR
submission; the challenge summary paper attributes "Qwen3" to Team 277 in the Track 3
table; and our camera-ready states a "frozen Qwen3-VL-8B-Instruct inference pipeline"
throughout. Our team also remains listed on the PSI-VQA board, which would be
inconsistent with a unified-system finding, since such a finding would apply across
leaderboards rather than to FETV alone.

We think we know where the impression came from, and both sources are our own fault:

1. The manuscript we submitted misstated the TAR backbone as Qwen2.5-VL-7B at 4-bit.
   We found this ourselves on 2026-08-03, published the correction in our public
   repository before results were released, and carried it into the camera-ready.
2. A track-level README in that repository repeated the same error and was missed in
   that sweep. It has since been corrected.

**The reproducibility gap, stated plainly.** Our FETV artifact is the last of an
eleven-step chain rather than the output of one run, and the commands behind two of
those steps were never recorded. Re-running our published pipeline reproduces none of
the 200 records exactly. We discovered this on 2026-08-13 and published it in the same
repository under the commit title "FETV does not reproduce, and the repository said it
would." We should have caught it before submitting, and we did not. By contrast our
PSI-VQA artifact does reproduce exactly, which we verified on all 328 records.

For completeness: measured against the metric rather than by byte comparison, the
re-run stays close where we can score it — 0.7820 against 0.7841 on
`answer_intersection_type`, and 162 of 200 timestamps inside the seven-second
tolerance. We mention this only for accuracy, not as a defense; two unrecorded steps
are two unrecorded steps.

**Our questions:**

1. Which finding does the FETV determination rest on in your records — model identity,
   reproducibility, or something else? We would like our internal record to be correct.
2. For future challenges, is a verified end-to-end reproduction of the submitted
   artifact from a single command the expected standard for award candidates? We have
   adopted that internally and would rather match your requirement than guess at it.

Thank you for your time, and congratulations to MR-CAS and UWIPL_ETRI.

Best regards,

Hyunwoo Kim, Yooseung Wang, Pusik Park
Korea Electronics Technology Institute (KETI) — Team 277, Korea Drive
hwkim3@keti.re.kr
Repository: https://github.com/hwkim3330/aicity2026
