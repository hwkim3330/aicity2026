# FETV disqualification — appeal draft

Korea Drive (Team 277), Track 7 / FETV, 2026 AI City Challenge.
Written 2026-09-08 after the on-site announcement that Korea Drive was disqualified
from FETV for using a different model. Every claim below is checkable from the
organizers' own published records or from the public repository cited in our paper,
`github.com/hwkim3330/aicity2026`.

## The rule

2026 FAQ, Q2:

> Task-specific prompts, parsing, and routing inside the pipeline are allowed, but
> teams may not submit separately tuned or specialized systems per task type or per
> test set.

Q7 repeats it per leaderboard. So the requirement is one system across TAR, FETV and
PSI-VQA — not membership of an approved model list. "A different model on FETV"
therefore means *different from the TAR submission*.

Q3 places the declaration duty in the technical report: "External data sources used
must be declared in the technical report." The FAQ does not define a portal
`models_used` field as the declaration of record.

## What the scored runs were

| task | scored artifact | model | precision |
| --- | --- | --- | --- |
| TAR | `submission_qwen3vl8b_v9.csv`, 0.4256 | `Qwen/Qwen3-VL-8B-Instruct` | bf16 |
| FETV | v11, 16 frames, 360x420 px/frame, greedy, 0.4634 | `Qwen/Qwen3-VL-8B-Instruct` | bf16 |
| PSI-VQA | `psi_vqa_submission_v7.csv`, 57.04 | `Qwen/Qwen3-VL-8B-Instruct` | bf16 |

One frozen backbone, no adapter, no per-task weights.

## The organizers' own records already say this

1. **The portal.** The scored TAR submission carries `models_used: qwen3`.
2. **The challenge summary paper.** *The 10th AI City Challenge* (arXiv 2608.17044)
   lists Team 277 in the Track 3 table at rank 24, mean 0.4256, attributing **Qwen3**
   — the organizers' own attribution, not ours. The same paper lists Korea Drive on
   the FETV board at 3/8 public (5/15 overall), 0.4634, with no ineligibility mark.
3. **The camera-ready.** Its abstract states that Korea Drive "uses a frozen
   Qwen3-VL-8B-Instruct inference pipeline with task-specific prompts, frame policies,
   output contracts, parsing, and optional calibration; **no task-specific parameter
   update or adapter generated the official predictions**," and that "the archived runs
   record the same model identifier, bf16 precision, and visual budget." The final PDF
   contains zero occurrences of "Qwen2.5" and zero of "4-bit".

## What we got wrong, stated plainly

The summary paper describes verification as: "Award-candidate teams were required to
provide reproducible code and models." At FETV rank 3 we were inside that band, and
the artifact a verifier would read is the repository cited in our paper. Two documents
in it contradicted the runs.

* **`track3_anomaly/README.md` — the most likely cause.** Under "Model choice" it read
  "**TAR baseline:** `Qwen/Qwen2.5-VL-7B-Instruct`, 4-bit NF4" and "Qwen3-VL-8B was
  later integrated for the official FETV and PSI-VQA runs. The earlier TAR baseline
  retained Qwen2.5-VL-7B for environment stability." Read literally that is a
  statement that TAR used one model and FETV/PSI used another — the exact violation
  described to us. It was written on 2026-07-23 about the July exploratory baseline and
  was never updated when Qwen3-VL-8B became the TAR submission. It was wrong, it was
  ours, and it stood in the cited public repository throughout the verification window.
  It is corrected as of this appeal; the git history shows both the original and the
  correction.

* **The *submitted* manuscript misstated the TAR backbone** as Qwen2.5-VL-7B at 4-bit.

Neither is a concealment, and the timeline is public and timestamped:

* commit `a07a930`, **2026-08-03**, "Identify all three official artifacts, and correct
  the TAR backbone" — pushed to `origin/main` before results were released. It added
  `OFFICIAL_RESULTS.md` ("The TAR backbone was Qwen3-VL-8B, not Qwen2.5-VL-7B … this
  contradicts the submitted paper … must be corrected in the camera-ready") and
  `REPRODUCE.md` gap 2.
* The camera-ready carried that correction through.
* We missed `track3_anomaly/README.md` in that sweep. That omission is ours.

The real Qwen2.5-VL-7B artifact in our logs is a *General*-type exploratory TAR entry
(`test`, 0.3480) that was never among the scored submissions.

## What we are asking

1. Re-check the FETV determination against the **scored artifacts** and the
   camera-ready rather than the July baseline notes, given that the portal field and
   the challenge summary paper both already attribute Qwen3 to Team 277.
2. Tell us which document the determination rested on. If it was
   `track3_anomaly/README.md`, we accept that the record we handed over said what it
   said, and we would ask only that the correction and its 2026-08-03 predecessor be
   noted.

We are not contesting the ranking on merit. FETV was 0.4634 against the winner's
0.4891.
