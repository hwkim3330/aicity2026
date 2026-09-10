# Track 7 (FETV) removal — the questions, and what the repository can prove

Korea Drive, Team 277. Written 2026-09-09, after the workshop announcement that our
FETV entry was removed. This file exists so the questions being asked internally have
one answer each, with something checkable behind it rather than a recollection.

The organizers' Challenge Winners page lists only MR-CAS and UWIPL_ETRI for FETV, with
a **Verified Score** column separate from the public score, and both teams moved in it
(UWIPL_ETRI 0.4891 → 0.4629, MR-CAS 0.4884 → 0.4848, reversing the award order). So
the final review re-runs a team's published code and scores the output.

---

## Q. Did all three scored runs use the same model?

**Yes.** `Qwen/Qwen3-VL-8B-Instruct`, bf16, no adapter, no task-specific weights.

| evidence | what it says |
|---|---|
| evaluation portal | `models_used: qwen3` on the scored TAR submission |
| challenge summary paper (arXiv 2608.17044) | attributes **Qwen3** to Team 277 in the Track 3 table |
| camera-ready abstract | "frozen Qwen3-VL-8B-Instruct inference pipeline … no task-specific parameter update or adapter generated the official predictions" |
| this repository | every reproduce script hardcodes or defaults to `Qwen/Qwen3-VL-8B-Instruct`; no adapter weights are tracked |
| PSI-VQA board | Team 277 still listed at rank 5, which a unified-system finding would not leave standing |
| portal `models_used` | present for 27 of 27 teams on the TAR board and for **0 of 8 on FETV, 0 of 7 on PSI-VQA** — those forms had no such field, so the empty FETV declaration is universal and says nothing about any team |

## Q. Was LoRA used? The paper mentions fine-tuning.

**Trained once, measured, dropped, never submitted.**

* Trained 2026-07-08 18:42 — 150 steps, 2 epochs, final train loss 1.045, ~21 minutes
  (`track3_anomaly/lora_train_run.log`).
* Evaluated on the local proxy sample: BCQ 0.550, MCQ 0.675, **overall 0.636**
  (`track3_anomaly/lora_eval_run.log`).
* The few-shot base model on a comparable proxy scored **0.772**
  (`track3_anomaly/eval_fewshot_sc5.log`). The adapter was worse, so it was abandoned.
* The checkpoint `lora_out_v1` no longer exists on disk. `eval_lora.py` and
  `build_finetune_data.py` remain as the record of the attempt.
* No submission on any track was produced with it.

## Q. Was an A100 used?

**No.** One RTX 3090, 24 GB, driver 580.173.02 (`REPRODUCE.md`, common environment).
No A100 appears in the paper or in any log in this repository.

## Q. Where was Qwen2.5-VL written, and was it true?

It was written in the **submitted manuscript** — abstract and Table 1 — stating the TAR
run used Qwen2.5-VL-7B at 4-bit. It was **not true of any scored submission**. The only
Qwen2.5-VL 4-bit entry is a General-type exploratory TAR submission (`test`, 0.3480,
2026-07-03) that was never scored.

We found this ourselves and published the correction in commit `a07a930` on
**2026-08-03**, before results, in `OFFICIAL_RESULTS.md` and `REPRODUCE.md` gap 2, and
carried it into the camera-ready — whose final PDF contains zero occurrences of
"Qwen2.5" and zero of "4-bit". A track-level README repeated the error and was missed
in that sweep; corrected 2026-09-09 in `9bc25e8`.

## Q. Does "task-routed" mean we tuned per leaderboard, and is that the violation?

This is the sharpest version of the worry, and the evidence points away from it.

What is actually per-leaderboard in our pipeline: task-specific prompts, frame policies,
output contracts and parsers — which the FAQ explicitly allows — plus two statistical
priors on PSI-VQA. Both priors are **fitted on the training split, not the test set**:
the Open-QA cue pair by greedy forward selection on the PSI training split
(`make_psi_v7_openqa_prior.py`), and the temporal window on all 227 training intervals
(`apply_psi_temporal_prior.py`). The FAQ prohibits "using test data during training or
for model selection"; training data is not that.

And the outcome runs the wrong way for this theory: **PSI-VQA carries the most
calibration of our three entries and kept its rank 5 listing, while FETV carries the
least and was removed.** If per-leaderboard calibration were the finding, the sanction
would have landed on PSI first.

## Q. Which submissions failed?

* **Track 6** — two portal evaluations returned `Failed` before the deadline, so it has
  no scored leaderboard row at all. This is the "failed track".
* **Track 7** — one General-type submission, `teat`, 2026-07-05 19:36, status `Failed`.
  Seven later FETV submissions scored normally.
* Track 3 (TAR) and Track 8 (PSI-VQA) had no failed submissions.

## Q. So what does the evidence actually support?

Not model identity. The FETV artifact has a reproducibility problem, and it is specific.

**What now works, and what that is worth.** `rebuild_fetv_v11.py --verify` regenerates
the scored submission with 200/200 rows identical, hashing to
`39abdb0a8cca7a7fa18dbd31374ee353e032977df9928d54734a53e9ec43e835`, the same SHA256 as
the submitted file. But it starts from **v7**, an archived intermediate, and two of the
steps upstream of v7 do not reproduce. Calling that "a byte-exact reproduction of our
submission" would be false, and an earlier version of this file came close to doing so.

The real model run is **v6**, not v7. Every model run left a log — v1 through v6, and
v8 — and there is no v7 log; v6 → v7 changed the description on all 200 rows and the
timestamp on 163, and **no categorical field at all**, which a re-inference could not do.

**What still does not reproduce.** Three points, and they are why the removal is
defensible:

1. **The model run (v6) does not regenerate.** Re-running the published inference
   matches no archived version exactly — 82 of 200 rows on the categorical fields alone.
   Eight explanations are ruled out by measurement in `REPRODUCE.md`: nondeterminism, the
   determinism pin, clip encoding, frame sampling, few-shot exemplars, the NumPy upgrade,
   the model weights (identical cache blob hashes), the committed prompt, and the video
   decoder (v4-v6 and v8 all logged decord, as our re-runs do; only v1-v3 used
   torchvision). What remains is that these runs happened between the commit of
   2026-07-06 and the archive commit of 2026-07-11 22:51, so the code that produced them
   was an uncommitted working tree that no longer exists.

2. **The v6 → v7 timestamp correction does not regenerate.** It rewrote 163 of 200
   timestamps. Those values appear in no earlier artifact, and the clock burned into the
   video puts them four to six seconds after each clip's first frame at no fixed offset —
   an uncommitted script read the on-screen clock. The description rewrite in the same
   step *is* recovered: `"On {date} at {time}, " + lowercased v6 description` matches 199
   of 200 rows, the exception being the one clip whose v6 date and time were placeholder
   garbage.

3. **Fifteen rows are a lookup table.** One step writes a violator type and colour into
   15 `no_violation` rows. Fourteen of the fifteen values are our own model's output in
   the first two submissions; `001_013.mp4`'s colour appears in no archived artifact;
   and the choice of those 15 clips out of 64 equally qualified ones follows no rule
   that survives testing — edited and unedited candidates are statistically
   indistinguishable (mean cross-version agreement 4.00 vs 3.63, mean distinct answers
   1.53 vs 1.55). Something outside the pipeline selected them.

By contrast the PSI-VQA artifact reproduces exactly, verified on all 328 records
(`make_psi_v7_openqa_prior.py --verify`), and PSI-VQA kept its rank.

**A caution about the verbal reason.** The reason given on site was "a different
model", and no reproducibility concern was mentioned. Everything above is our own
reading of the evidence, not a statement of what the organizers found. The first thing
to ask them is which finding the determination actually rests on — that question, and
one about the standard applied to award candidates, are the whole content of the draft
in [`EMAIL_TO_ORGANIZERS.md`](EMAIL_TO_ORGANIZERS.md).

## What is not worth doing

`postchallenge_analysis/fetv_submission_v12.json` scores 0.4724 against the official 0.4634 by setting
`answer_intersection_type` from ground truth recovered by inverting our own leaderboard
score. That is test-set annotation recovery and the FAQ prohibits it; it exists in this
repository as a measurement, not as a submission. The same insight taken honestly —
junction names burned into the video plus a per-source majority vote — reaches 0.4697
and would have been legitimate. Neither changes a closed board.
