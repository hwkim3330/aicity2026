# Where the frozen system actually stands (measured 2026-08-20)

I twice misread our position today before getting it right. Recording the correct
numbers and what explains them.

## Two corrections to my own reading

**First error:** matching `'277'` against the JSON dump text hit an unrelated row, so
I reported us at **rank 15, mean 0.6544, 0.024 behind first**. That row was Star
Gazer. We are **rank 55 of 76, mean 0.4256** — which is exactly our TAR submission
score, because the Track 3 general `mean` *is* the TAR nine-component mean.

**Second error:** seeing 0.7803 and 0.3778 repeat, I claimed
`temporal_localization_miou` was bimodal and probably a parse-failure default. It is
not — 41 distinct values across 76 teams. What is true is that **0.7803 appears for 13
separate teams**, which does suggest a shared method or prior circulating.

## The real gap

| component | ours | rank 1 | gap |
| --- | ---: | ---: | ---: |
| temporal_localization_miou | 0.1990 | 0.7803 | +0.581 |
| bcq_accuracy | 0.5437 | **1.0000** | +0.456 |
| mcq_accuracy | 0.5875 | 0.9500 | +0.363 |
| temporal_description | 0.1856 | 0.5137 | +0.328 |
| causal_linkage | 0.2874 | 0.5310 | +0.244 |
| **mean** | **0.4256** | **0.6788** | **+0.253** |

**Every component is worse, most by a lot.** Rank 1 reaches perfect BCQ accuracy.
This is not a prompt-tuning gap.

## What explains it

`NVIDIAAICITYCHALLENGE/2026AICITY_Code_From_Top_Teams` lists Track 3 entrants
including **Korea Drive — Team 277**, us. One of them, UWIPL_ETRI (Team 30), has
published their full method:

> *LoRA-finetune a video VLM (Cosmos-Reason1-7B / Qwen3-VL) on the official
> `tao-vl-reason` annotations with ms-swift, then run sharded inference and per-task
> answer fusion.*

Their reported TAR is **~0.6144** against our 0.4256, and their TAR specialists are
LoRA adapters on **`Qwen/Qwen3-VL-8B-Instruct` — our exact base model**, split into
generalist, temporal, and MCQ experts. Their PSI (66.78) and FETV (0.4908) are both
rank 1.

So the difference is not the backbone and not the prompts. **It is that the top of
this board fine-tunes and we deliberately did not.**

## What this means for the paper and poster

It does not weaken the frozen-backbone framing — it *locates* it. The paper's claim
is what a fixed checkpoint achieves when only the inference contract changes, and the
honest coordinate for that claim is now public: same base model, LoRA-tuned, +0.19 on
TAR. Stating it is stronger than omitting it, and a workshop audience will already
know the top entries fine-tune.

The poster currently says "No adapter, fine-tuning, or task-specific checkpoint was
used" without that context. One line placing us against the tuned systems would make
the contribution legible rather than looking like an unexplained low score.

## The one thing directly transferable

UWIPL_ETRI single out, as their own most transferable finding:

> *dense-frame inference for PSI — raising **inference** frame rate lifts
> perception-question accuracy **with the same weights**.*

No training, same checkpoint, test-time only. That is compatible with our frozen
design and the evaluation window is open. It is the one idea from the top of the
board we can legitimately adopt as-is.

**Line to hold:** their adapters are on the Hub, and running them would not be our
work. Learning the method is what the organizers publish these repositories for.

## Dense-frame: applicable, and how to test it without repeating today's mistake

`scripts/inference.py` reads `MAX_FRAMES = int(os.environ.get("TAR_MAX_FRAMES", 16))`,
so the frame budget is already an environment variable. No training, no code change,
same checkpoint — exactly the condition UWIPL_ETRI describe.

**Three constraints on doing it properly.**

The camera-ready fixes 16 frames in three places (poster header, Table 1,
`CAMERA_READY.md`). Any run at a different budget is a **different configuration**
and must be reported as post-deadline, not folded into the paper result — which is
what slide 9's own rule of thumb says.

Today's v8 lesson applies directly: a 24-item paired win did not predict the
leaderboard. So dense-frame gets **scored locally first** against the 227 labelled
PSI clips in `psi_mcq_cv_results/`, and only goes to the portal if that moves. Not
the other way round.

And the GPU is committed to the CASCADE description run for roughly two more hours.

**Order:** finish CASCADE → rerun PSI at `TAR_MAX_FRAMES=32` → score on the 227
labelled clips → submit only if it improves → keep "16 frames" in the paper either
way.
