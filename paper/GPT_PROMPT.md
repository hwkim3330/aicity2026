# Prompt for revising the camera-ready — paste everything below the line

Written 2026-08-14 against commit `65b444d` of
<https://github.com/hwkim3330/aicity2026>. Every number here was recomputed from
the repository, not copied from a draft; the check that produced each one is
named so it can be rerun.

---

You are revising a camera-ready ECCV workshop paper. The deadline is
2026-08-15 (AoE). Do not restructure the paper. Make the edits listed below and
nothing else.

## Ground rules

- **Never introduce a number that is not in this prompt.** Every figure in the
  paper has been checked against the challenge portal export and the archived
  artifacts. If a revision needs a value that is not here, write `[TODO: verify]`
  instead of estimating one.
- **Do not soften or strengthen a claim beyond what the numbers support.** The
  paper's value is that its negative results are load-bearing.
- **Ranks always name the board.** "3rd of 8 on the public leaderboard", never
  "3rd". The portal keeps a public board and a general board with different
  denominators for identical scores.

## Verified — do not change these

99 numeric claims were recomputed and all matched. Leave them alone:

- Full FETV public board, 8 teams × 3 columns (24 values).
- Full PSI-VQA public board, 7 teams × 5 columns (35 values).
- All 10 TAR component scores; the 9 scored ones average 0.42561 and reconstruct
  the official 0.4256.
- All 12 FETV field scores in Figure 2 (weather/lighting/date 1.000, time 0.940,
  intersection 0.784, violator 0.313, colour 0.243, initial lane 0.178, final
  lane 0.169, violation type 0.158, initial position 0.124, final position
  0.128).
- Description 0.0067 above the winner; OpenQA 0.6019 above the winner's 0.5833;
  temporal mIoU 0.5708, within 0.0043 of fourth place.
- Temporal study: random 0.2741, VLM 0.4617, centered window 0.5200, global mean
  interval 0.5369, blend 0.5495, duration prior 0.5566, shipped constants 0.5724
  (not held out; the 0.0158 gap is fit optimism).
- PSI MCQ ablation: generic 8/24, routed 3/24, box-aware 9/24, McNemar
  p = 0.0625 and p = 0.0312.
- TAR General board: 13 organizer baselines, best 0.5729, ours 0.4256 beats 11.
- QLoRA 49/77 = 0.636.
- The footnote commit `65b444df` exists and is current.

**"Between two late system versions, 67/200 clips changed at least one
actor-dependent field; in 51 clips the violation type and all six dependent
actor fields changed together."** — confirmed, and the pair is **v8 → v11**,
both of which were submitted to the portal. Name them in the text; the sentence
currently makes the reader guess, and no adjacent pair of versions produces
those counts.

## Edit 1 — Figure 1 is the main problem

Two versions exist and both are wrong in opposite directions.

The figure currently in the Overleaf source **overflows its boxes**: "Required
output format" touches the border, "Frozen Qwen3-VL-8B" spills outside, and
"Contract + parser norm./calibration" overflows on both sides. The
"TAR / FETV / PSI-VQA" label floats under an arrow with nothing to attach to.

The newer version **mixes configuration with results**. Boxes carrying frame
policies and prompt programs also carry "0.4256 | Public 24/27", "Beats 11/13
organizer baselines", "Controlled A/B: 15/25 -> 16/25". Those are Tables 1–5
redrawn as a diagram, and they make three branches read as three systems rather
than one shared backbone.

Redraw so that **Figure 1 shows only what is configured, and tables carry what
was measured**:

- Panel (a): the benchmark interface — video + task → participant system →
  required output format → official scorer. One annotation, not a box: organizer
  baselines exist on the TAR General board (13 entries, best 0.5729) and on no
  other Track 3 board.
- Panel (b): one shared frozen backbone box —
  `Qwen3-VL-8B-Instruct, bf16, revision 0c351dd0, ≤16 frames, 151,200 px/frame` —
  drawn **once**, with the router above it and three branches that differ only
  in their control layer:

  | Branch | Frame policy | Prompt program | Output contract and post-processing |
  |---|---|---|---|
  | TAR | 16 frames, `MM:SS` question-window extraction | exact-choice with short reasoning; descriptive tasks get longer budgets | final-token extraction, vocabulary normalization; 5-sample majority vote on BCQ |
  | FETV | 16 frames, uniform | one call for the complete 13-field record, few-shot enabled | schema validation, field legalization, description template |
  | PSI-VQA | 16 frames, real-fps metadata | per-subtask programs, greedy | letter extraction with type-correct fallback; duration-stratified temporal prior $\hat I(d)$ |

- Mark the three knobs §6 actually varied — question-window sampling on TAR,
  red-box re-localization on PSI MCQ, and the temporal prior's threshold and
  endpoints — with a single consistent visual cue, and say in the caption that
  those are the varied components. Do not put their results in the figure.
- Every text string must fit inside its box. The current overflow is the
  specific defect a co-author raised.

## Edit 2 — the Hub revision is recovered, and the paper still hedges in one place

Sections 1, 4 and 9 have already been updated. **Section 7.3 has not**, and still
says:

> "The official artifacts record the same model identifier and precision, while
> the missing revision hashes prevent a stronger checkpoint-identity claim."

That is now false and it contradicts Section 4 of the same paper. Replace with:

> The official artifacts record the same model identifier, precision, and weight
> revision, so the three benchmarks were scored under one checkpoint.

Both revisions are known: `Qwen/Qwen3-VL-8B-Instruct` at
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` for all three official runs, and
`Qwen/Qwen2.5-VL-7B-Instruct` at `cc594898137f460bfe9f0759e9844b3ce807cfb5` for
the General-type TAR entry that was never the scored submission. The first was
recovered because the Hub repo has had no commit since 2025-10-15, so `main` on
the run date can only have been that; the second was logged at download time in
`track3_anomaly/model_download.log`. Neither is a guess.

## Edit 3 — what reproduction actually establishes

Section 9 says byte-level reproduction is limited by unseeded voting and by the
FETV generation history. Both true, and one measured result should join them,
because it is stronger than a disclaimer:

> Re-running the FETV pipeline on the public clips reproduces `v6_fewshot`, the
> single-pass artifact whose configuration matches, on 67.0% of fields; the
> shipped v11 is the last of an eleven-version chain and only one code state was
> committed for the nine artifacts in it. Two identical reruns agree bit-for-bit
> on all thirteen fields, so the residual difference is systematic rather than
> run-to-run variance.

Six candidate explanations for that residual were tested and ruled out —
nondeterminism, cuDNN pinning, clip encoding, frame sampling, the few-shot
exemplars, and the NumPy upgrade. Do not offer a cause; the repository does not
establish one.

## Edit 4 — smaller items

- Section 6.3 says the official OpenQA score "is lower than the local one-to-one
  macro estimate". Give the number: the local estimate was 0.7816 against the
  official 0.6019.
- Section 5.4 and Section 6.2 both introduce the red-box re-localization result.
  Keep it in 6.2 and have 5.4 point forward.
- Check that "Section 4" cross-references still resolve; the section numbering
  moved when Controlled Analyses was renamed.

## What not to do

- Do not remove TAR. A reviewer asked for the full ten-component breakdown, the
  title claims three benchmarks, and the General-board baseline comparison is
  now the paper's clearest quantitative reference point.
- Do not claim the FETV artifact is reproducible from the released code.
- Do not describe the timestamp residual as run-to-run variance.
- Do not report a score that is not traceable to
  `leaderboards/submission_history.json` or `leaderboards/raw/`.
