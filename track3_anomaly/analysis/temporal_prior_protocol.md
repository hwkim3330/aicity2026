# PSI-VQA temporal-localization prior: fitting and validation protocol

This document specifies exactly how the duration-stratified temporal prior was
fitted, what it was compared against, and which reported numbers are held out.
It exists because the challenge paper reported a cross-validated mIoU without
publishing the search space, objective, or split construction, and because the
shipped constants were fitted once on the full training set rather than
per-split.

Everything below is reproduced by:

```bash
cd track3_anomaly/analysis
python3 temporal_prior_baselines.py \
  --gt      ../data/psi_vqa/train/temporal_localization.json \
  --videos  ../data/psi_vqa/train/videos/temporal \
  --out-csv  ../results/temporal_prior_cv.csv \
  --out-json ../results/temporal_prior_summary.json
```

Runtime is about 25 s on CPU after clip durations are cached; no GPU is needed
for any row except `vlm` and `vlm_prior_blend`.

## Data

- Source: `ise-ice-lab/PSI_VQA`, `train/temporal_localization.json`
  (227 items) and `train/videos/temporal/*.mp4` (227 clips).
- One item per clip, so the 227 items span 227 distinct videos.
- Ground truth is `{"start": "MM:SS.ss", "end": "MM:SS.ss"}` with
  hundredth-of-a-second resolution; both endpoints are parsed to float seconds.
- Clip duration comes from `ffprobe -show_entries format=duration` and is
  cached in `results/psi_temporal_durations.json`.

## Splits

- 5 folds, **video-grouped**: no video contributes to both the training and the
  held-out half of a split. PSI temporal happens to be one item per video, but
  the grouping is applied explicitly so the protocol stays correct if that
  changes.
- Fold assignment: `numpy.random.RandomState(42).permutation(n_videos)`,
  then `order[k::5]` for fold `k`. Seed is `--seed`, default 42.
- Resulting sizes: 181/46, 181/46, 182/45, 182/45, 182/45 (train/held-out).

## Objective and search space

Every parameterized baseline maximizes **mean IoU on that split's training half
only**, then is evaluated once on the held-out half. No held-out item
participates in any fit.

| Baseline | Fitted parameters | Search space |
|---|---|---|
| `random` | none | interval endpoints drawn `U(0, duration)` and sorted, averaged over 20 draws per split |
| `center` | window width ratio `w`; window is `[(1-w)/2·dur, (1+w)/2·dur]` | `w ∈ [0.05, 1.00]`, step 0.01 |
| `global_mean` | `lo = mean(start/dur)`, `hi = mean(end/dur)` over training items | closed form, no grid |
| `duration_prior` | threshold `T`, plus `(lo, hi)` for each of the two strata | `T ∈ {6,7,…,14} s`; `lo, hi ∈ [0.00, 1.00]` step 0.02 with `hi > lo`; all three fitted jointly by exhaustive search |
| `vlm` | none | Qwen3-VL-8B, bf16, 16 frames, 151200 px/frame, greedy — the official temporal configuration |
| `vlm_prior_blend` | none beyond `duration_prior` | endpoint-wise average of `vlm` and `duration_prior` |
| `shipped_prior_full_fit` | none — replays shipped constants | `T = 10 s`, short `(0.30, 0.70)`, long `(0.20, 0.62)`, rounded to whole seconds, `end > start` forced |

`shipped_prior_full_fit` is the rule that actually shipped in
`psi_vqa_submission_v6.csv` and `v7.csv`. Its constants were selected once on
all 227 training windows, so its per-split scores are **in-sample and not held
out**. It is reported anyway, and labelled as such, to quantify how much that
full-set fit flatters itself.

## Results

From `results/temporal_prior_cv.csv` (mIoU on held-out halves, 5 splits):

| split | train | test | random | vlm | center | global_mean | vlm_prior_blend | duration_prior | shipped_prior_full_fit* |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 181 | 46 | 0.2597 | 0.4765 | 0.4991 | 0.5128 | 0.5420 | 0.5440 | 0.5670 |
| 1 | 181 | 46 | 0.2989 | 0.4608 | 0.5558 | 0.5890 | 0.5876 | 0.5993 | 0.6171 |
| 2 | 182 | 45 | 0.2681 | 0.4652 | 0.4829 | 0.4835 | 0.5522 | 0.5154 | 0.5145 |
| 3 | 182 | 45 | 0.2788 | 0.4370 | 0.5417 | 0.5623 | 0.5275 | 0.5653 | 0.5904 |
| 4 | 182 | 45 | 0.2652 | 0.4691 | 0.5206 | 0.5370 | 0.5384 | 0.5591 | 0.5728 |
| **mean** | | | **0.2741** | **0.4617** | **0.5200** | **0.5369** | **0.5495** | **0.5566** | **0.5724*** |
| **std** | | | 0.0155 | 0.0150 | 0.0299 | 0.0412 | 0.0230 | 0.0307 | 0.0378 |

`*` not held out — see above.

The `vlm` row is Qwen3-VL-8B run on all 227 training clips in the official
temporal configuration (`analysis/psi_temporal_vlm_eval.py`, 227/227 completed,
0 errors, 2.07 s/item, 16.78 GiB peak VRAM). 225 of 227 outputs parsed as valid
`{"start", "end"}` JSON; the 2 unparseable ones are excluded rather than scored
as zero.

Fitted parameters per split:

| split | threshold | short stratum `(lo, hi)` | long stratum `(lo, hi)` | n short / long | center `w` | global mean `(lo, hi)` |
|---:|---:|---|---|---:|---:|---|
| 0 | 12 s | (0.30, 0.76) | (0.22, 0.60) | 33 / 148 | 0.47 | (0.268, 0.623) |
| 1 | 7 s | (0.32, 0.68) | (0.20, 0.64) | 29 / 152 | 0.47 | (0.265, 0.612) |
| 2 | 8 s | (0.32, 0.68) | (0.20, 0.60) | 29 / 153 | 0.47 | (0.261, 0.619) |
| 3 | 12 s | (0.28, 0.76) | (0.20, 0.58) | 34 / 148 | 0.49 | (0.257, 0.612) |
| 4 | 7 s | (0.32, 0.70) | (0.20, 0.60) | 30 / 152 | 0.48 | (0.261, 0.616) |

## What these numbers do and do not show

- **The properly held-out prior scores 0.5566 ± 0.0307**, not the 0.5724 that
  the shipped constants reach in-sample. The 0.0158 gap is the optimism from
  fitting on all 227 windows; the earlier 0.553 figure quoted in the paper is
  consistent with the honest held-out value, but its protocol was not stated.
- **Most of the gain is not from duration stratification.** A fixed centered
  window with a single fitted width already reaches 0.5200, and a global mean
  normalized interval reaches 0.5369. Duration stratification adds
  +0.0197 over `global_mean` and +0.0366 over `center`. Any metadata-only
  prior captures the bulk of the effect.
- **The stratum ratios are stable, the threshold is not.** Short-clip `lo`
  stays in 0.28–0.32 and long-clip `lo` in 0.20–0.22 across all five splits,
  but the best threshold moves between 7 s and 12 s because the short stratum
  holds only ~30 clips. The threshold should be read as weakly identified.
- **The VLM is not blind, but it loses to a one-parameter window.** At 0.4617
  the model is well clear of random (0.2741), so it does carry real temporal
  signal. It nonetheless falls below a fixed centered window whose single
  fitted parameter is a width ratio (0.5200), and below every other
  metadata-only prior. The honest statement is not "the model knows nothing
  about time" but "on this benchmark, the model's temporal output is worth less
  than the benchmark's own duration statistics."
- **Blending makes it worse.** Averaging the VLM's endpoints with the prior's
  gives 0.5495, below the prior alone at 0.5566. This reproduces, under a
  proper held-out protocol, the development-time observation that no blend beat
  the pure prior.
- **None of this is video understanding.** Every row except `vlm` and
  `vlm_prior_blend` predicts an interval from clip duration alone, without
  decoding a single frame. The official PSI temporal mIoU of 0.5708 is
  therefore substantially a measure of annotation regularity in this benchmark,
  not of temporal grounding, and should not be expected to transfer to a
  dataset with different clipping conventions.

## Reproducing the VLM rows

The `vlm` and `vlm_prior_blend` rows require model predictions on the training
split, produced with the official temporal configuration:

```bash
cd track3_anomaly/analysis
python3 psi_temporal_vlm_eval.py \
  --out   ../results/psi_temporal_vlm_train_preds.jsonl \
  --stats ../results/psi_temporal_vlm_train_stats.json

python3 temporal_prior_baselines.py \
  --vlm-preds ../results/psi_temporal_vlm_train_preds.jsonl
```

Unparseable model outputs are counted as absent rather than silently scored as
zero; `vlm_n` in the summary JSON records how many held-out items the VLM row
is actually computed over.
