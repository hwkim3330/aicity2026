# Reproduction record

Everything needed to re-run the three Track 3 evaluations Korea Drive entered,
plus an explicit account of what could not be recovered.

**No model training or fine-tuning is required.** All official runs used frozen
pretrained backbones downloaded from the Hugging Face Hub. The only fitted
component in the entire system is the PSI-VQA temporal prior, whose parameters
were derived exclusively from the released PSI training annotations; its search
space, objective, and per-split refit are published in
[`track3_anomaly/analysis/temporal_prior_protocol.md`](track3_anomaly/analysis/temporal_prior_protocol.md).
A LoRA fine-tuning experiment was run during development and **abandoned
because it scored worse than the frozen baseline**; it is documented in
[`ABLATIONS.md`](ABLATIONS.md) and contributed to no submission.

---

## Common environment

| | |
|---|---|
| OS | Linux 6.8 (Ubuntu 24.04), Python 3.12.3 |
| GPU | 1 × NVIDIA RTX 3090, 24 GB, driver 580.173.02 |
| CUDA build | PyTorch 2.10.0+cu128 |
| Transformers | 5.13.0 (inference) |
| Video decode | decord 0.6.0 via `qwen_vl_utils`, `ffprobe` 6.1.1 for durations |
| Sentence embeddings | sentence-transformers 5.6.0 (`all-MiniLM-L6-v2`), analysis only |
| NumPy | 2.5.1 at the time of this record |

```bash
pip install -r requirements-inference.txt
```

The organizer's TAR scorer needs a **different and incompatible** environment —
`evaluate.py` pins `transformers==4.57.0` because BERTScore shifts by up to
~0.02 absolute under transformers 5.x. Install it separately:

```bash
pip install -r requirements-tar-evaluator.txt
```

Environment variables read by the inference backend:

| Variable | Default | Meaning |
|---|---|---|
| `TAR_MODEL_ID` | `Qwen/Qwen3-VL-8B-Instruct` | Hub model id |
| `TAR_HF_CACHE` | `track3_anomaly/hf_cache` | weight cache directory |
| `TAR_MAX_FRAMES` | `16` | maximum sampled frames per clip |
| `TAR_MAX_PIXELS` | `151200` | pixel budget per frame (360 × 420) |

---

## Track 3 — TAR reproduction

| | |
|---|---|
| Model ID | `Qwen/Qwen3-VL-8B-Instruct` — the portal records `models_used: qwen3` for the scored submission |
| Hub revision | **not recorded in the original runs** |
| Precision | bf16 |
| Official artifact | `track3_anomaly/submissions/submission_qwen3vl8b_v9.csv` (portal submission `9`, 2026-07-11 16:14) |
| Expected SHA256 | `243a5e8b67310428096cfc760ddeedaf5bc9d280729ad73f4c940eb3da759f6f` |
| Official result | rank 24 of 27 (public board), mean 0.4256 |
| Frame sampling | 16 max / 4 min, uniform, with `MM:SS` question-window extraction |
| Pixel budget | 151,200 per frame |
| Prompt/config | `track3_anomaly/scripts/prompts.py` |
| Dataset | `track3_anomaly/data/test/test.json` (960 items, 80 clips) + clips fetched by `data/test/download_test_videos.py` |
| Command | `./scripts/reproduce_tar_official.sh` |
| Output | `track3_anomaly/submissions/reproduced_tar.csv` |
| Records | 960 |
| Validator | `python3 test/evaluate.py --gt test/test.json --submission <csv>` |
| GPU / runtime / peak VRAM | see [`BENCHMARKS.md`](BENCHMARKS.md) |

The earliest candidate `submission_qwen25vl_4bit.csv` used
`Qwen/Qwen2.5-VL-7B-Instruct` at NF4 4-bit. It was submitted once as a General
entry (0.3480) and was **not** the scored submission. Reproduce it with:

```bash
TAR_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct TAR_QUANT=4bit ./scripts/reproduce_tar_official.sh
```

One test clip (`v=ir8j5bGBTiE_0-32_0-40.mp4`) is permanently age-gated on
YouTube and cannot be fetched; its items receive type-correct fallbacks so
coverage stays at 960/960.

---

## Track 7 — FETV reproduction

| | |
|---|---|
| Model ID | `Qwen/Qwen3-VL-8B-Instruct` |
| Hub revision | **not recorded in the original run** (do not infer a hash) |
| Precision | bf16 |
| Frame sampling | 16 max / 4 min |
| Pixel budget | 151,200 per frame (360 × 420) |
| Decoding | greedy (`do_sample=False`), one structured JSON call per clip, few-shot enabled |
| Prompt/config | `track3_anomaly/scripts/fetv_submission.py::PROMPT`, budgets in `prompts.py` |
| Dataset | FETV public clips, distributed through the challenge portal (not redistributable here) |
| Command | `FETV_CLIPS=/path/to/FETV_public_clips ./scripts/reproduce_fetv_official.sh` |
| Output | `track3_anomaly/submissions/reproduced_fetv_v11.json` |
| Records | 200 |
| Official artifact | `track3_anomaly/submissions/fetv_submission_v11.json` |
| Expected SHA256 | `39abdb0a8cca7a7fa18dbd31374ee353e032977df9928d54734a53e9ec43e835` |
| Official result | rank 3, final 0.4634 (description 0.4238, categorical mean 0.5031) |

Seeds were not set by the official script; greedy decoding makes the run
deterministic up to kernel nondeterminism, but byte-level reproduction has not
been verified because the Hub revision was not preserved. New runs can pin both
— see [Determinism controls](#determinism-controls) — though that fixes future
runs rather than recovering this one.

---

## Track 8 — PSI-VQA reproduction

| | |
|---|---|
| Model ID | `Qwen/Qwen3-VL-8B-Instruct` |
| Hub revision | **not recorded in the original run** |
| Precision | bf16 |
| Frame sampling | 16 max / 4 min, real-fps metadata via decord (`return_video_metadata=True`) |
| Pixel budget | 151,200 per frame |
| Decoding | BCQ: 5-sample self-consistency vote; MCQ / OpenQA / temporal: greedy |
| Prompt/config | `prompts.py` (`psi_bcq`, `psi_mcq`), routing in `build_psi_test_json.py` |
| Dataset | `ise-ice-lab/PSI_VQA` on the Hub — `test_public/*.json` + `test_public/videos/` |
| Command | `./scripts/reproduce_psi_vqa_official.sh` |
| Output | `track3_anomaly/submissions/reproduced_psi_vqa_v7.csv` |
| Records | 328 (55 BCQ, 91 MCQ, 126 OpenQA, 56 temporal) |
| Official artifact | `track3_anomaly/submissions/psi_vqa_submission_v7.csv` |
| Expected SHA256 | `a3829a36f591907bb8838098b1cc61feb907fec1cc6215f6098094aafaafb110` |
| Official result | rank 5, final 57.0400 (BCQ 0.5045, OpenQA 0.6019, MCQ 0.6044, temporal mIoU 0.5708) |

Fetch the dataset with:

```bash
hf download ise-ice-lab/PSI_VQA --repo-type dataset \
  --local-dir track3_anomaly/data/psi_vqa \
  --include "test_public/*" --include "train/*"
```

### What v7 actually is

The shipped v7 differs from v6 in **exactly the 126 OpenQA rows** and nothing
else. The BCQ vote-margin rebalance described in
`scripts/make_psi_v7_bcq_rebalance.py` was written but **did not ship** — all
55 BCQ rows in v7 are byte-identical to v6, as are the 91 MCQ and 56 temporal
rows. That script is retained for the record; it does not describe the
submitted artifact.

The script that produced v7's OpenQA rows was never committed. It has been
reconstructed as `scripts/make_psi_v7_openqa_prior.py` and verified to
regenerate the shipped file exactly:

```bash
cd track3_anomaly/scripts && python3 make_psi_v7_openqa_prior.py --verify
# OK: reconstruction matches ../submissions/psi_vqa_submission_v7.csv on all 328 records
```

### Determinism

Steps 4 (temporal prior) and 5 (OpenQA cues) are fully deterministic. MCQ,
OpenQA, and temporal generation use greedy decoding. The 55 BCQ rows use
unseeded 5-sample voting and are **not** expected to reproduce bit-for-bit, so
a reproduced file will not match the recorded SHA256.

---

## What the official runs pinned, and what was recovered afterwards

The 2026-07-10 submissions that produced the final ranks were run with no seed
and no weight revision. One of those is recoverable and the other is not.

| Value | Status | Evidence |
|---|---|---|
| Weight revision | **Recovered** — `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | see below |
| Sampling seeds | **Unrecoverable** | never generated; no RNG was seeded, so no value exists to record |
| Model id, precision, frames, pixel budget | Already recorded | [Common environment](#common-environment) |
| Library and driver versions | Already recorded | [Common environment](#common-environment) |

### Why the revision is certain

The scripts pinned no `revision=`, so every load resolved whatever `main`
pointed at on the run date. The Hub repo `Qwen/Qwen3-VL-8B-Instruct` reports
`lastModified: 2025-10-15` and its newest commit is `0c351dd` of the same day —
there has been no commit since, so `main` on 2026-07-10 can only have been
`0c351dd`. The local cache agrees: it holds exactly one snapshot, `0c351dd`,
whose config blobs were fetched 2026-01-31, i.e. already after that commit.

```bash
curl -s https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sha'], d['lastModified'])"
# 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b 2025-10-15T16:16:59.000Z
```

It is now pinned in both backends as `MODEL_REVISION`, overridable with
`$TAR_MODEL_REVISION` / `$T2_MODEL_REVISION`. The pin applies only while
`MODEL_ID` is the default, so pointing the scripts at another checkpoint does
not attach this hash to it.

### How much of each result the missing seeds actually touch

Sampled voting is not spread across the system. Of the thirteen task types in
[`track3_anomaly/scripts/prompts.py`](track3_anomaly/scripts/prompts.py) exactly
**two** carry `self_consistency: True` — `bcq` and `psi_bcq`. The other eleven,
including `psi_mcq`, decode greedily and reproduce.

| Submission | Deterministic | Sampled |
|---|---|---|
| Track 7 FETV, 0.4634 | all of it | — |
| Track 8 PSI, 57.04 | `mcq_norm`, `open_qa_norm`, `temporal_norm` | `bcq_norm` (55 rows) |
| Track 3 TAR, 0.4256 | 8 of the 9 scored components | `bcq_accuracy` |

The ranks do not depend on the sampled part:

- **Track 8** is rank 5 on the public board at 57.04, with 54.2445 below and
  64.4161 above. `final` is the unweighted mean of the four `*_norm` values, so
  losing a place needs `bcq_norm` to fall 11.18 points. One of the 55 rows is
  worth 1.82 points, so **six or more rows would have to flip**. Gaining a place
  needs +29.50, i.e. BCQ macro-F1 from 0.5045 to about 0.80.
- **Track 3** is rank 24 at 0.4256, with 0.3575 below and 0.4867 above. `mean`
  averages nine components, so losing a place needs `bcq_accuracy` to fall
  0.6130 from 0.5437 — **below zero**, which is unreachable. Gaining a place
  needs 1.09, above the metric's ceiling. This rank is structurally immune.

So the unrecoverable seeds cost bit-exactness on two components and change no
placement.

### Why the seeds are not recoverable

Nothing seeded `random`, `numpy`, or `torch`, so the sampled draws consumed
whatever state the interpreter started with — a value that was never
materialised, never logged, and is not derivable from the outputs. The
five-sample BCQ/MCQ votes therefore cannot be replayed exactly, and no amount of
searching the artifacts will change that. This is a permanent property of those
submissions; the controls below fix subsequent runs only.

---

## Determinism controls

The official runs of 2026-07-10 were produced with no seeding and no cuDNN
pinning at all. That is a property of those runs and **cannot be repaired after
the fact**: the seeds were never recorded, so the shipped artifacts stay as
described above — greedy paths reproducible up to kernel nondeterminism, sampled
voting paths not reproducible. Nothing below changes the recorded SHA256s.

What it does give is a reviewer re-running the pipeline twice getting identical
bytes, so any difference they observe is a real difference and not sampling
noise.

[`shared/scripts/determinism.py`](shared/scripts/determinism.py) is wired into
both inference backends
([`track3_anomaly/scripts/inference.py`](track3_anomaly/scripts/inference.py),
[`track2_captioning/scripts/inference.py`](track2_captioning/scripts/inference.py)):

| | |
|---|---|
| Seed | `--seed N`, else `$AICITY_SEED`, else 1234 |
| Seeded | `random`, `PYTHONHASHSEED`, `numpy`, `torch`, `torch.cuda` (all devices) |
| cuDNN | `benchmark=False`, `deterministic=True`, applied before the model loads so the autotuner never runs |
| Strict | `--strict-determinism` adds `use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

Two details worth knowing before relying on it:

- **Sampled draws are seeded per question, not per process.** The seed for draw
  *k* of a clip is `blake2b(seed, basename, task_type, question) + k`. Seeding
  once at startup would make every answer depend on how many samples happened to
  be drawn before it, so a resumed or reordered run would diverge; this way it
  does not. blake2b rather than `hash()` because `PYTHONHASHSEED` randomises
  string hashing per process — measured: `hash()` gave three different values
  across three runs where blake2b gave one.
- **`--strict-determinism` is off by default on purpose.** It changes which
  kernels run, and therefore the numbers, so defaulting it on would silently
  diverge from what the repository's recorded results were produced with. It is
  also slower and raises on any op lacking a deterministic implementation.

Verify the plumbing without loading a model:

```bash
python3 -c "
import sys; sys.path.insert(0, 'shared/scripts')
import determinism, torch
determinism.pin(1234); a = torch.randn(4, device='cuda')
determinism.pin(1234, verbose=False); b = torch.randn(4, device='cuda')
assert torch.equal(a, b) and torch.backends.cudnn.deterministic
print('determinism OK')"
```

---

## Validation commands

```bash
# offline: hashes, record counts, structure, and the v6 -> v7 reconstruction
./scripts/validate_official_artifacts.sh

# TAR format/coverage against the redacted test set (needs the evaluator env)
cd track3_anomaly/data && python3 test/evaluate.py \
  --gt test/test.json --submission ../submissions/<candidate>.csv

# temporal-prior cross-validation (CPU only, ~25 s)
cd track3_anomaly/analysis && python3 temporal_prior_baselines.py

# controlled PSI MCQ prompt ablation (reads stored per-condition JSONL)
cd track3_anomaly/analysis && python3 prompt_ablation_psi_mcq.py

# FETV structured-prototype unit tests
python3 -m unittest discover -s track3_anomaly/tests -v
```

## Expected artifact hashes

| Artifact | Records | SHA256 |
|---|---:|---|
| `track3_anomaly/submissions/fetv_submission_v11.json` | 200 | `39abdb0a8cca7a7fa18dbd31374ee353e032977df9928d54734a53e9ec43e835` |
| `track3_anomaly/submissions/psi_vqa_submission_v7.csv` | 328 | `a3829a36f591907bb8838098b1cc61feb907fec1cc6215f6098094aafaafb110` |

Verified by `./scripts/validate_official_artifacts.sh`.

## Runtime and peak VRAM

Measured figures are in [`BENCHMARKS.md`](BENCHMARKS.md). Original official-run
wall-clock times were **not recorded**; the benchmark table reports
reproduction-run measurements and labels them as such.

---

## Known provenance gaps

These are recorded rather than filled with plausible-looking values.

1. ~~The scored TAR artifact is unidentified.~~ **Resolved 2026-08-03.** The
   portal's Track 3 submission page identifies submission `9`
   (2026-07-11 16:14, mean 0.4256) whose ten component scores match this
   repository's recorded TAR results exactly. By mtime it is
   `submission_qwen3vl8b_v9.csv`. All eight TAR submissions map 1:1 to
   candidate files; `submission_qwen3vl8b_v8.csv` was never submitted. See
   [`leaderboards/submission_history.json`](leaderboards/submission_history.json).
2. **The submitted paper misstates the TAR backbone.** Its abstract and Table 1
   say TAR used Qwen2.5-VL-7B with 4-bit inference. The portal records
   `models_used: qwen3` for the scored submission, and the only Qwen2.5-VL
   4-bit entry (`test`, General, 0.3480) was never scored. All three official
   runs used Qwen3-VL-8B-Instruct in bf16. **This must be corrected in the
   camera-ready**, and it removes the backbone/precision confound the paper
   cited as its reason for declining a controlled cross-domain claim.
3. **No Hub revision/commit was persisted for any official run.** The model ids
   are recorded; the exact weight revisions are not, and are not guessed.
4. **The PSI portal upload filename/ID was not retained.**
   `psi_vqa_submission_v7.csv` is the repository-side candidate associated with
   the final 57.0400 result. This is a repository-history association, not a
   portal-side confirmation.
5. **Official-run wall-clock and peak memory were not logged.** Only
   reproduction-run measurements exist.
6. ~~Leaderboard denominators were not archived.~~ **Resolved 2026-08-03.**
   Exported from the portal's public endpoint: TAR 24 of 27, FETV 3 of 8,
   PSI-VQA 5 of 7 on the public board. See
   [`leaderboards/README.md`](leaderboards/README.md) and refresh with
   `python3 scripts/fetch_leaderboards.py`.
7. ~~Team names predate the organizers' update.~~ **Resolved 2026-08-03** — the
   stored export postdates the 2026-08-02 notice.
8. ~~The PSI portal upload filename was not retained.~~ **Resolved 2026-08-03.**
   The Track 8 submission history shows submission `7` scoring 57.0400 at
   2026-07-11 16:15, and its component scores (BCQ 0.5045, Cue-F1 0.6019,
   MCQ 0.6044, mIoU 0.5708) match `psi_vqa_submission_v7.csv` exactly. Gap 4
   above is superseded.
9. **Some scores recorded in this repository during the challenge were wrong.**
   The portal history contradicts development notes in several places — most
   importantly, FETV v8 scored 0.4616 and did *not* regress against v7's 0.4584,
   though a note in this repository claimed 0.4505 versus 0.4621. Every score
   now cited in [`ABLATIONS.md`](ABLATIONS.md) comes from the portal export.
   Treat any score not traceable to the portal as unverified.

## Research-only artifacts

`track3_anomaly/submissions/psi_vqa_submission_v8_final.csv` is a
**post-deadline research artifact — not used for the official leaderboard
result**. `track3_anomaly/scripts/fetv_structured_pipeline.py` is a
**post-challenge prototype — not used for the official leaderboard result**.
Neither may be presented as an official submission.
