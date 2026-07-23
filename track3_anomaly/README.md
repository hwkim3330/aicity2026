# AI City Challenge 2026 — Track 3 (Anomalous Events in Transportation) baseline

Working baseline for the **TAR** leaderboard (in-domain). The official FETV
v11 artifact was generated with **Qwen/Qwen3-VL-8B-Instruct**, bf16, on a
single RTX 3090 (24GB). The same backend also supports 4-bit TAR experiments;
those settings are not the official FETV configuration.

## Directory layout

```
track3_anomaly/
├── data/                        # dataset (annotations + videos), see data/README.md
│   ├── train/*.json             # 10 task-type files, 44,040 items, 3,670 videos
│   ├── test/test.json           # 960 items, 80 videos, answers redacted
│   ├── test/evaluate.py         # organizer-provided validator/scorer
│   └── videos/                  # downloaded video files (not in git)
│       ├── tar_test/            # 80 test clips (yt-dlp from clip_manifest.csv)
│       ├── Vad-R1/, TAD/, ...   # training sources (background download, partial)
├── hf_cache/                    # local HF snapshot/cache (not in git)
├── scripts/
│   ├── prompts.py                # per-task-type prompt templates + gen configs
│   ├── inference.py               # QwenVLBackend: load model once, answer(video, task, question)
│   └── make_submission.py         # iterate test.json -> submissions/*.csv
├── submissions/                   # generated CSVs
└── README.md                      # this file
```

## Data schema (verified by reading the actual files)

Format: `tao-vl-reason-v1.0` envelope — `{"format", "metadata", "media_root", "items": [...]}`.

- **Training** (`data/train/<task>.json`, 10 files, 44,040 items / 3,670 videos):
  item = `{video_id, question, answer, reasoning}`. `question` already
  inlines task-specific formatting instructions (e.g. "Answer with only
  Yes or No."). `reasoning` is a chain-of-thought trace not used at
  inference time (could be used for future SFT/distillation).
- **Test** (`data/test/test.json`, 960 items / 80 videos, **10 task types**,
  answers redacted): item = `{item_index (16-hex id), task_type, video_id,
  question, answer: ""}`. `video_id` is `tar_test/v=<ytid>_<start>-<end>.mp4`.
- Task type counts in test.json: `bcq` 160, `bcq_openended` 160, `mcq` 80,
  `mcq_openended` 80, `open_qa` 80, `scene_description` 80,
  `video_summarization` 80, `causal_linkage` 80, `temporal_description` 80,
  `temporal_localization` 80 (per the 2026-07-01 notice, temporal_localization
  is excluded from scoring but must still be present/parseable in the CSV).
- **Submission**: CSV with exactly `item_index,prediction` for all 960 test
  items. `test/evaluate.py --gt test/test.json --submission <csv>` runs in
  validation-only mode against the redacted test set (format/coverage check
  only; real scoring happens server-side).
- Scoring (from `evaluate.py`): `bcq`/`mcq` → regex-extracted exact-match
  accuracy; `temporal_localization` → mean IoU (excluded per 07-01 notice);
  all other (open-ended) task types → BERTScore F1 (`roberta-large`,
  `rescale_with_baseline=True`, requires `transformers==4.57.0` — the
  released `evaluate.py` explicitly pins this because BERTScore under
  transformers 5.x shifts by up to ~0.02 absolute).

## Model choice

**Official FETV v11:** `Qwen/Qwen3-VL-8B-Instruct`, bf16, 16 frames, and
`360 * 420 = 151200` maximum pixels per frame. Decoding was greedy
(`do_sample=False`) with the task prompt in `scripts/fetv_submission.py`.
The original run did not persist a Hub revision/commit, so the revision is
explicitly recorded as **not available** in [`REPRODUCE.md`](../REPRODUCE.md).

**TAR baseline:** `Qwen/Qwen2.5-VL-7B-Instruct`, 4-bit NF4
(`bitsandbytes`, `bnb_4bit_use_double_quant=True`, compute dtype bf16).

Rationale (see also the researched comparison against Qwen3-VL-8B,
InternVL3-8B, VideoLLaMA3-7B):
- Native video input (dynamic FPS sampling, absolute-time-aware mRoPE) with
  explicit second-level timestamp grounding — directly useful for
  `temporal_localization`/`causal_linkage`/`temporal_description` prompts
  that reference `MM:SS` timestamps.
- Most mature `transformers`/`vLLM` integration of any open video-VLM;
  first-class `Qwen2_5_VLForConditionalGeneration` + `qwen_vl_utils`
  support, abundant community 4-bit/AWQ checkpoints.
- 7B model at 4-bit fits comfortably in the available VRAM headroom on the
  shared 3090 (other long-running processes on this box — a stock-trading
  bot and a CARLA simulation — hold ~9GB; the quantized model + KV cache
  fits in the remaining ~14GB, observed peak ~16GB total GPU usage during a
  smoke test).
- Qwen3-VL-8B would likely score higher on long-video/temporal benchmarks
  but has thinner ecosystem support; not worth the integration risk on an
  8-day budget. Noted as a drop-in future upgrade (same `transformers`
  loading pattern).

**Known gotcha (fixed during setup)**: loading Qwen2.5-VL with
`transformers==5.0.0`'s new `core_model_loading.py` path caused a large
peak-VRAM spike during 4-bit conversion (observed CUDA OOM at ~14GB for a
model that should need ~6-8GB at 4-bit). Downgrading to
`transformers==4.57.0` (which conveniently also matches the pin required by
`evaluate.py`'s BERTScore path) fixed this — 4-bit load now peaks well
under 16GB total GPU usage. `torchvision` also had to be installed
separately (`qwen_vl_utils` needs it to decode video via its `av`/
`torchvision` reader).

## Prompt design (`scripts/prompts.py`)

Each `task_type` gets a short suffix appended to the (already
task-formatted) `question` text, tuned to match the regex extractors in
`evaluate.py`:

| task_type | suffix strategy | max_new_tokens |
|---|---|---|
| `bcq` | force single-word Yes/No | 8 |
| `bcq_openended` | force leading "Yes."/"No." + 1 sentence | 96 |
| `mcq` | force single capital letter | 4 |
| `mcq_openended` | force leading "X) " + 1 sentence | 96 |
| `open_qa`, `scene_description`, `causal_linkage`, `temporal_description` | 2-4 concise sentences | 120-160 |
| `video_summarization` | 2-4 concise sentences | 180 |
| `temporal_localization` | force bare `{"start":"MM:SS","end":"MM:SS"}` JSON, no fences | 40 |

A shared system prompt frames the model as a traffic-surveillance analyst
and instructs it to skip AI disclaimers.

## Running the pipeline

```bash
cd track3_anomaly/scripts

# 1. one-off smoke test on a single downloaded clip
python inference.py --video ../data/videos/tar_test/v=-3nwOfm1Pdk_0-00_0-16.mp4 \
    --task_type bcq --question $'Does the white van flip over after the collision?\n\nAnswer with only Yes or No.'

# 2. full submission run (all 960 items / 80 videos)
python make_submission.py \
    --test_json ../data/test/test.json \
    --media_root ../data/videos \
    --out ../submissions/submission_qwen25vl_4bit.csv \
    --quant 4bit --resume

# 3. validate format locally
cd ../data
python test/evaluate.py --gt test/test.json --submission ../submissions/submission_qwen25vl_4bit.csv
```

`make_submission.py` writes the CSV incrementally after every video (so a
crash/timeout loses at most one video's worth of predictions) and supports
`--resume` to continue a partial run. Missing video files (e.g. an
age-gated YouTube source `yt-dlp` cannot fetch) get a task-appropriate
fallback answer so the CSV still covers all 960 items.

## Verification performed

1. **Environment**: fixed a broken `torchvision` import (missing, required
   by `qwen_vl_utils`' video reader) and downgraded `transformers` from the
   default `5.0.0` to `4.57.0` in `~/.local` (user site, shadows the system
   install) — `5.0.0`'s new model-loading path spiked peak VRAM to >13GB
   during 4-bit conversion alone and OOM'd; `4.57.0` loads cleanly at 4-bit
   with ~16GB total GPU usage including two unrelated processes already
   holding ~9GB (stock-bot + CARLA sim). `4.57.0` is also the exact pin
   `test/requirements.txt` specifies for `evaluate.py`'s BERTScore path, so
   this one downgrade serves both needs.
2. **Test video acquisition**: ran `test/download_test_videos.py` — 79/80
   clips fetched via `yt-dlp` (a few transient 403s cleared on retry with
   `--only-ytid`); 1 clip permanently blocked (age-gated YouTube source).
3. **Model load + single-item inference**: `scripts/inference.py` loads
   Qwen2.5-VL-7B-Instruct at 4-bit and answers a `bcq` question about a
   real downloaded clip end-to-end (`No`, plausible given the clip).
4. **Smoke test**: `scripts/make_submission.py --limit 3` ran the full
   10-task-type pipeline over 3 real videos (36 items) in 155s wall clock
   (~4.3s/item after model load), 0 errors, 0 missing videos. Output
   inspected manually (see `submissions/smoke_test.csv`) — answers are
   coherent and on-topic for every task type (e.g. correctly identifies
   Yes/No collisions, produces valid `{"start","end"}` JSON for temporal
   localization, single-letter MCQ answers, multi-sentence descriptions).
5. **Format validation**: `test/evaluate.py --gt test/test.json --submission
   submissions/smoke_test.csv --allow-missing` reports **all 36/36
   predictions parse cleanly** across all 10 task types — confirms the
   prompt-engineered output formats match the regex extractors the
   official scorer uses.
6. **Full run**: `make_submission.py` (no `--limit`) launched over all 80
   videos / 960 items to produce `submissions/submission_qwen25vl_4bit.csv`.
   At ~4-5s/item this is expected to take roughly 60-80 minutes end-to-end
   on this single 3090 (shared with other processes). See the file's
   timestamp / row count for the actual outcome; validate with
   `evaluate.py --gt test/test.json --submission submissions/submission_qwen25vl_4bit.csv`
   before submitting to the challenge evaluation server.

**Not verified**: actual accuracy/BERTScore against real ground truth
(test answers are redacted; only the challenge server can score this).
Training-video-based fine-tuning was not attempted (see limitations).

## Known limitations / remaining work

- **Test video coverage**: 79/80 `tar_test` clips downloaded successfully
  via `yt-dlp`; 1 clip (`v=ir8j5bGBTiE_0-32_0-40.mp4`) is permanently
  age-gated on YouTube and could not be fetched (yt-dlp reports "Sign in to
  confirm your age" even after retry). `make_submission.py` falls back to a
  generic answer for that item's task; this will presumably score 0 on
  whichever task type it is.
- **Training videos**: the 8 upstream training-video sources are still
  downloading in the background (`data/download_videos.py`); 3 of 8
  sources (TAD, HTV, barbados) require Kaggle credentials not yet
  configured on this box. The current pipeline is a **zero-shot / no
  fine-tuning baseline** — it does not use `train/*.json` at all. Given the
  8-day budget, this is a reasonable first baseline; a natural next step
  (not done here) is LoRA/QLoRA fine-tuning Qwen2.5-VL on the 44K training
  annotations once all training videos are available.
- **No prompt/answer-format tuning against real ground truth** — the test
  split's answers are redacted, so the only feedback loop available
  locally is `evaluate.py`'s format-validation report, not accuracy. Prompt
  suffixes were designed by inspecting the regex extractors in
  `evaluate.py`, not by iterating against scores.
- **No temporal-localization special-casing beyond the JSON-format
  instruction** — Qwen2.5-VL is prompted for `{"start","end"}` timestamps
  directly; no frame-level anomaly detector or dedicated grounding head is
  used. This task type is excluded from scoring per the 2026-07-01 notice
  anyway.
- **Single-pass greedy decoding** (`do_sample=False`) for speed/determinism;
  no self-consistency/majority voting, no reasoning-then-answer
  two-stage prompting (the `reasoning` field in training data was not
  leveraged for chain-of-thought prompting in this baseline).

## Final Track 7/8 results

- Track 7 (FETV): Korea Drive finished 3rd on the full test set. The final
  artifact is `submissions/fetv_submission_v11.json` (0.4634 official full-test
  score: description 0.4238, categorical mean 0.5031).
- Track 8 (PSI-VQA): Korea Drive finished 5th with 57.0400: BCQ 0.5045,
  OpenQA 0.6019, MCQ 0.6044, and temporal mIoU 0.5708.
- `submissions/psi_vqa_submission_v8_final.csv` is a **POST-DEADLINE RESEARCH
  ARTIFACT** and was not submitted. It contains the robust two-cue OpenQA prior and 87
  valid box-aware MCQ regenerations (39 answers differ from v7). The paired
  held-out MCQ check improved from 3/24 to 9/24 with six wins and no losses;
  the robust OpenQA prior reached 0.7816 held-out Cue-F1 under pessimistic
  one-to-one matching at k=2. These are local validation results, not an
  official leaderboard score.

## Post-challenge structured FETV prototype

**POST-DEADLINE RESEARCH ARTIFACT — NOT USED FOR THE OFFICIAL LEADERBOARD RESULT.**

`scripts/fetv_structured_pipeline.py` implements the Track 7 redesign from
the postmortem as a detector-agnostic postprocessor. It consumes tracked boxes,
lane polygons, OCR text, event flags, and scene metadata; then deterministically
selects a violator, calculates official center-square 3x3 positions, assigns
direction-relative lanes, applies conservative trajectory rules, and emits the
official 14-field FETV record.

```bash
cd track3_anomaly/scripts
python3 fetv_structured_pipeline.py \
  ../examples/fetv_structured_scene.json \
  --out /tmp/fetv_structured_output.json

cd ../../
python3 -m unittest discover -s track3_anomaly/tests \
  -p 'test_fetv_structured_pipeline.py' -v
```

This is executable research code, not a retroactively scored submission. An
upstream detector/tracker, lane segmenter, signal state estimator, and OCR
engine still need to populate the scene JSON for real videos.
