# Reproduction record

## Official FETV v11

- Submission: `track3_anomaly/submissions/fetv_submission_v11.json`
- Result: rank 3, final 0.4634; description 0.4238; categorical mean 0.5031
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Hub revision/commit: **not recorded in the original run** (do not infer a hash)
- Precision: bf16
- Frames: 16 maximum, 4 minimum
- Per-frame input budget: 151200 pixels (`360 * 420`)
- Decoding: `do_sample=False`; task-specific max-new-token settings in
  `track3_anomaly/scripts/prompts.py`; one structured JSON call per clip
- Prompt: `track3_anomaly/scripts/fetv_submission.py::PROMPT`
- Seed: not set by the official script; framework/runtime defaults
- Hardware: NVIDIA RTX 3090 24GB
- OS/Python: Linux, Python 3.12 (local run record)
- PyTorch: 2.10.0; Transformers: 5.13.0; CUDA: installed PyTorch CUDA build
- Runtime: not recorded in the original submission log

## Official PSI-VQA

- Submission: `track3_anomaly/submissions/psi_vqa_submission.csv`
- Result: rank 5, final 57.0400
- The post-deadline `psi_vqa_submission_v8_final.csv` is not official.

## Reproduction

Run `scripts/reproduce_fetv_official.sh` after downloading the challenge
clips and setting `TAR_HF_CACHE`. The command writes a new artifact; it does
not overwrite the official file. Validate JSON shape and clip coverage with
the challenge validator before any submission.
