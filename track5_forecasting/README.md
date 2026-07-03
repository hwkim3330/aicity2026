# Track 5 — Generative Traffic Video Forecasting

Status: **blocked on data access.** Dataset requires Google Form approval
(https://forms.gle/szQPk1TMR8JXzm327) — submitted but not yet approved as of
2026-07-03. Everything below is built against the task description in
`shared/notes/tracks_overview.md`, not a real sample, so field names/paths
are best-guess and will need a quick pass once real data arrives.

## Task (from public rules)

- Input: two captions + one initial frame per clip.
- Output: a sequence of future frames, same resolution as the input frame.
- 810 videos / 155 scenarios total. Pretraining on BDD_PC_5K is allowed.
- Metric: mean of PSNR / SSIM / LPIPS / CLIP-S / FVD.
- Submission: frames named `0.png` ... `N-1.png` per clip.
- Constraint: cannot train on WTS test data; must disclose any
  pretrained/external data used.

## What's here (runnable now, without real data)

```
schema.py            <- dataclasses for one forecasting "clip" request
make_dummy_sample.py <- generates a fake clip (random init frame + 2 captions)
                         so the rest of the pipeline is testable today
baseline_forecast.py <- naive baseline: holds the last frame (freeze-frame)
                         for N steps. Trivial, but validates the I/O contract
                         end-to-end: same resolution, correct frame count,
                         correct file naming.
write_submission.py  <- writes a clip's frames as 0.png..N-1.png
run_baseline.py       <- wires the three together, prints where output landed
```

Run the smoke test:

```bash
python3 make_dummy_sample.py
python3 run_baseline.py
```

## Once the dataset access is approved

1. Confirm the real field names for captions/init-frame/frame-count against
   `schema.py` — update it, this is currently a guess.
2. Point `run_baseline.py` at the real clip list instead of the dummy sample.
3. Replace `baseline_forecast.py`'s freeze-frame logic with an actual
   generative model. Candidates, roughly in order of effort:
   - Optical-flow extrapolation (RAFT flow forward-warped from init frame) —
     still cheap, usually beats freeze-frame on PSNR/SSIM.
   - Image-to-video diffusion (e.g. Stable Video Diffusion / CogVideoX-I2V)
     conditioned on the init frame, captions injected via text-to-video
     cross-attention or as a conditioning prefix — better FVD/CLIP-S, needs
     the pretrained checkpoint disclosed per the rules.
   - Fine-tune the above on BDD_PC_5K + the real training clips once
     downloaded.
4. Wire in the actual eval metrics (PSNR/SSIM/LPIPS/CLIP-S/FVD) locally
   before submitting, to catch resolution/frame-count mismatches early.
