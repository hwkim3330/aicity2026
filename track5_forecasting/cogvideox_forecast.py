#!/usr/bin/env python3
"""CogVideoX-5B-I2V based forecast: conditions on the last input frame +
pedestrian/vehicle captions to generate more visually/semantically plausible
continuations than the freeze-frame baseline. Falls back to freeze-frame
per-clip on any generation error so a bad clip never blocks the run.

Chains passes for clips needing >49 frames (the model's native output
length): each pass's last frame seeds the next pass's image conditioning.
"""
import gc
import glob
import os
import sys
import time

import torch
from diffusers import CogVideoXImageToVideoPipeline
from PIL import Image

sys.path.insert(0, ".")
from schema import ClipRequest, load_clip

NATIVE_FRAMES = 49
GEN_RESOLUTION = (720, 480)  # CogVideoX-5B-I2V native (w, h)
STEPS = 25
GUIDANCE = 6.0

_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
        )
        _pipe.enable_model_cpu_offload()
        _pipe.vae.enable_tiling()
    return _pipe


def forecast(clip: ClipRequest) -> list:
    if clip.frames_to_generate == 0:
        return []
    pipe = get_pipe()
    prompt = f"{clip.caption_pedestrian} {clip.caption_vehicle}".strip()[:900]
    seed_img = Image.open(clip.input_frame_paths[-1]).convert("RGB").resize(GEN_RESOLUTION)

    frames_out = []
    remaining = clip.frames_to_generate
    while remaining > 0:
        out = pipe(
            image=seed_img,
            prompt=prompt,
            num_frames=NATIVE_FRAMES,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
        ).frames[0]
        take = min(remaining, len(out))
        frames_out.extend(out[:take])
        remaining -= take
        if remaining > 0:
            seed_img = out[-1]  # chain: last generated frame seeds next pass

    return [f.resize((clip.width, clip.height)) for f in frames_out]


def write_frames(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(out_dir, f"{i}.png"))


def main():
    test_root = "wts_data/WTS_TRACK5_TEST"
    out_root = "submission_cogvideox"
    clip_dirs = sorted(glob.glob(os.path.join(test_root, "*")))
    print(f"[cogvideox_forecast] {len(clip_dirs)} clips", file=sys.stderr)

    n_ok, n_fallback = 0, 0
    for i, cd in enumerate(clip_dirs, 1):
        clip = load_clip(cd)
        out_dir = os.path.join(out_root, clip.clip_id)
        if os.path.isdir(out_dir) and len(glob.glob(os.path.join(out_dir, "*.png"))) == clip.frames_to_generate:
            continue  # resume support
        t0 = time.time()
        try:
            frames = forecast(clip)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {clip.clip_id}: {e} -- falling back to freeze-frame", file=sys.stderr)
            last = Image.open(clip.input_frame_paths[-1]).convert("RGB")
            frames = [last.copy() for _ in range(clip.frames_to_generate)]
            n_fallback += 1
            torch.cuda.empty_cache()
            gc.collect()
        write_frames(frames, out_dir)
        print(f"[{i}/{len(clip_dirs)}] {clip.clip_id}: {clip.frames_to_generate} frames, "
              f"{time.time()-t0:.0f}s", file=sys.stderr)

    print(f"done: ok={n_ok} fallback={n_fallback}")


if __name__ == "__main__":
    main()
