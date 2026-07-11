#!/usr/bin/env python3
import sys, time, os
sys.path.insert(0, ".")
from schema import load_clip
import torch
from diffusers import CogVideoXImageToVideoPipeline
from PIL import Image

t0 = time.time()
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
).to("cuda")
pipe.vae.enable_tiling()
print(f"load: {time.time()-t0:.1f}s", file=sys.stderr)

clip = load_clip("wts_data/WTS_TRACK5_TEST/20230707_14_CN16_T1_Camera2_3")
print(f"clip: {clip.clip_id}, need {clip.frames_to_generate} frames, {clip.width}x{clip.height}", file=sys.stderr)

last_frame = Image.open(clip.input_frame_paths[-1]).convert("RGB")
prompt = f"{clip.caption_pedestrian} {clip.caption_vehicle}".strip()

t1 = time.time()
out = pipe(
    image=last_frame.resize((720, 480)),
    prompt=prompt,
    num_frames=49,
    num_inference_steps=30,
    guidance_scale=6.0,
).frames[0]
print(f"generate (no offload): {time.time()-t1:.1f}s, got {len(out)} frames", file=sys.stderr)

os.makedirs("cogvideox_test_frames", exist_ok=True)
for i, f in enumerate(out):
    f.resize((1280, 720)).save(f"cogvideox_test_frames/{i}.png")
print("wrote frames to cogvideox_test_frames/", file=sys.stderr)
