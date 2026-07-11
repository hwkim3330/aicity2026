#!/usr/bin/env python3
import sys, time
sys.path.insert(0, ".")
from schema import load_clip
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image

t0 = time.time()
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
print(f"load: {time.time()-t0:.1f}s", file=sys.stderr)

clip = load_clip("wts_data/WTS_TRACK5_TEST/20230707_14_CN16_T1_Camera2_3")
print(f"clip: {clip.clip_id}, need {clip.frames_to_generate} frames, {clip.width}x{clip.height}", file=sys.stderr)

last_frame = Image.open(clip.input_frame_paths[-1]).convert("RGB")
prompt = f"{clip.caption_pedestrian} {clip.caption_vehicle}".strip()
print(f"prompt: {prompt[:200]}", file=sys.stderr)

t1 = time.time()
out = pipe(
    image=last_frame.resize((720, 480)),
    prompt=prompt,
    num_frames=49,
    num_inference_steps=30,
    guidance_scale=6.0,
).frames[0]
print(f"generate: {time.time()-t1:.1f}s, got {len(out)} frames", file=sys.stderr)

export_to_video(out, "cogvideox_test_out.mp4", fps=8)
print("wrote cogvideox_test_out.mp4", file=sys.stderr)
