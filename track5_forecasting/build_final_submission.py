#!/usr/bin/env python3
"""Assemble the final Track5 submission via plain file copy (fast) --
use CogVideoX output where generated, freeze-frame fallback (copy the
same source frame file, not re-encode via PIL) for the rest."""
import glob
import os
import shutil
import sys

sys.path.insert(0, ".")
from schema import load_clip

TEST_ROOT = "wts_data/WTS_TRACK5_TEST"
COGVIDEOX_DIR = "submission_cogvideox"
OUT_ROOT = "submission_final"

clip_dirs = sorted(glob.glob(os.path.join(TEST_ROOT, "*")))
n_cogvideox, n_freeze = 0, 0
for cd in clip_dirs:
    clip = load_clip(cd)
    out_dir = os.path.join(OUT_ROOT, clip.clip_id)
    os.makedirs(out_dir, exist_ok=True)
    if clip.frames_to_generate == 0:
        continue

    cog_dir = os.path.join(COGVIDEOX_DIR, clip.clip_id)
    cog_frames = sorted(glob.glob(os.path.join(cog_dir, "*.png")),
                         key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if len(cog_frames) == clip.frames_to_generate:
        for i, p in enumerate(cog_frames):
            shutil.copy(p, os.path.join(out_dir, f"{i}.png"))
        n_cogvideox += 1
    else:
        src = clip.input_frame_paths[-1]
        for i in range(clip.frames_to_generate):
            shutil.copy(src, os.path.join(out_dir, f"{i}.png"))
        n_freeze += 1

print(f"done: {n_cogvideox} clips from CogVideoX, {n_freeze} clips freeze-frame fallback")
