"""Write a clip's forecasted frames as 0.png..N-1.png, per the submission spec."""
import os

from PIL import Image


def write_frames(clip_id: str, frames: list[Image.Image], out_root: str = "submission") -> str:
    clip_dir = os.path.join(out_root, clip_id)
    os.makedirs(clip_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(clip_dir, f"{i}.png"))
    return clip_dir
