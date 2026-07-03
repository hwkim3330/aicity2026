"""Naive freeze-frame baseline: repeats the init frame for N steps.

Exists to validate the I/O contract (resolution, frame count, naming) before
a real generative model is wired in -- see README.md "Once the dataset
access is approved" for what should replace this.
"""
from PIL import Image

from schema import ClipRequest


def forecast(clip: ClipRequest) -> list[Image.Image]:
    init = Image.open(clip.init_frame_path).convert("RGB")
    assert init.size == (clip.width, clip.height), (
        f"init frame size {init.size} != declared ({clip.width}, {clip.height})"
    )
    return [init.copy() for _ in range(clip.num_frames)]
