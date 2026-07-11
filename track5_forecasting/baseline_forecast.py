"""Naive freeze-frame baseline: repeats the LAST input frame for however
many frames the clip needs generated.

Exists to get a real, valid submission in given very few competing teams
(3 total on the public leaderboard as of 2026-07-08) -- a correct freeze-
frame beats no submission at all. See README.md for stronger follow-ups
(optical-flow extrapolation, diffusion-based generation) if time allows.
"""
from PIL import Image

from schema import ClipRequest


def forecast(clip: ClipRequest) -> list[Image.Image]:
    last = Image.open(clip.input_frame_paths[-1]).convert("RGB")
    assert last.size == (clip.width, clip.height), (
        f"last input frame size {last.size} != declared ({clip.width}, {clip.height})"
    )
    return [last.copy() for _ in range(clip.frames_to_generate)]
