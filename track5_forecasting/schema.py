"""Best-guess data contract for a Track 5 forecasting clip.

Field names are not yet confirmed against the real dataset (pending Google
Form approval) -- update this once real samples are available.
"""
from dataclasses import dataclass


@dataclass
class ClipRequest:
    clip_id: str
    caption_1: str
    caption_2: str
    init_frame_path: str
    num_frames: int
    width: int
    height: int
