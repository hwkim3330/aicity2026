"""Data contract for a Track 5 forecasting clip, confirmed against the real
WTS_TRACK5_TEST dataset (received 2026-07-08).

Real layout per clip, under <root>/<clip_id>/:
    input/0.png .. input/{N-1}.png   -- N observed context frames (N varies
                                         per clip, e.g. 27, 42)
    caption.json                     -- {id, event_phase: [{labels,
                                         caption_pedestrian, caption_vehicle}],
                                         "frame length": <total clip length>}

"frame length" is the full clip's frame count; the number of frames we must
generate is `frame_length - len(input_frames)`. Submission format per the
official rules: write the GENERATED frames only, numbered fresh as
0.png..{M-1}.png (M = frames_to_generate), same resolution as the input
frames.
"""
import glob
import json
import os
from dataclasses import dataclass


@dataclass
class ClipRequest:
    clip_id: str
    caption_pedestrian: str
    caption_vehicle: str
    input_frame_paths: list  # sorted by frame index, 0.png first
    frames_to_generate: int
    width: int
    height: int


def load_clip(clip_dir: str) -> ClipRequest:
    clip_id = os.path.basename(clip_dir.rstrip("/"))
    with open(os.path.join(clip_dir, "caption.json")) as f:
        cap = json.load(f)
    phase = cap["event_phase"][0]
    input_paths = sorted(
        glob.glob(os.path.join(clip_dir, "input", "*.png")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    frame_length = cap["frame length"]
    frames_to_generate = max(0, frame_length - len(input_paths))
    from PIL import Image
    with Image.open(input_paths[-1]) as im:
        width, height = im.size
    return ClipRequest(
        clip_id=clip_id,
        caption_pedestrian=phase.get("caption_pedestrian", ""),
        caption_vehicle=phase.get("caption_vehicle", ""),
        input_frame_paths=input_paths,
        frames_to_generate=frames_to_generate,
        width=width,
        height=height,
    )
