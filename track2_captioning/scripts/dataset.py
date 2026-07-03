#!/usr/bin/env python3
"""
Enumerates SynWTS scenarios and pairs each event-phase caption segment /
VQA question set with a representative video file.

Directory layout consumed (matches data/README.md):
  <root>/videos/<split>/<scenario>/<view>/*.mp4
  <root>/annotations/caption/<split>/<scenario>/<view>/<scenario>_caption.json
  <root>/annotations/vqa/<split>/<scenario>/<view>/<scenario>.json

<view> in {overhead_view, vehicle_view, environment}. `environment` has no
video (it's a text-only VQA subset about weather/road/etc.), so it's
skipped for captioning and for VQA it falls back to the vehicle_view clip
if present, else overhead_view.
"""

import glob
import json
import os
from dataclasses import dataclass, field


@dataclass
class CaptionItem:
    scenario: str
    split: str
    view: str
    video_path: str
    event_phase: list  # raw list of {labels, caption_pedestrian, caption_vehicle, start_time, end_time}


@dataclass
class VqaItem:
    scenario: str
    split: str
    view: str
    video_path: str  # may be None for environment-only questions with no clip found
    q_index: int
    question: dict  # {question, a, b, c, d, correct}
    start_time: float = None  # None for `environment` (not time-scoped)
    end_time: float = None


def _pick_video(video_dir: str) -> str:
    """First-camera heuristic: sorted alphabetically, take the first mp4."""
    vids = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    return vids[0] if vids else None


def load_captions(root: str, split: str):
    items = []
    cap_root = os.path.join(root, "annotations", "caption", split)
    for scenario in sorted(os.listdir(cap_root)):
        scenario_dir = os.path.join(cap_root, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        for view in sorted(os.listdir(scenario_dir)):
            view_dir = os.path.join(scenario_dir, view)
            json_files = glob.glob(os.path.join(view_dir, "*_caption.json"))
            if not json_files:
                continue
            with open(json_files[0]) as f:
                data = json.load(f)
            video_dir = os.path.join(root, "videos", split, scenario, view)
            video_path = _pick_video(video_dir)
            items.append(
                CaptionItem(
                    scenario=scenario,
                    split=split,
                    view=view,
                    video_path=video_path,
                    event_phase=data.get("event_phase", []),
                )
            )
    return items


def load_vqa(root: str, split: str):
    items = []
    vqa_root = os.path.join(root, "annotations", "vqa", split)
    for scenario in sorted(os.listdir(vqa_root)):
        scenario_dir = os.path.join(vqa_root, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        for view in sorted(os.listdir(scenario_dir)):
            view_dir = os.path.join(scenario_dir, view)
            json_files = glob.glob(os.path.join(view_dir, "*.json"))
            if not json_files:
                continue
            with open(json_files[0]) as f:
                data = json.load(f)
            if view == "environment":
                # no dedicated clip; fall back to vehicle_view then overhead_view
                video_path = _pick_video(os.path.join(root, "videos", split, scenario, "vehicle_view"))
                if video_path is None:
                    ov_dir = os.path.join(root, "videos", split, scenario, "overhead_view")
                    video_path = _pick_video(ov_dir)
            else:
                video_path = _pick_video(os.path.join(root, "videos", split, scenario, view))

            for entry in data:
                if view == "environment":
                    # flat list of questions, not time-scoped
                    for qi, q in enumerate(entry.get("environment", [])):
                        items.append(
                            VqaItem(
                                scenario=scenario, split=split, view=view,
                                video_path=video_path, q_index=qi, question=q,
                            )
                        )
                else:
                    # event_phase: [{start_time, end_time, labels, conversations: [q, ...]}, ...]
                    qi = 0
                    for phase in entry.get("event_phase", []):
                        for q in phase.get("conversations", []):
                            items.append(
                                VqaItem(
                                    scenario=scenario, split=split, view=view,
                                    video_path=video_path, q_index=qi, question=q,
                                    start_time=float(phase["start_time"]) if phase.get("start_time") else None,
                                    end_time=float(phase["end_time"]) if phase.get("end_time") else None,
                                )
                            )
                            qi += 1
    return items


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "../data/data/data"
    caps = load_captions(root, "val")
    vqas = load_vqa(root, "val")
    print(f"caption items: {len(caps)}, vqa items: {len(vqas)}")
    if caps:
        c = caps[0]
        print("sample caption item:", c.scenario, c.view, c.video_path, len(c.event_phase))
    if vqas:
        v = vqas[0]
        print("sample vqa item:", v.scenario, v.view, v.video_path, v.q_index, v.question.get("question"))
