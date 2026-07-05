#!/usr/bin/env python3
"""
Track 7 (FETV -- Track 3 out-of-domain leaderboard) submission generator.

Fisheye traffic-camera clips; one structured JSON object per clip with 12
answer_* fields + answer_description (schema documented in
github.com/MoyoG/FETV README). One combined VLM call per clip asks for all
fields at once as JSON; invalid/missing fields fall back to safe defaults.

Usage:
    python3 fetv_submission.py \
        --clips ../data/fetv/FETV_public_clips \
        --out ../submissions/fetv_submission.json [--limit N] [--resume]
"""
import argparse
import json
import os
import re
import sys
import time

VIOLATION_TYPES = ["wrong_way", "jaywalking", "red_light", "uturn",
                   "lane_use_control", "lane_discipline", "no_violation"]
VIOLATOR_TYPES = ["bus", "truck", "car", "motorcycle", "pedestrian", "na"]
COLORS = ["dark", "light", "red", "green", "yellow", "blue", "mixed", "na"]
POSITIONS = ["Top-Left", "Top-Center", "Top-Right", "Middle-Left", "Middle-Center",
             "Middle-Right", "Bottom-Left", "Bottom-Center", "Bottom-Right", "na"]
INTERSECTIONS = ["T-intersection", "four-way"]
WEATHERS = ["clear", "cloudy", "rain", "snow", "fog"]
LIGHTS = ["day", "night", "dawn_dusk"]

PROMPT = f"""You are analyzing a fisheye traffic-surveillance clip for traffic violations.

Look carefully at the entire clip and answer ALL of the following as one JSON object (no markdown fences, JSON only):

{{{{
  "answer_date": "YYYY-MM-DD read from any timestamp overlay burned into the video frames; if none visible, your best guess",
  "answer_time": "HH:MM:SS approximate time the violation occurs, read from the timestamp overlay if present",
  "answer_violation_type": "one of {VIOLATION_TYPES}",
  "answer_violator_type": "one of {VIOLATOR_TYPES} ('na' only when no violation)",
  "answer_color": "violator's color, one of {COLORS}",
  "answer_initial_position": "where the violator STARTS in a 3x3 grid over the square center crop, one of {POSITIONS}",
  "answer_final_position": "where the violator ENDS/exits, same options",
  "answer_initial_lane": "lane number where violator starts, 1 = left-most lane from driver perspective, or 'na'",
  "answer_final_lane": "lane number where violator ends, or 'na'",
  "answer_intersection_type": "one of {INTERSECTIONS}",
  "answer_weather": "one of {WEATHERS}",
  "answer_light": "one of {LIGHTS}",
  "answer_description": "2-4 sentence description of the scene and the violation (or of normal traffic if no violation)"
}}}}

Pay close attention to: any timestamp text overlay (read it exactly), the direction of traffic flow (for wrong_way), traffic-light state (for red_light), pedestrians in the roadway (jaywalking), and lane markings."""

DEFAULTS = {
    "answer_date": "2023-05-26",
    "answer_time": "12:00:00",
    "answer_violation_type": "no_violation",
    "answer_violator_type": "na",
    "answer_color": "na",
    "answer_initial_position": "na",
    "answer_final_position": "na",
    "answer_initial_lane": "na",
    "answer_final_lane": "na",
    "answer_intersection_type": "four-way",
    "answer_weather": "clear",
    "answer_light": "day",
    "answer_description": "Traffic moves through the intersection without a visible violation.",
}

VALID = {
    "answer_violation_type": VIOLATION_TYPES,
    "answer_violator_type": VIOLATOR_TYPES,
    "answer_color": COLORS,
    "answer_initial_position": POSITIONS,
    "answer_final_position": POSITIONS,
    "answer_intersection_type": INTERSECTIONS,
    "answer_weather": WEATHERS,
    "answer_light": LIGHTS,
}


def parse_response(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def clean_record(clip_name, raw):
    rec = {"clip_name": clip_name}
    for k, default in DEFAULTS.items():
        v = raw.get(k, default)
        v = str(v).strip() if v is not None else default
        allowed = VALID.get(k)
        if allowed and v not in allowed:
            # tolerate case-mismatches before falling back
            match = next((a for a in allowed if a.lower() == v.lower()), None)
            v = match or default
        rec[k] = v
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="../data/fetv/FETV_public_clips")
    ap.add_argument("--out", default="../submissions/fetv_submission.json")
    ap.add_argument("--quant", default="bf16", choices=["4bit", "8bit", "bf16"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    clips = sorted(f for f in os.listdir(args.clips) if f.endswith(".mp4"))
    if args.limit:
        clips = clips[: args.limit]

    done = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            done = {r["clip_name"]: r for r in json.load(f)}
        print(f"[resume] {len(done)} clips already answered")

    from inference import QwenVLBackend
    backend = QwenVLBackend(quant=args.quant)

    records = list(done.values())
    t0 = time.time()
    for i, clip in enumerate(clips, 1):
        clip_name = os.path.splitext(clip)[0]
        if clip_name in done:
            continue
        video_path = os.path.join(args.clips, clip)
        try:
            # reuse the open-ended path (no final_answer extraction)
            out = backend.answer(video_path, "fetv_structured", PROMPT)
            raw = parse_response(out)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {clip_name}: {e}", file=sys.stderr)
            raw = {}
        records.append(clean_record(clip_name, raw))
        if i % 10 == 0 or i == len(clips):
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(records, f, indent=1)
            print(f"[{i}/{len(clips)}] elapsed {time.time()-t0:.0f}s")

    with open(args.out, "w") as f:
        json.dump(records, f, indent=1)
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
