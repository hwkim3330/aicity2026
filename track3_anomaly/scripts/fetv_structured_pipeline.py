#!/usr/bin/env python3
"""Structured FETV postprocessor for detector/tracker outputs.

POST-DEADLINE RESEARCH ARTIFACT
NOT USED FOR THE OFFICIAL LEADERBOARD RESULT

This is a post-challenge prototype of the Track 7 redesign. It deliberately
separates geometry and metadata from language generation. The input is JSON
containing tracked boxes and scene configuration; the output is one official
FETV record. Detection, tracking, lane segmentation, and OCR can be supplied
by any upstream implementation.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

POSITION_ROWS = ("Top", "Middle", "Bottom")
POSITION_COLS = ("Left", "Center", "Right")
VEHICLES = {"bus", "truck", "car", "motorcycle"}
VALID_VIOLATIONS = {
    "wrong_way", "jaywalking", "red_light", "uturn",
    "lane_use_control", "lane_discipline", "no_violation",
}


@dataclass(frozen=True)
class Observation:
    frame: int
    bbox: tuple[float, float, float, float]
    label: str
    confidence: float = 1.0

    @property
    def foot(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h


@dataclass
class Track:
    track_id: int
    observations: list[Observation]

    @property
    def label(self) -> str:
        weights: dict[str, float] = {}
        for obs in self.observations:
            weights[obs.label] = weights.get(obs.label, 0.0) + obs.confidence
        return max(weights, key=weights.get)

    @property
    def start(self) -> Observation:
        return min(self.observations, key=lambda item: item.frame)

    @property
    def end(self) -> Observation:
        return max(self.observations, key=lambda item: item.frame)

    @property
    def displacement(self) -> tuple[float, float]:
        sx, sy = self.start.foot
        ex, ey = self.end.foot
        return ex - sx, ey - sy

    @property
    def path_length(self) -> float:
        points = [obs.foot for obs in sorted(self.observations, key=lambda x: x.frame)]
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def grid_position(point: tuple[float, float], width: int, height: int) -> str:
    """Map a point to the official 3x3 grid over the centered square crop."""
    side = min(width, height)
    left = (width - side) / 2.0
    x = min(max((point[0] - left) / side, 0.0), 1.0 - 1e-9)
    y = min(max(point[1] / side, 0.0), 1.0 - 1e-9)
    return f"{POSITION_ROWS[int(y * 3)]}-{POSITION_COLS[int(x * 3)]}"


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon without an OpenCV dependency."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def lane_at(point: tuple[float, float], lanes: list[dict]) -> str:
    """Return direction-relative lane number from configured lane polygons."""
    for lane in lanes:
        if point_in_polygon(point, lane["polygon"]):
            number = str(lane["number"])
            if number not in {"1", "2", "3", "4"}:
                raise ValueError(f"invalid lane number: {number}")
            return number
    return "na"


def parse_overlay(text: str, fallback_date: str, fallback_time: str) -> tuple[str, str]:
    date_match = re.search(r"(20\d{2})[-/.](\d{2})[-/.](\d{2})", text)
    time_match = re.search(r"\b(\d{2}):(\d{2}):(\d{2})\b", text)
    date = "-".join(date_match.groups()) if date_match else fallback_date
    clock = ":".join(time_match.groups()) if time_match else fallback_time
    return date, clock


def cosine(a: tuple[float, float], b: tuple[float, float]) -> float:
    denom = math.hypot(*a) * math.hypot(*b)
    return (a[0] * b[0] + a[1] * b[1]) / denom if denom else 1.0


def infer_violation(track: Track, scene: dict, start_lane: str, end_lane: str) -> str:
    """Conservative trajectory rules; scene signals come from upstream models."""
    flags = scene.get("flags", {})
    actor = normalize_actor(track.label)
    if flags.get("red_signal_crossing"):
        return "red_light"
    if flags.get("prohibited_lane_use"):
        return "lane_use_control"

    dx, dy = track.displacement
    flow = tuple(scene.get("legal_flow", [0.0, 1.0]))
    if actor in VEHICLES and cosine((dx, dy), flow) < -0.5:
        return "wrong_way"

    points = [obs.foot for obs in sorted(track.observations, key=lambda x: x.frame)]
    if len(points) >= 3:
        first = (points[len(points) // 2][0] - points[0][0],
                 points[len(points) // 2][1] - points[0][1])
        second = (points[-1][0] - points[len(points) // 2][0],
                  points[-1][1] - points[len(points) // 2][1])
        if actor in VEHICLES and cosine(first, second) < -0.75:
            return "uturn"

    if actor == "pedestrian" and flags.get("outside_crosswalk"):
        if abs(dx) > scene.get("crossing_displacement_px", 80):
            return "jaywalking"
    if start_lane != "na" and end_lane != "na" and start_lane != end_lane:
        if flags.get("unjustified_lane_change"):
            return "lane_discipline"
    return "no_violation"


def select_violator(tracks: Iterable[Track], scene: dict) -> Track:
    """Rank candidates by event overlap, motion, and upstream violation score."""
    start, end = scene.get("event_frames", [0, 10**9])
    priors = scene.get("violation_scores", {})

    def score(track: Track) -> float:
        overlap = sum(start <= obs.frame <= end for obs in track.observations)
        confidence = sum(obs.confidence for obs in track.observations)
        prior = float(priors.get(str(track.track_id), 0.0))
        return 4.0 * prior + overlap + 0.01 * track.path_length + 0.1 * confidence

    candidates = [track for track in tracks if track.observations]
    if not candidates:
        raise ValueError("no non-empty tracks")
    return max(candidates, key=score)


def normalize_actor(label: str) -> str:
    label = label.lower()
    aliases = {"person": "pedestrian", "pickup": "truck", "van": "car"}
    return aliases.get(label, label) if aliases.get(label, label) in {
        "bus", "truck", "car", "motorcycle", "pedestrian"
    } else "car"


def build_record(payload: dict) -> dict:
    width, height = payload["frame_size"]
    tracks = [Track(
        int(item["track_id"]),
        [Observation(int(obs["frame"]), tuple(map(float, obs["bbox"])),
                     obs["label"], float(obs.get("confidence", 1.0)))
         for obs in item["observations"]],
    ) for item in payload["tracks"]]
    scene = payload.get("scene", {})
    violator = select_violator(tracks, scene)
    actor = normalize_actor(violator.label)
    start_pos = grid_position(violator.start.foot, width, height)
    end_pos = grid_position(violator.end.foot, width, height)
    lanes = scene.get("lanes", [])
    start_lane = "na" if actor == "pedestrian" else lane_at(violator.start.foot, lanes)
    end_lane = "na" if actor == "pedestrian" else lane_at(violator.end.foot, lanes)
    violation = infer_violation(violator, scene, start_lane, end_lane)
    date, clock = parse_overlay(
        payload.get("overlay_text", ""),
        scene.get("date", "2018-07-17"), scene.get("time", "12:00:00"),
    )
    no_violation = violation == "no_violation"
    actor_out = "na" if no_violation else actor
    color = "na" if no_violation else scene.get("track_colors", {}).get(
        str(violator.track_id), "dark")
    if no_violation:
        description = (f"On {date} at {clock}, traffic moved normally through "
                       f"the {scene.get('intersection_type', 'four-way intersection')} "
                       "with no violation.")
    else:
        description = (f"On {date} at {clock}, a {color} {actor_out} moved from "
                       f"the {start_pos} to the {end_pos} area and committed a "
                       f"{violation.replace('_', ' ')} violation.")
    return {
        "clip_name": payload["clip_name"],
        "answer_date": date,
        "answer_time": clock,
        "answer_violation_type": violation,
        "answer_violator_type": actor_out,
        "answer_color": color,
        "answer_initial_position": "na" if no_violation else start_pos,
        "answer_final_position": "na" if no_violation else end_pos,
        "answer_initial_lane": "na" if no_violation else start_lane,
        "answer_final_lane": "na" if no_violation else end_lane,
        "answer_intersection_type": scene.get("intersection_type", "four-way intersection"),
        "answer_weather": scene.get("weather", "clear"),
        "answer_light": scene.get("light", "daylight"),
        "answer_description": description,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="tracked-scene JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = build_record(json.loads(args.input.read_text()))
    if record["answer_violation_type"] not in VALID_VIOLATIONS:
        raise ValueError("invalid violation output")
    args.out.write_text(json.dumps([record], indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
