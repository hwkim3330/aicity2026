import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from fetv_structured_pipeline import (  # noqa: E402
    Observation, Track, build_record, grid_position, infer_violation,
    lane_at, parse_overlay, select_violator,
)


class StructuredFETVTest(unittest.TestCase):
    def test_center_square_grid_on_widescreen(self):
        self.assertEqual(grid_position((320, 100), 1280, 720), "Top-Left")
        self.assertEqual(grid_position((640, 360), 1280, 720), "Middle-Center")
        self.assertEqual(grid_position((1000, 700), 1280, 720), "Bottom-Right")

    def test_lane_polygon(self):
        lanes = [
            {"number": 1, "polygon": [[0, 0], [100, 0], [100, 200], [0, 200]]},
            {"number": 2, "polygon": [[100, 0], [200, 0], [200, 200], [100, 200]]},
        ]
        self.assertEqual(lane_at((50, 50), lanes), "1")
        self.assertEqual(lane_at((150, 50), lanes), "2")
        self.assertEqual(lane_at((250, 50), lanes), "na")

    def test_overlay_parser(self):
        self.assertEqual(parse_overlay("CAM 2026/07/11 18:52:11", "x", "y"),
                         ("2026-07-11", "18:52:11"))

    def test_wrong_way_rule(self):
        track = Track(1, [
            Observation(0, (100, 200, 20, 20), "car"),
            Observation(5, (100, 50, 20, 20), "car"),
        ])
        self.assertEqual(infer_violation(track, {"legal_flow": [0, 1]}, "1", "1"),
                         "wrong_way")

    def test_selects_event_candidate(self):
        quiet = Track(1, [Observation(1, (0, 0, 10, 10), "car")])
        event = Track(2, [Observation(i, (i * 10, 0, 10, 10), "car") for i in range(5)])
        self.assertEqual(select_violator([quiet, event], {"event_frames": [0, 4]}).track_id, 2)

    def test_build_record(self):
        payload = {
            "clip_name": "001_000.mp4", "frame_size": [300, 300],
            "overlay_text": "2018-07-17 06:03:44",
            "tracks": [{"track_id": 7, "observations": [
                {"frame": 0, "bbox": [10, 10, 20, 20], "label": "person"},
                {"frame": 5, "bbox": [210, 110, 20, 20], "label": "person"},
            ]}],
            "scene": {"flags": {"outside_crosswalk": True},
                      "intersection_type": "T-intersection", "weather": "clear",
                      "light": "daylight", "track_colors": {"7": "dark"}},
        }
        record = build_record(payload)
        self.assertEqual(record["answer_violation_type"], "jaywalking")
        self.assertEqual(record["answer_violator_type"], "pedestrian")
        self.assertEqual(record["answer_initial_lane"], "na")
        self.assertTrue(record["answer_description"].startswith(
            "On 2018-07-17 at 06:03:44,"))
        self.assertEqual(set(record), {
            "clip_name", "answer_date", "answer_time", "answer_violation_type",
            "answer_violator_type", "answer_color", "answer_initial_position",
            "answer_final_position", "answer_initial_lane", "answer_final_lane",
            "answer_intersection_type", "answer_weather", "answer_light",
            "answer_description",
        })


if __name__ == "__main__":
    unittest.main()
