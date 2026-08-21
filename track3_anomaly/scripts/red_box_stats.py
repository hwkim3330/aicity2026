#!/usr/bin/env python3
"""Measure the red box the PSI questions refer to, per split.

The harness scores 0.2804 on 321 labelled train MCQ items where the official
test score is 0.6044 (exactly 55/91), and nothing about the labels, prompts,
videos, metric or class balance explains it. Reading three of the disagreements
frame by frame points at one thing: in each case the red box marks a small,
distant subject while something much more salient moves through the frame, and
the model describes the salient one.

That is the grounding failure the poster already reports -- re-locating the
target explicitly took 3/24 to 9/24 on the paired MCQ study. If the train split
simply has smaller and more distant boxes than the test split, both the 2x gap
and the poster's finding have the same cause, and the harness is not broken so
much as measuring a harder split.

So: find the box and measure it. It is drawn in saturated red on the first
frame, which a colour threshold isolates without a model. Reported per split:
box area as a fraction of frame, and the vertical position of its centre, which
is a rough proxy for distance in a forward-facing camera.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def first_frame(path: Path):
    import av
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            return frame.to_ndarray(format="rgb24")
    return None


def red_box(rgb: np.ndarray) -> tuple[float, float, float] | None:
    """(area fraction, centre y fraction, centre x fraction) of the red overlay.

    The overlay is drawn, not photographed, so it is far more saturated than any
    real red in the scene: a wide margin over both other channels isolates it
    where a hue threshold would also catch brake lights and signage.
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mask = (r > 130) & (r - g > 70) & (r - b > 70)
    if mask.sum() < 12:
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    # the box is an outline, so its extent is the quantity of interest, not the
    # pixel count of the stroke
    area = ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / (h * w)
    return float(area), float((ys.min() + ys.max()) / 2 / h), float((xs.min() + xs.max()) / 2 / w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/psi_vqa")
    ap.add_argument("--out", default="data/psi_vqa/red_box_index.json")
    args = ap.parse_args()

    root = Path(args.data)
    splits = {
        "train": (root / "train/mcq.json", root / "train/videos"),
        "test": (root / "test_public/mcq_questions.json", root / "test_public/videos"),
    }
    index: dict[str, dict] = {}
    for name, (qfile, vroot) in splits.items():
        if not qfile.exists():
            print(f"{name}: {qfile} missing, skipped")
            continue
        blob = json.loads(qfile.read_text())
        items = blob["items"] if isinstance(blob, dict) else blob
        areas, ys, found = [], [], 0
        for it in items:
            rel = it["video_id"].split("PSI/", 1)[-1]
            path = vroot / rel
            if not path.exists():
                continue
            frame = first_frame(path)
            if frame is None:
                continue
            got = red_box(frame)
            if got is None:
                index[it["video_id"]] = {"found": False}
                continue
            area, cy, cx = got
            index[it["video_id"]] = {"found": True, "area": area, "cy": cy, "cx": cx}
            areas.append(area)
            ys.append(cy)
            found += 1
        a = np.array(areas)
        print(f"{name}: {len(items)} items, box found on {found}")
        if found:
            print(f"  area fraction   median {np.median(a):.5f}  "
                  f"p25 {np.percentile(a, 25):.5f}  p75 {np.percentile(a, 75):.5f}")
            print(f"  centre y        median {np.median(ys):.3f}  "
                  f"(higher = lower in frame = nearer)")
            print(f"  boxes under 0.1% of frame: {np.mean(a < 0.001):.1%}")

    Path(args.out).write_text(json.dumps(index))
    print(f"\nwrote {args.out} ({len(index)} entries)")


if __name__ == "__main__":
    main()
