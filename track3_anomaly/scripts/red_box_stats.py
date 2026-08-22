#!/usr/bin/env python3
"""Measure the red box the PSI questions refer to, per split.

Written to test whether the train split has harder grounding than test, back
when the shipped prompt's 0.2804 on train looked like it contradicted the
official 0.6044. It did not contradict anything: the submission history shows
the MCQ rows were written once on 2026-07-06 and carried forward unchanged
through six submissions, and the `psi_mcq` prompt was first committed five days
later, so the two numbers came from different configurations. See POSTMORTEM.md.

The measurement stands on its own and is why this file is kept. The boxes are
the same size in both splits -- median area 0.01770 on train against 0.01764 on
test -- so the split difference hypothesis is refuted. But box size does predict
accuracy within a split: items whose box covers at least 1% of the frame score
0.3008 against 0.2447 for the rest, 5.6 points, which is consistent with the
poster's finding that re-locating the target explicitly moved 3/24 to 9/24.

The overlay is drawn rather than photographed, so it is far more saturated than
any real red in the scene and a channel-margin threshold isolates it without a
model.
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
    # Deliberately not red_box_index.json: that name belongs to psi_box_detect.py,
    # whose schema is t0/t1/cx/cy/h and which box_hint() reads. This file is
    # area/cx/cy/found and overwriting the other one would have made --hint fail
    # silently.
    ap.add_argument("--out", default="data/psi_vqa/red_box_area_stats.json")
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
