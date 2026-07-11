#!/usr/bin/env python3
"""Detect the annotator-drawn red pedestrian bounding box in PSI clips.

The questions claim the box is 'visible at the start' but in reality it is
drawn only for a ~31-frame (~1s) window that can occur ANYWHERE in the clip
(verified on train/ambiguous). Uniform 16-frame sampling therefore often
never (clearly) shows the model which pedestrian is meant. This script scans
every frame for the pure-red outline, and records when the box is visible,
where it is (normalized centroid + bbox), so the prompt can tell the model
where/when to look.

Output: JSON {video_rel: {t0, t1, cx, cy, w, h, npx, nframes, fps, dur}}
"""
import cv2
import glob
import json
import os
import sys

import numpy as np


def scan(vp):
    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = 0
    hits = []  # (frame, count, cx, cy, w, h)
    W = H = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if W is None:
            H, W = fr.shape[:2]
        b = fr[:, :, 0].astype(np.int16)
        g = fr[:, :, 1].astype(np.int16)
        r = fr[:, :, 2].astype(np.int16)
        mask = (r > 150) & (g < 90) & (b < 90)
        cnt = int(mask.sum())
        if cnt > 15:
            ys, xs = np.nonzero(mask)
            hits.append((n, cnt, float(xs.mean()), float(ys.mean()),
                         int(xs.max() - xs.min()), int(ys.max() - ys.min())))
        n += 1
    cap.release()
    if not hits:
        return {"nframes": n, "fps": fps, "found": False}
    best = max(hits, key=lambda h: h[1])
    return {
        "nframes": n, "fps": fps, "found": True,
        "t0": round(hits[0][0] / fps, 2), "t1": round(hits[-1][0] / fps, 2),
        "nbox": len(hits),
        "cx": round(best[2] / W, 3), "cy": round(best[3] / H, 3),
        "w": best[4], "h": best[5], "npx": best[1],
    }


def main():
    out = {}
    roots = [
        ("train", "../data/psi_vqa/train/videos/ambiguous"),
        ("test", "../data/psi_vqa/test_public/videos/ambiguous"),
    ]
    for tag, root in roots:
        vids = sorted(glob.glob(os.path.join(root, "*.mp4")))
        print(f"{tag}: {len(vids)} videos", file=sys.stderr)
        for i, vp in enumerate(vids, 1):
            rel = "ambiguous/" + os.path.basename(vp)
            out[rel] = scan(vp)
            if i % 20 == 0:
                print(f"  {tag} {i}/{len(vids)}", file=sys.stderr)
    with open("../data/psi_vqa/red_box_index.json", "w") as f:
        json.dump(out, f, indent=1)
    nf = sum(1 for v in out.values() if not v.get("found"))
    print(f"wrote red_box_index.json: {len(out)} videos, {nf} without box")


if __name__ == "__main__":
    main()
