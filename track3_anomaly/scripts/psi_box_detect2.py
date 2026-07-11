#!/usr/bin/env python3
"""v2 red-box detector with temporal differencing: the annotation box is
drawn only during a sub-window of the clip, while other red objects (stop
signs, red cars, brake lights) persist across frames. Subtracting the red
mask of an outside-window reference frame isolates the drawn box, giving a
clean centroid + box size. Falls back to the outline-shaped connected
component when the box is drawn for (nearly) the whole clip."""
import glob
import json
import os
import sys

import cv2
import numpy as np


def red_mask(fr):
    b = fr[:, :, 0].astype(np.int16)
    g = fr[:, :, 1].astype(np.int16)
    r = fr[:, :, 2].astype(np.int16)
    return ((r > 150) & (g < 90) & (b < 90)).astype(np.uint8)


def best_component(mask):
    """largest connected component; returns (cx, cy, w, h, npx) or None"""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 10:
            continue
        if best is None or area > best[4]:
            best = (cents[i][0], cents[i][1], w, h, area)
    return best


def scan(vp):
    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    if n == 0:
        return {"found": False, "nframes": 0, "fps": fps}
    H, W = frames[0].shape[:2]
    masks = [red_mask(f) for f in frames]
    counts = np.array([int(m.sum()) for m in masks])
    base = int(np.median(counts))  # persistent red background level
    thr = max(15, base + 15)
    hit = counts > thr
    if not hit.any():
        # maybe box drawn whole clip: any red at all?
        if counts.max() <= 15:
            return {"found": False, "nframes": n, "fps": fps}
        hit = counts > 15
    idx = np.nonzero(hit)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    ibest = int(idx[np.argmax(counts[idx])])
    m = masks[ibest]
    if len(idx) < 0.9 * n:
        # temporal differencing: remove persistent red using a reference
        # frame far from the window
        ref_candidates = [i for i in range(n) if not hit[i]]
        if ref_candidates:
            ref = min(ref_candidates, key=lambda i: counts[i])
            refm = cv2.dilate(masks[ref], np.ones((7, 7), np.uint8))
            m = (m & (1 - refm)).astype(np.uint8)
    comp = best_component(m)
    if comp is None:
        comp = best_component(masks[ibest])
    if comp is None:
        return {"found": False, "nframes": n, "fps": fps}
    cx, cy, w, h, npx = comp
    return {
        "found": True, "nframes": n, "fps": fps,
        "t0": round(i0 / fps, 2), "t1": round(i1 / fps, 2),
        "nbox": int(len(idx)),
        "cx": round(cx / W, 3), "cy": round(cy / H, 3),
        "w": int(w), "h": int(h), "npx": int(npx),
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
    with open("../data/psi_vqa/red_box_index2.json", "w") as f:
        json.dump(out, f, indent=1)
    nf = sum(1 for v in out.values() if not v.get("found"))
    print(f"wrote red_box_index2.json: {len(out)} videos, {nf} without box")


if __name__ == "__main__":
    main()
