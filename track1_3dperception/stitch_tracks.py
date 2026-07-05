"""stitch_tracks.py -- offline tracklet stitching over fused global tracks.

The distance-based MTMC fusion fragments identities badly (Warehouse_023
ended up with 11k global ids for 9000 frames): every time a per-camera track
breaks (occlusion, missed detections) the fusion mints a fresh global id,
which wrecks AssA. This pass greedily links global tracks of the same class
when one ends and another begins shortly after at a spatially consistent
location -- the standard offline tracklet-association step.

Conservative by default: gap <= 60 frames, endpoint distance <= 2.0m plus a
velocity-extrapolation allowance. Runs on cache/global_tracks/<scene>.json in
place (rewrites global ids), so re-run export_submission.py afterwards.

Usage:
    python3 stitch_tracks.py --scene Warehouse_023 [--max-gap 60] [--max-dist 2.0]
"""
import argparse
import json
import math
import os
from collections import defaultdict


def load_tracks(data):
    """global_id -> {class, frames: sorted [(frame, x, y)]}"""
    tracks = defaultdict(lambda: {"cls": None, "pts": []})
    for fk, dets in data["frames"].items():
        f = int(fk)
        for d in dets:
            t = tracks[d["global_id"]]
            t["cls"] = d["target_class"]
            t["pts"].append((f, d["x"], d["y"]))
    for t in tracks.values():
        t["pts"].sort()
    return tracks


def endpoint_velocity(pts, k=5, from_end=True):
    seg = pts[-k:] if from_end else pts[:k]
    if len(seg) < 2:
        return 0.0, 0.0
    (f0, x0, y0), (f1, x1, y1) = seg[0], seg[-1]
    df = max(f1 - f0, 1)
    return (x1 - x0) / df, (y1 - y0) / df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--global-tracks-dir", default="cache/global_tracks")
    ap.add_argument("--max-gap", type=int, default=60)
    ap.add_argument("--max-dist", type=float, default=2.0)
    args = ap.parse_args()

    path = os.path.join(args.global_tracks_dir, args.scene + ".json")
    with open(path) as f:
        data = json.load(f)

    tracks = load_tracks(data)
    # order tracks by start frame; greedily attach each to the best earlier track
    order = sorted(tracks, key=lambda g: tracks[g]["pts"][0][0])
    merged_into = {}

    def resolve(g):
        while g in merged_into:
            g = merged_into[g]
        return g

    # ends[class] = list of (end_frame, gid) for candidate predecessors
    open_ends = defaultdict(list)
    for gid in order:
        t = tracks[gid]
        start_f, sx, sy = t["pts"][0]
        best = None
        for end_f, cand in open_ends[t["cls"]]:
            gap = start_f - end_f
            if gap < 1 or gap > args.max_gap:
                continue
            cand_t = tracks[cand]
            ef, ex, ey = cand_t["pts"][-1]
            vx, vy = endpoint_velocity(cand_t["pts"])
            px, py = ex + vx * gap, ey + vy * gap  # extrapolated position
            dist = min(math.hypot(sx - ex, sy - ey), math.hypot(sx - px, sy - py))
            if dist <= args.max_dist and (best is None or dist < best[0]):
                best = (dist, cand)
        if best is not None:
            target = resolve(best[1])
            merged_into[gid] = target
            # the merged track's new end is this track's end
            tracks[target]["pts"].extend(t["pts"])
            tracks[target]["pts"].sort()
            open_ends[t["cls"]] = [(e, g) for e, g in open_ends[t["cls"]] if g != best[1]]
            open_ends[t["cls"]].append((t["pts"][-1][0], target))
        else:
            open_ends[t["cls"]].append((t["pts"][-1][0], gid))

    n_before = len(tracks)
    for fk, dets in data["frames"].items():
        for d in dets:
            d["global_id"] = resolve(d["global_id"])
    n_after = len({d["global_id"] for dets in data["frames"].values() for d in dets})

    with open(path, "w") as f:
        json.dump(data, f)
    print(f"[stitch] {args.scene}: {n_before} -> {n_after} global ids "
          f"({len(merged_into)} merges) -> {path}")


if __name__ == "__main__":
    main()
