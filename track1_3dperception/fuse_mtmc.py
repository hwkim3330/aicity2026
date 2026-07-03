"""
fuse_mtmc.py -- cross-camera identity fusion for the MTMC baseline.

v1 approach (no ReID -- documented limitation, see README.md):
  For each frame, gather every camera's 3D-projected detections. Two detections from
  DIFFERENT cameras in the SAME frame are considered a candidate match if:
    - same target_class, and
    - Euclidean distance between (x,y) world positions <= DIST_THRESH meters.
  Candidate matches are solved frame-by-frame as a bipartite assignment (Hungarian,
  via `lap`) between every camera pair, greedily merged. A Union-Find (disjoint set)
  over (camera, local_track_id) keys accumulates these per-frame matches over the
  whole video, so two per-camera tracks that were matched in ANY frame end up under
  the same global object_id.

  This is intentionally simple. Code is structured so a ReID embedding distance can
  be added as an extra edge-weight term later (see `pair_cost()` -- currently pure
  geometric distance, would become `alpha*dist + beta*(1-cos_sim(embed_a,embed_b))`).

Output: cache/global_tracks/<scene>.json
{
  "scene": ...,
  "frames": {"<frame_idx>": [{"global_id":int, "camera":str, "track_id":int,
      "target_class":str, "x":..,"y":..,"z":..,"w":..,"l":..,"h":..,"yaw":0.0}, ...]}
}
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import lap

DIST_THRESH_M = 2.0  # meters; two detections in the same frame within this distance
                      # (and same class) from different cameras are considered the
                      # same physical object.


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            x, self.parent[x] = self.parent[x], self.parent[self.parent[x]]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def load_all_camera_tracks(scene, tracks3d_dir):
    """Returns dict: camera -> loaded json data."""
    cams = {}
    for fn in os.listdir(tracks3d_dir):
        if fn.startswith(scene + "__") and fn.endswith(".json"):
            cam = fn[len(scene) + 2:-5]
            with open(os.path.join(tracks3d_dir, fn)) as f:
                cams[cam] = json.load(f)
    return cams


def pair_cost(det_a, det_b):
    """Geometric-only cost; hook point for adding ReID embedding distance later."""
    if det_a["target_class"] != det_b["target_class"]:
        return None
    d = np.hypot(det_a["x"] - det_b["x"], det_a["y"] - det_b["y"])
    if d > DIST_THRESH_M:
        return None
    return d


def match_frame_across_cameras(frame_dets_by_cam, uf):
    """frame_dets_by_cam: {camera: [detection dicts]}. Updates uf in-place with matches."""
    cams = list(frame_dets_by_cam.keys())
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            ca, cb = cams[i], cams[j]
            da, db = frame_dets_by_cam[ca], frame_dets_by_cam[cb]
            if not da or not db:
                continue
            cost = np.full((len(da), len(db)), DIST_THRESH_M + 1.0)
            for ii, dda in enumerate(da):
                for jj, ddb in enumerate(db):
                    c = pair_cost(dda, ddb)
                    if c is not None:
                        cost[ii, jj] = c
            _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=DIST_THRESH_M)
            for ii, jj in enumerate(x):
                if jj >= 0 and cost[ii, jj] <= DIST_THRESH_M:
                    key_a = (ca, da[ii]["track_id"])
                    key_b = (cb, db[jj]["track_id"])
                    uf.union(key_a, key_b)


def fuse_scene(scene, tracks3d_dir="cache/tracks3d", out_dir="cache/global_tracks"):
    cams_data = load_all_camera_tracks(scene, tracks3d_dir)
    if not cams_data:
        raise RuntimeError(f"no tracks3d found for scene {scene} in {tracks3d_dir}")

    # collect union of frame keys
    all_frame_keys = set()
    for data in cams_data.values():
        all_frame_keys.update(data["frames"].keys())
    all_frame_keys = sorted(all_frame_keys, key=lambda k: int(k))

    uf = UnionFind()
    # register every (camera, track_id) that appears, even if never matched
    for cam, data in cams_data.items():
        for fk, dets in data["frames"].items():
            for d in dets:
                uf.find((cam, d["track_id"]))

    for fk in all_frame_keys:
        frame_dets_by_cam = {}
        for cam, data in cams_data.items():
            dets = data["frames"].get(fk, [])
            if dets:
                frame_dets_by_cam[cam] = dets
        if len(frame_dets_by_cam) >= 2:
            match_frame_across_cameras(frame_dets_by_cam, uf)

    # assign compact global ids
    root_to_gid = {}
    next_gid = 1

    def gid_for(key):
        nonlocal next_gid
        r = uf.find(key)
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        return root_to_gid[r]

    out_frames = defaultdict(list)
    for cam, data in cams_data.items():
        for fk, dets in data["frames"].items():
            for d in dets:
                gid = gid_for((cam, d["track_id"]))
                out_frames[fk].append({
                    "global_id": gid,
                    "camera": cam,
                    "track_id": d["track_id"],
                    "target_class": d["target_class"],
                    "x": d["x"], "y": d["y"], "z": d["z"],
                    "w": d["w"], "l": d["l"], "h": d["h"],
                    "yaw": d["yaw"],
                })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}.json")
    with open(out_path, "w") as f:
        json.dump({"scene": scene, "frames": dict(out_frames)}, f)
    n_global = len(root_to_gid)
    print(f"[fuse_mtmc] {scene}: {len(all_frame_keys)} frames, "
          f"{len(cams_data)} cams, {n_global} global ids")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tracks3d-dir", default="cache/tracks3d")
    ap.add_argument("--out-dir", default="cache/global_tracks")
    ap.add_argument("--dist-thresh", type=float, default=DIST_THRESH_M)
    args = ap.parse_args()
    globals()["DIST_THRESH_M"] = args.dist_thresh
    fuse_scene(args.scene, args.tracks3d_dir, args.out_dir)


if __name__ == "__main__":
    main()
