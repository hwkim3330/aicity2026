"""
fuse_mtmc.py -- cross-camera identity fusion for the MTMC baseline.

v2: geometric distance + ReID appearance cost (see extract_reid_embeds.py).
  For each frame, gather every camera's 3D-projected detections. Two detections from
  DIFFERENT cameras in the SAME frame are considered a candidate match if:
    - same target_class,
    - Euclidean distance between (x,y) world positions <= DIST_THRESH meters, and
    - (if embeddings are available for both tracks) cosine appearance distance
      <= REID_DIST_THRESH.
  Cost = ALPHA_GEOM * geom_dist + ALPHA_REID * (1 - cos_sim), falling back to
  pure geometry when no embedding exists for one/both sides (e.g. extraction
  wasn't run for that scene). This is the fix for the fragmentation seen in a
  local proxy eval on Warehouse_000 (a local proxy IDF1 of only ~38-51%
  depending on gating distance, with detection counts already close to GT --
  the problem is cross-camera identity linking, not detection quality) and
  matches what public top Track 1 solutions do (e.g. the AIC24 runner-up,
  github.com/riips/AIC24_Track1_YACHIYO_RIIPS, fuses BoT-SORT tracking with
  deep-person-reid features).
  Candidate matches are solved frame-by-frame as a bipartite assignment (Hungarian,
  via `lap`) between every camera pair, greedily merged. A Union-Find (disjoint set)
  over (camera, local_track_id) keys accumulates these per-frame matches over the
  whole video, so two per-camera tracks that were matched in ANY frame end up under
  the same global object_id.

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
REID_DIST_THRESH = 0.4   # cosine distance (1 - cos_sim) gate when embeddings exist
ALPHA_GEOM = 1.0
ALPHA_REID = 2.0          # weighted higher: appearance is the more discriminative
                          # signal once two candidates are already within DIST_THRESH_M


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


def load_reid_embeds(scene, cameras, reid_dir="cache/reid_embeds"):
    """Returns {camera: {track_id: np.array}}. Missing files -> empty dict for
    that camera (pair_cost falls back to pure geometry)."""
    out = {}
    for cam in cameras:
        path = os.path.join(reid_dir, f"{scene}__{cam}.json")
        if os.path.isfile(path):
            with open(path) as f:
                raw = json.load(f)
            out[cam] = {int(k): np.array(v) for k, v in raw.items()}
        else:
            out[cam] = {}
    return out


def pair_cost(det_a, det_b, embed_a=None, embed_b=None):
    """Geometric distance, blended with ReID appearance cosine distance when
    embeddings are available for both sides (see extract_reid_embeds.py)."""
    if det_a["target_class"] != det_b["target_class"]:
        return None
    d = np.hypot(det_a["x"] - det_b["x"], det_a["y"] - det_b["y"])
    if d > DIST_THRESH_M:
        return None
    if embed_a is None or embed_b is None:
        return ALPHA_GEOM * d
    cos_dist = 1.0 - float(np.dot(embed_a, embed_b))
    if cos_dist > REID_DIST_THRESH:
        return None
    return ALPHA_GEOM * d + ALPHA_REID * cos_dist


def match_frame_across_cameras(frame_dets_by_cam, uf, reid_embeds=None):
    """frame_dets_by_cam: {camera: [detection dicts]}. Updates uf in-place with matches."""
    cams = list(frame_dets_by_cam.keys())
    max_cost = DIST_THRESH_M + ALPHA_REID  # loosest possible cost still admissible
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            ca, cb = cams[i], cams[j]
            da, db = frame_dets_by_cam[ca], frame_dets_by_cam[cb]
            if not da or not db:
                continue
            embeds_a = reid_embeds.get(ca, {}) if reid_embeds else {}
            embeds_b = reid_embeds.get(cb, {}) if reid_embeds else {}
            cost = np.full((len(da), len(db)), max_cost + 1.0)
            for ii, dda in enumerate(da):
                ea = embeds_a.get(dda["track_id"])
                for jj, ddb in enumerate(db):
                    eb = embeds_b.get(ddb["track_id"])
                    c = pair_cost(dda, ddb, ea, eb)
                    if c is not None:
                        cost[ii, jj] = c
            _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=max_cost)
            for ii, jj in enumerate(x):
                if jj >= 0 and cost[ii, jj] <= max_cost:
                    key_a = (ca, da[ii]["track_id"])
                    key_b = (cb, db[jj]["track_id"])
                    uf.union(key_a, key_b)


def fuse_scene(scene, tracks3d_dir="cache/tracks3d", out_dir="cache/global_tracks",
               reid_dir="cache/reid_embeds"):
    cams_data = load_all_camera_tracks(scene, tracks3d_dir)
    if not cams_data:
        raise RuntimeError(f"no tracks3d found for scene {scene} in {tracks3d_dir}")

    reid_embeds = load_reid_embeds(scene, cams_data.keys(), reid_dir)
    n_embedded = sum(len(v) for v in reid_embeds.values())
    print(f"[fuse_mtmc] {scene}: {n_embedded} ReID-embedded tracks across {len(cams_data)} cams"
          + (" (none -- falling back to pure geometry)" if n_embedded == 0 else ""))

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
            match_frame_across_cameras(frame_dets_by_cam, uf, reid_embeds)

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
