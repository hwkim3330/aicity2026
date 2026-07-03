"""
project3d.py -- backproject per-camera 2D tracks into 3D world coordinates.

Method (baseline, monocular, no depth):
  1. Take the bottom-center pixel of the 2D bbox: ((x1+x2)/2, y2).
     This approximates the object's ground-contact point.
  2. Apply the camera's ground-plane homography inverse to get world (X, Y) at Z=0.
     (Verified against ground_truth.json: bottom of 3D box at Z~0 projects to
     bottom-center-ish of the reported 2D bbox via cameraMatrix -- see common.py docstring
     and README.md for the concrete numeric check.)
  3. "3d location" in ground_truth.json is the BOX CENTER, so we report location.z =
     class_height_prior / 2 (box center height), not 0. width/length/height come from
     common.CLASS_SIZE_PRIOR (per-class average measured from ground_truth.json).
  4. yaw = 0.0 baseline (documented limitation -- no heading estimation in v1).

Output: cache/tracks3d/<scene>__<camera>.json
{
  "camera":..., "scene":...,
  "frames": {"<frame_idx>": [{"track_id":int, "target_class":str, "conf":float,
     "x":..,"y":..,"z":.., "w":..,"l":..,"h":..,"yaw":0.0, "bbox":[..]}, ...]}
}
"""
import argparse
import json
import os

from common import CLASS_SIZE_PRIOR, load_camera_models


def project_camera_tracks(scene, camera, tracks_path, cam_model, out_dir="cache/tracks3d"):
    with open(tracks_path) as f:
        data = json.load(f)

    out_frames = {}
    for fk, dets in data["frames"].items():
        out_dets = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            u = (x1 + x2) / 2.0
            v = y2  # bottom-center
            X, Y = cam_model.pixel_to_ground(u, v)
            w, l, h = CLASS_SIZE_PRIOR.get(d["target_class"], (0.6, 0.6, 1.0))
            out_dets.append({
                "track_id": d["track_id"],
                "target_class": d["target_class"],
                "conf": d["conf"],
                "x": round(float(X), 2),
                "y": round(float(Y), 2),
                "z": round(float(h) / 2.0, 2),
                "w": round(float(w), 2),
                "l": round(float(l), 2),
                "h": round(float(h), 2),
                "yaw": 0.0,
                "bbox": d["bbox"],
            })
        out_frames[fk] = out_dets

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}__{camera}.json")
    with open(out_path, "w") as f:
        json.dump({"camera": camera, "scene": scene, "frames": out_frames}, f)
    print(f"[project3d] {scene}/{camera}: projected {len(out_frames)} frames")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--tracks-dir", default="cache/tracks2d")
    ap.add_argument("--out-dir", default="cache/tracks3d")
    args = ap.parse_args()

    cam_models = load_camera_models(args.scene, args.split)

    if args.cameras is None:
        cams = []
        for fn in os.listdir(args.tracks_dir):
            if fn.startswith(args.scene + "__") and fn.endswith(".json"):
                cams.append(fn[len(args.scene) + 2:-5])
        args.cameras = sorted(cams)

    for cam in args.cameras:
        tpath = os.path.join(args.tracks_dir, f"{args.scene}__{cam}.json")
        project_camera_tracks(args.scene, cam, tpath, cam_models[cam], args.out_dir)


if __name__ == "__main__":
    main()
