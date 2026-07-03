"""
export_submission.py -- write track1.txt from fuse_mtmc.py's global tracks.

Format (one line per detection):
  <scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <width> <length> <height> <yaw>
  - coordinates in meters, 2 decimal places
  - class_id per common.CLASS_NAME_TO_ID (0=Person .. 6=PalletTruck)
  - scene_id: numeric suffix of the Warehouse_XXX folder name (documented assumption,
    see README.md -- no separate scene-id mapping file was found in the dataset).
  - object_id = fused global_id from fuse_mtmc.py
"""
import argparse
import json
import os

from common import CLASS_NAME_TO_ID


def scene_to_id(scene_name):
    # "Warehouse_000" -> 0
    digits = "".join(c for c in scene_name.split("_")[-1] if c.isdigit())
    return int(digits) if digits else 0


def export_scene(scene, global_tracks_path, out_path="track1.txt"):
    with open(global_tracks_path) as f:
        data = json.load(f)

    scene_id = scene_to_id(scene)
    lines = []
    for fk, dets in sorted(data["frames"].items(), key=lambda kv: int(kv[0])):
        frame_id = int(fk)
        for d in dets:
            cls_id = CLASS_NAME_TO_ID.get(d["target_class"])
            if cls_id is None:
                continue
            line = (
                f"{scene_id} {cls_id} {d['global_id']} {frame_id} "
                f"{d['x']:.2f} {d['y']:.2f} {d['z']:.2f} "
                f"{d['w']:.2f} {d['l']:.2f} {d['h']:.2f} {d['yaw']:.2f}"
            )
            lines.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    validate_submission(out_path)
    return out_path


def validate_submission(path):
    n_lines = 0
    bad = 0
    class_ids = set()
    obj_ids = set()
    frame_ids = set()
    xyz_min = [1e18, 1e18, 1e18]
    xyz_max = [-1e18, -1e18, -1e18]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            parts = line.split()
            if len(parts) != 11:
                bad += 1
                continue
            try:
                scene_id, cls_id, obj_id, frame_id = map(int, parts[:4])
                floats = [float(v) for v in parts[4:]]
            except ValueError:
                bad += 1
                continue
            if any(f != f or abs(f) == float("inf") for f in floats):  # NaN/inf check
                bad += 1
                continue
            if not (0 <= cls_id <= 6) or obj_id < 0 or frame_id < 0:
                bad += 1
                continue
            class_ids.add(cls_id)
            obj_ids.add(obj_id)
            frame_ids.add(frame_id)
            for i in range(3):
                xyz_min[i] = min(xyz_min[i], floats[i])
                xyz_max[i] = max(xyz_max[i], floats[i])

    print(f"[export_submission] {path}: {n_lines} lines, {bad} malformed, "
          f"{len(class_ids)} distinct class_ids {sorted(class_ids)}, "
          f"{len(obj_ids)} distinct object_ids, {len(frame_ids)} distinct frame_ids")
    if n_lines - bad > 0:
        print(f"  x range [{xyz_min[0]:.2f}, {xyz_max[0]:.2f}]  "
              f"y range [{xyz_min[1]:.2f}, {xyz_max[1]:.2f}]  "
              f"z range [{xyz_min[2]:.2f}, {xyz_max[2]:.2f}]")
    if bad > 0:
        print(f"  WARNING: {bad} malformed lines found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--global-tracks-dir", default="cache/global_tracks")
    ap.add_argument("--out", default="track1.txt")
    args = ap.parse_args()
    gpath = os.path.join(args.global_tracks_dir, f"{args.scene}.json")
    export_scene(args.scene, gpath, args.out)


if __name__ == "__main__":
    main()
