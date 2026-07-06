"""
eval_track1.py -- local proxy evaluation of a track1.txt submission against
ground_truth.json, using 3D center-distance matching (MOTA/IDF1/MOTP via
motmetrics). This is NOT the official 3D-HOTA metric used by the challenge
server, but tracks the same signal (detection + association quality) well
enough to sanity-check changes before submitting.

Usage:
    python3 eval_track1.py --scene Warehouse_000 --pred track1.txt --max-dist 2.0
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import motmetrics as mm

from common import load_ground_truth, scene_dir
import os


def load_pred(path, scene_id):
    """frame_id -> list of (object_id, x, y, z)"""
    by_frame = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split()
            sid, cls_id, obj_id, frame_id = map(int, p[:4])
            if sid != scene_id:
                continue
            x, y, z = map(float, p[4:7])
            by_frame.setdefault(frame_id, []).append((obj_id, x, y, z))
    return by_frame


def load_gt(scene, split="train"):
    """frame_id -> list of (object_id, x, y, z)"""
    gt = load_ground_truth(scene, split)
    by_frame = {}
    for fk, objs in gt.items():
        fid = int(fk)
        for o in objs:
            x, y, z = o["3d location"]
            by_frame.setdefault(fid, []).append((o["object id"], x, y, z))
    return by_frame


def scene_to_id(scene_name):
    digits = "".join(c for c in scene_name.split("_")[-1] if c.isdigit())
    return int(digits) if digits else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-dist", type=float, default=2.0, help="meters, gating distance")
    ap.add_argument("--max-frames", type=int, default=None,
                     help="Only score frames [0, max_frames) -- for comparing a partial-run "
                          "prediction fairly against a full-length GT (else every frame beyond "
                          "the prediction's range counts as 100%% missed).")
    args = ap.parse_args()

    scene_id = scene_to_id(args.scene)
    gt_by_frame = load_gt(args.scene, args.split)
    pred_by_frame = load_pred(args.pred, scene_id)
    if args.max_frames is not None:
        gt_by_frame = {f: v for f, v in gt_by_frame.items() if f < args.max_frames}
        pred_by_frame = {f: v for f, v in pred_by_frame.items() if f < args.max_frames}

    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    for fid in frames:
        gt = gt_by_frame.get(fid, [])
        pr = pred_by_frame.get(fid, [])
        gt_ids = [g[0] for g in gt]
        pr_ids = [p[0] for p in pr]
        if gt and pr:
            gt_xy = np.array([[g[1], g[2]] for g in gt])
            pr_xy = np.array([[p[1], p[2]] for p in pr])
            dists = mm.distances.norm2squared_matrix(gt_xy, pr_xy, max_d2=args.max_dist ** 2)
            dists = np.sqrt(dists)
        else:
            dists = np.empty((len(gt_ids), len(pr_ids)))
        acc.update(gt_ids, pr_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["num_frames", "mota", "motp", "idf1", "idp", "idr",
                 "num_switches", "num_false_positives", "num_misses",
                 "num_objects", "num_predictions"],
        name=args.scene,
    )
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))


if __name__ == "__main__":
    main()
