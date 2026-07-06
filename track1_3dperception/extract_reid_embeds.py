"""extract_reid_embeds.py -- appearance (ReID-style) embeddings per 2D track,
to fix the single biggest gap vs top Track 1 teams: our MTMC fusion
(fuse_mtmc.py) is purely geometric, while every public top solution (e.g.
AIC24 runner-up github.com/riips/AIC24_Track1_YACHIYO_RIIPS) fuses geometry
with Re-ID appearance features. A local proxy eval on Warehouse_000 already
confirmed our bottleneck is track fragmentation/mismatches, not detection
recall -- this directly targets that.

Uses CLIP's image encoder (already a dependency via track4_reid, and "good
enough" as a generic appearance embedding without needing a dedicated
person-ReID model/training data) rather than installing torchreid.

For each (scene, camera, track_id), samples up to N_SAMPLES frames spread
across the track's lifetime, crops the bbox, embeds each crop, and averages
-> one L2-normalized embedding per track. Output:
  cache/reid_embeds/<scene>__<camera>.json  {"<track_id>": [float, ...], ...}

Usage:
    python3 extract_reid_embeds.py --scene Warehouse_000 [--cameras Camera_0000 ...]
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from common import video_path, list_cameras

N_SAMPLES = 3
MODEL_NAME = "openai/clip-vit-large-patch14"  # same as track4_reid -- has safetensors weights
                                              # (clip-vit-base-patch32 doesn't, and current
                                              # transformers refuses non-safetensors .bin loads)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def sample_frames_for_track(frames_dict, track_id, n=N_SAMPLES):
    have = sorted(int(fk) for fk, dets in frames_dict.items()
                  if any(d["track_id"] == track_id for d in dets))
    if not have:
        return []
    if len(have) <= n:
        return have
    idxs = np.linspace(0, len(have) - 1, n).round().astype(int)
    return [have[i] for i in idxs]


@torch.no_grad()
def embed_crops(crops):
    if not crops:
        return None
    model, processor = get_model()
    inputs = processor(images=crops, return_tensors="pt").to(DEVICE)
    feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.mean(dim=0)


def process_camera(scene, camera, tracks2d_dir="cache/tracks2d", out_dir="cache/reid_embeds"):
    path = os.path.join(tracks2d_dir, f"{scene}__{camera}.json")
    if not os.path.isfile(path):
        print(f"[reid] no tracks2d for {scene}/{camera}, skipping")
        return
    with open(path) as f:
        data = json.load(f)
    frames = data["frames"]

    track_ids = sorted({d["track_id"] for dets in frames.values() for d in dets})
    frames_needed = {}
    for tid in track_ids:
        for fk in sample_frames_for_track(frames, tid):
            frames_needed.setdefault(fk, []).append(tid)

    vpath = video_path(scene, camera, "train")
    if not os.path.isfile(vpath):
        vpath = video_path(scene, camera, "test")
    cap = cv2.VideoCapture(vpath)

    crops_by_track = {tid: [] for tid in track_ids}
    idx = 0
    while frames_needed:
        ok, img = cap.read()
        if not ok:
            break
        if idx in frames_needed:  # frames_needed is keyed by int frame index
            fk = str(idx)         # frames dict (from JSON) is keyed by str
            for tid in frames_needed.pop(idx):
                dets = [d for d in frames[fk] if d["track_id"] == tid]
                if dets:
                    x1, y1, x2, y2 = [int(v) for v in dets[0]["bbox"]]
                    x1, y1 = max(0, x1), max(0, y1)
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops_by_track[tid].append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    embeds = {}
    for tid, crops in crops_by_track.items():
        e = embed_crops(crops)
        if e is not None:
            embeds[str(tid)] = e.cpu().tolist()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}__{camera}.json")
    with open(out_path, "w") as f:
        json.dump(embeds, f)
    print(f"[reid] {scene}/{camera}: {len(embeds)}/{len(track_ids)} tracks embedded -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--cameras", nargs="*", default=None)
    args = ap.parse_args()

    cameras = args.cameras or list_cameras(args.scene, args.split)
    for cam in cameras:
        process_camera(args.scene, cam)


if __name__ == "__main__":
    main()
