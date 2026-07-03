"""
detect.py -- run a pretrained (COCO) YOLO model over a camera's video and cache
per-frame 2D detections to a JSON file.

Limitation (see README.md): the detector is COCO-pretrained, so only "person" is a
reliable detection; warehouse-specific classes (Forklift/PalletTruck/NovaCarter/...)
have no direct COCO analogue. We optionally map a couple of COCO vehicle classes as
*weak* proxies (see common.COCO_TO_TARGET) purely so the rest of the pipeline has
something to chew on; precision/recall for those classes will be poor. The #1 next
step to fix this is fine-tuning YOLO on ground_truth.json's 2D boxes + class labels.

Output cache format (JSON), one file per camera:
{
  "camera": "Camera_0000",
  "frames": {
     "<frame_idx>": [
        {"bbox": [x1,y1,x2,y2], "conf": 0.87, "coco_class": "person", "target_class": "Person"},
        ...
     ], ...
  }
}
"""
import argparse
import json
import os
import time

import cv2

from common import (
    COCO_TO_TARGET,
    video_path,
    list_cameras,
)


def run_detection(scene, camera, split="train", max_frames=None, device="cuda",
                   model_name="yolo11n.pt", conf_thres=0.25, out_dir="cache/detections"):
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}__{camera}.json")

    vpath = video_path(scene, camera, split)
    if not os.path.isfile(vpath):
        raise FileNotFoundError(vpath)

    model = YOLO(model_name)
    names = model.names  # id -> coco class name

    cap = cv2.VideoCapture(vpath)
    frames = {}
    idx = 0
    t0 = time.time()
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ok, img = cap.read()
        if not ok:
            break
        results = model.predict(img, device=device, conf=conf_thres, verbose=False)
        dets = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for box, c, k in zip(xyxy, conf, cls):
                coco_name = names.get(int(k), str(k))
                target = COCO_TO_TARGET.get(coco_name)
                if target is None:
                    continue  # not a class of interest -- drop
                dets.append({
                    "bbox": [round(float(v), 2) for v in box],
                    "conf": round(float(c), 4),
                    "coco_class": coco_name,
                    "target_class": target,
                })
        frames[str(idx)] = dets
        idx += 1
    cap.release()
    dt = time.time() - t0
    print(f"[detect] {scene}/{camera}: {idx} frames in {dt:.1f}s ({idx/max(dt,1e-6):.1f} fps), "
          f"total dets={sum(len(v) for v in frames.values())}")

    with open(out_path, "w") as f:
        json.dump({"camera": camera, "scene": scene, "frames": frames}, f)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--cameras", nargs="*", default=None, help="camera ids; default = all available")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out-dir", default="cache/detections")
    args = ap.parse_args()

    cameras = args.cameras or list_cameras(args.scene, args.split)
    print(f"[detect] scene={args.scene} cameras={cameras}")
    for cam in cameras:
        run_detection(args.scene, cam, args.split, args.max_frames, args.device,
                      args.model, args.conf, args.out_dir)


if __name__ == "__main__":
    main()
