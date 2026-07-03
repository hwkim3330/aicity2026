#!/usr/bin/env python3
"""
Inference / submission-file generator for Track 6.

Runs the fine-tuned model over a directory of test images and writes
predictions in the AI City Challenge evaluation format.

CONFIRMED WORKFLOW (from the public Track 6 page): benchmarking is a
two-step process --
  (1) This script runs INSIDE the Hafnia platform against the hidden
      benchmark set (Hafnia's convention: scripts/benchmark.py, paired
      with scripts/benchmark.schema.json for its CLI args -- see
      docs/benchmark.md in github.com/milestone-hafnia/hafnia). Hafnia's
      own SDK represents predictions as "hafnia primitives"
      (`Bbox(..., confidence=..., ground_truth=False)`) attached to the
      dataset's `/predictions` task -- if the `hafnia` SDK is importable,
      prefer emitting via that representation so Hafnia's own result
      collection recognizes it (see write_hafnia_predictions() below).
  (2) The participant then manually DOWNLOADS the generated prediction
      file(s) from the Hafnia platform and uploads them to the separate
      official AI City evaluation portal: https://eval.aicitychallenge.org/aicity2026/
      (login-gated; exact upload file schema for Track 6 not confirmed
      publicly -- see README.md "Remaining steps"). The plain-JSON
      COCO-ish writer below (write_submission()) is kept as a
      framework-agnostic fallback/local-inspection format and should be
      ADAPTED once the eval-portal's exact expected schema is visible
      after logging in.

Also supports simple ensembling (multiple --weights) via NMS-fused boxes,
executed as a single process/pipeline so it complies with the "ensembles
must run as a single inference pipeline" rule.
"""
import argparse
import json
import os
from pathlib import Path

CLASS_NAMES = [
    "Vehicle.Car", "Pickup Truck", "Single Truck", "Combo Truck",
    "Heavy Duty Vehicle", "Trailer", "Motorcycle", "Bicycle", "Van", "Person",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", required=True,
                    help="One or more .pt checkpoints; >1 triggers ensemble mode.")
    p.add_argument("--images", type=str, required=True, help="Dir of test images")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU, also used for ensemble box fusion")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--out", type=str, default="submission.json")
    return p.parse_args()


def run_single(weight, images_dir, imgsz, conf, iou, device):
    from ultralytics import YOLO
    model = YOLO(weight)
    results = model.predict(source=images_dir, imgsz=imgsz, conf=conf, iou=iou,
                             device=device, stream=True, verbose=False)
    per_image = {}
    for r in results:
        stem = Path(r.path).stem
        boxes = []
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            boxes.append({
                "category_id": int(b.cls.item()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(b.conf.item()),
            })
        per_image[stem] = boxes
    return per_image


def fuse_ensemble(per_model_results, iou_thresh):
    """Simple weighted-box-fusion-lite: concatenate all models' boxes per
    image, then greedy-NMS across the union so the ensemble still emits a
    single deduplicated detection list (keeps this a single inference
    pipeline as required by the Track 6 rules)."""
    import numpy as np

    def nms(boxes):
        if not boxes:
            return []
        boxes_sorted = sorted(boxes, key=lambda b: -b["score"])
        kept = []

        def iou(a, b):
            ax1, ay1, aw, ah = a["bbox"]; ax2, ay2 = ax1 + aw, ay1 + ah
            bx1, by1, bw, bh = b["bbox"]; bx2, by2 = bx1 + bw, by1 + bh
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0

        for b in boxes_sorted:
            if all(iou(b, k) < iou_thresh or b["category_id"] != k["category_id"] for k in kept):
                kept.append(b)
        return kept

    all_images = set()
    for res in per_model_results:
        all_images.update(res.keys())
    fused = {}
    for img in all_images:
        combined = []
        for res in per_model_results:
            combined.extend(res.get(img, []))
        fused[img] = nms(combined)
    return fused


def write_submission(per_image, out_path):
    """Framework-agnostic fallback writer (COCO-ish). See module docstring:
    the real AI City eval-portal schema for Track 6 is not confirmed
    publicly -- inspect the portal after login and adapt this function."""
    records = []
    for image_id, boxes in per_image.items():
        for b in boxes:
            records.append({
                "image_id": image_id,
                "category_id": b["category_id"],
                "category_name": CLASS_NAMES[b["category_id"]],
                "bbox": [round(v, 2) for v in b["bbox"]],
                "score": round(b["score"], 4),
            })
    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"[benchmark.py] wrote {len(records)} detections across {len(per_image)} images -> {out_path}")


def write_hafnia_predictions(per_image, dataset_name, dataset_version):
    """Best-effort writer using the hafnia SDK's own prediction primitives,
    so that when this script runs inside a real Hafnia benchmarking job
    the platform's own result-collection recognizes the output without a
    separate conversion step. Falls back silently (caller should also
    call write_submission()) if the SDK isn't importable, e.g. during a
    local/offline dry run.

    NOT YET VERIFIED end-to-end against a live Hafnia job -- the hafnia
    SDK's exact `Bbox`/dataset-attach API should be reconfirmed against
    docs/benchmark.md in github.com/milestone-hafnia/hafnia once logged
    in, then this function adjusted accordingly.
    """
    try:
        from hafnia.dataset import HafniaDataset
        from hafnia.data.primitives import Bbox  # name unconfirmed, adjust after login
    except ImportError:
        print("[benchmark.py] hafnia SDK not importable, skipping hafnia-native "
              "prediction export (use write_submission() output instead)")
        return None

    ds = HafniaDataset.from_name(dataset_name, version=dataset_version)
    predictions = []
    for image_id, boxes in per_image.items():
        for b in boxes:
            x, y, w, h = b["bbox"]
            predictions.append(Bbox(
                image_id=image_id,
                top_left_x=x, top_left_y=y, width=w, height=h,
                class_name=CLASS_NAMES[b["category_id"]],
                confidence=b["score"],
                ground_truth=False,
            ))
    # Exact attach-to-dataset call unconfirmed; placeholder:
    # ds.attach_predictions(predictions, task="predictions")
    print(f"[benchmark.py] built {len(predictions)} hafnia-native prediction primitives "
          f"(attach-to-dataset call is a placeholder, verify after login)")
    return predictions


def main():
    args = parse_args()
    per_model = [run_single(w, args.images, args.imgsz, args.conf, args.iou, args.device)
                 for w in args.weights]
    fused = per_model[0] if len(per_model) == 1 else fuse_ensemble(per_model, args.iou)
    write_submission(fused, args.out)
    write_hafnia_predictions(fused, dataset_name=os.environ.get("TRACK6_DATASET_NAME", ""),
                              dataset_version=os.environ.get("TRACK6_DATASET_VERSION", "1.0.0"))


if __name__ == "__main__":
    main()
