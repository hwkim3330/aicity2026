#!/usr/bin/env python3
"""
Inference / submission-file generator for Track 6 (Hafnia benchmark job).

Applies every platform lesson learned the hard way during the training
shakeouts (v1-v7):
  - no internet on instances: weights must be local (bundled or trained)
  - dataset arrives as a mounted HafniaDataset (annotations + data/), not a
    raw image dir
  - HafniaLogger must register the job with the platform's tracking or the
    job is killed at ~6 minutes
  - outputs must land in the collected artifact dir (/opt/ml/output/data)
    to be downloadable after the job

Two-step Track 6 workflow: this runs inside Hafnia against the hidden
benchmark set; the participant then downloads the prediction JSON from the
platform and uploads it to eval.aicitychallenge.org.

Usage (cloud benchmark job):
    python scripts/benchmark.py --weights /opt/ml/model/best.pt
Usage (local, sample dataset):
    python scripts/benchmark.py --weights runs/main/weights/best.pt --split test
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# fallback only -- real order is derived from the mounted dataset's class_idx
FALLBACK_CLASS_NAMES = [
    "Vehicle.Car", "Vehicle.Pickup Truck", "Vehicle.Single Truck",
    "Vehicle.Combo Truck", "Vehicle.Heavy Duty Vehicle", "Vehicle.Trailer",
    "Vehicle.Motorcycle", "Vehicle.Bicycle", "Vehicle.Van", "Person",
]

# where a trained checkpoint might live, in preference order (the exact
# convention Hafnia uses to hand a trained model to a benchmark job is
# undocumented -- search broadly and log what was found)
WEIGHT_SEARCH_PATHS = [
    "/opt/ml/model/best.pt",
    "/opt/ml/checkpoints/best.pt",
    "/opt/ml/input/model/best.pt",
    "runs/main/weights/best.pt",
    "runs/track6_run/weights/best.pt",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", default=None,
                    help="One or more .pt checkpoints; >1 = ensemble. "
                         "Default: search WEIGHT_SEARCH_PATHS.")
    p.add_argument("--images", type=str, default=None,
                    help="Optional explicit dir of test images; default = "
                         "resolve from the mounted Hafnia dataset.")
    p.add_argument("--split", type=str, default="test",
                    help="Dataset split to run on when resolving from the "
                         "mounted dataset ('all' = every sample).")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", type=str, default=os.environ.get("HAFNIA_DEVICE", "0"))
    p.add_argument("--out", type=str, default="submission.json")
    return p.parse_args()


def get_mount():
    from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job
    if is_hafnia_cloud_job():
        return Path(get_dataset_path_in_hafnia_cloud())
    # local fallback: the downloaded sample dataset
    from hafnia.utils import PATH_DATASETS
    return Path(PATH_DATASETS) / os.environ.get("TRACK6_DATASET_NAME", "eccv-cross-city")


def load_records(mount):
    ann = mount / "annotations.jsonl"
    if ann.exists():
        return [json.loads(l) for l in ann.read_text().splitlines() if l.strip()]
    from hafnia.dataset.hafnia_dataset import HafniaDataset
    return list(HafniaDataset.from_path(mount).samples.iter_rows(named=True))


def resolve_images_and_classes(args, mount):
    records = load_records(mount)
    class_names = {}
    for r in records:
        for b in (r.get("bboxes") or []):
            if b.get("class_name") is not None and b.get("class_idx") is not None:
                class_names[int(b["class_idx"])] = b["class_name"]
    nc = max(class_names) + 1 if class_names else len(FALLBACK_CLASS_NAMES)
    names = [class_names.get(i, FALLBACK_CLASS_NAMES[i] if i < len(FALLBACK_CLASS_NAMES) else f"class_{i}")
             for i in range(nc)]

    if args.images:
        return args.images, names, None

    wanted = None if args.split == "all" else args.split
    image_paths = []
    id_of = {}
    for r in records:
        if wanted and r.get("split") != wanted:
            continue
        fp = Path(r["file_path"])
        img = fp if fp.is_absolute() else mount / fp
        image_paths.append(str(img))
        id_of[img.stem] = {
            "file_path": r.get("file_path"),
            "sample_index": r.get("sample_index"),
            "remote_path": r.get("remote_path"),
        }
    print(f"[benchmark.py] resolved {len(image_paths)} images (split={args.split}) "
          f"from {mount}, {len(names)} classes", file=sys.stderr)
    return image_paths, names, id_of


def find_weights(args):
    if args.weights:
        return args.weights
    for cand in WEIGHT_SEARCH_PATHS:
        if Path(cand).exists():
            print(f"[benchmark.py] auto-discovered weights: {cand}", file=sys.stderr)
            return [cand]
    tried = "\n  ".join(WEIGHT_SEARCH_PATHS)
    raise FileNotFoundError(
        f"No --weights given and none of the default checkpoint locations exist:\n  {tried}"
    )


def run_single(weight, source, imgsz, conf, iou, device):
    from ultralytics import YOLO
    model = YOLO(weight)
    results = model.predict(source=source, imgsz=imgsz, conf=conf, iou=iou,
                             device=device, stream=True, verbose=False)
    per_image = {}
    for r in results:
        stem = Path(r.path).stem
        boxes = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append({
                "category_id": int(b.cls.item()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(b.conf.item()),
            })
        per_image[stem] = boxes
    return per_image


def fuse_ensemble(per_model_results, iou_thresh):
    """Greedy cross-model NMS so an ensemble still emits one deduplicated
    detection list (single-inference-pipeline rule)."""
    def iou(a, b):
        ax1, ay1, aw, ah = a["bbox"]; ax2, ay2 = ax1 + aw, ay1 + ah
        bx1, by1, bw, bh = b["bbox"]; bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0

    fused = {}
    all_images = set().union(*(r.keys() for r in per_model_results))
    for img in all_images:
        combined = sorted(
            (b for res in per_model_results for b in res.get(img, [])),
            key=lambda b: -b["score"],
        )
        kept = []
        for b in combined:
            if all(b["category_id"] != k["category_id"] or iou(b, k) < iou_thresh for k in kept):
                kept.append(b)
        fused[img] = kept
    return fused


def artifact_dir():
    from hafnia.utils import is_hafnia_cloud_job
    if is_hafnia_cloud_job():
        d = Path("/opt/ml/output/data")
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(".")


def main():
    args = parse_args()

    from hafnia.experiment import HafniaLogger
    logger = HafniaLogger(project_name="track6-cross-city-benchmark")

    mount = get_mount()
    source, class_names, id_of = resolve_images_and_classes(args, mount)
    weights = find_weights(args)
    logger.log_configuration({"weights": [str(w) for w in weights],
                              "split": args.split, "imgsz": args.imgsz,
                              "conf": args.conf})

    per_model = [run_single(w, source, args.imgsz, args.conf, args.iou, args.device)
                 for w in weights]
    fused = per_model[0] if len(per_model) == 1 else fuse_ensemble(per_model, args.iou)

    records = []
    for stem, boxes in fused.items():
        meta = (id_of or {}).get(stem, {})
        for b in boxes:
            records.append({
                "image_id": stem,
                "file_path": meta.get("file_path"),
                "sample_index": meta.get("sample_index"),
                "category_id": b["category_id"],
                "category_name": class_names[b["category_id"]] if b["category_id"] < len(class_names) else str(b["category_id"]),
                "bbox": [round(v, 2) for v in b["bbox"]],
                "score": round(b["score"], 4),
            })

    out_local = Path(args.out)
    out_artifact = artifact_dir() / out_local.name
    for path in {out_local.resolve(), out_artifact.resolve()}:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f)
    n_img = len(fused)
    print(f"[benchmark.py] wrote {len(records)} detections across {n_img} images -> "
          f"{out_local} and {out_artifact}")
    logger.log_metric(name="benchmark_images", value=float(n_img), step=0)
    logger.log_metric(name="benchmark_detections", value=float(len(records)), step=0)
    logger.end_run()


if __name__ == "__main__":
    main()
