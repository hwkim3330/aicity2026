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
    "/opt/ml/model/checkpoint_best_total.pth",
    "/opt/ml/checkpoints/checkpoint_best_total.pth",
    "runs/rfdetr_run/checkpoint_best_total.pth",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", default=None,
                    help="One or more checkpoints; >1 = ensemble. "
                         "Default: search WEIGHT_SEARCH_PATHS.")
    p.add_argument("--model-type", choices=["yolo", "rfdetr"], default="yolo",
                    help="Which library to load --weights with. All weights "
                         "in an ensemble must be the same type.")
    p.add_argument("--rfdetr-model-size", choices=["base", "large"], default="base",
                    help="Only used when --model-type rfdetr.")
    p.add_argument("--rfdetr-resolution", type=int, default=560,
                    help="Only used when --model-type rfdetr -- must match "
                         "the resolution the checkpoint was trained at.")
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
    # --- logdump: print the submission through stdout logs (the ONLY channel
    # retrievable from Hafnia after the job ends -- see scripts/logdump.py).
    # chunk-chars=4000 is conservative; raise to 8000 if log_probe.py showed
    # long lines survive intact (4000 needs >1000 log entries, which exceeds
    # the observed per-request cap unless pagination works).
    p.add_argument("--logdump-chunk-chars", type=int, default=4000)
    p.add_argument("--logdump-codec", choices=["lzma", "gzip"], default="lzma")
    p.add_argument("--logdump-sleep-ms", type=int, default=15)
    p.add_argument("--logdump-bbox-decimals", type=int, default=1)
    p.add_argument("--logdump-score-decimals", type=int, default=3)
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


def run_single_rfdetr(weight, source, model_size, resolution, conf, device):
    """RF-DETR equivalent of run_single(). Was entirely missing before --
    train_rfdetr.py (added because the real leaderboard shows RF-DETR
    beating YOLO11) had NO way to turn its trained checkpoint into a
    submission: this function and --model-type rfdetr are what's needed to
    actually use that checkpoint, not just produce it."""
    from rfdetr import RFDETRBase, RFDETRLarge
    ModelClass = RFDETRLarge if model_size == "large" else RFDETRBase
    # RF-DETR/pydantic requires a torch-style device spec ("cuda:0"), unlike
    # ultralytics YOLO which accepts a bare index ("0") -- normalize here
    # rather than changing the shared --device default used by run_single().
    rfdetr_device = f"cuda:{device}" if device.isdigit() else device
    model = ModelClass(resolution=resolution, pretrain_weights=str(weight), device=rfdetr_device)

    image_paths = source if isinstance(source, list) else sorted(
        str(p) for p in Path(source).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    per_image = {}
    for img_path in image_paths:
        stem = Path(img_path).stem
        dets = model.predict(img_path, threshold=conf)
        boxes = []
        for i in range(len(dets.xyxy)):
            x1, y1, x2, y2 = dets.xyxy[i].tolist()
            boxes.append({
                "category_id": int(dets.class_id[i]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(dets.confidence[i]),
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

    if args.model_type == "rfdetr":
        per_model = [run_single_rfdetr(w, source, args.rfdetr_model_size,
                                        args.rfdetr_resolution, args.conf, args.device)
                     for w in weights]
    else:
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

    # ------------------------------------------------------------------
    # Dump the submission through stdout: the artifact dir is NOT actually
    # downloadable from Hafnia (confirmed: no CLI/SDK/REST endpoint), so the
    # /experiments/{id}/logs channel is the only way to ever see this file.
    # This MUST be the last thing printed (the logs endpoint reliably returns
    # the newest ~1000 entries via ordering=-created_at). A failure here must
    # never crash the job -- the artifact copy still exists as a hail-mary.
    # Reassemble locally: scripts/logdump_client.py reassemble ...
    # ------------------------------------------------------------------
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import logdump
        payload = logdump.compact_payload(
            fused, id_of, class_names,
            bbox_decimals=args.logdump_bbox_decimals,
            score_decimals=args.logdump_score_decimals)
        n_chunks = logdump.dump_bytes(
            payload, tag="submission",
            chunk_chars=args.logdump_chunk_chars,
            codec=args.logdump_codec,
            sleep_ms=args.logdump_sleep_ms)
        print(f"[benchmark.py] logdump complete: {n_chunks} chunks of "
              f"{args.logdump_chunk_chars} chars ({args.logdump_codec}+b64, "
              f"payload {len(payload)} raw bytes)", flush=True)
    except Exception as e:  # noqa: BLE001 -- never lose the run to the dump
        print(f"[benchmark.py] LOGDUMP FAILED ({e!r}) -- submission.json still "
              f"written to {out_artifact}", file=sys.stderr, flush=True)

    logger.end_run()


if __name__ == "__main__":
    main()
