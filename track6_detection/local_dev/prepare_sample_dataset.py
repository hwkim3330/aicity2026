#!/usr/bin/env python3
"""
Placeholder / scaffold for converting the Track 6 "local development
sample dataset" (announced for release 2026-05-18) into Ultralytics YOLO
format once it can actually be downloaded.

STATUS: The exact download mechanism for this sample dataset could not be
confirmed without logging into the Hafnia community portal — see
README.md "TODO (requires Hafnia login)" for what to check first
(Hafnia CLI `hafnia dataset pull <name>`? direct link on the AI City
track page? a HuggingFace/Kaggle mirror?). Once you have the raw sample
data locally, point --src at it; this script assumes a generic annotation
layout (COCO json OR per-image txt) and will need a small tweak once the
real format is confirmed.

Usage (once dataset is obtained):
    python3 prepare_sample_dataset.py --src /path/to/raw_sample --dst ./yolo_sample
"""
import argparse
import json
import shutil
from pathlib import Path

CLASS_NAMES = [
    "Vehicle.Car", "Pickup Truck", "Single Truck", "Combo Truck",
    "Heavy Duty Vehicle", "Trailer", "Motorcycle", "Bicycle", "Van", "Person",
]
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def convert_coco_json(coco_json_path: Path, images_dir: Path, dst: Path, split: str):
    with open(coco_json_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_info = {im["id"]: im for im in coco["images"]}

    out_images = dst / "images" / split
    out_labels = dst / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, info in img_id_to_info.items():
        w, h = info["width"], info["height"]
        src_img = images_dir / info["file_name"]
        if not src_img.exists():
            print(f"WARNING: missing image {src_img}")
            continue
        shutil.copy(src_img, out_images / Path(info["file_name"]).name)

        lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_name = cat_id_to_name[ann["category_id"]]
            if cat_name not in NAME_TO_ID:
                print(f"WARNING: unmapped category '{cat_name}', skipping")
                continue
            cls_id = NAME_TO_ID[cat_name]
            x, y, bw, bh = ann["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (out_labels / (Path(info["file_name"]).stem + ".txt")).write_text("\n".join(lines))

    print(f"[prepare_sample_dataset] converted {len(img_id_to_info)} images for split={split}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Raw sample dataset root")
    p.add_argument("--dst", required=True, help="Output YOLO-format dataset root")
    p.add_argument("--split", default="train")
    p.add_argument("--coco-json", default=None,
                    help="Path to COCO-style annotation json, if that's the format Hafnia ships. "
                         "If the sample instead ships per-image .txt/.xml labels, write a small "
                         "variant of convert_coco_json() for that format instead.")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    coco_json = Path(args.coco_json) if args.coco_json else src / "annotations.json"
    images_dir = src / "images"

    if not coco_json.exists():
        raise SystemExit(
            f"Annotation file {coco_json} not found. This script assumes COCO-json until the "
            f"real sample-dataset format is confirmed (needs Hafnia login) -- inspect the raw "
            f"download and adjust this script's parsing accordingly."
        )

    convert_coco_json(coco_json, images_dir, dst, args.split)


if __name__ == "__main__":
    main()
