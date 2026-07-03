"""
prepare_yolo_dataset.py -- build a YOLO-format detection dataset from the
Track1 ground_truth.json 2D boxes, to fine-tune yolo11n on the warehouse
classes (Person/Forklift/PalletTruck -- the only ones present in the locally
downloaded scenes, see README.md limitations section).

Usage:
    python3 prepare_yolo_dataset.py --train-scenes Warehouse_000 Warehouse_001 \
        Warehouse_002 Warehouse_003 Warehouse_004 Warehouse_005 \
        --val-scenes Warehouse_006 --stride 45 --out yolo_finetune
"""
import argparse
import os
import json
import cv2

from common import DATA_ROOT, list_cameras, load_ground_truth, video_path, CLASS_NAME_TO_ID


def extract_split(scenes, split_name, out_dir, stride, source_split="train"):
    img_dir = os.path.join(out_dir, "images", split_name)
    lbl_dir = os.path.join(out_dir, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    n_images = 0
    n_boxes = 0
    for scene in scenes:
        gt = load_ground_truth(scene, source_split)
        cams = list_cameras(scene, source_split)
        for cam in cams:
            # frame_id -> list of (class_id, x1,y1,x2,y2) visible in this camera
            per_frame = {}
            for frame_id_str, objs in gt.items():
                fid = int(frame_id_str)
                if fid % stride != 0:
                    continue
                boxes = []
                for o in objs:
                    cls_name = o["object type"]
                    if cls_name not in CLASS_NAME_TO_ID:
                        continue
                    bbox = o.get("2d bounding box visible", {}).get(cam)
                    if not bbox:
                        continue
                    boxes.append((CLASS_NAME_TO_ID[cls_name], bbox))
                if boxes:
                    per_frame[fid] = boxes
            if not per_frame:
                continue

            vp = video_path(scene, cam, source_split)
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                print(f"WARN: cannot open {vp}")
                continue
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            target_fids = sorted(per_frame.keys())
            fid = 0
            ti = 0
            while ti < len(target_fids):
                want = target_fids[ti]
                if fid < want:
                    if not cap.grab():
                        break
                    fid += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                stem = f"{scene}__{cam}__{want:06d}"
                img_path = os.path.join(img_dir, stem + ".jpg")
                cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                lines = []
                for cls_id, (x1, y1, x2, y2) in per_frame[want]:
                    x1c, x2c = max(0, min(x1, w)), max(0, min(x2, w))
                    y1c, y2c = max(0, min(y1, h)), max(0, min(y2, h))
                    bw, bh = x2c - x1c, y2c - y1c
                    if bw <= 1 or bh <= 1:
                        continue
                    cx, cy = (x1c + x2c) / 2 / w, (y1c + y2c) / 2 / h
                    nw, nh = bw / w, bh / h
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                if lines:
                    with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                        f.write("\n".join(lines) + "\n")
                    n_images += 1
                    n_boxes += len(lines)
                else:
                    os.remove(img_path)
                fid += 1
                ti += 1
            cap.release()
            print(f"[{split_name}] {scene}/{cam}: done, running total images={n_images} boxes={n_boxes}")
    return n_images, n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-scenes", nargs="+", required=True)
    ap.add_argument("--val-scenes", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=45, help="keep 1 frame every N")
    ap.add_argument("--out", default="yolo_finetune")
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    os.makedirs(out_dir, exist_ok=True)

    n_tr, b_tr = extract_split(args.train_scenes, "train", out_dir, args.stride)
    n_va, b_va = extract_split(args.val_scenes, "val", out_dir, args.stride)

    names = {v: k for k, v in CLASS_NAME_TO_ID.items()}
    yaml_txt = (
        f"path: {out_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {names[i]}\n" for i in sorted(names))
    )
    with open(os.path.join(out_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_txt)

    print(f"\nDONE. train images={n_tr} boxes={b_tr} | val images={n_va} boxes={b_va}")
    print(f"dataset.yaml written to {os.path.join(out_dir, 'dataset.yaml')}")


if __name__ == "__main__":
    main()
