"""
train_yolo.py -- fine-tune yolo11n (COCO-pretrained) on the warehouse-specific
dataset built by prepare_yolo_dataset.py.

Usage:
    python3 train_yolo.py --data yolo_finetune/dataset.yaml --epochs 30 --imgsz 960
"""
import argparse
import os
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="yolo_finetune/dataset.yaml")
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs_finetune")
    ap.add_argument("--name", default="warehouse7")
    args = ap.parse_args()

    last_ckpt = os.path.join(args.project, args.name, "weights", "last.pt")
    if os.path.exists(last_ckpt):
        # crash recovery: continue from the last saved epoch instead of
        # restarting all 30 epochs from scratch
        model = YOLO(last_ckpt)
        model.train(resume=True, cache="ram")
        return

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=10,
        cache="ram",
        save_period=1,
    )


if __name__ == "__main__":
    main()
