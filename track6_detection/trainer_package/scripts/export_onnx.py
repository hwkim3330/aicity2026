#!/usr/bin/env python3
"""
Export the fine-tuned checkpoint to ONNX.

Included because the public `milestone-hafnia/trainer-object-detection`
reference repo ships a scripts/export_onnx.py alongside train.py and
benchmark.py -- mirroring that shape in case the Hafnia platform/dashboard
expects or invokes it (e.g. for a model-size check against the separate
2GB trained-model cap, or for a downloadable-model feature). Not yet
confirmed whether Hafnia calls this automatically or it's purely for the
participant's own convenience.
"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to a .pt checkpoint")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--out", default=None, help="Output .onnx path (default: alongside weights)")
    p.add_argument("--dynamic", action="store_true", default=True)
    p.add_argument("--simplify", action="store_true", default=True)
    args = p.parse_args()

    model = YOLO(args.weights)
    exported = model.export(format="onnx", imgsz=args.imgsz, dynamic=args.dynamic,
                             simplify=args.simplify)
    print(f"[export_onnx.py] exported -> {exported}")


if __name__ == "__main__":
    main()
