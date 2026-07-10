#!/usr/bin/env python3
"""
Track 6 alternative trainer: RF-DETR (Roboflow) instead of YOLO11.

Switched to this after seeing the actual Track 6 leaderboard: the top 3
teams all run RF-DETR variants (RFDETR-HR3 0.4710 mAP, DETR-Grayworld
0.4700, RFDETR-AiO), not YOLO -- general small-object benchmarks suggested
YOLO11 might edge out RF-DETR here, but real task-specific results say
otherwise, so real results win.

Reuses the same zero-copy dataset construction as train.py (symlink images,
write YOLO-format label txts -- avoids the ~70GB image duplication that
exhausted the Lite instance's disk during the YOLO shakeout), but lays out
directories the way rfdetr's YOLO loader expects: train/valid/test with
images/ + labels/ subdirs and a data.yaml at the root (confirmed by reading
rfdetr/datasets/yolo.py -- it hardcodes "valid", not "validation").

Usage:
    python scripts/train_rfdetr.py --epochs 40 --resolution 728 --name rfdetr-main
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

TRACK6_DATASET_NAME = os.environ.get("TRACK6_DATASET_NAME", "eccv-cross-city")
TRACK6_DATASET_VERSION = os.environ.get("TRACK6_DATASET_VERSION", "1.0.0")

FALLBACK_CLASS_NAMES = [
    "Vehicle.Car", "Vehicle.Pickup Truck", "Vehicle.Single Truck",
    "Vehicle.Combo Truck", "Vehicle.Heavy Duty Vehicle", "Vehicle.Trailer",
    "Vehicle.Motorcycle", "Vehicle.Bicycle", "Vehicle.Van", "Person",
]

SPLIT_RENAME = {"validation": "valid", "val": "valid"}  # rfdetr's yolo loader hardcodes "valid"


def parse_args():
    p = argparse.ArgumentParser(description="Track6 RF-DETR training")
    p.add_argument("--dataset-name", type=str, default=TRACK6_DATASET_NAME)
    p.add_argument("--dataset-version", type=str, default=TRACK6_DATASET_VERSION)
    p.add_argument("--model-size", type=str, default="base", choices=["base", "large"],
                   help="RFDETRBase (~29M params, res 560 default) or RFDETRLarge (~128M, slower).")
    p.add_argument("--resolution", type=int, default=560,
                   help="Must be divisible by patch_size*num_windows for the variant; "
                        "560 is RFDETRBase's native default (matches its DINOv2 positional encoding).")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-path", type=str, default=os.environ.get("HAFNIA_OUTPUT_PATH", "runs"))
    p.add_argument("--name", type=str, default="rfdetr_run")
    p.add_argument("--dry-run", action="store_true",
                   help="Tiny run (1 epoch) to validate the pipeline mechanics before spending credits.")
    return p.parse_args()


def get_mount():
    from hafnia.utils import get_dataset_path_in_hafnia_cloud, is_hafnia_cloud_job
    if is_hafnia_cloud_job():
        return Path(get_dataset_path_in_hafnia_cloud())
    HafniaDataset_from_name()
    from hafnia.utils import PATH_DATASETS
    return Path(PATH_DATASETS) / TRACK6_DATASET_NAME


def HafniaDataset_from_name():
    from hafnia.dataset.hafnia_dataset import HafniaDataset
    HafniaDataset.from_name(TRACK6_DATASET_NAME, version=TRACK6_DATASET_VERSION)


def load_records(mount):
    ann_path = mount / "annotations.jsonl"
    if ann_path.exists():
        records = [json.loads(l) for l in ann_path.read_text().splitlines() if l.strip()]
        print(f"[train_rfdetr] loaded {len(records)} records from annotations.jsonl", file=sys.stderr)
        return records
    from hafnia.dataset.hafnia_dataset import HafniaDataset
    ds = HafniaDataset.from_path(mount)
    records = list(ds.samples.iter_rows(named=True))
    print(f"[train_rfdetr] loaded {len(records)} records via HafniaDataset (parquet)", file=sys.stderr)
    return records


def build_rfdetr_yolo_dataset(mount, output_path):
    """Zero-copy: symlink images + write YOLO label txts, laid out as
    <root>/{train,valid,test}/{images,labels}/ + <root>/data.yaml."""
    records = load_records(mount)

    class_names = {}
    for r in records:
        for b in (r.get("bboxes") or []):
            if b.get("class_name") is not None and b.get("class_idx") is not None:
                class_names[int(b["class_idx"])] = b["class_name"]
    nc = max(class_names) + 1 if class_names else len(FALLBACK_CLASS_NAMES)
    names = [class_names.get(i, FALLBACK_CLASS_NAMES[i] if i < len(FALLBACK_CLASS_NAMES) else f"class_{i}")
             for i in range(nc)]

    root = (Path(output_path) / "rfdetr_dataset").resolve()
    counts = {}
    for r in records:
        split = SPLIT_RENAME.get(r["split"], r["split"])
        if split not in ("train", "valid", "test"):
            continue  # rfdetr's loader only knows these three
        fp = Path(r["file_path"])
        img_src = fp.resolve() if fp.is_absolute() else (mount / fp).resolve()
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        img_link = img_dir / img_src.name
        if not img_link.exists():
            img_link.symlink_to(img_src)

        lines = []
        for b in (r.get("bboxes") or []):
            cx = b["top_left_x"] + b["width"] / 2.0
            cy = b["top_left_y"] + b["height"] / 2.0
            lines.append(f'{int(b["class_idx"])} {cx:.6f} {cy:.6f} {b["width"]:.6f} {b["height"]:.6f}')
        (lbl_dir / (img_link.stem + ".txt")).write_text("\n".join(lines))
        counts[split] = counts.get(split, 0) + 1

    (root / "data.yaml").write_text(yaml.safe_dump({
        "train": "train/images", "val": "valid/images", "test": "test/images",
        "nc": nc, "names": names,
    }))
    print(f"[train_rfdetr] dataset built: {counts}, {nc} classes -> {root}", file=sys.stderr)
    return root


def stage_offline_weights(model_size):
    """Training instances have NO internet (same issue that killed the YOLO
    shakeout runs on the yolo11n.pt download). RF-DETR needs two things from
    the network that aren't obvious from its API:
      1. Its own checkpoint (RFDETRBase -> rf-detr-base.pth, RFDETRLarge ->
         rf-detr-large.pth), fetched by maybe_download_pretrain_weights()
         into ~/.roboflow/models/ (overridable via RF_HOME).
      2. Its DINOv2 backbone (facebook/dinov2-with-registers-small/base),
         fetched from the HF Hub by WindowedDinov2WithRegistersBackbone.
           from_pretrained() -- a plain transformers-style HF cache lookup.
    Both are pre-downloaded into trainer_package/weights and
    trainer_package/hf_cache at build time; stage them into the exact
    locations these libraries expect before touching rfdetr's imports.
    """
    import shutil

    pkg_root = Path(__file__).resolve().parent.parent
    weights_dir = pkg_root / "weights"
    checkpoint_name = "rf-detr-large.pth" if model_size == "large" else "rf-detr-base.pth"
    bundled_ckpt = weights_dir / checkpoint_name
    if bundled_ckpt.exists():
        rf_home = Path(os.environ.get("RF_HOME", os.path.expanduser("~/.roboflow/models")))
        rf_home.mkdir(parents=True, exist_ok=True)
        dst = rf_home / checkpoint_name
        if not dst.exists():
            shutil.copy2(bundled_ckpt, dst)
        print(f"[train_rfdetr] staged {checkpoint_name} -> {dst}", file=sys.stderr)
    else:
        print(f"[train_rfdetr] WARNING: {bundled_ckpt} not bundled, "
              f"RFDETR will try to download it (will fail offline)", file=sys.stderr)

    hf_cache_dir = pkg_root / "hf_cache"
    if hf_cache_dir.exists():
        os.environ["HF_HOME"] = str(hf_cache_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        print(f"[train_rfdetr] staged HF cache -> {hf_cache_dir}", file=sys.stderr)
    else:
        print(f"[train_rfdetr] WARNING: {hf_cache_dir} not bundled, "
              f"DINOv2 backbone download will fail offline", file=sys.stderr)


def main():
    args = parse_args()
    mount = get_mount()
    try:
        entries = sorted(p.name for p in mount.iterdir())
        print(f"[train_rfdetr] mount contents ({mount}): {entries[:20]}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[train_rfdetr] cannot list mount {mount}: {e}", file=sys.stderr)

    dataset_dir = build_rfdetr_yolo_dataset(mount, args.output_path)

    stage_offline_weights(args.model_size)

    from rfdetr import RFDETRBase, RFDETRLarge
    ModelClass = RFDETRLarge if args.model_size == "large" else RFDETRBase
    model = ModelClass(resolution=args.resolution)

    from hafnia.experiment import HafniaLogger
    logger = HafniaLogger(project_name="track6-cross-city-rfdetr")
    logger.log_hparams({
        "model_size": args.model_size, "resolution": args.resolution,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
    })

    epochs = 1 if args.dry_run else args.epochs
    output_dir = str(Path(args.output_path) / args.name)
    # NOTE: mlflow=True + log_per_class_metrics was tried as a redundant
    # metrics safety net, but it overloads Hafnia's shared MLflow tracking
    # server (mlflow.exceptions.RestException: QueuePool limit of size 2
    # overflow 3 reached) and crashes the whole run early -- confirmed cause
    # of a full training-run loss. rfdetr's own per-epoch COCOEvalCallback
    # logging (stdout _logger.info calls) already survives via the
    # /experiments/{id}/logs API (fetch with limit=5000, not the default
    # page size, which silently truncates the real tail) -- that's the only
    # metrics channel we actually need.
    model.train(
        dataset_dir=str(dataset_dir),
        dataset_file="yolo",
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=output_dir,
        device="cuda",
    )

    best_pt = Path(output_dir) / "checkpoint_best_total.pth"
    if best_pt.exists():
        import shutil
        shutil.copy2(best_pt, logger.path_model() / "checkpoint_best_total.pth")
    logger.end_run()
    print(f"[train_rfdetr] done. best weights: {best_pt}")


if __name__ == "__main__":
    main()
