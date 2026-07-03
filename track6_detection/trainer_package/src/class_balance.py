"""
Practical class-imbalance handling for Track 6 (10 classes, expected heavy
skew towards Vehicle.Car / Person and scarcity of Trailer, Combo Truck,
Heavy Duty Vehicle, Bicycle in typical traffic-camera footage).

Approach: image-level oversampling by inverse class frequency, implemented
by rewriting Ultralytics' train image list to repeat images containing rare
classes. This is dataset-format agnostic (works on YOLO .txt labels) and
does not require custom loss functions or a modified dataloader, so it
survives whatever exact directory layout Hafnia mounts the data into.

If a per-class weighted *loss* is preferred instead/also, see
focal_patch.py which additionally supports per-class alpha weighting.
"""
import os
import glob
import math
from collections import Counter
from pathlib import Path

import yaml


def _iter_label_files(images_dir: Path):
    labels_dir = Path(str(images_dir).replace("images", "labels"))
    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        lbl_path = labels_dir / (img_path.stem + ".txt")
        yield img_path, lbl_path


def compute_class_counts(images_dir: Path, nc: int) -> Counter:
    counts = Counter({c: 0 for c in range(nc)})
    for _, lbl_path in _iter_label_files(images_dir):
        if not lbl_path.exists():
            continue
        with open(lbl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls = int(line.split()[0])
                counts[cls] += 1
    return counts


def build_oversampled_list(images_dir: Path, nc: int, cap_multiplier: float = 8.0):
    """Return a list of image paths where images containing rarer classes
    are repeated proportionally to inverse sqrt frequency (sqrt to avoid
    extreme duplication ratios blowing up epoch time), capped at
    `cap_multiplier`x the image's natural count."""
    counts = compute_class_counts(images_dir, nc)
    total = sum(counts.values()) or 1
    freq = {c: counts[c] / total for c in counts}
    max_freq = max(freq.values()) if freq else 1.0

    weighted_list = []
    for img_path, lbl_path in _iter_label_files(images_dir):
        classes_in_img = set()
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        classes_in_img.add(int(line.split()[0]))
        if not classes_in_img:
            weighted_list.append(str(img_path))
            continue
        rarity = max((max_freq / max(freq.get(c, 1e-9), 1e-9)) ** 0.5 for c in classes_in_img)
        repeats = min(cap_multiplier, max(1.0, rarity))
        weighted_list.extend([str(img_path)] * int(round(repeats)))
    return weighted_list


def maybe_reweight_dataset(data_yaml_path: str) -> str:
    """Given a resolved Ultralytics data.yaml, generate a rare-class-
    oversampled train image list (train.oversampled.txt) and point a new
    yaml at it via the `train:` field (Ultralytics accepts a .txt file of
    image paths as well as a directory)."""
    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    train_rel = cfg["train"]
    images_dir = root / train_rel if not os.path.isabs(train_rel) else Path(train_rel)

    if not images_dir.exists():
        # Dataset not mounted yet (e.g. running --dry-run locally without
        # data). Skip oversampling silently rather than crash.
        print(f"[class_balance] {images_dir} not found, skipping oversampling")
        return data_yaml_path

    nc = cfg.get("nc", 10)
    oversampled = build_oversampled_list(images_dir, nc)
    list_path = "/tmp/track6_train_oversampled.txt"
    with open(list_path, "w") as f:
        f.write("\n".join(oversampled))

    cfg["train"] = list_path
    new_yaml = "/tmp/track6_data_oversampled.yaml"
    with open(new_yaml, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"[class_balance] wrote {len(oversampled)} oversampled train entries -> {list_path}")
    return new_yaml
