"""Fast local proxy metric: image<->caption retrieval recall@k on the held-out
val split from imgs_14 (same split finetune_clip.py trained against).

Not the real Track 4 metric (no distractors, in-domain images only) but it's a
free, fast signal to compare model configs before burning a real portal
submission -- run this after any fine-tune to sanity-check the direction is
right.

Usage:
    python3 eval_recall.py --model clip_finetuned
    python3 eval_recall.py --model openai/clip-vit-large-patch14   # zero-shot baseline
"""
import argparse
import json
import os
import random

import torch

from common import embed_images, embed_texts, load_clip

ANNOTATION = "data/train_annotation_all.jsonl"
TRAIN_ROOT = "data/train"


def recall_at_k(sims, k):
    n = sims.size(0)
    topk = sims.topk(k, dim=1).indices
    labels = torch.arange(n).unsqueeze(1)
    hits = (topk == labels).any(dim=1)
    return hits.float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(ANNOTATION) if l.strip()]
    random.Random(0).shuffle(records)  # same seed as finetune_clip.py -> same val split
    n_val = int(len(records) * args.val_frac)
    val_records = records[:n_val]

    image_paths = [
        os.path.join(TRAIN_ROOT, *r["image"].split("/")[1:]).replace(".jpg", ".webp") for r in val_records
    ]
    captions = [r["caption"] for r in val_records]

    model, processor = load_clip(args.model)
    image_embeds = embed_images(model, processor, image_paths)
    text_embeds = embed_texts(model, processor, captions)

    sims = text_embeds @ image_embeds.T  # text-to-image, matches the real task direction
    for k in (1, 5, 10):
        print(f"R@{k}: {recall_at_k(sims, k) * 100:.2f}%")


if __name__ == "__main__":
    main()
