"""Contrastive fine-tune of CLIP on the one recoverable train shard (imgs_14, 13.5k
image-caption pairs) to close the domain gap vs zero-shot on PAB-style anomaly captions.

Usage:
    python3 finetune_clip.py --epochs 3 --lr 1e-6 --out clip_finetuned
"""
import argparse
import json
import os
import random
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import DEVICE, MODEL_NAME
from transformers import CLIPModel, CLIPProcessor

TRAIN_ROOT = "data/train"
ANNOTATION = "data/train_annotation_all.jsonl"


class PairDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_path = os.path.join(TRAIN_ROOT, *rec["image"].split("/")[1:]).replace(".jpg", ".webp")
        image = Image.open(image_path).convert("RGB")
        return image, rec["caption"]


def collate(batch, processor):
    images, captions = zip(*batch)
    inputs = processor(
        text=list(captions), images=list(images), return_tensors="pt", padding=True, truncation=True
    )
    return inputs


def clip_contrastive_loss(model, inputs):
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(**inputs)
        logits_per_image = out.logits_per_image  # already scaled by logit_scale
        logits_per_text = out.logits_per_text
        labels = torch.arange(logits_per_image.size(0), device=DEVICE)
        loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="clip_finetuned")
    ap.add_argument("--freeze-vision", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--step-sleep", type=float, default=0.0, help="seconds to sleep after each step, to pace disk/GPU load")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(ANNOTATION) if l.strip()]
    random.Random(0).shuffle(records)
    n_val = int(len(records) * args.val_frac)
    val_records, train_records = records[:n_val], records[n_val:]
    print(f"train={len(train_records)} val={len(val_records)}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    if args.freeze_vision:
        # 13.5k pairs is too little to safely tune a 300M-param ViT without
        # overfitting/forgetting -- only the text tower + projections adapt.
        for param in model.vision_model.parameters():
            param.requires_grad = False

    train_loader = DataLoader(
        PairDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, processor),
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        PairDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, processor),
        num_workers=max(1, args.num_workers // 2),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, inputs in enumerate(train_loader):
            loss = clip_contrastive_loss(model, inputs)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            if args.step_sleep > 0:
                time.sleep(args.step_sleep)
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        print(f"epoch {epoch} train loss avg {total_loss / len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = sum(clip_contrastive_loss(model, inputs).item() for inputs in val_loader)
        print(f"epoch {epoch} val loss avg {val_loss / len(val_loader):.4f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
