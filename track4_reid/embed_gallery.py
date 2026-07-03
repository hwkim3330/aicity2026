"""Embed all gallery images (query + distractors) with CLIP and cache to disk.

Usage:
    python3 embed_gallery.py --gallery-dir data/gallery --out data/gallery_embeds.pt
"""
import argparse
import glob
import os
import time

import torch

from common import embed_images, load_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-dir", default="data/gallery")
    ap.add_argument("--out", default="data/gallery_embeds.pt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model", default=None, help="HF model name or local fine-tuned checkpoint dir")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.gallery_dir, "*.jpg")))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    print(f"found {len(paths)} gallery images")

    model, processor = load_clip(args.model) if args.model else load_clip()
    t0 = time.time()
    embeds = embed_images(model, processor, paths, batch_size=args.batch_size)
    print(f"embedded in {time.time() - t0:.1f}s -> {embeds.shape}")

    torch.save({"ids": ids, "embeds": embeds}, args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
