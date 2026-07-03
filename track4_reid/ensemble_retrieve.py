"""Score-fusion ensemble across multiple CLIP checkpoints for Track 4 retrieval.

Combining independently-trained/fine-tuned models is one of the most
reliable, low-risk ways to push retrieval mAP -- each model's embedding
space captures the domain slightly differently (e.g. text-tower-only
fine-tune vs full fine-tune), and averaging normalized similarities
usually beats any single model. This does NOT touch the test distribution
in any way (still just cosine similarity over model embeddings), so it's
clean for the Public leaderboard.

Usage:
    python3 ensemble_retrieve.py \
        --models openai/clip-vit-large-patch14 clip_finetuned clip_finetuned_full \
        --weights 1.0 1.0 1.5 \
        --out answer_ensemble.txt

Each model gets its own gallery embedding cache (data/gallery_embeds__<name>.pt),
computed once and reused on subsequent runs.
"""
import argparse
import glob
import json
import os

import torch

from common import embed_images, embed_texts, load_clip


def cache_path(model_name):
    safe = model_name.replace("/", "__")
    return f"data/gallery_embeds__{safe}.pt"


def get_gallery_embeds(model_name, gallery_dir, model, processor, batch_size):
    path = cache_path(model_name)
    if os.path.exists(path):
        cache = torch.load(path, weights_only=False)
        return cache["ids"], cache["embeds"]

    paths = sorted(glob.glob(os.path.join(gallery_dir, "*.jpg")))
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    embeds = embed_images(model, processor, paths, batch_size=batch_size)
    torch.save({"ids": ids, "embeds": embeds}, path)
    return ids, embeds


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="HF model names or local checkpoint dirs")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="one weight per model, default all 1.0")
    ap.add_argument("--gallery-dir", default="data/gallery")
    ap.add_argument("--query-text", default="data/query_text.json")
    ap.add_argument("--query-index", default="data/query_index.txt")
    ap.add_argument("--out", default="answer_ensemble.txt")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.models)
    assert len(weights) == len(args.models), "need one weight per model"

    queries = {q["query_index"]: q["caption"] for q in load_jsonl(args.query_text)}
    with open(args.query_index) as f:
        query_order = [line.strip() for line in f if line.strip()]
    captions = [queries[q] for q in query_order]

    fused_sims = None
    gallery_ids = None
    for model_name, weight in zip(args.models, weights):
        print(f"scoring with {model_name} (weight={weight})")
        model, processor = load_clip(model_name)
        ids, gallery_embeds = get_gallery_embeds(model_name, args.gallery_dir, model, processor, args.batch_size)
        if gallery_ids is None:
            gallery_ids = ids
        assert ids == gallery_ids, f"{model_name} gallery ordering doesn't match -- delete stale cache files"

        query_embeds = embed_texts(model, processor, captions, batch_size=args.batch_size)
        sims = query_embeds @ gallery_embeds.T
        fused_sims = sims * weight if fused_sims is None else fused_sims + sims * weight

    topk = fused_sims.topk(args.topk, dim=1).indices
    with open(args.out, "w") as f:
        for row in topk:
            f.write(" ".join(gallery_ids[i] for i in row.tolist()) + "\n")

    print(f"wrote {len(query_order)} rows x top-{args.topk} to {args.out}")


if __name__ == "__main__":
    main()
