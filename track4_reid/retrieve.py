"""Zero-shot CLIP text->image retrieval: embed queries, rank gallery, write answer.txt.

Usage:
    python3 retrieve.py --gallery-embeds data/gallery_embeds.pt \
        --query-text data/query_text.json --query-index data/query_index.txt \
        --out answer.txt
"""
import argparse
import json

import torch

from common import embed_texts, load_clip


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-embeds", default="data/gallery_embeds.pt")
    ap.add_argument("--query-text", default="data/query_text.json")
    ap.add_argument("--query-index", default="data/query_index.txt")
    ap.add_argument("--out", default="answer.txt")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--model", default=None, help="HF model name or local fine-tuned checkpoint dir")
    args = ap.parse_args()

    cache = torch.load(args.gallery_embeds)
    gallery_ids, gallery_embeds = cache["ids"], cache["embeds"]

    queries = {q["query_index"]: q["caption"] for q in load_jsonl(args.query_text)}
    with open(args.query_index) as f:
        query_order = [line.strip() for line in f if line.strip()]

    missing = [q for q in query_order if q not in queries]
    if missing:
        raise ValueError(f"{len(missing)} query indices have no caption, e.g. {missing[:3]}")

    model, processor = load_clip(args.model) if args.model else load_clip()
    captions = [queries[q] for q in query_order]
    query_embeds = embed_texts(model, processor, captions)

    sims = query_embeds @ gallery_embeds.T  # [num_queries, num_gallery]
    topk = sims.topk(args.topk, dim=1).indices

    with open(args.out, "w") as f:
        for row in topk:
            f.write(" ".join(gallery_ids[i] for i in row.tolist()) + "\n")

    print(f"wrote {len(query_order)} rows x top-{args.topk} to {args.out}")


if __name__ == "__main__":
    main()
