"""Retrieval with standard ReID-style post-processing on top of CLIP scores:

1. alpha-Query-Expansion (aQE): each query embedding is refined with a
   similarity-weighted average of its top-k gallery neighbors, then the
   gallery is re-scored. Helps when the caption wording is far from CLIP's
   image manifold but the right images cluster together.
2. Local k-reciprocal Jaccard re-ranking (Zhong et al., CVPR'17), restricted
   to each query's top-M candidates so the 36.7k gallery never needs a full
   36k x 36k matrix.

Both are inference-time post-processing on embeddings -- nothing is trained,
no test labels/distribution are fitted, so this stays clean for the Public
leaderboard and Track 4's "no test data may be used [for training]" rule.

Usage:
    python3 rerank_retrieve.py --model clip_finetuned_big \
        --gallery-embeds data/gallery_embeds__clip_finetuned_big.pt \
        --out answer_rerank.txt [--qe-k 3] [--qe-alpha 3.0] [--rerank-m 200] [--k1 20] [--lambda-val 0.3]
"""
import argparse
import json

import torch

from common import embed_texts, load_clip


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def alpha_query_expansion(query_embeds, gallery_embeds, k=3, alpha=3.0):
    sims = query_embeds @ gallery_embeds.T
    topk_sims, topk_idx = sims.topk(k, dim=1)
    weights = topk_sims.clamp(min=0) ** alpha  # [nq, k]
    neighbors = gallery_embeds[topk_idx]       # [nq, k, d]
    expanded = query_embeds + (weights.unsqueeze(-1) * neighbors).sum(dim=1)
    return expanded / expanded.norm(dim=-1, keepdim=True)


def local_k_reciprocal_rerank(query_embeds, gallery_embeds, top_m=200, k1=20, lambda_val=0.3):
    """Re-rank each query's top-M candidates with k-reciprocal Jaccard distance,
    computed only over the local candidate set (memory-safe on 36k galleries)."""
    nq = query_embeds.size(0)
    sims = query_embeds @ gallery_embeds.T
    base_top = sims.topk(top_m, dim=1).indices  # [nq, M]

    reranked_rows = []
    for qi in range(nq):
        cand_idx = base_top[qi]                       # [M]
        cand = gallery_embeds[cand_idx]               # [M, d]
        q = query_embeds[qi : qi + 1]                 # [1, d]

        # distances within the local set (append query as node 0)
        feats = torch.cat([q, cand], dim=0)           # [M+1, d]
        dist = 1 - feats @ feats.T                    # cosine distance
        # k-reciprocal neighbors of the query node
        k1_eff = min(k1, dist.size(0) - 1)
        q_nn = dist[0].topk(k1_eff + 1, largest=False).indices  # includes self
        recip = []
        for j in q_nn.tolist():
            if j == 0:
                continue
            j_nn = dist[j].topk(k1_eff + 1, largest=False).indices
            if 0 in j_nn.tolist():
                recip.append(j)

        # Jaccard-ish distance: candidates sharing reciprocal neighbors with
        # the query move up; blend with original cosine distance.
        jaccard = torch.ones(cand.size(0))
        if recip:
            recip_t = torch.tensor(recip)
            recip_feats = feats[recip_t]              # [R, d]
            cand_to_recip = 1 - cand @ recip_feats.T  # [M, R]
            jaccard = cand_to_recip.min(dim=1).values  # closer to any reciprocal neighbor = better

        final = lambda_val * dist[0, 1:] + (1 - lambda_val) * jaccard
        order = final.argsort()
        reranked_rows.append(cand_idx[order])

    return torch.stack(reranked_rows)  # [nq, M] reranked gallery indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gallery-embeds", required=True)
    ap.add_argument("--query-text", default="data/query_text.json")
    ap.add_argument("--query-index", default="data/query_index.txt")
    ap.add_argument("--out", default="answer_rerank.txt")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--qe-k", type=int, default=3)
    ap.add_argument("--qe-alpha", type=float, default=3.0)
    ap.add_argument("--rerank-m", type=int, default=200)
    ap.add_argument("--k1", type=int, default=20)
    ap.add_argument("--lambda-val", type=float, default=0.3)
    ap.add_argument("--no-qe", action="store_true")
    ap.add_argument("--no-krecip", action="store_true")
    args = ap.parse_args()

    cache = torch.load(args.gallery_embeds, weights_only=False)
    gallery_ids, gallery_embeds = cache["ids"], cache["embeds"]

    queries = {q["query_index"]: q["caption"] for q in load_jsonl(args.query_text)}
    with open(args.query_index) as f:
        query_order = [line.strip() for line in f if line.strip()]
    captions = [queries[q] for q in query_order]

    model, processor = load_clip(args.model)
    query_embeds = embed_texts(model, processor, captions)

    if not args.no_qe:
        print(f"alpha-QE: k={args.qe_k} alpha={args.qe_alpha}")
        query_embeds = alpha_query_expansion(query_embeds, gallery_embeds, k=args.qe_k, alpha=args.qe_alpha)

    if not args.no_krecip:
        print(f"local k-reciprocal rerank: M={args.rerank_m} k1={args.k1} lambda={args.lambda_val}")
        ranked = local_k_reciprocal_rerank(
            query_embeds, gallery_embeds,
            top_m=args.rerank_m, k1=args.k1, lambda_val=args.lambda_val,
        )
        topk = ranked[:, : args.topk]
    else:
        sims = query_embeds @ gallery_embeds.T
        topk = sims.topk(args.topk, dim=1).indices

    with open(args.out, "w") as f:
        for row in topk:
            f.write(" ".join(gallery_ids[i] for i in row.tolist()) + "\n")
    print(f"wrote {len(query_order)} rows x top-{args.topk} to {args.out}")


if __name__ == "__main__":
    main()
