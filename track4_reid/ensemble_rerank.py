"""Ensemble + re-ranking combined: fuse similarities across checkpoints
(what got 62.64 mAP), then apply alpha-QE and local k-reciprocal re-ranking
in the fused similarity space (what added +0.3 on the single model).

QE here operates per-model (expand each model's queries with its own gallery
neighbors chosen by the FUSED ranking, then re-fuse) so each embedding space
stays internally consistent.

Usage:
    python3 ensemble_rerank.py \
        --models openai/clip-vit-large-patch14 clip_finetuned clip_finetuned_big \
        --weights 0.5 1.0 1.5 --out answer_ensemble_rerank.txt
"""
import argparse
import json

import torch

from common import embed_texts, load_clip
from rerank_retrieve import local_k_reciprocal_rerank


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cache_path(model_name):
    return f"data/gallery_embeds__{model_name.replace('/', '__')}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--query-text", default="data/query_text.json")
    ap.add_argument("--query-index", default="data/query_index.txt")
    ap.add_argument("--out", default="answer_ensemble_rerank.txt")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--qe-k", type=int, default=3)
    ap.add_argument("--qe-alpha", type=float, default=3.0)
    ap.add_argument("--rerank-m", type=int, default=200)
    ap.add_argument("--k1", type=int, default=20)
    ap.add_argument("--lambda-val", type=float, default=0.3)
    args = ap.parse_args()

    queries = {q["query_index"]: q["caption"] for q in load_jsonl(args.query_text)}
    with open(args.query_index) as f:
        query_order = [line.strip() for line in f if line.strip()]
    captions = [queries[q] for q in query_order]

    galleries, query_embeds_per_model = [], []
    gallery_ids = None
    for name in args.models:
        cache = torch.load(cache_path(name), weights_only=False)
        if gallery_ids is None:
            gallery_ids = cache["ids"]
        assert cache["ids"] == gallery_ids
        galleries.append(cache["embeds"])
        model, processor = load_clip(name)
        query_embeds_per_model.append(embed_texts(model, processor, captions))
        del model, processor
        torch.cuda.empty_cache()

    # pass 1: fused base similarity
    fused = None
    for qe, ge, w in zip(query_embeds_per_model, galleries, args.weights):
        sims = qe @ ge.T
        fused = sims * w if fused is None else fused + sims * w

    # pass 2: per-model alpha-QE guided by the FUSED top-k, then re-fuse.
    # weights_qe must be a PER-QUERY max, not a global one -- a global
    # `.max()` (no dim=) collapses to the single strongest match across all
    # 1978 queries, so nearly every query's weight sits close to 1 regardless
    # of how confident THAT query's own top-k match actually is. That erases
    # the confidence-scaling alpha-QE is supposed to provide and expands
    # weak/ambiguous queries almost as aggressively as confident ones.
    topk_sims, topk_idx = fused.topk(args.qe_k, dim=1)
    weights_qe = (topk_sims / topk_sims.max(dim=1, keepdim=True).values).clamp(min=0) ** args.qe_alpha
    fused2 = None
    expanded_per_model = []
    for qe, ge, w in zip(query_embeds_per_model, galleries, args.weights):
        neighbors = ge[topk_idx]  # [nq, k, d] in THIS model's space
        expanded = qe + (weights_qe.unsqueeze(-1) * neighbors).sum(dim=1)
        expanded = expanded / expanded.norm(dim=-1, keepdim=True)
        expanded_per_model.append(expanded)
        sims = expanded @ ge.T
        fused2 = sims * w if fused2 is None else fused2 + sims * w

    # pass 3: local k-reciprocal reranking. Must run on the POST-QE expanded
    # embeddings (the ones that actually produced fused2), not the raw
    # pre-QE query_embeds_per_model -- using the stale pre-QE embeddings here
    # silently threw away the QE step for this pass and reintroduced a single
    # checkpoint's un-expanded bias into the final ranking, working against
    # the whole point of ensembling.
    strongest = max(range(len(args.weights)), key=lambda i: args.weights[i])
    ranked = local_k_reciprocal_rerank(
        expanded_per_model[strongest], galleries[strongest],
        top_m=args.rerank_m, k1=args.k1, lambda_val=args.lambda_val,
    )
    # blend: take fused2 ranking, but demote candidates the k-reciprocal pass
    # pushed far down. Rank-average fusion of the two orderings over top-M.
    final_rows = []
    m = args.rerank_m
    fused_top = fused2.topk(m, dim=1).indices
    for qi in range(fused2.size(0)):
        rank_a = {int(g): r for r, g in enumerate(fused_top[qi].tolist())}
        rank_b = {int(g): r for r, g in enumerate(ranked[qi].tolist())}
        cands = set(rank_a) | set(rank_b)
        scored = sorted(
            cands,
            key=lambda g: 0.7 * rank_a.get(g, m) + 0.3 * rank_b.get(g, m),
        )
        final_rows.append(scored[: args.topk])

    with open(args.out, "w") as f:
        for row in final_rows:
            f.write(" ".join(gallery_ids[i] for i in row) + "\n")
    print(f"wrote {len(final_rows)} rows x top-{args.topk} to {args.out}")


if __name__ == "__main__":
    main()
