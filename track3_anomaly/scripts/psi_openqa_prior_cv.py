#!/usr/bin/env python3
"""CV-honest evaluation of a prior-only (video-blind) answer for PSI Open QA.

Per direction (cross / not_cross / uncertain), greedily select a fixed cue
set from the TRAIN-FOLD cue vocabulary that maximizes mean item Cue-F1 on
the train folds, then score that fixed answer on the held-out fold.
5-fold CV grouped by video. Reports the mean held-out F1 -- an unbiased
estimate of what shipping a constant per-direction answer would score.
"""
import json
import re
from collections import Counter

import numpy as np

from psi_openqa_cuef1 import _model, parse_cues

D = json.load(open("../data/psi_vqa/train/open_qa.json"))["items"]


def direction(q):
    if "NOT intend" in q:
        return "not_cross"
    if "uncertain" in q:
        return "uncertain"
    return "cross"


for it in D:
    it["dir"] = direction(it["question"])
    it["gt_cues"] = parse_cues(it["answer"])
    it["gt_none"] = it["answer"].strip().lower() == "none"

videos = sorted(set(it["video_id"] for it in D))
rng = np.random.RandomState(42)
order = rng.permutation(len(videos))
folds = [set(videos[i] for i in order[k::5]) for k in range(5)]

m = _model()

# embed every unique GT cue once
all_cues = sorted(set(c for it in D for c in it["gt_cues"]))
emb = m.encode(all_cues, batch_size=256, normalize_embeddings=True)
cue_emb = {c: emb[i] for i, c in enumerate(all_cues)}


def item_f1(pred_cue_embs, it):
    """Score a fixed predicted cue set (embeddings) against one item."""
    if it["gt_none"]:
        return 0.0 if len(pred_cue_embs) else 1.0
    if not len(pred_cue_embs):
        return 0.0
    ge = np.stack([cue_emb[c] for c in it["gt_cues"]])
    sim = pred_cue_embs @ ge.T
    mp = (sim.max(axis=1) >= 0.55).sum()
    mg = (sim.max(axis=0) >= 0.55).sum()
    p = mp / len(pred_cue_embs)
    r = mg / len(ge)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def greedy_select(items, pool, max_k=4):
    """Greedy forward selection of cues maximizing mean item F1."""
    sel = []
    best = -1.0
    while len(sel) < max_k:
        best_c, best_s = None, best
        cand_scores = {}
        for c in pool:
            if c in sel:
                continue
            embs = np.stack([cue_emb[x] for x in sel + [c]])
            s = np.mean([item_f1(embs, it) for it in items])
            cand_scores[c] = s
            if s > best_s:
                best_s, best_c = s, c
        if best_c is None:
            break
        sel.append(best_c)
        best = best_s
    return sel, best


results = {}
for dr in ("cross", "not_cross", "uncertain"):
    items_d = [it for it in D if it["dir"] == dr]
    ho_scores = []
    picks = []
    for k in range(5):
        tr = [it for it in items_d if it["video_id"] not in folds[k]]
        ho = [it for it in items_d if it["video_id"] in folds[k]]
        # candidate pool: cues appearing in train folds of this direction,
        # capped to the 120 most frequent for tractability
        cnt = Counter(c for it in tr for c in it["gt_cues"])
        pool = [c for c, _ in cnt.most_common(120)]
        sel, tr_score = greedy_select(tr, pool)
        embs = np.stack([cue_emb[c] for c in sel])
        ho_f1 = np.mean([item_f1(embs, it) for it in ho])
        ho_scores.append(ho_f1)
        picks.append((tr_score, ho_f1, sel))
    results[dr] = (float(np.mean(ho_scores)), picks)
    print(f"[{dr}] held-out F1 per fold: "
          + " ".join(f"{p[1]:.3f}" for p in picks)
          + f"  mean={np.mean(ho_scores):.4f}")
    for tr_s, ho_s, sel in picks[:2]:
        print(f"   fold pick (train {tr_s:.3f} / ho {ho_s:.3f}):")
        for c in sel:
            print(f"     - {c}")

overall = np.mean([results[dr][0] for dr in results])
print(f"\nOVERALL CV-honest prior-only Cue-F1 = {overall:.4f}  (real v6 model score: 0.6001)")
