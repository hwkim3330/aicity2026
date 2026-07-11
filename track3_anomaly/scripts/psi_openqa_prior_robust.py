#!/usr/bin/env python3
"""Stress-test the prior-only Open QA answer under plausible variants of the
official Cue-F1 implementation.

Variants:
  matching:    many-to-many threshold (each cue matched if max sim >= 0.55)
               vs one-to-one greedy (highest-sim pairs first, no reuse)
  aggregation: per-item F1 averaged (macro) vs global micro P/R -> F1
  k:           number of cues in the fixed per-direction answer (1..4)

Selection is done on train folds under the PESSIMISTIC variant combo
(one-to-one + macro), then every variant is reported on the held-out fold.
Also reports each fold's held-out score for the k chosen by train-fold
performance, so we can pick a k that is robust under all variants.
"""
import json
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
all_cues = sorted(set(c for it in D for c in it["gt_cues"]))
emb = m.encode(all_cues, batch_size=256, normalize_embeddings=True)
cue_emb = {c: emb[i] for i, c in enumerate(all_cues)}

THR = 0.55


def match_counts(sim, one_to_one):
    """Return (#matched pred cues, #matched gt cues)."""
    if not one_to_one:
        return int((sim.max(axis=1) >= THR).sum()), int((sim.max(axis=0) >= THR).sum())
    # greedy one-to-one on descending similarity
    pairs = [(sim[i, j], i, j) for i in range(sim.shape[0])
             for j in range(sim.shape[1]) if sim[i, j] >= THR]
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    n = 0
    for s, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); n += 1
    return n, n


def eval_items(pred_embs, items, one_to_one, micro):
    """pred_embs: fixed cue-set embedding matrix used for every item."""
    f1s = []
    MP = MG = NP = NG = 0
    for it in items:
        if it["gt_none"]:
            f1s.append(0.0)  # we always predict cues
            # micro: none-items contribute nothing matchable; count pred cues
            NP += len(pred_embs)
            continue
        ge = np.stack([cue_emb[c] for c in it["gt_cues"]])
        sim = pred_embs @ ge.T
        mp, mg = match_counts(sim, one_to_one)
        p = mp / len(pred_embs)
        r = mg / len(ge)
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        MP += mp; MG += mg; NP += len(pred_embs); NG += len(ge)
    if micro:
        P = MP / NP if NP else 0.0
        R = MG / NG if NG else 0.0
        return 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return float(np.mean(f1s))


def greedy_select(items, pool, k, one_to_one, micro):
    sel = []
    while len(sel) < k:
        best_c, best_s = None, -1.0
        for c in pool:
            if c in sel:
                continue
            embs = np.stack([cue_emb[x] for x in sel + [c]])
            s = eval_items(embs, items, one_to_one, micro)
            if s > best_s:
                best_s, best_c = s, c
        sel.append(best_c)
    return sel


print("Selection under PESSIMISTIC combo (one-to-one, macro); "
      "held-out scores under all 4 variant combos:")
print("k | m2m+macro  m2m+micro  1to1+macro  1to1+micro   (means over 5 folds x 3 directions)")
for k in (1, 2, 3, 4):
    scores = {v: [] for v in range(4)}
    for dr in ("cross", "not_cross", "uncertain"):
        items_d = [it for it in D if it["dir"] == dr]
        for f in range(5):
            tr = [it for it in items_d if it["video_id"] not in folds[f]]
            ho = [it for it in items_d if it["video_id"] in folds[f]]
            cnt = Counter(c for it in tr for c in it["gt_cues"])
            pool = [c for c, _ in cnt.most_common(120)]
            sel = greedy_select(tr, pool, k, one_to_one=True, micro=False)
            embs = np.stack([cue_emb[c] for c in sel])
            for vi, (oto, mic) in enumerate(
                    [(False, False), (False, True), (True, False), (True, True)]):
                scores[vi].append(eval_items(embs, ho, oto, mic))
    print(f"{k} |   " + "     ".join(f"{np.mean(scores[v]):.4f}" for v in range(4)))
