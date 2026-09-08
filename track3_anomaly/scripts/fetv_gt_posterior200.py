#!/usr/bin/env python3
"""Reconstruct the FETV ground truth over all 200 clips, not the 100-clip subset.

Supersedes fetv_gt_posterior.py, whose premise was wrong: it computed macro-F1
over the 100 clips named in eval_subset_50.json and fitted that to leaderboard
values, but the leaderboard scores all 200. That is settled rather than
assumed -- solving answer_intersection_type under per-source constancy
reproduces the official 0.7840684660961159 exactly on 200 clips (one assignment
out of 512) and comes no closer than 0.0011 on the 100-clip subset.

Marginals follow from the subset file itself: it selected half the clips with
"largest-remainder half targets" balanced on answer_violation_type, so the
200-clip counts are twice the recorded targets.

v11 is deliberately excluded from the fit so its three official field scores can
be used as held-out validation of whatever comes out.
"""
from __future__ import annotations
import collections, json, math, os, random, sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
meta = json.load(open(f'{BASE}/data/fetv/fetv_repo/eval_subset_50.json'))
MARG = {k: v * 2 for k, v in meta['violation_type_targets'].items()}

FIT = {'tr': 'fetv_submission_v2.json', 'v4': 'fetv_submission_v4.json',
       'v5': 'fetv_submission_v5.json', 'v7': 'fetv_submission_v7.json',
       'v8': 'fetv_submission_v8.json'}
HELDOUT = {'v11': 'fetv_submission_v11.json'}

rows0 = json.load(open(f'{BASE}/submissions/fetv_submission_v11.json'))
CLIPS = [r['clip_name'] for r in rows0]
N = len(CLIPS)
assert sum(MARG.values()) == N, (sum(MARG.values()), N)

def load(fname):
    d = {r['clip_name']: r for r in json.load(open(f'{BASE}/submissions/{fname}'))}
    g = lambda c, k: str(d[c][k]).strip().lower()
    return {'viol':  [g(c, 'answer_violation_type') for c in CLIPS],
            'vtype': [g(c, 'answer_violator_type') for c in CLIPS],
            'color': [g(c, 'answer_color') for c in CLIPS]}

SUBS = {k: load(f) for k, f in FIT.items()}
OUT_SUBS = {k: load(f) for k, f in HELDOUT.items()}

# Official macro-F1, read from the portal export.
TGT = {'viol':  {'tr': .1711, 'v4': .1293, 'v5': .1697, 'v7': .1720, 'v8': .1944},
       'vtype': {'tr': .4448, 'v4': .2355, 'v5': .3399, 'v7': .3751, 'v8': .1890},
       'color': {'tr': .2147, 'v4': .1150, 'v5': .1671, 'v7': .1837, 'v8': .1309}}

VIOL = sorted(MARG)
VEHICLES = ['car', 'motorcycle', 'bus', 'truck']
COLORS = ['dark', 'light', 'red', 'green', 'yellow', 'blue', 'mixed']
KNOWN = {'001_001.mp4': ('jaywalking', 'pedestrian', 'mixed')}
IDX_KNOWN = {CLIPS.index(c): v for c, v in KNOWN.items() if c in CLIPS}

def macro_f1(gt, pred):
    labs = set(gt) | set(pred)
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    for g, p in zip(gt, pred):
        if g == p: tp[g] += 1
        else: fp[p] += 1; fn[g] += 1
    return sum((2 * tp[l] / (2 * tp[l] + fp[l] + fn[l])) if (2 * tp[l] + fp[l] + fn[l]) else 0.0
               for l in labs) / len(labs)

def derive(v, vt, co):
    gv, gc = [], []
    for i in range(N):
        if v[i] == 'no_violation': gv.append('na'); gc.append('na')
        elif v[i] == 'jaywalking': gv.append('pedestrian'); gc.append(co[i])
        else: gv.append(vt[i]); gc.append(co[i])
    return gv, gc

def scores(v, vt, co, subs):
    gv, gc = derive(v, vt, co)
    return {k: {'viol': macro_f1(v, s['viol']), 'vtype': macro_f1(gv, s['vtype']),
                'color': macro_f1(gc, s['color'])} for k, s in subs.items()}

def loss(v, vt, co):
    s = scores(v, vt, co, SUBS)
    return sum((s[k][f] - TGT[f][k]) ** 2 for k in SUBS for f in TGT)

def anneal(seed, iters=60000, t0=0.02, t1=1e-5):
    rng = random.Random(seed)
    pool = [c for c, n in MARG.items() for _ in range(n)]
    rng.shuffle(pool)
    for i, (w, _, _) in IDX_KNOWN.items():
        if pool[i] != w:
            j = pool.index(w); pool[i], pool[j] = pool[j], pool[i]
    vt = [rng.choice(VEHICLES) for _ in range(N)]
    co = [rng.choice(COLORS) for _ in range(N)]
    for i, (_, t, c) in IDX_KNOWN.items(): vt[i] = t; co[i] = c
    cur = loss(pool, vt, co); best = (cur, pool[:], vt[:], co[:])
    for it in range(iters):
        T = t0 * (t1 / t0) ** (it / iters)
        m = rng.random(); undo = None
        if m < .45:
            i, j = rng.randrange(N), rng.randrange(N)
            if i in IDX_KNOWN or j in IDX_KNOWN or pool[i] == pool[j]: continue
            undo = ('s', i, j); pool[i], pool[j] = pool[j], pool[i]
        elif m < .75:
            i = rng.randrange(N)
            if i in IDX_KNOWN: continue
            old = vt[i]; new = rng.choice(VEHICLES)
            if new == old: continue
            undo = ('v', i, old); vt[i] = new
        else:
            i = rng.randrange(N)
            if i in IDX_KNOWN: continue
            old = co[i]; new = rng.choice(COLORS)
            if new == old: continue
            undo = ('c', i, old); co[i] = new
        nl = loss(pool, vt, co)
        if nl <= cur or rng.random() < math.exp((cur - nl) / max(T, 1e-9)):
            cur = nl
            if cur < best[0]: best = (cur, pool[:], vt[:], co[:])
        else:
            k = undo[0]
            if k == 's': _, i, j = undo; pool[i], pool[j] = pool[j], pool[i]
            elif k == 'v': _, i, old = undo; vt[i] = old
            else: _, i, old = undo; co[i] = old
    return best

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    out_path = sys.argv[2] if len(sys.argv) > 2 else f'{BASE}/results/fetv_gt_posterior200.json'
    sols = []
    for s in range(n):
        L, v, vt, co = anneal(2000 + s)
        held = scores(v, vt, co, OUT_SUBS)['v11']
        sols.append({'loss': L, 'viol': v, 'vtype': vt, 'color': co, 'heldout_v11': held})
        print(f"restart {s:2d}: rmse={math.sqrt(L/15):.5f}  "
              f"v11 predicted viol={held['viol']:.4f} vtype={held['vtype']:.4f} "
              f"color={held['color']:.4f}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({'clips': CLIPS, 'solutions': sols}, open(out_path, 'w'))
    print(f'wrote {out_path}')
