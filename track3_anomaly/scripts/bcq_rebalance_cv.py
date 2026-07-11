#!/usr/bin/env python3
"""Validate vote-margin-based BCQ Yes->No rebalancing on the labeled train
split before applying it to the 55-item test predictions.

Rules evaluated (all operate on 5-sample vote tallies):
  - majority (baseline; what production ships)
  - thresh_k: predict Yes iff yes_count >= k  (k=3 == majority sans ties)
  - ratio_r:  start from majority, then flip the lowest-yes-count Yes
              predictions to No (ties: greedy-sample==No first) until the
              batch Yes rate is <= r. Batch-level rule, applied within
              each evaluation batch exactly as it would be applied to the
              55-item test set.

Evaluation:
  1. Full-train macro-F1 per rule + bootstrap CI of delta vs baseline.
  2. Repeated random 55-item holdouts (test-sized): per-rule delta vs
     baseline on the holdout only. Mean/std/frac-positive over reps.
  3. Honest nested selection: pick best rule on the other 190, apply to
     the held-out 55; report the achieved delta distribution.
"""
import json
import random
import sys
from collections import Counter

TRAIN = "../submissions/bcq_votes_train.jsonl"
REPS = 1000
HOLDOUT = 55
SEED = 20260711


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            votes = [v for v in r["votes"] if v in ("Yes", "No")]
            r["yes_count"] = sum(v == "Yes" for v in votes)
            r["n_votes"] = len(votes)
            r["greedy"] = r["votes"][0] if r["votes"] else None
            recs.append(r)
    return recs


def macro_f1(preds, gts):
    f1s = []
    for cls in ("Yes", "No"):
        tp = sum(p == cls and g == cls for p, g in zip(preds, gts))
        fp = sum(p == cls and g != cls for p, g in zip(preds, gts))
        fn = sum(p != cls and g == cls for p, g in zip(preds, gts))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return sum(f1s) / 2


def pred_majority(batch):
    return [r["majority"] for r in batch]


def pred_thresh(batch, k):
    return ["Yes" if r["yes_count"] >= k else "No" for r in batch]


def flip_order_key(r):
    # weakest Yes first: fewest yes votes, then greedy==No first
    return (r["yes_count"], 0 if r["greedy"] == "No" else 1)


def pred_ratio(batch, r_target):
    preds = {id(x): x["majority"] for x in batch}
    yes_items = sorted((x for x in batch if x["majority"] == "Yes"),
                       key=flip_order_key)
    n = len(batch)
    n_yes = len(yes_items)
    i = 0
    while n_yes / n > r_target and i < len(yes_items):
        preds[id(yes_items[i])] = "No"
        n_yes -= 1
        i += 1
    return [preds[id(x)] for x in batch]


def main():
    recs = load(TRAIN)
    gts = [r["gt"] for r in recs]
    print(f"n={len(recs)}  GT: {Counter(gts)}")
    base_preds = pred_majority(recs)
    print(f"majority preds: {Counter(base_preds)} "
          f"(Yes rate {sum(p=='Yes' for p in base_preds)/len(recs):.3f})")
    print(f"vote hist (yes_count): {sorted(Counter(r['yes_count'] for r in recs).items())}")
    acc = sum(p == g for p, g in zip(base_preds, gts)) / len(recs)
    print(f"baseline majority: acc={acc:.4f} macroF1={macro_f1(base_preds, gts):.4f}")
    # informativeness of margin: GT-No fraction among Yes-preds by yes_count
    for yc in range(6):
        sub = [r for r in recs if r["majority"] == "Yes" and r["yes_count"] == yc]
        if sub:
            frac_no = sum(r["gt"] == "No" for r in sub) / len(sub)
            print(f"  yes-pred items with yes_count={yc}: n={len(sub)}, GT-No frac={frac_no:.2f}")

    rules = [("majority", pred_majority)]
    for k in (3, 4, 5):
        rules.append((f"thresh_{k}", lambda b, k=k: pred_thresh(b, k)))
    for rt in (0.65, 0.62, 0.60, 0.58, 0.55, 0.52, 0.50):
        rules.append((f"ratio_{rt:.2f}", lambda b, rt=rt: pred_ratio(b, rt)))

    print("\n== full-train macro-F1 (+bootstrap 95% CI of delta vs majority) ==")
    rng = random.Random(SEED)
    n = len(recs)
    boot_idx = [[rng.randrange(n) for _ in range(n)] for _ in range(2000)]
    base_full = macro_f1(base_preds, gts)
    for name, fn in rules:
        preds = fn(recs)
        f1 = macro_f1(preds, gts)
        deltas = []
        for idx in boot_idx:
            b = [recs[i] for i in idx]
            g = [gts[i] for i in idx]
            deltas.append(macro_f1(fn(b), g) - macro_f1(pred_majority(b), g))
        deltas.sort()
        lo, hi = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))]
        print(f"  {name:12s} F1={f1:.4f} delta={f1-base_full:+.4f} CI[{lo:+.4f},{hi:+.4f}]")

    print(f"\n== repeated {HOLDOUT}-item holdout ({REPS} reps): delta vs majority ==")
    rng = random.Random(SEED + 1)
    splits = []
    for _ in range(REPS):
        idx = list(range(n))
        rng.shuffle(idx)
        splits.append((idx[:HOLDOUT], idx[HOLDOUT:]))
    for name, fn in rules[1:]:
        deltas = []
        for hold, _ in splits:
            b = [recs[i] for i in hold]
            g = [gts[i] for i in hold]
            deltas.append(macro_f1(fn(b), g) - macro_f1(pred_majority(b), g))
        mean = sum(deltas) / len(deltas)
        var = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
        fpos = sum(d > 0 for d in deltas) / len(deltas)
        fneg = sum(d < 0 for d in deltas) / len(deltas)
        print(f"  {name:12s} mean={mean:+.4f} std={var**0.5:.4f} "
              f"P(>0)={fpos:.2f} P(<0)={fneg:.2f}")

    print("\n== nested: tune rule on 190, apply to held-out 55 ==")
    cand = rules[1:]
    deltas, picks = [], Counter()
    for hold, tune in splits:
        tb = [recs[i] for i in tune]
        tg = [gts[i] for i in tune]
        tune_base = macro_f1(pred_majority(tb), tg)
        best_name, best_fn, best_f1 = "majority", pred_majority, tune_base
        for name, fn in cand:
            f1 = macro_f1(fn(tb), tg)
            if f1 > best_f1:
                best_name, best_fn, best_f1 = name, fn, f1
        picks[best_name] += 1
        hb = [recs[i] for i in hold]
        hg = [gts[i] for i in hold]
        deltas.append(macro_f1(best_fn(hb), hg) - macro_f1(pred_majority(hb), hg))
    mean = sum(deltas) / len(deltas)
    var = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
    fpos = sum(d > 0 for d in deltas) / len(deltas)
    fneg = sum(d < 0 for d in deltas) / len(deltas)
    print(f"  nested delta: mean={mean:+.4f} std={var**0.5:.4f} "
          f"P(>0)={fpos:.2f} P(<0)={fneg:.2f}")
    print(f"  rule picks: {picks.most_common()}")


if __name__ == "__main__":
    main()
