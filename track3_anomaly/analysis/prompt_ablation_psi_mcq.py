#!/usr/bin/env python3
"""Controlled prompt ablation for PSI-VQA multiple-choice, scored on the same
held-out training items with the same backbone, precision, frame budget, and
decoding settings.

Three prompt programs are compared:

  generic   the shared Track 3 multiple-choice prompt, with no PSI-specific
            instructions — the "system without routing" condition
  routed    the shipped `psi_mcq` prompt program from prompts.py, which adds
            PSI-specific crossing-intent instructions
  boxaware  routed plus an explicit instruction to re-locate the red-boxed
            pedestrian before answering (post-deadline variant)

Only items answered under all three conditions enter the comparison, so every
number is a paired measurement. Discordant pairs are reported with an exact
McNemar test, because the held-out sample is small enough that raw accuracy
differences are easy to over-read.

Inputs are the per-condition JSONL files written by `scripts/psi_mcq_cv.py`:

    cd track3_anomaly/scripts
    python3 psi_mcq_cv.py --n 60 --seed 11 --out ../psi_mcq_cv_results/mcq_generic.jsonl  --variant generic
    python3 psi_mcq_cv.py --n 60 --seed 11 --out ../psi_mcq_cv_results/mcq_baseline.jsonl
    python3 psi_mcq_cv.py --n 60 --seed 11 --out ../psi_mcq_cv_results/mcq_boxaware.jsonl --variant boxaware

    cd ../analysis && python3 prompt_ablation_psi_mcq.py
"""
import argparse
import itertools
import json
import os
from math import comb

CONDITIONS = [
    ("generic", "mcq_generic.jsonl", "shared Track 3 MCQ prompt, no PSI routing"),
    ("routed", "mcq_baseline.jsonl", "shipped psi_mcq prompt program"),
    ("boxaware", "mcq_boxaware.jsonl", "routed + explicit red-box re-location"),
]


def exact_mcnemar(b, c):
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../psi_mcq_cv_results")
    ap.add_argument("--out", default="../results/prompt_ablation_psi_mcq.json")
    args = ap.parse_args()

    loaded = {}
    for name, fname, _ in CONDITIONS:
        path = os.path.join(args.results_dir, fname)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run psi_mcq_cv.py for '{name}' first")
        rows = [json.loads(line) for line in open(path) if line.strip()]
        loaded[name] = {r["item_index"]: bool(r["ok"]) for r in rows}

    paired = set.intersection(*(set(v) for v in loaded.values()))
    paired = sorted(paired)

    print(f"per-file totals (unpaired):")
    for name, fname, desc in CONDITIONS:
        v = loaded[name]
        print(f"  {name:<9} n={len(v):<4} acc={sum(v.values())}/{len(v)} "
              f"= {sum(v.values())/len(v):.4f}   ({desc})")

    print(f"\npaired held-out subset: n={len(paired)}")
    acc = {}
    for name, _, _ in CONDITIONS:
        ok = sum(loaded[name][k] for k in paired)
        acc[name] = ok / len(paired)
        print(f"  {name:<9} {ok}/{len(paired)} = {acc[name]:.4f}")

    print("\npairwise exact McNemar (b = first correct & second wrong):")
    tests = []
    for a, b in itertools.combinations([c[0] for c in CONDITIONS], 2):
        n_ab = sum(loaded[a][k] and not loaded[b][k] for k in paired)
        n_ba = sum(loaded[b][k] and not loaded[a][k] for k in paired)
        p = exact_mcnemar(n_ab, n_ba)
        tests.append({"a": a, "b": b, "a_only": n_ab, "b_only": n_ba, "p_two_sided": p})
        print(f"  {a:<9} vs {b:<9}  {a}-only={n_ab}  {b}-only={n_ba}  p={p:.4f}")

    summary = {
        "n_paired": len(paired),
        "conditions": {name: {"description": desc, "accuracy": acc[name],
                              "correct": sum(loaded[name][k] for k in paired)}
                       for name, _, desc in CONDITIONS},
        "mcnemar": tests,
        "caveat": ("The paired subset holds only %d items. Directions are "
                   "informative; absolute differences are not precise."
                   % len(paired)),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
