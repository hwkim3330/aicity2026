#!/usr/bin/env python3
"""Reconstruct the shipped PSI-VQA v7 Open QA rows from v6.

Provenance note. The official v7 submission replaced all 126 Open QA answers
with one fixed two-cue answer per question direction, but the one-off script
that performed that substitution was never committed; only its output survived
in `submissions/psi_vqa_submission_v7.csv`. This script restores that step so
the v6 -> v7 transition is reproducible end to end. It is verified by
`--verify`, which regenerates v7 from v6 and compares against the shipped file
record by record.

What v7 actually changed. Compared with v6, the shipped v7 differs in exactly
the 126 `open_qa` rows. The BCQ vote-margin rebalance described in
`make_psi_v7_bcq_rebalance.py` did *not* ship: all 55 BCQ rows in v7 are
identical to v6. The 91 MCQ rows and 56 temporal rows are likewise unchanged.

Cue selection. The two cues per direction were chosen by greedy forward
selection on the PSI training split under the pessimistic Cue-F1 variant
(one-to-one greedy matching, macro averaging) — the same procedure as
`psi_openqa_prior_robust.py`, run at k=2 on all training items rather than
per fold. `--derive-cues` re-runs that selection and reports whether it
reproduces the shipped strings; it needs sentence-transformers and is not
required for `--verify`.

    python3 make_psi_v7_openqa_prior.py --verify
    python3 make_psi_v7_openqa_prior.py --out ../submissions/reproduced_psi_v7.csv
"""
import argparse
import csv
import json
import os
import sys

V6 = "../submissions/psi_vqa_submission_v6.csv"
V7 = "../submissions/psi_vqa_submission_v7.csv"
OPENQA_Q = "../data/psi_vqa/test_public/open_qa_questions.json"

# The exact strings shipped in psi_vqa_submission_v7.csv.
SHIPPED_CUES = {
    "cross": [
        "Pedestrian had left the sidewalk and/or entered the roadway or left lane.",
        "Pedestrian was crossing perpendicular to the flow of traffic / the car's trajectory.",
    ],
    "not_cross": [
        "Pedestrian is standing or stopped rather than moving into the road.",
        "Pedestrian was standing or waiting for cars to pass rather than crossing.",
    ],
    "uncertain": [
        "Pedestrian was moving left-to-right or toward the right side of the road.",
        "Pedestrian appeared to be crossing or walking into the road.",
    ],
}


def direction(question):
    """Question-direction router, identical to psi_openqa_prior_robust.py."""
    if "NOT intend" in question:
        return "not_cross"
    if "uncertain" in question:
        return "uncertain"
    return "cross"


def render(cues):
    return "\n".join("- " + c for c in cues)


def build(v6_path, openqa_path, cues):
    questions = json.load(open(openqa_path))["items"]
    replacement = {it["item_index"]: render(cues[direction(it["question"])])
                   for it in questions}
    rows = list(csv.DictReader(open(v6_path)))
    out, n = [], 0
    for r in rows:
        pred = r["prediction"]
        if r["item_index"] in replacement:
            pred = replacement[r["item_index"]]
            n += 1
        out.append((r["item_index"], pred))
    return out, n


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_index", "prediction"])
        w.writerows(rows)


def derive_cues(k=2):
    """Re-run the greedy k-cue selection on the training split."""
    import numpy as np
    from collections import Counter
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from psi_openqa_cuef1 import _model, parse_cues

    items = json.load(open("../data/psi_vqa/train/open_qa.json"))["items"]
    for it in items:
        it["dir"] = direction(it["question"])
        it["gt_cues"] = parse_cues(it["answer"])
        it["gt_none"] = it["answer"].strip().lower() == "none"

    m = _model()
    vocab = sorted({c for it in items for c in it["gt_cues"]})
    emb = m.encode(vocab, batch_size=256, normalize_embeddings=True)
    cue_emb = dict(zip(vocab, emb))
    THR = 0.55

    def score(sel, subset):
        pe = np.stack([cue_emb[c] for c in sel])
        f1s = []
        for it in subset:
            if it["gt_none"]:
                f1s.append(0.0)
                continue
            ge = np.stack([cue_emb[c] for c in it["gt_cues"]])
            sim = pe @ ge.T
            pairs = sorted(((sim[i, j], i, j)
                            for i in range(sim.shape[0]) for j in range(sim.shape[1])
                            if sim[i, j] >= THR), reverse=True)
            up, ug, n = set(), set(), 0
            for _, i, j in pairs:
                if i in up or j in ug:
                    continue
                up.add(i); ug.add(j); n += 1
            p, r = n / len(sel), n / len(ge)
            f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        return float(np.mean(f1s))

    derived = {}
    for d in ("cross", "not_cross", "uncertain"):
        subset = [it for it in items if it["dir"] == d]
        pool = [c for c, _ in Counter(c for it in subset
                                      for c in it["gt_cues"]).most_common(120)]
        sel = []
        while len(sel) < k:
            best_c, best_s = None, -1.0
            for c in pool:
                if c in sel:
                    continue
                s = score(sel + [c], subset)
                if s > best_s:
                    best_s, best_c = s, c
            sel.append(best_c)
        derived[d] = sel
        match = "MATCHES shipped" if sel == SHIPPED_CUES[d] else "DIFFERS from shipped"
        print(f"[{d}] train Cue-F1={score(sel, subset):.4f} — {match}")
        for c in sel:
            print(f"    - {c}")
    return derived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", default=V6)
    ap.add_argument("--v7", default=V7)
    ap.add_argument("--openqa-questions", default=OPENQA_Q)
    ap.add_argument("--out", default=None,
                    help="write the reconstruction here (never overwrites the "
                         "official file unless you point it there yourself)")
    ap.add_argument("--verify", action="store_true",
                    help="rebuild from v6 and diff against the shipped v7")
    ap.add_argument("--derive-cues", action="store_true",
                    help="re-run greedy cue selection on the training split")
    args = ap.parse_args()

    if args.derive_cues:
        derive_cues()
        return

    rows, n = build(args.v6, args.openqa_questions, SHIPPED_CUES)
    print(f"rebuilt {len(rows)} rows, {n} open_qa rows replaced")

    if args.out:
        write_csv(rows, args.out)
        print(f"wrote {args.out}")

    if args.verify:
        shipped = {r["item_index"]: r["prediction"]
                   for r in csv.DictReader(open(args.v7))}
        built = dict(rows)
        if set(shipped) != set(built):
            sys.exit(f"FAIL: item sets differ "
                     f"({len(shipped)} shipped vs {len(built)} rebuilt)")
        bad = [k for k in shipped if shipped[k] != built[k]]
        if bad:
            print(f"FAIL: {len(bad)} rows differ, e.g. {bad[:3]}")
            for k in bad[:2]:
                print(f"  shipped: {shipped[k]!r}\n  rebuilt: {built[k]!r}")
            sys.exit(1)
        print(f"OK: reconstruction matches {args.v7} on all {len(shipped)} records")


if __name__ == "__main__":
    main()
