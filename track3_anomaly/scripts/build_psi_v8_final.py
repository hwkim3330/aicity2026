#!/usr/bin/env python3
"""Build PSI v8 from v7, robust OpenQA priors, and MCQ checkpoints."""
import csv, json
from collections import Counter
import numpy as np
from psi_openqa_cuef1 import _model, parse_cues

BASE = "../submissions/psi_vqa_submission_v7.csv"
OUT = "../submissions/psi_vqa_submission_v8_final.csv"
CKPT = "../submissions/psi_mcq_boxaware_ckpt.jsonl"
DATA = "../data/psi_vqa"


def direction(question):
    if "NOT intend" in question:
        return "not_cross"
    return "uncertain" if "uncertain" in question else "cross"


def score(selected, items, embeddings):
    pred = np.stack([embeddings[c] for c in selected])
    scores = []
    for item in items:
        if item["none"]:
            scores.append(0.0)
            continue
        gt = np.stack([embeddings[c] for c in item["cues"]])
        pairs = sorted(((pred[i] @ gt[j], i, j) for i in range(len(pred))
                        for j in range(len(gt)) if pred[i] @ gt[j] >= .55),
                       reverse=True)
        used_p, used_g = set(), set()
        for _, i, j in pairs:
            if i not in used_p and j not in used_g:
                used_p.add(i); used_g.add(j)
        n = len(used_p)
        p, r = n / len(pred), n / len(gt)
        scores.append(2*p*r/(p+r) if n else 0.0)
    return float(np.mean(scores))


def select_answers():
    items = json.load(open(f"{DATA}/train/open_qa.json"))["items"]
    for x in items:
        x["direction"] = direction(x["question"])
        x["cues"] = parse_cues(x["answer"])
        x["none"] = x["answer"].strip().lower() == "none"
    cues = sorted({c for x in items for c in x["cues"]})
    vecs = _model().encode(cues, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=False)
    embeddings = dict(zip(cues, vecs))
    answers = {}
    for label in ("cross", "not_cross", "uncertain"):
        subset = [x for x in items if x["direction"] == label]
        pool = [c for c, _ in Counter(c for x in subset for c in x["cues"]).most_common(120)]
        selected = []
        for _ in range(2):
            selected.append(max((c for c in pool if c not in selected),
                                key=lambda c: score(selected+[c], subset, embeddings)))
        answers[label] = "\n".join(f"- {c}" for c in selected)
        print(label, f"F1={score(selected, subset, embeddings):.4f}", selected)
    return answers


def main():
    rows = list(csv.DictReader(open(BASE)))
    assert len(rows) == 328
    predictions = {x["item_index"]: x["prediction"] for x in rows}
    answers = select_answers()
    open_items = json.load(open(f"{DATA}/test_public/open_qa_questions.json"))["items"]
    for x in open_items:
        predictions[x["item_index"]] = answers[direction(x["question"])]
    mcq_ids = {x["item_index"] for x in json.load(open(f"{DATA}/test_public/mcq_questions.json"))["items"]}
    done = set()
    try:
        for line in open(CKPT):
            x = json.loads(line)
            assert x["item_index"] in mcq_ids and x["prediction"] in "ABCD"
            predictions[x["item_index"]] = x["prediction"]
            done.add(x["item_index"])
    except FileNotFoundError:
        pass
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_index", "prediction"])
        for x in rows: w.writerow([x["item_index"], predictions[x["item_index"]]])
    final = list(csv.DictReader(open(OUT)))
    assert len(final) == 328 and all(x["prediction"].strip() for x in final)
    assert [x["item_index"] for x in final] == [x["item_index"] for x in rows]
    print(f"wrote {OUT}: 126 OpenQA, {len(done)}/91 MCQ checkpointed")


if __name__ == "__main__": main()
