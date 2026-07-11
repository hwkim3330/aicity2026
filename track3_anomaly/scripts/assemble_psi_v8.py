#!/usr/bin/env python3
"""Assemble psi_vqa_submission_v8.csv from v7 + whatever MCQ rows the
box-aware regen has checkpointed so far (deadline-safe cutoff assembler).
Validates: 328 rows, header, all item_indexes match v7, only MCQ rows
changed, predictions are single letters A-D."""
import csv
import json
import sys

IN_CSV = "../submissions/psi_vqa_submission_v7.csv"
CKPT = "../submissions/psi_mcq_boxaware_ckpt.jsonl"
OUT_CSV = "../submissions/psi_vqa_submission_v8.csv"

mcq_ids = {it["item_index"] for it in
           json.load(open("../data/psi_vqa/test_public/mcq_questions.json"))["items"]}

rows = list(csv.DictReader(open(IN_CSV)))
assert len(rows) == 328, len(rows)
pred = {r["item_index"]: r["prediction"] for r in rows}

n_new = 0
for line in open(CKPT):
    d = json.loads(line)
    assert d["item_index"] in mcq_ids, d
    assert d["prediction"] in ("A", "B", "C", "D"), d
    if pred[d["item_index"]] != d["prediction"]:
        n_new += 1
    pred[d["item_index"]] = d["prediction"]

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item_index", "prediction"])
    for r in rows:
        w.writerow([r["item_index"], pred[r["item_index"]]])

# validate output
out = list(csv.DictReader(open(OUT_CSV)))
assert len(out) == 328
assert [r["item_index"] for r in out] == [r["item_index"] for r in rows]
changed = [r["item_index"] for r, o in zip(rows, out) if r["prediction"] != o["prediction"]]
assert all(c in mcq_ids for c in changed), "non-MCQ row changed!"
n_ck = sum(1 for _ in open(CKPT))
print(f"wrote {OUT_CSV}: 328 rows, {n_ck} MCQ rows regenerated "
      f"({len(changed)} actually differ from v7), non-MCQ rows byte-identical")
