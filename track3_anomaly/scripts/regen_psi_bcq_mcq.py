#!/usr/bin/env python3
"""Regenerate PSI-VQA's bcq+mcq test answers with self-consistency voting
(samples=5), using the current fixed inference.py. Merges into the v3 csv
(which already has the fps-fixed temporal answers) to produce v4."""
import csv
import json
import os
import sys

sys.path.insert(0, ".")
from inference import QwenVLBackend

VIDEO_ROOT = "../data/psi_vqa/test_public/videos"
IN_CSV = "../submissions/psi_vqa_submission_v3.csv"
OUT_CSV = "../submissions/psi_vqa_submission_v4.csv"
SAMPLES = 5


def resolve_video(video_id):
    rel = video_id.split("PSI/", 1)[-1]
    return os.path.join(VIDEO_ROOT, rel)


def main():
    existing = {}
    with open(IN_CSV) as f:
        for row in csv.DictReader(f):
            existing[row["item_index"]] = row["prediction"]

    backend = QwenVLBackend(quant="bf16")

    items = []
    # psi_bcq / psi_mcq: PSI-specific pedestrian-crossing-intent prompts
    # (see prompts.py) -- the generic Track3 "bcq"/"mcq" prompts assume a
    # collision-forensics clip and were a cross-track mismatch here.
    # samples=5 only takes effect for psi_bcq (self_consistency=True);
    # psi_mcq is greedy single-sample (voting confirmed to hurt MCQ).
    for task_type, path in (("psi_bcq", "../data/psi_vqa/test_public/bcq_questions.json"),
                             ("psi_mcq", "../data/psi_vqa/test_public/mcq_questions.json")):
        q = json.load(open(path))
        for it in q["items"]:
            items.append((task_type, it))
    print(f"[regen_psi_bcq_mcq] {len(items)} items", file=sys.stderr)

    n_ok = 0
    for i, (task_type, it) in enumerate(items, 1):
        vpath = resolve_video(it["video_id"])
        if not os.path.isfile(vpath):
            print(f"  MISSING {vpath}", file=sys.stderr)
            continue
        try:
            ans = backend.answer(vpath, task_type, it["question"], samples=SAMPLES)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {it['item_index']}: {e}", file=sys.stderr)
            continue
        existing[it["item_index"]] = ans
        n_ok += 1
        if i % 10 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] ok={n_ok}", file=sys.stderr)

    with open(IN_CSV) as f:
        order = [row["item_index"] for row in csv.DictReader(f)]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_index", "prediction"])
        for idx in order:
            w.writerow([idx, existing[idx]])
    print(f"wrote {OUT_CSV}, updated {n_ok}/{len(items)} bcq/mcq rows")


if __name__ == "__main__":
    main()
