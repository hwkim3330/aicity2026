#!/usr/bin/env python3
"""Regenerate ONLY the 91 PSI-VQA test MCQ answers with the box-aware
pipeline (corrected red-box prompt + CV-detected box timing/location hint +
higher per-frame pixel budget), CV-validated on held-out train items against
the current-pipeline baseline. All other rows are copied byte-identical
from v7.

Run with: TAR_MAX_PIXELS=307200 python3 regen_psi_mcq_boxaware.py
"""
import csv
import json
import os
import sys

sys.path.insert(0, ".")

import prompts
from psi_mcq_cv import VARIANTS, box_hint

prompts.TASK_CONFIG["psi_mcq"]["suffix"] = VARIANTS["box_aware"]

from inference import QwenVLBackend  # noqa: E402

VIDEO_ROOT = "../data/psi_vqa/test_public/videos"
IN_CSV = "../submissions/psi_vqa_submission_v7.csv"
OUT_CSV = "../submissions/psi_vqa_submission_v8.csv"
BOX_INDEX = "../data/psi_vqa/red_box_index2.json"


def resolve_video(video_id):
    rel = video_id.split("PSI/", 1)[-1]
    return os.path.join(VIDEO_ROOT, rel)


def main():
    assert int(os.environ.get("TAR_MAX_PIXELS", 0)) == 307200, \
        "launch with TAR_MAX_PIXELS=307200 to match the CV-validated config"
    existing = {}
    with open(IN_CSV) as f:
        for row in csv.DictReader(f):
            existing[row["item_index"]] = row["prediction"]

    box_index = json.load(open(BOX_INDEX))
    items = json.load(open("../data/psi_vqa/test_public/mcq_questions.json"))["items"]
    print(f"[regen_psi_mcq_boxaware] {len(items)} mcq items", file=sys.stderr)

    checkpoint_path = "../submissions/psi_mcq_boxaware_ckpt.jsonl"
    completed = {}
    if os.path.exists(checkpoint_path):
        for line in open(checkpoint_path):
            row = json.loads(line)
            if row.get("prediction") in ("A", "B", "C", "D"):
                completed[row["item_index"]] = row["prediction"]
                existing[row["item_index"]] = row["prediction"]
    print(f"[resume] loaded {len(completed)} checkpoints", file=sys.stderr)
    backend = QwenVLBackend(quant="bf16")

    n_ok = 0
    ckpt = open(checkpoint_path, "a")
    for i, it in enumerate(items, 1):
        if it["item_index"] in completed:
            n_ok += 1
            continue
        vpath = resolve_video(it["video_id"])
        if not os.path.isfile(vpath):
            print(f"  MISSING {vpath}", file=sys.stderr)
            continue
        question = it["question"] + box_hint(it["video_id"], box_index)
        try:
            ans = backend.answer(vpath, "psi_mcq", question)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {it['item_index']}: {e}", file=sys.stderr)
            continue
        if ans and ans.strip().upper() in "ABCD":
            existing[it["item_index"]] = ans.strip().upper()
            n_ok += 1
            ckpt.write(json.dumps({"item_index": it["item_index"],
                                   "prediction": ans.strip().upper()}) + "\n")
            ckpt.flush()
        else:
            print(f"  EMPTY/BAD {it['item_index']}: {ans!r} (keeping v7 answer)",
                  file=sys.stderr)
        if i % 10 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] ok={n_ok}", file=sys.stderr)
    ckpt.close()

    with open(IN_CSV) as f:
        order = [row["item_index"] for row in csv.DictReader(f)]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_index", "prediction"])
        for idx in order:
            w.writerow([idx, existing[idx]])
    print(f"wrote {OUT_CSV}, regenerated {n_ok}/{len(items)} mcq rows")


if __name__ == "__main__":
    main()
