#!/usr/bin/env python3
"""Merge the four PSI-VQA public question files into the single
`tao-vl-reason-v1.0` envelope that `make_submission.py` consumes.

PSI ships one file per task and labels items with the generic task types
`bcq` / `mcq` / `open_qa` / `temporal_localization`. The prompt table in
`prompts.py` routes PSI's binary and multiple-choice questions through
dedicated `psi_bcq` / `psi_mcq` entries, because the generic Track 3 suffixes
assume a collision that PSI clips do not contain. This script performs that
remapping explicitly, so the routing is visible instead of implied.

The original merged file was produced ad hoc during the challenge and never
committed; this restores it. Item order follows the shipped submission CSVs:
bcq, mcq, open_qa, temporal_localization.

    python3 build_psi_test_json.py --out ../data/psi_vqa/test_public/psi_test.json
"""
import argparse
import json
import os

SOURCES = [
    ("bcq_questions.json", "psi_bcq"),
    ("mcq_questions.json", "psi_mcq"),
    ("open_qa_questions.json", "open_qa"),
    ("temporal_localization_questions.json", "temporal_localization"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/psi_vqa/test_public")
    ap.add_argument("--out", default="../data/psi_vqa/test_public/psi_test.json")
    args = ap.parse_args()

    items, counts = [], {}
    for fname, routed_type in SOURCES:
        path = os.path.join(args.src, fname)
        doc = json.load(open(path))
        for it in doc["items"]:
            items.append({
                "item_index": it["item_index"],
                "task_type": routed_type,
                "source_task_type": it["task_type"],
                "video_id": it["video_id"].split("PSI/", 1)[-1],
                "question": it["question"],
                "answer": "",
            })
        counts[routed_type] = len(doc["items"])

    doc = {
        "format": "tao-vl-reason-v1.0",
        "metadata": {
            "type": "questions",
            "task": "psi_vqa_merged",
            "source": "ise-ice-lab/PSI_VQA test_public",
            "note": ("bcq -> psi_bcq and mcq -> psi_mcq are prompt-routing "
                     "renames; source_task_type preserves the original label"),
        },
        "media_root": "videos",
        "items": items,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(doc, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: {len(items)} items {counts}")


if __name__ == "__main__":
    main()
