#!/usr/bin/env python3
"""One-off: regenerate PSI-VQA's 56 temporal_localization test answers using
the current (fixed) inference.py -- decord backend + real fps extraction via
return_video_metadata=True. The original psi_vqa_submission.csv was produced
by an older inference.py that used torchvision and could not infer fps
("Defaulting to fps=24" while the real videos are 30fps), which breaks
Qwen3VL's M-RoPE temporal grounding -- the model's answers were suspiciously
always "start": "00:00" regardless of when the real event occurred. Merges
the new temporal answers into the existing v2 csv (leaving its already-
competitive bcq/mcq/open_qa rows untouched) to produce v3.
"""
import csv
import json
import os
import sys

sys.path.insert(0, ".")
from inference import QwenVLBackend

Q_PATH = "../data/psi_vqa/test_public/temporal_localization_questions.json"
VIDEO_ROOT = "../data/psi_vqa/test_public/videos"
IN_CSV = "../submissions/psi_vqa_submission_v2.csv"
OUT_CSV = "../submissions/psi_vqa_submission_v3.csv"


def resolve_video(video_id):
    # video_id like "PSI/temporal/video_0205.mp4" -> videos/temporal/video_0205.mp4
    rel = video_id.split("PSI/", 1)[-1]
    return os.path.join(VIDEO_ROOT, rel)


def main():
    q = json.load(open(Q_PATH))
    items = q["items"]
    print(f"[regen_psi_temporal] {len(items)} temporal items", file=sys.stderr)

    existing = {}
    with open(IN_CSV) as f:
        for row in csv.DictReader(f):
            existing[row["item_index"]] = row["prediction"]

    backend = QwenVLBackend(quant="bf16")

    n_ok = 0
    for i, it in enumerate(items, 1):
        vpath = resolve_video(it["video_id"])
        if not os.path.isfile(vpath):
            print(f"  MISSING {vpath}", file=sys.stderr)
            continue
        try:
            ans = backend.answer(vpath, "temporal_localization", it["question"])
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
    print(f"wrote {OUT_CSV}, updated {n_ok}/{len(items)} temporal rows")


if __name__ == "__main__":
    main()
