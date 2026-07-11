#!/usr/bin/env python3
"""Properly-scored MCQ diagnostic against PSI train data (extracts leading
letter from GT's 'B) - text...' format, unlike the quick eval_psi_local.py
which compared raw strings and made everything look wrong)."""
import json, os, random, re, sys
sys.path.insert(0, ".")
from inference import QwenVLBackend

VIDEO_ROOT = "../data/psi_vqa/train/videos"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = 11


def resolve(video_id):
    return os.path.join(VIDEO_ROOT, video_id.split("PSI/", 1)[-1])


def gt_letter(answer):
    m = re.match(r"^([A-D])\)", answer.strip())
    return m.group(1) if m else None


def main():
    d = json.load(open("../data/psi_vqa/train/mcq.json"))
    items = [it for it in d["items"] if os.path.exists(resolve(it["video_id"])) and gt_letter(it["answer"])]
    random.Random(SEED).shuffle(items)
    items = items[:N]

    backend = QwenVLBackend(quant="bf16")
    correct = 0
    pred_dist = {}
    gt_dist = {}
    for it in items:
        vpath = resolve(it["video_id"])
        try:
            pred = backend.answer(vpath, "psi_mcq", it["question"])
        except Exception as e:
            print(f"ERROR {it['video_id']}: {e}", file=sys.stderr)
            continue
        gt = gt_letter(it["answer"])
        ok = pred.strip().upper() == gt
        correct += ok
        pred_dist[pred] = pred_dist.get(pred, 0) + 1
        gt_dist[gt] = gt_dist.get(gt, 0) + 1
        print(f"pred={pred!r} gt={gt!r} ok={ok} q={it['question'][:100]!r}")
    print(f"\nacc: {correct}/{len(items)} = {correct/max(1,len(items)):.3f}")
    print("pred distribution:", pred_dist)
    print("gt distribution:", gt_dist)


if __name__ == "__main__":
    main()
