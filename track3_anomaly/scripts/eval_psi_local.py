#!/usr/bin/env python3
"""Quick local sanity check: run our current bcq/mcq pipeline against PSI's
own labeled train split (real GT) to see actual failure patterns before
guessing at prompt fixes."""
import json, os, random, sys
sys.path.insert(0, ".")
from inference import QwenVLBackend

VIDEO_ROOT = "../data/psi_vqa/train/videos"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SEED = 7


def resolve(video_id):
    rel = video_id.split("PSI/", 1)[-1]
    return os.path.join(VIDEO_ROOT, rel)


def load(path, n):
    d = json.load(open(path))
    items = [it for it in d["items"] if os.path.exists(resolve(it["video_id"]))]
    random.Random(SEED).shuffle(items)
    return items[:n]


def main():
    bcq = load("../data/psi_vqa/train/bcq.json", N)
    mcq = load("../data/psi_vqa/train/mcq.json", N)
    backend = QwenVLBackend(quant="bf16")

    for task_type, items in (("psi_bcq", bcq), ("psi_mcq", mcq)):
        correct = 0
        for it in items:
            vpath = resolve(it["video_id"])
            try:
                pred = backend.answer(vpath, task_type, it["question"])
            except Exception as e:
                print(f"ERROR {it['video_id']}: {e}", file=sys.stderr)
                continue
            gt = it["answer"].strip()
            ok = pred.strip().lower() == gt.strip().lower()
            correct += ok
            print(f"[{task_type}] pred={pred!r} gt={gt!r} ok={ok} q={it['question'][:80]!r}")
        print(f"[{task_type}] acc: {correct}/{len(items)}")


if __name__ == "__main__":
    main()
