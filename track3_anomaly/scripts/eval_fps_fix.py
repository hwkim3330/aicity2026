#!/usr/bin/env python3
"""Quick accuracy check on a train-set sample, to measure the impact of the
fps/video_metadata fix in inference.py's answer(). Loads the model once and
scores N stratified bcq+mcq items against ground truth answers."""
import json
import os
import random
import sys

from inference import QwenVLBackend

VIDEO_ROOT = "../data/videos"
N_PER_TASK = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 1
FEWSHOT = os.environ.get("TAR_FEWSHOT_EVAL", "0") == "1"
SEED = 42

# The few-shot demo examples in inference.py's FEWSHOT_EXAMPLES are pulled
# from this video -- exclude it from our own eval sample so we're not
# "testing" on something the model was just shown as a worked example.
_FEWSHOT_DEMO_VIDEO = "Accident-Bench/land_space/long/videos/000001.mp4"


def load_sample(path, task_type, n):
    d = json.load(open(path))
    items = d["items"]
    rng = random.Random(SEED)
    rng.shuffle(items)
    out = []
    for it in items:
        if not it.get("answer"):
            continue
        if it["video_id"] == _FEWSHOT_DEMO_VIDEO:
            continue
        if not os.path.exists(f"{VIDEO_ROOT}/{it['video_id']}"):
            continue
        out.append(it)
        if len(out) >= n:
            break
    return out


def main():
    bcq = load_sample("../data/train/bcq.json", "bcq", N_PER_TASK)
    mcq = load_sample("../data/train/mcq.json", "mcq", N_PER_TASK)
    print(f"[eval] {len(bcq)} bcq + {len(mcq)} mcq items, samples={SAMPLES}", file=sys.stderr)

    backend = QwenVLBackend(quant="bf16")

    correct = 0
    total = 0
    for task_type, items in (("bcq", bcq), ("mcq", mcq)):
        n_correct = 0
        for it in items:
            video_path = f"{VIDEO_ROOT}/{it['video_id']}"
            try:
                pred = backend.answer(video_path, task_type, it["question"], samples=SAMPLES, fewshot=FEWSHOT)
            except Exception as e:  # noqa: BLE001
                print(f"[eval] ERROR on {it['video_id']}: {e}", file=sys.stderr)
                continue
            gt = it["answer"].strip()
            ok = pred.strip().lower() == gt.strip().lower()
            n_correct += ok
            total += 1
            correct += ok
            print(f"[{task_type}] pred={pred!r} gt={gt!r} ok={ok}", file=sys.stderr)
        print(f"[eval] {task_type} acc: {n_correct}/{len(items)} = {n_correct/max(1,len(items)):.3f}")

    print(f"[eval] OVERALL acc: {correct}/{total} = {correct/max(1,total):.3f}")


if __name__ == "__main__":
    main()
