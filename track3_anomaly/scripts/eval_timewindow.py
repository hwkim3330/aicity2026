#!/usr/bin/env python3
"""Isolate the effect of the new video_start/video_end timestamp-windowing
fix: only test MCQ items whose question text actually references a MM:SS
window, comparing accuracy with the fix active vs a monkey-patched no-op
version (uniform full-clip sampling, the old behavior)."""
import json, os, random, sys
sys.path.insert(0, ".")
import inference
from inference import QwenVLBackend

VIDEO_ROOT = "../data/videos"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SEED = 3


def has_window(q):
    s, e = QwenVLBackend._extract_time_window(q)
    return s is not None


def load_sample(path, n):
    d = json.load(open(path))
    items = [it for it in d["items"] if it.get("answer") and has_window(it["question"])
              and os.path.exists(f"{VIDEO_ROOT}/{it['video_id']}")]
    random.Random(SEED).shuffle(items)
    return items[:n]


def run(items, backend, label):
    correct = 0
    for it in items:
        vpath = f"{VIDEO_ROOT}/{it['video_id']}"
        try:
            pred = backend.answer(vpath, "mcq", it["question"])
        except Exception as e:
            print(f"ERROR {it['video_id']}: {e}", file=sys.stderr)
            continue
        ok = pred.strip().lower() == it["answer"].strip().lower()
        correct += ok
        print(f"[{label}] pred={pred!r} gt={it['answer']!r} ok={ok}")
    print(f"[{label}] acc: {correct}/{len(items)} = {correct/max(1,len(items)):.3f}")


def main():
    items = load_sample("../data/train/mcq.json", N)
    print(f"{len(items)} timestamp-windowed mcq items", file=sys.stderr)
    backend = QwenVLBackend(quant="bf16")

    run(items, backend, "with_window")

    # monkey-patch to disable windowing -> old uniform-sampling behavior
    orig = QwenVLBackend._extract_time_window
    QwenVLBackend._extract_time_window = staticmethod(lambda q, pad_s=3.0: (None, None))
    run(items, backend, "no_window")
    QwenVLBackend._extract_time_window = orig


if __name__ == "__main__":
    main()
