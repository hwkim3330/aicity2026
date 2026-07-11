#!/usr/bin/env python3
"""Convert train/{bcq,mcq}.json into the LLaVA-style conversations format
expected by 2U1/Qwen-VL-Series-Finetune. Small subset only (per user request
to fine-tune with a little data) -- this is a targeted LoRA nudge, not a
full retrain."""
import json
import os
import random
import subprocess

VIDEO_ROOT = "../data/videos"
SEED = 42
N_MCQ = 400
N_BCQ = 200
MAX_DURATION_S = 20.0  # longer clips blow up VRAM at train time (no frame
                        # cap available in the finetune repo's video path,
                        # unlike our own inference.py's MAX_FRAMES=16)

_duration_cache = {}


def video_duration(path):
    if path not in _duration_cache:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            _duration_cache[path] = float(out)
        except Exception:  # noqa: BLE001
            _duration_cache[path] = None
    return _duration_cache[path]


def load(path, n, task_label):
    d = json.load(open(path))
    candidates = [it for it in d["items"] if it.get("answer") and it.get("reasoning")]
    rng = random.Random(SEED)
    rng.shuffle(candidates)

    items = []
    for it in candidates:
        if len(items) >= n:
            break
        vpath = os.path.join(VIDEO_ROOT, it["video_id"])
        if not os.path.exists(vpath):
            continue
        dur = video_duration(vpath)
        if dur is None or dur > MAX_DURATION_S:
            continue
        items.append(it)
    print(f"[{task_label}] {len(items)} usable items (of {len(d['items'])} total, "
          f"<= {MAX_DURATION_S}s)")
    return items


def to_conv(it, idx_prefix):
    gpt_text = it["reasoning"].strip() + f"\nFinal answer: {it['answer'].strip()}"
    return {
        "id": f"{idx_prefix}_{it['video_id'].replace('/', '_')}_{hash(it['question']) & 0xffffff}",
        "video": it["video_id"],
        "conversations": [
            {"from": "human", "value": "<video>\n" + it["question"]},
            {"from": "gpt", "value": gpt_text},
        ],
    }


def main():
    mcq = load("../data/train/mcq.json", N_MCQ, "mcq")
    bcq = load("../data/train/bcq.json", N_BCQ, "bcq")

    records = [to_conv(it, "mcq") for it in mcq] + [to_conv(it, "bcq") for it in bcq]
    random.Random(SEED).shuffle(records)

    out_path = "../data/finetune_lora_data.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=1)
    print(f"wrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
