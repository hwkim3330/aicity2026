#!/usr/bin/env python3
"""Evaluate the LoRA fine-tuned checkpoint (lora_out_v1) against the same
local proxy sample used in eval_fps_fix.py, to check whether the fine-tune
actually helped before committing to a full test-set resubmission."""
import json
import os
import random
import sys

sys.path.insert(0, ".")
from inference import QwenVLBackend

VIDEO_ROOT = "../data/videos"
N_PER_TASK = int(sys.argv[1]) if len(sys.argv) > 1 else 40
LORA_PATH = "../lora_out_v1"
SEED = 42


def load_training_keys():
    """(video_id, question) pairs the LoRA adapter was actually trained on --
    must be excluded from eval or accuracy would reflect memorization, not
    generalization."""
    d = json.load(open("../data/finetune_lora_data.json"))
    keys = set()
    for r in d:
        q = r["conversations"][0]["value"]
        if q.startswith("<video>\n"):
            q = q[len("<video>\n"):]
        keys.add((r["video"], q))
    return keys


_TRAIN_KEYS = load_training_keys()


def load_sample(path, n):
    d = json.load(open(path))
    items = d["items"]
    rng = random.Random(SEED)
    rng.shuffle(items)
    out = []
    for it in items:
        if not it.get("answer"):
            continue
        if not os.path.exists(f"{VIDEO_ROOT}/{it['video_id']}"):
            continue
        if (it["video_id"], it["question"]) in _TRAIN_KEYS:
            continue
        out.append(it)
        if len(out) >= n:
            break
    return out


def main():
    bcq = load_sample("../data/train/bcq.json", N_PER_TASK)
    mcq = load_sample("../data/train/mcq.json", N_PER_TASK)
    print(f"[eval_lora] {len(bcq)} bcq + {len(mcq)} mcq items", file=sys.stderr)

    backend = QwenVLBackend(quant="bf16")

    from peft import PeftModel
    print(f"[eval_lora] applying LoRA adapter from {LORA_PATH}", file=sys.stderr)
    backend.model = PeftModel.from_pretrained(backend.model, LORA_PATH)
    backend.model.eval()

    correct = 0
    total = 0
    for task_type, items in (("bcq", bcq), ("mcq", mcq)):
        n_correct = 0
        for it in items:
            video_path = f"{VIDEO_ROOT}/{it['video_id']}"
            try:
                pred = backend.answer(video_path, task_type, it["question"], fewshot=False)
            except Exception as e:  # noqa: BLE001
                print(f"[eval_lora] ERROR on {it['video_id']}: {e}", file=sys.stderr)
                continue
            gt = it["answer"].strip()
            ok = pred.strip().lower() == gt.strip().lower()
            n_correct += ok
            total += 1
            correct += ok
            print(f"[{task_type}] pred={pred!r} gt={gt!r} ok={ok}", file=sys.stderr)
        print(f"[eval_lora] {task_type} acc: {n_correct}/{len(items)} = {n_correct/max(1,len(items)):.3f}")

    print(f"[eval_lora] OVERALL acc: {correct}/{total} = {correct/max(1,total):.3f}")


if __name__ == "__main__":
    main()
