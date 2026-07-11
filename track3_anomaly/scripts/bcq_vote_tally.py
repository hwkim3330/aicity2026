#!/usr/bin/env python3
"""Collect per-sample self-consistency vote tallies for PSI-VQA BCQ items.

Runs the exact production pipeline (psi_bcq prompt, samples=5: sample 0
greedy + 4 sampled at temp 0.7/top_p 0.9) but records every individual
sample's extracted Yes/No, not just the majority. Used to (a) validate a
vote-margin-based Yes->No rebalancing rule on the labeled train split and
(b) rank the 55 test predictions by flip safety.

Writes JSONL incrementally so partial runs survive:
  ../submissions/bcq_votes_train.jsonl / bcq_votes_test.jsonl
Resumes: skips item_index values already present in the output file.
"""
import json
import os
import sys
import time

sys.path.insert(0, ".")
from inference import QwenVLBackend

SPLITS = {
    "train": ("../data/psi_vqa/train/bcq.json",
              "../data/psi_vqa/train/videos",
              "../submissions/bcq_votes_train.jsonl"),
    "test": ("../data/psi_vqa/test_public/bcq_questions.json",
             "../data/psi_vqa/test_public/videos",
             "../submissions/bcq_votes_test.jsonl"),
}
SAMPLES = 5


def main():
    which = sys.argv[1:] or ["train", "test"]
    backend = QwenVLBackend(quant="bf16")

    # Intercept every generation to capture per-sample outputs.
    raw_outputs = []
    orig_gen = backend._generate_once

    def wrapped(inputs, max_new_tokens, do_sample):
        out = orig_gen(inputs, max_new_tokens, do_sample)
        raw_outputs.append(out)
        return out

    backend._generate_once = wrapped

    for split in which:
        qpath, vroot, outpath = SPLITS[split]
        items = json.load(open(qpath))["items"]
        done = set()
        if os.path.exists(outpath):
            with open(outpath) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["item_index"])
                    except Exception:
                        pass
        todo = [it for it in items if it["item_index"] not in done]
        print(f"[{split}] {len(items)} items, {len(done)} done, {len(todo)} to go",
              file=sys.stderr, flush=True)
        t0 = time.time()
        with open(outpath, "a") as out:
            for i, it in enumerate(todo, 1):
                vpath = os.path.join(vroot, it["video_id"].split("PSI/", 1)[-1])
                raw_outputs.clear()
                try:
                    maj = backend.answer(vpath, "psi_bcq", it["question"],
                                         samples=SAMPLES)
                except Exception as e:  # noqa: BLE001
                    print(f"  ERROR {it['item_index']}: {e}",
                          file=sys.stderr, flush=True)
                    continue
                votes = []
                for txt in raw_outputs:
                    tok = QwenVLBackend._extract_final(txt, "yesno")
                    votes.append(tok)  # may be None if unextractable
                rec = {
                    "item_index": it["item_index"],
                    "video_id": it["video_id"],
                    "votes": votes,          # sample 0 = greedy
                    "majority": maj,
                    "gt": it.get("answer", ""),
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                if i % 5 == 0 or i == len(todo):
                    rate = (time.time() - t0) / i
                    print(f"[{split} {i}/{len(todo)}] {rate:.1f}s/item, "
                          f"eta {(len(todo)-i)*rate/60:.0f}m",
                          file=sys.stderr, flush=True)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
