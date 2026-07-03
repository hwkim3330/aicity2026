#!/usr/bin/env python3
"""
Runs the Qwen2.5-VL baseline over a SynWTS split and writes both
submission-format JSON files:
  - captioning: {scenario_id: [{labels, caption_pedestrian, caption_vehicle}, ...]}
  - vqa: [{id, correct}, ...]   (id is synthesized as "<scenario>__<view>__<q_index>"
    for local val runs; swap in the organizer-provided test question ids once
    the real test.json is available)

Usage:
    python make_submission.py --root ../data/data/data --split val \
        --out_caption ../submissions/val_caption.json \
        --out_vqa ../submissions/val_vqa.json \
        [--limit N] [--quant 4bit|8bit|bf16]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_captions, load_vqa  # noqa: E402
from inference import QwenVLBackend  # noqa: E402


def run_captions(backend, items, out_path, limit=None, resume=False):
    out = {}
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"[resume] {len(out)} caption scenarios already done, skipping them", file=sys.stderr)
    n = len(items) if limit is None else min(limit, len(items))
    for i, item in enumerate(items[:n]):
        if item.video_path is None:
            continue
        if resume and item.scenario in out:
            continue
        t0 = time.time()
        phases_out = []
        for phase in item.event_phase:
            try:
                ped, veh = backend.caption_segment(
                    item.video_path, phase.get("start_time"), phase.get("end_time")
                )
            except Exception as e:
                print(f"  [warn] {item.scenario}/{item.view} phase {phase.get('labels')}: {e}", file=sys.stderr)
                ped, veh = "", ""
            phases_out.append(
                {
                    "labels": phase.get("labels", []),
                    "caption_pedestrian": ped,
                    "caption_vehicle": veh,
                }
            )
        out[item.scenario] = phases_out
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{i+1}/{n}] {item.scenario} ({item.view}, {len(phases_out)} phases) {time.time()-t0:.1f}s", file=sys.stderr)
    return out


def run_vqa(backend, items, out_path, limit=None, resume=False):
    out = []
    done_ids = set()
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        done_ids = {r["id"] for r in out}
        print(f"[resume] {len(out)} vqa items already done, skipping them", file=sys.stderr)
    n = len(items) if limit is None else min(limit, len(items))
    t0 = time.time()
    for i, item in enumerate(items[:n]):
        if item.video_path is None:
            continue
        item_id = f"{item.scenario}__{item.view}__{item.q_index}"
        if resume and item_id in done_ids:
            continue
        q = item.question
        try:
            pred = backend.answer_mcq(
                item.video_path, q["question"], q,
                start_time=item.start_time, end_time=item.end_time,
            )
        except Exception as e:
            print(f"  [warn] {item.scenario}/{item.view} q{item.q_index}: {e}", file=sys.stderr)
            pred = "a"
        out.append(
            {
                "id": item_id,
                "correct": pred,
                "_gt": q.get("correct"),  # kept for local accuracy calc; strip before real submission
            }
        )
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"[{i+1}/{n}] {time.time()-t0:.1f}s elapsed", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/data/data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out_caption", default="../submissions/val_caption.json")
    ap.add_argument("--out_vqa", default="../submissions/val_vqa.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quant", default="bf16", choices=["4bit", "8bit", "bf16"])
    ap.add_argument("--skip_caption", action="store_true")
    ap.add_argument("--skip_vqa", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Skip items already present in the output file")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_caption), exist_ok=True)
    backend = QwenVLBackend(quant=args.quant)

    if not args.skip_caption:
        cap_items = load_captions(args.root, args.split)
        print(f"loaded {len(cap_items)} caption items", file=sys.stderr)
        cap_out = run_captions(backend, cap_items, args.out_caption, args.limit, args.resume)
        print(f"wrote {args.out_caption}", file=sys.stderr)

    if not args.skip_vqa:
        vqa_items = load_vqa(args.root, args.split)
        print(f"loaded {len(vqa_items)} vqa items", file=sys.stderr)
        vqa_out = run_vqa(backend, vqa_items, args.out_vqa, args.limit, args.resume)
        n_correct = sum(1 for r in vqa_out if r["correct"] == r["_gt"])
        print(f"wrote {args.out_vqa} — local val accuracy: {n_correct}/{len(vqa_out)} = {n_correct/max(1,len(vqa_out)):.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
