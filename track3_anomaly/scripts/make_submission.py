#!/usr/bin/env python3
"""
Generate a TAR (Track 3) submission CSV by running the Qwen2.5-VL-7B
baseline over every item in test/test.json.

Loads the model once, iterates items grouped by video_id (so a video's
frames only need to be decoded once per clip, even though each clip has
~12 questions attached), and writes item_index,prediction rows.

Missing video files (e.g. an age-gated YouTube source that yt-dlp could
not fetch) fall back to a task-appropriate default answer so the CSV
still validates and covers 100% of items.

Usage:
    python make_submission.py \
        --test_json ../data/test/test.json \
        --media_root ../data/videos \
        --out ../submissions/submission_qwen25vl_4bit.csv \
        [--limit N] [--quant 4bit|8bit|bf16] [--resume]
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FALLBACK_BY_TASK = {
    "bcq": "No.",
    "bcq_openended": "No. The video could not be processed.",
    "mcq": "A",
    "mcq_openended": "A) The video could not be processed.",
    "temporal_localization": '{"start": "00:00", "end": "00:01"}',
}
FALLBACK_DEFAULT = "The video could not be processed."


def load_items(test_json_path):
    with open(test_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc["items"]


def fallback_answer(task_type):
    return FALLBACK_BY_TASK.get(task_type, FALLBACK_DEFAULT)


def normalize_temporal_answer(raw):
    """The prompt asks for {"start": "MM:SS", "end": "MM:SS"}, but the model
    often ignores that and writes HH:MM:SS out of habit (these clips are all
    well under an hour, so HH is always "00"). The evaluator's mIoU parser
    presumably expects exactly MM:SS, so an unstripped HH:MM:SS value throws
    off every interval it touches -- this alone tanked Track8's Temporal
    mIoU to 0.0253 vs competitors' 0.38-0.71 while every other metric was
    competitive. Reformat in place; don't touch anything else about the
    answer."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if not isinstance(obj, dict) or "start" not in obj or "end" not in obj:
        return raw
    changed = False
    for key in ("start", "end"):
        val = str(obj[key])
        parts = val.split(":")
        if len(parts) == 3:
            obj[key] = f"{parts[1]}:{parts[2]}"
            changed = True
    return json.dumps(obj) if changed else raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", default="../data/test/test.json")
    ap.add_argument("--media_root", default="../data/videos")
    ap.add_argument("--out", default="../submissions/submission.csv")
    ap.add_argument("--quant", default="bf16", choices=["4bit", "8bit", "bf16"])
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process the first N *videos* (for smoke testing).")
    ap.add_argument("--resume", action="store_true",
                     help="Skip item_index values already present in --out.")
    ap.add_argument("--samples", type=int, default=1,
                     help="Self-consistency samples for bcq/mcq (majority vote). 1 = greedy only.")
    ap.add_argument("--fewshot", action="store_true",
                     help="Prepend worked bcq/mcq examples (validated +3.8pp locally).")
    args = ap.parse_args()

    items = load_items(args.test_json)

    # group by video_id, preserving first-seen order
    by_video = {}
    order = []
    for it in items:
        vid = it["video_id"]
        if vid not in by_video:
            by_video[vid] = []
            order.append(vid)
        by_video[vid].append(it)

    if args.limit:
        order = order[: args.limit]

    existing = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                existing[row["item_index"]] = row["prediction"]
        print(f"[resume] loaded {len(existing)} existing predictions from {args.out}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    need_model = any(
        it["item_index"] not in existing
        for vid in order
        for it in by_video[vid]
    )

    backend = None
    if need_model:
        from inference import QwenVLBackend
        backend = QwenVLBackend(quant=args.quant)

    results = dict(existing)
    n_videos = len(order)
    t0 = time.time()
    n_missing_video = 0
    n_errors = 0

    for vi, vid in enumerate(order, 1):
        video_items = by_video[vid]
        if all(it["item_index"] in existing for it in video_items):
            continue

        video_path = os.path.join(args.media_root, vid)
        video_exists = os.path.isfile(video_path)
        if not video_exists:
            n_missing_video += 1
            print(f"[{vi}/{n_videos}] MISSING video: {video_path} -> using fallback answers")

        for it in video_items:
            idx = it["item_index"]
            if idx in existing:
                continue
            task_type = it["task_type"]
            question = it["question"]
            if not video_exists:
                results[idx] = fallback_answer(task_type)
                continue
            try:
                ans = backend.answer(video_path, task_type, question, samples=args.samples,
                                      fewshot=args.fewshot)
                if not ans.strip():
                    ans = fallback_answer(task_type)
                elif task_type == "temporal_localization":
                    ans = normalize_temporal_answer(ans)
                results[idx] = ans
            except Exception as e:  # noqa: BLE001
                n_errors += 1
                print(f"  ERROR on {idx} ({task_type}) for {vid}: {e}", file=sys.stderr)
                results[idx] = fallback_answer(task_type)

        if vi % 5 == 0 or vi == n_videos:
            elapsed = time.time() - t0
            print(f"[{vi}/{n_videos}] videos processed, elapsed {elapsed:.0f}s, "
                  f"missing_video={n_missing_video}, errors={n_errors}")

        # write incrementally so progress survives a crash/timeout
        _write_csv(args.out, items, results)

    _write_csv(args.out, items, results)
    print(f"Wrote {args.out} with {len(results)}/{len(items)} predictions "
          f"({n_missing_video} videos missing, {n_errors} inference errors).")


def _write_csv(path, items, results):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_index", "prediction"])
        for it in items:
            idx = it["item_index"]
            if idx in results:
                w.writerow([idx, results[idx]])
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
