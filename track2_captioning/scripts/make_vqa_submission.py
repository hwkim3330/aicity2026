#!/usr/bin/env python3
"""
Generate the Track 2 VQA submission from the real WTS public-test VQA file
(WTS_VQA_PUBLIC_TEST.json), which is a flat list of {videos, event_phase:
[{start_time, end_time, conversations: [{id, question, a..d}]}]} entries --
different shape from dataset.py's load_vqa() (which expects the nested
per-scenario train/val folder layout SynWTS ships).

Video filenames in the test file are resolved against a filename->path index
built once from the two known video roots (WTS test set + BDD_PC_5K external
test set), since entries don't carry the full relative path.

Writes incrementally (every N questions) so a crash/timeout loses at most a
few minutes of work, and supports --resume to skip already-answered ids.

Usage:
    python make_vqa_submission.py \
        --vqa_json /path/to/WTS_VQA_PUBLIC_TEST.json \
        --video_roots ROOT1 ROOT2 \
        --out ../submissions/test_vqa.json \
        [--limit N] [--resume]
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import QwenVLBackend  # noqa: E402


def build_video_index(roots):
    index = {}
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".mp4") and fn not in index:
                    index[fn] = os.path.join(dirpath, fn)
    return index


def load_questions(vqa_json_path, video_index):
    with open(vqa_json_path, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    n_missing_video = 0
    n_toplevel = 0
    for entry in data:
        videos = entry.get("videos", [])
        video_path = None
        for v in videos:
            if v in video_index:
                video_path = video_index[v]
                break
        if video_path is None:
            n_missing_video += 1
        for phase in entry.get("event_phase", []):
            start = phase.get("start_time")
            end = phase.get("end_time")
            for q in phase.get("conversations", []):
                items.append({
                    "id": q["id"],
                    "question": q["question"],
                    "options": {k: q[k] for k in "abcd" if k in q},
                    "video_path": video_path,
                    "start_time": float(start) if start else None,
                    "end_time": float(end) if end else None,
                })
        # 435/1040 entries have no "event_phase" at all -- instead a
        # top-level "conversations" list of whole-video questions (weather,
        # road surface, traffic volume, etc), same shape as dataset.py's
        # "environment" VQA category. Missing this branch silently dropped
        # 7308/19624 questions (37% of the real test set) from the output.
        if "conversations" in entry:
            n_toplevel += 1
            for q in entry["conversations"]:
                items.append({
                    "id": q["id"],
                    "question": q["question"],
                    "options": {k: q[k] for k in "abcd" if k in q},
                    "video_path": video_path,
                    "start_time": None,
                    "end_time": None,
                })
    print(f"[load] {len(items)} questions ({n_toplevel} whole-video entries), "
          f"{n_missing_video} entries with unresolved video", file=sys.stderr)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa_json", required=True)
    ap.add_argument("--video_roots", nargs="+", required=True)
    ap.add_argument("--out", default="../submissions/test_vqa.json")
    ap.add_argument("--quant", default="bf16", choices=["4bit", "8bit", "bf16"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--write_every", type=int, default=20)
    ap.add_argument("--chunk_size", type=int, default=None,
                     help="Exit cleanly after answering this many NEW "
                          "questions (on top of whatever --resume already "
                          "loaded), instead of running to completion. Works "
                          "around a memory leak observed in this process "
                          "(RSS jumped from 2.4GB to 15GB after ~13 calls, "
                          "not diagnosed to a specific line -- restarting "
                          "the whole process periodically is the reliable "
                          "fix since OS process exit fully reclaims memory "
                          "that in-process del/gc.collect()/empty_cache() "
                          "did not. Run this in a loop with --resume.")
    args = ap.parse_args()

    video_index = build_video_index(args.video_roots)
    print(f"[index] {len(video_index)} unique video files found", file=sys.stderr)

    items = load_questions(args.vqa_json, video_index)
    if args.limit:
        items = items[: args.limit]

    results = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for r in json.load(f):
                results[r["id"]] = r["correct"]
        print(f"[resume] {len(results)} already answered", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    need_model = any(it["id"] not in results for it in items)
    backend = QwenVLBackend(quant=args.quant) if need_model else None

    n_missing_video = 0
    n_errors = 0
    n_done = 0
    t0 = time.time()
    for it in items:
        if it["id"] in results:
            continue
        if it["video_path"] is None:
            results[it["id"]] = "a"
            n_missing_video += 1
            continue
        try:
            ans = backend.answer_mcq(
                it["video_path"], it["question"], it["options"],
                start_time=it["start_time"], end_time=it["end_time"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR on {it['id']}: {e}", file=sys.stderr)
            ans = "a"
            n_errors += 1
        results[it["id"]] = ans
        n_done += 1
        if n_done % 10 == 0:
            gc.collect()
        if args.chunk_size and n_done >= args.chunk_size:
            _write(args.out, results)
            print(f"[chunk] exiting after {n_done} new answers this run "
                  f"({len(results)}/{len(items)} total) -- restart with "
                  f"--resume to continue", file=sys.stderr)
            return
        if n_done % args.write_every == 0:
            _write(args.out, results)
            elapsed = time.time() - t0
            rate = n_done / elapsed
            remaining = len(items) - len(results)
            eta_min = remaining / rate / 60 if rate > 0 else float("inf")
            print(f"[{len(results)}/{len(items)}] elapsed={elapsed:.0f}s "
                  f"rate={rate:.2f}/s eta={eta_min:.0f}min "
                  f"missing_video={n_missing_video} errors={n_errors}", file=sys.stderr)

    _write(args.out, results)
    print(f"Wrote {args.out} with {len(results)}/{len(items)} answers "
          f"({n_missing_video} missing video, {n_errors} errors).")


def _write(out_path, results):
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([{"id": k, "correct": v} for k, v in results.items()], f, indent=2)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
